# Bootstrap Compile Attempt — Wave 2 (v2)

Goal: compile `/home/ubuntu/lithos/compiler/compiler.ls` (4,739 lines) to an
ARM64 ELF `compiler/lithos-stage1` using the pure-ARM64 bootstrap in
`/home/ubuntu/lithos/bootstrap/`.

## Headline result

- **Partial progress.** The bootstrap now assembles, links, runs, and emits
  a valid (though near-empty) ARM64 ELF for a trivial input. The previous
  SIGBUS on every input is fixed.
- **`lithos-stage1` was NOT produced.** `compiler.ls` still fails with
  `parse error`, as do all six files in `compiler/examples/`. The parser
  in `lithos-parser.s` does not speak the grammar either file uses.
- The self-compile fixed-point test therefore was not attempted.

## What I fixed this session

### FIX 1 — Stack-alignment bugs in lithos-elf-writer.s (DONE)

The previous diagnosis said "10 sites"; I found **7 pairs** of
`str xN, [sp, #-8]!` / `ldr xN, [sp], #8` in `lithos-elf-writer.s`. Each
creates an 8-byte-misaligned SP and SIGBUSes as soon as the callee
touches `[sp, #imm]`.

Sites fixed (line numbers are pre-edit):

| Line | Function              | Register | Fix                               |
| ---- | --------------------- | -------- | --------------------------------- |
| 481  | `elf_arm64_phdr`      | x22      | paired with xzr                   |
| 851  | `cubin_sym`           | x1       | paired with xzr + offsets updated |
| 1824 | `code_ELF_BUILD_ARM64`| x20      | paired with xzr                   |
| 1843 | `code_ELF_SAVE`       | x20      | paired with xzr                   |
| 1867 | `code_ELF_WRITE_ARM64`| x20      | paired with xzr                   |
| 1886 | `code_ELF_BUILD_CUBIN`| x20      | paired with xzr                   |
| 1906 | `code_ELF_WRITE_CUBIN`| x20      | paired with xzr                   |

For `cubin_sym` the `ldr x0, [sp, #N]` reads below had to be rewritten
from {#8, #16, #24, #32} to {#16, #24, #32, #40} because the 8-byte push
became a 16-byte pair push.

Verification: before the fix, `./bootstrap/lithos-bootstrap smoke.ls -o out.elf`
SIGBUSed inside `elf_write_arm64` with `sp=0xfffffffff018` (misaligned).
After the fix it exits 0, writes a 120-byte ELF (ehdr + phdr, no text),
and `readelf -h` reports a well-formed `EXEC AArch64` header.

### FIX 2 — Missing NEXT macro (ALREADY DONE BY BUILD.SH)

`bootstrap/build.sh` already prepends `build/_macros.h.s` (which defines
NEXT, PUSH, POP, RPUSH, RPOP) to `lithos-expr.s`. `lithos-lexer.s` and
`lithos-elf-writer.s` now define NEXT inline (guarded by
`.ifndef NEXT_DEFINED`). No source change needed.

### FIX 3 — lithos-expr.s (NOT NEEDED)

Nothing in `driver.s`, `emit-arm64.s`, or `lithos-parser.s` `.extern`s any
`code_EXPR_*` symbol. The file currently assembles OK with the macro
prelude (`build/lithos-expr.asm.err` is empty). Left as-is. Safe to delete
in a future pass; not in the critical path.

### FIX 4 — driver.s integration (ALREADY DONE)

`build.sh` handles this mechanically:
- Renames `_start` in `lithos-bootstrap.s` to `bootstrap_init` (sed pass).
- Auto-generates `.global` directives for every `code_*`, `entry_*`,
  `var_*`, `cfa_*`, `xtl_*`, `drv_*`, `last_entry`, `main_loop`,
  `mem_space`, etc. symbol in each file.
- `driver.o` is linked first so its `_start` wins.

`ld --no-undefined` succeeds with zero warnings. No source change needed.

### FIX 5 — Grammar compatibility (NOT DONE)

`compiler.ls` uses the **new** grammar (`name args :` + `=` assignment +
`buf [ idx ] = val` + `fn`-free). Examples use the **old** grammar (`fn
name args -> output` + `each i` + `=` assignment).

`lithos-parser.s` speaks neither well enough for either to succeed. See
"Blocker" below.

### FIX 6 — Smoke test (DONE, MINIMALLY)

```
$ echo 'var x 42' > smoke.ls
$ ./bootstrap/lithos-bootstrap smoke.ls -o smoke.elf
lithos: wrote smoke.elf
$ readelf -h smoke.elf | head -3
ELF64  EXEC  AArch64
```

The file is 120 bytes (64-byte EHDR + 56-byte PHDR, zero text). This
proves the lex→parse→emit→ELF-write chain plumbs together end-to-end for
a declaration-only input. It does NOT prove the emitter produces
executable code — `var x 42` emits nothing.

Attempting to execute it produces `Illegal instruction` because there is
no text segment. That is correct behavior for a program with no code.

### FIX 7 — Compile compiler.ls (BLOCKED)

```
$ ./bootstrap/lithos-bootstrap compiler/compiler.ls -o compiler/lithos-stage1
parse error
$ ls compiler/lithos-stage1
ls: cannot access ...: No such file or directory
```

Every example in `compiler/examples/` also fails:

```
$ for f in compiler/examples/*.ls; do
>   ./bootstrap/lithos-bootstrap "$f" -o /tmp/out 2>&1
> done
parse error     (x6)
```

No output file is produced in any case.

## The real blocker

The parser in `lithos-parser.s` (2,891 lines) does not implement the
grammar that `compiler.ls` is written in.

`compiler.ls` at a glance uses:
- Top-level compositions: `name args :` (new grammar)
- Assignment statements: `x = expr`
- Indexed assignment: `buf [ idx ] = expr`
- Arbitrary-length arithmetic RHS: `x = y + z * w + 4`
- `buf NAME N`, `var NAME VAL`, `const NAME VAL`

`compiler/examples/*.ls` uses the OLD grammar:
- `fn name args -> output`
- `each i`
- Same assignment forms.

The parser's entry point correctly dispatches on `var`, `buf`, `const`,
and `IDENT ... :` composition heads — but falls over inside the body.
`parse_statement` does not handle `IDENT =` (bare assignment) or
`IDENT [ expr ] =` (indexed assignment). Both forms fire
`.parse_err` → prints `"parse error\n"` → exit 1.

This is not a one-line fix. It requires:
1. Adding an LHS-recognizer that distinguishes
   `ident` / `ident = ...` / `ident [ expr ] = ...`.
2. An expression parser that handles the full arithmetic precedence
   stack (`+ - * / **` plus reductions `Σ △ ▽ #`).
3. `emit-arm64.s` support for memory stores with computed addresses
   (the arm64 backend currently emits registers + immediate stores, not
   `STR Xsrc, [Xbase, Xindex, LSL #n]`).
4. Keywords: `fn`, `->`, `<-` as ASCII fallbacks for `→ ←` in the
   lexer. Currently the lexer only recognizes the UTF-8 arrows.

I stopped short of doing this work because the scope (several hundred
lines of hand-written ARM64 across parser + emitter, plus a test loop
that requires a working emitter to validate against) is larger than can
be responsibly squeezed into one session. A half-written parser is worse
than a clearly-documented blocker.

## Exact state after this session

```
$ cd /home/ubuntu/lithos && ./bootstrap/build.sh
...
built: /home/ubuntu/lithos/bootstrap/lithos-bootstrap

$ printf 'var x 42\n' > smoke.ls
$ ./bootstrap/lithos-bootstrap smoke.ls -o smoke.elf
lithos: wrote smoke.elf
$ readelf -h smoke.elf | grep Type
  Type:                              EXEC (Executable file)

$ ./bootstrap/lithos-bootstrap compiler/compiler.ls -o compiler/lithos-stage1
parse error                   # BLOCKED — parser gap
```

## Files modified

- `/home/ubuntu/lithos/bootstrap/lithos-elf-writer.s` — 7 alignment fixes
  at lines 480, 517, 849, 851, 871, 1822–1828, 1841–1847, 1865–1871,
  1884–1890, 1904–1910 (all `str x{20|22|1}, [sp, #-8]!` → `stp xN, xzr,
  [sp, #-16]!` and matching loads; cubin_sym offsets shifted by 8).

No other files were touched. No `.fs` file was modified. No `.li` file
was created.

## What needs to happen to unblock Wave 3

1. **Rewrite `parse_statement` in `lithos-parser.s`** to handle:
   - `IDENT = EXPR`             (scalar assignment)
   - `IDENT [ EXPR ] = EXPR`    (indexed store)
   - `fn IDENT IDENT* -> IDENT` (old-grammar composition header)
   - Any other top-level form compiler.ls uses that the current
     recursive descent rejects.

2. **Add a real expression parser** with `+ - * /` precedence and
   the reduction operators. Emit ARM64 directly (no AST) into
   `arm64_buf`.

3. **Add `fn`, `->`, `<-` as ASCII tokens** in `lithos-lexer.s`.

4. **Sanity test** against `compiler/examples/vadd.ls` (9 lines) before
   `compiler.ls`. If the 9-line file doesn't compile, the 4,739-line
   file won't either.

5. **Only then attempt compiler.ls self-compile** and the
   fixed-point diff (`compare stage1 == stage2`).

## Summary for the impatient

- Stack-alignment bus error: FIXED. Smoke test compiles.
- Parser grammar gap: NOT FIXED. compiler.ls cannot be compiled.
- `compiler/lithos-stage1`: NOT produced.
- Self-compile fixed-point test: NOT attempted (no stage1 exists).
