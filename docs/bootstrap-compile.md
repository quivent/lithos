# Bootstrap Compile Attempt — Wave 2

Goal: compile `/home/ubuntu/lithos/compiler/compiler.ls` (4,739 lines, new
Lithos grammar) to an ARM64 ELF binary `lithos-stage1` using the
Forth-era bootstrap in `/home/ubuntu/lithos/bootstrap/`.

**Result: the bootstrap cannot compile anything end-to-end in its current
state. `lithos-stage1` was not produced.** The pipeline needs substantive
repair across three source files before it will build any .ls file, let
alone the self-hosting compiler.

---

## What the existing binary is

```
$ file /home/ubuntu/lithos/bootstrap/lithos-bootstrap
ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV),
statically linked, not stripped
```

It is **only the Forth core**. `nm` shows a single `_start` at `0x4000b0`
(the Forth bootstrap entry), `main_loop`, `__bss_start`. None of the
lithos-lexer / lithos-parser / emit-arm64 / lithos-elf-writer /
driver symbols are linked in.

Timestamps confirm it:

```
lithos-bootstrap         19:04      # built before driver.s existed
driver.s                 19:22      # newer
lithos-lexer.s           19:17
lithos-parser.s          19:19
emit-arm64.s             19:17
lithos-elf-writer.s      19:17
```

Running it against a `.ls` file drops into the Forth outer interpreter,
which treats the source as Forth code and prints `? \\` or `? fn`
(unknown word) — exactly what you see today.

```
$ /home/ubuntu/lithos/bootstrap/lithos-bootstrap compiler/examples/hello.ls
? fn
```

## Command(s) attempted

Build the real pipeline from the six .s files:

```
cd /home/ubuntu/lithos/bootstrap
as -o lithos-bootstrap.o lithos-bootstrap.s        # OK
as -o lithos-lexer.o     lithos-lexer.s            # FAIL: missing NEXT macro
as -o lithos-parser.o    lithos-parser.s           # OK
as -o emit-arm64.o       emit-arm64.s              # OK
as -o lithos-elf-writer.o lithos-elf-writer.s      # OK
as -o driver.o           driver.s                  # OK
as -o lithos-expr.o      lithos-expr.s             # FAIL: many errors
ld -o lithos-new driver.o lithos-bootstrap.o ...   # FAIL: undefined refs
```

## Problems encountered, and fixes applied in this pass

### 1. `lithos-lexer.s` will not assemble on its own (fixable)

It uses `NEXT` but never defines the macro. Designed to be textually
appended to `lithos-bootstrap.s`. Fix: prepend a `.macro NEXT` guarded
by `.ifndef NEXT_DEFINED`. Then it assembles.

### 2. `lithos-expr.s` has dozens of assembler errors (NOT fixed)

```
lithos-expr.s: unknown mnemonic `next'                 (lowercase, not NEXT)
lithos-expr.s: unknown mnemonic `pop'
lithos-expr.s: undefined symbol SYS_WRITE
lithos-expr.s: undefined symbol SYS_EXIT
lithos-expr.s: immediate cannot be moved by a single instruction  (~12 sites)
```

This file is referenced by nothing else in the build (driver.s does not
`.extern` any `code_EXPR_*`). Dropped from the link.

### 3. `_start` collision between driver.s and lithos-bootstrap.s (fixed here)

driver.s supplies its own `_start` and expects the bootstrap's `_start`
to be removed or made weak (per comment at end of driver.s, item 1).
Fix: renamed `_start` → `_start_bootstrap` in a patched copy of
lithos-bootstrap.s.

### 4. Almost every symbol needed by the driver is file-local (fixed here)

The bootstrap only declares `.global _start`. The driver needs:

```
saved_argc   saved_argv   data_stack_top   ret_stack_top
mem_space    var_state    var_latest       last_entry
var_source_addr var_source_len var_to_in   var_input_fd
main_loop    code_INTERPRET  entry_pad
```

All added via injected `.global` directives before `.end`. Same for
`code_LITHOS_LEX`, `code_LEX_TOKENS`, `code_LEX_COUNT`,
`code_LEX_TOKEN_FETCH`, `code_PARSE_TOKENS`, `code_CODE_BUF`,
`code_CODE_POS`, `code_CODE_RESET`, `code_ELF_WRITE_ARM64`,
`code_ELF_WRITE_CUBIN`, `code_CUBIN_PARAMS`.

### 5. driver.s uses native-code addresses as execution tokens (FIXED)

The DTC `NEXT` macro dereferences twice:

```
ldr x25, [x26], #8    ; x25 = xt (address of CFA cell)
ldr x16, [x25]        ; x16 = native entry = *CFA
br  x16
```

driver.s put raw `code_LITHOS_LEX` (etc.) into its `xtl_*` lists. On the
very first `br x16` the CPU jumped to whatever bytes the lexer's first
instruction happened to decode to — `PC = 0x200000aa1603e1`, SIGBUS.
Fix: inserted `cfa_*` data cells and made the xt lists point to the
CFA addresses. This matches the existing `cfa_drv_RET` pattern that was
already correct.

### 6. `lithos-elf-writer.s` has 10 stack-alignment bugs (NOT fixed)

Every code_ELF_WRITE_* entry ends with

```
    str     x20, [sp, #-8]!       ; 8-byte push — breaks 16-byte SP alignment
    bl      elf_...               ; SIGBUS when callee touches [sp,#imm]
    ldr     x20, [sp], #8
```

This is the cause of the bus error seen when invoking the rebuilt
binary on any input, including an empty file:

```
$ gdb -batch -ex 'file /tmp/lithos-new' -ex 'run /tmp/mini3.ls' -ex bt
Program received signal SIGBUS, Bus error.
0x4083c0 in elf_write_arm64 ()
#0 0x4083c0 in elf_write_arm64 ()
#1 0x4090b0 in code_ELF_WRITE_ARM64 ()
```

Each of the 10 sites needs `stp x20, xzr, [sp, #-16]!` / `ldp x20, xzr, [sp], #16`
(or equivalent 16-byte pair push) before the pipeline can run to completion.

### 7. Parser / lexer have not been end-to-end tested

Even once the alignment is fixed, these are unexercised:

- Lexer: handles `\\`, arrows `→ ← ↑ ↓`, `Σ △ ▽ #`, UTF-8 literals —
  but there is no keyword for `fn`, no `->` / `<-` ASCII fallback.
  `compiler/examples/hello.ls` uses the **old** `fn ... each ...`
  grammar, which the new lexer will not keyword-recognize (it becomes
  an ident stream, and the parser then emits `parse error`).
- Parser: `parse_toplevel` accepts `var`, `buf`, `const`, and
  `IDENT ... :` compositions. It does NOT handle bare expressions,
  top-level statements, or the `fn` form. So every example file in
  `compiler/examples/` except ones hand-crafted in the new grammar
  will fail.
- No smoke test of any kind has been run against the parser+emitter
  since they were written.

## What the patched build actually does

After the fixes in §1, §3, §4, §5 above:

```
$ /tmp/lithos-new /home/ubuntu/lithos/compiler/compiler.ls -o /tmp/lithos-stage1
parse error
$ /tmp/lithos-new /home/ubuntu/lithos/compiler/examples/hello.ls
parse error                # old grammar
$ /tmp/lithos-new /tmp/empty.ls
Bus error                  # §6 alignment bug
```

No output file is produced in any case.

## What needs to happen to unblock Wave 2

In priority order — these are the minimal edits to get a first
`lithos-stage1`:

1. **Fix the 10 stack-alignment bugs in `lithos-elf-writer.s`.** Change
   every `str x20, [sp, #-8]!` / matching load to a 16-byte paired
   push/pop. Without this, the ELF writer crashes before it can emit
   anything. Grep: `grep -n 'x20, \[sp, #-8\]!' lithos-elf-writer.s`.

2. **Move the global-symbol declarations into the `.s` files.** The
   hacks above (injecting `.global` before `.end`, wrapping the lexer
   with a NEXT macro) should be committed to the source rather than
   synthesized per-build. Specifically:
   - `lithos-bootstrap.s`: `.global` for the 15 symbols in §4; rename
     or `.weak` its `_start`.
   - `lithos-lexer.s`: add the `NEXT` macro guard and `.global
     code_LITHOS_LEX`, `code_LEX_TOKENS`, `code_LEX_COUNT`,
     `code_LEX_TOKEN_FETCH`.
   - `lithos-parser.s`: `.global code_PARSE_TOKENS`.
   - `emit-arm64.s`: `.global code_CODE_BUF`, `code_CODE_POS`,
     `code_CODE_RESET`.
   - `lithos-elf-writer.s`: `.global code_ELF_WRITE_ARM64`,
     `code_ELF_WRITE_CUBIN`, `code_CUBIN_PARAMS`.

3. **Commit the driver.s xt-list fix.** Done in this pass —
   `cfa_LITHOS_LEX` / `cfa_LEX_TOKENS` / etc. indirection cells now
   exist.

4. **Either delete or repair `lithos-expr.s`.** It is not referenced
   by anything in the build and does not assemble. If it is needed for
   parser expression-compile, port its content into `lithos-parser.s`
   or rewrite with consistent macros.

5. **Make the parser actually work on trivial input.** The minimum
   viable test is `var x 42\n` round-tripping into an ELF that exits 0.
   Right now even an empty file crashes (see §6).

6. **Decide whether `compiler/examples/*.ls` use the old or new
   grammar, and make the lexer+parser match.** Five of six examples
   start with `fn`, which the lexer does not recognize. Either:
   - add `fn` / `->` / `<-` to the lexer and parser (lexer has TOK_LOAD
     / TOK_STORE as arrows only, not ASCII), or
   - rewrite those examples in the composition grammar that the
     parser actually speaks.

7. **Only after a 10-line example round-trips**, try `compiler.ls`.
   It will flush out many more bugs (4,739 lines, ~1,500 compositions)
   that nothing smaller has exercised.

## Self-compile fixed-point test

Not attempted. Blocked on §1–§6 above. `lithos-stage1` was never
produced so there is no binary to re-run against `compiler.ls`.

## Files touched this session

- `/home/ubuntu/lithos/bootstrap/driver.s` — inserted `cfa_*` cells and
  rewired the seven `xtl_*` lists to point to CFAs (fix for §5).
  Driver will not work with the old xt-list layout; this change is a
  bug fix, not an experiment.

No other `.s` or `.ls` files were modified. The lexer NEXT-macro
wrapping, the `.global` injections, and the `_start` rename were applied
only to copies under `/tmp/` to see how far the pipeline could get.
They need to be made permanent in the source before the next build
attempt.
