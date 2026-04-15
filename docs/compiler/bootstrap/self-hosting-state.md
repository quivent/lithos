# Lithos Self-Hosting: Current State

As of 2026-04-15, branch `exp`, commit `6550aca`.

## Executive Summary

The Lithos bootstrap compiler (`bootstrap/lithos-bootstrap`, written in ARM64
assembly) now successfully compiles `compiler/compiler.ls` into a ~50KB ARM64
ELF binary (`lithos-stage1`). That binary runs, extracts argc/argv, opens the
input source file, mmaps it, runs the lexer through to completion, and then
crashes with infinite recursion inside the parser.

Reaching this point required fixing seven classes of bootstrap bug and
refactoring several thousand lines of `compiler.ls`. The remaining work is
tractable: identify which parser handler fails to advance `x19` (the token
pointer).

## Pipeline Status

| Stage | Status |
|-------|--------|
| `_start` stub (argc/argv/BSS base) | ✅ works |
| `main` → argc validation | ✅ works |
| `mmap_file` (openat/lseek/mmap/close) | ✅ works, verified via strace |
| `lex` (tokenizer) | ✅ runs end-to-end on 1138-byte source |
| `parse_file` → parser entry | ❌ infinite recursion, stack overflow |
| ELF writer | untested |
| File output | untested |

## Work Completed

### 1. Composition-level fixes in compiler.ls

**Argument count cap (bootstrap handles ≤8 args).** `elf_build` reduced 10→7
params; `elf_write_cubin` 11→8; `shdr64_emit` split into `shdr64_a`/`shdr64_b`
(5+5). Cooperative, gridsync, and exit-offset data moved to globals.

**REGCOUNT off-by-one.** `max_reg` is the highest register *index*; REGCOUNT
in the cubin EIATTR must be *count* (index + 1). Fixed in `elf_build`.

**EXIT_INSTR_OFFSETS hardcoded to 0x100.** Replaced with real tracking via a
`record_exit`/`exit_offsets` pattern mirroring `record_gridsync`. The EIATTR
now emits actual byte offsets.

**`elf_build` body exceeded register window.** Split into six sub-compositions
(`elf_emit_shstrtab`, `elf_emit_strtab_symtab`, `elf_emit_nvinfo`,
`elf_emit_nvinfo_k`, `elf_emit_text_const0`, `elf_emit_shdrs`) that share
state through globals. Each fits inside the 20-register window.

**`lex` rewritten flat.** Original used `while → if → elif → else` nested
five deep. The bootstrap’s `parse_body` uses indent detection that breaks on
elif chains after deep nesting (it emitted only 44 of ~750 expected
instructions). Rewrote as a single composition with `goto lex_loop` and
sequential `if==` dispatches — compiles to 747 instructions.

**`main` restructured.** Inlined the old ABI-extraction stub (`$31 + 16`
style) and moved `main` to the end of the file (the bootstrap enters at
`ls_last_comp_addr`, not the first composition). `main` now takes
`(argc, argv)` as normal parameters and the `_start` stub sets them up.

**`mmap_file` register discipline.** Every `↓ $N` to a low-numbered register
clobbers the caller’s param. Fixed by copying `path` to a safe register
(X7) before starting the syscall sequence, then saving `fd` in X6 and
`file_size` in X7 so the caller can read them back via `↑ $6` / `↑ $7`.
Global vars are register-backed and don’t survive calls, so results are
communicated through callee-saved-but-scratch-visible physical registers.

**All 79 global `var` declarations → `buf … 8`.** 264 references updated
across the file. Reads become `tmp → 64 buf_v`; writes become `← 64 buf_v
value`; read-modify-writes become the three-line pattern. The ELF writer
sections were especially dense (`elf_emit_shdrs` needed every section
metadata loaded into locals before the `shdr64_a/b` calls).

### 2. Bootstrap fixes (lithos-table.s, lithos-elf-writer.s, ls-shared.s)

**Expression-level BL encoding.** `.La_call_emit` used `sub w1, w1, w0` on
values loaded via `bl emit_cur`. The 32-bit sub produced wrong BL offsets
for some symbols (manifested as BL targets past end of file, e.g.,
`0x44xxxx` in a 48KB binary). Replaced with the `mov x1, x0; ldr w0,
[SYM_REG]; sub x2, x0, x1` pattern from `handle_call`.

**`handle_store` register preservation.** `handle_store` calls `parse_expr`
three times (width, addr, value). `parse_expr` is a `BL` that may clobber
`x4`/`x5`. When our buf ADD emission increased register pressure, `x5`
(addr register) was overwritten before the final `STR`, causing `STR Xval,
[Xval]` (NULL deref). Added save/restore around the second and third
parse_expr calls.

**`handle_binding` raises reg_floor.** The prior design deliberately did
not raise reg_floor, relying on `parse_body` to skip `reset_regs` after
binding statements (`w0 == 1` flag). That only protects bindings when all
following statements are also bindings. The moment a non-binding statement
fires (e.g., `← 32 buf_v val`), `reset_regs` resets `next_reg = reg_floor =
REG_FIRST = 9`, and the next binding reuses the earlier binding’s
register. This caused `pos` and `lp` in `lex` to both map to X9. Fixed by
raising `reg_floor = binding_reg + 1` after `sym_add`, matching how params
and the old approach (pre-optimization) protected locals.

**Spill count reset on composition entry.** `handle_composition` sets
`reg_floor` and `next_reg` on entry but left `spill_count` as whatever the
caller had. The first `reset_regs` inside the callee emitted spill-restore
code that popped frames from the caller’s stack context, corrupting the
stack. Added `str wzr, [spill_count]` after the reg_floor setup.

**Width-aware indexed loads.** `.La_load` with three operands (`→ W base
offset`) always emitted `LDRB Wd, [Xn, Xm]` regardless of width. Added a
saved `ls_load_width` global that records the last-parsed width literal
and a branch that emits `LDR Xd`, `LDR Wd`, or `LDRB Wd` accordingly. The
width is always an int literal in practice, so parsing the int token
directly before `parse_atom` consumes it works.

**`$N` register syntax in expressions.** `parse_dollar_reg` was only
wired into `handle_reg_read`/`handle_reg_write`. In expression context,
`$31` came through as `TOK_IDENT` and hit `sym_lookup` (missed). Added
a check in `.La_ident_lookup` for `$` prefix and routed to
`parse_dollar_reg`. Also updated `parse_dollar_reg` to reject $31 (kept
the previous cap at $30 after we realized R31 is context-sensitive on
ARM64 — see §Architecture, R31 problem).

**Forward-goto patching.** Forward gotos (label defined after the `goto`)
were emitted as `NOP` and never patched. Added a 256-entry patch table
(`ls_fwd_gotos`, `ls_n_fwd_gotos`) recording `(name[32], src_offset,
length)`. `handle_label` scans the table on label definition and emits
`B imm26` at the recorded source offset with the correct distance.

**Buffer address computation (X28 = BSS base).** This was the single
largest fix. See §Architecture below.

**`_start` stub expanded to 4 instructions.** Now:

```
LDR  X0, [SP]            ; argc
ADD  X1, SP, #8          ; argv
MOVZ X28, #0x50, LSL #16 ; BSS base = 0x500000
B    <main>              ; patched to last composition
```

The ELF program header `p_memsz` was increased to `0x100000 + bss_size`
so the kernel maps the gap from `0x400000` (code) through `0x500000` (BSS
start) + actual BSS size into one contiguous RWE LOAD segment.

### 3. Language discipline discovered

ARM64 register 31 is context-sensitive: SP in load/store, XZR in
arithmetic. One 5-bit encoding, two meanings. `$31 + 16` generates
`ADD XZR, XZR, X9` (writes to zero register, result lost); `→ 64 $31`
generates `LDR X, [SP]` (works). This is a hardware quirk that can't be
hidden. Lithos bans `$31` — `parse_dollar_reg` caps at $30 — and the
compiler’s ABI code uses `$29` (frame pointer) for stack-relative reads.

## Architecture Understanding

### Bootstrap compiler structure

The bootstrap is a DTC (direct-threaded code) Forth-like system in ARM64
assembly, extended with a table-driven `.ls` parser (`lithos-table.s`).
Key files:

- `lithos-bootstrap.s`: DTC core, stack machine
- `lithos-lexer.s`: tokenizer
- `lithos-table.s`: `.ls` statement dispatcher and handlers
- `emit-arm64.s`: ARM64 instruction emitters (LDR/STR/ADD/BL/etc.)
- `lithos-elf-writer.s`: ELF64 output
- `driver.s`: `_start`, argv parsing, dispatch to lexer→parser→emitter→elf
- `ls-shared.s`: shared buffers (code_buf, sym_table, token_buf)

The bootstrap runs as a single static binary. Its own state lives in its
BSS (ls_code_buf, ls_sym_table, etc.). When compiling a `.ls` file, it
reads source, tokenizes, walks tokens emitting ARM64 machine code into
`ls_code_buf`, then writes an ELF wrapping that code.

Register conventions (bootstrap-internal):
- X19: TOKP (current token pointer)
- X27: TOKEND
- X28: SRC (source buffer base)
- X29/X30: FP/LR
- X9–X18: scratch

### Symbol table

`ls_sym_table`: 1024 entries × 24 bytes each. Layout:
- `[0:4]` name offset (into `ls_source_buf`)
- `[4:8]` name length
- `[8:12]` kind (`KIND_PARAM=1`, `KIND_VAR=2`, `KIND_BUF=3`, `KIND_COMP=4`,
  `KIND_CONST=6`, `KIND_LOCAL_REG=0`)
- `[12:16]` SYM_REG — meaning depends on kind:
  - PARAM/LOCAL_REG/VAR: physical register number (X0–X28)
  - BUF: BSS offset (after our fix; was size before)
  - COMP: code address in `ls_code_buf` (truncated to 32 bits)
- `[16:20]` scope depth
- `[20:24]` padding

### Register allocation

Bump allocator with spill. `alloc_reg` returns `next_reg` and bumps it.
When `next_reg > REG_LAST` (28 before our fix, 27 after), `.Lalloc_spill`
emits `STR X_REG_FIRST, [SP, #-16]!`, shifts live locals down, and reuses
`REG_FIRST`. `reset_regs` restores `next_reg = reg_floor` and emits `LDR`
pops balancing all outstanding spills.

**Critical invariants** (both violated in pre-fix bootstrap):
1. Bindings must raise `reg_floor`, or mixed binding/non-binding sequences
   collide.
2. `spill_count` must reset on composition entry, or the callee pops the
   caller’s stack frames.

### Composition calling convention

Arguments 0–7 in X0–X7. Bootstrap caps compositions at 8 params.
Prologue `STP X29, X30, [SP, #-16]!; MOV X29, SP` emitted by
`handle_composition`. Epilogue `LDP X29, X30, [SP], #16; RET`. No stack
allocation for locals — everything in registers (or spilled via the
register allocator).

### Global variable problem (the big one)

The bootstrap originally mapped every global `var NAME 0` to a register.
It `alloc_reg`ed one on declaration and stored the register number in
`SYM_REG`. But every composition’s `handle_composition` resets
`reg_floor = REG_FIRST`, meaning the callee’s locals start at the same
X9, X10, etc. that the global vars occupy. Inside the callee, the
"globals" are gone — those registers hold callee locals.

The original `compiler.ls` worked around this by never actually reading a
global var from two compositions in a way that expected continuity. The
bootstrap handled the specific bufs (`tokens`, `gpu_buf`) because the
size-as-SYM_REG happened to mask to a reasonable register number. It was
a pile of coincidences.

The fix is memory-backed bufs:

- Every `var NAME 0` becomes `buf NAME_v 8`.
- `handle_buf` assigns a unique BSS offset via `ls_bss_offset` and
  stores that offset in `SYM_REG`.
- `parse_atom` for `KIND_BUF` emits `ADD Xr, X28, #offset` (with a second
  ADD for offsets > 4095 bits 12-23).
- `_start` initializes `X28 = 0x500000` (BSS base VA).
- The ELF program header's `p_memsz` covers `0x100000 + bss_size` so the
  BSS at 0x500000 is mapped.
- Reads: `tmp → 64 NAME_v` — the bootstrap resolves `NAME_v` via the
  `ADD X28, offset` path, yielding the address, then `handle_load` emits
  `LDR Xtmp, [Xaddr]`.
- Writes: `← 64 NAME_v value` — same address computation, then `STR`.

**Cost**: every global access is now 2-3 instructions (ADD + LDR/STR)
instead of a register move. Worth it — correctness first.

### Forward goto machinery

Labels are `KIND_LOCAL_REG` with `SYM_REG = emit_ptr at definition site`.
`handle_goto` calls `sym_lookup`. If found, emit `B` with computed offset.
If not found, record `(name, src_offset, length)` in `ls_fwd_gotos` and
emit `B` with `imm26 = 0` as a placeholder. `handle_label` scans the
patch table, matches by name + length, and writes the real `imm26` at
the recorded source offset. Entries are invalidated by zeroing length.

## What's Broken Right Now

The parser infinite-recurses. Symptoms:
- `SP = 0xffffff800000` at crash — ~8MB of stack used
- 500,000+ frames pushed
- X30 points into the middle of some parser composition (around 0x40aba4)
- That composition has 5 self-calls and appears to be parse_atom or
  parse_expr

The recursion pattern suggests a recursive-descent parser that doesn't
advance the token pointer. Causes in order of likelihood:

1. **A handler we modified stopped advancing `x19`.** Most likely culprit:
   one of the `←`/`→` / expression handlers where the var→buf conversion
   introduced a load/store but forgot `add x19, x19, #TOK_STRIDE_SZ`.
2. **A handler returns without consuming its tokens on an error path.**
   The parser keeps re-dispatching on the same token.
3. **A sym_lookup that used to find a KIND_VAR now finds nothing because
   we renamed everything to `_v`.** If a composition body references
   `emit_target` (the old name), the lookup fails, `.La_ident_unknown`
   fires, tokens aren't consumed, parser loops.

## Remaining Work

### Immediate (to finish self-hosting)

1. **Find the parser recursion.** Strategy: identify the composition at
   `0x40aba4` (check git log for parser body with ~105 instructions and
   5 self-calls). Most likely `parse_expr`, `parse_atom`, or
   `handle_ident_stmt`. Audit every line that touches `x19` to confirm
   advancement on every code path.

2. **Verify buf-address ADD for offsets ≥ 4096.** The second ADD
   (with `LSL #12`) has been written but never exercised (all our bufs
   currently fit in the first 4KB). Once many bufs are declared, this
   path runs. Bit encoding for ADD-imm with `sh=1` is
   `0x91400000 | imm12<<10 | Rn<<5 | Rd` — confirm against hardware.

3. **Test ELF output.** Once the parser terminates, verify that the
   output cubin or ARM64 ELF is well-formed:
   - `readelf -h`, `readelf -l`
   - `nvdisasm` for cubin, `aarch64-linux-gnu-objdump -d` for ARM64
   - Check REGCOUNT, EXIT_INSTR_OFFSETS, section layout

4. **Stage-1 self-check: `lithos-stage1 compiler.ls -o lithos-stage2`.**
   Then `diff lithos-stage1 lithos-stage2` — fixed point confirms
   self-hosting. Any diff means the stage1 compiler produces different
   code than the bootstrap, which is a bug.

### Follow-up

5. **Delete unused bootstrap helpers.** With `_v` bufs everywhere, the
   `KIND_VAR` path in parse_atom is still there but unused. Cleanup pass.

6. **Rewrite `compiler.ls` to use memory-backed globals natively** (no
   `_v` suffix). Once stage1 works, it can compile itself with cleaner
   syntax since it doesn’t have the bootstrap’s buf-address bug. Drop
   all the `tmp → 64 foo_v` boilerplate.

7. **Fix the buf offset cap.** Currently `0x100000 + bss_size` for the
   segment, hard-coded `0x500000` BSS base. Works up to ~16MB of BSS.
   If `compiler.ls` ever grows past that, ADD-imm runs out of bits
   (24-bit addressable space). Long-term: switch to ADRP+ADD for buf
   addresses (relocatable) or widen the address computation.

8. **Test GPU kernel compilation.** The GPU backend (`emit-gpu.ls`) has
   been partially tested but never run end-to-end through the new
   buf-backed compiler. A `.ls` kernel should produce a valid cubin
   that `nvdisasm` can read.

### Speculative / architecture

9. **The bootstrap itself should be discarded once stage1 works.** The
   ARM64-assembly bootstrap is ~3500 lines of very carefully tuned
   register-machine code. It's maintenance debt. Goal: have
   `compiler.ls` (written in clean Lithos) compile itself, and delete
   the bootstrap entirely.

10. **Consider a two-pass symbol resolution.** The current single-pass
    forward-goto patching works because labels are local to a
    composition. But forward *composition* references (calling a
    composition defined later in the file) still emit NOP/wrong BLs
    unless the file is written leaf-first. Adding a pre-scan pass that
    records all composition addresses would eliminate ordering
    constraints. For now, `compiler.ls` is written with leaf functions
    at the top and `main` at the bottom — workable but brittle.

11. **Spilling is broken architecture.** The register allocator’s
    spill-to-stack mechanism is fragile: unbalanced pushes between
    composition calls corrupt the stack. The reg_floor mechanism saves
    bindings but adds pressure. A cleaner design: every composition
    pre-allocates a stack frame for its locals (AAPCS-style) and loads
    them as needed. This is a real compiler, not a peephole optimizer —
    the "every local is a register" approach doesn't scale past ~20
    locals.

12. **Global vars via buf is the right primitive, not a workaround.**
    The `_v` suffix is cosmetic. Every cross-composition variable
    should be memory-backed. Register-backed "vars" were a bug
    pretending to be a feature. The language spec should say so.

13. **The GSP/CPU split should stay.** The CPU-side compiler produces
    ARM64 ELF for the host. The GPU side produces SM90 SASS in a cubin.
    They share a lexer and parser and differ only in the emit backend.
    This is correct. The mess came from using the CPU-side for
    everything, including the compiler's own state. The compiler is a
    CPU program; its state should live in memory, like any other CPU
    program. Bufs are memory. Done.

## Metrics

- Bootstrap binary: 139KB (was 128KB)
- `lithos-stage1`: ~50KB ARM64 ELF
- `compiler.ls`: ~5300 lines
- References converted: 264 (all var → buf load/store)
- Bootstrap bugs found and fixed: 7 distinct classes
- Pre-existing tests passing: ARM64 encoding tests still green

## Files Touched

Bootstrap (ARM64 asm):
- `bootstrap/lithos-table.s`: expression-level BL fix, handle_binding
  reg_floor, handle_store save/restore, handle_buf BSS allocation,
  parse_atom buf ADD emission, spill_count reset on composition entry,
  forward-goto patching via `.La_ident_buf`, width-aware load
- `bootstrap/lithos-elf-writer.s`: `_start` stub expansion, MOVZ X28,
  segment MemSiz
- `bootstrap/ls-shared.s`: `ls_last_comp_addr`, `ls_load_width`,
  `ls_bss_offset`, `ls_fwd_gotos`, `ls_n_fwd_gotos`

Compiler (`.ls`):
- `compiler/compiler.ls`: 79 var → buf, 264 references, lex rewrite,
  mmap_file register discipline, elf_build split into 6 sub-comps,
  main structure, prologue inlining

Configuration:
- REG_LAST reduced 28 → 27 (X28 reserved for BSS base)
- `parse_dollar_reg` cap 30 → banned 31 (ARM64 SP/XZR ambiguity)
