## Stage1 segfault root cause (2026-04-15)

After pulling `ee46577` (stride-12 + local-shadowing + `lex_src_v` publish fixes),
stage1 still segfaults on any non-trivial input. Investigation found the real
root cause is **not** in `compiler.ls` — it's in the Linux bootstrap's ELF BSS
sizing.

### Minimal reproduction (no compiler.ls involved)

```lithos
buf mybuf 128
main :
    ← 32 mybuf 99
    ↓ $8 93
    ↓ $0 42
    trap
```

Segfaults at `str x11, [x10]` where `x10 = 0x500000`. The crashing
instruction is the store into the first declared buf.

### The bug

`readelf -l` on the resulting binary:

```
LOAD  vaddr=0x400000  filesz=0xb4  memsz=0x100000  RWE
```

The segment covers `[0x400000, 0x500000)` — **exclusive upper bound**.
But `bootstrap/lithos-elf-writer.s:510-513` sets:

```asm
// p_memsz = 0x100000 + bss_size
mov  x0, #0x100000
add  x0, x0, x21    // x21 = bss_size from caller
```

With `bss_size = 0`, `memsz = 0x100000`, and x28 (BSS_BASE) is hardcoded
to `0x500000` — which is one byte past the end of the mapped segment.

### Why is bss_size zero?

`bootstrap/driver.s:458-487` walks the symbol table to compute
`bss_total` by finding `KIND_BUF=3` entries and summing their sizes
from offset 12 of a 24-byte sym entry.

For the minimal repro, this sum is `0` — meaning `buf mybuf 128` is
**not** landing in the symbol table as `KIND_BUF`, or the entry stride is
wrong. Needs verification in the parser (`lithos-parser.s` /
`lithos-table.s`).

### Why does compiler.ls (stage1) sometimes run further?

stage1's ELF has `memsz = 0x1564cca0` (~342 MB) — absurdly oversized
compared to compiler.ls's ~1.3 MB of declared bufs. This over-allocation
is **accidental** (some separate miscount in bootstrap), and happens to
give stage1 enough mapped memory to run through early initialization
before crashing at `ldr x10, [x9]` with `x9 = 0x100000000f150`
(bit-48-set garbage pointer) — which is almost certainly a
read-from-uninitialized-bss slot further downstream.

### Disassembly of the compiler.ls crash

```
add x27, x28, #0x0   ; x27 = bss + 0 (first buf: `buf tokens 262144`)
...
mul x12, x12, #4     ; x12 = index * 4  (Wrong — should be * 12)
...
add x27, x27, x12    ; x27 = tokens + index*4
ldr x9, [x27]        ; read 32 bits
```

The accessors in `compiler.ls` were fixed to `i * 12` in `ee46577`, but
the bootstrap is emitting `mul x12, x12, #4` anyway. **Also** confirmed
by a 5-line repro: a top-level `r i * 12` statement returns 36 correctly,
but the same `i * 12` inside `→ 32 base + i * 12` produces wrong code.
So there are **two** bootstrap bugs:

1. BSS sizing via `buf` declarations is broken (primary blocker).
2. `i * 12` in memory-address expressions miscompiles (secondary).

### What we ruled out

- ✖ `emit_subscript` / `emit_assign` from `114eeb4` — stubbing both
  to no-ops does not change the crash.
- ✖ `compiler.ls` local-variable naming collisions — those were fixed
  by `ee46577` and applied cleanly.
- ✖ The `x9 = 0x100000000f150` pattern is not read from any bss slot
  (gdb `find` over `[0x500000, 0x510000)` turns up empty), so it's
  computed, not loaded.

### Next steps

- **(1)** Fix `buf`-decl → KIND_BUF registration in the Linux bootstrap
  (parser or table-building code). This unblocks the minimal repro and
  is prerequisite for everything else.
- **(2)** Find where `i * 12` in an address expression drops the `*3`
  component; likely in the expression emitter for `→ W base + expr`.
- **(3)** Re-run stage1 on `elementwise.ls` once both are fixed.

### Files of interest

- `bootstrap/lithos-elf-writer.s:510` — memsz calculation.
- `bootstrap/driver.s:458` — bss_total scan.
- `bootstrap/lithos-parser.s` — where `buf NAME SIZE` declarations
  should register into the sym table.
- `bootstrap/lithos-expr.s` — expression emitter for address-arithmetic
  that drops `*12`.
