# Walker refactor: stack-resident bindings

## Problem

The walker treats bindings as register-allocated values. This can't work
because:

1. compiler.ls has functions with 24-44 bindings — no register file fits.
2. The bump allocator never reclaims registers, so even functions with few
   bindings exhaust the pool if they evaluate many expressions.
3. No register save/restore across composition calls — inline expansion
   is the only option, which doesn't work for recursive/complex call graphs.

The ASM bootstrap has the same structural problem and is being refactored
in parallel (the "spill is broken" analysis from the Linux side).

## Design

**Core change**: every binding is a stack slot.  Registers are scratch only.

### Emitters needed (compiler.ls ARM64 section)

```
emit_a64_stur rd rn simm9    — STUR Xt, [Xn, #simm9]
emit_a64_ldur rd rn simm9    — LDUR Xt, [Xn, #simm9]
emit_a64_sub_imm rd rn imm   — SUB Xd, Xn, #imm12
emit_a64_add_imm rd rn imm   — ADD Xd, Xn, #imm12
```

### Walker state changes

Replace:
```
buf next_reg_v 8          — bump register allocator
```

With:
```
buf frame_slot_v 8        — next available frame offset (starts at 8)
buf scratch_idx_v 8       — rotating index into scratch pool (0..5)
```

### Scratch pool

Fixed 6 registers: X9, X10, X11, X12, X13, X14.  `alloc_scratch` returns
`9 + (scratch_idx++ % 6)`.  Never spills.  If an expression needs more
than 6 simultaneous values, it's a parse error (real expressions never
need more than 3-4).

### Composition prologue/epilogue

walk_top_level emits:
```
STP X29, X30, [SP, #-16]!
MOV X29, SP
SUB SP, SP, #512            ; fixed 512-byte frame (64 slots)
```

And at end:
```
ADD SP, SP, #512
LDP X29, X30, [SP], #16
RET
```

### Binding (new name)

```
slot = frame_slot; frame_slot += 8
sym_add(name, slot)
emit STUR Xresult, [X29, #-slot]
```

### Binding read

```
slot = sym_find(name)
scratch = alloc_scratch()
emit LDUR Xscratch, [X29, #-slot]
vpush(scratch)
```

### Reassignment

```
slot = sym_find(name)
emit STUR Xresult, [X29, #-slot]
```

### Composition calls

No save/restore needed — bindings live on the frame, not in registers.
Scratch registers are transient and don't survive calls.

### What gets deleted

- `alloc_reg` (bump allocator)
- `reg_reset`
- The concept of "register number" in sym_reg_v (becomes slot offset)
- vpush_with_op's alloc_reg call (uses alloc_scratch instead)
- emit_save/restore_bindings references in walk_top_level

## Test plan

1. `host main : ↓ $8 93; ↓ $0 42; trap` — trivial, no bindings
2. `a 10; b 5; c a+b` — bindings + infix
3. `i 0; i i+1; i i+1` — reassignment
4. `tc → 32 buf; idx tc*12` — buf load + binding chain
5. compiler.ls compiling test-exit42.ls — end-to-end
6. compiler.ls compiling compiler.ls — self-hosting attempt
