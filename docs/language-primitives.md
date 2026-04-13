# Lithos Language Primitives

Two targets. One language. Every primitive maps to one instruction on the target.
No functions. No call/return. Compositions — named sequences that the compiler
flattens into instruction streams.

## Compositions

A composition is a name, its inputs, and its body. The colon begins the body.

```
rmsnorm x w D :
    each i
        sq → 32 x i * x i
        s Σ sq
        rms sqrt s / D
        ← 32 out i rms * w i
```

The compiler inlines everything. A composition of compositions is a flat
instruction stream.

## The symbols

```
→    load from memory     (data flows to you)
←    store to memory      (data flows away)
↑    read from register   (pull value up out of register)
↓    write to register    (push value down into register)
$    register marker      ($0, $8, $TID_X)
Σ    sum                  (sum a vector)
△    max                  (max of a vector)
▽    min                  (min of a vector)
#    index                (position, not value — composable with △ ▽)
trap syscall              (invoke the operating system)
\\   comment              (to end of line)
```

## Memory: `→` and `←`

`→` loads. `←` stores. The number is the bit width.

```
→ 8 addr            load 8 bits    → ARM64: LDRB  / GPU: LDS.U8
→ 16 addr           load 16 bits   → ARM64: LDRH  / GPU: LDS.U16
→ 32 addr           load 32 bits   → ARM64: LDR W / GPU: LDG
→ 64 addr           load 64 bits   → ARM64: LDR X / GPU: LDG.64

← 8 addr val        store 8 bits   → ARM64: STRB  / GPU: STS.U8
← 16 addr val       store 16 bits  → ARM64: STRH  / GPU: STS.U16
← 32 addr val       store 32 bits  → ARM64: STR W / GPU: STG
← 64 addr val       store 64 bits  → ARM64: STR X / GPU: STG.64
```

## Registers: `↑` `↓` `$`

`$N` names a register. `↑` reads it. `↓` writes it.

```
↑ $8                 read register 8
↓ $8 56              write 56 into register 8
↑ $0                 read register 0
↓ $0 addr            write addr into register 0
```

`$N` is the same on CPU and GPU. The target determines the register file:
- Host composition: $0 = ARM64 X0 (64-bit, 0-30)
- GPU composition: $0 = Hopper R0 (32-bit, 0-255)

Special registers use names from the architecture dictionary:

```
↑ $TID_X             GPU: S2R (which thread am I?)
↑ $LANEID            GPU: S2R (which lane in the warp?)
↑ $CNTVCT_EL0        ARM64: MRS (cycle counter)
```

Architecture dictionaries:

```
arch/hopper.dict:
    TID_X       S2R  0x21
    TID_Y       S2R  0x22
    CTAID_X     S2R  0x25
    LANEID      S2R  0x00

arch/arm64.dict:
    CNTVCT_EL0  MRS  0x5F01
    MPIDR_EL1   MRS  0xC005
```

NOTE: GPU cores are more complex than CPU cores (warp scheduling,
predication, control words, shared memory). Needs its own design session.

## Trap: `trap`

Invoke the operating system. One instruction, one word.

```
open path flags mode :
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 flags
    ↓ $3 mode
    trap
    ↑ $0
```

## Reductions: `Σ` `△` `▽` `#`

Four symbols. `Σ` sums. `△` finds max. `▽` finds min. `#` returns position.

```
Σ x                  sum all elements of x
△ x                  max value in x
▽ x                  min value in x
# △ x                index of max (argmax — the token ID for sampling)
# ▽ x                index of min (argmin)
```

On GPU, all reductions compile to shuffle trees (O(log N), not loops).
`#` modifies any reduction to return the position instead of the value.

## Arithmetic

```
+    add         → ARM64: ADD   / GPU: FADD (f32) or IADD3 (u32)
-    subtract    → ARM64: SUB   / GPU: FADD(neg) or IADD3(neg)
*    multiply    → ARM64: MUL   / GPU: FMUL (f32) or IMAD (u32)
/    divide      → ARM64: SDIV  / GPU: MUFU.RCP + FMUL
```

## Dimensional notation (GPU)

```
*    scalar      one instruction
**   vector      loop of instructions (elementwise)
***  matrix      nested loop
```

## Math intrinsics (GPU: MUFU family)

```
exp    → MUFU.EX2    (2^x, use with log2(e) prescale for e^x)
log    → MUFU.LG2    (log2)
rcp    → MUFU.RCP    (1/x)
rsqrt  → MUFU.RSQ    (1/√x)
sqrt   → MUFU.SQRT
sin    → MUFU.SIN
cos    → MUFU.COS
```

## Composites

Named sequences of primitives, defined as compositions:

```
sigmoid x :
    neg x * -1
    ex exp neg
    s 1 + ex
    rcp s

silu x :
    sig sigmoid x
    x * sig

l2norm x :
    sq x ** x
    s Σ sq
    irs rsqrt s
    x ** irs
```

## Control flow

```
for i start end step          counted loop
each i                        thread-parallel iteration (GPU)
stride i dim                  stride loop (GPU, blockDim stride)
if== a b                      conditional
if>= a b                      conditional
if< a b                       conditional
```

## Memory-mapped I/O

Same `→` and `←` — BAR0 registers are just memory addresses:

```
gsp_poke bar0 offset val :
    ← 32 bar0 + offset val

gsp_read bar0 offset :
    → 32 bar0 + offset
```
