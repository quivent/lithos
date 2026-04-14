# Lithos Language

Two targets. One language. Every primitive maps to one instruction on the target.
No functions. No call/return. Compositions — named sequences that the compiler
flattens into instruction streams.

## Complete Grammar

```
STRUCTURE
    name args :             composition (body follows, indented)
    \\                      comment (to end of line)

MEMORY
    → width addr            load (8, 16, 32, 64 bits)
    ← width addr val        store (8, 16, 32, 64 bits)

REGISTERS
    ↑ $N                    read register N
    ↓ $N val                write val into register N
    $NAME                   named special register (arch dictionary)

ARITHMETIC
    +                       add
    -                       subtract
    *                       multiply
    /                       divide

DIMENSIONAL (GPU)
    **                      vector (elementwise loop)
    ***                     matrix (nested loop)

REDUCTIONS (GPU)
    Σ                       sum
    △                       max
    ▽                       min
    #                       index (# △ x = argmax)

MATH — ONE INSTRUCTION
    2^                      MUFU.EX2
    log₂                    MUFU.LG2
    √                       MUFU.SQRT
    1/                      MUFU.RCP
    1/√                     MUFU.RSQ
    ≅                       MUFU.SIN
    ≡                       MUFU.COS

MATH — COMPOSITES (TWO INSTRUCTIONS)
    e^                      * 1.44269504 then 2^
    ln                      log₂ then * 0.69314718

CONTROL
    for i start end step    counted loop
    each i                  thread-parallel (GPU)
    stride i dim            stride loop (GPU)
    if== a b                conditional
    if>= a b                conditional
    if< a b                 conditional

SYSTEM
    trap                    syscall (ARM64: SVC #0)
```

## Examples

```
\\ RMSNorm — normalize a vector by its root mean square
rmsnorm x w D :
    each i
        sq → 32 x i * x i
        s Σ sq
        rms √ s / D
        ← 32 out i rms * w i

\\ Sigmoid — 1 / (1 + e^(-x))
sigmoid x :
    neg x * -1
    ex e^ neg
    s 1 + ex
    1/ s

\\ SiLU — x * sigmoid(x)
silu x :
    sig sigmoid x
    x * sig

\\ L2 normalize
l2norm x :
    sq x ** x
    s Σ sq
    irs 1/√ s
    x ** irs

\\ Token sampling — get the index of the largest logit
sample logits :
    # △ logits

\\ Syscall — open a file
open path flags mode :
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 flags
    ↓ $3 mode
    trap
    ↑ $0

\\ MMIO — poke a GPU register via BAR0
gsp_poke bar0 offset val :
    ← 32 bar0 + offset val

\\ DeltaNet layer — composition of compositions
deltanet_layer x X :
    rmsnorm x w_norm D
    project W_q x q
    project W_k x k
    project W_v x v
    l2norm k
    silu q
    l2norm q
    matvec S k recall
    ← 32 out recall
```

## Architecture Register Dictionaries

Each target has a dictionary mapping names to hardware register IDs.
`$N` is a numbered slot. `$NAME` looks up the dictionary.

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

## Symbol Table

```
→ ← ↑ ↓ $                  memory and registers
Σ △ ▽ #                     reductions
≅ ≡                          trig (sine, cosine)
√                            square root
+ - * / ** ***              arithmetic and dimensional
```

Unicode byte sequences for the lexer:

```
→   E2 86 92    (U+2192)
←   E2 86 90    (U+2190)
↑   E2 86 91    (U+2191)
↓   E2 86 93    (U+2193)
Σ   CE A3       (U+03A3)
△   E2 96 B3    (U+25B3)
▽   E2 96 BD    (U+25BD)
√   E2 88 9A    (U+221A)
≅   E2 89 85    (U+2245)
≡   E2 89 A1    (U+2261)
```
