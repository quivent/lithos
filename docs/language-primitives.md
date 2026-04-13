# Lithos Language Primitives

Two targets. One language. Every primitive maps to one instruction on the target.
No functions. No call/return. Compositions — named sequences that the compiler
flattens into instruction streams.

## Compositions

A composition is a name, its inputs, and its body. The colon begins the body.

```
rmsnorm x w D :
    each i
        sq <- 32 x i * x i
        sum Σ sq
        rms sqrt sum / D
        -> 32 out i rms * w i
```

No `fn` keyword. No return arrow. The compiler inlines everything.
A composition of compositions is still just a flat instruction stream:

```
deltanet_layer x X :
    rmsnorm x w_norm D
    project W_q x q
    project W_k x k
    project W_v x v
    l2norm k
    silu q
    l2norm q
    matvec S k recall
    -> 32 out recall
```

## Memory: `<-` and `->`

`<-` loads (data comes to you). `->` stores (data goes away). The number is the bit width.

```
<- 8 addr            load 8 bits    → ARM64: LDRB  / GPU: LDS.U8
<- 16 addr           load 16 bits   → ARM64: LDRH  / GPU: LDS.U16
<- 32 addr           load 32 bits   → ARM64: LDR W / GPU: LDG
<- 64 addr           load 64 bits   → ARM64: LDR X / GPU: LDG.64

-> 8 addr val        store 8 bits   → ARM64: STRB  / GPU: STS.U8
-> 16 addr val       store 16 bits  → ARM64: STRH  / GPU: STS.U16
-> 32 addr val       store 32 bits  → ARM64: STR W / GPU: STG
-> 64 addr val       store 64 bits  → ARM64: STR X / GPU: STG.64
```

Two symbols, one dispatch on a constant, one instruction on the silicon.

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
Σ    reduction   shuffle tree (O(log N))
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

Named sequences of primitives. Not built-in — defined as compositions:

```
sigmoid x :
    neg x * -1
    ex exp neg
    sum 1 + ex
    rcp sum

silu x :
    sig sigmoid x
    x * sig

softplus x :
    ex exp x
    sum 1 + ex
    log sum

normalise x w D :
    sq x ** x
    sum Σ sq
    mean sum / D
    rms sqrt mean
    x // rms
    x ** w

l2norm x :
    sq x ** x
    sum Σ sq
    irs rsqrt sum
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

## System (ARM64 host only)

```
syscall num                   → SVC #0 (Linux ARM64 syscall)
```

## Memory-mapped I/O

Same `<-` and `->` — BAR0 registers are just addresses:

```
gsp_poke bar0 offset val :
    -> 32 bar0 + offset val

gsp_read bar0 offset :
    <- 32 bar0 + offset
```

No special MMIO primitives. Memory is memory. Registers are addresses.
