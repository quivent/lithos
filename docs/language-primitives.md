# Lithos Language Primitives

Two targets. One language. Every primitive maps to one instruction on the target.
No functions. No call/return. Compositions — named sequences that the compiler
flattens into instruction streams.

## Compositions

A composition is a name, its inputs, and its body. The colon begins the body.

```
rmsnorm x w D :
    each i
        sq -> 32 x i * x i
        sum Σ sq
        rms sqrt sum / D
        <- 32 out i rms * w i
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
    <- 32 out recall
```

## Memory: `->` and `<-`

`->` loads (data flows to you). `<-` stores (data flows to memory).
The number is the bit width.

```
-> 8 addr            load 8 bits    → ARM64: LDRB  / GPU: LDS.U8
-> 16 addr           load 16 bits   → ARM64: LDRH  / GPU: LDS.U16
-> 32 addr           load 32 bits   → ARM64: LDR W / GPU: LDG
-> 64 addr           load 64 bits   → ARM64: LDR X / GPU: LDG.64

<- 8 addr val        store 8 bits   → ARM64: STRB  / GPU: STS.U8
<- 16 addr val       store 16 bits  → ARM64: STRH  / GPU: STS.U16
<- 32 addr val       store 32 bits  → ARM64: STR W / GPU: STG
<- 64 addr val       store 64 bits  → ARM64: STR X / GPU: STG.64
```

Two symbols, one dispatch on a constant, one instruction on the silicon.

## Registers: `^`

`^` marks a hardware register. `^N` is register N. Not an operator — a noun.
The instruction around it decides what happens.

A register is a numbered slot on the silicon. It holds bits.
The register does not know if the bits are an address, an integer, or a float.
The next instruction decides what the bits mean.

```
^8 56                put 56 in register 8         → ARM64: MOV X8, #56
^0 addr              put addr in register 0       → ARM64: MOV X0, addr
result ^0            read register 0 into result  → ARM64: MOV Xd, X0
```

Special registers (hardware state, read-only) are named in per-architecture
dictionaries. The name is the register. The compiler emits the right instruction.

```
tid ^TID_X           GPU: S2R rd, SR_TID_X    (which thread am I?)
lane ^LANEID         GPU: S2R rd, SR_LANEID   (which lane in the warp?)
blk ^CTAID_X         GPU: S2R rd, SR_CTAID_X  (which block?)
cycles ^CNTVCT_EL0   ARM64: MRS xd, CNTVCT_EL0 (cycle counter)
```

Architecture dictionaries live in `arch/`:

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

## Trap: `trap`

Invoke the operating system. One instruction. One word.

```
trap                 → ARM64: SVC #0
```

Syscall convention is register setup before the trap:

```
open path flags mode :
    ^8 56            syscall number (openat)
    ^0 -100          AT_FDCWD
    ^1 path
    ^2 flags
    ^3 mode
    trap
    fd ^0            return value
```

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

## Memory-mapped I/O

Same `->` and `<-` — BAR0 registers are just memory addresses:

```
gsp_poke bar0 offset val :
    <- 32 bar0 + offset val

gsp_read bar0 offset :
    -> 32 bar0 + offset
```

No special MMIO primitives. Memory is memory. GPU registers are addresses.
