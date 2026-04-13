# Primitives → SASS

The Lithos language is 12 primitives. The compiler maps each to a known SASS
implementation pattern on Hopper. Context (operand shape) selects the pattern.

## The 12 primitives

```
*    +    -    /    exp    log    √    1/    1/√    outer    project    matvec
```

Everything else is a named composition of these. sigmoid = `* -1, exp, + 1, 1 /`.
A DeltaNet layer is 66 of them in sequence.

## Mapping

### `*` — multiply

Context determines shape:

| Usage | Meaning | SASS |
|---|---|---|
| `* -1` | negate | `FMUL Rd, Ra, -1.0` (FP32 core) |
| `*` (self) | square | `FMUL Rd, Ra, Ra` (FP32 core) |
| `* w` | scale by learned weight vector | stride loop: `LDG` weight + `FMUL Rd, Ra, Rb` per element |
| `* (two vectors)` | elementwise multiply | stride loop: `LDG` both + `FMUL` per element |

`w` is a learned weight vector — one scalar per element of the hidden state,
stored in the model weights. Every RMSNorm has one. It's not a hyperparameter
or constant; it's a per-element scale factor that the model learned during training.

All cases: one `FMUL` per element, trivially parallel. Compiler emits
parallel stride loop when operating on vectors.

### `+` — add

| Usage | Meaning | SASS |
|---|---|---|
| `+ 1` | add scalar | `FADD Rd, Ra, 1.0` (FP32 core) |
| `+ (two vectors)` | elementwise add | stride loop: `LDG` both + `FADD` per element |
| `+` (reduce) | sum all elements | **reduction pattern** (see below) |

**Reduction pattern** (the only non-trivial implementation):
1. Stride loop: each thread accumulates partial sum via `FFMA`/`FADD`
2. Intra-warp: 5x `SHFL.BFLY` + `FADD` (butterfly reduction, 32 to 1)
3. Cross-warp: `STS` to shared memory, `BAR.SYNC`, first warp loads + reduces
4. Broadcast: `STS` result to smem[0], `BAR.SYNC`, all threads `LDS`

This is the ONLY primitive that needs shared memory, barriers, and shuffles.
The compiler derives all of it from the single word `+` applied to a vector.

### `-` — subtract

| Usage | Meaning | SASS |
|---|---|---|
| `- (two values)` | subtract | `FADD Rd, Ra, -Rb` (FP32 core, negate src modifier) |

One instruction. Trivially parallel on vectors.

### `/` — divide

| Usage | Meaning | SASS |
|---|---|---|
| `/ D` | divide by constant | `MUFU.RCP Rt, D` (SFU) then `FMUL Rd, Ra, Rt` (FP32) |
| `/ (two values)` | divide | `MUFU.RCP Rt, Rb` (SFU) then `FMUL Rd, Ra, Rt` (FP32) |

Two instructions (reciprocal + multiply). SFU for the reciprocal.

### `exp` — e^x

| Usage | SASS |
|---|---|
| `exp` | `FMUL Rt, Ra, 1.442695` (FP32) then `MUFU.EX2 Rd, Rt` (SFU) |

Two instructions. Convert to base-2 (multiply by log2(e)), then hardware exp2.

### `log` — ln(x)

| Usage | SASS |
|---|---|
| `log` | `MUFU.LG2 Rt, Ra` (SFU) then `FMUL Rd, Rt, 0.693147` (FP32) |

Two instructions. Hardware log2, then convert to natural log (multiply by ln(2)).

### `√` — square root

| Usage | SASS |
|---|---|
| `√` | `MUFU.RSQ Rt, Ra` (SFU) then `MUFU.RCP Rd, Rt` (SFU) |

Two SFU ops: reciprocal-square-root then reciprocal.

### `1/` — reciprocal

| Usage | SASS |
|---|---|
| `1/` | `MUFU.RCP Rd, Ra` (SFU) |

One instruction.

### `1/√` — reciprocal square root

| Usage | SASS |
|---|---|
| `1/√` | `MUFU.RSQ Rd, Ra` (SFU) |

One instruction. This is RMSNorm's core op.

### `outer` — outer product (v ⊗ k)

| Usage | SASS |
|---|---|
| `outer` | Nested loop: for each (i,j), `FFMA M[i,j], v[i], k[j], M[i,j]` |

head_dim x head_dim iterations. Each is one `FFMA`. Parallelized across threads
within a warp (each thread handles a tile of the output matrix). The tiling
strategy is a compiler decision.

Alternatively on Hopper: `HMMA` (tensor core) for FP16 outer products, if the
dimensions align to 16x16x16 tiles.

### `project` — matrix-vector multiply with learned weights

The heaviest primitive. Decomposes into arithmetic + memory ops:

```
for each output element:
  for each group of 8 packed weights:
    load packed u32           (8 weights in 32 bits)
    load scale                (one per group of 128 weights)
    8x: extract nibble        (shift, mask)
        - integer to float
        - subtract zero point  (- 8)
        - * scale
        - * input element
        + accumulate
  + reduce across threads     (sum partial products)
```

SASS per loop iteration (one packed u32 = 8 weights):
1. `LDG` packed u32, `LDG` scale, `LDG` input element
2. 8x unroll: `LOP3` extract nibble, `I2F` convert, `FADD` zero-point, `FMUL` scale, `FFMA` accumulate
3. After all groups: warp reduction (5x `SHFL.BFLY` + `FADD`) then cross-warp smem reduction (same as `+` reduce)

40 SASS instructions per loop iteration (8 weights x 5 ops each). Memory-bandwidth
bound. Compiler's job is coalesced loads and maximal FMA throughput per byte.

Quantization scheme (W4A16, W8A16, NF4, etc.) parameterizes the dequant sequence
(step 2). The outer structure is identical.

### `matvec` — matrix-vector multiply against state matrix

| Usage | SASS |
|---|---|
| `matvec` | Dot product: for each head, `FFMA` across head_dim, then reduce |

Simpler than `project` — state matrix is FP32, no dequant. Per head:
1. Stride loop: `LDG` state row element + `LDG` query element
2. `FFMA` accumulate
3. Warp reduction (same shuffle tree)

---

## What the compiler derives (never in source)

| Concept | Derived from | Implementation |
|---|---|---|
| Thread indexing | vector operand | `S2R SR_TID.X` + `S2R SR_CTAID.X` + `IMAD` |
| Stride loops | vector length > blockDim | `ISETP` + `@P BRA` + `IADD3` stride |
| Warp shuffle tree | `+` reduce on vector | 5x `SHFL.BFLY` + `FADD` |
| Shared memory alloc | `+` reduce needing cross-warp | `STS` / `LDS` / `BAR.SYNC` |
| Barriers | cross-warp communication | `BAR.SYNC` placed by compiler |
| Register allocation | all live values | linear-scan allocator |
| Coalesced access | array indexing pattern | address arithmetic |
| Loop unrolling | `project` inner loop | compiler decides unroll factor |
| Tiling | `outer`, `project` | compiler decides tile shape |
| Grid sync | layer boundary in megakernel | cooperative barrier between layers |

None of these appear in the source language. The 66 steps in STEPS are the
complete source for a DeltaNet layer. The compiler handles everything below.

---

## Composition examples (from VOCABULARY.md)

```
sigmoid     = * -1, exp, + 1, 1 /           4 SASS instructions
SiLU        = * -1, exp, + 1, 1 /, *        5 SASS instructions
RMSNorm     = *, +, / D, 1/√, * w           ~25 SASS instructions (reduction dominates)
softplus    = exp, + 1, log                  5 SASS instructions
L2Norm      = *, +, 1/√, *                  ~25 SASS instructions (reduction dominates)
```

Every composite is a sequence of primitives. The compiler concatenates their SASS
patterns. Fusion is the default. There is nothing to fuse because nothing was
ever separated.
