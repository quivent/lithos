# Primitives → SASS

The Lithos language is 12 primitive families. The compiler maps each to a known SASS
implementation pattern on Hopper. Context (operand shape) selects the pattern.

## The primitives

```
*   **   ***   ****        multiply
+   ++   +++   ++++        add
-   --   ---   ----        subtract
/   //   ///   ////        divide
Σ                          reduce (shuffle tree)
exp   log   √   1/   1/√  scalar math
project   matvec           GEMV
```

Every extra symbol = one more `for` loop around the inner SASS.
`outer` is gone — it is `***`.
`exp`, `log`, `√`, `1/`, `1/√` are scalar-only; apply elementwise by wrapping in `**`.

Everything else is a named composition of these. sigmoid = `* -1, exp, ++, //`.
A DeltaNet layer is 66 of them in sequence.

## Mapping

### `*` / `**` / `***` / `****` — multiply

Loop depth = number of `*` symbols. Inner SASS is always `FMUL` (or `FFMA`).

| Symbol | Loop depth | Usage | SASS |
|--------|-----------|-------|------|
| `*`    | 0 (scalar) | `* -1` negate; `* w` scale | `FMUL Rd, Ra, Rb` (FP32 core) |
| `**`   | 1          | elementwise vector multiply | `for i: FMUL Rd[i], Ra[i], Rb[i]` |
| `***`  | 2          | outer product — for each (i,j) | `for i: for j: FMUL M[i,j], A[i], B[j]` |
| `****` | 3          | matmul index space | `for i: for j: for k: FFMA C[i,j], A[i,k], B[k,j], C[i,j]` |

`**` is the workhorse: stride loop, each thread handles a tile, trivially parallel.
`***` (formerly `outer`): head_dim × head_dim iterations. Each is one `FMUL`.
Parallelized across threads within a warp; each thread owns a tile of the output matrix.
Tiling strategy is a compiler decision.

On Hopper: `***` may emit `HMMA` (tensor core) for FP16 outer products when dimensions
align to 16×16×16 tiles.

`w` in `* w` is a learned weight vector — one scalar per element of the hidden state,
stored in the model weights. Every RMSNorm has one.

### `+` / `++` / `+++` / `++++` — add

| Symbol | Loop depth | Usage | SASS |
|--------|-----------|-------|------|
| `+`    | 0 (scalar) | `+ 1` add scalar constant | `FADD Rd, Ra, 1.0` (FP32 core) |
| `++`   | 1          | elementwise vector add | `for i: FADD Rd[i], Ra[i], Rb[i]` |
| `+++`  | 2          | elementwise matrix add | `for i: for j: FADD M[i,j], A[i,j], B[i,j]` |
| `++++` | 3          | elementwise tensor add | `for i: for j: for k: FADD T[i,j,k], A[i,j,k], B[i,j,k]` |

`+++` is the state-update add: `S = decay_S +++ correction_matrix`.

### `-` / `--` / `---` / `----` — subtract

| Symbol | Loop depth | Usage | SASS |
|--------|-----------|-------|------|
| `-`    | 0 (scalar) | scalar subtract | `FADD Rd, Ra, -Rb` (negate src modifier) |
| `--`   | 1          | elementwise vector subtract | `for i: FADD Rd[i], Ra[i], -Rb[i]` |
| `---`  | 2          | elementwise matrix subtract | `for i: for j: FADD M[i,j], A[i,j], -B[i,j]` |
| `----` | 3          | elementwise tensor subtract | `for i: for j: for k: FADD T[i,j,k], A[i,j,k], -B[i,j,k]` |

All cases use `FADD` with a negate source modifier — no dedicated subtract instruction.

### `/` / `//` / `///` / `////` — divide

| Symbol | Loop depth | Usage | SASS |
|--------|-----------|-------|------|
| `/`    | 0 (scalar) | scalar divide | `MUFU.RCP Rt, Rb` then `FMUL Rd, Ra, Rt` |
| `//`   | 1          | elementwise reciprocal/divide | `for i: MUFU.RCP Rt, Rb[i]; FMUL Rd[i], Ra[i], Rt` |
| `///`  | 2          | elementwise matrix divide | `for i: for j: MUFU.RCP Rt, B[i,j]; FMUL M[i,j], A[i,j], Rt` |
| `////` | 3          | elementwise tensor divide | `for i: for j: for k: MUFU.RCP + FMUL` |

Two instructions per element (SFU reciprocal + FP32 multiply).

### `Σ` — reduction (collapse one dimension)

| Usage | SASS |
|---|---|
| `Σ` | Shuffle tree: stride loop partial sums → 5× `SHFL.BFLY` + `FADD` → smem cross-warp |

`Σ` is NOT a loop. It collapses one dimension via the shuffle tree — O(log N).
This is the ONLY primitive that requires shared memory, barriers, and shuffles.

Implementation:
1. Stride loop: each thread accumulates partial sum via `FFMA`/`FADD`
2. Intra-warp: 5× `SHFL.BFLY` + `FADD` (butterfly reduction, 32→1)
3. Cross-warp: `STS` to shared memory, `BAR.SYNC`, first warp loads + reduces
4. Broadcast: `STS` result to smem[0], `BAR.SYNC`, all threads `LDS`

The compiler derives all of this from the single symbol `Σ` applied to a vector.

### `exp` — e^x

| SASS |
|---|
| `FMUL Rt, Ra, 1.442695` (FP32) then `MUFU.EX2 Rd, Rt` (SFU) |

Two instructions. Convert to base-2 (multiply by log2(e)), then hardware exp2.

### `log` — ln(x)

| SASS |
|---|
| `MUFU.LG2 Rt, Ra` (SFU) then `FMUL Rd, Rt, 0.693147` (FP32) |

Two instructions. Hardware log2, then convert to natural log (multiply by ln(2)).

### `√` — square root

| SASS |
|---|
| `MUFU.RSQ Rt, Ra` (SFU) then `MUFU.RCP Rd, Rt` (SFU) |

Two SFU ops: reciprocal-square-root then reciprocal.

### `1/` — reciprocal

| SASS |
|---|
| `MUFU.RCP Rd, Ra` (SFU) |

One instruction.

### `1/√` — reciprocal square root

| SASS |
|---|
| `MUFU.RSQ Rd, Ra` (SFU) |

One instruction. This is RMSNorm's core op.

### `project` — matrix-vector multiply with learned weights

The heaviest primitive. W4A16 GPTQ dequant GEMV. Decomposes into:

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
  Σ across threads            (sum partial products)
```

SASS per loop iteration (one packed u32 = 8 weights):
1. `LDG` packed u32, `LDG` scale, `LDG` input element
2. 8× unroll: `LOP3` extract nibble, `I2F` convert, `FADD` zero-point, `FMUL` scale, `FFMA` accumulate
3. After all groups: warp reduction (5× `SHFL.BFLY` + `FADD`) then cross-warp smem reduction

40 SASS instructions per loop iteration (8 weights × 5 ops each). Memory-bandwidth
bound. Compiler's job is coalesced loads and maximal FMA throughput per byte.

This operation is effectively `****Σ` (matmul index space + reduction + W4A16 dequant).
The name `project` is kept because the dequant sequence is too specific to encode in stars.

Quantization scheme (W4A16, W8A16, NF4, etc.) parameterizes the dequant sequence.
The outer structure is identical across schemes.

### `matvec` — matrix-vector multiply against state matrix

| SASS |
|---|
| `matvec` | Dot product: for each head, `FFMA` across head_dim, then `Σ` |

Simpler than `project` — state matrix is FP32, no dequant. Per head:
1. Stride loop: `LDG` state row element + `LDG` query element
2. `FFMA` accumulate
3. Warp reduction (same shuffle tree as `Σ`)

This operation is effectively `***Σ` (matrix × vector + reduction per row, FP32).
The name `matvec` is kept because the FP32 state access pattern is structurally distinct.

---

## What the compiler derives (never in source)

| Concept | Derived from | Implementation |
|---|---|---|
| Thread indexing | vector operand | `S2R SR_TID.X` + `S2R SR_CTAID.X` + `IMAD` |
| Stride loops | vector length > blockDim | `ISETP` + `@P BRA` + `IADD3` stride |
| Warp shuffle tree | `Σ` on vector | 5× `SHFL.BFLY` + `FADD` |
| Shared memory alloc | `Σ` needing cross-warp | `STS` / `LDS` / `BAR.SYNC` |
| Barriers | cross-warp communication | `BAR.SYNC` placed by compiler |
| Register allocation | all live values | linear-scan allocator |
| Coalesced access | array indexing pattern | address arithmetic |
| Loop unrolling | `project` inner loop | compiler decides unroll factor |
| Tiling | `***`, `project` | compiler decides tile shape |
| Grid sync | layer boundary in megakernel | cooperative barrier between layers |

None of these appear in the source language. The 66 steps in STEPS are the
complete source for a DeltaNet layer. The compiler handles everything below.

---

## Composition examples

```
sigmoid     = * -1, exp, ++, //               4 SASS patterns
SiLU        = * -1, exp, ++, //, **           5 SASS patterns
RMSNorm     = **,   Σ,   / D, 1/√, **        ~25 SASS instructions (Σ dominates)
softplus    = exp,  ++,  log                   5 SASS instructions
L2Norm      = **,   Σ,   1/√,   **           ~25 SASS instructions (Σ dominates)
```

State update (the delta rule):

```
S = decay ** S  +++  beta ** (v -- recall) *** k
```

- `decay ** S`      — elementwise decay of state matrix (O(N×M) loops)
- `v -- recall`     — elementwise vector subtract (O(N))
- `beta **`         — elementwise scale by beta (O(N))
- `*** k`           — outer product to form correction matrix (O(N×M))
- `+++`             — matrix add: decayed S plus correction (O(N×M))

Every composite is a sequence of primitives. The compiler concatenates their SASS
patterns. Fusion is the default. There is nothing to fuse because nothing was
ever separated.
