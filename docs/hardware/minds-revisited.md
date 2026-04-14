# Minds Revisited: GPTQ GEMV Variants vs Current Lithos Architecture

The 12 PTX "minds" in `/home/ubuntu/lithos/minds/` were written when Lithos
was a CUDA host driving hand-written PTX kernels, parameterised at runtime
by `(qweight_ptr, scales_ptr, input_ptr, output_ptr, N, K, K_SPLITS,
k_packed_per_split)`. The architecture has since moved to:

1. **Weights-as-code (corrected form).** Q4 matrix weights stay in HBM as
   LDG-loaded data (SIMT forbids per-lane immediates), but every pointer,
   stride, N, K, group count, per-group scale offset and K_SPLITS is baked
   in at compile time by `compiler/`. No runtime parameters survive.
2. **Per-layer, per-projection specialisation.** Q_proj@L0 and Q_proj@L1
   are distinct ELF sections with distinct hardcoded dimensions. There is
   no "one kernel fits all sizes" any more.
3. **Two cooperative-grid megakernels.** GEMV is a fragment inside a
   larger fused stream, not a standalone launch. No PTX, direct SASS.
4. **Target = Qwen 3.5 27B** (hidden 5120; MLP 17408; 48 V-heads * 128;
   LM head 5120 -> 248320; 48 DeltaNet + 16 full-attn layers on GH200).

Below, each mind is evaluated against that reality. The headers inside
the `.ptx` files are sometimes mislabelled (`gptq_gemv_safe.ptx` actually
contains ultra's "4 cols/thread, v4 loads, 64 threads" code; `v8` and
`ultra` share the same strategy comment verbatim). Classifications below
reflect the *code* present in each file, not just the filename prose.

---

## 1. Strategy summary per mind

| File | Strategy (1-2 sentences) |
|---|---|
| **gptq_gemv_safe** | Despite the name, the body is the "ultra" strategy: 64 threads/block, 4 columns/thread, `ld.global.v4.u32` for 4 consecutive packed columns per load, 4 independent accumulators, split-K via grid.y + atomicAdd. Column bounds checked per lane. |
| **gptq_gemv_v2** | 256 threads/block, 1 col/thread, `ld.global.v4.u32` along **K** (4 packed words = 32 K-elements per load), 4x unroll, `ld.global.L2::128B` prefetch hints, input preloaded to smem as f32, split-K + atomicAdd. |
| **gptq_gemv_v3** | 128 threads/block, 2 cols/thread (two independent load streams per thread to double MLP), 4x unroll, v4.f32 smem loads for input, split-K + atomicAdd. |
| **gptq_gemv_v4** | Same 128t/2-col shape as v3 but does dequant + accumulate in **f16** (2x f16 throughput on SM90), input stored as f16 in smem (halves smem footprint), only the final atomicAdd is f32. |
| **gptq_gemv_v5** | Branch-free, 256t/1-col, 8x K-unroll, **scale reloaded every iteration** (no group-boundary predicate) to kill the scale-cache branch and keep the pipeline straight. |
| **gptq_gemv_v6** | Double-buffered tile prefetch of the weight matrix into smem with `cp.async` (buffer A processed while buffer B is filled); 256t, 1 col/thread. |
| **gptq_gemv_v7** | True software pipelining: issue `ld.global` for iter N+1 **before** processing iter N, so the ~200 cycle HBM latency overlaps the 32-instruction dequant chain. 128t / 2 cols. |
| **gptq_gemv_v8** | Same source strategy as `ultra`/`safe`: 64t, 4 cols/thread, v4.u32 across columns, shared input from smem, split-K. Body differs only in minor bookkeeping. |
| **gptq_gemv_v9** | 256t/1-col with **depth-4 pipeline**: 4 outstanding `ld.global.u32` at all times (`pk0..pk3` rotated), giving the assembler an explicit MLP hint. |
| **gptq_gemv_tiled** | Intended to smem-tile weights in the transposed layout, ended up reverting to the original [K/8, N] layout with streaming cache hints (`ld.global.cs`), input in smem, 4-way K unroll. |
| **gptq_gemv_transposed** | Requires a pre-transposed [N, K/8] weight layout so that each thread walks its own column stride-1 in K; uses `ld.global.v4.u32` to pull 4 packed K-values (32 K-elements) in one load per iteration. |
| **gptq_gemv_ultra** | Identical header and strategy to `safe`/`v8`: 64t, 4 cols/thread, v4.u32 across columns, shared input in smem, split-K + atomicAdd. |
| **shannon_smrs_v3_gptq.py** | Offline weight-quantizer, not a kernel. Computes `H = X^T X`, `H_inv` once, then row-block quantizes with GPTQ error-compensation `W_remaining -= (H_inv_cross / diag) @ Err`, against Shannon v2's 8-level / 16-level float codebooks. |
| **transpose_gptq_weights.py** | One-shot preprocessor that rewrites GPTQ tensors from [K/8, N] row-major to [N, K/8] column-major for use with `gptq_gemv_transposed`. |
| **test_gptq_gemv_\*.py** | Pytest-style harnesses that dequantize a reference with numpy, launch the kernel via cuda_driver, and compare with a tolerance; all they verified is numerical agreement with a synthetic f32 reference (correctness only). |

---

## 2. Relevance ranking given current architecture

Ranked most useful to the current Lithos gemv.ls -> least useful:

1. **v7 (software pipelining)** - MOST RELEVANT. "Issue load for iter N+1
   before compute iter N" is a pure scheduling idea that survives any
   weight-layout or address-compilation change. Per-layer compilation
   *helps* it, because the compiler knows the exact K of each projection
   and can size the pipeline depth precisely (e.g., Q-proj of 5120->1024
   has different latency-hiding needs than LM-head 5120->248320).
2. **v2 (v4.u32 loads along K, prefetch hints, input in smem)** - Still
   excellent. Vectorising along K (4 packed u32 = 32 K-elements per load)
   is an HBM-coalescing win independent of parameter-vs-immediate
   address handling. Every current Lithos GEMV body should emit v4
   weight loads.
3. **transposed (column-major [N, K/8] layout)** - Very relevant and now
   **cheap to deploy**: the compiler already reads safetensors at compile
   time, so it can emit weights in a transposed layout for free.
   Transposed gives stride-1 K access per thread, which is the only way
   to make v4 K-wise loads coalesce naturally. This is the single biggest
   layout-level win available.
4. **v9 (deep load pipeline, rotating registers)** - A weaker form of
   v7, but for small-K projections (e.g., DeltaNet K=128 heads, attn
   head K=256) where v7's tail overhead dominates, v9's constant-depth
   pipeline is simpler and measurably fast.
5. **v3 (2 cols/thread for MLP) / ultra-family (4 cols/thread)** - Still
   useful, and **now per-projection tuneable**. For narrow projections
   (K=1024 attn KV), 1 col/thread is right; for LM head (N=248320) 4
   cols/thread packs better. The compiler knows which is which at emit
   time.
6. **v5 (branch-free, reload scale every iter)** - Only wins when the
   group boundary is not aligned with the unroll. With per-layer
   compilation we can always align the unroll to `group_size=128`
   (16 packed u32), so v5's whole premise is dissolved by the compiler.
   Mildly useful as a fallback.
7. **v4 (f16 dequant + accumulate)** - The 2x throughput is real, but
   the accuracy risk on long K-reductions (K=5120, K=17408) is large
   enough that it needs a numerics audit before production. Park it.
8. **v6 (cp.async double-buffered weight tiling)** - Mostly obsolete for
   GEMV (a single vector of activations fits in smem easily; weights are
   the streaming side and are too large to tile usefully). cp.async is
   the *right* hammer for GEMM, not GEMV.
9. **tiled** - Same author admits mid-file that tiling the weight side
   doesn't help; the code devolves into "v2 with streaming hints". No
   unique idea remains.
10. **v8 / ultra / safe** - These three files are the same strategy
    written three times. Keep one representative (v8 body is the
    cleanest), retire the other two.

---

## 3. Ideas to incorporate into current gemv.ls

First, a correctness issue unrelated to the minds, spotted while reading
`inference/gemv.ls`: the kernel **loads the per-group scale** (`sc =
scales [ grp ]`) but **never multiplies by it**, and it never subtracts
the GPTQ zero-point (8) bias. The FMA chain uses the raw nibble as the
weight. This kernel therefore cannot be emitting correct outputs for any
GPTQ W4A16 model. Any perf numbers from it are void until the dequant
step `dq = sc * (nib - 8)` is applied before each FMA. This must be
fixed before any of the optimisations below matter.

Concrete upgrades to apply to `inference/gemv.ls`, in order of
expected impact:

### 3a. Vectorised weight loads along K (from v2)

Replace the inner `packed = W_packed[wp_idx]` (one u32 per iteration)
with a 4-wide load:

    v4.u32 (pk0, pk1, pk2, pk3) = W_packed[wp_idx..wp_idx+3]

That is 128 bits per load and 32 K-elements of work per load (4 packed
words * 8 nibbles). Unroll the inner body 4x across `pk0..pk3` sharing
one cached scale (still safe because 4 packed words = 32 K-elements
which is strictly less than group_size=128).

### 3b. Software-pipelined load (from v7)

Prologue issues `v4.u32 pkA` for iter 0. The K-loop body issues
`v4.u32 pkB` for iter i+1 **before** doing the dequant+FMA chain on
`pkA`, then swaps. This hides HBM latency behind the 32-instruction
dequant chain and typically gives 30-60% on SM90 L2-resident GEMVs.

### 3c. Per-projection specialisation (new, only possible now)

Because the compiler knows K and N of each projection at emit time, it
can pick the best shape per projection instead of a single one-size
kernel:

| Projection | K | N | Recommended shape |
|---|---|---|---|
| DeltaNet QK (heads) | 5120 | 2048 | 256t * 1col, v4-K, v7 pipeline |
| DeltaNet V        | 5120 | 6144 | 128t * 2col, v4-K, v7 pipeline |
| Full-attn QKV     | 5120 | 1024-6144 | 256t * 1col |
| MLP gate/up       | 5120 | 17408 | 64t * 4col (ultra-family), v4 across N |
| MLP down          | 17408 | 5120 | 256t * 1col, v4-K, split-K 4x |
| LM head           | 5120 | 248320 | 64t * 4col + split-N, atomicAdd |

The compiler can emit one of five template bodies chosen by a
`(K,N)` classifier. This is the single largest pay-off from
weights-as-code's per-layer compilation. None of the PTX minds could
express this because they had runtime `N`, `K` parameters.

### 3d. Transposed weight layout (from `transposed` mind + `transpose_gptq_weights.py`)

If the compiler also writes the weight-data section in [N, K/8]
column-major form at compile time, every thread's K stride becomes 1,
and the v4-K load in (3a) is a single coalesced 128-byte transaction
per warp instead of 32 scattered 4-byte lanes. This is free at compile
time and pays back on every forward pass. Strongly recommended as the
default data layout emitted by `compiler/`.

### 3e. Align K-unroll to group_size at compile time (retires v5)

The compiler should emit the inner loop with a factor chosen so that
the group boundary (every 16 packed words) coincides with an unroll
edge, removing the branch that v5 fought. This is a one-line decision
at emit time for every K that is a multiple of 128 (all current Qwen
3.5 projections qualify). v5's problem is therefore dissolved, not
solved.

---

## 4. Ideas obsolete (and why)

- **All four of "faster address arithmetic" in v2/v3/v7/v9.** The minds
  spent many registers precomputing `qw_row_stride = N*4`, `sc_row_stride
  = N*2`, etc. The current compiler bakes these as 32-bit immediates in
  the instruction stream; they cost zero registers and zero cycles.
- **Split-K + atomicAdd plumbing (all variants).** Split-K was needed to
  turn up occupancy on small N. With per-projection compilation, the
  compiler *chooses* split-K per projection based on `(K, N, SM_count)`.
  The atomicAdd race is replaced by a deterministic grid-sync partial
  reduction. All the `param_K_SPLITS`/`param_k_packed_per_split`
  parameter logic is dead code.
- **Column bounds checks (`p11/p12/p13` predicates in safe/ultra/v8).**
  These guarded the "did tid*4 overshoot N?" case at runtime. At compile
  time N is known; the compiler either sizes the grid so no lane
  overshoots, or emits a residual tail block. Predicates gone.
- **v6 (weight cp.async double buffering)** is wrong tool for GEMV:
  weights stream once (no reuse across rows within a projection), so
  tiling into smem doesn't shorten the critical path. cp.async *is* the
  right tool for activations in GEMM, not weights in GEMV.
- **tiled.** Its own source comment concedes it degenerates into v2;
  nothing new to keep.
- **v4's f16 accumulation.** On K=17408 MLP rows the f16 accumulator
  loses enough bits that outputs drift. Not worth the 2x until someone
  does a per-projection error audit.
- **safe == v8 == ultra.** Three files, one strategy. Keep v8.

---

## 5. Shannon quantizer verdict

**shannon_smrs_v3_gptq.py** implements GPTQ itself - not an inference
kernel. Specifically:

- **Algorithm**: classic Frantar-Alistarh GPTQ row-block quantization.
  Builds `H = X^T X` from calibration activations, computes `H_inv` once,
  then for each block of rows:
  1. Quantize the block against a Shannon v2 float **codebook** (8-level
     for non-sensitive, 16-level for `down_proj` / `out_proj`),
  2. Compute per-element error,
  3. Compensate unseen rows: `W_remaining -= (H_inv_cross / diag) @ Err`,
     with a 0.1 * stddev clip for stability.
- **Difference from standard GPTQ**: the quantizer is **non-uniform /
  codebook-based** (Shannon SMRS codebooks, 8 or 16 float levels
  chosen by Lloyd-Max-style design), not the uniform int4 grid that
  HuggingFace GPTQ uses. Also uses LS-refined per-group (scale, zero)
  re-fit inside each block, whereas vanilla GPTQ just picks scale/zero
  from range.
- **Relevance today**: **not relevant for the current inference path.**
  Lithos targets `Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16`, which is
  already quantized to the uniform int4 + f16 scale + packed u32 layout
  that `gemv.ls` and the compiler expect. The Shannon codebook is
  incompatible with that layout (dequant would need a codebook lookup,
  not a `scale*(nib-8)` FMA chain). So using `shannon_smrs_v3` on the
  deployed model would require rebuilding both the weight format and
  every GEMV kernel.
- **Future role**: keep it around as a **re-quantization research tool**.
  Its measured wins on layer 0 (output-cosine +0.005 vs v2, reaching
  0.997-0.999 on 4-bit matrices) are genuine and would justify a
  retrain-quantize pass if Lithos ever ships its own quantized weights.
  Until then, it is parked, not promoted.

---

## 6. One-line takeaway

The minds were all hand-tunings of a parametric CUDA kernel. Per-layer
compilation collapses half of them into compile-time decisions
(addresses, strides, split-K, bounds) and elevates the other half (v2
v4-K loads, v7 software pipelining, transposed layout, per-(K,N)
shape choice) into templates the compiler should pick between. Fix the
missing scale+zero dequant in `gemv.ls` first; then bring in (3a), (3b),
(3c), (3d) in that order.
