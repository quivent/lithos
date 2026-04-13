# Lithos Project Status — April 13, 2026

## What Lithos Is

A Forth-derived language for writing GPU compute kernels that compile directly to SASS cubins. One language, two backends (SASS binary for NVIDIA GPUs, ARM64 native for the host). The compiler is written in Forth, hosted by an ARM64 assembly bootstrap interpreter.

## Correct Inference: YES

"The capital of France is" → " Paris" (rank 1, logit 18.057)

Verified layer-by-layer against PyTorch — cosine similarity 1.0 at all 64 layers. Six bugs were found and fixed to get here:

1. Missing SiLU after conv1d
2. Swapped in_proj_a / in_proj_b
3. Missing L2 normalization on Q and K
4. Missing 1/√d scaling on Q
5. RMSNorm uses (1+weight) not weight
6. GPTQ zero-point: 8 not 7 (auto-GPTQ stores zero_point - 1)

## Performance

| Metric | Value |
|---|---|
| Current tok/s (correct output) | **5.8** |
| vLLM on same hardware | **179** |
| Gap | **31×** |
| Cold startup | **228 ms** (vs vLLM 30-120s) |

### Why 5.8 and not 179

97% of wall-clock time is Python/ctypes dispatch overhead. The GPU finishes each kernel in microseconds, then waits milliseconds for Python to set up the next call. 448 kernel launches × ~0.7ms Python overhead each = 314ms of Python per token. GPU work is ~9ms.

### Optimization Attempts

| Approach | tok/s | Notes |
|---|---|---|
| Initial (unoptimized kernels) | 2.0 | 35ms per projection, scalar FMA |
| Tiled projection kernel | 5.6 | 0.28ms per projection, still scalar |
| Fast GEMV (1610 GB/s) | 5.6 | Faster kernel, same Python overhead |
| Multi-block fused (132 SMs) | 7.4 | One launch, grid sync between layers |
| Carmack fused (optimized) | 16.0 | 24ms GPU time, MLP-only |
| Hybrid (correct output) | 5.8 | Carmack MLP + Python DeltaNet/attention |

### Key Finding: Cooperative Grid Sync Is Wrong

Seymour Cray tested: shared memory caching, vectorized loads, higher occupancy, register reduction, prefetching. All made it slower or negligibly faster. The cooperative single-kernel with 256 grid syncs is fundamentally bottlenecked by sync overhead. More blocks = more spinning on atomics.

John Carmack got the best result (24ms GPU time) by fixing memory access patterns within the cooperative architecture. That's near the ceiling for this approach.

### Key Finding: Tensor Cores Don't Help GEMV

At batch=1, GEMV is purely memory-bandwidth-bound. Tensor cores add shared memory staging overhead that scalar FMA avoids. Our tensor core kernel was 3× SLOWER than scalar (170μs vs 55μs). cuBLAS achieves 3460 GB/s — the gap is memory access patterns (coalescing, prefetching), not compute instructions.

### The Real Path Forward

1. **Kill Python dispatch.** The launcher must be native code — Forth compiled through Eighth/S3 to ARM64 ELF, linked against libcuda.so. Or CUDA graphs replayed from native code. This alone should take 5.8 → 50+ tok/s.
2. **CUDA graph for the kernel sequence.** Record the ~130 kernel launches (fused layers), replay with one call. cuBLAS-level per-kernel performance + graph-level amortized dispatch.
3. **Fix the GEMV bandwidth gap.** 1610 GB/s vs cuBLAS 3460 GB/s. The Carmack analysis says: coalescing (warp-per-output), split-K reduction, nanosleep on sync. Applied to individual kernels (not cooperative fused), this should approach cuBLAS speeds.

## The Lithos Language

### Compiler

Written in Forth. 647 lines across 8 files. Hosted by forth-bootstrap (4831 lines ARM64 assembly). All 10 required features implemented:

- For loops, bitwise ops, warp shuffles, shared memory
- Scalar params, math intrinsics (exp/rcp/rsqrt/sqrt/sin/cos)
- Float constants (IEEE 754 hex), type conversions
- Predication, bounds checks

GEMV test kernel compiles from .li source through the Lithos compiler to a SASS cubin that loads and runs on GH200.

### Primitives (in design)

```
+  -  *  /  √
```

Five arithmetic primitives. Everything else decomposes:
- Subtraction is add with negation (* -1)
- Division by a constant is multiplication by reciprocal
- Square root via Newton's method (+ * / iterated)
- Exp via Taylor series (+ * / repeated)
- Sigmoid: * -1, exp, + 1, 1 /

Constants: e (2.71828...), computed from primitives via Taylor series to 12 terms for float32 precision.

### Kernel as Composed Math

A DeltaNet layer is 22 named operations that decompose to ~60 lines of primitives (+ * / √ on vectors and matrices). The language should express this math directly.

5 .li files replace 21 hand-written kernels:
- gemv.li (replaces 6 kernels)
- elementwise.li (replaces 4)
- reduce.li (replaces 2)
- attend.li (replaces 3)
- recur.li (replaces 4)

## Infrastructure

### GPU Kernels

21 hand-written kernels (legacy PTX artifacts, being replaced by 5 .li kernels compiled to SASS directly):
- gptq_matvec, gptq_matvec_tc, gptq_gemv_tc, gptq_gemv_fast — projection variants
- embed, embed_f16 — token embedding
- norm — RMSNorm
- activate — SiLU
- lm_head — vocabulary projection (FP16)
- residual_add, add_store, elemwise_mul — elementwise ops
- attention_score, rotate — attention
- recurrence, recurrence_rollback, conv1d — DeltaNet
- fused_attention, fused_mlp, fused_deltanet — fused layer kernels
- forward_pass, forward_pass_multi, forward_pass_carmack, forward_pass_cray, forward_pass_bellard, forward_pass_full — fused forward pass variants
- sample — argmax

### SASS Emitter

47 Hopper opcodes mapped. Automated probing tool for any NVIDIA architecture. Vector-add cubin emitted directly (no ptxas) and executes correctly on GH200. Memory op encodings fixed (descriptor-based LDG/STG, cbank param base 0x210).

### Model Loading

mmap safetensors: 9ms address setup for 18.21 GB (1999 tensors, 4 shards). Host pointers passed directly to GPU kernels — GH200 unified memory confirmed working. No cudaMalloc, no cudaMemcpy.

### Quantization

5 hardware-matched schemes designed by brilliant minds, implemented, validated:

| Scheme | Bits/weight | Cosine | Fits 66MB |
|---|---|---|---|
| Shannon SMRS v2 | 3.076 | 0.984 (v2 improved) | Yes |
| Von Neumann HGQ | 3.000 | 0.966 | Yes |
| Turing-2.93 | 2.932 | 0.954 | Yes |
| Ferrucci EBAQ | 2.993 | 0.947 | Yes |
| Lovelace ALE | 2.852 | 0.929 | Yes |

Shannon proved: cosine 0.999 at 3 bpw is impossible (rate-distortion bound). Need activation-aware calibration (GPTQ-style) or 4.5+ bpw.

Shannon v1 requantization through 64 layers: failed (cumulative error). Per-matrix cosine 0.975 is insufficient for 64-layer propagation. v2 with mixed precision (sensitive matrices get 4 bits) improves weakest link from 0.743 to 0.993.

### Documentation

23+ pages deployed at https://docs-delta-mauve.vercel.app:
- Main spec, architecture (8 subtab pages), kernels, recipe
- Performance, bandwidth, multicore, memory
- Engine comparison (11 engines), TensorRT deep dive
- Lithos vs PyTorch, language design, compiler architecture
- Quantization (overview + five minds), token journey
- Glossary (72+ terms with hover tooltips on every page)
- Warp scheduler

### Lore

10 chapters documenting the session honestly:
- The Session, Twenty-One Kernels, The Language Nobody Used
- Five Geniuses One Shelf, Scalar on Silicon
- Six Bugs, The mmap Revelation, The Bootstrap Chain
- The Brilliant Minds, The Addiction

## The Sixth Compiler

### E1 (Eighth)

137 benchmarks pass on GH200. Full Linux port (all syscalls mapped macOS→Linux). Self-hosting blocked on forward-reference depth bug (~120 forward refs with non-zero stack effects in compile.fs). Fix requires two-pass compilation or source reordering.

### Bootstrap

forth-bootstrap.s: 4831 lines ARM64 assembly. Include STATE save/restore bug fixed. Runs on GH200. Hosts the Lithos compiler.

### S3-linux

376KB ARM64 ELF binary. Runs on GH200. Can compile Forth to native ARM64. Potential host for the native launcher (replacing Python).

## What To Do Next

1. **Write the launcher in Forth, compile with S3-linux to native ARM64, link against libcuda.so.** This replaces Python entirely. The 31× gap to vLLM is mostly Python overhead.

2. **Use CUDA graphs from the native launcher.** Record the kernel sequence once, replay per token. Zero dispatch overhead.

3. **Use the Lithos compiler for the actual kernels.** The compiler works. The 5 .li files exist. Wire them into the pipeline instead of the legacy hand-written kernels.

4. **Close the GEMV bandwidth gap.** 1610 GB/s → 3460 GB/s. Carmack's coalescing + split-K techniques applied to individual kernel launches.

5. **Complete the full fused kernel.** forward_pass_full has DeltaNet, but attention layers are placeholder. Add attention for the 16 full-attention layers.

6. **Language design.** The primitives are + * / √. Everything decomposes. The language should express the math directly. A DeltaNet layer is 60 lines of primitive math operations.

7. **Deploy Shannon v2 quantization with activation-aware calibration.** The on-chip fitting thesis is proven. Quality at 3 bpw needs GPTQ-style calibration to reach 0.999 cosine.

## Repository

GitHub: https://github.com/quivent/lithos
Docs: https://docs-delta-mauve.vercel.app
