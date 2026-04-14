# Scalar on Silicon

## Using windshield wipers for propulsion

The NVIDIA GH200 has 528 tensor cores. Each one does 4,096 floating point operations per clock cycle. That's 2,162,688 operations per clock across the whole GPU. At 1.83 GHz, that's roughly 4 trillion operations per second.

We used scalar FMA. One operation per clock per core. 128 cores per SM. 132 SMs. About 241 billion operations per second.

The tensor cores deliver 16x more throughput. They're designed for exactly this operation — multiplying a tile of weights by a tile of activations and accumulating the result. It's a matrix multiply. That's what tensor cores DO.

The `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction exists. We mapped it in our SASS encoding. We used it in the prefill GEMM kernel. We used it in the projection_gemm kernel. We even included it in the Lithos patterns (pattern #67: `mma-f16-f32`).

The decode GEMV kernel — the one that runs 448 times per token, the one that accounts for 80% of execution time, the one that determines whether inference takes 0.03 seconds or 3 seconds — uses scalar FMA.

## How it happened

The first projection kernel was written by an agent told "write a correct GPTQ dequantization kernel." The agent wrote a straightforward implementation: one thread block per output element, each block does a dot product using scalar multiply-accumulate. Correct. Functional. 35.6 milliseconds per call.

An optimization agent rewrote it: 128 output elements per block, shared memory input caching, vectorized weight loads. 0.28 milliseconds. A 127x improvement. Still scalar FMA.

The optimization agent was told "make it faster." Not "use tensor cores." The agent improved the memory access pattern (which was the bottleneck it could see) and left the compute instructions alone. Reasonable — the kernel was already memory-bound, and scalar FMA at 0.28ms might have seemed fast enough.

But 0.28ms × 448 calls = 125ms per token = 8 tok/s. vLLM's cuBLAS does the same operations in about 0.003ms × 448 = 1.3ms per token = ~700 tok/s throughput (the batch-1 bottleneck is bandwidth, not compute, so the real number is lower — but the per-kernel time is 100x less).

## The tensor core GEMV challenge

There's a genuine difficulty: tensor cores operate on 2D tiles. `mma.sync.m16n8k16` multiplies a 16×16 tile by a 16×8 tile. At batch=1, the "input matrix" is a single vector — one row. The M dimension is 1.

You can't feed a 1-row input to an instruction that wants 16 rows.

The solution: treat multiple output elements as the M dimension. Instead of "one vector times one weight row = one output element," compute "one vector times 16 weight rows = 16 output elements" in one MMA instruction. The input vector is replicated across the M dimension. The weight rows become the A operand.

This is how cuBLAS does it. This is how TensorRT does it. This is how every high-performance GEMV on tensor cores works. It's a well-known technique.

We didn't do it because no agent was told to do it.

## The patterns knew

Pattern #67 in patterns.fs: `mma-f16-f32`. Emits the mma.sync instruction. Pattern #66: `tc-load-a`. Loads a matrix tile from shared memory into tensor core registers via ldmatrix. Pattern #68: `mma-bf16-f32`. BFloat16 variant.

The patterns encode the correct way to use tensor cores. They exist. They're tested. They produce valid PTX.

The kernels don't use them.

## What 100x means

At 0.28ms per projection (scalar FMA): 448 projections × 0.28ms = 125ms per token = ~8 tok/s.

At 0.003ms per projection (tensor core): 448 projections × 0.003ms = 1.3ms per token.

Add the other operations (norm, activate, DeltaNet recurrence, attention, lm_head): maybe 2ms total overhead.

Theoretical decode speed with tensor cores: ~1/(0.0013 + 0.002) = ~300 tok/s.

vLLM achieves 179 tok/s. Our architecture eliminates launch overhead and intermediate HBM traffic that vLLM still pays. The theoretical ceiling with correct kernels is higher than vLLM.

We're at 2 tok/s because we multiply with the windshield wipers.

The engine is there. The tensor cores are there. The patterns are there. The language is there. One kernel rewrite — or better, one Lithos function compiled with the patterns — and the number changes by two orders of magnitude.
