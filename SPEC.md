# Lithos Language Specification

## What Lithos Is

Lithos is a Forth-derived language that writes computation directly onto GPU silicon. It emits PTX (and eventually SASS) for NVIDIA GPUs, dispatched from ARM64 CPUs. No C++, no nvcc, no framework.

## Target Hardware

Primary: NVIDIA GH200 480GB (Hopper die, sm_90)
- 96GB HBM3 @ ~4 TB/s
- 72-core Grace ARM64 CPU
- NVLink-C2C coherent unified memory @ 900 GB/s
- Hardware page-level coherence between CPU and GPU address spaces

Secondary: Any NVIDIA GPU via PTX forward compatibility.

## Target Silicon: Hopper SM (GH200)

Every Lithos pattern is designed against these exact numbers.

### Per SM (132 active on GH200)

| Component | Count | Notes |
|-----------|-------|-------|
| FP32 CUDA cores | 128 | Two 16-wide datapaths per sub-partition × 4 sub-partitions |
| FP64 cores | 64 | Full-rate double precision (1:2 of FP32) |
| INT32 cores | 64 | Shares one datapath with FP32 |
| Tensor cores (4th gen) | 4 | One per sub-partition |
| Load/store units | 16 | 4 per sub-partition |
| Special function units | 16 | 4 per sub-partition (rcp, rsqrt, sin, cos, ex2, lg2) |
| Warp schedulers | 4 | One per sub-partition, 1 instruction/clock each |
| Register file | 256 KB | 65,536 × 32-bit registers, 16,384 per sub-partition |
| L1 cache + shared memory | 256 KB | Unified, configurable split (up to 228 KB shared) |
| Max resident warps | 64 | 2,048 threads, 16 warps per sub-partition |
| Max thread blocks | 32 | 8 per sub-partition |
| Max registers per thread | 255 | |

### Tensor Core Throughput Per SM

| Precision | FLOPs/clock | Shape |
|-----------|-------------|-------|
| FP8 (E4M3/E5M2) | 2,048 | m16n8k32 |
| FP16 / BF16 | 1,024 | m16n8k16 |
| TF32 | 512 | m16n8k8 |
| FP64 | 256 | m16n8k4 |
| INT8 | 2,048 | m16n8k32 |

### GPU-Wide (132 SMs, ~1830 MHz boost)

| Resource | Total |
|----------|-------|
| SMs | 132 (from 144 on die) |
| Tensor cores | 528 |
| L2 cache | 50 MB (12 partitions, 128-byte lines) |
| HBM3e stacks | 6 (4.0 TB/s total, ~667 GB/s per stack) |
| HBM3e capacity | 96 GB (on-GPU) + 480 GB LPDDR5X (Grace CPU) |
| NVLink-C2C | 900 GB/s bidirectional (CPU↔GPU coherent) |
| FP16 tensor TFLOPS | ~990 dense, ~1979 with 2:4 sparsity |
| FP8 tensor TFLOPS | ~1979 dense, ~3958 with 2:4 sparsity |

### Memory Hierarchy Latencies

| Level | Latency | Bandwidth | Line/Sector Size |
|-------|---------|-----------|------------------|
| Register file | 1 cycle | — | — |
| Shared memory | ~23 cycles | 128 bytes/clock/SM | 32 banks × 4 bytes |
| L1 data cache | ~33 cycles | 128 bytes/clock/SM | 128-byte lines |
| L2 cache | ~200 cycles | ~12 TB/s internal | 128-byte lines |
| HBM3e (global) | ~400 cycles | 4.0 TB/s | 32-byte sectors |

### Design Rules for Lithos Patterns

These are not guidelines. They are constraints. Patterns that violate them waste hardware.

1. **Coalesce at 32 bytes.** Every global memory access from a warp must hit contiguous 32-byte sectors. Misaligned access wastes bandwidth 1:1 with the misalignment.

2. **Avoid shared memory bank conflicts.** 32 banks, 4 bytes each. If two threads in a warp access the same bank (different addresses), one stalls. Pad or swizzle access patterns.

3. **Fill warps for latency hiding.** 6-8 warps per sub-partition to hide shared memory latency. 12-16 to hide global memory latency. Below this, the scheduler has nothing to issue while waiting.

4. **Match tensor core shapes.** m16n8k16 for FP16/BF16. m16n8k32 for FP8/INT8. Tiles that don't match these shapes waste tensor core cycles.

5. **Use async copy.** `cp.async` and TMA bypass registers — global memory flows directly to shared memory. Register file is the scarcest resource per thread (255 max). Don't waste it on data staging.

6. **Minimize register pressure.** More registers per thread = fewer warps = less latency hiding. Target ≤128 registers/thread for memory-bound kernels, ≤255 for compute-bound.

7. **Fuse to eliminate global memory round-trips.** Every kernel boundary writes results to HBM and the next kernel reads them back. A fused kernel keeps intermediate values in registers or shared memory. At 4 TB/s HBM vs 128 bytes/clock shared memory, the savings are 10-100x per intermediate value.

## The Problems Lithos Solves

### 1. Kernel Granularity

Current inference stacks launch 20-50 kernels per transformer layer. Each kernel boundary forces hardware synchronization: drain warps, flush pipeline, re-schedule. Inter-kernel gaps: 1-5μs each. At 64 layers, thousands of gaps. Measured result on GH200: 13% bandwidth utilization at batch=1. The GPU is idle 87% of the time.

Lithos eliminates kernel boundaries by fusing entire layers — or entire models — into single kernels. The vocabulary contains composite patterns (fused RMSNorm+residual, fused attention+softmax, fused GEMV+activation) that emit as one continuous PTX kernel. No inter-kernel gaps. No re-scheduling. The GPU stays saturated.

### 2. Compiler Quality

nvcc passes through 6-7 intermediate representations. ptxas makes generic scheduling decisions. NVIDIA's own cuBLAS uses hand-written SASS for peak GEMM performance because the compiler can't match it.

Lithos emits PTX directly. The programmer controls instruction interleaving, register allocation, shared memory access patterns, and async copy pipelines. Every pattern in the vocabulary encodes a known-good scheduling decision. The eventual path to SASS emission gives complete control over the execution pipeline.

### 3. Memory Management on Unified Memory

cudaMalloc/cudaMemcpy exist for discrete GPUs with separate memory spaces. GH200 has hardware-coherent unified memory. The entire CUDA memory management layer — page table duplication, driver ioctls, memory pool caching — is unnecessary overhead for this machine.

Lithos on GH200: allocate with mmap, pass the pointer to the kernel. The hardware handles coherence. No cudaMalloc, no memory pools, no transfer scheduling.

### 4. Dispatch Overhead

The current path: Python → PyTorch dispatcher → ATen → cuBLAS → CUDA Runtime → Driver → GPU. Measured: 10-40μs per operation. With CUDA graphs: 0.3-1μs amortized per kernel.

Lithos: Forth → driver API (cuLaunchKernel) → GPU. One call. For the fused mega-kernel approach, this is one launch per forward pass, not thousands.

### 5. Compilation Latency

nvcc compiles a FlashAttention kernel in 10-60 seconds. CUTLASS template instantiation takes hours in aggregate. TensorRT model compilation takes minutes.

Lithos compilation is string concatenation. The vocabulary words append PTX text to a buffer. A full inference kernel compiles in microseconds. This makes compilation a runtime operation — specialize kernels for the exact model config, sequence length, and batch size, recompile instantly when conditions change.

## Architecture

```
┌─────────────────────────────────────┐
│         Lithos Source                │
│  (Forth words that emit PTX)        │
└──────────────┬──────────────────────┘
               │ microseconds
               ▼
┌─────────────────────────────────────┐
│         PTX Text Buffer             │
│  (valid PTX, ready to load)         │
└──────────────┬──────────────────────┘
               │ cuModuleLoadData (driver JITs to SASS)
               ▼                      
┌─────────────────────────────────────┐
│         SASS on GPU                 │
│  (cached after first load)          │
└──────────────┬──────────────────────┘
               │ cuLaunchKernel
               ▼
┌─────────────────────────────────────┐
│         GPU Execution               │
└─────────────────────────────────────┘

Future: Lithos → SASS binary directly (skip driver JIT)
```

## Inference Engine Architecture

The inference engine's job: take model weights and a token sequence, produce the next token. Repeat.

### The Work Per Token

1. **Read weights from memory.** ~8GB for 27B W4. Every token.
2. **Matrix-vector multiply.** Activation × weights. 64 layers.
3. **Attention.** Score against all previous tokens' KV cache. Read/write KV entries.
4. **Normalize.** RMSNorm before and after attention + MLP. Elementwise.
5. **Activate.** SiLU in the MLP. Elementwise.
6. **Residual add.** Skip connections. Elementwise.
7. **Sample.** Logits → token. Once per forward pass.

Steps 1-6 repeat per layer. Step 7 once at the end.

### Engine Decisions

- **Weight placement.** HBM for hot layers, CPU memory for cold layers. GH200 unified memory makes this a soft boundary — no explicit transfer, just access latency.
- **KV cache placement.** Grows with every token. Starts in HBM, spills to CPU memory at NVLink speed when full. Not a cliff — a gradient.
- **CPU vs GPU dispatch.** Matmuls: GPU. Sampling: CPU (72 Grace cores). KV cache management: CPU. Speculative decode verification: CPU. Attention scoring: flexible — could split.
- **Overlap.** Next layer's weight prefetch overlaps current layer's compute. Attention and MLP can pipeline. CPU sampling overlaps GPU next-token prefill.
- **Batching.** Multiple requests share the same weight read. Linear throughput scaling until compute-bound. GH200 at batch=1 uses 13% bandwidth — batch=8 approaches ceiling.

### Kernel Structure

Instead of 20-50 kernels per layer:

**One kernel per layer type:**
- `attention-layer` — QKV projection, RoPE, attention scores, softmax, output projection, residual add, RMSNorm. One kernel.
- `mlp-layer` — gate projection, up projection, SiLU, element-wise multiply, down projection, residual add, RMSNorm. One kernel.
- `mtp-head` — single MTP layer for speculative drafting. One kernel.

**Or one kernel per forward pass:**
- `forward-pass` — all 64 layers fused. Weight reads pipelined. No kernel boundaries anywhere. The GPU runs a single program from token input to logit output.

### Memory Model on GH200

```
┌─────────────────────────────────────────────────────┐
│                 Unified Address Space                │
│                                                     │
│  ┌──────────────┐  ┌────────────────────────────┐  │
│  │  HBM3 96GB   │  │  LPDDR5X 480GB             │  │
│  │  ~4 TB/s     │  │  ~500 GB/s                 │  │
│  │              │  │                            │  │
│  │  Weights     │  │  KV cache overflow         │  │
│  │  KV cache    │  │  Cold weights              │  │
│  │  Activations │  │  CPU-side computation      │  │
│  │              │  │                            │  │
│  └──────┬───────┘  └──────────┬─────────────────┘  │
│         │     NVLink-C2C      │                     │
│         │     900 GB/s        │                     │
│         │   coherent          │                     │
│         └────────┬────────────┘                     │
│                  │                                  │
│    Both CPU and GPU access any address              │
│    Hardware maintains coherence                     │
│    No explicit transfers needed                     │
└─────────────────────────────────────────────────────┘
```

Lithos allocates memory once (mmap). Weights load from disk into this space. The GPU reads them directly. No cudaMalloc, no cudaMemcpy, no staging buffers.

## Vocabulary Structure

### Layer 0: PTX Primitives
Direct 1:1 mapping to PTX instructions. Every PTX instruction has a Lithos word.

```
add.f32  sub.f32  mul.f32  fma.f32  div.f32
add.f16  sub.f16  mul.f16  fma.f16
ld.global  st.global  ld.shared  st.shared
setp  selp  bra  bar.sync
mma.sync  ldmatrix  cp.async
mov  cvt  shfl.sync
```

~65 instruction words. ~20 type modifiers. ~6 memory space qualifiers.

### Layer 1: Access Patterns
Cached optimizations for common GPU memory and compute patterns.

```
global-tid          \ threadIdx.x + blockIdx.x * blockDim.x
bounds-check        \ if (tid >= n) exit
coalesced-f32@      \ addr + tid*4 → f32
coalesced-f16@      \ addr + tid*2 → f16
vec4-f32@           \ ld.global.v4.f32
warp-reduce-add     \ butterfly shuffle reduction
warp-broadcast      \ lane 0 → all lanes
block-reduce-add    \ warp reduce + shared memory reduce
```

~30 pattern words.

### Layer 2: Fused Operations
Multiple PTX instructions that always appear together in inference kernels. The fusion eliminates intermediate memory traffic.

```
fused-rmsnorm       \ square + reduce + rsqrt + scale + multiply
fused-silu          \ x * sigmoid(x) in 5 instructions
fused-rope          \ rotary position embedding, sin/cos interleaved
fused-softmax       \ max-reduce + exp + sum-reduce + divide
fused-add-rmsnorm   \ residual add + rmsnorm in one pass (one read, one write)
fused-gqa-score     \ grouped query attention score computation
w4a16-dequant-gemv  \ 4-bit weight dequant + matrix-vector in one pass
```

~30 composite words.

### Layer 3: Layer Kernels
Complete transformer layer implementations as single kernels.

```
attention-kernel    \ full attention layer: QKV + RoPE + score + softmax + output + residual + norm
mlp-kernel          \ full MLP layer: gate + up + SiLU + mul + down + residual + norm
decoder-layer       \ attention-kernel + mlp-kernel fused
mtp-draft-kernel    \ MTP head for speculative decoding
```

~10 kernel words.

### Layer 4: Engine
Forward pass orchestration. Launches kernels, manages KV cache, handles batching.

```
forward-pass        \ all layers, one token
prefill             \ process prompt tokens (batched matmul)
decode-step         \ generate one token
speculative-draft   \ generate K draft tokens via MTP
speculative-verify  \ verify draft tokens against full model
sample-token        \ logits → token (CPU-side)
```

~10 engine words.

## Compilation Model

Lithos words emit PTX text. Compilation is string concatenation — microsecond latency.

```
\ Define a kernel
s" my_kernel" kernel{
  s" ptr_weights" param-u64 ,param
  s" ptr_input"   param-u64 ,param
  s" ptr_output"  param-u64 ,param
  s" n"           param-u32
  )params

  global-tid
  bounds-check
  coalesced-f16@    \ load input
  w4a16-dequant     \ dequantize weight
  fma.f16           \ multiply-accumulate
  coalesced-f16!    \ store output
}kernel
```

This produces complete, valid PTX text in the ptx-buf. Load it with cuModuleLoadData, get a function handle, launch.

Because compilation is instantaneous, kernels can be specialized at runtime:
- Generate a kernel tuned for the exact hidden_dim, num_heads, head_dim
- Regenerate when batch size changes
- Inline constants (no parameter passing for fixed values)
- Unroll loops to exact layer count

## Execution Model

The CPU runs Lithos (Forth). Lithos compiles PTX. The driver loads PTX and JITs to SASS. The GPU executes SASS.

The GPU never runs Forth. The GPU runs whatever Lithos compiled for it.

The CPU orchestrates: token input, KV cache management, sampling, speculative decode logic. The GPU computes: matmuls, attention, normalization, activation.

On GH200, both share the same memory. The orchestration is pointer passing, not data copying.

## Quantitative Targets

| Metric | vLLM Current | Lithos Target | Why |
|--------|-------------|---------------|-----|
| Bandwidth utilization (batch=1) | 13% | >70% | Fused kernels eliminate inter-kernel gaps |
| Kernel launches per token | ~3000 | 1-64 | One per layer or one per forward pass |
| Dispatch overhead per token | ~45ms (no graphs) / ~50μs (graphs) | <10μs | One launch, no framework |
| Compilation time for kernel change | 10-60 seconds (nvcc) | <1ms | String concatenation |
| Memory management overhead | cudaMalloc pools + transfers | Zero | Unified memory, mmap only |
| Single-stream tok/s (Qwen 27B W4) | ~180 | >400 | Bandwidth utilization + fused kernels |

## File Structure

```
lithos/
├── SPEC.md          This document
├── core.fs          PTX buffer, register pools, kernel structure
├── patterns.fs      100 cached optimization patterns
├── layers.fs        Fused transformer layer kernels
├── engine.fs        Inference orchestration
└── launch.fs        CUDA driver API interface (cuModuleLoadData, cuLaunchKernel)
```

## Name

Lithos: from lithography — writing on stone. The process that puts circuits on silicon. Lithos writes on the GPU.
