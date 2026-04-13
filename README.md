# Lithos

Writing on silicon — a GPU compute language and inference engine for the GH200

## What It Is

Lithos is a from-scratch GPU compute stack targeting NVIDIA GH200 (Hopper, sm_90). It generates PTX directly from high-level kernel patterns written in Forth, compiles to cubins via ptxas, and loads/launches them through the CUDA driver API. No C++, no nvcc, no PyTorch, no Python in the inference path.

## Why It Exists

Five problems with existing inference stacks make batch=1 LLM inference unacceptably slow:

1. **Kernel granularity.** PyTorch dispatches dozens of tiny kernels per layer. Each launch costs 5-15us. A 32-layer model hits 1000+ launches per token — launch overhead alone exceeds compute time.

2. **Compiler quality.** nvcc/ptxas generates generic code optimized for throughput. At batch=1 on GH200, measured HBM bandwidth utilization is 13% of theoretical 4 TB/s. The compiler doesn't know the access pattern is purely sequential streaming.

3. **Memory management.** PyTorch's caching allocator fragments GPU memory with allocation pools, metadata, and reference counting. A 3B parameter model that needs ~6GB of weight storage consumes 2-4x that in PyTorch due to framework overhead.

4. **Dispatch overhead.** Python -> C++ -> CUDA runtime -> driver is a deep call stack. Each kernel launch traverses it. At batch=1 where kernels run for microseconds, the dispatch path is a significant fraction of wall time.

5. **Compilation latency.** torch.compile takes 30-120 seconds for initial compilation. Model loading through HuggingFace/transformers takes 10-30 seconds. The entire stack assumes compilation cost is amortized over long runs.

## What's Built

| Component | Description |
|-----------|-------------|
| 13 cubins | 132KB total — embed, norm, rotate, attention_score, fused_attention, fused_deltanet, conv1d, fused_mlp, activate, projection, recurrence, recurrence_rollback, sample |
| Model loader | `src/loader.py` — SafeTensors -> GPU, direct mmap |
| Engine | `src/engine.py` — kernel dispatch loop |
| Factory | `src/factory.py` — cubin compilation from PTX |
| Tokenizer | `src/tokenizer.py` — SentencePiece wrapper |
| Server | `src/server.py` — HTTP inference endpoint |
| SASS encoding | `sass/ENCODING.md` — reverse-engineered sm_90 instruction encoding |
| Benchmarks | `bench/benchmark.py` — bandwidth, latency, kernel timing |
| Kernel patterns | `core.fs`, `patterns.fs` — Forth DSL for kernel generation |

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| HBM bandwidth measured | 3.59 TB/s | 89.7% of 4 TB/s theoretical on GH200 |
| Kernel launch latency | 2.3 us | Driver API, no runtime overhead |
| Model load time | 9 ms | SafeTensors mmap, no framework |
| Cubin load time | 0.092 ms | 13 cubins, 132KB total |
| Total cubin size | 132 KB | vs ~800 MB for cuBLAS + cuDNN + PyTorch kernels |

## Architecture

```
Lithos Forth patterns  ->  PTX assembly  ->  cubin (ptxas)  ->  GPU

No C++ compiler. No nvcc. No PyTorch. No Python in inference path.
Only ptxas (NVIDIA's PTX assembler) and the CUDA driver API.
```

The pipeline:
1. Kernel patterns in Forth (`core.fs`, `patterns.fs`) emit PTX source
2. `ptxas` compiles PTX to cubins (ELF binaries with GPU machine code)
3. CUDA driver API (`cuModuleLoad`, `cuLaunchKernel`) loads and launches
4. Model weights loaded via SafeTensors mmap — no deserialization

## Quick Start

```bash
# Run proof of life — verifies the full pipeline on this machine
python src/proof_of_life.py
```

This runs three tests:
- **Vector Add**: generates PTX, compiles via ptxas, loads, launches, verifies
- **RMSNorm**: loads `norm.cubin`, launches, verifies against CPU reference
- **Embed**: loads `embed.cubin`, launches, verifies row lookup

## Kernel Inventory

13 cubins in `kernels/`:

| Cubin | Purpose |
|-------|---------|
| `embed.cubin` | Token embedding lookup — vectorized row copy |
| `norm.cubin` | RMSNorm with fused residual add |
| `rotate.cubin` | Rotary position embedding (RoPE) |
| `attention_score.cubin` | Q*K dot product with causal mask |
| `fused_attention.cubin` | Fused QKV attention (score + softmax + V multiply) |
| `fused_deltanet.cubin` | Delta network recurrence (linear attention variant) |
| `conv1d.cubin` | 1D convolution for state-space layers |
| `fused_mlp.cubin` | Fused gate + up projection + activation + down projection |
| `activate.cubin` | SiLU / GELU activation functions |
| `projection.cubin` | Matrix-vector multiply (weight * hidden) |
| `recurrence.cubin` | State-space model recurrence step |
| `recurrence_rollback.cubin` | Recurrence state rollback for speculative decoding |
| `sample.cubin` | Temperature-scaled softmax + argmax sampling |

## Documentation

Full documentation: [https://docs-delta-mauve.vercel.app](https://docs-delta-mauve.vercel.app)

## License

MIT

## Status

Early development. Proof of life passing. Not yet running full inference.
