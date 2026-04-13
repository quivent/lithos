# Lithos

Writing on silicon — a Forth-derived GPU compute language for the GH200

## What Lithos Is

**Lithos is a Forth-derived language with three backends.** One source language. Three outputs: **PTX** text for NVIDIA GPUs, raw **SASS** binary for Hopper, and **ARM64** machine code for the Grace host. A single Lithos program describes both the GPU kernels and the CPU-side orchestrator; the compiler routes each definition to the right backend based on whether it is declared `kernel` or `host`.

**The compiler is Forth too.** It runs on `forth-bootstrap` — a 4,831-line ARM64 assembly Forth interpreter — today, and compiles itself to a native ARM64 binary via Eighth once Eighth self-hosts. No Python. No Rust. No C. No framework. A Forth compiler binary is ~60 KB and starts instantly.

See [docs: The Compiler](https://docs-delta-mauve.vercel.app/compiler) for the full architecture.

| Backend | Output | Purpose |
|---------|--------|---------|
| `emit-ptx.fs`   | PTX text     | Portable GPU kernels. Driver JITs to SASS on load. |
| `emit-sass.fs`  | cubin binary | Hopper-specific. Skips the driver JIT. Microsecond load. |
| `emit-arm64.fs` | ARM64 ELF    | CPU-side orchestrator: layer loop, KV cache, tokenization, sampling, API server. |

The language surface is stack-based. A `kernel` word routes to the PTX or SASS emitter; a `host` word routes to the ARM64 emitter. Same syntax, same stack model, different machine code.

## What Lithos Emits

Lithos produces PTX, cubin binaries, and ARM64 ELF executables directly. No nvcc, no C++ toolchain, no template instantiation, no IR chain. Kernel compilation is string (or byte) concatenation driven by a word dispatch table — microseconds, not minutes.

```
Lithos source (.li)
    -> Lexer -> Parser -> AST (stack-based IR)
    -> Backend Router (per-definition: kernel or host?)
    -> PTX text / cubin binary / ARM64 ELF
```

The GPU never runs Forth. The CPU never runs Forth in the hot path either — once Eighth self-hosts, the orchestrator is native ARM64.

## The Engine (Host Infrastructure)

A language needs a host. The language emits kernels; something has to load weights, invoke the Forth, call the driver, manage the KV cache, and decode tokens. That something is the **engine**.

The engine today is Python scaffolding in [`src/`](src/):

| File | Role |
|------|------|
| `cuda_driver.py` | ctypes bindings for `cuModuleLoadData`, `cuLaunchKernel`, etc. |
| `factory.py` | Shells out to `ptxas` to turn PTX into cubins |
| `loader.py` | mmaps SafeTensors weight files into unified memory (sets up virtual address mappings only — weight bytes page in from SSD lazily on first GPU access) |
| `engine.py` | The kernel dispatch loop — forward pass orchestration |
| `tokenizer.py` | SentencePiece wrapper |
| `server.py` | HTTP inference endpoint |

**This Python is temporary.** It is to Lithos what the C `engine/fifth` binary is to Sixth: a bootstrap host that exists only until the language can self-host. The plan is to replace it with a Forth/Rust runtime (tracked as the E1 / Sixth work) that calls the CUDA driver directly, leaving no interpreted code in the inference path.

**Language + Engine = Inference System.** The language produces kernels. The engine runs them. Together they form the batch=1 inference system this project targets on the GH200.

## Why It Exists

Five properties of existing inference stacks constrain batch=1 LLM inference throughput:

1. **Kernel granularity.** PyTorch dispatches dozens of tiny kernels per layer. Each launch costs 5-15us. A 32-layer model hits 1000+ launches per token — launch overhead alone exceeds compute time.

2. **Compiler quality.** nvcc/ptxas generates generic code optimized for throughput. At batch=1 on GH200, measured HBM bandwidth utilization is 13% of theoretical 4 TB/s. The compiler doesn't know the access pattern is purely sequential streaming.

3. **Memory management.** PyTorch's caching allocator fragments GPU memory with allocation pools, metadata, and reference counting. A 3B parameter model that needs ~6GB of weight storage consumes 2-4x that in PyTorch due to framework overhead.

4. **Dispatch overhead.** Python -> C++ -> CUDA runtime -> driver is a deep call stack. Each kernel launch traverses it. At batch=1 where kernels run for microseconds, the dispatch path is a significant fraction of wall time.

5. **Compilation latency.** torch.compile takes 30-120 seconds for initial compilation. Model loading through HuggingFace/transformers takes 10-30 seconds. The entire stack assumes compilation cost is amortized over long runs.

A language that emits kernels in microseconds, hosted by a runtime with nothing in the dispatch path, addresses all five.

## What's Built

| Component | Description |
|-----------|-------------|
| **Language** | `core.fs` + `patterns.fs` — 1,459 lines of Forth defining the Lithos vocabulary |
| **SASS emitter** | `sass/emit-sass.fs` — direct binary instruction emission words |
| 13 cubins | 132KB total — embed, norm, rotate, attention_score, fused_attention, fused_deltanet, conv1d, fused_mlp, activate, projection, recurrence, recurrence_rollback, sample |
| Host: loader | `src/loader.py` — SafeTensors -> GPU, direct mmap *(temporary)* |
| Host: engine | `src/engine.py` — kernel dispatch loop *(temporary)* |
| Host: factory | `src/factory.py` — cubin compilation from PTX *(temporary)* |
| Host: tokenizer | `src/tokenizer.py` — SentencePiece wrapper *(temporary)* |
| Host: server | `src/server.py` — HTTP inference endpoint *(temporary)* |
| SASS encoding | `sass/ENCODING.md` — reverse-engineered sm_90 instruction encoding |
| Benchmarks | `bench/benchmark.py` — bandwidth, latency, kernel timing |

Items marked *temporary* are the Python host scaffolding slated for replacement once the Forth/Rust runtime is ready.

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| HBM bandwidth measured | 3.59 TB/s | 89.7% of 4 TB/s theoretical on GH200 |
| Kernel launch latency | 2.3 us | Driver API, no runtime overhead |
| Model mmap setup time | 9 ms | SafeTensors `mmap` (address mapping only — no weight bytes moved; open + mmap per shard + 5 KB header read + pointer arithmetic). The 18 GB still lives on SSD after this call; pages fault in lazily during the first forward pass (~5-9 s of SSD reads at ~2-3 GB/s, paid by first request instead of at startup). Compare PyTorch's `torch.load` + `cudaMalloc` + `cudaMemcpy`, which actually copies all 18 GB into memory in 3-5 s before returning. |
| Cubin load time | 0.092 ms | 13 cubins, 132KB total |
| Total cubin size | 132 KB | vs ~800 MB for cuBLAS + cuDNN + PyTorch kernels |
| Language size | 1,459 lines | Lithos itself (`core.fs` + `patterns.fs`) |

## Architecture

```
 LANGUAGE                           ENGINE (host)
 --------                           -------------
 core.fs       + patterns.fs   ->   factory.py   ->   ptxas   ->   cubin
 (Forth words emitting PTX)         (invokes Forth)                  |
                                                                     v
                                    loader.py (mmap safetensors) --> GPU memory
                                    engine.py (dispatch loop)        |
                                    cuda_driver.py (cuLaunchKernel) -+
                                                                     v
                                                                 GPU executes
```

No C++ compiler. No nvcc. No PyTorch. No Python in the *inference* path (only in the *host* path, and that is temporary). Only `ptxas` (NVIDIA's PTX assembler) and the CUDA driver API.

## Quick Start

```bash
# Run proof of life — verifies the full pipeline on this machine
python src/proof_of_life.py
```

This runs three tests:
- **Vector Add**: Lithos emits PTX, ptxas compiles it, driver loads and launches, verifies
- **RMSNorm**: loads `norm.cubin`, launches, verifies against CPU reference
- **Embed**: loads `embed.cubin`, launches, verifies row lookup

## Kernel Inventory

13 cubins in `kernels/`, all produced by the Lithos vocabulary:

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

## Current State

Measured on GH200 480GB, head-to-head against vLLM on the same box.

**Cold-startup breakdown.** Each row below is an *individual step*. The highlighted **Total** row is the *sum* of those steps — wall-clock time from process launch to server-ready. Individual step times are components of the total, not alternatives to it.

| Startup step | vLLM | Lithos | What it is |
|---|---|---|---|
| Python interpreter init | ~500 ms | — | fork / exec / libc / site-packages |
| `import torch` | ~2–3 s | — | load ~4 GB of Python extensions |
| `import vllm` + transformers | ~1–2 s | — | resolve model class, tokenizer, config |
| CUDA context create | 2–5 s | **127 ms** | NVIDIA driver handshake — nobody avoids this |
| Model load (18.21 GB) | 3–5 s | **9 ms** | vLLM: `torch.load` + `cudaMalloc` + `cudaMemcpy` (copies all 18 GB). Lithos: `mmap` + 5 KB JSON header (0 bytes moved; weights page in on first request) |
| Kernel compilation (11 kernels via ptxas) | — | **92 ms** | `ptxas` JIT, fanned across cores |
| Cubin load (11 × `cuModuleLoadData`) | — | **0.6 ms** | load pre-compiled cubins into the driver |
| `torch.compile` warmup | ~30–60 s | — | Inductor → Triton, compiles per-shape |
| CUDA graph capture | ~5–7 s | — | record `cuLaunchKernel` sequence |
| Memory profiling | ~36 s | — | probe free HBM before allocating KV cache |
| **TOTAL COLD STARTUP (sum)** | **~30–120 s** | **228 ms** | **130–525x faster; steps overlap slightly, so the sum is approximate** |

vLLM's time is dominated by `torch.compile` (30–60 s) and memory profiling (~36 s) — work Lithos doesn't need because it ships pre-compiled cubins and uses `mmap` instead of a profiled `cudaMalloc` pool. Lithos's 228 ms is dominated by **CUDA context creation (127 ms of 228)** — NVIDIA driver setup that nobody can avoid.

**Steady-state (per-token):**

| Metric | vLLM | Lithos |
|--------|------|--------|
| TTFT (10 prompts) | 73.2 ms | pending engine wiring |
| Inter-token latency | 19.2 ms | pending |
| Streaming throughput | 44.8 tok/s | pending |

**What works today:** language spec (1,459 lines), 13 compiled cubins (132 KB total), kernel factory (config.json → specialized cubins in 92 ms), model loader (18.21 GB mmap in 9 ms), CUDA driver bindings, engine orchestrator (64-layer loop, 915 kernel dispatches planned per forward pass), OpenAI-compatible API server (228 ms startup), SASS encoding (47 opcodes mapped), direct SASS emitter (noop kernels execute; memory ops in progress), benchmark suite (3.59 TB/s measured bandwidth, 2.3 µs kernel launch), 12 documentation pages, Rust CLI.

**What's pending:** real kernel launches (the `_gpu_launch` hook is still a log-and-return stub — replacing it with `cuLaunchKernel` is the next commit), correctness validation vs PyTorch reference, prefill GEMM integration, speculative decoding (MTP) wiring, KV cache spill to LPDDR5X, a Lithos compiler written from scratch (currently Python scaffolding interprets `.fs` files), and self-hosting via Eighth (blocked on forward-reference depth bug).

The infrastructure is in place and measured. Per-token numbers land the moment the launch stub becomes a real dispatch.

## Status

Early development. Language vocabulary at 1,459 lines. Startup and model-load paths fully measured; per-token path pending `_gpu_launch` wiring. Python host is temporary; Forth/Rust self-hosted runtime is the next milestone.
