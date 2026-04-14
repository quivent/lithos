# What Ships

## The whole deployment

Two things on disk:

```
lithos                        compiler + runtime, ~1 MB
qwen3_w4a16.safetensors       weights, 18 GB
```

Run it:

```
./lithos qwen3_w4a16.safetensors
```

Done. No Python, no CUDA toolkit, no PyTorch, no Docker, no conda, no framework. `scp` two files, run the binary.

## What each thing is

**`lithos`** — one ARM64 ELF binary. Compiler and runtime in the same file. Reads the safetensors header, pulls architecture metadata (hidden_dim, num_heads, num_kv_heads, num_layers, hybrid pattern, RoPE theta), templates the two layer kernels from that, compiles them to raw Hopper binary in memory, hands them to the NVIDIA driver, loops `cuLaunchKernel` per token. Produced by S3-linux from the compiler's Forth source.

**`.safetensors`** — model weights. 18 GB of GPTQ-quantized tensors. `mmap`'d directly into GH200 unified memory in 9 ms. No copy, no `cudaMalloc`, no `cudaMemcpy`. The header also describes the architecture — the compiler reads it to know what kernels to generate.

That's it.

## What's NOT on disk

- No `.ls` source files. The compiler generates them from the model metadata on the fly.
- No `.cubin` / kernel files. Compiler emits raw Hopper binary in memory, hands it to the driver, done.
- No configs, no YAMLs, no tokenizer wrapper scripts, no "model card" files.

## Startup budget (model → ready to decode)

| Step | Time |
|---|---|
| Read safetensors JSON header → architecture metadata | ~1 ms |
| Template two layer kernels from architecture parameters | sub-ms |
| Parse + emit Hopper binary + wrap + `cuModuleLoadData` for both kernels | ~10–100 ms |
| `mmap` 18 GB of weights | ~9 ms |
| **Total** | **< 1 second** |

One binary, one weights file, sub-second cold start. Any safetensors that matches a supported architecture family runs without recompilation or reconfiguration.

## What a kernel is

A kernel is bytes the GPU executes. Raw Hopper instruction encodings, 128 bits each. That's what the hardware runs.

The NVIDIA driver won't accept loose instruction bytes — it wants a tiny metadata wrapper (register count, parameter layout, entry point symbol) in an ELF format called a "cubin." The word "cubin" is just "kernel in the format the driver wants."

**Kernel = cubin.** Same thing. I'll stop using "cubin."

## Where kernels live

**In memory.** The compiler produces them at runtime and hands them to the driver immediately. They don't touch the filesystem.

```
.ls source (disk)
    │
    ▼
[lithos compiler]  ──►  kernel bytes (RAM)  ──►  driver  ──►  GPU
```

The `.cubin` files sitting in `kernels/` right now are development artifacts — outputs you inspect with `nvdisasm` while building the compiler. In production they don't exist.

## Why it's this small

The conventional stack ships a Python runtime + PyTorch + CUDA toolkit + framework glue before it ships a single weight. That's gigabytes of infrastructure to express "multiply these matrices in order."

The Lithos stack expresses the same thing as:
- A language (primitives: `+ − × ÷ √` and hardware intrinsics)
- A compiler that turns the language into raw Hopper binary
- A runtime that calls the NVIDIA driver

All three fit in one 1 MB binary. The weights are the weights. There is nothing else.
