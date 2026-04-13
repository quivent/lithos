# Lithos

Writing on silicon -- a self-hosting GPU compute language for the GH200.

**Documentation:** [docs-delta-mauve.vercel.app](https://docs-delta-mauve.vercel.app)
**GitHub:** [github.com/quivent/lithos](https://github.com/quivent/lithos)

## What Lithos Is

Lithos is a language that compiles .ls source files directly to SASS (GPU machine code) and ARM64 (host machine code). One source language, two backends. No Python. No C. No CUDA toolkit. No framework.

The compiler is written in Lithos (.ls), bootstrapped from pure ARM64 assembly (bootstrap/*.s). The runtime replaces libcuda.so with direct GPU register writes via vfio-pci. The entire system -- compiler, runtime, and inference kernels -- ships as one ARM64 binary.

Target: **Qwen 3.5 27B** (Huihui-abliterated, GPTQ W4A16) on GH200 480GB.
64 hybrid layers: 3 DeltaNet + 1 full attention x 16.

## Architecture

The self-hosting compiler (compiler/compiler.ls, 4739 lines) has 7 sections:
ARM64 backend, GPU backend, Lexer, Parser, Safetensors reader, ELF writer, Main entry.

Supporting infrastructure:
- bootstrap/*.s -- pure ARM64 assembly bootstrap (builds compiler)
- runtime/*.ls -- libcuda replacement (vfio-pci, QMD, GPFIFO, doorbell)
- inference/*.ls -- 10 kernel source files (new grammar)
- GSP/*.s -- GPU System Processor boot (ARM64 assembly)

## The Language

Compositions (named sequences), not functions. The compiler flattens them into instruction streams. See docs/language-primitives.md for the full grammar.

## Inference Kernels

10 .ls source files in inference/, covering the full Qwen 3.5 27B forward pass:
attend.ls, attend_full.ls, decay_gate.ls, delta_update.ls, deltanet_fused.ls,
elementwise.ls, embed.ls, gemv.ls, recur.ls, reduce.ls.

## Runtime (libcuda replacement)

11 .ls files in runtime/ replace all 42 libcuda.so call sites.

## What Ships

Two things on disk: lithos (compiler + runtime, ~1 MB ARM64 ELF) and the safetensors weights file (18 GB). Run: ./lithos qwen3_w4a16.safetensors

No Python, no CUDA toolkit, no PyTorch, no Docker, no conda.

## Current State

See PLAN.md for detailed progress. The compiler, kernels, and runtime are written.
Bootstrap parser work is blocking first compilation.

## Documentation

26 pages at docs-delta-mauve.vercel.app. Key documents:
- docs/language-primitives.md -- grammar spec
- docs/model-config.md -- Qwen 3.5 27B configuration
- PLAN.md -- execution plan and progress tracking

## License

MIT
