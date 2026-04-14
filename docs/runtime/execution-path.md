# SASS to Silicon — Execution Path

How a raw SM90 binary reaches the GPU's streaming multiprocessors,
bypassing libcuda entirely.

```
Your SASS binary (raw bytes, SM90 encoding)
    ↓ loaded into GPU-visible memory (BAR4 or GPU VA)
    ↓
QMD (Queue Meta Data) — 64-byte struct pointing at:
    - code address (where your SASS lives)
    - shared memory size
    - grid dimensions (blocks × threads)
    - register count
    ↓ QMD written into a compute method in the pushbuffer
    ↓
Pushbuffer — sequence of (method, data) pairs:
    - SET_PROGRAM_A/B (code address)
    - SET_QMD (QMD address)
    - LAUNCH_DMA (fire)
    ↓ pushbuffer sits in GPFIFO-accessible memory
    ↓
GPFIFO entry — (base address, length) pointing at pushbuffer
    ↓ written to the GPFIFO ring buffer
    ↓
Doorbell MMIO write — poke BAR0 register to notify GPU
    ↓ GPU's PBDMA reads GPFIFO entry
    ↓ PBDMA parses pushbuffer methods
    ↓ Host/ESCHED routes to compute engine
    ↓ CWD distributes thread blocks to SMs
    ↓
SM fetches your SASS from memory and executes it
```

No firmware in the execution path. No signing. Your bytes hit the ALUs.

## What each stage does

**SASS binary** — raw SM90 machine code. Each instruction is 128 bits:
lower 64 bits are the opcode, upper 64 bits are control codes (stall
counts, yield flags, dependency barriers, write barriers). Emitted by
`compiler/emit-gpu.ls`, encoded per `sass/encoding_sm90.json`.

**GPU-visible memory** — the kernel binary must live in memory the GPU
can fetch from. On GH200 with C2C/ATS, CPU and GPU share the same
physical address space, so BAR4 allocations or pinned system memory
both work. `runtime/mem.ls` handles allocation.

**QMD** — the GPU's equivalent of a thread launch descriptor. Contains
everything the hardware needs to dispatch a grid: code address, grid
dimensions, shared memory size, register count, constant buffer
bindings. Built by `runtime/qmd.ls`, field layout documented in
`docs/runtime/qmd-hopper.md`.

**Pushbuffer** — a sequence of GPU class methods (32-bit method ID +
32-bit data). The compute class methods configure the engine state and
trigger the launch. Built by `runtime/pushbuffer.ls`.

**GPFIFO** — a ring buffer of (address, length) entries, each pointing
at a pushbuffer segment. The GPU's PBDMA (Push Buffer DMA engine)
reads entries from this ring. Managed by `runtime/launch.ls`.

**Doorbell** — a single 32-bit MMIO write to a BAR0 register that
tells the GPU "new GPFIFO entries are available." This is the moment
the GPU starts executing. `runtime/doorbell.ls`.

**PBDMA → Host/ESCHED → Compute Engine → CWD → SM** — all hardware.
PBDMA parses the pushbuffer methods, Host unit routes to the compute
engine, CWD (Compute Work Distributor) assigns thread blocks to SMs
based on available resources, and the SM's instruction fetch unit
starts pulling your SASS from memory.

## What's NOT in the path

- **GSP firmware** — manages power, memory, error handling. Not in the
  execution datapath. Needed for setup, not for running kernels.
- **libcuda.so** — lithos replaces this entirely with `runtime/*.ls`.
- **ptxas** — lithos emits SM90 binary directly, no PTX intermediate.
- **CUDA runtime** — no cudaLaunchKernel, no driver API, no context.

## Lithos source map

| Stage | Source | Purpose |
|---|---|---|
| SASS encoding | `sass/encoding_sm90.json` | Opcode field definitions |
| SASS emission | `compiler/emit-gpu.ls` | Compiler GPU backend |
| Kernel binaries | `kernels/*.sass`, `kernels/*.cubin` | Compiled kernels |
| Memory allocation | `runtime/mem.ls` | GPU-visible memory |
| QMD construction | `runtime/qmd.ls` | Launch descriptor |
| Pushbuffer | `runtime/pushbuffer.ls` | Method stream |
| GPFIFO + launch | `runtime/launch.ls` | Ring buffer management |
| Doorbell | `runtime/doorbell.ls` | MMIO notify |
| Channel setup | `runtime/init.ls`, `kernel/lithos_channel.c` | Hardware channel allocation |
| Synchronization | `runtime/sync.ls` | Completion detection |
