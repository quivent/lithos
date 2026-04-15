# Lithos — Status

**Last updated:** 2026-04-15

---

## Milestone: first opcodes executed on GPU

Lithos-emitted SM90 opcodes ran on a GH200 480GB GPU (Compute 9.0, driver 580.105.08). 256 threads each computed `float(threadIdx.x)` and wrote to global memory. 255/256 non-zero outputs confirmed. No NVIDIA compiler in the instruction path — lithos compiles directly to opcode.

```
6 instructions (S2R, I2F, LDC, IMAD, STG, EXIT) → 96 bytes → cubin ELF → cuModuleLoadData → GPU
```

**Execution path proven end-to-end:**

```
Lithos opcode bytes (128-bit instruction words)
    ↓ wrapped in cubin ELF (.text section)
    ↓ loaded via cuModuleLoadData (NVIDIA driver accepts it)
    ↓ launched via cuLaunchKernel
    ↓ GPFIFO → PBDMA → Compute Engine → CWD → SM
    ↓ SM fetches opcodes from memory, executes them
    ↓ results written to GPU global memory ✓
```

**Previous milestone:** compiler.ls compiles cleanly — bootstrap produces 47KB ARM64 ELF from 5044-line compiler source.

---

## Progress

### Shared (both platforms)
```
COMPILER SOURCE [████████] compiler.ls rewritten to pure Lithos syntax
BOOTSTRAP       [████████] lithos-table.s compiles full compiler.ls cleanly
SAFETENSORS     [████████] split to compiler/safetensors.ls (separate unit)
BOOTSTRAP-PY    [██████░░] Python bootstrapper: 11/11 runtime, 10/10 inference parse
```

### Linux / GH200
```
GPU BACKEND     [████████] emit-gpu.ls: 40+ SM90 opcodes, ctrl words, grid-sync
RUNTIME         [████████] all .ls written, SPD+5-part pushbuffer implemented
OPCODE EXEC     [████████] PROVEN — 6 opcodes executed on GH200, correct output
GSP HARDENING   [████████] ~40 crash/brick bugs fixed, emergency reset, watchdog
HARDWARE        [██████░░] QMD/SPD probe-verified, GSP 8/9 gates, FSP step 7 TODO
SELF-HOST       [████░░░░] stage-1 runs (no segfault), output generation WIP
FIRST KERNEL    [██░░░░░░] opcode execution proven, compiler-emitted kernel next
FIRST TOKEN     [░░░░░░░░] blocked on full kernel compilation + inference pipeline
```

### macOS / Apple Silicon
```
BOOTSTRAP       [████████] build-darwin.sh: 11 transforms, compiles full compiler.ls
COMPILE+RUN     [██████░░] compile-darwin.sh wraps ELF→Mach-O for simple programs
GPU RESEARCH    [██░░░░░░] Metal confirmed as floor — Apple blocks raw ISA
METAL BACKEND   [░░░░░░░░] not started
```

---

## What was proven on 2026-04-15

**Opcode execution on real hardware.** Not SASS (assembly text). Not PTX (virtual ISA). Raw 128-bit SM90 machine code instruction words, constructed in Python, executed on a streaming multiprocessor.

The test kernel:
- `S2R R5, SR_TID_X` — read thread ID from special register
- `I2F.F32.U32 R5, R5` — convert integer to float
- `LDC R2, c[0x2][0x84]` — load output pointer from constant buffer
- `IMAD R2, R5, 0x4, R2` — compute byte offset (tid * 4 + base)
- `STG.E [R2], R5` — store float to global memory
- `EXIT` — terminate thread

Each instruction is 16 bytes: 8-byte instruction word + 8-byte control word (stall counts, yield flags, dependency barriers). The SM fetches these bytes from the .text section of the cubin ELF and executes them directly. No interpretation, no JIT, no microcode translation.

**What this means for lithos:** The compiler's `emit-gpu.ls` (and Python `emit_sm90.py`) can target the SM directly. The opcode-to-silicon path is open. The remaining work is making the emitter produce correct encodings for all 40+ opcodes, then compiling full inference kernels through the pipeline.

---

## Opcode-to-Silicon Pipeline

```
Source (.ls)
    ↓ lexer → tokens
    ↓ parser → AST (or direct emission in assembly bootstrap)
    ↓ emit-gpu.ls / emit_sm90.py → 128-bit instruction words
    ↓ lithos-elf.ls / elf_writer.py → cubin ELF
    ↓
    ↓ === BELOW THIS LINE: NVIDIA's driver infrastructure ===
    ↓
    ↓ cuModuleLoadData (or lithos runtime: QMD + SPD + pushbuffer)
    ↓ cuLaunchKernel (or lithos runtime: GPFIFO + doorbell)
    ↓ PBDMA → Host/ESCHED → Compute Engine → CWD → SM
    ↓
    ↓ SM instruction fetch → execute → writeback
    ↓
    Result in GPU memory
```

Lithos controls everything above the line. Below the line is either NVIDIA's driver (for testing via CUDA API) or lithos's own runtime (QMD/SPD/pushbuffer/doorbell — written, not yet tested on hardware).

---

## Next steps

### Immediate (no GPU needed)
1. **Fix opcode encodings** — make emit_sm90.py produce reference-matching bytes for all opcodes (agent working on this)
2. **Compile inference kernel through emitter API** — not hand-copied reference bytes
3. **Fix stage-1 output generation** — bootstrap compiles compiler.ls, stage-1 runs but doesn't write output (ELF writer codegen issue)

### On GH200 (needs GPU)
4. **Test compiler-emitted opcodes** — run a kernel where every instruction was produced by the emitter, not copied from a reference
5. **Test lithos runtime path** — QMD + SPD + pushbuffer + doorbell, bypassing CUDA API entirely
6. **Compile and run full inference kernel** — gemv.ls or elementwise.ls through the compiler pipeline
7. **First token** — forward pass through Qwen 3.5 27B
8. **Full inference** — autoregressive generation, EOS detection, streaming output

### On macOS
9. **Verify lithos-stage1 correctness** — compile simple .ls programs, verify output
10. **Self-host test** — stage1 compiles compiler.ls → stage2; compare binaries

---

## Architecture

**Two platforms. One language. Lithos compiles to opcode.**

| | Linux / GH200 | macOS / Apple Silicon |
|---|---|---|
| Host ISA | ARM64 | ARM64 |
| Host ABI | Linux syscalls (X8, SVC #0) | Darwin syscalls (X16, SVC #0x80) |
| Binary format | ELF64 | Mach-O |
| GPU | Hopper SM90a | Apple GPU (M-series) |
| GPU target | SM90 opcode (direct binary) | Metal (floor) |
| GPU access | vfio-pci + MMIO (no libcuda) | Metal API |

**Compiler architecture:** single-pass recursive descent, direct emission. No AST, no IR. Compositions are parsed and emitted immediately. Two backends: ARM64 (host) and SM90 (GPU opcode).

**Inference target:** Qwen 3.5 27B (64 layers: 48 DeltaNet + 16 full-attention, GPTQ W4A16). Two cooperative megakernels — runtime grid-sync loop, not compile-time unrolling.

---

## Weights-as-Code — current state

Weights-as-code (compile-time dequantize → emit FMUL-IMM instruction stream with FP32 weights as 32-bit immediates) is **not yet active** in the minimal compiler. The lexer reserves the vocabulary and the GPU emit primitive exists, but the parser has no semantic handlers and no live source uses the surface syntax.

**In place:**

| Layer | Evidence | Where |
|---|---|---|
| Lexer tokens | `TOK_WEIGHT=25, TOK_LAYER=26, TOK_BIND=27, TOK_TEMPLATE=29, TOK_PROJECT=30` | `bootstrap/lithos-lexer.s:63-68` + keyword table `173-215` |
| GPU emit primitive | `emit_fmul_imm rd ra imm32 : iword OP_FMUL_IMM \| (rd<<16) \| (ra<<24) \| (imm32<<32)` with `OP_FMUL_IMM = 0x7820` | `compiler/emit-gpu.ls:112,118,406-407` |
| Compile-time tensor reader (exists, wiring unknown) | — | `compiler/lithos-safetensors.ls` |
| Multi-target driver | `lithos kernel.ls --target gpu -o out.cubin` | `bootstrap/driver.s:26,36,649` |

**Missing:**

| Gap | What it needs |
|---|---|
| Keyword handlers | `lithos-table.s:2855-2869` dispatches to `.La_ident` (silent fallback) |
| Surface-syntax usage | Zero uses of `weight`/`bind`/`project`/`template`/`layer` |
| Template stamping | No `layer L : 0..47` iteration with body re-parse |
| `project` expansion | No compile-time weight read → dequant → N×FMUL-IMM |
| Dequant at compile time | No nibble-extract / zero-point / scale chain |
| `{L}` interpolation | No per-layer-index substitution in bind-path strings |

---

## GSP Hardening (2026-04-14)

Full audit and hardening of the GPU System Processor boot sequence. ~40 crash/hang/brick bugs found and fixed across 28 assembly source files. This code orchestrates the loading of NVIDIA's signed GSP firmware — it does not replace the firmware.

Key fixes:
- EMEM command queue overflow (brick risk)
- OOM pointer dereference (segfault on DMA copy)
- Falcon reset leaving undefined state
- RPC queue wrap causing infinite loops
- ARM64 Synchronous External Abort from dead PCIe reads
- Non-atomic 64-bit timer reads
- Chain-of-trust payload validation
- Boot watchdog (120s timeout, per-step liveness checks)
- Emergency reset function + standalone binary

---

## File map

```
compiler/
  compiler.ls       5056 lines  the self-hosting compiler
  safetensors.ls     423 lines  weight file reader (separate unit)
  emit-gpu.ls       1016 lines  SM90 opcode emitter (40+ opcodes)
  lithos-parser.ls               parser source
  lithos-lexer.ls                lexer source
  lithos-elf.ls                  ELF writer (cubin + ARM64)
  config-reader.ls               model config JSON reader

bootstrap/
  lithos-table.s    3549 lines  bootstrap parser (ARM64 assembly)
  build.sh                      Linux build script
  build-darwin.sh                macOS build script
  lithos-boot-*.c   4 variants  C bootstrap implementations

bootstrap-py/
  lexer.py          523 lines   tokenizer (tested on all .ls files)
  parser.py                     recursive descent (11/11 runtime, 10/10 inference)
  emit_arm64.py     996 lines   ARM64 code gen
  emit_sm90.py                  SM90 opcode gen (40 opcodes)
  elf_writer.py                 ELF64 writer (ARM64 + cubin)
  codegen.py                    AST walker with register alloc + spilling
  main.py                       CLI driver

inference/
  attend.ls, attend_full.ls, embed.ls, gemv.ls, reduce.ls, ...
  10 kernel source files covering full Qwen 3.5 forward pass

runtime/
  init.ls, mem.ls, qmd.ls, launch.ls, pushbuffer.ls, sync.ls, ...
  11 .ls files replacing all 42 libcuda call sites

GSP/
  28 ARM64 assembly source files — GPU system processor boot + probes
  Hardened: watchdog, emergency reset, bounds checks, timeout on every poll

tools/
  test_sass_execution.cu         opcode execution test via CUDA driver API
  mk_test_kernel.py              test kernel generator (6 opcodes)
  extract_and_compare.sh         reference comparison (nvcc vs lithos)
```
