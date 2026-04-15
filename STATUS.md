# Lithos — Status

**Last updated:** 2026-04-15

---

## Milestone: Lithos emitter API produces opcodes that execute on GPU

**`v0.2-emitter-api`** — Opcodes emitted through the compiler's emitter functions (`emit_s2r`, `emit_i2f`, `emit_ldc`, `emit_imad_imm`, `emit_stg`, `emit_exit`) executed correctly on a GH200 480GB GPU. 255/256 threads wrote `float(threadIdx.x)` to global memory. Every instruction byte matches the NVIDIA reference encoding byte-exact.

**Previous:** `v0.1-first-opcode` — Hand-copied reference opcode bytes executed on silicon, proving the cubin/execution path works.

**New:** `v0.2-emitter-api` — The compiler's emitter **functions** produce correct opcodes from Python API calls. This is the last verification needed before full compilation from .ls source.

```
Python API (emit_s2r, emit_ldc, ...) → 96 bytes → cubin ELF → GH200 SM → correct output ✓
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
EMITTER API     [████████] VERIFIED — emitter functions produce correct opcodes
RUNTIME         [████████] all .ls written, SPD+5-part pushbuffer implemented
OPCODE EXEC     [████████] PROVEN — emitter-emitted kernel executed on GH200
GSP HARDENING   [████████] ~40 crash/brick bugs fixed, emergency reset, watchdog
HARDWARE        [██████░░] QMD/SPD probe-verified, GSP 8/9 gates, FSP step 7 TODO
CODEGEN         [████░░░░] register allocator picks bad registers (R0/R1 issue)
SELF-HOST       [████░░░░] stage-1 runs (no segfault), output generation WIP
FIRST KERNEL    [██████░░] emitter proven, codegen allocator is the blocker
FIRST TOKEN     [░░░░░░░░] blocked on codegen fix + inference pipeline
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

### Immediate next step (has been unblocked)

**Test `elementwise.ls` on GPU.** Two critical bugs were fixed that previously produced illegal instructions:

1. **IADD3/IMAD ctrl word encoding** (emit_sm90.py) — extra41 bits weren't setting predicate sources to PT. Result: `IMAD.U32.X R12, R13, R13, ~R13, !PT` garbage instead of clean `IMAD`. Fixed: `extra41 = 0x0FF1E000 | rs3` (IADD3), `0x078E0200 | rs3` (IMAD).

2. **`temp()` returned duplicate registers** (codegen.py) — used `id(object())` for names, but Python GCs the throwaway and recycles memory addresses. Every temp got the same register, so `IMAD Rd, tid, tid, tid` instead of `IMAD Rd, ctaid, blockdim, tid`. Fixed with monotonic counter.

After fixes, `elementwise.ls` disassembles to semantically correct SASS:
```
S2R R13, SR_TID.X
S2R R14, SR_CTAID.X
IADD3 R15, RZ, 0x100, RZ        ; blockdim = 256
IMAD R12, R14, R15, R13          ; global_idx = ctaid * 256 + tid
IADD3 R14, RZ, 0x4, RZ           ; stride = 4
IMAD R13, R12, R14, RZ           ; byte_offset = idx * 4
IADD3 R15, P0, R9, UR13, RZ      ; addr = input + offset
LDG.E R16, desc[UR0][R15.64]     ; load input[idx]
...
```

**To test on 192.222.50.140:**
```bash
cd ~/lithos && git pull origin exp
cd bootstrap-py && rm -rf __pycache__
python3 main.py ../inference/elementwise.ls -o /tmp/elem.cubin --target gpu
/tmp/test_sass /tmp/elem.cubin
```

If it runs correctly, **lithos has compiled .ls source to executable GPU opcodes end-to-end** — the v0.3 milestone.

### Likely next bugs if elementwise.ls fails

The SASS looks correct but there may still be issues:

- **LDG/STG with UR registers** — the `desc[UR0][R15.64]` syntax means the memory operation uses a uniform register descriptor. We're emitting this, but if the uniform register isn't populated the load fails. May need a ULDC instruction to set up UR0/UR4 first.
- **Unused reg count** — if get_register_count() returns too few, SM may allocate insufficient registers for the kernel.
- **FADD on uninitialized reg** — if we emit `FADD R13, R13, R13` where R13 holds the loaded value, that's fine. But if the load went to a different reg and we FADD the wrong one, garbage out.

### After elementwise works

4. **Compile `gemv.ls`** (186 lines, GPTQ W4A16 GEMV) — real inference kernel. Uses shared memory, warp reductions, dequantization. Will surface gaps in codegen for more complex patterns.

5. **Fix stage-1 output generation** — the assembly-bootstrap-produced stage-1 compiler runs (no segfault, BSS fix worked) but doesn't write its output file. Bug is in compiler.ls's own ELF writer composition.

6. **Self-host** — stage-1 compiles compiler.ls → stage-2. Fixed point: stage-2 compiles compiler.ls → stage-3, and stage-2 == stage-3 byte-exact.

### On GH200 (needs GPU)

7. **Lithos runtime path** — QMD + SPD + pushbuffer + doorbell, bypassing `cuModuleLoadData` and `cuLaunchKernel`. Runtime encoding validated byte-exact against nvcc (tools/test_runtime_encoding.py passes). Actual submission requires either:
   - lithos.ko loaded (means unbinding nvidia, risky on single machine)
   - vfio-pci (generic Linux, no NVIDIA code)
   - Fresh machine we don't mind reconfiguring

8. **Full inference pipeline** — all 10 inference kernels compiled, runtime wired, weights loaded from safetensors.

9. **First token** — forward pass through Qwen 3.5 27B. 64 layers (48 DeltaNet + 16 full-attention), GPTQ W4A16.

10. **Autoregressive generation** — EOS detection, streaming output, performance tuning.

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
