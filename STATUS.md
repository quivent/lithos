# Lithos — Status

**Last updated:** 2026-04-14

---

## Milestone: compiler.ls compiles cleanly

The bootstrap compiler (`bootstrap/lithos-table.s`) compiles `compiler/compiler.ls` (5044 lines) to a 47KB ARM64 ELF binary. Clean compile — no hacks, no skipped statements. Both platforms (Linux and macOS) use the same parser.

`compiler/safetensors.ls` (423 lines) split out as a separate compilation unit. It reads weight file headers at inference time and will be compiled independently by the stage-1 compiler.

```
compiler.ls    5044 lines → 47,264 bytes ARM64 ELF (lithos-stage1)
safetensors.ls  423 lines → separate compilation unit
```

---

## Progress

### Shared (both platforms)
```
COMPILER SOURCE [████████] compiler.ls rewritten to pure Lithos syntax
BOOTSTRAP       [████████] lithos-table.s compiles full compiler.ls cleanly
SAFETENSORS     [████████] split to compiler/safetensors.ls (separate unit)
```

### Linux / GH200
```
GPU BACKEND     [████████] emit-gpu.ls: 35+ raw SM90 opcodes, ctrl words, grid-sync
RUNTIME         [████████] all .ls written, register_count found (SPD 0x094)
HARDWARE        [██████░░] GSP+FSP wired, cbuf0 probe DONE, QMD builder untested
SELF-HOST       [░░░░░░░░] lithos-stage1 exists but not yet tested on GH200
FIRST KERNEL    [░░░░░░░░] blocked on self-host verification
FIRST TOKEN     [░░░░░░░░] blocked on kernel execution
```

### macOS / Apple Silicon
```
BOOTSTRAP       [████████] build-darwin.sh: 11 transforms, compiles full compiler.ls
COMPILE+RUN     [██████░░] compile-darwin.sh wraps ELF→Mach-O for simple programs
GPU RESEARCH    [██░░░░░░] Metal confirmed as floor — Apple blocks raw ISA
METAL BACKEND   [░░░░░░░░] not started
```

---

## Next steps

### On macOS (this machine)
1. **Verify lithos-stage1 correctness** — compile simple .ls programs through the stage-1 compiler (via compile-darwin.sh wrapper) and verify output
2. **Self-host test** — lithos-stage1 compiles compiler.ls → lithos-stage2; compare binaries
3. **Compile safetensors.ls** — lithos-stage1 compiles safetensors.ls as separate unit
4. **Metal GPU research** — when ready to target Apple Silicon GPU

### On GH200 (Linux ARM64)
1. **Build bootstrap on Linux** — `cd bootstrap && ./build.sh` (uses same lithos-table.s)
2. **Compile compiler.ls** — bootstrap produces lithos-stage1 (ARM64 ELF, runs natively)
3. **Test lithos-stage1** — run it on a simple .ls file, verify output
4. **Self-host** — lithos-stage1 compiles compiler.ls → lithos-stage2
5. **Fixed point** — lithos-stage2 compiles compiler.ls → lithos-verify; `diff lithos-stage2 lithos-verify` must match
6. **Compile safetensors.ls** — separate unit, link with launcher
7. **Compile inference kernels** — inference/*.ls → SM90 cubins
8. **First kernel execution** — GSP+GPFIFO+QMD runs one compiled kernel on Hopper
9. **First token** — one forward pass through Qwen 3.5 27B
10. **Full inference** — autoregressive generation, EOS detection, streaming output

---

## Architecture

**Two platforms. One language. Zero shared GPU paths.**

| | Linux / GH200 | macOS / Apple Silicon |
|---|---|---|
| Host ISA | ARM64 | ARM64 |
| Host ABI | Linux syscalls (X8, SVC #0) | Darwin syscalls (X16, SVC #0x80) |
| Binary format | ELF64 | Mach-O |
| GPU | Hopper SM90a | Apple GPU (M-series) |
| GPU backend | Raw SM90 binary (probe-verified) | Metal (floor) |
| GPU access | vfio-pci + MMIO (no libcuda) | Metal API |

**Compiler architecture:** single-pass recursive descent, direct emission. No AST, no IR. Compositions are parsed and emitted immediately. Two backends: ARM64 (host) and SM90 (GPU, raw binary).

**Inference target:** Qwen 3.5 27B (64 layers: 48 DeltaNet + 16 full-attention, GPTQ W4A16). Two cooperative megakernels — runtime grid-sync loop, not compile-time unrolling.

---

## File map

```
compiler/
  compiler.ls       5044 lines  the self-hosting compiler
  safetensors.ls     423 lines  weight file reader (separate unit)
  emit-gpu.ls                   SM90 raw binary emitter
  lithos-parser.ls               parser source (compiled by stage-1)
  lithos-lexer.ls                lexer source
  lithos-elf.ls                  ELF writer source
  config-reader.ls               model config JSON reader

bootstrap/
  lithos-table.s    3502 lines  bootstrap parser (ARM64 assembly)
  build.sh                      Linux build script
  build-darwin.sh                macOS build script (11 transforms)
  compile-darwin.sh              ELF→Mach-O wrapper

inference/
  attend.ls, attend_full.ls, embed.ls, gemv.ls, reduce.ls, ...
  10 kernel source files covering full Qwen 3.5 forward pass

runtime/
  init.ls, mem.ls, qmd.ls, launch.ls, sync.ls, ...
  11 .ls files replacing all 42 libcuda call sites
```
