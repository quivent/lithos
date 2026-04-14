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

## Weights-as-Code — current state

Weights-as-code (compile-time dequantize → emit FMUL-IMM instruction stream with FP32 weights as 32-bit immediates) is **not yet active** in the minimal compiler. The lexer reserves the vocabulary and the GPU emit primitive exists, but the parser has no semantic handlers and no live source uses the surface syntax.

**In place:**

| Layer | Evidence | Where |
|---|---|---|
| Lexer tokens | `TOK_WEIGHT=25, TOK_LAYER=26, TOK_BIND=27, TOK_TEMPLATE=29, TOK_PROJECT=30` | `bootstrap/lithos-lexer.s:63-68` + keyword table `173-215` |
| GPU emit primitive | `emit_fmul_imm rd ra imm32 : iword OP_FMUL_IMM \| (rd<<16) \| (ra<<24) \| (imm32<<32)` with `OP_FMUL_IMM = 0x7820` | `compiler/emit-gpu.ls:112,118,406-407` |
| Compile-time tensor reader (exists, wiring unknown) | — | `compiler/lithos-safetensors.ls` |
| Multi-target driver | `lithos kernel.ls --target gpu -o out.cubin` | `bootstrap/driver.s:26,36,649` |
| Encoding breadcrumb from older parsers | `ORRIMM w5, 0x78206800, w16` — direct host-side encoding of OP_FMUL_IMM | `bootstrap/lithos-parser.s:724`, `-orig.s:720`, `-regfix.s:720`, `lithos-table.s:715` |

**Missing:**

| Gap | What it needs | Rough shape |
|---|---|---|
| Keyword handlers | `lithos-table.s:2855-2869` dispatches all five weights-as-code tokens to `.La_ident` (silent fallback) | Real semantic handlers per keyword |
| Surface-syntax usage | Zero uses of `weight <name> : <shape> <quant>`, `bind L ...`, `project W x -> y`, `template ...`, `layer L : 0..N` anywhere in `compiler.ls`, `inference/`, or `runtime/` | Depend on handlers + syntax freeze |
| Template stamping | Parser has no scope that iterates `layer L : 0..47` and re-parses the template body with L bound | Macro-expansion pre-pass, or capture-and-replay handler |
| `project` expansion | No pipeline from keyword to compile-time weight read → dequant → N×FMUL-IMM emission | Connect `lithos-safetensors.ls` + `emit_fmul_imm` through a new parser handler |
| Dequant at compile time | No compile-time nibble-extract / zero-point / scale chain | Can live in `compiler.ls` once bootstrap parses enough |
| `{L}` interpolation | No per-layer-index substitution in bind-path strings | String-interpolation primitive for the parser |

**Strategy (not a plan — a reading):** the path to weights-as-code is most likely through `compiler.ls` handling the semantics itself, once the bootstrap parses enough of it. The bootstrap's job is to pass `weight/bind/project/template/layer` constructs through faithfully — today it silently treats them as identifiers, which is wrong-but-not-erroring. Before the real handlers land, the bootstrap should probably error on these keywords used in unexpected positions.

**Old parser variants preserved as hint banks.** `bootstrap/lithos-parser{.s, -orig.s, -regfix.s, -v2.s, -v3.s, -v4.s, -v5.s}` contain FMUL-IMM encoding attempts and other experiments. Do not delete. A separate extraction pass is mining them for specific pieces.

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
