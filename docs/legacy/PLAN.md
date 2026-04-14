> **DEPRECATED** — superseded by STATUS.md

# Lithos Execution Plan

Two platforms. One language. Zero shared GPU paths.

All sources use `.ls` extension. Zero Forth. Zero Python. Zero CUDA runtime.

---

## PLATFORMS

| | **Linux / GH200** | **macOS / Apple Silicon** |
|---|---|---|
| **Host ISA** | ARM64 | ARM64 |
| **Host ABI** | Linux syscalls (X8, SVC #0) | Darwin syscalls (X16, SVC #0x80) |
| **Binary format** | ELF64 | Mach-O |
| **GPU** | Hopper SM90a | Apple GPU (M-series) |
| **GPU backend** | Raw SM90 binary (probe-verified) | TBD — Metal or lower (research needed) |
| **GPU access** | vfio-pci + MMIO (no libcuda) | IOKit / Metal / TBD |
| **Target workload** | Qwen 3.5 27B inference | TBD |

The compiler, lexer, parser, and ARM64 instruction emitter are **platform-common**.
Syscalls, binary writers, GPU backends, and device runtimes **diverge per platform**.

### Terminology

- **SM90 binary** — raw 128-bit Hopper instructions emitted directly. Not SASS assembly, not PTX. No ptxas in the loop. Opcodes empirically verified on GH200 via probe disassembly.
- **Apple GPU ISA** — partially reverse-engineered (Asahi Linux project). Apple restricts access below Metal. How far Lithos can go is an open research question.

---

## TOP-LEVEL PROGRESS

### Shared (both platforms)
```
FOUNDATIONS   [████████] complete   language, kernels, grammar, dicts, docs
COMPILER      [████████] DONE — rewritten to pure Lithos syntax (zero =, fn, ->, load_u, syscall)
BOOTSTRAP     [██████░░] links, compositions compile, compiler.ls parses to line 247
```

### Linux / GH200
```
GPU BACKEND   [████████] emit-gpu.ls: 35+ SM90 opcodes, ctrl words, grid-sync
RUNTIME       [████████] all .ls written, register_count found (SPD 0x094)
HARDWARE      [██████░░] GSP+FSP wired, cbuf0 probe DONE, QMD builder untested
INTEGRATION   [██░░░░░░] launcher.ls written (162 lines), not compiled
FIRST TOKEN   [░░░░░░░░] blocked on bootstrap + integration
```

### macOS / Apple Silicon
```
PLATFORM SPLIT [██████░░] bootstrap builds+runs, output still ELF (Mach-O writer next)
GPU RESEARCH   [██░░░░░░] Metal confirmed as floor — Apple blocks raw ISA on macOS
GPU BACKEND    [░░░░░░░░] no Metal emitter exists
METAL RUNTIME  [░░░░░░░░] Metal compute path not started
```

---

## BOOTSTRAP (pure ARM64 assembly, `bootstrap/*.s`)

### Platform-common (lexer, parser, expr compiler, ARM64 emitter)
```
[✓] Lithos bootstrap binary builds, 35KB .bss (not 528MB)
[✓] Entry, memory, DTC runtime wired (_start in driver.s, not lithos-bootstrap.s)
[✓] 7 stack-alignment SIGBUS bugs in lithos-elf-writer.s fixed
[✓] Smoke test: `var x 42` → valid ARM64 ELF produced
[✓] Build: bootstrap/build.sh, Makefile, macros prelude working
[✓] Expression parsing works: `var y x + 1` produces valid ELF
[ ] Composition parsing: `name args :` → parse error (code at line 1196, crashes) ← BLOCKING
[ ] Binding syntax: `name expr` (no = sign) — bootstrap treats first token as name ← BLOCKING
[ ] Audit: compiler.ls parses cleanly through bootstrap                            ← BLOCKING
```

### Platform split needed

The bootstrap has two platform-specific surfaces:

**1. Syscall layer** — DONE via build-time transform
```
[✓] build-darwin.sh applies 9 mechanical transforms to Linux sources:
    syscall register (X8→X16), trap (svc#0→svc#0x80), syscall numbers,
    ADRP/lo12 dialect, open flags, MAP_ANONYMOUS, AT_FDCWD,
    argc/argv convention (stack→registers), uxtw register class,
    .equ forward references, text relocation (.quad to .data)
[✓] lithos-bootstrap-darwin: 153KB Mach-O, compiles .ls files
[✓] Source files unchanged — transforms at build time only
```

**2. Binary output format** (lithos-elf-writer.s, ~2026 lines)
```
[✓] ELF writer works on both platforms (for Linux targets + GPU cubins)
[ ] Write macho-writer.s for macOS host executables
    Mach-O header, LC_SEGMENT_64, LC_MAIN, __TEXT/__DATA sections
    Base address: 0x100000000 (not 0x400000)
[ ] --target flag: arm64-linux (ELF), arm64-darwin (Mach-O)
```

Everything else in the bootstrap — lexer, parser, expression compiler, ARM64
instruction emitter — is platform-common. ARM64 is ARM64. The instructions
don't change, only the syscall ABI and the binary wrapper around them.

The language has NO `=` assignment. Bindings: first token = name, rest = expression.
Compositions: `name args :` followed by indented body.
compiler.ls is being rewritten to pure Lithos syntax (worker dispatched).

---

## COMPILER (`compiler/compiler.ls`, 4739 lines)

### Platform-common sections
```
[✓] Lexer (Section 3): UTF-8 multi-byte for → ← ↑ ↓ Σ △ ▽ √ ≅ ≡
[✓] ARM64 backend (Section 1): 90+ emitters, branch patching (ARM64 is ARM64 on both platforms)
[✓] Parser (Section 4): recursive descent, dual backend routing
[✓] Safetensors reader (Section 5): JSON header + tensor index
[✓] Main entry (Section 7): argv → mmap → lex → parse → emit → write
```

### Linux / GH200 — GPU backend
```
[✓] SM90 backend (Section 2, emit-gpu.ls): 35+ raw Hopper opcodes, ctrl words, grid-sync
    128-bit instructions (8-byte inst + 8-byte ctrl), probe-verified on sm_90a.
    No ptxas, no PTX, no SASS assembler — direct binary emission.
[✓] ELF writer (Section 6): 9-section cubin ELF64 structure (for GPU binary)
[✓] config.json reader — compiler/config-reader.ls (873 lines), untested
    Parses all 17 Qwen 3.5 fields, classifies layer_types[], provides accessors
[ ] No per-layer dispatch loop (hybrid-layers.md design unimplemented)
    NOTE: design is single megakernel with compile-time unrolling, not runtime dispatch
[ ] No megakernel linker
[ ] cubin_buf is 512KB; 64-layer megakernel needs ~300MB (600× too small)
```

### macOS / Apple Silicon — GPU backend (not started)
```
Metal is the floor. Apple blocks raw GPU ISA access on macOS despite
Asahi reverse engineering of the instruction set. No vfio-pci, no IOKit
bypass, no raw command buffer submission. Metal API is the only path.

[ ] Metal compute shader emitter — emit MSL source from .ls kernel IR
    Lithos compiler generates Metal Shading Language, compiled at runtime
    via MTLDevice newLibraryWithSource or pre-compiled via metal/metallib
[ ] Metal runtime bridge — call Metal C API from ARM64 Lithos host code
    MTLCreateSystemDefaultDevice → MTLDevice → MTLCommandQueue →
    MTLComputePipelineState → MTLComputeCommandEncoder → dispatch
    (metal-cpp provides C function interface, no Objective-C needed)
[ ] Map Lithos kernel concepts to Metal compute model:
    threadgroup = NVIDIA block, threads_per_threadgroup = blockDim,
    threadgroup_position_in_grid = blockIdx, thread_position_in_threadgroup = threadIdx
    device memory = global, threadgroup memory = shared
[ ] Determine target workload for macOS GPU
```

### Platform-specific host I/O (compiler.ls lines 5244-5390)
```
[ ] mmap_file (line 5334): hardcoded Linux syscall numbers (openat=56, lseek=62, mmap=222)
    Needs Darwin variant (different numbers, X16 not X8, SVC #0x80)
[ ] write_file (line 5368): hardcoded Linux syscall numbers (openat=56, write=64)
    Same platform split needed
[ ] elf_build_arm64 (line 5244): emits ELF64 at 0x400000
    Needs parallel macho_build_arm64 for macOS host executables
    (GPU cubins remain ELF regardless of host platform)
```

### Shared open items
```
[ ] `$` register prefix not a lexer token
[ ] arch/hopper.dict and arch/arm64.dict never read
```

---

## INFERENCE KERNELS (`inference/*.ls`, new grammar)

### Linux / GH200 — SM90 kernels (written, not compiled)
```
[✓] attend.ls — DeltaNet attention, RoPE (theta 10M, partial rotary 0.25)
[✓] attend_full.ls — full attention kernel for hybrid layers (GQA 24/4, output gate)
[✓] decay_gate.ls — softplus + exp decay
[✓] delta_update.ls — delta rule state update
[✓] deltanet_fused.ls — fused kernel with beta param
[✓] elementwise.ls — residual_add, elemwise_mul, activate_silu, scale
[✓] embed.ls — token embedding (FP16 and FP32 variants)
[✓] gemv.ls — W4A16 GEMV, GPTQ dequant chain CORRECT (scale + zero-point fixed)
[✓] recur.ls — conv1d, gate_sigmoid, deltanet_step, state_rollback
[✓] reduce.ls — rmsnorm, rmsnorm_residual, l2norm, sample_argmax

Stale gaps (resolved by scouts 2026-04-13):
[✓] Conv1D 4-tap — fully implemented in recur.ls (lines 12-40), not a stub
[✓] Computed-offset indexing — .ls files use x[var] pattern, works today

Still open from minds-revisited:
[ ] Apply v7 software pipelining to gemv.ls
[ ] Apply v2 v4.u32 K-wise weight loads
[ ] Per-projection shape selection (compile-time specialization per layer)
[ ] Transposed [N, K/8] weight layout (preprocess step)
```

### macOS / Apple Silicon — Metal kernels (not started)
```
Metal is the floor. The compiler's macOS GPU backend emits MSL (Metal Shading
Language) instead of raw SM90 binary. The .ls kernel sources describe algorithms;
the compiler translates to MSL compute kernels.

[ ] MSL emitter — translate .ls kernel compositions to Metal Shading Language
    kernel void name(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]])
    Metal uses SIMD-groups (like warps), threadgroups (like blocks), grids
[ ] Map Lithos memory model to Metal:
    → (load) / ← (store) → device memory reads/writes
    shared → threadgroup memory
    barrier → threadgroup_barrier(mem_flags::mem_threadgroup)
[ ] Metal has no inline assembly — all GPU logic must go through MSL
    This is the fundamental constraint. Lithos cannot "write on silicon"
    for Apple GPU. It writes on Metal, which writes on silicon.
```

---

## RUNTIME — Linux / GH200 (`runtime/*.ls` — the libcuda replacement)

Replaces all 42 libcuda call sites in `src/launcher.s`.
**Linux-only.** Direct GPU register access via vfio-pci. No macOS equivalent.

```
[✓] init.ls — vfio-pci + BAR0/BAR4 mmap, PMC sanity, GSP boot dispatch
[✓] mem.ls — BAR4 bump allocator (coherent HBM, no dma_alloc_coherent)
[✓] elf_load.ls — parse compiled GPU ELF, extract entry_pc via .symtab walk
[✓] qmd.ls — 528-byte descriptor with empirically-probed offsets
[✓] cbuf0.ls — constant buffer
[✓] register_count: NOT in QMD or cbuf0. In 384-byte Shader Program Descriptor (SPD) at 0x094
[ ] pushbuffer.ls needs update: launch is 5 inline loads (QMD + fence + context + SPD + patch), not just QMD
[✓] pushbuffer.ls — 6-method GPFIFO submission sequence
[✓] doorbell.ls — USERD+0x8c GPPut write + USERD+0x90 doorbell
[✓] launch.ls — kernel dispatch, ties QMD+cbuf0+pushbuffer+doorbell
[✓] sync.ls — completion polling (flag-based + GPGet)
[✓] teardown.ls — munmap + close
[✓] dispatch.ls — top-level dispatch_kernel composition

None of these have been executed yet — the compiler must work first.
```

## RUNTIME — macOS / Apple Silicon (not started)

```
Metal is the floor. Apple GPU runtime = Metal API.

[ ] Metal runtime in .ls (runtime-metal/*.ls)
    init_metal.ls   — MTLCreateSystemDefaultDevice, create command queue
    mem_metal.ls    — MTLBuffer allocation (MTLResourceStorageModeShared for unified memory)
    launch_metal.ls — create compute pipeline, encode dispatch, commit
    sync_metal.ls   — waitUntilCompleted or addCompletedHandler
    teardown_metal.ls — release device, queue, buffers

    All Metal calls go through dylib function pointers loaded at init.
    Metal's C API (via metal-cpp or raw dlsym from Metal.framework):
      MTLCreateSystemDefaultDevice()
      [device newCommandQueue]  → objc_msgSend with selector
      [device newBufferWithLength:options:]
      etc.

    Key difference from Linux: Metal manages its own command buffers.
    No GPFIFO, no QMD, no doorbell, no pushbuffer. The runtime is
    simpler — submit work via API, wait for completion via API.

[ ] Objective-C bridge or raw objc_msgSend
    Metal has no pure C function API — it's Objective-C under the hood.
    Options: (a) thin .m shim compiled separately, (b) raw objc_msgSend
    from ARM64 assembly (sel_registerName + objc_msgSend — Lithos style).
    Option (b) is ~20 extra assembly wrappers.
```

---

## HARDWARE PATH — Linux / GH200 (`GSP/*.s` — ARM64 assembly)

**Linux-only.** NVIDIA GSP/Falcon/GPFIFO — no macOS equivalent.

```
[✓] bar_map.s — sysfs resource0/resource4 mmap (BDF hard-coded)
[✓] pmc_check.s — PMC_BOOT_0 read, PMC_INTR_0 clear
[✓] falcon_reset.s — Falcon reset/deassert sequence
[✓] hbm_alloc.s — bump allocator from BAR4
[✓] fw_load.s — ELF64 parse, PT_LOAD copy to HBM
[✓] bcr_start.s — RISC-V BCR register programming
[✓] poll_lockdown.s — PRIV_LOCKDOWN release polling
[✓] rpc_channel.s — channel allocation via GSP RPC

FSP subsystem wired:
[✓] FSP wired into GSP/boot.s at step 6 (before BCR at step 7)
[✓] BCR/FSP ordering fixed per NVIDIA reference
[✓] fsp_queue_advance_head added to emem_xfer.s
[✓] Stale .include directives removed from mctp/response/diag
[ ] vfio-pci: not implemented (sysfs mmap only)
[✓] register_count: in SPD at offset 0x094 (bits 23:16). 3-point verified.
[✓] Launch is 5 inline loads: QMD(528B) + fence(8B) + context(384B) + SPD(384B) + patch(4B)
```

---

## LAUNCHER INTEGRATION

```
[✓] src/launcher.ls — 162 lines, replaces src/launcher.s
[ ] Removes all 42 libcuda call sites
[ ] Single dispatch per token (megakernel, not per-kernel)
[ ] Tokenize + emit stubs (full tokenizer later)
```

---

## DOCUMENTATION

```
[✓] docs/language-primitives.md — grammar spec
[✓] docs/model-config.md — Qwen 3.5 27B (num_layers 64, vocab 248320)
[✓] docs/hybrid-layers.md — 3+1 schedule, dims CORRECTED (24/4 heads, MLP 17408)
[✓] docs/gpu-register-model.md — $U, @P, $0:1, @smem, regcap
[✓] docs/qmd_fields.md — empirically probed offsets
[✓] docs/arm64-encodings.md — every instruction Lithos uses
[✓] docs/launcher-audit.md — 42 libcuda sites mapped
[✓] docs/minds-revisited.md — 12 GPTQ variants ranked under weights-as-code
[✓] docs/bootstrap-compile.md + v2 — bootstrap diagnosis
[✓] docs/weights-as-code-{compiler,language,inference}.md — corrected analyses
[✓] docs/gsp-native.md — eliminate C via vfio-pci
[✓] docs/self-hosting-compiler.md — architecture
[✓] docs/regalloc_design.md
[✓] HTML versions live on Vercel: /language, /pipeline, /roadmap, architecture,
   compiler, kernels, quantization, performance, bandwidth, glossary (26 pages, all 200)
[✓] /status dashboard live (hand-crafted HTML, progress cards, blocker banner)
[✓] bin/build-status generates from PLAN.md, local server on port 8080
```

---

## DEAD CODE REMOVED

```
[✓] 21 Forth files deleted (compiler/*.fs, gpu/*.fs, core.fs, patterns.fs, etc.)
[✓] 34 Python files deleted (src/, bench/, compiler/, kernels/, quantization/, tools/)
[✓] 21 orphan cubins deleted
[✓] 36 sass/ probe files deleted
[✓] Fake inference claims deleted (index.html "Paris" logit 18.057, cosine 1.0)
[ ] kernel/ (C kernel module) — keep until GSP boot fully runs, then delete
[ ] src/launcher.s (4096 lines) — delete when src/launcher.ls compiled + running
```

---

## MISSING LINKS (found by 12-scout gap hunt, 2026-04-13)

These are gaps NOT covered elsewhere in this plan. Each blocks full inference.
**All Linux / GH200 specific** — macOS inference path does not exist yet.

### M1. Tokenizer — **BLOCKING, no implementation exists**

The inference loop cannot start without text → token ID conversion.
Qwen 3.5 uses BPE with 248K vocab + ~130K merges. Files on disk:
- `/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16/vocab.json` (6.7MB)
- `/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16/tokenizer.json` (12.8MB)

Zero tokenizer code exists in Lithos. Options:
1. Implement BPE in `.ls` (hard: needs JSON parsing of 12MB merge table + regex pre-split)
2. Implement BPE in ARM64 assembly in the bootstrap layer
3. Pre-tokenize externally and feed raw token IDs (simplest, creates external dependency)

Decision needed. For first-token milestone, option 3 (hardcoded token IDs) is sufficient.

### M2. Autoregressive generation loop — **BLOCKING**

`sample_argmax` in reduce.ls is fully implemented (not a stub). But:
- No generation loop exists — system produces exactly one token per forward pass
- No EOS token detection (no EOS ID hardcoded or parameterized anywhere)
- No max-length parameter
- No streaming output (token written to device memory, no host readback/print)
- No temperature scaling, top-p, top-k, repetition penalty

For first-token milestone: hardcoded single forward pass is sufficient.
For usable inference: needs a loop in launcher.ls that calls forward pass,
reads argmax result, checks EOS, feeds token back as next input.

### M3. KV cache + DeltaNet state lifecycle — **BLOCKING**

Kernels accept `k_cache`/`v_cache`/`dn_state` pointers but nothing allocates
or manages them. Missing layer between `mem.ls` (raw bump) and kernels:

```
[ ] inference_init.ls — allocate KV cache (16 attn layers × 4 KV heads × 256 dim × max_seq)
[ ] inference_init.ls — allocate DeltaNet state (48 layers × 16 heads × 128 × 128 × f32 = ~48MB)
[ ] inference_init.ls — zero-initialize DeltaNet state at start of sequence
[ ] Per-token KV append — write new K/V at position seq_len, increment counter
[ ] Pre-allocate max sequence length or implement grow-on-demand
```

Memory ceiling: ~131KB/token for KV across 16 attention layers. At 128GB BAR4
minus weights (~14GB) and activations, theoretical max ~500K-800K tokens.

### M4. FP16 compute opcodes — **PERFORMANCE GAP**

`emit-gpu.ls` has zero FP16 opcodes. Only FP32: FADD, FMUL, FFMA, FMNMX.
No HADD, HMUL, HFMA, HMNMX, HSETP, or packed FP16 operations.

The current correct inference (cosine 1.0 at all 64 layers) used FP32 throughout.
For matching vLLM throughput, FP16 compute is essential — 2× ALU throughput
on Hopper, 2× memory bandwidth efficiency for activation traffic.

Not blocking for correctness. Blocking for closing the performance gap.

```
[ ] Add emit_hadd, emit_hmul, emit_hfma (FP16 opcodes) to emit-gpu.ls
[ ] Add emit_hfma2 (packed 2×FP16) for bandwidth-bound kernels
[ ] Probe and verify FP16 instruction encodings via ptxas/nvdisasm
[ ] Add f32>f16 / f16>f32 conversion opcodes
```

### M5. Error handling — **CRITICAL for development velocity**

Zero error handling exists. Silent failures will waste debugging time.

**Error model:** Lithos has no error handling as a language feature, and this
is correct. Compiled Forth has none either — `ABORT` resets stacks and returns
to the interpreter, `CATCH`/`THROW` are interpreter-level and don't survive
compilation to threaded code. Lithos compiles to bare machine code (ARM64 and
SM90 binary). Machine code doesn't have error handling — it has branches.

The error model is a **convention, not a language feature:**
- Compositions that can fail return a status (register or stack value).
- Caller checks via `if< result 0` or equivalent branch.
- No stack unwinding, no exception frames, no `try/catch`.
- Like C's `errno` or Go's `if err != nil` — the language doesn't enforce it,
  the programmer follows it.

**Three error surfaces:**

1. **ARM64 host (syscalls):** `openat`, `mmap`, `ioctl`, `fstat`, `write` return
   negative values on failure. ~5 call sites in the launcher. Each needs a
   `if< result 0` → trap or exit. ~3-5 ARM64 instructions per check.

2. **GPU (SM90):** No error concept — the SM either executes or hangs. The only
   detection is the completion semaphore never being written. `sync_wait_flag_timeout`
   (sync.ls line 52) polls N times then gives up. "Gives up" means exit —
   a hung GPU context may need `nvidia-smi -r` or process kill. Lithos can
   detect but not fix.

3. **Inference (numerical):** NaN propagation, softmax overflow, wrong tensor
   shapes are silent in FP32/FP16 arithmetic. No runtime detection — these are
   caught at validation time (cosine similarity vs reference), not at runtime.

**Total scope:** ~20 branch instructions across the entire codebase. Not a
language feature — a handful of guards at known points using existing control
flow (`if>=`, `label`, `goto`).

```
Critical now:
[ ] Check openat/fstat/mmap return values in launcher.ls load_file
    (failed open → negative fd → silent corruption or SIGSEGV)
[ ] Wire sync_wait_flag_timeout (exists in sync.ls line 52, unused)
    into launcher — infinite hang on any GPU fault currently
[ ] pmc_sanity_check in init.ls — actually branch on PMC_BOOT_0 read
    (currently reads but never checks; comment says "zero means dead silicon")

Deferrable:
[ ] ECC / thermal / Xid error register monitoring
[ ] Watchdog / health check loop
[ ] Graceful GPU reset on hang
```

---

## STALE ITEMS (corrected by scouts 2026-04-13)

Items previously listed as gaps that are actually resolved:

| Item | Status | Evidence |
|------|--------|----------|
| config.json reader | **EXISTS** | `compiler/config-reader.ls` (873 lines), parses 17 fields, untested |
| Conv1D 4-tap | **IMPLEMENTED** | `recur.ls` lines 12-40, complete with computed-offset workaround |
| Computed-offset indexing | **WORKING** | `.ls` files use `x[var]` pattern throughout |
| Weight pointer table | **BUILT** | `launcher.s` lines 298-304 (64 layers × 26 slots × 8 bytes) |
| RoPE partial rotary | **CORRECT** | `attend.ls` rotates dims 0-63, passes through 64-255 |

---

## CRITICAL BLOCKERS

### Shared (unblocks both platforms)

1. **Bootstrap parser** — compositions (`name args :`) + bindings (`name expr`) — NO `=`
2. ~~**compiler.ls rewrite**~~ — DONE. Pure Lithos syntax. Zero =, fn, ->, load_u*, syscall.
3. **First compile**: bootstrap parses compiler.ls → lithos-stage1
4. **Self-compile**: lithos-stage1 compiles compiler.ls → lithos (fixed point)

### macOS / Apple Silicon

5. ~~**Bootstrap platform split**~~ — DONE. build-darwin.sh, 9 transforms, 153KB Mach-O.
5b. **Mach-O output writer** — so compiled .ls runs natively on macOS (not just ELF)
6. **Metal compute emitter** — emit MSL from .ls kernel source
   Metal is the floor. No raw ISA. Compiler generates Metal Shading Language.
7. **Metal runtime** — objc_msgSend bridge to Metal.framework from ARM64
8. **First Metal kernel** — one .ls kernel compiled to MSL, dispatched, verified

### Linux / GH200

9. ~~**FSP boot wiring**~~ — DONE (wired at step 6, BCR/FSP order fixed)
10. ~~**register_count probe**~~ — DONE. SPD offset 0x094
11. **First kernel execution**: GSP+GPFIFO+QMD runs one compiled .ls kernel
12. **KV cache + state allocation** (M3) — allocate and wire inference buffers
13. **Tokenizer** (M1) — at minimum, hardcoded token IDs for test prompts
14. **One token through one layer**: all 71 steps, diff < 1e-3 vs reference
15. **Autoregressive loop** (M2) — generate sequences, detect EOS
16. **Error handling** (M5) — syscall checks + poll timeout
17. **FP16 opcodes** (M4) — after correctness, before performance
18. **Full 64-layer inference**: Qwen 3.5 27B produces coherent text

---

## E2E SAFETY

- `bin/check-site` — verifies all 26 nav links return 200 before deploy
- Run before every `vercel --prod`
