# Lithos — Master Status

**Target model:** Qwen 3.5 27B (64 layers, 48 DeltaNet + 16 full-attention, GPTQ W4A16)
**Architecture:** Two cooperative megakernels — one cubin per layer type, grid-sync loop at runtime
**Current phase:** Bootstrap compiler stabilization → first self-compile
**Last updated:** 2026-04-13

---

## Architecture (confirmed)

- **Two megakernels** because two layer types (DeltaNet + full attention) have different register/memory footprints.
- One cubin covers all 48 DeltaNet layers. One cubin covers all 16 attention layers.
- Within each megakernel, the loop over layers runs **at runtime** with a cooperative grid-sync barrier (`bar.sync` / `membar.gl`) between layer iterations — NOT compile-time unrolling.
- Compile-time unrolling would copy the layer body N times (48× or 16×), producing enormous, inflexible code. Runtime grid-sync is fast (Hopper hardware primitive) and keeps code compact.
- Both cubins emitted from `.ls` source by the Lithos compiler as raw Hopper binary (direct instruction encoding). No SASS assembler, no ptxas, no PTX, no CUDA runtime.
- Launcher calls dispatch a handful of times per token, not per-layer.
- `compiler.ls` — **5,467 lines** (not 4,739 as previously listed in PLAN.md).
- DeltaNet state memory: **48MB** fixed per sequence (48 layers × 16 heads × 128 × 128 × f32). Not 12.6MB.
- `register_count` lives in the **Shader Program Descriptor (SPD) at offset 0x094** — not in cbuf0 or QMD.
- `qmd.ls` comment claiming "register_count lives in cbuf0" is wrong. SPD 0x094 is correct (3-point verified via cbuf0_fields.md probe).
- Launch is **5 inline loads**: QMD (528B) + fence (8B) + context (384B) + SPD (384B) + patch (4B).
- `megakernel-link.ls` — 833 lines, implemented. Covers the linker for emitting the two megakernel cubins.

**Platforms:**

| | Linux / GH200 | macOS / Apple Silicon |
|---|---|---|
| Host ISA | ARM64 | ARM64 |
| Host ABI | Linux syscalls (X8, SVC #0) | Darwin syscalls (X16, SVC #0x80) |
| Binary format | ELF64 | Mach-O |
| GPU | Hopper SM90a | Apple GPU (M-series) |
| GPU backend | Raw SM90 binary (probe-verified) | TBD — Metal or lower (research needed) |
| GPU access | vfio-pci + MMIO (no libcuda) | IOKit / Metal / TBD |

---

## Progress

### Shared (both platforms)
```
FOUNDATIONS   [████████] complete   language, kernels, grammar, dicts, docs
COMPILER      [████████] DONE — rewritten to pure Lithos syntax (zero =, fn, ->, load_u, syscall)
BOOTSTRAP     [██████░░] links; compositions compile; compiler.ls parses to line ~247 (see TBD §1)
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
PLATFORM SPLIT [░░░░░░░░] bootstrap syscall + Mach-O writer not started
GPU RESEARCH   [░░░░░░░░] Apple GPU ISA access depth unknown
GPU BACKEND    [░░░░░░░░] no Apple GPU emitter exists
METAL FALLBACK [░░░░░░░░] Metal compute shader path not explored
```

### Subsystem checklist

**Bootstrap (pure ARM64 assembly, `bootstrap/*.s`)**
```
[✓] Lithos bootstrap binary builds, 35KB .bss (not 528MB)
[✓] Entry, memory, DTC runtime wired (_start in driver.s, not lithos-bootstrap.s)
[✓] 7 stack-alignment SIGBUS bugs in lithos-elf-writer.s fixed
[✓] Smoke test: `var x 42` → valid ARM64 ELF produced
[✓] Build: bootstrap/build.sh, Makefile, macros prelude working
[✓] Expression parsing works: `var y x + 1` produces valid ELF
[ ] Composition parsing: see TBD §1 — contradictory evidence, requires investigation
[ ] Binding syntax: see TBD §3 — `=` vs no-`=` contradiction
[ ] Audit: compiler.ls parses cleanly through bootstrap                      ← BLOCKING

Platform split needed:
[ ] Extract syscall callsites into syscall-linux.s / syscall-darwin.s
    Linux:  syscall number in X8, SVC #0, error = negative return
    Darwin: syscall number in X16 (0x2000000 | num), SVC #0x80, error = carry flag
    Calls: openat, close, read, write, mmap, exit (~6 wrappers)
[ ] Update build.sh to detect uname and select platform syscall file
[ ] Keep ELF writer for Linux targets + GPU cubins (both platforms need cubin ELF)
[ ] Write macho-writer.s for macOS host executables
    Mach-O header, LC_SEGMENT_64, LC_MAIN, __TEXT/__DATA sections
    Base address: 0x100000000 (not 0x400000)
[ ] --target flag in driver.s to select output format
```

**Compiler (`compiler/compiler.ls`, 5,467 lines)**
```
Platform-common:
[✓] Lexer (Section 3): UTF-8 multi-byte for → ← ↑ ↓ Σ △ ▽ √ ≅ ≡
[✓] ARM64 backend (Section 1): 90+ emitters, branch patching
[✓] Parser (Section 4): recursive descent, dual backend routing
[✓] Safetensors reader (Section 5): JSON header + tensor index
[✓] Main entry (Section 7): argv → mmap → lex → parse → emit → write
[✓] config-reader.ls (873 lines, untested): parses all 17 Qwen 3.5 fields, classifies layer_types[], provides accessors

Linux / GH200 — GPU backend:
[✓] SM90 backend (emit-gpu.ls): 35+ raw Hopper opcodes, ctrl words, grid-sync
    128-bit instructions (8-byte inst + 8-byte ctrl), probe-verified on sm_90a.
[✓] ELF writer (Section 6): 9-section cubin ELF64 structure
[ ] No per-layer dispatch loop (hybrid-layers.md design unimplemented)
[ ] No megakernel linker (megakernel-link.ls exists — 833 lines — but not wired)
[ ] cubin_buf is 512KB; 64-layer megakernel needs ~300MB (600× too small)

macOS / Apple Silicon — GPU backend (not started):
[ ] Apple GPU ISA research (Metal Level 1–3 depth TBD)
[ ] Apple GPU emitter
[ ] Apple GPU binary format
[ ] Determine target workload for macOS GPU

Platform-specific host I/O (compiler.ls lines 5244–5390):
[ ] mmap_file (line 5334): hardcoded Linux syscall numbers — needs Darwin variant
[ ] write_file (line 5368): hardcoded Linux syscall numbers — needs Darwin variant
[ ] elf_build_arm64 (line 5244): emits ELF64 at 0x400000
    Needs parallel macho_build_arm64 for macOS host executables

Shared open items:
[ ] `$` register prefix not a lexer token
[ ] arch/hopper.dict and arch/arm64.dict never read
```

**Inference kernels (`inference/*.ls`)**
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
[✓] reduce.ls — rmsnorm, rmsnorm_residual, l2norm
[✓] Conv1D 4-tap — fully implemented in recur.ls (lines 12–40)
[✓] Computed-offset indexing — .ls files use x[var] pattern, works today
[ ] sample_argmax — see TBD §4: contradictory evidence (PLAN.md says done; GAPS.md says stub at reduce.ls:209–217)
[ ] Apply v7 software pipelining to gemv.ls
[ ] Apply v2 v4.u32 K-wise weight loads
[ ] Per-projection shape selection (compile-time specialization per layer)
[ ] Transposed [N, K/8] weight layout (preprocess step)
[ ] FP16 opcodes (HADD, HMUL, HFMA, HMNMX) — performance gap, not correctness
```

**Runtime (`runtime/*.ls`)**
```
[✓] init.ls — vfio-pci + BAR0/BAR4 mmap, PMC sanity, GSP boot dispatch
[✓] mem.ls — BAR4 bump allocator (coherent HBM, no dma_alloc_coherent)
[✓] elf_load.ls — parse compiled GPU ELF, extract entry_pc via .symtab walk
[✓] qmd.ls — 528-byte descriptor with empirically-probed offsets
[✓] cbuf0.ls — constant buffer
[✓] pushbuffer.ls — 6-method GPFIFO submission sequence
[✓] doorbell.ls — USERD+0x8c GPPut write + USERD+0x90 doorbell
[✓] launch.ls — kernel dispatch, ties QMD+cbuf0+pushbuffer+doorbell
[✓] sync.ls — completion polling (flag-based + GPGet)
[✓] teardown.ls — munmap + close
[✓] dispatch.ls — top-level dispatch_kernel composition
[ ] pushbuffer.ls needs update: launch is 5 inline loads (QMD + fence + context + SPD + patch), not just QMD
None of these have been executed — the compiler must work first.
```

**Hardware path (`GSP/*.s`)**
```
[✓] bar_map.s — sysfs resource0/resource4 mmap
[✓] pmc_check.s — PMC_BOOT_0 read, PMC_INTR_0 clear
[✓] falcon_reset.s — Falcon reset/deassert sequence
[✓] hbm_alloc.s — bump allocator from BAR4
[✓] fw_load.s — ELF64 parse, PT_LOAD copy to HBM
[✓] bcr_start.s — RISC-V BCR register programming
[✓] poll_lockdown.s — PRIV_LOCKDOWN release polling
[✓] rpc_channel.s — channel allocation via GSP RPC
[✓] FSP wired into GSP/boot.s at step 6 (BCR/FSP ordering fixed)
[✓] register_count: in SPD at offset 0x094 (bits 23:16). 3-point verified.
[ ] vfio-pci: not implemented (sysfs mmap only)
```

**Launcher integration**
```
[✓] src/launcher.ls — 162 lines, replaces src/launcher.s
[ ] Removes all 42 libcuda call sites
[ ] Single dispatch per token (megakernel, not per-kernel)
[ ] Tokenize + emit stubs (full tokenizer later)
```

**Documentation**
```
[✓] docs/language/primitives.md
[✓] docs/inference/model-config.md
[✓] docs/inference/hybrid-layers.md
[✓] docs/gpu-register-model.md
[✓] docs/qmd_fields.md
[✓] docs/arm64-encodings.md
[✓] docs/launcher-audit.md (42 libcuda sites mapped)
[✓] docs/minds-revisited.md
[✓] docs/bootstrap-compile.md + v2
[✓] docs/weights-as-code-{compiler,language,inference}.md
[✓] docs/gsp-native.md
[✓] docs/self-hosting-compiler.md
[✓] docs/regalloc_design.md
[✓] HTML site on Vercel: /language, /pipeline, /roadmap, architecture, compiler,
    kernels, quantization, performance, bandwidth, glossary (26 pages, all 200)
[✓] /status dashboard live (hand-crafted HTML, progress cards, blocker banner)
[✓] bin/build-status generates from PLAN.md, local server on port 8080
```

**Dead code removed**
```
[✓] 34 Python files deleted
[✓] 21 orphan cubins deleted
[✓] 36 sass/ probe files deleted
[✓] Fake inference claims deleted (index.html "Paris" logit 18.057, cosine 1.0)
[ ] kernel/ (C kernel module) — keep until GSP boot fully runs, then delete
[ ] src/launcher.s (4096 lines) — delete when src/launcher.ls compiled + running
```

> **Forth files — see TBD §5:** PLAN.md says "Zero Forth, 21 files deleted." GAPS.md references `elf-wrap.fs`, `gpu/emit.fs`, `forth-bootstrap.s` as current. Investigate before marking as done.

---

## Bootstrap

### File inventory (9 source files — corrected)

> Note: PLAN.md listed `lithos-parser.s` in the build. Corrected: the build uses `lithos-table.s` + `lithos-glue.s`. The file list below reflects the actual 9-file build.

| File | Lines | Size | Role |
|---|---|---|---|
| lithos-bootstrap.s | 5,334 | 112 KB | DTC Forth runtime (87 words) |
| lithos-parser.s | ~2,961 | ~77 KB | Recursive-descent parser + codegen |
| emit-arm64.s | 2,833 | 68 KB | ARM64 instruction encoder (100+ words) |
| lithos-expr.s | 2,765 | 77 KB | Expression compiler (DTC, dual-target) |
| lithos-elf-writer.s | 2,026 | 56 KB | ELF64 writer (ARM64 + cubin) |
| lithos-lexer.s | 909 | 24 KB | Tokenizer (UTF-8 aware) |
| driver.s | 744 | 23 KB | CLI driver + pipeline orchestration |
| lithos-glue.s | — | — | Glue and cross-file wiring |
| ls-shared.s | 102 | 4 KB | Shared buffers and cursors |
| **Total** | ~17,674+ | ~441 KB+ | |

### Link order
```
driver.o → lithos-bootstrap.o → lithos-lexer.o → lithos-parser.o →
lithos-expr.o → emit-arm64.o → lithos-elf-writer.o → lithos-glue.o → ls-shared.o
```

### Current build status

**BROKEN** — actual failure cause: **GNU ld flags rejected by Apple ld** (not `undefined emit_ldr_x_zero` as previously documented in bootstrap/STATUS.md — that undefined-symbol issue has been resolved; it now lives in `lithos-glue.s`).

Workers actively modifying `lithos-parser.s`. The `.bak` files contain last known-good versions (2,952 lines for parser, 938 for lexer).

### What works

- **Lexer** (lithos-lexer.s, 909 lines): fully implemented. Tokenizes `.ls` into (type, offset, length) triples. Integer/float literals, identifiers, `$`-prefixed register names, full UTF-8 math set, all operators, 30+ keywords, comment syntax, indentation tracking.
- **Emitter** (emit-arm64.s, 2,833 lines): 100+ DTC words. Full ARM64 coverage: arithmetic, logic, shift, compare, move, memory (LDR/STR/LDRB/LDRH/LDP/STP), branches (all 14 conditions, CBZ, CBNZ, TBZ, TBNZ), conditional select, system ops, ADR/ADRP, forward branch patch helpers.
- **ELF Writer** (lithos-elf-writer.s, 2,026 lines): two complete ELF64 writers — ARM64 Linux executable (ET_EXEC, EM_AARCH64, PT_LOAD at 0x400000) and GPU cubin (ET_EXEC, EM_CUDA, 9-section layout including `.nv.info` attributes: REGCOUNT, FRAME_SIZE, PARAM_CBANK, etc.).
- **Driver** (driver.s, 744 lines): `.ls` → Lithos pipeline, `.fs` → Forth interpreter, argument parsing, source mmap, 5-stage pipeline, DTC trampoline.
- **DTC Bootstrap** (lithos-bootstrap.s, 5,334 lines): 87-word DTC Forth runtime.
- **Parser** (lithos-parser.s, ~2,961 lines): integer literals, arithmetic with precedence, comparisons, bitwise, shifts, register read/write, memory load/store, if/elif/else, while, for loops, var/buf/const/label/return, composition calls, parenthesized expressions.

### Known bugs

1. **Build broken**: GNU ld flags rejected by Apple ld. `lithos-parser.s` also has in-progress width-specific memory op work. Revert from `.bak` for a linking build on Linux.
2. **Width ignored in memory ops**: `→ 8 addr` and `→ 64 addr` both emit 64-bit LDR. Width argument parsed but not used to select LDRB/LDRH/LDR-W/LDR-X.
3. **TOK_TRAP/lexer mismatch**: Parser defines TOK_TRAP=89, lexer never emits it (emits TOK_IDENT for "trap"). Statement-level check is dead code; atom parser string-match works around it.
4. **Float literals treated as integers**: TOK_FLOAT tokens go through `parse_int_literal`, decimal point skipped. No NEON/FP codegen.
5. **Unary math ops are pass-through stubs**: `√`, `≅`, `≡` parse their operand but emit nothing.
6. **Reductions are pass-through stubs**: `Σ`, `△`, `▽` parse operand, no loop-based or warp-based reduction code.
7. **`each` is a stub**: Allocates register for loop variable, no loop or thread-index codegen.
8. **Register allocator is linear-only**: Allocates X9..X15 (7 total), no spilling. Complex expressions trigger `parse_error_regspill`.
9. **No `$NAME` dictionary lookup**: Named special registers not recognized; only numeric `$N` works.
10. **Composition call result convention**: `parse_comp_call` always returns X0 regardless of what the composition produces.

### Register conventions
```
X26 = IP     (DTC instruction pointer)
X25 = W      (DTC working register)
X24 = DSP    (data stack pointer, full descending)
X23 = RSP    (return stack pointer)
X22 = TOS    (top of stack, cached)
X20 = HERE   (dictionary pointer)
X21 = BASE   (number base, default 16)
X19 = TOKP   (parser: current token pointer)
X27 = TOKEND (parser: past-last-token pointer)
X28 = SRC    (parser: source buffer pointer)
```

---

## Open Gaps (ordered by critical path)

### 1a. Cubin ELF wrapping — BLOCKING

The ELF writer in `lithos-elf-writer.s` emits the 9-section cubin layout. However, GAPS.md flags `compiler/elf-wrap.fs` as only emitting a 64-byte ELF header (but see TBD §5 on whether `.fs` files exist).

Missing from any confirmed implementation:
- `.nv.info` structured attributes: `EIATTR_REGCOUNT`, `EIATTR_FRAME_SIZE`, `EIATTR_PARAM_CBANK`, `EIATTR_KPARAM_INFO`, `EIATTR_MIN/MAX_STACK_SIZE`, `EIATTR_CUDA_API_VERSION`, `EIATTR_MAX_THREADS`
- Symbol table entry with `STB_GLOBAL` / `STT_FUNC` / `STO_CUDA_ENTRY` (0x10)
- `EIATTR_COOPERATIVE_GROUP_INSTR_OFFSETS` in `.nv.info.<kernel>` (required for cooperative launch)
- Until confirmed complete, `cuModuleLoadData` will reject any Lithos-emitted cubin.

### 1b. Register allocator — BLOCKING AT SCALE

Bootstrap parser uses monotone counters (`freg+`, `rreg+`, `rdreg+`, `preg+`): no liveness, no reuse. Works for small programs. DeltaNet has 22+ ops with hundreds of live values — will exhaust Hopper's 255-per-thread register budget and wreck occupancy before that.

Needed: linear-scan allocator with a liveness pass. Graph coloring is overkill for the straight-line-plus-short-loops shape of these kernels. `docs/regalloc_design.md` exists; implementation does not.

### 1c. Backend dispatch coverage

The `li-backend @ if` selector at `parser.fs:686` routes per-op to PTX text or SASS binary.

Already routed: arithmetic ops, for-loops, setp/predication, branches.

Not yet routed: shared memory (LDS/STS), warp shuffles (SHFL.BFLY is in SASS emitter but parser still emits PTX for it), type conversions, bounds checks, barriers. Each unrouted op keeps the PTX path alive. Grunt work, not research.

### 1d. Instruction scheduling

SASS emitter uses hardcoded control-word stall/barrier values from `tools/sass_probe.py` probing. Correct for simple LDG/STG patterns, not tunable for deep pipelining. Acceptable for first working kernel; a latency-aware scheduler is a later pass.

### 1e. Cooperative grid-sync primitives — HIGH PRIORITY

Both cubins are cooperative megakernels: layer loop runs inside the kernel with grid-wide sync between layers. SASS emitter needs:
- Grid-wide barrier (cooperative groups `this_grid().sync()` equivalent at SASS level — Hopper combination of release/acquire fences + global counter barrier)
- Proper encoding for `cuLaunchCooperativeKernel` (cubin must advertise `EIATTR_COOPERATIVE_GROUP_INSTR_OFFSETS`)
- Block-level `BAR.SYNC` patterns tuned for grid-sync frequency (48 syncs per DeltaNet cubin invocation)
- Language-level construct in `.ls` to express "run body N times with grid sync between iterations"

### 1b½. Kernel-building parameterization (factory)

The compiler can turn a hand-written `.ls` file into a SASS cubin. It cannot yet synthesize `.ls` for a specific model from architecture metadata.

Missing:
- **Compile-time parameters**: values known at parse time, substituted before SASS emission (`param hidden_dim : u32 = compile_time`)
- **Template selection**: dispatch keyed on architectural fingerprint from safetensors header
- **Quantization parameterization**: GEMV template parameterized over `bits_per_weight`, `group_size`, dequant layout

Frontend-only change; no new SASS emission, no register allocator changes.

### 2a. Computed-offset array indexing — BLOCKS Conv1D

GAPS.md §2a reports a stub comment in `recur.ls`: *"Hand-unrolled in PTX below; Lithos lacks computed-offset indexing."* However PLAN.md's stale-gaps table marks computed-offset indexing as resolved (`x[var]` pattern works). See TBD §2.

### 2b. Outer-product / rank-1 update syntax

`delta_update.ls` and `recur.ls` nest `for` loops for `S += β·(V⊗K − S·(K⊗K))`. Native syntax `x ⊗ y → M` is nice-to-have, not blocking.

### 2c. `max` intrinsic

Online softmax uses `setp.ge` + predicated branch for max. FMNMX is in SASS emitter — needs language-level `max` mapping to it.

### 3a–3c. Layer kernel gaps

- **Attention layer** (~95% complete): composes into one kernel once backend emits it.
- **DeltaNet layer** (~85% complete): blocked on Conv1D (see 2a / TBD §2). Delta rule cleanup is polish.
- **MLP / embed / lm_head / sampling**: SiLU, residual add, GEMV individually present. No fused MLP kernel. Embed not in `.ls` yet. lm_head needs FP16 variant. Sampling — see TBD §4.

### M1. Tokenizer — BLOCKING, no implementation exists

Qwen 3.5 uses BPE with 248K vocab + ~130K merges. Zero tokenizer code exists in Lithos.

Options:
1. Implement BPE in `.ls` (hard: needs JSON parsing of 12MB merge table + regex pre-split)
2. Implement BPE in ARM64 assembly in bootstrap layer
3. Pre-tokenize externally, feed raw token IDs (simplest, external dependency)

For first-token milestone, option 3 (hardcoded token IDs) is sufficient.

### M2. Autoregressive generation loop — BLOCKING

No generation loop exists. Missing:
- Loop in launcher.ls: forward pass → read argmax → check EOS → feed back as next input
- EOS token detection (no EOS ID hardcoded anywhere)
- Max-length parameter
- Streaming output (no host readback/print)
- Temperature scaling, top-p, top-k, repetition penalty

For first-token milestone: single forward pass is sufficient.

### M3. KV cache + DeltaNet state lifecycle — BLOCKING

Kernels accept `k_cache`/`v_cache`/`dn_state` pointers but nothing allocates or manages them.

```
[ ] inference_init.ls — allocate KV cache (16 attn layers × 4 KV heads × 256 dim × max_seq)
[ ] inference_init.ls — allocate DeltaNet state (48 layers × 16 heads × 128 × 128 × f32 = ~48MB)
[ ] inference_init.ls — zero-initialize DeltaNet state at sequence start
[ ] Per-token KV append — write new K/V at position seq_len, increment counter
[ ] Pre-allocate max sequence length or implement grow-on-demand
```

Memory ceiling: ~131KB/token for KV across 16 attention layers. At 128GB BAR4 minus weights (~14GB) and activations, theoretical max ~500K–800K tokens.

### M4. FP16 compute opcodes — PERFORMANCE GAP

`emit-gpu.ls` has zero FP16 opcodes. Only FP32: FADD, FMUL, FFMA, FMNMX. Not blocking for correctness; blocking for closing the performance gap with vLLM.

```
[ ] Add emit_hadd, emit_hmul, emit_hfma (FP16 opcodes) to emit-gpu.ls
[ ] Add emit_hfma2 (packed 2×FP16) for bandwidth-bound kernels
[ ] Probe and verify FP16 instruction encodings via ptxas/nvdisasm
[ ] Add f32>f16 / f16>f32 conversion opcodes
```

### M5. Error handling — CRITICAL for development velocity

Zero error handling exists. Silent failures will waste debugging time.

Error model: compositions that can fail return a status register. Caller checks via `if< result 0` or equivalent. No stack unwinding, no exception frames.

```
Critical now:
[ ] Check openat/fstat/mmap return values in launcher.ls load_file
    (failed open → negative fd → silent corruption or SIGSEGV)
[ ] Wire sync_wait_flag_timeout (exists in sync.ls line 52, unused)
    into launcher — infinite hang on any GPU fault currently
[ ] pmc_sanity_check in init.ls — actually branch on PMC_BOOT_0 read
    (currently reads but never checks; "zero means dead silicon")

Deferrable:
[ ] ECC / thermal / Xid error register monitoring
[ ] Watchdog / health check loop
[ ] Graceful GPU reset on hang
```

### Native launcher / libcuda replacement

The hot dispatch path is zero-syscall (pure userspace memory writes: QMD into mapped USERD page + doorbell). Context creation is one-time cost (~450 ioctls). Total ioctl surface: 25 unique calls — finite and enumerable.

Scope: ~few hundred lines of Lithos. Replaces ~80MB of `libcuda.so`.

Still needed:
- Lithos syscall wrappers (`openat`, `close`, `ioctl`, `mmap`, `munmap`) as first-class Lithos words
- Struct marshaling helpers (`put-u32 @ offset`, `put-u64 @ offset`, field-packing for NVIDIA RM ioctl structs)
- ABI reference: precise struct layouts and ioctl numbers for all 25 calls from `open-gpu-kernel-modules`
- QMD encoder: Hopper compute launch descriptor (`HOPPER_COMPUTE_A` / `HOPPER_COMPUTE_B` fields)

### Language frontend gaps

- `$` register prefix not a lexer token
- `arch/hopper.dict` and `arch/arm64.dict` never read
- `goto` / `continue` tokens defined (TOK_GOTO=92), no parser handler
- String literals: no string token type, no `.data` section emission
- Width-specific memory ops: `→ 8/16/32 addr` always emits 64-bit LDR/STR
- `data` sections: parser references TOK_DATA but it's undefined in lexer/parser enums
- `host` / `kernel` prefixed compositions: parsed at top level, handlers not implemented
- `stride` loop: token defined, no parser handler
- Dimensional ops (`** ***`): tokens defined, no codegen
- Float arithmetic: float tokens parsed but treated as integer bits

### Self-hosting path (S1–S4)

```
S1. Update .li compiler files to new grammar (fn/-> → →/←/↑/↓/$ + composition syntax)
    Prerequisite: bootstrap can compile basic .ls programs end-to-end
S2. Write parser in .li — lithos-parser.li (recursive descent, single-pass, direct emission)
    Dependency: S1 complete
S3. Write SM90 emitter in .li
    Risk: Forth GPU emitter was deleted. Opcode data must be recovered from git history
          or re-probed (commit 17d5103).
S4. Bootstrap compile:
    forth-bootstrap lithos.fs compiler.li → lithos-stage1
    ./lithos-stage1 compiler.li → lithos
    ./lithos compiler.li → lithos-verify
    diff lithos lithos-verify  (must match)
```

---

## TBD — Requires Investigation

### 1. Composition parsing

**Status: TBD — contradictory evidence, requires investigation**

- Option A (PLAN.md line 69): `name args :` → parse error at line 1196, crashes — listed as BLOCKING.
- Option B (bootstrap/STATUS.md parser table): Compositions (`name args :`) — **Works**, emits STP/LDP prologue/epilogue, params in X0-X7.
- Action: run `./lithos-bootstrap simple_composition.ls` and verify whether a composition compiles without error. Check if the STATUS.md "Works" row reflects a version of `lithos-parser.s` that was later broken by the in-progress width-specific memory op work (see Bug #1 — parser broken, revert to `.bak`).

### 2. Computed-offset indexing + Conv1D

**Status: TBD — contradictory evidence, requires investigation**

- Option A (PLAN.md stale-gaps table, line 183–184): Both marked `[✓]` resolved by scouts 2026-04-13. "Conv1D 4-tap — fully implemented in recur.ls (lines 12–40)." "Computed-offset indexing — .ls files use `x[var]` pattern, works today."
- Option B (GAPS.md §2a, line 86–88): "Has a stub comment: 'Hand-unrolled in PTX below; Lithos lacks computed-offset indexing'." DeltaNet layer marked 85% complete, Conv1D listed as "✗ stub — blocked on 2a."
- Action: `grep -n "computed-offset\|Hand-unrolled\|x\[var\]" inference/recur.ls` to determine which description matches the current file. If the stub comment exists, PLAN.md's scout resolution was premature.

### 3. `=` assignment in the language

**Status: TBD — contradictory evidence, requires investigation**

- Option A (PLAN.md line 100): "The language has NO `=` assignment. Bindings: first token = name, rest = expression." Listed as a BLOCKING parse gap: "Binding syntax: `name expr` (no = sign) — bootstrap treats first token as name."
- Option B (bootstrap/STATUS.md parser table, line 58): Assignment (`x = expr`) — **Works**, allocates register, adds to symbol table. Lexer tokenizes `=` as a single-char operator.
- Action: Determine whether `=` is (a) a working feature of the current bootstrap parser that will be eliminated from the language spec, or (b) already absent from compiler.ls and the bootstrap just happens to support it legacy. Run `grep -n " = " compiler/compiler.ls | head -20` to see if the target source uses `=` syntax.

### 4. `sample_argmax`

**Status: TBD — contradictory evidence, requires investigation**

- Option A (PLAN.md line 311): "`sample_argmax` in reduce.ls is fully implemented (not a stub)."
- Option B (GAPS.md §3c, line 132): "Sampling (argmax / top-k): stub in `reduce.ls:209-217`."
- Action: `sed -n '205,220p' inference/reduce.ls` to read the implementation. If it contains substantive warp-reduce argmax logic, Option A is correct. If it contains a comment like "// stub" or a single-register pass-through, Option B is correct.

### 5. Forth files — existence

**Status: RESOLVED — No Forth. PLAN.md is correct.**

21 Forth files (compiler/*.fs, gpu/*.fs, core.fs, patterns.fs, etc.) are deleted. GAPS.md references to `elf-wrap.fs`, `gpu/emit.fs`, `forth-bootstrap.s`, and `parser.fs:686` are stale — written against pre-deletion state. Those line-number references are invalid. Any gap analysis that depended on `.fs` files must be re-evaluated against the current pure-Lithos + ARM64-assembly stack.

### 6. `=` syntax elimination timing

**Status: TBD — requires investigation**

- Is `=` being eliminated from the language spec (future work), or is it already absent from compiler.ls's target output and only the bootstrap retains it for legacy parsing?
- This determines whether the PLAN.md BLOCKING item ("bootstrap treats first token as name") is an active parse regression or a planned migration.
- Action: Check whether compiler.ls uses `x = expr` syntax anywhere. If yes, the bootstrap must support `=`. If no, the bootstrap's `=` support is legacy-only and the PLAN.md goal is to ensure bindings like `x 42` work instead.

---

## Milestones (M1–M5)

### Critical blocker sequence
1. **Bootstrap parser stabilization** — compositions + bindings (resolve TBD §1 and §3)
2. **Compiler.ls parses cleanly** through bootstrap (audit pass)
3. **First compile**: bootstrap parses compiler.ls → lithos-stage1
4. **Self-compile**: lithos-stage1 compiles compiler.ls → lithos (fixed point)
5. **Register allocator** (1b) — linear-scan with liveness pass
6. **Cubin ELF wrapping** (1a) confirmed complete — verify `cuModuleLoadData` accepts output
7. **Cooperative grid-sync primitives** (1e) — SASS encoding + `.ls` language construct
8. **Backend dispatch closure** (1c)
9. **Computed-offset indexing** (2a) — unblocks Conv1D
10. **First kernel execution**: GSP + GPFIFO + QMD runs one compiled `.ls` kernel
11. **KV cache + state allocation** (M3)
12. **Tokenizer** (M1) — at minimum, hardcoded token IDs for test prompts
13. **One token through one layer**: all 71 steps, diff < 1e-3 vs reference
14. **Autoregressive loop** (M2)
15. **Error handling** (M5) — syscall checks + poll timeout
16. **FP16 opcodes** (M4) — after correctness, before performance
17. **Full 64-layer inference**: Qwen 3.5 27B produces coherent text

### Milestone definitions

**M1. Tokenizer** — hardcoded token IDs sufficient for first-token test.

**M2. Autoregressive loop** — launcher.ls loops forward pass → argmax → EOS check → next token feed.

**M3. KV cache + DeltaNet state lifecycle** — inference_init.ls allocates and manages all per-sequence state (~48MB DeltaNet state + ~131KB/token KV cache). Pre-allocated at max sequence length or grow-on-demand.

**M4. FP16 compute opcodes** — HADD, HMUL, HFMA, HFMA2 in emit-gpu.ls. 2× ALU throughput, 2× memory bandwidth efficiency.

**M5. Error handling** — ~20 branch instructions at known fault points. openat/fstat/mmap return checks, sync_wait_flag_timeout wired, pmc_sanity_check actually branches.

### Validation targets

- Per-kernel: bit-identical output vs PyTorch reference for same inputs.
- End-to-end: "The capital of France is" → top-1 `"Paris"` (logit 18.057, cosine 1.0 at every one of 64 layers). Proven with Python pipeline. Same test must pass with two Lithos-compiled cubins replacing 21 hand-written PTX kernels.
- No tok/s numbers until end-to-end cosine = 1.0 at every layer.

### E2E safety

`bin/check-site` — verifies all 26 nav links return 200 before deploy. Run before every `vercel --prod`.
