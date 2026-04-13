# Gaps to Full Inference

Last updated: 2026-04-13

Target: Qwen3-30B-A3B (64 layers, 48 DeltaNet + 16 full-attention, GPTQ W4A16) running per-token decode on GH200 via Lithos-compiled SASS cubins, dispatched by a native (non-Python) launcher.

Architecture (confirmed): **two cooperative megakernels.** One cubin for all 48 DeltaNet layers, one cubin for all 16 attention layers. Each cubin is a *cooperative grid-sync kernel* — the 48-layer (or 16-layer) iteration lives inside the kernel, with grid-wide sync between layers. Both emitted from `.li` source by the Lithos compiler straight to Hopper SASS. The launcher calls `cuLaunchKernel` a handful of times per token, not per-layer.

CUDA graphs aren't needed — when dispatch is measured in native code nanoseconds and there are only a handful of launches per token, there's nothing for graphs to amortize. The earlier PTX-based cooperative-megakernel attempts (`forward_pass_*`, `fused_*`) weren't wrong conceptually — they were implemented through PTX + ptxas, which removed the level of control needed to make grid-sync cheap. Direct SASS gives back control over stall counts, barrier slots, control-word reuse flags, and scheduling — the things that actually determine whether cooperative grid-sync is fast or slow.

---

## 1. Compiler backend

### 1a. Cubin ELF wrapping — **BLOCKING**
- `compiler/elf-wrap.fs` currently emits only the 64-byte ELF header.
- Missing: section header table (`.shstrtab`, `.strtab`, `.symtab`, `.nv.info`, `.nv.info.<kernel>`, `.nv.constant0.<kernel>`, `.text.<kernel>`, `.nv.callgraph`, `.nv.shared.<kernel>` when needed).
- Missing: `.nv.info` structured attributes (`EIATTR_REGCOUNT`, `EIATTR_FRAME_SIZE`, `EIATTR_PARAM_CBANK`, `EIATTR_KPARAM_INFO`, `EIATTR_MIN/MAX_STACK_SIZE`, `EIATTR_CUDA_API_VERSION`, `EIATTR_MAX_THREADS`).
- Missing: symbol table entry with `STB_GLOBAL` / `STT_FUNC` / `STO_CUDA_ENTRY` (0x10).
- Until this lands, `cuModuleLoadData` rejects any Lithos-emitted cubin regardless of SASS correctness. *(Worker dispatched on this.)*

### 1b. Register allocator — **BLOCKING at scale**
- Current scheme in `parser.fs`: monotone counters (`freg+`, `rreg+`, `rdreg+`, `preg+`) that only hand out fresh register IDs — no liveness, no reuse.
- Works for PTX because `ptxas` does the real allocation afterward. Works for tiny SASS programs (vector-add fits). Does **not** work for full-layer kernels — DeltaNet has 22+ ops with hundreds of live values and will exhaust Hopper's 255-per-thread budget (and wreck occupancy long before that).
- Needed: linear-scan allocator over the existing integer IDs, with a liveness pass. Graph coloring is overkill for the straight-line-plus-short-loops shape of these kernels.

### 1c. Backend dispatch coverage in parser.fs
- The `li-backend @ if` selector at `parser.fs:686` cleanly routes per-op to either PTX text or SASS binary.
- **Already routed:** arithmetic ops (`+`, `-`, `*`, `/`, `fma`), for-loops, `setp`/predication, branches.
- **Not yet routed:** shared memory (LDS/STS), warp shuffles (SHFL.BFLY is in the SASS emitter but parser still emits PTX), type conversions, bounds checks, barriers.
- Each unrouted op keeps the PTX path alive. Closing these is grunt work, not research.

### 1d. Instruction scheduling
- SASS emitter uses hardcoded control-word stall/barrier values derived from `tools/sass_probe.py` probing. Correct for simple LDG/STG patterns, not tuneable for deep pipelining or latency hiding.
- Acceptable for first working kernel. A real latency-aware scheduler is a later pass.

### 1e. Cooperative grid-sync primitives — **HIGH PRIORITY**
- Each of the two cubins is a cooperative megakernel: the layer loop runs inside the kernel, with grid-wide sync between layers. That means the SASS emitter needs first-class cooperative-launch support:
  - Grid-wide barrier (cooperative groups `this_grid().sync()` equivalent at SASS level — on Hopper this is a combination of release/acquire fences and a barrier through a global counter).
  - Proper encoding for kernels launched via `cuLaunchCooperativeKernel` (cubin must advertise `EIATTR_COOPERATIVE_GROUP_INSTR_OFFSETS` in `.nv.info.<kernel>`).
  - Block-level `BAR.SYNC` patterns tuned for the grid-sync frequency (48 syncs per DeltaNet cubin invocation).
- The PTX-based attempts (`forward_pass_carmack`, `_cray`, `_bellard`) hit ~24ms GPU time floor because ptxas was choosing barrier/sync patterns. With direct SASS control (stall counts, write/read barrier slots, reuse flags in control words) there's real headroom — that's the whole reason for the SASS pipeline.
- Language-level support: a construct in `.li` to express "run this body N times with grid sync between iterations" so the layer loop compiles down to inline layer N, `grid_sync`, inline layer N+1, etc. — no function call, one code block, grid sync between them.

---

## 1b½. Kernel-building (the factory itself)

The word "factory" has been loose. Today the compiler can turn a hand-written `.li` file into a SASS cubin. It cannot yet *synthesize* the `.li` for a specific model from that model's architecture metadata. That synthesis step is what makes this a factory rather than a one-off compiler.

### Missing: compile-time parameterization

The `.li` language needs compile-time parameters: values known at parse time, substituted into expressions before SASS emission. Something like:

```
param hidden_dim : u32 = compile_time
param n_heads    : u32 = compile_time
param head_dim   : u32 = compile_time

fn deltanet_layer ...
  each i from 0 to hidden_dim
    ...
```

The compiler reads the safetensors JSON header (plain JSON, sub-ms) to get `hidden_dim=5120`, `n_heads=64`, etc., binds those to the compile-time parameters, then parses the `.li` template. The SASS emitter sees resolved constants and doesn't know the difference between a literal `5120` and a resolved `hidden_dim`. Frontend change only; backend untouched.

### Missing: template selection

Different model families (DeltaNet-hybrid, pure-attention, MoE, state-space) want different templates. A small dispatch keyed on architectural fingerprint from the safetensors header picks the right pair of templates (or triplet for MoE) to compile.

### Missing: quantization parameterization

The GEMV template should parameterize over `bits_per_weight`, `group_size`, and the dequant layout (GPTQ zero-point vs NF4 codebook vs the Shannon/Turing schemes in `quantization/`). Template body is a single GEMV shape; dequant inline varies by scheme.

### Scope

This is frontend work — parse-time parameter resolution, a small metadata reader, a template-family dispatch. No new SASS emission, no change to the register allocator, no change to cubin wrapping. Bounded in scope but load-bearing: without it, the compiler works for exactly one hand-written model and nothing else.

---

## 2. Language frontend (`.li` gaps)

Covered today: 5 primitives (`+ − × ÷ √`), hardware intrinsics (`exp`, `rcp`, `rsqrt`, `sqrt`, `sin`, `cos`, `fma`, `neg`), bitwise ops (shift/and/or/xor), warp shuffles, shared memory decl, scalar params, predication, type conversions, bounds checks, barriers, `each` (parallel) and `for` (sequential).

### 2a. Computed-offset array indexing — **blocks Conv1D**
- `inference/recur.li` has a stub comment: *"Hand-unrolled in PTX below; Lithos lacks computed-offset indexing"*.
- Needed: `a[i + k]` or `a[expr]` where `expr` is not a literal. Parser-level change.

### 2b. Outer-product / rank-1 update syntax
- `delta_update.li` and `recur.li` nest `for` loops to express `S += β·(V⊗K − S·(K⊗K))`.
- Nice-to-have: native syntax for `x ⊗ y → M`. Not blocking — loops work.

### 2c. `max` intrinsic
- Online softmax in `attend.li` uses `setp.ge` + predicated branch to get max. FMNMX opcode is already in the SASS emitter — just needs a language-level `max` that maps to it.

### 2d. Intrinsic philosophy
- VOCABULARY.md describes `exp`/`sigmoid`/etc. as decomposable into primitives via Taylor / Newton iteration. Reality: `.li` files call `exp`, `rcp`, `rsqrt`, `sin`, `cos`, `fma` as hardware intrinsics and none of them are decomposed.
- This is fine and intentional — intrinsics map to Hopper `MUFU.EX2 / MUFU.RCP / MUFU.RSQ / MUFU.SIN / MUFU.COS` and `FFMA` which are already opcodes in the SASS emitter. The "5 primitive" story is conceptual; the real primitive set is `+ − × ÷ √ exp rcp rsqrt sin cos fma`. Document this and move on.

---

## 3. Layer kernels (`.li` source)

Status from audit of `inference/*.li` (833 lines across 12 files):

### 3a. Attention layer — **~95% complete**
In one file, `inference/attend.li`:
- RoPE rotation (38 lines, uses `sin`/`cos`)
- Attention scores with online softmax (116 lines, causal mask, GQA-aware, warp-reduce dot product, inline weighted sum)
- Reuses `reduce.li` RMSNorm and `gemv.li` GPTQ GEMV for projections

Missing: none critical. It composes into one kernel once the backend can emit it.

### 3b. DeltaNet layer — **~85% complete**
- RMSNorm ✓ (`reduce.li`)
- Q/K/V/gate/beta projections ✓ (`gemv.li` GPTQ W4A16)
- Conv1D 4-tap ✗ *stub* — blocked on 2a (computed-offset indexing)
- Decay gate ✓ (`decay_gate.li`)
- Gate sigmoid ✓ (`recur.li`)
- Delta rule / state update ~partial (`delta_update.li` rank-1 update; `recur.li` has a simplified version) — outer product works via nested loops but is verbose
- L2Norm on Q/K ✓ (`reduce.li`)
- Residual add ✓ (`elementwise.li`)
- Output projection ✓ (via `gemv.li`)
- SiLU ✓ (`elementwise.li`)

Missing: Conv1D (language-blocked). Delta rule cleanup is polish, not blocking.

### 3c. MLP / embed / lm_head / sampling
- SiLU, residual add, GEMV individually present. No fused MLP kernel composed yet.
- Embed (token ID → vector lookup): trivial, not yet written in `.li`.
- lm_head (final GEMV to vocab): reuses `gemv.li` shape, may need FP16 variant.
- Sampling (argmax / top-k): stub in `reduce.li:209-217`.

Decision point: these fold into the launcher (few lines of C/Lithos each) or into one of the two cubins. Probably cleanest in the launcher since they run once per token, not per-layer.

---

## 4. Native launcher + libcuda replacement

Current Python dispatch is 97% of wall-clock time per STATUS.md — that goes away entirely. But the scope is now larger than just "replace Python": the empirical strace work shows we can also replace `libcuda.so` itself with a few hundred lines of Lithos, because the hot path doesn't go through the kernel at all.

### 4.0 What libcuda actually does (empirical)

Captured via `strace` on a minimal CUDA program on GH200 (driver 580.105.08, CUDA 12.8). Full trace at `docs/libcuda_ioctl_trace.md`. Three findings change the scope:

**Finding 1 — the hot path is zero-syscall.** `cuModuleLoadData`, `cuLaunchKernel`, `cuCtxSynchronize` fire **0 ioctls, 0 mmaps** during kernel dispatch. All GPU work is CPU-side memory writes:
- Build a **QMD** (Queue Memory Descriptor: entry PC, register count, shared-mem bytes, param block, grid/block dims) in a mapped USERD page.
- Write to a mapped doorbell register.
- That's the entire launch path.

**Finding 2 — setup tax is in `cuCtxCreate`, not `cuInit`.** Context creation is 449 ioctls + 18 mmaps. This is the real one-time cost. Our replacement eats it once at startup, then the per-token loop is pure userspace memory stores.

**Finding 3 — total ioctl surface is 25 unique calls.** 11 on `/dev/nvidiactl`, 14 on `/dev/nvidia-uvm`. Finite. Enumerable. All documented in `open-gpu-kernel-modules`.

### 4.1 Replacement path (Level 1 of the size analysis)

| Piece | Approach | Size |
|---|---|---|
| `cuInit` + `cuCtxCreate` equivalent | ~450 ioctls in right sequence, mostly `RM_ALLOC` / `UVM_REGISTER_GPU` shaped. Mechanical once ABI doc is in hand. | ~few hundred lines of Lithos |
| `cuModuleLoadData` equivalent | Allocate GPU memory, copy cubin SASS bytes, record kernel metadata (entry PC, reg count, shmem, param layout) that we already know from compile | ~50 lines |
| `cuModuleGetFunction` equivalent | Zero work — we emitted the cubin, we already have every field libcuda would otherwise parse out | (eliminated) |
| `cuLaunchKernel` equivalent | Pack QMD into mapped USERD page, store doorbell. Pure memory writes, no syscalls. | ~50 lines |
| `cuCtxSynchronize` equivalent | Poll a 32-bit fence value in mapped GPU-visible memory | ~10 lines |
| `cuMemAlloc` / `cuMemcpy*` | Not needed on GH200 — `mmap` of safetensors + NVLink-C2C makes host VA directly GPU-visible (confirmed empirically, zero ioctls for `cuMemcpyHtoD` on a host-mapped buffer) | (eliminated) |
| **Total** | | **~few hundred lines of Lithos** |

Replaces ~80 MB of `libcuda.so`. Binary size impact: negligible (Lithos words compile to tight ARM64).

### 4.2 What's still needed to make this work

- **Lithos syscall wrappers** — `openat`, `close`, `ioctl`, `mmap`, `munmap` as Lithos words callable from the compiler/launcher. Exist as inline ARM64 in `src/launcher.s` but not as first-class Lithos words in `forth-bootstrap.s`. FFI survey estimate: ~1 day to lift them.
- **Struct marshaling helpers** — generalized `put-u32 @ offset`, `put-u64 @ offset`, field-packing for the NVIDIA RM ioctl structs. Pattern exists in `gpu/emit.fs` for ELF headers; generalize it.
- **ABI reference** — precise struct layouts and ioctl numbers for all 25 calls, from `open-gpu-kernel-modules` source. (Research worker in flight.)
- **QMD encoder** — the Queue Memory Descriptor format for Hopper compute launches. Documented in open-gpu-kernel-modules under class `HOPPER_COMPUTE_A` / `HOPPER_COMPUTE_B`. Fields: entry PC, register count, shared mem, param block base + size, grid/block dims, barrier config, constant bank bindings. ~128 bytes.

### 4.3 What the native binary does (revised)

```
main:
  open /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm
  cuInit-equivalent: ~20 ioctls
  cuCtxCreate-equivalent: ~450 ioctls, mmap USERD + GPFIFO + BAR1 + semaphore pool
  mmap safetensors (weights, 18 GB, zero-copy on GH200)
  lithos-compile deltanet_layer.li  → SASS bytes + metadata in memory
  lithos-compile attention_layer.li → SASS bytes + metadata in memory
  for each kernel:
    copy SASS into GPU instruction memory (direct write to mapped BAR region)
    record kernel metadata for QMD building

  per token:
    pack QMD for deltanet kernel into USERD, doorbell
    poll fence until layer completes
    pack QMD for attention kernel into USERD (when layer is attention), doorbell
    poll fence
    ... (grid-sync happens inside the cubin, not the launcher)
    sample token, emit
```

No libcuda, no driver userspace beyond the open kernel module. Pure Lithos → syscalls → mapped-memory writes.

### 4a. Launcher skeleton
- Load cubins via `cuModuleLoadData` / `cuModuleGetFunction` (libcuda.so direct).
- Per-token loop: embed → 64× (dispatch deltanet OR attention cubin based on layer index) → final norm → lm_head → sample.
- ~200 lines of Lithos, compiles through S3-linux to ARM64 ELF.

### 4b. Weight residency
- mmap safetensors already works (`src/loader.py` Python → to be ported). 18.21 GB of GPTQ weights map in 9ms on GH200 unified memory. No cudaMalloc, no cudaMemcpy.
- The launcher needs: mmap the safetensors, build a weight-pointer table, pass it to each kernel launch.

### 4c. State management
- DeltaNet state: 16 heads × 128 × 128 × f32 × 48 layers = ~12.6 MB fixed. Allocate once at startup, reuse every token.
- KV cache for attention layers: grows per token. 16 layers × head_dim × n_kv_heads × 2 (K,V). Pre-allocate max sequence length or grow in chunks.
- RoPE cos/sin tables: precompute and pin.

### 4d. Streams and async
- First working launcher: single stream, synchronous per-kernel launch. That's already a huge win over Python.
- Later: async launches, but without CUDA graphs. Hopper's launch overhead is nanoseconds when called from native code.

---

## 5. Validation

### 5a. Per-kernel correctness
- Each compiled cubin must produce bit-identical output to a PyTorch reference for the same inputs. Same kernel, same weights, same inputs → same outputs.
- Tests belong next to each `.li` — pattern: `compiler/examples/test_<name>_load.py` checks loadability, `compiler/examples/test_<name>_exec.py` compares to reference.

### 5b. End-to-end verification
- The known-good target: "The capital of France is" → top-1 prediction `"Paris"` (logit 18.057, cosine 1.0 against PyTorch at every one of 64 layers). This is already proven with the current Python pipeline per STATUS.md:937.
- Same test must pass once two Lithos-compiled cubins replace the 21 hand-written PTX kernels.

### 5c. No perf numbers until correct
- Do not report tok/s numbers until end-to-end cosine = 1.0 against PyTorch at every layer. "Fast but wrong" numbers have burned us before.

---

## 6. Order of operations

Ordered by "blocks the most downstream work":

1. Cubin ELF wrapping *(1a — in flight)*
2. Register allocator *(1b)*
3. Cooperative grid-sync primitives *(1e)* — both at SASS encoding level and as a `.li` language construct
4. Backend dispatch closure *(1c)*
5. Language frontend — computed-offset indexing *(2a)* *(unblocks Conv1D)*
6. Compile the DeltaNet cubin (48-layer cooperative megakernel) and verify per-layer cosine vs PyTorch
7. Compile the attention cubin (16-layer cooperative megakernel) and verify per-layer cosine vs PyTorch
8. Lithos launcher *(section 4)*
9. End-to-end "Paris" test — bit-identical top-1 token + cosine 1.0 at every layer
10. Polish: outer-product syntax *(2b)*, `max` intrinsic *(2c)*, scheduler *(1d)*

Items 1, 4, and 5 are parallel-safe (different files). Items 2 and 4 serialize on `parser.fs`. Item 3 touches both `gpu/emit.fs` and `parser.fs`/language spec.
