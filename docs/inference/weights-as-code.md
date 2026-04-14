# Weights-as-Code Inference Architecture

Engineering analysis: can Q4 model weights be encoded as immediates in GPU
instructions, eliminating weight loads from HBM?

**Verdict: SIMT breaks the naive model. A workable hybrid exists but trades
simplicity for complexity. The honest analysis follows.**

---

## 1. What Execution Looks Like Per Token (the vision)

The current engine (`src/engine.py`) orchestrates inference as a sequence of
kernel launches: for each of the 64 layers, it launches separate kernels for
RMSNorm, GEMV projection (Q/K/V/gate/up/down), attention or DeltaNet
recurrence, SiLU activation, residual add, and so on. Each GEMV kernel
(`inference/gemv.ls`) receives a pointer to packed Q4 weights in HBM and
streams them through the LSU.

The weights-as-code vision replaces this with:

1. The compiler reads safetensors at compile time.
2. For each weight element, it emits an FMUL or FFMA instruction with the
   (dequantized) weight baked into a 32-bit immediate field.
3. At inference time, the SM executes these instructions. Weights arrive
   through the instruction fetch pipeline, not the data load pipeline.
4. Between projections, elementwise ops (SiLU, sigmoid, residual add,
   RMSNorm) run as short instruction sequences -- these are tiny compared
   to the ~26M multiply-accumulates per projection.
5. Grid-sync separates layers within two cooperative megakernels.
6. Total instruction stream per token = 48 DeltaNet layers + 16
   full-attention layers, each with ~7 projections + elementwise/reduction
   work.

---

## 2. Memory Layout at Inference Time

### What lives in HBM

| Allocation | Size (Qwen 3.5-27B) | Notes |
|---|---|---|
| ELF .text (weight-instruction stream) | ~28-55 GB (see calculation below) | The "weights as code" |
| Activation double-buffer | ~20 KB | 2 x 5120 x bf16 |
| KV cache (16 full-attn layers) | ~512 MB at 32K seq | 16 layers x 32K x 4 heads x 128 dim x 2 x 2B |
| DeltaNet state matrices (48 layers) | ~2.4 GB | 48 x 28 heads x 128 x 128 x 2B |
| Residual accumulator | ~10 KB | 5120 x bf16 |
| Embedding table | ~2.4 GB | 248320 x 5120 x bf16 (not weight-as-code) |

### What disappears

Weight tensors as separate data allocations. No more `W_packed`, `scales`,
`zeros` buffers passed as kernel parameters. The weights are literally inside
the instruction stream.

### Does this save HBM? No.

Qwen 3.5-27B at Q4 has roughly 27 billion parameters. At 4 bits per weight,
that is:

    27e9 / 2 = 13.5 GB raw weight data

The weight-as-code instruction stream encodes one weight per instruction.
Each Hopper binary instruction is 16 bytes. At 8 weights unpacked from each 32-bit
packed word in the current GEMV, we need one FMUL-IMM per weight element:

    27e9 weights x 16 bytes/instruction = 432 GB

This is absurd. Even if we pack 2 FMUL-IMMs per 128-bit instruction slot
(which sm90 does not support), it is still 216 GB.

The realistic encoding: one FFMA instruction with a dequantized fp32
immediate per weight element. 16 bytes per instruction, one weight per
instruction. The instruction stream is **32x larger** than the packed Q4
weight data.

**There is no HBM savings. The instruction stream is dramatically larger
than the raw weights.**

The argument for the architecture is not about saving memory. It is about
the ACCESS PATTERN: instruction fetch vs data load use different hardware
paths.

---

## 3. Instruction Cache Behavior

### sm90 instruction cache hierarchy

| Level | Scope | Size | Line width |
|---|---|---|---|
| L0I | Per-SM, per-subpartition | ~16 KB (estimated) | 128 bytes |
| L1I | Per-SM | ~128 KB | 128 bytes |
| L2 | Shared across all SMs | 50 MB (GH200) | 128 bytes |

### Does the instruction stream fit?

A single projection: 3584 output x 3584 input = ~12.8M weights.
At 16 bytes per instruction: **~205 MB per projection.**

For the smallest projections (head projections), the numbers are smaller,
but even a single Q/K/V combined projection for one layer far exceeds L2.

The instruction stream does NOT fit in any cache level. The I-fetch unit
will stream continuously from L2, and L2 will stream from HBM. This is
functionally identical to data streaming from HBM -- the instruction fetch
unit becomes a second DMA engine pulling weight data.

### Separate ports or shared bandwidth?

The key question: does instruction fetch share L2 bandwidth with data loads?

On sm90 (Hopper):

- **L1I and L1D are physically separate caches** with separate ports to the SM.
- **L2 is unified.** Both instruction fetch misses and data load misses
  compete for L2 bandwidth. The L2 has a crossbar with multiple ports, but
  total bandwidth is finite (~5 TB/s on GH200).
- **HBM bandwidth is fully shared.** There is one HBM interface. Instruction
  fetch and data loads both consume the same ~4 TB/s (GH200) of HBM bandwidth.

**Therefore: at the L2 and HBM level, instruction fetch and data loads
compete for the same bandwidth.** The claim that "weight-fetch and
activation-fetch can happen simultaneously without contention" is FALSE at
the memory controller level.

The separation exists only at L1: the L1I path and L1D path are independent
within an SM. But for a streaming workload that blows through all cache
levels, the bottleneck is HBM bandwidth, which is shared.

### Comparison: LDG path vs I-fetch path

| Property | LDG (current) | I-fetch (weights-as-code) |
|---|---|---|
| SM unit | LSU (Load Store Unit) | I-fetch unit |
| L1 cache | L1D (~256 KB, configurable) | L1I (~128 KB) |
| L1 → L2 port | Data port | Instruction port |
| L2 → HBM | Shared HBM controller | Same shared HBM controller |
| Bandwidth at HBM | 4 TB/s shared | 4 TB/s shared |
| Data per operation | 4 bits (packed Q4) | 128 bits (16-byte instruction) |

The LDG path is **32x more efficient per byte of HBM bandwidth consumed**
because it reads packed Q4 data (4 bits per weight) rather than 128-bit
instructions (one weight per instruction). Weights-as-code turns a memory
bandwidth problem into a WORSE memory bandwidth problem.

---

## 4. The Activation Register Problem

For a GEMV with weights-as-immediates, the inner loop for one output
element looks like:

    acc = 0.0
    for i in 0..5119:
        acc += activation[i] * WEIGHT_IMMEDIATE_i   // FFMA with baked weight

This requires:
- 1 accumulator register (f32)
- 1 temporary for the loaded activation element
- 1 or 2 address registers

Total: 3-4 registers per output element. Register pressure is not the
constraint.

The activation vector (3584 or 5120 elements depending on the projection)
must be loaded from HBM once per layer, then each element accessed
sequentially. At 4 bytes x 5120 = 20 KB, this fits in shared memory. Each
thread reads `activation[i]` from smem as it executes `FFMA_IMM_i`.

The register problem is solved. The activation vector lives in shared
memory, the accumulator lives in a register, and the weight is in the
instruction immediate.

---

## 5. Thread Mapping

### Current GEMV (gemv.ls)

The current kernel launches with `gridDim.x = N` (one block per output row),
`blockDim.x = 256`. Each block of 256 threads cooperates on ONE output
element. Each thread strides over K/8 packed weight words, accumulates a
partial sum, then warp-shuffles + smem reduce to get the final result.

The K-dimension is split across 256 threads: each thread handles
ceil(K_packed / 256) packed words. This is inner-dimension parallelism.

### Weights-as-code threading

If weights are in instruction immediates, all threads execute the same
instruction stream (SIMT). Consider two options:

**Option A: Each thread computes one output element independently.**
Each thread streams through all 5120 FFMA-IMM instructions, reading
activations from shared memory. A full warp of 32 threads computes 32
output elements simultaneously, all executing the same FFMA-IMM at each
cycle.

Problem: see Section 6.

**Option B: Threads split the inner dimension as before.**
256 threads split the 5120-element inner dimension into ~20 elements each.
But each thread needs DIFFERENT weight immediates (different portions of
the same row). Since all threads in a warp execute the SAME instruction
(same immediate), this does not work.

---

## 6. The SIMT Constraint -- Critical Analysis

**This is the showstopper.**

In NVIDIA's SIMT model, all 32 threads in a warp execute the same
instruction at the same time. If the weight is an instruction immediate,
every thread in the warp multiplies by the SAME weight value at each step.

### What this means for GEMV

A GEMV computes `y[j] = sum_i(W[j,i] * x[i])` for each output element j.

If 32 threads in a warp compute 32 different output elements (j=0..31),
they each need a different row of W. At step i, thread 0 needs W[0,i],
thread 1 needs W[1,i], etc. But if W[j,i] is the instruction immediate,
all 32 threads see the SAME immediate W[?,i]. They cannot each have
different weights.

### What actually happens with SIMT + immediates

All 32 threads execute: `acc += smem_activation[i] * IMMEDIATE_WEIGHT`

- `smem_activation[i]` is the same for all threads (same shared memory
  address, same value)
- `IMMEDIATE_WEIGHT` is the same for all threads (it is in the instruction)
- `acc` is per-thread (register), but will contain the SAME value

**Result: all 32 threads compute the SAME output element.** A warp of 32
threads doing 5120 FFMA-IMMs produces 1 output element, not 32. This is
a 32x reduction in parallelism compared to the current GEMV.

### Could threads read DIFFERENT activation elements?

If each thread reads a different `smem_activation[lane_id + offset]`,
then at each step:

    thread_k: acc_k += activation[k + offset] * W_immediate

All threads multiply a DIFFERENT activation element by the SAME weight.
This is an outer-product decomposition: each instruction step computes one
column of the outer product `w_i * x^T`, where w_i is the weight immediate
and each thread holds a different element of x.

But this computes `y[j] = sum_i(W_col_immediate_i * x[j])` -- every output
element gets multiplied by the same weight column. This is NOT a GEMV. A
GEMV needs each output to use a DIFFERENT row of W.

---

## 7. The Resolution: What Actually Works

### 7a. One output per warp, serialized (works but slow)

Each warp computes one output element by streaming through the full
weight-immediate sequence (one row of W encoded as 5120 FFMA-IMMs). To
compute all N output elements, we serialize: the first pass computes
output[0], the second output[1], etc.

For the GH200 with 132 SMs x 4 warp schedulers = 528 concurrent warps:
- 5120-dim output: 5120 / 528 = ~10 serial passes
- Each pass: 5120 FFMA-IMMs = 5120 cycles at best

This means: ~10 passes x 5120 cycles = ~51,200 cycles per projection.
Current GEMV: ~5120 / 256 threads x reduction = ~20 cycles per element
with full occupancy. The weights-as-code path is orders of magnitude
slower.

### 7b. Warp-divergent immediates via shuffle (does not exist)

There is no mechanism in the Hopper ISA to have per-lane immediates. The immediate
field is part of the instruction encoding, shared by all lanes. SHFL can
distribute a register value across lanes, but the value must come from a
register in the first place -- defeating the purpose of having weights in
immediates.

### 7c. Outer-product form (mathematically valid, architecturally expensive)

Reformulate the GEMV as a sum of outer products:

    Y = sum_i( W_col_i * x_i )

where W_col_i is column i of W (a vector of length N) and x_i is a scalar.

Each instruction step broadcasts one weight column element as an immediate,
and each thread multiplies it by a different x element loaded from smem.
But wait -- this requires N different weight values per column (one per
output element), and all threads see the same immediate.

The outer product form does not resolve the SIMT constraint either.

### 7d. The actual workable architecture: weight registers, not immediates

Instead of encoding weights as instruction immediates, the compiler
generates a load-compute sequence:

    LDG.128 R4, [weight_ptr + offset]    // load 4 weights (16 bytes of Q4 packed)
    // dequantize in registers
    FFMA R0, R4, R8, R0                   // acc += weight * activation

This is... exactly what the current GEMV does. The "compiler bakes weights"
concept dissolves into "the compiler generates the exact same load-compute
pattern but with hardcoded addresses instead of parameterized ones."

### 7e. The one place weights-as-immediates DOES work: scalar operations

For elementwise operations where every thread computes the same function
with the same learned parameter, immediates work perfectly:

- **RMSNorm scale weights**: `y[i] = normalized[i] * weight[i]` -- if
  each thread handles one element and the weight is baked into the FMUL,
  this works because each thread needs a DIFFERENT weight (different i)
  and can execute a different instruction... except SIMT still requires
  all threads in a warp to execute the SAME instruction.

Even for elementwise ops, SIMT means all 32 threads in a warp apply the
SAME learned weight to 32 different positions. The weight is correct for
only one of those positions.

**SIMT makes per-element learned constants as immediates impossible for
ANY operation where different lanes need different constants.**

### 7f. Warp-sequential execution (the nuclear option)

Run one thread per warp (mask off lanes 1-31). Each surviving thread
executes its own instruction stream with its own weight immediates.
128 SMs x 4 schedulers x 1 thread = 512 independent execution streams.

This throws away 31/32 = 97% of the GPU's compute capacity. A 132-SM
GH200 would compute at the effective rate of ~4 SMs worth of single-lane
execution. This is roughly 500x slower than the current architecture.

---

## 8. Honest Verdict

**Weights-as-instruction-immediates is fundamentally incompatible with
SIMT execution for matrix-vector multiplication.**

The core constraint:
- GEMV requires different output elements to use different weight rows.
- SIMT requires all threads in a warp to execute the same instruction
  (same immediate).
- These two requirements are contradictory.

No amount of clever encoding resolves this. It is not an engineering
difficulty -- it is a structural impossibility given NVIDIA's execution
model. The immediate field of an instruction is architecturally shared
across all 32 lanes of a warp.

### What weights-as-code CAN mean (reframed)

The useful interpretation of "weights are code" is not "weights are
instruction immediates." It is:

1. **The compiler hardcodes weight addresses and strides** into the
   instruction stream, eliminating pointer arithmetic and parameter
   passing at runtime. The GEMV loop body becomes a fixed sequence of
   LDG instructions with compile-time-resolved addresses, followed by
   FFMA. No `W_packed` pointer, no `scales` pointer, no `K_packed`
   parameter. Just raw addresses baked into load instructions.

2. **The compiler fuses the entire layer** (norm + projection + activation
   + residual) into one instruction sequence with zero kernel launch
   overhead. Weight addresses, activation buffer addresses, and norm
   weights are all known at compile time and embedded in the code.

3. **Dequantization is compile-time**. The compiler reads Q4 packed
   weights, dequantizes them, and emits the dequantized values directly
   as data loads from a `.rodata`-like section. The runtime never touches
   scale factors or zero points.

This is the architecture Lithos already pursues with the megakernel
approach: two cooperative grid-sync megakernels per model, with all
addresses resolved at compile time, no parameter buffers, no kernel
launch overhead. The weights remain data (loaded via LDG from HBM), but
the entire execution plan -- which weights to load, when, in what order,
with what dequantization -- is compiled into the instruction stream.

### Performance comparison

| Architecture | Bytes from HBM per weight | Instruction bytes per weight | Total HBM traffic per weight |
|---|---|---|---|
| Current GEMV (Q4) | 0.5 B (4 bits packed) + ~0.03 B (scale) | ~2 B (shared GEMV loop) | ~0.53 B |
| Weights-as-immediates | 0 B (no data load) | 16 B (one FFMA-IMM instruction) | 16 B |
| Compiled megakernel (realistic) | 0.5 B (Q4 from .rodata, hardcoded addr) | ~4 B (unrolled LDG+FFMA) | ~4.5 B |

The weights-as-immediates path uses **30x more HBM bandwidth** than the
current Q4 GEMV. The compiled megakernel with hardcoded addresses is a
reasonable middle ground: slightly more instruction traffic than a generic
GEMV loop (due to unrolling), but with zero parameter overhead and zero
launch overhead.

---

## 9. What Changes in the Lithos Execution Path

Given this analysis, the weights-as-code architecture for Lithos means:

### Per token, the execution is:

1. **Megakernel launch** (one SEND_PCAS_A via QMD, 528-byte descriptor).
2. All 132 SMs begin executing layer 0.
3. For each projection (Q, K, V, gate, up, down, output):
   - Threads load activation vector into shared memory (one cooperative load).
   - Threads execute an **unrolled GEMV loop** where weight addresses are
     hardcoded constants in the LDG instructions. No pointer parameters.
   - Dequantization is baked in: LDG loads pre-resolved Q4 packed words,
     shift/mask/scale sequences use compile-time constants.
   - Warp shuffle + smem reduction produces output elements.
4. Between projections: short instruction sequences for RMSNorm, SiLU,
   residual add (matching the 71-step decomposition from `ops/deltanet/inference/STEPS`).
5. Grid-sync (`bar.sync` across all CTAs) separates layers.
6. After 64 layers: final norm, lm_head projection, argmax.

### Memory layout:

- **HBM**: activation buffers, KV cache (16 layers), DeltaNet state (48 layers),
  weight data in `.rodata` section of ELF (Q4 packed, ~13.5 GB), instruction
  stream in `.text` (~200 MB for the megakernel code itself).
- **No separate weight allocations**: weights are part of the ELF binary.
  The `LoadedModel.weight_offsets` dictionary is eliminated. The compiler
  resolves all offsets at compile time.
- **No `LoadedKernels` dispatch table**: there is one megakernel. No
  `projection_func`, `norm_func`, `attention_score_func` -- just one
  function pointer.

### What the compiler does:

1. Reads safetensors (or GGUF) at compile time.
2. Emits Q4 weight data into `.rodata` section with known GPU VA offsets.
3. For each layer, emits the complete instruction sequence:
   - LDG instructions with hardcoded `.rodata` addresses for weight loads.
   - Dequant sequences with compile-time scale factors.
   - FFMA sequences for accumulation.
   - Reduction sequences (shuffle + smem).
   - Elementwise ops (SiLU, sigmoid, norm) with hardcoded norm-weight addresses.
   - Grid-sync barriers between layers.
4. Outputs a single ELF binary containing both `.text` and `.rodata`.

### What disappears:

- `engine.py` (the Python orchestrator)
- `LoadedModel`, `LoadedKernels`, `ActivationBuffers` classes
- Per-kernel launch overhead (QMD construction per kernel)
- Weight pointer passing via cbuf0
- Runtime dequantization parameter setup

### What remains:

- HBM bandwidth as the bottleneck (weights still come from HBM, just via
  LDG with hardcoded addresses instead of parameterized pointers)
- The 71-step decomposition per DeltaNet layer (unchanged)
- The DeltaNet state update loop (128x128 state matrix, FP32, same as
  `deltanet_fused.ls`)
- KV cache management for the 16 full-attention layers

---

## 10. Summary

The literal "weights as instruction immediates" idea fails due to SIMT.
All 32 warp lanes execute the same instruction with the same immediate,
making it impossible for different lanes to apply different weights. This
is not a solvable engineering problem -- it is how the hardware works.

The productive version of "weights are code" is: **the compiler resolves
all weight addresses at compile time and emits a single monolithic
instruction stream that contains the complete inference pass.** Weights
remain data loaded via LDG, but the addresses, strides, dequantization
constants, and execution schedule are all baked into the code. No runtime
orchestration, no parameter passing, no kernel launches beyond the initial
megakernel dispatch.

This is what Lithos should build.
