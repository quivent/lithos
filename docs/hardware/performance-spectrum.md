# Performance Spectrum: Full Compilation Stack Analysis

*Speculative analysis — not measured results. Based on GH200 hardware specs,
Qwen 3.5 27B W4A16 model dimensions, and known GPU performance characteristics.
No numbers here should be cited as benchmarks until verified on hardware.*

---

## The Ceiling

Before any tier is discussed, the absolute ceiling must be established.

**GH200 HBM3e bandwidth:** 3.35 TB/s
**Qwen 3.5 27B W4A16 weight size:** ~14 GB (27B params × 4 bits + scales)
**Theoretical maximum tokens/second at batch=1:**

```
3,350 GB/s ÷ 14 GB = ~240 tok/s
```

This is a hard physical limit. At batch=1, autoregressive decode reads every weight
once per token. No software path can exceed 240 tok/s on this hardware with this model
at batch=1, because reading 14GB from HBM takes at least 4.2ms regardless of how the
reads are issued.

Every tier below is measured against this ceiling.

**Arithmetic intensity at batch=1:**
Each weight (4-bit) produces one multiply-accumulate with one activation element.
That is ~2 FLOPs per 0.5 bytes of weight = ~4 FLOP/byte.
GH200 ridge point (FP32): 67 TFLOPS ÷ 3.35 TB/s ≈ 20 FLOP/byte.
At batch=1, inference is firmly **memory-bandwidth-bound at every tier**.
Compute throughput is irrelevant until batch ≥ ~5 on this model.

---

## The Spectrum

### Tier 1: Python + PyTorch + vLLM + CUDA

**Stack:** Python inference loop → vLLM engine → PyTorch ops → CUDA runtime →
ptxas-compiled kernels → SASS → opcode.

**Estimated tok/s:** 40–70 (batch=1) · 250–450 (batch=8)
**Estimated ms/token:** 14–25ms
**Bandwidth utilization:** 15–30% of peak
**Kernel launches per token:** 200–500 (per-layer embed, norm, qkv, attn, ffn × 64 layers)
**Occupancy:** 25–50% (varies by kernel type)

**What dominates the loss:**

Python has a floor. Even with CUDA graphs amortizing kernel launch overhead,
the Python interpreter must prepare inputs, run graph replay, and handle outputs.
At small batch sizes this overhead is 5–15ms per token — more than the GPU
computation itself. The GPU sits idle while Python runs.

The kernel launch overhead (200–500 launches × ~5µs each = 1–2.5ms) is
secondary to Python but significant. Generic cuBLAS kernels are tuned for
large matrix multiplications, not for the 5120×128-vector GEMVs that
dominate single-token decode. They use more registers than necessary and
achieve 30–50% of peak bandwidth on these shapes.

The 48 DeltaNet layers each trigger separate kernel launches for:
Q/K/V/gate/beta/decay projections, delta state update, output projection, RMSNorm,
residual add, MLP gate+up projection, SiLU, elementwise multiply, MLP down projection.
That is ~15 kernel launches per DeltaNet layer × 48 layers = 720 launches
before counting the 16 full attention layers. In practice vLLM fuses some of
these, reducing the count to 200–400.

**Batch scaling:** Batching helps dramatically here because it amortizes the Python
overhead across multiple token streams. At batch=8, Python overhead is roughly
constant while GPU work scales linearly, so bandwidth utilization improves
substantially. Beyond batch=16 on this model, the memory bandwidth becomes the
dominant constraint and gains taper.

---

### Tier 2: C++ → CUDA (remove Python)

**Stack:** C++ launcher → CUDA API → ptxas kernels → SASS → opcode.

**Estimated tok/s:** 70–110 (batch=1) · 500–750 (batch=8)
**Estimated ms/token:** 9–14ms
**Bandwidth utilization:** 25–40% of peak
**Kernel launches per token:** 100–200
**Occupancy:** 25–50%

**What is gained:** The Python dispatch floor (5–15ms) is eliminated. The C++ launcher
calls cuLaunchKernel directly, paying ~1–5µs per launch instead of ~50µs per Python
call. This alone recovers 5–15ms per token, roughly doubling throughput at batch=1.

**What remains:** Generic kernel quality is unchanged. The GEMV kernels are the same
ptxas-compiled binaries with the same occupancy characteristics. The 200-launch waterfall
still exists; C++ just navigates it faster. Bandwidth utilization improves modestly because
the GPU spends less time idle between launches.

The libcuda hot path is already largely userspace memory writes (QMD construction +
doorbell ring). The per-launch overhead in C++ is genuinely low (~1µs). The improvement
over Python is real but most of it comes from removing the Python interpreter,
not from improving the CUDA dispatch path.

---

### Tier 3: C++ → PTX → GPU (hand-tuned PTX, remove ptxas choices)

**Stack:** C++ launcher → hand-written PTX → ptxas → SASS → opcode.

**Estimated tok/s:** 90–140 (batch=1) · 650–950 (batch=8)
**Estimated ms/token:** 7–11ms
**Bandwidth utilization:** 35–55% of peak
**Kernel launches per token:** 50–150 (model-specific kernels fuse more)
**Occupancy:** 40–65%

**What is gained:** Hand-written PTX for the specific projection shapes
(5120×2048, 5120×17408, etc.) eliminates the overhead of general-purpose cuBLAS
routing. The kernels are sized exactly for Qwen 3.5 27B's dimensions. ptxas can
make better register allocation decisions when the input PTX is written with tight
live ranges rather than the bloated intermediate-variable style that C++ compilers
tend to produce.

Kernel fusion becomes practical: the Q/K/V projections for one layer can be
fused into one kernel launch. The 6 DeltaNet projections (Q, K, V, Z, beta, decay)
become 2 or 3 launches instead of 6. This reduces launch overhead and improves cache
utilization (the input activation is read once, used for multiple projections).

**What remains:** ptxas controls the final register assignment. It is conservative —
it targets a wide range of programs and tends to err toward using more registers to
avoid spilling, which reduces occupancy. For the specific shapes and patterns of
these kernels, a human or a model-specific register allocator would make better choices.
The activation loads still go through LDG with all associated latency, requiring
occupancy to hide that latency.

---

### Tier 4: C++ → SASS (direct SASS, remove ptxas)

**Stack:** C++ launcher → hand-emitted SASS → opcode.

**Estimated tok/s:** 120–170 (batch=1) · 800–1100 (batch=8)
**Estimated ms/token:** 5.9–8.3ms
**Bandwidth utilization:** 45–65% of peak
**Kernel launches per token:** 30–80
**Occupancy:** 50–75%

**What is gained:** Full control over register allocation, instruction scheduling,
and control words. A skilled SASS emitter can assign registers to minimize the peak
live set, place reuse hints (control word bits 62:58) for values used in consecutive
instructions, and set stall counts precisely rather than conservatively.

For the GEMV inner loop specifically: the 5-register streaming pattern (pack, scale,
acc, tmp, xval) is achievable only when the allocator knows that tmp and xval are dead
after each FFMA. ptxas, working from PTX with many named intermediate variables, may
assign 10–15 registers to the same loop, reducing occupancy and hiding less latency.

Direct SASS also enables instruction patterns that ptxas does not emit: predicated
prefetches, explicit software pipelining of LDG (issue the load 8 instructions before
it is consumed, hiding latency without extra warp switching), and reuse-cache exploitation
for repeated reads of the same register within a group of instructions.

**What remains:** libcuda dispatch overhead (~1µs per launch) is small but real.
For 50 launches per token it is ~50µs, or about 1% of a 5ms token budget. Not dominant.
The activation loads still use LDG, paying full global memory latency.

---

### Tier 5: Lithos → opcode, naive register allocation

**Stack:** Native ARM64 launcher → Lithos compiler → raw sm_90a binary.
*This is the current state of the Lithos compiler (minus bootstrap completion).*

**Estimated tok/s:** 90–140 (batch=1) · 550–850 (batch=8)
**Estimated ms/token:** 7–11ms
**Bandwidth utilization:** 30–50% of peak
**Kernel launches per token:** **1** (cooperative megakernel)
**Occupancy:** 10–20% (monotone register allocator → 150–255 regs/thread → 8–16 warps/SM)

**The megakernel advantage:** One QMD dispatch per token instead of 200–500.
One doorbell write per token. The ARM64 launcher executes in nanoseconds.
The GPU sees one enormous kernel with 67 internal grid-syncs rather than
500 separate kernel submissions. The scheduler overhead is negligible.

**The occupancy problem:** This is where the naive allocator damages performance.
The monotone bump allocator hands out a fresh register for every named value and
never reuses any. A fully-inlined DeltaNet layer has hundreds of named values:
Q/K/V/gate projections each with their own accumulator, decay gate intermediates,
delta rule loop variables, RMSNorm running sums, MLP gate/up/down intermediates.
With no reuse, the compiler reaches 150–200 registers by the first DeltaNet layer
and exhausts the 255-register budget before the third.

At 150–200 registers per thread, only 8–10 warps can reside on the SM simultaneously.
With only 8 warps to switch between, and LDG latency of 200–400 cycles, the SM
frequently exhausts its warp pool and stalls waiting for memory. Bandwidth
utilization drops to 30–50% of peak despite the hardware being capable of much more.

**A counterintuitive result:** Tier 5 (Lithos megakernel, naive allocation) may
perform similarly to or slightly worse than Tier 3 (C++/PTX, many launches) despite
having a superior dispatch architecture. The megakernel design is sound; the register
allocation is the bottleneck. This is the exact problem the register-packing work addresses.

---

### Tier 6: Lithos → register-packed opcode (the target)

**Stack:** Native ARM64 launcher → Lithos compiler with linear-scan allocator →
raw sm_90a binary with optimal register utilization.

**Estimated tok/s:** 180–230 (batch=1) · 1,000–1,400 (batch=8)
**Estimated ms/token:** 4.3–5.6ms
**Bandwidth utilization:** 75–90% of peak
**Kernel launches per token:** 1
**Occupancy:** 50–100% (depending on regcap target: 64 regs → 32 warps/SM)

**What changes from Tier 5:** The linear-scan allocator with precise liveness analysis
reduces the peak live register set to 30–50 registers for the GEMV inner loop and
35–60 registers for the full DeltaNet layer. With 64 registers per thread, 32 warps
reside on the SM simultaneously. 32 warps × 200-cycle LDG latency = 6,400 cycles of
latency the SM can hide by switching between warps. The arithmetic pipeline rarely stalls.

At 32 warps/SM with full latency hiding, bandwidth utilization approaches 75–90% of
peak. The remaining 10–25% loss comes from:
- Grid-sync overhead (67 synchronization points per megakernel, each briefly
  serializing all SMs)
- DeltaNet state update compute (the 128×128 matrix operations are partially
  compute-bound, not memory-bound)
- Instruction fetch latency for the large (~300MB) instruction stream
- RoPE and softmax in the 16 full-attention layers (less parallelism than GEMV)

**75–90% of 240 tok/s ceiling = 180–216 tok/s at batch=1.**

**The weights-as-code contribution:** At Tier 6, the Lithos megakernel embeds
Q4 weights as FP32 immediates in FMUL-IMM instructions rather than loading them
via LDG. This changes the weight delivery path from data cache (L1D → L2 → HBM)
to instruction cache (L1I → L2 → HBM). The HBM bandwidth consumed is the same.
The difference is:

- *No dequantization arithmetic:* The 6 ALU instructions per weight element
  (SHF, LOP3, I2F, FADD_IMM, FMUL, FFMA) collapse to 1 (FMUL-IMM). This frees
  ALU pipeline slots for activation loads and accumulation, improving instruction-level
  throughput.
- *Sequential instruction fetch:* The PC increments monotonically through the weight
  stream. HBM sequential prefetch is maximally efficient for this pattern.
  LDG with strided row-major access may incur more HBM row-activation overhead.
- *No activation-weight synchronization stalls:* A traditional GEMV must wait for
  both the weight LDG and the activation LDG before issuing FFMA. With weights-as-code,
  only the activation LDG (for x[k]) gates the FMUL-IMM. Half the memory dependency
  chain is eliminated.

The weights-as-code contribution at batch=1 is estimated at 10–20% throughput
improvement over an equivalent direct-SASS kernel using LDG for weights, primarily
from the dequant elimination and the cleaner memory dependency graph.

---

## Summary Table

*All numbers are speculative estimates for Qwen 3.5 27B W4A16, single GH200, autoregressive decode.*

| Tier | Path | tok/s BS=1 | tok/s BS=8 | ms/token | BW util | Warps/SM | Launches/tok |
|------|------|-----------|-----------|----------|---------|----------|-------------|
| 1 | Python + vLLM + CUDA | 40–70 | 250–450 | 14–25ms | 15–30% | 8–16 | 200–500 |
| 2 | C++ + CUDA | 70–110 | 500–750 | 9–14ms | 25–40% | 8–16 | 100–200 |
| 3 | C++ → PTX (model-specific) | 90–140 | 650–950 | 7–11ms | 35–55% | 16–32 | 50–150 |
| 4 | C++ → SASS (direct) | 120–170 | 800–1100 | 5.9–8.3ms | 45–65% | 16–32 | 30–80 |
| 5 | Lithos → opcode (naive alloc) | 90–140 | 550–850 | 7–11ms | 30–50% | 8–16 | **1** |
| 6 | Lithos → opcode (register-packed) | **180–230** | **1000–1400** | **4.3–5.6ms** | **75–90%** | 32+ | **1** |
| — | Theoretical ceiling | 240 | ~1400 | 4.2ms | 100% | 64 | — |

---

## Notable Observations

### Tier 5 is not better than Tier 3

This is the most important non-obvious result. A single-dispatch megakernel with naive
register allocation performs approximately the same as a multi-dispatch ptxas pipeline,
because the occupancy damage from naive allocation cancels the dispatch overhead savings.
The megakernel architecture is correct and will pay off — but only after the register
allocator is working.

### Tier 6 breaks the vLLM ceiling by 3–5×

The tier 6 estimate (180–230 tok/s) is 3–5× higher than a well-tuned vLLM deployment
on the same hardware. The sources of that gain:

| Source | Approximate gain |
|--------|----------------|
| Remove Python dispatch floor | ~2× at BS=1 |
| Megakernel (1 launch vs 200–500) | ~1.2× |
| Register packing (occupancy 8→32 warps) | ~2.5× |
| Weights-as-code (dequant elimination) | ~1.1–1.2× |
| Direct opcode (vs ptxas suboptimality) | ~1.1× |

These multiply rather than add. 2.0 × 1.2 × 2.5 × 1.15 × 1.1 ≈ **7.6×** over the
Python baseline at batch=1. The estimate of 3–5× is conservative — it accounts for
the fact that not all gains are fully additive and that some theoretical gains are
partially realized.

### The DeltaNet architecture uniquely favors this approach

A pure transformer (all softmax attention layers) scales with context length — the KV
cache grows and O(N) attention dominates at long contexts. Qwen 3.5 27B runs 48 of 64
layers as DeltaNet recurrence (O(1) per token, 230MB fixed state regardless of context
length). Only 16 layers accumulate KV cache.

This means the 240 tok/s bandwidth ceiling holds even at 64k-token contexts for 75% of
the model. Full attention layers (16 of 64) do grow with context, adding ~256MB of KV
reads per token at 64k context. But this is a linear additive term, not multiplicative.

At 64k context, the full attention layers add approximately 256MB × 16 layers ÷ 3,350 GB/s
= ~1.2ms per token. Against a 4.2ms baseline, the degradation is ~30% at 64k tokens —
versus a pure transformer where 64k context degrades performance by 3–10× depending
on implementation.

This is the architectural advantage of the hybrid model, and it is fully captured
by the Lithos compiler because the megakernel lays out all 64 layers at compile time
with no runtime dispatch logic.

### Context length scaling

| Context length | Traditional transformer | Qwen 3.5 27B Lithos (est.) |
|---------------|------------------------|---------------------------|
| 1k tokens | baseline | baseline |
| 8k tokens | ~0.5× (attention growing) | ~0.9× (only 16 layers affected) |
| 32k tokens | ~0.15× (attention dominant) | ~0.6× |
| 64k tokens | ~0.08× (memory saturated) | ~0.5× |
| 128k tokens | ~0.04× (likely OOM) | ~0.35× (state still fixed) |

The DeltaNet layers preserve near-linear throughput scaling with context length.
This is not a compiler achievement — it is a model architecture achievement that
the compiler faithfully preserves.

### Batch scaling

The theoretical ceiling scales linearly with batch size up to the compute-bound
crossover (~batch=40 on this model). Below that, batching improves bandwidth
utilization by amortizing weight reads across multiple activation streams.

For Tier 6 at batch=8, the estimate (1,000–1,400 tok/s total = 125–175 tok/s per
stream) reflects that the single megakernel handles 8 batch items without structural
change — the grid size increases to cover 8 items, and the same register budget
handles 8 independent accumulations in different thread groups. No architectural
change required.

---

## What Must Be True for Tier 6 to Reach 180 tok/s

Three things must hold simultaneously:

1. **Register allocator produces ≤64 regs/thread for the full DeltaNet megakernel.**
   Verified by reading SPD offset 0x094 (bits 23:16) from the compiled cubin. If this
   number is above 64, occupancy is below 50% and the estimate does not hold.

2. **Correctness is verified before performance is claimed.**
   The kernel must produce cosine ≥ 0.9999 against PyTorch reference at every layer.
   A kernel producing wrong output at 200 tok/s is not a 200 tok/s kernel.

3. **Grid-sync overhead is bounded.**
   67 grid-syncs per megakernel entry must each complete in ≤10µs (total sync overhead
   ≤670µs against a ~4.2ms token budget = ~16% overhead). This requires the
   MEMBAR + ATOMS pattern to clear quickly on Hopper's cooperative grid implementation.
   This is achievable but must be verified empirically.

If any of these three fails, the performance falls back toward Tier 5. The register
allocator is the most uncertain; the correctness check is the most important.

---

*Generated 2026-04-14. Update this document when hardware measurements are available.
The first measured data point should be the vadd.ls kernel — even one real measurement
anchors the rest of the estimates.*
