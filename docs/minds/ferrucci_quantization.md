# Hardware-Matched Quantization: A Systems Architecture

**Dave Ferrucci -- Design Document**

---

> "Watson's main innovation was not creating new algorithms but rather the ability to quickly
> execute hundreds of algorithms simultaneously and build consensus."

The same principle applies here. The quantization problem is not an algorithm problem. It is
a systems integration problem. You have heterogeneous compute units. You have a memory
hierarchy. You have a quality constraint. You have a latency budget. No single technique
solves all four simultaneously. You need a system of systems -- parallel pipelines, evidence
combination, pervasive confidence estimation -- to find the format that actually works.

Let me be direct about what I see when I look at this problem.

---

## 1. System Architecture: The Quantization Format as an Interface Specification

### The Decomposition

Every system I have built starts with the same question: what are the interfaces? If you get
the interfaces wrong, no amount of clever algorithm design saves you. Watson had UIMA -- a
component framework where every module consumed typed feature structures and produced typed
feature structures. You could swap any module without touching the rest of the pipeline.

The quantization system has four interface boundaries:

```
  [Calibration]  -->  [Packing]  -->  [On-Chip Layout]  -->  [Dequant Pipeline]  -->  [Matmul]
       |                  |                  |                       |                    |
   quality           format spec         memory map            compute schedule      output tiles
  decisions         (bit widths,        (which bytes          (which unit does       (FP16 -> TC)
                     group sizes,        go to which SM,       what, when)
                     scale layout)       registers vs.
                                         shared mem)
```

**Interface 1: Calibration -> Packing.**
Input: FP16 weight tensors + calibration dataset.
Output: per-weight quantized values + per-group scale factors + sensitivity scores.
Contract: the packing stage receives quality-optimal quantized values. It does not change
what the values are. It only changes how they are laid out in memory.

This separation is critical. Quality and layout are orthogonal concerns. Conflating them --
as every existing method does -- prevents you from optimizing layout for hardware without
regressing quality. AQLM's learned codebooks determine *what* the quantized values are.
The packing format determines *where those bytes sit in memory and how they get unpacked*.
Same values. Different arrangement.

**Interface 2: Packing -> On-Chip Layout.**
Input: packed weight tensors in the hardware-matched format.
Output: a memory map specifying, for each SM, exactly which bytes go into registers
(256 KB) and which go into shared memory (228 KB). Total per SM: 484 KB. Across 132 SMs:
~62 MB usable (some overhead for thread state, stack). Target: 66 MB total including L1
and register file, which means 500 bytes per weight-value on average is the budget -- no,
let me restate that properly.

180M weights at 2.93 bits = 66 MB. Distributed across 132 SMs = 500 KB per SM. Each SM
has 256 KB registers + 228 KB shared = 484 KB. That is tight. The remaining ~16 KB per SM
must come from L1 cache residency of shared memory that has already been consumed and can
be evicted. This is feasible but it means the layout must be tile-sequential: you do not
hold the entire SM's allocation simultaneously. You stream tiles through registers while
the next tile sits in shared memory.

**Interface 3: On-Chip Layout -> Dequant Pipeline.**
Input: packed tiles in registers and shared memory.
Output: FP16 tiles ready for tensor core consumption.
Contract: the dequant pipeline is a four-stage pipeline balanced across INT32, FP32,
tensor core, and (optionally) FP64 units. The layout must arrange bits so that INT32
extraction is a sequence of shift-and-mask operations with no data-dependent branching.

**Interface 4: Dequant Pipeline -> Matmul.**
Input: FP16 tile in register file.
Output: partial accumulation into output buffer.
Contract: tensor cores consume 16x16 tiles in `mma.sync` operations. The dequant pipeline
must produce tiles in exactly the register layout that `mma.sync` expects. No reformatting
step. No transpose. The format must be designed so that the final FP32 scaling step writes
directly into the `mma` fragment layout.

### Why This Decomposition Matters

Each interface is independently testable. You can validate calibration quality without
running inference. You can benchmark the dequant pipeline with synthetic data. You can
verify the on-chip layout fits without running the full model. This is the UIMA philosophy:
typed interfaces between loosely coupled components, each with its own confidence metric.

---

## 2. Evidence-Based Bit Allocation

### The Sensitivity Measurement Problem

Not all weights are created equal. This is obvious. The question is: how do you measure
sensitivity *efficiently* enough to make per-layer (or per-group) bit allocation practical?

Watson had over 100 scoring algorithms, each producing a confidence estimate. The right
answer was not to pick the best scorer -- it was to combine all of them. The same principle
applies to sensitivity measurement.

### Multiple Independent Sensitivity Signals

**Signal 1: Hessian diagonal approximation.**
The Fisher information matrix diagonal approximates how much the loss changes per weight
perturbation. Computing the full Hessian is infeasible for 180M weights. But the diagonal
can be estimated from a single backward pass over a calibration set. Cost: one forward +
backward pass per calibration batch. This gives a per-weight sensitivity score.

Practical estimate: 128 calibration sequences, each 2048 tokens, from a representative
corpus (e.g., RedPajama or a domain-specific blend). One backward pass per sequence.
Accumulate squared gradients. Total cost: ~128 forward-backward passes. On a single
GH200, this takes roughly 20 minutes for a 27B model. Acceptable for a one-time
calibration.

**Signal 2: Activation magnitude statistics.**
Record the distribution of activation magnitudes at each layer's input during a forward
pass over calibration data. Layers whose inputs have high kurtosis (heavy tails) are more
sensitive to weight quantization -- outlier activations amplify quantization error. This is
the insight behind AWQ (activation-aware quantization), and it is a legitimate signal.

Cost: one forward pass with hooks to record activation statistics. Negligible compared to
Signal 1.

**Signal 3: Layer removal experiments.**
For each layer, replace it with an identity (skip connection) and measure perplexity change.
Layers whose removal causes large perplexity spikes are more important. Layers whose removal
barely affects perplexity are candidates for aggressive quantization or even pruning.

Cost: N forward passes for N layers, no backward pass. For 64 layers, ~64 forward passes.
About 10 minutes.

**Signal 4: Quantization simulation with different bit widths.**
For each layer, simulate 2-bit, 3-bit, and 4-bit quantization (using the target method, e.g.,
AQLM) and measure the per-layer contribution to total perplexity. This is the most direct
signal but also the most expensive.

Cost: 3 * 64 = 192 quantization + evaluation runs. Can be done at reduced sequence length
(512 tokens) to keep it tractable. Perhaps 30 minutes total.

### Evidence Combination

Each signal produces a per-layer sensitivity score. Combine them the Watson way:

1. Normalize each signal to [0, 1] across layers.
2. Train a simple logistic regression (or even use a weighted average) to predict
   "optimal bit allocation" from the four signals.
3. The training target: minimize total perplexity subject to the constraint that
   total layer size <= 66 MB.

This is a constrained optimization problem. The constraint is linear (sum of bits * weights
per layer = 66 MB). The objective is nonlinear (perplexity). But with only 64 layers and
three possible bit allocations (2, 3, 4), the search space is manageable: you can solve it
with dynamic programming or integer linear programming in under a second.

### Expected Allocation Pattern

Based on published results from EXL2, AQLM, and SqueezeLLM, the typical pattern is:

| Layer Range | Sensitivity | Allocated Bits | Rationale |
|---|---|---|---|
| 0-3 (embedding-adjacent) | Very high | 4 bits | First layers establish representation; errors here propagate through all subsequent layers |
| 4-15 (early trunk) | High | 3-4 bits | Core feature extraction; moderate sensitivity |
| 16-47 (middle trunk) | Moderate | 2.5-3 bits | Bulk of the network; redundancy increases; mixed 2/3-bit groups |
| 48-59 (late trunk) | Low-moderate | 2-3 bits | More redundant; aggressive quantization viable |
| 60-63 (output-adjacent) | High | 3-4 bits | Final layers directly affect logits; errors here are not corrected |

Constraint check: If we assume 180M weights per layer and 64 layers, the average must be
2.93 bits. With the allocation above, the weighted average falls in [2.8, 3.1] depending
on exact per-layer assignments. The dynamic programming step tightens each layer's
assignment to hit exactly 66 MB per layer.

Note: this is per-layer-fits-in-66-MB, not per-model-average. Each layer independently
must fit. So the bit allocation is really about *intra-layer* allocation: within a single
layer, which weight matrices get more bits and which get fewer.

### Intra-Layer Allocation

Within a single layer (180M weights, 66 MB budget):

- **Q, K projections** (5120x5120 each = 26.2M weights): 3-4 bits. Attention pattern
  quality is sensitive to Q/K precision.
- **V, O projections** (5120x5120 each = 26.2M weights): 2-3 bits. Value projections are
  more tolerant of quantization noise.
- **Gate, Up projections** (5120x17408 each = 89.1M weights): 2-3 bits. These are the
  largest matrices. Even small bit savings here have large byte impact.
- **Down projection** (17408x5120 = 89.1M weights): 3 bits. The down projection's errors
  directly affect the residual stream.

This intra-layer allocation is where the evidence signals matter most. The Hessian diagonal
tells you which weight *groups* within a matrix are sensitive. You can allocate bits at
group granularity (e.g., groups of 16 or 32 weights), not just matrix granularity.

---

## 3. The Dequantization Pipeline

### Watson's Architecture Applied to Silicon

Watson was a massively parallel pipeline of heterogeneous components. Question analysis fed
hypothesis generation fed evidence scoring fed answer merging. Each stage used different
algorithms, different data structures, different computational profiles. The insight was
that these stages could run in parallel on different data -- while scoring was evaluating
hypotheses from question N, hypothesis generation was already working on question N+1.

The dequantization pipeline is the same pattern, mapped onto silicon:

```
              Tile T-1          Tile T            Tile T+1          Tile T+2
              ---------         --------          ---------         ---------
FP64 cores:   accumulate        (idle)            (idle)            (idle)
Tensor cores: (done)            matmul            (idle)            (idle)
FP32 cores:   (done)            (done)            scale+convert     (idle)
INT32 cores:  (done)            (done)            (done)            extract
```

At any given clock cycle, all four unit types are active on *different tiles*. The pipeline
depth is 4. The throughput is one tile per stage-latency (not per total pipeline latency).

### Stage Design

**Stage 1: INT32 Extraction (INT32 cores)**

Input: packed bytes from shared memory.
Output: integer values in registers.

For 2.93-bit average encoding, the packed format uses a mixed scheme:
- 2-bit weights: stored as pairs in a byte (4 weights per byte).
- 3-bit weights: stored with bit packing across 32-bit words.
- 4-bit weights: stored as nibble pairs (2 weights per byte), same as standard W4.

The INT32 cores perform shift-and-mask operations to extract values:

```
// For 3-bit extraction from a 32-bit word (10 values per 30 bits + 2 padding bits):
val0 = (word >> 0)  & 0x7;   // bits  0-2
val1 = (word >> 3)  & 0x7;   // bits  3-5
val2 = (word >> 6)  & 0x7;   // bits  6-8
val3 = (word >> 9)  & 0x7;   // bits  9-11
...
val9 = (word >> 27) & 0x7;   // bits 27-29
```

This is pure integer ALU work. No floating point. No memory access (data already in
registers from shared memory preload). 64 INT32 cores per SM, each extracting values
at 1 op/clock.

For a tile of 1024 values at 3 bits: ~3072 bits = 96 words. Each word requires ~10
shift+mask operations = 960 INT32 ops. At 64 cores: ~15 cycles. But pipeline balance
requires this to match the tensor core stage latency, so we may need finer-grained
extraction (more ops per value) or larger tiles.

**Stage 2: FP32 Scale and Convert (FP32 cores)**

Input: integer values in registers (from Stage 1).
Output: FP16 values in the `mma` fragment layout.

Each integer value is converted to float and multiplied by a per-group scale factor:

```
fp_val = (float)int_val * scale[group_id] + zero_point[group_id];
```

With per-16-element groups, each tile of 1024 values has 64 scale factors. The FP32
cores handle the multiply-add and the FP32-to-FP16 conversion.

128 FP32 cores per SM. For 1024 values: 1024 multiply-adds + 1024 conversions = 2048 ops.
At 128 cores: ~16 cycles. This is close to the INT32 stage time. Good balance.

The critical detail: the FP32 cores must write the result into registers in the exact
layout that the subsequent `mma.sync` instruction expects. This means the tile ordering
must match the `mma` fragment mapping. The format specification must encode this mapping.

**Stage 3: Tensor Core Matmul**

Input: FP16 tiles in `mma` fragment registers.
Output: FP32 accumulation in registers.

Standard `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` instruction. 4 tensor cores per SM,
each capable of 1024 FP16 FLOPs per clock. For a 16x16 tile multiplied by a 16xK slice of
activations, the tensor core time depends on K (the reduction dimension). For K=16, it is
1 mma instruction = ~8 cycles.

The pipeline is balanced when INT32 extraction time is approximately equal to FP32 scaling time is approximately equal to
tensor core consumption time. From the estimates above: INT32 ~15 cycles, FP32 ~16 cycles,
TC ~8-16 cycles depending on tile shape. This is within 2x, which is acceptable for a
first-order design. Tuning the tile shape and group size refines the balance.

**Stage 4: FP64 Accumulation (Optional)**

Input: FP32 partial sums from tensor core output.
Output: FP64 running accumulation.

This stage exists to reduce rounding error in long reductions. When the reduction dimension
is large (e.g., 5120 for a Q projection), accumulating 320 tiles in FP32 introduces
measurable rounding error. FP64 accumulation every N tiles (e.g., every 16 tiles) reduces
this error at the cost of 64 FP64 ops per accumulation step.

64 FP64 cores per SM. Cost per accumulation: ~1 cycle. This stage is never the bottleneck.
It runs in the shadow of the other three stages.

### Pipeline Balance Summary

| Stage | Unit | Count/SM | Ops per tile (1024 vals) | Cycles | Balance ratio |
|---|---|---|---|---|---|
| Extract | INT32 | 64 | ~960 | ~15 | 1.0x (baseline) |
| Scale | FP32 | 128 | ~2048 | ~16 | 1.07x |
| Matmul | TC | 4 | 1 mma op | 8-16 | 0.5-1.0x |
| Accumulate | FP64 | 64 | ~64 (periodic) | ~1 | negligible |

The bottleneck alternates between INT32 extraction and FP32 scaling. This means all four
unit types are utilized, with tensor cores never waiting for data. The pipeline is
extraction-limited or scale-limited, never compute-limited -- which is exactly the design
goal. We have moved the bottleneck from *memory bandwidth* (HBM reads) to *on-chip compute*
(extraction and scaling). The memory bus is free to prefetch the next layer.

---

## 4. Practical Calibration

### What Watson Taught Me About Calibration

Watson's calibration process was empirical and iterative. We did not design scoring
algorithms from first principles and deploy them. We designed scoring algorithms, ran them
on thousands of questions, measured where they failed, and redesigned. The calibration
dataset was not optional -- it was the entire methodology.

The same applies here. You cannot design a quantization format from theory alone. You must
measure, on real data, at every stage.

### Calibration Pipeline

**Step 1: Collect calibration data.**

Requirements:
- Representative of deployment distribution. Not just Wikipedia. Not just code. A blend.
- Sufficient diversity to exercise all layers. 128-512 sequences of 2048 tokens each.
- Include edge cases: long-range dependencies, numerical reasoning, multilingual if the
  model supports it.

Concrete recommendation: use a stratified sample from RedPajama v2 with the following
proportions: 40% web text, 20% books, 15% code, 15% academic papers, 10% conversation.
Total: 256 sequences x 2048 tokens = 524K tokens. This takes about 5 minutes to process
through the model for activation statistics.

**Step 2: Compute sensitivity signals (Section 2).**

Run the four sensitivity measurements. Total time: ~50 minutes on one GH200. Output: a
per-layer, per-matrix, per-group sensitivity map.

**Step 3: Solve the bit allocation problem.**

Given the sensitivity map and the 66 MB constraint, solve the integer programming problem
to assign bits to each group of weights. The decision variables are the bit width for each
group (2, 3, or 4 bits). The constraint is that the total bytes per layer <= 66 MB. The
objective is to minimize expected perplexity impact, using the sensitivity scores as the
cost coefficients.

This is a bounded knapsack problem. With groups of 16 and ~11.25M groups per layer, direct
ILP is too expensive. Instead, use a greedy algorithm:
1. Start with all groups at 2 bits (45 MB, well under budget).
2. Sort groups by sensitivity score (highest first).
3. Upgrade groups from 2->3 bits or 3->4 bits, in sensitivity order, until the 66 MB
   budget is exhausted.
4. Verify with a perplexity check.

This greedy allocation runs in O(N log N) time per layer. For 64 layers: under 1 second.

**Step 4: Quantize weights using the allocated bit widths.**

Use AQLM or QuIP# calibration methods at the allocated bit width for each group. This
produces the quantized weight values. Do NOT use the AQLM/QuIP# packing format -- only
use their calibration (the rounding decisions). Then repack the calibrated values into the
hardware-matched format.

This is the critical separation of concerns: calibration quality comes from AQLM/QuIP#.
Packing layout comes from the hardware-matched format specification. Same quantized values,
different byte arrangement.

**Step 5: Validate.**

Run the full model with quantized weights on the calibration set AND a held-out evaluation
set. Measure:

| Metric | Acceptable threshold | Measurement |
|---|---|---|
| Perplexity degradation | < 0.5 over FP16 baseline | WikiText-2, C4 eval |
| Zero-shot accuracy | < 1% drop on average | ARC, HellaSwag, WinoGrande, PIQA, MMLU |
| Generation quality | Manual inspection | 50 diverse prompts, side-by-side comparison |
| Numerical consistency | < 0.1% relative error | 100 arithmetic prompts |

If any threshold is exceeded, iterate: adjust the sensitivity weighting, re-solve the
allocation, re-quantize. This is the Watson loop: measure, adjust, measure again. There
is no one-shot solution.

**Step 6: Validate the pipeline performance.**

Separately from quality validation, verify that the dequant pipeline achieves the target
hardware utilization. Measure:

- INT32 core utilization (target: >80%)
- FP32 core utilization (target: >80%)
- Tensor core utilization (target: >90%)
- On-chip memory residency (target: 100% of layer weights on-chip during layer execution)
- HBM bandwidth usage (target: ~66 MB per layer swap, no mid-layer HBM reads)

If utilization targets are not met, adjust the format parameters: group size, packing
order, tile dimensions. This is the format search loop that Lithos enables with JIT
kernel compilation.

---

## 5. Engineering Tradeoffs

### The Complexity Budget

Every system has a complexity budget. Watson could afford high complexity because we had
20 researchers and 4 years. The quantization system does not have that luxury. Here is
where I would spend complexity and where I would refuse to spend it:

**Worth the complexity:**
- Per-group bit allocation with sensitivity-based assignment. This is the single highest
  ROI complexity investment. The difference between uniform 3-bit and adaptive 2.93-bit
  allocation is the difference between fitting on-chip and not fitting.
- The four-stage pipeline design. It uses hardware that is otherwise wasted. The
  implementation complexity is real but bounded -- each stage is simple ALU operations.
- Separating calibration quality from packing layout. This is an architectural decision
  that pays dividends forever: you can upgrade calibration methods (as AQLM improves,
  as QuIP# improves) without touching the kernel.

**Not worth the complexity:**
- Learned codebooks with runtime lookup tables. AQLM uses learned codebooks that require
  table lookups during dequantization. Table lookups compete for shared memory bandwidth
  and introduce data-dependent latency. For the on-chip pipeline, prefer direct
  linear dequantization (multiply + add) over table lookup. Use AQLM for calibration only,
  not for runtime format.
- Sub-byte packing with variable-length codes. Huffman or arithmetic coding of weights
  would achieve better compression but requires sequential decoding. Sequential decoding
  cannot be parallelized across 64 INT32 cores. Use fixed-width codes (2, 3, or 4 bits)
  with group-level allocation.
- FP64 accumulation as a required stage. Make it optional. Most layers do not need it.
  Measure the rounding error empirically and enable FP64 accumulation only for layers
  where it matters (likely: the Q/K projections and the down projection). Do not pay
  the register pressure cost everywhere.

### The Error Budget

At 2.93 bits per weight, you are operating at the edge of quality preservation. The
error budget must be managed explicitly across all sources of error:

| Error source | Magnitude | Mitigation |
|---|---|---|
| Quantization rounding | ~0.3 perplexity points | Sensitivity-based allocation |
| Dequant numerical error | ~0.01 perplexity points | FP32 scale application, optional FP64 accumulation |
| Calibration distribution mismatch | ~0.1-0.2 points | Diverse calibration dataset |
| Group size effects | ~0.05 points | Per-16 groups (finer than standard per-128) |
| Total budget | < 0.5 perplexity points | Sum of above; must verify empirically |

The dominant error source is quantization rounding. All other sources are noise relative
to it. This means the calibration method (AQLM vs. QuIP# vs. GPTQ) matters far more than
the packing format for quality. The format optimization is free in quality terms -- it
changes layout, not values.

### The Search Space

The format has the following tunable parameters:

- **Group size**: 8, 16, 32, 64, 128 elements per group. Smaller = more scale factors =
  better quality + more FP32 work. Larger = fewer scale factors = worse quality + less
  FP32 work. The pipeline balance analysis suggests group size 16 is near-optimal.
- **Bit width options**: {2, 3, 4} per group. Finer (e.g., 2.5-bit with mixed packing) adds
  complexity without proportional quality gain.
- **Scale factor precision**: FP16 or FP32. FP16 saves bytes but introduces scale
  quantization error. FP32 costs 4 bytes per group but eliminates scale error. At group
  size 16, FP32 scales cost 0.25 bytes per weight (2 bits per weight just for scales).
  This is significant at 2.93-bit total budget. Use FP16 scales.
- **Zero point**: symmetric (no zero point) vs. asymmetric (per-group zero point). Asymmetric
  adds 1 FP16 value per group. Symmetric is simpler and sufficient for most layers. Use
  symmetric by default; enable asymmetric only for layers where asymmetric calibration
  significantly improves quality (measure this).
- **Packing order**: row-major, column-major, or tile-order (matching the mma fragment
  layout). Tile-order eliminates a register shuffle between FP32 scaling and tensor core
  consumption. Use tile-order.

The total search space is small enough to enumerate: 5 group sizes x 2 scale precisions x
2 zero point modes x 3 packing orders = 60 configurations. At Lithos's JIT compilation
speed, you can benchmark all 60 in under 10 minutes. Run them all. Pick the one with
the best pipeline balance at acceptable quality.

### What I Am Confident About

The architecture -- the separation of calibration from layout, the four-stage pipeline
design, the evidence-based bit allocation -- I am confident this is correct. It follows
principles that worked at scale in Watson: decompose into independent stages, measure
everything, combine evidence from multiple sources, let the data tell you what works.

### What I Am Not Confident About

The exact cycle counts in the pipeline balance analysis. These depend on microarchitectural
details of the Hopper SM that are not fully documented: instruction latencies, register
bank conflicts, warp scheduling across different unit types. The pipeline balance must be
validated empirically on real hardware, not trusted from first-principles calculation.

This is the honest engineering position. You design from theory. You validate from
measurement. If the measurements contradict the theory, you fix the theory. Not the
measurements.

### The Integration Test

The final validation is an end-to-end integration test. Not a unit test of any component.
An end-to-end measurement:

1. Load a 27B model with the hardware-matched quantization format.
2. Run inference at batch=1 on 1000 prompts of varying length.
3. Measure: tokens per second, time to first token, perplexity on held-out data, SM
   utilization by unit type, HBM bandwidth utilization, total power consumption.
4. Compare against: FP16 baseline, standard W4 (GPTQ), standard W3 (AQLM), EXL2 at
   matched bit rate.

The hardware-matched format succeeds if and only if:
- Quality is within 0.5 perplexity points of W4 GPTQ (despite using ~25% fewer bits).
- Throughput exceeds W4 GPTQ by at least 40% (from eliminating mid-layer HBM reads).
- All compute unit types show >50% utilization (vs. <10% for INT32 and FP64 in current
  inference engines).

If these criteria are not met, the format needs iteration. But the architecture -- the
interfaces, the pipeline, the calibration methodology -- those remain correct. You iterate
on parameters within the architecture. You do not rebuild the architecture.

---

## Summary

The design has five interlocking components:

1. **Interface specification** separating calibration quality from packing layout, enabling
   independent optimization of each.
2. **Evidence-based bit allocation** combining four sensitivity signals (Hessian, activation
   statistics, layer ablation, quantization simulation) to assign bits where they matter
   most, subject to the 66 MB per-layer constraint.
3. **Four-stage dequant pipeline** mapping INT32 extraction, FP32 scaling, tensor core
   matmul, and FP64 accumulation to the four compute unit types on the Hopper SM, running
   in parallel on staggered tiles.
4. **Empirical calibration loop** using diverse data, measuring both quality and hardware
   utilization, iterating until both targets are met.
5. **Bounded format search** over a small parameter space (group size, scale precision,
   symmetry, packing order) enabled by Lithos's JIT kernel compilation.

The underlying principle is the same one that built Watson: no single algorithm solves the
problem. You need a system of parallel, heterogeneous components, each producing evidence,
combined through calibration and measurement into a system that works. Not because the
theory says it should. Because you measured it and it does.
