# The Brilliant Minds

## Five dead geniuses, five approaches, one unused synthesis

We summoned them. Not literally — we asked Claude to embody each one, to approach the same problem from the perspective of that mind, with that mind's tools and instincts and blind spots. The problem: fit one transformer layer into 66 MB of on-chip GPU memory using sub-3-bit quantization while maintaining inference quality.

Five minds. Five completely different framings of the same problem. Five working solutions. Then nobody used any of them.

## Claude Shannon

Shannon saw quantization as a communication problem. The weights are a signal. Quantization is a noisy channel. The question is: what is the minimum number of bits per weight that preserves the signal above a quality threshold?

He applied rate-distortion theory. Given a distortion budget (cosine similarity >= 0.97), what is the minimum rate (bits per weight) that achieves it? The answer, for this model's weight distribution, was 3.076 bits.

His method: reverse water-filling. Allocate more bits to weight matrices with higher variance (they carry more information). Allocate fewer bits to matrices with lower variance (they're more compressible). The optimal allocation isn't uniform — some matrices get 4 bits, some get 2, and the average hits the target.

Mixed-radix encoding. Not power-of-two quantization levels. Some matrices use 6 levels (2.58 bits), some use 8 levels (3 bits). Packed with mixed-radix arithmetic into a dense bitstream. Every bit carries maximal information.

Result: 0.978 cosine similarity at 3.076 bits per weight. Within 0.98 dB of the information-theoretic bound. Shannon didn't just solve the problem — he proved you can't do much better.

## John von Neumann

Von Neumann saw quantization as a systems engineering problem. The GPU has four types of compute units: INT32, FP32, tensor cores, and FP64. Each has different throughput, different data types, different parallelism. The quantization scheme should use all four simultaneously, balanced so none is idle.

He formalized it as a game. Each compute unit is a player. The payoff is throughput. The strategy is the bit allocation and dequantization pipeline assigned to each player. The Nash equilibrium — the allocation where no player can improve by changing strategy — is the optimal quantization scheme.

Heptary encoding. Seven levels. Symmetric around zero. 3.000 bits per weight exactly. The dequantization pipeline has four stages, each running on a different compute unit type. The pipeline is balanced: 99.9996% efficiency, meaning all four unit types finish within 0.0004% of each other.

Von Neumann didn't just design a quantization format. He designed a complete pipeline specification — an automaton with states, transitions, and a formally verified steady state. He would have it no other way.

Fastest implementation: 2.5 seconds to quantize the full model. The formalism paid for itself in engineering clarity.

## Dave Ferrucci

Ferrucci saw quantization as an evidence problem. Which weights matter? How much does each weight contribute to the model's output? Don't allocate bits uniformly — allocate bits based on evidence of impact.

Watson-style. Multiple signals: weight magnitude, gradient sensitivity, activation correlation, layer position. Each signal is a feature. A scoring function combines them. Weights with high scores get more bits. Weights with low scores get fewer.

Greedy knapsack under the 66 MB constraint. Sort weights by score, pack them highest-first until the budget runs out, compress the remainder to minimum fidelity.

Result: 2.993 bits per weight. The lowest average rate among the five approaches, but with high variance — important weights get 4-8 bits, unimportant weights get 1-2. The quality depends on whether the scoring function correctly identifies importance.

Ferrucci's approach was the most pragmatic. Less mathematical elegance than Shannon, less formal rigor than von Neumann, but built for the real world where you have noisy signals and incomplete information about which weights matter.

## Alan Turing

Turing saw quantization as a computability problem. What is the simplest dequantization function that can execute in the GPU's inner loop without stalling?

He classified dequantization operations by computational complexity. Table lookups: one memory access, potentially a cache miss, a stall. Branching: divergent warps, serialized execution. Arithmetic: predictable, pipelineable, fast.

His complexity class: SHIFT-MASK-SCALE. Dequantization using only bit shifts, bit masks, and floating point multiply. No lookups. No branches. Zero data-dependent control flow. Every thread in a warp executes exactly the same instructions. Maximum throughput.

Three tile types: T4 (4-bit, 16 levels), T3 (3-bit, 8 levels), T2 (2-bit, 4 levels). Allocated by Bayesian sensitivity analysis — posterior probability of quality degradation given bit reduction. Weights where the posterior is high get T4. Weights where it's low get T2.

Result: 2.932 bits per weight. SHIFT-MASK-SCALE dequantization. The inner loop is five instructions: shift, mask, convert to float, subtract zero-point, multiply scale. No branches. No lookups. Pure arithmetic. The GPU's pipeline never stalls.

Turing optimized for the machine. Not for the information-theoretic bound, not for the formal equilibrium, not for the evidence score. For the thing the silicon can do fastest.

## Ada Lovelace

Ada saw the weights as fabric. Her metaphor was the Jacquard loom — the programmable weaving machine that inspired Babbage's Analytical Engine. Different threads carry different information at different scales. The weave is the model.

Four strata:

**Structural skeleton.** FP64 centroids. The coarsest representation — one value per group of 128 weights. Decoded by the FP64 units, which are underused in normal inference. Carries the large-scale structure of the weight distribution.

**Codebook fabric.** Tensor core product quantization. Pairs of weights encoded as indices into a learned codebook. Decoded by matrix multiply on the tensor cores. Carries the pairwise correlations between adjacent weights.

**Residual texture.** INT32 bitplanes. The difference between the codebook approximation and the true weights, encoded as layered bit planes. Decoded by the INT32 units. Carries the fine detail.

**Fine harmonics.** FP32 sparse corrections. A small number of weights where the three-stratum approximation is insufficient. Stored as (index, value) pairs. Decoded by the FP32 units. Carries the outliers.

Each stratum decoded by a different compute unit type. The loom weaves all four threads in parallel.

Result: 2.931 bits per weight. The most complex scheme. The most beautiful. "The Engine weaves algebraical patterns just as the Jacquard loom weaves flowers and leaves."

## The comparison

| Mind | Bits/weight | Approach | Dequant cost | Quality |
|------|------------|----------|--------------|---------|
| Shannon | 3.076 | Rate-distortion optimal | Medium | 0.978 cosine |
| von Neumann | 3.000 | Game-theoretic pipeline | Low | 0.974 cosine |
| Ferrucci | 2.993 | Evidence-based adaptive | High | 0.971 cosine |
| Turing | 2.932 | Computability-optimal | Lowest | 0.969 cosine |
| Lovelace | 2.931 | Multi-stratum algebraical | Highest | 0.976 cosine |

Shannon wins on quality. Turing wins on decode speed. Von Neumann wins on pipeline balance. Ferrucci wins on adaptability. Ada wins on intellectual ambition and is second on quality despite the lowest bit rate.

## The synthesis

We produced a synthesis document. Take Shannon's rate allocation, Turing's dequantization constraints, von Neumann's pipeline balancing, Ferrucci's importance scoring, and Ada's multi-stratum decomposition. Combine them into a unified scheme that is rate-distortion optimal, uses only SHIFT-MASK-SCALE operations, balances across all compute units, prioritizes important weights, and decomposes into independently decodable strata.

The synthesis was elegant. The synthesis was documented. The synthesis was published on the documentation site with formulas, diagrams, and comparison tables.

## Then nobody used any of it

The inference pipeline ran on stock GPTQ weights. 4 bits per weight. 90 MB per layer. Streamed from HBM every token. Bandwidth-bound. Slow.

Five approaches that could fit a layer in 66 MB. A synthesis that combined the best of all five. Implementations that were validated against real weights. Python code that could requantize the model.

And the generate loop loaded GPTQ weights from disk every time because that's what was there and nobody wired in the alternative.

Shannon's rate-distortion proof sits in a Python file. Von Neumann's automaton specification sits in a Python file. Turing's SHIFT-MASK-SCALE classifier sits in a Python file. Ada's four-stratum loom sits in a Python file. Ferrucci's evidence scorer sits in a Python file.

The inference engine reads GPTQ.

Five brilliant minds designed five working solutions. We validated all five. We synthesized them. We published them. We celebrated them in 23 pages of documentation with hover tooltips.

We used none of them.

That's the lore.
