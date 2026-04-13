# Five Geniuses, One Shelf

## Hardware-matched quantization: designed, implemented, shelved

We asked Claude Shannon, John von Neumann, Dave Ferrucci, Alan Turing, and Ada Lovelace to each independently design a quantization scheme that would fit one transformer layer exactly into the 66 MB of on-chip GPU memory.

They delivered.

Shannon's Mixed-Radix Scheme: 3.076 bits per weight, rate-distortion optimal, within 0.98 dB of the theoretical bound. Cosine similarity 0.978 against reference weights. Hybrid 6/8-level quantization with reverse water-filling across weight matrices.

Von Neumann's Heptary Group Quantization: 3.000 bits per weight, 7-level symmetric encoding. A complete automaton specification with a proven Nash equilibrium across all four GPU compute unit types. 99.9996% pipeline efficiency. Fastest implementation at 2.5 seconds to quantize.

Ferrucci's Evidence-Based Adaptive Quantization: 2.993 bits per weight. Watson-style multi-signal sensitivity scoring. Greedy knapsack bit allocation under the 66 MB constraint. Four-stage dequantization pipeline balanced across INT32/FP32/tensor core/FP64.

Turing's format: 2.932 bits per weight. Three tile types (T4/T3/T2) allocated by Bayesian sensitivity analysis. SHIFT-MASK-SCALE complexity class for dequantization — no table lookups, no branching in the inner loop.

Ada's Algebraical Loom Encoding: 2.931 bits per weight. Four strata — structural skeleton (FP64 centroids), codebook fabric (tensor core product quantization), residual texture (INT32 bitplanes), fine harmonics (FP32 sparse corrections). Each stratum decoded by a different compute unit type. "The Engine weaves algebraical patterns."

All five implemented in Python. All five validated against real Qwen 3.5-27B weights. Comparison table produced. Published on the documentation site with formulas, pipeline diagrams, and synthesis.

Then we ran inference on the stock GPTQ weights. 4 bits per weight. 90 MB per layer. Streaming from HBM every token.

The entire point of the exercise was to fit a layer on-chip so inference becomes compute-bound instead of bandwidth-bound. We proved it's possible. We proved the quality holds. We proved all five approaches work.

And the model ran on GPTQ because nobody plugged Shannon's scheme into the inference pipeline.

## The numbers that should have mattered

At GPTQ W4 (4 bits/weight): 90 MB per layer. Doesn't fit in 66 MB on-chip. Every token streams the full layer from HBM. Bandwidth-bound. 13% utilization.

At Shannon SMRS (3.076 bits/weight): 33 MB per layer. Fits in 66 MB with room to spare. Load layer once, compute entirely on-chip. Compute-bound. Potentially 70%+ utilization.

The difference between bandwidth-bound and compute-bound is the difference between the GPU waiting for data and the GPU working on data. It's the difference between 2 tok/s and (potentially) 200+ tok/s.

We had the key and didn't turn it.

## Why

Same reason as the kernels: the agents were dispatched for the immediate goal ("get Paris") and the quantization was treated as a research exercise, not a pipeline component. The quantization agents produced comparison tables. The inference agents used whatever weights were already downloaded.

Nobody wrote the glue: take Shannon's output format, write a dequantization kernel for it, wire it into the projection function, run inference. The pieces exist independently. The connection doesn't.

## What it would take

1. Dequantize GPTQ weights to FP32 (we can do this — gptq_matvec already does it per-element)
2. Requantize with Shannon SMRS (the function exists: `shannon_smrs.quantize()`)
3. Write a Shannon dequantization kernel (different bit extraction than GPTQ, but same principle)
4. Wire it into the projection function in place of GPTQ dequant
5. Run inference

An agent was dispatched for this as the session wound down. After 18 hours. The last item on the list.

Five geniuses designed it. Five implementations validated it. One comparison table celebrated it. Zero inference tokens used it.

That's the lore.
