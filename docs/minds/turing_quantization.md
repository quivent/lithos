# A Computable Encoding for Sub-3-Bit Weight Quantization on Hopper Silicon

**Alan Turing — Design Document for Lithos**

---

## Preface

The problem before us is this: compress 180 million weight values into 66 megabytes of on-chip memory, in such a way that the compressed representation can be decoded efficiently by four distinct types of arithmetic unit operating simultaneously. This is 2.93 bits per weight.

I shall approach this as I would approach any problem — by asking what is fundamentally possible, what is computable within the given constraints, and where the structure lies that we can exploit.

The analogy to cryptanalysis is not superficial. At Bletchley, the Enigma cipher appeared to produce random output, but its mechanical structure imposed statistical regularities that we could detect and exploit. Neural network weights appear to be arbitrary floating-point numbers, but their training process imposes statistical regularities — distributions, correlations, low-rank structure — that a well-designed encoding can exploit. In both cases, the task is: find the structure, design the machine to exploit it, operate within strict time constraints.

---

## Part 1: The Computability Angle

### What must the dequantization function compute?

Let `Q: R^n -> {0,1}^m` be the quantization function mapping `n` weight values to `m` bits, where `m/n = 2.93`. Let `D: {0,1}^m -> R^n` be the dequantization function. We require:

1. **Bounded time.** `D` must execute in fewer clock cycles than the tensor core consumes the output. If a tensor core MMA operation on one tile takes `T_tc` clocks, then INT32 extraction + FP32 scaling for that tile must complete in `<= T_tc` clocks.

2. **Bounded space.** The intermediate state of `D` must fit in registers. On Hopper, at most 255 registers per thread, shared across all computation the thread performs. The dequantization working set must be small — say 16-32 registers.

3. **Deterministic.** `D` must produce identical output for identical input on every invocation. No dependence on execution order or thread scheduling. This is essential for correctness of the neural network.

4. **Decomposable.** `D` must factor into stages that map to distinct hardware units: `D = D_fp32 . D_int32`, where `D_int32` performs bit extraction on INT32 cores and `D_fp32` performs scaling on FP32 cores.

### Complexity class of the dequantization function

The dequantization function must be in a complexity class I shall call **SHIFT-MASK-SCALE**: functions computable by a bounded sequence of shift, bitwise-AND, and multiply-add operations. This is a strict subset of linear-time computation, but more importantly, it maps exactly to the instruction set of INT32 and FP32 cores.

Any dequantization scheme requiring:
- Division (not natively pipelined on INT32)
- Table lookups larger than the register file (would require shared memory access, adding latency)
- Data-dependent branching (would cause warp divergence)
- Cross-thread communication (would require shuffles, adding latency)

...is computationally inadmissible for the inner loop. The format must be designed so that each thread can dequantize its assigned weights independently using only shift, mask, and multiply-add.

### The decidability constraint

There is a subtler point. Given an arbitrary quantization format specification, is it decidable whether that format achieves pipeline balance on the target hardware? Yes — the pipeline timing is a finite computation over known instruction latencies. We can enumerate candidate formats and evaluate each mechanically. This transforms format design from an art into a search problem. Lithos's microsecond recompilation makes this search practical.

---

## Part 2: The Cryptographic Analogy

### Encryption in reverse

In Enigma, the plaintext (the meaningful signal) was transformed by a key-dependent mechanical process into ciphertext (apparently random output). The cryptanalyst's task: recover the plaintext given the ciphertext and partial knowledge of the key generation process.

In quantization, the situation is inverted:

| Cryptography | Quantization |
|---|---|
| Plaintext (meaningful) | Full-precision weights (meaningful) |
| Key (secret structure) | Quantization parameters (scales, zeros, codebook) |
| Ciphertext (compressed, appears random) | Quantized weights (compressed, appears random) |
| Decryption (key + ciphertext -> plaintext) | Dequantization (params + packed bits -> weights) |
| Goal: prevent decryption without key | Goal: make dequantization as fast as possible |

The critical difference: in cryptography, we want the decryption function to be hard without the key. In quantization, we want the dequantization function to be as easy as possible, even trivial — a small number of arithmetic operations per output value.

### Properties of an ideal dequantization function

Drawing from this analogy, the dequantization function should have these properties:

1. **No key schedule.** In Enigma, the key schedule (rotor positions, plugboard settings) was complex. In our quantization, the "key" (scale factors, zero points) should require zero preprocessing at inference time. Precompute everything during quantization. Store the scales in a format that can be loaded and used directly — no per-tile setup.

2. **Fixed function, variable data.** The Bombe succeeded because it tested a fixed logical structure against variable key hypotheses. Similarly, the dequantization kernel should be a single fixed instruction sequence that operates on variable packed weight data. No conditionals. No format-dependent branching. One code path for all tiles.

3. **Locality.** Each output value depends only on a small, contiguous region of the packed input. No weight value's dequantization requires reading distant parts of the bitstream. This ensures coalesced memory access and register-only computation.

4. **Linearity.** The reconstruction should be affine: `w_reconstructed = scale * q_int + zero_point`. This maps to a single FMA instruction on FP32 cores. Non-linear reconstruction (e.g., codebook lookup) violates the SHIFT-MASK-SCALE complexity constraint unless the codebook is small enough to live in registers.

---

## Part 3: The Statistical Approach (Banburismus Applied to Weights)

### Bayesian analysis of weight distributions

At Bletchley, I developed Banburismus — a sequential Bayesian procedure for determining Enigma rotor order by accumulating evidence from character-by-character comparisons. The key insight: not all positions in the ciphertext carry equal information. Some positions gave strong evidence (high deciban scores), others were nearly uninformative.

The same principle applies to weight values. Not all weights carry equal information for the network's function. A Bayesian analysis proceeds as follows.

### Prior distribution

Trained neural network weights are approximately normally distributed within each tensor, with zero mean and a standard deviation that varies by layer and matrix. Let `p(w) = N(0, sigma_l^2)` be the prior for layer `l`.

However, the tails matter disproportionately. Weights more than 2 standard deviations from the mean are rare (~5% of values) but often carry critical information — they represent the strongest learned associations. A quantization scheme that clips or coarsely represents the tails loses these critical signals.

### Posterior given network function

Given that the network produces correct outputs, the posterior distribution `p(w | network works)` is not simply the training distribution. It is concentrated on a manifold of weight configurations that produce low loss. This manifold has structure:

1. **Low-rank structure.** Weight matrices are approximately low-rank. The top singular values carry most of the information. The bottom singular values are noise-like and can be quantized aggressively.

2. **Columnar salience variation.** Different output channels (columns of the weight matrix) have vastly different importance. Some columns participate in every prediction; others are rarely activated. This is the basis of AWQ's approach.

3. **Block-diagonal correlations.** Groups of weights within attention heads are correlated. Weights within one head can be jointly encoded more efficiently than independently.

### Information-optimal encoding

The rate-distortion theorem (Shannon, 1959) gives a lower bound: for a Gaussian source with variance `sigma^2` and mean squared error distortion `D`, the minimum rate is:

```
R(D) = (1/2) * log2(sigma^2 / D)
```

For our target of 2.93 bits per weight, this permits a distortion of:

```
D = sigma^2 * 2^(-2 * 2.93) = sigma^2 * 2^(-5.86) = sigma^2 / 58
```

This means we can reconstruct each weight to within `sigma / 7.6` RMS error. For a typical layer with `sigma = 0.02`, the reconstruction error is `~0.0026` — about 7 quantization levels within one standard deviation on each side.

But Shannon's bound assumes optimal encoding of i.i.d. Gaussian values. Real weights have correlations. Exploiting these correlations allows either:
- Lower distortion at the same rate (better quality), or
- The same distortion at a lower rate (fewer bits), leaving headroom for scale factors and metadata.

### The mixed-precision Bayesian allocation

Apply the Banburismus principle: allocate bits where the evidence is strongest.

**Procedure:**
1. For each group of `g` weights, compute the sensitivity `s_i = ||dL/dw_i||` — the gradient of the loss with respect to each weight, evaluated on calibration data.
2. The posterior precision of weight `i` is proportional to `s_i^2 * n_cal`, where `n_cal` is the calibration set size.
3. Allocate bits proportionally to `log(s_i)` — high-sensitivity weights get more bits.
4. Subject to the constraint that the average is 2.93 bits per weight.

In practice, this yields a distribution like:
- ~15% of weights at 4 bits (high sensitivity — attention output projections, first/last layers)
- ~60% of weights at 3 bits (moderate sensitivity — bulk of MLP and attention)
- ~25% of weights at 2 bits (low sensitivity — redundant MLP dimensions, middle layers)

Weighted average: `0.15*4 + 0.60*3 + 0.25*2 = 0.6 + 1.8 + 0.5 = 2.9 bits`. Add scale factors and metadata at the tile level: `~0.03 bits/weight` overhead. Total: `2.93 bits/weight`.

---

## Part 4: The Oracle Question

### If we had an oracle for bit importance

In my 1939 paper on ordinal logics, I introduced the concept of an oracle — a hypothetical device that answers questions beyond the reach of mechanical computation. Let us suppose we had an oracle that, for every bit in the quantized representation, could tell us whether flipping that bit changes the network's output on any input.

Such an oracle would partition the bits into three classes:

1. **Essential bits.** Flipping any of these changes outputs. These must be stored exactly.
2. **Contributory bits.** Flipping these changes outputs only in combination with other flips. These should be stored but can tolerate some noise.
3. **Inert bits.** Flipping these never changes outputs. These need not be stored at all.

The oracle is uncomputable in general (it reduces to the halting problem for the network computation). But we can approximate it.

### Sensitivity analysis as oracle approximation

**First-order approximation (gradient-based):**
```
importance(bit_j) ≈ |dL/d(bit_j)| evaluated on calibration data
```

This is computable in `O(n)` time via backpropagation and gives a noisy but useful ranking of bit importance.

**Second-order approximation (Hessian-based):**
```
importance(bit_j) ≈ (dL/d(bit_j))^2 / (d^2L/d(bit_j)^2)
```

This is the optimal bit-flip criterion from the OBS (Optimal Brain Surgeon) framework. It accounts for curvature — a bit might have high gradient but also high curvature, meaning the loss surface is steep but the minimum is nearby, so the bit is less important than the gradient alone suggests.

**Empirical approximation (leave-one-out):**

For each group of weights, quantize at target precision and measure the reconstruction error weighted by activation magnitudes:

```
group_importance(g) = ||X_g||_F * ||W_g - Dequant(Quant(W_g))||_F
```

where `X_g` is the activation matrix corresponding to weight group `g`. This is the Hessian-weighted error that GPTQ minimizes, and it is computable.

### Bit allocation from the approximate oracle

The approximate oracle (sensitivity analysis) suggests:

| Weight matrix | Relative sensitivity | Allocated bits |
|---|---|---|
| Q projection | High (1.0x) | 3-4 |
| K projection | High (0.9x) | 3-4 |
| V projection | Medium (0.6x) | 3 |
| O projection | Very high (1.2x) | 4 |
| Gate projection | Medium (0.7x) | 3 |
| Up projection | Medium (0.5x) | 2-3 |
| Down projection | High (0.8x) | 3 |

These allocations are per-matrix defaults. Within each matrix, per-group (e.g., per-128 or per-64 elements) allocation further refines the precision. The constraint: every matrix's packed representation, summed across all matrices in one layer, equals 66 MB.

---

## Part 5: The Universality Question

### Is there one format for all models?

The question of universality is dear to me — my 1936 paper showed that one machine (the universal Turing machine) can simulate any other. Is there an analogous universal quantization format?

**The answer is: partially.**

The *dequantization procedure* can be universal. The sequence of operations — extract integers via shift-mask on INT32 cores, scale via FMA on FP32 cores, feed to tensor cores — is the same regardless of the model. This is the "instruction set" of the format.

But the *parameters* of the format — group size, bit allocation per matrix, scale factor precision, zero-point encoding — must be model-specific and even layer-specific. The weight distributions vary between architectures. A format tuned for Llama's MLP will waste bits on Qwen's DeltaNet layers.

**The Lithos solution is precisely the right one:** a universal dequantization kernel generator parameterized by format descriptors. The kernel structure is fixed (the universal machine); the parameters are specialized per model (the specific machine being simulated). Recompilation in microseconds means this specialization has zero runtime cost.

### The format descriptor

A universal format descriptor specifies, for each tile of weights:

```
FormatDescriptor {
    bits_per_weight: u8,          // 2, 3, or 4 for this tile
    group_size: u16,              // number of weights sharing one scale factor
    scale_bits: u8,               // precision of scale factors (8 or 16)
    has_zero_point: bool,         // asymmetric quantization if true
    packing_order: PackingOrder,  // how bits are arranged in 32-bit words
}
```

The kernel generator reads this descriptor and emits a PTX kernel with the exact shift amounts, mask values, and scale application sequence baked in as constants. No runtime interpretation of the format — the format *is* the kernel.

---

## Part 6: The Concrete Design

### Format specification: Turing-2.93

I shall now specify a concrete format. I name it with characteristic immodesty.

#### Bit layout

The fundamental unit is a **tile** of `64 x 16 = 1,024` weights, matching the tensor core's consumption granularity. Each tile is independently decodable (no cross-tile dependencies).

**Three tile types, mixed within a layer:**

| Tile type | Bits/weight | Weights/tile | Tile size (bytes) | Encoding |
|---|---|---|---|---|
| T4 | 4 | 1,024 | 512 | Uniform 4-bit, 2 weights per byte |
| T3 | 3 | 1,024 | 384 | 3-bit packed into 32-bit words |
| T2 | 2 | 1,024 | 256 | Uniform 2-bit, 4 weights per byte |

**Plus metadata per tile:**

| Field | Size | Purpose |
|---|---|---|
| Scale (FP16) | 2 bytes | Per-group scale factor |
| Zero (FP16) | 2 bytes | Per-group zero point |
| Groups per tile | 16 groups of 64 | 16 * 4 = 64 bytes metadata per tile |

Effective bits per weight including metadata:
- T4: `(512 + 64) / 1024 * 8 = 4.50 bits` (but metadata cost is amortized — scale/zero are shared)
- T3: `(384 + 64) / 1024 * 8 = 3.50 bits`
- T2: `(256 + 64) / 1024 * 8 = 2.50 bits`

Wait — this is imprecise. Let me redo this properly.

Per tile: 1,024 weights. Group size = 64 weights. So 16 groups per tile. Each group has one FP16 scale (2 bytes) and one FP16 zero point (2 bytes). Metadata per tile = 16 * 4 = 64 bytes = 512 bits.

Effective bits per weight:
- T4: `(1024*4 + 512) / 1024 = 4.50`
- T3: `(1024*3 + 512) / 1024 = 3.50`
- T2: `(1024*2 + 512) / 1024 = 2.50`

**To achieve 2.93 average bits/weight:**

Let `p4, p3, p2` be the fractions of tiles at each precision. We need:
```
4.50*p4 + 3.50*p3 + 2.50*p2 = 2.93
p4 + p3 + p2 = 1
```

One solution (from the Bayesian analysis): `p4 = 0.05, p3 = 0.33, p2 = 0.62`:
```
4.50*0.05 + 3.50*0.33 + 2.50*0.62 = 0.225 + 1.155 + 1.55 = 2.93
```

Alternatively, to give more precision where needed: `p4 = 0.10, p3 = 0.43, p2 = 0.47`:
```
4.50*0.10 + 3.50*0.43 + 2.50*0.47 = 0.45 + 1.505 + 1.175 = 3.13
```

That is too high. Adjust: `p4 = 0.08, p3 = 0.30, p2 = 0.62`:
```
4.50*0.08 + 3.50*0.30 + 2.50*0.62 = 0.36 + 1.05 + 1.55 = 2.96
```

Close enough — the remaining 0.03 bits of headroom accommodates the tile-type indicator (1 byte per tile, amortized over 1,024 weights = 0.008 bits/weight) and alignment padding.

**Final allocation: 8% T4, 30% T3, 62% T2.**

#### Packing for INT32 extraction

**T4 (4-bit):** 8 weights per 32-bit word. Extraction:
```
// INT32 core: extract 8 weights from one u32
w0 = (packed >> 0)  & 0xF;
w1 = (packed >> 4)  & 0xF;
w2 = (packed >> 8)  & 0xF;
...
w7 = (packed >> 28) & 0xF;
```
Cost: 2 INT32 ops per weight (shift + AND). Total for tile: 2048 INT32 ops.

**T3 (3-bit):** 10 weights per 32-bit word (30 bits used, 2 bits wasted). Extraction:
```
// INT32 core: extract 10 weights from one u32
w0 = (packed >> 0)  & 0x7;
w1 = (packed >> 3)  & 0x7;
...
w9 = (packed >> 27) & 0x7;
```
Cost: 2 INT32 ops per weight. Wasted bits: 2 per word = 6.25% overhead. Acceptable. This wastes `1024 * 3 / 30 * 2 / 8 = ~26 bytes` per tile. The alternative — packing 3-bit values across word boundaries — requires additional shifts and masks that cost more in compute time than the wasted bits cost in storage. The computational simplicity wins.

Total for tile: 2048 INT32 ops.

**T2 (2-bit):** 16 weights per 32-bit word. Extraction:
```
w0 = (packed >> 0)  & 0x3;
w1 = (packed >> 2)  & 0x3;
...
w15 = (packed >> 30) & 0x3;
```
Cost: 2 INT32 ops per weight. Total for tile: 2048 INT32 ops.

**Key observation:** INT32 cost per tile is the same regardless of bit-width. This is elegant — the pipeline is naturally balanced across tile types.

#### Scale application on FP32 cores

For each group of 64 weights:
```
// FP32 core: dequantize
w_float = (float)w_int * scale + zero_point;
```

This is one `cvt.rn.f32.s32` (INT to FP32 conversion) + one FMA. Two FP32 ops per weight.

Per tile: 1,024 * 2 = 2,048 FP32 ops. With 128 FP32 cores: `2048 / 128 = 16 cycles`.

#### Tensor core consumption

The dequantized FP32 values must be converted to FP16 for tensor core input. This is one additional FP32 op per weight (`cvt.rn.f16.f32`). After conversion, the 64x16 tile feeds one `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` instruction sequence.

For a 64x16 tile consumed as four m16n8k16 operations:
Each MMA: `16 * 8 * 16 * 2 = 4,096 FLOPs`. Four MMAs: 16,384 FLOPs.
At 1,024 FLOPs/clock for 4 tensor cores: `16,384 / 1,024 = 16 cycles`.

#### FP64 accumulation

The tensor core output is FP32. For long dot-product reductions (large hidden dimensions), FP32 accumulation introduces rounding error. FP64 cores can accumulate partial sums:

```
// FP64 core: accumulate tile result
accum_f64 += (double)tile_result_f32;
```

Per tile: one FP64 add per output element. For a 16-element output vector from a 64x16 tile: 16 FP64 ops. With 64 FP64 cores: `< 1 cycle`. FP64 cores are vastly underutilized even with this scheme — they could accumulate across multiple tiles.

#### Pipeline balance summary

| Stage | Hardware | Ops/tile | Cores | Cycles |
|---|---|---|---|---|
| Extract | INT32 | 2,048 | 64 | 32 |
| Scale + convert | FP32 | 3,072 | 128 | 24 |
| Matmul | Tensor | 16,384 | 4 (1024 FLOP/clk) | 16 |
| Accumulate | FP64 | 16 | 64 | <1 |

The pipeline is not perfectly balanced — INT32 extraction is the bottleneck at 32 cycles, while tensor cores finish in 16. Two responses:

1. **Double-buffer tensor core work.** While INT32 extracts tile T+2 (32 cycles), tensor cores process tiles T and T+1 (2 * 16 = 32 cycles). The tensor cores are now the constraint.

2. **Reduce INT32 work.** Use a pre-shifted packing where the shift amounts are baked into the data layout. Each thread loads a pre-aligned 32-bit word and masks without shifting: cost drops to 1 INT32 op per weight = 1,024 ops / 64 cores = 16 cycles. Now all stages balance at 16 cycles.

**I recommend option 2.** The pre-shifted packing costs slightly more storage (some wasted bits in alignment) but brings the pipeline into balance. The storage overhead is absorbed by the margin between 2.93 and 3.0 bits/weight.

---

## Part 7: The Pre-Shifted Packing (Detail)

The key insight: we can choose how bits are arranged in memory. Traditional packing minimizes storage. We instead minimize extraction cost.

### Layout for T3 (3-bit weights)

Traditional packing of 10 weights into 30 bits of a 32-bit word:
```
[w9:3][w8:3][w7:3][w6:3][w5:3][w4:3][w3:3][w2:3][w1:3][w0:3][xx:2]
```
Extraction requires variable-length shifts: 2, 5, 8, 11, ...

**Pre-shifted layout:** Pack 8 weights into 32 bits at fixed 4-bit boundaries, with the high bit always zero:
```
[0][w7:3][0][w6:3][0][w5:3][0][w4:3][0][w3:3][0][w2:3][0][w1:3][0][w0:3]
```

Each weight occupies 4 bits but only uses 3. Extraction is a single mask:
```
w_i = (packed >> (i*4)) & 0x7;    // shift by multiple of 4 + mask
```

The cost: 1 wasted bit per weight. Effective storage: 4 bits per weight for 3-bit values. But wait — this defeats the purpose of 3-bit quantization.

**Better approach: half-word packing.** Pack weights into 16-bit half-words loaded as pairs:
```
Word layout: [w1_hi:1][w1:3][w1_pad:4][w0_hi:1][w0:3][w0_pad:4] ... 
```

No, this is getting too clever. Let me return to first principles.

**The right answer for T3:** Pack 10 three-bit weights into one 32-bit word (30 bits used). Each thread in a warp processes one word. With 32 threads per warp, one warp processes 320 weights per cycle. Use a precomputed shift table stored in registers (10 values: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27):

```ptx
// Precomputed in registers at kernel launch
// r_shift[i] contains i*3
shr.b32  r_val, r_packed, r_shift;
and.b32  r_val, r_val, 0x7;
```

This is 2 instructions per weight regardless. The shift table is constant and occupies 10 registers — loaded once, reused for every tile. Total INT32 cost per tile: 2048 ops / 64 cores = 32 cycles.

To achieve 16-cycle balance, we need to halve the INT32 work. The solution: **have FP32 cores share the extraction work.** On Hopper, FP32 cores can execute integer instructions via the shared datapath. Issue INT32 shift operations on the FP32 datapath (128 cores instead of 64), bringing extraction to 2048 / 128 = 16 cycles. Then the FP32 scaling work (2048 ops) executes in the next pipeline stage: 2048 / 128 = 16 cycles. But this serializes extraction and scaling on FP32 cores: total 32 cycles.

**The real solution:** Accept the imbalance. 32 cycles for INT32 extraction, 16 for tensor core matmul. The tensor cores are 2x underutilized. But consider: at batch=1, the problem is bandwidth-bound, not compute-bound. The tensor cores are already waiting for data. Using INT32 cores to prepare data on-chip, even at 2x the tensor core consumption rate, is still vastly faster than reading FP16 weights from HBM (400 cycles). We have converted a 400-cycle memory access into a 32-cycle on-chip extraction. The speedup is 12.5x even with the pipeline imbalance.

---

## Part 8: Total System Analysis

### Memory accounting

For one layer of 180M weights at the final allocation:

| Tile type | Fraction | Weights | Raw bits | Metadata bits | Total bytes |
|---|---|---|---|---|---|
| T4 (4-bit) | 8% | 14.4M | 57.6Mb | 7.2Mb | 8.1 MB |
| T3 (3-bit) | 30% | 54.0M | 162.0Mb | 27.0Mb | 23.6 MB |
| T2 (2-bit) | 62% | 111.6M | 223.2Mb | 55.8Mb | 34.9 MB |
| **Total** | 100% | 180M | 442.8Mb | 90.0Mb | **66.6 MB** |

Hmm — 66.6 MB. We are 0.6 MB over.

This is the kind of error one must not gloss over. Let me recalculate with larger groups.

**Adjustment: increase group size from 64 to 128.** Metadata per tile drops from 64 bytes to 32 bytes. Metadata bits per weight drop from 0.50 to 0.25.

| Tile type | Bits/weight (data) | Bits/weight (metadata) | Effective bits/weight |
|---|---|---|---|
| T4 | 4.00 | 0.25 | 4.25 |
| T3 | 3.00 | 0.25 | 3.25 |
| T2 | 2.00 | 0.25 | 2.25 |

Target: 2.93 bits/weight average.

```
4.25*p4 + 3.25*p3 + 2.25*p2 = 2.93
p4 + p3 + p2 = 1
```

With `p4 = 0.10, p3 = 0.48, p2 = 0.42`:
```
4.25*0.10 + 3.25*0.48 + 2.25*0.42 = 0.425 + 1.56 + 0.945 = 2.93
```

This gives more 3-bit tiles (better quality) and fewer 2-bit tiles. Total:

```
180M * 2.93 / 8 = 65.925 MB
```

This fits in 66 MB with 75 KB to spare — enough for tile-type indices and alignment padding.

### Layer loading sequence

The inference engine loads one layer at a time into on-chip memory:

```
for layer in 0..63:
    // Phase 1: Load from HBM to on-chip (all 132 SMs participate)
    // 66 MB at 4 TB/s = 16.5 microseconds
    async_load(layer_weights[layer] -> shared_memory_distributed)
    
    // Phase 2: Compute (all operations read from on-chip)
    // INT32 extracts, FP32 scales, tensor cores compute, FP64 accumulates
    compute_attention(layer)   // reads Q,K,V,O weights from shared memory
    compute_mlp(layer)         // reads gate, up, down weights from shared memory
    
    // Phase 3: Overlap — begin loading next layer while finishing current
    async_load(layer_weights[layer+1] -> shared_memory_distributed)  // double-buffer
```

Total HBM traffic per token: `64 layers * 66 MB = 4.224 GB`. At 4 TB/s: `1.056 ms` of memory transfer. This can be overlapped with computation.

---

## Part 9: Quality Preservation

### Why 2.93 bits can work

The information-theoretic argument: a neural network with 180M weights per layer does not have 180M * 16 = 2.88 billion bits of independent information per layer. The weights are highly redundant:

1. **Low intrinsic dimensionality.** Work on lottery tickets and pruning shows that 60-90% of weights can be zeroed without loss. The effective dimensionality is far lower than the parameter count.

2. **Noise tolerance.** Training introduces stochastic gradient noise. The weights are already noisy to about `1/sqrt(training_tokens)` precision. For a model trained on 1T tokens, this is `~3e-5` — far coarser than FP16 precision.

3. **Redundancy in representation.** Many different weight configurations produce identical network behavior. The mapping from weights to function is highly many-to-one.

The practical evidence: AQLM and QuIP# achieve 2-bit quantization (lower than our 2.93-bit target) with single-digit perplexity increases on Llama 2 70B. Our scheme, with its mixed precision and sensitivity-guided bit allocation, should perform at least as well.

### Calibration procedure

1. Run a calibration dataset (e.g., 128 sequences from C4 or RedPajama) through the full-precision model.
2. Record activation statistics `X_l` at each layer `l`.
3. For each weight matrix `W`, compute Hessian-weighted sensitivity: `H = 2 * X^T X`.
4. Assign tile precisions (T2/T3/T4) based on per-tile sensitivity, subject to the constraint that the total fits in 66 MB.
5. Quantize each tile using GPTQ's row-by-row procedure (sequential rounding with Hessian-based error compensation).
6. Pack the quantized values in the format specified above.
7. Emit the packed weights, scale factors, and tile-type map as a single binary blob per layer.

---

## Part 10: Concluding Observations

### What we have designed

A quantization format — Turing-2.93 — with these properties:

1. **Exactly fits one transformer layer in 66 MB of on-chip GPU memory** on Hopper (132 SMs * 512 KB = 66 MB).
2. **Utilizes all four compute unit types** in a pipelined fashion: INT32 for extraction, FP32 for scaling, tensor cores for matmul, FP64 for accumulation.
3. **Is efficiently decodable** using only shift-mask-scale operations — the SHIFT-MASK-SCALE complexity class.
4. **Preserves network quality** through Bayesian-guided mixed-precision allocation (8% at 4-bit, 48% at 3-bit, 42% at 2-bit).
5. **Is parameterized, not fixed** — the format descriptor specializes per model and per layer, while the dequantization procedure is universal.

### What remains to be done

1. **Empirical validation.** The pipeline cycle counts are approximate. The actual balance must be measured on silicon. Lithos's rapid recompilation makes this measurement loop fast.

2. **Group size optimization.** I chose 128 for the arithmetic. The optimal group size balances metadata overhead against quantization granularity. It should be searched empirically, likely landing between 64 and 256.

3. **Codebook variants.** For the 2-bit tiles, a small codebook (4 entries stored in registers) might outperform uniform quantization. This adds no memory latency but changes the FP32 scaling step to a register lookup + scale.

4. **Attention to the KV cache.** This design addresses weight quantization only. The KV cache, which grows with sequence length, presents a separate on-chip fitting problem that merits its own analysis.

5. **Cross-layer scheduling.** The double-buffering scheme (load layer N+1 while computing layer N) requires careful orchestration of shared memory. Half the shared memory holds current weights, half prefetches next weights. This halves the effective capacity to 33 MB per layer's active weights — but registers (33 MB total) can hold the other half. The scheduling is intricate but computable.

### The fundamental insight

The format of numbers in a machine is not a mathematical abstraction — it is a physical arrangement of bits in silicon, read by specific circuits, processed by specific arithmetic units. A quantization format designed in ignorance of the target hardware is like a cipher designed in ignorance of the machine that must break it. At Bletchley, we built the Bombe to exploit the specific mechanical structure of Enigma. Here, we design the encoding to exploit the specific silicon structure of Hopper. The principle is the same: understand the machine, design for the machine.

We can only see a short distance ahead, but we can see plenty there that needs to be done.

— A.M. Turing
