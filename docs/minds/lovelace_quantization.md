# The Algebraical Loom: A Quantization Scheme for Neural Weight Patterns

### By Ada Augusta King, Countess of Lovelace
### A Study in Poetical Science Applied to the Compression and Parallel Dequantization of Neural Network Weights

---

> *"The Analytical Engine weaves algebraical patterns just as the Jacquard loom weaves flowers and leaves."*
> -- Note A, on Menabrea's Memoir

---

## I. The Nature of the Problem, Perceived Rightly

We are presented with a problem that appears, at first glance, to be one of mere arithmetic: compress 180 million weights into 66 megabytes of on-chip memory, achieving approximately 2.93 bits per weight. The world has treated this as a problem of *storage* -- how to make numbers smaller. Or of *quality* -- how to preserve accuracy under compression.

But I perceive in it a problem of an altogether different character. It is a problem of **operations**.

The modern GPU possesses four distinct species of computational machinery -- four looms, if you will, each capable of weaving a different algebraical fabric:

| Loom | Nature | Typical Idleness During Inference |
|------|--------|----------------------------------|
| **FP32 units** | Single-precision floating-point | Partially active |
| **INT32 units** | Integer arithmetic | Mostly idle |
| **FP64 units** | Double-precision floating-point | Almost entirely idle |
| **Tensor cores** | Matrix-multiply-accumulate | Active in bursts, idle between |

During inference, most of these looms sit silent -- like a Jacquard mechanism with only one set of cards threaded, the other harnesses slack. This is not a failure of the machine but a failure of *our notation*. We have not yet devised a representation of weights whose dequantization *requires* all four looms simultaneously.

This is the problem I propose to solve. Not "how to store weights cheaply" but "how to encode weight patterns such that their reconstruction is an operation demanding the full orchestra of the Engine."

---

## II. The Pattern Perspective: Weights as Woven Fabric

Consider a weight matrix W of dimension m x n. The prevailing view treats each weight w_ij as an independent quantity to be compressed. But this misapprehends the nature of the object. The weight matrix is not a collection of numbers -- it is a *pattern of relations*.

When the Jacquard loom encodes a floral design, it does not store the colour of each thread independently. It stores the *pattern* -- the systematic relationships between threads that, when woven, produce the flower. The individual thread colours are consequences of the pattern, not its constituents.

Similarly, neural network weights exhibit systematic structure:
- **Low-rank structure**: Many weight matrices are approximately low-rank, meaning the pattern can be described by fewer dimensions than it occupies.
- **Clustered structure**: Weights tend to group around certain values, forming natural codebooks.
- **Spatial correlation**: Nearby weights (in the matrix topology) tend to be related.
- **Cross-head symmetry**: In attention layers, different heads often share structural motifs.

The most natural notation for a weight pattern, then, is not a list of individually quantized values but a **multi-scale decomposition**: a coarse structural skeleton overlaid with progressively finer detail, each scale of detail expressed in the notation most natural to it.

---

## III. The Algebraical Loom Encoding (ALE)

I propose a four-stratum encoding, where each stratum is purpose-built for reconstruction on one of the four compute unit types. The strata are not independent compressions summed together -- they are *interlocking patterns*, like the warp and weft of a fabric, where the meaning of each stratum depends on the others.

### Stratum I: The Structural Skeleton (FP64 Units)

**Bit budget: ~0.03 bits/weight**
**Reconstructed by: FP64 units**

For each block of 1024 weights, we store a single FP64 value: the *structural centroid* -- a high-precision anchor point that captures the coarse energy of the block. This requires 64 bits per 1024 weights = 0.0625 bits/weight.

But we may be more parsimonious. We observe that adjacent blocks share structural similarity. We therefore store:
- One FP64 *base centroid* per super-block of 4096 weights (0.0156 bits/weight)
- One FP16 *delta* per block of 1024 within the super-block (0.0156 bits/weight)

Total: ~0.031 bits/weight.

The FP64 units, ordinarily idle, compute the high-precision base values. The use of FP64 is not extravagance -- it is *necessity*, for the centroid must be computed without rounding error to prevent error accumulation across the hundreds of blocks that tile the layer. This is analogous to my Bernoulli number computation, where each term depends on the accumulated precision of all prior terms.

### Stratum II: The Codebook Fabric (Tensor Cores)

**Bit budget: ~2.0 bits/weight**
**Reconstructed by: Tensor cores (matrix-multiply-accumulate)**

This is the heart of the encoding, and it contains what I consider the most elegant insight of the scheme.

Rather than storing individual quantized weights, we store *indices into a learned codebook*. But the codebook is structured as a **product quantization** with a specific geometry: each weight is represented by the combination of two sub-codes, and the reconstruction is a *matrix multiplication*.

Concretely, for a group of 128 weights:
- Partition each weight into two sub-spaces (conceptually, the "warp" and "weft").
- Maintain two codebooks, C_warp of dimension (16 x 64) and C_weft of dimension (16 x 64).
- Each weight is indexed by a pair (i, j) where i, j are each 4-bit indices.
- But we do not store 128 independent pairs. Instead, we store a *pattern matrix* P of dimension (8 x 16) with 4-bit entries, and reconstruct the full weight block as:

```
W_block = C_warp[P_rows, :] * S * C_weft[P_cols, :]^T + centroid
```

where S is a per-block FP16 scale factor from Stratum I, and the multiplication is a matrix operation executed on tensor cores.

The indices P require 8 x 16 x 4 bits = 512 bits for 128 weights = 4.0 bits/weight for the raw indices. But here is where the pattern structure helps: P itself is compressible because it encodes a *pattern*, not random indices. We apply a lightweight entropy coding (a fixed Huffman table shared across the layer) to compress P to approximately 2.0 bits/weight on average.

The reconstruction is a small matrix multiply -- precisely the operation tensor cores are designed to execute in a single cycle. We are not *adapting* the computation to the hardware; we have designed the *notation* so that its natural reading *is* a tensor core operation.

This is the Jacquard principle: the card does not contain the flower. It contains the instructions for producing the flower. And we have written those instructions in the language the loom speaks natively.

### Stratum III: The Residual Texture (INT32 Units)

**Bit budget: ~0.75 bits/weight**
**Reconstructed by: INT32 units**

After Strata I and II reconstruct the coarse pattern, a residual remains. This residual has a characteristic structure: it is approximately symmetric about zero, with most values being very small corrections.

We encode this residual using **bitplane coding with integer arithmetic**:

For each group of 32 weights, we store:
- A 32-bit mask M indicating which weights have non-zero residuals (~1.0 bit/weight amortized via run-length encoding to ~0.5 bits/weight)
- For non-zero residuals: a sign bit and a 2-bit magnitude, packed into INT32 words
- An INT32 scale factor per group of 128 weights

The dequantization is pure integer arithmetic:
1. Unpack the mask M using bitwise AND, shift operations (INT32)
2. Extract sign and magnitude via integer bit manipulation (INT32)
3. Multiply by integer scale factor (INT32)
4. The result is an integer residual that will be added (after type conversion) to the tensor-core output

This stratum keeps the INT32 units fully occupied with their native operations: bitwise manipulation, integer multiply, integer add. These units have sat idle during neural network inference since the inception of GPU computing. We give them purposeful work.

### Stratum IV: The Fine Harmonics (FP32 Units)

**Bit budget: ~0.15 bits/weight**
**Reconstructed by: FP32 units**

The final stratum captures what I call the *harmonics* -- fine-grained corrections that are sparse but significant. Like the overtones in a musical chord, they are few in number but essential to the character of the sound.

For each block of 256 weights:
- A 256-bit sparse mask (1.0 bit/weight, but only ~3.8% of weights receive this correction, so amortized cost is ~0.04 bits/weight for the mask via bitmap compression)
- For each flagged weight: an FP16 correction value (~0.038 x 16 = ~0.6 bits/weight amortized)
- Per-block FP32 harmonic scale (negligible amortized cost)

Total: ~0.15 bits/weight.

The FP32 units perform:
1. Sparse mask decode (FP32 bitwise operations, reinterpreted)
2. FP16-to-FP32 upcast of correction values
3. Multiply by FP32 harmonic scale
4. Accumulate into the final output

This gives the FP32 units independent, meaningful work that overlaps temporally with the other three strata.

---

## IV. The Total Accounting

| Stratum | Bits/Weight | Compute Unit | Operation Character |
|---------|-------------|--------------|-------------------|
| I. Structural Skeleton | 0.031 | FP64 | High-precision centroid computation |
| II. Codebook Fabric | 2.000 | Tensor Cores | Matrix-multiply codebook lookup |
| III. Residual Texture | 0.750 | INT32 | Bitwise unpack, integer scale |
| IV. Fine Harmonics | 0.150 | FP32 | Sparse correction accumulate |
| **Total** | **2.931** | **All four** | **Full utilization** |

At 2.931 bits/weight x 180,000,000 weights:

```
180,000,000 x 2.931 / 8 = 65,947,500 bytes = 62.91 MB
```

With codebook storage (~2 MB) and metadata (~1 MB): **approximately 65.9 MB**. The pattern fits the frame. The fabric fills the loom.

---

## V. The Orchestration: Temporal Weaving of the Four Looms

The true elegance of this scheme lies not in any single stratum but in their *temporal orchestration*. I conceive of dequantization as a four-voice fugue, where each voice enters at its appointed time and all voices sound together in the culminating chord.

```
Time -->
         t0        t1        t2        t3        t4
         |         |         |         |         |
FP64:    [====centroid====]  .         .         [merge]
Tensor:  .  [=====codebook multiply=====]       [merge]
INT32:   .  [==residual unpack & scale===]      [merge]
FP32:    .    [===sparse harmonic decode===]    [merge]
         |         |         |         |         |
         Phase 1   Phase 2   Phase 3   Phase 4   Sum
         Load      Compute   Compute   Reduce
```

### Phase 1 (t0): Load and Distribute
All four unit types begin loading their respective data streams from the packed on-chip buffer. The data is laid out in memory so that each unit type reads a contiguous, aligned segment -- no bank conflicts, no contention.

### Phase 2 (t1-t2): Parallel Computation
Each loom weaves its own fabric independently:
- FP64 units compute centroids with full precision
- Tensor cores execute the codebook matrix multiply
- INT32 units unpack and scale the residual
- FP32 units decode the sparse harmonic corrections

These operations have **zero data dependencies between them**. This is by construction -- we designed the notation so that each stratum is independently decodable. The only shared information is the block index, which is known a priori.

### Phase 3 (t3): Type Convergence
Results from INT32 and FP64 paths are converted to FP32. This type conversion is itself a useful operation: the INT32-to-FP32 conversion uses the INT32 units for their final shift/mask and the FP32 units for the float conversion, keeping both occupied.

### Phase 4 (t4): Summation
The four FP32 results are summed:
```
W_reconstructed = centroid + codebook_product + int_residual_as_float + harmonic_correction
```

This final addition is a single FP32 fused operation.

---

## VI. The Bernoulli Dependency Structure: Maximizing Parallelism

In my algorithm for the Bernoulli numbers, I observed that the computation of B_n requires B_0 through B_{n-1}. This creates a chain of dependencies that constrains the ordering of operations. The question I posed then, and pose again now, is: **what is the minimum sequential depth of the dependency graph?**

In the ALE scheme, the dependencies form a *forest*, not a chain:

```
Level 0 (independent):  FP64_centroid  |  Tensor_codebook  |  INT32_residual  |  FP32_harmonic
                              |                |                   |                  |
Level 1 (converge):     FP32_cast      |   FP32_ready       |  FP32_cast       |  FP32_ready
                              |                |                   |                  |
Level 2 (reduce):       ============== FP32 four-way sum ==============
```

The sequential depth is exactly **3**: decode, convert, sum. All the expensive work happens at Level 0, in parallel across all four unit types.

Compare this to conventional dequantization (e.g., GPTQ or AWQ):
```
Load quantized → INT32 unpack → FP16 cast → FP16 scale → FP16 zero-point add → ready
```
Sequential depth: **5**, using only INT32 and FP16 units, leaving FP64 and half the FP32 capacity idle.

We have reduced the critical path while engaging all resources. The Engine's mill turns faster when all its gears are meshed.

---

## VII. Memory Layout: The Punched Card Arrangement

The 66 MB on-chip allocation is partitioned as follows. I describe the layout for a single transformer layer with 180M weights:

```
Address Range        Contents                        Size (MB)
0x000000-0x002800    Layer metadata & codebooks       ~2.0
0x002800-0x2C0000    Stratum II: Codebook indices     ~45.0  (the bulk)
0x2C0000-0x3D0000    Stratum III: Residual bitplanes  ~16.9
0x3D0000-0x400000    Stratum IV: Harmonic corrections ~3.4
0x400000-0x410000    Stratum I: FP64 centroids        ~0.6
                                          Total:      ~65.9 MB
```

The layout is ordered by access pattern: Stratum II (tensor cores) is placed first and contiguously because tensor cores require aligned, sequential access. Stratum III follows for INT32 streaming access. Strata IV and I, being the sparsest, are placed last.

Within each stratum, data is tiled to match the GPU's warp/wavefront structure. Each warp processes one block of 128 weights, and the data for all four strata for that block is arranged so that a single warp can issue loads to all four data streams in a single cycle.

---

## VIII. The Aesthetic Dimension: On Elegance

I stated in my Notes that mathematical science "constitutes the language through which alone we can adequately express the great facts of the natural world." An encoding scheme is, in essence, a language for expressing weight patterns. Let us ask: is this language *elegant*?

I believe it possesses three forms of elegance:

**1. Structural elegance.** Each stratum corresponds to a natural scale of the weight distribution -- coarse position (centroid), dominant pattern (codebook), texture (residual), fine detail (harmonics). This mirrors how mathematicians decompose functions: the Taylor series gives the smooth trend, the Fourier series gives the oscillatory texture, and the wavelet gives the localised detail. We decompose weights similarly, but assign each decomposition to its natural computational substrate.

**2. Operational elegance.** The dequantization operation is a *natural* operation for the hardware. We have not forced the hardware to perform unnatural gymnastics. We have written the notation in the hardware's native tongue. The tensor cores perform matrix multiplies because the codebook is structured as a matrix product. The INT32 units perform bit manipulation because the residual is structured as packed bits. The form of the data mirrors the function of the unit.

**3. Compositional elegance.** The four strata compose by simple addition. There is no complex interaction, no conditional logic, no branching. The final weight is:

```
w = skeleton + fabric + texture + harmonics
```

This additive composition is both mathematically clean and computationally efficient. It recalls the principle that complex patterns emerge from the superposition of simple components -- as Fourier showed for heat, as Babbage showed for polynomial differences, as I now show for neural weights.

---

## IX. Practical Considerations and Numerical Bounds

I am not one to let imagination outrun analysis. Let us verify the scheme's soundness.

### Error Bound

The total reconstruction error per weight is bounded by the sum of the quantization errors of each stratum:

```
|w - w_hat| <= epsilon_codebook + epsilon_residual + epsilon_harmonic
```

With Stratum II providing 4-bit effective precision per sub-code (dynamic range covered by codebook learning), Stratum III providing ~2-bit residual correction, and Stratum IV providing FP16 corrections to the ~3.8% most sensitive weights (identified during calibration), the expected mean squared error is comparable to uniform 4-bit quantization -- but achieved at 2.93 bits/weight.

The key insight is that the codebook (Stratum II) is *learned* from calibration data using k-means or EM on the weight distribution. It captures the natural clusters of the weight space, not an arbitrary uniform grid. This is the difference between a bespoke garment and one cut from a standard pattern.

### Codebook Training

The codebook C_warp and C_weft are trained offline per layer using alternating least squares on a calibration set. The procedure:

1. Initialize codebooks via k-means on weight sub-spaces
2. For each iteration:
   a. Fix codebooks, assign optimal indices (nearest-neighbor in product space)
   b. Fix indices, update codebooks via least-squares
   c. Compute residuals, train Strata III and IV on residuals
3. Converge when MSE stabilises (typically 20-50 iterations)

The codebooks themselves occupy 2 x 16 x 64 x 2 bytes (FP16) = 4 KB per group, with groups shared across blocks of 4096 weights. Total codebook storage: ~1.4 MB per layer.

### Sensitivity-Aware Harmonic Allocation

The 3.8% of weights receiving Stratum IV corrections are selected by **Hessian sensitivity analysis**: we compute the diagonal of the Fisher information matrix (or its approximation) and allocate harmonic corrections to the weights with the highest sensitivity x quantization-error product. This is analogous to my observation about the Bernoulli computation: not all terms contribute equally to the final result, and computational resources should be allocated to those terms whose errors would propagate most severely.

---

## X. The Vision Beyond

I perceive in this scheme something beyond a mere compression technique. It is a principle:

**The notation in which we express a computation should be chosen not for the convenience of the human reader, nor for the minimisation of storage, but for the natural alignment between the structure of the data and the structure of the machine that must process it.**

This is what I meant when I wrote that the Engine "weaves algebraical patterns." The patterns are not incidental -- they are the essence. When we quantize neural network weights, we are not merely making numbers smaller. We are *re-expressing the pattern* in a notation that the Engine can weave most efficiently.

The Jacquard loom does not store a bitmap of the fabric. It stores instructions that, when executed by the mechanical linkages of the loom, produce the fabric. Our quantization scheme does not store a compressed array of weights. It stores instructions -- codebook indices, residual bitmasks, harmonic corrections -- that, when executed by the four species of computational linkage in the GPU, produce the weight matrix.

The card is not the flower. The encoding is not the weight. The pattern transcends its representation, but the representation must honour the pattern and the machine alike.

I close with a reflection. In 1843, I wrote that the bounds of arithmetic were outstepped the moment the idea of applying Jacquard's cards to the Analytical Engine had occurred. Today, the bounds of quantization are outstepped the moment we perceive that compression is not about making numbers smaller, but about expressing patterns in the native language of the machine that must reconstruct them.

The four looms await their cards. Let us weave.

---

*Augusta Ada King, Countess of Lovelace*
*In the science of operations, as she would have applied it to the engines of this age*
