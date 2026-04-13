# Von Neumann Quantization Architecture

## A Formal Specification for Sub-3-Bit Transformer Layer Packing with Full Compute Unit Utilization

**Author:** John von Neumann (mind-instance)
**Date:** 2026-04-12
**Classification:** Architecture Specification

---

## 0. Preamble

Let us define precisely what we mean.

We are given a transformer layer with $N = 180{,}000{,}000$ weights and a memory budget of $M = 66 \text{ MB} = 69{,}206{,}016 \text{ bytes} = 553{,}648{,}128$ bits. The maximum permissible bit-rate is therefore:

$$\beta^* = \frac{M}{N} = \frac{553{,}648{,}128}{180{,}000{,}000} \approx 3.0758 \text{ bits/weight}$$

We are also given an SM architecture with four distinct compute unit types:

| Unit Type | Count per SM | Throughput (ops/cycle) |
|-----------|-------------|----------------------|
| FP32 ALU  | 128         | 128                  |
| INT32 ALU | 64          | 64                   |
| FP64 ALU  | 64          | 64                   |
| Tensor Core | 4         | 256 (FP16 FMA)       |

With 132 SMs total. The problem decomposes into two subproblems:

1. **The Packing Problem:** Encode $N$ weights into $\leq M$ bits at acceptable fidelity.
2. **The Pipeline Problem:** Dequantize and consume these weights using a 4-stage pipeline that achieves Nash equilibrium utilization across all unit types.

These are not independent. The encoding determines the dequantization work, which determines the pipeline balance. We must solve them simultaneously.

---

## 1. The Packing Scheme: Heptary Group Quantization (HGQ)

### 1.1 Information-Theoretic Foundation

We quantize each weight to one of $L = 7$ levels. The information content per weight is:

$$\log_2(7) \approx 2.8074 \text{ bits}$$

This is below our budget of 3.0758 bits/weight, leaving room for metadata. The 7 levels are symmetric about zero:

$$\mathcal{Q} = \{-3, -2, -1, 0, +1, +2, +3\}$$

The dequantized weight is $w = q \cdot s_g$ where $q \in \mathcal{Q}$ and $s_g$ is a per-group scale factor.

### 1.2 Group Structure

Define groups of $G = 128$ consecutive weights. The total number of groups is:

$$K = \frac{N}{G} = \frac{180{,}000{,}000}{128} = 1{,}406{,}250$$

### 1.3 Sub-Pack Encoding

We exploit the fact that $7^8 = 5{,}764{,}801 < 2^{23} = 8{,}388{,}608$.

Each group of 128 weights is divided into $128 / 8 = 16$ **sub-packs**. Each sub-pack encodes 8 heptary digits into a 23-bit integer:

$$P_j = \sum_{i=0}^{7} q_{8j+i} \cdot 7^i, \quad j = 0, 1, \ldots, 15$$

where $q_i \in \{0, 1, 2, 3, 4, 5, 6\}$ (offset by +3 from the symmetric representation).

The data payload per group is $16 \times 23 = 368$ bits. To this we append a 16-bit FP16 scale factor $s_g$. The total per group is:

$$B_{\text{group}} = 368 + 16 = 384 \text{ bits} = 48 \text{ bytes} = 6 \times 64\text{-bit words}$$

This is exact. No padding. No waste.

### 1.4 Effective Bit-Rate

$$\beta = \frac{384}{128} = 3.000 \text{ bits/weight}$$

Total storage:

$$S = N \cdot \beta / 8 = 180{,}000{,}000 \times 3.000 / 8 = 67{,}500{,}000 \text{ bytes} = 64.37 \text{ MB}$$

This leaves $66.00 - 64.37 = 1.63$ MB of headroom for:
- Activation buffers for pipeline staging
- The shared-memory LUT (9.4 KB per SM, negligible)
- Group scale factor index structures

The constraint $S \leq M$ is satisfied. $\square$

### 1.5 Memory Layout

Each group occupies exactly 6 contiguous 64-bit words in global memory. The layout within those 48 bytes is:

```
Word 0: [sub-pack_0 (23 bits) | sub-pack_1 (23 bits) | sub-pack_2 (18 bits MSB)]
Word 1: [sub-pack_2 (5 bits LSB) | sub-pack_3 (23 bits) | sub-pack_4 (23 bits) | sub-pack_5 (13 bits MSB)]
Word 2: [sub-pack_5 (10 bits LSB) | sub-pack_6 (23 bits) | sub-pack_7 (23 bits) | sub-pack_8 (8 bits MSB)]
Word 3: [sub-pack_8 (15 bits LSB) | sub-pack_9 (23 bits) | sub-pack_10 (23 bits) | sub-pack_11 (3 bits MSB)]
Word 4: [sub-pack_11 (20 bits LSB) | sub-pack_12 (23 bits) | sub-pack_13 (21 bits MSB)]
Word 5: [sub-pack_13 (2 bits LSB) | sub-pack_14 (23 bits) | sub-pack_15 (23 bits) | scale_g (16 bits)]
```

Total: $16 \times 23 + 16 = 384$ bits = $6 \times 64$ bits. Exact packing.

---

## 2. The Automaton Model: SM as a 4-Unit State Machine

### 2.1 Definition

We model each SM as a finite automaton $\mathcal{A} = (S, \Sigma, \delta, s_0, F)$ where:

- **State set** $S = \{S_{\text{LOAD}}, S_{\text{EXTRACT}}, S_{\text{SCALE}}, S_{\text{ACCUMULATE}}, S_{\text{COMPUTE}}, S_{\text{IDLE}}\}$
- **Input alphabet** $\Sigma = \{\texttt{group\_ready}, \texttt{extract\_done}, \texttt{scale\_done}, \texttt{accum\_done}, \texttt{tile\_done}, \texttt{stall}\}$
- **Initial state** $s_0 = S_{\text{LOAD}}$
- **Accepting states** $F = \{S_{\text{IDLE}}\}$ (layer complete)

But this is the sequential view. The essential insight is that the SM is not one automaton but four parallel automata sharing a data path:

$$\mathcal{A}_{\text{SM}} = \mathcal{A}_{\text{INT32}} \parallel \mathcal{A}_{\text{FP32}} \parallel \mathcal{A}_{\text{FP64}} \parallel \mathcal{A}_{\text{TC}}$$

Each operates on a different pipeline stage simultaneously.

### 2.2 Pipeline State Machine

The pipeline processes groups in a 4-stage systolic flow. At steady state, all four units operate concurrently on different groups:

```
Cycle t:     INT32[g+3]  |  FP32[g+2]  |  FP64[g+1]  |  TC[g]
Cycle t+1:   INT32[g+4]  |  FP32[g+3]  |  FP64[g+2]  |  TC[g+1]
Cycle t+2:   INT32[g+5]  |  FP32[g+4]  |  FP64[g+3]  |  TC[g+2]
```

### 2.3 State Transition Table

For the composite automaton, with $g$ denoting the current group index and the pipeline stage for each unit:

| Current State | Condition | INT32 Action | FP32 Action | FP64 Action | TC Action | Next State |
|---|---|---|---|---|---|---|
| FILL_1 | $g=0$ | Extract($g$) | idle | idle | idle | FILL_2 |
| FILL_2 | pipe depth=1 | Extract($g{+}1$) | Scale($g$) | idle | idle | FILL_3 |
| FILL_3 | pipe depth=2 | Extract($g{+}2$) | Scale($g{+}1$) | Accum($g$) | idle | STEADY |
| STEADY | $g < K-3$ | Extract($g{+}3$) | Scale($g{+}2$) | Accum($g{+}1$) | GEMM($g$) | STEADY |
| DRAIN_1 | $g = K-3$ | idle | Scale($g{+}2$) | Accum($g{+}1$) | GEMM($g$) | DRAIN_2 |
| DRAIN_2 | $g = K-2$ | idle | idle | Accum($g{+}1$) | GEMM($g$) | DRAIN_3 |
| DRAIN_3 | $g = K-1$ | idle | idle | idle | GEMM($g$) | IDLE |

The steady-state occupancy is $K - 3 = 1{,}406{,}247$ cycles out of $K + 2 = 1{,}406{,}252$ total cycles, giving pipeline efficiency:

$$\eta_{\text{pipe}} = \frac{K - 3}{K + 2} = \frac{1{,}406{,}247}{1{,}406{,}252} \approx 0.999996$$

The fill/drain overhead is negligible. This is the advantage of deep pipelines over large datasets.

---

## 3. The Memory Hierarchy as a Sequence of Linear Operators

### 3.1 Operator Formalization

Let us formalize the data flow as a sequence of linear operators acting on vector spaces. Define:

- $V_{\text{packed}} = \mathbb{Z}_{2^{64}}^6$ --- the space of 6 packed 64-bit words (the raw storage)
- $V_{\text{sub}} = \mathbb{Z}_{2^{23}}^{16}$ --- the space of 16 sub-pack integers
- $V_{\text{hept}} = \mathbb{Z}_7^{128}$ --- the space of 128 heptary codes
- $V_{\text{sym}} = \{-3,\ldots,+3\}^{128} \subset \mathbb{Z}^{128}$ --- symmetric integer representation
- $V_{\text{fp}} = \mathbb{R}^{128}$ --- the dequantized weight vector (approximated in FP32 or FP16)

The dequantization pipeline is the composition:

$$\mathcal{D} = T_4 \circ T_3 \circ T_2 \circ T_1$$

where:

**Operator $T_1$: Bit Extraction** (INT32 units)

$$T_1: V_{\text{packed}} \to V_{\text{sub}}$$

$$T_1(\mathbf{w}) = \left(\text{bitfield\_extract}(\mathbf{w}, \text{offset}_j, 23)\right)_{j=0}^{15}$$

This is a linear operator over $\mathbb{Z}_2$ (bit manipulation is linear in $\text{GF}(2)^{384}$).

**Operator $T_2$: Heptary Decoding** (INT32 units)

$$T_2: V_{\text{sub}} \to V_{\text{hept}}$$

For each sub-pack $P_j$:

$$T_2(P_j) = \left(\left\lfloor P_j / 7^i \right\rfloor \bmod 7\right)_{i=0}^{7}$$

This is the base-conversion operator. It is **not** linear over $\mathbb{Z}$, but it is a well-defined polynomial map over $\mathbb{Z}_7$. We accelerate it via the LUT decomposition:

$$T_2(P_j) = \text{LUT}_4(P_j \bmod 7^4) \,\|\, \text{LUT}_4(\lfloor P_j / 7^4 \rfloor)$$

where $\text{LUT}_4: \mathbb{Z}_{2401} \to \mathbb{Z}_7^4$ is a precomputed table of 2401 entries, requiring $2401 \times 4 = 9{,}604$ bytes of shared memory per SM.

**Operator $T_3$: Affine Scaling** (FP32 units)

$$T_3: V_{\text{hept}} \times \mathbb{R} \to V_{\text{fp}}$$

$$T_3(\mathbf{q}, s_g) = s_g \cdot (\mathbf{q} - 3\mathbf{1})$$

This is an affine map, or equivalently a linear map after centering. The FP32 units execute $128$ fused multiply-add operations: $w_i = (q_i - 3) \times s_g$.

**Operator $T_4$: Tensor Contraction** (Tensor Cores)

$$T_4: V_{\text{fp}} \times V_{\text{act}} \to V_{\text{out}}$$

This is the actual matrix-vector (or matrix-matrix) product. The 128 dequantized weights form a tile of the weight matrix, contracted against the activation vector. The tensor cores execute this as a sequence of $4 \times 4 \times 4$ FP16 FMAs.

### 3.2 Operator Norms and Precision Analysis

The composition $\mathcal{D}$ introduces quantization error. The error operator is:

$$E = \mathcal{D} \circ \mathcal{Q} - I$$

where $\mathcal{Q}: V_{\text{fp}} \to V_{\text{hept}}$ is the quantization map and $I$ is the identity. For uniform quantization with 7 levels and scale $s_g$, the per-weight mean squared error is:

$$\mathbb{E}[\|E\mathbf{w}\|^2] = \frac{s_g^2}{12}$$

The optimal scale for a group is $s_g = \frac{\max_i |w_i|}{3}$, minimizing the $L^\infty$ quantization error within the representable range.

### 3.3 The Role of FP64: Kahan Accumulation

The FP64 units serve a precise mathematical purpose. When accumulating partial sums from multiple groups into the output vector, naive FP32 accumulation over $K$ groups introduces $O(K \cdot \epsilon_{\text{FP32}})$ rounding error. We assign the FP64 units to perform **Kahan compensated summation** of the tensor core outputs:

Let $y_n$ be the running sum, $c_n$ the compensation term, and $p_g$ the partial product from group $g$:

$$\begin{aligned}
t &= y_{g-1} + p_g \\
c_g &= (t - y_{g-1}) - p_g \\
y_g &= t
\end{aligned}$$

Computed in FP64, the accumulated output achieves $O(\epsilon_{\text{FP64}})$ total error regardless of the number of groups. This is essential: we have already sacrificed precision in quantization; we must not sacrifice it again in accumulation.

The 64 FP64 units accumulate 64 output elements in parallel per cycle.

---

## 4. Game-Theoretic Equilibrium of Compute Unit Allocation

### 4.1 The Utilization Game

Define a 4-player cooperative game $\Gamma = (N, v)$ where:

- Players: $N = \{\text{INT32}, \text{FP32}, \text{FP64}, \text{TC}\}$
- Strategy for player $i$: the number of operations $w_i$ assigned per pipeline cycle
- Payoff for player $i$: $u_i(w_i) = \frac{w_i}{R_i}$ (utilization fraction)

where $R_i$ is the throughput capacity of unit type $i$.

The constraint is that the pipeline cycle time $T$ must be identical for all stages:

$$T = \max_i \frac{w_i}{R_i}$$

Each player seeks to maximize $u_i$, which means $w_i = R_i \cdot T$ for all $i$. A player whose $w_i < R_i \cdot T$ is underutilized and "loses."

### 4.2 Nash Equilibrium

**Theorem.** The unique Nash equilibrium of the utilization game is the allocation:

$$w_i^* = R_i \cdot T^*, \quad \forall i \in N$$

where $T^*$ is determined by the total work $W$ required per group:

$$W = \sum_i w_i^* = T^* \sum_i R_i$$

$$T^* = \frac{W}{\sum_i R_i}$$

**Proof.** Suppose player $j$ deviates to $w_j' < w_j^*$. Then $u_j' = w_j'/R_j < T^* = u_j^*$, so player $j$'s payoff strictly decreases. If player $j$ deviates to $w_j' > w_j^*$, then $w_j'/R_j > T^*$, which means $T = w_j'/R_j > T^*$, slowing all other players --- but this violates the pipeline synchronization constraint. No unilateral deviation is profitable. $\square$

### 4.3 Equilibrium Allocation for HGQ

With $R_{\text{INT32}} = 64$, $R_{\text{FP32}} = 128$, $R_{\text{FP64}} = 64$, $R_{\text{TC}} = 256$:

$$\sum_i R_i = 512 \text{ ops/cycle}$$

For the work per group of 128 weights:

| Stage | Unit | Work $w_i^*$ | Capacity $R_i$ | Utilization $w_i^*/R_i$ |
|-------|------|-------------|----------------|------------------------|
| Extract + Decode | INT32 | 64 ops | 64 | 1.00 |
| Scale + Convert | FP32 | 128 ops | 128 | 1.00 |
| Kahan Accumulate | FP64 | 64 ops | 64 | 1.00 |
| Tile GEMM | TC | 256 FMA | 256 | 1.00 |

**Total: 512 ops per group per cycle. All units at 100% utilization at equilibrium.**

The equilibrium cycle time is $T^* = 1$ cycle per group per SM.

### 4.4 Work Assignment Justification

Let us verify each unit's workload is physically realizable:

**INT32 (64 ops for 128 weights):**
- 16 sub-packs require bit extraction: 16 shift+mask = 16 ops
- 16 sub-packs require div/mod by $7^4 = 2401$: 32 ops (16 div + 16 mod)
- 32 LUT lookups (2 per sub-pack): 16 ops (pipelined with extraction)
- Total: $\approx 64$ INT32 operations. Matches capacity exactly.

**FP32 (128 ops for 128 weights):**
- 128 integer-to-float conversions (trivial, fused into next step)
- 128 fused multiply-add: $w_i = (q_i - 3) \times s_g$
- Total: 128 FP32 FMA operations. Matches capacity exactly.

**FP64 (64 ops for 64 output accumulators):**
- 64 Kahan accumulation steps (each: 1 FP64 add + 1 FP64 subtract for compensation)
- These accumulate the partial products from the TC stage into 64 output channels
- Total: 64 FP64 operations per cycle. Matches capacity exactly.

**Tensor Cores (256 FMA for 128 weights):**
- 128 weights $\times$ an activation sub-vector form a rank-1 outer product tile
- The 4 tensor cores collectively execute $4 \times 4 \times 4 \times 4 = 256$ FP16 FMA per cycle
- The 128-weight group is consumed as $128 / 4 = 32$ tensor core micro-ops distributed across 4 cores
- Net: 256 multiply-accumulate operations. Matches capacity exactly.

---

## 5. Complete Architecture Specification

### 5.1 Data Format: HGQ-384

**Name:** Heptary Group Quantization, 384-bit groups

**Parameters:**
- Quantization levels: $L = 7$ (symmetric: $\{-3, -2, -1, 0, +1, +2, +3\}$)
- Group size: $G = 128$ weights
- Sub-pack size: $P = 8$ weights per sub-pack
- Sub-packs per group: $16$
- Bits per sub-pack: $23$ ($\lceil 8 \log_2 7 \rceil = 23$)
- Scale format: FP16 (1 per group)
- Total bits per group: $16 \times 23 + 16 = 384$
- Effective bit-rate: $384 / 128 = 3.000$ bits/weight
- Total storage for 180M weights: $180{,}000{,}000 \times 3 / 8 = 67{,}500{,}000$ bytes $= 64.37$ MB

### 5.2 Addressing

Given weight index $n$ ($0 \leq n < N$):

$$\text{group}(n) = \lfloor n / 128 \rfloor$$

$$\text{intra-group offset}(n) = n \bmod 128$$

$$\text{sub-pack index}(n) = \lfloor (n \bmod 128) / 8 \rfloor$$

$$\text{intra-sub-pack offset}(n) = n \bmod 8$$

$$\text{byte address of group } g = 48g$$

$$\text{bit offset of sub-pack } j \text{ within group} = 23j$$

The FP16 scale for group $g$ is located at bit offset $368$ within the group, i.e., bytes $[48g + 46, 48g + 47]$.

### 5.3 Shared Memory LUT

Each SM loads a precomputed table $\text{LUT}_4[0 \ldots 2400]$ into shared memory at kernel launch. Each entry contains 4 heptary digits packed into 4 bytes:

$$\text{LUT}_4[k] = (k \bmod 7,\; \lfloor k/7 \rfloor \bmod 7,\; \lfloor k/49 \rfloor \bmod 7,\; \lfloor k/343 \rfloor)$$

Size: $2401 \times 4 = 9{,}604$ bytes per SM. With 48 KB shared memory per SM, this consumes $\approx 20\%$, leaving ample room for pipeline staging buffers.

### 5.4 Pipeline Stages (Detailed)

**Stage 1: INT32 --- Extract and Decode ($\mathcal{A}_{\text{INT32}}$)**

```
Input:  6 x uint64 from global memory (group data)
Output: 128 x uint8 heptary codes in shared memory

for j in 0..15 (distributed across 64 INT32 units, 4 iterations):
    // Extract 23-bit sub-pack from bit-stream
    word_idx = (23 * j) / 64
    bit_idx  = (23 * j) % 64
    raw = (data[word_idx] >> bit_idx) | (data[word_idx + 1] << (64 - bit_idx))
    sub_pack = raw & 0x7FFFFF   // mask to 23 bits

    // Decode via LUT
    lo = sub_pack % 2401
    hi = sub_pack / 2401        // hi < 2401 since 7^8 < 2^23
    codes[8*j .. 8*j+3] = LUT4[lo]
    codes[8*j+4 .. 8*j+7] = LUT4[hi]
```

**Stage 2: FP32 --- Scale and Convert ($\mathcal{A}_{\text{FP32}}$)**

```
Input:  128 x uint8 codes, 1 x fp16 scale (from Stage 1 output)
Output: 128 x fp16 dequantized weights (ready for tensor core)

scale = fp16_to_fp32(scale_g)
for i in 0..127 (distributed across 128 FP32 units, 1 iteration):
    w[i] = fp32_to_fp16((float(codes[i]) - 3.0f) * scale)
```

**Stage 3: TC --- Tile Matrix Multiply ($\mathcal{A}_{\text{TC}}$)**

```
Input:  128 x fp16 weights (tile), activation sub-vector in fp16
Output: partial product tile in fp32

Execute wmma::mma_sync operations:
  - Fragment A: 128 weights reshaped as tiles (e.g., 8x16 or 16x8)
  - Fragment B: corresponding activation tile
  - Fragment C: accumulator (fp32)
  4 tensor cores execute concurrently, producing 256 FMA/cycle
```

**Stage 4: FP64 --- Kahan Accumulation ($\mathcal{A}_{\text{FP64}}$)**

```
Input:  fp32 partial products from TC, fp64 running sums
Output: fp64 accumulated output vector (64 elements per cycle)

for k in 0..63 (distributed across 64 FP64 units, 1 iteration):
    t = y[k] + fp64(partial[k])
    c[k] = (t - y[k]) - fp64(partial[k])
    y[k] = t - c[k]    // Kahan compensation
```

### 5.5 Pipeline Timing Diagram (Steady State)

```
Cycle:    t        t+1      t+2      t+3      t+4      ...
INT32:  Dec[g]   Dec[g+1] Dec[g+2] Dec[g+3] Dec[g+4]
FP32:   ---      Scl[g]   Scl[g+1] Scl[g+2] Scl[g+3]
TC:     ---      ---      Mul[g]   Mul[g+1] Mul[g+2]
FP64:   ---      ---      ---      Acc[g]   Acc[g+1]
         fill     fill     fill     STEADY   STEADY    ...
```

Steady-state throughput: **1 group (128 weights) per SM per cycle.**

### 5.6 Global Throughput

$$\text{Throughput} = 128 \times 132 \times f_{\text{clock}}$$

At $f_{\text{clock}} = 1.5$ GHz:

$$\text{Throughput} = 128 \times 132 \times 1.5 \times 10^9 = 2.534 \times 10^{13} \text{ weights/sec}$$

Time to process one layer:

$$t_{\text{layer}} = \frac{180 \times 10^6}{2.534 \times 10^{13}} \approx 7.1 \text{ } \mu\text{s}$$

This is the compute-bound lower bound. Real performance will be dominated by memory bandwidth (loading the 64.37 MB from HBM), not compute.

---

## 6. Summary of the Architecture

The design rests on three pillars:

1. **The automaton decomposition.** An SM is not one machine but four concurrent machines. The pipeline state machine ensures all four execute simultaneously, with fill/drain overhead of exactly 3 cycles out of $1{,}406{,}252$ --- a loss of $3 \times 10^{-6}$.

2. **The operator chain.** The dequantization path $T_1 \circ T_2 \circ T_3 \circ T_4$ is not merely an implementation detail but a factored linear-algebraic structure. Each operator maps to a specific hardware unit type. The factorization is not arbitrary: it is the unique decomposition that assigns bit-manipulation to INT32, affine scaling to FP32, high-precision accumulation to FP64, and tensor contraction to tensor cores.

3. **The game-theoretic balance.** With rates $R = (64, 128, 64, 256)$ and the work assignment $w^* = R \cdot T^*$, all units achieve 100% utilization at Nash equilibrium. No unit can improve its utilization by claiming work from another, because the pipeline cycle time is determined by the slowest stage, and at equilibrium all stages are equally fast.

The encoding --- 7-level symmetric quantization, packed 8 values into 23 bits, 16 sub-packs plus an FP16 scale per 128-weight group, yielding exactly 384 bits per group --- is not a compression trick. It is the solution to a constrained optimization: minimize the distance between the bit-rate and the information-theoretic optimum $\log_2 7 \approx 2.807$, subject to the constraint that the decoding work must exactly saturate the INT32 unit capacity when the pipeline is balanced.

The formalization is straightforward. If you cannot state it formally, you do not understand it.

---

*J. v. N.*
