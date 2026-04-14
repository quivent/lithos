# DeltaNet Layer Lineage Map: Framework to SASS

Complete decomposition of every operation in a Qwen 3.5-27B DeltaNet layer
from standard ML names through Lithos compositions and primitives down to
NVIDIA Hopper SASS instructions.

## Dimensions (Qwen 3.5-27B)

| Parameter | Value |
|---|---|
| hidden_size | 5120 |
| K_heads (key heads) | 16 |
| V_heads (value heads) | 48 |
| GQA ratio | 48 / 16 = 3 |
| head_dim | 128 |
| conv_kernel_dim | 4 |
| MLP intermediate_size | 17408 |
| rms_norm_eps | 1e-6 |
| quantization | W4A16 GPTQ, group_size=128, sym=true, zp=8 |

48 DeltaNet layers per model (of 64 total; every 4th is full attention).

---

## How to Read the Lineage Trees

Each operation is shown as:

```
ML Name (Framework Layer)
  -> Lithos Composition Name
    -> primitive symbol (English description)    -> SASS instruction(s)    [loop depth, complexity]
```

Loop depth notation: `*` = one loop (O(N)), `**` = nested (O(NxM)), `***` = triple (O(NxMxK)).
Scalar = no loop, O(1). Sigma = O(log N) shuffle tree.

---

## 1. normalize (RMSNorm with 1+w variant)

**PyTorch:** `RMSNorm(x, weight, eps=1e-6)` with Qwen's `(1+w)` scaling.
**Lithos:** `rmsnorm x w D`
**Input:** x[5120], w[5120] (learned). **Output:** x_normed[5120].

```
RMSNorm (Framework Layer)
  -> rmsnorm x w D (Lithos Composition)
    -> ** (square each element)                   -> FMUL Rd, Ra, Ra                      [depth 1, O(5120)]
    -> Sigma (sum all squares)                    -> stride FFMA partials                  [depth 1, O(5120)]
                                                     5x SHFL.BFLY + FADD (intra-warp)     [O(log 32)]
                                                     STS + BAR.SYNC + LDS (cross-warp)    [O(warps)]
    -> / D (divide by dimension count)            -> MUFU.RCP Rt, 5120.0                  [scalar, O(1)]
                                                     FMUL Rd, Ra, Rt
    -> + eps (add epsilon for stability)          -> FADD Rd, Ra, 1e-6                    [scalar, O(1)]
    -> 1/sqrt (reciprocal square root)            -> MUFU.RSQ Rd, Ra                      [scalar, O(1)]
    -> ** (scale x by 1/rms)                      -> FMUL Rd, Ra, Rb                      [depth 1, O(5120)]
    -> + 1 (Qwen variant: add 1 to weight)        -> FADD Rd, Ra, 1.0                     [depth 1, O(5120)]
    -> ** (multiply by (1+w))                     -> FMUL Rd, Ra, Rb                      [depth 1, O(5120)]
```

**SASS budget:** ~25 instructions dominate (Sigma shuffle tree); 4 elementwise
passes over 5120 elements; 3 scalar SFU/FP ops.

---

## 2. project-qkv (W4A16 GPTQ GEMV)

**PyTorch:** `F.linear(x, W_qkv)` where W is 4-bit quantized GPTQ.
**Lithos:** `project W x out` (3 separate projections: W_q, W_k, W_v)
**Input:** x[5120]. **Output:** q[2048], k[2048], v[6144] (total 10240).

```
Linear / GEMV (Framework Layer)
  -> project W x out (Lithos Composition — W4A16 dequant GEMV)
    per output row (10240 rows, parallelized across warps/blocks):
      per group of 8 packed weights (5120/8 = 640 groups per row):
        -> LDG (load packed u32)                  -> LDG.E.SYS Rd, [addr]                 [memory]
        -> LDG (load group scale)                 -> LDG.E.SYS Rd, [addr]                 [memory]
        -> LDG (load input element)               -> LDG.E.SYS Rd, [addr]                 [memory]
        8x unrolled per nibble:
          -> extract (shift + mask 4-bit)          -> LOP3.LUT Rd, Ra, 0xF, Rb             [scalar, O(1)]
          -> int-to-float                          -> I2F.F32.U32 Rd, Ra                    [scalar, O(1)]
          -> - zp (subtract zero point 8)          -> FADD Rd, Ra, -8.0                     [scalar, O(1)]
          -> * scale (multiply by group scale)     -> FMUL Rd, Ra, Rb                       [scalar, O(1)]
          -> FMA (multiply-accumulate with input)  -> FFMA Rd, Ra, Rb, Rc                   [scalar, O(1)]
      -> Sigma (warp reduction)                    -> 5x SHFL.BFLY + FADD                   [O(log 32)]
      -> Sigma (cross-warp reduction)              -> STS + BAR.SYNC + LDS + FADD           [O(warps)]
```

**SASS budget:** 40 instructions per packed u32 (8 weights x 5 ops), 640 groups per
row, 10240 rows. ~26M FMAs total. Memory-bandwidth bound.

**Projection breakdown:**
- W_q: [2048, 5120] -> q[16 heads x 128] = 2048
- W_k: [2048, 5120] -> k[16 heads x 128] = 2048
- W_v: [6144, 5120] -> v[48 heads x 128] = 6144

---

## 3. project-z (Gate Projection)

**PyTorch:** `F.linear(x, W_z)` (same mechanism as project-qkv).
**Lithos:** `project W_z x z`
**Input:** x_normed[5120]. **Output:** z[6144] (48 heads x 128).

```
Linear / GEMV (Framework Layer)
  -> project W_z x z (Lithos Composition — W4A16 dequant GEMV)
    [identical SASS structure to project-qkv above]
    -> LDG + LOP3 + I2F + FADD + FMUL + FFMA   per nibble         [depth 3, O(6144 x 640 x 8)]
    -> 5x SHFL.BFLY + FADD                      per row reduction  [O(log 32)]
    -> STS + BAR.SYNC + LDS + FADD              cross-warp         [O(warps)]
```

**SASS budget:** Same dequant GEMV pattern. 6144 output rows x 640 groups x 8 weights.
This projection produces the gating signal used in post-process (step 10b).

---

## 4. conv1d (Depthwise Temporal Convolution)

**PyTorch:** `F.conv1d(x, w, groups=C)` with kernel_size=4 over a rolling buffer.
**Lithos:** `conv_w ** x` + `Sigma` over taps (elementwise FIR filter per channel).
**Input:** qkv[10240], history[10240, 3], conv_w[10240, 4].
**Output:** convolved[10240].

```
Depthwise Conv1d (Framework Layer)
  -> ** (elementwise multiply: window x kernel weights)
    per channel (10240 channels, parallelized):
      per tap (4 taps):
        -> * (multiply channel value by kernel weight)  -> FMUL Rd, Ra, Rb                [scalar, O(1)]
  -> Sigma-taps (sum over 4 taps per channel)
        -> FADD (accumulate 4 products)                 -> 3x FADD Rd, Ra, Rb             [O(4), unrolled]
  -> shift-buffer (update history)
        -> <- 32 (store new value to history buffer)    -> STG.E.SYS [addr], Rd           [memory]
```

**Note:** With only 4 taps, the "reduction" is 3 additions fully unrolled -- no
shuffle tree needed. Each channel is independent (depthwise).

**SASS budget:** 4 FMULs + 3 FADDs per channel = 7 FP ops x 10240 channels.
History shift is a memory operation (3 loads + 3 stores per channel to slide the window).

---

## 5. activate (SiLU)

**PyTorch:** `F.silu(x)` = `x * torch.sigmoid(x)`.
**Lithos:** `silu x` = `dup * -1 exp ++ // **`
**Input:** qkv[10240]. **Output:** activated[10240].

```
SiLU Activation (Framework Layer)
  -> silu x (Lithos Composition)
    applied elementwise across 10240 elements (depth 1):
      -> * -1 (negate x)                          -> FMUL Rd, Ra, -1.0                    [scalar, O(1)]
                                                     (or FADD Rd, -Ra, 0 with negate mod)
      -> exp (e^(-x))                             -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
      -> ++ (add 1 elementwise)                   -> FADD Rd, Ra, 1.0                     [scalar, O(1)]
      -> // (reciprocal elementwise = sigmoid)     -> MUFU.RCP Rd, Ra                      [SFU, O(1)]
      -> ** (multiply x * sigmoid(x))             -> FMUL Rd, Ra, Rb                      [scalar, O(1)]
```

**Decomposition detail:**
```
sigmoid(x)  =  1 / (1 + e^(-x))
            =  * -1  ->  exp  ->  + 1  ->  1/
SiLU(x)     =  x * sigmoid(x)
            =  sigmoid  ->  ** (multiply by original x)
```

**SASS budget:** 6 instructions per element (FMUL, FMUL, MUFU.EX2, FADD, MUFU.RCP, FMUL).
10240 elements. 2 SFU ops per element.

---

## 6. normalize-qk (L2 Norm + Scale)

**PyTorch:** `F.normalize(q, dim=-1) * (head_dim ** -0.5)` and `F.normalize(k, dim=-1)`.
**Lithos:** `l2norm q` then `* 0.08839` (= 1/sqrt(128)); `l2norm k`
**Input:** Q[16, 128], K[16, 128]. **Output:** Q_scaled[16, 128], K_normed[16, 128].

```
L2 Normalize + Scale (Framework Layer)
  -> l2norm x (Lithos Composition), applied per head (16 heads):
    -> ** (square each element)                   -> FMUL Rd, Ra, Ra                      [depth 1, O(128)]
    -> Sigma (sum all 128 squares)                -> stride FFMA partials                  [depth 1, O(128)]
                                                     5x SHFL.BFLY + FADD (intra-warp)     [O(log 32)]
                                                     STS + BAR.SYNC + LDS (cross-warp)    [O(warps)]
    -> + eps (add epsilon)                        -> FADD Rd, Ra, 1e-6                    [scalar, O(1)]
    -> 1/sqrt (reciprocal square root of sum)     -> MUFU.RSQ Rd, Ra                      [scalar, O(1)]
    -> ** (scale each element to unit length)     -> FMUL Rd, Ra, Rb                      [depth 1, O(128)]

  -> * 0.08839 (scale Q by 1/sqrt(128))          -> FMUL Rd, Ra, 0.08839                 [depth 1, O(128)]
     (applied to Q only, not K)
```

**SASS budget per head:** ~25 instructions (dominated by Sigma shuffle tree) + 2 elementwise
passes over 128 elements + 1 RSQ. Applied to 16 K-heads for Q + 16 K-heads for K = 32 passes.
Q gets an extra scaling pass.

---

## 7. compute-gates (Sigmoid for beta, Softplus+Exp for decay)

**PyTorch:** `torch.sigmoid(linear_beta(x))` for beta; `torch.exp(-softplus(linear_decay(x) + bias) * A.exp())` for decay.
**Lithos:** Two sub-compositions.
**Input:** x_normed[5120]. **Output:** beta[16] (broadcast to 48 via GQA), decay[48].

### 7a. Beta (sigmoid)

```
Sigmoid Gate (Framework Layer)
  -> sigmoid (Lithos Composition)
    -> project W_beta x raw_beta                  [W4A16 GEMV: 5120 -> 16]
       (same SASS as project-qkv, 16 output rows)
    per element (16 values):
      -> * -1 (negate)                            -> FMUL Rd, Ra, -1.0                    [scalar, O(1)]
      -> exp (e^(-x))                             -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
      -> + 1 (add one)                            -> FADD Rd, Ra, 1.0                     [scalar, O(1)]
      -> 1/ (reciprocal = sigmoid output)         -> MUFU.RCP Rd, Ra                      [SFU, O(1)]
```

### 7b. Decay (softplus + exponential decay)

```
Exponential Decay Gate (Framework Layer)
  -> project + bias + softplus + exp chain (Lithos Composition)
    -> project W_decay x raw_dt                   [W4A16 GEMV: 5120 -> 48]
    per element (48 values):
      -> + bias (add learned bias)                -> FADD Rd, Ra, bias_imm                [scalar, O(1)]
      -> exp (for softplus: e^x)                  -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
      -> + 1 (softplus: 1 + e^x)                 -> FADD Rd, Ra, 1.0                     [scalar, O(1)]
      -> log (softplus: ln(1 + e^x))              -> MUFU.LG2 Rt, Ra                      [SFU, O(1)]
                                                     FMUL Rd, Rt, 0.693147                [scalar, O(1)]
      -> * A_exp (multiply by learned rate)       -> FMUL Rd, Ra, Rb                      [scalar, O(1)]
      -> * -1 (negate)                            -> FMUL Rd, Ra, -1.0                    [scalar, O(1)]
      -> exp (final: e^(-A*dt) = decay)           -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
```

**Decomposition detail:**
```
softplus(x)  =  ln(1 + e^x)  =  exp -> + 1 -> log     (3 primitives, 5 SASS)
decay        =  e^(-A * softplus(x + bias))
             =  softplus -> * A_exp -> * -1 -> exp      (4 more primitives)
```

**SASS budget:**
- Beta: GEMV(5120->16) + 5 FP ops x 16 = 80 scalar ops.
- Decay: GEMV(5120->48) + 10 FP ops x 48 = 480 scalar ops. 4 SFU ops per element.

---

## 8. recurrence (Delta Rule -- State Update + Query)

**PyTorch:** Custom DeltaNet recurrence (no standard PyTorch equivalent).
**Lithos:** `deltanet_fused q k v z decay beta state out`
**Input:** S[128,128] (persistent state per head), Q[128], K[128], V[128], beta (scalar), decay (scalar).
**Output:** out[128] per head. S updated in place. Applied per head (48 V-heads, grouped by 16 K-heads).

```
DeltaNet Recurrence (Framework Layer)
  -> deltanet_fused (Lithos Composition), per head (48 heads):

    Step 1: Decay old state — S = S * decay
      -> ** S decay (elementwise matrix scale)    -> FMUL Rd, Ra, Rb                      [depth 2, O(128x128)]
         for each row i (128):
           for each col j (128):
             S[i,j] *= decay

    Step 2: Read from state — recall = S @ K (matvec)
      -> ** S K (element products per row)        -> FMUL Rd, Ra, Rb                      [depth 2, O(128x128)]
      -> Sigma (reduce each row)                  -> stride FFMA partials                  [per row, O(128)]
                                                     5x SHFL.BFLY + FADD                  [O(log 32)]
                                                     STS + BAR.SYNC + LDS                 [O(warps)]
         produces recall[128]

    Step 3: Compute delta — delta = beta * (V - recall)
      -> -- (subtract: V - recall)                -> FADD Rd, Ra, -Rb                     [depth 1, O(128)]
      -> ** beta (scale by beta)                  -> FMUL Rd, Ra, Rb                      [depth 1, O(128)]

    Step 4: Update state — S += outer(delta, K)
      -> *** delta K (outer product)              -> FMUL Rd, Ra, Rb                      [depth 2, O(128x128)]
         for each i (128):
           for each j (128):
             update[i,j] = delta[i] * K[j]
      -> +++ S update (matrix add)                -> FADD Rd, Ra, Rb                      [depth 2, O(128x128)]
         S[i,j] += update[i,j]

    Step 5: Query state — output = S_new @ Q (matvec)
      -> ** S Q (element products per row)        -> FMUL Rd, Ra, Rb                      [depth 2, O(128x128)]
      -> Sigma (reduce each row)                  -> stride FFMA partials                  [per row, O(128)]
                                                     5x SHFL.BFLY + FADD                  [O(log 32)]
                                                     STS + BAR.SYNC + LDS                 [O(warps)]
         produces output[128]
```

**Decomposition detail:**
```
Full recurrence per head:
  S = decay ** S  +++  beta ** (V -- recall) *** K
  output = S_new @ Q

Primitive sequence:
  ** (decay state)  ->  ** + Sigma (matvec S@K)  ->  -- (subtract)  ->  ** (scale beta)
  ->  *** (outer product)  ->  +++ (matrix add)  ->  ** + Sigma (matvec S@Q)
```

**SASS budget per head:**
- Decay: 128x128 = 16384 FMULs
- Matvec S@K: 128x128 FMULs + 128 Sigma reductions
- Subtract + scale: 128 FADDs + 128 FMULs
- Outer product: 128x128 = 16384 FMULs
- Matrix add: 128x128 = 16384 FADDs
- Matvec S@Q: 128x128 FMULs + 128 Sigma reductions
- **Total per head:** ~65K FP ops + 256 shuffle-tree reductions.
- **Total across 48 heads:** ~3.1M FP ops.

---

## 9. post-process (Group-Norm, Gate, Project, Residual)

**PyTorch:** Group norm per head, SiLU-gated output, linear projection, residual add.
**Lithos:** `rmsnorm` + `silu ** **` + `project` + `++`
**Input:** recurrence output[48, 128], z[48, 128], x_input[5120].
**Output:** layer_output[5120].

### 9a. Group norm (RMSNorm per head, plain w -- NOT 1+w)

```
GroupNorm / RMSNorm (Framework Layer)
  -> rmsnorm out w 128 (Lithos Composition), per head (48 heads):
    -> ** (square each element)                   -> FMUL Rd, Ra, Ra                      [depth 1, O(128)]
    -> Sigma (sum 128 squares)                    -> stride FFMA partials                  [O(128)]
                                                     5x SHFL.BFLY + FADD                  [O(log 32)]
                                                     STS + BAR.SYNC + LDS                 [O(warps)]
    -> / 128 (divide by head_dim)                 -> MUFU.RCP Rt, 128.0                   [scalar, O(1)]
                                                     FMUL Rd, Ra, Rt
    -> + eps (add epsilon)                        -> FADD Rd, Ra, 1e-6                    [scalar, O(1)]
    -> 1/sqrt (reciprocal square root)            -> MUFU.RSQ Rd, Ra                      [scalar, O(1)]
    -> ** (scale by 1/rms)                        -> FMUL Rd, Ra, Rb                      [depth 1, O(128)]
    -> ** (multiply by weight w)                  -> FMUL Rd, Ra, Rb                      [depth 1, O(128)]
```

### 9b. Gate with Z (SiLU on z, then element-multiply)

```
SiLU Gate (Framework Layer)
  -> silu z then ** (Lithos Composition), per head (48 heads):
    -> * -1 (negate z)                            -> FMUL Rd, Ra, -1.0                    [scalar, O(1)]
    -> exp (e^(-z))                               -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
    -> + 1 (add one)                              -> FADD Rd, Ra, 1.0                     [scalar, O(1)]
    -> 1/ (reciprocal = sigmoid(z))               -> MUFU.RCP Rd, Ra                      [SFU, O(1)]
    -> ** (z * sigmoid(z) = SiLU(z))              -> FMUL Rd, Ra, Rb                      [scalar, O(1)]
    -> ** (normed_output * SiLU(z) = gated)       -> FMUL Rd, Ra, Rb                      [depth 1, O(128)]
```

### 9c. Output projection

```
Linear / GEMV (Framework Layer)
  -> project W_o gated X (Lithos Composition — W4A16 dequant GEMV)
    [identical SASS structure to project-qkv]
    Input: gated[6144] (48 heads concatenated). Output: projected[5120].
    -> LDG + LOP3 + I2F + FADD + FMUL + FFMA   per nibble         [depth 3, O(5120 x 768 x 8)]
    -> 5x SHFL.BFLY + FADD                      per row reduction  [O(log 32)]
```

### 9d. Residual add

```
Residual Connection (Framework Layer)
  -> ++ (elementwise vector add)                  -> FADD Rd, Ra, Rb                      [depth 1, O(5120)]
```

---

## 10. MLP Block (RMSNorm + Gate/Up + SiLU + Down + Residual)

**PyTorch:** Standard gated MLP: `x + W_down(SiLU(W_gate(norm(x))) * W_up(norm(x)))`.
**Lithos:** `rmsnorm` + `project` x2 + `silu` + `**` + `project` + `++`
**Input:** x[5120] (post-attention residual). **Output:** x'[5120].

### 10a. MLP RMSNorm (same as step 1, with 1+w variant)

```
RMSNorm (Framework Layer)
  -> rmsnorm x w 5120 (Lithos Composition)
    -> ** (square)                                -> FMUL Rd, Ra, Ra                      [depth 1, O(5120)]
    -> Sigma (sum squares)                        -> stride FFMA + 5x SHFL.BFLY + FADD    [O(5120) + O(log 32)]
                                                     STS + BAR.SYNC + LDS                 [O(warps)]
    -> / 5120 (mean)                              -> MUFU.RCP + FMUL                      [scalar, O(1)]
    -> + eps                                      -> FADD Rd, Ra, 1e-6                    [scalar, O(1)]
    -> 1/sqrt                                     -> MUFU.RSQ Rd, Ra                      [scalar, O(1)]
    -> ** (scale)                                 -> FMUL Rd, Ra, Rb                      [depth 1, O(5120)]
    -> + 1 then ** (Qwen 1+w)                     -> FADD + FMUL                          [depth 1, O(5120)]
```

### 10b. Gate projection (W4A16 GEMV)

```
Linear / GEMV (Framework Layer)
  -> project W_gate x gate (Lithos — W4A16 dequant GEMV)
    Input: x_normed[5120]. Output: gate[17408].
    -> LDG + LOP3 + I2F + FADD + FMUL + FFMA   per nibble         [depth 3, O(17408 x 640 x 8)]
    -> 5x SHFL.BFLY + FADD                      per row reduction  [O(log 32)]
```

### 10c. Up projection (W4A16 GEMV)

```
Linear / GEMV (Framework Layer)
  -> project W_up x up (Lithos — W4A16 dequant GEMV)
    Input: x_normed[5120]. Output: up[17408].
    -> LDG + LOP3 + I2F + FADD + FMUL + FFMA   per nibble         [depth 3, O(17408 x 640 x 8)]
    -> 5x SHFL.BFLY + FADD                      per row reduction  [O(log 32)]
```

### 10d. SiLU on gate

```
SiLU Activation (Framework Layer)
  -> silu gate (Lithos Composition), elementwise across 17408 elements:
    -> * -1                                       -> FMUL Rd, Ra, -1.0                    [scalar, O(1)]
    -> exp                                        -> FMUL Rt, Ra, 1.442695                [scalar, O(1)]
                                                     MUFU.EX2 Rd, Rt                      [SFU, O(1)]
    -> + 1                                        -> FADD Rd, Ra, 1.0                     [scalar, O(1)]
    -> 1/                                         -> MUFU.RCP Rd, Ra                      [SFU, O(1)]
    -> ** (gate * sigmoid(gate))                  -> FMUL Rd, Ra, Rb                      [scalar, O(1)]
```

### 10e. Gate * Up (elementwise multiply)

```
Hadamard Product (Framework Layer)
  -> ** (elementwise multiply gate_activated * up) -> FMUL Rd, Ra, Rb                     [depth 1, O(17408)]
```

### 10f. Down projection (W4A16 GEMV)

```
Linear / GEMV (Framework Layer)
  -> project W_down intermediate X (Lithos — W4A16 dequant GEMV)
    Input: intermediate[17408]. Output: projected[5120].
    -> LDG + LOP3 + I2F + FADD + FMUL + FFMA   per nibble         [depth 3, O(5120 x 2176 x 8)]
    -> 5x SHFL.BFLY + FADD                      per row reduction  [O(log 32)]
```

### 10g. Residual add

```
Residual Connection (Framework Layer)
  -> ++ (elementwise vector add)                  -> FADD Rd, Ra, Rb                      [depth 1, O(5120)]
```

**MLP SASS budget:** Dominated by 3 GEMV projections (gate: 5120->17408, up: 5120->17408,
down: 17408->5120). ~205M FMA-equivalent ops. SiLU adds 6 ops x 17408 = 104K ops (negligible).

---

## Complete Primitive Inventory (41 Steps)

| Step | Operation | Lithos Primitive(s) | SASS Core Instruction(s) | Loop Depth |
|------|-----------|---------------------|--------------------------|------------|
| 1 | normalize | `** Sigma / + 1/sqrt ** + **` | FMUL, SHFL.BFLY+FADD, RCP+FMUL, FADD, RSQ, FMUL, FADD, FMUL | 0-1 |
| 2 | project qkv | `project` (x3) | LDG, LOP3, I2F, FADD, FMUL, FFMA, SHFL.BFLY | 3 |
| 3 | project z | `project` | LDG, LOP3, I2F, FADD, FMUL, FFMA, SHFL.BFLY | 3 |
| 4 | conv1d multiply | `**` | FMUL | 1 |
| 5 | conv1d reduce | `Sigma` (4-tap, unrolled) | 3x FADD | 0 |
| 6 | silu negate | `* -1` | FMUL | 0 |
| 7 | silu exp | `exp` | FMUL + MUFU.EX2 | 0 |
| 8 | silu add-one | `++` | FADD | 1 |
| 9 | silu reciprocal | `//` | MUFU.RCP | 1 |
| 10 | silu multiply | `**` | FMUL | 1 |
| 11 | l2norm square | `**` | FMUL | 1 |
| 12 | l2norm sum | `Sigma` | SHFL.BFLY + FADD + STS + BAR.SYNC + LDS | log N |
| 13 | l2norm rsqrt | `1/sqrt` | MUFU.RSQ | 0 |
| 14 | l2norm scale | `**` | FMUL | 1 |
| 15 | scale Q | `* 0.08839` | FMUL | 1 |
| 16 | sigmoid (beta) | `* -1 exp ++ //` | FMUL, FMUL+EX2, FADD, RCP | 0 |
| 17 | softplus exp | `exp` | FMUL + MUFU.EX2 | 0 |
| 18 | softplus add-one | `++` | FADD | 1 |
| 19 | softplus log | `log` | MUFU.LG2 + FMUL | 0 |
| 20 | matvec S@K products | `**` | FMUL | 2 |
| 21 | matvec S@K reduce | `Sigma` | SHFL.BFLY + FADD + STS + BAR.SYNC + LDS | log N |
| 22 | subtract V-recall | `--` | FADD (negate modifier) | 1 |
| 23 | scale by beta | `**` | FMUL | 1 |
| 24 | outer product | `***` | FMUL | 2 |
| 25 | decay state | `**` | FMUL | 2 |
| 26 | state add | `+++` | FADD | 2 |
| 27 | matvec S@Q products | `**` | FMUL | 2 |
| 28 | matvec S@Q reduce | `Sigma` | SHFL.BFLY + FADD + STS + BAR.SYNC + LDS | log N |
| 29 | group norm | `** Sigma / + 1/sqrt ** **` | FMUL, SHFL.BFLY+FADD, RCP+FMUL, FADD, RSQ, FMUL, FMUL | 0-1 |
| 30 | sigmoid (gate z) | `* -1 exp ++ //` | FMUL, FMUL+EX2, FADD, RCP | 0 |
| 31 | silu z | `**` | FMUL | 1 |
| 32 | output * gate | `**` | FMUL | 1 |
| 33 | project out | `project` | LDG, LOP3, I2F, FADD, FMUL, FFMA, SHFL.BFLY | 3 |
| 34 | residual add | `++` | FADD | 1 |
| 35 | MLP rmsnorm | `** Sigma / + 1/sqrt ** + **` | same as step 1 | 0-1 |
| 36 | MLP gate+up proj | `project` (x2) | LDG, LOP3, I2F, FADD, FMUL, FFMA, SHFL.BFLY | 3 |
| 37 | MLP silu gate | `* -1 exp ++ //` | FMUL, FMUL+EX2, FADD, RCP | 0 |
| 38 | MLP silu multiply | `**` | FMUL | 1 |
| 39 | MLP gate * up | `**` | FMUL | 1 |
| 40 | MLP down proj | `project` | LDG, LOP3, I2F, FADD, FMUL, FFMA, SHFL.BFLY | 3 |
| 41 | MLP residual add | `++` | FADD | 1 |

---

## SASS Instruction Summary

Every SASS instruction used by a DeltaNet layer, organized by family:

| SASS Instruction | Lithos Primitive | Usage |
|---|---|---|
| `FMUL Rd, Ra, Rb` | `*`, `**`, `***` | Multiply (elementwise, outer product, scale) |
| `FFMA Rd, Ra, Rb, Rc` | `project` inner loop, `Sigma` partials | Fused multiply-add (dequant accumulate, partial sums) |
| `FADD Rd, Ra, Rb` | `+`, `++`, `+++` | Add (scalar, elementwise, matrix) |
| `FADD Rd, Ra, -Rb` | `-`, `--` | Subtract (negate source modifier, no dedicated SUB) |
| `MUFU.RCP Rd, Ra` | `1/`, `//` | SFU reciprocal (division, sigmoid) |
| `MUFU.RSQ Rd, Ra` | `1/sqrt` | SFU reciprocal square root (RMSNorm, L2Norm) |
| `MUFU.EX2 Rd, Ra` | `exp` (after base conversion) | SFU 2^x (exponential via log2(e) multiply) |
| `MUFU.LG2 Rd, Ra` | `log` (before base conversion) | SFU log2(x) (logarithm via ln(2) multiply) |
| `SHFL.BFLY Rd, Ra, lane` | `Sigma` | Butterfly shuffle (intra-warp reduction) |
| `STS [addr], Rd` | `Sigma` (cross-warp) | Store to shared memory |
| `LDS Rd, [addr]` | `Sigma` (cross-warp) | Load from shared memory |
| `BAR.SYNC` | `Sigma` (cross-warp) | Warp barrier synchronization |
| `LDG.E.SYS Rd, [addr]` | `project`, `matvec` | Global memory load (weights, activations, state) |
| `STG.E.SYS [addr], Rd` | state write, history update | Global memory store |
| `LOP3.LUT Rd, Ra, Rb, Rc` | `project` (nibble extract) | Bitwise operation (extract 4-bit from packed u32) |
| `I2F.F32.U32 Rd, Ra` | `project` (dequant) | Integer to float conversion |
| `S2R Rd, SR_TID.X` | (compiler-derived) | Read thread/block ID for indexing |
| `IMAD Rd, Ra, Rb, Rc` | (compiler-derived) | Integer multiply-add for address computation |
| `ISETP Pd, Ra, Rb` | (compiler-derived) | Integer comparison for loop bounds |
| `@P BRA target` | (compiler-derived) | Conditional branch for stride loops |
| `IADD3 Rd, Ra, Rb, Rc` | (compiler-derived) | Integer add for loop increment |

---

## Compute Budget per DeltaNet Layer

| Operation | Dominant SASS | FLOPs (approx) | % of layer |
|---|---|---|---|
| normalize (step 1) | FMUL, Sigma | 20K | <0.01% |
| project-qkv (step 2) | FFMA (dequant GEMV) | 105M | 19.6% |
| project-z (step 3) | FFMA (dequant GEMV) | 63M | 11.8% |
| conv1d (step 4-5) | FMUL, FADD | 72K | <0.01% |
| activate SiLU (step 6-10) | FMUL, EX2, RCP | 62K | <0.01% |
| normalize-qk (step 11-15) | FMUL, RSQ, Sigma | 16K | <0.01% |
| compute-gates (step 16-19) | FMUL, EX2, LG2, RCP | 1.4M | 0.3% |
| recurrence (step 20-28) | FMUL, FADD, Sigma | 3.1M | 0.6% |
| post-process norm+gate (step 29-32) | FMUL, RSQ, EX2, RCP | 80K | <0.01% |
| project out (step 33) | FFMA (dequant GEMV) | 63M | 11.8% |
| MLP rmsnorm (step 35) | FMUL, Sigma | 20K | <0.01% |
| MLP gate+up proj (step 36) | FFMA (dequant GEMV) | 178M | 33.3% |
| MLP SiLU + multiply (step 37-39) | FMUL, EX2, RCP | 140K | <0.01% |
| MLP down proj (step 40) | FFMA (dequant GEMV) | 178M | 33.3% |
| MLP residual (step 41) | FADD | 5K | <0.01% |
| **Total** | | **~535M** | |

The 7 `project` calls (GPTQ dequant GEMV) account for **99%** of compute.
The recurrence (the actual DeltaNet innovation) is 0.6%. This is why the
model is memory-bandwidth-bound, not compute-bound: the bottleneck is
streaming 14 GB of quantized weights through the SMs, not the arithmetic.

---

## Compiler-Derived (Never in Source)

These appear in the final SASS but are NOT Lithos primitives. The compiler
infers them from operand shapes and primitive context:

| Concept | Source primitive | Generated SASS |
|---|---|---|
| Thread indexing | any `**` or `***` | `S2R SR_TID.X` + `S2R SR_CTAID.X` + `IMAD` |
| Stride loops | vector length > blockDim | `ISETP` + `@P BRA` + `IADD3` |
| Warp shuffle tree | `Sigma` on vector | 5x `SHFL.BFLY` + `FADD` |
| Shared memory | `Sigma` cross-warp | `STS` / `LDS` / `BAR.SYNC` |
| Register allocation | all live values | linear-scan allocator |
| Coalesced access | array indexing | address arithmetic via `IMAD` |
| Loop unrolling | `project` inner loop | 8x unroll for nibble extraction |
| Tiling | `***` outer product | compiler decides tile shape per SM |
| Grid sync | layer boundary | `BAR.SYNC.DEFER_BLOCKING` + `MEMBAR.GPU` |
