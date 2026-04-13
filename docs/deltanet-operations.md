# DeltaNet Layer Operations — Kernel → English → Primitives → Stack

One complete DeltaNet layer for Qwen3.5-27B. 10 operations in sequence.
Each operation: **name → plain english → primitive breakdown → stack language**.

Dimensions: hidden=5120, K_heads=16, V_heads=48, head_dim=128, conv_kernel=4.

---

## 1. normalize

**English:** RMSNorm the input hidden state. Compute root-mean-square across 5120 elements, scale each element by the reciprocal, multiply by learned weight. Qwen uses (1+w) variant.

**Primitives:** square, sum, divide-by-count, add-epsilon, rsqrt, scale, add-one-to-weight, multiply

```forth
x                           // ( x[5120] -- x[5120] )
dup square                  // ( x x² -- ) square each element
reduce-sum                  // ( x² -- sum_sq ) sum all 5120 squared values
5120 /                      // ( sum_sq -- mean_sq ) divide by element count
1e-6 +                      // ( mean_sq -- mean_sq+eps ) numerical stability
rsqrt                       // ( mean_sq+eps -- inv_rms ) reciprocal square root
*                           // ( x inv_rms -- x_normed ) scale input by inv_rms
weight 1.0 + *              // ( x_normed w+1 -- output ) multiply by (1 + learned weight)
```

---

## 2. project-qkv

**English:** Multiply normed input by a weight matrix to produce Q, K, V concatenated. Weight is 4-bit quantized (GPTQ). Input [5120] → output [10240]. Each output element is a dot product of the input with one row of weights.

**Primitives:** load-packed-int4, extract-4bit, subtract-zero-point, multiply-by-scale, dot-product, accumulate

```forth
normed[5120]                // ( normed -- normed ) input on stack
w4_packed[row]              // ( normed w4 -- normed w4 ) load packed weight row
unpack-8x4bit               // ( w4 -- i0 i1 i2 i3 i4 i5 i6 i7 ) extract 8 values from 32-bit word
each: zero-point - scale *  // ( int4 -- fp16 ) dequantize: (val - zp) * group_scale
normed *                    // ( fp16 normed_elem -- product ) multiply activation by weight
accumulate                  // ( product acc -- acc' ) running sum across K=5120
\ repeat for all rows → output[10240] = [Q:2048 | K:2048 | V:6144]
```

---

## 3. project-z

**English:** Same as project-qkv but separate: multiply normed input by gate weight matrix. Input [5120] → Z [6144]. This produces the gating signal used later.

**Primitives:** same as project-qkv, different weight matrix and output size.

```forth
normed[5120]                // ( normed -- normed )
w4_z_packed[row]            // ( normed w4 -- normed w4 ) gate projection weights
unpack-8x4bit               // extract 4-bit values
each: zero-point - scale *  // dequantize
normed *                    // multiply
accumulate                  // sum across K=5120 → Z[6144]
```

---

## 4. conv1d

**English:** Depthwise 1D convolution on the QKV vector using a rolling history of the last 3 timesteps. Each of 10240 channels is convolved independently with a 4-tap kernel. History shifts forward each token.

**Primitives:** concatenate-history, element-multiply, sum-over-taps, shift-buffer

```forth
history[10240, 3]           // ( hist -- hist ) last 3 timesteps per channel
qkv[10240]                  // ( hist qkv -- hist qkv ) current timestep
concat-along-time           // ( hist qkv -- window[10240, 4] ) [history | current]
conv_weight[10240, 4] *     // ( window w -- products[10240, 4] ) element-wise multiply
sum-over-taps               // ( products -- convolved[10240] ) sum the 4 values per channel
swap                        // prepare to update history
drop-oldest append-current  // ( hist qkv -- hist' ) shift buffer left, append new
```

---

## 5. activate

**English:** Apply SiLU activation element-wise to the convolved QKV. SiLU(x) = x * sigmoid(x). Smooth gating — values near zero are suppressed, large values pass through.

**Primitives:** negate, clamp, exp, add-one, reciprocal, multiply-by-original

```forth
x                           // ( x -- x ) convolved value
dup                         // ( x -- x x ) keep original
negate                      // ( x x -- x -x )
-80 80 clamp                // ( -x -- -x_clamped ) numerical stability
exp                         // ( -x_clamped -- e^-x )
1.0 +                       // ( e^-x -- 1+e^-x )
1.0 swap /                  // ( 1+e^-x -- sigmoid ) reciprocal = sigmoid(x)
*                           // ( x sigmoid -- x*sigmoid ) SiLU = x * sigmoid(x)
\ apply to all 10240 elements
```

---

## 6. split-reshape

**English:** Split the 10240-element vector into Q, K, V and reshape into per-head arrays. No math — just pointer arithmetic.

**Primitives:** slice, reshape

```forth
qkv[10240]                  // ( qkv -- qkv )
dup 0    2048  slice reshape[16,128]   // ( qkv -- qkv Q[16,128] ) first 2048 = Q
swap
dup 2048 4096  slice reshape[16,128]   // ( qkv -- qkv K[16,128] ) next 2048 = K
swap
    4096 10240 slice reshape[48,128]   // ( qkv -- V[48,128] ) remaining 6144 = V
```

---

## 7. normalize-qk

**English:** L2 normalize Q and K per head (make unit vectors), then scale Q by 1/sqrt(head_dim). This stabilizes the dot products in the recurrence.

**Primitives:** square, sum, add-epsilon, rsqrt, scale, constant-multiply

```forth
\ per head h (16 heads):
Q[h]                        // ( q -- q ) one head's query, 128 elements
dup square reduce-sum       // ( q -- q sum_sq )
1e-6 +                      // ( q sum_sq+eps -- )
rsqrt                       // ( -- q inv_norm )
*                           // ( q inv_norm -- q_unit ) L2 normalized
0.08839 *                   // ( q_unit -- q_scaled ) scale by 1/sqrt(128)

K[h]                        // same for K, without the final scaling
dup square reduce-sum 1e-6 + rsqrt *  // ( k -- k_unit )
```

---

## 8. compute-gates

**English:** Compute two gating signals from the normed input. Beta (per K-head, 16 values): sigmoid of a learned projection. Decay (per V-head, 48 values): exponential decay rate from a separate projection + bias + softplus + exponential.

**Primitives:** matvec, sigmoid, add-bias, softplus, exp, negate, multiply

```forth
\ Beta: controls how much new information enters the state
normed[5120]                // ( normed -- normed )
w_beta[16, 5120] matvec     // ( normed w -- raw_beta[16] ) projection
sigmoid                     // ( raw_beta -- beta[16] ) squash to (0,1)

\ Decay: controls how much old state is retained
normed[5120]                // ( normed -- normed )
w_decay[48, 5120] matvec    // ( normed w -- raw_dt[48] ) projection
dt_bias[48] +               // ( raw_dt bias -- biased[48] ) add learned bias
softplus                    // ( biased -- dt[48] ) softplus = log(1 + exp(x))
A_log exp                   // ( -- A_exp[48] ) learned positive rate
dt * negate                 // ( A_exp dt -- -A_exp*dt )
exp                         // ( -A*dt -- decay[48] ) decay rate in (0,1)
```

---

## 9. recurrence

**English:** The core DeltaNet update. A persistent state matrix S [48, 128, 128] is updated each token. Per head: (1) decay old state, (2) read what the state remembers about this key, (3) compute what's new (delta), (4) write the new information into state, (5) query the state with Q to produce output.

**Primitives:** scale-matrix, matvec, subtract, scale-vector, outer-product, add-matrix, matvec

```forth
\ per head h (48 V-heads; Q/K expanded from 16 K-heads via GQA 3:1 repeat):
\ beta[16] → beta_exp[48] via repeat (each K-head's beta serves 3 V-heads)

S[h]                        // ( S -- S ) state matrix [128, 128], persistent
decay[h] *                  // ( S decay -- S' ) scale entire matrix: forget old info

S[h] K[h] matvec            // ( S K -- kv_mem ) what state remembers about this key [128]
V[h] swap -                 // ( V kv_mem -- residual ) what's new = V - remembered
beta_exp[h] *               // ( residual beta -- delta ) gate the update (beta broadcast from K-head)

delta K[h] outer-product    // ( delta K -- update[128,128] ) rank-1 update matrix
S[h] +                      // ( update S -- S_new ) add new information to state
                            // S[h] := S[h] * decay + outer(delta, K) — the full recurrence

S[h] Q[h] matvec            // ( S Q -- output[128] ) query the updated state
```

---

## 10. post-process

**English:** Four steps to produce the final layer output from the recurrence output. (a) Group norm each head. (b) Gate with Z (from step 3). (c) Project back to hidden dim. (d) Residual add.

**Primitives:** rms-norm, silu, element-multiply, project, add

```forth
\ (a) Group norm per head
output[h]                   // ( out -- out ) recurrence output, 128 elements
dup square reduce-mean      // ( out -- out mean_sq )
epsilon + rsqrt             // ( out mean_sq -- out inv_rms )
* weight *                  // ( out inv_rms w -- normed ) RMSNorm (plain w, not 1+w)

\ (b) Gate with Z
Z[h]                        // ( normed z -- normed z ) gate projection from step 3
dup sigmoid *               // ( z -- z_silu ) SiLU activation on Z
*                           // ( normed z_silu -- gated ) element-wise gate

\ (c) Output projection
gated[6144] flatten         // ( gated -- flat[6144] ) concatenate all 48 heads
w_out[5120, 6144] matvec    // ( flat w -- projected[5120] ) project to hidden dim (GPTQ W4A16)

\ (d) Residual add
input[5120] +               // ( projected input -- output[5120] ) skip connection
```

---

## Full Pipeline Summary

```
input[5120]
  → normalize               // RMSNorm (1+w)
  → project-qkv             // [5120] → [10240] via GPTQ W4A16
  → project-z               // [5120] → [6144] via GPTQ W4A16 (gate, separate)
  → conv1d                  // temporal mixing with 4-tap history
  → activate                // SiLU element-wise
  → split-reshape            // [10240] → Q[16,128] K[16,128] V[48,128]
  → normalize-qk            // L2 norm + scale Q by 1/√128
  → compute-gates            // beta[16] via sigmoid, decay[48] via softplus+exp
  → recurrence              // S *= decay; delta = (V - S@K)*β; S += outer(δ,K); out = S@Q
  → post-process            // group-norm → gate(Z) → project → residual add
output[5120]
```
