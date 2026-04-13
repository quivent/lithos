# DeltaNet Layer — Pure Primitive Stack Operations

One complete DeltaNet layer. x is a vector accumulator.
Only: `+ - / * ^ e √ Σ @ dup swap` and constants.
`*` alone = squared. `* 7` = times 7. `@` = matrix-vector multiply.
`Σ` = sum of vector elements. `⊗` = outer product.
`log` = natural logarithm (inverse of e^).

x = input vector [5120]. w = weight. W = weight matrix.

---

## 1. Normalize (RMSNorm with 1+w)

```
x                     // input [5120]
dup                   // x x
*                     // x x²          — element-wise square
Σ                     // x s           — sum of squares
/ 5120                // x m           — mean of squares
+ 1e-6                // x m'          — add epsilon
√                     // x rms         — root mean square
1 swap /              // x 1/rms       — reciprocal
*                     // x_norm        — x * (1/rms) = normalized
w 1 + *               // output        — x_norm * (1 + w)
```

---

## 2. Project QKV (W4A16 dequant + matvec)

```
x                     // normalized input [5120]
Wqkv @                // qkv [10240]    — matrix-vector multiply (W is [10240, 5120])
```

Note: `@` here means for each output row i: `Σ(W[i] * x)`.
W is 4-bit packed; `@` includes dequant: `(int4 - zero) * scale` per element before multiply.

---

## 3. Project Z (gate projection)

```
x                     // same normalized input [5120] (kept from step 1)
Wz @                  // z [6144]       — gate projection
```

---

## 4. Conv1d (depthwise, 4-tap, with history)

```
qkv                   // current QKV [10240]
history               // previous 3 timesteps [10240, 3]
concat                // window [10240, 4] — [history | current]
conv_w *              // window * kernel_weights — element-wise [10240, 4]
Σ-over-taps           // convolved [10240] — sum along the 4-tap dimension
```

Update: `history ← window[:, 1:]` (shift left, drop oldest, keep current).

---

## 5. Activate (SiLU = x * sigmoid(x))

Applied element-wise to all 10240 values:

```
x                     // convolved value
dup                   // x x           — keep original
* -1                  // x -x          — negate
e swap ^              // x e^(-x)      — exponential of negated
+ 1                   // x (1+e^(-x))
1 swap /              // x 1/(1+e^(-x)) — this IS sigmoid, decomposed
*                     // x * sigmoid(x) — this IS SiLU, decomposed
```

---

## 6. Split + Reshape

```
qkv [10240]           // after SiLU
→ Q [0:2048]    reshape [16, 128]    // 16 key heads, 128 dim each
→ K [2048:4096] reshape [16, 128]
→ V [4096:]     reshape [48, 128]    // 48 value heads, 128 dim each
```

Pointer arithmetic only. No computation.

---

## 7. L2 Normalize Q and K, Scale Q

Per head h (16 heads):

```
Q[h]                  // query vector [128]
dup                   // Q Q
*                     // Q Q²          — element-wise square
Σ                     // Q s           — sum of squares
+ 1e-6                // Q s'
√                     // Q ‖Q‖
1 swap /              // Q 1/‖Q‖
*                     // Q̂             — unit vector
* 0.08839             // Q̂ * 1/√128   — scaled query

K[h]                  // key vector [128]
dup * Σ + 1e-6 √ 1 swap / *   // K̂  — L2 normalized (same ops, no scale)
```

---

## 8. Compute Gates

### 8a. Beta (input gate, 16 values → broadcast to 48)

```
x_normed              // normalized hidden state [5120] (from step 1)
Wβ @                  // raw [16]       — beta projection
* -1                  // -raw
e swap ^              // e^(-raw)
+ 1                   // 1 + e^(-raw)
1 swap /              // 1/(1+e^(-raw)) — this IS sigmoid, decomposed
→ β [16]              // broadcast to [48] via 3:1 GQA repeat
```

### 8b. Decay (forgetting gate, 48 values)

```
x_normed              // [5120]
Wα @                  // raw [48]       — decay projection
+ bias                // raw + dt_bias [48]
dup                   // keep for softplus
* -1                  // negate
e swap ^              // e^(-x)
+ 1                   // 1 + e^(-x)
log                   // log(1 + e^(-x))
+                     // x + log(1+e^(-x)) — this IS softplus, decomposed
→ dt [48]             // softplus output

A_log                 // learned log-rate [48]
e swap ^              // e^(A_log) = A [48] — positive rate
* dt                  // A * dt
* -1                  // -(A * dt)
e swap ^              // e^(-(A*dt)) — this IS decay, decomposed
→ decay [48]          // in (0, 1)
```

---

## 9. Recurrence (the core — per head, 48 heads)

S is [128, 128] persistent state. Q, K are [128]. V is [128]. β is scalar. decay is scalar.

```
// Step 1: Decay old state
S                     // state [128, 128]
* decay               // S * decay      — element-wise, forget old information

// Step 2: Read from state
S K @                 // kv_mem [128]    — what state remembers about this key
                      // (@ here = for each row i: Σ(S[i] * K))

// Step 3: Compute delta (what's new)
V                     // value vector [128]
kv_mem -              // V - kv_mem     — residual: what's NOT in state yet
* β                   // (V - kv_mem) * β — gated delta

// Step 4: Update state (rank-1)
delta K ⊗             // outer product [128, 128] — delta[i] * K[j] for all i,j
S +                   // S = S + outer(delta, K) — add new information to state

// Step 5: Query state
S Q @                 // output [128]    — read from updated state using query
```

---

## 10. Post-process

### 10a. Group norm per head (48 heads, plain w — NOT 1+w)

```
output[h]             // recurrence output [128]
dup * Σ / 128 + ε √ 1 swap / *    // RMSNorm (same as step 1 but / 128 not / 5120)
* w_head              // multiply by head norm weight (plain w, not 1+w)
```

### 10b. Gate with Z (SiLU on Z, then element-multiply)

```
z[h]                  // gate value [128] (from step 3)
dup                   // z z
* -1 e swap ^ + 1 1 swap /   // sigmoid(z) — decomposed
*                     // z * sigmoid(z) = SiLU(z)
* output[h]           // gated = output * SiLU(z)
```

### 10c. Output projection

```
gated [6144]          // all 48 heads concatenated
Wo @                  // projected [5120] — W4A16 matvec back to hidden dim
```

### 10d. Residual add

```
+ x_input             // output = projected + original input — skip connection
```

---

## Complete Linear Sequence (Summary)

```
x[5120]               // INPUT

// 1. Normalize
dup * Σ / 5120 + ε √ 1 swap / * w 1 + *

// 2-3. Project
dup  Wqkv @  swap  Wz @  swap     // qkv[10240] and z[6144] on stack, normed kept

// 4. Conv1d
history concat conv_w * Σ-taps

// 5. SiLU (element-wise)
dup * -1 e swap ^ + 1 1 swap / *

// 6. Split
→ Q[16,128] K[16,128] V[48,128]

// 7. L2 norm Q, K; scale Q
each-head: dup * Σ + ε √ 1 swap / *    Q: * 0.08839

// 8. Gates
x_normed Wβ @ dup * -1 e swap ^ + 1 1 swap / → β
x_normed Wα @ + bias dup * -1 e swap ^ + 1 log + → dt
A_log e swap ^ * dt * -1 e swap ^ → decay

// 9. Recurrence (per head)
S * decay              // forget
S K @ → kv_mem         // read
V kv_mem - * β → delta // delta
S delta K ⊗ + → S      // write
S Q @ → output         // query

// 10. Post-process
each-head: dup * Σ / 128 + ε √ 1 swap / * * w_head   // group norm
z dup * -1 e swap ^ + 1 1 swap / * * output           // gate
flatten Wo @           // project
+ x_input              // residual

→ output[5120]         // OUTPUT
```

---

## Decomposition Reference

| Kernel | Named function | Primitive decomposition |
|--------|---------------|------------------------|
| normalize | RMSNorm(x,w) | `dup * Σ / n + ε √ 1 swap / * w 1 + *` |
| project | W @ x (GPTQ) | `W @ ` (dequant is inside @: `(int4 - zp) * scale` per element) |
| conv1d | depthwise conv | `concat conv_w * Σ-taps` |
| activate | SiLU(x) | `dup * -1 e swap ^ + 1 1 swap / *` |
| sigmoid | σ(x) | `* -1 e swap ^ + 1 1 swap /` |
| softplus | sp(x) | `dup * -1 e swap ^ + 1 log +` |
| l2norm | x/‖x‖ | `dup * Σ + ε √ 1 swap / *` |
| decay | e^(-A·dt) | `A_log e swap ^ * dt * -1 e swap ^` |
| recurrence | S update | `S * decay; S K @; V - * β; S + δ K ⊗; S Q @` |
| group-norm | RMSNorm(x,w) plain | `dup * Σ / 128 + ε √ 1 swap / * * w` |
| gate | x·SiLU(z) | `z dup * -1 e swap ^ + 1 1 swap / * *` |
| residual | x + skip | `+ x_input` |
