# Weights as Code: Lithos Language v2

## The Core Insight

With Q4 quantized models, every learned weight is a 4-bit integer. After
dequantization (subtract zero point, multiply by scale), each weight becomes an
FP32 constant known at compile time. The compiler reads safetensors once, performs
dequantization, and emits SASS instructions with the resulting FP32 values encoded
as 32-bit immediates in the instruction word itself.

There is no weight memory. There are no weight pointers. There are no runtime
dequantization instructions. The model IS the program.

A 7B-parameter Q4 model produces roughly 3.5 billion FMUL-IMM instructions. Each
instruction is 16 bytes of SASS. The "model binary" is the executable.

## What Changes, What Stays

### Changes fundamentally: `project`

The old `project` primitive meant "loop over packed weight memory, dequant each
nibble, multiply by input, accumulate." This entire loop, its address arithmetic,
its nibble extraction, its scale lookups -- all of it disappears from the kernel
source.

In v2, `project` becomes a **compile-time expansion directive**. It tells the
compiler: "emit the weight-immediate instruction stream for this projection here."
The .ls source does not contain the weight data. The .ls source is a template.
The compiler fills it in.

### Stays the same: `matvec`

`matvec` (FP32 state GEMV, e.g., S @ K in DeltaNet) operates on runtime
activation data. State matrices live in HBM, loaded via LDG, stored via STG.
These are not learned weights. `matvec` keeps its current semantics unchanged.

### Stays the same: elementwise ops

`**`, `++`, `--`, `//` all operate on activation vectors in registers. No change.

### Stays the same: reductions

The shuffle-tree reduction primitive stays identical.

### Stays the same: scalar math

`exp`, `log`, `1/`, `1/sqrt`, `sqrt` -- all unchanged.

---

## The Template Model

### .ls files become templates

A .ls file no longer describes a complete, standalone kernel. It describes a
**template** -- a parameterized instruction pattern that the compiler instantiates
once per layer by binding concrete weight tensors to the template's weight
declarations.

The same template produces 64 different SASS streams (one per layer in a 64-layer
model), because each layer has different weight values. The instruction structure
is identical; only the immediates differ.

### Compilation pipeline

```
  .ls template
       |
       v
  lithos compiler  <---  model.safetensors
       |
       v
  per-layer SASS   (layer_00.bin, layer_01.bin, ..., layer_47.bin)
       |
       v
  megakernel linker  (concatenates into 2 cooperative grid-sync megakernels)
```

---

## New Language Elements

### 1. Weight tensor declarations

```
weight W_q  : [5120, 5120]  q4    \ Q projection, quantized 4-bit
weight W_k  : [1024, 5120]  q4    \ K projection (GQA: fewer heads)
weight W_v  : [3072, 5120]  q4    \ V projection
weight W_z  : [3072, 5120]  q4    \ output gate projection
weight W_o  : [5120, 3072]  q4    \ output projection
weight W_g  : [13824, 5120] q4    \ MLP gate projection
weight W_u  : [13824, 5120] q4    \ MLP up projection
weight W_d  : [5120, 13824] q4    \ MLP down projection

weight w_norm1 : [5120]     f32   \ RMSNorm weight (pre-attention)
weight w_norm2 : [5120]     f32   \ RMSNorm weight (post-attention)
weight w_norm3 : [5120]     f32   \ RMSNorm weight (pre-MLP)

weight A_log   : [48]       f32   \ decay log-parameter (per-head)
weight dt_bias : [48]       f32   \ softplus bias (per-head)
```

The `weight` keyword declares a compile-time constant tensor. The compiler:
1. Reads the named tensor from safetensors at compile time
2. For q4 tensors: dequantizes every element to FP32
3. Embeds the resulting constants as instruction immediates

The shape annotation `[rows, cols]` tells the compiler how many FMUL-IMM
instructions to emit when this weight appears in a `project` directive.

The quantization tag (`q4`, `f32`, etc.) tells the compiler what dequantization
to apply at compile time. For `f32` weights (norm scales, small per-head
parameters), the values are used as-is.

### 2. Layer parameterization

```
template deltanet_layer
    layer L : 0..47
```

The `layer L : 0..47` declaration tells the compiler to stamp out this template
48 times, binding L to each layer index. The compiler uses L to select the
correct weight tensors from safetensors (e.g., `model.layers.{L}.self_attn.q_proj`).

### 3. Binding syntax

The compiler needs to map .ls weight names to safetensors tensor paths. This is
declared in a `bind` block:

```
bind L
    W_q     <- model.layers.{L}.self_attn.q_proj
    W_k     <- model.layers.{L}.self_attn.k_proj
    W_v     <- model.layers.{L}.self_attn.v_proj
    W_z     <- model.layers.{L}.self_attn.z_proj
    W_o     <- model.layers.{L}.self_attn.o_proj
    W_g     <- model.layers.{L}.mlp.gate_proj
    W_u     <- model.layers.{L}.mlp.up_proj
    W_d     <- model.layers.{L}.mlp.down_proj
    w_norm1 <- model.layers.{L}.input_layernorm.weight
    w_norm2 <- model.layers.{L}.post_attention_layernorm.weight
    w_norm3 <- model.layers.{L}.pre_mlp_layernorm.weight
    A_log   <- model.layers.{L}.self_attn.A_log
    dt_bias <- model.layers.{L}.self_attn.dt_bias
```

The `{L}` interpolation substitutes the layer index. The compiler resolves these
paths against the safetensors file at compile time. If a tensor is missing, it is
a compile error.

### 4. The `project` directive (new semantics)

Old syntax (v1 -- dies):
```
fn gptq_gemv W_packed scales x -> y
    for k_p tid_local K_packed 256
        packed = W_packed [ wp_idx ]     \ <-- runtime load from HBM
        ...nibble extraction...          \ <-- runtime dequant
        fma acc fnib0 xval0 acc          \ <-- runtime multiply
    endfor
```

New syntax (v2):
```
project W_q x -> q
```

That is the entire statement. One line. The compiler expands it into the
FMUL-IMM instruction stream: for a [5120, 5120] Q4 weight matrix with input
vector x of dimension 5120, the compiler emits 5120 output elements, each
requiring 5120 multiply-accumulate operations. The weight value is the immediate
in each FMUL instruction.

The compiler decides:
- How to partition rows across thread blocks
- How to schedule instructions to hide latency
- How to use register file capacity (the input vector x stays in registers)
- How to reduce partial sums (warp shuffle tree -- same as before)

The .ls author specifies WHAT (project this weight matrix against this vector)
and the compiler decides HOW (instruction scheduling, thread mapping, reduction
strategy). This is a higher level of abstraction than v1, but only for weight
projections -- everything else stays low-level.

### 5. Small learned parameters

Some learned parameters are scalars or small vectors, not large matrices:

- `w_norm1 : [5120] f32` -- RMSNorm scale weights
- `A_log : [48] f32` -- decay log-parameters, one per head
- `dt_bias : [48] f32` -- softplus bias, one per head

These are also compile-time constants. They appear as immediates in the
instructions that use them. For example:

```
** x w_norm1     \ elementwise multiply by norm weight
```

In v1 this loaded `w_norm1[i]` from a pointer at runtime. In v2 the compiler
knows every element of `w_norm1` at compile time and emits:

```sass
FMUL R4, R4, 0.9823 ;    // w_norm1[0]
FMUL R5, R5, 1.0012 ;    // w_norm1[1]
FMUL R6, R6, 0.9956 ;    // w_norm1[2]
...                       // 5120 instructions, one per element
```

There is no load. There is no pointer. The weight IS the instruction.

The same applies to `A_log` and `dt_bias` -- they are per-head scalars known at
compile time. The decay computation:

```
g = * -1 (exp A_log) * softplus(a_proj + dt_bias)
```

becomes, for head h:

```sass
FADD R10, R10, 0.3421 ;   // + dt_bias[h], immediate
...softplus sequence...
FMUL R10, R10, -2.1547 ;  // * (-exp(A_log[h])), pre-computed immediate
```

The compiler pre-computes `-exp(A_log[h])` at compile time and emits a single
FMUL-IMM. Two runtime operations (exp + negate) collapse into zero runtime
instructions.

---

## What Disappears from the Language

### Gone: `param weights ptr`

Weight pointers do not exist. Weights are not runtime data. The function
signature no longer takes weight buffer pointers.

Old:
```
fn gptq_gemv W_packed scales x -> y
    param K u32
```

New:
```
fn layer x X state -> X_out state_out
    \ No weight parameters. Weights are declared at template level.
```

### Gone: indexed weight loads

```
packed = W_packed [ wp_idx ]    \ DEAD
sc = scales [ grp ]             \ DEAD
```

No weight indexing syntax exists in v2. Weights are not addressable at runtime.

### Gone: runtime dequantization

The entire nibble-extraction, zero-point-subtract, scale-multiply pipeline in
the old `gptq_gemv` kernel is gone from the .ls source. The compiler performs
dequantization once at compile time. The kernel source never mentions packed
formats, nibbles, scales, or zero points.

### Gone: the weight reduction loop

The `for k_p ... endfor` loop in `gptq_gemv` that iterated over packed weight
words is gone. The compiler generates the linear instruction stream directly.
There is no loop in the emitted SASS -- just a straight-line sequence of
FMUL-IMM instructions (possibly millions per projection).

---

## What Stays as Runtime Parameters

### Activation vectors: always runtime

```
runtime x : [5120] f32         \ input hidden state
runtime X : [5120] f32         \ residual accumulator
```

Input activations arrive at runtime. They are loaded from HBM via LDG into
registers. Every elementwise op, every reduction, every matvec reads activations
from registers. This is unchanged.

### KV cache / state matrices: always runtime

```
runtime state : [16, 3, 128, 128] f32    \ DeltaNet S matrices
runtime kv_cache : [seq_len, n_heads, d] f32
```

Recurrent state and KV cache are runtime data that persists across tokens. They
live in HBM, accessed by LDG/STG. Unchanged.

### Position index: always runtime

```
runtime pos : u32              \ current sequence position (for RoPE)
```

### Decay and beta scalars: weights-as-code

In GatedDeltaNet, `decay` and `beta` are per-head learned parameters. They are
NOT runtime. They are model weights read from safetensors. In v2 they become
compile-time constants:

```
weight A_log   : [48] f32     \ decay = exp(-exp(A_log) * softplus(...))
weight dt_bias : [48] f32     \ bias for softplus gate
weight beta    : [48] f32     \ correction scaling, per-head
```

The compiler pre-computes what it can (e.g., `exp(A_log)`) and embeds the
results as immediates.

---

## Full Example: DeltaNet Layer in the New Language

```
\ deltanet_layer.ls -- one DeltaNet decode layer, weights-as-code
\
\ This is a TEMPLATE. The compiler instantiates it once per layer,
\ binding weight tensors from safetensors.

template deltanet_layer
    layer L : 0..47

\ ---- Weight declarations (compile-time constants) ----

weight W_q     : [2048, 5120]  q4     \ Q projection [num_kv_heads * d_head, hidden]
weight W_k     : [2048, 5120]  q4     \ K projection
weight W_v     : [6144, 5120]  q4     \ V projection [num_v_heads * d_head, hidden]
weight W_z     : [6144, 5120]  q4     \ output gate projection
weight W_a     : [48, 5120]    q4     \ decay input projection (a_proj)
weight W_o     : [5120, 6144]  q4     \ output projection
weight W_g     : [13824, 5120] q4     \ MLP gate
weight W_u     : [13824, 5120] q4     \ MLP up
weight W_d     : [5120, 13824] q4     \ MLP down

weight w_pre   : [5120]        f32    \ RMSNorm pre-attention
weight w_post  : [5120]        f32    \ RMSNorm post-attention
weight w_mlp   : [5120]        f32    \ RMSNorm pre-MLP

weight A_log   : [48]          f32    \ decay log-parameter
weight dt_bias : [48]          f32    \ softplus bias
weight beta    : [48]          f32    \ correction scale

\ ---- Binding (safetensors tensor paths) ----

bind L
    W_q     <- model.layers.{L}.self_attn.q_proj.qweight
    W_k     <- model.layers.{L}.self_attn.k_proj.qweight
    W_v     <- model.layers.{L}.self_attn.v_proj.qweight
    W_z     <- model.layers.{L}.self_attn.z_proj.qweight
    W_a     <- model.layers.{L}.self_attn.a_proj.qweight
    W_o     <- model.layers.{L}.self_attn.o_proj.qweight
    W_g     <- model.layers.{L}.mlp.gate_proj.qweight
    W_u     <- model.layers.{L}.mlp.up_proj.qweight
    W_d     <- model.layers.{L}.mlp.down_proj.qweight
    w_pre   <- model.layers.{L}.input_layernorm.weight
    w_post  <- model.layers.{L}.post_attention_layernorm.weight
    w_mlp   <- model.layers.{L}.pre_mlp_layernorm.weight
    A_log   <- model.layers.{L}.self_attn.A_log
    dt_bias <- model.layers.{L}.self_attn.dt_bias
    beta    <- model.layers.{L}.self_attn.beta

\ ---- Runtime parameters (activation data, not weights) ----

runtime x     : [5120]              f32    \ input hidden state
runtime X     : [5120]              f32    \ residual accumulator (read/write)
runtime state : [16, 3, 128, 128]   f32    \ DeltaNet recurrent state (read/write)

\ ---- The layer computation ----
\ 71 steps from the STEPS decomposition, annotated with what the compiler does.

fn deltanet_layer x X state -> X state

    \ ============================================================
    \ RMSNorm on input hidden state (steps 1-6)
    \ Steps 1-4: pure activation compute (no weights involved)
    \ Step 5-6: uses w_pre (small f32 weight vector -> immediates)
    \ ============================================================

    \  1  **     square each element                      [activation]
    \  2  Sigma  sum all squares                          [activation]
    \  3  / D    mean of squares                          [activation]
    \  4  1/sqrt root-mean-square reciprocal              [activation]
    \  5  //     x / rms (elementwise)                    [activation]
    \  6  ** w   scale by learned weight                  [WEIGHT-IMM: w_pre]
    rmsnorm x w_pre -> x_norm

    \ ============================================================
    \ Projections (steps 7-9)
    \ WEIGHT-IMMEDIATE STREAMS: the compiler emits millions of
    \ FMUL-IMM instructions per projection.
    \ ============================================================

    \  7  project  Q = W_q @ x_norm                      [WEIGHT-IMM: W_q]
    project W_q x_norm -> q

    \  8  project  K = W_k @ x_norm                      [WEIGHT-IMM: W_k]
    project W_k x_norm -> k

    \  9  project  V = W_v @ x_norm                      [WEIGHT-IMM: W_v]
    project W_v x_norm -> v

    \ Also project the gate and decay-input:
    \     project  Z = W_z @ x_norm                      [WEIGHT-IMM: W_z]
    project W_z x_norm -> z

    \     project  a = W_a @ x_norm                      [WEIGHT-IMM: W_a]
    project W_a x_norm -> a_proj

    \ ============================================================
    \ L2-norm on K (steps 10-13)
    \ Pure activation compute.
    \ ============================================================

    \ 10  **     square each element of K                 [activation]
    \ 11  Sigma  sum all squares                          [activation]
    \ 12  sqrt   sqrt(sum)                                [activation]
    \ 13  //     K / sqrt(sum) = L2-norm(K)               [activation]
    l2norm k -> k

    \ ============================================================
    \ SiLU on Q (steps 14-18)
    \ Pure activation compute.
    \ ============================================================

    \ 14  * -1    -Q                                      [activation]
    \ 15  exp     e^(-Q)                                  [activation]
    \ 16  ++      1 + e^(-Q)                              [activation]
    \ 17  //      sigmoid(Q)                              [activation]
    \ 18  **      Q * sigmoid(Q) = SiLU(Q)                [activation]
    silu q -> q

    \ ============================================================
    \ L2-norm on Q (steps 19-22)
    \ Pure activation compute.
    \ ============================================================

    \ 19  **     square                                   [activation]
    \ 20  Sigma  sum                                      [activation]
    \ 21  sqrt   root                                     [activation]
    \ 22  //     normalize                                [activation]
    l2norm q -> q

    \ ============================================================
    \ Sigmoid on beta (steps 23-26)
    \ In v2, beta is a compile-time weight. sigmoid(beta) is
    \ pre-computed at compile time. The compiler emits a single
    \ MOV-IMM per head with the pre-computed sigmoid value.
    \ ZERO RUNTIME INSTRUCTIONS for this block.
    \ ============================================================

    \ 23  * -1                                            [ELIDED: compile-time]
    \ 24  exp                                             [ELIDED: compile-time]
    \ 25  ++                                              [ELIDED: compile-time]
    \ 26  //                                              [ELIDED: compile-time]
    \ beta_sig = sigmoid(beta) -- computed at compile time, emitted as immediate

    \ ============================================================
    \ Decay gate (steps 27-34)
    \ dt_bias and A_log are compile-time weights.
    \ a_proj is a runtime activation (it depends on input x).
    \
    \ Step 27: a_proj + dt_bias -- dt_bias is immediate
    \ Steps 28-30: softplus(a_proj + dt_bias) -- runtime (depends on a_proj)
    \ Steps 31-32: -exp(A_log) -- compile-time constant, becomes one immediate
    \ Step 33: g = (-exp(A_log)) * softplus(...) -- one FMUL-IMM
    \ Step 34: decay = exp(g) -- runtime
    \ ============================================================

    \ 27  ++    t = a_proj + dt_bias                      [WEIGHT-IMM: dt_bias]
    ++ a_proj dt_bias -> t

    \ 28  exp   exp(t)                                    [activation]
    \ 29  ++    1 + exp(t)                                [activation]
    \ 30  log   softplus(t)                               [activation]
    softplus t -> sp

    \ 31-32 (-exp(A_log)) is a compile-time constant     [WEIGHT-IMM: neg_exp_A]
    \ 33  **    g = neg_exp_A * sp                        [WEIGHT-IMM: pre-computed]
    ** neg_exp_A sp -> g

    \ 34  exp   decay = exp(g)                            [activation]
    ** exp g -> decay

    \ ============================================================
    \ Delta rule with decay (steps 35-41)
    \ ALL runtime. State S is in HBM. K, V, Q are activations.
    \ matvec uses LDG/STG for state, no weight immediates.
    \ ============================================================

    \ 35  matvec  recall = S @ K                          [activation: state GEMV]
    matvec state k -> recall

    \ 36  --      correction_v = V - recall               [activation]
    -- v recall -> correction_v

    \ 37  **      correction = beta_sig * correction_v    [WEIGHT-IMM: beta_sig]
    ** beta_sig correction_v -> correction

    \ 38  ***     update_mat = correction outer K         [activation: outer product]
    *** correction k -> update_mat

    \ 39  **      decay_S = decay * S                     [activation]
    ** decay state -> decay_S

    \ 40  +++     S = decay_S + update_mat                [activation: matrix add]
    +++ decay_S update_mat -> state

    \ 41  matvec  output = S @ Q                          [activation: state GEMV]
    matvec state q -> attn_out

    \ ============================================================
    \ Output gate (steps 42-47)
    \ Step 42 is a weight-immediate projection.
    \ Steps 43-47 are activation compute (sigmoid, gate multiply).
    \ ============================================================

    \ 42  project  z = W_z @ x (already done above)      [WEIGHT-IMM: W_z]
    \ (z was projected above with the other projections)

    \ 43  * -1                                            [activation]
    \ 44  exp                                             [activation]
    \ 45  ++                                              [activation]
    \ 46  //     sigmoid(z)                               [activation]
    \ 47  **     output * sigmoid(z)                      [activation]
    silu z -> z_gate
    ** attn_out z_gate -> gated_out

    \ ============================================================
    \ RMSNorm on gated output (steps 48-53)
    \ Steps 48-52: activation compute
    \ Step 53: uses w_post (small weight vector -> immediates)
    \ ============================================================

    \ 48-52  RMSNorm reduction                            [activation]
    \ 53     ** w  scale by learned weight                [WEIGHT-IMM: w_post]
    rmsnorm gated_out w_post -> normed_out

    \ ============================================================
    \ Output projection + residual (steps 54-55)
    \ Step 54: weight-immediate projection (W_o)
    \ Step 55: residual add (activation)
    \ ============================================================

    \ 54  project  out = W_o @ normed_out                 [WEIGHT-IMM: W_o]
    project W_o normed_out -> proj_out

    \ 55  ++  X = X + proj_out                            [activation]
    ++ X proj_out -> X

    \ ============================================================
    \ RMSNorm on X for MLP (steps 56-61)
    \ ============================================================

    \ 56-60  RMSNorm reduction                            [activation]
    \ 61     ** w  scale by norm weight                   [WEIGHT-IMM: w_mlp]
    rmsnorm X w_mlp -> mlp_in

    \ ============================================================
    \ MLP: gate + up projections (steps 62-63)
    \ Both are weight-immediate streams.
    \ ============================================================

    \ 62  project  gate = W_g @ mlp_in                    [WEIGHT-IMM: W_g]
    project W_g mlp_in -> gate

    \ 63  project  up = W_u @ mlp_in                      [WEIGHT-IMM: W_u]
    project W_u mlp_in -> up

    \ ============================================================
    \ SiLU on gate (steps 64-68)
    \ Pure activation compute.
    \ ============================================================

    \ 64-68  SiLU(gate)                                   [activation]
    silu gate -> gate

    \ ============================================================
    \ Gate * up (step 69)
    \ ============================================================

    \ 69  **  gate * up                                   [activation]
    ** gate up -> mlp_hidden

    \ ============================================================
    \ Down projection + residual (steps 70-71)
    \ Step 70: weight-immediate stream (W_d)
    \ Step 71: residual add (activation)
    \ ============================================================

    \ 70  project  down = W_d @ mlp_hidden                [WEIGHT-IMM: W_d]
    project W_d mlp_hidden -> down_out

    \ 71  ++  X = X + down_out                            [activation]
    ++ X down_out -> X
```

---

## Step-by-Step Compilation Summary

| Step | Primitive      | Category      | Compiler action                                             |
|------|---------------|---------------|-------------------------------------------------------------|
| 1-5  | ** Sigma / 1/sqrt // | activation | Emit standard SASS: FMUL, shuffle reduce, FMUL RSQRT, FMUL |
| 6    | ** w_pre       | WEIGHT-IMM    | 5120 FMUL-IMM instructions (one per hidden dim element)     |
| 7    | project W_q    | WEIGHT-IMM    | ~26M FMUL-IMM instructions (5120 x 5120 / thread partition) |
| 8    | project W_k    | WEIGHT-IMM    | ~10M FMUL-IMM (2048 x 5120 / thread partition)              |
| 9    | project W_v    | WEIGHT-IMM    | ~31M FMUL-IMM (6144 x 5120 / thread partition)              |
|      | project W_z    | WEIGHT-IMM    | ~31M FMUL-IMM (6144 x 5120)                                |
|      | project W_a    | WEIGHT-IMM    | ~245K FMUL-IMM (48 x 5120)                                 |
| 10-13| ** Sigma sqrt //    | activation | Standard SASS for L2-norm                                  |
| 14-18| silu           | activation    | Standard SASS for SiLU                                     |
| 19-22| l2norm         | activation    | Standard SASS for L2-norm                                  |
| 23-26| sigmoid(beta)  | ELIDED        | Pre-computed at compile time. Zero runtime instructions.    |
| 27   | ++ dt_bias     | WEIGHT-IMM    | 48 FADD-IMM instructions                                   |
| 28-30| softplus       | activation    | Standard SASS                                               |
| 31-33| * neg_exp_A    | WEIGHT-IMM    | 48 FMUL-IMM instructions (pre-computed constant)            |
| 34   | exp            | activation    | Standard SASS                                               |
| 35   | matvec S K     | activation    | LDG from state HBM, FMA, shuffle reduce                    |
| 36   | -- V recall    | activation    | FSUB                                                        |
| 37   | ** beta_sig    | WEIGHT-IMM    | 48 FMUL-IMM (pre-computed sigmoid(beta))                    |
| 38   | *** outer      | activation    | FMA outer product                                           |
| 39   | ** decay S     | activation    | FMUL elementwise on state                                  |
| 40   | +++ mat add    | activation    | FADD matrix add                                             |
| 41   | matvec S Q     | activation    | LDG from state, FMA, shuffle reduce                        |
| 42-47| sigmoid, gate  | activation    | Standard SASS for silu + elementwise mul                    |
| 48-52| rmsnorm reduce | activation    | Standard SASS                                               |
| 53   | ** w_post      | WEIGHT-IMM    | 5120 FMUL-IMM                                              |
| 54   | project W_o    | WEIGHT-IMM    | ~26M FMUL-IMM (5120 x 6144 / thread partition)             |
| 55   | ++ residual    | activation    | FADD                                                        |
| 56-60| rmsnorm reduce | activation    | Standard SASS                                               |
| 61   | ** w_mlp       | WEIGHT-IMM    | 5120 FMUL-IMM                                              |
| 62   | project W_g    | WEIGHT-IMM    | ~71M FMUL-IMM (13824 x 5120)                               |
| 63   | project W_u    | WEIGHT-IMM    | ~71M FMUL-IMM (13824 x 5120)                               |
| 64-68| silu           | activation    | Standard SASS                                               |
| 69   | ** gate up     | activation    | FMUL elementwise                                            |
| 70   | project W_d    | WEIGHT-IMM    | ~71M FMUL-IMM (5120 x 13824)                               |
| 71   | ++ residual    | activation    | FADD                                                        |

---

## Design Principles

### 1. Templates, not kernels

The .ls file specifies the computation graph. The compiler instantiates it per
layer. The same template with different weight bindings produces different SASS
but identical structure.

### 2. Weights are instruction selection, not data

In traditional GPU computing, weights are data loaded from memory. In Lithos v2,
weights are instruction selection -- they determine WHICH instructions are emitted
(specifically, the immediate field of each FMUL). The "model" and the "program"
are the same artifact.

### 3. No dequantization at runtime

All quantization formats (Q4, Q8, etc.) are resolved at compile time. The
compiler reads packed tensors, unpacks them, applies scales and zero points, and
produces FP32 constants. The generated SASS has no knowledge of quantization.

### 4. Pre-computation of constant expressions

When a computation involves only compile-time weights (e.g., `sigmoid(beta)`,
`-exp(A_log)`), the compiler evaluates it at compile time and emits the result
as a single immediate. This is constant folding applied to neural network
parameters.

### 5. Clear runtime/compile-time boundary

Every value in the system is either:
- **Compile-time** (`weight`): known from safetensors, becomes an instruction immediate
- **Runtime** (`runtime`): depends on input data, lives in registers or HBM

There is no ambiguity. The `weight` and `runtime` keywords make the distinction
explicit in the source language. The compiler enforces it: you cannot index into
a `weight` at runtime, and you cannot embed a `runtime` value as an immediate.

### 6. Composites remain as named sequences

`rmsnorm`, `l2norm`, `silu`, `softplus` remain as composite names for sequences
of primitives. They are not new primitives. The compiler expands them into their
constituent primitive sequences, then applies weight-immediate substitution where
applicable (e.g., the `** w` step inside rmsnorm uses weight immediates for the
norm scale vector).

---

## Impact on Binary Size

For a single layer with the dimensions above, the weight-immediate projections
produce on the order of hundreds of millions of FMUL-IMM instructions. At 16
bytes per SASS instruction, a single layer's projections occupy several
gigabytes of instruction memory.

For a 64-layer model, the total binary size is roughly equivalent to the model
size in FP32 (since each Q4 weight expands to an FP32 immediate). This is the
fundamental tradeoff: we trade memory bandwidth (no weight loads from HBM) for
instruction cache pressure (the program itself is huge).

This works because:
- GPU instruction caches stream sequentially through the projection blocks
- There is no random access into weights -- the instruction pointer advances linearly
- The instruction fetch unit has dedicated bandwidth separate from the memory subsystem
- Each SM executes its partition of rows independently (same as old tiled GEMV)

The "model binary" for a 7B Q4 model is approximately 14 GB of SASS (3.5B
parameters x 4 bytes FP32 immediate per instruction, encoded in 16-byte SASS
instructions, but the immediate replaces what would have been a memory operand,
so the instruction count is the same -- only the encoding changes).

---

## Transition from v1

| v1 concept                    | v2 replacement                              |
|-------------------------------|---------------------------------------------|
| `param W_packed ptr`          | `weight W_q : [M, N] q4`                    |
| `W_packed [ idx ]`            | (gone -- no runtime weight access)           |
| nibble extract + dequant loop | (gone -- compile-time dequant)               |
| `fn gptq_gemv W_packed ...`   | `project W_q x -> y`                        |
| one kernel for all layers     | template instantiated per layer              |
| safetensors loaded at runtime | safetensors read at compile time             |
| `scales [ grp ]`              | (gone -- scale applied at compile time)      |
| runtime dequant instructions  | (gone -- zero runtime dequant)               |
