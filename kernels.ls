# kernels — composed operations
# Each body line is a stack op: `op operand` applies acc = acc OP operand.
# Bare op = unary (* alone = square). First line is initial value.
# Called functions (exp, sigmoid, ...) act on the accumulator as their arg.


# ==== top-level composition ====

DeltaNet_Inference
  Embed
  Layer × 64
  RMSNorm
  LMHead
  Sample


Layer
  RMSNorm
  Attention
  Residual
  RMSNorm
  MLP
  Residual


Attention
  dispatch to DeltaNet or FullAttention by layer index


DeltaNet
  QKVProjection
  Conv1d on Q
  Conv1d on K
  L2Norm on K
  SiLU on Q
  L2Norm on Q
  DecayGate
  DeltaUpdate
  MatVec S Q
  ZProjection
  OutputGate
  RMSNorm
  OutputProjection


FullAttention
  QKVProjection
  RoPE
  ScaledDotProduct
  Softmax
  MatVec V_cache scores
  OutputProjection


MLP    # SwiGLU
  GateProjection
  UpProjection
  SiLU on gate
  multiply gate by up
  DownProjection


# ==== scalar utilities ====

negate x
  0
  - x


reciprocal x
  1
  / x


exp x
  e
  ^ x


square x
  x
  *


abs x
  x
  *
  √


sigmoid x
  0
  - x                // -x
  exp                // e^(-x)
  + 1                // 1 + e^(-x)
  reciprocal         // 1 / (1 + e^(-x))


SiLU x
  x
  sigmoid            // sigmoid(x)
  * x                // x * sigmoid(x)


softplus x
  x
  exp                // e^x
  + 1                // 1 + e^x
  log                // log(1 + e^x)


# ==== 2-D rotation (pair) ====

rotate_x x y c s
  x
  * c
  - (y * s)          // x*c - y*s


rotate_y x y c s
  x
  * s
  + (y * c)          // x*s + y*c


# ==== vector ops ====

dot u v
  u
  * v                // elementwise u[i]*v[i]
  Σ


Normalize x
  x
  *                  // square each element
  Σ
  / N
  √
  reciprocal
  * x                // scalar * x → vector


L2Norm x
  x
  *                  // square each
  Σ
  √
  reciprocal
  * x


RMSNorm x γ
  x
  Normalize
  * γ                // scale by γ


Residual x saved
  x
  + saved


Softmax x
  x
  - (max x)          // subtract max elementwise
  exp                // exp each
  / (Σ acc)          // divide each by sum of current acc


Sample logits
  logits
  argmax


Embed token_id
  lookup token_id in embedding table


# ==== projections ====

MatVec W x
  each row i of W
    dot W[i] x


QKVProjection x
  MatVec W_q x
  MatVec W_k x
  MatVec W_v x


OutputProjection x    → MatVec W_o x
ZProjection      x    → MatVec W_z x
GateProjection   x    → MatVec W_gate x
UpProjection     x    → MatVec W_up x
DownProjection   x    → MatVec W_down x
LMHead           x    → MatVec W_vocab x


# ==== DeltaNet pieces ====

DecayGate a_proj dt_bias A_log
  a_proj
  + dt_bias
  softplus
  * (negate (exp A_log))     // multiply by -exp(A_log)
  exp                         // exp of the product


DeltaUpdate S K V β decay
  outer K K
  matmul S (outer K K)
  outer V K
  subtract
  * β
  + (decay * S)


OutputGate x output
  ZProjection x
  sigmoid
  * output


ScaledDotProduct Q K_cache d_head
  each K in K_cache
    dot Q K
    / √d_head


RoPE Q K pos
  each pair (Q_x, Q_y) at position pos
    rotate_x Q_x Q_y cos[pos] sin[pos]
    rotate_y Q_x Q_y cos[pos] sin[pos]
  each pair (K_x, K_y)
    rotate_x K_x K_y cos[pos] sin[pos]
    rotate_y K_x K_y cos[pos] sin[pos]


Conv1d input weights history
  shift_history
  input
  * weights
  Σ


shift_history
  history[0] = history[1]
  history[1] = history[2]
  history[2] = history[3]
  history[3] = new input


outer u v
  each i
    each j
      u[i] * v[j]


matmul A B
  each row i of A
    each col j of B
      dot A[i] B[:,j]


# ==== leaves still needing primitives in primitives.ls ====
# log, sin, cos
