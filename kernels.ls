\\ kernels — Lithos composition language
\\
\\ - Each body line is one op on the stack.
\\ - A bare name (arg, literal, primitive constant) pushes to the stack.
\\ - `op operand` = push operand, then pop-two binary op.
\\ - Bare binary op = pop-two, combine.
\\ - Bare `*` with stack depth 1 = square (implicit dup).
\\ - Bare unary (√, Σ, ln, ≅, ≡) = apply to top.
\\ - A named kernel consumes its args from the stack, pushes its result.
\\ - `acc` = dup top.


\\ ==== top-level composition ====

DeltaNet_Inference :
  Embed
  Layer * 64
  RMSNorm
  LMHead
  Sample


Layer :
  RMSNorm
  Attention
  Residual
  RMSNorm
  MLP
  Residual


Attention :
  dispatch to DeltaNet or FullAttention by layer index


DeltaNet :
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


FullAttention :
  QKVProjection
  RoPE
  ScaledDotProduct
  Softmax
  MatVec V_cache scores
  OutputProjection


MLP :    \\ SwiGLU
  GateProjection
  UpProjection
  SiLU on gate
  ** gate by up
  DownProjection


\\ ==== scalar utilities ====

negate x :
  0
  - x


reciprocal x :
  1
  / x


exp x :
  e^ x


square x :
  x
  *


abs x :
  x
  *
  √


sigmoid x :
  0
  - x
  e^
  + 1
  reciprocal


SiLU x :
  x
  sigmoid
  * x


softplus x :
  x
  e^
  + 1
  ln


\\ ==== 2-D rotation (pair) ====

rotate_x x y c s :
  x
  * c
  y
  * s
  -


rotate_y x y c s :
  x
  * s
  y
  * c
  +


\\ ==== vector ops ====

dot u v :
  u
  * v
  Σ


Normalize x :
  x
  *
  Σ
  / N
  √
  reciprocal
  * x


L2Norm x :
  x
  *
  Σ
  √
  reciprocal
  * x


RMSNorm x γ :
  x
  Normalize
  * γ


Residual x saved :
  x
  + saved


Softmax x :
  x
  x
  △
  -
  e^
  acc
  Σ
  /


Sample logits :
  logits
  # △


Embed token_id :
  lookup token_id in embedding table


\\ ==== projections ====

MatVec W x :
  each row i of W
    dot W[i] x


QKVProjection x :
  MatVec W_q x
  MatVec W_k x
  MatVec W_v x


OutputProjection x :
  MatVec W_o x

ZProjection x :
  MatVec W_z x

GateProjection x :
  MatVec W_gate x

UpProjection x :
  MatVec W_up x

DownProjection x :
  MatVec W_down x

LMHead x :
  MatVec W_vocab x


\\ ==== DeltaNet pieces ====

DecayGate a_proj dt_bias A_log :
  a_proj
  + dt_bias
  softplus
  e^ A_log
  negate
  *
  e^


DeltaUpdate S K V β decay :
  V
  K
  ***
  S
  K
  K
  ***
  matmul
  ---
  * β
  S
  * decay
  +++


OutputGate x output :
  ZProjection x
  sigmoid
  * output


ScaledDotProduct Q K_cache d_head :
  each K in K_cache
    Q
    K
    dot
    √ d_head
    /


RoPE Q K pos :
  each pair Q_x Q_y
    rotate_x Q_x Q_y ≡ pos ≅ pos
    rotate_y Q_x Q_y ≡ pos ≅ pos
  each pair K_x K_y
    rotate_x K_x K_y ≡ pos ≅ pos
    rotate_y K_x K_y ≡ pos ≅ pos


Conv1d input weights history :
  shift_history
  input
  * weights
  Σ


shift_history :
  ← 32 history 0 → 32 history 1
  ← 32 history 1 → 32 history 2
  ← 32 history 2 → 32 history 3
  ← 32 history 3 input


outer u v :
  each i
    each j
      u[i] * v[j]


matmul A B :
  each row i of A
    each col j of B
      dot A[i] B[:,j]
