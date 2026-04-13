DeltaNet Inference
  Embed
  Layer × 64
  FinalNorm
  LMHead
  Sample


Layer
  RMSNorm
  Attention
  Residual
  RMSNorm
  MLP
  Residual


RMSNorm: Normalize then scale by γ

Normalize
  square each element
  Σ
  / N
  √
  reciprocal
  multiply each element


Attention: either DeltaNet or FullAttention depending on layer index

DeltaNet
  QKVProjection
  Conv1d on Q
  Conv1d on K
  L2Norm on K
  SiLU on Q
  L2Norm on Q
  DecayGate
  DeltaUpdate
  S @ Q
  ZProjection
  OutputGate
  RMSNorm
  OutputProjection


FullAttention
  QKVProjection
  RoPE
  ScaledDotProduct
  Softmax
  × V
  OutputProjection


MLP: SwiGLU
  GateProjection
  UpProjection
  SiLU on gate
  multiply gate by up
  DownProjection


QKVProjection
  Q = W_q · x
  K = W_k · x
  V = W_v · x


OutputProjection
  W_o · x


L2Norm
  square each element
  Σ
  √
  reciprocal
  multiply each element


SiLU
  -x
  exp
  + 1
  reciprocal
  multiply by x


Softmax
  max
  subtract max
  exp each
  Σ
  reciprocal
  multiply each


DecayGate
  a = a_proj + dt_bias
  softplus a
  exp A_log
  -
  multiply
  exp


softplus
  exp
  + 1
  log


DeltaUpdate
  K ⊗ K
  S · (K ⊗ K)
  V ⊗ K
  -
  β ·
  decay · S
  +


OutputGate
  Z = W_z · x
  sigmoid Z
  multiply


sigmoid
  -x
  exp
  + 1
  reciprocal


RoPE
  rotate Q by (cos, sin)
  rotate K by (cos, sin)


Conv1d
  shift history
  multiply by weights
  Σ


Embed
  lookup token id in embedding table


LMHead
  W_vocab · x


Sample
  argmax
