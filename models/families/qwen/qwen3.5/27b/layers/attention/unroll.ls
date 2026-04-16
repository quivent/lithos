RMSNorm []                \\ vector
  normalize
    **
    Σ
    / D
    + 1e-6
    1/√
    ** x
    + 1
    ** w

QProjection [][]          \\ matrix
  project
    ****

KProjection [][]          \\ matrix
  project
    ****

VProjection [][]          \\ matrix
  project
    ****

GateProjection [][]       \\ matrix
  project
    ****

PerHeadRMSNormQ [][]      \\ matrix
  normalize
    **
    Σ
    / D
    + 1e-6
    1/√
    ** x
    ** w

PerHeadRMSNormK [][]      \\ matrix
  normalize
    **
    Σ
    / D
    + 1e-6
    1/√
    ** x
    ** w

MRoPEQ [][]               \\ matrix
  rotate
    * 2
    / D
    * -23.25
    2^
    * pos
    ≡
    ≅
    *
    -
    *
    +

MRoPEK [][]               \\ matrix
  rotate
    * 2
    / D
    * -23.25
    2^
    * pos
    ≡
    ≅
    *
    -
    *
    +

KVCacheAppendK [][][]     \\ layer
  append
    ←

KVCacheAppendV [][][]     \\ layer
  append
    ←

AttentionScores [][]      \\ matrix
  score
    **** K
    * 1/√ D
    softmax
      △
      -
      e^
      Σ
      1/
      **

AttentionOutput [][]      \\ matrix
  weight
    **** V
    sigmoid
      * -1
      e^
      + 1
      1/
    **

OutputProjection [][]     \\ matrix
  project
    ****

Residual []               \\ vector
  add
    +
