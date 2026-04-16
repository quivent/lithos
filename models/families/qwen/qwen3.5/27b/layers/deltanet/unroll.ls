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

QKVProjection [][]        \\ matrix
  project
    ****

ZProjection [][]          \\ matrix
  project
    ****

ShortConv1D []            \\ vector
  convolve
    ** w
    Σ

SiLU []                   \\ vector
  activate
    sigmoid
      * -1
      e^
      + 1
      1/
    *

Split []                  \\ vector
  slice

L2Norm []                 \\ vector
  unitize
    **
    Σ
    1/√
    **

Scale []                  \\ vector
  rescale
    * 0.08838834764831845

BetaProjection [][]       \\ matrix
  project
    ****

Sigmoid []                \\ vector
  squash
    * -1
    e^
    + 1
    1/

DecayProjection [][]      \\ matrix
  project
    ****

DecayGate []              \\ vector
  decay
    + dt_bias
    softplus
      e^
      + 1
      ln
    e^
    * -1
    *
    e^

Recurrence [][][]         \\ layer
  recur
    ** decay
    *** K
    -
    * beta
    *** K
    ++
    *** Q

PerHeadRMSNorm [][]       \\ matrix
  normalize
    **
    Σ
    / D
    + 1e-6
    1/√
    ** x
    ** w

OutputGate [][]           \\ matrix
  gate
    silu
      sigmoid
        * -1
        e^
        + 1
        1/
      *
    **

OutputProjection [][]     \\ matrix
  project
    ****

Residual []               \\ vector
  add
    +
