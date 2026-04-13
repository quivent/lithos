# DeltaNet Primitive Decomposition Map
#
# Format: Name = alias = [decomposition]
# Primitives: + - / * ^ e √ Σ @
# * alone after value = square (x * = x²)
# * between two values = element-wise multiply
# A value can be used in multiple places — that's just math, not an operation.

# --- Activations ---
sigmoid         = gate     = [* -1, e ^, + 1, 1 /]
SiLU            = swish    = [x * sigmoid(x)]
softplus        = write    = [e ^, + 1, ln]

# --- Norms ---
RMSNorm         = normalize = [x *, Σ, / n, + ε, √, 1 /, *, * (1 + w)]
L2Norm          = unitize   = [x *, Σ, + ε, √, 1 /, *]

# --- Linear algebra ---
Projection      = project   = [W @]
OuterProduct    = bind       = [v ⊗ k]
MatVecReadout   = recall     = [S @]

# --- Temporal ---
Convolution     = slide      = [drop oldest, append new, * kernel, Σ]

# --- Recurrence (delta rule) ---
Decay           = forget     = [α *]
Write           = engram     = [softplus(β) * OuterProduct(v, k)]
StateUpdate     = remember   = [Decay(S), + Write]
Readout         = attend     = [MatVecReadout(S, q)]

# --- Gating ---
OutputGate      = filter     = [SiLU(z) * attn_out]

# --- Structural ---
Residual        = accrete    = [+ residual]
LayerScale      = temper     = [* (scale + 1)]

# --- Full DeltaNet layer (attn-only) ---
# x[5120] -> out[5120]
#
# 1. RMSNorm(x)           = normalize
# 2. Projection(x) -> qkv = project
# 3. Projection(x) -> z   = project
# 4. Projection(x) -> β,α = project
# 5. Convolution(qkv)     = slide
# 6. StateUpdate(S,k,v,α,β) = remember
# 7. Readout(S,q)         = attend
# 8. L2Norm(attn)         = unitize       (36/48 layers)
# 9. OutputGate(z, attn)  = filter
# 10. Projection(gated)   = project
# 11. LayerScale           = temper
# 12. Residual             = accrete

# --- PROBLEMS ---
# softplus uses ln (natural log) which is not in the primitive set.
# ln is the inverse of e^ but we have no way to express it with
# just + - / * ^ e √ Σ @. Either:
#   (a) add ln as a primitive, or
#   (b) note that softplus ≈ x for large x, ≈ 0 for very negative x,
#       and ask whether an approximation suffices
