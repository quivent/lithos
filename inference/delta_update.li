# LEGACY: .li dialect. Replaced by kernels.ls (.ls dialect).
# Multi-function .li files produced broken cubins (only last function kept).

\ delta_update.li — GatedDeltaNet state recurrence (step 35 of STEPS decomposition)
\
\ S <- decay * S + beta * (V *** K - S . (K *** K))
\
\ Equivalent expanded form per element:
\   delta[j]   = v[j] - sum_i( k[i] * S[i,j] )     (kT @ S, per column)
\   S[i,j]     = decay * S[i,j] + beta * k[i] * delta[j]   (gated rank-1 update)
\
\ This is the GatedDeltaNet delta rule for single-token inference.
\ decay = exp(g), a per-head scalar (48 values). Gates state retention.
\ State S is [d, d] per head. K, V are [d] per head. beta is scalar per head.
\
\ Launch: gridDim.x = num_heads, blockDim.x = 128
\ Each thread owns one column j of the state matrix.
\ Thread j loops over d rows to compute kT_S[j], then updates all d rows.
\
\ Memory: S in global (64KB per head @ d=128, FP32). In-place update.
\ K, V per-head vectors offset by caller. beta, decay scalars passed as params.
\
\ Phase 1: kT_S[j] = sum_i k[i] * S[i,j]    — matrix-vector product
\ Phase 2: delta[j] = v[j] - kT_S[j]         — correction term
\ Phase 3: S[i,j] = decay * S[i,j] + beta * k[i] * delta[j]  — gated rank-1 update

fn delta_update state k v -> state_out
    param d u32
    param beta f32
    param decay f32

    each j
    if>= j d exit

    \ ---- Phase 1: kT_S[j] = sum_i k[i] * S[i,j] ----
    kts = 0.0
    for row 0 d 1
        mad sidx row d j
        ki = k [ row ]
        si = state [ sidx ]
        fma kts ki si kts
    endfor

    \ ---- Phase 2: delta[j] = v[j] - kT_S[j] ----
    vj = v [ j ]
    delta = vj - kts

    \ ---- Phase 3: S[i,j] = decay * S[i,j] + beta * k[i] * delta[j] ----
    bd = beta * delta
    for row2 0 d 1
        mad sidx2 row2 d j
        s_old = state [ sidx2 ]
        ki2 = k [ row2 ]
        s_decayed = decay * s_old
        fma s_new ki2 bd s_decayed
        state_out [ sidx2 ] = s_new
    endfor
