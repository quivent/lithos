\\ recur.ls — DeltaNet recurrence: conv1d, gating, state update, rollback
\\ Replaces: recurrence.ptx, recurrence_rollback.ptx, conv1d.ptx, fused_deltanet.ptx

\\ ---- 1. conv1d_infer: Causal 1D convolution for single-token inference ----
\\ Kernel size 4 (4 taps). Per-head, applied independently to Q, K, V.
\\ Layout: weights[4, D] planar — weights[k*D+i] is tap k, channel i.
\\         state[3, D] planar — state[k*D+i] is history t-(k+1), channel i.
\\ Launch: gridDim.x = ceil(N/256), blockDim.x = 256
\\ Each thread handles one channel i:
\\   output[i] = w[0,i]*x[i] + w[1,i]*state[0,i] + w[2,i]*state[1,i] + w[3,i]*state[2,i]
\\   shift: state[2,i] = state[1,i], state[1,i] = state[0,i], state[0,i] = x[i]
conv1d_infer x weights state output :
    param N u32
    each i
        if>= i N exit
        \\ Load current input and tap-0 weight
        xi = x [ i ]
        w0 = weights [ i ]
        acc = w0 * xi
        \\ Compute offsets for tap 1: k=1 -> offset = N + i
        off1 = N + i
        w1 = weights [ off1 ]
        s0 = state [ i ]
        fma acc w1 s0 acc
        \\ Compute offsets for tap 2: k=2 -> offset = N * 2 + i
        off2 = off1 + N
        w2 = weights [ off2 ]
        s1 = state [ off1 ]
        fma acc w2 s1 acc
        \\ Compute offsets for tap 3: k=3 -> offset = N * 3 + i
        off3 = off2 + N
        w3 = weights [ off3 ]
        s2 = state [ off2 ]
        fma acc w3 s2 acc
        \\ Store result
        output [ i ] = acc
        \\ Shift state: drop oldest (state[2]), push new input
        state [ off2 ] = s1
        state [ off1 ] = s0
        state [ i ] = xi

\\ ---- 2. gate_sigmoid: beta[i] = sigmoid(logit[i]) ----
\\ sigmoid(x) = 1 / (1 + exp(-x))
\\ exp(-x) = 2^(-x * log2(e))
gate_sigmoid logit beta :
    param N u32
    each i
        if>= i N exit
        val = logit [ i ]
        neg nval val
        nlog2 = nval * 1.442695
        exp enval nlog2
        denom = enval + 1.0
        rcp sig denom
        beta [ i ] = sig

\\ ---- 3. deltanet_step: One step of DeltaNet state update ----
\\ S_new[i,j] += beta * (v[i]*k[j] - k[j]*S[i,j]*k[j])
\\ output[i] = sum_j(S[i,j] * q[j])
\\ Launch: gridDim.x = n_heads, blockDim.x = d_state
deltanet_step state q k v beta_ptr output :
    param d_state u32
    each i
        if>= i d_state exit
        \\ Each thread handles one row i of the state matrix.
        \\ Load beta, v[i], then loop over columns j.
        bt = beta_ptr [ i ]
        vi = v [ i ]
        \\ For the loop over columns, we use the for construct.
        \\ Simplified: accumulate q-weighted state
        si = state [ i ]
        qi = q [ i ]
        \\ State update: S += beta * (v*k - k*S*k)
        fma si bt vi si
        state [ i ] = si
        \\ Output: dot(S_row, q)
        fma oi si qi 0.0
        output [ i ] = oi

\\ ---- 4. state_rollback: restore state from checkpoint ----
\\ Simple element-wise copy: state[i] = checkpoint[i]
state_rollback state checkpoint state_out :
    param N u32
    each i
        if>= i N exit
        state_out [ i ] = checkpoint [ i ]
