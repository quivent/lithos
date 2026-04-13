# LEGACY: .li dialect. Replaced by kernels.ls (.ls dialect).
# Multi-function .li files produced broken cubins (only last function kept).

\ decay_gate.li — GatedDeltaNet decay gate computation
\
\ Computes per-head state decay factor for GatedDeltaNet inference.
\
\ g = -exp(A_log) * softplus(a_proj + dt_bias)
\ decay = exp(g)
\
\ Where:
\   softplus(x) = log(1 + exp(x))
\   a_proj[i]   = output of in_proj_a @ hidden_state  (per head)
\   A_log[i]    = learned log decay constant           (per head)
\   dt_bias[i]  = learned timestep bias                (per head)
\
\ Decomposed into primitives:
\   1. add       t = a_proj[i] + dt_bias[i]
\   2. exp       e = exp(t)
\   3. + 1       e1 = e + 1.0
\   4. log       sp = log(e1)             = softplus(t)
\   5. exp       A = exp(A_log[i])
\   6. * -1      nA = -A
\   7. *         g = nA * sp              = -exp(A_log) * softplus(t)
\   8. exp       decay[i] = exp(g)
\
\ Launch: gridDim.x = 1, blockDim.x = 48 (one thread per value head)
\ All inputs/outputs are [48] vectors (num_value_heads for Qwen3.5-27B).

fn decay_gate a_proj a_log dt_bias -> decay
    param N u32
    each i
        if>= i N exit

        \ Step 1: t = a_proj + dt_bias
        ai = a_proj [ i ]
        bi = dt_bias [ i ]
        t = ai + bi

        \ Steps 2-4: softplus(t) = log(1 + exp(t))
        \ Use exp via ex2: exp(t) = 2^(t * log2(e))
        t_log2 = t * 1.442695
        exp e_t t_log2
        e1 = e_t + 1.0
        lg sp e1

        \ Step 5: A = exp(A_log)
        alog = a_log [ i ]
        alog_log2 = alog * 1.442695
        exp A alog_log2

        \ Steps 6-7: g = -A * softplus(t)
        neg nA A
        g = nA * sp

        \ Step 8: decay = exp(g)
        g_log2 = g * 1.442695
        exp dec g_log2
        decay [ i ] = dec
