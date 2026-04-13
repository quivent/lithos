# LEGACY: .li dialect. Replaced by kernels.ls (.ls dialect).
# Multi-function .li files produced broken cubins (only last function kept).

\ attend.li — Attention: RoPE + causal softmax attention with KV cache + GQA
\ Replaces: attention_score.ptx, rotate.ptx, fused_attention.ptx
\
\ 1. RoPE: rotate Q and K vectors using sin/cos of position-dependent frequencies
\ 2. attention_score: online softmax(Q * K^T / sqrt(d)) * V with GQA and causal mask

\ ---- 1. rope ----
\ Launch: gridDim.x = n_heads, blockDim.x = d_head/2
\ Each thread rotates one pair of (even, odd) elements in Q or K.
fn rope q k -> q_out k_out
    param d_head u32
    param pos u32
    each i

    \ Compute rotation angle: theta_i = 1/(10000^(2i/d)) = exp2(-log2(10000)*2i/d)
    \ log2(10000) = 13.2877
    u32>f32 fi i
    u32>f32 fd d_head
    freq_ratio = fi * 2.0 / fd              \ 2*i / d
    freq = freq_ratio * -13.2877            \ -log2(10000) * 2i/d
    theta_i = exp2 freq                     \ MUFU.EX2: theta_i = 10000^(-2i/d)
    u32>f32 fpos pos
    angle = fpos * theta_i                  \ pos * theta_i

    cos cs angle
    sin sn angle

    \ Load even element at i and odd element at i+d_head/2
    q0 = q [ i ]
    shr half_d d_head 1
    add i_odd i half_d
    q1 = q [ i_odd ]

    \ Complex rotation: (q0 + j*q1) * (cos + j*sin)
    q_rot_re = q0 * cs - q1 * sn
    q_rot_im = q0 * sn + q1 * cs
    q_out [ i ]     = q_rot_re
    q_out [ i_odd ] = q_rot_im

    \ Same rotation for K
    k0 = k [ i ]
    k1 = k [ i_odd ]
    k_rot_re = k0 * cs - k1 * sn
    k_rot_im = k0 * sn + k1 * cs
    k_out [ i ]     = k_rot_re
    k_out [ i_odd ] = k_rot_im

\ ---- 2. attention_score ----
\ Real online-softmax attention (Flash Attention decode style).
\ Launch: gridDim.x = n_q_heads, blockDim.x = d_head (one thread per head dim)
\
\ Each thread owns one d-dimension lane across all KV positions.
\ For each KV position:
\   1. Load K into shared memory with a barrier.
\   2. Compute partial dot product Q[lane] * K_smem[lane], warp-reduce to get score.
\   3. Apply causal mask (score = -inf if pos > current_pos).
\   4. Online softmax: update running_max and running_sum, rescale accumulator.
\   5. Load V[lane] and accumulate: output[lane] += exp(score-max) * V[lane].
\ Final: output[lane] /= running_sum.
fn attention_score q k_cache v_cache -> output
    param d_head u32
    param seq_len u32
    param current_pos u32
    param n_kv_heads u32
    param n_q_heads u32
    shared smem_k 1024 f32

    each tid

    \ Exit if lane >= d_head
    if>= tid d_head exit

    \ GQA: map q_head -> kv_head
    \ head index is ctaid.x = r0; tid.x = r2 in each preamble
    \ Reconstruct: lane = tid % d_head, head = tid / d_head
    shr head tid d_head   \ head = global_tid / d_head (approximate for blockDim == d_head)
    and lane tid 255      \ lane = tid & (d_head-1); works when d_head is power-of-2

    \ GQA ratio = n_q_heads / n_kv_heads (e.g. 3 for Qwen 2.5 27B: 48Q / 16KV)
    \ kv_head = head / gqa_ratio via integer division
    \ Compute gqa_ratio = n_q_heads / n_kv_heads, then kv_head = head / gqa_ratio
    \ Integer division: kv_head = head * n_kv_heads / n_q_heads
    mul head_kv head n_kv_heads
    div kv_head head_kv n_q_heads

    \ Scale = 1 / sqrt(d_head)
    u32>f32 fd_head d_head
    scale = rsqrt fd_head

    \ Online softmax state
    running_max = -8.0      \ approximates -inf (practical lower bound)
    running_sum = 0.0
    acc         = 0.0

    \ Load Q[lane] once — stays in register across seq loop
    q_val = q [ lane ]

    \ ---- Loop over KV positions with causal mask ----
    for kv_pos 0 seq_len 1

        \ Causal mask: skip positions after current_pos
        if>= kv_pos current_pos exit

        \ Load K for this head/position into shared memory
        barrier

        \ k_cache layout: [seq_len, n_kv_heads, d_head]
        \ index = kv_pos * n_kv_heads * d_head + kv_head * d_head + lane
        mad k_base kv_pos n_kv_heads kv_head  \ kv_pos*n_kv_heads + kv_head
        mad k_idx  k_base d_head lane         \ * d_head + lane
        k_val = k_cache [ k_idx ]
        smem_k [ lane ] = k_val

        barrier

        \ Dot product: q_val * smem_k[lane], then warp butterfly reduce
        dot_partial = q_val * smem_k [ lane ]
        shfl.bfly dot_partial dot_partial 16
        shfl.bfly dot_partial dot_partial 8
        shfl.bfly dot_partial dot_partial 4
        shfl.bfly dot_partial dot_partial 2
        shfl.bfly dot_partial dot_partial 1

        \ Scale score
        score = dot_partial * scale

        \ ---- Online softmax update ----
        old_max = running_max
        \ new_max = max(old_max, score)
        \ Lithos has no max intrinsic; use conditional via predication
        setp lt f32 p_update old_max score
        @p_update bra do_update_max
        bra skip_update_max

        label do_update_max
        running_max = score

        label skip_update_max
        \ correction = exp(old_max - new_max)
        correction_arg = old_max - running_max
        correction = exp correction_arg

        running_sum = running_sum * correction
        acc         = acc * correction

        \ exp(score - new_max)
        exp_arg  = score - running_max
        exp_val  = exp exp_arg
        running_sum = running_sum + exp_val

        \ v_cache layout same as k_cache
        v_val = v_cache [ k_idx ]
        fma acc exp_val v_val acc

    endfor

    \ ---- Normalize ----
    inv_sum = rcp running_sum
    acc = acc * inv_sum

    \ Write output[head, lane]
    mad out_idx head d_head lane
    output [ out_idx ] = acc
