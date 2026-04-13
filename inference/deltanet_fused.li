\ deltanet_fused.li — Fused GatedDeltaNet single-token decode recurrence
\ Qwen3.5-27B linear_attention layers (linear_key_head_dim=128, GQA ratio=3)
\
\ Fuses: delta-rule state update + output gate (silu(z)) in one kernel.
\ Replaces separate calls to: deltanet_step + gate_sigmoid + elemwise_mul
\
\ Memory layout after QKV+Z projection (activation buffer, f32):
\   q[16, 128]   offset 0           (2048 floats)
\   k[16, 128]   offset 2048        (2048 floats)
\   v[48, 128]   offset 4096        (6144 floats)
\   z[48, 128]   offset 10240       (6144 floats)
\ decay[48]      separate pointer   (scalar per value head)
\ state[16, 3, 128, 128] f32        separate pointer (per-layer, persistent)
\
\ Launch: gridDim.x = 16 (num_key_heads), blockDim.x = 128 (key_head_dim)
\ Thread j = lane = column index in state matrix = tid.x (0..127)
\   khead = blockIdx.x = global_tid >> 7
\   lane  = threadIdx.x = global_tid & 127
\
\ Per block (one key head):
\   1. Load K[khead, *] into smem_k cooperatively
\   2. Loop over 3 value heads (voff = 0..2):
\      a. kT_S[lane] = sum_row smem_k[row] * S[khead,voff,row,lane]
\      b. delta[lane] = V[vhead, lane] - kT_S[lane]
\      c. S[khead,voff,row,lane] = decay[vhead] * S[*] + K[row] * delta[lane]
\      d. out[lane]  = sum_row S_updated[row,lane] * Q[khead, row]  (via smem_q)
\      e. out[lane] *= silu(Z[vhead, lane])
\      f. output[vhead, lane] = out[lane]
\
\ Notes:
\   - smem_k loaded once before value-head loop (K shared across all 3 value heads)
\   - smem_q loaded once before value-head loop (Q same for all 3 value heads)
\   - State stride: d*d = 16384 floats per (khead, voff) slice
\   - khead * 3 * 16384 = khead * 49152 floats to start of key head's state
\   - Uses shl 7 for multiply-by-128 (d=128=2^7), shl 14 for d*d

fn deltanet_fused q k v z decay beta state -> output
    param d u32         \ key/value head dim = 128
    param gqa u32       \ value heads per key head = 3
    shared smem_k 128 f32
    shared smem_q 128 f32

    each gtid
    \ gtid = global thread ID = khead * 128 + lane
    \ Decompose: khead = gtid >> 7,  lane = gtid & 127
    shr khead gtid 7
    and lane  gtid 127

    \ ---- Cooperatively load K[khead, 0..127] and Q[khead, 0..127] ----
    \ k_base = khead * 128 (= khead << 7)
    shl k_base khead 7
    add k_abs k_base lane
    k_lane = k [ k_abs ]
    smem_k [ lane ] = k_lane

    shl q_base khead 7
    add q_abs q_base lane
    q_lane = q [ q_abs ]
    smem_q [ lane ] = q_lane

    barrier

    \ ---- State base for this key head ----
    \ khead_state_base = khead * gqa * d * d
    \   = khead * 3 * 128 * 128 = khead * 49152
    \   = khead * 3 * (1 << 14)
    \ Compute as: khead * 3, then << 14
    mad kh3 khead 3 0         \ kh3 = khead * 3
    shl khead_s_base kh3 14   \ khead_s_base = kh3 * 16384

    \ ---- First value head for this key head ----
    mad vhead_base khead 3 0  \ vhead_base = khead * gqa = khead * 3

    \ ---- Loop over 3 value heads (voff = 0..2) ----
    for voff 0 gqa 1

        \ vhead = vhead_base + voff
        add vhead vhead_base voff

        \ Decay scalar for this value head
        dec = decay [ vhead ]

        \ Beta scalar for this value head (same per-head shape as decay)
        bet = beta [ vhead ]

        \ V[vhead, lane] — v_base = vhead * 128 + lane
        shl v_vbase vhead 7
        add v_abs v_vbase lane
        v_val = v [ v_abs ]

        \ State slice base for [khead, voff]:
        \ s_base = khead_s_base + voff * d * d
        \        = khead_s_base + voff << 14
        shl voff_s voff 14
        add s_base khead_s_base voff_s

        \ ---- Phase 1: kT_S[lane] = sum_row K[row] * S[khead,voff,row,lane] ----
        \ Thread j (lane) accumulates the dot product of K with column lane of S.
        kts = 0.0
        for row 0 d 1
            \ S index: s_base + row * d + lane = s_base + (row << 7) + lane
            shl row_s row 7
            add s_rabs s_base row_s
            add s_abs2 s_rabs lane
            s_val = state [ s_abs2 ]
            k_row = smem_k [ row ]
            fma kts k_row s_val kts
        endfor

        \ ---- Phase 2: delta[lane] = V[lane] - kT_S[lane] ----
        delta = v_val - kts
        \ Scale delta by beta: delta = beta * delta
        delta = bet * delta

        \ ---- Phase 3: S update in place ----
        \ S[row, lane] = decay * S[row, lane] + K[row] * delta[lane]
        \ Also accumulate output = S_updated @ Q in same pass (post-update attention).
        acc = 0.0
        for row2 0 d 1
            shl row2_s row2 7
            add s2_rabs s_base row2_s
            add s2_abs s2_rabs lane
            s_old = state [ s2_abs ]
            k_r = smem_k [ row2 ]
            s_dec = dec * s_old
            fma s_new k_r delta s_dec
            state [ s2_abs ] = s_new
            \ Accumulate output: out[lane] += S_updated[row2, lane] * Q[khead, row2]
            q_row = smem_q [ row2 ]
            fma acc s_new q_row acc
        endfor

        \ ---- Phase 4: Output gate: out *= silu(Z[vhead, lane]) ----
        \ silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
        shl z_vbase vhead 7
        add z_abs z_vbase lane
        z_val = z [ z_abs ]
        neg nz z_val
        nz_l2 = nz * 1.442695
        exp ez nz_l2
        denom = ez + 1.0
        rcp sig denom
        silu_z = z_val * sig
        out_val = acc * silu_z

        \ ---- Write output[vhead, lane] ----
        shl o_vbase vhead 7
        add o_abs o_vbase lane
        output [ o_abs ] = out_val

    endfor
