\\ attend_full.ls — Full multi-head attention for Qwen 3.5 27B hybrid
\\ Every 4th layer uses full attention (the rest are DeltaNet).
\\
\\ Config:
\\   num_attention_heads    = 24
\\   num_key_value_heads    = 4        \\ GQA ratio 6 (each KV head -> 6 Q heads)
\\   head_dim               = 256
\\   attn_output_gate       = true     \\ output *= sigmoid(gate)
\\   rope_theta             = 10,000,000
\\   mrope_interleaved      = true
\\   mrope_section          = [11, 11, 10]   \\ text / height / width rotation pairs
\\   partial_rotary_factor  = 0.25            \\ only 64 of 256 head dims rotated
\\
\\ Rotation layout (partial rotary, 64 dims = 32 pairs split as [11,11,10]):
\\   dims  [ 0..22)   -> text   (11 pairs,  (i, i+11))
\\   dims  [22..44)   -> height (11 pairs,  (i, i+11))
\\   dims  [44..64)   -> width  (10 pairs,  (i, i+10))
\\   dims  [64..256)  -> pass-through (no rotation)
\\
\\ Frequency base: -log2(10_000_000) = -23.2534966642
\\ theta_i = 2^(-23.2534966642 * 2i / 64)


\\ ---- 1. softmax ----
\\ Numerically stable softmax over a vector x of length N.
\\   m = max(x)
\\   e = exp(x - m)
\\   s = sum(e)
\\   y = e / s
softmax x y :
    param N u32

    \\ Pass 1: max reduction
    m △ x

    \\ Pass 2: exp(x - m) into scratch, accumulate sum
    each i
        d = x [ i ] - m
        e e^ d
        y [ i ] = e

    \\ Sum over y
    s Σ y

    \\ Pass 3: divide
    inv_s 1/ s
    each j
        y [ j ] = y [ j ] * inv_s


\\ ---- 2. mrope_apply ----
\\ Apply mRoPE to one 256-dim Q or K head vector in-place style.
\\ Only dims 0..64 rotate. Sections: text [0,22), height [22,44), width [44,64).
\\ Within each section, pair (i, i + section_len/2) rotates by angle = pos * theta.
\\
\\ Signature: x is the head vector (length 256), pos is token position,
\\ head_idx reserved for future head-specific behavior (unused today).
mrope_apply x pos head_idx :
    param head_dim u32         \\ = 256
    param rot_dim u32          \\ = 64  (partial_rotary_factor * head_dim)

    \\ log2(10_000_000) = 23.2534966642
    \\ freq_i = 2 * i / rot_dim  (i in [0, rot_dim/2))
    \\ theta_i = 2^(-23.2534966642 * freq_i)
    \\ angle_i = pos * theta_i
    \\ Rotation: (x[a], x[b]) = (x[a]*cos - x[b]*sin, x[a]*sin + x[b]*cos)

    u32>f32 fpos pos

    \\ ---- Section A: text, dims [0, 22), 11 pairs, pair (i, i+11) ----
    each i
        if>= i 11 skip_text

        \\ global frequency index for text section: ti = i  (i in [0,11))
        u32>f32 fi i
        freq_ratio = fi * 2.0 / 64.0             \\ 2*ti / rot_dim
        freq = freq_ratio * -23.2534966642
        theta 2^ freq
        angle = fpos * theta
        cs ≡ angle
        sn ≅ angle

        a = i
        b = i + 11
        x0 = x [ a ]
        x1 = x [ b ]
        x [ a ] = x0 * cs - x1 * sn
        x [ b ] = x0 * sn + x1 * cs

        label skip_text

    \\ ---- Section B: height, dims [22, 44), 11 pairs, pair (22+j, 22+j+11) ----
    each j
        if>= j 11 skip_height

        \\ global frequency index for height section: ti = 11 + j
        ti_h = 11 + j
        u32>f32 fi_h ti_h
        freq_ratio_h = fi_h * 2.0 / 64.0
        freq_h = freq_ratio_h * -23.2534966642
        theta_h 2^ freq_h
        angle_h = fpos * theta_h
        cs_h ≡ angle_h
        sn_h ≅ angle_h

        a_h = 22 + j
        b_h = a_h + 11
        xh0 = x [ a_h ]
        xh1 = x [ b_h ]
        x [ a_h ] = xh0 * cs_h - xh1 * sn_h
        x [ b_h ] = xh0 * sn_h + xh1 * cs_h

        label skip_height

    \\ ---- Section C: width, dims [44, 64), 10 pairs, pair (44+k, 44+k+10) ----
    each k
        if>= k 10 skip_width

        \\ global frequency index for width section: ti = 22 + k
        ti_w = 22 + k
        u32>f32 fi_w ti_w
        freq_ratio_w = fi_w * 2.0 / 64.0
        freq_w = freq_ratio_w * -23.2534966642
        theta_w 2^ freq_w
        angle_w = fpos * theta_w
        cs_w ≡ angle_w
        sn_w ≅ angle_w

        a_w = 44 + k
        b_w = a_w + 10
        xw0 = x [ a_w ]
        xw1 = x [ b_w ]
        x [ a_w ] = xw0 * cs_w - xw1 * sn_w
        x [ b_w ] = xw0 * sn_w + xw1 * cs_w

        label skip_width

    \\ Dims [64, 256) pass through untouched (partial rotary).


\\ ---- 3. attention_score_full ----
\\ Compute softmax( Q · K^T / sqrt(head_dim) ) with causal mask and GQA.
\\
\\ Q       : [24, 256]
\\ K       : [4, 256, seq_len]           (KV heads shared 6:1)
\\ mask    : causal — position t valid iff t <= current_pos
\\ scores  : [24, seq_len]
\\
\\ Launch: gridDim.x = 24 (one block per query head), blockDim.x = seq_len (one thread per key).
attention_score_full Q K mask scores :
    param head_dim u32          \\ = 256
    param n_q_heads u32         \\ = 24
    param n_kv_heads u32        \\ = 4
    param seq_len u32
    param current_pos u32
    shared smem_row 4096 f32    \\ per-block scratch for softmax over keys

    each q
        if>= q n_q_heads exit

        \\ GQA: kv_head = q / 6  (= q * n_kv_heads / n_q_heads)
        qk = q * n_kv_heads
        kv_head = qk / n_q_heads

        \\ Scale = 1 / sqrt(head_dim) = 1/16 for head_dim=256
        u32>f32 fd head_dim
        scale 1/√ fd

        \\ ---- Inner: for each key position t, compute score(q, t) ----
        stride t seq_len
            \\ Causal mask: score = -inf if t > current_pos
            if> t current_pos mask_out

            \\ Dot product Q[q] · K[kv_head][:, t] over d in [0, head_dim)
            dot = 0.0
            for d 0 head_dim 1
                qv = Q [ q * head_dim + d ]
                \\ K layout: [n_kv_heads, head_dim, seq_len]
                k_idx = kv_head * head_dim * seq_len + d * seq_len + t
                kv = K [ k_idx ]
                dot = dot + qv * kv
            endfor
            smem_row [ t ] = dot * scale
            bra mask_done
            label mask_out
            smem_row [ t ] = -340282346638528859811704183484516925440.0
            label mask_done

        barrier

        \\ ---- Softmax over smem_row[0..seq_len] ----
        m △ smem_row
        stride t2 seq_len
            d2 = smem_row [ t2 ] - m
            ev e^ d2
            smem_row [ t2 ] = ev
        barrier

        s Σ smem_row
        inv_s 1/ s

        stride t3 seq_len
            out_idx = q * seq_len + t3
            scores [ out_idx ] = smem_row [ t3 ] * inv_s


\\ ---- 4. attention_output_full ----
\\ Weighted sum of V, then per-head output gate (sigmoid).
\\
\\ scores  : [24, seq_len]
\\ V       : [4, 256, seq_len]
\\ gate    : [24, 256]            (per-head output gate, logits)
\\ out     : [24, 256]
\\
\\ Launch: gridDim.x = 24, blockDim.x = 256 (one thread per d).
attention_output_full scores V gate out :
    param head_dim u32          \\ = 256
    param n_q_heads u32         \\ = 24
    param n_kv_heads u32        \\ = 4
    param seq_len u32

    each q
        if>= q n_q_heads exit

        \\ GQA mapping
        qk = q * n_kv_heads
        kv_head = qk / n_q_heads

        \\ Each thread computes one output dim d for this query head.
        stride d head_dim

            \\ acc = sum_t scores[q, t] * V[kv_head, d, t]
            acc = 0.0
            for t 0 seq_len 1
                w = scores [ q * seq_len + t ]
                v_idx = kv_head * head_dim * seq_len + d * seq_len + t
                vv = V [ v_idx ]
                acc = acc + w * vv
            endfor

            \\ Output gate: sigmoid(gate[q, d]) = 1 / (1 + e^(-gate))
            g = gate [ q * head_dim + d ]
            neg_g = g * -1.0
            eg e^ neg_g
            denom = 1.0 + eg
            sig 1/ denom

            out [ q * head_dim + d ] = acc * sig
