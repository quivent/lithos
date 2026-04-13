\\ reduce.ls — Reduction operations for LLM inference
\\ Replaces: norm.ptx, sample.ptx
\\
\\ 1. RMSNorm: y[i] = x[i] * weight[i] * rsqrt(mean(x^2) + eps)
\\ 2. Fused RMSNorm + residual add: y[i] = (x[i]+r[i]) * w[i] * rsqrt(mean((x+r)^2) + eps)
\\ 3. L2Norm: y[i] = x[i] * rsqrt(sum(x^2))
\\ 4. Sampling: argmax over logits
\\
\\ Launch: blockDim.x = 256, each block handles one row.
\\ Two-pass design:
\\   Pass 1: stride loop accumulates partial sum-of-squares per thread
\\   Reduce: warp shuffle down + cross-warp via shared memory
\\   Pass 2: stride loop applies normalize * weight
\\ This reads global memory twice but keeps the reduction correct.
\\ The hand-written PTX (norm.ptx, forward_pass.ptx) follows this exact pattern.

\\ ---- 1. rmsnorm ----
\\ gridDim.x = batch_rows, blockDim.x = 256
\\ Each block normalizes one row of hidden_dim elements.
\\ Two-pass: (1) reduce sum-of-squares, (2) apply norm+scale.
rmsnorm x weight y :
    param hidden_dim u32
    param eps f32
    shared smem_reduce 32 f32

    \\ Pass 1: accumulate sum of squares (strided over hidden_dim)
    partial_ss = 0.0
    stride i hidden_dim
        val = x [ i ]
        partial_ss = partial_ss + val * val

    \\ Intra-warp reduction (shuffle down)
    shfl.down t0 partial_ss 16
    partial_ss = partial_ss + t0
    shfl.down t1 partial_ss 8
    partial_ss = partial_ss + t1
    shfl.down t2 partial_ss 4
    partial_ss = partial_ss + t2
    shfl.down t3 partial_ss 2
    partial_ss = partial_ss + t3
    shfl.down t4 partial_ss 1
    partial_ss = partial_ss + t4

    \\ Cross-warp reduction via shared memory
    \\ Lane 0 of each warp stores to smem_reduce[warp_id]
    lane_id = tid & 31
    warp_id = tid >> 5
    if== lane_id 0
        smem_reduce [ warp_id ] = partial_ss
    barrier

    \\ First warp reads all partial sums and reduces
    if< tid 32
        num_warps = blockDim >> 5
        if< tid num_warps
            total = smem_reduce [ tid ]
        else
            total = 0.0
        shfl.down s0 total 16
        total = total + s0
        shfl.down s1 total 8
        total = total + s1
        shfl.down s2 total 4
        total = total + s2
        shfl.down s3 total 2
        total = total + s3
        shfl.down s4 total 1
        total = total + s4
        \\ Thread 0 broadcasts rms_inv
        if== tid 0
            mean_ss = total / hidden_dim
            mean_ss = mean_ss + eps
            rsqrt rms_inv mean_ss
            smem_reduce [ 0 ] = rms_inv
    barrier

    \\ All threads load the broadcast rms_inv
    rms_inv = smem_reduce [ 0 ]

    \\ Pass 2: normalize each element
    stride j hidden_dim
        y [ j ] = x [ j ] * weight [ j ] * rms_inv

\\ ---- 2. rmsnorm_residual ----
\\ Fused residual-add + RMSNorm. One read of x and residual, one write.
\\ y[i] = (x[i] + residual[i]) * weight[i] * rsqrt(mean((x+residual)^2) + eps)
\\ Also writes the sum x+residual back (needed for next layer's skip connection).
\\ gridDim.x = batch_rows, blockDim.x = 256
rmsnorm_residual x residual weight y :
    param hidden_dim u32
    param eps f32
    shared smem_reduce 32 f32

    \\ Pass 1: sum of squares of (x + residual)
    partial_ss = 0.0
    stride i hidden_dim
        val = x [ i ] + residual [ i ]
        partial_ss = partial_ss + val * val

    \\ Intra-warp reduction
    shfl.down t0 partial_ss 16
    partial_ss = partial_ss + t0
    shfl.down t1 partial_ss 8
    partial_ss = partial_ss + t1
    shfl.down t2 partial_ss 4
    partial_ss = partial_ss + t2
    shfl.down t3 partial_ss 2
    partial_ss = partial_ss + t3
    shfl.down t4 partial_ss 1
    partial_ss = partial_ss + t4

    \\ Cross-warp
    lane_id = tid & 31
    warp_id = tid >> 5
    if== lane_id 0
        smem_reduce [ warp_id ] = partial_ss
    barrier

    if< tid 32
        num_warps = blockDim >> 5
        if< tid num_warps
            total = smem_reduce [ tid ]
        else
            total = 0.0
        shfl.down s0 total 16
        total = total + s0
        shfl.down s1 total 8
        total = total + s1
        shfl.down s2 total 4
        total = total + s2
        shfl.down s3 total 2
        total = total + s3
        shfl.down s4 total 1
        total = total + s4
        if== tid 0
            mean_ss = total / hidden_dim
            mean_ss = mean_ss + eps
            rsqrt rms_inv mean_ss
            smem_reduce [ 0 ] = rms_inv
    barrier

    rms_inv = smem_reduce [ 0 ]

    \\ Pass 2: normalize
    stride j hidden_dim
        val = x [ j ] + residual [ j ]
        y [ j ] = val * weight [ j ] * rms_inv

\\ ---- 3. l2norm ----
\\ L2-normalization: y[i] = x[i] * rsqrt(sum(x^2))
\\ No learned weight, no mean division, no epsilon.
\\ gridDim.x = batch_rows, blockDim.x = 256
l2norm x y :
    param hidden_dim u32
    shared smem_reduce 32 f32

    \\ Pass 1: sum of squares
    partial_ss = 0.0
    stride i hidden_dim
        val = x [ i ]
        partial_ss = partial_ss + val * val

    \\ Intra-warp reduction
    shfl.down t0 partial_ss 16
    partial_ss = partial_ss + t0
    shfl.down t1 partial_ss 8
    partial_ss = partial_ss + t1
    shfl.down t2 partial_ss 4
    partial_ss = partial_ss + t2
    shfl.down t3 partial_ss 2
    partial_ss = partial_ss + t3
    shfl.down t4 partial_ss 1
    partial_ss = partial_ss + t4

    \\ Cross-warp
    lane_id = tid & 31
    warp_id = tid >> 5
    if== lane_id 0
        smem_reduce [ warp_id ] = partial_ss
    barrier

    if< tid 32
        num_warps = blockDim >> 5
        if< tid num_warps
            total = smem_reduce [ tid ]
        else
            total = 0.0
        shfl.down s0 total 16
        total = total + s0
        shfl.down s1 total 8
        total = total + s1
        shfl.down s2 total 4
        total = total + s2
        shfl.down s3 total 2
        total = total + s3
        shfl.down s4 total 1
        total = total + s4
        if== tid 0
            rsqrt rscale total
            smem_reduce [ 0 ] = rscale
    barrier

    rscale = smem_reduce [ 0 ]

    \\ Pass 2: scale each element
    stride j hidden_dim
        y [ j ] = x [ j ] * rscale

\\ ---- 4. sample_argmax ----
\\ Argmax over logit vector: returns index of maximum value (predicted token ID).
\\ gridDim.x = 1, blockDim.x = 256
\\ Single-pass: stride loop finds per-thread max, then parallel reduction
\\ carries both value and index through warp shuffles and shared memory.
sample_argmax logits vocab_size output_idx :
    shared smem_vals 32 f32
    shared smem_idxs 32 u32

    \\ Pass 1: stride loop — each thread finds its local max value and index
    \\ Initialize to -inf so any real logit wins
    my_val = -340282346638528859811704183484516925440.0
    my_idx = 0

    stride i vocab_size
        candidate = logits [ i ]
        if> candidate my_val
            my_val = candidate
            my_idx = i

    \\ Intra-warp argmax reduction (butterfly shuffle)
    \\ For each delta, exchange value and index with partner lane.
    \\ If partner's value is strictly greater, take their value and index.
    shfl.bfly nb_val16 my_val 16
    shfl.bfly nb_idx16 my_idx 16
    if> nb_val16 my_val
        my_val = nb_val16
        my_idx = nb_idx16

    shfl.bfly nb_val8 my_val 8
    shfl.bfly nb_idx8 my_idx 8
    if> nb_val8 my_val
        my_val = nb_val8
        my_idx = nb_idx8

    shfl.bfly nb_val4 my_val 4
    shfl.bfly nb_idx4 my_idx 4
    if> nb_val4 my_val
        my_val = nb_val4
        my_idx = nb_idx4

    shfl.bfly nb_val2 my_val 2
    shfl.bfly nb_idx2 my_idx 2
    if> nb_val2 my_val
        my_val = nb_val2
        my_idx = nb_idx2

    shfl.bfly nb_val1 my_val 1
    shfl.bfly nb_idx1 my_idx 1
    if> nb_val1 my_val
        my_val = nb_val1
        my_idx = nb_idx1

    \\ Cross-warp reduction via shared memory
    \\ Lane 0 of each warp stores its max value and index
    lane_id = tid & 31
    warp_id = tid >> 5
    if== lane_id 0
        smem_vals [ warp_id ] = my_val
        smem_idxs [ warp_id ] = my_idx
    barrier

    \\ First warp reads all warp-level maxima and reduces
    if< tid 32
        num_warps = blockDim >> 5
        if< tid num_warps
            my_val = smem_vals [ tid ]
            my_idx = smem_idxs [ tid ]
        else
            my_val = -340282346638528859811704183484516925440.0
            my_idx = 0

        shfl.bfly nb_val16 my_val 16
        shfl.bfly nb_idx16 my_idx 16
        if> nb_val16 my_val
            my_val = nb_val16
            my_idx = nb_idx16

        shfl.bfly nb_val8 my_val 8
        shfl.bfly nb_idx8 my_idx 8
        if> nb_val8 my_val
            my_val = nb_val8
            my_idx = nb_idx8

        shfl.bfly nb_val4 my_val 4
        shfl.bfly nb_idx4 my_idx 4
        if> nb_val4 my_val
            my_val = nb_val4
            my_idx = nb_idx4

        shfl.bfly nb_val2 my_val 2
        shfl.bfly nb_idx2 my_idx 2
        if> nb_val2 my_val
            my_val = nb_val2
            my_idx = nb_idx2

        shfl.bfly nb_val1 my_val 1
        shfl.bfly nb_idx1 my_idx 1
        if> nb_val1 my_val
            my_val = nb_val1
            my_idx = nb_idx1

        \\ Thread 0 writes the winning index to output
        if== tid 0
            output_idx [ 0 ] = my_idx
