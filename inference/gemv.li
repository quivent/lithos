# LEGACY: .li dialect. Replaced by kernels.ls (.ls dialect).
# Multi-function .li files produced broken cubins (only last function kept).

\ gemv.li — GPTQ W4A16 matrix-vector multiply
\ Replaces: gptq_matvec.ptx, projection.ptx, lm_head.ptx
\
\ y[row] = sum_k( dequant(W[row,k]) * x[k] )
\ W is 4-bit quantized (packed 8 per u32), scale per group, zero_point=8.
\ Group size = 128 elements = 16 packed u32s.
\
\ Launch: gridDim.x = N (one block per output row), blockDim.x = 256
\ Each thread strides over K_packed = K/8 packed words.
\ Warp shuffle + shared memory for final reduction.
\
\ K=3584: K_packed = 3584/8 = 448 packed u32s per row
\ Thread stride: threads=256, so each thread handles ceil(448/256) words
\ Row offset into W_packed: row * K_packed (row = blockIdx.x = %r0)
\
\ Register map after each row:
\   %r0 = blockIdx.x  (= output row index)
\   %r1 = blockDim.x  (= 256)
\   %r2 = tid.x       (intra-block thread id, 0..255)
\   %r3 = global_tid  (= row * 256 + tid.x — NOT used as row here)

fn gptq_gemv W_packed scales x -> y
    param K u32
    param N u32
    shared smem_reduce 256 f32
    each row
    \ row = blockIdx.x (one block per output row)
    \ Use %r0 for row-index, %r2 for tid.x
    \ Bounds check: if blockIdx.x >= N, exit
    if>= row N exit
    \ acc = 0.0
    acc = 0.0
    \ k_packed = K / 8  (number of packed u32s per row)
    \ K-reduction loop: thread tid.x handles k_p = tid.x, tid.x+256, tid.x+512, ...
    \ tid_local = tid.x = %r2 (= global_tid AND 255, since blockDim=256=2^8)
    and tid_local row 255
    \ row_base = blockIdx.x * (K/8)
    \ K_packed = K >> 3
    shr K_packed K 3
    \ row_base = row_idx * K_packed  (row_idx = blockIdx.x = %r0)
    \ We need blockIdx.x as a separate symbol; %r0 holds it after each
    \ Use: row_idx = (row - tid_local) / 256 = row >> 8
    shr row_idx row 8
    mul row_base row_idx K_packed
    \ Inner K loop: k_p strides from tid_local to K_packed by blockDim (256)
    for k_p tid_local K_packed 256
        \ Absolute packed index into W_packed for this row
        add wp_idx row_base k_p
        \ Load one packed u32 at W_packed[wp_idx]
        packed = W_packed [ wp_idx ]
        \ Load scale for this group: scales[wp_idx / 16]
        \ (group size 128 elements = 16 packed u32s, divide by 16 = shr 4)
        shr grp wp_idx 4
        sc = scales [ grp ]
        \ x base index for this packed word: k_p * 8
        shl x_base k_p 3
        \ Unroll 8 nibbles from packed word, each multiplies a distinct x element
        \ nibble 0: bits [3:0]  -> x[x_base + 0]
        and nib0 packed 15
        s32>f32 fnib0 nib0
        xval0 = x [ x_base ]
        fma acc fnib0 xval0 acc
        \ nibble 1: bits [7:4]   -> x[x_base + 1]
        shr tmp1 packed 4
        and nib1 tmp1 15
        s32>f32 fnib1 nib1
        add xi1 x_base 1
        xval1 = x [ xi1 ]
        fma acc fnib1 xval1 acc
        \ nibble 2: bits [11:8]  -> x[x_base + 2]
        shr tmp2 packed 8
        and nib2 tmp2 15
        s32>f32 fnib2 nib2
        add xi2 x_base 2
        xval2 = x [ xi2 ]
        fma acc fnib2 xval2 acc
        \ nibble 3: bits [15:12] -> x[x_base + 3]
        shr tmp3 packed 12
        and nib3 tmp3 15
        s32>f32 fnib3 nib3
        add xi3 x_base 3
        xval3 = x [ xi3 ]
        fma acc fnib3 xval3 acc
        \ nibble 4: bits [19:16] -> x[x_base + 4]
        shr tmp4 packed 16
        and nib4 tmp4 15
        s32>f32 fnib4 nib4
        add xi4 x_base 4
        xval4 = x [ xi4 ]
        fma acc fnib4 xval4 acc
        \ nibble 5: bits [23:20] -> x[x_base + 5]
        shr tmp5 packed 20
        and nib5 tmp5 15
        s32>f32 fnib5 nib5
        add xi5 x_base 5
        xval5 = x [ xi5 ]
        fma acc fnib5 xval5 acc
        \ nibble 6: bits [27:24] -> x[x_base + 6]
        shr tmp6 packed 24
        and nib6 tmp6 15
        s32>f32 fnib6 nib6
        add xi6 x_base 6
        xval6 = x [ xi6 ]
        fma acc fnib6 xval6 acc
        \ nibble 7: bits [31:28] -> x[x_base + 7]
        shr tmp7 packed 28
        and nib7 tmp7 15
        s32>f32 fnib7 nib7
        add xi7 x_base 7
        xval7 = x [ xi7 ]
        fma acc fnib7 xval7 acc
    endfor
    \ Warp reduction across 32 lanes (butterfly; all lanes of each warp sum)
    shfl.bfly t0 acc 16
    acc = acc + t0
    shfl.bfly t1 acc 8
    acc = acc + t1
    shfl.bfly t2 acc 4
    acc = acc + t2
    shfl.bfly t3 acc 2
    acc = acc + t3
    shfl.bfly t4 acc 1
    acc = acc + t4
    \ Cross-warp reduction: 8 warps (blockDim=256) reduce to one value.
    \ Step 1: compute warp_id and lane_id from tid.x
    \ tid_local = tid.x (already in tid_local; recompute for clarity)
    \ warp_id = tid_local >> 5
    shr warp_id tid_local 5
    \ lane_id = tid_local & 31
    and lane_id tid_local 31
    \ Step 2: lane 0 of each warp stores partial sum to smem[warp_id]
    \ Guard: skip STS if lane_id != 0 (i.e. lane_id >= 1)
    if>= lane_id 1 skip_sts
    smem_reduce [ warp_id ] = acc
    label skip_sts
    \ Step 3: block-wide barrier — all warp partial sums now in smem
    bar.sync
    \ Step 4: first warp (tid_local < 32) loads and butterfly-reduces
    \ Default: partial = 0.0 for threads tid >= 8 (inactive warp slots)
    warp_acc = 0.0
    \ Only threads 0..7 (one per warp slot) load; tid >= 8 contribute 0.0
    if>= tid_local 8 skip_lds
    warp_acc = smem_reduce [ tid_local ]
    label skip_lds
    \ Butterfly reduce within first warp (tid_local < 32 participate)
    \ Threads with tid_local >= 32 are in other warps and hold 0.0 anyway.
    shfl.bfly w0 warp_acc 16
    warp_acc = warp_acc + w0
    shfl.bfly w1 warp_acc 8
    warp_acc = warp_acc + w1
    shfl.bfly w2 warp_acc 4
    warp_acc = warp_acc + w2
    shfl.bfly w3 warp_acc 2
    warp_acc = warp_acc + w3
    shfl.bfly w4 warp_acc 1
    warp_acc = warp_acc + w4
    \ Step 5: thread 0 stores final sum to smem[0]
    if>= tid_local 1 skip_final_sts
    smem_reduce [ 0 ] = warp_acc
    label skip_final_sts
    \ Step 6: second barrier — broadcast result now in smem[0]
    bar.sync
    \ Step 7: all threads load broadcast result from smem[0]
    result = smem_reduce [ 0 ]
    \ Step 8: thread 0 (lane 0 of warp 0) stores to output
    if>= tid_local 1 skip_store
    y [ row_idx ] = result
    label skip_store
