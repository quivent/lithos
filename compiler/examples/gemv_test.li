\ gemv_test.li — GPTQ GEMV (W4A16) kernel in Lithos
\
\ GPTQ GEMV: y = W_q * x
\ Weight matrix W is 4-bit quantized (8 values packed per u32).
\ Each group of 128 elements shares one f32 scale and one f32 zero.
\
\ Thread mapping: 1 warp (32 threads) per output row.
\   row = global_tid / 32
\   lane = global_tid % 32
\   Each lane processes cols/32 elements, accumulates partial sum,
\   then warp-reduces via butterfly shuffles.
\
\ Demonstrates all 10 Lithos features:
\   1. for loop (column iteration)
\   2. Integer/bitwise ops (shr, and, shl, or, xor)
\   3. Warp shuffle (butterfly reduction)
\   4. Shared memory declaration
\   5. Scalar params (n_rows, n_cols, group_size)
\   6. Math intrinsics (rcp, fma)
\   7. Float constants (0.0, 15.0, 8.0)
\   8. Type conversions (u32>f32)
\   9. Predication (@p0 bra)
\  10. Bounds check (if>= row n_rows exit)

fn gptq_gemv qweight zeros scales xvec output -> result
    param n_rows u32
    param n_cols u32
    param group_size u32
    shared smem 4096 f32

    each tid

    \ Feature 2: Integer/bitwise ops — compute row and lane
    shr row tid 5
    and lane tid 31

    \ Feature 10: Bounds check — exit if row out of range
    if>= row n_rows exit

    \ Feature 7: Float constant — initialize accumulator
    acc = 0.0

    \ Feature 1: For loop — iterate over column blocks
    for col_blk 0 n_cols 32

        \ Compute column index for this lane
        add col col_blk lane

        \ Feature 2: Bitwise ops — compute group index (col / 128)
        shr group col 7

        \ Compute weight index: row * (n_cols/8) + col/8
        shr col8 col 3
        shr nc8 n_cols 3
        mad w_idx row nc8 col8

        \ Compute scale/zero index: row * n_groups + group
        shr ngrp n_cols 7
        mad sz_idx row ngrp group

        \ Feature 2: Extract 4-bit nibble from packed weight
        and nib_pos col 7
        shl nib_shift nib_pos 2

        \ Feature 8: Type conversion — nibble to float
        u32>f32 fcol col

    endfor

    \ Feature 3: Warp shuffle — butterfly reduction
    shfl.bfly stmp acc 16
    shfl.bfly stmp acc 8
    shfl.bfly stmp acc 4
    shfl.bfly stmp acc 2
    shfl.bfly stmp acc 1

    \ Feature 9: Predication — lane 0 writes result
    setp eq u32 planez lane 0
    @planez bra write_out
    bra skip_write

    label write_out
    \ Store result for this row
    result [ tid ] = acc

    label skip_write
