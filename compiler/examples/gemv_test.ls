\\ gemv_test.ls -- GPTQ GEMV (W4A16) kernel in Lithos
\\
\\ GPTQ GEMV: y = W_q * x
\\ Weight matrix W is 4-bit quantized (8 values packed per u32).
\\ Each group of 128 elements shares one f32 scale and one f32 zero.

gptq_gemv qweight zeros scales xvec output result :
    param n_rows u32
    param n_cols u32
    param group_size u32
    shared smem 4096 f32

    each tid

    \\ Compute row and lane
    shr row tid 5
    and lane tid 31

    \\ Bounds check
    if>= row n_rows exit

    acc = 0.0

    for col_blk 0 n_cols 32
        add col col_blk lane
        shr group col 7
        shr col8 col 3
        shr nc8 n_cols 3
        mad w_idx row nc8 col8
        shr ngrp n_cols 7
        mad sz_idx row ngrp group
        and nib_pos col 7
        shl nib_shift nib_pos 2
        u32>f32 fcol col
    endfor

    \\ Warp shuffle reduction
    shfl.bfly stmp acc 16
    shfl.bfly stmp acc 8
    shfl.bfly stmp acc 4
    shfl.bfly stmp acc 2
    shfl.bfly stmp acc 1

    \\ Lane 0 writes result
    setp eq u32 planez lane 0
    @planez bra write_out
    bra skip_write

    label write_out
    result [ tid ] = acc

    label skip_write
