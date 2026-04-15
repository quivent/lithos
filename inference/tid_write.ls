\\ tid_write.ls — minimal test kernel
\\
\\ Each thread writes float(threadIdx.x) to output[tid].
\\ Uses only proven opcodes: S2R, I2F, LDC, IMAD_IMM, STG, EXIT.

kernel out :
    param out ptr

    each tid
        x tid
        y x i2f
        stride 4
        offset x * stride
        addr out + offset
        addr[0] = y
