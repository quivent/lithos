\\ ##  SECTION 5 — SAFETENSORS READER (lithos-safetensors)                  ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================
\\ CONSTANTS
\\ ============================================================

128 constant ST_MAX_TENSORS
256 constant ST_MAX_NAME_LEN

0 constant DTYPE_UNKNOWN
1 constant DTYPE_F16
2 constant DTYPE_F32
3 constant DTYPE_BF16
4 constant DTYPE_I32
5 constant DTYPE_I16
6 constant DTYPE_I8
7 constant DTYPE_U8

\\ ============================================================
\\ SYSCALL WRAPPERS
\\ ============================================================

sys_openat dirfd path flags mode :
    trap fd 56 dirfd path flags mode

sys_close fd :
    trap ret 57 fd

sys_lseek fd offset whence :
    trap pos 62 fd offset whence

sys_mmap addr length prot flags fd offset :
    trap ptr 222 addr length prot flags fd offset

\\ ============================================================
\\ st_open — Open and mmap a safetensors file
\\ ============================================================

st_open path :
    fd sys_openat -100 path 0 0
    file_size sys_lseek fd 0 2
    base sys_mmap 0 file_size 1 2 fd 0
    sys_close fd

\\ ============================================================
\\ BYTE-LEVEL HELPERS
\\ ============================================================

read_u64_le ptr :
    val → 64 ptr

read_u32_le ptr :
    val → 32 ptr

\\ ============================================================
\\ JSON SCANNER
\\ ============================================================

st_skip_ws ptr end :
    pos ptr
    loop_skip:
        if>= pos end
            out pos
            return
        ch → 8 pos
        if== ch 32
            pos pos + 1
            goto loop_skip
        if== ch 9
            pos pos + 1
            goto loop_skip
        if== ch 10
            pos pos + 1
            goto loop_skip
        if== ch 13
            pos pos + 1
            goto loop_skip
        out pos

st_find_char ptr end ch :
    pos ptr
    loop_fc:
        if>= pos end
            found end
            return
        b → 8 pos
        if== b ch
            found pos
            return
        pos pos + 1
        goto loop_fc

st_parse_quoted ptr end :
    start ptr + 1
    close st_find_char start end 34
    name_start start
    name_len close - start

st_parse_u64_dec ptr end :
    val 0
    pos ptr
    loop_num:
        if>= pos end
            next pos
            return
        ch → 8 pos
        if< ch 48
            next pos
            return
        if> ch 57
            next pos
            return
        digit ch - 48
        val val * 10 + digit
        pos pos + 1
        goto loop_num

st_match_bytes a b n :
    eq 1
    for i 0 n 1
        ca → 8 (a + i)
        cb → 8 (b + i)
        if!= ca cb
            eq 0
            return
    endfor

\\ ============================================================
\\ DTYPE PARSER
\\ ============================================================

st_parse_dtype ptr len :
    dtype DTYPE_UNKNOWN
    if== len 3
        c0 → 8 ptr
        c1 → 8 (ptr + 1)
        c2 → 8 (ptr + 2)
        if== c0 70
            if== c1 49
                if== c2 54
                    dtype DTYPE_F16
                    return
        if== c0 70
            if== c1 51
                if== c2 50
                    dtype DTYPE_F32
                    return
        if== c0 73
            if== c1 51
                if== c2 50
                    dtype DTYPE_I32
                    return
        if== c0 73
            if== c1 49
                if== c2 54
                    dtype DTYPE_I16
                    return
        if== c0 70
            if== c1 54
                if== c2 52
                    dtype DTYPE_UNKNOWN
                    return
    if== len 4
        c0 → 8 ptr
        c1 → 8 (ptr + 1)
        if== c0 66
            if== c1 70
                dtype DTYPE_BF16
                return
    if== len 2
        c0 → 8 ptr
        c1 → 8 (ptr + 1)
        if== c0 73
            if== c1 56
                dtype DTYPE_I8
                return
        if== c0 85
            if== c1 56
                dtype DTYPE_U8
                return

\\ ============================================================
\\ DESCRIPTOR PARSER
\\ ============================================================

st_parse_descriptor ptr end :
    dtype DTYPE_UNKNOWN
    shape0 0
    shape1 0
    off_start 0
    off_end 0
    pos ptr

    loop_desc:
        pos st_skip_ws pos end
        if>= pos end
            next pos
            return
        ch → 8 pos
        if== ch 125
            next pos + 1
            return

        if== ch 44
            pos pos + 1
            goto loop_desc

        if!= ch 34
            pos pos + 1
            goto loop_desc

        key_start key_len st_parse_quoted pos end
        pos key_start + key_len + 1

        pos st_find_char pos end 58
        pos pos + 1
        pos st_skip_ws pos end

        if== key_len 5
            k0 → 8 key_start
            if== k0 100
                pos st_skip_ws pos end
                val_start val_len st_parse_quoted pos end
                dtype st_parse_dtype val_start val_len
                pos val_start + val_len + 1
                goto loop_desc

        if== key_len 5
            k0 → 8 key_start
            if== k0 115
                pos st_find_char pos end 91
                pos pos + 1
                pos st_skip_ws pos end
                peek → 8 pos
                if== peek 93
                    pos pos + 1
                    goto loop_desc
                shape0 next_pos st_parse_u64_dec pos end
                pos next_pos
                pos st_skip_ws pos end
                peek2 → 8 pos
                if== peek2 44
                    pos pos + 1
                    pos st_skip_ws pos end
                    shape1 next_pos2 st_parse_u64_dec pos end
                    pos next_pos2
                pos st_find_char pos end 93
                pos pos + 1
                goto loop_desc

        if== key_len 12
            k0 → 8 key_start
            if== k0 100
                pos st_find_char pos end 91
                pos pos + 1
                pos st_skip_ws pos end
                off_start next_pos st_parse_u64_dec pos end
                pos next_pos
                pos st_find_char pos end 44
                pos pos + 1
                pos st_skip_ws pos end
                off_end next_pos2 st_parse_u64_dec pos end
                pos next_pos2
                pos st_find_char pos end 93
                pos pos + 1
                goto loop_desc

        depth 0
        loop_skip_val:
            if>= pos end
                goto loop_desc
            sv_ch → 8 pos
            if== sv_ch 123
                depth depth + 1
            if== sv_ch 91
                depth depth + 1
            if== sv_ch 125
                if== depth 0
                    goto loop_desc
                depth depth - 1
            if== sv_ch 93
                if> depth 0
                    depth depth - 1
            if== sv_ch 44
                if== depth 0
                    goto loop_desc
            pos pos + 1
            goto loop_skip_val

\\ ============================================================
\\ st_parse_header
\\ ============================================================

st_parse_header base file_size :
    header_len read_u64_le base
    json_start base + 8
    json_end base + 8 + header_len
    data_base json_end

    count 0

    pos st_skip_ws json_start json_end
    ch0 → 8 pos
    if== ch0 123
        pos pos + 1

    loop_tensors:
        pos st_skip_ws pos json_end
        if>= pos json_end
            header_end data_base
            tensor_count count
            return

        tch → 8 pos
        if== tch 125
            header_end data_base
            tensor_count count
            return

        if== tch 44
            pos pos + 1
            goto loop_tensors

        if!= tch 34
            pos pos + 1
            goto loop_tensors

        tname_start tname_len st_parse_quoted pos json_end
        pos tname_start + tname_len + 1

        if== tname_len 12
            m0 → 8 tname_start
            m1 → 8 (tname_start + 1)
            if== m0 95
                if== m1 95
                    pos st_find_char pos json_end 123
                    pos pos + 1
                    depth_m 1
                    loop_skip_meta:
                        if>= pos json_end
                            goto loop_tensors
                        mc → 8 pos
                        if== mc 123
                            depth_m depth_m + 1
                        if== mc 125
                            depth_m depth_m - 1
                            if== depth_m 0
                                pos pos + 1
                                goto loop_tensors
                        pos pos + 1
                        goto loop_skip_meta

        pos st_find_char pos json_end 58
        pos pos + 1
        pos st_skip_ws pos json_end

        desc_ch → 8 pos
        if== desc_ch 123
            pos pos + 1

        dtype shape0 shape1 off_start off_end pos st_parse_descriptor pos json_end

        if< count ST_MAX_TENSORS
            ← 64 tbl_name_ptr + count * 8 tname_start
            ← 32 tbl_name_len + count * 4 tname_len
            ← 32 tbl_dtype + count * 4 dtype
            ← 64 tbl_data_start + count * 8 off_start
            ← 64 tbl_data_end + count * 8 off_end
            ← 32 tbl_shape_0 + count * 4 shape0
            ← 32 tbl_shape_1 + count * 4 shape1
            data_ptr data_base + off_start
            ← 64 tbl_data_ptr + count * 8 data_ptr
            count count + 1

        goto loop_tensors

\\ ============================================================
\\ st_find_tensor
\\ ============================================================

st_find_tensor name name_len tensor_count :
    data_ptr 0
    dtype DTYPE_UNKNOWN
    shape0 0
    shape1 0
    nbytes 0

    for idx 0 tensor_count 1
        entry_name → 64 tbl_name_ptr + idx * 8
        entry_len → 32 tbl_name_len + idx * 4

        if== entry_len name_len
            eq st_match_bytes name entry_name name_len
            if== eq 1
                data_ptr → 64 tbl_data_ptr + idx * 8
                dtype → 32 tbl_dtype + idx * 4
                shape0 → 32 tbl_shape_0 + idx * 4
                shape1 → 32 tbl_shape_1 + idx * 4
                off_s → 64 tbl_data_start + idx * 8
                off_e → 64 tbl_data_end + idx * 8
                nbytes off_e - off_s
                return
    endfor

\\ ============================================================
\\ st_close
\\ ============================================================

st_close base file_size :
    trap ret 215 base file_size

\\ ============================================================
\\ st_load — One-call open + parse
\\ ============================================================

st_load path :
    base file_size st_open path
    header_end tensor_count st_parse_header base file_size

\\ ############################################################################
\\ ##                                                                        ##
