\\ lithos-safetensors.li — Safetensors file reader for Lithos self-hosting compiler
\\
\\ Reads HuggingFace safetensors binary format:
\\   Bytes 0-7:                      u64 header_length (little-endian)
\\   Bytes 8..(8+header_length):     JSON header (tensor metadata)
\\   Bytes (8+header_length)..EOF:   raw tensor data (contiguous)
\\
\\ The JSON header maps tensor names to {dtype, shape, data_offsets}.
\\ data_offsets are relative to the start of the data region.
\\
\\ This is HOST code (ARM64 target), not GPU kernel code.
\\ Uses Linux syscalls: open, fstat, mmap, close.

\\ ============================================================
\\ ARM64 LINUX SYSCALL NUMBERS
\\ ============================================================
\\ NR_openat   = 56
\\ NR_close    = 57
\\ NR_fstat    = 80   (or newfstatat = 79 on aarch64)
\\ NR_mmap     = 222
\\ NR_lseek    = 62
\\ AT_FDCWD    = -100
\\ O_RDONLY    = 0
\\ PROT_READ   = 1
\\ MAP_PRIVATE = 2
\\ SEEK_END    = 2

\\ ============================================================
\\ CONSTANTS
\\ ============================================================
\\ Maximum number of tensor entries we can index
\\ 512 tensors covers a 48-layer model with ~10 tensors/layer

128 constant MAX_TENSORS
256 constant MAX_NAME_LEN

\\ Dtype encoding (matches safetensors spec)
\\ 0 = unknown, 1 = F16, 2 = F32, 3 = BF16, 4 = I32, 5 = I16, 6 = I8, 7 = U8
0 constant DTYPE_UNKNOWN
1 constant DTYPE_F16
2 constant DTYPE_F32
3 constant DTYPE_BF16
4 constant DTYPE_I32
5 constant DTYPE_I16
6 constant DTYPE_I8
7 constant DTYPE_U8

\\ ============================================================
\\ TENSOR INDEX — flat table of parsed tensor metadata
\\ ============================================================
\\ Each entry:
\\   name_ptr    (u64) — pointer into mmap'd JSON header
\\   name_len    (u32) — byte length of tensor name
\\   dtype       (u32) — DTYPE_* constant
\\   data_start  (u64) — byte offset from data region start
\\   data_end    (u64) — byte offset from data region start
\\   shape_0     (u32) — first dimension
\\   shape_1     (u32) — second dimension (0 if 1-D)
\\   data_ptr    (u64) — resolved pointer into mmap'd data
\\
\\ Total: 48 bytes per entry
\\ Storage: global arrays, one per field (SOA layout for simplicity)

\\ [NOTE: .li host extension needed] Static arrays for tensor index.
\\ These would be global buffers in the ARM64 .bss section.
\\ For now, expressed as named storage with known sizes.

\\ global tbl_name_ptr   MAX_TENSORS u64
\\ global tbl_name_len   MAX_TENSORS u32
\\ global tbl_dtype      MAX_TENSORS u32
\\ global tbl_data_start MAX_TENSORS u64
\\ global tbl_data_end   MAX_TENSORS u64
\\ global tbl_shape_0    MAX_TENSORS u32
\\ global tbl_shape_1    MAX_TENSORS u32
\\ global tbl_data_ptr   MAX_TENSORS u64
\\ global tbl_count      1 u32            — number of entries parsed
\\ global st_base        1 u64            — mmap base address
\\ global st_file_size   1 u64            — total file size
\\ global st_data_base   1 u64            — start of raw data region

\\ ============================================================
\\ SYSCALL WRAPPERS
\\ ============================================================
\\ [NOTE: .li host extension needed] Inline syscall via SVC #0 on ARM64.
\\ Convention: x8 = syscall number, x0-x5 = args, x0 = return.
\\ These are pseudo-functions showing the ARM64 syscall ABI.

sys_openat dirfd path flags mode :
    \\ syscall(56, dirfd, path, flags, mode)
    \\ [HOST INTRINSIC] svc 0 with x8=56
    \\ Returns file descriptor in x0, negative on error.
    \\ dirfd = -100 (AT_FDCWD) for path relative to cwd.
    syscall fd 56 dirfd path flags mode

sys_close fd :
    \\ syscall(57, fd)
    syscall ret 57 fd

sys_lseek fd offset whence :
    \\ syscall(62, fd, offset, whence)
    \\ whence: 0=SET, 1=CUR, 2=END
    syscall pos 62 fd offset whence

sys_mmap addr length prot flags fd offset :
    \\ syscall(222, addr, length, prot, flags, fd, offset)
    syscall ptr 222 addr length prot flags fd offset

\\ ============================================================
\\ st_open — Open and mmap a safetensors file
\\ ============================================================
\\ Opens the file read-only, determines size via lseek, mmaps it,
\\ then closes the fd. Returns base pointer and file size.
\\
\\ path: null-terminated file path string
\\ Returns:
\\   base      — mmap'd base address (pointer to byte 0 of file)
\\   file_size — total file size in bytes

st_open path :
    \\ Open read-only: openat(AT_FDCWD, path, O_RDONLY, 0)
    fd = sys_openat -100 path 0 0

    \\ [NOTE: error check needed] if fd < 0, return 0 0

    \\ Get file size via lseek to end
    file_size = sys_lseek fd 0 2

    \\ mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    base = sys_mmap 0 file_size 1 2 fd 0

    \\ Close fd — the mmap keeps the mapping alive
    sys_close fd

    \\ Store globals for later use by st_parse_header / st_find_tensor
    \\ [HOST] ← base st_base
    \\ [HOST] ← file_size st_file_size

\\ ============================================================
\\ BYTE-LEVEL HELPERS
\\ ============================================================
\\ [NOTE: .li host extension needed] Byte load/store operations.
\\ → 8  ptr byte     (LDRB on ARM64)
\\ → 64 ptr val      (LDR on ARM64)
\\ These are intrinsics the ARM64 backend must provide.

\\ Read a little-endian u64 from a byte pointer
read_u64_le ptr :
    \\ [HOST INTRINSIC] val = *(u64*)ptr
    \\ ARM64 is little-endian natively, so a plain LDR X works.
    load_u64 val ptr

\\ Read a little-endian u32 from a byte pointer
read_u32_le ptr :
    \\ [HOST INTRINSIC] val = *(u32*)ptr
    load_u32 val ptr

\\ ============================================================
\\ JSON SCANNER — minimal byte-scanning parser
\\ ============================================================
\\ The safetensors JSON header is flat: top-level object whose keys
\\ are tensor names, values are {dtype, shape, data_offsets} objects.
\\ No nested objects beyond the tensor descriptor. No arrays of objects.
\\
\\ Strategy:
\\   1. Skip leading '{'
\\   2. Loop: find next '"' -> tensor name -> find ':' -> find '{'
\\   3. Inside descriptor: find "data_offsets", "dtype", "shape"
\\   4. Repeat until closing '}'

\\ skip_whitespace: advance ptr past spaces, tabs, newlines
skip_ws ptr end :
    pos = ptr
    loop_skip:
        if>= pos end
            out = pos
            return
        ch = load_u8 pos
        \\ space=32, tab=9, newline=10, cr=13
        if== ch 32
            pos = pos + 1
            goto loop_skip
        if== ch 9
            pos = pos + 1
            goto loop_skip
        if== ch 10
            pos = pos + 1
            goto loop_skip
        if== ch 13
            pos = pos + 1
            goto loop_skip
        out = pos

\\ find_char: scan forward for a specific byte, return pointer to it
find_char ptr end ch :
    pos = ptr
    loop_fc:
        if>= pos end
            found = end
            return
        b = load_u8 pos
        if== b ch
            found = pos
            return
        pos = pos + 1
        goto loop_fc

\\ parse_quoted_string: ptr points at opening '"', returns name_start, name_len
\\ name_start points to first char after the quote; name_len excludes quotes.
parse_quoted ptr end :
    \\ Skip opening '"'
    start = ptr + 1
    \\ Find closing '"'  (no escape handling — tensor names have no backslashes)
    close = find_char start end 34    \\ 34 = '"'
    name_start = start
    name_len = close - start

\\ parse_u64_decimal: parse ASCII decimal integer starting at ptr
parse_u64_dec ptr end :
    val = 0
    pos = ptr
    loop_num:
        if>= pos end
            next = pos
            return
        ch = load_u8 pos
        \\ '0'=48, '9'=57
        if< ch 48
            next = pos
            return
        if> ch 57
            next = pos
            return
        digit = ch - 48
        val = val * 10 + digit
        pos = pos + 1
        goto loop_num

\\ match_bytes: compare n bytes at a and b, return 1 if equal, 0 otherwise
match_bytes a b n :
    eq = 1
    for i 0 n 1
        ca = load_u8 (a + i)
        cb = load_u8 (b + i)
        if!= ca cb
            eq = 0
            return
    endfor

\\ ============================================================
\\ DTYPE PARSER
\\ ============================================================
\\ Parse dtype string like "F16", "F32", "BF16", "I32", etc.
\\ ptr points to first char after opening quote of dtype value.

parse_dtype ptr len :
    dtype = DTYPE_UNKNOWN
    \\ Check common dtypes by length and first chars
    if== len 3
        c0 = load_u8 ptr
        c1 = load_u8 (ptr + 1)
        c2 = load_u8 (ptr + 2)
        \\ "F16" = 70,49,54
        if== c0 70
            if== c1 49
                if== c2 54
                    dtype = DTYPE_F16
                    return
        \\ "F32" = 70,51,50
        if== c0 70
            if== c1 51
                if== c2 50
                    dtype = DTYPE_F32
                    return
        \\ "I32" = 73,51,50
        if== c0 73
            if== c1 51
                if== c2 50
                    dtype = DTYPE_I32
                    return
        \\ "I16" = 73,49,54
        if== c0 73
            if== c1 49
                if== c2 54
                    dtype = DTYPE_I16
                    return
        \\ "F64" = 70,54,52
        if== c0 70
            if== c1 54
                if== c2 52
                    dtype = DTYPE_UNKNOWN
                    return
    if== len 4
        c0 = load_u8 ptr
        c1 = load_u8 (ptr + 1)
        \\ "BF16" = 66,70,49,54
        if== c0 66
            if== c1 70
                dtype = DTYPE_BF16
                return
    if== len 2
        c0 = load_u8 ptr
        c1 = load_u8 (ptr + 1)
        \\ "I8" = 73,56
        if== c0 73
            if== c1 56
                dtype = DTYPE_I8
                return
        \\ "U8" = 85,56
        if== c0 85
            if== c1 56
                dtype = DTYPE_U8
                return

\\ ============================================================
\\ DESCRIPTOR PARSER
\\ ============================================================
\\ Parse one tensor descriptor: { "dtype": "...", "shape": [...], "data_offsets": [start, end] }
\\ ptr points just past the opening '{' of the descriptor.
\\ Returns parsed dtype, shape dims, and data offsets.

parse_descriptor ptr end :
    dtype = DTYPE_UNKNOWN
    shape0 = 0
    shape1 = 0
    off_start = 0
    off_end = 0
    pos = ptr

    \\ Scan for keys within this descriptor until we hit '}'
    loop_desc:
        pos = skip_ws pos end
        if>= pos end
            next = pos
            return
        ch = load_u8 pos
        if== ch 125    \\ '}' = 125 — end of descriptor
            next = pos + 1
            return

        if== ch 44    \\ ',' = 44 — separator between fields
            pos = pos + 1
            goto loop_desc

        \\ Expect a key string
        if!= ch 34    \\ not '"' — skip unexpected byte
            pos = pos + 1
            goto loop_desc

        \\ Parse key name
        key_start key_len = parse_quoted pos end
        pos = key_start + key_len + 1   \\ skip past closing '"'

        \\ Skip to ':'
        pos = find_char pos end 58    \\ ':' = 58
        pos = pos + 1
        pos = skip_ws pos end

        \\ ---- Match key and parse value ----

        \\ "dtype" (5 chars): 100,116,121,112,101
        if== key_len 5
            k0 = load_u8 key_start
            if== k0 100    \\ 'd'
                \\ Value is a quoted string: "F16", "I32", etc.
                pos = skip_ws pos end
                \\ skip opening '"'
                val_start val_len = parse_quoted pos end
                dtype = parse_dtype val_start val_len
                pos = val_start + val_len + 1   \\ past closing '"'
                goto loop_desc

        \\ "shape" (5 chars): 115,104,97,112,101
        if== key_len 5
            k0 = load_u8 key_start
            if== k0 115    \\ 's'
                \\ Value is array: [dim0] or [dim0, dim1]
                pos = find_char pos end 91    \\ '[' = 91
                pos = pos + 1
                pos = skip_ws pos end
                \\ Check for empty array
                peek = load_u8 pos
                if== peek 93    \\ ']' = 93
                    pos = pos + 1
                    goto loop_desc
                \\ Parse first dimension
                shape0 next_pos = parse_u64_dec pos end
                pos = next_pos
                pos = skip_ws pos end
                peek2 = load_u8 pos
                if== peek2 44    \\ ','
                    pos = pos + 1
                    pos = skip_ws pos end
                    shape1 next_pos2 = parse_u64_dec pos end
                    pos = next_pos2
                \\ Skip to ']'
                pos = find_char pos end 93    \\ ']'
                pos = pos + 1
                goto loop_desc

        \\ "data_offsets" (12 chars): 100,97,116,97,95,111,102,102,115,101,116,115
        if== key_len 12
            k0 = load_u8 key_start
            if== k0 100    \\ 'd'
                \\ Value is [start, end]
                pos = find_char pos end 91    \\ '['
                pos = pos + 1
                pos = skip_ws pos end
                off_start next_pos = parse_u64_dec pos end
                pos = next_pos
                pos = find_char pos end 44    \\ ','
                pos = pos + 1
                pos = skip_ws pos end
                off_end next_pos2 = parse_u64_dec pos end
                pos = next_pos2
                pos = find_char pos end 93    \\ ']'
                pos = pos + 1
                goto loop_desc

        \\ Unknown key — skip value (find next ',' or '}')
        \\ Simple: scan forward for ',' or '}' at this nesting level
        depth = 0
        loop_skip_val:
            if>= pos end
                goto loop_desc
            sv_ch = load_u8 pos
            if== sv_ch 123    \\ '{'
                depth = depth + 1
            if== sv_ch 91     \\ '['
                depth = depth + 1
            if== sv_ch 125    \\ '}'
                if== depth 0
                    goto loop_desc
                depth = depth - 1
            if== sv_ch 93     \\ ']'
                if> depth 0
                    depth = depth - 1
            if== sv_ch 44     \\ ','
                if== depth 0
                    goto loop_desc
            pos = pos + 1
            goto loop_skip_val

\\ ============================================================
\\ st_parse_header — Parse JSON header, build tensor index
\\ ============================================================
\\ base: mmap'd file base pointer (from st_open)
\\ Reads header_length from first 8 bytes, then iterates the JSON
\\ to populate the tensor index tables.
\\ Returns header_end (pointer to start of raw data region).

st_parse_header base file_size :
    \\ Read 8-byte header length
    header_len = read_u64_le base
    json_start = base + 8
    json_end = base + 8 + header_len
    data_base = json_end    \\ raw tensor data starts here

    \\ Store data_base globally for st_find_tensor
    \\ [HOST] ← data_base st_data_base

    \\ Initialize tensor count
    count = 0

    \\ Skip opening '{' of top-level JSON object
    pos = skip_ws json_start json_end
    ch0 = load_u8 pos
    if== ch0 123    \\ '{'
        pos = pos + 1

    \\ Main parse loop: iterate tensor entries
    loop_tensors:
        pos = skip_ws pos json_end
        if>= pos json_end
            header_end = data_base
            tensor_count = count
            return

        tch = load_u8 pos
        \\ End of top-level object
        if== tch 125    \\ '}'
            header_end = data_base
            tensor_count = count
            return

        \\ Skip comma between entries
        if== tch 44    \\ ','
            pos = pos + 1
            goto loop_tensors

        \\ Expect '"' for tensor name
        if!= tch 34    \\ not '"'
            pos = pos + 1
            goto loop_tensors

        \\ Parse tensor name
        tname_start tname_len = parse_quoted pos json_end
        pos = tname_start + tname_len + 1    \\ past closing '"'

        \\ Skip "__metadata__" key (safetensors may include it)
        if== tname_len 12
            m0 = load_u8 tname_start
            m1 = load_u8 (tname_start + 1)
            if== m0 95     \\ '_'
                if== m1 95 \\ '_'
                    \\ Skip to matching '}'
                    pos = find_char pos json_end 123    \\ find '{'
                    pos = pos + 1
                    depth_m = 1
                    loop_skip_meta:
                        if>= pos json_end
                            goto loop_tensors
                        mc = load_u8 pos
                        if== mc 123
                            depth_m = depth_m + 1
                        if== mc 125
                            depth_m = depth_m - 1
                            if== depth_m 0
                                pos = pos + 1
                                goto loop_tensors
                        pos = pos + 1
                        goto loop_skip_meta

        \\ Skip ':' between key and value
        pos = find_char pos json_end 58    \\ ':'
        pos = pos + 1
        pos = skip_ws pos json_end

        \\ Parse descriptor object { dtype, shape, data_offsets }
        desc_ch = load_u8 pos
        if== desc_ch 123    \\ '{'
            pos = pos + 1

        dtype shape0 shape1 off_start off_end pos = parse_descriptor pos json_end

        \\ Store into index tables
        if< count MAX_TENSORS
            \\ [HOST] tbl_name_ptr[count] = tname_start
            \\ [HOST] tbl_name_len[count] = tname_len
            \\ [HOST] tbl_dtype[count]    = dtype
            \\ [HOST] tbl_data_start[count] = off_start
            \\ [HOST] tbl_data_end[count]   = off_end
            \\ [HOST] tbl_shape_0[count]  = shape0
            \\ [HOST] tbl_shape_1[count]  = shape1
            \\ [HOST] tbl_data_ptr[count] = data_base + off_start
            store_u64 tbl_name_ptr count tname_start
            store_u32 tbl_name_len count tname_len
            store_u32 tbl_dtype count dtype
            store_u64 tbl_data_start count off_start
            store_u64 tbl_data_end count off_end
            store_u32 tbl_shape_0 count shape0
            store_u32 tbl_shape_1 count shape1
            data_ptr = data_base + off_start
            store_u64 tbl_data_ptr count data_ptr
            count = count + 1

        goto loop_tensors

\\ ============================================================
\\ st_find_tensor — Look up a tensor by name
\\ ============================================================
\\ name:     pointer to tensor name string (not null-terminated)
\\ name_len: byte length of name
\\ Returns:
\\   data_ptr  — pointer into mmap'd raw data (ready to use)
\\   dtype     — DTYPE_* constant
\\   shape_ptr — pointer to (shape0, shape1) pair in index table
\\   nbytes    — data_end - data_start (total bytes of tensor data)
\\
\\ Returns data_ptr=0 if tensor not found.

st_find_tensor name name_len tensor_count :
    data_ptr = 0
    dtype = DTYPE_UNKNOWN
    shape0 = 0
    shape1 = 0
    nbytes = 0

    for idx 0 tensor_count 1
        \\ Load entry name pointer and length
        entry_name = load_u64_idx tbl_name_ptr idx
        entry_len  = load_u32_idx tbl_name_len idx

        \\ Quick length check
        if== entry_len name_len
            \\ Compare bytes
            eq = match_bytes name entry_name name_len
            if== eq 1
                data_ptr = load_u64_idx tbl_data_ptr idx
                dtype    = load_u32_idx tbl_dtype idx
                shape0   = load_u32_idx tbl_shape_0 idx
                shape1   = load_u32_idx tbl_shape_1 idx
                off_s    = load_u64_idx tbl_data_start idx
                off_e    = load_u64_idx tbl_data_end idx
                nbytes   = off_e - off_s
                return
    endfor

\\ ============================================================
\\ st_close — Unmap the safetensors file (optional cleanup)
\\ ============================================================
\\ base:      mmap'd base address
\\ file_size: total mapped size
\\ Calls munmap(base, file_size). NR_munmap = 215 on aarch64.

st_close base file_size :
    syscall ret 215 base file_size

\\ ============================================================
\\ CONVENIENCE: st_load — One-call open + parse
\\ ============================================================
\\ Opens the safetensors file, mmaps it, parses the header,
\\ returns the base pointer and tensor count for subsequent
\\ st_find_tensor calls.

st_load path :
    base file_size = st_open path
    header_end tensor_count = st_parse_header base file_size
