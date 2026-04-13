\\ config-reader.ls — Parse model config.json for the hybrid layer schedule
\\
\\ Reads HuggingFace config.json (Qwen 3.5 style) to extract:
\\   - Model dimension parameters (hidden_size, vocab_size, etc.)
\\   - Attention head counts and dimensions (full + DeltaNet)
\\   - RoPE parameters (theta, partial_rotary_factor)
\\   - The layer_types[] array that drives hybrid compilation
\\
\\ The JSON is nested: most fields live under "text_config": { ... }
\\ and RoPE fields live under "text_config": { "rope_parameters": { ... } }
\\
\\ This is HOST code (ARM64 target). Uses Linux syscalls: open, lseek, mmap, close.
\\ Reuses the byte-scanning approach from lithos-safetensors.ls.

\\ ============================================================
\\ CONSTANTS
\\ ============================================================

64 constant MAX_LAYERS
\\ Layer type codes: 0 = DeltaNet (linear_attention), 1 = full (softmax) attention
0 constant LAYER_LINEAR
1 constant LAYER_FULL

\\ ============================================================
\\ GLOBAL STATE — parsed config fields
\\ ============================================================
\\ The compiler reads these after config_load returns.
\\
\\ Integers (u32):
\\   ← 32 cfg_num_hidden_layers      val
\\   ← 32 cfg_hidden_size             val
\\   ← 32 cfg_intermediate_size       val
\\   ← 32 cfg_vocab_size              val
\\   ← 32 cfg_head_dim                val
\\   ← 32 cfg_num_attention_heads     val
\\   ← 32 cfg_num_key_value_heads     val
\\   ← 32 cfg_linear_num_key_heads    val
\\   ← 32 cfg_linear_num_value_heads  val
\\   ← 32 cfg_linear_key_head_dim     val
\\   ← 32 cfg_linear_value_head_dim   val
\\   ← 32 cfg_linear_conv_kernel_dim  val
\\   ← 32 cfg_full_attention_interval val
\\   ← 32 cfg_attn_output_gate        val    (0=false, 1=true)
\\
\\ Floats:
\\   ← 32 cfg_rms_norm_eps            val    (f32)
\\   ← 64 cfg_rope_theta              val    (f64)
\\   ← 32 cfg_partial_rotary_factor   val    (f32)
\\
\\ Layer schedule:
\\   cfg_layer_types : u8[MAX_LAYERS]   (0=linear, 1=full)

\\ ============================================================
\\ SYSCALL WRAPPERS (same ABI as lithos-safetensors.ls)
\\ ============================================================

cfg_sys_openat dirfd path flags mode :
    syscall fd 56 dirfd path flags mode

cfg_sys_close fd :
    syscall ret 57 fd

cfg_sys_lseek fd offset whence :
    syscall pos 62 fd offset whence

cfg_sys_mmap addr length prot flags fd offset :
    syscall ptr 222 addr length prot flags fd offset

\\ ============================================================
\\ BYTE-LEVEL HELPERS (reused from safetensors scanner)
\\ ============================================================

\\ skip_whitespace: advance ptr past spaces, tabs, newlines, carriage returns
cfg_skip_ws ptr end :
    pos = ptr
    loop_ws:
        if>= pos end
            out = pos
            return
        ch = load_u8 pos
        if== ch 32
            pos = pos + 1
            goto loop_ws
        if== ch 9
            pos = pos + 1
            goto loop_ws
        if== ch 10
            pos = pos + 1
            goto loop_ws
        if== ch 13
            pos = pos + 1
            goto loop_ws
        out = pos

\\ find_char: scan forward for byte ch, return pointer to it (or end)
cfg_find_char ptr end ch :
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

\\ match_bytes: compare n bytes at a and b, return 1 if equal else 0
cfg_match_bytes a b n :
    eq = 1
    for i 0 n 1
        ca = load_u8 (a + i)
        cb = load_u8 (b + i)
        if!= ca cb
            eq = 0
            return
    endfor

\\ parse_u64_dec: parse ASCII decimal integer at ptr, return val and next pos
cfg_parse_u64_dec ptr end :
    val = 0
    pos = ptr
    loop_num:
        if>= pos end
            next = pos
            return
        ch = load_u8 pos
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

\\ parse_f64_dec: parse ASCII decimal float (e.g. "0.25", "1e-06", "10000000.0")
\\ Returns an f64 value and next position.
\\ Handles: integer part, optional '.' + fractional part, optional 'e'/'E' + exponent
cfg_parse_f64_dec ptr end :
    val = 0.0
    pos = ptr
    sign = 1.0

    \\ Optional leading minus
    ch0 = load_u8 pos
    if== ch0 45
        sign = -1.0
        pos = pos + 1

    \\ Integer part
    loop_int:
        if>= pos end
            goto done_number
        ch = load_u8 pos
        if< ch 48
            goto check_dot
        if> ch 57
            goto check_dot
        digit = ch - 48
        val = val * 10.0 + digit
        pos = pos + 1
        goto loop_int

    check_dot:
        ch = load_u8 pos
        if!= ch 46
            goto check_exp
        pos = pos + 1
        frac = 0.1

    loop_frac:
        if>= pos end
            goto done_number
        ch = load_u8 pos
        if< ch 48
            goto check_exp
        if> ch 57
            goto check_exp
        digit = ch - 48
        val = val + digit * frac
        frac = frac * 0.1
        pos = pos + 1
        goto loop_frac

    check_exp:
        if>= pos end
            goto done_number
        ch = load_u8 pos
        \\ 'e' = 101, 'E' = 69
        if!= ch 101
            if!= ch 69
                goto done_number
        pos = pos + 1

        \\ Exponent sign
        exp_sign = 1
        if>= pos end
            goto done_number
        ech = load_u8 pos
        if== ech 45
            exp_sign = -1
            pos = pos + 1
        if== ech 43
            pos = pos + 1

        \\ Exponent digits
        exp_val = 0
    loop_exp:
        if>= pos end
            goto apply_exp
        ch = load_u8 pos
        if< ch 48
            goto apply_exp
        if> ch 57
            goto apply_exp
        digit = ch - 48
        exp_val = exp_val * 10 + digit
        pos = pos + 1
        goto loop_exp

    apply_exp:
        \\ Compute 10^exp_val and multiply/divide
        pow10 = 1.0
        for j 0 exp_val 1
            pow10 = pow10 * 10.0
        endfor
        if== exp_sign -1
            val = val / pow10
        if== exp_sign 1
            val = val * pow10

    done_number:
        val = val * sign
        next = pos

\\ parse_quoted: ptr at opening '"', return start (after quote) and length
cfg_parse_quoted ptr end :
    start = ptr + 1
    close = cfg_find_char start end 34    \\ 34 = '"'
    name_start = start
    name_len = close - start

\\ ============================================================
\\ JSON VALUE SKIPPER
\\ ============================================================
\\ Skips one JSON value starting at pos (string, number, object, array, bool, null).
\\ Returns pointer past the skipped value.

cfg_skip_value pos end :
    pos = cfg_skip_ws pos end
    if>= pos end
        out = pos
        return
    ch = load_u8 pos

    \\ String: skip to closing quote
    if== ch 34
        inner = pos + 1
        close = cfg_find_char inner end 34
        out = close + 1
        return

    \\ Object: skip balanced { }
    if== ch 123
        depth = 1
        pos = pos + 1
    loop_skip_obj:
        if>= pos end
            out = pos
            return
        sc = load_u8 pos
        if== sc 123
            depth = depth + 1
        if== sc 125
            depth = depth - 1
            if== depth 0
                out = pos + 1
                return
        \\ Skip strings (to avoid counting braces inside strings)
        if== sc 34
            pos = pos + 1
            pos = cfg_find_char pos end 34
        pos = pos + 1
        goto loop_skip_obj

    \\ Array: skip balanced [ ]
    if== ch 91
        depth = 1
        pos = pos + 1
    loop_skip_arr:
        if>= pos end
            out = pos
            return
        sc = load_u8 pos
        if== sc 91
            depth = depth + 1
        if== sc 93
            depth = depth - 1
            if== depth 0
                out = pos + 1
                return
        if== sc 34
            pos = pos + 1
            pos = cfg_find_char pos end 34
        pos = pos + 1
        goto loop_skip_arr

    \\ true / false / null / number: scan until delimiter
    loop_skip_prim:
        if>= pos end
            out = pos
            return
        sc = load_u8 pos
        \\ Stop at , } ] or whitespace
        if== sc 44
            out = pos
            return
        if== sc 125
            out = pos
            return
        if== sc 93
            out = pos
            return
        if== sc 32
            out = pos
            return
        if== sc 10
            out = pos
            return
        if== sc 13
            out = pos
            return
        if== sc 9
            out = pos
            return
        pos = pos + 1
        goto loop_skip_prim

\\ ============================================================
\\ KEY MATCHER CONSTANTS
\\ ============================================================
\\ Key names encoded as byte literals for matching.
\\ We match by (length, first-few-bytes) to avoid storing string constants.
\\
\\ Key lengths:
\\   "num_hidden_layers"        = 17
\\   "hidden_size"              = 11
\\   "intermediate_size"        = 17
\\   "vocab_size"               = 10
\\   "head_dim"                 = 8
\\   "num_attention_heads"      = 19
\\   "num_key_value_heads"      = 19
\\   "linear_num_key_heads"     = 20
\\   "linear_num_value_heads"   = 22
\\   "linear_key_head_dim"      = 19
\\   "linear_value_head_dim"    = 21
\\   "linear_conv_kernel_dim"   = 22
\\   "full_attention_interval"  = 23
\\   "rms_norm_eps"             = 12
\\   "rope_theta"               = 10
\\   "partial_rotary_factor"    = 21
\\   "attn_output_gate"         = 16
\\   "layer_types"              = 11
\\   "text_config"              = 11
\\   "rope_parameters"          = 15

\\ ============================================================
\\ LAYER_TYPES ARRAY PARSER
\\ ============================================================
\\ Parses: "layer_types": ["linear_attention", "full_attention", ...]
\\ pos points just after '['. Fills cfg_layer_types[0..count].
\\ "linear_attention" (16 chars, starts with 'l') → 0
\\ "full_attention"   (14 chars, starts with 'f') → 1

cfg_parse_layer_types pos end :
    idx = 0
    loop_lt:
        pos = cfg_skip_ws pos end
        if>= pos end
            count = idx
            next = pos
            return
        ch = load_u8 pos

        \\ End of array
        if== ch 93    \\ ']'
            count = idx
            next = pos + 1
            return

        \\ Comma separator
        if== ch 44    \\ ','
            pos = pos + 1
            goto loop_lt

        \\ Expect quoted string
        if!= ch 34    \\ not '"'
            pos = pos + 1
            goto loop_lt

        \\ Parse the string value
        str_start str_len = cfg_parse_quoted pos end
        pos = str_start + str_len + 1    \\ past closing '"'

        \\ Classify: "linear_attention" = 16 chars, "full_attention" = 14 chars
        \\ Robust check: first char 'l' (108) → linear, 'f' (102) → full
        if< idx MAX_LAYERS
            first = load_u8 str_start
            if== first 108    \\ 'l' for linear_attention
                store_u8 cfg_layer_types idx 0
            if== first 102    \\ 'f' for full_attention
                store_u8 cfg_layer_types idx 1
            idx = idx + 1

        goto loop_lt

\\ ============================================================
\\ PARSE BOOL — match "true" or "false"
\\ ============================================================
\\ pos points at first char of value (after ':'). Returns 0 or 1 and next pos.

cfg_parse_bool pos end :
    pos = cfg_skip_ws pos end
    ch = load_u8 pos
    \\ 't' = 116 → true (4 chars)
    if== ch 116
        val = 1
        next = pos + 4
        return
    \\ 'f' = 102 → false (5 chars)
    if== ch 102
        val = 0
        next = pos + 5
        return
    \\ Fallback
    val = 0
    next = pos

\\ ============================================================
\\ OBJECT KEY DISPATCHER
\\ ============================================================
\\ Given a key (start, len) and value position, parse the value
\\ and store into the appropriate cfg_* global.
\\ Returns the position after the value.
\\
\\ This is the core dispatch: match key name → extract value → store.

cfg_dispatch_key key_start key_len val_pos end :
    pos = val_pos

    \\ ---- "num_hidden_layers" (17 chars) ----
    \\ First 3 bytes: n(110) u(117) m(109)
    if== key_len 17
        k0 = load_u8 key_start
        k4 = load_u8 (key_start + 4)
        \\ "num_hidden_layers": k0=110('n'), k4=104('h')
        if== k0 110
            if== k4 104
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_num_hidden_layers v
                out = next
                return

    \\ ---- "intermediate_size" (17 chars) ----
    \\ k0=105('i'), k4=114('r')
    if== key_len 17
        k0 = load_u8 key_start
        k4 = load_u8 (key_start + 4)
        if== k0 105
            if== k4 114
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_intermediate_size v
                out = next
                return

    \\ ---- "hidden_size" (11 chars) ----
    \\ k0=104('h'), k1=105('i')
    if== key_len 11
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 104
            if== k1 105
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_hidden_size v
                out = next
                return

    \\ ---- "layer_types" (11 chars) ----
    \\ k0=108('l'), k1=97('a')
    if== key_len 11
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 108
            if== k1 97
                \\ Find opening '['
                pos = cfg_find_char pos end 91
                pos = pos + 1
                count next = cfg_parse_layer_types pos end
                out = next
                return

    \\ ---- "vocab_size" (10 chars) ----
    \\ k0=118('v'), k1=111('o')
    if== key_len 10
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 118
            if== k1 111
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_vocab_size v
                out = next
                return

    \\ ---- "rope_theta" (10 chars) ----
    \\ k0=114('r'), k1=111('o')
    if== key_len 10
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 114
            if== k1 111
                v next = cfg_parse_f64_dec pos end
                ← 64 cfg_rope_theta v
                out = next
                return

    \\ ---- "head_dim" (8 chars) ----
    \\ k0=104('h'), k1=101('e')
    if== key_len 8
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 104
            if== k1 101
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_head_dim v
                out = next
                return

    \\ ---- "num_attention_heads" (19 chars) ----
    \\ k0=110('n'), k4=97('a')  — "num_attention_heads"
    if== key_len 19
        k0 = load_u8 key_start
        k4 = load_u8 (key_start + 4)
        \\ Disambiguate from "num_key_value_heads" (also 19): k4='a' vs k4='k'
        if== k0 110
            if== k4 97
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_num_attention_heads v
                out = next
                return

    \\ ---- "num_key_value_heads" (19 chars) ----
    \\ k0=110('n'), k4=107('k')
    if== key_len 19
        k0 = load_u8 key_start
        k4 = load_u8 (key_start + 4)
        if== k0 110
            if== k4 107
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_num_key_value_heads v
                out = next
                return

    \\ ---- "linear_key_head_dim" (19 chars) ----
    \\ k0=108('l'), k4=97('a')  — starts with "line", k7=107('k')
    if== key_len 19
        k0 = load_u8 key_start
        k7 = load_u8 (key_start + 7)
        if== k0 108
            if== k7 107
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_linear_key_head_dim v
                out = next
                return

    \\ ---- "linear_num_key_heads" (20 chars) ----
    \\ k0=108('l'), k11=107('k')
    if== key_len 20
        k0 = load_u8 key_start
        k11 = load_u8 (key_start + 11)
        if== k0 108
            if== k11 107
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_linear_num_key_heads v
                out = next
                return

    \\ ---- "linear_value_head_dim" (21 chars) ----
    \\ k0=108('l'), k7=118('v')
    if== key_len 21
        k0 = load_u8 key_start
        k7 = load_u8 (key_start + 7)
        if== k0 108
            if== k7 118
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_linear_value_head_dim v
                out = next
                return

    \\ ---- "linear_num_value_heads" (22 chars) ----
    \\ k0=108('l'), k7=110('n') for "linear_num..."
    if== key_len 22
        k0 = load_u8 key_start
        k7 = load_u8 (key_start + 7)
        if== k0 108
            if== k7 110
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_linear_num_value_heads v
                out = next
                return

    \\ ---- "linear_conv_kernel_dim" (22 chars) ----
    \\ k0=108('l'), k7=99('c') for "linear_conv..."
    if== key_len 22
        k0 = load_u8 key_start
        k7 = load_u8 (key_start + 7)
        if== k0 108
            if== k7 99
                v next = cfg_parse_u64_dec pos end
                ← 32 cfg_linear_conv_kernel_dim v
                out = next
                return

    \\ ---- "partial_rotary_factor" (21 chars) ----
    \\ k0=112('p') — unique first byte at this length
    if== key_len 21
        k0 = load_u8 key_start
        if== k0 112
            v next = cfg_parse_f64_dec pos end
            ← 32 cfg_partial_rotary_factor v
            out = next
            return

    \\ ---- "full_attention_interval" (23 chars) ----
    \\ k0=102('f'), k5=116('t')
    if== key_len 23
        k0 = load_u8 key_start
        if== k0 102
            v next = cfg_parse_u64_dec pos end
            ← 32 cfg_full_attention_interval v
            out = next
            return

    \\ ---- "rms_norm_eps" (12 chars) ----
    \\ k0=114('r'), k1=109('m')
    if== key_len 12
        k0 = load_u8 key_start
        k1 = load_u8 (key_start + 1)
        if== k0 114
            if== k1 109
                v next = cfg_parse_f64_dec pos end
                ← 32 cfg_rms_norm_eps v
                out = next
                return

    \\ ---- "attn_output_gate" (16 chars) ----
    \\ k0=97('a'), k5=111('o')
    if== key_len 16
        k0 = load_u8 key_start
        k5 = load_u8 (key_start + 5)
        if== k0 97
            if== k5 111
                v next = cfg_parse_bool pos end
                ← 32 cfg_attn_output_gate v
                out = next
                return

    \\ ---- Unrecognized key: skip value ----
    out = cfg_skip_value pos end

\\ ============================================================
\\ OBJECT PARSER — iterate keys in a JSON object
\\ ============================================================
\\ Parses { "key": value, "key": value, ... }
\\ For each key-value pair, calls cfg_dispatch_key.
\\ Recursively enters "text_config" and "rope_parameters" objects
\\ so nested fields are dispatched through the same key matcher.
\\
\\ pos points just past the opening '{'.

cfg_parse_object pos end :
    loop_obj:
        pos = cfg_skip_ws pos end
        if>= pos end
            next = pos
            return
        ch = load_u8 pos

        \\ End of object
        if== ch 125    \\ '}'
            next = pos + 1
            return

        \\ Comma separator
        if== ch 44    \\ ','
            pos = pos + 1
            goto loop_obj

        \\ Expect '"' for key
        if!= ch 34
            pos = pos + 1
            goto loop_obj

        \\ Parse key
        key_start key_len = cfg_parse_quoted pos end
        pos = key_start + key_len + 1    \\ past closing '"'

        \\ Skip to ':'
        pos = cfg_find_char pos end 58    \\ ':'
        pos = pos + 1
        pos = cfg_skip_ws pos end

        \\ ---- Check for nested objects we need to descend into ----

        \\ "text_config" (11 chars): k0=116('t'), k1=101('e')
        if== key_len 11
            k0 = load_u8 key_start
            k1 = load_u8 (key_start + 1)
            if== k0 116
                if== k1 101
                    \\ Descend into text_config object
                    brace = load_u8 pos
                    if== brace 123    \\ '{'
                        pos = pos + 1
                        pos = cfg_parse_object pos end
                        goto loop_obj

        \\ "rope_parameters" (15 chars): k0=114('r'), k5=112('p')
        if== key_len 15
            k0 = load_u8 key_start
            k5 = load_u8 (key_start + 5)
            if== k0 114
                if== k5 112
                    \\ Descend into rope_parameters object
                    brace = load_u8 pos
                    if== brace 123
                        pos = pos + 1
                        pos = cfg_parse_object pos end
                        goto loop_obj

        \\ ---- Dispatch to key matcher for known fields ----
        pos = cfg_dispatch_key key_start key_len pos end

        goto loop_obj

\\ ============================================================
\\ config_load — Open, mmap, and parse config.json
\\ ============================================================
\\ path: null-terminated path to config.json
\\ Populates all cfg_* globals and cfg_layer_types[].

config_load path :
    \\ Open read-only: openat(AT_FDCWD, path, O_RDONLY, 0)
    fd = cfg_sys_openat -100 path 0 0

    \\ Get file size via lseek to end
    file_size = cfg_sys_lseek fd 0 2

    \\ mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    base = cfg_sys_mmap 0 file_size 1 2 fd 0

    \\ Close fd — mmap keeps the mapping alive
    cfg_sys_close fd

    \\ Initialize defaults for fields that might be absent
    ← 32 cfg_num_hidden_layers 0
    ← 32 cfg_hidden_size 0
    ← 32 cfg_intermediate_size 0
    ← 32 cfg_vocab_size 0
    ← 32 cfg_head_dim 0
    ← 32 cfg_num_attention_heads 0
    ← 32 cfg_num_key_value_heads 0
    ← 32 cfg_linear_num_key_heads 0
    ← 32 cfg_linear_num_value_heads 0
    ← 32 cfg_linear_key_head_dim 0
    ← 32 cfg_linear_value_head_dim 0
    ← 32 cfg_linear_conv_kernel_dim 0
    ← 32 cfg_full_attention_interval 0
    ← 32 cfg_attn_output_gate 0
    ← 32 cfg_rms_norm_eps 0
    ← 64 cfg_rope_theta 0
    ← 32 cfg_partial_rotary_factor 0

    \\ Zero the layer_types array
    for i 0 MAX_LAYERS 1
        store_u8 cfg_layer_types i 0
    endfor

    \\ Parse: find top-level '{' and enter the object
    json_end = base + file_size
    pos = cfg_skip_ws base json_end
    ch = load_u8 pos
    if== ch 123    \\ '{'
        pos = pos + 1
        pos = cfg_parse_object pos json_end

\\ ============================================================
\\ ACCESSORS — layer type queries
\\ ============================================================

\\ config_get_layer_type: returns 0 (DeltaNet/linear) or 1 (full attention)
config_get_layer_type layer_idx :
    type = load_u8_idx cfg_layer_types layer_idx

\\ config_is_full_attention: returns 1 if layer uses full softmax attention
config_is_full_attention layer_idx :
    type = load_u8_idx cfg_layer_types layer_idx
    if== type 1
        result = 1
        return
    result = 0

\\ config_is_deltanet: returns 1 if layer uses DeltaNet linear attention
config_is_deltanet layer_idx :
    type = load_u8_idx cfg_layer_types layer_idx
    if== type 0
        result = 1
        return
    result = 0

\\ ============================================================
\\ TEST PLAN (verify against known Qwen 3.5 27B config values)
\\ ============================================================
\\
\\ After calling:
\\   config_load "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16/config.json"
\\
\\ Verify each field:
\\   → 32 cfg_num_hidden_layers        == 64
\\   → 32 cfg_hidden_size              == 5120
\\   → 32 cfg_intermediate_size        == 17408
\\   → 32 cfg_vocab_size               == 248320
\\   → 32 cfg_head_dim                 == 256
\\   → 32 cfg_num_attention_heads      == 24
\\   → 32 cfg_num_key_value_heads      == 4
\\   → 32 cfg_linear_num_key_heads     == 16
\\   → 32 cfg_linear_num_value_heads   == 48
\\   → 32 cfg_linear_key_head_dim      == 128
\\   → 32 cfg_linear_value_head_dim    == 128
\\   → 32 cfg_linear_conv_kernel_dim   == 4
\\   → 32 cfg_full_attention_interval  == 4
\\   → 32 cfg_attn_output_gate         == 1  (true)
\\   → 32 cfg_rms_norm_eps             == 1e-6  (f32)
\\   → 64 cfg_rope_theta               == 10000000.0  (f64)
\\   → 32 cfg_partial_rotary_factor    == 0.25  (f32)
\\
\\ Verify layer_types array (64 entries, pattern 3+1 repeating):
\\   for i 0 64 1
\\       type = config_get_layer_type i
\\       mod = i % 4
\\       if== mod 3
\\           assert type == 1   \\ full attention at indices 3,7,11,...,63
\\       else
\\           assert type == 0   \\ linear attention at all other indices
\\   endfor
\\
\\ Verify convenience queries:
\\   config_is_full_attention 3   == 1
\\   config_is_full_attention 7   == 1
\\   config_is_full_attention 0   == 0
\\   config_is_full_attention 1   == 0
\\   config_is_deltanet 0         == 1
\\   config_is_deltanet 3         == 0
\\
\\ Edge cases:
\\   - rope_theta = 10000000 (integer in JSON, no decimal point) → parsed as 10000000.0
\\   - rms_norm_eps = 1e-06 (scientific notation) → parsed as 0.000001
\\   - partial_rotary_factor = 0.25 (decimal) → parsed as 0.25
\\   - attn_output_gate = true (bare keyword, not quoted) → parsed as 1
