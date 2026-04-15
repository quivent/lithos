\\ ============================================================================
\\ parts/06-walker.ls — Lithos minimal compiler token walker
\\ ============================================================================
\\
\\ Two-phase dispatch over the token stream from 01-lex.ls.
\\
\\   Phase 1 (walk_collect): one linear pass to index every top-level
\\     "name args :" composition into a composition table.
\\
\\   Phase 2 (walk_top_level): for every non-host composition, emit code
\\     by walking its body tokens. Composition references inline the body
\\     of the referenced composition (spec §4.1 "flattening"). There is no
\\     call/return — each kernel is a flat instruction stream.
\\
\\ The walker uses a tiny virtual stack of register numbers. Literals and
\\ arg references push; operators pop. Registers are bump-allocated per
\\ kernel — no spilling because kernels are bounded.
\\
\\ Inputs:
\\   tokens buffer + token_count_buf (from 01-lex.ls, 12 bytes per token).
\\   lex_src_v — source base pointer (from 01-lex.ls).
\\
\\ Outputs:
\\   gpu_buf + gpu_pos_v for SASS (from 04-cubin-elf.ls / SASS emitters).
\\   arm64_buf + arm64_pos_v for ARM64 (from 05-arm64-elf.ls).
\\
\\ Token layout reminder: tokens[i] = { type:u32, offset:u32, length:u32 }
\\ addressed as 32-bit words at (tokens + i*3 + k).

\\ ============================================================================
\\ Walker-local globals
\\ ============================================================================

\\ Composition table (up to 64 compositions).
buf comp_name_off_v  256      \\ u32 source offset of name
buf comp_name_len_v  256      \\ u32 length of name
buf comp_arg_count_v 256      \\ u32 arg count
buf comp_body_start_v 256     \\ u32 token index of first body token
buf comp_body_end_v   256     \\ u32 token index one past last body token
buf comp_is_host_v    256     \\ u32 0=GPU, 1=ARM64
buf comp_count_v       8

\\ Symbol table for current kernel: arg name → register (up to 32 entries).
buf sym_name_off_v   128
buf sym_name_len_v   128
buf sym_reg_v        128
buf sym_count_v        8

\\ Virtual value stack for current expression (register numbers).
buf vstack_v         128
buf vstack_sp_v        8

\\ Bump register allocator for current kernel.
buf next_reg_v         8

\\ Scan cursor used by body walker.
buf walk_pos_v         8

\\ Emit target: 0 = GPU (SASS), 1 = ARM64 (host).
buf emit_target_v      8

\\ Bookkeeping: track whether we emitted any GPU / ARM64 code.
buf any_gpu_v          8
buf any_arm64_v        8

\\ ============================================================================
\\ Token accessors — one token = 3 consecutive u32 words.
\\ ============================================================================

tok_type i :
    t → 32 tokens + i * 3
    t

tok_offset i :
    o → 32 tokens + (i * 3) + 1
    o

tok_length i :
    l → 32 tokens + (i * 3) + 2
    l

\\ ============================================================================
\\ Symbol table helpers
\\ ============================================================================

sym_reset :
    ← 64 sym_count_v 0

\\ Add name→reg binding; name lives in source buffer at (offset, len).
sym_add name_off name_len reg :
    n → 64 sym_count_v
    ← 32 sym_name_off_v + n * 4 name_off
    ← 32 sym_name_len_v + n * 4 name_len
    ← 32 sym_reg_v + n * 4 reg
    ← 64 sym_count_v n + 1

\\ Find name in table; returns register index, or -1 if not found.
sym_find name_off name_len :
    src → 64 lex_src_v
    n → 64 sym_count_v
    i 0
    label sf_loop
    if>= i n
        goto sf_miss
    slen → 32 sym_name_len_v + i * 4
    if== slen name_len
        soff → 32 sym_name_off_v + i * 4
        match 1
        j 0
        label sf_cmp
        if>= j slen
            goto sf_cmp_done
        a → 8 src + soff + j
        b → 8 src + name_off + j
        if!= a b
            match 0
            goto sf_cmp_done
        j j + 1
        goto sf_cmp
        label sf_cmp_done
        if== match 1
            reg → 32 sym_reg_v + i * 4
            return reg
    i i + 1
    goto sf_loop
    label sf_miss
    -1

\\ ============================================================================
\\ Composition table helpers
\\ ============================================================================

comp_reset :
    ← 64 comp_count_v 0

comp_add name_off name_len arg_count body_start body_end is_host :
    n → 64 comp_count_v
    ← 32 comp_name_off_v + n * 4 name_off
    ← 32 comp_name_len_v + n * 4 name_len
    ← 32 comp_arg_count_v + n * 4 arg_count
    ← 32 comp_body_start_v + n * 4 body_start
    ← 32 comp_body_end_v + n * 4 body_end
    ← 32 comp_is_host_v + n * 4 is_host
    ← 64 comp_count_v n + 1

comp_find name_off name_len :
    src → 64 lex_src_v
    n → 64 comp_count_v
    i 0
    label cf_loop
    if>= i n
        goto cf_miss
    clen → 32 comp_name_len_v + i * 4
    if== clen name_len
        coff → 32 comp_name_off_v + i * 4
        match 1
        j 0
        label cf_cmp
        if>= j clen
            goto cf_cmp_done
        a → 8 src + coff + j
        b → 8 src + name_off + j
        if!= a b
            match 0
            goto cf_cmp_done
        j j + 1
        goto cf_cmp
        label cf_cmp_done
        if== match 1
            return i
    i i + 1
    goto cf_loop
    label cf_miss
    -1

\\ ============================================================================
\\ Register allocator (bump) and virtual stack
\\ ============================================================================

reg_reset :
    \\ Reserve R0..R3 for hardware-owned quantities; start at R4.
    ← 64 next_reg_v 4
    ← 64 vstack_sp_v 0

alloc_reg :
    r → 64 next_reg_v
    ← 64 next_reg_v r + 1
    r

vpush reg :
    sp → 64 vstack_sp_v
    ← 32 vstack_v + sp * 4 reg
    ← 64 vstack_sp_v sp + 1

vpop :
    sp → 64 vstack_sp_v
    if== sp 0
        return 0
    sp sp - 1
    ← 64 vstack_sp_v sp
    r → 32 vstack_v + sp * 4
    r

\\ ============================================================================
\\ Phase 1 — Scan for top-level compositions.
\\ ============================================================================
\\
\\ A composition header is:  IDENT (IDENT*) COLON NEWLINE
\\ Body tokens are every token after NEWLINE until the next token sitting at
\\ indent 0 (another header) or EOF. We record start/end in token-index units.

walk_collect :
    ← 64 emit_target_v 0
    comp_reset
    tc → 32 token_count_buf
    i 0
    host_pending 0

    label wc_loop
    if>= i tc
        goto wc_done
    t tok_type i

    \\ EOF
    if== t 0
        goto wc_done

    \\ HOST keyword flips the next composition to the ARM64 backend.
    if== t 35
        host_pending 1
        i i + 1
        goto wc_loop

    \\ Skip NEWLINE/INDENT at file scope.
    if== t 1
        i i + 1
        goto wc_loop
    if== t 2
        i i + 1
        goto wc_loop

    \\ Composition header must start on an IDENT.
    if!= t 5
        i i + 1
        goto wc_loop

    \\ Record name.
    name_off tok_offset i
    name_len tok_length i
    i i + 1

    \\ Count args: consecutive IDENTs until a COLON.
    arg_count 0
    label wc_args
    t tok_type i
    if== t 5
        arg_count arg_count + 1
        i i + 1
        goto wc_args

    \\ Require COLON; otherwise this isn't a header — skip the line.
    if!= t 72
        label wc_skip_line
        tt tok_type i
        if== tt 0
            goto wc_loop
        if== tt 1
            i i + 1
            host_pending 0
            goto wc_loop
        i i + 1
        goto wc_skip_line

    \\ Consume COLON and the NEWLINE that follows the header.
    i i + 1
    tn tok_type i
    if== tn 1
        i i + 1

    \\ Body starts here.
    body_start i

    \\ Walk until we hit either EOF or a NEWLINE followed by a non-INDENT
    \\ token (i.e. a new top-level header).
    label wc_body
    if>= i tc
        goto wc_body_done
    bt tok_type i
    if== bt 0
        goto wc_body_done
    if== bt 1
        \\ Newline: peek ahead — an INDENT means the body continues; any other
        \\ token (IDENT, HOST, ...) means a new top-level item begins.
        j i + 1
        if>= j tc
            goto wc_body_done
        nt tok_type j
        if== nt 2
            \\ Still indented — keep scanning.
            i i + 2
            goto wc_body
        if== nt 1
            \\ Blank line — consume and continue.
            i j
            goto wc_body
        \\ Top-level token ahead: body ends at this newline.
        goto wc_body_done
    i i + 1
    goto wc_body

    label wc_body_done
    body_end i
    comp_add name_off name_len arg_count body_start body_end host_pending
    host_pending 0
    goto wc_loop

    label wc_done
    0

\\ ============================================================================
\\ Phase 2 — Emit code by walking each top-level composition's body.
\\ ============================================================================

\\ Bind composition args to registers. First argument → R4, next → R5, ...
bind_args comp_idx :
    sym_reset
    reg_reset
    \\ For each arg token in the header, consume one register slot.
    n → 32 comp_arg_count_v + comp_idx * 4
    bs → 32 comp_body_start_v + comp_idx * 4
    \\ Arg tokens live immediately before body_start. Header layout is:
    \\   name arg1 arg2 ... argN : NEWLINE body_start
    \\ So arg tokens occupy the (name+1 .. name+n) slots. We recover them by
    \\ scanning backwards from (bs - 2) through (bs - 2 - n + 1).
    if== n 0
        return 0
    first bs - 2 - n + 1
    i 0
    label ba_loop
    if>= i n
        return 0
    ti first + i
    off tok_offset ti
    len tok_length ti
    r alloc_reg
    sym_add off len r
    i i + 1
    goto ba_loop

\\ Emit a single primitive by token type. Assumes top-of-stack holds the LHS
\\ (or, for binary ops, the right operand; LHS is the next one down).
\\
\\ Returns 1 if the token was consumed, 0 otherwise.
emit_primitive t :
    et → 64 emit_target_v

    \\ Binary FP: + - * /
    if== t 50
        rb vpop
        ra vpop
        rd alloc_reg
        if== et 0
            emit_fadd rd ra rb
        if== et 1
            emit_a64_fadd rd ra rb
        vpush rd
        return 1
    if== t 51
        rb vpop
        ra vpop
        rd alloc_reg
        if== et 0
            emit_fsub rd ra rb
        if== et 1
            emit_a64_fsub rd ra rb
        vpush rd
        return 1
    if== t 52
        rb vpop
        ra vpop
        rd alloc_reg
        if== et 0
            emit_fmul rd ra rb
        if== et 1
            emit_a64_fmul rd ra rb
        vpush rd
        return 1
    if== t 53
        rb vpop
        ra vpop
        rd alloc_reg
        if== et 0
            emit_fdiv rd ra rb rd
        if== et 1
            emit_a64_fdiv rd ra rb
        vpush rd
        return 1

    \\ Unary SFU: √ ≅ ≡
    if== t 79
        ra vpop
        rd alloc_reg
        if== et 0
            emit_sqrt rd ra
        vpush rd
        return 1
    if== t 80
        ra vpop
        rd alloc_reg
        if== et 0
            emit_sin rd ra
        if== et 1
            emit_a64_sin rd ra
        vpush rd
        return 1
    if== t 81
        ra vpop
        rd alloc_reg
        if== et 0
            emit_cos rd ra
        if== et 1
            emit_a64_cos rd ra
        vpush rd
        return 1

    \\ Σ — sum reduction (5x SHFL.BFLY + FADD). GPU only.
    if== t 75
        ra vpop
        if== et 0
            \\ delta = 16
            r1 alloc_reg
            emit_shfl_bfly r1 ra 16 0x1F
            r2 alloc_reg
            emit_fadd r2 ra r1
            \\ delta = 8
            r3 alloc_reg
            emit_shfl_bfly r3 r2 8 0x1F
            r4 alloc_reg
            emit_fadd r4 r2 r3
            \\ delta = 4
            r5 alloc_reg
            emit_shfl_bfly r5 r4 4 0x1F
            r6 alloc_reg
            emit_fadd r6 r4 r5
            \\ delta = 2
            r7 alloc_reg
            emit_shfl_bfly r7 r6 2 0x1F
            r8 alloc_reg
            emit_fadd r8 r6 r7
            \\ delta = 1
            r9 alloc_reg
            emit_shfl_bfly r9 r8 1 0x1F
            r10 alloc_reg
            emit_fadd r10 r8 r9
            vpush r10
            return 1
        \\ ARM64 has no warp shuffle — drop the value, leave result on stack.
        vpush ra
        return 1

    \\ △ ▽ # — TODO, consume and re-push input so walk continues.
    if== t 76
        ra vpop
        vpush ra
        return 1
    if== t 77
        ra vpop
        vpush ra
        return 1
    if== t 78
        return 1

    \\ → width addr  —  memory load.
    if== t 36
        return emit_load
    \\ ← width addr val  —  memory store.
    if== t 37
        return emit_store
    \\ ↑ $N  —  register read
    if== t 38
        return emit_reg_read
    \\ ↓ $N val  —  register write
    if== t 39
        return emit_reg_write

    0

\\ → width addr — consumes next two subexpressions from the token stream.
\\ Rather than a mini-parser, we read width literally and then consume the
\\ next already-pushed stack value as the address.
emit_load :
    et → 64 emit_target_v
    p → 64 walk_pos_v
    \\ Width is the next int token.
    wt tok_type p
    if!= wt 3
        return 1
    w_off tok_offset p
    w_len tok_length p
    width parse_int_tok w_off w_len
    ← 64 walk_pos_v p + 1
    \\ Address comes from the virtual stack (the next atom, which the main
    \\ walk loop will push before control returns here via the post-hook).
    \\ For the minimal compiler we instead treat the next token as an address
    \\ register directly.
    p2 → 64 walk_pos_v
    at tok_type p2
    addr_reg 0
    if== at 5
        ao tok_offset p2
        al tok_length p2
        r sym_find ao al
        if>= r 0
            addr_reg r
        ← 64 walk_pos_v p2 + 1
    rd alloc_reg
    if== et 0
        emit_ldg rd addr_reg
    vpush rd
    1

\\ ← width addr val — minimal: consumes width, then pops val and addr from stack.
emit_store :
    et → 64 emit_target_v
    p → 64 walk_pos_v
    wt tok_type p
    if== wt 3
        ← 64 walk_pos_v p + 1
    rv vpop
    ra vpop
    if== et 0
        emit_stg ra rv
    1

emit_reg_read :
    et → 64 emit_target_v
    p → 64 walk_pos_v
    t tok_type p
    if== t 97
        ← 64 walk_pos_v p + 1
    p2 → 64 walk_pos_v
    nt tok_type p2
    rd alloc_reg
    if== nt 3
        off tok_offset p2
        len tok_length p2
        sr parse_int_tok off len
        ← 64 walk_pos_v p2 + 1
        if== et 0
            emit_s2r rd sr
    vpush rd
    1

emit_reg_write :
    p → 64 walk_pos_v
    t tok_type p
    if== t 97
        ← 64 walk_pos_v p + 1
    p2 → 64 walk_pos_v
    nt tok_type p2
    if== nt 3
        ← 64 walk_pos_v p2 + 1
    vpop
    1

\\ ============================================================================
\\ Tiny numeric parser for literal tokens (no fractional/exp support).
\\ ============================================================================

parse_int_tok offset length :
    src → 64 lex_src_v
    val 0
    neg 0
    i 0
    c → 8 src + offset
    if== c 45
        neg 1
        i 1
    label pit_loop
    if>= i length
        goto pit_done
    c → 8 src + offset + i
    if< c 48
        goto pit_done
    if> c 57
        goto pit_done
    val val * 10 + (c - 48)
    i i + 1
    goto pit_loop
    label pit_done
    if== neg 1
        val 0 - val
    val

\\ ============================================================================
\\ Main body walker — single left-to-right pass over the body tokens.
\\ ============================================================================

walk_body body_start body_end :
    ← 64 walk_pos_v body_start
    label wb_loop
    p → 64 walk_pos_v
    if>= p body_end
        return 0
    t tok_type p

    \\ Skip structural tokens.
    if== t 1
        ← 64 walk_pos_v p + 1
        goto wb_loop
    if== t 2
        ← 64 walk_pos_v p + 1
        goto wb_loop
    if== t 0
        return 0

    \\ Integer / float literal — load immediate, push.
    if== t 3
        off tok_offset p
        len tok_length p
        val parse_int_tok off len
        rd alloc_reg
        et → 64 emit_target_v
        if== et 0
            emit_mov_imm rd val
        vpush rd
        ← 64 walk_pos_v p + 1
        goto wb_loop
    if== t 4
        rd alloc_reg
        et → 64 emit_target_v
        if== et 0
            emit_mov_imm rd 0
        vpush rd
        ← 64 walk_pos_v p + 1
        goto wb_loop

    \\ Keywords: for / each / stride / if — minimal stubs.
    if== t 16
        walk_for_kw
        goto wb_loop
    if== t 18
        walk_each_kw
        goto wb_loop
    if== t 19
        walk_stride_kw
        goto wb_loop
    if== t 13
        walk_if_kw
        goto wb_loop

    \\ Identifier: arg lookup, composition reference, or pseudo-primitive.
    if== t 5
        off tok_offset p
        len tok_length p
        ← 64 walk_pos_v p + 1
        r sym_find off len
        if>= r 0
            vpush r
            goto wb_loop
        cidx comp_find off len
        if>= cidx 0
            \\ Inline expansion (spec §4.1): recursively walk the referenced
            \\ composition's body. Symbols/regs are shared with the caller —
            \\ this is compile-time concatenation, not a function call.
            cbs → 32 comp_body_start_v + cidx * 4
            cbe → 32 comp_body_end_v + cidx * 4
            saved → 64 walk_pos_v
            walk_body cbs cbe
            ← 64 walk_pos_v saved
            goto wb_loop
        \\ Unknown identifier — allocate a placeholder register so downstream
        \\ operators still have something to pop.
        rd alloc_reg
        vpush rd
        goto wb_loop

    \\ Primitive operator?
    ← 64 walk_pos_v p + 1
    ok emit_primitive t
    if== ok 1
        goto wb_loop

    \\ Unrecognised — advance and continue.
    goto wb_loop

\\ ============================================================================
\\ Loop / conditional keyword stubs — spec §3.8.
\\ ============================================================================
\\
\\ These consume the associated operands from the token stream and emit the
\\ minimum viable sequence. A full implementation (branch patching, indent
\\ tracking, etc.) is left as a TODO — the scalar path is what matters now.

walk_for_kw :
    et → 64 emit_target_v
    \\ for i start end step — consume 4 operand tokens.
    consume_operands 4
    \\ ISETP + @P BRA + IADD3 placeholder: emit a zero-offset self-branch.
    if== et 0
        emit_isetp_ge 0 RZ RZ
        emit_bra_pred 0 0
    0

walk_each_kw :
    et → 64 emit_target_v
    \\ each i — consume loop-variable ident.
    consume_operands 1
    \\ S2R SR_TID.X + S2R SR_CTAID.X + IMAD (thread-parallel dispatch).
    if== et 0
        r0 alloc_reg
        emit_s2r r0 0x21
        r1 alloc_reg
        emit_s2r r1 0x25
        r2 alloc_reg
        emit_imad r2 r0 r1 RZ
    0

walk_stride_kw :
    et → 64 emit_target_v
    consume_operands 2
    if== et 0
        emit_isetp_ge 0 RZ RZ
        emit_bra_pred 0 0
    0

walk_if_kw :
    et → 64 emit_target_v
    \\ if== / if>= / if<  — lexed as a single IF token; operands are two expressions.
    consume_operands 2
    if== et 0
        emit_isetp_eq 0 RZ RZ
        emit_bra_pred 0 0
    0

\\ consume_operands — advance walk_pos past `n` non-structural tokens.
consume_operands n :
    i 0
    label co_loop
    if>= i n
        return 0
    p → 64 walk_pos_v
    t tok_type p
    if== t 0
        return 0
    if== t 1
        return 0
    ← 64 walk_pos_v p + 1
    i i + 1
    goto co_loop

\\ ============================================================================
\\ Top-level driver: emit code for every composition.
\\ ============================================================================

walk_top_level :
    walk_collect
    ← 64 any_gpu_v 0
    ← 64 any_arm64_v 0
    n → 64 comp_count_v
    i 0
    label wtl_loop
    if>= i n
        goto wtl_done

    is_host → 32 comp_is_host_v + i * 4
    ← 64 emit_target_v is_host

    bind_args i

    bs → 32 comp_body_start_v + i * 4
    be → 32 comp_body_end_v + i * 4
    walk_body bs be

    \\ Kernel terminator.
    if== is_host 0
        emit_exit
        ← 64 any_gpu_v 1
    if== is_host 1
        emit_a64_ret
        ← 64 any_arm64_v 1

    i i + 1
    goto wtl_loop
    label wtl_done
    0
