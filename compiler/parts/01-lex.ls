\\ ============================================================================
\\ 01-lex.ls — Lithos minimal compiler tokenizer
\\ ============================================================================
\\
\\ Input:  (src_ptr, src_len) — source bytes in memory.
\\ Output: tokens buffer (pre-declared global `buf tokens 262144`)
\\         token_count_buf (u32) — count of emitted tokens.
\\ Each token is 12 bytes: [type:u32, source_offset:u32, length:u32].
\\
\\ Token type constants:
\\   0  EOF          1  NEWLINE       2  INDENT
\\   3  INT          4  FLOAT         5  IDENT
\\   13 IF           16 FOR           17 ENDFOR
\\   18 EACH         19 STRIDE        33 LABEL
\\   34 EXIT         35 HOST          36 LOAD (→)
\\   37 STORE (←)    38 REG_READ (↑)  39 REG_WRITE (↓)
\\   50 PLUS         51 MINUS         52 STAR
\\   53 SLASH        54 EQ            55 EQEQ
\\   56 NEQ          57 LT            58 GT
\\   59 LTE          60 GTE           61 AMP
\\   62 PIPE         63 CARET         64 SHL
\\   65 SHR          67 LBRACK        68 RBRACK
\\   69 LPAREN       70 RPAREN        71 COMMA
\\   72 COLON        73 DOT           74 AT
\\   75 SIGMA (Σ)    76 MAX (△)       77 MIN (▽)
\\   78 INDEX (#)    79 SQRT (√)      80 SIN (≅)
\\   81 COS (≡)      90 NEWLINE_LF    92 BACKSLASH
\\   96 CONSTANT     97 DOLLAR
\\
\\ Note: `tokens` buffer and `token_count_buf` are declared in 00-constants.ls.

\\ ============================================================================
\\ Lexer-local globals (scratch state lifted to globals for robustness)
\\ ============================================================================

buf lex_pos_v 8
buf lex_lp_v 8
buf lex_src_v 8
buf lex_len_v 8

\\ ============================================================================
\\ Emit one (type, offset, length) token into the tokens buffer.
\\ ============================================================================

emit_token type offset length :
    tc → 32 token_count_buf
    idx tc * 3
    ← 32 tokens + idx type
    idx1 idx + 1
    ← 32 tokens + idx1 offset
    idx2 idx + 2
    ← 32 tokens + idx2 length
    ← 32 token_count_buf tc + 1

\\ ============================================================================
\\ Character classification
\\ ============================================================================

is_alpha c :
    upper c >= 65 & c <= 90
    lower c >= 97 & c <= 122
    under c == 95
    result upper | lower | under

is_digit c :
    result c >= 48 & c <= 57

is_hex_digit c :
    d c >= 48 & c <= 57
    u c >= 65 & c <= 70
    l c >= 97 & c <= 102
    result d | u | l

is_alnum c :
    a is_alpha c
    d is_digit c
    result a | d

is_newline c :
    lf c == 10
    cr c == 13
    result lf | cr

\\ ============================================================================
\\ Token scanning helpers (goto-based — no `while`, no tail recursion)
\\ ============================================================================

\\ scan_ident — advance past an identifier / keyword.
\\ Accepts [a-zA-Z0-9_] plus a few punctuation chars used in keyword-ish tokens
\\ (like `if==`, `if>=`, `if<`) so the whole thing is one token we dispatch on.
scan_ident src pos end :
    np pos
    label si_loop
    if>= np end
        goto si_done
    c → 8 (src + np)
    a is_alnum c
    dot c == 46
    al c == 60
    ar c == 62
    eq c == 61
    bg c == 33
    ok a | dot | al | ar | eq | bg
    if== ok 0
        goto si_done
    np np + 1
    goto si_loop
    label si_done
    np

\\ scan_number — advance past an integer or float literal, including 0x.. hex.
scan_number src pos end :
    np pos
    c → 8 (src + np)

    if== c 48
        if< np + 1 end
            c2 → 8 (src + np + 1)
            if== c2 120
                goto sn_hex
            if== c2 88
                goto sn_hex

    goto sn_dec

    label sn_hex
    np np + 2
    label sn_hex_loop
    if>= np end
        goto sn_done
    ch → 8 (src + np)
    ok is_hex_digit ch
    if== ok 0
        goto sn_done
    np np + 1
    goto sn_hex_loop

    label sn_dec
    label sn_dec_loop
    if>= np end
        goto sn_done
    ch → 8 (src + np)
    d is_digit ch
    dot ch == 46
    ok d | dot
    if== ok 0
        goto sn_done
    np np + 1
    goto sn_dec_loop

    label sn_done
    np

\\ scan_to_eol — skip to the next newline (or end-of-input).
scan_to_eol src pos end :
    np pos
    label se_loop
    if>= np end
        goto se_done
    c → 8 (src + np)
    nl is_newline c
    if nl
        goto se_done
    np np + 1
    goto se_loop
    label se_done
    np

\\ ============================================================================
\\ Keyword matching — returns token type (IDENT=5 if no keyword matches)
\\ ============================================================================

match_keyword src offset length :
    tok_type 5

    if== length 2
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        \\ "if" -> IF (13)
        if== b0 105
            if== b1 102
                tok_type 13
                return

    if== length 3
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        b2 → 8 (src + offset + 2)
        \\ "for" -> FOR (16)
        if== b0 102
            if== b1 111
                if== b2 114
                    tok_type 16
                    return

    if== length 4
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        b2 → 8 (src + offset + 2)
        b3 → 8 (src + offset + 3)
        \\ "each" -> EACH (18)
        if== b0 101
            if== b1 97
                if== b2 99
                    if== b3 104
                        tok_type 18
                        return
        \\ "exit" -> EXIT (34)
        if== b0 101
            if== b1 120
                if== b2 105
                    if== b3 116
                        tok_type 34
                        return
        \\ "host" -> HOST (35)
        if== b0 104
            if== b1 111
                if== b2 115
                    if== b3 116
                        tok_type 35
                        return

    if== length 5
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        b2 → 8 (src + offset + 2)
        b3 → 8 (src + offset + 3)
        b4 → 8 (src + offset + 4)
        \\ "label" -> LABEL (33)
        if== b0 108
            if== b1 97
                if== b2 98
                    if== b3 101
                        if== b4 108
                            tok_type 33
                            return

    if== length 6
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        b2 → 8 (src + offset + 2)
        b3 → 8 (src + offset + 3)
        b4 → 8 (src + offset + 4)
        b5 → 8 (src + offset + 5)
        \\ "stride" -> STRIDE (19)
        if== b0 115
            if== b1 116
                if== b2 114
                    if== b3 105
                        if== b4 100
                            if== b5 101
                                tok_type 19
                                return
        \\ "endfor" -> ENDFOR (17)
        if== b0 101
            if== b1 110
                if== b2 100
                    if== b3 102
                        if== b4 111
                            if== b5 114
                                tok_type 17
                                return

    if== length 8
        b0 → 8 (src + offset)
        b1 → 8 (src + offset + 1)
        b2 → 8 (src + offset + 2)
        b3 → 8 (src + offset + 3)
        b4 → 8 (src + offset + 4)
        b5 → 8 (src + offset + 5)
        b6 → 8 (src + offset + 6)
        b7 → 8 (src + offset + 7)
        \\ "constant" -> CONSTANT (96)
        if== b0 99
            if== b1 111
                if== b2 110
                    if== b3 115
                        if== b4 116
                            if== b5 97
                                if== b6 110
                                    if== b7 116
                                        tok_type 96
                                        return

    tok_type

\\ ============================================================================
\\ Number type classification — INT (3) vs FLOAT (4) by presence of '.'
\\ ============================================================================

classify_number src offset length :
    tok_type 3
    i 0
    label cn_loop
    if>= i length
        goto cn_done
    c → 8 (src + offset + i)
    if== c 46
        tok_type 4
        goto cn_done
    i i + 1
    goto cn_loop
    label cn_done
    tok_type

\\ ============================================================================
\\ Multi-character operator tables
\\ ============================================================================

\\ lex_match_twochar — two-char ASCII ops. Returns token type or 0.
lex_match_twochar c cn :
    tok 0
    if== c 61
        if== cn 61
            tok 55
            return
    if== c 33
        if== cn 61
            tok 56
            return
    if== c 60
        if== cn 61
            tok 59
            return
        if== cn 60
            tok 64
            return
    if== c 62
        if== cn 61
            tok 60
            return
        if== cn 62
            tok 65
            return
    tok

\\ lex_match_e2 — 3-byte UTF-8 tokens starting with 0xE2. Returns type or 0.
lex_match_e2 b1 b2 :
    tok 0
    if== b1 0x86
        if== b2 0x92
            tok 36
            return
        if== b2 0x90
            tok 37
            return
        if== b2 0x91
            tok 38
            return
        if== b2 0x93
            tok 39
            return
    if== b1 0x96
        if== b2 0xB3
            tok 76
            return
        if== b2 0xBD
            tok 77
            return
    if== b1 0x88
        if== b2 0x9A
            tok 79
            return
    if== b1 0x89
        if== b2 0x85
            tok 80
            return
        if== b2 0xA1
            tok 81
            return
    tok

\\ ============================================================================
\\ Main lexer — flat control flow using goto/label.
\\ ============================================================================

lex src src_len :
    pos 0
    ← 32 token_count_buf 0
    lp 1

    label lex_loop
    if>= pos src_len
        goto lex_done
    c → 8 (src + pos)

    \\ ---- Indent at start of line ----
    if== lp 0
        goto lex_skip_indent
    indent 0
    label lex_indent_loop
    if>= pos src_len
        goto lex_indent_done
    c → 8 (src + pos)
    if== c 32
        indent indent + 1
        pos pos + 1
        goto lex_indent_loop
    if== c 9
        indent indent + 4
        pos pos + 1
        goto lex_indent_loop
    label lex_indent_done
    emit_token 2 pos indent
    lp 0
    if>= pos src_len
        goto lex_done
    c → 8 (src + pos)
    label lex_skip_indent

    \\ ---- Whitespace (space, tab) ----
    if== c 32
        pos pos + 1
        goto lex_loop
    if== c 9
        pos pos + 1
        goto lex_loop

    \\ ---- Newline LF ----
    if== c 10
        emit_token 1 pos 1
        pos pos + 1
        lp 1
        goto lex_loop

    \\ ---- Newline CR ----
    if== c 13
        emit_token 1 pos 1
        pos pos + 1
        lp 1
        goto lex_loop

    \\ ---- Comment \\ ... (double backslash to end of line) ----
    if== c 92
        goto lex_try_comment
    goto lex_not_comment
    label lex_try_comment
    if>= pos + 1 src_len
        goto lex_not_comment
    t → 8 (src + pos + 1)
    if== t 92
        pos scan_to_eol src pos src_len
        goto lex_loop
    label lex_not_comment

    \\ ---- Number ----
    t is_digit c
    if t
        start pos
        pos scan_number src pos src_len
        len pos - start
        t classify_number src start len
        emit_token t start len
        goto lex_loop

    \\ ---- Negative number (unary minus attached to digit) ----
    if== c 45
        goto lex_try_neg
    goto lex_not_neg
    label lex_try_neg
    if>= pos + 1 src_len
        goto lex_not_neg
    t → 8 (src + pos + 1)
    t is_digit t
    if== t 0
        goto lex_not_neg
    start pos
    pos pos + 1
    pos scan_number src pos src_len
    len pos - start
    t classify_number src start len
    emit_token t start len
    goto lex_loop
    label lex_not_neg

    \\ ---- Identifier / keyword ----
    t is_alpha c
    if t
        start pos
        pos scan_ident src pos src_len
        len pos - start
        t match_keyword src start len
        emit_token t start len
        goto lex_loop

    \\ ---- Two-byte ASCII operators (==, !=, <=, >=, <<, >>) ----
    if>= pos + 1 src_len
        goto lex_no_twochar
    t → 8 (src + pos + 1)
    t lex_match_twochar c t
    if t
        emit_token t pos 2
        pos pos + 2
        goto lex_loop
    label lex_no_twochar

    \\ ---- UTF-8 multi-byte tokens ----
    if== c 0xE2
        goto lex_try_e2
    if== c 0xCE
        goto lex_try_ce
    goto lex_no_utf8

    label lex_try_e2
    if>= pos + 2 src_len
        goto lex_no_utf8
    start → 8 (src + pos + 1)
    t → 8 (src + pos + 2)
    t lex_match_e2 start t
    if t
        emit_token t pos 3
        pos pos + 3
        goto lex_loop
    goto lex_no_utf8

    label lex_try_ce
    if>= pos + 1 src_len
        goto lex_no_utf8
    t → 8 (src + pos + 1)
    if== t 0xA3
        emit_token 75 pos 2
        pos pos + 2
        goto lex_loop

    label lex_no_utf8

    \\ ---- Single-character operators ----
    if== c 43
        emit_token 50 pos 1
        pos pos + 1
        goto lex_loop
    if== c 42
        emit_token 52 pos 1
        pos pos + 1
        goto lex_loop
    if== c 47
        emit_token 53 pos 1
        pos pos + 1
        goto lex_loop
    if== c 61
        emit_token 54 pos 1
        pos pos + 1
        goto lex_loop
    if== c 60
        emit_token 57 pos 1
        pos pos + 1
        goto lex_loop
    if== c 62
        emit_token 58 pos 1
        pos pos + 1
        goto lex_loop
    if== c 38
        emit_token 61 pos 1
        pos pos + 1
        goto lex_loop
    if== c 124
        emit_token 62 pos 1
        pos pos + 1
        goto lex_loop
    if== c 94
        emit_token 63 pos 1
        pos pos + 1
        goto lex_loop
    if== c 91
        emit_token 67 pos 1
        pos pos + 1
        goto lex_loop
    if== c 93
        emit_token 68 pos 1
        pos pos + 1
        goto lex_loop
    if== c 40
        emit_token 69 pos 1
        pos pos + 1
        goto lex_loop
    if== c 41
        emit_token 70 pos 1
        pos pos + 1
        goto lex_loop
    if== c 44
        emit_token 71 pos 1
        pos pos + 1
        goto lex_loop
    if== c 58
        emit_token 72 pos 1
        pos pos + 1
        goto lex_loop
    if== c 46
        emit_token 73 pos 1
        pos pos + 1
        goto lex_loop
    if== c 64
        emit_token 74 pos 1
        pos pos + 1
        goto lex_loop
    if== c 36
        emit_token 97 pos 1
        pos pos + 1
        goto lex_loop
    if== c 35
        emit_token 78 pos 1
        pos pos + 1
        goto lex_loop
    if== c 45
        emit_token 51 pos 1
        pos pos + 1
        goto lex_loop
    if== c 92
        emit_token 92 pos 1
        pos pos + 1
        goto lex_loop

    \\ Unknown byte — skip it silently.
    pos pos + 1
    goto lex_loop

    label lex_done
    emit_token 0 pos 0

\\ Top-level entry used by the driver.
lithos_lex src src_len :
    lex src src_len
