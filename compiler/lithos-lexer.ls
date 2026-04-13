\\ lithos-lexer.li — Self-hosting lexer for the Lithos .li format
\\ Target: ARM64 (host code, not GPU kernel)
\\
\\ First stage of the self-hosting compiler pipeline:
\\   lithos-lexer.li  ->  tokenize .li source into flat token buffer
\\   lithos-parser.li ->  parse token stream into AST
\\   lithos-emit.li   ->  emit ARM64 or SASS from AST
\\
\\ Input:  byte buffer (mmap'd .li source file) + length
\\ Output: flat u32 buffer of token triples (type, offset, length)
\\
\\ Token encoding: 3 consecutive u32 values per token
\\   [0] type    — token type enum (see TOK_* constants below)
\\   [1] offset  — byte offset in source buffer
\\   [2] length  — byte length of token text
\\
\\ The lexer is line-oriented. Newlines and leading indentation are
\\ significant tokens. Comments (\\) consume to end of line.

\\ ============================================================================
\\ Token type constants
\\ ============================================================================
\\
\\ NOTE: .li does not yet have const declarations that produce named
\\ integer constants at compile time. The values below define the token
\\ type enum. The compiler will treat bare integers in comparisons until
\\ const is implemented. Each constant is numbered so downstream code
\\ can use the raw integer if needed.
\\
\\ --- Structural ---
\\ TOK_EOF        = 0    end of input
\\ TOK_NEWLINE    = 1    newline character (LF)
\\ TOK_INDENT     = 2    leading whitespace at start of line (length = space count)
\\
\\ --- Literals ---
\\ TOK_INT        = 3    decimal or hex integer literal
\\ TOK_FLOAT      = 4    floating-point literal (has decimal point)
\\ TOK_IDENT      = 5    identifier (not a keyword)
\\
\\ --- Keywords ---
\\ TOK_FN         = 10   (REMOVED — no fn keyword. Compositions are name args :)
\\ TOK_KERNEL     = 11
\\ TOK_PARAM      = 12
\\ TOK_IF         = 13
\\ TOK_ELSE       = 14
\\ TOK_ELIF       = 15
\\ TOK_FOR        = 16
\\ TOK_ENDFOR     = 17
\\ TOK_EACH       = 18
\\ TOK_STRIDE     = 19
\\ TOK_WHILE      = 20
\\ TOK_RETURN     = 21
\\ TOK_CONST      = 22
\\ TOK_VAR        = 23
\\ TOK_BUF        = 24
\\ TOK_WEIGHT     = 25
\\ TOK_LAYER      = 26
\\ TOK_BIND       = 27
\\ TOK_RUNTIME    = 28
\\ TOK_TEMPLATE   = 29
\\ TOK_PROJECT    = 30
\\ TOK_SHARED     = 31
\\ TOK_BARRIER    = 32
\\ TOK_LABEL      = 33
\\ TOK_EXIT       = 34
\\ TOK_HOST       = 35   host (marks a host composition vs gpu composition)
\\ TOK_LOAD       = 36   →   (memory load: → width addr)
\\ TOK_STORE      = 37   ←   (memory store: ← width addr val)
\\ TOK_REG_READ   = 38   ↑   (register read: ↑ $N)
\\ TOK_REG_WRITE  = 39   ↓   (register write: ↓ $N val)
\\
\\ --- Types ---
\\ TOK_F32        = 40
\\ TOK_U32        = 41
\\ TOK_S32        = 42
\\ TOK_F16        = 43
\\ TOK_PTR        = 44
\\ TOK_VOID       = 45
\\
\\ --- Operators ---
\\ TOK_PLUS       = 50   +
\\ TOK_MINUS      = 51   -
\\ TOK_STAR       = 52   *
\\ TOK_SLASH      = 53   /
\\ TOK_EQ         = 54   =
\\ TOK_EQEQ       = 55   ==
\\ TOK_NEQ        = 56   !=
\\ TOK_LT         = 57   <   (comparison)
\\ TOK_GT         = 58   >   (comparison)
\\ TOK_LTE        = 59   <=
\\ TOK_GTE        = 60   >=
\\ TOK_AMP        = 61   &
\\ TOK_PIPE       = 62   |
\\ TOK_CARET      = 63   ^
\\ TOK_SHL        = 64   <<
\\ TOK_SHR        = 65   >>
\\ TOK_ARROW      = 66   (REMOVED — compositions use : not ->)
\\
\\ --- Reduction / Math Unicode tokens ---
\\ TOK_SUM        = 75   Σ   (sum reduction)
\\ TOK_MAX        = 76   △   (max reduction)
\\ TOK_MIN        = 77   ▽   (min reduction)
\\ TOK_INDEX      = 78   #   (index operator: # △ x = argmax)
\\ TOK_SQRT       = 79   √   (square root)
\\ TOK_SIN        = 80   ≅   (MUFU.SIN)
\\ TOK_COS        = 81   ≡   (MUFU.COS)
\\ TOK_LBRACK     = 67   [
\\ TOK_RBRACK     = 68   ]
\\ TOK_LPAREN     = 69   (
\\ TOK_RPAREN     = 70   )
\\
\\ --- Punctuation ---
\\ TOK_COMMA      = 71   ,
\\ TOK_COLON      = 72   :
\\ TOK_DOT        = 73   .
\\ TOK_AT         = 74   @

\\ ============================================================================
\\ Token output buffer
\\ ============================================================================
\\ Each token is 3 u32 values: type, offset, length.
\\ 262144 u32 slots = room for 87381 tokens.

buf tokens 262144
var token_count 0

\\ ============================================================================
\\ Keyword table
\\ ============================================================================
\\ The keyword table is a flat array of (string-pointer, length, token-type)
\\ triples. After scanning an identifier, we linear-scan this table.
\\
\\ NOTE: .li does not yet support string literals or initialized data arrays.
\\ The keyword table here is represented as a sequence of comparisons in
\\ match_keyword. When string literals and initialized buffers are added to
\\ the language, this can become a proper table lookup.

\\ ============================================================================
\\ Emit a token into the output buffer
\\ ============================================================================

emit_token type offset length :
    idx = token_count * 3
    tokens [ idx ] = type
    idx1 = idx + 1
    tokens [ idx1 ] = offset
    idx2 = idx + 2
    tokens [ idx2 ] = length
    token_count = token_count + 1

\\ ============================================================================
\\ Character classification
\\ ============================================================================
\\ These operate on a single byte value (0-255).
\\ Result is nonzero (true) or zero (false).

is_alpha c :
    \\ A-Z: 65-90, a-z: 97-122, underscore: 95
    upper = c >= 65 & c <= 90
    lower = c >= 97 & c <= 122
    under = c == 95
    result = upper | lower | under

is_digit c :
    \\ 0-9: 48-57
    result = c >= 48 & c <= 57

is_hex_digit c :
    \\ 0-9, A-F, a-f
    digit = c >= 48 & c <= 57
    upper = c >= 65 & c <= 70
    lower = c >= 97 & c <= 102
    result = digit | upper | lower

is_alnum c :
    \\ alpha or digit
    alpha = is_alpha c
    digit = is_digit c
    result = alpha | digit

is_space c :
    \\ space (32) or tab (9)
    sp = c == 32
    tb = c == 9
    result = sp | tb

is_newline c :
    \\ LF (10) or CR (13)
    lf = c == 10
    cr = c == 13
    result = lf | cr

\\ ============================================================================
\\ Token scanning helpers
\\ ============================================================================
\\ Each scanner advances from pos, returning the new position.
\\ NOTE: src[pos] means byte-load from address (src + pos).
\\ .li does not yet have byte-level memory access syntax.
\\ We write src[pos] to mean "load byte at src+pos" and mark
\\ where the compiler needs to emit LDRB (ARM64) instead of LDR.

scan_ident src pos end :
    \\ Advance while character is alphanumeric, underscore, or dot/angle
    \\ (for identifiers like shfl.bfly, u32>f32, if>=, if==, if<, if>)
    new_pos = pos
    while new_pos < end
        c = src [ new_pos ]          \\ BYTE LOAD: ldrb
        alpha = is_alnum c
        dot   = c == 46              \\ '.'
        angle_l = c == 60            \\ '<'  (for u32>f32 style)
        angle_r = c == 62            \\ '>'  (for u32>f32 style)
        eq    = c == 61              \\ '='  (for if>=, if==)
        bang  = c == 33              \\ '!'  (for if!=)
        ok = alpha | dot | angle_l | angle_r | eq | bang
        if ok == 0
            return
        new_pos = new_pos + 1

scan_number src pos end :
    \\ Advance over a number literal.
    \\ Handles: decimal integers, hex (0x...), floats (with '.')
    new_pos = pos
    c = src [ new_pos ]              \\ BYTE LOAD

    \\ Check for hex prefix: 0x or 0X
    if c == 48                       \\ '0'
        if new_pos + 1 < end
            c2 = src [ new_pos + 1 ] \\ BYTE LOAD
            if c2 == 120 | c2 == 88  \\ 'x' or 'X'
                new_pos = new_pos + 2
                while new_pos < end
                    ch = src [ new_pos ]     \\ BYTE LOAD
                    ok = is_hex_digit ch
                    if ok == 0
                        return
                    new_pos = new_pos + 1
                return

    \\ Decimal integer or float
    while new_pos < end
        ch = src [ new_pos ]         \\ BYTE LOAD
        digit = is_digit ch
        dot   = ch == 46             \\ '.'
        ok = digit | dot
        if ok == 0
            return
        new_pos = new_pos + 1

scan_to_eol src pos end :
    \\ Advance to next newline or end of input (for comments)
    new_pos = pos
    while new_pos < end
        c = src [ new_pos ]          \\ BYTE LOAD
        nl = is_newline c
        if nl
            return
        new_pos = new_pos + 1

\\ ============================================================================
\\ Keyword matching
\\ ============================================================================
\\ After scanning an identifier, check if the span [offset, offset+length)
\\ matches a known keyword. Returns the keyword token type, or TOK_IDENT (5)
\\ if no match.
\\
\\ NOTE: This uses a byte-by-byte comparison idiom. .li does not yet have
\\ string comparison built-in. Each check tests:
\\   1. Length matches expected keyword length
\\   2. Each byte matches (unrolled for short keywords)
\\
\\ The helper cmp_kw compares src[offset..offset+length) against a known
\\ keyword. Since we cannot express string constants yet, each keyword
\\ match is inlined with byte comparisons against ASCII values.
\\
\\ When .li gains string literals: replace with table lookup.

match_keyword src offset length :
    \\ Default to identifier
    tok_type = 5

    \\ --- Length 2 keywords ---
    if length == 2
        b0 = src [ offset ]              \\ BYTE LOAD
        b1 = src [ offset + 1 ]          \\ BYTE LOAD

        \\ fn — REMOVED (no fn keyword; compositions use name args :)
        \\ or
        if b0 == 111 & b1 == 114         \\ 'o' 'r'
            tok_type = 5                  \\ 'or' is not a keyword token — treated as ident/intrinsic
            return

    \\ --- Length 3 keywords ---
    if length == 3
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]
        b2 = src [ offset + 2 ]

        \\ for
        if b0 == 102 & b1 == 111 & b2 == 114
            tok_type = 16
            return
        \\ var
        if b0 == 118 & b1 == 97 & b2 == 114
            tok_type = 23
            return
        \\ buf
        if b0 == 98 & b1 == 117 & b2 == 102
            tok_type = 24
            return
        \\ f32
        if b0 == 102 & b1 == 51 & b2 == 50
            tok_type = 40
            return
        \\ u32
        if b0 == 117 & b1 == 51 & b2 == 50
            tok_type = 41
            return
        \\ s32
        if b0 == 115 & b1 == 51 & b2 == 50
            tok_type = 42
            return
        \\ f16
        if b0 == 102 & b1 == 49 & b2 == 54
            tok_type = 43
            return
        \\ ptr
        if b0 == 112 & b1 == 116 & b2 == 114
            tok_type = 44
            return

    \\ --- Length 4 keywords ---
    if length == 4
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]
        b2 = src [ offset + 2 ]
        b3 = src [ offset + 3 ]

        \\ each
        if b0 == 101 & b1 == 97 & b2 == 99 & b3 == 104
            tok_type = 18
            return
        \\ else
        if b0 == 101 & b1 == 108 & b2 == 115 & b3 == 101
            tok_type = 14
            return
        \\ elif
        if b0 == 101 & b1 == 108 & b2 == 105 & b3 == 102
            tok_type = 15
            return
        \\ void
        if b0 == 118 & b1 == 111 & b2 == 105 & b3 == 100
            tok_type = 45
            return
        \\ bind
        if b0 == 98 & b1 == 105 & b2 == 110 & b3 == 100
            tok_type = 27
            return
        \\ exit
        if b0 == 101 & b1 == 120 & b2 == 105 & b3 == 116
            tok_type = 34
            return
        \\ host
        if b0 == 104 & b1 == 111 & b2 == 115 & b3 == 116
            tok_type = 35
            return
        \\ goto
        if b0 == 103 & b1 == 111 & b2 == 116 & b3 == 111
            tok_type = 93
            return
        \\ trap
        if b0 == 116 & b1 == 114 & b2 == 97 & b3 == 112
            tok_type = 94
            return

    \\ --- Length 5 keywords ---
    if length == 5
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]
        b2 = src [ offset + 2 ]
        b3 = src [ offset + 3 ]
        b4 = src [ offset + 4 ]

        \\ param
        if b0 == 112 & b1 == 97 & b2 == 114 & b3 == 97 & b4 == 109
            tok_type = 12
            return
        \\ while
        if b0 == 119 & b1 == 104 & b2 == 105 & b3 == 108 & b4 == 101
            tok_type = 20
            return
        \\ const
        if b0 == 99 & b1 == 111 & b2 == 110 & b3 == 115 & b4 == 116
            tok_type = 22
            return
        \\ layer
        if b0 == 108 & b1 == 97 & b2 == 121 & b3 == 101 & b4 == 114
            tok_type = 26
            return
        \\ label
        if b0 == 108 & b1 == 97 & b2 == 98 & b3 == 101 & b4 == 108
            tok_type = 33
            return

    \\ --- Length 6 keywords ---
    if length == 6
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]
        b2 = src [ offset + 2 ]
        b3 = src [ offset + 3 ]
        b4 = src [ offset + 4 ]
        b5 = src [ offset + 5 ]

        \\ kernel
        if b0 == 107 & b1 == 101 & b2 == 114 & b3 == 110 & b4 == 101 & b5 == 108
            tok_type = 11
            return
        \\ stride
        if b0 == 115 & b1 == 116 & b2 == 114 & b3 == 105 & b4 == 100 & b5 == 101
            tok_type = 19
            return
        \\ return
        if b0 == 114 & b1 == 101 & b2 == 116 & b3 == 117 & b4 == 114 & b5 == 110
            tok_type = 21
            return
        \\ weight
        if b0 == 119 & b1 == 101 & b2 == 105 & b3 == 103 & b4 == 104 & b5 == 116
            tok_type = 25
            return
        \\ shared
        if b0 == 115 & b1 == 104 & b2 == 97 & b3 == 114 & b4 == 101 & b5 == 100
            tok_type = 31
            return
        \\ endfor
        if b0 == 101 & b1 == 110 & b2 == 100 & b3 == 102 & b4 == 111 & b5 == 114
            tok_type = 17
            return

    \\ --- Length 7 keywords ---
    if length == 7
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]

        \\ runtime (114 117 110 116 105 109 101)
        if b0 == 114 & b1 == 117
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            if b2 == 110 & b3 == 116 & b4 == 105 & b5 == 109 & b6 == 101
                tok_type = 28
                return
        \\ barrier (98 97 114 114 105 101 114)
        if b0 == 98 & b1 == 97
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            if b2 == 114 & b3 == 114 & b4 == 105 & b5 == 101 & b6 == 114
                tok_type = 32
                return
        \\ project (112 114 111 106 101 99 116)
        if b0 == 112 & b1 == 114
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            if b2 == 111 & b3 == 106 & b4 == 101 & b5 == 99 & b6 == 116
                tok_type = 30
                return

    \\ --- Length 8 keywords ---
    if length == 8
        b0 = src [ offset ]
        b1 = src [ offset + 1 ]

        \\ template (116 101 109 112 108 97 116 101)
        if b0 == 116 & b1 == 101
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            b7 = src [ offset + 7 ]
            if b2 == 109 & b3 == 112 & b4 == 108 & b5 == 97 & b6 == 116 & b7 == 101
                tok_type = 29
                return
        \\ constant (99 111 110 115 116 97 110 116)
        if b0 == 99 & b1 == 111
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            b7 = src [ offset + 7 ]
            if b2 == 110 & b3 == 115 & b4 == 116 & b5 == 97 & b6 == 110 & b7 == 116
                tok_type = 96
                return
        \\ continue (99 111 110 116 105 110 117 101)
        if b0 == 99 & b1 == 111
            b2 = src [ offset + 2 ]
            b3 = src [ offset + 3 ]
            b4 = src [ offset + 4 ]
            b5 = src [ offset + 5 ]
            b6 = src [ offset + 6 ]
            b7 = src [ offset + 7 ]
            if b2 == 110 & b3 == 116 & b4 == 105 & b5 == 110 & b6 == 117 & b7 == 101
                tok_type = 95
                return

    \\ No keyword matched — tok_type remains TOK_IDENT (5)

\\ ============================================================================
\\ Number type classification
\\ ============================================================================
\\ After scanning a number, determine if it is integer or float.
\\ Returns TOK_INT (3) or TOK_FLOAT (4).

classify_number src offset length :
    tok_type = 3                     \\ default: integer
    i = 0
    while i < length
        c = src [ offset + i ]       \\ BYTE LOAD
        if c == 46                   \\ '.' decimal point
            tok_type = 4             \\ it is a float
            return
        i = i + 1

\\ ============================================================================
\\ Main lexer
\\ ============================================================================
\\ Tokenizes the entire source buffer. Writes tokens to the global
\\ tokens[] buffer and increments token_count.
\\
\\ Line-oriented: at the start of each line, we emit a TOK_INDENT token
\\ whose length field encodes the number of leading spaces. Then we
\\ scan tokens until newline or end-of-input, emitting TOK_NEWLINE at
\\ each line boundary.

lex src src_len :
    pos = 0
    token_count = 0
    line_start = 1                   \\ flag: next char is start of line

    while pos < src_len
        c = src [ pos ]              \\ BYTE LOAD

        \\ ---- Handle start of line: measure indentation ----
        if line_start
            indent = 0
            while pos < src_len
                ic = src [ pos ]     \\ BYTE LOAD
                if ic == 32          \\ space
                    indent = indent + 1
                    pos = pos + 1
                elif ic == 9         \\ tab = 4 spaces
                    indent = indent + 4
                    pos = pos + 1
                else
                    \\ done counting indent
                    \\ break out of indent loop
                    \\ NOTE: .li lacks break; we use a flag + continue pattern
                    \\ For now, express as: set pos past indent, fall through
                    goto indent_done
            label indent_done
            \\ Emit indent token (even if 0 — parser needs to know line boundaries)
            emit_token 2 pos indent  \\ TOK_INDENT, offset=first non-space, length=indent count
            line_start = 0
            \\ Re-read current char after advancing past indent
            if pos >= src_len
                return
            c = src [ pos ]          \\ BYTE LOAD

        \\ ---- Newline ----
        if c == 10                   \\ LF
            emit_token 1 pos 1       \\ TOK_NEWLINE
            pos = pos + 1
            line_start = 1
            \\ Skip CR after LF (or LF after CR) for Windows line endings
            if pos < src_len
                c2 = src [ pos ]     \\ BYTE LOAD
                if c2 == 13
                    pos = pos + 1
            continue

        if c == 13                   \\ CR
            emit_token 1 pos 1       \\ TOK_NEWLINE
            pos = pos + 1
            line_start = 1
            if pos < src_len
                c2 = src [ pos ]
                if c2 == 10
                    pos = pos + 1
            continue

        \\ ---- Skip inline whitespace (space, tab) ----
        if c == 32 | c == 9
            pos = pos + 1
            continue

        \\ ---- # is TOK_INDEX (reduction index operator) ----
        if c == 35                   \\ '#'
            emit_token 78 pos 1      \\ TOK_INDEX
            pos = pos + 1
            continue

        \\ ---- $ is TOK_DOLLAR (register sigil) ----
        if c == 36                   \\ '$'
            emit_token 97 pos 1      \\ TOK_DOLLAR
            pos = pos + 1
            continue

        \\ ---- Comments: \\ to end of line (double backslash) ----
        if c == 92                   \\ first '\'
            if pos + 1 < src_len
                c2 = src [ pos + 1 ] \\ BYTE LOAD
                if c2 == 92          \\ second '\' -> comment
                    pos = scan_to_eol src pos src_len
                    continue

        \\ ---- Number literals ----
        \\ A digit starts a number. Also a negative sign followed by digit.
        digit = is_digit c
        if digit
            start = pos
            pos = scan_number src pos src_len
            length = pos - start
            num_type = classify_number src start length
            emit_token num_type start length
            continue

        \\ Negative number: '-' followed immediately by digit
        if c == 45                   \\ '-'
            if pos + 1 < src_len
                cnext = src [ pos + 1 ]  \\ BYTE LOAD
                ndig = is_digit cnext
                if ndig
                    start = pos
                    pos = pos + 1    \\ skip the '-'
                    pos = scan_number src pos src_len
                    length = pos - start
                    num_type = classify_number src start length
                    emit_token num_type start length
                    continue

        \\ ---- Identifiers and keywords ----
        alpha = is_alpha c
        if alpha
            start = pos
            pos = scan_ident src pos src_len
            length = pos - start
            kw = match_keyword src start length
            emit_token kw start length
            continue

        \\ ---- Two-character operators ----
        \\ Must check before single-char operators.
        \\ We need the next char for lookahead.
        if pos + 1 < src_len
            cnext = src [ pos + 1 ]  \\ BYTE LOAD

            \\ ==
            if c == 61 & cnext == 61
                emit_token 55 pos 2
                pos = pos + 2
                continue
            \\ !=
            if c == 33 & cnext == 61
                emit_token 56 pos 2
                pos = pos + 2
                continue
            \\ <=
            if c == 60 & cnext == 61
                emit_token 59 pos 2
                pos = pos + 2
                continue
            \\ >=
            if c == 62 & cnext == 61
                emit_token 60 pos 2
                pos = pos + 2
                continue
            \\ <<
            if c == 60 & cnext == 60
                emit_token 64 pos 2
                pos = pos + 2
                continue
            \\ >>
            if c == 62 & cnext == 62
                emit_token 65 pos 2
                pos = pos + 2
                continue
            \\ -> removed (compositions use :, not ->)

        \\ ---- Multi-byte UTF-8 tokens (Unicode arrows and math symbols) ----
        \\ → (E2 86 92) = TOK_LOAD (36)
        \\ ← (E2 86 90) = TOK_STORE (37)
        \\ ↑ (E2 86 91) = TOK_REG_READ (38)
        \\ ↓ (E2 86 93) = TOK_REG_WRITE (39)
        \\ Σ (CE A3)    = TOK_SUM (75)
        \\ △ (E2 96 B3) = TOK_MAX (76)
        \\ ▽ (E2 96 BD) = TOK_MIN (77)
        \\ √ (E2 88 9A) = TOK_SQRT (79)
        \\ ≅ (E2 89 85) = TOK_SIN (80)
        \\ ≡ (E2 89 A1) = TOK_COS (81)
        if c == 0xE2
            if pos + 2 < src_len
                b1 = src [ pos + 1 ]    \\ BYTE LOAD
                b2 = src [ pos + 2 ]    \\ BYTE LOAD
                \\ → E2 86 92
                if b1 == 0x86 & b2 == 0x92
                    emit_token 36 pos 3
                    pos = pos + 3
                    continue
                \\ ← E2 86 90
                if b1 == 0x86 & b2 == 0x90
                    emit_token 37 pos 3
                    pos = pos + 3
                    continue
                \\ ↑ E2 86 91
                if b1 == 0x86 & b2 == 0x91
                    emit_token 38 pos 3
                    pos = pos + 3
                    continue
                \\ ↓ E2 86 93
                if b1 == 0x86 & b2 == 0x93
                    emit_token 39 pos 3
                    pos = pos + 3
                    continue
                \\ △ E2 96 B3
                if b1 == 0x96 & b2 == 0xB3
                    emit_token 76 pos 3
                    pos = pos + 3
                    continue
                \\ ▽ E2 96 BD
                if b1 == 0x96 & b2 == 0xBD
                    emit_token 77 pos 3
                    pos = pos + 3
                    continue
                \\ √ E2 88 9A
                if b1 == 0x88 & b2 == 0x9A
                    emit_token 79 pos 3
                    pos = pos + 3
                    continue
                \\ ≅ E2 89 85
                if b1 == 0x89 & b2 == 0x85
                    emit_token 80 pos 3
                    pos = pos + 3
                    continue
                \\ ≡ E2 89 A1
                if b1 == 0x89 & b2 == 0xA1
                    emit_token 81 pos 3
                    pos = pos + 3
                    continue
        \\ Σ (CE A3) — 2-byte UTF-8
        if c == 0xCE
            if pos + 1 < src_len
                b1 = src [ pos + 1 ]    \\ BYTE LOAD
                if b1 == 0xA3
                    emit_token 75 pos 2
                    pos = pos + 2
                    continue

        \\ ---- Single-character operators and punctuation ----
        if c == 43                   \\ '+'
            emit_token 50 pos 1
            pos = pos + 1
            continue
        if c == 45                   \\ '-'  (not followed by digit or '>', already checked)
            emit_token 51 pos 1
            pos = pos + 1
            continue
        if c == 42                   \\ '*'
            emit_token 52 pos 1
            pos = pos + 1
            continue
        if c == 47                   \\ '/'
            emit_token 53 pos 1
            pos = pos + 1
            continue
        if c == 61                   \\ '='  (single, not '==')
            emit_token 54 pos 1
            pos = pos + 1
            continue
        if c == 60                   \\ '<'  (single, not '<<' or '<=')
            emit_token 57 pos 1
            pos = pos + 1
            continue
        if c == 62                   \\ '>'  (single, not '>>' or '>=')
            emit_token 58 pos 1
            pos = pos + 1
            continue
        if c == 38                   \\ '&'
            emit_token 61 pos 1
            pos = pos + 1
            continue
        if c == 124                  \\ '|'
            emit_token 62 pos 1
            pos = pos + 1
            continue
        if c == 94                   \\ '^'
            emit_token 63 pos 1
            pos = pos + 1
            continue
        if c == 91                   \\ '['
            emit_token 67 pos 1
            pos = pos + 1
            continue
        if c == 93                   \\ ']'
            emit_token 68 pos 1
            pos = pos + 1
            continue
        if c == 40                   \\ '('
            emit_token 69 pos 1
            pos = pos + 1
            continue
        if c == 41                   \\ ')'
            emit_token 70 pos 1
            pos = pos + 1
            continue
        if c == 44                   \\ ','
            emit_token 71 pos 1
            pos = pos + 1
            continue
        if c == 58                   \\ ':'
            emit_token 72 pos 1
            pos = pos + 1
            continue
        if c == 46                   \\ '.'
            emit_token 73 pos 1
            pos = pos + 1
            continue
        if c == 64                   \\ '@'
            emit_token 74 pos 1
            pos = pos + 1
            continue

        \\ ---- Unknown character: skip ----
        \\ In a production compiler this would emit an error token.
        \\ For now, advance past it silently.
        pos = pos + 1

    \\ ---- Emit final EOF token ----
    emit_token 0 pos 0               \\ TOK_EOF

\\ ============================================================================
\\ Entry point
\\ ============================================================================
\\ Called by the compiler driver after mmap'ing the source file.
\\ src:     pointer to file contents (byte buffer)
\\ src_len: file size in bytes
\\
\\ After lex returns:
\\   token_count contains the number of tokens emitted
\\   tokens[0..token_count*3-1] contains the token triples

lithos_lex src src_len :
    lex src src_len
