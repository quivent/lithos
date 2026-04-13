// lithos-lexer.s — ARM64 assembly lexer for Lithos .ls source files
//
// Tokenizes a source buffer into a flat array of (type, offset, length)
// triples, each field a u32. Designed to be appended to lithos-bootstrap.s
// or linked alongside it.
//
// Interface (DTC Forth words):
//   LITHOS-LEX  ( src len -- )         tokenize buffer into ls_token_buf
//   LEX-TOKENS  ( -- addr )            address of token triple array
//   LEX-COUNT   ( -- n )               number of tokens emitted
//   LEX-TOKEN@  ( idx -- type off len ) fetch idx-th token triple
//
// Register map (inherited from lithos-bootstrap.s):
//   X26=IP  X25=W  X24=DSP  X23=RSP  X22=TOS  X20=HERE  X21=BASE
//
// Build: append to lithos-bootstrap.s before the .end directive,
//        updating last_entry and entry_pad's link accordingly.
//
// Token encoding: 3 consecutive u32 values per token
//   [0] type    — token type enum (TOK_* constants)
//   [1] offset  — byte offset in source buffer
//   [2] length  — byte length of token text (or indent count for TOK_INDENT)

// ============================================================
// DTC macro NEXT — defined here only if the shared header in
// build.sh (DTC_MACROS_DEFINED) hasn't already provided it.
// ============================================================
.ifndef DTC_MACROS_DEFINED
.set DTC_MACROS_DEFINED, 1
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.endif

// ============================================================
// Token type constants
// ============================================================

.equ TOK_EOF,        0
.equ TOK_NEWLINE,    1
.equ TOK_INDENT,     2
.equ TOK_INT,        3
.equ TOK_FLOAT,      4
.equ TOK_IDENT,      5

// Keywords
.equ TOK_KERNEL,    11
.equ TOK_PARAM,     12
.equ TOK_IF,        13
.equ TOK_ELSE,      14
.equ TOK_ELIF,      15
.equ TOK_FOR,       16
.equ TOK_ENDFOR,    17
.equ TOK_EACH,      18
.equ TOK_STRIDE,    19
.equ TOK_WHILE,     20
.equ TOK_RETURN,    21
.equ TOK_CONST,     22
.equ TOK_VAR,       23
.equ TOK_BUF,       24
.equ TOK_WEIGHT,    25
.equ TOK_LAYER,     26
.equ TOK_BIND,      27
.equ TOK_RUNTIME,   28
.equ TOK_TEMPLATE,  29
.equ TOK_PROJECT,   30
.equ TOK_SHARED,    31
.equ TOK_BARRIER,   32
.equ TOK_LABEL,     33
.equ TOK_EXIT_KW,   34
.equ TOK_HOST,      35
.equ TOK_TRAP,      89      // trap (syscall)

// Memory / register arrows (UTF-8 multi-byte)
.equ TOK_LOAD,      36      // → E2 86 92
.equ TOK_STORE,     37      // ← E2 86 90
.equ TOK_REG_READ,  38      // ↑ E2 86 91
.equ TOK_REG_WRITE, 39      // ↓ E2 86 93

// Type keywords
.equ TOK_F32,       40
.equ TOK_U32,       41
.equ TOK_S32,       42
.equ TOK_F16,       43
.equ TOK_PTR,       44
.equ TOK_VOID,      45

// Operators
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_EQ,        54
.equ TOK_EQEQ,     55
.equ TOK_NEQ,       56
.equ TOK_LT,        57
.equ TOK_GT,        58
.equ TOK_LTE,       59
.equ TOK_GTE,       60
.equ TOK_AMP,       61
.equ TOK_PIPE,      62
.equ TOK_CARET,     63
.equ TOK_SHL,       64
.equ TOK_SHR,       65

// Brackets / punctuation
.equ TOK_LBRACK,    67
.equ TOK_RBRACK,    68
.equ TOK_LPAREN,    69
.equ TOK_RPAREN,    70
.equ TOK_COMMA,     71
.equ TOK_COLON,     72
.equ TOK_DOT,       73
.equ TOK_AT,        74

// Reduction / math unicode
.equ TOK_SUM,       75      // Σ CE A3
.equ TOK_MAX,       76      // △ E2 96 B3
.equ TOK_MIN,       77      // ▽ E2 96 BD
.equ TOK_INDEX,     78      // #
.equ TOK_SQRT,      79      // √ E2 88 9A
.equ TOK_SIN,       80      // ≅ E2 89 85
.equ TOK_COS,       81      // ≡ E2 89 A1

.equ LEX_MAX_TOKENS, 87381  // 262143 / 3

// ============================================================
// Data section
// ============================================================
// Shared token buffer + count live in ls-shared.s
.extern ls_token_buf
.extern ls_token_count

.data
.align 3

// Keyword table: packed entries of (len:u8, tok_type:u8, chars:len bytes)
// Linear scan. Sentinel: len=0.
.align 2
lex_kw_table:
    // 2-char
    .byte 2, TOK_IF
    .ascii "if"
    // 3-char
    .byte 3, TOK_FOR
    .ascii "for"
    .byte 3, TOK_VAR
    .ascii "var"
    .byte 3, TOK_BUF
    .ascii "buf"
    .byte 3, TOK_F32
    .ascii "f32"
    .byte 3, TOK_U32
    .ascii "u32"
    .byte 3, TOK_S32
    .ascii "s32"
    .byte 3, TOK_F16
    .ascii "f16"
    .byte 3, TOK_PTR
    .ascii "ptr"
    // 4-char
    .byte 4, TOK_EACH
    .ascii "each"
    .byte 4, TOK_ELSE
    .ascii "else"
    .byte 4, TOK_ELIF
    .ascii "elif"
    .byte 4, TOK_VOID
    .ascii "void"
    .byte 4, TOK_BIND
    .ascii "bind"
    .byte 4, TOK_EXIT_KW
    .ascii "exit"
    .byte 4, TOK_HOST
    .ascii "host"
    .byte 4, TOK_TRAP
    .ascii "trap"
    // 5-char
    .byte 5, TOK_PARAM
    .ascii "param"
    .byte 5, TOK_WHILE
    .ascii "while"
    .byte 5, TOK_CONST
    .ascii "const"
    .byte 5, TOK_LAYER
    .ascii "layer"
    .byte 5, TOK_LABEL
    .ascii "label"
    // 6-char
    .byte 6, TOK_KERNEL
    .ascii "kernel"
    .byte 6, TOK_STRIDE
    .ascii "stride"
    .byte 6, TOK_RETURN
    .ascii "return"
    .byte 6, TOK_WEIGHT
    .ascii "weight"
    .byte 6, TOK_SHARED
    .ascii "shared"
    .byte 6, TOK_ENDFOR
    .ascii "endfor"
    // 7-char
    .byte 7, TOK_RUNTIME
    .ascii "runtime"
    .byte 7, TOK_BARRIER
    .ascii "barrier"
    .byte 7, TOK_PROJECT
    .ascii "project"
    // 8-char
    .byte 8, TOK_TEMPLATE
    .ascii "template"
    // sentinel
    .byte 0

// Single-char operator lookup: 95 bytes for ASCII 32..126
// Index = (char - 32). Value = token type, 0 = not an operator.
.align 2
op1_table:
    // 32 sp  33 !   34 "   35 #        36 $   37 %   38 &        39 '
    .byte 0,  0,     0,     TOK_INDEX,  0,     0,     TOK_AMP,    0
    // 40 (         41 )          42 *        43 +         44 ,         45 -         46 .        47 /
    .byte TOK_LPAREN, TOK_RPAREN, TOK_STAR,  TOK_PLUS,   TOK_COMMA,  TOK_MINUS,   TOK_DOT,   TOK_SLASH
    // 48-57: '0'-'9' — all 0
    .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // 58 :          59 ;  60 <       61 =       62 >       63 ?
    .byte TOK_COLON, 0,   TOK_LT,   TOK_EQ,   TOK_GT,    0
    // 64 @        65-90 A-Z: all 0
    .byte TOK_AT
    .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // A-J
    .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // K-T
    .byte 0, 0, 0, 0, 0, 0               // U-Z
    // 91 [           92 \  93 ]           94 ^          95 _
    .byte TOK_LBRACK, 0,   TOK_RBRACK,   TOK_CARET,   0
    // 96 `   97-122 a-z: all 0
    .byte 0
    .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // a-j
    .byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // k-t
    .byte 0, 0, 0, 0, 0, 0               // u-z
    // 123 {  124 |         125 }  126 ~
    .byte 0,  TOK_PIPE,    0,     0

// ============================================================
// BSS — token output buffer lives in ls-shared.s (ls_token_buf)
// ============================================================

// ============================================================
// Text section — lexer machine code
// ============================================================
.text
.align 4

// ============================================================
// emit_tok: append one token to ls_token_buf
//   w0 = type, w1 = offset, w2 = length
//   Clobbers: x3, x4, x5, x6
// ============================================================
emit_tok:
    adrp    x3, ls_token_count
    add     x3, x3, :lo12:ls_token_count
    ldr     x4, [x3]
    mov     x5, #12
    madd    x5, x4, x5, xzr
    adrp    x6, ls_token_buf
    add     x6, x6, :lo12:ls_token_buf
    add     x6, x6, x5
    str     w0, [x6]
    str     w1, [x6, #4]
    str     w2, [x6, #8]
    add     x4, x4, #1
    str     x4, [x3]
    ret

// ============================================================
// match_kw: check identifier against keyword table
//   x0 = src base, w1 = offset, w2 = length
//   Returns: w0 = token type (TOK_IDENT=5 if no match)
//   Clobbers: x3-x11
// ============================================================
match_kw:
    adrp    x3, lex_kw_table
    add     x3, x3, :lo12:lex_kw_table
    add     x4, x0, x1, uxtw       // x4 = &src[offset]
.Llex_kw_loop:
    ldrb    w5, [x3]               // kw_len
    cbz     w5, .Llex_kw_miss
    ldrb    w6, [x3, #1]           // kw_tok
    cmp     w5, w2
    b.ne    .Llex_kw_skip
    // byte-by-byte compare
    add     x7, x3, #2             // kw text
    mov     x8, x4                 // src text
    mov     w9, w5
.Llex_kw_cmp:
    ldrb    w10, [x7], #1
    ldrb    w11, [x8], #1
    cmp     w10, w11
    b.ne    .Llex_kw_skip
    subs    w9, w9, #1
    b.ne    .Llex_kw_cmp
    mov     w0, w6                  // matched
    ret
.Llex_kw_skip:
    add     x3, x3, #2
    add     x3, x3, x5             // advance past entry
    b       .Llex_kw_loop
.Llex_kw_miss:
    mov     w0, #TOK_IDENT
    ret

// ============================================================
// scan_ident_asm: advance past identifier characters
//   x0 = src, w1 = pos, w2 = end
//   Returns: w1 = new pos
//   Ident chars: [A-Za-z0-9_$]
//   Clobbers: w3
// ============================================================
scan_ident_asm:
.Lsi_loop:
    cmp     w1, w2
    b.ge    .Lsi_done
    ldrb    w3, [x0, x1]
    // a-z
    sub     w4, w3, #97
    cmp     w4, #25
    b.ls    .Lsi_next
    // A-Z
    sub     w4, w3, #65
    cmp     w4, #25
    b.ls    .Lsi_next
    // 0-9
    sub     w4, w3, #48
    cmp     w4, #9
    b.ls    .Lsi_next
    // _ or $
    cmp     w3, #95
    b.eq    .Lsi_next
    cmp     w3, #36
    b.ne    .Lsi_done
.Lsi_next:
    add     w1, w1, #1
    b       .Lsi_loop
.Lsi_done:
    ret

// ============================================================
// scan_number_asm: advance past number literal
//   x0 = src, w1 = pos, w2 = end
//   Returns: w1 = new pos, w3 = has_dot (0 or 1)
//   Handles decimal, hex (0x...), and float (with '.')
//   Clobbers: w3-w5
// ============================================================
scan_number_asm:
    mov     w3, #0                  // has_dot = 0
    ldrb    w4, [x0, x1]
    cmp     w4, #48                 // '0'
    b.ne    .Lsn_dec
    add     w5, w1, #1
    cmp     w5, w2
    b.ge    .Lsn_dec
    ldrb    w4, [x0, x5]
    cmp     w4, #120                // 'x'
    b.eq    .Lsn_hex_start
    cmp     w4, #88                 // 'X'
    b.ne    .Lsn_dec
.Lsn_hex_start:
    add     w1, w1, #2
.Lsn_hex_loop:
    cmp     w1, w2
    b.ge    .Lsn_done
    ldrb    w4, [x0, x1]
    sub     w5, w4, #48
    cmp     w5, #9
    b.ls    .Lsn_hex_next
    sub     w5, w4, #65
    cmp     w5, #5
    b.ls    .Lsn_hex_next
    sub     w5, w4, #97
    cmp     w5, #5
    b.ls    .Lsn_hex_next
    b       .Lsn_done
.Lsn_hex_next:
    add     w1, w1, #1
    b       .Lsn_hex_loop

.Lsn_dec:
.Lsn_dec_loop:
    cmp     w1, w2
    b.ge    .Lsn_done
    ldrb    w4, [x0, x1]
    sub     w5, w4, #48
    cmp     w5, #9
    b.ls    .Lsn_dec_next
    cmp     w4, #46                 // '.'
    b.ne    .Lsn_done
    mov     w3, #1
.Lsn_dec_next:
    add     w1, w1, #1
    b       .Lsn_dec_loop
.Lsn_done:
    ret

// ============================================================
// do_lithos_lex: main lexer engine
//   x0 = src buffer, x1 = src length
//   Writes tokens to ls_token_buf[], count to ls_token_count.
//
//   Internal register usage (callee-saved, stacked on entry):
//     x19 = src base
//     x27 = pos
//     x28 = end (src_len)
//     x29 = line_start flag
// ============================================================
.align 4
do_lithos_lex:
    stp     x19, x20, [sp, #-80]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x27, x28, [sp, #48]
    stp     x29, x30, [sp, #64]

    mov     x19, x0
    mov     w28, w1
    mov     w27, #0                 // pos = 0
    mov     w29, #1                 // line_start = 1

    // Reset count
    adrp    x0, ls_token_count
    add     x0, x0, :lo12:ls_token_count
    str     xzr, [x0]

// ---- Main loop ----
.Lml_top:
    cmp     w27, w28
    b.ge    .Lml_eof

    ldrb    w15, [x19, x27]

    // ---- Start of line: indentation ----
    cbz     w29, .Lml_not_sol
    mov     w16, #0                 // indent count
.Lml_indent:
    cmp     w27, w28
    b.ge    .Lml_indent_emit
    ldrb    w15, [x19, x27]
    cmp     w15, #32
    b.ne    1f
    add     w16, w16, #1
    add     w27, w27, #1
    b       .Lml_indent
1:  cmp     w15, #9
    b.ne    .Lml_indent_emit
    add     w16, w16, #4
    add     w27, w27, #1
    b       .Lml_indent
.Lml_indent_emit:
    mov     w0, #TOK_INDENT
    mov     w1, w27                 // offset = first non-ws byte
    mov     w2, w16                 // length = indent count
    bl      emit_tok
    mov     w29, #0
    cmp     w27, w28
    b.ge    .Lml_eof
    ldrb    w15, [x19, x27]

.Lml_not_sol:

    // ---- LF (10) ----
    cmp     w15, #10
    b.ne    .Lml_not_lf
    mov     w0, #TOK_NEWLINE
    mov     w1, w27
    mov     w2, #1
    bl      emit_tok
    add     w27, w27, #1
    mov     w29, #1
    // skip trailing CR
    cmp     w27, w28
    b.ge    .Lml_top
    ldrb    w15, [x19, x27]
    cmp     w15, #13
    b.ne    .Lml_top
    add     w27, w27, #1
    b       .Lml_top
.Lml_not_lf:

    // ---- CR (13) ----
    cmp     w15, #13
    b.ne    .Lml_not_cr
    mov     w0, #TOK_NEWLINE
    mov     w1, w27
    mov     w2, #1
    bl      emit_tok
    add     w27, w27, #1
    mov     w29, #1
    cmp     w27, w28
    b.ge    .Lml_top
    ldrb    w15, [x19, x27]
    cmp     w15, #10
    b.ne    .Lml_top
    add     w27, w27, #1
    b       .Lml_top
.Lml_not_cr:

    // ---- Inline whitespace ----
    cmp     w15, #32
    b.eq    .Lml_skip1
    cmp     w15, #9
    b.eq    .Lml_skip1
    b       .Lml_not_ws
.Lml_skip1:
    add     w27, w27, #1
    b       .Lml_top
.Lml_not_ws:

    // ---- Comment: \\ to EOL ----
    cmp     w15, #92
    b.ne    .Lml_not_comment
    add     w3, w27, #1
    cmp     w3, w28
    b.ge    .Lml_not_comment
    ldrb    w4, [x19, x3]
    cmp     w4, #92
    b.ne    .Lml_not_comment
    add     w27, w27, #2
.Lml_comment_scan:
    cmp     w27, w28
    b.ge    .Lml_top
    ldrb    w4, [x19, x27]
    cmp     w4, #10
    b.eq    .Lml_top
    cmp     w4, #13
    b.eq    .Lml_top
    add     w27, w27, #1
    b       .Lml_comment_scan
.Lml_not_comment:

    // ---- 3-byte UTF-8: E2 xx xx ----
    cmp     w15, #0xE2
    b.ne    .Lml_not_e2
    add     w3, w27, #2
    cmp     w3, w28
    b.ge    .Lml_not_e2
    add     x5, x19, x27
    ldrb    w4, [x5, #1]           // b1
    ldrb    w6, [x5, #2]           // b2

    cmp     w4, #0x86
    b.ne    .Le2_not_86
    // E2 86 9x — arrows
    cmp     w6, #0x90
    b.eq    .Le2_store
    cmp     w6, #0x91
    b.eq    .Le2_regrd
    cmp     w6, #0x92
    b.eq    .Le2_load
    cmp     w6, #0x93
    b.eq    .Le2_regwr
    b       .Lml_not_e2
.Le2_load:
    mov     w0, #TOK_LOAD
    b       .Le2_emit3
.Le2_store:
    mov     w0, #TOK_STORE
    b       .Le2_emit3
.Le2_regrd:
    mov     w0, #TOK_REG_READ
    b       .Le2_emit3
.Le2_regwr:
    mov     w0, #TOK_REG_WRITE
    b       .Le2_emit3

.Le2_not_86:
    cmp     w4, #0x96
    b.ne    .Le2_not_96
    cmp     w6, #0xB3
    b.eq    .Le2_max
    cmp     w6, #0xBD
    b.eq    .Le2_min
    b       .Lml_not_e2
.Le2_max:
    mov     w0, #TOK_MAX
    b       .Le2_emit3
.Le2_min:
    mov     w0, #TOK_MIN
    b       .Le2_emit3

.Le2_not_96:
    cmp     w4, #0x88
    b.ne    .Le2_not_88
    cmp     w6, #0x9A
    b.ne    .Lml_not_e2
    mov     w0, #TOK_SQRT
    b       .Le2_emit3
.Le2_not_88:
    cmp     w4, #0x89
    b.ne    .Lml_not_e2
    cmp     w6, #0x85
    b.eq    .Le2_sin
    cmp     w6, #0xA1
    b.ne    .Lml_not_e2
    mov     w0, #TOK_COS
    b       .Le2_emit3
.Le2_sin:
    mov     w0, #TOK_SIN

.Le2_emit3:
    mov     w1, w27
    mov     w2, #3
    bl      emit_tok
    add     w27, w27, #3
    b       .Lml_top

.Lml_not_e2:
    // ---- 2-byte UTF-8: Σ = CE A3 ----
    cmp     w15, #0xCE
    b.ne    .Lml_not_ce
    add     w3, w27, #1
    cmp     w3, w28
    b.ge    .Lml_not_ce
    add     x5, x19, x27
    ldrb    w4, [x5, #1]
    cmp     w4, #0xA3
    b.ne    .Lml_not_ce
    mov     w0, #TOK_SUM
    mov     w1, w27
    mov     w2, #2
    bl      emit_tok
    add     w27, w27, #2
    b       .Lml_top
.Lml_not_ce:

    // ---- Number literal: starts with digit ----
    sub     w3, w15, #48
    cmp     w3, #9
    b.hi    .Lml_not_digit
    mov     w16, w27                // start
    mov     x0, x19
    mov     w1, w27
    mov     w2, w28
    bl      scan_number_asm
    mov     w27, w1
    mov     w7, w3                  // has_dot
    sub     w2, w27, w16            // length
    cbz     w7, 1f
    mov     w0, #TOK_FLOAT
    b       2f
1:  mov     w0, #TOK_INT
2:  mov     w1, w16
    bl      emit_tok
    b       .Lml_top
.Lml_not_digit:

    // ---- Negative number: '-' + digit ----
    cmp     w15, #45
    b.ne    .Lml_not_neg
    add     w3, w27, #1
    cmp     w3, w28
    b.ge    .Lml_not_neg
    ldrb    w4, [x19, x3]
    sub     w5, w4, #48
    cmp     w5, #9
    b.hi    .Lml_not_neg
    mov     w16, w27                // start (includes '-')
    add     w27, w27, #1
    mov     x0, x19
    mov     w1, w27
    mov     w2, w28
    bl      scan_number_asm
    mov     w27, w1
    mov     w7, w3
    sub     w2, w27, w16
    cbz     w7, 1f
    mov     w0, #TOK_FLOAT
    b       2f
1:  mov     w0, #TOK_INT
2:  mov     w1, w16
    bl      emit_tok
    b       .Lml_top
.Lml_not_neg:

    // ---- Identifier / keyword ----
    // starts with [A-Za-z_$]
    sub     w3, w15, #97
    cmp     w3, #25
    b.ls    .Lml_do_ident
    sub     w3, w15, #65
    cmp     w3, #25
    b.ls    .Lml_do_ident
    cmp     w15, #95
    b.eq    .Lml_do_ident
    cmp     w15, #36
    b.ne    .Lml_not_ident
.Lml_do_ident:
    mov     w16, w27                // start
    mov     x0, x19
    mov     w1, w27
    mov     w2, w28
    bl      scan_ident_asm
    mov     w27, w1                 // new pos
    sub     w2, w27, w16            // length
    // Save length for emit
    mov     w17, w2
    mov     x0, x19
    mov     w1, w16
    // w2 = length
    bl      match_kw                // returns w0 = tok type
    mov     w1, w16
    mov     w2, w17
    bl      emit_tok
    b       .Lml_top
.Lml_not_ident:

    // ---- Two-char operators ----
    add     w3, w27, #1
    cmp     w3, w28
    b.ge    .Lml_not_2ch
    ldrb    w4, [x19, x3]          // next char

    // ==
    cmp     w15, #61
    b.ne    .Ltc_not_eq
    cmp     w4, #61
    b.ne    .Lml_not_2ch            // single '=' falls through to 1-char
    mov     w0, #TOK_EQEQ
    b       .Ltc_emit2
.Ltc_not_eq:
    // !=
    cmp     w15, #33
    b.ne    .Ltc_not_neq
    cmp     w4, #61
    b.ne    .Lml_skip_unknown       // bare '!' not a valid token
    mov     w0, #TOK_NEQ
    b       .Ltc_emit2
.Ltc_not_neq:
    // < variants: <= <<
    cmp     w15, #60
    b.ne    .Ltc_not_lt
    cmp     w4, #61
    b.ne    1f
    mov     w0, #TOK_LTE
    b       .Ltc_emit2
1:  cmp     w4, #60
    b.ne    .Lml_not_2ch
    mov     w0, #TOK_SHL
    b       .Ltc_emit2
.Ltc_not_lt:
    // > variants: >= >>
    cmp     w15, #62
    b.ne    .Lml_not_2ch
    cmp     w4, #61
    b.ne    1f
    mov     w0, #TOK_GTE
    b       .Ltc_emit2
1:  cmp     w4, #62
    b.ne    .Lml_not_2ch
    mov     w0, #TOK_SHR

.Ltc_emit2:
    mov     w1, w27
    mov     w2, #2
    bl      emit_tok
    add     w27, w27, #2
    b       .Lml_top
.Lml_not_2ch:

    // ---- Single-char operators (table lookup) ----
    cmp     w15, #32
    b.lt    .Lml_skip_unknown
    cmp     w15, #126
    b.gt    .Lml_skip_unknown
    sub     w3, w15, #32
    adrp    x5, op1_table
    add     x5, x5, :lo12:op1_table
    ldrb    w4, [x5, x3]
    cbz     w4, .Lml_skip_unknown
    mov     w0, w4
    mov     w1, w27
    mov     w2, #1
    bl      emit_tok
    add     w27, w27, #1
    b       .Lml_top

.Lml_skip_unknown:
    add     w27, w27, #1
    b       .Lml_top

.Lml_eof:
    mov     w0, #TOK_EOF
    mov     w1, w27
    mov     w2, #0
    bl      emit_tok

    ldp     x29, x30, [sp, #64]
    ldp     x27, x28, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #80
    ret


// ============================================================
// DTC word: LITHOS-LEX ( src len -- )
// ============================================================
.align 4
code_LITHOS_LEX:
    mov     x1, x22                 // len = TOS
    ldr     x0, [x24], #8          // src = NOS
    ldr     x22, [x24], #8         // pop: restore TOS

    // Save DTC state
    stp     x26, x25, [sp, #-48]!
    stp     x24, x23, [sp, #16]
    stp     x22, x20, [sp, #32]

    bl      do_lithos_lex

    // Restore DTC state
    ldp     x22, x20, [sp, #32]
    ldp     x24, x23, [sp, #16]
    ldp     x26, x25, [sp], #48
    NEXT

// ============================================================
// DTC word: LEX-TOKENS ( -- addr )
// ============================================================
.align 4
code_LEX_TOKENS:
    str     x22, [x24, #-8]!
    adrp    x22, ls_token_buf
    add     x22, x22, :lo12:ls_token_buf
    NEXT

// ============================================================
// DTC word: LEX-COUNT ( -- n )
// ============================================================
.align 4
code_LEX_COUNT:
    str     x22, [x24, #-8]!
    adrp    x0, ls_token_count
    add     x0, x0, :lo12:ls_token_count
    ldr     x22, [x0]
    NEXT

// ============================================================
// DTC word: LEX-TOKEN@ ( idx -- type offset len )
//   Fetch the idx-th token triple. idx consumed, 3 values returned.
// ============================================================
.align 4
code_LEX_TOKEN_FETCH:
    mov     x0, #12
    mul     x0, x22, x0
    adrp    x1, ls_token_buf
    add     x1, x1, :lo12:ls_token_buf
    add     x1, x1, x0
    ldr     w2, [x1]               // type (u32)
    ldr     w3, [x1, #4]           // offset (u32)
    ldr     w4, [x1, #8]           // length (u32)
    // Result: ( -- type offset len )
    // type goes deepest on stack, len becomes TOS
    mov     x22, x2                 // overwrite TOS (idx) with type
    str     x22, [x24, #-8]!       // push type
    mov     x22, x3
    str     x22, [x24, #-8]!       // push offset
    mov     x22, x4                 // TOS = len
    NEXT

// ============================================================
// Dictionary entries
//
// To integrate into lithos-bootstrap.s:
//   1. Replace the "last_entry:" label to point to entry_lex_token_fetch
//   2. Set entry_lithos_lex's link to the current last entry (entry_pad)
// ============================================================

.data
.align 3

// First lexer entry — links to bootstrap chain tail
entry_lithos_lex:
    .quad   entry_ls_loop_peek
    .byte   0
    .byte   10
    .ascii  "lithos-lex"
    .align  3
    .quad   code_LITHOS_LEX

entry_lex_tokens:
    .quad   entry_lithos_lex
    .byte   0
    .byte   10
    .ascii  "lex-tokens"
    .align  3
    .quad   code_LEX_TOKENS

entry_lex_count:
    .quad   entry_lex_tokens
    .byte   0
    .byte   9
    .ascii  "lex-count"
    .align  3
    .quad   code_LEX_COUNT

.globl entry_lex_token_fetch
entry_lex_token_fetch:
    .quad   entry_lex_count
    .byte   0
    .byte   10
    .ascii  "lex-token@"
    .align  3
    .quad   code_LEX_TOKEN_FETCH
