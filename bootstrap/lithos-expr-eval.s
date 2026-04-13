// lithos-expr-eval.s — Expression parser/evaluator for Lithos bootstrap
//
// Parses infix expressions from the token stream with correct operator
// precedence and emits ARM64 instructions that compute the result into
// a destination register. Returns the register number holding the result.
//
// Linked alongside lithos-parser.s — shares its symbol table, token
// cursor, register allocator, and emit helpers.
//
// Threading: Native ARM64 subroutine calls (bl/ret). The DTC wrapper
// at the bottom (code_PARSE_EXPR) bridges from the DTC bootstrap.
//
// Register conventions (inherited from lithos-parser.s):
//   X19 = TOKP   — pointer to current token triple
//   X27 = TOKEND — end of token buffer
//   X28 = SRC    — pointer to source buffer
//   X26 = IP     — DTC instruction pointer
//   X24 = DSP    — data stack pointer
//   X22 = TOS    — top of data stack
//   X20 = HERE   — dictionary pointer
//
// Token triple layout: [+0] u32 type, [+4] u32 offset, [+8] u32 length
// Token stride: 12 bytes
//
// Emitted code uses X9-X15 as the expression register pool.
// Result of parse_expr is in w0 = register number holding the value.

// ============================================================
// Token type constants (must match lithos-parser.s / lithos-lexer.s)
// ============================================================
.equ TOK_EOF,       0
.equ TOK_NEWLINE,   1
.equ TOK_INDENT,    2
.equ TOK_INT,       3
.equ TOK_FLOAT,     4
.equ TOK_IDENT,     5
.equ TOK_STRING,    6
.equ TOK_LOAD,      36
.equ TOK_REG_READ,  38
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_AMP,       61
.equ TOK_PIPE,      62
.equ TOK_CARET,     63
.equ TOK_SHL,       64
.equ TOK_SHR,       65
.equ TOK_LPAREN,    69
.equ TOK_RPAREN,    70
.equ TOK_SQRT,      79

.equ TOK_STRIDE_SZ, 12

// Symbol table entry layout (matches lithos-parser.s)
.equ SYM_SIZE,      24
.equ SYM_NAME_OFF,  0
.equ SYM_NAME_LEN,  4
.equ SYM_KIND,      8
.equ SYM_REG,       12

.equ KIND_LOCAL_REG, 0

// ORRIMM macro — OR a 32-bit immediate into Wd via a tmp register
.macro ORRIMM Wd, imm, Wtmp
    mov     \Wtmp, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wtmp, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
    orr     \Wd, \Wd, \Wtmp
.endm

// ============================================================
// External symbols from lithos-parser.s
// ============================================================
.extern ls_sym_count
.extern ls_sym_table
.extern next_reg
.extern emit_ptr

// External helpers from lithos-parser.s
.extern alloc_reg
.extern free_reg
.extern emit32
.extern emit_mov_imm16
.extern emit_movk_imm16
.extern emit_mov_imm64
.extern emit_add_reg
.extern emit_sub_reg
.extern emit_mul_reg
.extern emit_sdiv_reg
.extern emit_and_reg
.extern emit_orr_reg
.extern emit_eor_reg
.extern emit_lsl_reg
.extern emit_lsr_reg
.extern emit_ldr_reg
.extern emit_ldrb_reg
.extern emit_ldrh_reg
.extern emit_ldr_w_zero
.extern emit_ldr_x_zero
.extern parse_error

// DTC macros (for the DTC wrapper only)
.ifndef DTC_MACROS_DEFINED
.set DTC_MACROS_DEFINED, 1
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.macro PUSH reg
    str     x22, [x24, #-8]!
    mov     x22, \reg
.endm
.macro POP reg
    mov     \reg, x22
    ldr     x22, [x24], #8
.endm
.endif

// ============================================================
// .text — Expression evaluator
// ============================================================
.text
.align 4

// ============================================================
// Helper: peek current token type into w0
// ============================================================
ee_tok_type:
    ldr     w0, [x19]
    ret

// ============================================================
// Helper: advance token pointer by one triple
// ============================================================
ee_advance:
    add     x19, x19, #TOK_STRIDE_SZ
    ret

// ============================================================
// Helper: get current token text pointer in x0, length in w1
// ============================================================
ee_tok_text:
    ldr     w1, [x19, #8]          // length
    ldr     w0, [x19, #4]          // offset
    add     x0, x28, x0            // src + offset
    ret

// ============================================================
// parse_number — parse decimal integer from token text
//   Reads current token (must be TOK_INT). Returns value in x0.
//   Does NOT advance the token pointer.
// ============================================================
parse_number:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!

    ldr     w1, [x19, #8]          // token length
    ldr     w2, [x19, #4]          // token offset
    add     x2, x28, x2            // text pointer

    // Check for hex prefix 0x / 0X
    cmp     w1, #2
    b.lt    .Lnum_decimal
    ldrb    w3, [x2]
    cmp     w3, #'0'
    b.ne    .Lnum_decimal
    ldrb    w3, [x2, #1]
    cmp     w3, #'x'
    b.eq    .Lnum_hex
    cmp     w3, #'X'
    b.eq    .Lnum_hex

.Lnum_decimal:
    mov     x0, #0
    mov     w3, #0                  // index
.Lnum_dec_loop:
    cmp     w3, w1
    b.ge    .Lnum_done
    ldrb    w4, [x2, x3]
    sub     w4, w4, #'0'
    // x0 = x0 * 10 + digit
    mov     x5, #10
    mul     x0, x0, x5
    and     x4, x4, #0xFF
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lnum_dec_loop

.Lnum_hex:
    // Skip "0x" prefix
    add     x2, x2, #2
    sub     w1, w1, #2
    mov     x0, #0
    mov     w3, #0
.Lnum_hex_loop:
    cmp     w3, w1
    b.ge    .Lnum_done
    ldrb    w4, [x2, x3]
    // Convert hex digit
    cmp     w4, #'9'
    b.le    .Lhex_digit
    cmp     w4, #'F'
    b.le    .Lhex_upper
    // lowercase a-f
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lhex_accum
.Lhex_upper:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lhex_accum
.Lhex_digit:
    sub     w4, w4, #'0'
.Lhex_accum:
    lsl     x0, x0, #4
    and     x4, x4, #0xFF
    orr     x0, x0, x4
    add     w3, w3, #1
    b       .Lnum_hex_loop

.Lnum_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// sym_lookup_tok — look up current token in symbol table
//   Returns: x0 = pointer to sym entry, or 0 if not found
//   Searches from most recent to oldest.
// ============================================================
sym_lookup_tok:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w0, [x19, #4]          // token offset
    ldr     w1, [x19, #8]          // token length

    adrp    x2, ls_sym_count
    add     x2, x2, :lo12:ls_sym_count
    ldr     w3, [x2]
    cbz     w3, .Lslt_notfound

    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table

    sub     w3, w3, #1
    mov     w5, #SYM_SIZE
    madd    x6, x3, x5, x4         // x6 = &sym_table[n-1]

.Lslt_loop:
    ldr     w7, [x6, #SYM_NAME_LEN]
    cmp     w7, w1
    b.ne    .Lslt_next

    // Lengths match — compare bytes
    ldr     w7, [x6, #SYM_NAME_OFF]
    add     x7, x28, x7            // sym name pointer
    add     x2, x28, x0            // token text pointer
    mov     w5, #0
.Lslt_cmp:
    cmp     w5, w1
    b.ge    .Lslt_found
    ldrb    w3, [x7, x5]
    ldrb    w4, [x2, x5]
    cmp     w3, w4
    b.ne    .Lslt_next
    add     w5, w5, #1
    b       .Lslt_cmp

.Lslt_found:
    mov     x0, x6
    b       .Lslt_done

.Lslt_next:
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
    cmp     x6, x4
    b.ls    .Lslt_notfound
    sub     x6, x6, #SYM_SIZE
    b       .Lslt_loop

.Lslt_notfound:
    mov     x0, #0

.Lslt_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// tok_match — compare current token text to a literal
//   x0 = string pointer, w1 = string length
//   Returns w0 = 1 if match, 0 if not
// ============================================================
ee_tok_match:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    mov     x4, x0
    mov     w5, w1
    ldr     w2, [x19, #8]          // token length
    cmp     w2, w5
    b.ne    .Ltm_no
    ldr     w3, [x19, #4]
    add     x3, x28, x3            // token text
    mov     w2, #0
.Ltm_loop:
    cmp     w2, w5
    b.ge    .Ltm_yes
    ldrb    w0, [x3, x2]
    ldrb    w1, [x4, x2]
    cmp     w0, w1
    b.ne    .Ltm_no
    add     w2, w2, #1
    b       .Ltm_loop
.Ltm_yes:
    mov     w0, #1
    b       .Ltm_done
.Ltm_no:
    mov     w0, #0
.Ltm_done:
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// emit_binop — emit a 3-register binary op instruction
//   w0 = rd, w1 = rn (left), w2 = rm (right)
//   x3 = base opcode (32-bit, in low 32 bits of x3)
//   Encodes: base | Rm<<16 | Rn<<5 | Rd
// ============================================================
emit_binop:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w2, #16             // Rm field
    lsl     w5, w1, #5              // Rn field
    orr     w6, w0, w5
    orr     w6, w6, w4
    orr     w0, w6, w3              // | base opcode
    bl      emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// parse_atom — lowest level: literals, identifiers, parens,
//              memory loads (→), register reads (↑)
//
// Returns: w0 = register number holding the result
// ============================================================
.global parse_atom
parse_atom:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!

    ldr     w0, [x19]              // current token type

    // --- Integer literal ---
    cmp     w0, #TOK_INT
    b.eq    .Latom_int

    // --- Identifier (symbol lookup) ---
    cmp     w0, #TOK_IDENT
    b.eq    .Latom_ident

    // --- Parenthesized expression ---
    cmp     w0, #TOK_LPAREN
    b.eq    .Latom_paren

    // --- Memory load: → width addr ---
    cmp     w0, #TOK_LOAD
    b.eq    .Latom_load

    // --- Register read: ↑ $N ---
    cmp     w0, #TOK_REG_READ
    b.eq    .Latom_reg_read

    // Unknown token in expression — error
    b       parse_error

// --- Integer or hex literal ---
.Latom_int:
    bl      parse_number            // x0 = value
    mov     x9, x0                  // save value
    bl      ee_advance              // consume the token

    // Allocate a register and emit MOVZ/MOVK to load the value
    mov     x10, x9                 // x10 = value to load
    bl      alloc_reg               // w0 = dest register number
    mov     w9, w0                  // w9 = dest reg

    // Emit MOVZ Xd, #(imm & 0xFFFF)
    mov     w0, w9
    mov     x1, x10
    bl      emit_mov_imm64          // handles full 64-bit with MOVZ+MOVK chain

    mov     w0, w9                  // return register number
    b       .Latom_done

// --- Identifier: symbol table lookup ---
.Latom_ident:
    bl      sym_lookup_tok          // x0 = sym entry or 0
    cbz     x0, parse_error         // unknown symbol

    ldr     w9, [x0, #SYM_REG]     // register number holding the value
    bl      ee_advance              // consume the identifier token
    mov     w0, w9                  // return the symbol's register
    b       .Latom_done

// --- Parenthesized expression ---
.Latom_paren:
    bl      ee_advance              // consume '('
    bl      parse_expr              // parse inner expression
    mov     w9, w0                  // save result register

    ldr     w0, [x19]              // expect ')'
    cmp     w0, #TOK_RPAREN
    b.ne    parse_error
    bl      ee_advance              // consume ')'

    mov     w0, w9
    b       .Latom_done

// --- Memory load: → width addr ---
//   → is already lexed as TOK_LOAD
//   Next token = width (integer: 8, 16, 32, 64)
//   Next token = address expression
.Latom_load:
    bl      ee_advance              // consume '→'

    // Parse width
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    parse_error
    bl      parse_number            // x0 = width value
    mov     w10, w0                 // w10 = width
    bl      ee_advance              // consume width token

    // Parse address expression
    bl      parse_expr              // w0 = register holding address
    mov     w9, w0                  // w9 = addr register

    // Allocate result register
    bl      alloc_reg               // w0 = dest register
    mov     w1, w9                  // Xn = address reg
    mov     w2, w0                  // save dest
    stp     w2, w10, [sp, #-16]!

    // Emit LDR/LDRB/LDRH based on width
    //   LDR Xd, [Xn]  — unsigned offset 0
    //   We emit: LDR Xd, [Xn, #0]
    ldp     w2, w10, [sp], #16
    mov     w0, w2                  // rd
    // w1 = base reg (addr), emit LDR Xd, [Xn, XZR]
    // Simplification: emit LDR with zero offset
    cmp     w10, #8
    b.eq    .Lload_8
    cmp     w10, #16
    b.eq    .Lload_16
    cmp     w10, #32
    b.eq    .Lload_32
    // Default: 64-bit load
    bl      emit_ldr_x_zero         // LDR Xd, [Xn, #0]
    b       .Lload_done
.Lload_8:
    bl      emit_ldrb_reg_zero
    b       .Lload_done
.Lload_16:
    bl      emit_ldrh_reg_zero
    b       .Lload_done
.Lload_32:
    bl      emit_ldr_w_zero         // LDR Wd, [Xn, #0]
.Lload_done:
    // Free the address register if it was a temp
    // (simplified: we don't track this yet)
    mov     w0, w2                  // return dest register
    b       .Latom_done

// Fallback stubs for load widths not in parser's emit helpers
emit_ldrb_reg_zero:
    // LDRB Wd, [Xn, #0] = 0x39400000 | Rn<<5 | Rd
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w1, #5
    orr     w3, w3, w0
    ORRIMM  w3, 0x39400000, w16
    mov     w0, w3
    bl      emit32
    ldp     x30, xzr, [sp], #16
    ret

emit_ldrh_reg_zero:
    // LDRH Wd, [Xn, #0] = 0x79400000 | Rn<<5 | Rd
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w1, #5
    orr     w3, w3, w0
    ORRIMM  w3, 0x79400000, w16
    mov     w0, w3
    bl      emit32
    ldp     x30, xzr, [sp], #16
    ret

// --- Register read: ↑ $N ---
.Latom_reg_read:
    bl      ee_advance              // consume '↑'

    // Next token should be the register designator (e.g., "$0", identifier)
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    .Latom_rr_ident

    // Numeric register: $N where N is an integer
    bl      parse_number            // x0 = register number
    mov     w9, w0
    bl      ee_advance
    mov     w0, w9                  // return the hardware register directly
    b       .Latom_done

.Latom_rr_ident:
    // Named register — look up in symbol table
    bl      sym_lookup_tok
    cbz     x0, parse_error
    ldr     w9, [x0, #SYM_REG]
    bl      ee_advance
    mov     w0, w9
    b       .Latom_done

.Latom_done:
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_unary_expr — unary minus, sqrt (√), reciprocal (1/)
//
// Returns: w0 = register number holding the result
// ============================================================
.global parse_unary_expr
parse_unary_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!

    ldr     w0, [x19]

    // --- Unary minus ---
    cmp     w0, #TOK_MINUS
    b.eq    .Lunary_neg

    // --- Square root (√) ---
    cmp     w0, #TOK_SQRT
    b.eq    .Lunary_sqrt

    // --- Check for "1/" reciprocal ---
    cmp     w0, #TOK_INT
    b.ne    .Lunary_atom

    // Peek: is this "1" followed by "/"?
    bl      parse_number
    cmp     x0, #1
    b.ne    .Lunary_atom            // not "1", treat as atom

    // Check next token
    add     x1, x19, #TOK_STRIDE_SZ
    ldr     w2, [x1]
    cmp     w2, #TOK_SLASH
    b.ne    .Lunary_atom            // not "1/", treat as atom

    // It is "1/" — consume both tokens
    bl      ee_advance              // consume "1"
    bl      ee_advance              // consume "/"

    // Parse operand
    bl      parse_unary_expr        // w0 = operand register
    mov     w9, w0

    // 1/x on ARM64: emit FMOV + FDIV or use integer reciprocal approximation
    // For integer expressions, we emit: MOVZ Xd, #1; SDIV Xd, Xd, Xoperand
    bl      alloc_reg               // w0 = dest reg
    mov     w10, w0
    mov     x1, #1
    bl      emit_mov_imm64          // MOV Xdest, #1

    mov     w0, w10                 // rd
    mov     w1, w10                 // rn = dest (holds 1)
    mov     w2, w9                  // rm = operand
    bl      emit_sdiv_reg           // SDIV Xd, X(1), Xoperand

    mov     w0, w10
    b       .Lunary_done

// --- Unary minus: negate ---
.Lunary_neg:
    bl      ee_advance              // consume '-'
    bl      parse_unary_expr        // w0 = operand register
    mov     w9, w0

    bl      alloc_reg               // w0 = dest reg
    mov     w10, w0

    // Emit SUB Xd, XZR, Xoperand  (NEG)
    lsl     w3, w9, #16             // Rm field
    mov     w4, #31                 // Rn = XZR
    lsl     w4, w4, #5
    orr     w5, w10, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCB000000, w16     // SUB X
    mov     w0, w5
    stp     x9, x10, [sp, #-16]!
    bl      emit32
    ldp     x9, x10, [sp], #16

    mov     w0, w10
    b       .Lunary_done

// --- Square root (integer approx: not directly available, emit placeholder) ---
.Lunary_sqrt:
    bl      ee_advance              // consume '√'
    bl      parse_unary_expr        // w0 = operand register
    // For integer bootstrap, sqrt is rarely used. Pass through the operand.
    // A full implementation would emit FSQRT for float expressions.
    b       .Lunary_done

// --- Not a unary op: fall through to parse_atom ---
.Lunary_atom:
    bl      parse_atom
    b       .Lunary_done

.Lunary_done:
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// Binary operator parser template
//
// Each level calls the next-higher-precedence parser for left and
// right operands, checks for its operator token(s), and emits the
// corresponding ARM64 instruction.
//
// Pattern:
//   left = parse_higher()
//   while (current token is my operator) {
//       consume operator
//       right = parse_higher()
//       dest = alloc_reg()
//       emit OP Xdest, Xleft, Xright
//       left = dest
//   }
//   return left
// ============================================================

// ============================================================
// parse_mul_expr — handles *, /
// ============================================================
.global parse_mul_expr
parse_mul_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_unary_expr        // w0 = left
    mov     w9, w0                  // w9 = left register

.Lmul_loop:
    ldr     w0, [x19]              // peek token type

    cmp     w0, #TOK_STAR
    b.eq    .Lmul_star
    cmp     w0, #TOK_SLASH
    b.eq    .Lmul_slash
    b       .Lmul_done

.Lmul_star:
    bl      ee_advance              // consume '*'
    bl      parse_unary_expr        // w0 = right
    mov     w10, w0

    bl      alloc_reg               // w0 = dest
    mov     w11, w0
    mov     w0, w11                 // rd
    mov     w1, w9                  // rn = left
    mov     w2, w10                 // rm = right
    bl      emit_mul_reg

    mov     w9, w11                 // left = dest
    b       .Lmul_loop

.Lmul_slash:
    bl      ee_advance              // consume '/'
    bl      parse_unary_expr        // w0 = right
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_sdiv_reg

    mov     w9, w11
    b       .Lmul_loop

.Lmul_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_add_expr — handles +, -
// ============================================================
.global parse_add_expr
parse_add_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_mul_expr          // w0 = left
    mov     w9, w0

.Ladd_loop:
    ldr     w0, [x19]

    cmp     w0, #TOK_PLUS
    b.eq    .Ladd_plus
    cmp     w0, #TOK_MINUS
    b.eq    .Ladd_minus
    b       .Ladd_done

.Ladd_plus:
    bl      ee_advance
    bl      parse_mul_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_add_reg

    mov     w9, w11
    b       .Ladd_loop

.Ladd_minus:
    bl      ee_advance
    bl      parse_mul_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_sub_reg

    mov     w9, w11
    b       .Ladd_loop

.Ladd_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_shift_expr — handles <<, >>
// ============================================================
.global parse_shift_expr
parse_shift_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_add_expr
    mov     w9, w0

.Lshift_loop:
    ldr     w0, [x19]

    cmp     w0, #TOK_SHL
    b.eq    .Lshift_left
    cmp     w0, #TOK_SHR
    b.eq    .Lshift_right
    b       .Lshift_done

.Lshift_left:
    bl      ee_advance
    bl      parse_add_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_lsl_reg

    mov     w9, w11
    b       .Lshift_loop

.Lshift_right:
    bl      ee_advance
    bl      parse_add_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_lsr_reg

    mov     w9, w11
    b       .Lshift_loop

.Lshift_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_and_expr — handles 'and', '&'
// ============================================================
.global parse_and_expr
parse_and_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_shift_expr
    mov     w9, w0

.Land_loop:
    ldr     w0, [x19]

    cmp     w0, #TOK_AMP
    b.eq    .Land_op

    // Check for "and" keyword (TOK_IDENT with text "and")
    cmp     w0, #TOK_IDENT
    b.ne    .Land_done
    adrp    x0, str_and
    add     x0, x0, :lo12:str_and
    mov     w1, #3
    bl      ee_tok_match
    cbz     w0, .Land_done

.Land_op:
    bl      ee_advance
    bl      parse_shift_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_and_reg

    mov     w9, w11
    b       .Land_loop

.Land_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_xor_expr — handles 'xor', '^'
// ============================================================
.global parse_xor_expr
parse_xor_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_and_expr
    mov     w9, w0

.Lxor_loop:
    ldr     w0, [x19]

    cmp     w0, #TOK_CARET
    b.eq    .Lxor_op

    // Check for "xor" keyword
    cmp     w0, #TOK_IDENT
    b.ne    .Lxor_done
    adrp    x0, str_xor
    add     x0, x0, :lo12:str_xor
    mov     w1, #3
    bl      ee_tok_match
    cbz     w0, .Lxor_done

.Lxor_op:
    bl      ee_advance
    bl      parse_and_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_eor_reg

    mov     w9, w11
    b       .Lxor_loop

.Lxor_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_or_expr — handles '|' (bitwise OR, lowest precedence)
// ============================================================
.global parse_or_expr
parse_or_expr:
    stp     x30, x19, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!
    stp     x11, x12, [sp, #-16]!

    bl      parse_xor_expr
    mov     w9, w0

.Lor_loop:
    ldr     w0, [x19]

    cmp     w0, #TOK_PIPE
    b.ne    .Lor_done

    bl      ee_advance
    bl      parse_xor_expr
    mov     w10, w0

    bl      alloc_reg
    mov     w11, w0
    mov     w0, w11
    mov     w1, w9
    mov     w2, w10
    bl      emit_orr_reg

    mov     w9, w11
    b       .Lor_loop

.Lor_done:
    mov     w0, w9
    ldp     x11, x12, [sp], #16
    ldp     x9, x10, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_expr — top-level entry point
//   Parses a full expression with correct precedence.
//   Returns: w0 = register number holding the computed result.
//   Side effect: emits ARM64 instructions into the code buffer.
// ============================================================
.global parse_expr
parse_expr:
    // Delegate to the lowest-precedence level
    b       parse_or_expr

// ============================================================
// DTC wrapper — call parse_expr from threaded bootstrap
//   ( -- reg )  Pushes result register number onto DTC data stack
// ============================================================
.align 4
.global code_PARSE_EXPR
code_PARSE_EXPR:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x20, [sp, #-16]!

    bl      parse_expr

    // Restore DTC state and push result
    ldp     x22, x20, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16

    // Push result register number onto DTC data stack
    PUSH    x0
    NEXT

// ============================================================
// .data — string constants
// ============================================================
.data
.align 3

str_and:
    .ascii "and"
str_xor:
    .ascii "xor"

// ============================================================
// Dictionary entry — extends the chain
// ============================================================
.align 3
entry_ee_parse_expr:
    .quad   0                       // link — set by linker/bootstrap
    .byte   0                       // flags
    .byte   10                      // name length
    .ascii  "parse-expr"
    .align  3
    .quad   code_PARSE_EXPR
