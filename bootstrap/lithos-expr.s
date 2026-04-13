// lithos-expr.s — Expression compiler for Lithos
//
// Compiles infix expressions from the token stream into ARM64 or SASS
// machine code. Operates on the token buffer produced by lithos-lexer.ls.
//
// Architecture: recursive descent with operator precedence.
// Single-pass, no AST — parses tokens and emits instructions directly.
//
// Threading: Direct Threaded Code (DTC), same register map as lithos-bootstrap.s
//   X26 = IP    X25 = W     X24 = DSP    X23 = RSP
//   X22 = TOS   X20 = HERE  X21 = BASE
//
// Token buffer layout (from lexer): u32 triples [type, offset, length]
// Token constants match lithos-lexer.ls (TOK_* enum).
//
// Build: assembled as part of the lithos-bootstrap image.
//        Included after lithos-bootstrap.s core words.

// ============================================================
// DTC macros — defined only if build.sh hasn't injected them.
// ============================================================
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
// Token type constants (match lithos-lexer.ls)
// ============================================================
.equ TOK_EOF,       0
.equ TOK_NEWLINE,   1
.equ TOK_INDENT,    2
.equ TOK_INT,       3
.equ TOK_FLOAT,     4
.equ TOK_IDENT,     5
.equ TOK_KERNEL,    11
.equ TOK_IF,        13
.equ TOK_FOR,       16
.equ TOK_CONST,     22
.equ TOK_VAR,       23
.equ TOK_BUF,       24
.equ TOK_SHARED,    31
.equ TOK_BARRIER,   32
.equ TOK_EXIT,      34
.equ TOK_HOST,      35
.equ TOK_LOAD,      36
.equ TOK_STORE,     37
.equ TOK_REG_READ,  38
.equ TOK_REG_WRITE, 39
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_EQ,        54
.equ TOK_EQEQ,      55
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
.equ TOK_LBRACK,    67
.equ TOK_RBRACK,    68
.equ TOK_LPAREN,    69
.equ TOK_RPAREN,    70
.equ TOK_COLON,     72

// Operator precedence levels (higher = tighter binding)
.equ PREC_OR,       1    // |
.equ PREC_AND,      2    // &
.equ PREC_CMP,      3    // == != < > <= >=
.equ PREC_ADD,      4    // + -
.equ PREC_MUL,      5    // * / % >> <<
.equ PREC_UNARY,    6    // unary - !
.equ PREC_ATOM,     7    // literals, idents, parens, array index

// Target backend constants
.equ TARGET_GPU,    0
.equ TARGET_HOST,   1

// ARM64 instruction encodings (fixed constants)
.equ ARM64_ADD_REG,   0x8B000000   // ADD Xd, Xn, Xm
.equ ARM64_SUB_REG,   0xCB000000   // SUB Xd, Xn, Xm
.equ ARM64_MUL,       0x9B007C00   // MUL Xd, Xn, Xm (MADD Xd,Xn,Xm,XZR)
.equ ARM64_SDIV,      0x9AC00C00   // SDIV Xd, Xn, Xm
.equ ARM64_UDIV,      0x9AC00800   // UDIV Xd, Xn, Xm
.equ ARM64_AND_REG,   0x8A000000   // AND Xd, Xn, Xm
.equ ARM64_ORR_REG,   0xAA000000   // ORR Xd, Xn, Xm
.equ ARM64_EOR_REG,   0xCA000000   // EOR Xd, Xn, Xm
.equ ARM64_LSL_REG,   0x9AC02000   // LSLV Xd, Xn, Xm
.equ ARM64_LSR_REG,   0x9AC02400   // LSRV Xd, Xn, Xm
.equ ARM64_SUBS_REG,  0xEB000000   // SUBS Xd, Xn, Xm (CMP when Rd=XZR)
.equ ARM64_CSINC,     0x9A800400   // CSINC Xd, Xn, Xm, cond
.equ ARM64_MOVZ,      0xD2800000   // MOVZ Xd, #imm16, LSL #hw*16
.equ ARM64_MOVK,      0xF2800000   // MOVK Xd, #imm16, LSL #hw*16
.equ ARM64_LDRB_REG,  0x38606800   // LDRB Wt, [Xn, Xm]
.equ ARM64_LDRB_IMM,  0x39400000   // LDRB Wt, [Xn, #imm12]
.equ ARM64_LDR_IMM,   0xF9400000   // LDR Xt, [Xn, #imm12*8]
.equ ARM64_LDR_REG,   0xF8606800   // LDR Xt, [Xn, Xm]
.equ ARM64_STR_IMM,   0xF9000000   // STR Xt, [Xn, #imm12*8]
.equ ARM64_STR_REG,   0xF8206800   // STR Xt, [Xn, Xm]
.equ ARM64_STRB_IMM,  0x39000000   // STRB Wt, [Xn, #imm12]
.equ ARM64_MOV_REG,   0xAA0003E0   // MOV Xd, Xm (ORR Xd, XZR, Xm)
.equ ARM64_BL,        0x94000000   // BL offset
.equ ARM64_ADD_IMM,   0x91000000   // ADD Xd, Xn, #imm12
.equ ARM64_MSUB,      0x9B008000   // MSUB Xd, Xn, Xm, Xa (for MOD)

// Condition codes for CSET
.equ COND_EQ,   0
.equ COND_NE,   1
.equ COND_LT,   11
.equ COND_GT,   12
.equ COND_LE,   13
.equ COND_GE,   10

// SM90 SASS opcodes (16-byte instructions)
.equ SASS_FADD,      0x7221
.equ SASS_FMUL,      0x7220
.equ SASS_FFMA,      0x7223
.equ SASS_IADD3,     0x7210
.equ SASS_IMAD,      0x7224
.equ SASS_LOP3,      0x7212
.equ SASS_SHF,       0x7819
.equ SASS_LDG,       0x7981
.equ SASS_STG,       0x7986
.equ SASS_MOV_IMM,   0x7802
.equ SASS_MUFU,      0x7308
.equ SASS_ISETP,     0x720C
.equ SASS_FSETP,     0x720B
.equ SASS_IADD3_IMM, 0x7810
.equ SASS_FMNMX,     0x7209

// Default control word: stall=4, yield=1, wbar=7, rbar=7, wait=0, reuse=0
.equ SASS_CTRL_DEFAULT, 0x000fca0000000000

// ============================================================
// Expression compiler state
//
// Shared state (code buf/cursor, sym table, comp table) lives in
// ls-shared.s. Expr-private scratch (token cursor, target flag,
// SASS buffer, register allocator) stays here.
// ============================================================
.extern ls_code_buf
.extern ls_code_pos
.extern ls_sym_table
.extern ls_sym_count
.extern ls_comp_table
.extern ls_comp_count

.bss
.align 3

// Token stream pointers — expr-private cursor state
expr_tok_buf:       .skip 8     // pointer to token buffer base (set at init)
expr_tok_pos:       .skip 8     // current token index (in triples)
expr_tok_total:     .skip 8     // total token count
expr_src_buf:       .skip 8     // pointer to source text (for reading ident/number text)

// Code emission target
expr_target:        .skip 8     // 0 = GPU (SASS), 1 = HOST (ARM64)

// SASS code buffer — expr-private (GPU target only)
expr_sass_buf:      .skip 65536     // 64KB for SASS code
expr_sass_pos:      .skip 8         // current write position in sass buf

// Register allocator — monotone bump
// ARM64: X9-X15 are scratch (7 regs), X0-X7 for args
// GPU: R4+ for general purpose (R0-R3 reserved)
expr_next_reg:      .skip 8     // next register to allocate
expr_max_reg:       .skip 8     // high-water mark

// Sizing constants retained for reference (actual storage in ls-shared.s)
.equ SYM_ENTRY_SIZE, 48
.equ SYM_MAX, 128
.equ COMP_ENTRY_SIZE, 48
.equ COMP_MAX, 64

.text
.align 4

// ============================================================
// INITIALIZATION
// ============================================================
// expr_init ( tok_buf tok_total src_buf target -- )
// Initialize the expression compiler state.
.align 4
code_EXPR_INIT:
    // TOS = target
    POP     x0          // src_buf
    POP     x1          // tok_total
    POP     x2          // tok_buf

    adrp    x3, expr_target
    add     x3, x3, :lo12:expr_target
    str     x22, [x3]

    adrp    x3, expr_src_buf
    add     x3, x3, :lo12:expr_src_buf
    str     x0, [x3]

    adrp    x3, expr_tok_total
    add     x3, x3, :lo12:expr_tok_total
    str     x1, [x3]

    adrp    x3, expr_tok_buf
    add     x3, x3, :lo12:expr_tok_buf
    str     x2, [x3]

    adrp    x3, expr_tok_pos
    add     x3, x3, :lo12:expr_tok_pos
    str     xzr, [x3]

    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    str     xzr, [x3]

    adrp    x3, expr_sass_pos
    add     x3, x3, :lo12:expr_sass_pos
    str     xzr, [x3]

    adrp    x3, expr_next_reg
    add     x3, x3, :lo12:expr_next_reg
    mov     x4, #9          // ARM64: start at X9
    str     x4, [x3]

    adrp    x3, expr_max_reg
    add     x3, x3, :lo12:expr_max_reg
    str     xzr, [x3]

    adrp    x3, ls_sym_count
    add     x3, x3, :lo12:ls_sym_count
    str     xzr, [x3]

    adrp    x3, ls_comp_count
    add     x3, x3, :lo12:ls_comp_count
    str     xzr, [x3]

    ldr     x22, [x24], #8
    NEXT

// ============================================================
// TOKEN STREAM ACCESS
// ============================================================

// expr_peek_type ( -- type )
// Return the type of the current token without consuming it.
.align 4
code_EXPR_PEEK_TYPE:
    str     x22, [x24, #-8]!
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]           // tok_pos (index in triples)
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]           // base of token buffer
    // token[tok_pos] = 3 u32 values, type is first
    // byte offset = tok_pos * 3 * 4 = tok_pos * 12
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w22, [x4]          // load u32 type
    NEXT

// expr_peek_offset ( -- offset )
.align 4
code_EXPR_PEEK_OFFSET:
    str     x22, [x24, #-8]!
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w22, [x4, #4]      // offset is second u32
    NEXT

// expr_peek_length ( -- length )
.align 4
code_EXPR_PEEK_LENGTH:
    str     x22, [x24, #-8]!
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w22, [x4, #8]      // length is third u32
    NEXT

// expr_consume ( -- )
// Advance to the next token.
.align 4
code_EXPR_CONSUME:
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]
    NEXT

// expr_expect ( type -- )
// Consume current token, panic if it doesn't match type.
.align 4
code_EXPR_EXPECT:
    // TOS = expected type
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]           // actual type
    cmp     w5, w22
    b.ne    expr_expect_fail
    add     x1, x1, #1
    str     x1, [x0]
    ldr     x22, [x24], #8
    NEXT

expr_expect_fail:
    // Write error message and exit
    adrp    x1, expr_err_expect
    add     x1, x1, :lo12:expr_err_expect
    mov     x2, #24
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// REGISTER ALLOCATOR
// ============================================================

// expr_alloc_reg ( -- reg )
// Allocate the next available register. Bump allocator.
// ARM64 target: returns X9, X10, ... X15, then X0-X7 overflow
// GPU target: returns R4, R5, R6, ...
.align 4
code_EXPR_ALLOC_REG:
    str     x22, [x24, #-8]!
    adrp    x0, expr_next_reg
    add     x0, x0, :lo12:expr_next_reg
    ldr     x22, [x0]
    add     x1, x22, #1
    str     x1, [x0]
    // Update high-water mark
    adrp    x2, expr_max_reg
    add     x2, x2, :lo12:expr_max_reg
    ldr     x3, [x2]
    cmp     x22, x3
    b.le    1f
    str     x22, [x2]
1:  NEXT

// expr_reset_regs ( -- )
// Reset register allocator for a new expression/composition.
.align 4
code_EXPR_RESET_REGS:
    adrp    x0, expr_next_reg
    add     x0, x0, :lo12:expr_next_reg
    adrp    x1, expr_target
    add     x1, x1, :lo12:expr_target
    ldr     x2, [x1]
    cbz     x2, 1f
    // HOST: start at X9
    mov     x3, #9
    str     x3, [x0]
    NEXT
1:  // GPU: start at R4
    mov     x3, #4
    str     x3, [x0]
    NEXT

// ============================================================
// CODE EMISSION — ARM64
// ============================================================

// expr_emit_arm64 ( u32 -- )
// Append a 32-bit ARM64 instruction to the code buffer.
.align 4
code_EXPR_EMIT_ARM64:
    adrp    x0, ls_code_pos
    add     x0, x0, :lo12:ls_code_pos
    ldr     x1, [x0]
    adrp    x2, ls_code_buf
    add     x2, x2, :lo12:ls_code_buf
    str     w22, [x2, x1]
    add     x1, x1, #4
    str     x1, [x0]
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// CODE EMISSION — SASS (SM90 GPU)
// ============================================================

// expr_emit_sass ( inst_lo inst_hi ctrl_lo ctrl_hi -- )
// Append a 16-byte SASS instruction (8-byte inst + 8-byte ctrl).
.align 4
code_EXPR_EMIT_SASS:
    // TOS = ctrl_hi
    POP     x3          // ctrl_lo
    POP     x2          // inst_hi
    POP     x1          // inst_lo
    adrp    x0, expr_sass_pos
    add     x0, x0, :lo12:expr_sass_pos
    ldr     x4, [x0]
    adrp    x5, expr_sass_buf
    add     x5, x5, :lo12:expr_sass_buf
    add     x5, x5, x4
    // Combine inst halves into 64-bit
    orr     x6, x1, x2, lsl #32
    str     x6, [x5]
    // Combine ctrl halves into 64-bit
    orr     x7, x3, x22, lsl #32
    str     x7, [x5, #8]
    add     x4, x4, #16
    str     x4, [x0]
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// SYMBOL TABLE
// ============================================================

// expr_sym_find ( addr len -- index|-1 )
// Linear search the symbol table for a name match.
.align 4
code_EXPR_SYM_FIND:
    // TOS = len, NOS = addr
    POP     x0              // addr
    mov     x1, x22         // len
    adrp    x2, ls_sym_count
    add     x2, x2, :lo12:ls_sym_count
    ldr     x3, [x2]       // n_syms
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
    mov     x5, #0          // index
sym_find_loop:
    cmp     x5, x3
    b.ge    sym_find_notfound
    mov     x6, #SYM_ENTRY_SIZE
    mul     x7, x5, x6
    add     x7, x4, x7     // entry base
    // Check name_len at offset 32
    ldr     w8, [x7, #32]
    cmp     w8, w1
    b.ne    sym_find_next
    // Compare name bytes
    mov     x9, #0
sym_cmp_loop:
    cmp     x9, x1
    b.ge    sym_find_found
    ldrb    w10, [x7, x9]
    ldrb    w11, [x0, x9]
    cmp     w10, w11
    b.ne    sym_find_next
    add     x9, x9, #1
    b       sym_cmp_loop
sym_find_found:
    mov     x22, x5
    NEXT
sym_find_next:
    add     x5, x5, #1
    b       sym_find_loop
sym_find_notfound:
    mov     x22, #-1
    NEXT

// expr_sym_add ( addr len kind reg -- )
// Add a symbol to the table.
.align 4
code_EXPR_SYM_ADD:
    // TOS = reg
    POP     x3          // kind
    POP     x1          // len
    POP     x0          // addr
    adrp    x4, ls_sym_count
    add     x4, x4, :lo12:ls_sym_count
    ldr     x5, [x4]       // index
    adrp    x6, ls_sym_table
    add     x6, x6, :lo12:ls_sym_table
    mov     x7, #SYM_ENTRY_SIZE
    mul     x8, x5, x7
    add     x8, x6, x8     // entry base
    // Copy name (up to 31 bytes)
    mov     x9, #0
    cmp     x1, #31
    b.le    1f
    mov     x1, #31
1:  cbz     x1, 3f
2:  ldrb    w10, [x0, x9]
    strb    w10, [x8, x9]
    add     x9, x9, #1
    cmp     x9, x1
    b.lt    2b
3:  // Store name_len, kind, reg
    str     w1, [x8, #32]
    str     w3, [x8, #36]
    str     w22, [x8, #40]
    // Increment count
    add     x5, x5, #1
    str     x5, [x4]
    ldr     x22, [x24], #8
    NEXT

// expr_sym_reg ( index -- reg )
// Get the register number for symbol at index.
.align 4
code_EXPR_SYM_REG:
    adrp    x0, ls_sym_table
    add     x0, x0, :lo12:ls_sym_table
    mov     x1, #SYM_ENTRY_SIZE
    mul     x2, x22, x1
    add     x2, x0, x2
    ldr     w22, [x2, #40]
    NEXT

// ============================================================
// NUMBER PARSING
// ============================================================

// expr_parse_int ( -- value )
// Parse the current token as an integer literal and consume it.
// Handles decimal and hex (0x prefix).
.align 4
code_EXPR_PARSE_INT:
    str     x22, [x24, #-8]!
    // Get current token offset and length
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4, #4]       // offset in source
    ldr     w6, [x4, #8]       // length
    // Consume token
    add     x1, x1, #1
    str     x1, [x0]
    // Get source text pointer
    adrp    x7, expr_src_buf
    add     x7, x7, :lo12:expr_src_buf
    ldr     x7, [x7]
    add     x7, x7, x5         // ptr to number text
    // Check for negative sign
    mov     x22, #0             // accumulator
    mov     x8, #0              // negative flag
    ldrb    w9, [x7]
    cmp     w9, #'-'
    b.ne    parse_int_nosign
    mov     x8, #1
    add     x7, x7, #1
    sub     x6, x6, #1
parse_int_nosign:
    // Check for 0x prefix
    cmp     x6, #2
    b.lt    parse_int_decimal
    ldrb    w9, [x7]
    cmp     w9, #'0'
    b.ne    parse_int_decimal
    ldrb    w10, [x7, #1]
    cmp     w10, #'x'
    b.eq    parse_int_hex
    cmp     w10, #'X'
    b.eq    parse_int_hex
parse_int_decimal:
    cbz     x6, parse_int_done
    ldrb    w9, [x7], #1
    sub     w9, w9, #'0'
    cmp     w9, #9
    b.hi    parse_int_done
    mov     x10, #10
    mul     x22, x22, x10
    add     x22, x22, x9
    sub     x6, x6, #1
    b       parse_int_decimal
parse_int_hex:
    add     x7, x7, #2
    sub     x6, x6, #2
parse_int_hex_loop:
    cbz     x6, parse_int_done
    ldrb    w9, [x7], #1
    // 0-9
    sub     w10, w9, #'0'
    cmp     w10, #9
    b.hi    1f
    lsl     x22, x22, #4
    add     x22, x22, x10
    sub     x6, x6, #1
    b       parse_int_hex_loop
1:  // a-f
    sub     w10, w9, #'a'
    cmp     w10, #5
    b.hi    2f
    lsl     x22, x22, #4
    add     w10, w10, #10
    add     x22, x22, x10
    sub     x6, x6, #1
    b       parse_int_hex_loop
2:  // A-F
    sub     w10, w9, #'A'
    cmp     w10, #5
    b.hi    parse_int_done
    lsl     x22, x22, #4
    add     w10, w10, #10
    add     x22, x22, x10
    sub     x6, x6, #1
    b       parse_int_hex_loop
parse_int_done:
    cbz     x8, 1f
    neg     x22, x22
1:  NEXT

// ============================================================
// ARM64 INSTRUCTION ENCODERS
// ============================================================
// Each takes register operands on the stack and emits one ARM64 instruction.

// expr_emit_a64_rrr ( opcode rd rn rm -- )
// Emit a register-register-register instruction:
//   opcode | Rm<<16 | Rn<<5 | Rd
.align 4
code_EXPR_EMIT_A64_RRR:
    // TOS = rm
    POP     x2          // rn
    POP     x1          // rd
    POP     x0          // opcode
    orr     x0, x0, x22, lsl #16   // | Rm<<16
    orr     x0, x0, x2, lsl #5     // | Rn<<5
    orr     x0, x0, x1              // | Rd
    // Write to buffer
    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    ldr     x4, [x3]
    adrp    x5, ls_code_buf
    add     x5, x5, :lo12:ls_code_buf
    str     w0, [x5, x4]
    add     x4, x4, #4
    str     x4, [x3]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_cmp_cset ( rd rn rm cond -- )
// Emit CMP Xn, Xm then CSET Xd, cond
// CMP = SUBS XZR, Xn, Xm = 0xEB000000 | Rm<<16 | Rn<<5 | 31
// CSET Xd, cond = CSINC Xd, XZR, XZR, invert(cond)
.align 4
code_EXPR_EMIT_A64_CMP_CSET:
    // TOS = cond
    POP     x2          // rm
    POP     x1          // rn
    POP     x0          // rd
    // Emit CMP
    mov     x3, #ARM64_SUBS_REG
    orr     x3, x3, x2, lsl #16    // Rm
    orr     x3, x3, x1, lsl #5     // Rn
    orr     x3, x3, #31            // Rd = XZR
    adrp    x4, ls_code_pos
    add     x4, x4, :lo12:ls_code_pos
    ldr     x5, [x4]
    adrp    x6, ls_code_buf
    add     x6, x6, :lo12:ls_code_buf
    str     w3, [x6, x5]
    add     x5, x5, #4
    // Emit CSET = CSINC Xd, XZR, XZR, invert(cond)
    eor     x7, x22, #1            // invert condition
    mov     x3, #ARM64_CSINC
    orr     x3, x3, #(31 << 16)    // Xm = XZR
    orr     x3, x3, x7, lsl #12    // inverted cond
    orr     x3, x3, #(31 << 5)     // Xn = XZR
    orr     x3, x3, x0             // Rd
    str     w3, [x6, x5]
    add     x5, x5, #4
    str     x5, [x4]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_mov_imm ( rd imm64 -- )
// Emit MOVZ + MOVK sequence to load a 64-bit immediate.
.align 4
code_EXPR_EMIT_A64_MOV_IMM:
    // TOS = imm64
    POP     x0          // rd
    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    ldr     x4, [x3]
    adrp    x5, ls_code_buf
    add     x5, x5, :lo12:ls_code_buf
    // MOVZ Xd, #chunk0, LSL #0
    and     x1, x22, #0xFFFF
    mov     x2, #ARM64_MOVZ
    orr     x2, x2, x1, lsl #5
    orr     x2, x2, x0
    str     w2, [x5, x4]
    add     x4, x4, #4
    // MOVK Xd, #chunk1, LSL #16
    ubfx    x1, x22, #16, #16
    cbz     x1, 2f
    mov     x2, #ARM64_MOVK
    orr     x2, x2, #(1 << 21)     // hw=1
    orr     x2, x2, x1, lsl #5
    orr     x2, x2, x0
    str     w2, [x5, x4]
    add     x4, x4, #4
2:  // MOVK Xd, #chunk2, LSL #32
    ubfx    x1, x22, #32, #16
    cbz     x1, 3f
    mov     x2, #ARM64_MOVK
    orr     x2, x2, #(2 << 21)     // hw=2
    orr     x2, x2, x1, lsl #5
    orr     x2, x2, x0
    str     w2, [x5, x4]
    add     x4, x4, #4
3:  // MOVK Xd, #chunk3, LSL #48
    ubfx    x1, x22, #48, #16
    cbz     x1, 4f
    mov     x2, #ARM64_MOVK
    orr     x2, x2, #(3 << 21)     // hw=3
    orr     x2, x2, x1, lsl #5
    orr     x2, x2, x0
    str     w2, [x5, x4]
    add     x4, x4, #4
4:  str     x4, [x3]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_mov_reg ( rd rs -- )
// Emit MOV Xd, Xs (ORR Xd, XZR, Xs)
.align 4
code_EXPR_EMIT_A64_MOV_REG:
    // TOS = rs
    POP     x0          // rd
    mov     x1, #ARM64_MOV_REG
    orr     x1, x1, x22, lsl #16   // Rm
    orr     x1, x1, x0             // Rd
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    ldr     x3, [x2]
    adrp    x4, ls_code_buf
    add     x4, x4, :lo12:ls_code_buf
    str     w1, [x4, x3]
    add     x3, x3, #4
    str     x3, [x2]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_ldrb_reg ( rd base offset_reg -- )
// Emit LDRB Wd, [Xbase, Xoffset]
.align 4
code_EXPR_EMIT_A64_LDRB_REG:
    // TOS = offset_reg
    POP     x1          // base
    POP     x0          // rd
    mov     x2, #ARM64_LDRB_REG
    orr     x2, x2, x22, lsl #16   // Rm = offset
    orr     x2, x2, x1, lsl #5     // Rn = base
    orr     x2, x2, x0             // Rt = rd
    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    ldr     x4, [x3]
    adrp    x5, ls_code_buf
    add     x5, x5, :lo12:ls_code_buf
    str     w2, [x5, x4]
    add     x4, x4, #4
    str     x4, [x3]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_ldr_reg ( rd base offset_reg -- )
// Emit LDR Xd, [Xbase, Xoffset]
.align 4
code_EXPR_EMIT_A64_LDR_REG:
    // TOS = offset_reg
    POP     x1          // base
    POP     x0          // rd
    mov     x2, #ARM64_LDR_REG
    orr     x2, x2, x22, lsl #16
    orr     x2, x2, x1, lsl #5
    orr     x2, x2, x0
    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    ldr     x4, [x3]
    adrp    x5, ls_code_buf
    add     x5, x5, :lo12:ls_code_buf
    str     w2, [x5, x4]
    add     x4, x4, #4
    str     x4, [x3]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_str_reg ( val_reg base offset_reg -- )
// Emit STR Xval, [Xbase, Xoffset]
.align 4
code_EXPR_EMIT_A64_STR_REG:
    // TOS = offset_reg
    POP     x1          // base
    POP     x0          // val_reg
    mov     x2, #ARM64_STR_REG
    orr     x2, x2, x22, lsl #16   // Rm = offset
    orr     x2, x2, x1, lsl #5     // Rn = base
    orr     x2, x2, x0             // Rt = val
    adrp    x3, ls_code_pos
    add     x3, x3, :lo12:ls_code_pos
    ldr     x4, [x3]
    adrp    x5, ls_code_buf
    add     x5, x5, :lo12:ls_code_buf
    str     w2, [x5, x4]
    add     x4, x4, #4
    str     x4, [x3]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_a64_bl ( target_offset -- )
// Emit BL with PC-relative offset.
.align 4
code_EXPR_EMIT_A64_BL:
    // TOS = byte offset from this instruction
    asr     x0, x22, #2            // offset / 4
    and     x0, x0, #0x3FFFFFF     // mask to 26 bits
    mov     x1, #ARM64_BL
    orr     x1, x1, x0
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    ldr     x3, [x2]
    adrp    x4, ls_code_buf
    add     x4, x4, :lo12:ls_code_buf
    str     w1, [x4, x3]
    add     x3, x3, #4
    str     x3, [x2]
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// SASS INSTRUCTION ENCODERS
// ============================================================

// expr_emit_sass_alu ( opcode rd ra rb -- )
// Emit a SASS ALU instruction: opcode | rd<<16 | ra<<24 | rb<<32
// with default control word.
.align 4
code_EXPR_EMIT_SASS_ALU:
    // TOS = rb
    POP     x2          // ra
    POP     x1          // rd
    POP     x0          // opcode
    // Build inst word
    orr     x3, x0, x1, lsl #16    // | rd<<16
    orr     x3, x3, x2, lsl #24    // | ra<<24
    orr     x3, x3, x22, lsl #32   // | rb<<32
    // Write 16-byte instruction
    adrp    x4, expr_sass_pos
    add     x4, x4, :lo12:expr_sass_pos
    ldr     x5, [x4]
    adrp    x6, expr_sass_buf
    add     x6, x6, :lo12:expr_sass_buf
    add     x6, x6, x5
    str     x3, [x6]               // inst word
    mov     x7, #SASS_CTRL_DEFAULT
    movk    x7, #0x000f, lsl #48
    str     x7, [x6, #8]           // ctrl word
    add     x5, x5, #16
    str     x5, [x4]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_sass_mov_imm ( rd imm32 -- )
// Emit SASS MOV-IMM: opcode | rd<<16 | imm32<<32
.align 4
code_EXPR_EMIT_SASS_MOV_IMM:
    // TOS = imm32
    POP     x0          // rd
    mov     x1, #SASS_MOV_IMM
    orr     x1, x1, x0, lsl #16
    orr     x1, x1, x22, lsl #32
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    // ctrl for MOV-IMM
    ldr     x5, =0x000fe200078e00ff
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_sass_ldg ( rd base_reg -- )
// Emit LDG.E Rd, [Rbase]
.align 4
code_EXPR_EMIT_SASS_LDG:
    // TOS = base_reg
    POP     x0          // rd
    mov     x1, #SASS_LDG
    orr     x1, x1, x0, lsl #16
    orr     x1, x1, x22, lsl #24   // base in ra position
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    // ctrl for LDG: stall=4, yield=1, wbar=2, extra41=0x0c1e1900
    ldr     x5, =0x000d4a000c1e1900
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    ldr     x22, [x24], #8
    NEXT

// expr_emit_sass_stg ( val_reg base_reg -- )
// Emit STG.E [Rbase], Rval
.align 4
code_EXPR_EMIT_SASS_STG:
    // TOS = base_reg
    POP     x0          // val_reg
    mov     x1, #SASS_STG
    orr     x1, x1, x0, lsl #16    // val in rd position
    orr     x1, x1, x22, lsl #24   // base in ra position
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    // ctrl for STG: stall=1, extra41=0x0c101904
    ldr     x5, =0x000fca000c101904
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// EXPRESSION COMPILER — RECURSIVE DESCENT (Colon definitions)
// ============================================================
// These are DTC colon definitions. They use the token stream,
// register allocator, and code emitters above.
//
// Precedence hierarchy:
//   expr_compile_expr     -> expr_compile_or
//   expr_compile_or       -> expr_compile_and     (|)
//   expr_compile_and      -> expr_compile_cmp     (&)
//   expr_compile_cmp      -> expr_compile_add     (== != < > <= >=)
//   expr_compile_add      -> expr_compile_mul     (+ -)
//   expr_compile_mul      -> expr_compile_atom    (* / % >> <<)
//   expr_compile_atom     -> literal | ident | (expr) | load | index

// ---- expr_compile_atom ----
// Parse and compile an atomic expression. Returns register holding result.
// ( -- reg )
.align 4
code_EXPR_COMPILE_ATOM:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    // Peek at current token type
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]           // token type

    // --- Integer literal ---
    cmp     w5, #TOK_INT
    b.ne    atom_not_int
    // Parse integer, load into register
    bl      code_EXPR_PARSE_INT
    mov     x10, x22           // value in x10
    bl      code_EXPR_ALLOC_REG
    mov     x11, x22           // rd in x11
    // Dispatch on target
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, atom_int_arm64
    // GPU: emit SASS MOV-IMM
    str     x22, [x24, #-8]!   // push rd
    mov     x22, x10            // imm
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    mov     x22, x11
    b       atom_done
atom_int_arm64:
    str     x22, [x24, #-8]!   // push rd
    mov     x22, x10            // imm64
    bl      code_EXPR_EMIT_A64_MOV_IMM
    mov     x22, x11
    b       atom_done

atom_not_int:
    // --- Float literal ---
    cmp     w5, #TOK_FLOAT
    b.ne    atom_not_float
    // Parse float — treat as integer bits for now (IEEE 754)
    bl      code_EXPR_PARSE_INT
    mov     x10, x22
    bl      code_EXPR_ALLOC_REG
    mov     x11, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, atom_float_arm64
    str     x22, [x24, #-8]!
    mov     x22, x10
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    mov     x22, x11
    b       atom_done
atom_float_arm64:
    str     x22, [x24, #-8]!
    mov     x22, x10
    bl      code_EXPR_EMIT_A64_MOV_IMM
    mov     x22, x11
    b       atom_done

atom_not_float:
    // --- Identifier (variable reference or composition call) ---
    cmp     w5, #TOK_IDENT
    b.ne    atom_not_ident
    // Get name pointer and length
    ldr     w6, [x4, #4]       // offset
    ldr     w7, [x4, #8]       // length
    adrp    x8, expr_src_buf
    add     x8, x8, :lo12:expr_src_buf
    ldr     x8, [x8]
    add     x8, x8, x6         // name ptr
    // Consume token
    add     x1, x1, #1
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    str     x1, [x0]
    // Look up in symbol table
    str     x22, [x24, #-8]!
    mov     x22, x7            // len
    str     x8, [x24, #-8]!    // push addr
    // Save name info for potential composition call
    stp     x8, x7, [sp, #-16]!
    bl      code_EXPR_SYM_FIND
    ldp     x8, x7, [sp], #16
    // x22 = index or -1
    cmn     x22, #1
    b.eq    atom_ident_unknown
    // Found: return the symbol's register
    bl      code_EXPR_SYM_REG
    // x22 = reg number
    // Check if next token is [ for array index
    stp     x22, xzr, [sp, #-16]!  // save reg
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]
    ldp     x14, xzr, [sp], #16    // restore base reg to x14
    cmp     w5, #TOK_LBRACK
    b.ne    atom_ident_no_index
    // Array index: base [ expr ]
    // Consume [
    add     x1, x1, #1
    str     x1, [x0]
    // Save base reg
    str     x14, [sp, #-16]!
    // Compile index expression
    mov     x22, x14            // not needed but restore TOS state
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x15, x22            // index reg
    ldr     x14, [sp], #16      // base reg
    // Consume ]
    mov     x22, #TOK_RBRACK
    bl      code_EXPR_EXPECT
    // Emit load: rd = base[index]
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22            // result reg
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, atom_index_arm64
    // GPU: emit LDG after computing address
    // IADD3 addr_reg, base, index, RZ
    str     x22, [x24, #-8]!       // push opcode (IADD3)
    mov     x22, #0xFF             // RZ
    str     x22, [x24, #-8]!
    mov     x22, x15               // rb = index
    str     x22, [x24, #-8]!
    mov     x22, x14               // ra = base
    str     x22, [x24, #-8]!
    mov     x22, x16               // rd
    str     x22, [x24, #-8]!
    mov     x22, #SASS_IADD3
    bl      code_EXPR_EMIT_SASS_ALU
    // LDG rd, [addr_reg]
    str     x16, [x24, #-8]!   // push rd
    mov     x22, x16            // base = addr_reg (reuse)
    bl      code_EXPR_EMIT_SASS_LDG
    mov     x22, x16
    b       atom_done
atom_index_arm64:
    // ARM64: LDRB Wd, [Xbase, Xindex]
    str     x16, [x24, #-8]!   // push rd
    str     x14, [x24, #-8]!   // push base
    mov     x22, x15            // offset_reg
    bl      code_EXPR_EMIT_A64_LDRB_REG
    mov     x22, x16
    b       atom_done

atom_ident_no_index:
    mov     x22, x14
    b       atom_done

atom_ident_unknown:
    // Unknown identifier — check if it's a composition call
    // For now, allocate a register and return it (will be resolved later)
    // The composition call handling:
    // name arg1 arg2 — prefix call syntax
    // Check if next tokens are expression-starters (arguments)
    bl      code_EXPR_ALLOC_REG
    // x22 = allocated reg (placeholder)
    b       atom_done

atom_not_ident:
    // --- Memory load: → width addr ---
    cmp     w5, #TOK_LOAD
    b.ne    atom_not_load
    // Consume →
    add     x1, x1, #1
    str     x1, [x0]
    // Parse width (next token should be integer)
    bl      code_EXPR_PARSE_INT
    mov     x10, x22            // width
    // Parse address expression
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x11, x22            // addr reg
    // Allocate result register
    bl      code_EXPR_ALLOC_REG
    mov     x12, x22            // rd
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, atom_load_arm64
    // GPU: LDG rd, [addr]
    str     x12, [x24, #-8]!
    mov     x22, x11
    bl      code_EXPR_EMIT_SASS_LDG
    mov     x22, x12
    b       atom_done
atom_load_arm64:
    // ARM64: dispatch on width
    // width 8 -> LDRB, width 32 -> LDR W, width 64 -> LDR X
    cmp     x10, #8
    b.ne    atom_load_a64_wide
    // LDRB: emit with zero offset reg (XZR acts as zero offset)
    str     x12, [x24, #-8]!
    str     x11, [x24, #-8]!
    mov     x22, #31            // XZR as offset
    bl      code_EXPR_EMIT_A64_LDRB_REG
    mov     x22, x12
    b       atom_done
atom_load_a64_wide:
    // LDR Xd, [Xbase, XZR] (zero offset via register)
    str     x12, [x24, #-8]!
    str     x11, [x24, #-8]!
    mov     x22, #31
    bl      code_EXPR_EMIT_A64_LDR_REG
    mov     x22, x12
    b       atom_done

atom_not_load:
    // --- Parenthesized expression: ( expr ) ---
    cmp     w5, #TOK_LPAREN
    b.ne    atom_not_paren
    // Consume (
    add     x1, x1, #1
    str     x1, [x0]
    // Parse inner expression
    bl      code_EXPR_COMPILE_EXPR_INNER
    // Consume )
    mov     x13, x22            // save result reg
    mov     x22, #TOK_RPAREN
    bl      code_EXPR_EXPECT
    mov     x22, x13
    b       atom_done

atom_not_paren:
    // --- Unary minus: - expr ---
    cmp     w5, #TOK_MINUS
    b.ne    atom_not_unary_neg
    // Consume -
    add     x1, x1, #1
    str     x1, [x0]
    // Parse operand
    bl      code_EXPR_COMPILE_ATOM
    mov     x10, x22            // src reg
    bl      code_EXPR_ALLOC_REG
    mov     x11, x22            // rd
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, atom_neg_arm64
    // GPU: IADD3 rd, RZ, -src, RZ (negate via subtract from zero)
    // Actually use IMAD rd, src, -1, RZ — but simpler to use FADD NEG
    // For simplicity, emit IADD3 RZ, RZ, src with NEG on src
    str     x22, [x24, #-8]!       // push IADD3_IMM opcode
    mov     x22, #0xFF             // rb = RZ
    str     x22, [x24, #-8]!
    mov     x22, x10               // ra = src (will be negated)
    str     x22, [x24, #-8]!
    mov     x22, x11               // rd
    str     x22, [x24, #-8]!
    mov     x22, #SASS_IADD3
    bl      code_EXPR_EMIT_SASS_ALU
    // TODO: set NEG bit in instruction — for now this is a placeholder
    mov     x22, x11
    b       atom_done
atom_neg_arm64:
    // ARM64: SUB Xd, XZR, Xsrc (NEG alias)
    str     x22, [x24, #-8]!
    mov     x22, x10            // rm = src
    str     x22, [x24, #-8]!
    mov     x22, #31            // rn = XZR
    str     x22, [x24, #-8]!
    mov     x22, x11            // rd
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_SUB_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x22, x11
    b       atom_done

atom_not_unary_neg:
    // Unknown atom — return -1 as error sentinel
    mov     x22, #-1

atom_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_mul ----
// Parse multiplicative: atom (* | / | % | >> | <<) atom
// ( -- reg )
.align 4
code_EXPR_COMPILE_MUL:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    bl      code_EXPR_COMPILE_ATOM
    mov     x14, x22            // left reg

mul_loop:
    // Peek at next token
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]

    // Check for * / % >> <<
    cmp     w5, #TOK_STAR
    b.eq    mul_op_star
    cmp     w5, #TOK_SLASH
    b.eq    mul_op_slash
    cmp     w5, #TOK_SHR
    b.eq    mul_op_shr
    cmp     w5, #TOK_SHL
    b.eq    mul_op_shl
    // Also check for % (modulo) — represented as TOK_IDENT with text "%"
    // Actually % is not in the lexer token list — skip for now
    // No more multiplicative operators — return left
    mov     x22, x14
    b       mul_done

mul_op_star:
    // Consume *
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_ATOM
    mov     x15, x22            // right reg
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22            // result reg
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, mul_star_arm64
    // GPU: FMUL rd, ra, rb
    str     x22, [x24, #-8]!       // rb
    mov     x22, x15
    str     x22, [x24, #-8]!       // ra (left)
    mov     x22, x14
    str     x22, [x24, #-8]!       // rd
    mov     x22, x16
    str     x22, [x24, #-8]!       // opcode
    mov     x22, #SASS_FMUL
    bl      code_EXPR_EMIT_SASS_ALU
    mov     x14, x16
    b       mul_loop
mul_star_arm64:
    // ARM64: MUL Xd, Xn, Xm
    str     x22, [x24, #-8]!       // rm = right
    mov     x22, x15
    str     x22, [x24, #-8]!       // rn = left
    mov     x22, x14
    str     x22, [x24, #-8]!       // rd
    mov     x22, x16
    str     x22, [x24, #-8]!       // opcode
    mov     x22, #ARM64_MUL
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       mul_loop

mul_op_slash:
    // Consume /
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_ATOM
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, mul_slash_arm64
    // GPU: MUFU.RCP then FMUL
    // Allocate temp for reciprocal
    bl      code_EXPR_ALLOC_REG
    mov     x17, x22           // rcp reg
    // MUFU.RCP rcp, right (opcode=SASS_MUFU, subop in ctrl)
    mov     x1, #SASS_MUFU
    orr     x1, x1, x17, lsl #16
    orr     x1, x1, x15, lsl #32
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    ldr     x5, =0x000e640000001000  // ctrl for MUFU.RCP
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    // FMUL rd, left, rcp
    str     x22, [x24, #-8]!
    mov     x22, x17            // rb = rcp
    str     x22, [x24, #-8]!
    mov     x22, x14            // ra = left
    str     x22, [x24, #-8]!
    mov     x22, x16            // rd
    str     x22, [x24, #-8]!
    mov     x22, #SASS_FMUL
    bl      code_EXPR_EMIT_SASS_ALU
    mov     x14, x16
    b       mul_loop
mul_slash_arm64:
    // ARM64: SDIV Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15            // rm = right
    str     x22, [x24, #-8]!
    mov     x22, x14            // rn = left
    str     x22, [x24, #-8]!
    mov     x22, x16            // rd
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_SDIV
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       mul_loop

mul_op_shr:
    // Consume >>
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_ATOM
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, mul_shr_arm64
    // GPU: SHF (shift funnel)
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #SASS_SHF
    bl      code_EXPR_EMIT_SASS_ALU
    mov     x14, x16
    b       mul_loop
mul_shr_arm64:
    // ARM64: LSR Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_LSR_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       mul_loop

mul_op_shl:
    // Consume <<
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_ATOM
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, mul_shl_arm64
    // GPU: SHF left direction
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #SASS_SHF
    bl      code_EXPR_EMIT_SASS_ALU
    mov     x14, x16
    b       mul_loop
mul_shl_arm64:
    // ARM64: LSL Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_LSL_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       mul_loop

mul_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_add ----
// Parse additive: mul (+ | -) mul
// ( -- reg )
.align 4
code_EXPR_COMPILE_ADD:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    bl      code_EXPR_COMPILE_MUL
    mov     x14, x22            // left reg

add_loop:
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]

    cmp     w5, #TOK_PLUS
    b.eq    add_op_plus
    cmp     w5, #TOK_MINUS
    b.eq    add_op_minus
    mov     x22, x14
    b       add_done

add_op_plus:
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_MUL
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, add_plus_arm64
    // GPU: FADD rd, left, right
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #SASS_FADD
    bl      code_EXPR_EMIT_SASS_ALU
    mov     x14, x16
    b       add_loop
add_plus_arm64:
    // ARM64: ADD Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_ADD_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       add_loop

add_op_minus:
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_MUL
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, add_minus_arm64
    // GPU: FADD rd, left, -right (NEG bit on src2)
    // Build instruction word with NEG bit (bit 63)
    mov     x1, #SASS_FADD
    orr     x1, x1, x16, lsl #16
    orr     x1, x1, x14, lsl #24
    orr     x1, x1, x15, lsl #32
    mov     x3, #1
    orr     x1, x1, x3, lsl #63    // NEG bit
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    mov     x5, #SASS_CTRL_DEFAULT
    movk    x5, #0x000f, lsl #48
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    mov     x14, x16
    b       add_loop
add_minus_arm64:
    // ARM64: SUB Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_SUB_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       add_loop

add_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_cmp ----
// Parse comparison: add (== | != | < | > | <= | >=) add
// ( -- reg )
.align 4
code_EXPR_COMPILE_CMP:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    bl      code_EXPR_COMPILE_ADD
    mov     x14, x22

    // Peek at next token
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]

    // Determine comparison condition
    cmp     w5, #TOK_EQEQ
    b.eq    cmp_eq
    cmp     w5, #TOK_NEQ
    b.eq    cmp_ne
    cmp     w5, #TOK_LT
    b.eq    cmp_lt
    cmp     w5, #TOK_GT
    b.eq    cmp_gt
    cmp     w5, #TOK_LTE
    b.eq    cmp_le
    cmp     w5, #TOK_GTE
    b.eq    cmp_ge
    // No comparison — return left as-is
    mov     x22, x14
    b       cmp_done

cmp_eq:
    mov     x17, #COND_EQ
    b       cmp_common
cmp_ne:
    mov     x17, #COND_NE
    b       cmp_common
cmp_lt:
    mov     x17, #COND_LT
    b       cmp_common
cmp_gt:
    mov     x17, #COND_GT
    b       cmp_common
cmp_le:
    mov     x17, #COND_LE
    b       cmp_common
cmp_ge:
    mov     x17, #COND_GE

cmp_common:
    // Consume the comparison token
    add     x1, x1, #1
    str     x1, [x0]
    // Parse right side
    stp     x14, x17, [sp, #-16]!
    bl      code_EXPR_COMPILE_ADD
    mov     x15, x22            // right reg
    ldp     x14, x17, [sp], #16
    // Allocate result register
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22            // result reg
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, cmp_arm64
    // GPU: ISETP/FSETP
    // Use ISETP for now: isetp pred, left, right
    // Then use SEL to materialize into integer register
    // For simplicity, use ISETP which sets a predicate,
    // then IMAD with predicate to select 1 or 0
    // Simplified: emit ISETP then use predicated MOV
    mov     x1, #SASS_ISETP
    orr     x1, x1, x14, lsl #24   // ra = left
    orr     x1, x1, x15, lsl #32   // rb = right
    // Predicate result in P0 position
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    mov     x5, #SASS_CTRL_DEFAULT
    movk    x5, #0x000f, lsl #48
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    // MOV-IMM result, 1 (predicated)
    str     x16, [x24, #-8]!
    mov     x22, #1
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    mov     x22, x16
    b       cmp_done
cmp_arm64:
    // ARM64: CMP + CSET
    str     x16, [x24, #-8]!       // push rd
    str     x14, [x24, #-8]!       // push rn
    str     x15, [x24, #-8]!       // push rm
    mov     x22, x17               // cond
    bl      code_EXPR_EMIT_A64_CMP_CSET
    mov     x22, x16

cmp_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_and ----
// Parse bitwise AND: cmp (& cmp)*
// ( -- reg )
.align 4
code_EXPR_COMPILE_AND:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    bl      code_EXPR_COMPILE_CMP
    mov     x14, x22

and_loop:
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]

    cmp     w5, #TOK_AMP
    b.ne    and_done_result
    // Consume &
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_CMP
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, and_arm64
    // GPU: LOP3 with AND truth table
    mov     x1, #SASS_LOP3
    orr     x1, x1, x16, lsl #16
    orr     x1, x1, x14, lsl #24
    orr     x1, x1, x15, lsl #32
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    // ctrl with LOP3_AND truth table in extra41[15:8]
    ldr     x5, =0x000fca00000000c0
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    mov     x14, x16
    b       and_loop
and_arm64:
    // ARM64: AND Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_AND_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       and_loop

and_done_result:
    mov     x22, x14
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_or ----
// Parse bitwise OR: and (| and)*
// ( -- reg )
.align 4
code_EXPR_COMPILE_OR:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    bl      code_EXPR_COMPILE_AND
    mov     x14, x22

or_loop:
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]

    cmp     w5, #TOK_PIPE
    b.ne    or_done_result
    // Consume |
    add     x1, x1, #1
    str     x1, [x0]
    stp     x14, xzr, [sp, #-16]!
    bl      code_EXPR_COMPILE_AND
    mov     x15, x22
    ldp     x14, xzr, [sp], #16
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, or_arm64
    // GPU: LOP3 with OR truth table
    mov     x1, #SASS_LOP3
    orr     x1, x1, x16, lsl #16
    orr     x1, x1, x14, lsl #24
    orr     x1, x1, x15, lsl #32
    adrp    x2, expr_sass_pos
    add     x2, x2, :lo12:expr_sass_pos
    ldr     x3, [x2]
    adrp    x4, expr_sass_buf
    add     x4, x4, :lo12:expr_sass_buf
    add     x4, x4, x3
    str     x1, [x4]
    ldr     x5, =0x000fca00000000fc    // LOP3_OR
    str     x5, [x4, #8]
    add     x3, x3, #16
    str     x3, [x2]
    mov     x14, x16
    b       or_loop
or_arm64:
    // ARM64: ORR Xd, Xn, Xm
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x14
    str     x22, [x24, #-8]!
    mov     x22, x16
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_ORR_REG
    bl      code_EXPR_EMIT_A64_RRR
    mov     x14, x16
    b       or_loop

or_done_result:
    mov     x22, x14
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ---- expr_compile_expr ----
// Top-level expression entry point.
// ( -- reg )
// Returns the register number holding the expression result.
.align 4
code_EXPR_COMPILE_EXPR:
    // Just delegate to the or-level (lowest precedence)
    b       code_EXPR_COMPILE_OR

// Inner entry point (callable from within atom for sub-expressions)
.align 4
code_EXPR_COMPILE_EXPR_INNER:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    bl      code_EXPR_COMPILE_OR
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    ret

// ============================================================
// STATEMENT COMPILATION
// ============================================================

// expr_compile_assign ( -- )
// Parse: name = expr
// Evaluates expr, stores result to name's register/location.
.align 4
code_EXPR_COMPILE_ASSIGN:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    // Current token should be TOK_IDENT (the name)
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    adrp    x2, expr_tok_buf
    add     x2, x2, :lo12:expr_tok_buf
    ldr     x2, [x2]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]
    cmp     w5, #TOK_IDENT
    b.ne    assign_not_ident

    // Get name
    ldr     w6, [x4, #4]       // offset
    ldr     w7, [x4, #8]       // length
    adrp    x8, expr_src_buf
    add     x8, x8, :lo12:expr_src_buf
    ldr     x8, [x8]
    add     x8, x8, x6         // name ptr

    // Consume name token
    add     x1, x1, #1
    str     x1, [x0]

    // Check next token for = (assignment)
    ldr     x1, [x0]
    mov     x3, #12
    mul     x4, x1, x3
    add     x4, x2, x4
    ldr     w5, [x4]
    cmp     w5, #TOK_EQ
    b.ne    assign_new_var

    // Assignment: name = expr
    // Consume =
    add     x1, x1, #1
    str     x1, [x0]

    // Save name info
    stp     x8, x7, [sp, #-16]!

    // Compile expression
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x10, x22            // expr result reg

    // Restore name info
    ldp     x8, x7, [sp], #16

    // Look up symbol
    str     x22, [x24, #-8]!
    mov     x22, x7
    str     x8, [x24, #-8]!
    bl      code_EXPR_SYM_FIND

    cmn     x22, #1
    b.eq    assign_new_sym

    // Known symbol — move result into its register
    bl      code_EXPR_SYM_REG
    mov     x11, x22            // target reg
    // Emit MOV target, result
    str     x11, [x24, #-8]!
    mov     x22, x10
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, assign_mov_arm64
    // GPU: IMAD rd, rs, 1, RZ (register move idiom)
    bl      code_EXPR_EMIT_SASS_MOV_IMM      // placeholder — should be MOV reg
    b       assign_done
assign_mov_arm64:
    bl      code_EXPR_EMIT_A64_MOV_REG
    b       assign_done

assign_new_sym:
    // New symbol — allocate register, add to table
    bl      code_EXPR_ALLOC_REG
    mov     x11, x22            // new reg
    // MOV new_reg, result_reg
    str     x11, [x24, #-8]!
    mov     x22, x10
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, assign_newsym_arm64
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    b       assign_newsym_add
assign_newsym_arm64:
    bl      code_EXPR_EMIT_A64_MOV_REG
assign_newsym_add:
    // Add to symbol table
    str     x8, [x24, #-8]!    // addr
    str     x7, [x24, #-8]!    // len
    mov     x22, x11            // reg
    str     x22, [x24, #-8]!
    mov     x22, #2             // kind = local-f32
    str     x22, [x24, #-8]!
    mov     x22, x11
    bl      code_EXPR_SYM_ADD
    b       assign_done

assign_new_var:
    // Name not followed by = : this is a Lithos-style implicit assignment
    // name expr  — means name = expr
    stp     x8, x7, [sp, #-16]!
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x10, x22
    ldp     x8, x7, [sp], #16
    // Look up or create
    str     x22, [x24, #-8]!
    mov     x22, x7
    str     x8, [x24, #-8]!
    bl      code_EXPR_SYM_FIND
    cmn     x22, #1
    b.eq    assign_newvar_create
    // Known — move
    bl      code_EXPR_SYM_REG
    mov     x11, x22
    str     x11, [x24, #-8]!
    mov     x22, x10
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, assign_newvar_arm64
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    b       assign_done
assign_newvar_arm64:
    bl      code_EXPR_EMIT_A64_MOV_REG
    b       assign_done
assign_newvar_create:
    bl      code_EXPR_ALLOC_REG
    mov     x11, x22
    str     x11, [x24, #-8]!
    mov     x22, x10
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, assign_newvar_create_arm64
    bl      code_EXPR_EMIT_SASS_MOV_IMM
    b       assign_newvar_add
assign_newvar_create_arm64:
    bl      code_EXPR_EMIT_A64_MOV_REG
assign_newvar_add:
    str     x8, [x24, #-8]!
    str     x7, [x24, #-8]!
    mov     x22, x11
    str     x22, [x24, #-8]!
    mov     x22, #2
    str     x22, [x24, #-8]!
    mov     x22, x11
    bl      code_EXPR_SYM_ADD
    b       assign_done

assign_not_ident:
    // Not an ident — could be ← (store)
    cmp     w5, #TOK_STORE
    b.ne    assign_done
    // ← width addr val
    add     x1, x1, #1
    str     x1, [x0]
    // Parse width
    bl      code_EXPR_PARSE_INT
    mov     x10, x22            // width
    // Parse addr
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x11, x22            // addr reg
    // Parse val
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x12, x22            // val reg
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, store_arm64
    // GPU: STG [addr], val
    str     x12, [x24, #-8]!
    mov     x22, x11
    bl      code_EXPR_EMIT_SASS_STG
    b       assign_done
store_arm64:
    // ARM64: STR val, [addr, XZR]
    str     x12, [x24, #-8]!
    str     x11, [x24, #-8]!
    mov     x22, #31            // XZR offset
    bl      code_EXPR_EMIT_A64_STR_REG

assign_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ============================================================
// COMPOSITION CALL COMPILATION
// ============================================================

// expr_compile_call ( name_ptr name_len -- reg )
// Compile a composition call: name arg1 arg2 ...
// Evaluates arguments, emits BL (ARM64) or inlines body (GPU).
.align 4
code_EXPR_COMPILE_CALL:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    // TOS = name_len, NOS = name_ptr
    POP     x8              // name_ptr
    mov     x7, x22         // name_len

    // Look up in composition table
    adrp    x0, ls_comp_count
    add     x0, x0, :lo12:ls_comp_count
    ldr     x1, [x0]
    adrp    x2, ls_comp_table
    add     x2, x2, :lo12:ls_comp_table
    mov     x3, #0          // search index
call_search:
    cmp     x3, x1
    b.ge    call_not_found
    mov     x4, #COMP_ENTRY_SIZE
    mul     x5, x3, x4
    add     x5, x2, x5
    // Check name length
    ldr     w6, [x5, #32]
    cmp     w6, w7
    b.ne    call_search_next
    // Compare name bytes
    mov     x9, #0
call_name_cmp:
    cmp     x9, x7
    b.ge    call_found
    ldrb    w10, [x5, x9]
    ldrb    w11, [x8, x9]
    cmp     w10, w11
    b.ne    call_search_next
    add     x9, x9, #1
    b       call_name_cmp
call_search_next:
    add     x3, x3, #1
    b       call_search

call_found:
    // Found composition at index x3
    // Get arg_count and token start
    ldr     w12, [x5, #36]     // arg_count
    ldr     w13, [x5, #40]     // tok_start

    // Evaluate arguments into consecutive registers (X0-X7 on ARM64)
    mov     x14, #0             // arg counter
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
call_args_loop:
    cmp     x14, x12
    b.ge    call_args_done
    // Save state
    stp     x12, x14, [sp, #-16]!
    stp     x13, x0, [sp, #-16]!
    // Compile argument expression
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x15, x22            // arg value reg
    ldp     x13, x0, [sp], #16
    ldp     x12, x14, [sp], #16
    // Move arg value to argument register
    cbnz    x0, call_arg_arm64
    // GPU: arguments go into sequential registers (handled by inlining)
    b       call_arg_next
call_arg_arm64:
    // ARM64: move to X0+n
    stp     x12, x14, [sp, #-16]!
    stp     x13, x0, [sp, #-16]!
    str     x14, [x24, #-8]!   // rd = arg index
    mov     x22, x15            // rs = value reg
    bl      code_EXPR_EMIT_A64_MOV_REG
    ldp     x13, x0, [sp], #16
    ldp     x12, x14, [sp], #16
call_arg_next:
    add     x14, x14, #1
    b       call_args_loop

call_args_done:
    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, call_emit_arm64
    // GPU: inline the composition body (not implemented here — placeholder)
    // Would save/restore tok_pos, replay from tok_start
    bl      code_EXPR_ALLOC_REG
    b       call_done
call_emit_arm64:
    // ARM64: emit BL to the composition's code offset
    // For now, emit a BL placeholder (offset 0, to be patched)
    mov     x22, #0
    bl      code_EXPR_EMIT_A64_BL
    // Result is in X0 after call returns
    bl      code_EXPR_ALLOC_REG
    mov     x16, x22
    str     x16, [x24, #-8]!
    mov     x22, #0             // MOV Xresult, X0
    bl      code_EXPR_EMIT_A64_MOV_REG
    mov     x22, x16
    b       call_done

call_not_found:
    // Unknown composition — allocate a register and return it
    bl      code_EXPR_ALLOC_REG

call_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ============================================================
// COMPOSITION REGISTRATION
// ============================================================

// expr_register_comp ( name_ptr name_len arg_count tok_start -- )
// Register a composition in the composition table for later calls.
.align 4
code_EXPR_REGISTER_COMP:
    // TOS = tok_start
    POP     x3          // arg_count
    POP     x1          // name_len
    POP     x0          // name_ptr
    adrp    x4, ls_comp_count
    add     x4, x4, :lo12:ls_comp_count
    ldr     x5, [x4]
    adrp    x6, ls_comp_table
    add     x6, x6, :lo12:ls_comp_table
    mov     x7, #COMP_ENTRY_SIZE
    mul     x8, x5, x7
    add     x8, x6, x8
    // Copy name
    mov     x9, #0
    cmp     x1, #31
    b.le    1f
    mov     x1, #31
1:  cbz     x1, 3f
2:  ldrb    w10, [x0, x9]
    strb    w10, [x8, x9]
    add     x9, x9, #1
    cmp     x9, x1
    b.lt    2b
3:  str     w1, [x8, #32]      // name_len
    str     w3, [x8, #36]      // arg_count
    str     w22, [x8, #40]     // tok_start
    add     x5, x5, #1
    str     x5, [x4]
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// ARRAY INDEX STORE
// ============================================================

// expr_compile_array_store ( -- )
// Parse: name [ index_expr ] = value_expr
// Emits the store instruction.
.align 4
code_EXPR_COMPILE_ARRAY_STORE:
    stp     x26, x30, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!

    // Assume name has been consumed and its sym index is known.
    // TOS = sym_index
    mov     x14, x22        // sym index

    // Consume [
    adrp    x0, expr_tok_pos
    add     x0, x0, :lo12:expr_tok_pos
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]

    // Compile index expression
    str     x14, [sp, #-16]!
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x15, x22        // index reg
    ldr     x14, [sp], #16

    // Consume ]
    mov     x22, #TOK_RBRACK
    bl      code_EXPR_EXPECT

    // Consume =
    mov     x22, #TOK_EQ
    bl      code_EXPR_EXPECT

    // Compile value expression
    stp     x14, x15, [sp, #-16]!
    bl      code_EXPR_COMPILE_EXPR_INNER
    mov     x16, x22        // val reg
    ldp     x14, x15, [sp], #16

    // Get base register from symbol table
    mov     x22, x14
    bl      code_EXPR_SYM_REG
    mov     x17, x22        // base reg

    // Compute address = base + index
    bl      code_EXPR_ALLOC_REG
    mov     x18, x22        // addr reg

    adrp    x0, expr_target
    add     x0, x0, :lo12:expr_target
    ldr     x0, [x0]
    cbnz    x0, array_store_arm64

    // GPU: IADD3 addr, base, index, RZ then STG [addr], val
    str     x22, [x24, #-8]!
    mov     x22, #0xFF         // rb = RZ
    str     x22, [x24, #-8]!
    mov     x22, x15           // ra = index (second operand)
    str     x22, [x24, #-8]!
    mov     x22, x17           // rd's first src = base
    // Actually: IADD3 addr, base, index, RZ
    str     x22, [x24, #-8]!
    mov     x22, x18
    str     x22, [x24, #-8]!
    mov     x22, #SASS_IADD3
    bl      code_EXPR_EMIT_SASS_ALU
    // STG [addr], val
    str     x16, [x24, #-8]!
    mov     x22, x18
    bl      code_EXPR_EMIT_SASS_STG
    b       array_store_done

array_store_arm64:
    // ARM64: ADD addr, base, index then STR val, [addr, XZR]
    str     x22, [x24, #-8]!
    mov     x22, x15
    str     x22, [x24, #-8]!
    mov     x22, x17
    str     x22, [x24, #-8]!
    mov     x22, x18
    str     x22, [x24, #-8]!
    mov     x22, #ARM64_ADD_REG
    bl      code_EXPR_EMIT_A64_RRR
    // STR val, [addr, XZR]
    str     x16, [x24, #-8]!
    str     x18, [x24, #-8]!
    mov     x22, #31
    bl      code_EXPR_EMIT_A64_STR_REG

array_store_done:
    ldp     x24, x23, [sp], #16
    ldp     x26, x30, [sp], #16
    NEXT

// ============================================================
// QUERY FUNCTIONS
// ============================================================

// expr_get_arm64_buf ( -- addr len )
// Return the ARM64 code buffer address and current length.
.align 4
code_EXPR_GET_ARM64_BUF:
    str     x22, [x24, #-8]!
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    str     x0, [x24, #-8]!
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]
    NEXT

// expr_get_sass_buf ( -- addr len )
// Return the SASS code buffer address and current length.
.align 4
code_EXPR_GET_SASS_BUF:
    str     x22, [x24, #-8]!
    adrp    x0, expr_sass_buf
    add     x0, x0, :lo12:expr_sass_buf
    str     x0, [x24, #-8]!
    adrp    x1, expr_sass_pos
    add     x1, x1, :lo12:expr_sass_pos
    ldr     x22, [x1]
    NEXT

// expr_get_max_reg ( -- n )
// Return the high-water register count.
.align 4
code_EXPR_GET_MAX_REG:
    str     x22, [x24, #-8]!
    adrp    x0, expr_max_reg
    add     x0, x0, :lo12:expr_max_reg
    ldr     x22, [x0]
    NEXT

// ============================================================
// STRING CONSTANTS
// ============================================================
.data
.align 3

expr_err_expect:
    .ascii  "expr: unexpected token\n\0"

.end
