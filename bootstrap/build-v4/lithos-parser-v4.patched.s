.global code_PARSE_TOKENS
.global entry_p_parse_tokens
.global parse_advance
.global parse_dollar_reg
.global parse_emit32
.global parse_emit_add
.global parse_emit_and
.global parse_emit_cur
.global parse_emit_eor
.global parse_emit_ldr_w_zero
.global parse_emit_ldr_x_zero
.global parse_emit_ldrb_zero
.global parse_emit_ldrh_zero
.global parse_emit_lsl
.global parse_emit_lsr
.global parse_emit_mov_imm16
.global parse_emit_mov_imm64
.global parse_emit_mov_reg
.global parse_emit_movk16
.global parse_emit_mul
.global parse_emit_nop
.global parse_emit_orr
.global parse_emit_ret
.global parse_emit_sdiv
.global parse_emit_str_w_zero
.global parse_emit_str_x_zero
.global parse_emit_strb_zero
.global parse_emit_strh_zero
.global parse_emit_sub
.global parse_emit_svc
.global parse_skip_nl
.global parse_tokens
.global parse_v4_alloc_reg
.global parse_v4_body
.global parse_v4_comp_or_stmt
.global parse_v4_composition
.global parse_v4_emit_ptr
.global parse_v4_err_msg
.global parse_v4_err_regspill
.global parse_v4_error
.global parse_v4_error_regspill
.global parse_v4_int_literal
.global parse_v4_line
.global parse_v4_line_rest
.global parse_v4_main_offset
.global parse_v4_next_reg
.global parse_v4_reset_regs
.global parse_v4_scope_depth
.global parse_v4_skip_to_eol
.global parse_v4_sym_add
.global parse_v4_sym_lookup
.global parse_v4_sym_pop
.global parse_v4_toplevel
.global parse_v4_value_token
.global parse_v4_vs_sp
.global parse_v4_vs_stack

// lithos-parser-v4.s — Stack-language parser for Lithos .ls source files
//
// Fourth independent implementation. Clean-room from the language spec.
//
// Design: Lithos is a stack language. The compiler maintains a virtual
// register stack (array of register numbers + stack pointer). Each line
// does one thing to the stack. No precedence climbing. No expression
// trees. One token dispatches one action.
//
// parse_line logic is a switch on the first token of each body line.
//
// Threading: Native ARM64 subroutine calls (bl/ret), not DTC.
//   Called FROM the DTC bootstrap via code_PARSE_TOKENS wrapper.
//
// Register conventions (inherited from lithos-bootstrap.s):
//   X26 = IP   (DTC instruction pointer)
//   X25 = W    (DTC working register)
//   X24 = DSP  (data stack pointer)
//   X23 = RSP  (return stack pointer)
//   X22 = TOS  (top of data stack)
//   X20 = HERE (dictionary/code-space pointer)
//   X21 = BASE
//
// Parser-private registers (callee-saved):
//   X19 = TOKP   — pointer to current token triple
//   X27 = TOKEND — pointer past last token
//   X28 = SRC    — pointer to source buffer

// ============================================================
// Token type constants (from lithos-lexer.s)
// ============================================================
.equ TOK_EOF,       0
.equ TOK_NEWLINE,   1
.equ TOK_INDENT,    2
.equ TOK_INT,       3
.equ TOK_FLOAT,     4
.equ TOK_IDENT,     5
.equ TOK_STRING,    6
.equ TOK_KERNEL,    11
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
.equ TOK_LABEL,     33
.equ TOK_EXIT_KW,   34
.equ TOK_HOST,      35
.equ TOK_LOAD,      36      // →
.equ TOK_STORE,     37      // ←
.equ TOK_REG_READ,  38      // ↑
.equ TOK_REG_WRITE, 39      // ↓
.equ TOK_TRAP,      89      // trap
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
.equ TOK_LBRACK,    67
.equ TOK_RBRACK,    68
.equ TOK_LPAREN,    69
.equ TOK_RPAREN,    70
.equ TOK_COLON,     72
.equ TOK_SUM,       75      // Σ
.equ TOK_MAX,       76      // △
.equ TOK_MIN,       77      // ▽
.equ TOK_INDEX,     78      // #
.equ TOK_SQRT,      79      // √
.equ TOK_SIN,       80      // ≅
.equ TOK_COS,       81      // ≡

.equ TOK_STRIDE_SZ, 12

// ============================================================
// Symbol table entry layout
// ============================================================
.equ SYM_SIZE,      24
.equ SYM_NAME_OFF,  0
.equ SYM_NAME_LEN,  4
.equ SYM_KIND,      8
.equ SYM_REG,       12
.equ SYM_DEPTH,     16

.equ KIND_LOCAL_REG, 0
.equ KIND_PARAM,     1
.equ KIND_COMP,      4

.equ MAX_SYMS,      512

// Virtual register stack: tracks which ARM64 register holds each
// stack slot. vs_stack[0..vs_sp-1] are live slots.
// vs_stack[i] = ARM64 register number (9-15).
.equ VS_MAX,        32

// Register allocator range
.equ REG_FIRST,     9
.equ REG_LAST,      15

// ARM64 instruction constants
.equ ARM64_NOP,     0xD503201F
.equ ARM64_RET,     0xD65F03C0
.equ ARM64_SVC_0,   0xD4000001

// ORRIMM macro — OR a 32-bit immediate into Wd via tmp register
.macro ORRIMM Wd, imm, Wtmp
    mov     \Wtmp, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wtmp, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
    orr     \Wd, \Wd, \Wtmp
.endm

.macro MOVI32 Wd, imm
    mov     \Wd, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wd, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
.endm

// DTC NEXT macro
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm

// ============================================================
// Extern declarations
// ============================================================
.extern ls_code_buf
.extern ls_code_pos
.extern ls_token_buf
.extern ls_token_count
.extern ls_sym_table
.extern ls_sym_count
.extern ls_data_buf
.extern ls_data_pos

// ============================================================
// Global declarations — only what v4 needs
// ============================================================

// ============================================================
// .text — Parser code
// ============================================================
.text
.align 4

// (v4 core — no aliases needed for standalone build)

// ============================================================
// Token accessors
// ============================================================
// tok_type: return type of current token in W0
tok_type:
    ldr     w0, [x19]
    ret

// tok_text: return (x0=ptr, w1=len) of current token's text
tok_text:
    ldr     w1, [x19, #8]
    ldr     w0, [x19, #4]
    add     x0, x28, x0
    ret

// advance: move to next token
parse_advance:
    add     x19, x19, #TOK_STRIDE_SZ
    ret

// skip_newlines: advance past TOK_NEWLINE and TOK_INDENT
parse_skip_nl:
1:  cmp     x19, x27
    b.hs    2f
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    3f
    cmp     w0, #TOK_INDENT
    b.eq    3f
2:  ret
3:  add     x19, x19, #TOK_STRIDE_SZ
    b       1b

// ============================================================
// emit32 — emit a 32-bit instruction word
//   w0 = instruction word
// ============================================================
parse_emit32:
    adrp    x1, parse_v4_emit_ptr
    add     x1, x1, :lo12:parse_v4_emit_ptr
    ldr     x2, [x1]
    str     w0, [x2], #4
    str     x2, [x1]
    ret

// ============================================================
// emit_cur — return current emit address in x0
// ============================================================
parse_emit_cur:
    adrp    x0, parse_v4_emit_ptr
    add     x0, x0, :lo12:parse_v4_emit_ptr
    ldr     x0, [x0]
    ret

// ============================================================
// emit_mov_imm16 — MOVZ Xd, #imm16
//   w0 = dest reg, w1 = imm16
// ============================================================
parse_emit_mov_imm16:
    and     w2, w1, #0xFFFF
    lsl     w2, w2, #5
    orr     w2, w2, w0          // Rd field
    ORRIMM  w2, 0xD2800000, w16
    mov     w0, w2
    b       parse_emit32

// ============================================================
// emit_movk_imm16 — MOVK Xd, #imm16, LSL #shift
//   w0 = dest reg, w1 = imm16, w2 = shift (0, 16, 32, 48)
// ============================================================
parse_emit_movk16:
    lsr     w3, w2, #4          // hw field
    lsl     w3, w3, #21
    and     w4, w1, #0xFFFF
    lsl     w4, w4, #5
    orr     w4, w4, w0
    orr     w4, w4, w3
    ORRIMM  w4, 0xF2800000, w16
    mov     w0, w4
    b       parse_emit32

// ============================================================
// emit_mov_imm64 — load full 64-bit immediate into Xd
//   w0 = dest reg, x1 = imm64
// ============================================================
parse_emit_mov_imm64:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!

    // MOVZ Xd, #(lo16)
    and     w2, w1, #0xFFFF
    mov     w3, w0
    mov     w1, w2
    mov     w0, w3
    bl      parse_emit_mov_imm16

    ldp     x0, x1, [sp]

    // MOVK Xd, #((imm>>16) & 0xFFFF), LSL #16
    lsr     x4, x1, #16
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv4_mov64_c32
    mov     w1, w5
    mov     w2, #16
    bl      parse_emit_movk16
    ldp     x0, x1, [sp]

.Lv4_mov64_c32:
    lsr     x4, x1, #32
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv4_mov64_c48
    mov     w1, w5
    mov     w2, #32
    bl      parse_emit_movk16
    ldp     x0, x1, [sp]

.Lv4_mov64_c48:
    lsr     x4, x1, #48
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv4_mov64_done
    mov     w1, w5
    mov     w2, #48
    bl      parse_emit_movk16

.Lv4_mov64_done:
    add     sp, sp, #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// emit_mov_reg — MOV Xd, Xm  (ORR Xd, XZR, Xm)
//   w0 = d, w1 = m
// ============================================================
parse_emit_mov_reg:
    lsl     w2, w1, #16         // Rm
    mov     w3, #31
    lsl     w3, w3, #5          // Rn = XZR
    orr     w4, w0, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    b       parse_emit32

// ============================================================
// Arithmetic emit helpers
//   w0 = d, w1 = n, w2 = m
// ============================================================
parse_emit_add:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8B000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_sub:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCB000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_mul:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9B007C00, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_sdiv:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC00C00, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_and:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8A000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_orr:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xAA000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_eor:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCA000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_lsl:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_lsr:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02400, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Memory emit helpers — LDR/STR with zero offset
//   w0 = Rt, w1 = Rn
// ============================================================
parse_emit_ldrb_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39400000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_strb_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_ldrh_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79400000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_strh_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_ldr_w_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9400000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_str_w_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_ldr_x_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9400000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

parse_emit_str_x_zero:
    stp     x30, xzr, [sp, #-16]!
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9000000, w16
    mov     w0, w5
    bl      parse_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// emit_svc — SVC #0
// ============================================================
parse_emit_svc:
    MOVI32  w0, 0xD4000001
    b       parse_emit32

// ============================================================
// emit_ret — RET
// ============================================================
parse_emit_ret:
    MOVI32  w0, 0xD65F03C0
    b       parse_emit32

// ============================================================
// emit_nop — NOP
// ============================================================
parse_emit_nop:
    MOVI32  w0, 0xD503201F
    b       parse_emit32

// ============================================================
// Virtual register stack operations
// ============================================================
// The virtual stack tracks which ARM64 registers hold stack values.
// vs_stack is an array of register numbers. vs_sp is the count.

// vs_push — push a register onto the virtual stack
//   w0 = register number to push
vs_push:
    adrp    x1, parse_v4_vs_sp
    add     x1, x1, :lo12:parse_v4_vs_sp
    ldr     w2, [x1]
    adrp    x3, parse_v4_vs_stack
    add     x3, x3, :lo12:parse_v4_vs_stack
    str     w0, [x3, x2, lsl #2]
    add     w2, w2, #1
    str     w2, [x1]
    ret

// vs_pop — pop top register from virtual stack, return in w0
vs_pop:
    adrp    x1, parse_v4_vs_sp
    add     x1, x1, :lo12:parse_v4_vs_sp
    ldr     w2, [x1]
    cbz     w2, parse_v4_error
    sub     w2, w2, #1
    str     w2, [x1]
    adrp    x3, parse_v4_vs_stack
    add     x3, x3, :lo12:parse_v4_vs_stack
    ldr     w0, [x3, x2, lsl #2]
    ret

// vs_top — peek top register, return in w0
vs_top:
    adrp    x1, parse_v4_vs_sp
    add     x1, x1, :lo12:parse_v4_vs_sp
    ldr     w2, [x1]
    cbz     w2, parse_v4_error
    sub     w2, w2, #1
    adrp    x3, parse_v4_vs_stack
    add     x3, x3, :lo12:parse_v4_vs_stack
    ldr     w0, [x3, x2, lsl #2]
    ret

// vs_depth — return stack depth in w0
vs_depth:
    adrp    x1, parse_v4_vs_sp
    add     x1, x1, :lo12:parse_v4_vs_sp
    ldr     w0, [x1]
    ret

// vs_reset — clear virtual stack
vs_reset:
    adrp    x1, parse_v4_vs_sp
    add     x1, x1, :lo12:parse_v4_vs_sp
    str     wzr, [x1]
    ret

// ============================================================
// Register allocator — trivial linear, X9-X15
// ============================================================
parse_v4_alloc_reg:
    adrp    x0, parse_v4_next_reg
    add     x0, x0, :lo12:parse_v4_next_reg
    ldr     w1, [x0]
    cmp     w1, #REG_LAST
    b.gt    parse_v4_error_regspill
    add     w2, w1, #1
    str     w2, [x0]
    mov     w0, w1
    ret

parse_v4_reset_regs:
    adrp    x0, parse_v4_next_reg
    add     x0, x0, :lo12:parse_v4_next_reg
    mov     w1, #REG_FIRST
    str     w1, [x0]
    ret

// ============================================================
// Symbol table operations
// ============================================================

// sym_lookup — find symbol matching current token
//   Returns: x0 = pointer to sym entry, or 0 if not found
parse_v4_sym_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w0, [x19, #4]      // token offset
    ldr     w1, [x19, #8]      // token length

    adrp    x2, ls_sym_count
    add     x2, x2, :lo12:ls_sym_count
    ldr     w3, [x2]
    cbz     w3, .Lv4sl_notfound

    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table

    sub     w3, w3, #1
    mov     w5, #SYM_SIZE
    madd    x6, x3, x5, x4

.Lv4sl_loop:
    ldr     w7, [x6, #SYM_NAME_LEN]
    cmp     w7, w1
    b.ne    .Lv4sl_next

    ldr     w7, [x6, #SYM_NAME_OFF]
    add     x7, x28, x7
    add     x2, x28, x0
    mov     w5, #0
.Lv4sl_cmp:
    cmp     w5, w1
    b.ge    .Lv4sl_found
    ldrb    w3, [x7, x5]
    ldrb    w4, [x2, x5]
    cmp     w3, w4
    b.ne    .Lv4sl_next2
    add     w5, w5, #1
    b       .Lv4sl_cmp

.Lv4sl_found:
    mov     x0, x6
    b       .Lv4sl_done

.Lv4sl_next2:
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
.Lv4sl_next:
    cmp     x6, x4
    b.ls    .Lv4sl_notfound
    sub     x6, x6, #SYM_SIZE
    b       .Lv4sl_loop

.Lv4sl_notfound:
    mov     x0, #0

.Lv4sl_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// sym_add — add symbol for current token
//   w1 = kind, w2 = reg_or_offset, w3 = scope_depth
//   Returns: x0 = pointer to new entry
parse_v4_sym_add:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!

    adrp    x4, ls_sym_count
    add     x4, x4, :lo12:ls_sym_count
    ldr     w5, [x4]

    adrp    x0, ls_sym_table
    add     x0, x0, :lo12:ls_sym_table
    mov     w6, #SYM_SIZE
    madd    x0, x5, x6, x0

    ldr     w6, [x19, #4]
    str     w6, [x0, #SYM_NAME_OFF]
    ldr     w6, [x19, #8]
    str     w6, [x0, #SYM_NAME_LEN]
    str     w1, [x0, #SYM_KIND]
    str     w2, [x0, #SYM_REG]
    str     w3, [x0, #SYM_DEPTH]

    add     w5, w5, #1
    str     w5, [x4]

    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// sym_pop_scope — remove all symbols with scope_depth > w0
parse_v4_sym_pop:
    stp     x30, xzr, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    mov     w2, w0

    adrp    x3, ls_sym_count
    add     x3, x3, :lo12:ls_sym_count
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table

.Lv4pop_loop:
    ldr     w0, [x3]
    cbz     w0, .Lv4pop_done
    sub     w0, w0, #1
    mov     w1, #SYM_SIZE
    madd    x1, x0, x1, x4
    ldr     w1, [x1, #SYM_DEPTH]
    cmp     w1, w2
    b.le    .Lv4pop_done
    str     w0, [x3]
    b       .Lv4pop_loop

.Lv4pop_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// parse_int_literal — parse integer from current token text
//   Returns value in x0. Does NOT advance token pointer.
// ============================================================
parse_v4_int_literal:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x1, x28, x1

    mov     x0, #0
    mov     w3, #0
    mov     w7, #0              // negative flag

    ldrb    w4, [x1]
    cmp     w4, #'-'
    b.ne    .Lv4il_chex
    mov     w7, #1
    add     w3, w3, #1

.Lv4il_chex:
    sub     w5, w2, w3
    cmp     w5, #2
    b.lt    .Lv4il_dec
    ldrb    w4, [x1, x3]
    cmp     w4, #'0'
    b.ne    .Lv4il_dec
    add     w6, w3, #1
    ldrb    w4, [x1, x6]
    cmp     w4, #'x'
    b.eq    .Lv4il_hex
    cmp     w4, #'X'
    b.ne    .Lv4il_dec

.Lv4il_hex:
    add     w3, w3, #2
.Lv4il_hexl:
    cmp     w3, w2
    b.ge    .Lv4il_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'9'
    b.le    .Lv4il_h09
    cmp     w4, #'F'
    b.le    .Lv4il_hAF
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lv4il_hacc
.Lv4il_hAF:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lv4il_hacc
.Lv4il_h09:
    sub     w4, w4, #'0'
.Lv4il_hacc:
    lsl     x0, x0, #4
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv4il_hexl

.Lv4il_dec:
.Lv4il_decl:
    cmp     w3, w2
    b.ge    .Lv4il_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'.'
    b.eq    .Lv4il_sign
    sub     w4, w4, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv4il_decl

.Lv4il_sign:
    cbz     w7, .Lv4il_done
    neg     x0, x0
.Lv4il_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_dollar_reg — parse $N token, return register number in w0
//   Current token must be TOK_IDENT with text "$N".
//   Parses the number after '$'. Does NOT advance.
// ============================================================
parse_dollar_reg:
    stp     x30, xzr, [sp, #-16]!
    ldr     w1, [x19, #4]      // offset
    ldr     w2, [x19, #8]      // length
    add     x1, x28, x1        // text ptr

    // Skip '$'
    ldrb    w3, [x1]
    cmp     w3, #'$'
    b.ne    parse_v4_error      // not a $N token

    mov     x0, #0
    mov     w4, #1              // start after $
.Lv4dr_loop:
    cmp     w4, w2
    b.ge    .Lv4dr_done
    ldrb    w3, [x1, x4]
    sub     w3, w3, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x3
    add     w4, w4, #1
    b       .Lv4dr_loop
.Lv4dr_done:
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// is_dollar_token — check if current token text starts with '$'
//   Returns w0 = 1 if yes, 0 if no
// ============================================================
is_dollar_token:
    ldr     w1, [x19, #4]      // offset
    ldr     w2, [x19, #8]      // length
    cbz     w2, .Lv4idt_no
    add     x1, x28, x1
    ldrb    w3, [x1]
    cmp     w3, #'$'
    b.ne    .Lv4idt_no
    mov     w0, #1
    ret
.Lv4idt_no:
    mov     w0, #0
    ret

// ============================================================
// is_end_of_line — check if current token ends a line
//   Returns w0 = 1 if NEWLINE/EOF/end-of-buffer/INDENT, 0 otherwise
// ============================================================
is_eol:
    cmp     x19, x27
    b.hs    .Lv4eol_yes
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv4eol_yes
    cmp     w0, #TOK_EOF
    b.eq    .Lv4eol_yes
    mov     w0, #0
    ret
.Lv4eol_yes:
    mov     w0, #1
    ret

// ============================================================
// PARSER ENTRY POINT
// ============================================================
// parse_tokens — main entry, called from DTC or native code
//   x0 = pointer to token buffer (u32 triples)
//   x1 = token count
//   x2 = pointer to source buffer
.global parse_tokens
.align 4
parse_tokens:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x27, [sp, #-16]!
    stp     x28, x20, [sp, #-16]!

    mov     x19, x0
    mov     w3, #TOK_STRIDE_SZ
    mul     x27, x1, x3
    add     x27, x27, x0
    mov     x28, x2

    // Initialize emit pointer to code buffer
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, parse_v4_emit_ptr
    add     x1, x1, :lo12:parse_v4_emit_ptr
    str     x0, [x1]

    // Clear symbol table
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    str     wzr, [x0]

    // Clear scope depth
    adrp    x0, parse_v4_scope_depth
    add     x0, x0, :lo12:parse_v4_scope_depth
    str     wzr, [x0]

    // Reset register allocator
    bl      parse_v4_reset_regs

    // Reset virtual stack
    bl      vs_reset

    // Parse top-level declarations
    bl      parse_v4_toplevel

    // Sync ls_code_pos
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, parse_v4_emit_ptr
    add     x1, x1, :lo12:parse_v4_emit_ptr
    ldr     x1, [x1]
    sub     x1, x1, x0
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    str     x1, [x2]

    ldp     x28, x20, [sp], #16
    ldp     x19, x27, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_toplevel — loop over top-level declarations
// ============================================================
parse_v4_toplevel:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Lv4top_loop:
    cmp     x19, x27
    b.hs    .Lv4top_done

    bl      parse_skip_nl
    cmp     x19, x27
    b.hs    .Lv4top_done

    ldr     w0, [x19]

    cmp     w0, #TOK_EOF
    b.eq    .Lv4top_done

    // IDENT — could be composition definition (name args :) or statement
    cmp     w0, #TOK_IDENT
    b.eq    .Lv4top_maybe_comp

    // Statement tokens at top level
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_LOAD
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_STORE
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_TRAP
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_INT
    b.eq    .Lv4top_stmt

    // Operator tokens at top level (bare ops in stack language)
    cmp     w0, #TOK_PLUS
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_MINUS
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_STAR
    b.eq    .Lv4top_stmt
    cmp     w0, #TOK_SLASH
    b.eq    .Lv4top_stmt

    // Skip unknown
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4top_loop

.Lv4top_stmt:
    bl      parse_v4_line
    b       .Lv4top_loop

.Lv4top_maybe_comp:
    bl      parse_v4_comp_or_stmt
    b       .Lv4top_loop

.Lv4top_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_comp_or_stmt — disambiguate composition def vs statement
//   Lookahead: scan for COLON before NEWLINE/EOF on same line
// ============================================================
parse_v4_comp_or_stmt:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, xzr, [sp, #-16]!

    mov     x4, x19
.Lv4cos_scan:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lv4cos_stmt
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lv4cos_comp
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv4cos_stmt
    cmp     w0, #TOK_EOF
    b.eq    .Lv4cos_stmt
    b       .Lv4cos_scan

.Lv4cos_comp:
    ldp     x19, xzr, [sp], #16
    bl      parse_v4_composition
    ldp     x29, x30, [sp], #16
    ret

.Lv4cos_stmt:
    ldp     x19, xzr, [sp], #16
    bl      parse_v4_line
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_composition — parse: name arg1 arg2 ... :
//   Then parse indented body until dedent.
// ============================================================
parse_v4_composition:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!

    // Save sym count for cleanup
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    ldr     w4, [x0]
    stp     x4, xzr, [sp, #-16]!

    // Increment scope depth
    adrp    x0, parse_v4_scope_depth
    add     x0, x0, :lo12:parse_v4_scope_depth
    ldr     w5, [x0]
    add     w6, w5, #1
    str     w6, [x0]

    // Record composition code address
    bl      parse_emit_cur
    mov     x7, x0

    // Add composition to symbol table
    mov     w1, #KIND_COMP
    mov     w2, w7
    mov     w3, w6
    bl      parse_v4_sym_add
    str     w7, [x0, #SYM_REG]

    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Record main's offset if this is "main"
    // Check: is name "main"?
    sub     x8, x19, #TOK_STRIDE_SZ
    ldr     w9, [x8, #8]       // name length
    cmp     w9, #4
    b.ne    .Lv4comp_not_main
    ldr     w9, [x8, #4]       // name offset
    add     x9, x28, x9
    ldrb    w10, [x9]
    cmp     w10, #'m'
    b.ne    .Lv4comp_not_main
    ldrb    w10, [x9, #1]
    cmp     w10, #'a'
    b.ne    .Lv4comp_not_main
    ldrb    w10, [x9, #2]
    cmp     w10, #'i'
    b.ne    .Lv4comp_not_main
    ldrb    w10, [x9, #3]
    cmp     w10, #'n'
    b.ne    .Lv4comp_not_main
    // Record main offset
    adrp    x10, ls_code_buf
    add     x10, x10, :lo12:ls_code_buf
    sub     x7, x7, x10        // offset from code_buf start
    adrp    x10, parse_v4_main_offset
    add     x10, x10, :lo12:parse_v4_main_offset
    str     x7, [x10]

.Lv4comp_not_main:
    // Emit function prologue: STP X29, X30, [SP, #-16]!
    MOVI32  w0, 0xA9BF7BFD
    bl      parse_emit32
    // MOV X29, SP
    MOVI32  w0, 0x910003FD
    bl      parse_emit32

    // Reset register allocator for this function
    bl      parse_v4_reset_regs
    bl      vs_reset

    // Parse parameters (idents before the colon)
    mov     w8, #0
.Lv4comp_params:
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lv4comp_colon

    cmp     w0, #TOK_IDENT
    b.ne    parse_v4_error

    // Add param to symbol table — ARM64 ABI: X0-X7
    mov     w1, #KIND_PARAM
    mov     w2, w8
    adrp    x3, parse_v4_scope_depth
    add     x3, x3, :lo12:parse_v4_scope_depth
    ldr     w3, [x3]
    stp     x8, xzr, [sp, #-16]!
    bl      parse_v4_sym_add
    ldp     x8, xzr, [sp], #16

    add     w8, w8, #1
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4comp_params

.Lv4comp_colon:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ':'

    bl      parse_skip_nl

    // Record body indent level
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lv4comp_indent0
    ldr     w9, [x19, #8]
    b       .Lv4comp_body
.Lv4comp_indent0:
    mov     w9, #2

.Lv4comp_body:
    bl      parse_v4_body       // w9 = expected indent

    // Emit epilogue: LDP X29, X30, [SP], #16
    MOVI32  w0, 0xA8C17BFD
    bl      parse_emit32
    // RET
    bl      parse_emit_ret

    // Pop scope
    adrp    x0, parse_v4_scope_depth
    add     x0, x0, :lo12:parse_v4_scope_depth
    ldr     w1, [x0]
    sub     w1, w1, #1
    str     w1, [x0]
    mov     w0, w1
    bl      parse_v4_sym_pop

    add     sp, sp, #16         // drop saved sym count

    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_body — parse indented body lines
//   w9 = minimum indent level for body membership
// ============================================================
parse_v4_body:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    str     w9, [sp, #-16]!

.Lv4body_loop:
    cmp     x19, x27
    b.hs    .Lv4body_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv4body_done
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv4body_skip
    cmp     w0, #TOK_INDENT
    b.ne    .Lv4body_line

    // Check indent level
    ldr     w1, [x19, #8]
    ldr     w9, [sp, #-16]
    cmp     w1, w9
    b.lt    .Lv4body_done

    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4body_loop

.Lv4body_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4body_loop

.Lv4body_line:
    bl      parse_v4_line
    bl      parse_v4_reset_regs
    bl      vs_reset
    b       .Lv4body_loop

.Lv4body_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_line — THE CORE: parse one stack-language line
//
// Switch on first token:
//   INT literal → MOVZ Xnext, #val; push Xnext
//   known arg name (IDENT in symtab as PARAM) → push arg's register
//   ↓ $N val → eval val; MOV XN, result
//   ↑ $N → MOV Xnext, XN; push Xnext
//   trap → SVC #0
//   → width addr → LDR into Xnext; push
//   ← width addr val → STR
//   binary_op (+,-,*,/,&,|,^,<<,>>) → pop/apply/push
//   known composition → BL; push X0
//   unknown IDENT → named intermediate (name binds to result of rest)
// ============================================================
parse_v4_line:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    cmp     x19, x27
    b.hs    .Lv4line_done

    ldr     w0, [x19]

    // ---- Integer literal ----
    cmp     w0, #TOK_INT
    b.eq    .Lv4line_int

    // ---- Register write: ↓ $N val ----
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv4line_reg_write

    // ---- Register read: ↑ $N ----
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv4line_reg_read

    // ---- trap ----
    cmp     w0, #TOK_TRAP
    b.eq    .Lv4line_trap

    // ---- Load: → width addr ----
    cmp     w0, #TOK_LOAD
    b.eq    .Lv4line_load

    // ---- Store: ← width addr val ----
    cmp     w0, #TOK_STORE
    b.eq    .Lv4line_store

    // ---- Binary operators ----
    cmp     w0, #TOK_PLUS
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_AMP
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_PIPE
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_CARET
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_SHL
    b.eq    .Lv4line_binop
    cmp     w0, #TOK_SHR
    b.eq    .Lv4line_binop

    // ---- Unary operators ----
    cmp     w0, #TOK_SQRT
    b.eq    .Lv4line_unary
    cmp     w0, #TOK_SIN
    b.eq    .Lv4line_unary
    cmp     w0, #TOK_COS
    b.eq    .Lv4line_unary

    // ---- IDENT — could be arg ref, named intermediate, or composition call ----
    cmp     w0, #TOK_IDENT
    b.eq    .Lv4line_ident

    // Skip unknown
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4line_done

// ---- INT literal: emit MOVZ Xnext, #value; push ----
.Lv4line_int:
    bl      parse_v4_int_literal
    mov     x4, x0
    bl      parse_v4_alloc_reg
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    str     w5, [sp, #-16]!
    bl      parse_emit_mov_imm64
    ldr     w5, [sp], #16
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    bl      vs_push
    // Continue parsing rest of line (might have more tokens)
    bl      parse_v4_line_rest
    b       .Lv4line_done

// ---- ↓ $N val: write val into register N ----
.Lv4line_reg_write:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↓

    // Parse $N — expect IDENT starting with $
    bl      parse_dollar_reg
    mov     w4, w0              // target register number
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N

    // Parse value — save w4 across call (caller-saved)
    str     w4, [sp, #-16]!
    bl      parse_v4_value_token  // result reg in w0
    mov     w5, w0
    ldr     w4, [sp], #16

    // Emit: MOV XN, Xval
    mov     w0, w4
    mov     w1, w5
    bl      parse_emit_mov_reg

    b       .Lv4line_done

// ---- ↑ $N: read register N, push onto virtual stack ----
.Lv4line_reg_read:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↑

    bl      parse_dollar_reg
    mov     w4, w0              // source register number
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N

    str     w4, [sp, #-16]!
    bl      parse_v4_alloc_reg
    mov     w5, w0
    ldr     w4, [sp], #16

    // Emit: MOV Xdest, Xsrc
    mov     w0, w5
    mov     w1, w4
    bl      parse_emit_mov_reg

    mov     w0, w5
    bl      vs_push

    b       .Lv4line_done

// ---- trap: emit SVC #0 ----
.Lv4line_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_emit_svc
    b       .Lv4line_done

// ---- → width addr: load from memory ----
.Lv4line_load:
    add     x19, x19, #TOK_STRIDE_SZ  // skip →

    // Parse width
    bl      parse_v4_int_literal
    mov     x4, x0             // width
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse address token — save x4 across call
    str     x4, [sp, #-16]!
    bl      parse_v4_value_token  // addr reg in w0
    mov     w5, w0
    ldr     x4, [sp], #16

    // Allocate result register — save x4, w5 across call
    stp     x4, x5, [sp, #-16]!
    bl      parse_v4_alloc_reg
    mov     w6, w0
    ldp     x4, x5, [sp], #16

    cmp     x4, #8
    b.eq    .Lv4ld_8
    cmp     x4, #16
    b.eq    .Lv4ld_16
    cmp     x4, #32
    b.eq    .Lv4ld_32
    // Default: 64-bit
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_ldr_x_zero
    b       .Lv4ld_push
.Lv4ld_8:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_ldrb_zero
    b       .Lv4ld_push
.Lv4ld_16:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_ldrh_zero
    b       .Lv4ld_push
.Lv4ld_32:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_ldr_w_zero
.Lv4ld_push:
    mov     w0, w6
    bl      vs_push
    b       .Lv4line_done

// ---- ← width addr val: store to memory ----
.Lv4line_store:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ←

    // Parse width
    bl      parse_v4_int_literal
    mov     x4, x0
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse address token — save x4
    str     x4, [sp, #-16]!
    bl      parse_v4_value_token
    mov     w5, w0              // addr reg
    ldr     x4, [sp], #16

    // Parse value token — save x4, w5
    stp     x4, x5, [sp, #-16]!
    bl      parse_v4_value_token
    mov     w6, w0              // val reg
    ldp     x4, x5, [sp], #16

    cmp     x4, #8
    b.eq    .Lv4st_8
    cmp     x4, #16
    b.eq    .Lv4st_16
    cmp     x4, #32
    b.eq    .Lv4st_32
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_str_x_zero
    b       .Lv4line_done
.Lv4st_8:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_strb_zero
    b       .Lv4line_done
.Lv4st_16:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_strh_zero
    b       .Lv4line_done
.Lv4st_32:
    mov     w0, w6
    mov     w1, w5
    bl      parse_emit_str_w_zero
    b       .Lv4line_done

// ---- Binary operator: +, -, *, /, &, |, ^, <<, >> ----
// Two modes:
//   "op operand" = pop one, eval operand, apply, push result
//   bare "op" = pop two, apply, push result
.Lv4line_binop:
    mov     w4, w0              // save operator token type
    add     x19, x19, #TOK_STRIDE_SZ  // skip operator

    // Check if there's an operand after the op (not EOL)
    str     w4, [sp, #-16]!    // save op across is_eol
    bl      is_eol
    ldr     w4, [sp], #16
    cbnz    w0, .Lv4binop_bare

    // "op operand" form: pop one from stack, eval operand
    str     w4, [sp, #-16]!
    bl      vs_pop              // w0 = left operand reg
    mov     w5, w0
    stp     w4, w5, [sp]       // save op and left reg (reuse slot)
    bl      parse_v4_value_token  // w0 = right operand reg
    mov     w6, w0
    ldp     w4, w5, [sp], #16  // w4 = op, w5 = left reg

    // alloc result reg — save w4,w5,w6 across call
    sub     sp, sp, #16
    stp     w4, w5, [sp]
    str     w6, [sp, #8]
    bl      parse_v4_alloc_reg
    mov     w7, w0
    ldp     w4, w5, [sp]
    ldr     w6, [sp, #8]
    add     sp, sp, #16

    b       .Lv4binop_emit

.Lv4binop_bare:
    // Bare op: pop two from stack
    str     w4, [sp, #-16]!
    bl      vs_pop
    mov     w6, w0              // second popped = right
    bl      vs_pop
    mov     w5, w0              // first popped = left
    ldr     w4, [sp], #16

    // alloc result reg — save w4,w5,w6 across call
    sub     sp, sp, #16
    stp     w4, w5, [sp]
    str     w6, [sp, #8]
    bl      parse_v4_alloc_reg
    mov     w7, w0
    ldp     w4, w5, [sp]
    ldr     w6, [sp, #8]
    add     sp, sp, #16

.Lv4binop_emit:
    // w4=op, w5=left, w6=right, w7=result
    cmp     w4, #TOK_PLUS
    b.eq    .Lv4bo_add
    cmp     w4, #TOK_MINUS
    b.eq    .Lv4bo_sub
    cmp     w4, #TOK_STAR
    b.eq    .Lv4bo_mul
    cmp     w4, #TOK_SLASH
    b.eq    .Lv4bo_div
    cmp     w4, #TOK_AMP
    b.eq    .Lv4bo_and
    cmp     w4, #TOK_PIPE
    b.eq    .Lv4bo_or
    cmp     w4, #TOK_CARET
    b.eq    .Lv4bo_eor
    cmp     w4, #TOK_SHL
    b.eq    .Lv4bo_lsl
    cmp     w4, #TOK_SHR
    b.eq    .Lv4bo_lsr
    b       .Lv4bo_push         // fallthrough: NOP

.Lv4bo_add:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_add
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_sub:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_sub
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_mul:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_mul
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_div:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_sdiv
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_and:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_and
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_or:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_orr
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_eor:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_eor
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_lsl:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_lsl
    ldr     w7, [sp], #16
    b       .Lv4bo_push
.Lv4bo_lsr:
    mov     w0, w7; mov     w1, w5; mov     w2, w6
    str     w7, [sp, #-16]!
    bl      parse_emit_lsr
    ldr     w7, [sp], #16

.Lv4bo_push:
    mov     w0, w7
    bl      vs_push
    b       .Lv4line_done

// ---- Unary operator: √, ≅, ≡ ----
.Lv4line_unary:
    mov     w4, w0              // save op type
    add     x19, x19, #TOK_STRIDE_SZ

    // Pop operand from stack
    bl      vs_pop
    mov     w5, w0

    // For ARM64: these map to library calls or NEON.
    // Stub: pass through (result = operand)
    mov     w0, w5
    bl      vs_push
    b       .Lv4line_done

// ---- IDENT at start of line ----
// Could be:
//   1. Known arg/param name → push its register
//   2. Known composition name → BL call, push X0
//   3. Unknown ident → named intermediate (bind name to line result)
.Lv4line_ident:
    // First check if it's a known symbol
    bl      parse_v4_sym_lookup
    cbz     x0, .Lv4line_named_inter

    // Found in symbol table
    ldr     w1, [x0, #SYM_KIND]

    // Param or local — push its register
    cmp     w1, #KIND_PARAM
    b.eq    .Lv4line_push_sym
    cmp     w1, #KIND_LOCAL_REG
    b.eq    .Lv4line_push_sym

    // Composition — call it
    cmp     w1, #KIND_COMP
    b.eq    .Lv4line_comp_call

    // Default: push its register
    b       .Lv4line_push_sym

.Lv4line_push_sym:
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w4
    bl      vs_push
    // Continue parsing rest of line
    bl      parse_v4_line_rest
    b       .Lv4line_done

.Lv4line_comp_call:
    // Parse composition call: name arg1 arg2 ...
    ldr     w7, [x0, #SYM_REG]  // code address
    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Parse arguments and move to X0-X7
    mov     w8, #0
.Lv4cc_args:
    stp     w7, w8, [sp, #-16]!
    bl      is_eol
    ldp     w7, w8, [sp], #16
    cbnz    w0, .Lv4cc_emit

    stp     w7, w8, [sp, #-16]!
    bl      parse_v4_value_token  // result in w0
    ldp     w7, w8, [sp], #16
    mov     w5, w0

    // MOV X<arg>, Xresult
    cmp     w5, w8
    b.eq    .Lv4cc_next
    stp     w7, w8, [sp, #-16]!
    mov     w0, w8
    mov     w1, w5
    bl      parse_emit_mov_reg
    ldp     w7, w8, [sp], #16

.Lv4cc_next:
    add     w8, w8, #1
    b       .Lv4cc_args

.Lv4cc_emit:
    // Emit BL to composition
    str     w7, [sp, #-16]!
    bl      parse_emit_cur
    ldr     w7, [sp], #16
    mov     x1, x0              // current addr
    sub     x2, x7, x1
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      parse_emit32

    // Push X0 (return value)
    mov     w0, #0
    bl      vs_push
    b       .Lv4line_done

// ---- Named intermediate: unknown ident = name for line result ----
.Lv4line_named_inter:
    // Save the name token position
    mov     x4, x19
    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Parse rest of line — this should produce a result on the virtual stack
    bl      parse_v4_line

    // The line should have pushed a result. Peek the top of the virtual stack.
    bl      vs_top
    mov     w5, w0              // register holding the result

    // Bind the name to this register
    mov     x19, x4             // point back to name token
    mov     w1, #KIND_LOCAL_REG
    mov     w2, w5
    adrp    x3, parse_v4_scope_depth
    add     x3, x3, :lo12:parse_v4_scope_depth
    ldr     w3, [x3]
    bl      parse_v4_sym_add

    // Restore x19 past the name
    add     x19, x4, #TOK_STRIDE_SZ
    // Skip to end of line (the sub-parse already consumed the rest)
    bl      parse_v4_skip_to_eol

    b       .Lv4line_done

.Lv4line_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_line_rest — continue parsing after pushing a value
//   If the next token on this line is a binary op, handle it.
//   Otherwise return (the push stands alone).
// ============================================================
parse_v4_line_rest:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    bl      is_eol
    cbnz    w0, .Lv4lr_done

    ldr     w0, [x19]

    // If next is a binary op, dispatch to binop handler
    cmp     w0, #TOK_PLUS
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_AMP
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_PIPE
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_CARET
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_SHL
    b.eq    .Lv4lr_binop
    cmp     w0, #TOK_SHR
    b.eq    .Lv4lr_binop

    b       .Lv4lr_done

.Lv4lr_binop:
    bl      parse_v4_line       // recurse into line handler for the op
.Lv4lr_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_value_token — evaluate one value token, return reg in w0
//   Handles: INT, IDENT ($N or known symbol), and pushes onto
//   virtual stack as side effect for compositions.
//   For simple value evaluation, just returns the register.
// ============================================================
parse_v4_value_token:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    cmp     x19, x27
    b.hs    parse_v4_error

    ldr     w0, [x19]

    cmp     w0, #TOK_INT
    b.eq    .Lv4vt_int
    cmp     w0, #TOK_IDENT
    b.eq    .Lv4vt_ident
    // REG_READ as value: ↑ $N
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv4vt_reg_read

    b       parse_v4_error

.Lv4vt_int:
    bl      parse_v4_int_literal
    mov     x4, x0              // x4 = value
    bl      parse_v4_alloc_reg  // w0 = reg (x4 survives)
    mov     w5, w0              // w5 = dest reg
    mov     w0, w5
    mov     x1, x4
    str     w5, [sp, #-16]!    // save w5 across emit (clobbers w4/w5)
    bl      parse_emit_mov_imm64
    ldr     w5, [sp], #16      // restore reg number
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.Lv4vt_ident:
    // Check if $N
    bl      is_dollar_token
    cbnz    w0, .Lv4vt_dollar

    // Look up symbol
    bl      parse_v4_sym_lookup
    cbz     x0, parse_v4_error  // unknown ident in value position
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w4
    ldp     x29, x30, [sp], #16
    ret

.Lv4vt_dollar:
    // $N — return the hardware register number directly
    bl      parse_dollar_reg
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w4
    ldp     x29, x30, [sp], #16
    ret

.Lv4vt_reg_read:
    // ↑ $N as value expression
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↑
    bl      parse_dollar_reg
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N

    str     w4, [sp, #-16]!
    bl      parse_v4_alloc_reg
    mov     w5, w0
    ldr     w4, [sp], #16
    mov     w0, w5
    mov     w1, w4
    str     w5, [sp, #-16]!
    bl      parse_emit_mov_reg
    ldr     w5, [sp], #16
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_v4_skip_to_eol — advance past remaining tokens on line
// ============================================================
parse_v4_skip_to_eol:
.Lv4sk_loop:
    cmp     x19, x27
    b.hs    .Lv4sk_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv4sk_done
    cmp     w0, #TOK_EOF
    b.eq    .Lv4sk_done
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv4sk_loop
.Lv4sk_done:
    ret

// ============================================================
// Error handlers
// ============================================================
parse_v4_error:
    adrp    x1, parse_v4_err_msg
    add     x1, x1, :lo12:parse_v4_err_msg
    mov     x2, #19
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

parse_v4_error_regspill:
    adrp    x1, parse_v4_err_regspill
    add     x1, x1, :lo12:parse_v4_err_regspill
    mov     x2, #25
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

// ============================================================
// DTC wrapper — call parser from threaded bootstrap
//   Stack: ( tok-buf tok-count src-buf -- )
// ============================================================
.align 4
.global code_PARSE_TOKENS
code_PARSE_TOKENS:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x20, [sp, #-16]!

    mov     x2, x22             // TOS = src-buf
    ldr     x1, [x24], #8      // tok-count
    ldr     x0, [x24], #8      // tok-buf
    ldr     x22, [x24], #8     // new TOS

    bl      parse_tokens

    ldp     x22, x20, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// ============================================================
// .data section
// ============================================================
.data
.align 3

parse_v4_err_msg:
    .ascii "v4: parse error\n"
    .byte 0
    .byte 0
    .byte 0

parse_v4_err_regspill:
    .ascii "v4: register spill error\n"

.align 3

// ============================================================
// .bss section — parser state
// ============================================================
.bss
.align 3

scope_depth:
parse_v4_scope_depth:   .space 4
    .align 3
parse_v4_next_reg:      .space 4
    .align 3
parse_v4_emit_ptr:      .space 8
    .align 3
parse_v4_main_offset:   .space 8
    .align 3

// Virtual register stack
parse_v4_vs_sp:         .space 4
    .align 3
parse_v4_vs_stack:      .space (VS_MAX * 4)
    .align 3

// ============================================================
// Dictionary entry — links into the DTC chain
// ============================================================
.data
.align 3

// We need to link to the tail of the emit-arm64.s chain.
// The existing parser declares entry_p_parse_tokens linking to entry_e_cbnz_fwd.
// We replicate that exactly.
.extern entry_e_cbnz_fwd

entry_p_parse_tokens:
    .quad   entry_e_cbnz_fwd
    .byte   0
    .byte   12
    .ascii  "parse-tokens"
    .align  3
    .quad   code_PARSE_TOKENS
