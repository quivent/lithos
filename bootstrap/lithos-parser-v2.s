// lithos-parser-v2.s — Stack-language parser for Lithos .ls token streams
//
// Compilation model:
//   - Stack positions map to ARM64 registers X9..X15 (7 slots)
//   - A "stack pointer" (sp_reg) tracks the current top register index
//   - Push = allocate next register. Pop = decrement sp_reg.
//   - Each body line is one stack operation
//   - Named intermediate: first unknown ident on a line binds to line result
//   - Composition args: X0-X7 per AAPCS64
//
// Threading: Native ARM64 subroutine calls (bl/ret), not DTC.
//   The parser is called FROM the DTC bootstrap via code_PARSE_TOKENS_V2.
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
//
// Token triple layout (from lithos-lexer.s):
//   [+0] u32 type
//   [+4] u32 offset
//   [+8] u32 length
//   Stride = 12 bytes

// Token type constants
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
.equ TOK_LOAD,      36
.equ TOK_STORE,     37
.equ TOK_REG_READ,  38
.equ TOK_REG_WRITE, 39
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
.equ TOK_SUM,       75
.equ TOK_MAX,       76
.equ TOK_MIN,       77
.equ TOK_INDEX,     78
.equ TOK_SQRT,      79
.equ TOK_SIN,       80
.equ TOK_COS,       81
.equ TOK_DATA,      82
.equ TOK_TRAP,      89

.equ TOK_STRIDE_SZ, 12

// Symbol table entry layout (matches v1 for interop)
.equ SYM_SIZE,      24
.equ SYM_NAME_OFF,  0
.equ SYM_NAME_LEN,  4
.equ SYM_KIND,      8
.equ SYM_REG,       12
.equ SYM_DEPTH,     16

.equ KIND_LOCAL_REG, 0
.equ KIND_PARAM,     1
.equ KIND_VAR,       2
.equ KIND_BUF,       3
.equ KIND_COMP,      4
.equ KIND_DATA,      5
.equ KIND_CONST,     6
.equ KIND_LABEL,     7

.equ MAX_SYMS,      512
.equ MAX_PATCH,     64
.equ MAX_LOOP,      16

// Stack register range: X9..X15
.equ SREG_FIRST,  9
.equ SREG_LAST,   15
.equ SREG_COUNT,  7

// ARM64 instruction constants
.equ ARM64_NOP,     0xD503201F
.equ ARM64_RET,     0xD65F03C0
.equ ARM64_SVC_0,   0xD4000001
.equ ARM64_BRK_1,   0xD4200020

// Code buffer size
.equ CODE_BUF_SIZE, 1048576

// ORRIMM macro
.macro ORRIMM Wd, imm, Wtmp
    mov     \Wtmp, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wtmp, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
    orr     \Wd, \Wd, \Wtmp
.endm

// MOVI32 macro
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
// PUSH — push a value onto the virtual stack
//   Increments sp_reg and returns the new top register index in w0.
//   Guards against overflow (>X15).
// ============================================================
.macro V2_PUSH
    bl      v2_stack_push
.endm

// ============================================================
// POP — pop the top of the virtual stack
//   Decrements sp_reg and returns the popped register index in w0.
//   Guards against underflow (<X9).
// ============================================================
.macro V2_POP
    bl      v2_stack_pop
.endm

// ============================================================
// .text
// ============================================================
.text
.align 4

// ============================================================
// Virtual stack operations
// ============================================================

// v2_stack_push — allocate next stack register
//   Returns: w0 = register number of new top
v2_stack_push:
    adrp    x0, v2_sp_reg
    add     x0, x0, :lo12:v2_sp_reg
    ldr     w1, [x0]
    cmp     w1, #SREG_LAST
    b.gt    v2_err_overflow
    add     w2, w1, #1
    str     w2, [x0]
    mov     w0, w1
    ret

// v2_stack_pop — release top stack register
//   Returns: w0 = register number of the popped slot
v2_stack_pop:
    adrp    x0, v2_sp_reg
    add     x0, x0, :lo12:v2_sp_reg
    ldr     w1, [x0]
    cmp     w1, #SREG_FIRST
    b.le    v2_err_underflow
    sub     w1, w1, #1
    str     w1, [x0]
    mov     w0, w1
    ret

// v2_stack_top — return current top register without popping
//   Returns: w0 = register number of current top (sp_reg - 1)
v2_stack_top:
    adrp    x0, v2_sp_reg
    add     x0, x0, :lo12:v2_sp_reg
    ldr     w0, [x0]
    sub     w0, w0, #1
    ret

// v2_stack_depth — return number of items on stack
//   Returns: w0 = depth
v2_stack_depth:
    adrp    x0, v2_sp_reg
    add     x0, x0, :lo12:v2_sp_reg
    ldr     w0, [x0]
    sub     w0, w0, #SREG_FIRST
    ret

// v2_stack_reset — reset stack to empty (sp_reg = SREG_FIRST)
v2_stack_reset:
    adrp    x0, v2_sp_reg
    add     x0, x0, :lo12:v2_sp_reg
    mov     w1, #SREG_FIRST
    str     w1, [x0]
    ret

// ============================================================
// Token access helpers
// ============================================================

tok_type_v2:
    ldr     w0, [x19]
    ret

tok_text_v2:
    ldr     w1, [x19, #8]          // length
    ldr     w0, [x19, #4]          // offset
    add     x0, x28, x0            // src + offset
    ret

advance_v2:
    add     x19, x19, #TOK_STRIDE_SZ
    ret

skip_newlines_v2:
1:  ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    2f
    cmp     w0, #TOK_INDENT
    b.eq    2f
    ret
2:  add     x19, x19, #TOK_STRIDE_SZ
    b       1b

// at_line_end — check if current token ends a line
//   Returns: w0=1 if at line end (NEWLINE/EOF/INDENT/past end), 0 otherwise
at_line_end:
    cmp     x19, x27
    b.hs    1f
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    1f
    cmp     w0, #TOK_EOF
    b.eq    1f
    // INDENT with lower level also means line end for body parsing
    mov     w0, #0
    ret
1:  mov     w0, #1
    ret

// ============================================================
// String comparison helpers
// ============================================================

// tok_match_4 — check if current token matches a 4-byte string
//   x0 = pointer to 4-char string, returns w0 = 1 if match, 0 if not
tok_match_4:
    ldr     w1, [x19, #8]          // length
    cmp     w1, #4
    b.ne    .Lm4_no
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    ldrb    w4, [x0]
    cmp     w3, w4
    b.ne    .Lm4_no
    ldrb    w3, [x2, #1]
    ldrb    w4, [x0, #1]
    cmp     w3, w4
    b.ne    .Lm4_no
    ldrb    w3, [x2, #2]
    ldrb    w4, [x0, #2]
    cmp     w3, w4
    b.ne    .Lm4_no
    ldrb    w3, [x2, #3]
    ldrb    w4, [x0, #3]
    cmp     w3, w4
    b.ne    .Lm4_no
    mov     w0, #1
    ret
.Lm4_no:
    mov     w0, #0
    ret

// tok_is_trap — check if token is "trap"
tok_is_trap:
    stp     x30, xzr, [sp, #-16]!
    adrp    x0, str_trap_v2
    add     x0, x0, :lo12:str_trap_v2
    bl      tok_match_4
    ldp     x30, xzr, [sp], #16
    ret

// tok_is_main — check if token is "main"
tok_is_main:
    stp     x30, xzr, [sp, #-16]!
    adrp    x0, str_main_v2
    add     x0, x0, :lo12:str_main_v2
    bl      tok_match_4
    ldp     x30, xzr, [sp], #16
    ret

// tok_match_str_v2 — general string match
//   x0 = ptr, w1 = len
//   Returns: w0 = 1 match, 0 no match
tok_match_str_v2:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    mov     x4, x0
    mov     w5, w1
    ldr     w2, [x19, #8]
    cmp     w2, w5
    b.ne    .Lmsv2_no
    ldr     w3, [x19, #4]
    add     x3, x28, x3
    mov     w2, #0
.Lmsv2_loop:
    cmp     w2, w5
    b.ge    .Lmsv2_yes
    ldrb    w0, [x3, x2]
    ldrb    w1, [x4, x2]
    cmp     w0, w1
    b.ne    .Lmsv2_no
    add     w2, w2, #1
    b       .Lmsv2_loop
.Lmsv2_yes:
    mov     w0, #1
    b       .Lmsv2_done
.Lmsv2_no:
    mov     w0, #0
.Lmsv2_done:
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// Symbol table operations (shared with v1 via ls_sym_*)
// ============================================================

sym_lookup_v2:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w0, [x19, #4]          // token offset
    ldr     w1, [x19, #8]          // token length

    adrp    x2, ls_sym_count
    add     x2, x2, :lo12:ls_sym_count
    ldr     w3, [x2]
    cbz     w3, .Lsv2_notfound

    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table

    sub     w3, w3, #1
    mov     w5, #SYM_SIZE
    madd    x6, x3, x5, x4

.Lsv2_loop:
    ldr     w7, [x6, #SYM_NAME_LEN]
    cmp     w7, w1
    b.ne    .Lsv2_next

    ldr     w7, [x6, #SYM_NAME_OFF]
    add     x7, x28, x7
    add     x2, x28, x0
    mov     w5, #0
.Lsv2_cmp:
    cmp     w5, w1
    b.ge    .Lsv2_found
    ldrb    w3, [x7, x5]
    ldrb    w4, [x2, x5]
    cmp     w3, w4
    b.ne    .Lsv2_next
    add     w5, w5, #1
    b       .Lsv2_cmp

.Lsv2_found:
    mov     x0, x6
    b       .Lsv2_done

.Lsv2_next:
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
    cmp     x6, x4
    b.ls    .Lsv2_notfound
    sub     x6, x6, #SYM_SIZE
    b       .Lsv2_loop

.Lsv2_notfound:
    mov     x0, #0

.Lsv2_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

sym_add_v2:
    // w1 = kind, w2 = reg_or_offset, w3 = scope_depth
    // Returns: x0 = pointer to new entry
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

sym_pop_scope_v2:
    stp     x30, xzr, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    mov     w2, w0

    adrp    x3, ls_sym_count
    add     x3, x3, :lo12:ls_sym_count
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table

.Lpop_v2:
    ldr     w0, [x3]
    cbz     w0, .Lpop_v2_done
    sub     w0, w0, #1
    mov     w1, #SYM_SIZE
    madd    x1, x0, x1, x4
    ldr     w1, [x1, #SYM_DEPTH]
    cmp     w1, w2
    b.le    .Lpop_v2_done
    str     w0, [x3]
    b       .Lpop_v2

.Lpop_v2_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// ARM64 code emission helpers
// ============================================================

emit32_v2:
    adrp    x1, v2_emit_ptr
    add     x1, x1, :lo12:v2_emit_ptr
    ldr     x2, [x1]
    str     w0, [x2], #4
    str     x2, [x1]
    ret

emit_cur_v2:
    adrp    x0, v2_emit_ptr
    add     x0, x0, :lo12:v2_emit_ptr
    ldr     x0, [x0]
    ret

// emit_mov_imm16_v2 — MOVZ Xd, #imm16
//   w0 = dest reg, w1 = imm16
emit_mov_imm16_v2:
    and     w2, w1, #0xFFFF
    lsl     w2, w2, #5
    orr     w2, w2, w0
    ORRIMM  w2, 0xD2800000, w16
    mov     w0, w2
    b       emit32_v2

// emit_movk_imm16_v2 — MOVK Xd, #imm16, LSL #shift
//   w0 = dest, w1 = imm16, w2 = shift (0,16,32,48)
emit_movk_imm16_v2:
    lsr     w3, w2, #4
    lsl     w3, w3, #21
    and     w4, w1, #0xFFFF
    lsl     w4, w4, #5
    orr     w4, w4, w0
    orr     w4, w4, w3
    ORRIMM  w4, 0xF2800000, w16
    mov     w0, w4
    b       emit32_v2

// emit_mov_imm64_v2 — load 64-bit immediate into Xd
//   w0 = dest reg, x1 = imm64
emit_mov_imm64_v2:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!

    // Check if value fits in 16 bits
    // MOVZ Xd, #(imm & 0xFFFF)
    and     w2, w1, #0xFFFF
    mov     w3, w0                 // save dest reg
    mov     w1, w2
    mov     w0, w3
    bl      emit_mov_imm16_v2

    ldp     x0, x1, [sp]

    // MOVK for bits [31:16]
    lsr     x4, x1, #16
    and     w1, w4, #0xFFFF
    cbz     w1, .Lv2m64_check32
    mov     w0, w3
    mov     w2, #16
    bl      emit_movk_imm16_v2
    ldp     x0, x1, [sp]

.Lv2m64_check32:
    lsr     x4, x1, #32
    and     w1, w4, #0xFFFF
    cbz     w1, .Lv2m64_check48
    mov     w0, w3
    mov     w2, #32
    bl      emit_movk_imm16_v2
    ldp     x0, x1, [sp]

.Lv2m64_check48:
    lsr     x4, x1, #48
    and     w1, w4, #0xFFFF
    cbz     w1, .Lv2m64_done
    mov     w0, w3
    mov     w2, #48
    bl      emit_movk_imm16_v2

.Lv2m64_done:
    add     sp, sp, #16
    ldp     x30, x19, [sp], #16
    ret

// emit_mov_reg_v2 — MOV Xd, Xm  (ORR Xd, XZR, Xm)
//   w0 = d, w1 = m
emit_mov_reg_v2:
    lsl     w2, w1, #16            // Rm
    mov     w3, #31
    lsl     w3, w3, #5             // Rn = XZR
    orr     w4, w0, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    b       emit32_v2

// emit_add_reg_v2 — ADD Xd, Xn, Xm
emit_add_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8B000000, w16
    mov     w0, w5
    b       emit32_v2

// emit_sub_reg_v2 — SUB Xd, Xn, Xm
emit_sub_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCB000000, w16
    mov     w0, w5
    b       emit32_v2

// emit_mul_reg_v2 — MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR)
emit_mul_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9B007C00, w16
    mov     w0, w5
    b       emit32_v2

// emit_sdiv_reg_v2 — SDIV Xd, Xn, Xm
emit_sdiv_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC00C00, w16
    mov     w0, w5
    b       emit32_v2

// emit_and_reg_v2 — AND Xd, Xn, Xm
emit_and_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8A000000, w16
    mov     w0, w5
    b       emit32_v2

// emit_orr_reg_v2 — ORR Xd, Xn, Xm
emit_orr_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xAA000000, w16
    mov     w0, w5
    b       emit32_v2

// emit_eor_reg_v2 — EOR Xd, Xn, Xm
emit_eor_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCA000000, w16
    mov     w0, w5
    b       emit32_v2

// emit_lsl_reg_v2 — LSLV Xd, Xn, Xm
emit_lsl_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02000, w16
    mov     w0, w5
    b       emit32_v2

// emit_lsr_reg_v2 — LSRV Xd, Xn, Xm
emit_lsr_reg_v2:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02400, w16
    mov     w0, w5
    b       emit32_v2

// emit_cmp_reg_v2 — SUBS XZR, Xn, Xm
emit_cmp_reg_v2:
    lsl     w2, w1, #16
    lsl     w3, w0, #5
    mov     w4, #31
    orr     w4, w4, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xEB000000, w16
    mov     w0, w4
    b       emit32_v2

// emit_b_v2 — B target
emit_b_v2:
    stp     x30, xzr, [sp, #-16]!
    adrp    x1, v2_emit_ptr
    add     x1, x1, :lo12:v2_emit_ptr
    ldr     x2, [x1]
    sub     x3, x0, x2
    asr     x3, x3, #2
    and     w3, w3, #0x3FFFFFF
    ORRIMM  w3, 0x14000000, w16
    mov     w0, w3
    ldp     x30, xzr, [sp], #16
    b       emit32_v2

// emit_bl_v2 — BL target (from current emit position)
//   x0 = target address
emit_bl_v2:
    stp     x30, xzr, [sp, #-16]!
    mov     x5, x0                 // save target
    bl      emit_cur_v2
    sub     x3, x5, x0
    asr     x3, x3, #2
    and     w3, w3, #0x3FFFFFF
    ORRIMM  w3, 0x94000000, w16
    mov     w0, w3
    ldp     x30, xzr, [sp], #16
    b       emit32_v2

// emit_b_cond_v2 — B.cond target
//   w0 = condition code, x1 = target
emit_b_cond_v2:
    stp     x30, x0, [sp, #-16]!
    adrp    x2, v2_emit_ptr
    add     x2, x2, :lo12:v2_emit_ptr
    ldr     x3, [x2]
    sub     x4, x1, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5
    ldp     x30, x0, [sp], #16
    orr     w4, w4, w0
    ORRIMM  w4, 0x54000000, w16
    mov     w0, w4
    b       emit32_v2

// emit_cbz_v2 — CBZ Xn, target
emit_cbz_v2:
    stp     x30, xzr, [sp, #-16]!
    mov     w5, w0
    adrp    x2, v2_emit_ptr
    add     x2, x2, :lo12:v2_emit_ptr
    ldr     x3, [x2]
    sub     x4, x1, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5
    orr     w4, w4, w5
    ORRIMM  w4, 0xB4000000, w16
    mov     w0, w4
    ldp     x30, xzr, [sp], #16
    b       emit32_v2

emit_svc_v2:
    MOVI32  w0, ARM64_SVC_0
    b       emit32_v2

emit_ret_v2:
    MOVI32  w0, ARM64_RET
    b       emit32_v2

emit_nop_v2:
    MOVI32  w0, ARM64_NOP
    b       emit32_v2

// patch_b_v2 — patch B at x0 to target x1
patch_b_v2:
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x14000000, w16
    str     w2, [x0]
    ret

// patch_cbz_v2 — patch CBZ at x0 to target x1
patch_cbz_v2:
    ldr     w3, [x0]
    and     w3, w3, #0xFF00001F    // keep opcode + Rt
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x7FFFF
    lsl     w2, w2, #5
    orr     w3, w3, w2
    str     w3, [x0]
    ret

// patch_b_cond_v2 — patch B.cond at x0 to target x1
patch_b_cond_v2:
    ldr     w3, [x0]
    and     w3, w3, #0xF           // preserve cond
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x7FFFF
    lsl     w2, w2, #5
    orr     w2, w2, w3
    ORRIMM  w2, 0x54000000, w16
    str     w2, [x0]
    ret

// Memory load/store emission (width-dispatched)
// emit_load_width_v2: w0=dest, w1=addr_reg, x2=width(8/16/32/64)
emit_load_width_v2:
    cmp     x2, #8
    b.eq    .Llw_8
    cmp     x2, #16
    b.eq    .Llw_16
    cmp     x2, #32
    b.eq    .Llw_32
    // default 64-bit
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9400000, w16   // LDR X, [Xn]
    mov     w0, w5
    b       emit32_v2
.Llw_8:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39400000, w16   // LDRB W, [Xn]
    mov     w0, w5
    b       emit32_v2
.Llw_16:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79400000, w16   // LDRH W, [Xn]
    mov     w0, w5
    b       emit32_v2
.Llw_32:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9400000, w16   // LDR W, [Xn]
    mov     w0, w5
    b       emit32_v2

// emit_store_width_v2: w0=val_reg, w1=addr_reg, x2=width
emit_store_width_v2:
    cmp     x2, #8
    b.eq    .Lsw_8
    cmp     x2, #16
    b.eq    .Lsw_16
    cmp     x2, #32
    b.eq    .Lsw_32
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9000000, w16   // STR X, [Xn]
    mov     w0, w5
    b       emit32_v2
.Lsw_8:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39000000, w16
    mov     w0, w5
    b       emit32_v2
.Lsw_16:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79000000, w16
    mov     w0, w5
    b       emit32_v2
.Lsw_32:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9000000, w16
    mov     w0, w5
    b       emit32_v2

// ============================================================
// Number parsing
// ============================================================
parse_int_v2:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x1, x28, x1

    mov     x0, #0
    mov     w3, #0
    mov     w7, #0

    ldrb    w4, [x1]
    cmp     w4, #'-'
    b.ne    .Lv2i_chkhex
    mov     w7, #1
    add     w3, w3, #1

.Lv2i_chkhex:
    sub     w5, w2, w3
    cmp     w5, #2
    b.lt    .Lv2i_dec
    ldrb    w4, [x1, x3]
    cmp     w4, #'0'
    b.ne    .Lv2i_dec
    add     w6, w3, #1
    ldrb    w4, [x1, x6]
    cmp     w4, #'x'
    b.eq    .Lv2i_hex
    cmp     w4, #'X'
    b.ne    .Lv2i_dec

.Lv2i_hex:
    add     w3, w3, #2
.Lv2i_hexloop:
    cmp     w3, w2
    b.ge    .Lv2i_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'9'
    b.le    .Lv2i_h09
    cmp     w4, #'F'
    b.le    .Lv2i_hAF
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lv2i_haccum
.Lv2i_hAF:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lv2i_haccum
.Lv2i_h09:
    sub     w4, w4, #'0'
.Lv2i_haccum:
    lsl     x0, x0, #4
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv2i_hexloop

.Lv2i_dec:
.Lv2i_decloop:
    cmp     w3, w2
    b.ge    .Lv2i_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'.'
    b.eq    .Lv2i_sign
    sub     w4, w4, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv2i_decloop

.Lv2i_sign:
    cbz     w7, .Lv2i_done
    neg     x0, x0
.Lv2i_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// parse_dollar_reg — parse "$N" token, returns register number in w0
//   Current token should be an IDENT starting with '$'.
//   Parses the number after '$'.
// ============================================================
parse_dollar_reg_v2:
    stp     x30, xzr, [sp, #-16]!
    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x1, x28, x1

    // Skip the '$' character
    ldrb    w3, [x1]
    cmp     w3, #'$'
    b.ne    .Ldollar_fail

    mov     x0, #0
    mov     w3, #1                 // start after '$'
.Ldollar_loop:
    cmp     w3, w2
    b.ge    .Ldollar_done
    ldrb    w4, [x1, x3]
    sub     w4, w4, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Ldollar_loop
.Ldollar_done:
    ldp     x30, xzr, [sp], #16
    ret
.Ldollar_fail:
    mov     x0, #0
    ldp     x30, xzr, [sp], #16
    ret

// check if current token starts with '$'
tok_is_dollar_v2:
    ldr     w1, [x19, #8]         // length
    cbz     w1, .Ldol_no
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'$'
    b.ne    .Ldol_no
    mov     w0, #1
    ret
.Ldol_no:
    mov     w0, #0
    ret

// ============================================================
// PARSER ENTRY POINT
// ============================================================
.global parse_tokens_v2
.align 4
parse_tokens_v2:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x27, [sp, #-16]!
    stp     x28, x20, [sp, #-16]!

    mov     x19, x0                // TOKP
    mov     w3, #TOK_STRIDE_SZ
    mul     x27, x1, x3
    add     x27, x27, x0           // TOKEND
    mov     x28, x2                // SRC

    // Initialize emit pointer
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v2_emit_ptr
    add     x1, x1, :lo12:v2_emit_ptr
    str     x0, [x1]

    // Clear symbol table
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    str     wzr, [x0]

    // Clear scope depth
    adrp    x0, v2_scope_depth
    add     x0, x0, :lo12:v2_scope_depth
    str     wzr, [x0]

    // Clear main entry pointer
    adrp    x0, v2_main_addr
    add     x0, x0, :lo12:v2_main_addr
    str     xzr, [x0]

    // Reset virtual stack
    bl      v2_stack_reset

    // Emit placeholder B instruction at code start (will be patched to jump to main)
    bl      emit_cur_v2
    adrp    x4, v2_entry_patch
    add     x4, x4, :lo12:v2_entry_patch
    str     x0, [x4]              // save address of the B instruction
    bl      emit_nop_v2            // placeholder (will be patched)

    // Parse top-level
    bl      v2_parse_toplevel

    // Patch the entry B to jump to main (if main was found)
    adrp    x0, v2_main_addr
    add     x0, x0, :lo12:v2_main_addr
    ldr     x1, [x0]              // main address
    cbz     x1, .Lv2_no_main_patch
    adrp    x0, v2_entry_patch
    add     x0, x0, :lo12:v2_entry_patch
    ldr     x0, [x0]              // address of placeholder B
    bl      patch_b_v2             // patch B to main
.Lv2_no_main_patch:

    // Sync ls_code_pos
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v2_emit_ptr
    add     x1, x1, :lo12:v2_emit_ptr
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
// v2_parse_toplevel — scan for composition definitions
// ============================================================
v2_parse_toplevel:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Lv2top_loop:
    cmp     x19, x27
    b.hs    .Lv2top_done

    bl      skip_newlines_v2
    cmp     x19, x27
    b.hs    .Lv2top_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv2top_done

    // Only compositions at top level: IDENT ... COLON
    cmp     w0, #TOK_IDENT
    b.eq    .Lv2top_comp_check

    // Skip unknown tokens
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2top_loop

.Lv2top_comp_check:
    // Lookahead for COLON on this line
    mov     x4, x19
.Lv2top_scan:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lv2top_skip
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lv2top_comp
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv2top_skip
    cmp     w0, #TOK_EOF
    b.eq    .Lv2top_skip
    b       .Lv2top_scan

.Lv2top_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2top_loop

.Lv2top_comp:
    bl      v2_parse_composition
    b       .Lv2top_loop

.Lv2top_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v2_parse_composition — name arg1 arg2 ... :
//   Parse body lines using stack model.
// ============================================================
v2_parse_composition:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #48            // locals: [0]=old_sym_count [8]=comp_name_is_main

    // Save sym count for scope cleanup
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    ldr     w4, [x0]
    str     w4, [sp, #0]

    // Increment scope depth
    adrp    x0, v2_scope_depth
    add     x0, x0, :lo12:v2_scope_depth
    ldr     w5, [x0]
    add     w6, w5, #1
    str     w6, [x0]

    // Check if this is "main"
    bl      tok_is_main
    str     w0, [sp, #8]

    // Record composition start address
    bl      emit_cur_v2
    mov     x7, x0

    // If main, record the address
    ldr     w0, [sp, #8]
    cbz     w0, .Lv2comp_not_main
    adrp    x0, v2_main_addr
    add     x0, x0, :lo12:v2_main_addr
    str     x7, [x0]
.Lv2comp_not_main:

    // Add composition to symbol table
    mov     w1, #KIND_COMP
    mov     w2, w7                 // code address (truncated to 32-bit offset is ok for sym_add)
    adrp    x3, v2_scope_depth
    add     x3, x3, :lo12:v2_scope_depth
    ldr     w3, [x3]
    bl      sym_add_v2

    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Emit prologue (skip for "main" — it's the bare entry point)
    ldr     w0, [sp, #8]
    cbnz    w0, .Lv2comp_skip_prologue
    MOVI32  w0, 0xA9BF7BFD            // STP X29, X30, [SP, #-16]!
    bl      emit32_v2
    MOVI32  w0, 0x910003FD             // MOV X29, SP
    bl      emit32_v2
.Lv2comp_skip_prologue:

    // Reset virtual stack for this composition
    bl      v2_stack_reset

    // Parse parameters before colon
    mov     w8, #0                 // param index
.Lv2comp_params:
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lv2comp_colon

    cmp     w0, #TOK_IDENT
    b.ne    v2_parse_error

    // Add param to symbol table (X0-X7)
    mov     w1, #KIND_PARAM
    mov     w2, w8
    adrp    x3, v2_scope_depth
    add     x3, x3, :lo12:v2_scope_depth
    ldr     w3, [x3]
    stp     x8, xzr, [sp, #16]
    bl      sym_add_v2
    ldp     x8, xzr, [sp, #16]

    add     w8, w8, #1
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2comp_params

.Lv2comp_colon:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ':'

    // Skip newlines/whitespace
    bl      skip_newlines_v2

    // Determine body indent level
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lv2comp_indent0
    ldr     w9, [x19, #8]
    b       .Lv2comp_body
.Lv2comp_indent0:
    mov     w9, #2

.Lv2comp_body:
    // Parse body lines
    bl      v2_parse_body          // w9 = body indent

    // Emit epilogue
    // If stack has a value, move top to X0 (return value)
    bl      v2_stack_depth
    cbz     w0, .Lv2comp_noretval
    bl      v2_stack_top
    cmp     w0, #0                 // if already X0, skip
    b.eq    .Lv2comp_noretval
    mov     w1, w0
    mov     w0, #0
    bl      emit_mov_reg_v2
.Lv2comp_noretval:

    // Emit epilogue (skip for "main")
    ldr     w0, [sp, #8]
    cbnz    w0, .Lv2comp_skip_epilogue
    MOVI32  w0, 0xA8C17BFD            // LDP X29, X30, [SP], #16
    bl      emit32_v2
    bl      emit_ret_v2
.Lv2comp_skip_epilogue:

    // Pop scope
    adrp    x0, v2_scope_depth
    add     x0, x0, :lo12:v2_scope_depth
    ldr     w1, [x0]
    sub     w1, w1, #1
    str     w1, [x0]
    mov     w0, w1
    bl      sym_pop_scope_v2

    add     sp, sp, #48
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v2_parse_body — parse indented body lines (stack model)
//   w9 = minimum indent level
// ============================================================
v2_parse_body:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16
    str     w9, [sp]               // save expected indent at [sp]

.Lv2body_loop:
    cmp     x19, x27
    b.hs    .Lv2body_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv2body_done

    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv2body_skip_nl
    cmp     w0, #TOK_INDENT
    b.ne    .Lv2body_stmt

    // Check indent level
    ldr     w1, [x19, #8]         // actual indent
    ldr     w9, [sp]              // expected indent
    cmp     w1, w9
    b.lt    .Lv2body_done          // dedent = end of body

    add     x19, x19, #TOK_STRIDE_SZ  // consume INDENT token
    b       .Lv2body_loop

.Lv2body_skip_nl:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2body_loop

.Lv2body_stmt:
    bl      v2_parse_line
    b       .Lv2body_loop

.Lv2body_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v2_parse_line — parse one body line (the core stack-language logic)
//
// Each body line is one stack operation:
//   INTEGER          → push literal
//   IDENT (known)    → push value (param reg or local reg)
//   IDENT (unknown)  → check if rest of line produces a value → named intermediate
//   + operand        → push operand, ADD top two, result in new top
//   - operand        → push operand, SUB top two, result in new top
//   * operand        → push operand, MUL ...
//   / operand        → push operand, SDIV ...
//   & | ^ << >>      → same pattern
//   ↓ $N val         → write val into ARM64 register N
//   ↑ $N             → read ARM64 register N, push onto stack
//   trap             → SVC #0
//   → width addr     → load from memory, push result
//   ← width addr val → store to memory
//   if== a b         → compare and conditionally branch
//   if>= a b         → compare and conditionally branch
//   if< a b          → compare and conditionally branch
//   name arg1 arg2   → composition call (known comp ident followed by args)
//
// Named intermediate rule:
//   If first token is unknown IDENT and next tokens produce a value,
//   bind the name to whatever register holds the result.
// ============================================================
v2_parse_line:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    // --- Binary ops with operand: + - * / & | ^ << >> ---
    cmp     w0, #TOK_PLUS
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_AMP
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_PIPE
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_CARET
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_SHL
    b.eq    .Lv2line_binop
    cmp     w0, #TOK_SHR
    b.eq    .Lv2line_binop

    // --- Register write: ↓ ---
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv2line_reg_write

    // --- Register read: ↑ ---
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv2line_reg_read

    // --- Memory load: → ---
    cmp     w0, #TOK_LOAD
    b.eq    .Lv2line_mem_load

    // --- Memory store: ← ---
    cmp     w0, #TOK_STORE
    b.eq    .Lv2line_mem_store

    // --- Conditionals ---
    cmp     w0, #TOK_IF
    b.eq    .Lv2line_if

    // --- Integer literal → push ---
    cmp     w0, #TOK_INT
    b.eq    .Lv2line_push_int

    // --- Ident → various ---
    cmp     w0, #TOK_IDENT
    b.eq    .Lv2line_ident

    // --- trap (TOK_TRAP from lexer) ---
    cmp     w0, #TOK_TRAP
    b.eq    .Lv2line_trap

    // --- Unary math ---
    cmp     w0, #TOK_SQRT
    b.eq    .Lv2line_unary
    cmp     w0, #TOK_SIN
    b.eq    .Lv2line_unary
    cmp     w0, #TOK_COS
    b.eq    .Lv2line_unary
    cmp     w0, #TOK_SUM
    b.eq    .Lv2line_unary
    cmp     w0, #TOK_MAX
    b.eq    .Lv2line_unary
    cmp     w0, #TOK_MIN
    b.eq    .Lv2line_unary

    // Skip unknown
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2line_skip_rest

.Lv2line_done:
    ldp     x29, x30, [sp], #16
    ret

// ---- Push integer literal ----
.Lv2line_push_int:
    bl      parse_int_v2
    mov     x4, x0
    V2_PUSH                        // w0 = new top reg
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    bl      emit_mov_imm64_v2
    add     x19, x19, #TOK_STRIDE_SZ
    // Check for more tokens on this line (bare op following)
    bl      v2_try_continuation
    b       .Lv2line_done

// ---- Binary op with operand ----
// Pattern: OP [operand]
//   If operand present: push operand, then binary-op top two
//   If no operand (bare op): binary-op top two existing stack entries
.Lv2line_binop:
    mov     w10, w0                // save operator token type
    add     x19, x19, #TOK_STRIDE_SZ  // skip operator

    // Check if there's an operand on this line
    bl      at_line_end
    cbnz    w0, .Lv2line_binop_bare

    // Has operand — evaluate it, push, then operate
    stp     w10, wzr, [sp, #-16]!
    bl      v2_eval_operand        // pushes result onto stack
    ldp     w10, wzr, [sp], #16

.Lv2line_binop_bare:
    // Pop two, operate, push result
    // Top = right operand, below = left operand
    V2_POP
    mov     w6, w0                 // right reg
    V2_POP
    mov     w5, w0                 // left reg
    V2_PUSH
    mov     w7, w0                 // result reg

    // Dispatch operator
    cmp     w10, #TOK_PLUS
    b.eq    .Lv2bin_add
    cmp     w10, #TOK_MINUS
    b.eq    .Lv2bin_sub
    cmp     w10, #TOK_STAR
    b.eq    .Lv2bin_mul
    cmp     w10, #TOK_SLASH
    b.eq    .Lv2bin_div
    cmp     w10, #TOK_AMP
    b.eq    .Lv2bin_and
    cmp     w10, #TOK_PIPE
    b.eq    .Lv2bin_or
    cmp     w10, #TOK_CARET
    b.eq    .Lv2bin_xor
    cmp     w10, #TOK_SHL
    b.eq    .Lv2bin_shl
    cmp     w10, #TOK_SHR
    b.eq    .Lv2bin_shr
    b       .Lv2line_done          // shouldn't happen

.Lv2bin_add:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_add_reg_v2
    b       .Lv2line_done
.Lv2bin_sub:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_sub_reg_v2
    b       .Lv2line_done
.Lv2bin_mul:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_mul_reg_v2
    b       .Lv2line_done
.Lv2bin_div:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_sdiv_reg_v2
    b       .Lv2line_done
.Lv2bin_and:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_and_reg_v2
    b       .Lv2line_done
.Lv2bin_or:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_orr_reg_v2
    b       .Lv2line_done
.Lv2bin_xor:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_eor_reg_v2
    b       .Lv2line_done
.Lv2bin_shl:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_lsl_reg_v2
    b       .Lv2line_done
.Lv2bin_shr:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      emit_lsr_reg_v2
    b       .Lv2line_done

// ---- Register write: ↓ $N val ----
.Lv2line_reg_write:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↓

    // Parse $N — expect ident starting with '$'
    bl      tok_is_dollar_v2
    cbz     w0, v2_parse_error
    bl      parse_dollar_reg_v2
    mov     w4, w0                 // target hardware register number
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N

    // Parse value
    stp     w4, wzr, [sp, #-16]!
    bl      v2_eval_operand        // pushes onto stack
    ldp     w4, wzr, [sp], #16

    // Pop value from virtual stack
    V2_POP
    mov     w5, w0                 // value in stack register w5

    // Emit: MOV X<target>, X<value>
    mov     w0, w4
    mov     w1, w5
    bl      emit_mov_reg_v2
    b       .Lv2line_done

// ---- Register read: ↑ $N ----
.Lv2line_reg_read:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↑

    // Parse $N
    bl      tok_is_dollar_v2
    cbz     w0, v2_parse_error
    bl      parse_dollar_reg_v2
    mov     w4, w0                 // hardware register number
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N

    // Push: allocate stack register, emit MOV Xstack, X<hw>
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2
    b       .Lv2line_done

// ---- Memory load: → width addr ----
.Lv2line_mem_load:
    add     x19, x19, #TOK_STRIDE_SZ  // skip →

    // Parse width
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    v2_parse_error
    bl      parse_int_v2
    mov     x10, x0               // width
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse address (pushes onto stack)
    stp     x10, xzr, [sp, #-16]!
    bl      v2_eval_operand
    ldp     x10, xzr, [sp], #16

    // Pop address, push result
    V2_POP
    mov     w4, w0                 // addr reg
    V2_PUSH
    mov     w5, w0                 // dest reg

    mov     w0, w5
    mov     w1, w4
    mov     x2, x10
    bl      emit_load_width_v2
    b       .Lv2line_done

// ---- Memory store: ← width addr val ----
.Lv2line_mem_store:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ←

    // Parse width
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    v2_parse_error
    bl      parse_int_v2
    mov     x10, x0
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse addr
    stp     x10, xzr, [sp, #-16]!
    bl      v2_eval_operand
    // Parse val
    bl      v2_eval_operand
    ldp     x10, xzr, [sp], #16

    // Pop val, pop addr
    V2_POP
    mov     w6, w0                 // val reg
    V2_POP
    mov     w5, w0                 // addr reg

    mov     w0, w6
    mov     w1, w5
    mov     x2, x10
    bl      emit_store_width_v2
    b       .Lv2line_done

// ---- Conditional: if== if>= if< ----
// These are parsed as:  if== / if>= / if<  followed by
// the condition operands on the rest of the line, then an indented body.
.Lv2line_if:
    add     x19, x19, #TOK_STRIDE_SZ  // skip 'if' token

    // The actual condition type is encoded in the IF token.
    // We need to check the next token for ==, >=, <
    // Actually the lexer might encode if== as a single token, or
    // if followed by == as two tokens. Check both.
    ldr     w0, [x19]
    cmp     w0, #TOK_EQEQ
    b.eq    .Lv2if_eq
    cmp     w0, #TOK_GTE
    b.eq    .Lv2if_ge
    cmp     w0, #TOK_LT
    b.eq    .Lv2if_lt
    // Treat bare 'if' as if!=0 (truthy check)
    b       .Lv2if_truthy

.Lv2if_eq:
    add     x19, x19, #TOK_STRIDE_SZ
    // Parse two operands
    bl      v2_eval_operand
    bl      v2_eval_operand
    V2_POP
    mov     w6, w0                 // b
    V2_POP
    mov     w5, w0                 // a
    // CMP a, b
    mov     w0, w5
    mov     w1, w6
    bl      emit_cmp_reg_v2
    // B.NE skip (placeholder)
    bl      emit_cur_v2
    mov     x10, x0
    mov     w0, #1                 // CC_NE
    mov     x1, #0
    bl      emit_b_cond_v2
    b       .Lv2if_body

.Lv2if_ge:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v2_eval_operand
    bl      v2_eval_operand
    V2_POP
    mov     w6, w0
    V2_POP
    mov     w5, w0
    mov     w0, w5
    mov     w1, w6
    bl      emit_cmp_reg_v2
    bl      emit_cur_v2
    mov     x10, x0
    mov     w0, #11                // CC_LT (inverse of GE)
    mov     x1, #0
    bl      emit_b_cond_v2
    b       .Lv2if_body

.Lv2if_lt:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v2_eval_operand
    bl      v2_eval_operand
    V2_POP
    mov     w6, w0
    V2_POP
    mov     w5, w0
    mov     w0, w5
    mov     w1, w6
    bl      emit_cmp_reg_v2
    bl      emit_cur_v2
    mov     x10, x0
    mov     w0, #10                // CC_GE (inverse of LT)
    mov     x1, #0
    bl      emit_b_cond_v2
    b       .Lv2if_body

.Lv2if_truthy:
    // Check top of stack != 0
    bl      v2_stack_top
    mov     w5, w0
    bl      emit_cur_v2
    mov     x10, x0
    mov     w0, w5
    mov     x1, #0
    bl      emit_cbz_v2
    b       .Lv2if_body

.Lv2if_body:
    // Skip to body
    stp     x10, xzr, [sp, #-16]!
    bl      skip_newlines_v2
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    1f
    ldr     w9, [x19, #8]
1:  bl      v2_parse_body
    ldp     x10, xzr, [sp], #16

    // Patch branch to here
    bl      emit_cur_v2
    mov     x1, x0
    mov     x0, x10
    bl      patch_b_cond_v2

    b       .Lv2line_done

// ---- Unary ops (applied to top of stack) ----
.Lv2line_unary:
    // For ARM64 host, unary math ops are stubs (would need NEON/libm).
    // Just consume the token and pass through.
    add     x19, x19, #TOK_STRIDE_SZ
    // If there's an operand, push it
    bl      at_line_end
    cbz     w0, .Lv2line_unary_has_arg
    b       .Lv2line_done
.Lv2line_unary_has_arg:
    bl      v2_eval_operand
    b       .Lv2line_done

// ---- Identifier at start of line ----
// Could be:
//   1. "trap" → SVC #0
//   2. known param/local → push its register value
//   3. known composition → call it with remaining args
//   4. unknown ident → named intermediate (bind to line result)
.Lv2line_ident:
    // Check for "trap"
    bl      tok_is_trap
    cbnz    w0, .Lv2line_trap

    // Look up in symbol table
    bl      sym_lookup_v2
    cbz     x0, .Lv2line_named_intermediate

    // Known symbol
    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_COMP
    b.eq    .Lv2line_comp_call

    // Known param or local — push its value
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ  // consume ident
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2

    // Check for continuation (e.g. bare binary op on same line after push)
    bl      v2_try_continuation
    b       .Lv2line_done

// ---- Composition call ----
.Lv2line_comp_call:
    mov     x5, x0                // sym entry
    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Parse args — each goes to X0, X1, X2...
    mov     w8, #0
.Lv2call_args:
    bl      at_line_end
    cbnz    w0, .Lv2call_emit
    cmp     x19, x27
    b.hs    .Lv2call_emit

    // Evaluate argument, push onto stack
    stp     x5, x8, [sp, #-16]!
    bl      v2_eval_operand
    ldp     x5, x8, [sp], #16

    // Pop into arg register
    V2_POP
    mov     w4, w0                 // stack reg holding value
    cmp     w4, w8
    b.eq    .Lv2call_arg_skip
    mov     w0, w8                 // target = arg index (X0, X1...)
    mov     w1, w4
    bl      emit_mov_reg_v2
.Lv2call_arg_skip:
    add     w8, w8, #1
    b       .Lv2call_args

.Lv2call_emit:
    // Emit BL to composition
    ldr     w0, [x5, #SYM_REG]
    // BL target
    bl      emit_cur_v2
    mov     x1, x0
    ldr     w0, [x5, #SYM_REG]    // code addr (32-bit in sym table)
    // Sign-extend w0 to x0 for address arithmetic
    sxtw    x0, w0
    sub     x2, x0, x1
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      emit32_v2

    // Result in X0 — push onto stack
    V2_PUSH
    mov     w5, w0
    cmp     w5, #0
    b.eq    .Lv2call_done
    mov     w0, w5
    mov     w1, #0                 // MOV Xstack, X0
    bl      emit_mov_reg_v2
.Lv2call_done:
    b       .Lv2line_done

// ---- Trap (SVC #0) ----
.Lv2line_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc_v2
    b       .Lv2line_done

// ---- Named intermediate ----
// Unknown ident at start of line: name REST_OF_LINE
// Parse the rest of the line, bind the result to name.
.Lv2line_named_intermediate:
    // Save name token position
    mov     x10, x19
    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Check what follows — if line end, just push as unknown (error recovery)
    bl      at_line_end
    cbnz    w0, .Lv2line_named_skip

    // Parse rest of line as a sub-expression/line
    bl      v2_parse_line_rest

    // Bind name to current top of stack
    bl      v2_stack_top
    mov     w4, w0                 // register holding result

    // Temporarily point x19 to name token for sym_add
    mov     x11, x19              // save current position
    mov     x19, x10
    mov     w1, #KIND_LOCAL_REG
    mov     w2, w4
    adrp    x3, v2_scope_depth
    add     x3, x3, :lo12:v2_scope_depth
    ldr     w3, [x3]
    bl      sym_add_v2
    mov     x19, x11              // restore position
    b       .Lv2line_done

.Lv2line_named_skip:
    // Unknown ident at line end — skip
    b       .Lv2line_done

// Skip rest of line (consume tokens until NEWLINE/EOF/INDENT)
.Lv2line_skip_rest:
    cmp     x19, x27
    b.hs    .Lv2line_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv2line_done
    cmp     w0, #TOK_EOF
    b.eq    .Lv2line_done
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2line_skip_rest

// ============================================================
// v2_parse_line_rest — parse the remaining tokens on a line
//   Used after a named intermediate consumes the first ident.
//   This is essentially the same dispatch as v2_parse_line but
//   without the named-intermediate check.
// ============================================================
v2_parse_line_rest:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    bl      at_line_end
    cbnz    w0, .Lv2lr_done

    ldr     w0, [x19]

    // Binary ops
    cmp     w0, #TOK_PLUS
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_AMP
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_PIPE
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_CARET
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_SHL
    b.eq    .Lv2lr_binop
    cmp     w0, #TOK_SHR
    b.eq    .Lv2lr_binop

    // Integer
    cmp     w0, #TOK_INT
    b.eq    .Lv2lr_int

    // Ident
    cmp     w0, #TOK_IDENT
    b.eq    .Lv2lr_ident

    // Register ops
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv2lr_rr
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv2lr_rw

    // Memory ops
    cmp     w0, #TOK_LOAD
    b.eq    .Lv2lr_load
    cmp     w0, #TOK_STORE
    b.eq    .Lv2lr_store

    // Trap
    cmp     w0, #TOK_TRAP
    b.eq    .Lv2lr_trap

    // Unary
    cmp     w0, #TOK_SQRT
    b.eq    .Lv2lr_unary
    cmp     w0, #TOK_SIN
    b.eq    .Lv2lr_unary
    cmp     w0, #TOK_COS
    b.eq    .Lv2lr_unary

    // Unknown — skip
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2lr_done

.Lv2lr_binop:
    // Dispatch to main line binop handler
    // The rest of the line logic is: evaluate operand, then apply binary op
    mov     w10, w0
    add     x19, x19, #TOK_STRIDE_SZ

    bl      at_line_end
    cbnz    w0, .Lv2lr_binop_bare

    stp     w10, wzr, [sp, #-16]!
    bl      v2_eval_operand
    ldp     w10, wzr, [sp], #16

.Lv2lr_binop_bare:
    V2_POP
    mov     w6, w0
    V2_POP
    mov     w5, w0
    V2_PUSH
    mov     w7, w0

    cmp     w10, #TOK_PLUS
    b.eq    .Lv2lr_ba
    cmp     w10, #TOK_MINUS
    b.eq    .Lv2lr_bs
    cmp     w10, #TOK_STAR
    b.eq    .Lv2lr_bm
    cmp     w10, #TOK_SLASH
    b.eq    .Lv2lr_bd
    cmp     w10, #TOK_AMP
    b.eq    .Lv2lr_band
    cmp     w10, #TOK_PIPE
    b.eq    .Lv2lr_bor
    cmp     w10, #TOK_CARET
    b.eq    .Lv2lr_bxor
    cmp     w10, #TOK_SHL
    b.eq    .Lv2lr_bshl
    cmp     w10, #TOK_SHR
    b.eq    .Lv2lr_bshr
    b       .Lv2lr_done

.Lv2lr_ba:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_add_reg_v2; b .Lv2lr_done
.Lv2lr_bs:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_sub_reg_v2; b .Lv2lr_done
.Lv2lr_bm:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_mul_reg_v2; b .Lv2lr_done
.Lv2lr_bd:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_sdiv_reg_v2; b .Lv2lr_done
.Lv2lr_band:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_and_reg_v2; b .Lv2lr_done
.Lv2lr_bor:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_orr_reg_v2; b .Lv2lr_done
.Lv2lr_bxor:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_eor_reg_v2; b .Lv2lr_done
.Lv2lr_bshl:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_lsl_reg_v2; b .Lv2lr_done
.Lv2lr_bshr:
    mov     w0, w7; mov w1, w5; mov w2, w6; bl emit_lsr_reg_v2; b .Lv2lr_done

.Lv2lr_int:
    bl      parse_int_v2
    mov     x4, x0
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    bl      emit_mov_imm64_v2
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v2_try_continuation
    b       .Lv2lr_done

.Lv2lr_ident:
    bl      tok_is_trap
    cbnz    w0, .Lv2lr_trap
    bl      sym_lookup_v2
    cbz     x0, .Lv2lr_unknown_ident
    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_COMP
    b.eq    .Lv2lr_comp_call
    // Known var/param — push
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2
    bl      v2_try_continuation
    b       .Lv2lr_done

.Lv2lr_comp_call:
    // Composition call in line-rest context
    mov     x5, x0
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w8, #0
.Lv2lr_call_args:
    bl      at_line_end
    cbnz    w0, .Lv2lr_call_emit
    stp     x5, x8, [sp, #-16]!
    bl      v2_eval_operand
    ldp     x5, x8, [sp], #16
    V2_POP
    mov     w4, w0
    cmp     w4, w8
    b.eq    .Lv2lr_call_skip
    mov     w0, w8
    mov     w1, w4
    bl      emit_mov_reg_v2
.Lv2lr_call_skip:
    add     w8, w8, #1
    b       .Lv2lr_call_args
.Lv2lr_call_emit:
    ldr     w0, [x5, #SYM_REG]
    bl      emit_cur_v2
    mov     x1, x0
    ldr     w0, [x5, #SYM_REG]
    sxtw    x0, w0
    sub     x2, x0, x1
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      emit32_v2
    V2_PUSH
    mov     w5, w0
    cmp     w5, #0
    b.eq    .Lv2lr_done
    mov     w0, w5
    mov     w1, #0
    bl      emit_mov_reg_v2
    b       .Lv2lr_done

.Lv2lr_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc_v2
    b       .Lv2lr_done

.Lv2lr_unknown_ident:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2lr_done

.Lv2lr_rr:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      tok_is_dollar_v2
    cbz     w0, .Lv2lr_done
    bl      parse_dollar_reg_v2
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2
    b       .Lv2lr_done

.Lv2lr_rw:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      tok_is_dollar_v2
    cbz     w0, .Lv2lr_done
    bl      parse_dollar_reg_v2
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v2_eval_operand
    V2_POP
    mov     w5, w0
    mov     w0, w4
    mov     w1, w5
    bl      emit_mov_reg_v2
    b       .Lv2lr_done

.Lv2lr_load:
    add     x19, x19, #TOK_STRIDE_SZ
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    .Lv2lr_done
    bl      parse_int_v2
    mov     x10, x0
    add     x19, x19, #TOK_STRIDE_SZ
    stp     x10, xzr, [sp, #-16]!
    bl      v2_eval_operand
    ldp     x10, xzr, [sp], #16
    V2_POP
    mov     w4, w0
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    mov     x2, x10
    bl      emit_load_width_v2
    b       .Lv2lr_done

.Lv2lr_store:
    add     x19, x19, #TOK_STRIDE_SZ
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    .Lv2lr_done
    bl      parse_int_v2
    mov     x10, x0
    add     x19, x19, #TOK_STRIDE_SZ
    stp     x10, xzr, [sp, #-16]!
    bl      v2_eval_operand
    bl      v2_eval_operand
    ldp     x10, xzr, [sp], #16
    V2_POP
    mov     w6, w0
    V2_POP
    mov     w5, w0
    mov     w0, w6
    mov     w1, w5
    mov     x2, x10
    bl      emit_store_width_v2
    b       .Lv2lr_done

.Lv2lr_unary:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      at_line_end
    cbnz    w0, .Lv2lr_done
    bl      v2_eval_operand
    b       .Lv2lr_done

.Lv2lr_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v2_eval_operand — evaluate one operand token, push onto stack
//   Handles: INT, known IDENT (param/local), $N register, composition call
// ============================================================
v2_eval_operand:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_INT
    b.eq    .Lv2op_int

    cmp     w0, #TOK_IDENT
    b.eq    .Lv2op_ident

    cmp     w0, #TOK_MINUS
    b.eq    .Lv2op_neg_int

    cmp     w0, #TOK_TRAP
    b.eq    .Lv2op_trap

    // Unknown — skip
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2op_done

.Lv2op_int:
    bl      parse_int_v2
    mov     x4, x0
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    bl      emit_mov_imm64_v2
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2op_done

.Lv2op_neg_int:
    // Check if next token is INT (negative literal)
    mov     x4, x19
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lv2op_done
    ldr     w0, [x4]
    cmp     w0, #TOK_INT
    b.ne    .Lv2op_done
    // Parse as negative: consume '-' then parse int
    add     x19, x19, #TOK_STRIDE_SZ  // skip '-'
    bl      parse_int_v2
    neg     x4, x0
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    bl      emit_mov_imm64_v2
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2op_done

.Lv2op_ident:
    // Check if starts with '$'
    bl      tok_is_dollar_v2
    cbnz    w0, .Lv2op_dollar

    // Check for trap
    bl      tok_is_trap
    cbnz    w0, .Lv2op_trap

    // Look up
    bl      sym_lookup_v2
    cbz     x0, .Lv2op_unknown

    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_COMP
    b.eq    .Lv2op_comp

    // Known param/local — push
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2
    b       .Lv2op_done

.Lv2op_dollar:
    bl      parse_dollar_reg_v2
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ
    V2_PUSH
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg_v2
    b       .Lv2op_done

.Lv2op_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc_v2
    // Push X0 (return value) onto stack
    V2_PUSH
    mov     w5, w0
    cmp     w5, #0
    b.eq    .Lv2op_done
    mov     w0, w5
    mov     w1, #0
    bl      emit_mov_reg_v2
    b       .Lv2op_done

.Lv2op_comp:
    // Composition call as operand — parse args, call, push result
    mov     x5, x0
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w8, #0
.Lv2op_comp_args:
    bl      at_line_end
    cbnz    w0, .Lv2op_comp_emit
    stp     x5, x8, [sp, #-16]!
    bl      v2_eval_operand
    ldp     x5, x8, [sp], #16
    V2_POP
    mov     w4, w0
    cmp     w4, w8
    b.eq    .Lv2op_comp_skip
    mov     w0, w8
    mov     w1, w4
    bl      emit_mov_reg_v2
.Lv2op_comp_skip:
    add     w8, w8, #1
    b       .Lv2op_comp_args
.Lv2op_comp_emit:
    ldr     w0, [x5, #SYM_REG]
    bl      emit_cur_v2
    mov     x1, x0
    ldr     w0, [x5, #SYM_REG]
    sxtw    x0, w0
    sub     x2, x0, x1
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      emit32_v2
    V2_PUSH
    mov     w5, w0
    cmp     w5, #0
    b.eq    .Lv2op_done
    mov     w0, w5
    mov     w1, #0
    bl      emit_mov_reg_v2
    b       .Lv2op_done

.Lv2op_unknown:
    // Unknown ident — skip
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv2op_done

.Lv2op_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v2_try_continuation — after pushing a value, check if there's
//   a binary op remaining on the same line and apply it.
// ============================================================
v2_try_continuation:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    bl      at_line_end
    cbnz    w0, .Lv2cont_done

    ldr     w0, [x19]
    cmp     w0, #TOK_PLUS
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_AMP
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_PIPE
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_CARET
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_SHL
    b.eq    .Lv2cont_binop
    cmp     w0, #TOK_SHR
    b.eq    .Lv2cont_binop
    b       .Lv2cont_done

.Lv2cont_binop:
    // There's a binary op — delegate to line_rest
    bl      v2_parse_line_rest
    b       .Lv2cont_done

.Lv2cont_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Error handlers
// ============================================================

v2_parse_error:
    adrp    x1, v2_err_parse
    add     x1, x1, :lo12:v2_err_parse
    mov     x2, #16
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

v2_err_overflow:
    adrp    x1, v2_err_overflow_str
    add     x1, x1, :lo12:v2_err_overflow_str
    mov     x2, #22
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

v2_err_underflow:
    adrp    x1, v2_err_underflow_str
    add     x1, x1, :lo12:v2_err_underflow_str
    mov     x2, #23
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

// ============================================================
// DTC wrapper — code_PARSE_TOKENS_V2
//   Stack: ( tok-buf tok-count src-buf -- )
// ============================================================
.align 4
.global code_PARSE_TOKENS_V2
code_PARSE_TOKENS_V2:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x20, [sp, #-16]!

    mov     x2, x22               // TOS = src-buf
    ldr     x1, [x24], #8         // tok-count
    ldr     x0, [x24], #8         // tok-buf
    ldr     x22, [x24], #8        // new TOS

    bl      parse_tokens_v2

    ldp     x22, x20, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// ============================================================
// .data section
// ============================================================
.data
.align 3

v2_err_parse:
    .ascii "v2: parse error\n"

v2_err_overflow_str:
    .ascii "v2: stack overflow\n"
    .byte 0, 0, 0

v2_err_underflow_str:
    .ascii "v2: stack underflow\n"
    .byte 0, 0, 0

.align 3
str_trap_v2:
    .ascii "trap"
str_main_v2:
    .ascii "main"

// ============================================================
// .bss section — parser state (v2-private)
// ============================================================

.extern ls_sym_count
.extern ls_sym_table
.extern ls_code_buf
.extern ls_code_pos

.bss
.align 3

v2_scope_depth:     .space 4
    .align 3

v2_sp_reg:          .space 4
    .align 3

v2_emit_ptr:        .space 8
    .align 3

v2_main_addr:       .space 8
    .align 3

v2_entry_patch:     .space 8
    .align 3

// ============================================================
// Dictionary entry — chains after entry_e_patch_cbnz in emit-arm64.s
// ============================================================
.data
.align 3

// Provide entry_p_parse_tokens for the dictionary chain (lithos-elf-writer.s
// chains from it). When v2 replaces v1, this entry takes v1's place in the chain.
entry_p_parse_tokens:
    .quad   entry_e_cbnz_fwd
    .byte   0
    .byte   12
    .ascii  "parse-tokens"
    .align  3
    .quad   code_PARSE_TOKENS_V2

entry_p_parse_tokens_v2:
    .quad   entry_p_parse_tokens
    .byte   0
    .byte   15
    .ascii  "parse-tokens-v2"
    .align  3
    .quad   code_PARSE_TOKENS_V2
