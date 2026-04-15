// lithos-table.s — Table-driven compiler for Lithos .ls source files
//
// Replaces lithos-parser.s, lithos-expr-eval.s, lithos-compose.s,
// and lithos-control.s with a single flat lookup-table approach.
//
// The insight: every Lithos construct maps to 1-3 ARM64 instructions.
// No AST, no precedence climbing, no expression trees. Just:
//   read token → look up handler → emit bytes.
//
// For expressions: left-to-right with explicit parens for grouping.
// Each operator emits one instruction. Bump register allocator,
// reset between statements.
//
// Register conventions (inherited from lithos-bootstrap.s):
//   X19 = TOKP   — pointer to current token triple
//   X27 = TOKEND — pointer past last token
//   X28 = SRC    — pointer to source buffer
//
// Token triple layout: [+0] u32 type, [+4] u32 offset, [+8] u32 length
//   Stride = 12 bytes per token.
//
// Build: replaces lithos-parser.s + lithos-expr-eval.s + lithos-compose.s
//        + lithos-control.s in build.sh

// ============================================================
// Token type constants
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
.equ TOK_LOAD,      36
.equ TOK_STORE,     37
.equ TOK_REG_READ,  38
.equ TOK_REG_WRITE, 39
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_EQ,        54
// All keyword tokens that compiler.ls may use as identifiers
.equ TOK_PARAM,     12
.equ TOK_WEIGHT,    25
.equ TOK_LAYER,     26
.equ TOK_BIND,      27
.equ TOK_RUNTIME,   28
.equ TOK_TEMPLATE,  29
.equ TOK_PROJECT,   30
.equ TOK_SHARED,    31
.equ TOK_BARRIER,   32
.equ TOK_F32,       40
.equ TOK_U32,       41
.equ TOK_S32,       42
.equ TOK_F16,       43
.equ TOK_PTR,       44
.equ TOK_VOID,      45

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
.equ TOK_COMMA,     71
.equ TOK_COLON,     72
.equ TOK_DOT,       73
.equ TOK_SUM,       75
.equ TOK_MAX,       76
.equ TOK_MIN,       77
.equ TOK_INDEX,     78
.equ TOK_SQRT,      79
.equ TOK_SIN,       80
.equ TOK_COS,       81
.equ TOK_TRAP,      89
.equ TOK_DOLLAR,    97

.equ TOK_STRIDE_SZ, 12

// Symbol table entry layout
.equ SYM_SIZE,      24
.equ SYM_NAME_OFF,  0
.equ SYM_NAME_LEN,  4
.equ SYM_KIND,      8
.equ SYM_REG,       12
.equ SYM_DEPTH,     16

// Symbol kinds
.equ KIND_LOCAL_REG, 0
.equ KIND_PARAM,     1
.equ KIND_VAR,       2
.equ KIND_BUF,       3
.equ KIND_COMP,      4
.equ KIND_CONST,     6

.equ MAX_SYMS,      1024
.equ MAX_PATCH,     64
.equ MAX_LOOP,      16

// ARM64 instruction constants
.equ ARM64_NOP,     0xD503201F
.equ ARM64_RET,     0xD65F03C0
// TARGET program SVC — must stay Linux encoding even on macOS builds.
// Split into hi/lo to prevent darwin transform from matching "svc #0".
.equ ARM64_SVC_0,   (0xD400 << 16) | 0x0001

// Register allocator range for TARGET program (NOT the parser's own registers).
// The target program uses X9-X28 freely. The parser's own X19-X28 are separate.
// X18 is reserved on macOS but usable in the target (Linux GH200).
.equ REG_FIRST,     9
.equ REG_LAST,      27
.equ REG_BSS_BASE,  28     // X28 reserved to hold BSS base address

// Condition codes
.equ CC_EQ, 0
.equ CC_NE, 1
.equ CC_LT, 11
.equ CC_GE, 10
.equ CC_GT, 12
.equ CC_LE, 13

// Code buffer size
.equ CODE_BUF_SIZE, 1048576

// ============================================================
// Macros
// ============================================================
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm

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

// ============================================================
// External symbols from ls-shared.s
// ============================================================
.extern ls_code_buf
.extern ls_code_pos
.extern ls_sym_count
.extern ls_sym_table
.extern ls_token_buf
.extern ls_bss_offset
.extern ls_fwd_gotos
.extern ls_n_fwd_gotos

// ============================================================
// .text — all code
// ============================================================
.text
.align 4

// ============================================================
// Token access helpers
// ============================================================

tok_type:
    ldr     w0, [x19]
    ret

tok_text:
    ldr     w1, [x19, #8]
    ldr     w0, [x19, #4]
    add     x0, x28, x0
    ret

advance:
    add     x19, x19, #TOK_STRIDE_SZ
    ret

skip_newlines:
    // Skip ONLY newline tokens — preserve indents for body detection
1:  cmp     x19, x27
    b.hs    3f
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.ne    3f
    add     x19, x19, #TOK_STRIDE_SZ
    b       1b
3:  ret

// at_line_end — check if current token ends a line
//   Returns: Z flag set if at line end (NEWLINE, EOF, or past end)
at_line_end:
    cmp     x19, x27
    b.hs    1f
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    1f
    cmp     w0, #TOK_EOF
    b.eq    1f
    cmp     w0, #TOK_INDENT
1:  ret

// ============================================================
// Symbol table — flat array, linear scan
// ============================================================

sym_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    ldr     w0, [x19, #4]          // token offset
    ldr     w1, [x19, #8]          // token length
    adrp    x2, ls_sym_count
    add     x2, x2, :lo12:ls_sym_count
    ldr     w3, [x2]
    cbz     w3, .Lsym_notfound
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
    sub     w3, w3, #1
    mov     w5, #SYM_SIZE
    madd    x6, x3, x5, x4
.Lsym_loop:
    ldr     w7, [x6, #SYM_NAME_LEN]
    cmp     w7, w1
    b.ne    .Lsym_next
    ldr     w7, [x6, #SYM_NAME_OFF]
    add     x7, x28, x7
    add     x2, x28, x0
    mov     w5, #0
.Lsym_cmp:
    cmp     w5, w1
    b.ge    .Lsym_found
    ldrb    w3, [x7, x5]
    ldrb    w4, [x2, x5]
    cmp     w3, w4
    b.ne    .Lsym_next
    add     w5, w5, #1
    b       .Lsym_cmp
.Lsym_found:
    mov     x0, x6
    b       .Lsym_done
.Lsym_next:
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
    cmp     x6, x4
    b.ls    .Lsym_notfound
    sub     x6, x6, #SYM_SIZE
    b       .Lsym_loop
.Lsym_notfound:
    mov     x0, #0
.Lsym_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

sym_add:
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

sym_pop_scope:
    stp     x30, xzr, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    mov     w2, w0
    adrp    x3, ls_sym_count
    add     x3, x3, :lo12:ls_sym_count
    adrp    x4, ls_sym_table
    add     x4, x4, :lo12:ls_sym_table
.Lpop_loop:
    ldr     w0, [x3]
    cbz     w0, .Lpop_done
    sub     w0, w0, #1
    mov     w1, #SYM_SIZE
    madd    x1, x0, x1, x4
    ldr     w1, [x1, #SYM_DEPTH]
    cmp     w1, w2
    b.le    .Lpop_done
    str     w0, [x3]
    b       .Lpop_loop
.Lpop_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Register allocator — bump within statement, reset between
// ============================================================

alloc_reg:
    adrp    x0, next_reg
    add     x0, x0, :lo12:next_reg
    ldr     w1, [x0]
    cmp     w1, #REG_LAST
    b.gt    .Lalloc_spill
    add     w2, w1, #1
    str     w2, [x0]
    mov     w0, w1
    ret

.Lalloc_spill:
    // Spill the register at min(reg_floor, REG_LAST) — the first
    // temporary slot, but never x28 (BSS_BASE) or above.  If
    // reg_floor has walked off the end of the allocator pool we must
    // still recycle something in the legal range; REG_LAST is the
    // highest register our code is allowed to touch.
    stp     x29, x30, [sp, #-16]!
    adrp    x2, reg_floor
    add     x2, x2, :lo12:reg_floor
    ldr     w2, [x2]
    cmp     w2, #REG_LAST
    b.le    1f
    mov     w2, #REG_LAST
1:  // ARM64 encoding: STR Xt, [SP, #-16]! = 0xF81F0FE0 | Rt
    mov     w0, #0x0FE0
    movk    w0, #0xF81F, lsl #16
    add     w0, w0, w2                  // Rt = spill reg
    bl      emit32
    // Increment spill count
    adrp    x0, spill_count
    add     x0, x0, :lo12:spill_count
    ldr     w1, [x0]
    add     w1, w1, #1
    str     w1, [x0]
    ldp     x29, x30, [sp], #16
    // Return the capped spill reg (min(reg_floor, REG_LAST))
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    ldr     w0, [x0]
    cmp     w0, #REG_LAST
    b.le    2f
    mov     w0, #REG_LAST
2:  ret

free_reg:
    // Reclaim: set next_reg = w0 (frees everything above)
    // If spilled registers exist and we're freeing below the spill point,
    // emit fills to restore them.
    adrp    x1, next_reg
    add     x1, x1, :lo12:next_reg
    str     w0, [x1]
    // Check if we need to restore spilled registers
    stp     x29, x30, [sp, #-16]!
    adrp    x2, spill_count
    add     x2, x2, :lo12:spill_count
    ldr     w3, [x2]
    cbz     w3, .Lfree_done
    // Emit LDR X<reg_floor>, [SP], #16 for each spilled register (LIFO).
    // Must match .Lalloc_spill which pushed reg_floor, not REG_FIRST.
.Lfree_fill_loop:
    cbz     w3, .Lfree_fill_done
    // ARM64 encoding: LDR Xt, [SP], #16 = 0xF84107E0 | Rt  (post-index)
    // Fill register must match what .Lalloc_spill pushed, which is
    // min(reg_floor, REG_LAST).
    stp     w0, w3, [sp, #-16]!
    stp     x1, x2, [sp, #-16]!
    adrp    x4, reg_floor
    add     x4, x4, :lo12:reg_floor
    ldr     w4, [x4]
    cmp     w4, #REG_LAST
    b.le    3f
    mov     w4, #REG_LAST
3:  mov     w0, #0x07E0
    movk    w0, #0xF841, lsl #16
    add     w0, w0, w4                 // Rt = capped fill reg
    bl      emit32
    ldp     x1, x2, [sp], #16
    ldp     w0, w3, [sp], #16
    sub     w3, w3, #1
    b       .Lfree_fill_loop
.Lfree_fill_done:
    str     wzr, [x2]              // spill_count = 0
.Lfree_done:
    ldp     x29, x30, [sp], #16
    ret

reset_regs:
    stp     x29, x30, [sp, #-16]!
    adrp    x0, next_reg
    add     x0, x0, :lo12:next_reg
    adrp    x1, reg_floor
    add     x1, x1, :lo12:reg_floor
    ldr     w2, [x1]
    str     w2, [x0]
    // Emit fills for any outstanding spills (balance target stack)
    adrp    x0, spill_count
    add     x0, x0, :lo12:spill_count
    ldr     w3, [x0]
    cbz     w3, .Lreset_no_spills
.Lreset_fill:
    cbz     w3, .Lreset_fill_done
    stp     x0, x3, [sp, #-16]!
    // Emit: LDR X<REG_FIRST>, [SP], #16
    mov     w0, #0x0FE0
    movk    w0, #0xF841, lsl #16
    add     w0, w0, #REG_FIRST
    bl      emit32
    ldp     x0, x3, [sp], #16
    sub     w3, w3, #1
    b       .Lreset_fill
.Lreset_fill_done:
    str     wzr, [x0]              // spill_count = 0
.Lreset_no_spills:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// ARM64 code emission
// ============================================================

emit32:
    adrp    x1, emit_ptr
    add     x1, x1, :lo12:emit_ptr
    ldr     x2, [x1]
    str     w0, [x2], #4
    str     x2, [x1]
    ret

emit_cur:
    adrp    x0, emit_ptr
    add     x0, x0, :lo12:emit_ptr
    ldr     x0, [x0]
    ret

emit_svc:
    MOVI32  w0, ARM64_SVC_0
    b       emit32

emit_ret_inst:
    MOVI32  w0, ARM64_RET
    b       emit32

emit_nop:
    MOVI32  w0, ARM64_NOP
    b       emit32

// emit_mov_imm16 — MOVZ Xd, #imm16
emit_mov_imm16:
    and     w2, w1, #0xFFFF
    lsl     w2, w2, #5
    orr     w2, w2, w0
    ORRIMM  w2, 0xD2800000, w16
    mov     w0, w2
    b       emit32

// emit_movk_imm16 — MOVK Xd, #imm16, LSL #shift
emit_movk_imm16:
    lsr     w3, w2, #4
    lsl     w3, w3, #21
    and     w4, w1, #0xFFFF
    lsl     w4, w4, #5
    orr     w4, w4, w0
    orr     w4, w4, w3
    ORRIMM  w4, 0xF2800000, w16
    mov     w0, w4
    b       emit32

// emit_mov_imm64 — load full 64-bit immediate into Xd
//   w0 = dest reg, x1 = imm64
emit_mov_imm64:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!
    and     w2, w1, #0xFFFF
    mov     w3, w0
    mov     w1, w2
    bl      emit_mov_imm16
    ldp     x0, x1, [sp]
    lsr     x4, x1, #16
    and     w1, w4, #0xFFFF
    cbz     w1, .Lm64_c32
    mov     w0, w3
    mov     w2, #16
    bl      emit_movk_imm16
    ldp     x0, x1, [sp]
.Lm64_c32:
    lsr     x4, x1, #32
    and     w1, w4, #0xFFFF
    cbz     w1, .Lm64_c48
    mov     w0, w3
    mov     w2, #32
    bl      emit_movk_imm16
    ldp     x0, x1, [sp]
.Lm64_c48:
    lsr     x4, x1, #48
    and     w1, w4, #0xFFFF
    cbz     w1, .Lm64_done
    mov     w0, w3
    mov     w2, #48
    bl      emit_movk_imm16
.Lm64_done:
    add     sp, sp, #16
    ldp     x30, x19, [sp], #16
    ret

// emit_mov_reg — MOV Xd, Xm (ORR Xd, XZR, Xm)
//   w0 = d, w1 = m
emit_mov_reg:
    lsl     w2, w1, #16
    mov     w3, #31
    lsl     w3, w3, #5
    orr     w4, w0, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    b       emit32

// Arithmetic emitters: ADD, SUB, MUL, SDIV, AND, ORR, EOR, LSL, LSR
//   w0=d, w1=n, w2=m

emit_add_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8B000000, w16
    mov     w0, w5
    b       emit32

emit_sub_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCB000000, w16
    mov     w0, w5
    b       emit32

emit_mul_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9B007C00, w16
    mov     w0, w5
    b       emit32

emit_sdiv_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC00C00, w16
    mov     w0, w5
    b       emit32

emit_and_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8A000000, w16
    mov     w0, w5
    b       emit32

emit_orr_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xAA000000, w16
    mov     w0, w5
    b       emit32

emit_eor_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCA000000, w16
    mov     w0, w5
    b       emit32

emit_lsl_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02000, w16
    mov     w0, w5
    b       emit32

emit_lsr_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC02400, w16
    mov     w0, w5
    b       emit32

// Compare and condition
emit_cmp_reg:
    lsl     w2, w1, #16
    lsl     w3, w0, #5
    mov     w4, #31
    orr     w4, w4, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xEB000000, w16
    mov     w0, w4
    b       emit32

emit_cset:
    eor     w2, w1, #1
    lsl     w2, w2, #12
    mov     w3, #31
    lsl     w3, w3, #5
    mov     w4, #31
    lsl     w4, w4, #16
    orr     w5, w0, w3
    orr     w5, w5, w4
    orr     w5, w5, w2
    ORRIMM  w5, 0x9A800400, w16
    mov     w0, w5
    b       emit32

// Memory access emitters
emit_ldr_imm:
    lsr     w3, w2, #3
    and     w3, w3, #0xFFF
    lsl     w3, w3, #10
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xF9400000, w16
    mov     w0, w5
    b       emit32

emit_str_imm:
    lsr     w3, w2, #3
    and     w3, w3, #0xFFF
    lsl     w3, w3, #10
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xF9000000, w16
    mov     w0, w5
    b       emit32

emit_ldr_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xF8606800, w16
    mov     w0, w5
    b       emit32

emit_str_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xF8206800, w16
    mov     w0, w5
    b       emit32

emit_ldrb_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x38606800, w16
    mov     w0, w5
    b       emit32

emit_strb_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x38206800, w16
    mov     w0, w5
    b       emit32

emit_ldrh_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x78606800, w16
    mov     w0, w5
    b       emit32

emit_strh_reg:
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x78206800, w16
    mov     w0, w5
    b       emit32

// Branch emitters
emit_b:
    stp     x30, xzr, [sp, #-16]!
    adrp    x1, emit_ptr
    add     x1, x1, :lo12:emit_ptr
    ldr     x2, [x1]
    sub     x3, x0, x2
    asr     x3, x3, #2
    and     w3, w3, #0x3FFFFFF
    ORRIMM  w3, 0x14000000, w16
    mov     w0, w3
    ldp     x30, xzr, [sp], #16
    b       emit32

emit_b_cond:
    stp     x30, x0, [sp, #-16]!
    adrp    x2, emit_ptr
    add     x2, x2, :lo12:emit_ptr
    ldr     x3, [x2]
    sub     x4, x1, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5
    ldp     x30, x0, [sp], #16
    orr     w4, w4, w0
    ORRIMM  w4, 0x54000000, w16
    mov     w0, w4
    b       emit32

emit_cbz:
    stp     x30, xzr, [sp, #-16]!
    mov     w5, w0
    adrp    x2, emit_ptr
    add     x2, x2, :lo12:emit_ptr
    ldr     x3, [x2]
    sub     x4, x1, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5
    orr     w4, w4, w5
    ORRIMM  w4, 0xB4000000, w16
    mov     w0, w4
    ldp     x30, xzr, [sp], #16
    b       emit32

emit_cbnz:
    stp     x30, xzr, [sp, #-16]!
    mov     w5, w0
    adrp    x2, emit_ptr
    add     x2, x2, :lo12:emit_ptr
    ldr     x3, [x2]
    sub     x4, x1, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5
    orr     w4, w4, w5
    ORRIMM  w4, 0xB5000000, w16
    mov     w0, w4
    ldp     x30, xzr, [sp], #16
    b       emit32

emit_bl:
    // x0 = target address
    stp     x30, xzr, [sp, #-16]!
    adrp    x1, emit_ptr
    add     x1, x1, :lo12:emit_ptr
    ldr     x2, [x1]
    sub     x3, x0, x2
    asr     x3, x3, #2
    and     w3, w3, #0x3FFFFFF
    ORRIMM  w3, 0x94000000, w16
    mov     w0, w3
    ldp     x30, xzr, [sp], #16
    b       emit32

// Patch helpers
patch_b:
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x14000000, w16
    str     w2, [x0]
    ret

patch_b_cond:
    ldr     w3, [x0]
    and     w3, w3, #0xF
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x7FFFF
    lsl     w2, w2, #5
    orr     w2, w2, w3
    ORRIMM  w2, 0x54000000, w16
    str     w2, [x0]
    ret

patch_cbz:
    // Patch CBZ at x0 to target x1
    ldr     w2, [x0]
    sub     x3, x1, x0
    asr     x3, x3, #2
    and     w3, w3, #0x7FFFF
    lsl     w3, w3, #5
    and     w2, w2, #0xFF00001F     // keep opcode + Rt
    orr     w2, w2, w3
    str     w2, [x0]
    ret

// ============================================================
// Number parsing
// ============================================================
parse_int_literal:
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
    b.ne    .Lint_check_hex
    mov     w7, #1
    add     w3, w3, #1
.Lint_check_hex:
    sub     w5, w2, w3
    cmp     w5, #2
    b.lt    .Lint_decimal
    ldrb    w4, [x1, x3]
    cmp     w4, #'0'
    b.ne    .Lint_decimal
    add     w6, w3, #1
    ldrb    w4, [x1, x6]
    cmp     w4, #'x'
    b.eq    .Lint_hex
    cmp     w4, #'X'
    b.ne    .Lint_decimal
.Lint_hex:
    add     w3, w3, #2
.Lint_hex_loop:
    cmp     w3, w2
    b.ge    .Lint_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'9'
    b.le    .Lhex_09
    cmp     w4, #'F'
    b.le    .Lhex_AF
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lhex_accum
.Lhex_AF:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lhex_accum
.Lhex_09:
    sub     w4, w4, #'0'
.Lhex_accum:
    lsl     x0, x0, #4
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lint_hex_loop
.Lint_decimal:
.Ldec_loop:
    cmp     w3, w2
    b.ge    .Lint_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'.'
    b.eq    .Lint_sign
    sub     w4, w4, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Ldec_loop
.Lint_sign:
    cbz     w7, .Lint_done
    neg     x0, x0
.Lint_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// parse_dollar_reg — parse "$N" token, return N in w0, advance x19
parse_dollar_reg:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error
    ldr     w1, [x19, #8]
    cmp     w1, #2
    b.lt    parse_error
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'$'
    b.ne    parse_error
    mov     w0, #0
    mov     w4, #1
.Ldollar_lp:
    cmp     w4, w1
    b.ge    .Ldollar_end
    ldrb    w5, [x2, x4]
    sub     w6, w5, #'0'
    cmp     w6, #9
    b.hi    parse_error
    mov     w7, #10
    mul     w0, w0, w7
    add     w0, w0, w6
    add     w4, w4, #1
    b       .Ldollar_lp
.Ldollar_end:
    cmp     w0, #30
    b.hi    parse_error
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// PARSER ENTRY POINT
// ============================================================
// parse_tokens — main entry
//   x0 = pointer to token buffer
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

    // Initialize emit pointer
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, emit_ptr
    add     x1, x1, :lo12:emit_ptr
    str     x0, [x1]

    // Clear symbol table
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    str     wzr, [x0]

    // Reset BSS allocation offset (for buf declarations)
    adrp    x0, ls_bss_offset
    add     x0, x0, :lo12:ls_bss_offset
    str     xzr, [x0]

    // Reset forward-goto patch table count
    adrp    x0, ls_n_fwd_gotos
    add     x0, x0, :lo12:ls_n_fwd_gotos
    str     wzr, [x0]

    // Clear scope
    adrp    x0, scope_depth
    add     x0, x0, :lo12:scope_depth
    str     wzr, [x0]

    // Initialize register allocator
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    mov     w1, #REG_FIRST
    str     w1, [x0]
    bl      reset_regs

    // Main loop — the table-driven core
    bl      parse_toplevel

    // Sync ls_code_pos
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, emit_ptr
    add     x1, x1, :lo12:emit_ptr
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
// parse_toplevel — token dispatch loop
//   Read token type, look up handler, call it. Repeat.
// ============================================================
parse_toplevel:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Ltop_loop:
    cmp     x19, x27
    b.hs    .Ltop_done
    bl      skip_newlines
    cmp     x19, x27
    b.hs    .Ltop_done

    ldr     w0, [x19]

    cmp     w0, #TOK_EOF
    b.eq    .Ltop_done
    cmp     w0, #TOK_CONST
    b.eq    .Ltop_const
    cmp     w0, #TOK_VAR
    b.eq    .Ltop_var
    cmp     w0, #TOK_BUF
    b.eq    .Ltop_buf
    cmp     w0, #TOK_IDENT
    b.eq    .Ltop_ident
    // "host" and "kernel" prefixes on compositions — skip the prefix
    cmp     w0, #TOK_HOST
    b.eq    .Ltop_skip_prefix
    cmp     w0, #TOK_KERNEL
    b.eq    .Ltop_skip_prefix
    // Forth-style constant declaration:  INT  IDENT("constant")  IDENT(NAME)
    cmp     w0, #TOK_INT
    b.eq    .Ltop_maybe_forth_const

    // Any keyword 11-45 not already dispatched → treat as identifier
    cmp     w0, #11
    b.lt    .Ltop_skip_unknown
    cmp     w0, #45
    b.le    .Ltop_ident
.Ltop_skip_unknown:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Ltop_loop

.Ltop_maybe_forth_const:
    // Need next two tokens: IDENT("constant"), IDENT(NAME).
    add     x4, x19, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Ltop_skip_unknown
    ldr     w5, [x4]
    cmp     w5, #TOK_IDENT
    b.ne    .Ltop_skip_unknown
    // Length must be 8 ("constant" is 8 chars).
    ldr     w5, [x4, #8]
    cmp     w5, #8
    b.ne    .Ltop_skip_unknown
    // Compare source bytes to "constant".
    ldr     w5, [x4, #4]            // source offset
    add     x5, x28, x5
    ldr     x6, [x5]                // 8 bytes starting at the ident
    movz    x7, #0x6f63
    movk    x7, #0x736e, lsl #16
    movk    x7, #0x6174, lsl #32
    movk    x7, #0x746e, lsl #48
    cmp     x6, x7
    b.ne    .Ltop_skip_unknown
    // Third token must be IDENT (the name we're defining).
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Ltop_skip_unknown
    ldr     w5, [x4]
    cmp     w5, #TOK_IDENT
    b.ne    .Ltop_skip_unknown
    // Parse the int literal at x19.
    stp     x4, xzr, [sp, #-16]!
    bl      parse_int_literal
    ldp     x4, xzr, [sp], #16
    mov     w6, w0                  // const value
    // sym_add reads name from x19 — point it at the NAME token.
    mov     x19, x4
    mov     w1, #KIND_CONST
    mov     w2, w6
    mov     w3, #0
    bl      sym_add
    // Advance past INT, "constant", NAME.
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Ltop_loop
.Ltop_skip_prefix:
    // Skip "host" or "kernel" token, then re-enter the loop
    // to parse the actual composition name
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Ltop_loop

.Ltop_const:
    bl      handle_const
    b       .Ltop_loop
.Ltop_var:
    bl      handle_var
    b       .Ltop_loop
.Ltop_buf:
    bl      handle_buf
    b       .Ltop_loop
.Ltop_ident:
    bl      handle_toplevel_ident
    b       .Ltop_loop
.Ltop_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_const — const name [=] value
// ============================================================
handle_const:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'const'
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error
    mov     x4, x19                      // save name position
    add     x19, x19, #TOK_STRIDE_SZ   // skip name
    // Optional '='
    ldr     w0, [x19]
    cmp     w0, #TOK_EQ
    b.ne    1f
    add     x19, x19, #TOK_STRIDE_SZ
1:  bl      parse_int_literal
    mov     x5, x0                      // value
    mov     x19, x4                      // restore to name token
    mov     w1, #KIND_CONST
    mov     w2, w5
    mov     w3, #0
    bl      sym_add
    // Advance past: name [=] value
    mov     x19, x4
    add     x19, x19, #TOK_STRIDE_SZ   // skip name
    ldr     w0, [x19]
    cmp     w0, #TOK_EQ
    b.ne    2f
    add     x19, x19, #TOK_STRIDE_SZ
2:  add     x19, x19, #TOK_STRIDE_SZ   // skip value
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_var — var name [value]
// ============================================================
handle_var:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'var'
    // Accept any token as name
    ldr     w0, [x19]
    ldr     w1, [x19, #8]
    cbz     w1, parse_error

    stp     x19, xzr, [sp, #-16]!
    bl      alloc_reg
    ldp     x19, xzr, [sp], #16
    mov     w4, w0

    mov     w1, #KIND_VAR
    mov     w2, w4
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    // Same approach: parse_body skips reset after var declarations.
.Lvar_floor_ok:

    // Check for initial value
    cmp     x19, x27
    b.hs    .Lvar_zero
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lvar_zero
    cmp     w0, #TOK_EOF
    b.eq    .Lvar_zero
    cmp     w0, #TOK_INDENT
    b.eq    .Lvar_zero

    // Has initial value — parse expression
    stp     x4, xzr, [sp, #-16]!
    bl      parse_expr
    ldp     x4, xzr, [sp], #16
    cmp     w0, w4
    b.eq    .Lvar_done
    // MOV Xvar, Xresult
    mov     w1, w0
    mov     w0, w4
    bl      emit_mov_reg
    b       .Lvar_done

.Lvar_zero:
    mov     w0, w4
    mov     x1, #0
    bl      emit_mov_imm64
.Lvar_done:
    mov     w0, #1                 // return 1 = var decl (don't reset regs)
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_buf — buf name size
// ============================================================
.globl parse_buf_decl
handle_buf:
parse_buf_decl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x20, x21, [sp, #-16]!
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'buf'
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error
    mov     x20, x19                    // x20 = save name token pos
    // Check if size follows the name
    add     x21, x19, #TOK_STRIDE_SZ   // peek at next token
    ldr     w0, [x21]
    mov     w2, #0                      // default buf size = 0
    cmp     w0, #TOK_INT
    b.ne    .Lbuf_add_sym
    // Parse the int literal to get the size value
    mov     x19, x21                    // point x19 at the int token
    bl      parse_int_literal           // x0 = parsed integer value
    mov     w2, w0                      // w2 = buf size for SYM_REG
    mov     x19, x20                    // restore x19 to name token
.Lbuf_add_sym:
    // Save size on stack (w2 gets clobbered by sym_add)
    stp     x2, xzr, [sp, #-16]!
    // Compute BSS offset: current ls_bss_offset
    adrp    x4, ls_bss_offset
    add     x4, x4, :lo12:ls_bss_offset
    ldr     x5, [x4]
    // Store BSS offset in SYM_REG instead of size.
    mov     w2, w5
    mov     w1, #KIND_BUF
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    // Restore size to x6
    ldp     x6, xzr, [sp], #16
    // Advance ls_bss_offset by buf size (aligned to 8 bytes)
    adrp    x4, ls_bss_offset
    add     x4, x4, :lo12:ls_bss_offset
    ldr     x5, [x4]
    add     x5, x5, x6
    add     x5, x5, #7
    and     x5, x5, #-8
    str     x5, [x4]
    add     x19, x19, #TOK_STRIDE_SZ   // skip name
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    .Lbuf_no_size
    add     x19, x19, #TOK_STRIDE_SZ   // skip size (already parsed)
.Lbuf_no_size:
    ldp     x20, x21, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// parse_data_decl is provided by lithos-glue.s

// ============================================================
// handle_toplevel_ident — disambiguate composition vs statement
//   Scan forward for COLON before NEWLINE/EOF to detect composition.
// ============================================================
handle_toplevel_ident:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, xzr, [sp, #-16]!

    // Lookahead: is there a COLON on this line?
    mov     x4, x19
.Lti_scan:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lti_stmt
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lti_comp
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lti_stmt
    cmp     w0, #TOK_EOF
    b.eq    .Lti_stmt
    b       .Lti_scan

.Lti_comp:
    ldp     x19, xzr, [sp], #16
    bl      handle_composition
    ldp     x29, x30, [sp], #16
    ret

.Lti_stmt:
    ldp     x19, xzr, [sp], #16
    bl      handle_ident_stmt
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_composition — name arg1 arg2 ... :
//   Emit function prologue, parse body, emit epilogue.
// ============================================================
handle_composition:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #48             // local storage

    // Save sym_count for scope cleanup
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    ldr     w4, [x0]
    str     w4, [sp, #0]           // [sp+0] = old sym_count

    // Save reg_floor and next_reg
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    ldr     w4, [x0]
    str     w4, [sp, #4]           // [sp+4] = old reg_floor
    adrp    x1, next_reg
    add     x1, x1, :lo12:next_reg
    ldr     w5, [x1]
    str     w5, [sp, #8]           // [sp+8] = old next_reg
    // Compositions always start with a fresh register window.  The
    // previous composition's bindings are dead once we pop its scope,
    // so reg_floor and next_reg both restart at REG_FIRST.  Inheriting
    // the prior next_reg leaked state across compositions and made
    // late-defined functions start with reg_floor near REG_LAST,
    // which immediately blew the spill path.
    mov     w5, #REG_FIRST
    str     w5, [x0]               // reg_floor = REG_FIRST
    str     w5, [x1]               // next_reg = REG_FIRST
    // Reset spill count — new composition has its own stack frame
    adrp    x0, spill_count
    add     x0, x0, :lo12:spill_count
    str     wzr, [x0]

    // Increment scope
    adrp    x0, scope_depth
    add     x0, x0, :lo12:scope_depth
    ldr     w5, [x0]
    add     w6, w5, #1
    str     w6, [x0]

    // Record composition name: register the emit address as symbol
    bl      emit_cur
    mov     w2, w0                  // code address
    str     x0, [sp, #16]          // save comp start address
    // Track last composition for entry point (bottom-of-file = entry)
    adrp    x6, ls_last_comp_addr
    add     x6, x6, :lo12:ls_last_comp_addr
    str     x0, [x6]
    mov     w1, #KIND_COMP
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    sub     w3, w3, #1             // comp itself at outer scope
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    // Emit prologue: STP X29, X30, [SP, #-16]!
    MOVI32  w0, 0xA9BF7BFD
    bl      emit32
    // MOV X29, SP
    MOVI32  w0, 0x910003FD
    bl      emit32

    // Parse arguments — identifiers (or type keywords used as names) before the colon
    mov     w8, #0                  // arg index
.Lcomp_args:
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lcomp_args_done
    // Accept IDENT or any keyword-as-name (tokens 11-45) as parameter names
    cmp     w0, #TOK_IDENT
    b.eq    .Lcomp_arg_ok
    cmp     w0, #11
    b.lt    parse_error
    cmp     w0, #45
    b.gt    parse_error
.Lcomp_arg_ok:

    // Register arg as param with register = arg index (X0-X7)
    mov     w1, #KIND_PARAM
    mov     w2, w8
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ
    add     w8, w8, #1
    cmp     w8, #8
    b.lt    .Lcomp_args
.Lcomp_args_done:
    add     x19, x19, #TOK_STRIDE_SZ   // skip ':'

    // Parse body
    bl      skip_newlines
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lcomp_no_body          // no indent token → empty body
    ldr     w9, [x19, #8]           // w9 = indent count (0 = column 0)
    cbz     w9, .Lcomp_no_body      // indent 0 = not indented → empty body
    bl      parse_body
.Lcomp_no_body:

    // Flush any pending register spills before epilogue
    bl      reset_regs

    // Emit epilogue: LDP X29, X30, [SP], #16; RET
    MOVI32  w0, 0xA8C17BFD
    bl      emit32
    bl      emit_ret_inst

    // Pop scope
    adrp    x0, scope_depth
    add     x0, x0, :lo12:scope_depth
    ldr     w1, [x0]
    sub     w1, w1, #1
    str     w1, [x0]

    // Pop symbols
    ldr     w0, [sp, #0]
    adrp    x1, ls_sym_count
    add     x1, x1, :lo12:ls_sym_count
    // Keep composition name, pop everything else after it
    // Actually: restore count to old+1 (composition entry stays)
    add     w0, w0, #1
    str     w0, [x1]

    // Restore reg_floor and next_reg
    ldr     w4, [sp, #4]
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    str     w4, [x0]
    ldr     w5, [sp, #8]
    adrp    x0, next_reg
    add     x0, x0, :lo12:next_reg
    str     w5, [x0]

    add     sp, sp, #48
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_body — parse indented block until dedent
//   w9 = minimum indent level
// ============================================================
parse_body:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    str     w9, [sp, #-16]!
.Lbody_loop:
    cmp     x19, x27
    b.hs    .Lbody_done
    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lbody_done
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lbody_newline
    cmp     w0, #TOK_INDENT
    b.ne    .Lbody_stmt
    // Check indent level
    ldr     w1, [x19, #8]
    ldr     w9, [sp]
    cmp     w1, w9
    b.ge    .Lbody_indent_ok
    // Indent < body level — but check for blank line first.
    // Blank line = INDENT(0) + NEWLINE → skip, don't exit body.
    cbnz    w1, .Lbody_done         // non-zero indent below body = real dedent
    add     x4, x19, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lbody_done
    ldr     w2, [x4]
    cmp     w2, #TOK_NEWLINE
    b.ne    .Lbody_done             // INDENT(0) + non-newline = real dedent
    // Blank line: skip INDENT(0) + NEWLINE
    add     x19, x19, #TOK_STRIDE_SZ
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lbody_loop
.Lbody_indent_ok:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lbody_loop
.Lbody_newline:
    add     x19, x19, #TOK_STRIDE_SZ
    // After newline: peek at next token to detect dedent.
    cmp     x19, x27
    b.hs    .Lbody_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lbody_loop         // blank line (NEWLINE NEWLINE) — skip
    cmp     w0, #TOK_EOF
    b.eq    .Lbody_done
    cmp     w0, #TOK_INDENT
    b.ne    .Lbody_done         // non-indent after newline = column 0 = done
    // It's an INDENT — check if it's a blank line (INDENT(0) + NEWLINE)
    ldr     w1, [x19, #8]      // indent level
    cbnz    w1, .Lbody_loop    // real indent → let loop check level
    // INDENT(0): peek ahead — if next is NEWLINE, it's a blank line, skip both
    add     x4, x19, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lbody_done
    ldr     w2, [x4]
    cmp     w2, #TOK_NEWLINE
    b.ne    .Lbody_done         // INDENT(0) + non-newline = real dedent
    // Blank line: skip INDENT(0) + NEWLINE
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lbody_newline      // process the NEWLINE (will advance + peek again)
.Lbody_stmt:
    bl      parse_statement
    // parse_statement returns w0: handle_binding/handle_var return 1,
    // all others return arbitrary values. Only skip reset on exactly 1.
    cmp     w0, #1
    b.eq    .Lbody_loop            // binding — don't recycle its register
    bl      reset_regs
    b       .Lbody_loop
.Lbody_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_statement — THE LOOKUP TABLE
//   Read token type, dispatch to handler.
// ============================================================
parse_statement:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_STORE
    b.eq    .Ls_store
    cmp     w0, #TOK_LOAD
    b.eq    .Ls_load
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Ls_reg_write
    cmp     w0, #TOK_REG_READ
    b.eq    .Ls_reg_read
    cmp     w0, #TOK_FOR
    b.eq    .Ls_for
    cmp     w0, #TOK_IF
    b.eq    .Ls_if
    cmp     w0, #TOK_WHILE
    b.eq    .Ls_while
    cmp     w0, #TOK_EACH
    b.eq    .Ls_each
    cmp     w0, #TOK_RETURN
    b.eq    .Ls_return
    cmp     w0, #TOK_VAR
    b.eq    .Ls_var
    cmp     w0, #TOK_BUF
    b.eq    .Ls_buf
    cmp     w0, #TOK_LABEL
    b.eq    .Ls_label
    cmp     w0, #TOK_IDENT
    b.eq    .Ls_ident
    // Any keyword 11-45 not already dispatched → treat as identifier
    cmp     w0, #11
    b.lt    .Ls_check_trap
    cmp     w0, #45
    b.le    .Ls_ident
.Ls_check_trap:
    cmp     w0, #TOK_TRAP
    b.eq    .Ls_trap
    // Bare expressions as statements (return values): INT, MINUS (negative), LOAD
    cmp     w0, #TOK_INT
    b.eq    .Ls_bare_expr
    cmp     w0, #TOK_MINUS
    b.eq    .Ls_bare_expr
    cmp     w0, #TOK_LOAD
    b.eq    .Ls_bare_expr
    cmp     w0, #TOK_LPAREN
    b.eq    .Ls_bare_expr

    // Unknown — skip
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

.Ls_bare_expr:
    // Bare expression as statement — evaluate and emit MOV X0, result
    bl      parse_expr
    cmp     w0, #0
    b.eq    1f
    mov     w1, w0
    mov     w0, #0
    bl      emit_mov_reg
1:  ldp     x29, x30, [sp], #16
    ret

.Ls_store:
    bl      handle_store
    ldp     x29, x30, [sp], #16
    ret
.Ls_load:
    bl      handle_load
    ldp     x29, x30, [sp], #16
    ret
.Ls_reg_write:
    bl      handle_reg_write
    ldp     x29, x30, [sp], #16
    ret
.Ls_reg_read:
    bl      handle_reg_read
    ldp     x29, x30, [sp], #16
    ret
.Ls_for:
    bl      handle_for
    ldp     x29, x30, [sp], #16
    ret
.Ls_if:
    bl      handle_if
    ldp     x29, x30, [sp], #16
    ret
.Ls_while:
    bl      handle_while
    ldp     x29, x30, [sp], #16
    ret
.Ls_each:
    bl      handle_each
    ldp     x29, x30, [sp], #16
    ret
.Ls_return:
    bl      handle_return
    ldp     x29, x30, [sp], #16
    ret
.Ls_var:
    bl      handle_var
    ldp     x29, x30, [sp], #16
    ret
.Ls_buf:
    bl      handle_buf
    ldp     x29, x30, [sp], #16
    ret
.Ls_label:
    bl      handle_label
    ldp     x29, x30, [sp], #16
    ret
.Ls_ident:
    bl      handle_ident_stmt
    ldp     x29, x30, [sp], #16
    ret
.Ls_trap:
    add     x19, x19, #TOK_STRIDE_SZ                    // skip 'trap'
    // Two forms:
    //   `trap`                   — bare SVC, caller has set X8 / X0..X7.
    //   `trap NAME NUM args...`  — expand to:
    //                                MOV X8, <num>
    //                                MOV X0..X7, <args>
    //                                SVC #0
    //                                MOV <name_reg>, X0
    //                              and add NAME as a local symbol.
    cmp     x19, x27
    b.hs    .Ls_trap_bare
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    .Ls_trap_bare
    // Forth-style trap:  NAME NUM args...
    str     x19, [sp, #-16]!                             // [sp] = NAME tok
    add     x19, x19, #TOK_STRIDE_SZ                    // past NAME
    bl      parse_expr                                   // w0 = num_reg
    mov     w1, w0
    mov     w0, #8                                       // dest = X8
    bl      emit_mov_reg
    mov     w4, #0                                       // arg idx
.Ls_trap_arg_loop:
    cmp     x19, x27
    b.hs    .Ls_trap_svc
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Ls_trap_svc
    cmp     w0, #TOK_EOF
    b.eq    .Ls_trap_svc
    cmp     w0, #TOK_INDENT
    b.eq    .Ls_trap_svc
    cmp     w4, #8
    b.ge    .Ls_trap_svc
    str     x4, [sp, #-16]!                              // save arg idx
    bl      parse_expr
    ldr     x4, [sp], #16
    mov     w1, w0                                       // arg reg
    mov     w0, w4                                       // dest = X<arg idx>
    str     x4, [sp, #-16]!
    bl      emit_mov_reg
    ldr     x4, [sp], #16
    add     w4, w4, #1
    b       .Ls_trap_arg_loop
.Ls_trap_svc:
    bl      emit_svc                                     // SVC #0
    // Bind NAME to X0 as a new local-register symbol.
    bl      alloc_reg
    mov     w5, w0                                       // name_reg
    mov     w1, #0                                       // src = X0
    mov     w0, w5                                       // dest = name_reg
    bl      emit_mov_reg
    ldr     x19, [sp], #16                               // restore NAME tok
    mov     w1, #KIND_VAR
    mov     w2, w5
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ                    // past NAME
    // Skip back past remaining args & num — x19 currently at NAME;
    // but the arg-loop already advanced past them before we restored.
    // Re-scan the line to EOL so the parser is positioned at NEWLINE.
    // Actually the arg-loop left x19 PAST the last arg already; we
    // just need to leave x19 at the original post-trap position so
    // the NAME advance above is correct.  Arg loop already consumed
    // through NEWLINE-peek; saved NAME pos is before everything, so
    // after restoring x19 = NAME_pos and adding one stride, x19 sits
    // at NUM.  That's wrong — the caller expects x19 at NEWLINE.
    // Quick recovery: spin forward until NEWLINE/EOF/INDENT.
.Ls_trap_eol:
    cmp     x19, x27
    b.hs    .Ls_trap_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Ls_trap_done
    cmp     w0, #TOK_EOF
    b.eq    .Ls_trap_done
    cmp     w0, #TOK_INDENT
    b.eq    .Ls_trap_done
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Ls_trap_eol
.Ls_trap_done:
    ldp     x29, x30, [sp], #16
    ret
.Ls_trap_bare:
    bl      emit_svc
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_ident_stmt — IDENT at start of statement
//   Disambiguate: name = expr | name expr (binding) | name (bare call)
// ============================================================
handle_ident_stmt:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Check for "trap" keyword
    ldr     w0, [x19, #8]
    cmp     w0, #4
    b.ne    .Lhi_not_trap
    ldr     w1, [x19, #4]
    add     x1, x28, x1
    ldrb    w2, [x1]
    cmp     w2, #'t'
    b.ne    .Lhi_not_trap
    ldrb    w2, [x1, #1]
    cmp     w2, #'r'
    b.ne    .Lhi_not_trap
    ldrb    w2, [x1, #2]
    cmp     w2, #'a'
    b.ne    .Lhi_not_trap
    ldrb    w2, [x1, #3]
    cmp     w2, #'p'
    b.ne    .Lhi_not_trap
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc
    ldp     x29, x30, [sp], #16
    ret

.Lhi_not_trap:
    // Check for "goto" keyword
    ldr     w0, [x19, #8]
    cmp     w0, #4
    b.ne    .Lhi_not_goto
    ldr     w1, [x19, #4]
    add     x1, x28, x1
    ldrb    w2, [x1]
    cmp     w2, #'g'
    b.ne    .Lhi_not_goto
    ldrb    w2, [x1, #1]
    cmp     w2, #'o'
    b.ne    .Lhi_not_goto
    ldrb    w2, [x1, #2]
    cmp     w2, #'t'
    b.ne    .Lhi_not_goto
    ldrb    w2, [x1, #3]
    cmp     w2, #'o'
    b.ne    .Lhi_not_goto
    add     x19, x19, #TOK_STRIDE_SZ
    // Next token is label name — look up
    bl      sym_lookup
    cbz     x0, .Lhi_goto_fwd
    ldr     w0, [x0, #SYM_REG]
    // x0 = target address as u32 — emit B
    bl      emit_b
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret
.Lhi_goto_fwd:
    // Forward goto — record in patch table, emit B placeholder (0x14000000)
    // x19 points at the label name token.
    // Record: name bytes + source offset in code_buf.
    ldr     w2, [x19, #4]           // source offset of name token
    ldr     w3, [x19, #8]           // name length
    // Save tokp for later
    str     x19, [sp, #-16]!
    // Get current emit position (code_buf offset)
    bl      emit_cur                 // x0 = emit_ptr
    adrp    x4, ls_code_buf
    add     x4, x4, :lo12:ls_code_buf
    sub     x5, x0, x4              // x5 = offset in code_buf
    // Append to patch table
    adrp    x6, ls_n_fwd_gotos
    add     x6, x6, :lo12:ls_n_fwd_gotos
    ldr     w7, [x6]
    adrp    x8, ls_fwd_gotos
    add     x8, x8, :lo12:ls_fwd_gotos
    mov     x9, #40
    mul     x9, x7, x9
    add     x8, x8, x9              // x8 = &fwd_gotos[w7]
    // Copy label name (up to 32 bytes)
    ldr     x19, [sp]                // restore tokp
    ldr     w2, [x19, #4]           // source offset
    ldr     w3, [x19, #8]           // length
    cmp     w3, #32
    b.lt    1f
    mov     w3, #32
1:  add     x9, x28, x2             // src + offset
    mov     w10, #0
.Lfg_copy:
    cmp     w10, w3
    b.ge    .Lfg_copy_done
    ldrb    w11, [x9, x10]
    strb    w11, [x8, x10]
    add     w10, w10, #1
    b       .Lfg_copy
.Lfg_copy_done:
    // Pad remaining bytes with 0
    cmp     w10, #32
    b.ge    .Lfg_pad_done
.Lfg_pad:
    strb    wzr, [x8, x10]
    add     w10, w10, #1
    cmp     w10, #32
    b.lt    .Lfg_pad
.Lfg_pad_done:
    // Store source offset at [x8 + 32]
    str     w5, [x8, #32]
    // Also store length at [x8 + 36]
    str     w3, [x8, #36]
    // Increment fwd_gotos count
    add     w7, w7, #1
    str     w7, [x6]
    ldr     x19, [sp], #16
    // Emit B with placeholder offset 0 (0x14000000)
    mov     w0, #0x14000000
    movk    w0, #0x1400, lsl #16
    bl      emit32
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

.Lhi_not_goto:
    // Check for "continue" keyword (full string compare)
    ldr     w0, [x19, #8]
    cmp     w0, #8
    b.ne    .Lhi_not_continue
    ldr     w1, [x19, #4]
    add     x1, x28, x1
    ldrb    w2, [x1]
    cmp     w2, #'c'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #1]
    cmp     w2, #'o'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #2]
    cmp     w2, #'n'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #3]
    cmp     w2, #'t'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #4]
    cmp     w2, #'i'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #5]
    cmp     w2, #'n'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #6]
    cmp     w2, #'u'
    b.ne    .Lhi_not_continue
    ldrb    w2, [x1, #7]
    cmp     w2, #'e'
    b.ne    .Lhi_not_continue
    add     x19, x19, #TOK_STRIDE_SZ
    // Emit B to loop_top (from loop stack)
    // For simplicity: emit NOP (continue support is a stub)
    bl      emit_nop
    ldp     x29, x30, [sp], #16
    ret

.Lhi_not_continue:
    // First check: is this name a KNOWN composition? If so, it's a call.
    bl      sym_lookup
    cbz     x0, .Lhi_not_known
    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_COMP
    b.eq    .Lhi_comp_call

.Lhi_not_known:
    // Check if it's a known VARIABLE — then "name expr" is reassignment
    bl      sym_lookup
    cbz     x0, .Lhi_truly_unknown
    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_VAR
    b.eq    .Lhi_reassign
    cmp     w1, #KIND_LOCAL_REG
    b.eq    .Lhi_reassign
    cmp     w1, #KIND_PARAM
    b.eq    .Lhi_reassign

.Lhi_truly_unknown:
    // Peek at next token
    mov     x4, x19
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lhi_bare_call

    ldr     w0, [x4]
    cmp     w0, #TOK_EQ
    b.eq    .Lhi_assign
    cmp     w0, #TOK_COLON
    b.eq    .Lhi_label

    // If next is NEWLINE/EOF/INDENT → bare call (no args)
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lhi_bare_call
    cmp     w0, #TOK_EOF
    b.eq    .Lhi_bare_call
    cmp     w0, #TOK_INDENT
    b.eq    .Lhi_bare_call

    // Unknown name followed by tokens → binding (name expr)
    bl      handle_binding
    ldp     x29, x30, [sp], #16
    ret

.Lhi_reassign:
    // Known variable reassignment. Two patterns:
    //   "cmp_type 2"     → name followed by value → skip name, parse "2"
    //   "count + 1"      → name followed by operator → DON'T skip, parse "count + 1"
    ldr     w5, [x0, #SYM_REG]     // existing register
    // Peek at token AFTER the name to decide
    add     x4, x19, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lhi_reassign_skip
    ldr     w0, [x4]
    // Operators: parse full expression including name
    cmp     w0, #TOK_PLUS
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_MINUS
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_STAR
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_SLASH
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_AMP
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_PIPE
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_CARET
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_SHL
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_SHR
    b.eq    .Lhi_reassign_full
.Lhi_reassign_skip:
    // Check for multi-line: NEWLINE + INDENT + operator means keep name
    mov     w3, w0                     // save original next-token for later
    cmp     w0, #TOK_NEWLINE
    b.ne    .Lhi_reassign_do_skip
    // Peek further: NEWLINE + INDENT + operator?
    add     x4, x4, #TOK_STRIDE_SZ    // past NEWLINE
    cmp     x4, x27
    b.hs    .Lhi_reassign_do_skip
    ldr     w0, [x4]
    cmp     w0, #TOK_INDENT
    b.ne    .Lhi_reassign_do_skip
    add     x4, x4, #TOK_STRIDE_SZ    // past INDENT
    cmp     x4, x27
    b.hs    .Lhi_reassign_do_skip
    ldr     w0, [x4]
    cmp     w0, #TOK_PIPE
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_AMP
    b.eq    .Lhi_reassign_full
    cmp     w0, #TOK_PLUS
    b.eq    .Lhi_reassign_full
    // NOTE: TOK_MINUS not included — "-1" on next line is a separate statement
.Lhi_reassign_do_skip:
    // Restore original next-token (multi-line peek may have clobbered w0)
    mov     w0, w3
    // Check if there's actually a value expression to parse.
    // If next is NEWLINE/EOF, it's a bare read (return value), not reassignment.
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lhi_reassign_bare
    cmp     w0, #TOK_EOF
    b.eq    .Lhi_reassign_bare
    // Value follows name: skip name, parse value
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lhi_reassign_full
.Lhi_reassign_bare:
    // Bare variable read — just emit MOV X0, Xvar (return value convention)
    mov     w1, w5                  // source = variable's register
    mov     w0, #0                  // dest = X0
    bl      emit_mov_reg
    add     x19, x19, #TOK_STRIDE_SZ   // skip the variable name
    ldp     x29, x30, [sp], #16
    ret
.Lhi_reassign_full:
    // Operator follows name: parse full expression (includes name)
    stp     w5, wzr, [sp, #-16]!
    bl      parse_expr
    ldp     w5, wzr, [sp], #16
    cmp     w0, w5
    b.eq    .Lhi_reassign_done      // same register, no MOV needed
    mov     w1, w0
    mov     w0, w5
    bl      emit_mov_reg
.Lhi_reassign_done:
    ldp     x29, x30, [sp], #16
    ret

.Lhi_comp_call:
    bl      handle_call
    ldp     x29, x30, [sp], #16
    ret

.Lhi_bare_call:
    bl      handle_call
    ldp     x29, x30, [sp], #16
    ret

.Lhi_label:
    // Label definition: "name:" — record current code address as symbol
    bl      emit_cur
    mov     w2, w0                  // code address
    mov     w1, #KIND_COMP          // labels use KIND_COMP (goto targets)
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ   // skip name
    add     x19, x19, #TOK_STRIDE_SZ   // skip ':'
    ldp     x29, x30, [sp], #16
    ret

.Lhi_assign:
    bl      handle_assign
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_binding — name expr
//   Evaluate expression, register name → result register
// ============================================================
.globl parse_binding_compose
handle_binding:
parse_binding_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    str     x19, [sp, #-16]!       // save name token pointer
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    bl      parse_expr              // result reg in w0
    mov     w4, w0

    // Save post-expr position, restore x19 to name for sym_add
    ldr     x5, [sp]
    str     x19, [sp]
    mov     x19, x5

    mov     w1, #KIND_LOCAL_REG
    mov     w2, w4
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add

    // Raise reg_floor to protect this binding's register from being recycled
    // by future reset_regs calls. Only raise if w4 is in the allocated range
    // (not a param register X0-X8).
    cmp     w4, #REG_FIRST
    b.lt    .Lbind_no_floor
    adrp    x5, reg_floor
    add     x5, x5, :lo12:reg_floor
    ldr     w6, [x5]
    add     w7, w4, #1              // new floor = binding_reg + 1
    cmp     w7, w6
    b.le    .Lbind_no_floor         // don't lower the floor
    str     w7, [x5]
    // Also update next_reg so future allocs start above this
    adrp    x5, next_reg
    add     x5, x5, :lo12:next_reg
    str     w7, [x5]
.Lbind_no_floor:

    ldr     x19, [sp], #16         // restore post-expr position
    mov     w0, #1                 // return 1 = binding (don't reset regs)
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_assign — name = expr
// ============================================================
handle_assign:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    bl      sym_lookup
    stp     x0, x19, [sp, #-16]!   // save sym entry + name tokp
    add     x19, x19, #TOK_STRIDE_SZ   // skip name
    add     x19, x19, #TOK_STRIDE_SZ   // skip '='

    bl      parse_expr
    mov     w8, w0

    ldp     x5, x6, [sp], #16

    cbnz    x5, .Lassign_existing

    // New symbol
    mov     x7, x19
    mov     x19, x6
    mov     w1, #KIND_LOCAL_REG
    mov     w2, w8
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    mov     x19, x7
    ldp     x29, x30, [sp], #16
    ret

.Lassign_existing:
    ldr     w1, [x5, #SYM_REG]
    cmp     w1, w8
    b.eq    .Lassign_done2
    mov     w0, w1
    mov     w1, w8
    bl      emit_mov_reg
.Lassign_done2:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// emit_save_bindings — push caller's live binding registers onto
// the target program's stack.  Iterates over [REG_FIRST, reg_floor)
// in pairs, emitting STP Xn,Xm,[SP,#-16]! for each pair, and a lone
// STR for a trailing odd register.  Preserves reg_floor/next_reg.
emit_save_bindings:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    ldr     w0, [x0]
    mov     w1, #REG_FIRST
    cmp     w1, w0
    b.ge    .Lesb_done
.Lesb_loop:
    add     w2, w1, #1
    cmp     w2, w0
    b.ge    .Lesb_tail
    // STP Xw1, Xw2, [SP, #-16]!  encoding:
    // 0xA9BF0000 | (Xw2 << 10) | (SP=31 << 5) | Xw1
    movz    w3, #0x0000
    movk    w3, #0xA9BF, lsl #16
    lsl     w4, w2, #10
    orr     w3, w3, w4
    mov     w4, #(31 << 5)
    orr     w3, w3, w4
    orr     w3, w3, w1
    stp     w0, w1, [sp, #-16]!
    mov     w0, w3
    bl      emit32
    ldp     w0, w1, [sp], #16
    add     w1, w1, #2
    cmp     w1, w0
    b.lt    .Lesb_loop
    b       .Lesb_done
.Lesb_tail:
    // Lone register left.  STR Xw1, [SP, #-16]! = 0xF81F0FE0 | w1
    movz    w3, #0x0FE0
    movk    w3, #0xF81F, lsl #16
    orr     w3, w3, w1
    mov     w0, w3
    bl      emit32
.Lesb_done:
    ldp     x29, x30, [sp], #16
    ret

// emit_restore_bindings — counterpart to emit_save_bindings.
// Pops the saved registers in reverse order.  For paired push,
// emits LDP Xn,Xm,[SP],#16.  For the trailing STR, emits LDR X,[SP],#16.
emit_restore_bindings:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    ldr     w0, [x0]                   // w0 = reg_floor
    mov     w1, #REG_FIRST
    cmp     w1, w0
    b.ge    .Lerb_done
    // Determine highest pair start.  If (reg_floor - REG_FIRST) is odd,
    // a lone register was pushed last — pop it first.
    sub     w2, w0, w1                 // count
    and     w3, w2, #1
    cbz     w3, .Lerb_pairs
    // Pop lone register = REG_FIRST + (count - 1)
    sub     w4, w0, #1
    // LDR Xw4, [SP], #16 = 0xF84107E0 | w4  (post-index, bits 11:10=01)
    movz    w3, #0x07E0
    movk    w3, #0xF841, lsl #16
    orr     w3, w3, w4
    stp     w0, w1, [sp, #-16]!
    mov     w0, w3
    bl      emit32
    ldp     w0, w1, [sp], #16
    sub     w0, w0, #1                 // one fewer to pop
.Lerb_pairs:
    cmp     w1, w0
    b.ge    .Lerb_done
    // Pop pair from top: Xtop-2, Xtop-1
    sub     w4, w0, #2
    sub     w5, w0, #1
    // LDP Xw4, Xw5, [SP], #16 = 0xA8C10000 | (Xw5 << 10) | (SP << 5) | Xw4
    movz    w3, #0x0000
    movk    w3, #0xA8C1, lsl #16
    lsl     w6, w5, #10
    orr     w3, w3, w6
    mov     w6, #(31 << 5)
    orr     w3, w3, w6
    orr     w3, w3, w4
    stp     w0, w1, [sp, #-16]!
    mov     w0, w3
    bl      emit32
    ldp     w0, w1, [sp], #16
    sub     w0, w0, #2
    b       .Lerb_pairs
.Lerb_done:
    ldp     x29, x30, [sp], #16
    ret

// handle_call — name [arg1 arg2 ...]
//   Look up composition, emit args into X0-X7, emit BL.
// ============================================================
handle_call:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    bl      sym_lookup
    mov     x5, x0
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    // Parse args
    mov     w8, #0
.Lcall_args:
    cmp     x19, x27
    b.hs    .Lcall_emit
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lcall_emit
    cmp     w0, #TOK_EOF
    b.eq    .Lcall_emit
    cmp     w0, #TOK_INDENT
    b.eq    .Lcall_emit

    stp     x8, x5, [sp, #-16]!
    bl      parse_expr
    ldp     x8, x5, [sp], #16

    cmp     w0, w8
    b.eq    .Lcall_next_arg
    // MOV Xarg, Xresult
    mov     w1, w0
    mov     w0, w8
    bl      emit_mov_reg
.Lcall_next_arg:
    add     w8, w8, #1
    b       .Lcall_args

.Lcall_emit:
    cbz     x5, .Lcall_unknown
    // Save caller-live binding registers [REG_FIRST, reg_floor) onto
    // the target program's stack before BL, then restore after.
    // Bindings live below reg_floor and would otherwise be clobbered
    // by the callee, which starts its allocator fresh at REG_FIRST.
    stp     x5, xzr, [sp, #-16]!
    bl      emit_save_bindings
    ldp     x5, xzr, [sp], #16
    ldr     w0, [x5, #SYM_REG]
    bl      emit_cur
    mov     x1, x0
    ldr     w0, [x5, #SYM_REG]
    sub     x2, x0, x1
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      emit32
    bl      emit_restore_bindings
    ldp     x29, x30, [sp], #16
    ret

.Lcall_unknown:
    bl      emit_nop
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_store — ← width addr val
// ============================================================
.globl parse_mem_store
handle_store:
parse_mem_store:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip '←'

    bl      parse_expr              // width
    mov     w4, w0
    stp     x4, xzr, [sp, #-16]!
    bl      parse_expr              // addr (base, possibly with +offset inline)
    mov     w5, w0
    ldp     x4, xzr, [sp], #16
    stp     x4, x5, [sp, #-16]!
    bl      parse_expr              // value (or offset if there's a 4th operand)
    mov     w6, w0
    ldp     x4, x5, [sp], #16
    // Check for 4th operand: if present, w6 is offset and next expr is value
    cmp     x19, x27
    b.hs    .Lstore_emit
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lstore_emit
    cmp     w0, #TOK_EOF
    b.eq    .Lstore_emit
    cmp     w0, #TOK_INDENT
    b.eq    .Lstore_emit
    // 4th operand exists: addr = base + offset, value = this operand
    // Emit ADD Xaddr, Xbase, Xoffset
    stp     w4, w5, [sp, #-16]!
    bl      alloc_reg
    mov     w7, w0              // combined addr register
    mov     w0, w7
    mov     w1, w5              // old base
    mov     w2, w6              // offset
    bl      emit_add_reg
    ldp     w4, w5, [sp], #16
    mov     w5, w7              // addr = combined
    bl      parse_expr          // real value
    mov     w6, w0

.Lstore_emit:
    // STR Xval, [Xaddr, #0]
    mov     w0, w6
    mov     w1, w5
    mov     w2, #0
    bl      emit_str_imm

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_load — → width addr
// ============================================================
.globl parse_mem_load
handle_load:
parse_mem_load:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip '→'

    bl      parse_expr              // width (often a literal — temp)
    mov     w4, w0
    bl      parse_expr              // addr
    mov     w5, w0

    bl      alloc_reg
    mov     w6, w0

    // LDR Xresult, [Xaddr, #0]
    mov     w0, w6
    mov     w1, w5
    mov     w2, #0
    bl      emit_ldr_imm

    // The width and addr temps (w4, w5) are dead now that the LDR is
    // in the code stream.  Compact the result down to w4's slot and
    // free everything above, so each `name → W addr` only consumes a
    // single register slot from the caller's pool.  Without this every
    // load wasted 2 regs and `walk_top_level` exhausted the allocator
    // before its body ran.
    adrp    x0, reg_floor
    add     x0, x0, :lo12:reg_floor
    ldr     w7, [x0]
    cmp     w4, w7
    b.lt    1f                      // w4 is a binding, leave alone
    cmp     w4, w6
    b.eq    1f                      // already compact
    // emit MOV Xw4, Xw6
    mov     w0, w4
    mov     w1, w6
    bl      emit_mov_reg
    // free everything above w4 (releases w5 and w6's slot)
    add     w0, w4, #1
    bl      free_reg
    mov     w6, w4                  // result is now in w4
1:  ldp     x29, x30, [sp], #16
    mov     w0, w6
    ret

// ============================================================
// handle_reg_write — ↓ $N val
// ============================================================
.globl parse_reg_write
handle_reg_write:
parse_reg_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip arrow

    bl      parse_dollar_reg
    str     w0, [sp, #-16]!

    // Fast path: integer literal
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    .Lrw_expr

    bl      parse_int_literal
    mov     x1, x0
    ldr     w0, [sp]
    bl      emit_mov_imm64
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lrw_done

.Lrw_expr:
    bl      parse_expr
    mov     w5, w0
    ldr     w4, [sp]
    mov     w0, w4
    mov     w1, w5
    bl      emit_mov_reg

.Lrw_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_reg_read — ↑ $N
// ============================================================
.globl parse_reg_read
handle_reg_read:
parse_reg_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip arrow

    bl      parse_dollar_reg
    mov     w4, w0

    bl      alloc_reg
    mov     w5, w0

    // MOV Xdest, XN
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg

    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_label — label name
// ============================================================
handle_label:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'label'

    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error

    bl      emit_cur
    str     x0, [sp, #-16]!        // save label address
    mov     w2, w0
    mov     w1, #KIND_LOCAL_REG
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add

    // Patch any forward gotos that target this label.
    // Scan ls_fwd_gotos; for each entry whose name matches, emit B fix at the source offset.
    ldr     x22, [sp]               // label address (emit_ptr absolute)
    adrp    x8, ls_n_fwd_gotos
    add     x8, x8, :lo12:ls_n_fwd_gotos
    ldr     w9, [x8]                // count
    cbz     w9, .Lhl_no_patches
    adrp    x10, ls_fwd_gotos
    add     x10, x10, :lo12:ls_fwd_gotos
    mov     w11, #0                 // index
    // Get label name info
    ldr     w12, [x19, #4]          // name source offset
    ldr     w13, [x19, #8]          // name length
    cmp     w13, #32
    b.lt    .Lhl_len_ok
    mov     w13, #32
.Lhl_len_ok:
    add     x14, x28, x12           // label name ptr (src + offset)
.Lhl_scan:
    cmp     w11, w9
    b.ge    .Lhl_no_patches
    mov     x15, #40
    mul     x15, x11, x15
    add     x15, x10, x15           // &fwd_gotos[i]
    // Check if name length matches
    ldr     w16, [x15, #36]         // stored length
    cmp     w16, w13
    b.ne    .Lhl_next
    // Compare name bytes
    mov     w17, #0
.Lhl_cmp:
    cmp     w17, w13
    b.ge    .Lhl_match
    ldrb    w20, [x14, x17]
    ldrb    w21, [x15, x17]
    cmp     w20, w21
    b.ne    .Lhl_next
    add     w17, w17, #1
    b       .Lhl_cmp
.Lhl_match:
    // Found matching forward goto — patch it
    // Source offset in code_buf is at [x15 + 32]
    ldr     w17, [x15, #32]        // source offset in code_buf
    adrp    x20, ls_code_buf
    add     x20, x20, :lo12:ls_code_buf
    add     x20, x20, x17           // absolute address of the B instruction
    // Compute offset: (label_addr - source_addr) / 4
    sub     x21, x22, x20
    asr     x21, x21, #2
    and     w21, w21, #0x3FFFFFF
    // Build B instruction: 0x14000000 | imm26
    mov     w2, #0x14000000
    movk    w2, #0x1400, lsl #16
    orr     w2, w2, w21
    str     w2, [x20]
    // Invalidate this entry by setting length to 0 so it won't rematch
    str     wzr, [x15, #36]
.Lhl_next:
    add     w11, w11, #1
    b       .Lhl_scan
.Lhl_no_patches:
    add     sp, sp, #16            // discard saved label address
    add     x19, x19, #TOK_STRIDE_SZ

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_return — return [expr]
// ============================================================
.globl parse_return_compose
handle_return:
parse_return_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'return'

    // Check if there's an expression
    cmp     x19, x27
    b.hs    .Lret_void
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lret_void
    cmp     w0, #TOK_EOF
    b.eq    .Lret_void
    cmp     w0, #TOK_INDENT
    b.eq    .Lret_void

    bl      parse_expr
    cmp     w0, #0
    b.eq    .Lret_emit
    // MOV X0, Xresult
    mov     w1, w0
    mov     w0, #0
    bl      emit_mov_reg

.Lret_emit:
    MOVI32  w0, 0xA8C17BFD         // LDP X29, X30, [SP], #16
    bl      emit32
    bl      emit_ret_inst
    ldp     x29, x30, [sp], #16
    ret

.Lret_void:
    MOVI32  w0, 0xA8C17BFD
    bl      emit32
    bl      emit_ret_inst
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_if — if expr : body [elif/else]
// ============================================================
handle_if:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'if'

    // Check for compound conditional: if>= if< if== if!= if> if<=
    ldr     w0, [x19]
    cmp     w0, #TOK_GTE
    b.eq    .Lif_compound
    cmp     w0, #TOK_LTE
    b.eq    .Lif_compound
    cmp     w0, #TOK_LT
    b.eq    .Lif_compound
    cmp     w0, #TOK_GT
    b.eq    .Lif_compound
    cmp     w0, #TOK_EQEQ
    b.eq    .Lif_compound
    cmp     w0, #TOK_NEQ
    b.eq    .Lif_compound

    // Simple if: if expr → CBZ
    bl      parse_expr
    mov     w4, w0
    bl      emit_cur
    str     x0, [sp, #0]           // patch address
    str     wzr, [sp, #16]         // patch type 0 = CBZ
    mov     w0, w4
    mov     x1, #0
    bl      emit_cbz
    b       .Lif_body

.Lif_compound:
    // Compound conditional: if>= a b → CMP a,b + B.!cond
    add     x19, x19, #TOK_STRIDE_SZ   // skip comparison operator
    // Save comparison token on stack (parse_expr clobbers caller-saved regs)
    stp     x0, xzr, [sp, #-16]!   // push comparison token
    bl      parse_expr              // left operand
    str     w0, [sp, #8]           // save left result in second slot
    bl      parse_expr              // right operand
    mov     w6, w0                  // right operand register
    ldr     w4, [sp, #8]           // restore left operand
    ldp     x5, xzr, [sp], #16    // restore comparison token into w5
    // Emit CMP
    mov     w0, w4
    mov     w1, w6
    bl      emit_cmp_reg
    // Emit B.!cond placeholder (inverted condition)
    bl      emit_cur
    str     x0, [sp, #0]           // patch address
    // Map comparison to INVERTED condition for skip
    cmp     w5, #TOK_GTE
    b.eq    .Lif_cc_lt
    cmp     w5, #TOK_LT
    b.eq    .Lif_cc_ge
    cmp     w5, #TOK_GT
    b.eq    .Lif_cc_le
    cmp     w5, #TOK_LTE
    b.eq    .Lif_cc_gt
    cmp     w5, #TOK_EQEQ
    b.eq    .Lif_cc_ne
    // TOK_NEQ → skip if EQ
    mov     w1, #CC_EQ
    b       .Lif_cc_emit
.Lif_cc_lt:
    mov     w1, #CC_LT
    b       .Lif_cc_emit
.Lif_cc_ge:
    mov     w1, #CC_GE
    b       .Lif_cc_emit
.Lif_cc_le:
    mov     w1, #CC_LE
    b       .Lif_cc_emit
.Lif_cc_gt:
    mov     w1, #CC_GT
    b       .Lif_cc_emit
.Lif_cc_ne:
    mov     w1, #CC_NE
    b       .Lif_cc_emit
.Lif_cc_emit:
    // Emit B.cond placeholder (condition in w1)
    mov     w0, #0                  // placeholder offset
    bl      emit_b_cond
    mov     w0, #1
    str     w0, [sp, #16]          // patch type 1 = B.cond
    b       .Lif_body

.Lif_body:

    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    1f
    ldr     w9, [x19, #8]
    cbz     w9, 1f              // indent 0 = column 0 = no body
1:  bl      parse_body

    // Emit B past else (placeholder)
    bl      emit_cur
    str     x0, [sp, #8]           // end-patch
    mov     x0, #0
    bl      emit_b

    // Patch branch to here (CBZ or B.cond depending on type)
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #0]
    ldr     w2, [sp, #16]          // patch type
    cbnz    w2, .Lif_patch_bcond
    bl      patch_cbz
    b       .Lif_patch_done
.Lif_patch_bcond:
    bl      patch_b_cond
.Lif_patch_done:

    // Check for elif/else — peek past NEWLINE + INDENT without consuming
    bl      skip_newlines
    cmp     x19, x27
    b.hs    .Lif_end
    ldr     w0, [x19]
    // If INDENT, peek past it for elif/else but don't consume yet
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_check_kw
    add     x4, x19, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lif_end
    ldr     w0, [x4]
    cmp     w0, #TOK_ELIF
    b.eq    .Lif_skip_indent
    cmp     w0, #TOK_ELSE
    b.eq    .Lif_skip_indent
    b       .Lif_end
.Lif_skip_indent:
    // Found elif/else after INDENT — consume the INDENT
    add     x19, x19, #TOK_STRIDE_SZ
    ldr     w0, [x19]
.Lif_check_kw:
    cmp     w0, #TOK_ELIF
    b.eq    .Lif_elif
    cmp     w0, #TOK_ELSE
    b.eq    .Lif_else
    b       .Lif_end

.Lif_elif:
    bl      handle_if               // recurse (elif = if)
    b       .Lif_end

.Lif_else:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    2f
    ldr     w9, [x19, #8]
    cbz     w9, 2f              // indent 0 = column 0 = no body
2:  bl      parse_body

.Lif_end:
    // Patch end-branch
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_while — while expr : body
// ============================================================
handle_while:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'while'

    bl      emit_cur
    mov     x5, x0                  // loop_top

    bl      parse_expr
    mov     w4, w0

    // Emit CBZ to exit (placeholder)
    bl      emit_cur
    mov     x6, x0
    mov     w0, w4
    mov     x1, #0
    bl      emit_cbz

    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    1f
    ldr     w9, [x19, #8]
    cbz     w9, 1f              // indent 0 = column 0 = no body
1:  stp     x5, x6, [sp, #-16]!
    bl      parse_body
    ldp     x5, x6, [sp], #16

    // B loop_top
    mov     x0, x5
    bl      emit_b

    // Patch CBZ to here
    bl      emit_cur
    mov     x1, x0
    mov     x0, x6
    bl      patch_cbz

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_for — for i start end [step]
// ============================================================
handle_for:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #48

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'for'

    // Parse loop var name
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error

    bl      alloc_reg
    mov     w10, w0
    str     w10, [sp, #0]           // loop var reg

    mov     w1, #KIND_LOCAL_REG
    mov     w2, w10
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse start
    bl      parse_expr
    str     w0, [sp, #4]

    // Parse end
    bl      parse_expr
    str     w0, [sp, #8]

    // Parse optional step
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lfor_default_step
    cmp     w0, #TOK_EOF
    b.eq    .Lfor_default_step
    cmp     w0, #TOK_INDENT
    b.eq    .Lfor_default_step

    bl      parse_expr
    str     w0, [sp, #12]
    b       .Lfor_init

.Lfor_default_step:
    bl      alloc_reg
    mov     w13, w0
    str     w13, [sp, #12]
    mov     w0, w13
    mov     x1, #1
    bl      emit_mov_imm64

.Lfor_init:
    // MOV Xi, Xstart
    ldr     w10, [sp, #0]
    ldr     w11, [sp, #4]
    mov     w0, w10
    mov     w1, w11
    bl      emit_mov_reg

    // loop_top:
    bl      emit_cur
    str     x0, [sp, #16]

    // CMP Xi, Xend
    ldr     w10, [sp, #0]
    ldr     w12, [sp, #8]
    mov     w0, w10
    mov     w1, w12
    bl      emit_cmp_reg

    // B.GE exit (placeholder)
    bl      emit_cur
    str     x0, [sp, #24]
    mov     w0, #CC_GE
    mov     x1, #0
    bl      emit_b_cond

    // Parse body
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    1f
    ldr     w9, [x19, #8]
    cbz     w9, 1f              // indent 0 = column 0 = no body
1:  bl      parse_body

    // ADD Xi, Xi, Xstep
    ldr     w10, [sp, #0]
    ldr     w13, [sp, #12]
    mov     w0, w10
    mov     w1, w10
    mov     w2, w13
    bl      emit_add_reg

    // B loop_top
    ldr     x0, [sp, #16]
    bl      emit_b

    // Patch B.GE
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #24]
    bl      patch_b_cond

    add     sp, sp, #48
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// handle_each — each i
//   On ARM64 host: emit MOV Xi, #0 (thread index placeholder)
// ============================================================
handle_each:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'each'

    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error

    bl      alloc_reg
    mov     w4, w0

    mov     w1, #KIND_LOCAL_REG
    mov     w2, w4
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ

    // MOV Xi, #0
    mov     w0, w4
    mov     x1, #0
    bl      emit_mov_imm64

    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    1f
    ldr     w9, [x19, #8]
    cbz     w9, 1f              // indent 0 = column 0 = no body
1:  bl      parse_body

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// EXPRESSION PARSER — with operator precedence
//
// Despite the "no precedence" aspiration, compiler.ls uses
// unparenthesized chains like: a + b * c
// So we keep the standard 5-level precedence:
//   0: comparison (== != < > <= >=)
//   1: bitwise (& | ^)
//   2: shift (<< >>)
//   3: additive (+ -)
//   4: multiplicative (* /)
//   5: atom (literal, ident, (expr), unary, mem ops)
//
// But each level is ~20 lines. No expression trees. Each op
// is one instruction emitted immediately.
// ============================================================

parse_expr:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      parse_bitwise

.Lexpr_cmp_loop:
    cmp     x19, x27
    b.hs    .Lexpr_cmp_done
    ldr     w1, [x19]
    cmp     w1, #TOK_EQEQ
    b.eq    .Lexpr_cmp_op
    cmp     w1, #TOK_NEQ
    b.eq    .Lexpr_cmp_op
    cmp     w1, #TOK_LT
    b.eq    .Lexpr_cmp_op
    cmp     w1, #TOK_GT
    b.eq    .Lexpr_cmp_op
    cmp     w1, #TOK_LTE
    b.eq    .Lexpr_cmp_op
    cmp     w1, #TOK_GTE
    b.eq    .Lexpr_cmp_op
    b       .Lexpr_cmp_done

.Lexpr_cmp_op:
    mov     w4, w0
    mov     w5, w1
    add     x19, x19, #TOK_STRIDE_SZ
    stp     w4, w5, [sp, #-16]!
    bl      parse_bitwise
    ldp     w4, w5, [sp], #16
    mov     w6, w0

    // CMP
    stp     w4, w5, [sp, #-16]!
    str     w6, [sp, #-16]!
    mov     w0, w4
    mov     w1, w6
    bl      emit_cmp_reg
    ldr     w6, [sp], #16
    ldp     w4, w5, [sp], #16

    // Reuse left only if it's a free temporary (>= reg_floor)
    adrp    x8, reg_floor
    add     x8, x8, :lo12:reg_floor
    ldr     w8, [x8]
    cmp     w4, w8
    b.lt    .Lcmp_alloc
    mov     w7, w4
    b       .Lcmp_reclaim
.Lcmp_alloc:
    bl      alloc_reg
    mov     w7, w0
.Lcmp_reclaim:
    cmp     w7, #REG_FIRST
    b.lt    .Lcmp_dispatch
    add     w0, w7, #1
    bl      free_reg
.Lcmp_dispatch:
    // Map token to condition code
    cmp     w5, #TOK_EQEQ
    b.eq    .Lcmp_eq
    cmp     w5, #TOK_NEQ
    b.eq    .Lcmp_ne
    cmp     w5, #TOK_LT
    b.eq    .Lcmp_lt
    cmp     w5, #TOK_GT
    b.eq    .Lcmp_gt
    cmp     w5, #TOK_LTE
    b.eq    .Lcmp_le
    mov     w1, #CC_GE
    b       .Lcmp_emit
.Lcmp_eq:
    mov     w1, #CC_EQ
    b       .Lcmp_emit
.Lcmp_ne:
    mov     w1, #CC_NE
    b       .Lcmp_emit
.Lcmp_lt:
    mov     w1, #CC_LT
    b       .Lcmp_emit
.Lcmp_gt:
    mov     w1, #CC_GT
    b       .Lcmp_emit
.Lcmp_le:
    mov     w1, #CC_LE
.Lcmp_emit:
    mov     w0, w7
    bl      emit_cset
    mov     w0, w7
    b       .Lexpr_cmp_loop

.Lexpr_cmp_done:
    ldp     x29, x30, [sp], #16
    ret

// parse_bitwise — & | ^
// Handles multi-line continuation: if NEWLINE+INDENT is followed by
// & | ^, skip the whitespace and continue the expression.
parse_bitwise:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      parse_shift
.Lbit_loop:
    cmp     x19, x27
    b.hs    .Lbit_done
    ldr     w1, [x19]
    cmp     w1, #TOK_AMP
    b.eq    .Lbit_op
    cmp     w1, #TOK_PIPE
    b.eq    .Lbit_op
    cmp     w1, #TOK_CARET
    b.eq    .Lbit_op
    // Check for multi-line continuation: NEWLINE + INDENT + operator
    cmp     w1, #TOK_NEWLINE
    b.ne    .Lbit_done
    mov     x4, x19
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lbit_done
    ldr     w2, [x4]
    cmp     w2, #TOK_INDENT
    b.ne    .Lbit_done
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lbit_done
    ldr     w2, [x4]
    cmp     w2, #TOK_PIPE
    b.eq    .Lbit_cont
    cmp     w2, #TOK_AMP
    b.eq    .Lbit_cont
    cmp     w2, #TOK_CARET
    b.eq    .Lbit_cont
    b       .Lbit_done
.Lbit_cont:
    // Skip NEWLINE + INDENT, continue at the operator
    add     x19, x19, #TOK_STRIDE_SZ
    add     x19, x19, #TOK_STRIDE_SZ
    ldr     w1, [x19]
    b       .Lbit_op
.Lbit_op:
    mov     w4, w0
    mov     w5, w1
    add     x19, x19, #TOK_STRIDE_SZ
    stp     w4, w5, [sp, #-16]!
    bl      parse_shift
    ldp     w4, w5, [sp], #16
    mov     w6, w0
    adrp    x8, reg_floor
    add     x8, x8, :lo12:reg_floor
    ldr     w8, [x8]
    cmp     w4, w8
    b.lt    .Lbit_alloc
    mov     w7, w4
    b       .Lbit_emit
.Lbit_alloc:
    bl      alloc_reg
    mov     w7, w0
.Lbit_emit:
    cmp     w5, #TOK_AMP
    b.eq    .Lbit_and
    cmp     w5, #TOK_PIPE
    b.eq    .Lbit_or
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_eor_reg
    b       .Lbit_reclaim
.Lbit_and:
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_and_reg
    b       .Lbit_reclaim
.Lbit_or:
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_orr_reg
.Lbit_reclaim:
    cmp     w7, #REG_FIRST
    b.lt    .Lbit_next
    add     w0, w7, #1
    bl      free_reg
.Lbit_next:
    mov     w0, w7
    b       .Lbit_loop
.Lbit_done:
    ldp     x29, x30, [sp], #16
    ret

// parse_shift — << >>
parse_shift:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      parse_additive
.Lshift_loop:
    cmp     x19, x27
    b.hs    .Lshift_done
    ldr     w1, [x19]
    cmp     w1, #TOK_SHL
    b.eq    .Lshift_op
    cmp     w1, #TOK_SHR
    b.eq    .Lshift_op
    b       .Lshift_done
.Lshift_op:
    mov     w4, w0
    mov     w5, w1
    add     x19, x19, #TOK_STRIDE_SZ
    stp     w4, w5, [sp, #-16]!
    bl      parse_additive
    ldp     w4, w5, [sp], #16
    mov     w6, w0
    adrp    x8, reg_floor
    add     x8, x8, :lo12:reg_floor
    ldr     w8, [x8]
    cmp     w4, w8
    b.lt    .Lshift_alloc
    mov     w7, w4
    b       .Lshift_emit
.Lshift_alloc:
    bl      alloc_reg
    mov     w7, w0
.Lshift_emit:
    cmp     w5, #TOK_SHL
    b.eq    .Lshift_left
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_lsr_reg
    b       .Lshift_reclaim
.Lshift_left:
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_lsl_reg
.Lshift_reclaim:
    cmp     w7, #REG_FIRST
    b.lt    .Lshift_next
    add     w0, w7, #1
    bl      free_reg
.Lshift_next:
    mov     w0, w7
    b       .Lshift_loop
.Lshift_done:
    ldp     x29, x30, [sp], #16
    ret

// parse_additive — + -
parse_additive:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      parse_multiplicative
.Ladd_loop:
    cmp     x19, x27
    b.hs    .Ladd_done
    ldr     w1, [x19]
    cmp     w1, #TOK_PLUS
    b.eq    .Ladd_op
    cmp     w1, #TOK_MINUS
    b.eq    .Ladd_op
    b       .Ladd_done
.Ladd_op:
    mov     w4, w0
    mov     w5, w1
    add     x19, x19, #TOK_STRIDE_SZ
    stp     w4, w5, [sp, #-16]!
    bl      parse_multiplicative
    ldp     w4, w5, [sp], #16
    mov     w6, w0
    // Same rule as parse_multiplicative: only reuse w4 if it is a free
    // temporary (>= reg_floor), not a live binding's register.
    adrp    x8, reg_floor
    add     x8, x8, :lo12:reg_floor
    ldr     w8, [x8]
    cmp     w4, w8
    b.lt    .Ladd_alloc
    mov     w7, w4
    b       .Ladd_emit
.Ladd_alloc:
    bl      alloc_reg
    mov     w7, w0
.Ladd_emit:
    cmp     w5, #TOK_PLUS
    b.eq    .Ladd_plus
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_sub_reg
    b       .Ladd_reclaim
.Ladd_plus:
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_add_reg
.Ladd_reclaim:
    cmp     w7, #REG_FIRST
    b.lt    .Ladd_next
    add     w0, w7, #1
    bl      free_reg
.Ladd_next:
    mov     w0, w7
    b       .Ladd_loop
.Ladd_done:
    ldp     x29, x30, [sp], #16
    ret

// parse_multiplicative — * /
parse_multiplicative:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      parse_atom
.Lmul_loop:
    cmp     x19, x27
    b.hs    .Lmul_done
    ldr     w1, [x19]
    cmp     w1, #TOK_STAR
    b.eq    .Lmul_op
    cmp     w1, #TOK_SLASH
    b.eq    .Lmul_op
    b       .Lmul_done
.Lmul_op:
    mov     w4, w0
    mov     w5, w1
    add     x19, x19, #TOK_STRIDE_SZ
    stp     w4, w5, [sp, #-16]!
    bl      parse_atom
    ldp     w4, w5, [sp], #16
    mov     w6, w0
    // Reuse w4 (left operand) as destination ONLY if it is a free
    // temporary — i.e. >= reg_floor.  Regs below reg_floor belong to
    // live bindings (handle_binding raised the floor to protect them),
    // and overwriting them here would clobber the source variable.
    adrp    x8, reg_floor
    add     x8, x8, :lo12:reg_floor
    ldr     w8, [x8]
    cmp     w4, w8
    b.lt    .Lmul_alloc
    mov     w7, w4
    b       .Lmul_emit
.Lmul_alloc:
    bl      alloc_reg
    mov     w7, w0
.Lmul_emit:
    cmp     w5, #TOK_STAR
    b.eq    .Lmul_star
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_sdiv_reg
    b       .Lmul_reclaim
.Lmul_star:
    mov     w0, w7
    mov     w1, w4
    mov     w2, w6
    bl      emit_mul_reg
.Lmul_reclaim:
    cmp     w7, #REG_FIRST
    b.lt    .Lmul_next
    add     w0, w7, #1
    bl      free_reg
.Lmul_next:
    mov     w0, w7
    b       .Lmul_loop
.Lmul_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_atom — leaf expressions
//   INT, IDENT, (expr), unary -, memory ops, register ops,
//   trap, unary math (pass-through)
// ============================================================
parse_atom:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_INT
    b.eq    .La_int
    cmp     w0, #TOK_FLOAT
    b.eq    .La_int             // treat float as int for bootstrap
    cmp     w0, #TOK_IDENT
    b.eq    .La_ident
    // Keywords that may be used as variable/param names in compiler.ls.
    // Treat all non-control keywords as identifiers in expression context.
    cmp     w0, #TOK_PTR
    b.eq    .La_ident
    cmp     w0, #TOK_F32
    b.eq    .La_ident
    cmp     w0, #TOK_U32
    b.eq    .La_ident
    cmp     w0, #TOK_F16
    b.eq    .La_ident
    cmp     w0, #TOK_S32
    b.eq    .La_ident
    cmp     w0, #TOK_VOID
    b.eq    .La_ident
    cmp     w0, #TOK_KERNEL
    b.eq    .La_ident
    cmp     w0, #TOK_PARAM
    b.eq    .La_ident
    cmp     w0, #TOK_WEIGHT
    b.eq    .La_ident
    cmp     w0, #TOK_SHARED
    b.eq    .La_ident
    cmp     w0, #TOK_HOST
    b.eq    .La_ident
    cmp     w0, #TOK_TEMPLATE
    b.eq    .La_ident
    cmp     w0, #TOK_BARRIER
    b.eq    .La_ident
    cmp     w0, #TOK_LAYER
    b.eq    .La_ident
    cmp     w0, #TOK_PROJECT
    b.eq    .La_ident
    cmp     w0, #TOK_BIND
    b.eq    .La_ident
    cmp     w0, #TOK_RUNTIME
    b.eq    .La_ident
    cmp     w0, #TOK_LPAREN
    b.eq    .La_paren
    cmp     w0, #TOK_MINUS
    b.eq    .La_neg
    cmp     w0, #TOK_LOAD
    b.eq    .La_load
    cmp     w0, #TOK_STORE
    b.eq    .La_store
    cmp     w0, #TOK_REG_READ
    b.eq    .La_reg_read
    cmp     w0, #TOK_REG_WRITE
    b.eq    .La_reg_write
    cmp     w0, #TOK_SQRT
    b.eq    .La_unary_math
    cmp     w0, #TOK_SIN
    b.eq    .La_unary_math
    cmp     w0, #TOK_COS
    b.eq    .La_unary_math
    cmp     w0, #TOK_SUM
    b.eq    .La_unary_math
    cmp     w0, #TOK_MAX
    b.eq    .La_unary_math
    cmp     w0, #TOK_MIN
    b.eq    .La_unary_math
    cmp     w0, #TOK_INDEX
    b.eq    .La_unary_math
    cmp     w0, #TOK_DOLLAR
    b.eq    .La_dollar

    // Keyword token used as variable name? Try sym_lookup.
    ldr     w1, [x19, #8]
    cbz     w1, .La_error
    bl      sym_lookup
    cbz     x0, .La_error
    ldr     w0, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

.La_int:
    bl      parse_int_literal
    mov     x4, x0
    bl      alloc_reg
    mov     w5, w0
    mov     w0, w5
    mov     x1, x4
    bl      emit_mov_imm64
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.La_ident:
    // Check for "trap"
    ldr     w1, [x19, #8]
    cmp     w1, #4
    b.ne    .La_ident_lookup
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'t'
    b.ne    .La_ident_lookup
    ldrb    w3, [x2, #1]
    cmp     w3, #'r'
    b.ne    .La_ident_lookup
    ldrb    w3, [x2, #2]
    cmp     w3, #'a'
    b.ne    .La_ident_lookup
    ldrb    w3, [x2, #3]
    cmp     w3, #'p'
    b.ne    .La_ident_lookup
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc
    mov     w0, #0              // result in X0
    ldp     x29, x30, [sp], #16
    ret

.La_ident_lookup:
    // Check for $N register reference (lexed as TOK_IDENT starting with '$')
    ldr     w2, [x19, #4]           // source offset
    ldrb    w3, [x28, x2]           // first character
    cmp     w3, #'$'
    b.ne    .La_ident_sym
    // $N — call parse_dollar_reg (expects x19 on current token)
    bl      parse_dollar_reg        // w0 = register number
    ldp     x29, x30, [sp], #16
    ret
.La_ident_sym:
    bl      sym_lookup
    cbz     x0, .La_ident_unknown
    // Check if it's a composition — needs call dispatch, not register return
    ldr     w1, [x0, #SYM_KIND]
    cmp     w1, #KIND_COMP
    b.eq    .La_ident_unknown       // dispatch as expression-level call
    // Check if it's a buf — emit ADD Xresult, X28 (BSS_BASE), #offset
    cmp     w1, #KIND_BUF
    b.ne    .La_ident_reg
    // Buf: compute address as BSS_BASE + offset
    ldr     w2, [x0, #SYM_REG]      // w2 = BSS offset
    add     x19, x19, #TOK_STRIDE_SZ
    // Save offset on stack; alloc_reg clobbers caller-saved regs
    stp     x2, xzr, [sp, #-16]!
    bl      alloc_reg               // w0 = fresh register for result
    mov     w4, w0                  // result register
    ldp     x2, xzr, [sp], #16
    // Emit: ADD Xresult, X28, #(offset & 0xFFF)
    // ARM64 ADD imm encoding: 0x91000000 | (imm12 << 10) | (Rn << 5) | Rd
    // imm12 lives at bits 21:10, so bits 6..11 of imm12 land in the
    // upper-16 region that `movk #0x9100, lsl #16` would otherwise
    // overwrite.  Fold those bits into the movk constant.
    and     w5, w2, #0xFFF          // w5 = imm12 (low 12 bits of offset)
    and     w7, w5, #0x3F           // w7 = imm12[5:0]   → encoded bits 15:10
    lsr     w8, w5, #6              // w8 = imm12[11:6]  → encoded bits 21:16
    lsl     w7, w7, #10
    mov     w6, #28
    lsl     w6, w6, #5
    orr     w0, w4, w7              // Rd | (low_imm << 10)
    orr     w0, w0, w6              // | (Rn << 5)
    movz    w9, #0x9100
    orr     w9, w9, w8              // 0x9100 | high_imm[5:0]
    orr     w0, w0, w9, lsl #16
    stp     x2, x4, [sp, #-16]!    // save offset and result reg
    bl      emit32
    ldp     x2, x4, [sp], #16
    // If offset >= 4096, emit another ADD with LSL #12
    lsr     w5, w2, #12
    cbz     w5, .La_buf_done
    and     w5, w5, #0xFFF
    and     w7, w5, #0x3F
    lsr     w8, w5, #6
    lsl     w7, w7, #10
    lsl     w6, w4, #5              // Rn=Xresult
    orr     w0, w4, w7
    orr     w0, w0, w6
    movz    w9, #0x9140             // ADD (imm, LSL #12): sh=1 (bit 22)
    orr     w9, w9, w8
    orr     w0, w0, w9, lsl #16
    bl      emit32
.La_buf_done:
    mov     w0, w4                  // return result register
    ldp     x29, x30, [sp], #16
    ret
.La_ident_reg:
    ldr     w0, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    // Check for array subscript: name[expr]
    cmp     x19, x27
    b.hs    .La_ident_done
    ldr     w1, [x19]
    cmp     w1, #TOK_LBRACK
    b.ne    .La_ident_done
    // Array subscript: emit LDRB Xresult, [Xbase, Xindex]
    add     x19, x19, #TOK_STRIDE_SZ   // skip '['
    mov     w4, w0                      // base register
    stp     w4, wzr, [sp, #-16]!
    bl      parse_expr                  // index expression
    ldp     w4, wzr, [sp], #16
    mov     w6, w0                      // index register
    // Skip ']'
    cmp     x19, x27
    b.hs    .La_subscript_emit
    ldr     w1, [x19]
    cmp     w1, #TOK_RBRACK
    b.ne    .La_subscript_emit
    add     x19, x19, #TOK_STRIDE_SZ
.La_subscript_emit:
    // Emit: LDRB Wresult, [Xbase, Xindex]
    // Encoding: 0x38606800 | (Rm << 16) | (Rn << 5) | Rd
    bl      alloc_reg
    mov     w7, w0                      // result register
    lsl     w1, w6, #16                 // Rm (index)
    lsl     w2, w4, #5                  // Rn (base)
    orr     w0, w7, w1
    orr     w0, w0, w2
    movz    w3, #0x6800
    movk    w3, #0x3860, lsl #16
    orr     w0, w0, w3
    bl      emit32
    mov     w0, w7
.La_ident_done:
    ldp     x29, x30, [sp], #16
    ret

.La_ident_unknown:
    // Composition call inside expression — parse ONE argument only.
    // Statement-level calls use handle_call (greedy, until newline).
    // Expression-level calls are "func arg" — single arg, result in X0.
    bl      sym_lookup
    mov     x5, x0                  // sym entry (or NULL)
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    // Check if next token is a valid expression start (not operator/newline)
    cmp     x19, x27
    b.hs    .La_call_no_arg
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_EOF
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_RPAREN
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_INDENT
    b.eq    .La_call_no_arg
    // Check for binary operators — these mean no arg (e.g., "track_rd << 16")
    cmp     w0, #TOK_PLUS
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_MINUS
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_STAR
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_SLASH
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_AMP
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_PIPE
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_CARET
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_SHL
    b.eq    .La_call_no_arg
    cmp     w0, #TOK_SHR
    b.eq    .La_call_no_arg

    // Has arguments — parse until operator or ) or newline
    // This handles both "func arg" and "(func arg1 arg2)" patterns
    mov     w8, #0                  // arg index
.La_call_arg_loop:
    cmp     x19, x27
    b.hs    .La_call_emit
    ldr     w0, [x19]
    // Stop at operators, delimiters, and line boundaries
    cmp     w0, #TOK_NEWLINE
    b.eq    .La_call_emit
    cmp     w0, #TOK_EOF
    b.eq    .La_call_emit
    cmp     w0, #TOK_RPAREN
    b.eq    .La_call_emit
    cmp     w0, #TOK_INDENT
    b.eq    .La_call_emit
    cmp     w0, #TOK_PLUS
    b.eq    .La_call_emit
    cmp     w0, #TOK_MINUS
    b.eq    .La_call_emit
    cmp     w0, #TOK_STAR
    b.eq    .La_call_emit
    cmp     w0, #TOK_SLASH
    b.eq    .La_call_emit
    cmp     w0, #TOK_AMP
    b.eq    .La_call_emit
    cmp     w0, #TOK_PIPE
    b.eq    .La_call_emit
    cmp     w0, #TOK_CARET
    b.eq    .La_call_emit
    cmp     w0, #TOK_SHL
    b.eq    .La_call_emit
    cmp     w0, #TOK_SHR
    b.eq    .La_call_emit
    // Parse one arg expression (atom-level to avoid consuming operators)
    stp     x5, x8, [sp, #-16]!
    bl      parse_atom
    ldp     x5, x8, [sp], #16
    cmp     w0, w8
    b.eq    .La_call_arg_next
    // Move result to arg register
    stp     x5, x8, [sp, #-16]!
    mov     w1, w0
    mov     w0, w8
    bl      emit_mov_reg
    ldp     x5, x8, [sp], #16
.La_call_arg_next:
    add     w8, w8, #1
    cmp     w8, #8
    b.lt    .La_call_arg_loop
    b       .La_call_emit

.La_call_no_arg:
    // No argument — bare call
.La_call_emit:
    cbz     x5, .La_call_nop        // symbol not found
    ldr     w0, [x5, #SYM_KIND]
    cmp     w0, #KIND_COMP
    b.ne    .La_call_nop
    // Emit BL to composition (use same pattern as handle_call)
    ldr     w0, [x5, #SYM_REG]     // code address
    bl      emit_cur
    mov     x1, x0                  // x1 = current emit_ptr
    ldr     w0, [x5, #SYM_REG]     // re-load target (emit_cur clobbered x0)
    sub     x2, x0, x1             // offset = target - current
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      emit32
    b       .La_call_done
.La_call_nop:
    // Unknown composition — emit NOP
    mov     w0, #0x201F
    movk    w0, #0xD503, lsl #16
    bl      emit32
.La_call_done:
    mov     w0, #0                  // result assumed in X0
    ldp     x29, x30, [sp], #16
    ret

.La_paren:
    add     x19, x19, #TOK_STRIDE_SZ   // consume '('
    bl      parse_expr
    mov     w4, w0
    cmp     w4, #REG_FIRST
    b.lt    1f
    add     w0, w4, #1
    bl      free_reg
1:  cmp     x19, x27
    b.hs    .La_error
    ldr     w1, [x19]
    cmp     w1, #TOK_RPAREN
    b.ne    .La_error
    add     x19, x19, #TOK_STRIDE_SZ   // consume ')'
    mov     w0, w4
    ldp     x29, x30, [sp], #16
    ret

.La_neg:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_atom
    mov     w4, w0
    bl      alloc_reg
    mov     w5, w0
    // SUB Xd, XZR, Xoperand
    mov     w0, w5
    mov     w1, #31             // XZR
    mov     w2, w4
    bl      emit_sub_reg
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.La_dollar:
    // $N — raw register reference. Returns register N as the atom value.
    bl      parse_dollar_reg        // w0 = register number, x19 advanced
    ldp     x29, x30, [sp], #16
    ret

.La_load:
    add     x19, x19, #TOK_STRIDE_SZ   // skip '→'
    // Save width literal value before parse_atom consumes the token.
    // Width is always an INT literal (8, 32, or 64).
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    1f
    ldr     w1, [x19, #4]              // source offset
    ldr     w2, [x19, #8]              // token length
    // Quick decimal parse for 1-2 digit widths
    ldrb    w3, [x28, x1]              // first digit
    sub     w3, w3, #'0'
    cmp     w2, #1
    b.eq    2f
    mov     w4, #10
    mul     w3, w3, w4
    add     w1, w1, #1
    ldrb    w4, [x28, x1]
    sub     w4, w4, #'0'
    add     w3, w3, w4
2:  adrp    x0, ls_load_width
    add     x0, x0, :lo12:ls_load_width
    str     w3, [x0]
1:
    bl      parse_atom                  // width (8, 16, 32, 64)
    mov     w4, w0                      // width register
    bl      parse_expr                  // address/base expression
    mov     w5, w0                      // base register
    // Check for offset operand (token on same line, not an operator)
    cmp     x19, x27
    b.hs    .La_load_simple
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .La_load_simple
    cmp     w0, #TOK_EOF
    b.eq    .La_load_simple
    cmp     w0, #TOK_RPAREN
    b.eq    .La_load_simple
    cmp     w0, #TOK_INDENT
    b.eq    .La_load_simple
    // Has offset — parse it and emit width-aware indexed load
    stp     w4, w5, [sp, #-16]!
    bl      parse_atom                  // offset
    ldp     w4, w5, [sp], #16
    mov     w6, w0                      // offset register
    bl      alloc_reg
    mov     w7, w0                      // result register
    lsl     w1, w6, #16                 // Rm (offset)
    lsl     w2, w5, #5                  // Rn (base)
    orr     w0, w7, w1
    orr     w0, w0, w2
    // Select opcode by width stored in w4 (register containing width literal).
    // The width was parsed as an integer literal → MOV Xw4, #width.
    // We saved the width VALUE before alloc_reg in the parse_atom call.
    // Actually, w4 is the register number. We need to recover the width.
    // Trick: the .La_load code parsed width via parse_atom which for an
    // INT literal emits MOV Xreg, #imm and returns the register in w0.
    // We saved w0→w4. But we can't read the register value at compile time.
    //
    // Simpler: read the width token directly BEFORE calling parse_atom.
    // But we already consumed it. So we save the literal value separately.
    // For now, use the saved width literal from ls_load_width (set below).
    adrp    x3, ls_load_width
    add     x3, x3, :lo12:ls_load_width
    ldr     w3, [x3]
    cmp     w3, #64
    b.eq    .La_load_off_64
    cmp     w3, #32
    b.eq    .La_load_off_32
    // Default: 8-bit (LDRB)
    movz    w3, #0x6800
    movk    w3, #0x3860, lsl #16        // LDRB Wd, [Xn, Xm]
    b       .La_load_off_emit
.La_load_off_64:
    movz    w3, #0x6800
    movk    w3, #0xF860, lsl #16        // LDR Xd, [Xn, Xm]
    b       .La_load_off_emit
.La_load_off_32:
    movz    w3, #0x6800
    movk    w3, #0xB860, lsl #16        // LDR Wd, [Xn, Xm]
.La_load_off_emit:
    orr     w0, w0, w3
    bl      emit32
    mov     w0, w7
    ldp     x29, x30, [sp], #16
    ret
.La_load_simple:
    bl      alloc_reg
    mov     w6, w0
    mov     w0, w6
    mov     w1, w5
    mov     w2, #0
    bl      emit_ldr_imm
    mov     w0, w6
    ldp     x29, x30, [sp], #16
    ret

.La_store:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_atom
    mov     w4, w0
    bl      parse_atom
    mov     w5, w0
    bl      parse_atom
    mov     w6, w0
    mov     w0, w6
    mov     w1, w5
    mov     w2, #0
    bl      emit_str_imm
    mov     w0, w6
    ldp     x29, x30, [sp], #16
    ret

.La_reg_read:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_dollar_reg
    mov     w4, w0
    bl      alloc_reg
    mov     w5, w0
    mov     w0, w5
    mov     w1, w4
    bl      emit_mov_reg
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.La_reg_write:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_dollar_reg
    mov     w4, w0
    bl      parse_atom
    mov     w5, w0
    mov     w0, w4
    mov     w1, w5
    bl      emit_mov_reg
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.La_unary_math:
    // Unary prefix ops — pass through operand for bootstrap
    add     x19, x19, #TOK_STRIDE_SZ
    bl      parse_atom
    ldp     x29, x30, [sp], #16
    ret

.La_error:
    b       parse_error

// ============================================================
// Error handlers
// ============================================================
parse_error:
    adrp    x1, err_parse
    add     x1, x1, :lo12:err_parse
    mov     x2, #13
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    adrp    x1, err_at_tok
    add     x1, x1, :lo12:err_at_tok
    mov     x2, #10
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    ldr     w0, [x19]
    bl      print_dec
    adrp    x1, err_at_tok
    add     x1, x1, :lo12:err_at_tok
    mov     x2, #10
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    ldr     w0, [x19, #4]
    bl      print_dec
    adrp    x1, err_nl
    add     x1, x1, :lo12:err_nl
    mov     x2, #1
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #93             // SYS_EXIT
    svc     #0

parse_error_regspill:
    adrp    x1, err_regspill
    add     x1, x1, :lo12:err_regspill
    mov     x2, #22
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #93             // SYS_EXIT
    svc     #0

// print_dec — print w0 as decimal to stderr
print_dec:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32
    mov     w1, w0
    add     x2, sp, #31            // end of buffer
    mov     w3, #0                  // digit count
    mov     w4, #10
.Lpd_loop:
    udiv    w5, w1, w4
    msub    w6, w5, w4, w1
    add     w6, w6, #'0'
    strb    w6, [x2]
    sub     x2, x2, #1
    add     w3, w3, #1
    mov     w1, w5
    cbnz    w1, .Lpd_loop
    add     x1, x2, #1
    mov     x2, x3
    mov     x0, #2
    mov     x8, #64             // SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// DTC wrapper — code_PARSE_TOKENS
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
// Exports for lithos-glue.s and other modules
// ============================================================
.globl parse_expr
.globl parse_body
.globl parse_mem_load
.globl parse_mem_store
.globl parse_reg_read
.globl parse_reg_write
.globl parse_return_compose
.globl parse_binding_compose
.globl parse_buf_decl
.globl parse_stmt_compose
// parse_stmt_compose is just an alias for parse_statement
parse_stmt_compose:
    b       parse_statement

.globl parse_body_compose
parse_body_compose:
    b       parse_body

.globl parse_ident_compose
parse_ident_compose:
    b       handle_ident_stmt

// Additional globals for cross-module use
.globl parse_var_decl
parse_var_decl:
    b       handle_var

.globl parse_const_decl
parse_const_decl:
    b       handle_const

// ============================================================
// .data section — strings
// ============================================================
.data
.align 3

err_parse:
    .ascii "parse error\n"
    .byte 0
dbg_body:
    .ascii "BODY\n"

err_regspill:
    .ascii "register spill error\n"
    .byte 0

err_at_tok:
    .ascii " at offset "

err_nl:
    .byte 10

.align 3
str_trap:
    .ascii "trap"
str_goto:
    .ascii "goto"
str_continue:
    .ascii "continue"

// ============================================================
// .bss section — parser state
// ============================================================
.extern ls_sym_count
.extern ls_sym_table
.extern ls_code_buf

.bss
.align 3

scope_depth: .space 4
    .align 3
current_indent: .space 4
    .align 3
next_reg:   .space 4
    .align 3
reg_floor:  .space 4
    .align 3
spill_count: .space 4
    .align 3
emit_ptr:   .space 8
    .align 3
patch_stack:    .space (8 * MAX_PATCH)
patch_sp:       .space 4
    .align 3
loop_stack:     .space (16 * MAX_LOOP)
loop_sp:        .space 4
    .align 3

// ============================================================
// Dictionary entry — continues chain from emit-arm64.s
// ============================================================
.data
.align 3

entry_p_parse_tokens:
    .quad   entry_e_cbnz_fwd
    .byte   0
    .byte   12
    .ascii  "parse-tokens"
    .align  3
    .quad   code_PARSE_TOKENS
