.global code_PARSE_TOKENS
.global entry_p_parse_tokens
.global parse_tokens
.global parse_v3_error
.global parse_v3_error_regspill
.global v3_advance
.global v3_alloc_scratch
.global v3_at_line_end
.global v3_comp_add
.global v3_comp_count
.global v3_comp_lookup
.global v3_comp_table
.global v3_emit32
.global v3_emit_add
.global v3_emit_bl
.global v3_emit_cur
.global v3_emit_ldp_fp_lr
.global v3_emit_ldr
.global v3_emit_mov_imm64
.global v3_emit_mov_reg
.global v3_emit_mov_sp_fp
.global v3_emit_movk
.global v3_emit_movz
.global v3_emit_mul
.global v3_emit_ptr
.global v3_emit_ret
.global v3_emit_sdiv
.global v3_emit_stp_fp_lr
.global v3_emit_str
.global v3_emit_sub
.global v3_emit_svc
.global v3_err_parse
.global v3_err_regspill
.global v3_eval_operand
.global v3_find_main
.global v3_fixup_count
.global v3_fixup_table
.global v3_is_dollar
.global v3_match_str
.global v3_next_scratch
.global v3_parse_body
.global v3_parse_composition
.global v3_parse_int
.global v3_parse_line
.global v3_parse_line_rest
.global v3_parse_toplevel
.global v3_prescan_compositions
.global v3_reset_scratch
.global v3_resolve_fixups
.global v3_skip_ws
.global v3_sym_add
.global v3_sym_count
.global v3_sym_lookup
.global v3_sym_pop_to
.global v3_sym_table
.global v3_tok_text
.global v3_tok_type
.global v3_vsp
.global v3_vstack
.global vdepth
.global vpeek
.global vpop
.global vpush
.global vreset

// lithos-parser-v3.s — Stack-language parser for Lithos .ls files (v3)
//
// This is the third independent implementation. It implements the Lithos
// stack-language model: each line is one stack operation, the compiler
// tracks a virtual register stack, and binary/unary ops pop/push.
//
// Interface:
//   parse_tokens(x0=tok_buf, x1=tok_count, x2=src_buf)
//   code_PARSE_TOKENS — DTC wrapper ( tok-buf tok-count src-buf -- )
//
// Register conventions (inherited from lithos-bootstrap.s):
//   X26=IP  X25=W  X24=DSP  X23=RSP  X22=TOS  X20=HERE  X21=BASE
//
// Parser-private registers:
//   X19 = TOKP     — current token pointer
//   X27 = TOKEND   — past last token
//   X28 = SRC      — source buffer base
//
// Token triple layout: [+0] u32 type, [+4] u32 offset, [+8] u32 length
// Stride = 12 bytes.
//
// Virtual register stack:
//   vstack[0..vsp-1] holds register numbers (w8 values: 0-30).
//   vpush: store reg number at vstack[vsp++]
//   vpop:  load reg number from vstack[--vsp]
//   On entry to a composition body, params bound to X0-X7.
//   Scratch registers allocated from X9 upward (X8 reserved).

// ============================================================
// Token type constants (from lithos-lexer.s)
// ============================================================
.equ TOK_EOF,        0
.equ TOK_NEWLINE,    1
.equ TOK_INDENT,     2
.equ TOK_INT,        3
.equ TOK_FLOAT,      4
.equ TOK_IDENT,      5
.equ TOK_COLON,      72
.equ TOK_PLUS,       50
.equ TOK_MINUS,      51
.equ TOK_STAR,       52
.equ TOK_SLASH,      53
.equ TOK_LOAD,       36      // →
.equ TOK_STORE,      37      // ←
.equ TOK_REG_READ,   38      // ↑
.equ TOK_REG_WRITE,  39      // ↓
.equ TOK_TRAP,       89      // trap
.equ TOK_SQRT,       79      // √
.equ TOK_SUM,        75      // Σ
.equ TOK_MAX,        76      // △
.equ TOK_MIN,        77      // ▽
.equ TOK_INDEX,      78      // #
.equ TOK_SIN,        80      // ≅
.equ TOK_COS,        81      // ≡

.equ TOK_STRIDE_SZ,  12

// Symbol kinds
.equ SYM_SIZE,       24
.equ SYM_NAME_OFF,   0
.equ SYM_NAME_LEN,   4
.equ SYM_KIND,       8
.equ SYM_REG,        12
.equ SYM_DEPTH,      16

.equ KIND_PARAM,     1
.equ KIND_LOCAL,     0
.equ KIND_COMP,      4

.equ MAX_SYMS,       512
.equ MAX_VSTACK,     32
.equ MAX_COMPS,      64

// ARM64 instruction constants
.equ ARM64_RET,      0xD65F03C0
.equ ARM64_SVC_0,    0xD4000001
.equ ARM64_NOP,      0xD503201F

// ============================================================
// Macros
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

// ORRIMM macro — materialize constant via MOVZ+MOVK then ORR
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

// ============================================================
// External symbols from ls-shared.s
// ============================================================
.extern ls_code_buf
.extern ls_code_pos
.extern ls_token_buf
.extern ls_token_count
.extern ls_sym_table
.extern ls_sym_count
.extern ls_source_buf_ptr
.extern ls_data_pos

// ============================================================
// .text
// ============================================================
.text
.align 4

// ============================================================
// Virtual register stack operations
//
// The vstack holds register numbers. vsp is the count.
// vpush: store w0 (reg number) at vstack[vsp], increment vsp
// vpop:  decrement vsp, load reg number into w0
// ============================================================

// vpush — push register number w0 onto virtual stack
vpush:
    adrp    x1, v3_vsp
    add     x1, x1, :lo12:v3_vsp
    ldr     w2, [x1]
    adrp    x3, v3_vstack
    add     x3, x3, :lo12:v3_vstack
    str     w0, [x3, x2, lsl #2]
    add     w2, w2, #1
    str     w2, [x1]
    ret

// vpop — pop register number into w0
vpop:
    adrp    x1, v3_vsp
    add     x1, x1, :lo12:v3_vsp
    ldr     w2, [x1]
    cbz     w2, parse_v3_error
    sub     w2, w2, #1
    str     w2, [x1]
    adrp    x3, v3_vstack
    add     x3, x3, :lo12:v3_vstack
    ldr     w0, [x3, x2, lsl #2]
    ret

// vpeek — peek top register number into w0 (no pop)
vpeek:
    adrp    x1, v3_vsp
    add     x1, x1, :lo12:v3_vsp
    ldr     w2, [x1]
    cbz     w2, parse_v3_error
    sub     w2, w2, #1
    adrp    x3, v3_vstack
    add     x3, x3, :lo12:v3_vstack
    ldr     w0, [x3, x2, lsl #2]
    ret

// vdepth — return stack depth in w0
vdepth:
    adrp    x1, v3_vsp
    add     x1, x1, :lo12:v3_vsp
    ldr     w0, [x1]
    ret

// vreset — clear virtual stack
vreset:
    adrp    x1, v3_vsp
    add     x1, x1, :lo12:v3_vsp
    str     wzr, [x1]
    ret

// ============================================================
// Scratch register allocator
// X9-X15 are scratch (7 registers). X8 reserved for syscall nr.
// next_scratch tracks the next available (9..15).
// ============================================================
.equ SCRATCH_FIRST, 9
.equ SCRATCH_LAST,  15

v3_alloc_scratch:
    adrp    x0, v3_next_scratch
    add     x0, x0, :lo12:v3_next_scratch
    ldr     w1, [x0]
    cmp     w1, #SCRATCH_LAST
    b.gt    parse_v3_error_regspill
    add     w2, w1, #1
    str     w2, [x0]
    mov     w0, w1
    ret

v3_reset_scratch:
    adrp    x0, v3_next_scratch
    add     x0, x0, :lo12:v3_next_scratch
    mov     w1, #SCRATCH_FIRST
    str     w1, [x0]
    ret

// ============================================================
// Code emission helpers
// ============================================================

// v3_emit32 — emit 32-bit instruction word in w0
v3_emit32:
    adrp    x1, v3_emit_ptr
    add     x1, x1, :lo12:v3_emit_ptr
    ldr     x2, [x1]
    str     w0, [x2], #4
    str     x2, [x1]
    ret

// v3_emit_cur — return current emit address in x0
v3_emit_cur:
    adrp    x0, v3_emit_ptr
    add     x0, x0, :lo12:v3_emit_ptr
    ldr     x0, [x0]
    ret

// v3_emit_movz — MOVZ Xd, #imm16
//   w0 = dest reg, w1 = imm16
v3_emit_movz:
    stp     x30, xzr, [sp, #-16]!
    and     w2, w1, #0xFFFF
    lsl     w2, w2, #5         // imm16 field at [20:5]
    orr     w2, w2, w0         // Rd field at [4:0]
    ORRIMM  w2, 0xD2800000, w16  // MOVZ X
    mov     w0, w2
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_movk — MOVK Xd, #imm16, LSL #shift
//   w0 = dest reg, w1 = imm16, w2 = shift (0,16,32,48)
v3_emit_movk:
    stp     x30, xzr, [sp, #-16]!
    lsr     w3, w2, #4         // hw = shift/16
    lsl     w3, w3, #21        // hw field at [22:21]
    and     w4, w1, #0xFFFF
    lsl     w4, w4, #5         // imm16 field
    orr     w4, w4, w0         // Rd
    orr     w4, w4, w3         // hw
    ORRIMM  w4, 0xF2800000, w16  // MOVK X
    mov     w0, w4
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_mov_imm64 — load full 64-bit immediate into Xd
//   w0 = dest reg, x1 = imm64
v3_emit_mov_imm64:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x0, x1, [sp, #-16]!

    // MOVZ Xd, #(imm & 0xFFFF)
    and     w2, w1, #0xFFFF
    mov     w3, w0
    mov     w1, w2
    mov     w0, w3
    bl      v3_emit_movz

    ldp     x0, x1, [sp]      // reload

    // MOVK Xd, #((imm>>16) & 0xFFFF), LSL #16
    lsr     x4, x1, #16
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv3_mi64_c32
    mov     w1, w5
    mov     w2, #16
    bl      v3_emit_movk
    ldp     x0, x1, [sp]

.Lv3_mi64_c32:
    lsr     x4, x1, #32
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv3_mi64_c48
    mov     w1, w5
    mov     w2, #32
    bl      v3_emit_movk
    ldp     x0, x1, [sp]

.Lv3_mi64_c48:
    lsr     x4, x1, #48
    and     w5, w4, #0xFFFF
    cbz     w5, .Lv3_mi64_done
    mov     w1, w5
    mov     w2, #48
    bl      v3_emit_movk

.Lv3_mi64_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// v3_emit_add — ADD Xd, Xn, Xm
//   w0=d, w1=n, w2=m
v3_emit_add:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16        // Rm
    lsl     w4, w1, #5         // Rn
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x8B000000, w16  // ADD X
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_sub — SUB Xd, Xn, Xm
v3_emit_sub:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0xCB000000, w16
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_mul — MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR)
v3_emit_mul:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9B007C00, w16  // MADD Xd,Xn,Xm,XZR
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_sdiv — SDIV Xd, Xn, Xm
v3_emit_sdiv:
    stp     x30, xzr, [sp, #-16]!
    lsl     w3, w2, #16
    lsl     w4, w1, #5
    orr     w5, w0, w4
    orr     w5, w5, w3
    ORRIMM  w5, 0x9AC00C00, w16
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_mov_reg — MOV Xd, Xm  (ORR Xd, XZR, Xm)
//   w0=d, w1=m
v3_emit_mov_reg:
    stp     x30, xzr, [sp, #-16]!
    lsl     w2, w1, #16        // Rm
    mov     w3, #31
    lsl     w3, w3, #5         // Rn = XZR
    orr     w4, w0, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_svc — SVC #0
v3_emit_svc:
    stp     x30, xzr, [sp, #-16]!
    MOVI32  w0, ARM64_SVC_0
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_ret — RET
v3_emit_ret:
    stp     x30, xzr, [sp, #-16]!
    MOVI32  w0, ARM64_RET
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_bl — BL to address x0 (relative to current emit ptr)
//   x0 = target absolute address
v3_emit_bl:
    stp     x30, xzr, [sp, #-16]!
    mov     x5, x0
    bl      v3_emit_cur
    sub     x2, x5, x0         // byte offset from current PC
    asr     x2, x2, #2         // instruction offset
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x94000000, w16
    mov     w0, w2
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_stp_fp_lr — STP X29, X30, [SP, #-16]!
v3_emit_stp_fp_lr:
    stp     x30, xzr, [sp, #-16]!
    MOVI32  w0, 0xA9BF7BFD
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_ldp_fp_lr — LDP X29, X30, [SP], #16
v3_emit_ldp_fp_lr:
    stp     x30, xzr, [sp, #-16]!
    MOVI32  w0, 0xA8C17BFD
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_mov_sp_fp — MOV X29, SP
v3_emit_mov_sp_fp:
    stp     x30, xzr, [sp, #-16]!
    MOVI32  w0, 0x910003FD     // ADD X29, SP, #0
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// Memory load/store emitters by width
// v3_emit_ldr_w0 — LDR variant Xd, [Xn] (zero offset)
//   w0=d, w1=n, w2=width (8,16,32,64)
v3_emit_ldr:
    stp     x30, xzr, [sp, #-16]!
    cmp     w2, #8
    b.eq    .Lv3_ldrb
    cmp     w2, #16
    b.eq    .Lv3_ldrh
    cmp     w2, #32
    b.eq    .Lv3_ldrw
    // 64-bit
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9400000, w16
    b       .Lv3_ldr_emit
.Lv3_ldrb:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39400000, w16
    b       .Lv3_ldr_emit
.Lv3_ldrh:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79400000, w16
    b       .Lv3_ldr_emit
.Lv3_ldrw:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9400000, w16
.Lv3_ldr_emit:
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// v3_emit_str — STR variant Xd, [Xn] (zero offset)
//   w0=val_reg, w1=addr_reg, w2=width
v3_emit_str:
    stp     x30, xzr, [sp, #-16]!
    cmp     w2, #8
    b.eq    .Lv3_strb
    cmp     w2, #16
    b.eq    .Lv3_strh
    cmp     w2, #32
    b.eq    .Lv3_strw
    // 64-bit
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xF9000000, w16
    b       .Lv3_str_emit
.Lv3_strb:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x39000000, w16
    b       .Lv3_str_emit
.Lv3_strh:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0x79000000, w16
    b       .Lv3_str_emit
.Lv3_strw:
    lsl     w4, w1, #5
    orr     w5, w0, w4
    ORRIMM  w5, 0xB9000000, w16
.Lv3_str_emit:
    mov     w0, w5
    bl      v3_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Token helpers
// ============================================================

// v3_tok_type — type of current token in w0
v3_tok_type:
    ldr     w0, [x19]
    ret

// v3_tok_text — (x0=ptr, w1=len) of current token
v3_tok_text:
    ldr     w1, [x19, #8]
    ldr     w0, [x19, #4]
    add     x0, x28, x0
    ret

// v3_advance — move to next token
v3_advance:
    add     x19, x19, #TOK_STRIDE_SZ
    ret

// v3_skip_whitespace — skip NEWLINE and INDENT tokens
v3_skip_ws:
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

// v3_at_line_end — check if at end of line (NEWLINE, EOF, INDENT, or past end)
//   Returns w0=1 if at line end, w0=0 otherwise
v3_at_line_end:
    cmp     x19, x27
    b.hs    .Lv3_ale_yes
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_ale_yes
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_ale_yes
    cmp     w0, #TOK_INDENT
    b.eq    .Lv3_ale_yes
    mov     w0, #0
    ret
.Lv3_ale_yes:
    mov     w0, #1
    ret

// v3_match_str — check if current token text matches
//   x0 = compare str ptr, w1 = compare str len
//   Returns w0 = 1 match, 0 no match
v3_match_str:
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    mov     x4, x0
    mov     w5, w1
    ldr     w2, [x19, #8]      // token length
    cmp     w2, w5
    b.ne    .Lv3_ms_no
    ldr     w3, [x19, #4]
    add     x3, x28, x3        // token text
    mov     w2, #0
.Lv3_ms_loop:
    cmp     w2, w5
    b.ge    .Lv3_ms_yes
    ldrb    w0, [x3, x2]
    ldrb    w1, [x4, x2]
    cmp     w0, w1
    b.ne    .Lv3_ms_no
    add     w2, w2, #1
    b       .Lv3_ms_loop
.Lv3_ms_yes:
    mov     w0, #1
    b       .Lv3_ms_done
.Lv3_ms_no:
    mov     w0, #0
.Lv3_ms_done:
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ret

// ============================================================
// Symbol table — parser-local
//
// v3_sym_table: up to MAX_SYMS entries, each SYM_SIZE bytes
// v3_sym_count: current count
// Layout per entry: name_off(4) name_len(4) kind(4) reg(4) depth(4) pad(4)
// ============================================================

// v3_sym_lookup — find symbol matching current token
//   Returns x0 = pointer to entry, or 0
v3_sym_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w0, [x19, #4]      // token offset
    ldr     w1, [x19, #8]      // token length

    adrp    x2, v3_sym_count
    add     x2, x2, :lo12:v3_sym_count
    ldr     w3, [x2]
    cbz     w3, .Lv3_sl_notfound

    adrp    x4, v3_sym_table
    add     x4, x4, :lo12:v3_sym_table

    sub     w3, w3, #1
    mov     w5, #SYM_SIZE
    madd    x6, x3, x5, x4     // last entry

.Lv3_sl_loop:
    ldr     w7, [x6, #SYM_NAME_LEN]
    cmp     w7, w1
    b.ne    .Lv3_sl_next

    // Compare bytes
    ldr     w7, [x6, #SYM_NAME_OFF]
    add     x7, x28, x7
    add     x2, x28, x0
    mov     w5, #0
.Lv3_sl_cmp:
    cmp     w5, w1
    b.ge    .Lv3_sl_found
    ldrb    w3, [x7, x5]
    ldrb    w4, [x2, x5]
    cmp     w3, w4
    b.ne    .Lv3_sl_next
    add     w5, w5, #1
    b       .Lv3_sl_cmp

.Lv3_sl_found:
    mov     x0, x6
    b       .Lv3_sl_done

.Lv3_sl_next:
    adrp    x4, v3_sym_table
    add     x4, x4, :lo12:v3_sym_table
    cmp     x6, x4
    b.ls    .Lv3_sl_notfound
    sub     x6, x6, #SYM_SIZE
    b       .Lv3_sl_loop

.Lv3_sl_notfound:
    mov     x0, #0

.Lv3_sl_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// v3_sym_add — add symbol for current token
//   w1=kind, w2=reg, w3=scope_depth
//   Returns x0 = new entry pointer
v3_sym_add:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!

    adrp    x4, v3_sym_count
    add     x4, x4, :lo12:v3_sym_count
    ldr     w5, [x4]

    adrp    x0, v3_sym_table
    add     x0, x0, :lo12:v3_sym_table
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

// v3_sym_pop_to — restore sym count to w0
v3_sym_pop_to:
    adrp    x1, v3_sym_count
    add     x1, x1, :lo12:v3_sym_count
    str     w0, [x1]
    ret

// ============================================================
// Composition table — maps names to code addresses
// Each entry: name_off(4) name_len(4) code_addr(8) = 16 bytes
// ============================================================
.equ COMP_SIZE, 16

// v3_comp_lookup — find composition by current token
//   Returns x0 = code address (may be 0 for forward ref)
//   Returns x1 = 1 if found, 0 if not found
v3_comp_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w0, [x19, #4]
    ldr     w1, [x19, #8]

    adrp    x2, v3_comp_count
    add     x2, x2, :lo12:v3_comp_count
    ldr     w3, [x2]
    cbz     w3, .Lv3_cl_nf

    adrp    x4, v3_comp_table
    add     x4, x4, :lo12:v3_comp_table
    mov     w5, #0

.Lv3_cl_loop:
    cmp     w5, w3
    b.ge    .Lv3_cl_nf
    mov     w6, #COMP_SIZE
    madd    x6, x5, x6, x4

    ldr     w7, [x6, #4]       // name_len
    cmp     w7, w1
    b.ne    .Lv3_cl_next

    ldr     w7, [x6, #0]       // name_off
    add     x7, x28, x7
    add     x2, x28, x0
    mov     w8, #0
.Lv3_cl_cmp:
    cmp     w8, w1
    b.ge    .Lv3_cl_found
    ldrb    w9, [x7, x8]
    ldrb    w10, [x2, x8]
    cmp     w9, w10
    b.ne    .Lv3_cl_next
    add     w8, w8, #1
    b       .Lv3_cl_cmp

.Lv3_cl_found:
    ldr     x0, [x6, #8]       // code_addr (may be 0)
    mov     x1, #1              // found
    b       .Lv3_cl_done

.Lv3_cl_next:
    add     w5, w5, #1
    b       .Lv3_cl_loop

.Lv3_cl_nf:
    mov     x0, #0
    mov     x1, #0              // not found
.Lv3_cl_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// v3_comp_add — register composition: current token = name, x1 = code addr
//   If name already exists in table (from prescan), update its code addr.
//   Otherwise add a new entry.
v3_comp_add:
    stp     x30, xzr, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    stp     x8, x9, [sp, #-16]!
    mov     x9, x1             // save code addr

    // Search for existing entry
    ldr     w2, [x19, #4]      // token offset
    ldr     w3, [x19, #8]      // token length

    adrp    x4, v3_comp_count
    add     x4, x4, :lo12:v3_comp_count
    ldr     w5, [x4]

    adrp    x6, v3_comp_table
    add     x6, x6, :lo12:v3_comp_table
    mov     w7, #0

.Lv3_ca_search:
    cmp     w7, w5
    b.ge    .Lv3_ca_new
    mov     w8, #COMP_SIZE
    madd    x8, x7, x8, x6

    ldr     w0, [x8, #4]       // entry name_len
    cmp     w0, w3
    b.ne    .Lv3_ca_snext

    ldr     w0, [x8, #0]       // entry name_off
    add     x0, x28, x0
    add     x1, x28, x2        // token text
    mov     w10, #0
.Lv3_ca_scmp:
    cmp     w10, w3
    b.ge    .Lv3_ca_update
    ldrb    w11, [x0, x10]
    ldrb    w12, [x1, x10]
    cmp     w11, w12
    b.ne    .Lv3_ca_snext
    add     w10, w10, #1
    b       .Lv3_ca_scmp

.Lv3_ca_update:
    // Found — update code addr
    str     x9, [x8, #8]
    b       .Lv3_ca_done

.Lv3_ca_snext:
    add     w7, w7, #1
    b       .Lv3_ca_search

.Lv3_ca_new:
    // Add new entry
    adrp    x0, v3_comp_table
    add     x0, x0, :lo12:v3_comp_table
    mov     w6, #COMP_SIZE
    madd    x0, x5, x6, x0

    ldr     w6, [x19, #4]
    str     w6, [x0, #0]
    ldr     w6, [x19, #8]
    str     w6, [x0, #4]
    str     x9, [x0, #8]

    add     w5, w5, #1
    str     w5, [x4]

.Lv3_ca_done:
    ldp     x8, x9, [sp], #16
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Integer literal parser
// v3_parse_int — parse current token as integer, result in x0
// ============================================================
v3_parse_int:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!

    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x1, x28, x1

    mov     x0, #0
    mov     w3, #0
    mov     w7, #0              // negative flag

    // Leading '-'
    ldrb    w4, [x1]
    cmp     w4, #'-'
    b.ne    .Lv3_pi_hex
    mov     w7, #1
    add     w3, w3, #1

.Lv3_pi_hex:
    sub     w5, w2, w3
    cmp     w5, #2
    b.lt    .Lv3_pi_dec
    ldrb    w4, [x1, x3]
    cmp     w4, #'0'
    b.ne    .Lv3_pi_dec
    add     w6, w3, #1
    ldrb    w4, [x1, x6]
    cmp     w4, #'x'
    b.eq    .Lv3_pi_hex_go
    cmp     w4, #'X'
    b.ne    .Lv3_pi_dec

.Lv3_pi_hex_go:
    add     w3, w3, #2
.Lv3_pi_hex_loop:
    cmp     w3, w2
    b.ge    .Lv3_pi_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'9'
    b.le    .Lv3_pi_h09
    cmp     w4, #'F'
    b.le    .Lv3_pi_hAF
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lv3_pi_hacc
.Lv3_pi_hAF:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lv3_pi_hacc
.Lv3_pi_h09:
    sub     w4, w4, #'0'
.Lv3_pi_hacc:
    lsl     x0, x0, #4
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv3_pi_hex_loop

.Lv3_pi_dec:
.Lv3_pi_dec_loop:
    cmp     w3, w2
    b.ge    .Lv3_pi_sign
    ldrb    w4, [x1, x3]
    sub     w4, w4, #'0'
    cmp     w4, #9
    b.hi    .Lv3_pi_sign        // non-digit: stop
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv3_pi_dec_loop

.Lv3_pi_sign:
    cbz     w7, .Lv3_pi_done
    neg     x0, x0
.Lv3_pi_done:
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// v3_is_dollar — check if current token starts with '$'
//   Returns w0=1 if yes, w0=0 if no.
//   If yes, also returns the register number after $ in w1.
// ============================================================
v3_is_dollar:
    stp     x30, xzr, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    ldr     w2, [x19, #8]      // length
    cmp     w2, #2
    b.lt    .Lv3_id_no
    ldr     w3, [x19, #4]
    add     x3, x28, x3
    ldrb    w4, [x3]
    cmp     w4, #'$'
    b.ne    .Lv3_id_no
    // Parse number after $
    mov     x0, #0
    mov     w5, #1
.Lv3_id_num:
    cmp     w5, w2
    b.ge    .Lv3_id_yes
    ldrb    w4, [x3, x5]
    sub     w4, w4, #'0'
    cmp     w4, #9
    b.hi    .Lv3_id_no          // non-digit
    mov     x6, #10
    mul     x0, x0, x6
    add     x0, x0, x4
    add     w5, w5, #1
    b       .Lv3_id_num
.Lv3_id_yes:
    mov     w1, w0
    mov     w0, #1
    b       .Lv3_id_done
.Lv3_id_no:
    mov     w0, #0
    mov     w1, #0
.Lv3_id_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// MAIN ENTRY: parse_tokens
//   x0 = token buffer, x1 = token count, x2 = source buffer
// ============================================================
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

    // Init emit pointer
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v3_emit_ptr
    add     x1, x1, :lo12:v3_emit_ptr
    str     x0, [x1]

    // Clear state
    adrp    x0, v3_sym_count
    add     x0, x0, :lo12:v3_sym_count
    str     wzr, [x0]
    adrp    x0, v3_comp_count
    add     x0, x0, :lo12:v3_comp_count
    str     wzr, [x0]

    bl      vreset
    bl      v3_reset_scratch

    // Emit placeholder B instruction at start (will be patched to main)
    bl      v3_emit_cur
    str     x0, [sp, #-16]!   // save addr of placeholder B
    MOVI32  w0, 0x14000000     // B +0 (NOP-equivalent placeholder)
    bl      v3_emit32

    // Pass 1: pre-scan to register all composition names
    str     x19, [sp, #-16]!  // save token position
    bl      v3_prescan_compositions
    ldr     x19, [sp], #16    // restore token position

    // Clear fixup table
    adrp    x0, v3_fixup_count
    add     x0, x0, :lo12:v3_fixup_count
    str     wzr, [x0]

    // Pass 2: parse all top-level compositions
    bl      v3_parse_toplevel

    // Pass 3: resolve forward-reference fixups
    bl      v3_resolve_fixups

    // Patch the initial B to jump to "main" composition
    bl      v3_find_main       // returns x0 = main code addr, or 0
    cbz     x0, .Lv3_no_main
    mov     x1, x0            // target
    ldr     x0, [sp]          // addr of placeholder B
    sub     x2, x1, x0
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    ORRIMM  w2, 0x14000000, w16
    str     w2, [x0]
.Lv3_no_main:
    add     sp, sp, #16        // drop saved placeholder addr

    // Sync ls_code_pos
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v3_emit_ptr
    add     x1, x1, :lo12:v3_emit_ptr
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
// v3_parse_toplevel — scan for compositions (name args :)
// ============================================================
v3_parse_toplevel:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Lv3_tl_loop:
    cmp     x19, x27
    b.hs    .Lv3_tl_done

    bl      v3_skip_ws
    cmp     x19, x27
    b.hs    .Lv3_tl_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_tl_done

    // Must be IDENT to start a composition (or operator token for top-level stmts)
    cmp     w0, #TOK_IDENT
    b.eq    .Lv3_tl_check_comp

    // Skip non-composition tokens at top level
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_tl_loop

.Lv3_tl_check_comp:
    // Lookahead for COLON on this line
    mov     x4, x19
.Lv3_tl_scan:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lv3_tl_skip_line
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lv3_tl_is_comp
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_tl_skip_line
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_tl_skip_line
    b       .Lv3_tl_scan

.Lv3_tl_is_comp:
    bl      v3_parse_composition
    b       .Lv3_tl_loop

.Lv3_tl_skip_line:
    // Skip to next newline
    cmp     x19, x27
    b.hs    .Lv3_tl_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_tl_skip_nl
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_tl_done
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_tl_skip_line
.Lv3_tl_skip_nl:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_tl_loop

.Lv3_tl_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_prescan_compositions — register all composition names with addr=0
// ============================================================
v3_prescan_compositions:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Lv3_ps_loop:
    cmp     x19, x27
    b.hs    .Lv3_ps_done
    bl      v3_skip_ws
    cmp     x19, x27
    b.hs    .Lv3_ps_done
    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_ps_done
    cmp     w0, #TOK_IDENT
    b.ne    .Lv3_ps_skip

    // Lookahead for COLON
    mov     x4, x19
.Lv3_ps_scan:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lv3_ps_skipline
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lv3_ps_found
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_ps_skipline
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_ps_skipline
    b       .Lv3_ps_scan

.Lv3_ps_found:
    // Register composition name (x19 points to name token) with addr=0
    mov     x1, #0             // address = 0 (placeholder)
    bl      v3_comp_add
    // Skip past this composition (to next newline after colon, then skip body)
    // Just skip to next top-level line (indent=0 after a body)
    add     x19, x19, #TOK_STRIDE_SZ  // skip name
    b       .Lv3_ps_loop

.Lv3_ps_skipline:
.Lv3_ps_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_ps_loop

.Lv3_ps_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_find_main — find "main" in comp table, return code addr in x0
// ============================================================
v3_find_main:
    stp     x30, xzr, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!

    adrp    x2, v3_comp_count
    add     x2, x2, :lo12:v3_comp_count
    ldr     w3, [x2]
    cbz     w3, .Lv3_fm_nf

    adrp    x4, v3_comp_table
    add     x4, x4, :lo12:v3_comp_table
    mov     w5, #0

.Lv3_fm_loop:
    cmp     w5, w3
    b.ge    .Lv3_fm_nf
    mov     w6, #COMP_SIZE
    madd    x6, x5, x6, x4

    ldr     w7, [x6, #4]       // name_len
    cmp     w7, #4
    b.ne    .Lv3_fm_next
    ldr     w7, [x6, #0]       // name_off
    add     x7, x28, x7
    ldrb    w8, [x7]
    cmp     w8, #'m'
    b.ne    .Lv3_fm_next
    ldrb    w8, [x7, #1]
    cmp     w8, #'a'
    b.ne    .Lv3_fm_next
    ldrb    w8, [x7, #2]
    cmp     w8, #'i'
    b.ne    .Lv3_fm_next
    ldrb    w8, [x7, #3]
    cmp     w8, #'n'
    b.ne    .Lv3_fm_next

    // Found main
    ldr     x0, [x6, #8]       // code addr
    b       .Lv3_fm_done

.Lv3_fm_next:
    add     w5, w5, #1
    b       .Lv3_fm_loop

.Lv3_fm_nf:
    mov     x0, #0
.Lv3_fm_done:
    ldp     x2, x3, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// v3_resolve_fixups — patch forward-reference BL instructions
// ============================================================
v3_resolve_fixups:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    stp     x8, x9, [sp, #-16]!

    adrp    x0, v3_fixup_count
    add     x0, x0, :lo12:v3_fixup_count
    ldr     w1, [x0]
    cbz     w1, .Lv3_rf_done

    adrp    x2, v3_fixup_table
    add     x2, x2, :lo12:v3_fixup_table
    mov     w3, #0

.Lv3_rf_loop:
    cmp     w3, w1
    b.ge    .Lv3_rf_done
    lsl     w4, w3, #4
    add     x4, x2, x4
    ldr     x5, [x4, #0]       // emit addr
    ldr     w6, [x4, #8]       // name offset
    ldr     w7, [x4, #12]      // name length

    // Search comp table for this name
    adrp    x8, v3_comp_count
    add     x8, x8, :lo12:v3_comp_count
    ldr     w9, [x8]
    adrp    x8, v3_comp_table
    add     x8, x8, :lo12:v3_comp_table
    mov     w10, #0

.Lv3_rf_search:
    cmp     w10, w9
    b.ge    .Lv3_rf_next       // not found — leave as NOP
    mov     w11, #COMP_SIZE
    madd    x11, x10, x11, x8

    ldr     w12, [x11, #4]     // comp name_len
    cmp     w12, w7
    b.ne    .Lv3_rf_snext

    ldr     w12, [x11, #0]     // comp name_off
    add     x12, x28, x12
    add     x13, x28, x6       // fixup name text
    mov     w14, #0
.Lv3_rf_cmp:
    cmp     w14, w7
    b.ge    .Lv3_rf_patch
    ldrb    w15, [x12, x14]
    ldrb    w16, [x13, x14]
    cmp     w15, w16
    b.ne    .Lv3_rf_snext
    add     w14, w14, #1
    b       .Lv3_rf_cmp

.Lv3_rf_patch:
    // Found — patch BL at x5 to jump to comp code addr
    ldr     x0, [x11, #8]      // target code addr
    cbz     x0, .Lv3_rf_next   // still 0 — can't patch
    sub     x0, x0, x5          // byte offset
    asr     x0, x0, #2          // instruction offset
    and     w0, w0, #0x3FFFFFF
    ORRIMM  w0, 0x94000000, w16 // BL
    str     w0, [x5]
    b       .Lv3_rf_next

.Lv3_rf_snext:
    add     w10, w10, #1
    b       .Lv3_rf_search

.Lv3_rf_next:
    add     w3, w3, #1
    b       .Lv3_rf_loop

.Lv3_rf_done:
    ldp     x8, x9, [sp], #16
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_parse_composition — parse `name args : \n body`
//
// Stack-language model:
//   - Each body line is one stack operation
//   - Virtual register stack tracks values
//   - Args bound to X0-X7
//   - Scratch from X9 upward
// ============================================================
v3_parse_composition:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #48         // local storage

    // Save sym count for scope cleanup
    adrp    x0, v3_sym_count
    add     x0, x0, :lo12:v3_sym_count
    ldr     w4, [x0]
    str     w4, [sp, #0]       // saved sym count

    // Record composition name and code address
    bl      v3_emit_cur
    str     x0, [sp, #8]       // composition code address

    // Register in composition table
    mov     x1, x0
    bl      v3_comp_add

    // Also add to symbol table as KIND_COMP
    bl      v3_emit_cur
    mov     w2, w0
    mov     w1, #KIND_COMP
    mov     w3, #0
    bl      v3_sym_add

    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Emit function prologue
    bl      v3_emit_stp_fp_lr
    bl      v3_emit_mov_sp_fp

    // Reset scratch and vstack for this composition
    bl      v3_reset_scratch
    bl      vreset

    // Parse parameters (idents before colon)
    mov     w8, #0              // param index
.Lv3_pc_params:
    cmp     x19, x27
    b.hs    parse_v3_error
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lv3_pc_colon

    cmp     w0, #TOK_IDENT
    b.ne    parse_v3_error

    // Add param: kind=PARAM, reg=param_index (X0-X7)
    str     w8, [sp, #-16]!
    mov     w1, #KIND_PARAM
    mov     w2, w8
    mov     w3, #1              // scope depth 1
    bl      v3_sym_add
    ldr     w8, [sp], #16

    add     w8, w8, #1
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_pc_params

.Lv3_pc_colon:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ':'
    bl      v3_skip_ws

    // Determine body indent level
    mov     w9, #0              // default: any indent counts
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lv3_pc_body
    ldr     w9, [x19, #8]      // body indent level
    str     w9, [sp, #16]

.Lv3_pc_body:
    str     w9, [sp, #16]
    // Parse body lines
    bl      v3_parse_body

    // If vstack non-empty and top != X0, emit MOV X0, Xtop (return value)
    bl      vdepth
    cbz     w0, .Lv3_pc_no_retval
    bl      vpeek
    cmp     w0, #0
    b.eq    .Lv3_pc_no_retval
    // Emit MOV X0, Xtop
    mov     w1, w0
    mov     w0, #0
    bl      v3_emit_mov_reg
.Lv3_pc_no_retval:

    // Emit epilogue
    bl      v3_emit_ldp_fp_lr
    bl      v3_emit_ret

    // Restore sym count
    ldr     w0, [sp, #0]
    bl      v3_sym_pop_to

    // Reset vstack
    bl      vreset

    add     sp, sp, #48
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_parse_body — parse indented body lines
//   Reads lines until indent drops or EOF
// ============================================================
v3_parse_body:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    str     w9, [sp, #-16]!    // save body indent level

.Lv3_body_loop:
    cmp     x19, x27
    b.hs    .Lv3_body_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_body_done

    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_body_skip
    cmp     w0, #TOK_INDENT
    b.ne    .Lv3_body_line

    // Check indent level
    ldr     w9, [sp]           // reload saved indent
    ldr     w1, [x19, #8]
    cmp     w1, w9
    b.lt    .Lv3_body_done      // dedent
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_body_loop

.Lv3_body_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_body_loop

.Lv3_body_line:
    // Parse one stack-language line
    bl      v3_parse_line
    // Reset scratch regs between lines (but NOT vstack — it persists!)
    bl      v3_reset_scratch
    b       .Lv3_body_loop

.Lv3_body_done:
    add     sp, sp, #16        // drop saved indent
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_parse_line — parse one line of stack operations
//
// The first token determines the operation:
//   Integer literal → MOVZ into scratch, vpush
//   Known arg/local name → vpush its register (no codegen)
//   + - * / with operand → eval operand, pop from vstack, binary op
//   Bare + - * / → pop two from vstack, binary op
//   ↓ $N val → eval val, MOV to XN
//   ↑ $N → MOV from XN to scratch, vpush
//   trap → SVC #0
//   → width addr → eval addr, LDR, vpush
//   ← width addr val → eval addr, eval val, STR
//   Named intermediate: name rest → compile rest, bind name to result
//   Composition call: name args → eval args to X0-X7, BL
// ============================================================
v3_parse_line:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Skip to first real token on this line
    bl      v3_at_line_end
    cbnz    w0, .Lv3_pl_done

    ldr     w0, [x19]

    // --- Integer literal ---
    cmp     w0, #TOK_INT
    b.eq    .Lv3_pl_int

    // --- Operators (check for binary ops) ---
    cmp     w0, #TOK_PLUS
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv3_pl_binop

    // --- Register write: ↓ $N val ---
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv3_pl_reg_write

    // --- Register read: ↑ $N ---
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv3_pl_reg_read

    // --- trap ---
    cmp     w0, #TOK_TRAP
    b.eq    .Lv3_pl_trap

    // --- Load: → width addr ---
    cmp     w0, #TOK_LOAD
    b.eq    .Lv3_pl_load

    // --- Store: ← width addr val ---
    cmp     w0, #TOK_STORE
    b.eq    .Lv3_pl_store

    // --- Unary math ---
    cmp     w0, #TOK_SQRT
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_SUM
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_MAX
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_MIN
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_SIN
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_COS
    b.eq    .Lv3_pl_unary

    // --- Identifier ---
    cmp     w0, #TOK_IDENT
    b.eq    .Lv3_pl_ident

    // --- Unknown: skip ---
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_pl_skip_rest

.Lv3_pl_done:
    ldp     x29, x30, [sp], #16
    ret

// ---- Integer literal: MOVZ Xscratch, #val; vpush scratch ----
.Lv3_pl_int:
    bl      v3_parse_int
    str     x0, [sp, #-16]!   // save value
    bl      v3_alloc_scratch
    str     w0, [sp, #-16]!   // save scratch reg
    ldr     x1, [sp, #16]     // reload value
    bl      v3_emit_mov_imm64
    ldr     w5, [sp], #16     // restore scratch reg
    add     sp, sp, #16       // drop saved value
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    bl      vpush
    b       .Lv3_pl_skip_rest

// ---- Binary operator: + - * / ----
// Check if there's an operand after the operator on the same line.
// If operand: eval it, pop one from vstack, binary op, push result.
// If bare: pop two from vstack, binary op, push result.
.Lv3_pl_binop:
    mov     w10, w0            // save operator token type
    add     x19, x19, #TOK_STRIDE_SZ  // skip operator

    // Check if there's an operand
    bl      v3_at_line_end
    cbnz    w0, .Lv3_pl_binop_bare

    // There's an operand — evaluate it
    str     w10, [sp, #-16]!
    bl      v3_eval_operand     // result reg in w0
    ldr     w10, [sp], #16
    mov     w6, w0             // operand reg (right)

    // Pop one from vstack (left)
    stp     w6, w10, [sp, #-16]!
    bl      vpop
    ldp     w6, w10, [sp], #16
    mov     w5, w0             // left reg

    b       .Lv3_pl_binop_emit

.Lv3_pl_binop_bare:
    // Bare operator: pop two
    str     w10, [sp, #-16]!
    bl      vpop               // right (top of stack)
    mov     w6, w0
    bl      vpop               // left
    mov     w5, w0
    ldr     w10, [sp], #16

.Lv3_pl_binop_emit:
    // Allocate result register
    // w5=left, w6=right, w10=op token — save all on stack
    sub     sp, sp, #32
    str     w5, [sp, #0]
    str     w6, [sp, #4]
    str     w10, [sp, #8]
    bl      v3_alloc_scratch
    str     w0, [sp, #12]     // w7 = result reg

    // Reload and emit binary instruction
    ldr     w10, [sp, #8]
    ldr     w5, [sp, #0]
    ldr     w6, [sp, #4]
    ldr     w7, [sp, #12]

    cmp     w10, #TOK_PLUS
    b.eq    .Lv3_pl_bo_add
    cmp     w10, #TOK_MINUS
    b.eq    .Lv3_pl_bo_sub
    cmp     w10, #TOK_STAR
    b.eq    .Lv3_pl_bo_mul
    // slash = divide
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_sdiv
    b       .Lv3_pl_bo_push

.Lv3_pl_bo_add:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_add
    b       .Lv3_pl_bo_push

.Lv3_pl_bo_sub:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_sub
    b       .Lv3_pl_bo_push

.Lv3_pl_bo_mul:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_mul

.Lv3_pl_bo_push:
    ldr     w7, [sp, #12]     // reload result reg (clobbered by emit)
    add     sp, sp, #32
    mov     w0, w7
    bl      vpush
    b       .Lv3_pl_skip_rest

// ---- Register write: ↓ $N val ----
.Lv3_pl_reg_write:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↓

    // Parse $N — expect IDENT starting with $, or INT
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    .Lv3_rw_int_reg

    bl      v3_is_dollar
    cbz     w0, parse_v3_error
    mov     w4, w1             // target hardware register
    add     x19, x19, #TOK_STRIDE_SZ  // skip $N
    b       .Lv3_rw_val

.Lv3_rw_int_reg:
    cmp     w0, #TOK_INT
    b.ne    parse_v3_error
    bl      v3_parse_int
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_rw_val

.Lv3_rw_val:
    // Parse value operand
    str     w4, [sp, #-16]!
    bl      v3_eval_operand
    ldr     w4, [sp], #16
    mov     w5, w0             // value reg

    // Emit: MOV X<target>, Xvalue
    mov     w0, w4
    mov     w1, w5
    bl      v3_emit_mov_reg
    b       .Lv3_pl_skip_rest

// ---- Register read: ↑ $N ----
.Lv3_pl_reg_read:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ↑

    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    .Lv3_rr_int

    bl      v3_is_dollar
    cbz     w0, parse_v3_error
    mov     w4, w1             // source hardware register
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_rr_emit

.Lv3_rr_int:
    cmp     w0, #TOK_INT
    b.ne    parse_v3_error
    bl      v3_parse_int
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ

.Lv3_rr_emit:
    // Allocate scratch, MOV Xscratch, X<source>
    str     w4, [sp, #-16]!   // save source reg
    bl      v3_alloc_scratch
    mov     w5, w0
    stp     w5, w4, [sp, #-16]!  // save scratch reg + dummy
    ldr     w4, [sp, #16]     // reload source from outer save
    mov     w0, w5
    mov     w1, w4
    bl      v3_emit_mov_reg
    ldp     w5, w4, [sp], #16
    add     sp, sp, #16
    // Push result onto vstack
    mov     w0, w5
    bl      vpush
    b       .Lv3_pl_skip_rest

// ---- trap ----
.Lv3_pl_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v3_emit_svc
    b       .Lv3_pl_skip_rest

// ---- Load: → width addr ----
.Lv3_pl_load:
    add     x19, x19, #TOK_STRIDE_SZ  // skip →

    // Parse width (integer)
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    parse_v3_error
    bl      v3_parse_int
    mov     w4, w0             // width
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse addr operand
    str     w4, [sp, #-16]!
    bl      v3_eval_operand
    ldr     w4, [sp], #16
    mov     w5, w0             // addr reg

    // Allocate result
    stp     w4, w5, [sp, #-16]!
    bl      v3_alloc_scratch
    mov     w6, w0
    ldp     w4, w5, [sp], #16

    // Emit LDR variant — save w6 across call
    str     w6, [sp, #-16]!
    mov     w0, w6
    mov     w1, w5
    mov     w2, w4
    bl      v3_emit_ldr

    ldr     w6, [sp], #16
    mov     w0, w6
    bl      vpush
    b       .Lv3_pl_skip_rest

// ---- Store: ← width addr val ----
.Lv3_pl_store:
    add     x19, x19, #TOK_STRIDE_SZ  // skip ←

    // Width
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    parse_v3_error
    bl      v3_parse_int
    mov     w4, w0
    add     x19, x19, #TOK_STRIDE_SZ

    // Addr
    str     w4, [sp, #-16]!
    bl      v3_eval_operand
    ldr     w4, [sp], #16
    mov     w5, w0

    // Val
    stp     w4, w5, [sp, #-16]!
    bl      v3_eval_operand
    ldp     w4, w5, [sp], #16
    mov     w6, w0

    // Emit STR variant
    mov     w0, w6             // val
    mov     w1, w5             // addr
    mov     w2, w4             // width
    bl      v3_emit_str
    b       .Lv3_pl_skip_rest

// ---- Unary operators (√, Σ, etc.) ----
.Lv3_pl_unary:
    // Pop top of vstack, apply unary, push result
    // For ARM64 host, these are stubs (NOP/passthrough)
    add     x19, x19, #TOK_STRIDE_SZ
    // Result passes through top of stack
    b       .Lv3_pl_skip_rest

// ---- Identifier ----
// Could be:
//   1. Known arg/local name → vpush its register
//   2. "trap" keyword
//   3. Named intermediate: if next tokens form a line, compile the rest
//      and bind name to result
//   4. Composition call
.Lv3_pl_ident:
    // Check for "trap" text
    ldr     w1, [x19, #8]
    cmp     w1, #4
    b.ne    .Lv3_pl_ident_check
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'t'
    b.ne    .Lv3_pl_ident_check
    ldrb    w3, [x2, #1]
    cmp     w3, #'r'
    b.ne    .Lv3_pl_ident_check
    ldrb    w3, [x2, #2]
    cmp     w3, #'a'
    b.ne    .Lv3_pl_ident_check
    ldrb    w3, [x2, #3]
    cmp     w3, #'p'
    b.ne    .Lv3_pl_ident_check
    add     x19, x19, #TOK_STRIDE_SZ
    bl      v3_emit_svc
    b       .Lv3_pl_skip_rest

.Lv3_pl_ident_check:
    // Check for "acc" (dup top)
    ldr     w1, [x19, #8]
    cmp     w1, #3
    b.ne    .Lv3_pl_ident_lookup
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'a'
    b.ne    .Lv3_pl_ident_lookup
    ldrb    w3, [x2, #1]
    cmp     w3, #'c'
    b.ne    .Lv3_pl_ident_lookup
    ldrb    w3, [x2, #2]
    cmp     w3, #'c'
    b.ne    .Lv3_pl_ident_lookup
    // acc = dup top of vstack
    add     x19, x19, #TOK_STRIDE_SZ
    bl      vpeek
    bl      vpush
    b       .Lv3_pl_skip_rest

.Lv3_pl_ident_lookup:
    // Look up in symbol table
    bl      v3_sym_lookup
    cbnz    x0, .Lv3_pl_ident_found

    // Not found — check if it's a known composition (possibly forward ref)
    bl      v3_comp_lookup
    cbnz    x1, .Lv3_pl_comp_call   // x1=1 means found; x0=code addr (may be 0)

    // Not a known symbol or composition.
    // Check if next token is also on this line (not an operator) —
    // if so, this is a named intermediate: name rest-of-line
    // Otherwise skip.
    mov     x4, x19            // save position
    add     x5, x19, #TOK_STRIDE_SZ
    cmp     x5, x27
    b.hs    .Lv3_pl_ident_skip
    ldr     w0, [x5]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_pl_ident_skip
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_pl_ident_skip
    cmp     w0, #TOK_INDENT
    b.eq    .Lv3_pl_ident_skip

    // Named intermediate: save name token, skip name, parse rest of line
    mov     x4, x19            // name token position
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse rest of line as sub-operations
    stp     x4, xzr, [sp, #-16]!
    bl      v3_parse_line_rest
    ldp     x4, xzr, [sp], #16

    // Peek top of vstack — that's the result
    bl      vpeek
    mov     w5, w0             // result reg

    // Bind name to this register
    mov     x19, x4            // point to name token
    mov     w1, #KIND_LOCAL
    mov     w2, w5
    mov     w3, #1
    bl      v3_sym_add
    mov     x19, x4
    // Skip to end of this line (already parsed by parse_line_rest)
    add     x19, x19, #TOK_STRIDE_SZ  // skip name token (already parsed rest)
    // skip to line end
    b       .Lv3_pl_skip_to_eol

.Lv3_pl_ident_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_pl_skip_rest

.Lv3_pl_ident_found:
    // Known symbol — push its register onto vstack
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w4
    bl      vpush

    // Check for more tokens on this line — could be operators
    bl      v3_at_line_end
    cbnz    w0, .Lv3_pl_skip_rest

    // More tokens — these could be operator+operand patterns
    // Continue parsing them as additional operations on this line
    b       .Lv3_pl_check_more

.Lv3_pl_check_more:
    bl      v3_at_line_end
    cbnz    w0, .Lv3_pl_skip_rest
    ldr     w0, [x19]
    cmp     w0, #TOK_PLUS
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv3_pl_binop
    cmp     w0, #TOK_SQRT
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_SUM
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_MAX
    b.eq    .Lv3_pl_unary
    cmp     w0, #TOK_MIN
    b.eq    .Lv3_pl_unary
    // Unknown extra token — skip rest
    b       .Lv3_pl_skip_rest

// ---- Composition call ----
// x0 = composition code address from v3_comp_lookup (may be 0 for forward ref)
.Lv3_pl_comp_call:
    mov     x5, x0             // code address (0 = forward ref)
    // Save the composition name info for potential fixup
    ldr     w14, [x19, #4]     // name offset in source
    ldr     w15, [x19, #8]     // name length
    add     x19, x19, #TOK_STRIDE_SZ  // skip name

    // Evaluate arguments into X0, X1, ...
    // Pop from vstack for each pending arg, or parse inline args
    // First: parse any inline arguments on this line
    mov     w8, #0              // arg count

.Lv3_cc_args:
    bl      v3_at_line_end
    cbnz    w0, .Lv3_cc_emit

    // Evaluate next arg operand — save x5, x8, x14, x15
    sub     sp, sp, #32
    str     x5, [sp, #0]
    str     x8, [sp, #8]
    str     x14, [sp, #16]
    str     x15, [sp, #24]
    bl      v3_eval_operand
    ldr     x5, [sp, #0]
    ldr     x8, [sp, #8]
    ldr     x14, [sp, #16]
    ldr     x15, [sp, #24]
    add     sp, sp, #32
    mov     w6, w0             // arg reg

    // Move to X<arg_index>
    cmp     w6, w8
    b.eq    .Lv3_cc_no_mov
    sub     sp, sp, #32
    str     x5, [sp, #0]
    str     x8, [sp, #8]
    str     x14, [sp, #16]
    str     x15, [sp, #24]
    mov     w0, w8
    mov     w1, w6
    bl      v3_emit_mov_reg
    ldr     x5, [sp, #0]
    ldr     x8, [sp, #8]
    ldr     x14, [sp, #16]
    ldr     x15, [sp, #24]
    add     sp, sp, #32
.Lv3_cc_no_mov:
    add     w8, w8, #1
    b       .Lv3_cc_args

.Lv3_cc_emit:
    // Emit BL to composition
    cbnz    x5, .Lv3_cc_emit_known
    // Forward reference: emit placeholder BL and record fixup
    bl      v3_emit_cur
    mov     x6, x0             // save address of BL instruction
    MOVI32  w0, 0x94000000     // BL +0 (placeholder)
    bl      v3_emit32
    // Record fixup: (emit_addr, name_off, name_len)
    adrp    x0, v3_fixup_count
    add     x0, x0, :lo12:v3_fixup_count
    ldr     w1, [x0]
    adrp    x2, v3_fixup_table
    add     x2, x2, :lo12:v3_fixup_table
    // Each fixup = 16 bytes: emit_addr(8) + name_off(4) + name_len(4)
    lsl     w3, w1, #4         // * 16
    add     x2, x2, x3
    str     x6, [x2, #0]       // emit addr
    str     w14, [x2, #8]      // name offset
    str     w15, [x2, #12]     // name length
    add     w1, w1, #1
    str     w1, [x0]
    b       .Lv3_cc_emit_done

.Lv3_cc_emit_known:
    mov     x0, x5
    bl      v3_emit_bl

.Lv3_cc_emit_done:
    // Result in X0 — push onto vstack
    mov     w0, #0
    bl      vpush
    b       .Lv3_pl_skip_rest

// ---- Skip to end of line ----
.Lv3_pl_skip_rest:
.Lv3_pl_skip_to_eol:
    cmp     x19, x27
    b.hs    .Lv3_pl_done2
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv3_pl_done2
    cmp     w0, #TOK_EOF
    b.eq    .Lv3_pl_done2
    cmp     w0, #TOK_INDENT
    b.eq    .Lv3_pl_done2
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_pl_skip_rest

.Lv3_pl_done2:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_parse_line_rest — parse remaining tokens on current line
//   as additional stack operations (after a named intermediate skip)
// ============================================================
v3_parse_line_rest:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Lv3_plr_loop:
    bl      v3_at_line_end
    cbnz    w0, .Lv3_plr_done

    ldr     w0, [x19]

    cmp     w0, #TOK_INT
    b.eq    .Lv3_plr_int
    cmp     w0, #TOK_PLUS
    b.eq    .Lv3_plr_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv3_plr_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv3_plr_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv3_plr_binop
    cmp     w0, #TOK_IDENT
    b.eq    .Lv3_plr_ident

    // Unknown — skip
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_plr_loop

.Lv3_plr_int:
    bl      v3_parse_int
    str     x0, [sp, #-16]!   // save value
    bl      v3_alloc_scratch
    str     w0, [sp, #-16]!   // save scratch reg
    ldr     x1, [sp, #16]     // reload value
    bl      v3_emit_mov_imm64
    ldr     w5, [sp], #16     // restore scratch reg
    add     sp, sp, #16       // drop saved value
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    bl      vpush
    b       .Lv3_plr_loop

.Lv3_plr_binop:
    mov     w10, w0
    add     x19, x19, #TOK_STRIDE_SZ

    bl      v3_at_line_end
    cbnz    w0, .Lv3_plr_binop_bare

    // Has operand
    str     w10, [sp, #-16]!
    bl      v3_eval_operand
    ldr     w10, [sp], #16
    mov     w6, w0
    stp     w6, w10, [sp, #-16]!
    bl      vpop
    ldp     w6, w10, [sp], #16
    mov     w5, w0
    b       .Lv3_plr_bo_emit

.Lv3_plr_binop_bare:
    str     w10, [sp, #-16]!
    bl      vpop
    mov     w6, w0
    bl      vpop
    mov     w5, w0
    ldr     w10, [sp], #16

.Lv3_plr_bo_emit:
    sub     sp, sp, #32
    str     w5, [sp, #0]
    str     w6, [sp, #4]
    str     w10, [sp, #8]
    bl      v3_alloc_scratch
    str     w0, [sp, #12]     // result reg
    ldr     w10, [sp, #8]
    ldr     w5, [sp, #0]
    ldr     w6, [sp, #4]
    ldr     w7, [sp, #12]

    cmp     w10, #TOK_PLUS
    b.eq    .Lv3_plr_bo_add
    cmp     w10, #TOK_MINUS
    b.eq    .Lv3_plr_bo_sub
    cmp     w10, #TOK_STAR
    b.eq    .Lv3_plr_bo_mul
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_sdiv
    b       .Lv3_plr_bo_push
.Lv3_plr_bo_add:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_add
    b       .Lv3_plr_bo_push
.Lv3_plr_bo_sub:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_sub
    b       .Lv3_plr_bo_push
.Lv3_plr_bo_mul:
    mov     w0, w7
    mov     w1, w5
    mov     w2, w6
    bl      v3_emit_mul
.Lv3_plr_bo_push:
    ldr     w7, [sp, #12]
    add     sp, sp, #32
    mov     w0, w7
    bl      vpush
    b       .Lv3_plr_loop

.Lv3_plr_ident:
    bl      v3_sym_lookup
    cbz     x0, .Lv3_plr_ident_skip
    ldr     w4, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w4
    bl      vpush
    b       .Lv3_plr_loop
.Lv3_plr_ident_skip:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lv3_plr_loop

.Lv3_plr_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v3_eval_operand — evaluate the next token as a single operand
//   Returns the register number holding the value in w0.
//   Handles: INT literal, IDENT (known name), $N register ref
// ============================================================
v3_eval_operand:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_INT
    b.eq    .Lv3_eo_int

    cmp     w0, #TOK_IDENT
    b.eq    .Lv3_eo_ident

    // Fallback: try parsing from vstack
    bl      vpop
    ldp     x29, x30, [sp], #16
    ret

.Lv3_eo_int:
    bl      v3_parse_int
    mov     x4, x0
    bl      v3_alloc_scratch
    mov     w5, w0
    // Save scratch reg number across emit call
    str     w5, [sp, #-16]!
    mov     w0, w5
    mov     x1, x4
    bl      v3_emit_mov_imm64
    ldr     w5, [sp], #16
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

.Lv3_eo_ident:
    // Check for $N
    bl      v3_is_dollar
    cbnz    w0, .Lv3_eo_dollar

    // Look up symbol
    bl      v3_sym_lookup
    cbz     x0, .Lv3_eo_unknown
    ldr     w0, [x0, #SYM_REG]
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

.Lv3_eo_dollar:
    // w1 = register number
    mov     w0, w1
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

.Lv3_eo_unknown:
    // Unknown ident as operand — allocate scratch, load 0
    bl      v3_alloc_scratch
    str     w0, [sp, #-16]!
    mov     x1, #0
    bl      v3_emit_mov_imm64
    ldr     w5, [sp], #16
    add     x19, x19, #TOK_STRIDE_SZ
    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Error handlers
// ============================================================

parse_v3_error:
    adrp    x1, v3_err_parse
    add     x1, x1, :lo12:v3_err_parse
    mov     x2, #16
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

parse_v3_error_regspill:
    adrp    x1, v3_err_regspill
    add     x1, x1, :lo12:v3_err_regspill
    mov     x2, #25
    mov     x0, #2
    mov     x8, #64
    svc     #0
    mov     x0, #1
    mov     x8, #93
    svc     #0

// ============================================================
// DTC wrapper
// Stack interface: ( tok-buf tok-count src-buf -- )
// ============================================================
.align 4
.global code_PARSE_TOKENS
code_PARSE_TOKENS:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x20, [sp, #-16]!

    mov     x2, x22            // TOS = src-buf
    ldr     x1, [x24], #8     // tok-count
    ldr     x0, [x24], #8     // tok-buf
    ldr     x22, [x24], #8    // new TOS

    bl      parse_tokens

    ldp     x22, x20, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// ============================================================
// .data
// ============================================================
.data
.align 3

v3_err_parse:
    .ascii "v3: parse error\n"

v3_err_regspill:
    .ascii "v3: register spill error\n"

// ============================================================
// .bss — parser-private state
// ============================================================
.bss
.align 3

// Virtual register stack
v3_vstack:      .space (MAX_VSTACK * 4)
v3_vsp:         .space 4
    .align 3

// Scratch register allocator
v3_next_scratch: .space 4
    .align 3

// Emit cursor
v3_emit_ptr:    .space 8
    .align 3

// Symbol table (parser-local scope)
v3_sym_table:   .space (MAX_SYMS * SYM_SIZE)
v3_sym_count:   .space 4
    .align 3

// Composition table
v3_comp_table:  .space (MAX_COMPS * COMP_SIZE)
v3_comp_count:  .space 4
    .align 3

// Forward-reference fixup table
// Each entry: emit_addr(8) + name_off(4) + name_len(4) = 16 bytes
.equ MAX_FIXUPS, 128
v3_fixup_table: .space (MAX_FIXUPS * 16)
v3_fixup_count: .space 4
    .align 3

// ============================================================
// Dictionary entry
// ============================================================
.data
.align 3

// Dictionary chain: emit-arm64.s tail → here → lithos-elf-writer.s
// The emit-arm64.s tail entry label varies; find it:
.extern entry_e_cbnz_fwd

entry_p_parse_tokens:
    .quad   entry_e_cbnz_fwd
    .byte   0
    .byte   12
    .ascii  "parse-tokens"
    .align  3
    .quad   code_PARSE_TOKENS
