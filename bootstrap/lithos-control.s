// lithos-control.s — Control flow and memory operation parsing for Lithos bootstrap
//
// Linked alongside lithos-parser.s (do NOT modify that file).
// Provides: parse_for, parse_if_eq, parse_if_ge, parse_if_lt,
//           parse_mem_load, parse_mem_store, parse_reg_read,
//           parse_reg_write, parse_trap
//
// Build: aarch64-linux-gnu-as -o lithos-control.o lithos-control.s
//        Link with lithos-parser.o, emit-arm64.o, lithos-bootstrap.o
//
// Register conventions (inherited from lithos-parser.s):
//   X19 = TOKP   — pointer to current token triple
//   X27 = TOKEND — pointer past last token
//   X28 = SRC    — pointer to source buffer
//   X20 = HERE   — dictionary/code-space pointer
//
// Token triple layout: [+0] u32 type, [+4] u32 offset, [+8] u32 length
//   Stride = 12 bytes per token.

// ============================================================
// Token type constants (must match lithos-parser.s)
// ============================================================
.equ TOK_EOF,       0
.equ TOK_NEWLINE,   1
.equ TOK_INDENT,    2
.equ TOK_INT,       3
.equ TOK_IDENT,     5
.equ TOK_IF,        13
.equ TOK_ELSE,      14
.equ TOK_FOR,       16
.equ TOK_ENDFOR,    17
.equ TOK_LOAD,      36
.equ TOK_STORE,     37
.equ TOK_REG_READ,  38
.equ TOK_REG_WRITE, 39
.equ TOK_EQEQ,     55
.equ TOK_LT,       57
.equ TOK_GTE,      60

.equ TOK_STRIDE_SZ, 12

// Symbol table constants
.equ SYM_SIZE,      24
.equ SYM_NAME_OFF,  0
.equ SYM_NAME_LEN,  4
.equ SYM_KIND,      8
.equ SYM_REG,       12
.equ SYM_DEPTH,     16
.equ KIND_LOCAL_REG, 0

// ARM64 instruction constants
.equ ARM64_SVC_0,   0xD4000001

// Condition code constants
.equ CC_EQ, 0
.equ CC_NE, 1
.equ CC_LT, 11
.equ CC_GE, 10

// For-loop nesting stack: max 8 deep
// Each entry: (loop_top_addr, patch_addr, counter_reg, end_reg, step_reg)
//   = 5 * 8 = 40 bytes per entry
.equ LOOP_ENTRY_SZ, 40
.equ MAX_LOOP_DEPTH, 8

// ORRIMM macro — OR a 32-bit immediate into Wd via a tmp register
.macro ORRIMM Wd, imm, Wtmp
    mov     \Wtmp, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wtmp, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
    orr     \Wd, \Wd, \Wtmp
.endm

// MOVI32 macro — load arbitrary 32-bit immediate into Wd
.macro MOVI32 Wd, imm
    mov     \Wd, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wd, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
.endm

// ============================================================
// External symbols (from lithos-parser.s and friends)
// ============================================================
.extern parse_expr          // ( -- w0=result_reg )
.extern parse_body          // w9=min_indent
.extern parse_int_literal   // ( -- x0=value )
.extern parse_error         // noreturn
.extern parse_error_regspill
.extern alloc_reg           // ( -- w0=reg_number )
.extern free_reg            // w0=reg to free
.extern reset_regs
.extern sym_add             // token at X19; w1=kind, w2=reg, w3=depth
.extern sym_lookup          // ( -- x0=sym_entry or 0 )
.extern skip_newlines
.extern emit32              // w0=instruction word
.extern emit_cur            // ( -- x0=current emit address )
.extern emit_mov_imm64      // w0=dest_reg, x1=imm64
.extern emit_cmp_reg        // w0=Xn, w1=Xm
.extern emit_add_reg        // w0=d, w1=n, w2=m
.extern emit_b              // x0=target address
.extern emit_b_cond         // w0=cond, x1=target
.extern emit_nop            // emit NOP placeholder
.extern patch_b             // x0=addr, x1=target
.extern patch_b_cond        // x0=addr, x1=target
.extern emit_ldrb_zero      // w0=d, w1=n  (LDRB Wd, [Xn])
.extern emit_strb_zero      // w0=d, w1=n
.extern emit_ldrh_zero      // w0=d, w1=n
.extern emit_strh_zero      // w0=d, w1=n
.extern emit_ldr_w_zero     // w0=d, w1=n  (LDR Wd, [Xn])
.extern emit_str_w_zero     // w0=d, w1=n
.extern emit_ldr_x_zero     // w0=d, w1=n  (LDR Xd, [Xn])
.extern emit_str_x_zero     // w0=d, w1=n
.extern scope_depth
.extern emit_ptr

// ============================================================
// .text — Control flow and memory parsers
// ============================================================

.text
.align 4

// ============================================================
// 1. parse_for — for i start end step ... endfor
//
//    Emits:
//      MOV Xi, Xstart
//    loop_top:
//      CMP Xi, Xend
//      B.GE exit                (placeholder, patched later)
//      <body>
//    endfor:
//      ADD Xi, Xi, Xstep
//      B loop_top
//    exit:
//
//    Uses frame-local storage for loop state.
// ============================================================
.global parse_for
parse_for:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #64             // local frame:
                                    //   [sp+0]  w: counter reg
                                    //   [sp+4]  w: start reg
                                    //   [sp+8]  w: end reg
                                    //   [sp+12] w: step reg
                                    //   [sp+16] x: loop_top address
                                    //   [sp+24] x: patch address (B.GE)
                                    //   [sp+32] w: loop_stack_idx

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'for' token

    // --- Parse loop variable name ---
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error

    // Allocate register for loop counter
    bl      alloc_reg
    mov     w10, w0
    str     w10, [sp, #0]

    // Add loop variable to symbol table
    mov     w1, #KIND_LOCAL_REG
    mov     w2, w10
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add
    add     x19, x19, #TOK_STRIDE_SZ   // skip variable name token

    // --- Parse start expression ---
    bl      parse_expr
    str     w0, [sp, #4]               // start reg

    // --- Parse end expression ---
    bl      parse_expr
    str     w0, [sp, #8]               // end reg

    // --- Parse step expression (optional, default 1) ---
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lfor_default_step
    cmp     w0, #TOK_EOF
    b.eq    .Lfor_default_step
    cmp     w0, #TOK_INDENT
    b.eq    .Lfor_default_step

    bl      parse_expr
    str     w0, [sp, #12]
    b       .Lfor_emit_init

.Lfor_default_step:
    // Allocate register, emit MOV Xstep, #1
    bl      alloc_reg
    str     w0, [sp, #12]
    mov     x1, #1
    bl      emit_mov_imm64

.Lfor_emit_init:
    // Push loop entry onto loop_stack
    adrp    x0, ctrl_loop_sp
    add     x0, x0, :lo12:ctrl_loop_sp
    ldr     w1, [x0]
    cmp     w1, #MAX_LOOP_DEPTH
    b.ge    parse_error                 // too many nested loops
    str     w1, [sp, #32]              // save index

    // --- Emit: MOV Xcounter, Xstart ---
    // ORR Xd, XZR, Xm  (MOV alias: 0xAA000000)
    ldr     w10, [sp, #0]              // counter
    ldr     w11, [sp, #4]              // start
    lsl     w2, w11, #16               // Rm field
    mov     w3, #31
    lsl     w3, w3, #5                 // Rn = XZR
    orr     w4, w10, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    bl      emit32

    // --- loop_top: record address ---
    bl      emit_cur
    str     x0, [sp, #16]             // loop_top

    // --- Emit: CMP Xcounter, Xend ---
    ldr     w10, [sp, #0]
    ldr     w12, [sp, #8]
    mov     w0, w10
    mov     w1, w12
    bl      emit_cmp_reg

    // --- Emit: B.GE exit (placeholder) ---
    bl      emit_cur
    str     x0, [sp, #24]             // patch address
    mov     w0, #CC_GE
    mov     x1, #0                     // placeholder target
    bl      emit_b_cond

    // --- Store into loop_stack ---
    adrp    x0, ctrl_loop_stack
    add     x0, x0, :lo12:ctrl_loop_stack
    ldr     w1, [sp, #32]
    mov     w2, #LOOP_ENTRY_SZ
    madd    x0, x1, x2, x0            // x0 = &ctrl_loop_stack[idx]
    ldr     x3, [sp, #16]
    str     x3, [x0, #0]              // loop_top_addr
    ldr     x3, [sp, #24]
    str     x3, [x0, #8]              // patch_addr
    ldr     w3, [sp, #0]
    str     w3, [x0, #16]             // counter_reg
    ldr     w3, [sp, #8]
    str     w3, [x0, #20]             // end_reg
    ldr     w3, [sp, #12]
    str     w3, [x0, #24]             // step_reg

    // Increment loop_sp
    adrp    x0, ctrl_loop_sp
    add     x0, x0, :lo12:ctrl_loop_sp
    ldr     w1, [x0]
    add     w1, w1, #1
    str     w1, [x0]

    // --- Parse body ---
    bl      skip_newlines
    mov     w9, #4                     // default indent
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lfor_body
    ldr     w9, [x19, #8]
.Lfor_body:
    bl      parse_body

    // --- At endfor: skip optional 'endfor' token ---
    ldr     w0, [x19]
    cmp     w0, #TOK_ENDFOR
    b.ne    .Lfor_no_endfor
    add     x19, x19, #TOK_STRIDE_SZ
.Lfor_no_endfor:

    // --- Emit: ADD Xcounter, Xcounter, Xstep ---
    ldr     w10, [sp, #0]
    ldr     w13, [sp, #12]
    mov     w0, w10
    mov     w1, w10
    mov     w2, w13
    bl      emit_add_reg

    // --- Emit: B loop_top ---
    ldr     x0, [sp, #16]
    bl      emit_b

    // --- Patch B.GE to here (exit) ---
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #24]
    bl      patch_b_cond

    // Pop loop_stack
    adrp    x0, ctrl_loop_sp
    add     x0, x0, :lo12:ctrl_loop_sp
    ldr     w1, [x0]
    sub     w1, w1, #1
    str     w1, [x0]

    add     sp, sp, #64
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 2. parse_if_eq — if== a b
//
//    Emits:
//      CMP Ra, Rb
//      B.NE skip             (placeholder, patched later)
//      <body>
//    skip:
//      [optional else block]
// ============================================================
.global parse_if_eq
parse_if_eq:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'if' token
    // Skip the '==' operator token that follows
    ldr     w0, [x19]
    cmp     w0, #TOK_EQEQ
    b.ne    .Lif_eq_no_op
    add     x19, x19, #TOK_STRIDE_SZ
.Lif_eq_no_op:

    // Parse two expressions
    bl      parse_expr
    str     w0, [sp, #0]               // reg a
    bl      parse_expr
    str     w0, [sp, #4]               // reg b

    // Emit: CMP Ra, Rb
    ldr     w0, [sp, #0]
    ldr     w1, [sp, #4]
    bl      emit_cmp_reg

    // Emit: B.NE skip (placeholder)
    bl      emit_cur
    str     x0, [sp, #8]               // patch address
    mov     w0, #CC_NE
    mov     x1, #0
    bl      emit_b_cond

    // Parse body
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_eq_body
    ldr     w9, [x19, #8]
.Lif_eq_body:
    bl      parse_body

    // Check for else
    bl      skip_newlines
    ldr     w0, [x19]
    cmp     w0, #TOK_ELSE
    b.eq    .Lif_eq_else

    // No else: patch B.NE to here
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

.Lif_eq_else:
    // Emit: B end (placeholder, skip over else body)
    bl      emit_cur
    str     x0, [sp, #16]              // end-patch address
    mov     x0, #0
    bl      emit_b

    // Patch B.NE to here (start of else)
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    // Parse else body
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'else'
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_eq_else_body
    ldr     w9, [x19, #8]
.Lif_eq_else_body:
    bl      parse_body

    // Patch B end to here
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #16]
    bl      patch_b

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 3. parse_if_ge — if>= a b
//
//    Skip condition: B.LT (skip if a < b)
// ============================================================
.global parse_if_ge
parse_if_ge:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'if'
    // Skip '>=' operator token
    ldr     w0, [x19]
    cmp     w0, #TOK_GTE
    b.ne    .Lif_ge_no_op
    add     x19, x19, #TOK_STRIDE_SZ
.Lif_ge_no_op:

    // Parse two expressions
    bl      parse_expr
    str     w0, [sp, #0]
    bl      parse_expr
    str     w0, [sp, #4]

    // Emit: CMP Ra, Rb
    ldr     w0, [sp, #0]
    ldr     w1, [sp, #4]
    bl      emit_cmp_reg

    // Emit: B.LT skip (placeholder)
    bl      emit_cur
    str     x0, [sp, #8]
    mov     w0, #CC_LT
    mov     x1, #0
    bl      emit_b_cond

    // Parse body
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_ge_body
    ldr     w9, [x19, #8]
.Lif_ge_body:
    bl      parse_body

    // Check for else
    bl      skip_newlines
    ldr     w0, [x19]
    cmp     w0, #TOK_ELSE
    b.eq    .Lif_ge_else

    // No else: patch B.LT to here
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

.Lif_ge_else:
    bl      emit_cur
    str     x0, [sp, #16]
    mov     x0, #0
    bl      emit_b

    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    add     x19, x19, #TOK_STRIDE_SZ
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_ge_else_body
    ldr     w9, [x19, #8]
.Lif_ge_else_body:
    bl      parse_body

    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #16]
    bl      patch_b

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 4. parse_if_lt — if< a b
//
//    Skip condition: B.GE (skip if a >= b)
// ============================================================
.global parse_if_lt
parse_if_lt:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'if'
    // Skip '<' operator token
    ldr     w0, [x19]
    cmp     w0, #TOK_LT
    b.ne    .Lif_lt_no_op
    add     x19, x19, #TOK_STRIDE_SZ
.Lif_lt_no_op:

    // Parse two expressions
    bl      parse_expr
    str     w0, [sp, #0]
    bl      parse_expr
    str     w0, [sp, #4]

    // Emit: CMP Ra, Rb
    ldr     w0, [sp, #0]
    ldr     w1, [sp, #4]
    bl      emit_cmp_reg

    // Emit: B.GE skip (placeholder)
    bl      emit_cur
    str     x0, [sp, #8]
    mov     w0, #CC_GE
    mov     x1, #0
    bl      emit_b_cond

    // Parse body
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_lt_body
    ldr     w9, [x19, #8]
.Lif_lt_body:
    bl      parse_body

    // Check for else
    bl      skip_newlines
    ldr     w0, [x19]
    cmp     w0, #TOK_ELSE
    b.eq    .Lif_lt_else

    // No else: patch B.GE to here
    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

.Lif_lt_else:
    bl      emit_cur
    str     x0, [sp, #16]
    mov     x0, #0
    bl      emit_b

    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #8]
    bl      patch_b_cond

    add     x19, x19, #TOK_STRIDE_SZ
    bl      skip_newlines
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lif_lt_else_body
    ldr     w9, [x19, #8]
.Lif_lt_else_body:
    bl      parse_body

    bl      emit_cur
    mov     x1, x0
    ldr     x0, [sp, #16]
    bl      patch_b

    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 5. parse_mem_load — → width addr
//
//    Parse width token (8, 16, 32, 64), parse addr expression,
//    allocate result register, emit appropriate LDR variant.
//    Returns result reg in w0.
// ============================================================
.global parse_mem_load
parse_mem_load:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16

    add     x19, x19, #TOK_STRIDE_SZ   // skip '→' (TOK_LOAD)

    // Parse width — must be integer literal
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    parse_error
    bl      parse_int_literal
    str     x0, [sp, #0]               // width
    add     x19, x19, #TOK_STRIDE_SZ   // skip width token

    // Parse address expression
    bl      parse_expr
    str     w0, [sp, #8]               // addr reg

    // Allocate result register
    bl      alloc_reg
    mov     w6, w0                      // result reg

    // Dispatch on width
    ldr     w5, [sp, #8]               // addr reg
    ldr     x4, [sp, #0]               // width

    cmp     x4, #8
    b.eq    .Lml2_8
    cmp     x4, #16
    b.eq    .Lml2_16
    cmp     x4, #32
    b.eq    .Lml2_32
    // Default: 64-bit
    mov     w0, w6
    mov     w1, w5
    bl      emit_ldr_x_zero
    b       .Lml2_done

.Lml2_8:
    mov     w0, w6
    mov     w1, w5
    bl      emit_ldrb_zero
    b       .Lml2_done

.Lml2_16:
    mov     w0, w6
    mov     w1, w5
    bl      emit_ldrh_zero
    b       .Lml2_done

.Lml2_32:
    mov     w0, w6
    mov     w1, w5
    bl      emit_ldr_w_zero

.Lml2_done:
    mov     w0, w6                      // return result reg
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 6. parse_mem_store — ← width addr val
//
//    Parse width token, addr expression, value expression,
//    emit appropriate STR variant.
// ============================================================
.global parse_mem_store
parse_mem_store:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32

    add     x19, x19, #TOK_STRIDE_SZ   // skip '←' (TOK_STORE)

    // Parse width
    ldr     w0, [x19]
    cmp     w0, #TOK_INT
    b.ne    parse_error
    bl      parse_int_literal
    str     x0, [sp, #0]               // width
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse address expression
    bl      parse_expr
    str     w0, [sp, #8]               // addr reg

    // Parse value expression
    bl      parse_expr
    str     w0, [sp, #16]              // val reg

    // Dispatch on width
    ldr     x4, [sp, #0]
    ldr     w5, [sp, #8]               // addr
    ldr     w6, [sp, #16]              // val

    cmp     x4, #8
    b.eq    .Lms2_8
    cmp     x4, #16
    b.eq    .Lms2_16
    cmp     x4, #32
    b.eq    .Lms2_32
    // Default: 64-bit
    mov     w0, w6
    mov     w1, w5
    bl      emit_str_x_zero
    b       .Lms2_done

.Lms2_8:
    mov     w0, w6
    mov     w1, w5
    bl      emit_strb_zero
    b       .Lms2_done

.Lms2_16:
    mov     w0, w6
    mov     w1, w5
    bl      emit_strh_zero
    b       .Lms2_done

.Lms2_32:
    mov     w0, w6
    mov     w1, w5
    bl      emit_str_w_zero

.Lms2_done:
    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 7. parse_reg_read — ↑ $N or ↑ $NAME
//
//    Parse register reference. If $N (number), the value IS
//    the hardware register. Emit MOV to capture into a scratch reg.
//    If $NAME, error for now (arch dictionary not yet wired).
//    Returns result reg in w0.
// ============================================================
.global parse_reg_read
parse_reg_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ   // skip '↑' (TOK_REG_READ)

    // Parse the register reference as an expression.
    // For $N forms, parse_expr returns a register holding the value N.
    // The convention: $0-$7 map to ARM64 x0-x7 (syscall regs).
    bl      parse_expr
    mov     w4, w0                      // source register (the hardware reg number)

    // Allocate a scratch register to receive the value
    bl      alloc_reg
    mov     w5, w0                      // destination reg

    // Emit: MOV Xdest, Xsource  (ORR Xd, XZR, Xm)
    lsl     w2, w4, #16                // Rm = source
    mov     w3, #31
    lsl     w3, w3, #5                 // Rn = XZR
    orr     w4, w5, w3
    orr     w4, w4, w2
    ORRIMM  w4, 0xAA000000, w16
    mov     w0, w4
    bl      emit32

    mov     w0, w5                      // return result reg
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 8. parse_reg_write — ↓ $N val
//
//    Parse register number, parse value expression,
//    emit MOV XN, Xval.
// ============================================================
.global parse_reg_write
parse_reg_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ   // skip '↓' (TOK_REG_WRITE)

    // Parse register number (the $N expression resolves to a reg holding N)
    bl      parse_expr
    mov     w4, w0                      // target hardware register

    // Parse value expression
    bl      parse_expr
    mov     w5, w0                      // value register

    // Emit: MOV X<target>, Xvalue  (ORR Xd, XZR, Xm)
    lsl     w1, w5, #16               // Rm = value reg
    mov     w2, #31
    lsl     w2, w2, #5                 // Rn = XZR
    orr     w3, w4, w2
    orr     w3, w3, w1
    ORRIMM  w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 9. parse_trap — trap
//
//    Emit: SVC #0 (0xD4000001)
// ============================================================
.global parse_trap
parse_trap:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ   // skip 'trap' token

    // Emit SVC #0
    MOVI32  w0, ARM64_SVC_0
    bl      emit32

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .bss — Control flow parser state
// ============================================================

.bss
.align 3

// Loop nesting stack: 8 entries of 40 bytes each = 320 bytes
ctrl_loop_stack:    .space (LOOP_ENTRY_SZ * MAX_LOOP_DEPTH)
ctrl_loop_sp:       .space 4
    .align 3
