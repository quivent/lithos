// lithos-parser-v5.s — Fifth independent Lithos stack-language parser
//
// THE KEY INSIGHT: The compiler is a register allocator with a token switch.
// Nothing more.
//
// Data structures (all parser-private):
//   reg_stack[32]      — virtual register stack. reg_sp = index of top.
//   names[64]          — (name_off, name_len, reg_num). n_names = count.
//   compositions[32]   — (name_off, name_len, code_off). n_comps = count.
//   next_scratch        = 9 (X9 is first scratch register; X0-X8 reserved)
//
// Operations:
//   vpush(reg)   : reg_stack[++reg_sp] = reg
//   vpop()       : return reg_stack[reg_sp--]
//   alloc()      : return next_scratch++
//   name_lookup  : linear scan names[], return reg or -1
//   comp_lookup  : linear scan compositions[], return code_off or -1
//
// Exports: code_PARSE_TOKENS (DTC entry), parse_tokens (native entry)
// Reads from: ls_token_buf (via args), ls_source_buf_ptr
// Writes to:  ls_code_buf, ls_code_pos
//
// Build: assemble as lithos-parser-v5.o, link in place of lithos-parser.o
//
// Register conventions (DTC bootstrap):
//   X22 = TOS, X24 = DSP, X26 = IP, X25 = W, X23 = RSP, X20 = HERE
//
// Parser-internal registers (callee-saved):
//   X19 = TOKP   — current token pointer
//   X27 = TOKEND — past last token
//   X28 = SRC    — source buffer base

// ============================================================
// Token type constants (from lithos-lexer.s)
// ============================================================
.equ TOK_EOF,       0
.equ TOK_NEWLINE,   1
.equ TOK_INDENT,    2
.equ TOK_INT,       3
.equ TOK_IDENT,     5
.equ TOK_COLON,     72
.equ TOK_LOAD,      36        // →
.equ TOK_STORE,     37        // ←
.equ TOK_REG_READ,  38        // ↑
.equ TOK_REG_WRITE, 39        // ↓
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_IF,        13
.equ TOK_ELSE,      14
.equ TOK_TRAP,      89

.equ TOK_STRIDE,    12        // bytes per token triple

// ARM64 fixed encodings
.equ ARM64_SVC_0,   0xD4000001
.equ ARM64_RET,     0xD65F03C0
.equ ARM64_NOP,     0xD503201F

// ============================================================
// Externs
// ============================================================
.extern ls_code_buf
.extern ls_code_pos
.extern ls_sym_table
.extern ls_sym_count
.extern ls_data_pos
.extern entry_e_cbnz_fwd

// ============================================================
// DTC macros
// ============================================================
.ifndef NEXT_DEFINED
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.set NEXT_DEFINED, 1
.endif

// ============================================================
// .text
// ============================================================
.text
.align 4

// ============================================================
// emit32 — write w0 to code buffer, advance emit_ptr
// ============================================================
v5_emit32:
    adrp    x1, v5_emit_ptr
    add     x1, x1, :lo12:v5_emit_ptr
    ldr     x2, [x1]
    str     w0, [x2], #4
    str     x2, [x1]
    ret

// ============================================================
// v5_emit_cur — return current emit address in x0
// ============================================================
v5_emit_cur:
    adrp    x0, v5_emit_ptr
    add     x0, x0, :lo12:v5_emit_ptr
    ldr     x0, [x0]
    ret

// ============================================================
// Virtual stack operations
// ============================================================

// vpush — push register number w0 onto virtual stack
v5_vpush:
    adrp    x1, v5_reg_sp
    add     x1, x1, :lo12:v5_reg_sp
    ldr     w2, [x1]
    add     w2, w2, #1
    str     w2, [x1]
    adrp    x3, v5_reg_stack
    add     x3, x3, :lo12:v5_reg_stack
    // w2 is reg_sp index
    add     x4, x3, x2, uxtw #2
    str     w0, [x4]
    ret

// vpop — pop top of virtual stack into w0
v5_vpop:
    adrp    x1, v5_reg_sp
    add     x1, x1, :lo12:v5_reg_sp
    ldr     w2, [x1]
    adrp    x3, v5_reg_stack
    add     x3, x3, :lo12:v5_reg_stack
    add     x4, x3, x2, uxtw #2
    ldr     w0, [x4]
    sub     w2, w2, #1
    str     w2, [x1]
    ret

// v5_alloc — allocate next scratch register, return in w0
v5_alloc:
    adrp    x1, v5_next_scratch
    add     x1, x1, :lo12:v5_next_scratch
    ldr     w0, [x1]
    add     w2, w0, #1
    str     w2, [x1]
    ret

// v5_alloc_reset — reset scratch allocator to X9
v5_alloc_reset:
    adrp    x1, v5_next_scratch
    add     x1, x1, :lo12:v5_next_scratch
    mov     w0, #9
    str     w0, [x1]
    // Also reset virtual stack
    adrp    x1, v5_reg_sp
    add     x1, x1, :lo12:v5_reg_sp
    mov     w0, #-1
    str     w0, [x1]
    ret

// ============================================================
// name_lookup — find name in names table
//   x0 = text ptr, w1 = text len
//   Returns: register number in w0, or -1 if not found
// ============================================================
v5_name_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x20, x0              // save text ptr
    mov     w21, w1              // save text len

    adrp    x2, v5_n_names
    add     x2, x2, :lo12:v5_n_names
    ldr     w3, [x2]             // count
    adrp    x4, v5_names
    add     x4, x4, :lo12:v5_names
    mov     w5, #0               // index
.Lnl_loop:
    cmp     w5, w3
    b.ge    .Lnl_notfound
    // entry: [+0] u32 name_off, [+4] u32 name_len, [+8] u32 reg
    ldr     w6, [x4, #4]        // stored len
    cmp     w6, w21
    b.ne    .Lnl_next
    // Compare bytes
    ldr     w7, [x4]            // stored offset
    add     x7, x28, x7         // stored text ptr
    mov     w8, #0
.Lnl_cmp:
    cmp     w8, w21
    b.ge    .Lnl_found
    ldrb    w9, [x20, x8]
    ldrb    w10, [x7, x8]
    cmp     w9, w10
    b.ne    .Lnl_next
    add     w8, w8, #1
    b       .Lnl_cmp
.Lnl_found:
    ldr     w0, [x4, #8]        // register number
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret
.Lnl_next:
    add     x4, x4, #12
    add     w5, w5, #1
    b       .Lnl_loop
.Lnl_notfound:
    mov     w0, #-1
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// v5_name_add — add name to names table
//   w0 = name offset in source, w1 = name len, w2 = register number
// ============================================================
v5_name_add:
    adrp    x3, v5_n_names
    add     x3, x3, :lo12:v5_n_names
    ldr     w4, [x3]
    adrp    x5, v5_names
    add     x5, x5, :lo12:v5_names
    mov     w6, #12
    mul     w7, w4, w6
    add     x5, x5, x7
    str     w0, [x5]             // name_off
    str     w1, [x5, #4]         // name_len
    str     w2, [x5, #8]         // reg
    add     w4, w4, #1
    str     w4, [x3]
    ret

// ============================================================
// comp_lookup — find composition in comp table
//   x0 = text ptr, w1 = text len
//   Returns: code offset in x0, or -1 if not found
// ============================================================
v5_comp_lookup:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x20, x0
    mov     w21, w1

    adrp    x2, v5_n_comps
    add     x2, x2, :lo12:v5_n_comps
    ldr     w3, [x2]
    adrp    x4, v5_comps
    add     x4, x4, :lo12:v5_comps
    mov     w5, #0
.Lcl_loop:
    cmp     w5, w3
    b.ge    .Lcl_notfound
    ldr     w6, [x4, #4]        // stored len
    cmp     w6, w21
    b.ne    .Lcl_next
    ldr     w7, [x4]            // stored offset
    add     x7, x28, x7
    mov     w8, #0
.Lcl_cmp:
    cmp     w8, w21
    b.ge    .Lcl_found
    ldrb    w9, [x20, x8]
    ldrb    w10, [x7, x8]
    cmp     w9, w10
    b.ne    .Lcl_next
    add     w8, w8, #1
    b       .Lcl_cmp
.Lcl_found:
    ldr     w0, [x4, #8]        // code offset
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret
.Lcl_next:
    add     x4, x4, #16         // stride 16 for comps
    add     w5, w5, #1
    b       .Lcl_loop
.Lcl_notfound:
    mov     x0, #-1
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// v5_comp_add — register a composition
//   w0 = name offset, w1 = name len, x2 = code address
// ============================================================
v5_comp_add:
    adrp    x3, v5_n_comps
    add     x3, x3, :lo12:v5_n_comps
    ldr     w4, [x3]
    adrp    x5, v5_comps
    add     x5, x5, :lo12:v5_comps
    lsl     w6, w4, #4           // *16
    add     x5, x5, x6
    str     w0, [x5]             // name_off
    str     w1, [x5, #4]         // name_len
    str     x2, [x5, #8]        // code addr (64-bit)
    add     w4, w4, #1
    str     w4, [x3]
    ret

// ============================================================
// v5_parse_int — parse integer from current token text
//   Returns value in x0. Does NOT advance token pointer.
// ============================================================
v5_parse_int:
    stp     x30, x19, [sp, #-16]!

    ldr     w1, [x19, #4]       // offset
    ldr     w2, [x19, #8]       // length
    add     x1, x28, x1         // text ptr

    mov     x0, #0               // accumulator
    mov     w3, #0               // index
    mov     w7, #0               // negative flag

    // Check leading '-'
    ldrb    w4, [x1]
    cmp     w4, #'-'
    b.ne    .Lv5_hex_check
    mov     w7, #1
    add     w3, w3, #1

.Lv5_hex_check:
    sub     w5, w2, w3
    cmp     w5, #2
    b.lt    .Lv5_dec
    ldrb    w4, [x1, x3]
    cmp     w4, #'0'
    b.ne    .Lv5_dec
    add     w6, w3, #1
    ldrb    w4, [x1, x6]
    cmp     w4, #'x'
    b.eq    .Lv5_hex
    cmp     w4, #'X'
    b.ne    .Lv5_dec

.Lv5_hex:
    add     w3, w3, #2
.Lv5_hex_loop:
    cmp     w3, w2
    b.ge    .Lv5_sign
    ldrb    w4, [x1, x3]
    cmp     w4, #'9'
    b.le    .Lv5_hex_09
    cmp     w4, #'F'
    b.le    .Lv5_hex_AF
    sub     w4, w4, #'a'
    add     w4, w4, #10
    b       .Lv5_hex_acc
.Lv5_hex_AF:
    sub     w4, w4, #'A'
    add     w4, w4, #10
    b       .Lv5_hex_acc
.Lv5_hex_09:
    sub     w4, w4, #'0'
.Lv5_hex_acc:
    lsl     x0, x0, #4
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv5_hex_loop

.Lv5_dec:
.Lv5_dec_loop:
    cmp     w3, w2
    b.ge    .Lv5_sign
    ldrb    w4, [x1, x3]
    sub     w4, w4, #'0'
    mov     x5, #10
    mul     x0, x0, x5
    add     x0, x0, x4
    add     w3, w3, #1
    b       .Lv5_dec_loop

.Lv5_sign:
    cbz     w7, .Lv5_int_done
    neg     x0, x0
.Lv5_int_done:
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// v5_emit_mov_imm — emit MOVZ Xd, #imm16 (+ MOVK if needed)
//   w0 = dest reg, x1 = immediate value (up to 64 bits)
// ============================================================
v5_emit_mov_imm:
    stp     x29, x30, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     w20, w0              // dest reg
    mov     x21, x1              // imm value

    // MOVZ Xd, #(imm & 0xFFFF)
    // Encoding: 0xD2800000 | (imm16 << 5) | Rd
    and     w2, w21, #0xFFFF
    lsl     w2, w2, #5           // imm16 field [20:5]
    orr     w0, w20, w2          // Rd | imm16
    mov     w3, #0xD280
    lsl     w3, w3, #16          // 0xD2800000
    orr     w0, w0, w3
    bl      v5_emit32

    // MOVK Xd, #((imm>>16) & 0xFFFF), LSL #16
    // Encoding: 0xF2A00000 | (imm16 << 5) | Rd   (hw=1)
    lsr     x2, x21, #16
    and     w2, w2, #0xFFFF
    cbz     w2, .Lv5_mi_check32
    lsl     w2, w2, #5
    orr     w0, w20, w2
    mov     w3, #0xF2A0
    lsl     w3, w3, #16
    orr     w0, w0, w3
    bl      v5_emit32

.Lv5_mi_check32:
    // MOVK Xd, #((imm>>32) & 0xFFFF), LSL #32
    // Encoding: 0xF2C00000 | (imm16 << 5) | Rd   (hw=2)
    lsr     x2, x21, #32
    and     w2, w2, #0xFFFF
    cbz     w2, .Lv5_mi_check48
    lsl     w2, w2, #5
    orr     w0, w20, w2
    mov     w3, #0xF2C0
    lsl     w3, w3, #16
    orr     w0, w0, w3
    bl      v5_emit32

.Lv5_mi_check48:
    // MOVK Xd, #((imm>>48) & 0xFFFF), LSL #48
    // Encoding: 0xF2E00000 | (imm16 << 5) | Rd   (hw=3)
    lsr     x2, x21, #48
    and     w2, w2, #0xFFFF
    cbz     w2, .Lv5_mi_done
    lsl     w2, w2, #5
    orr     w0, w20, w2
    mov     w3, #0xF2E0
    lsl     w3, w3, #16
    orr     w0, w0, w3
    bl      v5_emit32

.Lv5_mi_done:
    ldp     x20, x21, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v5_emit_mov_reg — emit MOV Xd, Xm  (ORR Xd, XZR, Xm)
//   w0 = Rd, w1 = Rm
// ============================================================
v5_emit_mov_reg:
    stp     x30, xzr, [sp, #-16]!
    // ORR Xd, XZR, Xm = 0xAA0003E0 | Rd | (Rm << 16)
    lsl     w2, w1, #16          // Rm field [20:16]
    orr     w0, w0, w2           // Rd | Rm
    orr     w0, w0, #(31 << 5)  // Rn = XZR [9:5]
    // Now set opcode: bits [31:21] = 0b10101010000 = 0xAA0 << 21
    // But we can't use movk since it clobbers Rm.
    // Build full word: 0xAA000000 | rest
    // Use a temp approach: build in w3, then emit
    mov     w3, #0xAA00
    lsl     w3, w3, #16          // 0xAA000000
    orr     w0, w0, w3
    bl      v5_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// v5_emit_arith — emit arithmetic: ADD/SUB/MUL/SDIV Xd, Xn, Xm
//   w0 = Rd, w1 = Rn, w2 = Rm, w3 = op (TOK_PLUS/MINUS/STAR/SLASH)
// ============================================================
v5_emit_arith:
    stp     x30, xzr, [sp, #-16]!

    // Build: Rd | (Rn << 5) | (Rm << 16)
    lsl     w4, w1, #5           // Rn field
    orr     w5, w0, w4
    lsl     w4, w2, #16          // Rm field
    orr     w5, w5, w4

    cmp     w3, #TOK_PLUS
    b.eq    .Lv5_arith_add
    cmp     w3, #TOK_MINUS
    b.eq    .Lv5_arith_sub
    cmp     w3, #TOK_STAR
    b.eq    .Lv5_arith_mul
    // default: SDIV = 0x9AC00C00 | fields
    mov     w6, #0x0C00
    movk    w6, #0x9AC0, lsl #16
    orr     w5, w5, w6
    b       .Lv5_arith_emit
.Lv5_arith_add:
    // ADD X = 0x8B000000 | fields
    mov     w6, #0x8B00
    lsl     w6, w6, #16
    orr     w5, w5, w6
    b       .Lv5_arith_emit
.Lv5_arith_sub:
    // SUB X = 0xCB000000 | fields
    mov     w6, #0xCB00
    lsl     w6, w6, #16
    orr     w5, w5, w6
    b       .Lv5_arith_emit
.Lv5_arith_mul:
    // MADD Xd, Xn, Xm, XZR  = 0x9B007C00 | Rd | (Rn<<5) | (Rm<<16)
    mov     w6, #0x7C00
    movk    w6, #0x9B00, lsl #16
    orr     w5, w5, w6
    b       .Lv5_arith_emit
.Lv5_arith_emit:
    mov     w0, w5
    bl      v5_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// v5_emit_stp_prologue — STP X29, X30, [SP, #-16]! ; MOV X29, SP
// ============================================================
v5_emit_prologue:
    stp     x30, xzr, [sp, #-16]!
    // STP X29, X30, [SP, #-16]!  = 0xA9BF7BFD
    mov     w0, #0x7BFD
    movk    w0, #0xA9BF, lsl #16
    bl      v5_emit32
    // MOV X29, SP  =  ADD X29, SP, #0
    mov     w0, #0x03FD
    movk    w0, #0x9100, lsl #16  // 910003FD
    bl      v5_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// v5_emit_epilogue — LDP X29, X30, [SP], #16 ; RET
// ============================================================
v5_emit_epilogue:
    stp     x30, xzr, [sp, #-16]!
    // LDP X29, X30, [SP], #16  = A8C17BFD
    mov     w0, #0x7BFD
    movk    w0, #0xA8C1, lsl #16
    bl      v5_emit32
    // RET = 0xD65F03C0
    mov     w0, #0x03C0
    movk    w0, #0xD65F, lsl #16
    bl      v5_emit32
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// v5_tok_advance — advance token pointer by one triple
// ============================================================
v5_tok_advance:
    add     x19, x19, #TOK_STRIDE
    ret

// ============================================================
// v5_skip_ws — skip NEWLINE and INDENT tokens
// ============================================================
v5_skip_ws:
.Lv5_ws:
    cmp     x19, x27
    b.hs    .Lv5_ws_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv5_ws_skip
    cmp     w0, #TOK_INDENT
    b.eq    .Lv5_ws_skip
    ret
.Lv5_ws_skip:
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_ws
.Lv5_ws_done:
    ret

// ============================================================
// v5_tok_is_eol — check if current token is EOL/EOF/INDENT
//   Returns: w0 = 1 if EOL-ish, 0 if not
// ============================================================
v5_tok_is_eol:
    cmp     x19, x27
    b.hs    .Lv5_eol_yes
    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv5_eol_yes
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv5_eol_yes
    mov     w0, #0
    ret
.Lv5_eol_yes:
    mov     w0, #1
    ret

// ============================================================
// v5_parse_dollar_reg — parse $N token, return register number in w0
//   Current token must be TOK_IDENT starting with '$'.
//   Advances token pointer.
// ============================================================
v5_parse_dollar_reg:
    ldr     w1, [x19, #4]       // offset
    ldr     w2, [x19, #8]       // length
    add     x3, x28, x1         // text ptr

    // Skip '$'
    mov     x0, #0
    mov     w4, #1               // start at index 1
.Lv5_dr_loop:
    cmp     w4, w2
    b.ge    .Lv5_dr_done
    ldrb    w5, [x3, x4]
    sub     w5, w5, #'0'
    mov     x6, #10
    mul     x0, x0, x6
    add     x0, x0, x5
    add     w4, w4, #1
    b       .Lv5_dr_loop
.Lv5_dr_done:
    add     x19, x19, #TOK_STRIDE
    ret

// ============================================================
// v5_tok_text_match — check current token text against string
//   x0 = string ptr, w1 = string len
//   Returns: w0 = 1 if match, 0 if not. Does NOT advance.
// ============================================================
v5_tok_text_match:
    ldr     w2, [x19, #8]       // token length
    cmp     w2, w1
    b.ne    .Lv5_tm_no
    ldr     w3, [x19, #4]       // token offset
    add     x3, x28, x3         // token text ptr
    mov     w4, #0
.Lv5_tm_loop:
    cmp     w4, w1
    b.ge    .Lv5_tm_yes
    ldrb    w5, [x0, x4]
    ldrb    w6, [x3, x4]
    cmp     w5, w6
    b.ne    .Lv5_tm_no
    add     w4, w4, #1
    b       .Lv5_tm_loop
.Lv5_tm_yes:
    mov     w0, #1
    ret
.Lv5_tm_no:
    mov     w0, #0
    ret

// ============================================================
// v5_is_dollar — check if current TOK_IDENT starts with '$'
//   Returns w0 = 1 if yes, 0 if no
// ============================================================
v5_is_dollar:
    ldr     w1, [x19, #4]       // offset
    add     x1, x28, x1
    ldrb    w0, [x1]
    cmp     w0, #'$'
    b.eq    .Lv5_isdol_y
    mov     w0, #0
    ret
.Lv5_isdol_y:
    mov     w0, #1
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

    mov     x19, x0              // TOKP
    mov     w3, #TOK_STRIDE
    mul     x27, x1, x3
    add     x27, x27, x0         // TOKEND
    mov     x28, x2              // SRC

    // Initialize emit pointer
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v5_emit_ptr
    add     x1, x1, :lo12:v5_emit_ptr
    str     x0, [x1]

    // Clear names table
    adrp    x0, v5_n_names
    add     x0, x0, :lo12:v5_n_names
    str     wzr, [x0]

    // Clear compositions table
    adrp    x0, v5_n_comps
    add     x0, x0, :lo12:v5_n_comps
    str     wzr, [x0]

    // Reset allocator
    bl      v5_alloc_reset

    // Main loop: parse top-level items
.Lv5_top:
    cmp     x19, x27
    b.hs    .Lv5_top_done
    bl      v5_skip_ws
    cmp     x19, x27
    b.hs    .Lv5_top_done
    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv5_top_done

    // Only handle: IDENT (composition start) at top level
    cmp     w0, #TOK_IDENT
    b.eq    .Lv5_top_ident

    // Skip unknown top-level token
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_top

.Lv5_top_ident:
    // Lookahead: scan for COLON on this line
    mov     x4, x19
.Lv5_top_scan:
    add     x4, x4, #TOK_STRIDE
    cmp     x4, x27
    b.hs    .Lv5_top_skip
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lv5_top_comp
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv5_top_skip
    cmp     w0, #TOK_EOF
    b.eq    .Lv5_top_skip
    b       .Lv5_top_scan

.Lv5_top_skip:
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_top

.Lv5_top_comp:
    bl      v5_parse_composition
    b       .Lv5_top

.Lv5_top_done:
    // Sync ls_code_pos
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, v5_emit_ptr
    add     x1, x1, :lo12:v5_emit_ptr
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
// v5_parse_composition — parse: name args... :
//   Then indented body lines until dedent.
// ============================================================
v5_parse_composition:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x20, x21, [sp, #-16]!

    // Save current names count for scope cleanup
    adrp    x0, v5_n_names
    add     x0, x0, :lo12:v5_n_names
    ldr     w20, [x0]            // saved name count

    // Record composition: name -> current code address
    bl      v5_emit_cur
    mov     x21, x0              // code address of this composition

    // Register composition name
    ldr     w0, [x19, #4]        // name offset
    ldr     w1, [x19, #8]        // name len
    mov     x2, x21
    bl      v5_comp_add

    add     x19, x19, #TOK_STRIDE  // skip name

    // Emit prologue
    bl      v5_emit_prologue

    // Reset register allocator for this composition
    bl      v5_alloc_reset

    // Parse args (idents before colon) — bind to X0, X1, ...
    mov     w10, #0               // arg index
.Lv5_comp_args:
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lv5_comp_colon
    cmp     w0, #TOK_IDENT
    b.ne    .Lv5_comp_colon       // safety: stop at non-ident

    // Bind arg name to register w10
    ldr     w0, [x19, #4]        // name offset
    ldr     w1, [x19, #8]        // name len
    mov     w2, w10               // register = param index
    bl      v5_name_add

    add     w10, w10, #1
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_comp_args

.Lv5_comp_colon:
    add     x19, x19, #TOK_STRIDE  // skip ':'
    bl      v5_skip_ws

    // Determine body indent level
    mov     w9, #4                // default
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lv5_comp_body
    ldr     w9, [x19, #8]        // indent level from token

.Lv5_comp_body:
    // Parse body lines
    bl      v5_parse_body

    // Emit epilogue
    bl      v5_emit_epilogue

    // Restore names count (pop scope)
    adrp    x0, v5_n_names
    add     x0, x0, :lo12:v5_n_names
    str     w20, [x0]

    ldp     x20, x21, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v5_parse_body — parse indented body lines
//   w9 = minimum indent level
// ============================================================
v5_parse_body:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    str     w9, [sp, #-16]!      // save indent level

.Lv5_body_loop:
    cmp     x19, x27
    b.hs    .Lv5_body_done
    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lv5_body_done
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv5_body_skip
    cmp     w0, #TOK_INDENT
    b.ne    .Lv5_body_stmt

    // Check indent level
    ldr     w1, [x19, #8]
    ldr     w9, [sp, #-16]
    cmp     w1, w9
    b.lt    .Lv5_body_done
    add     x19, x19, #TOK_STRIDE  // consume indent
    b       .Lv5_body_loop

.Lv5_body_skip:
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_body_loop

.Lv5_body_stmt:
    bl      v5_parse_line
    bl      v5_alloc_reset       // reset scratch between lines
    b       .Lv5_body_loop

.Lv5_body_done:
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v5_parse_line — parse one body line (the token switch)
//
// This is the core. Read first token, switch on type/text,
// emit ARM64, manipulate virtual stack. Done. Next line.
//
// Uses a 48-byte local frame for saving values across calls.
// sp+0  = slot0, sp+8 = slot1, sp+16 = slot2,
// sp+24 = slot3, sp+32 = slot4, sp+40 = slot5
// ============================================================
.equ FRAME_SZ, 64    // 48 for slots + 16 for x29/x30
v5_parse_line:
    stp     x29, x30, [sp, #-FRAME_SZ]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lv5_line_reg_write
    cmp     w0, #TOK_REG_READ
    b.eq    .Lv5_line_reg_read
    cmp     w0, #TOK_LOAD
    b.eq    .Lv5_line_load
    cmp     w0, #TOK_STORE
    b.eq    .Lv5_line_store
    cmp     w0, #TOK_PLUS
    b.eq    .Lv5_line_binop
    cmp     w0, #TOK_MINUS
    b.eq    .Lv5_line_binop
    cmp     w0, #TOK_STAR
    b.eq    .Lv5_line_binop
    cmp     w0, #TOK_SLASH
    b.eq    .Lv5_line_binop
    cmp     w0, #TOK_IF
    b.eq    .Lv5_line_if
    cmp     w0, #TOK_TRAP
    b.eq    .Lv5_line_trap
    cmp     w0, #TOK_INT
    b.eq    .Lv5_line_int
    cmp     w0, #TOK_IDENT
    b.eq    .Lv5_line_ident

    // unknown: skip
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_line_ret

.Lv5_line_ret:
    ldp     x29, x30, [sp], #FRAME_SZ
    ret

// ---- Register write: ↓ $N val ----
.Lv5_line_reg_write:
    add     x19, x19, #TOK_STRIDE  // skip ↓
    bl      v5_parse_operand     // target reg
    str     w0, [x29, #16]      // slot0 = target reg
    bl      v5_parse_operand     // value reg
    mov     w1, w0              // Rm = value
    ldr     w0, [x29, #16]      // Rd = target
    bl      v5_emit_mov_reg
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Register read: ↑ $N ----
.Lv5_line_reg_read:
    add     x19, x19, #TOK_STRIDE  // skip ↑
    bl      v5_parse_operand     // source reg
    str     w0, [x29, #16]      // slot0 = source
    bl      v5_alloc             // dest reg
    str     w0, [x29, #24]      // slot1 = dest
    ldr     w1, [x29, #16]      // Rm = source
    bl      v5_emit_mov_reg      // MOV Xdest, Xsource
    ldr     w0, [x29, #24]      // push dest
    bl      v5_vpush
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Load: → width addr ----
.Lv5_line_load:
    add     x19, x19, #TOK_STRIDE
    bl      v5_parse_operand     // width (register holding width)
    str     w0, [x29, #16]      // slot0 = width reg (unused for now)
    bl      v5_parse_operand     // addr reg
    str     w0, [x29, #24]      // slot1 = addr reg
    bl      v5_alloc             // dest reg
    str     w0, [x29, #32]      // slot2 = dest reg
    // Emit LDR Xdest, [Xaddr]
    ldr     w1, [x29, #24]      // addr
    lsl     w1, w1, #5          // Rn field
    ldr     w0, [x29, #32]      // dest
    orr     w0, w0, w1
    movk    w0, #0xF940, lsl #16
    bl      v5_emit32
    ldr     w0, [x29, #32]
    bl      v5_vpush
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Store: ← width addr val ----
.Lv5_line_store:
    add     x19, x19, #TOK_STRIDE
    bl      v5_parse_operand     // width reg
    str     w0, [x29, #16]
    bl      v5_parse_operand     // addr reg
    str     w0, [x29, #24]
    bl      v5_parse_operand     // value reg
    str     w0, [x29, #32]
    // Emit STR Xval, [Xaddr]
    ldr     w1, [x29, #24]
    lsl     w1, w1, #5
    ldr     w0, [x29, #32]
    orr     w0, w0, w1
    movk    w0, #0xF900, lsl #16
    bl      v5_emit32
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Binary op: + - * / [operand] ----
.Lv5_line_binop:
    str     w0, [x29, #16]      // slot0 = op type
    add     x19, x19, #TOK_STRIDE
    bl      v5_tok_is_eol
    cbnz    w0, .Lv5_binop_stack
    // Has operand on line
    bl      v5_parse_operand     // right
    str     w0, [x29, #32]      // slot2 = right
    bl      v5_vpop              // left
    str     w0, [x29, #24]      // slot1 = left
    b       .Lv5_binop_emit
.Lv5_binop_stack:
    bl      v5_vpop              // right
    str     w0, [x29, #32]
    bl      v5_vpop              // left
    str     w0, [x29, #24]
.Lv5_binop_emit:
    bl      v5_alloc             // result reg
    str     w0, [x29, #40]      // slot3 = result
    ldr     w1, [x29, #24]      // Rn = left
    ldr     w2, [x29, #32]      // Rm = right
    ldr     w3, [x29, #16]      // op type
    bl      v5_emit_arith
    ldr     w0, [x29, #40]
    bl      v5_vpush
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Integer literal ----
.Lv5_line_int:
    bl      v5_parse_int
    str     x0, [x29, #16]      // slot0 = parsed value
    add     x19, x19, #TOK_STRIDE
    bl      v5_alloc
    str     w0, [x29, #24]      // slot1 = allocated reg
    ldr     x1, [x29, #16]      // imm value
    bl      v5_emit_mov_imm
    ldr     w0, [x29, #24]
    bl      v5_vpush
    // Check for trailing op
    bl      v5_tok_is_eol
    cbnz    w0, .Lv5_line_int_done
    ldr     w0, [x19]
    cmp     w0, #TOK_PLUS
    b.eq    .Lv5_line_int_trail_op
    cmp     w0, #TOK_MINUS
    b.eq    .Lv5_line_int_trail_op
    cmp     w0, #TOK_STAR
    b.eq    .Lv5_line_int_trail_op
    cmp     w0, #TOK_SLASH
    b.eq    .Lv5_line_int_trail_op
    b       .Lv5_line_int_done
.Lv5_line_int_trail_op:
    str     w0, [x29, #16]      // slot0 = op type
    add     x19, x19, #TOK_STRIDE
    bl      v5_parse_operand     // right
    str     w0, [x29, #32]      // slot2 = right
    bl      v5_vpop              // left
    str     w0, [x29, #24]      // slot1 = left
    bl      v5_alloc
    str     w0, [x29, #40]      // slot3 = result
    ldr     w1, [x29, #24]
    ldr     w2, [x29, #32]
    ldr     w3, [x29, #16]
    bl      v5_emit_arith
    ldr     w0, [x29, #40]
    bl      v5_vpush
.Lv5_line_int_done:
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- Identifier (name, trap, composition call) ----
.Lv5_line_ident:
    // Check "trap"
    adrp    x0, v5_str_trap
    add     x0, x0, :lo12:v5_str_trap
    mov     w1, #4
    bl      v5_tok_text_match
    cbnz    w0, .Lv5_line_trap

    // Check "exit"
    adrp    x0, v5_str_exit
    add     x0, x0, :lo12:v5_str_exit
    mov     w1, #4
    bl      v5_tok_text_match
    cbnz    w0, .Lv5_line_exit_kw

    // Name lookup
    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x0, x28, x1
    mov     w1, w2
    bl      v5_name_lookup
    cmn     w0, #1
    b.eq    .Lv5_line_comp_call

    // Known name: push register
    bl      v5_vpush
    add     x19, x19, #TOK_STRIDE

    // Trailing op?
    bl      v5_tok_is_eol
    cbnz    w0, .Lv5_line_ident_done
    ldr     w0, [x19]
    cmp     w0, #TOK_PLUS
    b.eq    .Lv5_line_ident_trail_op
    cmp     w0, #TOK_MINUS
    b.eq    .Lv5_line_ident_trail_op
    cmp     w0, #TOK_STAR
    b.eq    .Lv5_line_ident_trail_op
    cmp     w0, #TOK_SLASH
    b.eq    .Lv5_line_ident_trail_op
    b       .Lv5_line_ident_done

.Lv5_line_ident_trail_op:
    str     w0, [x29, #16]      // slot0 = op type
    add     x19, x19, #TOK_STRIDE
    bl      v5_parse_operand     // right
    str     w0, [x29, #32]      // slot2 = right
    bl      v5_vpop              // left
    str     w0, [x29, #24]      // slot1 = left
    bl      v5_alloc
    str     w0, [x29, #40]      // slot3 = result
    ldr     w1, [x29, #24]
    ldr     w2, [x29, #32]
    ldr     w3, [x29, #16]
    bl      v5_emit_arith
    ldr     w0, [x29, #40]
    bl      v5_vpush

.Lv5_line_ident_done:
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- trap ----
.Lv5_line_trap:
    add     x19, x19, #TOK_STRIDE
    mov     w0, #0x0001
    movk    w0, #0xD400, lsl #16
    bl      v5_emit32
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- exit keyword ----
.Lv5_line_exit_kw:
    add     x19, x19, #TOK_STRIDE
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- composition call ----
.Lv5_line_comp_call:
    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x0, x28, x1
    mov     w1, w2
    bl      v5_comp_lookup
    str     x0, [x29, #16]      // slot0 = code address
    add     x19, x19, #TOK_STRIDE
    str     wzr, [x29, #24]     // slot1 = arg index (0)
.Lv5_cc_args:
    bl      v5_tok_is_eol
    cbnz    w0, .Lv5_cc_emit
    bl      v5_parse_operand     // result reg
    ldr     w1, [x29, #24]      // arg index
    cmp     w0, w1
    b.eq    .Lv5_cc_arg_ok
    // MOV Xarg, Xresult
    str     w0, [x29, #32]      // save result reg
    mov     w1, w0              // source
    ldr     w0, [x29, #24]      // dest = arg index
    bl      v5_emit_mov_reg
    b       .Lv5_cc_arg_ok
.Lv5_cc_arg_ok:
    ldr     w0, [x29, #24]
    add     w0, w0, #1
    str     w0, [x29, #24]
    b       .Lv5_cc_args
.Lv5_cc_emit:
    ldr     x0, [x29, #16]      // code address
    cmn     x0, #1
    b.eq    .Lv5_cc_unknown
    str     x0, [x29, #32]      // save code addr
    bl      v5_emit_cur
    ldr     x1, [x29, #32]
    sub     x2, x1, x0          // byte offset (target - current)
    asr     x2, x2, #2
    and     w2, w2, #0x3FFFFFF
    movk    w2, #0x9400, lsl #16
    mov     w0, w2
    bl      v5_emit32
    b       .Lv5_cc_done
.Lv5_cc_unknown:
    mov     w0, #0x201F
    movk    w0, #0xD503, lsl #16
    bl      v5_emit32
.Lv5_cc_done:
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ---- if (conditional) ----
.Lv5_line_if:
    add     x19, x19, #TOK_STRIDE
    bl      v5_parse_operand     // a
    str     w0, [x29, #16]      // slot0 = a reg
    bl      v5_parse_operand     // b
    str     w0, [x29, #24]      // slot1 = b reg
    // CMP
    mov     w0, #31
    ldr     w1, [x29, #16]
    lsl     w1, w1, #5
    orr     w0, w0, w1
    ldr     w2, [x29, #24]
    lsl     w2, w2, #16
    orr     w0, w0, w2
    mov     w3, #0xEB00
    lsl     w3, w3, #16
    orr     w0, w0, w3
    bl      v5_emit32
    // B.NE placeholder
    bl      v5_emit_cur
    str     x0, [x29, #32]      // slot2 = patch addr
    mov     w0, #0x0001
    movk    w0, #0x5400, lsl #16
    bl      v5_emit32
    // Body
    bl      v5_skip_ws
    mov     w9, #4
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lv5_if_body
    ldr     w9, [x19, #8]
.Lv5_if_body:
    bl      v5_parse_body
    // Patch
    bl      v5_emit_cur
    ldr     x1, [x29, #32]      // patch addr
    sub     x0, x0, x1
    asr     x0, x0, #2
    and     w0, w0, #0x7FFFF
    lsl     w0, w0, #5
    ldr     w2, [x1]
    and     w2, w2, #0xFF00001F
    orr     w2, w2, w0
    str     w2, [x1]
    bl      v5_skip_to_eol
    b       .Lv5_line_ret

// ============================================================
// v5_parse_operand — parse one operand token, return reg in w0
//
// Handles: $N (register ref), integer literal, named variable.
// This is the workhorse: every operand goes through here.
// ============================================================
v5_parse_operand:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    // Integer literal
    cmp     w0, #TOK_INT
    b.eq    .Lv5_op_int

    // Identifier
    cmp     w0, #TOK_IDENT
    b.eq    .Lv5_op_ident

    // Fallback: treat as 0
    mov     w0, #0
    ldp     x29, x30, [sp], #16
    ret

.Lv5_op_int:
    bl      v5_parse_int
    str     x0, [sp, #-16]!     // save parsed int value
    add     x19, x19, #TOK_STRIDE

    bl      v5_alloc             // allocate scratch reg
    str     w0, [sp, #-16]!     // save allocated reg number
    ldr     x1, [sp, #16]       // retrieve int value
    bl      v5_emit_mov_imm      // emit MOVZ Xreg, #imm
    ldr     w0, [sp], #16       // pop reg number
    add     sp, sp, #16          // drop saved int value
    ldp     x29, x30, [sp], #16
    ret

.Lv5_op_ident:
    // Check if $N
    bl      v5_is_dollar
    cbnz    w0, .Lv5_op_dollar

    // Named variable lookup
    ldr     w1, [x19, #4]
    ldr     w2, [x19, #8]
    add     x0, x28, x1
    mov     w1, w2
    bl      v5_name_lookup
    cmn     w0, #1
    b.eq    .Lv5_op_ident_unknown

    add     x19, x19, #TOK_STRIDE
    // w0 = register number
    ldp     x29, x30, [sp], #16
    ret

.Lv5_op_dollar:
    bl      v5_parse_dollar_reg  // advances token, returns reg in w0
    ldp     x29, x30, [sp], #16
    ret

.Lv5_op_ident_unknown:
    // Unknown ident — allocate reg, load 0
    add     x19, x19, #TOK_STRIDE
    bl      v5_alloc
    mov     w11, w0
    mov     w0, w11
    mov     x1, #0
    bl      v5_emit_mov_imm
    mov     w0, w11
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// v5_skip_to_eol — advance past remaining tokens on this line
// ============================================================
v5_skip_to_eol:
.Lv5_eol:
    cmp     x19, x27
    b.hs    .Lv5_eol_done
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lv5_eol_done
    cmp     w0, #TOK_EOF
    b.eq    .Lv5_eol_done
    cmp     w0, #TOK_INDENT
    b.eq    .Lv5_eol_done
    add     x19, x19, #TOK_STRIDE
    b       .Lv5_eol
.Lv5_eol_done:
    ret

// ============================================================
// DTC wrapper — code_PARSE_TOKENS
// Stack: ( tok-buf tok-count src-buf -- )
// ============================================================
.align 4
.global code_PARSE_TOKENS
code_PARSE_TOKENS:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x20, [sp, #-16]!

    // Pop args from DTC stack
    mov     x2, x22              // TOS = src-buf
    ldr     x1, [x24], #8       // tok-count
    ldr     x0, [x24], #8       // tok-buf
    ldr     x22, [x24], #8      // new TOS

    bl      parse_tokens

    ldp     x22, x20, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// ============================================================
// .data — strings
// ============================================================
.data
.align 3

v5_str_trap:    .ascii "trap"
v5_str_exit:    .ascii "exit"
v5_str_main:    .ascii "main"

// ============================================================
// .bss — parser state
// ============================================================
.extern ls_code_buf
.extern ls_code_pos

.bss
.align 3

v5_emit_ptr:        .quad 0

v5_reg_stack:       .space (32 * 4)      // 32 entries, u32 each
v5_reg_sp:          .space 4             // index of top (-1 = empty, set by v5_alloc_reset)
    .align 3

v5_names:           .space (64 * 12)     // 64 entries: (off, len, reg)
v5_n_names:         .word 0
    .align 3

v5_comps:           .space (32 * 16)     // 32 entries: (off, len, code_addr)
v5_n_comps:         .word 0
    .align 3

v5_next_scratch:    .word 0              // initialized to 9 at runtime by v5_alloc_reset
    .align 3

// ============================================================
// Dictionary entry — extends chain
// ============================================================
.data
.align 3

.global entry_p_parse_tokens
entry_p_parse_tokens:
    .quad   entry_e_cbnz_fwd     // link to tail of emit-arm64.s chain
    .byte   0
    .byte   12
    .ascii  "parse-tokens"
    .align  3
    .quad   code_PARSE_TOKENS

.end
