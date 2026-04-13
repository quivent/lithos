// lithos-glue.s — Safety-net stubs for symbols still undefined after
//                  other workers export existing ones.
//
// Provides the 9 symbols not defined anywhere in the current build:
//   - emit_ldrb_zero, emit_strb_zero, emit_ldrh_zero, emit_strh_zero
//   - emit_ldr_w_zero, emit_str_w_zero, emit_ldr_x_zero, emit_str_x_zero
//   - parse_data_decl
//
// Also notes 6 symbols defined in lithos-parser.s as LOCAL (t/b) that
// should be promoted to GLOBAL for cross-module use:
//   alloc_reg, emit32, patch_b, patch_b_cond, reset_regs, skip_newlines
//   These are handled by updating globals_for in build.sh (scope_depth
//   is .bss b-type, also local).
//
// All zero-offset load/store emitters are self-contained: they write
// directly to ls_code_buf + ls_code_pos without calling emit32 (which
// is local in lithos-parser.o and thus inaccessible from here).
//
// ARM64 encoding reference: docs/arm64-encodings.md
//   LDRB Wt, [Xn]  = 0x39400000 | (Rn<<5) | Rt
//   STRB Wt, [Xn]  = 0x39000000 | (Rn<<5) | Rt
//   LDRH Wt, [Xn]  = 0x79400000 | (Rn<<5) | Rt
//   STRH Wt, [Xn]  = 0x79000000 | (Rn<<5) | Rt
//   LDR  Wt, [Xn]  = 0xB9400000 | (Rn<<5) | Rt
//   STR  Wt, [Xn]  = 0xB9000000 | (Rn<<5) | Rt
//   LDR  Xt, [Xn]  = 0xF9400000 | (Rn<<5) | Rt
//   STR  Xt, [Xn]  = 0xF9000000 | (Rn<<5) | Rt

.extern ls_code_buf
.extern ls_code_pos
.extern parse_buf_decl

// ============================================================
// Global exports
// ============================================================
.global emit_ldrb_zero
.global emit_strb_zero
.global emit_ldrh_zero
.global emit_strh_zero
.global emit_ldr_w_zero
.global emit_str_w_zero
.global emit_ldr_x_zero
.global emit_str_x_zero
.global parse_data_decl

// ============================================================
// Helper macro: inline emit of a 32-bit instruction word.
// w0 = instruction word to emit.
// Clobbers x1, x2, x3. Self-contained (no external calls).
// ============================================================
.macro GLUE_EMIT32
    adrp    x1, ls_code_buf
    add     x1, x1, :lo12:ls_code_buf
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    ldr     x3, [x2]           // x3 = current offset
    str     w0, [x1, x3]       // write 32-bit LE word
    add     x3, x3, #4
    str     x3, [x2]           // advance pos by 4
.endm

// ============================================================
// Helper macro: load a 32-bit immediate into Wd via movz+movk.
// ============================================================
.macro LOADW32 Wd, val
    movz    \Wd, #((\val) >> 16) & 0xFFFF, lsl #16
    movk    \Wd, #(\val) & 0xFFFF
.endm

// ============================================================
// Zero-offset load/store emitters
//
// Convention: w0 = Rt (dest/src register), w1 = Rn (base register)
// Emit: <op> Wt/Xt, [Xn]  (unsigned offset = 0)
// Encoding: base_opcode | (Rn << 5) | Rt
// ============================================================

.text
.align 4

// emit_ldrb_zero — LDRB Wt, [Xn]
emit_ldrb_zero:
    lsl     w4, w1, #5             // Rn << 5
    orr     w0, w0, w4             // Rt | Rn<<5
    LOADW32 w5, 0x39400000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_strb_zero — STRB Wt, [Xn]
emit_strb_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0x39000000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_ldrh_zero — LDRH Wt, [Xn]
emit_ldrh_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0x79400000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_strh_zero — STRH Wt, [Xn]
emit_strh_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0x79000000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_ldr_w_zero — LDR Wt, [Xn]
emit_ldr_w_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0xB9400000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_str_w_zero — STR Wt, [Xn]
emit_str_w_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0xB9000000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_ldr_x_zero — LDR Xt, [Xn]
emit_ldr_x_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0xF9400000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

.align 4
// emit_str_x_zero — STR Xt, [Xn]
emit_str_x_zero:
    lsl     w4, w1, #5
    orr     w0, w0, w4
    LOADW32 w5, 0xF9000000
    orr     w0, w0, w5
    GLUE_EMIT32
    ret

// ============================================================
// parse_data_decl — parse "data name size" declaration
//
// Semantically identical to buf declarations. Delegates to
// parse_buf_decl which handles the "buf name size" pattern.
// ============================================================
.align 4
parse_data_decl:
    b       parse_buf_decl
