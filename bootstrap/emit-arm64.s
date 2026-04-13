// emit-arm64.s — ARM64 machine code emitter for the Lithos self-hosting compiler
//
// Purpose: DTC words that encode ARM64 instructions into a code buffer.
//          Called by the Lithos parser/compiler to emit native ARM64 code.
//          Companion to lithos-bootstrap.s — uses the same register map,
//          macros, and dictionary entry format.
//
// Build: Include after lithos-bootstrap.s dictionary entries, before .end.
//        Or assemble standalone and link with lithos-bootstrap.o.
//
// Register map (inherited from lithos-bootstrap.s):
//   X22 = TOS   (top of data stack, cached)
//   X24 = DSP   (data stack pointer, full descending)
//   X26 = IP    (instruction pointer — next CFA)
//   X25 = W     (working register — current CFA)
//   X23 = RSP   (return stack pointer)
//   X20 = HERE  (dictionary/data-space pointer)
//
// Stack effects are Forth-style: ( inputs -- outputs )
// All emitter words consume operands from the data stack and emit a 32-bit
// instruction word into the code buffer. No outputs unless noted.
//
// Encoding reference: docs/arm64-encodings.md
// Design reference:   compiler/emit-arm64.ls

// ============================================================
// Macros (duplicated from lithos-bootstrap.s for standalone assembly)
// ============================================================

.ifndef LITHOS_MACROS_DEFINED

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

.macro RPUSH reg
    str     \reg, [x23, #-8]!
.endm

.macro RPOP reg
    ldr     \reg, [x23], #8
.endm

.endif

// ============================================================
// CODE BUFFER
// ============================================================
// 1 MB code buffer. code_pos is a byte offset from ls_code_buf.
// emit_word writes a 32-bit LE word at ls_code_buf+code_pos, advances by 4.

.equ CODE_BUF_SIZE, 1048576

// Shared code buffer and cursor live in ls-shared.s
.extern ls_code_buf
.extern ls_code_pos

// ============================================================
// emit_word ( u32 -- )
// Write a 32-bit word to ls_code_buf at code_pos, advance code_pos by 4.
// This is the primitive all emitter words call.
// ============================================================
.text
.align 4
code_EMIT_WORD:
    // TOS = the 32-bit instruction word to emit
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x2, [x1]               // x2 = code_pos
    str     w22, [x0, x2]          // store 32-bit LE word
    add     x2, x2, #4
    str     x2, [x1]               // code_pos += 4
    POP     x22                    // drop TOS, restore from stack
    NEXT

// ============================================================
// code_here ( -- addr )
// Push address of next emission point: ls_code_buf + code_pos
// ============================================================
.align 4
code_CODE_HERE:
    PUSH    x22
    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x2, [x1]
    add     x22, x0, x2
    NEXT

// ============================================================
// code_pos@ ( -- u )
// Push current code_pos (byte offset)
// ============================================================
.align 4
code_CODE_POS:
    PUSH    x22
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]
    NEXT

// ============================================================
// code_reset ( -- )
// Reset code_pos to 0
// ============================================================
.align 4
code_CODE_RESET:
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    str     xzr, [x1]
    NEXT

// ============================================================
// code_buf_addr ( -- addr )
// Push base address of code buffer
// ============================================================
.align 4
code_CODE_BUF:
    PUSH    x22
    adrp    x22, ls_code_buf
    add     x22, x22, :lo12:ls_code_buf
    NEXT

// ============================================================
// Helper: encode and emit
// Many emitter words share the pattern: build a 32-bit word from
// stack operands using ORR/shift, then call emit_word logic inline.
// We inline the emit rather than calling code_EMIT_WORD to avoid
// threading overhead.
// ============================================================

// Inline emit: x0 = 32-bit instruction word to write.
// Clobbers x1, x2. Preserves x22/x24/x26.
.macro EMIT_INLINE
    adrp    x1, ls_code_buf
    add     x1, x1, :lo12:ls_code_buf
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    ldr     x3, [x2]
    str     w0, [x1, x3]
    add     x3, x3, #4
    str     x3, [x2]
.endm

// ============================================================
// 1. DATA PROCESSING — REGISTER
// ============================================================

// emit-add-reg ( rd rn rm -- )
// ADD Xd, Xn, Xm = 0x8B000000 | Rm<<16 | Rn<<5 | Rd
.align 4
code_EMIT_ADD_REG:
    POP     x3                      // x3 = rm (was TOS)
    // Now TOS = rn
    POP     x4                      // x4 = rn (was TOS after first POP)
    // Now TOS = rd ... wait, let me redo the stack logic.
    // Stack: rd rn rm -- with rm = TOS (x22)
    mov     x3, x22                 // x3 = rm
    ldr     x4, [x24], #8          // x4 = rn
    ldr     x22, [x24], #8         // x22 = rd (will be consumed)
    mov     x0, x22                 // x0 = rd
    ldr     x22, [x24], #8         // restore TOS from stack
    mov     x5, #0x8B000000
    orr     x0, x5, x0             // | Rd
    orr     x0, x0, x4, lsl #5    // | Rn<<5
    orr     x0, x0, x3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT

// Generic 3-register encoding helper.
// On entry: x22=rm (TOS), next on stack=rn, then rd.
// x5 = base opcode. Emits base | Rm<<16 | Rn<<5 | Rd.
.macro EMIT_RD_RN_RM base
    mov     x3, x22                 // rm
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    movz    x5, #((\base) >> 48) & 0xFFFF, lsl #48
    movk    x5, #((\base) >> 32) & 0xFFFF, lsl #32
    movk    x5, #((\base) >> 16) & 0xFFFF, lsl #16
    movk    x5, #(\base) & 0xFFFF
    orr     x0, x5, x0             // | Rd
    orr     x0, x0, x4, lsl #5    // | Rn<<5
    orr     x0, x0, x3, lsl #16   // | Rm<<16
    EMIT_INLINE
.endm

// Simpler: load 32-bit base into w5 directly
.macro LOAD_BASE32 reg, val
    movz    \reg, #((\val) >> 16) & 0xFFFF, lsl #16
    movk    \reg, #(\val) & 0xFFFF
.endm

// Load arbitrary 32-bit immediate into a W register (always 2 insns).
.macro LOADW reg, val
    movz    \reg, #((\val) >> 16) & 0xFFFF, lsl #16
    movk    \reg, #(\val) & 0xFFFF
.endm

// 3-register emitter: ( rd rn rm -- )
// TOS=rm, stack has rn then rd below
.macro CODE_3REG base
    mov     x3, x22                 // rm = TOS
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, \base
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT
.endm

// 2-register + imm12 emitter: ( rd rn imm12 -- )
// TOS=imm12, stack has rn then rd
.macro CODE_RD_RN_IMM12 base
    mov     x3, x22                 // imm12 = TOS
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, \base
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w3, lsl #10   // | imm12<<10
    EMIT_INLINE
    NEXT
.endm

// ---- Actual emitter words ----

// emit-add-reg ( rd rn rm -- )  ADD Xd, Xn, Xm
.align 4
code_E_ADD_REG:
    CODE_3REG 0x8B000000

// emit-sub-reg ( rd rn rm -- )  SUB Xd, Xn, Xm
.align 4
code_E_SUB_REG:
    CODE_3REG 0xCB000000

// emit-adds-reg ( rd rn rm -- )  ADDS Xd, Xn, Xm
.align 4
code_E_ADDS_REG:
    CODE_3REG 0xAB000000

// emit-subs-reg ( rd rn rm -- )  SUBS Xd, Xn, Xm
.align 4
code_E_SUBS_REG:
    CODE_3REG 0xEB000000

// emit-mul ( rd rn rm -- )  MUL Xd, Xn, Xm = MADD Xd, Xn, Xm, XZR
// 0x9B007C00 | Rm<<16 | Rn<<5 | Rd
.align 4
code_E_MUL:
    CODE_3REG 0x9B007C00

// emit-sdiv ( rd rn rm -- )  SDIV Xd, Xn, Xm
.align 4
code_E_SDIV:
    CODE_3REG 0x9AC00C00

// emit-udiv ( rd rn rm -- )  UDIV Xd, Xn, Xm
.align 4
code_E_UDIV:
    CODE_3REG 0x9AC00800

// emit-and-reg ( rd rn rm -- )  AND Xd, Xn, Xm
.align 4
code_E_AND_REG:
    CODE_3REG 0x8A000000

// emit-orr-reg ( rd rn rm -- )  ORR Xd, Xn, Xm
.align 4
code_E_ORR_REG:
    CODE_3REG 0xAA000000

// emit-eor-reg ( rd rn rm -- )  EOR Xd, Xn, Xm
.align 4
code_E_EOR_REG:
    CODE_3REG 0xCA000000

// emit-ands-reg ( rd rn rm -- )  ANDS Xd, Xn, Xm
.align 4
code_E_ANDS_REG:
    CODE_3REG 0xEA000000

// emit-lsl-reg ( rd rn rm -- )  LSLV Xd, Xn, Xm
.align 4
code_E_LSL_REG:
    CODE_3REG 0x9AC02000

// emit-lsr-reg ( rd rn rm -- )  LSRV Xd, Xn, Xm
.align 4
code_E_LSR_REG:
    CODE_3REG 0x9AC02400

// emit-asr-reg ( rd rn rm -- )  ASRV Xd, Xn, Xm
.align 4
code_E_ASR_REG:
    CODE_3REG 0x9AC02800

// ============================================================
// MADD / MSUB — 4-register encodings
// ============================================================

// 4-register emitter: ( rd rn rm ra -- )
// TOS=ra, stack has rm, rn, rd
// Encoding: base | Rm<<16 | Ra<<10 | Rn<<5 | Rd
.macro CODE_4REG base
    mov     x6, x22                 // ra = TOS
    ldr     x3, [x24], #8          // rm
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, \base
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w6, lsl #10   // | Ra<<10
    orr     w0, w0, w3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT
.endm

// emit-madd ( rd rn rm ra -- )  MADD Xd, Xn, Xm, Xa
.align 4
code_E_MADD:
    CODE_4REG 0x9B000000

// emit-msub ( rd rn rm ra -- )  MSUB Xd, Xn, Xm, Xa
.align 4
code_E_MSUB:
    CODE_4REG 0x9B008000

// ============================================================
// 2. DATA PROCESSING — IMMEDIATE
// ============================================================

// emit-add-imm ( rd rn imm12 -- )  ADD Xd, Xn, #imm12
.align 4
code_E_ADD_IMM:
    CODE_RD_RN_IMM12 0x91000000

// emit-add-imm-lsl12 ( rd rn imm12 -- )  ADD Xd, Xn, #imm12, LSL #12
.align 4
code_E_ADD_IMM_LSL12:
    CODE_RD_RN_IMM12 0x91400000

// emit-sub-imm ( rd rn imm12 -- )  SUB Xd, Xn, #imm12
.align 4
code_E_SUB_IMM:
    CODE_RD_RN_IMM12 0xD1000000

// emit-adds-imm ( rd rn imm12 -- )  ADDS Xd, Xn, #imm12
.align 4
code_E_ADDS_IMM:
    CODE_RD_RN_IMM12 0xB1000000

// emit-subs-imm ( rd rn imm12 -- )  SUBS Xd, Xn, #imm12
.align 4
code_E_SUBS_IMM:
    CODE_RD_RN_IMM12 0xF1000000

// ============================================================
// ALIASES (CMP, CMN, TST, NEG, MOV, MVN)
// These are thin wrappers that push XZR and call the underlying word.
// Implemented inline for speed.
// ============================================================

// emit-cmp-reg ( rn rm -- )  CMP Xn, Xm = SUBS XZR, Xn, Xm
.align 4
code_E_CMP_REG:
    mov     x3, x22                 // rm = TOS
    ldr     x4, [x24], #8          // rn
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xEB000000
    mov     w0, #31                 // Rd = XZR
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// emit-cmp-imm ( rn imm12 -- )  CMP Xn, #imm12
.align 4
code_E_CMP_IMM:
    mov     x3, x22                 // imm12 = TOS
    ldr     x4, [x24], #8          // rn
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xF1000000
    mov     w0, #31                 // Rd = XZR
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-cmn-imm ( rn imm12 -- )  CMN Xn, #imm12 = ADDS XZR, Xn, #imm12
.align 4
code_E_CMN_IMM:
    mov     x3, x22                 // imm12 = TOS
    ldr     x4, [x24], #8          // rn
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xB1000000
    mov     w0, #31
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-tst-reg ( rn rm -- )  TST Xn, Xm = ANDS XZR, Xn, Xm
.align 4
code_E_TST_REG:
    mov     x3, x22                 // rm
    ldr     x4, [x24], #8          // rn
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xEA000000
    mov     w0, #31
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// emit-neg ( rd rm -- )  NEG Xd, Xm = SUB Xd, XZR, Xm
.align 4
code_E_NEG:
    mov     x3, x22                 // rm
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xCB000000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, #(31 << 5)    // | Rn=XZR<<5
    orr     w0, w0, w3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT

// emit-mov ( rd rm -- )  MOV Xd, Xm = ORR Xd, XZR, Xm
.align 4
code_E_MOV:
    mov     x3, x22                 // rm (source)
    ldr     x0, [x24], #8          // rd (dest)
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xAA000000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, #(31 << 5)    // | Rn=XZR<<5
    orr     w0, w0, w3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT

// emit-mvn ( rd rm -- )  MVN Xd, Xm = ORN Xd, XZR, Xm
.align 4
code_E_MVN:
    mov     x3, x22                 // rm
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xAA200000
    orr     w0, w5, w0
    orr     w0, w0, #(31 << 5)    // Rn = XZR
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// ============================================================
// 3. MOVE WIDE IMMEDIATE
// ============================================================

// emit-movz ( rd imm16 hw -- )  MOVZ Xd, #imm16, LSL #(hw*16)
// 0xD2800000 | hw<<21 | imm16<<5 | Rd
.align 4
code_E_MOVZ:
    mov     x6, x22                 // hw
    ldr     x3, [x24], #8          // imm16
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xD2800000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w3, lsl #5    // | imm16<<5
    orr     w0, w0, w6, lsl #21   // | hw<<21
    EMIT_INLINE
    NEXT

// emit-movk ( rd imm16 hw -- )  MOVK Xd, #imm16, LSL #(hw*16)
// 0xF2800000 | hw<<21 | imm16<<5 | Rd
.align 4
code_E_MOVK:
    mov     x6, x22                 // hw
    ldr     x3, [x24], #8          // imm16
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xF2800000
    orr     w0, w5, w0
    orr     w0, w0, w3, lsl #5
    orr     w0, w0, w6, lsl #21
    EMIT_INLINE
    NEXT

// emit-movn ( rd imm16 hw -- )  MOVN Xd, #imm16, LSL #(hw*16)
// 0x92800000 | hw<<21 | imm16<<5 | Rd
.align 4
code_E_MOVN:
    mov     x6, x22                 // hw
    ldr     x3, [x24], #8          // imm16
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0x92800000
    orr     w0, w5, w0
    orr     w0, w0, w3, lsl #5
    orr     w0, w0, w6, lsl #21
    EMIT_INLINE
    NEXT

// emit-mov64 ( rd imm64 -- )
// Load a full 64-bit immediate using MOVZ + up to 3 MOVKs.
// Emits 1-4 instructions.
.align 4
code_E_MOV64:
    mov     x3, x22                 // imm64
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS

    // chunk0 = imm64[15:0]
    and     x6, x3, #0xFFFF

    // Emit MOVZ Xd, #chunk0, LSL #0
    LOAD_BASE32 w5, 0xD2800000
    orr     w7, w5, w0             // | Rd
    orr     w7, w7, w6, lsl #5    // | imm16<<5
    // hw=0, no shift needed
    mov     w0, w0                  // preserve rd in w0
    // Actually emit
    adrp    x1, ls_code_buf
    add     x1, x1, :lo12:ls_code_buf
    adrp    x2, ls_code_pos
    add     x2, x2, :lo12:ls_code_pos
    ldr     x8, [x2]
    str     w7, [x1, x8]
    add     x8, x8, #4

    // chunk1 = imm64[31:16]
    ubfx    x6, x3, #16, #16
    cbz     x6, .Lmov64_c2
    LOAD_BASE32 w5, 0xF2800000
    orr     w7, w5, w0             // | Rd
    orr     w7, w7, w6, lsl #5    // | imm16<<5
    orr     w7, w7, #(1 << 21)    // | hw=1
    str     w7, [x1, x8]
    add     x8, x8, #4

.Lmov64_c2:
    // chunk2 = imm64[47:32]
    ubfx    x6, x3, #32, #16
    cbz     x6, .Lmov64_c3
    LOAD_BASE32 w5, 0xF2800000
    orr     w7, w5, w0
    orr     w7, w7, w6, lsl #5
    orr     w7, w7, #(2 << 21)    // hw=2
    str     w7, [x1, x8]
    add     x8, x8, #4

.Lmov64_c3:
    // chunk3 = imm64[63:48]
    ubfx    x6, x3, #48, #16
    cbz     x6, .Lmov64_done
    LOAD_BASE32 w5, 0xF2800000
    orr     w7, w5, w0
    orr     w7, w7, w6, lsl #5
    orr     w7, w7, #(3 << 21)    // hw=3
    str     w7, [x1, x8]
    add     x8, x8, #4

.Lmov64_done:
    str     x8, [x2]               // update code_pos
    NEXT

// ============================================================
// 4. SHIFT — IMMEDIATE
// ============================================================

// emit-lsl-imm ( rd rn shift -- )
// LSL Xd, Xn, #shift = UBFM Xd, Xn, #(64-shift), #(63-shift)
// 0xD3400000 | immr<<16 | imms<<10 | Rn<<5 | Rd
.align 4
code_E_LSL_IMM:
    mov     x3, x22                 // shift
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    // immr = (64 - shift) & 63
    mov     x6, #64
    sub     x6, x6, x3
    and     x6, x6, #63
    // imms = 63 - shift
    mov     x7, #63
    sub     x7, x7, x3
    LOAD_BASE32 w5, 0xD3400000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w7, lsl #10   // | imms<<10
    orr     w0, w0, w6, lsl #16   // | immr<<16
    EMIT_INLINE
    NEXT

// emit-lsr-imm ( rd rn shift -- )
// LSR Xd, Xn, #shift = UBFM Xd, Xn, #shift, #63
// 0xD340FC00 | shift<<16 | Rn<<5 | Rd
.align 4
code_E_LSR_IMM:
    mov     x3, x22                 // shift
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xD340FC00
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// emit-asr-imm ( rd rn shift -- )
// ASR Xd, Xn, #shift = SBFM Xd, Xn, #shift, #63
// 0x9340FC00 | shift<<16 | Rn<<5 | Rd
.align 4
code_E_ASR_IMM:
    mov     x3, x22                 // shift
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0x9340FC00
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// ============================================================
// 5. LOADS AND STORES
// ============================================================

// --- 64-bit unsigned offset ---

// emit-ldr ( rt rn imm -- )  LDR Xt, [Xn, #imm]
// 0xF9400000 | (imm/8)<<10 | Rn<<5 | Rt
.align 4
code_E_LDR:
    mov     x3, x22                 // imm (byte offset, must be multiple of 8)
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rt
    ldr     x22, [x24], #8         // restore TOS
    lsr     x3, x3, #3             // imm/8
    LOAD_BASE32 w5, 0xF9400000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-str ( rt rn imm -- )  STR Xt, [Xn, #imm]
// 0xF9000000 | (imm/8)<<10 | Rn<<5 | Rt
.align 4
code_E_STR:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #3
    LOAD_BASE32 w5, 0xF9000000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// --- 64-bit register offset ---

// emit-ldr-reg ( rt rn rm -- )  LDR Xt, [Xn, Xm]
// 0xF8606800 | Rm<<16 | Rn<<5 | Rt
.align 4
code_E_LDR_REG:
    CODE_3REG 0xF8606800

// emit-str-reg ( rt rn rm -- )  STR Xt, [Xn, Xm]
// 0xF8206800 | Rm<<16 | Rn<<5 | Rt
.align 4
code_E_STR_REG:
    CODE_3REG 0xF8206800

// --- Byte loads/stores (unsigned offset) ---

// emit-ldrb ( rt rn imm12 -- )  LDRB Wt, [Xn, #imm12]
// 0x39400000 | imm12<<10 | Rn<<5 | Rt
.align 4
code_E_LDRB:
    CODE_RD_RN_IMM12 0x39400000

// emit-strb ( rt rn imm12 -- )  STRB Wt, [Xn, #imm12]
// 0x39000000 | imm12<<10 | Rn<<5 | Rt
.align 4
code_E_STRB:
    CODE_RD_RN_IMM12 0x39000000

// --- Halfword loads/stores (unsigned offset, scaled by 2) ---

// emit-ldrh ( rt rn imm -- )  LDRH Wt, [Xn, #imm]
// 0x79400000 | (imm/2)<<10 | Rn<<5 | Rt
.align 4
code_E_LDRH:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #1             // imm/2
    LOAD_BASE32 w5, 0x79400000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-strh ( rt rn imm -- )  STRH Wt, [Xn, #imm]
// 0x79000000 | (imm/2)<<10 | Rn<<5 | Rt
.align 4
code_E_STRH:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #1
    LOAD_BASE32 w5, 0x79000000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// --- 32-bit loads/stores (unsigned offset, scaled by 4) ---

// emit-ldr-w ( rt rn imm -- )  LDR Wt, [Xn, #imm]
// 0xB9400000 | (imm/4)<<10 | Rn<<5 | Rt
.align 4
code_E_LDR_W:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #2
    LOAD_BASE32 w5, 0xB9400000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-str-w ( rt rn imm -- )  STR Wt, [Xn, #imm]
// 0xB9000000 | (imm/4)<<10 | Rn<<5 | Rt
.align 4
code_E_STR_W:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #2
    LOAD_BASE32 w5, 0xB9000000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// --- Pre-indexed loads/stores (simm9, writeback) ---

// Helper macro for pre/post indexed: ( rt rn simm9 -- )
// base | (simm9 & 0x1FF)<<12 | Rn<<5 | Rt
.macro CODE_PRE_POST base
    mov     x3, x22                 // simm9
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rt
    ldr     x22, [x24], #8         // restore TOS
    and     x3, x3, #0x1FF         // mask to 9 bits
    LOAD_BASE32 w5, \base
    orr     w0, w5, w0             // | Rt
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w3, lsl #12   // | simm9<<12
    EMIT_INLINE
    NEXT
.endm

// emit-ldr-pre ( rt rn simm9 -- )  LDR Xt, [Xn, #simm9]!
.align 4
code_E_LDR_PRE:
    CODE_PRE_POST 0xF8400C00

// emit-str-pre ( rt rn simm9 -- )  STR Xt, [Xn, #simm9]!
.align 4
code_E_STR_PRE:
    CODE_PRE_POST 0xF8000C00

// emit-ldr-post ( rt rn simm9 -- )  LDR Xt, [Xn], #simm9
.align 4
code_E_LDR_POST:
    CODE_PRE_POST 0xF8400400

// emit-str-post ( rt rn simm9 -- )  STR Xt, [Xn], #simm9
.align 4
code_E_STR_POST:
    CODE_PRE_POST 0xF8000400

// --- Load/Store Pair ---

// emit-ldp ( rt1 rt2 rn imm -- )  LDP Xt1, Xt2, [Xn, #imm]
// 0xA9400000 | (imm/8 & 0x7F)<<15 | Rt2<<10 | Rn<<5 | Rt1
.macro CODE_PAIR base
    mov     x6, x22                 // imm (byte offset)
    ldr     x3, [x24], #8          // rn
    ldr     x4, [x24], #8          // rt2
    ldr     x0, [x24], #8          // rt1
    ldr     x22, [x24], #8         // restore TOS
    asr     x6, x6, #3             // imm/8 (signed)
    and     x6, x6, #0x7F          // mask to 7 bits
    LOAD_BASE32 w5, \base
    orr     w0, w5, w0             // | Rt1
    orr     w0, w0, w4, lsl #10   // | Rt2<<10
    orr     w0, w0, w3, lsl #5    // | Rn<<5
    orr     w0, w0, w6, lsl #15   // | imm7<<15
    EMIT_INLINE
    NEXT
.endm

// emit-ldp ( rt1 rt2 rn imm -- )
.align 4
code_E_LDP:
    CODE_PAIR 0xA9400000

// emit-stp ( rt1 rt2 rn imm -- )
.align 4
code_E_STP:
    CODE_PAIR 0xA9000000

// emit-ldp-pre ( rt1 rt2 rn imm -- )
.align 4
code_E_LDP_PRE:
    CODE_PAIR 0xA9C00000

// emit-stp-pre ( rt1 rt2 rn imm -- )
.align 4
code_E_STP_PRE:
    CODE_PAIR 0xA9800000

// emit-ldp-post ( rt1 rt2 rn imm -- )
.align 4
code_E_LDP_POST:
    CODE_PAIR 0xA8C00000

// emit-stp-post ( rt1 rt2 rn imm -- )
.align 4
code_E_STP_POST:
    CODE_PAIR 0xA8800000

// ============================================================
// 6. BRANCHES
// ============================================================

// emit-b ( offset -- )  B label
// 0x14000000 | (offset/4 & 0x3FFFFFF)
.align 4
code_E_B:
    mov     x0, x22                 // offset (bytes, signed)
    ldr     x22, [x24], #8         // restore TOS
    asr     x0, x0, #2             // offset/4
    and     w0, w0, #0x3FFFFFF     // mask 26 bits
    LOAD_BASE32 w5, 0x14000000
    orr     w0, w5, w0
    EMIT_INLINE
    NEXT

// emit-bl ( offset -- )  BL label
// 0x94000000 | (offset/4 & 0x3FFFFFF)
.align 4
code_E_BL:
    mov     x0, x22
    ldr     x22, [x24], #8
    asr     x0, x0, #2
    and     w0, w0, #0x3FFFFFF
    LOAD_BASE32 w5, 0x94000000
    orr     w0, w5, w0
    EMIT_INLINE
    NEXT

// emit-bcond ( cond offset -- )  B.cond label
// 0x54000000 | ((offset/4 & 0x7FFFF) << 5) | cond
.align 4
code_E_BCOND:
    mov     x3, x22                 // offset
    ldr     x6, [x24], #8          // cond
    ldr     x22, [x24], #8         // restore TOS
    asr     x3, x3, #2             // offset/4
    and     w3, w3, #0x7FFFF       // 19-bit mask
    LOAD_BASE32 w5, 0x54000000
    orr     w0, w5, w6             // | cond
    orr     w0, w0, w3, lsl #5    // | imm19<<5
    EMIT_INLINE
    NEXT

// Condition code constants (for stack use)
.equ COND_EQ, 0
.equ COND_NE, 1
.equ COND_CS, 2
.equ COND_HS, 2
.equ COND_CC, 3
.equ COND_LO, 3
.equ COND_MI, 4
.equ COND_PL, 5
.equ COND_VS, 6
.equ COND_VC, 7
.equ COND_HI, 8
.equ COND_LS, 9
.equ COND_GE, 10
.equ COND_LT, 11
.equ COND_GT, 12
.equ COND_LE, 13
.equ COND_AL, 14

// emit-cbz ( rt offset -- )  CBZ Xt, label
// 0xB4000000 | ((offset/4 & 0x7FFFF) << 5) | Rt
.align 4
code_E_CBZ:
    mov     x3, x22                 // offset
    ldr     x0, [x24], #8          // rt
    ldr     x22, [x24], #8         // restore TOS
    asr     x3, x3, #2
    and     w3, w3, #0x7FFFF
    LOAD_BASE32 w5, 0xB4000000
    orr     w0, w5, w0             // | Rt
    orr     w0, w0, w3, lsl #5    // | imm19<<5
    EMIT_INLINE
    NEXT

// emit-cbnz ( rt offset -- )  CBNZ Xt, label
// 0xB5000000 | ((offset/4 & 0x7FFFF) << 5) | Rt
.align 4
code_E_CBNZ:
    mov     x3, x22
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    asr     x3, x3, #2
    and     w3, w3, #0x7FFFF
    LOAD_BASE32 w5, 0xB5000000
    orr     w0, w5, w0
    orr     w0, w0, w3, lsl #5
    EMIT_INLINE
    NEXT

// emit-tbz ( rt bit offset -- )  TBZ Xt, #bit, label
// 0x36000000 | (b5<<31) | (b40<<19) | ((imm14 & 0x3FFF)<<5) | Rt
.align 4
code_E_TBZ:
    mov     x3, x22                 // offset
    ldr     x6, [x24], #8          // bit
    ldr     x0, [x24], #8          // rt
    ldr     x22, [x24], #8         // restore TOS
    asr     x3, x3, #2
    and     w3, w3, #0x3FFF         // 14-bit offset
    lsr     w7, w6, #5              // b5 = bit[5]
    and     w6, w6, #0x1F           // b40 = bit[4:0]
    LOAD_BASE32 w5, 0x36000000
    orr     w0, w5, w0             // | Rt
    orr     w0, w0, w3, lsl #5    // | imm14<<5
    orr     w0, w0, w6, lsl #19   // | b40<<19
    orr     w0, w0, w7, lsl #31   // | b5<<31
    EMIT_INLINE
    NEXT

// emit-tbnz ( rt bit offset -- )  TBNZ Xt, #bit, label
// 0x37000000 | (b5<<31) | (b40<<19) | ((imm14 & 0x3FFF)<<5) | Rt
.align 4
code_E_TBNZ:
    mov     x3, x22
    ldr     x6, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    asr     x3, x3, #2
    and     w3, w3, #0x3FFF
    lsr     w7, w6, #5
    and     w6, w6, #0x1F
    LOAD_BASE32 w5, 0x37000000
    orr     w0, w5, w0
    orr     w0, w0, w3, lsl #5
    orr     w0, w0, w6, lsl #19
    orr     w0, w0, w7, lsl #31
    EMIT_INLINE
    NEXT

// emit-br ( rn -- )  BR Xn
// 0xD61F0000 | Rn<<5
.align 4
code_E_BR:
    mov     x0, x22
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD61F0000
    orr     w0, w5, w0, lsl #5
    EMIT_INLINE
    NEXT

// emit-blr ( rn -- )  BLR Xn
// 0xD63F0000 | Rn<<5
.align 4
code_E_BLR:
    mov     x0, x22
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD63F0000
    orr     w0, w5, w0, lsl #5
    EMIT_INLINE
    NEXT

// emit-ret ( -- )  RET (to LR/X30)
// 0xD65F03C0
.align 4
code_E_RET:
    LOADW   w0, 0xD65F03C0
    EMIT_INLINE
    NEXT

// emit-ret-reg ( rn -- )  RET Xn
// 0xD65F0000 | Rn<<5
.align 4
code_E_RET_REG:
    mov     x0, x22
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD65F0000
    orr     w0, w5, w0, lsl #5
    EMIT_INLINE
    NEXT

// ============================================================
// 7. CONDITIONAL SELECT
// ============================================================

// emit-csel ( rd rn rm cond -- )  CSEL Xd, Xn, Xm, cond
// 0x9A800000 | Rm<<16 | cond<<12 | Rn<<5 | Rd
.align 4
code_E_CSEL:
    mov     x6, x22                 // cond
    ldr     x3, [x24], #8          // rm
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0x9A800000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w6, lsl #12   // | cond<<12
    orr     w0, w0, w3, lsl #16   // | Rm<<16
    EMIT_INLINE
    NEXT

// emit-csinc ( rd rn rm cond -- )  CSINC Xd, Xn, Xm, cond
// 0x9A800400 | Rm<<16 | cond<<12 | Rn<<5 | Rd
.align 4
code_E_CSINC:
    mov     x6, x22                 // cond
    ldr     x3, [x24], #8          // rm
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0x9A800400
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w6, lsl #12
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// emit-cset ( rd cond -- )  CSET Xd, cond
// = CSINC Xd, XZR, XZR, invert(cond)
// 0x9A9F07E0 | invert(cond)<<12 | Rd
.align 4
code_E_CSET:
    mov     x6, x22                 // cond
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    eor     x6, x6, #1             // invert cond
    LOAD_BASE32 w5, 0x9A9F07E0
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w6, lsl #12   // | inv_cond<<12
    EMIT_INLINE
    NEXT

// emit-csneg ( rd rn rm cond -- )  CSNEG Xd, Xn, Xm, cond
// 0xDA800400 | Rm<<16 | cond<<12 | Rn<<5 | Rd
.align 4
code_E_CSNEG:
    mov     x6, x22
    ldr     x3, [x24], #8
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xDA800400
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w6, lsl #12
    orr     w0, w0, w3, lsl #16
    EMIT_INLINE
    NEXT

// ============================================================
// 8. SYSTEM INSTRUCTIONS
// ============================================================

// emit-svc ( imm16 -- )  SVC #imm16
// 0xD4000001 | imm16<<5
.align 4
code_E_SVC:
    mov     x0, x22
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD4000001
    orr     w0, w5, w0, lsl #5
    EMIT_INLINE
    NEXT

// emit-syscall ( -- )  SVC #0
.align 4
code_E_SYSCALL:
    LOADW   w0, 0xD4000001
    EMIT_INLINE
    NEXT

// emit-nop ( -- )  NOP
.align 4
code_E_NOP:
    LOADW   w0, 0xD503201F
    EMIT_INLINE
    NEXT

// emit-brk ( imm16 -- )  BRK #imm16
// 0xD4200000 | imm16<<5
.align 4
code_E_BRK:
    mov     x0, x22
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD4200000
    orr     w0, w5, w0, lsl #5
    EMIT_INLINE
    NEXT

// emit-mrs ( rt sysreg -- )  MRS Xt, <sysreg>
// 0xD5300000 | sysreg<<5 | Rt
.align 4
code_E_MRS:
    mov     x3, x22                 // sysreg encoding (15-bit)
    ldr     x0, [x24], #8          // rt
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xD5300000
    orr     w0, w5, w0             // | Rt
    orr     w0, w0, w3, lsl #5    // | sysreg<<5
    EMIT_INLINE
    NEXT

// emit-msr ( sysreg rt -- )  MSR <sysreg>, Xt
// 0xD5100000 | sysreg<<5 | Rt
.align 4
code_E_MSR:
    mov     x3, x22                 // rt
    ldr     x6, [x24], #8          // sysreg
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0xD5100000
    orr     w0, w5, w3             // | Rt
    orr     w0, w0, w6, lsl #5    // | sysreg<<5
    EMIT_INLINE
    NEXT

// emit-dsb-ish ( -- )
.align 4
code_E_DSB_ISH:
    LOADW   w0, 0xD5033B9F
    EMIT_INLINE
    NEXT

// emit-dsb-sy ( -- )
.align 4
code_E_DSB_SY:
    LOADW   w0, 0xD5033F9F
    EMIT_INLINE
    NEXT

// emit-dmb-ish ( -- )
.align 4
code_E_DMB_ISH:
    LOADW   w0, 0xD5033BBF
    EMIT_INLINE
    NEXT

// emit-isb ( -- )
.align 4
code_E_ISB:
    LOADW   w0, 0xD5033FDF
    EMIT_INLINE
    NEXT

// ============================================================
// 9. PC-RELATIVE ADDRESS GENERATION
// ============================================================

// emit-adr ( rd offset -- )  ADR Xd, label
// ((offset&3)<<29) | 0x10000000 | ((offset>>2)&0x7FFFF)<<5 | Rd
.align 4
code_E_ADR:
    mov     x3, x22                 // offset
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    and     x6, x3, #3             // immlo = offset & 3
    asr     x7, x3, #2
    and     x7, x7, #0x7FFFF       // immhi = (offset>>2) & 0x7FFFF
    LOAD_BASE32 w5, 0x10000000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w7, lsl #5    // | immhi<<5
    orr     w0, w0, w6, lsl #29   // | immlo<<29
    EMIT_INLINE
    NEXT

// emit-adrp ( rd offset -- )  ADRP Xd, label
// ((offset&3)<<29) | 0x90000000 | ((offset>>2)&0x7FFFF)<<5 | Rd
.align 4
code_E_ADRP:
    mov     x3, x22
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    and     x6, x3, #3
    asr     x7, x3, #2
    and     x7, x7, #0x7FFFF
    LOAD_BASE32 w5, 0x90000000
    orr     w0, w5, w0
    orr     w0, w0, w7, lsl #5
    orr     w0, w0, w6, lsl #29
    EMIT_INLINE
    NEXT

// ============================================================
// 10. LOGICAL IMMEDIATE (raw N/immr/imms encoding)
// ============================================================

// emit-and-imm ( rd rn N immr imms -- )
// 0x92000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
.align 4
code_E_AND_IMM:
    mov     x9, x22                 // imms
    ldr     x8, [x24], #8          // immr
    ldr     x7, [x24], #8          // N
    ldr     x4, [x24], #8          // rn
    ldr     x0, [x24], #8          // rd
    ldr     x22, [x24], #8         // restore TOS
    LOAD_BASE32 w5, 0x92000000
    orr     w0, w5, w0             // | Rd
    orr     w0, w0, w4, lsl #5    // | Rn<<5
    orr     w0, w0, w9, lsl #10   // | imms<<10
    orr     w0, w0, w8, lsl #16   // | immr<<16
    orr     w0, w0, w7, lsl #22   // | N<<22
    EMIT_INLINE
    NEXT

// emit-orr-imm ( rd rn N immr imms -- )
// 0xB2000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
.align 4
code_E_ORR_IMM:
    mov     x9, x22
    ldr     x8, [x24], #8
    ldr     x7, [x24], #8
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xB2000000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w9, lsl #10
    orr     w0, w0, w8, lsl #16
    orr     w0, w0, w7, lsl #22
    EMIT_INLINE
    NEXT

// emit-eor-imm ( rd rn N immr imms -- )
// 0xD2000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
.align 4
code_E_EOR_IMM:
    mov     x9, x22
    ldr     x8, [x24], #8
    ldr     x7, [x24], #8
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    LOAD_BASE32 w5, 0xD2000000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w9, lsl #10
    orr     w0, w0, w8, lsl #16
    orr     w0, w0, w7, lsl #22
    EMIT_INLINE
    NEXT

// ============================================================
// 11. BRANCH PATCHING (forward reference resolution)
// ============================================================

// emit-mark ( -- pos )
// Push current code_pos for later patching.
.align 4
code_E_MARK:
    PUSH    x22
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]
    NEXT

// emit-patch-bcond ( mark_pos -- )
// Patch a B.cond at mark_pos to branch to current code_pos.
.align 4
code_E_PATCH_BCOND:
    mov     x3, x22                 // mark_pos
    ldr     x22, [x24], #8         // restore TOS

    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x2, [x1]               // current code_pos

    sub     x4, x2, x3             // offset = current - mark
    asr     x4, x4, #2             // offset/4
    and     w4, w4, #0x7FFFF       // 19-bit mask
    lsl     w4, w4, #5             // shift into position

    ldr     w5, [x0, x3]           // read existing instruction
    orr     w5, w5, w4             // OR in the offset
    str     w5, [x0, x3]           // write back
    NEXT

// emit-patch-b ( mark_pos -- )
// Patch a B (unconditional) at mark_pos.
.align 4
code_E_PATCH_B:
    mov     x3, x22
    ldr     x22, [x24], #8

    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x2, [x1]

    sub     x4, x2, x3
    asr     x4, x4, #2
    and     w4, w4, #0x3FFFFFF     // 26-bit mask

    ldr     w5, [x0, x3]
    orr     w5, w5, w4
    str     w5, [x0, x3]
    NEXT

// emit-patch-cbz ( mark_pos -- )
// Patch a CBZ/CBNZ at mark_pos.
.align 4
code_E_PATCH_CBZ:
    mov     x3, x22
    ldr     x22, [x24], #8

    adrp    x0, ls_code_buf
    add     x0, x0, :lo12:ls_code_buf
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x2, [x1]

    sub     x4, x2, x3
    asr     x4, x4, #2
    and     w4, w4, #0x7FFFF
    lsl     w4, w4, #5

    ldr     w5, [x0, x3]
    orr     w5, w5, w4
    str     w5, [x0, x3]
    NEXT

// Forward-reference helpers: emit placeholder + push mark

// emit-bcond-fwd ( cond -- mark )
.align 4
code_E_BCOND_FWD:
    // Save cond, push mark (current pos), then emit B.cond with offset=0
    mov     x6, x22                 // cond
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]              // TOS = mark (current pos)
    // Emit: 0x54000000 | cond
    LOAD_BASE32 w5, 0x54000000
    orr     w0, w5, w6
    EMIT_INLINE
    NEXT

// emit-b-fwd ( -- mark )
.align 4
code_E_B_FWD:
    PUSH    x22
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]              // TOS = mark
    LOAD_BASE32 w0, 0x14000000     // B with offset=0
    EMIT_INLINE
    NEXT

// emit-cbz-fwd ( rt -- mark )
.align 4
code_E_CBZ_FWD:
    mov     x6, x22                 // rt
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]              // TOS = mark
    LOAD_BASE32 w5, 0xB4000000
    orr     w0, w5, w6             // | Rt
    EMIT_INLINE
    NEXT

// emit-cbnz-fwd ( rt -- mark )
.align 4
code_E_CBNZ_FWD:
    mov     x6, x22
    adrp    x1, ls_code_pos
    add     x1, x1, :lo12:ls_code_pos
    ldr     x22, [x1]
    LOAD_BASE32 w5, 0xB5000000
    orr     w0, w5, w6
    EMIT_INLINE
    NEXT

// ============================================================
// 12. SIGN-EXTENDING LOADS
// ============================================================

// emit-ldrsb ( rt rn imm12 -- )  LDRSB Xt, [Xn, #imm12]
.align 4
code_E_LDRSB:
    CODE_RD_RN_IMM12 0x39800000

// emit-ldrsh ( rt rn imm -- )  LDRSH Xt, [Xn, #imm]
// 0x79800000 | (imm/2)<<10 | Rn<<5 | Rt
.align 4
code_E_LDRSH:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #1
    LOAD_BASE32 w5, 0x79800000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// emit-ldrsw ( rt rn imm -- )  LDRSW Xt, [Xn, #imm]
// 0xB9800000 | (imm/4)<<10 | Rn<<5 | Rt
.align 4
code_E_LDRSW:
    mov     x3, x22
    ldr     x4, [x24], #8
    ldr     x0, [x24], #8
    ldr     x22, [x24], #8
    lsr     x3, x3, #2
    LOAD_BASE32 w5, 0xB9800000
    orr     w0, w5, w0
    orr     w0, w0, w4, lsl #5
    orr     w0, w0, w3, lsl #10
    EMIT_INLINE
    NEXT

// ============================================================
// DICTIONARY ENTRIES
// ============================================================
// Link these into the dictionary chain. The last entry in
// lithos-bootstrap.s is entry_pad. We continue from there.
//
// Entry format: [link(8)] [flags(1)] [nlen(1)] [name(padded)] [CFA(8)]

.data
.align 3

// First emitter entry — links to lexer chain tail
entry_emit_word:
    .quad   entry_lex_token_fetch
    .byte   0
    .byte   9
    .ascii  "emit-word"
    .align  3
    .quad   code_EMIT_WORD

entry_code_here:
    .quad   entry_emit_word
    .byte   0
    .byte   9
    .ascii  "code-here"
    .align  3
    .quad   code_CODE_HERE

entry_code_pos:
    .quad   entry_code_here
    .byte   0
    .byte   8
    .ascii  "code-pos"
    .align  3
    .quad   code_CODE_POS

entry_code_reset:
    .quad   entry_code_pos
    .byte   0
    .byte   10
    .ascii  "code-reset"
    .align  3
    .quad   code_CODE_RESET

entry_code_buf:
    .quad   entry_code_reset
    .byte   0
    .byte   8
    .ascii  "code-buf"
    .align  3
    .quad   code_CODE_BUF

// --- Data processing register ---

entry_e_add_reg:
    .quad   entry_code_buf
    .byte   0
    .byte   12
    .ascii  "emit-add-reg"
    .align  3
    .quad   code_E_ADD_REG

entry_e_sub_reg:
    .quad   entry_e_add_reg
    .byte   0
    .byte   12
    .ascii  "emit-sub-reg"
    .align  3
    .quad   code_E_SUB_REG

entry_e_adds_reg:
    .quad   entry_e_sub_reg
    .byte   0
    .byte   13
    .ascii  "emit-adds-reg"
    .align  3
    .quad   code_E_ADDS_REG

entry_e_subs_reg:
    .quad   entry_e_adds_reg
    .byte   0
    .byte   13
    .ascii  "emit-subs-reg"
    .align  3
    .quad   code_E_SUBS_REG

entry_e_mul:
    .quad   entry_e_subs_reg
    .byte   0
    .byte   8
    .ascii  "emit-mul"
    .align  3
    .quad   code_E_MUL

entry_e_sdiv:
    .quad   entry_e_mul
    .byte   0
    .byte   9
    .ascii  "emit-sdiv"
    .align  3
    .quad   code_E_SDIV

entry_e_udiv:
    .quad   entry_e_sdiv
    .byte   0
    .byte   9
    .ascii  "emit-udiv"
    .align  3
    .quad   code_E_UDIV

entry_e_and_reg:
    .quad   entry_e_udiv
    .byte   0
    .byte   12
    .ascii  "emit-and-reg"
    .align  3
    .quad   code_E_AND_REG

entry_e_orr_reg:
    .quad   entry_e_and_reg
    .byte   0
    .byte   12
    .ascii  "emit-orr-reg"
    .align  3
    .quad   code_E_ORR_REG

entry_e_eor_reg:
    .quad   entry_e_orr_reg
    .byte   0
    .byte   12
    .ascii  "emit-eor-reg"
    .align  3
    .quad   code_E_EOR_REG

entry_e_ands_reg:
    .quad   entry_e_eor_reg
    .byte   0
    .byte   13
    .ascii  "emit-ands-reg"
    .align  3
    .quad   code_E_ANDS_REG

entry_e_lsl_reg:
    .quad   entry_e_ands_reg
    .byte   0
    .byte   12
    .ascii  "emit-lsl-reg"
    .align  3
    .quad   code_E_LSL_REG

entry_e_lsr_reg:
    .quad   entry_e_lsl_reg
    .byte   0
    .byte   12
    .ascii  "emit-lsr-reg"
    .align  3
    .quad   code_E_LSR_REG

entry_e_asr_reg:
    .quad   entry_e_lsr_reg
    .byte   0
    .byte   12
    .ascii  "emit-asr-reg"
    .align  3
    .quad   code_E_ASR_REG

entry_e_madd:
    .quad   entry_e_asr_reg
    .byte   0
    .byte   9
    .ascii  "emit-madd"
    .align  3
    .quad   code_E_MADD

entry_e_msub:
    .quad   entry_e_madd
    .byte   0
    .byte   9
    .ascii  "emit-msub"
    .align  3
    .quad   code_E_MSUB

// --- Data processing immediate ---

entry_e_add_imm:
    .quad   entry_e_msub
    .byte   0
    .byte   12
    .ascii  "emit-add-imm"
    .align  3
    .quad   code_E_ADD_IMM

entry_e_add_imm_lsl12:
    .quad   entry_e_add_imm
    .byte   0
    .byte   18
    .ascii  "emit-add-imm-lsl12"
    .align  3
    .quad   code_E_ADD_IMM_LSL12

entry_e_sub_imm:
    .quad   entry_e_add_imm_lsl12
    .byte   0
    .byte   12
    .ascii  "emit-sub-imm"
    .align  3
    .quad   code_E_SUB_IMM

entry_e_adds_imm:
    .quad   entry_e_sub_imm
    .byte   0
    .byte   13
    .ascii  "emit-adds-imm"
    .align  3
    .quad   code_E_ADDS_IMM

entry_e_subs_imm:
    .quad   entry_e_adds_imm
    .byte   0
    .byte   13
    .ascii  "emit-subs-imm"
    .align  3
    .quad   code_E_SUBS_IMM

// --- Aliases ---

entry_e_cmp_reg:
    .quad   entry_e_subs_imm
    .byte   0
    .byte   12
    .ascii  "emit-cmp-reg"
    .align  3
    .quad   code_E_CMP_REG

entry_e_cmp_imm:
    .quad   entry_e_cmp_reg
    .byte   0
    .byte   12
    .ascii  "emit-cmp-imm"
    .align  3
    .quad   code_E_CMP_IMM

entry_e_cmn_imm:
    .quad   entry_e_cmp_imm
    .byte   0
    .byte   12
    .ascii  "emit-cmn-imm"
    .align  3
    .quad   code_E_CMN_IMM

entry_e_tst_reg:
    .quad   entry_e_cmn_imm
    .byte   0
    .byte   12
    .ascii  "emit-tst-reg"
    .align  3
    .quad   code_E_TST_REG

entry_e_neg:
    .quad   entry_e_tst_reg
    .byte   0
    .byte   8
    .ascii  "emit-neg"
    .align  3
    .quad   code_E_NEG

entry_e_mov:
    .quad   entry_e_neg
    .byte   0
    .byte   8
    .ascii  "emit-mov"
    .align  3
    .quad   code_E_MOV

entry_e_mvn:
    .quad   entry_e_mov
    .byte   0
    .byte   8
    .ascii  "emit-mvn"
    .align  3
    .quad   code_E_MVN

// --- Move wide ---

entry_e_movz:
    .quad   entry_e_mvn
    .byte   0
    .byte   9
    .ascii  "emit-movz"
    .align  3
    .quad   code_E_MOVZ

entry_e_movk:
    .quad   entry_e_movz
    .byte   0
    .byte   9
    .ascii  "emit-movk"
    .align  3
    .quad   code_E_MOVK

entry_e_movn:
    .quad   entry_e_movk
    .byte   0
    .byte   9
    .ascii  "emit-movn"
    .align  3
    .quad   code_E_MOVN

entry_e_mov64:
    .quad   entry_e_movn
    .byte   0
    .byte   10
    .ascii  "emit-mov64"
    .align  3
    .quad   code_E_MOV64

// --- Shift immediate ---

entry_e_lsl_imm:
    .quad   entry_e_mov64
    .byte   0
    .byte   12
    .ascii  "emit-lsl-imm"
    .align  3
    .quad   code_E_LSL_IMM

entry_e_lsr_imm:
    .quad   entry_e_lsl_imm
    .byte   0
    .byte   12
    .ascii  "emit-lsr-imm"
    .align  3
    .quad   code_E_LSR_IMM

entry_e_asr_imm:
    .quad   entry_e_lsr_imm
    .byte   0
    .byte   12
    .ascii  "emit-asr-imm"
    .align  3
    .quad   code_E_ASR_IMM

// --- Loads and stores ---

entry_e_ldr:
    .quad   entry_e_asr_imm
    .byte   0
    .byte   8
    .ascii  "emit-ldr"
    .align  3
    .quad   code_E_LDR

entry_e_str:
    .quad   entry_e_ldr
    .byte   0
    .byte   8
    .ascii  "emit-str"
    .align  3
    .quad   code_E_STR

entry_e_ldr_reg:
    .quad   entry_e_str
    .byte   0
    .byte   12
    .ascii  "emit-ldr-reg"
    .align  3
    .quad   code_E_LDR_REG

entry_e_str_reg:
    .quad   entry_e_ldr_reg
    .byte   0
    .byte   12
    .ascii  "emit-str-reg"
    .align  3
    .quad   code_E_STR_REG

entry_e_ldrb:
    .quad   entry_e_str_reg
    .byte   0
    .byte   9
    .ascii  "emit-ldrb"
    .align  3
    .quad   code_E_LDRB

entry_e_strb:
    .quad   entry_e_ldrb
    .byte   0
    .byte   9
    .ascii  "emit-strb"
    .align  3
    .quad   code_E_STRB

entry_e_ldrh:
    .quad   entry_e_strb
    .byte   0
    .byte   9
    .ascii  "emit-ldrh"
    .align  3
    .quad   code_E_LDRH

entry_e_strh:
    .quad   entry_e_ldrh
    .byte   0
    .byte   9
    .ascii  "emit-strh"
    .align  3
    .quad   code_E_STRH

entry_e_ldr_w:
    .quad   entry_e_strh
    .byte   0
    .byte   10
    .ascii  "emit-ldr-w"
    .align  3
    .quad   code_E_LDR_W

entry_e_str_w:
    .quad   entry_e_ldr_w
    .byte   0
    .byte   10
    .ascii  "emit-str-w"
    .align  3
    .quad   code_E_STR_W

entry_e_ldr_pre:
    .quad   entry_e_str_w
    .byte   0
    .byte   12
    .ascii  "emit-ldr-pre"
    .align  3
    .quad   code_E_LDR_PRE

entry_e_str_pre:
    .quad   entry_e_ldr_pre
    .byte   0
    .byte   12
    .ascii  "emit-str-pre"
    .align  3
    .quad   code_E_STR_PRE

entry_e_ldr_post:
    .quad   entry_e_str_pre
    .byte   0
    .byte   13
    .ascii  "emit-ldr-post"
    .align  3
    .quad   code_E_LDR_POST

entry_e_str_post:
    .quad   entry_e_ldr_post
    .byte   0
    .byte   13
    .ascii  "emit-str-post"
    .align  3
    .quad   code_E_STR_POST

entry_e_ldp:
    .quad   entry_e_str_post
    .byte   0
    .byte   8
    .ascii  "emit-ldp"
    .align  3
    .quad   code_E_LDP

entry_e_stp:
    .quad   entry_e_ldp
    .byte   0
    .byte   8
    .ascii  "emit-stp"
    .align  3
    .quad   code_E_STP

entry_e_ldp_pre:
    .quad   entry_e_stp
    .byte   0
    .byte   12
    .ascii  "emit-ldp-pre"
    .align  3
    .quad   code_E_LDP_PRE

entry_e_stp_pre:
    .quad   entry_e_ldp_pre
    .byte   0
    .byte   12
    .ascii  "emit-stp-pre"
    .align  3
    .quad   code_E_STP_PRE

entry_e_ldp_post:
    .quad   entry_e_stp_pre
    .byte   0
    .byte   13
    .ascii  "emit-ldp-post"
    .align  3
    .quad   code_E_LDP_POST

entry_e_stp_post:
    .quad   entry_e_ldp_post
    .byte   0
    .byte   13
    .ascii  "emit-stp-post"
    .align  3
    .quad   code_E_STP_POST

// --- Sign-extending loads ---

entry_e_ldrsb:
    .quad   entry_e_stp_post
    .byte   0
    .byte   10
    .ascii  "emit-ldrsb"
    .align  3
    .quad   code_E_LDRSB

entry_e_ldrsh:
    .quad   entry_e_ldrsb
    .byte   0
    .byte   10
    .ascii  "emit-ldrsh"
    .align  3
    .quad   code_E_LDRSH

entry_e_ldrsw:
    .quad   entry_e_ldrsh
    .byte   0
    .byte   10
    .ascii  "emit-ldrsw"
    .align  3
    .quad   code_E_LDRSW

// --- Branches ---

entry_e_b:
    .quad   entry_e_ldrsw
    .byte   0
    .byte   6
    .ascii  "emit-b"
    .align  3
    .quad   code_E_B

entry_e_bl:
    .quad   entry_e_b
    .byte   0
    .byte   7
    .ascii  "emit-bl"
    .align  3
    .quad   code_E_BL

entry_e_bcond:
    .quad   entry_e_bl
    .byte   0
    .byte   10
    .ascii  "emit-bcond"
    .align  3
    .quad   code_E_BCOND

entry_e_cbz:
    .quad   entry_e_bcond
    .byte   0
    .byte   8
    .ascii  "emit-cbz"
    .align  3
    .quad   code_E_CBZ

entry_e_cbnz:
    .quad   entry_e_cbz
    .byte   0
    .byte   9
    .ascii  "emit-cbnz"
    .align  3
    .quad   code_E_CBNZ

entry_e_tbz:
    .quad   entry_e_cbnz
    .byte   0
    .byte   8
    .ascii  "emit-tbz"
    .align  3
    .quad   code_E_TBZ

entry_e_tbnz:
    .quad   entry_e_tbz
    .byte   0
    .byte   9
    .ascii  "emit-tbnz"
    .align  3
    .quad   code_E_TBNZ

entry_e_br:
    .quad   entry_e_tbnz
    .byte   0
    .byte   7
    .ascii  "emit-br"
    .align  3
    .quad   code_E_BR

entry_e_blr:
    .quad   entry_e_br
    .byte   0
    .byte   8
    .ascii  "emit-blr"
    .align  3
    .quad   code_E_BLR

entry_e_ret:
    .quad   entry_e_blr
    .byte   0
    .byte   8
    .ascii  "emit-ret"
    .align  3
    .quad   code_E_RET

entry_e_ret_reg:
    .quad   entry_e_ret
    .byte   0
    .byte   12
    .ascii  "emit-ret-reg"
    .align  3
    .quad   code_E_RET_REG

// --- Conditional select ---

entry_e_csel:
    .quad   entry_e_ret_reg
    .byte   0
    .byte   9
    .ascii  "emit-csel"
    .align  3
    .quad   code_E_CSEL

entry_e_csinc:
    .quad   entry_e_csel
    .byte   0
    .byte   10
    .ascii  "emit-csinc"
    .align  3
    .quad   code_E_CSINC

entry_e_cset:
    .quad   entry_e_csinc
    .byte   0
    .byte   9
    .ascii  "emit-cset"
    .align  3
    .quad   code_E_CSET

entry_e_csneg:
    .quad   entry_e_cset
    .byte   0
    .byte   10
    .ascii  "emit-csneg"
    .align  3
    .quad   code_E_CSNEG

// --- System ---

entry_e_svc:
    .quad   entry_e_csneg
    .byte   0
    .byte   8
    .ascii  "emit-svc"
    .align  3
    .quad   code_E_SVC

entry_e_syscall:
    .quad   entry_e_svc
    .byte   0
    .byte   12
    .ascii  "emit-syscall"
    .align  3
    .quad   code_E_SYSCALL

entry_e_nop:
    .quad   entry_e_syscall
    .byte   0
    .byte   8
    .ascii  "emit-nop"
    .align  3
    .quad   code_E_NOP

entry_e_brk:
    .quad   entry_e_nop
    .byte   0
    .byte   8
    .ascii  "emit-brk"
    .align  3
    .quad   code_E_BRK

entry_e_mrs:
    .quad   entry_e_brk
    .byte   0
    .byte   8
    .ascii  "emit-mrs"
    .align  3
    .quad   code_E_MRS

entry_e_msr:
    .quad   entry_e_mrs
    .byte   0
    .byte   8
    .ascii  "emit-msr"
    .align  3
    .quad   code_E_MSR

entry_e_dsb_ish:
    .quad   entry_e_msr
    .byte   0
    .byte   12
    .ascii  "emit-dsb-ish"
    .align  3
    .quad   code_E_DSB_ISH

entry_e_dsb_sy:
    .quad   entry_e_dsb_ish
    .byte   0
    .byte   11
    .ascii  "emit-dsb-sy"
    .align  3
    .quad   code_E_DSB_SY

entry_e_dmb_ish:
    .quad   entry_e_dsb_sy
    .byte   0
    .byte   12
    .ascii  "emit-dmb-ish"
    .align  3
    .quad   code_E_DMB_ISH

entry_e_isb:
    .quad   entry_e_dmb_ish
    .byte   0
    .byte   8
    .ascii  "emit-isb"
    .align  3
    .quad   code_E_ISB

// --- PC-relative ---

entry_e_adr:
    .quad   entry_e_isb
    .byte   0
    .byte   8
    .ascii  "emit-adr"
    .align  3
    .quad   code_E_ADR

entry_e_adrp:
    .quad   entry_e_adr
    .byte   0
    .byte   9
    .ascii  "emit-adrp"
    .align  3
    .quad   code_E_ADRP

// --- Logical immediate ---

entry_e_and_imm:
    .quad   entry_e_adrp
    .byte   0
    .byte   12
    .ascii  "emit-and-imm"
    .align  3
    .quad   code_E_AND_IMM

entry_e_orr_imm:
    .quad   entry_e_and_imm
    .byte   0
    .byte   12
    .ascii  "emit-orr-imm"
    .align  3
    .quad   code_E_ORR_IMM

entry_e_eor_imm:
    .quad   entry_e_orr_imm
    .byte   0
    .byte   12
    .ascii  "emit-eor-imm"
    .align  3
    .quad   code_E_EOR_IMM

// --- Branch patching ---

entry_e_mark:
    .quad   entry_e_eor_imm
    .byte   0
    .byte   9
    .ascii  "emit-mark"
    .align  3
    .quad   code_E_MARK

entry_e_patch_bcond:
    .quad   entry_e_mark
    .byte   0
    .byte   16
    .ascii  "emit-patch-bcond"
    .align  3
    .quad   code_E_PATCH_BCOND

entry_e_patch_b:
    .quad   entry_e_patch_bcond
    .byte   0
    .byte   12
    .ascii  "emit-patch-b"
    .align  3
    .quad   code_E_PATCH_B

entry_e_patch_cbz:
    .quad   entry_e_patch_b
    .byte   0
    .byte   14
    .ascii  "emit-patch-cbz"
    .align  3
    .quad   code_E_PATCH_CBZ

entry_e_bcond_fwd:
    .quad   entry_e_patch_cbz
    .byte   0
    .byte   14
    .ascii  "emit-bcond-fwd"
    .align  3
    .quad   code_E_BCOND_FWD

entry_e_b_fwd:
    .quad   entry_e_bcond_fwd
    .byte   0
    .byte   10
    .ascii  "emit-b-fwd"
    .align  3
    .quad   code_E_B_FWD

entry_e_cbz_fwd:
    .quad   entry_e_b_fwd
    .byte   0
    .byte   12
    .ascii  "emit-cbz-fwd"
    .align  3
    .quad   code_E_CBZ_FWD

// last emitter entry — chain continues into lithos-parser.s
// (last_entry is defined at the tail of lithos-expr.s)
emit_last_entry:
entry_e_cbnz_fwd:
    .quad   entry_e_cbz_fwd
    .byte   0
    .byte   13
    .ascii  "emit-cbnz-fwd"
    .align  3
    .quad   code_E_CBNZ_FWD

// ============================================================
// To link into lithos-bootstrap.s:
// 1. Change entry_pad's link to point to emit_last_entry
//    (or change last_entry to point here)
// 2. Or: set var_latest to emit_last_entry at boot
// ============================================================
