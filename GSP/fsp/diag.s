// fsp/diag.s -- FSP diagnostic routines for bringup debugging
//
// These routines are OPTIONAL. They are called from bootcmd.s on failure
// paths, or invoked ad-hoc from a debugger during FSP bringup. They are
// useful because the FSP boot path is the most failure-prone step of GSP
// boot: without visibility into the FSP's scratch groups and channel
// queues we cannot distinguish "FSP rejected COT payload" from "FSP hung"
// from "host never advanced QUEUE_HEAD".
//
// Exports:
//   fsp_diag_dump_scratch(bar0)              -- x0
//   fsp_diag_dump_queues(bar0, n_channels)   -- x0, w1
//   fsp_diag_decode_error(status_code)       -- w0
//   print_hex32(value, prefix, prefix_len)   -- w0, x1, w2
//
// Style: AAPCS64, raw SYS_WRITE syscalls, no libc. Hex printing pattern
// adapted from GSP/pmc_check.s (pmc_print_hex32).

// ---- Syscall numbers ----
.equ SYS_WRITE,     64

// ---- FSP register offsets (from fsp_plan.md table) ----
// NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2(i) = 0x8F0320 + i*4  (8 dwords)
// NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_3(i) = 0x8F0330 + i*4  (8 dwords)
// NV_PFSP_QUEUE_HEAD(i)                    = 0x8F2C00 + i*8
// NV_PFSP_QUEUE_TAIL(i)                    = 0x8F2C04 + i*8
// NV_PFSP_MSGQ_HEAD(i)                     = 0x8F2C80 + i*8
// NV_PFSP_MSGQ_TAIL(i)                     = 0x8F2C84 + i*8
.equ SCRATCH_GROUP_2,   0x8F0320
.equ SCRATCH_GROUP_3,   0x8F0330
.equ QUEUE_HEAD,        0x8F2C00
.equ QUEUE_TAIL,        0x8F2C04
.equ MSGQ_HEAD,         0x8F2C80
.equ MSGQ_TAIL,         0x8F2C84

// ============================================================
// Data
// ============================================================
.data
.align 3

// Shared hex digit table (local to this TU -- pmc_check.s has its own).
diag_hex_chars:     .ascii "0123456789abcdef"

// ---- Scratch dump labels (fixed width so caller can skip length math) ----
// Each label is exactly 15 bytes: "FSP_SCRATCHN[i]=" minus the '=' which
// print_hex32 does not emit; print_hex32 emits prefix then ": 0xXXXXXXXX\n".
// We use the prefix exactly as-provided.
lbl_s2_0:   .asciz "FSP_SCRATCH2[0]"
lbl_s2_0_l = . - lbl_s2_0 - 1
lbl_s2_1:   .asciz "FSP_SCRATCH2[1]"
lbl_s2_1_l = . - lbl_s2_1 - 1
lbl_s2_2:   .asciz "FSP_SCRATCH2[2]"
lbl_s2_2_l = . - lbl_s2_2 - 1
lbl_s2_3:   .asciz "FSP_SCRATCH2[3]"
lbl_s2_3_l = . - lbl_s2_3 - 1
lbl_s2_4:   .asciz "FSP_SCRATCH2[4]"
lbl_s2_4_l = . - lbl_s2_4 - 1
lbl_s2_5:   .asciz "FSP_SCRATCH2[5]"
lbl_s2_5_l = . - lbl_s2_5 - 1
lbl_s2_6:   .asciz "FSP_SCRATCH2[6]"
lbl_s2_6_l = . - lbl_s2_6 - 1
lbl_s2_7:   .asciz "FSP_SCRATCH2[7]"
lbl_s2_7_l = . - lbl_s2_7 - 1

lbl_s3_0:   .asciz "FSP_SCRATCH3[0]"
lbl_s3_0_l = . - lbl_s3_0 - 1
lbl_s3_1:   .asciz "FSP_SCRATCH3[1]"
lbl_s3_1_l = . - lbl_s3_1 - 1
lbl_s3_2:   .asciz "FSP_SCRATCH3[2]"
lbl_s3_2_l = . - lbl_s3_2 - 1
lbl_s3_3:   .asciz "FSP_SCRATCH3[3]"
lbl_s3_3_l = . - lbl_s3_3 - 1
lbl_s3_4:   .asciz "FSP_SCRATCH3[4]"
lbl_s3_4_l = . - lbl_s3_4 - 1
lbl_s3_5:   .asciz "FSP_SCRATCH3[5]"
lbl_s3_5_l = . - lbl_s3_5 - 1
lbl_s3_6:   .asciz "FSP_SCRATCH3[6]"
lbl_s3_6_l = . - lbl_s3_6 - 1
lbl_s3_7:   .asciz "FSP_SCRATCH3[7]"
lbl_s3_7_l = . - lbl_s3_7 - 1

// Per-channel queue labels (short -- we only dump up to 8 channels).
lbl_qh:     .asciz " QH"
lbl_qh_l = . - lbl_qh - 1
lbl_qt:     .asciz " QT"
lbl_qt_l = . - lbl_qt - 1
lbl_mh:     .asciz " MH"
lbl_mh_l = . - lbl_mh - 1
lbl_mt:     .asciz " MT"
lbl_mt_l = . - lbl_mt - 1

lbl_ch_prefix:   .asciz "CH["
lbl_ch_prefix_l = . - lbl_ch_prefix - 1
lbl_ch_suffix:   .asciz "]"
lbl_ch_suffix_l = . - lbl_ch_suffix - 1

// ---- Error decode table ----
// Each entry: 4-byte code, 4-byte reserved, 8-byte pointer to asciz str,
// 4-byte string length, 4-byte pad (to 24 bytes).
// Codes taken from NVIDIA kern_fsp_retval.h (approximate; used as tags).
.align 3
err_ok_s:       .asciz "FSP_OK"
err_ok_l = . - err_ok_s - 1
err_task_s:     .asciz "FSP_ERR_IFR_FILE_NOT_FOUND"
err_task_l = . - err_task_s - 1
err_sig_s:     .asciz "FSP_ERR_SIGNATURE_INVALID"
err_sig_l = . - err_sig_s - 1
err_cot_s:     .asciz "FSP_ERR_COT_PAYLOAD_REJECTED"
err_cot_l = . - err_cot_s - 1
err_seq_s:     .asciz "FSP_ERR_SEQUENCE_VIOLATION"
err_seq_l = . - err_seq_s - 1
err_perm_s:    .asciz "FSP_ERR_PERMISSION_DENIED"
err_perm_l = . - err_perm_s - 1
err_lock_s:    .asciz "FSP_ERR_LOCKDOWN_NOT_RELEASED"
err_lock_l = . - err_lock_s - 1
err_unk_s:     .asciz "FSP_ERR_UNKNOWN"
err_unk_l = . - err_unk_s - 1

.align 3
// Table: { u32 code; u32 pad; u64 str_addr; u32 str_len; u32 pad; }  (24B)
err_table:
    .word   0x00000000
    .word   0
    .quad   err_ok_s
    .word   err_ok_l
    .word   0

    .word   0x00000001
    .word   0
    .quad   err_task_s
    .word   err_task_l
    .word   0

    .word   0x00000002
    .word   0
    .quad   err_sig_s
    .word   err_sig_l
    .word   0

    .word   0x00000003
    .word   0
    .quad   err_cot_s
    .word   err_cot_l
    .word   0

    .word   0x00000004
    .word   0
    .quad   err_seq_s
    .word   err_seq_l
    .word   0

    .word   0x00000005
    .word   0
    .quad   err_perm_s
    .word   err_perm_l
    .word   0

    .word   0x00000006
    .word   0
    .quad   err_lock_s
    .word   err_lock_l
    .word   0
err_table_end:
.equ err_table_count, (err_table_end - err_table) / 24

err_decode_prefix:   .asciz "FSP_ERR: code=0x"
err_decode_prefix_l = . - err_decode_prefix - 1
err_decode_sep:      .asciz " -> "
err_decode_sep_l = . - err_decode_sep - 1
err_decode_nl:       .ascii "\n"

// ============================================================
// Text
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// diag_write -- raw write(stderr, x1, x2)
// Clobbers: x0, x8
// ------------------------------------------------------------
.align 4
diag_write:
    mov     x0, #2              // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// print_hex32(value, prefix_addr, prefix_len)
//   w0 = value
//   x1 = prefix address
//   w2 = prefix length (bytes)
//
// Emits: "<prefix>: 0xXXXXXXXX\n" to stderr.
// Clobbers: x0-x8, x16-x17.
// ------------------------------------------------------------
.align 4
.global print_hex32
print_hex32:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    mov     w19, w0                   // save value
    mov     x20, x1                   // save prefix addr
    uxtw    x21, w2                   // save prefix len

    // ---- 1. Write prefix ----
    mov     x1, x20
    mov     x2, x21
    bl      diag_write

    // ---- 2. Build ": 0xXXXXXXXX\n" in a 13-byte stack buffer ----
    // Layout: ':' ' ' '0' 'x' h7 h6 h5 h4 h3 h2 h1 h0 '\n'   = 13 bytes
    sub     sp, sp, #16
    mov     x22, sp                   // buffer

    mov     w3, #':'
    strb    w3, [x22, #0]
    mov     w3, #' '
    strb    w3, [x22, #1]
    mov     w3, #'0'
    strb    w3, [x22, #2]
    mov     w3, #'x'
    strb    w3, [x22, #3]

    adrp    x4, diag_hex_chars
    add     x4, x4, :lo12:diag_hex_chars

    // Emit 8 hex digits, MSB first, into buffer offsets [4..11].
    mov     w5, #28                   // starting shift
    mov     x6, x22
    add     x6, x6, #4                // write ptr
.ph32_loop:
    lsr     w7, w19, w5
    and     w7, w7, #0xF
    ldrb    w7, [x4, w7, uxtw]
    strb    w7, [x6], #1
    subs    w5, w5, #4
    b.ge    .ph32_loop

    mov     w3, #'\n'
    strb    w3, [x22, #12]

    mov     x1, x22
    mov     x2, #13
    bl      diag_write

    add     sp, sp, #16

    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ------------------------------------------------------------
// fsp_diag_dump_scratch(bar0)
//   x0 = bar0 base
//
// Reads SCRATCH_GROUP_2 and SCRATCH_GROUP_3 (8 dwords each) and prints
// each register as "FSP_SCRATCHn[i]: 0xXXXXXXXX\n" to stderr.
// Clobbers: x0-x8, x16-x17.
// ------------------------------------------------------------
.align 4
.global fsp_diag_dump_scratch
fsp_diag_dump_scratch:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    mov     x19, x0                   // bar0 base

    // Precompute SCRATCH_GROUP_2 base in x20 (callee-saved).
    // (ldr unsigned-offset imm for 32-bit loads is capped at 16380, so we
    // cannot use [x19, #0x8F0320] directly.)
    movz    x20, #(SCRATCH_GROUP_2 & 0xFFFF)
    movk    x20, #((SCRATCH_GROUP_2 >> 16) & 0xFFFF), lsl #16
    add     x20, x19, x20             // x20 = &SCRATCH_GROUP_2[0]

    // ---- SCRATCH_GROUP_2 (8 dwords) ----
    ldr     w0, [x20, #(0*4)]
    adrp    x1, lbl_s2_0
    add     x1, x1, :lo12:lbl_s2_0
    mov     w2, #lbl_s2_0_l
    bl      print_hex32

    ldr     w0, [x20, #(1*4)]
    adrp    x1, lbl_s2_1
    add     x1, x1, :lo12:lbl_s2_1
    mov     w2, #lbl_s2_1_l
    bl      print_hex32

    ldr     w0, [x20, #(2*4)]
    adrp    x1, lbl_s2_2
    add     x1, x1, :lo12:lbl_s2_2
    mov     w2, #lbl_s2_2_l
    bl      print_hex32

    ldr     w0, [x20, #(3*4)]
    adrp    x1, lbl_s2_3
    add     x1, x1, :lo12:lbl_s2_3
    mov     w2, #lbl_s2_3_l
    bl      print_hex32

    ldr     w0, [x20, #(4*4)]
    adrp    x1, lbl_s2_4
    add     x1, x1, :lo12:lbl_s2_4
    mov     w2, #lbl_s2_4_l
    bl      print_hex32

    ldr     w0, [x20, #(5*4)]
    adrp    x1, lbl_s2_5
    add     x1, x1, :lo12:lbl_s2_5
    mov     w2, #lbl_s2_5_l
    bl      print_hex32

    ldr     w0, [x20, #(6*4)]
    adrp    x1, lbl_s2_6
    add     x1, x1, :lo12:lbl_s2_6
    mov     w2, #lbl_s2_6_l
    bl      print_hex32

    ldr     w0, [x20, #(7*4)]
    adrp    x1, lbl_s2_7
    add     x1, x1, :lo12:lbl_s2_7
    mov     w2, #lbl_s2_7_l
    bl      print_hex32

    // ---- SCRATCH_GROUP_3 (8 dwords) ----
    // Repoint x20 to SCRATCH_GROUP_3 base.
    movz    x20, #(SCRATCH_GROUP_3 & 0xFFFF)
    movk    x20, #((SCRATCH_GROUP_3 >> 16) & 0xFFFF), lsl #16
    add     x20, x19, x20

    ldr     w0, [x20, #(0*4)]
    adrp    x1, lbl_s3_0
    add     x1, x1, :lo12:lbl_s3_0
    mov     w2, #lbl_s3_0_l
    bl      print_hex32

    ldr     w0, [x20, #(1*4)]
    adrp    x1, lbl_s3_1
    add     x1, x1, :lo12:lbl_s3_1
    mov     w2, #lbl_s3_1_l
    bl      print_hex32

    ldr     w0, [x20, #(2*4)]
    adrp    x1, lbl_s3_2
    add     x1, x1, :lo12:lbl_s3_2
    mov     w2, #lbl_s3_2_l
    bl      print_hex32

    ldr     w0, [x20, #(3*4)]
    adrp    x1, lbl_s3_3
    add     x1, x1, :lo12:lbl_s3_3
    mov     w2, #lbl_s3_3_l
    bl      print_hex32

    ldr     w0, [x20, #(4*4)]
    adrp    x1, lbl_s3_4
    add     x1, x1, :lo12:lbl_s3_4
    mov     w2, #lbl_s3_4_l
    bl      print_hex32

    ldr     w0, [x20, #(5*4)]
    adrp    x1, lbl_s3_5
    add     x1, x1, :lo12:lbl_s3_5
    mov     w2, #lbl_s3_5_l
    bl      print_hex32

    ldr     w0, [x20, #(6*4)]
    adrp    x1, lbl_s3_6
    add     x1, x1, :lo12:lbl_s3_6
    mov     w2, #lbl_s3_6_l
    bl      print_hex32

    ldr     w0, [x20, #(7*4)]
    adrp    x1, lbl_s3_7
    add     x1, x1, :lo12:lbl_s3_7
    mov     w2, #lbl_s3_7_l
    bl      print_hex32

    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ------------------------------------------------------------
// fsp_diag_dump_queues(bar0, n_channels)
//   x0 = bar0 base
//   w1 = number of channels to dump
//
// Emits per-channel:
//   "CH[i] QH: 0xXXXXXXXX\n"
//   "CH[i] QT: 0xXXXXXXXX\n"
//   "CH[i] MH: 0xXXXXXXXX\n"
//   "CH[i] MT: 0xXXXXXXXX\n"
//
// (We split onto four lines rather than one concatenated line because the
// print_hex32 primitive works per-value; parsing is trivial with grep.)
// Clobbers: x0-x8, x16-x17.
// ------------------------------------------------------------
.align 4
.global fsp_diag_dump_queues
fsp_diag_dump_queues:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]

    mov     x19, x0                   // bar0
    uxtw    x20, w1                   // n_channels
    mov     x21, #0                   // i

    // Reserve 16 bytes of stack for a per-iter prefix buffer:
    //   "CH[i] QH"    (i is single hex digit; channels 0..7)
    // max length = 3 + 1 + 1 + 2 = 7 bytes; round to 16.
    sub     sp, sp, #16
    mov     x22, sp

    adrp    x23, diag_hex_chars
    add     x23, x23, :lo12:diag_hex_chars

.dq_loop:
    cmp     x21, x20
    b.ge    .dq_done

    // Build "CH[i]" into x22[0..4]
    mov     w3, #'C'
    strb    w3, [x22, #0]
    mov     w3, #'H'
    strb    w3, [x22, #1]
    mov     w3, #'['
    strb    w3, [x22, #2]
    and     w4, w21, #0xF
    ldrb    w4, [x23, w4, uxtw]
    strb    w4, [x22, #3]
    mov     w3, #']'
    strb    w3, [x22, #4]

    // ---- QH ----
    mov     w3, #' '
    strb    w3, [x22, #5]
    mov     w3, #'Q'
    strb    w3, [x22, #6]
    mov     w3, #'H'
    strb    w3, [x22, #7]

    // Compute BAR0 + QUEUE_HEAD + i*8
    lsl     x5, x21, #3
    movz    x6, #(QUEUE_HEAD & 0xFFFF)
    movk    x6, #((QUEUE_HEAD >> 16) & 0xFFFF), lsl #16
    add     x6, x6, x5
    ldr     w0, [x19, x6]
    mov     x1, x22
    mov     w2, #8
    bl      print_hex32

    // ---- QT ----
    mov     w3, #'Q'
    strb    w3, [x22, #6]
    mov     w3, #'T'
    strb    w3, [x22, #7]
    lsl     x5, x21, #3
    movz    x6, #(QUEUE_TAIL & 0xFFFF)
    movk    x6, #((QUEUE_TAIL >> 16) & 0xFFFF), lsl #16
    add     x6, x6, x5
    ldr     w0, [x19, x6]
    mov     x1, x22
    mov     w2, #8
    bl      print_hex32

    // ---- MH ----
    mov     w3, #'M'
    strb    w3, [x22, #6]
    mov     w3, #'H'
    strb    w3, [x22, #7]
    lsl     x5, x21, #3
    movz    x6, #(MSGQ_HEAD & 0xFFFF)
    movk    x6, #((MSGQ_HEAD >> 16) & 0xFFFF), lsl #16
    add     x6, x6, x5
    ldr     w0, [x19, x6]
    mov     x1, x22
    mov     w2, #8
    bl      print_hex32

    // ---- MT ----
    mov     w3, #'M'
    strb    w3, [x22, #6]
    mov     w3, #'T'
    strb    w3, [x22, #7]
    lsl     x5, x21, #3
    movz    x6, #(MSGQ_TAIL & 0xFFFF)
    movk    x6, #((MSGQ_TAIL >> 16) & 0xFFFF), lsl #16
    add     x6, x6, x5
    ldr     w0, [x19, x6]
    mov     x1, x22
    mov     w2, #8
    bl      print_hex32

    add     x21, x21, #1
    b       .dq_loop

.dq_done:
    add     sp, sp, #16
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// ------------------------------------------------------------
// fsp_diag_decode_error(status_code)
//   w0 = status code (from FSP MSGQ response, kern_fsp_retval.h)
//
// Writes "FSP_ERR: code=0xXXXXXXXX -> <TAG>\n" to stderr.
// Codes not in the small internal table map to FSP_ERR_UNKNOWN.
// Clobbers: x0-x8, x16-x17.
// ------------------------------------------------------------
.align 4
.global fsp_diag_decode_error
fsp_diag_decode_error:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    mov     w19, w0                   // save status code

    // ---- 1. Write "FSP_ERR: code=0x" ----
    adrp    x1, err_decode_prefix
    add     x1, x1, :lo12:err_decode_prefix
    mov     x2, #err_decode_prefix_l
    bl      diag_write

    // ---- 2. Write 8 hex digits of status code ----
    sub     sp, sp, #16
    mov     x3, sp                    // buffer
    adrp    x4, diag_hex_chars
    add     x4, x4, :lo12:diag_hex_chars
    mov     w5, #28
.dec_hex_loop:
    lsr     w6, w19, w5
    and     w6, w6, #0xF
    ldrb    w6, [x4, w6, uxtw]
    strb    w6, [x3], #1
    subs    w5, w5, #4
    b.ge    .dec_hex_loop

    mov     x1, sp
    mov     x2, #8
    bl      diag_write
    add     sp, sp, #16

    // ---- 3. Write " -> " ----
    adrp    x1, err_decode_sep
    add     x1, x1, :lo12:err_decode_sep
    mov     x2, #err_decode_sep_l
    bl      diag_write

    // ---- 4. Look up code in err_table ----
    adrp    x3, err_table
    add     x3, x3, :lo12:err_table
    mov     x4, #err_table_count
    mov     x5, #0                    // index
.dec_lookup:
    cmp     x5, x4
    b.ge    .dec_unknown
    // Load 32-bit code at entry
    ldr     w6, [x3]
    cmp     w6, w19
    b.eq    .dec_found
    add     x3, x3, #24
    add     x5, x5, #1
    b       .dec_lookup

.dec_found:
    // entry +8 = str pointer, entry +16 = str length (w)
    ldr     x1, [x3, #8]
    ldr     w2, [x3, #16]
    uxtw    x2, w2
    bl      diag_write
    b       .dec_nl

.dec_unknown:
    adrp    x1, err_unk_s
    add     x1, x1, :lo12:err_unk_s
    mov     x2, #err_unk_l
    bl      diag_write

.dec_nl:
    adrp    x1, err_decode_nl
    add     x1, x1, :lo12:err_decode_nl
    mov     x2, #1
    bl      diag_write

    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
