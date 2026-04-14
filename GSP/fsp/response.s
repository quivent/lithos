// GSP/fsp/response.s -- FSP response polling, parsing, and ACK.
//
// After sending a COT (or any NVDM) command to FSP, we:
//   1) Poll NV_PFSP_MSGQ_HEAD(ch) until it advances past MSGQ_TAIL(ch),
//      indicating the FSP has staged a response in EMEM.
//   2) Parse the 4-DW (16-byte) response packet to extract the status
//      code; non-zero values are FSP error codes (see kern_fsp_retval.h).
//   3) ACK the response by advancing MSGQ_TAIL(ch) to match MSGQ_HEAD.
//
// Style mirrors GSP/poll_lockdown.s: AAPCS64, raw syscalls, no libc.
// Timing uses clock_gettime(CLOCK_MONOTONIC) via svc.
//
// Register map:
//   NV_PFSP_MSGQ_HEAD(i) = BAR0 + 0x8F2C80 + i*8    (FSP -> CPU head)
//   NV_PFSP_MSGQ_TAIL(i) = BAR0 + 0x8F2C84 + i*8    (CPU ack tail)
//
// Response packet layout (the 4-DW status block we parse):
//   DW0 = status / error code  (0 = FSP_OK)
//   DW1 = response type / command echo
//   DW2 = additional data
//   DW3 = checksum / reserved
//
// Build:
//   as -o response.o response.s

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------
.equ MSGQ_HEAD_BASE,        0x8F2C80        // + ch*8
.equ MSGQ_TAIL_BASE,        0x8F2C84        // + ch*8
.equ MSGQ_STRIDE,           8

.equ SYS_CLOCK_GETTIME,     113
.equ SYS_WRITE,             64
.equ CLOCK_MONOTONIC,       1
.equ STDERR_FD,             2

// ---- FSP status codes (kern_fsp_retval.h) ----
.equ FSP_OK,                            0x00
.equ FSP_ERR_IFS_ERR_INVALID_STATE,     0x9E
.equ FSP_ERR_IFR_FILE_NOT_FOUND,        0x9F
.equ FSP_ERR_IFS_ERR_NOT_SUPPORTED,     0xA0
.equ FSP_ERR_IFS_ERR_INVALID_DATA,      0xA1
.equ FSP_ERR_PRC_ERROR_INVALID_KNOB_ID, 0x1E3

// ---------------------------------------------------------------
.text

// =========================================================================
// fsp_response_poll(bar0, channel, timeout_secs)
//   x0 = bar0 mmap'd base
//   w1 = channel index (0..7)
//   w2 = timeout in whole seconds
// Returns:
//   x0 =  0 on response available (MSGQ_HEAD != MSGQ_TAIL)
//   x0 = -1 on timeout
//
// Stack frame (80 bytes):
//   sp+0..15  : saved x29, x30
//   sp+16..31 : saved x19, x20
//   sp+32..47 : saved x21, x22
//   sp+48..55 : saved x23
//   sp+64..79 : struct timespec (tv_sec, tv_nsec)
// =========================================================================
.globl fsp_response_poll
.type  fsp_response_poll, %function
.balign 4
fsp_response_poll:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    str x23, [sp, #48]

    mov x19, x0                     // x19 = bar0
    mov w20, w2                     // w20 = timeout_secs

    // Compute channel offset into MSGQ: ch*8
    and w1, w1, #0x7
    lsl w1, w1, #3                  // w1 = ch*8

    // x21 = bar0 + MSGQ_HEAD_BASE + ch*8
    // MSGQ_HEAD_BASE = 0x008F2C80 needs movz + movk.
    mov x2, #0x2C80
    movk x2, #0x008F, lsl #16       // x2 = 0x008F2C80
    add x21, x19, x2                // x21 = bar0 + MSGQ_HEAD_BASE
    add x21, x21, w1, uxtw          // + ch*8

    // x22 = bar0 + MSGQ_TAIL_BASE + ch*8 = x21 + 4
    add x22, x21, #4

    // ----- capture start time -----
    mov x0, #CLOCK_MONOTONIC
    add x1, sp, #64
    mov x8, #SYS_CLOCK_GETTIME
    svc #0
    ldr x23, [sp, #64]              // x23 = start_seconds

.Lrp_loop:
    // Read MSGQ_HEAD and MSGQ_TAIL
    ldr w0, [x21]                   // head (FSP writes)
    ldr w1, [x22]                   // tail (CPU owns)
    cmp w0, w1
    b.ne .Lrp_ready                 // head advanced -> response available

    // Timeout check
    mov x0, #CLOCK_MONOTONIC
    add x1, sp, #64
    mov x8, #SYS_CLOCK_GETTIME
    svc #0
    ldr x0, [sp, #64]
    sub x0, x0, x23                 // elapsed seconds
    cmp x0, x20, uxtw
    b.ge .Lrp_timeout

    // Short backoff (same rhythm as poll_lockdown.s)
    isb
    mov w0, #100
.Lrp_delay:
    nop
    subs w0, w0, #1
    b.ne .Lrp_delay
    b .Lrp_loop

.Lrp_ready:
    mov x0, #0
    b .Lrp_done

.Lrp_timeout:
    adr x0, .Lmsg_timeout
    mov x1, #.Lmsg_timeout_end - .Lmsg_timeout
    bl  _fsp_stderr_write
    mov x0, #-1

.Lrp_done:
    ldr x23, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
.size fsp_response_poll, . - fsp_response_poll


// =========================================================================
// fsp_response_parse(response_buf)
//   x0 = pointer to 16-byte response (4 DW)
// Returns:
//   w0 = 0 on FSP_OK, otherwise the FSP error code (positive), with
//        a short diagnostic tag emitted to stderr.
//
// We do NOT return negative values directly; callers treat non-zero
// as failure. The DW0 field from FSP is an unsigned status code.
// =========================================================================
.globl fsp_response_parse
.type  fsp_response_parse, %function
.balign 4
fsp_response_parse:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]

    mov x19, x0                     // x19 = response buffer
    // Raw EMEM data has 8 bytes of MCTP+NVDM headers (SOM packet):
    //   DW0 = MCTP transport constBlob (SOM|EOM|SEQ|EID)
    //   DW1 = NVDM msgType + vendorId + nvdmType
    //   DW2 = FSP status code (the actual payload)
    ldr w0, [x19, #8]              // DW2 = status (skip 8-byte header)
    cbz w0, .Lpar_ok                // 0 -> success, no print

    // Non-zero: classify and print a short tag.
    mov w1, w0                      // keep status in w1 for printing
    cmp w0, #FSP_ERR_IFS_ERR_INVALID_STATE
    b.eq .Lpar_e_state
    cmp w0, #FSP_ERR_IFR_FILE_NOT_FOUND
    b.eq .Lpar_e_notfound
    cmp w0, #FSP_ERR_IFS_ERR_NOT_SUPPORTED
    b.eq .Lpar_e_notsup
    cmp w0, #FSP_ERR_IFS_ERR_INVALID_DATA
    b.eq .Lpar_e_data
    mov w2, #0x01E3
    cmp w0, w2
    b.eq .Lpar_e_knob

    // Unknown status: print generic tag
    adr x0, .Lmsg_err_unknown
    mov x1, #.Lmsg_err_unknown_end - .Lmsg_err_unknown
    bl  _fsp_stderr_write
    b   .Lpar_return

.Lpar_e_state:
    adr x0, .Lmsg_err_state
    mov x1, #.Lmsg_err_state_end - .Lmsg_err_state
    bl  _fsp_stderr_write
    b   .Lpar_return
.Lpar_e_notfound:
    adr x0, .Lmsg_err_notfound
    mov x1, #.Lmsg_err_notfound_end - .Lmsg_err_notfound
    bl  _fsp_stderr_write
    b   .Lpar_return
.Lpar_e_notsup:
    adr x0, .Lmsg_err_notsup
    mov x1, #.Lmsg_err_notsup_end - .Lmsg_err_notsup
    bl  _fsp_stderr_write
    b   .Lpar_return
.Lpar_e_data:
    adr x0, .Lmsg_err_data
    mov x1, #.Lmsg_err_data_end - .Lmsg_err_data
    bl  _fsp_stderr_write
    b   .Lpar_return
.Lpar_e_knob:
    adr x0, .Lmsg_err_knob
    mov x1, #.Lmsg_err_knob_end - .Lmsg_err_knob
    bl  _fsp_stderr_write
    b   .Lpar_return

.Lpar_return:
    // Reload the raw status DW so the caller sees the exact code.
    ldr w0, [x19]
    b   .Lpar_done
.Lpar_ok:
    mov w0, #0
.Lpar_done:
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
.size fsp_response_parse, . - fsp_response_parse


// =========================================================================
// fsp_response_ack(bar0, channel, consumed_bytes)
//   x0 = bar0
//   w1 = channel
//   w2 = consumed_bytes (amount by which to advance MSGQ_TAIL)
//
// Advances MSGQ_TAIL(ch) by consumed_bytes. The FSP observes tail
// catching up to head and considers the response ACKed.
// =========================================================================
.globl fsp_response_ack
.type  fsp_response_ack, %function
.balign 4
fsp_response_ack:
    and w1, w1, #0x7
    lsl w1, w1, #3                  // w1 = ch*8
    mov x3, #0x2C84
    movk x3, #0x008F, lsl #16       // x3 = MSGQ_TAIL_BASE
    add x3, x0, x3
    add x3, x3, w1, uxtw            // x3 = bar0 + MSGQ_TAIL(ch)

    ldr w4, [x3]                    // current tail
    add w4, w4, w2                  // += consumed_bytes
    str w4, [x3]
    dsb     st                          // ensure MSGQ_TAIL write is committed
    // MMIO posted-write barrier: read-back to force ordering.
    ldr w4, [x3]
    ret
.size fsp_response_ack, . - fsp_response_ack


// =========================================================================
// Internal: write(stderr, buf=x0, len=x1). Clobbers x0, x1, x2, x8.
// Preserves LR via the caller (this is a leaf syscall helper with a
// single svc so no prologue is needed).
// =========================================================================
.type  _fsp_stderr_write, %function
.balign 4
_fsp_stderr_write:
    mov x2, x1                      // len
    mov x1, x0                      // buf
    mov x0, #STDERR_FD
    mov x8, #SYS_WRITE
    svc #0
    ret
.size _fsp_stderr_write, . - _fsp_stderr_write


// ---------------------------------------------------------------
// Read-only diagnostic strings (kept in .text via adr; they are
// short and do not need .rodata relocation.)
// ---------------------------------------------------------------
.balign 4
.Lmsg_timeout:
    .ascii "fsp_response_poll: timeout waiting for MSGQ_HEAD\n"
.Lmsg_timeout_end:

.Lmsg_err_unknown:
    .ascii "fsp_response_parse: FSP returned unknown error status\n"
.Lmsg_err_unknown_end:

.Lmsg_err_state:
    .ascii "fsp_response_parse: FSP_ERR_IFS_ERR_INVALID_STATE (0x9E)\n"
.Lmsg_err_state_end:

.Lmsg_err_notfound:
    .ascii "fsp_response_parse: FSP_ERR_IFR_FILE_NOT_FOUND (0x9F)\n"
.Lmsg_err_notfound_end:

.Lmsg_err_notsup:
    .ascii "fsp_response_parse: FSP_ERR_IFS_ERR_NOT_SUPPORTED (0xA0)\n"
.Lmsg_err_notsup_end:

.Lmsg_err_data:
    .ascii "fsp_response_parse: FSP_ERR_IFS_ERR_INVALID_DATA (0xA1)\n"
.Lmsg_err_data_end:

.Lmsg_err_knob:
    .ascii "fsp_response_parse: FSP_ERR_PRC_ERROR_INVALID_KNOB_ID (0x1E3)\n"
.Lmsg_err_knob_end:
