// fsp/init.s -- Step 7a: FSP (Firmware Security Processor) initialization
//
// Verifies the FSP (second Falcon at BAR0+0x8F2000 on GH200) is alive,
// not in error state, and ready to accept MCTP/COT boot commands on
// channel 0. Must run AFTER bar_map / pmc_check / falcon_reset and
// BEFORE any EMEMC/EMEMD writes from fsp/bootcmd.s.
//
// Exports:
//   fsp_init   -- probe FSP, poll readiness, validate scratch state
//
// Calling convention: ARM64 AAPCS64
//   Input:  x0 = bar0_base (BAR0 mmap base pointer, from bar_map_init)
//   Output: x0 =  0  on success
//           x0 = -1  BAR0 null
//           x0 = -2  FSP not ready (scratch group 2 never asserted ready)
//           x0 = -3  CPU->FSP command queue not empty (head != tail)
//           x0 = -4  FSP->CPU msg queue not empty (head != tail)
//           x0 = -5  FSP scratch group 3 indicates prior error
//
// Register map (all 32-bit MMIO, BAR0 offsets from dev_fsp_pri.h / fsp_plan.md):
//
//   NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2(0) = 0x008F0320
//     -- FSP boot-state sentinel.  FSP ROM/bootloader writes a non-zero
//        magic here once EMEM queues are live.  A zero value means FSP
//        has not finished its own boot; a value with bit[31] set
//        conventionally indicates an error code rather than a ready
//        sentinel, but on GH100/GH200 any non-zero value is treated as
//        "FSP has progressed past its own reset" and the finer error
//        decode lives in SCRATCH_GROUP_3.
//
//   NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_3(0) = 0x008F0330
//     -- FSP error/diag scratch.  Must be zero at init time; any set
//        bit indicates the FSP has latched a fault (signature mismatch
//        from an earlier run, EMEM parity, etc.).  If non-zero we abort
//        rather than poking queues and arming another fault.
//
//   NV_PFSP_QUEUE_HEAD(0) = 0x008F2C00   CPU->FSP cmd queue head (bytes)
//   NV_PFSP_QUEUE_TAIL(0) = 0x008F2C04   CPU->FSP cmd queue tail (bytes)
//   NV_PFSP_MSGQ_HEAD(0)  = 0x008F2C80   FSP->CPU msg queue head  (bytes)
//   NV_PFSP_MSGQ_TAIL(0)  = 0x008F2C84   FSP->CPU msg queue tail  (bytes)
//
// Sequence:
//   1. Poll SCRATCH_GROUP_2(0) != 0 with timeout (FSP boot sentinel).
//   2. Verify SCRATCH_GROUP_3(0) == 0 (no latched FSP error).
//   3. Verify QUEUE_HEAD(0) == QUEUE_TAIL(0) (cmd queue empty).
//   4. Verify MSGQ_HEAD(0)  == MSGQ_TAIL(0)  (msg queue empty).
//
// Timeout is ~100M iterations (~1-2s on Grace at ~3GHz), matching the
// convention used in falcon_reset.s.

// Shared constants (syscalls, etc.)
.include "gsp_common.s"

// ---- FSP register offsets (absolute BAR0 offsets) ----
// Constructed via movz/movk at use sites; listed here for reference.
.equ FSP_SCRATCH_GROUP_2_0,         0x008F0320
.equ FSP_SCRATCH_GROUP_3_0,         0x008F0330
.equ FSP_QUEUE_HEAD_0,              0x008F2C00
.equ FSP_QUEUE_TAIL_0,              0x008F2C04
.equ FSP_MSGQ_HEAD_0,               0x008F2C80
.equ FSP_MSGQ_TAIL_0,               0x008F2C84

// ---- Timeout: ~100M iterations ----
.equ FSP_POLL_TIMEOUT,              100000000   // = 0x05F5E100

// ============================================================
// Data section
// ============================================================
.data
.align 3

fsp_msg_start:
    .asciz "fsp: init -- probing FSP at BAR0+0x8F2000...\n"
fsp_msg_start_len      = . - fsp_msg_start - 1

fsp_msg_ready:
    .asciz "fsp: SCRATCH_GROUP_2 non-zero -- FSP boot sentinel asserted\n"
fsp_msg_ready_len      = . - fsp_msg_ready - 1

fsp_msg_queues_empty:
    .asciz "fsp: command and message queues empty -- FSP ready for COT\n"
fsp_msg_queues_empty_len = . - fsp_msg_queues_empty - 1

fsp_msg_done:
    .asciz "fsp: init complete\n"
fsp_msg_done_len       = . - fsp_msg_done - 1

fsp_msg_bar0_null:
    .asciz "fsp: ERROR: BAR0 base is null\n"
fsp_msg_bar0_null_len  = . - fsp_msg_bar0_null - 1

fsp_msg_not_ready:
    .asciz "fsp: ERROR: SCRATCH_GROUP_2 never became non-zero (FSP not ready)\n"
fsp_msg_not_ready_len  = . - fsp_msg_not_ready - 1

fsp_msg_scratch3_err:
    .asciz "fsp: ERROR: SCRATCH_GROUP_3 non-zero -- FSP has latched error\n"
fsp_msg_scratch3_err_len = . - fsp_msg_scratch3_err - 1

fsp_msg_cmdq_busy:
    .asciz "fsp: ERROR: CPU->FSP command queue not empty (head != tail)\n"
fsp_msg_cmdq_busy_len  = . - fsp_msg_cmdq_busy - 1

fsp_msg_msgq_busy:
    .asciz "fsp: ERROR: FSP->CPU message queue not empty (head != tail)\n"
fsp_msg_msgq_busy_len  = . - fsp_msg_msgq_busy - 1

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// fsp_init -- verify FSP liveness and queue idleness
//
// Input:  x0 = bar0_base
// Output: x0 = 0 on success, negative on failure (see codes above)
// Clobbers: x0-x8, x16-x17, x19-x22
// ------------------------------------------------------------
.global fsp_init
fsp_init:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Save bar0_base
    mov     x19, x0
    cbz     x19, .fsp_bar0_null

    // Announce
    adrp    x1, fsp_msg_start
    add     x1, x1, :lo12:fsp_msg_start
    mov     x2, #fsp_msg_start_len
    bl      fsp_print_msg

    // --------------------------------------------------------
    // Step 1: Poll SCRATCH_GROUP_2(0) != 0 with timeout
    // addr = BAR0 + 0x008F0320
    // --------------------------------------------------------
    movz    x20, #0x0320
    movk    x20, #0x008F, lsl #16       // x20 = 0x008F0320
    add     x20, x19, x20               // x20 = &SCRATCH_GROUP_2(0)

    // Timeout counter = 100000000 = 0x05F5E100
    movz    x1, #0xE100
    movk    x1, #0x05F5, lsl #16
.fsp_poll_sg2:
    ldr     w0, [x20]                   // read SCRATCH_GROUP_2(0)
    cbnz    w0, .fsp_sg2_ready
    subs    x1, x1, #1
    isb                                     // serialize between MMIO poll iterations
    b.ne    .fsp_poll_sg2
    // Timeout: SCRATCH_GROUP_2 stayed zero
    adrp    x1, fsp_msg_not_ready
    add     x1, x1, :lo12:fsp_msg_not_ready
    mov     x2, #fsp_msg_not_ready_len
    bl      fsp_print_msg
    mov     x0, #-2
    b       .fsp_return

.fsp_sg2_ready:
    adrp    x1, fsp_msg_ready
    add     x1, x1, :lo12:fsp_msg_ready
    mov     x2, #fsp_msg_ready_len
    bl      fsp_print_msg

    // --------------------------------------------------------
    // Step 2: Verify SCRATCH_GROUP_3(0) == 0
    // addr = BAR0 + 0x008F0330
    // --------------------------------------------------------
    movz    x21, #0x0330
    movk    x21, #0x008F, lsl #16       // x21 = 0x008F0330
    add     x21, x19, x21               // x21 = &SCRATCH_GROUP_3(0)
    ldr     w0, [x21]
    cbnz    w0, .fsp_scratch3_err

    // --------------------------------------------------------
    // Step 3: Verify QUEUE_HEAD(0) == QUEUE_TAIL(0)
    // QUEUE_HEAD(0) = BAR0 + 0x008F2C00
    // QUEUE_TAIL(0) = BAR0 + 0x008F2C04
    // --------------------------------------------------------
    movz    x22, #0x2C00
    movk    x22, #0x008F, lsl #16       // x22 = 0x008F2C00
    add     x22, x19, x22               // x22 = &QUEUE_HEAD(0)
    ldr     w0, [x22]                   // QUEUE_HEAD(0)
    ldr     w1, [x22, #4]               // QUEUE_TAIL(0)
    cmp     w0, w1
    b.ne    .fsp_cmdq_busy

    // --------------------------------------------------------
    // Step 4: Verify MSGQ_HEAD(0) == MSGQ_TAIL(0)
    // MSGQ_HEAD(0) = BAR0 + 0x008F2C80
    // MSGQ_TAIL(0) = BAR0 + 0x008F2C84
    // --------------------------------------------------------
    movz    x22, #0x2C80
    movk    x22, #0x008F, lsl #16       // x22 = 0x008F2C80
    add     x22, x19, x22               // x22 = &MSGQ_HEAD(0)
    ldr     w0, [x22]                   // MSGQ_HEAD(0)
    ldr     w1, [x22, #4]               // MSGQ_TAIL(0)
    cmp     w0, w1
    b.ne    .fsp_msgq_busy

    // --------------------------------------------------------
    // Success
    // --------------------------------------------------------
    adrp    x1, fsp_msg_queues_empty
    add     x1, x1, :lo12:fsp_msg_queues_empty
    mov     x2, #fsp_msg_queues_empty_len
    bl      fsp_print_msg

    adrp    x1, fsp_msg_done
    add     x1, x1, :lo12:fsp_msg_done
    mov     x2, #fsp_msg_done_len
    bl      fsp_print_msg

    mov     x0, #0
    b       .fsp_return

.fsp_scratch3_err:
    adrp    x1, fsp_msg_scratch3_err
    add     x1, x1, :lo12:fsp_msg_scratch3_err
    mov     x2, #fsp_msg_scratch3_err_len
    bl      fsp_print_msg
    mov     x0, #-5
    b       .fsp_return

.fsp_cmdq_busy:
    adrp    x1, fsp_msg_cmdq_busy
    add     x1, x1, :lo12:fsp_msg_cmdq_busy
    mov     x2, #fsp_msg_cmdq_busy_len
    bl      fsp_print_msg
    mov     x0, #-3
    b       .fsp_return

.fsp_msgq_busy:
    adrp    x1, fsp_msg_msgq_busy
    add     x1, x1, :lo12:fsp_msg_msgq_busy
    mov     x2, #fsp_msg_msgq_busy_len
    bl      fsp_print_msg
    mov     x0, #-4
    b       .fsp_return

.fsp_bar0_null:
    adrp    x1, fsp_msg_bar0_null
    add     x1, x1, :lo12:fsp_msg_bar0_null
    mov     x2, #fsp_msg_bar0_null_len
    bl      fsp_print_msg
    mov     x0, #-1

.fsp_return:
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ------------------------------------------------------------
// fsp_print_msg -- write string to stderr via raw SYS_WRITE
// x1 = string address, x2 = length
// Clobbers: x0, x8
// ------------------------------------------------------------
.align 4
fsp_print_msg:
    mov     x0, #2                      // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret
