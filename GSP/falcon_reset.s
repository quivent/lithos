// falcon_reset.s -- Step 3: GSP Falcon reset sequence
//
// Asserts and deasserts reset on the GSP Falcon engine, waiting for
// hardware acknowledgment at each step.  This is a direct translation
// of gsp_reset_hw() from lithos_gsp.c.
//
// Exports:
//   falcon_reset   -- full reset sequence, returns 0 on success
//
// Requires: bar0_base must be populated (call bar_map_init first).
//
// Register map (BAR0 offsets from gsp-native.md Section 3.3):
//
//   0x1103C0 = NV_PGSP_FALCON_ENGINE
//              bit[0]     = RESET (1 = assert, 0 = deassert)
//              bits[10:8] = RESET_STATUS
//                           0b000 = ASSERTED
//                           0b010 = DEASSERTED
//                           0b001 = DEASSERT_IN_PROGRESS
//
// Sequence:
//   1. Store BAR0+0x1103C0 <- 0x00000001    (assert reset, bit 0 = 1)
//   2. Poll  BAR0+0x1103C0 bits[10:8] == 0  (wait for ASSERTED)
//   3. Store BAR0+0x1103C0 <- 0x00000000    (deassert reset, bit 0 = 0)
//   4. Poll  BAR0+0x1103C0 bits[10:8] == 2  (wait for DEASSERTED)
//
// The polling loops use clock_gettime(CLOCK_MONOTONIC) for wall-clock
// timeout to avoid dependence on CPU frequency or pipeline stalls.
// Timeout is 5 seconds for falcon reset.
//
// Calling convention: ARM64 AAPCS.
// Returns: x0 = 0 success, -1 assert timeout, -2 deassert timeout,
//          -3 BAR0 not mapped.

// Shared constants (syscalls, clock IDs, Falcon offsets, etc.)
.include "gsp_common.s"

// ---- File-specific constants ----
.equ SYS_RT_SIGPROCMASK, 135

.equ SIG_BLOCK,          0
.equ SIG_UNBLOCK,        1

// Reset status field -- bits[10:8] of FALCON_ENGINE register.
// (gsp_common.s defines FALCON_STATUS_MASK et al.; these short aliases
// are used throughout this file.)
.equ STATUS_MASK,        0x700         // bits[10:8]
.equ STATUS_ASSERTED,    0x000         // 0b000 << 8
.equ STATUS_DEASSERTED,  0x200         // 0b010 << 8

// Timeout: wall-clock seconds
.equ FALCON_TIMEOUT_SECS, 5

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_reset_start:    .asciz "gsp: falcon reset -- asserting...\n"
msg_reset_start_len = . - msg_reset_start - 1
msg_reset_asserted: .asciz "gsp: falcon reset -- asserted, deasserting...\n"
msg_reset_asserted_len = . - msg_reset_asserted - 1
msg_reset_done:     .asciz "gsp: falcon reset -- deasserted, reset complete\n"
msg_reset_done_len = . - msg_reset_done - 1
msg_assert_timeout: .asciz "gsp: ERROR: falcon reset assert timeout\n"
msg_assert_timeout_len = . - msg_assert_timeout - 1
msg_deassert_timeout: .asciz "gsp: ERROR: falcon reset deassert timeout, re-asserting\n"
msg_deassert_timeout_len = . - msg_deassert_timeout - 1
falcon_msg_bar0_null:      .asciz "gsp: ERROR: BAR0 not mapped for falcon reset\n"
falcon_msg_bar0_null_len = . - falcon_msg_bar0_null - 1

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// falcon_reset -- assert/deassert GSP Falcon engine reset
//
// Returns: x0 = 0 on success, -1 on failure (timeout or no BAR0)
// Clobbers: x0-x8, x16-x17
// ------------------------------------------------------------
.global falcon_reset
falcon_reset:
    // Prologue: save callee-saved regs and allocate timespec on stack.
    // Stack layout:
    //   sp+0..15:  saved x29, x30
    //   sp+16..31: saved x19, x20
    //   sp+32..47: saved x21, x22
    //   sp+48..63: struct timespec { tv_sec(8), tv_nsec(8) }
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Block SIGTERM/SIGINT/SIGHUP during critical MMIO
    mov     x0, #SIG_BLOCK
    adr     x1, .Lsigmask        // pointer to signal set
    mov     x2, #0               // don't save old set
    mov     x3, #8               // sigsetsize
    mov     x8, #SYS_RT_SIGPROCMASK
    svc     #0

    // Load BAR0 base
    adrp    x19, bar0_base
    add     x19, x19, :lo12:bar0_base
    ldr     x19, [x19]

    // Check BAR0 is mapped
    cbz     x19, .falcon_bar0_null

    // Compute register address: BAR0 + 0x1103C0
    movz    x20, #0x03C0              // low 16 bits of 0x1103C0
    movk    x20, #0x0011, lsl #16     // high 16 bits -> 0x001103C0
    add     x20, x19, x20             // x20 = &FALCON_ENGINE

    // ---- Step 1: Assert reset (bit 0 = 1) ----
    adrp    x1, msg_reset_start
    add     x1, x1, :lo12:msg_reset_start
    mov     x2, #msg_reset_start_len
    bl      falcon_print_msg

    mov     w0, #1
    str     w0, [x20]                 // BAR0+0x1103C0 <- 1
    dsb     st                        // ensure MMIO write reaches device

    // ---- Step 2: Poll bits[10:8] == 0 (ASSERTED) ----
    // Capture start time for wall-clock timeout
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48              // timespec at sp+48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .falcon_assert_timeout    // clock_gettime failed -- treat as timeout
    ldr     x21, [sp, #48]           // x21 = start_seconds

    dsb     sy                        // full barrier: ensure write visible before read
    isb                               // serialize pipeline before first MMIO read
.poll_asserted:
    ldr     w0, [x20]                 // read FALCON_ENGINE
    and     w0, w0, #STATUS_MASK      // isolate bits[10:8]
    cmp     w0, #STATUS_ASSERTED      // == 0?
    b.eq    .asserted_ok

    // Check wall-clock timeout
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .falcon_assert_timeout    // clock_gettime failed -- treat as timeout
    ldr     x0, [sp, #48]            // current_seconds
    sub     x0, x0, x21              // elapsed seconds
    cmp     x0, #FALCON_TIMEOUT_SECS
    b.ge    .falcon_assert_timeout

    dsb     ld                        // ensure previous MMIO read completed
    isb
    mov     w0, #100
.assert_delay:
    nop
    subs    w0, w0, #1
    b.ne    .assert_delay

    b       .poll_asserted

.asserted_ok:
    adrp    x1, msg_reset_asserted
    add     x1, x1, :lo12:msg_reset_asserted
    mov     x2, #msg_reset_asserted_len
    bl      falcon_print_msg

    // ---- Step 3: Deassert reset (bit 0 = 0) ----
    mov     w0, #0
    str     w0, [x20]                 // BAR0+0x1103C0 <- 0
    dsb     st                        // ensure MMIO write reaches device

    // ---- Step 4: Poll bits[10:8] == 2 (DEASSERTED) ----
    // Capture start time for wall-clock timeout
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48              // timespec at sp+48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .falcon_deassert_timeout  // clock_gettime failed -- treat as timeout
    ldr     x21, [sp, #48]           // x21 = start_seconds

    dsb     sy                        // full barrier: ensure write visible before read
    isb                               // serialize pipeline before first MMIO read
.poll_deasserted:
    ldr     w0, [x20]                 // read FALCON_ENGINE
    and     w0, w0, #STATUS_MASK      // isolate bits[10:8]
    cmp     w0, #STATUS_DEASSERTED    // == 0x200?
    b.eq    .deasserted_ok

    // Check wall-clock timeout
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .falcon_deassert_timeout  // clock_gettime failed -- treat as timeout
    ldr     x0, [sp, #48]            // current_seconds
    sub     x0, x0, x21              // elapsed seconds
    cmp     x0, #FALCON_TIMEOUT_SECS
    b.ge    .falcon_deassert_timeout

    dsb     ld                        // ensure previous MMIO read completed
    isb
    mov     w0, #100
.deassert_delay:
    nop
    subs    w0, w0, #1
    b.ne    .deassert_delay

    b       .poll_deasserted

.deasserted_ok:
    adrp    x1, msg_reset_done
    add     x1, x1, :lo12:msg_reset_done
    mov     x2, #msg_reset_done_len
    bl      falcon_print_msg

    // ---- Success ----
    mov     x0, #0
    b       .falcon_return

.falcon_assert_timeout:
    // Deassert reset to undo the partial assert and restore Falcon to
    // its pre-call state.  Without this, Falcon is stuck with bit[0]=1
    // written but RESET_STATUS never confirmed -- an undefined limbo.
    mov     w0, #0
    str     w0, [x20]                 // BAR0+0x1103C0 <- 0 (deassert)
    dsb     sy                        // ensure deassert reaches device

    adrp    x1, msg_assert_timeout
    add     x1, x1, :lo12:msg_assert_timeout
    mov     x2, #msg_assert_timeout_len
    bl      falcon_print_msg
    mov     x0, #-1                   // -1 = assert timeout
    b       .falcon_return

.falcon_deassert_timeout:
    // Re-assert reset to leave Falcon in a known held-in-reset state,
    // then poll briefly to confirm the re-assert actually took effect.
    mov     w22, #1
    str     w22, [x20]               // BAR0+0x1103C0 <- 1 (re-assert)
    dsb     sy                        // full barrier: ensure write reaches device
    isb

    // Brief poll (up to ~1000 iterations) to confirm re-assert.
    // If this also fails, we still return -2 -- but at least we tried.
    mov     w22, #1000
.reassert_poll:
    ldr     w0, [x20]
    and     w0, w0, #STATUS_MASK
    cmp     w0, #STATUS_ASSERTED
    b.eq    .reassert_confirmed
    subs    w22, w22, #1
    b.ne    .reassert_poll
.reassert_confirmed:

    adrp    x1, msg_deassert_timeout
    add     x1, x1, :lo12:msg_deassert_timeout
    mov     x2, #msg_deassert_timeout_len
    bl      falcon_print_msg
    mov     x0, #-2                   // -2 = deassert timeout
    b       .falcon_return

.falcon_bar0_null:
    adrp    x1, falcon_msg_bar0_null
    add     x1, x1, :lo12:falcon_msg_bar0_null
    mov     x2, #falcon_msg_bar0_null_len
    bl      falcon_print_msg
    mov     x0, #-3                   // -3 = BAR0 not mapped

.falcon_return:
    // Save return value across sigprocmask call
    mov     x19, x0

    // Unblock signals
    mov     x0, #SIG_UNBLOCK
    adr     x1, .Lsigmask
    mov     x2, #0
    mov     x3, #8
    mov     x8, #SYS_RT_SIGPROCMASK
    svc     #0

    // Restore return value
    mov     x0, x19

    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// ------------------------------------------------------------
// falcon_print_msg -- write string to stderr
// x1 = string address, x2 = length
// ------------------------------------------------------------
.align 4
falcon_print_msg:
    mov     x0, #2                    // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// Signal mask literal in .text (must be in same section as adr references)
.align 3
.Lsigmask:
    .quad   0x4003               // SIGHUP(1)=bit0, SIGINT(2)=bit1, SIGTERM(15)=bit14

// ---- Mailbox register offsets ----
.equ FALCON_MAILBOX0,    0x110040
.equ FALCON_MAILBOX1,    0x110044

// ============================================================
// Data section -- emergency reset messages
// ============================================================
.data
.align 3

emsg_start:         .asciz "gsp: emergency reset -- reading falcon state...\n"
emsg_start_len      = . - emsg_start - 1
emsg_deassert:      .asciz "gsp: emergency reset -- reset was asserted, deasserting...\n"
emsg_deassert_len   = . - emsg_deassert - 1
emsg_deassert_ok:   .asciz "gsp: emergency reset -- deassert verified\n"
emsg_deassert_ok_len = . - emsg_deassert_ok - 1
emsg_deassert_fail: .asciz "gsp: ERROR: emergency reset -- deassert verification timeout\n"
emsg_deassert_fail_len = . - emsg_deassert_fail - 1
emsg_mbox_clear:    .asciz "gsp: emergency reset -- clearing mailboxes...\n"
emsg_mbox_clear_len = . - emsg_mbox_clear - 1
emsg_cycle:         .asciz "gsp: emergency reset -- full reset cycle (assert+deassert)...\n"
emsg_cycle_len      = . - emsg_cycle - 1
emsg_cycle_assert_fail: .asciz "gsp: ERROR: emergency reset -- assert timeout during cycle\n"
emsg_cycle_assert_fail_len = . - emsg_cycle_assert_fail - 1
emsg_cycle_deassert_fail: .asciz "gsp: ERROR: emergency reset -- deassert timeout during cycle\n"
emsg_cycle_deassert_fail_len = . - emsg_cycle_deassert_fail - 1
emsg_ok:            .asciz "gsp: emergency reset -- complete, falcon recovered\n"
emsg_ok_len         = . - emsg_ok - 1
emsg_bar0_null:     .asciz "gsp: ERROR: BAR0 not mapped for emergency reset\n"
emsg_bar0_null_len  = . - emsg_bar0_null - 1

// ============================================================
// Text section -- gsp_emergency_reset
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// gsp_emergency_reset -- recover falcon from a failed boot
//
// 1. Reads FALCON_ENGINE to determine current state
// 2. If reset asserted (bit 0 = 1), deasserts with dsb sy + poll
// 3. Clears MAILBOX0 and MAILBOX1 to reset communication state
// 4. Performs a full reset cycle (assert then deassert) with timeouts
// 5. Returns 0 on success, negative on failure
//
// Returns: x0 = 0 success, -1 deassert verify timeout,
//          -2 cycle assert timeout, -3 cycle deassert timeout,
//          -4 BAR0 not mapped
// Clobbers: x0-x8, x16-x17
// ------------------------------------------------------------
.global gsp_emergency_reset
gsp_emergency_reset:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]

    // Block signals during critical MMIO
    mov     x0, #SIG_BLOCK
    adr     x1, .Lsigmask_emerg
    mov     x2, #0
    mov     x3, #8
    mov     x8, #SYS_RT_SIGPROCMASK
    svc     #0

    // Load BAR0 base
    adrp    x19, bar0_base
    add     x19, x19, :lo12:bar0_base
    ldr     x19, [x19]
    cbz     x19, .emerg_bar0_null

    // Compute FALCON_ENGINE address
    movz    x20, #0x03C0
    movk    x20, #0x0011, lsl #16
    add     x20, x19, x20             // x20 = &FALCON_ENGINE

    // Compute MAILBOX0 address
    movz    x23, #0x0040
    movk    x23, #0x0011, lsl #16
    add     x23, x19, x23             // x23 = &FALCON_MAILBOX0

    // Compute MAILBOX1 address
    movz    x24, #0x0044
    movk    x24, #0x0011, lsl #16
    add     x24, x19, x24             // x24 = &FALCON_MAILBOX1

    // ---- Step 1: Read current falcon state ----
    adrp    x1, emsg_start
    add     x1, x1, :lo12:emsg_start
    mov     x2, #emsg_start_len
    bl      falcon_print_msg

    dsb     sy
    isb
    ldr     w22, [x20]                // w22 = FALCON_ENGINE value

    // ---- Step 2: If reset asserted (bit 0 = 1), deassert it ----
    tst     w22, #1
    b.eq    .emerg_skip_deassert

    adrp    x1, emsg_deassert
    add     x1, x1, :lo12:emsg_deassert
    mov     x2, #emsg_deassert_len
    bl      falcon_print_msg

    mov     w0, #0
    str     w0, [x20]                 // deassert reset
    dsb     sy

    // Poll for DEASSERTED status with timeout
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_deassert_fail
    ldr     x21, [sp, #48]            // x21 = start_seconds

    dsb     sy
    isb
.emerg_poll_deassert:
    ldr     w0, [x20]
    and     w0, w0, #STATUS_MASK
    cmp     w0, #STATUS_DEASSERTED
    b.eq    .emerg_deassert_verified

    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_deassert_fail
    ldr     x0, [sp, #48]
    sub     x0, x0, x21
    cmp     x0, #FALCON_TIMEOUT_SECS
    b.ge    .emerg_deassert_fail

    dsb     ld
    isb
    mov     w0, #100
.emerg_deassert_delay:
    nop
    subs    w0, w0, #1
    b.ne    .emerg_deassert_delay

    b       .emerg_poll_deassert

.emerg_deassert_verified:
    adrp    x1, emsg_deassert_ok
    add     x1, x1, :lo12:emsg_deassert_ok
    mov     x2, #emsg_deassert_ok_len
    bl      falcon_print_msg

.emerg_skip_deassert:
    // ---- Step 3: Clear MAILBOX0 and MAILBOX1 ----
    adrp    x1, emsg_mbox_clear
    add     x1, x1, :lo12:emsg_mbox_clear
    mov     x2, #emsg_mbox_clear_len
    bl      falcon_print_msg

    mov     w0, #0
    str     w0, [x23]                 // MAILBOX0 = 0
    dsb     st
    str     w0, [x24]                 // MAILBOX1 = 0
    dsb     st

    // ---- Step 4: Full reset cycle (assert then deassert) ----
    adrp    x1, emsg_cycle
    add     x1, x1, :lo12:emsg_cycle
    mov     x2, #emsg_cycle_len
    bl      falcon_print_msg

    // 4a. Assert reset (write 1)
    mov     w0, #1
    str     w0, [x20]
    dsb     st

    // Poll for ASSERTED status
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_cycle_assert_fail
    ldr     x21, [sp, #48]

    dsb     sy
    isb
.emerg_poll_assert:
    ldr     w0, [x20]
    and     w0, w0, #STATUS_MASK
    cmp     w0, #STATUS_ASSERTED
    b.eq    .emerg_assert_ok

    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_cycle_assert_fail
    ldr     x0, [sp, #48]
    sub     x0, x0, x21
    cmp     x0, #FALCON_TIMEOUT_SECS
    b.ge    .emerg_cycle_assert_fail

    dsb     ld
    isb
    mov     w0, #100
.emerg_assert_delay:
    nop
    subs    w0, w0, #1
    b.ne    .emerg_assert_delay

    b       .emerg_poll_assert

.emerg_assert_ok:
    // 4b. Deassert reset (write 0)
    mov     w0, #0
    str     w0, [x20]
    dsb     st

    // Poll for DEASSERTED status
    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_cycle_deassert_fail
    ldr     x21, [sp, #48]

    dsb     sy
    isb
.emerg_poll_cycle_deassert:
    ldr     w0, [x20]
    and     w0, w0, #STATUS_MASK
    cmp     w0, #STATUS_DEASSERTED
    b.eq    .emerg_cycle_done

    mov     x0, #CLOCK_MONOTONIC
    add     x1, sp, #48
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    cmp     x0, #0
    b.lt    .emerg_cycle_deassert_fail
    ldr     x0, [sp, #48]
    sub     x0, x0, x21
    cmp     x0, #FALCON_TIMEOUT_SECS
    b.ge    .emerg_cycle_deassert_fail

    dsb     ld
    isb
    mov     w0, #100
.emerg_cycle_deassert_delay:
    nop
    subs    w0, w0, #1
    b.ne    .emerg_cycle_deassert_delay

    b       .emerg_poll_cycle_deassert

.emerg_cycle_done:
    adrp    x1, emsg_ok
    add     x1, x1, :lo12:emsg_ok
    mov     x2, #emsg_ok_len
    bl      falcon_print_msg

    mov     x0, #0
    b       .emerg_return

.emerg_deassert_fail:
    adrp    x1, emsg_deassert_fail
    add     x1, x1, :lo12:emsg_deassert_fail
    mov     x2, #emsg_deassert_fail_len
    bl      falcon_print_msg
    mov     x0, #-1
    b       .emerg_return

.emerg_cycle_assert_fail:
    // Try to deassert to leave falcon in a safe state
    mov     w0, #0
    str     w0, [x20]
    dsb     sy
    adrp    x1, emsg_cycle_assert_fail
    add     x1, x1, :lo12:emsg_cycle_assert_fail
    mov     x2, #emsg_cycle_assert_fail_len
    bl      falcon_print_msg
    mov     x0, #-2
    b       .emerg_return

.emerg_cycle_deassert_fail:
    // Re-assert to leave falcon in known held-in-reset state
    mov     w0, #1
    str     w0, [x20]
    dsb     sy
    adrp    x1, emsg_cycle_deassert_fail
    add     x1, x1, :lo12:emsg_cycle_deassert_fail
    mov     x2, #emsg_cycle_deassert_fail_len
    bl      falcon_print_msg
    mov     x0, #-3
    b       .emerg_return

.emerg_bar0_null:
    adrp    x1, emsg_bar0_null
    add     x1, x1, :lo12:emsg_bar0_null
    mov     x2, #emsg_bar0_null_len
    bl      falcon_print_msg
    mov     x0, #-4

.emerg_return:
    // Save return value across sigprocmask call
    mov     x19, x0

    // Unblock signals
    mov     x0, #SIG_UNBLOCK
    adr     x1, .Lsigmask_emerg
    mov     x2, #0
    mov     x3, #8
    mov     x8, #SYS_RT_SIGPROCMASK
    svc     #0

    mov     x0, x19

    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// Signal mask for emergency reset (same as falcon_reset)
.align 3
.Lsigmask_emerg:
    .quad   0x4003
