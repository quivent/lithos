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
// The polling loops include a timeout counter to avoid infinite hangs
// if the hardware is unresponsive.  Timeout is ~100M iterations
// (~1-2 seconds on Grace at ~3GHz).
//
// Calling convention: ARM64 AAPCS.
// Returns: x0 = 0 success, -1 timeout or BAR0 not mapped.

// ---- Syscall numbers ----
.equ SYS_WRITE,     64

// ---- Falcon register offset ----
.equ FALCON_ENGINE,     0x1103C0

// ---- Reset status field ----
// bits[10:8] of FALCON_ENGINE register
.equ STATUS_MASK,       0x700         // bits[10:8]
.equ STATUS_ASSERTED,   0x000         // 0b000 << 8
.equ STATUS_DEASSERTED, 0x200         // 0b010 << 8

// ---- Timeout: ~100M iterations ----
.equ POLL_TIMEOUT,      100000000

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
msg_reset_timeout:  .asciz "gsp: ERROR: falcon reset timeout\n"
msg_reset_timeout_len = . - msg_reset_timeout - 1
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
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

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

    // ---- Step 2: Poll bits[10:8] == 0 (ASSERTED) ----
    movz    x1, #0xE100              // POLL_TIMEOUT = 100000000 = 0x05F5E100
    movk    x1, #0x05F5, lsl #16
.poll_asserted:
    ldr     w0, [x20]                 // read FALCON_ENGINE
    and     w0, w0, #STATUS_MASK      // isolate bits[10:8]
    cmp     w0, #STATUS_ASSERTED      // == 0?
    b.eq    .asserted_ok
    subs    x1, x1, #1
    b.ne    .poll_asserted
    b       .falcon_timeout           // exhausted counter

.asserted_ok:
    adrp    x1, msg_reset_asserted
    add     x1, x1, :lo12:msg_reset_asserted
    mov     x2, #msg_reset_asserted_len
    bl      falcon_print_msg

    // ---- Step 3: Deassert reset (bit 0 = 0) ----
    mov     w0, #0
    str     w0, [x20]                 // BAR0+0x1103C0 <- 0

    // ---- Step 4: Poll bits[10:8] == 2 (DEASSERTED) ----
    movz    x1, #0xE100              // POLL_TIMEOUT = 100000000 = 0x05F5E100
    movk    x1, #0x05F5, lsl #16
.poll_deasserted:
    ldr     w0, [x20]                 // read FALCON_ENGINE
    and     w0, w0, #STATUS_MASK      // isolate bits[10:8]
    cmp     w0, #STATUS_DEASSERTED    // == 0x200?
    b.eq    .deasserted_ok
    subs    x1, x1, #1
    b.ne    .poll_deasserted
    b       .falcon_timeout

.deasserted_ok:
    adrp    x1, msg_reset_done
    add     x1, x1, :lo12:msg_reset_done
    mov     x2, #msg_reset_done_len
    bl      falcon_print_msg

    // ---- Success ----
    mov     x0, #0
    b       .falcon_return

.falcon_timeout:
    adrp    x1, msg_reset_timeout
    add     x1, x1, :lo12:msg_reset_timeout
    mov     x2, #msg_reset_timeout_len
    bl      falcon_print_msg
    mov     x0, #-1
    b       .falcon_return

.falcon_bar0_null:
    adrp    x1, falcon_msg_bar0_null
    add     x1, x1, :lo12:falcon_msg_bar0_null
    mov     x2, #falcon_msg_bar0_null_len
    bl      falcon_print_msg
    mov     x0, #-1

.falcon_return:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
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
