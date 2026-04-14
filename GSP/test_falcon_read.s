// test_falcon_read.s -- Read-only Falcon status check (safe, no writes)
//
// Maps BAR0 via bar_map_init, then reads two Falcon registers and
// prints their values.  No GPU state is modified.
//
//   1. Call bar_map_init to get BAR0
//   2. Read FALCON_ENGINE (BAR0 + 0x1103C0), print raw hex value
//   3. Check bit 0: if 1, "Falcon: RESET ASSERTED"; if 0, "Falcon: RUNNING"
//   4. Read FALCON_CPUCTL (BAR0 + 0x110100), print raw hex value
//
// Requires root and nvidia.ko unbound from target GPU.
//
// Build: make test_falcon
//   Or manually:
//     as test_falcon_read.s -o test_falcon_read.o
//     ld test_falcon_read.o bar_map.o gsp_common.o -o test_falcon
//
// Exit codes:
//   0 = success (registers read and printed)
//   1 = bar_map_init failed

// ---- Syscall numbers (aarch64) ----
.equ SYS_WRITE, 64
.equ SYS_EXIT,  93

// ---- File descriptors ----
.equ FD_STDERR, 2

// ---- Falcon register offsets from BAR0 ----
.equ FALCON_ENGINE_OFF, 0x1103C0
.equ FALCON_CPUCTL_OFF, 0x110100

// ============================================================
// External symbols
// ============================================================
.extern bar_map_init
.extern bar0_base

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_banner:     .ascii "=== test_falcon_read: Falcon status probe ===\n"
msg_banner_len = . - msg_banner

msg_bar_fail:   .ascii "  bar_map_init: FAILED\n"
msg_bar_fail_len = . - msg_bar_fail

msg_engine_pre: .ascii "  FALCON_ENGINE (0x1103C0) = 0x"
msg_engine_pre_len = . - msg_engine_pre

msg_cpuctl_pre: .ascii "  FALCON_CPUCTL (0x110100) = 0x"
msg_cpuctl_pre_len = . - msg_cpuctl_pre

msg_newline:    .ascii "\n"
msg_newline_len = . - msg_newline

msg_reset:      .ascii "  Falcon: RESET ASSERTED\n"
msg_reset_len = . - msg_reset

msg_running:    .ascii "  Falcon: RUNNING\n"
msg_running_len = . - msg_running

// Hex conversion buffer (8 chars for 32-bit value)
.align 3
hex_buf:        .ascii "00000000"
hex_buf_len = . - hex_buf

hex_digits:     .ascii "0123456789ABCDEF"

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ---------------------------------------------------------------
// to_hex32 -- convert w0 to 8-char hex string in hex_buf
// Input:  w0 = 32-bit value
// Output: hex_buf filled with ASCII hex
// Clobbers: x1, x2, x3, x4, x5
// ---------------------------------------------------------------
to_hex32:
    adrp    x1, hex_buf
    add     x1, x1, :lo12:hex_buf
    adrp    x5, hex_digits
    add     x5, x5, :lo12:hex_digits
    mov     x2, #7                  // index into hex_buf (start at rightmost)
.hex_loop:
    and     x3, x0, #0xF           // low nibble
    ldrb    w4, [x5, x3]           // hex digit
    strb    w4, [x1, x2]           // store in buffer
    lsr     x0, x0, #4             // shift right by 4
    subs    x2, x2, #1
    b.ge    .hex_loop
    ret

// ---------------------------------------------------------------
// print_hex_buf -- write hex_buf to stderr
// ---------------------------------------------------------------
print_hex_buf:
    mov     x0, #FD_STDERR
    adrp    x1, hex_buf
    add     x1, x1, :lo12:hex_buf
    mov     x2, #hex_buf_len
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ---------------------------------------------------------------
// print_newline -- write newline to stderr
// ---------------------------------------------------------------
print_newline:
    mov     x0, #FD_STDERR
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #msg_newline_len
    mov     x8, #SYS_WRITE
    svc     #0
    ret

.global _start
_start:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    // ---- Banner ----
    mov     x0, #FD_STDERR
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Map BAR0 ----
    bl      bar_map_init
    cmp     x0, #0
    b.ne    .bar_fail

    // Load BAR0 base address
    adrp    x19, bar0_base
    add     x19, x19, :lo12:bar0_base
    ldr     x19, [x19]              // x19 = BAR0 virtual address

    // ---- Read FALCON_ENGINE register ----
    // Build offset 0x1103C0
    movz    x20, #0x03C0
    movk    x20, #0x0011, lsl #16   // x20 = 0x1103C0
    ldr     w20, [x19, x20]         // w20 = FALCON_ENGINE value

    // Print "  FALCON_ENGINE (0x1103C0) = 0x"
    mov     x0, #FD_STDERR
    adrp    x1, msg_engine_pre
    add     x1, x1, :lo12:msg_engine_pre
    mov     x2, #msg_engine_pre_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Convert and print value
    mov     w0, w20
    bl      to_hex32
    bl      print_hex_buf
    bl      print_newline

    // ---- Check bit 0 of FALCON_ENGINE ----
    tst     w20, #1
    b.ne    .falcon_in_reset

    // Bit 0 == 0: RUNNING
    mov     x0, #FD_STDERR
    adrp    x1, msg_running
    add     x1, x1, :lo12:msg_running
    mov     x2, #msg_running_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .read_cpuctl

.falcon_in_reset:
    // Bit 0 == 1: RESET ASSERTED
    mov     x0, #FD_STDERR
    adrp    x1, msg_reset
    add     x1, x1, :lo12:msg_reset
    mov     x2, #msg_reset_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Read FALCON_CPUCTL register ----
.read_cpuctl:
    // Build offset 0x110100
    movz    x20, #0x0100
    movk    x20, #0x0011, lsl #16   // x20 = 0x110100
    ldr     w20, [x19, x20]         // w20 = FALCON_CPUCTL value

    // Print "  FALCON_CPUCTL (0x110100) = 0x"
    mov     x0, #FD_STDERR
    adrp    x1, msg_cpuctl_pre
    add     x1, x1, :lo12:msg_cpuctl_pre
    mov     x2, #msg_cpuctl_pre_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Convert and print value
    mov     w0, w20
    bl      to_hex32
    bl      print_hex_buf
    bl      print_newline

    // ---- Success: exit 0 ----
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.bar_fail:
    mov     x0, #FD_STDERR
    adrp    x1, msg_bar_fail
    add     x1, x1, :lo12:msg_bar_fail
    mov     x2, #msg_bar_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
