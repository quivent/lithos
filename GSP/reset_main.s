// reset_main.s -- Minimal driver for gsp_emergency_reset
//
// Standalone entry point that:
//   1. Calls bar_map_init to map BAR0/BAR4
//   2. Calls gsp_emergency_reset to recover falcon
//   3. Prints success or failure
//   4. Exits with 0 on success, 1 on failure
//
// Build:
//   as -o reset_main.o reset_main.s
//   as -o bar_map.o bar_map.s
//   as -o gsp_common.o gsp_common.s
//   as -o falcon_reset.o falcon_reset.s
//   ld -o gsp_reset reset_main.o bar_map.o gsp_common.o falcon_reset.o

.equ SYS_WRITE, 64
.equ SYS_EXIT,  93

.extern bar_map_init
.extern gsp_emergency_reset

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_banner:     .asciz "gsp_reset: emergency falcon recovery tool\n"
msg_banner_len  = . - msg_banner - 1
msg_bar_fail:   .asciz "gsp_reset: ERROR: bar_map_init failed\n"
msg_bar_fail_len = . - msg_bar_fail - 1
msg_ok:         .asciz "gsp_reset: SUCCESS -- falcon recovered\n"
msg_ok_len      = . - msg_ok - 1
msg_fail:       .asciz "gsp_reset: FAILED -- emergency reset returned error\n"
msg_fail_len    = . - msg_fail - 1

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // Print banner
    mov     x0, #2                    // stderr
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Step 1: bar_map_init
    bl      bar_map_init
    cmp     x0, #0
    b.lt    .reset_bar_fail

    // Step 2: gsp_emergency_reset
    bl      gsp_emergency_reset
    cmp     x0, #0
    b.lt    .reset_fail

    // Success
    mov     x0, #2
    adrp    x1, msg_ok
    add     x1, x1, :lo12:msg_ok
    mov     x2, #msg_ok_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.reset_bar_fail:
    mov     x0, #2
    adrp    x1, msg_bar_fail
    add     x1, x1, :lo12:msg_bar_fail
    mov     x2, #msg_bar_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.reset_fail:
    mov     x0, #2
    adrp    x1, msg_fail
    add     x1, x1, :lo12:msg_fail
    mov     x2, #msg_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
