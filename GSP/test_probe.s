// test_probe.s -- Sanity test for GSP boot steps 1 & 2
//
// Read-only probe: verifies that we can
//   1. mmap BAR0 via /sys/bus/pci/.../resource0  (bar_map_init)
//   2. Read PMC_BOOT_0 and confirm Hopper arch  (pmc_check)
//
// This test does NOT modify GPU state.  It is safe to run before the
// full 9-step boot sequence.  However it still requires:
//   - root (to open /sys/.../resource0 for mmap)
//   - nvidia.ko unbound from the target GPU
//
// Build: make test   (produces gsp_probe)
//   Or manually:
//     as test_probe.s -o test_probe.o
//     ld test_probe.o bar_map.o pmc_check.o gsp_common.o -o gsp_probe
//
// Exit codes:
//   0 = all checks passed
//   1 = bar_map_init failed (cannot open/mmap sysfs resource files)
//   2 = pmc_check failed (not Hopper or BAR0 not mapped)
//
// Calling convention: ARM64 AAPCS.  Raw Linux syscalls, no libc.

// ---- Syscall numbers (aarch64) ----
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93

// ---- File descriptors ----
.equ FD_STDOUT,     1
.equ FD_STDERR,     2

// ============================================================
// External symbols
// ============================================================
.extern bar_map_init
.extern pmc_check
.extern bar0_base
.extern bar4_base
.extern bar4_phys

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_banner:     .ascii "=== GSP probe: read-only hardware check ===\n"
msg_banner_len = . - msg_banner

msg_step1_ok:   .ascii "  [1/2] bar_map_init: OK (BAR0 + BAR4 mapped)\n"
msg_step1_ok_len = . - msg_step1_ok
msg_step1_fail: .ascii "  [1/2] bar_map_init: FAILED\n"
msg_step1_fail_len = . - msg_step1_fail

msg_step2_ok:   .ascii "  [2/2] pmc_check:    OK (Hopper GPU detected)\n"
msg_step2_ok_len = . - msg_step2_ok
msg_step2_fail: .ascii "  [2/2] pmc_check:    FAILED (not Hopper or BAR0 null)\n"
msg_step2_fail_len = . - msg_step2_fail

msg_bar0_nz:    .ascii "  check: bar0_base != 0: OK\n"
msg_bar0_nz_len = . - msg_bar0_nz
msg_bar0_zero:  .ascii "  check: bar0_base != 0: FAILED (bar0_base is zero after bar_map_init)\n"
msg_bar0_zero_len = . - msg_bar0_zero

msg_bar4_nz:    .ascii "  check: bar4_base != 0: OK\n"
msg_bar4_nz_len = . - msg_bar4_nz
msg_bar4_zero:  .ascii "  check: bar4_base != 0: FAILED (bar4_base is zero after bar_map_init)\n"
msg_bar4_zero_len = . - msg_bar4_zero

msg_phys_nz:    .ascii "  check: bar4_phys != 0: OK\n"
msg_phys_nz_len = . - msg_phys_nz
msg_phys_zero:  .ascii "  check: bar4_phys != 0: FAILED (bar4_phys is zero -- sysfs parse broken?)\n"
msg_phys_zero_len = . - msg_phys_zero

msg_pass:       .ascii "=== GSP probe: PASSED ===\n"
msg_pass_len  = . - msg_pass

msg_fail:       .ascii "=== GSP probe: FAILED ===\n"
msg_fail_len  = . - msg_fail

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // ---- Banner ----
    mov     x0, #FD_STDERR
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Step 1: map BAR0 (and BAR4) via sysfs ----
    bl      bar_map_init
    cmp     x0, #0
    b.ne    .fail_step1

    // Print step 1 OK
    mov     x0, #FD_STDERR
    adrp    x1, msg_step1_ok
    add     x1, x1, :lo12:msg_step1_ok
    mov     x2, #msg_step1_ok_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Post-condition checks for step 1 ----
    // Verify bar0_base was actually populated (not left zero)
    adrp    x9, bar0_base
    add     x9, x9, :lo12:bar0_base
    ldr     x9, [x9]
    cbz     x9, .fail_bar0_zero
    mov     x0, #FD_STDERR
    adrp    x1, msg_bar0_nz
    add     x1, x1, :lo12:msg_bar0_nz
    mov     x2, #msg_bar0_nz_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Verify bar4_base was populated
    adrp    x9, bar4_base
    add     x9, x9, :lo12:bar4_base
    ldr     x9, [x9]
    cbz     x9, .fail_bar4_zero
    mov     x0, #FD_STDERR
    adrp    x1, msg_bar4_nz
    add     x1, x1, :lo12:msg_bar4_nz
    mov     x2, #msg_bar4_nz_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Verify bar4_phys was populated (sysfs resource parse worked)
    adrp    x9, bar4_phys
    add     x9, x9, :lo12:bar4_phys
    ldr     x9, [x9]
    cbz     x9, .fail_phys_zero
    mov     x0, #FD_STDERR
    adrp    x1, msg_phys_nz
    add     x1, x1, :lo12:msg_phys_nz
    mov     x2, #msg_phys_nz_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Step 2: read PMC_BOOT_0, verify Hopper ----
    bl      pmc_check
    cmp     x0, #0
    b.ne    .fail_step2

    // Print step 2 OK
    mov     x0, #FD_STDERR
    adrp    x1, msg_step2_ok
    add     x1, x1, :lo12:msg_step2_ok
    mov     x2, #msg_step2_ok_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- All passed ----
    mov     x0, #FD_STDOUT
    adrp    x1, msg_pass
    add     x1, x1, :lo12:msg_pass
    mov     x2, #msg_pass_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Failure paths -- each prints a specific message, then exits
// ============================================================
.fail_step1:
    mov     x0, #FD_STDERR
    adrp    x1, msg_step1_fail
    add     x1, x1, :lo12:msg_step1_fail
    mov     x2, #msg_step1_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .exit_fail_1

.fail_bar0_zero:
    mov     x0, #FD_STDERR
    adrp    x1, msg_bar0_zero
    add     x1, x1, :lo12:msg_bar0_zero
    mov     x2, #msg_bar0_zero_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .exit_fail_1

.fail_bar4_zero:
    mov     x0, #FD_STDERR
    adrp    x1, msg_bar4_zero
    add     x1, x1, :lo12:msg_bar4_zero
    mov     x2, #msg_bar4_zero_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .exit_fail_1

.fail_phys_zero:
    mov     x0, #FD_STDERR
    adrp    x1, msg_phys_zero
    add     x1, x1, :lo12:msg_phys_zero
    mov     x2, #msg_phys_zero_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .exit_fail_1

.fail_step2:
    mov     x0, #FD_STDERR
    adrp    x1, msg_step2_fail
    add     x1, x1, :lo12:msg_step2_fail
    mov     x2, #msg_step2_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .exit_fail_2

.exit_fail_1:
    mov     x0, #FD_STDERR
    adrp    x1, msg_fail
    add     x1, x1, :lo12:msg_fail
    mov     x2, #msg_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.exit_fail_2:
    mov     x0, #FD_STDERR
    adrp    x1, msg_fail
    add     x1, x1, :lo12:msg_fail
    mov     x2, #msg_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0
