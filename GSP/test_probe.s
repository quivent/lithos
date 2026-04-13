// test_probe.s -- Minimal sanity test for GSP boot steps 1 & 2
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
// Link: as test_probe.s -o test_probe.o &&
//       ld test_probe.o bar_map.o pmc_check.o -o gsp_probe
//
// Calling convention: ARM64 AAPCS.  Raw Linux syscalls, no libc.

// ---- Syscall numbers (aarch64) ----
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93

// ---- File descriptors ----
.equ FD_STDOUT,     1
.equ FD_STDERR,     2

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_ok:         .ascii "GSP probe: Hopper GPU detected\n"
msg_ok_len    = . - msg_ok

msg_fail:       .ascii "GSP probe: FAILED\n"
msg_fail_len  = . - msg_fail

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // ---- Step 1: map BAR0 (and BAR4) via sysfs ----
    bl      bar_map_init
    cmp     x0, #0
    b.ne    .probe_fail

    // ---- Step 2: read PMC_BOOT_0, verify Hopper ----
    bl      pmc_check
    cmp     x0, #0
    b.ne    .probe_fail

    // ---- Success: print to stdout, exit 0 ----
    mov     x0, #FD_STDOUT
    adrp    x1, msg_ok
    add     x1, x1, :lo12:msg_ok
    mov     x2, #msg_ok_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.probe_fail:
    // ---- Failure: print to stderr, exit 1 ----
    mov     x0, #FD_STDERR
    adrp    x1, msg_fail
    add     x1, x1, :lo12:msg_fail
    mov     x2, #msg_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
