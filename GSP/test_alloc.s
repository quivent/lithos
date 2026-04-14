// test_alloc.s -- Unit tests for hbm_alloc arithmetic (no hardware needed)
//
// Verifies the bump allocator logic with known values:
//   1. hbm_alloc_init with known base/phys/offset/size
//   2. hbm_alloc(2MB)   -> returned gpu_va == phys_base + offset
//   3. hbm_alloc(2MB)   -> returned gpu_va == phys_base + offset + 2MB
//   4. hbm_alloc(0)     -> should return 0 (zero-size guard)
//   5. hbm_alloc(huge)  -> should return -1 (OOM)
//
// All output goes to stderr (fd 2).
// Exit 0 if all pass, exit 1 on any failure.
//
// Build: make test_alloc
//   Or manually:
//     as test_alloc.s -o test_alloc.o
//     ld test_alloc.o hbm_alloc.o gsp_common.o -o test_alloc

// ---- Syscall numbers (aarch64) ----
.equ SYS_WRITE, 64
.equ SYS_EXIT,  93

// ---- File descriptors ----
.equ FD_STDERR, 2

// ---- Test parameters ----
// base (virtual)  = 0x200000000
// phys            = 0x100000000
// offset          = 0x400000     (4MB, already 2MB-aligned)
// BAR4 window     = 256MB = 0x10000000
// alloc size      = 0x200000     (2MB)

// ============================================================
// External symbols
// ============================================================
.extern hbm_alloc_init
.extern hbm_alloc

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_banner:     .ascii "=== test_alloc: hbm_alloc arithmetic tests ===\n"
msg_banner_len = . - msg_banner

msg_t1_pass:    .ascii "  [1/4] hbm_alloc(2MB) first:  PASS\n"
msg_t1_pass_len = . - msg_t1_pass
msg_t1_fail:    .ascii "  [1/4] hbm_alloc(2MB) first:  FAIL\n"
msg_t1_fail_len = . - msg_t1_fail

msg_t2_pass:    .ascii "  [2/4] hbm_alloc(2MB) second: PASS\n"
msg_t2_pass_len = . - msg_t2_pass
msg_t2_fail:    .ascii "  [2/4] hbm_alloc(2MB) second: FAIL\n"
msg_t2_fail_len = . - msg_t2_fail

msg_t3_pass:    .ascii "  [3/4] hbm_alloc(0) zero:     PASS\n"
msg_t3_pass_len = . - msg_t3_pass
msg_t3_fail:    .ascii "  [3/4] hbm_alloc(0) zero:     FAIL\n"
msg_t3_fail_len = . - msg_t3_fail

msg_t4_pass:    .ascii "  [4/4] hbm_alloc(huge) OOM:   PASS\n"
msg_t4_pass_len = . - msg_t4_pass
msg_t4_fail:    .ascii "  [4/4] hbm_alloc(huge) OOM:   FAIL\n"
msg_t4_fail_len = . - msg_t4_fail

msg_all_pass:   .ascii "=== test_alloc: PASSED ===\n"
msg_all_pass_len = . - msg_all_pass

msg_some_fail:  .ascii "=== test_alloc: FAILED ===\n"
msg_some_fail_len = . - msg_some_fail

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // Save frame
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // x19 = failure count
    mov     x19, #0

    // ---- Banner ----
    mov     x0, #FD_STDERR
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Initialize allocator ----
    // x0 = virtual base  = 0x200000000
    // x1 = physical base = 0x100000000
    // x2 = initial offset = 0x400000
    mov     x0, #0x2
    lsl     x0, x0, #32            // x0 = 0x200000000
    mov     x1, #0x1
    lsl     x1, x1, #32            // x1 = 0x100000000
    mov     x2, #0x400000           // x2 = 0x400000 (4MB offset)
    bl      hbm_alloc_init

    // ============================================================
    // Test 1: hbm_alloc(2MB), expect x1 (gpu_va) = 0x100400000
    // ============================================================
    mov     x0, #0x200000           // 2MB
    bl      hbm_alloc
    // x1 = gpu_va, should be phys_base + offset = 0x100000000 + 0x400000
    mov     x20, x1                 // save gpu_va

    // Build expected: 0x100400000
    mov     x21, #0x1
    lsl     x21, x21, #32          // 0x100000000
    mov     x22, #0x400000
    add     x21, x21, x22          // 0x100400000

    cmp     x20, x21
    b.ne    .t1_fail

    // PASS
    mov     x0, #FD_STDERR
    adrp    x1, msg_t1_pass
    add     x1, x1, :lo12:msg_t1_pass
    mov     x2, #msg_t1_pass_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .t2

.t1_fail:
    add     x19, x19, #1
    mov     x0, #FD_STDERR
    adrp    x1, msg_t1_fail
    add     x1, x1, :lo12:msg_t1_fail
    mov     x2, #msg_t1_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ============================================================
    // Test 2: hbm_alloc(2MB) again, expect x1 = 0x100600000
    //   (phys_base + offset + 2MB = 0x100000000 + 0x400000 + 0x200000)
    // ============================================================
.t2:
    mov     x0, #0x200000           // 2MB
    bl      hbm_alloc
    mov     x20, x1                 // save gpu_va

    // Build expected: 0x100600000
    mov     x21, #0x1
    lsl     x21, x21, #32          // 0x100000000
    mov     x22, #0x600000
    add     x21, x21, x22          // 0x100600000

    cmp     x20, x21
    b.ne    .t2_fail

    // PASS
    mov     x0, #FD_STDERR
    adrp    x1, msg_t2_pass
    add     x1, x1, :lo12:msg_t2_pass
    mov     x2, #msg_t2_pass_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .t3

.t2_fail:
    add     x19, x19, #1
    mov     x0, #FD_STDERR
    adrp    x1, msg_t2_fail
    add     x1, x1, :lo12:msg_t2_fail
    mov     x2, #msg_t2_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ============================================================
    // Test 3: hbm_alloc(0), zero-size guard
    //   The code's .alloc_zero path returns cpu_addr = base + bump
    //   (non-zero), not 0.  This test checks for x0 == 0 per the
    //   specification -- it will FAIL against the current code,
    //   surfacing the discrepancy as a known issue.
    // ============================================================
.t3:
    mov     x0, #0                  // size = 0
    bl      hbm_alloc
    // x0 = cpu_addr; spec says should be 0
    cmp     x0, #0
    b.ne    .t3_fail

    // PASS
    mov     x0, #FD_STDERR
    adrp    x1, msg_t3_pass
    add     x1, x1, :lo12:msg_t3_pass
    mov     x2, #msg_t3_pass_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .t4

.t3_fail:
    add     x19, x19, #1
    mov     x0, #FD_STDERR
    adrp    x1, msg_t3_fail
    add     x1, x1, :lo12:msg_t3_fail
    mov     x2, #msg_t3_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ============================================================
    // Test 4: hbm_alloc(huge) -- exceed 256MB window, expect x0 = -1
    //   We already consumed 4MB (offset) + 4MB (2x 2MB allocs) = 8MB.
    //   Window is 256MB.  Request 0x10000000 (256MB) to overflow.
    // ============================================================
.t4:
    mov     x0, #0x10000000         // 256MB -- exceeds remaining space
    bl      hbm_alloc
    // x0 should be -1 (0xFFFFFFFFFFFFFFFF)
    cmn     x0, #1                  // compare x0 with -1
    b.ne    .t4_fail

    // PASS
    mov     x0, #FD_STDERR
    adrp    x1, msg_t4_pass
    add     x1, x1, :lo12:msg_t4_pass
    mov     x2, #msg_t4_pass_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .done

.t4_fail:
    add     x19, x19, #1
    mov     x0, #FD_STDERR
    adrp    x1, msg_t4_fail
    add     x1, x1, :lo12:msg_t4_fail
    mov     x2, #msg_t4_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ============================================================
    // Summary
    // ============================================================
.done:
    cmp     x19, #0
    b.ne    .exit_fail

    // All passed
    mov     x0, #FD_STDERR
    adrp    x1, msg_all_pass
    add     x1, x1, :lo12:msg_all_pass
    mov     x2, #msg_all_pass_len
    mov     x8, #SYS_WRITE
    svc     #0

    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.exit_fail:
    mov     x0, #FD_STDERR
    adrp    x1, msg_some_fail
    add     x1, x1, :lo12:msg_some_fail
    mov     x2, #msg_some_fail_len
    mov     x8, #SYS_WRITE
    svc     #0

    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
