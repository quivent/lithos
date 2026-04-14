// GSP/poll_lockdown.s -- Step 8: Poll for PRIV_LOCKDOWN release
//
// Translates gsp_wait_for_lockdown_release() from kernel/lithos_gsp.c
// to raw ARM64. No libc.
//
// Source: _kgspLockdownReleasedOrFmcError() -- kernel_gsp_gh100.c:498
//
// After FSP authorizes the FMC and releases the target mask, the
// RISCV_BR_PRIV_LOCKDOWN bit (bit 13) in NV_PFALCON_FALCON_HWCFG2
// clears. We poll until clear or timeout.
//
// Additionally checks MAILBOX0 for FMC error codes: if MAILBOX0 is
// non-zero and does not match fmc_params_pa[31:0], FMC has reported
// an error.
//
// Inputs:
//   x0 = bar0 base (mmap'd BAR0 virtual address)
//   x1 = fmc_params_pa (to distinguish FMC error from normal boot state)
//
// Returns:
//   x0 = 0 on success (lockdown released)
//   x0 = -1 on timeout (~30 seconds)
//   x0 = -2 on FMC error (error code in MAILBOX0)
//
// Timing: uses clock_gettime(CLOCK_MONOTONIC) via raw syscall.
//   SYS_clock_gettime = 113 on aarch64.
//   30-second timeout matches NVIDIA's GPU_TIMEOUT_DEFAULT.
//
// Build:
//   as -o poll_lockdown.o poll_lockdown.s

// ---- BAR0 register offsets ----
// These exceed str immediate range, so loaded into a register.
//   NV_PFALCON_FALCON_HWCFG2: BAR0+0x1100F4
//   NV_PGSP_FALCON_MAILBOX0:  BAR0+0x110040
.equ PRIV_LOCKDOWN_BIT,     13          // bit 13 = RISCV_BR_PRIV_LOCKDOWN

// ---- Syscall numbers (aarch64) ----
.equ SYS_CLOCK_GETTIME,     113
.equ CLOCK_MONOTONIC,       1

// ---- Timeout ----
.equ TIMEOUT_SECS,          30

.text
.globl gsp_poll_lockdown
.type  gsp_poll_lockdown, %function
.balign 4

gsp_poll_lockdown:
    // Prologue: save callee-saved regs and allocate timespec on stack.
    // Stack layout:
    //   sp+0..15:  saved x29, x30
    //   sp+16..31: saved x19, x20
    //   sp+32..47: saved x21, x22
    //   sp+48..63: struct timespec { tv_sec(8), tv_nsec(8) }
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    str x23, [sp, #48]

    mov x19, x0                         // x19 = bar0 base
    mov x20, x1                         // x20 = fmc_params_pa (full 64-bit)

    // Pre-compute BAR0 offset registers (too large for immediate str/ldr)
    mov x22, #0x0040
    movk x22, #0x0011, lsl #16          // x22 = 0x110040 (FALCON_MAILBOX0)
    mov x23, #0x00F4
    movk x23, #0x0011, lsl #16          // x23 = 0x1100F4 (FALCON_HWCFG2)

    // ----------------------------------------------------------------
    // Read start time via clock_gettime(CLOCK_MONOTONIC, &ts)
    // ----------------------------------------------------------------
    mov x0, #CLOCK_MONOTONIC
    add x1, sp, #64                     // timespec at sp+64
    mov x8, #SYS_CLOCK_GETTIME
    svc #0
    cmp x0, #0
    b.lt .timeout                        // clock_gettime failed -- treat as timeout
    ldr x21, [sp, #64]                  // x21 = start_seconds

.poll_loop:
    // ----------------------------------------------------------------
    // Check MAILBOX0 for FMC error code
    // If MAILBOX0 != 0 and MAILBOX0 != fmc_params_pa[31:0] -> error
    // Source: gsp_wait_for_lockdown_release() in lithos_gsp.c
    // ----------------------------------------------------------------
    ldr w0, [x19, x22]                  // read BAR0+0x110040
    cbz w0, .check_lockdown             // MAILBOX0 == 0 -> skip error check
    mov w1, w20                         // w1 = fmc_params_pa[31:0]
    cmp w0, w1
    b.eq .check_lockdown                // matches boot args PA -> not an error
    // FMC error: MAILBOX0 contains an error code
    mov x0, #-2
    b .done

.check_lockdown:
    // ----------------------------------------------------------------
    // Read HWCFG2 and check bit 13 (RISCV_BR_PRIV_LOCKDOWN)
    // bit 13 = 1 -> still locked, keep polling
    // bit 13 = 0 -> lockdown released, success
    // Source: NV_PFALCON_FALCON_HWCFG2 at BAR0+0x1100F4
    // ----------------------------------------------------------------
    ldr w0, [x19, x23]                  // read BAR0+0x1100F4
    tbz w0, #PRIV_LOCKDOWN_BIT, .success // bit 13 clear -> lockdown released

    // ----------------------------------------------------------------
    // Check timeout: read current time, compare to start + 30s
    // ----------------------------------------------------------------
    mov x0, #CLOCK_MONOTONIC
    add x1, sp, #64
    mov x8, #SYS_CLOCK_GETTIME
    svc #0
    cmp x0, #0
    b.lt .timeout                        // clock_gettime failed -- treat as timeout
    ldr x0, [sp, #64]                   // current_seconds
    sub x0, x0, x21                     // elapsed seconds
    cmp x0, #TIMEOUT_SECS
    b.ge .timeout

    // Brief yield between polls (~100ns).
    // The MMIO load itself takes ~microseconds over C2C.
    // ISB serializes and a small NOP loop approximates the
    // usleep_range(100, 200) from the C implementation.
    isb
    mov w0, #100
.delay_loop:
    nop
    subs w0, w0, #1
    b.ne .delay_loop

    b .poll_loop

.success:
    mov x0, #0                          // return 0 = success
    b .done

.timeout:
    mov x0, #-1                         // return -1 = timeout

.done:
    // Epilogue: restore callee-saved regs
    ldr x23, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

.size gsp_poll_lockdown, . - gsp_poll_lockdown
