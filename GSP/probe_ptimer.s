// probe_ptimer.s — Standalone ARM64 PTIMER probe for NVIDIA GPU
// Reads GPU PTIMER registers via BAR0 mmap, measures tick rate.
// Build: as -o probe_ptimer.o probe_ptimer.s && ld -o probe_ptimer probe_ptimer.o
// Run:   sudo ./probe_ptimer

.equ SYS_WRITE,            64
.equ SYS_OPENAT,           56
.equ SYS_CLOSE,            57
.equ SYS_MMAP,             222
.equ SYS_MUNMAP,           215
.equ SYS_EXIT,             93
.equ SYS_CLOCK_GETTIME,    113

.equ AT_FDCWD,             -100
.equ O_RDONLY,              0
.equ PROT_READ,            1
.equ MAP_SHARED,           1
.equ CLOCK_MONOTONIC,      1

.equ STDOUT,               1

// BAR0 offsets
.equ PTIMER_PRI_TIMEOUT,   0x9080
.equ PTIMER_TIME_0,        0x9400     // low 32 bits (ns)
.equ PTIMER_TIME_1,        0x9410     // high 32 bits (ns)

.equ MAP_SIZE,             0x10000    // 64K is enough to cover 0x9xxx

.equ NUM_SAMPLES,          10

.global _start
.text

// ============================================================
// _start
// ============================================================
_start:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // --- Print banner ---
    adr     x1, banner_str
    mov     x2, #banner_len
    bl      print_stdout

    // --- Open BAR0 resource file ---
    mov     x0, #AT_FDCWD
    adr     x1, bar0_path
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    open_fail
    mov     x19, x0                 // x19 = fd

    // --- mmap BAR0 ---
    mov     x0, #0                  // addr = NULL
    mov     x1, #MAP_SIZE           // length
    mov     x2, #PROT_READ          // prot
    mov     x3, #MAP_SHARED         // flags
    mov     x4, x19                 // fd
    mov     x5, #0                  // offset = 0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    mmap_fail
    mov     x20, x0                 // x20 = BAR0 base

    // --- Close fd (no longer needed after mmap) ---
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // --- Read and print PRI_TIMEOUT ---
    adr     x1, pri_timeout_str
    mov     x2, #pri_timeout_len
    bl      print_stdout

    // ldr with large offset: add offset to base in a temp reg
    movz    x0, #PTIMER_PRI_TIMEOUT
    add     x0, x20, x0
    ldr     w0, [x0]
    bl      print_hex32
    bl      print_newline

    // --- Print samples header ---
    adr     x1, samples_hdr_str
    mov     x2, #samples_hdr_len
    bl      print_stdout

    // --- Collect 10 PTIMER samples with wall-clock timestamps ---
    // x21 = pointer to sample storage (on stack)
    // Each sample: 8 bytes GPU time + 16 bytes wall-clock (tv_sec, tv_nsec)
    // = 24 bytes per sample, 10 samples = 240 bytes
    sub     sp, sp, #256            // align to 16
    mov     x21, sp

    mov     x22, #0                 // sample index

sample_loop:
    // Compute entry offset: x23 = index * 24
    mov     x0, x22
    mov     x1, #24
    mul     x23, x0, x1

    // Read wall-clock time into entry+8
    add     x1, x21, x23
    add     x1, x1, #8             // &wall_clock (offset 8 within entry)
    mov     x0, #CLOCK_MONOTONIC
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0

    // Read GPU PTIMER atomically: read hi, lo, re-read hi.
    // If hi changed (lo wrapped between reads), retry.
    movz    x0, #PTIMER_TIME_1
    add     x0, x20, x0                    // &TIME_1
    movz    x1, #PTIMER_TIME_0
    add     x1, x20, x1                    // &TIME_0

.Lptimer_read_retry:
    ldr     w24, [x0]                      // high 32  (first read)
    ldr     w25, [x1]                      // low 32
    ldr     w26, [x0]                      // high 32  (re-read)
    cmp     w24, w26
    b.ne    .Lptimer_read_retry            // hi changed -> lo wrapped, retry
    // w26 == w24, safe to combine
    orr     x24, x25, x24, lsl #32         // combine: x24 = (hi<<32)|lo

    // Store GPU time at entry+0
    // Recompute x23 = index * 24 (it's still valid, not clobbered)
    str     x24, [x21, x23]

    // Print "  [N] 0x"
    adr     x1, sample_prefix
    mov     x2, #3                          // "  ["
    bl      print_stdout
    mov     x0, x22
    bl      print_digit
    adr     x1, sample_mid
    mov     x2, #4                          // "] 0x"
    bl      print_stdout
    mov     x0, x24
    bl      print_hex64
    bl      print_newline

    // If not last sample, spin-wait ~100ms
    add     x22, x22, #1
    cmp     x22, #NUM_SAMPLES
    b.ge    samples_done

    // Spin delay: read clock_gettime until 100ms passes
    // Get start time
    sub     sp, sp, #32
    mov     x0, #CLOCK_MONOTONIC
    mov     x1, sp
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    ldr     x26, [sp]               // start_sec
    ldr     x27, [sp, #8]           // start_nsec

    movz    x28, #0x4C, lsl #16     // max iterations (~5M) as timeout guard
    movk    x28, #0x4B40            // x28 = 0x004C4B40 = 5,000,000
spin_wait:
    subs    x28, x28, #1
    b.eq    .Lspin_timeout          // safety: break out after max iters
    mov     x0, #CLOCK_MONOTONIC
    mov     x1, sp
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    ldr     x0, [sp]                // cur_sec
    ldr     x1, [sp, #8]            // cur_nsec

    // elapsed_ns = (cur_sec - start_sec) * 1e9 + (cur_nsec - start_nsec)
    sub     x2, x0, x26             // delta_sec
    sub     x3, x1, x27             // delta_nsec (may be negative)
    // Load 1_000_000_000 = 0x3B9ACA00
    movz    x4, #0xCA00
    movk    x4, #0x3B9A, lsl #16
    mul     x2, x2, x4
    add     x2, x2, x3              // total elapsed ns
    // Compare to 100ms = 100_000_000 = 0x05F5E100
    movz    x4, #0xE100
    movk    x4, #0x05F5, lsl #16
    cmp     x2, x4
    b.lt    spin_wait
.Lspin_timeout:

    add     sp, sp, #32
    b       sample_loop

samples_done:
    bl      print_newline

    // --- Compute delta (last - first GPU time) ---
    ldr     x24, [x21, #0]          // first
    // last at index 9: offset = 9*24 = 216
    mov     x0, #216
    ldr     x25, [x21, x0]          // last
    sub     x26, x25, x24           // delta GPU ns

    // SAFETY: check if GPU timer is stuck (delta == 0)
    cbnz    x26, .Ltimer_ok
    adr     x1, timer_stuck_str
    mov     x2, #timer_stuck_len
    bl      print_stdout
    add     sp, sp, #256
    mov     x0, x20
    mov     x1, #MAP_SIZE
    mov     x8, #SYS_MUNMAP
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0
.Ltimer_ok:

    adr     x1, delta_str
    mov     x2, #delta_len
    bl      print_stdout
    mov     x0, x26
    bl      print_hex64
    bl      print_newline

    // --- Compute wall-clock elapsed ---
    // first wall at [x21 + 8], [x21 + 16]
    // last  wall at [x21 + 224], [x21 + 232]
    ldr     x0, [x21, #8]           // first_sec
    ldr     x1, [x21, #16]          // first_nsec
    mov     x2, #224                // 9*24 + 8
    ldr     x3, [x21, x2]           // last_sec
    mov     x2, #232                // 9*24 + 16
    ldr     x4, [x21, x2]           // last_nsec

    sub     x5, x3, x0              // delta_sec
    sub     x6, x4, x1              // delta_nsec
    // Load 1_000_000_000
    movz    x7, #0xCA00
    movk    x7, #0x3B9A, lsl #16
    mul     x5, x5, x7
    add     x27, x5, x6             // total wall-clock ns
    // x27 = wall elapsed ns

    // Print wall-clock elapsed as ~NNN ms
    adr     x1, wallclock_str
    mov     x2, #wallclock_len
    bl      print_stdout

    // Convert ns to ms: divide by 1_000_000 = 0x000F4240
    movz    x1, #0x4240
    movk    x1, #0x000F, lsl #16
    udiv    x0, x27, x1
    bl      print_decimal
    adr     x1, ms_suffix
    mov     x2, #ms_suffix_len
    bl      print_stdout
    bl      print_newline

    // --- Compute tick rate ---
    // GPU timer is in nanoseconds, so rate = delta_gpu / delta_wall
    // If timer is 1 GHz (ticking every ns), delta_gpu ~= delta_wall
    // rate = delta_gpu * 1000 / delta_wall => thousandths (for X.XXX display)
    adr     x1, rate_str
    mov     x2, #rate_len
    bl      print_stdout

    mov     x0, #1000
    mul     x0, x26, x0             // delta_gpu * 1000
    udiv    x28, x0, x27            // thousandths of GHz

    // whole part = thousandths / 1000
    mov     x1, #1000
    udiv    x0, x28, x1
    bl      print_decimal
    // print dot
    adr     x1, dot_str
    mov     x2, #1
    bl      print_stdout
    // fractional part = thousandths % 1000, print as 3 digits
    mov     x1, #1000
    udiv    x2, x28, x1
    msub    x0, x2, x1, x28         // remainder (0-999)
    // Print 3-digit fractional: hundreds digit
    mov     x19, x0
    mov     x1, #100
    udiv    x0, x19, x1
    bl      print_digit
    // tens digit
    mov     x1, #100
    udiv    x2, x19, x1
    msub    x0, x2, x1, x19
    mov     x1, #10
    udiv    x0, x0, x1
    bl      print_digit
    // ones digit
    mov     x1, #10
    udiv    x2, x19, x1
    msub    x0, x2, x1, x19
    bl      print_digit

    adr     x1, ghz_suffix
    mov     x2, #ghz_suffix_len
    bl      print_stdout

    // --- Cleanup and exit ---
    add     sp, sp, #256

    // munmap
    mov     x0, x20
    mov     x1, #MAP_SIZE
    mov     x8, #SYS_MUNMAP
    svc     #0

    ldp     x29, x30, [sp], #16
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Error handlers
// ============================================================
open_fail:
    adr     x1, open_fail_str
    mov     x2, #open_fail_len
    bl      print_stdout
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

mmap_fail:
    adr     x1, mmap_fail_str
    mov     x2, #mmap_fail_len
    bl      print_stdout
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// print_stdout: x1=buf, x2=len
// ============================================================
print_stdout:
    stp     x29, x30, [sp, #-16]!
    mov     x0, #STDOUT
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// print_newline
// ============================================================
print_newline:
    stp     x29, x30, [sp, #-16]!
    adr     x1, newline_ch
    mov     x2, #1
    bl      print_stdout
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// print_hex64: x0 = value, prints 16 hex digits
// ============================================================
print_hex64:
    stp     x29, x30, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    mov     x19, x0
    sub     sp, sp, #16
    mov     x20, #16                // 16 digits

.Lhex64_loop:
    sub     x20, x20, #1
    and     x1, x19, #0xf
    adr     x2, hex_chars
    ldrb    w1, [x2, x1]
    strb    w1, [sp, x20]
    lsr     x19, x19, #4
    cbnz    x20, .Lhex64_loop

    mov     x0, #STDOUT
    mov     x1, sp
    mov     x2, #16
    mov     x8, #SYS_WRITE
    svc     #0

    add     sp, sp, #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// print_hex32: w0 = value, prints 8 hex digits
// ============================================================
print_hex32:
    stp     x29, x30, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    mov     w19, w0
    sub     sp, sp, #16
    mov     x20, #8

.Lhex32_loop:
    sub     x20, x20, #1
    and     x1, x19, #0xf
    adr     x2, hex_chars
    ldrb    w1, [x2, x1]
    strb    w1, [sp, x20]
    lsr     w19, w19, #4
    cbnz    x20, .Lhex32_loop

    mov     x0, #STDOUT
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0

    add     sp, sp, #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// print_digit: x0 = single digit 0-9
// ============================================================
print_digit:
    stp     x29, x30, [sp, #-16]!
    sub     sp, sp, #16
    add     w1, w0, #'0'
    strb    w1, [sp]
    mov     x0, #STDOUT
    mov     x1, sp
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// print_decimal: x0 = unsigned integer, prints decimal string
// ============================================================
print_decimal:
    stp     x29, x30, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    sub     sp, sp, #32
    mov     x19, x0
    mov     x20, #0                 // digit count

    // Handle zero
    cbnz    x19, .Ldec_loop
    mov     w1, #'0'
    strb    w1, [sp]
    mov     x20, #1
    b       .Ldec_print

.Ldec_loop:
    cbz     x19, .Ldec_reverse
    mov     x1, #10
    udiv    x2, x19, x1
    msub    x3, x2, x1, x19        // remainder
    add     w3, w3, #'0'
    strb    w3, [sp, x20]
    add     x20, x20, #1
    mov     x19, x2
    b       .Ldec_loop

.Ldec_reverse:
    // Reverse digits in place
    mov     x1, #0                  // left
    sub     x2, x20, #1             // right
.Ldec_rev_loop:
    cmp     x1, x2
    b.ge    .Ldec_print
    ldrb    w3, [sp, x1]
    ldrb    w4, [sp, x2]
    strb    w4, [sp, x1]
    strb    w3, [sp, x2]
    add     x1, x1, #1
    sub     x2, x2, #1
    b       .Ldec_rev_loop

.Ldec_print:
    mov     x0, #STDOUT
    mov     x1, sp
    mov     x2, x20
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Data
// ============================================================
.section .rodata

bar0_path:
    .asciz  "/sys/bus/pci/devices/0000:dd:00.0/resource0"

banner_str:
    .ascii  "\n=== Lithos PTIMER Probe ===\n\n"
    .equ    banner_len, . - banner_str

pri_timeout_str:
    .ascii  "PTIMER_PRI_TIMEOUT = 0x"
    .equ    pri_timeout_len, . - pri_timeout_str

samples_hdr_str:
    .ascii  "PTIMER samples (64-bit ns counter):\n"
    .equ    samples_hdr_len, . - samples_hdr_str

sample_prefix:
    .ascii  "  ["

sample_mid:
    .ascii  "] 0x"

delta_str:
    .ascii  "Delta (last - first) = 0x"
    .equ    delta_len, . - delta_str

wallclock_str:
    .ascii  "Wall-clock elapsed   = ~"
    .equ    wallclock_len, . - wallclock_str

ms_suffix:
    .ascii  " ms"
    .equ    ms_suffix_len, . - ms_suffix

rate_str:
    .ascii  "GPU timer rate       = ~"
    .equ    rate_len, . - rate_str

dot_str:
    .ascii  "."

ghz_suffix:
    .ascii  " GHz\n"
    .equ    ghz_suffix_len, . - ghz_suffix

timer_stuck_str:
    .ascii  "CRITICAL: GPU PTIMER is stuck (delta=0). Timer not ticking!\n"
    .equ    timer_stuck_len, . - timer_stuck_str

open_fail_str:
    .ascii  "ERROR: Failed to open BAR0 resource0\n"
    .equ    open_fail_len, . - open_fail_str

mmap_fail_str:
    .ascii  "ERROR: Failed to mmap BAR0\n"
    .equ    mmap_fail_len, . - mmap_fail_str

hex_chars:
    .ascii  "0123456789abcdef"

newline_ch:
    .ascii  "\n"
