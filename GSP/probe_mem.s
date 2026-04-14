// probe_mem.s — Lithos HBM Memory Probe for GH200
// Characterizes BAR4 memory system: size, accessibility, latency, bandwidth
//
// Build: as -o probe_mem.o probe_mem.s && ld -o probe_mem probe_mem.o
// Run:   sudo ./probe_mem

.global _start

// ---- syscall numbers (Linux aarch64) ----
.equ SYS_read,        63
.equ SYS_write,       64
.equ SYS_openat,      56
.equ SYS_close,       57
.equ SYS_mmap,        222
.equ SYS_munmap,      215
.equ SYS_exit,        93
.equ SYS_clock_gettime, 113

.equ AT_FDCWD,        -100
.equ O_RDONLY,         0
.equ O_RDWR,           2
.equ PROT_READ,        1
.equ PROT_WRITE,       2
.equ PROT_RW,          3
.equ MAP_SHARED,       1
.equ CLOCK_MONOTONIC,  1

.equ PAGE_SIZE,        4096
.equ MAP_2MB,          0x200000

// Stack frame layout (from sp):
//   sp+0    : saved x29,x30 (16 bytes)
//   sp+16   : ts_before (16 bytes)
//   sp+32   : ts_after  (16 bytes)
//   sp+48   : readbuf   (1024 bytes)
// Total: 1072 → round to 1088 (16-byte aligned)
.equ STK_TS_BEFORE,   16
.equ STK_TS_AFTER,    32
.equ STK_READBUF,     48
.equ STK_SIZE,        1088

// ---- Read-only string data (embedded in .text for adr reach) ----
.section .text

resource_path:
    .asciz "/sys/bus/pci/devices/0000:dd:00.0/resource"
    .align 2
bar4_path:
    .asciz "/sys/bus/pci/devices/0000:dd:00.0/resource4"
    .align 2

str_banner:     .asciz "\n=== Lithos HBM Memory Probe ===\n\n"
    .align 2
str_bar4_size:  .asciz "BAR4 size          = "
    .align 2
str_gb:         .asciz " GB\n"
    .align 2
str_bar4_base:  .asciz "BAR4 phys base     = 0x"
    .align 2
str_nl:         .asciz "\n"
    .align 2
str_access:     .asciz "\nAccessibility:\n"
    .align 2
str_off0:       .asciz "  Offset 0GB        = 0x"
    .align 2
str_off64:      .asciz "  Offset 64GB       = 0x"
    .align 2
str_off96:      .asciz "  Offset 96GB       = 0x"
    .align 2
str_ok:         .asciz " (OK)\n"
    .align 2
str_fail:       .asciz " (FAIL)\n"
    .align 2
str_latency:    .asciz "\nLatency (1000 reads, same addr):\n"
    .align 2
str_rdlat:      .asciz "  Avg read latency  = "
    .align 2
str_wrlat:      .asciz "  Avg write latency = "
    .align 2
str_ns:         .asciz " ns\n"
    .align 2
str_bw:         .asciz "\nBandwidth (2MB sequential read):\n"
    .align 2
str_tp:         .asciz "  Throughput         = "
    .align 2
str_mbs:        .asciz " MB/s\n"
    .align 2
str_allfs:      .asciz " (WARN: all-Fs, likely dead)\n"
    .align 2
str_err_res:    .asciz "ERROR: cannot open resource file\n"
    .align 2
str_err_bar4:   .asciz "ERROR: cannot open BAR4\n"
    .align 2

// ============================================================
// _start
// ============================================================
.align 2
_start:
    sub     sp, sp, STK_SIZE
    stp     x29, x30, [sp]
    mov     x29, sp

    // Print banner
    adr     x1, str_banner
    bl      print_str

    // ---- Step 1: Parse BAR4 size from resource file ----
    mov     x0, AT_FDCWD
    adr     x1, resource_path
    mov     x2, O_RDONLY
    mov     x3, 0
    mov     x8, SYS_openat
    svc     0
    cmp     x0, 0
    b.lt    err_resource
    mov     x19, x0

    // Read contents
    mov     x0, x19
    add     x1, sp, STK_READBUF
    mov     x2, 1024
    mov     x8, SYS_read
    svc     0
    mov     x20, x0             // bytes read

    mov     x0, x19
    mov     x8, SYS_close
    svc     0

    // Parse line 5 (index 4) — skip 4 newlines
    add     x0, sp, STK_READBUF
    add     x1, x0, x20
    mov     x2, 4
.skip_line:
    cbz     x2, .found_line
.skip_char:
    cmp     x0, x1
    b.ge    err_resource
    ldrb    w3, [x0], 1
    cmp     w3, '\n'
    b.ne    .skip_char
    sub     x2, x2, 1
    b       .skip_line
.found_line:
    bl      skip_spaces
    bl      parse_hex
    mov     x21, x0             // BAR4 phys start
    mov     x0, x1
    bl      skip_spaces
    bl      parse_hex
    mov     x22, x0             // BAR4 phys end

    sub     x23, x22, x21
    add     x23, x23, 1         // BAR4 size bytes

    // Print BAR4 size in GB
    adr     x1, str_bar4_size
    bl      print_str
    mov     x0, x23
    lsr     x0, x0, 30
    bl      print_dec
    adr     x1, str_gb
    bl      print_str

    // Print BAR4 phys base
    adr     x1, str_bar4_base
    bl      print_str
    mov     x0, x21
    bl      print_hex16
    adr     x1, str_nl
    bl      print_str

    // ---- Step 2: Accessibility probes ----
    adr     x1, str_access
    bl      print_str

    mov     x0, AT_FDCWD
    adr     x1, bar4_path
    mov     x2, O_RDWR
    mov     x3, 0
    mov     x8, SYS_openat
    svc     0
    cmp     x0, 0
    b.lt    err_bar4
    mov     x24, x0             // BAR4 fd

    // Probe offset 0GB
    adr     x1, str_off0
    bl      print_str
    mov     x0, x24
    mov     x1, 0
    bl      probe_offset
    cmn     x1, #2
    b.eq    .off0_allfs
    cbnz    x1, .off0_fail
    bl      print_hex16
    adr     x1, str_ok
    b       .off0_pr
.off0_allfs:
    bl      print_hex16
    adr     x1, str_allfs
    b       .off0_pr
.off0_fail:
    bl      print_hex16
    adr     x1, str_fail
.off0_pr:
    bl      print_str

    // Probe offset 64GB
    adr     x1, str_off64
    bl      print_str
    mov     x0, x24
    mov     x1, 64
    lsl     x1, x1, 30
    bl      probe_offset
    cmn     x1, #2
    b.eq    .off64_allfs
    cbnz    x1, .off64_fail
    bl      print_hex16
    adr     x1, str_ok
    b       .off64_pr
.off64_allfs:
    bl      print_hex16
    adr     x1, str_allfs
    b       .off64_pr
.off64_fail:
    bl      print_hex16
    adr     x1, str_fail
.off64_pr:
    bl      print_str

    // Probe offset 96GB
    adr     x1, str_off96
    bl      print_str
    mov     x0, x24
    mov     x1, 96
    lsl     x1, x1, 30
    bl      probe_offset
    cmn     x1, #2
    b.eq    .off96_allfs
    cbnz    x1, .off96_fail
    bl      print_hex16
    adr     x1, str_ok
    b       .off96_pr
.off96_allfs:
    bl      print_hex16
    adr     x1, str_allfs
    b       .off96_pr
.off96_fail:
    bl      print_hex16
    adr     x1, str_fail
.off96_pr:
    bl      print_str

    // ---- Step 3: Latency measurement ----
    adr     x1, str_latency
    bl      print_str

    // Map 4KB at offset 0
    mov     x0, 0
    mov     x1, PAGE_SIZE
    mov     x2, PROT_RW
    mov     x3, MAP_SHARED
    mov     x4, x24
    mov     x5, 0
    mov     x8, SYS_mmap
    svc     0
    cmn     x0, #4096
    b.hi    err_bar4
    mov     x25, x0             // mapped ptr

    // Save original value for restore
    ldr     x26, [x25]

    // --- Read latency ---
    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_BEFORE
    mov     x8, SYS_clock_gettime
    svc     0

    mov     x9, 1000
    mov     x10, 0
.read_loop:
    ldr     x11, [x25]
    add     x10, x10, x11
    sub     x9, x9, 1
    cbnz    x9, .read_loop

    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_AFTER
    mov     x8, SYS_clock_gettime
    svc     0

    bl      compute_elapsed_ns  // total ns in x0
    mov     x1, 1000
    udiv    x0, x0, x1

    mov     x27, x0
    adr     x1, str_rdlat
    bl      print_str
    mov     x0, x27
    bl      print_dec
    adr     x1, str_ns
    bl      print_str

    // --- Write latency ---
    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_BEFORE
    mov     x8, SYS_clock_gettime
    svc     0

    mov     x9, 1000
    movz    x11, 0xDEAD, lsl 16
    movk    x11, 0xBEEF
.write_loop:
    str     x11, [x25]
    sub     x9, x9, 1
    cbnz    x9, .write_loop

    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_AFTER
    mov     x8, SYS_clock_gettime
    svc     0

    // Restore original value
    str     x26, [x25]

    bl      compute_elapsed_ns
    mov     x1, 1000
    udiv    x0, x0, x1

    mov     x27, x0
    adr     x1, str_wrlat
    bl      print_str
    mov     x0, x27
    bl      print_dec
    adr     x1, str_ns
    bl      print_str

    // Unmap latency page
    mov     x0, x25
    mov     x1, PAGE_SIZE
    mov     x8, SYS_munmap
    svc     0

    // ---- Step 4: Bandwidth estimate ----
    adr     x1, str_bw
    bl      print_str

    // Map 2MB at offset 0
    mov     x0, 0
    mov     x1, MAP_2MB
    mov     x2, PROT_READ
    mov     x3, MAP_SHARED
    mov     x4, x24
    mov     x5, 0
    mov     x8, SYS_mmap
    svc     0
    cmn     x0, #4096
    b.hi    err_bar4
    mov     x25, x0

    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_BEFORE
    mov     x8, SYS_clock_gettime
    svc     0

    // Sequential read: stride 64 bytes, 2MB / 64 = 32768 iters
    mov     x9, 32768
    mov     x10, x25
    mov     x11, 0
    mov     x12, 0
.bw_loop:
    ldp     x13, x14, [x10], 64
    add     x11, x11, x13
    add     x12, x12, x14
    sub     x9, x9, 1
    cbnz    x9, .bw_loop

    dsb     sy
    isb
    mov     x0, CLOCK_MONOTONIC
    add     x1, sp, STK_TS_AFTER
    mov     x8, SYS_clock_gettime
    svc     0

    bl      compute_elapsed_ns  // elapsed_ns in x0

    // MB/s = 2097152 * 1e9 / elapsed_ns
    // To avoid overflow: elapsed_us = elapsed_ns / 1000
    //   MB/s = 2097152 * 1e6 / elapsed_us
    //   2097152 * 1e6 = 0x1E8_4800_0000 (fits 64-bit)
    mov     x15, x0
    mov     x1, 1000
    udiv    x2, x15, x1         // elapsed_us
    cbz     x2, .bw_zero

    movz    x3, 0x01E8, lsl 32
    movk    x3, 0x4800, lsl 16
    movk    x3, 0x0000
    udiv    x0, x3, x2
    b       .bw_print
.bw_zero:
    mov     x0, 0
.bw_print:
    mov     x27, x0
    adr     x1, str_tp
    bl      print_str
    mov     x0, x27
    bl      print_dec
    adr     x1, str_mbs
    bl      print_str

    // Unmap 2MB
    mov     x0, x25
    mov     x1, MAP_2MB
    mov     x8, SYS_munmap
    svc     0

    // Close BAR4
    mov     x0, x24
    mov     x8, SYS_close
    svc     0

    // Exit
    ldp     x29, x30, [sp]
    add     sp, sp, STK_SIZE
    mov     x0, 0
    mov     x8, SYS_exit
    svc     0

// ============================================================
// Error handlers
// ============================================================
err_resource:
    adr     x1, str_err_res
    bl      print_str
    mov     x0, 1
    mov     x8, SYS_exit
    svc     0

err_bar4:
    adr     x1, str_err_bar4
    bl      print_str
    mov     x0, 1
    mov     x8, SYS_exit
    svc     0

// ============================================================
// compute_elapsed_ns — reads ts_before/ts_after from stack.
// Returns total nanoseconds in x0.
// Note: caller must have sp pointing to _start's frame.
// ============================================================
compute_elapsed_ns:
    ldp     x2, x3, [sp, STK_TS_AFTER]     // sec_after, nsec_after
    ldp     x4, x5, [sp, STK_TS_BEFORE]    // sec_before, nsec_before
    sub     x0, x2, x4                      // delta_sec
    movz    x1, 0x3B9A, lsl 16
    movk    x1, 0xCA00                      // 1000000000
    mul     x0, x0, x1                      // delta_sec * 1e9
    sub     x1, x3, x5                      // delta_nsec (2's complement handles borrow)
    add     x0, x0, x1                      // total elapsed ns
    ret

// ============================================================
// probe_offset(fd=x0, offset=x1) -> value=x0, status=x1 (0=ok, -1=fail)
// ============================================================
probe_offset:
    stp     x29, x30, [sp, -48]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    str     x21, [sp, 32]

    mov     x15, x0             // fd
    mov     x16, x1             // offset

    mov     x0, 0
    mov     x1, PAGE_SIZE
    mov     x2, PROT_RW
    mov     x3, MAP_SHARED
    mov     x4, x15
    mov     x5, x16
    mov     x8, SYS_mmap
    svc     0
    cmn     x0, 1
    b.eq    .probe_fail

    mov     x17, x0
    ldr     x20, [x17]

    mov     x0, x17
    mov     x1, PAGE_SIZE
    mov     x8, SYS_munmap
    svc     0

    // Check for PCIe all-Fs (0xFFFFFFFF_FFFFFFFF) = dead/unmapped region
    mov     x0, x20
    mvn     x1, xzr             // x1 = 0xFFFFFFFFFFFFFFFF
    cmp     x20, x1
    b.ne    .probe_val_ok
    mov     x1, -2              // status -2 = all-Fs (suspect dead read)
    b       .probe_ret
.probe_val_ok:
    mov     x1, 0
    b       .probe_ret

.probe_fail:
    mov     x0, 0
    mov     x1, -1
.probe_ret:
    ldp     x19, x20, [sp, 16]
    ldr     x21, [sp, 32]
    ldp     x29, x30, [sp], 48
    ret

// ============================================================
// skip_spaces — x0 = ptr, advances past spaces/tabs
// ============================================================
skip_spaces:
    ldrb    w1, [x0]
    cmp     w1, ' '
    b.eq    .ss_next
    cmp     w1, '\t'
    b.eq    .ss_next
    ret
.ss_next:
    add     x0, x0, 1
    b       skip_spaces

// ============================================================
// parse_hex — x0 = ptr to "0xHHHH..."
// Returns: x0 = value, x1 = updated ptr
// ============================================================
parse_hex:
    ldrb    w2, [x0]
    cmp     w2, '0'
    b.ne    .ph_digits
    ldrb    w2, [x0, 1]
    cmp     w2, 'x'
    b.eq    .ph_skip2
    cmp     w2, 'X'
    b.eq    .ph_skip2
    b       .ph_digits
.ph_skip2:
    add     x0, x0, 2
.ph_digits:
    mov     x3, 0
.ph_loop:
    ldrb    w2, [x0]
    sub     w4, w2, '0'
    cmp     w4, 9
    b.hi    .ph_try_af
    lsl     x3, x3, 4
    add     x3, x3, x4
    add     x0, x0, 1
    b       .ph_loop
.ph_try_af:
    sub     w4, w2, 'a'
    cmp     w4, 5
    b.hi    .ph_try_AF
    add     w4, w4, 10
    lsl     x3, x3, 4
    add     x3, x3, x4
    add     x0, x0, 1
    b       .ph_loop
.ph_try_AF:
    sub     w4, w2, 'A'
    cmp     w4, 5
    b.hi    .ph_done
    add     w4, w4, 10
    lsl     x3, x3, 4
    add     x3, x3, x4
    add     x0, x0, 1
    b       .ph_loop
.ph_done:
    mov     x1, x0
    mov     x0, x3
    ret

// ============================================================
// print_str — x1 = null-terminated string
// ============================================================
print_str:
    mov     x3, x1
    mov     x2, 0
.ps_len:
    ldrb    w0, [x1, x2]
    cbz     w0, .ps_write
    add     x2, x2, 1
    b       .ps_len
.ps_write:
    mov     x0, 1
    mov     x1, x3
    mov     x8, SYS_write
    svc     0
    ret

// ============================================================
// print_hex16 — x0 = 64-bit value, prints 16-char hex
// ============================================================
print_hex16:
    sub     sp, sp, 16
    mov     x5, sp
    mov     x2, 60
.ph16_loop:
    lsr     x3, x0, x2
    and     x3, x3, 0xF
    cmp     x3, 10
    b.lt    .ph16_digit
    add     x3, x3, 'a' - 10
    b       .ph16_store
.ph16_digit:
    add     x3, x3, '0'
.ph16_store:
    strb    w3, [x5], 1
    sub     x2, x2, 4
    cmp     x2, 0
    b.ge    .ph16_loop

    mov     x0, 1
    mov     x1, sp
    mov     x2, 16
    mov     x8, SYS_write
    svc     0
    add     sp, sp, 16
    ret

// ============================================================
// print_dec — x0 = 64-bit unsigned, prints decimal
// ============================================================
print_dec:
    sub     sp, sp, 32
    add     x1, sp, 31
    mov     x5, x1
    mov     x3, 10

    cbnz    x0, .pd_loop
    sub     x1, x1, 1
    mov     w2, '0'
    strb    w2, [x1]
    b       .pd_write
.pd_loop:
    cbz     x0, .pd_write
    udiv    x4, x0, x3
    msub    x2, x4, x3, x0
    add     w2, w2, '0'
    sub     x1, x1, 1
    strb    w2, [x1]
    mov     x0, x4
    b       .pd_loop
.pd_write:
    mov     x0, 1
    sub     x2, x5, x1
    mov     x8, SYS_write
    svc     0
    add     sp, sp, 32
    ret
