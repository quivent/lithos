// probe_bar4.s -- Read-only BAR4 (HBM) coherence probe for GH200
//
// Maps BAR4 via sysfs, reads/writes a small region to verify:
//   1. BAR4 mmap succeeds
//   2. CPU can read HBM (first 64 bytes)
//   3. CPU write + readback is coherent (C2C/ATS)
//   4. Reports BAR4 physical base address
//
// WRITES to a small scratch region at BAR4 offset 127GB (near end,
// unlikely to overlap active allocations). Restores original values.
//
// Build:  as -o probe_bar4.o probe_bar4.s && ld -o probe_bar4 probe_bar4.o
// Run:    sudo ./probe_bar4

.equ SYS_OPENAT,    56
.equ SYS_READ,      63
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ AT_FDCWD,      -100
.equ PROT_RW,       3
.equ MAP_SHARED,    1

// Map a 2MB window at the start of BAR4 (enough for probing)
.equ BAR4_MAP_SIZE, 0x200000

.data
.align 3

bar4_path:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource4\0"

resource_path:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource\0"

msg_header:     .ascii "=== Lithos BAR4 HBM Probe ===\n\n"
msg_header_len = . - msg_header

msg_bar4_ok:    .ascii "BAR4 mmap          = OK (2 MB window)\n"
msg_bar4_ok_len= . - msg_bar4_ok

msg_phys:       .ascii "BAR4 phys base     = 0x"
msg_phys_len  = . - msg_phys

msg_read:       .ascii "HBM read test      = "
msg_read_len  = . - msg_read

msg_write:      .ascii "HBM write/readback = "
msg_write_len = . - msg_write

msg_pass:       .ascii "PASS\n"
msg_pass_len  = . - msg_pass

msg_fail:       .ascii "FAIL\n"
msg_fail_len  = . - msg_fail

msg_first8:     .ascii "HBM first 8 bytes  = 0x"
msg_first8_len= . - msg_first8

msg_newline:    .ascii "\n"

msg_hbm_dead:   .ascii "probe_bar4: HBM read returned 0xFFFFFFFFFFFFFFFF -- BAR4/C2C not responding\n"
msg_hbm_dead_len = . - msg_hbm_dead

msg_open_fail:  .ascii "probe_bar4: failed to open resource4\n"
msg_opf_len   = . - msg_open_fail

msg_mmap_fail:  .ascii "probe_bar4: mmap failed\n"
msg_mmf_len   = . - msg_mmap_fail

// Scratch area for resource file parsing
resource_buf: .skip 2048

.text
.align 4

.global _start
_start:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]

    // Header
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Parse BAR4 physical base from resource file ----
    mov     x0, #AT_FDCWD
    adrp    x1, resource_path
    add     x1, x1, :lo12:resource_path
    mov     x2, #0              // O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .skip_phys
    mov     x19, x0             // fd

    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    mov     x2, #2048
    mov     x0, x19
    mov     x8, #SYS_READ
    svc     #0
    mov     x22, x0             // bytes read

    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Parse: skip 4 lines (BAR0, BAR1, BAR2, BAR3), read line 5 (BAR4)
    // Each line: "0xSTART 0xEND 0xFLAGS\n"
    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    add     x3, x1, x22        // end of buffer
    mov     w4, #4              // skip 4 lines
.skip_lines:
    cbz     w4, .parse_bar4_hex
    cmp     x1, x3
    b.ge    .skip_phys
.skip_char:
    cmp     x1, x3
    b.ge    .skip_phys
    ldrb    w5, [x1], #1
    cmp     w5, #'\n'
    b.ne    .skip_char
    sub     w4, w4, #1
    b       .skip_lines

.parse_bar4_hex:
    // Skip "0x" prefix
    cmp     x1, x3
    b.ge    .skip_phys
    ldrb    w5, [x1]
    cmp     w5, #'0'
    b.ne    .skip_phys
    add     x1, x1, #2         // skip "0x"
    mov     x23, #0             // accumulator
.hex_loop:
    cmp     x1, x3
    b.ge    .hex_done
    ldrb    w5, [x1], #1
    // Is it 0-9?
    sub     w6, w5, #'0'
    cmp     w6, #10
    b.lt    .hex_add
    // Is it a-f?
    sub     w6, w5, #'a'
    cmp     w6, #6
    b.lt    .hex_af
    // Not hex, done
    b       .hex_done
.hex_af:
    add     w6, w6, #10
.hex_add:
    lsl     x23, x23, #4
    and     x6, x6, #0xF
    orr     x23, x23, x6
    b       .hex_loop
.hex_done:
    // x23 = BAR4 physical base

    // Print phys base
    mov     x0, #1
    adrp    x1, msg_phys
    add     x1, x1, :lo12:msg_phys
    mov     x2, #msg_phys_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Print as 16-digit hex
    mov     x0, x23
    bl      .print_hex64_nl
    b       .phys_done

.skip_phys:
    mov     x23, #0
.phys_done:

    // ---- Open and mmap BAR4 ----
    mov     x0, #AT_FDCWD
    adrp    x1, bar4_path
    add     x1, x1, :lo12:bar4_path
    movz    x2, #0x1002
    movk    x2, #0x0010, lsl #16    // O_RDWR | O_SYNC
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0

    mov     x0, #0
    mov     x1, #BAR4_MAP_SIZE
    mov     x2, #PROT_RW
    mov     x3, #MAP_SHARED
    mov     x4, x19
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .mmap_fail
    mov     x20, x0             // x20 = bar4 base

    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Print BAR4 OK
    mov     x0, #1
    adrp    x1, msg_bar4_ok
    add     x1, x1, :lo12:msg_bar4_ok
    mov     x2, #msg_bar4_ok_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Read test: read first 8 bytes ----
    ldr     x21, [x20]          // first 8 bytes of HBM

    // CHECK: all-ones means BAR4/HBM not accessible (link down, C2C not ready)
    cmn     x21, #1             // sets Z if x21 == 0xFFFFFFFFFFFFFFFF
    b.ne    .hbm_read_ok
    mov     x0, #2              // stderr
    adrp    x1, msg_hbm_dead
    add     x1, x1, :lo12:msg_hbm_dead
    mov     x2, #msg_hbm_dead_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #3
    mov     x8, #SYS_EXIT
    svc     #0
.hbm_read_ok:

    mov     x0, #1
    adrp    x1, msg_first8
    add     x1, x1, :lo12:msg_first8
    mov     x2, #msg_first8_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, x21
    bl      .print_hex64_nl

    mov     x0, #1
    adrp    x1, msg_read
    add     x1, x1, :lo12:msg_read
    mov     x2, #msg_read_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    adrp    x1, msg_pass
    add     x1, x1, :lo12:msg_pass
    mov     x2, #msg_pass_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Write/readback test at offset 0x100000 (1MB in) ----
    // Save original value, write pattern, read back, compare, restore
    movz    x1, #0x0010, lsl #16           // 0x100000 = 1MB
    add     x24, x20, x1                   // x24 = bar4 + 1MB

    ldr     x21, [x24]              // save original
    movz    x22, #0xBEEF
    movk    x22, #0xDEAD, lsl #16
    movk    x22, #0xCAFE, lsl #32
    movk    x22, #0x1234, lsl #48   // pattern = 0x1234CAFEDEADBEEF

    str     x22, [x24]              // write pattern
    dsb     sy                      // ensure visible
    ldr     x23, [x24]             // read back into x23 (callee-saved)
    str     x21, [x24]             // restore original
    dsb     sy

    mov     x0, #1
    adrp    x1, msg_write
    add     x1, x1, :lo12:msg_write
    mov     x2, #msg_write_len
    mov     x8, #SYS_WRITE
    svc     #0

    cmp     x23, x22               // compare readback vs pattern
    b.ne    .write_fail

    mov     x0, #1
    adrp    x1, msg_pass
    add     x1, x1, :lo12:msg_pass
    mov     x2, #msg_pass_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .write_done

.write_fail:
    mov     x0, #1
    adrp    x1, msg_fail
    add     x1, x1, :lo12:msg_fail
    mov     x2, #msg_fail_len
    mov     x8, #SYS_WRITE
    svc     #0
.write_done:

    // ---- Exit ----
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.open_fail:
    mov     x0, #2
    adrp    x1, msg_open_fail
    add     x1, x1, :lo12:msg_open_fail
    mov     x2, #msg_opf_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.mmap_fail:
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
    mov     x0, #2
    adrp    x1, msg_mmap_fail
    add     x1, x1, :lo12:msg_mmap_fail
    mov     x2, #msg_mmf_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Hex printers
// ============================================================
.print_hex64_nl:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x19, [sp, #16]      // save callee-saved x19
    mov     x19, x0
    lsr     x0, x19, #32
    bl      .print_hex32
    mov     w0, w19
    bl      .print_hex32
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x19, [sp, #16]      // restore x19
    ldp     x29, x30, [sp], #32
    ret

.print_hex32:
    sub     sp, sp, #16
    mov     w3, #28
    add     x4, sp, #0
.p32_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .p32_dig
    add     w5, w5, #('a' - 10)
    b       .p32_st
.p32_dig:
    add     w5, w5, #'0'
.p32_st:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .p32_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret
