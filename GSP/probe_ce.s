// probe_ce.s -- Copy Engine / NVLink / C2C discovery probe for GH200
//
// Maps BAR0 read-only via sysfs, reads top-level engine status,
// then scans the 0x100000-0x140000 range to discover live CE, NVLink,
// C2C, and memory controller registers.
//
// Read-only, safe with nvidia.ko loaded.
// No libc, raw syscalls.
//
// BDF:  0000:dd:00.0
// BAR0: /sys/bus/pci/devices/0000:dd:00.0/resource0
//
// Build:  as -o probe_ce.o probe_ce.s && ld -o probe_ce probe_ce.o
// Run:    sudo ./probe_ce

.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ AT_FDCWD,      -100
.equ PROT_READ,     1
.equ MAP_SHARED,    1
.equ BAR0_SIZE,     0x1000000       // 16 MB

// Known register offsets
.equ NV_PMC_ENABLE,         0x000200
.equ NV_PMC_DEVICE_ENABLE,  0x000600

// Scan range
.equ SCAN_START,    0x100000
.equ SCAN_END,      0x140000
.equ SCAN_STEP,     0x4             // read every dword

// Dead-register sentinels
.equ BADF_UPPER,    0xBADF          // upper 16 bits of 0xBADFxxxx

.data
.align 3

bar0_path:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource0\0"

// ---- Messages ----
msg_header:     .ascii "=== Lithos CE/NVLink/C2C Probe (read-only) ===\n\n"
msg_header_len = . - msg_header

msg_pmc_en:     .ascii "NV_PMC_ENABLE      (0x000200) = 0x"
msg_pmc_en_len = . - msg_pmc_en

msg_pmc_dev:    .ascii "NV_PMC_DEVICE_EN   (0x000600) = 0x"
msg_pmc_dev_len = . - msg_pmc_dev

msg_scan_hdr:   .ascii "\n--- Scanning 0x100000 - 0x140000 (live registers) ---\n"
msg_scan_hdr_len = . - msg_scan_hdr

// Block header labels (printed when entering a new range)
lbl_pfb:        .ascii "\n[0x100000] NV_PFB -- Framebuffer / Memory Controller\n"
lbl_pfb_len   = . - lbl_pfb

lbl_ce0:        .ascii "\n[0x104000] NV_PCE0 -- Copy Engine 0\n"
lbl_ce0_len   = . - lbl_ce0

lbl_ce1:        .ascii "\n[0x105000] NV_PCE1 -- Copy Engine 1\n"
lbl_ce1_len   = . - lbl_ce1

lbl_ce2:        .ascii "\n[0x106000] NV_PCE2 -- Copy Engine 2\n"
lbl_ce2_len   = . - lbl_ce2

lbl_ce3:        .ascii "\n[0x107000] NV_PCE3 -- Copy Engine 3\n"
lbl_ce3_len   = . - lbl_ce3

lbl_ce4:        .ascii "\n[0x108000] NV_PCE4 -- Copy Engine 4\n"
lbl_ce4_len   = . - lbl_ce4

lbl_ce5:        .ascii "\n[0x109000] NV_PCE5 -- Copy Engine 5\n"
lbl_ce5_len   = . - lbl_ce5

lbl_10a:        .ascii "\n[0x10A000] (unknown block)\n"
lbl_10a_len   = . - lbl_10a

lbl_10b:        .ascii "\n[0x10B000] (unknown block)\n"
lbl_10b_len   = . - lbl_10b

lbl_10c:        .ascii "\n[0x10C000] (unknown block)\n"
lbl_10c_len   = . - lbl_10c

lbl_unk:        .ascii "\n[0x???000] (unknown block)\n"
lbl_unk_len   = . - lbl_unk

lbl_nvl:        .ascii "\n[0x130000] NV_PNVL -- NVLink / C2C\n"
lbl_nvl_len   = . - lbl_nvl

lbl_nvl2:       .ascii "\n[0x138000] NV_PNVL (upper) -- NVLink / C2C\n"
lbl_nvl2_len  = . - lbl_nvl2

// Prefix for each register line: "  0x"
msg_prefix:     .ascii "  0x"
msg_prefix_len = . - msg_prefix

// " = 0x"
msg_eq:         .ascii " = 0x"
msg_eq_len    = . - msg_eq

msg_newline:    .ascii "\n"

// Summary
msg_summary:    .ascii "\n--- Scan complete ---\n"
msg_summary_len = . - msg_summary

msg_live:       .ascii "Live registers found: "
msg_live_len  = . - msg_live

msg_dead:       .ascii "Dead (0/badf) skipped: "
msg_dead_len  = . - msg_dead

msg_open_fail:  .ascii "probe_ce: failed to open resource0\n"
msg_open_flen = . - msg_open_fail

msg_mmap_fail:  .ascii "probe_ce: mmap failed\n"
msg_mmap_flen = . - msg_mmap_fail

.text
.align 4

.global _start
_start:
    // Save callee-saved regs we'll use
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    // ---- Print header ----
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Open BAR0 ----
    mov     x0, #AT_FDCWD
    adrp    x1, bar0_path
    add     x1, x1, :lo12:bar0_path
    // O_RDONLY | O_SYNC = 0x101000
    movz    x2, #0x1000
    movk    x2, #0x0010, lsl #16
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0             // x19 = fd

    // ---- mmap BAR0 read-only ----
    mov     x0, #0              // addr = NULL
    mov     x1, #BAR0_SIZE
    mov     x2, #PROT_READ
    mov     x3, #MAP_SHARED
    mov     x4, x19             // fd
    mov     x5, #0              // offset
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .mmap_fail
    mov     x20, x0             // x20 = bar0 base

    // Close fd
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // ======== Top-level engine status ========

    // NV_PMC_ENABLE
    ldr     w21, [x20, #NV_PMC_ENABLE]
    adrp    x1, msg_pmc_en
    add     x1, x1, :lo12:msg_pmc_en
    mov     w2, #msg_pmc_en_len
    mov     w0, w21
    bl      print_reg

    // NV_PMC_DEVICE_ENABLE
    movz    x1, #NV_PMC_DEVICE_ENABLE
    ldr     w21, [x20, x1]
    adrp    x1, msg_pmc_dev
    add     x1, x1, :lo12:msg_pmc_dev
    mov     w2, #msg_pmc_dev_len
    mov     w0, w21
    bl      print_reg

    // ======== Scan header ========
    mov     x0, #1
    adrp    x1, msg_scan_hdr
    add     x1, x1, :lo12:msg_scan_hdr
    mov     x2, #msg_scan_hdr_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ======== Discovery scan ========
    // x23 = current offset
    // x24 = live count
    // x25 = dead count
    // x26 = current block base (for block header printing)
    // x27 = scan end
    movz    x23, #0x0000
    movk    x23, #0x0010, lsl #16   // 0x100000
    mov     x24, #0
    mov     x25, #0
    mov     w26, #-1               // no block ID printed yet
    movz    x27, #0x0000
    movk    x27, #0x0014, lsl #16   // 0x140000

.scan_loop:
    cmp     x23, x27
    b.ge    .scan_done

    // Read register
    ldr     w21, [x20, x23]

    // Check if dead: 0x00000000
    cbz     w21, .skip_dead

    // Check if dead: 0xBADFxxxx
    lsr     w0, w21, #16
    movz    w1, #0xBADF
    cmp     w0, w1
    b.eq    .skip_dead

    // ---- Live register found ----
    add     x24, x24, #1

    // Check if we need a new block header
    // Classify offset into a block ID (0-12) based on address range
    // x26 holds the last block ID printed (-1 = none)
    bl      classify_block      // w0 = offset in x23, returns block ID in w0
    mov     w28, w0             // w28 = block ID
    cmp     w28, w26
    b.eq    .no_new_block

    mov     w26, w28            // update current block ID

    // Dispatch on block ID in w28
    // Block IDs: 0=PFB, 1=CE0, 2=CE1, 3=CE2, 4=CE3, 5=CE4, 6=CE5,
    //            7=10A, 8=10B, 9=10C, 10=NVL, 11=NVL2, 99=unknown
    cmp     w28, #99
    b.eq    .no_new_block       // skip unknown blocks

    cmp     w28, #0
    b.eq    .blk_pfb
    cmp     w28, #1
    b.eq    .blk_ce0
    cmp     w28, #2
    b.eq    .blk_ce1
    cmp     w28, #3
    b.eq    .blk_ce2
    cmp     w28, #4
    b.eq    .blk_ce3
    cmp     w28, #5
    b.eq    .blk_ce4
    cmp     w28, #6
    b.eq    .blk_ce5
    cmp     w28, #7
    b.eq    .blk_10a
    cmp     w28, #8
    b.eq    .blk_10b
    cmp     w28, #9
    b.eq    .blk_10c
    cmp     w28, #10
    b.eq    .blk_nvl
    cmp     w28, #11
    b.eq    .blk_nvl2
    b       .no_new_block

.blk_pfb:
    adrp    x1, lbl_pfb
    add     x1, x1, :lo12:lbl_pfb
    mov     x2, #lbl_pfb_len
    b       .print_blk
.blk_ce0:
    adrp    x1, lbl_ce0
    add     x1, x1, :lo12:lbl_ce0
    mov     x2, #lbl_ce0_len
    b       .print_blk
.blk_ce1:
    adrp    x1, lbl_ce1
    add     x1, x1, :lo12:lbl_ce1
    mov     x2, #lbl_ce1_len
    b       .print_blk
.blk_ce2:
    adrp    x1, lbl_ce2
    add     x1, x1, :lo12:lbl_ce2
    mov     x2, #lbl_ce2_len
    b       .print_blk
.blk_ce3:
    adrp    x1, lbl_ce3
    add     x1, x1, :lo12:lbl_ce3
    mov     x2, #lbl_ce3_len
    b       .print_blk
.blk_ce4:
    adrp    x1, lbl_ce4
    add     x1, x1, :lo12:lbl_ce4
    mov     x2, #lbl_ce4_len
    b       .print_blk
.blk_ce5:
    adrp    x1, lbl_ce5
    add     x1, x1, :lo12:lbl_ce5
    mov     x2, #lbl_ce5_len
    b       .print_blk
.blk_10a:
    adrp    x1, lbl_10a
    add     x1, x1, :lo12:lbl_10a
    mov     x2, #lbl_10a_len
    b       .print_blk
.blk_10b:
    adrp    x1, lbl_10b
    add     x1, x1, :lo12:lbl_10b
    mov     x2, #lbl_10b_len
    b       .print_blk
.blk_10c:
    adrp    x1, lbl_10c
    add     x1, x1, :lo12:lbl_10c
    mov     x2, #lbl_10c_len
    b       .print_blk
.blk_nvl:
    adrp    x1, lbl_nvl
    add     x1, x1, :lo12:lbl_nvl
    mov     x2, #lbl_nvl_len
    b       .print_blk
.blk_nvl2:
    adrp    x1, lbl_nvl2
    add     x1, x1, :lo12:lbl_nvl2
    mov     x2, #lbl_nvl2_len
    b       .print_blk

.print_blk:
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0

.no_new_block:
    // Print "  0x" <offset> " = 0x" <value> "\n"
    mov     x0, #1
    adrp    x1, msg_prefix
    add     x1, x1, :lo12:msg_prefix
    mov     x2, #msg_prefix_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Print offset as hex (reuse w0 = offset, but x23 is 32-bit enough)
    mov     w0, w23
    bl      print_hex32

    // Print " = 0x"
    mov     x0, #1
    adrp    x1, msg_eq
    add     x1, x1, :lo12:msg_eq
    mov     x2, #msg_eq_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Print value
    mov     w0, w21
    bl      print_hex32

    // Print newline
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    b       .scan_next

.skip_dead:
    add     x25, x25, #1

.scan_next:
    add     x23, x23, #SCAN_STEP
    b       .scan_loop

.scan_done:
    // ---- Summary ----
    mov     x0, #1
    adrp    x1, msg_summary
    add     x1, x1, :lo12:msg_summary
    mov     x2, #msg_summary_len
    mov     x8, #SYS_WRITE
    svc     #0

    // "Live registers found: "
    mov     x0, #1
    adrp    x1, msg_live
    add     x1, x1, :lo12:msg_live
    mov     x2, #msg_live_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w24
    bl      print_decimal
    // newline
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    // "Dead skipped: "
    mov     x0, #1
    adrp    x1, msg_dead
    add     x1, x1, :lo12:msg_dead
    mov     x2, #msg_dead_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w25
    bl      print_decimal
    // newline
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Exit 0 ----
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ---- Error paths ----
.open_fail:
    mov     x0, #2
    adrp    x1, msg_open_fail
    add     x1, x1, :lo12:msg_open_fail
    mov     x2, #msg_open_flen
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
    mov     x2, #msg_mmap_flen
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// classify_block -- map offset to block ID
//   Input:  x23 = BAR0 offset
//   Output: w0  = block ID
//   Block IDs: 0=PFB(100-103), 1=CE0(104), 2=CE1(105), 3=CE2(106),
//              4=CE3(107), 5=CE4(108), 6=CE5(109), 7=10A, 8=10B,
//              9=10C, 10=NVL(130-137), 11=NVL2(138-13F), 99=unknown
// ============================================================
classify_block:
    // Extract bits [19:12] of offset => page index within scan range
    lsr     w0, w23, #12        // w0 = offset >> 12

    // 0x100-0x103 => PFB (block 0)
    cmp     w0, #0x100
    b.lt    .cls_unk
    cmp     w0, #0x104
    b.lt    .cls_ret0
    // 0x104 => CE0 (block 1)
    cmp     w0, #0x104
    b.eq    .cls_ret1
    // 0x105 => CE1 (block 2)
    cmp     w0, #0x105
    b.eq    .cls_ret2
    // 0x106 => CE2 (block 3)
    cmp     w0, #0x106
    b.eq    .cls_ret3
    // 0x107 => CE3 (block 4)
    cmp     w0, #0x107
    b.eq    .cls_ret4
    // 0x108 => CE4 (block 5)
    cmp     w0, #0x108
    b.eq    .cls_ret5
    // 0x109 => CE5 (block 6)
    cmp     w0, #0x109
    b.eq    .cls_ret6
    // 0x10A => block 7
    cmp     w0, #0x10A
    b.eq    .cls_ret7
    // 0x10B => block 8
    cmp     w0, #0x10B
    b.eq    .cls_ret8
    // 0x10C => block 9
    cmp     w0, #0x10C
    b.eq    .cls_ret9
    // 0x10D-0x12F => unknown
    cmp     w0, #0x130
    b.lt    .cls_unk
    // 0x130-0x137 => NVL (block 10)
    cmp     w0, #0x138
    b.lt    .cls_ret10
    // 0x138-0x13F => NVL2 (block 11)
    cmp     w0, #0x140
    b.lt    .cls_ret11
    b       .cls_unk

.cls_ret0:  mov w0, #0;  ret
.cls_ret1:  mov w0, #1;  ret
.cls_ret2:  mov w0, #2;  ret
.cls_ret3:  mov w0, #3;  ret
.cls_ret4:  mov w0, #4;  ret
.cls_ret5:  mov w0, #5;  ret
.cls_ret6:  mov w0, #6;  ret
.cls_ret7:  mov w0, #7;  ret
.cls_ret8:  mov w0, #8;  ret
.cls_ret9:  mov w0, #9;  ret
.cls_ret10: mov w0, #10; ret
.cls_ret11: mov w0, #11; ret
.cls_unk:   mov w0, #99; ret

// ============================================================
// print_reg -- print "<prefix>XXXXXXXX\n"
//   w0 = value, x1 = prefix addr, w2 = prefix len
// ============================================================
print_reg:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x22, [sp, #16]
    mov     w22, w0
    // Print prefix
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    // Print hex value
    mov     w0, w22
    bl      print_hex32
    // Print newline
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x22, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ============================================================
// print_hex32 -- print 8 hex digits to stdout
//   w0 = value
// ============================================================
print_hex32:
    sub     sp, sp, #16
    mov     w3, #28
    add     x4, sp, #0
.hex_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex_digit
    add     w5, w5, #('a' - 10)
    b       .hex_store
.hex_digit:
    add     w5, w5, #'0'
.hex_store:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .hex_loop
    // Write 8 bytes
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret

// ============================================================
// print_decimal -- print unsigned decimal w0 to stdout
// ============================================================
print_decimal:
    sub     sp, sp, #32
    add     x4, sp, #20        // end of buffer
    mov     w1, w0
    mov     w2, #0              // digit count

    // Handle zero
    cbnz    w1, .dec_loop
    mov     w5, #'0'
    sub     x4, x4, #1
    strb    w5, [x4]
    mov     w2, #1
    b       .dec_print

.dec_loop:
    cbz     w1, .dec_print
    mov     w6, #10
    udiv    w7, w1, w6
    msub    w5, w7, w6, w1     // remainder
    add     w5, w5, #'0'
    sub     x4, x4, #1
    strb    w5, [x4]
    add     w2, w2, #1
    mov     w1, w7
    b       .dec_loop

.dec_print:
    mov     x0, #1
    mov     x1, x4
    mov     x2, x2             // length (already in w2, zero-extend)
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ret
