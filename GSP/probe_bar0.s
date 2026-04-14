// probe_bar0.s -- Read-only BAR0 probe for GH200
//
// Maps BAR0 via sysfs, reads key GPU registers, prints results.
// Does NOT modify GPU state. Safe to run with nvidia.ko loaded.
// Requires root (to mmap sysfs resource files).
//
// Build:  as -o probe_bar0.o probe_bar0.s && ld -o probe_bar0 probe_bar0.o
// Run:    sudo ./probe_bar0
//
// PCI BDF is hardcoded below — patch if needed.

.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ AT_FDCWD,      -100
.equ O_RDWR,        2
.equ O_SYNC,        0x101000
.equ PROT_READ,     1
.equ PROT_WRITE,    2
.equ MAP_SHARED,    1
.equ BAR0_SIZE,     0x1000000       // 16 MB

// GPU register offsets (read-only safe)
.equ PMC_BOOT_0,            0x000000
.equ PMC_BOOT_42,           0x000088
.equ NV_PMC_ENABLE,         0x000200
.equ NV_PTIMER_TIME_0,      0x009400
.equ NV_PTIMER_TIME_1,      0x009410
.equ THERM_I2CS_SCRATCH,    0x0200BC    // FSP boot-complete sentinel
.equ FALCON_ENGINE,         0x1103C0    // GSP Falcon
.equ FALCON_HWCFG2,        0x1100F4    // PRIV_LOCKDOWN

.data
.align 3

bar0_path:
    .ascii "/sys/bus/pci/devices/0000:"
bar0_bdf:
    .ascii "dd:00"
    .ascii ".0/resource0\0"

msg_header:     .ascii "=== Lithos BAR0 Probe (read-only) ===\n"
msg_header_len = . - msg_header

msg_pmc0:       .ascii "PMC_BOOT_0        = 0x"
msg_pmc0_len  = . - msg_pmc0

msg_pmc42:      .ascii "PMC_BOOT_42       = 0x"
msg_pmc42_len = . - msg_pmc42

msg_enable:     .ascii "PMC_ENABLE        = 0x"
msg_enable_len = . - msg_enable

msg_timer:      .ascii "PTIMER_TIME_1:0   = 0x"
msg_timer_len = . - msg_timer

msg_fsp:        .ascii "THERM_I2CS_SCRATCH= 0x"
msg_fsp_len   = . - msg_fsp

msg_falcon:     .ascii "GSP_FALCON_ENGINE = 0x"
msg_falcon_len= . - msg_falcon

msg_hwcfg2:     .ascii "FALCON_HWCFG2     = 0x"
msg_hwcfg2_len= . - msg_hwcfg2

msg_arch:       .ascii "  -> Architecture   = "
msg_arch_len  = . - msg_arch

msg_hopper:     .ascii "HOPPER (0xA) ✓\n"
msg_hopper_len= . - msg_hopper

msg_not_hopper: .ascii "UNKNOWN (expected 0xA)\n"
msg_not_hop_len=. - msg_not_hopper

msg_fsp_ready:  .ascii "  -> FSP boot       = COMPLETE (0xFF)\n"
msg_fsp_rdy_len=. - msg_fsp_ready

msg_fsp_not:    .ascii "  -> FSP boot       = not ready\n"
msg_fsp_not_len=. - msg_fsp_not

msg_lockdown:   .ascii "  -> PRIV_LOCKDOWN  = "
msg_lock_len  = . - msg_lockdown

msg_locked:     .ascii "LOCKED (bit 13 set)\n"
msg_locked_len= . - msg_locked

msg_unlocked:   .ascii "RELEASED\n"
msg_unlock_len= . - msg_unlocked

msg_newline:    .ascii "\n"

msg_dev_dead:   .ascii "probe: PMC_BOOT_0=0xFFFFFFFF -- device not responding (link down?)\n"
msg_dev_dead_len = . - msg_dev_dead

msg_open_fail:  .ascii "probe: failed to open resource0\n"
msg_open_flen = . - msg_open_fail

msg_mmap_fail:  .ascii "probe: mmap failed\n"
msg_mmap_flen = . - msg_mmap_fail

.text
.align 4

.global _start
_start:
    // ---- Print header ----
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ---- Open BAR0 via sysfs ----
    mov     x0, #AT_FDCWD
    adrp    x1, bar0_path
    add     x1, x1, :lo12:bar0_path
    // O_RDWR | O_SYNC = 0x101002
    movz    x2, #0x1002
    movk    x2, #0x0010, lsl #16
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0             // x19 = fd

    // ---- mmap BAR0 ----
    mov     x0, #0              // addr = NULL
    mov     x1, #BAR0_SIZE
    mov     x2, #PROT_READ      // read-only probe
    mov     x3, #MAP_SHARED
    mov     x4, x19             // fd
    mov     x5, #0              // offset
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .mmap_fail
    mov     x20, x0             // x20 = bar0 base

    // Close fd (mapping persists)
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // ---- Read and print registers ----

    // PMC_BOOT_0
    ldr     w21, [x20, #PMC_BOOT_0]

    // CHECK: 0xFFFFFFFF means device not responding (link down, D3, BAR disabled)
    cmn     w21, #1             // cmn w21, #1 sets Z if w21 == 0xFFFFFFFF
    b.ne    .device_ok
    mov     x0, #2              // stderr
    adrp    x1, msg_dev_dead
    add     x1, x1, :lo12:msg_dev_dead
    mov     x2, #msg_dev_dead_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #3
    mov     x8, #SYS_EXIT
    svc     #0
.device_ok:

    adrp    x1, msg_pmc0
    add     x1, x1, :lo12:msg_pmc0
    mov     w2, #msg_pmc0_len
    mov     w0, w21
    bl      .print_reg

    // Check architecture: bits[7:4] on GH200
    ubfx    w0, w21, #4, #4
    cmp     w0, #0xA
    b.ne    .not_hopper
    mov     x0, #1
    adrp    x1, msg_arch
    add     x1, x1, :lo12:msg_arch
    mov     x2, #msg_arch_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    adrp    x1, msg_hopper
    add     x1, x1, :lo12:msg_hopper
    mov     x2, #msg_hopper_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .arch_done
.not_hopper:
    mov     x0, #1
    adrp    x1, msg_arch
    add     x1, x1, :lo12:msg_arch
    mov     x2, #msg_arch_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    adrp    x1, msg_not_hopper
    add     x1, x1, :lo12:msg_not_hopper
    mov     x2, #msg_not_hop_len
    mov     x8, #SYS_WRITE
    svc     #0
.arch_done:

    // PMC_BOOT_42
    ldr     w21, [x20, #PMC_BOOT_42]
    adrp    x1, msg_pmc42
    add     x1, x1, :lo12:msg_pmc42
    mov     w2, #msg_pmc42_len
    mov     w0, w21
    bl      .print_reg

    // PMC_ENABLE
    ldr     w21, [x20, #NV_PMC_ENABLE]
    adrp    x1, msg_enable
    add     x1, x1, :lo12:msg_enable
    mov     w2, #msg_enable_len
    mov     w0, w21
    bl      .print_reg

    // PTIMER (64-bit: read TIME_1 first, then TIME_0)
    mov     x1, #NV_PTIMER_TIME_1
    ldr     w22, [x20, x1]
    mov     x1, #NV_PTIMER_TIME_0
    ldr     w21, [x20, x1]
    // Combine and print TIME_1 (upper)
    adrp    x1, msg_timer
    add     x1, x1, :lo12:msg_timer
    mov     w2, #msg_timer_len
    mov     w0, w22
    bl      .print_reg_no_nl
    mov     w0, w21
    bl      .print_hex32
    // newline
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    // THERM_I2CS_SCRATCH (FSP boot sentinel)
    movz    x1, #0x00BC
    movk    x1, #0x0002, lsl #16    // 0x000200BC
    ldr     w21, [x20, x1]
    adrp    x1, msg_fsp
    add     x1, x1, :lo12:msg_fsp
    mov     w2, #msg_fsp_len
    mov     w0, w21
    bl      .print_reg

    // Check FSP ready
    and     w0, w21, #0xFF
    cmp     w0, #0xFF
    b.ne    .fsp_not_ready
    mov     x0, #1
    adrp    x1, msg_fsp_ready
    add     x1, x1, :lo12:msg_fsp_ready
    mov     x2, #msg_fsp_rdy_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .fsp_done
.fsp_not_ready:
    mov     x0, #1
    adrp    x1, msg_fsp_not
    add     x1, x1, :lo12:msg_fsp_not
    mov     x2, #msg_fsp_not_len
    mov     x8, #SYS_WRITE
    svc     #0
.fsp_done:

    // GSP FALCON_ENGINE
    movz    x1, #0x03C0
    movk    x1, #0x0011, lsl #16    // 0x001103C0
    ldr     w21, [x20, x1]
    adrp    x1, msg_falcon
    add     x1, x1, :lo12:msg_falcon
    mov     w2, #msg_falcon_len
    mov     w0, w21
    bl      .print_reg

    // FALCON_HWCFG2 (PRIV_LOCKDOWN bit 13)
    movz    x1, #0x00F4
    movk    x1, #0x0011, lsl #16    // 0x001100F4
    ldr     w21, [x20, x1]
    adrp    x1, msg_hwcfg2
    add     x1, x1, :lo12:msg_hwcfg2
    mov     w2, #msg_hwcfg2_len
    mov     w0, w21
    bl      .print_reg

    // Check lockdown
    mov     x0, #1
    adrp    x1, msg_lockdown
    add     x1, x1, :lo12:msg_lockdown
    mov     x2, #msg_lock_len
    mov     x8, #SYS_WRITE
    svc     #0
    tbnz    w21, #13, .locked
    mov     x0, #1
    adrp    x1, msg_unlocked
    add     x1, x1, :lo12:msg_unlocked
    mov     x2, #msg_unlock_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .lock_done
.locked:
    mov     x0, #1
    adrp    x1, msg_locked
    add     x1, x1, :lo12:msg_locked
    mov     x2, #msg_locked_len
    mov     x8, #SYS_WRITE
    svc     #0
.lock_done:

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
// .print_reg -- print "PREFIX0xXXXXXXXX\n"
//   w0 = value, x1 = prefix addr, w2 = prefix len
// ============================================================
.print_reg:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x22, [sp, #16]
    // Save value
    mov     w22, w0
    // Print prefix
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    // Print hex value
    mov     w0, w22
    bl      .print_hex32
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
// .print_reg_no_nl -- same but no trailing newline
// ============================================================
.print_reg_no_nl:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x22, [sp, #16]
    mov     w22, w0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w22
    bl      .print_hex32
    ldr     x22, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ============================================================
// .print_hex32 -- print 8 hex digits to stdout
//   w0 = value to print
// ============================================================
.print_hex32:
    sub     sp, sp, #16
    mov     w3, #28             // shift start (7*4 = 28 for MSB nibble)
    add     x4, sp, #0          // buffer pointer
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
    // Write 8 bytes from stack
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret
