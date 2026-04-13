// probe_gsp_falcon.s -- Read-only GSP Falcon state probe for GH200
//
// Reads GSP Falcon registers to determine if GSP firmware is running,
// what state it's in, and key configuration.
//
// Does NOT modify GPU state. Safe with nvidia.ko loaded.
//
// Build:  as -o probe_gsp_falcon.o probe_gsp_falcon.s && ld -o probe_gsp_falcon probe_gsp_falcon.o
// Run:    sudo ./probe_gsp_falcon

.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ AT_FDCWD,      -100
.equ PROT_READ,     1
.equ MAP_SHARED,    1
.equ BAR0_SIZE,     0x1000000

// GSP Falcon registers (BAR0 offsets)
.equ FALCON_IRQSTAT,    0x110008
.equ FALCON_IRQMASK,    0x110018
.equ FALCON_MAILBOX0,   0x110040
.equ FALCON_MAILBOX1,   0x110044
.equ FALCON_ITFEN,      0x110048
.equ FALCON_OS,          0x110080
.equ FALCON_CPUCTL,     0x110100
.equ FALCON_BOOTVEC,    0x110104
.equ FALCON_HWCFG,      0x1100F0
.equ FALCON_HWCFG2,     0x1100F4
.equ FALCON_ENGINE,     0x1103C0
.equ RISCV_CPUCTL,      0x111388
.equ RISCV_BR_RETCODE,  0x11138C

// NV_PGSP_EMEMC/EMEMD (GSP-side, distinct from FSP EMEM)
.equ GSP_QUEUE_HEAD,    0x110C00
.equ GSP_QUEUE_TAIL,    0x110C04

.data
.align 3

bar0_path:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource0\0"

msg_header:     .ascii "=== Lithos GSP Falcon Probe (read-only) ===\n\n"
msg_header_len= . - msg_header

// Register labels (fixed-width for alignment)
msg_engine:     .ascii "FALCON_ENGINE   = 0x"
msg_engine_len= . - msg_engine

msg_cpuctl:     .ascii "FALCON_CPUCTL   = 0x"
msg_cpuctl_len= . - msg_cpuctl

msg_hwcfg:      .ascii "FALCON_HWCFG    = 0x"
msg_hwcfg_len = . - msg_hwcfg

msg_hwcfg2:     .ascii "FALCON_HWCFG2   = 0x"
msg_hwcfg2_len= . - msg_hwcfg2

msg_mbox0:      .ascii "FALCON_MAILBOX0 = 0x"
msg_mbox0_len = . - msg_mbox0

msg_mbox1:      .ascii "FALCON_MAILBOX1 = 0x"
msg_mbox1_len = . - msg_mbox1

msg_irqstat:    .ascii "FALCON_IRQSTAT  = 0x"
msg_irq_len   = . - msg_irqstat

msg_irqmask:    .ascii "FALCON_IRQMASK  = 0x"
msg_irqm_len  = . - msg_irqmask

msg_os:         .ascii "FALCON_OS       = 0x"
msg_os_len    = . - msg_os

msg_itfen:      .ascii "FALCON_ITFEN    = 0x"
msg_itfen_len = . - msg_itfen

msg_bootvec:    .ascii "FALCON_BOOTVEC  = 0x"
msg_bv_len    = . - msg_bootvec

msg_riscv_cpu:  .ascii "RISCV_CPUCTL    = 0x"
msg_rv_cpu_len= . - msg_riscv_cpu

msg_riscv_ret:  .ascii "RISCV_BR_RETCODE= 0x"
msg_rv_ret_len= . - msg_riscv_ret

msg_gsp_qh:     .ascii "GSP_QUEUE_HEAD  = 0x"
msg_gsp_qh_len= . - msg_gsp_qh

msg_gsp_qt:     .ascii "GSP_QUEUE_TAIL  = 0x"
msg_gsp_qt_len= . - msg_gsp_qt

// Interpretation
msg_reset:      .ascii "  -> Falcon reset state: "
msg_reset_len = . - msg_reset

msg_in_reset:   .ascii "IN RESET\n"
msg_inr_len   = . - msg_in_reset

msg_running:    .ascii "RUNNING\n"
msg_run_len   = . - msg_running

msg_lockdown:   .ascii "  -> PRIV_LOCKDOWN:      "
msg_ld_len    = . - msg_lockdown

msg_locked:     .ascii "LOCKED\n"
msg_lk_len    = . - msg_locked

msg_released:   .ascii "RELEASED\n"
msg_rel_len   = . - msg_released

msg_riscv_st:   .ascii "  -> RISC-V CPU:         "
msg_rv_st_len = . - msg_riscv_st

msg_rv_started: .ascii "STARTED\n"
msg_rv_s_len  = . - msg_rv_started

msg_rv_stopped: .ascii "STOPPED\n"
msg_rv_x_len  = . - msg_rv_stopped

msg_newline:    .ascii "\n"

msg_open_fail:  .ascii "probe_gsp_falcon: failed to open resource0\n"
msg_opf_len   = . - msg_open_fail

msg_mmap_fail:  .ascii "probe_gsp_falcon: mmap failed\n"
msg_mmf_len   = . - msg_mmap_fail

.text
.align 4

// Macro: load large BAR0 offset, read register
// Clobbers x1
.macro read_bar0_reg dst, base, offset
    movz    x1, #(\offset & 0xFFFF)
    .if (\offset >> 16) != 0
    movk    x1, #(\offset >> 16), lsl #16
    .endif
    ldr     \dst, [\base, x1]
.endm

.global _start
_start:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Header
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Open + mmap BAR0
    mov     x0, #AT_FDCWD
    adrp    x1, bar0_path
    add     x1, x1, :lo12:bar0_path
    movz    x2, #0x1002
    movk    x2, #0x0010, lsl #16
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0

    mov     x0, #0
    mov     x1, #BAR0_SIZE
    mov     x2, #PROT_READ
    mov     x3, #MAP_SHARED
    mov     x4, x19
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .mmap_fail
    mov     x20, x0

    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // ---- Read all Falcon registers ----

    // FALCON_ENGINE
    read_bar0_reg w21, x20, FALCON_ENGINE
    adrp    x1, msg_engine
    add     x1, x1, :lo12:msg_engine
    mov     w2, #msg_engine_len
    mov     w0, w21
    bl      .print_reg

    // Reset state interpretation
    and     w0, w21, #0x700
    mov     x0, #1
    adrp    x1, msg_reset
    add     x1, x1, :lo12:msg_reset
    mov     x2, #msg_reset_len
    mov     x8, #SYS_WRITE
    svc     #0
    and     w0, w21, #0x700
    cmp     w0, #0x200
    b.eq    .falcon_running
    mov     x0, #1
    adrp    x1, msg_in_reset
    add     x1, x1, :lo12:msg_in_reset
    mov     x2, #msg_inr_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .reset_done
.falcon_running:
    mov     x0, #1
    adrp    x1, msg_running
    add     x1, x1, :lo12:msg_running
    mov     x2, #msg_run_len
    mov     x8, #SYS_WRITE
    svc     #0
.reset_done:

    // FALCON_CPUCTL
    read_bar0_reg w21, x20, FALCON_CPUCTL
    adrp    x1, msg_cpuctl
    add     x1, x1, :lo12:msg_cpuctl
    mov     w2, #msg_cpuctl_len
    mov     w0, w21
    bl      .print_reg

    // FALCON_HWCFG
    read_bar0_reg w21, x20, FALCON_HWCFG
    adrp    x1, msg_hwcfg
    add     x1, x1, :lo12:msg_hwcfg
    mov     w2, #msg_hwcfg_len
    mov     w0, w21
    bl      .print_reg

    // FALCON_HWCFG2
    read_bar0_reg w21, x20, FALCON_HWCFG2
    adrp    x1, msg_hwcfg2
    add     x1, x1, :lo12:msg_hwcfg2
    mov     w2, #msg_hwcfg2_len
    mov     w0, w21
    bl      .print_reg

    // PRIV_LOCKDOWN interpretation
    mov     x0, #1
    adrp    x1, msg_lockdown
    add     x1, x1, :lo12:msg_lockdown
    mov     x2, #msg_ld_len
    mov     x8, #SYS_WRITE
    svc     #0
    tbnz    w21, #13, .is_locked
    mov     x0, #1
    adrp    x1, msg_released
    add     x1, x1, :lo12:msg_released
    mov     x2, #msg_rel_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .lock_done
.is_locked:
    mov     x0, #1
    adrp    x1, msg_locked
    add     x1, x1, :lo12:msg_locked
    mov     x2, #msg_lk_len
    mov     x8, #SYS_WRITE
    svc     #0
.lock_done:

    // MAILBOX0 / MAILBOX1
    read_bar0_reg w21, x20, FALCON_MAILBOX0
    adrp    x1, msg_mbox0
    add     x1, x1, :lo12:msg_mbox0
    mov     w2, #msg_mbox0_len
    mov     w0, w21
    bl      .print_reg

    read_bar0_reg w21, x20, FALCON_MAILBOX1
    adrp    x1, msg_mbox1
    add     x1, x1, :lo12:msg_mbox1
    mov     w2, #msg_mbox1_len
    mov     w0, w21
    bl      .print_reg

    // IRQSTAT / IRQMASK
    read_bar0_reg w21, x20, FALCON_IRQSTAT
    adrp    x1, msg_irqstat
    add     x1, x1, :lo12:msg_irqstat
    mov     w2, #msg_irq_len
    mov     w0, w21
    bl      .print_reg

    read_bar0_reg w21, x20, FALCON_IRQMASK
    adrp    x1, msg_irqmask
    add     x1, x1, :lo12:msg_irqmask
    mov     w2, #msg_irqm_len
    mov     w0, w21
    bl      .print_reg

    // FALCON_OS / ITFEN / BOOTVEC
    read_bar0_reg w21, x20, FALCON_OS
    adrp    x1, msg_os
    add     x1, x1, :lo12:msg_os
    mov     w2, #msg_os_len
    mov     w0, w21
    bl      .print_reg

    read_bar0_reg w21, x20, FALCON_ITFEN
    adrp    x1, msg_itfen
    add     x1, x1, :lo12:msg_itfen
    mov     w2, #msg_itfen_len
    mov     w0, w21
    bl      .print_reg

    read_bar0_reg w21, x20, FALCON_BOOTVEC
    adrp    x1, msg_bootvec
    add     x1, x1, :lo12:msg_bootvec
    mov     w2, #msg_bv_len
    mov     w0, w21
    bl      .print_reg

    // RISC-V CPU status
    read_bar0_reg w21, x20, RISCV_CPUCTL
    adrp    x1, msg_riscv_cpu
    add     x1, x1, :lo12:msg_riscv_cpu
    mov     w2, #msg_rv_cpu_len
    mov     w0, w21
    bl      .print_reg

    // RISC-V interpretation
    mov     x0, #1
    adrp    x1, msg_riscv_st
    add     x1, x1, :lo12:msg_riscv_st
    mov     x2, #msg_rv_st_len
    mov     x8, #SYS_WRITE
    svc     #0
    tst     w21, #1
    b.eq    .rv_stopped
    mov     x0, #1
    adrp    x1, msg_rv_started
    add     x1, x1, :lo12:msg_rv_started
    mov     x2, #msg_rv_s_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .rv_done
.rv_stopped:
    mov     x0, #1
    adrp    x1, msg_rv_stopped
    add     x1, x1, :lo12:msg_rv_stopped
    mov     x2, #msg_rv_x_len
    mov     x8, #SYS_WRITE
    svc     #0
.rv_done:

    // RISC-V return code
    read_bar0_reg w21, x20, RISCV_BR_RETCODE
    adrp    x1, msg_riscv_ret
    add     x1, x1, :lo12:msg_riscv_ret
    mov     w2, #msg_rv_ret_len
    mov     w0, w21
    bl      .print_reg

    // GSP message queue HEAD/TAIL
    read_bar0_reg w21, x20, GSP_QUEUE_HEAD
    adrp    x1, msg_gsp_qh
    add     x1, x1, :lo12:msg_gsp_qh
    mov     w2, #msg_gsp_qh_len
    mov     w0, w21
    bl      .print_reg

    read_bar0_reg w21, x20, GSP_QUEUE_TAIL
    adrp    x1, msg_gsp_qt
    add     x1, x1, :lo12:msg_gsp_qt
    mov     w2, #msg_gsp_qt_len
    mov     w0, w21
    bl      .print_reg

    // Exit 0
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
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
.print_reg:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     w22, w0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w22
    bl      .print_hex32
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #16
    ret

.print_hex32:
    sub     sp, sp, #16
    mov     w3, #28
    add     x4, sp, #0
.h32_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .h32_d
    add     w5, w5, #('a' - 10)
    b       .h32_s
.h32_d:
    add     w5, w5, #'0'
.h32_s:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .h32_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret
