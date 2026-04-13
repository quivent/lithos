// probe_sm.s — Lithos SM Topology Probe (ARM64, raw syscalls, no libc)
// Reads GH200 GPU BAR0 registers via mmap of sysfs resource0.
// Build: as -o probe_sm.o probe_sm.s && ld -o probe_sm probe_sm.o
// Run:   sudo ./probe_sm

.global _start

// ── syscall numbers (aarch64) ──────────────────────────────────────
.equ SYS_write,     64
.equ SYS_openat,    56
.equ SYS_close,     57
.equ SYS_mmap,      222
.equ SYS_munmap,    215
.equ SYS_exit,      93
.equ AT_FDCWD,      -100

// mmap flags
.equ PROT_READ,     1
.equ MAP_SHARED,    1
.equ O_RDONLY,      0

// BAR0 size (16 MiB)
.equ BAR0_SIZE,     0x01000000

// GPU priv-fault sentinel
.equ PRIV_FAULT,    0xbadf5040

// ── BAR0 register offsets ──────────────────────────────────────────
.equ PMC_BOOT_0,                0x000000
.equ PMC_BOOT_42,               0x000088
.equ FUSE_STATUS_OPT_GPC,       0x021C1C
.equ FUSE_STATUS_OPT_TPC_GPC0,  0x021C38
.equ PTOP_SCAL_NUM_GPCS,        0x022430
.equ PTOP_SCAL_NUM_TPC_PER_GPC, 0x022434
.equ PTOP_SCAL_NUM_FBPS,        0x022438
.equ PTOP_SCAL_NUM_FBPA_PER_FBP,0x02243C
.equ PTOP_SCAL_NUM_LTC_PER_FBP, 0x02244C

// ── .data ──────────────────────────────────────────────────────────
.section .data

bar0_path:
    .asciz "/sys/bus/pci/devices/0000:dd:00.0/resource0"

s_banner:    .asciz "\n=== Lithos SM Topology Probe ===\n\n"
s_err_open:  .asciz "ERROR: cannot open BAR0 resource0\n"
s_err_mmap:  .asciz "ERROR: mmap BAR0 failed\n"

s_boot0:         .asciz "PMC_BOOT_0                = 0x"
s_num_gpcs:      .asciz "PTOP_SCAL_NUM_GPCS        = "
s_num_tpc:       .asciz "PTOP_SCAL_NUM_TPC_PER_GPC = "
s_fuse_gpc:      .asciz "FUSE_STATUS_OPT_GPC       = "
s_fuse_ok_0x:    .asciz "0x"
s_gpc_dis:       .asciz " ("
s_gpc_dis2:      .asciz " GPCs disabled)\n"
s_fuse_tpc_hdr:  .asciz "FUSE_STATUS_OPT_TPC_GPC:\n"
s_gpc_prefix:    .asciz "  GPC["
s_eq_0x:         .asciz "] = "
s_tpc_dis:       .asciz " ("
s_tpc_dis2:      .asciz " TPCs disabled)\n"
s_newline:       .asciz "\n"
s_num_fbps:      .asciz "PTOP_SCAL_NUM_FBPS        = "
s_fbpa_per_fbp:  .asciz "PTOP_SCAL_NUM_FBPA_PER_FBP= "
s_ltc_per_fbp:   .asciz "PTOP_SCAL_NUM_LTC_PER_FBP = "
s_boot42:        .asciz "PMC_BOOT_42               = "
s_summary:       .asciz "\nSummary:\n"
s_act_gpcs:      .asciz "  Active GPCs     = "
s_act_tpcs:      .asciz "  Active TPCs     = "
s_act_sms:       .asciz "  Active SMs      = "
s_cuda_cores:    .asciz "  Est. CUDA cores = "
s_mem_part:      .asciz "  Memory partitions = "
s_priv_fault:    .asciz "PRIV-FAULTED (0xbadf5040)\n"
s_priv_short:    .asciz "PRIV-FAULTED"
s_fuse_note:     .asciz "  (fuse regs priv-faulted; PTOP_SCAL values are post-fuse active counts)\n"

// ── .bss ───────────────────────────────────────────────────────────
.section .bss
.align 4
num_buf:   .skip 32

// ── .text ──────────────────────────────────────────────────────────
.section .text

// ── Macro: load 32-bit BAR0 reg into Wd via Xbase, using x9 scratch
.macro LOAD_BAR0_REG rd, base, offset
    movz x9, #(\offset & 0xffff)
    .if (\offset >> 16) != 0
    movk x9, #(\offset >> 16), lsl #16
    .endif
    ldr  \rd, [\base, x9]
.endm

// ── Macro: load 32-bit PRIV_FAULT constant into Wd
.macro LOAD_PRIV_FAULT rd
    movz \rd, #(PRIV_FAULT & 0xffff)
    movk \rd, #(PRIV_FAULT >> 16), lsl #16
.endm

// ────────────────────────────────────────────────────────────────────
// print null-terminated string at x0
// ────────────────────────────────────────────────────────────────────
print_str:
    stp x29, x30, [sp, #-16]!
    mov x9, x0
    mov x1, x0
1:  ldrb w2, [x1], #1
    cbnz w2, 1b
    sub x2, x1, x9
    sub x2, x2, #1
    mov x0, #1
    mov x1, x9
    mov x8, #SYS_write
    svc #0
    ldp x29, x30, [sp], #16
    ret

// ────────────────────────────────────────────────────────────────────
// print unsigned 32-bit decimal in w0 + newline
// ────────────────────────────────────────────────────────────────────
print_dec_nl:
    stp x29, x30, [sp, #-16]!
    mov w10, w0
    adrp x11, num_buf
    add  x11, x11, :lo12:num_buf
    add  x13, x11, #30
    mov  w14, #'\n'
    strb w14, [x13]
    add  x15, x13, #1

    mov  x12, x13
    cbz  w10, 2f
1:  cbz  w10, 3f
    mov  w16, #10
    udiv w17, w10, w16
    msub w18, w17, w16, w10
    add  w18, w18, #'0'
    sub  x12, x12, #1
    strb w18, [x12]
    mov  w10, w17
    b    1b
2:  mov  w18, #'0'
    sub  x12, x12, #1
    strb w18, [x12]
3:  mov  x0, #1
    mov  x1, x12
    sub  x2, x15, x12
    mov  x8, #SYS_write
    svc  #0
    ldp  x29, x30, [sp], #16
    ret

// ────────────────────────────────────────────────────────────────────
// print unsigned 32-bit decimal in w0, no newline
// ────────────────────────────────────────────────────────────────────
print_dec:
    stp x29, x30, [sp, #-16]!
    mov w10, w0
    adrp x11, num_buf
    add  x11, x11, :lo12:num_buf
    add  x13, x11, #30
    mov  x12, x13

    cbz  w10, 2f
1:  cbz  w10, 3f
    mov  w16, #10
    udiv w17, w10, w16
    msub w18, w17, w16, w10
    add  w18, w18, #'0'
    sub  x12, x12, #1
    strb w18, [x12]
    mov  w10, w17
    b    1b
2:  mov  w18, #'0'
    sub  x12, x12, #1
    strb w18, [x12]
3:  mov  x0, #1
    mov  x1, x12
    sub  x2, x13, x12
    mov  x8, #SYS_write
    svc  #0
    ldp  x29, x30, [sp], #16
    ret

// ────────────────────────────────────────────────────────────────────
// print 32-bit hex (8 digits), no newline
// ────────────────────────────────────────────────────────────────────
print_hex32:
    stp x29, x30, [sp, #-16]!
    mov w10, w0
    adrp x11, num_buf
    add  x11, x11, :lo12:num_buf
    mov w12, #8
1:  sub w12, w12, #1
    and w14, w10, #0xf
    cmp w14, #10
    b.lt 2f
    add w14, w14, #('a' - 10)
    b 3f
2:  add w14, w14, #'0'
3:  strb w14, [x11, x12]
    lsr w10, w10, #4
    cbnz w12, 1b

    mov  x0, #1
    mov  x1, x11
    mov  x2, #8
    mov  x8, #SYS_write
    svc  #0
    ldp  x29, x30, [sp], #16
    ret

// ────────────────────────────────────────────────────────────────────
// print 8-bit hex (2 digits), no newline
// ────────────────────────────────────────────────────────────────────
print_hex8:
    stp x29, x30, [sp, #-16]!
    and w10, w0, #0xff
    adrp x11, num_buf
    add  x11, x11, :lo12:num_buf
    lsr  w12, w10, #4
    cmp  w12, #10
    b.lt 1f
    add  w12, w12, #('a' - 10)
    b    2f
1:  add  w12, w12, #'0'
2:  strb w12, [x11]
    and  w12, w10, #0xf
    cmp  w12, #10
    b.lt 3f
    add  w12, w12, #('a' - 10)
    b    4f
3:  add  w12, w12, #'0'
4:  strb w12, [x11, #1]

    mov  x0, #1
    mov  x1, x11
    mov  x2, #2
    mov  x8, #SYS_write
    svc  #0
    ldp  x29, x30, [sp], #16
    ret

// ────────────────────────────────────────────────────────────────────
// popcount w0 -> w0
// ────────────────────────────────────────────────────────────────────
popcount32:
    fmov s0, w0
    cnt  v0.8b, v0.8b
    addv b0, v0.8b
    fmov w0, s0
    ret

// ════════════════════════════════════════════════════════════════════
// ENTRY
// ════════════════════════════════════════════════════════════════════
_start:
    // ── banner ─────────────────────────────────────────────────────
    adrp x0, s_banner
    add  x0, x0, :lo12:s_banner
    bl   print_str

    // ── open BAR0 ──────────────────────────────────────────────────
    mov  x0, #AT_FDCWD
    adrp x1, bar0_path
    add  x1, x1, :lo12:bar0_path
    mov  x2, #O_RDONLY
    mov  x3, #0
    mov  x8, #SYS_openat
    svc  #0
    tbnz x0, #63, .Lerr_open
    mov  x19, x0

    // ── mmap BAR0 ──────────────────────────────────────────────────
    mov  x0, #0
    mov  x1, #BAR0_SIZE
    mov  x2, #PROT_READ
    mov  x3, #MAP_SHARED
    mov  x4, x19
    mov  x5, #0
    mov  x8, #SYS_mmap
    svc  #0
    cmn  x0, #4096
    b.hs .Lerr_mmap
    mov  x20, x0              // BAR0 base

    mov  x0, x19
    mov  x8, #SYS_close
    svc  #0

    // ════════════════════════════════════════════════════════════════
    // Read registers
    // ════════════════════════════════════════════════════════════════

    // ── PMC_BOOT_0 (always accessible) ─────────────────────────────
    adrp x0, s_boot0
    add  x0, x0, :lo12:s_boot0
    bl   print_str
    ldr  w3, [x20, #PMC_BOOT_0]
    mov  w0, w3
    bl   print_hex32
    adrp x0, s_newline
    add  x0, x0, :lo12:s_newline
    bl   print_str

    // ── PTOP_SCAL_NUM_GPCS ─────────────────────────────────────────
    adrp x0, s_num_gpcs
    add  x0, x0, :lo12:s_num_gpcs
    bl   print_str
    LOAD_BAR0_REG w21, x20, PTOP_SCAL_NUM_GPCS
    mov  w0, w21
    bl   print_dec_nl

    // ── PTOP_SCAL_NUM_TPC_PER_GPC ──────────────────────────────────
    adrp x0, s_num_tpc
    add  x0, x0, :lo12:s_num_tpc
    bl   print_str
    LOAD_BAR0_REG w22, x20, PTOP_SCAL_NUM_TPC_PER_GPC
    mov  w0, w22
    bl   print_dec_nl

    // ── FUSE_STATUS_OPT_GPC ────────────────────────────────────────
    adrp x0, s_fuse_gpc
    add  x0, x0, :lo12:s_fuse_gpc
    bl   print_str
    LOAD_BAR0_REG w23, x20, FUSE_STATUS_OPT_GPC
    // Check for priv-fault
    LOAD_PRIV_FAULT w9
    cmp  w23, w9
    b.ne .Lfuse_gpc_ok
    // faulted
    adrp x0, s_priv_fault
    add  x0, x0, :lo12:s_priv_fault
    bl   print_str
    mov  w24, #0              // assume 0 disabled
    b    .Lfuse_gpc_done
.Lfuse_gpc_ok:
    adrp x0, s_fuse_ok_0x
    add  x0, x0, :lo12:s_fuse_ok_0x
    bl   print_str
    mov  w0, w23
    bl   print_hex32
    // count disabled GPCs
    mov  w0, #1
    lsl  w0, w0, w21
    sub  w0, w0, #1
    and  w0, w23, w0
    bl   popcount32
    mov  w24, w0
    adrp x0, s_gpc_dis
    add  x0, x0, :lo12:s_gpc_dis
    bl   print_str
    mov  w0, w24
    bl   print_dec
    adrp x0, s_gpc_dis2
    add  x0, x0, :lo12:s_gpc_dis2
    bl   print_str
.Lfuse_gpc_done:

    // ── FUSE_STATUS_OPT_TPC_GPC per GPC ────────────────────────────
    adrp x0, s_fuse_tpc_hdr
    add  x0, x0, :lo12:s_fuse_tpc_hdr
    bl   print_str

    mov  w25, #0             // i = 0
    mov  w26, #0             // total disabled TPCs
    LOAD_PRIV_FAULT w9       // keep fault sentinel handy

.Ltpc_loop:
    cmp  w25, w21
    b.ge .Ltpc_done

    adrp x0, s_gpc_prefix
    add  x0, x0, :lo12:s_gpc_prefix
    bl   print_str
    mov  w0, w25
    bl   print_dec
    adrp x0, s_eq_0x
    add  x0, x0, :lo12:s_eq_0x
    bl   print_str

    // read FUSE_STATUS_OPT_TPC_GPC(i) = 0x021C38 + i*4
    movz x9, #(FUSE_STATUS_OPT_TPC_GPC0 & 0xffff)
    movk x9, #(FUSE_STATUS_OPT_TPC_GPC0 >> 16), lsl #16
    add  x9, x9, x25, lsl #2
    ldr  w27, [x20, x9]

    // check priv-fault
    LOAD_PRIV_FAULT w9
    cmp  w27, w9
    b.ne .Ltpc_val_ok
    // faulted
    adrp x0, s_priv_fault
    add  x0, x0, :lo12:s_priv_fault
    bl   print_str
    b    .Ltpc_next

.Ltpc_val_ok:
    adrp x0, s_fuse_ok_0x
    add  x0, x0, :lo12:s_fuse_ok_0x
    bl   print_str
    mov  w0, w27
    bl   print_hex8
    // count disabled TPCs
    mov  w0, #1
    lsl  w0, w0, w22
    sub  w0, w0, #1
    and  w0, w27, w0
    bl   popcount32
    mov  w28, w0
    add  w26, w26, w28
    adrp x0, s_tpc_dis
    add  x0, x0, :lo12:s_tpc_dis
    bl   print_str
    mov  w0, w28
    bl   print_dec
    adrp x0, s_tpc_dis2
    add  x0, x0, :lo12:s_tpc_dis2
    bl   print_str

.Ltpc_next:
    add  w25, w25, #1
    b    .Ltpc_loop
.Ltpc_done:

    // ── newline ────────────────────────────────────────────────────
    adrp x0, s_newline
    add  x0, x0, :lo12:s_newline
    bl   print_str

    // ── PTOP_SCAL_NUM_FBPS ─────────────────────────────────────────
    adrp x0, s_num_fbps
    add  x0, x0, :lo12:s_num_fbps
    bl   print_str
    LOAD_BAR0_REG w3, x20, PTOP_SCAL_NUM_FBPS
    stp  x3, xzr, [sp, #-16]!
    mov  w0, w3
    bl   print_dec_nl
    ldp  x3, xzr, [sp], #16
    mov  w28, w3             // w28 = num_fbps

    // ── PTOP_SCAL_NUM_FBPA_PER_FBP ─────────────────────────────────
    adrp x0, s_fbpa_per_fbp
    add  x0, x0, :lo12:s_fbpa_per_fbp
    bl   print_str
    LOAD_BAR0_REG w3, x20, PTOP_SCAL_NUM_FBPA_PER_FBP
    mov  w0, w3
    bl   print_dec_nl

    // ── PTOP_SCAL_NUM_LTC_PER_FBP ──────────────────────────────────
    adrp x0, s_ltc_per_fbp
    add  x0, x0, :lo12:s_ltc_per_fbp
    bl   print_str
    LOAD_BAR0_REG w3, x20, PTOP_SCAL_NUM_LTC_PER_FBP
    mov  w0, w3
    bl   print_dec_nl

    // ── newline ────────────────────────────────────────────────────
    adrp x0, s_newline
    add  x0, x0, :lo12:s_newline
    bl   print_str

    // ── PMC_BOOT_42 ────────────────────────────────────────────────
    adrp x0, s_boot42
    add  x0, x0, :lo12:s_boot42
    bl   print_str
    ldr  w3, [x20, #PMC_BOOT_42]
    LOAD_PRIV_FAULT w9
    cmp  w3, w9
    b.ne .Lboot42_ok
    adrp x0, s_priv_fault
    add  x0, x0, :lo12:s_priv_fault
    bl   print_str
    b    .Lboot42_done
.Lboot42_ok:
    adrp x0, s_fuse_ok_0x
    add  x0, x0, :lo12:s_fuse_ok_0x
    bl   print_str
    mov  w0, w3
    bl   print_hex32
    adrp x0, s_newline
    add  x0, x0, :lo12:s_newline
    bl   print_str
.Lboot42_done:

    // ════════════════════════════════════════════════════════════════
    // Summary
    // PTOP_SCAL values are already post-fuse (active counts).
    // w21 = num_gpcs (active), w22 = tpc_per_gpc (max active per GPC)
    // w26 = disabled TPCs (from fuse, or 0 if faulted)
    // w28 = num_fbps
    // ════════════════════════════════════════════════════════════════
    adrp x0, s_summary
    add  x0, x0, :lo12:s_summary
    bl   print_str

    // Note about fuse regs
    // Check if fuse was faulted (w24 == 0 and w23 == PRIV_FAULT)
    LOAD_PRIV_FAULT w9
    cmp  w23, w9
    b.ne .Lno_fuse_note
    adrp x0, s_fuse_note
    add  x0, x0, :lo12:s_fuse_note
    bl   print_str
.Lno_fuse_note:

    // Active GPCs = num_gpcs - disabled_gpcs
    sub  w25, w21, w24
    adrp x0, s_act_gpcs
    add  x0, x0, :lo12:s_act_gpcs
    bl   print_str
    mov  w0, w25
    bl   print_dec_nl

    // Active TPCs = (num_gpcs * tpc_per_gpc) - disabled_tpcs
    mul  w27, w21, w22
    sub  w27, w27, w26
    adrp x0, s_act_tpcs
    add  x0, x0, :lo12:s_act_tpcs
    bl   print_str
    mov  w0, w27
    bl   print_dec_nl

    // Active SMs = TPCs * 2 (Hopper: 2 SMs per TPC)
    lsl  w29, w27, #1
    adrp x0, s_act_sms
    add  x0, x0, :lo12:s_act_sms
    bl   print_str
    mov  w0, w29
    bl   print_dec_nl

    // Est. CUDA cores = SMs * 128
    lsl  w3, w29, #7
    stp  x3, xzr, [sp, #-16]!
    adrp x0, s_cuda_cores
    add  x0, x0, :lo12:s_cuda_cores
    bl   print_str
    ldp  x3, xzr, [sp], #16
    mov  w0, w3
    bl   print_dec_nl

    // Memory partitions
    adrp x0, s_mem_part
    add  x0, x0, :lo12:s_mem_part
    bl   print_str
    mov  w0, w28
    bl   print_dec_nl

    adrp x0, s_newline
    add  x0, x0, :lo12:s_newline
    bl   print_str

    // ── cleanup ────────────────────────────────────────────────────
    mov  x0, x20
    mov  x1, #BAR0_SIZE
    mov  x8, #SYS_munmap
    svc  #0

    mov  x0, #0
    mov  x8, #SYS_exit
    svc  #0

// ── error paths ────────────────────────────────────────────────────
.Lerr_open:
    adrp x0, s_err_open
    add  x0, x0, :lo12:s_err_open
    bl   print_str
    mov  x0, #1
    mov  x8, #SYS_exit
    svc  #0

.Lerr_mmap:
    adrp x0, s_err_mmap
    add  x0, x0, :lo12:s_err_mmap
    bl   print_str
    mov  x0, #1
    mov  x8, #SYS_exit
    svc  #0
