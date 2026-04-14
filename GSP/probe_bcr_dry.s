// probe_bcr_dry.s -- Dry-run BCR probe: programs BCR registers, does NOT start RISC-V
//
// Runs steps 1-5 of the GSP boot sequence, then programs all BCR registers
// exactly as bcr_start.s does (MAILBOX0/1, BCR_FMCCODE_LO/HI, BCR_FMCDATA_LO/HI,
// BCR_PKCPARAM_LO/HI, BCR_DMACFG with LOCK).  Critically, it does NOT write
// CPUCTL=1, so the RISC-V never starts.  After programming, it reads back
// every register and prints the values for validation.
//
// Build:
//   as -o probe_bcr_dry.o probe_bcr_dry.s
//
// Link (requires the same objects as gsp_boot):
//   ld -o probe_bcr_dry probe_bcr_dry.o bar_map.o pmc_check.o \
//      falcon_reset.o hbm_alloc.o fw_load.o -static
//
// Run:
//   sudo ./probe_bcr_dry
//
// Exit codes:
//   0 = dry run complete (all BCR regs programmed and verified)
//   1 = bar_map_init failed
//   2 = pmc_check failed
//   3 = falcon_reset failed
//   4 = hbm_alloc_init failed
//   5 = BCR DMACFG LOCK failed

.equ SYS_WRITE,     64
.equ SYS_EXIT,      93

// BAR4 reservation -- same as boot.s
.equ GSP_RESERVED_BYTES, 0x04000000           // 64 MB

// BCR address shift (8-byte alignment)
.equ RISCV_BR_ADDR_SHIFT, 3

// ============================================================
// External symbols (same as boot.s)
// ============================================================
.extern bar_map_init
.extern pmc_check
.extern falcon_reset
.extern hbm_alloc_init
.extern gsp_fw_load

.extern bar0_base
.extern bar4_base
.extern bar4_phys

.extern fw_bar4_phys
.extern fw_code_offset
.extern fw_data_offset
.extern fw_manifest_offset

// ============================================================
// Data section -- messages
// ============================================================
.data
.align 3

msg_step1_begin:     .asciz "probe_bcr_dry: [1/5] bar_map_init...\n"
msg_step1_begin_len  = . - msg_step1_begin - 1
msg_step1_ok:        .asciz "probe_bcr_dry: [1/5] bar_map_init -- OK\n"
msg_step1_ok_len     = . - msg_step1_ok - 1
msg_step1_fail:      .asciz "probe_bcr_dry: [1/5] bar_map_init -- FAILED\n"
msg_step1_fail_len   = . - msg_step1_fail - 1

msg_step2_begin:     .asciz "probe_bcr_dry: [2/5] pmc_check...\n"
msg_step2_begin_len  = . - msg_step2_begin - 1
msg_step2_ok:        .asciz "probe_bcr_dry: [2/5] pmc_check -- OK\n"
msg_step2_ok_len     = . - msg_step2_ok - 1
msg_step2_fail:      .asciz "probe_bcr_dry: [2/5] pmc_check -- FAILED\n"
msg_step2_fail_len   = . - msg_step2_fail - 1

msg_step3_begin:     .asciz "probe_bcr_dry: [3/5] falcon_reset...\n"
msg_step3_begin_len  = . - msg_step3_begin - 1
msg_step3_ok:        .asciz "probe_bcr_dry: [3/5] falcon_reset -- OK\n"
msg_step3_ok_len     = . - msg_step3_ok - 1
msg_step3_fail:      .asciz "probe_bcr_dry: [3/5] falcon_reset -- FAILED\n"
msg_step3_fail_len   = . - msg_step3_fail - 1

msg_step4_begin:     .asciz "probe_bcr_dry: [4/5] hbm_alloc_init...\n"
msg_step4_begin_len  = . - msg_step4_begin - 1
msg_step4_ok:        .asciz "probe_bcr_dry: [4/5] hbm_alloc_init -- OK\n"
msg_step4_ok_len     = . - msg_step4_ok - 1
msg_step4_fail:      .asciz "probe_bcr_dry: [4/5] hbm_alloc_init -- FAILED\n"
msg_step4_fail_len   = . - msg_step4_fail - 1

msg_step5_begin:     .asciz "probe_bcr_dry: [5/5] gsp_fw_load...\n"
msg_step5_begin_len  = . - msg_step5_begin - 1
msg_step5_ok:        .asciz "probe_bcr_dry: [5/5] gsp_fw_load -- OK\n"
msg_step5_ok_len     = . - msg_step5_ok - 1

// BCR dry run banners
msg_banner:          .asciz "=== BCR DRY RUN (no CPUCTL write) ===\n"
msg_banner_len       = . - msg_banner - 1

msg_warn_params:     .asciz "WARNING: fmc_params_pa = 0 (placeholder, FSP not implemented)\n"
msg_warn_params_len  = . - msg_warn_params - 1

// Input value labels
msg_fmc_params_pa:   .asciz "fmc_params_pa   = 0x"
msg_fmc_params_len   = . - msg_fmc_params_pa - 1
msg_fmc_image_pa:    .asciz "fmc_image_pa    = 0x"
msg_fmc_image_len    = . - msg_fmc_image_pa - 1
msg_code_offset:     .asciz "code_offset     = 0x"
msg_code_off_len     = . - msg_code_offset - 1
msg_data_offset:     .asciz "data_offset     = 0x"
msg_data_off_len     = . - msg_data_offset - 1
msg_manifest_offset: .asciz "manifest_offset = 0x"
msg_manifest_off_len = . - msg_manifest_offset - 1

// Computed BCR value labels
msg_bcr_fmccode:     .asciz "BCR_FMCCODE  = 0x"
msg_bcr_fmccode_len  = . - msg_bcr_fmccode - 1
msg_bcr_fmcdata:     .asciz "BCR_FMCDATA  = 0x"
msg_bcr_fmcdata_len  = . - msg_bcr_fmcdata - 1
msg_bcr_pkcparam:    .asciz "BCR_PKCPARAM = 0x"
msg_bcr_pkcparam_len = . - msg_bcr_pkcparam - 1

// Write confirmation
msg_writing:         .asciz "Writing BCR registers to hardware...\n"
msg_writing_len      = . - msg_writing - 1

// Readback labels
msg_rb_mbox0:        .asciz "MAILBOX0 readback  = 0x"
msg_rb_mbox0_len     = . - msg_rb_mbox0 - 1
msg_rb_mbox1:        .asciz "MAILBOX1 readback  = 0x"
msg_rb_mbox1_len     = . - msg_rb_mbox1 - 1
msg_rb_fmccode_lo:   .asciz "BCR_FMCCODE_LO rb  = 0x"
msg_rb_fmccode_lo_len = . - msg_rb_fmccode_lo - 1
msg_rb_fmccode_hi:   .asciz "BCR_FMCCODE_HI rb  = 0x"
msg_rb_fmccode_hi_len = . - msg_rb_fmccode_hi - 1
msg_rb_fmcdata_lo:   .asciz "BCR_FMCDATA_LO rb  = 0x"
msg_rb_fmcdata_lo_len = . - msg_rb_fmcdata_lo - 1
msg_rb_fmcdata_hi:   .asciz "BCR_FMCDATA_HI rb  = 0x"
msg_rb_fmcdata_hi_len = . - msg_rb_fmcdata_hi - 1
msg_rb_pkcparam_lo:  .asciz "BCR_PKCPARAM_LO rb = 0x"
msg_rb_pkcparam_lo_len = . - msg_rb_pkcparam_lo - 1
msg_rb_pkcparam_hi:  .asciz "BCR_PKCPARAM_HI rb = 0x"
msg_rb_pkcparam_hi_len = . - msg_rb_pkcparam_hi - 1
msg_rb_dmacfg:       .asciz "BCR_DMACFG rb      = 0x"
msg_rb_dmacfg_len    = . - msg_rb_dmacfg - 1

msg_lock_ok:         .asciz "LOCK = OK\n"
msg_lock_ok_len      = . - msg_lock_ok - 1
msg_lock_fail:       .asciz "LOCK = FAILED\n"
msg_lock_fail_len    = . - msg_lock_fail - 1

// Final banners
msg_no_cpuctl:       .asciz "=== CPUCTL NOT WRITTEN -- RISC-V NOT STARTED ===\n"
msg_no_cpuctl_len    = . - msg_no_cpuctl - 1
msg_complete:        .asciz "=== DRY RUN COMPLETE ===\n"
msg_complete_len     = . - msg_complete - 1

msg_newline:         .ascii "\n"

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // Save frame pointer / link register for the duration
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // =============================================================
    // Step 1: bar_map_init
    // =============================================================
    adrp    x1, msg_step1_begin
    add     x1, x1, :lo12:msg_step1_begin
    mov     x2, #msg_step1_begin_len
    bl      probe_print

    bl      bar_map_init
    cmp     x0, #0
    b.lt    .fail_step1

    adrp    x1, msg_step1_ok
    add     x1, x1, :lo12:msg_step1_ok
    mov     x2, #msg_step1_ok_len
    bl      probe_print

    // =============================================================
    // Step 2: pmc_check
    // =============================================================
    adrp    x1, msg_step2_begin
    add     x1, x1, :lo12:msg_step2_begin
    mov     x2, #msg_step2_begin_len
    bl      probe_print

    bl      pmc_check
    cmp     x0, #0
    b.lt    .fail_step2

    adrp    x1, msg_step2_ok
    add     x1, x1, :lo12:msg_step2_ok
    mov     x2, #msg_step2_ok_len
    bl      probe_print

    // =============================================================
    // Step 3: falcon_reset
    // =============================================================
    adrp    x1, msg_step3_begin
    add     x1, x1, :lo12:msg_step3_begin
    mov     x2, #msg_step3_begin_len
    bl      probe_print

    bl      falcon_reset
    cmp     x0, #0
    b.lt    .fail_step3

    adrp    x1, msg_step3_ok
    add     x1, x1, :lo12:msg_step3_ok
    mov     x2, #msg_step3_ok_len
    bl      probe_print

    // =============================================================
    // Step 4: hbm_alloc_init
    //   x0 = bar4 CPU va, x1 = bar4 phys, x2 = reserved offset
    // =============================================================
    adrp    x1, msg_step4_begin
    add     x1, x1, :lo12:msg_step4_begin
    mov     x2, #msg_step4_begin_len
    bl      probe_print

    adrp    x0, bar4_base
    add     x0, x0, :lo12:bar4_base
    ldr     x0, [x0]
    adrp    x1, bar4_phys
    add     x1, x1, :lo12:bar4_phys
    ldr     x1, [x1]
    mov     x2, #GSP_RESERVED_BYTES
    bl      hbm_alloc_init
    cmp     x0, #0
    b.lt    .fail_step4

    adrp    x1, msg_step4_ok
    add     x1, x1, :lo12:msg_step4_ok
    mov     x2, #msg_step4_ok_len
    bl      probe_print

    // =============================================================
    // Step 5: gsp_fw_load
    //   Exits internally on error; no return check needed.
    // =============================================================
    adrp    x1, msg_step5_begin
    add     x1, x1, :lo12:msg_step5_begin
    mov     x2, #msg_step5_begin_len
    bl      probe_print

    bl      gsp_fw_load

    adrp    x1, msg_step5_ok
    add     x1, x1, :lo12:msg_step5_ok
    mov     x2, #msg_step5_ok_len
    bl      probe_print

    // =============================================================
    // === DRY RUN BCR PROGRAMMING ===
    // =============================================================
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    bl      probe_print

    // ---- Load the same args boot.s loads for bcr_start ----
    // x19 = bar0_base (callee-saved, persists across prints)
    adrp    x19, bar0_base
    add     x19, x19, :lo12:bar0_base
    ldr     x19, [x19]

    // x20 = fmc_params_pa (placeholder 0)
    mov     x20, #0

    // x21 = fw_bar4_phys (fmc_image_pa)
    adrp    x21, fw_bar4_phys
    add     x21, x21, :lo12:fw_bar4_phys
    ldr     x21, [x21]

    // x22 = fw_code_offset
    adrp    x22, fw_code_offset
    add     x22, x22, :lo12:fw_code_offset
    ldr     x22, [x22]

    // x23 = fw_data_offset
    adrp    x23, fw_data_offset
    add     x23, x23, :lo12:fw_data_offset
    ldr     x23, [x23]

    // x24 = fw_manifest_offset
    adrp    x24, fw_manifest_offset
    add     x24, x24, :lo12:fw_manifest_offset
    ldr     x24, [x24]

    // ---- Print warning about placeholder fmc_params_pa ----
    adrp    x1, msg_warn_params
    add     x1, x1, :lo12:msg_warn_params
    mov     x2, #msg_warn_params_len
    bl      probe_print

    // ---- Print each input value ----

    // fmc_params_pa
    adrp    x1, msg_fmc_params_pa
    add     x1, x1, :lo12:msg_fmc_params_pa
    mov     x2, #msg_fmc_params_len
    mov     x0, x20
    bl      print_labeled_hex64

    // fmc_image_pa
    adrp    x1, msg_fmc_image_pa
    add     x1, x1, :lo12:msg_fmc_image_pa
    mov     x2, #msg_fmc_image_len
    mov     x0, x21
    bl      print_labeled_hex64

    // code_offset
    adrp    x1, msg_code_offset
    add     x1, x1, :lo12:msg_code_offset
    mov     x2, #msg_code_off_len
    mov     x0, x22
    bl      print_labeled_hex64

    // data_offset
    adrp    x1, msg_data_offset
    add     x1, x1, :lo12:msg_data_offset
    mov     x2, #msg_data_off_len
    mov     x0, x23
    bl      print_labeled_hex64

    // manifest_offset
    adrp    x1, msg_manifest_offset
    add     x1, x1, :lo12:msg_manifest_offset
    mov     x2, #msg_manifest_off_len
    mov     x0, x24
    bl      print_labeled_hex64

    // ---- Compute and print BCR values (without writing to hardware yet) ----

    // fmc_code_addr = (fmc_image_pa + code_offset) >> 3
    add     x25, x21, x22
    lsr     x25, x25, #RISCV_BR_ADDR_SHIFT
    adrp    x1, msg_bcr_fmccode
    add     x1, x1, :lo12:msg_bcr_fmccode
    mov     x2, #msg_bcr_fmccode_len
    mov     x0, x25
    bl      print_labeled_hex64

    // fmc_data_addr = (fmc_image_pa + data_offset) >> 3
    add     x26, x21, x23
    lsr     x26, x26, #RISCV_BR_ADDR_SHIFT
    adrp    x1, msg_bcr_fmcdata
    add     x1, x1, :lo12:msg_bcr_fmcdata
    mov     x2, #msg_bcr_fmcdata_len
    mov     x0, x26
    bl      print_labeled_hex64

    // fmc_manifest = (fmc_image_pa + manifest_offset) >> 3
    add     x27, x21, x24
    lsr     x27, x27, #RISCV_BR_ADDR_SHIFT
    adrp    x1, msg_bcr_pkcparam
    add     x1, x1, :lo12:msg_bcr_pkcparam
    mov     x2, #msg_bcr_pkcparam_len
    mov     x0, x27
    bl      print_labeled_hex64

    // ---- NOW write to hardware (all BCR regs EXCEPT CPUCTL) ----
    adrp    x1, msg_writing
    add     x1, x1, :lo12:msg_writing
    mov     x2, #msg_writing_len
    bl      probe_print

    // 1. MAILBOX0/1 = fmc_params_pa lo/hi
    mov     x8, #0x0040
    movk    x8, #0x0011, lsl #16           // x8 = 0x110040 (FALCON_MAILBOX0)
    str     w20, [x19, x8]                 // MAILBOX0 <- fmc_params_pa[31:0]
    add     x8, x8, #4                     // x8 = 0x110044 (FALCON_MAILBOX1)
    lsr     x7, x20, #32
    str     w7, [x19, x8]                  // MAILBOX1 <- fmc_params_pa[63:32]

    // 2. BCR_FMCCODE_LO/HI
    mov     x8, #0x1678
    movk    x8, #0x0011, lsl #16           // x8 = 0x111678
    str     w25, [x19, x8]                 // BCR_FMCCODE_LO
    lsr     x7, x25, #32
    add     x8, x8, #4                     // x8 = 0x11167C
    str     w7, [x19, x8]                  // BCR_FMCCODE_HI

    // 3. BCR_FMCDATA_LO/HI
    mov     x8, #0x1680
    movk    x8, #0x0011, lsl #16           // x8 = 0x111680
    str     w26, [x19, x8]                 // BCR_FMCDATA_LO
    lsr     x7, x26, #32
    add     x8, x8, #4                     // x8 = 0x111684
    str     w7, [x19, x8]                  // BCR_FMCDATA_HI

    // 4. BCR_PKCPARAM_LO/HI
    mov     x8, #0x1670
    movk    x8, #0x0011, lsl #16           // x8 = 0x111670
    str     w27, [x19, x8]                 // BCR_PKCPARAM_LO
    lsr     x7, x27, #32
    add     x8, x8, #4                     // x8 = 0x111674
    str     w7, [x19, x8]                  // BCR_PKCPARAM_HI

    // Barrier: ensure all BCR address writes committed before lock
    dsb     st

    // 5. BCR_DMACFG = LOCK | COHERENT (0x80000001)
    mov     w7, #0x0001
    movk    w7, #0x8000, lsl #16           // w7 = 0x80000001
    mov     x8, #0x166C
    movk    x8, #0x0011, lsl #16           // x8 = 0x11166C
    str     w7, [x19, x8]                  // BCR_DMACFG <- LOCK|COHERENT
    dsb     st

    // ---- Read back EVERY register and print ----

    // MAILBOX0 readback
    mov     x8, #0x0040
    movk    x8, #0x0011, lsl #16           // 0x110040
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_mbox0
    add     x1, x1, :lo12:msg_rb_mbox0
    mov     x2, #msg_rb_mbox0_len
    mov     w0, w28
    bl      print_labeled_hex32

    // MAILBOX1 readback
    mov     x8, #0x0044
    movk    x8, #0x0011, lsl #16           // 0x110044
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_mbox1
    add     x1, x1, :lo12:msg_rb_mbox1
    mov     x2, #msg_rb_mbox1_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_FMCCODE_LO readback
    mov     x8, #0x1678
    movk    x8, #0x0011, lsl #16           // 0x111678
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_fmccode_lo
    add     x1, x1, :lo12:msg_rb_fmccode_lo
    mov     x2, #msg_rb_fmccode_lo_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_FMCCODE_HI readback
    mov     x8, #0x167C
    movk    x8, #0x0011, lsl #16           // 0x11167C
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_fmccode_hi
    add     x1, x1, :lo12:msg_rb_fmccode_hi
    mov     x2, #msg_rb_fmccode_hi_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_FMCDATA_LO readback
    mov     x8, #0x1680
    movk    x8, #0x0011, lsl #16           // 0x111680
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_fmcdata_lo
    add     x1, x1, :lo12:msg_rb_fmcdata_lo
    mov     x2, #msg_rb_fmcdata_lo_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_FMCDATA_HI readback
    mov     x8, #0x1684
    movk    x8, #0x0011, lsl #16           // 0x111684
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_fmcdata_hi
    add     x1, x1, :lo12:msg_rb_fmcdata_hi
    mov     x2, #msg_rb_fmcdata_hi_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_PKCPARAM_LO readback
    mov     x8, #0x1670
    movk    x8, #0x0011, lsl #16           // 0x111670
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_pkcparam_lo
    add     x1, x1, :lo12:msg_rb_pkcparam_lo
    mov     x2, #msg_rb_pkcparam_lo_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_PKCPARAM_HI readback
    mov     x8, #0x1674
    movk    x8, #0x0011, lsl #16           // 0x111674
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_pkcparam_hi
    add     x1, x1, :lo12:msg_rb_pkcparam_hi
    mov     x2, #msg_rb_pkcparam_hi_len
    mov     w0, w28
    bl      print_labeled_hex32

    // BCR_DMACFG readback
    mov     x8, #0x166C
    movk    x8, #0x0011, lsl #16           // 0x11166C
    ldr     w28, [x19, x8]
    adrp    x1, msg_rb_dmacfg
    add     x1, x1, :lo12:msg_rb_dmacfg
    mov     x2, #msg_rb_dmacfg_len
    mov     w0, w28
    bl      print_labeled_hex32

    // Check DMACFG bit 31 (LOCK)
    tbnz    w28, #31, .lock_ok
    // LOCK failed
    adrp    x1, msg_lock_fail
    add     x1, x1, :lo12:msg_lock_fail
    mov     x2, #msg_lock_fail_len
    bl      probe_print
    b       .lock_checked
.lock_ok:
    adrp    x1, msg_lock_ok
    add     x1, x1, :lo12:msg_lock_ok
    mov     x2, #msg_lock_ok_len
    bl      probe_print
.lock_checked:

    // ---- Final banners ----
    adrp    x1, msg_no_cpuctl
    add     x1, x1, :lo12:msg_no_cpuctl
    mov     x2, #msg_no_cpuctl_len
    bl      probe_print

    adrp    x1, msg_complete
    add     x1, x1, :lo12:msg_complete
    mov     x2, #msg_complete_len
    bl      probe_print

    // ---- Exit 0 ----
    ldp     x29, x30, [sp], #16
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Failure paths
// ============================================================
.fail_step1:
    adrp    x1, msg_step1_fail
    add     x1, x1, :lo12:msg_step1_fail
    mov     x2, #msg_step1_fail_len
    bl      probe_print
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.fail_step2:
    adrp    x1, msg_step2_fail
    add     x1, x1, :lo12:msg_step2_fail
    mov     x2, #msg_step2_fail_len
    bl      probe_print
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

.fail_step3:
    adrp    x1, msg_step3_fail
    add     x1, x1, :lo12:msg_step3_fail
    mov     x2, #msg_step3_fail_len
    bl      probe_print
    mov     x0, #3
    mov     x8, #SYS_EXIT
    svc     #0

.fail_step4:
    adrp    x1, msg_step4_fail
    add     x1, x1, :lo12:msg_step4_fail
    mov     x2, #msg_step4_fail_len
    bl      probe_print
    mov     x0, #4
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// probe_print -- write x2 bytes from x1 to stderr (fd 2)
// Clobbers: x0, x8
// ============================================================
.align 4
probe_print:
    mov     x0, #2                          // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ============================================================
// print_labeled_hex64 -- print "LABEL0xHHHHHHHHHHHHHHHH\n"
//   x0 = 64-bit value to print
//   x1 = label address
//   x2 = label length
// Clobbers: x0-x8, x28 (uses x28 to save value across calls)
// ============================================================
.align 4
print_labeled_hex64:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]            // save callee-saved we use globally
    mov     x9, x0                          // save value in scratch reg

    // Print label
    mov     x0, #2                          // stderr
    mov     x8, #SYS_WRITE
    svc     #0

    // Print upper 32 bits
    lsr     w0, w9, #0                      // need full 64-bit value
    lsr     x10, x9, #32
    mov     w0, w10                         // upper 32 bits
    bl      print_hex32

    // Print lower 32 bits
    mov     w0, w9                          // lower 32 bits
    bl      print_hex32

    // Print newline
    mov     x0, #2
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ============================================================
// print_labeled_hex32 -- print "LABEL0xHHHHHHHH\n"
//   w0 = 32-bit value to print
//   x1 = label address
//   x2 = label length
// ============================================================
.align 4
print_labeled_hex32:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     w28, [sp, #16]                  // save w28
    mov     w9, w0                          // save value

    // Print label
    mov     x0, #2                          // stderr
    mov     x8, #SYS_WRITE
    svc     #0

    // Print hex value
    mov     w0, w9
    bl      print_hex32

    // Print newline
    mov     x0, #2
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0

    ldr     w28, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ============================================================
// print_hex32 -- print 8 hex digits to stderr
//   w0 = value to print
// Pattern adapted from probe_bar0.s
// ============================================================
.align 4
print_hex32:
    sub     sp, sp, #16
    mov     w3, #28                         // shift start (MSB nibble)
    add     x4, sp, #0                      // buffer pointer
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
    // Write 8 hex chars from stack
    mov     x0, #2                          // stderr (not stdout)
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret
