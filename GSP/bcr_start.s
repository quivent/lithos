// GSP/bcr_start.s -- Step 6: Program RISC-V BCR registers and start FMC
//
// Translates gsp_bootstrap_fmc() from kernel/lithos_gsp.c to raw ARM64.
// No libc. All BAR0 offsets from dev_gsp.h / dev_riscv_pri.h.
//
// Source: _kgspBootstrapGspFmc_GH100() -- kernel_gsp_gh100.c:716
//
// Inputs (arguments):
//   x0 = bar0 base (mmap'd BAR0 virtual address)
//   x1 = fmc_params_pa (physical address of GSP_FMC_BOOT_PARAMS in BAR4)
//   x2 = fmc_image_pa  (physical address of FMC binary image in BAR4)
//   x3 = code_offset   (RM_RISCV_UCODE_DESC.monitorCodeOffset)
//   x4 = data_offset   (RM_RISCV_UCODE_DESC.monitorDataOffset)
//   x5 = manifest_offset (RM_RISCV_UCODE_DESC.manifestOffset)
//
// Returns: x0 = 0 on success, -1 null params, -2 lock verify fail
//
// Register allocation:
//   x0  = bar0 base (preserved)
//   x6  = scratch for computed shifted address
//   x7  = scratch for register value
//   x8  = scratch for BAR0 offset (too large for str immediate)
//
// Build:
//   as -o bcr_start.o bcr_start.s

// FMC addresses are >> 3 (8-byte alignment) before writing to BCR regs
.equ RISCV_BR_ADDR_SHIFT,   3

.text
.globl gsp_bcr_start
.type  gsp_bcr_start, %function
.balign 4

gsp_bcr_start:
    // No stack frame needed -- leaf function, no callee-saved regs used.
    // All work done in x0-x9.
    //
    // BAR0 offsets exceed the 12-bit unsigned immediate range for str
    // (max 16380 for 32-bit str), so we load each offset into x8 and
    // use str w_, [x0, x8].

    // ----------------------------------------------------------------
    // 0. Null-check fmc_params_pa
    // ----------------------------------------------------------------
    cbz     x1, .bcr_bad_params

    // ----------------------------------------------------------------
    // 1. Write fmc_params_pa to MAILBOX0/1
    //    GSP-FMC reads its boot args PA from these two 32-bit registers.
    //    Source: kgspSetupGspFmcArgs_GH100() -> FALCON_MAILBOX0/1
    // ----------------------------------------------------------------
    mov x8, #0x0040
    movk x8, #0x0011, lsl #16           // x8 = 0x110040 (FALCON_MAILBOX0)
    str w1, [x0, x8]                    // BAR0+0x110040 <- fmc_params_pa[31:0]
    add x8, x8, #4                      // x8 = 0x110044 (FALCON_MAILBOX1)
    lsr x7, x1, #32
    str w7, [x0, x8]                    // BAR0+0x110044 <- fmc_params_pa[63:32]

    // ----------------------------------------------------------------
    // 2. Program BCR_FMCCODE_LO/HI with (fmc_image_pa + code_offset) >> 3
    //    FMC code segment address, 8-byte aligned and shifted.
    //    Source: _kgspBootstrapGspFmc_GH100() BCR code addr programming
    // ----------------------------------------------------------------
    add x6, x2, x3                      // x6 = fmc_image_pa + code_offset
    lsr x6, x6, #RISCV_BR_ADDR_SHIFT    // x6 >>= 3 (8-byte alignment)
    mov x8, #0x1678
    movk x8, #0x0011, lsl #16           // x8 = 0x111678 (BCR_FMCCODE_LO)
    str w6, [x0, x8]                    // BAR0+0x111678 <- fmc_code[31:0]
    lsr x7, x6, #32
    add x8, x8, #4                      // x8 = 0x11167C (BCR_FMCCODE_HI)
    str w7, [x0, x8]                    // BAR0+0x11167C <- fmc_code[63:32]

    // ----------------------------------------------------------------
    // 3. Program BCR_FMCDATA_LO/HI with (fmc_image_pa + data_offset) >> 3
    //    FMC data segment address.
    // ----------------------------------------------------------------
    add x6, x2, x4                      // x6 = fmc_image_pa + data_offset
    lsr x6, x6, #RISCV_BR_ADDR_SHIFT    // x6 >>= 3
    mov x8, #0x1680
    movk x8, #0x0011, lsl #16           // x8 = 0x111680 (BCR_FMCDATA_LO)
    str w6, [x0, x8]                    // BAR0+0x111680 <- fmc_data[31:0]
    lsr x7, x6, #32
    add x8, x8, #4                      // x8 = 0x111684 (BCR_FMCDATA_HI)
    str w7, [x0, x8]                    // BAR0+0x111684 <- fmc_data[63:32]

    // ----------------------------------------------------------------
    // 4. Program BCR_PKCPARAM_LO/HI with (fmc_image_pa + manifest_offset) >> 3
    //    FMC manifest (PKC signature parameters).
    // ----------------------------------------------------------------
    add x6, x2, x5                      // x6 = fmc_image_pa + manifest_offset
    lsr x6, x6, #RISCV_BR_ADDR_SHIFT    // x6 >>= 3
    mov x8, #0x1670
    movk x8, #0x0011, lsl #16           // x8 = 0x111670 (BCR_PKCPARAM_LO)
    str w6, [x0, x8]                    // BAR0+0x111670 <- manifest[31:0]
    lsr x7, x6, #32
    add x8, x8, #4                      // x8 = 0x111674 (BCR_PKCPARAM_HI)
    str w7, [x0, x8]                    // BAR0+0x111674 <- manifest[63:32]

    // Barrier: ensure all 6 BCR address writes are committed before lock
    dsb     st

    // ----------------------------------------------------------------
    // 5. Lock BCR and set DMA target = coherent sysmem
    //    bit 31 = LOCK (prevent further BCR modification)
    //    bit  0 = TARGET = COHERENT_SYSMEM
    //    Source: NV_PRISCV_RISCV_BCR_DMACFG from dev_riscv_pri.h
    // ----------------------------------------------------------------
    mov w7, #0x0001
    movk w7, #0x8000, lsl #16           // w7 = 0x80000001
    mov x8, #0x166C
    movk x8, #0x0011, lsl #16           // x8 = 0x11166C (BCR_DMACFG)
    str w7, [x0, x8]                    // BAR0+0x11166C <- LOCK|COHERENT
    dsb     st                           // drain stores before readback
    ldr w9, [x0, x8]                    // read back DMACFG
    tbnz    w9, #31, .bcr_locked        // bit 31 set = locked, good
    // lock failed
    mov x0, #-2
    ret
.bcr_locked:

    // ----------------------------------------------------------------
    // 6. Start RISC-V CPU
    //    Writing STARTCPU=1 begins FMC execution on the GSP RISC-V core.
    //    Source: NV_PRISCV_RISCV_CPUCTL from dev_riscv_pri.h
    // ----------------------------------------------------------------
    mov w7, #1
    mov x8, #0x1388
    movk x8, #0x0011, lsl #16           // x8 = 0x111388 (CPUCTL)
    str w7, [x0, x8]                    // BAR0+0x111388 <- 1 (start CPU)

    // Memory barrier: ensure all stores to BAR0 (device memory) are
    // ordered and visible to the GPU before we return.
    dsb sy

    mov x0, #0                          // success
    ret

.bcr_bad_params:
    mov x0, #-1                         // null fmc_params_pa
    ret

.size gsp_bcr_start, . - gsp_bcr_start
