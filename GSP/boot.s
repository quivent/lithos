// boot.s -- GSP cold-boot entry point
//
// Calls the 9 GSP boot stages in sequence:
//   1. bar_map_init        -- mmap BAR0 + BAR4 from PCI sysfs
//   2. pmc_check           -- verify Hopper architecture via PMC_BOOT_0
//   3. falcon_reset        -- 4-step Falcon engine reset
//   4. hbm_alloc_init      -- init BAR4 bump allocator (64MB reserved header)
//   5. gsp_fw_load         -- parse firmware ELF, copy .fwimage to BAR4
//   6. gsp_bcr_start       -- program BCR, start RISC-V FMC
//   7. FSP communication   -- TODO -- NOT IMPLEMENTED (placeholder notice)
//   8. gsp_poll_lockdown   -- wait for PRIV_LOCKDOWN release
//   9. gsp_rpc_alloc_channel -- first RPC round-trip, allocate GPFIFO channel
//
// NOTE: NVIDIA's reference flow performs FSP handshake BEFORE BCR start.
// We currently skip it (works on GH200 where SBIOS pre-seeds the state),
// so the [7/9] slot is just a TODO notice printed after [6/9].
//
// No libc. Raw Linux syscalls only (SYS_WRITE for progress, SYS_EXIT on end).
// Matches style of src/launcher.s.
//
// Build:
//   as -o boot.o boot.s
//   as -o bar_map.o bar_map.s
//   as -o pmc_check.o pmc_check.s
//   as -o falcon_reset.o falcon_reset.s
//   as -o hbm_alloc.o hbm_alloc.s
//   as -o fw_load.o fw_load.s
//   as -o bcr_start.o bcr_start.s
//   as -o poll_lockdown.o poll_lockdown.s
//   as -o rpc_channel.o rpc_channel.s
//   ld -o gsp-boot *.o -static
//
// Exit codes:
//   0 = success (all 9 stages passed)
//   1 = bar_map_init failed
//   2 = pmc_check failed (not Hopper)
//   3 = falcon_reset failed
//   4 = gsp_poll_lockdown failed (lockdown timeout or FMC error)
//   5 = gsp_rpc_alloc_channel failed
//   6 = hbm_alloc_init failed
//   7 = gsp_bcr_start failed
//   8 = gsp_fw_load failed (if it returns error instead of SYS_EXIT)
// Note: fw_load.s also exits directly with codes 1-12 on internal errors.

.equ SYS_WRITE,          64
.equ SYS_EXIT,           93
.equ SYS_CLOCK_GETTIME,  113
.equ CLOCK_MONOTONIC,    1

// ---- Boot watchdog ----
// If the full 9-step boot exceeds this wall-clock budget, abort.
// Individual step timeouts (falcon 5s, lockdown 30s) sum to ~40s;
// the 120s outer envelope catches unexpected hangs (e.g., MMIO
// returning 0xFFFFFFFF from a powered-down BAR).
.equ BOOT_WATCHDOG_SECS, 120

// ---- BAR4 reservation for GSP control structures ----
// The first 64 MB of BAR4 is reserved for things the bump allocator
// does not manage (WPR metadata, RM subregions, etc.).  The bump
// allocator starts at offset 0x04000000.
.equ GSP_RESERVED_BYTES, 0x04000000           // 64 MB

// ---- RPC message queue offsets within BAR4 ----
// These are placeholder offsets for the command/status queues.
// The real values are negotiated via FSP/booter (step 7, TODO).
// For now we pick stable offsets high in the reserved region.
.equ CMD_QUEUE_OFFSET,   0x03000000           // 48 MB into BAR4
.equ STAT_QUEUE_OFFSET,  0x03400000           // 52 MB into BAR4

// ---- GPFIFO sizing passed to gsp_rpc_alloc_channel ----
.equ GPFIFO_ENTRIES,     1024

// ============================================================
// External symbols
// ============================================================
.extern bar_map_init
.extern pmc_check
.extern falcon_reset
.extern hbm_alloc_init
.extern gsp_fw_load
.extern gsp_bcr_start
.extern gsp_poll_lockdown
.extern gsp_rpc_alloc_channel

.extern bar0_base
.extern bar4_base
.extern bar4_phys

.extern fw_bar4_phys
.extern fw_code_offset
.extern fw_data_offset
.extern fw_manifest_offset

// ============================================================
// Data section -- progress messages
// ============================================================
.data
.align 3

msg_banner:          .asciz "gsp: ================== Lithos GSP cold-boot ==================\n"
msg_banner_len       = . - msg_banner - 1

msg_step1_begin:     .asciz "gsp: [1/9] bar_map_init -- mapping BAR0 and BAR4...\n"
msg_step1_begin_len  = . - msg_step1_begin - 1
msg_step1_done:      .asciz "gsp: [1/9] bar_map_init -- OK\n"
msg_step1_done_len   = . - msg_step1_done - 1
msg_step1_fail:      .asciz "gsp: [1/9] bar_map_init -- FAILED, aborting\n"
msg_step1_fail_len   = . - msg_step1_fail - 1

msg_step2_begin:     .asciz "gsp: [2/9] pmc_check -- verifying Hopper...\n"
msg_step2_begin_len  = . - msg_step2_begin - 1
msg_step2_done:      .asciz "gsp: [2/9] pmc_check -- OK\n"
msg_step2_done_len   = . - msg_step2_done - 1
msg_step2_fail:      .asciz "gsp: [2/9] pmc_check -- FAILED, aborting\n"
msg_step2_fail_len   = . - msg_step2_fail - 1

msg_step3_begin:     .asciz "gsp: [3/9] falcon_reset -- resetting GSP Falcon engine...\n"
msg_step3_begin_len  = . - msg_step3_begin - 1
msg_step3_done:      .asciz "gsp: [3/9] falcon_reset -- OK\n"
msg_step3_done_len   = . - msg_step3_done - 1
msg_step3_fail:      .asciz "gsp: [3/9] falcon_reset -- FAILED, aborting\n"
msg_step3_fail_len   = . - msg_step3_fail - 1

msg_step4_begin:     .asciz "gsp: [4/9] hbm_alloc_init -- init BAR4 bump allocator (64MB reserved)...\n"
msg_step4_begin_len  = . - msg_step4_begin - 1
msg_step4_done:      .asciz "gsp: [4/9] hbm_alloc_init -- OK\n"
msg_step4_done_len   = . - msg_step4_done - 1

msg_step5_begin:     .asciz "gsp: [5/9] gsp_fw_load -- loading firmware ELF into BAR4...\n"
msg_step5_begin_len  = . - msg_step5_begin - 1
msg_step5_done:      .asciz "gsp: [5/9] gsp_fw_load -- OK\n"
msg_step5_done_len   = . - msg_step5_done - 1
msg_step5_fail:      .asciz "gsp: [5/9] gsp_fw_load -- FAILED, aborting\n"
msg_step5_fail_len   = . - msg_step5_fail - 1

// Step 7 placeholder -- FSP communication (TODO -- NOT IMPLEMENTED)
msg_step_fsp_todo:   .asciz "gsp: [7/9] FSP communication -- TODO -- NOT IMPLEMENTED (see docs/gsp-native.md sec 7)\n"
msg_step_fsp_todo_len = . - msg_step_fsp_todo - 1

msg_step4_fail:      .asciz "gsp: [4/9] hbm_alloc_init -- FAILED, aborting\n"
msg_step4_fail_len   = . - msg_step4_fail - 1

msg_step6_begin:     .asciz "gsp: [6/9] gsp_bcr_start -- programming BCR, starting RISC-V FMC...\n"
msg_step6_begin_len  = . - msg_step6_begin - 1
msg_step6_done:      .asciz "gsp: [6/9] gsp_bcr_start -- OK (CPU started)\n"
msg_step6_done_len   = . - msg_step6_done - 1
msg_step6_fail:      .asciz "gsp: [6/9] gsp_bcr_start -- FAILED, aborting\n"
msg_step6_fail_len   = . - msg_step6_fail - 1

msg_step7_begin:     .asciz "gsp: [8/9] gsp_poll_lockdown -- waiting for PRIV_LOCKDOWN release...\n"
msg_step7_begin_len  = . - msg_step7_begin - 1
msg_step7_done:      .asciz "gsp: [8/9] gsp_poll_lockdown -- OK (lockdown released)\n"
msg_step7_done_len   = . - msg_step7_done - 1
msg_step7_fail:      .asciz "gsp: [8/9] gsp_poll_lockdown -- FAILED (timeout or FMC error), aborting\n"
msg_step7_fail_len   = . - msg_step7_fail - 1

msg_step8_begin:     .asciz "gsp: [9/9] gsp_rpc_alloc_channel -- first RPC round-trip...\n"
msg_step8_begin_len  = . - msg_step8_begin - 1
msg_step8_done:      .asciz "gsp: [9/9] gsp_rpc_alloc_channel -- OK (channel allocated)\n"
msg_step8_done_len   = . - msg_step8_done - 1
msg_step8_fail:      .asciz "gsp: [9/9] gsp_rpc_alloc_channel -- FAILED, aborting\n"
msg_step8_fail_len   = . - msg_step8_fail - 1

msg_success:         .asciz "gsp: ================== GSP boot complete ==================\n"
msg_success_len      = . - msg_success - 1

msg_watchdog:        .asciz "gsp: FATAL: boot watchdog expired (>120s) -- aborting\n"
msg_watchdog_len     = . - msg_watchdog - 1

// Bump pointer variable for RPC channel allocator (separate from the
// main hbm_alloc bump -- RPC module owns its own u64 cursor).
.align 3
rpc_bump:   .quad 0

// Boot start time (tv_sec from CLOCK_MONOTONIC, captured at _start).
.align 3
boot_start_sec:  .quad 0

// ============================================================
// Text section
// ============================================================
.text
.align 4

.global _start
_start:
    // ---- Capture boot start time for watchdog ----
    sub     sp, sp, #16                       // allocate timespec on stack
    mov     x0, #CLOCK_MONOTONIC
    mov     x1, sp
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    ldr     x9, [sp]                          // x9 = boot start seconds
    adrp    x10, boot_start_sec
    add     x10, x10, :lo12:boot_start_sec
    str     x9, [x10]
    add     sp, sp, #16                       // free timespec

    // ---- Banner ----
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    bl      boot_print

    // =============================================================
    // Step 1: bar_map_init
    // =============================================================
    adrp    x1, msg_step1_begin
    add     x1, x1, :lo12:msg_step1_begin
    mov     x2, #msg_step1_begin_len
    bl      boot_print

    bl      bar_map_init
    cmp     x0, #0
    b.lt    .boot_fail_step1

    adrp    x1, msg_step1_done
    add     x1, x1, :lo12:msg_step1_done
    mov     x2, #msg_step1_done_len
    bl      boot_print

    // =============================================================
    // Step 2: pmc_check
    // =============================================================
    adrp    x1, msg_step2_begin
    add     x1, x1, :lo12:msg_step2_begin
    mov     x2, #msg_step2_begin_len
    bl      boot_print

    bl      pmc_check
    cmp     x0, #0
    b.lt    .boot_fail_step2

    adrp    x1, msg_step2_done
    add     x1, x1, :lo12:msg_step2_done
    mov     x2, #msg_step2_done_len
    bl      boot_print

    // =============================================================
    // Step 3: falcon_reset
    // =============================================================
    adrp    x1, msg_step3_begin
    add     x1, x1, :lo12:msg_step3_begin
    mov     x2, #msg_step3_begin_len
    bl      boot_print

    bl      falcon_reset
    cmp     x0, #0
    b.lt    .boot_fail_step3

    adrp    x1, msg_step3_done
    add     x1, x1, :lo12:msg_step3_done
    mov     x2, #msg_step3_done_len
    bl      boot_print

    // =============================================================
    // Step 4: hbm_alloc_init -- bump allocator for BAR4
    //   x0 = BAR4 CPU virtual address (from bar4_base)
    //   x1 = BAR4 physical base       (from bar4_phys, == GPU VA on GH200)
    //   x2 = initial bump offset      (skip 64MB reserved for GSP metadata)
    // =============================================================
    adrp    x1, msg_step4_begin
    add     x1, x1, :lo12:msg_step4_begin
    mov     x2, #msg_step4_begin_len
    bl      boot_print

    adrp    x0, bar4_base
    add     x0, x0, :lo12:bar4_base
    ldr     x0, [x0]                        // x0 = bar4 CPU va
    adrp    x1, bar4_phys
    add     x1, x1, :lo12:bar4_phys
    ldr     x1, [x1]                        // x1 = bar4 phys
    mov     x2, #GSP_RESERVED_BYTES         // x2 = 64 MB initial offset
    bl      hbm_alloc_init
    cmp     x0, #0
    b.lt    .boot_fail_step4_alloc

    adrp    x1, msg_step4_done
    add     x1, x1, :lo12:msg_step4_done
    mov     x2, #msg_step4_done_len
    bl      boot_print

    // =============================================================
    // Step 5: gsp_fw_load -- parse ELF, copy .fwimage into BAR4
    //   Returns: x0 = BAR4 phys addr of firmware image
    //            x1 = total .fwimage size
    //   Side effects: fw_bar4_cpu, fw_bar4_phys, fw_code_offset,
    //                 fw_data_offset, fw_manifest_offset populated.
    // =============================================================
    adrp    x1, msg_step5_begin
    add     x1, x1, :lo12:msg_step5_begin
    mov     x2, #msg_step5_begin_len
    bl      boot_print

    bl      gsp_fw_load
    // gsp_fw_load currently SYS_EXITs on failure internally, but add a
    // defensive check: if it ever changes to return errors, catch them.
    cmp     x0, #0
    b.le    .boot_fail_step5

    adrp    x1, msg_step5_done
    add     x1, x1, :lo12:msg_step5_done
    mov     x2, #msg_step5_done_len
    bl      boot_print

    // =============================================================
    // Step 6: gsp_bcr_start
    //   x0 = bar0 base
    //   x1 = fmc_params_pa  -- TODO: real boot-args PA goes here once
    //        step 7 produces it.  For now pass 0 so the GSP reads a
    //        well-defined sentinel.
    //   x2 = fmc_image_pa (= fw_bar4_phys)
    //   x3 = code_offset  (= fw_code_offset)
    //   x4 = data_offset  (= fw_data_offset)
    //   x5 = manifest_offset (= fw_manifest_offset)
    // =============================================================
    adrp    x1, msg_step6_begin
    add     x1, x1, :lo12:msg_step6_begin
    mov     x2, #msg_step6_begin_len
    bl      boot_print

    // x0 = bar0 base
    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]

    // x1 = fmc_params_pa (placeholder: 0 until step 7 implemented)
    mov     x1, #0

    // x2 = fmc_image_pa
    adrp    x6, fw_bar4_phys
    add     x6, x6, :lo12:fw_bar4_phys
    ldr     x2, [x6]

    // x3 = code_offset
    adrp    x6, fw_code_offset
    add     x6, x6, :lo12:fw_code_offset
    ldr     x3, [x6]

    // x4 = data_offset
    adrp    x6, fw_data_offset
    add     x6, x6, :lo12:fw_data_offset
    ldr     x4, [x6]

    // x5 = manifest_offset
    adrp    x6, fw_manifest_offset
    add     x6, x6, :lo12:fw_manifest_offset
    ldr     x5, [x6]

    bl      gsp_bcr_start
    cmp     x0, #0
    b.lt    .boot_fail_step6

    adrp    x1, msg_step6_done
    add     x1, x1, :lo12:msg_step6_done
    mov     x2, #msg_step6_done_len
    bl      boot_print

    // Step 7: FSP -- not yet wired
    adrp    x1, msg_step_fsp_todo
    add     x1, x1, :lo12:msg_step_fsp_todo
    mov     x2, #msg_step_fsp_todo_len
    bl      boot_print

    // =============================================================
    // Step 8: gsp_poll_lockdown
    //   x0 = bar0 base
    //   x1 = fmc_params_pa (must match what BCR step passed)
    //   Returns 0 on success, -1 on timeout, -2 on FMC error.
    // =============================================================
    adrp    x1, msg_step7_begin
    add     x1, x1, :lo12:msg_step7_begin
    mov     x2, #msg_step7_begin_len
    bl      boot_print

    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    mov     x1, #0                            // fmc_params_pa placeholder
    bl      gsp_poll_lockdown
    cmp     x0, #0
    b.lt    .boot_fail_step8

    adrp    x1, msg_step7_done
    add     x1, x1, :lo12:msg_step7_done
    mov     x2, #msg_step7_done_len
    bl      boot_print

    // =============================================================
    // Step 9: gsp_rpc_alloc_channel
    //   x0 = bar0 base
    //   x1 = bar4 base (CPU va)
    //   x2 = cmd_queue_offset
    //   x3 = stat_queue_offset
    //   x4 = gpfifo_gpu_va (0 -> bump-allocate)
    //   x5 = gpfifo_entries
    //   x6 = pointer to u64 bump variable
    // =============================================================
    adrp    x1, msg_step8_begin
    add     x1, x1, :lo12:msg_step8_begin
    mov     x2, #msg_step8_begin_len
    bl      boot_print

    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]

    adrp    x1, bar4_base
    add     x1, x1, :lo12:bar4_base
    ldr     x1, [x1]

    movz    x2, #(CMD_QUEUE_OFFSET & 0xFFFF)
    movk    x2, #((CMD_QUEUE_OFFSET >> 16) & 0xFFFF), lsl #16
    movz    x3, #(STAT_QUEUE_OFFSET & 0xFFFF)
    movk    x3, #((STAT_QUEUE_OFFSET >> 16) & 0xFFFF), lsl #16
    mov     x4, #0                            // gpfifo_gpu_va = 0 -> alloc
    mov     x5, #GPFIFO_ENTRIES
    adrp    x6, rpc_bump
    add     x6, x6, :lo12:rpc_bump

    bl      gsp_rpc_alloc_channel
    cmp     x0, #0
    b.lt    .boot_fail_step9

    adrp    x1, msg_step8_done
    add     x1, x1, :lo12:msg_step8_done
    mov     x2, #msg_step8_done_len
    bl      boot_print

    // =============================================================
    // Success -- exit 0
    // =============================================================
    adrp    x1, msg_success
    add     x1, x1, :lo12:msg_success
    mov     x2, #msg_success_len
    bl      boot_print

    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Failure paths
// ============================================================
.boot_fail_step1:
    adrp    x1, msg_step1_fail
    add     x1, x1, :lo12:msg_step1_fail
    mov     x2, #msg_step1_fail_len
    bl      boot_print
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step2:
    adrp    x1, msg_step2_fail
    add     x1, x1, :lo12:msg_step2_fail
    mov     x2, #msg_step2_fail_len
    bl      boot_print
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step3:
    adrp    x1, msg_step3_fail
    add     x1, x1, :lo12:msg_step3_fail
    mov     x2, #msg_step3_fail_len
    bl      boot_print
    mov     x0, #3
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step5:
    adrp    x1, msg_step5_fail
    add     x1, x1, :lo12:msg_step5_fail
    mov     x2, #msg_step5_fail_len
    bl      boot_print
    mov     x0, #8                            // exit code for fw_load failure
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step6:
    adrp    x1, msg_step6_fail
    add     x1, x1, :lo12:msg_step6_fail
    mov     x2, #msg_step6_fail_len
    bl      boot_print
    mov     x0, #7
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step8:
    adrp    x1, msg_step7_fail
    add     x1, x1, :lo12:msg_step7_fail
    mov     x2, #msg_step7_fail_len
    bl      boot_print
    mov     x0, #4
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step9:
    adrp    x1, msg_step8_fail
    add     x1, x1, :lo12:msg_step8_fail
    mov     x2, #msg_step8_fail_len
    bl      boot_print
    mov     x0, #5
    mov     x8, #SYS_EXIT
    svc     #0

.boot_fail_step4_alloc:
    adrp    x1, msg_step4_fail
    add     x1, x1, :lo12:msg_step4_fail
    mov     x2, #msg_step4_fail_len
    bl      boot_print
    mov     x0, #6
    mov     x8, #SYS_EXIT
    svc     #0

// ------------------------------------------------------------
// boot_watchdog_fail -- boot exceeded BOOT_WATCHDOG_SECS
// Prints fatal message and exits with code 99.
// ------------------------------------------------------------
.boot_watchdog_fail:
    adrp    x1, msg_watchdog
    add     x1, x1, :lo12:msg_watchdog
    mov     x2, #msg_watchdog_len
    bl      boot_print
    mov     x0, #99
    mov     x8, #SYS_EXIT
    svc     #0

// ------------------------------------------------------------
// boot_print -- write x2 bytes from x1 to stderr (fd 2)
// Clobbers: x0, x8
// ------------------------------------------------------------
.align 4
boot_print:
    mov     x0, #2                            // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// boot_check_watchdog -- check total elapsed boot time
// If > BOOT_WATCHDOG_SECS, jumps to .boot_watchdog_fail (no return).
// Otherwise returns normally.
// Clobbers: x0, x1, x8, x9, x10
// Uses 16 bytes of stack for timespec.
// ------------------------------------------------------------
.align 4
boot_check_watchdog:
    sub     sp, sp, #16
    mov     x0, #CLOCK_MONOTONIC
    mov     x1, sp
    mov     x8, #SYS_CLOCK_GETTIME
    svc     #0
    ldr     x9, [sp]                          // current seconds
    add     sp, sp, #16
    adrp    x10, boot_start_sec
    add     x10, x10, :lo12:boot_start_sec
    ldr     x10, [x10]                        // start seconds
    sub     x9, x9, x10                       // elapsed
    cmp     x9, #BOOT_WATCHDOG_SECS
    b.ge    .boot_watchdog_fail
    ret
