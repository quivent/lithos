// GSP/gsp_common.s -- Shared constants and macros for GSP boot assembly
//
// Source of truth for .equ values used by more than one file in this
// directory.  The per-step files (bar_map.s, pmc_check.s, falcon_reset.s,
// hbm_alloc.s, fw_load.s, bcr_start.s, poll_lockdown.s, rpc_channel.s)
// currently each redefine their own copy; this file collects them so a
// future refactor can remove the duplicates.
//
// This file contributes NO code and NO data symbols.  It is .equ-only,
// plus preprocessor-style comments.  Assembling it produces an empty
// object; including it (via `.include "gsp_common.s"`) in each step
// file would let us drop the local .equ blocks without name collisions.
//
// Why this matters:
//   - Duplicate .equ across translation units is benign at link time
//     (each .s becomes its own .o and .equ values are local),
//     but if any two step files are ever assembled together as a
//     single translation unit (e.g. via .include), the redefinitions
//     would error out unless all copies agree exactly.
//   - Duplicate *data labels* (see below) would conflict at link
//     time with global symbols, and silently shadow each other
//     with local symbols.  Today the following data labels are
//     duplicated across files and should be unified here once the
//     step files are refactored:
//       * msg_bar0_null / msg_bar0_null_len
//           defined in  pmc_check.s  AND  falcon_reset.s
//           (local .asciz strings, different text contents -- both
//            labeled "msg_bar0_null" -- each .o has its own copy,
//            so there is no link-time collision *today* because
//            neither is .global, but the name reuse is a latent
//            hazard and confuses anyone searching for the symbol.)
//   - The "09:00" BDF ASCII fragment is duplicated inside bar_map.s
//     itself (three copies: pci_bdf_slot, pci_bdf_slot4,
//     pci_bdf_slot_r).  That is an intra-file concern, not shared,
//     and is left to bar_map.s.
//
// Assemble check:
//   as -o /dev/null GSP/gsp_common.s

// ============================================================
// 1. Linux aarch64 syscall numbers
//    (arch/arm64/include/uapi/asm/unistd.h, generic unistd.h)
// ============================================================

.equ SYS_READ,              63      // bar_map.s, fw_load.s
.equ SYS_WRITE,             64      // bar_map.s, pmc_check.s,
                                    // falcon_reset.s, fw_load.s
.equ SYS_OPENAT,            56      // bar_map.s, fw_load.s
.equ SYS_CLOSE,             57      // bar_map.s, fw_load.s
.equ SYS_LSEEK,             62      // fw_load.s
.equ SYS_MMAP,              222     // bar_map.s, fw_load.s
.equ SYS_EXIT,              93      // bar_map.s, fw_load.s
.equ SYS_CLOCK_GETTIME,     113     // poll_lockdown.s

// ============================================================
// 2. openat() / mmap() / lseek() flag values
//    (bits/fcntl-linux.h, bits/mman-linux.h)
//    Note: O_SYNC has an aarch64-specific encoding (0x101000),
//    not the generic 0x1000.  O_RDWR|O_SYNC == 0x101002.
// ============================================================

.equ AT_FDCWD,              -100    // bar_map.s, fw_load.s
.equ O_RDONLY,              0       // fw_load.s (implicit in bar_map.s)
.equ O_RDWR,                2       // bar_map.s
.equ O_SYNC,                0x101000 // bar_map.s (aarch64 encoding)

.equ PROT_READ,             1       // bar_map.s, fw_load.s
.equ PROT_WRITE,            2       // bar_map.s
.equ PROT_RW,               3       // bar_map.s (PROT_READ|PROT_WRITE)

.equ MAP_SHARED,            1       // bar_map.s
.equ MAP_PRIVATE,           2       // fw_load.s

.equ SEEK_SET,              0       // fw_load.s
.equ SEEK_END,              2       // fw_load.s

// ============================================================
// 3. BAR0 register offsets (Hopper / GH100)
//    Sources: dev_gsp.h, dev_riscv_pri.h, dev_falcon_v4.h,
//             gsp-native.md Section 3.
//    Several of these exceed the 12-bit str/ldr immediate range
//    and must be materialized into a register at use sites.
// ============================================================

// ---- PMC (Power Management Controller) ----
.equ PMC_BOOT_0,            0x000000    // pmc_check.s
.equ ARCH_HOPPER,           0xA        // pmc_check.s (bits[23:20])

// ---- Falcon engine control ----
.equ FALCON_ENGINE,         0x1103C0    // falcon_reset.s
                                        //   bit[0]     = RESET
                                        //   bits[10:8] = RESET_STATUS
.equ FALCON_STATUS_MASK,       0x700    // falcon_reset.s: bits[10:8]
.equ FALCON_STATUS_ASSERTED,   0x000
.equ FALCON_STATUS_DEASSERTED, 0x200

// ---- Falcon mailboxes / HWCFG2 ----
.equ FALCON_MAILBOX0,       0x110040    // bcr_start.s, poll_lockdown.s
.equ FALCON_MAILBOX1,       0x110044    // bcr_start.s
.equ FALCON_HWCFG2,         0x1100F4    // poll_lockdown.s
.equ PRIV_LOCKDOWN_BIT,     13          // poll_lockdown.s (bit in HWCFG2)

// ---- RISC-V Boot Control Registers (BCR) ----
// NV_PRISCV_RISCV_BCR_* -- dev_riscv_pri.h
.equ BCR_DMACFG,            0x11166C    // bcr_start.s (LOCK|TARGET)
.equ BCR_PKCPARAM_LO,       0x111670    // bcr_start.s (manifest addr lo)
.equ BCR_PKCPARAM_HI,       0x111674    // bcr_start.s (manifest addr hi)
.equ BCR_FMCCODE_LO,        0x111678    // bcr_start.s
.equ BCR_FMCCODE_HI,        0x11167C    // bcr_start.s
.equ BCR_FMCDATA_LO,        0x111680    // bcr_start.s
.equ BCR_FMCDATA_HI,        0x111684    // bcr_start.s
.equ RISCV_BR_ADDR_SHIFT,   3           // bcr_start.s (>>3 for 8B align)

// ---- RISC-V CPU control ----
.equ CPUCTL,                0x111388    // bcr_start.s (STARTCPU bit 0)

// ---- GSP message queue head/tail (QUEUE_HEAD(i) = base + i*8) ----
.equ GSP_QUEUE_HEAD_BASE,   0x110C00    // rpc_channel.s (idx 0)
.equ GSP_QUEUE_TAIL_BASE,   0x110C04    // rpc_channel.s (idx 0)

// ---- USERD (channel doorbell pages) ----
.equ USERD_BAR0_BASE,       0xFC0000    // rpc_channel.s
.equ USERD_STRIDE,          0x200       // rpc_channel.s (per-channel page)
.equ USERD_GPPUT_OFF,       0x08C       // rpc_channel.s

// ============================================================
// 4. HBM / BAR4 sizing (from bar_map.s, hbm_alloc.s)
// ============================================================

.equ BAR0_SIZE,             0x1000000       // 16 MB GPU MMIO
.equ BAR4_SIZE_LO,          0x10000000      // 256 MB initial window
.equ ALIGN_2MB,             0x200000        // hbm_alloc.s
.equ ALIGN_2MB_MASK,        0x1FFFFF

// ============================================================
// 5. Poll timeouts / clock
// ============================================================

.equ CLOCK_MONOTONIC,       1               // poll_lockdown.s
.equ TIMEOUT_SECS,          30              // poll_lockdown.s
.equ POLL_TIMEOUT,          100000000       // falcon_reset.s (~1-2 s)
.equ RPC_POLL_LIMIT,        10000000        // rpc_channel.s (~10 s)

// ============================================================
// 6. RPC / message-queue protocol constants (rpc_channel.s)
//    Sources: rpc_global_enums.h, rpc_headers.h,
//             g_rpc-structures.h, message_queue_cpu.c
// ============================================================

// Queue indices
.equ RPC_CMD_QUEUE_IDX,     0
.equ RPC_STAT_QUEUE_IDX,    1

// RPC function codes (NV_VGPU_MSG_FUNCTION_*)
.equ NV_VGPU_MSG_FUNCTION_ALLOC_ROOT,        2
.equ NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE,      3
.equ NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA, 6
.equ NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE,   19

// Object classes
.equ NV01_DEVICE_0,         0x0080
.equ NV20_SUBDEVICE_0,      0x20E0
.equ HOPPER_CHANNEL_GPFIFO, 0xC86F

// RPC object handles (fixed for single-client boot)
.equ HANDLE_CLIENT,         0x01000000
.equ HANDLE_DEVICE,         0x01000001
.equ HANDLE_SUBDEVICE,      0x01000002
.equ HANDLE_CHANNEL,        0x01000003

// RPC header / payload sizes
.equ RPC_HDR_SIZE,              0x20    // 32-byte RPC header
.equ RPC_ALLOC_ROOT_PAYLOAD,    4
.equ RPC_ALLOC_DEVICE_PAYLOAD,  12
.equ RPC_ALLOC_SUBDEVICE_PAYLOAD, 16
.equ RPC_ALLOC_CHANNEL_PAYLOAD, 40

// Msgq element wrapper (8 bytes of length + flags before RPC header)
.equ MQ_ELEM_HDR_SIZE,      8

// GPFIFO ring sizing
.equ GPFIFO_SIZE,           8192        // 8 KB = 1024 * 8B entries
.equ GPFIFO_ALIGN,          4096        // page-aligned

// ============================================================
// 7. ELF64 field offsets (fw_load.s)
// ============================================================

.equ E_SHOFF,               0x28
.equ E_SHENTSIZE,           0x3A
.equ E_SHNUM,               0x3C
.equ E_SHSTRNDX,            0x3E
.equ SH_NAME,               0x00
.equ SH_OFFSET,             0x18
.equ SH_SIZE,               0x20

// ============================================================
// 8. Watchdog / boot timeout constants
//
// The GSP boot sequence has no hardware watchdog.  If any step hangs
// (e.g., an MMIO read to a powered-down BAR returns 0xFFFFFFFF
// forever), the process blocks indefinitely and the machine appears
// hung.
//
// Recommendation: boot.s should capture CLOCK_MONOTONIC at _start
// and check elapsed time between steps.  If total boot exceeds
// BOOT_WATCHDOG_SECS, abort with a diagnostic and exit.
//
// A host-side watchdog (systemd WatchdogSec= on the lithos service,
// or an external ipmitool-based monitor) is the ultimate safety net
// -- it can power-cycle the node if the GSP process wedges.
//
// These constants define the timeout budget.  Individual step
// timeouts (FALCON_TIMEOUT_SECS=5, TIMEOUT_SECS=30, etc.) should
// sum to less than BOOT_WATCHDOG_SECS.
// ============================================================

.equ BOOT_WATCHDOG_SECS,    120     // 2 minutes for full 9-step boot
.equ STEP_WATCHDOG_SECS,    45      // max time for any single step

// ============================================================
// 9. Shared data strings -- INTENTIONALLY NOT DEFINED HERE
//
// Today the only cross-file duplicated data label is:
//
//     msg_bar0_null / msg_bar0_null_len
//         pmc_check.s:    "gsp: ERROR: BAR0 not mapped (bar0_base == 0)\n"
//         falcon_reset.s: "gsp: ERROR: BAR0 not mapped for falcon reset\n"
//
// Both copies are file-local (not .global) and carry *different
// message text*, so emitting a single canonical string here would
// change observable stderr output.  Unification is deferred to the
// refactor step; for now each file keeps its own wording.
//
// If/when unified, the canonical definition would live here as:
//
//     .section .rodata
//     .align 3
//     .globl msg_bar0_null
//     msg_bar0_null:     .asciz "gsp: ERROR: BAR0 not mapped\n"
//     .equ   msg_bar0_null_len, . - msg_bar0_null - 1
//
// and every caller would adrp/add :lo12:msg_bar0_null.
//
// No other data labels are shared across these 8 files.
// ============================================================
