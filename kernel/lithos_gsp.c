// SPDX-License-Identifier: GPL-2.0-only
/*
 * lithos_gsp.c — GH100/GH200 GSP firmware bootstrap for lithos.ko
 *
 * This file implements the minimal GSP boot sequence needed to get GPFIFO
 * channels working on Hopper.  It is ported directly from the NVIDIA
 * open-source kernel module at /tmp/ogkm/ (open-gpu-kernel-modules, tag
 * matching 580.105.08).
 *
 * -----------------------------------------------------------------------
 * ARCHITECTURE OVERVIEW
 * -----------------------------------------------------------------------
 *
 * On GH100/GH200, the GPU Resource Manager (RM) runs inside the GSP — a
 * dedicated RISC-V coprocessor on the GPU die.  The boot chain is:
 *
 *   Host CPU
 *     |
 *     | (1) request_firmware("nvidia/580.105.08/gsp_ga10x.bin")
 *     |     The .fwimage ELF section is the GSP-RM RISC-V binary (~74 MB).
 *     |
 *     | (2) Allocate GSP-FMC boot params in sysmem (coherent with GPU).
 *     |     Populate: WPR meta PA, GSP-RM ELF PA, GSP init args PA.
 *     |
 *     | (3) Write FMC args PA to PGSP_MAILBOX[0/1].
 *     |     Program RISC-V BCR regs with FMC code/data/manifest addresses.
 *     |     Write NV_PRISCV_RISCV_CPUCTL_STARTCPU → FMC starts running.
 *     |
 *     | (4) FSP (Firmware Security Processor) verifies FMC signature and
 *     |     releases PRIV_LOCKDOWN (GPU_FLD_TEST_DRF: RISCV_BR_PRIV_LOCKDOWN).
 *     |     Source: kgspBootstrap_GH100() → kfspSendBootCommands_HAL()
 *     |
 *     | (5) ACR (Access Control Region) sets up WPR2 in HBM, copies GSP-RM
 *     |     ELF, boots GSP-RM RISC-V OS.
 *     |
 *     | (6) Host polls GspStatusQueueInit() — GSP-RM signals "ready" by
 *     |     writing a handshake token to the status queue.
 *     |
 *     | (7) Host sends init RPCs over the message queue:
 *     |       NV_RM_RPC_GSP_SET_SYSTEM_INFO
 *     |       NV_RM_RPC_SET_REGISTRY
 *     |     Then kgspWaitForRmInitDone() polls for GSP_INIT_DONE.
 *     |
 *     | (8) After init done: channel allocation is an RPC:
 *     |       NV_RM_RPC_ALLOC_ROOT → hClient
 *     |       NV_RM_RPC_ALLOC_DEVICE → hDevice
 *     |       NV_RM_RPC_ALLOC_CHANNEL (class 0xC86F) → hChannel, channel_id
 *
 * -----------------------------------------------------------------------
 * SOURCE REFERENCES (all in /tmp/ogkm/)
 * -----------------------------------------------------------------------
 *
 *   src/nvidia/src/kernel/gpu/gsp/arch/hopper/kernel_gsp_gh100.c
 *     kgspResetHw_GH100()         — reset/deassert GSP Falcon
 *     kgspBootstrap_GH100()       — top-level bootstrap (FSP + FMC path)
 *     _kgspBootstrapGspFmc_GH100() — direct FMC bootstrap (no FSP)
 *     kgspSetupGspFmcArgs_GH100() — populate GSP_FMC_BOOT_PARAMS
 *     kgspPopulateWprMeta_GH100() — populate GspFwWprMeta
 *
 *   src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c
 *     kgspInitRm_IMPL()           — top-level GSP init sequence
 *     kgspSendInitRpcs_IMPL()     — send SET_REGISTRY, SET_SYSTEM_INFO
 *     kgspWaitForRmInitDone_IMPL() — poll for GSP_INIT_DONE
 *     GspMsgQueuesInit()          — allocate command/status queues
 *     GspStatusQueueInit()        — link status queue after GSP boots
 *     GspMsgQueueSendCommand()    — send one RPC command
 *     GspMsgQueueReceiveStatus()  — poll for RPC response
 *
 *   src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h
 *     GspFwWprMeta                — WPR layout descriptor (256 bytes)
 *
 *   src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h
 *     GSP_ARGUMENTS_CACHED        — init args passed to GSP-RM
 *     MESSAGE_QUEUE_INIT_ARGUMENTS
 *
 *   src/nvidia/arch/nvalloc/common/inc/rmRiscvUcode.h
 *     RM_RISCV_UCODE_DESC         — FMC binary descriptor
 *
 *   src/common/inc/swref/published/hopper/gh100/dev_gsp.h
 *     NV_PGSP_FALCON_MAILBOX0     0x110040
 *     NV_PGSP_FALCON_MAILBOX1     0x110044
 *     NV_PGSP_FALCON_ENGINE       0x1103c0
 *     NV_PGSP_MAILBOX(i)          0x110804 + i*4
 *     NV_PGSP_QUEUE_HEAD(i)       0x110c00 + i*8
 *
 *   src/common/inc/swref/published/hopper/gh100/dev_falcon_v4.h
 *     NV_PFALCON_FALCON_MAILBOX0  (relative to PGSP base 0x110000) → 0x110040
 *     NV_PFALCON_FALCON_MAILBOX1                                    → 0x110044
 *     NV_PFALCON_FALCON_HWCFG2                                      → 0x1100f4
 *     NV_PFALCON_FALCON_HWCFG2_RISCV_BR_PRIV_LOCKDOWN bit 13
 *
 *   src/common/inc/swref/published/hopper/gh100/dev_riscv_pri.h
 *     NV_FALCON2_GSP_BASE = 0x111000 (RISC-V BCR register window)
 *     NV_PRISCV_RISCV_BCR_DMACFG    (at RISC-V base + 0x66c) → 0x11166c
 *     NV_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_LO  → 0x111678
 *     NV_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_HI  → 0x11167c
 *     NV_PRISCV_RISCV_BCR_DMAADDR_FMCDATA_LO  → 0x111680
 *     NV_PRISCV_RISCV_BCR_DMAADDR_FMCDATA_HI  → 0x111684
 *     NV_PRISCV_RISCV_BCR_DMAADDR_PKCPARAM_LO → 0x111670
 *     NV_PRISCV_RISCV_BCR_DMAADDR_PKCPARAM_HI → 0x111674
 *     NV_PRISCV_RISCV_CPUCTL                   → 0x111388
 *
 * -----------------------------------------------------------------------
 * WHAT IS IMPLEMENTED vs STUBBED
 * -----------------------------------------------------------------------
 *
 * IMPLEMENTED (real code):
 *   - Firmware load via request_firmware()
 *   - ELF section extraction (.fwimage from gsp_ga10x.bin)
 *   - GSP Falcon reset/deassert sequence (exact register writes)
 *   - RISC-V BCR register programming for FMC code/data/manifest
 *   - GspFwWprMeta structure population (the 256-byte descriptor)
 *   - GSP_FMC_BOOT_PARAMS population
 *   - FMC args PA stuffed in PGSP_FALCON_MAILBOX[0/1]
 *   - FMC CPU start via NV_PRISCV_RISCV_CPUCTL_STARTCPU
 *   - PRIV_LOCKDOWN release polling loop
 *   - Message queue memory allocation (command + status queues, 512 KB total)
 *   - RPC send/receive structure layout documented
 *   - Channel alloc RPC structure documented
 *
 * STUBBED (with exact source pointers):
 *   A) FSP boot commands — kfspSendBootCommands_HAL()/kfspPrepareBootCommands_HAL()
 *      Source: src/nvidia/src/kernel/gpu/fsp/kern_fsp.c
 *      Reason: FSP communicates via NV_PFSP_* registers at 0x8F2xxx — a
 *              separate Falcon that requires its own closed-source COT
 *              (Chain-of-Trust) payload.  Without FSP, production GH200
 *              hardware will refuse to release PRIV_LOCKDOWN.
 *      TODO: Port kern_fsp.c kfspSendAndReadMessage() / kfspSendBootCommands()
 *            which write the COT payload via NV_PFSP_EMEMC/EMEMD registers.
 *
 *   B) FMC binary image — pKernelGsp->pGspRmBootUcodeImage / RM_RISCV_UCODE_DESC
 *      Source: nv-kernel.o_binary (prebuilt) symbols:
 *              __kgspGetBinArchiveGspRmBoot_GH100
 *              __kgspGetBinArchiveConcatenatedFMC_GH100
 *      Reason: The FMC is a signed RISC-V binary embedded in the closed
 *              nv-kernel.o_binary blob.  We cannot extract/parse it without
 *              the RM object model (BINDATA_ARCHIVE).
 *      TODO: Implement BINDATA_ARCHIVE parsing to extract the FMC binary,
 *            or provide a standalone extraction tool.
 *
 *   C) Radix3 ELF mapping — kgspCreateRadix3()
 *      Source: src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:5704
 *      Reason: This creates a 3-level page table mapping the GSP-RM ELF image
 *              in a format the GSP-RM OS expects.  Needs LIBOS memory layout.
 *      TODO: Port kgspCreateRadix3_IMPL() — it's ~150 lines of pure C with
 *            no external dependencies.
 *
 *   D) LibOS init args — kgspSetupLibosInitArgs() + GSP_ARGUMENTS_CACHED
 *      Source: src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:~4925
 *      Reason: Needs GPU instance info and the message queue physical addrs.
 *      TODO: Implement after (C) is done.
 *
 *   E) RPC message queue protocol — GspMsgQueueSendCommand/ReceiveStatus
 *      Source: src/nvidia/src/kernel/gpu/gsp/message_queue_cpu.c
 *              src/nvidia/libraries/msgq/ (msgq ring buffer library)
 *      Reason: The queue uses the msgq_priv library (src/nvidia/libraries/msgq/)
 *              which manages producer/consumer ring buffers with alignment and
 *              encryption (CC mode).  350 LOC to port standalone.
 *      TODO: Port message_queue_cpu.c + msgq library.
 *
 *   F) Init RPCs — kgspSendInitRpcs_IMPL()
 *      Source: src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:4558
 *      Reason: NV_RM_RPC_GSP_SET_SYSTEM_INFO and NV_RM_RPC_SET_REGISTRY
 *              RPCs need system info structures populated.
 *      TODO: After (E), build the rpc_structures.h payloads and send.
 *
 *   G) Channel alloc RPC — NV_RM_RPC_ALLOC_CHANNEL (class 0xC86F)
 *      Source: src/nvidia/src/kernel/gpu/fifo/kernel_channel.c:_kchannelSendChannelAllocRpc
 *      Reason: Needs the full RPC infrastructure from (E).
 *      TODO: After (F), send ALLOC_ROOT + ALLOC_DEVICE + ALLOC_CHANNEL RPCs.
 *
 * -----------------------------------------------------------------------
 * CRITICAL BLOCKER: FSP on production GH200
 * -----------------------------------------------------------------------
 *
 * The production GH200 boot path ALWAYS goes through FSP (Firmware Security
 * Processor).  The direct _kgspBootstrapGspFmc_GH100() path only applies
 * when pKernelFsp == NULL, which happens on emulation/pre-silicon or when
 * the FSP is explicitly disabled (PDB_PROP_KFSP_DISABLE_GSPFMC).
 *
 * On a shipping GH200:
 *   kgspBootstrap_GH100():
 *     if (pKernelFsp != NULL && !PDB_PROP_KFSP_DISABLE_GSPFMC):
 *         kfspSendBootCommands_HAL()   ← THIS PATH
 *     else if (pKernelSec2 != NULL && PDB_PROP_KSEC2_BOOT_GSPFMC):
 *         ksec2SendBootCommands_HAL()
 *     else:
 *         _kgspBootstrapGspFmc_GH100() ← only for no-FSP platforms
 *
 * The FSP communicates via its own FALCON (at BAR0 offset 0x8F2000+) using
 * EMEMC/EMEMD registers to transfer a Chain-of-Trust (COT) payload that
 * authorizes the FMC to run.  This payload is constructed by kern_fsp.c
 * (src/nvidia/src/kernel/gpu/fsp/kern_fsp.c, ~2000 LOC).
 *
 * NEXT STEP: Port kern_fsp.c to lithos_fsp.c.
 * Key functions:
 *   kfspPrepareBootCommands_HAL()   — build COT payload
 *   kfspSendBootCommands_HAL()      — write payload via EMEMC/EMEMD
 *   kfspWaitForGspTargetMaskReleased_HAL() — poll 0x8F2xxx register
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/firmware.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/slab.h>
#include <linux/elf.h>
#include <linux/string.h>
#include <linux/jiffies.h>

#include "lithos.h"
#include "lithos_dev.h"
#include "lithos_gsp.h"

/* -----------------------------------------------------------------------
 * GSP firmware path
 * Source: nv-firmware.h — GH100 → NV_FIRMWARE_CHIP_FAMILY_GH100 → "gsp_ga10x"
 * ----------------------------------------------------------------------- */
#define LITHOS_GSP_FW_PATH   "nvidia/580.105.08/gsp_ga10x.bin"

/* -----------------------------------------------------------------------
 * Exact BAR0 register offsets for GSP Falcon and RISC-V BCR
 *
 * Source: /tmp/ogkm/src/common/inc/swref/published/hopper/gh100/dev_gsp.h
 *         and dev_falcon_v4.h and dev_riscv_pri.h
 *
 * The GSP Falcon register window starts at BAR0+0x110000 (NV_PGSP base).
 * The NV_PFALCON_FALCON_* offsets are RELATIVE to this base — the absolute
 * BAR0 addresses are obtained by adding 0x110000.
 *
 * The RISC-V BCR registers are at a separate window: BAR0+0x111000
 * (NV_FALCON2_GSP_BASE).  The NV_PRISCV_RISCV_* offsets are relative to
 * this base.
 * ----------------------------------------------------------------------- */

/* GSP Falcon absolute BAR0 offsets (NV_PGSP_* from dev_gsp.h) */
#define NV_PGSP_BASE                0x110000u
#define NV_GSP_FALCON_MAILBOX0      0x110040u  /* NV_PGSP_FALCON_MAILBOX0 */
#define NV_GSP_FALCON_MAILBOX1      0x110044u  /* NV_PGSP_FALCON_MAILBOX1 */
#define NV_GSP_FALCON_ENGINE        0x1103c0u  /* NV_PGSP_FALCON_ENGINE   */
#define NV_GSP_FALCON_ENGINE_RESET_ASSERT   0x00000001u
#define NV_GSP_FALCON_ENGINE_RESET_STATUS   0x00000700u  /* bits [10:8] */
#define NV_GSP_FALCON_ENGINE_RESET_ASSERTED  0x00000000u /* RESET_STATUS = 0 */
#define NV_GSP_FALCON_ENGINE_RESET_DEASSERTED 0x00000200u /* RESET_STATUS = 2 */

/* NV_PFALCON_FALCON_HWCFG2 at PGSP base + 0xf4 = 0x1100f4 */
#define NV_GSP_FALCON_HWCFG2        0x1100f4u
#define NV_GSP_HWCFG2_RISCV_BR_PRIV_LOCKDOWN_BIT   (1u << 13)
/* bit 13 = 0 → UNLOCK (lockdown released = GSP ready to accept init) */

/* NV_PGSP_MAILBOX(0) for CC regkeys: 0x110804 + 0*4 = 0x110804 */
#define NV_GSP_MAILBOX0             0x110804u

/* GSP queue registers (NV_PGSP_QUEUE_HEAD/TAIL from dev_gsp.h) */
#define NV_GSP_QUEUE_HEAD(i)        (0x110c00u + (i) * 8u)
#define NV_GSP_QUEUE_TAIL(i)        (0x110c04u + (i) * 8u)

/* RISC-V BCR register window: BAR0+0x111000 (NV_FALCON2_GSP_BASE) */
#define NV_FALCON2_GSP_BASE         0x111000u
/* BCR register offsets relative to NV_FALCON2_GSP_BASE (dev_riscv_pri.h) */
#define NV_PRISCV_BCR_DMACFG_OFF    0x66cu  /* absolute = 0x11166c */
#define NV_PRISCV_BCR_PKCPARAM_LO_OFF  0x670u  /* 0x111670 */
#define NV_PRISCV_BCR_PKCPARAM_HI_OFF  0x674u  /* 0x111674 */
#define NV_PRISCV_BCR_FMCCODE_LO_OFF   0x678u  /* 0x111678 */
#define NV_PRISCV_BCR_FMCCODE_HI_OFF   0x67cu  /* 0x11167c */
#define NV_PRISCV_BCR_FMCDATA_LO_OFF   0x680u  /* 0x111680 */
#define NV_PRISCV_BCR_FMCDATA_HI_OFF   0x684u  /* 0x111684 */
#define NV_PRISCV_CPUCTL_OFF        0x388u  /* 0x111388 */

/* BCR_DMACFG field values (dev_riscv_pri.h) */
#define NV_PRISCV_BCR_DMACFG_TARGET_COHERENT_SYSMEM  0x1u
#define NV_PRISCV_BCR_DMACFG_LOCK_LOCKED             (1u << 31)
#define NV_PRISCV_CPUCTL_STARTCPU   0x1u

/* Address alignment: FMC addresses are >> 3 (8-byte aligned) before writing */
#define RISCV_BR_ADDR_ALIGNMENT     8u

/* -----------------------------------------------------------------------
 * GspFwWprMeta — 256-byte WPR layout descriptor sent to ACR
 * Source: /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h
 *
 * On GH200, most fields are 0-initialized because ACR determines the actual
 * offsets.  We fill in what we know: sizes from the firmware image and FMC.
 * ----------------------------------------------------------------------- */
#define GSP_FW_WPR_META_MAGIC       0xdc3aae21371a60b3ULL
#define GSP_FW_WPR_META_REVISION    1ULL
#define GSP_FW_WPR_META_VERIFIED    0xa0a0a0a0a0a0a0a0ULL

/* DMA target encodings (gspifpub.h) */
#define GSP_DMA_TARGET_COHERENT_SYSTEM  1u

struct lithos_gsp_wpr_meta {
    /* magic + revision (offset 0) */
    u64 magic;
    u64 revision;
    /* sysmem addresses of GSP-RM ELF radix3 page table and FMC image */
    u64 sysmemAddrOfRadix3Elf;
    u64 sizeOfRadix3Elf;
    u64 sysmemAddrOfBootloader;   /* FMC (GSP-FMC) image PA */
    u64 sizeOfBootloader;
    /* offsets within the FMC image (from RM_RISCV_UCODE_DESC) */
    u64 bootloaderCodeOffset;
    u64 bootloaderDataOffset;
    u64 bootloaderManifestOffset;
    /* signature of the GSP-RM ELF */
    u64 sysmemAddrOfSignature;
    u64 sizeOfSignature;
    /* FB layout — filled in by ACR on GH200, leave 0 */
    u64 gspFwRsvdStart;
    u64 nonWprHeapOffset;
    u64 nonWprHeapSize;
    u64 gspFwWprStart;
    u64 gspFwHeapOffset;
    u64 gspFwHeapSize;
    u64 gspFwOffset;
    u64 bootBinOffset;
    u64 frtsOffset;
    u64 frtsSize;
    u64 gspFwWprEnd;
    u64 fbSize;
    u64 vgaWorkspaceOffset;
    u64 vgaWorkspaceSize;
    u64 bootCount;
    /* partition RPC fields (unused for first boot, union) */
    u64 partitionRpcAddr;
    u16 partitionRpcRequestOffset;
    u16 partitionRpcReplyOffset;
    u32 elfCodeOffset;
    u32 elfDataOffset;
    u32 elfCodeSize;
    u32 elfDataSize;
    u32 lsUcodeVersion;
    /* heap partition count + flags */
    u8  gspFwHeapVfPartitionCount;
    u8  flags;
    u8  padding[2];
    u32 pmuReservedSize;
    /* verified token written by Booter (read back to confirm) */
    u64 verified;
} __packed;

/* Verify at compile time: structure must be exactly 256 bytes.
 * _Static_assert works at file scope in C11 (kernel uses -std=gnu11). */
_Static_assert(sizeof(struct lithos_gsp_wpr_meta) == 256,
               "lithos_gsp_wpr_meta must be exactly 256 bytes");

/* -----------------------------------------------------------------------
 * RM_RISCV_UCODE_DESC — header at the start of the FMC binary image
 * Source: /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/rmRiscvUcode.h
 * ----------------------------------------------------------------------- */
struct lithos_riscv_ucode_desc {
    u32 version;
    u32 bootloaderOffset;
    u32 bootloaderSize;
    u32 bootloaderParamOffset;
    u32 bootloaderParamSize;
    u32 riscvElfOffset;
    u32 riscvElfSize;
    u32 appVersion;
    u32 manifestOffset;
    u32 manifestSize;
    u32 monitorDataOffset;
    u32 monitorDataSize;
    u32 monitorCodeOffset;
    u32 monitorCodeSize;
    u32 bIsMonitorEnabled;
    u32 swbromCodeOffset;
    u32 swbromCodeSize;
    u32 swbromDataOffset;
    u32 swbromDataSize;
    u32 fbReservedSize;
    u32 bSignedAsCode;
    u32 bIsSmp;
    u32 bIsPlicEnabled;
} __packed;

/* -----------------------------------------------------------------------
 * GSP_FMC_BOOT_PARAMS — struct written to sysmem, PA stuffed in MAILBOX
 * Source: /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/gsp/gspifpub.h
 * ----------------------------------------------------------------------- */
struct lithos_gsp_fmc_init_params {
    u32 regkeys;
} __packed;

struct lithos_gsp_acr_boot_params {
    u32  target;          /* GSP_DMA_TARGET */
    u32  gspRmDescSize;
    u32  _pad0;
    u64  gspRmDescOffset; /* PA of GspFwWprMeta */
    u64  wprCarveoutOffset;
    u32  wprCarveoutSize;
    u8   bIsGspRmBoot;
    u8   bInstInSysMode;
    u8   _pad1[2];
} __packed;

struct lithos_gsp_rm_params {
    u32  target;
    u32  _pad;
    u64  bootArgsOffset;  /* PA of GSP_ARGUMENTS_CACHED */
} __packed;

struct lithos_gsp_spdm_params {
    u32  target;
    u32  _pad;
    u64  payloadBufferOffset;
    u32  payloadBufferSize;
    u32  _pad2;
} __packed;

struct lithos_gsp_rm_mem_params {
    u32  flushSysmemAddrValLo;
    u32  flushSysmemAddrValHi;
} __packed;

struct lithos_gsp_fmc_boot_params {
    struct lithos_gsp_fmc_init_params   initParams;
    struct lithos_gsp_acr_boot_params   bootGspRmParams;
    struct lithos_gsp_rm_params         gspRmParams;
    struct lithos_gsp_spdm_params       gspSpdmParams;
    struct lithos_gsp_rm_mem_params     gspRmMemParams;
} __packed;

/* -----------------------------------------------------------------------
 * MESSAGE_QUEUE_INIT_ARGUMENTS — passed in GSP_ARGUMENTS_CACHED.
 * Source: /tmp/ogkm/src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h
 * ----------------------------------------------------------------------- */
struct lithos_mq_init_args {
    u64 sharedMemPhysAddr;
    u32 pageTableEntryCount;
    u32 _pad;
    u64 cmdQueueOffset;
    u64 statQueueOffset;
    u64 queueElementHdrSize;
    u64 queueElementSizeMin;
    u64 queueElementSizeMax;
    u32 queueHeaderAlign;
    u32 queueElementAlign;
} __packed;

struct lithos_gsp_sr_init_args {
    u32 oldLevel;
    u32 flags;
    u8  bInPMTransition;
    u8  _pad[3];
} __packed;

struct lithos_gsp_args_cached {
    struct lithos_mq_init_args     messageQueueInitArguments;
    struct lithos_gsp_sr_init_args srInitArguments;
    u32 gpuInstance;
    u8  bDmemStack;
    u8  _pad[3];
    /* profilerArgs */
    u64 profilerArgsPa;
    u64 profilerArgsSize;
    /* sysmemHeapArgs */
    u64 sysmemHeapPa;
    u64 sysmemHeapSize;
    /* rmStateMonitorBufferArgs */
    u64 rmStateMonitorPa;
    u64 rmStateMonitorSize;
} __packed;

/* -----------------------------------------------------------------------
 * GSP message queue sizing (from message_queue_cpu.c)
 *   commandQueueSize = 256 KB
 *   statusQueueSize  = 256 KB
 *   Total shared buffer (including page table) ≈ 512+ KB
 * ----------------------------------------------------------------------- */
#define GSP_CMD_QUEUE_SIZE   (256 * 1024)
#define GSP_STAT_QUEUE_SIZE  (256 * 1024)
#define GSP_QUEUE_TOTAL_SIZE (GSP_CMD_QUEUE_SIZE + GSP_STAT_QUEUE_SIZE)

/* -----------------------------------------------------------------------
 * lithos_gsp_state — everything allocated for GSP boot
 * ----------------------------------------------------------------------- */
struct lithos_gsp_state {
    /* GSP firmware ELF blob */
    const struct firmware *fw;
    const void            *fw_image_data;  /* pointer into fw->data (.fwimage) */
    size_t                 fw_image_size;

    /* FMC boot params (coherent sysmem, GPU-accessible) */
    struct lithos_gsp_fmc_boot_params *fmc_params;
    dma_addr_t                         fmc_params_pa;

    /* WPR metadata (coherent sysmem) */
    struct lithos_gsp_wpr_meta *wpr_meta;
    dma_addr_t                  wpr_meta_pa;

    /* GSP init args (coherent sysmem) */
    struct lithos_gsp_args_cached *gsp_args;
    dma_addr_t                     gsp_args_pa;

    /* Message queues (coherent sysmem, 512 KB total) */
    void      *mq_buf;
    dma_addr_t mq_pa;
    size_t     mq_size;
};

/* -----------------------------------------------------------------------
 * BAR0 helper — absolute offset from BAR0 base
 * ----------------------------------------------------------------------- */
static inline u32 gsp_bar0_rd(struct lithos_device *ldev, u32 off)
{
    return ioread32(ldev->bar0 + off);
}

static inline void gsp_bar0_wr(struct lithos_device *ldev, u32 off, u32 val)
{
    iowrite32(val, ldev->bar0 + off);
}

/* RISC-V BCR register: absolute BAR0 = NV_FALCON2_GSP_BASE + relative_off */
static inline void gsp_riscv_wr(struct lithos_device *ldev, u32 rel_off, u32 val)
{
    gsp_bar0_wr(ldev, NV_FALCON2_GSP_BASE + rel_off, val);
}

/* -----------------------------------------------------------------------
 * Step 1: Find a named ELF section in the firmware blob
 *
 * The gsp_ga10x.bin ELF has these sections:
 *   .fwimage       — raw GSP-RM RISC-V binary (the payload, ~74 MB)
 *   .fwversion     — version string (e.g. "580.105.08")
 *   .fwsignatureXX — RSA signatures (one per key family)
 *   .note.gnu.build-id
 *
 * Source: kernel_gsp.c:_kgspFwContainerGetSection()
 * ----------------------------------------------------------------------- */
static int gsp_elf64_find_section(const void *elf_data, size_t elf_size,
                                   const char *name,
                                   const void **section_data,
                                   size_t *section_size)
{
    const Elf64_Ehdr *ehdr;
    const Elf64_Shdr *shdr;
    const char       *shstrtab;
    int               i;

    if (elf_size < sizeof(Elf64_Ehdr))
        return -EINVAL;

    ehdr = (const Elf64_Ehdr *)elf_data;

    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
        ehdr->e_ident[EI_CLASS] != ELFCLASS64)
        return -EINVAL;

    if (ehdr->e_shoff == 0 || ehdr->e_shstrndx >= ehdr->e_shnum)
        return -EINVAL;

    shdr = (const Elf64_Shdr *)((const u8 *)elf_data + ehdr->e_shoff);
    shstrtab = (const char *)elf_data + shdr[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        const char *sec_name = shstrtab + shdr[i].sh_name;
        if (strcmp(sec_name, name) == 0) {
            *section_data = (const u8 *)elf_data + shdr[i].sh_offset;
            *section_size = (size_t)shdr[i].sh_size;
            return 0;
        }
    }
    return -ENOENT;
}

/* -----------------------------------------------------------------------
 * Step 2: Reset and deassert the GSP Falcon
 *
 * Source: kgspResetHw_GH100()
 *   kernel_gsp_gh100.c:112
 *
 *   GPU_FLD_WR_DRF_DEF(pGpu, _PGSP, _FALCON_ENGINE, _RESET, _ASSERT);
 *   [poll until RESET_STATUS == ASSERTED, timeout 10 us]
 *   GPU_FLD_WR_DRF_DEF(pGpu, _FALCON_ENGINE, _RESET, _DEASSERT);
 *   [poll until RESET_STATUS == DEASSERTED, timeout 10 us]
 *
 * NV_PGSP_FALCON_ENGINE (0x1103c0):
 *   bit 0:    RESET = 1 → assert, 0 → deassert
 *   bits[10:8]: RESET_STATUS: 0=ASSERTED, 2=DEASSERTED
 * ----------------------------------------------------------------------- */
static int gsp_reset_hw(struct lithos_device *ldev)
{
    u32 val;
    int retries;

    /* Assert reset (set bit 0) */
    gsp_bar0_wr(ldev, NV_GSP_FALCON_ENGINE, NV_GSP_FALCON_ENGINE_RESET_ASSERT);

    /* Poll for RESET_STATUS == ASSERTED (bits[10:8] = 0) */
    for (retries = 0; retries < 1000; retries++) {
        val = gsp_bar0_rd(ldev, NV_GSP_FALCON_ENGINE);
        if ((val & NV_GSP_FALCON_ENGINE_RESET_STATUS) ==
            NV_GSP_FALCON_ENGINE_RESET_ASSERTED)
            break;
        udelay(1);
    }
    if (retries >= 1000) {
        pr_err("lithos/gsp: timeout waiting for GSP reset to ASSERT\n");
        return -ETIMEDOUT;
    }

    /* Deassert reset (clear bit 0) */
    gsp_bar0_wr(ldev, NV_GSP_FALCON_ENGINE, 0);

    /* Poll for RESET_STATUS == DEASSERTED (bits[10:8] = 2 = 0x200) */
    for (retries = 0; retries < 1000; retries++) {
        val = gsp_bar0_rd(ldev, NV_GSP_FALCON_ENGINE);
        if ((val & NV_GSP_FALCON_ENGINE_RESET_STATUS) ==
            NV_GSP_FALCON_ENGINE_RESET_DEASSERTED)
            break;
        udelay(1);
    }
    if (retries >= 1000) {
        pr_err("lithos/gsp: timeout waiting for GSP reset to DEASSERT\n");
        return -ETIMEDOUT;
    }

    pr_info("lithos/gsp: GSP Falcon reset complete\n");
    return 0;
}

/* -----------------------------------------------------------------------
 * Step 3: Program RISC-V BCR registers and start the FMC
 *
 * Source: _kgspBootstrapGspFmc_GH100() — kernel_gsp_gh100.c:716
 *
 * The FMC (Firmware Management Controller) is a signed RISC-V binary
 * embedded in the closed-source nv-kernel.o_binary.  lithos.ko does NOT
 * have access to it — this function documents the EXACT register write
 * sequence and is called with the correct addresses when the FMC binary
 * is available (stubbed TODO below).
 *
 * Register sequence (from the source):
 *   1. Reset GSP HW (kgspResetHw_GH100)
 *   2. Write FMC args PA to PGSP_FALCON_MAILBOX[0/1]
 *   3. Program FMC code address → NV_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_LO/HI
 *   4. Program FMC data address → NV_PRISCV_RISCV_BCR_DMAADDR_FMCDATA_LO/HI
 *   5. Program manifest address → NV_PRISCV_RISCV_BCR_DMAADDR_PKCPARAM_LO/HI
 *   6. Write BCR_DMACFG: TARGET=COHERENT_SYSMEM, LOCK=LOCKED
 *   7. Write CPUCTL: STARTCPU=1
 *
 * @fmc_image_pa:  physical address of the FMC binary image
 * @fmc_code_off:  RM_RISCV_UCODE_DESC.monitorCodeOffset
 * @fmc_data_off:  RM_RISCV_UCODE_DESC.monitorDataOffset
 * @fmc_manifest_off: RM_RISCV_UCODE_DESC.manifestOffset
 * @fmc_params_pa: physical address of lithos_gsp_fmc_boot_params
 * ----------------------------------------------------------------------- */
static void gsp_bootstrap_fmc(struct lithos_device *ldev,
                               dma_addr_t fmc_image_pa,
                               u32 fmc_code_off,
                               u32 fmc_data_off,
                               u32 fmc_manifest_off,
                               dma_addr_t fmc_params_pa)
{
    u64 phys;

    /* Stuff FMC args PA into MAILBOX[0/1] */
    gsp_bar0_wr(ldev, NV_GSP_FALCON_MAILBOX0, (u32)(fmc_params_pa & 0xffffffffULL));
    gsp_bar0_wr(ldev, NV_GSP_FALCON_MAILBOX1, (u32)(fmc_params_pa >> 32));

    /* Program FMC code address (shifted right by 3 = 8-byte alignment) */
    phys = ((u64)fmc_image_pa + fmc_code_off) >> RISCV_BR_ADDR_ALIGNMENT;
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_FMCCODE_LO_OFF, (u32)(phys & 0xffffffffULL));
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_FMCCODE_HI_OFF, (u32)(phys >> 32));

    /* Program FMC data address */
    phys = ((u64)fmc_image_pa + fmc_data_off) >> RISCV_BR_ADDR_ALIGNMENT;
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_FMCDATA_LO_OFF, (u32)(phys & 0xffffffffULL));
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_FMCDATA_HI_OFF, (u32)(phys >> 32));

    /* Program manifest address */
    phys = ((u64)fmc_image_pa + fmc_manifest_off) >> RISCV_BR_ADDR_ALIGNMENT;
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_PKCPARAM_LO_OFF, (u32)(phys & 0xffffffffULL));
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_PKCPARAM_HI_OFF, (u32)(phys >> 32));

    /* Lock BCR_DMACFG: TARGET=COHERENT_SYSMEM (1), LOCK=1 */
    gsp_riscv_wr(ldev, NV_PRISCV_BCR_DMACFG_OFF,
                 NV_PRISCV_BCR_DMACFG_TARGET_COHERENT_SYSMEM |
                 NV_PRISCV_BCR_DMACFG_LOCK_LOCKED);

    /* Start the RISC-V CPU */
    gsp_riscv_wr(ldev, NV_PRISCV_CPUCTL_OFF, NV_PRISCV_CPUCTL_STARTCPU);

    pr_info("lithos/gsp: RISC-V CPUCTL_STARTCPU written — FMC should be running\n");
}

/* -----------------------------------------------------------------------
 * Step 4: Poll for PRIV_LOCKDOWN release
 *
 * Source: _kgspLockdownReleasedOrFmcError() — kernel_gsp_gh100.c:498
 *
 * After FSP releases the target mask, poll NV_PFALCON_FALCON_HWCFG2 bit 13.
 * When bit 13 (RISCV_BR_PRIV_LOCKDOWN) = 0 → lockdown released, GSP alive.
 * If MAILBOX0 is non-zero and doesn't match boot args PA → FMC error code.
 *
 * Timeout: 30 seconds (NVIDIA uses GPU_TIMEOUT_DEFAULT ≈ a few seconds)
 * ----------------------------------------------------------------------- */
static int gsp_wait_for_lockdown_release(struct lithos_device *ldev,
                                          dma_addr_t fmc_params_pa)
{
    unsigned long deadline = jiffies + msecs_to_jiffies(30000);
    u32 hwcfg2, mailbox0;

    while (time_before(jiffies, deadline)) {
        mailbox0 = gsp_bar0_rd(ldev, NV_GSP_FALCON_MAILBOX0);

        /* If mailbox0 non-zero and doesn't match boot args PA → FMC error */
        if (mailbox0 != 0) {
            if ((u32)(fmc_params_pa & 0xffffffffULL) != mailbox0) {
                pr_err("lithos/gsp: FMC reported error code in MAILBOX0: 0x%08x\n",
                       mailbox0);
                return -EIO;
            }
            /*
             * mailbox0 matches boot args PA — FMC hasn't read/cleared it yet.
             * This is the normal case early in boot; keep polling.
             */
        }

        /* Check HWCFG2 bit 13 (RISCV_BR_PRIV_LOCKDOWN) */
        hwcfg2 = gsp_bar0_rd(ldev, NV_GSP_FALCON_HWCFG2);
        if (!(hwcfg2 & NV_GSP_HWCFG2_RISCV_BR_PRIV_LOCKDOWN_BIT)) {
            pr_info("lithos/gsp: PRIV_LOCKDOWN released — GSP RISC-V running\n");
            return 0;
        }

        usleep_range(100, 200);
    }

    pr_err("lithos/gsp: timeout waiting for PRIV_LOCKDOWN release (30s)\n");
    pr_err("lithos/gsp: HWCFG2=0x%08x MAILBOX0=0x%08x\n",
           gsp_bar0_rd(ldev, NV_GSP_FALCON_HWCFG2),
           gsp_bar0_rd(ldev, NV_GSP_FALCON_MAILBOX0));
    return -ETIMEDOUT;
}

/* -----------------------------------------------------------------------
 * Step 5: Link status queue
 *
 * Source: GspStatusQueueInit() — message_queue_cpu.c
 *         kgspBootstrap_GH100() line 1104: GspStatusQueueInit()
 *
 * After PRIV_LOCKDOWN is released, the host writes the status queue
 * head/tail pointers to NV_PGSP_QUEUE_HEAD/TAIL registers so GSP-RM can
 * find them.  The exact queue index for RM is RPC_TASK_RM_QUEUE_IDX = 0.
 *
 * Source: message_queue_cpu.c:GspStatusQueueInit()
 * ----------------------------------------------------------------------- */
static void gsp_link_status_queue(struct lithos_device *ldev,
                                   dma_addr_t stat_queue_pa)
{
    /*
     * Write status queue physical address to NV_PGSP_QUEUE_HEAD(0).
     * GSP-RM polls this register to find the status queue after init.
     * Source: GspStatusQueueInit() writes pMQI->pStatusQueue addr to HW.
     *
     * TODO: Implement the full GspStatusQueueInit() protocol once the
     * msgq library is ported.  The register write here is the correct
     * action; the exact payload format needs msgq_priv.h structures.
     */
    gsp_bar0_wr(ldev, NV_GSP_QUEUE_HEAD(0), (u32)(stat_queue_pa & 0xffffffffULL));
    pr_info("lithos/gsp: status queue head written: PA=0x%llx\n",
            (unsigned long long)stat_queue_pa);
}

/* -----------------------------------------------------------------------
 * Step 6: Wait for GSP-RM init done
 *
 * Source: kgspWaitForRmInitDone_IMPL() — kernel_gsp.c:6044
 *         Polls the status queue for NV_VGPU_MSG_FUNCTION_GSP_INIT_DONE (0x1001)
 *
 * TODO: Implement after message queue protocol is ported.
 * For now: poll MAILBOX0 for a non-zero "ready" sentinel.
 * ----------------------------------------------------------------------- */
static int gsp_wait_for_rm_init_done(struct lithos_device *ldev)
{
    unsigned long deadline = jiffies + msecs_to_jiffies(30000);
    u32 mailbox0;

    /*
     * TODO: Replace with proper GspMsgQueueReceiveStatus() polling.
     * GSP-RM signals init done by posting NV_VGPU_MSG_FUNCTION_GSP_INIT_DONE
     * (0x1001) to the status queue.
     *
     * kgspWaitForRmInitDone_IMPL():
     *   poll until GspMsgQueueReceiveStatus() returns a message with
     *   function == NV_VGPU_MSG_FUNCTION_GSP_INIT_DONE
     *
     * Source: kernel_gsp.c:6044
     */

    pr_warn("lithos/gsp: kgspWaitForRmInitDone stub — polling MAILBOX0 for ready\n");
    pr_warn("lithos/gsp: TODO: implement GspMsgQueueReceiveStatus() from\n");
    pr_warn("lithos/gsp:   /tmp/ogkm/src/nvidia/src/kernel/gpu/gsp/message_queue_cpu.c\n");
    pr_warn("lithos/gsp:   and msgq library: /tmp/ogkm/src/nvidia/libraries/msgq/\n");

    /*
     * Approximate fallback: MAILBOX0 should be cleared by GSP-FMC during
     * normal boot (see _kgspBootstrapGspFmc_GH100 comment).  If it reads
     * 0 now (after PRIV_LOCKDOWN released), GSP may be running normally.
     */
    while (time_before(jiffies, deadline)) {
        mailbox0 = gsp_bar0_rd(ldev, NV_GSP_FALCON_MAILBOX0);
        if (mailbox0 == 0) {
            pr_info("lithos/gsp: MAILBOX0=0 — GSP may be ready (approximate)\n");
            return 0;
        }
        msleep(10);
    }

    pr_err("lithos/gsp: GSP-RM init done timeout (mailbox0=0x%08x)\n",
           gsp_bar0_rd(ldev, NV_GSP_FALCON_MAILBOX0));
    return -ETIMEDOUT;
}

/* -----------------------------------------------------------------------
 * lithos_gsp_boot — main entry point
 * ----------------------------------------------------------------------- */
int lithos_gsp_boot(struct lithos_device *dev)
{
    struct lithos_gsp_state *gsp;
    const void *fw_image_data;
    size_t      fw_image_size;
    int ret;

    pr_info("lithos/gsp: starting GSP boot sequence for GH200\n");

    gsp = kzalloc(sizeof(*gsp), GFP_KERNEL);
    if (!gsp)
        return -ENOMEM;

    dev->gsp = gsp;

    /* ---- Step 1a: Load GSP firmware ELF -------------------------------- */
    pr_info("lithos/gsp: requesting firmware: %s\n", LITHOS_GSP_FW_PATH);
    ret = request_firmware(&gsp->fw, LITHOS_GSP_FW_PATH, &dev->pdev->dev);
    if (ret) {
        pr_err("lithos/gsp: request_firmware() failed: %d\n", ret);
        goto err_free_gsp;
    }
    pr_info("lithos/gsp: firmware loaded: %zu bytes\n", gsp->fw->size);

    /* ---- Step 1b: Extract .fwimage ELF section -------------------------
     *
     * Source: _kgspFwContainerGetSection() kernel_gsp.c
     *         GSP_IMAGE_SECTION_NAME = ".fwimage"
     *         (g_kernel_gsp_nvoc.h:272)
     *
     * The .fwimage section contains the raw GSP-RM RISC-V binary.
     * This is what gets DMA'd to WPR2 by the FMC/ACR during boot.
     * -------------------------------------------------------------------*/
    ret = gsp_elf64_find_section(gsp->fw->data, gsp->fw->size,
                                  ".fwimage",
                                  &fw_image_data, &fw_image_size);
    if (ret) {
        pr_err("lithos/gsp: failed to find .fwimage section in ELF: %d\n", ret);
        goto err_release_fw;
    }
    gsp->fw_image_data = fw_image_data;
    gsp->fw_image_size = fw_image_size;
    pr_info("lithos/gsp: .fwimage section: %zu bytes\n", fw_image_size);

    /* ---- Step 2: Allocate coherent sysmem buffers ----------------------
     *
     * All buffers must be DMA-coherent (GPU-visible via C2C on GH200).
     * On GH200 with ATS, dma_alloc_coherent returns a PA that is the GPU VA.
     *
     * We allocate:
     *   (a) GSP-FMC boot params    (lithos_gsp_fmc_boot_params, ~64 bytes)
     *   (b) WPR metadata           (lithos_gsp_wpr_meta, exactly 256 bytes)
     *   (c) GSP init args          (lithos_gsp_args_cached, ~128 bytes)
     *   (d) Message queues         (512 KB: 256 KB cmd + 256 KB stat)
     * -------------------------------------------------------------------*/

    /* (a) FMC boot params */
    gsp->fmc_params = dma_alloc_coherent(&dev->pdev->dev,
                                          sizeof(*gsp->fmc_params),
                                          &gsp->fmc_params_pa,
                                          GFP_KERNEL | __GFP_ZERO);
    if (!gsp->fmc_params) {
        pr_err("lithos/gsp: failed to alloc FMC boot params\n");
        ret = -ENOMEM;
        goto err_release_fw;
    }

    /* (b) WPR metadata */
    gsp->wpr_meta = dma_alloc_coherent(&dev->pdev->dev,
                                        sizeof(*gsp->wpr_meta),
                                        &gsp->wpr_meta_pa,
                                        GFP_KERNEL | __GFP_ZERO);
    if (!gsp->wpr_meta) {
        pr_err("lithos/gsp: failed to alloc WPR metadata\n");
        ret = -ENOMEM;
        goto err_free_fmc_params;
    }

    /* (c) GSP init args */
    gsp->gsp_args = dma_alloc_coherent(&dev->pdev->dev,
                                        sizeof(*gsp->gsp_args),
                                        &gsp->gsp_args_pa,
                                        GFP_KERNEL | __GFP_ZERO);
    if (!gsp->gsp_args) {
        pr_err("lithos/gsp: failed to alloc GSP init args\n");
        ret = -ENOMEM;
        goto err_free_wpr_meta;
    }

    /* (d) Message queues (512 KB) */
    gsp->mq_size = GSP_QUEUE_TOTAL_SIZE;
    gsp->mq_buf  = dma_alloc_coherent(&dev->pdev->dev,
                                        gsp->mq_size,
                                        &gsp->mq_pa,
                                        GFP_KERNEL | __GFP_ZERO);
    if (!gsp->mq_buf) {
        pr_err("lithos/gsp: failed to alloc message queues (%zu bytes)\n",
               gsp->mq_size);
        ret = -ENOMEM;
        goto err_free_gsp_args;
    }

    pr_info("lithos/gsp: allocations: fmc_params PA=0x%llx wpr_meta PA=0x%llx\n",
            (u64)gsp->fmc_params_pa, (u64)gsp->wpr_meta_pa);
    pr_info("lithos/gsp:              gsp_args PA=0x%llx mq PA=0x%llx (%zu KB)\n",
            (u64)gsp->gsp_args_pa, (u64)gsp->mq_pa, gsp->mq_size / 1024);

    /* ---- Step 3: Populate WPR metadata ---------------------------------
     *
     * Source: kgspPopulateWprMeta_GH100() — kernel_gsp_gh100.c:302
     *
     * On GH200 most FB layout fields are 0 because ACR fills them in.
     * We provide: magic, revision, image sizes, and the sysmem addresses
     * of the GSP-RM ELF.
     *
     * NOTE: We don't have the Radix3 ELF mapping or the FMC binary yet.
     * sysmemAddrOfRadix3Elf and sysmemAddrOfBootloader are stubs.
     *
     * TODO (C): Implement kgspCreateRadix3() to build the 3-level page table
     * mapping of the GSP-RM ELF image:
     *   Source: kernel_gsp.c:5704 kgspCreateRadix3_IMPL()
     *   ~150 lines, no external dependencies, page table in sysmem.
     *
     * TODO (B): Extract FMC binary from nv-kernel.o_binary BINDATA_ARCHIVE
     *   to get the correct sysmemAddrOfBootloader and RM_RISCV_UCODE_DESC.
     * -------------------------------------------------------------------*/
    gsp->wpr_meta->magic    = GSP_FW_WPR_META_MAGIC;
    gsp->wpr_meta->revision = GSP_FW_WPR_META_REVISION;

    /* GSP-RM ELF image in sysmem — we don't have radix3 yet (TODO C) */
    gsp->wpr_meta->sizeOfRadix3Elf        = fw_image_size;
    gsp->wpr_meta->sysmemAddrOfRadix3Elf  = 0; /* TODO: set to radix3 PA */

    /* FMC binary — we don't have it from nv-kernel.o_binary (TODO B) */
    gsp->wpr_meta->sizeOfBootloader       = 0; /* TODO: FMC size */
    gsp->wpr_meta->sysmemAddrOfBootloader = 0; /* TODO: FMC PA */
    gsp->wpr_meta->bootloaderCodeOffset   = 0; /* TODO: FMC monitorCodeOffset */
    gsp->wpr_meta->bootloaderDataOffset   = 0; /* TODO: FMC monitorDataOffset */
    gsp->wpr_meta->bootloaderManifestOffset = 0; /* TODO: FMC manifestOffset */

    /* Heap sizes from kgspGetNonWprHeapSize/kgspGetFwHeapSize (defaults) */
    gsp->wpr_meta->nonWprHeapSize  = 0x100000;    /* 1 MB non-WPR heap  */
    gsp->wpr_meta->gspFwHeapSize   = 0x2000000;   /* 32 MB WPR heap     */
    gsp->wpr_meta->vgaWorkspaceSize = 128 * 1024; /* 128 KB VGA reserve */
    gsp->wpr_meta->pmuReservedSize  = 0;           /* set by kpmuReservedMemorySizeGet */

    /* FRTS: GH200 has no FRTS (frtsSize = 0 per GA100 note in header) */
    gsp->wpr_meta->frtsSize = 0;

    /* Flags: no PPCIE, no clock boost */
    gsp->wpr_meta->flags = 0;

    pr_info("lithos/gsp: WPR metadata populated (PA=0x%llx)\n",
            (u64)gsp->wpr_meta_pa);

    /* ---- Step 4: Populate FMC boot params ------------------------------
     *
     * Source: kgspSetupGspFmcArgs_GH100() — kernel_gsp_gh100.c:418
     *
     * bootGspRmParams.gspRmDescOffset = PA of WPR meta
     * gspRmParams.bootArgsOffset      = PA of GSP init args
     * -------------------------------------------------------------------*/
    gsp->fmc_params->initParams.regkeys = 0;  /* no debug regkeys */

    gsp->fmc_params->bootGspRmParams.target       = GSP_DMA_TARGET_COHERENT_SYSTEM;
    gsp->fmc_params->bootGspRmParams.gspRmDescSize    = sizeof(*gsp->wpr_meta);
    gsp->fmc_params->bootGspRmParams.gspRmDescOffset  = gsp->wpr_meta_pa;
    gsp->fmc_params->bootGspRmParams.bIsGspRmBoot = 1;

    gsp->fmc_params->gspRmParams.target         = GSP_DMA_TARGET_COHERENT_SYSTEM;
    gsp->fmc_params->gspRmParams.bootArgsOffset = gsp->gsp_args_pa;

    /* No SPDM on this system */
    gsp->fmc_params->gspSpdmParams.target = 0;

    pr_info("lithos/gsp: FMC boot params populated (PA=0x%llx)\n",
            (u64)gsp->fmc_params_pa);

    /* ---- Step 5: Populate GSP init args (message queue addresses) ------
     *
     * Source: kgspPopulateGspRmInitArgs() — kernel_gsp.c:~4928
     *         MESSAGE_QUEUE_INIT_ARGUMENTS populated with queue addresses.
     *
     * The command queue starts at mq_pa.
     * The status queue starts at mq_pa + GSP_CMD_QUEUE_SIZE.
     * -------------------------------------------------------------------*/
    gsp->gsp_args->messageQueueInitArguments.sharedMemPhysAddr  = gsp->mq_pa;
    gsp->gsp_args->messageQueueInitArguments.pageTableEntryCount = 0; /* TODO after radix3 */
    gsp->gsp_args->messageQueueInitArguments.cmdQueueOffset = 0;
    gsp->gsp_args->messageQueueInitArguments.statQueueOffset = GSP_CMD_QUEUE_SIZE;
    gsp->gsp_args->messageQueueInitArguments.queueElementHdrSize = 0;  /* sizeof header */
    gsp->gsp_args->messageQueueInitArguments.queueElementSizeMin = 4096;
    gsp->gsp_args->messageQueueInitArguments.queueElementSizeMax = 4096 * 16;
    gsp->gsp_args->messageQueueInitArguments.queueHeaderAlign    = 4;
    gsp->gsp_args->messageQueueInitArguments.queueElementAlign   = 12; /* RM_PAGE_SHIFT */
    gsp->gsp_args->gpuInstance = 0;

    pr_info("lithos/gsp: GSP init args populated: cmdQ=PA+0x%x statQ=PA+0x%x\n",
            0, GSP_CMD_QUEUE_SIZE);

    /* ---- Step 6: Reset GSP Falcon -------------------------------------- */
    ret = gsp_reset_hw(dev);
    if (ret)
        goto err_free_mq;

    /* ---- Step 7: Bootstrap GSP-FMC ------------------------------------
     *
     * Source: _kgspBootstrapGspFmc_GH100() — kernel_gsp_gh100.c:716
     *
     * CRITICAL BLOCKER: On production GH200, this goes through FSP:
     *   kgspBootstrap_GH100() line 957:
     *     if (pKernelFsp && !PDB_PROP_KFSP_DISABLE_GSPFMC):
     *         kfspSendBootCommands_HAL()   ← required on real hardware
     *
     * Without the FSP boot command sequence, the PRIV_LOCKDOWN will
     * never be released and GSP-RM won't boot.
     *
     * TODO (A): Port kern_fsp.c (~2000 LOC):
     *   kfspPrepareBootCommands_HAL() — builds COT payload
     *   kfspSendBootCommands_HAL()    — writes via NV_PFSP_EMEMC/EMEMD
     *   Source: /tmp/ogkm/src/nvidia/src/kernel/gpu/fsp/kern_fsp.c
     *   FSP registers: /tmp/ogkm/src/common/inc/swref/published/hopper/
     *                  gh100/dev_fsp_pri.h
     *     NV_PFSP_EMEMC(i)   0x8F2ac0 + i*8
     *     NV_PFSP_EMEMD(i)   0x8F2ac4 + i*8
     *     NV_PFSP_QUEUE_HEAD 0x8F2c00
     *     NV_PFSP_MSGQ_HEAD  0x8F2c80
     *
     * For now: attempt the direct FMC path (no FSP).
     * This will succeed on emulation/no-COT platforms but fail on
     * production GH200 — PRIV_LOCKDOWN will not release.
     *
     * TODO (B): Extract FMC binary from nv-kernel.o_binary.
     * Without it we cannot program the BCR registers correctly.
     * The FMC binary is at:
     *   nm nv-kernel.o_binary | grep __kgspGetBinArchiveGspRmBoot_GH100
     *   (returns a BINDATA_ARCHIVE structure pointer)
     * -------------------------------------------------------------------*/

    pr_warn("lithos/gsp: *** STUB *** FSP boot not implemented\n");
    pr_warn("lithos/gsp: TODO (A): port kern_fsp.c kfspSendBootCommands_HAL()\n");
    pr_warn("lithos/gsp:   /tmp/ogkm/src/nvidia/src/kernel/gpu/fsp/kern_fsp.c\n");
    pr_warn("lithos/gsp: TODO (B): extract FMC binary from nv-kernel.o_binary\n");
    pr_warn("lithos/gsp:   symbol: __kgspGetBinArchiveGspRmBoot_GH100\n");
    pr_warn("lithos/gsp: Without (A)+(B), PRIV_LOCKDOWN will not release on GH200\n");

    /*
     * Call gsp_bootstrap_fmc() with zeroed FMC addresses (will fail on
     * real hardware, but documents the correct call site for when (B) is done).
     *
     * When (B) is implemented, replace the 0 values with:
     *   fmc_image_pa = DMA address of FMC binary extracted from BINDATA_ARCHIVE
     *   fmc_code_off = riscv_desc.monitorCodeOffset
     *   fmc_data_off = riscv_desc.monitorDataOffset
     *   fmc_manifest_off = riscv_desc.manifestOffset
     */
    gsp_bootstrap_fmc(dev,
                       0,  /* TODO (B): fmc_image_pa */
                       0,  /* TODO (B): fmc_code_off */
                       0,  /* TODO (B): fmc_data_off */
                       0,  /* TODO (B): fmc_manifest_off */
                       gsp->fmc_params_pa);

    /* ---- Step 8: Wait for PRIV_LOCKDOWN release -----------------------
     *
     * On production GH200 without FSP this will time out.
     * Return 0 anyway so the module loads, but channels will not work.
     * -------------------------------------------------------------------*/
    ret = gsp_wait_for_lockdown_release(dev, gsp->fmc_params_pa);
    if (ret) {
        pr_warn("lithos/gsp: PRIV_LOCKDOWN not released — expected without FSP\n");
        pr_warn("lithos/gsp: module loaded in degraded mode; channels unavailable\n");
        /* Non-fatal: return 0 so module loads for debugging */
        return 0;
    }

    /* ---- Step 9: Link status queue ------------------------------------ */
    gsp_link_status_queue(dev, gsp->mq_pa + GSP_CMD_QUEUE_SIZE);

    /* ---- Step 10: Wait for GSP-RM init done --------------------------- */
    ret = gsp_wait_for_rm_init_done(dev);
    if (ret) {
        pr_warn("lithos/gsp: GSP-RM init done not received\n");
        pr_warn("lithos/gsp: TODO (E): port GspMsgQueueReceiveStatus()\n");
        return 0;
    }

    pr_info("lithos/gsp: GSP boot sequence complete\n");
    return 0;

err_free_mq:
    dma_free_coherent(&dev->pdev->dev, gsp->mq_size, gsp->mq_buf, gsp->mq_pa);
err_free_gsp_args:
    dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->gsp_args),
                      gsp->gsp_args, gsp->gsp_args_pa);
err_free_wpr_meta:
    dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->wpr_meta),
                      gsp->wpr_meta, gsp->wpr_meta_pa);
err_free_fmc_params:
    dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->fmc_params),
                      gsp->fmc_params, gsp->fmc_params_pa);
err_release_fw:
    release_firmware(gsp->fw);
err_free_gsp:
    kfree(gsp);
    dev->gsp = NULL;
    return ret;
}

/* -----------------------------------------------------------------------
 * lithos_gsp_alloc_channel — allocate a GPFIFO channel via GSP RPC
 *
 * Source: _kchannelSendChannelAllocRpc() — kernel_channel.c
 *         NV_RM_RPC_ALLOC_CHANNEL(pGpu, hclient, hparent, hchannel, hclass,
 *                                  pbufferBase, bufferSize, hveaddr,
 *                                  gpFifoEntries, veaSpace, tsgid, status)
 *
 * The RPC sequence for a fresh channel on a fresh client:
 *   1. NV_RM_RPC_ALLOC_ROOT(pGpu, hClient, status)
 *      → allocates client handle
 *   2. NV_RM_RPC_ALLOC_DEVICE(pGpu, hClient, hDevice, status)
 *      → allocates device object under client
 *   3. NV_RM_RPC_ALLOC_SUBDEVICE(pGpu, hClient, hDevice, hSubdevice, status)
 *      → allocates subdevice
 *   4. NV_RM_RPC_ALLOC_CHANNEL(pGpu, hClient, hSubdevice, hChannel,
 *                               0xC86F, gpfifo_gpu_va, 0, 0, entries, ...)
 *      → GSP allocates the channel and returns channel_id
 *
 * All RPCs go through GspMsgQueueSendCommand() / GspMsgQueueReceiveStatus().
 *
 * TODO (F + G): Implement when message queue is ported.
 * ----------------------------------------------------------------------- */
int lithos_gsp_alloc_channel(struct lithos_device *dev,
                               u64 gpfifo_gpu_va,
                               u32 entries,
                               u32 *channel_id_out)
{
    /*
     * TODO: Send the following RPC sequence via GspMsgQueueSendCommand():
     *
     *   hClient    = 0x01000000  (arbitrary client handle)
     *   hDevice    = 0x01000001  (arbitrary device handle)
     *   hSubdevice = 0x01000002  (arbitrary subdevice handle)
     *   hChannel   = 0x01000003  (arbitrary channel handle)
     *
     *   RPC 1: NV_VGPU_MSG_FUNCTION_ALLOC_ROOT (2)
     *     params: { hClient }
     *
     *   RPC 2: NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE (3)
     *     params: { hClient, hDevice, class=0x0080 (NV01_DEVICE_0) }
     *
     *   RPC 3: NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE (19)
     *     params: { hClient, hDevice, hSubdevice, class=0x20E0 (NV20_SUBDEVICE_0) }
     *
     *   RPC 4: NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA (6)
     *     params: {
     *       hClient, hSubdevice, hChannel,
     *       class = 0xC86F,        // HOPPER_CHANNEL_GPFIFO_A
     *       gpFifoBase = gpfifo_gpu_va,
     *       gpFifoEntries = entries,
     *       // all other fields = 0
     *     }
     *
     *   Response: { channel_id }  written to channel_id_out
     *
     * Source:
     *   src/nvidia/inc/kernel/vgpu/rpc_global_enums.h — function codes
     *   src/nvidia/src/kernel/gpu/fifo/kernel_channel.c:_kchannelSendChannelAllocRpc
     *   src/nvidia/generated/g_rpc-structures.h — RPC payload structs
     */

    pr_warn("lithos/gsp: alloc_channel RPC not implemented (GSP not fully booted)\n");
    pr_warn("lithos/gsp: TODO (G): implement NV_RM_RPC_ALLOC_CHANNEL via\n");
    pr_warn("lithos/gsp:   GspMsgQueueSendCommand(function=0x06, payload=...)\n");
    pr_warn("lithos/gsp:   Source: kernel_channel.c:_kchannelSendChannelAllocRpc\n");

    return -ENODEV;
}

/* -----------------------------------------------------------------------
 * lithos_gsp_shutdown — tear down GSP and free resources
 * ----------------------------------------------------------------------- */
void lithos_gsp_shutdown(struct lithos_device *dev)
{
    struct lithos_gsp_state *gsp = dev->gsp;

    if (!gsp)
        return;

    /*
     * TODO: Send NV_RM_RPC_UNLOAD_GSP_RM before reset to let GSP-RM
     * flush state and tear down WPR2.
     * Source: kgspUnloadRm_IMPL() — kernel_gsp.c
     */

    /* Reset the GSP Falcon to halt the RISC-V core */
    gsp_bar0_wr(dev, NV_GSP_FALCON_ENGINE, NV_GSP_FALCON_ENGINE_RESET_ASSERT);
    udelay(100);

    /* Free DMA buffers */
    if (gsp->mq_buf)
        dma_free_coherent(&dev->pdev->dev, gsp->mq_size,
                          gsp->mq_buf, gsp->mq_pa);
    if (gsp->gsp_args)
        dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->gsp_args),
                          gsp->gsp_args, gsp->gsp_args_pa);
    if (gsp->wpr_meta)
        dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->wpr_meta),
                          gsp->wpr_meta, gsp->wpr_meta_pa);
    if (gsp->fmc_params)
        dma_free_coherent(&dev->pdev->dev, sizeof(*gsp->fmc_params),
                          gsp->fmc_params, gsp->fmc_params_pa);

    if (gsp->fw)
        release_firmware(gsp->fw);

    kfree(gsp);
    dev->gsp = NULL;

    pr_info("lithos/gsp: shutdown complete\n");
}
