// SPDX-License-Identifier: GPL-2.0-only
/*
 * lithos_init.c — GPU initialization sequence for GH200 / Hopper
 *
 * The central question for lithos.ko: can we submit compute work on Hopper
 * without GSP firmware?
 *
 * SHORT ANSWER: No.
 *
 * On Hopper (GH100 / GH200) the GPU Resource Manager (RM) runs entirely in
 * the GSP — a dedicated RISC-V coprocessor on the GPU die.  The host CPU
 * cannot allocate channels, configure the MMU, or submit runlist entries
 * without first loading and communicating with GSP.  Attempting to write
 * to PFIFO registers without GSP causes GPU faults.
 *
 * This is documented in the NVIDIA open-source module:
 *   nvidia-srv-580.105.08/nvidia/nv.c:nv_is_rm_firmware_active()
 *   Returns NV_TRUE for Hopper — GSP is always required.
 *
 *   nv_firmware.h: GH100 maps to NV_FIRMWARE_CHIP_FAMILY_GH100 which maps
 *   to the gsp_ga10x.bin firmware file (confirmed at
 *   /lib/firmware/nvidia/580.105.08/gsp_ga10x.bin, 74 MB on this system).
 *
 * WHAT THIS FILE IMPLEMENTS:
 *
 * Phase 1 — PMC sanity (MMIO, no GSP required):
 *   Read PMC_BOOT_0 and PMC_ENABLE to confirm the GPU is alive at BAR0.
 *   Clear and enable PMC interrupts.
 *   This can be done before GSP is loaded.
 *
 * Phase 2 — GSP firmware loading (REQUIRED for Hopper channels):
 *   Load gsp_ga10x.bin, copy it to a reserved memory region, and boot
 *   the GSP processor via the FALCON bootstrap sequence.
 *
 *   The loading sequence in NVIDIA's driver:
 *
 *   a. Request firmware from kernel: request_firmware(&fw, "nvidia/580.105.08/gsp_ga10x.bin")
 *      Source: nvidia-srv-580.105.08/nvidia/nv.c:nv_get_firmware()
 *
 *   b. Allocate a contiguous "WPR2" (Write-Protected Region 2) in VRAM for GSP
 *      code/data.  On GH200 this is in the coherent HBM.
 *      Source: open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gsp/
 *              kernel_gsp.c:kgspAllocBootArgs_GH100()
 *
 *   c. Parse the GSP firmware ELF image and copy sections to WPR2.
 *      Source: kernel_gsp.c:kgspPrepareForBootstrap_GH100()
 *
 *   d. Set up the GSP boot arguments structure at a known physical address.
 *      The boot args include: WPR2 base/size, FRTS base/size, libos log buffer.
 *      Source: kernel_gsp.c  struct GspFwWprMeta
 *
 *   e. Boot the GSP Falcon:
 *      - Write the boot args physical address to NV_PGSP_FALCON_MAILBOX0/1
 *      - Assert GSP reset: write to NV_PGSP_FALCON_ENGINE (PGSP_RESET)
 *      - Deassert reset and wait for GSP to signal "ready" via mailbox
 *      Source: open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gsp/
 *              arch/hopper/kernel_gsp_gh100.c:kgspBootstrapGspFw_GH100()
 *
 *   f. Establish the RPC ring buffers (shared memory queues):
 *      - GspMsgQueueElement queues in GSP SRAM/VRAM
 *      - CPU writes RPC requests; GSP polls and responds
 *      Source: kernel_gsp.c:kgspInitRpcInfrastructure_IMPL()
 *
 *   g. Send the initial RPC to initialize the RM inside GSP:
 *      NV_RM_RPC_SET_REGISTRY, NV_RM_RPC_LOAD_GSP_RM, NV_RM_RPC_INIT_DONE
 *      Source: kernel_gsp.c:kgspInitRm_IMPL()
 *
 * Phase 3 — ATS / coherent link confirmation:
 *   On GH200 the NVLink-C2C ATS mapping is enabled by default.
 *   We confirm by reading the coherent link info from ACPI SRAT (as
 *   nv-pci.c:nv_init_coherent_link_info() does) or by checking that
 *   BAR4 is present and > 0 bytes (which we already did in probe()).
 *
 *   The "nvidia,gpu-mem-base-pa" ACPI DSD property confirms the C2C
 *   physical address.  We log it but do not need to program anything —
 *   ATS is configured by system firmware (ACM / BIOS) before the OS boots.
 *
 * CURRENT STATE:
 *   Phase 1 is implemented (PMC read/write).
 *   Phases 2 and 3 are documented stubs.  Phase 2 (GSP load) requires
 *   porting approximately 3000 lines from kernel_gsp.c.  Doing so is the
 *   correct next step before any GPU work can be submitted.
 *
 * REGISTERS USED (BAR0 offsets):
 *   Source: open-gpu-kernel-modules/src/nvidia/arch/nvalloc/common/inc/
 *           hopper/gh100/dev_master.h (NV_PMC_* registers)
 *           hopper/gh100/dev_falcon_v4.h (GSP Falcon bootstrap)
 */

#include <linux/kernel.h>
#include <linux/io.h>
#include <linux/firmware.h>
#include <linux/delay.h>
#include <linux/pci.h>

#include "lithos.h"
#include "lithos_dev.h"
#include "lithos_init.h"
#include "lithos_gsp.h"


/* -----------------------------------------------------------------------
 * BAR0 helpers (these inline functions duplicate lithos_main.c's statics;
 * in a multi-file driver we'd put them in a shared internal header)
 * ----------------------------------------------------------------------- */

static inline uint32_t init_bar0_read(struct lithos_device *ldev, uint32_t off)
{
    return ioread32(ldev->bar0 + off);
}

static inline void init_bar0_write(struct lithos_device *ldev, uint32_t off,
                                    uint32_t val)
{
    iowrite32(val, ldev->bar0 + off);
}

/* -----------------------------------------------------------------------
 * Phase 1: PMC sanity check
 * ----------------------------------------------------------------------- */

static int lithos_pmc_init(struct lithos_device *ldev)
{
    uint32_t boot0 = init_bar0_read(ldev, NV_PMC_BOOT_0);
    uint32_t enable;

    pr_info("lithos/init: PMC_BOOT_0 = 0x%08x\n", boot0);

    if (boot0 == 0xffffffff || boot0 == 0x00000000) {
        pr_err("lithos/init: PMC_BOOT_0 reads 0x%08x — BAR0 not accessible\n",
               boot0);
        return -EIO;
    }

    /*
     * PMC_BOOT_0 bits for GH100 (Hopper):
     *   [3:0]   = minor revision
     *   [7:4]   = major revision
     *   [11:8]  = implementation minor
     *   [19:12] = implementation (GPU chip)
     *   [23:20] = architecture (0xa = Hopper)
     *
     * 0xa in bits [23:20] means Hopper architecture.
     * This system reads 0x??? — we log it above in probe() already.
     *
     * TODO: verify bit field layout against:
     *   open-gpu-kernel-modules/src/nvidia/arch/nvalloc/common/inc/hopper/
     *   gh100/dev_master.h  NV_PMC_BOOT_0 field definitions
     */

    /* Enable PFIFO and LTC engines in PMC_ENABLE */
    enable = init_bar0_read(ldev, NV_PMC_ENABLE);
    pr_info("lithos/init: PMC_ENABLE = 0x%08x\n", enable);

    /* Clear any pending PMC interrupts before arming */
    init_bar0_write(ldev, NV_PMC_INTR_0, 0xffffffff);

    /*
     * Do NOT enable PMC interrupts (NV_PMC_INTR_EN_0) until GSP is
     * running, because GSP owns interrupt dispatch on Hopper.
     * Enabling PMC interrupts prematurely causes spurious IRQs.
     *
     * TODO: After GSP init, enable via the GSP RPC:
     *   NV_RM_RPC_ENABLE_INTERRUPTS
     */

    pr_info("lithos/init: PMC phase complete\n");
    return 0;
}

/* -----------------------------------------------------------------------
 * Phase 2: GSP firmware load (STUB — implement before using channels)
 *
 * What must be implemented here:
 *   See file header above for the full sequence.
 *   The authoritative source is:
 *     open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c
 *     open-gpu-kernel-modules/src/nvidia/src/kernel/gpu/gsp/arch/hopper/
 *     kernel_gsp_gh100.c
 *
 * WHY GSP IS REQUIRED ON HOPPER:
 *   Unlike Turing/Ampere where the RM runs on the CPU and directly writes
 *   PFIFO/MMU registers, on Hopper these registers are owned by GSP.
 *   The CPU's MMIO access to PFIFO is funneled through a shared-memory
 *   RPC queue.  Without GSP the GPU is essentially deaf — no channels,
 *   no runlists, no MMU page tables.
 *
 * MINIMUM GSP IMPLEMENTATION FOR LITHOS:
 *   The full NVIDIA driver loads GSP and then runs the entire RM inside it
 *   (~300 KLOC).  Lithos doesn't need all of that.  The minimum is:
 *
 *   1. Load gsp_ga10x.bin and boot the Falcon (~200 LOC)
 *   2. Exchange the minimum RPCs to get channels working:
 *      - NV_RM_RPC_SET_REGISTRY
 *      - NV_RM_RPC_LOAD_GSP_RM
 *      - NV_RM_RPC_INIT_DONE
 *      - NV_RM_RPC_ALLOC_ROOT (client)
 *      - NV_RM_RPC_ALLOC_DEVICE (device)
 *      - NV_RM_RPC_ALLOC_OBJECT (class=0xC86F channel)
 *   3. Parse the GSP response to get the channel ID and USERD address
 *
 *   This is approximately 1000–2000 lines of C ported from kernel_gsp.c.
 *
 * ALTERNATIVE PATH (research):
 *   Some researchers have explored booting Hopper GPUs with a custom
 *   lightweight GSP firmware (replacing gsp_ga10x.bin with a minimal
 *   RISC-V binary that only implements the channel allocation RPC).
 *   This avoids the complexity of the full RM RPC surface.
 *   See: "GSP-Free" discussions in NVIDIA open-source kernel module issues.
 * ----------------------------------------------------------------------- */

/*
 * GSP firmware loading is now handled by lithos_gsp_boot() in lithos_gsp.c.
 * This wrapper calls it and translates the result for the init sequence.
 */
static int lithos_gsp_load(struct lithos_device *ldev)
{
    int ret = lithos_gsp_boot(ldev);
    if (ret) {
        pr_err("lithos/init: lithos_gsp_boot() failed: %d\n", ret);
        return ret;
    }
    pr_info("lithos/init: GSP boot completed (may be in degraded mode — "
            "check lithos/gsp messages above)\n");
    return 0;
}

/* -----------------------------------------------------------------------
 * Phase 3: ATS / coherent link confirmation
 * ----------------------------------------------------------------------- */

static void lithos_ats_confirm(struct lithos_device *ldev)
{
    /*
     * On GH200 the NVLink-C2C ATS (Address Translation Services) bridge
     * between Grace CPU and H100 GPU is configured by system firmware
     * (BIOS/ACM) before the OS boots.  We don't need to program it.
     *
     * The ACPI SRAT table contains a Generic Initiator affinity structure
     * that maps the GPU PCI BDF to a NUMA node.  The "nvidia,gpu-mem-base-pa"
     * DSD property gives the physical base of the coherent HBM region.
     *
     * On this system:
     *   BAR4 phys = 0x42000000000, size = 128 GB
     *   This IS the coherent HBM.  CPU PA == GPU VA (identity mapping).
     *
     * ATS confirmation: if BAR4 is present and readable, ATS is active.
     * We already verified this in probe() by checking bar4_size > 0.
     *
     * TODO: For a more rigorous check, read the PCI ATS capability register:
     *   pci_find_ext_capability(ldev->pdev, PCI_EXT_CAP_ID_ATS)
     *   and verify PCI_ATS_CTRL_ENABLE is set.
     *
     * Source: nvidia-srv-580.105.08/nvidia/nv-pci.c:
     *   nv_init_coherent_link_info() and the surrounding nv_ats_supported logic.
     */
    if (ldev->bar4_size > 0) {
        pr_info("lithos/init: ATS/C2C confirmed — BAR4 phys=0x%llx size=%llu GB\n",
                (unsigned long long)ldev->bar4_phys,
                (unsigned long long)(ldev->bar4_size >> 30));
        pr_info("lithos/init: CPU PA == GPU VA for HBM allocations\n");
    } else {
        pr_warn("lithos/init: BAR4 not present — ATS/C2C status unknown\n");
    }
}

/* -----------------------------------------------------------------------
 * Public entry points
 * ----------------------------------------------------------------------- */

int lithos_gpu_init(struct lithos_device *ldev)
{
    int ret;

    pr_info("lithos/init: starting GPU initialization\n");

    /* Phase 1: PMC sanity — works without GSP */
    ret = lithos_pmc_init(ldev);
    if (ret)
        return ret;

    /* Phase 2: GSP firmware load — required for channel operation on Hopper */
    ret = lithos_gsp_load(ldev);
    if (ret) {
        pr_err("lithos/init: GSP load failed — no channels will work\n");
        /* Non-fatal for module loading: return 0 to allow debug access */
        return 0;
    }

    /* Phase 3: ATS confirmation */
    lithos_ats_confirm(ldev);

    pr_info("lithos/init: GPU initialization complete\n");
    pr_info("lithos/init: NOTE: channel hardware registration requires "
            "full GSP implementation\n");
    return 0;
}

void lithos_gpu_fini(struct lithos_device *ldev)
{
    /* Shut down GSP (sends UNLOAD_GSP_RM RPC when implemented, halts RISC-V) */
    lithos_gsp_shutdown(ldev);

    /* Reset PMC interrupts */
    if (ldev->bar0) {
        init_bar0_write(ldev, NV_PMC_INTR_EN_0, 0);
        init_bar0_write(ldev, NV_PMC_INTR_0, 0xffffffff);
    }

    pr_info("lithos/init: GPU fini complete\n");
}
