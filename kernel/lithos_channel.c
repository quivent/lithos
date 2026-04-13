// SPDX-License-Identifier: GPL-2.0-only
/*
 * lithos_channel.c — GPFIFO channel creation and USERD doorbell setup
 *
 * A GPFIFO channel is the ring buffer through which the CPU submits GPU work.
 * Each entry in the ring is 8 bytes (a GPFIFO descriptor) pointing to a
 * pushbuffer segment and its length.
 *
 * On Hopper the channel class is 0xC86F (HOPPER_CHANNEL_GPFIFO_A, defined in
 * nvidia-srv-580.105.08/nvidia-uvm/clc86f.h).  The USERD page layout is
 * given by Nvc86fControl (same file):
 *   offset 0x040: Put      — pushbuffer put index  (unused, GPFIFO uses GPPut)
 *   offset 0x044: Get      — pushbuffer get index  (read-only)
 *   offset 0x088: GPGet    — GPFIFO get index      (read-only for CPU)
 *   offset 0x08c: GPPut    — GPFIFO put index      (CPU writes to submit work)
 *
 * The launch sequence after filling N GPFIFO entries is:
 *   1. Write the N GPFIFO descriptors to the GPFIFO ring (CPU writes to HBM)
 *   2. Increment local put pointer: gpfifo_put = (gpfifo_put + N) & (ENTRIES-1)
 *   3. Store gpfifo_put to USERD+0x08c  ← this is the doorbell, 0 syscalls
 *
 * Channel hardware registration:
 *   NVIDIA's RM uses the GSP RPC mechanism (NV0080_CTRL_CMD_GPU_GET_CLASSLIST,
 *   then NVC86F allocation via the object model) to allocate channels on Hopper.
 *   Without GSP, direct MMIO channel allocation is blocked — Hopper requires GSP
 *   for all resource manager operations.
 *
 *   For now this file implements:
 *     (a) GPFIFO ring buffer allocation (kzalloc, page-aligned)
 *     (b) Software channel tracking
 *     (c) The MMIO write sequence that WOULD be required if GSP were bypassed
 *         (stubbed with TODO comments pointing to exact RM source locations)
 *
 *   The stub prints clearly so the missing hardware steps are obvious.
 *
 * Source references:
 *   nvidia-srv-580.105.08/nvidia-uvm/clc86f.h      — Hopper USERD layout
 *   nvidia-srv-580.105.08/nvidia-uvm/uvm_channel.h — channel manager structure
 *   Open-GPU-Kernel-Modules GitHub:
 *     src/nvidia/src/kernel/gpu/fifo/kernel_fifo.c — kfifoConstructChannel
 *     src/nvidia/src/kernel/gpu/fifo/kernel_channel.c — channel allocation
 *     src/nvidia/src/kernel/gpu/fifo/arch/hopper/    — Hopper-specific FIFO
 */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/io.h>
#include <linux/pci.h>

#include "lithos.h"
#include "lithos_dev.h"
#include "lithos_channel.h"
#include "lithos_gsp.h"

/* -----------------------------------------------------------------------
 * MMIO helpers for channel registration
 *
 * These register offsets control the channel engine (PFIFO / NV_PFIFO).
 * On Hopper all PFIFO operations are proxied through GSP.  The offsets
 * below are the BAR0 addresses for direct MMIO access (pre-GSP path,
 * retained as documentation of what GSP does internally).
 *
 * TODO: Verify against:
 *   open-gpu-kernel-modules/src/nvidia/arch/nvalloc/common/inc/hopper/
 *   gh100/dev_fifo.h  (NV_PFIFO_* register definitions for GH100)
 * ----------------------------------------------------------------------- */

/*
 * NV_PFIFO_CHANNEL_INST — channel instance block base (one per channel).
 * The instance block describes the channel to the GPU MMU and FIFO engine.
 * Layout is defined in:
 *   open-gpu-kernel-modules/src/nvidia/arch/nvalloc/common/inc/hopper/
 *   gh100/dev_ram.h  (NV_RAMFC_* fields = channel instance block)
 *
 * Key RAMFC fields (byte offsets within the 4KB instance block):
 *   0x00: GP_BASE_LO    — GPFIFO ring base address [31:0]
 *   0x04: GP_BASE_HI    — GPFIFO ring base address [39:32] | flags
 *   0x08: GP_ENTRIES    — number of GPFIFO entries (log2 or count)
 *   0x10: USERD_ADDR_LO — USERD page physical address [31:0]
 *   0x14: USERD_ADDR_HI — USERD page physical address [39:32]
 *
 * TODO: Read exact offsets from:
 *   open-gpu-kernel-modules/src/nvidia/arch/nvalloc/common/inc/hopper/
 *   gh100/dev_ram.h
 *   Function: kfifoChannelInstBuildHopper() or equivalent
 */

static inline void bar0_write_ch(void __iomem *bar0, uint32_t off, uint32_t val)
{
    iowrite32(val, bar0 + off);
}

static inline uint32_t bar0_read_ch(void __iomem *bar0, uint32_t off)
{
    return ioread32(bar0 + off);
}

/* -----------------------------------------------------------------------
 * Allocate a free channel slot
 * ----------------------------------------------------------------------- */

static int alloc_channel_id(struct lithos_device *ldev)
{
    unsigned long flags;
    int i;

    spin_lock_irqsave(&ldev->chan_lock, flags);
    for (i = 0; i < LITHOS_MAX_CHANNELS; i++) {
        if (!ldev->channels[i].valid) {
            ldev->channels[i].valid = true;
            spin_unlock_irqrestore(&ldev->chan_lock, flags);
            return i;
        }
    }
    spin_unlock_irqrestore(&ldev->chan_lock, flags);
    return -ENOSPC;
}

/* -----------------------------------------------------------------------
 * lithos_create_channel — allocate GPFIFO and register channel
 * ----------------------------------------------------------------------- */

int lithos_create_channel(struct lithos_device *ldev,
                           struct lithos_channel *out)
{
    struct lithos_chan *ch;
    void              *gpfifo_cpu;
    dma_addr_t         gpfifo_dma;
    int                chid;

    chid = alloc_channel_id(ldev);
    if (chid < 0) {
        pr_err("lithos/channel: no free channel slots\n");
        return -ENOSPC;
    }

    ch = &ldev->channels[chid];

    /*
     * Allocate the GPFIFO ring.
     *
     * The ring must be GPU-visible.  On GH200 with ATS the physical address
     * of a kernel allocation IS the GPU VA — no explicit DMA mapping needed.
     *
     * We use dma_alloc_coherent so the allocation is:
     *   - physically contiguous
     *   - cache-coherent from both CPU and GPU perspectives
     *   - below the 4 GB DMA boundary (safe for 32-bit GPU VA fields)
     *
     * dma_alloc_coherent on ARM64 with ATS returns a CPU VA whose PA is
     * directly usable as a GPU VA (the GPU MMU identity-maps system memory
     * through C2C on GH200).
     */
    gpfifo_cpu = dma_alloc_coherent(&ldev->pdev->dev,
                                     LITHOS_GPFIFO_SIZE,
                                     &gpfifo_dma,
                                     GFP_KERNEL | __GFP_ZERO);
    if (!gpfifo_cpu) {
        pr_err("lithos/channel: dma_alloc_coherent failed for GPFIFO "
               "(size=%u)\n", LITHOS_GPFIFO_SIZE);
        ch->valid = false;
        return -ENOMEM;
    }

    ch->gpfifo_cpu   = gpfifo_cpu;
    ch->gpfifo_phys  = (uint64_t)gpfifo_dma;
    ch->gpfifo_gpu_va = (uint64_t)gpfifo_dma;  /* ATS: PA == GPU VA */
    ch->gpfifo_put   = 0;

    pr_info("lithos/channel: ch%d GPFIFO cpu=%px phys=0x%llx entries=%d\n",
            chid, gpfifo_cpu,
            (unsigned long long)ch->gpfifo_phys,
            LITHOS_GPFIFO_ENTRIES);

    /*
     * Hardware channel registration via GSP RPC.
     *
     * On Hopper (GH100/GH200), all FIFO/channel operations are owned by
     * GSP firmware.  Direct MMIO channel allocation is not possible —
     * the hardware faults if the CPU touches PFIFO registers that GSP owns.
     *
     * lithos_gsp_alloc_channel() sends the following RPC sequence:
     *   NV_RM_RPC_ALLOC_ROOT     → allocate client handle
     *   NV_RM_RPC_ALLOC_DEVICE   → allocate device object
     *   NV_RM_RPC_ALLOC_SUBDEVICE → allocate subdevice
     *   NV_RM_RPC_ALLOC_CHANNEL  → allocate GPFIFO channel class 0xC86F
     *
     * Source: _kchannelSendChannelAllocRpc()
     *   /tmp/ogkm/src/nvidia/src/kernel/gpu/fifo/kernel_channel.c
     */
    {
        u32 hw_channel_id = 0;
        int gsp_ret = lithos_gsp_alloc_channel(ldev,
                                                ch->gpfifo_gpu_va,
                                                LITHOS_GPFIFO_ENTRIES,
                                                &hw_channel_id);
        if (gsp_ret == 0) {
            pr_info("lithos/channel: ch%d GSP RPC alloc succeeded: hw_id=%u\n",
                    chid, hw_channel_id);
            /* Update software channel ID to the hardware-assigned ID */
            /* (chid is our software slot; hw_channel_id is the GPU channel) */
        } else {
            pr_warn("lithos/channel: ch%d GSP RPC alloc failed (%d) — "
                    "channel is software-only (requires full GSP boot)\n",
                    chid, gsp_ret);
            pr_warn("lithos/channel: TODO: implement lithos_gsp_alloc_channel()\n");
            pr_warn("lithos/channel:   See lithos_gsp.c lithos_gsp_alloc_channel() stub\n");
        }
    }

    out->gpfifo_gpu_va  = ch->gpfifo_gpu_va;
    out->gpfifo_entries = LITHOS_GPFIFO_ENTRIES;
    out->channel_id     = (uint32_t)chid;

    return 0;
}

/*
 * lithos_destroy_channel — release a channel's resources
 */
void lithos_destroy_channel(struct lithos_device *ldev, int chid)
{
    struct lithos_chan *ch;

    if (chid < 0 || chid >= LITHOS_MAX_CHANNELS)
        return;

    ch = &ldev->channels[chid];

    if (!ch->valid)
        return;

    if (ch->gpfifo_cpu) {
        dma_free_coherent(&ldev->pdev->dev,
                          LITHOS_GPFIFO_SIZE,
                          ch->gpfifo_cpu,
                          (dma_addr_t)ch->gpfifo_phys);
        ch->gpfifo_cpu   = NULL;
        ch->gpfifo_phys  = 0;
        ch->gpfifo_gpu_va = 0;
    }

    ch->valid = false;
    pr_info("lithos/channel: ch%d destroyed\n", chid);
}
