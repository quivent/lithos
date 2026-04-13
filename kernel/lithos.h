/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * lithos.h — shared header for the lithos.ko kernel module
 *
 * lithos.ko replaces nvidia.ko for the Lithos GPU inference engine.
 * It targets the GH200 (Grace Hopper Superchip, device 0x2342) running
 * Linux 6.8 on ARM64.  The only functionality exposed is the 4 ioctls
 * required to run compute kernels: VRAM allocation, GPFIFO channel
 * creation, USERD doorbell mapping, and ATS coherent-link confirmation.
 *
 * BAR layout (confirmed via /sys/bus/pci/devices/0000:dd:00.0/resource):
 *   BAR0 (resource0): 0x44080000000–0x44080ffffff   16 MB  MMIO registers
 *   BAR2 (resource2): 0x44000000000–0x4407fffffff    2 GB  FB aperture window
 *   BAR4 (resource4): 0x42000000000–0x43fffffffff  128 GB  coherent HBM (C2C)
 *
 * On GH200 the CPU and GPU share HBM via NVLink-C2C (ATS).  BAR4 is the
 * coherent window: CPU stores are GPU-visible without any DMA mapping.
 * BAR0 is the register BAR; PMC_BOOT_0 at BAR0+0x000000 identifies the GPU.
 */

#ifndef _LITHOS_H
#define _LITHOS_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* -----------------------------------------------------------------------
 * PCI identifiers
 * ----------------------------------------------------------------------- */

#define LITHOS_PCI_VENDOR_ID        0x10de   /* NVIDIA */
#define LITHOS_PCI_DEVICE_ID_GH200  0x2342   /* GH100 self-hosted (GH200 SXM) */

/*
 * The self-hosted Hopper range is 0x2340–0x237f (see
 * nvidia-srv-580.105.08/nvidia/detect-self-hosted.h).
 * We match the exact silicon we have; add further IDs to the pci_device_id
 * table in lithos_main.c if needed.
 */

/* -----------------------------------------------------------------------
 * BAR indices (PCI resource numbers, not internal NVIDIA bar_index_t)
 * These are the raw Linux PCI BAR indices as seen in /sys/.../resource.
 * ----------------------------------------------------------------------- */

#define LITHOS_BAR_REGS     0   /* 16 MB MMIO register space */
#define LITHOS_BAR_FB       2   /* 2 GB framebuffer aperture */
#define LITHOS_BAR_COHERENT 4   /* 128 GB coherent HBM (C2C shared memory) */

/* -----------------------------------------------------------------------
 * Key MMIO register offsets within BAR0
 *
 * All offsets are relative to the start of the BAR0 mapping (bar0_base).
 * Source: NVIDIA open-gpu-kernel-modules + manual cross-reference with
 * nvdisasm PTX dumps from the Hopper architecture manual.
 *
 * PMC_BOOT_0: GPU identity register.  Bits [23:20] = architecture (0xa =
 * Hopper), bits [19:15] = implementation, bits [7:4] = revision.
 * Reading this register immediately after BAR0 mapping confirms the GPU.
 *
 * PFIFO_INTR_0 / PFIFO_INTR_EN_0: channel interrupt status and enable.
 * These must be cleared / enabled before submitting work via GPFIFO.
 *
 * USERD_BASE: The USERD (User Engine Ring Data) aperture starts here.
 * Each channel gets a 512-byte USERD page within this aperture.
 * The USERD page for channel N is at: USERD_BASE + N * 0x200.
 * Within each USERD page the relevant offsets (from Nvc86fControl in
 * nvidia-srv-580.105.08/nvidia-uvm/clc86f.h) are:
 *   0x040  Put       — GPFIFO put pointer (CPU writes to advance)
 *   0x044  Get       — GPFIFO get pointer (read-only for CPU)
 *   0x088  GPGet     — GP FIFO get (read-only for CPU)
 *   0x08c  GPPut     — GP FIFO put (CPU writes to ring the doorbell)
 *
 * After writing GPPut the GPU scheduler sees new GPFIFO entries.
 * No syscall is required; the store to the mmap'd USERD page is the
 * entire launch path.
 *
 * TODO (lithos_init.c / lithos_channel.c): confirm exact USERD_BASE
 * offset by reading NV_PFIFO_USERD_BAR_OFFSET from open-source RM:
 *   src/nvidia/arch/nvalloc/unix/src/os.c  (nv_os_map_userd_region)
 *   src/common/sdk/nvidia/inc/class/cl906f.h (historical USERD layout)
 * For Hopper the USERD aperture is in BAR0 at the offset below.
 * ----------------------------------------------------------------------- */

/* PMC — Power Management Controller */
#define NV_PMC_BOOT_0               0x000000u   /* GPU identity */
#define NV_PMC_ENABLE               0x000200u   /* engine enable bits */
#define NV_PMC_INTR_0               0x000100u   /* pending interrupt mask */
#define NV_PMC_INTR_EN_0            0x000140u   /* interrupt enable */

/* PFIFO — GPU FIFO / channel engine */
#define NV_PFIFO_INTR_0             0x002100u   /* channel interrupt status */
#define NV_PFIFO_INTR_EN_0          0x002140u   /* channel interrupt enable */
#define NV_PFIFO_RUNLIST_BASE       0x002270u   /* runlist base address lo32 */
#define NV_PFIFO_RUNLIST_BASE_HI    0x002274u   /* runlist base address hi32 */
#define NV_PFIFO_RUNLIST            0x002278u   /* runlist control */
#define NV_PFIFO_CHANNEL_ENABLE     0x0060f4u   /* channel enable bitmap */

/*
 * USERD aperture — channel doorbell pages mapped to userspace.
 * Hopper USERD base in BAR0:
 *   TODO: verify exact offset from open-gpu-kernel-modules:
 *   src/nvidia/arch/nvalloc/common/inc/hopper/gh100/dev_userd.h
 *   or grep for NV_UDISP_UCODE_USERD_LIMIT in the RM source tree.
 * The value below is the Ampere/Hopper conventional offset; confirm before
 * enabling real hardware access.
 */
#define NV_USERD_BASE               0xfc0000u   /* start of USERD aperture in BAR0 */
#define NV_USERD_PAGE_SIZE          0x200u      /* 512 bytes per channel USERD page */

/* Within each USERD page (Hopper GPFIFO A = class 0xC86F, from clc86f.h) */
#define NVC86F_USERD_PUT            0x040u      /* GP FIFO put index  */
#define NVC86F_USERD_GET            0x044u      /* GP FIFO get index  */
#define NVC86F_USERD_GP_GET         0x088u      /* GPFIFO get pointer */
#define NVC86F_USERD_GP_PUT         0x08cu      /* GPFIFO put pointer — write to launch */

/* -----------------------------------------------------------------------
 * Channel / GPFIFO constants
 * ----------------------------------------------------------------------- */

#define LITHOS_MAX_CHANNELS         64
#define LITHOS_GPFIFO_ENTRIES       1024        /* must be power of 2 */
#define LITHOS_GPFIFO_ENTRY_SIZE    8           /* bytes per GPFIFO entry */
#define LITHOS_GPFIFO_SIZE          (LITHOS_GPFIFO_ENTRIES * LITHOS_GPFIFO_ENTRY_SIZE)

/*
 * A GPFIFO entry is a 64-bit descriptor:
 *   bits [63:42] — length of pushbuffer segment in 32-bit words (22 bits)
 *   bits [41:2]  — GPU VA of pushbuffer, 4-byte aligned (40 bits)
 *   bits [1:0]   — flags (0 = normal entry)
 */
#define LITHOS_GPFIFO_ENTRY(gpu_va, words) \
    (((uint64_t)(words) << 42) | ((uint64_t)(gpu_va) & ~3ULL))

/* -----------------------------------------------------------------------
 * ioctl interface
 * ----------------------------------------------------------------------- */

/*
 * LITHOS_IOCTL_ALLOC_VRAM
 *   Allocate a contiguous region of GPU-accessible memory.
 *   On GH200 with ATS the coherent HBM (BAR4) is the primary pool;
 *   host vmalloc is also GPU-visible via C2C.
 *   IN:  size    — bytes to allocate (page-aligned by driver)
 *   OUT: gpu_va  — GPU virtual address for use in pushbuffers
 *        cpu_va  — CPU virtual address (0 if not CPU-mappable)
 */
#define LITHOS_IOCTL_ALLOC_VRAM     _IOWR('L', 1, struct lithos_alloc)

/*
 * LITHOS_IOCTL_CREATE_CHANNEL
 *   Allocate a GPFIFO ring and register it with the hardware.
 *   OUT: gpfifo_gpu_va  — GPU VA of the GPFIFO ring buffer
 *        gpfifo_entries — number of entries in the ring (always LITHOS_GPFIFO_ENTRIES)
 *        channel_id     — hardware channel ID (0–63)
 */
#define LITHOS_IOCTL_CREATE_CHANNEL _IOR ('L', 2, struct lithos_channel)

/*
 * LITHOS_IOCTL_MAP_USERD
 *   Map the USERD doorbell page for channel_id into the calling process's
 *   address space.  After this call the process can ring the doorbell by
 *   writing GPPut at userd_cpu_va + NVC86F_USERD_GP_PUT with no syscall.
 *   IN:  channel_id    — channel returned by CREATE_CHANNEL
 *   OUT: userd_cpu_va  — CPU virtual address of the 512-byte USERD page
 */
#define LITHOS_IOCTL_MAP_USERD      _IOR ('L', 3, struct lithos_userd)

/*
 * LITHOS_IOCTL_FREE_VRAM
 *   Release a VRAM allocation returned by ALLOC_VRAM.
 *   IN:  gpu_va  — address from ALLOC_VRAM
 *        size    — same size as ALLOC_VRAM
 */
#define LITHOS_IOCTL_FREE_VRAM      _IOW ('L', 4, struct lithos_free)

struct lithos_alloc {
    uint64_t size;      /* IN:  bytes to allocate */
    uint64_t gpu_va;    /* OUT: GPU virtual address */
    uint64_t cpu_va;    /* OUT: CPU virtual address (if mappable, else 0) */
};

struct lithos_channel {
    uint64_t gpfifo_gpu_va;    /* OUT: GPU VA of GPFIFO ring */
    uint32_t gpfifo_entries;   /* OUT: entries in ring */
    uint32_t channel_id;       /* OUT: hardware channel ID */
};

struct lithos_userd {
    uint64_t userd_cpu_va;     /* OUT: CPU VA of USERD page */
    uint32_t channel_id;       /* IN:  channel to map */
    uint32_t pad;
};

struct lithos_free {
    uint64_t gpu_va;    /* IN: address from ALLOC_VRAM */
    uint64_t size;      /* IN: same size */
};

/* -----------------------------------------------------------------------
 * Internal channel descriptor (kernel-only, not exposed via ioctl)
 * ----------------------------------------------------------------------- */

#ifdef __KERNEL__

#include <linux/spinlock.h>
#include <linux/mm_types.h>

struct lithos_chan {
    int             id;             /* hardware channel ID */
    bool            valid;
    void           *gpfifo_cpu;     /* kernel VA of GPFIFO ring (kzalloc'd) */
    uint64_t        gpfifo_phys;    /* physical address of GPFIFO ring */
    uint64_t        gpfifo_gpu_va;  /* GPU VA (== phys on GH200 with ATS) */
    uint32_t        gpfifo_put;     /* software put pointer */
    spinlock_t      lock;
};

#endif /* __KERNEL__ */

#endif /* _LITHOS_H */
