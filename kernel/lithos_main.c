// SPDX-License-Identifier: GPL-2.0-only
/*
 * lithos_main.c — module init/exit, PCI probe, char device, ioctl dispatch
 *
 * Targets the GH200 (PCI device 10de:2342) on Linux 6.8 / ARM64.
 * BAR0 (16 MB MMIO registers) is mapped with pci_iomap(); PMC_BOOT_0 is
 * read to confirm GPU identity.  The char device /dev/lithos0 exposes
 * the 4-ioctl interface to userspace.
 *
 * Source references:
 *   nvidia-srv-580.105.08/nvidia/nv-pci.c   — probe/remove structure
 *   nvidia-srv-580.105.08/nvidia/nv.c        — ioctl dispatch pattern
 *   nvidia-srv-580.105.08/nvidia/nv-mmap.c   — mmap helper pattern
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/io.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/mm.h>

#include "lithos.h"
#include "lithos_mem.h"
#include "lithos_channel.h"
#include "lithos_init.h"

/* -----------------------------------------------------------------------
 * Module metadata
 * ----------------------------------------------------------------------- */

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Lithos");
MODULE_DESCRIPTION("Lithos GPU driver for GH200 — minimal GPFIFO inference engine");
MODULE_VERSION("0.1");

/* -----------------------------------------------------------------------
 * Global driver state
 * ----------------------------------------------------------------------- */

struct lithos_device {
    struct pci_dev     *pdev;

    /* BAR0: MMIO register space */
    void __iomem       *bar0;
    resource_size_t     bar0_phys;
    resource_size_t     bar0_size;

    /* BAR4: 128 GB coherent HBM window (GH200 NVLink-C2C) */
    resource_size_t     bar4_phys;   /* physical base of coherent HBM */
    resource_size_t     bar4_size;   /* 128 GB */

    /* char device */
    struct cdev         cdev;
    dev_t               devno;

    /* channel table */
    struct lithos_chan  channels[LITHOS_MAX_CHANNELS];
    spinlock_t          chan_lock;  /* protects channel table allocation */

    /* device identity — read from PMC_BOOT_0 after BAR0 map */
    uint32_t            pmc_boot_0;
};

static struct lithos_device *lithos_dev;   /* single-device driver */
static struct class        *lithos_class;
static int                  lithos_major;

/* -----------------------------------------------------------------------
 * Helper: read/write BAR0 MMIO
 * ----------------------------------------------------------------------- */

static inline uint32_t bar0_read(struct lithos_device *ldev, uint32_t off)
{
    return ioread32(ldev->bar0 + off);
}

static inline void bar0_write(struct lithos_device *ldev, uint32_t off,
                               uint32_t val)
{
    iowrite32(val, ldev->bar0 + off);
}

/* -----------------------------------------------------------------------
 * File operations
 * ----------------------------------------------------------------------- */

static int lithos_open(struct inode *inode, struct file *filp)
{
    filp->private_data = lithos_dev;
    return 0;
}

static int lithos_release(struct inode *inode, struct file *filp)
{
    return 0;
}

/*
 * lithos_mmap — map the USERD page for a channel into user VA space.
 *
 * The USERD doorbell lives in BAR0 at:
 *   NV_USERD_BASE + channel_id * NV_USERD_PAGE_SIZE
 *
 * We use remap_pfn_range to create a write-combining mapping so that
 * the CPU store to GPPut is flushed directly to the GPU without a syscall.
 *
 * Userspace must first call LITHOS_IOCTL_MAP_USERD which stores the
 * channel_id in file->private_data's mmap_channel_id field (via the
 * ioctl handler below), then call mmap() to get the actual CPU VA.
 *
 * For the doorbell mapping we map exactly NV_USERD_PAGE_SIZE bytes
 * from the BAR0 USERD aperture.
 *
 * TODO: If GSP firmware is active, USERD may be in a separate aperture
 * rather than BAR0.  Cross-reference:
 *   nvidia-srv-580.105.08/nvidia/nv-mmap.c:nvidia_mmap_helper()
 *   src/nvidia/arch/nvalloc/unix/src/os.c:nv_os_map_userd_region()
 */

struct lithos_file_priv {
    struct lithos_device *ldev;
    int mmap_channel_id;          /* set by MAP_USERD ioctl before mmap() */
};

static int lithos_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct lithos_file_priv *priv = filp->private_data;
    struct lithos_device    *ldev;
    unsigned long            size;
    unsigned long            userd_phys;
    unsigned long            pfn;
    int                      chid;
    int                      ret;

    if (!priv || !priv->ldev)
        return -EINVAL;

    ldev = priv->ldev;
    chid = priv->mmap_channel_id;

    if (chid < 0 || chid >= LITHOS_MAX_CHANNELS)
        return -EINVAL;

    if (!ldev->channels[chid].valid)
        return -ENODEV;

    size = vma->vm_end - vma->vm_start;
    if (size != NV_USERD_PAGE_SIZE)
        return -EINVAL;

    /* Physical address of this channel's USERD page in BAR0 */
    userd_phys = ldev->bar0_phys + NV_USERD_BASE
                 + (unsigned long)chid * NV_USERD_PAGE_SIZE;

    pfn = userd_phys >> PAGE_SHIFT;

    /*
     * Write-combining so that the store to GPPut reaches the GPU without
     * a wbinvd or explicit cache flush from the CPU side.
     */
    vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
    vma->vm_flags    |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;

    ret = remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
    if (ret) {
        pr_err("lithos: remap_pfn_range failed for USERD ch%d: %d\n",
               chid, ret);
        return ret;
    }

    pr_info("lithos: mapped USERD ch%d phys=0x%lx to user VA 0x%lx\n",
            chid, userd_phys, vma->vm_start);
    return 0;
}

/* -----------------------------------------------------------------------
 * ioctl handlers
 * ----------------------------------------------------------------------- */

static long lithos_ioctl_alloc_vram(struct lithos_device *ldev,
                                     unsigned long arg)
{
    struct lithos_alloc req;
    int ret;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    ret = lithos_alloc_vram(ldev, req.size, &req.gpu_va, &req.cpu_va);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    return 0;
}

static long lithos_ioctl_create_channel(struct lithos_device *ldev,
                                         unsigned long arg)
{
    struct lithos_channel req;
    int ret;

    ret = lithos_create_channel(ldev, &req);
    if (ret)
        return ret;

    if (copy_to_user((void __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    return 0;
}

static long lithos_ioctl_map_userd(struct lithos_device *ldev,
                                    struct file *filp,
                                    unsigned long arg)
{
    struct lithos_file_priv *priv = filp->private_data;
    struct lithos_userd req;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    if (req.channel_id >= LITHOS_MAX_CHANNELS)
        return -EINVAL;

    if (!ldev->channels[req.channel_id].valid)
        return -ENODEV;

    /*
     * Record which channel the next mmap() call should map.
     * The actual VA is assigned by the subsequent mmap() call.
     * We return userd_cpu_va = 0 here; the caller gets the real VA
     * from mmap(fd, ..., MAP_SHARED, 0).
     *
     * Alternative: use vm_pgoff encoding as NVIDIA does in nv-mmap.c —
     * encode the channel ID in the mmap offset and decode in lithos_mmap.
     * That avoids the per-fd state but requires the caller to pass
     * (channel_id << PAGE_SHIFT) as the mmap offset.  We use the simpler
     * per-fd approach here.
     */
    priv->mmap_channel_id = (int)req.channel_id;
    req.userd_cpu_va = 0;   /* filled after mmap() */

    if (copy_to_user((void __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    return 0;
}

static long lithos_ioctl_free_vram(struct lithos_device *ldev,
                                    unsigned long arg)
{
    struct lithos_free req;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    return lithos_free_vram(ldev, req.gpu_va, req.size);
}

static long lithos_unlocked_ioctl(struct file *filp, unsigned int cmd,
                                   unsigned long arg)
{
    struct lithos_file_priv *priv = filp->private_data;
    struct lithos_device    *ldev;

    if (!priv || !priv->ldev)
        return -EINVAL;

    ldev = priv->ldev;

    switch (cmd) {
    case LITHOS_IOCTL_ALLOC_VRAM:
        return lithos_ioctl_alloc_vram(ldev, arg);

    case LITHOS_IOCTL_CREATE_CHANNEL:
        return lithos_ioctl_create_channel(ldev, arg);

    case LITHOS_IOCTL_MAP_USERD:
        return lithos_ioctl_map_userd(ldev, filp, arg);

    case LITHOS_IOCTL_FREE_VRAM:
        return lithos_ioctl_free_vram(ldev, arg);

    default:
        return -ENOTTY;
    }
}

static int lithos_open_alloc_priv(struct inode *inode, struct file *filp)
{
    struct lithos_file_priv *priv;

    priv = kzalloc(sizeof(*priv), GFP_KERNEL);
    if (!priv)
        return -ENOMEM;

    priv->ldev           = lithos_dev;
    priv->mmap_channel_id = -1;
    filp->private_data   = priv;
    return 0;
}

static int lithos_release_free_priv(struct inode *inode, struct file *filp)
{
    kfree(filp->private_data);
    filp->private_data = NULL;
    return 0;
}

static const struct file_operations lithos_fops = {
    .owner          = THIS_MODULE,
    .open           = lithos_open_alloc_priv,
    .release        = lithos_release_free_priv,
    .mmap           = lithos_mmap,
    .unlocked_ioctl = lithos_unlocked_ioctl,
};

/* -----------------------------------------------------------------------
 * PCI probe
 * ----------------------------------------------------------------------- */

static int lithos_pci_probe(struct pci_dev *pdev,
                             const struct pci_device_id *id)
{
    struct lithos_device *ldev;
    uint32_t boot0;
    int ret;
    int i;

    pr_info("lithos: probing %04x:%04x (rev %02x)\n",
            pdev->vendor, pdev->device, pdev->revision);

    /* Confirm self-hosted Hopper device range */
    if (pdev->device < 0x2340 || pdev->device > 0x237f) {
        pr_err("lithos: unexpected device ID 0x%04x; expected GH100 "
               "self-hosted (0x2340–0x237f)\n", pdev->device);
        return -ENODEV;
    }

    ldev = kzalloc(sizeof(*ldev), GFP_KERNEL);
    if (!ldev)
        return -ENOMEM;

    ldev->pdev = pdev;
    spin_lock_init(&ldev->chan_lock);
    for (i = 0; i < LITHOS_MAX_CHANNELS; i++) {
        ldev->channels[i].id    = i;
        ldev->channels[i].valid = false;
        spin_lock_init(&ldev->channels[i].lock);
    }

    ret = pci_enable_device_mem(pdev);
    if (ret) {
        pr_err("lithos: pci_enable_device_mem failed: %d\n", ret);
        goto err_free;
    }

    pci_set_master(pdev);

    /*
     * Request BAR0 (register space).  On GH200 this is 16 MB at
     * physical 0x44080000000 (confirmed via /sys/.../resource).
     *
     * We do NOT request BAR4 (coherent HBM) because nvidia.ko is still
     * loaded and owns that region.  In a production lithos.ko deployment
     * where nvidia.ko is absent, also call:
     *   request_mem_region(bar4_start, bar4_size, "lithos_hbm")
     */
    ret = pci_request_region(pdev, LITHOS_BAR_REGS, "lithos_regs");
    if (ret) {
        pr_err("lithos: cannot request BAR0: %d\n", ret);
        goto err_disable;
    }

    ldev->bar0_phys = pci_resource_start(pdev, LITHOS_BAR_REGS);
    ldev->bar0_size = pci_resource_len(pdev, LITHOS_BAR_REGS);

    ldev->bar0 = pci_iomap(pdev, LITHOS_BAR_REGS, ldev->bar0_size);
    if (!ldev->bar0) {
        pr_err("lithos: pci_iomap BAR0 failed\n");
        ret = -ENOMEM;
        goto err_release_regs;
    }

    /* Record coherent HBM window (BAR4) physical base — do not iomap,
     * it is 128 GB and we address it via GPU VA from the bump allocator */
    if (pci_resource_len(pdev, LITHOS_BAR_COHERENT) > 0) {
        ldev->bar4_phys = pci_resource_start(pdev, LITHOS_BAR_COHERENT);
        ldev->bar4_size = pci_resource_len(pdev, LITHOS_BAR_COHERENT);
        pr_info("lithos: coherent HBM BAR4 phys=0x%llx size=%llu GB\n",
                (unsigned long long)ldev->bar4_phys,
                (unsigned long long)(ldev->bar4_size >> 30));
    } else {
        pr_warn("lithos: BAR4 (coherent HBM) not present — "
                "will fall back to vmalloc pool\n");
    }

    /* Read PMC_BOOT_0 to confirm GPU identity */
    boot0 = bar0_read(ldev, NV_PMC_BOOT_0);
    ldev->pmc_boot_0 = boot0;
    pr_info("lithos: PMC_BOOT_0 = 0x%08x  (arch=%u impl=%u rev=%u)\n",
            boot0,
            (boot0 >> 20) & 0xf,   /* bits 23:20 = architecture */
            (boot0 >> 15) & 0x1f,  /* bits 19:15 = implementation */
            (boot0 >>  4) & 0xf);  /* bits  7:4  = revision */

    /*
     * Hopper architecture ID in PMC_BOOT_0[23:20] = 0xa (decimal 10).
     * If we read something else, something is wrong with BAR0 mapping.
     */
    if (((boot0 >> 20) & 0xf) != 0xa) {
        pr_warn("lithos: PMC_BOOT_0 arch field = 0x%x, expected 0xa (Hopper)\n",
                (boot0 >> 20) & 0xf);
        /* Continue anyway — may be pre-GSP state */
    }

    /* Initialize allocator over the coherent HBM pool */
    ret = lithos_mem_init(ldev, ldev->bar4_phys, ldev->bar4_size);
    if (ret) {
        pr_err("lithos: memory allocator init failed: %d\n", ret);
        goto err_unmap;
    }

    /* Minimal GPU init: enable PFIFO, load GSP if required */
    ret = lithos_gpu_init(ldev);
    if (ret) {
        pr_err("lithos: GPU init failed: %d\n", ret);
        goto err_mem;
    }

    /* Register char device */
    ret = alloc_chrdev_region(&ldev->devno, 0, 1, "lithos");
    if (ret) {
        pr_err("lithos: alloc_chrdev_region failed: %d\n", ret);
        goto err_gpu;
    }
    lithos_major = MAJOR(ldev->devno);

    cdev_init(&ldev->cdev, &lithos_fops);
    ldev->cdev.owner = THIS_MODULE;
    ret = cdev_add(&ldev->cdev, ldev->devno, 1);
    if (ret) {
        pr_err("lithos: cdev_add failed: %d\n", ret);
        goto err_chrdev;
    }

    lithos_class = class_create("lithos");
    if (IS_ERR(lithos_class)) {
        ret = PTR_ERR(lithos_class);
        pr_err("lithos: class_create failed: %d\n", ret);
        goto err_cdev;
    }

    if (IS_ERR(device_create(lithos_class, &pdev->dev,
                              ldev->devno, NULL, "lithos0"))) {
        pr_err("lithos: device_create failed\n");
        ret = -ENODEV;
        goto err_class;
    }

    pci_set_drvdata(pdev, ldev);
    lithos_dev = ldev;

    pr_info("lithos: /dev/lithos0 ready — GH200 online\n");
    return 0;

err_class:
    class_destroy(lithos_class);
err_cdev:
    cdev_del(&ldev->cdev);
err_chrdev:
    unregister_chrdev_region(ldev->devno, 1);
err_gpu:
    lithos_gpu_fini(ldev);
err_mem:
    lithos_mem_fini(ldev);
err_unmap:
    pci_iounmap(pdev, ldev->bar0);
err_release_regs:
    pci_release_region(pdev, LITHOS_BAR_REGS);
err_disable:
    pci_disable_device(pdev);
err_free:
    kfree(ldev);
    return ret;
}

static void lithos_pci_remove(struct pci_dev *pdev)
{
    struct lithos_device *ldev = pci_get_drvdata(pdev);

    if (!ldev)
        return;

    device_destroy(lithos_class, ldev->devno);
    class_destroy(lithos_class);
    cdev_del(&ldev->cdev);
    unregister_chrdev_region(ldev->devno, 1);

    lithos_gpu_fini(ldev);
    lithos_mem_fini(ldev);

    pci_iounmap(pdev, ldev->bar0);
    pci_release_region(pdev, LITHOS_BAR_REGS);
    pci_disable_device(pdev);

    lithos_dev = NULL;
    kfree(ldev);

    pr_info("lithos: removed\n");
}

/* -----------------------------------------------------------------------
 * PCI device table
 *
 * Matches GH100 self-hosted devices (0x2340–0x237f).
 * The specific SKU on this system is 0x2342 (GH200 SXM).
 * We also match the full range in case other GH100 variants appear.
 * ----------------------------------------------------------------------- */

static const struct pci_device_id lithos_pci_ids[] = {
    /* GH200 SXM — confirmed device on this system */
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, LITHOS_PCI_DEVICE_ID_GH200) },
    /* Other GH100 self-hosted variants (0x2340–0x237f, skip 0x2342) */
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2340) },
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2341) },
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2343) },
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2344) },
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2345) },
    { PCI_DEVICE(LITHOS_PCI_VENDOR_ID, 0x2346) },
    { 0 }
};
MODULE_DEVICE_TABLE(pci, lithos_pci_ids);

static struct pci_driver lithos_pci_driver = {
    .name     = "lithos",
    .id_table = lithos_pci_ids,
    .probe    = lithos_pci_probe,
    .remove   = lithos_pci_remove,
};

/* -----------------------------------------------------------------------
 * Module init / exit
 * ----------------------------------------------------------------------- */

static int __init lithos_init(void)
{
    pr_info("lithos: loading — targeting GH200 (10de:2342)\n");
    return pci_register_driver(&lithos_pci_driver);
}

static void __exit lithos_exit(void)
{
    pci_unregister_driver(&lithos_pci_driver);
    pr_info("lithos: unloaded\n");
}

module_init(lithos_init);
module_exit(lithos_exit);
