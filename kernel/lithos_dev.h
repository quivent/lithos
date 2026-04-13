/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * lithos_dev.h — internal struct lithos_device definition shared across
 * lithos_main.c, lithos_mem.c, lithos_channel.c, lithos_init.c.
 *
 * Not exported to userspace.
 */

#ifndef _LITHOS_DEV_H
#define _LITHOS_DEV_H

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/spinlock.h>
#include "lithos.h"

struct lithos_gsp_state;   /* forward declaration — defined in lithos_gsp.c */

struct lithos_device {
    struct pci_dev     *pdev;

    /* BAR0: MMIO register space (16 MB) */
    void __iomem       *bar0;
    resource_size_t     bar0_phys;
    resource_size_t     bar0_size;

    /* BAR4: 128 GB coherent HBM window (GH200 NVLink-C2C / ATS) */
    resource_size_t     bar4_phys;
    resource_size_t     bar4_size;

    /* char device */
    struct cdev         cdev;
    dev_t               devno;

    /* channel table */
    struct lithos_chan  channels[LITHOS_MAX_CHANNELS];
    spinlock_t          chan_lock;

    /* GPU identity — read from PMC_BOOT_0 after BAR0 map */
    uint32_t            pmc_boot_0;

    /* GSP firmware state (allocated by lithos_gsp_boot, freed by lithos_gsp_shutdown) */
    struct lithos_gsp_state *gsp;
};

#endif /* _LITHOS_DEV_H */
