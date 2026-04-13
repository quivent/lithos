/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * lithos_gsp.h — GSP firmware bootstrap interface for lithos.ko
 *
 * Provides the three public entry points for GSP management:
 *   lithos_gsp_boot()          — load firmware, boot GSP RISC-V, init RPC
 *   lithos_gsp_alloc_channel() — allocate a GPFIFO channel via RPC
 *   lithos_gsp_shutdown()      — halt GSP and free resources
 */

#ifndef LITHOS_GSP_H
#define LITHOS_GSP_H

#include "lithos_dev.h"

/*
 * lithos_gsp_boot — bootstrap GSP firmware on GH200/Hopper
 *
 * Loads "nvidia/580.105.08/gsp_ga10x.bin", allocates the required coherent
 * DMA buffers, populates WPR metadata and FMC boot params, resets and starts
 * the GSP Falcon RISC-V CPU, and waits for GSP-RM to signal init complete.
 *
 * Returns 0 on success (or degraded success with a warning printed), or a
 * negative errno if a hard error occurred before the module should load.
 *
 * On production GH200 this will complete partially until the FSP boot
 * command sequence is implemented (lithos_gsp.c TODO A).
 */
int lithos_gsp_boot(struct lithos_device *dev);

/*
 * lithos_gsp_alloc_channel — allocate a GPFIFO channel via GSP RPC
 *
 * @dev:          lithos device
 * @gpfifo_gpu_va: GPU virtual address of the pre-allocated GPFIFO ring buffer
 * @entries:       number of GPFIFO entries (must be power of 2)
 * @channel_id_out: receives the hardware channel ID assigned by GSP
 *
 * Sends the ALLOC_ROOT + ALLOC_DEVICE + ALLOC_CHANNEL RPC sequence to
 * GSP-RM and returns the assigned channel ID.
 *
 * Returns 0 on success, -ENODEV if GSP is not ready, or other negative errno.
 */
int lithos_gsp_alloc_channel(struct lithos_device *dev,
                               u64 gpfifo_gpu_va,
                               u32 entries,
                               u32 *channel_id_out);

/*
 * lithos_gsp_shutdown — tear down GSP and release all resources
 *
 * Sends the UNLOAD_GSP_RM RPC (when implemented), asserts GSP Falcon reset,
 * and frees all DMA buffers and the firmware image.
 *
 * Safe to call even if lithos_gsp_boot() failed partway through.
 */
void lithos_gsp_shutdown(struct lithos_device *dev);

#endif /* LITHOS_GSP_H */
