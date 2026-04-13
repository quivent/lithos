/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LITHOS_MEM_H
#define _LITHOS_MEM_H

#include <linux/types.h>

struct lithos_device;   /* forward declaration */

int  lithos_mem_init(struct lithos_device *ldev,
                     resource_size_t pool_phys, resource_size_t pool_size);
void lithos_mem_fini(struct lithos_device *ldev);

int  lithos_alloc_vram(struct lithos_device *ldev, uint64_t size,
                        uint64_t *gpu_va_out, uint64_t *cpu_va_out);
int  lithos_free_vram(struct lithos_device *ldev,
                       uint64_t gpu_va, uint64_t size);

#endif /* _LITHOS_MEM_H */
