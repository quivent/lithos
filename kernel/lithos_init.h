/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LITHOS_INIT_H
#define _LITHOS_INIT_H

struct lithos_device;

int  lithos_gpu_init(struct lithos_device *ldev);
void lithos_gpu_fini(struct lithos_device *ldev);

#endif /* _LITHOS_INIT_H */
