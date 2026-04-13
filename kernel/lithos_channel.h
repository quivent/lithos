/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LITHOS_CHANNEL_H
#define _LITHOS_CHANNEL_H

#include "lithos.h"

struct lithos_device;

int lithos_create_channel(struct lithos_device *ldev,
                           struct lithos_channel *out);

void lithos_destroy_channel(struct lithos_device *ldev, int chid);

#endif /* _LITHOS_CHANNEL_H */
