# SPDX-License-Identifier: GPL-2.0-only
#
# lithos.ko — out-of-tree kernel module Makefile
#
# Usage:
#   make                   — build against running kernel
#   make clean             — remove build artifacts
#   make KDIR=/path/...    — build against a specific kernel tree
#
# The module targets Linux 6.8 on ARM64 (aarch64) as installed on this system.
# Kernel build directory: /lib/modules/$(uname -r)/build

KDIR ?= /lib/modules/$(shell uname -r)/build

# Module object files
lithos-y := lithos_main.o   \
            lithos_mem.o    \
            lithos_channel.o \
            lithos_init.o   \
            lithos_gsp.o

obj-m := lithos.o

# Extra compiler flags
# -DDEBUG enables pr_debug() output (optional)
ccflags-y := -Wall -Wno-unused-function

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

.PHONY: all clean install
