# GSP/ — Userspace GSP boot in ARM64 assembly

This directory contains raw ARM64 assembly that boots NVIDIA's GSP
(GPU System Processor) firmware on a GH200 directly from a Lithos
userspace process. It replaces the C kernel module (`lithos.ko`,
formerly `kernel/lithos_gsp.c` and friends) with `.s` files that
issue Linux syscalls and perform 32-bit MMIO loads/stores against
mmap'd PCI BARs.

Why: on GH200 the C2C/ATS link makes BAR4 a 128 GB coherent HBM
window with CPU PA == GPU VA. Combined with sysfs PCI resource files
(or `vfio-pci`), every operation `lithos.ko` performed — BAR mapping,
coherent allocation, Falcon reset, RISC-V bring-up, FSP Chain-of-Trust,
RPC channel creation — reduces to memory-mapped register pokes from
userspace. No custom kernel module, no libc, no C.

Design rationale and the full register sequence are documented in
`../docs/gsp-native.md`.

## Boot sequence

The nine steps below run in order. Each `.s` file is a standalone
AAPCS-callable routine (or small vocabulary of routines) built with
`as` and linked into the Lithos binary.

1. **BAR mapping** — `bar_map.s`
   Opens `/sys/bus/pci/devices/<BDF>/resource{0,4}` and mmaps BAR0
   (16 MB GPU MMIO) and BAR4 (coherent HBM window). Also parses the
   sysfs `resource` file to recover BAR4's physical base.

2. **PMC identity check** — `pmc_check.s`
   Reads `PMC_BOOT_0` at `BAR0+0x0` and verifies bits[23:20] == 0xA
   (Hopper). Rejects any other architecture.

3. **Falcon reset** — `falcon_reset.s`
   Asserts then deasserts the reset bit at `BAR0+0x1103C0`, polling
   `RESET_STATUS` (bits[10:8]) for ASSERTED then DEASSERTED with a
   bounded timeout.

4. **BAR4 bump allocator** — `hbm_alloc.s`
   Single-threaded 2 MB-aligned bump allocator over BAR4 that hands
   out `(cpu_va, gpu_pa)` pairs for FMC params, WPR metadata, GSP
   init args, message queues, and GPFIFO rings.

5. **GSP firmware load** — `fw_load.s`
   Opens `/lib/firmware/nvidia/.../gsp_ga10x.bin`, walks the ELF64
   section headers to find `.fwimage`, and copies the payload into
   a BAR4 allocation; returns the code/data/manifest offsets that
   step 6 needs.

6. **BCR program + RISC-V start** — `bcr_start.s`
   Writes FMC params pointer to MAILBOX0/1, programs the six BCR
   address registers (`BCR_FMCCODE/FMCDATA/PKCPARAM` lo/hi, each
   shifted right by 3), locks the BCR with DMACFG = `0x80000001`,
   then stores 1 to `CPUCTL` to start the RISC-V core.

7. **FSP communication / Chain-of-Trust** — **NOT YET IMPLEMENTED**
   On production GH200 the FSP (second Falcon at `BAR0+0x8F2000`)
   must authorize the FMC via the EMEMC/EMEMD mailbox protocol before
   PRIV_LOCKDOWN releases. This is a direct port of `kern_fsp.c`
   — approximately 2000 lines of payload construction plus MMIO
   handshakes. It is the bulk of the remaining work.

8. **Lockdown poll** — `poll_lockdown.s`
   Polls `NV_PFALCON_FALCON_HWCFG2` (BAR0+0x1100F4) bit 13 until
   clear with a 30 s `CLOCK_MONOTONIC` timeout, and cross-checks
   MAILBOX0 for FMC error codes.

9. **RPC channel allocation** — `rpc_channel.s`
   After GSP is live, sends `ALLOC_ROOT` / `ALLOC_DEVICE` /
   `ALLOC_CHANNEL` RPCs over the two shared-memory ring buffers in
   BAR4, bumping the queue head/tail registers at
   `NV_PGSP_QUEUE_HEAD/TAIL`, and returns a usable GPFIFO channel.

## Status

8 of 9 steps are implemented in assembly (files for steps 1–6, 8, 9
are present). **Step 7 (FSP Chain-of-Trust) is the remaining work**
and is the largest single piece — everything else combined is under
~1500 lines of assembly; FSP alone is ~2000.

Without step 7, the RISC-V core starts in step 6 but PRIV_LOCKDOWN
never releases in step 8, so the RPC queues in step 9 are not
serviced. End-to-end boot on production silicon requires completing
FSP.

## Build / test

A Makefile is being added separately. In the meantime each unit
assembles standalone:

```
as -o bar_map.o       bar_map.s
as -o pmc_check.o     pmc_check.s
as -o falcon_reset.o  falcon_reset.s
as -o hbm_alloc.o     hbm_alloc.s
as -o fw_load.o       fw_load.s
as -o bcr_start.o     bcr_start.s
as -o poll_lockdown.o poll_lockdown.s
as -o rpc_channel.o   rpc_channel.s
```

Link order is driven by the caller. The eventual Makefile will
stitch these into a single `gsp_boot` entry point invoked from the
Lithos bootstrap.

Test harness: `tools/dump_qmd` and the Lithos launcher exercise the
post-boot path; the pre-boot path is exercised by running the
`gpu-probe` Forth word after `bar_map_init` + `pmc_check`.

## Assumptions

- **GH200 only.** Grace (ARM64) CPU + Hopper GPU, joined by NVLink
  C2C with ATS. The code issues `aarch64` syscalls directly and
  assumes the Hopper register map.
- **BAR4 is coherent and identity-mapped.** C2C/ATS gives the CPU a
  128 GB window where PA == GPU VA and stores are GPU-visible with
  no explicit flush. No DMA API, no IOMMU programming.
- **`nvidia.ko` is unbound** from the target GPU before the Lithos
  process runs. Either sysfs `resource{0,4}` files are readable
  (root or udev ACL) or the device is bound to `vfio-pci`.
- **No interrupts.** Completion is detected by polling — register
  bits during boot, fence values in HBM at runtime. Lithos runs
  cooperative megakernels, so there is nothing to preempt.
- **The PCI BDF is hard-coded.** `bar_map.s` embeds the BDF
  (default `0000:09:00.0`) as an ASCII string in three places; all
  three must be patched together.
