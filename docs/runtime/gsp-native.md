# Eliminating C: Lithos-Native GSP Boot and GPU Access

## Status: Design Proposal

This document describes how to replace lithos.ko (the C kernel module) with
a pure userspace implementation.  The goal: zero C in the project.  GPU
register access, GSP firmware boot, channel creation, and work submission
all happen from the Lithos bootstrap Forth interpreter or from ARM64
machine code emitted by the compiler.

---

## 1. What the kernel module does today

The kernel module (kernel/lithos_main.c and siblings) performs four jobs:

| Job | Files | Kernel API used |
|-----|-------|-----------------|
| Map BAR0 (16 MB GPU registers) | lithos_main.c | pci_iomap |
| Map BAR4 (128 GB coherent HBM) | lithos_main.c | pci_resource_start (phys only) |
| Boot GSP RISC-V coprocessor | lithos_gsp.c | request_firmware, dma_alloc_coherent, iowrite32 |
| Create GPFIFO channel + map USERD doorbell | lithos_channel.c, lithos_main.c | dma_alloc_coherent, remap_pfn_range |

Every one of these reduces to: **map a physical address range and
read/write 32-bit values to it**.  No interrupt handling is on the critical
path.  No DMA engine is programmed (the GPU fetches from coherent memory
on its own).  No kernel-mode page table manipulation is needed beyond the
initial mapping.

---

## 2. Mapping BAR0 and BAR4 from userspace

### 2.1 Options evaluated

| Mechanism | Requires kernel code? | Root? | Maps arbitrary PCI BARs? | Mapping type |
|-----------|----------------------|-------|--------------------------|--------------|
| /sys/bus/pci/devices/.../resource0 | No | Root or file ACL | Yes | mmap, uncacheable |
| /dev/mem + mmap | No | CAP_SYS_RAWIO | Yes (any phys addr) | mmap, any pgprot |
| VFIO (vfio-pci) | No (stock kernel module) | Root + IOMMU group | Yes | ioctl + mmap |
| UIO (uio_pci_generic) | No (stock kernel module) | Root or file ACL | Yes | mmap via /dev/uioN |
| Custom kernel stub | Yes (tiny) | Root | Yes | mmap via char dev |

### 2.2 Recommended: VFIO (primary) with sysfs resource files (fallback)

**VFIO** is the correct production mechanism.  It was designed specifically
for userspace device drivers.  The Linux kernel already ships vfio-pci.
The sequence:

```
# One-time setup (root):
echo 0000:dd:00.0 > /sys/bus/pci/devices/0000:dd:00.0/driver/unbind
echo vfio-pci > /sys/bus/pci/devices/0000:dd:00.0/driver_override
echo 0000:dd:00.0 > /sys/bus/pci/drivers/vfio-pci/bind

# Userspace (Lithos process, needs /dev/vfio/N access):
fd = open("/dev/vfio/GROUP", O_RDWR)
device_fd = ioctl(fd, VFIO_GROUP_GET_DEVICE_FD, "0000:dd:00.0")
info = ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, {index=0})  # BAR0
bar0 = mmap(NULL, 16MB, PROT_READ|PROT_WRITE, MAP_SHARED, device_fd, info.offset)
info = ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, {index=4})  # BAR4
bar4 = mmap(NULL, 128GB, PROT_READ|PROT_WRITE, MAP_SHARED, device_fd, info.offset)
```

On GH200 there is no discrete IOMMU (the C2C link uses ATS); VFIO works
in "no-IOMMU" mode (`echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode`).

**Sysfs resource files** are simpler for development:

```
# BAR0:
fd = open("/sys/bus/pci/devices/0000:dd:00.0/resource0", O_RDWR|O_SYNC)
bar0 = mmap(NULL, 16MB, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)

# BAR4:
fd = open("/sys/bus/pci/devices/0000:dd:00.0/resource4", O_RDWR|O_SYNC)
bar4 = mmap(NULL, 128GB, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)
```

The sysfs path requires: (a) root or a udev rule granting access,
(b) the nvidia driver must be unbound from the device first.
The mapping is uncacheable by default, which is correct for BAR0 (MMIO
registers).  For BAR4 (coherent HBM), write-combining may improve
throughput; /dev/mem with pgprot_writecombine achieves this, or VFIO's
mmap with WC flag.

**Both methods require unbinding nvidia.ko first**.  This is already the
assumption for lithos.ko (it replaces nvidia.ko).  The userspace path
has the same prerequisite but eliminates lithos.ko too.

### 2.3 What about /dev/mem?

Works but is the least safe option.  Requires `CONFIG_DEVMEM=y` and
`CAP_SYS_RAWIO`.  On many distro kernels `/dev/mem` is restricted to the
first 1 MB by `CONFIG_STRICT_DEVMEM`.  GH200 may have `iomem=relaxed` or
a custom kernel — if so, /dev/mem works as a one-liner.  Not recommended
for production.

---

## 3. The GSP boot sequence translated to memory-mapped writes

The entire GSP boot (lithos_gsp.c) is a sequence of 32-bit stores to BAR0
offsets plus allocation of coherent memory buffers.  Here is the exact
register sequence translated to userspace:

### 3.1 Phase 0: Discovery

```
# Read BAR0 physical address from sysfs (or from VFIO region info)
bar0_phys = read_sysfs("resource")  # parse line 0: 0x44080000000
bar4_phys = read_sysfs("resource")  # parse line 4: 0x42000000000

# mmap them
bar0 = mmap(resource0_fd, 16MB)
bar4 = mmap(resource4_fd, 128GB)  # or partial map of needed regions
```

### 3.2 Phase 1: PMC identity check

```forth
\ Read PMC_BOOT_0 at BAR0+0x000000
bar0 @ constant boot0
\ Verify bits[23:20] = 0xa (Hopper)
boot0 20 rshift $f and $a <> abort" Not Hopper"
```

In ARM64 assembly:
```
ldr w0, [x20]            // x20 = bar0 base, offset 0 = PMC_BOOT_0
ubfx w1, w0, #20, #4     // extract arch field
cmp w1, #0xa
b.ne .not_hopper
```

### 3.3 Phase 2: GSP Falcon reset

Exact register sequence from lithos_gsp.c `gsp_reset_hw()`:

```
Store BAR0+0x1103C0 <- 0x00000001    // Assert reset (bit 0)
Poll  BAR0+0x1103C0 bits[10:8] == 0  // Wait for ASSERTED
Store BAR0+0x1103C0 <- 0x00000000    // Deassert reset
Poll  BAR0+0x1103C0 bits[10:8] == 2  // Wait for DEASSERTED
```

### 3.4 Phase 3: Populate structures in coherent memory

The FMC boot params, WPR metadata, GSP init args, and message queues are
all structures written to GPU-visible coherent memory.  In the kernel
module these use `dma_alloc_coherent()`.  In the userspace path:

**BAR4 IS the coherent memory**.  On GH200 with C2C/ATS:
- CPU PA == GPU VA for BAR4 addresses
- CPU stores to BAR4 are GPU-visible without any cache flush
- No DMA mapping, no IOMMU programming, no page table setup

So the replacement for `dma_alloc_coherent()` is:

```
# Bump-allocate from BAR4 (already mapped via sysfs/VFIO)
fmc_params = bar4 + bump_offset
bump_offset += ALIGN(sizeof(fmc_boot_params), 4096)
fmc_params_pa = bar4_phys + (fmc_params - bar4)  # == GPU VA
```

This is exactly what lithos_mem.c already does for VRAM allocations.
The kernel module's `dma_alloc_coherent` was unnecessary overhead --
it allocated from system RAM when the entire 128 GB HBM is already
coherent and CPU-mappable through BAR4.

### 3.5 Phase 4: Load GSP firmware

The kernel module uses `request_firmware()` to load
`nvidia/580.105.08/gsp_ga10x.bin`.  In userspace:

```c
int fd = open("/lib/firmware/nvidia/580.105.08/gsp_ga10x.bin", O_RDONLY);
// read into BAR4 allocation (or mmap the file + memcpy to BAR4)
// parse ELF, find .fwimage section (same logic as gsp_elf64_find_section)
// copy .fwimage payload to BAR4 WPR region
```

In Forth:
```forth
s" /lib/firmware/nvidia/580.105.08/gsp_ga10x.bin" r/o open-file throw
\ slurp into BAR4 at the allocated offset
\ parse ELF64 header, find .fwimage section, record offset+size
```

### 3.6 Phase 5: Program RISC-V BCR and start FMC

Exact register writes from `gsp_bootstrap_fmc()`:

```
Store BAR0+0x110040 <- fmc_params_pa[31:0]    // MAILBOX0
Store BAR0+0x110044 <- fmc_params_pa[63:32]   // MAILBOX1

// FMC code address (shifted right 3 for 8-byte alignment)
fmc_code_pa = (fmc_image_pa + code_offset) >> 3
Store BAR0+0x111678 <- fmc_code_pa[31:0]      // BCR_FMCCODE_LO
Store BAR0+0x11167C <- fmc_code_pa[63:32]     // BCR_FMCCODE_HI

// FMC data address
fmc_data_pa = (fmc_image_pa + data_offset) >> 3
Store BAR0+0x111680 <- fmc_data_pa[31:0]      // BCR_FMCDATA_LO
Store BAR0+0x111684 <- fmc_data_pa[63:32]     // BCR_FMCDATA_HI

// Manifest address
fmc_manifest_pa = (fmc_image_pa + manifest_offset) >> 3
Store BAR0+0x111670 <- fmc_manifest_pa[31:0]  // BCR_PKCPARAM_LO
Store BAR0+0x111674 <- fmc_manifest_pa[63:32] // BCR_PKCPARAM_HI

// Lock BCR and set DMA target = coherent sysmem
Store BAR0+0x11166C <- 0x80000001             // DMACFG: LOCK|COHERENT

// Start RISC-V CPU
Store BAR0+0x111388 <- 0x00000001             // CPUCTL: STARTCPU
```

### 3.7 Phase 6: FSP boot commands (the hard part)

On production GH200, the FSP (Firmware Security Processor) must authorize
the FMC before PRIV_LOCKDOWN releases.  The FSP is a second Falcon at
BAR0+0x8F2000.  Its communication protocol uses EMEMC/EMEMD registers:

```
NV_PFSP_EMEMC(i) = 0x8F2AC0 + i*8
NV_PFSP_EMEMD(i) = 0x8F2AC4 + i*8
NV_PFSP_QUEUE_HEAD = 0x8F2C00
NV_PFSP_MSGQ_HEAD  = 0x8F2C80
```

These are all BAR0 MMIO registers.  **Nothing about FSP communication
requires kernel mode**.  The COT (Chain-of-Trust) payload is built from
data structures and written through the same 32-bit store mechanism.
This is ~2000 lines of logic to port from kern_fsp.c, but it is purely
computational + MMIO writes.  It can run in Forth or ARM64 assembly
identically to how it runs in C.

### 3.8 Phase 7: Poll for lockdown release

```
// Poll BAR0+0x1100F4 bit 13 until clear (timeout 30s)
loop:
    Load BAR0+0x1100F4 -> hwcfg2
    if (hwcfg2 & (1<<13)) == 0: goto done
    usleep(100)
    goto loop
done:
```

### 3.9 Phase 8: Message queue RPC for channel allocation

After GSP boots, channel allocation uses an RPC protocol over shared
memory queues.  The queues live in BAR4 coherent HBM.  The CPU writes
RPC request structures, then writes the queue head register at BAR0.
GSP reads the request, processes it, writes a response to the status
queue, and the CPU polls the status queue head register.

All of this is memory writes to BAR0 (registers) and BAR4 (shared
memory).  No kernel involvement.

---

## 4. Replacing dma_alloc_coherent with BAR4 HBM allocations

### The key insight

On GH200, `dma_alloc_coherent()` allocates from system RAM and sets up
DMA mappings so the GPU can see it.  But BAR4 already IS 128 GB of
coherent GPU-visible memory with identity-mapped PA == GPU VA.  Every
buffer the kernel module allocates via dma_alloc_coherent can instead
be carved from BAR4:

| Buffer | Kernel module | Userspace replacement |
|--------|--------------|----------------------|
| FMC boot params (64B) | dma_alloc_coherent | BAR4 bump alloc |
| WPR metadata (256B) | dma_alloc_coherent | BAR4 bump alloc |
| GSP init args (128B) | dma_alloc_coherent | BAR4 bump alloc |
| Message queues (512KB) | dma_alloc_coherent | BAR4 bump alloc |
| GPFIFO ring (8KB) | dma_alloc_coherent | BAR4 bump alloc |
| Model weights (GBs) | BAR4 bump alloc | BAR4 bump alloc (same) |
| KV cache (GBs) | BAR4 bump alloc | BAR4 bump alloc (same) |

The BAR4 bump allocator from lithos_mem.c translates directly -- the
logic is identical, just with a userspace mmap pointer instead of a
kernel ioremap pointer.

### DMA coherency guarantee

The C2C/ATS link on GH200 provides hardware cache coherence.  CPU stores
to BAR4 addresses are visible to the GPU without any explicit flush.  The
GPU's view of BAR4 physical addresses is identity-mapped (PA == GPU VA).
This is confirmed by:
- lithos.h comment: "CPU PA == GPU VA for HBM allocations"
- lithos_init.c ATS confirmation: BAR4 presence = ATS active
- lithos_channel.c: `ch->gpfifo_gpu_va = (uint64_t)gpfifo_dma` (PA == GPU VA)

No IOMMU programming is needed.  No DMA API is needed.  The hardware
coherence is set up by system firmware before Linux boots.

---

## 5. What minimal kernel support is truly unavoidable?

### 5.1 Things that require NO kernel support

| Function | Why no kernel needed |
|----------|---------------------|
| BAR0/BAR4 mapping | sysfs resource files or VFIO (stock kernel modules) |
| Register reads/writes | Loads/stores to mmap'd memory |
| GSP firmware loading | read() from /lib/firmware/ |
| Coherent memory allocation | Bump allocator over BAR4 mmap |
| ELF parsing | Pure computation |
| Structure population | Pure computation |
| Falcon reset/start | BAR0 register writes |
| FSP communication | BAR0 register writes |
| RPC message queues | BAR4 memory + BAR0 queue head registers |
| USERD doorbell | BAR0 mmap (already userspace-accessible) |
| Work submission | Store to USERD GPPut offset |

### 5.2 Things that DO require kernel support

| Function | Why | Mitigation |
|----------|-----|------------|
| **PCI BAR exposure** | Need some mechanism to mmap PCI BARs | vfio-pci (stock kernel module, zero custom code) |
| **Interrupt delivery** | GPU completion interrupts need a kernel path | **Not needed for Lithos**: polling is correct for inference (the megakernel runs to completion, no preemption) |
| **IOMMU programming** | Needed on discrete GPU systems | **Not needed on GH200**: C2C/ATS provides identity-mapped coherent access |
| **Firmware file access** | request_firmware() loads from /lib/firmware | Direct file read from userspace (open + read, trivial) |

### 5.3 Verdict: Yes, C can be eliminated entirely

The only kernel involvement is **vfio-pci** (or sysfs resource file
exposure), both of which are stock Linux kernel modules requiring zero
custom code.  After PCI BARs are mapped, every remaining operation is
loads and stores to memory-mapped addresses plus file I/O -- all of which
work from userspace.

**Interrupts are not needed.**  The inference engine uses two cooperative
megakernels.  Work submission is a store to USERD GPPut.  Completion
detection is polling GPGet or a fence value in HBM.  There is no
preemption, no context switching, no multi-tenant scheduling.  Polling
is not just acceptable -- it is optimal (lower latency than interrupt
delivery + context switch).

---

## 6. The Forth/ARM64 implementation

### 6.1 Architecture

The Lithos bootstrap Forth interpreter runs on Grace (ARM64).  It already
emits ARM64 machine code and SASS GPU binaries.  The GPU driver becomes
a Forth vocabulary:

```
VOCABULARY gpu

: gpu-init  ( -- )
    bar0-map bar4-map     \ mmap PCI BARs via sysfs/VFIO
    pmc-check             \ read PMC_BOOT_0, verify Hopper
    gsp-boot              \ full GSP bootstrap sequence
    channel-create ;      \ allocate GPFIFO via GSP RPC

: gpu-launch  ( kernel-addr n-words -- )
    gpfifo-entry!         \ write GPFIFO descriptor to ring
    gpput-advance         \ increment software put pointer
    userd-gpput! ;        \ store to USERD+0x8C → GPU starts
```

### 6.2 BAR mapping words

```forth
\ Open sysfs resource files and mmap
: bar0-map  ( -- addr )
    s" /sys/bus/pci/devices/0000:dd:00.0/resource0" r/o open-file throw
    0 $1000000  ( offset=0 size=16MB )
    PROT_READ PROT_WRITE or  MAP_SHARED
    mmap-file ;

: bar4-map  ( -- addr )
    s" /sys/bus/pci/devices/0000:dd:00.0/resource4" r/o open-file throw
    0 $2000000000  ( offset=0 size=128GB — or map a smaller window )
    PROT_READ PROT_WRITE or  MAP_SHARED
    mmap-file ;
```

The `mmap-file` word wraps the Linux mmap syscall (syscall number 222 on
aarch64).  This is a single `svc #0` instruction.  The Forth interpreter
already has file I/O; adding mmap is one word definition.

### 6.3 Register access words

```forth
variable bar0-base

: reg@   ( offset -- value )   bar0-base @ + @ ;     \ 32-bit load
: reg!   ( value offset -- )   bar0-base @ + ! ;     \ 32-bit store

\ GSP Falcon registers
$1103C0 constant FALCON_ENGINE
$110040 constant FALCON_MAILBOX0
$110044 constant FALCON_MAILBOX1
$1100F4 constant FALCON_HWCFG2

\ RISC-V BCR registers
$111678 constant BCR_FMCCODE_LO
$11167C constant BCR_FMCCODE_HI
$111680 constant BCR_FMCDATA_LO
$111684 constant BCR_FMCDATA_HI
$111670 constant BCR_PKCPARAM_LO
$111674 constant BCR_PKCPARAM_HI
$11166C constant BCR_DMACFG
$111388 constant CPUCTL

: gsp-reset  ( -- )
    1 FALCON_ENGINE reg!              \ assert reset
    begin FALCON_ENGINE reg@ $700 and 0= until  \ wait ASSERTED
    0 FALCON_ENGINE reg!              \ deassert
    begin FALCON_ENGINE reg@ $700 and $200 = until ; \ wait DEASSERTED

: fmc-start  ( params-pa -- )
    dup $FFFFFFFF and FALCON_MAILBOX0 reg!
    32 rshift FALCON_MAILBOX1 reg!
    \ ... BCR programming (6 register pairs) ...
    $80000001 BCR_DMACFG reg!         \ LOCK | COHERENT_SYSMEM
    1 CPUCTL reg! ;                   \ start RISC-V
```

### 6.4 HBM allocator

```forth
variable hbm-base      \ BAR4 mmap address
variable hbm-phys      \ BAR4 physical base (0x42000000000)
variable hbm-bump      \ current offset from base

: hbm-alloc  ( size -- cpu-addr gpu-va )
    $1FFFFF + $1FFFFF invert and   \ align to 2MB
    hbm-bump @ over +              \ new bump = old + aligned_size
    hbm-bump !
    hbm-base @ hbm-bump @ + swap - \ cpu-addr = base + old_bump
    dup hbm-base @ - hbm-phys @ +  \ gpu-va = phys + offset
    ;
```

### 6.5 Alternatively: ARM64 assembly

For maximum control, the GSP boot sequence can be emitted as ARM64
machine code by the Lithos compiler.  The compiler already has an ARM64
backend (compiler/arm64-wrap.fs, compiler/elf-wrap.fs).  The entire
boot sequence is approximately 500 ARM64 instructions:

- ~20 instructions for mmap syscalls (BAR0, BAR4)
- ~30 instructions for PMC check
- ~50 instructions for Falcon reset + BCR programming
- ~200 instructions for structure population (memset + field stores)
- ~100 instructions for FSP communication
- ~50 instructions for polling loops
- ~50 instructions for ELF section parsing

This produces a standalone ARM64 binary (~2KB of code) that boots the
GPU with zero dependencies.  No libc, no kernel module, no C compiler.

---

## 7. Migration plan

### Phase 1: Prove the path (week 1)

1. **Unbind nvidia.ko**, bind vfio-pci (or leave unbound for sysfs access)
2. **Add `mmap` syscall word** to the Forth interpreter (one definition,
   wraps syscall 222)
3. **Map BAR0 via sysfs resource0**, read PMC_BOOT_0 from Forth
4. **Confirm register access works** -- if PMC_BOOT_0 reads 0x????????
   with arch=0xa, the path is proven

Deliverable: `gpu-probe` Forth word that prints PMC_BOOT_0.

### Phase 2: GSP boot from Forth (weeks 2-3)

1. **Map BAR4**, implement bump allocator in Forth
2. **Port ELF64 section parser** (gsp_elf64_find_section is 30 lines)
3. **Port Falcon reset** (gsp_reset_hw is 20 lines of register pokes)
4. **Port FMC bootstrap** (gsp_bootstrap_fmc is 15 register writes)
5. **Port FSP communication** (kern_fsp.c, ~2000 lines of C → ~800
   lines of Forth or ~500 ARM64 instructions)
6. **Port message queue protocol** (msgq library, ~350 lines)
7. **Send init RPCs**, wait for GSP_INIT_DONE

Deliverable: `gsp-boot` Forth word that boots GSP to the "ready" state.

### Phase 3: Channel creation and launch (week 4)

1. **Send ALLOC_ROOT/DEVICE/CHANNEL RPCs** via message queue
2. **Implement GPFIFO ring** in BAR4 (bump-allocate 8KB)
3. **Map USERD page** (it is in BAR0 at 0xFC0000 + chid*0x200, already
   mmap'd as part of BAR0)
4. **Implement gpu-launch** word: write GPFIFO entry, store GPPut

Deliverable: First Lithos kernel launch with zero C on the system.

### Phase 4: Delete kernel/ directory (week 4)

Once gpu-launch works from Forth:
1. `rm -rf kernel/`
2. Remove all references to lithos.ko from documentation
3. Add `gpu-setup.sh` script that unbinds nvidia.ko and sets permissions

### What to keep

- **All register offsets and structure definitions** from lithos_gsp.c --
  these are hardware constants, language-independent
- **The boot sequence logic** -- translates 1:1 from C to Forth/ARM64
- **The bump allocator design** from lithos_mem.c
- **The GPFIFO entry format** and USERD layout from lithos.h

### What to rewrite

- lithos_gsp.c (GSP boot) → Forth words or ARM64 code
- lithos_channel.c (channel creation) → Forth words
- lithos_mem.c (allocator) → Forth words (simpler: no spinlocks needed,
  single-threaded bootstrap)
- lithos_main.c (PCI probe, char device) → replaced by sysfs/VFIO mmap

### What to delete

- lithos_main.c — PCI driver framework, char device, ioctl dispatch
  (all replaced by 3 mmap syscalls)
- lithos_init.c — wrapper around gsp_boot (folded into boot word)
- lithos_dev.h — kernel-internal struct (replaced by Forth variables)
- Makefile, Module.symvers, *.ko, *.o — kernel build artifacts
- All linux/ kernel headers

---

## 8. Summary

**Can C be eliminated entirely?  Yes.**

The GH200's BAR4 coherent HBM window is the key enabler.  On a discrete
GPU you need kernel-mode DMA mapping to make system RAM visible to the
GPU.  On GH200, BAR4 gives the CPU direct coherent access to all GPU
memory -- no DMA API, no IOMMU, no kernel.  BAR0 gives access to all GPU
registers.  Both can be mmap'd from userspace via stock Linux mechanisms
(vfio-pci or sysfs resource files).

The GSP boot sequence is pure register poking + structure population in
coherent memory.  There is nothing in the 1245 lines of lithos_gsp.c that
requires kernel mode.  The FSP communication (the one remaining hard part)
is also pure MMIO register writes to BAR0.

The only kernel involvement is the vfio-pci module (ships with Linux,
zero custom code) or sysfs PCI resource file exposure (also stock Linux).
After the BARs are mapped, the entire GPU driver -- from PMC identity
check through GSP boot through channel creation through kernel launch --
is loads and stores to memory-mapped addresses.  This can be expressed
in Forth, ARM64 assembly, or any language that can issue the mmap
syscall and perform 32/64-bit aligned memory accesses.

**No interrupts needed.**  Lithos uses cooperative megakernels with
polling completion.  No preemption, no scheduling, no multi-tenancy.

**No DMA API needed.**  BAR4 IS coherent memory.  PA == GPU VA.
CPU stores are GPU-visible without flushes.

**No custom kernel code needed.**  vfio-pci or sysfs resource files
provide the BAR mappings.  Everything else is userspace.
