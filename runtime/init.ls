\\ init.ls — replaces CUDA init with vfio-pci + BAR mmap.
\\
\\ Replaces cuInit + cuDeviceGet + cuCtxCreate + cuCtxSetCurrent.
\\ Opens PCI sysfs resource files (no vfio group complexity for bring-up),
\\ mmaps BAR0 (16 MB GPU registers) and BAR4 (128 GB coherent HBM via
\\ NVLink-C2C identity mapping), and returns the two base virtual addresses
\\ subsequent runtime operations use.
\\
\\ On GH200:
\\   BAR0 = 16 MB, uncacheable MMIO window for GPU registers
\\   BAR4 = 128 GB, coherent HBM. CPU PA == GPU VA. No dma_alloc_coherent.
\\
\\ Prerequisites:
\\   nvidia.ko unbound from the device
\\   root (or udev ACL) on /sys/bus/pci/devices/.../resource{0,4}
\\   PRIV_LOCKDOWN must be released by gsp_boot before GPFIFO work.
\\
\\ Syscall numbers (aarch64):
\\   openat = 56
\\   mmap   = 222
\\
\\ Constants:
\\   AT_FDCWD    = -100
\\   O_RDONLY    = 0
\\   O_RDWR      = 2
\\   PROT_READ   = 1
\\   PROT_WRITE  = 2
\\   MAP_SHARED  = 1
\\   BAR0 size   = 0x1000000      \\ 16 MB
\\   BAR4 size   = 0x2000000000   \\ 128 GB

\\ ------------------------------------------------------------
\\ Raw syscall wrappers
\\ ------------------------------------------------------------

\\ openat(AT_FDCWD, path, flags, mode=0) → fd
openat fd path flags :
    ↓ $8 56
    ↓ $0 fd
    ↓ $1 path
    ↓ $2 flags
    ↓ $3 0
    trap
    ↑ $0

\\ mmap(addr, len, prot, flags, fd, offset) → va
mmap addr len prot flags fd offset :
    ↓ $8 222
    ↓ $0 addr
    ↓ $1 len
    ↓ $2 prot
    ↓ $3 flags
    ↓ $4 fd
    ↓ $5 offset
    trap
    ↑ $0

\\ ------------------------------------------------------------
\\ PCI device open via sysfs
\\
\\ bdf_path is a pointer to a NUL-terminated string of the form
\\   "/sys/bus/pci/devices/0000:01:00.0/resource0"
\\ The BDF and resource index are baked into the string by the caller.
\\ ------------------------------------------------------------
pci_device_open bdf_path :
    openat -100 bdf_path 2

\\ ------------------------------------------------------------
\\ vfio_open
\\
\\ For now this uses the sysfs resource file path rather than
\\ /dev/vfio/vfio; the "fd" returned is a sysfs resource0 fd that
\\ can be mmap'd for BAR0 directly. bar4_map opens its own fd for
\\ resource4. This sidesteps VFIO group / noiommu mode setup during
\\ bring-up; swap to /dev/vfio/vfio + VFIO_DEVICE_GET_REGION_INFO
\\ for production.
\\ ------------------------------------------------------------
vfio_open :
    pci_device_open $BAR0_SYSFS_PATH

\\ ------------------------------------------------------------
\\ bar0_map — mmap the 16 MB register window
\\
\\ vfio_fd: open fd on .../resource0
\\ Returns the virtual address the kernel chose for the mapping.
\\ prot = PROT_READ|PROT_WRITE = 3, flags = MAP_SHARED = 1.
\\ ------------------------------------------------------------
bar0_map vfio_fd :
    mmap 0 0x1000000 3 1 vfio_fd 0

\\ ------------------------------------------------------------
\\ bar4_map — mmap the 128 GB coherent HBM window
\\
\\ vfio_fd here is a fresh fd on .../resource4.
\\ Identity-mapped: CPU PA == GPU VA on GH200 (NVLink-C2C + ATS).
\\ ------------------------------------------------------------
bar4_map vfio_fd :
    mmap 0 0x2000000000 3 1 vfio_fd 0

\\ ------------------------------------------------------------
\\ pmc_sanity_check — verify the GPU is alive before touching anything else
\\
\\ PMC_BOOT_0 at BAR0+0x000000 holds the architecture / implementation
\\ magic. On Hopper GH200 the upper nibble [23:20] reads as 0xa. A zero
\\ read means the BAR map did not land on live silicon (wrong BDF,
\\ device powered down, or nvidia.ko still bound).
\\
\\ Also clears PMC_INTR_0 at BAR0+0x100 to mask any pending interrupts
\\ left over from a previous owner of the device.
\\ ------------------------------------------------------------
pmc_sanity_check bar0 :
    boot0 → 32 bar0 + 0x0
    ← 32 bar0 + 0x100 0

\\ ------------------------------------------------------------
\\ gsp_boot — full GSP RISC-V coprocessor bootstrap
\\
\\ Delegates to the 9-stage sequence already implemented in GSP/*.s:
\\   1. bar_map_init (already done here — reuses bar0/bar4)
\\   2. pmc_check
\\   3. falcon_reset
\\   4. hbm_alloc_init
\\   5. gsp_fw_load
\\   6. gsp_bcr_start
\\   7. FSP communication (TODO)
\\   8. gsp_poll_lockdown
\\   9. gsp_rpc_alloc_channel
\\
\\ After this returns the GPU is out of PRIV_LOCKDOWN and GPFIFO
\\ submission is legal.
\\ ------------------------------------------------------------
gsp_boot bar0 bar4 :
    pmc_sanity_check bar0
    $GSP_BOOT_SEQUENCE bar0 bar4

\\ ------------------------------------------------------------
\\ runtime_init — top-level. Replaces cuInit/cuDeviceGet/cuCtxCreate.
\\
\\ Steps:
\\   1. Open sysfs resource0, mmap BAR0
\\   2. Open sysfs resource4, mmap BAR4
\\   3. Sanity-check PMC_BOOT_0
\\   4. Boot GSP, release PRIV_LOCKDOWN
\\
\\ Returns: bar0 (register window va), bar4 (HBM window va).
\\ Both are passed to every subsequent runtime composition.
\\ ------------------------------------------------------------
runtime_init :
    fd0 pci_device_open $BAR0_SYSFS_PATH
    bar0 bar0_map fd0

    fd4 pci_device_open $BAR4_SYSFS_PATH
    bar4 bar4_map fd4

    pmc_sanity_check bar0
    gsp_boot bar0 bar4

    ↑ bar0
    ↑ bar4
