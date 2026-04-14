# NVIDIA driver ABI reference (from open-gpu-kernel-modules)

Source: open-gpu-kernel-modules commit `db0c4e65c8e34c678d745ddb1317f53f90d1072b` (HEAD of `main`, shallow clone, cloned 2026-04-13).
Target: Hopper (GH100 / sm_90a, GH200 superchip) on Linux ARM64 (aarch64). The
GSP-RM-proprietary firmware runs inside the GPU; the open-source kernel module
talks to it via the RPC path, but ioctl shape seen from userspace is identical
to the proprietary module.

All paths below are relative to the repo root of `open-gpu-kernel-modules`.

---

## 1. Character devices and how ioctls are encoded

Defined in `kernel-open/common/inc/nv-chardev-numbers.h`:

```
#define NV_MAJOR_DEVICE_NUMBER                195
#define NV_MINOR_DEVICE_NUMBER_REGULAR_MAX    247
#define NV_MINOR_DEVICE_NUMBER_CONTROL_DEVICE 255
```

Userspace opens these nodes in this order (standard CUDA init):

| Path | Minor | Purpose |
|---|---|---|
| `/dev/nvidiactl` | 255 | Global control FD: version check, attach set, GPU query. |
| `/dev/nvidia-uvm` | misc dev (allocated by `nvidia-uvm` miscdevice) | Unified memory (ATS, HMM, pageable memory, range groups). |
| `/dev/nvidia-uvm-tools` | misc dev | Profiler-only, not needed for inference. |
| `/dev/nvidiaN` | 0+N | Per-GPU FD: allocations, kernel launch, mmap of BAR regions. |

`nv_major` and miscdev registration happen in `kernel-open/nvidia/nv.c`; UVM
registers itself as a miscdevice in `kernel-open/nvidia-uvm/uvm.c`.

### Ioctl number encoding (RM path, `/dev/nvidiaN`, `/dev/nvidiactl`)

`kernel-open/common/inc/nv-ioctl-numbers.h`:

```
#define NV_IOCTL_MAGIC      'F'     /* 0x46 */
#define NV_IOCTL_BASE       200
#define NV_ESC_CARD_INFO            (NV_IOCTL_BASE + 0)    /* 200 / 0xC8 */
#define NV_ESC_REGISTER_FD          (NV_IOCTL_BASE + 1)    /* 201 / 0xC9 */
#define NV_ESC_ALLOC_OS_EVENT       (NV_IOCTL_BASE + 6)    /* 206 */
#define NV_ESC_FREE_OS_EVENT        (NV_IOCTL_BASE + 7)
#define NV_ESC_STATUS_CODE          (NV_IOCTL_BASE + 9)
#define NV_ESC_CHECK_VERSION_STR    (NV_IOCTL_BASE + 10)   /* 210 / 0xD2 */
#define NV_ESC_IOCTL_XFER_CMD       (NV_IOCTL_BASE + 11)   /* 211 / 0xD3 */
#define NV_ESC_ATTACH_GPUS_TO_FD    (NV_IOCTL_BASE + 12)   /* 212 / 0xD4 */
#define NV_ESC_QUERY_DEVICE_INTR    (NV_IOCTL_BASE + 13)
#define NV_ESC_SYS_PARAMS           (NV_IOCTL_BASE + 14)
#define NV_ESC_EXPORT_TO_DMABUF_FD  (NV_IOCTL_BASE + 17)
#define NV_ESC_WAIT_OPEN_COMPLETE   (NV_IOCTL_BASE + 18)
```

Per `src/nvidia/arch/nvalloc/unix/include/nv_escape.h`:

```
#define NV_ESC_RM_ALLOC_MEMORY           0x27
#define NV_ESC_RM_ALLOC_OBJECT           0x28
#define NV_ESC_RM_FREE                   0x29
#define NV_ESC_RM_CONTROL                0x2A
#define NV_ESC_RM_ALLOC                  0x2B
#define NV_ESC_RM_DUP_OBJECT             0x34
#define NV_ESC_RM_MAP_MEMORY             0x4E
#define NV_ESC_RM_UNMAP_MEMORY           0x4F
#define NV_ESC_RM_MAP_MEMORY_DMA         0x57
#define NV_ESC_RM_UNMAP_MEMORY_DMA       0x58
#define NV_ESC_RM_IDLE_CHANNELS          0x41
#define NV_ESC_RM_VID_HEAP_CONTROL       0x4A
```

The full ioctl number given to `ioctl(2)` is the Linux `_IOC` encoding:

```
cmd = _IOC(dir, type, nr, size)
    = (dir << 30) | (type << 8) | nr | (size << 16)
```

where `type = 'F' = 0x46`, `nr` is one of the constants above, `size` is
`sizeof(struct)` of the corresponding params (max 2^14-1 = 16383). `dir` in
the kernel is decoded by `_IOC_NR(cmd)` / `_IOC_SIZE(cmd)` only — the
driver ignores the direction bits (see `kernel-open/nvidia/nv.c:2410-2445`
`validate_ioctl()`: it reads only `_IOC_NR(cmd)` and `_IOC_SIZE(cmd)`).

Because `nr` is only 8 bits but `size` is 14 bits, structs larger than
~16 KB must be passed indirectly through the `NV_ESC_IOCTL_XFER_CMD` wrapper:

```c
/* src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h:49-54 */
typedef struct nv_ioctl_xfer {
    NvU32   cmd;                        /* real NV_ESC_* number  */
    NvU32   size;                       /* real struct size      */
    NvP64   ptr  NV_ALIGN_BYTES(8);     /* pointer to real args  */
} nv_ioctl_xfer_t;
```

Practically, most CUDA-relevant ioctls (RM_ALLOC, RM_CONTROL) always go
through `NV_ESC_IOCTL_XFER_CMD` because the `cmd`-specific params can be
large. The shape of the call then is:

```
ioctl(fd, _IOC(_IOC_NONE, 'F', NV_ESC_IOCTL_XFER_CMD, sizeof(nv_ioctl_xfer_t)),
      &xfer)
```

### UVM ioctl encoding (`/dev/nvidia-uvm`)

From `kernel-open/nvidia-uvm/uvm_ioctl.h:38-46`:

```
#   define UVM_IOCTL_BASE(i) i      /* Linux */
#define UVM_RESERVE_VA          UVM_IOCTL_BASE(1)
...
```

So the UVM cmd IS the raw integer (no `_IOC` wrapping). The dispatcher in
`kernel-open/nvidia-uvm/uvm.c` switches on the raw cmd. Two special exceptions
with different numbering for backwards-compat (`kernel-open/nvidia-uvm/uvm_linux_ioctl.h`):

```
#define UVM_INITIALIZE    0x30000001    /* MUST be first ioctl on FD    */
#define UVM_DEINITIALIZE  0x30000002
```

---

## 2. Per-operation ABI

### 2.1 cuInit-equivalent sequence

The exact sequence the proprietary libcuda performs is not in open-gpu-kernel-modules
(the driver side is; the userspace side is proprietary). But every path that
allocates any GPU object requires, in order:

1. `openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR | O_CLOEXEC)` → `fd_ctl`.

2. **Version check.** `NV_ESC_CHECK_VERSION_STR`, struct at
   `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h:100-105`:

   ```c
   #define NV_RM_API_VERSION_STRING_LENGTH 64
   typedef struct nv_ioctl_rm_api_version {
       NvU32 cmd;                                          /* NV_RM_API_VERSION_CMD_* */
       NvU32 reply;                                        /* _RECOGNIZED / _UNRECOGNIZED */
       char  versionString[NV_RM_API_VERSION_STRING_LENGTH];
   } nv_ioctl_rm_api_version_t;
   ```

   `cmd` = `'2'` (`NV_RM_API_VERSION_CMD_QUERY`) to read the driver's version,
   or `'1'` to accept driver's version (relaxed), or `0` to require an exact
   match. The string format must match the running kernel module's version
   (e.g. `"545.23.08"`). Mismatch → all subsequent RM ioctls return
   `NV_ERR_LIB_VERSION_MISMATCH`.

3. **Attach GPUs.** `NV_ESC_ATTACH_GPUS_TO_FD` with an array of `NvU32`
   gpu_ids. `isArgumentArray=NV_TRUE` in the table at
   `kernel-open/nvidia/nv.c:2425`. Enumerate first via `NV_ESC_CARD_INFO`:

   ```c
   /* src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h:56-68 */
   typedef struct nv_ioctl_card_info {
       NvBool        valid;
       nv_pci_info_t pci_info;
       NvU32         gpu_id;
       NvU16         interrupt_line;
       NvU64         reg_address;      /* BAR0 phys            */
       NvU64         reg_size;         /* BAR0 size (16 MiB)   */
       NvU64         fb_address;       /* BAR1 phys (VRAM)     */
       NvU64         fb_size;          /* BAR1 size            */
       NvU32         minor_number;     /* for /dev/nvidia%d    */
       NvU8          dev_name[10];
   } nv_ioctl_card_info_t;
   ```

   `NV_ESC_CARD_INFO` is called with an array of this struct; kernel fills
   valid=true for each attached GPU (`kernel-open/nvidia/nv.c:2358`
   `nvidia_read_card_info`).

4. **Open per-GPU FD.** `openat("/dev/nvidiaN", O_RDWR | O_CLOEXEC)` → `fd_dev`
   using `minor_number` from step 3.

5. **Register FD.** `NV_ESC_REGISTER_FD` with
   `nv_ioctl_register_fd_t { int ctl_fd; }`
   (`nv-ioctl.h:126-129`) — passes `fd_ctl` into `fd_dev` so they share an RM
   client.

6. **Allocate root client.** `NV_ESC_RM_ALLOC` with hClass=`NV01_ROOT_CLIENT`
   (= `0x41`, from `nvos.h:189`). Wrapper is `NVOS21_PARAMETERS` or
   `NVOS64_PARAMETERS` (`nvos.h:467-493`):

   ```c
   typedef struct {
       NvHandle hRoot;                          /* 0 for root-of-roots  */
       NvHandle hObjectParent;                  /* 0                    */
       NvHandle hObjectNew;                     /* [INOUT] 0 → autogen  */
       NvV32    hClass;                         /* NV01_ROOT_CLIENT     */
       NvP64    pAllocParms NV_ALIGN_BYTES(8);  /* NULL for root        */
       NvP64    pRightsRequested;               /* NULL                 */
       NvU32    paramsSize;                     /* 0                    */
       NvU32    flags;                          /* NVOS64_FLAGS_NONE    */
       NvV32    status;                         /* [OUT]                */
   } NVOS64_PARAMETERS;
   ```

   Output `hObjectNew` is the client handle used as `hRoot` for everything
   afterwards.

7. **Query attached GPUs.** `NV_ESC_RM_CONTROL` with `NVOS54_PARAMETERS`
   (`nvos.h:2232-2242`):

   ```c
   typedef struct {
       NvHandle hClient;        /* client handle from step 6     */
       NvHandle hObject;        /* object being controlled       */
       NvV32    cmd;            /* NV0000_CTRL_CMD_GPU_*         */
       NvU32    flags;
       NvP64    params;
       NvU32    paramsSize;
       NvV32    status;
   } NVOS54_PARAMETERS;
   ```

   Useful commands (from `src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h`):

   - `NV0000_CTRL_CMD_GPU_GET_PROBED_IDS` (0x214)
   - `NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS` (0x201) — `NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS { NvU32 gpuIds[32]; }`
   - `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2` (0x205) — returns subDeviceInstance, gpuInstance, UUID
   - `NV0000_CTRL_CMD_GPU_GET_DEVICE_IDS` (0x204)

8. **Allocate NV01_DEVICE_0 and NV20_SUBDEVICE_0.** Two further
   `NV_ESC_RM_ALLOC` calls:

   - hClass `NV01_DEVICE_0` (0x80, `cl0080.h:36`), params =
     `NV0080_ALLOC_PARAMETERS` (`cl0080.h:54-64`):

     ```c
     typedef struct NV0080_ALLOC_PARAMETERS {
         NvU32    deviceId;             /* the gpu_id/instance        */
         NvHandle hClientShare;         /* 0 or a client to share VA  */
         NvHandle hTargetClient;        /* 0                          */
         NvHandle hTargetDevice;        /* 0                          */
         NvV32    flags;
         NvU64    vaSpaceSize;          /* 0 = default                */
         NvU64    vaStartInternal;
         NvU64    vaLimitInternal;
         NvV32    vaMode;               /* VAMODE_SINGLE_VASPACE etc. */
     } NV0080_ALLOC_PARAMETERS;
     ```

   - hClass `NV20_SUBDEVICE_0` (0x2080, `cl2080.h:36`), params
     `NV2080_ALLOC_PARAMETERS { NvU32 subDeviceId; }`.

9. **Wait open complete** (optional but proprietary CUDA does it):
   `NV_ESC_WAIT_OPEN_COMPLETE` with
   `nv_ioctl_wait_open_complete_t { int rc; NvU32 adapterStatus; }`
   (`nv-ioctl.h:152-156`). Blocks until GSP-RM finished boot.

**Summary files touched for section 2.1:** `nv-ioctl.h`, `nv-ioctl-numbers.h`,
`nv_escape.h`, `nvos.h`, `cl0080.h`, `cl2080.h`, `ctrl0000gpu.h`, `nv.c:2358,
2410-2470`.

### 2.2 Memory mapping for unified memory (GH200)

On GH200, HBM is exposed to the CPU through the NVLink-C2C chip-to-chip
coherent link. The driver surfaces this as a NUMA node; on the GPU side,
host pageable memory is accessed through ATS. Both use the **UVM** character
device, not `/dev/nvidiaN`.

Sequence (from `kernel-open/nvidia-uvm/uvm_ioctl.h`):

1. `openat("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC)` → `fd_uvm`.

2. `ioctl(fd_uvm, UVM_INITIALIZE, UVM_INITIALIZE_PARAMS)`
   (`uvm_linux_ioctl.h:32-38`):

   ```c
   typedef struct {
       NvU64     flags     NV_ALIGN_BYTES(8); /* IN  */
       NV_STATUS rmStatus;                    /* OUT */
   } UVM_INITIALIZE_PARAMS;
   ```

   This MUST be the first ioctl on the FD (the comment at line 28-31 says so).

3. `UVM_PAGEABLE_MEM_ACCESS` (number 39, `uvm_ioctl.h:458-464`) — reads the
   `pageableMemAccess` bool; on GH200 this returns NvTrue.

4. `UVM_REGISTER_GPU` (37, `uvm_ioctl.h:434-445`):

   ```c
   typedef struct {
       NvProcessorUuid gpu_uuid;    /* IN  */
       NvBool          numaEnabled; /* OUT - true on GH200  */
       NvS32           numaNodeId;  /* OUT - NUMA node of HBM */
       NvS32           rmCtrlFd;    /* IN  - fd of /dev/nvidiactl */
       NvHandle        hClient;     /* IN  - root client handle   */
       NvHandle        hSmcPartRef; /* IN  - 0 for non-MIG        */
       NV_STATUS       rmStatus;    /* OUT */
   } UVM_REGISTER_GPU_PARAMS;
   ```

   `rmCtrlFd` + `hClient` connect the UVM FD to the already-initialized RM
   session. UVM then reads the HBM NUMA node via the kernel's numa maps.

5. `UVM_REGISTER_GPU_VASPACE` (25): binds the VA space created under NV01_DEVICE_0
   to the UVM driver so UVM can install GPU PTEs for migratable/pageable ranges.

6. **For each managed range** (cudaMallocManaged / cudaMallocAsync):
   - Caller first does anonymous `mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)`.
     On GH200 with HMM, this is already GPU-accessible; no extra UVM call needed
     unless explicit residency hints are required.
   - `UVM_CREATE_RANGE_GROUP` (23) if you need to batch-control preferred location.
   - `UVM_SET_PREFERRED_LOCATION` (42, `uvm_ioctl.h:512-521`):
     ```c
     NvU64           requestedBase;
     NvU64           length;
     NvProcessorUuid preferredLocation;    /* GPU UUID or special "CPU" */
     NvS32           preferredCpuNumaNode;
     ```
   - `UVM_MIGRATE` (51, `uvm_ioctl.h:674-687`) to force-move pages now.

7. To expose a **GPU-allocated buffer** (from NV_ESC_RM_ALLOC of
   `NV01_MEMORY_LOCAL_USER` / hClass `0x003e`) to the UVM-managed VA space,
   use `UVM_MAP_EXTERNAL_ALLOCATION` (33, `uvm_ioctl.h:394-407`):

   ```c
   typedef struct {
       NvU64                   base;               /* CPU virtual address returned by mmap of /dev/nvidiaN */
       NvU64                   length;
       NvU64                   offset;             /* into the underlying GPU mem */
       UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];
       NvU64                   gpuAttributesCount;
       NvS32                   rmCtrlFd;           /* /dev/nvidiactl                */
       NvU32                   hClient;
       NvU32                   hMemory;            /* handle returned by RM_ALLOC   */
       NV_STATUS               rmStatus;
   } UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;
   ```

**For plain CPU-side mmap of HBM (the BAR1 path, what cuMemHostAlloc /
cuMemAllocHost uses):** skip UVM entirely and use the RM path:

1. `NV_ESC_RM_ALLOC` with hClass `NV01_MEMORY_SYSTEM` (0x0e) or
   `NV01_MEMORY_LOCAL_USER` (0x003e).
2. `NV_ESC_RM_MAP_MEMORY` (0x4E) using `nv_ioctl_nvos33_parameters_with_fd`
   wrapper (`nv-unix-nvos-params-wrappers.h:39-46`), which embeds
   `NVOS33_PARAMETERS` + an `int fd`:
   ```c
   typedef struct {
       NvHandle hClient;
       NvHandle hDevice;
       NvHandle hMemory;
       NvU64    offset;
       NvU64    length;
       NvP64    pLinearAddress;   /* [OUT]  */
       NvU32    status;
       NvU32    flags;             /* NVOS33_FLAGS_*  */
   } NVOS33_PARAMETERS;              /* nvos.h:1848-1858 */
   ```
3. Userspace then `mmap(NULL, length, PROT_..., MAP_SHARED, fd_dev, pLinearAddress)`.
   The `pLinearAddress` returned by the kernel is the offset you pass to
   `mmap` — it is a cookie, not a real address. See
   `kernel-open/nvidia/nv-mmap.c` (not quoted here).

### 2.3 Module load (cubin)

There is no kernel-side "load module" ioctl. The proprietary libcuda does:

1. Parse the ELF/cubin in userspace (extract `.text.<kernel>`, `.nv.info`,
   constant banks `.nv.constant0.<kernel>`, shared-size, local-size,
   reg count, QMD template bytes, relocation info).

2. Allocate GPU memory for instruction storage (`iSize`) via
   `NV_ESC_RM_ALLOC` with hClass `NV01_MEMORY_LOCAL_USER` (0x3e). For
   Hopper this must be placed in video memory (HBM) at
   `NVOS32_ATTR_LOCATION_VIDMEM`. The alloc param struct is defined at
   `nvos.h:884` (`NVOS32_PARAMETERS`, function `NVOS32_FUNCTION_ALLOC_SIZE`)
   — very large; see `nvos.h:667-884`.

3. Allocate VA space reservation via `NV_ESC_RM_VID_HEAP_CONTROL` /
   `NVOS32_PARAMETERS` with `function=NVOS32_FUNCTION_ALLOC_SIZE` and the
   `VIRTUAL` attribute set.

4. Bind the backing allocation into the VA space via `NV_ESC_RM_MAP_MEMORY_DMA`
   using `NVOS46_PARAMETERS` (`nvos.h:2170-2187`):

   ```c
   typedef struct {
       NvHandle hClient, hDevice, hDma, hMemory;
       NvU64    offset, length;
       NvV32    flags, flags2;
       NvV32    kindOverride;
       NvU64    dmaOffset;   /* [INOUT] GPU virtual address */
       NvV32    status;
   } NVOS46_PARAMETERS;
   ```

   Use `NVOS46_FLAGS_PAGE_SIZE_HUGE` (2 MB) or `_512M` for code allocations
   on Hopper.

5. Copy the cubin bytes: either
   - Map the backing memory to CPU via `NV_ESC_RM_MAP_MEMORY` +
     `mmap(fd_dev)` and `memcpy` directly (works for HBM on GH200 because
     coherent C2C makes HBM host-cached), OR
   - Use a CE (copy-engine) channel to DMA from a staging buffer.

6. For each constant bank, repeat the allocate-map-copy pattern.

7. The kernel "entry PC" for a launch is `base_va + func_offset`, where
   `func_offset` is the offset of `.text.<name>` inside the cubin.

**Gap:** the QMD (Queue Memory Descriptor) template encoded in the cubin's
`.nv.info` section — the field layout in bytes — is not in the open-source
repo for Hopper. `cla0c0qmd.h` (Kepler) is the only QMD header present;
Hopper QMD fields are proprietary. See Unknowns.

### 2.4 Kernel launch (Hopper compute)

Class IDs (`src/common/sdk/nvidia/inc/class/`):

| Class ID | Name | Source |
|---|---|---|
| `0xC86F` | `HOPPER_CHANNEL_GPFIFO_A` | `clc86f.h:27` |
| `0xCBC0` | `HOPPER_COMPUTE_A` | `clcbc0.h:26` |
| `0xC661` | `HOPPER_USERMODE_A` | `clc661.h:26` |
| `0xC96F` | `BLACKWELL_CHANNEL_GPFIFO_A` | `clc96f.h` |
| `0xCEC0` | `BLACKWELL_COMPUTE_B` | `clcec0.h` |

Setup sequence (Hopper), derived from `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c:5541-5700`
which is effectively the open-source equivalent of the libcuda launch path:

1. **Allocate HOPPER_USERMODE_A** via `NV_ESC_RM_ALLOC`, parent = subdevice,
   params = `NV_HOPPER_USERMODE_A_PARAMS` (`nvos.h:3320-3328`):
   ```c
   typedef struct {
       NvBool bBar1Mapping;   /* IN: NV_TRUE for BAR1 mapping        */
       NvBool bPriv;          /* IN: NV_FALSE for normal user access */
   } NV_HOPPER_USERMODE_A_PARAMS;
   ```
   Then `NV_ESC_RM_MAP_MEMORY` + `mmap` for `NVC361_NV_USERMODE__SIZE = 65536`
   bytes (`clc361.h:30`). The doorbell is at offset
   `NVC361_NOTIFY_CHANNEL_PENDING = 0x90` (`clc361.h:33`).

2. **Allocate a VA space** — hClass `FERMI_VASPACE_A` (`0x90F1`) if you
   need a private one, else use the implicit one from NV01_DEVICE_0.

3. **Allocate UserD / GPFIFO backing memory** (system or video; HBM on GH200)
   via `NV_ESC_RM_ALLOC` with `NV01_MEMORY_SYSTEM` or `NV01_MEMORY_LOCAL_USER`.
   GPFIFO buffer size = `gpFifoEntries * 8 bytes` (entry format below).

4. **Allocate the channel object**: `NV_ESC_RM_ALLOC` with
   hClass `HOPPER_CHANNEL_GPFIFO_A` (0xC86F), params =
   `NV_CHANNEL_ALLOC_PARAMS` (`src/common/sdk/nvidia/inc/alloc/alloc_channel.h:296-342`):

   ```c
   typedef struct NV_CHANNEL_ALLOC_PARAMS {
       NvHandle hObjectError;       /* error notifier memory handle */
       NvHandle hObjectBuffer;      /* unused                        */
       NvU64    gpFifoOffset;       /* GPU VA of GPFIFO ring         */
       NvU32    gpFifoEntries;      /* power-of-two count            */
       NvU32    flags;
       NvHandle hContextShare;
       NvHandle hVASpace;
       NvHandle hUserdMemory[NV_MAX_SUBDEVICES];  /* NV_MAX_SUBDEVICES=8 */
       NvU64    userdOffset[NV_MAX_SUBDEVICES];
       NvU32    engineType;         /* NV2080_ENGINE_TYPE_GR0 = 1    */
       NvU32    cid;
       NvU32    subDeviceId;        /* 1 for single-GPU              */
       NvHandle hObjectEccError;
       NV_MEMORY_DESC_PARAMS instanceMem, userdMem, ramfcMem, mthdbufMem;
       /* ... reserved fields for Confidential Compute ... */
   } NV_CHANNEL_ALLOC_PARAMS;
   ```

5. **Allocate the compute object** on the channel: `NV_ESC_RM_ALLOC`,
   parent = channel handle, hClass = `HOPPER_COMPUTE_A` (0xCBC0), paramsSize=0.
   This binds the compute engine methods to subchannel (conventionally 1).

6. **Call `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE`** (0xA06F0103, shared with
   Kepler base class). Params at
   `ctrl/ctrla06f/ctrla06fgpfifo.h:69-73`:

   ```c
   typedef struct NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS {
       NvBool bEnable;       /* NV_TRUE to put channel on a runlist */
       NvBool bSkipSubmit;
       NvBool bSkipEnable;
   } NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;
   ```

7. **Get work-submit token** via RM_CONTROL cmd
   `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` (`0xc36f0108`, `ctrlc36f.h:79-85`):

   ```c
   typedef struct NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS {
       NvU32 workSubmitToken;
   } ...;
   ```

   The token identifies this channel to the Host scheduler.

8. **USERD control structure** (mapped into user VA):
   `Nvc86fControl` at `clc86f.h:29-45`:

   ```c
   typedef volatile struct Nvc86fControl_struct {
       NvU32 Ignored00[0x010];     /* 0x0000-0x003f */
       NvU32 Put;                  /* 0x0040       */
       NvU32 Get;                  /* 0x0044 ro    */
       NvU32 Reference;            /* 0x0048 ro    */
       NvU32 PutHi;                /* 0x004c       */
       NvU32 Ignored01[2];         /* 0x0050-0x0057 */
       NvU32 TopLevelGet;          /* 0x0058 ro    */
       NvU32 TopLevelGetHi;        /* 0x005c ro    */
       NvU32 GetHi;                /* 0x0060 ro    */
       NvU32 Ignored02[7];
       NvU32 Ignored03;            /* 0x0080       */
       NvU32 Ignored04[1];
       NvU32 GPGet;                /* 0x0088 ro    */
       NvU32 GPPut;                /* 0x008c       */
       NvU32 Ignored05[0x5c];
   } Nvc86fControl, HopperAControlGPFifo;
   ```

   Userspace writes `GPPut` (0x8c) after it has written a new GPFIFO entry.

9. **GPFIFO entry format** (`clc86f.h:168-186`), 8 bytes:

   ```
   entry[0]:
     [31:2]   GET   — low 30 bits of PB GPU VA shifted right 2
     [0]      FETCH — 0 unconditional, 1 conditional
   entry[1]:
     [7:0]    GET_HI  (bits [39:32] of PB VA)
     [9]      LEVEL   (0=main, 1=subroutine)
     [30:10]  LENGTH  — number of 4-byte methods in pushbuffer slice
     [31]     SYNC    (0=proceed, 1=wait)
     [7:0]    OPCODE  — overlap w/ GET_HI; 0=NOP / 1=ILLEGAL
   ```

10. **Pushbuffer** (referenced by a GPFIFO entry) contains methods encoded
    as:

    ```
    method header (1 dword):
      [12:0]  method address >> 2 (e.g. NVC6C0_SET_INLINE_QMD_LO = 0x02cc >> 2)
      [15:13] subchannel (the one bound to HOPPER_COMPUTE_A, typically 1)
      [28:16] count of data dwords
      [31:29] opcode: INC_METHOD=1, NON_INC=3, ONE_INC=5, IMM_DATA=4
    data dwords (count of them)
    ```

    (Method encoding is documented by convention in `clc86f.h` / the
    `GP_ENTRY` fields; the three-word method header layout itself is NOT
    in the open source — it's derived from `cla16f.h` / the fermi manual.
    See Unknowns.)

11. **Launch** a compute kernel: write methods onto the pushbuffer that
    perform `SET_INLINE_QMD` to hand the Host a filled QMD structure
    (Hopper compute methods inherited from `clc6c0.h:357-370`):

    ```
    NVC6C0_SEND_PCAS_A                0x02b4   QMD address >> 8
    NVC6C0_SEND_PCAS_B                0x02b8   from/delta for PCAS batch
    NVC6C0_SEND_SIGNALING_PCAS_B      0x02bc   invalidate / schedule bits
    NVC6C0_SEND_SIGNALING_PCAS2_B     0x02c0   PCAS action
    ```

    The QMD itself is ~256–384 bytes of packed bitfields (grid dims,
    block dims, shared mem, register count, constant bank 0 address,
    entry PC, barriers, texture state, etc.). For Hopper, the specific
    field layout is **NOT in the open-source tree**; only Kepler's
    (`cla0c0qmd.h`) is. See Unknowns.

12. **Ring the doorbell**: write the `workSubmitToken` (from step 7) to
    offset `0x90` of the USERMODE region (the mmaped HOPPER_USERMODE_A):

    ```c
    /* Per src/nvidia/src/kernel/rmapi/nv_gpu_ops.c:5631 */
    channel->workSubmissionOffset =
        (NvU32 *)((NvU8*)clientRegionMapping + NVC361_NOTIFY_CHANNEL_PENDING);
    *channel->workSubmissionOffset = channel->workSubmissionToken;
    ```

    That write goes out on the NVLink-C2C / BAR0 path directly to the Host
    engine, which then picks up new GPFIFO entries using `GPPut`.

### 2.5 Fence / sync

Two independent sync mechanisms:

**(a) Inline semaphore release via channel methods** (`clc86f.h:142-162`):

```
NVC86F_SEM_ADDR_LO        0x005c  bits [31:2] of VA
NVC86F_SEM_ADDR_HI        0x0060  bits [56:32] of VA
NVC86F_SEM_PAYLOAD_LO     0x0064
NVC86F_SEM_PAYLOAD_HI     0x0068  (for 64-bit semaphores)
NVC86F_SEM_EXECUTE        0x006c
  [2:0]   OPERATION: 0=ACQUIRE, 1=RELEASE, 3=ACQ_CIRC_GEQ
  [12]    ACQUIRE_SWITCH_TSG: 0=DIS, 1=EN
  [20]    RELEASE_WFI: 0=DIS, 1=EN
  [24]    PAYLOAD_SIZE: 0=32BIT (64-bit requires extra bit)
  [25]    RELEASE_TIMESTAMP: 0=DIS, 1=EN
```

Post a `SEM_RELEASE` as the last thing in the pushbuffer slice that runs a
kernel. CPU polls the VA. For GH200, the VA can point into HBM (cached on
CPU over C2C) or into host memory pinned via `NV01_MEMORY_SYSTEM`; the CPU
reads converge through coherence.

**(b) Error/notifier interrupt via `NV_ESC_ALLOC_OS_EVENT`** (not typically
used for fast-path sync — it's for async RC / fault events):

```c
/* src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h:71-77 */
typedef struct nv_ioctl_alloc_os_event {
    NvHandle hClient;
    NvHandle hDevice;
    NvU32    fd;        /* eventfd-style: poll this fd */
    NvU32    Status;
} nv_ioctl_alloc_os_event_t;
```

Combined with `NV_ESC_RM_ALLOC` of hClass `NV01_EVENT_OS_EVENT` (0x79,
`nvos.h:393`) to wire an RM event to a userland `eventfd`. For kernel
launch completion you use (a); for catastrophic errors you use (b).

**(c) GP_GET / GP_PUT self-check**: the USERD region is readable via mmap,
so the CPU can read `GPGet` (offset 0x88, `clc86f.h:42`) and wait for it
to catch up to a known `GPPut`. But this only guarantees that the Host has
*dispatched* the work, not that the compute engine has finished it —
use (a) for compute completion.

---

## 3. Minimum ABI surface

| ioctl | fd | struct (sizeof on aarch64) | purpose |
|---|---|---|---|
| `NV_ESC_CHECK_VERSION_STR` (0xD2, 'F') | `/dev/nvidiactl` | `nv_ioctl_rm_api_version_t` (72 B) | Handshake, no-ops otherwise |
| `NV_ESC_CARD_INFO` (0xC8, 'F') | `/dev/nvidiactl` | array of `nv_ioctl_card_info_t` (56 B each) | Enumerate GPUs |
| `NV_ESC_ATTACH_GPUS_TO_FD` (0xD4, 'F') | `/dev/nvidiactl` | array of `NvU32` | Pin set of GPUs to this FD |
| `NV_ESC_REGISTER_FD` (0xC9, 'F') | `/dev/nvidiaN` | `nv_ioctl_register_fd_t` (4 B) | Link per-GPU FD to ctl FD |
| `NV_ESC_WAIT_OPEN_COMPLETE` (0xDA, 'F') | `/dev/nvidiaN` | `nv_ioctl_wait_open_complete_t` (8 B) | Wait for GSP-RM up |
| `NV_ESC_IOCTL_XFER_CMD` (0xD3, 'F') | either | `nv_ioctl_xfer_t` (16 B) → wraps any big struct | Wrapper for large args |
| `NV_ESC_RM_ALLOC` (0x2B, 'F') | either | `NVOS64_PARAMETERS` (48 B) | Allocate RM object (client/device/channel/memory/compute) |
| `NV_ESC_RM_CONTROL` (0x2A, 'F') | either | `NVOS54_PARAMETERS` (40 B) + payload | Invoke NV0000/NV2080/NVA06F/NVC36F control commands |
| `NV_ESC_RM_FREE` (0x29, 'F') | either | `NVOS00_PARAMETERS` (16 B) | Drop RM object |
| `NV_ESC_RM_MAP_MEMORY` (0x4E, 'F') | per-GPU | `nv_ioctl_nvos33_parameters_with_fd` (48+4 B) | Get mmap-cookie for RM memory handle |
| `NV_ESC_RM_UNMAP_MEMORY` (0x4F, 'F') | per-GPU | `NVOS34_PARAMETERS` (40 B) | Undo above |
| `NV_ESC_RM_MAP_MEMORY_DMA` (0x57, 'F') | per-GPU | `NVOS46_PARAMETERS` (56 B) | Install GPU PTEs for memory handle into VA space |
| `NV_ESC_RM_UNMAP_MEMORY_DMA` (0x58, 'F') | per-GPU | `NVOS47_PARAMETERS` (40 B) | Undo above |
| `NV_ESC_RM_VID_HEAP_CONTROL` (0x4A, 'F') | per-GPU | `NVOS32_PARAMETERS` (≈1 KB, see nvos.h:884) | Video-heap alloc/free/info/VA reserve |
| `NV_ESC_ALLOC_OS_EVENT` (0xCE, 'F') | either | `nv_ioctl_alloc_os_event_t` (12 B) | Bind RM event to eventfd |
| `UVM_INITIALIZE` (0x30000001) | `/dev/nvidia-uvm` | `UVM_INITIALIZE_PARAMS` (16 B) | Must be first ioctl |
| `UVM_REGISTER_GPU` (37) | UVM | `UVM_REGISTER_GPU_PARAMS` (≈48 B) | Tell UVM about an RM-attached GPU |
| `UVM_REGISTER_GPU_VASPACE` (25) | UVM | `UVM_REGISTER_GPU_VASPACE_PARAMS` | Share VASPACE with UVM |
| `UVM_MAP_EXTERNAL_ALLOCATION` (33) | UVM | `UVM_MAP_EXTERNAL_ALLOCATION_PARAMS` | Bring RM mem into UVM range |
| `UVM_MIGRATE` (51) | UVM | `UVM_MIGRATE_PARAMS` (≈80 B) | Sync or async migrate VA range |
| `UVM_PAGEABLE_MEM_ACCESS` (39) | UVM | `UVM_PAGEABLE_MEM_ACCESS_PARAMS` (8 B) | Query ATS/HMM availability |

Control commands (all via `NV_ESC_RM_CONTROL` payload):

| cmd | class+iface | purpose |
|---|---|---|
| `NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS` | 0x201 | List gpu_ids attached |
| `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2` | 0x205 | UUID, subdevice index |
| `NV2080_CTRL_CMD_GPU_GET_NAME_STRING` | 0x20800110 | "NVIDIA H100 80GB HBM3" etc. |
| `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` | 0xA06F0103 | Put channel on runlist |
| `NVA06F_CTRL_CMD_BIND` | 0xA06F0104 | Bind channel to engine (GR0) |
| `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` | 0xC36F0108 | Token to write to doorbell |

---

## 4. Unknowns / gotchas

The open-source tree is sufficient for steps 2.1 (cuInit), 2.2 (UVM/memory),
and the **setup** of 2.4 (channel create, compute-object bind, doorbell
mapping). The parts that will need empirical strace / reverse-engineering
work:

1. **Hopper QMD field layout (big one).** `cla0c0qmd.h` (Kepler A0C0) is the
   *only* QMD header shipped in open-gpu-kernel-modules. The Hopper QMD
   is a superset (~ 384 bytes, HMMA/WGMMA specific fields, TMA descriptors,
   cluster dims, optional barriers). CUDA embeds a QMD template inside
   each kernel's `.nv.info` and patches grid/block/smem/regcount/entryPC at
   launch time. Gleaning the layout from the ELF is the only option; a
   strace won't help here because the QMD is written to GPU memory as
   pushbuffer data, not through an ioctl. Likely reference: NVIDIA's
   `nvdisasm -qmd` (proprietary) or cross-referencing the cubin's
   `.nv.info.<kernel>` entries (`EIATTR_PARAM_CBANK`,
   `EIATTR_MAXREG_COUNT`, `EIATTR_KPARAM_INFO`, …).

2. **Method/pushbuffer header encoding.** The 1-dword header packing
   (opcode in bits 31:29, subchannel in 15:13, count in 28:16, method
   address in 12:0) is *convention*, cross-driver since Fermi. It is not
   spelled out in a single header in this repo. Can be confirmed via
   strace on a running CUDA workload or by reading Mesa's
   `nouveau` classes/ files which mirror this.

3. **`NV01_MEMORY_LOCAL_USER` / `NV01_MEMORY_SYSTEM` alloc params**.
   These go through `NV_ESC_RM_VID_HEAP_CONTROL` with `NVOS32_PARAMETERS`
   (`nvos.h:667-884`). The struct is huge, many union branches, and the
   field combinations CUDA actually uses (which `function`, which `attr`
   bitfield, which PTE kind) are not documented per-use-case. Cross-check
   with `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c` which itself calls these
   paths.

4. **Per-architecture class auto-selection.** CUDA's libcuda picks the
   `*_COMPUTE_*` class per GPU revision. Hopper pre-H100: it might be
   `ADA_COMPUTE_A` (0xC9C0) if NVIDIA classes GH100 as a variant. On
   GH200 specifically, `HOPPER_COMPUTE_A = 0xCBC0` is correct (verified
   by `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c:8693-8695`).

5. **GSP firmware dependence.** All of the `NV_ESC_RM_ALLOC` and
   `NV_ESC_RM_CONTROL` ioctls on Hopper are proxied to GSP-RM via RPC;
   the kernel module is thin. A given driver version ties to a specific
   GSP firmware blob. `NV_ESC_CHECK_VERSION_STR` verifies this. An
   independent libcuda replacement will be bound to whatever
   `nvidia.ko` + `gsp_gh100.bin` is installed — you cannot pick a
   version, you must read it.

6. **BAR1 window size and NUMA node of HBM on GH200.** The
   `nv_ioctl_card_info_t` returns `fb_size` but on GH200 the "fb" is the
   remote HBM reachable through NVLink-C2C. The UVM-returned `numaNodeId`
   from `UVM_REGISTER_GPU` is what you actually want — mmap allocations
   on that NUMA node with `mbind()` or use `numa_alloc_onnode()`. This is
   documented in comments, not in any parseable struct field.

7. **`nv_ioctl_register_fd_t::ctl_fd`.** Must be the same `fd_ctl` across
   all per-GPU FDs, otherwise RM treats each per-GPU FD as a separate
   client. Not enforced by struct layout; enforced by convention.

8. **`ioctl direction bits are ignored`.** The kernel reads only
   `_IOC_NR` and `_IOC_SIZE`. libcuda uses `_IOWR` direction but any of
   `_IOC_NONE / _IOR / _IOW / _IOWR` would work. Don't rely on matching
   direction bits.

9. **Confidential Compute / CC mode.** Numerous fields in
   `NV_CHANNEL_ALLOC_PARAMS` (`encryptIv`, `decryptIv`, `hmacNonce`) and
   `NVOS02_FLAGS_MEMORY_PROTECTION` are only relevant when HCC is on.
   For a non-CC inference stack, zero them. GH200 supports CC but does
   not require it.

10. **Channel group vs bare channel.** Modern CUDA allocates a
    `KEPLER_CHANNEL_GROUP_A` (TSG, 0xA06C) first, then channels under it.
    `NV_CHANNEL_ALLOC_PARAMS` requires a `hContextShare` that dangles off
    the group. Skipping the group and allocating a bare channel directly
    *may* still work on recent drivers but is deprecated — empirically
    confirm.

11. **`NV_ESC_RM_ALLOC` size wrap.** `NVOS64_PARAMETERS` itself is 48
    bytes, which fits in the 14-bit size field, but many alloc-param
    sub-structures (`NVOS32_PARAMETERS`, `NV_CHANNEL_ALLOC_PARAMS`) do
    NOT. Any call where `paramsSize > 16383 - sizeof(NVOS64_PARAMETERS)`
    must wrap in `NV_ESC_IOCTL_XFER_CMD`. Libcuda always wraps.

12. **`UVM_INITIALIZE_PARAMS::flags`.** Valid flags are defined in
    `kernel-open/nvidia-uvm/uvm_api.h` (not read here); includes bits for
    "multi-process" (`UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE`) and
    "disable teardown on last close". For single-process inference,
    passing `flags=0` works.

---

## Appendix A: file citation index

| Topic | File | Lines |
|---|---|---|
| Ioctl numbering | `kernel-open/common/inc/nv-ioctl-numbers.h` | 33-47 |
| RM escape numbers | `src/nvidia/arch/nvalloc/unix/include/nv_escape.h` | 31-53 |
| `nv_ioctl_xfer_t` | `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h` | 49-54 |
| Card info struct | `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h` | 56-68 |
| Version struct | `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h` | 100-105 |
| `nv_ioctl_register_fd_t` | `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h` | 126-129 |
| `nv_ioctl_wait_open_complete_t` | `src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h` | 152-156 |
| NVOS00/21/64 parameters | `src/common/sdk/nvidia/inc/nvos.h` | 165-171, 467-493 |
| NVOS33 (map memory) | `src/common/sdk/nvidia/inc/nvos.h` | 1848-1858 |
| NVOS46 (map memory DMA) | `src/common/sdk/nvidia/inc/nvos.h` | 2170-2187 |
| NVOS54 (RM control) | `src/common/sdk/nvidia/inc/nvos.h` | 2232-2242 |
| `NV_HOPPER_USERMODE_A_PARAMS` | `src/common/sdk/nvidia/inc/nvos.h` | 3320-3328 |
| Device class alloc | `src/common/sdk/nvidia/inc/class/cl0080.h` | 36-64 |
| Subdevice class | `src/common/sdk/nvidia/inc/class/cl2080.h` | 36 |
| Hopper channel FIFO | `src/common/sdk/nvidia/inc/class/clc86f.h` | 27-186 |
| Hopper compute class ID | `src/common/sdk/nvidia/inc/class/clcbc0.h` | 26 |
| Hopper usermode class | `src/common/sdk/nvidia/inc/class/clc661.h` | 26 |
| Volta usermode layout (doorbell @0x90) | `src/common/sdk/nvidia/inc/class/clc361.h` | 27-33 |
| Channel alloc params | `src/common/sdk/nvidia/inc/alloc/alloc_channel.h` | 296-344 |
| GPFIFO schedule ctrl | `src/common/sdk/nvidia/inc/ctrl/ctrla06f/ctrla06fgpfifo.h` | 66-73 |
| Work-submit token ctrl | `src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h` | 79-85 |
| Ioctl dispatcher | `kernel-open/nvidia/nv.c` | 2358, 2410-2470 |
| UVM ioctl base | `kernel-open/nvidia-uvm/uvm_ioctl.h` | 38-46 |
| `UVM_INITIALIZE` | `kernel-open/nvidia-uvm/uvm_linux_ioctl.h` | 32-42 |
| `UVM_REGISTER_GPU` | `kernel-open/nvidia-uvm/uvm_ioctl.h` | 434-445 |
| `UVM_MAP_EXTERNAL_ALLOCATION` | `kernel-open/nvidia-uvm/uvm_ioctl.h` | 394-407 |
| `UVM_MIGRATE` | `kernel-open/nvidia-uvm/uvm_ioctl.h` | 674-687 |
| Chardev major/minor | `kernel-open/common/inc/nv-chardev-numbers.h` | 29-40 |
| Usermode/doorbell mapping (reference impl) | `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c` | 5541-5700 |
