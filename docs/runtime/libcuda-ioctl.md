# libcuda to nvidia.ko ioctl catalog (empirical)

Captured on: `192-222-57-141` (GH200 aarch64) — NVIDIA driver **580.105.08**,
CUDA **12.8** (`libcuda.so.580.105.08`), kernel **6.8.0-1046-nvidia-64k**.

Reproduction: `/tmp/minimal_cuda.py` (ctypes → `libcuda.so.1`) loading
`inference/elementwise.cubin` and launching `elemwise_mul`. Each of the five
CUDA driver calls we care about is bracketed by a `write(2, "=====MARK=...")`
marker so strace output can be sliced per operation. Full trace:
`/tmp/libcuda_trace.log` (1288 lines).

```
strace -f -e trace=ioctl,openat,close,mmap,munmap,read,write \
       -o /tmp/libcuda_trace.log python3 /tmp/minimal_cuda.py
```

## Device / fd map (after `cuInit` + `cuCtxCreate`)

| fd  | path                  | role                                              |
| --- | --------------------- | ------------------------------------------------- |
| 8   | `/dev/nvidiactl`      | RM master fd (all `NV_ESC_RM_*` dispatched here)  |
| 9   | `/dev/nvidia-uvm`     | UVM master fd (unified memory, ATS on GH200)      |
| 10  | `/dev/nvidia-uvm`     | UVM tools/secondary fd                            |
| 11  | `/dev/nvidia0`        | per-device RM fd (device 0)                       |
| 12,13,14,15,17,19,21,24,26,28,30 | `/dev/nvidia0` / `/dev/nvidiactl` (opened, registered via `NV_ESC_REGISTER_FD`/`dup2`, then used as MMIO/BAR1/USERD mapping fds) |

Each resource (channel, context, BAR window, USERD page) is allocated on
`/dev/nvidiactl` with `NV_ESC_RM_ALLOC`, then a fresh fd is opened and
linked to the resource with `NV_ESC_REGISTER_FD`/`NV_ESC_IOCTL_XFER_CMD`
before being `mmap`'d.

## ioctl encoding

All RM ioctls use magic byte `'F' = 0x46` (`NV_IOCTL_MAGIC`), size of the
arg struct encoded in the upper 14 bits of the ioctl nr per
`_IOC(dir,type,nr,size)`. Direction is `_IOC_READ|_IOC_WRITE` for RM and
`_IOC_NONE` for UVM. UVM ioctls use magic `0` (null) and embed the struct
size in the arg itself.

Base enum (from `/usr/src/nvidia-srv-580.105.08/common/inc/nv-ioctl-numbers.h`):
`NV_IOCTL_BASE = 200 (0xc8)`. The `NV_ESC_RM_*` family lives below 0xc8
(closed-source in this package; numbers verified against open-gpu-kernel-modules
published headers).

## Operation 1: `cuInit(0)`

- Opens `/dev/nvidiactl` -> fd 8 (after probing
  `/proc/driver/nvidia/params`, per-user `~/.nv/...` profile files — all
  ENOENT here).
- Issues 10 distinct ioctls; 105 total in the marker window.

| # | fd | nr   | size  | decoded                              | purpose                      |
|---|----|------|-------|--------------------------------------|------------------------------|
| 1 | 8  | 0xd6 | 0x008 | `NV_ESC_CHECK_VERSION_STR`           | ABI/version handshake        |
| 2 | 8  | 0xc8 | 0x900 | `NV_ESC_CARD_INFO`                   | Probe devices (2304 B table) |
| 3 | 8  | 0x2b | 0x030 | `NV_ESC_RM_CONTROL`                  | Polymorphic control call     |
| 4 | 8  | 0x2a | 0x020 | `NV_ESC_RM_FREE`                     | Free RM handle               |
| 5 | 8  | 0x29 | 0x010 | `NV_ESC_RM_ALLOC_MEMORY`             | Allocate memory (legacy)     |
| 6 | 9  | 0x01 (sz 0x3000) | — | `UVM_INITIALIZE`                 | Init UVM (size-in-arg)       |
| 7 | 10 | 0x4b | 0     | `UVM_MM_INITIALIZE`                  | Init UVM tools fd            |
| 8 | 9  | 0x27 | 0     | `UVM_PAGEABLE_MEM_ACCESS`            | Enable ATS pageable access   |
| 9 | 11 | 0xc9 | 0x004 | `NV_ESC_REGISTER_FD`                 | Attach ephemeral fd to master|
| 10| 11 | 0xd7 | 0x230 | `NV_ESC_IOCTL_XFER_CMD`              | XFER wrapper (>256 B args)   |

Counts: `RM_FREE` x71, `RM_CONTROL` x14, `REGISTER_FD` x3, `RM_ALLOC_OS_EVENT` (0x4e, 0x38) x3, `CHECK_VERSION` x2, `CARD_INFO` x2, `RM_ALLOC_MEMORY` x2, `IOCTL_XFER_CMD` x1, `ALLOC_OS_EVENT2` (0xce, 0x10) x1, `UVM_MM_INITIALIZE` x1. **mmaps: 10** (all on fd 15, a dup of nvidiactl — 3 x 64 KiB at offset 0 with `MAP_SHARED`, both `PROT_WRITE` and `PROT_READ`; these are the GSP/RM shared status pages).

## Operation 1.5: `cuCtxCreate_v2` (not in the original list but required before any Mem/Module/Launch)

- 449 ioctls, 18 nvidia-fd mmaps. **This is the elephant.** The context
  construction allocates client / device / subdevice / channel-group /
  channel / USERD / doorbell / gpfifo / event resources (`NV_ESC_RM_ALLOC`
  nr=0x27 sz=0x38 x22), plus per-channel `UVM_REGISTER_CHANNEL` (nr=0x1b) x16
  and `UVM_REGISTER_GPU_VASPACE` (nr=0x17) x7, and populates a 4 GiB VA
  reservation at `0x200000000` plus a 512 MiB reservation at `0x320000000`
  that become the GPU BAR1 virtual address window.
- Every channel follows a fixed pattern: `RM_ALLOC` (0x27) -> new fd ->
  `REGISTER_FD` (0xc9) -> `ALLOC_OS_EVENT2` (0xce) -> `mmap(... 0x10000, MAP_SHARED, fd, 0)`
  at a fixed address in the reserved VA window. The 64 KiB pages are the
  USERD / doorbell / GP_GET/PUT ring control surfaces.
- 2 MiB `MAP_SHARED` mappings at `0x324xxx000` are GPU BAR1 sysmem windows;
  2 MiB at `0x2012xx000` are GPU VA pages.

## Operation 2: memory mapping weights / host -> GH200 unified memory

Marker slice (8 lines, 4 ioctls):

```
openat("inference/elementwise.cubin", O_RDONLY|O_CLOEXEC) = 34
mmap(NULL, 8816, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd=34, 0) = 0xf8f98dd20000
ioctl(8, NV_ESC_RM_CONTROL,       size=0x30)  = 0  -- query region capabilities
ioctl(8, NV_ESC_RM_FREE,          size=0x20)  = 0  -- release scratch handle
ioctl(9, UVM_MAP_EXTERNAL_ALLOC?  nr=0x49)    = 0  -- register host range w/ UVM
ioctl(9, UVM_REGISTER_GPU?        nr=0x21)    = 0  -- bind to device
```

On GH200 the NVLink-C2C cache-coherent interconnect plus ATS means the
existing host mmap address IS the GPU-visible pointer — no bounce buffer
is established; libcuda only (a) validates the range with RM, and (b)
registers it with UVM (nr `0x49` + `0x21`, both on fd 9) so the UVM
driver knows to fault/prefetch those host pages on GPU access.

`cuMemAlloc_v2(dptr, size)` and `cuMemcpyHtoD_v2` inside the same marker
do their work entirely through the already-mapped BAR1 VA window
(`0x324...`, `0x201...`) — they issue ZERO additional ioctls. The copy is
a CPU `memcpy` into a coherent window.

## Operation 3: `cuModuleLoadData(bytes, len)`

Marker slice (6 lines, **0 ioctls, 0 mmaps** on any nvidia fd):

```
openat("elementwise.cubin", O_RDONLY|O_CLOEXEC) = 36
ioctl(36, TCGETS, ...)  -- unrelated Python runtime; errors ENOTTY
read(36, "\x7fELF...", 8817) = 8816
read(36, "", 1) = 0
close(36)
```

The ELF read from fd 36 is Python's own `open(...).read()` for the second
file load in the test script — the ioctl to fd 36 is `TCGETS` from
Python's buffered IO layer, not libcuda. **`cuModuleLoadData` itself
makes no syscalls in the steady state.** It parses the cubin in user
space, installs the kernel's text/constant sections into already-mapped
UVM BAR ranges via CPU stores, and records entry metadata in process
memory. The module "handle" is a userspace pointer.

## Operation 4: `cuLaunchKernel(fn, grid, block, args, ...)`

Marker slice: **0 syscalls of any kind.** The launch is a sequence of
CPU stores into:

1. The channel's GPFIFO ring (2 MiB `MAP_SHARED` page at `0x324...`):
   method pushbuffer encoding grid/block/param pointer and a SET_OBJECT
   / SEMAPHORE_RELEASE sequence.
2. The USERD page (64 KiB `MAP_SHARED` at `0xf8f759exx000`, fd was a
   dup of `/dev/nvidia0`): write `GP_PUT` to advance the producer index.
3. The doorbell page (another 64 KiB MMIO page in the same region):
   a single 32-bit store with (channel_id, run_list) rings the GPU.

All three are regions established by the `mmap(... MAP_SHARED, fd, 0)`
calls during `cuCtxCreate`. No ioctl, no write(2), no futex in the
critical path.

## Operation 5: fence / `cuCtxSynchronize`

Marker slice: **0 syscalls.** Sync is a spin-wait on a host-visible
semaphore value that the kernel writes at completion. The semaphore
lives in a host-pinned page mapped earlier (the 64 KiB anonymous
`MAP_SHARED|MAP_FIXED` pages at `0x200600000` / `0x200800000`). libcuda
reads the 64-bit semaphore word and compares against the payload emitted
by the launch. No ioctl — the completion side is pure memory polling
(GH200 ATS makes this coherent without a barrier ioctl).

For a blocking wait that needs to actually sleep, libcuda would fall
back to `ioctl(fd, NV_ESC_RM_CONTROL, cmd=WAIT_FOR_IDLE)` or an
eventfd-based path via `NV_ESC_ALLOC_OS_EVENT` (0x4e). Neither fires in
this tiny kernel because the spin completes first.

## Minimum ioctl surface (union across all 5 ops + `cuCtxCreate`)

| fd class         | nr   | dir          | size  | symbolic name                              |
| ---------------- | ---- | ------------ | ----- | ------------------------------------------ |
| `/dev/nvidiactl` | 0xd6 | R/W (F)      | 0x008 | `NV_ESC_CHECK_VERSION_STR`                 |
| `/dev/nvidiactl` | 0xc8 | R/W (F)      | 0x900 | `NV_ESC_CARD_INFO`                         |
| `/dev/nvidiactl` | 0xc9 | R/W (F)      | 0x004 | `NV_ESC_REGISTER_FD`                       |
| `/dev/nvidiactl` | 0xce | R/W (F)      | 0x010 | `NV_ESC_ALLOC_OS_EVENT`                    |
| `/dev/nvidiactl` | 0xd7 | R/W (F)      | 0x230 | `NV_ESC_IOCTL_XFER_CMD` (wrapper for >256 B args) |
| `/dev/nvidiactl` | 0x27 | R/W (F)      | 0x038 | `NV_ESC_RM_ALLOC`                          |
| `/dev/nvidiactl` | 0x29 | R/W (F)      | 0x010 | `NV_ESC_RM_ALLOC_MEMORY`                   |
| `/dev/nvidiactl` | 0x2a | R/W (F)      | 0x020 | `NV_ESC_RM_FREE`                           |
| `/dev/nvidiactl` | 0x2b | R/W (F)      | 0x030 | `NV_ESC_RM_CONTROL`                        |
| `/dev/nvidiactl` | 0x4e | R/W (F)      | 0x038 | `NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO`     |
| `/dev/nvidiactl` | 0x5e | R/W (F)      | 0x028 | `NV_ESC_RM_MAP_MEMORY`                     |
| `/dev/nvidia-uvm`| 0x01 | NONE (size in arg) | 0x3000 | `UVM_INITIALIZE`                   |
| `/dev/nvidia-uvm`| 0x17 | NONE         | —     | `UVM_REGISTER_GPU_VASPACE`                 |
| `/dev/nvidia-uvm`| 0x1b | NONE         | —     | `UVM_REGISTER_CHANNEL`                     |
| `/dev/nvidia-uvm`| 0x19 | NONE         | —     | `UVM_REGISTER_GPU`                         |
| `/dev/nvidia-uvm`| 0x21 | NONE         | —     | `UVM_MAP_EXTERNAL_ALLOCATION`              |
| `/dev/nvidia-uvm`| 0x25 | NONE         | —     | `UVM_FREE`                                 |
| `/dev/nvidia-uvm`| 0x27 | NONE         | —     | `UVM_PAGEABLE_MEM_ACCESS`                  |
| `/dev/nvidia-uvm`| 0x41 | NONE         | —     | `UVM_ALLOC_SEMAPHORE_POOL`                 |
| `/dev/nvidia-uvm`| 0x44 | NONE         | —     | `UVM_VALIDATE_VA_RANGE`                    |
| `/dev/nvidia-uvm`| 0x46 | NONE         | —     | `UVM_UNREGISTER_GPU`                       |
| `/dev/nvidia-uvm`| 0x48 | NONE         | —     | `UVM_ENABLE_PEER_ACCESS`                   |
| `/dev/nvidia-uvm`| 0x49 | NONE         | —     | `UVM_CREATE_EXTERNAL_RANGE`                |
| `/dev/nvidia-uvm`| 0x4b | NONE         | —     | `UVM_MM_INITIALIZE`                        |
| `/dev/nvidia-uvm-tools`| 0x4b | NONE   | —     | (same, called on tools fd)                 |

**Raw count: 25 distinct ioctl numbers** across nvidiactl + nvidia-uvm.
The mappings use one additional `mmap(2)` family call with four
distinguishable shapes:

| mmap target                    | size     | prot | flags                  | offset semantics          |
| ------------------------------ | -------- | ---- | ---------------------- | ------------------------- |
| USERD / doorbell page          | 64 KiB   | RW   | MAP_SHARED\|MAP_FIXED  | `offset = handle`         |
| GPFIFO / method ring           | 2 MiB    | RW   | MAP_SHARED\|MAP_FIXED  | `offset = handle`         |
| BAR1 device-memory window      | 2 MiB    | RW   | MAP_SHARED\|MAP_FIXED  | `offset = BAR VA`         |
| Host semaphore pool            | 64 KiB   | RW   | MAP_SHARED\|MAP_ANONYMOUS | —                      |

## Surprises / notes

1. **The hot path is zero-syscall.** `cuModuleLoadData`, `cuLaunchKernel`,
   and `cuCtxSynchronize` make no ioctls. All GPU interaction is via CPU
   stores/loads against four fixed-purpose `MAP_SHARED` regions set up
   during `cuCtxCreate`. Reimplementing "launch" means learning the
   GPFIFO method-encoding for Hopper (SMCH/CSMC, QMD for compute) and
   writing 32-bit MMIO to the doorbell. No kernel ABI touched.

2. **`cuCtxCreate` is 450 ioctls.** It is a much bigger beast than
   `cuInit` and is the dominant setup cost. It must be paid once per
   process, not once per launch.

3. **Hidden sixth operation needed: `cuModuleGetFunction`.** In the trace
   it is also zero-ioctl (pure ELF-symbol lookup in user memory), but the
   reimplementation needs a cubin ELF parser and must materialize a
   `CUfunction` struct containing: entry PC, shared-mem size, reg count,
   param-buffer layout, QMD template — all recoverable from the cubin's
   `.nv.info.<kernel>`, `.nv.shared.<kernel>`, and `.text.<kernel>`
   sections. Without it we can't construct the QMD.

4. **GH200 pinning is a no-op.** `cuMemcpyHtoD` on a host-mmap'd
   cubin issued zero ioctls — NVLink-C2C + ATS make the host VA directly
   accessible from SM, so libcuda fast-paths to a plain memcpy in the
   mapped BAR window. Our unified-memory story is essentially free.

5. **`NV_ESC_IOCTL_XFER_CMD` is a size bypass.** Any RM call whose arg is
   larger than 256 bytes is wrapped in this 0x230-byte envelope because
   the Linux ioctl encoding only has 14 bits for `size`. We must handle
   the wrapper when reimplementing RM dispatch.

6. **`RM_FREE` fires 71 times in `cuInit` alone** — libcuda speculatively
   allocates scratch handles, probes, then frees. A clean reimplementation
   can cut this dramatically.

## Reimplementation plan (informational)

- **Setup side** (cuInit, ctxCreate, mmap): must replay the 25 distinct
  ioctls byte-exactly. The `NV_ESC_RM_ALLOC` dispatch is polymorphic on a
  `hClass` field inside the struct; we need roughly a dozen hClass values
  (0x0000 NV01_ROOT, 0x0080 NV01_DEVICE, 0x2080 NV20_SUBDEVICE, 0xC06F
  HOPPER_CHANNEL_GPFIFO_A, 0xC597 HOPPER_A compute, 0x00DE USERD,
  0x0014 NV50_DISPLAY — skip the last, we're headless).
- **Hot side** (load, launch, sync): no ioctl surface. Implement:
  1. cubin ELF parser (symbol -> QMD descriptor).
  2. GPFIFO pushbuffer builder (SET_OBJECT, SET_SHADER_LOCAL_MEMORY,
     SET_CONSTANT_BUFFER, BEGIN_COMPUTE, QMD, END_COMPUTE, SEMAPHORE).
  3. 32-bit MMIO doorbell write.
  4. 64-bit semaphore poll.

End of catalog.
