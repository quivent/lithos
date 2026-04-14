# launcher.s libcuda audit

Source: `/home/ubuntu/lithos/src/launcher.s` (4096 lines). Linked via
`ld ... -lcuda` (line 13). All `bl cuXXX` call sites enumerated below.

## Summary

- Total `bl cuXXX` call sites: **42**
- Unique CUDA driver APIs used: **14**
  - `cuInit`, `cuDeviceGet`, `cuCtxCreate_v2`, `cuCtxSynchronize`
  - `cuStreamCreate`, `cuStreamWaitEvent`
  - `cuEventCreate`, `cuEventRecord`
  - `cuMemAlloc`, `cuMemcpyDtoH`, `cuMemcpyAsync`
  - `cuModuleLoadData`, `cuModuleGetFunction`
  - `cuLaunchKernel`
- Additional related references (data / comments): `CU_MEMHOSTREGISTER_DEVICEMAP` (L60), `cuda_device/context/stream{_q,_k,_v}/event_{q,k,v}/logits_dev` globals (L264-L360).

## By category

### Init / Context (runtime/init.ls)

| Line | API | Purpose |
|------|-----|---------|
| 417 | `cuInit(0)` | Initialize CUDA driver (one-shot). Replaced by opening `/dev/vfio/<grp>`, `VFIO_GROUP_GET_DEVICE_FD`, `VFIO_DEVICE_GET_REGION_INFO`. |
| 430 | `cuDeviceGet(&cuda_device, 0)` | Resolve ordinal 0 to a device handle. Replaced by selecting the vfio group bound to the target BDF. Output: `cuda_device` (L264). |
| 440 | `cuCtxCreate_v2(&cuda_context, 0, device)` | Create + push a primary context (TLB, page tables, channel). Replaced by mmap BAR0 (PCIe config/MMIO) + BAR4 (framebuffer/VRAM) and allocating a GPFIFO channel via ioctl. Output: `cuda_context` (L265). |

Inputs/outputs of interest: writes `cuda_device`, `cuda_context`. After this
point `x19` holds the stream handle and all launches use the primary context.

### Memory (runtime/mem.ls)

| Line | API | Purpose |
|------|-----|---------|
| 3133 | `cuMemAlloc(&cuda_logits_dev, VOCAB_SIZE*2)` | Allocate VRAM for logits output buffer. Replaced by a BAR4 bump allocator returning a device VA. Output: `cuda_logits_dev` (L360). |
| 960 | `cuMemcpyDtoH(logits_buf, cuda_logits_dev, bytes)` | Device-to-host blocking copy of logits after lm_head. Replaced by direct CPU-side read through BAR4 CPU mapping (GH200 cache-coherent) — no copy needed, or a DMA descriptor pushed via `runtime/pushbuffer.ls`. |
| 2234 | `cuMemcpyAsync(dst, src, bytes, cuda_stream)` | Async D2D copy (K into KV cache slot at seq_pos). Replaced by a DMA copy command on the pushbuffer (`runtime/pushbuffer.ls`) or an inline memcpy kernel primitive. |
| 2242 | `cuMemcpyAsync(dst, src, bytes, cuda_stream)` | Async D2D copy (V into KV cache slot). Same replacement. |

Not present (intentionally disabled): `cuMemHostRegister` (comment L60, L220,
L2998 says it returns 801 NOT_SUPPORTED on GH200 and is skipped).

### ELF loading (runtime/elf_load.ls)

| Line | API | Purpose |
|------|-----|---------|
| 3934 | `cuModuleLoadData(&module, buffer)` | Parse an in-memory cubin, upload code+constants to device, relocate. Called once per cubin from `load_one_cubin` (L3861). Replaced by parsing the compiled Lithos GPU ELF, copying `.text.*` to a VRAM code segment via BAR4, and recording `entry_pc`. |
| 3941 | `cuModuleGetFunction(&func, module, name)` | Resolve a kernel symbol inside a loaded module to a `CUfunction`. Replaced by symbol-table lookup in the Lithos ELF; returns the kernel's `entry_pc` stored in the kernel function table at `x28 + KF_*` (L78-L89). |

Call sites of `load_one_cubin` (each triggers one `cuModuleLoadData` + one
`cuModuleGetFunction`): L3735, L3746, L3757, L3768, L3779, L3790, L3801,
L3812, L3823, L3834, L3845 (11 cubins: embed, norm, proj, attn, recur,
activate, rope, gate_sig, conv1d, l2norm, lm_head).

### Launch (runtime/launch.ls)

Every `cuLaunchKernel` takes `(CUfunction, gridX/Y/Z, blockX/Y/Z, smem,
stream, kernelParams**, extra)`. Replaced by building a QMD (Queue Meta
Descriptor) with grid/block/smem/entry_pc/param-buffer, appending a
COMPUTE_LAUNCH method to the pushbuffer (`runtime/pushbuffer.ls`), and
ringing the doorbell (`runtime/doorbell.ls`).

| Line | API | Purpose / kernel dispatched |
|------|-----|-----------------------------|
| 761  | `cuLaunchKernel` | embed (token id -> hidden). Stream = `cuda_stream` (L755). |
| 1218 | `cuLaunchKernel` | RMSNorm (pre-attn / pre-MLP / final). Stream = `cuda_stream`. |
| 1534 | `cuLaunchKernel` | GPTQ GEMV projection (generic path, DN non-QKV and all FA non-QKV). |
| 1638 | `cuLaunchKernel` | FA Q projection on `cuda_stream_q`. |
| 1714 | `cuLaunchKernel` | FA K projection on `cuda_stream_k`. |
| 1789 | `cuLaunchKernel` | FA V projection on `cuda_stream_v`. |
| 1949 | `cuLaunchKernel` | FA attention score (softmax attn). Stream = `cuda_stream`. |
| 2063 | `cuLaunchKernel` | DeltaNet recurrence (`deltanet_step`). Stream = `cuda_stream`. |
| 2142 | `cuLaunchKernel` | SwiGLU activation. Stream = `cuda_stream`. |
| 2354 | `cuLaunchKernel` | RoPE rotate (applied to Q/K post-projection). Stream = `cuda_stream`. |
| 2451 | `cuLaunchKernel` | L2 norm (reduce) on DeltaNet q/k. Stream = `cuda_stream`. |
| 2527 | `cuLaunchKernel` | gate_sigmoid (DeltaNet output gate). Stream = `cuda_stream`. |
| 2633 | `cuLaunchKernel` | z-projection (DeltaNet gate path, separate launch to reuse original x). |
| 2801 | `cuLaunchKernel` | conv1d_infer (DeltaNet short conv). Stream = `cuda_stream`. |
| 2895 | `cuLaunchKernel` | final RMSNorm before lm_head. Stream = `cuda_stream`. |
| 2969 | `cuLaunchKernel` | lm_head GEMV (hidden -> vocab logits, writes `cuda_logits_dev`). |

17 `cuLaunchKernel` sites total. Each becomes one QMD build + pushbuffer
append. Grid/block/smem values must be moved from ARM64 registers into the
QMD fields directly (not as API args).

### Sync (runtime/sync.ls)

| Line | API | Purpose |
|------|-----|---------|
| 452  | `cuStreamCreate(&cuda_stream, 0)` | Main compute stream. Replaced by allocating a GPFIFO slot + a 4-byte completion flag in BAR4 (`runtime/sync.ls`). |
| 459  | `cuStreamCreate(&cuda_stream_q, 0)` | FA Q-projection stream. |
| 465  | `cuStreamCreate(&cuda_stream_k, 0)` | FA K-projection stream. |
| 471  | `cuStreamCreate(&cuda_stream_v, 0)` | FA V-projection stream. |
| 479  | `cuEventCreate(&cuda_event_q, CU_EVENT_DISABLE_TIMING=2)` | Completion event for Q. Replaced by a semaphore word in BAR4 released by a SEMAPHORE_RELEASE method appended after the launch. |
| 485  | `cuEventCreate(&cuda_event_k, 2)` | Completion event for K. |
| 491  | `cuEventCreate(&cuda_event_v, 2)` | Completion event for V. |
| 1648 | `cuEventRecord(event_q, stream_q)` | Publish Q-done. Replaced by pushbuffer SEMAPHORE_RELEASE on q channel. |
| 1724 | `cuEventRecord(event_k, stream_k)` | Publish K-done. |
| 1799 | `cuEventRecord(event_v, stream_v)` | Publish V-done. |
| 1814 | `cuStreamWaitEvent(cuda_stream, event_q, 0)` | Main stream waits for Q. Replaced by SEMAPHORE_ACQUIRE on main channel. |
| 1823 | `cuStreamWaitEvent(cuda_stream, event_k, 0)` | Main stream waits for K. |
| 1832 | `cuStreamWaitEvent(cuda_stream, event_v, 0)` | Main stream waits for V. |
| 931  | `cuCtxSynchronize()` | Block CPU until every stream drains. Called once per decoded token before the DtoH copy. Replaced by spin-polling the completion flag in BAR4. |
| 4060 | `cuCtxSynchronize()` | Same, inside `debug_sync_tag` (L4044) — diagnostic barrier. |

Outputs of interest: `cuda_stream{,_q,_k,_v}` (L266-L269),
`cuda_event_{q,k,v}` (L270-L272).

### Teardown (runtime/teardown.ls)

No explicit `cuCtxDestroy` / `cuStreamDestroy` / `cuEventDestroy` /
`cuMemFree` / `cuModuleUnload` calls are present — `_start` calls
`SYS_EXIT` (L44) and lets the kernel reclaim everything. Replacement plan:
`runtime/teardown.ls` performs `munmap(BAR0/BAR4)` + `close(vfio_fd)` + the
same `SYS_EXIT 93`. No libcuda calls to audit here.

## Lithos replacement entry points

Each `runtime/*.ls` file owns the following call sites.

### runtime/init.ls — 3 sites
- L417 `cuInit`
- L430 `cuDeviceGet`
- L440 `cuCtxCreate_v2`

Also owns (non-libcuda) setup state: `cuda_device` (L264), `cuda_context`
(L265).

### runtime/mem.ls — 4 sites
- L3133 `cuMemAlloc` (logits device buffer)
- L960  `cuMemcpyDtoH` (logits readback)
- L2234 `cuMemcpyAsync` (K -> KV cache)
- L2242 `cuMemcpyAsync` (V -> KV cache)

Also owns: `cuda_logits_dev` (L360), the shard/weight mmap region
(cuMemHostRegister path already dead, L2998).

### runtime/elf_load.ls — 2 APIs, 11 logical cubins
- L3934 `cuModuleLoadData` (inside `load_one_cubin`)
- L3941 `cuModuleGetFunction` (inside `load_one_cubin`)

Callers: L3735, L3746, L3757, L3768, L3779, L3790, L3801, L3812, L3823,
L3834, L3845. Populates kernel function table at `x28 + KF_*`
(L78-L89).

### runtime/launch.ls — 17 sites
All `cuLaunchKernel`: L761, L1218, L1534, L1638, L1714, L1789, L1949,
L2063, L2142, L2354, L2451, L2527, L2633, L2801, L2895, L2969. (Plus
L744 comment-only.)

### runtime/sync.ls — 15 sites
- Stream create: L452, L459, L465, L471
- Event create: L479, L485, L491
- Event record: L1648, L1724, L1799
- Stream wait event: L1814, L1823, L1832
- Ctx sync: L931, L4060

### runtime/teardown.ls — 0 libcuda sites
Matches the existing `SYS_EXIT` shutdown. Adds `munmap` + `close`.

### runtime/qmd.ls, runtime/pushbuffer.ls, runtime/doorbell.ls
Lower-level primitives invoked by `runtime/launch.ls` (for every
`cuLaunchKernel` site) and by `runtime/mem.ls` (for the two
`cuMemcpyAsync` sites, which lower to DMA methods on the pushbuffer).
No direct 1:1 libcuda mapping — these replace the opaque work the
driver does inside `cuLaunchKernel` (QMD assembly, GPFIFO append,
USERD doorbell write).

## Cross-reference: API -> site count

| API | Sites | Owner |
|-----|-------|-------|
| cuInit              | 1  | init.ls |
| cuDeviceGet         | 1  | init.ls |
| cuCtxCreate_v2      | 1  | init.ls |
| cuStreamCreate      | 4  | sync.ls |
| cuEventCreate       | 3  | sync.ls |
| cuEventRecord       | 3  | sync.ls |
| cuStreamWaitEvent   | 3  | sync.ls |
| cuCtxSynchronize    | 2  | sync.ls |
| cuMemAlloc          | 1  | mem.ls |
| cuMemcpyDtoH        | 1  | mem.ls |
| cuMemcpyAsync       | 2  | mem.ls |
| cuModuleLoadData    | 1  | elf_load.ls |
| cuModuleGetFunction | 1  | elf_load.ls |
| cuLaunchKernel      | 17 | launch.ls (via qmd/pushbuffer/doorbell) |
| **Total**           | **42** | |
