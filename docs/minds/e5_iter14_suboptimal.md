# e5 iter14 — New Suboptimal Findings

Date: 2026-04-13

## 1. Kernel param array rebuilt every launch (hot path)

`src/cuda_driver.py:launch()` builds a fresh Python list (`packed`), allocates a new ctypes array (`(c_void_p * n)()`), and fills it via a Python for-loop on every `cuLaunchKernel` call. For the fused single-launch path this is invisible. For the embed loop (`pipeline_exec.py:124-136`, one launch per token, n_tokens iterations) this is pure Python allocation overhead per launch. The array is identical across calls except for `out_addr` and `tid` — the whole structure should be pre-allocated and mutated in place.

## 2. Single stream, all launches serialized

Every launch passes `stream=None` (default stream). The driver has `cuStreamCreate` bound but no production path uses it. Embed, norm, projection, and lm_head kernels are independent given the activation buffer — the embed loop and the first norm launch could overlap on separate streams. This is a GAPS.md §4d known deferral, but it is a real latency left on the table.

## 3. Norm weight BF16→F32 conversion on CPU per session

`generate_paris_fused.py:185-190`: all 64×2 = 128 norm weight vectors are converted BF16→F32 on CPU and uploaded individually with 128 separate `mem_alloc` + `memcpy_htod` calls. These could be batch-uploaded in a single allocation and a single async copy, or converted in-place by a tiny GPU kernel, removing 127 extra round-trips.

## 4. Weight table uploaded via `view(float32)` type alias

`generate_paris_fused.py:234`: the `uint64` pointer table is reinterpreted as `float32` to use `upload_f32`. This aliases 8-byte integers through a 4-byte float view — fragile and semantically wrong even if it works on little-endian. A proper `mem_alloc` + `memcpy_htod` with the raw `uint64` buffer avoids the type pun.

## 5. Embed token row bytes() copy at startup

`generate_paris_fused.py:153`: `bytes(model.weight_bytes(...))` materializes the *entire* embed weight (152 MB for Qwen 27B vocab × 5120 × 2B) as a Python bytes object just to slice one row. Should index directly into the mmap buffer with `embed_raw[offset:offset+row_bytes]` without forcing a full copy.

## 6. `pipeline_exec.py` norm launches with `shared_mem=128` for a 256-thread block

Line 218 and 270: the norm kernel is launched with `shared_mem=128` bytes dynamic shared. The comment at line 125 of `generate_paris_fused.py` shows the static shared footprint is ~41KB declared in PTX. Passing 128 bytes dynamic on top of 41KB static is harmless but documents confusion — if the kernel ever switches to dynamic allocation the 128-byte ceiling becomes the real bug.

---

Dance vector: [param-rebuild, single-stream, norm-batch-upload, u64-pun, embed-full-copy, smem-confusion]
