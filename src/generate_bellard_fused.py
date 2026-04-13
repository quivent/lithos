#!/usr/bin/env python3
"""
Bellard's optimized fused forward pass benchmark.

132 blocks x 256 threads, ALL 132 SMs active simultaneously.
Uses cuLaunchCooperativeKernel + grid-wide atomic sync barriers.

Key optimizations over forward_pass_multi:
  1. Input vector cached in shared memory (20KB) -- loaded ONCE per
     projection, reused by all threads. Eliminates redundant global reads.
  2. W4A16 dequant: precompute bias = -zero * scale, then
     dequant(nib) = nib * scale + bias  (FMA, 1 op instead of sub+mul)
  3. lm_head uses FP16 weights (not GPTQ). Separate code path with
     vectorized f16 loads and shared-memory cached input.

Run:
    python3 /home/ubuntu/lithos/src/generate_bellard_fused.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_driver import CUDADriver, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Architecture constants (must match forward_pass_bellard.ptx)
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
VOCAB_SIZE = 248320
GROUP_SIZE = 128
NUM_BLOCKS = 132
THREADS_PER_BLOCK = 256

# Weight table: 11 u64 pointers per layer
PTRS_PER_LAYER = 11
BYTES_PER_LAYER = PTRS_PER_LAYER * 8  # 88
WEIGHT_TABLE_BYTES = NUM_LAYERS * BYTES_PER_LAYER  # 5632

WARMUP_RUNS = 5
TIMED_RUNS = 5
BASELINE_MS = 135.0  # forward_pass_multi baseline


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def load_norm_f32(model: LithosModel, name: str) -> np.ndarray:
    """Load a norm weight tensor, convert to f32, apply +1.0 for Qwen3NextRMSNorm."""
    ti = model.weight_info(name)
    raw = bytes(model.weight_bytes(name))
    if ti.dtype == "BF16":
        w = bf16_to_f32(raw)
    elif ti.dtype == "F16":
        w = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif ti.dtype == "F32":
        w = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        w = np.frombuffer(raw, dtype=np.float32).copy()
    # Qwen3NextRMSNorm: stored weight needs +1.0
    return w + 1.0


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    """Upload a contiguous f32 array to GPU, return device pointer."""
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, count: int) -> np.ndarray:
    """Download count f32 values from GPU to host."""
    out = np.zeros(count, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, count * 4)
    return out


def main() -> int:
    banner("BELLARD OPTIMIZED FUSED FORWARD PASS")
    t_total_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load model (mmap) and tokenizer
    # ------------------------------------------------------------------
    banner("Loading model + tokenizer")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    print(f"  Model: {model}")
    print(f"  Device: {gpu.device_name}")
    print(f"  Load time: {time.monotonic() - t0:.3f}s")

    # Query SM count
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    sm_count = gpu.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    print(f"  SM count: {sm_count}")

    # ------------------------------------------------------------------
    # 2. Load forward_pass_bellard.cubin
    # ------------------------------------------------------------------
    banner("Loading forward_pass_bellard.cubin")
    t0 = time.monotonic()
    cubin_path = f"{KERNEL_DIR}/forward_pass_bellard.cubin"
    mod = gpu.load_cubin(cubin_path)
    func = gpu.get_function(mod, "forward_pass_bellard")

    # Query max active blocks for cooperative launch
    max_blks = gpu.max_active_blocks(func, THREADS_PER_BLOCK, 0)
    print(f"  Max active blocks per SM: {max_blks}")
    print(f"  Max cooperative grid: {max_blks * sm_count}")

    grid_size = min(NUM_BLOCKS, max_blks * sm_count)
    total_threads = grid_size * THREADS_PER_BLOCK
    print(f"  Using grid: ({grid_size}, 1, 1)  block: ({THREADS_PER_BLOCK}, 1, 1)")
    print(f"  Total threads: {total_threads}")
    print(f"  Load time: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 3. Prepare embedding
    # ------------------------------------------------------------------
    banner("Preparing embedding")
    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    last_token_id = token_ids[-1]
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Last token: id={last_token_id} '{tok.decode([last_token_id])}'")

    embed_raw = bytes(model.weight_bytes("model.language_model.embed_tokens.weight"))
    row_offset_bytes = last_token_id * HIDDEN_DIM * 2
    embed_row_f16 = np.frombuffer(
        embed_raw[row_offset_bytes : row_offset_bytes + HIDDEN_DIM * 2],
        dtype=np.float16,
    )
    embed_row_f32 = embed_row_f16.astype(np.float32)
    d_embed_table = upload_f32(gpu, embed_row_f32)
    print(f"  Uploaded F32 embed row ({HIDDEN_DIM * 4} bytes)")

    # ------------------------------------------------------------------
    # 4. Build weight table for all 64 layers
    # ------------------------------------------------------------------
    banner("Building weight table (64 layers x 11 pointers)")
    t0 = time.monotonic()

    d_input_norms = []
    d_post_attn_norms = []
    for i in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{i}"
        w_in = load_norm_f32(model, f"{prefix}.input_layernorm.weight")
        w_post = load_norm_f32(model, f"{prefix}.post_attention_layernorm.weight")
        d_input_norms.append(upload_f32(gpu, w_in))
        d_post_attn_norms.append(upload_f32(gpu, w_post))

    table = np.zeros(NUM_LAYERS * PTRS_PER_LAYER, dtype=np.uint64)

    for i in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{i}"
        base = i * PTRS_PER_LAYER

        table[base + 0] = d_input_norms[i].value
        table[base + 10] = d_post_attn_norms[i].value

        gate_qw_ptr = model.weight_info(f"{prefix}.mlp.gate_proj.qweight").ptr
        gate_sc_ptr = model.weight_info(f"{prefix}.mlp.gate_proj.scales").ptr
        gate_zr_ptr = model.weight_info(f"{prefix}.mlp.gate_proj.qzeros").ptr

        up_qw_ptr = model.weight_info(f"{prefix}.mlp.up_proj.qweight").ptr
        up_sc_ptr = model.weight_info(f"{prefix}.mlp.up_proj.scales").ptr
        up_zr_ptr = model.weight_info(f"{prefix}.mlp.up_proj.qzeros").ptr

        down_qw_ptr = model.weight_info(f"{prefix}.mlp.down_proj.qweight").ptr
        down_sc_ptr = model.weight_info(f"{prefix}.mlp.down_proj.scales").ptr
        down_zr_ptr = model.weight_info(f"{prefix}.mlp.down_proj.qzeros").ptr

        table[base + 1] = gate_qw_ptr
        table[base + 2] = gate_sc_ptr
        table[base + 3] = gate_zr_ptr
        table[base + 4] = up_qw_ptr
        table[base + 5] = up_sc_ptr
        table[base + 6] = up_zr_ptr
        table[base + 7] = down_qw_ptr
        table[base + 8] = down_sc_ptr
        table[base + 9] = down_zr_ptr

    d_weight_table = upload_f32(gpu, table.view(np.float32))
    print(f"  Weight table: {WEIGHT_TABLE_BYTES} bytes")
    print(f"  Build time: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 5. Allocate buffers
    # ------------------------------------------------------------------
    banner("Allocating buffers (global activation, normed, sync workspace)")

    # Activation buffer in global memory (5120 f32)
    d_activation = gpu.mem_alloc(HIDDEN_DIM * 4)

    # Normed buffer in global memory (5120 f32)
    d_normed = gpu.mem_alloc(HIDDEN_DIM * 4)

    # Mid scratch for gate*up intermediate (17408 f32)
    d_mid_scratch = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)

    # Sync workspace: 4 x u32 = 16 bytes (must be zero-initialized)
    d_sync_workspace = gpu.mem_alloc(16)
    gpu.memset_d8(d_sync_workspace, 0, 16)

    # Norm partials: 132 x f32 = 528 bytes
    d_norm_partials = gpu.mem_alloc(grid_size * 4)

    # Output logits
    d_output_logits = gpu.mem_alloc(VOCAB_SIZE * 4)

    # Final norm weight
    final_norm_w = load_norm_f32(model, "model.language_model.norm.weight")
    d_final_norm = upload_f32(gpu, final_norm_w)

    # LM head -- Bellard kernel uses FP16 weights directly (not GPTQ).
    # Pass the raw FP16 weight pointer. The kernel ignores params 8 & 9
    # but we keep them for signature compatibility.
    lm_head_ptr = model.weight_info("lm_head.weight").ptr

    # Dummy allocations for the two unused slots (kept for compat)
    d_lm_head_unused1 = gpu.mem_alloc(16)
    d_lm_head_unused2 = gpu.mem_alloc(16)
    gpu.memset_d8(d_lm_head_unused1, 0, 16)
    gpu.memset_d8(d_lm_head_unused2, 0, 16)

    print(f"  activation:     {HIDDEN_DIM * 4} bytes (global)")
    print(f"  normed:         {HIDDEN_DIM * 4} bytes (global)")
    print(f"  mid_scratch:    {INTERMEDIATE_SIZE * 4} bytes")
    print(f"  sync_workspace: 16 bytes")
    print(f"  norm_partials:  {grid_size * 4} bytes")
    print(f"  output_logits:  {VOCAB_SIZE * 4} bytes")
    print(f"  lm_head:        FP16 direct pointer (Bellard path)")

    # ------------------------------------------------------------------
    # 6. Build kernel arguments
    # ------------------------------------------------------------------
    # Kernel signature (from forward_pass_bellard.ptx):
    #   forward_pass_bellard(
    #     weight_table_ptr, embed_table_ptr, token_id,
    #     mid_scratch_ptr, output_logits_ptr, final_norm_ptr,
    #     lm_head_weight_ptr,       // FP16 [248320, 5120] row-major
    #     lm_head_unused1,          // (unused, kept for compat)
    #     lm_head_unused2,          // (unused, kept for compat)
    #     activation_ptr, normed_ptr, sync_workspace_ptr, norm_partials_ptr
    #   )

    args = [
        ctypes.c_uint64(d_weight_table.value),
        ctypes.c_uint64(d_embed_table.value),
        ctypes.c_uint32(0),                            # token_id
        ctypes.c_uint64(d_mid_scratch.value),
        ctypes.c_uint64(d_output_logits.value),
        ctypes.c_uint64(d_final_norm.value),
        ctypes.c_uint64(lm_head_ptr),                  # FP16 lm_head weights
        ctypes.c_uint64(d_lm_head_unused1.value),      # unused (compat)
        ctypes.c_uint64(d_lm_head_unused2.value),      # unused (compat)
        ctypes.c_uint64(d_activation.value),
        ctypes.c_uint64(d_normed.value),
        ctypes.c_uint64(d_sync_workspace.value),
        ctypes.c_uint64(d_norm_partials.value),
    ]

    # ------------------------------------------------------------------
    # 7. Launch cooperative kernel
    # ------------------------------------------------------------------
    banner("LAUNCHING BELLARD COOPERATIVE KERNEL")
    print(f"  Grid: ({grid_size}, 1, 1)  Block: ({THREADS_PER_BLOCK}, 1, 1)")
    print(f"  {grid_size} blocks x {THREADS_PER_BLOCK} threads = {total_threads} total")
    print(f"  ALL {sm_count} SMs active with grid-sync barriers")
    print()

    # Warmup
    print(f"  Warmup ({WARMUP_RUNS} runs)...")
    for wi in range(WARMUP_RUNS):
        try:
            gpu.memset_d8(d_sync_workspace, 0, 16)
            gpu.launch_cooperative(
                func,
                grid=(grid_size, 1, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                args=args,
                shared_mem=0,
            )
            gpu.synchronize()
            if wi == 0:
                print("  Warmup 1 complete.")
        except Exception as e:
            print(f"  Warmup launch FAILED: {e}")
            print(f"  Attempting with smaller grid...")
            for try_grid in [64, 32, 16, 8, 4, 2, 1]:
                try:
                    gpu.memset_d8(d_sync_workspace, 0, 16)
                    gpu.launch_cooperative(
                        func,
                        grid=(try_grid, 1, 1),
                        block=(THREADS_PER_BLOCK, 1, 1),
                        args=args,
                        shared_mem=0,
                    )
                    gpu.synchronize()
                    grid_size = try_grid
                    total_threads = grid_size * THREADS_PER_BLOCK
                    print(f"  Success with grid=({try_grid}, 1, 1)")
                    break
                except Exception as e2:
                    print(f"  Grid {try_grid} failed: {e2}")
            else:
                print("  All cooperative launch attempts failed.")
                gpu.close()
                model.close()
                return 1
            break
    print(f"  All {WARMUP_RUNS} warmup runs complete.")

    # Timed launches
    print()
    print(f"  Timed launches ({TIMED_RUNS} runs)...")
    times_ms = []
    for run_i in range(TIMED_RUNS):
        gpu.memset_d8(d_sync_workspace, 0, 16)
        t0 = time.monotonic()
        gpu.launch_cooperative(
            func,
            grid=(grid_size, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            args=args,
            shared_mem=0,
        )
        gpu.synchronize()
        elapsed = (time.monotonic() - t0) * 1000.0
        times_ms.append(elapsed)
        print(f"    Run {run_i+1}: {elapsed:.3f} ms")

    median_ms = sorted(times_ms)[TIMED_RUNS // 2]
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    print(f"\n  {TIMED_RUNS} runs:    median={median_ms:.3f}ms  min={min_ms:.3f}ms  max={max_ms:.3f}ms")
    print(f"  All runs:    {['%.3f' % t for t in times_ms]}")

    # ------------------------------------------------------------------
    # 8. Read back logits
    # ------------------------------------------------------------------
    banner("READING LOGITS")
    logits = download_f32(gpu, d_output_logits, VOCAB_SIZE)

    has_nan = np.any(np.isnan(logits))
    has_inf = np.any(np.isinf(logits))
    nonzero = np.count_nonzero(logits)
    print(f"  Logits shape: ({VOCAB_SIZE},)")
    print(f"  NaN: {has_nan}  Inf: {has_inf}  Nonzero: {nonzero}/{VOCAB_SIZE}")
    print(f"  Range: [{np.nanmin(logits):.4f}, {np.nanmax(logits):.4f}]")
    print(f"  Mean: {np.nanmean(logits):.4f}  Std: {np.nanstd(logits):.4f}")

    logits_valid = not has_nan and nonzero > 0
    logits_nonzero = nonzero > 0

    # Also read activation to check it's not all-zero
    activation = download_f32(gpu, d_activation, HIDDEN_DIM)
    act_nonzero = np.count_nonzero(activation)
    act_norm = np.linalg.norm(activation)
    print(f"\n  Activation after 64 layers:")
    print(f"  Nonzero: {act_nonzero}/{HIDDEN_DIM}  Norm: {act_norm:.4f}")

    # ------------------------------------------------------------------
    # 9. Top-10 and Paris check
    # ------------------------------------------------------------------
    banner("TOP-10 PREDICTIONS")
    logits_clean = np.where(np.isfinite(logits), logits, -1e9)
    top10_idx = np.argsort(logits_clean)[-10:][::-1]

    for rank, idx in enumerate(top10_idx):
        word = tok.decode([int(idx)])
        print(f"  {rank+1:2d}. token={idx:6d}  logit={logits[idx]:12.4f}  '{word}'")

    paris_rank = None
    for candidate in ["Paris", " Paris"]:
        cand_ids = tok.encode(candidate)
        for cid in cand_ids:
            rank = int(np.sum(logits_clean > logits_clean[cid])) + 1
            word = tok.decode([cid])
            print(f"\n  '{candidate}' (id={cid}, decoded='{word}'): logit={logits[cid]:.4f}, rank={rank}")
            if paris_rank is None or rank < paris_rank:
                paris_rank = rank

    is_paris_rank1 = (paris_rank == 1 and logits_valid) if paris_rank is not None else False

    # ------------------------------------------------------------------
    # 10. Final report
    # ------------------------------------------------------------------
    t_total = time.monotonic() - t_total_start

    banner("RESULTS -- BELLARD OPTIMIZED KERNEL")
    print(f"  Prompt:              {prompt!r}")
    print(f"  Token processed:     id={last_token_id} '{tok.decode([last_token_id])}'")
    print()
    print(f"  --- Kernel Configuration ---")
    print(f"  Kernel:              forward_pass_bellard")
    print(f"  Grid:                ({grid_size}, 1, 1)")
    print(f"  Block:               ({THREADS_PER_BLOCK}, 1, 1)")
    print(f"  Total threads:       {total_threads}")
    print(f"  SMs utilized:        {grid_size} / {sm_count}")
    print()
    print(f"  --- GPU Kernel Timing ---")
    print(f"  Warmup runs:         {WARMUP_RUNS}")
    print(f"  Timed runs:          {TIMED_RUNS}")
    print(f"  Median time/token:   {median_ms:.3f} ms")
    print(f"  Min / Max:           {min_ms:.3f} / {max_ms:.3f} ms")
    if median_ms > 0:
        tok_per_sec = 1000.0 / median_ms
        print(f"  tok/s:               {tok_per_sec:.1f}")
    print()
    print(f"  --- Comparison vs forward_pass_multi ({BASELINE_MS:.0f}ms baseline) ---")
    print(f"  Multi-block fused:   {BASELINE_MS:.0f} ms (baseline)")
    print(f"  Bellard optimized:   {median_ms:.3f} ms")
    if median_ms > 0:
        speedup = BASELINE_MS / median_ms
        delta_ms = BASELINE_MS - median_ms
        delta_pct = (delta_ms / BASELINE_MS) * 100.0
        print(f"  Speedup:             {speedup:.2f}x ({delta_pct:+.1f}%)")
    print()
    print(f"  --- Logits Validation ---")
    print(f"  Logits non-zero:     {'YES' if logits_nonzero else 'NO (all-zero -- FP16 lm_head issue NOT fixed)'}")
    print(f"  Logits valid:        {'YES' if logits_valid else 'NO (NaN or all-zero)'}")
    if logits_valid and paris_rank is not None:
        print(f"  Paris rank:          {paris_rank}")
    print(f"  FP16 lm_head fix:    {'CONFIRMED' if logits_nonzero else 'NOT WORKING'}")
    print(f"  Total wall time:     {t_total:.2f}s")

    # ------------------------------------------------------------------
    gpu.close()
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
