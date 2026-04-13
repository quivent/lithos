#!/usr/bin/env python3
"""
Benchmark: forward_pass_carmack.cubin wall-clock timing with real weights.

Measures:
  - Wall-clock time per single forward pass (one token), including Python overhead
  - Wall-clock time for 5-token prefill (5 sequential launches)
  - Actual tok/s

Uses the same weight-loading and argument setup as generate_multi_fused.py,
but launches the Carmack kernel instead of forward_pass_multi.

Run:
    python3 /home/ubuntu/lithos/src/benchmark_carmack.py
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
# Architecture constants (must match forward_pass_carmack kernel)
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
    return w + 1.0


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, count * 4)
    return out


def main() -> int:
    banner("CARMACK KERNEL WALL-CLOCK BENCHMARK")
    t_total_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    banner("Loading model + tokenizer")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    print(f"  Model: {model}")
    print(f"  Device: {gpu.device_name}")
    print(f"  Load time: {time.monotonic() - t0:.3f}s")

    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    sm_count = gpu.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    print(f"  SM count: {sm_count}")

    # ------------------------------------------------------------------
    # 2. Load forward_pass_carmack.cubin
    # ------------------------------------------------------------------
    banner("Loading forward_pass_carmack.cubin")
    t0 = time.monotonic()
    cubin_path = f"{KERNEL_DIR}/forward_pass_carmack.cubin"
    mod = gpu.load_cubin(cubin_path)
    func = gpu.get_function(mod, "forward_pass_carmack")

    max_blks = gpu.max_active_blocks(func, THREADS_PER_BLOCK, 0)
    print(f"  Max active blocks per SM: {max_blks}")
    print(f"  Max cooperative grid: {max_blks * sm_count}")

    grid_size = min(NUM_BLOCKS, max_blks * sm_count)
    total_threads = grid_size * THREADS_PER_BLOCK
    print(f"  Using grid: ({grid_size}, 1, 1)  block: ({THREADS_PER_BLOCK}, 1, 1)")
    print(f"  Total threads: {total_threads}")
    print(f"  Load time: {time.monotonic() - t0:.3f}s")

    # Also load the original for comparison
    banner("Loading forward_pass_multi.cubin (for comparison)")
    orig_mod = gpu.load_cubin(f"{KERNEL_DIR}/forward_pass_multi.cubin")
    orig_func = gpu.get_function(orig_mod, "forward_pass_multi")
    orig_max_blks = gpu.max_active_blocks(orig_func, THREADS_PER_BLOCK, 0)
    orig_grid = min(NUM_BLOCKS, orig_max_blks * sm_count)
    print(f"  Original grid: ({orig_grid}, 1, 1)")

    # ------------------------------------------------------------------
    # 3. Prepare embedding (last token of prompt)
    # ------------------------------------------------------------------
    banner("Preparing embedding")
    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    last_token_id = token_ids[-1]
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Last token: id={last_token_id} '{tok.decode([last_token_id])}'")

    # Pre-extract all token embeddings for prefill
    embed_raw = bytes(model.weight_bytes("model.language_model.embed_tokens.weight"))
    embed_rows_f32 = []
    for tid in token_ids:
        row_off = tid * HIDDEN_DIM * 2
        row_f16 = np.frombuffer(
            embed_raw[row_off : row_off + HIDDEN_DIM * 2], dtype=np.float16
        )
        embed_rows_f32.append(row_f16.astype(np.float32))

    # Upload the last token embedding (for single-token benchmark)
    d_embed_table = upload_f32(gpu, embed_rows_f32[-1])

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

        table[base + 1] = model.weight_info(f"{prefix}.mlp.gate_proj.qweight").ptr
        table[base + 2] = model.weight_info(f"{prefix}.mlp.gate_proj.scales").ptr
        table[base + 3] = model.weight_info(f"{prefix}.mlp.gate_proj.qzeros").ptr
        table[base + 4] = model.weight_info(f"{prefix}.mlp.up_proj.qweight").ptr
        table[base + 5] = model.weight_info(f"{prefix}.mlp.up_proj.scales").ptr
        table[base + 6] = model.weight_info(f"{prefix}.mlp.up_proj.qzeros").ptr
        table[base + 7] = model.weight_info(f"{prefix}.mlp.down_proj.qweight").ptr
        table[base + 8] = model.weight_info(f"{prefix}.mlp.down_proj.scales").ptr
        table[base + 9] = model.weight_info(f"{prefix}.mlp.down_proj.qzeros").ptr

    d_weight_table = upload_f32(gpu, table.view(np.float32))
    print(f"  Weight table: {WEIGHT_TABLE_BYTES} bytes")
    print(f"  Build time: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 5. Allocate buffers
    # ------------------------------------------------------------------
    banner("Allocating buffers")

    d_activation = gpu.mem_alloc(HIDDEN_DIM * 4)
    d_normed = gpu.mem_alloc(HIDDEN_DIM * 4)
    d_mid_scratch = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
    d_sync_workspace = gpu.mem_alloc(16)
    gpu.memset_d8(d_sync_workspace, 0, 16)
    d_norm_partials = gpu.mem_alloc(grid_size * 4)
    d_output_logits = gpu.mem_alloc(VOCAB_SIZE * 4)

    final_norm_w = load_norm_f32(model, "model.language_model.norm.weight")
    d_final_norm = upload_f32(gpu, final_norm_w)

    lm_head_ptr = model.weight_info("lm_head.weight").ptr
    lm_head_scales_size = VOCAB_SIZE * 40 * 2
    lm_head_zeros_size = VOCAB_SIZE * 40 * 4
    d_lm_head_scales = gpu.mem_alloc(lm_head_scales_size)
    d_lm_head_zeros = gpu.mem_alloc(lm_head_zeros_size)
    gpu.memset_d8(d_lm_head_scales, 0, lm_head_scales_size)
    gpu.memset_d8(d_lm_head_zeros, 0, lm_head_zeros_size)

    print(f"  All buffers allocated")

    # ------------------------------------------------------------------
    # 6. Build kernel arguments
    # ------------------------------------------------------------------
    # Same signature as forward_pass_multi:
    #   (weight_table_ptr, embed_table_ptr, token_id,
    #    mid_scratch_ptr, output_logits_ptr, final_norm_ptr,
    #    lm_head_qw_ptr, lm_head_scales_ptr, lm_head_zeros_ptr,
    #    activation_ptr, normed_ptr, sync_workspace_ptr, norm_partials_ptr)

    def make_args(d_embed):
        return [
            ctypes.c_uint64(d_weight_table.value),
            ctypes.c_uint64(d_embed.value),
            ctypes.c_uint32(0),  # token_id (unused when embed is pre-extracted)
            ctypes.c_uint64(d_mid_scratch.value),
            ctypes.c_uint64(d_output_logits.value),
            ctypes.c_uint64(d_final_norm.value),
            ctypes.c_uint64(lm_head_ptr),
            ctypes.c_uint64(d_lm_head_scales.value),
            ctypes.c_uint64(d_lm_head_zeros.value),
            ctypes.c_uint64(d_activation.value),
            ctypes.c_uint64(d_normed.value),
            ctypes.c_uint64(d_sync_workspace.value),
            ctypes.c_uint64(d_norm_partials.value),
        ]

    def reset_state():
        """Reset sync workspace and activation buffers between launches."""
        gpu.memset_d8(d_sync_workspace, 0, 16)
        gpu.memset_d8(d_activation, 0, HIDDEN_DIM * 4)
        gpu.memset_d8(d_mid_scratch, 0, INTERMEDIATE_SIZE * 4)
        gpu.memset_d8(d_output_logits, 0, VOCAB_SIZE * 4)
        gpu.memset_d8(d_normed, 0, HIDDEN_DIM * 4)
        gpu.memset_d8(d_norm_partials, 0, grid_size * 4)

    args_carmack = make_args(d_embed_table)
    args_orig = make_args(d_embed_table)

    # ------------------------------------------------------------------
    # 7. Warmup
    # ------------------------------------------------------------------
    banner("WARMUP")
    for label, kernel_func, kernel_grid, kernel_args in [
        ("Carmack", func, grid_size, args_carmack),
        ("Original", orig_func, orig_grid, args_orig),
    ]:
        print(f"  Warming up {label}...")
        for _ in range(3):
            reset_state()
            try:
                gpu.launch_cooperative(
                    kernel_func,
                    grid=(kernel_grid, 1, 1),
                    block=(THREADS_PER_BLOCK, 1, 1),
                    args=kernel_args,
                    shared_mem=0,
                )
                gpu.synchronize()
            except Exception as e:
                print(f"    {label} warmup failed: {e}")
                break
        print(f"    {label} warmup done")

    # ------------------------------------------------------------------
    # 8. BENCHMARK: Single forward pass (one token)
    # ------------------------------------------------------------------
    banner("BENCHMARK: Single Forward Pass (1 token)")
    NUM_RUNS = 20

    for label, kernel_func, kernel_grid, kernel_args in [
        ("Carmack", func, grid_size, args_carmack),
        ("Original", orig_func, orig_grid, args_orig),
    ]:
        times_ms = []
        for _ in range(NUM_RUNS):
            reset_state()
            gpu.synchronize()

            t_start = time.perf_counter()
            gpu.launch_cooperative(
                kernel_func,
                grid=(kernel_grid, 1, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                args=kernel_args,
                shared_mem=0,
            )
            gpu.synchronize()
            t_end = time.perf_counter()

            times_ms.append((t_end - t_start) * 1000.0)

        times_ms.sort()
        median_ms = times_ms[NUM_RUNS // 2]
        min_ms = times_ms[0]
        max_ms = times_ms[-1]
        avg_ms = sum(times_ms) / len(times_ms)
        p10_ms = times_ms[NUM_RUNS // 10]
        tok_s = 1000.0 / median_ms if median_ms > 0 else 0

        print(f"\n  {label} ({NUM_RUNS} runs):")
        print(f"    Median:  {median_ms:.3f} ms")
        print(f"    Min:     {min_ms:.3f} ms")
        print(f"    Max:     {max_ms:.3f} ms")
        print(f"    Avg:     {avg_ms:.3f} ms")
        print(f"    P10:     {p10_ms:.3f} ms")
        print(f"    tok/s:   {tok_s:.1f} (from median)")
        print(f"    All:     {['%.2f' % t for t in times_ms]}")

    # ------------------------------------------------------------------
    # 9. BENCHMARK: 5-token prefill (sequential launches)
    # ------------------------------------------------------------------
    banner("BENCHMARK: 5-Token Prefill (sequential launches)")

    # Upload all 5 token embeddings
    d_token_embeds = []
    for i, row in enumerate(embed_rows_f32):
        d_token_embeds.append(upload_f32(gpu, row))

    NUM_PREFILL_RUNS = 10

    for label, kernel_func, kernel_grid in [
        ("Carmack", func, grid_size),
        ("Original", orig_func, orig_grid),
    ]:
        times_ms = []
        for _ in range(NUM_PREFILL_RUNS):
            # Full reset before each prefill sequence
            reset_state()
            gpu.synchronize()

            t_start = time.perf_counter()
            for tok_idx in range(len(token_ids)):
                # Reset sync workspace between tokens (required for cooperative kernel)
                gpu.memset_d8(d_sync_workspace, 0, 16)
                gpu.memset_d8(d_mid_scratch, 0, INTERMEDIATE_SIZE * 4)
                gpu.memset_d8(d_normed, 0, HIDDEN_DIM * 4)
                gpu.memset_d8(d_norm_partials, 0, grid_size * 4)

                token_args = make_args(d_token_embeds[tok_idx])

                gpu.launch_cooperative(
                    kernel_func,
                    grid=(kernel_grid, 1, 1),
                    block=(THREADS_PER_BLOCK, 1, 1),
                    args=token_args,
                    shared_mem=0,
                )
                gpu.synchronize()
            t_end = time.perf_counter()

            times_ms.append((t_end - t_start) * 1000.0)

        times_ms.sort()
        n_tokens = len(token_ids)
        median_ms = times_ms[NUM_PREFILL_RUNS // 2]
        min_ms = times_ms[0]
        max_ms = times_ms[-1]
        avg_ms = sum(times_ms) / len(times_ms)
        per_token_ms = median_ms / n_tokens
        tok_s = n_tokens * 1000.0 / median_ms if median_ms > 0 else 0

        print(f"\n  {label} ({NUM_PREFILL_RUNS} runs, {n_tokens} tokens each):")
        print(f"    Total median:    {median_ms:.3f} ms")
        print(f"    Total min:       {min_ms:.3f} ms")
        print(f"    Total max:       {max_ms:.3f} ms")
        print(f"    Per-token avg:   {per_token_ms:.3f} ms")
        print(f"    tok/s:           {tok_s:.1f} (from median)")
        print(f"    All:             {['%.2f' % t for t in times_ms]}")

    # ------------------------------------------------------------------
    # 10. Correctness check: read logits from last Carmack run
    # ------------------------------------------------------------------
    banner("CORRECTNESS CHECK (Carmack, last token)")

    # Do one clean Carmack run with last token
    reset_state()
    gpu.launch_cooperative(
        func,
        grid=(grid_size, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        args=make_args(d_embed_table),
        shared_mem=0,
    )
    gpu.synchronize()

    logits = download_f32(gpu, d_output_logits, VOCAB_SIZE)
    has_nan = np.any(np.isnan(logits))
    has_inf = np.any(np.isinf(logits))
    nonzero = np.count_nonzero(logits)
    print(f"  Logits: NaN={has_nan}  Inf={has_inf}  Nonzero={nonzero}/{VOCAB_SIZE}")
    print(f"  Range: [{np.nanmin(logits):.4f}, {np.nanmax(logits):.4f}]")

    logits_clean = np.where(np.isfinite(logits), logits, -1e9)
    top10_idx = np.argsort(logits_clean)[-10:][::-1]
    print(f"\n  Top-10 predictions:")
    for rank, idx in enumerate(top10_idx):
        word = tok.decode([int(idx)])
        print(f"    {rank+1:2d}. token={idx:6d}  logit={logits[idx]:12.4f}  '{word}'")

    paris_rank = None
    for candidate in ["Paris", " Paris"]:
        cand_ids = tok.encode(candidate)
        for cid in cand_ids:
            rank = int(np.sum(logits_clean > logits_clean[cid])) + 1
            word = tok.decode([cid])
            print(f"  '{candidate}' (id={cid}, decoded='{word}'): logit={logits[cid]:.4f}, rank={rank}")
            if paris_rank is None or rank < paris_rank:
                paris_rank = rank

    # Also do one clean original run for comparison
    banner("CORRECTNESS CHECK (Original, last token)")
    reset_state()
    gpu.launch_cooperative(
        orig_func,
        grid=(orig_grid, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        args=make_args(d_embed_table),
        shared_mem=0,
    )
    gpu.synchronize()

    logits_orig = download_f32(gpu, d_output_logits, VOCAB_SIZE)
    logits_orig_clean = np.where(np.isfinite(logits_orig), logits_orig, -1e9)
    top10_orig = np.argsort(logits_orig_clean)[-5:][::-1]
    print(f"  Top-5 (original):")
    for rank, idx in enumerate(top10_orig):
        word = tok.decode([int(idx)])
        print(f"    {rank+1:2d}. token={idx:6d}  logit={logits_orig[idx]:12.4f}  '{word}'")

    # Compare Carmack vs Original logits
    diff = np.abs(logits_clean - logits_orig_clean)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"\n  Carmack vs Original logit difference:")
    print(f"    Max abs diff:  {max_diff:.6f}")
    print(f"    Mean abs diff: {mean_diff:.6f}")

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------
    t_total = time.monotonic() - t_total_start
    banner("SUMMARY")
    print(f"  Kernel:              forward_pass_carmack.cubin")
    print(f"  Grid:                ({grid_size}, 1, 1) x ({THREADS_PER_BLOCK}, 1, 1)")
    print(f"  Model:               Qwen3.5-27B GPTQ W4A16")
    print(f"  Scope:               MLP-only (64 layers fused), no DeltaNet/attention")
    print(f"  Note:                This is the MLP-only forward pass kernel.")
    print(f"                       Real inference also needs per-layer DeltaNet/attention.")
    print(f"  Total benchmark time: {t_total:.2f}s")
    print()
    print(f"  For hybrid inference (Carmack MLP + per-layer attention):")
    print(f"    The Carmack kernel handles all 64 MLP layers in one launch.")
    print(f"    Attention/DeltaNet would add ~150-200ms per token (from generate_paris.py profile).")
    print(f"    Expected hybrid tok/s depends on attention overhead.")

    # Cleanup
    for d in d_token_embeds:
        gpu.mem_free(d)
    gpu.close()
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
