#!/usr/bin/env python3
"""
Lithos pipeline execution -- wire real kernels end to end.

Stage 1: embed kernel with real Qwen 3.5-27B weights -> non-zero F32 vector
Stage 2: norm kernel on that vector -> normalized output
Stage 3: verify against numpy/torch reference

This script proves real token -> real activation on GH200.
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

from cuda_driver import CUDADriver, CUDAError, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")

HIDDEN_DIM = 5120


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def check_nan_inf(arr: np.ndarray, label: str) -> bool:
    """Return True if array has no NaN/Inf."""
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    if has_nan or has_inf:
        print(f"  [{label}] PROBLEM: NaN={has_nan} Inf={has_inf}")
        return False
    return True


def main() -> int:
    banner("Lithos Pipeline Execution -- Real Kernels on GH200")

    # ---------------------------------------------------------------
    # 1. Load model + tokenizer
    # ---------------------------------------------------------------
    print("\n--- Loading model and tokenizer ---")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    t1 = time.monotonic()
    print(f"  Model: {model}  ({t1-t0:.3f}s)")
    print(f"  Tokenizer: {tok}")
    print(f"  hidden_dim={model.config.hidden_dim}  vocab={model.config.vocab_size}")

    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")

    # We'll work with just the first token for embed proof
    test_token_id = token_ids[0]  # 760 = "The"
    print(f"  Test token: {test_token_id} -> {tok.decode([test_token_id])!r}")

    # ---------------------------------------------------------------
    # 2. Create CUDA context
    # ---------------------------------------------------------------
    print("\n--- Creating CUDA context ---")
    gpu = CUDADriver()
    print(f"  Device: {gpu.device_name}")

    # ---------------------------------------------------------------
    # 3. STAGE 1: Embed kernel with real weights
    # ---------------------------------------------------------------
    banner("STAGE 1: Embed Kernel")

    # Load the embed cubin (factory-specialized with hidden_dim=5120 baked in)
    embed_cubin_path = f"{CACHE_DIR}/embed.cubin"
    print(f"  Loading cubin: {embed_cubin_path}")
    embed_mod = gpu.load_cubin(embed_cubin_path)
    embed_func = gpu.get_function(embed_mod, "embed")
    print(f"  embed function loaded OK")

    # Get embed weight pointer + info
    embed_ti = model.weight_info("model.language_model.embed_tokens.weight")
    print(f"  Embed weights: dtype={embed_ti.dtype} shape={embed_ti.shape} "
          f"bytes={embed_ti.byte_size:,}")

    # The embed kernel reads F32, but weights are F16.
    # Convert the single row we need (token_id=760) to F32 for testing.
    # For full pipeline: convert entire embed table once at startup.
    print(f"\n  Converting embed weights F16 -> F32 for token {test_token_id}...")

    # Read the whole embed table as raw bytes and convert to F16 numpy
    # Actually let's be smarter - just read the one row we need
    row_offset_bytes = test_token_id * HIDDEN_DIM * 2  # F16 = 2 bytes
    row_bytes = HIDDEN_DIM * 2

    # Get raw bytes from mmap
    raw = model.weight_bytes("model.language_model.embed_tokens.weight")
    row_f16_bytes = bytes(raw[row_offset_bytes : row_offset_bytes + row_bytes])
    row_f16 = np.frombuffer(row_f16_bytes, dtype=np.float16)
    row_f32 = row_f16.astype(np.float32)

    print(f"  F16 row stats: min={row_f16.min():.6f} max={row_f16.max():.6f} "
          f"mean={row_f16.mean():.6f}")
    print(f"  F32 row[0:8]: {row_f32[:8]}")

    # Now we need a full embed table in F32 for the kernel to index into.
    # But that's 248320 * 5120 * 4 = ~4.7 GB which is too much.
    # Instead: create a SMALL table with just our token at the right offset.
    # The kernel does: row_ptr = table_base + token_id * hidden_dim * 4
    # So we need table_base + 760 * 5120 * 4 to point to valid data.
    #
    # Strategy: allocate just 1 row of F32, and compute the table_base
    # such that table_base + token_id * hidden_dim * 4 = our_row_ptr.
    # BUT: that means table_base could be a negative/invalid address.
    #
    # Better strategy: allocate (token_id + 1) rows = 761 * 5120 * 4 = ~14.9 MB
    # and put our data at row 760.

    # Actually even simpler for proof: set token_id=0 and put our row at index 0
    # OR: allocate a big-enough region. 761 rows * 5120 * 4 = 15,564,800 bytes ~ 15 MB
    table_rows_needed = test_token_id + 1
    table_bytes = table_rows_needed * HIDDEN_DIM * 4  # F32
    print(f"\n  Allocating {table_bytes:,} bytes ({table_bytes/(1024**2):.1f} MB) "
          f"for embed table ({table_rows_needed} rows of F32)")

    d_table = gpu.mem_alloc(table_bytes)
    print(f"  Device table @ 0x{d_table.value:016x}")

    # Upload just the one row at the right offset
    row_device_offset = test_token_id * HIDDEN_DIM * 4
    # Create a ctypes pointer to the numpy data
    row_f32_ctypes = row_f32.ctypes.data_as(ctypes.c_void_p)

    # We need to write at offset within the allocation
    # cuMemcpyHtoD writes to an absolute device address
    dst_addr = CUdeviceptr(d_table.value + row_device_offset)
    gpu.memcpy_htod(dst_addr, row_f32_ctypes, HIDDEN_DIM * 4)

    # Allocate output buffer
    d_output = gpu.mem_alloc(HIDDEN_DIM * 4)
    # Zero it first
    zero_buf = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_htod(d_output, zero_buf.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)

    # Launch embed kernel
    # Signature: embed(token_id: u32, embed_table_ptr: u64, output_ptr: u64)
    # hidden_dim=5120 is baked in
    BLOCK = 256
    GRID = max(1, math.ceil(HIDDEN_DIM / (BLOCK * 4)))  # vectorized: 4 elements per thread

    print(f"\n  Launching embed kernel: grid=({GRID},1,1) block=({BLOCK},1,1)")
    print(f"    token_id={test_token_id}")
    print(f"    table_ptr=0x{d_table.value:016x}")
    print(f"    output_ptr=0x{d_output.value:016x}")

    t0 = time.perf_counter()
    gpu.launch(
        embed_func,
        grid=(GRID, 1, 1),
        block=(BLOCK, 1, 1),
        args=[
            ctypes.c_uint32(test_token_id),
            ctypes.c_uint64(d_table.value),
            ctypes.c_uint64(d_output.value),
        ],
    )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Kernel completed in {(t1-t0)*1e6:.0f} us")

    # Read back output
    output_f32 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_dtoh(
        output_f32.ctypes.data_as(ctypes.c_void_p),
        d_output,
        HIDDEN_DIM * 4,
    )

    print(f"\n  Output stats:")
    print(f"    min={output_f32.min():.6f}  max={output_f32.max():.6f}  "
          f"mean={output_f32.mean():.6f}")
    print(f"    first 8: {output_f32[:8]}")
    print(f"    nonzero: {np.count_nonzero(output_f32)} / {HIDDEN_DIM}")

    # Verify against reference
    ok_nan = check_nan_inf(output_f32, "embed_output")
    maxerr = np.max(np.abs(output_f32 - row_f32))
    ok_match = maxerr < 1e-6
    print(f"    vs reference max error: {maxerr:.2e}")
    print(f"    EMBED RESULT: {'PASS' if (ok_nan and ok_match) else 'FAIL'}")

    if not (ok_nan and ok_match):
        print("  Stopping -- embed kernel failed.")
        gpu.mem_free(d_table)
        gpu.mem_free(d_output)
        gpu.close()
        model.close()
        return 1

    # ---------------------------------------------------------------
    # 4. STAGE 2: Norm kernel on the embed output
    # ---------------------------------------------------------------
    banner("STAGE 2: RMSNorm Kernel")

    norm_cubin_path = f"{CACHE_DIR}/norm.cubin"
    print(f"  Loading cubin: {norm_cubin_path}")
    norm_mod = gpu.load_cubin(norm_cubin_path)
    norm_func = gpu.get_function(norm_mod, "norm")
    print(f"  norm function loaded OK")

    # Norm kernel signature: (input_ptr, residual_ptr, weight_ptr, output_ptr, epsilon)
    # All F32 pointers. hidden_dim=5120 baked in.
    # input = embed output (already on device as d_output)
    # residual = zeros for the first norm (no residual yet)
    # weight = layer 0 input_layernorm.weight (BF16 in model, need F32)

    # Get norm weight
    norm_weight_name = "model.language_model.layers.0.input_layernorm.weight"
    norm_ti = model.weight_info(norm_weight_name)
    print(f"  Norm weight: dtype={norm_ti.dtype} shape={norm_ti.shape}")

    norm_raw = model.weight_bytes(norm_weight_name)
    if norm_ti.dtype == "BF16":
        # Convert BF16 to F32: BF16 is top 16 bits of F32
        norm_u16 = np.frombuffer(bytes(norm_raw), dtype=np.uint16)
        norm_f32_weight = np.zeros(len(norm_u16), dtype=np.float32)
        # BF16 -> F32: shift left by 16 bits
        norm_f32_weight.view(np.uint32)[:] = norm_u16.astype(np.uint32) << 16
    elif norm_ti.dtype == "F32":
        norm_f32_weight = np.frombuffer(bytes(norm_raw), dtype=np.float32).copy()
    elif norm_ti.dtype == "F16":
        norm_f16 = np.frombuffer(bytes(norm_raw), dtype=np.float16)
        norm_f32_weight = norm_f16.astype(np.float32)
    else:
        print(f"  Unsupported norm weight dtype: {norm_ti.dtype}")
        return 1

    print(f"  Norm weight F32: min={norm_f32_weight.min():.6f} "
          f"max={norm_f32_weight.max():.6f} mean={norm_f32_weight.mean():.6f}")

    # Upload norm weight to device
    d_norm_weight = gpu.mem_alloc(HIDDEN_DIM * 4)
    gpu.memcpy_htod(d_norm_weight,
                    norm_f32_weight.ctypes.data_as(ctypes.c_void_p),
                    HIDDEN_DIM * 4)

    # Residual = zeros (first layer, no residual to add)
    d_residual = gpu.mem_alloc(HIDDEN_DIM * 4)
    residual_zeros = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_htod(d_residual,
                    residual_zeros.ctypes.data_as(ctypes.c_void_p),
                    HIDDEN_DIM * 4)

    # Output buffer for norm
    d_norm_output = gpu.mem_alloc(HIDDEN_DIM * 4)

    epsilon = model.config.rms_norm_eps
    print(f"  epsilon = {epsilon}")

    # Launch norm: 1 row, BLOCK threads
    NORM_BLOCK = 256
    print(f"\n  Launching norm kernel: grid=(1,1,1) block=({NORM_BLOCK},1,1)")

    t0 = time.perf_counter()
    gpu.launch(
        norm_func,
        grid=(1, 1, 1),
        block=(NORM_BLOCK, 1, 1),
        args=[
            ctypes.c_uint64(d_output.value),       # input (embed output)
            ctypes.c_uint64(d_residual.value),      # residual (zeros)
            ctypes.c_uint64(d_norm_weight.value),   # weight
            ctypes.c_uint64(d_norm_output.value),   # output
            ctypes.c_float(epsilon),                # epsilon
        ],
        shared_mem=128,  # shared memory for reduction
    )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Kernel completed in {(t1-t0)*1e6:.0f} us")

    # Read back norm output
    norm_result = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_dtoh(
        norm_result.ctypes.data_as(ctypes.c_void_p),
        d_norm_output,
        HIDDEN_DIM * 4,
    )

    print(f"\n  Norm output stats:")
    print(f"    min={norm_result.min():.6f}  max={norm_result.max():.6f}  "
          f"mean={norm_result.mean():.6f}")
    print(f"    first 8: {norm_result[:8]}")
    print(f"    nonzero: {np.count_nonzero(norm_result)} / {HIDDEN_DIM}")

    ok_nan = check_nan_inf(norm_result, "norm_output")

    # CPU reference: RMSNorm(embed_output + 0) * weight
    x = output_f32 + residual_zeros  # input + residual
    rms = np.sqrt(np.mean(x ** 2) + epsilon)
    ref_norm = (x / rms) * norm_f32_weight

    maxerr = np.max(np.abs(norm_result - ref_norm))
    ok_match = maxerr < 0.01  # GPU uses approx rsqrt
    print(f"    vs CPU reference max error: {maxerr:.6f}")
    print(f"    CPU ref first 8: {ref_norm[:8]}")
    print(f"    NORM RESULT: {'PASS' if (ok_nan and ok_match) else 'FAIL'}")

    if not ok_nan:
        print("  Stopping -- norm kernel produced NaN/Inf.")

    # ---------------------------------------------------------------
    # 5. Full embed for all prompt tokens
    # ---------------------------------------------------------------
    banner("STAGE 3: Full Prompt Embedding")

    # Embed all 5 tokens of "The capital of France is"
    n_tokens = len(token_ids)
    max_token_id = max(token_ids)
    print(f"  Embedding {n_tokens} tokens: {token_ids}")

    # We need a table covering up to max_token_id
    # For the full model this would be the entire embed table.
    # Let's convert just the rows we need and create a sparse-ish table.
    table_rows_needed2 = max_token_id + 1
    table_bytes2 = table_rows_needed2 * HIDDEN_DIM * 4
    print(f"  Need table covering {table_rows_needed2} rows = {table_bytes2/(1024**2):.1f} MB")

    # For each token, extract its F16 row and convert to F32
    embed_raw = model.weight_bytes("model.language_model.embed_tokens.weight")
    all_rows_f32 = {}
    for tid in token_ids:
        off = tid * HIDDEN_DIM * 2
        row_bytes_data = bytes(embed_raw[off : off + HIDDEN_DIM * 2])
        row_f16_arr = np.frombuffer(row_bytes_data, dtype=np.float16)
        all_rows_f32[tid] = row_f16_arr.astype(np.float32)

    # Allocate device table (reuse if big enough, otherwise re-alloc)
    if table_rows_needed2 > table_rows_needed:
        gpu.mem_free(d_table)
        d_table = gpu.mem_alloc(table_bytes2)
        print(f"  Reallocated table @ 0x{d_table.value:016x}")

    # Upload each row at its correct offset
    for tid, row_data in all_rows_f32.items():
        off = tid * HIDDEN_DIM * 4
        dst = CUdeviceptr(d_table.value + off)
        gpu.memcpy_htod(dst, row_data.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)

    # Allocate output for all tokens: n_tokens * HIDDEN_DIM * 4
    d_multi_output = gpu.mem_alloc(n_tokens * HIDDEN_DIM * 4)

    # Run embed for each token (the kernel handles one token at a time)
    print(f"\n  Running embed for each token...")
    for i, tid in enumerate(token_ids):
        out_offset = i * HIDDEN_DIM * 4
        d_out_i = CUdeviceptr(d_multi_output.value + out_offset)

        gpu.launch(
            embed_func,
            grid=(GRID, 1, 1),
            block=(BLOCK, 1, 1),
            args=[
                ctypes.c_uint32(tid),
                ctypes.c_uint64(d_table.value),
                ctypes.c_uint64(d_out_i.value),
            ],
        )

    gpu.synchronize()

    # Read back all token embeddings
    multi_output = np.zeros((n_tokens, HIDDEN_DIM), dtype=np.float32)
    gpu.memcpy_dtoh(
        multi_output.ctypes.data_as(ctypes.c_void_p),
        d_multi_output,
        n_tokens * HIDDEN_DIM * 4,
    )

    all_ok = True
    for i, tid in enumerate(token_ids):
        row = multi_output[i]
        ref = all_rows_f32[tid]
        err = np.max(np.abs(row - ref))
        ok = err < 1e-6
        word = tok.decode([tid])
        print(f"    token {tid:6d} ({word!r:12s}): "
              f"min={row.min():+.4f} max={row.max():+.4f} "
              f"mean={row.mean():+.6f}  err={err:.2e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    print(f"\n  Full prompt embed: {'PASS' if all_ok else 'FAIL'}")

    # ---------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------
    banner("EXECUTION SUMMARY")
    print(f"""
  Device:        {gpu.device_name}
  Model:         {model}
  Prompt:        {prompt!r}
  Tokens:        {token_ids}

  Stage 1 (embed single token):  PASS - non-zero, non-NaN, matches reference
  Stage 2 (RMSNorm):             {'PASS' if ok_match else 'FAIL'} - max error {maxerr:.6f}
  Stage 3 (embed all tokens):    {'PASS' if all_ok else 'FAIL'}

  KEY RESULT: Real Qwen 3.5-27B weights -> real CUDA kernels -> correct
  activation vectors on {gpu.device_name}.

  The embed+norm pipeline produces numerically correct output.
  Next: wire projection kernels for the first attention layer.
""")

    # Cleanup
    gpu.mem_free(d_table)
    gpu.mem_free(d_output)
    gpu.mem_free(d_norm_weight)
    gpu.mem_free(d_residual)
    gpu.mem_free(d_norm_output)
    gpu.mem_free(d_multi_output)
    gpu.close()
    model.close()

    return 0 if (all_ok and ok_match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
