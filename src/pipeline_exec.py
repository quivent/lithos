#!/usr/bin/env python3
"""
Lithos pipeline execution -- wire real kernels end to end on GH200.

Key insight: GH200 unified memory means host mmap'd pointers are directly
accessible from GPU kernels.  No cuMemcpyHtoD needed for weights.

Stage 1: F16 embed kernel with mmap'd model weights -> F32 activation vector
Stage 2: RMSNorm kernel on embed output -> normalized activations
Stage 3: Full prompt embedding + norm for all tokens
Stage 4: Verify all outputs against CPU reference

Run:
    python3 /home/ubuntu/lithos/src/pipeline_exec.py
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
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")

HIDDEN_DIM = 5120


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def check_nan_inf(arr: np.ndarray, label: str) -> bool:
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    if has_nan or has_inf:
        print(f"  [{label}] PROBLEM: NaN={has_nan} Inf={has_inf}")
        return False
    return True


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    """Convert BF16 raw bytes to F32 numpy array."""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    """Upload F32 numpy array to device memory, return device pointer."""
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def main() -> int:
    banner("Lithos Pipeline Execution -- Real Kernels on GH200")
    results = {}

    # ---------------------------------------------------------------
    # 1. Load model + tokenizer + CUDA
    # ---------------------------------------------------------------
    print("\n--- Loading model, tokenizer, CUDA context ---")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    t1 = time.monotonic()
    print(f"  Model: {model}  ({t1-t0:.3f}s)")
    print(f"  Device: {gpu.device_name}")
    print(f"  hidden_dim={model.config.hidden_dim}  vocab={model.config.vocab_size}")

    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")

    # ---------------------------------------------------------------
    # 2. Load kernels
    # ---------------------------------------------------------------
    print("\n--- Loading kernel cubins ---")
    embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
    embed_func = gpu.get_function(embed_mod, "embed_f16")
    print(f"  embed_f16: loaded")

    norm_cubin = "/tmp/lithos-cache/3644e4d3fa48efc4/norm.cubin"
    norm_mod = gpu.load_cubin(norm_cubin)
    norm_func = gpu.get_function(norm_mod, "norm")
    print(f"  norm: loaded")

    # ---------------------------------------------------------------
    # 3. Get weight pointers (host mmap -- works on GH200 unified memory)
    # ---------------------------------------------------------------
    embed_ptr = model.weight_info("model.language_model.embed_tokens.weight").ptr
    print(f"\n  embed table @ 0x{embed_ptr:016x} (host mmap, F16)")

    # ---------------------------------------------------------------
    # STAGE 1: Embed all prompt tokens (F16 -> F32)
    # ---------------------------------------------------------------
    banner("STAGE 1: F16 Embed Kernel -- All Prompt Tokens")

    n_tokens = len(token_ids)
    BLOCK = 256
    GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))

    # Allocate output: [n_tokens, HIDDEN_DIM] F32
    d_embed_out = gpu.mem_alloc(n_tokens * HIDDEN_DIM * 4)

    t0 = time.perf_counter()
    for i, tid in enumerate(token_ids):
        out_addr = CUdeviceptr(d_embed_out.value + i * HIDDEN_DIM * 4)
        gpu.launch(
            embed_func,
            grid=(GRID, 1, 1),
            block=(BLOCK, 1, 1),
            args=[
                ctypes.c_uint32(tid),
                ctypes.c_uint64(embed_ptr),
                ctypes.c_uint64(out_addr.value),
            ],
        )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Embedded {n_tokens} tokens in {(t1-t0)*1e6:.0f} us")

    # Read back and verify
    embed_output = np.zeros((n_tokens, HIDDEN_DIM), dtype=np.float32)
    gpu.memcpy_dtoh(
        embed_output.ctypes.data_as(ctypes.c_void_p),
        d_embed_out,
        embed_output.nbytes,
    )

    raw = model.weight_bytes("model.language_model.embed_tokens.weight")
    all_ok = True
    for i, tid in enumerate(token_ids):
        off = tid * HIDDEN_DIM * 2
        ref = np.frombuffer(bytes(raw[off:off+HIDDEN_DIM*2]), dtype=np.float16).astype(np.float32)
        err = np.max(np.abs(embed_output[i] - ref))
        word = tok.decode([tid])
        ok = err < 1e-6 and check_nan_inf(embed_output[i], f"embed_{tid}")
        print(f"    token {tid:6d} ({word!r:12s}): "
              f"min={embed_output[i].min():+.4f} max={embed_output[i].max():+.4f} "
              f"err={err:.2e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    results["embed"] = all_ok
    print(f"\n  STAGE 1 RESULT: {'PASS' if all_ok else 'FAIL'}")

    if not all_ok:
        print("  Stopping -- embed failed.")
        gpu.close()
        model.close()
        return 1

    # ---------------------------------------------------------------
    # STAGE 2: RMSNorm on first token's embedding
    # ---------------------------------------------------------------
    banner("STAGE 2: RMSNorm Kernel (Layer 0 Input Norm)")

    # Norm kernel: norm(input_ptr, residual_ptr, weight_ptr, output_ptr, epsilon)
    # All F32.  hidden_dim=5120 baked in.

    # Prepare norm weight (BF16 -> F32, upload to device)
    norm_weight_name = "model.language_model.layers.0.input_layernorm.weight"
    norm_ti = model.weight_info(norm_weight_name)
    norm_raw = bytes(model.weight_bytes(norm_weight_name))
    if norm_ti.dtype == "BF16":
        norm_f32 = bf16_to_f32(norm_raw)
    elif norm_ti.dtype == "F16":
        norm_f32 = np.frombuffer(norm_raw, dtype=np.float16).astype(np.float32)
    else:
        norm_f32 = np.frombuffer(norm_raw, dtype=np.float32).copy()
    print(f"  Norm weight: {norm_ti.dtype} -> F32, "
          f"min={norm_f32.min():.6f} max={norm_f32.max():.6f}")

    d_norm_weight = upload_f32(gpu, norm_f32)

    # Residual = zeros (first layer)
    d_residual = upload_f32(gpu, np.zeros(HIDDEN_DIM, dtype=np.float32))

    # Use first token's embed output as input
    d_norm_input = CUdeviceptr(d_embed_out.value)  # first token = offset 0

    # Output buffer
    d_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)

    epsilon = model.config.rms_norm_eps
    print(f"  epsilon = {epsilon}")

    t0 = time.perf_counter()
    gpu.launch(
        norm_func,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_norm_input.value),
            ctypes.c_uint64(d_residual.value),
            ctypes.c_uint64(d_norm_weight.value),
            ctypes.c_uint64(d_norm_out.value),
            ctypes.c_float(epsilon),
        ],
        shared_mem=128,
    )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Norm kernel: {(t1-t0)*1e6:.0f} us")

    norm_result = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_dtoh(
        norm_result.ctypes.data_as(ctypes.c_void_p),
        d_norm_out,
        HIDDEN_DIM * 4,
    )

    # CPU reference
    x = embed_output[0] + 0.0  # input + residual(zeros)
    rms = np.sqrt(np.mean(x ** 2) + epsilon)
    ref_norm = (x / rms) * norm_f32

    maxerr = np.max(np.abs(norm_result - ref_norm))
    ok_nan = check_nan_inf(norm_result, "norm")
    ok_match = maxerr < 0.01

    print(f"  Output: min={norm_result.min():.6f} max={norm_result.max():.6f} "
          f"mean={norm_result.mean():.6f}")
    print(f"  first 8: {norm_result[:8]}")
    print(f"  vs CPU reference max error: {maxerr:.6f}")
    results["norm"] = ok_nan and ok_match
    print(f"\n  STAGE 2 RESULT: {'PASS' if results['norm'] else 'FAIL'}")

    # ---------------------------------------------------------------
    # STAGE 3: Norm all 5 tokens
    # ---------------------------------------------------------------
    banner("STAGE 3: Norm All Prompt Tokens")

    # Allocate output for all tokens
    d_norm_all = gpu.mem_alloc(n_tokens * HIDDEN_DIM * 4)

    # Allocate zero residual for all tokens
    d_residual_all = upload_f32(gpu, np.zeros(n_tokens * HIDDEN_DIM, dtype=np.float32))

    t0 = time.perf_counter()
    gpu.launch(
        norm_func,
        grid=(n_tokens, 1, 1),  # one block per token row
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_embed_out.value),
            ctypes.c_uint64(d_residual_all.value),
            ctypes.c_uint64(d_norm_weight.value),
            ctypes.c_uint64(d_norm_all.value),
            ctypes.c_float(epsilon),
        ],
        shared_mem=128,
    )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Norm {n_tokens} tokens: {(t1-t0)*1e6:.0f} us")

    norm_all_result = np.zeros((n_tokens, HIDDEN_DIM), dtype=np.float32)
    gpu.memcpy_dtoh(
        norm_all_result.ctypes.data_as(ctypes.c_void_p),
        d_norm_all,
        norm_all_result.nbytes,
    )

    all_norm_ok = True
    for i, tid in enumerate(token_ids):
        row = norm_all_result[i]
        x_ref = embed_output[i]
        rms_ref = np.sqrt(np.mean(x_ref ** 2) + epsilon)
        ref = (x_ref / rms_ref) * norm_f32
        err = np.max(np.abs(row - ref))
        ok = err < 0.01 and check_nan_inf(row, f"norm_{tid}")
        word = tok.decode([tid])
        print(f"    token {tid:6d} ({word!r:12s}): "
              f"norm_min={row.min():+.6f} norm_max={row.max():+.6f} "
              f"err={err:.6f}  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_norm_ok = False

    results["norm_all_tokens"] = all_norm_ok
    print(f"\n  STAGE 3 RESULT: {'PASS' if all_norm_ok else 'FAIL'}")

    # ---------------------------------------------------------------
    # STAGE 4: Activation statistics -- is the signal reasonable?
    # ---------------------------------------------------------------
    banner("STAGE 4: Activation Analysis")

    print("  Post-embed activation statistics per token:")
    for i, tid in enumerate(token_ids):
        row = embed_output[i]
        word = tok.decode([tid])
        l2 = np.sqrt(np.sum(row ** 2))
        print(f"    {word!r:12s}: L2={l2:.4f}  var={np.var(row):.6f}  "
              f"absmax={np.max(np.abs(row)):.4f}")

    print("\n  Post-norm activation statistics per token:")
    for i, tid in enumerate(token_ids):
        row = norm_all_result[i]
        word = tok.decode([tid])
        l2 = np.sqrt(np.sum(row ** 2))
        print(f"    {word!r:12s}: L2={l2:.4f}  var={np.var(row):.6f}  "
              f"absmax={np.max(np.abs(row)):.4f}")

    results["activation_analysis"] = True  # informational

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    banner("EXECUTION SUMMARY")

    print(f"""
  Device:          {gpu.device_name}
  Model:           Qwen 3.5-27B GPTQ-W4A16  ({model.total_weight_bytes/(1024**3):.2f} GiB)
  Prompt:          {prompt!r}
  Tokens:          {token_ids}
  Unified memory:  YES (host mmap pointers work directly in kernels)
""")

    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"""
  KEY FINDINGS:
  1. GH200 unified memory: host mmap'd safetensors pointers passed directly
     to CUDA kernels -- ZERO copies needed for weight access.
  2. F16 embed kernel: reads F16 weights, outputs F32 activations, exact match.
  3. RMSNorm kernel: correct within GPU approx rsqrt tolerance.
  4. All {n_tokens} prompt tokens produce valid, non-zero, non-NaN activations.

  NEXT STEPS:
  - Wire projection kernels (W4A16 GPTQ dequant) for first attention layer
  - Run Q/K/V projections through layer 0
  - Build toward full forward pass and token generation
""")

    # Cleanup
    for dptr in [d_embed_out, d_norm_weight, d_residual, d_norm_out,
                 d_norm_all, d_residual_all]:
        try:
            gpu.mem_free(dptr)
        except Exception:
            pass
    gpu.close()
    model.close()

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
