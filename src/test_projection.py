#!/usr/bin/env python3
"""
Test GPTQ W4A16 projection kernel on GH200 with real Qwen 3.5-27B weights.

Uses layer 3 q_proj (first full-attention layer) as the test case.
Compares GPU kernel output against a Python reference dequant + matmul.

Run:
    python3 /home/ubuntu/lithos/src/test_projection.py
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
GROUP_SIZE = 128
ZERO_POINT = 7  # symmetric quantization


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def dequant_gptq_reference(qweight_bytes: bytes, scales_bytes: bytes,
                            K: int, N: int) -> np.ndarray:
    """
    CPU reference: dequantize GPTQ W4 weights to f32 matrix [K, N].

    qweight: [K/8, N] int32, row-major
    scales:  [K/128, N] float16, row-major
    """
    K_packed = K // 8
    n_groups = K // GROUP_SIZE

    qw = np.frombuffer(qweight_bytes, dtype=np.int32).reshape(K_packed, N)
    sc = np.frombuffer(scales_bytes, dtype=np.float16).reshape(n_groups, N).astype(np.float32)

    # Dequantize
    weight = np.zeros((K, N), dtype=np.float32)
    for k_packed in range(K_packed):
        for bit in range(8):
            k = k_packed * 8 + bit
            group = k // GROUP_SIZE
            # Extract nibbles for all N outputs at once
            nibbles = (qw[k_packed, :].view(np.uint32) >> (bit * 4)) & 0xF
            weight[k, :] = (nibbles.astype(np.float32) - ZERO_POINT) * sc[group, :]

    return weight


def dequant_gptq_fast(qweight_bytes: bytes, scales_bytes: bytes,
                       K: int, N: int) -> np.ndarray:
    """
    Faster vectorized CPU reference dequantization.
    """
    K_packed = K // 8
    n_groups = K // GROUP_SIZE

    qw = np.frombuffer(qweight_bytes, dtype=np.uint32).reshape(K_packed, N)
    sc = np.frombuffer(scales_bytes, dtype=np.float16).reshape(n_groups, N).astype(np.float32)

    weight = np.zeros((K, N), dtype=np.float32)
    for bit in range(8):
        nibbles = (qw >> (bit * 4)) & 0xF  # [K_packed, N]
        for k_packed in range(K_packed):
            k = k_packed * 8 + bit
            group = k // GROUP_SIZE
            weight[k, :] = (nibbles[k_packed].astype(np.float32) - ZERO_POINT) * sc[group, :]

    return weight


def main() -> int:
    banner("GPTQ W4A16 Projection Kernel Test")

    # ---------------------------------------------------------------
    # 1. Load model + CUDA
    # ---------------------------------------------------------------
    print("\n--- Loading model and CUDA ---")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    t1 = time.monotonic()
    print(f"  Model: {model}  ({t1-t0:.3f}s)")
    print(f"  Device: {gpu.device_name}")

    # ---------------------------------------------------------------
    # 2. Get embed+norm output as input vector
    # ---------------------------------------------------------------
    banner("STEP 1: Generate input via embed + norm")

    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")

    # Load embed kernel
    embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
    embed_func = gpu.get_function(embed_mod, "embed_f16")

    # Load norm kernel
    norm_cubin = "/tmp/lithos-cache/3644e4d3fa48efc4/norm.cubin"
    try:
        norm_mod = gpu.load_cubin(norm_cubin)
    except Exception:
        # Try assembling norm if cached cubin missing
        import subprocess
        norm_ptx = f"{KERNEL_DIR}/norm.ptx"
        norm_cubin = "/tmp/gptq_test_norm.cubin"
        subprocess.run(["ptxas", "-arch=sm_90", "-o", norm_cubin, norm_ptx], check=True)
        norm_mod = gpu.load_cubin(norm_cubin)
    norm_func = gpu.get_function(norm_mod, "norm")

    # Embed first token
    embed_ptr = model.weight_info("model.language_model.embed_tokens.weight").ptr
    d_embed_out = gpu.mem_alloc(HIDDEN_DIM * 4)
    BLOCK = 256
    GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))

    tid = token_ids[0]
    gpu.launch(
        embed_func,
        grid=(GRID, 1, 1),
        block=(BLOCK, 1, 1),
        args=[
            ctypes.c_uint32(tid),
            ctypes.c_uint64(embed_ptr),
            ctypes.c_uint64(d_embed_out.value),
        ],
    )
    gpu.synchronize()

    # Norm
    norm_weight_name = "model.language_model.layers.3.input_layernorm.weight"
    norm_ti = model.weight_info(norm_weight_name)
    norm_raw = bytes(model.weight_bytes(norm_weight_name))
    if norm_ti.dtype == "BF16":
        norm_f32 = bf16_to_f32(norm_raw)
    elif norm_ti.dtype == "F16":
        norm_f32 = np.frombuffer(norm_raw, dtype=np.float16).astype(np.float32)
    else:
        norm_f32 = np.frombuffer(norm_raw, dtype=np.float32).copy()

    d_norm_weight = upload_f32(gpu, norm_f32)
    d_residual = upload_f32(gpu, np.zeros(HIDDEN_DIM, dtype=np.float32))
    d_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)

    epsilon = model.config.rms_norm_eps
    gpu.launch(
        norm_func,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_embed_out.value),
            ctypes.c_uint64(d_residual.value),
            ctypes.c_uint64(d_norm_weight.value),
            ctypes.c_uint64(d_norm_out.value),
            ctypes.c_float(epsilon),
        ],
        shared_mem=128,
    )
    gpu.synchronize()

    # Read back norm output (this is our input to projection)
    input_vec = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_dtoh(
        input_vec.ctypes.data_as(ctypes.c_void_p),
        d_norm_out,
        HIDDEN_DIM * 4,
    )
    print(f"  Input vector (post-norm): min={input_vec.min():.6f} max={input_vec.max():.6f} "
          f"L2={np.sqrt(np.sum(input_vec**2)):.4f}")

    # ---------------------------------------------------------------
    # 3. Load projection kernel
    # ---------------------------------------------------------------
    banner("STEP 2: Load GPTQ projection kernel")

    proj_cubin = f"{KERNEL_DIR}/gptq_matvec.cubin"
    proj_mod = gpu.load_cubin(proj_cubin)
    proj_func = gpu.get_function(proj_mod, "gptq_matvec")
    print(f"  gptq_matvec kernel loaded from {proj_cubin}")

    # ---------------------------------------------------------------
    # 4. Get weight pointers
    # ---------------------------------------------------------------
    banner("STEP 3: Get q_proj weight pointers")

    layer_prefix = "model.language_model.layers.3.self_attn.q_proj"
    qw_ti = model.weight_info(f"{layer_prefix}.qweight")
    sc_ti = model.weight_info(f"{layer_prefix}.scales")
    qz_ti = model.weight_info(f"{layer_prefix}.qzeros")

    K = HIDDEN_DIM  # 5120
    N = qw_ti.shape[1]  # 12288 (output features = num_heads * head_dim * 2?)

    # Actually: qweight shape is [K/8, N] = [640, 12288]
    assert qw_ti.shape[0] == K // 8, f"Expected qweight rows = {K//8}, got {qw_ti.shape[0]}"

    print(f"  q_proj: K={K} N={N}")
    print(f"  qweight: shape={qw_ti.shape} dtype={qw_ti.dtype} @ 0x{qw_ti.ptr:016x}")
    print(f"  scales:  shape={sc_ti.shape} dtype={sc_ti.dtype} @ 0x{sc_ti.ptr:016x}")
    print(f"  qzeros:  shape={qz_ti.shape} dtype={qz_ti.dtype}")

    qweight_ptr = qw_ti.ptr
    scales_ptr = sc_ti.ptr

    # ---------------------------------------------------------------
    # 5. Upload input to device and run kernel
    # ---------------------------------------------------------------
    banner("STEP 4: Launch projection kernel")

    d_input = upload_f32(gpu, input_vec)
    d_output = gpu.mem_alloc(N * 4)

    # Zero output
    zero_buf = np.zeros(N, dtype=np.float32)
    gpu.memcpy_htod(d_output, zero_buf.ctypes.data_as(ctypes.c_void_p), N * 4)

    print(f"  Launching: grid=({N}, 1, 1), block=(256, 1, 1)")
    print(f"  qweight_ptr = 0x{qweight_ptr:016x}")
    print(f"  scales_ptr  = 0x{scales_ptr:016x}")
    print(f"  input_ptr   = 0x{d_input.value:016x}")
    print(f"  output_ptr  = 0x{d_output.value:016x}")
    print(f"  N={N}, K={K}")

    t0 = time.perf_counter()
    gpu.launch(
        proj_func,
        grid=(N, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(qweight_ptr),
            ctypes.c_uint64(scales_ptr),
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ],
        shared_mem=32,  # 8 warps * 4 bytes
    )
    gpu.synchronize()
    t1 = time.perf_counter()
    print(f"  Kernel completed in {(t1-t0)*1e3:.2f} ms")

    # Read back
    gpu_output = np.zeros(N, dtype=np.float32)
    gpu.memcpy_dtoh(
        gpu_output.ctypes.data_as(ctypes.c_void_p),
        d_output,
        N * 4,
    )

    has_nan = np.any(np.isnan(gpu_output))
    has_inf = np.any(np.isinf(gpu_output))
    print(f"  GPU output: min={gpu_output.min():.6f} max={gpu_output.max():.6f}")
    print(f"  NaN={has_nan} Inf={has_inf}")
    print(f"  First 10: {gpu_output[:10]}")
    print(f"  Last 10:  {gpu_output[-10:]}")

    # ---------------------------------------------------------------
    # 6. CPU reference
    # ---------------------------------------------------------------
    banner("STEP 5: CPU reference dequant + matvec")

    qw_raw = bytes(model.weight_bytes(f"{layer_prefix}.qweight"))
    sc_raw = bytes(model.weight_bytes(f"{layer_prefix}.scales"))

    print(f"  Dequantizing {K}x{N} weight matrix (this may take a moment)...")
    t0 = time.monotonic()
    weight_f32 = dequant_gptq_fast(qw_raw, sc_raw, K, N)
    t1 = time.monotonic()
    print(f"  Dequant done in {t1-t0:.2f}s")
    print(f"  Weight stats: min={weight_f32.min():.6f} max={weight_f32.max():.6f} "
          f"mean={weight_f32.mean():.6f}")

    # Matvec: output = input @ weight  (input is [K], weight is [K, N], output is [N])
    t0 = time.monotonic()
    ref_output = input_vec @ weight_f32
    t1 = time.monotonic()
    print(f"  Matvec done in {t1-t0:.2f}s")
    print(f"  Ref output: min={ref_output.min():.6f} max={ref_output.max():.6f}")
    print(f"  First 10: {ref_output[:10]}")
    print(f"  Last 10:  {ref_output[-10:]}")

    # ---------------------------------------------------------------
    # 7. Compare
    # ---------------------------------------------------------------
    banner("STEP 6: Comparison")

    abs_err = np.abs(gpu_output - ref_output)
    max_err = abs_err.max()
    mean_err = abs_err.mean()

    # Relative error (avoid div by zero)
    denom = np.maximum(np.abs(ref_output), 1e-8)
    rel_err = abs_err / denom
    max_rel = rel_err.max()
    mean_rel = rel_err.mean()

    # Cosine similarity
    dot = np.dot(gpu_output, ref_output)
    norm_gpu = np.sqrt(np.dot(gpu_output, gpu_output))
    norm_ref = np.sqrt(np.dot(ref_output, ref_output))
    cosine = dot / (norm_gpu * norm_ref + 1e-20)

    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  Max relative error:  {max_rel:.6f}")
    print(f"  Mean relative error: {mean_rel:.6f}")
    print(f"  Cosine similarity:   {cosine:.8f}")
    print(f"  L2 norm (GPU):       {norm_gpu:.4f}")
    print(f"  L2 norm (ref):       {norm_ref:.4f}")

    # Element-by-element for first 10
    print("\n  Element-by-element (first 20):")
    print(f"  {'idx':>5s} {'GPU':>12s} {'Ref':>12s} {'AbsErr':>12s} {'RelErr':>10s}")
    for i in range(min(20, N)):
        print(f"  {i:5d} {gpu_output[i]:12.6f} {ref_output[i]:12.6f} "
              f"{abs_err[i]:12.6f} {rel_err[i]:10.6f}")

    # Find worst elements
    worst_idx = np.argsort(abs_err)[-5:][::-1]
    print(f"\n  Worst 5 elements:")
    for i in worst_idx:
        print(f"  [{i:5d}] GPU={gpu_output[i]:12.6f} Ref={ref_output[i]:12.6f} "
              f"AbsErr={abs_err[i]:.6f}")

    # Pass criteria
    PASS = max_err < 0.1 and cosine > 0.99 and not has_nan and not has_inf

    banner("RESULT")
    print(f"  Max abs error:     {max_err:.6f}  (threshold < 0.1)")
    print(f"  Cosine similarity: {cosine:.8f}  (threshold > 0.99)")
    print(f"  NaN/Inf:           {has_nan or has_inf}")
    print(f"  VERDICT:           {'PASS' if PASS else 'FAIL'}")

    if not PASS and not has_nan:
        # Check if the pattern matches (maybe an offset issue)
        # Try shifting by 1
        for shift in [1, -1, 8, -8]:
            if 0 <= shift < N and N + shift <= N:
                shifted = gpu_output[max(0,shift):min(N,N+shift)]
                ref_shifted = ref_output[max(0,-shift):min(N,N-shift)]
                if len(shifted) == len(ref_shifted) and len(shifted) > 0:
                    c = np.dot(shifted, ref_shifted) / (
                        np.sqrt(np.dot(shifted, shifted)) * np.sqrt(np.dot(ref_shifted, ref_shifted)) + 1e-20)
                    if c > cosine + 0.01:
                        print(f"  NOTE: shift={shift} gives cosine={c:.6f} (better than {cosine:.6f})")

    # Cleanup
    for dptr in [d_embed_out, d_norm_weight, d_residual, d_norm_out, d_input, d_output]:
        try:
            gpu.mem_free(dptr)
        except Exception:
            pass
    gpu.close()
    model.close()

    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
