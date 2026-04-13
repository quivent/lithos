#!/usr/bin/env python3
"""
Test gptq_matvec_tc kernel: compare against known-correct gptq_matvec output.

Run:
    python3 /home/ubuntu/lithos/src/test_matvec_tc.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
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


def main() -> int:
    banner("GPTQ W4A16 Tensor-Core Matvec Test")

    # 1. Load model + CUDA
    print("\n--- Loading model and CUDA ---")
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    print(f"  Device: {gpu.device_name}")

    # 2. Generate input via embed + norm (same as test_projection.py)
    banner("STEP 1: Generate input via embed + norm")

    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")

    embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
    embed_func = gpu.get_function(embed_mod, "embed_f16")

    norm_cubin = "/tmp/lithos-cache/3644e4d3fa48efc4/norm.cubin"
    try:
        norm_mod = gpu.load_cubin(norm_cubin)
    except Exception:
        import subprocess
        norm_ptx = f"{KERNEL_DIR}/norm.ptx"
        norm_cubin = "/tmp/gptq_test_norm.cubin"
        subprocess.run(["ptxas", "-arch=sm_90", "-o", norm_cubin, norm_ptx], check=True)
        norm_mod = gpu.load_cubin(norm_cubin)
    norm_func = gpu.get_function(norm_mod, "norm")

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

    input_vec = np.zeros(HIDDEN_DIM, dtype=np.float32)
    gpu.memcpy_dtoh(
        input_vec.ctypes.data_as(ctypes.c_void_p),
        d_norm_out,
        HIDDEN_DIM * 4,
    )
    print(f"  Input vector (post-norm): min={input_vec.min():.6f} max={input_vec.max():.6f}")

    # 3. Get weight pointers
    layer_prefix = "model.language_model.layers.3.self_attn.q_proj"
    qw_ti = model.weight_info(f"{layer_prefix}.qweight")
    sc_ti = model.weight_info(f"{layer_prefix}.scales")

    K = HIDDEN_DIM
    N = qw_ti.shape[1]
    qweight_ptr = qw_ti.ptr
    scales_ptr = sc_ti.ptr
    print(f"  q_proj: K={K} N={N}")

    d_input = upload_f32(gpu, input_vec)

    # 4. Run reference kernel (gptq_matvec)
    banner("STEP 2: Run reference gptq_matvec")

    ref_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
    ref_func = gpu.get_function(ref_mod, "gptq_matvec")

    d_ref_output = gpu.mem_alloc(N * 4)
    zero_buf = np.zeros(N, dtype=np.float32)
    gpu.memcpy_htod(d_ref_output, zero_buf.ctypes.data_as(ctypes.c_void_p), N * 4)

    # Warmup
    gpu.launch(
        ref_func,
        grid=(N, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(qweight_ptr),
            ctypes.c_uint64(scales_ptr),
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_ref_output.value),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ],
        shared_mem=32,
    )
    gpu.synchronize()

    # Timed run
    NRUNS = 20
    gpu.memcpy_htod(d_ref_output, zero_buf.ctypes.data_as(ctypes.c_void_p), N * 4)
    t0 = time.perf_counter()
    for _ in range(NRUNS):
        gpu.launch(
            ref_func,
            grid=(N, 1, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(qweight_ptr),
                ctypes.c_uint64(scales_ptr),
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_ref_output.value),
                ctypes.c_uint32(N),
                ctypes.c_uint32(K),
            ],
            shared_mem=32,
        )
    gpu.synchronize()
    t1 = time.perf_counter()
    ref_time = (t1 - t0) / NRUNS * 1000
    print(f"  Reference kernel time: {ref_time:.2f} ms (avg of {NRUNS} runs)")

    ref_output = np.zeros(N, dtype=np.float32)
    gpu.memcpy_dtoh(
        ref_output.ctypes.data_as(ctypes.c_void_p),
        d_ref_output,
        N * 4,
    )
    print(f"  Ref output: min={ref_output.min():.6f} max={ref_output.max():.6f}")
    print(f"  First 10: {ref_output[:10]}")

    # 5. Run tensor-core kernel (gptq_matvec_tc)
    banner("STEP 3: Run gptq_matvec_tc")

    tc_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec_tc.cubin")
    tc_func = gpu.get_function(tc_mod, "gptq_matvec_tc")

    GRID_TC = math.ceil(N / 64)
    d_tc_output = gpu.mem_alloc(N * 4)
    gpu.memcpy_htod(d_tc_output, zero_buf.ctypes.data_as(ctypes.c_void_p), N * 4)

    print(f"  Launching: grid=({GRID_TC}, 1, 1), block=(256, 1, 1)")

    # Warmup
    gpu.launch(
        tc_func,
        grid=(GRID_TC, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(qweight_ptr),
            ctypes.c_uint64(scales_ptr),
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_tc_output.value),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ],
        shared_mem=20768,
    )
    gpu.synchronize()

    # Timed run
    gpu.memcpy_htod(d_tc_output, zero_buf.ctypes.data_as(ctypes.c_void_p), N * 4)
    t0 = time.perf_counter()
    for _ in range(NRUNS):
        gpu.launch(
            tc_func,
            grid=(GRID_TC, 1, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(qweight_ptr),
                ctypes.c_uint64(scales_ptr),
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_tc_output.value),
                ctypes.c_uint32(N),
                ctypes.c_uint32(K),
            ],
            shared_mem=20768,
        )
    gpu.synchronize()
    t1 = time.perf_counter()
    tc_time = (t1 - t0) / NRUNS * 1000
    print(f"  TC kernel time: {tc_time:.2f} ms (avg of {NRUNS} runs)")

    tc_output = np.zeros(N, dtype=np.float32)
    gpu.memcpy_dtoh(
        tc_output.ctypes.data_as(ctypes.c_void_p),
        d_tc_output,
        N * 4,
    )

    has_nan = np.any(np.isnan(tc_output))
    has_inf = np.any(np.isinf(tc_output))
    print(f"  TC output: min={tc_output.min():.6f} max={tc_output.max():.6f}")
    print(f"  NaN={has_nan} Inf={has_inf}")
    print(f"  First 10: {tc_output[:10]}")

    # 6. Compare
    banner("STEP 4: Comparison")

    abs_err = np.abs(tc_output - ref_output)
    max_err = abs_err.max()
    mean_err = abs_err.mean()

    denom = np.maximum(np.abs(ref_output), 1e-8)
    rel_err = abs_err / denom
    max_rel = rel_err.max()

    dot = np.dot(tc_output, ref_output)
    norm_tc = np.sqrt(np.dot(tc_output, tc_output))
    norm_ref = np.sqrt(np.dot(ref_output, ref_output))
    cosine = dot / (norm_tc * norm_ref + 1e-20)

    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  Max relative error:  {max_rel:.6f}")
    print(f"  Cosine similarity:   {cosine:.8f}")

    print(f"\n  Element-by-element (first 20):")
    print(f"  {'idx':>5s} {'TC':>12s} {'Ref':>12s} {'AbsErr':>12s}")
    for i in range(min(20, N)):
        print(f"  {i:5d} {tc_output[i]:12.6f} {ref_output[i]:12.6f} {abs_err[i]:12.6f}")

    worst_idx = np.argsort(abs_err)[-5:][::-1]
    print(f"\n  Worst 5 elements:")
    for i in worst_idx:
        print(f"  [{i:5d}] TC={tc_output[i]:12.6f} Ref={ref_output[i]:12.6f} AbsErr={abs_err[i]:.6f}")

    # Check how many are zero
    n_zero = np.sum(tc_output == 0.0)
    print(f"\n  Zero outputs: {n_zero}/{N}")

    speedup = ref_time / tc_time if tc_time > 0 else 0
    PASS = max_err < 0.000006 and cosine > 0.9999 and not has_nan and not has_inf

    banner("RESULT")
    print(f"  Reference kernel: {ref_time:.2f} ms")
    print(f"  TC kernel:        {tc_time:.2f} ms")
    print(f"  Speedup:          {speedup:.1f}x")
    print(f"  Max abs error:    {max_err:.6f}  (threshold < 0.000006)")
    print(f"  Cosine similarity:{cosine:.8f}  (threshold > 0.9999)")
    print(f"  VERDICT:          {'PASS' if PASS else 'FAIL'}")

    # Cleanup
    for dptr in [d_embed_out, d_norm_weight, d_residual, d_norm_out, d_input, d_ref_output, d_tc_output]:
        try:
            gpu.mem_free(dptr)
        except Exception:
            pass
    gpu.close()
    model.close()

    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
