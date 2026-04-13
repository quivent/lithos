#!/usr/bin/env python3
"""
Test causal conv1d kernel with real Qwen 3.5-27B DeltaNet weights.

Loads conv1d.weight from layer 0 linear_attn and verifies the 4-tap
causal FIR kernel produces correct output against a Python reference.

Kernel signature (from conv1d.ptx):
    conv1d(input_ptr, weight_ptr, hist_ptr, output_ptr, num_heads, channels)

    input_ptr:   [num_heads, channels] FP16  -- current token activation
    weight_ptr:  [num_heads, channels, 4] FP32 -- 4-tap FIR weights
    hist_ptr:    [num_heads, channels, 3] FP32 -- shift-register history
    output_ptr:  [num_heads, channels] FP32  -- convolution output
    num_heads:   u32
    channels:    u32 (128 -- one thread per channel)

Model conv1d weight tensor:
    model.language_model.layers.0.linear_attn.conv1d.weight
    shape = [10240, 1, 4], dtype = BF16
    10240 = total channels across all heads (80 groups of 128)
    Middle dim = 1 is PyTorch depthwise conv1d convention (groups=channels)
    Reshaped to [80, 128, 4] and cast to FP32 for the kernel.

Grid:   (num_heads, 1, 1)  = (80, 1, 1)
Block:  (128, 1, 1)        = one thread per channel

Run:
    python3 /home/ubuntu/lithos/src/test_conv1d.py
"""

from __future__ import annotations

import ctypes
import numpy as np
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_driver import CUDADriver, CUdeviceptr
from loader import LithosModel

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")

# Kernel grid/block config: 128 threads per block, one block per head-group
CHANNELS = 128


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    """Convert raw BF16 bytes to a float32 numpy array."""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def f32_to_f16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to float16."""
    return arr.astype(np.float16)


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    """Upload a contiguous float32 array to the device."""
    assert data.dtype == np.float32
    buf = np.ascontiguousarray(data)
    dptr = gpu.mem_alloc(buf.nbytes)
    gpu.memcpy_htod(dptr, buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)
    return dptr


def upload_f16(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    """Upload a contiguous float16 array to the device."""
    assert data.dtype == np.float16
    buf = np.ascontiguousarray(data)
    dptr = gpu.mem_alloc(buf.nbytes)
    gpu.memcpy_htod(dptr, buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, n: int) -> np.ndarray:
    """Download n float32 values from device to host."""
    out = np.zeros(n, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, n * 4)
    return out


def causal_conv1d_reference(
    x_t: np.ndarray,
    hist: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Python reference for 4-tap causal conv1d (single-token decode step).

    Args:
        x_t:     [num_heads, channels] float32 -- current input
        hist:    [num_heads, channels, 3] float32 -- history [x_{t-3}, x_{t-2}, x_{t-1}]
        weights: [num_heads, channels, 4] float32 -- FIR taps

    Returns:
        y:        [num_heads, channels] float32 -- conv output
        new_hist: [num_heads, channels, 3] float32 -- updated history
    """
    num_heads, channels = x_t.shape
    assert weights.shape == (num_heads, channels, 4)
    assert hist.shape == (num_heads, channels, 3)

    # y = w[0]*x_t + w[1]*x_{t-1} + w[2]*x_{t-2} + w[3]*x_{t-3}
    y = (weights[:, :, 0] * x_t +
         weights[:, :, 1] * hist[:, :, 2] +   # x_{t-1}
         weights[:, :, 2] * hist[:, :, 1] +   # x_{t-2}
         weights[:, :, 3] * hist[:, :, 0])     # x_{t-3}

    # Shift history: [x_{t-2}, x_{t-1}, x_t]
    new_hist = np.stack([hist[:, :, 1], hist[:, :, 2], x_t], axis=-1)

    return y, new_hist


def main() -> int:
    banner("Conv1d Kernel Test -- Qwen 3.5-27B DeltaNet Layer 0")

    # ------------------------------------------------------------------
    # 1. Load model and verify conv1d weight shape/dtype
    # ------------------------------------------------------------------
    print("\n--- Loading model ---")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    t1 = time.monotonic()
    print(f"  Model: {model}  ({t1 - t0:.3f}s)")

    conv_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    ti = model.weight_info(conv_name)
    print(f"\n  Weight tensor: {conv_name}")
    print(f"    dtype:  {ti.dtype}")
    print(f"    shape:  {ti.shape}")
    print(f"    bytes:  {ti.byte_size:,}")

    # Validate shape and dtype
    assert ti.shape == [10240, 1, 4], f"Expected shape [10240, 1, 4], got {ti.shape}"
    assert ti.dtype == "BF16", f"Expected BF16, got {ti.dtype}"

    total_channels = ti.shape[0]  # 10240
    kernel_dim = ti.shape[2]      # 4
    assert kernel_dim == 4, f"Expected kernel_dim=4, got {kernel_dim}"
    assert total_channels % CHANNELS == 0, (
        f"Total channels {total_channels} not divisible by block size {CHANNELS}"
    )
    num_heads = total_channels // CHANNELS  # 80
    print(f"    num_heads (grid blocks): {num_heads}")
    print(f"    channels per head:       {CHANNELS}")

    # Verify model config matches
    cfg = model.config
    print(f"\n  Model config:")
    print(f"    linear_conv_kernel_dim:  {cfg.linear_conv_kernel_dim}")
    print(f"    linear_num_key_heads:    {cfg.linear_num_key_heads}")
    print(f"    linear_key_head_dim:     {cfg.linear_key_head_dim}")
    print(f"    linear_num_value_heads:  {cfg.linear_num_value_heads}")
    print(f"    linear_value_head_dim:   {cfg.linear_value_head_dim}")
    assert cfg.linear_conv_kernel_dim == 4, (
        f"Config says kernel_dim={cfg.linear_conv_kernel_dim}, expected 4"
    )

    # ------------------------------------------------------------------
    # 2. Load and reshape weights: [10240, 1, 4] BF16 -> [80, 128, 4] FP32
    # ------------------------------------------------------------------
    banner("STEP 1: Load and convert conv1d weights")

    raw = bytes(model.weight_bytes(conv_name))
    weights_flat = bf16_to_f32(raw)  # [10240 * 4] flat
    # Original layout is [10240, 1, 4] -- drop the middle dim -> [10240, 4]
    # Then reshape to [num_heads, channels, 4] = [80, 128, 4]
    weights = weights_flat.reshape(num_heads, CHANNELS, 4)

    print(f"  Weights (FP32): shape={weights.shape}")
    print(f"    min={weights.min():.6f}  max={weights.max():.6f}  "
          f"mean={weights.mean():.6f}")
    print(f"    First head, first channel taps: {weights[0, 0, :]}")
    print(f"    Last head, last channel taps:   {weights[-1, -1, :]}")

    # ------------------------------------------------------------------
    # 3. Initialize CUDA and load kernel
    # ------------------------------------------------------------------
    banner("STEP 2: Load CUDA kernel")

    gpu = CUDADriver()
    print(f"  Device: {gpu.device_name}")

    cubin_path = f"{KERNEL_DIR}/conv1d.cubin"
    mod = gpu.load_cubin(cubin_path)
    func = gpu.get_function(mod, "conv1d")
    print(f"  Loaded conv1d kernel from {cubin_path}")

    # ------------------------------------------------------------------
    # 4. Test A: Single step with zero history
    # ------------------------------------------------------------------
    banner("TEST A: Single decode step, zero history")

    np.random.seed(42)
    x_t = np.random.randn(num_heads, CHANNELS).astype(np.float32)
    hist = np.zeros((num_heads, CHANNELS, 3), dtype=np.float32)

    # Reference
    ref_out, ref_hist = causal_conv1d_reference(x_t, hist, weights)

    # Convert input to FP16 for kernel (kernel reads FP16)
    x_t_f16 = f32_to_f16(x_t)

    # Upload
    d_input = upload_f16(gpu, x_t_f16)
    d_weights = upload_f32(gpu, weights)
    d_hist = upload_f32(gpu, hist)
    d_output = gpu.mem_alloc(num_heads * CHANNELS * 4)

    # Launch
    gpu.launch(
        func,
        grid=(num_heads, 1, 1),
        block=(CHANNELS, 1, 1),
        args=[
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_weights.value),
            ctypes.c_uint64(d_hist.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(CHANNELS),
        ],
    )
    gpu.synchronize()

    # Download output
    gpu_out = download_f32(gpu, d_output, num_heads * CHANNELS).reshape(num_heads, CHANNELS)

    # The kernel reads FP16 input and converts to FP32 internally, so the
    # reference should also use the FP16-rounded input for a fair comparison.
    x_t_rounded = x_t_f16.astype(np.float32)
    ref_out_rounded, _ = causal_conv1d_reference(x_t_rounded, hist, weights)

    abs_err = np.abs(gpu_out - ref_out_rounded)
    max_abs = abs_err.max()
    mean_abs = abs_err.mean()

    denom = np.maximum(np.abs(ref_out_rounded), 1e-8)
    rel_err = abs_err / denom
    max_rel = rel_err.max()

    # Cosine similarity
    g_flat = gpu_out.ravel()
    r_flat = ref_out_rounded.ravel()
    cosine = np.dot(g_flat, r_flat) / (
        np.linalg.norm(g_flat) * np.linalg.norm(r_flat) + 1e-20
    )

    print(f"  GPU output:  min={gpu_out.min():.6f}  max={gpu_out.max():.6f}")
    print(f"  Ref output:  min={ref_out_rounded.min():.6f}  max={ref_out_rounded.max():.6f}")
    print(f"  Max abs err: {max_abs:.6e}")
    print(f"  Mean abs err:{mean_abs:.6e}")
    print(f"  Max rel err: {max_rel:.6e}")
    print(f"  Cosine sim:  {cosine:.10f}")

    # With zero history and FP16 input rounding, only w[0]*x_t contributes,
    # so error should be extremely small (just FP32 rounding).
    pass_a = max_abs < 1e-3 and cosine > 0.9999
    print(f"  VERDICT: {'PASS' if pass_a else 'FAIL'}")

    # ------------------------------------------------------------------
    # 5. Verify history was updated
    # ------------------------------------------------------------------
    banner("TEST B: Verify history buffer update")

    gpu_hist = download_f32(gpu, d_hist, num_heads * CHANNELS * 3).reshape(
        num_heads, CHANNELS, 3
    )

    # After one step from zero history with input x_t:
    #   hist[0] = x_{t-2} = 0  (was hist[1] = 0)
    #   hist[1] = x_{t-1} = 0  (was hist[2] = 0)
    #   hist[2] = x_t          (the new input, stored as FP32 from FP16)
    expected_hist = np.zeros_like(gpu_hist)
    expected_hist[:, :, 2] = x_t_rounded  # x_t was converted from FP16 in kernel

    hist_err = np.abs(gpu_hist - expected_hist).max()
    pass_b = hist_err < 1e-6
    print(f"  History max err: {hist_err:.6e}")
    print(f"  hist[:,:,0] (should be 0): max={np.abs(gpu_hist[:,:,0]).max():.6e}")
    print(f"  hist[:,:,1] (should be 0): max={np.abs(gpu_hist[:,:,1]).max():.6e}")
    print(f"  hist[:,:,2] (should be x_t): err={np.abs(gpu_hist[:,:,2] - x_t_rounded).max():.6e}")
    print(f"  VERDICT: {'PASS' if pass_b else 'FAIL'}")

    # ------------------------------------------------------------------
    # 6. Test C: Multi-step sequence (4 steps to fill history)
    # ------------------------------------------------------------------
    banner("TEST C: Multi-step sequence (4 decode steps)")

    np.random.seed(123)
    steps = 4
    inputs = [np.random.randn(num_heads, CHANNELS).astype(np.float32) for _ in range(steps)]

    # Reset history to zero
    hist_cpu = np.zeros((num_heads, CHANNELS, 3), dtype=np.float32)
    gpu.memcpy_htod(
        d_hist,
        hist_cpu.ctypes.data_as(ctypes.c_void_p),
        hist_cpu.nbytes,
    )

    ref_outputs = []
    gpu_outputs = []

    for step in range(steps):
        x = inputs[step]
        x_f16 = f32_to_f16(x)
        x_rounded = x_f16.astype(np.float32)

        # Reference
        ref_y, hist_cpu = causal_conv1d_reference(x_rounded, hist_cpu, weights)
        ref_outputs.append(ref_y)

        # GPU: upload new input
        gpu.memcpy_htod(
            d_input,
            np.ascontiguousarray(x_f16).ctypes.data_as(ctypes.c_void_p),
            x_f16.nbytes,
        )

        # Launch (history buffer persists on device across steps)
        gpu.launch(
            func,
            grid=(num_heads, 1, 1),
            block=(CHANNELS, 1, 1),
            args=[
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_weights.value),
                ctypes.c_uint64(d_hist.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_uint32(num_heads),
                ctypes.c_uint32(CHANNELS),
            ],
        )
        gpu.synchronize()

        gpu_y = download_f32(gpu, d_output, num_heads * CHANNELS).reshape(
            num_heads, CHANNELS
        )
        gpu_outputs.append(gpu_y)

    # Compare each step
    pass_c = True
    for step in range(steps):
        err = np.abs(gpu_outputs[step] - ref_outputs[step])
        max_e = err.max()
        g = gpu_outputs[step].ravel()
        r = ref_outputs[step].ravel()
        cos = np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-20)
        ok = max_e < 1e-3 and cos > 0.9999
        if not ok:
            pass_c = False
        print(f"  Step {step}: max_abs_err={max_e:.6e}  cosine={cos:.10f}  {'PASS' if ok else 'FAIL'}")

    # Step 3 (4th step) uses all 4 taps -- this is the full test
    print(f"\n  Step 3 uses all 4 FIR taps (full history).")
    print(f"  VERDICT: {'PASS' if pass_c else 'FAIL'}")

    # ------------------------------------------------------------------
    # 7. Test D: Verify final history state after 4 steps
    # ------------------------------------------------------------------
    banner("TEST D: Final history state after 4 steps")

    gpu_hist_final = download_f32(gpu, d_hist, num_heads * CHANNELS * 3).reshape(
        num_heads, CHANNELS, 3
    )

    # After 4 steps with inputs x0..x3 (all FP16-rounded):
    #   hist[0] = x1, hist[1] = x2, hist[2] = x3
    x_rounded = [f32_to_f16(inputs[i]).astype(np.float32) for i in range(steps)]
    expected_final_hist = np.stack(
        [x_rounded[1], x_rounded[2], x_rounded[3]], axis=-1
    )

    hist_final_err = np.abs(gpu_hist_final - expected_final_hist).max()
    pass_d = hist_final_err < 1e-6
    print(f"  Final history max err: {hist_final_err:.6e}")
    print(f"  VERDICT: {'PASS' if pass_d else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("SUMMARY")

    all_pass = pass_a and pass_b and pass_c and pass_d
    results = [
        ("A: Zero-history single step", pass_a),
        ("B: History update",           pass_b),
        ("C: Multi-step (4 taps)",      pass_c),
        ("D: Final history state",      pass_d),
    ]
    for name, ok in results:
        print(f"  {name:40s} {'PASS' if ok else 'FAIL'}")

    print(f"\n  Kernel format compatibility:")
    print(f"    Model weight:   [{ti.shape[0]}, {ti.shape[1]}, {ti.shape[2]}] {ti.dtype}")
    print(f"    Kernel expects: [num_heads, channels, 4] FP32")
    print(f"    Reshape:        [{ti.shape[0]}, {ti.shape[2]}] -> [{num_heads}, {CHANNELS}, {kernel_dim}]")
    print(f"    BF16->FP32 conversion required: YES")
    print(f"    Causal (no future taps): YES")
    print(f"    Input dtype: FP16 (kernel does FP16->FP32 on load)")
    print(f"    Output dtype: FP32 (feeds into DeltaNet recurrence)")

    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # Cleanup
    for dptr in [d_input, d_weights, d_hist, d_output]:
        try:
            gpu.mem_free(dptr)
        except Exception:
            pass
    gpu.close()
    model.close()

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
