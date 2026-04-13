#!/usr/bin/env python3
"""
Comparison script: load one layer from the Qwen GPTQ model, dequantize to FP32,
re-quantize with each of the 5 schemes, and report metrics.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shannon_smrs import (
    quantize as shannon_quantize,
    dequantize as shannon_dequantize,
    bits_per_weight as shannon_bpw,
    layer_size_bytes as shannon_size,
    validate as shannon_validate,
)
from vonneumann_hgq import (
    quantize as vn_quantize,
    dequantize as vn_dequantize,
    bits_per_weight as vn_bpw,
    layer_size_bytes as vn_size,
    validate as vn_validate,
)
from ferrucci_ebaq import (
    quantize as ferrucci_quantize,
    dequantize as ferrucci_dequantize,
    bits_per_weight as ferrucci_bpw,
    layer_size_bytes as ferrucci_size,
    validate as ferrucci_validate,
)
from turing_293 import (
    quantize as turing_quantize,
    dequantize as turing_dequantize,
    bits_per_weight as turing_bpw,
    layer_size_bytes as turing_size,
    validate as turing_validate,
)
from lovelace_ale import (
    quantize as lovelace_quantize,
    dequantize as lovelace_dequantize,
    bits_per_weight as lovelace_bpw,
    layer_size_bytes as lovelace_size,
    validate as lovelace_validate,
)

GPTQ_MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
BUDGET_MB = 66.0


def dequantize_gptq_layer(model_path: str, layer_idx: int = 0,
                            matrix_name: str = "mlp.gate_proj") -> np.ndarray:
    """
    Dequantize a single weight matrix from GPTQ format to FP32.

    GPTQ W4A16 symmetric:
        qweight: (in_features // 8, out_features) int32
            each int32 holds 8 x 4-bit quantized values
        scales: (in_features // group_size, out_features) float16
        qzeros: (in_features // group_size, out_features // 8) int32
            each int32 holds 8 x 4-bit zero-point values
    """
    from safetensors import safe_open
    import json

    # Load quantization config
    with open(f"{model_path}/quantization_config.json") as f:
        qconfig = json.load(f)

    group_size = qconfig["group_size"]  # 128
    bits = qconfig["bits"]  # 4
    sym = qconfig["sym"]  # True

    # Find which shard has this layer
    with open(f"{model_path}/model.safetensors.index.json") as f:
        index = json.load(f)

    prefix = f"model.language_model.layers.{layer_idx}.{matrix_name}"
    qweight_key = f"{prefix}.qweight"
    scales_key = f"{prefix}.scales"
    qzeros_key = f"{prefix}.qzeros"

    shard = index["weight_map"][qweight_key]
    shard_path = f"{model_path}/{shard}"

    f = safe_open(shard_path, framework="numpy")

    qweight = f.get_tensor(qweight_key)  # (in//8, out) int32
    scales = f.get_tensor(scales_key)     # (in//gs, out) float16
    qzeros = f.get_tensor(qzeros_key)     # (in//gs, out//8) int32

    # Unpack qweight: each int32 has 8 x 4-bit values
    in_packed, out_features = qweight.shape
    in_features = in_packed * 8

    # Unpack to (in_features, out_features)
    unpacked = np.zeros((in_features, out_features), dtype=np.int32)
    for i in range(8):
        unpacked[i::8, :] = (qweight >> (bits * i)) & ((1 << bits) - 1)

    # Unpack qzeros similarly
    n_groups = scales.shape[0]
    out_packed_z = qzeros.shape[1]
    zeros_unpacked = np.zeros((n_groups, out_features), dtype=np.int32)
    for i in range(8):
        zeros_unpacked[:, i::8] = (qzeros >> (bits * i)) & ((1 << bits) - 1)

    # Dequantize: w = scale * (q - zero)
    scales_f32 = scales.astype(np.float32)

    weights = np.zeros((in_features, out_features), dtype=np.float32)
    for g in range(n_groups):
        row_start = g * group_size
        row_end = min(row_start + group_size, in_features)
        q_slice = unpacked[row_start:row_end, :].astype(np.float32)
        z_slice = zeros_unpacked[g, :].astype(np.float32)
        s_slice = scales_f32[g, :]
        weights[row_start:row_end, :] = s_slice[None, :] * (q_slice - z_slice[None, :])

    return weights


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def relative_mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE / variance of original."""
    v = np.var(a.astype(np.float64))
    if v < 1e-20:
        return 0.0
    return mse(a, b) / v


def main():
    print("=" * 90)
    print("LITHOS QUANTIZATION COMPARISON")
    print("=" * 90)
    print()

    # Step 1: Load and dequantize GPTQ weights
    print("[1/3] Loading GPTQ weights from layer 0, mlp.gate_proj ...")
    t0 = time.time()
    ref_weights = dequantize_gptq_layer(GPTQ_MODEL_PATH, layer_idx=0,
                                          matrix_name="mlp.gate_proj")
    t1 = time.time()
    print(f"      Shape: {ref_weights.shape}")
    print(f"      Dtype: {ref_weights.dtype}")
    print(f"      N weights: {ref_weights.size:,}")
    print(f"      FP32 size: {ref_weights.nbytes / 1e6:.1f} MB")
    print(f"      Stats: mean={ref_weights.mean():.6f}, std={ref_weights.std():.6f}, "
          f"min={ref_weights.min():.6f}, max={ref_weights.max():.6f}")
    print(f"      Loaded in {t1 - t0:.1f}s")
    print()

    # Step 2: Re-quantize with each scheme
    schemes = [
        ("Shannon SMRS",    "3.076", shannon_quantize,   shannon_dequantize,   shannon_bpw,   shannon_size,   shannon_validate,   {"group_size": 128}),
        ("VonNeumann HGQ",  "3.000", vn_quantize,        vn_dequantize,        vn_bpw,        vn_size,        vn_validate,        {"group_size": 128}),
        ("Ferrucci EBAQ",   "2.930", ferrucci_quantize,   ferrucci_dequantize,  ferrucci_bpw,  ferrucci_size,  ferrucci_validate,  {"group_size": 32}),
        ("Turing-2.93",     "2.930", turing_quantize,     turing_dequantize,    turing_bpw,    turing_size,    turing_validate,    {"group_size": 128}),
        ("Lovelace ALE",    "2.931", lovelace_quantize,   lovelace_dequantize,  lovelace_bpw,  lovelace_size,  lovelace_validate,  {"group_size": 128}),
    ]

    results = []

    print("[2/3] Quantizing with each scheme ...")
    print()

    for name, target, q_fn, dq_fn, bpw_fn, size_fn, val_fn, kwargs in schemes:
        print(f"  {name} (target {target} bpw) ...", end=" ", flush=True)
        t0 = time.time()

        try:
            packed = q_fn(ref_weights, **kwargs)
            reconstructed = dq_fn(packed)
            bpw = bpw_fn(packed)
            size_bytes = size_fn(packed)
            size_mb = size_bytes / (1024 * 1024)
            cos_sim = cosine_similarity(ref_weights, reconstructed)
            mse_val = mse(ref_weights, reconstructed)
            rmse = relative_mse(ref_weights, reconstructed)
            valid = val_fn(ref_weights, packed, threshold=0.20)
            fits = size_mb <= BUDGET_MB

            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            results.append({
                "name": name,
                "target_bpw": target,
                "actual_bpw": bpw,
                "size_mb": size_mb,
                "mse": mse_val,
                "rmse": rmse,
                "cos_sim": cos_sim,
                "valid": valid,
                "fits_66mb": fits,
                "time_s": elapsed,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": name,
                "target_bpw": target,
                "actual_bpw": -1,
                "size_mb": -1,
                "mse": -1,
                "rmse": -1,
                "cos_sim": -1,
                "valid": False,
                "fits_66mb": False,
                "time_s": elapsed,
            })

    # Step 3: Print comparison table
    print()
    print("[3/3] Results")
    print()
    print("=" * 120)
    hdr = f"{'Scheme':<20} {'Target':>6} {'Actual':>7} {'Size MB':>8} {'Fits 66MB':>9} {'MSE':>12} {'Rel MSE':>10} {'Cosine':>10} {'Valid':>6} {'Time':>6}"
    print(hdr)
    print("-" * 120)

    for r in results:
        if r["actual_bpw"] < 0:
            print(f"{r['name']:<20} {r['target_bpw']:>6} {'FAILED':>7} {'--':>8} {'--':>9} {'--':>12} {'--':>10} {'--':>10} {'--':>6} {r['time_s']:>5.1f}s")
        else:
            fits_str = "YES" if r["fits_66mb"] else "NO"
            valid_str = "PASS" if r["valid"] else "FAIL"
            print(f"{r['name']:<20} {r['target_bpw']:>6} {r['actual_bpw']:>7.3f} {r['size_mb']:>8.2f} {fits_str:>9} {r['mse']:>12.2e} {r['rmse']:>10.6f} {r['cos_sim']:>10.6f} {valid_str:>6} {r['time_s']:>5.1f}s")

    print("=" * 120)
    print()

    # Summary for 180M weight layer extrapolation
    print("Extrapolation to 180M-weight layer (typical transformer layer):")
    print("-" * 90)
    n_weights = ref_weights.size
    for r in results:
        if r["actual_bpw"] > 0:
            extrapolated_mb = r["actual_bpw"] * 180_000_000 / 8 / (1024 * 1024)
            fits = "YES" if extrapolated_mb <= 66.01 else "NO"  # 10KB tolerance
            print(f"  {r['name']:<20}  {r['actual_bpw']:.3f} bpw  x 180M weights = {extrapolated_mb:>7.1f} MB  fits 66MB: {fits}")
    print()


if __name__ == "__main__":
    main()
