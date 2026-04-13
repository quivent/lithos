#!/usr/bin/env python3
"""
Requantize Qwen 3.5-27B from GPTQ W4A16 to Shannon Mixed-Radix Scheme (SMRS).

Pipeline:
  1. Load GPTQ-quantized model via LithosModel (mmap'd safetensors)
  2. For each weight matrix: dequantize GPTQ -> F32 -> requantize with Shannon SMRS
  3. Save Shannon-packed format as .npz files (one per layer)

GPTQ format (4-bit, group_size=128, symmetric with zero_point=8):
  - qweight: I32 [K/8, N] -- 8 4-bit weights packed per int32
  - scales:  F16 [K/128, N] -- per-group scale
  - qzeros:  I32 [K/128, N/8] -- packed 4-bit zeros (all 8 for symmetric)

Usage:
    python3 requantize_model.py [--layers 0-3] [--output-dir /tmp/shannon_weights]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import LithosModel
from shannon_smrs import quantize as shannon_quantize, dequantize as shannon_dequantize
from shannon_smrs import bits_per_weight, layer_size_bytes

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
GPTQ_BITS = 4
GPTQ_ZERO_POINT = 8  # symmetric quantization zero point


def dequantize_gptq(model: LithosModel, weight_prefix: str) -> np.ndarray:
    """Dequantize a single GPTQ weight matrix to F32.

    GPTQ 4-bit symmetric format:
      qweight[i, j]: int32 packing 8 consecutive 4-bit weights along K dimension
      scales[g, j]: FP16 scale for group g of column j
      qzeros[g, j//8]: int32 packing 8 4-bit zero points

    For row k in column j:
      - group g = k // 128
      - pack_row = k // 8, bit_offset = (k % 8) * 4
      - qval = (qweight[pack_row, j] >> bit_offset) & 0xF
      - zero = (qzeros[g, j // 8] >> ((j % 8) * 4)) & 0xF  (typically 8)
      - weight = (qval - zero) * scale[g, j]
    """
    qw_name = f"{weight_prefix}.qweight"
    sc_name = f"{weight_prefix}.scales"
    zr_name = f"{weight_prefix}.qzeros"

    qw_info = model.weight_info(qw_name)
    sc_info = model.weight_info(sc_name)
    zr_info = model.weight_info(zr_name)

    # Read raw bytes
    qw_raw = bytes(model.weight_bytes(qw_name))
    sc_raw = bytes(model.weight_bytes(sc_name))
    zr_raw = bytes(model.weight_bytes(zr_name))

    # Parse shapes
    qw_shape = qw_info.shape  # [K/8, N]
    sc_shape = sc_info.shape  # [n_groups, N]
    zr_shape = zr_info.shape  # [n_groups, N/8]

    K = qw_shape[0] * 8
    N = qw_shape[1]
    n_groups = sc_shape[0]

    # Convert to numpy -- fully vectorized dequantization
    qweight = np.frombuffer(qw_raw, dtype=np.int32).reshape(qw_shape).copy().view(np.uint32)
    scales = np.frombuffer(sc_raw, dtype=np.float16).reshape(sc_shape).astype(np.float32)
    qzeros_raw = np.frombuffer(zr_raw, dtype=np.int32).reshape(zr_shape).copy().view(np.uint32)

    # Unpack all 4-bit qweight values: [K/8, N] -> [K, N]
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 8, 1)
    qw_3d = qweight[:, np.newaxis, :]  # [K/8, 1, N]
    unpacked = ((qw_3d >> shifts) & 0xF).astype(np.float32).reshape(K, N)

    # Unpack qzeros: [n_groups, N/8] -> [n_groups, N]
    shifts_z = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 8, 1)
    zr_3d = qzeros_raw[:, np.newaxis, :]
    zeros_unpacked = ((zr_3d >> shifts_z) & 0xF).astype(np.float32).reshape(n_groups, N)

    # Dequantize per group
    result = np.zeros((K, N), dtype=np.float32)
    for g in range(n_groups):
        k_start = g * GROUP_SIZE
        k_end = min(k_start + GROUP_SIZE, K)
        result[k_start:k_end, :] = (unpacked[k_start:k_end, :] - zeros_unpacked[g]) * scales[g]

    return result


def dequantize_bf16(model: LithosModel, weight_name: str) -> np.ndarray:
    """Read a BF16 weight and convert to F32."""
    ti = model.weight_info(weight_name)
    raw = bytes(model.weight_bytes(weight_name))
    if ti.dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = np.zeros(len(u16), dtype=np.float32)
        f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
        return f32.reshape(ti.shape)
    elif ti.dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(ti.shape)
    elif ti.dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(ti.shape)
    else:
        raise ValueError(f"Unsupported dtype {ti.dtype} for {weight_name}")


def requantize_weight(f32_weights: np.ndarray, name: str, group_size: int = 128) -> dict:
    """Requantize F32 weights with Shannon SMRS and report quality."""
    packed = shannon_quantize(f32_weights, group_size=group_size)

    # Check roundtrip quality
    reconstructed = shannon_dequantize(packed)
    orig_flat = f32_weights.flatten().astype(np.float32)
    recon_flat = reconstructed.flatten()

    # Cosine similarity
    dot = np.dot(orig_flat, recon_flat)
    norm_o = np.linalg.norm(orig_flat)
    norm_r = np.linalg.norm(recon_flat)
    cosine = dot / (norm_o * norm_r + 1e-30)

    # Relative MSE
    mse = np.mean((orig_flat - recon_flat) ** 2)
    var = np.var(orig_flat) + 1e-30
    rel_mse = mse / var

    bpw = bits_per_weight(packed)
    size_bytes = layer_size_bytes(packed)

    print(f"    {name}: shape={f32_weights.shape} bpw={bpw:.3f} "
          f"cos={cosine:.4f} rMSE={rel_mse:.6f} size={size_bytes/1024:.1f}KB")

    return packed


def save_shannon_layer(layer_data: dict, output_path: str):
    """Save all Shannon-packed weight matrices for a layer to a single .npz file."""
    save_dict = {}
    for weight_name, packed in layer_data.items():
        prefix = weight_name.replace(".", "_")
        save_dict[f"{prefix}__n_weights"] = np.array([packed['n_weights']], dtype=np.int64)
        save_dict[f"{prefix}__group_size"] = np.array([packed['group_size']], dtype=np.int32)
        save_dict[f"{prefix}__n_groups"] = np.array([packed['n_groups']], dtype=np.int32)
        save_dict[f"{prefix}__n_class_s"] = np.array([packed['n_class_s']], dtype=np.int32)
        save_dict[f"{prefix}__original_shape"] = np.array(packed['original_shape'], dtype=np.int64)
        save_dict[f"{prefix}__scales"] = packed['scales']
        save_dict[f"{prefix}__zeros"] = packed['zeros']
        save_dict[f"{prefix}__is_class_s"] = packed['is_class_s']
        save_dict[f"{prefix}__packed_6"] = packed['packed_6']
        save_dict[f"{prefix}__packed_8"] = packed['packed_8']
        save_dict[f"{prefix}__group_class_s_indices"] = packed['group_class_s_indices']
        save_dict[f"{prefix}__group_class_i_indices"] = packed['group_class_i_indices']

    np.savez_compressed(output_path, **save_dict)


def load_shannon_layer(path: str) -> dict:
    """Load Shannon-packed weights from .npz file. Returns {weight_name: packed_dict}."""
    data = np.load(path)
    # Discover weight names from keys
    weight_names = set()
    for key in data.files:
        # key format: prefix__field
        name = key.rsplit("__", 1)[0]
        weight_names.add(name)

    result = {}
    for name in weight_names:
        packed = {
            'scheme': 'SMRS',
            'n_weights': int(data[f"{name}__n_weights"][0]),
            'group_size': int(data[f"{name}__group_size"][0]),
            'n_groups': int(data[f"{name}__n_groups"][0]),
            'n_class_s': int(data[f"{name}__n_class_s"][0]),
            'original_shape': tuple(data[f"{name}__original_shape"]),
            'scales': data[f"{name}__scales"],
            'zeros': data[f"{name}__zeros"],
            'is_class_s': data[f"{name}__is_class_s"],
            'packed_6': data[f"{name}__packed_6"],
            'packed_8': data[f"{name}__packed_8"],
            'group_class_s_indices': data[f"{name}__group_class_s_indices"],
            'group_class_i_indices': data[f"{name}__group_class_i_indices"],
        }
        # Convert prefix back to dotted name
        dotted_name = name.replace("_", ".", 1)  # Won't work perfectly; use saved name
        result[name] = packed

    return result


def requantize_layer(model: LithosModel, layer_idx: int, layer_types: list,
                     output_dir: str) -> dict:
    """Requantize all weight matrices for a single layer."""
    layer_type = layer_types[layer_idx]
    prefix = f"model.language_model.layers.{layer_idx}"

    print(f"\n  Layer {layer_idx} ({layer_type}):")
    layer_data = {}
    layer_stats = {"total_bytes": 0, "total_weights": 0}

    # MLP weights (all layers have MLP)
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        weight_prefix = f"{prefix}.mlp.{proj}"
        t0 = time.monotonic()
        f32 = dequantize_gptq(model, weight_prefix)
        t_deq = time.monotonic() - t0
        packed = requantize_weight(f32, f"mlp.{proj}", group_size=128)
        layer_data[f"mlp_{proj}"] = packed
        layer_stats["total_bytes"] += layer_size_bytes(packed)
        layer_stats["total_weights"] += packed['n_weights']
        del f32

    if layer_type == "full_attention":
        # Full attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_prefix = f"{prefix}.self_attn.{proj}"
            f32 = dequantize_gptq(model, weight_prefix)
            packed = requantize_weight(f32, f"self_attn.{proj}", group_size=128)
            layer_data[f"self_attn_{proj}"] = packed
            layer_stats["total_bytes"] += layer_size_bytes(packed)
            layer_stats["total_weights"] += packed['n_weights']
            del f32
    else:
        # DeltaNet projections (GPTQ-quantized)
        for proj in ["in_proj_qkv", "in_proj_z", "out_proj"]:
            weight_prefix = f"{prefix}.linear_attn.{proj}"
            f32 = dequantize_gptq(model, weight_prefix)
            packed = requantize_weight(f32, f"linear_attn.{proj}", group_size=128)
            layer_data[f"linear_attn_{proj}"] = packed
            layer_stats["total_bytes"] += layer_size_bytes(packed)
            layer_stats["total_weights"] += packed['n_weights']
            del f32

    total_mb = layer_stats["total_bytes"] / (1024 * 1024)
    total_bpw = (layer_stats["total_bytes"] * 8) / layer_stats["total_weights"]
    print(f"  Layer {layer_idx} totals: {total_mb:.2f} MB, {total_bpw:.3f} bpw, "
          f"{layer_stats['total_weights']:,} weights")

    # Save
    output_path = os.path.join(output_dir, f"layer_{layer_idx:03d}.npz")
    save_shannon_layer(layer_data, output_path)
    actual_file_size = os.path.getsize(output_path + "" if output_path.endswith(".npz") else output_path)
    print(f"  Saved: {output_path} ({actual_file_size / (1024*1024):.2f} MB on disk)")

    return layer_stats


def main():
    parser = argparse.ArgumentParser(description="Requantize Qwen 3.5-27B with Shannon SMRS")
    parser.add_argument("--layers", type=str, default="0-63",
                        help="Layer range, e.g. '0-3' or '0' or '0-63'")
    parser.add_argument("--output-dir", type=str, default="/tmp/shannon_weights",
                        help="Output directory for Shannon-packed weights")
    args = parser.parse_args()

    # Parse layer range
    if "-" in args.layers:
        start, end = args.layers.split("-")
        layer_range = range(int(start), int(end) + 1)
    else:
        layer_range = range(int(args.layers), int(args.layers) + 1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading GPTQ model from {MODEL_DIR}...")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    print(f"  Loaded in {time.monotonic() - t0:.2f}s")
    print(f"  {model}")

    layer_types = model.config.layer_types

    total_stats = {"total_bytes": 0, "total_weights": 0}
    t_all = time.monotonic()

    for layer_idx in layer_range:
        t_layer = time.monotonic()
        stats = requantize_layer(model, layer_idx, layer_types, args.output_dir)
        total_stats["total_bytes"] += stats["total_bytes"]
        total_stats["total_weights"] += stats["total_weights"]
        elapsed = time.monotonic() - t_layer
        print(f"  Layer {layer_idx} done in {elapsed:.1f}s")

    total_elapsed = time.monotonic() - t_all
    total_mb = total_stats["total_bytes"] / (1024 * 1024)
    total_bpw = (total_stats["total_bytes"] * 8) / max(total_stats["total_weights"], 1)

    print(f"\n{'='*72}")
    print(f"  REQUANTIZATION COMPLETE")
    print(f"  Layers: {list(layer_range)}")
    print(f"  Total Shannon size: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
    print(f"  Average bpw: {total_bpw:.3f}")
    print(f"  Total weights: {total_stats['total_weights']:,}")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*72}")

    model.close()


if __name__ == "__main__":
    main()
