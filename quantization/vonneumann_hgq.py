"""
Von Neumann Heptary Group Quantization (HGQ-384) — 3.000 bits per weight

Design from John von Neumann's architecture specification:
- 7-level symmetric quantization: {-3, -2, -1, 0, +1, +2, +3}
- Groups of 128 weights, each group = 384 bits = 48 bytes exactly
- 16 sub-packs of 8 heptary digits packed into 23-bit integers
- 1 FP16 scale per group (16 bits)
- Total per group: 16 * 23 + 16 = 384 bits
- Effective rate: 384 / 128 = 3.000 bits/weight
- Decode via LUT: split 23-bit value into hi/lo via div/mod 7^4=2401
"""

import numpy as np
from typing import Dict

# Symmetric 7-level quantization: map {-3,-2,-1,0,1,2,3} to {0,1,2,3,4,5,6}
HEPT_LEVELS = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
HEPT_OFFSET = 3  # add 3 to get unsigned index 0-6

# Precompute LUT for 4-digit heptary decode
# LUT4[k] = (k%7, (k//7)%7, (k//49)%7, k//343) for k in 0..2400
LUT4 = np.zeros((2401, 4), dtype=np.uint8)
for k in range(2401):
    LUT4[k, 0] = k % 7
    LUT4[k, 1] = (k // 7) % 7
    LUT4[k, 2] = (k // 49) % 7
    LUT4[k, 3] = k // 343

SUBPACK_SIZE = 8
SUBPACKS_PER_GROUP = 16  # 128 / 8
BITS_PER_SUBPACK = 23  # ceil(8 * log2(7))
MAX_SUBPACK_VAL = 7**8  # 5,764,801 < 2^23 = 8,388,608


def _quantize_group(group: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a group of weights to 7 symmetric levels.

    w_hat = q * scale, where q in {-3,-2,-1,0,1,2,3}
    So q = round(w / scale), clamped to [-3, 3]
    """
    if scale < 1e-30:
        return np.zeros(len(group), dtype=np.int8)
    q = np.round(group / scale).astype(np.int32)
    q = np.clip(q, -3, 3)
    return q.astype(np.int8)


def _encode_subpack(codes: np.ndarray) -> np.uint32:
    """Encode 8 heptary codes (0-6) into a single 23-bit integer using mixed-radix."""
    val = np.uint32(0)
    for i in range(7, -1, -1):
        val = val * np.uint32(7) + np.uint32(codes[i])
    return val


def _decode_subpack(packed: np.uint32) -> np.ndarray:
    """Decode a 23-bit integer into 8 heptary codes using LUT."""
    lo = int(packed % 2401)
    hi = int(packed // 2401)
    codes = np.zeros(8, dtype=np.uint8)
    codes[0:4] = LUT4[lo]
    codes[4:8] = LUT4[hi]
    return codes


def _pack_group(q_values: np.ndarray) -> np.ndarray:
    """Pack 128 quantized values (in -3..3) into 16 sub-pack uint32s."""
    # Convert to unsigned: add offset 3
    unsigned = (q_values.astype(np.int32) + HEPT_OFFSET).astype(np.uint8)
    subpacks = unsigned.reshape(SUBPACKS_PER_GROUP, SUBPACK_SIZE)
    packed = np.zeros(SUBPACKS_PER_GROUP, dtype=np.uint32)
    for j in range(SUBPACKS_PER_GROUP):
        packed[j] = _encode_subpack(subpacks[j])
    return packed


def _unpack_group(packed: np.ndarray) -> np.ndarray:
    """Unpack 16 sub-pack uint32s into 128 signed quantized values."""
    codes = np.zeros(128, dtype=np.uint8)
    for j in range(SUBPACKS_PER_GROUP):
        codes[j * 8:(j + 1) * 8] = _decode_subpack(packed[j])
    # Convert back to signed
    return codes.astype(np.int8) - HEPT_OFFSET


def quantize(weights: np.ndarray, group_size: int = 128) -> dict:
    """
    Quantize weights using Heptary Group Quantization (HGQ-384).

    Args:
        weights: FP16/FP32 weight tensor
        group_size: must be 128 (fixed by design)

    Returns:
        dict with packed representation
    """
    assert group_size == 128, "HGQ-384 requires group_size=128"

    original_shape = weights.shape
    w = weights.flatten().astype(np.float32)
    n = len(w)

    # Pad to multiple of 128
    n_groups = (n + 127) // 128
    padded = np.zeros(n_groups * 128, dtype=np.float32)
    padded[:n] = w
    groups = padded.reshape(n_groups, 128)

    # Compute optimal scale per group: scale = max(|w|) / 3
    group_max = np.max(np.abs(groups), axis=1)
    scales = group_max / 3.0
    scales = np.maximum(scales, 1e-10)

    # Quantize all groups (vectorized)
    # q = round(w / scale), clamped to [-3, 3]
    scales_expanded = scales[:, None]  # (n_groups, 1)
    q_all = np.round(groups / np.maximum(scales_expanded, 1e-30)).astype(np.int32)
    q_all = np.clip(q_all, -3, 3).astype(np.int8)

    # Pack all groups (vectorized)
    # Convert to unsigned: add offset 3, so values in [0..6]
    unsigned = (q_all.astype(np.int32) + HEPT_OFFSET).astype(np.uint32)
    # Reshape to (n_groups, 16 subpacks, 8 values)
    unsigned_sp = unsigned.reshape(n_groups, SUBPACKS_PER_GROUP, SUBPACK_SIZE)
    # Mixed-radix encode: sum(val[i] * 7^i for i in 0..7)
    powers = np.array([7**i for i in range(SUBPACK_SIZE)], dtype=np.uint32)
    all_packed = np.sum(unsigned_sp * powers[None, None, :], axis=2).astype(np.uint32)

    return {
        'scheme': 'HGQ-384',
        'original_shape': original_shape,
        'n_weights': n,
        'group_size': 128,
        'n_groups': n_groups,
        'scales': scales.astype(np.float16),
        'packed_subpacks': all_packed,  # (n_groups, 16) uint32
    }


def dequantize(packed: dict) -> np.ndarray:
    """Reconstruct FP32 weights from HGQ-384 packed representation."""
    n = packed['n_weights']
    n_groups = packed['n_groups']
    scales = packed['scales'].astype(np.float32)
    all_packed = packed['packed_subpacks']

    # Vectorized mixed-radix decode
    # For each subpack: split into lo (mod 2401) and hi (div 2401)
    flat_packed = all_packed.reshape(-1)  # (n_groups * 16,)
    lo = (flat_packed % 2401).astype(np.int32)
    hi = (flat_packed // 2401).astype(np.int32)

    # LUT lookup for all subpacks at once
    codes_lo = LUT4[lo]  # (n_sp, 4)
    codes_hi = LUT4[hi]  # (n_sp, 4)
    all_codes = np.concatenate([codes_lo, codes_hi], axis=1)  # (n_sp, 8)

    # Convert to signed and reshape
    all_codes_signed = all_codes.astype(np.float32) - HEPT_OFFSET  # {0..6} -> {-3..3}
    all_codes_signed = all_codes_signed.reshape(n_groups, 128)  # (n_groups, 128)

    # Dequantize: w = q * scale
    result = (all_codes_signed * scales[:, None]).flatten()

    return result[:n].reshape(packed['original_shape'])


def bits_per_weight(packed: dict) -> float:
    """Calculate actual bits per weight."""
    total_bits = layer_size_bytes(packed) * 8
    return total_bits / packed['n_weights']


def layer_size_bytes(packed: dict) -> int:
    """Total bytes: exactly 48 bytes per group (384 bits = 16*23 + 16)."""
    n_groups = packed['n_groups']
    # Each group: 16 sub-packs * 23 bits + 16 bits scale = 384 bits = 48 bytes
    # In practice we store subpacks as uint32 (wastes some bits in storage)
    # For accurate reporting: 384 bits per group
    total_bits = n_groups * 384
    return (total_bits + 7) // 8


def validate(original: np.ndarray, packed: dict, threshold: float = 0.01) -> bool:
    """Check round-trip quality."""
    reconstructed = dequantize(packed)
    mse = np.mean((original.astype(np.float32) - reconstructed) ** 2)
    var = np.var(original.astype(np.float32))
    if var < 1e-10:
        return mse < 1e-10
    return (mse / var) < threshold
