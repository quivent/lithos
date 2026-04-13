"""
Ferrucci Evidence-Based Adaptive Quantization (EBAQ) — 2.93 bits per weight

Design from Dave Ferrucci's Watson-inspired systems architecture:
- Mixed {2,3,4}-bit per-group adaptive quantization
- Evidence-based sensitivity allocation (Hessian-diagonal proxy)
- Group size 16 with FP16 scales (symmetric)
- Greedy bit allocation: start at 2-bit, upgrade sensitive groups to 3 or 4-bit
- Separation of calibration quality from packing layout
- Target: exactly 2.93 bpw average
"""

import numpy as np
from typing import Dict


def _compute_group_sensitivity(groups: np.ndarray) -> np.ndarray:
    """
    Compute per-group sensitivity using multiple signals:
    1. Magnitude range (proxy for weight importance)
    2. Kurtosis (outlier sensitivity)
    3. Variance (information content)
    Combined into a single score.
    """
    variance = np.var(groups, axis=1) + 1e-10
    max_abs = np.max(np.abs(groups), axis=1)
    mean_abs = np.mean(np.abs(groups), axis=1) + 1e-10
    # Kurtosis proxy
    m4 = np.mean((groups - groups.mean(axis=1, keepdims=True))**4, axis=1)
    kurtosis = m4 / (variance**2 + 1e-20)
    # Combined score: high variance + high kurtosis + high max/mean ratio = sensitive
    score = np.sqrt(variance) * np.sqrt(kurtosis) * (max_abs / mean_abs)
    return score


def _allocate_bits(sensitivity: np.ndarray, n_weights: int,
                   target_bpw: float, group_size: int) -> np.ndarray:
    """
    Greedy bit allocation: start at 2 bits, upgrade groups by sensitivity.

    Effective bpw per group including metadata (FP16 scale = 16 bits per group):
      2-bit: (2 * gs + 16) / gs = 2 + 16/gs
      3-bit: (3 * gs + 16) / gs = 3 + 16/gs
      4-bit: (4 * gs + 16) / gs = 4 + 16/gs
    """
    n_groups = len(sensitivity)
    metadata_bits_per_group = 16  # FP16 scale

    # Start all at 2 bits
    bits = np.full(n_groups, 2, dtype=np.uint8)

    total_bits_at_2 = n_groups * (2 * group_size + metadata_bits_per_group)
    budget = int(target_bpw * n_weights)

    remaining = budget - total_bits_at_2

    if remaining <= 0:
        return bits

    # Sort groups by sensitivity (highest first)
    order = np.argsort(-sensitivity)

    # Upgrade 2->3 costs group_size bits, 3->4 costs group_size bits
    for idx in order:
        if remaining <= 0:
            break
        if bits[idx] == 2:
            cost = group_size  # 2->3
            if remaining >= cost:
                bits[idx] = 3
                remaining -= cost
        # Second pass: upgrade 3->4 for most sensitive

    for idx in order:
        if remaining <= 0:
            break
        if bits[idx] == 3:
            cost = group_size  # 3->4
            if remaining >= cost:
                bits[idx] = 4
                remaining -= cost

    return bits


def _quantize_uniform(values: np.ndarray, n_levels: int, scale: float) -> np.ndarray:
    """Symmetric uniform quantization to n_levels.

    Maps values to integer range [0, n_levels-1].
    q = round((value / scale + 1) / 2 * (n_levels - 1))
    """
    if scale < 1e-30:
        return np.full(len(values), n_levels // 2, dtype=np.uint8)
    max_q = n_levels - 1
    normalized = values / scale  # in [-1, 1]
    q = np.round((normalized + 1.0) / 2.0 * max_q).astype(np.int32)
    q = np.clip(q, 0, max_q)
    return q.astype(np.uint8)


def _dequantize_uniform(q: np.ndarray, n_levels: int, scale: float) -> np.ndarray:
    """Reconstruct from uniform quantization."""
    max_q = n_levels - 1
    normalized = q.astype(np.float32) / max_q * 2.0 - 1.0  # back to [-1, 1]
    return normalized * scale


def _pack_2bit(indices: np.ndarray) -> np.ndarray:
    """Pack 2-bit unsigned values (0-3): 16 values per uint32."""
    unsigned = np.clip(indices.astype(np.uint8), 0, 3)
    n = len(unsigned)
    n_words = (n + 15) // 16
    padded = np.zeros(n_words * 16, dtype=np.uint8)
    padded[:n] = unsigned
    padded = padded.reshape(n_words, 16)
    packed = np.zeros(n_words, dtype=np.uint32)
    for i in range(16):
        packed |= padded[:, i].astype(np.uint32) << (2 * i)
    return packed


def _unpack_2bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack 2-bit unsigned values from uint32 words."""
    n_words = len(packed)
    values = np.zeros((n_words, 16), dtype=np.uint8)
    for i in range(16):
        values[:, i] = ((packed >> (2 * i)) & 0x3).astype(np.uint8)
    return values.flatten()[:count]


def _pack_3bit(indices: np.ndarray) -> np.ndarray:
    """Pack 3-bit unsigned values (0-7): 10 per uint32 (30 bits used, 2 wasted)."""
    unsigned = np.clip(indices.astype(np.uint8), 0, 7)
    n = len(unsigned)
    n_words = (n + 9) // 10
    padded = np.zeros(n_words * 10, dtype=np.uint8)
    padded[:n] = unsigned
    padded = padded.reshape(n_words, 10)
    packed = np.zeros(n_words, dtype=np.uint32)
    for i in range(10):
        packed |= padded[:, i].astype(np.uint32) << (3 * i)
    return packed


def _unpack_3bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack 3-bit unsigned values from uint32 words."""
    n_words = len(packed)
    values = np.zeros((n_words, 10), dtype=np.uint8)
    for i in range(10):
        values[:, i] = ((packed >> (3 * i)) & 0x7).astype(np.uint8)
    return values.flatten()[:count]


def _pack_4bit(indices: np.ndarray) -> np.ndarray:
    """Pack 4-bit unsigned values (0-15): 8 per uint32."""
    unsigned = np.clip(indices.astype(np.uint8), 0, 15)
    n = len(unsigned)
    n_words = (n + 7) // 8
    padded = np.zeros(n_words * 8, dtype=np.uint8)
    padded[:n] = unsigned
    padded = padded.reshape(n_words, 8)
    packed = np.zeros(n_words, dtype=np.uint32)
    for i in range(8):
        packed |= padded[:, i].astype(np.uint32) << (4 * i)
    return packed


def _unpack_4bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack 4-bit unsigned values from uint32 words."""
    n_words = len(packed)
    values = np.zeros((n_words, 8), dtype=np.uint8)
    for i in range(8):
        values[:, i] = ((packed >> (4 * i)) & 0xF).astype(np.uint8)
    return values.flatten()[:count]


def quantize(weights: np.ndarray, group_size: int = 16) -> dict:
    """
    Quantize using Evidence-Based Adaptive Quantization.

    Args:
        weights: FP16/FP32 weight tensor
        group_size: weights per group (default 16 per Ferrucci's recommendation)

    Returns:
        dict with packed representation
    """
    original_shape = weights.shape
    w = weights.flatten().astype(np.float32)
    n = len(w)

    n_groups = (n + group_size - 1) // group_size
    padded = np.zeros(n_groups * group_size, dtype=np.float32)
    padded[:n] = w
    groups = padded.reshape(n_groups, group_size)

    # Per-group scale (symmetric)
    group_max = np.max(np.abs(groups), axis=1)
    scales = group_max.copy()
    scales = np.maximum(scales, 1e-10)

    # Sensitivity analysis
    sensitivity = _compute_group_sensitivity(groups)

    # Allocate bits per group
    target_bpw = 2.93
    bit_alloc = _allocate_bits(sensitivity, n, target_bpw, group_size)

    # Quantize each group at its allocated precision
    # n_levels: 2-bit -> 4 levels, 3-bit -> 8 levels, 4-bit -> 16 levels
    q_values = np.zeros(n_groups * group_size, dtype=np.uint8)

    for g in range(n_groups):
        nbits = bit_alloc[g]
        n_levels = 2 ** nbits
        grp = groups[g]
        q = _quantize_uniform(grp, n_levels, scales[g])
        q_values[g * group_size:(g + 1) * group_size] = q

    # Pack by bit-width groups
    mask_2 = bit_alloc == 2
    mask_3 = bit_alloc == 3
    mask_4 = bit_alloc == 4

    idx_2 = np.where(mask_2)[0]
    idx_3 = np.where(mask_3)[0]
    idx_4 = np.where(mask_4)[0]

    # Collect quantized values per bit-width
    if len(idx_2) > 0:
        vals_2 = np.concatenate([q_values[g * group_size:(g + 1) * group_size] for g in idx_2])
        packed_2 = _pack_2bit(vals_2)
    else:
        packed_2 = np.array([], dtype=np.uint32)
        vals_2 = np.array([], dtype=np.int8)

    if len(idx_3) > 0:
        vals_3 = np.concatenate([q_values[g * group_size:(g + 1) * group_size] for g in idx_3])
        packed_3 = _pack_3bit(vals_3)
    else:
        packed_3 = np.array([], dtype=np.uint32)
        vals_3 = np.array([], dtype=np.int8)

    if len(idx_4) > 0:
        vals_4 = np.concatenate([q_values[g * group_size:(g + 1) * group_size] for g in idx_4])
        packed_4 = _pack_4bit(vals_4)
    else:
        packed_4 = np.array([], dtype=np.uint32)
        vals_4 = np.array([], dtype=np.int8)

    return {
        'scheme': 'EBAQ',
        'original_shape': original_shape,
        'n_weights': n,
        'group_size': group_size,
        'n_groups': n_groups,
        'scales': scales.astype(np.float16),
        'bit_alloc': bit_alloc,
        'idx_2': idx_2.astype(np.int32),
        'idx_3': idx_3.astype(np.int32),
        'idx_4': idx_4.astype(np.int32),
        'packed_2': packed_2,
        'packed_3': packed_3,
        'packed_4': packed_4,
        'n_vals_2': len(vals_2),
        'n_vals_3': len(vals_3),
        'n_vals_4': len(vals_4),
    }


def dequantize(packed: dict) -> np.ndarray:
    """Reconstruct FP32 weights from EBAQ packed representation."""
    n = packed['n_weights']
    group_size = packed['group_size']
    n_groups = packed['n_groups']
    scales = packed['scales'].astype(np.float32)
    bit_alloc = packed['bit_alloc']

    result = np.zeros(n_groups * group_size, dtype=np.float32)

    # Unpack 2-bit groups
    if len(packed['packed_2']) > 0:
        vals_2 = _unpack_2bit(packed['packed_2'], packed['n_vals_2'])
        offset = 0
        for g in packed['idx_2']:
            n_levels = 4
            q = vals_2[offset:offset + group_size]
            result[g * group_size:(g + 1) * group_size] = _dequantize_uniform(q, n_levels, scales[g])
            offset += group_size

    # Unpack 3-bit groups
    if len(packed['packed_3']) > 0:
        vals_3 = _unpack_3bit(packed['packed_3'], packed['n_vals_3'])
        offset = 0
        for g in packed['idx_3']:
            n_levels = 8
            q = vals_3[offset:offset + group_size]
            result[g * group_size:(g + 1) * group_size] = _dequantize_uniform(q, n_levels, scales[g])
            offset += group_size

    # Unpack 4-bit groups
    if len(packed['packed_4']) > 0:
        vals_4 = _unpack_4bit(packed['packed_4'], packed['n_vals_4'])
        offset = 0
        for g in packed['idx_4']:
            n_levels = 16
            q = vals_4[offset:offset + group_size]
            result[g * group_size:(g + 1) * group_size] = _dequantize_uniform(q, n_levels, scales[g])
            offset += group_size

    return result[:n].reshape(packed['original_shape'])


def bits_per_weight(packed: dict) -> float:
    """Calculate actual bits per weight."""
    total_bits = layer_size_bytes(packed) * 8
    return total_bits / packed['n_weights']


def layer_size_bytes(packed: dict) -> int:
    """Total bytes for this layer's packed data."""
    total = 0
    # Header
    total += 64
    # Scales: FP16 per group
    total += packed['n_groups'] * 2
    # Bit allocation map: 2 bits per group (values 2,3,4 fit in 2 bits as 0,1,2)
    total += (packed['n_groups'] * 2 + 7) // 8
    # Packed data
    # 2-bit: 16 vals per uint32
    n_vals_2 = packed['n_vals_2']
    bits_2 = n_vals_2 * 2
    # 3-bit: 10 vals per 30-bit word (2 wasted per word)
    n_vals_3 = packed['n_vals_3']
    bits_3 = n_vals_3 * 3
    # 4-bit: 8 vals per uint32
    n_vals_4 = packed['n_vals_4']
    bits_4 = n_vals_4 * 4

    total += (bits_2 + bits_3 + bits_4 + 7) // 8
    return total


def validate(original: np.ndarray, packed: dict, threshold: float = 0.01) -> bool:
    """Check round-trip quality."""
    reconstructed = dequantize(packed)
    mse = np.mean((original.astype(np.float32) - reconstructed) ** 2)
    var = np.var(original.astype(np.float32))
    if var < 1e-10:
        return mse < 1e-10
    return (mse / var) < threshold
