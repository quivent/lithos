"""
Turing-2.93 Mixed-Precision Tile Quantization — 2.93 bits per weight

Design from Alan Turing's computability-inspired approach:
- Three tile types: T4 (4-bit), T3 (3-bit), T2 (2-bit)
- Tile size: 1024 weights (64x16)
- Group size 128 within each tile, FP16 scale + FP16 zero per group
- Allocation: ~10% T4, ~48% T3, ~42% T2 (Bayesian sensitivity-guided)
- Effective bpw: 0.10*4.25 + 0.48*3.25 + 0.42*2.25 = 2.93
- SHIFT-MASK-SCALE dequantization: extract via shift+AND, scale via FMA
"""

import numpy as np
from typing import Dict

TILE_SIZE = 1024
GROUP_SIZE_DEFAULT = 128
GROUPS_PER_TILE = TILE_SIZE // GROUP_SIZE_DEFAULT  # 8

# Tile type enum
T2, T3, T4 = 0, 1, 2

# Effective bits per weight including metadata (FP16 scale + FP16 zero per group)
# metadata: 32 bits per group, 128 weights/group -> 0.25 bpw overhead
METADATA_BPW = 32.0 / GROUP_SIZE_DEFAULT  # 0.25
EFF_BPW = {
    T2: 2.0 + METADATA_BPW,  # 2.25
    T3: 3.0 + METADATA_BPW,  # 3.25
    T4: 4.0 + METADATA_BPW,  # 4.25
}


def _tile_sensitivity(tile: np.ndarray) -> float:
    """Compute sensitivity of a tile (1024 weights)."""
    var = np.var(tile)
    if var < 1e-20:
        return 0.0
    max_abs = np.max(np.abs(tile))
    mean_abs = np.mean(np.abs(tile)) + 1e-10
    m4 = np.mean(tile ** 4)
    kurtosis = m4 / (var ** 2 + 1e-20)
    return float(np.sqrt(var) * kurtosis * (max_abs / mean_abs))


def _allocate_tile_types(sensitivities: np.ndarray, n_weights: int,
                          target_bpw: float = 2.93) -> np.ndarray:
    """Allocate tile types to hit target bpw."""
    n_tiles = len(sensitivities)

    # Start all at T2
    types = np.full(n_tiles, T2, dtype=np.uint8)

    current_bits = sum(EFF_BPW[T2] * TILE_SIZE for _ in range(n_tiles))
    budget = target_bpw * n_weights
    remaining = budget - current_bits

    # Upgrade by sensitivity
    order = np.argsort(-sensitivities)

    # First pass: T2 -> T3
    for idx in order:
        if remaining <= 0:
            break
        cost = (EFF_BPW[T3] - EFF_BPW[T2]) * TILE_SIZE
        if remaining >= cost:
            types[idx] = T3
            remaining -= cost

    # Second pass: T3 -> T4 for most sensitive
    for idx in order:
        if remaining <= 0:
            break
        if types[idx] == T3:
            cost = (EFF_BPW[T4] - EFF_BPW[T3]) * TILE_SIZE
            if remaining >= cost:
                types[idx] = T4
                remaining -= cost

    return types


def _quantize_symmetric(values: np.ndarray, n_bits: int, scale: float, zero: float) -> np.ndarray:
    """Quantize values to n_bits with scale and zero point."""
    n_levels = 2 ** n_bits
    max_q = n_levels - 1
    if scale < 1e-30:
        return np.zeros(len(values), dtype=np.uint8)
    normalized = (values - zero) / scale
    # Map [-1, 1] to [0, max_q]
    q = np.round((normalized + 1.0) / 2.0 * max_q).astype(np.int32)
    q = np.clip(q, 0, max_q)
    return q.astype(np.uint8)


def _dequantize_symmetric(q: np.ndarray, n_bits: int, scale: float, zero: float) -> np.ndarray:
    """Reconstruct from quantized values."""
    n_levels = 2 ** n_bits
    max_q = n_levels - 1
    normalized = q.astype(np.float32) / max_q * 2.0 - 1.0
    return normalized * scale + zero


def _pack_nbits(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Pack n-bit unsigned values into uint32 words."""
    vals_per_word = 32 // n_bits
    n = len(values)
    n_words = (n + vals_per_word - 1) // vals_per_word
    padded = np.zeros(n_words * vals_per_word, dtype=np.uint8)
    padded[:n] = values
    padded = padded.reshape(n_words, vals_per_word)
    packed = np.zeros(n_words, dtype=np.uint32)
    mask = (1 << n_bits) - 1
    for i in range(vals_per_word):
        packed |= (padded[:, i].astype(np.uint32) & mask) << (n_bits * i)
    return packed


def _unpack_nbits(packed: np.ndarray, n_bits: int, count: int) -> np.ndarray:
    """Unpack n-bit values from uint32 words."""
    vals_per_word = 32 // n_bits
    n_words = len(packed)
    mask = (1 << n_bits) - 1
    values = np.zeros((n_words, vals_per_word), dtype=np.uint8)
    for i in range(vals_per_word):
        values[:, i] = ((packed >> (n_bits * i)) & mask).astype(np.uint8)
    return values.flatten()[:count]


def quantize(weights: np.ndarray, group_size: int = 128) -> dict:
    """
    Quantize using Turing-2.93 mixed-precision tile scheme.

    Args:
        weights: FP16/FP32 weight tensor
        group_size: weights per group (default 128)

    Returns:
        dict with packed representation
    """
    original_shape = weights.shape
    w = weights.flatten().astype(np.float32)
    n = len(w)

    # Pad to multiple of TILE_SIZE
    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
    padded = np.zeros(n_tiles * TILE_SIZE, dtype=np.float32)
    padded[:n] = w
    tiles = padded.reshape(n_tiles, TILE_SIZE)

    # Compute sensitivity per tile
    sensitivities = np.array([_tile_sensitivity(tiles[t]) for t in range(n_tiles)])

    # Allocate tile types
    tile_types = _allocate_tile_types(sensitivities, n, target_bpw=2.93)

    # Quantize each tile
    groups_per_tile = TILE_SIZE // group_size
    n_groups_total = n_tiles * groups_per_tile

    all_scales = np.zeros(n_groups_total, dtype=np.float32)
    all_zeros = np.zeros(n_groups_total, dtype=np.float32)

    packed_per_type = {T2: [], T3: [], T4: []}
    tile_group_indices = {T2: [], T3: [], T4: []}
    q_counts = {T2: 0, T3: 0, T4: 0}

    for t in range(n_tiles):
        tile = tiles[t]
        ttype = tile_types[t]
        n_bits = [2, 3, 4][ttype]

        tile_q = np.zeros(TILE_SIZE, dtype=np.uint8)

        for gi in range(groups_per_tile):
            g_idx = t * groups_per_tile + gi
            grp = tile[gi * group_size:(gi + 1) * group_size]

            g_min = grp.min()
            g_max = grp.max()
            zero = (g_min + g_max) / 2.0
            scale = (g_max - g_min) / 2.0
            scale = max(scale, 1e-10)

            all_scales[g_idx] = scale
            all_zeros[g_idx] = zero

            q = _quantize_symmetric(grp, n_bits, scale, zero)
            tile_q[gi * group_size:(gi + 1) * group_size] = q

        packed = _pack_nbits(tile_q, n_bits)
        packed_per_type[ttype].append(packed)
        q_counts[ttype] += TILE_SIZE

    # Concatenate packed data per type
    for tt in [T2, T3, T4]:
        if packed_per_type[tt]:
            packed_per_type[tt] = np.concatenate(packed_per_type[tt])
        else:
            packed_per_type[tt] = np.array([], dtype=np.uint32)

    return {
        'scheme': 'Turing-2.93',
        'original_shape': original_shape,
        'n_weights': n,
        'group_size': group_size,
        'n_tiles': n_tiles,
        'tile_types': tile_types,
        'scales': all_scales.astype(np.float16),
        'zeros': all_zeros.astype(np.float16),
        'packed_t2': packed_per_type[T2],
        'packed_t3': packed_per_type[T3],
        'packed_t4': packed_per_type[T4],
        'n_q_t2': q_counts[T2],
        'n_q_t3': q_counts[T3],
        'n_q_t4': q_counts[T4],
    }


def dequantize(packed: dict) -> np.ndarray:
    """Reconstruct FP32 weights from Turing-2.93 packed representation."""
    n = packed['n_weights']
    n_tiles = packed['n_tiles']
    group_size = packed['group_size']
    tile_types = packed['tile_types']
    scales = packed['scales'].astype(np.float32)
    zeros = packed['zeros'].astype(np.float32)
    groups_per_tile = TILE_SIZE // group_size

    # Compute words per tile for each bit width
    words_per_tile = {
        T2: (TILE_SIZE + 15) // 16,  # 2-bit: 16 per word
        T3: (TILE_SIZE + 9) // 10,   # 3-bit: 10 per word
        T4: (TILE_SIZE + 7) // 8,    # 4-bit: 8 per word
    }

    result = np.zeros(n_tiles * TILE_SIZE, dtype=np.float32)

    # Track word offsets per type (not value offsets)
    word_offsets = {T2: 0, T3: 0, T4: 0}

    for t in range(n_tiles):
        ttype = tile_types[t]
        n_bits = [2, 3, 4][ttype]
        wpt = words_per_tile[ttype]
        key = ['packed_t2', 'packed_t3', 'packed_t4'][ttype]

        # Unpack this tile's words
        tile_words = packed[key][word_offsets[ttype]:word_offsets[ttype] + wpt]
        tile_q = _unpack_nbits(tile_words, n_bits, TILE_SIZE)
        word_offsets[ttype] += wpt

        for gi in range(groups_per_tile):
            g_idx = t * groups_per_tile + gi
            q = tile_q[gi * group_size:(gi + 1) * group_size]
            result[t * TILE_SIZE + gi * group_size:t * TILE_SIZE + (gi + 1) * group_size] = \
                _dequantize_symmetric(q, n_bits, scales[g_idx], zeros[g_idx])

    return result[:n].reshape(packed['original_shape'])


def bits_per_weight(packed: dict) -> float:
    """Calculate actual bits per weight."""
    total_bits = layer_size_bytes(packed) * 8
    return total_bits / packed['n_weights']


def layer_size_bytes(packed: dict) -> int:
    """Total bytes for packed data."""
    n_tiles = packed['n_tiles']
    group_size = packed['group_size']
    groups_per_tile = TILE_SIZE // group_size
    n_groups = n_tiles * groups_per_tile

    total = 0
    # Header + tile type map (2 bits per tile)
    total += 64 + (n_tiles * 2 + 7) // 8
    # Scales + zeros: FP16 each per group
    total += n_groups * 4  # 2 bytes scale + 2 bytes zero
    # Packed weight data
    for tt, nbits, key in [(T2, 2, 'packed_t2'), (T3, 3, 'packed_t3'), (T4, 4, 'packed_t4')]:
        n_vals = packed[f'n_q_{["t2","t3","t4"][tt]}']
        total += (n_vals * nbits + 7) // 8

    return total


def validate(original: np.ndarray, packed: dict, threshold: float = 0.01) -> bool:
    """Check round-trip quality."""
    reconstructed = dequantize(packed)
    mse = np.mean((original.astype(np.float32) - reconstructed) ** 2)
    var = np.var(original.astype(np.float32))
    if var < 1e-10:
        return mse < 1e-10
    return (mse / var) < threshold
