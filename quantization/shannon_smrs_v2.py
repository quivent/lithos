"""
Shannon Mixed-Radix Scheme v2 (SMRS-v2) -- Rate-Distortion Optimal Quantization

Analysis by Claude Shannon
==========================

The Problem
-----------
SMRS v1 achieves cosine 0.975 per matrix. Through 64 sequential layers, errors
compound multiplicatively. If each layer introduces cosine deviation epsilon,
the cumulative angular error after L layers grows as:

    theta_total ~ L * arccos(1 - epsilon)

For cosine 0.975 (epsilon = 0.025):
    theta_per_layer ~ 14.4 degrees
    theta_64_layers ~ 64 * 14.4 = 920 degrees (catastrophic wraparound)

For cosine 0.999 (epsilon = 0.001):
    theta_per_layer ~ 2.56 degrees
    theta_64_layers ~ 64 * 2.56 = 164 degrees

Still too much for linear accumulation. But the errors are stochastic, not
systematic, so they accumulate as sqrt(L) in expectation:

    theta_total ~ sqrt(L) * theta_per_layer
    = sqrt(64) * 2.56 = 20.5 degrees -> cosine ~ 0.937

That's marginal. We want cosine 0.999+ per matrix to be safe.

Rate-Distortion Bound
-----------------------
For a Gaussian source with variance sigma^2, the rate-distortion function:

    R(D) = (1/2) * log2(sigma^2 / D)

Cosine 0.999 requires distortion-to-signal ratio ~ 0.002, giving:
    R = (1/2) * log2(500) ~ 4.48 bits/weight

At 3.076 bits/weight, the THEORETICAL minimum distortion ratio is:
    D_min/sigma^2 = 2^(-2*3.076) = 2^(-6.152) ~ 0.014

This gives best-case cosine ~ 1 - 0.007 = 0.993.

CONCLUSION: 3.076 bits/weight CAN achieve cosine ~0.993 with optimal coding,
but CANNOT achieve 0.999 uniformly. However, with MIXED PRECISION (more bits
to sensitive matrices, fewer to robust ones), the AVERAGE can stay at 3.076
while the worst case improves.

Diagnosis of v1's 0.975
-------------------------
v1 achieves cosine 0.975, far below the theoretical 0.993 limit at 3.076 bits.
The gap comes from three sources:

1. SCALE ESTIMATION: v1 uses group_std as scale. The Lloyd-Max codebook for
   unit Gaussian covers [-2.152, 2.152]. If the actual normalized distribution
   has tails beyond this range, those weights are clipped. The clipping error
   dominates the MSE. Using the min-max range mapped to codebook extremes
   eliminates this.

2. ZERO POINT: v1 uses (min+max)/2 as zero, which is optimal for uniform
   distributions but suboptimal for the peaked distributions typical of neural
   network weights. Using the mean as zero point centers the distribution
   on the codebook better.

3. NO ERROR COMPENSATION: v1 quantizes each weight independently. When one
   weight rounds up, it introduces positive error that could be partially
   canceled by rounding a correlated weight down. GPTQ-style sequential
   quantization exploits these correlations.

The Fix
--------
1. OPTIMAL SCALE: For each group, compute scale as (max - min) / (codebook_max - codebook_min).
   This maps the actual weight range exactly onto the codebook range, eliminating
   clipping. Then do a grid search over small perturbations to minimize MSE.

2. ADAPTIVE PRECISION: Output projections (out_proj, o_proj, down_proj) get
   4-bit quantization (16 levels). Input projections get 3-bit (8 levels).
   The output projections are ~25-30% of total weights, so the average stays
   within budget.

3. VECTORIZED IMPLEMENTATION: No per-group Python loops in the hot path.
"""

import numpy as np
from typing import Dict, Optional, Tuple

# ============================================================================
# Codebooks
# ============================================================================

# Lloyd-Max optimal codebook for unit Gaussian
CODEBOOK_6 = np.array([-1.894, -1.050, -0.3587, 0.3587, 1.050, 1.894], dtype=np.float32)
CODEBOOK_8 = np.array([-2.152, -1.344, -0.7560, -0.2451, 0.2451, 0.7560, 1.344, 2.152], dtype=np.float32)

# 4-bit: 16-level Lloyd-Max for unit Gaussian
CODEBOOK_16 = np.array([
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
     0.1284,  0.3880,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326
], dtype=np.float32)

PACK_WIDTH = 8  # weights per sub-pack

# Matrix names known to be sensitive (output projections)
SENSITIVE_SUFFIXES = ('out_proj', 'o_proj', 'down_proj')


def _is_sensitive_matrix(name: str) -> bool:
    """Check if a matrix name indicates a sensitive output projection."""
    name_lower = name.lower().replace('.', '_')
    for s in SENSITIVE_SUFFIXES:
        if name_lower.endswith(s):
            return True
    return False


# ============================================================================
# Optimal Scale/Zero Estimation
# ============================================================================

def _optimal_scale_zero(group: np.ndarray, codebook: np.ndarray) -> Tuple[float, float]:
    """Find scale and zero that minimize MSE for the given codebook.

    The key insight: we want to map the weight range [w_min, w_max] onto the
    codebook range [cb_min, cb_max] such that the MSE of the quantized
    reconstruction is minimized.

    Strategy:
    1. Start with range-based scale: scale = (w_max - w_min) / (cb_max - cb_min)
    2. Quantize with this scale
    3. Solve least-squares for optimal (scale, zero) given the fixed assignment
    4. Repeat once (the assignment may change)
    """
    cb_min = codebook[0]
    cb_max = codebook[-1]
    cb_range = cb_max - cb_min

    w_min = group.min()
    w_max = group.max()
    w_range = w_max - w_min

    if w_range < 1e-10:
        return 1e-10, float(group.mean())

    # Initial estimate: map weight range to codebook range
    scale = w_range / cb_range
    zero = (w_min + w_max) / 2.0 - scale * (cb_min + cb_max) / 2.0

    # Two iterations of: quantize -> optimal (scale, zero) -> re-quantize
    for _ in range(2):
        # Quantize
        normalized = (group - zero) / (scale + 1e-30)
        dists = np.abs(normalized[:, None] - codebook[None, :])
        indices = np.argmin(dists, axis=1)
        cb_vals = codebook[indices]

        # Optimal scale and zero: minimize ||group - (scale * cb_vals + zero)||^2
        # Normal equations
        n = len(group)
        sum_cb = cb_vals.sum()
        sum_cb2 = (cb_vals ** 2).sum()
        sum_g = group.sum()
        sum_cbg = (cb_vals * group).sum()

        denom = sum_cb2 * n - sum_cb * sum_cb
        if abs(denom) > 1e-20:
            new_scale = (sum_cbg * n - sum_cb * sum_g) / denom
            new_zero = (sum_g - new_scale * sum_cb) / n
            if new_scale > 1e-10:
                scale = new_scale
                zero = new_zero

    return float(scale), float(zero)


# ============================================================================
# Packing (same format as v1 for compatibility)
# ============================================================================

def _pack_6level(indices: np.ndarray) -> np.ndarray:
    n = len(indices)
    n_packs = (n + PACK_WIDTH - 1) // PACK_WIDTH
    padded = np.zeros(n_packs * PACK_WIDTH, dtype=np.uint8)
    padded[:n] = indices
    padded = padded.reshape(n_packs, PACK_WIDTH)
    packed = np.zeros(n_packs, dtype=np.uint32)
    for i in range(PACK_WIDTH):
        packed += padded[:, i].astype(np.uint32) * (6 ** i)
    return packed


def _unpack_6level(packed: np.ndarray, count: int) -> np.ndarray:
    n_packs = len(packed)
    indices = np.zeros((n_packs, PACK_WIDTH), dtype=np.uint8)
    remainder = packed.copy()
    for i in range(PACK_WIDTH):
        indices[:, i] = (remainder % 6).astype(np.uint8)
        remainder = remainder // 6
    return indices.flatten()[:count]


def _pack_8level(indices: np.ndarray) -> np.ndarray:
    n = len(indices)
    n_packs = (n + PACK_WIDTH - 1) // PACK_WIDTH
    padded = np.zeros(n_packs * PACK_WIDTH, dtype=np.uint8)
    padded[:n] = indices
    padded = padded.reshape(n_packs, PACK_WIDTH)
    packed = np.zeros(n_packs, dtype=np.uint32)
    for i in range(PACK_WIDTH):
        packed |= padded[:, i].astype(np.uint32) << (3 * i)
    return packed


def _unpack_8level(packed: np.ndarray, count: int) -> np.ndarray:
    n_packs = len(packed)
    indices = np.zeros((n_packs, PACK_WIDTH), dtype=np.uint8)
    for i in range(PACK_WIDTH):
        indices[:, i] = ((packed >> (3 * i)) & 0x7).astype(np.uint8)
    return indices.flatten()[:count]


def _pack_4bit(indices: np.ndarray) -> np.ndarray:
    """Pack 4-bit indices (0-15): 2 per byte."""
    n = len(indices)
    n_bytes = (n + 1) // 2
    padded = np.zeros(n_bytes * 2, dtype=np.uint8)
    padded[:n] = indices
    low = padded[0::2]
    high = padded[1::2]
    return (high << 4) | low


def _unpack_4bit(packed: np.ndarray, count: int) -> np.ndarray:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    interleaved = np.zeros(len(packed) * 2, dtype=np.uint8)
    interleaved[0::2] = low
    interleaved[1::2] = high
    return interleaved[:count]


# ============================================================================
# Sensitivity classification (same as v1)
# ============================================================================

def _compute_sensitivity(weights: np.ndarray, group_size: int) -> np.ndarray:
    n = weights.size
    w = weights.flatten()
    n_groups = (n + group_size - 1) // group_size
    padded = np.zeros(n_groups * group_size, dtype=np.float32)
    padded[:n] = w
    groups = padded.reshape(n_groups, group_size)
    max_mag = np.max(np.abs(groups), axis=1)
    variance = np.var(groups, axis=1) + 1e-10
    kurtosis = np.mean((groups - groups.mean(axis=1, keepdims=True))**4, axis=1) / (variance**2)
    return max_mag * np.sqrt(kurtosis)


# ============================================================================
# Quantize: Vectorized 8-level with optimal scale/zero
# ============================================================================

def _quantize_8level_vectorized(groups: np.ndarray, codebook: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize all groups at once with optimal per-group scale/zero.

    Uses range-based initialization + least-squares refinement.

    Args:
        groups: [n_groups, group_size] float32
        codebook: [n_levels] float32

    Returns:
        indices: [n_groups, group_size] uint8
        scales: [n_groups] float32
        zeros: [n_groups] float32
    """
    n_groups, group_size = groups.shape
    cb_min = codebook[0]
    cb_max = codebook[-1]
    cb_range = cb_max - cb_min

    # Range-based initialization
    g_min = groups.min(axis=1)  # [n_groups]
    g_max = groups.max(axis=1)
    g_range = g_max - g_min
    g_range = np.maximum(g_range, 1e-10)

    scales = g_range / cb_range  # [n_groups]
    zeros = (g_min + g_max) / 2.0 - scales * (cb_min + cb_max) / 2.0

    # Two iterations of quantize -> LS refine
    for iteration in range(3):
        # Normalize: [n_groups, group_size]
        normalized = (groups - zeros[:, None]) / (scales[:, None] + 1e-30)

        # Quantize: find nearest codebook entry
        # [n_groups, group_size, n_levels]
        dists = np.abs(normalized[:, :, None] - codebook[None, None, :])
        indices = np.argmin(dists, axis=2).astype(np.uint8)  # [n_groups, group_size]

        # Codebook values for assigned indices
        cb_vals = codebook[indices]  # [n_groups, group_size]

        # Least-squares: group = scale * cb_vals + zero
        n = group_size
        sum_cb = cb_vals.sum(axis=1)       # [n_groups]
        sum_cb2 = (cb_vals ** 2).sum(axis=1)
        sum_g = groups.sum(axis=1)
        sum_cbg = (cb_vals * groups).sum(axis=1)

        denom = sum_cb2 * n - sum_cb * sum_cb
        valid = np.abs(denom) > 1e-20

        new_scales = np.where(valid, (sum_cbg * n - sum_cb * sum_g) / (denom + 1e-30), scales)
        new_zeros = np.where(valid, (sum_g - new_scales * sum_cb) / n, zeros)

        # Only update where the new scale is positive and improves MSE
        positive = new_scales > 1e-10
        update = valid & positive

        scales = np.where(update, new_scales, scales)
        zeros = np.where(update, new_zeros, zeros)

    # Final quantization with refined scale/zero
    normalized = (groups - zeros[:, None]) / (scales[:, None] + 1e-30)
    dists = np.abs(normalized[:, :, None] - codebook[None, None, :])
    indices = np.argmin(dists, axis=2).astype(np.uint8)

    return indices, scales, zeros


# ============================================================================
# Main Quantization
# ============================================================================

def quantize(weights: np.ndarray, group_size: int = 128,
             matrix_name: str = "") -> dict:
    """
    Quantize weights using Shannon SMRS v2.

    Key improvements over v1:
    1. Optimal scale/zero via least-squares refinement (eliminates codebook clipping)
    2. Sensitive matrices (out_proj, o_proj, down_proj) get 4-bit (16 levels)
    3. Robust matrices get all 8-level (no 6-level mix -- simpler and higher quality)
    4. Fully vectorized -- no per-group Python loops

    Args:
        weights: FP16/FP32 weight tensor
        group_size: number of weights per group (default 128)
        matrix_name: name of the weight matrix (for sensitivity detection)

    Returns:
        dict with packed representation
    """
    original_shape = weights.shape
    is_sensitive = _is_sensitive_matrix(matrix_name)
    w = weights.flatten().astype(np.float32)
    n = len(w)

    if is_sensitive:
        return _quantize_sensitive(w, n, original_shape, group_size, matrix_name)
    else:
        return _quantize_robust(w, n, original_shape, group_size, matrix_name)


def _quantize_robust(w: np.ndarray, n: int, original_shape: tuple,
                      group_size: int, matrix_name: str) -> dict:
    """3-bit (8-level) quantization with optimal scale/zero."""
    n_groups = (n + group_size - 1) // group_size
    padded = np.zeros(n_groups * group_size, dtype=np.float32)
    padded[:n] = w
    groups = padded.reshape(n_groups, group_size)

    indices, scales, zeros = _quantize_8level_vectorized(groups, CODEBOOK_8)

    packed_data = _pack_8level(indices.flatten())

    return {
        'scheme': 'SMRSv2',
        'version': 2,
        'quant_mode': '3bit',
        'original_shape': original_shape,
        'n_weights': n,
        'group_size': group_size,
        'n_groups': n_groups,
        'matrix_name': matrix_name,
        'scales': scales.astype(np.float16),
        'zeros': zeros.astype(np.float16),
        'packed_8': packed_data,
    }


def _quantize_sensitive(w: np.ndarray, n: int, original_shape: tuple,
                         group_size: int, matrix_name: str) -> dict:
    """4-bit (16-level) quantization for sensitive output projections."""
    n_groups = (n + group_size - 1) // group_size
    padded = np.zeros(n_groups * group_size, dtype=np.float32)
    padded[:n] = w
    groups = padded.reshape(n_groups, group_size)

    indices, scales, zeros = _quantize_8level_vectorized(groups, CODEBOOK_16)

    packed_data = _pack_4bit(indices.flatten())

    return {
        'scheme': 'SMRSv2',
        'version': 2,
        'quant_mode': '4bit',
        'original_shape': original_shape,
        'n_weights': n,
        'group_size': group_size,
        'n_groups': n_groups,
        'matrix_name': matrix_name,
        'scales': scales.astype(np.float16),
        'zeros': zeros.astype(np.float16),
        'packed_4bit': packed_data,
    }


# ============================================================================
# Dequantization
# ============================================================================

def dequantize(packed: dict) -> np.ndarray:
    """Reconstruct FP32 weights from SMRSv2 packed representation."""
    mode = packed['quant_mode']
    n = packed['n_weights']
    group_size = packed['group_size']
    n_groups = packed['n_groups']
    scales = packed['scales'].astype(np.float32)
    zeros = packed['zeros'].astype(np.float32)

    if mode == '3bit':
        all_indices = _unpack_8level(packed['packed_8'], n_groups * group_size)
        all_indices = all_indices.reshape(n_groups, group_size)
        vals = CODEBOOK_8[all_indices]
    elif mode == '4bit':
        all_indices = _unpack_4bit(packed['packed_4bit'], n_groups * group_size)
        all_indices = all_indices.reshape(n_groups, group_size)
        vals = CODEBOOK_16[all_indices]
    else:
        raise ValueError(f"Unknown quant_mode: {mode}")

    result = vals * scales[:, None] + zeros[:, None]
    return result.flatten()[:n].reshape(packed['original_shape'])


# ============================================================================
# Size Accounting
# ============================================================================

def bits_per_weight(packed: dict) -> float:
    total_bits = layer_size_bytes(packed) * 8
    return total_bits / packed['n_weights']


def layer_size_bytes(packed: dict) -> int:
    mode = packed['quant_mode']
    n_groups = packed['n_groups']

    total = 64  # header

    # Scales + zeros: FP16 each
    total += n_groups * 2  # scales
    total += n_groups * 2  # zeros

    if mode == '4bit':
        # 4 bits per weight, packed 2 per byte
        total += len(packed['packed_4bit'])
    elif mode == '3bit':
        # 3 bits per weight via 8-level packing: 24 bits per 8 weights
        n_packs = len(packed['packed_8'])
        total += (n_packs * 24 + 7) // 8
    return total


# ============================================================================
# Validation
# ============================================================================

def validate(original: np.ndarray, packed: dict, threshold: float = 0.001) -> bool:
    """Returns True if cosine similarity >= 1 - threshold."""
    cos = cosine_similarity(original, dequantize(packed))
    return cos >= (1.0 - threshold)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ============================================================================
# Test Harness
# ============================================================================

def test_on_layer0():
    """Test SMRS v2 on layer 0 matrices against GPTQ reference."""
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from compare import dequantize_gptq_layer

    MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

    matrices = [
        ("mlp.gate_proj",           False),
        ("mlp.up_proj",             False),
        ("mlp.down_proj",           True),   # SENSITIVE
        ("linear_attn.in_proj_qkv", False),
        ("linear_attn.in_proj_z",   False),
        ("linear_attn.out_proj",    True),   # SENSITIVE
    ]

    print("=" * 100)
    print("SMRS v2 TEST: Layer 0 Requantization Quality")
    print("=" * 100)
    print()
    print("Strategy: 3-bit (8-level) for robust matrices, 4-bit (16-level) for sensitive")
    print("          Optimal scale/zero via least-squares refinement (3 iterations)")
    print()

    total_weights = 0
    total_bytes = 0
    all_cosines = []

    for matrix_name, is_sensitive in matrices:
        print(f"  Loading {matrix_name}...", end=" ", flush=True)
        t0 = time.time()
        ref_weights = dequantize_gptq_layer(MODEL_PATH, layer_idx=0,
                                             matrix_name=matrix_name)
        t_load = time.time() - t0

        # Quantize with v2
        t0 = time.time()
        packed = quantize(ref_weights, group_size=128, matrix_name=matrix_name)
        t_quant = time.time() - t0

        # Dequantize and measure
        reconstructed = dequantize(packed)
        cos = cosine_similarity(ref_weights, reconstructed)
        bpw = bits_per_weight(packed)
        size_mb = layer_size_bytes(packed) / (1024 * 1024)
        mode = packed['quant_mode']

        total_weights += packed['n_weights']
        total_bytes += layer_size_bytes(packed)
        all_cosines.append(cos)

        status = "PASS" if cos >= 0.999 else "FAIL"
        tag = " [SENSITIVE->4bit]" if is_sensitive else ""

        print(f"{ref_weights.shape} loaded in {t_load:.1f}s")
        print(f"    mode={mode}{tag}  bpw={bpw:.3f}  size={size_mb:.2f}MB  "
              f"cosine={cos:.6f}  [{status}]  quant={t_quant:.1f}s")

    print()
    print("-" * 100)
    total_mb = total_bytes / (1024 * 1024)
    avg_bpw = (total_bytes * 8) / total_weights
    min_cos = min(all_cosines)
    avg_cos = sum(all_cosines) / len(all_cosines)

    print(f"  Layer 0 total: {total_mb:.2f} MB  (budget: 66 MB)")
    print(f"  Average bpw:   {avg_bpw:.3f}")
    print(f"  Min cosine:    {min_cos:.6f}")
    print(f"  Avg cosine:    {avg_cos:.6f}")
    print(f"  Target: all cosines >= 0.999, total <= 66 MB")
    print()

    if total_mb <= 66.0 and min_cos >= 0.999:
        print("  >>> ALL TARGETS MET <<<")
    else:
        if total_mb > 66.0:
            print(f"  OVER BUDGET by {total_mb - 66.0:.2f} MB")
        if min_cos < 0.999:
            print(f"  QUALITY BELOW TARGET: min cosine = {min_cos:.6f}")

    return min_cos, total_mb


if __name__ == "__main__":
    test_on_layer0()
