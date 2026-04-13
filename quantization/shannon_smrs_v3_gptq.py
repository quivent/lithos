#!/usr/bin/env python3
"""
Shannon SMRS v3: Activation-Aware Calibrated Quantization (GPTQ-style)

APPROACH:
- Compute H = X^T X from calibration data (once)
- Pre-compute H_inv (once, O(K^3))
- Process rows of W in blocks: quantize, compute error, compensate remaining
- Compensation: W[k', :] -= H_inv[k, k'] / H_inv[k, k] * Error[k, :]
- All operations vectorized via numpy (no per-row Python loops)

Key performance insight: the Hessian inverse is computed ONCE and reused.
The per-row quantization is just vectorized codebook lookup + LS refinement.
The compensation is a matrix multiply: W_remaining -= factors @ Error_block.

EXPERIMENTAL RESULTS (Layer 0, Qwen 3.5-27B)
=============================================

Matrix                  Mode   v2 o_cos   v3 o_cos   delta
mlp.gate_proj           3bit   0.98470    0.99384   +0.00915
mlp.up_proj             3bit   0.98506    0.99392   +0.00886
mlp.down_proj           4bit   0.99874    0.99968   +0.00094
linear_attn.in_proj_qkv 3bit   0.99444    0.99786   +0.00342
linear_attn.in_proj_z   3bit   0.99454    0.99786   +0.00333
linear_attn.out_proj    4bit   0.99342    0.99748   +0.00405
AVERAGE                        0.99182    0.99677   +0.00496

The output-space cosine (what matters for inference quality) improves by +0.005
average. The 4-bit matrices reach 0.997-0.999, and 3-bit matrices reach 0.994.

Weight-space cosine is essentially unchanged (-0.0002), confirming that GPTQ
trades per-weight fidelity for output fidelity -- exactly as designed.
"""

import gc
import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import dequantize_gptq_layer, cosine_similarity
from shannon_smrs_v2 import (
    CODEBOOK_8, CODEBOOK_16,
    _is_sensitive_matrix,
    _pack_8level, _pack_4bit,
    quantize as v2_quantize,
    dequantize as v2_dequantize,
    bits_per_weight as v2_bpw,
    layer_size_bytes as v2_layer_size,
    cosine_similarity as v2_cosine,
)


# ============================================================================
# Vectorized row-batch quantization
# ============================================================================

def _quantize_rows_batch(
    rows: np.ndarray,
    codebook: np.ndarray,
    group_size: int = 128,
    n_refine: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a batch of rows, each split into groups.

    All rows are processed simultaneously (fully vectorized).

    Args:
        rows: [B, N] weight rows
        codebook: quantization codebook
        group_size: group size along N
        n_refine: LS iterations

    Returns:
        q_rows: [B, N] dequantized values
        errors: [B, N] quantization errors
    """
    B, N = rows.shape
    n_groups = (N + group_size - 1) // group_size
    N_pad = n_groups * group_size

    # Pad rows to multiple of group_size
    rows_pad = np.zeros((B, N_pad), dtype=np.float32)
    rows_pad[:, :N] = rows
    # Reshape to [B, n_groups, group_size]
    groups = rows_pad.reshape(B, n_groups, group_size)

    cb_min, cb_max = codebook[0], codebook[-1]
    cb_range = cb_max - cb_min
    cb_mid = (cb_min + cb_max) / 2.0
    n_levels = len(codebook)

    # Initialize scale/zero from range: [B, n_groups]
    g_min = groups.min(axis=2)
    g_max = groups.max(axis=2)
    g_range = np.maximum(g_max - g_min, 1e-10)
    scales = g_range / cb_range
    zeros = (g_min + g_max) / 2.0 - scales * cb_mid

    # Iterative LS refinement
    for _ in range(n_refine):
        # Normalize: [B, n_groups, group_size]
        normalized = (groups - zeros[:, :, None]) / (scales[:, :, None] + 1e-30)

        # Find nearest codebook entry
        # Process in chunks to avoid memory blowup
        indices = np.zeros((B, n_groups, group_size), dtype=np.uint8)
        chunk = 32  # process 32 groups at a time
        for g_start in range(0, n_groups, chunk):
            g_end = min(g_start + chunk, n_groups)
            # [B, chunk, gs, 1] vs [1, 1, 1, nL]
            norm_chunk = normalized[:, g_start:g_end, :, None]
            dists = np.abs(norm_chunk - codebook[None, None, None, :])
            indices[:, g_start:g_end, :] = np.argmin(dists, axis=3).astype(np.uint8)
            del norm_chunk, dists

        cb_vals = codebook[indices]  # [B, n_groups, group_size]

        # LS normal equations (vectorized over B and n_groups)
        n = group_size
        sum_cb = cb_vals.sum(axis=2)       # [B, n_groups]
        sum_cb2 = (cb_vals ** 2).sum(axis=2)
        sum_g = groups.sum(axis=2)
        sum_cbg = (cb_vals * groups).sum(axis=2)

        denom = sum_cb2 * n - sum_cb * sum_cb
        valid = np.abs(denom) > 1e-20

        new_scales = np.where(valid,
            (sum_cbg * n - sum_cb * sum_g) / (denom + 1e-30), scales)
        new_zeros = np.where(valid,
            (sum_g - new_scales * sum_cb) / n, zeros)

        update = valid & (new_scales > 1e-10)
        scales = np.where(update, new_scales, scales)
        zeros = np.where(update, new_zeros, zeros)

    # Final quantization
    normalized = (groups - zeros[:, :, None]) / (scales[:, :, None] + 1e-30)
    indices = np.zeros((B, n_groups, group_size), dtype=np.uint8)
    for g_start in range(0, n_groups, 32):
        g_end = min(g_start + 32, n_groups)
        norm_chunk = normalized[:, g_start:g_end, :, None]
        dists = np.abs(norm_chunk - codebook[None, None, None, :])
        indices[:, g_start:g_end, :] = np.argmin(dists, axis=3).astype(np.uint8)
        del norm_chunk, dists

    q_groups = codebook[indices] * scales[:, :, None] + zeros[:, :, None]
    q_rows = q_groups.reshape(B, N_pad)[:, :N]
    errors = rows - q_rows

    return q_rows, errors


# ============================================================================
# GPTQ with vectorized compensation
# ============================================================================

def gptq_quantize_matrix(
    W: np.ndarray,
    H_inv: np.ndarray,
    codebook: np.ndarray,
    group_size: int = 128,
    block_size: int = 128,
    n_refine: int = 3,
) -> np.ndarray:
    """GPTQ row-wise quantization with pre-computed H_inv.

    Processes rows in blocks of `block_size`. For each block:
    1. Quantize all rows in the block simultaneously
    2. Compute quantization errors
    3. Compensate remaining rows: W_remaining -= factors @ Errors

    The compensation factors come from H_inv:
        factor[k, k'] = H_inv[k, k'] / H_inv[k, k]

    Args:
        W: [K, N] weight matrix
        H_inv: [K, K] inverse Hessian (pre-computed)
        codebook: quantization codebook
        group_size: group size
        block_size: rows per block
        n_refine: LS iterations

    Returns:
        W_q: [K, N] quantized-dequantized weight matrix
    """
    K, N = W.shape

    h_inv_diag = np.diag(H_inv).copy()
    h_inv_diag = np.maximum(h_inv_diag, 1e-10)

    W_work = W.copy()
    W_q = np.zeros_like(W)

    for block_start in range(0, K, block_size):
        block_end = min(block_start + block_size, K)
        bs = block_end - block_start

        # Quantize all rows in this block at once
        q_rows, errors = _quantize_rows_batch(
            W_work[block_start:block_end, :], codebook, group_size, n_refine)

        W_q[block_start:block_end, :] = q_rows

        # Compensate remaining rows
        if block_end < K:
            remaining = K - block_end

            # factors[i, j] = H_inv[block_start+i, block_end+j] / H_inv[block_start+i, block_start+i]
            # We want: W_work[block_end:, :] -= factors^T @ errors
            # factors: [bs, remaining]
            H_inv_cross = H_inv[block_start:block_end, block_end:K]  # [bs, remaining]
            diag = h_inv_diag[block_start:block_end]  # [bs]
            factors = H_inv_cross / diag[:, None]  # [bs, remaining]

            # compensation = factors^T @ errors  =>  [remaining, N]
            compensation = factors.T @ errors  # [remaining, N]

            # Clip to prevent instability
            max_comp = 0.1 * np.std(W_work[block_end:K, :])
            if max_comp > 1e-10:
                compensation = np.clip(compensation, -max_comp, max_comp)

            W_work[block_end:K, :] -= compensation

    return W_q


# ============================================================================
# Calibration and Hessian
# ============================================================================

def generate_calibration_activations(
    K: int, n_samples: int = 128, seed: int = 42,
) -> np.ndarray:
    """Generate synthetic calibration activations with non-uniform variance."""
    rng = np.random.RandomState(seed)
    base_var = rng.power(0.5, size=K).astype(np.float32) * 3.0 + 0.1
    rng.shuffle(base_var)
    X = rng.randn(n_samples, K).astype(np.float32) * np.sqrt(base_var)[None, :]
    rank = min(32, K // 16)
    U = rng.randn(n_samples, rank).astype(np.float32) * 0.3
    V = rng.randn(rank, K).astype(np.float32)
    X += U @ V
    return X


def compute_hessian_inv(X: np.ndarray, damping: float = 0.01) -> np.ndarray:
    """Compute H_inv where H = X^T X / n + damping * I."""
    n, K = X.shape
    H = (X.T @ X) / n
    diag_mean = np.mean(np.diag(H))
    H += damping * diag_mean * np.eye(K, dtype=np.float32)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H += 0.1 * diag_mean * np.eye(K, dtype=np.float32)
        H_inv = np.linalg.inv(H)

    return H_inv


# ============================================================================
# Main entry points
# ============================================================================

def _adjust_calibration_dim(X: np.ndarray, K: int) -> np.ndarray:
    """Adjust calibration data dimension to match weight matrix K dimension."""
    cal_K = X.shape[1]
    if cal_K == K:
        return X
    if cal_K < K:
        X_adj = np.zeros((X.shape[0], K), dtype=np.float32)
        X_adj[:, :cal_K] = X
        rng = np.random.RandomState(42)
        X_adj[:, cal_K:] = rng.randn(X.shape[0], K - cal_K).astype(np.float32) * np.std(X) * 0.1
        return X_adj
    return X[:, :K].copy()


def quantize_v3(
    weights: np.ndarray,
    calibration_X: np.ndarray,
    group_size: int = 128,
    matrix_name: str = "",
    n_refine: int = 3,
    block_size: int = 128,
) -> dict:
    """GPTQ-style quantization producing Shannon v2-compatible packed output."""
    K, N = weights.shape
    is_sensitive = _is_sensitive_matrix(matrix_name)
    codebook = CODEBOOK_16 if is_sensitive else CODEBOOK_8

    calibration_X = _adjust_calibration_dim(calibration_X, K)

    # Compute H_inv (once)
    H_inv = compute_hessian_inv(calibration_X, damping=0.01)

    # GPTQ quantize
    W_q = gptq_quantize_matrix(
        weights, H_inv, codebook,
        group_size=group_size, block_size=block_size, n_refine=n_refine,
    )

    # Pack using v2 format
    packed = v2_quantize(W_q, group_size=group_size, matrix_name=matrix_name)
    packed['version'] = 3
    return packed


def quantize_v3_direct(
    weights: np.ndarray,
    calibration_X: np.ndarray,
    group_size: int = 128,
    matrix_name: str = "",
    n_refine: int = 3,
    block_size: int = 128,
) -> Tuple[dict, float, float, float, float]:
    """GPTQ quantization with diagnostic metrics.

    Returns:
        packed, weight_cos_raw, output_cos_raw, weight_cos_repacked, output_cos_repacked
    """
    K, N = weights.shape
    is_sensitive = _is_sensitive_matrix(matrix_name)
    codebook = CODEBOOK_16 if is_sensitive else CODEBOOK_8

    calibration_X = _adjust_calibration_dim(calibration_X, K)

    H_inv = compute_hessian_inv(calibration_X, damping=0.01)

    W_q = gptq_quantize_matrix(
        weights, H_inv, codebook,
        group_size=group_size, block_size=block_size, n_refine=n_refine,
    )

    # Raw metrics (GPTQ output before v2 repacking)
    Y_orig = calibration_X @ weights
    Y_q = calibration_X @ W_q
    w_cos_raw = cosine_similarity(weights, W_q)
    o_cos_raw = cosine_similarity(Y_orig, Y_q)

    # Pack and repack metrics
    packed = v2_quantize(W_q, group_size=group_size, matrix_name=matrix_name)
    packed['version'] = 3

    recon = v2_dequantize(packed)
    Y_rp = calibration_X @ recon
    w_cos_rp = cosine_similarity(weights, recon)
    o_cos_rp = cosine_similarity(Y_orig, Y_rp)

    return packed, w_cos_raw, o_cos_raw, w_cos_rp, o_cos_rp


def quantize(
    weights: np.ndarray,
    group_size: int = 128,
    matrix_name: str = "",
    calibration_X: Optional[np.ndarray] = None,
) -> dict:
    """Main entry point. Falls back to v2 without calibration."""
    if calibration_X is None:
        return v2_quantize(weights, group_size=group_size, matrix_name=matrix_name)
    return quantize_v3(weights, calibration_X, group_size=group_size,
                        matrix_name=matrix_name)


# ============================================================================
# Test
# ============================================================================

def test_on_layer0():
    MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

    matrices = [
        ("mlp.gate_proj",           False),
        ("mlp.up_proj",             False),
        ("mlp.down_proj",           True),
        ("linear_attn.in_proj_qkv", False),
        ("linear_attn.in_proj_z",   False),
        ("linear_attn.out_proj",    True),
    ]

    print("=" * 110)
    print("SMRS v3: GPTQ Row-Wise Error Compensation (Layer 0)")
    print("=" * 110)
    print()
    print("Metrics: weight_cos = cosine(W, W_q), output_cos = cosine(X@W, X@W_q)")
    print()

    # Try loading real calibration data
    cal_path = "/tmp/calibration/activations_layer_000.npz"
    has_real_cal = False
    try:
        cal_data = np.load(cal_path)
        has_real_cal = True
        print("  Using REAL calibration activations from", cal_path)
    except FileNotFoundError:
        print("  No real calibration data found, using synthetic only")
    print()

    all_results = []

    for matrix_name, is_sensitive in matrices:
        print(f"  {matrix_name}...")
        W = dequantize_gptq_layer(MODEL_PATH, layer_idx=0, matrix_name=matrix_name)
        K, N = W.shape

        # Synthetic calibration
        seed = hash(matrix_name) % 10000
        X_synth = generate_calibration_activations(K, n_samples=128, seed=seed)

        # Real calibration (if available)
        X_real = None
        if has_real_cal:
            key = matrix_name.replace(".", "_")
            if key in cal_data:
                X_real = cal_data[key].astype(np.float32)

        # v2 baseline
        t0 = time.time()
        packed_v2 = v2_quantize(W, group_size=128, matrix_name=matrix_name)
        recon_v2 = v2_dequantize(packed_v2)
        cos_v2_w = v2_cosine(W, recon_v2)
        t_v2 = time.time() - t0

        # v3 synthetic
        t0 = time.time()
        _, w_s, o_s, w_s_rp, o_s_rp = quantize_v3_direct(
            W, X_synth, group_size=128, matrix_name=matrix_name, block_size=128)
        t_v3s = time.time() - t0

        # Evaluate v2 output cosine with synthetic activations
        Y_orig_s = X_synth @ W
        Y_v2_s = X_synth @ recon_v2
        cos_v2_o_s = cosine_similarity(Y_orig_s, Y_v2_s)

        result = {
            'name': matrix_name,
            'mode': "4bit" if is_sensitive else "3bit",
            'v2_w': cos_v2_w,
            'v2_o_s': cos_v2_o_s,
            'v3s_w': w_s_rp,
            'v3s_o': o_s_rp,
        }

        print(f"    v2:          w_cos={cos_v2_w:.6f}  o_cos={cos_v2_o_s:.6f}  ({t_v2:.1f}s)")
        print(f"    v3(synth):   w_cos={w_s_rp:.6f}  o_cos={o_s_rp:.6f}  "
              f"dw={w_s_rp - cos_v2_w:+.6f}  do={o_s_rp - cos_v2_o_s:+.6f}  ({t_v3s:.1f}s)")

        if X_real is not None:
            t0 = time.time()
            _, w_r, o_r, w_r_rp, o_r_rp = quantize_v3_direct(
                W, X_real, group_size=128, matrix_name=matrix_name, block_size=128)
            t_v3r = time.time() - t0

            # Also evaluate v2 output cosine with real activations
            X_real_adj = _adjust_calibration_dim(X_real, K)
            Y_orig_r = X_real_adj @ W
            Y_v2_r = X_real_adj @ recon_v2
            cos_v2_o_r = cosine_similarity(Y_orig_r, Y_v2_r)

            result['v2_o_r'] = cos_v2_o_r
            result['v3r_w'] = w_r_rp
            result['v3r_o'] = o_r_rp

            print(f"    v3(REAL):    w_cos={w_r_rp:.6f}  o_cos={o_r_rp:.6f}  "
                  f"dw={w_r_rp - cos_v2_w:+.6f}  do={o_r_rp - cos_v2_o_r:+.6f}  ({t_v3r:.1f}s)")

        print()
        all_results.append(result)

        del W, X_synth, packed_v2, recon_v2
        if X_real is not None:
            del X_real
        gc.collect()

    # Summary
    print("=" * 110)
    print("SUMMARY")
    print("-" * 110)

    if has_real_cal:
        print(f"{'Matrix':<28s} {'Mode':>4} {'v2 w':>8} {'v2 o_r':>8} "
              f"{'v3r w':>8} {'v3r o':>8} {'dw':>8} {'do':>8}")
        print("-" * 110)
        for r in all_results:
            if 'v3r_o' in r:
                dw = r['v3r_w'] - r['v2_w']
                do = r['v3r_o'] - r.get('v2_o_r', r['v2_o_s'])
                print(f"  {r['name']:<26s} {r['mode']:>4} {r['v2_w']:>8.5f} "
                      f"{r.get('v2_o_r', r['v2_o_s']):>8.5f} "
                      f"{r['v3r_w']:>8.5f} {r['v3r_o']:>8.5f} "
                      f"{dw:>+8.5f} {do:>+8.5f}")
    else:
        print(f"{'Matrix':<28s} {'Mode':>4} {'v2 w':>8} {'v2 o':>8} "
              f"{'v3s w':>8} {'v3s o':>8} {'dw':>8} {'do':>8}")
        print("-" * 110)
        for r in all_results:
            dw = r['v3s_w'] - r['v2_w']
            do = r['v3s_o'] - r['v2_o_s']
            print(f"  {r['name']:<26s} {r['mode']:>4} {r['v2_w']:>8.5f} "
                  f"{r['v2_o_s']:>8.5f} {r['v3s_w']:>8.5f} {r['v3s_o']:>8.5f} "
                  f"{dw:>+8.5f} {do:>+8.5f}")

    print("=" * 110)

    # Averages
    if has_real_cal and all('v3r_o' in r for r in all_results):
        avg_v2 = np.mean([r.get('v2_o_r', r['v2_o_s']) for r in all_results])
        avg_v3 = np.mean([r['v3r_o'] for r in all_results])
        print(f"\n  Average output cosine: v2={avg_v2:.6f}  v3(real)={avg_v3:.6f}  "
              f"delta={avg_v3 - avg_v2:+.6f}")

    return all_results


if __name__ == "__main__":
    test_on_layer0()
