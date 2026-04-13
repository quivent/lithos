#!/usr/bin/env python3
"""
Test Shannon v3 GPTQ quantization with REAL calibration activations.

Loads activations collected by calibrate_activations.py and uses them
for GPTQ-style error compensation. Compares against v2 baseline.
"""

import gc
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import dequantize_gptq_layer, cosine_similarity
from shannon_smrs_v2 import (
    quantize as v2_quantize,
    dequantize as v2_dequantize,
    bits_per_weight as v2_bpw,
    cosine_similarity as v2_cosine,
)
from shannon_smrs_v3_gptq import (
    quantize_v3_direct,
    generate_calibration_activations,
)

MODEL_PATH = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
CALIBRATION_DIR = "/tmp/calibration"


def load_calibration(layer_idx: int, matrix_name: str) -> np.ndarray:
    """Load real calibration activations from saved .npz file."""
    path = f"{CALIBRATION_DIR}/activations_layer_{layer_idx:03d}.npz"
    data = np.load(path)
    key = matrix_name.replace(".", "_")
    return data[key].astype(np.float32)


def test_real_vs_synthetic():
    matrices = [
        ("mlp.gate_proj",           False),
        ("mlp.up_proj",             False),
        ("mlp.down_proj",           True),
        ("linear_attn.in_proj_qkv", False),
        ("linear_attn.in_proj_z",   False),
        ("linear_attn.out_proj",    True),
    ]

    print("=" * 120)
    print("SMRS v3: REAL vs SYNTHETIC Calibration Activations (Layer 0)")
    print("=" * 120)
    print()

    all_results = []

    for matrix_name, is_sensitive in matrices:
        print(f"  {matrix_name}...")

        W = dequantize_gptq_layer(MODEL_PATH, layer_idx=0, matrix_name=matrix_name)
        K, N = W.shape

        # Load real calibration activations
        X_real = load_calibration(0, matrix_name)
        # Generate synthetic
        seed = hash(matrix_name) % 10000
        X_synth = generate_calibration_activations(K, n_samples=128, seed=seed)

        # Check activation statistics
        real_std = np.std(X_real, axis=0)  # [K]
        real_var_ratio = real_std.max() / (real_std.min() + 1e-10)
        synth_std = np.std(X_synth, axis=0)
        synth_var_ratio = synth_std.max() / (synth_std.min() + 1e-10)
        print(f"    Real X:  shape={X_real.shape}, var_ratio={real_var_ratio:.1f}, "
              f"mean_std={real_std.mean():.4f}")
        print(f"    Synth X: shape={X_synth.shape}, var_ratio={synth_var_ratio:.1f}, "
              f"mean_std={synth_std.mean():.4f}")

        # v2 baseline
        packed_v2 = v2_quantize(W, group_size=128, matrix_name=matrix_name)
        recon_v2 = v2_dequantize(packed_v2)
        cos_v2_w = v2_cosine(W, recon_v2)

        Y_orig_real = X_real @ W
        Y_v2_real = X_real @ recon_v2
        cos_v2_out_real = cosine_similarity(Y_orig_real, Y_v2_real)

        Y_orig_synth = X_synth @ W
        Y_v2_synth = X_synth @ recon_v2
        cos_v2_out_synth = cosine_similarity(Y_orig_synth, Y_v2_synth)

        # v3 with synthetic activations
        t0 = time.time()
        packed_v3s, w_cos_s, out_cos_s, w_cos_s_rp, out_cos_s_rp = \
            quantize_v3_direct(W, X_synth, group_size=128, matrix_name=matrix_name, block_size=32)
        t_synth = time.time() - t0

        # v3 with REAL activations
        t0 = time.time()
        packed_v3r, w_cos_r, out_cos_r, w_cos_r_rp, out_cos_r_rp = \
            quantize_v3_direct(W, X_real, group_size=128, matrix_name=matrix_name, block_size=32)
        t_real = time.time() - t0

        # Measure output cosine with REAL activations for all methods
        recon_v3s = v2_dequantize(packed_v3s)
        recon_v3r = v2_dequantize(packed_v3r)

        Y_v3s_real = X_real @ recon_v3s
        Y_v3r_real = X_real @ recon_v3r

        cos_v3s_out_real = cosine_similarity(Y_orig_real, Y_v3s_real)
        cos_v3r_out_real = cosine_similarity(Y_orig_real, Y_v3r_real)

        cos_v3s_w = v2_cosine(W, recon_v3s)
        cos_v3r_w = v2_cosine(W, recon_v3r)

        mode = "4bit" if is_sensitive else "3bit"

        print(f"    {mode}  K={K}, N={N}")
        print(f"    v2 baseline:     w_cos={cos_v2_w:.6f}  o_cos(real)={cos_v2_out_real:.6f}")
        print(f"    v3 synthetic:    w_cos={cos_v3s_w:.6f}  o_cos(real)={cos_v3s_out_real:.6f}  "
              f"dw={cos_v3s_w - cos_v2_w:+.6f}  do={cos_v3s_out_real - cos_v2_out_real:+.6f}  ({t_synth:.1f}s)")
        print(f"    v3 REAL calib:   w_cos={cos_v3r_w:.6f}  o_cos(real)={cos_v3r_out_real:.6f}  "
              f"dw={cos_v3r_w - cos_v2_w:+.6f}  do={cos_v3r_out_real - cos_v2_out_real:+.6f}  ({t_real:.1f}s)")
        print()

        all_results.append({
            'name': matrix_name,
            'mode': mode,
            'v2_w': cos_v2_w,
            'v2_o': cos_v2_out_real,
            'v3s_w': cos_v3s_w,
            'v3s_o': cos_v3s_out_real,
            'v3r_w': cos_v3r_w,
            'v3r_o': cos_v3r_out_real,
        })

        del W, X_real, X_synth, packed_v2, packed_v3s, packed_v3r
        del recon_v2, recon_v3s, recon_v3r
        gc.collect()

    # Summary
    print("=" * 120)
    print("SUMMARY: Output-space cosine measured with REAL activations")
    print("-" * 120)
    print(f"{'Matrix':<28s} {'Mode':>4} {'v2 w':>8} {'v2 o':>8} "
          f"{'v3s w':>8} {'v3s o':>8} {'v3r w':>8} {'v3r o':>8} "
          f"{'do(s)':>8} {'do(r)':>8}")
    print("-" * 120)

    for r in all_results:
        dos = r['v3s_o'] - r['v2_o']
        dor = r['v3r_o'] - r['v2_o']
        print(f"  {r['name']:<26s} {r['mode']:>4} "
              f"{r['v2_w']:>8.5f} {r['v2_o']:>8.5f} "
              f"{r['v3s_w']:>8.5f} {r['v3s_o']:>8.5f} "
              f"{r['v3r_w']:>8.5f} {r['v3r_o']:>8.5f} "
              f"{dos:>+8.5f} {dor:>+8.5f}")

    print("=" * 120)

    avg_v2_o = np.mean([r['v2_o'] for r in all_results])
    avg_v3s_o = np.mean([r['v3s_o'] for r in all_results])
    avg_v3r_o = np.mean([r['v3r_o'] for r in all_results])

    print(f"\n  Average output cosine (real activations):")
    print(f"    v2 baseline:     {avg_v2_o:.6f}")
    print(f"    v3 synthetic:    {avg_v3s_o:.6f}  delta={avg_v3s_o - avg_v2_o:+.6f}")
    print(f"    v3 REAL calib:   {avg_v3r_o:.6f}  delta={avg_v3r_o - avg_v2_o:+.6f}")

    improvement_ratio = (avg_v3r_o - avg_v2_o) / (avg_v3s_o - avg_v2_o + 1e-10)
    print(f"\n  Real vs synthetic improvement ratio: {improvement_ratio:.2f}x")


if __name__ == "__main__":
    test_real_vs_synthetic()
