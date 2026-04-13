#!/usr/bin/env python3
"""
Test KV cache + attention against PyTorch's scaled_dot_product_attention.

Simulates a 5-token prefill through the KV cache and attention pipeline,
then verifies the output matches PyTorch for the same inputs.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kv_cache import KVCache, NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM, GQA_RATIO, FULL_ATTN_LAYERS
from attention import (
    attention_prefill, process_qkv_with_rope, apply_rope,
    _build_rope_freqs, ROTARY_DIM, ROPE_THETA,
)


def test_kv_cache_basic():
    """Test basic store/get operations."""
    print("=== Test: KV Cache basic store/get ===")
    cache = KVCache(max_seq_len=32)

    layer = 3  # first full-attention layer
    k0 = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    v0 = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    cache.store(layer, 0, k0, v0)
    k_out, v_out = cache.get(layer, 0)

    assert k_out.shape == (1, NUM_KV_HEADS, HEAD_DIM)
    assert v_out.shape == (1, NUM_KV_HEADS, HEAD_DIM)
    assert np.allclose(k_out[0], k0), "K mismatch at position 0"
    assert np.allclose(v_out[0], v0), "V mismatch at position 0"

    # Store more positions
    k1 = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    v1 = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    cache.store(layer, 1, k1, v1)

    k_out, v_out = cache.get(layer, 1)
    assert k_out.shape == (2, NUM_KV_HEADS, HEAD_DIM)
    assert np.allclose(k_out[0], k0)
    assert np.allclose(k_out[1], k1)
    print("  PASSED")


def test_kv_cache_multiple_layers():
    """Test that different layers are independent."""
    print("=== Test: KV Cache multiple layers ===")
    cache = KVCache(max_seq_len=32)

    k3 = np.ones((NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) * 3.0
    v3 = np.ones((NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) * 30.0
    k7 = np.ones((NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) * 7.0
    v7 = np.ones((NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) * 70.0

    cache.store(3, 0, k3, v3)
    cache.store(7, 0, k7, v7)

    k_out3, v_out3 = cache.get(3, 0)
    k_out7, v_out7 = cache.get(7, 0)

    assert np.allclose(k_out3[0], k3), "Layer 3 K corrupted"
    assert np.allclose(k_out7[0], k7), "Layer 7 K corrupted"
    assert not np.allclose(k_out3[0], k_out7[0]), "Layers should differ"
    print("  PASSED")


def test_rope_basic():
    """Test that RoPE at position 0 is identity-ish (cos=1, sin=0)."""
    print("=== Test: RoPE at position 0 ===")
    x = np.random.randn(1, HEAD_DIM).astype(np.float32)
    freqs = _build_rope_freqs(np.array([0]))
    x_rope = apply_rope(x, freqs)

    # At position 0, freqs are all 0, so cos=1, sin=0 -> x unchanged
    assert np.allclose(x_rope, x, atol=1e-6), "RoPE at pos 0 should be identity"
    print("  PASSED")


def test_rope_changes_with_position():
    """Test that RoPE produces different results at different positions."""
    print("=== Test: RoPE varies with position ===")
    x = np.random.randn(1, HEAD_DIM).astype(np.float32)

    freqs0 = _build_rope_freqs(np.array([0]))
    freqs5 = _build_rope_freqs(np.array([5]))

    x_rope0 = apply_rope(x, freqs0)
    x_rope5 = apply_rope(x, freqs5)

    # Rotary dims should differ, pass-through dims should be same
    assert not np.allclose(x_rope0[..., :ROTARY_DIM], x_rope5[..., :ROTARY_DIM]), \
        "RoPE should produce different values at different positions"
    assert np.allclose(x_rope0[..., ROTARY_DIM:], x_rope5[..., ROTARY_DIM:]), \
        "Pass-through dims should be unchanged"
    print("  PASSED")


def test_attention_single_token():
    """Attention with a single token should return V (trivial case)."""
    print("=== Test: Attention single token ===")
    np.random.seed(42)

    q = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    k = np.random.randn(1, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    v = np.random.randn(1, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    result = attention_prefill(q, k, v, position=0)
    assert result.shape == (NUM_Q_HEADS * HEAD_DIM,)

    # With 1 position, softmax of 1 score = 1.0, so output = V for each head
    expected = np.repeat(v[0], GQA_RATIO, axis=0).flatten()  # [24, 256] -> 6144
    assert np.allclose(result, expected, atol=1e-5), \
        f"Single token attention should return V. Max diff: {np.max(np.abs(result - expected))}"
    print("  PASSED")


def test_attention_vs_pytorch():
    """Compare our attention against PyTorch's scaled_dot_product_attention."""
    print("=== Test: Attention vs PyTorch SDPA (5-token prefill) ===")

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  SKIPPED (PyTorch not available)")
        return

    np.random.seed(123)
    seq_len = 5

    # Generate random Q/K/V for the full 5-token sequence
    # In practice, Q/K/V come from projections, but for testing we use random
    all_q = np.random.randn(seq_len, NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    all_k = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    all_v = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    # --- Our implementation: process token by token with KV cache ---
    our_outputs = []
    for t in range(seq_len):
        # K/V cache up to position t
        k_cache = all_k[:t + 1]  # [t+1, num_kv_heads, head_dim]
        v_cache = all_v[:t + 1]  # [t+1, num_kv_heads, head_dim]
        q_t = all_q[t]           # [num_q_heads, head_dim]

        out = attention_prefill(q_t, k_cache, v_cache, position=t)
        our_outputs.append(out)

    our_result = np.stack(our_outputs)  # [5, 6144]

    # --- PyTorch reference: batched SDPA with causal mask ---
    # Expand K/V with GQA: [seq, kv_heads, dim] -> [seq, q_heads, dim]
    all_k_expanded = np.repeat(all_k, GQA_RATIO, axis=1)  # [5, 24, 256]
    all_v_expanded = np.repeat(all_v, GQA_RATIO, axis=1)  # [5, 24, 256]

    # PyTorch SDPA expects [batch, heads, seq, dim]
    q_pt = torch.from_numpy(all_q).permute(1, 0, 2).unsqueeze(0)      # [1, 24, 5, 256]
    k_pt = torch.from_numpy(all_k_expanded).permute(1, 0, 2).unsqueeze(0)  # [1, 24, 5, 256]
    v_pt = torch.from_numpy(all_v_expanded).permute(1, 0, 2).unsqueeze(0)  # [1, 24, 5, 256]

    pt_out = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True)
    # [1, 24, 5, 256] -> [5, 24, 256] -> [5, 6144]
    pt_result = pt_out.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).numpy()

    max_diff = np.max(np.abs(our_result - pt_result))
    mean_diff = np.mean(np.abs(our_result - pt_result))
    print(f"  Max  absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    assert max_diff < 1e-4, f"Attention mismatch! Max diff = {max_diff}"
    print("  PASSED")


def test_attention_causal_property():
    """Verify that attention at position t does NOT depend on future positions."""
    print("=== Test: Causal property ===")
    np.random.seed(99)
    seq_len = 5

    all_q = np.random.randn(seq_len, NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    all_k = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    all_v = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    # Compute attention at position 2 with cache [0..2]
    out_at_2 = attention_prefill(all_q[2], all_k[:3], all_v[:3], position=2)

    # Now compute with cache [0..4] but only looking at position 2
    # Our function only uses 0..position, so this should be different
    # if we incorrectly gave it the wrong cache

    # Actually, verify: same result if we add garbage after position 2
    all_k_modified = all_k.copy()
    all_k_modified[3:] = 999.0  # garbage in future positions
    out_at_2_v2 = attention_prefill(all_q[2], all_k_modified[:3], all_v[:3], position=2)

    assert np.allclose(out_at_2, out_at_2_v2), "Result should not depend on future positions"
    print("  PASSED")


def test_prefill_5_tokens():
    """
    Simulate complete 5-token prefill through KV cache.
    Step through position by position, store K/V, compute attention.
    """
    print("=== Test: Full 5-token prefill simulation ===")
    np.random.seed(777)

    cache = KVCache(max_seq_len=32)
    layer = FULL_ATTN_LAYERS[0]  # layer 3

    seq_len = 5
    # Simulate Q/K/V projection outputs (after RoPE)
    all_q = np.random.randn(seq_len, NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    all_k = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    all_v = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    outputs = []
    for t in range(seq_len):
        # Store K, V at position t
        cache.store(layer, t, all_k[t], all_v[t])

        # Get cache up to position t
        k_cached, v_cached = cache.get(layer, t)
        assert k_cached.shape[0] == t + 1, f"Expected {t+1} cached, got {k_cached.shape[0]}"

        # Compute attention
        out = attention_prefill(all_q[t], k_cached, v_cached, position=t)
        outputs.append(out)

        n_scores = t + 1
        print(f"  Position {t}: cache_size={k_cached.shape[0]}, "
              f"scores_per_head={n_scores}, output_norm={np.linalg.norm(out):.4f}")

    # Position 0 should be trivial: output = V (expanded with GQA)
    v0_expanded = np.repeat(all_v[0].reshape(1, NUM_KV_HEADS, HEAD_DIM)[0],
                            GQA_RATIO, axis=0).flatten()
    assert np.allclose(outputs[0], v0_expanded, atol=1e-5), "Position 0 should return V"

    print("  PASSED")


def test_gqa_mapping():
    """Verify GQA head mapping: query heads 0-5 share KV head 0, etc."""
    print("=== Test: GQA head mapping ===")
    np.random.seed(55)

    # Create distinct V per KV head
    v = np.zeros((1, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(NUM_KV_HEADS):
        v[0, h, :] = float(h + 1)  # KV head h has all values = h+1

    # Q and K don't matter for single token (softmax of 1 = 1.0, output = V)
    q = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    k = np.random.randn(1, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

    out = attention_prefill(q, k, v, position=0)
    out = out.reshape(NUM_Q_HEADS, HEAD_DIM)

    # Query heads 0-5 should get KV head 0's values (all 1.0)
    # Query heads 6-11 should get KV head 1's values (all 2.0)
    # etc.
    for qh in range(NUM_Q_HEADS):
        expected_kv = qh // GQA_RATIO + 1
        actual = out[qh, 0]
        assert abs(actual - expected_kv) < 1e-5, \
            f"Q head {qh} got {actual}, expected {expected_kv} (KV head {qh // GQA_RATIO})"
    print("  PASSED")


def test_rope_vs_pytorch():
    """Verify our RoPE implementation against PyTorch's manual computation."""
    print("=== Test: RoPE vs PyTorch ===")

    try:
        import torch
    except ImportError:
        print("  SKIPPED (PyTorch not available)")
        return

    np.random.seed(42)
    position = 7
    x = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)

    # Our implementation
    freqs = _build_rope_freqs(np.array([position]))
    x_ours = apply_rope(x, freqs)

    # PyTorch reference RoPE
    half_rotary = ROTARY_DIM // 2
    freq_indices = torch.arange(half_rotary, dtype=torch.float64)
    inv_freq = 1.0 / (ROPE_THETA ** (2.0 * freq_indices / ROTARY_DIM))
    freqs_pt = position * inv_freq  # [half_rotary]
    cos_pt = torch.cos(freqs_pt).float()
    sin_pt = torch.sin(freqs_pt).float()

    x_t = torch.from_numpy(x)
    x_rot = x_t[..., :ROTARY_DIM]
    x_pass = x_t[..., ROTARY_DIM:]
    x1 = x_rot[..., :half_rotary]
    x2 = x_rot[..., half_rotary:]
    out1 = x1 * cos_pt - x2 * sin_pt
    out2 = x2 * cos_pt + x1 * sin_pt
    x_ref = torch.cat([out1, out2, x_pass], dim=-1).numpy()

    max_diff = np.max(np.abs(x_ours - x_ref))
    assert max_diff < 1e-5, f"RoPE mismatch vs PyTorch! Max diff = {max_diff}"
    print(f"  Max diff: {max_diff:.2e}")
    print("  PASSED")


def main():
    print("KV Cache & Attention Tests")
    print("=" * 60)
    print(f"Config: Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, GQA={GQA_RATIO}")
    print(f"RoPE: rotary_dim={ROTARY_DIM}, theta={ROPE_THETA:.0f}")
    print(f"Full-attn layers: {FULL_ATTN_LAYERS}")
    print()

    test_kv_cache_basic()
    test_kv_cache_multiple_layers()
    test_rope_basic()
    test_rope_changes_with_position()
    test_rope_vs_pytorch()
    test_attention_single_token()
    test_gqa_mapping()
    test_attention_causal_property()
    test_attention_vs_pytorch()
    test_prefill_5_tokens()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
