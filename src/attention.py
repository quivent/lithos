"""
Scaled dot-product attention with GQA and RoPE for Qwen 3.5-27B full-attention layers.

Runs entirely on CPU -- for prefill of short prompts (5-10 tokens) the
attention math is microseconds and not worth a GPU kernel.

Architecture:
  - 24 query heads, 4 KV heads (GQA ratio = 6)
  - head_dim = 256
  - partial_rotary_factor = 0.25 -> 64 dims get RoPE, 192 pass through
  - rope_theta = 10,000,000
  - Causal masking (each position attends only to itself and earlier positions)
"""

from __future__ import annotations

import math
import numpy as np
from kv_cache import NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, GQA_RATIO

# RoPE configuration
ROPE_THETA = 10_000_000.0
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64


def _build_rope_freqs(positions: np.ndarray) -> np.ndarray:
    """
    Build RoPE frequency table for the given positions.

    Args:
        positions: [seq_len] array of position indices

    Returns:
        freqs: [seq_len, rotary_dim // 2] array of angles (theta * position)
    """
    half_rotary = ROTARY_DIM // 2  # 32
    freq_indices = np.arange(half_rotary, dtype=np.float64)
    # inv_freq[i] = 1.0 / (theta ^ (2i / rotary_dim))
    inv_freq = 1.0 / (ROPE_THETA ** (2.0 * freq_indices / ROTARY_DIM))
    # freqs[pos, i] = pos * inv_freq[i]
    freqs = np.outer(positions.astype(np.float64), inv_freq)  # [seq_len, half_rotary]
    return freqs


def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Apply rotary positional embedding to the first ROTARY_DIM dimensions of x.

    Args:
        x: [..., head_dim] -- last dim is 256, first 64 get rotated
        freqs: [seq_len, rotary_dim // 2] or [1, rotary_dim // 2] for single token

    Returns:
        x with RoPE applied to first ROTARY_DIM dims
    """
    x_rot = x[..., :ROTARY_DIM].copy()
    x_pass = x[..., ROTARY_DIM:]

    half = ROTARY_DIM // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]

    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    # Standard rotary: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    return np.concatenate([out1, out2, x_pass], axis=-1)


def attention_prefill(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    position: int,
) -> np.ndarray:
    """
    Compute causal scaled dot-product attention for ONE query token against
    all cached K/V positions 0..position.

    This is the function called during prefill for each token.

    Args:
        q: [num_q_heads, head_dim] -- query for the current token, AFTER RoPE
        k_cache: [seq_len, num_kv_heads, head_dim] -- cached keys 0..position, AFTER RoPE
        v_cache: [seq_len, num_kv_heads, head_dim] -- cached values 0..position (no RoPE)
        position: current position index (0-based); seq_len = position + 1

    Returns:
        output: [num_q_heads * head_dim] = [6144] flattened attention output
    """
    seq_len = position + 1
    assert k_cache.shape[0] == seq_len
    assert v_cache.shape[0] == seq_len

    scale = 1.0 / math.sqrt(HEAD_DIM)
    output_heads = np.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=np.float32)

    for qh in range(NUM_Q_HEADS):
        kv_idx = qh // GQA_RATIO  # which KV head this query head uses

        q_vec = q[qh]  # [head_dim]
        k_mat = k_cache[:, kv_idx, :]  # [seq_len, head_dim]
        v_mat = v_cache[:, kv_idx, :]  # [seq_len, head_dim]

        # Attention scores: [seq_len]
        scores = (k_mat @ q_vec) * scale

        # Causal mask is trivially satisfied -- we only have positions 0..position
        # and the current token at `position` should attend to all of them.
        # No future positions exist in the cache, so no masking needed.

        # Softmax
        scores_max = np.max(scores)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores)

        # Weighted sum of values
        output_heads[qh] = attn_weights @ v_mat  # [head_dim]

    return output_heads.flatten()  # [6144]


def process_qkv_with_rope(
    q_raw: np.ndarray,
    k_raw: np.ndarray,
    position: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply RoPE to Q and K vectors for a single token at the given position.

    Args:
        q_raw: [num_q_heads * head_dim] = [6144] raw Q projection output
        k_raw: [num_kv_heads * head_dim] = [1024] raw K projection output
        position: sequence position for this token

    Returns:
        q_rope: [num_q_heads, head_dim] with RoPE applied
        k_rope: [num_kv_heads, head_dim] with RoPE applied
    """
    q = q_raw.reshape(NUM_Q_HEADS, HEAD_DIM)     # [24, 256]
    k = k_raw.reshape(NUM_KV_HEADS, HEAD_DIM)    # [4, 256]

    # Build RoPE frequencies for this single position
    positions = np.array([position])
    freqs = _build_rope_freqs(positions)  # [1, 32]

    # Apply to all heads
    q_rope = apply_rope(q, freqs)  # [24, 256]
    k_rope = apply_rope(k, freqs)  # [4, 256]

    return q_rope, k_rope
