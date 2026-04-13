"""
KV cache for the 16 full-attention layers in Qwen 3.5-27B.

Stores K and V projections at each sequence position so that
causal attention can be computed during prefill and generation.

Architecture:
  - 16 full-attention layers (indices 3, 7, 11, ..., 63)
  - 4 KV heads per layer, 256-dim each
  - 24 query heads (GQA ratio = 6 query heads per KV head)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# Full-attention layer indices in the 64-layer model
FULL_ATTN_LAYERS: List[int] = [i for i in range(64) if i % 4 == 3]
NUM_CACHE_LAYERS = len(FULL_ATTN_LAYERS)  # 16

# Model dimensions
NUM_KV_HEADS = 4
HEAD_DIM = 256
NUM_Q_HEADS = 24
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 6


class KVCache:
    """
    Pre-allocated KV cache for full-attention layers.

    Storage layout:
        k_cache: [num_cache_layers, max_seq_len, num_kv_heads, head_dim]  float32
        v_cache: [num_cache_layers, max_seq_len, num_kv_heads, head_dim]  float32

    The cache is indexed by a *cache layer index* (0..15), not the global
    layer index. Use layer_to_cache_idx() to convert.
    """

    def __init__(self, max_seq_len: int = 8192):
        self.max_seq_len = max_seq_len
        self.seq_len = 0  # number of positions filled so far

        # Map global layer index -> cache layer index
        self._layer_map = {layer: idx for idx, layer in enumerate(FULL_ATTN_LAYERS)}

        # Allocate storage
        self.k_cache = np.zeros(
            (NUM_CACHE_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM),
            dtype=np.float32,
        )
        self.v_cache = np.zeros(
            (NUM_CACHE_LAYERS, max_seq_len, NUM_KV_HEADS, HEAD_DIM),
            dtype=np.float32,
        )

    def layer_to_cache_idx(self, global_layer: int) -> int:
        """Convert a global layer index (e.g. 3) to a cache slot index (e.g. 0)."""
        return self._layer_map[global_layer]

    def store(self, global_layer: int, position: int,
              k: np.ndarray, v: np.ndarray) -> None:
        """
        Write K and V for a single token at the given position.

        Args:
            global_layer: the global layer index (3, 7, 11, ...)
            position: sequence position (0-indexed)
            k: shape [num_kv_heads, head_dim] or [num_kv_heads * head_dim] float32
            v: shape [num_kv_heads, head_dim] or [num_kv_heads * head_dim] float32
        """
        cache_idx = self._layer_map[global_layer]
        k_2d = k.reshape(NUM_KV_HEADS, HEAD_DIM)
        v_2d = v.reshape(NUM_KV_HEADS, HEAD_DIM)
        self.k_cache[cache_idx, position] = k_2d
        self.v_cache[cache_idx, position] = v_2d

    def get(self, global_layer: int, up_to_position: int
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve cached K and V for positions 0..up_to_position (inclusive).

        Returns:
            k: shape [up_to_position + 1, num_kv_heads, head_dim]
            v: shape [up_to_position + 1, num_kv_heads, head_dim]
        """
        cache_idx = self._layer_map[global_layer]
        seq = up_to_position + 1
        return (
            self.k_cache[cache_idx, :seq],  # [seq, num_kv_heads, head_dim]
            self.v_cache[cache_idx, :seq],  # [seq, num_kv_heads, head_dim]
        )

    def reset(self) -> None:
        """Clear the cache (zero out and reset position counter)."""
        self.k_cache[:] = 0.0
        self.v_cache[:] = 0.0
        self.seq_len = 0

    def memory_bytes(self) -> int:
        """Total bytes used by the cache arrays."""
        return self.k_cache.nbytes + self.v_cache.nbytes
