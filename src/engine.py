"""Lithos inference engine orchestrator.

Walks through model layers and launches kernels for the Qwen 3.5-27B
hybrid DeltaNet / full-attention architecture.

Python prototype — marked for rewrite into Lithos Forth once kernel
launches are wired to real cubins.

Architecture (Qwen 3.5-27B, 64 layers, 3:1 pattern):
  Layer  0: DeltaNet     Layer  1: DeltaNet     Layer  2: DeltaNet     Layer  3: FullAttention
  Layer  4: DeltaNet     Layer  5: DeltaNet     Layer  6: DeltaNet     Layer  7: FullAttention
  ...
  Layer 60: DeltaNet     Layer 61: DeltaNet     Layer 62: DeltaNet     Layer 63: FullAttention

DeltaNet layers use the RECURRENCE kernel (linear-time state update).
FullAttention layers use the ATTENTION-SCORE kernel (standard KV-cache lookup).
All layers share: PROJECTION (QKV, output, MLP gates), NORM, ACTIVATE kernels.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from weight_stage import WeightStager

# ---------------------------------------------------------------------------
# Model configuration — matches Qwen 3.5-27B with DeltaNet hybrid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 64
    hidden_dim: int = 3584          # Qwen 3.5-27B
    num_heads: int = 28             # attention heads
    num_kv_heads: int = 4           # GQA groups
    head_dim: int = 128
    intermediate_dim: int = 18944   # MLP intermediate
    vocab_size: int = 152064
    max_seq_len: int = 32768
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    dtype_bytes: int = 2            # bf16 activations
    # 3:1 pattern — layers 0,1,2 = DeltaNet; layer 3 = FullAttention; repeat
    deltanet_period: int = 4        # every 4th layer is full attention
    deltanet_full_attn_index: int = 3  # index within the period

    def layer_type(self, layer_idx: int) -> str:
        """Return 'deltanet' or 'full_attention' for a given layer index."""
        if (layer_idx % self.deltanet_period) == self.deltanet_full_attn_index:
            return "full_attention"
        return "deltanet"

    @property
    def num_deltanet_layers(self) -> int:
        return sum(
            1 for i in range(self.num_layers) if self.layer_type(i) == "deltanet"
        )

    @property
    def num_full_attention_layers(self) -> int:
        return self.num_layers - self.num_deltanet_layers


# ---------------------------------------------------------------------------
# Lightweight handle types for loaded resources
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Opaque handle to model weights in memory (mmap-ed safetensors / GGUF).

    The engine never interprets weight bytes — it passes base pointers and
    per-layer offsets to the kernels.  The loader is responsible for
    populating these fields.
    """
    config: ModelConfig
    base_ptr: int = 0               # mmap base address
    # Per-layer weight offsets from base_ptr (populated by loader)
    # Keys: "layers.{i}.attn.q_proj", "layers.{i}.mlp.gate_proj", etc.
    weight_offsets: Dict[str, int] = field(default_factory=dict)
    weight_sizes: Dict[str, int] = field(default_factory=dict)
    embed_ptr: int = 0
    lm_head_ptr: int = 0
    final_norm_ptr: int = 0


@dataclass
class LoadedKernels:
    """Handles to pre-loaded cubin modules and function pointers.

    When running without a GPU the fields stay at 0 and all launches
    are no-ops (the orchestration logic still executes).
    """
    # module handles (ctypes.c_void_p values stored as int)
    projection_func: int = 0
    attention_score_func: int = 0
    recurrence_func: int = 0        # DeltaNet recurrence kernel
    norm_func: int = 0
    activate_func: int = 0
    rotate_func: int = 0
    embed_func: int = 0             # F16 embed kernel (reads F16, outputs F32)
    sample_func: int = 0


# ---------------------------------------------------------------------------
# Activation double-buffer
# ---------------------------------------------------------------------------

class ActivationBuffers:
    """Double-buffered activation storage for layer-to-layer data flow.

    Two buffers of shape [max_tokens, hidden_dim] in bf16.  The engine
    alternates which is input and which is output so that we never read
    and write the same allocation in a single layer.
    """

    def __init__(self, config: ModelConfig, max_tokens: int = 1):
        self.config = config
        self.max_tokens = max_tokens
        buf_bytes = max_tokens * config.hidden_dim * config.dtype_bytes
        # Align to 256 bytes for GPU coalescing
        buf_bytes = (buf_bytes + 255) & ~255

        self._buf_bytes = buf_bytes
        self._mmap_a: Optional[mmap.mmap] = None
        self._mmap_b: Optional[mmap.mmap] = None
        self.ptr_a: int = 0
        self.ptr_b: int = 0
        self._current: int = 0  # 0 → A is input, 1 → B is input

        self._allocate()

    def _allocate(self) -> None:
        """Pre-allocate two activation buffers via anonymous mmap."""
        for label in ("a", "b"):
            mm = mmap.mmap(-1, self._buf_bytes, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            # memset to zero
            mm[:] = b"\x00" * self._buf_bytes
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            setattr(self, f"_mmap_{label}", mm)
            setattr(self, f"ptr_{label}", ptr)

    @property
    def input_ptr(self) -> int:
        return self.ptr_a if self._current == 0 else self.ptr_b

    @property
    def output_ptr(self) -> int:
        return self.ptr_b if self._current == 0 else self.ptr_a

    def swap(self) -> None:
        """Flip input/output roles after a layer completes."""
        self._current ^= 1

    def resize(self, max_tokens: int) -> None:
        """Reallocate if the token count grows (e.g. prefill vs decode)."""
        if max_tokens <= self.max_tokens:
            return
        self.max_tokens = max_tokens
        self._buf_bytes = (
            (max_tokens * self.config.hidden_dim * self.config.dtype_bytes + 255) & ~255
        )
        self._allocate()

    def close(self) -> None:
        for label in ("a", "b"):
            mm = getattr(self, f"_mmap_{label}", None)
            if mm is not None:
                mm.close()


# ---------------------------------------------------------------------------
# KV cache for full-attention layers
# ---------------------------------------------------------------------------

class KVCache:
    """Pre-allocated KV cache for the full-attention layers.

    Layout per full-attention layer:
        K: [max_seq_len, num_kv_heads, head_dim] in bf16
        V: [max_seq_len, num_kv_heads, head_dim] in bf16

    Only full-attention layers (every 4th) need a KV cache.
    DeltaNet layers use a fixed-size state matrix instead.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.position: int = 0  # next write position (= number of tokens seen)

        # Identify which layer indices are full attention
        self.attn_layer_indices: List[int] = [
            i for i in range(config.num_layers) if config.layer_type(i) == "full_attention"
        ]
        num_attn_layers = len(self.attn_layer_indices)

        # Bytes per layer: 2 (K+V) * max_seq_len * num_kv_heads * head_dim * dtype_bytes
        self._per_layer_bytes = (
            2 * config.max_seq_len * config.num_kv_heads * config.head_dim * config.dtype_bytes
        )
        self._total_bytes = num_attn_layers * self._per_layer_bytes

        self._mmap: Optional[mmap.mmap] = None
        self._base_ptr: int = 0
        self._allocate()

        # Map from absolute layer index to (K_ptr, V_ptr)
        self._layer_ptrs: Dict[int, Tuple[int, int]] = {}
        self._build_pointer_table()

    def _allocate(self) -> None:
        if self._total_bytes == 0:
            return
        self._mmap = mmap.mmap(
            -1, self._total_bytes, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        self._mmap[:] = b"\x00" * self._total_bytes
        self._base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))

    def _build_pointer_table(self) -> None:
        half = self._per_layer_bytes // 2  # K half, V half
        for slot, layer_idx in enumerate(self.attn_layer_indices):
            layer_base = self._base_ptr + slot * self._per_layer_bytes
            k_ptr = layer_base
            v_ptr = layer_base + half
            self._layer_ptrs[layer_idx] = (k_ptr, v_ptr)

    def kv_ptrs(self, layer_idx: int) -> Tuple[int, int]:
        """Return (K_ptr, V_ptr) for a full-attention layer."""
        return self._layer_ptrs[layer_idx]

    def write_offset_bytes(self) -> int:
        """Byte offset within a single head's sequence dimension for current position."""
        return self.position * self.config.num_kv_heads * self.config.head_dim * self.config.dtype_bytes

    def advance(self, n_tokens: int = 1) -> None:
        self.position += n_tokens
        if self.position > self.config.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: position {self.position} > max_seq_len {self.config.max_seq_len}"
            )

    def reset(self) -> None:
        self.position = 0
        if self._mmap is not None:
            self._mmap[:] = b"\x00" * self._total_bytes

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()


# ---------------------------------------------------------------------------
# DeltaNet recurrent state
# ---------------------------------------------------------------------------

class DeltaNetState:
    """Fixed-size state matrices for all DeltaNet layers.

    Each DeltaNet layer maintains a state matrix of shape:
        [num_heads, head_dim, head_dim] in bf16

    This is the recurrent hidden state that gets updated with each token
    via the delta rule:  S_{t} = S_{t-1} + beta_t * (v_t - S_{t-1}^T q_t) * k_t^T

    For speculative decoding we keep a *snapshot* of the state so that
    we can roll back if verification rejects draft tokens.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        self.deltanet_layer_indices: List[int] = [
            i for i in range(config.num_layers) if config.layer_type(i) == "deltanet"
        ]
        num_dn_layers = len(self.deltanet_layer_indices)  # 48 for Qwen 3.5-27B

        # Per-layer state: num_heads * head_dim * head_dim * dtype_bytes
        self._per_layer_bytes = (
            config.num_heads * config.head_dim * config.head_dim * config.dtype_bytes
        )
        self._total_bytes = num_dn_layers * self._per_layer_bytes

        # Primary state
        self._mmap: Optional[mmap.mmap] = None
        self._base_ptr: int = 0
        # Snapshot for speculative decoding rollback
        self._snap_mmap: Optional[mmap.mmap] = None
        self._snap_ptr: int = 0

        self._allocate()

        # Map absolute layer index -> state pointer
        self._layer_ptrs: Dict[int, int] = {}
        self._snap_layer_ptrs: Dict[int, int] = {}
        self._build_pointer_table()

    def _allocate(self) -> None:
        if self._total_bytes == 0:
            return
        for attr in ("_mmap", "_snap_mmap"):
            mm = mmap.mmap(
                -1, self._total_bytes, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            )
            mm[:] = b"\x00" * self._total_bytes
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            setattr(self, attr, mm)
            ptr_attr = "_base_ptr" if attr == "_mmap" else "_snap_ptr"
            setattr(self, ptr_attr, ptr)

    def _build_pointer_table(self) -> None:
        for slot, layer_idx in enumerate(self.deltanet_layer_indices):
            self._layer_ptrs[layer_idx] = self._base_ptr + slot * self._per_layer_bytes
            self._snap_layer_ptrs[layer_idx] = self._snap_ptr + slot * self._per_layer_bytes

    def state_ptr(self, layer_idx: int) -> int:
        """Return the state matrix pointer for a DeltaNet layer."""
        return self._layer_ptrs[layer_idx]

    def snapshot(self) -> None:
        """Copy current state to the snapshot buffer (for speculative decoding)."""
        if self._mmap is not None and self._snap_mmap is not None:
            self._snap_mmap[:] = self._mmap[:]

    def rollback(self) -> None:
        """Restore state from the last snapshot (rejected speculative tokens)."""
        if self._mmap is not None and self._snap_mmap is not None:
            self._mmap[:] = self._snap_mmap[:]

    def reset(self) -> None:
        if self._mmap is not None:
            self._mmap[:] = b"\x00" * self._total_bytes
        if self._snap_mmap is not None:
            self._snap_mmap[:] = b"\x00" * self._total_bytes

    def close(self) -> None:
        for attr in ("_mmap", "_snap_mmap"):
            mm = getattr(self, attr, None)
            if mm is not None:
                mm.close()


# ---------------------------------------------------------------------------
# Residual accumulator
# ---------------------------------------------------------------------------

class ResidualAccumulator:
    """Tracks the residual stream pointer.

    In the transformer, each sublayer (attention / MLP) adds its output
    to the residual.  The engine passes the residual pointer so that the
    NORM and PROJECTION kernels can read/write it in-place.
    """

    def __init__(self, config: ModelConfig, max_tokens: int = 1):
        self.config = config
        buf_bytes = max_tokens * config.hidden_dim * config.dtype_bytes
        buf_bytes = (buf_bytes + 255) & ~255
        self._buf_bytes = buf_bytes
        self._mmap = mmap.mmap(-1, buf_bytes, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self._mmap[:] = b"\x00" * buf_bytes
        self.ptr: int = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()


# ---------------------------------------------------------------------------
# GPU launch -- real cuLaunchKernel via CUDADriver + fallback logging
# ---------------------------------------------------------------------------

_LAUNCH_LOG: List[str] = []  # for testing/debugging

# Global CUDADriver instance -- set by LithosEngine when in real-execution mode
_CUDA_DRIVER: Optional[Any] = None

# Set of kernel names (prefixes) that are safe to actually launch on GPU.
# Kernels not in this set will be logged but not launched.
# This allows incremental bring-up: start with embed+norm, add more as they work.
_LIVE_KERNELS: set[str] = set()


def _gpu_launch(
    func_handle: int,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    shared_bytes: int,
    params: Sequence[int],
    *,
    name: str = "",
    param_types: Optional[Sequence[str]] = None,
) -> None:
    """Launch a CUDA kernel via the driver API.

    When a CUDADriver is registered and func_handle is non-zero, this calls
    cuLaunchKernel with the given parameters.  Otherwise it falls back to
    logging (for orchestration-only testing without hardware).

    *params* is a list of raw integer values.  *param_types* optionally
    specifies the ctypes type for each parameter:
        "u32" -> ctypes.c_uint32
        "u64" -> ctypes.c_uint64
        "f32" -> ctypes.c_float
        "i32" -> ctypes.c_int32
    If param_types is None, values >= (1 << 32) are treated as u64 (pointers),
    otherwise as u32.
    """
    entry = (
        f"LAUNCH {name}: grid={grid} block={block} "
        f"smem={shared_bytes} params=[{', '.join(f'0x{p:x}' for p in params)}]"
    )
    _LAUNCH_LOG.append(entry)

    # Check if this kernel is in the live set (for incremental bring-up)
    should_launch = _CUDA_DRIVER is not None and func_handle != 0
    if should_launch and _LIVE_KERNELS:
        # Only launch if the kernel name matches a live prefix
        should_launch = any(name.startswith(prefix) for prefix in _LIVE_KERNELS)

    if should_launch:
        typed_args: list[Any] = []
        for i, val in enumerate(params):
            if param_types is not None and i < len(param_types):
                ptype = param_types[i]
                if ptype == "u64":
                    typed_args.append(ctypes.c_uint64(val))
                elif ptype == "u32":
                    typed_args.append(ctypes.c_uint32(val))
                elif ptype == "f32":
                    # val is actually a float passed as int via struct
                    typed_args.append(ctypes.c_float(struct.unpack('f', struct.pack('I', val))[0]))
                elif ptype == "i32":
                    typed_args.append(ctypes.c_int32(val))
                else:
                    typed_args.append(ctypes.c_uint64(val))
            else:
                # Auto-detect: pointers are u64, small values are u32
                if val >= (1 << 32) or val < 0:
                    typed_args.append(ctypes.c_uint64(val))
                else:
                    typed_args.append(ctypes.c_uint32(val))

        func = ctypes.c_void_p(func_handle)
        _CUDA_DRIVER.launch(
            func,
            grid=grid,
            block=block,
            args=typed_args,
            shared_mem=shared_bytes,
        )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class LithosEngine:
    """Inference engine orchestrator for Qwen 3.5-27B hybrid DeltaNet.

    Usage:
        model = LoadedModel(config=ModelConfig(), ...)  # from loader
        kernels = LoadedKernels(...)                     # from load_all.py
        engine = LithosEngine(model, kernels)
        engine.prefill(token_ids)
        while not done:
            token = engine.decode_step()
    """

    def __init__(self, model: LoadedModel, kernels: LoadedKernels,
                 cuda_driver: Optional[Any] = None,
                 weight_stager: Optional[WeightStager] = None):
        global _CUDA_DRIVER
        self.model = model
        self.kernels = kernels
        self.config = model.config
        self._gpu = cuda_driver
        self._weight_stager = weight_stager

        # Register the CUDA driver globally so _gpu_launch can use it
        if cuda_driver is not None:
            _CUDA_DRIVER = cuda_driver

        # State
        self.kv_cache = KVCache(self.config)
        self.dn_state = DeltaNetState(self.config)
        self.act_bufs = ActivationBuffers(self.config, max_tokens=1)
        self.residual = ResidualAccumulator(self.config, max_tokens=1)

        # Sequence position (total tokens processed so far)
        self.seq_pos: int = 0

        # Last logits pointer (set after final lm_head projection)
        self._logits_ptr: int = 0
        self._logits_count: int = 0  # number of f32 logits at _logits_ptr
        self._logits_are_real: bool = False  # True only when lm_head actually ran

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(self, token_ids: Sequence[int]) -> None:
        """Process the entire prompt in one pass.

        During prefill every layer sees all prompt tokens at once.
        Full-attention layers do causal self-attention over the prompt.
        DeltaNet layers scan sequentially to build their state.
        """
        n_tokens = len(token_ids)
        if n_tokens == 0:
            return

        # Resize buffers for the prompt length
        self.act_bufs.resize(n_tokens)

        # --- Embedding lookup ---
        self._launch_embed(token_ids)

        # Copy embedding output into the residual stream
        self._copy_to_residual(n_tokens)

        # --- Layer loop ---
        for layer_idx in range(self.config.num_layers):
            self._run_layer(layer_idx, n_tokens, is_prefill=True)

        # --- Final norm + LM head ---
        self._launch_final_norm(n_tokens)
        self._launch_lm_head(n_tokens)

        # Advance sequence position
        self.kv_cache.advance(n_tokens)
        self.seq_pos += n_tokens
        # Remember last token for decode_step's embed
        self._last_token_id = token_ids[-1] if token_ids else 0

    def decode_step(self) -> int:
        """Generate a single token.

        Returns the sampled token id.
        """
        n_tokens = 1

        # Snapshot DeltaNet state (for speculative decode rollback)
        # In normal decode we still snapshot cheaply — the copy is small
        # relative to the forward pass.
        self.dn_state.snapshot()

        # The previous token's id is used for embedding.  In a real
        # implementation the caller passes it; here we use a placeholder.
        # The important thing is the orchestration logic.

        # --- Embedding ---
        self._launch_embed_single()

        self._copy_to_residual(n_tokens)

        # --- Layer loop ---
        for layer_idx in range(self.config.num_layers):
            self._run_layer(layer_idx, n_tokens, is_prefill=False)

        # --- Final norm + LM head ---
        self._launch_final_norm(n_tokens)
        self._launch_lm_head(n_tokens)

        # --- Sample ---
        token_id = self._launch_sample()

        # Advance
        self.kv_cache.advance(1)
        self.seq_pos += 1
        self._last_token_id = token_id

        return token_id

    def reset(self) -> None:
        """Clear all state for a new sequence."""
        self.kv_cache.reset()
        self.dn_state.reset()
        self.seq_pos = 0

    def close(self) -> None:
        self.kv_cache.close()
        self.dn_state.close()
        self.act_bufs.close()
        self.residual.close()

    # ------------------------------------------------------------------
    # Per-layer orchestration
    # ------------------------------------------------------------------

    def _run_layer(self, layer_idx: int, n_tokens: int, *, is_prefill: bool) -> None:
        """Execute one transformer layer (attention/deltanet sublayer + MLP)."""
        layer_type = self.config.layer_type(layer_idx)

        # ---- Attention / DeltaNet sublayer ----

        # Pre-attention norm (RMSNorm)
        self._launch_norm(layer_idx, "attn_norm", n_tokens)

        # QKV projection
        self._launch_qkv_proj(layer_idx, n_tokens)

        # Rotary position embedding (RoPE) — applied to Q and K
        self._launch_rope(layer_idx, n_tokens)

        if layer_type == "full_attention":
            # Write K, V into KV cache at current position
            self._launch_kv_cache_write(layer_idx, n_tokens)
            # Attention scores: Q @ K^T, softmax, @ V
            self._launch_attention_score(layer_idx, n_tokens)
        else:
            # DeltaNet recurrence: update state matrix, produce output
            self._launch_deltanet_recurrence(layer_idx, n_tokens)

        # Output projection (attention output -> hidden_dim)
        self._launch_output_proj(layer_idx, n_tokens)

        # Residual add (output += residual)
        self._launch_residual_add(layer_idx, "attn", n_tokens)

        # Swap activation buffers
        self.act_bufs.swap()

        # ---- MLP sublayer ----

        # Pre-MLP norm (RMSNorm)
        self._launch_norm(layer_idx, "mlp_norm", n_tokens)

        # Gate projection + Up projection (can be fused into one GEMM)
        self._launch_gate_up_proj(layer_idx, n_tokens)

        # SiLU activation + element-wise multiply (gate * up after SiLU)
        self._launch_activate(layer_idx, n_tokens)

        # Down projection
        self._launch_down_proj(layer_idx, n_tokens)

        # Residual add
        self._launch_residual_add(layer_idx, "mlp", n_tokens)

        # Swap activation buffers
        self.act_bufs.swap()

    # ------------------------------------------------------------------
    # Kernel launch wrappers
    # ------------------------------------------------------------------

    def _weight_ptr(self, name: str) -> int:
        """Resolve a weight name to an absolute pointer."""
        offset = self.model.weight_offsets.get(name, 0)
        return self.model.base_ptr + offset

    def _launch_embed(self, token_ids: Sequence[int]) -> None:
        """Embedding lookup for a batch of tokens.

        The specialized embed kernel takes (token_id: u32, table_ptr: u64, output_ptr: u64)
        with hidden_dim baked in.  We launch once per token.
        """
        hidden = self.config.hidden_dim
        BLOCK = 256
        GRID = max(1, math.ceil(hidden / (BLOCK * 4)))  # vectorized: 4 elems/thread

        for i, tid in enumerate(token_ids):
            out_offset = i * hidden * 4  # F32 output, 4 bytes per element
            _gpu_launch(
                self.kernels.embed_func,
                grid=(GRID, 1, 1),
                block=(BLOCK, 1, 1),
                shared_bytes=0,
                params=[
                    tid,
                    self.model.embed_ptr,
                    self.act_bufs.output_ptr + out_offset,
                ],
                param_types=["u32", "u64", "u64"],
                name=f"embed_tok{tid}",
            )
        self.act_bufs.swap()

    def _launch_embed_single(self) -> None:
        """Embedding lookup for a single token (decode step).

        Uses the last generated token id stored in self._last_token_id.
        """
        hidden = self.config.hidden_dim
        BLOCK = 256
        GRID = max(1, math.ceil(hidden / (BLOCK * 4)))

        token_id = getattr(self, '_last_token_id', 0)
        _gpu_launch(
            self.kernels.embed_func,
            grid=(GRID, 1, 1),
            block=(BLOCK, 1, 1),
            shared_bytes=0,
            params=[
                token_id,
                self.model.embed_ptr,
                self.act_bufs.output_ptr,
            ],
            param_types=["u32", "u64", "u64"],
            name="embed_single",
        )
        self.act_bufs.swap()

    def _copy_to_residual(self, n_tokens: int) -> None:
        """Copy current activation input into the residual accumulator.

        In a real implementation this is a device memcpy or the embed
        kernel writes directly to the residual buffer.
        """
        # Mock: just note the pointer relationship
        pass

    def _norm_weight_ptr(self, layer_idx: int, sublayer: str) -> int:
        """Resolve a norm weight pointer, using staged F32 weights when available.

        Falls back to the raw model pointer if no stager is present.
        """
        if self._weight_stager is not None:
            if sublayer == "attn_norm":
                return self._weight_stager.input_norm_ptr(layer_idx)
            elif sublayer == "mlp_norm":
                return self._weight_stager.post_attn_norm_ptr(layer_idx)
        # Fallback: use the raw weight pointer (caller must ensure dtype is F32)
        weight_name = f"layers.{layer_idx}.{sublayer}.weight"
        return self._weight_ptr(weight_name)

    def _launch_norm(self, layer_idx: int, sublayer: str, n_tokens: int) -> None:
        """RMSNorm before attention or MLP sublayer.

        Kernel signature: norm(input_ptr, residual_ptr, weight_ptr, output_ptr, epsilon)
        All u64 pointers + f32 epsilon.  hidden_dim=5120 baked in.

        If a WeightStager is attached, the norm weight pointer comes from the
        staged F32 buffer instead of the raw BF16 model weights.
        """
        eps_bits = struct.unpack('I', struct.pack('f', self.config.norm_eps))[0]
        _gpu_launch(
            self.kernels.norm_func,
            grid=(n_tokens, 1, 1),
            block=(256, 1, 1),
            shared_bytes=128,
            params=[
                self.act_bufs.input_ptr,     # input
                self.residual.ptr,           # residual
                self._norm_weight_ptr(layer_idx, sublayer),  # weight (F32)
                self.act_bufs.output_ptr,    # output
                eps_bits,                    # epsilon as f32 bits
            ],
            param_types=["u64", "u64", "u64", "u64", "f32"],
            name=f"norm_L{layer_idx}_{sublayer}",
        )
        self.act_bufs.swap()

    def _launch_qkv_proj(self, layer_idx: int, n_tokens: int) -> None:
        """QKV projection — three GEMV (decode) or GEMM (prefill) operations.

        For decode (n_tokens=1) this is three matrix-vector multiplies.
        For prefill this is three matrix-matrix multiplies.

        Q: [n_tokens, hidden_dim] @ [hidden_dim, num_heads * head_dim]
        K: [n_tokens, hidden_dim] @ [hidden_dim, num_kv_heads * head_dim]
        V: [n_tokens, hidden_dim] @ [hidden_dim, num_kv_heads * head_dim]
        """
        q_out_dim = self.config.num_heads * self.config.head_dim
        kv_out_dim = self.config.num_kv_heads * self.config.head_dim

        for proj, out_dim in [("q_proj", q_out_dim), ("k_proj", kv_out_dim), ("v_proj", kv_out_dim)]:
            weight_name = f"layers.{layer_idx}.attn.{proj}.weight"
            _gpu_launch(
                self.kernels.projection_func,
                grid=(math.ceil(out_dim / 128), n_tokens, 1),
                block=(128, 1, 1),
                shared_bytes=0,
                params=[
                    self.act_bufs.input_ptr,
                    self._weight_ptr(weight_name),
                    self.act_bufs.output_ptr,
                    n_tokens,
                    self.config.hidden_dim,  # M (input dim)
                    out_dim,                 # N (output dim)
                ],
                name=f"proj_L{layer_idx}_{proj}",
            )
        # After QKV, the output buffer holds Q, K, V packed
        self.act_bufs.swap()

    def _launch_rope(self, layer_idx: int, n_tokens: int) -> None:
        """Apply rotary position embeddings to Q and K."""
        _gpu_launch(
            self.kernels.rotate_func,
            grid=(math.ceil(self.config.num_heads * self.config.head_dim / 256), n_tokens, 1),
            block=(256, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.input_ptr,  # Q, K packed
                self.seq_pos,             # base position
                n_tokens,
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
            ],
            name=f"rope_L{layer_idx}",
        )

    def _launch_kv_cache_write(self, layer_idx: int, n_tokens: int) -> None:
        """Write K and V into the KV cache for a full-attention layer."""
        k_ptr, v_ptr = self.kv_cache.kv_ptrs(layer_idx)
        write_offset = self.kv_cache.write_offset_bytes()

        # K and V are in the activation buffer after QKV projection
        # The write kernel copies them into the cache at the right position
        _gpu_launch(
            self.kernels.projection_func,  # reuse projection kernel for memcpy-like write
            grid=(math.ceil(self.config.num_kv_heads * self.config.head_dim / 256), n_tokens, 1),
            block=(256, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.input_ptr,
                k_ptr + write_offset,
                v_ptr + write_offset,
                n_tokens,
                self.config.num_kv_heads,
                self.config.head_dim,
            ],
            name=f"kv_write_L{layer_idx}",
        )

    def _launch_attention_score(self, layer_idx: int, n_tokens: int) -> None:
        """Full causal attention: Q @ K^T / sqrt(d), softmax, @ V."""
        k_ptr, v_ptr = self.kv_cache.kv_ptrs(layer_idx)
        seq_len = self.seq_pos + n_tokens  # total sequence length including current

        _gpu_launch(
            self.kernels.attention_score_func,
            grid=(self.config.num_heads, n_tokens, 1),
            block=(128, 1, 1),
            shared_bytes=self.config.head_dim * 4,  # for softmax reduction
            params=[
                self.act_bufs.input_ptr,     # Q
                k_ptr,                       # K cache
                v_ptr,                       # V cache
                self.act_bufs.output_ptr,    # attention output
                seq_len,                     # total sequence length
                n_tokens,                    # current tokens
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
            ],
            name=f"attention_score_L{layer_idx}",
        )
        self.act_bufs.swap()

    def _launch_deltanet_recurrence(self, layer_idx: int, n_tokens: int) -> None:
        """DeltaNet linear-time recurrence.

        Updates the state matrix S via:
            S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T @ q_t) @ k_t^T

        The kernel reads the current state, Q, K, V, beta, and writes
        the updated state + the layer output.
        """
        state_ptr = self.dn_state.state_ptr(layer_idx)

        _gpu_launch(
            self.kernels.recurrence_func,
            grid=(self.config.num_heads, 1, 1),
            block=(self.config.head_dim, 1, 1),
            shared_bytes=self.config.head_dim * self.config.head_dim * 2,  # state tile in smem
            params=[
                self.act_bufs.input_ptr,     # Q, K, V, beta packed
                state_ptr,                   # state matrix (read + write)
                self.act_bufs.output_ptr,    # layer output
                n_tokens,
                self.config.num_heads,
                self.config.head_dim,
            ],
            name=f"deltanet_recurrence_L{layer_idx}",
        )
        self.act_bufs.swap()

    def _launch_output_proj(self, layer_idx: int, n_tokens: int) -> None:
        """Output projection: [n_tokens, num_heads * head_dim] -> [n_tokens, hidden_dim]."""
        weight_name = f"layers.{layer_idx}.attn.o_proj.weight"
        out_dim = self.config.hidden_dim

        _gpu_launch(
            self.kernels.projection_func,
            grid=(math.ceil(out_dim / 128), n_tokens, 1),
            block=(128, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.input_ptr,
                self._weight_ptr(weight_name),
                self.act_bufs.output_ptr,
                n_tokens,
                self.config.num_heads * self.config.head_dim,  # input dim
                out_dim,
            ],
            name=f"proj_L{layer_idx}_o_proj",
        )
        self.act_bufs.swap()

    def _launch_residual_add(self, layer_idx: int, sublayer: str, n_tokens: int) -> None:
        """Add sublayer output to the residual stream.

        residual += sublayer_output

        In a fused kernel this is part of the norm — fused-add-rmsnorm.
        Here it is a separate step for clarity.
        """
        # Mock: in the real kernel, residual.ptr and act_bufs.input_ptr
        # are passed and the kernel does element-wise add in-place.
        _gpu_launch(
            self.kernels.norm_func,  # residual add reuses norm kernel in fused mode
            grid=(math.ceil(n_tokens * self.config.hidden_dim / 256), 1, 1),
            block=(256, 1, 1),
            shared_bytes=0,
            params=[
                self.residual.ptr,
                self.act_bufs.input_ptr,
                n_tokens,
                self.config.hidden_dim,
            ],
            name=f"residual_add_L{layer_idx}_{sublayer}",
        )

    def _launch_gate_up_proj(self, layer_idx: int, n_tokens: int) -> None:
        """MLP gate and up projections.

        gate: [n_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
        up:   [n_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
        """
        for proj in ("gate_proj", "up_proj"):
            weight_name = f"layers.{layer_idx}.mlp.{proj}.weight"
            _gpu_launch(
                self.kernels.projection_func,
                grid=(math.ceil(self.config.intermediate_dim / 128), n_tokens, 1),
                block=(128, 1, 1),
                shared_bytes=0,
                params=[
                    self.act_bufs.input_ptr,
                    self._weight_ptr(weight_name),
                    self.act_bufs.output_ptr,
                    n_tokens,
                    self.config.hidden_dim,
                    self.config.intermediate_dim,
                ],
                name=f"proj_L{layer_idx}_{proj}",
            )

    def _launch_activate(self, layer_idx: int, n_tokens: int) -> None:
        """SiLU activation on gate output, then element-wise multiply with up.

        out = SiLU(gate) * up
        """
        _gpu_launch(
            self.kernels.activate_func,
            grid=(math.ceil(n_tokens * self.config.intermediate_dim / 256), 1, 1),
            block=(256, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.output_ptr,  # gate values (in-place SiLU)
                self.act_bufs.output_ptr,  # up values (offset by intermediate_dim)
                n_tokens,
                self.config.intermediate_dim,
            ],
            name=f"activate_L{layer_idx}",
        )

    def _launch_down_proj(self, layer_idx: int, n_tokens: int) -> None:
        """MLP down projection: [n_tokens, intermediate_dim] -> [n_tokens, hidden_dim]."""
        weight_name = f"layers.{layer_idx}.mlp.down_proj.weight"
        _gpu_launch(
            self.kernels.projection_func,
            grid=(math.ceil(self.config.hidden_dim / 128), n_tokens, 1),
            block=(128, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.input_ptr,
                self._weight_ptr(weight_name),
                self.act_bufs.output_ptr,
                n_tokens,
                self.config.intermediate_dim,
                self.config.hidden_dim,
            ],
            name=f"proj_L{layer_idx}_down_proj",
        )
        self.act_bufs.swap()

    def _launch_final_norm(self, n_tokens: int) -> None:
        """RMSNorm after all layers, before the LM head.

        Same kernel signature as per-layer norm.  Uses the staged F32
        final norm weight when a WeightStager is attached.
        """
        if self._weight_stager is not None:
            norm_ptr = self._weight_stager.final_norm_ptr()
        else:
            norm_ptr = self.model.final_norm_ptr

        eps_bits = struct.unpack('I', struct.pack('f', self.config.norm_eps))[0]
        _gpu_launch(
            self.kernels.norm_func,
            grid=(n_tokens, 1, 1),
            block=(256, 1, 1),
            shared_bytes=128,
            params=[
                self.act_bufs.input_ptr,
                self.residual.ptr,
                norm_ptr,
                self.act_bufs.output_ptr,
                eps_bits,
            ],
            param_types=["u64", "u64", "u64", "u64", "f32"],
            name="final_norm",
        )
        self.act_bufs.swap()

    def _launch_lm_head(self, n_tokens: int) -> None:
        """Project hidden states to vocab logits.

        [n_tokens, hidden_dim] @ [hidden_dim, vocab_size] -> [n_tokens, vocab_size]

        Only the last token's logits matter for next-token prediction,
        but during prefill we compute all for potential parallel verification.
        """
        _gpu_launch(
            self.kernels.projection_func,
            grid=(math.ceil(self.config.vocab_size / 128), n_tokens, 1),
            block=(128, 1, 1),
            shared_bytes=0,
            params=[
                self.act_bufs.input_ptr,
                self.model.lm_head_ptr,
                self.act_bufs.output_ptr,
                n_tokens,
                self.config.hidden_dim,
                self.config.vocab_size,
            ],
            name="lm_head",
        )
        self._logits_ptr = self.act_bufs.output_ptr
        self._logits_count = n_tokens * self.config.vocab_size
        # Check if lm_head actually launched (vs log-only)
        self._logits_are_real = (
            _CUDA_DRIVER is not None
            and self.kernels.projection_func != 0
            and (not _LIVE_KERNELS or any("lm_head".startswith(p) for p in _LIVE_KERNELS))
        )
        self.act_bufs.swap()

    def _launch_sample(self) -> int:
        """Sample next token from logits via CPU argmax.

        If a CUDA driver is available and logits are on device, copies them
        back to host first.  On GH200 unified memory, the logits pointer
        may already be host-accessible.

        For now: greedy argmax (temperature=0).  Top-k/top-p can be added later.
        """
        import numpy as np

        vocab = self.config.vocab_size
        logits_ptr = self._logits_ptr

        if logits_ptr == 0 or not self._logits_are_real:
            _LAUNCH_LOG.append("LAUNCH sample: logits not computed (kernels log-only), returning 0")
            return 0

        # Synchronize GPU before reading logits
        if self._gpu is not None:
            self._gpu.synchronize()

        # Read the LAST token's logits (for next-token prediction)
        # logits layout: [n_tokens, vocab_size] in F32
        # We want the last row.
        last_row_offset = (self._logits_count - vocab) * 4  # bytes

        # Try to read logits from the pointer.
        # On GH200 unified memory, mmap'd host pointers are GPU-accessible
        # and GPU device pointers are host-accessible.
        try:
            logits_host = np.zeros(vocab, dtype=np.float32)
            if self._gpu is not None:
                # Copy from device to host
                from cuda_driver import CUdeviceptr
                src = CUdeviceptr(logits_ptr + last_row_offset)
                self._gpu.memcpy_dtoh(
                    logits_host.ctypes.data_as(ctypes.c_void_p),
                    src,
                    vocab * 4,
                )
            else:
                # Direct host memory read (logits in mmap'd buffer)
                src_array = (ctypes.c_float * vocab).from_address(
                    logits_ptr + last_row_offset
                )
                for i in range(vocab):
                    logits_host[i] = src_array[i]

            # Greedy argmax
            token_id = int(np.argmax(logits_host))
            _LAUNCH_LOG.append(
                f"LAUNCH sample: CPU argmax over {vocab} logits -> token_id={token_id}"
            )
            return token_id

        except Exception as e:
            _LAUNCH_LOG.append(f"LAUNCH sample: FAILED ({e}), returning 0")
            return 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def get_launch_log() -> List[str]:
        """Return the mock launch log (for testing)."""
        return list(_LAUNCH_LOG)

    @staticmethod
    def clear_launch_log() -> None:
        _LAUNCH_LOG.clear()

    @staticmethod
    def set_live_kernels(prefixes: set[str]) -> None:
        """Control which kernels actually launch on GPU.

        Pass a set of name prefixes (e.g. {"embed", "norm"}).
        Only kernels whose name starts with one of these will be
        dispatched to cuLaunchKernel; others are logged only.
        Pass an empty set to launch ALL kernels (no filter).
        """
        global _LIVE_KERNELS
        _LIVE_KERNELS = set(prefixes)

    def print_layer_map(self) -> None:
        """Print the layer type assignment for all 64 layers."""
        for i in range(self.config.num_layers):
            lt = self.config.layer_type(i)
            marker = "FA" if lt == "full_attention" else "DN"
            end = "\n" if (i + 1) % 8 == 0 else "  "
            print(f"L{i:02d}:{marker}", end=end)

    def memory_report(self) -> Dict[str, int]:
        """Report memory usage in bytes."""
        kv_bytes = self.kv_cache._total_bytes
        dn_bytes = self.dn_state._total_bytes * 2  # primary + snapshot
        act_bytes = self.act_bufs._buf_bytes * 2    # double buffer
        res_bytes = self.residual._buf_bytes
        return {
            "kv_cache_bytes": kv_bytes,
            "deltanet_state_bytes": dn_bytes,
            "activation_buffer_bytes": act_bytes,
            "residual_buffer_bytes": res_bytes,
            "total_engine_bytes": kv_bytes + dn_bytes + act_bytes + res_bytes,
        }


# ---------------------------------------------------------------------------
# Convenience: build engine from paths (will integrate with real loader)
# ---------------------------------------------------------------------------

def build_engine(
    model_path: Optional[str] = None,
    kernel_dir: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> LithosEngine:
    """Construct a LithosEngine with default config.

    Until the loader and kernel loading are wired up, this creates
    a mock engine suitable for testing the orchestration logic.
    """
    if config is None:
        config = ModelConfig()

    model = LoadedModel(config=config)
    kernels = LoadedKernels()
    return LithosEngine(model, kernels)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke test: prefill + one decode step, verify launch counts."""
    print("=== Lithos Engine Self-Test ===\n")

    engine = build_engine()
    engine.print_layer_map()

    print()
    mem = engine.memory_report()
    for k, v in mem.items():
        print(f"  {k}: {v / (1024**2):.1f} MB")

    print(f"\n  Config: {engine.config.num_layers} layers "
          f"({engine.config.num_deltanet_layers} DeltaNet, "
          f"{engine.config.num_full_attention_layers} FullAttention)")

    # Prefill
    LithosEngine.clear_launch_log()
    prompt = list(range(128))  # 128 mock token ids
    engine.prefill(prompt)
    prefill_launches = len(LithosEngine.get_launch_log())
    print(f"\n  Prefill ({len(prompt)} tokens): {prefill_launches} kernel launches")

    # Decode
    LithosEngine.clear_launch_log()
    token = engine.decode_step()
    decode_launches = len(LithosEngine.get_launch_log())
    print(f"  Decode step: {decode_launches} kernel launches, token={token}")

    # Verify structure
    log = LithosEngine.get_launch_log()
    dn_count = sum(1 for e in log if "deltanet_recurrence" in e)
    attn_count = sum(1 for e in log if "attention_score" in e)
    print(f"  Decode: {dn_count} DeltaNet recurrences, {attn_count} attention scores")

    assert dn_count == 48, f"Expected 48 DeltaNet layers, got {dn_count}"
    assert attn_count == 16, f"Expected 16 attention layers, got {attn_count}"
    print("\n  All assertions passed.")

    engine.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    _self_test()
