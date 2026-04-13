"""
Automatic weight staging for dtype conversion at startup.

The RMSNorm kernel expects F32 weights, but the model stores them as BF16.
Previously, test_full_layer.py and generate_first_token.py manually converted
each norm weight from BF16 -> F32 and uploaded it to a GPU buffer before every
kernel call.  This module replaces that pattern with a single, up-front staging
pass that:

  1. Scans all weight tensors for dtype mismatches
  2. Converts BF16 -> F32 (norm weights)
  3. Stores every converted tensor in one contiguous mmap'd buffer
  4. Provides O(1) pointer lookup by tensor name (or layer index + sublayer)

Weight dtype inventory for the Qwen 3.5-27B GPTQ model:

  qweight        int32   GPTQ packed bits -- no conversion
  scales         f16     used directly by the GPTQ matvec kernel
  embed_tokens   f16     handled by the embed_f16 kernel
  lm_head        f16     same layout as embed; projection kernel reads f16
  *_layernorm    bf16    ** NEEDS f32 conversion ** (this module)
  model.norm     bf16    final norm -- same treatment

The class is designed to be instantiated once at engine startup and kept alive
for the duration of inference.  The converted buffer is backed by an anonymous
mmap so the OS can page it in/out without fragmenting the Python heap.
"""

from __future__ import annotations

import ctypes
import mmap
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from loader import LithosModel, TensorInfo


# ---------------------------------------------------------------------------
# BF16 -> F32 conversion (bit-level, no PyTorch dependency)
# ---------------------------------------------------------------------------

def _bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convert raw BF16 bytes to an F32 numpy array.

    BF16 is the upper 16 bits of an IEEE-754 float32, so conversion is
    a left-shift by 16 bits.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = np.empty(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def _f16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convert raw F16 bytes to an F32 numpy array."""
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# Weight staging rules
# ---------------------------------------------------------------------------

# Maps source dtype -> (target dtype, converter function)
# Only dtypes that actually need conversion are listed here.
_CONVERSION_TABLE: Dict[str, Tuple[str, type]] = {
    "BF16": ("F32", _bf16_bytes_to_f32),
    # F16 norm weights are rare but possible in other models.
    "F16":  ("F32", _f16_bytes_to_f32),
}


def _needs_staging(tensor_name: str, dtype: str) -> bool:
    """Return True if this tensor must be staged (dtype-converted).

    Current policy: stage all norm weights that are not already F32.
    """
    if dtype == "F32":
        return False
    if dtype not in _CONVERSION_TABLE:
        return False
    # Match norm weight tensors by name pattern:
    #   model.language_model.layers.N.input_layernorm.weight
    #   model.language_model.layers.N.post_attention_layernorm.weight
    #   model.language_model.norm.weight
    # General rule: any tensor whose name contains "layernorm.weight" or
    # ends with "norm.weight".
    if "layernorm.weight" in tensor_name:
        return True
    if tensor_name.endswith("norm.weight"):
        return True
    return False


# ---------------------------------------------------------------------------
# StagedWeight -- metadata for a single converted tensor
# ---------------------------------------------------------------------------

class StagedWeight:
    """Metadata for a single staged (converted) weight tensor."""

    __slots__ = ("name", "offset", "byte_size", "numel", "src_dtype", "dst_dtype")

    def __init__(self, name: str, offset: int, byte_size: int, numel: int,
                 src_dtype: str, dst_dtype: str):
        self.name = name
        self.offset = offset        # byte offset into the staging buffer
        self.byte_size = byte_size   # bytes in the staging buffer (f32)
        self.numel = numel           # number of elements
        self.src_dtype = src_dtype   # original dtype in the model file
        self.dst_dtype = dst_dtype   # dtype after conversion (always F32)


# ---------------------------------------------------------------------------
# WeightStager -- the main class
# ---------------------------------------------------------------------------

class WeightStager:
    """Scans a LithosModel for weights needing dtype conversion, converts them
    into a single contiguous mmap'd buffer, and provides pointer lookup.

    Typical usage::

        model = LithosModel("/path/to/model")
        stager = WeightStager(model)          # scans + converts at init

        # Get the host pointer for layer 3 input norm (usable with ctypes):
        ptr = stager.ptr("model.language_model.layers.3.input_layernorm.weight")

        # Or by layer index:
        ptr = stager.input_norm_ptr(3)
        ptr = stager.post_attn_norm_ptr(3)
        ptr = stager.final_norm_ptr()

    The returned pointers are valid as long as the WeightStager is alive.
    """

    def __init__(self, model: LithosModel, *, verbose: bool = True):
        self._model = model
        self._staged: Dict[str, StagedWeight] = {}
        self._buf: Optional[mmap.mmap] = None
        self._buf_ptr: int = 0
        self._total_bytes: int = 0

        # Layer-index lookup caches (filled during scan)
        self._input_norm_names: Dict[int, str] = {}
        self._post_attn_norm_names: Dict[int, str] = {}
        self._final_norm_name: Optional[str] = None

        self._scan_and_stage(verbose=verbose)

    # ------------------------------------------------------------------
    # Public API -- pointer access
    # ------------------------------------------------------------------

    def ptr(self, tensor_name: str) -> int:
        """Return the host virtual address of the staged (F32) tensor.

        Raises KeyError if the tensor was not staged.
        """
        sw = self._staged[tensor_name]
        return self._buf_ptr + sw.offset

    def has(self, tensor_name: str) -> bool:
        """Return True if the tensor has been staged."""
        return tensor_name in self._staged

    def input_norm_ptr(self, layer_idx: int) -> int:
        """Return the F32 pointer for layers.{layer_idx}.input_layernorm.weight."""
        return self.ptr(self._input_norm_names[layer_idx])

    def post_attn_norm_ptr(self, layer_idx: int) -> int:
        """Return the F32 pointer for layers.{layer_idx}.post_attention_layernorm.weight."""
        return self.ptr(self._post_attn_norm_names[layer_idx])

    def final_norm_ptr(self) -> int:
        """Return the F32 pointer for the final RMSNorm weight."""
        if self._final_norm_name is None:
            raise KeyError("No final norm weight found in model")
        return self.ptr(self._final_norm_name)

    def staged_names(self) -> List[str]:
        """Return names of all staged tensors, sorted."""
        return sorted(self._staged.keys())

    @property
    def total_bytes(self) -> int:
        """Total bytes consumed by the staging buffer."""
        return self._total_bytes

    @property
    def num_staged(self) -> int:
        """Number of tensors that were staged."""
        return len(self._staged)

    # ------------------------------------------------------------------
    # Convenience: get F32 numpy view (for CPU reference / testing)
    # ------------------------------------------------------------------

    def as_f32(self, tensor_name: str) -> np.ndarray:
        """Return a numpy F32 array backed by the staging buffer (zero-copy)."""
        sw = self._staged[tensor_name]
        return np.frombuffer(
            self._buf,
            dtype=np.float32,
            count=sw.numel,
            offset=sw.offset,
        )

    # ------------------------------------------------------------------
    # Internal: scan model and build the staging buffer
    # ------------------------------------------------------------------

    def _scan_and_stage(self, *, verbose: bool) -> None:
        """Walk all tensors, identify those needing conversion, allocate one
        contiguous buffer, and perform the conversions."""

        # --- Pass 1: identify tensors and compute total buffer size ---
        to_stage: List[Tuple[str, TensorInfo, int]] = []  # (name, ti, f32_bytes)
        total = 0

        for name in self._model.weight_names():
            ti = self._model.weight_info(name)
            if not _needs_staging(name, ti.dtype):
                continue

            numel = ti.byte_size // ti.element_size
            f32_bytes = numel * 4  # F32 = 4 bytes per element

            # Align each tensor to 64 bytes for cache-line alignment
            f32_bytes_aligned = (f32_bytes + 63) & ~63

            to_stage.append((name, ti, f32_bytes_aligned))
            total += f32_bytes_aligned

            # Build layer-index caches
            self._cache_norm_name(name)

        if total == 0:
            if verbose:
                print("[WeightStager] No tensors need staging.")
            return

        # --- Allocate contiguous buffer via anonymous mmap ---
        self._buf = mmap.mmap(-1, total, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self._buf_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        self._total_bytes = total

        # --- Pass 2: convert and copy into the buffer ---
        offset = 0
        for name, ti, f32_bytes_aligned in to_stage:
            raw = bytes(self._model.weight_bytes(name))
            converter = _CONVERSION_TABLE[ti.dtype][1]
            f32_data = converter(raw)

            numel = len(f32_data)
            actual_bytes = numel * 4

            # Copy into the mmap buffer
            self._buf[offset:offset + actual_bytes] = f32_data.tobytes()

            self._staged[name] = StagedWeight(
                name=name,
                offset=offset,
                byte_size=actual_bytes,
                numel=numel,
                src_dtype=ti.dtype,
                dst_dtype="F32",
            )
            offset += f32_bytes_aligned

        if verbose:
            n = len(self._staged)
            mb = total / (1024 * 1024)
            print(f"[WeightStager] Staged {n} tensors "
                  f"({mb:.2f} MiB, {total:,} bytes)")
            # Breakdown by source dtype
            from collections import Counter
            dtype_counts = Counter(sw.src_dtype for sw in self._staged.values())
            for dt, cnt in dtype_counts.most_common():
                print(f"  {dt} -> F32: {cnt} tensors")

    def _cache_norm_name(self, name: str) -> None:
        """Extract layer index from norm weight names for fast lookup."""
        # model.language_model.layers.N.input_layernorm.weight
        if "input_layernorm.weight" in name:
            idx = self._extract_layer_idx(name)
            if idx is not None:
                self._input_norm_names[idx] = name
        # model.language_model.layers.N.post_attention_layernorm.weight
        elif "post_attention_layernorm.weight" in name:
            idx = self._extract_layer_idx(name)
            if idx is not None:
                self._post_attn_norm_names[idx] = name
        # model.language_model.norm.weight  (final norm before lm_head)
        # Be specific: match the language model's final norm, not visual/MTP norms
        elif name.endswith("norm.weight") and "layers." not in name:
            # Prefer the language_model norm; only set if not already set or
            # this is the language_model one.
            if "language_model" in name or self._final_norm_name is None:
                self._final_norm_name = name

    @staticmethod
    def _extract_layer_idx(name: str) -> Optional[int]:
        """Extract the layer index from a tensor name like
        'model.language_model.layers.42.input_layernorm.weight'."""
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the staging buffer."""
        self._staged.clear()
        if self._buf is not None:
            try:
                self._buf.close()
            except BufferError:
                pass
            self._buf = None
        self._buf_ptr = 0
        self._total_bytes = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        mb = self._total_bytes / (1024 * 1024)
        return (f"WeightStager({self.num_staged} tensors, "
                f"{mb:.2f} MiB)")
