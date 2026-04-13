"""
Lithos model loader -- reads safetensors files and provides weight pointers to kernels.

Pure Python (stdlib + mmap). No PyTorch dependency.
Marked for rewrite to C/CUDA once the kernel interface stabilizes.

Usage:
    model = LithosModel("/path/to/model")
    ptr, dtype, shape = model.weight("model.language_model.layers.0.self_attn.q_proj.qweight")
    print(model.config)
    print(model.kv_cache_per_token)
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Safetensors dtype -> (numpy dtype string, element size in bytes)
# ---------------------------------------------------------------------------
SAFETENSOR_DTYPES = {
    "F64":  ("float64",  8),
    "F32":  ("float32",  4),
    "F16":  ("float16",  2),
    "BF16": ("bfloat16", 2),
    "I64":  ("int64",    8),
    "I32":  ("int32",    4),
    "I16":  ("int16",    2),
    "I8":   ("int8",     1),
    "U8":   ("uint8",    1),
    "BOOL": ("bool",     1),
}


@dataclass
class TensorInfo:
    """Metadata for a single tensor in the weight table."""
    shard_path: str           # absolute path to the .safetensors file
    mmap_buf: mmap.mmap       # mmap object for the shard
    data_offset: int          # byte offset from start of mmap to tensor data
    byte_size: int            # total bytes of this tensor's data
    dtype: str                # safetensors dtype string (e.g. "F16", "I32")
    shape: List[int]          # tensor shape
    element_size: int         # bytes per element

    @property
    def ptr(self) -> int:
        """Return the virtual address of the tensor data (for passing to kernels via ctypes).

        Works with read-only mmaps by reading the internal buffer pointer
        directly from the mmap object via ctypes.
        """
        # mmap objects in CPython expose their buffer through the buffer protocol.
        # For read-only mmaps we cannot use from_buffer, so we use the
        # /proc/self/maps approach or the ctypes hack to get the base address.
        # The most portable CPython approach: use ctypes to call into the
        # buffer protocol via PyObject_GetBuffer.
        import ctypes as ct

        class Py_buffer(ct.Structure):
            _fields_ = [
                ("buf", ct.c_void_p),
                ("obj", ct.py_object),
                ("len", ct.c_ssize_t),
                ("itemsize", ct.c_ssize_t),
                ("readonly", ct.c_int),
                ("ndim", ct.c_int),
                ("format", ct.c_char_p),
                ("shape", ct.POINTER(ct.c_ssize_t)),
                ("strides", ct.POINTER(ct.c_ssize_t)),
                ("suboffsets", ct.POINTER(ct.c_ssize_t)),
                ("internal", ct.c_void_p),
            ]

        buf = Py_buffer()
        # PyBUF_SIMPLE = 0
        ret = ct.pythonapi.PyObject_GetBuffer(ct.py_object(self.mmap_buf), ct.byref(buf), 0)
        if ret != 0:
            raise RuntimeError("Failed to get buffer from mmap")
        base_ptr = buf.buf
        ct.pythonapi.PyBuffer_Release(ct.byref(buf))
        return base_ptr + self.data_offset


@dataclass
class ModelConfig:
    """Extracted model dimensions relevant for inference."""
    arch: str = ""
    hidden_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    layer_types: List[str] = field(default_factory=list)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    max_position_embeddings: int = 262144
    # Linear attention (Mamba-style) params
    linear_key_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_value_head_dim: int = 0
    linear_num_value_heads: int = 0
    linear_conv_kernel_dim: int = 0
    # Quantization
    quant_method: str = ""
    quant_bits: int = 0
    quant_group_size: int = 0
    quant_sym: bool = True
    # Raw dicts for anything we didn't parse
    raw_text_config: Dict[str, Any] = field(default_factory=dict)
    raw_quant_config: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [
            f"ModelConfig(",
            f"  arch            = {self.arch!r}",
            f"  hidden_dim      = {self.hidden_dim}",
            f"  num_heads       = {self.num_heads}",
            f"  num_kv_heads    = {self.num_kv_heads}",
            f"  head_dim        = {self.head_dim}",
            f"  intermediate    = {self.intermediate_size}",
            f"  num_layers      = {self.num_layers}",
            f"  vocab_size      = {self.vocab_size}",
            f"  layer_types     = {self._layer_type_summary()}",
            f"  quant           = {self.quant_method} w{self.quant_bits} g{self.quant_group_size}",
            f"  linear_attn     = k_heads={self.linear_num_key_heads} k_dim={self.linear_key_head_dim}"
            f" v_heads={self.linear_num_value_heads} v_dim={self.linear_value_head_dim}"
            f" conv_k={self.linear_conv_kernel_dim}",
            f")",
        ]
        return "\n".join(lines)

    def _layer_type_summary(self) -> str:
        from collections import Counter
        c = Counter(self.layer_types)
        parts = [f"{v}x {k}" for k, v in c.most_common()]
        return ", ".join(parts) + f" ({len(self.layer_types)} total)"


class LithosModel:
    """
    Loads a safetensors model directory and provides zero-copy weight access via mmap.

    Supports both single-file models (model.safetensors) and sharded models
    (model.safetensors.index.json + model-NNNNN.safetensors).
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._mmaps: Dict[str, mmap.mmap] = {}          # shard_path -> mmap
        self._fds: Dict[str, int] = {}                    # shard_path -> fd
        self._weights: Dict[str, TensorInfo] = {}         # tensor_name -> TensorInfo
        self.config: ModelConfig = ModelConfig()

        self._load_config()
        self._load_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weight(self, name: str) -> Tuple[int, str, List[int]]:
        """
        Look up a tensor by name.

        Returns:
            (ptr, dtype, shape) where ptr is a virtual memory address usable
            with ctypes / kernel launches, dtype is the safetensors dtype
            string, and shape is the tensor shape.

        Raises KeyError if the tensor is not found.
        """
        ti = self._weights[name]
        return (ti.ptr, ti.dtype, list(ti.shape))

    def weight_info(self, name: str) -> TensorInfo:
        """Return the full TensorInfo for a tensor."""
        return self._weights[name]

    def weight_names(self) -> List[str]:
        """Return all tensor names, sorted."""
        return sorted(self._weights.keys())

    def weight_bytes(self, name: str) -> memoryview:
        """Return a zero-copy memoryview of the raw tensor bytes."""
        ti = self._weights[name]
        return memoryview(ti.mmap_buf)[ti.data_offset : ti.data_offset + ti.byte_size]

    @property
    def total_weight_bytes(self) -> int:
        """Total bytes across all weight tensors."""
        return sum(ti.byte_size for ti in self._weights.values())

    @property
    def kv_cache_per_token(self) -> int:
        """
        Bytes of KV cache needed per token (across all layers), stored in float16.

        For this hybrid model:
        - full_attention layers: standard KV cache (num_kv_heads * head_dim * 2 for K+V)
        - linear_attention layers: Mamba-style SSM state, not per-token KV in the
          traditional sense. We report the recurrent state size per layer instead.

        All values assume float16 (2 bytes per element).

        For this hybrid architecture:
        - full_attention layers need per-token KV cache that grows with sequence length
        - linear_attention layers use fixed-size recurrent state (conv + SSM) that
          does NOT grow with sequence length

        This property returns the per-token cost for full_attention layers only.
        Use kv_cache_report() for the complete picture including fixed state.
        """
        cfg = self.config
        bytes_per_element = 2  # float16

        full_attn_per_layer = (
            cfg.num_kv_heads * cfg.head_dim * 2 * bytes_per_element  # K + V
        )

        num_full = sum(1 for lt in cfg.layer_types if lt == "full_attention")
        num_linear = len(cfg.layer_types) - num_full

        per_token = full_attn_per_layer * num_full

        # Linear attention fixed recurrent state (does NOT scale with tokens):
        # Conv state per layer: conv1d channels * (kernel_size - 1)
        #   conv1d channels = linear_num_key_heads * key_head_dim +
        #                     linear_num_value_heads * value_head_dim
        conv_channels = (cfg.linear_num_key_heads * cfg.linear_key_head_dim +
                         cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        conv_state_per_layer = conv_channels * max(cfg.linear_conv_kernel_dim - 1, 0) * bytes_per_element

        # SSM recurrent state per layer: num_value_heads * value_head_dim * key_head_dim
        # This is the (d_state x d_inner) matrix that accumulates over time
        ssm_state_per_layer = (
            cfg.linear_num_value_heads * cfg.linear_value_head_dim *
            cfg.linear_key_head_dim * bytes_per_element
        )
        linear_state_per_layer = conv_state_per_layer + ssm_state_per_layer
        linear_state_total = linear_state_per_layer * num_linear

        self._kv_cache_detail = {
            "full_attention_layers": num_full,
            "linear_attention_layers": num_linear,
            "full_attn_per_layer_per_token_bytes": full_attn_per_layer,
            "full_attn_per_token_total_bytes": per_token,
            "linear_state_per_layer_bytes": linear_state_per_layer,
            "linear_conv_state_per_layer_bytes": conv_state_per_layer,
            "linear_ssm_state_per_layer_bytes": ssm_state_per_layer,
            "linear_state_total_bytes": linear_state_total,
        }
        return per_token

    def kv_cache_report(self, max_tokens: int = 8192) -> str:
        """Human-readable report of KV cache / state requirements."""
        per_tok = self.kv_cache_per_token
        detail = self._kv_cache_detail
        kv_alloc = per_tok * max_tokens
        linear_total = detail["linear_state_total_bytes"]
        total_alloc = kv_alloc + linear_total

        lines = [
            "=== KV Cache / Recurrent State Report ===",
            f"",
            f"Full attention layers:    {detail['full_attention_layers']}",
            f"  Per layer per token:    {detail['full_attn_per_layer_per_token_bytes']:,} B"
            f"  (KV heads={self.config.num_kv_heads} x head_dim={self.config.head_dim} x 2 x fp16)",
            f"  Per token (all layers): {per_tok:,} B ({per_tok / 1024:.1f} KiB)",
            f"  For {max_tokens:,} tokens:       {kv_alloc:,} B ({kv_alloc / (1024**2):.1f} MiB)",
            f"",
            f"Linear attention layers:  {detail['linear_attention_layers']}",
            f"  Per layer (fixed state):{detail['linear_state_per_layer_bytes']:,} B",
            f"    conv state:           {detail['linear_conv_state_per_layer_bytes']:,} B",
            f"    ssm state:            {detail['linear_ssm_state_per_layer_bytes']:,} B",
            f"  Total fixed state:      {linear_total:,} B ({linear_total / (1024**2):.1f} MiB)",
            f"",
            f"Total allocation ({max_tokens:,} tokens + linear state):",
            f"  {total_alloc:,} B ({total_alloc / (1024**2):.1f} MiB, {total_alloc / (1024**3):.3f} GiB)",
        ]
        return "\n".join(lines)

    def close(self):
        """Unmap all shards and close file descriptors."""
        # Clear weights first so no TensorInfo holds mmap refs
        self._weights.clear()
        for mm in self._mmaps.values():
            try:
                mm.close()
            except BufferError:
                pass  # Outstanding memoryview references; GC will handle it
        for fd in self._fds.values():
            os.close(fd)
        self._mmaps.clear()
        self._fds.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        n = len(self._weights)
        total_mb = self.total_weight_bytes / (1024 ** 2)
        return f"LithosModel({self.model_dir.name!r}, {n} tensors, {total_mb:.0f} MiB)"

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self):
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            return

        with open(config_path) as f:
            raw = json.load(f)

        cfg = self.config
        cfg.arch = raw.get("architectures", ["unknown"])[0] if "architectures" in raw else raw.get("model_type", "unknown")

        # Qwen3.5 nests text model config under "text_config"
        text = raw.get("text_config", raw)
        cfg.raw_text_config = text

        cfg.hidden_dim = text.get("hidden_size", 0)
        cfg.num_heads = text.get("num_attention_heads", 0)
        cfg.num_kv_heads = text.get("num_key_value_heads", cfg.num_heads)
        cfg.head_dim = text.get("head_dim", cfg.hidden_dim // cfg.num_heads if cfg.num_heads else 0)
        cfg.intermediate_size = text.get("intermediate_size", 0)
        cfg.num_layers = text.get("num_hidden_layers", 0)
        cfg.vocab_size = text.get("vocab_size", 0)
        cfg.layer_types = text.get("layer_types", ["full_attention"] * cfg.num_layers)
        cfg.rms_norm_eps = text.get("rms_norm_eps", 1e-6)
        cfg.max_position_embeddings = text.get("max_position_embeddings", 131072)

        # Rope params
        rope_params = text.get("rope_parameters", text.get("rope_scaling", {}))
        if rope_params:
            cfg.rope_theta = rope_params.get("rope_theta", text.get("rope_theta", 10_000_000.0))
        else:
            cfg.rope_theta = text.get("rope_theta", 10_000_000.0)

        # Linear attention params
        cfg.linear_key_head_dim = text.get("linear_key_head_dim", 0)
        cfg.linear_num_key_heads = text.get("linear_num_key_heads", 0)
        cfg.linear_value_head_dim = text.get("linear_value_head_dim", 0)
        cfg.linear_num_value_heads = text.get("linear_num_value_heads", 0)
        cfg.linear_conv_kernel_dim = text.get("linear_conv_kernel_dim", 0)

        # Quantization config -- can be nested in config.json or separate file
        qcfg = raw.get("quantization_config", {})
        if not qcfg:
            qcfg_path = self.model_dir / "quantization_config.json"
            if qcfg_path.exists():
                with open(qcfg_path) as f:
                    qcfg = json.load(f)

        cfg.raw_quant_config = qcfg
        cfg.quant_method = qcfg.get("quant_method", "")
        cfg.quant_bits = qcfg.get("bits", 0)
        cfg.quant_group_size = qcfg.get("group_size", 0)
        cfg.quant_sym = qcfg.get("sym", True)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self):
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"

        if index_path.exists():
            self._load_sharded(index_path)
        elif single_path.exists():
            self._load_single_file(single_path)
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.safetensors.index.json in {self.model_dir}"
            )

    def _load_sharded(self, index_path: Path):
        """Load a multi-shard safetensors model."""
        with open(index_path) as f:
            index = json.load(f)

        weight_map: Dict[str, str] = index["weight_map"]

        # Collect unique shard files
        shard_files = set(weight_map.values())

        # mmap and parse each shard
        shard_headers: Dict[str, dict] = {}
        for shard_name in shard_files:
            shard_path = str(self.model_dir / shard_name)
            self._mmap_shard(shard_path)
            shard_headers[shard_name] = self._parse_shard_header(shard_path)

        # Build weight table
        for tensor_name, shard_name in weight_map.items():
            shard_path = str(self.model_dir / shard_name)
            header = shard_headers[shard_name]
            if tensor_name not in header:
                raise KeyError(f"Tensor {tensor_name!r} not found in shard {shard_name}")
            self._register_tensor(tensor_name, shard_path, header)

    def _load_single_file(self, path: Path):
        """Load a single safetensors file."""
        shard_path = str(path)
        self._mmap_shard(shard_path)
        header = self._parse_shard_header(shard_path)

        for tensor_name in header:
            if tensor_name == "__metadata__":
                continue
            self._register_tensor(tensor_name, shard_path, header)

    def _mmap_shard(self, shard_path: str):
        """Memory-map a shard file with PROT_READ, MAP_PRIVATE."""
        if shard_path in self._mmaps:
            return
        fd = os.open(shard_path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        mm = mmap.mmap(
            fd,
            size,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ,
        )
        self._fds[shard_path] = fd
        self._mmaps[shard_path] = mm

    def _parse_shard_header(self, shard_path: str) -> dict:
        """
        Parse the safetensors JSON header from an mmap'd shard.

        Safetensors format:
            bytes [0..8):   little-endian u64 = header_size
            bytes [8..8+header_size): JSON header
            bytes [8+header_size..):  raw tensor data

        The JSON header maps tensor names to:
            {"dtype": "F16", "shape": [M, N], "data_offsets": [start, end]}
        where offsets are relative to the start of the data section (byte 8+header_size).
        """
        mm = self._mmaps[shard_path]
        header_size = struct.unpack_from("<Q", mm, 0)[0]
        header_json = mm[8 : 8 + header_size]
        header = json.loads(header_json)

        # Store the data section offset for this shard
        header["__data_offset__"] = 8 + header_size
        return header

    def _register_tensor(self, tensor_name: str, shard_path: str, header: dict):
        """Create a TensorInfo entry from parsed header data."""
        info = header[tensor_name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        data_offsets = info["data_offsets"]  # [start, end] relative to data section

        if dtype_str not in SAFETENSOR_DTYPES:
            raise ValueError(f"Unknown safetensors dtype: {dtype_str!r} for {tensor_name}")

        _, elem_size = SAFETENSOR_DTYPES[dtype_str]

        # Absolute offset into the mmap
        data_section_start = header["__data_offset__"]
        abs_offset = data_section_start + data_offsets[0]
        byte_size = data_offsets[1] - data_offsets[0]

        self._weights[tensor_name] = TensorInfo(
            shard_path=shard_path,
            mmap_buf=self._mmaps[shard_path],
            data_offset=abs_offset,
            byte_size=byte_size,
            dtype=dtype_str,
            shape=shape,
            element_size=elem_size,
        )


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

def main():
    import sys
    import time

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

    print(f"Loading model from {model_dir} ...")
    t0 = time.monotonic()
    model = LithosModel(model_dir)
    t1 = time.monotonic()
    print(f"Loaded in {t1 - t0:.3f}s\n")

    print(model)
    print()
    print(model.config)
    print()

    # Total weight bytes
    total = model.total_weight_bytes
    print(f"Total weight bytes: {total:,} ({total / (1024**3):.2f} GiB)")
    print(f"Total tensors:      {len(model.weight_names())}")
    print()

    # KV cache report
    print(model.kv_cache_report(max_tokens=8192))
    print()

    # Sample weight lookups
    test_names = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.qweight",
        "model.language_model.layers.3.self_attn.q_proj.qweight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ]
    for name in test_names:
        try:
            ptr, dtype, shape = model.weight(name)
            ti = model.weight_info(name)
            print(f"{name}")
            print(f"  ptr=0x{ptr:016x}  dtype={dtype}  shape={shape}  bytes={ti.byte_size:,}")
        except KeyError:
            print(f"{name} -- NOT FOUND")

    # Verify data integrity: read first few bytes of a tensor
    print("\n=== Data integrity check ===")
    ti = model.weight_info("model.language_model.layers.0.linear_attn.A_log")
    raw = model.weight_bytes("model.language_model.layers.0.linear_attn.A_log")
    # A_log is F32, shape [48] -> 192 bytes
    assert len(raw) == 48 * 4, f"Expected 192 bytes, got {len(raw)}"
    vals = struct.unpack_from(f"<{48}f", raw)
    print(f"A_log (F32, shape=[48]): first 4 values = {vals[:4]}")
    print(f"  min={min(vals):.6f} max={max(vals):.6f}")
    print("Data integrity check PASSED")

    model.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
