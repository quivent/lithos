#!/usr/bin/env python3
"""
Layer-by-layer comparison of Lithos inference vs PyTorch (transformers).

Loads the GPTQ model weights from safetensors, manually dequantizes them,
builds the same model forward pass in pure NumPy/CPU, and compares hidden
states at each layer boundary against the Lithos GPU engine.

This identifies the FIRST layer where the two implementations diverge
(cosine similarity < 0.99), pinpointing whether the error is in:
  - GPTQ dequantization (gptq_matvec kernel)
  - DeltaNet recurrence
  - Full attention (RoPE, QK norm, etc.)
  - MLP (SiLU activation, projections)
  - RMSNorm

Run:
    python3 /home/ubuntu/lithos/src/layer_by_layer_compare.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import struct
import sys
import time

import numpy as np
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_driver import CUDADriver, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer
from kv_cache import KVCache
from attention import process_qkv_with_rope, attention_prefill

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
ZERO_POINT = 7
NUM_LAYERS = 64

# DeltaNet dimensions
NUM_KEY_HEADS = 16
KEY_HEAD_DIM = 128
NUM_VALUE_HEADS = 48
VALUE_HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HEADS_PER_KEY = NUM_VALUE_HEADS // NUM_KEY_HEADS  # 3

# Full attention dimensions
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256

EPSILON = 1e-6

# ============================================================================
# Utility functions
# ============================================================================

def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.flatten() - b.flatten())))


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


# ============================================================================
# GPTQ Dequantization (CPU reference)
# ============================================================================

class GPTQDequantizer:
    """Dequantize GPTQ 4-bit weights from safetensors on CPU.

    Format:
      qweight: [K//8, N] int32 -- 8 x 4-bit weights packed per int32
      scales:  [K//group_size, N] float16 -- per-group scale
      qzeros:  [K//group_size, N//8] int32 -- 8 x 4-bit zeros packed per int32

    Dequantization:  w = (q_unpacked - z_unpacked) * scale
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.index = json.load(f)
        self._shard_cache_np = {}  # for numpy-safe dtypes (int32, float16, float32)
        self._shard_cache_pt = {}  # for bfloat16 via torch

    def _get_shard_np(self, shard_name: str):
        if shard_name not in self._shard_cache_np:
            path = os.path.join(self.model_dir, shard_name)
            self._shard_cache_np[shard_name] = safe_open(path, framework="numpy")
        return self._shard_cache_np[shard_name]

    def _get_shard_pt(self, shard_name: str):
        import torch
        if shard_name not in self._shard_cache_pt:
            path = os.path.join(self.model_dir, shard_name)
            self._shard_cache_pt[shard_name] = safe_open(path, framework="pt")
        return self._shard_cache_pt[shard_name]

    def get_tensor(self, name: str) -> np.ndarray:
        """Get tensor as float32 numpy array, handling bfloat16 via torch."""
        shard_name = self.index["weight_map"][name]
        try:
            shard = self._get_shard_np(shard_name)
            t = shard.get_tensor(name)
            if t.dtype == np.float16:
                return t.astype(np.float32)
            return t
        except TypeError:
            # bfloat16 - use torch
            import torch
            shard = self._get_shard_pt(shard_name)
            t = shard.get_tensor(name)
            return t.float().numpy()

    def dequantize(self, prefix: str) -> np.ndarray:
        """Dequantize a GPTQ linear layer, returning [N, K] float32 weight matrix.

        The GPTQ matvec computes output = W @ input where W is [N, K].
        But qweight is stored as [K//8, N], so we need to unpack and transpose.

        Actually for GPTQ: qweight is [K_packed, N] where K_packed = K // 8.
        Each int32 in qweight packs 8 4-bit values along the K dimension.
        So unpacked shape is [K, N], and the weight matrix for y = Wx is W = unpacked.T = [N, K].
        """
        qweight = self.get_tensor(f"{prefix}.qweight")   # [K//8, N]
        scales = self.get_tensor(f"{prefix}.scales")      # [num_groups, N]
        qzeros = self.get_tensor(f"{prefix}.qzeros")      # [num_groups, N//8]

        K_packed, N = qweight.shape
        K = K_packed * 8
        num_groups = scales.shape[0]

        # Unpack qweight: each int32 -> 8 x 4-bit values
        # Bit layout: bits [0:4] = row 0, bits [4:8] = row 1, ..., bits [28:32] = row 7
        qw_unpacked = np.zeros((K, N), dtype=np.int32)
        for bit in range(8):
            qw_unpacked[bit::8, :] = (qweight >> (bit * 4)) & 0xF

        # Unpack qzeros: each int32 -> 8 x 4-bit values
        N_z_packed = qzeros.shape[1]
        qz_unpacked = np.zeros((num_groups, N), dtype=np.int32)
        for bit in range(8):
            qz_unpacked[:, bit::8] = (qzeros >> (bit * 4)) & 0xF

        # Dequantize: w = (q - z) * scale
        scales_f32 = scales if scales.dtype == np.float32 else scales.astype(np.float32)  # [num_groups, N]

        w_dequant = np.zeros((K, N), dtype=np.float32)
        for g in range(num_groups):
            row_start = g * GROUP_SIZE
            row_end = min(row_start + GROUP_SIZE, K)
            w_dequant[row_start:row_end, :] = (
                (qw_unpacked[row_start:row_end, :] - qz_unpacked[g, :]) *
                scales_f32[g, :]
            )

        # Return as [N, K] for matrix-vector multiply: y = W @ x
        return w_dequant.T

    def close(self):
        for s in self._shard_cache.values():
            pass  # safe_open handles cleanup
        self._shard_cache.clear()


# ============================================================================
# PyTorch-equivalent forward pass in NumPy (CPU reference)
# ============================================================================

class ReferenceModel:
    """Pure CPU/NumPy reference implementation matching PyTorch's forward pass.

    Uses dequantized GPTQ weights to compute the exact same operations
    as PyTorch would, but without needing auto-gptq or GPU.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.dequant = GPTQDequantizer(model_dir)

        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        text_cfg = self.config.get("text_config", self.config)
        self.layer_types = text_cfg["layer_types"]
        self.epsilon = text_cfg.get("rms_norm_eps", 1e-6)

        # Preload norm weights
        self._preload_norms()
        self._preload_deltanet_weights()
        self._init_deltanet_state()
        self._preload_qk_norms()
        self.kv_cache = KVCache(max_seq_len=64)

    def _load_bf16_weight(self, name: str) -> np.ndarray:
        t = self.dequant.get_tensor(name)
        return t.astype(np.float32) if t.dtype != np.float32 else t

    def _preload_norms(self):
        self.input_norms = []
        self.post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            inw = self._load_bf16_weight(f"{prefix}.input_layernorm.weight")
            self.input_norms.append(inw.flatten() + 1.0)
            panw = self._load_bf16_weight(f"{prefix}.post_attention_layernorm.weight")
            self.post_attn_norms.append(panw.flatten() + 1.0)
        fnw = self._load_bf16_weight("model.language_model.norm.weight")
        self.final_norm_w = fnw.flatten() + 1.0
        print(f"  Reference: {NUM_LAYERS * 2 + 1} norm weights loaded")

    def _preload_deltanet_weights(self):
        self.dn_conv_w = {}
        self.dn_a_weight = {}
        self.dn_b_weight = {}
        self.dn_dt_bias = {}
        self.dn_A_log = {}
        self.dn_head_norm_w = {}

        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            prefix = f"model.language_model.layers.{i}.linear_attn"
            self.dn_conv_w[i] = self._load_bf16_weight(f"{prefix}.conv1d.weight").reshape(10240, 1, CONV_KERNEL_SIZE)
            self.dn_a_weight[i] = self._load_bf16_weight(f"{prefix}.in_proj_a.weight").reshape(NUM_VALUE_HEADS, HIDDEN_DIM)
            self.dn_b_weight[i] = self._load_bf16_weight(f"{prefix}.in_proj_b.weight").reshape(NUM_VALUE_HEADS, HIDDEN_DIM)
            self.dn_dt_bias[i] = self._load_bf16_weight(f"{prefix}.dt_bias").flatten()
            a_log = self.dequant.get_tensor(f"{prefix}.A_log")
            self.dn_A_log[i] = a_log.astype(np.float32).flatten()
            norm_w = self.dequant.get_tensor(f"{prefix}.norm.weight")
            self.dn_head_norm_w[i] = norm_w.astype(np.float32).flatten()

        print(f"  Reference: DeltaNet weights loaded for {len(self.dn_conv_w)} layers")

    def _init_deltanet_state(self):
        self.dn_S = {}
        self.dn_conv_buf = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            self.dn_S[i] = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=np.float32)
            self.dn_conv_buf[i] = np.zeros((10240, CONV_KERNEL_SIZE - 1), dtype=np.float32)

    def _preload_qk_norms(self):
        self.q_norm_weights = {}
        self.k_norm_weights = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "full_attention":
                continue
            prefix = f"model.language_model.layers.{i}.self_attn"
            self.q_norm_weights[i] = self._load_bf16_weight(f"{prefix}.q_norm.weight").flatten() + 1.0
            self.k_norm_weights[i] = self._load_bf16_weight(f"{prefix}.k_norm.weight").flatten() + 1.0

    def rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2) + self.epsilon)
        return (x / rms) * weight

    def silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x.clip(-80, 80))))

    def matvec(self, prefix: str, x: np.ndarray) -> np.ndarray:
        """Dequantize GPTQ weight and compute W @ x."""
        W = self.dequant.dequantize(prefix)  # [N, K]
        return W @ x

    def embed(self, token_id: int) -> np.ndarray:
        embed_w = self._load_bf16_weight("model.language_model.embed_tokens.weight")
        return embed_w[token_id].flatten()

    def run_deltanet(self, layer_idx: int, normed: np.ndarray) -> np.ndarray:
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        # QKV projection
        qkv = self.matvec(f"{prefix}.in_proj_qkv", normed)
        z = self.matvec(f"{prefix}.in_proj_z", normed)

        # Conv1d
        conv_buf = self.dn_conv_buf[layer_idx]
        conv_w = self.dn_conv_w[layer_idx]
        conv_window = np.concatenate([conv_buf, qkv[:, np.newaxis]], axis=1)
        qkv_conv = np.sum(conv_window * conv_w[:, 0, :], axis=1)
        self.dn_conv_buf[layer_idx] = conv_window[:, 1:]

        # SiLU after conv
        qkv_conv = self.silu(qkv_conv)

        # Split QKV
        q = qkv_conv[:2048].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        k = qkv_conv[2048:4096].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        v = qkv_conv[4096:].reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)

        # Q/K L2 norm + Q scaling
        for hi in range(NUM_KEY_HEADS):
            q_norm = np.sqrt(np.sum(q[hi] ** 2) + 1e-6)
            q[hi] = q[hi] / q_norm
            k_norm = np.sqrt(np.sum(k[hi] ** 2) + 1e-6)
            k[hi] = k[hi] / k_norm
        q = q * (1.0 / np.sqrt(float(KEY_HEAD_DIM)))

        # Beta
        beta = 1.0 / (1.0 + np.exp(-(self.dn_b_weight[layer_idx] @ normed).clip(-80, 80)))

        # dt/decay
        dt = self.dn_a_weight[layer_idx] @ normed + self.dn_dt_bias[layer_idx]
        dt = np.log1p(np.exp(dt.clip(-20, 20)))

        A_exp = np.exp(self.dn_A_log[layer_idx])
        decay = np.exp(-A_exp * dt)

        # DeltaNet recurrence
        S = self.dn_S[layer_idx]
        output_heads = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM), dtype=np.float32)

        for h in range(NUM_VALUE_HEADS):
            key_idx = h // HEADS_PER_KEY
            q_h = q[key_idx]
            k_h = k[key_idx]
            v_h = v[h]
            S[h] *= decay[h]
            kv_mem = S[h] @ k_h
            delta = (v_h - kv_mem) * beta[h]
            S[h] += np.outer(delta, k_h)
            output_heads[h] = S[h] @ q_h

        self.dn_S[layer_idx] = S

        # Group norm
        head_norm_w = self.dn_head_norm_w[layer_idx]
        for h in range(NUM_VALUE_HEADS):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w

        # Gate with Z
        z_heads = z.reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu

        # Output projection
        attn_output_flat = output_heads.flatten()
        return self.matvec(f"{prefix}.out_proj", attn_output_flat)

    def run_full_attention(self, layer_idx: int, normed: np.ndarray, position: int) -> np.ndarray:
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        q_raw = self.matvec(f"{prefix}.q_proj", normed)
        k_raw = self.matvec(f"{prefix}.k_proj", normed)
        v_raw = self.matvec(f"{prefix}.v_proj", normed)

        # QK RMSNorm
        q_heads = q_raw.reshape(NUM_Q_HEADS, HEAD_DIM)
        k_heads = k_raw.reshape(NUM_KV_HEADS, HEAD_DIM)
        q_norm_w = self.q_norm_weights[layer_idx]
        k_norm_w = self.k_norm_weights[layer_idx]
        for h in range(NUM_Q_HEADS):
            rms = np.sqrt(np.mean(q_heads[h] ** 2) + self.epsilon)
            q_heads[h] = (q_heads[h] / rms) * q_norm_w
        for h in range(NUM_KV_HEADS):
            rms = np.sqrt(np.mean(k_heads[h] ** 2) + self.epsilon)
            k_heads[h] = (k_heads[h] / rms) * k_norm_w
        q_raw = q_heads.flatten()
        k_raw = k_heads.flatten()

        # RoPE
        q_rope, k_rope = process_qkv_with_rope(q_raw, k_raw, position)

        # KV cache
        v_2d = v_raw.reshape(NUM_KV_HEADS, HEAD_DIM)
        self.kv_cache.store(layer_idx, position, k_rope, v_2d)
        k_cache, v_cache = self.kv_cache.get(layer_idx, position)

        # Attention
        attended = attention_prefill(q_rope, k_cache, v_cache, position)

        # O projection
        return self.matvec(f"{prefix}.o_proj", attended)

    def run_layer(self, layer_idx: int, x: np.ndarray, position: int) -> np.ndarray:
        layer_type = self.layer_types[layer_idx]
        residual = x.copy()

        # Input norm
        normed = self.rms_norm(x, self.input_norms[layer_idx])

        # Attention
        if layer_type == "full_attention":
            attn_out = self.run_full_attention(layer_idx, normed, position)
        else:
            attn_out = self.run_deltanet(layer_idx, normed)

        x = residual + attn_out
        residual = x.copy()

        # MLP
        mlp_normed = self.rms_norm(x, self.post_attn_norms[layer_idx])
        gate = self.matvec(f"model.language_model.layers.{layer_idx}.mlp.gate_proj", mlp_normed)
        up = self.matvec(f"model.language_model.layers.{layer_idx}.mlp.up_proj", mlp_normed)
        act = self.silu(gate) * up
        down = self.matvec(f"model.language_model.layers.{layer_idx}.mlp.down_proj", act)
        x = residual + down

        return x

    def process_prompt(self, prompt: str) -> list:
        """Process all prompt tokens and return hidden states after each layer of the last token."""
        tok = Tokenizer(self.model_dir)
        token_ids = tok.encode(prompt)
        print(f"  Reference: tokens = {token_ids}")

        # Process each token
        x = None
        for pos, tid in enumerate(token_ids):
            x = self.embed(tid)

            if pos < len(token_ids) - 1:
                # For non-last tokens, just run through all layers (update state)
                for layer_idx in range(NUM_LAYERS):
                    x = self.run_layer(layer_idx, x, pos)
                print(f"  Reference: token {pos} done, norm={np.linalg.norm(x):.4f}")
            else:
                # For the last token, capture hidden states at each layer boundary
                hidden_states = [x.copy()]  # After embedding
                for layer_idx in range(NUM_LAYERS):
                    x = self.run_layer(layer_idx, x, pos)
                    hidden_states.append(x.copy())
                print(f"  Reference: last token done, norm={np.linalg.norm(x):.4f}")

        # Final norm + logits
        x_normed = self.rms_norm(x, self.final_norm_w)

        return hidden_states, x_normed

    def close(self):
        self.dequant.close()


# ============================================================================
# Lithos Engine (GPU) -- adapted from generate_paris.py
# ============================================================================

def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, size * 4)
    return out


def load_norm_weight(model: LithosModel, name: str) -> np.ndarray:
    ti = model.weight_info(name)
    raw = bytes(model.weight_bytes(name))
    if ti.dtype == "BF16":
        return bf16_to_f32(raw)
    elif ti.dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif ti.dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    else:
        return np.frombuffer(raw, dtype=np.float32).copy()


class LithosEngine:
    """Wraps the Lithos GPU engine to capture hidden states at each layer boundary."""

    def __init__(self):
        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self.layer_types = self.model.config.layer_types

        self._load_kernels()
        self._alloc_buffers()
        self._preload_norms()
        self._preload_deltanet_weights()
        self._init_deltanet_state()
        self.kv_cache = KVCache(max_seq_len=64)
        self._preload_qk_norms()

    def _load_kernels(self):
        gpu = self.gpu
        embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = gpu.get_function(embed_mod, "embed_f16")
        norm_mod = gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = gpu.get_function(norm_mod, "norm")
        proj_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
        self.proj_func = gpu.get_function(proj_mod, "gptq_matvec")
        activate_mod = gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = gpu.get_function(activate_mod, "activate")

    def _alloc_buffers(self):
        gpu = self.gpu
        self.d_x = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_residual = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_weight = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_gate = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_up = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_act = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_down = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_mlp_in = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_attn_scratch1 = gpu.mem_alloc(12288 * 4)
        self.d_attn_scratch2 = gpu.mem_alloc(6144 * 4)
        self.d_attn_out = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_zero = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.gpu.memcpy_htod(self.d_zero,
                             np.zeros(HIDDEN_DIM, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)

    def _preload_norms(self):
        self.input_norms = []
        self.post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            self.input_norms.append(
                load_norm_weight(self.model, f"{prefix}.input_layernorm.weight") + 1.0)
            self.post_attn_norms.append(
                load_norm_weight(self.model, f"{prefix}.post_attention_layernorm.weight") + 1.0)
        self.final_norm_w = load_norm_weight(self.model, "model.language_model.norm.weight") + 1.0

    def _preload_deltanet_weights(self):
        model = self.model
        self.dn_conv_w = {}
        self.dn_a_weight = {}
        self.dn_b_weight = {}
        self.dn_dt_bias = {}
        self.dn_A_log = {}
        self.dn_head_norm_w = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            prefix = f"model.language_model.layers.{i}.linear_attn"
            conv_w_raw = bytes(model.weight_bytes(f"{prefix}.conv1d.weight"))
            self.dn_conv_w[i] = bf16_to_f32(conv_w_raw).reshape(10240, 1, CONV_KERNEL_SIZE)
            a_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_a.weight"))
            self.dn_a_weight[i] = bf16_to_f32(a_w_raw).reshape(NUM_VALUE_HEADS, HIDDEN_DIM)
            b_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_b.weight"))
            self.dn_b_weight[i] = bf16_to_f32(b_w_raw).reshape(NUM_VALUE_HEADS, HIDDEN_DIM)
            dt_bias_raw = bytes(model.weight_bytes(f"{prefix}.dt_bias"))
            self.dn_dt_bias[i] = bf16_to_f32(dt_bias_raw)
            a_log_raw = bytes(model.weight_bytes(f"{prefix}.A_log"))
            self.dn_A_log[i] = np.frombuffer(a_log_raw, dtype=np.float32).copy()
            norm_w_raw = bytes(model.weight_bytes(f"{prefix}.norm.weight"))
            self.dn_head_norm_w[i] = np.frombuffer(norm_w_raw, dtype=np.float32).copy()

    def _init_deltanet_state(self):
        self.dn_S = {}
        self.dn_conv_buf = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            self.dn_S[i] = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=np.float32)
            self.dn_conv_buf[i] = np.zeros((10240, CONV_KERNEL_SIZE - 1), dtype=np.float32)

    def _preload_qk_norms(self):
        self.q_norm_weights = {}
        self.k_norm_weights = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "full_attention":
                continue
            prefix = f"model.language_model.layers.{i}.self_attn"
            self.q_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.q_norm.weight") + 1.0
            self.k_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.k_norm.weight") + 1.0

    # GPU kernel wrappers
    def gpu_embed(self, token_id: int) -> np.ndarray:
        embed_ptr = self.model.weight_info("model.language_model.embed_tokens.weight").ptr
        BLOCK = 256
        GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))
        self.gpu.launch(
            self.embed_func, grid=(GRID, 1, 1), block=(BLOCK, 1, 1),
            args=[
                ctypes.c_uint32(token_id),
                ctypes.c_uint64(embed_ptr),
                ctypes.c_uint64(self.d_x.value),
            ],
        )
        self.gpu.synchronize()
        return download_f32(self.gpu, self.d_x, HIDDEN_DIM)

    def gpu_norm(self, d_input, d_residual, norm_w, d_output):
        self.gpu.memcpy_htod(self.d_norm_weight,
                             norm_w.ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)
        self.gpu.launch(
            self.norm_func, grid=(1, 1, 1), block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_residual.value),
                ctypes.c_uint64(self.d_norm_weight.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_float(self.epsilon),
            ],
            shared_mem=128,
        )

    def gpu_gptq_matvec(self, prefix, d_input, d_output, K, N):
        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr
        self.gpu.memcpy_htod(d_output,
                             np.zeros(N, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             N * 4)
        self.gpu.launch(
            self.proj_func, grid=(N, 1, 1), block=(256, 1, 1),
            args=[
                ctypes.c_uint64(qw_ptr),
                ctypes.c_uint64(sc_ptr),
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_uint32(N),
                ctypes.c_uint32(K),
            ],
            shared_mem=32,
        )

    def gpu_activate(self, d_gate, d_up, d_output, size):
        GRID_ACT = max(1, math.ceil(size / 256))
        self.gpu.launch(
            self.activate_func, grid=(GRID_ACT, 1, 1), block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_up.value),
                ctypes.c_uint64(d_gate.value),
                ctypes.c_uint64(d_output.value),
            ],
        )

    def run_mlp(self, layer_idx: int, d_residual):
        prefix = f"model.language_model.layers.{layer_idx}"
        self.gpu_norm(d_residual, self.d_zero,
                      self.post_attn_norms[layer_idx], self.d_norm_out)
        self.gpu_gptq_matvec(f"{prefix}.mlp.gate_proj",
                              self.d_norm_out, self.d_gate,
                              HIDDEN_DIM, INTERMEDIATE_SIZE)
        self.gpu_gptq_matvec(f"{prefix}.mlp.up_proj",
                              self.d_norm_out, self.d_up,
                              HIDDEN_DIM, INTERMEDIATE_SIZE)
        self.gpu_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)
        self.gpu_gptq_matvec(f"{prefix}.mlp.down_proj",
                              self.d_act, self.d_down,
                              INTERMEDIATE_SIZE, HIDDEN_DIM)
        self.gpu.synchronize()

    def run_full_attention(self, layer_idx: int, d_normed, position: int):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        self.gpu_gptq_matvec(f"{prefix}.q_proj", d_normed, self.d_attn_scratch1, HIDDEN_DIM, 6144)
        self.gpu.synchronize()
        q_raw = download_f32(self.gpu, self.d_attn_scratch1, 6144)
        self.gpu_gptq_matvec(f"{prefix}.k_proj", d_normed, self.d_attn_scratch1, HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        k_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)
        self.gpu_gptq_matvec(f"{prefix}.v_proj", d_normed, self.d_attn_scratch1, HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        v_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)

        q_heads = q_raw.reshape(NUM_Q_HEADS, HEAD_DIM)
        k_heads = k_raw.reshape(NUM_KV_HEADS, HEAD_DIM)
        q_norm_w = self.q_norm_weights[layer_idx]
        k_norm_w = self.k_norm_weights[layer_idx]
        for h in range(NUM_Q_HEADS):
            rms = np.sqrt(np.mean(q_heads[h] ** 2) + self.epsilon)
            q_heads[h] = (q_heads[h] / rms) * q_norm_w
        for h in range(NUM_KV_HEADS):
            rms = np.sqrt(np.mean(k_heads[h] ** 2) + self.epsilon)
            k_heads[h] = (k_heads[h] / rms) * k_norm_w
        q_raw = q_heads.flatten()
        k_raw = k_heads.flatten()

        q_rope, k_rope = process_qkv_with_rope(q_raw, k_raw, position)

        v_2d = v_raw.reshape(NUM_KV_HEADS, HEAD_DIM)
        self.kv_cache.store(layer_idx, position, k_rope, v_2d)
        k_cache, v_cache = self.kv_cache.get(layer_idx, position)
        attended = attention_prefill(q_rope, k_cache, v_cache, position)

        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attended.ctypes.data_as(ctypes.c_void_p), 6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.o_proj", self.d_attn_scratch2, self.d_attn_out, 6144, HIDDEN_DIM)
        self.gpu.synchronize()

    def run_deltanet(self, layer_idx: int, d_normed):
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        self.gpu_gptq_matvec(f"{prefix}.in_proj_qkv", d_normed, self.d_attn_scratch1, HIDDEN_DIM, 10240)
        self.gpu_gptq_matvec(f"{prefix}.in_proj_z", d_normed, self.d_attn_scratch2, HIDDEN_DIM, 6144)
        self.gpu.synchronize()

        qkv = download_f32(self.gpu, self.d_attn_scratch1, 10240)
        z = download_f32(self.gpu, self.d_attn_scratch2, 6144)
        normed_cpu = download_f32(self.gpu, d_normed, HIDDEN_DIM)

        conv_buf = self.dn_conv_buf[layer_idx]
        conv_w = self.dn_conv_w[layer_idx]
        conv_window = np.concatenate([conv_buf, qkv[:, np.newaxis]], axis=1)
        qkv_conv = np.sum(conv_window * conv_w[:, 0, :], axis=1)
        self.dn_conv_buf[layer_idx] = conv_window[:, 1:]
        qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv.clip(-80, 80))))

        q = qkv_conv[:2048].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        k = qkv_conv[2048:4096].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        v = qkv_conv[4096:].reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)

        for hi in range(NUM_KEY_HEADS):
            q_norm = np.sqrt(np.sum(q[hi] ** 2) + 1e-6)
            q[hi] = q[hi] / q_norm
            k_norm = np.sqrt(np.sum(k[hi] ** 2) + 1e-6)
            k[hi] = k[hi] / k_norm
        q = q * (1.0 / np.sqrt(float(KEY_HEAD_DIM)))

        beta = 1.0 / (1.0 + np.exp(-(self.dn_b_weight[layer_idx] @ normed_cpu).clip(-80, 80)))
        dt = self.dn_a_weight[layer_idx] @ normed_cpu + self.dn_dt_bias[layer_idx]
        dt = np.log1p(np.exp(dt.clip(-20, 20)))
        A_exp = np.exp(self.dn_A_log[layer_idx])
        decay = np.exp(-A_exp * dt)

        S = self.dn_S[layer_idx]
        output_heads = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM), dtype=np.float32)
        for h in range(NUM_VALUE_HEADS):
            key_idx = h // HEADS_PER_KEY
            q_h = q[key_idx]
            k_h = k[key_idx]
            v_h = v[h]
            S[h] *= decay[h]
            kv_mem = S[h] @ k_h
            delta = (v_h - kv_mem) * beta[h]
            S[h] += np.outer(delta, k_h)
            output_heads[h] = S[h] @ q_h

        self.dn_S[layer_idx] = S

        head_norm_w = self.dn_head_norm_w[layer_idx]
        for h in range(NUM_VALUE_HEADS):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w

        z_heads = z.reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu

        attn_output_flat = output_heads.flatten()
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output_flat.ctypes.data_as(ctypes.c_void_p), 6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.out_proj", self.d_attn_scratch2, self.d_attn_out, 6144, HIDDEN_DIM)
        self.gpu.synchronize()

    def run_layer(self, layer_idx: int, x: np.ndarray, position: int) -> np.ndarray:
        layer_type = self.layer_types[layer_idx]
        gpu = self.gpu
        residual = x.copy()

        d_x_gpu = upload_f32(gpu, x)
        self.gpu_norm(d_x_gpu, self.d_zero,
                      self.input_norms[layer_idx], self.d_norm_out)
        gpu.synchronize()

        if layer_type == "full_attention":
            self.run_full_attention(layer_idx, self.d_norm_out, position)
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
        else:
            self.run_deltanet(layer_idx, self.d_norm_out)
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
        x = residual + attn_out
        residual = x.copy()
        gpu.mem_free(d_x_gpu)

        gpu.memcpy_htod(self.d_residual,
                        x.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)
        self.run_mlp(layer_idx, self.d_residual)
        mlp_out = download_f32(gpu, self.d_down, HIDDEN_DIM)
        x = x + mlp_out

        return x

    def process_prompt(self, prompt: str) -> list:
        """Process all prompt tokens and return hidden states after each layer of the last token."""
        token_ids = self.tok.encode(prompt)
        print(f"  Lithos: tokens = {token_ids}")

        x = None
        for pos, tid in enumerate(token_ids):
            x = self.gpu_embed(tid)

            if pos < len(token_ids) - 1:
                for layer_idx in range(NUM_LAYERS):
                    x = self.run_layer(layer_idx, x, pos)
                print(f"  Lithos: token {pos} done, norm={np.linalg.norm(x):.4f}")
            else:
                hidden_states = [x.copy()]
                for layer_idx in range(NUM_LAYERS):
                    x = self.run_layer(layer_idx, x, pos)
                    hidden_states.append(x.copy())
                print(f"  Lithos: last token done, norm={np.linalg.norm(x):.4f}")

        # Final norm
        x_normed = np.sqrt(np.mean(x ** 2) + self.epsilon)
        x_final = (x / x_normed) * self.final_norm_w

        return hidden_states, x_final

    def close(self):
        self.gpu.close()
        self.model.close()


# ============================================================================
# Detailed sub-operation comparison for a single layer
# ============================================================================

def compare_layer_ops(lithos: LithosEngine, ref: ReferenceModel,
                      layer_idx: int, x_lithos: np.ndarray, x_ref: np.ndarray,
                      position: int):
    """Compare sub-operations within a single layer to find the exact divergence point."""
    layer_type = lithos.layer_types[layer_idx]
    print(f"\n  === Detailed comparison of layer {layer_idx} ({layer_type}) ===")

    # Step 1: Input norm
    # Reference
    ref_normed = ref.rms_norm(x_ref, ref.input_norms[layer_idx])
    # Lithos
    gpu = lithos.gpu
    d_x_gpu = upload_f32(gpu, x_lithos)
    lithos.gpu_norm(d_x_gpu, lithos.d_zero,
                    lithos.input_norms[layer_idx], lithos.d_norm_out)
    gpu.synchronize()
    lithos_normed = download_f32(gpu, lithos.d_norm_out, HIDDEN_DIM)

    cs_norm = cosine_similarity(lithos_normed, ref_normed)
    mad_norm = max_abs_diff(lithos_normed, ref_normed)
    print(f"    After RMSNorm:    cos={cs_norm:.8f}  max_diff={mad_norm:.6e}")

    if layer_type == "linear_attention":
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        # QKV projection comparison
        ref_qkv = ref.matvec(f"{prefix}.in_proj_qkv", ref_normed)
        lithos.gpu_gptq_matvec(f"{prefix}.in_proj_qkv",
                                lithos.d_norm_out, lithos.d_attn_scratch1,
                                HIDDEN_DIM, 10240)
        gpu.synchronize()
        lithos_qkv = download_f32(gpu, lithos.d_attn_scratch1, 10240)

        cs_qkv = cosine_similarity(lithos_qkv, ref_qkv)
        mad_qkv = max_abs_diff(lithos_qkv, ref_qkv)
        print(f"    QKV projection:   cos={cs_qkv:.8f}  max_diff={mad_qkv:.6e}")

        # Z projection comparison
        ref_z = ref.matvec(f"{prefix}.in_proj_z", ref_normed)
        lithos.gpu_gptq_matvec(f"{prefix}.in_proj_z",
                                lithos.d_norm_out, lithos.d_attn_scratch2,
                                HIDDEN_DIM, 6144)
        gpu.synchronize()
        lithos_z = download_f32(gpu, lithos.d_attn_scratch2, 6144)

        cs_z = cosine_similarity(lithos_z, ref_z)
        mad_z = max_abs_diff(lithos_z, ref_z)
        print(f"    Z projection:     cos={cs_z:.8f}  max_diff={mad_z:.6e}")

        if cs_qkv < 0.999:
            print(f"    ** GPTQ dequant mismatch in QKV projection **")
            # Compare first few values
            print(f"       Lithos QKV[:8] = {lithos_qkv[:8]}")
            print(f"       Ref   QKV[:8] = {ref_qkv[:8]}")

    elif layer_type == "full_attention":
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        # Q projection
        ref_q = ref.matvec(f"{prefix}.q_proj", ref_normed)
        lithos.gpu_gptq_matvec(f"{prefix}.q_proj",
                                lithos.d_norm_out, lithos.d_attn_scratch1,
                                HIDDEN_DIM, 6144)
        gpu.synchronize()
        lithos_q = download_f32(gpu, lithos.d_attn_scratch1, 6144)

        cs_q = cosine_similarity(lithos_q, ref_q)
        mad_q = max_abs_diff(lithos_q, ref_q)
        print(f"    Q projection:     cos={cs_q:.8f}  max_diff={mad_q:.6e}")

        # K projection
        ref_k = ref.matvec(f"{prefix}.k_proj", ref_normed)
        lithos.gpu_gptq_matvec(f"{prefix}.k_proj",
                                lithos.d_norm_out, lithos.d_attn_scratch1,
                                HIDDEN_DIM, 1024)
        gpu.synchronize()
        lithos_k = download_f32(gpu, lithos.d_attn_scratch1, 1024)

        cs_k = cosine_similarity(lithos_k, ref_k)
        mad_k = max_abs_diff(lithos_k, ref_k)
        print(f"    K projection:     cos={cs_k:.8f}  max_diff={mad_k:.6e}")

        if cs_q < 0.999:
            print(f"    ** GPTQ dequant mismatch in Q projection **")
            print(f"       Lithos Q[:8] = {lithos_q[:8]}")
            print(f"       Ref   Q[:8] = {ref_q[:8]}")

    # MLP comparison (using the post-attention hidden state)
    # We skip this for now -- the attention sub-ops above are the main suspects

    gpu.mem_free(d_x_gpu)
    return cs_norm


# ============================================================================
# Main comparison
# ============================================================================

def main():
    prompt = "The capital of France is"

    banner("Loading Reference Model (CPU dequantized)")
    t0 = time.monotonic()
    ref = ReferenceModel(MODEL_DIR)
    t_ref_load = time.monotonic() - t0
    print(f"  Reference model loaded in {t_ref_load:.1f}s")

    banner("Loading Lithos Engine (GPU)")
    t0 = time.monotonic()
    lithos = LithosEngine()
    t_lithos_load = time.monotonic() - t0
    print(f"  Lithos engine loaded in {t_lithos_load:.1f}s")

    # Verify same tokenization
    tok = Tokenizer(MODEL_DIR)
    token_ids = tok.encode(prompt)
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {token_ids}")
    for i, tid in enumerate(token_ids):
        print(f"  [{i}] id={tid} '{tok.decode([tid])}'")

    # ---- Process all tokens except the last through both engines ----
    banner("Processing prefix tokens (building DeltaNet state)")
    for pos in range(len(token_ids) - 1):
        tid = token_ids[pos]
        word = tok.decode([tid])
        print(f"\n  Token {pos}: '{word}' (id={tid})")

        # Reference
        x_ref = ref.embed(tid)
        for layer_idx in range(NUM_LAYERS):
            x_ref = ref.run_layer(layer_idx, x_ref, pos)
        print(f"    Ref  norm: {np.linalg.norm(x_ref):.4f}")

        # Lithos
        x_lit = lithos.gpu_embed(tid)
        for layer_idx in range(NUM_LAYERS):
            x_lit = lithos.run_layer(layer_idx, x_lit, pos)
        print(f"    Lit  norm: {np.linalg.norm(x_lit):.4f}")

        cs = cosine_similarity(x_lit, x_ref)
        print(f"    Cosine sim after all layers: {cs:.8f}")

    # ---- Process the LAST token with per-layer comparison ----
    last_pos = len(token_ids) - 1
    last_tid = token_ids[last_pos]
    last_word = tok.decode([last_tid])

    banner(f"LAYER-BY-LAYER COMPARISON: token {last_pos} = '{last_word}' (id={last_tid})")

    # Embeddings
    x_ref = ref.embed(last_tid)
    x_lit = lithos.gpu_embed(last_tid)

    cs_embed = cosine_similarity(x_lit, x_ref)
    mad_embed = max_abs_diff(x_lit, x_ref)
    print(f"  Embedding: cos={cs_embed:.8f}  max_diff={mad_embed:.6e}")

    # Track where divergence first occurs
    first_diverge_layer = None
    results = []

    for layer_idx in range(NUM_LAYERS):
        layer_type = lithos.layer_types[layer_idx]

        # Run layer on both
        x_ref_next = ref.run_layer(layer_idx, x_ref.copy(), last_pos)
        x_lit_next = lithos.run_layer(layer_idx, x_lit.copy(), last_pos)

        cs = cosine_similarity(x_lit_next, x_ref_next)
        mad = max_abs_diff(x_lit_next, x_ref_next)

        status = "OK" if cs >= 0.99 else "DIVERGED"
        if cs < 0.999:
            status = "WARN" if cs >= 0.99 else "DIVERGED"

        layer_label = "DeltaNet" if layer_type == "linear_attention" else "FullAttn"
        print(f"  Layer {layer_idx:2d} [{layer_label:8s}]: cos={cs:.8f}  max_diff={mad:.6e}  {status}")

        results.append({
            "layer": layer_idx,
            "type": layer_label,
            "cosine": cs,
            "max_diff": mad,
            "status": status,
        })

        if cs < 0.99 and first_diverge_layer is None:
            first_diverge_layer = layer_idx
            # Do detailed sub-operation comparison
            compare_layer_ops(lithos, ref, layer_idx, x_lit, x_ref, last_pos)

        x_ref = x_ref_next
        x_lit = x_lit_next

    # ---- Final norm + logits comparison ----
    banner("FINAL NORM + LOGITS")

    # Reference final norm
    ref_final = ref.rms_norm(x_ref, ref.final_norm_w)
    # Lithos final norm
    lit_final_rms = np.sqrt(np.mean(x_lit ** 2) + EPSILON)
    lit_final = (x_lit / lit_final_rms) * lithos.final_norm_w

    cs_final = cosine_similarity(lit_final, ref_final)
    print(f"  Final norm: cos={cs_final:.8f}")

    # Logits (both on CPU, using dequantized lm_head for reference)
    lm_head_raw = bytes(lithos.model.weight_bytes("lm_head.weight"))
    vocab_size = 248320
    CHUNK = 8192

    ref_logits = np.zeros(vocab_size, dtype=np.float32)
    lit_logits = np.zeros(vocab_size, dtype=np.float32)

    for start in range(0, vocab_size, CHUNK):
        end = min(start + CHUNK, vocab_size)
        chunk_size = end - start
        offset = start * HIDDEN_DIM * 2
        chunk_bytes = chunk_size * HIDDEN_DIM * 2
        w_chunk = np.frombuffer(
            lm_head_raw[offset:offset + chunk_bytes],
            dtype=np.float16
        ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
        ref_logits[start:end] = w_chunk @ ref_final
        lit_logits[start:end] = w_chunk @ lit_final

    # Compare top predictions
    ref_top = int(np.argmax(ref_logits))
    lit_top = int(np.argmax(lit_logits))
    print(f"\n  Reference top-1: id={ref_top} '{tok.decode([ref_top])}'")
    print(f"  Lithos    top-1: id={lit_top} '{tok.decode([lit_top])}'")

    print(f"\n  Reference top-10:")
    for rank, idx in enumerate(np.argsort(ref_logits)[-10:][::-1]):
        print(f"    {rank+1}. id={idx:6d} logit={ref_logits[idx]:8.3f} '{tok.decode([int(idx)])}'")

    print(f"\n  Lithos top-10:")
    for rank, idx in enumerate(np.argsort(lit_logits)[-10:][::-1]):
        print(f"    {rank+1}. id={idx:6d} logit={lit_logits[idx]:8.3f} '{tok.decode([int(idx)])}'")

    # Paris ranking
    for candidate in ["Paris", " Paris"]:
        cand_ids = tok.encode(candidate)
        for cid in cand_ids:
            ref_rank = int(np.sum(ref_logits > ref_logits[cid])) + 1
            lit_rank = int(np.sum(lit_logits > lit_logits[cid])) + 1
            print(f"\n  '{candidate}' (id={cid}):")
            print(f"    Reference: rank={ref_rank}  logit={ref_logits[cid]:.3f}")
            print(f"    Lithos:    rank={lit_rank}  logit={lit_logits[cid]:.3f}")

    # ---- Summary ----
    banner("SUMMARY")
    if first_diverge_layer is not None:
        r = results[first_diverge_layer]
        print(f"  First divergence at layer {first_diverge_layer} ({r['type']})")
        print(f"  Cosine similarity: {r['cosine']:.8f}")
        print(f"  Max absolute diff: {r['max_diff']:.6e}")
        if r['type'] == 'DeltaNet':
            print(f"  This is a DeltaNet (linear_attention) layer.")
            print(f"  Check: GPTQ dequant, conv1d, DeltaNet recurrence, or group norm.")
        else:
            print(f"  This is a full attention layer.")
            print(f"  Check: GPTQ dequant, QK norm, RoPE, or attention computation.")
    else:
        # Find lowest cosine similarity
        worst = min(results, key=lambda r: r["cosine"])
        print(f"  No layer dropped below cos=0.99 (threshold)")
        print(f"  Worst layer: {worst['layer']} ({worst['type']}) cos={worst['cosine']:.8f}")
        if cs_final < 0.99:
            print(f"  But final norm diverged: cos={cs_final:.8f}")
            print(f"  This suggests accumulated small errors across many layers.")

    print(f"\n  Embedding cos: {cs_embed:.8f}")
    print(f"  Final norm cos: {cs_final:.8f}")
    print(f"  Reference predicts: '{tok.decode([ref_top])}'")
    print(f"  Lithos predicts:    '{tok.decode([lit_top])}'")

    lithos.close()
    ref.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
