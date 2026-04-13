#!/usr/bin/env python3
"""
Shannon SMRS Inference for Qwen 3.5-27B.

Proves that Shannon Mixed-Radix Scheme quantization (3.076 bpw) produces
correct inference output. Uses the existing GPTQ model as source, dequantizes
to F32, requantizes with Shannon SMRS, and runs full inference on CPU+GPU.

Strategy:
  - GPTQ-quantized weights: dequantize -> F32 -> Shannon requantize on-the-fly
  - Small weights (norms, biases, conv1d, etc.): kept as F32 (same as generate_paris.py)
  - Matmul: Shannon dequantize to F32 on CPU, then matvec (CPU numpy)
  - Embedding + lm_head: kept on GPU (FP16), same as original

Run:
    python3 /home/ubuntu/lithos/src/shannon_inference.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import struct
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantization"))

from cuda_driver import CUDADriver, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer
from kv_cache import KVCache
from attention import process_qkv_with_rope, attention_prefill
from shannon_smrs import (
    quantize as shannon_quantize,
    dequantize as shannon_dequantize,
    bits_per_weight as shannon_bpw,
    layer_size_bytes as shannon_size_bytes,
    CODEBOOK_6, CODEBOOK_8, PACK_WIDTH,
    _unpack_6level, _unpack_8level,
)

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
GPTQ_ZERO_POINT = 8
NUM_LAYERS = 64

# DeltaNet dimensions
NUM_KEY_HEADS = 16
KEY_HEAD_DIM = 128
NUM_VALUE_HEADS = 48
VALUE_HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HEADS_PER_KEY = NUM_VALUE_HEADS // NUM_KEY_HEADS  # 3

SHANNON_WEIGHTS_DIR = "/tmp/shannon_weights"


# ============================================================================
# GPTQ Dequantization (from requantize_model.py)
# ============================================================================

def dequantize_gptq_matrix(model: LithosModel, weight_prefix: str) -> np.ndarray:
    """Dequantize a GPTQ 4-bit weight matrix to F32. Returns shape [K, N]."""
    qw_name = f"{weight_prefix}.qweight"
    sc_name = f"{weight_prefix}.scales"
    zr_name = f"{weight_prefix}.qzeros"

    qw_info = model.weight_info(qw_name)
    sc_info = model.weight_info(sc_name)
    zr_info = model.weight_info(zr_name)

    qw_raw = bytes(model.weight_bytes(qw_name))
    sc_raw = bytes(model.weight_bytes(sc_name))
    zr_raw = bytes(model.weight_bytes(zr_name))

    qw_shape = qw_info.shape
    sc_shape = sc_info.shape
    zr_shape = zr_info.shape

    K = qw_shape[0] * 8
    N = qw_shape[1]
    n_groups = sc_shape[0]

    qweight = np.frombuffer(qw_raw, dtype=np.int32).reshape(qw_shape).copy()
    scales = np.frombuffer(sc_raw, dtype=np.float16).reshape(sc_shape).astype(np.float32)
    qzeros = np.frombuffer(zr_raw, dtype=np.int32).reshape(zr_shape).copy()

    result = np.zeros((K, N), dtype=np.float32)

    for g in range(n_groups):
        k_start = g * GROUP_SIZE
        k_end = min(k_start + GROUP_SIZE, K)
        scale_row = scales[g]

        zeros = np.zeros(N, dtype=np.float32)
        for j_pack in range(zr_shape[1]):
            zr_val = qzeros[g, j_pack]
            for bit in range(8):
                j = j_pack * 8 + bit
                if j < N:
                    zeros[j] = float((zr_val >> (bit * 4)) & 0xF)

        for k in range(k_start, k_end):
            pack_row = k // 8
            bit_offset = (k % 8) * 4
            qvals = ((qweight[pack_row, :].astype(np.uint32) >> bit_offset) & 0xF).astype(np.float32)
            result[k, :] = (qvals - zeros) * scale_row

    return result


# ============================================================================
# Shannon-aware Matvec: dequantize column groups on-the-fly and accumulate
# ============================================================================

def shannon_matvec(packed: dict, x: np.ndarray) -> np.ndarray:
    """Compute y = W @ x where W is stored in Shannon SMRS format.

    W has shape [K, N] (original_shape), x has shape [K].
    Returns y with shape [N].

    This does full dequantization then matvec. For a proof-of-concept,
    this is fine; a production kernel would fuse the dequant into the matvec.
    """
    W = shannon_dequantize(packed)
    # W shape is original_shape = [K, N]
    # y[n] = sum_k W[k, n] * x[k]  =>  y = x @ W
    return x @ W


# ============================================================================
# Utility functions (same as generate_paris.py)
# ============================================================================

def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


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


def rms_norm_cpu(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def silu_cpu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x.clip(-80, 80))))


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


# ============================================================================
# Shannon Prefill Engine
# ============================================================================

class ShannonPrefillEngine:
    """Processes prompt tokens through the full model using Shannon SMRS weights.

    All linear projections (GPTQ in the original) are replaced with:
      GPTQ -> F32 -> Shannon SMRS quantize -> Shannon dequantize -> CPU matvec

    This proves Shannon quantization works for real inference.
    """

    def __init__(self):
        banner("Initializing Shannon SMRS Prefill Engine")
        t0 = time.monotonic()

        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self.layer_types = self.model.config.layer_types

        print(f"  Model: {self.model}")
        print(f"  Device: {self.gpu.device_name}")
        print(f"  Layers: {NUM_LAYERS}")
        print(f"  Quantization: Shannon SMRS (~3.076 bpw)")

        self._load_gpu_kernels()
        self._alloc_gpu_buffers()
        self._preload_norms()
        self._preload_deltanet_weights()
        self._init_deltanet_state()
        self.kv_cache = KVCache(max_seq_len=64)
        self._preload_qk_norms()

        # Shannon weight cache: populated on first use per layer
        self._shannon_cache = {}  # (layer_idx, weight_name) -> packed dict
        self._shannon_stats = {"total_bytes": 0, "total_weights": 0, "layers_done": set()}

        t1 = time.monotonic()
        print(f"  Init time: {t1-t0:.3f}s")

    def _load_gpu_kernels(self):
        """Load only embed and lm_head kernels (projections are done on CPU with Shannon)."""
        gpu = self.gpu
        embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = gpu.get_function(embed_mod, "embed_f16")

        lm_head_mod = gpu.load_cubin(f"{KERNEL_DIR}/lm_head.cubin")
        self.lm_head_func = gpu.get_function(lm_head_mod, "lm_head")
        print("    GPU kernels: embed_f16, lm_head loaded")

    def _alloc_gpu_buffers(self):
        gpu = self.gpu
        self.d_x = gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_lm_head_out = gpu.mem_alloc(248320 * 4)
        print("    GPU buffers allocated (minimal -- projections on CPU)")

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
        print(f"    {NUM_LAYERS * 2 + 1} norm weights preloaded")

    def _preload_deltanet_weights(self):
        """Preload DeltaNet CPU-side weights (small, kept as FP32)."""
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

        print(f"    DeltaNet auxiliary weights preloaded for {len(self.dn_conv_w)} layers")

    def _init_deltanet_state(self):
        self.dn_S = {}
        self.dn_conv_buf = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            self.dn_S[i] = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=np.float32)
            self.dn_conv_buf[i] = np.zeros((10240, CONV_KERNEL_SIZE - 1), dtype=np.float32)
        print(f"    DeltaNet state initialized ({len(self.dn_S)} layers)")

    def _preload_qk_norms(self):
        self.q_norm_weights = {}
        self.k_norm_weights = {}
        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "full_attention":
                continue
            prefix = f"model.language_model.layers.{i}.self_attn"
            self.q_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.q_norm.weight")
            self.k_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.k_norm.weight")
        print(f"    Q/K norm weights preloaded for {len(self.q_norm_weights)} full attention layers")

    # ------------------------------------------------------------------
    # Shannon-quantized matvec
    # ------------------------------------------------------------------

    def _get_shannon_weight(self, weight_prefix: str) -> dict:
        """Get or create Shannon-packed weight. Caches after first use."""
        if weight_prefix in self._shannon_cache:
            return self._shannon_cache[weight_prefix]

        # Check if we have it saved on disk
        # For now, always requantize on-the-fly from GPTQ
        t0 = time.monotonic()
        f32 = dequantize_gptq_matrix(self.model, weight_prefix)
        packed = shannon_quantize(f32, group_size=128)

        bpw = shannon_bpw(packed)
        size_b = shannon_size_bytes(packed)
        self._shannon_stats["total_bytes"] += size_b
        self._shannon_stats["total_weights"] += packed['n_weights']

        elapsed = time.monotonic() - t0
        self._shannon_cache[weight_prefix] = packed
        return packed

    def shannon_matvec(self, weight_prefix: str, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W where W is Shannon-quantized.

        weight_prefix: e.g. "model.language_model.layers.0.mlp.gate_proj"
        x: input vector [K]
        Returns: output vector [N]
        """
        packed = self._get_shannon_weight(weight_prefix)
        W = shannon_dequantize(packed)  # [K, N]
        return x @ W

    # ------------------------------------------------------------------
    # GPU helpers (embed + lm_head only)
    # ------------------------------------------------------------------

    def gpu_embed(self, token_id: int) -> np.ndarray:
        embed_ptr = self.model.weight_info("model.language_model.embed_tokens.weight").ptr
        BLOCK = 256
        GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))
        self.gpu.launch(
            self.embed_func,
            grid=(GRID, 1, 1),
            block=(BLOCK, 1, 1),
            args=[
                ctypes.c_uint32(token_id),
                ctypes.c_uint64(embed_ptr),
                ctypes.c_uint64(self.d_x.value),
            ],
        )
        self.gpu.synchronize()
        return download_f32(self.gpu, self.d_x, HIDDEN_DIM)

    # ------------------------------------------------------------------
    # Layer execution (all projections via Shannon CPU matvec)
    # ------------------------------------------------------------------

    def run_mlp(self, layer_idx: int, x_normed: np.ndarray) -> np.ndarray:
        """MLP sublayer with Shannon-quantized gate/up/down projections."""
        prefix = f"model.language_model.layers.{layer_idx}"

        # Gate and up projections: [5120] -> [17408]
        gate = self.shannon_matvec(f"{prefix}.mlp.gate_proj", x_normed)
        up = self.shannon_matvec(f"{prefix}.mlp.up_proj", x_normed)

        # SiLU(gate) * up
        act = silu_cpu(gate) * up

        # Down projection: [17408] -> [5120]
        down = self.shannon_matvec(f"{prefix}.mlp.down_proj", act)
        return down

    def run_full_attention(self, layer_idx: int, x_normed: np.ndarray, position: int) -> np.ndarray:
        """Full attention with Shannon-quantized Q/K/V/O projections."""
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        # Q: [5120] -> [12288]
        q_gate_raw = self.shannon_matvec(f"{prefix}.q_proj", x_normed)

        # K: [5120] -> [1024]
        k_raw = self.shannon_matvec(f"{prefix}.k_proj", x_normed)

        # V: [5120] -> [1024]
        v_raw = self.shannon_matvec(f"{prefix}.v_proj", x_normed)

        # Split Q into Q and gate
        q_gate_heads = q_gate_raw.reshape(24, 512)
        q_heads = q_gate_heads[:, :256].copy()
        gate_heads = q_gate_heads[:, 256:].copy()
        gate_flat = gate_heads.flatten()

        # QK RMSNorm
        k_heads = k_raw.reshape(4, 256)
        q_norm_w = self.q_norm_weights[layer_idx]
        k_norm_w = self.k_norm_weights[layer_idx]
        for h in range(24):
            rms = np.sqrt(np.mean(q_heads[h] ** 2) + self.epsilon)
            q_heads[h] = (q_heads[h] / rms) * q_norm_w
        for h in range(4):
            rms = np.sqrt(np.mean(k_heads[h] ** 2) + self.epsilon)
            k_heads[h] = (k_heads[h] / rms) * k_norm_w

        q_raw = q_heads.flatten()
        k_raw = k_heads.flatten()

        # RoPE
        q_rope, k_rope = process_qkv_with_rope(q_raw, k_raw, position)

        # KV cache
        v_2d = v_raw.reshape(4, 256)
        self.kv_cache.store(layer_idx, position, k_rope, v_2d)
        k_cache, v_cache = self.kv_cache.get(layer_idx, position)

        # Attention
        attended = attention_prefill(q_rope, k_cache, v_cache, position)

        # Gate
        gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_flat.clip(-80, 80)))
        attended = attended * gate_sigmoid

        # O projection: [6144] -> [5120]
        attn_out = self.shannon_matvec(f"{prefix}.o_proj", attended)
        return attn_out

    def run_deltanet(self, layer_idx: int, x_normed: np.ndarray) -> np.ndarray:
        """DeltaNet with Shannon-quantized QKV/Z/out projections."""
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        # QKV projection: [5120] -> [10240]
        qkv = self.shannon_matvec(f"{prefix}.in_proj_qkv", x_normed)

        # Z projection: [5120] -> [6144]
        z = self.shannon_matvec(f"{prefix}.in_proj_z", x_normed)

        # Conv1d with history buffer
        conv_buf = self.dn_conv_buf[layer_idx]
        conv_w = self.dn_conv_w[layer_idx]
        conv_window = np.concatenate([conv_buf, qkv[:, np.newaxis]], axis=1)
        qkv_conv = np.sum(conv_window * conv_w[:, 0, :], axis=1)
        self.dn_conv_buf[layer_idx] = conv_window[:, 1:]

        # SiLU
        qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv.clip(-80, 80))))

        # Split QKV
        q = qkv_conv[:2048].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        k = qkv_conv[2048:4096].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        v = qkv_conv[4096:].reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)

        # Q, K L2 norm + Q scaling
        for hi in range(NUM_KEY_HEADS):
            q_norm = np.sqrt(np.sum(q[hi] ** 2) + 1e-6)
            q[hi] = q[hi] / q_norm
            k_norm = np.sqrt(np.sum(k[hi] ** 2) + 1e-6)
            k[hi] = k[hi] / k_norm
        q = q * (1.0 / np.sqrt(float(KEY_HEAD_DIM)))

        # Beta and dt
        beta = 1.0 / (1.0 + np.exp(-(self.dn_b_weight[layer_idx] @ x_normed).clip(-80, 80)))
        dt = self.dn_a_weight[layer_idx] @ x_normed + self.dn_dt_bias[layer_idx]
        dt = np.log1p(np.exp(dt.clip(-20, 20)))

        A_exp = np.exp(self.dn_A_log[layer_idx])
        decay = np.exp(-A_exp * dt)

        # DeltaNet recurrence
        S = self.dn_S[layer_idx]
        k_exp = np.repeat(k, HEADS_PER_KEY, axis=0)
        q_exp = np.repeat(q, HEADS_PER_KEY, axis=0)

        S *= decay[:, None, None]
        kv_mem = np.einsum('hvk,hk->hv', S, k_exp)
        delta = (v - kv_mem) * beta[:, None]
        S += np.einsum('hv,hk->hvk', delta, k_exp)
        output_heads = np.einsum('hvk,hk->hv', S, q_exp)
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

        # Output projection: [6144] -> [5120]
        attn_output_flat = output_heads.flatten()
        attn_out = self.shannon_matvec(f"{prefix}.out_proj", attn_output_flat)
        return attn_out

    def run_layer(self, layer_idx: int, x: np.ndarray, position: int) -> np.ndarray:
        """Run one transformer layer."""
        layer_type = self.layer_types[layer_idx]

        # Input norm
        x_normed = rms_norm_cpu(x, self.input_norms[layer_idx], self.epsilon)

        # Attention
        if layer_type == "full_attention":
            attn_out = self.run_full_attention(layer_idx, x_normed, position)
        else:
            attn_out = self.run_deltanet(layer_idx, x_normed)

        # Residual
        x = x + attn_out

        # MLP sublayer
        mlp_normed = rms_norm_cpu(x, self.post_attn_norms[layer_idx], self.epsilon)
        mlp_out = self.run_mlp(layer_idx, mlp_normed)
        x = x + mlp_out

        return x

    def process_token(self, token_id: int, token_pos: int) -> np.ndarray:
        """Process a single token through all 64 layers."""
        x = self.gpu_embed(token_id)

        for layer_idx in range(NUM_LAYERS):
            t_layer = time.monotonic()
            x = self.run_layer(layer_idx, x, token_pos)
            elapsed = time.monotonic() - t_layer

            if layer_idx == 0 or (layer_idx + 1) % 16 == 0:
                x_norm = np.linalg.norm(x)
                has_nan = np.any(np.isnan(x))
                has_inf = np.any(np.isinf(x))
                status = ""
                if has_nan or has_inf:
                    status = " *** NaN/Inf DETECTED ***"
                print(f"    Layer {layer_idx:2d}: norm={x_norm:.4f} {elapsed:.2f}s{status}")
                if has_nan or has_inf:
                    return None

            # Track layer stats
            self._shannon_stats["layers_done"].add(layer_idx)

        return x

    def generate(self, prompt: str) -> tuple:
        """Process prompt tokens and generate the next token."""
        token_ids = self.tok.encode(prompt)
        print(f"\nPrompt: {prompt!r}")
        print(f"Token IDs: {token_ids}")
        for i, tid in enumerate(token_ids):
            print(f"  Token {i}: id={tid} '{self.tok.decode([tid])}'")

        banner("PREFILL (Shannon SMRS)")

        x = None
        for pos, tid in enumerate(token_ids):
            word = self.tok.decode([tid])
            t0 = time.monotonic()
            print(f"\n--- Token {pos+1}/{len(token_ids)}: '{word}' (id={tid}) ---")

            x = self.process_token(tid, pos)

            if x is None:
                print("  ERROR: NaN/Inf detected!")
                return -1, "<error>", None

            elapsed = time.monotonic() - t0
            print(f"  Token done: norm={np.linalg.norm(x):.4f} time={elapsed:.1f}s")

        banner("GENERATE NEXT TOKEN (Shannon SMRS)")

        # Final RMSNorm
        x_normed = rms_norm_cpu(x, self.final_norm_w, self.epsilon)
        print(f"  Final norm output: norm={np.linalg.norm(x_normed):.4f}")

        # lm_head: FP16 matmul on GPU (not requantized -- it's already FP16)
        print(f"  Computing lm_head (248320 x 5120 FP16 matmul on GPU)...")
        t_lm = time.monotonic()
        vocab_size = 248320
        lm_weight_ptr = self.model.weight_info("lm_head.weight").ptr
        d_normed_input = upload_f32(self.gpu, x_normed)

        self.gpu.launch(
            self.lm_head_func,
            grid=(62080, 1, 1),
            block=(128, 1, 1),
            args=[
                ctypes.c_uint64(lm_weight_ptr),
                ctypes.c_uint64(d_normed_input.value),
                ctypes.c_uint64(self.d_lm_head_out.value),
            ],
            shared_mem=20544,
        )
        self.gpu.synchronize()
        logits = download_f32(self.gpu, self.d_lm_head_out, vocab_size)
        self.gpu.mem_free(d_normed_input)
        print(f"  lm_head: {time.monotonic() - t_lm:.2f}s")

        # Argmax
        token_out = int(np.argmax(logits))
        token_text = self.tok.decode([token_out])

        # Top-10
        top10_idx = np.argsort(logits)[-10:][::-1]
        print(f"\n  Top-10 predictions:")
        for rank, idx in enumerate(top10_idx):
            word = self.tok.decode([int(idx)])
            print(f"    {rank+1}. token={idx:6d} logit={logits[idx]:8.3f} '{word}'")

        print(f"\n  Generated token: {token_out} = '{token_text}'")

        # Paris check
        for candidate in ["Paris", " Paris"]:
            cand_ids = self.tok.encode(candidate)
            for cid in cand_ids:
                rank = int(np.sum(logits > logits[cid])) + 1
                cword = self.tok.decode([cid])
                print(f"  '{candidate}' (id={cid}, decoded='{cword}'): logit={logits[cid]:.3f}, rank={rank}")

        return token_out, token_text, logits

    def report_stats(self):
        """Print Shannon quantization statistics."""
        banner("SHANNON SMRS STATISTICS")

        total_bytes = self._shannon_stats["total_bytes"]
        total_weights = self._shannon_stats["total_weights"]
        n_layers = len(self._shannon_stats["layers_done"])

        if total_weights > 0:
            avg_bpw = (total_bytes * 8) / total_weights
        else:
            avg_bpw = 0

        total_mb = total_bytes / (1024 * 1024)
        per_layer_mb = total_mb / max(n_layers, 1)

        print(f"  Layers requantized:    {n_layers}")
        print(f"  Total Shannon weights: {total_weights:,}")
        print(f"  Total Shannon size:    {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
        print(f"  Per-layer average:     {per_layer_mb:.2f} MB")
        print(f"  Average bpw:           {avg_bpw:.3f}")
        print(f"  Weight matrices cached:{len(self._shannon_cache)}")

        # Compare to GPTQ
        gptq_total_mb = self.model.total_weight_bytes / (1024 * 1024)
        print(f"\n  GPTQ model size:       {gptq_total_mb:.0f} MB")
        print(f"  Shannon model size:    {total_mb:.2f} MB (projections only)")
        if total_mb > 0:
            ratio = gptq_total_mb / total_mb
            print(f"  Compression vs GPTQ:   {ratio:.2f}x smaller")

    def close(self):
        self.gpu.close()
        self.model.close()


def main() -> int:
    engine = ShannonPrefillEngine()

    prompt = "The capital of France is"
    t0 = time.monotonic()
    token_out, token_text, logits = engine.generate(prompt)
    total = time.monotonic() - t0

    banner("RESULT")
    print(f"  Prompt:    {prompt!r}")
    print(f"  Generated: '{token_text}' (token_id={token_out})")
    print(f"  Expected:  'Paris' or ' Paris'")
    print(f"  Total time: {total:.1f}s")

    # Paris check
    paris_ids = []
    for candidate in ["Paris", " Paris"]:
        ids = engine.tok.encode(candidate)
        paris_ids.extend(ids)
    is_paris = token_out in paris_ids
    print(f"  Match: {'YES' if is_paris else 'NO'}")

    if logits is not None:
        # Report Paris logit
        for cid in paris_ids:
            word = engine.tok.decode([cid])
            print(f"  Paris logit (id={cid}, '{word}'): {logits[cid]:.3f}")

    engine.report_stats()

    engine.close()
    return 0 if is_paris else 1


if __name__ == "__main__":
    raise SystemExit(main())
