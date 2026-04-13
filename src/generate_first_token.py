#!/usr/bin/env python3
"""
Generate the first token from Qwen 3.5-27B on GH200.

Phases:
  1. MLP-only pass through all 64 layers (skip attention/DeltaNet)
  2. Add full attention for the 16 full_attention layers (first token = trivial)
  3. Add DeltaNet for the 48 linear_attention layers (first token, zero state)
  4. Final norm + lm_head -> argmax -> token

Run:
    python3 /home/ubuntu/lithos/src/generate_first_token.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_driver import CUDADriver, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer
from kv_cache import KVCache
from attention import attention_prefill, process_qkv_with_rope

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
ZERO_POINT = 8
NUM_LAYERS = 64


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


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


class InferenceEngine:
    """Manages GPU resources and runs the 64-layer pipeline."""

    def __init__(self):
        banner("Initializing Inference Engine")
        t0 = time.monotonic()

        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self.layer_types = self.model.config.layer_types

        print(f"  Model: {self.model}")
        print(f"  Device: {self.gpu.device_name}")
        print(f"  Layers: {NUM_LAYERS} ({sum(1 for t in self.layer_types if t == 'linear_attention')} DeltaNet, "
              f"{sum(1 for t in self.layer_types if t == 'full_attention')} full_attn)")
        print(f"  Epsilon: {self.epsilon}")

        # Load kernels
        self._load_kernels()

        # Preallocate GPU buffers for the hot path
        self._alloc_buffers()

        # Preload all norm weights (BF16 -> F32) -- small, 64 layers * 2 norms * 5120 * 4 = 2.5 MB
        self._preload_norms()

        # KV cache for the 16 full-attention layers
        self.kv_cache = KVCache(max_seq_len=256)
        print(f"  KV cache: {self.kv_cache.memory_bytes() / (1024**2):.1f} MiB "
              f"(max_seq_len=256)")

        t1 = time.monotonic()
        print(f"  Init time: {t1-t0:.3f}s")

    def _load_kernels(self):
        print("  Loading kernels...")
        gpu = self.gpu

        embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = gpu.get_function(embed_mod, "embed_f16")

        norm_mod = gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = gpu.get_function(norm_mod, "norm")

        proj_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
        self.proj_func = gpu.get_function(proj_mod, "gptq_matvec")

        activate_mod = gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = gpu.get_function(activate_mod, "activate")

        print("    embed_f16, norm, gptq_matvec, activate: loaded")

    def _alloc_buffers(self):
        """Preallocate reusable GPU buffers."""
        gpu = self.gpu
        # We need buffers for: norm input, norm output, gate, up, activate, down, mlp_input
        # Plus embed output and residual stream
        # Max intermediate size is 17408 (gate/up), everything else is 5120
        # For attention: QKV is 10240 (DeltaNet) or Q=12288,K=1024,V=1024 (full_attn)
        # We'll allocate the biggest we need and reuse.

        self.d_x = gpu.mem_alloc(HIDDEN_DIM * 4)           # current activation (5120)
        self.d_residual = gpu.mem_alloc(HIDDEN_DIM * 4)     # residual stream (5120)
        self.d_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)     # norm output (5120)
        self.d_norm_weight = gpu.mem_alloc(HIDDEN_DIM * 4)  # norm weight (5120)
        self.d_gate = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)  # gate projection (17408)
        self.d_up = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)    # up projection (17408)
        self.d_act = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)   # activate output (17408)
        self.d_down = gpu.mem_alloc(HIDDEN_DIM * 4)         # down projection (5120)
        self.d_mlp_in = gpu.mem_alloc(HIDDEN_DIM * 4)       # MLP input (copy of norm_out)

        # For attention projections (reusable scratch)
        # DeltaNet: QKV = 10240, Z = 6144
        # Full attn: Q = 12288, K = 1024, V = 1024, o_proj input = 6144
        self.d_attn_scratch1 = gpu.mem_alloc(12288 * 4)  # biggest projection output
        self.d_attn_scratch2 = gpu.mem_alloc(6144 * 4)   # for Z / V expansion / o_proj input
        self.d_attn_out = gpu.mem_alloc(HIDDEN_DIM * 4)  # attention output -> 5120

        # Zero buffer for norm residual=0 trick
        self.d_zero = gpu.mem_alloc(HIDDEN_DIM * 4)
        gpu.memcpy_htod(self.d_zero,
                        np.zeros(HIDDEN_DIM, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                        HIDDEN_DIM * 4)

        print(f"    GPU buffers allocated")

    def _preload_norms(self):
        """Preload all norm weights into CPU arrays.

        Qwen3NextRMSNorm uses (1 + weight) * norm(x), but our GPU norm kernel
        multiplies by weight directly.  We add 1.0 to all norm weights here
        so the kernel produces the correct result.
        """
        self.input_norms = []
        self.post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            self.input_norms.append(
                load_norm_weight(self.model, f"{prefix}.input_layernorm.weight") + 1.0)
            self.post_attn_norms.append(
                load_norm_weight(self.model, f"{prefix}.post_attention_layernorm.weight") + 1.0)

        self.final_norm_w = load_norm_weight(self.model, "model.language_model.norm.weight") + 1.0
        print(f"    {NUM_LAYERS * 2 + 1} norm weights preloaded (with +1.0 for Qwen3NextRMSNorm)")

    # ------------------------------------------------------------------
    # GPU kernel wrappers
    # ------------------------------------------------------------------

    def gpu_embed(self, token_id: int) -> np.ndarray:
        """Embed a single token: FP16 table -> F32 vector."""
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

    def gpu_norm(self, d_input: CUdeviceptr, d_residual: CUdeviceptr,
                 norm_w: np.ndarray, d_output: CUdeviceptr):
        """Run RMSNorm kernel: output = norm(input + residual) * weight."""
        # Upload norm weight
        self.gpu.memcpy_htod(self.d_norm_weight,
                             norm_w.ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)
        self.gpu.launch(
            self.norm_func,
            grid=(1, 1, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_residual.value),
                ctypes.c_uint64(self.d_norm_weight.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_float(self.epsilon),
            ],
            shared_mem=128,
        )

    def gpu_gptq_matvec(self, prefix: str, d_input: CUdeviceptr,
                         d_output: CUdeviceptr, K: int, N: int):
        """Run GPTQ W4A16 matvec: output = input @ dequant(qweight)."""
        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr

        # Zero output buffer
        self.gpu.memcpy_htod(d_output,
                             np.zeros(N, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             N * 4)

        self.gpu.launch(
            self.proj_func,
            grid=(N, 1, 1),
            block=(256, 1, 1),
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

    def gpu_activate(self, d_gate: CUdeviceptr, d_up: CUdeviceptr,
                      d_output: CUdeviceptr, size: int):
        """SiLU activation: output = silu(gate) * up.

        The kernel computes param1 * silu(param2), so we swap to get silu(gate) * up.
        """
        GRID_ACT = max(1, math.ceil(size / 256))
        self.gpu.launch(
            self.activate_func,
            grid=(GRID_ACT, 1, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_up.value),     # param1 = up (straight multiply)
                ctypes.c_uint64(d_gate.value),   # param2 = gate (gets SiLU)
                ctypes.c_uint64(d_output.value),
            ],
        )

    # ------------------------------------------------------------------
    # Layer execution
    # ------------------------------------------------------------------

    def run_mlp(self, layer_idx: int, d_residual: CUdeviceptr):
        """Run the MLP sublayer for one layer.

        Input: d_residual contains the residual stream (after attention).
        The norm kernel reads (input + residual), but we want norm(residual)
        since there's no separate input. We pass residual as input and zero as residual.

        After MLP: residual += down_proj output (done on CPU).
        Returns the MLP output (down_proj result) as numpy.
        """
        prefix = f"model.language_model.layers.{layer_idx}"

        # Post-attention norm: output = norm(residual + 0) * weight = norm(residual) * weight
        self.gpu_norm(d_residual, self.d_zero,
                      self.post_attn_norms[layer_idx], self.d_norm_out)

        # Copy norm output for MLP input (gate and up both read from it)
        self.gpu.memcpy_htod(self.d_mlp_in,
                             ctypes.c_void_p(0), 0)  # dummy, we'll use d_norm_out directly

        # Gate projection: 5120 -> 17408
        self.gpu_gptq_matvec(f"{prefix}.mlp.gate_proj",
                              self.d_norm_out, self.d_gate,
                              HIDDEN_DIM, INTERMEDIATE_SIZE)

        # Up projection: 5120 -> 17408
        self.gpu_gptq_matvec(f"{prefix}.mlp.up_proj",
                              self.d_norm_out, self.d_up,
                              HIDDEN_DIM, INTERMEDIATE_SIZE)

        # SiLU(gate) * up
        self.gpu_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)

        # Need to upload activate result for down_proj input
        # d_act is already on GPU, but down_proj kernel reads from a different buffer
        # Actually, d_act IS on GPU. We can use it directly.

        # Down projection: 17408 -> 5120
        self.gpu_gptq_matvec(f"{prefix}.mlp.down_proj",
                              self.d_act, self.d_down,
                              INTERMEDIATE_SIZE, HIDDEN_DIM)

        self.gpu.synchronize()

    def run_full_attention(self, layer_idx: int, d_normed: CUdeviceptr,
                           position: int):
        """Run full attention with KV cache for a token at the given position.

        Steps:
          1. Project Q (5120 -> 6144), K (5120 -> 1024), V (5120 -> 1024)
          2. Apply RoPE to Q and K
          3. Store K, V in the KV cache at this position
          4. Retrieve cached K, V for positions 0..position
          5. Compute scaled dot-product attention with GQA
          6. O projection (6144 -> 5120)

        d_normed: input (after input_layernorm), on GPU
        position: sequence position (0-based)
        Writes result to self.d_attn_out
        """
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        # Q projection: 5120 -> 6144 (24 q_heads * 256 head_dim)
        self.gpu_gptq_matvec(f"{prefix}.q_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 6144)

        # K projection: 5120 -> 1024 (4 kv_heads * 256 head_dim)
        # Reuse d_attn_scratch2 for K temporarily
        self.gpu_gptq_matvec(f"{prefix}.k_proj",
                              d_normed, self.d_attn_scratch2,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()

        q_raw = download_f32(self.gpu, self.d_attn_scratch1, 6144)
        k_raw = download_f32(self.gpu, self.d_attn_scratch2, 1024)

        # V projection: 5120 -> 1024 (4 kv_heads * 256 head_dim)
        self.gpu_gptq_matvec(f"{prefix}.v_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()

        v_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)

        # Apply RoPE to Q and K
        q_rope, k_rope = process_qkv_with_rope(q_raw, k_raw, position)

        # Store K (with RoPE) and V (no RoPE) in the cache
        self.kv_cache.store(layer_idx, position, k_rope, v_raw)

        # Retrieve all cached K, V for positions 0..position
        k_cached, v_cached = self.kv_cache.get(layer_idx, position)

        # Compute attention
        attn_output = attention_prefill(q_rope, k_cached, v_cached, position)
        # attn_output shape: [6144] = [24 * 256]

        # Upload for O projection
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)

        # O projection: 6144 -> 5120
        self.gpu_gptq_matvec(f"{prefix}.o_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()

    def run_deltanet_first_token(self, layer_idx: int, d_normed: CUdeviceptr):
        """Run DeltaNet linear attention for the first token (zero-initialized state).

        DeltaNet for first token with zero state S=0:
          1. QKV projection: 5120 -> 10240 (Q:2048, K:2048, V:6144)
          2. conv1d: with zero padding, first token = input * conv_weight[:, :, -1]
             (only the last conv position matters for first token)
          3. beta = sigmoid(in_proj_a @ x) where x is original input (pre-QKV)
          4. gate (decay) = sigmoid(-exp(A_log)) per head -- NOT dependent on input for DeltaNet
             Actually: gating uses dt computed from in_proj_b
          5. For zero state: retrieval = 0, so output comes only from new state
          6. Apply group norm per head on output
          7. Gate with Z: output = output * silu(Z)
          8. o_proj: 6144 -> 5120

        For simplicity on the first token, we'll compute this on CPU using the
        GPTQ kernel for projections and CPU for the recurrence math.

        d_normed: input after input_layernorm, on GPU
        Writes result to self.d_attn_out
        """
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        model = self.model

        # --- Step 1: QKV projection (GPTQ, 5120 -> 10240) ---
        self.gpu_gptq_matvec(f"{prefix}.in_proj_qkv",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 10240)

        # --- Step 1b: Z projection (GPTQ, 5120 -> 6144) ---
        self.gpu_gptq_matvec(f"{prefix}.in_proj_z",
                              d_normed, self.d_attn_scratch2,
                              HIDDEN_DIM, 6144)
        self.gpu.synchronize()

        # Download QKV and Z
        qkv = download_f32(self.gpu, self.d_attn_scratch1, 10240)
        z = download_f32(self.gpu, self.d_attn_scratch2, 6144)

        # Download the normed input for beta computation
        normed_cpu = download_f32(self.gpu, d_normed, HIDDEN_DIM)

        # Split QKV: Q=[0:2048], K=[2048:4096], V=[4096:10240]
        q_raw = qkv[:2048]   # 16 key_heads * 128 key_head_dim
        k_raw = qkv[2048:4096]
        v_raw = qkv[4096:]   # 48 value_heads * 128 value_head_dim

        # --- Step 2: Conv1d ---
        # conv1d.weight shape: [10240, 1, 4] -- depthwise conv, kernel_size=4
        # For the FIRST token with zero-padded history, only the last kernel position
        # (index 3) contributes: output[ch] = input[ch] * weight[ch, 0, 3]
        conv_w_raw = bytes(model.weight_bytes(f"{prefix}.conv1d.weight"))
        conv_w = bf16_to_f32(conv_w_raw).reshape(10240, 1, 4)

        # Apply conv: for first token, output = qkv * conv_w[:, 0, 3]
        qkv_conv = qkv * conv_w[:, 0, 3]

        # Apply SiLU activation after conv1d (matches reference: F.silu(conv1d(x)))
        qkv_conv = silu_cpu(qkv_conv)

        # Re-split after conv
        q = qkv_conv[:2048].reshape(16, 128)    # 16 key heads, 128 dim
        k = qkv_conv[2048:4096].reshape(16, 128)
        v = qkv_conv[4096:].reshape(48, 128)    # 48 value heads, 128 dim

        # L2 normalize Q and K per head
        for hi in range(16):
            q_norm = np.sqrt(np.sum(q[hi] ** 2) + 1e-6)
            q[hi] = q[hi] / q_norm
            k_norm = np.sqrt(np.sum(k[hi] ** 2) + 1e-6)
            k[hi] = k[hi] / k_norm

        # Scale Q by 1/sqrt(head_dim)
        q = q * (1.0 / np.sqrt(128.0))

        # --- Step 3: Compute beta (controls state update) ---
        # beta = sigmoid(in_proj_b @ x)  -- shape [48] (one per value head)
        b_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_b.weight"))
        b_weight = bf16_to_f32(b_w_raw).reshape(48, HIDDEN_DIM)
        beta = 1.0 / (1.0 + np.exp(-(b_weight @ normed_cpu).clip(-80, 80)))  # [48]

        # --- Step 4: Compute dt (timestep) from in_proj_a ---
        # dt controls the gating/decay: g = -exp(A_log) * softplus(a + dt_bias)
        a_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_a.weight"))
        a_weight = bf16_to_f32(a_w_raw).reshape(48, HIDDEN_DIM)
        dt_bias_raw = bytes(model.weight_bytes(f"{prefix}.dt_bias"))
        dt_bias = bf16_to_f32(dt_bias_raw)  # [48]

        dt = a_weight @ normed_cpu + dt_bias  # [48]
        dt = np.log1p(np.exp(dt.clip(-20, 20)))  # softplus

        # --- Step 5: Compute decay (A) ---
        a_log_raw = bytes(model.weight_bytes(f"{prefix}.A_log"))
        A_log = np.frombuffer(a_log_raw, dtype=np.float32).copy()  # [48]
        # A = -exp(A_log), then decay = exp(A * dt)
        A = -np.exp(A_log)  # [48], negative
        decay = np.exp(A * dt)  # [48], in (0, 1)

        # --- Step 6: DeltaNet recurrence with zero initial state ---
        # For each value head h (48 total):
        #   The state S_h is a (key_dim, value_dim) = (128, 128) matrix
        #   But key_heads=16, value_heads=48, so key_head_group = 48/16 = 3 value_heads per key head
        #
        #   For head h:
        #     key_head_idx = h // 3  (each key head serves 3 value heads)
        #     q_h = q[key_head_idx]  # [128]
        #     k_h = k[key_head_idx]  # [128]
        #     v_h = v[h]             # [128]
        #
        #   With zero state S=0:
        #     retrieval = q_h @ S = 0
        #     update: S_new = decay[h] * S + beta[h] * outer(k_h, (v_h - retrieval))
        #                   = beta[h] * outer(k_h, v_h)    (since S=0, decay*0=0, retrieval=0)
        #     output_h = q_h @ S_new = beta[h] * (q_h @ k_h^T) * v_h
        #              = beta[h] * dot(q_h, k_h) * v_h

        value_heads_per_key = 48 // 16  # = 3
        output_heads = np.zeros((48, 128), dtype=np.float32)

        for h in range(48):
            key_idx = h // value_heads_per_key
            q_h = q[key_idx]   # [128]
            k_h = k[key_idx]   # [128]
            v_h = v[h]         # [128]

            # With zero state: output = beta[h] * dot(q_h, k_h) * v_h
            qk_dot = np.dot(q_h, k_h)
            output_heads[h] = beta[h] * qk_dot * v_h

        # --- Step 7: Group norm per head ---
        # norm.weight shape: [128] -- shared across all 48 heads
        norm_w_raw = bytes(model.weight_bytes(f"{prefix}.norm.weight"))
        head_norm_w = np.frombuffer(norm_w_raw, dtype=np.float32).copy()  # [128]

        for h in range(48):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w

        # --- Step 8: Gate with Z ---
        # z shape: [6144] = [48, 128]
        z_heads = z.reshape(48, 128)
        # output = output * silu(z)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu

        # Flatten: [48, 128] -> 6144
        attn_output_flat = output_heads.flatten()

        # --- Step 9: Output projection (GPTQ, 6144 -> 5120) ---
        # Upload the attention output and run o_proj
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output_flat.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)

        self.gpu_gptq_matvec(f"{prefix}.out_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()

    def run_layer(self, layer_idx: int, x: np.ndarray, phase: int,
                  position: int = 0) -> np.ndarray:
        """Run one transformer layer.

        phase 1: MLP only (skip attention)
        phase 2: MLP + full attention for full_attn layers, skip DeltaNet
        phase 3: MLP + full attention + DeltaNet (complete)

        x: input activation [5120] on CPU
        position: sequence position for KV cache and RoPE (0-based)
        Returns: output activation [5120] on CPU
        """
        layer_type = self.layer_types[layer_idx]
        gpu = self.gpu
        residual = x.copy()

        if phase >= 2 and layer_type == "full_attention":
            # --- Input norm ---
            d_x_gpu = upload_f32(gpu, x)
            self.gpu_norm(d_x_gpu, self.d_zero,
                          self.input_norms[layer_idx], self.d_norm_out)
            gpu.synchronize()

            # --- Full attention with KV cache ---
            self.run_full_attention(layer_idx, self.d_norm_out, position)

            # Download attention output and add residual
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
            x = residual + attn_out
            residual = x.copy()
            gpu.mem_free(d_x_gpu)

        elif phase >= 3 and layer_type == "linear_attention":
            # --- Input norm ---
            d_x_gpu = upload_f32(gpu, x)
            self.gpu_norm(d_x_gpu, self.d_zero,
                          self.input_norms[layer_idx], self.d_norm_out)
            gpu.synchronize()

            # --- DeltaNet ---
            self.run_deltanet_first_token(layer_idx, self.d_norm_out)

            # Download attention output and add residual
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
            x = residual + attn_out
            residual = x.copy()
            gpu.mem_free(d_x_gpu)

        # --- MLP sublayer ---
        # Upload current x as residual for MLP norm
        gpu.memcpy_htod(self.d_residual,
                        x.ctypes.data_as(ctypes.c_void_p),
                        HIDDEN_DIM * 4)

        self.run_mlp(layer_idx, self.d_residual)

        # Download MLP output (down_proj) and add residual
        mlp_out = download_f32(gpu, self.d_down, HIDDEN_DIM)
        x = x + mlp_out

        return x

    def generate_token(self, prompt: str, phase: int) -> tuple:
        """Run the full pipeline and return (token_id, token_text, logits).

        For phase >= 2, processes ALL tokens in the prompt through the KV cache
        so that the final token's attention can see the full context.
        """
        token_ids = self.tok.encode(prompt)
        num_tokens = len(token_ids)

        phase_names = {1: "MLP-only", 2: "+FullAttn", 3: "+DeltaNet (complete)"}
        token_words = [self.tok.decode([tid]) for tid in token_ids]
        banner(f"Phase {phase}: {phase_names[phase]} -- "
               f"{num_tokens} tokens: {list(zip(token_ids, token_words))}")

        # Reset KV cache for this generation
        self.kv_cache.reset()

        t0 = time.monotonic()

        # For phase 1 (MLP-only), only process the first token (no attention anyway)
        # For phase >= 2, process ALL tokens to build up the KV cache
        tokens_to_process = token_ids if phase >= 2 else [token_ids[0]]

        for tok_pos, tid in enumerate(tokens_to_process):
            t_tok = time.monotonic()
            word = self.tok.decode([tid])

            if len(tokens_to_process) > 1:
                print(f"\n  --- Token {tok_pos}: '{word}' (id={tid}) ---")

            # Step 1: Embed
            x = self.gpu_embed(tid)

            # Check for NaN/Inf
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("  ERROR: NaN/Inf in embedding!")
                return -1, "<error>", None

            # Step 2: Run all 64 layers
            t_layers = time.monotonic()
            for layer_idx in range(NUM_LAYERS):
                x = self.run_layer(layer_idx, x, phase, position=tok_pos)

                # Periodic status (only for last token or single-token mode)
                if tok_pos == len(tokens_to_process) - 1:
                    if (layer_idx + 1) % 8 == 0 or layer_idx == 0:
                        x_norm = np.linalg.norm(x)
                        has_nan = np.any(np.isnan(x))
                        has_inf = np.any(np.isinf(x))
                        elapsed = time.monotonic() - t_layers
                        print(f"  Layer {layer_idx:2d}: norm={x_norm:.4f} "
                              f"NaN={has_nan} Inf={has_inf}  [{elapsed:.2f}s elapsed]")
                        if has_nan or has_inf:
                            print("  ERROR: NaN/Inf detected, aborting!")
                            return -1, "<error>", None

            t_tok_done = time.monotonic()
            if len(tokens_to_process) > 1:
                print(f"  Token {tok_pos} done: {t_tok_done - t_tok:.2f}s, "
                      f"norm={np.linalg.norm(x):.4f}")

        t_layers_done = time.monotonic()
        print(f"  {len(tokens_to_process)} token(s), 64 layers each: "
              f"{t_layers_done - t0:.2f}s total")

        # Step 3: Final RMSNorm
        x_normed = rms_norm_cpu(x, self.final_norm_w, self.epsilon)
        print(f"  Final norm: norm={np.linalg.norm(x_normed):.4f}")

        # Step 4: lm_head projection (FP16, 248320 x 5120)
        # This is a dense matmul: logits = x_normed @ lm_head^T
        # lm_head.weight is [248320, 5120] in FP16
        print(f"  Computing lm_head (248320 x 5120 FP16 matmul on CPU)...")
        t_lm = time.monotonic()

        lm_head_ti = self.model.weight_info("lm_head.weight")
        lm_head_raw = bytes(self.model.weight_bytes("lm_head.weight"))
        # Load as FP16, convert chunks to avoid huge memory spike
        vocab_size = 248320

        # Process in chunks to avoid allocating 248320 * 5120 * 4 = 5GB of f32
        CHUNK = 8192
        logits = np.zeros(vocab_size, dtype=np.float32)
        for start in range(0, vocab_size, CHUNK):
            end = min(start + CHUNK, vocab_size)
            chunk_size = end - start
            offset = start * HIDDEN_DIM * 2  # 2 bytes per fp16
            chunk_bytes = chunk_size * HIDDEN_DIM * 2
            w_chunk = np.frombuffer(
                lm_head_raw[offset:offset + chunk_bytes],
                dtype=np.float16
            ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
            logits[start:end] = w_chunk @ x_normed

        t_lm_done = time.monotonic()
        print(f"  lm_head: {t_lm_done - t_lm:.2f}s")

        # Step 5: Argmax
        token_out = int(np.argmax(logits))
        token_text = self.tok.decode([token_out])

        # Top-5
        top5_idx = np.argsort(logits)[-5:][::-1]
        print(f"\n  Top-5 predictions:")
        for rank, idx in enumerate(top5_idx):
            word = self.tok.decode([int(idx)])
            print(f"    {rank+1}. token={idx:6d} logit={logits[idx]:8.3f} '{word}'")

        total_time = time.monotonic() - t0
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Output token: {token_out} = '{token_text}'")

        return token_out, token_text, logits

    def close(self):
        self.gpu.close()
        self.model.close()


def main() -> int:
    engine = InferenceEngine()

    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {engine.tok.encode(prompt)}")

    # Phase 1: MLP only
    tid1, text1, logits1 = engine.generate_token(prompt, phase=1)
    if tid1 < 0:
        print("\nPhase 1 FAILED (NaN/Inf). Stopping.")
        engine.close()
        return 1

    # Phase 2: MLP + Full Attention
    tid2, text2, logits2 = engine.generate_token(prompt, phase=2)
    if tid2 < 0:
        print("\nPhase 2 FAILED (NaN/Inf). Stopping.")
        engine.close()
        return 1

    # Phase 3: MLP + Full Attention + DeltaNet (complete first-token inference)
    tid3, text3, logits3 = engine.generate_token(prompt, phase=3)

    banner("RESULTS SUMMARY")
    print(f"  Prompt: {prompt!r}")
    print(f"  Phase 1 (MLP only):      token={tid1:6d} '{text1}'")
    print(f"  Phase 2 (+FullAttn):     token={tid2:6d} '{text2}'")
    print(f"  Phase 3 (+DeltaNet):     token={tid3:6d} '{text3}'")
    print(f"\n  Expected answer: 'Paris' or ' Paris'")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
