#!/usr/bin/env python3
"""
Hybrid end-to-end generation: DeltaNet/attention + MLP per layer.

Runs "The capital of France is" through the full 64-layer pipeline and
measures wall-clock per token with a breakdown:
  - MLP time (GPU kernels: norm + gate/up + activate + down)
  - DeltaNet/attention time (GPU projections + CPU recurrence/attention)
  - Overhead (residual copies, uploads/downloads, etc.)

Reports: Paris rank, wall-clock per token, tok/s, and breakdown.

The Carmack fused kernel handles all 64 MLP layers in a single cooperative
launch (62ms for all 64 layers). It cannot be split per-layer. So for the
hybrid path we use the existing per-layer GPU MLP kernels (gptq_gemv_fast +
norm + activate), which gives us the correct interleaving with attention.

The Carmack MLP-only benchmark (benchmark_carmack.py) shows what the MLP
portion alone costs in a fused launch. This script shows the full picture.

Run:
    python3 /home/ubuntu/lithos/src/generate_paris_carmack.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import sys
import time
from pathlib import Path

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
ZERO_POINT = 8
NUM_LAYERS = 64

# DeltaNet dimensions
NUM_KEY_HEADS = 16
KEY_HEAD_DIM = 128
NUM_VALUE_HEADS = 48
VALUE_HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HEADS_PER_KEY = NUM_VALUE_HEADS // NUM_KEY_HEADS  # 3


# ---------------------------------------------------------------------------
# Timing accumulator with MLP/attention/overhead breakdown
# ---------------------------------------------------------------------------
class TimingBreakdown:
    """Accumulates wall-clock time into named buckets."""

    def __init__(self):
        self.buckets: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._t0: float = 0.0
        self._name: str = ""

    def start(self, name: str):
        self._name = name
        self._t0 = time.monotonic()

    def stop(self) -> float:
        dt = time.monotonic() - self._t0
        self.buckets[self._name] = self.buckets.get(self._name, 0.0) + dt
        self.counts[self._name] = self.counts.get(self._name, 0) + 1
        return dt

    def get(self, name: str) -> float:
        return self.buckets.get(name, 0.0)

    def report(self, label: str = ""):
        total = sum(self.buckets.values())
        print(f"\n{'=' * 72}")
        print(f"  TIMING BREAKDOWN  {label}  (total: {total:.4f}s)")
        print(f"{'=' * 72}")
        for name, t in sorted(self.buckets.items(), key=lambda x: -x[1]):
            cnt = self.counts[name]
            pct = 100.0 * t / total if total > 0 else 0.0
            avg_ms = 1000.0 * t / cnt if cnt > 0 else 0.0
            print(f"  {name:40s}  {t:8.4f}s  ({pct:5.1f}%)  n={cnt:5d}  avg={avg_ms:.3f}ms")
        print(f"{'=' * 72}")

        # Aggregate into MLP / Attention / Overhead
        mlp_keys = ["mlp_norm", "mlp_gate_up_proj", "mlp_activate", "mlp_down_proj"]
        attn_keys = [k for k in self.buckets if k.startswith("dn_") or k.startswith("fa_")]
        overhead_keys = [k for k in self.buckets
                        if k not in mlp_keys and k not in attn_keys]

        mlp_t = sum(self.buckets.get(k, 0) for k in mlp_keys)
        attn_t = sum(self.buckets.get(k, 0) for k in attn_keys)
        overhead_t = sum(self.buckets.get(k, 0) for k in overhead_keys)

        print(f"\n  AGGREGATE:")
        print(f"    MLP (GPU kernels):          {mlp_t:8.4f}s  ({100*mlp_t/total:.1f}%)")
        print(f"    DeltaNet/Attention:          {attn_t:8.4f}s  ({100*attn_t/total:.1f}%)")
        print(f"    Overhead (embed/norm/xfer):  {overhead_t:8.4f}s  ({100*overhead_t/total:.1f}%)")


T = TimingBreakdown()


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


class HybridEngine:
    """Full 64-layer engine: DeltaNet/attention + per-layer MLP on GPU."""

    def __init__(self):
        banner("Initializing Hybrid Engine")
        t0 = time.monotonic()

        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self.layer_types = self.model.config.layer_types

        n_dn = sum(1 for t in self.layer_types if t == "linear_attention")
        n_fa = sum(1 for t in self.layer_types if t == "full_attention")
        print(f"  Model: {self.model}")
        print(f"  Device: {self.gpu.device_name}")
        print(f"  Layers: {NUM_LAYERS} ({n_dn} DeltaNet, {n_fa} full_attn)")

        self._load_kernels()
        self._alloc_buffers()
        self._preload_norms()
        self._preload_deltanet_weights()
        self._init_deltanet_state()
        self.kv_cache = KVCache(max_seq_len=64)
        self._preload_qk_norms()

        self.stream0 = self.gpu.stream_create()
        self.stream1 = self.gpu.stream_create()

        print(f"  Init time: {time.monotonic() - t0:.3f}s")

    def _load_kernels(self):
        gpu = self.gpu

        embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = gpu.get_function(embed_mod, "embed_f16")

        norm_mod = gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = gpu.get_function(norm_mod, "norm")

        if getattr(self.model, "qweight_transposed", False):
            # Model was pre-transposed offline: use coalesced [N, K/8] kernel
            trans_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_gemv_transposed.cubin")
            self.proj_func_fast = gpu.get_function(trans_mod, "gptq_gemv_transposed")
            self._qweight_transposed = True
            print("    qweight layout: transposed [N, K/8] -- using gptq_gemv_transposed")
        else:
            fast_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_gemv_fast.cubin")
            self.proj_func_fast = gpu.get_function(fast_mod, "gptq_gemv_fast")
            self._qweight_transposed = False

        activate_mod = gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = gpu.get_function(activate_mod, "activate")

        lm_head_mod = gpu.load_cubin(f"{KERNEL_DIR}/lm_head.cubin")
        self.lm_head_func = gpu.get_function(lm_head_mod, "lm_head")

        layout = "transposed [N,K/8]" if self._qweight_transposed else "original [K/8,N]"
        print(f"    Kernels loaded: embed_f16, norm, gptq_gemv ({layout}), activate, lm_head")

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
        self.d_lm_head_out = gpu.mem_alloc(248320 * 4)
        self.d_zero = gpu.mem_alloc(HIDDEN_DIM * 4)
        gpu.memcpy_htod(self.d_zero,
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
            self.q_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.q_norm.weight")
            self.k_norm_weights[i] = load_norm_weight(self.model, f"{prefix}.k_norm.weight")

    # ------------------------------------------------------------------
    # GPU kernel wrappers
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

    def gpu_norm(self, d_input, d_residual, norm_w, d_output):
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

    def gpu_gptq_matvec(self, prefix, d_input, d_output, K, N, stream=None):
        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr

        self.gpu.memset_d8(d_output, 0, N * 4, stream=stream)

        K_SPLITS = 16
        K_packed = K // 8
        k_packed_per_split = K_packed // K_SPLITS
        fast_smem = k_packed_per_split * 8 * 4

        self.gpu.launch(
            self.proj_func_fast,
            grid=(math.ceil(N / 256), K_SPLITS, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(qw_ptr),
                ctypes.c_uint64(sc_ptr),
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_uint32(N),
                ctypes.c_uint32(K),
                ctypes.c_uint32(K_SPLITS),
                ctypes.c_uint32(k_packed_per_split),
            ],
            shared_mem=fast_smem,
            stream=stream,
        )

    def gpu_activate(self, d_gate, d_up, d_output, size):
        GRID_ACT = max(1, math.ceil(size / 256))
        self.gpu.launch(
            self.activate_func,
            grid=(GRID_ACT, 1, 1),
            block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_up.value),
                ctypes.c_uint64(d_gate.value),
                ctypes.c_uint64(d_output.value),
            ],
        )

    # ------------------------------------------------------------------
    # Layer execution
    # ------------------------------------------------------------------

    def run_mlp(self, layer_idx: int, d_residual):
        prefix = f"model.language_model.layers.{layer_idx}"
        T.start("mlp_norm")
        self.gpu_norm(d_residual, self.d_zero,
                      self.post_attn_norms[layer_idx], self.d_norm_out)
        self.gpu.synchronize()
        T.stop()

        T.start("mlp_gate_up_proj")
        self.gpu_gptq_matvec(f"{prefix}.mlp.gate_proj",
                              self.d_norm_out, self.d_gate,
                              HIDDEN_DIM, INTERMEDIATE_SIZE,
                              stream=self.stream0)
        self.gpu_gptq_matvec(f"{prefix}.mlp.up_proj",
                              self.d_norm_out, self.d_up,
                              HIDDEN_DIM, INTERMEDIATE_SIZE,
                              stream=self.stream1)
        self.gpu.stream_synchronize(self.stream0)
        self.gpu.stream_synchronize(self.stream1)
        T.stop()

        T.start("mlp_activate")
        self.gpu_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)
        self.gpu.synchronize()
        T.stop()

        T.start("mlp_down_proj")
        self.gpu_gptq_matvec(f"{prefix}.mlp.down_proj",
                              self.d_act, self.d_down,
                              INTERMEDIATE_SIZE, HIDDEN_DIM)
        self.gpu.synchronize()
        T.stop()

    def run_full_attention(self, layer_idx: int, d_normed, position: int):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        T.start("fa_qkv_proj_gpu")
        self.gpu_gptq_matvec(f"{prefix}.q_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 12288)
        self.gpu.synchronize()
        q_gate_raw = download_f32(self.gpu, self.d_attn_scratch1, 12288)

        self.gpu_gptq_matvec(f"{prefix}.k_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        k_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)

        self.gpu_gptq_matvec(f"{prefix}.v_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        v_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)
        T.stop()

        T.start("fa_attn_cpu")
        q_gate_heads = q_gate_raw.reshape(24, 512)
        q_heads = q_gate_heads[:, :256].copy()
        gate_heads = q_gate_heads[:, 256:].copy()
        gate_flat = gate_heads.flatten()

        k_heads = k_raw.reshape(4, 256)
        q_norm_w = self.q_norm_weights[layer_idx]
        k_norm_w = self.k_norm_weights[layer_idx]
        for h in range(24):
            rms = np.sqrt(np.mean(q_heads[h] ** 2) + self.epsilon)
            q_heads[h] = (q_heads[h] / rms) * q_norm_w
        for h in range(4):
            rms = np.sqrt(np.mean(k_heads[h] ** 2) + self.epsilon)
            k_heads[h] = (k_heads[h] / rms) * k_norm_w
        q_raw_normed = q_heads.flatten()
        k_raw_normed = k_heads.flatten()

        q_rope, k_rope = process_qkv_with_rope(q_raw_normed, k_raw_normed, position)

        v_2d = v_raw.reshape(4, 256)
        self.kv_cache.store(layer_idx, position, k_rope, v_2d)
        k_cache, v_cache = self.kv_cache.get(layer_idx, position)
        attended = attention_prefill(q_rope, k_cache, v_cache, position)

        gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_flat.clip(-80, 80)))
        attended = attended * gate_sigmoid
        T.stop()

        T.start("fa_out_proj_gpu")
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attended.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.o_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()
        T.stop()

    def run_deltanet(self, layer_idx: int, d_normed):
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        T.start("dn_proj_qkv")
        self.gpu_gptq_matvec(f"{prefix}.in_proj_qkv",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 10240)
        self.gpu_gptq_matvec(f"{prefix}.in_proj_z",
                              d_normed, self.d_attn_scratch2,
                              HIDDEN_DIM, 6144)
        self.gpu.synchronize()
        T.stop()

        T.start("dn_download_proj")
        qkv = download_f32(self.gpu, self.d_attn_scratch1, 10240)
        z = download_f32(self.gpu, self.d_attn_scratch2, 6144)
        normed_cpu = download_f32(self.gpu, d_normed, HIDDEN_DIM)
        T.stop()

        T.start("dn_conv1d_cpu")
        conv_buf = self.dn_conv_buf[layer_idx]
        conv_w = self.dn_conv_w[layer_idx]
        conv_window = np.concatenate([conv_buf, qkv[:, np.newaxis]], axis=1)
        qkv_conv = np.sum(conv_window * conv_w[:, 0, :], axis=1)
        self.dn_conv_buf[layer_idx] = conv_window[:, 1:]
        qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv.clip(-80, 80))))
        T.stop()

        T.start("dn_pre_recurrence_cpu")
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
        T.stop()

        T.start("dn_recurrence_cpu")
        S = self.dn_S[layer_idx]
        k_exp = np.repeat(k, HEADS_PER_KEY, axis=0)
        q_exp = np.repeat(q, HEADS_PER_KEY, axis=0)
        S *= decay[:, None, None]
        kv_mem = np.einsum('hvk,hk->hv', S, k_exp)
        delta = (v - kv_mem) * beta[:, None]
        S += np.einsum('hv,hk->hvk', delta, k_exp)
        output_heads = np.einsum('hvk,hk->hv', S, q_exp)
        self.dn_S[layer_idx] = S
        T.stop()

        T.start("dn_groupnorm_cpu")
        head_norm_w = self.dn_head_norm_w[layer_idx]
        for h in range(NUM_VALUE_HEADS):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w
        T.stop()

        T.start("dn_gate_z_cpu")
        z_heads = z.reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu
        T.stop()

        T.start("dn_out_proj_gpu")
        attn_output_flat = output_heads.flatten()
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output_flat.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.out_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()
        T.stop()

    def run_layer(self, layer_idx: int, x: np.ndarray, position: int) -> np.ndarray:
        layer_type = self.layer_types[layer_idx]
        gpu = self.gpu

        residual = x.copy()

        T.start("upload_x")
        d_x_gpu = upload_f32(gpu, x)
        T.stop()

        T.start("norm")
        self.gpu_norm(d_x_gpu, self.d_zero,
                      self.input_norms[layer_idx], self.d_norm_out)
        gpu.synchronize()
        T.stop()

        if layer_type == "full_attention":
            self.run_full_attention(layer_idx, self.d_norm_out, position)
        else:
            self.run_deltanet(layer_idx, self.d_norm_out)

        T.start("download_attn")
        attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
        T.stop()

        T.start("residual_add")
        x = residual + attn_out
        T.stop()

        T.start("mem_free")
        gpu.mem_free(d_x_gpu)
        T.stop()

        # MLP sublayer
        T.start("upload_mlp_residual")
        gpu.memcpy_htod(self.d_residual,
                        x.ctypes.data_as(ctypes.c_void_p),
                        HIDDEN_DIM * 4)
        T.stop()

        self.run_mlp(layer_idx, self.d_residual)

        T.start("download_mlp")
        mlp_out = download_f32(gpu, self.d_down, HIDDEN_DIM)
        T.stop()

        T.start("residual_add")
        x = x + mlp_out
        T.stop()

        return x

    def process_token(self, token_id: int, token_pos: int) -> np.ndarray:
        T.start("embed")
        x = self.gpu_embed(token_id)
        T.stop()

        for layer_idx in range(NUM_LAYERS):
            x = self.run_layer(layer_idx, x, token_pos)

        return x

    def generate(self, prompt: str):
        token_ids = self.tok.encode(prompt)
        print(f"\nPrompt: {prompt!r}")
        print(f"Token IDs: {token_ids}")
        for i, tid in enumerate(token_ids):
            print(f"  Token {i}: id={tid} '{self.tok.decode([tid])}'")

        banner("PREFILL: Processing prompt tokens")

        x = None
        token_times = []
        for pos, tid in enumerate(token_ids):
            word = self.tok.decode([tid])
            t0 = time.monotonic()
            print(f"\n--- Token {pos+1}/{len(token_ids)}: '{word}' (id={tid}) ---")

            x = self.process_token(tid, pos)

            elapsed = time.monotonic() - t0
            token_times.append(elapsed)
            x_norm = np.linalg.norm(x)
            has_nan = np.any(np.isnan(x))
            print(f"  Done: norm={x_norm:.4f} time={elapsed:.2f}s nan={has_nan}")

            if has_nan:
                print("  ERROR: NaN detected, aborting!")
                return -1, "<error>", None, token_times

        # --- Generate: decode the next token ---
        banner("GENERATE: Decoding next token")

        T.start("final_norm")
        x_normed = rms_norm_cpu(x, self.final_norm_w, self.epsilon)
        T.stop()

        T.start("lm_head")
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
        T.stop()

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

        # Check Paris rank
        for candidate in ["Paris", " Paris"]:
            cand_ids = self.tok.encode(candidate)
            for cid in cand_ids:
                rank = int(np.sum(logits > logits[cid])) + 1
                cword = self.tok.decode([cid])
                print(f"  '{candidate}' (id={cid}, decoded='{cword}'): logit={logits[cid]:.3f}, rank={rank}")

        return token_out, token_text, logits, token_times

    def close(self):
        self.gpu.close()
        self.model.close()


def main() -> int:
    engine = HybridEngine()

    prompt = "The capital of France is"
    t0 = time.monotonic()
    token_out, token_text, logits, token_times = engine.generate(prompt)
    total = time.monotonic() - t0

    banner("RESULT")
    print(f"  Prompt: {prompt!r}")
    print(f"  Generated: '{token_text}' (token_id={token_out})")

    # Check Paris
    paris_ids = []
    for candidate in ["Paris", " Paris"]:
        ids = engine.tok.encode(candidate)
        paris_ids.extend(ids)
    is_paris = token_out in paris_ids
    paris_rank = "1" if is_paris else "?"
    for candidate in ["Paris", " Paris"]:
        cand_ids = engine.tok.encode(candidate)
        for cid in cand_ids:
            r = int(np.sum(logits > logits[cid])) + 1
            if r == 1:
                paris_rank = "1"

    print(f"  Paris rank: {paris_rank}")
    print(f"  Match: {'YES' if is_paris else 'NO'}")

    # Per-token timing
    print(f"\n  Per-token wall-clock:")
    for i, t in enumerate(token_times):
        print(f"    Token {i}: {t:.3f}s ({t*1000:.1f}ms)")

    # Last token is the one that matters for decode-phase tok/s
    last_token_ms = token_times[-1] * 1000.0
    tok_s = 1000.0 / last_token_ms if last_token_ms > 0 else 0
    avg_ms = sum(token_times) / len(token_times) * 1000.0

    print(f"\n  --- Performance ---")
    print(f"  Last token wall-clock:  {last_token_ms:.1f} ms")
    print(f"  Average per token:      {avg_ms:.1f} ms")
    print(f"  tok/s (last token):     {tok_s:.2f}")
    print(f"  tok/s (average):        {1000.0/avg_ms:.2f}")
    print(f"  Total generation time:  {total:.2f}s")

    # Detailed breakdown
    T.report(f"(last token = {last_token_ms:.1f}ms)")

    # Carmack comparison
    print(f"\n  --- Carmack MLP-Only Comparison ---")
    print(f"  Carmack fused 64-layer MLP: ~62ms (single cooperative launch)")
    print(f"  This hybrid uses per-layer GPU MLP kernels interleaved with attention.")
    print(f"  The Carmack kernel fuses all 64 layers and cannot be split per-layer.")
    print(f"  To get Carmack-level MLP speed with attention, the Carmack kernel")
    print(f"  would need to be extended with DeltaNet/attention in PTX (Approach B).")

    engine.close()
    return 0 if is_paris else 1


if __name__ == "__main__":
    raise SystemExit(main())
