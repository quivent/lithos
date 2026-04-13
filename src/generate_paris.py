#!/usr/bin/env python3
"""
Generate the next token after "The capital of France is" by processing
all 5 prompt tokens sequentially through the full 64-layer pipeline.

DeltaNet state (S matrices) and conv1d history are maintained between tokens.
Full attention layers use real attention with KV cache and RoPE.

Run:
    python3 /home/ubuntu/lithos/src/generate_paris.py
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
# Profiling accumulator
# ---------------------------------------------------------------------------
class Profiler:
    """Accumulates timing across named categories."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._stack: list[tuple[str, float]] = []

    def start(self, name: str):
        self._stack.append((name, time.monotonic()))

    def stop(self):
        name, t0 = self._stack.pop()
        dt = time.monotonic() - t0
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
        return dt

    def report(self):
        total = sum(self.totals.values())
        print(f"\n{'=' * 72}")
        print(f"  PROFILE REPORT  (total accounted: {total:.4f}s)")
        print(f"{'=' * 72}")
        # Sort by total time descending
        for name, t in sorted(self.totals.items(), key=lambda x: -x[1]):
            cnt = self.counts[name]
            pct = 100.0 * t / total if total > 0 else 0.0
            avg_ms = 1000.0 * t / cnt if cnt > 0 else 0.0
            print(f"  {name:40s}  {t:8.4f}s  ({pct:5.1f}%)  n={cnt:5d}  avg={avg_ms:.3f}ms")
        print(f"{'=' * 72}")


PROF = Profiler()


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


class PrefillEngine:
    """Processes multiple tokens sequentially, maintaining DeltaNet state."""

    def __init__(self):
        banner("Initializing Prefill Engine")
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

        self._load_kernels()
        self._alloc_buffers()
        self._preload_norms()
        self._preload_deltanet_weights()
        self._init_deltanet_state()
        self.kv_cache = KVCache(max_seq_len=64)
        self._preload_qk_norms()
        print(f"  KV cache initialized ({self.kv_cache.memory_bytes() / 1024 / 1024:.1f} MB)")

        # CUDA streams for concurrent gate/up projections
        self.stream0 = self.gpu.stream_create()
        self.stream1 = self.gpu.stream_create()
        print(f"  CUDA streams created for concurrent MLP projections")

        t1 = time.monotonic()
        print(f"  Init time: {t1-t0:.3f}s")

    def _load_kernels(self):
        print("  Loading kernels...")
        gpu = self.gpu

        embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = gpu.get_function(embed_mod, "embed_f16")

        norm_mod = gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = gpu.get_function(norm_mod, "norm")

        proj_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec_tc.cubin")
        self.proj_func = gpu.get_function(proj_mod, "gptq_matvec_tc")
        # Allow dynamic shared memory up to 70KB for K=17408 (down_proj)
        gpu.set_max_dynamic_shared(self.proj_func, 69632)

        proj_mod_old = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
        self.proj_func_old = gpu.get_function(proj_mod_old, "gptq_matvec")

        fast_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_gemv_fast.cubin")
        self.proj_func_fast = gpu.get_function(fast_mod, "gptq_gemv_fast")

        activate_mod = gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = gpu.get_function(activate_mod, "activate")

        lm_head_mod = gpu.load_cubin(f"{KERNEL_DIR}/lm_head.cubin")
        self.lm_head_func = gpu.get_function(lm_head_mod, "lm_head")

        # Note: recurrence.cubin implements a different DeltaNet variant (output before
        # state update, undecayed kT@S). The correct recurrence uses vectorized numpy.
        # Kernel loaded for future use if the PTX is corrected.
        recurrence_mod = gpu.load_cubin(f"{KERNEL_DIR}/recurrence.cubin")
        self.recurrence_func = gpu.get_function(recurrence_mod, "recurrence")
        gpu.set_max_dynamic_shared(self.recurrence_func, 67080)

        print("    embed_f16, norm, gptq_matvec_tc, gptq_gemv_fast, activate, lm_head, recurrence: loaded")

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

        print(f"    GPU buffers allocated")

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
        print(f"    {NUM_LAYERS * 2 + 1} norm weights preloaded (with +1.0 for Qwen3NextRMSNorm)")

    def _preload_deltanet_weights(self):
        """Preload all DeltaNet CPU-side weights to avoid repeated I/O."""
        print("  Preloading DeltaNet weights...")
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

        print(f"    DeltaNet weights preloaded for {len(self.dn_conv_w)} layers")

    def _init_deltanet_state(self):
        """Initialize DeltaNet state: S matrices and conv1d history per layer."""
        self.dn_S = {}        # layer_idx -> np.ndarray [48, 128, 128]
        self.dn_conv_buf = {} # layer_idx -> np.ndarray [10240, CONV_KERNEL_SIZE-1]

        for i in range(NUM_LAYERS):
            if self.layer_types[i] != "linear_attention":
                continue
            # S: state matrix per value head, shape [num_value_heads, value_dim, key_dim]
            # Note: state layout is [Hv, Dv, Dk] following the fused_gdn kernel convention
            self.dn_S[i] = np.zeros((NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=np.float32)
            # Conv buffer: store last (kernel_size-1) = 3 QKV values per channel
            self.dn_conv_buf[i] = np.zeros((10240, CONV_KERNEL_SIZE - 1), dtype=np.float32)

        print(f"    DeltaNet state initialized ({len(self.dn_S)} layers, "
              f"S shape={NUM_VALUE_HEADS}x{VALUE_HEAD_DIM}x{KEY_HEAD_DIM})")

    def _preload_qk_norms(self):
        """Preload Q/K RMSNorm weights for full attention layers.

        Qwen3MoeRMSNorm (plain w, init to ones) is used in code, but empirically
        the stored weights (~0.22 mean) suggest they may actually be Qwen3NextRMSNorm
        (init to zeros, needs +1.0). Try both and pick whichever works better.
        Currently: NO +1.0 (plain w).
        """
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
    # GPU kernel wrappers (same as generate_first_token.py)
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

        # Zero the output buffer (required for atomicAdd in gptq_gemv_fast)
        self.gpu.memset_d8(d_output, 0, N * 4, stream=stream)

        K_SPLITS = 16
        K_packed = K // 8
        k_packed_per_split = K_packed // K_SPLITS
        fast_smem = k_packed_per_split * 8 * 4  # k_packed_per_split * 32

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
        PROF.start("mlp_norm")
        self.gpu_norm(d_residual, self.d_zero,
                      self.post_attn_norms[layer_idx], self.d_norm_out)
        self.gpu.synchronize()
        PROF.stop()

        PROF.start("mlp_gate_up_proj")
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
        PROF.stop()

        PROF.start("mlp_activate")
        self.gpu_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)
        self.gpu.synchronize()
        PROF.stop()

        PROF.start("mlp_down_proj")
        self.gpu_gptq_matvec(f"{prefix}.mlp.down_proj",
                              self.d_act, self.d_down,
                              INTERMEDIATE_SIZE, HIDDEN_DIM)
        self.gpu.synchronize()
        PROF.stop()

    def run_full_attention(self, layer_idx: int, d_normed, position: int):
        """Full attention with KV cache and RoPE."""
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        # Q projection: 5120 -> 12288 (24 heads * 256 dim * 2 for Q + gate)
        PROF.start("fa_qkv_proj_gpu")
        self.gpu_gptq_matvec(f"{prefix}.q_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 12288)
        self.gpu.synchronize()
        q_gate_raw = download_f32(self.gpu, self.d_attn_scratch1, 12288)

        # K projection: 5120 -> 1024 (4 heads * 256 dim)
        self.gpu_gptq_matvec(f"{prefix}.k_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        k_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)

        # V projection: 5120 -> 1024 (4 heads * 256 dim)
        self.gpu_gptq_matvec(f"{prefix}.v_proj",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 1024)
        self.gpu.synchronize()
        v_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)
        PROF.stop()

        # Split q_proj output into Q and gate
        PROF.start("fa_attn_cpu")
        q_gate_heads = q_gate_raw.reshape(24, 512)  # [num_q_heads, head_dim * 2]
        q_heads = q_gate_heads[:, :256].copy()       # [24, 256] -- Q
        gate_heads = q_gate_heads[:, 256:].copy()    # [24, 256] -- gate
        gate_flat = gate_heads.flatten()              # [6144] -- will be applied after attention

        # Apply QK RMSNorm per head (Qwen3NextRMSNorm with 1+w, already loaded)
        k_heads = k_raw.reshape(4, 256)   # [num_kv_heads, head_dim]
        q_norm_w = self.q_norm_weights[layer_idx]  # [256] with +1.0
        k_norm_w = self.k_norm_weights[layer_idx]  # [256] with +1.0
        for h in range(24):
            rms = np.sqrt(np.mean(q_heads[h] ** 2) + self.epsilon)
            q_heads[h] = (q_heads[h] / rms) * q_norm_w
        for h in range(4):
            rms = np.sqrt(np.mean(k_heads[h] ** 2) + self.epsilon)
            k_heads[h] = (k_heads[h] / rms) * k_norm_w
        q_raw = q_heads.flatten()
        k_raw = k_heads.flatten()

        # Apply RoPE to Q and K
        q_rope, k_rope = process_qkv_with_rope(q_raw, k_raw, position)

        # Store K (after RoPE) and V (no RoPE) in KV cache
        v_2d = v_raw.reshape(4, 256)
        self.kv_cache.store(layer_idx, position, k_rope, v_2d)

        # Get cached keys and values up to current position
        k_cache, v_cache = self.kv_cache.get(layer_idx, position)

        # Compute attention
        attended = attention_prefill(q_rope, k_cache, v_cache, position)

        # Apply output gate: attn_output = attn_output * sigmoid(gate)
        gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_flat.clip(-80, 80)))
        attended = attended * gate_sigmoid  # [6144] element-wise
        PROF.stop()

        # O projection: 6144 -> 5120
        PROF.start("fa_out_proj_gpu")
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attended.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.o_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()
        PROF.stop()

    def run_deltanet(self, layer_idx: int, d_normed):
        """Run DeltaNet with persistent state S and conv buffer.

        Unlike generate_first_token.py which assumes S=0, this version:
        - Reads S from self.dn_S[layer_idx]
        - Updates conv buffer with new QKV values
        - Applies conv1d using the buffer
        - Runs the DeltaNet recurrence with the existing state
        - Writes updated S back
        """
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        # --- QKV projection (GPTQ, 5120 -> 10240) ---
        PROF.start("dn_proj_qkv")
        self.gpu_gptq_matvec(f"{prefix}.in_proj_qkv",
                              d_normed, self.d_attn_scratch1,
                              HIDDEN_DIM, 10240)

        # --- Z projection (GPTQ, 5120 -> 6144) ---
        self.gpu_gptq_matvec(f"{prefix}.in_proj_z",
                              d_normed, self.d_attn_scratch2,
                              HIDDEN_DIM, 6144)
        self.gpu.synchronize()
        PROF.stop()

        PROF.start("dn_download_proj")
        qkv = download_f32(self.gpu, self.d_attn_scratch1, 10240)
        z = download_f32(self.gpu, self.d_attn_scratch2, 6144)
        normed_cpu = download_f32(self.gpu, d_normed, HIDDEN_DIM)
        PROF.stop()

        # --- Conv1d with history buffer ---
        PROF.start("dn_conv1d_cpu")
        conv_buf = self.dn_conv_buf[layer_idx]  # [10240, 3]
        conv_w = self.dn_conv_w[layer_idx]       # [10240, 1, 4]

        # Build the full conv window: [history..., current] = [10240, 4]
        conv_window = np.concatenate([conv_buf, qkv[:, np.newaxis]], axis=1)  # [10240, 4]

        # Apply depthwise conv: sum over kernel positions
        qkv_conv = np.sum(conv_window * conv_w[:, 0, :], axis=1)  # [10240]

        # Update conv buffer: shift left by 1, append current
        self.dn_conv_buf[layer_idx] = conv_window[:, 1:]  # [10240, 3] -- drop oldest

        # Apply SiLU activation after conv1d
        qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv.clip(-80, 80))))
        PROF.stop()

        # Split QKV after conv + SiLU
        PROF.start("dn_pre_recurrence_cpu")
        q = qkv_conv[:2048].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        k = qkv_conv[2048:4096].reshape(NUM_KEY_HEADS, KEY_HEAD_DIM)
        v = qkv_conv[4096:].reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)

        # --- Q and K L2 normalization + Q scaling ---
        for hi in range(NUM_KEY_HEADS):
            q_norm = np.sqrt(np.sum(q[hi] ** 2) + 1e-6)
            q[hi] = q[hi] / q_norm
            k_norm = np.sqrt(np.sum(k[hi] ** 2) + 1e-6)
            k[hi] = k[hi] / k_norm
        q = q * (1.0 / np.sqrt(float(KEY_HEAD_DIM)))  # scale Q by 1/sqrt(128)

        # --- Beta (from in_proj_b) ---
        beta = 1.0 / (1.0 + np.exp(-(self.dn_b_weight[layer_idx] @ normed_cpu).clip(-80, 80)))

        # --- dt/decay (from in_proj_a) ---
        dt = self.dn_a_weight[layer_idx] @ normed_cpu + self.dn_dt_bias[layer_idx]
        dt = np.log1p(np.exp(dt.clip(-20, 20)))  # softplus

        # Gate = exp(-A_exp * dt) where A_exp = exp(A_log) (positive)
        A_exp = np.exp(self.dn_A_log[layer_idx])  # [48], positive
        decay = np.exp(-A_exp * dt)  # [48], in (0, 1)
        PROF.stop()

        # --- DeltaNet recurrence with state (vectorized) ---
        PROF.start("dn_recurrence_cpu")
        S = self.dn_S[layer_idx]  # [48, 128, 128] = [Hv, Dv, Dk]

        # Expand k, q from [16, Dk] to [48, Dk] via GQA repeat
        k_exp = np.repeat(k, HEADS_PER_KEY, axis=0)  # [48, Dk]
        q_exp = np.repeat(q, HEADS_PER_KEY, axis=0)  # [48, Dk]

        # Step 1: Apply decay -- S[h] *= decay[h] for all heads
        S *= decay[:, None, None]  # [48, Dv, Dk]

        # Step 2: Compute kv_mem = S @ k for all heads: [48, Dv]
        kv_mem = np.einsum('hvk,hk->hv', S, k_exp)

        # Step 3: Delta
        delta = (v - kv_mem) * beta[:, None]  # [48, Dv]

        # Step 4: State update: S[h] += outer(delta[h], k[h])
        S += np.einsum('hv,hk->hvk', delta, k_exp)

        # Step 5: Output: output[h] = S[h] @ q[h]
        output_heads = np.einsum('hvk,hk->hv', S, q_exp)

        self.dn_S[layer_idx] = S
        PROF.stop()

        # --- Group norm per head ---
        PROF.start("dn_groupnorm_cpu")
        head_norm_w = self.dn_head_norm_w[layer_idx]
        for h in range(NUM_VALUE_HEADS):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w
        PROF.stop()

        # --- Gate with Z ---
        PROF.start("dn_gate_z_cpu")
        z_heads = z.reshape(NUM_VALUE_HEADS, VALUE_HEAD_DIM)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu
        PROF.stop()

        # --- Output projection ---
        PROF.start("dn_out_proj_gpu")
        attn_output_flat = output_heads.flatten()
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output_flat.ctypes.data_as(ctypes.c_void_p),
                             6144 * 4)
        self.gpu_gptq_matvec(f"{prefix}.out_proj",
                              self.d_attn_scratch2, self.d_attn_out,
                              6144, HIDDEN_DIM)
        self.gpu.synchronize()
        PROF.stop()

    def run_layer(self, layer_idx: int, x: np.ndarray, position: int) -> np.ndarray:
        """Run one transformer layer with full attention/DeltaNet."""
        layer_type = self.layer_types[layer_idx]
        gpu = self.gpu

        PROF.start("residual_copy")
        residual = x.copy()
        PROF.stop()

        # --- Input norm + Attention ---
        PROF.start("upload_x")
        d_x_gpu = upload_f32(gpu, x)
        PROF.stop()

        PROF.start("norm")
        self.gpu_norm(d_x_gpu, self.d_zero,
                      self.input_norms[layer_idx], self.d_norm_out)
        gpu.synchronize()
        PROF.stop()

        if layer_type == "full_attention":
            PROF.start("full_attention")
            self.run_full_attention(layer_idx, self.d_norm_out, position)
            PROF.stop()
            PROF.start("download_attn")
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
            PROF.stop()
        else:  # linear_attention
            PROF.start("deltanet")
            self.run_deltanet(layer_idx, self.d_norm_out)
            PROF.stop()
            PROF.start("download_attn")
            attn_out = download_f32(gpu, self.d_attn_out, HIDDEN_DIM)
            PROF.stop()

        PROF.start("residual_add")
        x = residual + attn_out
        residual = x.copy()
        PROF.stop()

        PROF.start("mem_free_x")
        gpu.mem_free(d_x_gpu)
        PROF.stop()

        # --- MLP sublayer ---
        PROF.start("upload_mlp_residual")
        gpu.memcpy_htod(self.d_residual,
                        x.ctypes.data_as(ctypes.c_void_p),
                        HIDDEN_DIM * 4)
        PROF.stop()

        PROF.start("mlp")
        self.run_mlp(layer_idx, self.d_residual)
        PROF.stop()

        PROF.start("download_mlp")
        mlp_out = download_f32(gpu, self.d_down, HIDDEN_DIM)
        PROF.stop()

        PROF.start("residual_add")
        x = x + mlp_out
        PROF.stop()

        return x

    def process_token(self, token_id: int, token_pos: int, token_text: str) -> np.ndarray:
        """Process a single token through all 64 layers."""
        PROF.start("embed")
        x = self.gpu_embed(token_id)
        PROF.stop()

        for layer_idx in range(NUM_LAYERS):
            x = self.run_layer(layer_idx, x, token_pos)

            if (layer_idx + 1) % 16 == 0:
                x_norm = np.linalg.norm(x)
                has_nan = np.any(np.isnan(x))
                has_inf = np.any(np.isinf(x))
                if has_nan or has_inf:
                    print(f"    Layer {layer_idx}: norm={x_norm:.4f} NaN={has_nan} Inf={has_inf} -- ABORTING")
                    return None

        # Print DeltaNet state statistics after processing this token
        if token_pos == 0 or token_pos == 4:  # first and last token
            s_norms = []
            for i in sorted(self.dn_S.keys())[:3]:  # first 3 DeltaNet layers
                s_norms.append(np.linalg.norm(self.dn_S[i]))
            print(f"    DeltaNet state norms (first 3 layers): {[f'{n:.4f}' for n in s_norms]}")

        return x

    def generate(self, prompt: str) -> tuple:
        """Process all prompt tokens, then generate the next token."""
        token_ids = self.tok.encode(prompt)
        print(f"\nPrompt: {prompt!r}")
        print(f"Token IDs: {token_ids}")
        for i, tid in enumerate(token_ids):
            print(f"  Token {i}: id={tid} '{self.tok.decode([tid])}'")

        # --- Prefill: process each prompt token ---
        banner("PREFILL: Processing prompt tokens")

        x = None
        for pos, tid in enumerate(token_ids):
            word = self.tok.decode([tid])
            t0 = time.monotonic()
            print(f"\n--- Token {pos+1}/{len(token_ids)}: '{word}' (id={tid}) ---")

            x = self.process_token(tid, pos, word)

            if x is None:
                print("  ERROR: NaN/Inf detected, aborting!")
                return -1, "<error>", None

            elapsed = time.monotonic() - t0
            x_norm = np.linalg.norm(x)
            print(f"  Done: norm={x_norm:.4f} time={elapsed:.2f}s")

        # --- Generate: decode the next token ---
        banner("GENERATE: Decoding next token")

        # Final RMSNorm
        PROF.start("final_norm")
        x_normed = rms_norm_cpu(x, self.final_norm_w, self.epsilon)
        PROF.stop()
        print(f"  Final norm: norm={np.linalg.norm(x_normed):.4f}")

        # lm_head projection (GPU kernel)
        print(f"  Computing lm_head (248320 x 5120 FP16 matmul on GPU)...")
        t_lm = time.monotonic()

        PROF.start("lm_head")
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
        PROF.stop()

        t_lm_done = time.monotonic()
        print(f"  lm_head: {t_lm_done - t_lm:.2f}s")

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

        # Check where "Paris" / " Paris" rank
        for candidate in ["Paris", " Paris"]:
            cand_ids = self.tok.encode(candidate)
            for cid in cand_ids:
                rank = int(np.sum(logits > logits[cid])) + 1
                cword = self.tok.decode([cid])
                print(f"  '{candidate}' (id={cid}, decoded='{cword}'): logit={logits[cid]:.3f}, rank={rank}")

        return token_out, token_text, logits

    def close(self):
        self.gpu.close()
        self.model.close()


def main() -> int:
    engine = PrefillEngine()

    prompt = "The capital of France is"
    t0 = time.monotonic()
    token_out, token_text, logits = engine.generate(prompt)
    total = time.monotonic() - t0

    banner("RESULT")
    print(f"  Prompt: {prompt!r}")
    print(f"  Generated: '{token_text}' (token_id={token_out})")
    print(f"  Expected: 'Paris' or ' Paris'")
    print(f"  Total time: {total:.2f}s")

    # Check if it's Paris
    paris_ids = []
    for candidate in ["Paris", " Paris"]:
        ids = engine.tok.encode(candidate)
        paris_ids.extend(ids)
    is_paris = token_out in paris_ids
    print(f"  Match: {'YES' if is_paris else 'NO'}")

    PROF.report()

    engine.close()
    return 0 if is_paris else 1


if __name__ == "__main__":
    raise SystemExit(main())
