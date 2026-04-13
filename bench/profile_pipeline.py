#!/usr/bin/env python3
"""
Lithos Pipeline Profiler -- Qwen 3.5-27B on GH200

Profiles a complete first-token forward pass through the inference pipeline,
measuring time per kernel type using CUDA events for GPU timing and
time.perf_counter for CPU-side timing.

Run:
    python3 /home/ubuntu/lithos/bench/profile_pipeline.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import sys
import time
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_driver import CUDADriver, CUdeviceptr, CUresult
from loader import LithosModel
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
VOCAB_SIZE = 248320

# CUDA event type
CUevent = c_void_p


# ---------------------------------------------------------------------------
# GPU Timer using CUDA events
# ---------------------------------------------------------------------------
class GPUTimer:
    """CUDA event-based GPU timer for accurate kernel timing."""

    def __init__(self, lib):
        self._lib = lib
        self.start_ev = CUevent()
        self.stop_ev = CUevent()
        self._check("cuEventCreate", lib.cuEventCreate(byref(self.start_ev), 0))
        self._check("cuEventCreate", lib.cuEventCreate(byref(self.stop_ev), 0))

    def _check(self, name, result):
        if result != 0:
            raise RuntimeError(f"{name} failed: {result}")

    def start(self):
        self._check("cuEventRecord", self._lib.cuEventRecord(self.start_ev, None))

    def stop(self):
        self._check("cuEventRecord", self._lib.cuEventRecord(self.stop_ev, None))
        self._check("cuEventSynchronize", self._lib.cuEventSynchronize(self.stop_ev))

    def elapsed_ms(self) -> float:
        ms = c_float()
        self._check("cuEventElapsedTime",
                     self._lib.cuEventElapsedTime(byref(ms), self.start_ev, self.stop_ev))
        return ms.value

    def destroy(self):
        self._lib.cuEventDestroy_v2(self.start_ev)
        self._lib.cuEventDestroy_v2(self.stop_ev)


# ---------------------------------------------------------------------------
# Helpers (from generate_first_token.py)
# ---------------------------------------------------------------------------
def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def load_norm_weight(model, name):
    ti = model.weight_info(name)
    raw = bytes(model.weight_bytes(name))
    if ti.dtype == "BF16":
        return bf16_to_f32(raw)
    elif ti.dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        return np.frombuffer(raw, dtype=np.float32).copy()


def upload_f32(gpu, data):
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu, dptr, size):
    out = np.zeros(size, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, size * 4)
    return out


# ---------------------------------------------------------------------------
# Setup CUDA event API on the raw libcuda handle
# ---------------------------------------------------------------------------
def setup_event_api(lib):
    """Add CUDA event function prototypes to the driver library."""
    lib.cuEventCreate.argtypes = [POINTER(CUevent), c_uint]
    lib.cuEventCreate.restype = CUresult
    lib.cuEventRecord.argtypes = [CUevent, c_void_p]
    lib.cuEventRecord.restype = CUresult
    lib.cuEventSynchronize.argtypes = [CUevent]
    lib.cuEventSynchronize.restype = CUresult
    lib.cuEventElapsedTime.argtypes = [POINTER(c_float), CUevent, CUevent]
    lib.cuEventElapsedTime.restype = CUresult
    lib.cuEventDestroy_v2.argtypes = [CUevent]
    lib.cuEventDestroy_v2.restype = CUresult


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
class PipelineProfiler:
    """Instruments one full forward pass and collects timing data."""

    def __init__(self):
        print("Initializing profiler...")
        t0 = time.perf_counter()

        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self.layer_types = self.model.config.layer_types

        # Setup CUDA event API on the underlying library
        self._lib = self.gpu._lib
        setup_event_api(self._lib)

        # Create reusable timer
        self.timer = GPUTimer(self._lib)

        # Load kernels
        embed_mod = self.gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
        self.embed_func = self.gpu.get_function(embed_mod, "embed_f16")

        norm_mod = self.gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = self.gpu.get_function(norm_mod, "norm")

        proj_mod = self.gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
        self.proj_func = self.gpu.get_function(proj_mod, "gptq_matvec")

        activate_mod = self.gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = self.gpu.get_function(activate_mod, "activate")

        # Allocate GPU buffers
        self.d_x = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_residual = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_out = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_weight = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_gate = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_up = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_act = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_down = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_attn_scratch1 = self.gpu.mem_alloc(12288 * 4)
        self.d_attn_scratch2 = self.gpu.mem_alloc(6144 * 4)
        self.d_attn_out = self.gpu.mem_alloc(HIDDEN_DIM * 4)

        self.d_zero = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.gpu.memcpy_htod(self.d_zero,
                             np.zeros(HIDDEN_DIM, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)

        # Preload norms
        self.input_norms = []
        self.post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            self.input_norms.append(
                load_norm_weight(self.model, f"{prefix}.input_layernorm.weight"))
            self.post_attn_norms.append(
                load_norm_weight(self.model, f"{prefix}.post_attention_layernorm.weight"))
        self.final_norm_w = load_norm_weight(self.model, "model.language_model.norm.weight")

        # Timing accumulators
        self.reset_timers()

        elapsed = time.perf_counter() - t0
        print(f"Init done in {elapsed:.2f}s -- {self.gpu.device_name}")

    def reset_timers(self):
        """Reset all timing accumulators."""
        # GPU times (ms) -- measured with CUDA events
        self.gpu_embed_ms = 0.0
        self.gpu_norm_ms = []        # per-call
        self.gpu_proj_ms = []        # per-call, with (K, N, label) metadata
        self.gpu_activate_ms = []    # per-call
        self.gpu_deltanet_recurrence_ms = []  # per-call (CPU-side, no kernel)
        self.gpu_lm_head_ms = 0.0

        # CPU overhead times (s)
        self.cpu_memcpy_s = 0.0
        self.cpu_launch_overhead_s = 0.0
        self.cpu_numpy_s = 0.0       # numpy/python compute (deltanet, residual adds, etc.)

        # Counts
        self.norm_count = 0
        self.proj_count = 0
        self.activate_count = 0
        self.launch_count = 0

    # ------------------------------------------------------------------
    # Timed kernel wrappers
    # ------------------------------------------------------------------

    def _timed_launch(self, func, grid, block, args, shared_mem=0):
        """Launch a kernel and measure both GPU time (CUDA events) and CPU launch overhead."""
        cpu_t0 = time.perf_counter()

        self.timer.start()
        self.gpu.launch(func, grid=grid, block=block, args=args, shared_mem=shared_mem)
        self.timer.stop()

        cpu_t1 = time.perf_counter()
        gpu_ms = self.timer.elapsed_ms()

        self.cpu_launch_overhead_s += (cpu_t1 - cpu_t0) - (gpu_ms / 1000.0)
        self.launch_count += 1
        return gpu_ms

    def timed_embed(self, token_id):
        embed_ptr = self.model.weight_info("model.language_model.embed_tokens.weight").ptr
        BLOCK = 256
        GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))
        gpu_ms = self._timed_launch(
            self.embed_func,
            grid=(GRID, 1, 1), block=(BLOCK, 1, 1),
            args=[
                ctypes.c_uint32(token_id),
                ctypes.c_uint64(embed_ptr),
                ctypes.c_uint64(self.d_x.value),
            ],
        )
        self.gpu_embed_ms = gpu_ms
        return download_f32(self.gpu, self.d_x, HIDDEN_DIM)

    def timed_norm(self, d_input, d_residual, norm_w, d_output):
        cpu_t0 = time.perf_counter()
        self.gpu.memcpy_htod(self.d_norm_weight,
                             norm_w.ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)
        cpu_t1 = time.perf_counter()
        self.cpu_memcpy_s += cpu_t1 - cpu_t0

        gpu_ms = self._timed_launch(
            self.norm_func,
            grid=(1, 1, 1), block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_input.value),
                ctypes.c_uint64(d_residual.value),
                ctypes.c_uint64(self.d_norm_weight.value),
                ctypes.c_uint64(d_output.value),
                ctypes.c_float(self.epsilon),
            ],
            shared_mem=128,
        )
        self.gpu_norm_ms.append(gpu_ms)
        self.norm_count += 1

    def timed_gptq_matvec(self, prefix, d_input, d_output, K, N, label=""):
        # Zero output
        cpu_t0 = time.perf_counter()
        self.gpu.memcpy_htod(d_output,
                             np.zeros(N, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             N * 4)
        cpu_t1 = time.perf_counter()
        self.cpu_memcpy_s += cpu_t1 - cpu_t0

        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr

        gpu_ms = self._timed_launch(
            self.proj_func,
            grid=(N, 1, 1), block=(256, 1, 1),
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
        self.gpu_proj_ms.append((gpu_ms, K, N, label))
        self.proj_count += 1

    def timed_activate(self, d_gate, d_up, d_output, size):
        GRID_ACT = max(1, math.ceil(size / 256))
        gpu_ms = self._timed_launch(
            self.activate_func,
            grid=(GRID_ACT, 1, 1), block=(256, 1, 1),
            args=[
                ctypes.c_uint64(d_up.value),
                ctypes.c_uint64(d_gate.value),
                ctypes.c_uint64(d_output.value),
            ],
        )
        self.gpu_activate_ms.append(gpu_ms)
        self.activate_count += 1

    # ------------------------------------------------------------------
    # Layer execution (instrumented)
    # ------------------------------------------------------------------

    def run_mlp(self, layer_idx, d_residual):
        prefix = f"model.language_model.layers.{layer_idx}"

        self.timed_norm(d_residual, self.d_zero,
                        self.post_attn_norms[layer_idx], self.d_norm_out)

        self.timed_gptq_matvec(f"{prefix}.mlp.gate_proj",
                               self.d_norm_out, self.d_gate,
                               HIDDEN_DIM, INTERMEDIATE_SIZE, label="gate 5120->17408")
        self.timed_gptq_matvec(f"{prefix}.mlp.up_proj",
                               self.d_norm_out, self.d_up,
                               HIDDEN_DIM, INTERMEDIATE_SIZE, label="up 5120->17408")

        self.timed_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)

        self.timed_gptq_matvec(f"{prefix}.mlp.down_proj",
                               self.d_act, self.d_down,
                               INTERMEDIATE_SIZE, HIDDEN_DIM, label="down 17408->5120")

        self.gpu.synchronize()

    def run_full_attention(self, layer_idx, d_normed):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        # V proj: 5120 -> 1024
        self.timed_gptq_matvec(f"{prefix}.v_proj",
                               d_normed, self.d_attn_scratch1,
                               HIDDEN_DIM, 1024, label="v_proj 5120->1024")
        self.gpu.synchronize()

        # CPU: GQA expansion
        cpu_t0 = time.perf_counter()
        v_raw = download_f32(self.gpu, self.d_attn_scratch1, 1024)
        v_heads = v_raw.reshape(4, 256)
        v_expanded = np.repeat(v_heads, 6, axis=0).flatten()
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             v_expanded.ctypes.data_as(ctypes.c_void_p), 6144 * 4)
        cpu_t1 = time.perf_counter()
        self.cpu_numpy_s += cpu_t1 - cpu_t0

        # O proj: 6144 -> 5120
        self.timed_gptq_matvec(f"{prefix}.o_proj",
                               self.d_attn_scratch2, self.d_attn_out,
                               6144, HIDDEN_DIM, label="o_proj 6144->5120")
        self.gpu.synchronize()

    def run_deltanet(self, layer_idx, d_normed):
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        # QKV: 5120 -> 10240
        self.timed_gptq_matvec(f"{prefix}.in_proj_qkv",
                               d_normed, self.d_attn_scratch1,
                               HIDDEN_DIM, 10240, label="qkv 5120->10240")

        # Z: 5120 -> 6144
        self.timed_gptq_matvec(f"{prefix}.in_proj_z",
                               d_normed, self.d_attn_scratch2,
                               HIDDEN_DIM, 6144, label="z 5120->6144")
        self.gpu.synchronize()

        # CPU-side recurrence (conv1d, beta, dt, decay, recurrence, group_norm, gate)
        cpu_t0 = time.perf_counter()

        qkv = download_f32(self.gpu, self.d_attn_scratch1, 10240)
        z = download_f32(self.gpu, self.d_attn_scratch2, 6144)
        normed_cpu = download_f32(self.gpu, d_normed, HIDDEN_DIM)

        # Conv1d
        conv_w_raw = bytes(self.model.weight_bytes(f"{prefix}.conv1d.weight"))
        conv_w = bf16_to_f32(conv_w_raw).reshape(10240, 1, 4)
        qkv_conv = qkv * conv_w[:, 0, 3]
        q = qkv_conv[:2048].reshape(16, 128)
        k = qkv_conv[2048:4096].reshape(16, 128)
        v = qkv_conv[4096:].reshape(48, 128)

        # Beta
        a_w_raw = bytes(self.model.weight_bytes(f"{prefix}.in_proj_a.weight"))
        a_weight = bf16_to_f32(a_w_raw).reshape(48, HIDDEN_DIM)
        beta = 1.0 / (1.0 + np.exp(-(a_weight @ normed_cpu).clip(-80, 80)))

        # dt
        b_w_raw = bytes(self.model.weight_bytes(f"{prefix}.in_proj_b.weight"))
        b_weight = bf16_to_f32(b_w_raw).reshape(48, HIDDEN_DIM)
        dt_bias_raw = bytes(self.model.weight_bytes(f"{prefix}.dt_bias"))
        dt_bias = bf16_to_f32(dt_bias_raw)
        dt = b_weight @ normed_cpu + dt_bias
        dt = np.log1p(np.exp(dt.clip(-20, 20)))

        # Decay
        a_log_raw = bytes(self.model.weight_bytes(f"{prefix}.A_log"))
        A_log = np.frombuffer(a_log_raw, dtype=np.float32).copy()
        A = -np.exp(A_log)
        decay = np.exp(A * dt)

        # Recurrence
        value_heads_per_key = 3
        output_heads = np.zeros((48, 128), dtype=np.float32)
        for h in range(48):
            key_idx = h // value_heads_per_key
            qk_dot = np.dot(q[key_idx], k[key_idx])
            output_heads[h] = beta[h] * qk_dot * v[h]

        # Group norm
        norm_w_raw = bytes(self.model.weight_bytes(f"{prefix}.norm.weight"))
        head_norm_w = np.frombuffer(norm_w_raw, dtype=np.float32).copy()
        for h in range(48):
            rms = np.sqrt(np.mean(output_heads[h] ** 2) + self.epsilon)
            output_heads[h] = (output_heads[h] / rms) * head_norm_w

        # Gate with Z
        z_heads = z.reshape(48, 128)
        z_silu = z_heads * (1.0 / (1.0 + np.exp(-z_heads.clip(-80, 80))))
        output_heads = output_heads * z_silu
        attn_output_flat = output_heads.flatten()

        cpu_t1 = time.perf_counter()
        self.gpu_deltanet_recurrence_ms.append((cpu_t1 - cpu_t0) * 1000.0)

        # Upload and o_proj
        cpu_t2 = time.perf_counter()
        self.gpu.memcpy_htod(self.d_attn_scratch2,
                             attn_output_flat.ctypes.data_as(ctypes.c_void_p), 6144 * 4)
        cpu_t3 = time.perf_counter()
        self.cpu_memcpy_s += cpu_t3 - cpu_t2

        self.timed_gptq_matvec(f"{prefix}.out_proj",
                               self.d_attn_scratch2, self.d_attn_out,
                               6144, HIDDEN_DIM, label="out_proj 6144->5120")
        self.gpu.synchronize()

    def run_layer(self, layer_idx, x):
        layer_type = self.layer_types[layer_idx]
        residual = x.copy()

        if layer_type == "full_attention":
            d_x_gpu = upload_f32(self.gpu, x)
            self.timed_norm(d_x_gpu, self.d_zero,
                            self.input_norms[layer_idx], self.d_norm_out)
            self.gpu.synchronize()
            self.run_full_attention(layer_idx, self.d_norm_out)
            cpu_t0 = time.perf_counter()
            attn_out = download_f32(self.gpu, self.d_attn_out, HIDDEN_DIM)
            x = residual + attn_out
            residual = x.copy()
            cpu_t1 = time.perf_counter()
            self.cpu_numpy_s += cpu_t1 - cpu_t0
            self.gpu.mem_free(d_x_gpu)

        elif layer_type == "linear_attention":
            d_x_gpu = upload_f32(self.gpu, x)
            self.timed_norm(d_x_gpu, self.d_zero,
                            self.input_norms[layer_idx], self.d_norm_out)
            self.gpu.synchronize()
            self.run_deltanet(layer_idx, self.d_norm_out)
            cpu_t0 = time.perf_counter()
            attn_out = download_f32(self.gpu, self.d_attn_out, HIDDEN_DIM)
            x = residual + attn_out
            residual = x.copy()
            cpu_t1 = time.perf_counter()
            self.cpu_numpy_s += cpu_t1 - cpu_t0
            self.gpu.mem_free(d_x_gpu)

        # MLP
        self.gpu.memcpy_htod(self.d_residual,
                             x.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)
        self.run_mlp(layer_idx, self.d_residual)

        cpu_t0 = time.perf_counter()
        mlp_out = download_f32(self.gpu, self.d_down, HIDDEN_DIM)
        x = x + mlp_out
        cpu_t1 = time.perf_counter()
        self.cpu_numpy_s += cpu_t1 - cpu_t0

        return x

    def run_lm_head(self, x_normed):
        """lm_head: FP16 dense matmul on CPU (matches generate_first_token.py)."""
        t0 = time.perf_counter()
        lm_head_raw = bytes(self.model.weight_bytes("lm_head.weight"))
        CHUNK = 8192
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for start in range(0, VOCAB_SIZE, CHUNK):
            end = min(start + CHUNK, VOCAB_SIZE)
            chunk_size = end - start
            offset = start * HIDDEN_DIM * 2
            chunk_bytes = chunk_size * HIDDEN_DIM * 2
            w_chunk = np.frombuffer(
                lm_head_raw[offset:offset + chunk_bytes],
                dtype=np.float16
            ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
            logits[start:end] = w_chunk @ x_normed
        t1 = time.perf_counter()
        self.gpu_lm_head_ms = (t1 - t0) * 1000.0
        return logits

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def profile_forward_pass(self):
        """Run a complete first-token forward pass and collect timing."""
        prompt = "The capital of France is"
        token_ids = self.tok.encode(prompt)
        tid = token_ids[0]

        print(f"\nProfiling forward pass (phase 3 = complete pipeline)")
        print(f"  Token: id={tid} '{self.tok.decode([tid])}'")

        self.reset_timers()
        total_t0 = time.perf_counter()

        # 1. Embed
        x = self.timed_embed(tid)

        # 2. All 64 layers
        for layer_idx in range(NUM_LAYERS):
            x = self.run_layer(layer_idx, x)
            if (layer_idx + 1) % 16 == 0:
                print(f"  ... layer {layer_idx + 1}/64 done")

        # 3. Final norm (CPU)
        cpu_t0 = time.perf_counter()
        rms = np.sqrt(np.mean(x ** 2) + self.epsilon)
        x_normed = (x / rms) * self.final_norm_w
        cpu_t1 = time.perf_counter()
        self.cpu_numpy_s += cpu_t1 - cpu_t0

        # 4. lm_head
        logits = self.run_lm_head(x_normed)

        total_t1 = time.perf_counter()
        total_ms = (total_t1 - total_t0) * 1000.0

        # Verify output
        token_out = int(np.argmax(logits))
        token_text = self.tok.decode([token_out])
        print(f"  Output: token={token_out} '{token_text}'")

        return total_ms

    # ------------------------------------------------------------------
    # Analysis and reporting
    # ------------------------------------------------------------------

    def report(self, total_ms):
        """Print the profiling summary table."""
        # Aggregate projections by size
        proj_by_size = {}
        for ms, K, N, label in self.gpu_proj_ms:
            key = f"{K}->{N}"
            if key not in proj_by_size:
                proj_by_size[key] = {"ms_list": [], "label": label, "K": K, "N": N}
            proj_by_size[key]["ms_list"].append(ms)

        # Count layer types
        n_full_attn = sum(1 for t in self.layer_types if t == "full_attention")
        n_deltanet = sum(1 for t in self.layer_types if t == "linear_attention")

        # Per-call averages
        avg_norm_ms = np.mean(self.gpu_norm_ms) if self.gpu_norm_ms else 0
        total_norm_ms = sum(self.gpu_norm_ms)

        avg_proj_ms = np.mean([m for m, _, _, _ in self.gpu_proj_ms]) if self.gpu_proj_ms else 0
        total_proj_ms = sum(m for m, _, _, _ in self.gpu_proj_ms)

        avg_activate_ms = np.mean(self.gpu_activate_ms) if self.gpu_activate_ms else 0
        total_activate_ms = sum(self.gpu_activate_ms)

        avg_deltanet_ms = np.mean(self.gpu_deltanet_recurrence_ms) if self.gpu_deltanet_recurrence_ms else 0
        total_deltanet_ms = sum(self.gpu_deltanet_recurrence_ms)

        total_kernel_gpu_ms = (self.gpu_embed_ms + total_norm_ms +
                               total_proj_ms + total_activate_ms)
        total_cpu_overhead_ms = (self.cpu_memcpy_s + self.cpu_launch_overhead_s +
                                 self.cpu_numpy_s) * 1000.0

        tps = 1000.0 / total_ms if total_ms > 0 else 0

        print()
        print("LITHOS PIPELINE PROFILE -- Qwen 3.5-27B, GH200")
        print("=" * 64)
        print(f"Embed:               {self.gpu_embed_ms:8.2f} ms")
        print(f"Per-layer norm:       {avg_norm_ms:8.4f} ms (x{len(self.gpu_norm_ms):3d} = {total_norm_ms:8.2f} ms total)")
        print(f"Per-layer projection: {avg_proj_ms:8.4f} ms (x{len(self.gpu_proj_ms):3d} = {total_proj_ms:8.2f} ms total)")
        print(f"Per-layer activate:   {avg_activate_ms:8.4f} ms (x{len(self.gpu_activate_ms):3d} = {total_activate_ms:8.2f} ms total)")
        print(f"DeltaNet recurrence:  {avg_deltanet_ms:8.2f} ms (x{len(self.gpu_deltanet_recurrence_ms):3d} = {total_deltanet_ms:8.2f} ms total)")
        print(f"lm_head:              {self.gpu_lm_head_ms:8.2f} ms")
        print(f"Total forward pass:   {total_ms:8.2f} ms")
        print(f"Tokens per second:    {tps:8.2f}")

        print()
        print("TIME BREAKDOWN")
        print("-" * 64)
        print(f"  GPU kernel execution:   {total_kernel_gpu_ms:10.2f} ms  ({100*total_kernel_gpu_ms/total_ms:5.1f}%)")
        print(f"  CPU overhead (memcpy):  {self.cpu_memcpy_s*1000:10.2f} ms  ({100*self.cpu_memcpy_s*1000/total_ms:5.1f}%)")
        print(f"  CPU overhead (launch):  {self.cpu_launch_overhead_s*1000:10.2f} ms  ({100*self.cpu_launch_overhead_s*1000/total_ms:5.1f}%)")
        print(f"  CPU numpy/python:       {self.cpu_numpy_s*1000:10.2f} ms  ({100*self.cpu_numpy_s*1000/total_ms:5.1f}%)")
        print(f"  DeltaNet CPU recurrence:{total_deltanet_ms:10.2f} ms  ({100*total_deltanet_ms/total_ms:5.1f}%)")
        print(f"  lm_head (CPU matmul):   {self.gpu_lm_head_ms:10.2f} ms  ({100*self.gpu_lm_head_ms/total_ms:5.1f}%)")
        accounted = total_kernel_gpu_ms + self.cpu_memcpy_s*1000 + self.cpu_launch_overhead_s*1000 + self.cpu_numpy_s*1000 + total_deltanet_ms + self.gpu_lm_head_ms
        print(f"  Unaccounted:            {total_ms - accounted:10.2f} ms  ({100*(total_ms - accounted)/total_ms:5.1f}%)")

        print()
        print("PROJECTION BREAKDOWN BY SIZE")
        print("-" * 64)
        print(f"  {'Size':<22s} {'Count':>5s} {'Avg ms':>10s} {'Total ms':>10s} {'% of proj':>10s}")
        for key in sorted(proj_by_size.keys(), key=lambda k: -np.mean(proj_by_size[k]["ms_list"])):
            info = proj_by_size[key]
            avg = np.mean(info["ms_list"])
            total = sum(info["ms_list"])
            pct = 100 * total / total_proj_ms if total_proj_ms > 0 else 0
            print(f"  {key:<22s} {len(info['ms_list']):5d} {avg:10.4f} {total:10.2f} {pct:9.1f}%")

        print()
        print("KERNEL LAUNCH OVERHEAD")
        print("-" * 64)
        print(f"  Total cuLaunchKernel calls: {self.launch_count}")
        if self.launch_count > 0:
            avg_launch_us = (self.cpu_launch_overhead_s / self.launch_count) * 1e6
            print(f"  Avg CPU launch overhead:    {avg_launch_us:.1f} us/call")
            print(f"  Total launch overhead:      {self.cpu_launch_overhead_s*1000:.2f} ms")

        print()

    def close(self):
        self.timer.destroy()
        self.gpu.close()
        self.model.close()


def main():
    profiler = PipelineProfiler()

    # Warmup run (1 layer only to warm caches)
    print("\nWarmup: running 1 layer...")
    prompt = "The capital of France is"
    token_ids = profiler.tok.encode(prompt)
    tid = token_ids[0]
    x = profiler.timed_embed(tid)
    x = profiler.run_layer(0, x)
    profiler.gpu.synchronize()
    print("Warmup done.")

    # Profile run
    total_ms = profiler.profile_forward_pass()

    # Report
    profiler.report(total_ms)

    profiler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
