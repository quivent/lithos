#!/usr/bin/env python3
"""
Lithos Benchmark Harness -- GH200 Inference Performance Tracker

Measures:
  1. Tokens/second (wall-clock, end-to-end)
  2. Time per token breakdown: GPU compute vs dispatch overhead vs CPU
  3. HBM bandwidth utilization per kernel class
  4. Total memory read/written per forward pass
  5. Per-kernel profiling via CUDA events (cuEventCreate/Record/Synchronize/ElapsedTime)
  6. Optional comparison mode against vLLM on the same prompt
  7. Appends results to a JSON log for tracking improvements over time

Output: human-readable table to stdout + machine-parseable JSON to
        /home/ubuntu/lithos/bench/results/<timestamp>.json
        and appended to /home/ubuntu/lithos/bench/results/history.jsonl

Hardware: GH200 -- 4 TB/s HBM3e peak, 132 SMs, 96 GB HBM3

Run:
    python3 /home/ubuntu/lithos/bench/harness.py
    python3 /home/ubuntu/lithos/bench/harness.py --compare-vllm
    python3 /home/ubuntu/lithos/bench/harness.py --runs 50
    python3 /home/ubuntu/lithos/bench/harness.py --json-only
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import numpy as np
import os
import statistics
import sys
import time
import urllib.request
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cuda_driver import CUDADriver, CUdeviceptr, CUresult
from loader import LithosModel
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(ROOT / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"
RESULTS_DIR = ROOT / "bench" / "results"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
VOCAB_SIZE = 248320
GROUP_SIZE = 128
NUM_BLOCKS = 132
THREADS_PER_BLOCK = 256

# Weight table layout for Carmack kernel
PTRS_PER_LAYER = 11
WEIGHT_TABLE_BYTES = NUM_LAYERS * PTRS_PER_LAYER * 8

# GH200 specs
GH200_PEAK_BW_TBS = 4.0      # TB/s HBM3e theoretical
GH200_SM_COUNT = 132

# vLLM
VLLM_URL = "http://localhost:8001"
VLLM_MODEL = "qwen3.5-27b"

# Prompt for all benchmarks
BENCH_PROMPT = "The capital of France is"

# CUDA event type
CUevent = c_void_p


# ---------------------------------------------------------------------------
# GPU Timer using CUDA events
# ---------------------------------------------------------------------------
class GPUTimer:
    """CUDA event pair for accurate GPU-side kernel timing."""

    def __init__(self, lib):
        self._lib = lib
        self.start_ev = CUevent()
        self.stop_ev = CUevent()
        self._check("cuEventCreate", lib.cuEventCreate(byref(self.start_ev), 0))
        self._check("cuEventCreate", lib.cuEventCreate(byref(self.stop_ev), 0))

    def _check(self, name, result):
        if result != 0:
            raise RuntimeError(f"{name} failed: {result}")

    def record_start(self, stream=None):
        self._check("cuEventRecord", self._lib.cuEventRecord(self.start_ev, stream))

    def record_stop(self, stream=None):
        self._check("cuEventRecord", self._lib.cuEventRecord(self.stop_ev, stream))
        self._check("cuEventSynchronize", self._lib.cuEventSynchronize(self.stop_ev))

    def elapsed_ms(self) -> float:
        ms = c_float()
        self._check("cuEventElapsedTime",
                     self._lib.cuEventElapsedTime(byref(ms), self.start_ev, self.stop_ev))
        return ms.value

    def destroy(self):
        self._lib.cuEventDestroy_v2(self.start_ev)
        self._lib.cuEventDestroy_v2(self.stop_ev)


def setup_event_api(lib):
    """Add CUDA event function prototypes to the driver library handle."""
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
# Helpers
# ---------------------------------------------------------------------------
def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def load_norm_f32(model: LithosModel, name: str) -> np.ndarray:
    ti = model.weight_info(name)
    raw = bytes(model.weight_bytes(name))
    if ti.dtype == "BF16":
        w = bf16_to_f32(raw)
    elif ti.dtype == "F16":
        w = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif ti.dtype == "F32":
        w = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        w = np.frombuffer(raw, dtype=np.float32).copy()
    return w + 1.0


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, count * 4)
    return out


def percentile(sorted_list, pct):
    """Return the pct-th percentile from a sorted list."""
    idx = max(0, int(len(sorted_list) * pct / 100.0) - 1)
    return sorted_list[idx]


# ---------------------------------------------------------------------------
# Bandwidth estimation for Qwen3.5-27B forward pass (MLP-only Carmack kernel)
# ---------------------------------------------------------------------------
def estimate_forward_pass_bytes() -> dict:
    """Estimate total HBM bytes read/written per forward pass (MLP-only).

    For W4A16 GPTQ, each weight element is 4 bits. Per layer MLP:
      gate_proj: 5120 x 17408 x 0.5 bytes (qweight) + scales + zeros
      up_proj:   5120 x 17408 x 0.5 bytes + scales + zeros
      down_proj: 17408 x 5120 x 0.5 bytes + scales + zeros

    Plus norm weights (5120 x 4 bytes x 2 per layer) and activations.
    """
    # Per-layer MLP weight reads (W4 packed)
    gate_qw = HIDDEN_DIM * INTERMEDIATE_SIZE // 2       # 4-bit packed
    up_qw = HIDDEN_DIM * INTERMEDIATE_SIZE // 2
    down_qw = INTERMEDIATE_SIZE * HIDDEN_DIM // 2

    # Scales: one f16 per group of 128 elements
    n_groups_gate = (HIDDEN_DIM * INTERMEDIATE_SIZE) // GROUP_SIZE
    n_groups_up = n_groups_gate
    n_groups_down = (INTERMEDIATE_SIZE * HIDDEN_DIM) // GROUP_SIZE
    gate_sc = n_groups_gate * 2
    up_sc = n_groups_up * 2
    down_sc = n_groups_down * 2

    # Zeros: packed, one per group
    gate_zr = n_groups_gate * 4
    up_zr = n_groups_up * 4
    down_zr = n_groups_down * 4

    per_layer_weight_read = (gate_qw + gate_sc + gate_zr +
                             up_qw + up_sc + up_zr +
                             down_qw + down_sc + down_zr)

    # Norm weights: 2 x 5120 x 4 bytes (input_norm + post_attn_norm)
    per_layer_norm_read = 2 * HIDDEN_DIM * 4

    # Activation reads/writes per layer (f32):
    #   read input vector:    5120 x 4  (for gate + up projections)
    #   write gate output:   17408 x 4
    #   write up output:     17408 x 4
    #   read gate + up:      17408 x 4 x 2 (for activation)
    #   write activated:     17408 x 4
    #   read activated:      17408 x 4 (for down proj)
    #   write down output:    5120 x 4
    per_layer_act_read = (HIDDEN_DIM * 4 +
                          INTERMEDIATE_SIZE * 4 * 2 +   # gate + up for activation
                          INTERMEDIATE_SIZE * 4)         # activated for down
    per_layer_act_write = (INTERMEDIATE_SIZE * 4 +       # gate
                           INTERMEDIATE_SIZE * 4 +       # up
                           INTERMEDIATE_SIZE * 4 +       # activated
                           HIDDEN_DIM * 4)               # down output

    per_layer_total = (per_layer_weight_read + per_layer_norm_read +
                       per_layer_act_read + per_layer_act_write)

    # lm_head: FP16 dense matrix 248320 x 5120 x 2 bytes
    lm_head_read = VOCAB_SIZE * HIDDEN_DIM * 2
    lm_head_write = VOCAB_SIZE * 4  # output logits f32

    # Embed: read one row of FP16 table
    embed_read = HIDDEN_DIM * 2

    total_read = (per_layer_weight_read + per_layer_norm_read + per_layer_act_read) * NUM_LAYERS + lm_head_read + embed_read
    total_write = per_layer_act_write * NUM_LAYERS + lm_head_write
    total = total_read + total_write

    return {
        "per_layer_weight_read_bytes": per_layer_weight_read,
        "per_layer_norm_read_bytes": per_layer_norm_read,
        "per_layer_activation_read_bytes": per_layer_act_read,
        "per_layer_activation_write_bytes": per_layer_act_write,
        "per_layer_total_bytes": per_layer_total,
        "all_layers_total_bytes": per_layer_total * NUM_LAYERS,
        "lm_head_bytes": lm_head_read + lm_head_write,
        "embed_bytes": embed_read,
        "total_read_bytes": total_read,
        "total_write_bytes": total_write,
        "total_bytes": total,
        "total_gib": total / (1024**3),
    }


# ---------------------------------------------------------------------------
# Carmack kernel benchmark (MLP-only fused)
# ---------------------------------------------------------------------------
class CarmackBenchmark:
    """Benchmark the fused Carmack MLP-only kernel with CUDA event timing."""

    def __init__(self):
        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self._lib = self.gpu._lib
        setup_event_api(self._lib)
        self.timer = GPUTimer(self._lib)

        sm_count = self.gpu.device_attribute(16)  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

        # Load Carmack kernel
        cubin_path = f"{KERNEL_DIR}/forward_pass_carmack.cubin"
        mod = self.gpu.load_cubin(cubin_path)
        self.func = self.gpu.get_function(mod, "forward_pass_carmack")
        max_blks = self.gpu.max_active_blocks(self.func, THREADS_PER_BLOCK, 0)
        self.grid_size = min(NUM_BLOCKS, max_blks * sm_count)

        # Prepare embedding
        token_ids = self.tok.encode(BENCH_PROMPT)
        last_token_id = token_ids[-1]
        embed_raw = bytes(self.model.weight_bytes("model.language_model.embed_tokens.weight"))
        row_off = last_token_id * HIDDEN_DIM * 2
        row_f16 = np.frombuffer(
            embed_raw[row_off : row_off + HIDDEN_DIM * 2], dtype=np.float16
        )
        self.d_embed = upload_f32(self.gpu, row_f16.astype(np.float32))

        # Build weight table
        self._build_weight_table()

        # Allocate buffers
        self.d_activation = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_normed = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_mid_scratch = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_sync_workspace = self.gpu.mem_alloc(16)
        self.d_norm_partials = self.gpu.mem_alloc(self.grid_size * 4)
        self.d_output_logits = self.gpu.mem_alloc(VOCAB_SIZE * 4)

        final_norm_w = load_norm_f32(self.model, "model.language_model.norm.weight")
        self.d_final_norm = upload_f32(self.gpu, final_norm_w)

        lm_head_ptr = self.model.weight_info("lm_head.weight").ptr
        lm_head_scales_size = VOCAB_SIZE * 40 * 2
        lm_head_zeros_size = VOCAB_SIZE * 40 * 4
        self.d_lm_head_scales = self.gpu.mem_alloc(lm_head_scales_size)
        self.d_lm_head_zeros = self.gpu.mem_alloc(lm_head_zeros_size)
        self.gpu.memset_d8(self.d_lm_head_scales, 0, lm_head_scales_size)
        self.gpu.memset_d8(self.d_lm_head_zeros, 0, lm_head_zeros_size)
        self.lm_head_ptr = lm_head_ptr

        self.args = self._make_args()

    def _build_weight_table(self):
        d_input_norms = []
        d_post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            w_in = load_norm_f32(self.model, f"{prefix}.input_layernorm.weight")
            w_post = load_norm_f32(self.model, f"{prefix}.post_attention_layernorm.weight")
            d_input_norms.append(upload_f32(self.gpu, w_in))
            d_post_attn_norms.append(upload_f32(self.gpu, w_post))

        table = np.zeros(NUM_LAYERS * PTRS_PER_LAYER, dtype=np.uint64)
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            base = i * PTRS_PER_LAYER
            table[base + 0] = d_input_norms[i].value
            table[base + 10] = d_post_attn_norms[i].value
            table[base + 1] = self.model.weight_info(f"{prefix}.mlp.gate_proj.qweight").ptr
            table[base + 2] = self.model.weight_info(f"{prefix}.mlp.gate_proj.scales").ptr
            table[base + 3] = self.model.weight_info(f"{prefix}.mlp.gate_proj.qzeros").ptr
            table[base + 4] = self.model.weight_info(f"{prefix}.mlp.up_proj.qweight").ptr
            table[base + 5] = self.model.weight_info(f"{prefix}.mlp.up_proj.scales").ptr
            table[base + 6] = self.model.weight_info(f"{prefix}.mlp.up_proj.qzeros").ptr
            table[base + 7] = self.model.weight_info(f"{prefix}.mlp.down_proj.qweight").ptr
            table[base + 8] = self.model.weight_info(f"{prefix}.mlp.down_proj.scales").ptr
            table[base + 9] = self.model.weight_info(f"{prefix}.mlp.down_proj.qzeros").ptr

        self.d_weight_table = upload_f32(self.gpu, table.view(np.float32))

    def _make_args(self):
        return [
            ctypes.c_uint64(self.d_weight_table.value),
            ctypes.c_uint64(self.d_embed.value),
            ctypes.c_uint32(0),
            ctypes.c_uint64(self.d_mid_scratch.value),
            ctypes.c_uint64(self.d_output_logits.value),
            ctypes.c_uint64(self.d_final_norm.value),
            ctypes.c_uint64(self.lm_head_ptr),
            ctypes.c_uint64(self.d_lm_head_scales.value),
            ctypes.c_uint64(self.d_lm_head_zeros.value),
            ctypes.c_uint64(self.d_activation.value),
            ctypes.c_uint64(self.d_normed.value),
            ctypes.c_uint64(self.d_sync_workspace.value),
            ctypes.c_uint64(self.d_norm_partials.value),
        ]

    def reset_state(self):
        self.gpu.memset_d8(self.d_sync_workspace, 0, 16)
        self.gpu.memset_d8(self.d_output_logits, 0, VOCAB_SIZE * 4)

    def run_once_wall(self) -> float:
        """Single forward pass, return wall-clock ms."""
        self.reset_state()
        self.gpu.synchronize()
        t0 = time.perf_counter()
        self.gpu.launch_cooperative(
            self.func,
            grid=(self.grid_size, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            args=self.args,
            shared_mem=0,
        )
        self.gpu.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def run_once_gpu(self) -> float:
        """Single forward pass, return GPU-timed ms via CUDA events."""
        self.reset_state()
        self.gpu.synchronize()
        self.timer.record_start()
        self.gpu.launch_cooperative(
            self.func,
            grid=(self.grid_size, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            args=self.args,
            shared_mem=0,
        )
        self.timer.record_stop()
        return self.timer.elapsed_ms()

    def run_benchmark(self, num_warmup: int = 5, num_runs: int = 20) -> dict:
        """Run full benchmark with both wall-clock and GPU event timing."""
        # Warmup
        for _ in range(num_warmup):
            self.run_once_wall()

        # Wall-clock runs
        wall_times = []
        for _ in range(num_runs):
            wall_times.append(self.run_once_wall())

        # GPU event runs
        gpu_times = []
        for _ in range(num_runs):
            gpu_times.append(self.run_once_gpu())

        wall_times.sort()
        gpu_times.sort()

        wall_median = wall_times[num_runs // 2]
        gpu_median = gpu_times[num_runs // 2]
        dispatch_overhead_ms = wall_median - gpu_median

        bw = estimate_forward_pass_bytes()

        # Bandwidth utilization based on GPU time
        bw_tbs_gpu = (bw["total_bytes"] / 1e12) / (gpu_median / 1e3)
        bw_pct_gpu = bw_tbs_gpu / GH200_PEAK_BW_TBS * 100.0

        tok_s_wall = 1000.0 / wall_median if wall_median > 0 else 0
        tok_s_gpu = 1000.0 / gpu_median if gpu_median > 0 else 0

        # Correctness check
        self.reset_state()
        self.gpu.launch_cooperative(
            self.func,
            grid=(self.grid_size, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            args=self.args,
            shared_mem=0,
        )
        self.gpu.synchronize()
        logits = download_f32(self.gpu, self.d_output_logits, VOCAB_SIZE)
        logits_clean = np.where(np.isfinite(logits), logits, -1e9)
        top_token = int(np.argmax(logits_clean))
        top_word = self.tok.decode([top_token])

        # Paris rank
        paris_rank = None
        for candidate in ["Paris", " Paris"]:
            cand_ids = self.tok.encode(candidate)
            for cid in cand_ids:
                rank = int(np.sum(logits_clean > logits_clean[cid])) + 1
                if paris_rank is None or rank < paris_rank:
                    paris_rank = rank

        return {
            "kernel": "forward_pass_carmack",
            "scope": "MLP-only (64 layers fused)",
            "prompt": BENCH_PROMPT,
            "num_runs": num_runs,
            "wall_clock": {
                "median_ms": round(wall_median, 3),
                "min_ms": round(wall_times[0], 3),
                "max_ms": round(wall_times[-1], 3),
                "p10_ms": round(percentile(wall_times, 10), 3),
                "p90_ms": round(percentile(wall_times, 90), 3),
                "mean_ms": round(statistics.fmean(wall_times), 3),
                "stdev_ms": round(statistics.stdev(wall_times), 3) if len(wall_times) > 1 else 0,
                "tok_s": round(tok_s_wall, 2),
            },
            "gpu_events": {
                "median_ms": round(gpu_median, 3),
                "min_ms": round(gpu_times[0], 3),
                "max_ms": round(gpu_times[-1], 3),
                "p10_ms": round(percentile(gpu_times, 10), 3),
                "p90_ms": round(percentile(gpu_times, 90), 3),
                "mean_ms": round(statistics.fmean(gpu_times), 3),
                "stdev_ms": round(statistics.stdev(gpu_times), 3) if len(gpu_times) > 1 else 0,
                "tok_s": round(tok_s_gpu, 2),
            },
            "dispatch_overhead_ms": round(dispatch_overhead_ms, 3),
            "bandwidth": {
                "total_bytes_per_pass": bw["total_bytes"],
                "total_gib_per_pass": round(bw["total_gib"], 3),
                "total_read_bytes": bw["total_read_bytes"],
                "total_write_bytes": bw["total_write_bytes"],
                "achieved_tbs": round(bw_tbs_gpu, 3),
                "peak_tbs": GH200_PEAK_BW_TBS,
                "utilization_pct": round(bw_pct_gpu, 1),
                "per_layer_weight_read_bytes": bw["per_layer_weight_read_bytes"],
                "lm_head_bytes": bw["lm_head_bytes"],
            },
            "correctness": {
                "top_token_id": top_token,
                "top_token_text": top_word,
                "paris_rank": paris_rank,
                "correct": paris_rank == 1,
            },
            "grid": f"({self.grid_size}, 1, 1)",
            "block": f"({THREADS_PER_BLOCK}, 1, 1)",
        }

    def close(self):
        self.timer.destroy()
        self.gpu.close()
        self.model.close()


# ---------------------------------------------------------------------------
# Per-kernel profiled benchmark (individual kernel timing)
# ---------------------------------------------------------------------------
class PerKernelBenchmark:
    """Profile individual kernel timings for a single forward pass using
    the per-layer GPU MLP path (not fused Carmack)."""

    def __init__(self):
        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.gpu = CUDADriver()
        self.epsilon = self.model.config.rms_norm_eps
        self._lib = self.gpu._lib
        setup_event_api(self._lib)
        self.timer = GPUTimer(self._lib)

        # Load kernels
        norm_mod = self.gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
        self.norm_func = self.gpu.get_function(norm_mod, "norm")

        fast_mod = self.gpu.load_cubin(f"{KERNEL_DIR}/gptq_gemv_fast.cubin")
        self.proj_func = self.gpu.get_function(fast_mod, "gptq_gemv_fast")

        activate_mod = self.gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
        self.activate_func = self.gpu.get_function(activate_mod, "activate")

        # Buffers
        self.d_x = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_out = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_norm_weight = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_gate = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_up = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_act = self.gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
        self.d_down = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.d_zero = self.gpu.mem_alloc(HIDDEN_DIM * 4)
        self.gpu.memcpy_htod(self.d_zero,
                             np.zeros(HIDDEN_DIM, dtype=np.float32).ctypes.data_as(ctypes.c_void_p),
                             HIDDEN_DIM * 4)

        # Preload norms for layer 0
        self.norms = {}
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            self.norms[i] = {
                "input": load_norm_f32(self.model, f"{prefix}.input_layernorm.weight"),
                "post": load_norm_f32(self.model, f"{prefix}.post_attention_layernorm.weight"),
            }

        # Upload dummy input
        dummy = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.01
        self.gpu.memcpy_htod(self.d_x, dummy.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)

    def _timed_launch(self, func, grid, block, args, shared_mem=0):
        """Launch kernel and return GPU time in ms from CUDA events."""
        self.timer.record_start()
        self.gpu.launch(func, grid=grid, block=block, args=args, shared_mem=shared_mem)
        self.timer.record_stop()
        return self.timer.elapsed_ms()

    def profile_mlp_kernels(self, layer_idx: int, num_runs: int = 10) -> dict:
        """Profile each MLP kernel for a single layer, return timing dict."""
        prefix = f"model.language_model.layers.{layer_idx}"
        norm_w = self.norms[layer_idx]["post"]

        norm_times = []
        gate_times = []
        up_times = []
        activate_times = []
        down_times = []
        dispatch_times = []

        for _ in range(num_runs):
            # Norm
            self.gpu.memcpy_htod(self.d_norm_weight,
                                 norm_w.ctypes.data_as(ctypes.c_void_p), HIDDEN_DIM * 4)
            cpu_t0 = time.perf_counter()
            t = self._timed_launch(
                self.norm_func,
                grid=(1, 1, 1), block=(256, 1, 1),
                args=[
                    ctypes.c_uint64(self.d_x.value),
                    ctypes.c_uint64(self.d_zero.value),
                    ctypes.c_uint64(self.d_norm_weight.value),
                    ctypes.c_uint64(self.d_norm_out.value),
                    ctypes.c_float(self.epsilon),
                ],
                shared_mem=128,
            )
            cpu_t1 = time.perf_counter()
            norm_times.append(t)
            dispatch_times.append((cpu_t1 - cpu_t0) * 1000.0 - t)

            # Gate projection
            qw_ptr = self.model.weight_info(f"{prefix}.mlp.gate_proj.qweight").ptr
            sc_ptr = self.model.weight_info(f"{prefix}.mlp.gate_proj.scales").ptr
            self.gpu.memset_d8(self.d_gate, 0, INTERMEDIATE_SIZE * 4)
            K, N = HIDDEN_DIM, INTERMEDIATE_SIZE
            K_SPLITS = 16
            k_packed_per_split = (K // 8) // K_SPLITS
            fast_smem = k_packed_per_split * 8 * 4

            cpu_t0 = time.perf_counter()
            t = self._timed_launch(
                self.proj_func,
                grid=(math.ceil(N / 256), K_SPLITS, 1), block=(256, 1, 1),
                args=[
                    ctypes.c_uint64(qw_ptr),
                    ctypes.c_uint64(sc_ptr),
                    ctypes.c_uint64(self.d_norm_out.value),
                    ctypes.c_uint64(self.d_gate.value),
                    ctypes.c_uint32(N), ctypes.c_uint32(K),
                    ctypes.c_uint32(K_SPLITS), ctypes.c_uint32(k_packed_per_split),
                ],
                shared_mem=fast_smem,
            )
            cpu_t1 = time.perf_counter()
            gate_times.append(t)
            dispatch_times.append((cpu_t1 - cpu_t0) * 1000.0 - t)

            # Up projection
            qw_ptr = self.model.weight_info(f"{prefix}.mlp.up_proj.qweight").ptr
            sc_ptr = self.model.weight_info(f"{prefix}.mlp.up_proj.scales").ptr
            self.gpu.memset_d8(self.d_up, 0, INTERMEDIATE_SIZE * 4)
            cpu_t0 = time.perf_counter()
            t = self._timed_launch(
                self.proj_func,
                grid=(math.ceil(N / 256), K_SPLITS, 1), block=(256, 1, 1),
                args=[
                    ctypes.c_uint64(qw_ptr),
                    ctypes.c_uint64(sc_ptr),
                    ctypes.c_uint64(self.d_norm_out.value),
                    ctypes.c_uint64(self.d_up.value),
                    ctypes.c_uint32(N), ctypes.c_uint32(K),
                    ctypes.c_uint32(K_SPLITS), ctypes.c_uint32(k_packed_per_split),
                ],
                shared_mem=fast_smem,
            )
            cpu_t1 = time.perf_counter()
            up_times.append(t)
            dispatch_times.append((cpu_t1 - cpu_t0) * 1000.0 - t)

            # Activate (SiLU * gate)
            GRID_ACT = max(1, math.ceil(INTERMEDIATE_SIZE / 256))
            cpu_t0 = time.perf_counter()
            t = self._timed_launch(
                self.activate_func,
                grid=(GRID_ACT, 1, 1), block=(256, 1, 1),
                args=[
                    ctypes.c_uint64(self.d_up.value),
                    ctypes.c_uint64(self.d_gate.value),
                    ctypes.c_uint64(self.d_act.value),
                ],
            )
            cpu_t1 = time.perf_counter()
            activate_times.append(t)
            dispatch_times.append((cpu_t1 - cpu_t0) * 1000.0 - t)

            # Down projection
            qw_ptr = self.model.weight_info(f"{prefix}.mlp.down_proj.qweight").ptr
            sc_ptr = self.model.weight_info(f"{prefix}.mlp.down_proj.scales").ptr
            K2, N2 = INTERMEDIATE_SIZE, HIDDEN_DIM
            k_packed_per_split2 = (K2 // 8) // K_SPLITS
            fast_smem2 = k_packed_per_split2 * 8 * 4
            self.gpu.memset_d8(self.d_down, 0, HIDDEN_DIM * 4)
            cpu_t0 = time.perf_counter()
            t = self._timed_launch(
                self.proj_func,
                grid=(math.ceil(N2 / 256), K_SPLITS, 1), block=(256, 1, 1),
                args=[
                    ctypes.c_uint64(qw_ptr),
                    ctypes.c_uint64(sc_ptr),
                    ctypes.c_uint64(self.d_act.value),
                    ctypes.c_uint64(self.d_down.value),
                    ctypes.c_uint32(N2), ctypes.c_uint32(K2),
                    ctypes.c_uint32(K_SPLITS), ctypes.c_uint32(k_packed_per_split2),
                ],
                shared_mem=fast_smem2,
            )
            cpu_t1 = time.perf_counter()
            down_times.append(t)
            dispatch_times.append((cpu_t1 - cpu_t0) * 1000.0 - t)

        def stats(times):
            s = sorted(times)
            return {
                "median_ms": round(s[len(s)//2], 4),
                "min_ms": round(s[0], 4),
                "max_ms": round(s[-1], 4),
                "mean_ms": round(statistics.fmean(s), 4),
            }

        # Bandwidth per kernel
        gate_bytes = HIDDEN_DIM * INTERMEDIATE_SIZE // 2  # W4 weight read dominates
        up_bytes = gate_bytes
        down_bytes = INTERMEDIATE_SIZE * HIDDEN_DIM // 2
        norm_bytes = HIDDEN_DIM * 4 * 3  # input + weight + output

        gate_med = sorted(gate_times)[len(gate_times)//2]
        up_med = sorted(up_times)[len(up_times)//2]
        down_med = sorted(down_times)[len(down_times)//2]

        return {
            "layer_idx": layer_idx,
            "num_runs": num_runs,
            "norm": stats(norm_times),
            "gate_proj_5120x17408": {
                **stats(gate_times),
                "bytes_read": gate_bytes,
                "bw_gbs": round((gate_bytes / 1e9) / (gate_med / 1e3), 1),
                "bw_pct": round((gate_bytes / 1e12) / (gate_med / 1e3) / GH200_PEAK_BW_TBS * 100, 1),
            },
            "up_proj_5120x17408": {
                **stats(up_times),
                "bytes_read": up_bytes,
                "bw_gbs": round((up_bytes / 1e9) / (up_med / 1e3), 1),
                "bw_pct": round((up_bytes / 1e12) / (up_med / 1e3) / GH200_PEAK_BW_TBS * 100, 1),
            },
            "activate": stats(activate_times),
            "down_proj_17408x5120": {
                **stats(down_times),
                "bytes_read": down_bytes,
                "bw_gbs": round((down_bytes / 1e9) / (down_med / 1e3), 1),
                "bw_pct": round((down_bytes / 1e12) / (down_med / 1e3) / GH200_PEAK_BW_TBS * 100, 1),
            },
            "dispatch_overhead": {
                "mean_per_call_ms": round(statistics.fmean(dispatch_times), 4),
                "total_per_mlp_layer_ms": round(sum(dispatch_times) / num_runs, 4),
            },
        }

    def close(self):
        self.timer.destroy()
        self.gpu.close()
        self.model.close()


# ---------------------------------------------------------------------------
# vLLM comparison
# ---------------------------------------------------------------------------
def measure_vllm(prompt: str = BENCH_PROMPT, max_tokens: int = 1,
                 num_warmup: int = 3, num_runs: int = 10) -> dict | None:
    """Hit vLLM server with streaming request, measure tok/s and TTFT."""
    try:
        # Quick health check
        urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=2.0)
    except Exception:
        return None

    def single_request():
        payload = json.dumps({
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{VLLM_URL}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        events = []
        t_send = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        events.append((time.perf_counter(), text))
        t_end = time.perf_counter()
        total = t_end - t_send
        ttft = (events[0][0] - t_send) * 1000.0 if events else float("nan")
        tps = len(events) / total if total > 0 and events else 0.0
        return {"ttft_ms": ttft, "total_s": total, "tps": tps,
                "chunks": len(events), "text": "".join(c for _, c in events)}

    # Warmup
    for _ in range(num_warmup):
        try:
            single_request()
        except Exception:
            pass

    # Bench
    results = []
    for _ in range(num_runs):
        try:
            results.append(single_request())
        except Exception as e:
            results.append({"ttft_ms": float("nan"), "total_s": 0, "tps": 0,
                            "chunks": 0, "text": f"ERROR: {e}"})

    valid = [r for r in results if r["chunks"] > 0]
    if not valid:
        return {"error": "all requests failed", "raw": results}

    ttft_list = sorted(r["ttft_ms"] for r in valid)
    tps_list = sorted(r["tps"] for r in valid)

    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "num_runs": num_runs,
        "valid_runs": len(valid),
        "ttft_median_ms": round(ttft_list[len(ttft_list)//2], 1),
        "ttft_p95_ms": round(percentile(ttft_list, 95), 1),
        "tps_median": round(tps_list[len(tps_list)//2], 2),
        "tps_mean": round(statistics.fmean(tps_list), 2),
        "sample_text": valid[0]["text"] if valid else "",
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_table(results: dict, per_kernel: dict | None, vllm: dict | None):
    """Print human-readable results table."""
    w = results["wall_clock"]
    g = results["gpu_events"]
    bw = results["bandwidth"]
    c = results["correctness"]

    print()
    print("=" * 78)
    print("  LITHOS BENCHMARK HARNESS -- GH200")
    print("=" * 78)

    print(f"\n  Prompt:  {results['prompt']!r}")
    print(f"  Kernel:  {results['kernel']}  ({results['scope']})")
    print(f"  Grid:    {results['grid']}  Block: {results['block']}")
    print(f"  Runs:    {results['num_runs']}")

    print(f"\n  TOKENS / SECOND")
    print(f"  {'-' * 50}")
    print(f"  {'Metric':<30s} {'Wall-clock':>12s} {'GPU events':>12s}")
    print(f"  {'-' * 50}")
    print(f"  {'tok/s':<30s} {w['tok_s']:>12.2f} {g['tok_s']:>12.2f}")
    print(f"  {'Median ms':<30s} {w['median_ms']:>12.3f} {g['median_ms']:>12.3f}")
    print(f"  {'Min ms':<30s} {w['min_ms']:>12.3f} {g['min_ms']:>12.3f}")
    print(f"  {'Max ms':<30s} {w['max_ms']:>12.3f} {g['max_ms']:>12.3f}")
    print(f"  {'P10 ms':<30s} {w['p10_ms']:>12.3f} {g['p10_ms']:>12.3f}")
    print(f"  {'P90 ms':<30s} {w['p90_ms']:>12.3f} {g['p90_ms']:>12.3f}")
    print(f"  {'Stdev ms':<30s} {w['stdev_ms']:>12.3f} {g['stdev_ms']:>12.3f}")

    print(f"\n  DISPATCH OVERHEAD")
    print(f"  {'-' * 50}")
    print(f"  Wall-clock median - GPU median: {results['dispatch_overhead_ms']:.3f} ms")

    print(f"\n  HBM BANDWIDTH UTILIZATION")
    print(f"  {'-' * 50}")
    print(f"  Total data per pass:   {bw['total_gib_per_pass']:.3f} GiB")
    print(f"    Read:                {bw['total_read_bytes'] / (1024**3):.3f} GiB")
    print(f"    Write:               {bw['total_write_bytes'] / (1024**3):.3f} GiB")
    print(f"  Achieved bandwidth:    {bw['achieved_tbs']:.3f} TB/s")
    print(f"  Peak bandwidth:        {bw['peak_tbs']:.1f} TB/s")
    print(f"  Utilization:           {bw['utilization_pct']:.1f}%")
    print(f"  Per-layer weight read: {bw['per_layer_weight_read_bytes'] / (1024**2):.1f} MiB")
    print(f"  lm_head data:          {bw['lm_head_bytes'] / (1024**2):.1f} MiB")

    print(f"\n  CORRECTNESS")
    print(f"  {'-' * 50}")
    print(f"  Top token: {c['top_token_id']} '{c['top_token_text']}'")
    print(f"  Paris rank: {c['paris_rank']}  {'PASS' if c['correct'] else 'FAIL'}")

    if per_kernel:
        print(f"\n  PER-KERNEL PROFILE (layer {per_kernel['layer_idx']}, MLP only)")
        print(f"  {'-' * 68}")
        print(f"  {'Kernel':<28s} {'Median ms':>10s} {'Min ms':>10s} {'BW GB/s':>10s} {'BW %':>8s}")
        print(f"  {'-' * 68}")
        for name in ["norm", "gate_proj_5120x17408", "up_proj_5120x17408",
                      "activate", "down_proj_17408x5120"]:
            k = per_kernel[name]
            bw_str = f"{k.get('bw_gbs', '-'):>10}" if 'bw_gbs' in k else f"{'--':>10}"
            pct_str = f"{k.get('bw_pct', '-'):>8}" if 'bw_pct' in k else f"{'--':>8}"
            print(f"  {name:<28s} {k['median_ms']:>10.4f} {k['min_ms']:>10.4f} {bw_str} {pct_str}")
        print(f"  {'-' * 68}")
        mlp_total = sum(per_kernel[n]["median_ms"]
                        for n in ["norm", "gate_proj_5120x17408", "up_proj_5120x17408",
                                  "activate", "down_proj_17408x5120"])
        print(f"  {'MLP total (1 layer)':<28s} {mlp_total:>10.4f}")
        print(f"  {'MLP total (64 layers est.)':<28s} {mlp_total * 64:>10.2f}")
        d = per_kernel["dispatch_overhead"]
        print(f"  Dispatch overhead:       {d['mean_per_call_ms']:.4f} ms/call  "
              f"({d['total_per_mlp_layer_ms']:.4f} ms/layer)")

    if vllm:
        print(f"\n  vLLM COMPARISON")
        print(f"  {'-' * 50}")
        if "error" in vllm:
            print(f"  Error: {vllm['error']}")
        else:
            print(f"  vLLM tok/s (median):   {vllm['tps_median']:.2f}")
            print(f"  vLLM TTFT (median):    {vllm['ttft_median_ms']:.1f} ms")
            print(f"  vLLM sample output:    {vllm['sample_text']!r}")
            print(f"  Lithos tok/s (wall):   {w['tok_s']:.2f}")
            ratio = w["tok_s"] / vllm["tps_median"] if vllm["tps_median"] > 0 else 0
            print(f"  Lithos / vLLM ratio:   {ratio:.3f}x")

    # Progress toward target
    print(f"\n  PROGRESS TOWARD TARGET")
    print(f"  {'-' * 50}")
    current = w["tok_s"]
    target = 400.0
    vllm_baseline = 179.0
    print(f"  Current:      {current:>8.2f} tok/s")
    print(f"  vLLM (ref):   {vllm_baseline:>8.2f} tok/s")
    print(f"  Target:       {target:>8.2f} tok/s")
    print(f"  Gap to vLLM:  {vllm_baseline - current:>8.2f} tok/s ({(current/vllm_baseline*100):.1f}% of vLLM)")
    print(f"  Gap to target:{target - current:>8.2f} tok/s ({(current/target*100):.1f}% of target)")

    print(f"\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------
def save_results(results: dict, per_kernel: dict | None, vllm: dict | None):
    """Save results to timestamped JSON and append to history log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lithos": results,
        "per_kernel": per_kernel,
        "vllm": vllm,
    }

    # Timestamped snapshot
    snapshot_path = RESULTS_DIR / f"{timestamp}.json"
    with open(snapshot_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # Append to history log (JSONL)
    history_path = RESULTS_DIR / "history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return snapshot_path, history_path


def print_history_summary():
    """Print a summary of recent benchmark history from the JSONL log."""
    history_path = RESULTS_DIR / "history.jsonl"
    if not history_path.exists():
        return

    entries = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if len(entries) < 2:
        return

    print(f"\n  HISTORY (last {min(10, len(entries))} runs)")
    print(f"  {'-' * 62}")
    print(f"  {'Timestamp':<22s} {'tok/s wall':>11s} {'tok/s GPU':>11s} {'BW TB/s':>9s} {'Paris':>6s}")
    print(f"  {'-' * 62}")

    for entry in entries[-10:]:
        ts = entry.get("timestamp", "?")[:19]
        lit = entry.get("lithos", {})
        wc = lit.get("wall_clock", {})
        ge = lit.get("gpu_events", {})
        bw_info = lit.get("bandwidth", {})
        cor = lit.get("correctness", {})
        print(f"  {ts:<22s} {wc.get('tok_s', 0):>11.2f} {ge.get('tok_s', 0):>11.2f} "
              f"{bw_info.get('achieved_tbs', 0):>9.3f} {'PASS' if cor.get('correct') else 'FAIL':>6s}")
    print(f"  {'-' * 62}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Lithos Benchmark Harness")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs (default: 20)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs (default: 5)")
    parser.add_argument("--compare-vllm", action="store_true", help="Include vLLM comparison")
    parser.add_argument("--skip-per-kernel", action="store_true", help="Skip per-kernel profiling")
    parser.add_argument("--json-only", action="store_true", help="Only output JSON, no table")
    parser.add_argument("--profile-layers", type=int, nargs="*", default=[0, 32, 63],
                        help="Layer indices to profile (default: 0 32 63)")
    args = parser.parse_args()

    if not args.json_only:
        print(f"Lithos Benchmark Harness -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: Qwen3.5-27B GPTQ W4A16")
        print(f"Device: GH200 -- {GH200_SM_COUNT} SMs, {GH200_PEAK_BW_TBS} TB/s HBM3e peak")
        print()

    # ---- Carmack fused kernel benchmark ----
    if not args.json_only:
        print("Loading Carmack fused kernel benchmark...")
    bench = CarmackBenchmark()
    results = bench.run_benchmark(num_warmup=args.warmup, num_runs=args.runs)
    bench.close()

    # ---- Per-kernel profiling ----
    per_kernel = None
    if not args.skip_per_kernel:
        if not args.json_only:
            print("Running per-kernel profiling...")
        pkb = PerKernelBenchmark()
        # Profile the first requested layer (use first as representative)
        layer_to_profile = args.profile_layers[0] if args.profile_layers else 0
        per_kernel = pkb.profile_mlp_kernels(layer_to_profile, num_runs=args.runs)
        pkb.close()

    # ---- vLLM comparison ----
    vllm = None
    if args.compare_vllm:
        if not args.json_only:
            print("Measuring vLLM baseline...")
        vllm = measure_vllm()

    # ---- Output ----
    if not args.json_only:
        print_table(results, per_kernel, vllm)
        print_history_summary()

    snapshot_path, history_path = save_results(results, per_kernel, vllm)

    if args.json_only:
        record = {"lithos": results, "per_kernel": per_kernel, "vllm": vllm}
        print(json.dumps(record, indent=2, default=str))
    else:
        print(f"\n  Results saved: {snapshot_path}")
        print(f"  History log:   {history_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
