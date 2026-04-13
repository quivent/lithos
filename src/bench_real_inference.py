#!/usr/bin/env python3
"""
Benchmark real inference: loads GPTQ weights, launches optimized GPU kernels,
measures actual tok/s through 64-layer MLP pipeline.

Wires together:
  - gptq_gemv_ultra.cubin (optimized GPTQ W4A16 GEMV, 256 cols/block)
  - norm.cubin (RMSNorm)
  - activate.cubin (SiLU gate)
  - embed_f16.cubin (embedding lookup)
  - Real GPTQ model weights (mmap'd safetensors, unified memory on GH200)

Usage:
    python3 bench_real_inference.py [--tokens N] [--prompt "text"]
"""

from __future__ import annotations

import argparse
import ctypes
import math
import numpy as np
import struct
import sys
import time
from ctypes import (
    POINTER, byref, c_float, c_int, c_size_t,
    c_uint, c_uint32, c_uint64, c_void_p,
)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import LithosModel
from tokenizer import Tokenizer

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
NUM_LAYERS = 64
VOCAB_SIZE = 248320
EOS_TOKEN_ID = 248044

# CUDA driver API types
CUresult = c_int
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUstream = c_void_p
CUdeviceptr = c_uint64
CUevent = c_void_p


def _check(name, result):
    if result != 0:
        raise RuntimeError(f"{name} failed: error {result}")


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


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


class NativeBench:
    """Benchmark engine using optimized GPU kernels with real weights."""

    def __init__(self):
        print("=" * 72)
        print("  Lithos Native Inference Benchmark")
        print("=" * 72)
        t0 = time.monotonic()

        # Load model (mmap safetensors)
        self.model = LithosModel(MODEL_DIR)
        self.tok = Tokenizer(MODEL_DIR)
        self.epsilon = self.model.config.rms_norm_eps

        # CUDA setup
        self.cuda = cuda = ctypes.CDLL("libcuda.so.1")
        for fn, at, rt in [
            ('cuInit', [c_uint], CUresult),
            ('cuDeviceGet', [POINTER(CUdevice), c_int], CUresult),
            ('cuCtxCreate_v2', [POINTER(CUcontext), c_uint, CUdevice], CUresult),
            ('cuCtxDestroy_v2', [CUcontext], CUresult),
            ('cuCtxSynchronize', [], CUresult),
            ('cuModuleLoad', [POINTER(CUmodule), ctypes.c_char_p], CUresult),
            ('cuModuleLoadData', [POINTER(CUmodule), c_void_p], CUresult),
            ('cuModuleGetFunction', [POINTER(CUfunction), CUmodule, ctypes.c_char_p], CUresult),
            ('cuMemAlloc_v2', [POINTER(CUdeviceptr), c_size_t], CUresult),
            ('cuMemFree_v2', [CUdeviceptr], CUresult),
            ('cuMemcpyHtoD_v2', [CUdeviceptr, c_void_p, c_size_t], CUresult),
            ('cuMemcpyDtoH_v2', [c_void_p, CUdeviceptr, c_size_t], CUresult),
            ('cuMemsetD8_v2', [CUdeviceptr, ctypes.c_ubyte, c_size_t], CUresult),
            ('cuStreamCreate', [POINTER(CUstream), c_uint], CUresult),
            ('cuStreamSynchronize', [CUstream], CUresult),
            ('cuEventCreate', [POINTER(CUevent), c_uint], CUresult),
            ('cuEventRecord', [CUevent, CUstream], CUresult),
            ('cuEventSynchronize', [CUevent], CUresult),
            ('cuEventElapsedTime', [POINTER(c_float), CUevent, CUevent], CUresult),
            ('cuEventDestroy_v2', [CUevent], CUresult),
        ]:
            f = getattr(cuda, fn)
            f.argtypes = at
            f.restype = rt
        cuda.cuLaunchKernel.argtypes = [
            CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
            c_uint, CUstream, c_void_p, c_void_p,
        ]
        cuda.cuLaunchKernel.restype = CUresult

        _check("cuInit", cuda.cuInit(0))
        dev = CUdevice()
        _check("cuDeviceGet", cuda.cuDeviceGet(byref(dev), 0))

        # Try to retain primary context instead of creating a new one
        # (uses less memory when VLLM has most of GPU memory)
        self.ctx = CUcontext()
        for fn_name in ['cuDevicePrimaryCtxRetain', 'cuCtxCreate_v2']:
            if fn_name == 'cuDevicePrimaryCtxRetain':
                try:
                    cuda.cuDevicePrimaryCtxRetain.argtypes = [POINTER(CUcontext), CUdevice]
                    cuda.cuDevicePrimaryCtxRetain.restype = CUresult
                    r = cuda.cuDevicePrimaryCtxRetain(byref(self.ctx), dev)
                    if r == 0:
                        # Push context to make it current
                        cuda.cuCtxPushCurrent_v2 = cuda.cuCtxPushCurrent_v2
                        cuda.cuCtxPushCurrent_v2.argtypes = [CUcontext]
                        cuda.cuCtxPushCurrent_v2.restype = CUresult
                        cuda.cuCtxPushCurrent_v2(self.ctx)
                        self._using_primary = True
                        break
                except Exception:
                    pass
            else:
                r = cuda.cuCtxCreate_v2(byref(self.ctx), 0, dev)
                _check("cuCtxCreate", r)
                self._using_primary = False
                break

        self.stream = CUstream()
        _check("cuStreamCreate", cuda.cuStreamCreate(byref(self.stream), 0))
        print(f"  CUDA initialized")

        # Load kernels
        self._load_kernels()

        # Allocate GPU buffers
        self._alloc_buffers()

        # Preload norm weights
        self._preload_norms()

        # Events for timing
        self.ev_start = CUevent()
        self.ev_stop = CUevent()
        _check("ev", cuda.cuEventCreate(byref(self.ev_start), 0))
        _check("ev", cuda.cuEventCreate(byref(self.ev_stop), 0))

        t1 = time.monotonic()
        print(f"  Init time: {t1-t0:.3f}s")

    def _load_kernels(self):
        cuda = self.cuda
        print("  Loading kernels...")

        # gptq_gemv_safe -- alignment-safe GEMV (scalar loads, no v4 requirement)
        mod = CUmodule()
        cubin_path = f"{KERNEL_DIR}/gptq_gemv_safe.cubin".encode()
        _check("load safe", cuda.cuModuleLoad(byref(mod), cubin_path))
        self.gemv_func = CUfunction()
        _check("getfunc safe", cuda.cuModuleGetFunction(byref(self.gemv_func), mod, b"gptq_gemv_safe"))

        # Also load gptq_matvec as fallback reference
        mod2 = CUmodule()
        _check("load matvec", cuda.cuModuleLoad(byref(mod2), f"{KERNEL_DIR}/gptq_matvec.cubin".encode()))
        self.matvec_func = CUfunction()
        _check("getfunc matvec", cuda.cuModuleGetFunction(byref(self.matvec_func), mod2, b"gptq_matvec"))

        # Norm
        # Try cached norm first, fall back to kernels/
        norm_path = f"{CACHE_DIR}/norm.cubin"
        if not Path(norm_path).exists():
            norm_path = f"{KERNEL_DIR}/norm.cubin"
        mod3 = CUmodule()
        _check("load norm", cuda.cuModuleLoad(byref(mod3), norm_path.encode()))
        self.norm_func = CUfunction()
        _check("getfunc norm", cuda.cuModuleGetFunction(byref(self.norm_func), mod3, b"norm"))

        # Activate
        act_path = f"{CACHE_DIR}/activate.cubin"
        if not Path(act_path).exists():
            act_path = f"{KERNEL_DIR}/activate.cubin"
        mod4 = CUmodule()
        _check("load activate", cuda.cuModuleLoad(byref(mod4), act_path.encode()))
        self.activate_func = CUfunction()
        _check("getfunc activate", cuda.cuModuleGetFunction(byref(self.activate_func), mod4, b"activate"))

        # Embed
        mod5 = CUmodule()
        _check("load embed", cuda.cuModuleLoad(byref(mod5), f"{KERNEL_DIR}/embed_f16.cubin".encode()))
        self.embed_func = CUfunction()
        _check("getfunc embed", cuda.cuModuleGetFunction(byref(self.embed_func), mod5, b"embed_f16"))

        print("    gptq_gemv_safe, gptq_matvec, norm, activate, embed_f16: loaded")

    def _alloc_buffers(self):
        cuda = self.cuda
        self.bufs = {}

        def alloc(name, nbytes):
            d = CUdeviceptr()
            _check(f"alloc {name}", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
            self.bufs[name] = d
            return d

        self.d_x = alloc("x", HIDDEN_DIM * 4)
        self.d_residual = alloc("residual", HIDDEN_DIM * 4)
        self.d_norm_out = alloc("norm_out", HIDDEN_DIM * 4)
        self.d_norm_weight = alloc("norm_w", HIDDEN_DIM * 4)
        self.d_gate = alloc("gate", INTERMEDIATE_SIZE * 4)
        self.d_up = alloc("up", INTERMEDIATE_SIZE * 4)
        self.d_act = alloc("act", INTERMEDIATE_SIZE * 4)
        self.d_down = alloc("down", HIDDEN_DIM * 4)
        self.d_zero = alloc("zero", HIDDEN_DIM * 4)
        # Zero out the zero buffer
        _check("memset", cuda.cuMemsetD8_v2(self.d_zero, 0, c_size_t(HIDDEN_DIM * 4)))

        # For lm_head output (vocab_size floats)
        self.d_logits = alloc("logits", VOCAB_SIZE * 4)

        # Scratch for attention projections
        self.d_attn_scratch1 = alloc("attn1", 12288 * 4)
        self.d_attn_scratch2 = alloc("attn2", 6144 * 4)
        self.d_attn_out = alloc("attn_out", HIDDEN_DIM * 4)

        total_mb = sum(1 for _ in self.bufs) * 0.1  # rough
        print(f"    GPU buffers allocated ({len(self.bufs)} buffers)")

    def _preload_norms(self):
        self.input_norms = []
        self.post_attn_norms = []
        for i in range(NUM_LAYERS):
            prefix = f"model.language_model.layers.{i}"
            self.input_norms.append(
                load_norm_weight(self.model, f"{prefix}.input_layernorm.weight") + 1.0)
            self.post_attn_norms.append(
                load_norm_weight(self.model, f"{prefix}.post_attention_layernorm.weight") + 1.0)
        self.final_norm_w = load_norm_weight(
            self.model, "model.language_model.norm.weight") + 1.0
        print(f"    {NUM_LAYERS * 2 + 1} norm weights preloaded")

    def _launch(self, func, grid, block, args, shared_mem=0):
        ptrs = (c_void_p * len(args))()
        for i, a in enumerate(args):
            ptrs[i] = ctypes.cast(byref(a), c_void_p).value
        r = self.cuda.cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            self.stream,
            ptrs, None
        )
        if r != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {r}")

    def sync(self):
        _check("sync", self.cuda.cuStreamSynchronize(self.stream))

    def upload_f32(self, dptr, data: np.ndarray):
        assert data.dtype == np.float32
        _check("HtoD", self.cuda.cuMemcpyHtoD_v2(
            dptr, data.ctypes.data_as(c_void_p), c_size_t(data.nbytes)))

    def download_f32(self, dptr, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        _check("DtoH", self.cuda.cuMemcpyDtoH_v2(
            out.ctypes.data_as(c_void_p), dptr, c_size_t(n * 4)))
        return out

    # ------------------------------------------------------------------
    # GPU kernel wrappers
    # ------------------------------------------------------------------

    def gpu_embed(self, token_id: int):
        """Embed token using F16 lookup kernel. Result in d_x."""
        embed_ptr = self.model.weight_info(
            "model.language_model.embed_tokens.weight").ptr
        BLOCK = 256
        GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))
        self._launch(
            self.embed_func,
            (GRID, 1, 1), (BLOCK, 1, 1),
            [c_uint32(token_id), c_uint64(embed_ptr), c_uint64(self.d_x.value)],
        )

    def gpu_norm(self, d_input, d_residual, norm_w: np.ndarray, d_output):
        """RMSNorm: output = norm(input + residual) * weight.

        Uses cached norm.cubin which has hidden_dim=5120 baked in (5 params).
        """
        self.upload_f32(self.d_norm_weight, norm_w)
        self._launch(
            self.norm_func,
            (1, 1, 1), (256, 1, 1),
            [c_uint64(d_input.value), c_uint64(d_residual.value),
             c_uint64(self.d_norm_weight.value), c_uint64(d_output.value),
             c_float(self.epsilon)],
            shared_mem=128,
        )

    def gpu_gemv_ultra(self, prefix: str, d_input, d_output, K: int, N: int):
        """Optimized GPTQ GEMV: output = input @ dequant(qweight).

        Uses gptq_gemv_safe with scalar loads (alignment-safe).
        """
        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr

        # Zero output
        _check("memset", self.cuda.cuMemsetD8_v2(d_output, 0, c_size_t(N * 4)))

        K_packed = K // 8
        # Choose K_SPLITS: find best divisor of K_packed that gives good parallelism
        # without exceeding shared memory (k_packed_per_split * 32 <= 48KB)
        K_SPLITS = 1
        for s in [80, 64, 40, 32, 20, 16, 10, 8, 5, 4, 2, 1]:
            if K_packed % s == 0:
                k_pps = K_packed // s
                smem = k_pps * 32
                if smem <= 48 * 1024:
                    K_SPLITS = s
                    break

        k_pps = K_packed // K_SPLITS
        grid_x = (N + 255) // 256
        smem = k_pps * 32

        self._launch(
            self.gemv_func,
            (grid_x, K_SPLITS, 1), (64, 1, 1),
            [c_uint64(qw_ptr), c_uint64(sc_ptr), c_uint64(d_input.value),
             c_uint64(d_output.value), c_uint32(N), c_uint32(K),
             c_uint32(K_SPLITS), c_uint32(k_pps)],
            shared_mem=smem,
        )

    def gpu_gemv_matvec(self, prefix: str, d_input, d_output, K: int, N: int):
        """Original GPTQ matvec: 1 block per output element (slow baseline)."""
        qw_ptr = self.model.weight_info(f"{prefix}.qweight").ptr
        sc_ptr = self.model.weight_info(f"{prefix}.scales").ptr

        _check("memset", self.cuda.cuMemsetD8_v2(d_output, 0, c_size_t(N * 4)))
        self._launch(
            self.matvec_func,
            (N, 1, 1), (256, 1, 1),
            [c_uint64(qw_ptr), c_uint64(sc_ptr), c_uint64(d_input.value),
             c_uint64(d_output.value), c_uint32(N), c_uint32(K)],
            shared_mem=32,
        )

    def gpu_activate(self, d_gate, d_up, d_output, size: int):
        """SiLU: output = silu(gate) * up."""
        GRID = max(1, math.ceil(size / 256))
        self._launch(
            self.activate_func,
            (GRID, 1, 1), (256, 1, 1),
            [c_uint64(d_up.value), c_uint64(d_gate.value), c_uint64(d_output.value)],
        )

    # ------------------------------------------------------------------
    # Layer execution (MLP-only for benchmarking)
    # ------------------------------------------------------------------

    def run_mlp_layer(self, layer_idx: int, use_ultra=True):
        """Run MLP sublayer: norm -> gate/up -> activate -> down."""
        prefix = f"model.language_model.layers.{layer_idx}"
        gemv = self.gpu_gemv_ultra if use_ultra else self.gpu_gemv_matvec

        # Post-attention norm
        self.gpu_norm(self.d_x, self.d_zero,
                      self.post_attn_norms[layer_idx], self.d_norm_out)

        # Gate: 5120 -> 17408
        gemv(f"{prefix}.mlp.gate_proj", self.d_norm_out, self.d_gate,
             HIDDEN_DIM, INTERMEDIATE_SIZE)

        # Up: 5120 -> 17408
        gemv(f"{prefix}.mlp.up_proj", self.d_norm_out, self.d_up,
             HIDDEN_DIM, INTERMEDIATE_SIZE)

        # SiLU(gate) * up
        self.gpu_activate(self.d_gate, self.d_up, self.d_act, INTERMEDIATE_SIZE)

        # Down: 17408 -> 5120
        gemv(f"{prefix}.mlp.down_proj", self.d_act, self.d_down,
             INTERMEDIATE_SIZE, HIDDEN_DIM)

    def cpu_embed(self, token_id: int) -> np.ndarray:
        """Embed token on CPU: read F16 row from mmap'd safetensors, convert to F32."""
        ti = self.model.weight_info("model.language_model.embed_tokens.weight")
        raw = bytes(self.model.weight_bytes("model.language_model.embed_tokens.weight"))
        row_bytes = HIDDEN_DIM * 2  # F16
        offset = token_id * row_bytes
        row = np.frombuffer(raw[offset:offset + row_bytes], dtype=np.float16).astype(np.float32)
        return row

    def run_full_token(self, token_id: int, use_ultra=True):
        """Run embed + 64 MLP layers + final norm.

        Skips attention/DeltaNet for now (MLP-only pass).
        Returns hidden state as numpy array.
        """
        # Embed on CPU (avoids unified memory alignment issues with GPU embed kernel)
        x = self.cpu_embed(token_id)

        for layer_idx in range(NUM_LAYERS):
            # Input norm
            self.upload_f32(self.d_x, x)
            self.gpu_norm(self.d_x, self.d_zero,
                          self.input_norms[layer_idx], self.d_norm_out)

            # MLP (attention skipped)
            self.run_mlp_layer(layer_idx, use_ultra)
            self.sync()

            # Residual add (CPU)
            mlp_out = self.download_f32(self.d_down, HIDDEN_DIM)
            x = x + mlp_out

        # Final norm (CPU, to avoid another round-trip)
        rms = np.sqrt(np.mean(x ** 2) + self.epsilon)
        x_normed = (x / rms) * self.final_norm_w
        return x_normed

    def compute_lm_head(self, x_normed: np.ndarray) -> int:
        """Project through lm_head and argmax (CPU, chunked)."""
        lm_head_raw = bytes(self.model.weight_bytes("lm_head.weight"))
        best_val = -1e30
        best_idx = 0
        CHUNK = 8192
        for start in range(0, VOCAB_SIZE, CHUNK):
            end = min(start + CHUNK, VOCAB_SIZE)
            chunk_size = end - start
            offset = start * HIDDEN_DIM * 2
            chunk_bytes = chunk_size * HIDDEN_DIM * 2
            w_chunk = np.frombuffer(
                lm_head_raw[offset:offset + chunk_bytes],
                dtype=np.float16,
            ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
            logits = w_chunk @ x_normed
            idx = np.argmax(logits)
            if logits[idx] > best_val:
                best_val = logits[idx]
                best_idx = start + int(idx)
        return best_idx

    def close(self):
        for d in self.bufs.values():
            self.cuda.cuMemFree_v2(d)
        self.cuda.cuEventDestroy_v2(self.ev_start)
        self.cuda.cuEventDestroy_v2(self.ev_stop)
        if getattr(self, '_using_primary', False):
            dev = CUdevice()
            self.cuda.cuDeviceGet(byref(dev), 0)
            self.cuda.cuDevicePrimaryCtxRelease = self.cuda.cuDevicePrimaryCtxRelease
            self.cuda.cuDevicePrimaryCtxRelease.argtypes = [CUdevice]
            self.cuda.cuDevicePrimaryCtxRelease.restype = CUresult
            self.cuda.cuDevicePrimaryCtxRelease(dev)
        else:
            self.cuda.cuCtxDestroy_v2(self.ctx)


def bench_single_layer(engine: NativeBench):
    """Benchmark a single MLP layer with both kernels."""
    print("\n" + "=" * 72)
    print("  Single Layer MLP Benchmark")
    print("=" * 72)

    # Warm up with embed
    engine.gpu_embed(760)
    engine.sync()

    for label, use_ultra in [("gptq_matvec (baseline)", False),
                              ("gptq_gemv_safe (optimized)", True)]:
        # Warm up
        for _ in range(3):
            engine.run_mlp_layer(0, use_ultra)
        engine.sync()

        # Benchmark
        _check("ev", engine.cuda.cuEventRecord(engine.ev_start, engine.stream))
        N_ITERS = 20
        for _ in range(N_ITERS):
            engine.run_mlp_layer(0, use_ultra)
        _check("ev", engine.cuda.cuEventRecord(engine.ev_stop, engine.stream))
        _check("ev", engine.cuda.cuEventSynchronize(engine.ev_stop))
        ms = c_float()
        _check("ev", engine.cuda.cuEventElapsedTime(byref(ms), engine.ev_start, engine.ev_stop))
        ms_per = ms.value / N_ITERS

        print(f"  {label}: {ms_per:.2f} ms/layer")
        print(f"    -> {ms_per * NUM_LAYERS:.1f} ms/token (MLP only)")
        print(f"    -> {1000.0 / (ms_per * NUM_LAYERS):.2f} tok/s (MLP only)")


def bench_full_token(engine: NativeBench, prompt: str, max_tokens: int):
    """Run full MLP-only inference and measure tok/s."""
    print("\n" + "=" * 72)
    print("  Full Token Generation (MLP-only, 64 layers)")
    print("=" * 72)

    token_ids = engine.tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Tokens: {token_ids}")

    # Process last prompt token
    tid = token_ids[-1]
    t0 = time.monotonic()
    x_normed = engine.run_full_token(tid, use_ultra=True)
    next_id = engine.compute_lm_head(x_normed)
    t1 = time.monotonic()
    print(f"  First token: {t1-t0:.3f}s -> {engine.tok.decode([next_id])!r}")

    # Generate more tokens
    times = []
    generated = [next_id]
    for i in range(max_tokens - 1):
        t0 = time.monotonic()
        x_normed = engine.run_full_token(next_id, use_ultra=True)
        next_id = engine.compute_lm_head(x_normed)
        t1 = time.monotonic()
        times.append(t1 - t0)
        generated.append(next_id)
        text = engine.tok.decode([next_id])
        print(f"  Token {i+2}: {t1-t0:.3f}s -> {text!r}")
        if next_id == EOS_TOKEN_ID:
            break

    if times:
        avg = sum(times) / len(times)
        print(f"\n  Avg decode: {avg:.3f}s/token = {1/avg:.2f} tok/s")
    print(f"  Output: {engine.tok.decode(generated)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--layer-bench", action="store_true",
                        help="Run single-layer benchmark only")
    args = parser.parse_args()

    engine = NativeBench()

    if args.layer_bench:
        bench_single_layer(engine)
    else:
        bench_single_layer(engine)
        bench_full_token(engine, args.prompt, args.tokens)

    engine.close()


if __name__ == "__main__":
    main()
