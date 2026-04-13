#!/usr/bin/env python3
"""
Lithos Benchmark Suite -- GH200 480GB

Measures Lithos kernel performance against cuBLAS baselines:
  1. Memory bandwidth (copy baseline)
  2. GEMV: Lithos PROJECTION vs cuBLAS sgemv
  3. RMSNorm: Lithos NORM kernel
  4. Kernel launch overhead
  5. Cubin load time (all 7 kernels)
  6. PTX compilation time (ptxas)

Requires: GH200 with CUDA 12.x, libcuda.so.1, libcublas.so
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import subprocess
import sys
import tempfile
import time
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
GH200_PEAK_BW_TBS = 4.0  # TB/s theoretical HBM3 peak
WARMUP_ITERS = 10
BENCH_ITERS = 100

# ---------------------------------------------------------------------------
# CUDA driver types
# ---------------------------------------------------------------------------
CUresult = c_int
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUstream = c_void_p
CUdeviceptr = c_uint64
CUevent = c_void_p


class CUDAError(RuntimeError):
    def __init__(self, func: str, code: int):
        super().__init__(f"{func} returned {code}")
        self.code = code


def _check(func: str, result: int):
    if result != 0:
        raise CUDAError(func, result)


# ---------------------------------------------------------------------------
# Load CUDA driver
# ---------------------------------------------------------------------------
def _setup_cuda_driver():
    lib = ctypes.CDLL("libcuda.so.1")

    lib.cuInit.argtypes = [c_uint]
    lib.cuInit.restype = CUresult
    lib.cuDeviceGet.argtypes = [POINTER(CUdevice), c_int]
    lib.cuDeviceGet.restype = CUresult
    lib.cuDeviceGetName.argtypes = [c_char_p, c_int, CUdevice]
    lib.cuDeviceGetName.restype = CUresult
    lib.cuCtxCreate_v2.argtypes = [POINTER(CUcontext), c_uint, CUdevice]
    lib.cuCtxCreate_v2.restype = CUresult
    lib.cuCtxDestroy_v2.argtypes = [CUcontext]
    lib.cuCtxDestroy_v2.restype = CUresult
    lib.cuCtxSynchronize.argtypes = []
    lib.cuCtxSynchronize.restype = CUresult
    lib.cuModuleLoadData.argtypes = [POINTER(CUmodule), c_void_p]
    lib.cuModuleLoadData.restype = CUresult
    lib.cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, c_char_p]
    lib.cuModuleGetFunction.restype = CUresult
    lib.cuModuleUnload.argtypes = [CUmodule]
    lib.cuModuleUnload.restype = CUresult

    lib.cuLaunchKernel.argtypes = [
        CUfunction,
        c_uint, c_uint, c_uint,
        c_uint, c_uint, c_uint,
        c_uint, CUstream, c_void_p, c_void_p,
    ]
    lib.cuLaunchKernel.restype = CUresult

    lib.cuMemAlloc_v2.argtypes = [POINTER(CUdeviceptr), c_size_t]
    lib.cuMemAlloc_v2.restype = CUresult
    lib.cuMemFree_v2.argtypes = [CUdeviceptr]
    lib.cuMemFree_v2.restype = CUresult
    lib.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, c_void_p, c_size_t]
    lib.cuMemcpyHtoD_v2.restype = CUresult
    lib.cuMemcpyDtoD_v2.argtypes = [CUdeviceptr, CUdeviceptr, c_size_t]
    lib.cuMemcpyDtoD_v2.restype = CUresult
    lib.cuMemsetD8_v2.argtypes = [CUdeviceptr, ctypes.c_ubyte, c_size_t]
    lib.cuMemsetD8_v2.restype = CUresult

    # Events
    lib.cuEventCreate.argtypes = [POINTER(CUevent), c_uint]
    lib.cuEventCreate.restype = CUresult
    lib.cuEventRecord.argtypes = [CUevent, CUstream]
    lib.cuEventRecord.restype = CUresult
    lib.cuEventSynchronize.argtypes = [CUevent]
    lib.cuEventSynchronize.restype = CUresult
    lib.cuEventElapsedTime.argtypes = [POINTER(c_float), CUevent, CUevent]
    lib.cuEventElapsedTime.restype = CUresult
    lib.cuEventDestroy_v2.argtypes = [CUevent]
    lib.cuEventDestroy_v2.restype = CUresult

    # Stream
    lib.cuStreamCreate.argtypes = [POINTER(CUstream), c_uint]
    lib.cuStreamCreate.restype = CUresult
    lib.cuStreamSynchronize.argtypes = [CUstream]
    lib.cuStreamSynchronize.restype = CUresult

    return lib


# ---------------------------------------------------------------------------
# Load cuBLAS
# ---------------------------------------------------------------------------
def _setup_cublas():
    lib = ctypes.CDLL("libcublas.so")

    # cublasCreate_v2
    lib.cublasCreate_v2.argtypes = [POINTER(c_void_p)]
    lib.cublasCreate_v2.restype = c_int
    # cublasDestroy_v2
    lib.cublasDestroy_v2.argtypes = [c_void_p]
    lib.cublasDestroy_v2.restype = c_int
    # cublasSetStream_v2
    lib.cublasSetStream_v2.argtypes = [c_void_p, CUstream]
    lib.cublasSetStream_v2.restype = c_int
    # cublasSgemv_v2:  handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
    lib.cublasSgemv_v2.argtypes = [
        c_void_p,  # handle
        c_int,     # trans (0=CUBLAS_OP_N, 1=CUBLAS_OP_T)
        c_int,     # m
        c_int,     # n
        POINTER(c_float),  # alpha
        CUdeviceptr,       # A (device)
        c_int,     # lda
        CUdeviceptr,       # x (device)
        c_int,     # incx
        POINTER(c_float),  # beta
        CUdeviceptr,       # y (device)
        c_int,     # incy
    ]
    lib.cublasSgemv_v2.restype = c_int

    return lib


# ---------------------------------------------------------------------------
# Helper: CUDA event pair for GPU timing
# ---------------------------------------------------------------------------
class GPUTimer:
    def __init__(self, cuda):
        self.cuda = cuda
        self.start_ev = CUevent()
        self.stop_ev = CUevent()
        _check("cuEventCreate", cuda.cuEventCreate(byref(self.start_ev), 0))
        _check("cuEventCreate", cuda.cuEventCreate(byref(self.stop_ev), 0))

    def start(self, stream=None):
        _check("cuEventRecord", self.cuda.cuEventRecord(self.start_ev, stream))

    def stop(self, stream=None):
        _check("cuEventRecord", self.cuda.cuEventRecord(self.stop_ev, stream))
        _check("cuEventSynchronize", self.cuda.cuEventSynchronize(self.stop_ev))

    def elapsed_ms(self) -> float:
        ms = c_float()
        _check("cuEventElapsedTime",
               self.cuda.cuEventElapsedTime(byref(ms), self.start_ev, self.stop_ev))
        return ms.value

    def destroy(self):
        self.cuda.cuEventDestroy_v2(self.start_ev)
        self.cuda.cuEventDestroy_v2(self.stop_ev)


# ---------------------------------------------------------------------------
# Helper: device alloc
# ---------------------------------------------------------------------------
def dev_alloc(cuda, nbytes):
    dptr = CUdeviceptr()
    _check("cuMemAlloc_v2", cuda.cuMemAlloc_v2(byref(dptr), c_size_t(nbytes)))
    # Zero-fill so values are deterministic
    _check("cuMemsetD8_v2", cuda.cuMemsetD8_v2(dptr, 0, c_size_t(nbytes)))
    return dptr


def dev_free(cuda, dptr):
    cuda.cuMemFree_v2(dptr)


# ---------------------------------------------------------------------------
# 1. Memory Bandwidth (D2D copy)
# ---------------------------------------------------------------------------
def bench_memory_bandwidth(cuda, timer):
    SIZE = 256 * 1024 * 1024  # 256 MiB
    src = dev_alloc(cuda, SIZE)
    dst = dev_alloc(cuda, SIZE)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _check("cuMemcpyDtoD_v2", cuda.cuMemcpyDtoD_v2(dst, src, c_size_t(SIZE)))
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    # Bench
    timer.start()
    for _ in range(BENCH_ITERS):
        _check("cuMemcpyDtoD_v2", cuda.cuMemcpyDtoD_v2(dst, src, c_size_t(SIZE)))
    timer.stop()

    elapsed_ms = timer.elapsed_ms()
    total_bytes = SIZE * BENCH_ITERS * 2  # read + write
    bw_tbs = (total_bytes / 1e12) / (elapsed_ms / 1e3)
    pct = bw_tbs / GH200_PEAK_BW_TBS * 100

    dev_free(cuda, src)
    dev_free(cuda, dst)
    return bw_tbs, pct


# ---------------------------------------------------------------------------
# 2. GEMV Benchmark -- cuBLAS sgemv
# ---------------------------------------------------------------------------
def bench_cublas_gemv(cuda, cublas, handle, timer, M, K):
    """cuBLAS sgemv: y = A*x where A is MxK (column-major), x is Kx1, y is Mx1."""
    mat_bytes = M * K * 4
    x_bytes = K * 4
    y_bytes = M * 4

    d_A = dev_alloc(cuda, mat_bytes)
    d_x = dev_alloc(cuda, x_bytes)
    d_y = dev_alloc(cuda, y_bytes)

    alpha = c_float(1.0)
    beta = c_float(0.0)

    # Warmup
    for _ in range(WARMUP_ITERS):
        cublas.cublasSgemv_v2(
            handle, 0,  # CUBLAS_OP_N
            M, K, byref(alpha),
            d_A, M,
            d_x, 1,
            byref(beta),
            d_y, 1,
        )
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    # Bench
    timer.start()
    for _ in range(BENCH_ITERS):
        cublas.cublasSgemv_v2(
            handle, 0,
            M, K, byref(alpha),
            d_A, M,
            d_x, 1,
            byref(beta),
            d_y, 1,
        )
    timer.stop()

    elapsed_ms = timer.elapsed_ms()
    # Data movement: read A (M*K*4) + read x (K*4) + write y (M*4), dominated by A
    total_bytes = (M * K * 4 + K * 4 + M * 4) * BENCH_ITERS
    bw_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3)
    pct = (bw_gbs / 1e3) / GH200_PEAK_BW_TBS * 100

    dev_free(cuda, d_A)
    dev_free(cuda, d_x)
    dev_free(cuda, d_y)
    return bw_gbs, pct, elapsed_ms / BENCH_ITERS


# ---------------------------------------------------------------------------
# 3. GEMV Benchmark -- Lithos PROJECTION kernel
# ---------------------------------------------------------------------------
def bench_lithos_projection(cuda, timer, M, K):
    """Lithos projection kernel: W4A16 GEMV.

    Kernel params: weight_ptr(u64), scale_ptr(u64), input_ptr(u64),
                   output_ptr(u64), M(u32), K(u32)

    For benchmarking bandwidth, we allocate realistic buffers.
    W4 packing: each weight is 4 bits, so weight matrix is M*K/2 bytes.
    Scales: one f16 per group of 128 elements, so M*(K/128)*2 bytes.
    Input: K f16 values = K*2 bytes.
    Output: M f32 values = M*4 bytes.
    """
    cubin_path = KERNEL_DIR / "projection.cubin"
    if not cubin_path.exists():
        return None, None, None

    cubin_data = cubin_path.read_bytes()
    mod = CUmodule()
    _check("cuModuleLoadData", cuda.cuModuleLoadData(byref(mod), cubin_data))
    func = CUfunction()
    _check("cuModuleGetFunction",
           cuda.cuModuleGetFunction(byref(func), mod, b"projection"))

    # Allocate buffers
    weight_bytes = M * K // 2  # 4-bit packed
    group_size = 128
    n_groups = M * (K // group_size)
    scale_bytes = n_groups * 2  # f16 scales
    input_bytes = K * 2  # f16
    output_bytes = M * 4  # f32

    d_weight = dev_alloc(cuda, weight_bytes)
    d_scale = dev_alloc(cuda, scale_bytes)
    d_input = dev_alloc(cuda, input_bytes)
    d_output = dev_alloc(cuda, output_bytes)

    # Grid: each block handles 16 rows
    grid_x = (M + 15) // 16
    block_x = 128  # typical for MMA kernels

    # Pack args
    args = [
        c_uint64(d_weight.value),
        c_uint64(d_scale.value),
        c_uint64(d_input.value),
        c_uint64(d_output.value),
        c_uint32(M),
        c_uint32(K),
    ]
    n = len(args)
    param_ptrs = (c_void_p * n)()
    for i, a in enumerate(args):
        param_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

    # Warmup
    for _ in range(WARMUP_ITERS):
        _check("cuLaunchKernel", cuda.cuLaunchKernel(
            func,
            grid_x, 1, 1,
            block_x, 1, 1,
            6144, None,
            param_ptrs, None,
        ))
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    # Bench
    timer.start()
    for _ in range(BENCH_ITERS):
        _check("cuLaunchKernel", cuda.cuLaunchKernel(
            func,
            grid_x, 1, 1,
            block_x, 1, 1,
            6144, None,
            param_ptrs, None,
        ))
    timer.stop()

    elapsed_ms = timer.elapsed_ms()
    # Effective data read: weight (W4) + scales + input + write output
    total_bytes = (weight_bytes + scale_bytes + input_bytes + output_bytes) * BENCH_ITERS
    bw_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3)
    pct = (bw_gbs / 1e3) / GH200_PEAK_BW_TBS * 100

    dev_free(cuda, d_weight)
    dev_free(cuda, d_scale)
    dev_free(cuda, d_input)
    dev_free(cuda, d_output)
    cuda.cuModuleUnload(mod)

    return bw_gbs, pct, elapsed_ms / BENCH_ITERS


# ---------------------------------------------------------------------------
# 4. RMSNorm Benchmark -- Lithos NORM kernel
# ---------------------------------------------------------------------------
def bench_lithos_norm(cuda, timer, hidden_dim=5120):
    """Lithos norm kernel: output = rmsnorm(input + residual) * weight.

    Params: input_ptr(u64), residual_ptr(u64), weight_ptr(u64),
            output_ptr(u64), hidden_dim(u32), epsilon(f32)
    """
    cubin_path = KERNEL_DIR / "norm.cubin"
    if not cubin_path.exists():
        return None

    cubin_data = cubin_path.read_bytes()
    mod = CUmodule()
    _check("cuModuleLoadData", cuda.cuModuleLoadData(byref(mod), cubin_data))
    func = CUfunction()
    _check("cuModuleGetFunction",
           cuda.cuModuleGetFunction(byref(func), mod, b"norm"))

    buf_bytes = hidden_dim * 4  # f32
    d_input = dev_alloc(cuda, buf_bytes)
    d_residual = dev_alloc(cuda, buf_bytes)
    d_weight = dev_alloc(cuda, buf_bytes)
    d_output = dev_alloc(cuda, buf_bytes)

    # Single row, one block
    args = [
        c_uint64(d_input.value),
        c_uint64(d_residual.value),
        c_uint64(d_weight.value),
        c_uint64(d_output.value),
        c_uint32(hidden_dim),
        c_float(1e-6),
    ]
    n = len(args)
    param_ptrs = (c_void_p * n)()
    for i, a in enumerate(args):
        param_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

    block_x = 256

    # Warmup
    for _ in range(WARMUP_ITERS):
        _check("cuLaunchKernel", cuda.cuLaunchKernel(
            func, 1, 1, 1, block_x, 1, 1, 128, None, param_ptrs, None,
        ))
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    # Bench (many iters for small kernel)
    iters = 1000
    timer.start()
    for _ in range(iters):
        _check("cuLaunchKernel", cuda.cuLaunchKernel(
            func, 1, 1, 1, block_x, 1, 1, 128, None, param_ptrs, None,
        ))
    timer.stop()

    elapsed_us = timer.elapsed_ms() * 1000.0 / iters

    dev_free(cuda, d_input)
    dev_free(cuda, d_residual)
    dev_free(cuda, d_weight)
    dev_free(cuda, d_output)
    cuda.cuModuleUnload(mod)

    return elapsed_us


# ---------------------------------------------------------------------------
# 5. Kernel Launch Overhead
# ---------------------------------------------------------------------------
def bench_launch_overhead(cuda, timer):
    """Measure the dispatch overhead of cuLaunchKernel with a pre-loaded cubin
    vs a cuBLAS sgemv call.

    We launch a minimal kernel (embed with tiny data) to measure pure overhead.
    """
    cubin_path = KERNEL_DIR / "embed.cubin"
    if not cubin_path.exists():
        return None, None

    cubin_data = cubin_path.read_bytes()
    mod = CUmodule()
    _check("cuModuleLoadData", cuda.cuModuleLoadData(byref(mod), cubin_data))
    func = CUfunction()
    _check("cuModuleGetFunction",
           cuda.cuModuleGetFunction(byref(func), mod, b"embed"))

    # Minimal buffers -- embed just needs table + output for 1 token
    hidden_dim = 16
    d_table = dev_alloc(cuda, hidden_dim * 4)
    d_output = dev_alloc(cuda, hidden_dim * 4)

    args = [
        c_uint32(0),                       # token_id
        c_uint64(d_table.value),           # embed_table_ptr
        c_uint64(d_output.value),          # output_ptr
        c_uint32(hidden_dim),              # hidden_dim
    ]
    n = len(args)
    param_ptrs = (c_void_p * n)()
    for i, a in enumerate(args):
        param_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

    # Warmup
    for _ in range(WARMUP_ITERS):
        _check("cuLaunchKernel", cuda.cuLaunchKernel(
            func, 1, 1, 1, 32, 1, 1, 0, None, param_ptrs, None,
        ))
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    # CPU-side timing: measures host dispatch overhead (not including GPU execution)
    iters = 10000
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())
    t0 = time.perf_counter()
    for _ in range(iters):
        cuda.cuLaunchKernel(
            func, 1, 1, 1, 32, 1, 1, 0, None, param_ptrs, None,
        )
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())
    t1 = time.perf_counter()
    lithos_overhead_us = (t1 - t0) / iters * 1e6

    dev_free(cuda, d_table)
    dev_free(cuda, d_output)
    cuda.cuModuleUnload(mod)

    # cuBLAS sgemv overhead -- tiny 4x4 problem
    cublas = _setup_cublas()
    handle = c_void_p()
    cublas.cublasCreate_v2(byref(handle))

    d_A = dev_alloc(cuda, 4 * 4 * 4)
    d_x = dev_alloc(cuda, 4 * 4)
    d_y = dev_alloc(cuda, 4 * 4)
    alpha = c_float(1.0)
    beta = c_float(0.0)

    # Warmup
    for _ in range(WARMUP_ITERS):
        cublas.cublasSgemv_v2(handle, 0, 4, 4, byref(alpha), d_A, 4, d_x, 1, byref(beta), d_y, 1)
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())

    t0 = time.perf_counter()
    for _ in range(iters):
        cublas.cublasSgemv_v2(handle, 0, 4, 4, byref(alpha), d_A, 4, d_x, 1, byref(beta), d_y, 1)
    _check("cuCtxSynchronize", cuda.cuCtxSynchronize())
    t1 = time.perf_counter()
    cublas_overhead_us = (t1 - t0) / iters * 1e6

    dev_free(cuda, d_A)
    dev_free(cuda, d_x)
    dev_free(cuda, d_y)
    cublas.cublasDestroy_v2(handle)

    return lithos_overhead_us, cublas_overhead_us


# ---------------------------------------------------------------------------
# 6. Cubin Load Time
# ---------------------------------------------------------------------------
def bench_cubin_load(cuda):
    """Measure time to load each of the 7 cubins via cuModuleLoadData."""
    kernel_names = [
        "projection", "attention_score", "norm", "activate",
        "rotate", "sample", "embed",
    ]

    cubin_data = {}
    for name in kernel_names:
        path = KERNEL_DIR / f"{name}.cubin"
        if path.exists():
            cubin_data[name] = path.read_bytes()

    if not cubin_data:
        return None, {}

    # Warmup: load/unload once
    for name, data in cubin_data.items():
        mod = CUmodule()
        cuda.cuModuleLoadData(byref(mod), data)
        cuda.cuModuleUnload(mod)

    # Bench
    times = {}
    total_ms = 0.0
    iters = 20
    for name, data in cubin_data.items():
        t0 = time.perf_counter()
        for _ in range(iters):
            mod = CUmodule()
            _check("cuModuleLoadData", cuda.cuModuleLoadData(byref(mod), data))
            cuda.cuModuleUnload(mod)
        t1 = time.perf_counter()
        per_ms = (t1 - t0) / iters * 1000.0
        times[name] = per_ms
        total_ms += per_ms

    return total_ms, times


# ---------------------------------------------------------------------------
# 7. PTX Compilation Time
# ---------------------------------------------------------------------------
def bench_ptx_compile():
    """Measure time for ptxas to compile projection.ptx to cubin."""
    ptx_path = KERNEL_DIR / "projection.ptx"
    if not ptx_path.exists():
        return None

    # Warmup: compile once
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=True) as tmp:
        subprocess.run(
            ["ptxas", "-arch=sm_90", "-o", tmp.name, str(ptx_path)],
            capture_output=True, check=True,
        )

    # Bench
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=True) as tmp:
            subprocess.run(
                ["ptxas", "-arch=sm_90", "-o", tmp.name, str(ptx_path)],
                capture_output=True, check=True,
            )
    t1 = time.perf_counter()

    return (t1 - t0) / iters * 1000.0  # ms


# ===========================================================================
# Main
# ===========================================================================
def main():
    cuda = _setup_cuda_driver()
    _check("cuInit", cuda.cuInit(0))

    # Device
    device = CUdevice()
    _check("cuDeviceGet", cuda.cuDeviceGet(byref(device), 0))
    name_buf = ctypes.create_string_buffer(256)
    _check("cuDeviceGetName", cuda.cuDeviceGetName(name_buf, 256, device))
    device_name = name_buf.value.decode()

    # Context
    ctx = CUcontext()
    _check("cuCtxCreate_v2", cuda.cuCtxCreate_v2(byref(ctx), 0, device))

    # cuBLAS
    cublas = _setup_cublas()
    cublas_handle = c_void_p()
    ret = cublas.cublasCreate_v2(byref(cublas_handle))
    if ret != 0:
        print(f"cublasCreate failed: {ret}")
        sys.exit(1)

    timer = GPUTimer(cuda)

    # ===================================================================
    print(f"\nLITHOS BENCHMARK SUITE -- {device_name}")
    print("=" * 56)

    # -- 1. Memory bandwidth -------------------------------------------
    bw_tbs, bw_pct = bench_memory_bandwidth(cuda, timer)
    print(f"Memory bandwidth (D2D copy):  {bw_tbs:.2f} TB/s ({bw_pct:.0f}% of {GH200_PEAK_BW_TBS:.0f} TB/s peak)")

    # -- 2. GEMV -------------------------------------------------------
    gemv_dims = [
        (5120, 5120),
        (5120, 17408),
        (17408, 5120),
    ]

    for M, K in gemv_dims:
        print(f"\nGEMV {M}x{K}:")

        # cuBLAS f32
        cb_gbs, cb_pct, cb_per_ms = bench_cublas_gemv(cuda, cublas, cublas_handle, timer, M, K)
        print(f"  cuBLAS sgemv:           {cb_gbs:8.1f} GB/s ({cb_pct:5.1f}% of peak)  [{cb_per_ms*1000:.1f} us/call]")

        # Lithos W4A16 projection
        li_gbs, li_pct, li_per_ms = bench_lithos_projection(cuda, timer, M, K)
        if li_gbs is not None:
            print(f"  Lithos PROJECTION (W4): {li_gbs:8.1f} GB/s ({li_pct:5.1f}% of peak)  [{li_per_ms*1000:.1f} us/call]")
        else:
            print(f"  Lithos PROJECTION:      (cubin not found)")

    # -- 3. RMSNorm ----------------------------------------------------
    print(f"\nRMSNorm hidden_dim=5120:")
    norm_us = bench_lithos_norm(cuda, timer, hidden_dim=5120)
    if norm_us is not None:
        print(f"  Lithos NORM:            {norm_us:.2f} us")
    else:
        print(f"  Lithos NORM:            (cubin not found)")

    # -- 4. Kernel launch overhead -------------------------------------
    print(f"\nKernel launch overhead (CPU-side, incl. sync):")
    lithos_oh, cublas_oh = bench_launch_overhead(cuda, timer)
    if lithos_oh is not None:
        print(f"  cuLaunchKernel (Lithos): {lithos_oh:.2f} us")
        print(f"  cuBLAS sgemv call:       {cublas_oh:.2f} us")
    else:
        print(f"  (cubin not found)")

    # -- 5. Cubin load time --------------------------------------------
    print(f"\nCubin load time (cuModuleLoadData):")
    total_ms, per_kernel = bench_cubin_load(cuda)
    if total_ms is not None:
        for name, ms in per_kernel.items():
            print(f"  {name:20s}  {ms:.3f} ms")
        print(f"  {'TOTAL (7 kernels)':20s}  {total_ms:.3f} ms")
    else:
        print(f"  (no cubins found)")

    # -- 6. PTX compile time -------------------------------------------
    print(f"\nPTX compile time (ptxas -arch=sm_90):")
    ptx_ms = bench_ptx_compile()
    if ptx_ms is not None:
        print(f"  projection.ptx:         {ptx_ms:.1f} ms")
    else:
        print(f"  (projection.ptx not found)")

    # ===================================================================
    # Summary table (uses already-collected results, no re-run)
    # ===================================================================
    print(f"\n{'=' * 56}")
    print(f"SUMMARY -- {device_name}")
    print(f"{'=' * 56}")
    print(f"Memory bandwidth (copy):     {bw_tbs:.2f} TB/s ({bw_pct:.0f}% of peak)")
    if norm_us is not None:
        print(f"RMSNorm 5120:")
        print(f"  Lithos NORM:               {norm_us:.2f} us")
    if lithos_oh is not None:
        print(f"Kernel launch overhead:")
        print(f"  cuLaunchKernel (Lithos):   {lithos_oh:.2f} us")
        print(f"  cuBLAS sgemv call:         {cublas_oh:.2f} us")
    if total_ms is not None:
        print(f"Cubin load time (7 kernels): {total_ms:.1f} ms")
    if ptx_ms is not None:
        print(f"PTX compile time:            {ptx_ms:.1f} ms")
    print()
    print("NOTE: Lithos PROJECTION uses W4A16 (4-bit weights), so it")
    print("moves ~8x fewer bytes than cuBLAS sgemv (f32). The GB/s")
    print("figures reflect actual bytes transferred, not f32-equivalent.")

    # Cleanup
    timer.destroy()
    cublas.cublasDestroy_v2(cublas_handle)
    cuda.cuCtxDestroy_v2(ctx)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
