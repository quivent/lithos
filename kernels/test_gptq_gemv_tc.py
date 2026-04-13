#!/usr/bin/env python3
"""
Test + benchmark for gptq_gemv_tc.ptx (tensor core GEMV)
against known-correct gptq_matvec_tc.ptx (scalar FMA GEMV).

Both compute: output[n] = sum_k( dequant(qweight[k/8,n], k%8) * input[k] )
  dequant: ((nibble - 8) * scale)

Uses CUDA driver API directly.
"""

import ctypes
import math
import os
import random
import struct
import sys
import time
from ctypes import (
    POINTER, byref, c_char_p, c_float, c_int, c_size_t,
    c_uint, c_uint32, c_uint64, c_void_p,
)

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

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
        raise RuntimeError(f"{name} failed with error code {result}")


def main():
    N = 12288
    K = 5120
    GROUP_SIZE = 128

    print(f"GPTQ GEMV Tensor Core Test & Benchmark")
    print(f"N={N}, K={K}, group_size={GROUP_SIZE}")
    print()

    # ---- Load CUDA driver ----
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [c_uint]; cuda.cuInit.restype = CUresult
    cuda.cuDeviceGet.argtypes = [POINTER(CUdevice), c_int]; cuda.cuDeviceGet.restype = CUresult
    cuda.cuCtxCreate_v2.argtypes = [POINTER(CUcontext), c_uint, CUdevice]; cuda.cuCtxCreate_v2.restype = CUresult
    cuda.cuCtxDestroy_v2.argtypes = [CUcontext]; cuda.cuCtxDestroy_v2.restype = CUresult
    cuda.cuCtxSynchronize.argtypes = []; cuda.cuCtxSynchronize.restype = CUresult
    cuda.cuModuleLoadData.argtypes = [POINTER(CUmodule), c_void_p]; cuda.cuModuleLoadData.restype = CUresult
    cuda.cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, c_char_p]; cuda.cuModuleGetFunction.restype = CUresult
    cuda.cuModuleUnload.argtypes = [CUmodule]; cuda.cuModuleUnload.restype = CUresult
    cuda.cuLaunchKernel.argtypes = [
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, c_void_p, c_void_p,
    ]; cuda.cuLaunchKernel.restype = CUresult
    cuda.cuMemAlloc_v2.argtypes = [POINTER(CUdeviceptr), c_size_t]; cuda.cuMemAlloc_v2.restype = CUresult
    cuda.cuMemFree_v2.argtypes = [CUdeviceptr]; cuda.cuMemFree_v2.restype = CUresult
    cuda.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, c_void_p, c_size_t]; cuda.cuMemcpyHtoD_v2.restype = CUresult
    cuda.cuMemcpyDtoH_v2.argtypes = [c_void_p, CUdeviceptr, c_size_t]; cuda.cuMemcpyDtoH_v2.restype = CUresult
    cuda.cuMemsetD8_v2.argtypes = [CUdeviceptr, ctypes.c_ubyte, c_size_t]; cuda.cuMemsetD8_v2.restype = CUresult
    cuda.cuEventCreate.argtypes = [POINTER(CUevent), c_uint]; cuda.cuEventCreate.restype = CUresult
    cuda.cuEventRecord.argtypes = [CUevent, CUstream]; cuda.cuEventRecord.restype = CUresult
    cuda.cuEventSynchronize.argtypes = [CUevent]; cuda.cuEventSynchronize.restype = CUresult
    cuda.cuEventElapsedTime.argtypes = [POINTER(c_float), CUevent, CUevent]; cuda.cuEventElapsedTime.restype = CUresult
    cuda.cuEventDestroy_v2.argtypes = [CUevent]; cuda.cuEventDestroy_v2.restype = CUresult

    _check("cuInit", cuda.cuInit(0))
    dev = CUdevice()
    _check("cuDeviceGet", cuda.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext()
    _check("cuCtxCreate", cuda.cuCtxCreate_v2(byref(ctx), 0, dev))

    # ---- Generate test data ----
    random.seed(42)
    K_packed = K // 8
    n_groups = K // GROUP_SIZE

    # qweight: [K/8, N] uint32 row-major
    qweight_data = bytearray(K_packed * N * 4)
    for i in range(K_packed * N):
        val = random.randint(0, 0xFFFFFFFF)
        struct.pack_into('<I', qweight_data, i * 4, val)

    # scales: [n_groups, N] float16 row-major
    scales_data = bytearray(n_groups * N * 2)
    for i in range(n_groups * N):
        # Use small random scales
        s = random.uniform(0.001, 0.01)
        struct.pack_into('<e', scales_data, i * 2, s)

    # input: [K] float32
    input_data = bytearray(K * 4)
    for i in range(K):
        v = random.uniform(-1.0, 1.0)
        struct.pack_into('<f', input_data, i * 4, v)

    # ---- Allocate device memory ----
    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("cuMemAlloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
        return d

    d_qweight = dev_alloc(len(qweight_data))
    d_scales = dev_alloc(len(scales_data))
    d_input = dev_alloc(len(input_data))
    d_output_ref = dev_alloc(N * 4)
    d_output_tc = dev_alloc(N * 4)

    qw_buf = (ctypes.c_ubyte * len(qweight_data)).from_buffer(qweight_data)
    sc_buf = (ctypes.c_ubyte * len(scales_data)).from_buffer(scales_data)
    in_buf = (ctypes.c_ubyte * len(input_data)).from_buffer(input_data)
    _check("HtoD qw", cuda.cuMemcpyHtoD_v2(d_qweight, ctypes.cast(qw_buf, c_void_p), c_size_t(len(qweight_data))))
    _check("HtoD sc", cuda.cuMemcpyHtoD_v2(d_scales, ctypes.cast(sc_buf, c_void_p), c_size_t(len(scales_data))))
    _check("HtoD in", cuda.cuMemcpyHtoD_v2(d_input, ctypes.cast(in_buf, c_void_p), c_size_t(len(input_data))))

    # ---- Load reference kernel (gptq_matvec_tc) ----
    ref_cubin_path = os.path.join(KERNEL_DIR, "gptq_matvec_tc.cubin")
    ref_data = open(ref_cubin_path, "rb").read()
    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod), ref_data))
    ref_func = CUfunction()
    _check("getfunc ref", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"gptq_matvec_tc"))

    # ---- Load new kernel (gptq_gemv_tc) ----
    tc_cubin_path = os.path.join(KERNEL_DIR, "gptq_gemv_tc.cubin")
    tc_data = open(tc_cubin_path, "rb").read()
    tc_mod = CUmodule()
    _check("load tc", cuda.cuModuleLoadData(byref(tc_mod), tc_data))
    tc_func = CUfunction()
    _check("getfunc tc", cuda.cuModuleGetFunction(byref(tc_func), tc_mod, b"gptq_gemv_tc"))

    # ---- Launch reference kernel ----
    # gptq_matvec_tc: grid=(ceil(N/128),1,1), block=(256,1,1), smem=K*4
    ref_grid_x = (N + 127) // 128
    ref_block_x = 256
    ref_smem = K * 4

    _check("memset ref", cuda.cuMemsetD8_v2(d_output_ref, 0, c_size_t(N * 4)))
    ref_args = [
        c_uint64(d_qweight.value),
        c_uint64(d_scales.value),
        c_uint64(d_input.value),
        c_uint64(d_output_ref.value),
        c_uint32(N),
        c_uint32(K),
    ]
    ref_ptrs = (c_void_p * len(ref_args))()
    for i, a in enumerate(ref_args):
        ref_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

    _check("launch ref", cuda.cuLaunchKernel(
        ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1,
        ref_smem, None, ref_ptrs, None))
    _check("sync", cuda.cuCtxSynchronize())

    # ---- Launch new TC kernel ----
    # gptq_gemv_tc: grid=(ceil(N/128),1,1), block=(128,1,1), smem=K*4+512+4096
    tc_grid_x = (N + 127) // 128
    tc_block_x = 128
    tc_smem = K * 4 + 512 + 4096

    _check("memset tc", cuda.cuMemsetD8_v2(d_output_tc, 0, c_size_t(N * 4)))
    tc_args = [
        c_uint64(d_qweight.value),
        c_uint64(d_scales.value),
        c_uint64(d_input.value),
        c_uint64(d_output_tc.value),
        c_uint32(N),
        c_uint32(K),
    ]
    tc_ptrs = (c_void_p * len(tc_args))()
    for i, a in enumerate(tc_args):
        tc_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

    _check("launch tc", cuda.cuLaunchKernel(
        tc_func, tc_grid_x, 1, 1, tc_block_x, 1, 1,
        tc_smem, None, tc_ptrs, None))
    _check("sync", cuda.cuCtxSynchronize())

    # ---- Read back results ----
    out_ref = bytearray(N * 4)
    out_tc = bytearray(N * 4)
    out_ref_buf = (ctypes.c_ubyte * (N * 4)).from_buffer(out_ref)
    out_tc_buf = (ctypes.c_ubyte * (N * 4)).from_buffer(out_tc)
    _check("DtoH ref", cuda.cuMemcpyDtoH_v2(ctypes.cast(out_ref_buf, c_void_p), d_output_ref, c_size_t(N * 4)))
    _check("DtoH tc", cuda.cuMemcpyDtoH_v2(ctypes.cast(out_tc_buf, c_void_p), d_output_tc, c_size_t(N * 4)))

    ref_vals = [struct.unpack_from('<f', out_ref, i * 4)[0] for i in range(N)]
    tc_vals = [struct.unpack_from('<f', out_tc, i * 4)[0] for i in range(N)]

    # ---- Compare ----
    max_abs_err = 0.0
    max_rel_err = 0.0
    n_wrong = 0
    for i in range(N):
        err = abs(ref_vals[i] - tc_vals[i])
        max_abs_err = max(max_abs_err, err)
        denom = max(abs(ref_vals[i]), 1e-8)
        rel = err / denom
        max_rel_err = max(max_rel_err, rel)
        if rel > 0.05:  # 5% tolerance for f16 rounding
            n_wrong += 1

    print(f"Correctness check:")
    print(f"  Max absolute error: {max_abs_err:.6e}")
    print(f"  Max relative error: {max_rel_err:.6e}")
    print(f"  Elements > 5% relative error: {n_wrong}/{N}")

    # Print a few sample values
    print(f"\n  Sample values (first 8):")
    print(f"  {'idx':>5} {'ref':>12} {'tc':>12} {'diff':>12}")
    for i in range(min(8, N)):
        print(f"  {i:5d} {ref_vals[i]:12.6f} {tc_vals[i]:12.6f} {ref_vals[i]-tc_vals[i]:12.6e}")

    if n_wrong == 0:
        print(f"\n  PASS: All outputs match within tolerance.")
    else:
        print(f"\n  FAIL: {n_wrong} outputs differ significantly.")
        # Show some failing indices
        shown = 0
        for i in range(N):
            err = abs(ref_vals[i] - tc_vals[i])
            denom = max(abs(ref_vals[i]), 1e-8)
            if err / denom > 0.05:
                print(f"    [{i}] ref={ref_vals[i]:.6f} tc={tc_vals[i]:.6f} err={err:.6e}")
                shown += 1
                if shown >= 10:
                    break

    # ---- Benchmark ----
    print(f"\nBenchmark:")

    ev_start = CUevent()
    ev_stop = CUevent()
    _check("evCreate", cuda.cuEventCreate(byref(ev_start), 0))
    _check("evCreate", cuda.cuEventCreate(byref(ev_stop), 0))

    WARMUP = 50
    ITERS = 200

    # Benchmark reference kernel
    for _ in range(WARMUP):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1,
                            ref_smem, None, ref_ptrs, None)
    _check("sync", cuda.cuCtxSynchronize())

    _check("evRec", cuda.cuEventRecord(ev_start, None))
    for _ in range(ITERS):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1,
                            ref_smem, None, ref_ptrs, None)
    _check("evRec", cuda.cuEventRecord(ev_stop, None))
    _check("evSync", cuda.cuEventSynchronize(ev_stop))
    ms_ref = c_float()
    _check("elapsed", cuda.cuEventElapsedTime(byref(ms_ref), ev_start, ev_stop))
    ms_per_ref = ms_ref.value / ITERS

    # Benchmark TC kernel
    for _ in range(WARMUP):
        cuda.cuLaunchKernel(tc_func, tc_grid_x, 1, 1, tc_block_x, 1, 1,
                            tc_smem, None, tc_ptrs, None)
    _check("sync", cuda.cuCtxSynchronize())

    _check("evRec", cuda.cuEventRecord(ev_start, None))
    for _ in range(ITERS):
        cuda.cuLaunchKernel(tc_func, tc_grid_x, 1, 1, tc_block_x, 1, 1,
                            tc_smem, None, tc_ptrs, None)
    _check("evRec", cuda.cuEventRecord(ev_stop, None))
    _check("evSync", cuda.cuEventSynchronize(ev_stop))
    ms_tc = c_float()
    _check("elapsed", cuda.cuEventElapsedTime(byref(ms_tc), ev_start, ev_stop))
    ms_per_tc = ms_tc.value / ITERS

    # Data moved: qweight (K/8 * N * 4 = K*N/2) + scales (K/128 * N * 2) + input (K*4) + output (N*4)
    qw_bytes = K * N // 2
    sc_bytes = n_groups * N * 2
    in_bytes = K * 4
    out_bytes = N * 4
    total_bytes = qw_bytes + sc_bytes + in_bytes + out_bytes

    bw_ref = total_bytes / (ms_per_ref / 1000.0) / 1e9
    bw_tc = total_bytes / (ms_per_tc / 1000.0) / 1e9

    print(f"  Reference (gptq_matvec_tc, scalar FMA):")
    print(f"    Time: {ms_per_ref*1000:.1f} us/call")
    print(f"    Bandwidth: {bw_ref:.1f} GB/s")
    print(f"  New (gptq_gemv_tc, tensor core):")
    print(f"    Time: {ms_per_tc*1000:.1f} us/call")
    print(f"    Bandwidth: {bw_tc:.1f} GB/s")
    print(f"  Speedup: {ms_per_ref/ms_per_tc:.2f}x")
    print(f"  Data moved per call: {total_bytes/1e6:.2f} MB")

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start)
    cuda.cuEventDestroy_v2(ev_stop)
    cuda.cuMemFree_v2(d_qweight)
    cuda.cuMemFree_v2(d_scales)
    cuda.cuMemFree_v2(d_input)
    cuda.cuMemFree_v2(d_output_ref)
    cuda.cuMemFree_v2(d_output_tc)
    cuda.cuModuleUnload(ref_mod)
    cuda.cuModuleUnload(tc_mod)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    main()
