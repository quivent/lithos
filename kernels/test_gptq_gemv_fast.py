#!/usr/bin/env python3
"""
Test + benchmark for gptq_gemv_fast.ptx (bandwidth-optimized GEMV)
against known-correct gptq_matvec_tc.ptx (scalar FMA GEMV).
"""

import ctypes
import os
import random
import struct
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

    print(f"GPTQ GEMV Fast (Bandwidth-Optimized) Test & Benchmark")
    print(f"N={N}, K={K}, group_size={GROUP_SIZE}")
    print()

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
    cuda.cuFuncGetAttribute = cuda.cuFuncGetAttribute
    cuda.cuFuncGetAttribute.argtypes = [POINTER(c_int), c_int, CUfunction]
    cuda.cuFuncGetAttribute.restype = CUresult

    _check("cuInit", cuda.cuInit(0))
    dev = CUdevice()
    _check("cuDeviceGet", cuda.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext()
    _check("cuCtxCreate", cuda.cuCtxCreate_v2(byref(ctx), 0, dev))

    # Generate test data
    random.seed(42)
    K_packed = K // 8
    n_groups = K // GROUP_SIZE

    qweight_data = bytearray(K_packed * N * 4)
    for i in range(K_packed * N):
        val = random.randint(0, 0xFFFFFFFF)
        struct.pack_into('<I', qweight_data, i * 4, val)

    scales_data = bytearray(n_groups * N * 2)
    for i in range(n_groups * N):
        s = random.uniform(0.001, 0.01)
        struct.pack_into('<e', scales_data, i * 2, s)

    input_data = bytearray(K * 4)
    for i in range(K):
        v = random.uniform(-1.0, 1.0)
        struct.pack_into('<f', input_data, i * 4, v)

    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("cuMemAlloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
        return d

    d_qweight = dev_alloc(len(qweight_data))
    d_scales = dev_alloc(len(scales_data))
    d_input = dev_alloc(len(input_data))
    d_output_ref = dev_alloc(N * 4)
    d_output_fast = dev_alloc(N * 4)

    qw_buf = (ctypes.c_ubyte * len(qweight_data)).from_buffer(qweight_data)
    sc_buf = (ctypes.c_ubyte * len(scales_data)).from_buffer(scales_data)
    in_buf = (ctypes.c_ubyte * len(input_data)).from_buffer(input_data)
    _check("HtoD qw", cuda.cuMemcpyHtoD_v2(d_qweight, ctypes.cast(qw_buf, c_void_p), c_size_t(len(qweight_data))))
    _check("HtoD sc", cuda.cuMemcpyHtoD_v2(d_scales, ctypes.cast(sc_buf, c_void_p), c_size_t(len(scales_data))))
    _check("HtoD in", cuda.cuMemcpyHtoD_v2(d_input, ctypes.cast(in_buf, c_void_p), c_size_t(len(input_data))))

    # Load reference kernel
    ref_cubin_path = os.path.join(KERNEL_DIR, "gptq_matvec_tc.cubin")
    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod), open(ref_cubin_path, "rb").read()))
    ref_func = CUfunction()
    _check("getfunc ref", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"gptq_matvec_tc"))

    # Load fast kernel
    fast_cubin_path = os.path.join(KERNEL_DIR, "gptq_gemv_fast.cubin")
    fast_mod = CUmodule()
    _check("load fast", cuda.cuModuleLoadData(byref(fast_mod), open(fast_cubin_path, "rb").read()))
    fast_func = CUfunction()
    _check("getfunc fast", cuda.cuModuleGetFunction(byref(fast_func), fast_mod, b"gptq_gemv_fast"))

    num_regs = c_int()
    cuda.cuFuncGetAttribute(byref(num_regs), 4, fast_func)
    print(f"Fast kernel register usage: {num_regs.value} regs/thread")

    # Launch reference
    ref_grid_x = (N + 127) // 128
    ref_block_x = 256
    ref_smem = K * 4

    _check("memset ref", cuda.cuMemsetD8_v2(d_output_ref, 0, c_size_t(N * 4)))
    ref_args = [c_uint64(d_qweight.value), c_uint64(d_scales.value), c_uint64(d_input.value),
                c_uint64(d_output_ref.value), c_uint32(N), c_uint32(K)]
    ref_ptrs = (c_void_p * len(ref_args))()
    for i, a in enumerate(ref_args):
        ref_ptrs[i] = ctypes.cast(byref(a), c_void_p).value
    _check("launch ref", cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1, ref_smem, None, ref_ptrs, None))
    _check("sync", cuda.cuCtxSynchronize())

    out_ref = bytearray(N * 4)
    out_ref_buf = (ctypes.c_ubyte * (N * 4)).from_buffer(out_ref)
    _check("DtoH ref", cuda.cuMemcpyDtoH_v2(ctypes.cast(out_ref_buf, c_void_p), d_output_ref, c_size_t(N * 4)))
    ref_vals = [struct.unpack_from('<f', out_ref, i * 4)[0] for i in range(N)]

    # Data moved
    qw_bytes = K * N // 2
    sc_bytes = n_groups * N * 2
    in_bytes = K * 4
    out_bytes = N * 4
    total_bytes = qw_bytes + sc_bytes + in_bytes + out_bytes

    ev_start = CUevent()
    ev_stop = CUevent()
    _check("evCreate", cuda.cuEventCreate(byref(ev_start), 0))
    _check("evCreate", cuda.cuEventCreate(byref(ev_stop), 0))

    WARMUP = 100
    ITERS = 500

    # Only test K_SPLITS that evenly divide K_packed
    K_SPLITS_OPTIONS = [s for s in [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640] if K_packed % s == 0]

    best_bw = 0
    best_splits = 1
    best_ms = 999

    print(f"\nSweeping K_SPLITS (only divisors of K_packed={K_packed}):")
    print(f"  {'SPLITS':>6} {'k/split':>7} {'blocks':>6} {'smem_KB':>7} {'time_us':>8} {'GB/s':>8} {'status':>6}")

    for K_SPLITS in K_SPLITS_OPTIONS:
        fast_grid_x = (N + 255) // 256
        fast_grid_y = K_SPLITS
        fast_block_x = 256
        k_packed_per_split = K_packed // K_SPLITS
        fast_smem = k_packed_per_split * 8 * 4  # input slice in shared memory

        # Max smem per block on GH200 is 228 KB (dynamic) with appropriate opt-in
        # Default max is 48 KB. Skip if too large.
        if fast_smem > 48 * 1024:
            continue

        # Correctness
        _check("memset fast", cuda.cuMemsetD8_v2(d_output_fast, 0, c_size_t(N * 4)))
        fast_args = [c_uint64(d_qweight.value), c_uint64(d_scales.value), c_uint64(d_input.value),
                     c_uint64(d_output_fast.value), c_uint32(N), c_uint32(K), c_uint32(K_SPLITS)]
        fast_ptrs = (c_void_p * len(fast_args))()
        for i, a in enumerate(fast_args):
            fast_ptrs[i] = ctypes.cast(byref(a), c_void_p).value

        _check("launch fast", cuda.cuLaunchKernel(fast_func, fast_grid_x, fast_grid_y, 1, fast_block_x, 1, 1,
                                                   fast_smem, None, fast_ptrs, None))
        _check("sync", cuda.cuCtxSynchronize())

        out_fast = bytearray(N * 4)
        out_fast_buf = (ctypes.c_ubyte * (N * 4)).from_buffer(out_fast)
        _check("DtoH fast", cuda.cuMemcpyDtoH_v2(ctypes.cast(out_fast_buf, c_void_p), d_output_fast, c_size_t(N * 4)))
        fast_vals = [struct.unpack_from('<f', out_fast, i * 4)[0] for i in range(N)]

        max_abs_err = 0.0
        n_wrong = 0
        for i in range(N):
            err = abs(ref_vals[i] - fast_vals[i])
            max_abs_err = max(max_abs_err, err)
            denom = max(abs(ref_vals[i]), 1e-8)
            if err / denom > 0.01:
                n_wrong += 1
        status = "PASS" if n_wrong == 0 else f"FAIL({n_wrong})"

        # Benchmark (kernel only, no memset in timed region)
        # Pre-zero and let atomicAdd accumulate. For benchmark we don't care about
        # correctness of repeated runs.
        for _ in range(WARMUP):
            cuda.cuLaunchKernel(fast_func, fast_grid_x, fast_grid_y, 1, fast_block_x, 1, 1,
                                fast_smem, None, fast_ptrs, None)
        _check("sync", cuda.cuCtxSynchronize())

        _check("evRec", cuda.cuEventRecord(ev_start, None))
        for _ in range(ITERS):
            cuda.cuLaunchKernel(fast_func, fast_grid_x, fast_grid_y, 1, fast_block_x, 1, 1,
                                fast_smem, None, fast_ptrs, None)
        _check("evRec", cuda.cuEventRecord(ev_stop, None))
        _check("evSync", cuda.cuEventSynchronize(ev_stop))
        ms_fast = c_float()
        _check("elapsed", cuda.cuEventElapsedTime(byref(ms_fast), ev_start, ev_stop))
        ms_per_fast = ms_fast.value / ITERS

        bw_fast = total_bytes / (ms_per_fast / 1000.0) / 1e9
        total_blocks = fast_grid_x * fast_grid_y
        print(f"  {K_SPLITS:6d} {k_packed_per_split:7d} {total_blocks:6d} {fast_smem/1024:7.1f} {ms_per_fast*1000:8.1f} {bw_fast:8.1f} {status:>6}")

        if bw_fast > best_bw:
            best_bw = bw_fast
            best_splits = K_SPLITS
            best_ms = ms_per_fast

    # Benchmark reference
    for _ in range(WARMUP):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1, ref_smem, None, ref_ptrs, None)
    _check("sync", cuda.cuCtxSynchronize())
    _check("evRec", cuda.cuEventRecord(ev_start, None))
    for _ in range(ITERS):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, ref_block_x, 1, 1, ref_smem, None, ref_ptrs, None)
    _check("evRec", cuda.cuEventRecord(ev_stop, None))
    _check("evSync", cuda.cuEventSynchronize(ev_stop))
    ms_ref = c_float()
    _check("elapsed", cuda.cuEventElapsedTime(byref(ms_ref), ev_start, ev_stop))
    ms_per_ref = ms_ref.value / ITERS
    bw_ref = total_bytes / (ms_per_ref / 1000.0) / 1e9

    print(f"\n{'='*60}")
    print(f"  Reference (gptq_matvec_tc):  {ms_per_ref*1000:7.1f} us  {bw_ref:7.1f} GB/s")
    print(f"  Best fast (K_SPLITS={best_splits}):  {best_ms*1000:7.1f} us  {best_bw:7.1f} GB/s")
    print(f"  Speedup: {bw_ref and best_bw/bw_ref:.2f}x")
    print(f"  Data moved per call: {total_bytes/1e6:.2f} MB")
    print(f"  Theoretical min @ 4000 GB/s: {total_bytes / 4e12 * 1e6:.1f} us")

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start)
    cuda.cuEventDestroy_v2(ev_stop)
    for d in [d_qweight, d_scales, d_input, d_output_ref, d_output_fast]:
        cuda.cuMemFree_v2(d)
    cuda.cuModuleUnload(ref_mod)
    cuda.cuModuleUnload(fast_mod)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    main()
