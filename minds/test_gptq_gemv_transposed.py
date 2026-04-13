#!/usr/bin/env python3
"""
Benchmark GPTQ GEMV: original [K/8, N] layout vs transposed [N, K/8] layout.

Tests:
1. gptq_gemv_ultra   -- original kernel, original layout (baseline)
2. gptq_gemv_transposed -- new kernel, transposed layout

Verifies correctness against gptq_matvec_tc reference, then benchmarks.
"""

import ctypes
import os
import random
import struct
import subprocess
import numpy as np
from ctypes import (
    POINTER, byref, c_float, c_int, c_size_t,
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
    N, K, GROUP_SIZE = 12288, 5120, 128
    K_packed = K // 8
    n_groups = K // GROUP_SIZE

    print(f"GPTQ GEMV Layout Comparison Benchmark")
    print(f"N={N}, K={K}, group_size={GROUP_SIZE}")
    print(f"K_packed={K_packed}, n_groups={n_groups}")
    print()

    # Compile kernels
    for name in ['gptq_gemv_ultra', 'gptq_gemv_transposed', 'gptq_gemv_tiled', 'gptq_matvec_tc']:
        ptx = os.path.join(KERNEL_DIR, f'{name}.ptx')
        cubin = os.path.join(KERNEL_DIR, f'{name}.cubin')
        r = subprocess.run(['ptxas', '-arch=sm_90', '-O3', '-o', cubin, ptx],
                          capture_output=True, text=True)
        if r.returncode != 0:
            print(f"COMPILE FAILED: {name}\n{r.stderr}")
            return

    # CUDA setup
    cuda = ctypes.CDLL("libcuda.so.1")
    for fn, at, rt in [
        ('cuInit', [c_uint], CUresult),
        ('cuDeviceGet', [POINTER(CUdevice), c_int], CUresult),
        ('cuCtxCreate_v2', [POINTER(CUcontext), c_uint, CUdevice], CUresult),
        ('cuCtxDestroy_v2', [CUcontext], CUresult),
        ('cuCtxSynchronize', [], CUresult),
        ('cuModuleLoadData', [POINTER(CUmodule), c_void_p], CUresult),
        ('cuModuleGetFunction', [POINTER(CUfunction), CUmodule, ctypes.c_char_p], CUresult),
        ('cuModuleUnload', [CUmodule], CUresult),
        ('cuMemAlloc_v2', [POINTER(CUdeviceptr), c_size_t], CUresult),
        ('cuMemFree_v2', [CUdeviceptr], CUresult),
        ('cuMemcpyHtoD_v2', [CUdeviceptr, c_void_p, c_size_t], CUresult),
        ('cuMemcpyDtoH_v2', [c_void_p, CUdeviceptr, c_size_t], CUresult),
        ('cuMemsetD8_v2', [CUdeviceptr, ctypes.c_ubyte, c_size_t], CUresult),
        ('cuEventCreate', [POINTER(CUevent), c_uint], CUresult),
        ('cuEventRecord', [CUevent, CUstream], CUresult),
        ('cuEventSynchronize', [CUevent], CUresult),
        ('cuEventElapsedTime', [POINTER(c_float), CUevent, CUevent], CUresult),
        ('cuEventDestroy_v2', [CUevent], CUresult),
        ('cuFuncGetAttribute', [POINTER(c_int), c_int, CUfunction], CUresult),
    ]:
        f = getattr(cuda, fn); f.argtypes = at; f.restype = rt
    cuda.cuLaunchKernel.argtypes = [
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, c_void_p, c_void_p,
    ]; cuda.cuLaunchKernel.restype = CUresult

    _check("cuInit", cuda.cuInit(0))
    dev = CUdevice(); _check("cuDeviceGet", cuda.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext(); _check("cuCtxCreate", cuda.cuCtxCreate_v2(byref(ctx), 0, dev))

    # Generate test data -- original layout [K_packed, N]
    random.seed(42)
    qw_orig = np.array([random.randint(0, 0xFFFFFFFF) for _ in range(K_packed * N)],
                        dtype=np.uint32).reshape(K_packed, N)
    scales_orig = np.array([struct.unpack('<e', struct.pack('<e', random.uniform(0.001, 0.01)))[0]
                            for _ in range(n_groups * N)],
                           dtype=np.float16).reshape(n_groups, N)
    input_data = np.array([random.uniform(-1.0, 1.0) for _ in range(K)], dtype=np.float32)

    # Transposed layout [N, K_packed]
    qw_trans = np.ascontiguousarray(qw_orig.T)  # [N, K_packed]
    scales_trans = np.ascontiguousarray(scales_orig.T)  # [N, n_groups]

    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("alloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
        return d

    def upload(d, arr):
        buf = arr.tobytes()
        b = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
        _check("HtoD", cuda.cuMemcpyHtoD_v2(d, ctypes.cast(b, c_void_p), c_size_t(len(buf))))

    # Allocate device memory
    d_qw_orig = dev_alloc(qw_orig.nbytes)
    d_sc_orig = dev_alloc(scales_orig.nbytes)
    d_qw_trans = dev_alloc(qw_trans.nbytes)
    d_sc_trans = dev_alloc(scales_trans.nbytes)
    d_in = dev_alloc(input_data.nbytes)
    d_out_ref = dev_alloc(N * 4)
    d_out_test = dev_alloc(N * 4)

    upload(d_qw_orig, qw_orig)
    upload(d_sc_orig, scales_orig)
    upload(d_qw_trans, qw_trans)
    upload(d_sc_trans, scales_trans)
    upload(d_in, input_data)

    # Reference: gptq_matvec_tc (original layout)
    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod), open(os.path.join(KERNEL_DIR, "gptq_matvec_tc.cubin"), "rb").read()))
    ref_func = CUfunction()
    _check("getfunc", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"gptq_matvec_tc"))
    _check("memset", cuda.cuMemsetD8_v2(d_out_ref, 0, c_size_t(N * 4)))
    ref_args = [c_uint64(d_qw_orig.value), c_uint64(d_sc_orig.value), c_uint64(d_in.value),
                c_uint64(d_out_ref.value), c_uint32(N), c_uint32(K)]
    ref_ptrs = (c_void_p * len(ref_args))()
    for i, a in enumerate(ref_args): ref_ptrs[i] = ctypes.cast(byref(a), c_void_p).value
    _check("launch", cuda.cuLaunchKernel(ref_func, (N+127)//128, 1, 1, 256, 1, 1, K*4, None, ref_ptrs, None))
    _check("sync", cuda.cuCtxSynchronize())
    out_ref = bytearray(N * 4)
    _check("DtoH", cuda.cuMemcpyDtoH_v2(ctypes.cast((ctypes.c_ubyte * (N*4)).from_buffer(out_ref), c_void_p), d_out_ref, c_size_t(N * 4)))
    ref_vals = [struct.unpack_from('<f', out_ref, i * 4)[0] for i in range(N)]

    qw_bytes = K * N // 2
    sc_bytes = n_groups * N * 2
    in_bytes = K * 4
    out_bytes = N * 4
    total_bytes = qw_bytes + sc_bytes + in_bytes + out_bytes

    ev_start = CUevent(); ev_stop = CUevent()
    _check("ev", cuda.cuEventCreate(byref(ev_start), 0))
    _check("ev", cuda.cuEventCreate(byref(ev_stop), 0))

    WARMUP, ITERS = 200, 1000
    K_SPLITS_OPTIONS = [s for s in [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640]
                        if K_packed % s == 0]

    # Test configurations
    configs = [
        # (label, cubin, entry, block_x, d_qw, d_sc, layout_desc)
        ('gptq_gemv_ultra [K/8,N]', 'gptq_gemv_ultra', 'gptq_gemv_ultra', 64, d_qw_orig, d_sc_orig, 'row-major'),
        ('gptq_gemv_transposed [N,K/8]', 'gptq_gemv_transposed', 'gptq_gemv_transposed', 256, d_qw_trans, d_sc_trans, 'col-major'),
        ('gptq_gemv_tiled [K/8,N] 4x-unroll', 'gptq_gemv_tiled', 'gptq_gemv_tiled', 256, d_qw_orig, d_sc_orig, 'row-major+streaming'),
    ]

    results = {}

    for label, cubin_name, entry, block_x, d_qw, d_sc, layout in configs:
        mod = CUmodule()
        _check("load", cuda.cuModuleLoadData(byref(mod), open(os.path.join(KERNEL_DIR, f"{cubin_name}.cubin"), "rb").read()))
        func = CUfunction()
        _check("getfunc", cuda.cuModuleGetFunction(byref(func), mod, entry.encode()))
        nregs = c_int()
        cuda.cuFuncGetAttribute(byref(nregs), 4, func)

        print(f"\n{'='*70}")
        print(f"  {label} ({nregs.value} regs, {block_x} threads/block)")
        print(f"  Layout: {layout} -- weight stride along K = {'4 bytes' if layout == 'col-major' else f'{N*4} bytes'}")
        print(f"{'='*70}")
        print(f"  {'SPLITS':>6} {'k/split':>7} {'blocks':>6} {'time_us':>8} {'GB/s':>8} {'status':>6}")

        best_bw, best_splits, best_ms = 0, 1, 999
        for K_SPLITS in K_SPLITS_OPTIONS:
            k_pps = K_packed // K_SPLITS
            smem = k_pps * 32
            if smem > 48 * 1024: continue

            grid_x = (N + 255) // 256
            grid_y = K_SPLITS

            _check("memset", cuda.cuMemsetD8_v2(d_out_test, 0, c_size_t(N * 4)))
            args = [c_uint64(d_qw.value), c_uint64(d_sc.value), c_uint64(d_in.value),
                    c_uint64(d_out_test.value), c_uint32(N), c_uint32(K),
                    c_uint32(K_SPLITS), c_uint32(k_pps)]
            ptrs = (c_void_p * len(args))()
            for i, a in enumerate(args): ptrs[i] = ctypes.cast(byref(a), c_void_p).value

            _check("launch", cuda.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, smem, None, ptrs, None))
            _check("sync", cuda.cuCtxSynchronize())

            out_buf = bytearray(N * 4)
            _check("DtoH", cuda.cuMemcpyDtoH_v2(ctypes.cast((ctypes.c_ubyte * (N*4)).from_buffer(out_buf), c_void_p), d_out_test, c_size_t(N * 4)))
            test_vals = [struct.unpack_from('<f', out_buf, i * 4)[0] for i in range(N)]
            n_wrong = sum(1 for i in range(N) if abs(ref_vals[i] - test_vals[i]) / max(abs(ref_vals[i]), 1e-8) > 0.01)
            status = "PASS" if n_wrong == 0 else f"FAIL({n_wrong})"

            if n_wrong > 0:
                print(f"  {K_SPLITS:6d} {k_pps:7d} {grid_x*grid_y:6d} {'---':>8} {'---':>8} {status:>6}")
                continue

            # Reset output for benchmark
            _check("memset", cuda.cuMemsetD8_v2(d_out_test, 0, c_size_t(N * 4)))
            for _ in range(WARMUP):
                cuda.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, smem, None, ptrs, None)
            _check("sync", cuda.cuCtxSynchronize())
            _check("memset", cuda.cuMemsetD8_v2(d_out_test, 0, c_size_t(N * 4)))
            _check("ev", cuda.cuEventRecord(ev_start, None))
            for _ in range(ITERS):
                cuda.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, smem, None, ptrs, None)
            _check("ev", cuda.cuEventRecord(ev_stop, None))
            _check("ev", cuda.cuEventSynchronize(ev_stop))
            ms = c_float()
            _check("ev", cuda.cuEventElapsedTime(byref(ms), ev_start, ev_stop))
            ms_per = ms.value / ITERS
            bw = total_bytes / (ms_per / 1000.0) / 1e9
            print(f"  {K_SPLITS:6d} {k_pps:7d} {grid_x*grid_y:6d} {ms_per*1000:8.1f} {bw:8.1f} {status:>6}")
            if bw > best_bw:
                best_bw, best_splits, best_ms = bw, K_SPLITS, ms_per

        print(f"\n  BEST: K_SPLITS={best_splits}, {best_ms*1000:.1f} us, {best_bw:.1f} GB/s")
        results[label] = (best_bw, best_splits, best_ms)
        cuda.cuModuleUnload(mod)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Data: {total_bytes/1e6:.2f} MB per call")
    print(f"  Theoretical min @ 4000 GB/s: {total_bytes / 4e12 * 1e6:.1f} us")
    print()
    for label, (bw, splits, ms) in results.items():
        print(f"  {label}:")
        print(f"    {bw:.1f} GB/s, {ms*1000:.1f} us, K_SPLITS={splits}")
    if len(results) >= 2:
        bws = list(results.values())
        baseline = bws[0][0]
        for label, (bw, _, _) in list(results.items())[1:]:
            print(f"  vs baseline: {label}: {bw/baseline:.2f}x")
    print(f"{'='*70}")

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start); cuda.cuEventDestroy_v2(ev_stop)
    for d in [d_qw_orig, d_sc_orig, d_qw_trans, d_sc_trans, d_in, d_out_ref, d_out_test]:
        cuda.cuMemFree_v2(d)
    cuda.cuModuleUnload(ref_mod); cuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    main()
