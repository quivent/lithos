#!/usr/bin/env python3
"""
Test + benchmark for gptq_gemv_ultra.ptx -- the optimized GPTQ W4A16 GEMV kernel.
Compares against gptq_gemv_fast.ptx baseline and gptq_matvec_tc.ptx reference.

Key optimizations in ultra:
- ld.global.v4.u32: 4 consecutive columns loaded per thread in one instruction
- 64 threads/block: high occupancy with small blocks, optimal for large K_SPLITS
- 4 independent accumulators per thread: hides FMA latency
- Vectorized scale loads: ld.global.v2.b32 for 4 f16 scales at once
"""

import ctypes
import os
import random
import struct
import subprocess
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

    print(f"GPTQ GEMV Ultra -- Production Kernel Benchmark")
    print(f"N={N}, K={K}, group_size={GROUP_SIZE}")
    print()

    # Compile
    for name in ['gptq_gemv_fast', 'gptq_gemv_ultra']:
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

    # Generate test data
    random.seed(42)
    qweight_data = bytearray(K_packed * N * 4)
    for i in range(K_packed * N):
        struct.pack_into('<I', qweight_data, i * 4, random.randint(0, 0xFFFFFFFF))
    scales_data = bytearray(n_groups * N * 2)
    for i in range(n_groups * N):
        struct.pack_into('<e', scales_data, i * 2, random.uniform(0.001, 0.01))
    input_data = bytearray(K * 4)
    for i in range(K):
        struct.pack_into('<f', input_data, i * 4, random.uniform(-1.0, 1.0))

    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("alloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
        return d

    d_qw = dev_alloc(len(qweight_data))
    d_sc = dev_alloc(len(scales_data))
    d_in = dev_alloc(len(input_data))
    d_out_ref = dev_alloc(N * 4)
    d_out_test = dev_alloc(N * 4)

    for d, buf in [(d_qw, qweight_data), (d_sc, scales_data), (d_in, input_data)]:
        b = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
        _check("HtoD", cuda.cuMemcpyHtoD_v2(d, ctypes.cast(b, c_void_p), c_size_t(len(buf))))

    # Reference
    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod), open(os.path.join(KERNEL_DIR, "gptq_matvec_tc.cubin"), "rb").read()))
    ref_func = CUfunction()
    _check("getfunc", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"gptq_matvec_tc"))
    _check("memset", cuda.cuMemsetD8_v2(d_out_ref, 0, c_size_t(N * 4)))
    ref_args = [c_uint64(d_qw.value), c_uint64(d_sc.value), c_uint64(d_in.value),
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

    # Test kernels
    for kern_name, entry, block_x in [
        ('gptq_gemv_fast', 'gptq_gemv_fast', 256),
        ('gptq_gemv_ultra', 'gptq_gemv_ultra', 64),
    ]:
        mod = CUmodule()
        _check("load", cuda.cuModuleLoadData(byref(mod), open(os.path.join(KERNEL_DIR, f"{kern_name}.cubin"), "rb").read()))
        func = CUfunction()
        _check("getfunc", cuda.cuModuleGetFunction(byref(func), mod, entry.encode()))
        nregs = c_int()
        cuda.cuFuncGetAttribute(byref(nregs), 4, func)

        print(f"\n{'='*65}")
        print(f"  {kern_name} ({nregs.value} regs, {block_x} threads/block)")
        print(f"{'='*65}")
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

            for _ in range(WARMUP):
                cuda.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, smem, None, ptrs, None)
            _check("sync", cuda.cuCtxSynchronize())
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
            if bw > best_bw and n_wrong == 0:
                best_bw, best_splits, best_ms = bw, K_SPLITS, ms_per

        print(f"\n  BEST: K_SPLITS={best_splits}, {best_ms*1000:.1f} us, {best_bw:.1f} GB/s")
        cuda.cuModuleUnload(mod)

    print(f"\n{'='*65}")
    print(f"  Data: {total_bytes/1e6:.2f} MB per call")
    print(f"  Theoretical min @ 4000 GB/s: {total_bytes / 4e12 * 1e6:.1f} us")
    print(f"{'='*65}")

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start); cuda.cuEventDestroy_v2(ev_stop)
    for d in [d_qw, d_sc, d_in, d_out_ref, d_out_test]: cuda.cuMemFree_v2(d)
    cuda.cuModuleUnload(ref_mod); cuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    main()
