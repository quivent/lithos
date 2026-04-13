#!/usr/bin/env python3
"""
Benchmark for gptq_gemv_v2.ptx and gptq_gemv_v3.ptx against gptq_gemv_fast.ptx.
Tests correctness and sweeps K_SPLITS for optimal bandwidth.
"""

import ctypes
import os
import random
import struct
import subprocess
import sys
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


def compile_ptx(ptx_path):
    """Compile .ptx to .cubin using ptxas."""
    cubin_path = ptx_path.replace('.ptx', '.cubin')
    cmd = ['ptxas', '-arch=sm_90', '-O3', '-o', cubin_path, ptx_path]
    print(f"  Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE FAILED for {ptx_path}:")
        print(f"  stderr: {result.stderr}")
        return None
    # Get register usage
    if 'registers' in result.stderr.lower():
        for line in result.stderr.split('\n'):
            if 'registers' in line.lower():
                print(f"    {line.strip()}")
    return cubin_path


def main():
    N = 12288
    K = 5120
    GROUP_SIZE = 128

    print(f"GPTQ GEMV Kernel Shootout")
    print(f"N={N}, K={K}, group_size={GROUP_SIZE}")
    print(f"GH200 HBM3e peak: 4000 GB/s")
    print()

    # Compile all kernels
    print("Compiling kernels...")
    kernels_to_test = {
        'gptq_gemv_fast': {
            'ptx': os.path.join(KERNEL_DIR, 'gptq_gemv_fast.ptx'),
            'entry': 'gptq_gemv_fast',
            'block_x': 256,
            'cols_per_block': 256,
            'smem_bytes_per_k': 32,
        },
        'gptq_gemv_v3': {
            'ptx': os.path.join(KERNEL_DIR, 'gptq_gemv_v3.ptx'),
            'entry': 'gptq_gemv_v3',
            'block_x': 128,
            'cols_per_block': 256,
            'smem_bytes_per_k': 32,
        },
        'gptq_gemv_v7': {
            'ptx': os.path.join(KERNEL_DIR, 'gptq_gemv_v7.ptx'),
            'entry': 'gptq_gemv_v7',
            'block_x': 128,
            'cols_per_block': 256,
            'smem_bytes_per_k': 32,
        },
        'gptq_gemv_v8': {
            'ptx': os.path.join(KERNEL_DIR, 'gptq_gemv_v8.ptx'),
            'entry': 'gptq_gemv_v8',
            'block_x': 64,
            'cols_per_block': 256,
            'smem_bytes_per_k': 32,
        },
        'gptq_gemv_v9': {
            'ptx': os.path.join(KERNEL_DIR, 'gptq_gemv_v9.ptx'),
            'entry': 'gptq_gemv_v9',
            'block_x': 256,
            'cols_per_block': 256,
            'smem_bytes_per_k': 32,
        },
    }

    cubins = {}
    for name, info in kernels_to_test.items():
        cubin = compile_ptx(info['ptx'])
        if cubin:
            cubins[name] = cubin
        else:
            print(f"  Skipping {name} (compilation failed)")
    print()

    if not cubins:
        print("No kernels compiled successfully!")
        return

    # Init CUDA
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
        struct.pack_into('<I', qweight_data, i * 4, random.randint(0, 0xFFFFFFFF))

    scales_data = bytearray(n_groups * N * 2)
    for i in range(n_groups * N):
        struct.pack_into('<e', scales_data, i * 2, random.uniform(0.001, 0.01))

    input_data = bytearray(K * 4)
    for i in range(K):
        struct.pack_into('<f', input_data, i * 4, random.uniform(-1.0, 1.0))

    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("cuMemAlloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(nbytes)))
        return d

    d_qweight = dev_alloc(len(qweight_data))
    d_scales = dev_alloc(len(scales_data))
    d_input = dev_alloc(len(input_data))
    d_output_ref = dev_alloc(N * 4)

    for d, buf in [(d_qweight, qweight_data), (d_scales, scales_data), (d_input, input_data)]:
        b = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
        _check("HtoD", cuda.cuMemcpyHtoD_v2(d, ctypes.cast(b, c_void_p), c_size_t(len(buf))))

    # --- Generate reference using gptq_matvec_tc ---
    print("Computing reference output with gptq_matvec_tc...")
    ref_cubin = os.path.join(KERNEL_DIR, "gptq_matvec_tc.cubin")
    if not os.path.exists(ref_cubin):
        print(f"  ERROR: {ref_cubin} not found. Cannot validate correctness.")
        return

    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod), open(ref_cubin, "rb").read()))
    ref_func = CUfunction()
    _check("getfunc ref", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"gptq_matvec_tc"))

    ref_grid_x = (N + 127) // 128
    _check("memset ref", cuda.cuMemsetD8_v2(d_output_ref, 0, c_size_t(N * 4)))
    ref_args = [c_uint64(d_qweight.value), c_uint64(d_scales.value), c_uint64(d_input.value),
                c_uint64(d_output_ref.value), c_uint32(N), c_uint32(K)]
    ref_ptrs = (c_void_p * len(ref_args))()
    for i, a in enumerate(ref_args):
        ref_ptrs[i] = ctypes.cast(byref(a), c_void_p).value
    _check("launch ref", cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, 256, 1, 1, K * 4, None, ref_ptrs, None))
    _check("sync", cuda.cuCtxSynchronize())

    out_ref = bytearray(N * 4)
    _check("DtoH ref", cuda.cuMemcpyDtoH_v2(ctypes.cast((ctypes.c_ubyte * (N*4)).from_buffer(out_ref), c_void_p), d_output_ref, c_size_t(N * 4)))
    ref_vals = [struct.unpack_from('<f', out_ref, i * 4)[0] for i in range(N)]

    # Bandwidth calculation
    qw_bytes = K * N // 2
    sc_bytes = n_groups * N * 2
    in_bytes = K * 4
    out_bytes = N * 4
    total_bytes = qw_bytes + sc_bytes + in_bytes + out_bytes

    ev_start = CUevent()
    ev_stop = CUevent()
    _check("evCreate", cuda.cuEventCreate(byref(ev_start), 0))
    _check("evCreate", cuda.cuEventCreate(byref(ev_stop), 0))

    WARMUP = 200
    ITERS = 1000

    K_SPLITS_OPTIONS = [s for s in [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640]
                        if K_packed % s == 0 and (K_packed // s) >= 1]

    # Benchmark reference kernel too
    print(f"\nBenchmarking reference kernel (gptq_matvec_tc)...")
    for _ in range(WARMUP):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, 256, 1, 1, K * 4, None, ref_ptrs, None)
    _check("sync", cuda.cuCtxSynchronize())
    _check("evRec", cuda.cuEventRecord(ev_start, None))
    for _ in range(ITERS):
        cuda.cuLaunchKernel(ref_func, ref_grid_x, 1, 1, 256, 1, 1, K * 4, None, ref_ptrs, None)
    _check("evRec", cuda.cuEventRecord(ev_stop, None))
    _check("evSync", cuda.cuEventSynchronize(ev_stop))
    ms_ref = c_float()
    _check("elapsed", cuda.cuEventElapsedTime(byref(ms_ref), ev_start, ev_stop))
    ms_per_ref = ms_ref.value / ITERS
    bw_ref = total_bytes / (ms_per_ref / 1000.0) / 1e9
    print(f"  Reference: {ms_per_ref*1000:.1f} us, {bw_ref:.1f} GB/s")

    # Test each kernel variant
    overall_best_bw = 0
    overall_best_name = ""

    for kern_name in sorted(cubins.keys()):
        cubin_path = cubins[kern_name]
        info = kernels_to_test[kern_name]

        print(f"\n{'='*70}")
        print(f"  Kernel: {kern_name}")
        print(f"  Block: {info['block_x']} threads, {info['cols_per_block']} cols/block")
        print(f"{'='*70}")

        mod = CUmodule()
        _check(f"load {kern_name}", cuda.cuModuleLoadData(byref(mod), open(cubin_path, "rb").read()))
        func = CUfunction()
        _check(f"getfunc {kern_name}", cuda.cuModuleGetFunction(byref(func), mod, info['entry'].encode()))

        num_regs = c_int()
        cuda.cuFuncGetAttribute(byref(num_regs), 4, func)
        print(f"  Register usage: {num_regs.value} regs/thread")

        d_output = dev_alloc(N * 4)

        best_bw = 0
        best_splits = 1
        best_ms = 999

        print(f"\n  {'SPLITS':>6} {'k/split':>7} {'blocks':>6} {'smem_KB':>7} {'time_us':>8} {'GB/s':>8} {'status':>6}")

        for K_SPLITS in K_SPLITS_OPTIONS:
            k_packed_per_split = K_packed // K_SPLITS
            smem_bytes = k_packed_per_split * info['smem_bytes_per_k']  # Input slice
            if smem_bytes > 48 * 1024:
                continue

            grid_x = (N + info['cols_per_block'] - 1) // info['cols_per_block']
            grid_y = K_SPLITS

            # Correctness check
            _check("memset", cuda.cuMemsetD8_v2(d_output, 0, c_size_t(N * 4)))
            args = [c_uint64(d_qweight.value), c_uint64(d_scales.value), c_uint64(d_input.value),
                    c_uint64(d_output.value), c_uint32(N), c_uint32(K),
                    c_uint32(K_SPLITS), c_uint32(k_packed_per_split)]
            ptrs = (c_void_p * len(args))()
            for i, a in enumerate(args):
                ptrs[i] = ctypes.cast(byref(a), c_void_p).value

            _check("launch", cuda.cuLaunchKernel(func, grid_x, grid_y, 1,
                                                  info['block_x'], 1, 1,
                                                  smem_bytes, None, ptrs, None))
            _check("sync", cuda.cuCtxSynchronize())

            out_buf = bytearray(N * 4)
            _check("DtoH", cuda.cuMemcpyDtoH_v2(
                ctypes.cast((ctypes.c_ubyte * (N*4)).from_buffer(out_buf), c_void_p),
                d_output, c_size_t(N * 4)))
            test_vals = [struct.unpack_from('<f', out_buf, i * 4)[0] for i in range(N)]

            # f16 kernels have lower precision -- use relaxed tolerance
            rtol = 0.05 if 'v4' in kern_name else 0.01
            n_wrong = 0
            max_abs_err = 0.0
            for i in range(N):
                err = abs(ref_vals[i] - test_vals[i])
                max_abs_err = max(max_abs_err, err)
                if err / max(abs(ref_vals[i]), 1e-8) > rtol:
                    n_wrong += 1
            status = "PASS" if n_wrong == 0 else f"FAIL({n_wrong})"

            if n_wrong > 0 and K_SPLITS <= 2:
                # Print some mismatches for debugging
                print(f"    First mismatches:")
                shown = 0
                for i in range(N):
                    err = abs(ref_vals[i] - test_vals[i])
                    if err / max(abs(ref_vals[i]), 1e-8) > 0.01:
                        print(f"      [{i}] ref={ref_vals[i]:.6f} got={test_vals[i]:.6f} err={err:.6f}")
                        shown += 1
                        if shown >= 5:
                            break

            # Benchmark
            for _ in range(WARMUP):
                _check("memset", cuda.cuMemsetD8_v2(d_output, 0, c_size_t(N * 4)))
                cuda.cuLaunchKernel(func, grid_x, grid_y, 1,
                                    info['block_x'], 1, 1,
                                    smem_bytes, None, ptrs, None)
            _check("sync", cuda.cuCtxSynchronize())

            # Time kernel only (output already zeroed from warmup -- but we need
            # to zero before each launch for atomicAdd correctness)
            # Actually for timing, we just want kernel throughput. The memset
            # adds overhead. Let's time kernel launches only and accept that
            # with K_SPLITS>1 the atomicAdd accumulates stale data -- doesn't
            # matter for timing.
            _check("evRec", cuda.cuEventRecord(ev_start, None))
            for _ in range(ITERS):
                cuda.cuLaunchKernel(func, grid_x, grid_y, 1,
                                    info['block_x'], 1, 1,
                                    smem_bytes, None, ptrs, None)
            _check("evRec", cuda.cuEventRecord(ev_stop, None))
            _check("evSync", cuda.cuEventSynchronize(ev_stop))
            ms_elapsed = c_float()
            _check("elapsed", cuda.cuEventElapsedTime(byref(ms_elapsed), ev_start, ev_stop))
            ms_per = ms_elapsed.value / ITERS

            bw = total_bytes / (ms_per / 1000.0) / 1e9
            total_blocks = grid_x * grid_y
            print(f"  {K_SPLITS:6d} {k_packed_per_split:7d} {total_blocks:6d} {smem_bytes/1024:7.1f} {ms_per*1000:8.1f} {bw:8.1f} {status:>6}")

            if bw > best_bw and n_wrong == 0:
                best_bw = bw
                best_splits = K_SPLITS
                best_ms = ms_per

        print(f"\n  Best for {kern_name}: K_SPLITS={best_splits}, {best_ms*1000:.1f} us, {best_bw:.1f} GB/s")
        if best_bw > overall_best_bw:
            overall_best_bw = best_bw
            overall_best_name = kern_name

        cuda.cuMemFree_v2(d_output)
        cuda.cuModuleUnload(mod)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Reference (gptq_matvec_tc): {ms_per_ref*1000:7.1f} us  {bw_ref:7.1f} GB/s")
    print(f"  Best overall: {overall_best_name} at {overall_best_bw:.1f} GB/s")
    print(f"  Data moved per call: {total_bytes/1e6:.2f} MB")
    print(f"  Theoretical min @ 4000 GB/s: {total_bytes / 4e12 * 1e6:.1f} us")
    print(f"  Achieved BW / peak: {overall_best_bw/4000*100:.1f}%")
    print(f"  vs cuBLAS target (3460 GB/s): {overall_best_bw/3460*100:.1f}%")

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start)
    cuda.cuEventDestroy_v2(ev_stop)
    for d in [d_qweight, d_scales, d_input, d_output_ref]:
        cuda.cuMemFree_v2(d)
    cuda.cuModuleUnload(ref_mod)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    main()
