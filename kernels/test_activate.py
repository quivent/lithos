#!/usr/bin/env python3
"""Test activate kernel (gate * silu(up)) with MLP dim = 17408."""

import ctypes
import numpy as np
import os
import sys

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_SUCCESS = 0
N = 17408  # MLP dimension to verify

def check(result, msg):
    if result != CUDA_SUCCESS:
        print(f"FAIL: {msg} (error {result})")
        sys.exit(1)

def silu(x):
    return x / (1.0 + np.exp(-x))

def main():
    cuda = ctypes.CDLL("libcuda.so.1")
    check(cuda.cuInit(0), "cuInit")

    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    # Load activate.cubin
    mod = ctypes.c_void_p()
    cubin = os.path.join(KERNEL_DIR, "activate.cubin")
    check(cuda.cuModuleLoad(ctypes.byref(mod), cubin.encode()), "cuModuleLoad")

    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"activate"), "cuModuleGetFunction")

    # Prepare data
    np.random.seed(42)
    gate = np.random.randn(N).astype(np.float32)
    up = np.random.randn(N).astype(np.float32)
    output = np.zeros(N, dtype=np.float32)

    # Expected result
    expected = gate * silu(up)

    # Allocate device memory
    d_gate = ctypes.c_uint64()
    d_up = ctypes.c_uint64()
    d_out = ctypes.c_uint64()
    nbytes = N * 4

    check(cuda.cuMemAlloc_v2(ctypes.byref(d_gate), nbytes), "cuMemAlloc gate")
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_up), nbytes), "cuMemAlloc up")
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes), "cuMemAlloc out")

    # Copy to device
    check(cuda.cuMemcpyHtoD_v2(d_gate, gate.ctypes.data, nbytes), "H2D gate")
    check(cuda.cuMemcpyHtoD_v2(d_up, up.ctypes.data, nbytes), "H2D up")

    # Launch: 256 threads/block, enough blocks
    block = 256
    grid = (N + block - 1) // block
    n_val = ctypes.c_uint32(N)

    # Build kernel params array - each element is a pointer to the argument
    p_gate = ctypes.c_uint64(d_gate.value)
    p_up = ctypes.c_uint64(d_up.value)
    p_out = ctypes.c_uint64(d_out.value)
    params = (ctypes.c_void_p * 4)(
        ctypes.addressof(p_gate),
        ctypes.addressof(p_up),
        ctypes.addressof(p_out),
        ctypes.addressof(n_val),
    )

    check(cuda.cuLaunchKernel(
        func,
        grid, 1, 1,
        block, 1, 1,
        0, ctypes.c_void_p(0),
        params, ctypes.c_void_p(0)
    ), "cuLaunchKernel")

    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    # Copy back
    check(cuda.cuMemcpyDtoH_v2(output.ctypes.data, d_out, nbytes), "D2H")

    # Check correctness
    # SiLU uses approx exp, so we need a reasonable tolerance
    max_err = np.max(np.abs(output - expected))
    rel_err = np.max(np.abs(output - expected) / (np.abs(expected) + 1e-7))

    print(f"activate kernel test: N={N}")
    print(f"  max absolute error: {max_err:.6e}")
    print(f"  max relative error: {rel_err:.6e}")

    if rel_err < 1e-3:
        print("  PASS")
    else:
        # Show worst mismatches
        worst = np.argsort(np.abs(output - expected))[-5:]
        for i in worst:
            print(f"    idx={i}: got={output[i]:.8f} expected={expected[i]:.8f}")
        print("  FAIL: relative error too large")
        sys.exit(1)

    # Cleanup
    cuda.cuMemFree_v2(d_gate)
    cuda.cuMemFree_v2(d_up)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

if __name__ == "__main__":
    main()
