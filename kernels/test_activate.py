#!/usr/bin/env python3
"""Test activate kernel (gate * silu(up)) with MLP dim = 17408."""

import numpy as np
import os
import sys

import pycuda.driver as cuda
import pycuda.autoinit

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
N = 17408  # MLP dimension

def silu(x):
    return x / (1.0 + np.exp(-x))

def main():
    cubin = os.path.join(KERNEL_DIR, "activate.cubin")
    mod = cuda.module_from_file(cubin)
    func = mod.get_function("activate")

    np.random.seed(42)
    gate = np.random.randn(N).astype(np.float32)
    up = np.random.randn(N).astype(np.float32)
    output = np.zeros(N, dtype=np.float32)
    expected = gate * silu(up)

    d_gate = cuda.mem_alloc(gate.nbytes)
    d_up = cuda.mem_alloc(up.nbytes)
    d_out = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_gate, gate)
    cuda.memcpy_htod(d_up, up)

    block = (256, 1, 1)
    grid = ((N + 255) // 256, 1)

    func(d_gate, d_up, d_out, np.uint32(N), block=block, grid=grid)

    cuda.memcpy_dtoh(output, d_out)

    max_err = np.max(np.abs(output - expected))
    rel_err = np.max(np.abs(output - expected) / (np.abs(expected) + 1e-7))

    print(f"activate kernel test: N={N}")
    print(f"  max absolute error: {max_err:.6e}")
    print(f"  max relative error: {rel_err:.6e}")

    if rel_err < 1e-3:
        print("  PASS")
    else:
        worst = np.argsort(np.abs(output - expected))[-5:]
        for i in worst:
            print(f"    idx={i}: got={output[i]:.8f} expected={expected[i]:.8f}")
        print("  FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
