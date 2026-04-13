#!/usr/bin/env python3
"""Test elemwise_mul kernel: output[i] = a[i] * b[i]."""

import numpy as np
import os
import sys

import pycuda.driver as cuda
import pycuda.autoinit

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

def test(N, label):
    mod = cuda.module_from_file(os.path.join(KERNEL_DIR, "elemwise_mul.cubin"))
    func = mod.get_function("elemwise_mul")

    np.random.seed(123)
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    output = np.zeros(N, dtype=np.float32)
    expected = a * b

    d_a = cuda.mem_alloc(a.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    d_out = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_a, a)
    cuda.memcpy_htod(d_b, b)

    block = (256, 1, 1)
    grid = ((N + 255) // 256, 1)
    func(d_a, d_b, d_out, np.uint32(N), block=block, grid=grid)

    cuda.memcpy_dtoh(output, d_out)

    max_err = np.max(np.abs(output - expected))
    print(f"elemwise_mul {label} (N={N}): max_err={max_err:.6e}", end=" ")
    if max_err < 1e-6:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

# Test with aligned size, non-aligned size, and large size
test(1024, "aligned")
test(1023, "non-aligned (tail)")
test(17408, "MLP dim")
test(1, "single element")
test(3, "3 elements (tail only)")
