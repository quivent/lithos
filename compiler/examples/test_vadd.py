#!/usr/bin/env python3
"""Test the Lithos-compiled vadd kernel on the GH200.

Compile with:
  forth-bootstrap lithos.fs examples/vadd.li --emit cubin -o /tmp/vadd.cubin
Then run this script.
"""

import ctypes
import ctypes.util
import numpy as np
import struct
import os

# Load CUDA driver API
cuda = ctypes.CDLL("libcuda.so.1")

CUDA_SUCCESS = 0

def check(err, msg=""):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error {err}: {msg}")

# Set proper argtypes for key functions
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int

cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDeviceGet.restype = ctypes.c_int

cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
cuda.cuCtxCreate_v2.restype = ctypes.c_int

cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
cuda.cuModuleLoad.restype = ctypes.c_int

cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
cuda.cuModuleLoadData.restype = ctypes.c_int

cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
cuda.cuModuleGetFunction.restype = ctypes.c_int

cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cuda.cuMemAlloc_v2.restype = ctypes.c_int

cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int

cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int

cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
cuda.cuMemFree_v2.restype = ctypes.c_int

cuda.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,  # function
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block
    ctypes.c_uint,  # shared mem
    ctypes.c_void_p,  # stream
    ctypes.POINTER(ctypes.c_void_p),  # params
    ctypes.POINTER(ctypes.c_void_p),  # extra
]
cuda.cuLaunchKernel.restype = ctypes.c_int

cuda.cuCtxSynchronize.argtypes = []
cuda.cuCtxSynchronize.restype = ctypes.c_int

cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
cuda.cuCtxDestroy_v2.restype = ctypes.c_int

# Initialize
check(cuda.cuInit(0), "cuInit")

dev = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

ctx = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

# Load the cubin — use cuModuleLoadData (in-memory) for lithos-emitted cubins
CUBIN_PATH = "/tmp/vadd.cubin"
module = ctypes.c_void_p()
with open(CUBIN_PATH, "rb") as f:
    cubin_bytes = f.read()
cubin_buf = ctypes.create_string_buffer(cubin_bytes, len(cubin_bytes))
check(cuda.cuModuleLoadData(ctypes.byref(module), cubin_buf), "cuModuleLoadData")

func = ctypes.c_void_p()
check(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"vadd"), "cuModuleGetFunction")

# Allocate device memory
N = 1024
nbytes = N * 4

d_a = ctypes.c_uint64()
d_b = ctypes.c_uint64()
d_c = ctypes.c_uint64()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nbytes), "cuMemAlloc a")
check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nbytes), "cuMemAlloc b")
check(cuda.cuMemAlloc_v2(ctypes.byref(d_c), nbytes), "cuMemAlloc c")

# Host data
h_a = np.arange(N, dtype=np.float32)
h_b = np.arange(N, dtype=np.float32) * 2.0

check(cuda.cuMemcpyHtoD_v2(d_a, h_a.ctypes.data, nbytes), "H2D a")
check(cuda.cuMemcpyHtoD_v2(d_b, h_b.ctypes.data, nbytes), "H2D b")

# Set up kernel parameters
# vadd takes 3 params: a(u64 ptr), b(u64 ptr), c(u64 ptr)
pa = ctypes.c_uint64(d_a.value)
pb = ctypes.c_uint64(d_b.value)
pc = ctypes.c_uint64(d_c.value)

params = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(pa), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(pb), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(pc), ctypes.c_void_p),
)

threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block

check(cuda.cuLaunchKernel(
    func,
    blocks, 1, 1,
    threads_per_block, 1, 1,
    0,
    None,
    params,
    None,
), "cuLaunchKernel")

check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

# Copy back
h_c = np.zeros(N, dtype=np.float32)
check(cuda.cuMemcpyDtoH_v2(h_c.ctypes.data, d_c, nbytes), "D2H c")

# Verify
expected = h_a + h_b
if np.allclose(h_c, expected):
    print(f"PASS: vadd({N}) correct. First 5: {h_c[:5]}")
else:
    diff = np.max(np.abs(h_c - expected))
    print(f"FAIL: max error = {diff}")
    print(f"  expected[:5] = {expected[:5]}")
    print(f"  got[:5]      = {h_c[:5]}")

# Cleanup
cuda.cuMemFree_v2(d_a)
cuda.cuMemFree_v2(d_b)
cuda.cuMemFree_v2(d_c)
cuda.cuCtxDestroy_v2(ctx)
