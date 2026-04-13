#!/usr/bin/env python3
"""
Lithos proof-of-life: end-to-end test that the pipeline works on this GH200.

Test 1 - Vector Add:  Generate PTX -> compile via ptxas -> load -> launch -> verify
Test 2 - RMSNorm:     Load norm.cubin -> launch -> verify normalization
Test 3 - Embed:       Load embed.cubin -> launch -> verify row lookup

Uses only ctypes against libcuda.so (CUDA driver API). No PyCUDA / CuPy dependency.
"""

import ctypes
import ctypes.util
import math
import numpy as np
import os
import struct
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# CUDA driver API constants
# ---------------------------------------------------------------------------
CUDA_SUCCESS = 0
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6

# ---------------------------------------------------------------------------
# Load libcuda
# ---------------------------------------------------------------------------
def _load_cuda():
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    print("FATAL: cannot load libcuda.so -- is the NVIDIA driver installed?")
    sys.exit(1)

cuda = _load_cuda()

# ---------------------------------------------------------------------------
# Thin wrappers
# ---------------------------------------------------------------------------
def check(result, tag=""):
    if result != CUDA_SUCCESS:
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(err))
        msg = err.value.decode() if err.value else f"error {result}"
        raise RuntimeError(f"CUDA {tag}: {msg} (code {result})")

def cu_init():
    check(cuda.cuInit(0), "cuInit")

def cu_device_get(ordinal=0):
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
    return dev

def cu_ctx_create(dev, flags=0):
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), flags, dev), "cuCtxCreate")
    return ctx

def cu_ctx_destroy(ctx):
    cuda.cuCtxDestroy_v2(ctx)

def cu_module_load(path):
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(mod), path.encode()), f"cuModuleLoad({path})")
    return mod

def cu_module_load_data(data: bytes):
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), data), "cuModuleLoadData")
    return mod

def cu_module_get_function(mod, name):
    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, name.encode()),
          f"cuModuleGetFunction({name})")
    return func

def cu_module_unload(mod):
    cuda.cuModuleUnload(mod)

def cu_mem_alloc(nbytes):
    dptr = ctypes.c_uint64(0)
    check(cuda.cuMemAlloc_v2(ctypes.byref(dptr), ctypes.c_size_t(nbytes)), "cuMemAlloc")
    return dptr

def cu_memcpy_htod(dptr, host_buf, nbytes):
    check(cuda.cuMemcpyHtoD_v2(dptr, host_buf, ctypes.c_size_t(nbytes)), "cuMemcpyHtoD")

def cu_memcpy_dtoh(host_buf, dptr, nbytes):
    check(cuda.cuMemcpyDtoH_v2(host_buf, dptr, ctypes.c_size_t(nbytes)), "cuMemcpyDtoH")

def cu_mem_free(dptr):
    cuda.cuMemFree_v2(dptr)

def cu_ctx_synchronize():
    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

def cu_launch_kernel(func, gx, gy, gz, bx, by, bz, shared_mem, stream, params):
    """
    Launch a CUDA kernel.  `params` is a list of ctypes objects whose addresses
    will be packed into the void** kernelParams array.
    """
    # Build the void** array: each element is a pointer to the parameter value
    param_ptrs = (ctypes.c_void_p * len(params))()
    for i, p in enumerate(params):
        param_ptrs[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)
    check(cuda.cuLaunchKernel(
        func,
        gx, gy, gz,
        bx, by, bz,
        shared_mem,
        stream,
        param_ptrs,
        None,  # extra
    ), "cuLaunchKernel")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")

def divup(a, b):
    return (a + b - 1) // b

def np_to_gpu(arr):
    """Copy a numpy array to GPU, return device pointer (c_uint64)."""
    buf = arr.ctypes.data_as(ctypes.c_void_p)
    nbytes = arr.nbytes
    dptr = cu_mem_alloc(nbytes)
    cu_memcpy_htod(dptr, buf, nbytes)
    return dptr

def gpu_to_np(dptr, shape, dtype=np.float32):
    """Copy GPU memory back to numpy array."""
    arr = np.empty(shape, dtype=dtype)
    cu_memcpy_dtoh(arr.ctypes.data_as(ctypes.c_void_p), dptr, arr.nbytes)
    return arr


# ===================================================================
# TEST 1: Vector Add -- generate PTX, compile, load, launch, verify
# ===================================================================
def generate_vecadd_ptx():
    """
    Generate PTX for a vector-add kernel using the Lithos pattern idioms:
    grid-stride loop, coalesced f32 loads/stores, f32 add.
    This mirrors what the Lithos Forth patterns would emit.

    Signature: vecadd(a_ptr: u64, b_ptr: u64, c_ptr: u64, n: u32)
    """
    return """\
.version 8.5
.target sm_90
.address_size 64

.visible .entry vecadd(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .u64   %rd<10>;
    .reg .u32   %r<10>;
    .reg .f32   %f<4>;
    .reg .pred  %p<2>;

    // Load params
    ld.param.u64    %rd0, [a_ptr];
    ld.param.u64    %rd1, [b_ptr];
    ld.param.u64    %rd2, [c_ptr];
    ld.param.u32    %r0, [n];

    // gtid.x = blockIdx.x * blockDim.x + threadIdx.x  (pattern [1])
    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;

    // stride = gridDim.x * blockDim.x  (pattern [8])
    mov.u32         %r5, %nctaid.x;
    mul.lo.u32      %r5, %r5, %r3;

LOOP:
    // bounds check  (pattern [7])
    setp.ge.u32     %p0, %r4, %r0;
    @%p0 bra        DONE;

    // byte offset
    shl.b32         %r6, %r4, 2;
    cvt.u64.u32     %rd3, %r6;

    // ld.global.f32  (pattern [9])
    add.u64         %rd4, %rd0, %rd3;
    ld.global.f32   %f0, [%rd4];

    add.u64         %rd5, %rd1, %rd3;
    ld.global.f32   %f1, [%rd5];

    // add.f32  (pattern [32])
    add.f32         %f2, %f0, %f1;

    // st.global.f32  (pattern [23])
    add.u64         %rd6, %rd2, %rd3;
    st.global.f32   [%rd6], %f2;

    add.u32         %r4, %r4, %r5;
    bra             LOOP;

DONE:
    ret;
}
"""


def test_vecadd():
    print("=" * 60)
    print("TEST 1: Vector Add  (generate PTX -> ptxas -> load -> launch)")
    print("=" * 60)

    N = 1 << 20  # 1M elements
    BLOCK = 256
    GRID = divup(N, BLOCK)

    # --- Step 1: Generate PTX ---
    t0 = time.perf_counter()
    ptx_src = generate_vecadd_ptx()
    t_gen = time.perf_counter() - t0

    # --- Step 2: Compile PTX -> cubin via ptxas ---
    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w", delete=False) as f:
        f.write(ptx_src)
        ptx_path = f.name
    cubin_path = ptx_path.replace(".ptx", ".cubin")

    t0 = time.perf_counter()
    result = subprocess.run(
        ["ptxas", "-arch=sm_90", "-o", cubin_path, ptx_path],
        capture_output=True, text=True,
    )
    t_compile = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ptxas FAILED:\n{result.stderr}")
        os.unlink(ptx_path)
        return False

    cubin_size = os.path.getsize(cubin_path)
    print(f"  PTX generated   : {len(ptx_src)} bytes  ({t_gen*1e3:.2f} ms)")
    print(f"  ptxas compiled  : {cubin_size} bytes cubin  ({t_compile*1e3:.2f} ms)")

    # --- Step 3: Load cubin ---
    t0 = time.perf_counter()
    with open(cubin_path, "rb") as f:
        cubin_data = f.read()
    mod = cu_module_load_data(cubin_data)
    func = cu_module_get_function(mod, "vecadd")
    t_load = time.perf_counter() - t0
    print(f"  Module loaded   : ({t_load*1e3:.2f} ms)")

    # Cleanup temp files
    os.unlink(ptx_path)
    os.unlink(cubin_path)

    # --- Steps 4-5: Allocate & fill ---
    a_host = np.arange(N, dtype=np.float32)
    b_host = np.arange(N, dtype=np.float32) * 2.0
    c_host = np.zeros(N, dtype=np.float32)

    d_a = np_to_gpu(a_host)
    d_b = np_to_gpu(b_host)
    d_c = cu_mem_alloc(c_host.nbytes)

    # --- Step 6: Launch ---
    params = [
        ctypes.c_uint64(d_a.value),
        ctypes.c_uint64(d_b.value),
        ctypes.c_uint64(d_c.value),
        ctypes.c_uint32(N),
    ]

    t0 = time.perf_counter()
    cu_launch_kernel(func, GRID, 1, 1, BLOCK, 1, 1, 0, None, params)
    cu_ctx_synchronize()
    t_launch = time.perf_counter() - t0
    print(f"  Kernel launched : ({t_launch*1e3:.2f} ms)")

    # --- Step 7: Read back ---
    c_host = gpu_to_np(d_c, (N,))

    # --- Step 8: Verify ---
    expected = a_host + b_host
    maxerr = np.max(np.abs(c_host - expected))
    ok = maxerr < 1e-5

    t_total = t_gen + t_compile + t_load + t_launch
    print(f"  Max error       : {maxerr:.2e}")
    print(f"  Timing          : gen={t_gen*1e3:.2f}ms  compile={t_compile*1e3:.2f}ms  "
          f"load={t_load*1e3:.2f}ms  launch={t_launch*1e3:.2f}ms  total={t_total*1e3:.2f}ms")
    print(f"  Result          : {'PASS' if ok else 'FAIL'}")

    # Cleanup
    cu_mem_free(d_a)
    cu_mem_free(d_b)
    cu_mem_free(d_c)
    cu_module_unload(mod)

    return ok


# ===================================================================
# TEST 2: RMSNorm kernel from norm.cubin
# ===================================================================
def test_rmsnorm():
    print()
    print("=" * 60)
    print("TEST 2: RMSNorm  (load norm.cubin -> launch -> verify)")
    print("=" * 60)

    cubin_path = os.path.join(KERNEL_DIR, "norm.cubin")
    if not os.path.exists(cubin_path):
        print(f"  SKIP: {cubin_path} not found")
        return None

    HIDDEN = 5120
    EPSILON = 1e-5
    BLOCK = 256  # must match what norm.cubin was compiled for
    ROWS = 1

    # Load cubin
    t0 = time.perf_counter()
    mod = cu_module_load(cubin_path)
    func = cu_module_get_function(mod, "norm")
    t_load = time.perf_counter() - t0
    print(f"  Module loaded   : ({t_load*1e3:.2f} ms)")

    # Create test data: input, residual, weight, output
    np.random.seed(42)
    inp = np.random.randn(ROWS, HIDDEN).astype(np.float32) * 0.1
    residual = np.random.randn(ROWS, HIDDEN).astype(np.float32) * 0.05
    weight = np.ones((HIDDEN,), dtype=np.float32)
    output = np.zeros((ROWS, HIDDEN), dtype=np.float32)

    # Upload
    d_inp = np_to_gpu(inp)
    d_res = np_to_gpu(residual)
    d_wt = np_to_gpu(weight)
    d_out = cu_mem_alloc(output.nbytes)

    # Launch: norm(input_ptr, residual_ptr, weight_ptr, output_ptr, hidden_dim, epsilon)
    params = [
        ctypes.c_uint64(d_inp.value),
        ctypes.c_uint64(d_res.value),
        ctypes.c_uint64(d_wt.value),
        ctypes.c_uint64(d_out.value),
        ctypes.c_uint32(HIDDEN),
        ctypes.c_float(EPSILON),
    ]

    t0 = time.perf_counter()
    cu_launch_kernel(func, ROWS, 1, 1, BLOCK, 1, 1, 128, None, params)
    cu_ctx_synchronize()
    t_launch = time.perf_counter() - t0
    print(f"  Kernel launched : ({t_launch*1e3:.2f} ms)")

    # Read back
    result = gpu_to_np(d_out, (ROWS, HIDDEN))

    # CPU reference: RMSNorm with fused residual add
    x = inp + residual
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + EPSILON)
    expected = (x / rms) * weight

    maxerr = np.max(np.abs(result - expected))
    # Use a tolerance that accounts for GPU approx rsqrt
    ok = maxerr < 0.01

    print(f"  Max error       : {maxerr:.6f}")
    print(f"  Sample output[0:5] : {result[0, :5]}")
    print(f"  Sample expected[0:5]: {expected[0, :5]}")
    print(f"  Result          : {'PASS' if ok else 'FAIL'}")

    # Cleanup
    cu_mem_free(d_inp)
    cu_mem_free(d_res)
    cu_mem_free(d_wt)
    cu_mem_free(d_out)
    cu_module_unload(mod)

    return ok


# ===================================================================
# TEST 3: Embed kernel from embed.cubin
# ===================================================================
def test_embed():
    print()
    print("=" * 60)
    print("TEST 3: Embed  (load embed.cubin -> launch -> verify)")
    print("=" * 60)

    cubin_path = os.path.join(KERNEL_DIR, "embed.cubin")
    if not os.path.exists(cubin_path):
        print(f"  SKIP: {cubin_path} not found")
        return None

    VOCAB = 100
    HIDDEN = 5120
    TOKEN_ID = 42
    BLOCK = 256
    GRID = divup(HIDDEN, BLOCK * 4)  # vectorized: each thread handles 4 elements
    if GRID < 1:
        GRID = 1

    # Load cubin
    t0 = time.perf_counter()
    mod = cu_module_load(cubin_path)
    func = cu_module_get_function(mod, "embed")
    t_load = time.perf_counter() - t0
    print(f"  Module loaded   : ({t_load*1e3:.2f} ms)")

    # Create fake embedding table
    np.random.seed(123)
    table = np.random.randn(VOCAB, HIDDEN).astype(np.float32)
    output = np.zeros((HIDDEN,), dtype=np.float32)

    d_table = np_to_gpu(table)
    d_out = cu_mem_alloc(output.nbytes)

    # Launch: embed(token_id, embed_table_ptr, output_ptr, hidden_dim)
    params = [
        ctypes.c_uint32(TOKEN_ID),
        ctypes.c_uint64(d_table.value),
        ctypes.c_uint64(d_out.value),
        ctypes.c_uint32(HIDDEN),
    ]

    t0 = time.perf_counter()
    cu_launch_kernel(func, GRID, 1, 1, BLOCK, 1, 1, 0, None, params)
    cu_ctx_synchronize()
    t_launch = time.perf_counter() - t0
    print(f"  Kernel launched : ({t_launch*1e3:.2f} ms)")

    # Read back
    result = gpu_to_np(d_out, (HIDDEN,))

    # Verify: output should match row 42 exactly
    expected = table[TOKEN_ID]
    maxerr = np.max(np.abs(result - expected))
    ok = maxerr < 1e-6

    print(f"  Token ID        : {TOKEN_ID}")
    print(f"  Max error       : {maxerr:.2e}")
    print(f"  Sample output[0:5]  : {result[:5]}")
    print(f"  Sample table[42,0:5]: {expected[:5]}")
    print(f"  Result          : {'PASS' if ok else 'FAIL'}")

    # Cleanup
    cu_mem_free(d_table)
    cu_mem_free(d_out)
    cu_module_unload(mod)

    return ok


# ===================================================================
# Main
# ===================================================================
def main():
    print("Lithos Proof-of-Life")
    print("=" * 60)

    # Init CUDA
    cu_init()
    dev = cu_device_get(0)

    # Get device name
    name_buf = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name_buf, 256, dev)
    print(f"Device: {name_buf.value.decode()}")

    ctx = cu_ctx_create(dev)
    print(f"Context created.\n")

    results = {}

    t_total_start = time.perf_counter()

    # Test 1: Vector Add
    try:
        results["vecadd"] = test_vecadd()
    except Exception as e:
        print(f"  FAIL (exception): {e}")
        results["vecadd"] = False

    # Test 2: RMSNorm
    try:
        results["rmsnorm"] = test_rmsnorm()
    except Exception as e:
        print(f"  FAIL (exception): {e}")
        results["rmsnorm"] = False

    # Test 3: Embed
    try:
        results["embed"] = test_embed()
    except Exception as e:
        print(f"  FAIL (exception): {e}")
        results["embed"] = False

    t_total = time.perf_counter() - t_total_start

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results.items():
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {name:12s} : {status}")

    print(f"\n  Total time: {t_total*1e3:.2f} ms")

    cu_ctx_destroy(ctx)

    if all_pass:
        print("\n  >>> ALL TESTS PASSED -- Lithos pipeline is alive on this GH200. <<<")
        return 0
    else:
        print("\n  >>> SOME TESTS FAILED <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
