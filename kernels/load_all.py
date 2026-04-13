#!/usr/bin/env python3
"""Load all 10 lithos inference cubins via the CUDA driver API and verify they load successfully."""

import ctypes
import os
import sys

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

KERNELS = [
    ("projection", "projection"),
    ("attention_score", "attention_score"),
    ("norm", "norm"),
    ("activate", "activate"),
    ("rotate", "rotate"),
    ("sample", "sample"),
    ("embed", "embed"),
    ("recurrence", "recurrence"),
    ("recurrence_rollback", "checkpoint"),
    ("recurrence_rollback", "rollback"),
    ("conv1d", "conv1d"),
]

# CUDA driver API result codes
CUDA_SUCCESS = 0

def cuda_check(result, msg):
    if result != CUDA_SUCCESS:
        print(f"  FAIL: {msg} (error code {result})")
        return False
    return True


def main():
    # Load CUDA driver library
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            cuda = ctypes.CDLL("libcuda.so")
        except OSError:
            print("ERROR: Could not load libcuda.so — is the NVIDIA driver installed?")
            sys.exit(1)

    # Initialize CUDA
    result = cuda.cuInit(0)
    if not cuda_check(result, "cuInit"):
        sys.exit(1)
    print("CUDA driver initialized.")

    # Get device and create context
    device = ctypes.c_int()
    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if not cuda_check(result, "cuDeviceGet"):
        sys.exit(1)

    context = ctypes.c_void_p()
    result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
    if not cuda_check(result, "cuCtxCreate_v2"):
        sys.exit(1)
    print("CUDA context created.\n")

    # Load each cubin
    total = len(KERNELS)
    passed = 0

    for cubin_name, entry_name in KERNELS:
        cubin_path = os.path.join(KERNEL_DIR, f"{cubin_name}.cubin")
        print(f"Loading {cubin_name}.cubin ...")

        if not os.path.exists(cubin_path):
            print(f"  FAIL: file not found: {cubin_path}")
            continue

        file_size = os.path.getsize(cubin_path)

        # Load module from cubin file
        module = ctypes.c_void_p()
        cubin_path_bytes = cubin_path.encode("utf-8")
        result = cuda.cuModuleLoad(ctypes.byref(module), cubin_path_bytes)
        if not cuda_check(result, "cuModuleLoad"):
            continue

        # Get the kernel function handle
        function = ctypes.c_void_p()
        entry_bytes = entry_name.encode("utf-8")
        result = cuda.cuModuleGetFunction(ctypes.byref(function), module, entry_bytes)
        if not cuda_check(result, f"cuModuleGetFunction('{entry_name}')"):
            cuda.cuModuleUnload(module)
            continue

        print(f"  OK: module loaded, entry '{entry_name}' found ({file_size} bytes)")
        passed += 1

        # Unload module
        cuda.cuModuleUnload(module)

    # Destroy context
    cuda.cuCtxDestroy_v2(context)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} kernels loaded successfully.")

    if passed == total:
        print("All kernels verified.")
        return 0
    else:
        print(f"WARNING: {total - passed} kernel(s) failed to load.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
