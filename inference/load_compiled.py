#!/usr/bin/env python3
"""Load compiled Lithos inference cubins and return kernel function handles.

The five .li files compile to cubins with multiple entry points each.
This module loads them and returns a dict mapping entry-point names to
CUDA function handles (as integers) suitable for cuLaunchKernel.

Usage:
    from load_compiled import load_compiled_kernels
    handles = load_compiled_kernels()
    # handles["activate_silu"] -> CUfunction handle
    # handles["rmsnorm"] -> CUfunction handle
    # etc.
"""

import ctypes
import os
import sys

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# Map: (cubin_filename, [entry_point_names])
COMPILED_CUBINS = [
    ("elementwise", ["residual_add", "elemwise_mul", "activate_silu", "scale"]),
    ("reduce", ["rmsnorm", "sample_argmax"]),
    ("gemv", ["gptq_gemv"]),
    ("attend", ["rope", "attention_score"]),
    ("recur", ["conv1d_infer", "gate_sigmoid", "deltanet_step", "state_rollback"]),
]

CUDA_SUCCESS = 0


def cuda_check(result, msg):
    if result != CUDA_SUCCESS:
        print(f"  FAIL: {msg} (error code {result})")
        return False
    return True


def load_compiled_kernels(cuda=None, verbose=True):
    """Load all compiled cubins and return a dict of entry-point handles.

    Parameters:
        cuda: ctypes.CDLL for libcuda.so (if None, will be loaded)
        verbose: print loading status

    Returns:
        dict mapping entry-point name -> int (CUfunction handle)
    """
    if cuda is None:
        try:
            cuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            cuda = ctypes.CDLL("libcuda.so")

    handles = {}
    total_entries = sum(len(entries) for _, entries in COMPILED_CUBINS)
    loaded = 0

    for cubin_name, entry_names in COMPILED_CUBINS:
        cubin_path = os.path.join(INFERENCE_DIR, f"{cubin_name}.cubin")
        if verbose:
            print(f"Loading {cubin_name}.cubin ...")

        if not os.path.exists(cubin_path):
            if verbose:
                print(f"  SKIP: {cubin_path} not found")
            continue

        # Load module
        module = ctypes.c_void_p()
        result = cuda.cuModuleLoad(
            ctypes.byref(module), cubin_path.encode("utf-8")
        )
        if not cuda_check(result, f"cuModuleLoad({cubin_name}.cubin)"):
            continue

        # Get each entry point
        for entry_name in entry_names:
            function = ctypes.c_void_p()
            result = cuda.cuModuleGetFunction(
                ctypes.byref(function), module, entry_name.encode("utf-8")
            )
            if cuda_check(result, f"cuModuleGetFunction('{entry_name}')"):
                handles[entry_name] = function.value
                loaded += 1
                if verbose:
                    print(f"  OK: '{entry_name}' loaded")
            else:
                if verbose:
                    print(f"  FAIL: '{entry_name}' not found in {cubin_name}.cubin")

    if verbose:
        print(f"\nCompiled kernels: {loaded}/{total_entries} loaded.")

    return handles


def populate_loaded_kernels(handles):
    """Create a LoadedKernels instance from compiled kernel handles.

    Maps the compiled entry points to the engine's kernel function slots.
    """
    # Import here to avoid circular dependency
    sys.path.insert(0, os.path.join(os.path.dirname(INFERENCE_DIR), "src"))
    from engine import LoadedKernels

    kernels = LoadedKernels()

    # Map compiled kernel entry points to engine kernel slots
    kernels.projection_func = handles.get("gptq_gemv", 0)
    kernels.attention_score_func = handles.get("attention_score", 0)
    kernels.recurrence_func = handles.get("deltanet_step", 0)
    kernels.norm_func = handles.get("rmsnorm", 0)
    kernels.activate_func = handles.get("activate_silu", 0)
    kernels.rotate_func = handles.get("rope", 0)
    kernels.sample_func = handles.get("sample_argmax", 0)
    # embed_func stays at 0 — embedding is not yet compiled from .li

    return kernels


if __name__ == "__main__":
    # Stand-alone test: initialize CUDA, load all cubins, print results
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        cuda = ctypes.CDLL("libcuda.so")

    result = cuda.cuInit(0)
    if not cuda_check(result, "cuInit"):
        sys.exit(1)

    device = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)

    handles = load_compiled_kernels(cuda)

    print(f"\nEntry points loaded: {list(handles.keys())}")

    cuda.cuCtxDestroy_v2(context)
    sys.exit(0 if handles else 1)
