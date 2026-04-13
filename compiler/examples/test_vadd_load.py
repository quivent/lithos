#!/usr/bin/env python3
"""Loadability test for the Lithos-compiled vadd cubin.

Verifies that:
  * cuModuleLoadData accepts the cubin produced by `lithos --emit cubin`
  * cuModuleGetFunction("vadd") returns a non-null handle

Does NOT launch the kernel — correctness of the generated SASS is a
separate concern; the goal here is ELF-wrapping correctness.

Usage (from repo root):
    compiler/run-lithos.sh compiler/examples/vadd.li --emit cubin \
        -o /tmp/vadd.cubin
    python3 compiler/examples/test_vadd_load.py /tmp/vadd.cubin
"""

from __future__ import annotations
import ctypes
import sys

CUBIN_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vadd.cubin"


def check(err: int, msg: str) -> None:
    if err != 0:
        # Fetch driver-side error string for nicer diagnostics
        s = ctypes.c_char_p()
        try:
            cuda.cuGetErrorName(err, ctypes.byref(s))
            name = s.value.decode() if s.value else "?"
        except Exception:
            name = "?"
        raise SystemExit(f"{msg}: CUDA error {err} ({name})")


cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuCtxCreate_v2.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int,
]
cuda.cuModuleLoadData.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
]
cuda.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
]
cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

check(cuda.cuInit(0), "cuInit")

dev = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

ctx = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate_v2")

with open(CUBIN_PATH, "rb") as f:
    cubin_bytes = f.read()
print(f"cubin size: {len(cubin_bytes)} bytes")

buf = ctypes.create_string_buffer(cubin_bytes, len(cubin_bytes))
mod = ctypes.c_void_p()
check(cuda.cuModuleLoadData(ctypes.byref(mod), buf), "cuModuleLoadData")
print(f"cuModuleLoadData OK, module handle = 0x{mod.value:x}")

fn = ctypes.c_void_p()
check(
    cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"vadd"),
    "cuModuleGetFunction(vadd)",
)
print(f"cuModuleGetFunction OK, fn handle = 0x{fn.value:x}")

cuda.cuCtxDestroy_v2(ctx)
print("PASS: vadd.cubin loads and exposes kernel 'vadd'")
