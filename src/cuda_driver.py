"""
Lithos CUDA Driver API bindings (ctypes, aarch64).

Minimal interface to load cubins and launch kernels via the CUDA driver API.
Marked for rewrite in Rust/C once the interface stabilises.

Usage:
    from cuda_driver import CUDADriver
    gpu = CUDADriver()
    print(gpu.device_name)
    module = gpu.load_cubin("kernels/embed.cubin")
    func = gpu.get_function(module, "embed")
    gpu.launch(func, grid=(1,1,1), block=(256,1,1), args=[...])
"""

from __future__ import annotations

import ctypes
import ctypes.util
from ctypes import (
    POINTER,
    byref,
    c_char,
    c_char_p,
    c_int,
    c_size_t,
    c_uint,
    c_uint64,
    c_void_p,
)
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Type aliases matching CUDA driver API on aarch64-linux (LP64)
# ---------------------------------------------------------------------------
CUresult = c_int
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUstream = c_void_p
CUdeviceptr = c_uint64  # 64-bit device pointer on aarch64


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class CUDAError(RuntimeError):
    """Raised when a CUDA driver call returns a non-zero status."""

    _ERROR_NAMES: dict[int, str] = {
        0: "CUDA_SUCCESS",
        1: "CUDA_ERROR_INVALID_VALUE",
        2: "CUDA_ERROR_OUT_OF_MEMORY",
        3: "CUDA_ERROR_NOT_INITIALIZED",
        100: "CUDA_ERROR_NO_DEVICE",
        101: "CUDA_ERROR_INVALID_DEVICE",
        200: "CUDA_ERROR_INVALID_IMAGE",
        201: "CUDA_ERROR_INVALID_CONTEXT",
        301: "CUDA_ERROR_FILE_NOT_FOUND",
        400: "CUDA_ERROR_INVALID_HANDLE",
        500: "CUDA_ERROR_NOT_FOUND",
        700: "CUDA_ERROR_LAUNCH_FAILED",
        701: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        702: "CUDA_ERROR_LAUNCH_TIMEOUT",
        999: "CUDA_ERROR_UNKNOWN",
    }

    def __init__(self, func_name: str, code: int) -> None:
        name = self._ERROR_NAMES.get(code, f"CUDA_ERROR_{code}")
        super().__init__(f"{func_name} failed: {name} ({code})")
        self.code = code


def _check(func_name: str, result: int) -> None:
    if result != 0:
        raise CUDAError(func_name, result)


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------
def _load_libcuda() -> ctypes.CDLL:
    """Load libcuda.so.1 and set up function prototypes."""
    lib = ctypes.CDLL("libcuda.so.1")

    # -- cuInit ---------------------------------------------------------------
    lib.cuInit.argtypes = [c_uint]
    lib.cuInit.restype = CUresult

    # -- cuDeviceGet ----------------------------------------------------------
    lib.cuDeviceGet.argtypes = [POINTER(CUdevice), c_int]
    lib.cuDeviceGet.restype = CUresult

    # -- cuDeviceGetName ------------------------------------------------------
    lib.cuDeviceGetName.argtypes = [c_char_p, c_int, CUdevice]
    lib.cuDeviceGetName.restype = CUresult

    # -- cuDeviceGetAttribute -------------------------------------------------
    lib.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, CUdevice]
    lib.cuDeviceGetAttribute.restype = CUresult

    # -- cuCtxCreate_v2 -------------------------------------------------------
    lib.cuCtxCreate_v2.argtypes = [POINTER(CUcontext), c_uint, CUdevice]
    lib.cuCtxCreate_v2.restype = CUresult

    # -- cuCtxDestroy_v2 ------------------------------------------------------
    lib.cuCtxDestroy_v2.argtypes = [CUcontext]
    lib.cuCtxDestroy_v2.restype = CUresult

    # -- cuCtxSynchronize -----------------------------------------------------
    lib.cuCtxSynchronize.argtypes = []
    lib.cuCtxSynchronize.restype = CUresult

    # -- cuModuleLoadData -----------------------------------------------------
    lib.cuModuleLoadData.argtypes = [POINTER(CUmodule), c_void_p]
    lib.cuModuleLoadData.restype = CUresult

    # -- cuModuleGetFunction --------------------------------------------------
    lib.cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, c_char_p]
    lib.cuModuleGetFunction.restype = CUresult

    # -- cuLaunchKernel -------------------------------------------------------
    lib.cuLaunchKernel.argtypes = [
        CUfunction,           # f
        c_uint, c_uint, c_uint,  # gridDimX/Y/Z
        c_uint, c_uint, c_uint,  # blockDimX/Y/Z
        c_uint,               # sharedMemBytes
        CUstream,             # hStream
        c_void_p,             # kernelParams (void**)
        c_void_p,             # extra
    ]
    lib.cuLaunchKernel.restype = CUresult

    # -- cuMemAlloc_v2 --------------------------------------------------------
    lib.cuMemAlloc_v2.argtypes = [POINTER(CUdeviceptr), c_size_t]
    lib.cuMemAlloc_v2.restype = CUresult

    # -- cuMemFree_v2 ---------------------------------------------------------
    lib.cuMemFree_v2.argtypes = [CUdeviceptr]
    lib.cuMemFree_v2.restype = CUresult

    # -- cuMemcpyHtoD_v2 ------------------------------------------------------
    lib.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, c_void_p, c_size_t]
    lib.cuMemcpyHtoD_v2.restype = CUresult

    # -- cuMemcpyDtoH_v2 ------------------------------------------------------
    lib.cuMemcpyDtoH_v2.argtypes = [c_void_p, CUdeviceptr, c_size_t]
    lib.cuMemcpyDtoH_v2.restype = CUresult

    # -- cuStreamCreate -------------------------------------------------------
    lib.cuStreamCreate.argtypes = [POINTER(CUstream), c_uint]
    lib.cuStreamCreate.restype = CUresult

    # -- cuStreamSynchronize --------------------------------------------------
    lib.cuStreamSynchronize.argtypes = [CUstream]
    lib.cuStreamSynchronize.restype = CUresult

    # -- cuMemcpyHtoDAsync_v2 -------------------------------------------------
    lib.cuMemcpyHtoDAsync_v2.argtypes = [CUdeviceptr, c_void_p, c_size_t, CUstream]
    lib.cuMemcpyHtoDAsync_v2.restype = CUresult

    # -- cuFuncSetAttribute ---------------------------------------------------
    lib.cuFuncSetAttribute.argtypes = [CUfunction, c_int, c_int]
    lib.cuFuncSetAttribute.restype = CUresult

    # -- cuMemsetD8_v2 --------------------------------------------------------
    lib.cuMemsetD8_v2.argtypes = [CUdeviceptr, ctypes.c_ubyte, c_size_t]
    lib.cuMemsetD8_v2.restype = CUresult

    # -- cuMemsetD8Async -------------------------------------------------------
    lib.cuMemsetD8Async.argtypes = [CUdeviceptr, ctypes.c_ubyte, c_size_t, CUstream]
    lib.cuMemsetD8Async.restype = CUresult

    # -- cuLaunchCooperativeKernel --------------------------------------------
    lib.cuLaunchCooperativeKernel.argtypes = [
        CUfunction,           # f
        c_uint, c_uint, c_uint,  # gridDimX/Y/Z
        c_uint, c_uint, c_uint,  # blockDimX/Y/Z
        c_uint,               # sharedMemBytes
        CUstream,             # hStream
        c_void_p,             # kernelParams (void**)
    ]
    lib.cuLaunchCooperativeKernel.restype = CUresult

    # -- cuOccupancyMaxActiveBlocksPerMultiprocessor --------------------------
    lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        POINTER(c_int), CUfunction, c_int, c_size_t
    ]
    lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = CUresult

    return lib


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class CUDADriver:
    """Thin wrapper around the CUDA driver API.

    Manages one device and one primary context.  All memory helpers operate
    on raw CUdeviceptr values so callers stay in control of layout.
    """

    def __init__(self, device_ordinal: int = 0) -> None:
        self._lib = _load_libcuda()
        _check("cuInit", self._lib.cuInit(0))

        # Device -----------------------------------------------------------
        self._device = CUdevice()
        _check("cuDeviceGet", self._lib.cuDeviceGet(byref(self._device), device_ordinal))

        name_buf = ctypes.create_string_buffer(256)
        _check("cuDeviceGetName", self._lib.cuDeviceGetName(name_buf, 256, self._device))
        self.device_name: str = name_buf.value.decode("utf-8")

        # Context ----------------------------------------------------------
        self._context = CUcontext()
        _check(
            "cuCtxCreate_v2",
            self._lib.cuCtxCreate_v2(byref(self._context), 0, self._device),
        )

        # Track allocations for cleanup
        self._allocs: list[CUdeviceptr] = []
        self._modules: list[CUmodule] = []

    # -- Device attributes ---------------------------------------------------
    def device_attribute(self, attr: int) -> int:
        """Query a CU_DEVICE_ATTRIBUTE_* value."""
        val = c_int()
        _check(
            "cuDeviceGetAttribute",
            self._lib.cuDeviceGetAttribute(byref(val), attr, self._device),
        )
        return val.value

    # -- Module / function ---------------------------------------------------
    def load_cubin(self, path: str | Path) -> CUmodule:
        """Load a cubin file and return the module handle."""
        data = Path(path).read_bytes()
        mod = CUmodule()
        _check("cuModuleLoadData", self._lib.cuModuleLoadData(byref(mod), data))
        self._modules.append(mod)
        return mod

    def load_cubin_bytes(self, data: bytes) -> CUmodule:
        """Load cubin from raw bytes and return the module handle."""
        mod = CUmodule()
        _check("cuModuleLoadData", self._lib.cuModuleLoadData(byref(mod), data))
        self._modules.append(mod)
        return mod

    def get_function(self, module: CUmodule, name: str) -> CUfunction:
        """Get a kernel function handle from a loaded module."""
        func = CUfunction()
        _check(
            "cuModuleGetFunction",
            self._lib.cuModuleGetFunction(byref(func), module, name.encode("utf-8")),
        )
        return func

    # -- Memory management ---------------------------------------------------
    def mem_alloc(self, nbytes: int) -> CUdeviceptr:
        """Allocate *nbytes* on the device. Returns a CUdeviceptr (uint64)."""
        dptr = CUdeviceptr()
        _check("cuMemAlloc_v2", self._lib.cuMemAlloc_v2(byref(dptr), c_size_t(nbytes)))
        self._allocs.append(dptr)
        return dptr

    def mem_free(self, dptr: CUdeviceptr) -> None:
        """Free a device allocation."""
        _check("cuMemFree_v2", self._lib.cuMemFree_v2(dptr))
        try:
            self._allocs.remove(dptr)
        except ValueError:
            pass

    def memcpy_htod(self, dst: CUdeviceptr, src: Any, nbytes: int) -> None:
        """Copy *nbytes* from host buffer *src* to device *dst*."""
        _check(
            "cuMemcpyHtoD_v2",
            self._lib.cuMemcpyHtoD_v2(dst, src, c_size_t(nbytes)),
        )

    def memcpy_dtoh(self, dst: Any, src: CUdeviceptr, nbytes: int) -> None:
        """Copy *nbytes* from device *src* to host buffer *dst*."""
        _check(
            "cuMemcpyDtoH_v2",
            self._lib.cuMemcpyDtoH_v2(dst, src, c_size_t(nbytes)),
        )

    def memcpy_htod_async(self, dst: CUdeviceptr, src: Any, nbytes: int, stream: CUstream) -> None:
        """Async copy *nbytes* from host *src* to device *dst* on *stream*."""
        _check(
            "cuMemcpyHtoDAsync_v2",
            self._lib.cuMemcpyHtoDAsync_v2(dst, src, c_size_t(nbytes), stream),
        )

    def memset_d8(self, dst: CUdeviceptr, value: int, nbytes: int, stream: Any = None) -> None:
        """Set *nbytes* at device address *dst* to *value* (0-255).
        If *stream* is provided, the operation is asynchronous on that stream.
        """
        if stream is not None:
            _check(
                "cuMemsetD8Async",
                self._lib.cuMemsetD8Async(dst, ctypes.c_ubyte(value), c_size_t(nbytes), stream),
            )
        else:
            _check(
                "cuMemsetD8_v2",
                self._lib.cuMemsetD8_v2(dst, ctypes.c_ubyte(value), c_size_t(nbytes)),
            )

    # -- Streams -------------------------------------------------------------
    def stream_create(self, flags: int = 0) -> CUstream:
        """Create a CUDA stream."""
        stream = CUstream()
        _check("cuStreamCreate", self._lib.cuStreamCreate(byref(stream), c_uint(flags)))
        return stream

    def stream_synchronize(self, stream: CUstream) -> None:
        """Block until all operations on *stream* complete."""
        _check("cuStreamSynchronize", self._lib.cuStreamSynchronize(stream))

    def synchronize(self) -> None:
        """Block until all work on the current context completes."""
        _check("cuCtxSynchronize", self._lib.cuCtxSynchronize())

    def set_max_dynamic_shared(self, func: CUfunction, size: int) -> None:
        """Set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for a kernel."""
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
        _check(
            "cuFuncSetAttribute",
            self._lib.cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size
            ),
        )

    # -- Kernel launch -------------------------------------------------------
    def launch(
        self,
        func: CUfunction,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        shared_mem: int = 0,
        stream: CUstream | None = None,
    ) -> None:
        """Launch a kernel.

        *args* is a sequence of ctypes scalars or instances whose address can
        be taken with ``ctypes.byref()``.  Each element is passed by pointer
        through ``kernelParams``.

        For convenience, plain Python ``int`` values are converted to
        ``ctypes.c_uint32`` and plain ``float`` values to ``ctypes.c_float``.
        For device pointers (CUdeviceptr / c_uint64) pass them directly.
        """
        # Materialise a C array of void* pointing at each argument.
        n = len(args)
        packed: list[Any] = []
        for a in args:
            if isinstance(a, int):
                packed.append(ctypes.c_uint32(a))
            elif isinstance(a, float):
                packed.append(ctypes.c_float(a))
            elif isinstance(a, ctypes.c_uint64):
                packed.append(a)
            elif isinstance(a, ctypes.c_uint32):
                packed.append(a)
            elif isinstance(a, ctypes.c_float):
                packed.append(a)
            elif isinstance(a, ctypes.c_int32):
                packed.append(a)
            else:
                # Assume it's already a ctypes type with addressable storage
                packed.append(a)

        param_ptrs = (c_void_p * n)()
        for i, p in enumerate(packed):
            param_ptrs[i] = ctypes.cast(byref(p), c_void_p).value

        _check(
            "cuLaunchKernel",
            self._lib.cuLaunchKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem,
                stream,             # NULL → default stream
                param_ptrs,
                None,               # extra
            ),
        )

    # -- Cooperative kernel launch -------------------------------------------
    def launch_cooperative(
        self,
        func: CUfunction,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        shared_mem: int = 0,
        stream: CUstream | None = None,
    ) -> None:
        """Launch a cooperative kernel (requires cuLaunchCooperativeKernel).

        All blocks must be resident simultaneously.  The grid size must not
        exceed the occupancy limit for the kernel.
        """
        n = len(args)
        packed: list[Any] = []
        for a in args:
            if isinstance(a, int):
                packed.append(ctypes.c_uint32(a))
            elif isinstance(a, float):
                packed.append(ctypes.c_float(a))
            elif isinstance(a, ctypes.c_uint64):
                packed.append(a)
            elif isinstance(a, ctypes.c_uint32):
                packed.append(a)
            elif isinstance(a, ctypes.c_float):
                packed.append(a)
            elif isinstance(a, ctypes.c_int32):
                packed.append(a)
            else:
                packed.append(a)

        param_ptrs = (c_void_p * n)()
        for i, p in enumerate(packed):
            param_ptrs[i] = ctypes.cast(byref(p), c_void_p).value

        _check(
            "cuLaunchCooperativeKernel",
            self._lib.cuLaunchCooperativeKernel(
                func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem,
                stream,
                param_ptrs,
            ),
        )

    def max_active_blocks(self, func: CUfunction, block_size: int, shared_mem: int = 0) -> int:
        """Query max active blocks per SM for a cooperative launch."""
        num_blocks = c_int()
        _check(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            self._lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                byref(num_blocks), func, block_size, c_size_t(shared_mem)
            ),
        )
        return num_blocks.value

    # -- Cleanup -------------------------------------------------------------
    def close(self) -> None:
        """Release context and all tracked allocations."""
        for dptr in self._allocs:
            self._lib.cuMemFree_v2(dptr)
        self._allocs.clear()
        if self._context:
            self._lib.cuCtxDestroy_v2(self._context)
            self._context = CUcontext()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"CUDADriver(device={self.device_name!r})"


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import array
    import struct

    print("=== Lithos CUDA Driver self-test ===\n")

    gpu = CUDADriver()
    print(f"Device : {gpu.device_name}")
    print(f"Driver : {gpu!r}")

    # -- Load embed.cubin ----------------------------------------------------
    cubin_path = Path(__file__).resolve().parent.parent / "kernels" / "embed.cubin"
    print(f"\nLoading cubin: {cubin_path}")
    module = gpu.load_cubin(cubin_path)
    print(f"Module handle: {module}")

    func = gpu.get_function(module, "embed")
    print(f"Function 'embed' handle: {func}")

    # -- Set up test data ----------------------------------------------------
    # embed kernel signature: (u32 token_id, u64 embed_table_ptr, u64 output_ptr, u32 hidden_dim)
    # We'll create a small embedding table: 4 tokens x 16 dims (f32)
    hidden_dim = 16
    num_tokens = 4
    table_size = num_tokens * hidden_dim  # in floats
    table_bytes = table_size * 4          # in bytes

    # Fill embedding table: token i, dim j → value i * 100 + j
    host_table = array.array("f", [float(t * 100 + d) for t in range(num_tokens) for d in range(hidden_dim)])
    host_output = array.array("f", [0.0] * hidden_dim)

    # Allocate device memory
    d_table = gpu.mem_alloc(table_bytes)
    d_output = gpu.mem_alloc(hidden_dim * 4)
    print(f"\nDevice table  @ 0x{d_table.value:016x}")
    print(f"Device output @ 0x{d_output.value:016x}")

    # Upload embedding table
    table_buf, _ = host_table.buffer_info()
    gpu.memcpy_htod(d_table, ctypes.c_void_p(table_buf), table_bytes)

    # Launch kernel: look up token 2
    token_id = ctypes.c_uint32(2)
    table_ptr = ctypes.c_uint64(d_table.value)
    output_ptr = ctypes.c_uint64(d_output.value)
    dim_arg = ctypes.c_uint32(hidden_dim)

    print(f"\nLaunching embed kernel (token_id=2, hidden_dim={hidden_dim})...")
    gpu.launch(
        func,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=[token_id, table_ptr, output_ptr, dim_arg],
    )
    gpu.synchronize()
    print("Kernel completed.")

    # Read back results
    out_buf, _ = host_output.buffer_info()
    gpu.memcpy_dtoh(ctypes.c_void_p(out_buf), d_output, hidden_dim * 4)

    expected = [float(2 * 100 + d) for d in range(hidden_dim)]
    actual = list(host_output)
    print(f"\nExpected: {expected}")
    print(f"Actual  : {actual}")

    if actual == expected:
        print("\n*** PASS — embed kernel produced correct output ***")
    else:
        print("\n*** FAIL — output mismatch ***")
        raise SystemExit(1)

    # -- Cleanup -------------------------------------------------------------
    gpu.mem_free(d_table)
    gpu.mem_free(d_output)
    gpu.close()
    print("\nCleanup done. All tests passed.")
