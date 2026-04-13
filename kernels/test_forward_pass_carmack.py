#!/usr/bin/env python3
"""
Test + benchmark: forward_pass_carmack.ptx vs forward_pass_multi.ptx
Validates correctness and measures wall-clock kernel time.
"""

import ctypes
import os
import struct
import math
from ctypes import (
    POINTER, byref, c_float, c_int, c_size_t,
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


def main():
    HIDDEN = 5120
    INTERMEDIATE = 17408
    NUM_LAYERS = 64
    VOCAB_SIZE = 248320
    GROUP_SIZE = 128
    NUM_BLOCKS = 132
    THREADS = 256

    K_packed_gate = HIDDEN // 8      # 640
    N_gate = INTERMEDIATE            # 17408
    n_groups_gate = HIDDEN // GROUP_SIZE  # 40

    K_packed_down = INTERMEDIATE // 8    # 2176
    N_down = HIDDEN                      # 5120
    n_groups_down = INTERMEDIATE // GROUP_SIZE  # 136

    print("=" * 70)
    print("Forward Pass Kernel Benchmark: Carmack vs Original")
    print("=" * 70)

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [c_uint]; cuda.cuInit.restype = CUresult
    cuda.cuDeviceGet.argtypes = [POINTER(CUdevice), c_int]; cuda.cuDeviceGet.restype = CUresult
    cuda.cuCtxCreate_v2.argtypes = [POINTER(CUcontext), c_uint, CUdevice]; cuda.cuCtxCreate_v2.restype = CUresult
    cuda.cuCtxDestroy_v2.argtypes = [CUcontext]; cuda.cuCtxDestroy_v2.restype = CUresult
    cuda.cuCtxSynchronize.argtypes = []; cuda.cuCtxSynchronize.restype = CUresult
    cuda.cuModuleLoadData.argtypes = [POINTER(CUmodule), c_void_p]; cuda.cuModuleLoadData.restype = CUresult
    cuda.cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, c_void_p]; cuda.cuModuleGetFunction.restype = CUresult
    cuda.cuModuleUnload.argtypes = [CUmodule]; cuda.cuModuleUnload.restype = CUresult
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
    cuda.cuFuncGetAttribute.argtypes = [POINTER(c_int), c_int, CUfunction]; cuda.cuFuncGetAttribute.restype = CUresult
    cuda.cuLaunchCooperativeKernel.argtypes = [
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, c_void_p,
    ]
    cuda.cuLaunchCooperativeKernel.restype = CUresult
    # cuMemGetInfo
    cuda.cuMemGetInfo_v2.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
    cuda.cuMemGetInfo_v2.restype = CUresult

    _check("cuInit", cuda.cuInit(0))
    dev = CUdevice()
    _check("cuDeviceGet", cuda.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext()
    _check("cuCtxCreate", cuda.cuCtxCreate_v2(byref(ctx), 0, dev))

    free_mem = c_size_t()
    total_mem = c_size_t()
    cuda.cuMemGetInfo_v2(byref(free_mem), byref(total_mem))
    print(f"  GPU memory: {free_mem.value/1e9:.2f} GB free / {total_mem.value/1e9:.2f} GB total")

    def dev_alloc(nbytes):
        d = CUdeviceptr()
        _check("cuMemAlloc", cuda.cuMemAlloc_v2(byref(d), c_size_t(max(nbytes, 4))))
        return d

    def upload(d, buf):
        b = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
        _check("HtoD", cuda.cuMemcpyHtoD_v2(d, ctypes.cast(b, c_void_p), c_size_t(len(buf))))

    # =========================================================================
    # Allocate all weight data on GPU using memset (fast, no CPU memory needed)
    # Both kernels see identical data, so correctness comparison is valid.
    # =========================================================================
    print("Allocating GPU weight data...")

    # Embedding: 5120 f32
    embed_size = HIDDEN * 4
    d_embed = dev_alloc(embed_size)
    # Fill with small values via host upload
    embed_data = bytearray(embed_size)
    for i in range(HIDDEN):
        struct.pack_into('<f', embed_data, i * 4, 0.01 * ((i % 100) - 50) / 50.0)
    upload(d_embed, embed_data)

    # Norm weights: fill with 1.0
    norm_size = HIDDEN * 4
    norm_data = bytearray(norm_size)
    for i in range(HIDDEN):
        struct.pack_into('<f', norm_data, i * 4, 1.0)
    d_norm_w = dev_alloc(norm_size); upload(d_norm_w, norm_data)
    d_final_norm = dev_alloc(norm_size); upload(d_final_norm, norm_data)

    # MLP weights -- allocate on GPU, fill with memset pattern
    gate_qw_size = K_packed_gate * N_gate * 4   # ~44.6MB
    gate_sc_size = n_groups_gate * N_gate * 2    # ~1.4MB
    gate_zr_size = n_groups_gate * N_gate * 4    # ~2.8MB

    down_qw_size = K_packed_down * N_down * 4    # ~44.6MB
    down_sc_size = n_groups_down * N_down * 2    # ~1.4MB
    down_zr_size = n_groups_down * N_down * 4    # ~2.8MB

    lm_qw_size = K_packed_gate * VOCAB_SIZE * 4  # ~635.7MB
    lm_sc_size = n_groups_gate * VOCAB_SIZE * 2  # ~19.9MB
    lm_zr_size = n_groups_gate * VOCAB_SIZE * 4  # ~39.7MB

    alloc_list = []

    def tracked_alloc(nbytes, name, fill=0x55):
        d = dev_alloc(nbytes)
        alloc_list.append(d)
        _check(f"memset {name}", cuda.cuMemsetD8_v2(d, fill, c_size_t(nbytes)))
        return d

    # Allocate MLP weights -- share gate/up to save memory (same format, same K)
    # We need to save every MB for the LM head
    d_gate_qw = tracked_alloc(gate_qw_size, "gate_qw")
    d_gate_sc = tracked_alloc(gate_sc_size, "gate_sc", 0x38)
    d_gate_zr = tracked_alloc(gate_zr_size, "gate_zr", 0x08)
    # Up shares with gate (both are [17408, 640] packed)
    d_up_qw = d_gate_qw
    d_up_sc = d_gate_sc
    d_up_zr = d_gate_zr
    # Down: different dimensions, allocate separately
    d_down_qw = tracked_alloc(down_qw_size, "down_qw")
    d_down_sc = tracked_alloc(down_sc_size, "down_sc", 0x38)
    d_down_zr = tracked_alloc(down_zr_size, "down_zr", 0x08)

    cuda.cuMemGetInfo_v2(byref(free_mem), byref(total_mem))
    print(f"  After MLP weights: {free_mem.value/1e9:.2f} GB free")

    # LM head -- this is the big one
    lm_total = lm_qw_size + lm_sc_size + lm_zr_size
    print(f"  LM head needs: {lm_total/1e9:.2f} GB")

    if free_mem.value > lm_total + 100*1024*1024:  # need 100MB headroom
        d_lm_qw = tracked_alloc(lm_qw_size, "lm_qw")
        d_lm_sc = tracked_alloc(lm_sc_size, "lm_sc", 0x38)
        d_lm_zr = tracked_alloc(lm_zr_size, "lm_zr", 0x08)
        print("  LM head: fully allocated")
    else:
        # Not enough memory -- allocate what we can
        # Use a smaller effective vocab that fits
        avail = free_mem.value - 100*1024*1024
        bytes_per_row = K_packed_gate * 4 + n_groups_gate * 2 + n_groups_gate * 4
        max_rows = avail // bytes_per_row
        eff_vocab = min(VOCAB_SIZE, max_rows)
        print(f"  LM head: only {avail/1e9:.2f}GB available, allocating {eff_vocab} of {VOCAB_SIZE} rows")
        # Allocate for eff_vocab rows -- kernel will read OOB for higher rows
        # but since both kernels do the same, comparison is valid
        d_lm_qw = tracked_alloc(K_packed_gate * eff_vocab * 4, "lm_qw")
        d_lm_sc = tracked_alloc(n_groups_gate * eff_vocab * 2, "lm_sc", 0x38)
        d_lm_zr = tracked_alloc(n_groups_gate * eff_vocab * 4, "lm_zr", 0x08)

    cuda.cuMemGetInfo_v2(byref(free_mem), byref(total_mem))
    print(f"  Final: {free_mem.value/1e6:.0f} MB free")

    # Weight descriptor table
    desc_size = NUM_LAYERS * 88
    desc_data = bytearray(desc_size)
    for layer in range(NUM_LAYERS):
        off = layer * 88
        struct.pack_into('<Q', desc_data, off + 0, d_norm_w.value)
        struct.pack_into('<Q', desc_data, off + 8, d_gate_qw.value)
        struct.pack_into('<Q', desc_data, off + 16, d_gate_sc.value)
        struct.pack_into('<Q', desc_data, off + 24, d_gate_zr.value)
        struct.pack_into('<Q', desc_data, off + 32, d_up_qw.value)
        struct.pack_into('<Q', desc_data, off + 40, d_up_sc.value)
        struct.pack_into('<Q', desc_data, off + 48, d_up_zr.value)
        struct.pack_into('<Q', desc_data, off + 56, d_down_qw.value)
        struct.pack_into('<Q', desc_data, off + 64, d_down_sc.value)
        struct.pack_into('<Q', desc_data, off + 72, d_down_zr.value)
        struct.pack_into('<Q', desc_data, off + 80, d_norm_w.value)
    d_desc = dev_alloc(desc_size); upload(d_desc, desc_data)
    alloc_list.append(d_desc)

    # Scratch buffers (2 copies for ref and new)
    d_act_ref = dev_alloc(HIDDEN * 4); alloc_list.append(d_act_ref)
    d_act_new = dev_alloc(HIDDEN * 4); alloc_list.append(d_act_new)
    d_normed_ref = dev_alloc(HIDDEN * 4); alloc_list.append(d_normed_ref)
    d_normed_new = dev_alloc(HIDDEN * 4); alloc_list.append(d_normed_new)
    d_mid_ref = dev_alloc(INTERMEDIATE * 4); alloc_list.append(d_mid_ref)
    d_mid_new = dev_alloc(INTERMEDIATE * 4); alloc_list.append(d_mid_new)
    d_logits_ref = dev_alloc(VOCAB_SIZE * 4); alloc_list.append(d_logits_ref)
    d_logits_new = dev_alloc(VOCAB_SIZE * 4); alloc_list.append(d_logits_new)
    d_sync_ref = dev_alloc(16); alloc_list.append(d_sync_ref)
    d_sync_new = dev_alloc(16); alloc_list.append(d_sync_new)
    d_part_ref = dev_alloc(NUM_BLOCKS * 4); alloc_list.append(d_part_ref)
    d_part_new = dev_alloc(NUM_BLOCKS * 4); alloc_list.append(d_part_new)

    # =========================================================================
    # Load kernels
    # =========================================================================
    print("\nLoading kernels...")

    ref_mod = CUmodule()
    _check("load ref", cuda.cuModuleLoadData(byref(ref_mod),
           open(os.path.join(KERNEL_DIR, "forward_pass_multi.cubin"), "rb").read()))
    ref_func = CUfunction()
    _check("getfunc ref", cuda.cuModuleGetFunction(byref(ref_func), ref_mod, b"forward_pass_multi"))

    new_mod = CUmodule()
    _check("load new", cuda.cuModuleLoadData(byref(new_mod),
           open(os.path.join(KERNEL_DIR, "forward_pass_carmack.cubin"), "rb").read()))
    new_func = CUfunction()
    _check("getfunc new", cuda.cuModuleGetFunction(byref(new_func), new_mod, b"forward_pass_carmack"))

    for name, func in [("Original", ref_func), ("Carmack", new_func)]:
        nregs = c_int()
        cuda.cuFuncGetAttribute(byref(nregs), 4, func)
        smem = c_int()
        cuda.cuFuncGetAttribute(byref(smem), 1, func)
        print(f"  {name}: {nregs.value} regs, {smem.value} bytes smem")

    # =========================================================================
    # Build kernel args
    # =========================================================================
    def make_args(d_act, d_normed, d_mid, d_logits, d_sync, d_part):
        args = [
            c_uint64(d_desc.value),
            c_uint64(d_embed.value),
            c_uint32(0),
            c_uint64(d_mid.value),
            c_uint64(d_logits.value),
            c_uint64(d_final_norm.value),
            c_uint64(d_lm_qw.value),
            c_uint64(d_lm_sc.value),
            c_uint64(d_lm_zr.value),
            c_uint64(d_act.value),
            c_uint64(d_normed.value),
            c_uint64(d_sync.value),
            c_uint64(d_part.value),
        ]
        ptrs = (c_void_p * len(args))()
        for i, a in enumerate(args):
            ptrs[i] = ctypes.cast(byref(a), c_void_p).value
        return args, ptrs

    def reset_buffers(d_act, d_mid, d_logits, d_sync, d_normed, d_part):
        cuda.cuMemsetD8_v2(d_sync, 0, c_size_t(16))
        cuda.cuMemsetD8_v2(d_act, 0, c_size_t(HIDDEN * 4))
        cuda.cuMemsetD8_v2(d_mid, 0, c_size_t(INTERMEDIATE * 4))
        cuda.cuMemsetD8_v2(d_logits, 0, c_size_t(VOCAB_SIZE * 4))
        cuda.cuMemsetD8_v2(d_normed, 0, c_size_t(HIDDEN * 4))
        cuda.cuMemsetD8_v2(d_part, 0, c_size_t(NUM_BLOCKS * 4))

    ref_args, ref_ptrs = make_args(d_act_ref, d_normed_ref, d_mid_ref, d_logits_ref, d_sync_ref, d_part_ref)
    new_args, new_ptrs = make_args(d_act_new, d_normed_new, d_mid_new, d_logits_new, d_sync_new, d_part_new)

    # =========================================================================
    # CORRECTNESS
    # =========================================================================
    print("\nCorrectness test...")
    reset_buffers(d_act_ref, d_mid_ref, d_logits_ref, d_sync_ref, d_normed_ref, d_part_ref)
    reset_buffers(d_act_new, d_mid_new, d_logits_new, d_sync_new, d_normed_new, d_part_new)

    _check("launch ref", cuda.cuLaunchCooperativeKernel(
        ref_func, NUM_BLOCKS, 1, 1, THREADS, 1, 1, 0, None, ref_ptrs))
    _check("sync ref", cuda.cuCtxSynchronize())

    _check("launch new", cuda.cuLaunchCooperativeKernel(
        new_func, NUM_BLOCKS, 1, 1, THREADS, 1, 1, 0, None, new_ptrs))
    _check("sync new", cuda.cuCtxSynchronize())

    # Compare logits
    CHECK_N = 1000
    out_ref = bytearray(CHECK_N * 4)
    out_new = bytearray(CHECK_N * 4)
    _check("DtoH ref", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (CHECK_N*4)).from_buffer(out_ref), c_void_p),
        d_logits_ref, c_size_t(CHECK_N * 4)))
    _check("DtoH new", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (CHECK_N*4)).from_buffer(out_new), c_void_p),
        d_logits_new, c_size_t(CHECK_N * 4)))

    n_wrong = 0
    max_abs = 0.0
    max_rel = 0.0
    n_nonzero = 0
    for i in range(CHECK_N):
        rv = struct.unpack_from('<f', out_ref, i * 4)[0]
        nv = struct.unpack_from('<f', out_new, i * 4)[0]
        if math.isnan(rv) or math.isnan(nv):
            if math.isnan(rv) != math.isnan(nv):
                n_wrong += 1
            continue
        if rv != 0.0:
            n_nonzero += 1
        ae = abs(rv - nv)
        max_abs = max(max_abs, ae)
        denom = max(abs(rv), 1e-8)
        re = ae / denom
        max_rel = max(max_rel, re)
        if re > 0.05:
            n_wrong += 1

    # Print a few values regardless
    print(f"  Sample logits (first 5):")
    for i in range(5):
        rv = struct.unpack_from('<f', out_ref, i * 4)[0]
        nv = struct.unpack_from('<f', out_new, i * 4)[0]
        print(f"    [{i}] ref={rv:.8e} new={nv:.8e}")

    if n_wrong == 0:
        print(f"  PASS: {CHECK_N} logits (max_abs={max_abs:.6e}, max_rel={max_rel:.4f}, nonzero={n_nonzero})")
    else:
        print(f"  FAIL: {n_wrong}/{CHECK_N} differ >5%")
        for i in range(min(20, CHECK_N)):
            rv = struct.unpack_from('<f', out_ref, i * 4)[0]
            nv = struct.unpack_from('<f', out_new, i * 4)[0]
            print(f"    [{i}] ref={rv:.8f} new={nv:.8f}")

    # Also compare mid_scratch (gate+up output) to isolate where differences come from
    mid_check = 100
    mid_ref_buf = bytearray(mid_check * 4)
    mid_new_buf = bytearray(mid_check * 4)
    _check("DtoH", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (mid_check*4)).from_buffer(mid_ref_buf), c_void_p),
        d_mid_ref, c_size_t(mid_check * 4)))
    _check("DtoH", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (mid_check*4)).from_buffer(mid_new_buf), c_void_p),
        d_mid_new, c_size_t(mid_check * 4)))
    mid_wrong = 0
    for i in range(mid_check):
        rv = struct.unpack_from('<f', mid_ref_buf, i * 4)[0]
        nv = struct.unpack_from('<f', mid_new_buf, i * 4)[0]
        ae = abs(rv - nv)
        denom = max(abs(rv), 1e-8)
        if ae / denom > 0.05:
            mid_wrong += 1
    print(f"  Mid scratch check: {mid_wrong}/{mid_check} differ >5%")

    # Compare activation (residual stream)
    act_check = 100
    act_ref_buf = bytearray(act_check * 4)
    act_new_buf = bytearray(act_check * 4)
    _check("DtoH", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (act_check*4)).from_buffer(act_ref_buf), c_void_p),
        d_act_ref, c_size_t(act_check * 4)))
    _check("DtoH", cuda.cuMemcpyDtoH_v2(
        ctypes.cast((ctypes.c_ubyte * (act_check*4)).from_buffer(act_new_buf), c_void_p),
        d_act_new, c_size_t(act_check * 4)))
    act_wrong = 0
    for i in range(act_check):
        rv = struct.unpack_from('<f', act_ref_buf, i * 4)[0]
        nv = struct.unpack_from('<f', act_new_buf, i * 4)[0]
        ae = abs(rv - nv)
        denom = max(abs(rv), 1e-8)
        if ae / denom > 0.05:
            act_wrong += 1
    print(f"  Activation check: {act_wrong}/{act_check} differ >5%")

    # =========================================================================
    # BENCHMARK
    # =========================================================================
    print("\nBenchmark (kernel wall-clock time):")

    ev_start = CUevent()
    ev_stop = CUevent()
    _check("evCreate", cuda.cuEventCreate(byref(ev_start), 0))
    _check("evCreate", cuda.cuEventCreate(byref(ev_stop), 0))

    WARMUP = 2
    ITERS = 5

    for label, func, ptrs, d_sync, d_act, d_mid_buf, d_log, d_nor, d_par in [
        ("Original (forward_pass_multi)", ref_func, ref_ptrs,
         d_sync_ref, d_act_ref, d_mid_ref, d_logits_ref, d_normed_ref, d_part_ref),
        ("Carmack  (forward_pass_carmack)", new_func, new_ptrs,
         d_sync_new, d_act_new, d_mid_new, d_logits_new, d_normed_new, d_part_new),
    ]:
        for _ in range(WARMUP):
            reset_buffers(d_act, d_mid_buf, d_log, d_sync, d_nor, d_par)
            cuda.cuLaunchCooperativeKernel(func, NUM_BLOCKS, 1, 1, THREADS, 1, 1, 0, None, ptrs)
            cuda.cuCtxSynchronize()

        times = []
        for _ in range(ITERS):
            reset_buffers(d_act, d_mid_buf, d_log, d_sync, d_nor, d_par)
            _check("sync", cuda.cuCtxSynchronize())
            _check("evRec", cuda.cuEventRecord(ev_start, None))
            _check("launch", cuda.cuLaunchCooperativeKernel(
                func, NUM_BLOCKS, 1, 1, THREADS, 1, 1, 0, None, ptrs))
            _check("evRec", cuda.cuEventRecord(ev_stop, None))
            _check("evSync", cuda.cuEventSynchronize(ev_stop))
            ms = c_float()
            _check("elapsed", cuda.cuEventElapsedTime(byref(ms), ev_start, ev_stop))
            times.append(ms.value)

        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"  {label}:")
        print(f"    avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms  ({ITERS} runs)")

    print()
    print("=" * 70)

    # Cleanup
    cuda.cuEventDestroy_v2(ev_start)
    cuda.cuEventDestroy_v2(ev_stop)
    for d in alloc_list:
        cuda.cuMemFree_v2(d)
    cuda.cuMemFree_v2(d_embed)
    cuda.cuMemFree_v2(d_norm_w)
    cuda.cuMemFree_v2(d_final_norm)
    cuda.cuModuleUnload(ref_mod)
    cuda.cuModuleUnload(new_mod)
    cuda.cuCtxDestroy_v2(ctx)


if __name__ == "__main__":
    main()
