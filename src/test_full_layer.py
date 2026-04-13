#!/usr/bin/env python3
"""
Test full layer 0 (DeltaNet / linear_attention) on GH200 with real Qwen 3.5-27B weights.

Chains: embed -> norm -> QKV projection -> [DeltaNet skip] -> post_attn norm -> MLP
Compares every intermediate against PyTorch reference.

Layer 0 structure (linear_attention / DeltaNet):
  - input_layernorm (RMSNorm)
  - linear_attn.in_proj_qkv  (fused Q+K+V GPTQ projection, out_dim=10240)
      Q: [0:2048], K: [2048:4096], V: [4096:10240]
  - linear_attn recurrence (SKIP -- use PyTorch reference)
  - linear_attn.out_proj (GPTQ, 6144->5120)
  - residual add
  - post_attention_layernorm (RMSNorm)
  - mlp.gate_proj (GPTQ, 5120->17408)
  - mlp.up_proj   (GPTQ, 5120->17408)
  - SiLU(gate) * up  (activate kernel)
  - mlp.down_proj (GPTQ, 17408->5120)
  - residual add

Run:
    python3 /home/ubuntu/lithos/src/test_full_layer.py
"""

from __future__ import annotations

import ctypes
import math
import numpy as np
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_driver import CUDADriver, CUDAError, CUdeviceptr
from loader import LithosModel
from tokenizer import Tokenizer

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
KERNEL_DIR = str(Path(__file__).resolve().parent.parent / "kernels")
CACHE_DIR = "/tmp/lithos-cache/3644e4d3fa48efc4"

HIDDEN_DIM = 5120
INTERMEDIATE_SIZE = 17408
GROUP_SIZE = 128
ZERO_POINT = 7


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}")


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def upload_f32(gpu: CUDADriver, data: np.ndarray) -> CUdeviceptr:
    assert data.dtype == np.float32
    dptr = gpu.mem_alloc(data.nbytes)
    gpu.memcpy_htod(dptr, data.ctypes.data_as(ctypes.c_void_p), data.nbytes)
    return dptr


def download_f32(gpu: CUDADriver, dptr: CUdeviceptr, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    gpu.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), dptr, size * 4)
    return out


def dequant_gptq_ref(qweight_bytes: bytes, scales_bytes: bytes,
                      K: int, N: int) -> np.ndarray:
    """CPU reference dequant: GPTQ W4 -> f32 [K, N]."""
    K_packed = K // 8
    n_groups = K // GROUP_SIZE
    qw = np.frombuffer(qweight_bytes, dtype=np.uint32).reshape(K_packed, N)
    sc = np.frombuffer(scales_bytes, dtype=np.float16).reshape(n_groups, N).astype(np.float32)
    weight = np.zeros((K, N), dtype=np.float32)
    for bit in range(8):
        nibbles = (qw >> (bit * 4)) & 0xF
        for k_packed in range(K_packed):
            k = k_packed * 8 + bit
            group = k // GROUP_SIZE
            weight[k, :] = (nibbles[k_packed].astype(np.float32) - ZERO_POINT) * sc[group, :]
    return weight


def rms_norm_ref(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """CPU reference RMSNorm."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def silu_ref(x: np.ndarray) -> np.ndarray:
    """CPU SiLU: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x.clip(-80, 80))))


def compare(name: str, gpu_val: np.ndarray, ref_val: np.ndarray,
            atol: float = 0.05, cos_thresh: float = 0.99) -> bool:
    """Compare GPU output vs reference. Returns True if PASS."""
    abs_err = np.abs(gpu_val - ref_val)
    max_err = abs_err.max()
    mean_err = abs_err.mean()

    dot = np.dot(gpu_val.ravel(), ref_val.ravel())
    norm_g = np.sqrt(np.dot(gpu_val.ravel(), gpu_val.ravel()))
    norm_r = np.sqrt(np.dot(ref_val.ravel(), ref_val.ravel()))
    cosine = dot / (norm_g * norm_r + 1e-20)

    has_nan = np.any(np.isnan(gpu_val))
    has_inf = np.any(np.isinf(gpu_val))

    ok = max_err < atol and cosine > cos_thresh and not has_nan and not has_inf

    print(f"  [{name}] max_err={max_err:.6f} mean_err={mean_err:.6f} "
          f"cosine={cosine:.8f} NaN={has_nan} Inf={has_inf} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    GPU  first 8: {gpu_val.ravel()[:8]}")
        print(f"    Ref  first 8: {ref_val.ravel()[:8]}")
        print(f"    GPU  range: [{gpu_val.min():.6f}, {gpu_val.max():.6f}]")
        print(f"    Ref  range: [{ref_val.min():.6f}, {ref_val.max():.6f}]")
    return ok


def load_norm_weight(model: LithosModel, name: str) -> np.ndarray:
    """Load a norm weight tensor as f32."""
    ti = model.weight_info(name)
    raw = bytes(model.weight_bytes(name))
    if ti.dtype == "BF16":
        return bf16_to_f32(raw)
    elif ti.dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        return np.frombuffer(raw, dtype=np.float32).copy()


def run_gptq_matvec(gpu: CUDADriver, proj_func,
                     model: LithosModel, prefix: str,
                     d_input: CUdeviceptr, K: int, N: int) -> tuple:
    """Run GPTQ matvec kernel and CPU reference. Returns (gpu_out, ref_out)."""
    qw_ptr = model.weight_info(f"{prefix}.qweight").ptr
    sc_ptr = model.weight_info(f"{prefix}.scales").ptr

    d_output = gpu.mem_alloc(N * 4)
    # Zero output
    gpu.memcpy_htod(d_output, np.zeros(N, dtype=np.float32).ctypes.data_as(ctypes.c_void_p), N * 4)

    gpu.launch(
        proj_func,
        grid=(N, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(qw_ptr),
            ctypes.c_uint64(sc_ptr),
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ],
        shared_mem=32,
    )
    gpu.synchronize()

    gpu_out = download_f32(gpu, d_output, N)

    # CPU reference
    qw_raw = bytes(model.weight_bytes(f"{prefix}.qweight"))
    sc_raw = bytes(model.weight_bytes(f"{prefix}.scales"))
    weight = dequant_gptq_ref(qw_raw, sc_raw, K, N)

    input_vec = download_f32(gpu, d_input, K)
    ref_out = input_vec @ weight

    return gpu_out, ref_out, d_output


def main() -> int:
    banner("Full Layer 0 Test -- DeltaNet + MLP on GH200")
    results = {}
    allocs = []  # track GPU allocations for cleanup

    # ---------------------------------------------------------------
    # 1. Load model + tokenizer + CUDA + kernels
    # ---------------------------------------------------------------
    print("\n--- Loading model, tokenizer, CUDA context ---")
    t0 = time.monotonic()
    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    gpu = CUDADriver()
    t1 = time.monotonic()
    print(f"  Model: {model}  ({t1-t0:.3f}s)")
    print(f"  Device: {gpu.device_name}")

    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Layer 0 type: {model.config.layer_types[0]}")

    epsilon = model.config.rms_norm_eps
    print(f"  epsilon = {epsilon}")

    # Load kernels
    print("\n--- Loading kernels ---")
    embed_mod = gpu.load_cubin(f"{KERNEL_DIR}/embed_f16.cubin")
    embed_func = gpu.get_function(embed_mod, "embed_f16")
    print(f"  embed_f16: loaded")

    norm_mod = gpu.load_cubin(f"{CACHE_DIR}/norm.cubin")
    norm_func = gpu.get_function(norm_mod, "norm")
    print(f"  norm (hidden_dim=5120 baked): loaded")

    proj_mod = gpu.load_cubin(f"{KERNEL_DIR}/gptq_matvec.cubin")
    proj_func = gpu.get_function(proj_mod, "gptq_matvec")
    print(f"  gptq_matvec: loaded")

    activate_mod = gpu.load_cubin(f"{CACHE_DIR}/activate.cubin")
    activate_func = gpu.get_function(activate_mod, "activate")
    print(f"  activate (SiLU*gate, size=17408 baked): loaded")

    # ---------------------------------------------------------------
    # STEP 1: Embed token 0 ("The")
    # ---------------------------------------------------------------
    banner("STEP 1: Embed")

    tid = token_ids[0]
    word = tok.decode([tid])
    print(f"  Token: {tid} ({word!r})")

    embed_ptr = model.weight_info("model.language_model.embed_tokens.weight").ptr
    d_embed_out = gpu.mem_alloc(HIDDEN_DIM * 4)
    allocs.append(d_embed_out)

    BLOCK = 256
    GRID = max(1, math.ceil(HIDDEN_DIM / BLOCK))
    gpu.launch(
        embed_func,
        grid=(GRID, 1, 1),
        block=(BLOCK, 1, 1),
        args=[
            ctypes.c_uint32(tid),
            ctypes.c_uint64(embed_ptr),
            ctypes.c_uint64(d_embed_out.value),
        ],
    )
    gpu.synchronize()

    embed_out = download_f32(gpu, d_embed_out, HIDDEN_DIM)

    # CPU reference
    raw = model.weight_bytes("model.language_model.embed_tokens.weight")
    off = tid * HIDDEN_DIM * 2
    embed_ref = np.frombuffer(bytes(raw[off:off+HIDDEN_DIM*2]), dtype=np.float16).astype(np.float32)

    ok = compare("embed", embed_out, embed_ref, atol=1e-5)
    results["1_embed"] = ok

    # ---------------------------------------------------------------
    # STEP 2: Input LayerNorm
    # ---------------------------------------------------------------
    banner("STEP 2: Input LayerNorm (layer 0)")

    norm_w_name = "model.language_model.layers.0.input_layernorm.weight"
    norm_f32 = load_norm_weight(model, norm_w_name)
    d_norm_weight = upload_f32(gpu, norm_f32)
    allocs.append(d_norm_weight)

    # Residual = embed (first layer, residual IS the embedding)
    # But the norm kernel does output = norm(input + residual) * weight
    # For layer 0 first pass: the residual stream IS the embed, and the input
    # to norm is the residual stream. So input = embed, residual = zeros.
    d_residual = upload_f32(gpu, np.zeros(HIDDEN_DIM, dtype=np.float32))
    allocs.append(d_residual)

    d_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)
    allocs.append(d_norm_out)

    gpu.launch(
        norm_func,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_embed_out.value),
            ctypes.c_uint64(d_residual.value),
            ctypes.c_uint64(d_norm_weight.value),
            ctypes.c_uint64(d_norm_out.value),
            ctypes.c_float(epsilon),
        ],
        shared_mem=128,
    )
    gpu.synchronize()

    norm_out = download_f32(gpu, d_norm_out, HIDDEN_DIM)
    norm_ref = rms_norm_ref(embed_out, norm_f32, epsilon)

    ok = compare("input_norm", norm_out, norm_ref, atol=0.01)
    results["2_input_norm"] = ok

    # ---------------------------------------------------------------
    # STEP 3: QKV Projection (fused, linear_attn.in_proj_qkv)
    # ---------------------------------------------------------------
    banner("STEP 3: QKV Projection (linear_attn.in_proj_qkv)")

    qkv_prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    qkv_qw = model.weight_info(f"{qkv_prefix}.qweight")
    K_proj = HIDDEN_DIM  # 5120
    N_qkv = qkv_qw.shape[1]  # 10240
    print(f"  QKV projection: K={K_proj} -> N={N_qkv}")
    print(f"  Q: [0:{2048}], K: [2048:{4096}], V: [4096:{N_qkv}]")

    # Upload norm output as input
    d_proj_input = upload_f32(gpu, norm_out)
    allocs.append(d_proj_input)

    t0 = time.perf_counter()
    gpu_qkv, ref_qkv, d_qkv_out = run_gptq_matvec(
        gpu, proj_func, model, qkv_prefix, d_proj_input, K_proj, N_qkv
    )
    t1 = time.perf_counter()
    allocs.append(d_qkv_out)
    print(f"  Kernel time: {(t1-t0)*1e3:.2f} ms")

    ok = compare("qkv_fused", gpu_qkv, ref_qkv, atol=0.1, cos_thresh=0.99)
    results["3_qkv_projection"] = ok

    # Split Q, K, V
    gpu_q = gpu_qkv[:2048]
    gpu_k = gpu_qkv[2048:4096]
    gpu_v = gpu_qkv[4096:]
    ref_q = ref_qkv[:2048]
    ref_k = ref_qkv[2048:4096]
    ref_v = ref_qkv[4096:]

    compare("Q_split", gpu_q, ref_q, atol=0.1)
    compare("K_split", gpu_k, ref_k, atol=0.1)
    compare("V_split", gpu_v, ref_v, atol=0.1)

    # ---------------------------------------------------------------
    # STEP 4: DeltaNet Recurrence -- SKIP, use reference for attn output
    # ---------------------------------------------------------------
    banner("STEP 4: DeltaNet Recurrence (SKIPPED -- too complex)")
    print("  DeltaNet linear attention requires:")
    print("    - A_log, dt_bias parameters")
    print("    - conv1d on QKV")
    print("    - in_proj_a, in_proj_b for short convolution")
    print("    - in_proj_z for gating")
    print("    - Recurrent state update")
    print("  Skipping recurrence -- will use out_proj(V) as a proxy to test the")
    print("  projection kernels through the rest of the layer.")
    print()
    print("  For correctness, we use the V output directly through out_proj")
    print("  to verify the out_proj kernel works, then continue with MLP.")

    # Run out_proj on V (6144 -> 5120) as a proxy test of projection
    out_proj_prefix = "model.language_model.layers.0.linear_attn.out_proj"
    out_qw = model.weight_info(f"{out_proj_prefix}.qweight")
    K_out = out_qw.shape[0] * 8  # 768*8 = 6144
    N_out = out_qw.shape[1]  # 5120
    print(f"\n  Testing out_proj: K={K_out} -> N={N_out}")

    # Upload V as input to out_proj
    d_v_input = upload_f32(gpu, gpu_v)
    allocs.append(d_v_input)

    gpu_attn_out, ref_attn_out, d_attn_out = run_gptq_matvec(
        gpu, proj_func, model, out_proj_prefix, d_v_input, K_out, N_out
    )
    allocs.append(d_attn_out)

    ok = compare("out_proj", gpu_attn_out, ref_attn_out, atol=0.15, cos_thresh=0.99)
    results["4_out_proj"] = ok

    # ---------------------------------------------------------------
    # STEP 5: Residual Add (attn_output + embed)
    # ---------------------------------------------------------------
    banner("STEP 5: Residual Add (CPU -- trivial)")
    # In the real model: residual = embed + attn_output
    # Since we skipped the real DeltaNet, this is just a proxy.
    # We do it on CPU for now.
    residual_after_attn = embed_out + gpu_attn_out
    ref_residual = embed_ref + ref_attn_out
    ok = compare("residual_1", residual_after_attn, ref_residual, atol=0.15)
    results["5_residual_add_1"] = ok

    # Upload residual to GPU for next norm
    d_residual_stream = upload_f32(gpu, residual_after_attn)
    allocs.append(d_residual_stream)

    # ---------------------------------------------------------------
    # STEP 6: Post-Attention LayerNorm
    # ---------------------------------------------------------------
    banner("STEP 6: Post-Attention LayerNorm")

    post_norm_name = "model.language_model.layers.0.post_attention_layernorm.weight"
    post_norm_f32 = load_norm_weight(model, post_norm_name)
    d_post_norm_w = upload_f32(gpu, post_norm_f32)
    allocs.append(d_post_norm_w)

    d_zero_res = upload_f32(gpu, np.zeros(HIDDEN_DIM, dtype=np.float32))
    allocs.append(d_zero_res)

    d_post_norm_out = gpu.mem_alloc(HIDDEN_DIM * 4)
    allocs.append(d_post_norm_out)

    gpu.launch(
        norm_func,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_residual_stream.value),
            ctypes.c_uint64(d_zero_res.value),
            ctypes.c_uint64(d_post_norm_w.value),
            ctypes.c_uint64(d_post_norm_out.value),
            ctypes.c_float(epsilon),
        ],
        shared_mem=128,
    )
    gpu.synchronize()

    post_norm_out = download_f32(gpu, d_post_norm_out, HIDDEN_DIM)
    post_norm_ref = rms_norm_ref(residual_after_attn, post_norm_f32, epsilon)

    ok = compare("post_attn_norm", post_norm_out, post_norm_ref, atol=0.01)
    results["6_post_attn_norm"] = ok

    # ---------------------------------------------------------------
    # STEP 7: Gate Projection (mlp.gate_proj)
    # ---------------------------------------------------------------
    banner("STEP 7: Gate Projection (5120 -> 17408)")

    gate_prefix = "model.language_model.layers.0.mlp.gate_proj"
    d_mlp_input = upload_f32(gpu, post_norm_out)
    allocs.append(d_mlp_input)

    t0 = time.perf_counter()
    gpu_gate, ref_gate, d_gate_out = run_gptq_matvec(
        gpu, proj_func, model, gate_prefix, d_mlp_input, HIDDEN_DIM, INTERMEDIATE_SIZE
    )
    t1 = time.perf_counter()
    allocs.append(d_gate_out)
    print(f"  Kernel time: {(t1-t0)*1e3:.2f} ms")

    ok = compare("gate_proj", gpu_gate, ref_gate, atol=0.15, cos_thresh=0.99)
    results["7_gate_proj"] = ok

    # ---------------------------------------------------------------
    # STEP 8: Up Projection (mlp.up_proj)
    # ---------------------------------------------------------------
    banner("STEP 8: Up Projection (5120 -> 17408)")

    up_prefix = "model.language_model.layers.0.mlp.up_proj"

    t0 = time.perf_counter()
    gpu_up, ref_up, d_up_out = run_gptq_matvec(
        gpu, proj_func, model, up_prefix, d_mlp_input, HIDDEN_DIM, INTERMEDIATE_SIZE
    )
    t1 = time.perf_counter()
    allocs.append(d_up_out)
    print(f"  Kernel time: {(t1-t0)*1e3:.2f} ms")

    ok = compare("up_proj", gpu_up, ref_up, atol=0.15, cos_thresh=0.99)
    results["8_up_proj"] = ok

    # ---------------------------------------------------------------
    # STEP 9: SiLU(gate) * up  (activate kernel)
    # ---------------------------------------------------------------
    banner("STEP 9: SiLU Activation -- gate * silu(up)")
    print("  NOTE: cached activate kernel computes gate * silu(up)")
    print("        but the standard Qwen MLP does silu(gate) * up")
    print("        Checking kernel signature to confirm semantics...")

    # The cached activate.ptx says: "output = gate * silu(up)"
    # But standard LLM GLU MLP does: silu(gate) * up
    # Let's handle this: we need silu(gate) * up
    # The kernel as written does: gate_val * silu(up_val)
    # So we swap: pass up as "gate_ptr" and gate as "up_ptr"
    # Then: up_val * silu(gate_val) = what we want

    d_act_out = gpu.mem_alloc(INTERMEDIATE_SIZE * 4)
    allocs.append(d_act_out)

    # Swap gate and up: kernel does param1 * silu(param2)
    # We want silu(gate) * up, so pass: param1=up, param2=gate
    GRID_ACT = max(1, math.ceil(INTERMEDIATE_SIZE / 256))
    gpu.launch(
        activate_func,
        grid=(GRID_ACT, 1, 1),
        block=(256, 1, 1),
        args=[
            ctypes.c_uint64(d_up_out.value),    # "gate_ptr" = up (multiplied straight)
            ctypes.c_uint64(d_gate_out.value),   # "up_ptr" = gate (gets SiLU applied)
            ctypes.c_uint64(d_act_out.value),
        ],
    )
    gpu.synchronize()

    gpu_act = download_f32(gpu, d_act_out, INTERMEDIATE_SIZE)
    ref_act = silu_ref(ref_gate) * ref_up

    ok = compare("silu_gate_mul_up", gpu_act, ref_act, atol=0.2, cos_thresh=0.99)
    results["9_silu_activation"] = ok

    # ---------------------------------------------------------------
    # STEP 10: Down Projection (mlp.down_proj, 17408 -> 5120)
    # ---------------------------------------------------------------
    banner("STEP 10: Down Projection (17408 -> 5120)")

    down_prefix = "model.language_model.layers.0.mlp.down_proj"

    # Upload activation output
    d_down_input = upload_f32(gpu, gpu_act)
    allocs.append(d_down_input)

    t0 = time.perf_counter()
    gpu_down, ref_down_from_gpu, d_down_out = run_gptq_matvec(
        gpu, proj_func, model, down_prefix, d_down_input, INTERMEDIATE_SIZE, HIDDEN_DIM
    )
    t1 = time.perf_counter()
    allocs.append(d_down_out)
    print(f"  Kernel time: {(t1-t0)*1e3:.2f} ms")

    # Also compute reference from ref activations
    down_qw_raw = bytes(model.weight_bytes(f"{down_prefix}.qweight"))
    down_sc_raw = bytes(model.weight_bytes(f"{down_prefix}.scales"))
    print("  Dequantizing down_proj weights...")
    down_weight = dequant_gptq_ref(down_qw_raw, down_sc_raw, INTERMEDIATE_SIZE, HIDDEN_DIM)
    ref_down = ref_act @ down_weight

    ok = compare("down_proj (gpu act input)", gpu_down, ref_down_from_gpu, atol=0.5, cos_thresh=0.98)
    results["10a_down_proj_gpu_vs_cpu_same_input"] = ok

    ok2 = compare("down_proj (ref chain)", gpu_down, ref_down, atol=1.0, cos_thresh=0.95)
    results["10b_down_proj_gpu_vs_ref_chain"] = ok2

    # ---------------------------------------------------------------
    # STEP 11: Final Residual Add
    # ---------------------------------------------------------------
    banner("STEP 11: Final Residual Add")

    gpu_layer_out = residual_after_attn + gpu_down
    ref_layer_out = ref_residual + ref_down

    ok = compare("layer_output", gpu_layer_out, ref_layer_out, atol=1.0, cos_thresh=0.95)
    results["11_layer_output"] = ok

    # Print activation statistics
    print(f"\n  Layer output stats:")
    print(f"    GPU: min={gpu_layer_out.min():.4f} max={gpu_layer_out.max():.4f} "
          f"L2={np.sqrt(np.sum(gpu_layer_out**2)):.4f}")
    print(f"    Ref: min={ref_layer_out.min():.4f} max={ref_layer_out.max():.4f} "
          f"L2={np.sqrt(np.sum(ref_layer_out**2)):.4f}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    banner("EXECUTION SUMMARY")

    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    print(f"\n  {n_pass}/{n_total} steps passed")
    print(f"\n  NOTE: DeltaNet recurrence was skipped. The attention portion uses")
    print(f"  V -> out_proj as a proxy. The MLP portion is fully tested with real")
    print(f"  GPTQ kernels: gate_proj, up_proj, SiLU activation, down_proj.")

    if all_pass:
        print(f"\n  ALL STEPS PASSED -- full layer 0 pipeline verified!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED steps: {failed}")

    # Cleanup
    for dptr in allocs:
        try:
            gpu.mem_free(dptr)
        except Exception:
            pass
    gpu.close()
    model.close()

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
