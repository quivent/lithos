"""
Lithos end-to-end pipeline test.

Wires together: loader -> factory -> cuda_driver -> engine -> tokenizer
to prove the orchestration works from prompt to generated token.

Many kernels aren't numerically complete yet; where that's the case we
either skip the kernel or zero-fill its output buffer. The point is to
validate that every component connects correctly and the engine can
produce a token id that round-trips through the tokenizer.

Run:
    python3 /home/ubuntu/lithos/src/pipeline_test.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
import traceback
from pathlib import Path

# Make sibling modules importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import LithosModel
from factory import KernelFactory
from cuda_driver import CUDADriver, CUDAError
from tokenizer import Tokenizer
from engine import (
    LithosEngine,
    LoadedModel,
    LoadedKernels,
    ModelConfig as EngineModelConfig,
)


MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def step(msg: str) -> None:
    print(f"\n--- {msg} ---")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    banner("Lithos pipeline end-to-end test")
    notes: list[str] = []
    worked: list[str] = []
    mocked: list[str] = []

    # ---------- 1. Load model (weights via mmap) ------------------------
    step("1. Loading model weights (mmap safetensors)")
    t0 = time.monotonic()
    try:
        model = LithosModel(MODEL_DIR)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 1
    t1 = time.monotonic()
    print(f"  {model}")
    print(f"  loaded in {t1 - t0:.3f}s  "
          f"({model.total_weight_bytes / (1024 ** 3):.2f} GiB, "
          f"{len(model.weight_names())} tensors)")
    cfg = model.config
    print(f"  arch={cfg.arch}  hidden={cfg.hidden_dim}  layers={cfg.num_layers}  "
          f"heads={cfg.num_heads} kv={cfg.num_kv_heads} head_dim={cfg.head_dim}  "
          f"vocab={cfg.vocab_size}")
    worked.append("LithosModel mmap load of 4 safetensors shards")

    # ---------- 2. Factory: generate / load 11 cubins -------------------
    step("2. KernelFactory: specialize + compile all cubins")
    try:
        factory = KernelFactory(MODEL_DIR)
        print(f"  config hash = {factory.config.config_hash()}")
        print(f"  cache dir   = {factory.cache_dir}")
        t0 = time.monotonic()
        cubins = factory.build_all()
        t1 = time.monotonic()
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 1
    print(f"  built/loaded {len(cubins)} cubins in {t1 - t0:.3f}s")
    for name in sorted(cubins):
        size = os.path.getsize(cubins[name])
        print(f"    {name:36s}  {size:>8,} B")
    worked.append(f"KernelFactory produced {len(cubins)} cubins")

    # ---------- 3. CUDA driver context ----------------------------------
    step("3. Creating CUDA driver + context")
    try:
        gpu = CUDADriver()
        print(f"  device: {gpu.device_name}")
        worked.append(f"CUDADriver context on {gpu.device_name}")
        cuda_available = True
    except (CUDAError, OSError) as e:
        print(f"  CUDA unavailable ({e}); falling back to CPU-only orchestration.")
        gpu = None
        cuda_available = False
        mocked.append("CUDA driver (no device / driver available)")

    # ---------- 4. Load cubins into modules, grab function handles ------
    step("4. Loading cubins into CUDA modules")
    kernel_funcs: dict[str, int] = {}  # logical name -> CUfunction int
    if cuda_available:
        try:
            # map factory kernel name -> (module path, function symbol in ptx)
            for name, path in cubins.items():
                mod = gpu.load_cubin(path)
                sym = name
                try:
                    func = gpu.get_function(mod, sym)
                except CUDAError:
                    base = name.split("_")[0]
                    func = gpu.get_function(mod, base)
                kernel_funcs[name] = ctypes.cast(func, ctypes.c_void_p).value or 0

            # Load the F16 embed kernel (reads F16 weights directly from mmap)
            embed_f16_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "embed_f16.cubin"
            )
            if os.path.exists(embed_f16_path):
                embed_f16_mod = gpu.load_cubin(embed_f16_path)
                embed_f16_func = gpu.get_function(embed_f16_mod, "embed_f16")
                kernel_funcs["embed"] = ctypes.cast(embed_f16_func, ctypes.c_void_p).value or 0
                print(f"  loaded embed_f16 kernel (reads F16 weights via unified memory)")

            print(f"  loaded {len(kernel_funcs)} kernel functions")
            worked.append("cuModuleLoadData + cuModuleGetFunction for all cubins + embed_f16")
        except Exception as e:
            print(f"  WARN: kernel load failed ({e}); continuing with zero handles")
            traceback.print_exc()
            mocked.append("cubin function handles (load failed)")
            kernel_funcs = {}
    else:
        mocked.append("cubin function handles (no CUDA)")

    # Build LoadedKernels — pick a representative projection for the generic slot
    def pick_proj() -> int:
        for k in kernel_funcs:
            if k.startswith("projection_"):
                return kernel_funcs[k]
        return 0

    loaded_kernels = LoadedKernels(
        projection_func=pick_proj(),
        attention_score_func=kernel_funcs.get("attention_score", 0),
        # No compiled DeltaNet recurrence cubin in the cache yet -> mock 0
        recurrence_func=0,
        norm_func=kernel_funcs.get("norm", 0),
        activate_func=kernel_funcs.get("activate", 0),
        rotate_func=kernel_funcs.get("rotate", 0),
        embed_func=kernel_funcs.get("embed", 0),
        sample_func=kernel_funcs.get("sample", 0),
    )
    if loaded_kernels.recurrence_func == 0:
        mocked.append("DeltaNet recurrence kernel (no cubin yet, engine logs launches)")

    # ---------- 5. Build engine with model-matched config ---------------
    step("5. Initializing LithosEngine")
    # Translate real-model dims into the engine's ModelConfig
    engine_cfg = EngineModelConfig(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_dim=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        max_seq_len=2048,  # keep test allocation small
        rope_theta=cfg.rope_theta,
        norm_eps=cfg.rms_norm_eps,
    )
    print(f"  engine cfg: layers={engine_cfg.num_layers} hidden={engine_cfg.hidden_dim} "
          f"vocab={engine_cfg.vocab_size} max_seq_len={engine_cfg.max_seq_len}")
    print(f"  deltanet layers: {engine_cfg.num_deltanet_layers}  "
          f"full-attn layers: {engine_cfg.num_full_attention_layers}")

    # Get real weight pointers for embed / lm_head / final norm
    def safe_weight_ptr(*names: str) -> int:
        for n in names:
            try:
                ptr, _, _ = model.weight(n)
                return ptr
            except KeyError:
                continue
        return 0

    embed_ptr = safe_weight_ptr(
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    )
    lm_head_ptr = safe_weight_ptr("lm_head.weight")
    final_norm_ptr = safe_weight_ptr(
        "model.language_model.norm.weight",
        "model.norm.weight",
    )
    print(f"  embed_ptr      = 0x{embed_ptr:016x}  "
          f"({'ok' if embed_ptr else 'MISSING'})")
    print(f"  lm_head_ptr    = 0x{lm_head_ptr:016x}  "
          f"({'ok' if lm_head_ptr else 'MISSING'})")
    print(f"  final_norm_ptr = 0x{final_norm_ptr:016x}  "
          f"({'ok' if final_norm_ptr else 'MISSING'})")

    loaded_model = LoadedModel(
        config=engine_cfg,
        base_ptr=0,  # we use absolute pointers directly
        embed_ptr=embed_ptr,
        lm_head_ptr=lm_head_ptr,
        final_norm_ptr=final_norm_ptr,
    )

    # Engine allocates KV cache + DeltaNet state (zeroed) on host-side mmap.
    # Pass the CUDADriver so the engine uses real cuLaunchKernel calls.
    # Only enable embed kernels for real GPU launch -- norm needs F32 weight
    # staging that isn't wired up in the engine yet (works in pipeline_exec.py).
    # All other kernels are log-only for orchestration testing.
    LithosEngine.set_live_kernels({"embed"})
    print(f"  live kernels: embed  (others: log-only, see pipeline_exec.py for norm proof)")
    try:
        engine = LithosEngine(loaded_model, loaded_kernels, cuda_driver=gpu)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 1
    mem = engine.memory_report()
    print("  pre-allocated buffers (zeroed):")
    for k, v in mem.items():
        print(f"    {k:28s}  {v / (1024 ** 2):8.1f} MiB")
    worked.append("KV cache + DeltaNet state + activation buffers pre-allocated (zeros)")

    # ---------- 6. Tokenize prompt --------------------------------------
    step("6. Tokenizing prompt")
    tok = Tokenizer(MODEL_DIR)
    prompt = "The capital of France is"
    token_ids = tok.encode(prompt)
    print(f"  prompt = {prompt!r}")
    print(f"  ids    = {token_ids}  ({len(token_ids)} tokens)")
    print(f"  round-trip decode: {tok.decode(token_ids)!r}")
    worked.append(f"Tokenizer encoded prompt into {len(token_ids)} ids")

    # ---------- 7. Prefill ----------------------------------------------
    step("7. Running prefill (engine orchestration)")
    LithosEngine.clear_launch_log()
    t0 = time.monotonic()
    try:
        engine.prefill(token_ids)
    except Exception as e:
        print(f"  FAIL during prefill: {e}")
        traceback.print_exc()
        return 1
    t1 = time.monotonic()
    prefill_launches = len(LithosEngine.get_launch_log())
    print(f"  prefill completed in {(t1 - t0) * 1000:.1f} ms  "
          f"({prefill_launches} kernel launches logged)")
    # Show a small sample of the log so it's clear orchestration actually ran
    sample = LithosEngine.get_launch_log()[:3] + ["..."] + LithosEngine.get_launch_log()[-3:]
    for line in sample:
        print(f"    {line}")
    worked.append(f"prefill orchestration issued {prefill_launches} logical launches over "
                  f"{engine_cfg.num_layers} layers")
    if cuda_available:
        worked.append("_gpu_launch calls real cuLaunchKernel via CUDADriver")
    else:
        mocked.append("actual GEMM math in _gpu_launch (stub logs launches instead of "
                      "invoking cuLaunchKernel)")

    # ---------- 8. Decode one step --------------------------------------
    step("8. Running one decode step")
    LithosEngine.clear_launch_log()
    t0 = time.monotonic()
    try:
        token_id = engine.decode_step()
    except Exception as e:
        print(f"  FAIL during decode: {e}")
        traceback.print_exc()
        return 1
    t1 = time.monotonic()
    decode_launches = len(LithosEngine.get_launch_log())
    print(f"  decode_step returned token_id={token_id}  "
          f"({decode_launches} launches, {(t1 - t0) * 1000:.1f} ms)")
    worked.append(f"decode_step orchestration issued {decode_launches} logical launches")
    if not cuda_available:
        mocked.append("_launch_sample returns token_id=0 (no CUDA)")
    # Note: _launch_sample now does real CPU argmax when logits are available

    # ---------- 9. Decode token back to text ----------------------------
    step("9. Decoding sampled token back to text")
    try:
        generated_text = tok.decode([token_id])
    except Exception as e:
        generated_text = f"<decode error: {e}>"
    print(f"  token_id = {token_id}")
    print(f"  decoded  = {generated_text!r}")
    worked.append("token id -> text via tokenizer decode")

    # ---------- 10. Final report ----------------------------------------
    banner("PIPELINE RESULT")
    print(f"  prompt:           {prompt!r}")
    print(f"  prompt tokens:    {token_ids}")
    print(f"  generated token:  {token_id} -> {generated_text!r}")
    print(f"  full text so far: {tok.decode(token_ids + [token_id])!r}")

    banner("WORKED")
    for w in worked:
        print(f"  [OK]   {w}")

    banner("MOCKED / STUBBED")
    for m in mocked:
        print(f"  [MOCK] {m}")

    banner("STATUS")
    print("""
  DONE:
    - engine._gpu_launch() calls real cuLaunchKernel via CUDADriver
    - GH200 unified memory: host mmap pointers work directly in kernels
      (no cuMemAlloc/cuMemcpyHtoD needed for weights)
    - embed_f16 kernel reads F16 safetensors weights, outputs F32 activations
    - RMSNorm kernel produces correct normalized output
    - _launch_sample does real CPU argmax over logits
    - Incremental bring-up: set_live_kernels() controls which kernels fire

  NEXT:
    - Enable projection kernels (need W4A16 GPTQ dequant)
    - Write F16-aware norm kernel (to read BF16 weights from mmap directly)
    - Run full first layer with real weights
    - Generate actual tokens from "The capital of France is"
""")

    # Cleanup
    try:
        engine.close()
    except Exception:
        pass
    if cuda_available and gpu is not None:
        try:
            gpu.close()
        except Exception:
            pass
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
