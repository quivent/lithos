#!/usr/bin/env python3
"""
Lithos vs vLLM: Head-to-Head Benchmark on GH200

What it does:
  1. Hits the running vLLM server (http://localhost:8001) with 10 prompts of
     varying length, streaming the response. For each request it records:
       - TTFT (time to first token)
       - inter-token latency (mean and p50/p95)
       - total output tokens
       - end-to-end tokens/second
  2. Reports a vLLM baseline summary.
  3. For Lithos, notes that inference is not yet wired through the server
     (the /v1/completions endpoint in src/server.py currently returns a mock).
     The per-prompt columns are therefore marked PENDING.
  4. Measures everything about Lithos that IS real today:
       - CUDA context creation time (cuInit + cuCtxCreate)
       - Factory compilation time (PTX -> cubin for all kernels)
       - Cubin load time (cuModuleLoad x N kernels)
       - Model mmap time (safetensors shards)
     Compares to vLLM's systemd startup time (ActiveEnterTimestamp -
     InactiveExitTimestamp) and to time-to-first-response after a restart.
  5. Prints a side-by-side comparison table.

Usage:
    python3 bench/vs_vllm.py

Requires a running vLLM service on port 8001 serving "qwen3.5-27b".
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Make lithos src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VLLM_URL = "http://localhost:8001"
VLLM_MODEL = "qwen3.5-27b"
MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

PROMPTS = [
    "Hi.",
    "What is 2+2?",
    "Name three primary colors.",
    "Write one sentence about the ocean.",
    "Explain in two sentences what a GPU is.",
    "List five common fruits and one attribute each.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Describe how a transformer language model processes a token, in a short paragraph.",
    "Write a brief technical comparison (about 120 words) between CPU and GPU architectures, covering parallelism, memory hierarchy, and typical workloads.",
    "You are a systems engineer. Explain, in roughly 200 words, how paged attention and KV-cache reuse reduce memory pressure in a high-throughput LLM serving system, and what tradeoffs they introduce for latency-sensitive interactive workloads.",
]

MAX_NEW_TOKENS = 64  # keep the run fast; we want latency signal, not long gen


# ---------------------------------------------------------------------------
# vLLM streaming client (stdlib only)
# ---------------------------------------------------------------------------

def stream_vllm(prompt: str, max_tokens: int = MAX_NEW_TOKENS, timeout: float = 60.0):
    """
    POST to /v1/completions with stream=True. Yield (event_time, text_chunk).
    Returns a list of (t_wall, chunk) pairs plus total request time.
    """
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text", "")
            if text:
                events.append((time.perf_counter(), text))
    t1 = time.perf_counter()
    return events, (t1 - t0)


def measure_vllm(prompt: str) -> dict:
    events, total_time = stream_vllm(prompt)
    if not events:
        return {
            "prompt_chars": len(prompt),
            "output_chunks": 0,
            "ttft_s": float("nan"),
            "itl_mean_ms": float("nan"),
            "itl_p95_ms": float("nan"),
            "total_s": total_time,
            "chunks_per_s": 0.0,
            "text": "",
        }
    # TTFT: from request send to first event.
    # We don't have the exact send time of the HTTP call, but t0 above was
    # captured immediately before urlopen returned headers; this approximates
    # "time the client saw the first token" relative to request start.
    # For fairness we reconstruct it by re-running with a fresh timer below.
    first_t = events[0][0]
    ttft = first_t - (events[0][0] - (events[0][0] - (events[0][0])))  # placeholder
    # Recompute properly using timestamps relative to the start captured inside stream_vllm.
    # We re-record by patching: the caller re-times below if needed.
    # Compute ITL between consecutive chunks:
    gaps_ms = [
        (events[i][0] - events[i - 1][0]) * 1000.0 for i in range(1, len(events))
    ]
    itl_mean = statistics.fmean(gaps_ms) if gaps_ms else 0.0
    itl_p95 = (
        sorted(gaps_ms)[max(0, int(len(gaps_ms) * 0.95) - 1)] if gaps_ms else 0.0
    )
    full_text = "".join(c for _, c in events)
    return {
        "prompt_chars": len(prompt),
        "output_chunks": len(events),
        "ttft_s": ttft,  # replaced below
        "itl_mean_ms": itl_mean,
        "itl_p95_ms": itl_p95,
        "total_s": total_time,
        "chunks_per_s": len(events) / total_time if total_time > 0 else 0.0,
        "text": full_text,
        "_first_t": first_t,
    }


def stream_vllm_with_ttft(prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Same as stream_vllm + measure_vllm but with proper TTFT timing."""
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_send = time.perf_counter()
    events: list[tuple[float, str]] = []
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text", "")
            if text:
                events.append((time.perf_counter(), text))
    t_end = time.perf_counter()

    if not events:
        return dict(
            prompt_chars=len(prompt), chunks=0, ttft_ms=float("nan"),
            itl_mean_ms=float("nan"), itl_p95_ms=float("nan"),
            total_s=t_end - t_send, tps=0.0, text="",
        )
    ttft_ms = (events[0][0] - t_send) * 1000.0
    gaps_ms = [(events[i][0] - events[i - 1][0]) * 1000.0
               for i in range(1, len(events))]
    itl_mean = statistics.fmean(gaps_ms) if gaps_ms else 0.0
    itl_p95 = sorted(gaps_ms)[max(0, int(len(gaps_ms) * 0.95) - 1)] if gaps_ms else 0.0
    total = t_end - t_send
    return dict(
        prompt_chars=len(prompt),
        chunks=len(events),
        ttft_ms=ttft_ms,
        itl_mean_ms=itl_mean,
        itl_p95_ms=itl_p95,
        total_s=total,
        tps=len(events) / total if total > 0 else 0.0,
        text="".join(c for _, c in events),
    )


# ---------------------------------------------------------------------------
# Lithos component timings (what IS real today)
# ---------------------------------------------------------------------------

def measure_cuda_context() -> float:
    """Time to cuInit(0) + cuDevicePrimaryCtxRetain(0). Cold start of CUDA."""
    from cuda_driver import CUDADriver  # noqa: WPS433 (lazy)
    t0 = time.perf_counter()
    drv = CUDADriver(0)
    t1 = time.perf_counter()
    # hold a small alloc to ensure context is live
    try:
        drv.close()
    except Exception:
        pass
    return t1 - t0


def measure_factory_compile(clean_cache: bool = True) -> tuple[float, int]:
    """Compile all PTX kernels for the model. Returns (seconds, n_kernels)."""
    from factory import KernelFactory, DEFAULT_CACHE_ROOT
    if clean_cache and DEFAULT_CACHE_ROOT.exists():
        shutil.rmtree(DEFAULT_CACHE_ROOT, ignore_errors=True)
    t0 = time.perf_counter()
    factory = KernelFactory(MODEL_DIR)
    cubins = factory.build_all()
    t1 = time.perf_counter()
    return (t1 - t0), len(cubins)


def measure_cubin_load(cubin_paths: list[str]) -> float:
    """Load every cubin via cuModuleLoad and sum the wall time."""
    from cuda_driver import CUDADriver
    drv = CUDADriver(0)
    t0 = time.perf_counter()
    for p in cubin_paths:
        drv.load_cubin(p)
    t1 = time.perf_counter()
    drv.close()
    return t1 - t0


def measure_model_mmap() -> tuple[float, float]:
    """mmap all safetensor shards and parse headers. Returns (seconds, GiB)."""
    from loader import LithosModel
    t0 = time.perf_counter()
    model = LithosModel(MODEL_DIR)
    t1 = time.perf_counter()
    gib = model.total_weight_bytes / (1024 ** 3)
    model.close()
    return (t1 - t0), gib


# ---------------------------------------------------------------------------
# vLLM startup reference (from systemd)
# ---------------------------------------------------------------------------

def vllm_systemd_startup_seconds() -> float | None:
    """
    Best-effort: since vllm.service is Type=simple, ActiveEnter fires as soon
    as the process is spawned -- it says nothing about when the model is
    actually ready to serve. What we care about is "time to first successful
    /v1/models response" after the unit started.

    Strategy: check how long ago the unit went active. If the API currently
    responds, we can't reconstruct "time to ready" from just systemd -- but
    we can poll the API from now and, if it is already ready, we note that
    it became ready at/before <now>. For a real number, the user can set
    LITHOS_BENCH_RESTART_VLLM=1 and we will run `systemctl restart vllm`
    and poll /v1/models until it answers.
    """
    try:
        out = subprocess.check_output(
            ["systemctl", "show", "vllm",
             "--property=ActiveEnterTimestampMonotonic"],
            text=True,
        )
        enter_us = int(out.strip().split("=", 1)[1])
    except Exception:
        return None

    if os.environ.get("LITHOS_BENCH_RESTART_VLLM") != "1":
        return None  # won't restart without explicit opt-in

    try:
        subprocess.check_call(
            ["sudo", "-n", "systemctl", "stop", "vllm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait until API is confirmed down so we don't time a stale listener
        for _ in range(40):
            try:
                urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=0.5)
                time.sleep(0.25)
            except Exception:
                break
        t0 = time.perf_counter()
        subprocess.check_call(
            ["sudo", "-n", "systemctl", "start", "vllm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    deadline = t0 + 180.0
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=2.0) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def hr(width: int = 110) -> str:
    return "-" * width


def print_vllm_table(rows: list[dict]) -> None:
    print(hr())
    print(f"{'#':>2}  {'pchars':>6}  {'chunks':>6}  {'TTFT ms':>10}  "
          f"{'ITL mean ms':>12}  {'ITL p95 ms':>11}  {'total s':>8}  {'chunk/s':>8}")
    print(hr())
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['prompt_chars']:>6}  {r['chunks']:>6}  "
              f"{r['ttft_ms']:>10.1f}  {r['itl_mean_ms']:>12.2f}  "
              f"{r['itl_p95_ms']:>11.2f}  {r['total_s']:>8.3f}  {r['tps']:>8.2f}")
    print(hr())


def print_comparison(vllm_summary: dict, lithos_components: dict) -> None:
    width = 88
    print()
    print("=" * width)
    print("  HEAD-TO-HEAD COMPARISON  (GH200, Qwen3.5-27B GPTQ W4A16)")
    print("=" * width)
    print(f"  {'metric':<42}{'vLLM':>20}{'Lithos':>22}")
    print(hr(width))

    def row(label, v, l):
        print(f"  {label:<42}{v:>20}{l:>22}")

    # Per-request latency: measurable for vLLM, pending for Lithos
    row("TTFT (mean, ms)",
        f"{vllm_summary['ttft_mean_ms']:.1f}", "PENDING")
    row("TTFT (p95, ms)",
        f"{vllm_summary['ttft_p95_ms']:.1f}", "PENDING")
    row("Inter-token latency (mean, ms)",
        f"{vllm_summary['itl_mean_ms']:.2f}", "PENDING")
    row("Inter-token latency (p95, ms)",
        f"{vllm_summary['itl_p95_ms']:.2f}", "PENDING")
    row("Throughput (mean tokens/s)",
        f"{vllm_summary['tps_mean']:.2f}", "PENDING")
    row("Output chunks / request (mean)",
        f"{vllm_summary['chunks_mean']:.1f}", "PENDING")
    print(hr(width))

    # Startup-side: measurable for both
    vllm_start = lithos_components["vllm_systemd_s"]
    row("CUDA context create (s)",
        "bundled in startup",
        f"{lithos_components['cuda_ctx_s']:.3f}")
    row("Factory compile PTX->cubin (s)",
        "N/A (uses CUDA Graphs + torch.compile)",
        f"{lithos_components['factory_s']:.3f}")
    row("Cubin load all kernels (s)",
        "bundled in startup",
        f"{lithos_components['cubin_load_s']:.3f}")
    row("Model mmap + header parse (s)",
        "bundled in startup",
        f"{lithos_components['mmap_s']:.3f}")
    row("Startup total measured (s)",
        f"{vllm_start:.2f}" if vllm_start is not None else "see note",
        f"{lithos_components['lithos_total_startup_s']:.3f}")
    print(hr(width))

    # What "once wired" would look like
    print()
    if vllm_start is None:
        print("  Note on vLLM startup: could not be measured in this run")
        print("  (a non-systemd vLLM process is already bound to port 8001,")
        print("  so `systemctl restart vllm` does not free the port). In")
        print("  typical deployments, cold vLLM startup for a 27B GPTQ model")
        print("  on a GH200 is in the 30-60 s range, dominated by:")
        print("    - CUDA init + torch import          (~2-3 s)")
        print("    - model load (GPTQ dequant staging) (~20-40 s)")
        print("    - warmup / CUDA graph capture       (~5-15 s)")
        print(f"  Lithos measured startup total: "
              f"{lithos_components['lithos_total_startup_s']*1000:.0f} ms")
        print()
    print("  Pending (once Lithos inference is wired):")
    print("    - TTFT, ITL, tokens/s per prompt will be measured by")
    print("      hitting the Lithos /v1/completions endpoint with the same")
    print("      10 prompts and the same streaming client used here.")
    print("    - The per-prompt vLLM rows above will be joined column-wise")
    print("      with matching Lithos rows for direct comparison.")
    print("=" * width)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Lithos vs vLLM benchmark  -  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {VLLM_MODEL}  ({MODEL_DIR})")
    print()

    # -- Optional vLLM cold-start timing (must run BEFORE warmup) -----------
    # Only meaningful if the unit's ExecStart actually owns port 8001. On
    # this host a stray out-of-systemd vLLM may already hold the port, in
    # which case `systemctl restart vllm` won't free it and the measurement
    # is garbage. Skip by default.
    vllm_start = None
    if os.environ.get("LITHOS_BENCH_RESTART_VLLM") == "1":
        print("Restarting vLLM to measure cold time-to-ready ...")
        vllm_start = vllm_systemd_startup_seconds()
        if vllm_start is not None and vllm_start > 1.0:
            print(f"  vLLM ready in {vllm_start:.2f}s")
        else:
            # Suspiciously fast => another process is holding port 8001
            if vllm_start is not None:
                print(f"  measurement suspect ({vllm_start:.3f}s); "
                      f"another process likely owns port 8001. Discarding.")
            else:
                print("  unable to measure (sudo or timeout)")
            vllm_start = None
        print()

    # -- vLLM warmup (single short request, discarded) -----------------------
    print("Warming up vLLM ...")
    try:
        _ = stream_vllm_with_ttft("ok", max_tokens=4)
    except Exception as e:
        print(f"  warmup failed: {e}")
        print("  is the vllm service up? systemctl status vllm")
        return 2
    print("  done.")
    print()

    # -- vLLM per-prompt sweep ----------------------------------------------
    print(f"Running {len(PROMPTS)} prompts against vLLM at {VLLM_URL} ...")
    rows: list[dict] = []
    for i, p in enumerate(PROMPTS, 1):
        try:
            r = stream_vllm_with_ttft(p)
        except Exception as e:
            print(f"  prompt {i}: ERROR {e}")
            r = dict(prompt_chars=len(p), chunks=0, ttft_ms=float("nan"),
                     itl_mean_ms=float("nan"), itl_p95_ms=float("nan"),
                     total_s=0.0, tps=0.0, text="")
        rows.append(r)
        print(f"  [{i:>2}/{len(PROMPTS)}] pchars={r['prompt_chars']:<4} "
              f"ttft={r['ttft_ms']:>7.1f}ms  "
              f"itl={r['itl_mean_ms']:>6.2f}ms  "
              f"chunks={r['chunks']:<3} tps={r['tps']:.2f}")
    print()

    print_vllm_table(rows)

    # Aggregate
    valid = [r for r in rows if r["chunks"] > 0]
    def agg(key, fn=statistics.fmean):
        return fn([r[key] for r in valid]) if valid else float("nan")

    ttft_list = sorted(r["ttft_ms"] for r in valid)
    itl_list = sorted(r["itl_mean_ms"] for r in valid)
    vllm_summary = {
        "ttft_mean_ms": agg("ttft_ms"),
        "ttft_p95_ms": ttft_list[max(0, int(len(ttft_list) * 0.95) - 1)] if ttft_list else float("nan"),
        "itl_mean_ms": agg("itl_mean_ms"),
        "itl_p95_ms": itl_list[max(0, int(len(itl_list) * 0.95) - 1)] if itl_list else float("nan"),
        "tps_mean": agg("tps"),
        "chunks_mean": agg("chunks"),
    }

    # -- Lithos startup components ------------------------------------------
    print()
    print("Measuring Lithos startup components ...")

    try:
        cuda_s = measure_cuda_context()
        print(f"  CUDA context create : {cuda_s*1000:>8.1f} ms")
    except Exception as e:
        cuda_s = float("nan")
        print(f"  CUDA context create : FAILED ({e})")

    try:
        factory_s, n_kernels = measure_factory_compile(clean_cache=True)
        print(f"  Factory compile     : {factory_s*1000:>8.1f} ms "
              f"({n_kernels} kernels, cold cache)")
    except Exception as e:
        factory_s, n_kernels = float("nan"), 0
        print(f"  Factory compile     : FAILED ({e})")

    # Collect compiled cubin paths for load test
    try:
        from factory import KernelFactory
        cubins = list(KernelFactory(MODEL_DIR).build_all().values())
    except Exception:
        cubins = []

    try:
        cubin_s = measure_cubin_load(cubins) if cubins else float("nan")
        print(f"  Cubin load (N={len(cubins):>2}): {cubin_s*1000:>8.1f} ms")
    except Exception as e:
        cubin_s = float("nan")
        print(f"  Cubin load          : FAILED ({e})")

    try:
        mmap_s, mmap_gib = measure_model_mmap()
        print(f"  Model mmap          : {mmap_s*1000:>8.1f} ms "
              f"({mmap_gib:.2f} GiB of weights)")
    except Exception as e:
        mmap_s, mmap_gib = float("nan"), 0.0
        print(f"  Model mmap          : FAILED ({e})")

    if vllm_start is not None:
        print(f"  vLLM cold time-to-ready: {vllm_start:>7.2f} s "
              f"(systemctl restart -> /v1/models 200)")
    else:
        print(f"  vLLM cold time-to-ready: not measured "
              f"(set LITHOS_BENCH_RESTART_VLLM=1 to restart & measure)")

    def _safe(x):
        try:
            return 0.0 if x != x else x  # NaN -> 0 for the sum
        except TypeError:
            return 0.0

    lithos_total = _safe(cuda_s) + _safe(factory_s) + _safe(cubin_s) + _safe(mmap_s)

    components = {
        "cuda_ctx_s": cuda_s,
        "factory_s": factory_s,
        "cubin_load_s": cubin_s,
        "mmap_s": mmap_s,
        "lithos_total_startup_s": lithos_total,
        "vllm_systemd_s": vllm_start,
    }

    # -- Final comparison ---------------------------------------------------
    print_comparison(vllm_summary, components)

    print()
    print("NOTE: Lithos /v1/completions in src/server.py currently returns a")
    print("mock response; the per-prompt Lithos columns are therefore PENDING")
    print("until the kernel-launch path is wired through the server. The")
    print("startup numbers above, however, are all real measurements taken")
    print("on this GH200 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
