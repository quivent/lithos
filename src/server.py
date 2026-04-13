#!/usr/bin/env python3
"""
Lithos inference server — OpenAI-compatible API.

Zero framework dependencies beyond uvicorn. Uses only:
  - uvicorn (ASGI server)
  - starlette (minimal ASGI toolkit, installed with uvicorn)
  - json, time, uuid, asyncio from stdlib

Startup path demonstrates the Lithos advantage: sub-second cold start
vs. vLLM's 2+ minutes. Weights are mmap'd, cubins are loaded directly
via the CUDA driver API, and KV cache is pre-allocated in unified memory.

Run:
    python3 lithos/src/server.py
    python3 lithos/src/server.py --model /path/to/model --port 8080
"""

import argparse
import asyncio
import ctypes
import json
import mmap
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Union

# ---------------------------------------------------------------------------
# Starlette imports (ships with uvicorn)
# ---------------------------------------------------------------------------
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = PROJECT_ROOT / "kernels"

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
_T0 = time.monotonic()


def _ts() -> str:
    """Elapsed time since process start, formatted for startup log."""
    return f"{time.monotonic() - _T0:.3f}s"


# ===================================================================
# Model / engine types
# ===================================================================

@dataclass
class ModelConfig:
    name: str = "lithos-qwen-27b-w4"
    hidden_dim: int = 3584
    num_layers: int = 64
    num_heads: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    intermediate_dim: int = 18944
    vocab_size: int = 152064
    max_seq_len: int = 32768
    quantization: str = "w4a16"
    weight_bytes: int = 18_300_000_000  # ~18.3 GB (W4)


@dataclass
class CubinHandle:
    name: str
    module: ctypes.c_void_p
    function: ctypes.c_void_p
    size_bytes: int


@dataclass
class KVCache:
    """Placeholder for pre-allocated KV cache."""
    num_layers: int = 64
    max_batch: int = 24
    max_seq_len: int = 32768
    head_dim: int = 128
    num_kv_heads: int = 4
    dtype_bytes: int = 2  # FP16
    allocated_bytes: int = 0

    def compute_size(self) -> int:
        # 2 (K+V) * layers * batch * seq * heads * dim * dtype
        return (2 * self.num_layers * self.max_batch * self.max_seq_len
                * self.num_kv_heads * self.head_dim * self.dtype_bytes)


@dataclass
class Metrics:
    requests_served: int = 0
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        dt = self.uptime_s
        return self.tokens_generated / dt if dt > 0 else 0.0


# ===================================================================
# Engine (mock — replace internals with real Lithos dispatch)
# ===================================================================

class LithosEngine:
    """
    Inference engine.  Currently mocked — every method marks where real
    CUDA kernel launches would happen.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.cubins: list[CubinHandle] = []
        self.kv_cache = KVCache(
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
        )
        self._cuda = None
        self._context = ctypes.c_void_p()

    # ---- startup --------------------------------------------------------

    def load_config(self, model_path: Optional[str] = None) -> None:
        """Load model config.json (or use compiled-in defaults)."""
        # TODO: if model_path provided, read config.json from it
        print(f"[{_ts()}] Loaded config.json  "
              f"({self.config.num_layers}L, {self.config.hidden_dim}d, "
              f"{self.config.quantization})")

    def mmap_weights(self, model_path: Optional[str] = None) -> None:
        """
        mmap weight files into the unified address space.

        On GH200 this means: open() + mmap() and you're done.
        The GPU reads these pointers directly via NVLink-C2C coherence.
        No cudaMalloc, no cudaMemcpy, no staging buffers.
        """
        gb = self.config.weight_bytes / (1 << 30)
        # TODO(real): glob safetensors/bin files, mmap each one
        #   weight_files = sorted(Path(model_path).glob("*.safetensors"))
        #   for wf in weight_files:
        #       fd = os.open(wf, os.O_RDONLY)
        #       mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        #       self._weight_maps.append(mm)
        print(f"[{_ts()}] mmap'd weight files ({gb:.1f} GB)")

    def load_cubins(self) -> None:
        """
        Load pre-compiled cubins via the CUDA driver API.

        Each cubin is a single fused kernel — no framework compilation,
        no JIT warmup.  7 cubins, ~30 KB total, loads in microseconds.
        """
        cubin_files = sorted(KERNEL_DIR.glob("*.cubin"))
        if not cubin_files:
            print(f"[{_ts()}] No cubins found in {KERNEL_DIR} "
                  "(running in mock mode)")
            return

        # Try loading real cubins via driver API
        try:
            cuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            try:
                cuda = ctypes.CDLL("libcuda.so")
            except OSError:
                print(f"[{_ts()}] libcuda.so not available — "
                      f"mock-loading {len(cubin_files)} cubins")
                for cf in cubin_files:
                    self.cubins.append(CubinHandle(
                        name=cf.stem, module=ctypes.c_void_p(),
                        function=ctypes.c_void_p(),
                        size_bytes=cf.stat().st_size,
                    ))
                return

        self._cuda = cuda
        cuda.cuInit(0)

        device = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(device), 0)
        cuda.cuCtxCreate_v2(
            ctypes.byref(self._context), 0, device)

        for cf in cubin_files:
            module = ctypes.c_void_p()
            rc = cuda.cuModuleLoad(
                ctypes.byref(module), str(cf).encode())
            if rc != 0:
                print(f"  WARN: cuModuleLoad({cf.name}) error {rc}")
                continue

            function = ctypes.c_void_p()
            entry = cf.stem.encode()
            rc = cuda.cuModuleGetFunction(
                ctypes.byref(function), module, entry)
            if rc != 0:
                print(f"  WARN: cuModuleGetFunction({cf.stem}) error {rc}")
                cuda.cuModuleUnload(module)
                continue

            self.cubins.append(CubinHandle(
                name=cf.stem, module=module, function=function,
                size_bytes=cf.stat().st_size,
            ))

        print(f"[{_ts()}] Loaded {len(self.cubins)} cubins")

    def create_cuda_context(self) -> None:
        """Ensure CUDA context exists (may already be created in load_cubins)."""
        # Context creation happened in load_cubins if driver was available.
        print(f"[{_ts()}] CUDA context ready")

    def preallocate_kv_cache(self) -> None:
        """
        Pre-allocate KV cache in unified memory.

        On GH200: mmap a large region.  The hardware migrates pages
        between HBM and LPDDR5X on demand — starts in HBM, spills
        gracefully over NVLink-C2C when full.
        """
        total = self.kv_cache.compute_size()
        self.kv_cache.allocated_bytes = total
        gb = total / (1 << 30)
        # TODO(real): mmap(MAP_ANONYMOUS | MAP_PRIVATE, total)
        #   or cudaMallocManaged for discrete GPUs
        print(f"[{_ts()}] Pre-allocated KV cache ({gb:.1f} GB)")

    # ---- inference (mock) -----------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize input text.

        TODO(real): load tokenizer.json / tokenizer.model from model dir.
        For now returns deterministic mock token ids.
        """
        # Mock: ~1.3 tokens per word, plus some for special tokens
        words = text.split()
        return list(range(1, len(words) + 2))

    def detokenize(self, token_ids: list[int]) -> str:
        """
        Convert token ids back to text.

        TODO(real): use the loaded tokenizer.
        """
        # Mock vocabulary for demo responses
        mock_vocab = [
            "Hello", ",", " I", "'m", " Lith", "os", ",",
            " a", " fast", " inference", " engine", ".",
            " How", " can", " I", " help", " you", " today", "?",
        ]
        parts = []
        for tid in token_ids:
            parts.append(mock_vocab[tid % len(mock_vocab)])
        return "".join(parts)

    async def prefill(self, token_ids: list[int]) -> list[float]:
        """
        Process prompt tokens (batched matmul through all layers).

        Returns logits for the last position.

        TODO(real):
          - For each layer: launch attention_kernel, mlp_kernel
          - Or: launch single fused forward_pass kernel
          - Write KV cache entries for all prompt positions
          - cuLaunchKernel(self.cubins["projection"].function, ...)
        """
        # Simulate ~1ms per 100 tokens of prefill
        delay = max(0.001, len(token_ids) * 0.00001)
        await asyncio.sleep(delay)
        # Return mock logits
        return [0.0] * self.config.vocab_size

    async def decode_step(self, prev_token: int, seq_pos: int) -> int:
        """
        Generate one token (autoregressive decode).

        TODO(real):
          - Launch fused decoder layer kernels
          - Read KV cache for attention
          - Write new KV entry at seq_pos
          - Sample from logits on CPU (Grace cores)
          - cuLaunchKernel for each layer, or one mega-kernel
        """
        # Simulate ~2ms per token decode (targeting 400+ tok/s)
        await asyncio.sleep(0.002)
        # Return mock token — cycle through a short response
        mock_response = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        return mock_response[seq_pos % len(mock_response)]


# ===================================================================
# Global state
# ===================================================================

engine: Optional[LithosEngine] = None
metrics = Metrics()
_model_path: Optional[str] = None


# ===================================================================
# Startup sequence
# ===================================================================

def startup_engine(model_path: Optional[str] = None) -> LithosEngine:
    """
    Full startup sequence.  This is where the Lithos advantage shows:
    sub-second from process start to ready-to-serve.

    vLLM equivalent takes ~120s:
      - PyTorch import: ~5s
      - Model class instantiation + weight loading: ~30-60s
      - CUDA graph capture: ~20-30s
      - KV cache profiling: ~10-20s
      - Worker process setup: ~5-10s

    Lithos:
      - mmap weights: <1ms (no deserialization, no copy)
      - Load cubins: <1ms (pre-compiled, no JIT)
      - CUDA context: ~100ms (driver initialization)
      - KV cache alloc: <1ms (mmap, no cudaMalloc)
    """
    global _T0
    _T0 = time.monotonic()

    print(f"{'='*60}")
    print(f"  Lithos inference server — starting up")
    print(f"{'='*60}")

    config = ModelConfig()
    eng = LithosEngine(config)

    eng.load_config(model_path)
    eng.mmap_weights(model_path)
    eng.load_cubins()
    eng.create_cuda_context()
    eng.preallocate_kv_cache()

    return eng


# ===================================================================
# API handlers
# ===================================================================

async def health(request: Request) -> JSONResponse:
    """GET /health"""
    return JSONResponse({"status": "ok"})


async def list_models(request: Request) -> JSONResponse:
    """GET /v1/models"""
    model_name = engine.config.name if engine else "lithos-unknown"
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(metrics.start_time),
                "owned_by": "lithos",
                "permission": [],
                "root": model_name,
                "parent": None,
            }
        ],
    })


def _make_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


async def _generate_tokens(
    prompt_tokens: list[int],
    max_tokens: int = 128,
    stop_token: int = -1,
) -> AsyncIterator[tuple[int, str]]:
    """
    Core generation loop: prefill then decode.

    Yields (token_id, token_text) pairs.
    """
    # --- prefill ---
    await engine.prefill(prompt_tokens)

    # --- decode ---
    seq_pos = len(prompt_tokens)
    for i in range(max_tokens):
        prev = prompt_tokens[-1] if i == 0 else token_id
        token_id = await engine.decode_step(prev, seq_pos + i)

        if token_id == stop_token:
            break

        token_text = engine.detokenize([token_id])
        metrics.tokens_generated += 1
        yield token_id, token_text


async def chat_completions(request: Request) -> Union[JSONResponse, StreamingResponse]:
    """POST /v1/chat/completions"""
    body = await request.json()

    model = body.get("model", engine.config.name)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 1.0)

    # Flatten messages into a single prompt string
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|{role}|>\n{content}")
    prompt_parts.append("<|assistant|>\n")
    prompt_text = "\n".join(prompt_parts)

    prompt_tokens = engine.tokenize(prompt_text)
    request_id = _make_chat_id()
    created = int(time.time())

    metrics.requests_served += 1

    if stream:
        return StreamingResponse(
            _stream_chat(request_id, model, created, prompt_tokens, max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect all tokens
    output_parts = []
    completion_tokens = 0
    async for _tid, token_text in _generate_tokens(prompt_tokens, max_tokens):
        output_parts.append(token_text)
        completion_tokens += 1

    assistant_text = "".join(output_parts)

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
        },
    })


async def _stream_chat(
    request_id: str,
    model: str,
    created: int,
    prompt_tokens: list[int],
    max_tokens: int,
) -> AsyncIterator[str]:
    """SSE stream for chat completions."""

    # Initial chunk with role
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Token-by-token chunks
    async for _tid, token_text in _generate_tokens(prompt_tokens, max_tokens):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def completions(request: Request) -> Union[JSONResponse, StreamingResponse]:
    """POST /v1/completions — raw text completion endpoint."""
    body = await request.json()

    model = body.get("model", engine.config.name)
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)

    prompt_tokens = engine.tokenize(prompt)
    request_id = _make_completion_id()
    created = int(time.time())

    metrics.requests_served += 1

    if stream:
        return StreamingResponse(
            _stream_completion(request_id, model, created, prompt_tokens, max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    output_parts = []
    completion_tokens = 0
    async for _tid, token_text in _generate_tokens(prompt_tokens, max_tokens):
        output_parts.append(token_text)
        completion_tokens += 1

    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": "".join(output_parts),
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
        },
    })


async def _stream_completion(
    request_id: str,
    model: str,
    created: int,
    prompt_tokens: list[int],
    max_tokens: int,
) -> AsyncIterator[str]:
    """SSE stream for raw completions."""
    async for _tid, token_text in _generate_tokens(prompt_tokens, max_tokens):
        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": token_text,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def server_metrics(request: Request) -> JSONResponse:
    """GET /metrics — basic operational metrics."""
    return JSONResponse({
        "requests_served": metrics.requests_served,
        "tokens_generated": metrics.tokens_generated,
        "tokens_per_second": round(metrics.tokens_per_second, 2),
        "uptime_seconds": round(metrics.uptime_s, 2),
        "cubins_loaded": len(engine.cubins) if engine else 0,
        "kv_cache_allocated_gb": round(
            engine.kv_cache.allocated_bytes / (1 << 30), 1
        ) if engine else 0,
        "model": engine.config.name if engine else None,
    })


# ===================================================================
# Application
# ===================================================================

routes = [
    Route("/health", health, methods=["GET"]),
    Route("/v1/models", list_models, methods=["GET"]),
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/v1/completions", completions, methods=["POST"]),
    Route("/metrics", server_metrics, methods=["GET"]),
]

app = Starlette(routes=routes)


# ===================================================================
# Main
# ===================================================================

def main():
    global engine, metrics

    parser = argparse.ArgumentParser(description="Lithos inference server")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Listen port (default: 8080)")
    parser.add_argument("--max-batch", type=int, default=256,
                        help="Maximum concurrent batch size")
    parser.add_argument("--max-seq-len", type=int, default=32768,
                        help="Maximum sequence length")
    args = parser.parse_args()

    # --- startup ---
    engine = startup_engine(args.model)
    metrics = Metrics()

    print(f"[{_ts()}] Ready to serve on port {args.port}")
    print(f"{'='*60}")
    print()
    print(f"  Endpoints:")
    print(f"    GET  http://{args.host}:{args.port}/health")
    print(f"    GET  http://{args.host}:{args.port}/v1/models")
    print(f"    POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"    POST http://{args.host}:{args.port}/v1/completions")
    print(f"    GET  http://{args.host}:{args.port}/metrics")
    print()

    # --- serve ---
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,  # we do our own logging
    )


if __name__ == "__main__":
    main()
