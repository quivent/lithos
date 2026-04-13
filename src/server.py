#!/usr/bin/env python3
"""
Lithos inference server — OpenAI-compatible API with REAL inference.

Uses the InferenceEngine from generate_first_token.py to run Qwen 3.5-27B
on GH200 via hand-written CUDA kernels (GPTQ W4A16 matvec, RMSNorm,
SiLU activation, embed).  No PyTorch, no vLLM.

Each token requires a full forward pass through 64 layers (~seconds per
token with current unoptimized kernels).  Responses stream via SSE as
tokens are generated.

Run:
    python3 lithos/src/server.py
    python3 lithos/src/server.py --port 8080
"""

import argparse
import asyncio
import ctypes
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Starlette imports (ships with uvicorn)
# ---------------------------------------------------------------------------
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Project imports — real inference engine
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from generate_first_token import (
    InferenceEngine,
    rms_norm_cpu,
    HIDDEN_DIM,
    NUM_LAYERS,
    download_f32,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = SRC_DIR.parent
KERNEL_DIR = PROJECT_ROOT / "kernels"

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
_T0 = time.monotonic()


def _ts() -> str:
    """Elapsed time since process start, formatted for startup log."""
    return f"{time.monotonic() - _T0:.3f}s"


# ===================================================================
# Metrics
# ===================================================================

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
# Wrapper around InferenceEngine for server use
# ===================================================================

class LithosServerEngine:
    """
    Wraps InferenceEngine from generate_first_token.py to provide
    tokenize / forward_token / sample methods suitable for serving.

    Each forward_token call runs a single token through all 64 layers
    (embed -> 64x {attn + MLP} -> final_norm -> lm_head -> logits).

    This is first-token-only inference (no KV cache reuse across
    positions), so every generated token is an independent forward pass.
    Slow but real.
    """

    def __init__(self):
        self.inf = InferenceEngine()
        self.model_name = "lithos-qwen3.5-27b-gptq-w4a16"
        # Cache the lm_head weight bytes once (they are large but read-only)
        print(f"[{_ts()}] Caching lm_head weights...")
        t0 = time.monotonic()
        self._lm_head_raw = bytes(self.inf.model.weight_bytes("lm_head.weight"))
        self._vocab_size = 248320
        print(f"[{_ts()}] lm_head cached ({len(self._lm_head_raw) / (1<<20):.0f} MB) "
              f"in {time.monotonic() - t0:.2f}s")

    @property
    def tokenizer(self):
        return self.inf.tok

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text using the real Qwen tokenizer."""
        return self.inf.tok.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.inf.tok.decode(token_ids)

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Format messages using the model's chat template."""
        return self.inf.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )

    def forward_token(self, token_id: int) -> np.ndarray:
        """
        Run a single token through the full 64-layer pipeline.

        Returns logits array of shape [vocab_size].

        This is a synchronous, blocking call that takes several seconds
        due to unoptimized per-layer kernel launches.
        """
        gpu = self.inf.gpu

        # Step 1: Embed
        x = self.inf.gpu_embed(token_id)

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise RuntimeError(f"NaN/Inf in embedding for token {token_id}")

        # Step 2: Run all 64 layers (phase=3 = complete pipeline)
        for layer_idx in range(NUM_LAYERS):
            x = self.inf.run_layer(layer_idx, x, phase=3)

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                raise RuntimeError(
                    f"NaN/Inf at layer {layer_idx} for token {token_id}"
                )

        # Step 3: Final RMSNorm
        x_normed = rms_norm_cpu(x, self.inf.final_norm_w, self.inf.epsilon)

        # Step 4: lm_head projection (FP16, 248320 x 5120 matmul on CPU)
        CHUNK = 8192
        logits = np.zeros(self._vocab_size, dtype=np.float32)
        for start in range(0, self._vocab_size, CHUNK):
            end = min(start + CHUNK, self._vocab_size)
            chunk_size = end - start
            offset = start * HIDDEN_DIM * 2  # 2 bytes per fp16
            chunk_bytes = chunk_size * HIDDEN_DIM * 2
            w_chunk = np.frombuffer(
                self._lm_head_raw[offset:offset + chunk_bytes],
                dtype=np.float16,
            ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
            logits[start:end] = w_chunk @ x_normed

        return logits

    def sample(self, logits: np.ndarray, temperature: float = 1.0,
               top_k: int = 50) -> int:
        """Sample a token from logits with temperature and top-k."""
        if temperature <= 0 or temperature < 1e-6:
            # Greedy
            return int(np.argmax(logits))

        # Temperature scaling
        scaled = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < len(scaled):
            top_k_idx = np.argpartition(scaled, -top_k)[-top_k:]
            mask = np.full_like(scaled, -np.inf)
            mask[top_k_idx] = scaled[top_k_idx]
            scaled = mask

        # Softmax
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)

        return int(np.random.choice(len(probs), p=probs))

    def close(self):
        self.inf.close()


# ===================================================================
# Global state
# ===================================================================

engine: Optional[LithosServerEngine] = None
metrics = Metrics()
_startup_duration: float = 0.0


# ===================================================================
# Startup sequence
# ===================================================================

def startup_engine() -> LithosServerEngine:
    """
    Full startup sequence: initialize InferenceEngine (loads model weights
    via mmap, loads cubins via CUDA driver API, allocates GPU buffers,
    preloads norm weights).
    """
    global _T0, _startup_duration
    _T0 = time.monotonic()

    print(f"{'='*60}")
    print(f"  Lithos inference server — starting up (REAL inference)")
    print(f"{'='*60}")

    eng = LithosServerEngine()

    _startup_duration = time.monotonic() - _T0
    print(f"[{_ts()}] Startup complete in {_startup_duration:.2f}s")

    return eng


# ===================================================================
# API handlers
# ===================================================================

async def health(request: Request) -> JSONResponse:
    """GET /health"""
    return JSONResponse({
        "status": "ok",
        "startup_seconds": round(_startup_duration, 2),
    })


async def list_models(request: Request) -> JSONResponse:
    """GET /v1/models"""
    model_name = engine.model_name if engine else "lithos-unknown"
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
    temperature: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
) -> AsyncIterator[tuple[int, str, float]]:
    """
    Core generation loop using REAL inference.

    For prefill: runs the last prompt token through the full pipeline to
    get the first output token.  Then for each decode step, runs the
    previously generated token through the pipeline again.

    Without KV cache reuse, each token is an independent forward pass
    through all 64 layers.  This is slow (~seconds per token) but produces
    real model outputs.

    Yields (token_id, token_text, token_time_seconds) triples.
    """
    if stop_token_ids is None:
        eos_id = engine.tokenizer.eos_token_id
        stop_token_ids = [eos_id] if eos_id is not None else []

    loop = asyncio.get_event_loop()

    # --- prefill: process last prompt token to get first output ---
    input_token = prompt_tokens[-1] if prompt_tokens else 0
    print(f"  [generate] prefill token_id={input_token}, "
          f"prompt_len={len(prompt_tokens)}")

    for i in range(max_tokens):
        t_tok = time.monotonic()

        # Run forward pass in executor to avoid blocking the event loop
        logits = await loop.run_in_executor(
            None, engine.forward_token, input_token
        )

        token_id = engine.sample(logits, temperature=temperature)
        token_time = time.monotonic() - t_tok

        if token_id in stop_token_ids:
            print(f"  [generate] stop token {token_id} after {i+1} tokens")
            break

        token_text = engine.detokenize([token_id])
        metrics.tokens_generated += 1

        tok_s = 1.0 / token_time if token_time > 0 else 0.0
        print(f"  [generate] token {i}: id={token_id} "
              f"text={token_text!r} time={token_time:.2f}s "
              f"({tok_s:.2f} tok/s)")

        yield token_id, token_text, token_time

        # Next iteration uses the token we just generated
        input_token = token_id


async def chat_completions(request: Request) -> Union[JSONResponse, StreamingResponse]:
    """POST /v1/chat/completions"""
    body = await request.json()

    model = body.get("model", engine.model_name)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.0)  # default greedy

    # Format prompt using the model's real chat template
    try:
        prompt_text = engine.apply_chat_template(messages)
    except Exception as e:
        # Fallback: simple concatenation
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|{role}|>\n{content}")
        prompt_parts.append("<|assistant|>\n")
        prompt_text = "\n".join(prompt_parts)
        print(f"  [chat] chat template failed ({e}), using fallback format")

    prompt_tokens = engine.tokenize(prompt_text)
    request_id = _make_chat_id()
    created = int(time.time())

    metrics.requests_served += 1

    print(f"\n{'='*60}")
    print(f"  [chat] request={request_id}")
    print(f"  [chat] prompt_tokens={len(prompt_tokens)}, "
          f"max_tokens={max_tokens}, temperature={temperature}")
    print(f"  [chat] startup_time={_startup_duration:.2f}s")
    t_request = time.monotonic()

    if stream:
        return StreamingResponse(
            _stream_chat(request_id, model, created, prompt_tokens,
                         max_tokens, temperature),
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
    total_gen_time = 0.0
    first_token_time = None
    async for _tid, token_text, tok_time in _generate_tokens(
        prompt_tokens, max_tokens, temperature
    ):
        output_parts.append(token_text)
        completion_tokens += 1
        total_gen_time += tok_time
        if first_token_time is None:
            first_token_time = tok_time

    assistant_text = "".join(output_parts)
    request_time = time.monotonic() - t_request
    avg_tok_s = completion_tokens / total_gen_time if total_gen_time > 0 else 0.0

    print(f"  [chat] done: {completion_tokens} tokens in {request_time:.2f}s "
          f"({avg_tok_s:.2f} tok/s avg)")
    if first_token_time is not None:
        print(f"  [chat] first_token_time={first_token_time:.2f}s")

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
        "lithos_timing": {
            "first_token_seconds": round(first_token_time, 3) if first_token_time else None,
            "total_seconds": round(request_time, 3),
            "tokens_per_second": round(avg_tok_s, 3),
            "startup_seconds": round(_startup_duration, 3),
        },
    })


async def _stream_chat(
    request_id: str,
    model: str,
    created: int,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 0.0,
) -> AsyncIterator[str]:
    """SSE stream for chat completions with real inference."""

    t_stream_start = time.monotonic()

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
    completion_tokens = 0
    first_token_time = None
    async for _tid, token_text, tok_time in _generate_tokens(
        prompt_tokens, max_tokens, temperature
    ):
        completion_tokens += 1
        if first_token_time is None:
            first_token_time = tok_time
            print(f"  [stream] first_token_time={first_token_time:.2f}s")

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

    total_time = time.monotonic() - t_stream_start
    avg_tok_s = completion_tokens / total_time if total_time > 0 else 0.0
    print(f"  [stream] done: {completion_tokens} tokens in {total_time:.2f}s "
          f"({avg_tok_s:.2f} tok/s avg)")

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
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
        },
        "lithos_timing": {
            "first_token_seconds": round(first_token_time, 3) if first_token_time else None,
            "total_seconds": round(total_time, 3),
            "tokens_per_second": round(avg_tok_s, 3),
        },
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def completions(request: Request) -> Union[JSONResponse, StreamingResponse]:
    """POST /v1/completions — raw text completion endpoint."""
    body = await request.json()

    model = body.get("model", engine.model_name)
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.0)

    prompt_tokens = engine.tokenize(prompt)
    request_id = _make_completion_id()
    created = int(time.time())

    metrics.requests_served += 1
    t_request = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  [completions] request={request_id}")
    print(f"  [completions] prompt_tokens={len(prompt_tokens)}, "
          f"max_tokens={max_tokens}")

    if stream:
        return StreamingResponse(
            _stream_completion(request_id, model, created, prompt_tokens,
                               max_tokens, temperature),
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
    total_gen_time = 0.0
    async for _tid, token_text, tok_time in _generate_tokens(
        prompt_tokens, max_tokens, temperature
    ):
        output_parts.append(token_text)
        completion_tokens += 1
        total_gen_time += tok_time

    request_time = time.monotonic() - t_request
    avg_tok_s = completion_tokens / total_gen_time if total_gen_time > 0 else 0.0

    print(f"  [completions] done: {completion_tokens} tokens in "
          f"{request_time:.2f}s ({avg_tok_s:.2f} tok/s)")

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
        "lithos_timing": {
            "total_seconds": round(request_time, 3),
            "tokens_per_second": round(avg_tok_s, 3),
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
