#!/usr/bin/env python3
"""
Continuous token generation loop for Lithos.

Prefills the prompt tokens through all 64 layers, then enters a decode loop
generating tokens one at a time until a stop condition is met.

Usage:
    python3 generate_loop.py "The capital of France is" --max-tokens 20
    python3 generate_loop.py "The capital of France is" --max-tokens 50 --temperature 0.7
"""

from __future__ import annotations

import argparse
import ctypes
import math
import numpy as np
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_first_token import (
    InferenceEngine,
    HIDDEN_DIM,
    NUM_LAYERS,
    bf16_to_f32,
    download_f32,
    rms_norm_cpu,
    upload_f32,
    banner,
)

EOS_TOKEN_ID = 248044
VOCAB_SIZE = 248320


def sample_token(logits: np.ndarray, temperature: float = 0.0) -> int:
    """Sample a token from logits.

    temperature=0.0 (or <=0): greedy argmax
    temperature>0: softmax sampling with temperature scaling
    """
    if temperature <= 0.0:
        return int(np.argmax(logits))

    # Temperature scaling
    scaled = logits / temperature

    # Numerically stable softmax
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def compute_lm_head(engine: InferenceEngine, x_normed: np.ndarray) -> np.ndarray:
    """Project the final hidden state through lm_head to get logits.

    Processes in chunks to avoid a huge memory allocation.
    """
    lm_head_raw = bytes(engine.model.weight_bytes("lm_head.weight"))
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    CHUNK = 8192
    for start in range(0, VOCAB_SIZE, CHUNK):
        end = min(start + CHUNK, VOCAB_SIZE)
        chunk_size = end - start
        offset = start * HIDDEN_DIM * 2  # 2 bytes per fp16
        chunk_bytes = chunk_size * HIDDEN_DIM * 2
        w_chunk = np.frombuffer(
            lm_head_raw[offset:offset + chunk_bytes],
            dtype=np.float16,
        ).reshape(chunk_size, HIDDEN_DIM).astype(np.float32)
        logits[start:end] = w_chunk @ x_normed
    return logits


def run_token_through_layers(engine: InferenceEngine, token_id: int,
                             phase: int = 3) -> np.ndarray:
    """Embed a single token and run it through all 64 layers.

    Returns the final hidden state vector [5120] before lm_head.
    """
    # Embed
    x = engine.gpu_embed(token_id)

    # Run all 64 layers
    for layer_idx in range(NUM_LAYERS):
        x = engine.run_layer(layer_idx, x, phase)
        # Early abort on numerical issues
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"\n  ERROR: NaN/Inf at layer {layer_idx}, aborting")
            return None

    # Final RMSNorm
    x = rms_norm_cpu(x, engine.final_norm_w, engine.epsilon)
    return x


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lithos continuous token generation",
    )
    parser.add_argument("prompt", type=str, help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy, default: 0.0)")
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3],
                        help="Inference phase: 1=MLP-only, 2=+FullAttn, 3=+DeltaNet (default: 3)")
    args = parser.parse_args()

    # Handle ctrl-c gracefully
    interrupted = False

    def sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    # Initialize engine
    engine = InferenceEngine()

    prompt = args.prompt
    max_tokens = args.max_tokens
    temperature = args.temperature
    phase = args.phase

    token_ids = engine.tok.encode(prompt)
    num_prompt_tokens = len(token_ids)

    phase_names = {1: "MLP-only", 2: "+FullAttn", 3: "+DeltaNet (complete)"}
    banner(f"Generate Loop -- phase {phase}: {phase_names[phase]}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Prompt tokens ({num_prompt_tokens}): {token_ids}")
    print(f"  Max generate: {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  EOS token ID: {EOS_TOKEN_ID}")

    # ================================================================
    # Phase 1: PREFILL -- process all prompt tokens through the model
    # ================================================================
    banner("Prefill")
    t_total_start = time.monotonic()

    # Process each prompt token through all 64 layers.
    # The last token's hidden state is what we use for the first prediction.
    # Without KV cache, earlier tokens don't directly influence later ones
    # through attention, but the model still processes each token individually.
    for i, tid in enumerate(token_ids):
        tok_text = engine.tok.decode([tid])
        t_tok = time.monotonic()

        x_normed = run_token_through_layers(engine, tid, phase)
        if x_normed is None:
            print(f"  Prefill failed at token {i}")
            engine.close()
            return 1

        elapsed = time.monotonic() - t_tok
        print(f"  Prefill [{i+1}/{num_prompt_tokens}]: "
              f"token={tid} '{tok_text}' -- {elapsed:.2f}s")

        if interrupted:
            print("\n  Interrupted during prefill.")
            engine.close()
            return 130

    t_prefill_done = time.monotonic()
    t_first_token_start = time.monotonic()

    # x_normed now holds the final hidden state of the last prompt token.
    # Compute logits to get the first generated token.
    logits = compute_lm_head(engine, x_normed)
    first_token_id = sample_token(logits, temperature)
    first_token_text = engine.tok.decode([first_token_id])

    t_first_token = time.monotonic() - t_first_token_start
    t_ttft = time.monotonic() - t_total_start  # time to first token

    # ================================================================
    # Phase 2: DECODE -- generate tokens one at a time
    # ================================================================
    banner("Decode")
    print(f"  Prompt: {prompt}", end="", flush=True)
    print(first_token_text, end="", flush=True)

    generated_ids = [first_token_id]
    decode_times = []
    current_token_id = first_token_id

    # Check stop conditions for first token
    if current_token_id == EOS_TOKEN_ID:
        print("\n  [EOS]", flush=True)
    else:
        # Continue generating
        for step in range(1, max_tokens):
            if interrupted:
                break

            t_step = time.monotonic()

            # Run current token through all layers
            x_normed = run_token_through_layers(engine, current_token_id, phase)
            if x_normed is None:
                print("\n  [numerical error, stopping]")
                break

            # Project to logits and sample
            logits = compute_lm_head(engine, x_normed)
            next_token_id = sample_token(logits, temperature)
            next_token_text = engine.tok.decode([next_token_id])

            t_step_done = time.monotonic() - t_step
            decode_times.append(t_step_done)

            generated_ids.append(next_token_id)
            current_token_id = next_token_id

            # Stream output
            print(next_token_text, end="", flush=True)

            # Stop conditions
            if next_token_id == EOS_TOKEN_ID:
                print("\n  [EOS]", flush=True)
                break

    print()  # newline after streaming

    # ================================================================
    # Timing report
    # ================================================================
    t_total = time.monotonic() - t_total_start
    t_prefill_total = t_prefill_done - t_total_start
    num_generated = len(generated_ids)

    banner("Timing Report")
    print(f"  Prompt tokens:     {num_prompt_tokens}")
    print(f"  Generated tokens:  {num_generated}")
    print(f"  Prefill time:      {t_prefill_total:.2f}s "
          f"({num_prompt_tokens / max(t_prefill_total, 1e-9):.2f} tok/s)")
    print(f"  Time to first token (TTFT): {t_ttft:.2f}s")

    if decode_times:
        avg_decode = sum(decode_times) / len(decode_times)
        decode_tps = 1.0 / avg_decode if avg_decode > 0 else 0
        print(f"  Decode tokens:     {len(decode_times)} (excluding first)")
        print(f"  Avg decode time:   {avg_decode:.2f}s/token")
        print(f"  Decode throughput: {decode_tps:.2f} tok/s")
    else:
        print(f"  Decode:            (only 1 token generated)")

    print(f"  Total time:        {t_total:.2f}s")
    print(f"  Overall:           {num_generated / max(t_total, 1e-9):.2f} tok/s")

    # Print the full generated text
    full_output = engine.tok.decode(generated_ids)
    banner("Output")
    print(f"  {prompt}{full_output}")

    if interrupted:
        print("\n  [interrupted by user]")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
