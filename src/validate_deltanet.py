#!/usr/bin/env python3
"""
Validate DeltaNet recurrence in generate_first_token.py against
the reference Qwen3NextGatedDeltaNet implementation from transformers.

Loads layer 0 weights, runs a single token through both code paths,
and compares intermediate values to identify divergences.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import LithosModel
from tokenizer import Tokenizer

MODEL_DIR = "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"
HIDDEN_DIM = 5120
GROUP_SIZE = 128
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM     # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM   # 6144
CONV_DIM = KEY_DIM * 2 + VALUE_DIM     # 10240
EPS = 1e-6


def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def dequant_gptq_matvec(model: LithosModel, prefix: str, x: np.ndarray, K: int, N: int) -> np.ndarray:
    """Dequantize GPTQ weights and do matvec on CPU (reference implementation)."""
    qw_raw = bytes(model.weight_bytes(f"{prefix}.qweight"))
    sc_raw = bytes(model.weight_bytes(f"{prefix}.scales"))

    qweight = np.frombuffer(qw_raw, dtype=np.int32).reshape(K // 8, N)
    scales = np.frombuffer(sc_raw, dtype=np.float16).astype(np.float32).reshape(K // GROUP_SIZE, N)

    output = np.zeros(N, dtype=np.float32)
    for row_group in range(K // GROUP_SIZE):
        scale_row = scales[row_group]
        for sub in range(GROUP_SIZE // 8):
            packed_row_idx = row_group * (GROUP_SIZE // 8) + sub
            packed = qweight[packed_row_idx]  # int32, 8 x 4-bit values
            for bit in range(8):
                k_idx = row_group * GROUP_SIZE + sub * 8 + bit
                w4 = (int(packed[0]) >> (bit * 4)) & 0xF if N == 1 else None
                # Vectorized: extract all N values at once
                break
            break
        break

    # Simpler vectorized approach
    output = np.zeros(N, dtype=np.float32)
    for k in range(K):
        group_idx = k // GROUP_SIZE
        packed_idx = k // 8
        bit_pos = k % 8

        # Extract 4-bit weights for all N output columns
        packed_row = qweight[packed_idx].view(np.uint32)
        w4 = (packed_row >> (bit_pos * 4)) & 0xF
        w4_signed = w4.astype(np.float32) - 7.0  # zero_point = 7 for sym quant
        dequantized = w4_signed * scales[group_idx]

        output += x[k] * dequantized

    return output


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -80, 80))))


def l2norm(x, eps=1e-6):
    """L2 normalize along last axis."""
    inv_norm = 1.0 / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inv_norm


def rms_norm(x, eps=1e-6):
    """RMS norm (no weight)."""
    return x / np.sqrt(np.mean(x ** 2) + eps)


def compare(name, a, b, rtol=1e-3, atol=1e-3):
    """Compare two arrays and report."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    rel_diff = max_diff / (max(a_norm, b_norm, 1e-12))
    match = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "MATCH" if match else "DIVERGE"
    print(f"  {name}: {status}  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  "
          f"rel={rel_diff:.6e}  norm_a={a_norm:.4f}  norm_b={b_norm:.4f}")
    if not match:
        # Show first few divergent elements
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    worst at {idx}: ours={a[idx]:.8f}  ref={b[idx]:.8f}")
    return match


def main():
    print("=" * 72)
    print("  DeltaNet Validation: generate_first_token.py vs Reference")
    print("=" * 72)

    model = LithosModel(MODEL_DIR)
    tok = Tokenizer(MODEL_DIR)
    epsilon = model.config.rms_norm_eps

    # Use first token of "The capital of France is"
    token_ids = tok.encode("The capital of France is")
    tid = token_ids[0]
    print(f"\nToken: '{tok.decode([tid])}' (id={tid})")

    # --- Step 0: Get embedding ---
    embed_raw = bytes(model.weight_bytes("model.language_model.embed_tokens.weight"))
    embed_f16 = np.frombuffer(embed_raw, dtype=np.float16).reshape(-1, HIDDEN_DIM)
    x_embed = embed_f16[tid].astype(np.float32)
    print(f"Embedding norm: {np.linalg.norm(x_embed):.4f}")

    # --- Step 1: Input layernorm ---
    norm_w_raw = bytes(model.weight_bytes("model.language_model.layers.0.input_layernorm.weight"))
    # Check if this is the Qwen3Next style (1+w) or standard (w)
    # From config, the norm is Qwen3NextRMSNorm which uses (1+w)*norm(x)
    # But our generate_first_token uses standard w*norm(x)
    # Let's check norm weight dtype
    ti = model.weight_info("model.language_model.layers.0.input_layernorm.weight")
    print(f"Norm weight dtype: {ti.dtype}")
    if ti.dtype == "BF16":
        norm_w = bf16_to_f32(norm_w_raw)
    elif ti.dtype == "F16":
        norm_w = np.frombuffer(norm_w_raw, dtype=np.float16).astype(np.float32)
    else:
        norm_w = np.frombuffer(norm_w_raw, dtype=np.float32).copy()

    # Reference: Qwen3NextRMSNorm uses (1 + weight) * norm(x)
    x_normed_ref = rms_norm(x_embed, epsilon) * (1.0 + norm_w)
    # Our code: weight * norm(x)
    x_normed_ours = rms_norm(x_embed, epsilon) * norm_w
    print(f"\nNorm comparison:")
    print(f"  Our norm output norm: {np.linalg.norm(x_normed_ours):.4f}")
    print(f"  Ref norm output norm: {np.linalg.norm(x_normed_ref):.4f}")
    # Note: if weights were saved from a model using (1+w), the stored w is the raw parameter.
    # The kernel norm code might already handle this. Let's check which one the GPU norm uses.

    # For now, use the input to DeltaNet layer (we'll compare the DeltaNet internals)
    # We'll use the reference norm since that matches the model's actual computation
    normed_input = x_normed_ref  # Use this for the reference path

    layer_idx = 0
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

    print(f"\n{'='*72}")
    print(f"  DeltaNet Layer {layer_idx} Internals")
    print(f"{'='*72}")

    # ====================================================================
    # QKV Projection
    # ====================================================================
    print("\n--- QKV Projection ---")
    qkv = dequant_gptq_matvec(model, f"{prefix}.in_proj_qkv", normed_input, HIDDEN_DIM, CONV_DIM)
    print(f"QKV raw output norm: {np.linalg.norm(qkv):.4f}")
    print(f"QKV[0:5]: {qkv[:5]}")

    # Z Projection
    z = dequant_gptq_matvec(model, f"{prefix}.in_proj_z", normed_input, HIDDEN_DIM, VALUE_DIM)
    print(f"Z raw output norm: {np.linalg.norm(z):.4f}")

    # ====================================================================
    # Conv1d (first token, zero padding)
    # ====================================================================
    print("\n--- Conv1d ---")
    conv_w_raw = bytes(model.weight_bytes(f"{prefix}.conv1d.weight"))
    conv_w = bf16_to_f32(conv_w_raw).reshape(CONV_DIM, 1, 4)

    # For first token with zero padding, conv output = qkv * conv_w[:, 0, 3]
    # (only the last kernel position touches the actual input)
    qkv_conv_no_silu = qkv * conv_w[:, 0, 3]
    qkv_conv_with_silu = silu(qkv_conv_no_silu)

    print(f"Conv output (no SiLU) norm: {np.linalg.norm(qkv_conv_no_silu):.4f}")
    print(f"Conv output (with SiLU) norm: {np.linalg.norm(qkv_conv_with_silu):.4f}")

    # BUG #1: Our code is MISSING SiLU after conv1d
    print("\n  >> BUG #1: generate_first_token.py does NOT apply SiLU after conv1d")
    print("     Reference code: F.silu(self.conv1d(mixed_qkv))")
    print("     Our code: just qkv * conv_w[:, 0, 3]")

    # Reference uses SiLU
    qkv_post_conv = qkv_conv_with_silu

    # Split Q, K, V after conv
    q = qkv_post_conv[:KEY_DIM].reshape(NUM_K_HEADS, HEAD_K_DIM)
    k = qkv_post_conv[KEY_DIM:KEY_DIM*2].reshape(NUM_K_HEADS, HEAD_K_DIM)
    v = qkv_post_conv[KEY_DIM*2:].reshape(NUM_V_HEADS, HEAD_V_DIM)

    print(f"\nQ shape: {q.shape}, norm: {np.linalg.norm(q):.4f}")
    print(f"K shape: {k.shape}, norm: {np.linalg.norm(k):.4f}")
    print(f"V shape: {v.shape}, norm: {np.linalg.norm(v):.4f}")

    # ====================================================================
    # Beta and Alpha (a) projections
    # ====================================================================
    print("\n--- Beta and Alpha projections ---")

    # Load weights
    a_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_a.weight"))
    a_weight = bf16_to_f32(a_w_raw).reshape(NUM_V_HEADS, HIDDEN_DIM)

    b_w_raw = bytes(model.weight_bytes(f"{prefix}.in_proj_b.weight"))
    b_weight = bf16_to_f32(b_w_raw).reshape(NUM_V_HEADS, HIDDEN_DIM)

    a_proj = a_weight @ normed_input  # [48]
    b_proj = b_weight @ normed_input  # [48]

    print(f"in_proj_a output: mean={a_proj.mean():.4f} std={a_proj.std():.4f}")
    print(f"in_proj_b output: mean={b_proj.mean():.4f} std={b_proj.std():.4f}")

    # Reference: b -> beta = sigmoid(b), a -> used in g = -exp(A_log) * softplus(a + dt_bias)
    # Our code (WRONG): in_proj_a -> beta, in_proj_b -> decay
    # Correct: in_proj_a -> a (for decay), in_proj_b -> b (for beta)

    # BUG #2: Our code swaps in_proj_a and in_proj_b
    print("\n  >> BUG #2: generate_first_token.py SWAPS in_proj_a and in_proj_b")
    print("     Reference: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)")
    print("     Our code uses in_proj_a for beta and in_proj_b for decay -- SWAPPED")

    # Correct assignment:
    # in_proj_a -> a (used for gating/decay)
    # in_proj_b -> b (used for beta)
    beta_ref = 1.0 / (1.0 + np.exp(-b_proj.clip(-80, 80)))  # sigmoid(b_proj)
    beta_ours = 1.0 / (1.0 + np.exp(-a_proj.clip(-80, 80)))  # sigmoid(a_proj) -- WRONG

    print(f"\nBeta (correct, from in_proj_b): mean={beta_ref.mean():.4f} std={beta_ref.std():.4f}")
    print(f"Beta (our code, from in_proj_a): mean={beta_ours.mean():.4f} std={beta_ours.std():.4f}")
    compare("beta", beta_ours, beta_ref)

    # ====================================================================
    # Decay (g) computation
    # ====================================================================
    print("\n--- Decay (g) computation ---")

    a_log_raw = bytes(model.weight_bytes(f"{prefix}.A_log"))
    A_log = np.frombuffer(a_log_raw, dtype=np.float32).copy()

    dt_bias_raw = bytes(model.weight_bytes(f"{prefix}.dt_bias"))
    dt_bias = bf16_to_f32(dt_bias_raw)

    # Reference: g = -exp(A_log) * softplus(a + dt_bias)
    # where 'a' comes from in_proj_a
    g_ref = -np.exp(A_log) * np.log1p(np.exp(np.clip(a_proj + dt_bias, -20, 20)))

    # Our code computes: dt = in_proj_b @ x + dt_bias, dt = softplus(dt), decay = exp(-exp(A_log) * dt)
    # Which is: decay = exp(-exp(A_log) * softplus(in_proj_b @ x + dt_bias))
    # But should be: decay = exp(g) = exp(-exp(A_log) * softplus(in_proj_a @ x + dt_bias))
    dt_ours = b_proj + dt_bias  # Our code uses in_proj_b (wrong)
    dt_ours_sp = np.log1p(np.exp(np.clip(dt_ours, -20, 20)))
    decay_ours = np.exp(-np.exp(A_log) * dt_ours_sp)

    dt_correct = a_proj + dt_bias  # Should use in_proj_a
    dt_correct_sp = np.log1p(np.exp(np.clip(dt_correct, -20, 20)))
    decay_correct = np.exp(-np.exp(A_log) * dt_correct_sp)  # = exp(g_ref)

    print(f"g_ref: mean={g_ref.mean():.4f} std={g_ref.std():.4f}")
    print(f"decay (correct, exp(g)): mean={decay_correct.mean():.4f}")
    print(f"decay (our code, wrong input): mean={decay_ours.mean():.4f}")
    compare("decay", decay_ours, decay_correct)

    # ====================================================================
    # L2 Normalization on Q and K
    # ====================================================================
    print("\n--- L2 Normalization ---")

    # BUG #3: Our code does NOT apply L2 norm to Q and K
    print("  >> BUG #3: generate_first_token.py does NOT apply L2 norm to Q and K")
    print("     Reference: use_qk_l2norm_in_kernel=True -> l2norm(q), l2norm(k)")

    # Expand Q and K: 16 key heads -> 48 value heads (3x repeat)
    q_expanded = np.repeat(q, NUM_V_HEADS // NUM_K_HEADS, axis=0)  # [48, 128]
    k_expanded = np.repeat(k, NUM_V_HEADS // NUM_K_HEADS, axis=0)  # [48, 128]

    q_l2 = l2norm(q_expanded)  # [48, 128]
    k_l2 = l2norm(k_expanded)  # [48, 128]

    print(f"Q before l2norm: per-head norms = {np.linalg.norm(q_expanded, axis=1)[:3]}...")
    print(f"Q after l2norm:  per-head norms = {np.linalg.norm(q_l2, axis=1)[:3]}...")
    print(f"K before l2norm: per-head norms = {np.linalg.norm(k_expanded, axis=1)[:3]}...")
    print(f"K after l2norm:  per-head norms = {np.linalg.norm(k_l2, axis=1)[:3]}...")

    # ====================================================================
    # Query scaling
    # ====================================================================
    print("\n--- Query Scaling ---")
    scale = 1.0 / np.sqrt(HEAD_K_DIM)
    print(f"  >> BUG #4: Missing 1/sqrt(d) scaling: scale = {scale:.6f}")
    q_scaled = q_l2 * scale

    # ====================================================================
    # DeltaNet Recurrence (zero state, single token)
    # ====================================================================
    print("\n--- DeltaNet Recurrence (single token, zero state) ---")

    # Reference recurrence for single token with zero state:
    # S = 0
    # g_t = g[h]  (already computed)
    # S = S * exp(g_t) = 0
    # kv_mem = 0
    # delta = (v - 0) * beta = v * beta
    # S = 0 + outer(k, delta) = outer(k, v * beta)
    # output = q @ S = dot(q, k) * v * beta   (per-head)

    output_ref = np.zeros((NUM_V_HEADS, HEAD_V_DIM), dtype=np.float32)
    output_ours_buggy = np.zeros((NUM_V_HEADS, HEAD_V_DIM), dtype=np.float32)

    for h in range(NUM_V_HEADS):
        # Reference (correct): uses l2norm q/k, scaling, correct beta
        q_h = q_scaled[h]
        k_h = k_l2[h]
        v_h = v[h]
        b_h = beta_ref[h]

        # With zero state: output = beta * dot(q_scaled, k_l2) * v
        qk_dot = np.dot(q_h, k_h)
        output_ref[h] = b_h * qk_dot * v_h

        # Our code (buggy): no l2norm, no scaling, wrong beta
        key_idx = h // (NUM_V_HEADS // NUM_K_HEADS)
        q_h_ours = q[key_idx]
        k_h_ours = k[key_idx]
        # Also uses conv output WITHOUT SiLU (bug #1)
        # Also uses wrong beta (bug #2)
        output_ours_buggy[h] = beta_ours[h] * np.dot(q_h_ours, k_h_ours) * v_h

    print(f"Reference output norm: {np.linalg.norm(output_ref):.6f}")
    print(f"Our (buggy) output norm: {np.linalg.norm(output_ours_buggy):.6f}")
    compare("recurrence_output", output_ours_buggy, output_ref)

    # Now compute what our code ACTUALLY produces (with all bugs from generate_first_token.py)
    print("\n--- Full comparison: our code vs reference ---")

    # Our code path (reproducing generate_first_token.py exactly):
    qkv_ours = dequant_gptq_matvec(model, f"{prefix}.in_proj_qkv", normed_input, HIDDEN_DIM, CONV_DIM)
    # Bug #1: no SiLU
    qkv_ours_conv = qkv_ours * conv_w[:, 0, 3]
    q_ours = qkv_ours_conv[:KEY_DIM].reshape(NUM_K_HEADS, HEAD_K_DIM)
    k_ours = qkv_ours_conv[KEY_DIM:KEY_DIM*2].reshape(NUM_K_HEADS, HEAD_K_DIM)
    v_ours = qkv_ours_conv[KEY_DIM*2:].reshape(NUM_V_HEADS, HEAD_V_DIM)

    output_ours_full = np.zeros((NUM_V_HEADS, HEAD_V_DIM), dtype=np.float32)
    for h in range(NUM_V_HEADS):
        key_idx = h // (NUM_V_HEADS // NUM_K_HEADS)
        # Bug #2: wrong beta (from in_proj_a instead of in_proj_b)
        # Bug #3: no l2norm
        # Bug #4: no scaling
        output_ours_full[h] = beta_ours[h] * np.dot(q_ours[key_idx], k_ours[key_idx]) * v_ours[h]

    print(f"\nOur full output norm: {np.linalg.norm(output_ours_full):.6f}")
    print(f"Reference output norm: {np.linalg.norm(output_ref):.6f}")
    compare("full_output", output_ours_full, output_ref)

    # ====================================================================
    # Group Norm
    # ====================================================================
    print("\n--- Group Norm ---")
    norm_w_raw2 = bytes(model.weight_bytes(f"{prefix}.norm.weight"))
    head_norm_w = np.frombuffer(norm_w_raw2, dtype=np.float32).copy()
    print(f"Norm weight: shape={head_norm_w.shape}, mean={head_norm_w.mean():.4f}")

    # Reference: Qwen3NextRMSNormGated: norm(x) * weight * silu(z)
    # (weight initialized to ones, stored as-is)
    # Our code: (x / rms) * weight * silu(z) -- same formula, correct

    z_heads = z.reshape(NUM_V_HEADS, HEAD_V_DIM)
    z_gate = silu(z_heads)

    # Apply to reference output
    normed_ref = np.zeros_like(output_ref)
    for h in range(NUM_V_HEADS):
        rms = np.sqrt(np.mean(output_ref[h] ** 2) + EPS)
        normed_ref[h] = (output_ref[h] / rms) * head_norm_w * z_gate[h]

    # Apply to our (buggy) output
    normed_ours = np.zeros_like(output_ours_full)
    for h in range(NUM_V_HEADS):
        rms = np.sqrt(np.mean(output_ours_full[h] ** 2) + EPS)
        normed_ours[h] = (output_ours_full[h] / rms) * head_norm_w * z_gate[h]

    print(f"Normed ref norm: {np.linalg.norm(normed_ref):.6f}")
    print(f"Normed ours norm: {np.linalg.norm(normed_ours):.6f}")
    compare("post_norm", normed_ours, normed_ref)

    # ====================================================================
    # Output Projection
    # ====================================================================
    print("\n--- Output Projection ---")
    out_ref = dequant_gptq_matvec(model, f"{prefix}.out_proj",
                                   normed_ref.flatten(), VALUE_DIM, HIDDEN_DIM)
    out_ours = dequant_gptq_matvec(model, f"{prefix}.out_proj",
                                    normed_ours.flatten(), VALUE_DIM, HIDDEN_DIM)
    print(f"Output ref norm: {np.linalg.norm(out_ref):.6f}")
    print(f"Output ours norm: {np.linalg.norm(out_ours):.6f}")
    compare("out_proj", out_ours, out_ref)

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"\n{'='*72}")
    print("  SUMMARY OF BUGS")
    print(f"{'='*72}")
    print("""
BUG #1: Missing SiLU activation after conv1d
  Location: generate_first_token.py line ~419
  Our code:  qkv_conv = qkv * conv_w[:, 0, 3]
  Should be: qkv_conv = silu(qkv * conv_w[:, 0, 3])
  Reference: F.silu(self.conv1d(mixed_qkv))

BUG #2: Swapped in_proj_a and in_proj_b usage
  Location: generate_first_token.py lines ~429-440
  Our code:  beta = sigmoid(in_proj_a @ x)   -- WRONG
             dt = in_proj_b @ x + dt_bias     -- WRONG
  Should be: beta = sigmoid(in_proj_b @ x)
             g = -exp(A_log) * softplus(in_proj_a @ x + dt_bias)
  Reference: beta = b.sigmoid() where b is from in_proj_b
             g = -A_log.exp() * softplus(a + dt_bias) where a is from in_proj_a

BUG #3: Missing L2 normalization on Q and K
  Location: generate_first_token.py line ~422
  Our code:  q, k used directly after conv1d
  Should be: q = l2norm(q), k = l2norm(k)
  Reference: use_qk_l2norm_in_kernel=True -> q = l2norm(q), k = l2norm(k)

BUG #4: Missing 1/sqrt(d) scaling on Q
  Location: generate_first_token.py line ~477
  Our code:  qk_dot = dot(q, k)
  Should be: qk_dot = dot(q * (1/sqrt(128)), k) = dot(q, k) / sqrt(128)
  Reference: query = query * scale  where scale = 1/sqrt(head_k_dim)

BUG #5: Q and K head expansion should use repeat_interleave (not repeat)
  Location: generate_first_token.py line ~467-476
  Our code:  key_idx = h // value_heads_per_key (correct indexing for repeat)
  Should be: expand Q and K from 16 to 48 heads via repeat_interleave
             then index directly by h
  Reference: query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
  Note: The indexing h // 3 is equivalent to repeat_interleave, so this is
        actually correct. Not a bug.

BUG #5 (CONFIRMED): RMSNorm uses (1+w) in Qwen3NextRMSNorm
  Location: generate_first_token.py GPU norm kernel / rms_norm_cpu
  Reference Qwen3NextRMSNorm: output = norm(x) * (1 + weight)
  Our code:  output = norm(x) * weight
  The stored weights have mean ~-0.027 (near zero), so using them directly
  instead of (1 + w) suppresses the output by ~10x.
  Verified: our norm output has norm 6.55 vs reference 76.25.
  Fix: change all norm computations to use (1 + w) * norm(x).
  Note: The DeltaNet gated norm (Qwen3NextRMSNormGated) uses plain w
        (initialized to 1), NOT (1+w). So only the layer norms are affected.
""")

    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
