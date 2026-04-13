# Hybrid Architectures in Lithos

How the Lithos compiler emits a single cooperative megakernel for models
that interleave recurrent (DeltaNet / linear) attention with full softmax
attention on a fixed periodic schedule. Concrete target: Qwen 3.5 27B.

---

## 1. Why Hybrid Models Matter

A pure transformer pays O(N) memory and O(N) compute per token at position
N, because softmax attention reads the entire KV cache on every step. A
pure recurrent model (DeltaNet, Mamba, RWKV) pays O(1) per token — the
state matrix is fixed size and updated in place — but cannot reach back
into distant tokens with full fidelity; the state is a lossy summary.

Hybrid models resolve the trade-off by scheduling them:

- **DeltaNet layers (majority).** Recurrent, O(1) per token, O(1) memory.
  Constant work regardless of context length. Good at local structure and
  short-range composition.
- **Full attention layers (minority).** O(N) per token, O(N) memory, but
  give every position direct access to every prior token. Inserted
  periodically so the residual stream is re-anchored to exact past tokens
  every few layers.

Qwen 3.5 27B's schedule: **3 DeltaNet + 1 full attention, repeating 16
times, 64 layers total**. The model spends 75% of its layers in O(1)
recurrence and 25% in O(N) global attention. For a 32k-token conversation,
only 16 layers actually materialise a KV cache — the other 48 layers store
only a 128x128 state matrix per head.

The compiler's job is to turn this schedule into one instruction stream
with no branches, no runtime layer dispatch, no conditionals on layer
type. Every layer is laid out statically in `.text` at compile time.

---

## 2. How the Compiler Reads the Schedule

### 2.1 Input

`config.json` (inside the safetensors bundle) contains:

```
"layer_types": [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ...
    "linear_attention", "linear_attention", "linear_attention", "full_attention"
],
"full_attention_interval": 4,
"num_hidden_layers": 64
```

The compiler's safetensors reader (Section 7 of `compiler.ls`) parses this
JSON at compile time. The `layer_types` array becomes a 64-entry u8 array
in the compiler's heap: `0 = linear_attention, 1 = full_attention`.

### 2.2 Dispatch loop (compile-time, not runtime)

```
for layer_idx 0 64 1
    type = layer_types [ layer_idx ]
    if== type 0
        emit_deltanet_layer layer_idx
    endif
    if== type 1
        emit_full_attention_layer layer_idx
    endif
    emit_grid_sync
endfor
```

This loop runs **inside the compiler**, on ARM64. Its output is 64 fully
expanded instruction sequences concatenated into the forward megakernel's
`.text` section. At GPU runtime there is no layer dispatch — the program
counter simply walks through 64 layers of already-flattened SASS.

### 2.3 Template selection

Each branch inlines a composition:
- `emit_deltanet_layer layer_idx` flattens `deltanet_layer` with
  `layer_idx` bound, resolving every `project` primitive against the
  DeltaNet weight names.
- `emit_full_attention_layer layer_idx` flattens `full_attention_layer`,
  resolving against the full-attention weight names.

Both templates share identical **residual, RMSNorm, MLP, and output
projection** code. Only the attention block differs. The compiler emits
them as two distinct compositions (no runtime conditional) because the
data shapes and state buffers differ.

---

## 3. Per-Layer Composition Templates

Both compositions take `x` (layer input activation vector) and `X` (the
residual stream — updated in place). `layer_idx` is a compile-time
integer; it selects which weight tensors to dequantize into immediates
and which state/cache slice to address.

### 3.1 DeltaNet layer

```
deltanet_layer x X layer_idx :
    \\ Attention sub-block
    rmsnorm x
    project W_q[layer_idx]    x q        \\ 5120 -> 2048  (16 heads x 128)
    project W_k[layer_idx]    x k        \\ 5120 -> 2048
    project W_v[layer_idx]    x v        \\ 5120 -> 6144  (48 heads x 128)
    project W_z[layer_idx]    x z        \\ 5120 -> 6144  (output gate)
    project W_beta[layer_idx] x beta     \\ 5120 -> 48    (per value-head)
    project W_decay[layer_idx] x decay   \\ 5120 -> 48

    \\ DeltaNet recurrence — see inference/deltanet_fused.ls
    \\ Reads state[layer_idx, 16, 3, 128, 128], updates in place
    deltanet_fused q k v z decay beta state[layer_idx] out

    project W_o[layer_idx] out X         \\ residual add into X

    \\ MLP sub-block (shared with full_attention_layer)
    rmsnorm X
    project W_up[layer_idx]   X up
    project W_gate[layer_idx] X gate
    silu gate
    gate * up
    project W_down[layer_idx] X X        \\ residual add into X
```

### 3.2 Full attention layer

```
full_attention_layer x X layer_idx :
    \\ Attention sub-block
    rmsnorm x
    project W_q_full[layer_idx]    x q     \\ 5120 -> 2048  (16 heads x 128)
    project W_k_full[layer_idx]    x k     \\ 5120 -> 1024  (8 KV heads x 128, GQA)
    project W_v_full[layer_idx]    x v     \\ 5120 -> 1024
    project W_gate_attn[layer_idx] x gate  \\ 5120 -> 2048  (attn_output_gate)

    mrope q position
    mrope k position

    \\ Append K, V to cache at [layer_idx, position, :]
    kv_write kv_cache[layer_idx] k v position

    \\ Softmax attention over positions 0..=position
    attention_score_full  q kv_cache[layer_idx].k position scores
    attention_output_full scores kv_cache[layer_idx].v gate out

    project W_o_full[layer_idx] out X      \\ residual add into X

    \\ MLP sub-block — identical to deltanet_layer
    rmsnorm X
    project W_up[layer_idx]   X up
    project W_gate[layer_idx] X gate
    silu gate
    gate * up
    project W_down[layer_idx] X X
```

Shapes (Qwen 3.5 27B):
- `hidden = 5120`, `q_heads = 16`, `kv_heads = 8` (full) / `kv_heads = 16`
  (DeltaNet key side), `head_dim = 128`.
- DeltaNet value heads per key head = 3 (GQA ratio 3), 48 value heads.
- Full-attention GQA ratio = 2.
- MLP intermediate `= 25600` (up and gate project to 25600, down projects
  back to 5120).

---

## 4. Weight Naming Conventions

Safetensors tensor names differ between the two layer flavours. The
compiler's template resolver uses `layer_idx` and the layer type to pick
the right key:

| Role        | DeltaNet tensor key                              | Full-attn tensor key                             |
|-------------|--------------------------------------------------|--------------------------------------------------|
| Q proj      | `model.layers.{N}.linear_attn.q_proj.qweight`    | `model.layers.{N}.self_attn.q_proj.qweight`      |
| K proj      | `model.layers.{N}.linear_attn.k_proj.qweight`    | `model.layers.{N}.self_attn.k_proj.qweight`      |
| V proj      | `model.layers.{N}.linear_attn.v_proj.qweight`    | `model.layers.{N}.self_attn.v_proj.qweight`      |
| Z / gate    | `model.layers.{N}.linear_attn.z_proj.qweight`    | `model.layers.{N}.self_attn.gate_proj.qweight`   |
| beta        | `model.layers.{N}.linear_attn.b_proj.qweight`    | —                                                |
| decay       | `model.layers.{N}.linear_attn.a_proj.qweight`    | —                                                |
| O proj      | `model.layers.{N}.linear_attn.o_proj.qweight`    | `model.layers.{N}.self_attn.o_proj.qweight`      |
| pre-attn LN | `model.layers.{N}.input_layernorm.weight`        | `model.layers.{N}.input_layernorm.weight`        |
| pre-MLP LN  | `model.layers.{N}.post_attention_layernorm.weight` | `model.layers.{N}.post_attention_layernorm.weight` |
| MLP up      | `model.layers.{N}.mlp.up_proj.qweight`           | `model.layers.{N}.mlp.up_proj.qweight`           |
| MLP gate    | `model.layers.{N}.mlp.gate_proj.qweight`         | `model.layers.{N}.mlp.gate_proj.qweight`         |
| MLP down    | `model.layers.{N}.mlp.down_proj.qweight`         | `model.layers.{N}.mlp.down_proj.qweight`         |

### Autodetection rule

The compiler does **not** trust `config.json` blindly. For each layer N it
probes the safetensors header for the two diagnostic keys:

- `model.layers.{N}.linear_attn.q_proj.qweight` present → DeltaNet.
- `model.layers.{N}.self_attn.q_proj.qweight` present → full attention.

If this probe disagrees with `layer_types[N]`, compilation aborts with a
hard error citing the tensor name and the claimed type. This catches
model-conversion bugs at compile time instead of at inference time.

Associated per-tensor inputs are walked once per layer:
`qweight` (Q4 packed u32), `qzeros`, `scales` (FP16), and, for full
attention, the rope inverse frequencies baked into `rotary_emb.inv_freq`.
Every scalar is dequantized at compile time into an FP32 immediate before
being embedded in an FMUL-IMM or HFMA2 instruction per
`docs/weights-as-code-compiler.md`.

---

## 5. Megakernel Layout

The forward megakernel is a single cooperative cubin. Its `.text` is a
concatenation of statically-scheduled sections, separated by
`grid.sync()` barriers implemented as one instruction per block plus a
membar:

```
Section  0  : token_embed              ~20 KB
Section  1  : layer_00  (DeltaNet)     ~5.0 MB
Section  2  : layer_01  (DeltaNet)     ~5.0 MB
Section  3  : layer_02  (DeltaNet)     ~5.0 MB
Section  4  : layer_03  (full attn)    ~5.3 MB
Section  5  : layer_04  (DeltaNet)
...
Section 64  : layer_63  (full attn)
Section 65  : final_rmsnorm              ~1 KB
Section 66  : lm_head_project          ~80 MB  (5120 x 152064, Q4)
Section 67  : argmax_sample             ~4 KB
```

Each layer section is approximately 5 MB of weight-embedded instruction
stream. The exact size depends on layer type: full-attention layers have
a marginally larger Q/K/V preamble but a smaller softmax kernel, while
DeltaNet layers pay for the fused state update loop. Both are dwarfed by
the MLP (which is identical across layer types and accounts for ~80% of
each layer's instruction bytes).

### Grid sync between layers

Between every pair of consecutive sections the compiler emits:

```
BAR.SYNC.DEFER_BLOCKING  0
MEMBAR.GPU
```

This is the standard sm90 cooperative-grid barrier. The single QMD
dispatch launches the megakernel once; all 64 layers execute inside that
one dispatch separated by grid syncs. One doorbell write per token.

### Dispatch accounting

- **QMD dispatches per token: 1.**
- **Host → GPU doorbell writes per token: 1.**
- **Megakernel entries per token: 1.**
- **Grid syncs per token: 67** (between sections).

No Python. No runtime kernel launch loop. The scheduler does not see 64
kernels — it sees one enormous kernel with an internal static schedule.

---

## 6. KV Cache and Recurrent State Management

Two kinds of persistent per-layer buffers live in HBM (BAR4), allocated
once at startup and updated in place:

### 6.1 Full attention KV cache

```
kv_cache : struct
    k : f16[16, max_seq_len, 8, 128]     \\ [full_layers, seq, kv_heads, head_dim]
    v : f16[16, max_seq_len, 8, 128]
```

- Only 16 of the 64 layers need a KV cache (layers 3, 7, 11, ..., 63).
- The compiler emits a compact index mapping: `full_layer_slot[layer_idx]
  = layer_idx / 4` (since the schedule is perfectly periodic). This lets
  each full-attention section address its own slice without a lookup
  table.
- For `max_seq_len = 32768`: per full layer, K+V = 2 * 32768 * 8 * 128 *
  2 bytes = 128 MB. Across 16 full layers: **2 GB** of KV cache in HBM.
- Append is a single `kv_write` primitive: thread-parallel store of the
  current-token K and V vectors at offset `position`.

### 6.2 DeltaNet recurrent state

```
deltanet_state : f32[48, 16, 3, 128, 128]
                  \\ [deltanet_layers, key_heads, value_heads_per_key, d, d]
```

- 48 DeltaNet layers, each with 16 key heads × 3 value heads × 128×128
  = 48 * 16 * 3 * 128 * 128 * 4 = **4.8 MB per layer**, **230 MB total**.
- The compiler emits the per-layer base pointer as an immediate in the
  `deltanet_fused` expansion. No table lookup at runtime.
- State is read, updated, and written in place every token. It replaces
  the KV cache for these layers: the 128×128 matrix is a compressed
  summary of all past tokens routed through that layer.
- No position dependency — the state matrix itself is the history.

### 6.3 Combined HBM footprint

```
Weights (embedded in .text)   ~14 GB  (27B params, Q4)
KV cache (16 full layers)      2.0 GB  (f16, 32k max_seq)
DeltaNet state (48 layers)    230 MB  (f32)
Activation scratch             ~2 MB  (single token)
                              --------
                              ~16.2 GB
```

Fits comfortably on a single H100 80GB with room for longer sequences or
larger batch.

---

## 7. Position Tracking

A **single `u32 position` counter** is shared across all 64 layers for
the current sequence. It lives in a fixed BAR4 scratch slot, written by
the host before the megakernel dispatch and incremented atomically by
the megakernel after `argmax_sample` writes the next token.

- **Full-attention layers read it** for two purposes:
  1. mRoPE: rotate Q and K by the angle table indexed by `position`.
     Qwen 3.5 uses multi-resolution RoPE with three frequency bands, all
     computed from the same `position`.
  2. KV cache indexing: write K, V at offset `position`; run softmax
     attention over positions `0..=position`.
- **DeltaNet layers never read it.** The state matrix encodes all
  history; there is no positional argument to the recurrence. The mRoPE
  code is not emitted into DeltaNet sections.

At compile time, the compiler binds the `position` pointer as a kernel
parameter passed in a constant-memory slot. Every full-attention section
reads it with a single `LDC` at section entry; DeltaNet sections never
touch it.

For batched or multi-sequence decoding (not yet implemented), `position`
becomes a per-sequence array indexed by `blockIdx.y`, with the grid laid
out as `(cores, batch)`.

---

## 8. Summary — What the Compiler Does

Given a safetensors file plus the `.ls` templates:

1. Parse `config.json` → read `layer_types[64]` and `full_attention_interval`.
2. For each `layer_idx in 0..64`:
   a. Probe tensor names; confirm layer type; fail fast on mismatch.
   b. Select the matching composition (`deltanet_layer` or
      `full_attention_layer`).
   c. Flatten the composition, binding `layer_idx` as a compile-time
      integer, and dequantizing every weight tensor into FMUL-IMM / HFMA2
      immediates per `weights-as-code-compiler.md`.
   d. Emit a grid-sync barrier after the layer.
3. Prepend `token_embed`; append `final_rmsnorm`, `lm_head_project`,
   `argmax_sample`.
4. Link all sections into one cooperative cubin; record the single QMD
   entry point.
5. Allocate the KV cache (16 slices) and DeltaNet state (48 slices) in
   BAR4 HBM at startup.

Output: a single ELF cubin whose `.text` is the entire forward pass for
Qwen 3.5 27B, one QMD dispatch per token, zero runtime dispatch logic.
