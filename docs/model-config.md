# Target Model Configuration

Canonical model spec for Lithos compilation and inference.

**Model:** Qwen 3.5 27B (Huihui-abliterated, GPTQ W4A16 variant)
**Source:** `/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16/config.json`
**Architecture:** `Qwen3_5ForConditionalGeneration` (multimodal: text + vision)
**Hybrid attention:** 3 linear_attention (DeltaNet) + 1 full_attention, repeating × 16 = 64 layers

---

## Text backbone

| Parameter | Value |
|---|---|
| `num_hidden_layers` | **64** |
| `hidden_size` | 5120 |
| `intermediate_size` | 17408 |
| `vocab_size` | **248,320** |
| `head_dim` | 256 |
| `rms_norm_eps` | 1e-6 |
| `hidden_act` | silu |
| `attention_bias` | false |
| `attn_output_gate` | true |
| `tie_word_embeddings` | false |
| `max_position_embeddings` | 262,144 |
| `dtype` | bfloat16 |

## DeltaNet (linear attention) layers — 48 of 64

| Parameter | Value |
|---|---|
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 48 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |

GQA ratio inside DeltaNet = 48 / 16 = 3.

## Full attention layers — 16 of 64

| Parameter | Value |
|---|---|
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4 |
| GQA ratio | 6 |
| Output gate | enabled (`attn_output_gate: true`) |
| `full_attention_interval` | 4 |

Every 4th layer (indices 3, 7, 11, 15, …, 63) is full attention.

## RoPE

| Parameter | Value |
|---|---|
| `rope_type` | default |
| `rope_theta` | **10,000,000** |
| `mrope_interleaved` | true |
| `mrope_section` | [11, 11, 10] (text, height, width) — sums to 32 pairs = 64 dims |
| `partial_rotary_factor` | 0.25 → rotated_dims = 64 (of 256) |

## Layer types (pattern — first 16 shown; repeats × 4)

```
index  0:  linear_attention
index  1:  linear_attention
index  2:  linear_attention
index  3:  full_attention      <- every 4th
index  4:  linear_attention
index  5:  linear_attention
index  6:  linear_attention
index  7:  full_attention
index  8:  linear_attention
index  9:  linear_attention
index 10:  linear_attention
index 11:  full_attention
index 12:  linear_attention
index 13:  linear_attention
index 14:  linear_attention
index 15:  full_attention
... (same 4-pattern repeats × 16 for total 64 layers) ...
```

Full attention layer indices: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63.

## Multimodal — vision encoder

Vision is a **separate compilation path** from the text backbone.

| Parameter | Value |
|---|---|
| `depth` | 27 |
| `hidden_size` | 1152 |
| `intermediate_size` | 4304 |
| `num_heads` | 16 |
| `patch_size` | 16 × 16 |
| `temporal_patch_size` | 2 (2-frame temporal) |
| `spatial_merge_size` | 2 |
| `in_channels` | 3 |
| `num_position_embeddings` | 2304 |
| `out_hidden_size` | 5120 (projects into text backbone) |
| `hidden_act` | gelu_pytorch_tanh |

Special vision tokens: `vision_start_token_id=248053`, `vision_end_token_id=248054`,
`image_token_id=248056`, `video_token_id=248057`.

## Quantization

| Parameter | Value |
|---|---|
| `quant_method` | gptq |
| `bits` | **4** |
| `group_size` | **128** |
| `sym` | **true** |
| `desc_act` | false |

Activations remain at 16-bit (W4A16). Per auto-GPTQ convention used during export,
zero-point is 8.0 for symmetric 4-bit quantization.

## MTP (multi-token prediction)

- `mtp_num_hidden_layers`: 1
- `mtp_use_dedicated_embeddings`: false
