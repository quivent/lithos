# Lithos Inference Kernels

Target: **sm_90** (NVIDIA Hopper / GH200)
Compiled with: `ptxas -arch=sm_90` (CUDA 12.8, V12.8.93)
Disassembled with: `nvdisasm -hex`

## Kernel Summary

### 1. projection

W4A16 matrix-vector multiply with tensor core MMA operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| weight_ptr | u64 | Pointer to packed 4-bit weight matrix |
| scale_ptr | u64 | Pointer to per-group dequantization scales |
| input_ptr | u64 | Pointer to input vector (f16) |
| output_ptr | u64 | Pointer to output vector (f32) |
| M | u32 | Number of output rows |
| K | u32 | Input dimension |

- **Registers:** 24
- **Shared memory:** 6144 bytes
- **Barriers:** 1
- **Spills:** 0 stores, 0 loads
- **Files:** projection.ptx (5379 B), projection.cubin (4560 B), projection.sass (25645 B)

### 2. attention_score

Flash attention decode kernel with online softmax for single-query vs KV cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| q_ptr | u64 | Query vector pointer |
| k_cache_ptr | u64 | Key cache pointer |
| v_cache_ptr | u64 | Value cache pointer |
| output_ptr | u64 | Output pointer |
| seq_len | u32 | Sequence length of KV cache |
| head_dim | u32 | Dimension per head (256) |
| num_heads | u32 | Number of attention heads |

- **Registers:** 20
- **Shared memory:** 8192 bytes
- **Barriers:** 1
- **Spills:** 0 stores, 0 loads
- **Files:** attention_score.ptx (5583 B), attention_score.cubin (5208 B), attention_score.sass (38398 B)

### 3. norm

RMSNorm with fused residual addition. Computes `norm(input + residual) * weight`.

| Parameter | Type | Description |
|-----------|------|-------------|
| input_ptr | u64 | Input tensor pointer |
| residual_ptr | u64 | Residual tensor pointer |
| weight_ptr | u64 | RMSNorm weight pointer |
| output_ptr | u64 | Output tensor pointer |
| hidden_dim | u32 | Hidden dimension size |
| epsilon | f32 | Normalization epsilon |

- **Registers:** 15
- **Shared memory:** 128 bytes
- **Barriers:** 1
- **Spills:** 0 stores, 0 loads
- **Files:** norm.ptx (4929 B), norm.cubin (5712 B), norm.sass (47285 B)

### 4. activate

SiLU activation with elementwise gate multiply: `output = gate * silu(up)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| gate_ptr | u64 | Gate projection pointer |
| up_ptr | u64 | Up projection pointer |
| output_ptr | u64 | Output pointer |
| size | u32 | Number of elements |

- **Registers:** 16
- **Shared memory:** 0 bytes
- **Barriers:** 0
- **Spills:** 0 stores, 0 loads
- **Files:** activate.ptx (2237 B), activate.cubin (3656 B), activate.sass (19541 B)

### 5. rotate

Rotary position embedding (RoPE) applied to Q and K vectors.

| Parameter | Type | Description |
|-----------|------|-------------|
| q_ptr | u64 | Query tensor pointer (in-place) |
| k_ptr | u64 | Key tensor pointer (in-place) |
| cos_ptr | u64 | Cosine table pointer |
| sin_ptr | u64 | Sine table pointer |
| seq_pos | u32 | Sequence position index |
| head_dim | u32 | Head dimension |
| num_heads | u32 | Number of heads |

- **Registers:** 22
- **Shared memory:** 0 bytes
- **Barriers:** 0
- **Spills:** 0 stores, 0 loads
- **Files:** rotate.ptx (3422 B), rotate.cubin (3928 B), rotate.sass (19596 B)

### 6. sample

Top-k sampling via parallel argmax reduction with temperature scaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| logits_ptr | u64 | Logits vector pointer |
| output_token_ptr | u64 | Output token ID pointer |
| vocab_size | u32 | Vocabulary size |
| temperature | f32 | Sampling temperature |
| top_k | u32 | Top-k parameter |

- **Registers:** 12
- **Shared memory:** 256 bytes
- **Barriers:** 1
- **Spills:** 0 stores, 0 loads
- **Files:** sample.ptx (5101 B), sample.cubin (5064 B), sample.sass (36044 B)

### 7. embed

Token embedding table lookup with vectorized 128-bit loads/stores.

| Parameter | Type | Description |
|-----------|------|-------------|
| token_id | u32 | Input token ID |
| embed_table_ptr | u64 | Embedding table pointer |
| output_ptr | u64 | Output buffer pointer |
| hidden_dim | u32 | Embedding dimension |

- **Registers:** 14
- **Shared memory:** 0 bytes
- **Barriers:** 0
- **Spills:** 0 stores, 0 loads
- **Files:** embed.ptx (2836 B), embed.cubin (3784 B), embed.sass (19243 B)

## Aggregate Statistics

| Kernel | Registers | Shared Mem | Cubin Size |
|--------|-----------|------------|------------|
| projection | 24 | 6144 B | 4560 B |
| attention_score | 20 | 8192 B | 5208 B |
| norm | 15 | 128 B | 5712 B |
| activate | 16 | 0 B | 3656 B |
| rotate | 22 | 0 B | 3928 B |
| sample | 12 | 256 B | 5064 B |
| embed | 14 | 0 B | 3784 B |
| **Total** | - | - | **31912 B** |

All kernels compile with zero spills (0 bytes stack frame, 0 spill stores, 0 spill loads).
