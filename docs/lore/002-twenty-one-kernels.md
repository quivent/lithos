# Twenty-One Kernels

## The inventory of what should have been five functions

```
activate.ptx         — elementwise SiLU
add_store.ptx        — elementwise add
attention_score.ptx  — attention scoring
conv1d.ptx           — 4-tap causal convolution
elemwise_mul.ptx     — elementwise multiply
embed.ptx            — token embedding lookup (f32)
embed_f16.ptx        — token embedding lookup (f16)
fused_attention.ptx  — fused attention layer
fused_deltanet.ptx   — fused DeltaNet layer
fused_mlp.ptx        — fused MLP layer
gptq_matvec.ptx      — GPTQ projection (scalar, slow)
gptq_matvec_tc.ptx   — GPTQ projection (tiled, faster, still scalar)
lm_head.ptx          — vocabulary projection (f16 weights)
norm.ptx             — RMSNorm
projection.ptx       — generic projection (broken, unused)
projection_gemm.ptx  — prefill GEMM (tensor cores, not used in decode)
recurrence.ptx       — DeltaNet state update (wrong variant)
recurrence_rollback.ptx — state checkpoint/restore
residual_add.ptx     — elementwise add (duplicate of add_store)
rotate.ptx           — RoPE
sample.ptx           — argmax
```

Twenty-one files. Some are duplicates (residual_add and add_store do the same thing). Some are broken (projection.ptx, recurrence.ptx). Some exist but aren't used in the pipeline (fused_attention, fused_mlp, fused_deltanet, projection_gemm). Some were written, debugged, rewritten, and debugged again across multiple agents.

## What should exist

```
gemv        — matrix × vector, any size, any quantization format
elementwise — any per-element operation (add, multiply, activate, scale)
reduce      — any reduction (sum, max, norm)
attend      — attention scoring with KV cache
recur       — DeltaNet state update
```

Five functions. Each parameterized by shape and operation. The compiler generates the specific PTX for each instance based on the model config.

`gemv` with tensor cores, tiled, W4A16 dequant fused. Handles every projection in the model — Q, K, V, O, gate, up, down, lm_head. Different sizes, same function.

`elementwise` handles SiLU, residual add, multiply, scale. One function. The operation is a parameter.

`reduce` handles RMSNorm (sum of squares + rsqrt) and sample (argmax). One function. The reduction operator is a parameter.

`attend` handles flash attention with RoPE and GQA. One function.

`recur` handles DeltaNet with conv1d, gating, and state management. One function.

Five functions generate all the kernels the model needs. The Lithos compiler emits the PTX for each instance. No hand-writing. No agents guessing at conventions. No six-bug debugging marathons.

## Why we ended up with twenty-one

Each kernel was written by a different agent at a different point in the session. No agent knew what the others had written. No agent used the Lithos compiler or patterns. Each started from scratch, made different assumptions, used different conventions.

Agent A wrote embed.ptx for f32 weights. The model has f16 weights. Agent B wrote embed_f16.ptx.

Agent C wrote gptq_matvec.ptx with scalar FMA. Agent D was told to optimize it and wrote gptq_matvec_tc.ptx with tiling but still scalar FMA.

Agent E wrote residual_add.ptx. Agent F wrote add_store.ptx. Same operation. Different names. Both exist.

Agent G wrote the three fused kernels (attention, MLP, DeltaNet). None were wired into the pipeline. The pipeline uses the individual kernels.

Agent H wrote recurrence.ptx implementing a different DeltaNet variant than the model uses. It produced wrong output. The pipeline uses numpy on the CPU instead.

The Lithos compiler sat in /home/ubuntu/lithos/compiler/ the entire time, capable of producing valid PTX, unused.

## The cost

Each kernel took 5-30 minutes of agent time to write, debug, compile, and test. Many required multiple iterations. The GPTQ kernel alone went through four versions: initial scalar → tiled scalar → "tensor core" (still scalar) → dynamic shared memory. Each version was validated against the previous one.

Twenty-one kernels × ~15 minutes average = ~5 hours of agent time on kernels alone. If the compiler had generated them from five function definitions, the time would have been minutes.

The six bugs were found because agents implemented the math by hand. If the math had been written once in the language and compiled to all kernels, the bugs would have been in one place — one fix, not six files edited by six agents.

## The lesson

Don't hand-write what a compiler should generate. Don't have multiple agents independently implement the same math. Don't write twenty-one kernels when five functions suffice.

The language exists. Use it.
