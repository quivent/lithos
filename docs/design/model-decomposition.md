# Model Decomposition Design

Lithos decomposes model inference into a hierarchy that maps directly from architecture config down to GPU opcodes. The structure lives in `models/families/` and is organized by model family, variant, and parameter count.

## Structure

```
models/families/qwen/qwen3.5/27b/
  config.json              model config (from HuggingFace)
  layers/
    deltanet/
      kernels.ls           linear pipeline of kernel operations
      unroll.ls            each kernel decomposed to Lithos primitives
    attention/
      kernels.ls           linear pipeline of kernel operations
      unroll.ls            each kernel decomposed to Lithos primitives
```

## What each file does

**config.json** is the model's own config, unmodified. It defines the layer types, dimensions, head counts, and quantization parameters. Everything else is derived from it.

**kernels.ls** lists the kernel operations for one layer type in execution order. Each line is one kernel. Data flows implicitly — no named intermediates. The file is the layer definition.

**unroll.ls** decomposes each kernel into Lithos primitives using the spec's vocabulary (`** Σ 1/√ e^ *** **** ≅ ≡ → ←`). Three levels of indentation:

- Level 0: kernel name with shape annotation (`[]` vector, `[][]` matrix, `[][][]` layer)
- Level 1: plain-English verb (what it does)
- Level 2+: Lithos primitives, with named compositions (sigmoid, silu, softplus) shown as intermediate steps before their atomic expansion

## Why

The goal is to express a complete model inference as a pipeline of Lithos primitives. Each primitive maps to a known SASS instruction sequence. If every kernel in `kernels.ls` can be unrolled through `unroll.ls` down to primitives, and every primitive has a proven opcode emission, then the path from `.ls` source to GPU execution is fully specified.

This decomposition also makes it possible to target any model. A new model means: drop in its config.json, write its kernels.ls (the layer pipeline), write its unroll.ls (the primitive decomposition), and the same compiler infrastructure produces the opcodes.

## Current target

Qwen 3.5 27B — 64 layers (48 DeltaNet + 16 full attention), GPTQ W4A16. The initial proof of concept is a single token inference: embed → 64 layers → output projection → argmax → token.

## What is not here

The MLP block (RMSNorm → gate/up projection → silu gate → down projection → residual) is shared between both layer types and is not yet written. Embedding and output projection (the first and last operations of a forward pass) are also not yet in this structure.
