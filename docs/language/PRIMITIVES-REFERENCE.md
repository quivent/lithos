# Lithos Primitives Reference

This is the COMPLETE Lithos primitive vocabulary. If an operation is not listed here, it does not exist in the language.

## Arithmetic (scalar)

| Symbol | Operation | SASS |
|--------|-----------|------|
| `+` | Add | `FADD` |
| `-` | Subtract | `FADD` (negate modifier) |
| `*` | Multiply | `FMUL` |
| `/` | Divide | `MUFU.RCP` + `FMUL` |

## Dimensional Variants

| Depth | Multiply | Add | Subtract | Divide | Complexity |
|-------|----------|-----|----------|--------|------------|
| 0 (scalar) | `*` | `+` | `-` | `/` | O(1) |
| 1 (vector) | `**` | `++` | `--` | `//` | O(N) |
| 2 (matrix) | `***` | `+++` | `---` | `///` | O(N x M) |
| 3 (tensor) | `****` | `++++` | `----` | `////` | O(N x M x K) |

## Reductions

| Symbol | Operation | Notes |
|--------|-----------|-------|
| `Σ` | Sum | Shuffle tree + shared memory |
| `△` | Max | Same structure, `FMNMX` max |
| `▽` | Min | Same structure, `FMNMX` min |
| `#` | Index modifier | Prefix: `# △ x` = argmax |

## Math (1 instruction, SFU)

| Symbol | Operation | SASS |
|--------|-----------|------|
| `1/` | Reciprocal | `MUFU.RCP` |
| `1/√` | Reciprocal sqrt | `MUFU.RSQ` |
| `2^` | Power of 2 | `MUFU.EX2` |
| `log₂` | Log base 2 | `MUFU.LG2` |
| `√` | Square root | `MUFU.RSQ` + `MUFU.RCP` (2 ops) |
| `≅` | Sine | `MUFU.SIN` |
| `≡` | Cosine | `MUFU.COS` |

## Math (2 instruction composites)

| Symbol | Operation | SASS |
|--------|-----------|------|
| `e^` | Natural exp | `FMUL` (by 1.442695) + `MUFU.EX2` |
| `ln` | Natural log | `MUFU.LG2` + `FMUL` (by 0.693147) |

## Memory

| Symbol | Operation | SASS |
|--------|-----------|------|
| `→` | Load (width addr) | `LDG` / `LDS` |
| `←` | Store (width addr val) | `STG` / `STS` |
| `↑` | Read register ($N) | `S2R` |
| `↓` | Write register ($N val) | `MOV` |

## Control

| Keyword | Syntax | Purpose |
|---------|--------|---------|
| `for` | `for i start end step` | Counted loop |
| `each` | `each i` | Thread-parallel dispatch (GPU) |
| `stride` | `stride i dim` | Stride loop for vectors > blockDim (GPU) |
| `if==` | `if== a b` | Conditional: equality |
| `if>=` | `if>= a b` | Conditional: greater or equal |
| `if<` | `if< a b` | Conditional: less than |
| `trap` | `trap` | Supervisor call (ARM64 only) |

## Named Primitives

| Name | Operation | Notes |
|------|-----------|-------|
| `project` | W4A16 GPTQ dequant GEMV | 40 SASS instructions per 8-weight group |
| `matvec` | FP32 state matrix-vector multiply | Structurally `***Σ` |

## Standard Compositions (section 6.1)

| Name | Primitives | Decomposition |
|------|-----------|---------------|
| `sigmoid x` | 4 | `* -1` `e^` `+ 1` `1/` |
| `silu x` | 5 | `* -1` `e^` `+ 1` `1/` `* x` |
| `softplus x` | 3 | `e^` `+ 1` `ln` |
| `rmsnorm x w D` | 6 | `** x x` `Σ` `/ D` `1/√` `** x` `** w` |
| `l2norm x` | 4 | `** x x` `Σ` `1/√` `** x` |
