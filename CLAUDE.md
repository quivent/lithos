# Lithos Project — CLAUDE.md

## Lithos Language — Mandatory Primitives

When writing `.ls` files, use ONLY these primitives and compositions. NEVER invent
English-named functions. The spec (`docs/language/SPECIFICATION.md`) is authoritative.

### Arithmetic (dimensional depth by symbol count)

| Scalar | Vector | Matrix | Tensor | Operation |
|--------|--------|--------|--------|-----------|
| `*`    | `**`   | `***`  | `****` | Multiply  |
| `+`    | `++`   | `+++`  | `++++` | Add       |
| `-`    | `--`   | `---`  | `----` | Subtract  |
| `/`    | `//`   | `///`  | `////` | Divide    |

### Reductions

| Symbol | Meaning |
|--------|---------|
| `Σ`    | Sum reduction (shuffle tree) |
| `△`    | Max reduction |
| `▽`    | Min reduction |
| `#`    | Index prefix (`# △ x` = argmax) |

### Memory and Registers

| Symbol | Meaning |
|--------|---------|
| `→`    | Load from memory |
| `←`    | Store to memory |
| `↑`    | Read register |
| `↓`    | Write register |

### Scalar Math (SFU)

| Symbol | Meaning |
|--------|---------|
| `1/`   | Reciprocal |
| `1/√`  | Reciprocal square root |
| `2^`   | Power of 2 |
| `log₂` | Log base 2 |
| `√`    | Square root |
| `e^`   | Natural exp (2 instr: FMUL + MUFU.EX2) |
| `ln`   | Natural log (2 instr: MUFU.LG2 + FMUL) |
| `≅`    | Sine |
| `≡`    | Cosine |

### Named Operations

| Name | Meaning |
|------|---------|
| `project` | W4A16 GPTQ dequant GEMV |
| `matvec`  | FP32 state matrix-vector multiply |

### Standard Compositions (section 6.1)

```
sigmoid x :       * -1, e^, + 1, 1/
silu x :          sigmoid x, * x
softplus x :      e^, + 1, ln
rmsnorm x w D :   ** x x, Σ, / D, 1/√, ** x, ** w
l2norm x :        ** x x, Σ, 1/√, ** x
```

### WRONG vs RIGHT

```
WRONG: square_each x         RIGHT: ** x x
WRONG: reduce_sum x          RIGHT: Σ
WRONG: multiply_by_weight w  RIGHT: ** w
WRONG: reciprocal_sqrt x     RIGHT: 1/√
WRONG: elementwise_add a b   RIGHT: ++ a b
```

The 25 primitive symbols and 5 standard compositions above are the COMPLETE
vocabulary. If you need something not listed here, check the spec first --
it does not exist and you are inventing.
