# Lithos Vocabulary — DeltaNet Layer Operations

A DeltaNet layer is a sequence of operations on a hidden state vector x,
accumulating into the residual stream X. There is no "kernel" concept.
Every operation is a math function. Composition is fusion.

## The Layer as Operations

 1. normalise
 2. project
 3. project
 4. convolve
 5. silu
 6. l2norm
 7. scale
 8. sigmoid
 9. softplus
10. matvec
11. --  **
12. ***  **  ++
13. matvec
14. rmsnorm
15. sigmoid  **  **
16. project
17. ++
18. rmsnorm
19. project
20. silu  **
21. project
22. ++

## Fully Decomposed — 41 Primitive Steps

Every composite operation broken into irreducible math:

| Step | Operation | Context |
|------|-----------|---------|
|  1 | normalise       | input hidden state |
|  2 | project         | → q k v |
|  3 | project         | → z |
|  4 | **              | convolve (FIR tap, elementwise) |
|  5 | Σ               | convolve (reduction) |
|  6 | * -1            | silu: negate |
|  7 | exp             | silu: e^x |
|  8 | ++              | silu: add 1 (elementwise) |
|  9 | //              | silu: 1/ (elementwise reciprocal) |
| 10 | **              | silu: elementwise multiply |
| 11 | **              | l2norm: square each element |
| 12 | Σ               | l2norm: sum all squares |
| 13 | 1/√             | l2norm: reciprocal root |
| 14 | **              | l2norm: elementwise scale |
| 15 | * 1/√d          | scale q by 1/√d |
| 16 | sigmoid         | → beta |
| 17 | exp             | softplus |
| 18 | ++              | softplus: + 1 |
| 19 | log             | softplus: ln |
| 20 | **              | matvec S @ k (element products) |
| 21 | Σ               | matvec S @ k (reduction) |
| 22 | --              | v - recall |
| 23 | **              | * beta → correction (elementwise) |
| 24 | ***             | correction ⊗ k (outer product) |
| 25 | **              | decay * S (elementwise) |
| 26 | +++             | S update (matrix add) |
| 27 | **              | matvec S @ q (element products) |
| 28 | Σ               | matvec S @ q (reduction) |
| 29 | rmsnorm         | output |
| 30 | sigmoid         | gate z |
| 31 | **              | silu(z) (elementwise) |
| 32 | **              | output * gate (elementwise) |
| 33 | project         | → hidden dim |
| 34 | ++              | residual X |
| 35 | rmsnorm         | MLP input |
| 36 | project         | → gate, up |
| 37 | sigmoid         | silu gate |
| 38 | **              | silu gate (elementwise) |
| 39 | **              | gate * up (elementwise) |
| 40 | project         | → down |
| 41 | ++              | residual X |

## The Primitives

Dimensional notation: the number of repeated symbols encodes loop depth.

```
*   **   ***   ****        multiply: 1 / N / N×M / N×M×K
+   ++   +++   ++++        add:      1 / N / N×M / N×M×K
-   --   ---   ----        subtract: 1 / N / N×M / N×M×K
/   //   ///   ////        divide:   1 / N / N×M / N×M×K
Σ                          reduce one dimension — O(log N), NOT a loop
exp   log   √   1/   1/√  scalar only; apply elementwise via **
project                    W4A16 dequant GEMV (too specific to encode in stars)
matvec                     FP32 state GEMV (ditto)
```

Total: 25 symbols / names. The 12 irreducible operation families:

```
*   **   ***   ****
+   ++   +++   ++++
-   --   ---   ----
/   //   ///   ////
Σ
exp   log   √   1/   1/√
project   matvec
```

`outer` is gone. It is `***`.

## Dimensional Table

| Symbols | Loop depth | Complexity | What it does |
|---------|-----------|------------|--------------|
| `*`     | 0 (scalar) | O(1)       | one FMUL |
| `**`    | 1          | O(N)       | loop of FMULs — elementwise |
| `***`   | 2          | O(N×M)     | nested loop of FMULs — outer product |
| `****`  | 3          | O(N×M×K)   | triple nested — matmul index space |
| `+`     | 0 (scalar) | O(1)       | one FADD |
| `++`    | 1          | O(N)       | loop of FADDs — elementwise |
| `+++`   | 2          | O(N×M)     | nested loop of FADDs — matrix add |
| `++++`  | 3          | O(N×M×K)   | triple nested — tensor add |
| `-`     | 0          | O(1)       | one FADD (negate modifier) |
| `--`    | 1          | O(N)       | elementwise subtract |
| `---`   | 2          | O(N×M)     | matrix subtract |
| `----`  | 3          | O(N×M×K)   | tensor subtract |
| `/`     | 0          | O(1)       | one RCP + FMUL |
| `//`    | 1          | O(N)       | elementwise divide |
| `///`   | 2          | O(N×M)     | matrix divide |
| `////`  | 3          | O(N×M×K)   | tensor divide |
| `Σ`     | —          | O(log N)   | shuffle tree — collapses one dimension |
| `exp`   | scalar     | O(1)       | one EX2 + FMUL |
| `log`   | scalar     | O(1)       | one LG2 + FMUL |
| `√`     | scalar     | O(1)       | RSQ + RCP |
| `1/`    | scalar     | O(1)       | one RCP |
| `1/√`   | scalar     | O(1)       | one RSQ |
| `project` | —        | O(N×M)     | W4A16 GEMV with dequant |
| `matvec`  | —        | O(N×M)     | FP32 state GEMV |

`Σ` is NOT a loop. It is the shuffle tree: O(log N) communication steps.
Single symbols are O(1). Every extra symbol = one more enclosing `for` loop.
`exp`, `log`, `√`, `1/`, `1/√` are scalar; wrap in `**` to apply elementwise.

## Why This Matters

A "fused kernel" is a redundant phrase. Every function is already a
composition of smaller functions. RMSNorm is `**`, Σ, `/`, `1/√`, `**`.
SiLU is `* -1`, `exp`, `++`, `//`, `**`. The word "fused" only exists
because other systems make separation the default and fusion the exception.

In Lithos, composition is the default. There is nothing to fuse because
nothing was ever separated. A DeltaNet layer is 41 math steps. The compiler
decides how many GPU launches that becomes.

## Composition Examples

```
rmsnorm   = **  Σ  / D  1/√  **          square → reduce → mean → reciproot → scale
silu      = * -1  exp  ++  //  **         negate → exp → add1 → recip → multiply
l2norm    = **  Σ  1/√  **               square → reduce → reciproot → scale
softplus  = exp  ++  log                  exp → add1 → ln
sigmoid   = * -1  exp  ++  //            negate → exp → add1 → recip
```

Outer product (formerly `outer`, now `***`):

```
correction ⊗ k  =  ***   (for each i,j: result[i,j] = correction[i] * k[j])
```

State update uses `***` and `+++`:

```
S = decay ** S  +++  beta ** (v -- recall) ***k
    └ elementwise ┘  └── elementwise ──┘ └outer┘
```
