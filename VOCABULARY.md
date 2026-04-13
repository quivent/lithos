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
11. subtract, multiply
12. outer, multiply, add
13. matvec
14. rmsnorm
15. sigmoid, multiply, multiply
16. project
17. add
18. rmsnorm
19. project
20. silu, multiply
21. project
22. add

## Fully Decomposed — 41 Primitive Steps

Every composite operation broken into irreducible math:

| Step | Operation | Context |
|------|-----------|---------|
| 1 | normalise | input hidden state |
| 2 | project | → q k v |
| 3 | project | → z |
| 4 | multiply | convolve (FIR tap) |
| 5 | sum | convolve (accumulate) |
| 6 | negate | silu |
| 7 | e^ | silu |
| 8 | + 1 | silu |
| 9 | 1 / | silu |
| 10 | multiply | silu |
| 11 | square | l2norm |
| 12 | sum | l2norm |
| 13 | 1 / √   | l2norm |
| 14 | multiply | l2norm |
| 15 | scale | q by 1/√d |
| 16 | sigmoid | → beta |
| 17 | e^ | softplus |
| 18 | + 1 | softplus |
| 19 | log | softplus |
| 20 | multiply | matvec S @ k |
| 21 | sum | matvec S @ k |
| 22 | subtract | v - recall |
| 23 | multiply | * beta → correction |
| 24 | outer | correction ⊗ k |
| 25 | multiply | decay * S |
| 26 | add | S update |
| 27 | multiply | matvec S @ q |
| 28 | sum | matvec S @ q |
| 29 | rmsnorm | output |
| 30 | sigmoid | gate z |
| 31 | multiply | silu(z) |
| 32 | multiply | output * gate |
| 33 | project | → hidden dim |
| 34 | add | residual X |
| 35 | rmsnorm | MLP input |
| 36 | project | → gate, up |
| 37 | sigmoid | silu gate |
| 38 | multiply | silu gate |
| 39 | multiply | gate * up |
| 40 | project | → down |
| 41 | add | residual X |

## The Primitives

```
normalise  project  rmsnorm  sigmoid  scale
multiply   sum      add      subtract
e^         log      1 /      1 / √      square
outer      negate
```

16 words. But `normalise`, `project`, `rmsnorm`, `sigmoid` are themselves
compositions of the others. The irreducible set:

```
multiply  add  subtract  e^  log  1/  1/√  square  outer  negate  sum  scale
```

12 operations. A DeltaNet layer is 41 of them in sequence.

## Why This Matters

A "fused kernel" is a redundant phrase. Every function is already a
composition of smaller functions. RMSNorm is square → sum → 1/√ → multiply.
SiLU is negate → e^ → +1 → 1/ → multiply. The word "fused" only exists
because other systems make separation the default and fusion the exception.

In Lithos, composition is the default. There is nothing to fuse because
nothing was ever separated. A DeltaNet layer is 41 math steps. The compiler
decides how many GPU launches that becomes.
