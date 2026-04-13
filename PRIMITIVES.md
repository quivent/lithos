# Lithos Primitives

## Operations

```
+       add
*       multiply
/       divide
-       subtract (* -1 +)
√       square root
```

## Constants

```
e       2.71828182845904523536...
```

## Computation of Constants

### e

e = exp(1) = the sum that never ends

```
1
+ 1
+ 1 * 1 / 2
+ 1 * 1 * 1 / 6
+ 1 * 1 * 1 * 1 / 24
+ 1 * 1 * 1 * 1 * 1 / 120
+ 1 * 1 * 1 * 1 * 1 * 1 / 720
...
```

Which simplifies (since 1 * 1 = 1):

```
1
+ 1
+ 1 / 2
+ 1 / 6
+ 1 / 24
+ 1 / 120
+ 1 / 720
+ 1 / 5040
+ 1 / 40320
+ 1 / 362880
+ 1 / 3628800
```

Each denominator is the previous denominator multiplied by the next integer:

```
1
1 * 2 = 2
2 * 3 = 6
6 * 4 = 24
24 * 5 = 120
120 * 6 = 720
720 * 7 = 5040
5040 * 8 = 40320
40320 * 9 = 362880
362880 * 10 = 3628800
```

At 10 terms: 2.71828180...
At 12 terms: 2.71828182845...
Float32 exact at 12 terms.

In stack form:

```
1       // denominator
1       // sum

swap 1 * swap    // denom = 1
1 over / +       // sum += 1/1

swap 2 * swap    // denom = 2
1 over / +       // sum += 1/2

swap 3 * swap    // denom = 6
1 over / +       // sum += 1/6

swap 4 * swap    // denom = 24
1 over / +       // sum += 1/24

...12 times total
```

e is computed from: `+ * /` and the integers 1 through 12.
