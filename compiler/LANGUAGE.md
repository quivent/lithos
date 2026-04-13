# Lithos Language Specification

Lithos is a language where everything is a math function. There is no "kernel"
concept. Functions compose. When the host launches a function on the GPU, that
function runs on the GPU. Functions call other functions, which inlines them.
That is what "fusion" means: composed math functions become one GPU launch.

## Design Principles

1. **A program is a collection of math function definitions.** No other
   construct exists at the top level.

2. **Functions take inputs, produce outputs.** The `->` separates inputs
   from outputs in the signature.

3. **`each` means parallel.** `each i` declares that the body runs in
   parallel over index `i`. The compiler generates thread indexing.

4. **Math looks like math.** Symbols where they naturally work (`+ - * /`),
   words where they do not (`reciprocal_sqrt`, `sum`, `length`).

5. **Function calls inline.** A function calling another function inlines
   the callee's body. This is fusion, and it is not a special operation.

6. **No GPU-isms in source.** No `tid`, no `thread.x`, no `block.dim.x`.
   The compiler generates those from `each`.

## Syntax

```
fn NAME param1 param2 ... -> output1 output2 ...
    each VAR
        output [ VAR ] = expr

fn NAME2 a b -> c
    each i
        c [ i ] = a [ i ] + b [ i ]
```

### Functions

A function begins with `fn`, followed by the function name, followed by
input parameter names (whitespace-separated), then `->`, then output
parameter names. All parameters are f32 array pointers.

### Parallel Iteration

`each VAR` introduces parallel iteration. The variable `VAR` (typically `i`)
becomes the parallel index. The compiler maps this to the global thread ID.

### Expressions

Infix math with standard precedence:
- `*` and `/` bind tighter than `+` and `-`
- Indexed access: `NAME [ VAR ]` loads element `VAR` from array `NAME`
- Indexed store: `NAME [ VAR ] = expr` stores to element `VAR`
- Assignment: `NAME = expr` creates a local variable

Operators: `+`  `-`  `*`  `/`

### Comments

Line comments start with `\` (Forth convention).

## Examples

### Vector Addition
```
fn vadd a b -> c
    each i
        c [ i ] = a [ i ] + b [ i ]
```

### Scale and Add (Fused)
```
fn fused_scale_bias x scale bias -> y
    each i
        y [ i ] = x [ i ] * scale [ i ] + bias [ i ]
```

### Residual Connection
```
fn residual_add projected residual -> output
    each i
        output [ i ] = projected [ i ] + residual [ i ]
```

## Compilation

```
forth-bootstrap lithos.fs input.li --emit ptx -o output.ptx
```

The compiler is written in Forth, hosted by forth-bootstrap. It reads `.li`
source files, parses the function definitions, and emits PTX text.

PTX is validated with `ptxas -arch=sm_90` and produces cubins that run
on the GH200.
