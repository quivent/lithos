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

### Scalar Parameters

`param NAME TYPE` declares a scalar kernel parameter (not a pointer).
TYPE is `u32` or `f32`.

### For Loops

`for COUNTER START BOUND STEP` ... `endfor` emits a PTX loop.
COUNTER is a new u32 variable. START/BOUND/STEP can be literals or variables.

### Integer / Bitwise Operations

`shr DST SRC AMT` — shift right  
`shl DST SRC AMT` — shift left  
`and DST SRC1 SRC2` — bitwise AND  
`or DST SRC1 SRC2` — bitwise OR  
`xor DST SRC1 SRC2` — bitwise XOR  

### Warp Shuffle

`shfl.bfly DST SRC OFFSET` emits `shfl.sync.bfly.b32`.

### Shared Memory

`shared NAME COUNT TYPE` declares shared memory (e.g., `shared buf 1024 f32`).

### Math Intrinsics

`exp DST SRC`, `rcp DST SRC`, `rsqrt DST SRC`, `sqrt DST SRC`,
`sin DST SRC`, `cos DST SRC`, `neg DST SRC` — unary math ops.
`fma DST A B C` — fused multiply-add.

### Float Constants

Literals like `0.0`, `1.0`, `0.5` are encoded as IEEE 754 hex in PTX.

### Type Conversions

`u32>f32 DST SRC`, `s32>f32 DST SRC`, `f32>u32 DST SRC`, `f32>s32 DST SRC`.

### Predication

`@pN INSTRUCTION` emits a predicated PTX instruction.
`setp CMP TYPE PRED SRC1 SRC2` sets a predicate register.

### Bounds Check / Early Exit

`if>= A B exit` emits `setp.ge.u32` + `@p bra $L_exit`.

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
