# Lithos Language Specification

Version 1.0 — Complete reference for the Lithos GPU compute language.

Lithos compiles `.ls` source files directly to SASS (NVIDIA GPU machine code) and
ARM64 (host machine code). One source language, two backends. No intermediate
representation, no runtime library, no framework dependencies.

Target hardware: NVIDIA GH200 (Hopper SM_90a + Grace ARM64).
Target workload: Qwen 3.5 27B (64 hybrid layers: 48 DeltaNet + 16 full attention).

---

## 1. Design Principles

1. **Compositions, not functions.** Named sequences that the compiler flattens into
   instruction streams. No call stack, no return address, no ABI.
2. **One primitive = one hardware pattern.** Every primitive maps to a known SASS
   instruction sequence. The compiler never invents new patterns.
3. **Dimensional encoding in syntax.** Loop depth is encoded by symbol repetition:
   `*` is scalar, `**` is vector, `***` is matrix, `****` is tensor.
4. **Fusion by default.** Compositions are concatenated instruction streams.
   There is nothing to fuse because nothing was ever separated.
5. **Two targets, one grammar.** The same `.ls` source emits SASS or ARM64
   depending on the backend selected at compile time.

---

## 2. Lexical Structure

### 2.1 Comments

```
\\  comment to end of line
```

### 2.2 Identifiers

Alphanumeric sequences starting with a letter or underscore: `[a-zA-Z_][a-zA-Z0-9_]*`

### 2.3 Numeric Literals

Integer and floating-point literals in decimal notation:
```
42          integer
3.14159     floating point
1e-6        scientific notation
-1          negative (unary minus attached to literal)
0.08839     fractional
```

### 2.4 Special Symbols (Unicode)

| Symbol | Unicode | UTF-8 Bytes | Meaning |
|--------|---------|-------------|---------|
| `→`    | U+2192  | E2 86 92    | Load from memory |
| `←`    | U+2190  | E2 86 90    | Store to memory |
| `↑`    | U+2191  | E2 86 91    | Read register |
| `↓`    | U+2193  | E2 86 93    | Write register |
| `Σ`    | U+03A3  | CE A3       | Reduction (sum) |
| `△`    | U+25B3  | E2 96 B3    | Reduction (max) |
| `▽`    | U+25BD  | E2 96 BD    | Reduction (min) |
| `√`    | U+221A  | E2 88 9A    | Square root |
| `≅`    | U+2245  | E2 89 85    | Sine (MUFU.SIN) |
| `≡`    | U+2261  | E2 89 A1    | Cosine (MUFU.COS) |

### 2.5 ASCII Symbols

```
+ - * / $ #           arithmetic, register prefix, index reduction
** *** ****            dimensional multiply (2, 3, 4 loop depth)
++ +++ ++++            dimensional add
-- --- ----            dimensional subtract
// /// ////            dimensional divide
:                      composition definition terminator
```

### 2.6 Keywords

```
for each stride         loop constructs
if== if>= if<           conditionals
trap                    system call
```

---

## 3. Grammar

### 3.1 Composition Definition

```
name arg1 arg2 ... argN :
    body
```

A composition is a named sequence of operations. The body is indented.
The compiler flattens compositions inline at every use site — there is no
call/return overhead. Compositions may reference other compositions.

### 3.2 Memory Operations

```
→ width addr            load (width = 8, 16, 32, or 64 bits)
← width addr val        store (width = 8, 16, 32, or 64 bits)
```

`width` determines the load/store instruction width.
`addr` is a base address expression. `val` is the value to store.

ARM64 backend: `LDR` / `STR` variants.
GPU backend: `LDG` / `STG` (global), `LDS` / `STS` (shared).

### 3.3 Register Operations

```
↑ $N                    read hardware register N (by number)
↓ $N val                write val into hardware register N
$NAME                   read named register (resolved via arch dictionary)
```

Architecture dictionaries map symbolic names to hardware register IDs:

```
arch/hopper.dict:           arch/arm64.dict:
    TID_X    S2R 0x21           CNTVCT_EL0  MRS 0x5F01
    TID_Y    S2R 0x22           MPIDR_EL1   MRS 0xC005
    CTAID_X  S2R 0x25
    LANEID   S2R 0x00
```

### 3.4 Arithmetic Operations

Scalar operations (loop depth 0):

| Symbol | Operation | GPU SASS | ARM64 |
|--------|-----------|----------|-------|
| `+`    | Add       | `FADD Rd, Ra, Rb` | `FADD Dd, Da, Db` |
| `-`    | Subtract  | `FADD Rd, Ra, -Rb` (negate modifier) | `FSUB Dd, Da, Db` |
| `*`    | Multiply  | `FMUL Rd, Ra, Rb` | `FMUL Dd, Da, Db` |
| `/`    | Divide    | `MUFU.RCP Rt, Rb` then `FMUL Rd, Ra, Rt` | `FDIV Dd, Da, Db` |

### 3.5 Dimensional Operations

Loop depth is encoded by symbol count. Each additional symbol adds one
enclosing `for` loop around the inner SASS instruction.

| Depth | Multiply | Add | Subtract | Divide | Complexity |
|-------|----------|-----|----------|--------|------------|
| 0 (scalar) | `*` | `+` | `-` | `/` | O(1) |
| 1 (vector) | `**` | `++` | `--` | `//` | O(N) |
| 2 (matrix) | `***` | `+++` | `---` | `///` | O(N x M) |
| 3 (tensor) | `****` | `++++` | `----` | `////` | O(N x M x K) |

The inner SASS instruction is always the same regardless of depth.
`**` (elementwise vector multiply) emits: `for i: FMUL Rd[i], Ra[i], Rb[i]`.
`***` (outer product) emits: `for i: for j: FMUL M[i,j], A[i], B[j]`.
`****` (matmul index space) emits: `for i: for j: for k: FFMA C[i,j], A[i,k], B[k,j], C[i,j]`.

On Hopper, `***` may emit `HMMA` (tensor core) for FP16 when dimensions align to 16x16x16.

### 3.6 Reductions

| Symbol | Operation | SASS Pattern |
|--------|-----------|--------------|
| `Σ`    | Sum       | Shuffle tree: stride loop partial sums, 5x `SHFL.BFLY` + `FADD`, then smem cross-warp |
| `△`    | Max       | Same structure, `FMNMX` (max) instead of `FADD` |
| `▽`    | Min       | Same structure, `FMNMX` (min) instead of `FADD` |
| `#`    | Index     | Prefix modifies a reduction: `# △ x` = argmax |

Reductions are NOT loops. They collapse one dimension via the shuffle tree in O(log N).
This is the ONLY primitive family that requires shared memory, barriers, and shuffles.

Implementation detail:
1. Stride loop: each thread accumulates partial result via `FFMA`/`FADD`
2. Intra-warp: 5x `SHFL.BFLY` + reduction op (butterfly, 32 threads to 1)
3. Cross-warp: `STS` to shared memory, `BAR.SYNC`, first warp loads + reduces
4. Broadcast: `STS` result to smem[0], `BAR.SYNC`, all threads `LDS`

### 3.7 Mathematical Functions

Single-instruction operations using the Special Function Unit (SFU):

| Symbol | Operation | SASS Instruction | Notes |
|--------|-----------|------------------|-------|
| `1/`   | Reciprocal | `MUFU.RCP Rd, Ra` | One SFU op |
| `1/√`  | Reciprocal sqrt | `MUFU.RSQ Rd, Ra` | One SFU op. Core of RMSNorm. |
| `2^`   | Power of 2 | `MUFU.EX2 Rd, Ra` | One SFU op |
| `log₂` | Log base 2 | `MUFU.LG2 Rd, Ra` | One SFU op |
| `√`    | Square root | `MUFU.RSQ Rt, Ra` then `MUFU.RCP Rd, Rt` | Two SFU ops |
| `≅`    | Sine | `MUFU.SIN Rd, Ra` | One SFU op |
| `≡`    | Cosine | `MUFU.COS Rd, Ra` | One SFU op |

Two-instruction composite math:

| Symbol | Operation | SASS Sequence |
|--------|-----------|---------------|
| `e^`   | Natural exp | `FMUL Rt, Ra, 1.442695` then `MUFU.EX2 Rd, Rt` |
| `ln`   | Natural log | `MUFU.LG2 Rt, Ra` then `FMUL Rd, Rt, 0.693147` |

All math functions are scalar-only. To apply elementwise, wrap in `**`:
`** 1/√` applies reciprocal square root to every element of a vector.

### 3.8 Control Flow

```
for i start end step        counted loop
each i                      thread-parallel dispatch (GPU only)
stride i dim                stride loop for vectors longer than blockDim (GPU only)
if== a b                    conditional (equality)
if>= a b                    conditional (greater or equal)
if< a b                     conditional (less than)
```

`for` emits `ISETP` + `@P BRA` + `IADD3` for the loop counter.
`each` maps to thread indexing: `S2R SR_TID.X` + `S2R SR_CTAID.X` + `IMAD`.
`stride` emits a stride loop for processing vectors that exceed the thread block size.

### 3.9 System Calls

```
trap                        ARM64: SVC #0 (supervisor call)
```

Used for host-side I/O (file open, mmap, etc.). Not available on the GPU backend.

### 3.10 Named Primitives

Two operations are too hardware-specific to encode in dimensional symbols:

| Name | Operation | Why it's named |
|------|-----------|----------------|
| `project` | W4A16 GPTQ dequant GEMV | 40 SASS instructions per 8-weight group: `LDG` + 8x(`LOP3` + `I2F` + `FADD` + `FMUL` + `FFMA`) + warp reduction. Quantization scheme parameterizes the dequant sequence. |
| `matvec` | FP32 state matrix-vector multiply | Simpler than `project` — no dequant. Stride loop of `LDG` + `FFMA`, then warp reduction. Structurally `***Σ`. |

---

## 4. Compilation Model

### 4.1 Flattening

Compositions are inlined at every use site. The compiler produces a flat
instruction stream with no call/return overhead.

```
sigmoid x :                 silu x :
    * -1                        sigmoid x
    e^                          * x
    + 1
    1/
```

`silu x` flattens to: `* -1`, `e^`, `+ 1`, `1/`, `* x` — five instructions.

### 4.2 Compiler-Derived Constructs

The following are NEVER written in source. The compiler derives them:

| Concept | Derived from | Implementation |
|---------|-------------|----------------|
| Thread indexing | Vector operand shape | `S2R SR_TID.X` + `S2R SR_CTAID.X` + `IMAD` |
| Stride loops | Vector length > blockDim | `ISETP` + `@P BRA` + `IADD3` |
| Warp shuffle tree | `Σ` / `△` / `▽` on vector | 5x `SHFL.BFLY` + reduction op |
| Shared memory allocation | Cross-warp communication | `STS` / `LDS` / `BAR.SYNC` |
| Barriers | Cross-warp synchronization | `BAR.SYNC` |
| Register allocation | All live values | Linear-scan allocator |
| Coalesced memory access | Array indexing pattern | Address arithmetic |
| Loop unrolling | `project` inner loop | Compiler decides unroll factor |
| Tiling | `***`, `project` | Compiler decides tile shape |
| Grid synchronization | Layer boundaries | Cooperative barrier |

### 4.3 Self-Hosting

The compiler (`compiler/compiler.ls`, 5,467 lines) is written in Lithos with
7 sections: ARM64 backend, GPU backend, Lexer, Parser, Safetensors reader,
ELF writer, Main entry. Bootstrapped from pure ARM64 assembly (`bootstrap/*.s`).

---

## 5. Type System

Lithos has no explicit type system. Types are determined by context:

- **Scalars**: FP32 by default. Integer when used in address arithmetic or loop counters.
- **Vectors**: Arrays of scalars. Length determined by the operand.
- **Matrices**: 2D arrays. Dimensions determined by the operand.
- **Width specifier**: Memory operations (`→` / `←`) take an explicit bit width (8/16/32/64).

The compiler infers types from usage context. There are no type declarations,
type annotations, or type errors. If you write `* -1` on a vector, you get
elementwise negation. If you write `** -1` on a vector, you get the same thing
with an explicit loop.

---

## 6. Composition Library

### 6.1 Standard Compositions

These compositions are defined in terms of primitives. They are the vocabulary
for building inference kernels.

```
sigmoid x :                             \\ 4 primitives
    * -1                                \\ negate
    e^                                  \\ exponential
    + 1                                 \\ add one
    1/                                  \\ reciprocal

silu x :                                \\ 5 primitives
    sigmoid x                           \\ (flattens to 4 above)
    * x                                 \\ multiply by original

softplus x :                            \\ 3 primitives
    e^                                  \\ exponential
    + 1                                 \\ add one
    ln                                  \\ natural log

rmsnorm x w D :                         \\ 6 primitives
    ** x x                              \\ square each element
    Σ                                   \\ sum of squares
    / D                                 \\ divide by dimension
    1/√                                 \\ reciprocal square root
    ** x                                \\ scale input
    ** w                                \\ multiply by weight

l2norm x :                              \\ 4 primitives
    ** x x                              \\ square each element
    Σ                                   \\ sum of squares
    1/√                                 \\ reciprocal square root
    ** x                                \\ scale to unit length
```

### 6.2 Composition Algebra

Compositions combine by concatenation. The SASS instruction count is the
sum of the constituent primitives:

| Composition | Primitives | SASS Instructions (approx.) |
|-------------|-----------|---------------------------|
| `sigmoid`   | `* -1`, `e^`, `+ 1`, `1/` | 5 (FMUL + FMUL + MUFU.EX2 + FADD + MUFU.RCP) |
| `silu`      | sigmoid + `**` | 6 |
| `softplus`  | `e^`, `+ 1`, `ln` | 5 (FMUL + MUFU.EX2 + FADD + MUFU.LG2 + FMUL) |
| `rmsnorm`   | `**`, `Σ`, `/`, `1/√`, `**` | ~25 (Σ dominates with shuffle tree) |
| `l2norm`    | `**`, `Σ`, `1/√`, `**` | ~25 (Σ dominates) |

---

## 7. The 12 Primitive Families

Complete enumeration of every irreducible operation in Lithos:

```
Family 1-4:  Arithmetic with dimensional variants
    *   **   ***   ****         multiply at depth 0/1/2/3
    +   ++   +++   ++++         add at depth 0/1/2/3
    -   --   ---   ----         subtract at depth 0/1/2/3
    /   //   ///   ////         divide at depth 0/1/2/3

Family 5:   Reductions
    Σ   △   ▽   #               sum, max, min, index

Family 6-10: Scalar math (SFU)
    1/                          reciprocal
    1/√                         reciprocal square root
    2^                          power of 2
    log₂                        log base 2
    √                           square root

Family 11-12: Named GEMV operations
    project                     W4A16 dequant GEMV
    matvec                      FP32 state GEMV
```

Total unique symbols/names: 25.
Total primitive families: 12.
Every Lithos program is a sequence drawn from these 12 families.

---

## 8. Memory Model

### 8.1 GPU Memory Spaces

The compiler selects memory space based on access pattern:

| Space | SASS Prefix | Latency | Usage |
|-------|-------------|---------|-------|
| Global (HBM) | `LDG` / `STG` | ~400 cycles | Weights, activations, state |
| Shared (SMEM) | `LDS` / `STS` | ~30 cycles | Cross-warp reductions |
| Registers | `MOV` | 0 cycles | All intermediate values |
| Constant | `LDC` | ~4 cycles | Compile-time constants |

### 8.2 State Persistence

The DeltaNet recurrence state matrix S[128, 128] FP32 persists per head in HBM
across tokens. Full-model footprint: **48 layers × 16 K-heads × 128 × 128 × 4 bytes
= 48 MB per sequence**. (The "S[48, 128, 128]" shorthand used elsewhere describes
a per-K-head slice, not the full state — see `docs/inference/hybrid-layers.md` for
the complete `f32[48, 16, 3, 128, 128]` layout including V-head fan-out.) It is
loaded, updated, and stored back each token. The compiler handles the load/store
scheduling.

---

## 9. Runtime

The Lithos runtime (`runtime/*.ls`, 11 files) replaces `libcuda.so` entirely.
It communicates with the GPU via direct register writes through `vfio-pci`:

- QMD (Queue Meta Data) construction
- GPFIFO (GPU FIFO) submission
- Doorbell register writes
- GSP (GPU System Processor) boot sequence

No CUDA toolkit, no `cuLaunchKernel`, no driver API.

---

## 10. Complete Example: DeltaNet Layer

A full DeltaNet layer is 41 primitive steps (see Section 6 compositions):

```
deltanet_layer x X :
    rmsnorm x w_norm 5120           \\ 1. normalize input
    project W_qkv x qkv            \\ 2. project to Q, K, V
    project W_z x z                 \\ 3. project gate signal
    conv1d qkv history              \\ 4. temporal convolution
    silu qkv                        \\ 5. activate
    l2norm k                        \\ 6. normalize keys
    l2norm q                        \\    normalize queries
    * q 0.08839                     \\    scale Q by 1/√128
    sigmoid beta_raw                \\ 7. input gate
    softplus dt_raw                 \\    compute dt
    e^ A_log                        \\    compute decay
    * decay dt                      \\
    * -1                            \\
    e^                              \\
    matvec S k recall               \\ 8. read from state
    -- v recall                     \\    compute residual
    ** beta                         \\    gate residual
    *** delta k                     \\    outer product (rank-1 update)
    ** decay S                      \\    decay old state
    +++ S update                    \\    update state
    matvec S q output               \\    query state
    rmsnorm output w_head 128       \\ 9. group norm per head
    silu z                          \\    gate activation
    ** output gate                  \\    apply gate
    project W_out gated hidden      \\    project to hidden dim
    ++ X hidden                     \\    residual connection
```

This is the complete source for one DeltaNet layer. The compiler handles
thread indexing, memory coalescing, tiling, register allocation, shared
memory, barriers, and GPU launch configuration.

---

## Appendix A: SASS Instruction Reference

| SASS Instruction | Lithos Primitive | Unit |
|------------------|-----------------|------|
| `FADD`           | `+`, `-` (negate modifier) | FP32 core |
| `FMUL`           | `*` | FP32 core |
| `FFMA`           | `*` + `+` (fused) | FP32 core |
| `MUFU.RCP`       | `1/` | SFU |
| `MUFU.RSQ`       | `1/√` | SFU |
| `MUFU.EX2`       | `2^` | SFU |
| `MUFU.LG2`       | `log₂` | SFU |
| `MUFU.SQRT`      | `√` | SFU |
| `MUFU.SIN`       | `≅` | SFU |
| `MUFU.COS`       | `≡` | SFU |
| `FMNMX`          | `△`, `▽` | FP32 core |
| `SHFL.BFLY`      | `Σ` (intra-warp) | Warp shuffle |
| `STS` / `LDS`    | `Σ` (cross-warp) | Shared memory |
| `BAR.SYNC`       | `Σ` (barrier) | Barrier unit |
| `LDG` / `STG`    | `→` / `←` (global) | Memory |
| `S2R`            | `↑ $NAME` | System register |
| `HMMA`           | `***` (FP16 aligned) | Tensor core |
| `LOP3`           | `project` (nibble extract) | Integer |
| `I2F`            | `project` (int-to-float) | Conversion |
| `ISETP` + `BRA`  | `for`, `if==`, `if>=`, `if<` | Integer + branch |
| `IADD3`          | `for` (loop counter) | Integer |

---

## Appendix B: Qwen 3.5 27B Dimensions

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 5120 |
| K heads (key/query) | 16 |
| V heads (value) | 48 |
| Head dimension | 128 |
| GQA ratio | 3:1 (each K-head serves 3 V-heads) |
| Conv kernel size | 4 |
| Total layers | 64 (48 DeltaNet + 16 full attention) |
| State matrix per head | [128, 128] FP32 |
| Quantization | W4A16 GPTQ |
