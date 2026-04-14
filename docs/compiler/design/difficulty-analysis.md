# Lithos Compiler: Difficulty Analysis and State Assessment

*Derived from a Shannon-identity analysis session, 2026-04-14.*

---

## What Lithos Actually Is

Lithos is not a programming language in the conventional sense. Strip the syntax away and what remains is a **compositional algebra over typed tensors**. The primitives — `+`, `-`, `*`, `/`, `Σ`, `△`, `▽`, `√`, `e^`, `ln`, `**`, `***` — are morphisms. A composition like:

```
L2Norm x :
  x
  *
  Σ
  √
  reciprocal
  * x
```

is not a program with steps. It is a mathematical expression in point-free style — a way of writing `x · (1/√(Σ(x²)))` without parentheses. The stack is notation. The meaning is the composition of functions, not the execution of instructions.

This matters for the compiler because the source language carries almost no execution model. It says *what* to compute with extremely high information density. It says nothing about *how* threads are organized, *which* dimension becomes `threadIdx`, *where* reductions cross warp boundaries, or *how many* registers are live at any point. All of that must be decided by the compiler.

---

## Difficulty Ratings: The Full Stack

The question was posed: how hard is each compiler target on a 0–10 scale?

### Lithos → PTX → GPU

**Difficulty: 3**

PTX uses virtual registers — the compiler never runs out. ptxas handles register allocation, instruction scheduling, stall insertion, reuse hints, and control word construction. The mapping from Lithos primitives to PTX instructions is nearly mechanical:

| Lithos | PTX |
|--------|-----|
| `+` | `add.f32` |
| `√` | `sqrt.approx.f32` |
| `e^` | `ex2.approx.f32` (with log₂e prescaling) |
| `Σ` | fixed warp shuffle reduction template |
| `→` / `←` | `ld.global` / `st.global` |

The genuinely hard parts at this level:
- `***` (outer product): scalar FMAs or tensor cores — a shape-dependent decision
- Multi-dimensional `each`: which axis maps to `threadIdx.x` vs `blockIdx.x`
- Reductions exceeding warp width: require shared memory + barrier, size must be known

If tensor shapes are known at compile time, it drops to a **2**. If `***` targets scalar FMAs only, it drops to a **2**.

### Lithos → raw Hopper binary → GPU (skipping PTX)

**Difficulty: 5**

Same math-to-execution-model decisions as the PTX path (the 3). What is added:

- Real register allocation — ptxas was doing this from virtual PTX registers; now the compiler must
- Instruction scheduling and stall insertion
- Encoding opcode and operand fields in binary
- Control word construction (stall cycles, barrier slots, reuse bits)

The control words are the bulk of the additional work. Each 16-byte sm_90a instruction is 8 bytes of math and 8 bytes of scheduling metadata. Filling those correctly requires knowing the full data dependency graph of surrounding instructions — information that is only complete after the entire instruction sequence is emitted.

### SASS → opcode (on a known, reverse-engineered architecture)

**Difficulty: 1**

If opcodes are empirically verified — which Lithos's are, probed on GH200 via ptxas/nvdisasm — then this step is pure bit-packing. Mechanical.

### Lithos → correct GPU opcode (combined)

**Difficulty: 6** (5 + 1)

### Lithos → GPU opcode with maximal FP32 register utilization and 4-bit weight streaming

**Difficulty: 9**

This is the compiler actually being built. The additional difficulty over the baseline 6 comes from:

- Register allocation must not just be correct but **optimal**: pack live values to maximize occupancy, interleave 4-bit dequantization chains without spilling
- The streaming pipeline design requires knowing what is live across instruction boundaries far in advance of emission
- Scheduling decisions (control words) interact with the register packing — a reuse hint that is wrong wastes bandwidth; a stall count that is too conservative wastes cycles

The first response in this session implicitly assumed this higher target and assigned 9 accordingly. When the components were analyzed in isolation (without the register packing constraint), the number came down to 6. Both numbers are correct for their respective targets.

---

## Comparison: Traditional Path vs Lithos Direct

The traditional pipeline for GPU code:

```
C++ → ASM → PTX → SASS → opcode
```

This is not one compiler. It is five compilers, each a multi-million-line engineering project with decades of accumulated work. NVIDIA has a team of hundreds of engineers on ptxas alone. The PTX → SASS step involves a full register allocator, a latency-aware scheduler with models for every instruction on every GPU variant, a peephole optimizer, and a binary layout engine.

Lithos bypasses all of that. It also starts from a *higher* abstraction than C++ (pure compositional math, no imperative scaffolding) and emits raw 16-byte Hopper binary instruction words with hand-constructed control fields directly — bypassing the SASS assembler layer entirely — launched via a pushbuffer/QMD path that bypasses libcuda entirely.

| Layer | Traditional | Lithos |
|-------|-------------|--------|
| C++ → ASM | compiler provides | not applicable — source is higher than C++ |
| ASM → PTX | compiler provides | not applicable |
| PTX → SASS | ptxas (NVIDIA) | Lithos compiler skips this layer — emits raw binary directly |
| SASS → opcode | ptxas (NVIDIA) | Lithos compiler does this (done, verified) — emits 128-bit binary words |
| Kernel dispatch | libcuda.so (~80MB) | pushbuffer.ls + qmd.ls + dispatch.ls (~few hundred lines) |

Lithos is attempting a strictly harder problem. The traditional pipeline stands on top of NVIDIA's entire stack and only solves the top layer. Lithos solves all layers simultaneously, starting from a more abstract source.

---

## The Fundamental Compilation Problem

The core difficulty is not the target level. It is the **semantic gap**.

A Lithos source file has very high information density: few bits express much mathematical content. A GPU binary has very high mechanical specificity: many bits express mostly scheduling and hardware-appeasing detail.

Compiling one to the other is expansion of a compressed message through a noisy channel. The compiler is the encoder. Its job is to add exactly the right redundancy — register numbering, barrier placement, stall counts, warp scheduling — without corrupting the mathematical signal.

The expansion requires global information (tensor shapes, data flow topology, reduction sizes) to make correct decisions, but a single-pass architecture only has local information (the current token) at emission time. This is the architectural tension the compiler must manage.

---

## Can Development Continue Without the Hopper GPU?

**Yes, for most remaining work. No, for execution validation.**

### What can be done on this machine (Apple Silicon, no GPU):

- Bootstrap parser: fixing `name args :` composition syntax and `name expr` binding syntax — pure ARM64, runs locally
- Self-hosting chain: bootstrap parses compiler.ls → stage1 → self-compile — entirely local
- GPU backend: writing emitters and verifying bit patterns against known-good encodings from the probe corpus
- Register allocator: design and implementation — no execution required
- Control word scheduling pass: design and implementation
- All language frontend work: lexer, parser, new syntax

### What requires the GH200:

- Kernel execution testing
- QMD/pushbuffer/doorbell path validation
- Completion polling
- Correctness validation against PyTorch reference (cosine similarity at every layer)
- Performance measurement

The compile-and-emit pipeline can be fully developed and made correct in structure before any execution is possible.

---

## Is the Compiler Doing the Right Thing?

**Mostly yes. One known gap that matters at scale.**

### What is right:

- **Architecture**: single-pass, no AST, direct emission. Appropriate for the source language and target. Lithos source is nearly linear; the straight-line structure maps naturally to single-pass emission.
- **Opcode table**: empirically verified on GH200 via ptxas 12.8.93 + nvdisasm 12.8.90. Not guessed, not derived from documentation alone — probed.
- **ELF/cubin structure**: 9-section cubin ELF64 with correct `.nv.info` attributes.
- **Dispatch path**: pushbuffer → QMD → SEND_PCAS_A, bypassing libcuda entirely. Correct.
- **register_count location**: found in the 384-byte Shader Program Descriptor (SPD) at offset 0x094, not in QMD or cbuf0. Three-point verified.

### What is incomplete:

**Register allocator.** The current allocator is monotone — it hands out fresh register IDs and never reuses them. This works for small kernels (vadd fits). It will exhaust Hopper's 255-per-thread register budget on a full DeltaNet layer, which has 22+ operations with hundreds of live values.

The fix is a **linear-scan allocator with a liveness pass** over the instruction sequence. Graph coloring is overkill for the straight-line-plus-short-loops shape of Lithos kernels. The shape is favorable: compute liveness backward over the linear sequence, assign registers greedily from a free list, spill only when necessary.

This is not a design flaw — it is a known incomplete piece. It is also the single most load-bearing remaining item for correctness at full model scale.

**Control word scheduling.** Currently uses hardcoded stall/barrier values from the probe corpus. Correct for simple patterns; conservative for deep pipelining. Acceptable for first working kernel. A proper latency-aware backward pass over the instruction sequence is the eventual fix, but it is not blocking correctness — only performance.

### Summary

The compiler is architecturally sound, empirically grounded, and structurally complete. The register allocator must be upgraded before full-layer kernels can be compiled. Everything else is bounded, known work.
