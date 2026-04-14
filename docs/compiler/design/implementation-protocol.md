# Lithos Compiler: Implementation Protocol

*Authoring a compiler that compiles pure math to raw GPU binary is a difficulty-9 problem.
This document is the protocol for doing it without losing track of where you are.*

---

## Part I: Implementation Order

Work must proceed in dependency order. Each item unlocks the next. Do not skip ahead.

### Stage 1: Bootstrap Parser (blocking everything)

The bootstrap is ARM64 assembly that produces the first native binary from Lithos source.
The parser in the bootstrap currently fails on two constructs that appear throughout compiler.ls:

1. **Composition syntax**: `name args :` followed by indented body
2. **Binding syntax**: `name expr` — first token is name, rest is expression, no `=`

Until these parse, compiler.ls cannot be ingested by the bootstrap, and nothing downstream can proceed.

**What to fix:**
- `bootstrap/` parser logic at the composition entry point (line 1196 area)
- Binding parser must not treat the first token as a standalone expression
- Test incrementally: `var x 42` works today; `name args :` body must work next

**Acceptance criterion:** `bootstrap/build.sh` produces a binary that parses compiler.ls without crashing.

---

### Stage 2: Compiler Parses Itself (stage1)

Once the bootstrap parser handles the full grammar, feed compiler.ls to it:

```
./bootstrap/lithos-bootstrap compiler/compiler.ls -o lithos-stage1
```

This produces a stage1 binary. It does not need to produce correct output yet — it needs to not crash and to emit *something* for every input construct.

**Acceptance criterion:** `lithos-stage1` runs without crashing on compiler.ls.

---

### Stage 3: Self-Hosting Fixed Point

Feed compiler.ls to lithos-stage1:

```
./lithos-stage1 compiler/compiler.ls -o lithos-stage2
./lithos-stage2 compiler/compiler.ls -o lithos-stage3
diff lithos-stage2 lithos-stage3  # must be identical
```

The fixed point — where the compiler compiles itself to an identical binary — is the correctness signal for the host (ARM64) path.

**Acceptance criterion:** `diff lithos-stage2 lithos-stage3` is empty.

---

### Stage 4: Register Allocator

The current allocator is monotone: it only hands out fresh register IDs. This must be replaced before any full-layer kernel can compile without exhausting the 255-register budget.

**Design (linear-scan with liveness):**

Lithos kernels are mostly straight-line sequences with short `for`/`each` loops. This shape is favorable — it does not require full graph coloring.

```
Pass 1 — forward emission pass (already exists):
  Emit instruction sequence into a flat buffer.
  Record for each instruction: (opcode, dst_reg_id, [src_reg_ids], latency_cycles)

Pass 2 — backward liveness pass (new):
  Walk instruction buffer in reverse.
  For each instruction:
    - Mark dst_reg as defined here
    - Mark src_regs as live from here to their next use
  Output: live_in[i] and live_out[i] sets for each instruction i

Pass 3 — register assignment (new):
  Walk forward.
  Maintain free list of physical registers (0..254).
  At each instruction:
    - For each src_reg in live_out[i] that is not in live_out[i+1]: return to free list
    - Assign physical register from free list to dst_reg
  Output: mapping from virtual ID → physical register number

Pass 4 — rewrite (new):
  Walk instruction buffer, substitute physical register numbers.
  This is the final form fed to the binary emitter.
```

**Acceptance criterion:** `vadd.ls` compiles to the same binary as before. A full `gemv.ls` compiles without spilling (register count at or below 128 for good occupancy).

---

### Stage 5: Control Word Scheduling Pass

Currently stall counts are hardcoded conservatively. A backward pass over the instruction sequence can fill them correctly.

**Design:**

After the register assignment pass, walk the instruction buffer backward:

```
For instruction i with latency L:
  Scan forward from i to find the nearest instruction j that reads dst_reg(i).
  stall_count(i) = max(0, L - (j - i - 1))
  If no reader found within the latency window: stall_count = 0
```

The Hopper latency table (from the probe corpus):
- FP32 arithmetic (FADD, FMUL, FFMA): 4 cycles
- Special function (MUFU): 6 cycles
- Global memory load (LDG): 20+ cycles (hide with software pipelining)
- Shared memory load (LDS): 20 cycles
- Integer arithmetic: 4 cycles

**Acceptance criterion:** The emitted control words for a vadd kernel match those produced by ptxas disassembly for the equivalent PTX. Within ±1 stall cycle.

---

### Stage 6: First Kernel Execution (requires GH200)

Compile `compiler/examples/vadd.ls` to a cubin. Launch it via dispatch.ls on a pair of known vectors. Compare output to expected result.

This is the first end-to-end test of: compiler → cubin → QMD → pushbuffer → GPU → result.

**Acceptance criterion:** vadd output matches CPU reference to bit precision.

---

### Stage 7: Full Layer Kernels

With register allocator and scheduling pass in place, compile the inference kernels:

1. `inference/gemv.ls` — W4A16 GEMV, the workhorse of every projection
2. `inference/reduce.ls` — RMSNorm, L2Norm (used by almost every layer)
3. `inference/elementwise.ls` — residual add, SiLU
4. `inference/recur.ls` — Conv1D, DeltaNet step (requires computed-offset indexing fix first)
5. `inference/attend.ls` — full attention (DeltaNet path)
6. `inference/attend_full.ls` — full attention (transformer path)

**Acceptance criterion for each:** per-kernel cosine similarity = 1.0 against PyTorch reference for the same inputs and weights.

---

### Stage 8: Megakernel Assembly

Two cooperative megakernels replace the per-layer dispatch:

- **DeltaNet cubin**: 48 DeltaNet layers, grid-wide sync between layers
- **Attention cubin**: 16 full-attention layers, grid-wide sync between layers

This requires:
- Cooperative grid-sync primitives in emit-gpu.ls (release/acquire fence + global barrier counter)
- `EIATTR_COOPERATIVE_GROUP_INSTR_OFFSETS` in `.nv.info.<kernel>`
- Language construct for "run body N times with grid sync between iterations"

**Acceptance criterion:** Each megakernel produces cosine = 1.0 at every layer against PyTorch.

---

### Stage 9: End-to-End Inference

Launcher.ls dispatches both megakernels per token. Known-good test:

```
Input:  "The capital of France is"
Output: "Paris"  (logit 18.057, cosine 1.0 at all 64 layers vs PyTorch)
```

This is the proof-of-correctness milestone. No performance numbers are reported until this passes.

---

## Part II: Implementation Details

### Register Allocator Data Structures

```
\\ Instruction record — flat buffer of 5 u32 per instruction
\\ [0] opcode
\\ [1] dst_reg (virtual ID, 0xFFFFFFFF = no destination)
\\ [2..4] src_regs[0..2] (virtual IDs, 0xFFFFFFFF = unused)
buf insn_buf 1048576    \\ room for 200K instructions
var insn_count 0

\\ Liveness sets — one u32 bitset per instruction, indexed by physical reg / 32
\\ For 255 registers: 8 u32 words per instruction
buf live_out_buf 6553600  \\ insn_count * 8 u32
buf live_in_buf  6553600

\\ Physical register free list
buf free_regs 1024         \\ 255 entries max
var free_regs_top 0

\\ Virtual → physical mapping
buf reg_map 65536          \\ one u32 per virtual ID
```

### Control Word Fields (Hopper sm_90a)

The 64-bit control word layout (from probe corpus, empirically verified):

```
bits [40:0]   — extra41: opaque barrier/descriptor fields (set to 0 for most instructions)
bits [44:41]  — stall: cycles to stall before next instruction (0-15)
bit  [45]     — yield: scheduler yield hint (0 = no yield)
bits [48:46]  — wbar: write barrier slot (7 = none)
bits [51:49]  — rbar: read barrier slot (7 = none)
bits [57:52]  — wait: barrier wait mask (6 bits, one per barrier slot)
bits [62:58]  — reuse: register reuse cache flags (5 bits)
bit  [63]     — reserved (0)
```

For a conservative but correct default: stall=15, yield=0, wbar=7, rbar=7, wait=0, reuse=0.
For a scheduled instruction: stall=computed, others as appropriate for memory/barrier use.

### Cubin ELF Section Layout

The 9 required sections (in order):

| # | Name | Purpose |
|---|------|---------|
| 0 | (null) | Required ELF null section |
| 1 | `.nv.info` | Global NVIDIA info attributes |
| 2 | `.nv.info.<kernel>` | Per-kernel attributes: REGCOUNT, FRAME_SIZE, PARAM_CBANK, KPARAM_INFO, MAX_THREADS |
| 3 | `.nv.constant0.<kernel>` | Constant bank 0 (parameter block) |
| 4 | `.text.<kernel>` | raw Hopper binary instruction words |
| 5 | `.nv.callgraph` | Call graph (empty for leaf kernels) |
| 6 | `.nv.shared.<kernel>` | Shared memory descriptor (if used) |
| 7 | `.symtab` | Symbol table: one STB_GLOBAL / STT_FUNC / STO_CUDA_ENTRY (0x10) entry per kernel |
| 8 | `.shstrtab` | Section name string table |

The symbol table entry for a kernel requires `st_other = 0x10` (STO_CUDA_ENTRY). Without this, the driver rejects the cubin.

### Cooperative Grid-Sync Encoding

For a Hopper cooperative megakernel, the grid-wide sync at binary encoding level is:

```
1. MEMBAR.SC.GL           — store-complete fence, global scope
2. ATOMS.ADD.U32 [sync_counter], 1  — atomic increment of global counter
3. ATOMS.CAS or polling loop        — wait until counter reaches grid_size
4. MEMBAR.SC.GL           — acquire fence before resuming
```

The cubin must advertise cooperative launch capability via:
- `EIATTR_COOPERATIVE_GROUP_INSTR_OFFSETS` in `.nv.info.<kernel>` — byte offsets of every grid-sync site
- The QMD `cooperative` flag must be set to 1
- Launch must use the cooperative launch method (`SEND_SIGNALING_PCAS2_B = 0x0a`)

The `record_gridsync` composition in emit-gpu.ls already tracks sync site offsets. The ELF writer must emit them into `.nv.info.<kernel>`.

---

## Part III: Iteration Protocol

Compiler authoring is hard because the failure modes are silent, the debug surface is minimal, and a single wrong bit can produce a kernel that hangs rather than crashes. The following protocol reduces the cost of each iteration.

### Principle 1: One Variable at a Time

Never change the compiler and the test kernel simultaneously. If a test fails, you do not know which one is wrong. The discipline:

- To test a new compiler feature: use an existing, known-good test kernel.
- To test a new kernel: use the existing, known-good compiler.
- To test both: test the compiler first, then the kernel.

### Principle 2: The Smallest Failing Case

When something breaks, reduce to the minimum input that reproduces the failure before attempting a fix. For a compiler:

```
Does vadd still work?          → if no, the change broke something fundamental
Does a 2-instruction kernel work?  → if no, isolate to one instruction
Does a 1-instruction kernel work?  → if no, the emitter itself is wrong
```

Binary search on the instruction sequence, not on the source file.

### Principle 3: Emit, Inspect, Then Run

Before executing a kernel, inspect the emitted binary:

```
1. Compile the .ls kernel to a cubin
2. Run nvdisasm on the cubin text section
3. Compare disassembly to what you expected
4. Only if the disassembly is correct: run the kernel
```

A kernel that disassembles incorrectly will not give useful output. Do not debug execution behavior caused by wrong emission — fix the emitter first.

### Principle 4: Reference Points

Maintain at least three reference points at all times:

| Reference | Purpose |
|-----------|---------|
| A PTX kernel compiled by ptxas for the same operation | Ground truth for binary instruction structure and control words |
| The probe corpus (docs/encoding/*.md) | Ground truth for individual instruction encodings |
| A known-passing Lithos kernel (vadd.ls) | Regression guard — if this breaks, stop |

When a new kernel fails, compare its disassembly against the PTX-derived reference. The difference is the bug.

### Principle 5: Checkpoint Before Every Change

Before modifying the compiler:

```
1. Run the regression suite (at minimum: compile and disassemble vadd.ls)
2. Record the current state in a comment block at the top of the file:
   \\ CHECKPOINT 2026-04-14: vadd passes, stage1 builds, gemv fails at insn 47
3. Make the change
4. Run the regression suite again
5. If it regresses: revert to the checkpoint, not to an intermediate state
```

The comment block is the checkpoint. Git is the backup.

### Principle 6: Control Words Last

Do not attempt to tune control words until the instruction words are correct. The order:

1. Get the opcode and operand fields right (check via nvdisasm)
2. Use conservative defaults for control words (stall=15, barriers=7, reuse=0)
3. Verify the kernel produces correct output with conservative words
4. Only then apply the scheduling pass to improve performance

A kernel with correct opcodes and conservative control words runs slowly but correctly. A kernel with tuned control words and wrong opcodes does not run at all.

### Principle 7: The Cosine Test

For every inference kernel, the correctness test is cosine similarity against PyTorch reference:

```python
import torch
import numpy as np

reference = torch.load("reference_output.pt").float().numpy()
lithos_out = np.frombuffer(gpu_output_buffer, dtype=np.float32)
cosine = np.dot(reference.flatten(), lithos_out.flatten()) / (
    np.linalg.norm(reference) * np.linalg.norm(lithos_out)
)
assert cosine > 0.9999, f"cosine={cosine:.6f}"
```

This catches: wrong register allocation (garbage values), wrong arithmetic order (wrong result), precision loss (low cosine), and silent kernel failures (cosine = 0 or NaN).

Do not report a kernel as working until its cosine = 1.0 (to 4 decimal places).

### Principle 8: The Iteration Log

Compiler bugs often reappear. Keep a flat log of every bug found and fixed:

```
docs/compiler-bugs.md

## 2026-04-14 — register ID collision in liveness pass
Symptom: gemv output NaN at position 128
Cause:   live_out bitset indexed by virtual ID mod 32; two IDs collided
Fix:     virtual ID space partitioned by type (freg 0-127, rreg 128-255, preg 256-263)
Regression test: gemv_256_element case added to test suite
```

When a new bug appears, check this log first. The second occurrence of a bug is always faster to fix than the first — if you recorded the first.

### Principle 9: What the Compiler Cannot Tell You

A GPU kernel that is wrong will often produce:
- Output that looks plausible but has cosine 0.97 against reference (off-by-one in a loop bound)
- NaN at a specific tensor position (uninitialized register, wrong memory address)
- Silent hang (wrong barrier count, wrong cooperative launch flag)
- Wrong token (correct kernel, wrong weight loaded)

None of these produce an error message. The only signals are:
- The completion flag (did the kernel finish at all)
- The output values (are they numerically correct)
- nvdisasm output (is the emitted code structurally correct)

Build the validation infrastructure (cosine test, completion timeout, disassembly inspection) before the kernels. Debugging without them is guessing.

---

## Appendix: Current Blocker Summary

As of 2026-04-14:

| Blocker | File | What is needed |
|---------|------|----------------|
| Bootstrap parser: composition syntax | `bootstrap/` parser | Parse `name args :` + indented body |
| Bootstrap parser: binding syntax | `bootstrap/` parser | Parse `name expr` without `=` |
| Register allocator | `compiler/compiler.ls` parser section | Linear-scan with liveness pass |
| pushbuffer 5-part launch | `runtime/pushbuffer.ls` | QMD + fence + context + SPD + patch |
| Cooperative grid-sync | `compiler/emit-gpu.ls` | MEMBAR + ATOMS pattern + ELF advertising |
| Tokenizer | new file | At minimum: hardcoded token IDs for test prompt |
| KV cache + state allocation | new file: `inference_init.ls` | Allocate and zero-initialize at startup |

Everything above the register allocator can be developed on this machine without the GH200.
Everything at or below cooperative grid-sync requires the GH200 for execution testing.
