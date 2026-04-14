# Register Utilization Strategy: The Central Design Constraint

*This is not a performance optimization. It is the primary design goal of the Lithos compiler.
Everything else — correctness, scheduling, ELF structure — is in service of this.*

---

## Why This Document Exists

The Lithos compiler is a difficulty-9 problem. Six points of that difficulty come from
compiling pure math to raw GPU binary. The remaining three points come from one requirement
that was not stated in the base analysis: **the compiler must make maximal use of the GPU's
FP32 register space by packing 4-bit model weights into registers and streaming them through
a pipeline.**

This is not an optimization applied after a working compiler exists. It is a design constraint
that must be present from the first architectural decision. A compiler that allocates registers
naively — assigning a fresh register to every value and never reusing — will produce kernels
that run at 10-20% of hardware capability regardless of how correct the opcodes are.

This document states precisely why, and what must be done about it.

---

## Part I: The Hardware Reality

### Hopper Register Architecture

Every thread on an SM90a Hopper GPU has access to 256 general-purpose 32-bit registers
(R0–R255). These are the fastest storage on the chip. An FP32 value in a register is
consumed by an FFMA instruction in 4 cycles. A value in global memory (DRAM) takes 200–400
cycles to arrive. Shared memory is 20 cycles. There is no faster place to hold a value
than a register.

The SM (streaming multiprocessor) has a **total register file of 65,536 32-bit registers**,
shared across all resident threads. This is fixed silicon. How those registers are divided
determines how many threads can run simultaneously.

### The Occupancy Table

The number of registers a kernel uses per thread directly controls how many warps
(groups of 32 threads) can be resident on the SM simultaneously:

| Registers per thread | Warps per SM | Threads per SM | Occupancy |
|---------------------|--------------|----------------|-----------|
| 32 | 64 | 2,048 | 100% |
| 48 | 42 | 1,344 | 66% |
| 64 | 32 | 1,024 | 50% |
| 80 | 25 | 800 | 39% |
| 96 | 21 | 672 | 33% |
| 128 | 16 | 512 | 25% |
| 192 | 10 | 320 | 16% |
| 255 | 8 | 256 | 12.5% |

This is not a soft guideline. It is arithmetic. 65,536 total registers ÷ register count per
thread = maximum resident threads. There is no way around it.

### Why Occupancy Determines Throughput

The GPU hides memory latency by switching between warps. When warp A issues an LDG
(global memory load) that will take 200 cycles, the SM switches to warp B, then C, then D,
while A's data is in flight. When A's data arrives, the SM switches back. This is the
fundamental throughput mechanism.

If only 8 warps are resident (255 registers/thread), and each warp can hide at most
~200 cycles of latency before it must stall waiting for data, the SM runs out of warps
to switch to and sits idle waiting. The arithmetic pipeline stalls.

With 32 warps resident (64 registers/thread), there are 4× more warps to switch between.
The SM can sustain the memory latency without stalling. Arithmetic throughput approaches
the theoretical maximum.

**The register count per thread is the single most important number the compiler produces.**

For the Lithos megakernels targeting DeltaNet and full-attention layers on the GH200,
the goal is **≤64 registers per thread** wherever achievable. This is the boundary between
50% and lower occupancy. Above 64, performance drops sharply and does not recover.

---

## Part II: 4-Bit Weights and the Packing Strategy

### The Problem Without Packing

The GPTQ W4A16 inference model (Qwen 3.5 27B) stores weights at 4 bits per element,
packed 8 per 32-bit integer. A GEMV row of K=5120 elements requires K/8 = 640 packed
u32 loads. Each load produces 8 weight values that must be dequantized and multiplied
against activation values before being discarded.

The naive register usage for one iteration of the GEMV inner loop (processing one packed u32):

```
packed       — the loaded u32                              (1 register)
nib0..nib7   — 8 nibble extractions                        (8 registers)
fnib0..fnib7 — 8 INT→FP32 conversions                     (8 registers)
zn0..zn7     — 8 zero-point subtractions (- 8.0)          (8 registers)
dq0..dq7     — 8 scale multiplications                     (8 registers)
xval0..xval7 — 8 activation loads                         (8 registers)
acc          — running accumulator                          (1 register)
sc           — scale for this group                        (1 register)
k_p, grp, etc — loop control                               (4 registers)
```

Naive total: **48 registers** for just the inner loop body, before any outer loop
control, RMSNorm state, or other kernel logic is included.

A fully-inlined DeltaNet layer (RMSNorm + Q/K/V/gate/beta projections + Conv1D +
decay gate + delta update + output gate + residual) with naive allocation runs well above
128 registers. That puts the kernel at 25% occupancy or below. The SM spends most of its
time waiting for data.

### The Packing Strategy

The key insight: **the 8 nibbles from a single packed u32 do not all need to be live
simultaneously.** The compiler can serialize the extraction, dequantize, multiply, and
accumulate one nibble at a time, reusing the same 3-4 registers for each.

The streaming pattern:

```
Phase 1 — load and hold:
  packed  = load W_packed[wp_idx]       → R_pack  (1 register, held for all 8 nibbles)
  sc      = load scales[grp]            → R_scale (1 register, held for all 8 nibbles)
  acc exists already                    → R_acc   (1 register, carried across loop)

Phase 2 — process nibble N (repeats 8 times, reusing R_tmp each time):
  R_tmp = packed >> (N*4)               IMAD.SHF or SHF instruction
  R_tmp = R_tmp & 0xF                   LOP3.IMM  
  R_tmp = INT→FP32(R_tmp)              I2FP
  R_tmp = R_tmp - 8.0                   FADD_IMM
  R_tmp = R_tmp * sc                    FMUL
  R_xval = load x[x_base + N]          LDG (one register for the activation)
  acc = fma(R_tmp, R_xval, acc)         FFMA
  // R_tmp and R_xval are dead — reused for nibble N+1
```

**This pattern uses 5 registers** (R_pack, R_scale, R_acc, R_tmp, R_xval) instead of 48.
The remaining space (59 registers to reach the 64-register target) is available for
loop control, outer kernel state, RMSNorm accumulators, and other live values.

The tradeoff: serializing the 8 nibbles means they cannot be computed in parallel.
On Hopper, the FFMA throughput (4 cycles latency, 1 per cycle throughput) means the
pipeline is not stalled by serialization — the register pressure benefit far outweighs
the instruction-level parallelism cost, because the real throughput limiter is memory
bandwidth and occupancy, not FP32 ALU.

### The Streaming Pipeline Model

The compiler's job is to emit code that maintains a **streaming register pipeline**:
at any given cycle, the register file holds exactly the live state needed for the
current and immediately upcoming instructions, no more.

For the GEMV kernel, the streaming state is:

```
Always live (loop duration):
  R_acc      — running dot product accumulator
  R_row_base — base address for this output row
  R_k_p      — current packed-word loop index
  R_k_max    — loop bound (K_packed)
  R_stride   — thread stride (blockDim = 256)

Live for one packed word (reused across 16 packed words per group):
  R_scale    — current group scale (reloaded every 16 iterations)
  R_x_base   — x array base for this packed word (= k_p * 8)

Live for one nibble (reused 8 times per packed word):
  R_pack     — current packed u32 (held while processing its 8 nibbles)
  R_tmp      — scratch: nibble → int → fp32 → centered → scaled
  R_xval     — one activation element

Total: 10 registers for the GEMV inner loop
```

A kernel with a 10-register inner loop and ~15 registers of outer state runs at
**25 registers total**, achieving near-100% occupancy on Hopper (64 warps/SM).
This is the target the compiler must be designed to hit.

---

## Part III: What the Compiler Must Do

The register allocator in `regalloc_design.md` describes *how* to build a correct
linear-scan allocator. This section describes what the allocator must *optimize for*
and what architectural decisions enable it.

### Requirement 1: Liveness Precision Must Be Tight

The allocator's output quality is limited by the precision of its liveness analysis.
A conservative liveness estimate (keeping values live longer than necessary) wastes
registers. A precise estimate allows maximum reuse.

The critical case is the nibble processing loop. Each nibble uses R_tmp to hold a
value that is dead after the FFMA that consumes it. If the liveness analysis marks
R_tmp as live from the nibble shift to the end of the loop body (instead of just to
the FFMA), it cannot be reused for the next nibble's operations.

**The allocator must mark a virtual register dead at its last use, not at the end
of its syntactic scope.** This is the difference between 5 registers and 48 registers
for the GEMV inner loop.

The rule: the `last-use` of a virtual register is the index of the last instruction
that reads it. The `expire-old` pass must free it at `last-use + 1`, not at the end
of the enclosing block.

### Requirement 2: Loop-Carried Values Must Be Identified Correctly

Three values in the GEMV kernel are carried across loop iterations:

- `R_acc` — accumulates across all K packed words. Defined at loop entry (MOV_IMM 0.0),
  live for the entire K loop, read at loop exit for the warp reduction.
- `R_k_p` — loop counter. Defined at loop entry, incremented at loop tail.
- `R_row_base` — row base address. Defined once before the loop, never modified inside.

These three must be assigned physical registers that are held across the entire loop body.
The allocator's loop extension rule (from `regalloc_design.md`) handles this: any vreg
whose raw `[def, last-use]` range crosses a loop boundary is extended to cover the full
`[loop_head, loop_tail]` range.

If this extension is missing or wrong, the allocator will free the accumulator register
mid-loop and assign it to a transient value. The FFMA at the bottom of the loop writes
a garbage value into what was the accumulator. The kernel produces incorrect output
silently.

**The loop extension rule is not an optimization — it is a correctness requirement.**

### Requirement 3: The Register Budget Is a Compiler Input

The Lithos `regcap` directive sets a hard ceiling. For the GEMV kernel:

```
gptq_gemv W_packed scales x y :
    regcap 64
    ...
```

If the compiler allocates more than 64 registers and cannot spill, it must fail loudly:

```
FATAL: register pressure exceeded in gptq_gemv
  peak live: 71 registers at IR instruction 847
  regcap:    64
  hint: the live set at IR 847 contains:
    R_acc (F32, live 0..2047)
    R_row_base (I64, live 0..2047)
    R_k_p (I32, live 184..847)
    R_scale (F32, live 832..847)
    R_tmp (F32, live 843..847)
    R_xval (F32, live 845..847)
    R_tmp_shift (I32, live 844..847)   <-- unexpected; should be dead by 843
```

This error message tells the programmer exactly where the pressure is and which value
is unexpectedly live. The programmer can then either restructure the kernel (serialize
more operations) or lower the `regcap` expectation.

**A compiler that silently uses 71 registers when 64 were requested is worse than one
that fails loudly.** The 71-register kernel runs at 50% occupancy and the programmer
has no idea why inference is slower than expected.

### Requirement 4: The Allocator Must Understand Register Classes

Hopper has multiple register files, and using the right one for the right value is
part of achieving the 64-register target.

**Uniform registers (UR0–UR63):** Values that are identical across all 32 threads in a warp.
These do not consume FP32 register space. Loop bounds, base addresses, and constants are
natural candidates.

In the GEMV kernel:
- `K_packed` (the loop bound) — same for every thread → UR
- `row_base` (base address for the row) — same for every thread in a block → UR
- `stride` (= 256, blockDim.x) — same for every thread → UR

Moving these three values to UR registers frees 3–4 FP32 registers, which may be the
difference between fitting in 64 and spilling.

**The compiler must infer which values are warp-uniform and allocate them to UR registers,
not to FP32 registers.** A value is warp-uniform if:
- It is a compile-time constant
- It is loaded from constant memory (LDC/ULDC)
- It is derived by arithmetic from only warp-uniform values
- It is `blockIdx.x/y/z`, `blockDim.x/y/z`, or `gridDim.x/y/z`

Warp-uniform values that are mistakenly placed in FP32 registers waste one FP32 slot
per value. In a tight budget, this is significant.

**Predicate registers (P0–P6):** Boolean conditions used for branches and predicated
execution. These do not consume FP32 registers at all. There are only 7 per thread.
The allocator has a separate 7-slot pool. The GEMV kernel uses at most 2 predicates
simultaneously (loop continuation + bounds check). This is safe.

### Requirement 5: The dequantization chain must be a single-register pipeline

This is the central implementation pattern that the compiler must recognize and enforce.

The 8-nibble dequantization chain for one packed u32 is not 8 independent computations.
It is a **sequential pipeline over one register**. The correct IR for this pattern is:

```
IR 0:  LDG     v_pack  ← W_packed[wp_idx]        // load packed u32
IR 1:  LDG     v_sc    ← scales[grp]             // load scale
IR 2:  LDG     v_x0    ← x[x_base + 0]           // load activation 0
IR 3:  LOP3    v_tmp   = v_pack & 0xF            // nibble 0
IR 4:  I2FP    v_tmp   = INT→FP32(v_tmp)         // convert (reuse v_tmp)
IR 5:  FADD    v_tmp   = v_tmp - 8.0             // center (reuse v_tmp)
IR 6:  FMUL    v_tmp   = v_tmp * v_sc            // scale (reuse v_tmp)
IR 7:  FFMA    v_acc   = v_tmp * v_x0 + v_acc    // accumulate; v_tmp dead, v_x0 dead
IR 8:  LDG     v_x1    ← x[x_base + 1]           // load activation 1
IR 9:  SHF     v_tmp   = v_pack >> 4             // nibble 1 (v_tmp reborn)
...
```

After IR 7, both `v_tmp` and `v_x0` are dead. The allocator must free their physical
registers before IR 8. Then `v_x1` at IR 8 and `v_tmp` at IR 9 take those same
physical registers.

**The two physical registers occupied by v_tmp and v_x0 are the only two registers
cycling through all 8 nibble iterations.** Everything else is stable for the duration
of the packed word. This is achievable only if liveness is computed to instruction
granularity and the expire-old pass runs before each new definition, not after.

---

## Part IV: The Megakernel Register Budget

A fully-inlined DeltaNet layer executing as a cooperative megakernel has more live state
than a single GEMV. The full layer register budget must be tracked as a whole.

### DeltaNet Layer Live State at Peak Pressure

The peak register pressure occurs during the delta rule state update, which requires
simultaneously live:

| Value | Registers | Reason |
|-------|-----------|--------|
| GEMV accumulator | 1 | Q/K/V projections (can be serialized) |
| DeltaNet state S | 16 heads × 1 pointer | Only pointer, not data |
| K vector (current) | 1 pointer + streaming window | Loaded in chunks |
| V vector (current) | 1 pointer + streaming window | Loaded in chunks |
| β (beta scalar) | 1 | Held for one delta update |
| decay scalar | 1 | Per-head, computed once |
| RMSNorm accumulator | 2 | sum_sq + count |
| Residual pointer | 1 | For residual add |
| Loop control | 4–6 | head index, dim index, loop bounds |
| Uniform registers | UR file | Base addresses, strides, block dims |
| **Estimated total** | **~35–45 FP32 registers** | With tight allocation |

This is within the 64-register budget — but only with tight allocation. A monotone
bump allocator would use 150+ registers for the same kernel, putting it at under 20%
occupancy.

The difference between a tight allocator and a naive one is not 10–20% performance.
It is the difference between a kernel that runs at 50% occupancy (32 warps/SM, adequate
latency hiding) and one that runs at 12.5% occupancy (8 warps/SM, constant stalls).
On memory-bound workloads like GEMV, this is roughly a **4× throughput difference**.

### The regcap Protocol for Megakernels

Each kernel must be compiled with an explicit regcap that reflects the hardware target:

```
\\ DeltaNet megakernel: target 50% occupancy (32 warps/SM)
\\ 65536 total regs / 32 warps / 32 threads/warp = 64 regs/thread
deltanet_megakernel :
    regcap 64
    ...

\\ Full attention megakernel: GQA means less state, can afford tighter
\\ Target 66% occupancy (42 warps/SM) ~ 48 regs/thread
attention_megakernel :
    regcap 48
    ...
```

If the compiler cannot meet the regcap, it must fail. The programmer must then either:
1. Split the kernel into smaller pieces (reduces fusion efficiency but enables the budget)
2. Serialize more operations in the source (reduces ILP but enables the budget)
3. Move more values to shared memory explicitly (adds latency but enables the budget)

No silent degradation. The register count must be visible and controlled.

---

## Part V: Verification

### How to Verify Register Count

After emitting a cubin, extract the register count from the `.nv.info.<kernel>` section:

```bash
nvdisasm --print-code cubin.cubin | grep REGCOUNT
```

Or read it from the SPD at offset 0x094 (bits 23:16) before launch. This is the actual
register count the driver will use. It must match what `regcap` specified.

If it is higher: the allocator is not expiring values correctly. Use `--dump-ra` to
identify which virtual register is staying live past its last use.

If it is lower: the allocator found more reuse than expected. This is fine — but verify
that the output is still correct (cosine = 1.0 against PyTorch reference). A too-low
register count can indicate that a loop-carried value was incorrectly freed mid-loop.

### The Occupancy-Correctness Correlation

A kernel that suddenly achieves suspiciously high occupancy (e.g., drops from 25 to 64
registers after a compiler change) should be treated as a bug suspect until proven
correct by the cosine test. The most common cause is the loop extension rule failing,
which allows an accumulator to be freed and reused mid-loop — reducing register count
while silently corrupting output.

**The cosine test is the only reliable correctness check.** Register count alone is not
sufficient. A kernel can use 30 registers and produce completely wrong output if one of
those registers was incorrectly shared between a live accumulator and a transient value.

---

## Summary

The register utilization strategy is not a late-stage optimization. It is the reason
the Lithos compiler is a difficulty-9 problem instead of a difficulty-6 problem.

The three non-negotiable requirements:

1. **The liveness analysis must be precise to instruction granularity.** Values must be
   freed at their last use, not at the end of their syntactic scope. The GEMV nibble
   pipeline depends on this absolutely.

2. **Loop-carried values must be identified and held correctly.** The accumulator that
   crosses the K loop boundary must never be freed mid-loop. This is correctness, not
   performance.

3. **Uniform values must go to UR registers.** Loop bounds, base addresses, and constants
   that are warp-invariant must not consume FP32 register budget. Every FP32 register
   recovered by moving a value to UR is one step closer to the 64-register target.

Everything else the compiler does — correct opcode encoding, valid ELF structure,
proper control word scheduling — is the foundation that makes these three things matter.
Without correct register utilization, a correct and well-scheduled kernel still runs
at a fraction of GPU capability. The goal is not a kernel that produces the right answer.
The goal is a kernel that produces the right answer at hardware speed.
