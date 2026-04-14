# Weights-as-Code Compiler Architecture

## The Fundamental Insight

With Q4-quantized models and direct 64-bit GPU instruction emission, model weights
can be encoded as immediates in arithmetic instructions. Instead of loading weights
from global memory at runtime (LDG), the compiler reads safetensors at compile time
and emits multiply-accumulate instructions where the 32-bit immediate fields ARE the
weights. The model becomes the program. .text IS the weights.

This eliminates the memory hierarchy for weight access entirely. Weights arrive via
the instruction fetch path: L1I -> instruction buffer -> dispatch. There is no LDG
latency, no DRAM bandwidth wall, no cache thrashing between weight tiles and
activations. The GPU's instruction delivery system becomes the weight delivery system.

---

## 1. New Compilation Pipeline

### Current pipeline (memory-bound GEMV)

```
.ls template  -->  parser.fs  -->  emit-gemv.fs  -->  Hopper binary
                                       |
                                  emits LDG loop:
                                  load packed u32 from DRAM
                                  dequant 8 nibbles
                                  FMA with activation
                                  stride, repeat
```

Weight matrix lives in DRAM. The kernel is small (~1 KB). The data is large (~14 GB
for 27B Q4). Every token requires streaming the entire weight matrix through the
memory hierarchy.

### New pipeline (weights-as-code)

```
.ls template  ----+
                  |
safetensors   ----+--> lithos-compile  -->  per-layer ELF binaries
(model file)      |
                  |
                  v
           For each layer:
             1. Parse safetensors header (JSON) to locate tensors
             2. For each `project` primitive in the .ls template:
                a. Read the weight tensor (Q4 packed u32s + scales)
                b. For each output row:
                   - For each packed u32 in that row:
                     Emit 8 FMUL-IMM instructions (one per nibble)
                     with pre-dequantized FP32 weight as immediate
                   - Interleave scale application every 128 weights
                c. Emit reduction (unchanged: SHFL.BFLY tree)
             3. Emit elementwise ops (unchanged)
             4. Emit grid-sync between layers
             5. Write ELF section
```

### Input artifacts

- **Template:** `.ls` file defining the layer structure (project, normalise, silu,
  etc.). One template covers all layers of the same type (e.g., all DeltaNet layers
  share one template; the LM head has its own).

- **Safetensors:** Standard HuggingFace format. Binary file with:
  - 8-byte little-endian header length
  - JSON header mapping tensor names to {dtype, shape, data_offsets}
  - Raw tensor data (contiguous, no padding between tensors)
  - Q4 GPTQ tensors: `qweight` (packed u32), `qzeros` (packed u32),
    `scales` (FP16), `g_idx` (optional permutation)

- **Output:** One ELF binary per layer (or per megakernel segment). The .text
  section contains the weight-embedded instruction stream. No .data section for
  weights.

---

## 2. FMUL-IMM Packing Format: Option Analysis

The central question: how to encode Q4 weights into instructions.

### Option A: Pre-dequantize to FP32, one FMUL-IMM per weight

Each Q4 nibble (0-15) is dequantized at compile time:

```
fp32_weight = (nibble - zero_point) * scale
```

This FP32 value is embedded directly in FMUL-IMM's 32-bit immediate field:

```
FMUL-IMM  Racc, Ractivation, <fp32_weight>    ; Racc = activation * weight
FADD      Raccum, Raccum, Racc                 ; accumulate
```

**Instructions per weight:** 2 (FMUL-IMM + FADD) or 1 (FFMA with immediate, if
available -- but FFMA on sm90 takes Rs3 from ctrl word, not an immediate field, so
this does not exist as a single instruction).

Actually: FMUL-IMM produces a product. To accumulate, we need FADD. So each weight
costs:
- 1 FMUL-IMM (multiply activation by weight-immediate): 16 bytes
- 1 FADD (accumulate into running sum): 16 bytes

**Total: 2 instructions = 32 bytes per weight.**

But wait -- the activation register must be loaded first. For a GEMV row of K
elements, the activation vector has K elements. Each thread handles a subset. The
activation for element k must be in a register before the FMUL-IMM. Since we are
doing straight-line code (no loop), the activation loads must also be unrolled.

Revised per-weight cost for Option A:
- If activations are pre-loaded into registers: 2 instructions (FMUL-IMM + FADD)
- If activations are loaded inline: 3 instructions (LDG + FMUL-IMM + FADD), but
  the LDG can be pipelined/hoisted

With register spilling considered (Hopper has 255 registers, so at most ~250
activations in registers at once), batching is needed: load a tile of activations,
execute the weight-immediate FMAs for that tile, repeat.

**Instruction bytes per weight: 32 bytes** (FMUL-IMM + FADD, ignoring activation
load overhead which is amortized across rows).

**Density: 32 bytes per weight.** A Q4 weight is 0.5 bytes in memory. This is a
**64x expansion** in size. The model is encoded 64 times larger than its weight data.

### Option B: Keep Q4 packed, integer extract at runtime

Pack 8 Q4 nibbles into one 32-bit immediate. Use MOV-IMM to load the packed word,
then the existing dequant-8 sequence (SHF, LOP3, I2F, FFMA) to unpack.

```
MOV-IMM  Rpacked, <packed_u32>    ; 8 weights in 32 bits
; then existing 48-instruction dequant-8 sequence (emit-dequant-8)
; but with x-elements loaded from activation registers, not LDG
```

**Instructions per 8 weights:** 1 (MOV-IMM) + ~40 (dequant-8 minus the 8 LDG loads
for x, which become register reads) = ~33 instructions.

Actually let us recount emit-dequant-8 from emit-gemv.fs. Per 8 nibbles:
- 1 MOV-IMM for neg8_scale setup
- 1 FFMA for neg8_scale computation
- Per nibble: 1 SHF (except nib0) + 1 LOP3 + 1 I2F + 1 FFMA (dequant) + 1 LDG (x) + 1 FFMA (accum)
- Nibble 0: no SHF, so 5 ops; nibbles 1-7: 6 ops each
- Total: 2 + 5 + 7*6 = 49 instructions

With weights-as-code, the LDG for packed word is replaced by MOV-IMM. The 8 LDG
loads for x elements are replaced by register moves (free if activations are
already in registers) or stay as LDG from activation buffer. The scale LDG becomes
either another MOV-IMM (embedding scale as immediate) or a register read.

Replacing packed LDG with MOV-IMM and keeping activation LDGs:
- 1 MOV-IMM (packed weights)
- 1 MOV-IMM (scale, if embedding scale too) or 1 LDG (scale from buffer)
- 2 (neg8_scale setup)
- 8 nibbles: same dequant sequence = 47 more instructions
- Total: ~50 instructions per 8 weights = ~6.25 instructions per weight

**Instruction bytes per weight: 6.25 * 16 = 100 bytes per 8 weights = 12.5 bytes
per weight.** But the packed immediate holds 8 weights in 4 bytes, so the marginal
cost of the weight data itself is 0.5 bytes per weight (same as memory). The
remaining 12 bytes/weight is compute instructions.

Wait -- this is worse than Option A's 32 bytes/weight in total instruction size.
But the runtime instruction count is better: 6.25 vs 2 instructions per weight for
Option A, but Option A lacks the dequant -- it pre-dequants. Let me re-examine.

**Option A instruction count per weight:** 2 (FMUL-IMM + FADD). No dequant at
runtime. Scale and zero-point are folded into the IEEE 754 immediate at compile
time.

**Option B instruction count per weight:** ~6.25 (includes runtime dequant). The
advantage: 8 weights in one 32-bit immediate = 16 bytes of instruction, vs 8 * 32 =
256 bytes for Option A. Option B is **16x denser in instruction bytes** but uses
**3x more runtime instructions** per weight.

### Option C: Pre-dequantize to FP16, pack 2 per HFMA2

HFMA2 (opcode $7235, verified in opcodes-sm90.fs) performs FP16x2 fused
multiply-add. The 32-bit immediate field holds two FP16 values (16 bits each).

```
HFMA2  Racc.H2, Ractivation.H2, <fp16_w0 | fp16_w1>
```

This processes 2 weights per instruction. Pre-dequantize each Q4 weight to FP16 at
compile time (FP16 has enough range for dequantized Q4 values: typical range is
roughly -1.0 to +1.0 after scale).

**Instructions per weight:** 0.5 HFMA2 = 8 bytes. But the accumulation is in FP16,
which loses precision. For a reduction over K=3584 elements, FP16 accumulation
introduces significant numerical error.

**Mitigation:** Accumulate in FP32. This means HFMA2 produces FP16x2 products, then
widen to FP32 and add. That adds 2 conversion + 2 FADD instructions per HFMA2:

```
HFMA2  Rtmp.H2, Ract.H2, <w0|w1>   ; 2 FP16 products
; unpack and accumulate in FP32:
; (requires HADD2 -> F2F widening -> FADD, or manual extraction)
```

This gets complicated and largely negates the density advantage.

### Verdict: Option A wins for sm90 Hopper

| Metric | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Instructions/weight | 2 | 6.25 | ~3 (with FP32 accum) |
| Bytes/weight (code) | 32 | 12.5 | ~24 |
| Runtime dequant | None | Yes | Partial (FP16 convert) |
| Precision | FP32 exact | FP32 (runtime dequant) | FP16 products, FP32 accum |
| Complexity | Trivial | Moderate | High (HFMA2 encoding unverified) |

**Option A is correct.** The reasoning:

1. **No runtime dequant.** Every cycle spent on SHF/LOP3/I2F at runtime is a cycle
   not spent on FMA. Pre-dequantizing at compile time converts weight delivery from
   a memory problem to a code-size problem.

2. **Instruction cache is not the bottleneck on Hopper.** The L1I cache is 32 KB per
   SM, and instructions stream sequentially. The instruction fetch unit on sm90 can
   deliver 1 instruction per cycle per warp (128 bits / 16 bytes). For straight-line
   code with no branches, the fetch unit achieves perfect prefetching. This is
   fundamentally different from data cache, which suffers from irregular access
   patterns.

3. **The 64x expansion is acceptable** because the instruction stream IS the
   computation. There is no separate weight data. The total memory footprint is the
   instruction stream, which replaces both the kernel code and the weight data.

4. **HFMA2 (Option C) is not probe-verified** for the immediate encoding format.
   OP-HFMA2 ($7235) appears in opcodes-sm90.fs but has no ctrl word constants and no
   emitter. Building on unverified encodings violates Lithos principles.

5. **FP32 immediate encoding is exact** for the dequantized weight range. Q4 weights
   dequantize to values like (nibble - 8) * scale, where scale is typically in the
   range [0.001, 0.1]. These are perfectly representable in IEEE 754 FP32 (24 bits
   of mantissa, more than enough for the ~4 bits of weight information).

**Option A is what we build.**

---

## 3. Activation Flow

Activations are data, not code. They live in registers and shared memory.

### Per-token activation path

For a single `project` primitive (GEMV: y = W @ x, where W is [N, K] and x is [K]):

1. **Activation vector x** lives in shared memory (loaded once per layer by the
   preceding operation's output, or from global memory for the first layer).

2. **Thread assignment:** One warp per output row (or one block per output row,
   depending on K). For inference with batch=1, each SM processes multiple output
   rows.

3. **Activation loading into registers:** Before executing the weight-immediate
   instruction stream for a given tile, threads cooperatively load a tile of the
   activation vector from shared memory into registers:

```
; Load activation tile [k_start .. k_start+TILE-1] into registers R10..R10+TILE-1
; Each thread loads its own elements (all threads in the row-warp need the same x)
; Use shared memory broadcast: all threads read x[k] from the same smem address

LDS  R10, [smem + k*4]        ; x[k+0]
LDS  R11, [smem + (k+1)*4]    ; x[k+1]
...
LDS  R10+T, [smem + (k+T)*4]  ; x[k+T-1]
```

4. **Weight-immediate multiply-accumulate:** For each weight in the row, the
   compiler has emitted:

```
FMUL-IMM  Rtmp, R10, <w[row][k+0]>    ; tmp = x[k+0] * w_0
FADD      Racc, Racc, Rtmp            ; acc += tmp
FMUL-IMM  Rtmp, R11, <w[row][k+1]>    ; tmp = x[k+1] * w_1
FADD      Racc, Racc, Rtmp            ; acc += tmp
...
```

5. **Tile boundary:** After TILE weights are processed, the next tile of activations
   must be loaded. The compiler emits another LDS block, then continues with the
   next tile of weight-immediates.

### Register budget for activation tiling

Hopper has 255 architectural registers. Reserve:
- R0-R3: thread indexing (tid, ctaid, etc.)
- Racc: accumulator
- Rtmp: scratch for FMUL-IMM result
- 2-3 registers: smem address computation

That leaves ~248 registers for activation tile. **Tile size = 248 activations.**
Every 248 weight-immediate instructions, the compiler inserts an LDS reload block.

For K=3584: 3584 / 248 = ~15 tiles. Each tile boundary costs 248 LDS instructions.
Total LDS overhead: 15 * 248 = 3720 LDS instructions vs 3584 * 2 = 7168
weight-immediate instructions. The LDS overhead is ~52% of the weight instruction
count, but LDS has 1-cycle throughput from shared memory and can be pipelined.

**Optimization: double-buffering.** Split activation registers into two halves
(124 each). While executing weight-immediates against tile A, prefetch tile B into
the other register set. This hides LDS latency completely.

With double-buffering:
- Tile size = 124 activations
- 3584 / 124 = 29 tiles
- LDS overhead per tile = 124 instructions
- Total LDS = 29 * 124 = 3596 instructions
- Total weight instructions = 3584 * 2 = 7168
- The LDS instructions overlap with weight-immediate execution (different functional
  units: LDS uses shared memory pipe, FMUL-IMM uses FP32 pipe)

### Thread-to-row mapping

For inference (batch=1), each output element requires reducing across K=3584 input
elements. With weights-as-code, each output row's entire instruction stream is
emitted sequentially. The mapping:

- **1 warp per output row:** 32 threads per row. Each thread handles K/32 = 112
  weights. After the straight-line weight-immediate sequence, a 5-stage SHFL.BFLY
  reduction gives the final output.

- **1 thread per output row:** Maximum parallelism. Each thread executes the full
  K=3584 weight-immediate stream (7168 instructions). No intra-row reduction needed.
  This is the simplest model for weights-as-code: the instruction stream for row r
  is the program for thread r.

The 1-thread-per-row model is cleanest for weights-as-code because the entire
weight sequence for a row is a single thread's instruction stream, requiring no
synchronization mid-row. With 3584 weights per row at 2 instructions each, that is
7168 instructions per row -- well within Hopper's instruction buffer capabilities
for straight-line code.

---

## 4. Scale Factors and Zero Points

### GPTQ quantization structure

GPTQ Q4 uses:
- **Group size:** 128 weights per group
- **Scale:** One FP16 scale factor per group (stored in `scales` tensor)
- **Zero point:** Packed 4-bit values in `qzeros` tensor (typically all 8, meaning
  unsigned Q4 values 0-15 are centered at 8)

### Compile-time folding

With Option A (pre-dequantize to FP32), scale and zero-point are folded at compile
time:

```python
# At compile time, for each weight:
nibble = extract_nibble(packed_u32, position)      # 0-15
zero_point = extract_nibble(qzeros[group], ...)    # typically 8
scale = float(scales[group])                       # FP16 -> FP32
dequantized = (nibble - zero_point) * scale        # FP32
# This FP32 value becomes the immediate in FMUL-IMM
```

**There are no scale-factor instructions in the output.** The scale is baked into
every weight immediate. This is the key advantage of Option A: the runtime
instruction stream is pure multiply-accumulate with no dequant overhead.

### Numerical precision

The dequantized value is exact in FP32. Proof:
- nibble - zero_point is an integer in [-15, +15] (at most 5 bits)
- scale is an FP16 value, which is a subset of FP32
- The product of a 5-bit integer and an FP16 value is exactly representable in FP32
  (5 + 11 = 16 bits of mantissa, well within FP32's 24-bit mantissa)

No precision is lost by pre-dequantizing.

### Per-group verification

The compiler should emit a comment (or debug annotation in the ELF) at every group
boundary showing the scale factor and zero point used, enabling post-hoc
verification that the weight embedding is correct.

---

## 5. ELF Size and Memory Layout

### Size calculation for one projection

Model: Qwen 3.5 27B (DeltaNet variant, using Lithos's 71-step decomposition).

Assumed dimensions (standard for 27B-class models):
- Hidden dimension D = 3584
- MLP intermediate = 4 * D = 14336 (or 3.5 * D with GQA adjustments)
- Number of layers = 36

For a standard Q/K/V projection: W is [D_out, D_in] where both are ~3584.

**Weights per projection:** 3584 * 3584 = 12,845,056

**Instructions per weight (Option A):** 2 (FMUL-IMM + FADD)

**Bytes per instruction:** 16 (sm90 128-bit instruction format)

**Instruction stream per projection:**
- 12,845,056 weights * 2 instructions * 16 bytes = 411,041,792 bytes
- **= 392 MB per projection**

Plus activation load overhead (~50% additional for LDS tiling):
- 392 MB * 1.5 = **~588 MB per projection** (upper bound; actual depends on tiling)

### Size for one DeltaNet layer

From the 71-step decomposition (STEPS file), each layer has:
- 7 `project` operations: Q, K, V (step 7-9), output gate (step 42), output
  projection (step 54), gate projection (step 62), up projection (step 63),
  down projection (step 70)

Wait -- that is 8 projections. Let me recount:
- Steps 7, 8, 9: Q, K, V projections (3 projections, each [D, D] or similar)
- Step 42: output gate z = W_z @ x (1 projection)
- Step 54: output projection (1 projection)
- Step 62: gate projection (1 projection, [MLP_DIM, D])
- Step 63: up projection (1 projection, [MLP_DIM, D])
- Step 70: down projection (1 projection, [D, MLP_DIM])

That is 8 projections per layer. Dimensions vary:
- Q, K, V: [D, D] = [3584, 3584] = 12.8M weights each
- Output gate, output proj: [D, D] = 12.8M weights each
- Gate, up: [14336, 3584] = 51.4M weights each
- Down: [3584, 14336] = 51.4M weights

Total weights per layer:
- 5 * 12.8M = 64.0M (D*D projections)
- 3 * 51.4M = 154.2M (MLP projections)
- **Total: 218.2M weights per layer**

Instructions per layer: 218.2M * 2 = 436.4M instructions
Bytes per layer: 436.4M * 16 = **~6.98 GB per layer**

### Full model

36 layers * 6.98 GB = **~251 GB for the full model instruction stream.**

Plus the LM head (one final projection [vocab_size, D]):
- vocab_size = 248,320 (Qwen 3.5)
- 248,320 * 3584 = 544.5M weights
- 544.5M * 2 * 16 = 17.4 GB
- **Total: ~269 GB**

### Does it fit?

**GH200 has 96 GB HBM3 + potential NVLink-connected memory.** The 269 GB instruction
stream does NOT fit in a single GH200's memory.

**H100 80GB SXM:** Also does not fit.

**Multi-GPU:** 4x H100 80GB = 320 GB. The instruction stream fits across 4 GPUs.

**This is the fundamental constraint.** The 64x expansion from Q4 (0.5 bytes/weight)
to instruction encoding (32 bytes/weight) means a 13.5 GB Q4 model becomes a 269 GB
instruction stream.

### Reducing the expansion factor

The 64x factor makes full-model weights-as-code impractical for single-GPU
deployment. Several mitigations exist:

**Mitigation 1: Use FFMA-IMM if it existed.** On sm90, FFMA takes three register
sources (Ra, Rb in inst word, Rc in ctrl word). There is no immediate form of FFMA.
If there were, each weight would cost 1 instruction instead of 2, halving the
expansion to 32x and the total to ~135 GB. Still too large for single GPU.

**Mitigation 2: Hybrid approach.** Embed weights-as-code for the hot path (MLP
projections are memory-bound; attention projections less so) and keep the smaller
projections as traditional LDG loops. The 3 MLP projections account for
154.2M / 218.2M = 71% of per-layer weights. Keeping only the 5 smaller projections
as LDG and embedding MLP would still be 71% * 251 = 178 GB.

**Mitigation 3: Per-layer streaming.** Do not load the entire instruction stream
into memory at once. Load one layer's instructions (~7 GB), execute, discard, load
next layer. This requires 7 GB of instruction buffer space (fits easily) but adds
PCIe/NVLink transfer latency between layers. At PCIe 5.0 x16 (64 GB/s): 7 GB /
64 GB/s = 109 ms per layer. 36 layers = 3.9 seconds per token. **Unacceptable.**
At NVLink 900 GB/s (GH200): 7 GB / 900 = 7.8 ms per layer, 280 ms per token.
Still too slow for real-time inference.

**Mitigation 4: Smaller model or smaller projections.** A 7B model with D=4096 and
28 layers would have ~60 GB of instruction stream -- fits on one 80 GB H100 with
room for activations.

**Mitigation 5: Partial embedding.** Embed only the weights that are on the critical
path for latency. For batch=1 inference, the bottleneck is memory bandwidth for
weight loading. Weights-as-code trades memory bandwidth for instruction bandwidth.
On H100, instruction fetch bandwidth is not publicly documented but is believed to
be comparable to or exceeding L1 data cache bandwidth (~200 GB/s per SM, 132 SMs =
~26 TB/s aggregate L1I bandwidth). If true, weights-as-code could be faster even
with the 64x expansion, because every byte of instruction fetch carries useful
weight data (no cache line waste, no TLB misses, perfect sequential access).

### Instruction cache hierarchy on sm90

Hopper instruction cache hierarchy (per SM):
- **L0 I-cache:** 16 KB, per sub-partition (4 sub-partitions per SM)
- **L1 I-cache:** 32 KB, shared per SM
- **L1.5 I-cache / L2:** Instruction misses go to L2 cache (50 MB on H100)
- **DRAM:** HBM3 at 3.35 TB/s (H100 SXM)

For a straight-line instruction stream with no branches:
- Prefetching is trivial (sequential addresses)
- L0/L1 miss rate is 100% (streaming, no reuse) but latency is hidden by prefetch
- L2 hit rate depends on whether the working set fits in 50 MB (it does not for a
  full layer; 7 GB >> 50 MB)
- **Effective instruction delivery rate = DRAM bandwidth**

On H100 SXM: 3.35 TB/s DRAM bandwidth, shared between instruction fetch and data
access. If the instruction stream is the dominant consumer, most of the 3.35 TB/s
serves instructions.

**Instruction throughput per weight:**
- 32 bytes per weight / 3.35 TB/s = 9.5 ps per weight (if bandwidth-saturated)
- Per row of K=3584: 34 ns
- Per layer of 218M weights: 624 us
- Per full model (36 layers): 22.5 ms per token

**For comparison, current memory-bound GEMV:**
- 0.5 bytes per weight (Q4) / 3.35 TB/s = 0.149 ps per weight
- Per full model: ~1 ms per token

The weights-as-code approach is **64x slower in bandwidth terms** if instruction
fetch and data fetch share the same DRAM bandwidth. This is the fundamental problem
with the 64x expansion.

### The real calculation: instruction issue rate, not bandwidth

However, instruction delivery is NOT bandwidth-bound in the same way. The GPU issues
instructions from the instruction buffer, which is filled by the I-cache hierarchy.
The issue rate is:

- sm90: 1 instruction per cycle per warp scheduler
- 4 warp schedulers per SM
- 132 SMs on H100
- 4 * 132 = 528 instructions per cycle
- At 1.98 GHz boost: 528 * 1.98e9 = 1.046 trillion instructions/second

**Per-weight: 2 instructions. Per-model-pass: 2 * 7.85 billion weights (27B model
Q4) = 15.7 billion instructions.**

15.7e9 / 1.046e12 = **15 ms per token** at 100% issue efficiency.

This ignores that not all warps are active, many warps stall on instruction fetch,
and issue efficiency is typically 30-60% for real workloads. At 40% efficiency:
**37.5 ms per token.** That is 26.7 tok/s -- competitive with current Q4 inference
on H100.

**But the instruction stream must fit in memory.** At 269 GB, it does not fit on
one H100. The calculation above assumes perfect streaming, which requires the
instructions to be resident in DRAM.

---

## 6. The Megakernel Question

### Current architecture

Two cooperative megakernels per model:
1. **Forward megakernel:** All layers' forward computations, connected by grid-sync.
2. **Recurrence megakernel:** DeltaNet state update (matvec, outer product, etc.).

Grid-sync between layers: all SMs complete layer N before any SM begins layer N+1.

### Weights-as-code megakernel structure

**Option: One megakernel containing ALL layers' weight-immediate streams.**

This means the forward megakernel's .text section is ~251 GB. This is impossible --
it does not fit in memory.

**Option: One ELF per layer, launched sequentially.**

Each layer's ELF is ~7 GB. Load into DRAM, launch kernel, wait for completion,
load next layer's ELF. This adds kernel launch overhead and PCIe transfer time
(if loading from host memory).

If all layer ELFs are pre-loaded into DRAM (requires 251 GB -- multi-GPU only),
kernel launches can be pipelined with near-zero gap.

**Option: Streaming megakernel with instruction-level paging.**

The megakernel is one launch. The GPU's instruction fetch unit pages in instruction
segments from DRAM as needed. This is the natural behavior -- the GPU does not
require the entire .text to be cache-resident. It fetches instructions on demand.

This is the correct architecture:
1. Allocate 251 GB across multiple GPUs (4x H100 or 3x GH200)
2. The megakernel's .text spans the full model
3. Grid-sync instructions are embedded between layers in the instruction stream
4. The instruction fetch unit streams through the .text linearly
5. Each SM executes its portion of each layer, hits grid-sync, proceeds to next

**The megakernel IS the model.** Grid-sync between layers is an instruction in the
stream, not a separate kernel launch.

### Grid-sync encoding in the weight stream

Between each layer's weight-immediate blocks, the compiler emits:

```
; Layer N complete -- grid sync
MEMBAR.SC.GPU           ; flush stores
ATOMG  [sync_counter], 1   ; signal completion
SPIN:  LDG  Rtmp, [sync_counter]   ; poll
       ISETP.LT P0, Rtmp, grid_size
       @P0 BRA SPIN
MEMBAR.SC.GPU           ; acquire barrier
; Layer N+1 begins
```

This is the same grid-sync pattern already used in Lithos (verified in emit.fs).

---

## 7. What Changes in the Forth Bootstrap

### New components needed

#### 7a. Safetensors parser (binary format)

New file: `compiler/safetensors.fs`

The safetensors format is:
1. 8 bytes: little-endian u64 = header_size
2. header_size bytes: JSON string mapping tensor names to metadata
3. Remaining bytes: raw tensor data

The Forth parser needs:
- **Read 8-byte LE integer:** Trivial in Forth (8 c@ bytes, shift-and-or).
- **Minimal JSON parser:** Only needs to extract `{name: {dtype, shape,
  data_offsets}}` entries. No nested objects, no arrays-of-arrays. A state machine
  that looks for quoted strings and colon-separated key-value pairs suffices.
- **Tensor accessor:** Given a tensor name, return (addr, dtype, shape) by looking
  up the JSON header and computing the file offset.

Words:
```forth
: st-open       ( filename-addr filename-len -- fd )      \ open safetensors file
: st-header     ( fd -- json-addr json-len )              \ read and return header
: st-tensor     ( fd name-addr name-len -- data-addr shape-addr dtype )
                                                          \ locate tensor by name
: st-close      ( fd -- )                                 \ close file
```

The raw tensor data is mmap'd (or read into a Forth buffer). For a 14 GB model
file, mmap is essential -- Forth's dictionary cannot hold 14 GB.

```forth
: st-mmap       ( fd offset len -- addr )   \ mmap tensor region
```

#### 7b. Weight-to-immediate converter

New file: `compiler/weight-embed.fs`

Reads Q4 packed tensors and produces FP32 immediates.

Words:
```forth
: q4-dequant    ( nibble zero-point scale -- fp32-bits )
                \ dequant one Q4 nibble to IEEE 754 FP32 bit pattern

: q4-row-emit   ( row-idx weight-addr scale-addr zeros-addr K -- )
                \ For one output row:
                \   for k = 0 to K-1:
                \     extract nibble from packed u32
                \     look up scale and zero point for this group
                \     dequant to FP32
                \     emit: FMUL-IMM Rtmp, R_activation[k%TILE], <fp32>
                \     emit: FADD Racc, Racc, Rtmp

: q4-project-emit  ( weight-tensor scale-tensor zeros-tensor N K -- )
                \ For each row 0..N-1:
                \   emit activation tile loads
                \   call q4-row-emit
                \   emit intra-warp reduction
```

The critical inner loop in Forth:

```forth
: q4-nibble@  ( packed-u32 nibble-idx -- nibble )
    4 * rshift $f and ;

: q4-dequant  ( nibble zero-point scale-fp32 -- fp32-bits )
    >r >r                    \ R: scale zero
    r> -                     \ nibble - zero_point (integer)
    s>f                      \ convert to float (Forth float stack)
    r> f*                    \ multiply by scale
    f>ieee754 ;              \ convert to 32-bit IEEE 754 pattern

: emit-weight-imm  ( activation-reg fp32-bits acc-reg tmp-reg -- )
    >r >r                    \ R: tmp acc
    r@ rot                   \ stack: tmp activation-reg fp32-bits; R: acc
    fmul-imm,               \ FMUL-IMM Rtmp, Ract, <fp32>
    r> r> dup >r             \ stack: acc tmp; R: acc
    r@ -rot RZ-G fadd, ;    \ FADD Racc, Racc, Rtmp
```

#### 7c. Per-layer code generation loop

New file: `compiler/layer-emit.fs`

Orchestrates the emission of one complete layer:

```forth
: emit-layer  ( layer-idx safetensors-fd -- )
    \ 1. Look up this layer's tensor names:
    \    "model.layers.{N}.self_attn.q_proj.qweight"
    \    "model.layers.{N}.self_attn.q_proj.scales"
    \    etc.
    \ 2. For each project primitive in the template:
    \    a. Resolve tensor name
    \    b. mmap tensor data
    \    c. Call q4-project-emit
    \ 3. Emit elementwise ops between projections (unchanged)
    \ 4. Emit grid-sync at layer boundary
    ;

: emit-model  ( safetensors-fd n-layers -- )
    0 do
        i over emit-layer
    loop
    drop ;
```

#### 7d. Changes to the `project` primitive

Current `project` in emit-gemv.fs emits:
- LDG loop over packed weight words
- Runtime dequant (SHF, LOP3, I2F, FFMA)
- LDG for activation elements
- Warp reduction + cross-warp reduction
- STG for output

New `project` emits:
- **No LDG for weights** (weights are in the instruction stream)
- **No runtime dequant** (pre-dequantized at compile time)
- LDS for activation tiles (from shared memory, not global)
- Straight-line FMUL-IMM + FADD sequence (the weight-embedded computation)
- Warp reduction (unchanged)
- STG for output (unchanged)

The word `emit-gemv` in emit-gemv.fs is replaced entirely. The new word:

```forth
: emit-project-wac  ( weight-data scales zeros N K -- )
    \ "WAC" = weights-as-code
    \ Emits the complete weight-embedded projection for one matrix
    \ No loops. Straight-line code. Size = N * K * 2 * 16 bytes.
    ;
```

The parser's dispatch for the `project` keyword changes:

```forth
\ Old:
\ : compile-project  ... emit-gemv ... ;

\ New:
: compile-project  ( -- )
    current-layer-tensors    \ look up weight/scale/zeros tensors
    emit-project-wac ;       \ emit weight-embedded straight-line code
```

---

## 8. Practical Viability Assessment

### The hard constraint

For a 27B model, the instruction stream is ~269 GB. This exceeds single-GPU memory.
The weights-as-code architecture as described (Option A, full embedding) is viable
only for:

1. **Small models (<=7B)** on a single 80 GB GPU
2. **Multi-GPU deployments** where the instruction stream is distributed

### Scaling analysis

| Model size | Q4 weights | Instruction stream (Option A) | Fits on... |
|------------|-----------|-------------------------------|------------|
| 1.5B       | ~0.75 GB  | ~48 GB                       | 1x H100 80GB |
| 3B         | ~1.5 GB   | ~96 GB                       | 2x H100 80GB |
| 7B         | ~3.5 GB   | ~224 GB                      | 3x H100 80GB |
| 14B        | ~7 GB     | ~448 GB                      | 6x H100 80GB |
| 27B        | ~13.5 GB  | ~864 GB (corrected*)         | 11x H100 80GB |

*Correction: The 269 GB figure above assumed only 8 projections per layer. With
full tensor accounting (including bias terms, norms, embeddings), and with the MLP
intermediate dimension being 3.5x hidden dim for Qwen architecture, the actual
figure is higher. Let me recalculate more carefully.

**Precise calculation for Qwen 3.5 27B (DeltaNet variant):**

Layer config (assumed): D=3584, MLP_DIM=18944 (5.29x, matching Qwen 3.5 architecture
which uses 18944 intermediate with SiLU gate), 36 layers.

Per-layer projection sizes:
- Q proj: [3584, 3584] = 12,845,056
- K proj: [3584, 3584] = 12,845,056
- V proj: [3584, 3584] = 12,845,056
- Output gate: [3584, 3584] = 12,845,056
- Output proj: [3584, 3584] = 12,845,056
- Gate proj: [18944, 3584] = 67,895,296
- Up proj: [18944, 3584] = 67,895,296
- Down proj: [3584, 18944] = 67,895,296

Per-layer total: 5 * 12,845,056 + 3 * 67,895,296 = 64,225,280 + 203,685,888
= **267,911,168 weights per layer**

36 layers: 267,911,168 * 36 = 9,644,802,048 weights (~9.6 billion)

Plus LM head: 248,320 * 3584 = 544,538,624 weights

Plus embeddings: 248,320 * 3584 = 544,538,624 weights (but embeddings are not
projections -- they are lookups, not GEMV)

Total GEMV weights: 9,644,802,048 + 544,538,624 = **~10.19 billion weights**

At Q4: 10.19e9 * 0.5 bytes = **5.09 GB** (matches expected 27B model Q4 size when
accounting for shared Q/K projections and GQA)

Wait -- 27B model should have more parameters. The discrepancy is because Qwen 3.5
27B uses GQA (grouped query attention) where K and V projections are smaller. Also,
the actual hidden dim might be larger. Let me use the standard figure: **27 billion
parameters at Q4 = 13.5 GB of weight data.**

Instruction stream: 27e9 * 2 * 16 = **864 GB.**

This is the corrected figure. A 27B Q4 model produces an 864 GB instruction stream.
This requires 11 H100 80GB GPUs just for instruction storage.

### Where weights-as-code makes sense

Given the 64x expansion, the sweet spot is:

1. **Very small models (<=1B)** where the instruction stream fits comfortably in one
   GPU's memory and the elimination of memory latency produces a measurable speedup.

2. **Single-layer or single-projection embedding** for profiling: embed one
   projection as weights-as-code to measure actual instruction fetch bandwidth vs.
   LDG bandwidth on Hopper. This gives empirical data on whether the approach is
   faster per-byte.

3. **Future architectures** where instruction fetch bandwidth significantly exceeds
   data fetch bandwidth, or where wider immediate fields allow denser packing.

### Alternative: FFMA-register with MOV-IMM pipeline

A more practical variant that preserves the "no LDG for weights" property:

```
MOV-IMM  R10, <w0_fp32>    ; weight 0
MOV-IMM  R11, <w1_fp32>    ; weight 1
MOV-IMM  R12, <w2_fp32>    ; weight 2
...
FFMA  Racc, R10, Ract0, Racc   ; acc += w0 * x0
FFMA  Racc, R11, Ract1, Racc   ; acc += w1 * x1
...
```

This is the same instruction count (2 per weight: MOV-IMM + FFMA) but allows the
MOV-IMM and FFMA to be scheduled on different pipelines and interleaved. The MOV-IMM
uses the integer pipe; FFMA uses the FP32 pipe. Dual-issue potential.

On sm90, MOV-IMM can issue on a different warp scheduler than FFMA, so with 4 warp
schedulers per SM, the throughput could approach 1 weight per cycle instead of 2
cycles per weight.

**This does not change the size calculation** (still 2 instructions * 16 bytes = 32
bytes per weight) but may improve throughput.

---

## 9. Recommended Path Forward

### Phase 1: Proof of concept (immediate)

1. Implement `compiler/safetensors.fs` -- minimal safetensors reader in Forth
2. Implement `compiler/weight-embed.fs` -- Q4 to FP32 immediate converter
3. Embed one small projection (e.g., a [256, 256] test matrix) as weights-as-code
4. Measure: instruction fetch latency, actual throughput, correctness vs. LDG path

### Phase 2: Single-layer prototype

1. Embed one full DeltaNet layer's projections
2. Run the forward pass with weights-as-code for that layer, LDG for others
3. Measure per-layer latency delta

### Phase 3: Architecture decision

Based on Phase 1-2 measurements:
- If instruction fetch throughput >= LDG throughput: pursue full model embedding
  (requires multi-GPU)
- If instruction fetch throughput < LDG throughput: abandon full embedding, but
  retain the compiler infrastructure for other uses (e.g., embedding lookup tables,
  small constant matrices, normalization weights)

### What NOT to do

- Do not attempt to embed the full 27B model on a single GPU. The math does not
  work: 864 GB does not fit in 80 GB or 96 GB.
- Do not assume instruction fetch bandwidth is "free." It competes with data fetch
  for DRAM bandwidth. The net effect depends on Hopper's internal arbitration, which
  is not publicly documented.
- Do not report performance numbers until Phase 1 measurements are complete and
  verified correct end-to-end.
