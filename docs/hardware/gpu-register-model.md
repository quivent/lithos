# GPU Register Model

How Lithos addresses the full Hopper (sm_90a) register architecture.
Every register file, every address space, every scheduling field --
what the language exposes, what the compiler manages, what the programmer controls.

---

## 1. General Registers: R0-R255 (32-bit, per-thread)

**Hardware:** 256 x 32-bit registers per thread. The main workhorse file.
Every ALU, memory, and conversion instruction reads/writes these.

**Language syntax:** `$0` through `$255`. Already implemented.

```
\\ Lithos .ls -- general register access
rmsnorm x w D :
    ↓ $4 → 32 x i          \\ load into R4
    ↓ $5 → 32 w i          \\ load into R5
    ↓ $6 $4 * $5            \\ R6 = R4 * R5
```

**Compiler:** Bump allocator assigns ascending register numbers to named values.
Named values in compositions resolve to `$N` during flattening.
The compiler tracks `max-reg-used` (high-water mark) for the ELF `.nv.info` section.

**Programmer controls:**
- Explicit `$N` when writing low-level binary-adjacent code
- Named values (preferred) when writing compositions -- compiler assigns registers
- The `regcap` directive to set a hard ceiling (see section 9)

---

## 2. Uniform Registers: UR0-UR63 (32-bit, per-warp)

**Hardware:** 64 x 32-bit registers shared across all 32 threads in a warp.
One copy, one value, read by every thread simultaneously. Used for base addresses,
loop counters, constant bank loads, and descriptor pointers. ULDC loads from
constant memory directly into a uniform register. LDG/STG use a UR as the
descriptor pointer field (encoded in the ctrl word, not the instruction word).

**Language syntax:** `$U0` through `$U63`. The `U` prefix distinguishes the file.

```
\\ Lithos .ls -- uniform registers for shared-across-warp values
gemv_setup base_ptr stride :
    ↓ $U4 base_ptr           \\ UR4 = base address (same for all 32 threads)
    ↓ $U5 stride             \\ UR5 = stride (same for all 32 threads)
    ↓ $0  → 32 $U4 $tid_x   \\ LDG R0, desc[UR4][R_tid.64] -- per-thread load
```

**Named uniform values** use a `uniform` qualifier:

```
\\ Lithos .ls -- compiler infers uniform file from qualifier
gemv W x y D :
    uniform base → 64 W      \\ compiler places in UR pair (UR4:UR5)
    uniform cols D            \\ compiler places in UR6
    each i
        ↓ $0 → 32 base i    \\ LDG with UR descriptor
```

**Compiler responsibilities:**
- Track a separate uniform register allocator (UR0-UR63, bump or linear scan)
- When a value is declared `uniform` or provably warp-invariant (e.g., block ID,
  constant, loop bound), allocate from the uniform file
- Emit ULDC (opcode `$7ab9`, ctrl `CTRL-ULDC`) for constant bank loads into URs
- Emit S2UR (opcode `$79c3`) for special-register-to-uniform moves
- Encode the UR index in the ctrl word for LDG/STG descriptor fields

**Programmer controls:**
- Explicit `$U0`-`$U63` for hand-scheduled code
- `uniform` keyword for named values -- compiler picks the UR slot
- Cannot store a per-thread-divergent value in a UR (compiler error)

---

## 3. Predicate Registers: P0-P6 (1-bit, per-thread)

**Hardware:** 7 usable predicate registers per thread (P0-P6). P7 = PT (always true),
not assignable. ISETP/FSETP write a predicate. BRA, conditional execution, and
predicated instructions read a predicate.

**Language syntax:** `@P0` through `@P6`. The `@` prefix matches NVIDIA's
predicate convention in SASS (`@P0 BRA target`).

```
\\ Lithos .ls -- explicit predicate usage
bounded_add a b limit result :
    ↓ $0 → 32 a i
    ↓ $1 → 32 b i
    ↓ $2 $0 + $1
    @P0 if>= $2 limit        \\ ISETP.GE writes P0
        ↓ $2 limit           \\ predicated: only executes where P0 is true
    ← 32 result i $2
```

**Implicit predicate usage (preferred):** The `if==`, `if>=`, `if<` control flow
words already allocate predicates internally. The programmer never names them:

```
\\ Lithos .ls -- implicit predicates (common case)
clamp x lo hi :
    if< x lo
        x lo                  \\ compiler allocates P0 for the comparison
    if>= x hi
        x hi                  \\ compiler allocates P1 (or reuses P0)
```

**Compiler responsibilities:**
- Allocate P0-P6 using a small free-list (7 registers, trivial allocation)
- `if==`, `if>=`, `if<` emit ISETP/FSETP writing to an allocated predicate,
  then BRA reading that predicate
- Free predicates when they fall out of scope (after the branch resolves)
- Error if more than 7 predicates are live simultaneously

**Programmer controls:**
- Explicit `@P0`-`@P6` for hand-written predicate logic
- Implicit predicates via `if==`/`if>=`/`if<` (no naming needed) -- the common path
- `@PT` as a read-only alias for "always true" (P7)

---

## 4. Uniform Predicates: UP0-UP6 (1-bit, per-warp)

**Hardware:** 7 uniform predicate registers shared across all 32 threads. If a
comparison has the same result for every thread (e.g., comparing two uniform values),
the result goes in a UP instead of a P. This enables uniform branch decisions
without divergence.

**Language syntax:** `@UP0` through `@UP6`.

```
\\ Lithos .ls -- uniform predicate for warp-uniform branch
loop_check iter max_iter :
    uniform i iter
    uniform bound max_iter
    @UP0 if< i bound          \\ uniform comparison -> UP0
        \\ all 32 threads take the same branch -- no divergence
```

**Compiler responsibilities:**
- When both operands of a comparison are uniform (in UR file), emit a uniform
  SETP variant and allocate from UP0-UP6
- Uniform predicate allocation is a separate 7-entry free-list
- Uniform branch (BRA with UP) avoids warp divergence overhead

**Programmer controls:**
- Explicit `@UP0`-`@UP6` for hand-written uniform control flow
- Normally invisible -- the compiler promotes P to UP when operands are uniform

---

## 5. 64-bit Values: Register Pairs

**Hardware:** A 64-bit value occupies two consecutive even/odd registers: R0:R1,
R2:R3, etc. GPU pointers are 64-bit. LDG/STG use 64-bit address registers.
IMAD.WIDE produces a 64-bit result in a register pair.

**Language syntax:** `$0:1` denotes the pair R0:R1. The compiler enforces
even-alignment on the first register.

```
\\ Lithos .ls -- 64-bit address computation
load_global base offset :
    ↓ $2:3 base               \\ 64-bit base address -> R2:R3
    ↓ $0 → 32 $2:3 offset    \\ LDG R0, [R2.64 + offset]
```

**For named values:** The compiler infers 64-bit width from context (pointer
operations, `→ 64` loads, address arithmetic) and pairs registers automatically:

```
\\ Lithos .ls -- compiler auto-pairs for 64-bit
gemv_load W_ptr row_offset :
    ptr → 64 W_ptr             \\ compiler allocates R2:R3, loads 64-bit pointer
    val → 32 ptr row_offset   \\ LDG uses the 64-bit pair as address
```

**Compiler responsibilities:**
- When allocating a 64-bit value, bump to the next even register index
- Track pair liveness as a unit (both registers occupied)
- Emit `.64` memory width bit in ctrl word (bit 9 of extra41 field)
- ULDC.64 for 64-bit constant bank loads into UR pairs (UR4:UR5, etc.)

**Programmer controls:**
- Explicit `$N:N+1` notation when hand-scheduling
- Named values with 64-bit context -- compiler handles pairing
- No half-pair access: you cannot write `$2` independently if `$2:3` is live

---

## 6. FP16 Packing: Two Values Per Register (Future)

**Hardware:** HFMA2 (opcode `$7235`) operates on two FP16 values packed into one
32-bit register. The upper 16 bits hold one value, the lower 16 bits hold another.
This doubles throughput for half-precision math.

**Status:** Deferred. Current Lithos targets FP32 and INT4-GPTQ inference.
FP16 packing will be addressed when FP16 inference kernels are prioritized.

**Sketch of future syntax:**

```
\\ Lithos .ls -- FP16 packed operations (FUTURE, not yet implemented)
hfma2 a b c :
    ↓ $0.h2 pack a_lo a_hi    \\ pack two FP16 into one register
    ↓ $1.h2 pack b_lo b_hi
    ↓ $2.h2 $0.h2 * $1.h2 + $3.h2   \\ HFMA2: two FMAs in parallel
```

**Design notes for future work:**
- `.h2` suffix on a register denotes FP16x2 packed format
- `pack` / `unpack` primitives to convert between scalar FP16 and packed
- Compiler must track register format (f32 vs h2) to select HFMA2 vs FFMA
- Register count is the same (still 32-bit registers), but arithmetic throughput doubles

---

## 7. Shared Memory

**Hardware:** On-SM scratchpad (up to 228KB on Hopper, configurable vs L1).
Addressed separately from global memory. Uses LDS/STS instructions (opcodes
`$7984`/`$7988` for UR-addressed, `$7388` for register-addressed STS).
Visible to all threads in a thread block. Primary use: inter-warp communication,
tiling, and reduction scratch space.

**Declaration syntax:**

```
\\ Lithos .ls -- shared memory declaration
gemv_kernel D :
    shared scratch 4096 f32    \\ 4096 x 4 bytes = 16KB shared buffer
    shared partial 128 f32     \\ 128 x 4 bytes = 512B partial sums
```

The `shared` keyword declares a named region. The compiler assigns byte offsets
from the start of the shared memory segment.

**Access syntax:** Same memory arrows, but with `@smem` address space qualifier:

```
\\ Lithos .ls -- shared memory load/store
reduce_block val :
    shared scratch 128 f32
    ↓ $0 val
    ← 32 @smem scratch $tid_x $0    \\ STS [scratch + tid.x * 4], R0
    bar 0                             \\ BAR.SYNC -- wait for all threads
    ↓ $1 → 32 @smem scratch 0       \\ LDS R1, [scratch + 0]
```

**How the compiler distinguishes address spaces:**

| Syntax | Instruction | Opcode |
|---|---|---|
| `→ 32 addr` | LDG (global) | `$7981` |
| `→ 32 @smem addr` | LDS (shared) | `$7984` |
| `← 32 addr val` | STG (global) | `$7986` |
| `← 32 @smem addr val` | STS (shared) | `$7988` |

The `@smem` qualifier is required. There is no ambiguity -- the compiler selects
the correct instruction based on the qualifier. Omitting `@smem` means global.

**UR-addressed vs register-addressed shared memory:**
- `STS [UR4+offset], Rs` (opcode `$7988`) -- when offset comes from a uniform register
- `STS [Rn+offset], Rs` (opcode `$7388`) -- when offset comes from a general register
The compiler selects the correct opcode based on whether the address register
is in the UR file or the R file.

**Compiler responsibilities:**
- Track shared memory usage and assign byte offsets to named regions
- Emit `.nv.shared.<kernel>` section in ELF with total shared memory size
- Emit correct opcode (LDS/STS vs LDG/STG) based on `@smem` qualifier
- Insert `BAR.SYNC` at `bar N` sites

**Programmer controls:**
- `shared name size type` declarations with explicit sizes
- `@smem` qualifier on loads/stores
- `bar N` for explicit synchronization barriers

---

## 8. Warp Scheduling / Control Words

**Hardware:** Every 128-bit Hopper instruction is 64-bit instruction word + 64-bit
control word. The control word fields:

| Bits | Field | Range |
|---|---|---|
| 44:41 | Stall count | 0-15 cycles |
| 45 | Yield hint | 0 or 1 |
| 48:46 | Write barrier | 0-6 (7=none) |
| 51:49 | Read barrier | 0-6 (7=none) |
| 57:52 | Wait barrier mask | 6-bit mask |
| 62:58 | Reuse cache flags | 5-bit mask |
| 40:0 | Extra41 | Opaque per-opcode |

**Language policy: compiler-managed, not programmer-exposed.**

The control word is scheduling microarchitecture. Exposing it in the language would
couple the source to a specific SM generation. The Forth emitter (`gpu/emit.fs`)
already handles this correctly: each instruction builder (`fadd,`, `ffma,`, `ldg,`, etc.)
calls the corresponding `ctrl-*` word with empirically verified scheduling parameters.

**What the compiler does:**
- Each instruction carries its own ctrl word constructor (from probe data)
- Memory operations set write barriers and stall counts based on latency
- The emitter tracks barrier slot allocation (7 slots, round-robin)
- Stall counts are per-instruction-class constants from nvdisasm probes

**What is NOT exposed:**
- Stall counts (always from verified probe data)
- Yield hints (always from verified probe data)
- Barrier slot assignment (compiler manages the 7-slot pool)
- Reuse cache flags (from probe data)
- Extra41 opaque bits (from probe data, parameterized where needed)

**Exception -- the `stall` escape hatch:**

For expert tuning of instruction-level scheduling, a future `pragma` could
override the default stall count:

```
\\ Lithos .ls -- hypothetical scheduling override (NOT YET IMPLEMENTED)
\\ This is an escape hatch, not the normal path
tight_loop :
    pragma stall 2             \\ override next instruction's stall to 2 cycles
    ↓ $0 $1 + $2
```

This is explicitly deferred. The current approach (hardcoded from probe data)
is correct for all verified instruction sequences. Scheduling overrides would
only matter for novel instruction sequences not yet probed.

---

## 9. Register Allocation Strategy

**Hardware constraint:** Register usage determines occupancy:

| Registers/thread | Warps/SM | Occupancy |
|---|---|---|
| 32 | 64 | 100% |
| 64 | 32 | 50% |
| 128 | 16 | 25% |
| 255 | 8 | 12.5% |

For Lithos megakernels (cooperative grid-sync, long-running), occupancy is
typically low by design -- these kernels own the GPU. The tradeoff is explicit.

**Current allocator: monotone bump.**
The Forth emitter increments `max-reg-used` for every destination register.
No register is ever reused. This is correct but wasteful -- a kernel with
sequential independent operations burns through registers unnecessarily.

**Language controls:**

The `regcap` directive sets a hard ceiling on register usage per kernel:

```
\\ Lithos .ls -- register budget control
gemv_kernel D :
    regcap 64                  \\ max 64 registers -> 32 warps/SM occupancy
    shared scratch 4096 f32
    each i
        val → 32 W i
        acc $acc + val * x_i
    ← 32 y $tid_x acc
```

**Compiler allocation roadmap:**

1. **Current: monotone bump** -- registers are never freed. Simple, correct,
   high register pressure. Suitable for small kernels (< 32 registers).

2. **Next: linear scan** -- track live ranges of named values. When a value's
   last use is reached, return its register to the free pool. This is the
   standard approach for JIT compilers (LLVM, LuaJIT).

3. **Future: graph coloring** -- build an interference graph of simultaneously
   live values, color with `regcap` colors. Optimal but expensive to compute.
   Only needed if linear scan proves insufficient.

**How linear scan works in Lithos:**

```
\\ Lithos .ls -- linear scan example
\\ The compiler sees that 'a' is dead after line 3, so R0 is reused for 'd'
example x y z :
    a → 32 x 0               \\ R0 = load x[0]        (a live: R0)
    b → 32 y 0               \\ R1 = load y[0]        (a,b live: R0,R1)
    c a + b                   \\ R2 = R0 + R1          (a dead after this)
    d → 32 z 0               \\ R0 = load z[0]        (R0 reused! a is dead)
    e c + d                   \\ R1 = R2 + R0          (b dead, R1 reused)
    ← 32 out 0 e
```

With linear scan, this kernel uses 3 registers instead of 5. For megakernels
with hundreds of operations, the savings determine whether the kernel fits
in the register budget.

**Register spilling:**

If live values exceed `regcap` (or 255), the compiler must spill to shared
memory or local memory. Lithos does not yet implement spilling. If the
allocator cannot fit within the budget, compilation fails with an error
showing the live range conflict. The programmer must restructure the kernel.

---

## Summary: Register File Syntax Table

| File | Hardware | Lithos Syntax | Count | Scope |
|---|---|---|---|---|
| General | R0-R255 | `$0`-`$255` | 256 | per-thread |
| Uniform | UR0-UR63 | `$U0`-`$U63` | 64 | per-warp |
| Predicate | P0-P6 | `@P0`-`@P6` | 7 | per-thread |
| Uniform pred | UP0-UP6 | `@UP0`-`@UP6` | 7 | per-warp |
| Register pair | R0:R1 etc. | `$0:1` etc. | 128 pairs | per-thread |
| Shared mem | -- | `@smem` qualifier | configurable | per-block |
| Always-true | PT | `@PT` | 1 (read-only) | constant |

## Naming Conventions

- `$` prefix = register (value storage)
- `@` prefix = predicate or address space qualifier
- `U` after prefix = uniform (warp-shared)
- `:` between numbers = register pair (64-bit)
- Lowercase named values = compiler-allocated (the common path)
- Explicit numbered registers = hand-scheduled (the escape hatch)

## What the Compiler Always Manages

- Control word scheduling (stall, yield, barriers, reuse)
- Barrier slot allocation (7 slots per warp)
- Instruction selection (LDG vs LDS, ISETP vs uniform SETP)
- Register pair alignment (even-index enforcement)
- `max-reg-used` tracking for ELF metadata

## What the Programmer Always Controls

- `regcap N` -- register budget ceiling
- `shared name size type` -- shared memory allocation
- `bar N` -- explicit synchronization points
- `uniform` qualifier -- force uniform register allocation
- Choice between named values (compiler-allocated) and explicit `$N` (hand-scheduled)
