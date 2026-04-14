# Register allocator design

## Status

Design (this document) -> Implementation (pending sign-off). K, 2026-04-13.

## Why

Today's allocator: four monotone counters in `parser.fs:78-81` (`freg+`,
`rreg+`, `rdreg+`, `preg+`). IDs are minted and never reclaimed. Fine for
PTX (ptxas re-allocates) but on SASS the IDs are physical registers --
`gpu/emit.fs` encodes them straight into instruction bytes. Hopper
caps threads at 255 regs and occupancy collapses above ~128. Fully inlined
DeltaNet layers will blow this on the first compile. We need reuse.

## Algorithm: linear scan

The source language is uniquely friendly to linear scan:

- **No arbitrary CFG.** `.ls` is straight-line + counted `for`/`stride` loops
  (parser.fs:723, 858) + predication + `if>= ... exit`. No `while`, no `goto`
  back-edges other than loop back-edges, no indirect branches.
- **No runtime calls.** `fn` inlines (LANGUAGE.md). One flat body per kernel.
  No caller/callee-saved conventions.
- **Single-pass today.** The parser walks tokens once and emits at the
  `li-backend @ if` dispatch (parser.fs:686). Linear scan matches this shape.
- **Small programs.** Largest `.ls` is `inference/reduce.ls` at 216 lines.
  Fully-inlined DeltaNet is under ~2000 IR instructions. Compile time is
  not a concern.

Graph coloring buys nothing: it wins only when interference structure defeats
linear scan, typically with many disjoint ranges in complex loops. Our loops
are shallow (max nesting 2) with few carried values. We would pay graph build,
simplify and color cost for ~zero register savings.

**Decision: linear scan with live ranges conservatively extended across whole
loop bodies.**

## Compilation model: two-pass (SASS path only)

Options: (1) stay single-pass with deferred binding -- works for expressions
but breaks for values crossing loops, fatally broken for our kernels;
(2) full two-pass with IR buffer + late emission; (3) hybrid: PTX stays
single-pass, SASS goes two-pass.

**Decision: option 3.** PTX already works -- ptxas is a real allocator, don't
re-implement it. SASS is where virtual-to-physical is our job. The
`li-backend @ if ... else ... then` at parser.fs:686 becomes the split: the
`else` branch emits PTX text as today; the `if` branch appends an IR record
instead of calling a SASS encoder. After each `fn`, a terminal `ra-finalize
( -- )` computes live ranges, assigns physical registers, then walks the IR
calling the existing SASS encoders (`fadd,`, `ffma,`, `imad-imm,` ...).

## IR

Flat array of fixed-size instruction records. One record per SASS instruction.
Stored in a single contiguous buffer `ir-buf` with `ir-count` entries.

Each record is 8 cells (64 bytes on 64-bit Forth):

```
cell 0: op         \ opcode id: OP_FADD, OP_FFMA, OP_IMAD_IMM, OP_LDG, ...
cell 1: vdst       \ virtual dest register id, or -1
cell 2: vsrc0      \ virtual src, or -1, or immediate tag
cell 3: vsrc1
cell 4: vsrc2
cell 5: pred       \ virtual predicate (-1 = unconditional)
cell 6: imm        \ immediate value when present
cell 7: flags      \ bits: [0]=dst-is-f32, [1]=dst-is-pred, [2]=is-loop-head,
                   \       [3]=is-loop-tail, [4]=is-barrier, [5]=is-gridsync,
                   \       [6..7]=reg-class (F32/I32/I64/PRED)
```

A virtual register is a small integer from a unified namespace (`next-vreg`)
replacing the four monotone counters. Example from `delta_update.ls`:

```
kts = 0.0
for row 0 d 1
    mad sidx row d j
    ki = k [ row ]
    si = state [ sidx ]
    fma kts ki si kts
endfor
```

becomes IR (F=float, I=int):

```
0: MOV_IMM    v0:F  imm=0x00000000
1: IMAD_IMM   v1:I  v_row, v_d, v_j     ; loop head (flag 2)
2: LDG_F32    v2:F  base=v_k_ptr off=v_row
3: LDG_F32    v3:F  base=v_state_ptr off=v1
4: FFMA       v0:F  v2, v3, v0          ; accumulate into kts
5: IADD_IMM   v_row:I v_row, 1          ; loop tail (flag 3)
6: BRA_BACK   target=1
```

After allocation, `kts` keeps a single physical slot across the loop because
its live range spans the loop boundary; `v2` and `v3` share a slot with next
iteration's `ki`/`si`.

## Register classes

Hopper R0..R254 is a flat pool of 32-bit slots. 64-bit values need two
adjacent even-aligned slots (Rn:Rn+1, n even). Predicates P0..P6 are
separate hardware; P7 is always-true and not allocated.

**Decision: one free-list for 32-bit slots, one for predicates; 64-bit
is an aligned pair drawn from the 32-bit pool.**

- `free32`: 32-byte bitset over R0..R254. Reserve R0..R3 for tid/ctaid as
  parser.fs:74-75 already does.
- Classes F32 and I32 both pull from `free32`; no physical distinction.
- Class I64 pulls an aligned pair by scanning `free32` for two consecutive
  free slots at even n; spills if none (see below).
- Class PRED pulls from `freeP` (7-bit bitset P0..P6).

Key words:

```
: vreg-new    ( class -- v )           \ fresh virtual reg, records class
: preg-alloc  ( class -- phys )        \ returns R-number or P-number, or -1
: preg-free   ( phys class -- )        \ returns slot(s) to pool
```

## Liveness

Computed at `ra-finalize` in one backwards pass:

- Walk IR from last instruction to first. Maintain `live-set`, a bitset of
  currently-live vregs.
- Per instruction: record `last-use` for every source the first time seen
  backwards; add sources to `live-set`; record `def-pt` for the destination;
  remove destination from `live-set`.

**Loop extension rule (conservative).** For any vreg whose `[def, last-use]`
range crosses a loop boundary, extend `def-pt` up to the loop head and
`last-use` down to the loop tail. One additional pass over the IR, no
fixpoint iteration. Never wrong; occasionally over-reserves by one slot on
values only used on one side of the loop.

Storage:

```
create vreg-def-pt  MAX-VREGS cells allot
create vreg-last-pt MAX-VREGS cells allot
create vreg-class   MAX-VREGS cells allot
create vreg-phys    MAX-VREGS cells allot
```

`MAX-VREGS` = 1024 initially; IR builder aborts with a clear message on
overflow. Realistic workloads need a few hundred.

## Allocation pass

Linear scan in IR order:

```
active = empty         \ (vreg, last-pt) sorted by last-pt
for i = 0 .. ir-count-1:
    expire-old(i)      \ for each v with last-pt < i: preg-free(v)
    for each vdst defined at i:
        p = preg-alloc(class)
        if p = NONE: spill-one(active, class); retry
        vreg-phys[vdst] = p
        insert into active ordered by last-pt
```

Reuse comes from `expire-old`: when last-use is past, the slot returns to
the pool and the next def takes it.

## Spilling

**Pressure estimate.** `attend.ls` attention_score peaks at ~35 live floats
(q_val, running_max, running_sum, acc, scale + online-softmax state +
short-lived shuffle temps). `delta_update.ls` Phase 3: ~12. A fully-inlined
DeltaNet layer (rope + attend + delta + norm + ffn) lands around 80-110 live
32-bit values. Headroom under 128 with reuse alone.

**Decision: spec it, don't ship it in v1.** If `preg-alloc` returns -1, abort
with `FATAL: register pressure exceeded at v=N class=F32`. Fix by splitting
the kernel. Spill is a follow-up.

**Spill design when needed:**

- Target: shared memory, not local. Hopper shared is ~1 cycle with known
  offset; local is 200+ cycles. Declare `__spill` shared buffer sized at
  `spills * 4` bytes; 32 spills = 128 bytes, negligible of 128 KB/SM.
- Policy: **reload-per-use.** Pick the active vreg with the furthest
  `last-pt`, emit `STS.32 [__spill+slot], R`, free the slot. Each later use
  emits `LDS.32 R_tmp, [__spill+slot]` into a fresh allocation.
- Predicates: no spill. 7 is always enough; real kernels use 2-3.

## Loop handling

For a loop spanning IR indices `[H, T]`:

- **Pre-loop def, in-loop use only.** Live range `[def, T]`. Slot occupied
  through the loop, freed at first `expire-old` past T. Example: `d` in
  `delta_update.ls` Phase 1 loop.
- **Pre-loop def, post-loop use.** `[def, last-use]` with `last-use > T`.
  Example: `kts` in Phase 1.
- **In-loop def, in-loop use (same iter).** Both endpoints inside `[H, T]`.
  Pure linear-scan reuse: `ki`, `si`, `sidx` share a slot across iterations.
- **In-loop def, post-loop use.** Loop-carried output. Extension moves
  `def-pt` up to `H` so the slot is reserved for the whole loop.
- **Loop counter.** (`for-counter-reg` at parser.fs:727.) Allocated at H,
  range `[H, T+1]`. IR builder emits its def at H.

**Nested loops.** `for-depth` (parser.fs:112) is already tracked in the
parser. Each loop open/close emits head/tail markers; the extension pass
extends a vreg's range across every enclosing loop its raw range crosses.

## Predication

A predicated instruction `@p INSTR` writes its dest only when `p` is true,
else the old value persists. Two consequences:

1. **Dest is also a use.** The old value is live through this instruction.
   IR builder adds vdst to the source set for predicated writes.
2. **Predicate is a source.** Handled by the `pred` field in the IR record.

For plain FFMA, vdst is def-only. For `@p FFMA`, vdst is both use and def,
which pins its slot correctly. Conditionally-updated accumulators like
`running_max` in `attend.ls` (line 129 onward) just fall out as loop-carried
values; no special case needed.

`setp` defines a predicate vreg. `preg-alloc PRED` returns a P-number.
Predicates live in their own pool.

## Grid-sync / cooperative megakernel

GAPS.md 1e describes a future `repeat N { body }` that inlines N copies of
the layer body separated by `grid.sync()`. Grid sync is a full barrier +
memory fence; by design no register value is carried across it -- each
iteration re-materializes from params/global.

**Rule: a `GRID_SYNC` IR opcode terminates all live ranges.** At the sync
point the allocator runs `expire-all`, clears `active`, and returns every
slot to the free pool. Next instruction starts fresh. This is the **only**
opcode that fully resets the pool. Plain `barrier` (parser.fs:1361,
`bar.sync`) is CTA-local; registers stay live.

## Integration with parser.fs

Scope: **SASS only**. PTX path is untouched.

The `li-backend @ if ... else ... then` dispatch becomes the split:
- `else` (PTX): existing `ptx+` writers; unchanged.
- `if` (SASS): replace direct encoder calls (`fadd,`, `ffma,`, `imad-imm,`,
  `isetp-ge,`, `bra-pred,`) with `ir-emit` appending a record.

Touchpoints in `parser.fs` (grep `li-backend @ if`):

1. `parse-expr` ~686 -- `fadd,` -> `OP_FADD ir-emit`.
2. `emit-fma` ~1184 -- `ffma,` -> `OP_FFMA ir-emit`.
3. `emit-for-v2` ~773-793 -- emit `OP_MOV_IMM`, `OP_IMAD_IMM`, `OP_ISETP_GE`,
   `OP_LOOP_HEAD`, `OP_BRA_BACK`.
4. `emit-endfor` ~825 -- emit `OP_LOOP_TAIL`.
5. `emit-ifge-exit` ~1317, `emit-iflt-exit` ~1331.
6. Other SASS encoder sites (`mul-r,`, `mad,`, `ld-global,`, `st-global,`,
   `barrier,`, `shfl-bfly,`). About **15 distinct sites** total across
   parser.fs (there are 47 counter-bump call sites but far fewer SASS-encoder
   sites).

The four counters (`freg+`, `rreg+`, `rdreg+`, `preg+` at parser.fs:78-81)
become dual-mode:

```
: freg+  ( -- id )  li-backend @ if CLASS-F32 vreg-new
                    else next-freg @ 1 next-freg +! then ;
```

PTX mode: mints physical IDs as today. SASS mode: mints virtual IDs, which
are resolved in `ra-finalize`. 47 existing call sites are unchanged.

Driver: `lithos.fs:104` calls `lithos-compile`. We add `ra-finalize` before
`write-sass-raw`/`write-cubin` (lines 110-118), guarded on `arg-emit @ 1 =
or arg-emit @ 3 =`.

## Debugging / observability

Three hooks, controlled by a `--dump-ra` CLI flag in `lithos.fs`:

1. **v-to-p map.** `<output>.ra.txt`, one line per vreg:
   `v0042 class=F32 def=17 last=89 phys=R23 spilled=no`.
2. **Live-range grid.** ASCII grid, rows = IR index, columns = physical reg,
   `#` where live. Good for spotting pressure cliffs. `<output>.ra.lsve`.
3. **Summary.** Max live 32-bit, max live predicates, spill count, IR count,
   printed to stdout.

Test hook: `tests/test-regalloc.fs` feeds synthetic IR directly into the
allocator. Cases: pre-loop-def / post-loop-use keeps one slot; disjoint
ranges share a slot; in-loop temps reuse across iterations; predicate pool
separate from R-file; grid-sync resets `active`.

## Implementation plan

1. IR buffer + record layout in parser.fs: `ir-buf`, `ir-count`,
   `ir-record-size`, `ir-emit ( op vdst vs0 vs1 vs2 pred imm flags -- )`,
   `ir-reset`. ~60 lines.
2. Vreg namespace: `next-vreg`, `vreg-class[]`, `vreg-def-pt[]`,
   `vreg-last-pt[]`, `vreg-phys[]`. Keep `freg+/rreg+/rdreg+/preg+` as
   dual-mode shims on `li-backend @`. ~40 lines.
3. Convert each `li-backend @ if` SASS branch to `ir-emit`. ~15 sites,
   ~2 lines each.
4. New `compiler/regalloc.fs`: `compute-liveness`, `extend-across-loops`,
   `expire-old`, `preg-alloc`, `preg-free`, `allocate-all`, `ra-finalize`.
   ~300 lines.
5. New `ir-to-sass`: walks IR post-allocation, calls existing encoders
   with physical regs. ~150 lines, one branch per opcode.
6. Driver: `ra-finalize` before `write-sass-raw`/`write-cubin` in
   `lithos.fs`. ~5 lines.
7. Debug dumps: `ra-dump-map`, `ra-dump-live`, `ra-dump-summary`. ~80 lines.
8. Tests: `tests/test-regalloc.fs`. ~200 lines.
9. Update `docs/compiler.html` with a short pointer to this doc.

Total: ~850 new lines of Forth, ~30 lines edited in parser.fs. No `.ls`
syntax changes.

## Open questions

- **Precise live-range extension inside nested loops with conditional arms.**
  Conservative extension is correct but may over-reserve. Revisit with a
  proper dataflow fixpoint if real pressure demands it.
- **Copy coalescing.** `mov a b` where `b` dies at the copy can assign the
  same phys reg. Not in v1.
- **Immediate rematerialization.** Cheaper than spilling a MOV_IMM vreg.
  Add when spill triggers.
- **R0..R3 reservation check.** Validate via a compile of `delta_update.ls`
  that nothing accidentally consumes R0..R3.
- **ARM64 backend unaffected.** Own register model; out of scope.
