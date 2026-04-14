# Weights-as-Code — Path Forward

Companion to the "Weights-as-Code — current state" section in `STATUS.md`. Where STATUS
describes the gap, this doc proposes the constructive path and its physical constraints.
Written 2026-04-14 after a survey of all parser variants, `compiler/emit-gpu.ls`,
`compiler/lithos-safetensors.ls`, `inference/gemv.ls`, and the three retired
`weights-as-code.md` design docs.

## The literal immediate design is not viable

Three independent hard constraints rule out the "every weight becomes a 32-bit float
immediate in an FMUL-IMM instruction" interpretation. All three are already documented
elsewhere in the tree; collected here so we don't accidentally rebuild the impossible.

### 1. No `FFMA-IMM` on SM90

`FFMA` takes its Rs3 accumulator from the control word, not from the instruction word
(`compiler/emit-gpu.ls:257-262`, opcode `0x7223`). FFMA with an immediate operand does
not exist in the ISA. Consequence: every weight becomes **two** instructions —
`FMUL-IMM` (32 bytes: 16-byte inst + 16-byte ctrl) plus a separate `FADD` to roll the
product into the accumulator. That is 32 bytes of `.text` per weight, not the ideal 16.

### 2. 543 GB of `.text` per Qwen 3.5-27B

At 2 instructions × 16 bytes = 32 bytes per weight, with 27B parameters the model
becomes a **543 GB instruction stream** (per the arithmetic in
`docs/compiler/design/weights-as-code.md:391-454` and reconfirmed independently). GH200
has 96 GB HBM3e. A single-GPU literal embedding fits at scale ≲ 1.5B parameters; Qwen
3.5-27B requires multi-GPU or a different design.

### 3. SIMT uniformity

All 32 lanes of a warp execute the same instruction with the same immediate. You cannot
assign a different weight to each lane within one FMUL-IMM issue. To emit weights as
immediates, each warp would have to produce only one output element, pushing per-output
instruction count up by 32× and destroying the tile structure the kernel needs to hide
LDG latency for activations. `docs/inference/weights-as-code.md:212-357` works this
through and concludes the naive model breaks.

## The viable design — compiled megakernel with compile-time dequant

This is the design `docs/inference/weights-as-code.md` (the most recent of the three
doc versions) settles on after discarding the literal-immediate version.

**Weights stay as data.** The packed Q4 `qweight` tensors, FP16 `scales` and packed
`qzeros` live in `.rodata` of the compiled ELF. 13.5 GB total for Qwen 3.5-27B,
unchanged from today's runtime layout.

**Addresses are hardcoded at compile time.** Every LDG instruction that fetches a
weight gets its address resolved at compile time from the safetensors index. No
`qweight_ptr` parameter in the kernel signature, no per-kernel parameter buffer, no
pointer arithmetic that depends on layer index at runtime.

**Dequant folds into the emitted sequence.** The nibble-extract → zero-point-subtract
→ scale-multiply pipeline (`inference/gemv.ls:58-130`) becomes compile-time
pre-computation where possible. The formula is `fp32_w = (nibble - 8.0) × scale` with
symmetric zp=8 and 128-element groups — all three inputs known to the compiler. The
runtime only does the FMA against the activation.

**One megakernel per layer type.** The `compiler/megakernel-link.ls:614-645` loop
already demonstrates the shape: iterate `for layer_idx 0 NUM_LAYERS 1`, call
`emit_layer_deltanet` or `emit_layer_full_attention` per layer. The template syntax
(`template deltanet_layer` + `layer L : 0..47`) from the retired language spec is
surface sugar over exactly this. The stamping mechanism is already working; what's
missing is the parser-visible keyword.

**Throughput claim** (from Agent 12's arithmetic, not yet empirically measured):
~10,240 instructions per output element vs. ~39,040 today (ignoring the dequant-setup
that runtime gemv does per group; the compile-time path emits a direct FMA once the
activation is loaded). HBM bandwidth per token drops because the dequant scaffold
disappears, though weight LDG bandwidth is unchanged.

## Delta from today, in priority order

Items labeled `required` are the critical path for the compiled-megakernel design.
Items labeled `sugar` are only needed for the surface-syntax template form.

| # | Piece | Label | What to do | Build-on |
|---|---|---|---|---|
| 1 | Keyword handlers in `lithos-table.s:2855-2869` | required | Add `parse_weight`, `parse_bind`, `parse_project` alongside existing control-flow parsers. Stop silently dispatching to `.La_ident`. | `parse_body` at `lithos-parser.ls:1844-1887` |
| 2 | Shape bracket `[rows, cols]` | required | Extend the array-subscript path at `lithos-table.s:2960-2980` to accept commas and return a shape descriptor. | Existing TOK_LBRACK/TOK_RBRACK |
| 3 | Compile-time Q4 dequant in `compiler.ls` | required | Implement `(nibble - 8.0) × fp16_scale → fp32` as a composition callable from `parse_project`. Formula matches `inference/gemv.ls:58,62-66`. | `compiler/emit-gpu.ls:emit_dequant_nibble` (runtime variant for reference) |
| 4 | Safetensors quantization metadata | required | Extend `lithos-safetensors.ls` to parse `quantization_config` from the JSON header (group_size, sym, bits, scale dtype, zero-point layout). Add Q4 as a recognized dtype enum. | `st_parse_header` at line 451 |
| 5 | Safetensors >2D shapes + capacity | required | `shape_2`, `shape_3` fields in the tensor index; raise MAX_TENSORS beyond 128. | Index layout at lines 48–79 |
| 6 | Wiring: project → safetensors → emit_fmul_imm | required | `parse_project` reads weight + scale tensors via `st_find_tensor`, iterates nibbles, emits the instruction stream. | All three pieces exist individually |
| 7 | `<-` arrow operator | sugar | Lex as a single token. Reserved today; unused. | — |
| 8 | `{L}` interpolation in bind paths | sugar | Only needed if `bind L` blocks use templated paths. Alternative: require explicit per-layer paths in a different syntactic form. | — |
| 9 | `0..47` range operator | sugar | Replace with `for L 0 48 1` positional syntax. Already works. | `parse_for` in live parser |
| 10 | Template body capture | sugar | Replace with direct `for L ...` in source over pre-written compositions. The stamping is already in `megakernel-link.ls`. | — |

## Honest scope estimate

Items 1–6 (the `required` set) are roughly:

- (1) parser handlers: few hundred lines of assembly across `lithos-table.s`
- (2) shape bracket: small extension to existing array-subscript path
- (3) compile-time dequant: a few dozen lines of Lithos in `compiler.ls`
- (4) safetensors metadata: moderate JSON-parsing extension in `lithos-safetensors.ls`
- (5) safetensors ergonomics: mechanical — adding fields and raising a constant
- (6) wiring: the hard part, because it's where the design decisions about emission
      order, register allocation for the accumulator, and warp-tile sizing get pinned
      down

Plus the open empirical question from `docs/inference/weights-as-code.md`:
**measured throughput delta vs. today's `gemv.ls`** has not been taken. Until a single
projection is compiled both ways and benchmarked on a real tensor, the "~4× fewer
instructions" claim is a calculation, not evidence. First concrete milestone should be:
one Q-projection compiled both ways, both producing numerically-identical output on a
small synthetic input, with wall-clock and issue-rate measured.

## What to preserve no matter what

The older parser variants (`bootstrap/lithos-parser{.s, -orig.s, -regfix.s, -v2.s,
-v3.s, -v4.s, -v5.s}`) carry hints — specifically the `ORRIMM w5, 0x78206800, w16`
byte-level FMUL-IMM encoding sequence at `lithos-parser.s:724`, `-orig.s:720`,
`-regfix.s:720`, and already-live `lithos-table.s:715`. Do not delete. Their parser
architectures also represent real iteration on the minimal-compiler design; each one
has a distinct approach that might be worth cannibalizing later.

## Pointers

- Current state (facts on the ground): `STATUS.md` → "Weights-as-Code — current state"
- Live FMUL-IMM emitter: `compiler/emit-gpu.ls:406-408`
- Runtime GEMV reference: `inference/gemv.ls`
- Safetensors reader: `compiler/lithos-safetensors.ls`
- Per-layer stamping loop: `compiler/megakernel-link.ls:614-645`
- Retired design docs (read for color, not as instructions): `docs/language/weights-as-code.md`, `docs/compiler/design/weights-as-code.md`, `docs/inference/weights-as-code.md`

The inference-side retired doc is the most accurate about what's actually physically
realizable. Read it first if you're about to touch any of items 1–6.
