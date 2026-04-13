# Lithos TODO

Confirmed bugs, missing pieces, and architecture work.
Debunked claims and fixed bugs at the bottom.

---

## COMPILER BUGS — still open

### C3. emit-bw-instr emits NO instructions
`emit-bw-instr` parses operands but returns without emitting anything.
`shr`/`shl`/`and`/`or`/`xor` are all silent no-ops.
Fix: call `shf-r-imm,`/`lop3-and-imm,` etc. based on op name.

### C4. emit-sub-u32, emit-mul-u32 are stubs (`drop 2drop`)
Parse args, emit nothing.  Any kernel using `sub` or `mul` is wrong.
Fix: add `imad,` (mul) and `imad,` with negation (sub) calls.

### C5. For-loop branch never patched, no back-branch emitted
`emit-for-v2` emits `0 bra-pred,` as placeholder; `emit-endfor` never patches it.
No back-branch to loop top is emitted either.  Loops do not loop.
Fix: save `sass-pos @` before loop body, patch placeholder at `emit-endfor`,
emit back-branch with correct negative dword offset.

---

## EMITTER BUGS — still open

### 1. emit-cross-warp-reduce — duplicate, incompatible signatures
Two definitions, different stack effects:
- `emit-reduce.fs:102` — `( partial smem-base tid -- result )`
- `emit-gemv.fs:252`  — `( partial tid -- result )` (no smem-base)

Whichever loads last wins and silently breaks the other caller.
Fix: give the gemv-local version a distinct name (`emit-cross-warp-reduce-gemv`).

### 2. emit-grid-sync shadowed with nop, in ls-compiler.fs:251
`gpu/emit.fs` has the real cooperative grid-sync implementation.
`ls-compiler.fs:251` redefines it as `nop,`. Any kernel using `TOK_REPEAT`
silently emits a NOP barrier. Layer pipelining is broken.
Fix: remove the shadow definition from ls-compiler.fs.

### 3. rreg+/freg+/preg+ defined in three places
`gpu/emit.fs`, `parser.fs`, and `ls-compiler.fs` all define these.
Load order determines which wins. Reset between kernels now works (C6 fixed),
but the triple definition is still messy.
Fix: define once in gpu/emit.fs (real), remove stubs and duplicates.

### 4. ls-compile-file pipeline not reachable under forth-bootstrap
The `.ls → .elf` path exists in ls-compiler.fs but is not wired into
`run-lithos.sh`. No script invokes it. No kernel has been compiled through it.
Fix: wire ls-compile-kernel into a `lithos-compile` entry point.

---

## KERNEL GAPS — blocking single-layer inference

### K3. conv1d_infer is an empty stub (P0 — DeltaNet correctness)
`recur.li:conv1d_infer` has no body.  Blocked on computed-offset array indexing
(`a[i + k]`) not implemented in parser.  Without 4-tap causal convolution,
Q/K/V short convolutions cannot run.
Fix: implement computed-index expressions in parser; fill kernel body.

### K4. sample_argmax is a stub (P0 — can't emit a token)
`reduce.li:sample_argmax` copies logits without computing argmax.
No softmax or top-k anywhere.  Cannot emit a token.
Fix: implement argmax over logits in reduce.li.

### K6. RoPE angle formula wrong — linear approximation (P1)
`attend.li` computes `angle = fpos * freq_exp` where `freq_exp = fi * 2.0 / fd`.
Correct formula: `pos / (10000 ^ (2i/d))` which requires log + exp.
Fix: rewrite rope angle computation in attend.li.

### K7. attend.li GQA mapping uses q_head as kv_head (P1)
For GQA ratio > 1, `kv_head = q_head / gqa_ratio` — not implemented.
Current code uses `head` directly, producing wrong K/V for non-first query groups.
Fix: add integer divide in attend.li.

### K8. Token embedding kernel missing
No `.li` source for token embedding lookup. Legacy `kernels/embed.cubin` exists
but has no source.  Trivial kernel: `output[i] = embed_table[token_id * dim + i]`.

---

## ARCHITECTURE — weights-as-code transition

Design documents written today at `docs/weights-as-code-*.md`.

### A1. Safetensors parser in Forth
The compiler must read model weight files at compile time.
Safetensors format: 8-byte header length + JSON header + raw tensor data.
Need: `compiler/safetensors.fs` — mmap file, parse JSON header for tensor
names/offsets/shapes, return base+offset for any named tensor.

### A2. Template language — .li v2
.li files become templates: `weight W_q : [5120, 5120] q4` declares compile-time
weight tensors. `layer L : 0..47` stamps out per-layer. `project W_q x -> y`
emits the full projection with hardcoded addresses and broadcast scalars as
immediates.  See `docs/weights-as-code-language.md`.

### A3. Model-to-binary compiler
The `project` primitive's emit word changes: reads Q4 weight tensor from
safetensors, hardcodes all LDG addresses, embeds scale/zero-point/bias as
FMUL-IMM/FADD-IMM immediates, emits one straight-line GEMV instruction stream
per projection per layer.  Matrix weights stay LDG (SIMT constraint).
See `docs/weights-as-code-compiler.md`.

### A4. Megakernel linker
Per-layer ELF sections must be linked into two cooperative megakernels
(forward pass + recurrence update).  Grid-sync instructions between layers.
Need: `compiler/megakernel-link.fs`.

### A5. GSP boot from Forth — eliminate C entirely
`docs/gsp-native.md` confirms: BAR0/BAR4 via vfio-pci (stock Linux),
all register writes from Forth or ARM64.  No custom kernel module needed.
Need: `compiler/gsp-boot.fs` — mmap BARs, poke Falcon registers, boot GSP.
Delete `kernel/` directory.

### A6. GPFIFO + doorbell from Forth
After GSP boot: create channel via GSP RPC, bump-allocate GPFIFO ring from
BAR4, write QMD pushbuffer, ring doorbell at USERD+0x90.
Need: `compiler/gpfifo.fs`.

---

## BUILD

- `bin/build.sh` only builds the launcher binary. No script rebuilds kernels.
- No dependency tracking. `.cubin`/`.elf` files in `kernels/` are stale vs `.li` sources.
- Need: `lithos-build` script that runs forth-bootstrap over each `.li` file and
  emits `.elf` into `kernels/`.

---

## GPU MACHINE CODE REFERENCE DOCS

| File | Contents |
|---|---|
| `docs/encoding/arithmetic.md` | FMUL, FADD, FFMA, FSETP, FNEG/FABS — source modifier bit fields |
| `docs/encoding/sfu.md` | MUFU all variants, subop table, exp/log two-instruction sequences |
| `docs/encoding/memory.md` | LDG, STG, LDS, STS, LDC, ULDC — offset encoding, width bit, cbuf layout |
| `docs/encoding/integer_atomic.md` | IMAD, IADD3, ISETP, LOP3, ATOMG — truth tables, ctrl constants |
| `docs/encoding/warp_control.md` | SHFL.BFLY, BAR.SYNC, S2R, BRA |

---

## FIXED TODAY — do not re-open

| Bug | Fix |
|---|---|
| C1. Dangling `drop` parser.fs:438 | Deleted |
| C2. emit-math-unary silent no-op | Added dispatch: neg→FMUL×-1, exp→ex2, rcp→rcp, rsqrt→rsqrt, sqrt→sqrt, sin→sin, cos→cos, log→lg2 |
| C6. rreg+/freg+/preg+ never reset | Added `_gs-r`/`_gs-f`/`_gs-p` reset in parse-fn |
| BRA dword offset | `bra,`/`bra-pred,` now divide by 4 internally |
| BRA back-edge sign | Removed `swap` in emit-reduce.fs:268 |
| BRA patch in emit-gemv/emit-elementwise | Added `4 /` in patch-bra-offset and patch-forward-bra |
| ISETP $7c0c→$720c | Fixed in gpu/emit.fs isetp-ge, and isetp-lt, |
| FMUL-IMM $7420→$7820 | Fixed in emit-elementwise.fs fmul-imm, |
| FADD-IMM hardcoded→OP-FADD-IMM | Fixed in emit-elementwise.fs fadd-imm, |
| IMAD/ISETP/IADD3 duplicate opcodes | Removed from emit.fs, single source in opcodes-sm90.fs |
| K1. GEMV cross-warp reduction | Added full smem+barrier+butterfly reduction in gemv.li |
| K2. Beta missing deltanet_fused | Added beta param, per-head load, multiply into delta |
| K5. sin/cos emission | sin,/cos, exist in emit.fs, emit-math-unary now dispatches to them |
| ldc,/uldc,/atom-add-f32,/redg-f32, | Added to gpu/emit.fs |
| sass/→gpu/ rename | Complete, all references updated |
| OP-I2F $7306→$7245 | Fixed |
| OP-ATOM-ADD $798b→$79a8/$79a3 | Fixed |
| SHFL.BFLY delta encoding | Fixed |
| BAR.SYNC barrier ID field | Fixed |

---

## DEBUNKED — do not act on these

| Claim | Why wrong |
|---|---|
| "Branch offsets 2× too large" | Our emitter emits 16 bytes/instruction (8B instruction + 8B control). 368 = 23×16 is correct. |
| "ls-tokenizer.fs uses 2>r/2r>" | Verified false. File uses only `>r`/`r>` which bootstrap supports. |
| "`[char]` → `char` change is wrong" | `char *` at top level (create statements) is correct. `[char]` is used correctly inside colon definitions at lines 132, 135, 143, 148, 237-240. |
| "OP-IMAD=$7c24 is correct" | Probe shows $7224. The $7c prefix is a different instruction class on sm90. |
| "Weights as literal FMUL-IMM immediates" | SIMT constraint: all 32 threads execute same immediate. 64x expansion (32 bytes/weight vs 0.5). Corrected: addresses/scalars hardcoded, matrix weights stay LDG. |
