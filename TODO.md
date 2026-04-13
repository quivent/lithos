# Lithos TODO

Language spec is at docs/language-primitives.md.
Self-hosting compiler architecture at docs/self-hosting-compiler.md.

---

## SELF-HOSTING COMPILER — the path to first token

### S1. Update .li compiler files to new grammar
The lexer, ARM64 emitter, safetensors reader, and ELF writer all use the
old syntax (fn, ->, etc.). Must be updated to the new symbol set:
→ ← ↑ ↓ $ Σ △ ▽ # ≅ ≡ √ and composition syntax (name args :).

Files: compiler/lithos-lexer.li, compiler/emit-arm64.li,
compiler/lithos-safetensors.li, compiler/lithos-elf.li.

### S2. Write the parser in .li — lithos-parser.li
The missing piece. Recursive descent, single-pass, direct emission.
Reads token stream from lexer, emits ARM64 or GPU machine code
depending on target. Dual backend dispatch.

### S3. Write the sm90 emitter in .li
Port gpu/emit.fs (the Forth GPU emitter) to .li. Each Forth word
(fadd,, fmul,, ldg,, sinst,) becomes a host composition that writes
16 bytes (instruction + control word) to a code buffer.
Opcode constants from gpu/opcodes-sm90.fs become data.

### S4. Bootstrap compile
The Forth bootstrap compiles compiler.li → lithos-stage1 (ARM64 binary).
lithos-stage1 self-compiles → lithos. Diff validates fixed point.
Then Forth is deleted.

### S5. Megakernel linker
Per-layer instruction streams linked into two cooperative megakernels.
Grid-sync instructions between layers.

### S6. GSP boot from .li
Port lithos_gsp.c register sequences to .li compositions using ← and →.
mmap BAR0/BAR4 via vfio-pci. Delete kernel/ directory.

### S7. GPFIFO + doorbell from .li
Channel creation via GSP RPC, QMD pushbuffer construction,
ring write, doorbell store. Zero syscalls at inference time.

---

## EMITTER CLEANUP — still open in Forth (frozen, will be deleted)

These are in the Forth files. We are NOT fixing them. The self-hosting
compiler replaces all of this. Listed for reference only.

- Duplicate emit-cross-warp-reduce signatures
- emit-grid-sync shadowed with nop in ls-compiler.fs
- Triple rreg+/freg+/preg+ definitions
- .ls pipeline not wired into entry point

---

## ARCHITECTURE REGISTER DICTIONARIES

### arch/hopper.dict
Per-thread: $0-$255 (R0-R255, 32-bit general purpose)
Per-warp: $U0-$U63 (UR0-UR63, 32-bit uniform, shared across 32 threads)
Predicates: P0-P6 (1-bit, per-thread), UP0-UP6 (1-bit, per-warp)
Special: TID_X, TID_Y, CTAID_X, LANEID, WARPID, etc.

64-bit: two consecutive R registers (R0:R1). FP16: two values packed per R.
Tensor cores read/write the same R file in matrix fragments.

### arch/arm64.dict
$0-$30 (X0-X30, 64-bit general purpose)
Special: CNTVCT_EL0, MPIDR_EL1, etc.

Design session needed: how to address uniform registers and predicates
in the language. Current spec covers $N for general file only.

---

## OPCODES — COMPLETE

gpu/opcodes-sm90.fs has all probe-verified encodings.
35+ opcodes, all from live ptxas + nvdisasm on GH200.
Single source of truth.

---

## .ls FILES — PRECURSORS

kernels.ls, primitives.ls, derivatives.ls use stack notation and
symbols (×, √, Σ) that are closer to the new language spec than .li.
These may inform the final language design. Not dead — reference material.

---

## FIXED — do not re-open

All compiler bugs (C1-C6), all kernel gaps (K1-K8), all opcode fixes,
BRA dword offset, GEMV cross-warp reduction, beta in deltanet_fused,
conv1d, sample_argmax, RoPE, GQA, token embedding. See git log.
