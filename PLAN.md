# Lithos Execution Plan

Target: **Qwen 3.5 27B** (Huihui-abliterated, GPTQ W4A16) on GH200.
64 hybrid layers: 3 DeltaNet + 1 full attention × 16.

All sources use `.ls` extension. Zero Forth. Zero Python. Zero CUDA runtime.

---

## TOP-LEVEL PROGRESS

```
FOUNDATIONS   [████████] complete   language, kernels, grammar, dicts, docs
COMPILER      [██████░░] wired, not compiling   compiler.ls self-parse blocked
BOOTSTRAP     [███░░░░░] smoke test works, parser gap blocks compiler.ls
RUNTIME       [████░░░░] all .ls written, untested, cbuf0 register_count unknown
HARDWARE      [████░░░░] GSP boot wired (FSP done), QMD builder untested, cbuf0 probe pending
INTEGRATION   [██░░░░░░] launcher.ls written (162 lines), not compiled, nothing executes end-to-end
FIRST TOKEN   [░░░░░░░░] blocked on all above
```

---

## BOOTSTRAP (pure ARM64 assembly, `bootstrap/*.s`)

```
[✓] Lithos bootstrap binary builds, 35KB .bss (not 528MB)
[✓] Entry, memory, DTC runtime wired (_start in driver.s, not lithos-bootstrap.s)
[✓] 7 stack-alignment SIGBUS bugs in lithos-elf-writer.s fixed
[✓] Smoke test: `var x 42` → valid ARM64 ELF produced
[✓] Build: bootstrap/build.sh, Makefile, macros prelude working
[ ] Parser: expression parsing (IDENT = expr, IDENT[i] = expr)  ← BLOCKING
[ ] Parser: `fn NAME ARGS ->` OR new composition syntax               ← BLOCKING
[ ] Parser: hundreds of lines of expression precedence + ARM64 emit   ← BLOCKING
[ ] Audit: compiler.ls parses cleanly through bootstrap               ← BLOCKING
```

Without bootstrap parser work, `compiler.ls` cannot be compiled to `lithos-stage1`.
That blocks every downstream wave.

---

## COMPILER (`compiler/compiler.ls`, 4739 lines)

```
[✓] Lexer (Section 3): UTF-8 multi-byte for → ← ↑ ↓ Σ △ ▽ √ ≅ ≡
[✓] ARM64 backend (Section 1): 90+ emitters, syscalls, branch patching
[✓] GPU backend (Section 2): 35+ opcodes from probes, ctrl words, grid-sync
[✓] Safetensors reader (Section 5): JSON header + tensor index
[✓] ELF writer (Section 6): 9-section cubin ELF64 structure
[✓] Parser (Section 4): recursive descent, dual backend routing
[✓] Main entry (Section 7): argv → mmap → lex → parse → emit → write

Internal inconsistencies (Scout 2):
[ ] Parser handles superset: `goto`, `label`, `endfor`, `load_u8/16/32/64`,
    `syscall`, `if!=`, 3-operand stmts — compiler uses them but lexer rejects
[ ] `param NAME TYPE` lexed but never parsed → n_kparams always 0
[ ] No config.json reader (separate from safetensors; needed for layer_types)
[ ] No per-layer dispatch loop (the hybrid-layers.md design is unimplemented)
[ ] No megakernel linker
[ ] cubin_buf is 512KB; 64-layer megakernel needs ~300MB (600× too small)
[ ] `$` register prefix not a lexer token
[ ] arch/hopper.dict and arch/arm64.dict never read
[ ] ARM64 ELF wrapper absent (main writes raw bytes for host output)
```

---

## INFERENCE KERNELS (`inference/*.ls`, new grammar)

```
[✓] attend.ls — DeltaNet attention, RoPE (theta 10M, partial rotary 0.25)
[✓] attend_full.ls — full attention kernel for hybrid layers (GQA 24/4, output gate)
[✓] decay_gate.ls — softplus + exp decay
[✓] delta_update.ls — delta rule state update
[✓] deltanet_fused.ls — fused kernel with beta param
[✓] elementwise.ls — residual_add, elemwise_mul, activate_silu, scale
[✓] embed.ls — token embedding (FP16 and FP32 variants)
[✓] gemv.ls — W4A16 GEMV, GPTQ dequant chain CORRECT (scale + zero-point fixed)
[✓] recur.ls — conv1d, gate_sigmoid, deltanet_step, state_rollback
[✓] reduce.ls — rmsnorm, rmsnorm_residual, l2norm, sample_argmax

Still open from minds-revisited:
[ ] Apply v7 software pipelining to gemv.ls
[ ] Apply v2 v4.u32 K-wise weight loads
[ ] Per-projection shape selection (compile-time specialization per layer)
[ ] Transposed [N, K/8] weight layout (preprocess step)
```

---

## RUNTIME (`runtime/*.ls` — the libcuda replacement)

Replaces all 42 libcuda call sites in `src/launcher.s`.

```
[✓] init.ls — vfio-pci + BAR0/BAR4 mmap, PMC sanity, GSP boot dispatch
[✓] mem.ls — BAR4 bump allocator (coherent HBM, no dma_alloc_coherent)
[✓] elf_load.ls — parse compiled GPU ELF, extract entry_pc via .symtab walk
[✓] qmd.ls — 528-byte descriptor with empirically-probed offsets
[✓] cbuf0.ls — constant buffer (register_count offset UNKNOWN — TODO(probe))
[✓] pushbuffer.ls — 6-method GPFIFO submission sequence
[✓] doorbell.ls — USERD+0x8c GPPut write + USERD+0x90 doorbell
[✓] launch.ls — kernel dispatch, ties QMD+cbuf0+pushbuffer+doorbell
[✓] sync.ls — completion polling (flag-based + GPGet)
[✓] teardown.ls — munmap + close
[✓] dispatch.ls — top-level dispatch_kernel composition

None of these have been executed yet — the compiler must work first.
```

---

## HARDWARE PATH (`GSP/*.s` — ARM64 assembly)

```
[✓] bar_map.s — sysfs resource0/resource4 mmap (BDF hard-coded)
[✓] pmc_check.s — PMC_BOOT_0 read, PMC_INTR_0 clear
[✓] falcon_reset.s — Falcon reset/deassert sequence
[✓] hbm_alloc.s — bump allocator from BAR4
[✓] fw_load.s — ELF64 parse, PT_LOAD copy to HBM
[✓] bcr_start.s — RISC-V BCR register programming
[✓] poll_lockdown.s — PRIV_LOCKDOWN release polling
[✓] rpc_channel.s — channel allocation via GSP RPC

FSP subsystem wired:
[✓] FSP wired into GSP/boot.s at step 6 (before BCR at step 7)
[✓] BCR/FSP ordering fixed per NVIDIA reference
[✓] fsp_queue_advance_head added to emem_xfer.s
[✓] Stale .include directives removed from mctp/response/diag
[ ] vfio-pci: not implemented (sysfs mmap only)
[ ] register_count byte offset in cbuf0 unknown          ← probe dispatched
```

---

## LAUNCHER INTEGRATION

```
[✓] src/launcher.ls — 162 lines, replaces src/launcher.s
[ ] Removes all 42 libcuda call sites
[ ] Single dispatch per token (megakernel, not per-kernel)
[ ] Tokenize + emit stubs (full tokenizer later)
```

---

## DOCUMENTATION

```
[✓] docs/language-primitives.md — grammar spec
[✓] docs/model-config.md — Qwen 3.5 27B (num_layers 64, vocab 248320)
[✓] docs/hybrid-layers.md — 3+1 schedule, dims CORRECTED (24/4 heads, MLP 17408)
[✓] docs/gpu-register-model.md — $U, @P, $0:1, @smem, regcap
[✓] docs/qmd_fields.md — empirically probed offsets
[✓] docs/arm64-encodings.md — every instruction Lithos uses
[✓] docs/launcher-audit.md — 42 libcuda sites mapped
[✓] docs/minds-revisited.md — 12 GPTQ variants ranked under weights-as-code
[✓] docs/bootstrap-compile.md + v2 — bootstrap diagnosis
[✓] docs/weights-as-code-{compiler,language,inference}.md — corrected analyses
[✓] docs/gsp-native.md — eliminate C via vfio-pci
[✓] docs/self-hosting-compiler.md — architecture
[✓] docs/regalloc_design.md
[✓] HTML versions live on Vercel: /language, /pipeline, /roadmap, architecture,
   compiler, kernels, quantization, performance, bandwidth, glossary (26 pages, all 200)
[ ] PLAN/STATUS as HTML
```

---

## DEAD CODE REMOVED

```
[✓] 21 Forth files deleted (compiler/*.fs, gpu/*.fs, core.fs, patterns.fs, etc.)
[✓] 34 Python files deleted (src/, bench/, compiler/, kernels/, quantization/, tools/)
[✓] 21 orphan cubins deleted
[✓] 36 sass/ probe files deleted
[✓] Fake inference claims deleted (index.html "Paris" logit 18.057, cosine 1.0)
[ ] kernel/ (C kernel module) — keep until GSP boot fully runs, then delete
[ ] src/launcher.s (4096 lines) — delete when src/launcher.ls compiled + running
```

---

## CRITICAL BLOCKERS (in order)

1. **Bootstrap parser** — add expression parsing + composition syntax handling
2. **compiler.ls self-consistency** — fix 8 internal grammar issues from Scout 2
3. ~~**FSP boot wiring**~~ — DONE (wired at step 6, BCR/FSP order fixed)
4. **register_count probe** — differential probe for cbuf0 offset
5. **First compile**: bootstrap parses compiler.ls → lithos-stage1
6. **Self-compile**: lithos-stage1 compiles compiler.ls → lithos (fixed point)
7. **First kernel execution**: GSP+GPFIFO+QMD runs one compiled .ls kernel
8. **One token through one layer**: all 71 steps, diff < 1e-3 vs reference
9. **Full 64-layer inference**: Qwen 3.5 27B produces coherent text

---

## E2E SAFETY

- `bin/check-site` — verifies all 26 nav links return 200 before deploy
- Run before every `vercel --prod`
