# Lithos Parallel Execution Plan

Everything that can run simultaneously does. Blocked items wait.

---

## PROGRESS

```
WAVE 1:  [███████░] 6/8 complete, 2 running (1a, 1d)
WAVE 2:  [░░░░░░░░] 0/1 blocked
WAVE 3:  [░░░░░░░░] 0/4 blocked
WAVE 4:  [░░░░░░░░] 0/3 blocked
WAVE 5:  [░░░░░░░░] 0/2 blocked
TOTAL:   [████░░░░] 6/18
```

---

## WAVE 1 — no dependencies, start now

```
1a. [░░] Wire compiler.ls              IN PROGRESS — worker running
1b. [██] Rewrite 9 inference kernels   DONE
1c. [██] arch/hopper.dict + arm64.dict DONE (99 + 27 lines)
1d. [░░] QMD differential probe        IN PROGRESS — 4 probe tools written, no fields.md yet
1e. [██] Update HTML docs (.li→.ls)    DONE (language.html live, pipeline/roadmap updated)
1f. [██] Reconcile .ls precursors      DONE
1g. [██] Delete dead code (93 files)   DONE
1h. [██] GPU register model design     DONE
```

### 1a. Wire compiler.ls
- [ ] Combine 6 .ls files into one compiler.ls
- [ ] Main entry point (args, mmap, lex, parse, emit, write)
- [ ] Resolve naming conflicts between pieces
- [ ] Verify: all compositions present, no .li references
- [ ] Audit: entry point callable by Forth bootstrap

### 1b. Rewrite 9 inference kernels to new grammar
- [ ] attend.ls — compositions, \\, → ← ↑ ↓
- [ ] decay_gate.ls
- [ ] delta_update.ls
- [ ] deltanet_fused.ls
- [ ] elementwise.ls
- [ ] embed.ls
- [ ] gemv.ls
- [ ] recur.ls
- [ ] reduce.ls
- [ ] Audit: no fn keyword, no ->, no # comments, no \ comments

### 1c. Architecture register dictionaries
- [ ] arch/hopper.dict — probe all S2R register IDs on live GPU
- [ ] arch/arm64.dict — list system registers we use
- [ ] Audit: every register ID empirically verified via nvdisasm

### 1d. QMD differential probe
- [ ] Experiment 1: find entry_pc byte offset
- [ ] Experiment 2: find register_count byte offset
- [ ] Experiment 3: find shared_mem_size byte offset
- [ ] Write results to docs/qmd_fields.md
- [ ] Audit: each offset confirmed by two independent diffs

### 1e. Update HTML docs
- [ ] .li → .ls across all HTML files
- [ ] Create docs/language.html (grammar spec page)
- [ ] Update pipeline diagram for self-hosting era
- [ ] Update roadmap status
- [ ] Add language.html to navigation
- [ ] Audit: no .li references remain in any HTML file

### 1f. Reconcile .ls precursor files ✓
- [x] kernels.ls — compositions, \\, △ ▽ # ≅ ≡
- [x] primitives.ls — all grammar primitives listed
- [x] derivatives.ls — compositions, △ ▽ #
- [x] Audit: exact match to language-primitives.md grammar

### 1g. Delete dead code ✓
- [x] 34 Python files deleted
- [x] 21 orphan cubins deleted (9 kept with .ls source)
- [x] 36 sass/ probe files deleted (kept encoding_sm90.json)
- [x] 5 build artifacts deleted
- [x] Audit: no Python importable, no orphan binaries

### 1h. GPU register model design ✓
- [x] $U0-$U63 uniform register syntax
- [x] @P0-@P6 predicate syntax
- [x] $0:1 64-bit pair syntax
- [x] @smem shared memory qualifier
- [x] regcap N register budget directive
- [x] Control words: compiler-managed only
- [x] Register allocation roadmap (bump → linear scan → graph coloring)
- [x] Audit: all 9 register model points covered with .ls examples

---

## WAVE 2 — blocked on 1a (compiler.ls wired)

```
2a. [░░] Bootstrap compile              BLOCKED on 1a
```

### 2a. Bootstrap compile
- [ ] Forth bootstrap compiles compiler.ls → lithos-stage1
- [ ] lithos-stage1 is a valid ARM64 ELF
- [ ] lithos-stage1 runs (prints usage or processes a .ls file)
- [ ] Audit: file -v confirms ARM64 ELF, ldd shows statically linked

---

## WAVE 3 — blocked on 2a (bootstrap succeeds)

```
3a. [░░] Self-compile                   BLOCKED on 2a
3b. [░░] GSP boot from .ls             BLOCKED on 2a
3c. [░░] GPFIFO + doorbell from .ls    BLOCKED on 2a + 1d
3d. [░░] Model-to-binary compiler      BLOCKED on 2a + 1b
```

### 3a. Self-compile + fixed point
- [ ] lithos-stage1 compiles compiler.ls → lithos
- [ ] diff lithos-stage1 lithos = identical (fixed point)
- [ ] lithos compiles compiler.ls → lithos-verify
- [ ] diff lithos lithos-verify = identical
- [ ] Audit: three-way identical binaries

### 3b. GSP boot from .ls
- [ ] Port lithos_gsp.c register pokes to ← → compositions
- [ ] FSP boot commands (~2000 lines)
- [ ] vfio-pci setup composition
- [ ] GSP firmware loads and responds
- [ ] Audit: GSP PRIV_LOCKDOWN releases, PMC_BOOT_0 readable post-boot

### 3c. GPFIFO + doorbell from .ls
- [ ] Channel creation via GSP RPC
- [ ] QMD pushbuffer construction (uses 1d probe results)
- [ ] GPFIFO ring write composition
- [ ] Doorbell store composition
- [ ] Audit: USERD doorbell page mapped, GPPut write succeeds

### 3d. Model-to-binary compiler
- [ ] Safetensors wired into compilation pipeline
- [ ] Per-layer weight address hardcoding
- [ ] Broadcast scalar embedding as FMUL-IMM/FADD-IMM immediates
- [ ] Per-layer instruction stream generation
- [ ] Audit: one projection layer compiles to valid GPU ELF

---

## WAVE 4 — blocked on wave 3

```
4a. [░░] Delete Forth + kernel/         BLOCKED on 3a
4b. [░░] Megakernel linker              BLOCKED on 3d
4c. [░░] First kernel execution         BLOCKED on 3b + 3c + 3d
```

### 4a. Delete Forth + kernel/
- [ ] All .fs files deleted
- [ ] kernel/ directory deleted
- [ ] compiler/arm64-wrap.fs, elf-wrap.fs deleted
- [ ] No Forth anywhere in the repo
- [ ] Audit: grep -r "\.fs" finds zero source files

### 4b. Megakernel linker
- [ ] Concatenate per-layer instruction streams
- [ ] Grid-sync instructions between layers
- [ ] Forward pass megakernel ELF
- [ ] Recurrence update megakernel ELF
- [ ] Audit: both ELFs have valid .nv.info cooperative launch attributes

### 4c. First kernel execution without CUDA
- [ ] GSP booted (3b)
- [ ] GPFIFO channel ready (3c)
- [ ] Valid GPU ELF loaded to HBM via BAR4 (3d)
- [ ] QMD dispatched, doorbell rung
- [ ] GPU executes, result readable from BAR4
- [ ] Audit: output buffer contains correct computation (compare vs CUDA reference)

---

## WAVE 5 — first token

```
5a. [░░] One token through one layer    BLOCKED on 4b + 4c
5b. [░░] Full 48-layer forward pass     BLOCKED on 5a
```

### 5a. One token, one layer
- [ ] All 71 DeltaNet decomposition steps execute
- [ ] RMSNorm → projections → delta rule → output gate → residual
- [ ] Output matches PyTorch reference (within FP32 tolerance)
- [ ] Audit: element-wise diff < 1e-3 vs reference

### 5b. Full model inference
- [ ] Qwen 2.5 27B, 48 layers, forward + recurrence
- [ ] Token embedding → 48 layers → LM head → argmax
- [ ] Produces coherent text
- [ ] Audit: matches reference model output for fixed prompt

---

## CRITICAL PATH

```
1a ──→ 2a ──→ 3a ──→ 4a    (self-hosting confirmed, delete Forth)
              ├──→ 3b ─┐
  1d ─────────┤        ├→ 4c ──→ 5a ──→ 5b
              ├──→ 3c ─┘
  1b ─────────┴──→ 3d ──→ 4b ─┘
```

Fastest path to first token: 1a → 2a → 3d → 4b+4c → 5a
