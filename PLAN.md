# Lithos Parallel Execution Plan

Everything that can run simultaneously does. Blocked items wait.

---

## WAVE 1 — no dependencies, start now

All of these can run in parallel immediately.

```
┌─────────────────────────────────────────────────────────────────┐
│ 1a. Wire compiler.ls                                            │
│     Combine lithos-lexer.ls + lithos-parser.ls + emit-arm64.ls  │
│     + emit-gpu.ls + lithos-safetensors.ls + lithos-elf.ls       │
│     into one compiler.ls with main entry point                  │
│     Output: compiler.ls that the Forth bootstrap can compile    │
├─────────────────────────────────────────────────────────────────┤
│ 1b. Rewrite 9 inference kernels to new grammar                  │
│     attend.ls, decay_gate.ls, delta_update.ls,                  │
│     deltanet_fused.ls, elementwise.ls, embed.ls,                │
│     gemv.ls, recur.ls, reduce.ls                                │
│     All independent — 9 workers in parallel                     │
├─────────────────────────────────────────────────────────────────┤
│ 1c. Write arch/hopper.dict + arch/arm64.dict                    │
│     Probe all S2R register IDs on live GPU                      │
│     List ARM64 system registers we use                          │
├─────────────────────────────────────────────────────────────────┤
│ 1d. QMD differential probe                                      │
│     Launch two kernels via CUDA, diff pushbuffers               │
│     Find entry_pc, register_count, shared_mem_size byte offsets │
│     Uses existing tools/dump_qmd.c                              │
├─────────────────────────────────────────────────────────────────┤
│ 1e. Update all HTML docs                                        │
│     .li → .ls everywhere                                        │
│     New grammar examples                                        │
│     Language spec as HTML page                                   │
│     Pipeline diagram for self-hosting era                       │
├─────────────────────────────────────────────────────────────────┤
│ 1f. Reconcile .ls precursor files                               │
│     kernels.ls, primitives.ls, derivatives.ls                   │
│     Update to exact new grammar                                 │
├─────────────────────────────────────────────────────────────────┤
│ 1g. Delete dead code                                            │
│     Python: src/engine.py, src/cuda_driver.py,                  │
│             inference/load_compiled.py                           │
│     Old cubins: kernels/*.cubin (no .ls source)                 │
│     sass/ remnants                                              │
│     (keep .fs files until bootstrap succeeds)                   │
├─────────────────────────────────────────────────────────────────┤
│ 1h. Design: GPU register model                                  │
│     Uniform registers ($U0-$U63) in the language                │
│     Predicate registers (P0-P6) in the language                 │
│     64-bit register pairs                                       │
│     Shared memory declaration syntax                            │
│     Warp scheduling / control word model                        │
│     Cooperative grid-sync expression                            │
└─────────────────────────────────────────────────────────────────┘
```

## WAVE 2 — blocked on 1a (compiler.ls wired)

```
┌─────────────────────────────────────────────────────────────────┐
│ 2a. Bootstrap compile                                           │
│     Forth bootstrap compiles compiler.ls → lithos-stage1        │
│     This is the first ARM64 binary produced by Lithos           │
│     BLOCKED ON: 1a                                              │
└─────────────────────────────────────────────────────────────────┘
```

## WAVE 3 — blocked on 2a (bootstrap succeeds)

```
┌─────────────────────────────────────────────────────────────────┐
│ 3a. Self-compile                                                │
│     lithos-stage1 compiles compiler.ls → lithos                 │
│     diff lithos-stage1 lithos (fixed point test)                │
│     BLOCKED ON: 2a                                              │
├─────────────────────────────────────────────────────────────────┤
│ 3b. GSP boot from .ls                                           │
│     Port lithos_gsp.c register sequences to ← → compositions   │
│     FSP boot commands (~2000 lines)                             │
│     vfio-pci setup composition                                  │
│     BLOCKED ON: 2a (needs working compiler for ARM64 output)    │
├─────────────────────────────────────────────────────────────────┤
│ 3c. GPFIFO + doorbell from .ls                                  │
│     Channel creation via GSP RPC                                │
│     QMD pushbuffer construction (uses 1d probe results)         │
│     Ring write + doorbell store                                 │
│     BLOCKED ON: 2a + 1d                                         │
├─────────────────────────────────────────────────────────────────┤
│ 3d. Model-to-binary compiler                                    │
│     Wire safetensors into compilation                           │
│     Per-layer weight address hardcoding                         │
│     Broadcast scalar embedding as immediates                    │
│     Per-layer instruction stream generation                     │
│     BLOCKED ON: 2a + 1b (needs working compiler + kernel specs) │
└─────────────────────────────────────────────────────────────────┘
```

## WAVE 4 — blocked on wave 3

```
┌─────────────────────────────────────────────────────────────────┐
│ 4a. Delete .fs files + kernel/ directory                        │
│     Only after 3a succeeds (self-hosting confirmed)             │
│     BLOCKED ON: 3a                                              │
├─────────────────────────────────────────────────────────────────┤
│ 4b. Megakernel linker                                           │
│     Concatenate per-layer instruction streams                   │
│     Grid-sync instructions between layers                       │
│     Two megakernels: forward + recurrence                       │
│     BLOCKED ON: 3d                                              │
├─────────────────────────────────────────────────────────────────┤
│ 4c. First kernel execution without CUDA                         │
│     GSP booted + GPFIFO ready + valid GPU ELF                   │
│     BLOCKED ON: 3b + 3c + 3d                                    │
└─────────────────────────────────────────────────────────────────┘
```

## WAVE 5 — first token

```
┌─────────────────────────────────────────────────────────────────┐
│ 5a. One token through one layer                                 │
│     All 71 steps of the DeltaNet decomposition                  │
│     BLOCKED ON: 4b + 4c                                         │
├─────────────────────────────────────────────────────────────────┤
│ 5b. Full 48-layer forward pass                                  │
│     Qwen 2.5 27B end-to-end                                    │
│     BLOCKED ON: 5a                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
WAVE 1:  8 parallel tracks (no blockers)
WAVE 2:  1 track (bootstrap compile)
WAVE 3:  4 parallel tracks
WAVE 4:  3 parallel tracks
WAVE 5:  2 sequential milestones

Critical path: 1a → 2a → 3a → 4a (self-hosting confirmed, delete Forth)
Hardware path: 1a → 2a → 3b + 3c → 4c → 5a → 5b
Model path:    1a + 1b → 2a → 3d → 4b → 5a → 5b
```
