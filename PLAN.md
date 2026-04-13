# Lithos Parallel Execution Plan

Target: Qwen 3.5 27B (Huihui-abliterated, GPTQ W4A16), 64 hybrid layers.
Everything that can run simultaneously does. Blocked items wait.

---

## PROGRESS

```
WAVE 1:  [████████] 8/8 DONE
WAVE 2:  [░░░░░░░░] 0/1 BLOCKED — bootstrap has 10 SIGBUS bugs + grammar mismatch
WAVE 3:  [░░░░░░░░] 0/4 blocked
WAVE 4:  [░░░░░░░░] 0/3 blocked
WAVE 5:  [░░░░░░░░] 0/2 blocked
TOTAL:   [████░░░░] 8/18
```

---

## BOOTSTRAP ARCHITECTURE

The bootstrap is pure ARM64 assembly in `bootstrap/` — there is no Forth.
Files:
  - `lithos-bootstrap.s` — entry, memory, core runtime
  - `lithos-lexer.s` — tokenizer
  - `lithos-parser.s` — parser
  - `emit-arm64.s` — ARM64 emitter
  - `lithos-elf-writer.s` — ELF output
  - `driver.s` — main loop orchestrating above
  - `ls-shared.s`, `lithos-expr.s`

Job: read `compiler.ls`, emit an ARM64 binary `lithos-stage1`.
After `lithos-stage1` self-compiles to `lithos` (fixed point), the bootstrap
can be retired but doesn't need to be deleted — it's a few thousand lines
of .s and has no dependencies to remove.

---

## WAVE 1 — done ✓

```
1a. [✓] compiler.ls wired               4739 lines, 7 sections
1b. [✓] 9 inference kernels rewritten   20 compositions, new grammar
1c. [✓] arch/ dictionaries              hopper.dict (99), arm64.dict (27)
1d. [✓] QMD fields probed               shared_mem@0x2c, entry_pc@0x118/0x11c
                                         register_count: NOT in QMD (in cbuf0)
1e. [✓] HTML docs updated               /language live, pipeline/roadmap current
1f. [✓] .ls precursors reconciled       kernels.ls, primitives.ls, derivatives.ls
1g. [✓] dead code deleted               93 files (Python, orphan cubins, probes)
1h. [✓] GPU register model designed     $U0-$U63, @P0-@P6, $0:1, @smem, regcap
```

---

## WAVE 2 — BLOCKED: bootstrap bugs

**Status: bootstrap worker ran, reported fundamental bugs, did NOT produce lithos-stage1.**

Diagnosis in `docs/bootstrap-compile.md`. The blockers:

```
2a. [ ] lithos-elf-writer.s — 10 stack-alignment bugs (str x20, [sp, #-8]!)
        Breaks 16-byte SP alignment → SIGBUS on every ELF emit
2b. [ ] lithos-lexer.s missing NEXT macro — can't self-assemble in isolation
2c. [ ] lithos-expr.s dead + unassembleable (12+ errors) — delete or repair
2d. [ ] driver.s needs .global decls and _start rename merged into runtime
2e. [ ] Grammar: lexer doesn't recognize `fn` in examples/*.ls — decide:
        migrate examples to new grammar, or add `fn` back to lexer
2f. [ ] Minimal smoke test: compile `var x 42` end-to-end before compiler.ls
2g. [ ] Re-run compiler.ls through bootstrap — expect parse failures initially
```

When these are fixed, Wave 2 = "bootstrap produces working `lithos-stage1` that
can process at least one .ls file end-to-end, write a valid binary."

---

## WAVE 3 — blocked on 2 (stage1 works)

```
3a. [ ] Self-compile                    stage1 compiles compiler.ls → lithos; diff
3b. [ ] GSP boot from .ls               port lithos_gsp.c register pokes to ← →
3c. [ ] GPFIFO + doorbell from .ls      channel via GSP RPC, QMD, USERD store
3d. [ ] Model-to-binary compiler        safetensors wired, per-layer weight hardcode
```

---

## WAVE 4 — blocked on wave 3

```
4a. [ ] Retire bootstrap + delete kernel/   after self-compile confirmed
4b. [ ] Megakernel linker                    forward + recurrence ELFs, grid-sync
4c. [ ] First kernel execution              GSP + GPFIFO + valid ELF → result
```

---

## WAVE 5 — first token

```
5a. [ ] One token through one layer     all 71 steps, diff < 1e-3 vs reference
5b. [ ] Full model inference            Qwen 3.5 27B, 64 layers (hybrid)
                                         Token embed → 64 layers → LM head → argmax
                                         Produces coherent text
```

---

## CRITICAL PATH

```
BOOTSTRAP BUGS ──→ stage1 ──→ self-compile ──→ retire bootstrap
                               ├──→ 3b GSP ──┐
           1d QMD ─────────────┤             ├→ first exec ──→ one token ──→ full model
                               ├──→ 3c GPFIFO┘
           1b kernels ─────────┴──→ 3d model compiler ──→ megakernel ──┘
```

Fastest path: fix bootstrap → stage1 → 3d → 4b+4c → 5a → 5b

---

## E2E SAFETY

- `bin/check-site`: verifies every nav link returns 200 before deploy
- Run before every `vercel --prod`
- 26 paths currently verified (as of last check)
