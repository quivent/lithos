# Documentation Cleanup Report

Date: 2026-04-13
Source: Scout 1 report (/tmp/scout1-report.md)

## Files Deleted

| File | Reason |
|------|--------|
| STATUS.md | Obsolete Python/vLLM era content. PLAN.md supersedes. |
| TODO.md | Referenced .li files, Forth compiler. PLAN.md supersedes. |
| SPEC.md | Old Forth-era language spec (core.fs, patterns.fs). docs/language-primitives.md supersedes. |
| compiler/LANGUAGE.md | Old fn/-> grammar, .li extension. docs/language-primitives.md supersedes. |
| minds/shannon_smrs_v3_gptq.py | Dead Python. |
| minds/test_gptq_gemv_transposed.py | Dead Python. |
| minds/test_gptq_gemv_ultra.py | Dead Python. |
| minds/test_gptq_gemv_v2v3.py | Dead Python. |
| minds/transpose_gptq_weights.py | Dead Python. |
| bench/harness.py | Dead Python benchmark harness. |
| kernels/__pycache__/transpose_gptq_weights.cpython-310.pyc | Stale bytecode. |

## Files Rewritten

| File | Changes |
|------|---------|
| README.md | Complete rewrite. Removed all Python/vLLM/Forth era content. Now describes .ls source, self-hosting compiler, ARM64 bootstrap, runtime/*.ls dispatch. |
| compiler/examples/hello.ls | Old fn/-> grammar and single-backslash comments -> new composition syntax (name args :) and double-backslash comments. |
| compiler/examples/vadd.ls | Same grammar update. |
| compiler/examples/scale_add.ls | Same grammar update. |
| compiler/examples/residual_add.ls | Same grammar update. |
| compiler/examples/fused_scale_bias.ls | Same grammar update. |
| compiler/examples/gemv_test.ls | Same grammar update. |

## Files Edited (targeted fixes)

### HIGH priority

| File | Fix |
|------|-----|
| docs/weights-as-code-inference.md | 3584->5120, 152064->248320, engine.py refs removed, .li->.ls |
| docs/weights-as-code-compiler.md | D=3584->5120, MLP=14336->17408, 36 layers->64, all bandwidth calcs recomputed, .li->.ls |

### MEDIUM priority

| File | Fix |
|------|-----|
| docs/self-hosting-compiler.md | All .li->.ls (~25 occurrences) |
| docs/regalloc_design.md | All .li->.ls |
| docs/weights-as-code-language.md | All .li->.ls |
| GAPS.md | .li->.ls, Qwen3-30B-A3B->Qwen 3.5 27B, removed Python references |
| SHIPPING.md | .li->.ls |
| docs/index.html | Added 9 missing nav links (memory, minds, recipe, sm-architecture, warp-scheduler, arch-cores, arch-gpu, arch-kernel, arch-registers) |
| cublas-catalog/README.md | vocab_size 151936->248320 |

### LOW priority (Qwen spelling normalization)

| File | Fix |
|------|-----|
| docs/deltanet-operations.md | Qwen3.5-27B -> Qwen 3.5 27B |
| docs/token-journey.html | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/kernels.html | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/quantization.html | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/tensorrt.html | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/recipe.html | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/search-data.js | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/weights-as-code-inference.md | Qwen 3.5-27B -> Qwen 3.5 27B |
| docs/minds/shannon_quantization.md | Qwen 3.5-27B -> Qwen 3.5 27B |

## Files Kept (not superseded)

| File | Reason |
|------|--------|
| PRIMITIVES.md | Unique content: primitives-to-binary mapping. Not duplicated elsewhere. |
| VOCABULARY.md | Unique content: DeltaNet layer decomposition into 41 primitive steps. |
| PLAN.md | Canonical status/plan document. |
| GAPS.md | Updated to current state (fixed .li, model name, Python refs). |
| SHIPPING.md | Updated .li->.ls. Content still current. |

## Not Changed (per instructions)

- No .ls source logic modified
- No bootstrap/*.s files touched
- No runtime/*.ls files touched
- No compiler/*.fs files touched

## Verification

Final grep for stale patterns returns zero hits (excluding legitimate forth-bootstrap binary references):
  grep -rn 'Qwen 2.5|.li |.li$|.li"|gforth|cubin-wrap|151936|152064' docs/ *.md compiler/examples/
