# Documentation Inconsistency Audit — Pass 1

Date: 2026-04-14
Status: OPEN — fixes pending review

---

## CRITICAL — Factual errors

### ~~1. Compiler line count: 4,739 vs 5,467~~ FIXED

### ~~2. Vocabulary size: 152,064 vs 248,320~~ FIXED

---

## HIGH — Broken file references (from reorg)

### ~~3. README.md references old doc paths~~ FIXED

### ~~4. STATUS.md references old doc paths~~ FIXED

### 5. REVIEW.md references root-level files that moved

- Line 5: References `PRIMITIVES.md`, `VOCABULARY.md`, `GAPS.md`, `PLAN.md` without paths
- **Actual locations**: `docs/legacy/PRIMITIVES.md`, `docs/language/VOCABULARY.md`, `docs/legacy/GAPS.md`, `docs/legacy/PLAN.md`
- **Fix**: Leave as-is — REVIEW.md is a historical document (Chuck Moore review from 2026-04-14). The references describe what existed at time of review. Adding a note at top is sufficient.

---

## HIGH — Design specification vs implementation

### 6. Primitive count: 12 vs 25 vs 30

- **docs/language/SPECIFICATION.md** Section 7: "Total primitive families: 12. Total unique symbols/names: 25."
- **docs/language/VOCABULARY.md**: "Total: 25 symbols / names. The 12 irreducible operation families"
- **docs/language/REVIEW.md** (Chuck Moore): "The honest primitive set is not 12 families. It is closer to 30 operations" — lists `fma`, `neg`, `rcp`, `rsqrt`, `shr`, `shl`, `and`, `or`, `xor`, `u32>f32`, `shfl.down`, `barrier` as additional primitives used in actual `inference/*.ls` code
- **Decision needed**: Are 12 the language-level primitives and 30 the compiler-level primitives? If so, state this explicitly. If the language genuinely has 30 primitives, update the spec.
- **Fix**: Add clarifying note to SPECIFICATION.md Section 7: "12 conceptual families at the language level. The compiler recognizes ~30 low-level operations including fma, bitwise ops, warp shuffles, and barriers (see REVIEW.md Section 8)."

### 7. Three-language problem (per Chuck Moore review)

The spec describes three overlapping syntaxes:

| Source | Syntax style | Assignment | Array access |
|--------|-------------|------------|--------------|
| `docs/language/primitives.md` | Arrow/Unicode | `name expr` (no `=`) | `→ width addr` |
| `inference/*.ls` | Register-transfer | `x = expr` | `a[i]` |
| `kernels.ls` / `VOCABULARY.md` | Stack/postfix | implicit stack | implicit stack |

- **REVIEW.md** Section 1: "The spec must describe the language the compiler actually parses."
- **Decision needed**: Which syntax is canonical? The compiler parses the register-transfer style (`=` and `[]`). The spec describes the arrow/Unicode style. One must change.
- **Fix**: This is a design decision, not a docs fix. Once decided, update SPECIFICATION.md to match.

---

## MEDIUM — Internal inconsistencies

### 8. DeltaNet state matrix shape description

- **docs/language/SPECIFICATION.md** Section 8.2: "S[48, 128, 128]" — implies 48 as a dimension
- **STATUS.md**: "48 layers x 16 heads x 128 x 128 x f32 = 48 MB"
- **docs/inference/hybrid-layers.md**: "deltanet_state: f32[48, 16, 3, 128, 128]" — 48 layers, 16 K-heads, 3 V-heads per K-head
- **Issue**: SPECIFICATION.md omits the per-head breakdown. "S[48, 128, 128]" is the per-head shape, not the full state.
- **Fix**: Clarify in SPECIFICATION.md: "State matrix S per head: [128, 128] FP32. Total: 48 layers x 16 heads x 128 x 128 x 4 bytes = 48 MB per sequence."

### 9. Bootstrap parser line count

- **STATUS.md** line 231: "lithos-parser.s | ~2,961 lines"
- **STATUS.md** line 252: ".bak files contain 2,952 lines for parser"
- **Issue**: 2,961 vs 2,952. Minor but should be consistent.
- **Fix**: Verify actual line count, update both references.

### 10. README.md compiler line count in "What Ships" context

- **README.md** line 42: "lithos (compiler + runtime, ~1 MB ARM64 ELF)"
- **SHIPPING.md** line 9: "qwen3_w4a16.safetensors"
- **SHIPPING.md** line 6: Lists binary + weights as the two shipping artifacts
- **Status**: Consistent between README and SHIPPING.

### 11. GH200 HBM bandwidth

- **docs/hardware/performance-spectrum.md**: Speculative analysis referencing "GH200 hardware specs"
- **docs/legacy/PRIMITIVES.md**: No bandwidth figure
- **docs/inference/model-config.md**: No bandwidth figure
- **Issue**: No docs state the canonical HBM3e bandwidth figure (4.8 TB/s for GH200 480GB or 900 GB/s for GH200 96GB). Different GH200 SKUs exist.
- **Fix**: Add canonical hardware spec to model-config.md or a hardware reference doc.

---

## LOW — Style and terminology

### 12. "sm_90" vs "SM_90a" vs "SM90"

- **docs/language/SPECIFICATION.md**: "Hopper SM_90a"
- **docs/runtime/qmd-hopper.md**: "Hopper sm_90"
- **docs/runtime/cbuf0-fields.md**: "Hopper sm_90"
- **STATUS.md**: Mixed "SM90" and "sm_90"
- **Fix**: Standardize to lowercase `sm_90a` in technical docs (matches NVIDIA convention).

### 13. README shorthand for layer architecture

- **README.md** line 15: "3 DeltaNet + 1 full attention x 16"
- **All other docs**: "48 DeltaNet + 16 full attention"
- **Issue**: README shorthand could be misread as "(3 DeltaNet + 1 full attention) x 16" which equals 64, or "3 DeltaNet + (1 full attention x 16)" which equals 19.
- **Fix**: Change to "64 hybrid layers: 48 DeltaNet + 16 full attention (pattern: 3+1, repeating x16)"

### 14. "weights-as-code" appears in three directories

- `docs/language/weights-as-code.md` — "Lithos Language v2"
- `docs/compiler/design/weights-as-code.md` — "Compiler Architecture"
- `docs/inference/weights-as-code.md` — "Inference Architecture"
- **Issue**: Three files with the same name in different directories. Not wrong, but confusing for navigation.
- **Fix**: Consider renaming to `weights-as-code-language.md`, `weights-as-code-compiler.md`, `weights-as-code-inference.md` for disambiguation. Or consolidate into one document.

### 15. SHIPPING.md refers to PLAN.md

- **SHIPPING.md** may reference PLAN.md as active
- **Actual**: PLAN.md is in docs/legacy/ and marked deprecated
- **Fix**: Check SHIPPING.md for any PLAN.md references and update to STATUS.md

### 16. qmd.ls source comment about register_count location

- **STATUS.md** line 21: "qmd.ls comment claiming 'register_count lives in cbuf0' is wrong. SPD 0x094 is correct"
- **Issue**: Source code comment is misleading. Not a docs issue per se, but STATUS.md flags it.
- **Fix**: Update `runtime/qmd.ls` comment to: "register_count lives in Shader Program Descriptor at offset 0x094"

---

## Summary

| Severity | Count | Status |
|----------|------:|--------|
| ~~CRITICAL~~ | ~~2~~ | ~~Factual errors (line count, vocab size)~~ FIXED |
| HIGH | 3 | ~~Broken paths (3-4) FIXED~~, REVIEW.md (5), design spec issues (6-7) |
| MEDIUM | 4 | Internal inconsistencies (8-11) |
| LOW | 5 | Style/terminology (12-16) |
| **Total** | **16** | |

## Next steps

1. Fix CRITICAL (#1, #2) — straightforward text edits
2. Fix HIGH broken paths (#3, #4) — find-and-replace
3. Decide on design questions (#6, #7) — owner decision
4. Fix MEDIUM (#8, #9, #11) — clarification edits
5. Standardize LOW (#12-16) — style pass
6. Re-audit after fixes (Pass 2)
