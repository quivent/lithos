# Documentation Inconsistency Audit — Pass 1

Date: 2026-04-14
Status: PARTIAL — pass 1 fixes landed 2026-04-14, remaining items are design decisions

---

## CRITICAL — Factual errors

### ~~1. Compiler line count: 4,739 vs 5,467~~ FIXED

### ~~2. Vocabulary size: 152,064 vs 248,320~~ FIXED

---

## HIGH — Broken file references (from reorg)

### ~~3. README.md references old doc paths~~ FIXED

### ~~4. STATUS.md references old doc paths~~ FIXED

### ~~5. REVIEW.md references root-level files that moved~~ FIXED

- Historical banner added at top of `docs/language/REVIEW.md` documenting that the doc is preserved as-is and listing the current locations of the cited files.

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

### ~~8. DeltaNet state matrix shape description~~ FIXED

- `docs/language/SPECIFICATION.md` Section 8.2 now states the per-head shape (`S[128, 128] FP32`) and the full-model footprint (`48 × 16 × 128 × 128 × 4 = 48 MB`), with a pointer to `docs/inference/hybrid-layers.md` for the complete layout.

### ~~9. Bootstrap parser line count~~ FIXED

- STATUS.md now cites the live parser (`lithos-table.s`, 3,450 lines) and flags `lithos-parser.s` (3,279 lines) as dead/superseded. The `.bak` file reference was removed because it was comparing against a file that isn't the live parser anyway.

### ~~10. README.md compiler line count in "What Ships" context~~ ALREADY CONSISTENT

- README and SHIPPING agree on the two shipping artifacts. No change needed.

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

### ~~13. README shorthand for layer architecture~~ ALREADY FIXED

- README line 15 already reads "64 hybrid layers: 48 DeltaNet + 16 full attention (pattern: 3+1, repeating x16)".

### 14. "weights-as-code" appears in three directories

- `docs/language/weights-as-code.md` — "Lithos Language v2"
- `docs/compiler/design/weights-as-code.md` — "Compiler Architecture"
- `docs/inference/weights-as-code.md` — "Inference Architecture"
- **Issue**: Three files with the same name in different directories. Not wrong, but confusing for navigation.
- **Fix**: Consider renaming to `weights-as-code-language.md`, `weights-as-code-compiler.md`, `weights-as-code-inference.md` for disambiguation. Or consolidate into one document.

### ~~15. SHIPPING.md refers to PLAN.md~~ NO ACTION NEEDED

- Verified: SHIPPING.md contains no PLAN.md references.

### ~~16. qmd.ls source comment about register_count location~~ ALREADY FIXED

- `runtime/qmd.ls:9-12` already says: "register_count is NOT in the QMD body and NOT in cbuf0. It lives in the Shader Program..." — correct, no change needed.

---

## Summary

| Severity | Count | Status |
|----------|------:|--------|
| ~~CRITICAL~~ | ~~2~~ | FIXED — factual errors (line count, vocab size) |
| HIGH | 3 | ~~Broken paths (3, 4) FIXED~~, ~~REVIEW.md banner (5) FIXED~~, design-decision items remaining (6, 7) |
| MEDIUM | 4 | ~~DeltaNet shape (8) FIXED~~, ~~parser line count (9) FIXED~~, ~~README/SHIPPING (10) consistent~~, bandwidth canonical doc (11) remains |
| LOW | 5 | sm_90 style (12), ~~README layer shorthand (13) already fixed~~, triple weights-as-code (14) design, ~~SHIPPING/PLAN (15) no refs~~, ~~qmd.ls comment (16) already correct~~ |
| **Total** | **16** | **11 closed, 5 remaining (all design decisions or style passes)** |

## Remaining items

- **#6** Primitive count (12 vs 25 vs 30): design decision. Add clarifying note once owner picks the framing.
- **#7** Three-language problem: design decision. The live parser accepts register-transfer style (`x = expr`, `a[i]`); spec describes arrow/Unicode style. One must change.
- **#11** GH200 HBM bandwidth: no canonical doc. Add a single hardware-reference section with 4.8 TB/s (480GB SKU) vs 900 GB/s (96GB SKU).
- **#12** sm_90 standardization: 18 files mix `sm_90`, `SM_90`, `SM_90a`. Pick one (NVIDIA convention is lowercase `sm_90a`) and sweep.
- **#14** `weights-as-code.md` in three directories: rename with suffix or consolidate. Navigation-friendliness issue, not correctness.

## Next steps

- Owner decisions on #6, #7, #14 (design).
- Mechanical style pass on #12 (can be done anytime).
- One-paragraph add to hardware doc for #11.
- Re-audit after all items closed.
