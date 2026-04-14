# Lithos Language Specification Review

**Reviewer**: Chuck Moore
**Date**: 2026-04-14
**Sources**: `docs/language-primitives.md`, `PRIMITIVES.md`, `VOCABULARY.md`, `GAPS.md`, `PLAN.md`, `SHIPPING.md`, `compiler/compiler.ls`, `inference/*.ls`, `kernels.ls`, `derivatives.ls`, `primitives.ls`

> **Historical document.** This review was written against the repo state on 2026-04-14 and cites filenames that have since moved or been retired: `PRIMITIVES.md` / `GAPS.md` / `PLAN.md` are now under `docs/legacy/` (deprecated); `VOCABULARY.md` is at `docs/language/VOCABULARY.md`. The review is preserved as-is for the record — do not patch filenames into the body.

---

## What I Like

The instinct is right. One language, two targets, compositions that flatten. No functions, no call stack, no ABI. The compiler sees everything and inlines everything. That is how you get to the metal.

"Fusion is the default. There is nothing to fuse because nothing was ever separated." That sentence is correct. It is also the most important sentence in the entire specification. Most GPU programmers spend their careers gluing kernels back together that should never have been apart. Lithos starts from the right place.

The shipping story is correct too. Two files on disk. One binary, one weights file. Sub-second cold start. `scp` two files, run the binary. That is how computing should work. Everything else -- Python, CUDA toolkit, PyTorch, Docker, conda -- is infrastructure to manage infrastructure. Lithos skips it. Good.

The dimensional notation -- `*`, `**`, `***`, `****` -- is the best idea in the language. Each symbol adds a loop. The programmer sees the dimensionality of the operation in the operator itself. `**` is a vector operation. `***` is a matrix operation. You do not need to read a type signature or count array indices. The loop depth is in the name. That is what notation should do: make the structure visible.

The MUFU mapping is clean. `1/` is one instruction. `1/sqrt` is one instruction. `sqrt` is two (reciprocal-square-root then reciprocal). The programmer sees the hardware cost in the symbol. No mystery about what is fast and what is not.

The libcuda replacement is the right ambition. 450 ioctls at startup, then pure memory writes for dispatch. A few hundred lines of Lithos replacing 80 MB of `libcuda.so`. That ratio tells you everything about what is wrong with conventional systems.

The megakernel architecture -- two cooperative cubins, 48 layers and 16 layers, grid sync between layers, no per-layer dispatch -- is the correct design. Per-kernel dispatch is a 1990s idea. The GPU runs a program. Let it run the whole program.

---

## What Is Missing

### 1. The spec has three languages, not one

Read `language-primitives.md`. It describes a language with arrows (`->`, `<-`), `each`, `for`, `stride`, `if==`, `trap`, and register syntax (`$N`, `$NAME`). Compositions are `name args :` with indented bodies. Bindings are `name expr` (no `=`).

Now read `inference/gemv.ls`. It uses `=` for assignment. It uses `[ ]` for array indexing. It uses `if>=` with labels. It uses `shr`, `shl`, `and`, `mul`, `add`, `fma`, `neg`, `rcp`, `rsqrt`, `u32>f32` as keywords. It uses `param K u32` and `shared smem_reduce 256 f32` as declarations. None of these appear in the grammar.

Now read `kernels.ls`. It uses a stack-oriented postfix notation where bare names push to the stack and bare operators pop-and-apply. `sigmoid x :` followed by `0`, `- x`, `e^`, `+ 1`, `reciprocal`. That is a third language -- concatenative, stack-based, implicit operands.

These three languages need to become one language. Right now an implementer cannot write a parser because the grammar in `language-primitives.md` does not describe the language that `inference/*.ls` actually uses. The grammar document says bindings have no `=`. The real code uses `=` on every line.

The compiler's token list (compiler.ls section 0) reveals the truth: there are 84 token types including `TOK_EQ`, `TOK_LBRACK`, `TOK_RBRACK`, brackets for array indexing, comparison operators, bitwise operators. The grammar document mentions none of this.

**The spec must describe the language the compiler actually parses.** Everything else is aspirational, not specification.

### 2. No formal grammar

There is no BNF, no PEG, no production rules. The "Complete Grammar" section is a categorized list of symbols with English descriptions. That is a cheat sheet, not a grammar.

An implementer needs to know:

- Is `acc = 0.0` a binding or an assignment? The spec says "no `=`" but the real code uses it everywhere.
- Is `output [ i ] = a [ i ] + b [ i ]` parsed as `store(output[i], add(load(a[i]), load(b[i])))`? Or is `[ ]` syntactic sugar for `->` / `<-`? What are the precedence rules?
- Is `fma acc dq0 xval0 acc` a composition call with four arguments? A keyword with three operands? How does the parser know?
- What is the difference between `each i` and `stride i hidden_dim` and `for k_p tid_local K_packed 256`? The grammar lists all three but does not define their scope rules. Where does the body of a `for` end? At dedent? At a blank line?

Write the grammar. Ten lines of BNF would eliminate these questions.

### 3. Types are present but undeclared

The grammar says nothing about types. The compiler has token types for `f32`, `u32`, `s32`, `f16`, `ptr`, `void`. The `param` declaration takes a type: `param K u32`. The `shared` declaration takes a type: `shared smem_reduce 256 f32`.

But compositions have no type annotations on their arguments. `rmsnorm x weight y :` -- what are the types of `x`, `weight`, `y`? Pointers? The compiler must know to emit `LDG` vs `FADD`. How does it know?

If types are inferred, say so and describe the inference rules. If types are implicit (everything is f32 unless declared otherwise), say so. Silence on this question is the worst option because every implementer will guess differently.

### 4. Memory model is undefined

The language has `->` (load) and `<-` (store) with widths. But `inference/*.ls` uses `x [ i ]` for loads and `output [ i ] = result` for stores. These are two different notations for the same operation. Which is canonical? Does `[ ]` desugar to arrow notation or is it a separate construct?

More importantly: what memory spaces exist? The GPU has global memory, shared memory, registers, constant memory, and local memory. The language has `shared` declarations but no `global` or `local`. Are composition arguments always in global memory? Are locals always in registers? What about spills?

The ARM64 target has a flat address space. The GPU target has five address spaces. The language must say which space each name lives in, or describe the rules by which the compiler decides.

### 5. The composition/function distinction is not enforced

The spec says: "No functions. No call/return. Compositions -- named sequences that the compiler flattens into instruction streams."

But `kernels.ls` has `DeltaNet_Inference :` calling `Embed`, `Layer * 64`, `RMSNorm`, `LMHead`, `Sample`. `Layer * 64` means "repeat Layer 64 times." That is not flattening -- that is a loop. Does the compiler unroll 64 copies of the layer body? If so, the megakernel cubin will be enormous (each DeltaNet layer is 41 primitives, each primitive emits multiple SASS instructions, times 64 layers). GAPS.md says the cubin buffer is 512KB but a 64-layer megakernel needs ~300MB. That is a real problem the spec ignores.

If compositions are truly flattened, the spec must address code size. If they are sometimes calls, the spec must say when.

### 6. Control flow scope rules

`for`, `each`, `stride`, `if==`, `if>=`, `if<` all introduce scope. How does that scope end? By indentation (Python-style)? By a closing keyword? By end-of-line?

In `elementwise.ls`, `each i` is followed by an indented body. In `gemv.ls`, `for k_p tid_local K_packed 256` is followed by an indented body that spans 60+ lines. In `reduce.ls`, `if== lane_id 0` is followed by an indented body, and `else` starts a new block.

So scope is by indentation. But the grammar does not say this. The compiler's parser must have indentation tracking (the token list includes `TOK_INDENT`). Document it. Say: "Bodies are indented. Scope ends at dedent." Simple rule, but it must be stated.

### 7. The register model has two dialects

`language-primitives.md` describes registers as `$N` (numbered) and `$NAME` (dictionary lookup). `compiler.ls` uses `%r0`, `%r1`, `%r2` in comments and presumably in emission. The inference files use bare names like `tid`, `lane_id`, `warp_id` that are clearly register-allocated.

What does the programmer write? `$0`? `%r0`? `tid`? All three appear in different documents. If the language has named bindings that the compiler maps to registers, then `$N` is a backend detail that should not appear in the language spec. If `$N` is user-facing (for syscall setup), that is a different semantic than named bindings.

### 8. Intrinsics vs. compositions: the honest primitive set

VOCABULARY.md says the language has 12 irreducible families. But the real `.ls` files use `fma`, `neg`, `rcp`, `rsqrt`, `shr`, `shl`, `and`, `or`, `xor`, `u32>f32`, `shfl.down`, `barrier` as primitive operations that are not compositions and cannot be decomposed further.

The honest primitive set is not 12 families. It is closer to 30 operations:
- Arithmetic: `+`, `-`, `*`, `/`, `fma`
- Bitwise: `and`, `or`, `xor`, `shr`, `shl`
- Conversion: `u32>f32`, `f32>f16`, etc.
- MUFU: `exp` (2^), `log` (log2), `sqrt`, `rcp`, `rsqrt`, `sin`, `cos`
- Warp: `shfl.down`, `shfl.bfly`
- Sync: `barrier`
- Memory: load, store (with width), shared load/store
- System: `trap` (syscall on ARM64)

That is still a small number. Thirty operations is fine. But say thirty, not twelve. The twelve-family story is elegant but the inference kernels use thirty.

---

## Ambiguities an Implementer Would Hit

### A. Binding vs. assignment vs. store

Three patterns appear in real `.ls` code:
1. `acc = 0.0` -- assignment to a local
2. `output [ i ] = result` -- store to memory via array syntax
3. `partial_ss = partial_ss + t0` -- mutation of an existing binding

Are these three operations? Two? One? Does the `=` in pattern 1 create a new binding or mutate an existing one? In a language with "no `=`" these all need explanation.

### B. When `=` disappears

The spec says bindings are `name expr` with no `=`. The real code uses `=` everywhere. PLAN.md says the compiler is being rewritten to remove all `=`. If `=` is going away, what replaces `output [ i ] = result`? The arrow `<- 32 output i result`? That changes every line of every inference kernel.

This is a language design decision that is unresolved. The spec should not describe a syntax that does not exist yet and will require rewriting 800+ lines of kernel source.

### C. Implicit thread model

`each i` parallelizes across GPU threads. But `i` is never declared and its range is never specified. In `residual_add`, `each i` iterates over... what? The length of `a`? The length of `b`? The length of `output`? All three? What if they differ?

The compiler derives thread indexing from "vector operands" per PRIMITIVES.md. But the rules for this derivation are not stated. Does the compiler look at the first array access inside the `each` body? The composition parameters? The `param` declarations?

### D. Stride loop mechanics

`stride i hidden_dim` appears in `reduce.ls`. This is different from `each i` (which is thread-parallel) and `for i start end step` (which is sequential). The difference matters enormously for GPU code generation. `stride` means each thread handles elements `tid, tid+blockDim, tid+2*blockDim, ...` -- but this is not stated anywhere in the grammar.

### E. Composition parameter passing

`rmsnorm x weight y :` takes three parameters. `residual_add a b output :` takes three parameters. When composed: `rmsnorm x w_norm D` appears in `language-primitives.md` examples -- is `D` passed as a runtime value or a compile-time constant? The `param` declaration exists but is separate from composition arguments. The spec needs to distinguish between runtime parameters (pointers, values on the GPU) and compile-time parameters (dimensions, constants known at parse time).

---

## What a Comprehensive Spec Should Clarify

1. **One grammar, formally stated.** BNF or PEG. Cover everything the parser accepts: assignments, array indexing, declarations, control flow, compositions, Unicode operators. Ten to twenty production rules.

2. **Scope rules.** Indentation-based scope with explicit rules for nesting depth, blank lines, and multi-line expressions.

3. **Type system.** Even if minimal: "Everything is f32 unless declared otherwise. `param` introduces typed compile-time constants. `shared` introduces typed shared-memory arrays. Composition arguments are untyped pointers to global memory." Something. Anything.

4. **Memory model.** Which names live in which memory spaces. How the compiler decides. How the programmer overrides (if they can).

5. **Thread model.** What `each` does. What `stride` does. How the compiler derives grid and block dimensions from composition arguments.

6. **Composition semantics.** Are they always inlined? What about `Layer * 64`? What is the code size model? When does the compiler emit a call instead of inlining?

7. **The real primitive set.** List every operation the compiler recognizes as a keyword. Not the conceptual twelve -- the actual thirty. Include `fma`, `neg`, `rcp`, `shr`, `and`, `shfl.down`, `barrier`, `u32>f32`, and everything else that appears in `inference/*.ls`.

8. **The binding model.** Resolve the `=` question. Either bindings use `=` (and the grammar document is wrong) or they do not (and 800 lines of kernel source must be rewritten). Pick one and document it.

9. **Error model.** GAPS.md gets this right: "Lithos has no error handling as a language feature, and this is correct." But the spec should say this explicitly rather than leaving it as an implicit absence.

10. **Architecture dictionaries.** The grammar mentions `arch/hopper.dict` and `arch/arm64.dict` but does not specify their format. One line of BNF: `entry = NAME OPCODE HEX_ID`.

---

## The Deeper Question

This language wants to be two things. The grammar document describes a Forth-like concatenative language with Unicode operators and stack semantics. The inference kernels are written in a register-transfer language with named variables, array indexing, C-like assignment syntax, and explicit control flow.

Both are valid choices. The concatenative style (`kernels.ls`, `derivatives.ls`) is beautiful for expressing mathematical compositions. Sigmoid is five lines. RMSNorm is six lines. You can see the whole DeltaNet layer in 41 steps. That is the language you show to a mathematician.

The register-transfer style (`inference/gemv.ls`, `inference/reduce.ls`) is what you need when you are hand-scheduling a W4A16 dequant loop with 8-way nibble extraction and warp shuffle reductions. That is the language you show to a GPU programmer.

I have spent my career arguing that one language can serve both. Forth does. But Forth achieves this by being genuinely stack-based -- there are no named variables, no array syntax, no assignment. Everything goes through the stack. The stack is the abstraction that unifies the mathematical view and the machine view.

Lithos is not stack-based. It has named bindings, and the inference kernels use them on every line. That is fine -- I am not saying it must be Forth. But it must be honest about what it is. Right now the spec describes a stack language and the code is a register language. The compiler parses the register language. The spec should describe the register language.

If the concatenative notation in `kernels.ls` and `VOCABULARY.md` is a high-level view that desugars into the register-transfer notation of `inference/*.ls`, say so. Define the desugaring. Make `kernels.ls` a macro layer over `inference/*.ls`. That would be clean.

If the concatenative notation is the real language and the register-transfer notation is temporary scaffolding that will be rewritten, say so and do the rewriting.

What you cannot do is ship a spec that describes language A while the compiler implements language B. That is not a specification. That is a wish.

---

## Summary

The design instinct is excellent. The shipping vision is correct. The dimensional notation is genuinely good. The megakernel architecture is the right choice. The libcuda replacement is bold and justified.

But the specification is not a specification. It is three documents describing three overlapping but incompatible views of the same language. An implementer would be blocked on line one of the parser because the grammar does not match the source files.

Write the real grammar. Resolve the `=` question. List the real primitives. Define scope rules. State the type discipline. Describe the thread model. One document, complete, no contradictions.

The language is small enough to specify completely in a few pages. That is its greatest strength. Do not waste it by leaving the spec in three inconsistent pieces.

---

*If you cannot hold the whole thing in your head, it is too complicated. This language is small enough. The spec should be too.*
