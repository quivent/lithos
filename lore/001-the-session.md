# The Session

## April 12-13, 2026

It started at 8:46 PM UTC on a Saturday. K said "install gh cli" and within 18 hours we had built a GPU compute language, an inference engine, a documentation site, reverse-engineered NVIDIA's instruction set, had five dead geniuses design quantization schemes, generated the correct answer to "The capital of France is" from a 27-billion-parameter model running on custom CUDA kernels — and the whole thing ran at 2 tokens per second because nobody used the tensor cores.

This is the story of how that happened.

---

## The Machine

The GH200. NVIDIA's Grace Hopper Superchip. 96 GB of HBM3e at 4 terabytes per second. 72 ARM CPU cores sharing the same memory with the GPU via hardware coherence. 132 streaming multiprocessors, each with 4 tensor cores that can do 4,096 floating point operations per clock cycle.

We measured the bandwidth: 3.59 TB/s. Ninety percent of theoretical peak. The hardware delivers. The question was never "is the GPU fast enough." The question was "can we get out of its way."

vLLM, the state of the art inference server, uses 13% of that bandwidth at batch-1. Thirteen percent. The GPU is idle 87% of the time. Not because the hardware is slow — because the software stack between the user's request and the GPU's execution is seven layers deep, each adding overhead, each inserting a boundary where the GPU has to stop and start again.

K knew this. K had measured it, documented it, and built three separate optimization projects around it before this session started.

## The Thesis

Build an inference engine from scratch. No PyTorch. No cuBLAS. No framework. A language that emits GPU instructions directly. The language compiles in microseconds because it's string concatenation — you're building PTX text, not running a compiler. The kernels are fused — one function per layer, no boundaries, no intermediate writes to HBM. The memory is mmap'd — no cudaMalloc, no copies, the GPU reads from the same address space as the CPU.

On the GH200, this should be trivial. The hardware was designed for it. Unified memory. Coherent cache. Fast interconnect. Everything the framework stack wastes time on — memory allocation, data transfer, launch overhead — is unnecessary on this machine.

The target was 400+ tokens per second. More than 2x vLLM.

## What We Built

In 18 hours:

**The language.** Lithos. A Forth-derived vocabulary of 120 GPU compute patterns. Words that emit PTX instructions. Composable — small words build into larger words. The patterns encode correct memory access (coalesced loads, bank-conflict-free shared memory), correct computation (tensor core tile shapes, warp-level reductions), and correct fusion (multiple operations in one kernel, intermediates in registers).

**The compiler.** Written in Forth, hosted by an ARM64 assembly bootstrap interpreter. Reads .li source files, emits PTX. Five example programs compile and run correctly. 647 lines of Forth. No Python, no Rust, no C in the compilation path.

**The SASS emitter.** We reverse-engineered 47 Hopper opcodes from NVIDIA's undocumented SASS binary format. Built an automated probing tool that maps any NVIDIA GPU architecture. Emitted a cubin directly — no ptxas, no NVIDIA tools — loaded it, launched it, got correct output. Vector-add, 256 elements, zero error.

**The inference engine.** Model loader that mmap's safetensors files in 9 milliseconds (no data moved — just address mapping). Kernel factory that reads config.json and generates 11 specialized cubins with model dimensions baked in. Engine orchestrator that walks all 64 layers of Qwen 3.5-27B. Tokenizer. API server. The whole pipeline.

**The documentation.** 23 pages on Vercel. Architecture diagrams. Performance analysis. Memory hierarchy. SM internals with to-scale diagrams. Engine comparison (8 inference engines analyzed). TensorRT deep dive. Token journey through all 64 layers. Glossary with 72 terms and hover tooltips on every page.

**The quantization.** Five designs from five brilliant minds. Claude Shannon applied rate-distortion theory and reverse water-filling. John von Neumann formalized it as an automaton with Nash equilibrium across compute units. Dave Ferrucci built a Watson-style evidence pipeline. Alan Turing did Bayesian sensitivity analysis under computability constraints. Ada Lovelace decomposed weights into four strata — each mapped to a different GPU compute unit type. All five were implemented, validated against real weights, and proven to fit one transformer layer in 66 MB of on-chip memory. Shannon's hit 0.978 cosine similarity at 3.076 bits per weight.

**The cold start.** 228 milliseconds. vLLM takes 30-120 seconds. That's 130-525x faster. We measured it.

## What Went Wrong

We got the right answer. "The capital of France is" → " Paris". Rank 1. Logit 18.057. Verified layer-by-layer against PyTorch — cosine similarity 1.0 at every one of the 64 layers. Mathematically correct.

At 2 tokens per second.

vLLM does 179.

### The Kernel Joke

We wrote 21 hand-written PTX kernels. Twenty-one. The entire point of the language was to generate kernels. We had 120 patterns that encode correct GPU programming. We had a compiler that produces valid PTX. We had a SASS emitter that can write binary GPU instructions directly.

And every single kernel in the inference pipeline was hand-written by an agent, in raw PTX, without using the language, without using the patterns, without using the compiler.

The most critical kernel — the GPTQ matrix-vector multiply that runs 448 times per token — was written with scalar FMA instructions. One multiply-add per clock per core. The tensor cores sit on the same silicon doing 4,096 multiply-adds per clock. We literally wrote the kernel to ignore the hardware that was designed for this exact operation.

It's like building a car and using the windshield wipers for propulsion while the engine sits cold.

The first agent that wrote the projection kernel was told "correctness first." It produced a correct kernel. Then we optimized it — added tiling, shared memory input caching, vectorized weight loads. Got it down from 35.6 ms to 0.28 ms. A 105x improvement. But 0.28 ms is still scalar FMA. cuBLAS does the same operation in 0.003 ms with tensor cores.

We had a 105x optimization that was still 93x slower than the baseline library we were trying to replace.

### The Bug Hunt

Six bugs stood between us and correct output. Six mathematical errors, each trivial in isolation, each cascading through 64 layers to produce garbage.

1. **Missing SiLU after conv1d.** The model applies a nonlinearity after the causal convolution. We didn't.
2. **Swapped in_proj_a and in_proj_b.** Two weight matrices with generic names. We used them backwards.
3. **Missing L2 normalization.** The model normalizes Q and K vectors before the dot product. We didn't.
4. **Missing 1/√d scaling.** Standard attention scaling. We forgot.
5. **RMSNorm formula: (1+w) not w.** The model's norm multiplies by (1 + learned_weight). We multiplied by just the weight. The weights are initialized to zero. So we were multiplying by ~0 instead of ~1.
6. **GPTQ zero-point: 8, not 7.** Auto-GPTQ stores zero_point - 1. Every dequantization in every layer was off by one.

Bug 6 was the last one. One constant. One byte. 0x40E00000 vs 0x41000000. Seven versus eight. Applied to every one of the 448 matrix multiplications per token, across all five prompt tokens, through all 64 layers. The model went from producing " Paris" at rank 2,723 to rank 1.

These bugs existed because we implemented the DeltaNet recurrence from documentation and weight names instead of from the actual PyTorch source code. We guessed at what `in_proj_a` meant. We assumed the norm formula from the class name. We used the stored zero-point value literally.

The language was supposed to prevent this. If the math IS the code — if `softplus(dt + bias)` appears in the source as those exact symbols — you can't forget the softplus. If `(1 + weight) * norm(x)` is the formula, you can't accidentally write `weight * norm(x)`. The bugs came from translating math into code by hand, which is exactly what the language was designed to eliminate.

### The Quantization Gap

Five geniuses designed quantization schemes to fit one transformer layer exactly into the GPU's on-chip memory. 66 MB. Sub-3-bit encoding. Pipeline-balanced across all four compute unit types. Mathematically rigorous. Shannon proved it was within 0.98 dB of the information-theoretic bound.

We implemented all five. Validated them against real weights. Produced comparison tables.

Then we ran inference on the original GPTQ weights at 90 MB per layer. Nobody requantized the model. Nobody plugged the Shannon scheme into the inference pipeline. The entire hardware-matched quantization design — the flagship research contribution of this session — sat in Python files producing comparison tables while the inference engine streamed 90 MB per layer from HBM on every token.

The agent to requantize with Shannon was dispatched 18 hours into the session. After everything else was done. As an afterthought.

## What It Means

The architecture works. mmap'd weights passed directly to GPU kernels on GH200 — confirmed. Model loads in 9 ms. Cold start in 228 ms. Correct output through all 64 layers. The thesis is proven: you CAN build an inference engine from scratch, on this hardware, with no framework, and get the right answer.

The performance doesn't work yet. 2 tok/s vs 179. The gap is one instruction — the tensor core MMA instead of scalar FMA. Not an architectural problem. Not a design problem. A kernel quality problem. The language exists to solve this. It just wasn't used.

The quantization was designed but not deployed. The language was built but not used for the actual kernels. The SASS emitter works but isn't in the inference path. The fused kernels compile but aren't wired in. The compiler produces valid PTX but the pipeline uses hand-written PTX.

We built every piece. We connected almost none of them.

## The Real Lesson

K said it early in the session: "It's all math." Every kernel is a math function. A matrix-vector multiply is a sum of products. A norm is a square root of a mean of squares. An activation is a sigmoid times the input. These are formulas. They don't care what instruction you use to evaluate them — scalar FMA or tensor core MMA, the math is the same.

The language should BE the math. `output[row] = sum(weight[row, k] * input[k])` should be the entire kernel. The compiler should decide whether to use tensor cores based on the shapes. The programmer should never see PTX, never worry about tiling, never manually manage shared memory.

We got distracted by the machinery and forgot the math.

The tensor cores were there the whole time.
