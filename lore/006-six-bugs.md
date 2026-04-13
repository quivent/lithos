# Six Bugs

## The distance between garbage and "Paris"

The model ran. All 64 layers executed. The GPU computed millions of multiply-accumulates across 448 projections, normalizations, activations, recurrences, and one final vocabulary projection. A number came out. The number was wrong.

Not slightly wrong. Rank 2,723 wrong. The correct token " Paris" was buried under two thousand seven hundred and twenty-two tokens that the model considered more likely. The model was, mathematically speaking, hallucinating with confidence.

Six bugs. Each one trivial. Each one invisible until you knew where to look. Each one the kind of error a human makes when translating a paper into code at 3 AM and an agent makes when translating weight names into math at any hour.

Here they are.

---

## Bug 1: The Missing SiLU

**What:** The DeltaNet block applies a SiLU activation after the causal conv1d. We didn't.

**How it was found:** Layer-by-layer comparison against PyTorch reference. The output after conv1d diverged. Cosine similarity dropped to 0.7 by layer 4.

**Why it existed:** The agent that wrote the DeltaNet pipeline read the architecture description, which says "causal convolution followed by gating." It interpreted "gating" as the multiplicative gate that comes later. The SiLU between conv1d and the rest of the block isn't mentioned in the architecture summary. It's in the code. The agent didn't read the code.

**What it taught us:** Architecture descriptions lie by omission. The code is the specification. Everything else is a summary that forgot something.

---

## Bug 2: Swapped in_proj_a and in_proj_b

**What:** Two weight matrices named `in_proj_a` and `in_proj_b`. One projects to the gating path. The other projects to the value path. We used them backwards.

**How it was found:** Same layer-by-layer comparison. After fixing Bug 1, the next divergence was in the projection outputs. The shapes were right. The values were wrong. Swapping the two matrices fixed it.

**Why it existed:** `in_proj_a` and `in_proj_b` are arbitrary names. There's no convention that says which is which. The agent guessed. It guessed wrong. A 50/50 coin flip that landed tails.

**What it taught us:** Never guess at weight semantics from names. Load the model in PyTorch, print which weight feeds which operation, and use that mapping. Or better: read the source code where the names are assigned to operations. The names mean nothing without the assignment.

---

## Bug 3: Missing L2 Normalization

**What:** The model normalizes Q and K vectors to unit length before computing attention scores. We didn't normalize.

**How it was found:** Attention scores were exploding. By layer 10, the softmax was saturating — one position getting 0.9999, everything else getting 0.0001. The attention was collapsing to a copy operation. Normalizing Q and K fixed it.

**Why it existed:** Standard transformer attention doesn't normalize Q and K. The original "Attention Is All You Need" paper scales by 1/sqrt(d) but doesn't L2-normalize. This model does both. The agent implemented the standard formula. The model uses a non-standard formula.

**What it taught us:** "Standard" is a dangerous word. Every model is its own specification. You can't implement "attention" — you have to implement this model's attention, which means reading this model's code.

---

## Bug 4: Missing 1/sqrt(d) Scaling

**What:** Even after L2 normalization, attention scores need to be scaled by 1/sqrt(head_dimension). We forgot.

**How it was found:** After fixing Bug 3, attention scores were still too large. Not exploding, but biased. Adding the scaling factor brought them into the expected range.

**Why it existed:** This one is embarrassing. Attention scaling is in every transformer tutorial. Every implementation guide. Every paper since 2017. The agent that wrote the attention kernel simply forgot. The line wasn't there. Not wrong — absent.

**What it taught us:** The most well-known operations are the ones you forget to double-check. Nobody reviews the obvious.

---

## Bug 5: RMSNorm (1+w), Not w

**What:** RMSNorm computes `norm(x) * weight`. This model's RMSNorm computes `norm(x) * (1 + weight)`. The learned weights are initialized to zero and stay near zero after training. Multiplying by `weight` means multiplying by approximately zero. Multiplying by `(1 + weight)` means multiplying by approximately one.

**How it was found:** After fixing Bugs 1-4, the output was still wrong. Not garbage anymore — the model was producing plausible tokens, just not the right ones. The layer comparison showed that the RMSNorm output was attenuated by a factor of ~1000x. The weights were all near zero. The `+1` makes the difference between "scale by 0.001" and "scale by 1.001."

**Why it existed:** The agent read the class name `RMSNorm` and implemented the standard formula. The `+1` variant is a design choice specific to this model family. It's one line in the source code: `return x * (1 + self.weight)`. The agent wrote `return x * self.weight`.

**What it taught us:** Same lesson as Bug 3. There is no "standard" implementation. There is only the code. Read the code.

---

## Bug 6: The Zero-Point

This one gets its own section.

### One byte, one constant, the difference between garbage and Paris

GPTQ quantization stores weights as 4-bit integers. To recover the original floating point value, you dequantize: `float_weight = (int4_value - zero_point) * scale`.

The zero-point for this model is 8. The midpoint of a 4-bit range [0, 15]. But Auto-GPTQ, the tool that quantized the model, stores `zero_point - 1` in the file. It stores 7.

We used 7.

Every weight in every layer was dequantized with an offset of negative one. Every one of the 448 matrix multiplications per token. Across all 64 layers. Across all five prompt tokens. The error was small per-weight — a fraction of a scale factor — but it accumulated through 64 layers of computation into total nonsense.

### How it was found

After fixing Bugs 1-5, the model was close. Layer-by-layer comparison showed cosine similarity above 0.99 for the first few layers. But it degraded. By layer 64, the logits were off. " Paris" was in the top 10 but not rank 1.

Someone printed the dequantized weights and compared them to PyTorch's dequantization. There was a systematic bias. Every dequantized value was shifted by exactly one scale unit in the same direction. A constant offset.

The zero-point was read from the model file and used directly. The model file said 7. The correct value is 8. The fix was changing one constant: `0x40E00000` (7.0 as IEEE 754 float) to `0x41000000` (8.0 as IEEE 754 float).

### The moment

Before the fix: " Paris" at rank 2,723. Logit lost in the noise.

After the fix: " Paris" at rank 1. Logit 18.057. The highest-confidence prediction in the vocabulary.

One byte. One constant. One bit flip in the exponent field of a 32-bit float. The entire session — the language, the compiler, the kernels, the quantization research, the documentation, eighteen hours of work — came down to whether a PTX immediate value ended in `E0` or `00`.

### Why it existed

The Auto-GPTQ storage convention is documented nowhere obvious. The code stores `zero_point - 1`. Why? Presumably an implementation detail that made bit-packing easier. The agent that wrote the dequantization kernel read the stored value and used it at face value. Why wouldn't you? If the file says the zero-point is 7, you use 7.

Unless you read Auto-GPTQ's source code and discover that they subtract 1 before storing. Unless you compare your dequantized weights against theirs and notice the systematic offset. Unless you already know this particular quirk of this particular tool.

The agent didn't know. Nobody told it. The documentation didn't mention it. The weight file doesn't flag it. You have to read the quantizer's source code to know that the number in the file isn't the number you should use.

### What it taught us

Don't trust stored values. Validate against the tool that created them. If you're reimplementing dequantization, compare your output to the original tool's output on the same weights before running inference. Not at the end, after 64 layers of accumulated error. At the beginning. On one weight matrix. Element by element.

---

## The aggregate lesson

Six bugs. Each one a failure to read source code. Each one a hand-translation error — math on paper turned into PTX by an agent that made assumptions instead of checking.

1. Assumed the architecture summary was complete. It wasn't.
2. Assumed weight names implied semantics. They didn't.
3. Assumed "attention" meant standard attention. It didn't.
4. Forgot a universally known scaling factor. Nobody checked.
5. Assumed "RMSNorm" meant the standard formula. It didn't.
6. Assumed the stored value was the true value. It wasn't.

The language was designed to prevent exactly this class of error. If the math is the code — if you write `(1 + weight) * norm(x)` and the compiler turns it into PTX — you can see the `+1` in the source. If you write `(nibble - zero_point) * scale` and `zero_point` is loaded from validated metadata — you can't get it wrong by one.

Hand-translating math into PTX is how you get six bugs. Compiling math into PTX is how you get zero.

We had the compiler. We hand-translated anyway. Six bugs later, " Paris" appeared at rank 1, and we moved on.

That's the lore.
