# The Language Nobody Used

## We wrote Lithos and then didn't use it

This is the most absurd part of the entire session.

We designed a language. We debated its syntax for hours — stack-based vs declarative, terse vs readable, Forth-derived vs from-scratch. We wrote a compiler in Forth. We wrote 120 GPU compute patterns. We wrote a SASS emitter that produces binary GPU instructions directly. We wrote example programs that compile and run correctly on the GH200.

Then we hand-wrote 21 PTX kernels by dispatching agents who had never heard of Lithos.

The compiler sat in `/home/ubuntu/lithos/compiler/`. The patterns sat in `/home/ubuntu/lithos/patterns.fs`. The examples sat in `/home/ubuntu/lithos/compiler/examples/`. Valid. Working. Tested.

And the inference pipeline at `/home/ubuntu/lithos/src/generate_paris.py` contains zero Lithos-compiled code. Every kernel was written in raw PTX by an agent that was told "write a PTX kernel for X." Not "compile a Lithos function for X." Not "use the patterns." Not "use the compiler."

## The timeline

**Hour 3:** Lithos language designed. Core vocabulary defined. 120 patterns written covering thread indexing, memory access, arithmetic, warp operations, tensor cores, and inference-specific composites.

**Hour 5:** Lithos compiler written in Forth. Lexer, parser, emitter. Produces valid PTX. Five example programs compile and execute correctly.

**Hour 6-16:** Twenty-one hand-written PTX kernels dispatched to agents. None of them told to use the compiler. None of them import patterns.fs. None of them know the language exists.

**Hour 17:** Someone asks "how many kernels did we dispatch?" The answer is twenty-one. The realization hits.

## Why it happened

The agents were dispatched for speed. "Write a projection kernel" is a concrete task an agent can execute in 15 minutes. "Write a projection kernel using the Lithos compiler" requires the agent to understand the compiler, load the patterns, and work within the language — a harder task that produces the same output (PTX text).

In the rush to get inference working — to see "Paris" come out of the model — the language was bypassed as overhead. The irony is total: the language was built to make kernel generation faster and more correct. Bypassing it made kernel generation slower (21 iterations instead of 5 definitions) and less correct (6 bugs from hand-translating math).

## What the language would have done

If the GPTQ matvec had been written in Lithos:

```
fn projection input weights scales -> output
    each row
        sum = 0
        each group in 0..K/128
            each sub in 0..128/8
                packed = weights[row, group*16 + sub]
                each nib in 0..8
                    w = (packed >> (nib*4)) & 0xF
                    w = (w - 8) * scales[group, row]
                    sum = sum + w * input[group*128 + sub*8 + nib]
        output[row] = sum
```

The compiler would see:
- `each row` → parallelize across thread blocks
- The inner sum → warp-level reduction
- The weight access pattern → dequantization
- The shapes from config.json → tensor core tiling decisions
- The memory access → coalesced loads, shared memory caching

One function. The compiler generates the PTX with tensor cores, tiling, shared memory, dequantization — all from the math.

Bug 6 (zero-point 7 vs 8) would have been: `w = (nibble - ZERO_POINT) * scale` where ZERO_POINT is read from the model's metadata, not hardcoded in a PTX constant. One source of truth. No hand-transcription error.

Bug 5 (RMSNorm 1+w) would have been: `norm(x) * (1 + weight)` in the source. Exactly what the math says. Not a comment that an agent forgot to implement.

## What happens next

Step one: use Lithos. Write the five functions in the language. Compile them. Wire them into the pipeline. Delete the 21 hand-written PTX files.

The compiler works. The patterns work. The SASS emitter works. The machinery exists and is tested. The only thing missing is the decision to use it.

We wrote a language to solve our exact problem and then solved the problem without it. That's the lore.
