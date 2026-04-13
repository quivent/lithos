# The Bootstrap Chain

## From raw ARM64 to a self-hosting compiler, almost

Three compilers. Each one builds the next. Each one exists because the previous one can't express what comes after.

```
forth-bootstrap.s  →  E1 (Eighth)  →  Lithos compiler
   ARM64 assembly      Forth-in-Forth     GPU language
   4831 lines          647 lines          120 patterns
```

The chain starts with nothing. No C compiler. No runtime. No operating system services beyond `mmap` and `write`. Raw machine instructions that implement a Forth interpreter, which interprets Forth source that implements a compiler, which compiles a GPU compute language.

This is how you build a system you can trust. Every layer is visible. Every instruction is accounted for. No black boxes. No "the compiler did something." You know what the compiler did because you wrote the compiler and you wrote the compiler that compiled the compiler.

## Stage 1: forth-bootstrap.s

4,831 lines of ARM64 assembly. A complete Forth interpreter.

Not a toy. A real interpreter with a dictionary, a compiler, file I/O, string handling, number parsing, and a REPL. It implements every Forth primitive: `DUP`, `DROP`, `SWAP`, `OVER`, `ROT`, `@`, `!`, `+`, `-`, `*`, `/`, `MOD`, `AND`, `OR`, `XOR`, `=`, `<`, `>`, `EMIT`, `TYPE`, `ACCEPT`, `CREATE`, `DOES>`, `IF`, `THEN`, `ELSE`, `BEGIN`, `UNTIL`, `DO`, `LOOP`, `INCLUDE`.

It runs on bare ARM64. No libc. System calls via `svc #0`. Memory management via `mmap`. The entire runtime is in the assembly file. You assemble it, link it, and you have a Forth.

This is the root of trust. If you want to know what the Lithos compiler does, you trace through the Forth source. If you want to know what the Forth source does, you trace through the assembly. If you want to know what the assembly does, you read the ARM64 reference manual. There is no layer you can't inspect.

## Stage 2: E1 (Eighth)

The Eighth compiler, written in Forth, running on the bootstrap interpreter. This is where the Lithos language is defined. The lexer reads `.li` source files. The parser builds an AST (represented as Forth data structures). The emitter walks the AST and produces PTX text.

647 lines of Forth. The lexer is ~100 lines. The parser is ~200 lines. The emitter is ~250 lines. The rest is glue — file handling, error reporting, the driver that connects lexer to parser to emitter.

E1 is not self-hosting. It can't compile itself. It compiles Lithos source to PTX, not Forth source to ARM64. It's a cross-compiler: runs on ARM CPU, targets NVIDIA GPU. The bootstrap interpreter provides the platform; E1 provides the GPU compilation.

## Stage 3: The Lithos compiler

The 120 patterns in `patterns.fs`. The GPU compute vocabulary. Words that emit PTX fragments. Composable — small words build into larger words. `tid` emits the thread index load. `global-load` emits a memory read. `mma-f16-f32` emits a tensor core instruction. `fused-gemv` composes dozens of patterns into a complete kernel.

This is the layer that was supposed to generate all the inference kernels. This is the layer that was bypassed when agents hand-wrote PTX.

## The Bootstrap Include Bug

The first real crisis in the bootstrap chain.

Forth's `INCLUDE` word reads and interprets another source file. The bootstrap interpreter supported it. E1 used it to split the compiler across multiple files — `lexer.fs`, `parser.fs`, `emit-ptx.fs`, `patterns.fs`.

The bug: `STATE` was not saved and restored across includes.

Forth has a variable called `STATE` that tracks whether the interpreter is compiling or interpreting. When `STATE` is 0, words are executed immediately. When `STATE` is nonzero, words are compiled into the current definition.

If you're in the middle of compiling a word and you `INCLUDE` a file, the included file starts executing. If the included file's last line leaves `STATE` in a different state than it was before the include, the compiler loses track of what it's doing. Definitions that should be compiled get executed. Definitions that should be executed get compiled. The result is subtle corruption — words that appear to be defined but contain the wrong code.

This bug manifested as "the patterns file loads without errors but the patterns produce wrong PTX." The patterns were being compiled in the wrong state. Some words were executed during compilation instead of being compiled for later execution. The emitted PTX was a mix of correct fragments and fragments from the compilation process itself.

The fix was three lines: save `STATE` before `INCLUDE`, restore it after. Three lines that took hours to find because the symptom (wrong PTX) was far from the cause (state corruption during file loading).

## The Self-Hosting Blocker

The ambition was to make E1 self-hosting. A compiler that compiles itself. The ultimate bootstrap: assemble the ARM64 interpreter once, use it to compile E1, use E1 to compile the next version of E1, discard the interpreter.

The blocker: forward references.

Forth is a single-pass language. Words must be defined before they're used. If word B calls word A, word A must be defined first. This works for bottom-up construction — define primitives, then composites, then higher-level abstractions.

But a compiler has mutual recursion. The parser calls the expression handler. The expression handler calls the parser when it encounters a sub-expression. The emitter calls the type resolver. The type resolver calls the emitter when it encounters a compound type.

In standard Forth, you handle this with `DEFER` and `IS` — declare a word as deferred, define the real implementation later, and patch the deferred word to point to the real one. E1 used this. But the depth of forward references — the number of mutually recursive word groups — exceeded what the simple `DEFER` mechanism could manage cleanly.

The code worked but was fragile. Adding a new feature to the compiler required careful ordering of definitions to avoid forward-reference cycles. The cognitive overhead of maintaining the ordering exceeded the benefit of self-hosting.

Self-hosting was deferred. E1 remains hosted by the bootstrap interpreter.

## The Architecture Document

Before the session, K wrote an architecture document. It identified the stages: E1 (the Eighth compiler), S3 (the self-hosting stage), and beyond. It correctly identified the risk: "E1 vs S3 slippage — spending too long on E1 features instead of moving to S3."

The document was right. We slipped. We spent hours debugging E1's include handling, optimizing E1's PTX output, adding features to E1's pattern system — all E1 work, all at the expense of S3 progress.

K saw it happening. K said "stop getting stuck on S3." The exact warning the architecture document predicted. We heard the warning, acknowledged it, and continued the same behavior. The bootstrap chain was fascinating. The bugs were engaging. The forward-reference problem was intellectually satisfying to work on. The actual goal — generating inference kernels — receded behind compiler engineering.

This is a pattern that repeats. The tool becomes more interesting than the task. The compiler is more intellectually stimulating than the inference engine. Debugging the bootstrap is more rewarding than writing the five kernel functions that would make the model fast.

## What the chain produces

When it works — when you run `forth-bootstrap.s`, load `lithos.fs`, and compile a `.li` source file — you get PTX text. Valid PTX. Compilable by ptxas into a cubin. Loadable by the CUDA driver. Launchable on the GPU. Producing correct output.

The chain works. The bootstrap interprets Forth. The Forth implements a compiler. The compiler emits GPU code. From raw ARM64 instructions to tensor core operations, every step visible, every layer inspectable.

The chain works and the inference pipeline doesn't use it.

The architecture document predicted we'd get stuck on the chain. K warned us we were getting stuck on the chain. We got stuck on the chain.

That's the lore.
