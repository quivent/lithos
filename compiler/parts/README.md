# Minimal compiler.ls parts

Each agent writes ONE file in this directory. Integrator concatenates them
into compiler/compiler.ls.

## Section order in final compiler.ls

1. parts/00-constants.ls    — token type constants, opcode constants
2. parts/01-lex.ls          — lexer
3. parts/02-sass-emit.ls    — SASS opcode emitters (one per primitive)
4. parts/03-arm64-emit.ls   — ARM64 opcode emitters
5. parts/04-cubin-elf.ls    — cubin ELF writer
6. parts/05-arm64-elf.ls    — ARM64 ELF writer
7. parts/06-walker.ls       — token walker that dispatches to emitters
8. parts/07-main.ls         — main entry point

## Conventions

- Use stack-based emit (no register allocator nightmare)
- One composition per Lithos primitive (emit_<name>)
- Globals via `buf NAME_v 8` + `← 64 NAME_v val` / `tmp → 64 NAME_v`
- Use `goto label` for loops, NOT tail recursion
- Use `host` keyword for ARM64-only compositions
