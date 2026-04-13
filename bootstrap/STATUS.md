# Lithos Bootstrap Status

Date: 2026-04-13

## 1. What Works

The bootstrap is an 8-file ARM64 assembly compiler that reads `.ls` source
files and emits static ARM64 ELF binaries or GPU cubins. It assembles and
links via `build.sh`. The binary also embeds a DTC Forth interpreter for
legacy `.fs` files.

### Lexer (lithos-lexer.s, 909 lines)
Fully implemented. Tokenizes `.ls` source into (type, offset, length) triples.

- Integer literals (decimal, hex with 0x prefix, negative)
- Float literals (decimal point detection)
- Identifiers (including `$`-prefixed register names like `$8`)
- UTF-8 multi-byte tokens: `→` (load), `←` (store), `↑` (reg read), `↓` (reg write)
- UTF-8 math: `Σ` (sum), `△` (max), `▽` (min), `√` (sqrt), `≅` (sin), `≡` (cos)
- All single-char operators: `+ - * / = < > & | ^ # ( ) [ ] : , .`
- Two-char operators: `== != <= >= << >> ** ***` and compound conditionals `if== if>= if< if!=`
- 30+ keywords: if, else, elif, for, while, each, stride, return, var, buf, const, kernel, host, param, shared, weight, label, exit, etc.
- Comment syntax: `\\` to end of line
- Indentation tracking (TOK_INDENT with count)

### Emitter (emit-arm64.s, 2833 lines)
Comprehensive ARM64 instruction encoder. 100+ DTC words covering:

- Arithmetic: ADD, SUB, MUL, SDIV, UDIV, MADD, MSUB (register and immediate)
- Logic: AND, ORR, EOR, ANDS (register and immediate)
- Shift: LSL, LSR, ASR (register and immediate)
- Compare: CMP, CMN, TST (register and immediate)
- Move: MOV, MOVZ, MOVK, MOVN, MVN, NEG, 64-bit immediate load
- Memory: LDR, STR, LDRB, STRB, LDRH, STRH (register offset, immediate offset, pre/post index)
- Load/store pair: LDP, STP (immediate, pre, post)
- Branch: B, B.cond (all 14 conditions), BL, BLR, BR, RET, CBZ, CBNZ, TBZ, TBNZ
- Conditional select: CSEL, CSINC, CSET, CSNEG
- System: SVC, NOP, BRK, MRS, MSR, DSB, DMB, ISB
- Address: ADR, ADRP
- Array: ARRAY-LOAD, ARRAY-STORE (combined offset + load/store)
- Forward branch patch helpers: MARK, PATCH-BCOND, PATCH-B, PATCH-CBZ, PATCH-CBNZ

### ELF Writer (lithos-elf-writer.s, 2026 lines)
Two complete ELF64 writers:

- ARM64 Linux executable: ET_EXEC, EM_AARCH64, PT_LOAD at 0x400000, .text + .data + .bss segments
- GPU cubin: ET_EXEC, EM_CUDA, 9-section layout (.shstrtab, .strtab, .symtab, .nv.info, .nv.info.<kernel>, .text.<kernel>, .nv.constant0.<kernel>, .nv.shared)
- File I/O via Linux syscalls (openat, write, close)
- Cubin nv.info attribute emission (REGCOUNT, FRAME_SIZE, PARAM_CBANK, etc.)

### Parser (lithos-parser.s, ~2961 lines stable)
Recursive-descent, single-pass, no AST. Implemented features:

| Feature | Status | Notes |
|---|---|---|
| Compositions (`name args :`) | **Works** | Emits STP/LDP prologue/epilogue, params in X0-X7 |
| Integer literals | **Works** | Decimal and hex, up to 64-bit via MOVZ/MOVK |
| Assignment (`x = expr`) | **Works** | Allocates register, adds to symbol table |
| Arithmetic (`+ - * /`) | **Works** | Recursive descent with precedence |
| Comparisons (`== != < > <= >=`) | **Works** | CMP + CSET emission |
| Bitwise (`& \| ^`) | **Works** | AND, ORR, EOR |
| Shifts (`<< >>`) | **Works** | LSLV, LSRV |
| Register write (`↓ $N val`) | **Works** | Emits MOV to target register |
| Register read (`↑ $N`) | **Works** | Emits MOV from source register |
| `trap` (SVC #0) | **Works** | Recognized as identifier text match, emits SVC #0 |
| Memory load (`→ width addr`) | **Works** | Emits LDR; width ignored (always 64-bit) |
| Memory store (`← width addr val`) | **Works** | Emits STR; width ignored (always 64-bit) |
| `if` / `elif` / `else` | **Works** | CBZ + forward branch patching |
| `while` | **Works** | Loop top + CBZ exit + backward branch |
| `for i start end step` | **Works** | Init + CMP + B.GE + body + ADD step + B loop |
| `each i` | **Stub** | Allocates register, no GPU thread mapping |
| `var` declarations | **Works** | Register-allocated variables |
| `buf` declarations | **Works** | Buffer allocation |
| `const` declarations | **Works** | Stores value in symbol table |
| `label` | **Works** | Records code address in symbol table |
| `return` | **Works** | Emits epilogue + RET |
| Composition calls | **Works** | Symbol lookup + BL to code offset |
| Parenthesized expressions | **Partial** | `(expr)` for grouping |

### Expression Compiler (lithos-expr.s, 2765 lines)
Separate DTC-threaded expression compiler (legacy path, parallel to parser):

- Operator precedence: `| & == != < > <= >= + - * / % >> <<`
- Atom types: integer, float, identifier, load, store, reg read/write, unary math
- Dual-target: ARM64 and SASS emission paths
- Register allocator: linear allocation with high-water mark
- Symbol table: find, add, get-register operations

### Driver (driver.s, 744 lines)
Complete CLI driver:

- File extension dispatch: `.ls` → Lithos pipeline, `.fs` → Forth interpreter
- Argument parsing: `--target {arm64|gpu}`, `-o output`, multiple source files
- Source file concatenation via mmap (16 MB buffer)
- 5-stage pipeline: parse argv → mmap sources → lex → parse+emit → write ELF
- DTC trampoline (`drv_run_xt_list`) for calling Forth words from native code

### Shared State (ls-shared.s, 102 lines)
Canonical shared buffers:

- `ls_code_buf`: 1 MB code emission buffer
- `ls_token_buf`: 1 MB token buffer (87,381 triples)
- `ls_sym_table`: 24 KB symbol table (512 entries x 48 bytes)
- `ls_comp_table`: 3 KB composition table (64 entries)
- `ls_elf_buf`: 512 KB ELF output buffer

### DTC Bootstrap (lithos-bootstrap.s, 5334 lines)
87-word DTC Forth runtime: stack ops, arithmetic, memory, I/O, dictionary,
string ops, outer interpreter. Provides the runtime substrate for the
DTC-threaded components (lexer, expr, emitter, ELF writer).


## 2. What Doesn't Work Yet

| Feature | Status | Details |
|---|---|---|
| Width-specific memory ops | **Missing** | `→ 8/16/32 addr` always emits 64-bit LDR/STR; no LDRB/LDRH/LDR-W dispatch |
| String literals | **Missing** | No string token type in lexer, no `.data` section emission |
| `data` sections | **Missing** | Parser references TOK_DATA but it's undefined (not in lexer or parser token enums) |
| `trap` as keyword token | **Mismatch** | Parser defines TOK_TRAP=89 but lexer doesn't emit it; works via string match in atom parser only |
| `host` / `kernel` prefixed compositions | **Stub** | Parsed at top level but handler functions not implemented |
| `shared` / `weight` / `param` declarations | **Stub** | Dispatch exists, handler functions called but unverified |
| GPU/SASS target | **Scaffolded** | Expr compiler has SASS emission stubs; parser has no SASS path |
| `each i` (GPU parallel) | **Stub** | Allocates register, no actual thread-parallel codegen |
| `stride` loop (GPU) | **Missing** | Token defined, no parser handler |
| Dimensional ops (`** ***`) | **Missing** | Tokens defined, no codegen |
| Reductions (`Σ △ ▽ #`) | **Stub** | Parsed, operand passed through without reduction codegen |
| Unary math (`√ ≅ ≡`) | **Stub** | Parsed, operand passed through without NEON/FP codegen |
| Float arithmetic | **Missing** | Float tokens parsed but treated as integer bits |
| `goto` / `continue` | **Missing** | Token defined (TOK_GOTO=92), no parser handler |
| Named special registers (`$NAME`) | **Missing** | `$`-prefixed idents parsed but no architecture dictionary lookup |
| Nested composition scoping | **Untested** | Scope push/pop exists but may have bugs with deep nesting |


## 3. Build System

### Build command
```
cd bootstrap && ./build.sh
```
Or: `make` (thin wrapper around build.sh).

### Build process
1. Generate DTC macro header (`_macros.h.s`)
2. Patch `lithos-bootstrap.s`: rename `_start` → `bootstrap_init`
3. Auto-generate `.global` directives for cross-file symbols
4. Assemble each `.s` file with optional macro prelude
5. Duplicate-symbol audit via `nm`
6. Link with `ld --warn-common --no-undefined`

### Link order
```
driver.o → lithos-bootstrap.o → lithos-lexer.o → lithos-parser.o →
lithos-expr.o → emit-arm64.o → lithos-elf-writer.o → ls-shared.o
```

### File list (8 source files)
| File | Lines | Size | Role |
|---|---|---|---|
| lithos-bootstrap.s | 5334 | 112 KB | DTC Forth runtime (87 words) |
| lithos-parser.s | ~2961 | ~77 KB | Recursive-descent parser + codegen |
| emit-arm64.s | 2833 | 68 KB | ARM64 instruction encoder (100+ words) |
| lithos-expr.s | 2765 | 77 KB | Expression compiler (DTC, dual-target) |
| lithos-elf-writer.s | 2026 | 56 KB | ELF64 writer (ARM64 + cubin) |
| lithos-lexer.s | 909 | 24 KB | Tokenizer (UTF-8 aware) |
| driver.s | 744 | 23 KB | CLI driver + pipeline orchestration |
| ls-shared.s | 102 | 4 KB | Shared buffers and cursors |
| **Total** | ~17,674 | ~441 KB | |

### Binary
- Output: `lithos-bootstrap` (ARM64 static ELF)
- Size: ~120 KB
- Dependencies: none (statically linked, Linux syscalls only)

### Current build status
**BROKEN** — `lithos-parser.s` is being actively modified by concurrent workers.
Link errors reference undefined symbols (`emit_ldr_x_zero`, `emit_ldrb_zero`,
`emit_str_x_zero`, etc.) added by in-progress work. The `.bak` files contain
the last known-good versions (2952 lines for parser, 938 for lexer).


## 4. Architecture

```
                     ┌─────────────┐
                     │  driver.s   │  CLI entry, argv parsing,
                     │  (_start)   │  pipeline orchestration
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌────────────┐ ┌─────────────┐ ┌────────────────┐
     │ lithos-    │ │ lithos-     │ │ lithos-        │
     │ lexer.s    │ │ parser.s    │ │ expr.s         │
     │            │ │             │ │                │
     │ Tokenize   │ │ Parse +     │ │ Expression     │
     │ .ls source │ │ emit ARM64  │ │ compiler (DTC) │
     └─────┬──────┘ └──────┬──────┘ └───────┬────────┘
           │               │                │
           │               ▼                ▼
           │        ┌─────────────┐  ┌──────────────┐
           │        │ emit-arm64.s│  │ (SASS path   │
           │        │             │  │  scaffolded)  │
           │        │ ARM64 instr │  └──────────────┘
           │        │ encoder     │
           │        └──────┬──────┘
           │               │
           │               ▼
           │        ┌──────────────────┐
           │        │ lithos-elf-      │
           │        │ writer.s         │
           │        │ ELF64 output     │
           │        │ (ARM64 + cubin)  │
           │        └──────────────────┘
           │
      ┌────┴──────────────────────────────┐
      │ ls-shared.s                       │
      │ Shared buffers: code, tokens,     │
      │ symbols, compositions, ELF output │
      └───────────────────────────────────┘
           │
      ┌────┴──────────────────────────────┐
      │ lithos-bootstrap.s                │
      │ DTC Forth runtime: stacks, dict,  │
      │ I/O, outer interpreter (87 words) │
      └───────────────────────────────────┘
```

### Data flow
1. **driver.s**: mmaps source files, concatenates into single buffer
2. **lithos-lexer.s**: scans buffer → token triples in `ls_token_buf`
3. **lithos-parser.s**: walks tokens, emits ARM64 instructions into `ls_code_buf`
4. **emit-arm64.s**: provides instruction encoding primitives called by parser
5. **lithos-elf-writer.s**: wraps `ls_code_buf` contents into ELF64 → output file

### Threading model
- **Parser**: native ARM64 subroutine calls (bl/ret) — can recurse
- **Lexer, emitter, ELF writer**: DTC Forth threading (NEXT macro, X26=IP)
- **Bridge**: `drv_run_xt_list` trampoline converts native→DTC and back

### Register conventions
```
X26 = IP    (DTC instruction pointer)
X25 = W     (DTC working register)
X24 = DSP   (data stack pointer, full descending)
X23 = RSP   (return stack pointer)
X22 = TOS   (top of stack, cached)
X20 = HERE  (dictionary pointer)
X21 = BASE  (number base, default 16)
X19 = TOKP  (parser: current token pointer)
X27 = TOKEND (parser: past-last-token pointer)
X28 = SRC   (parser: source buffer pointer)
```


## 5. Known Bugs

1. **Build broken**: `lithos-parser.s` references undefined symbols from
   in-progress width-specific memory operations. Restore from `.bak` to get
   a linking build.

2. **Width ignored in memory ops**: `→ 8 addr` and `→ 64 addr` both emit
   64-bit LDR. The width argument is parsed but not used to select
   LDRB/LDRH/LDR-W/LDR-X.

3. **TOK_TRAP/lexer mismatch**: Parser defines TOK_TRAP=89 and checks for it
   in `parse_statement`, but the lexer never emits this token (it emits
   TOK_IDENT for "trap"). The atom parser works around this via string
   comparison, but the statement-level check is dead code.

4. **Float literals treated as integers**: `TOK_FLOAT` tokens go through
   `parse_int_literal` which skips the decimal point. No NEON/FP codegen.

5. **Unary math ops are pass-through stubs**: `√`, `≅`, `≡` parse their
   operand but emit nothing — result is just the operand register.

6. **Reductions are pass-through stubs**: `Σ`, `△`, `▽` parse their operand
   but do not emit loop-based or warp-based reduction code.

7. **`each` is a stub**: Allocates a register for the loop variable but does
   not emit any loop or thread-index code.

8. **Register allocator is linear-only**: Registers are allocated X9..X15
   (7 total) with no spilling. Complex expressions exhaust registers and
   trigger `parse_error_regspill`.

9. **No `$NAME` dictionary lookup**: Named special registers (like `$CNTVCT_EL0`)
   are not recognized — only numeric `$N` works via the identifier→integer path.

10. **Composition call result convention**: `parse_comp_call` always returns X0
    regardless of what the called composition actually produces.


## 6. Next Steps for Self-Hosting (S1-S4)

### S1. Update .li compiler files to new grammar
The `.li` compiler files (compiler/lithos-lexer.li, compiler/emit-arm64.li,
compiler/lithos-safetensors.li, compiler/lithos-elf.li) need to be updated
from old syntax (fn, ->) to new symbol set (→ ← ↑ ↓ $ and composition syntax).

**Prerequisite**: The bootstrap must be able to compile basic `.ls` programs
end-to-end first. Current blocker: width-specific memory ops and the
build-breaking undefined symbols.

### S2. Write the parser in .li — lithos-parser.li
Recursive descent, single-pass, direct emission. Reads token stream from
lexer, emits ARM64 or GPU machine code. Dual backend dispatch.

**Dependency**: S1 complete, bootstrap compiles basic programs.

### S3. Write the sm90 emitter in .li
Port gpu/emit.fs (now deleted) to .li. Each Forth word becomes a host
composition that writes 16 bytes (instruction + control word) to a code
buffer. Opcode constants from gpu/opcodes-sm90.fs (deleted) need recreation
from the probe-verified encodings.

**Risk**: Forth GPU emitter was deleted in commit 17d5103. Opcode data
must be recovered from git history or re-probed.

### S4. Bootstrap compile
```
forth-bootstrap lithos.fs compiler.li → lithos-stage1
./lithos-stage1 compiler.li → lithos
./lithos compiler.li → lithos-verify
diff lithos lithos-verify  # must match (fixed point)
```

**Distance**: S4 requires S1-S3 complete. The bootstrap compiler needs:
- Working memory ops with width selection (8/16/32/64)
- String literals and data sections
- Float support (at least for constant embedding)
- Composition calls working end-to-end with correct ABI

### Immediate priorities
1. Fix the build: resolve undefined `emit_ldr_x_zero` etc. or revert parser to `.bak`
2. Add width dispatch to memory load/store (LDRB/LDRH/LDR-W/LDR-X)
3. Add "trap" to lexer keyword table (eliminate string-match workaround)
4. Test basic `.ls` programs end-to-end: compile + run + verify exit code
5. Add string literals and `.data` section support
