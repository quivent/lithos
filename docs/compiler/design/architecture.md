# Lithos Self-Hosting Compiler Architecture

## Overview

The self-hosting compiler is a single `.ls` source file (`compiler.ls`) that
compiles to a native ARM64 ELF binary. This binary replaces the Forth
bootstrap (`lithos.fs` + `parser.fs` + `emit-*.fs` + `gpu/emit.fs`)
entirely. It reads `.ls` template files and safetensors weight files, and
emits per-layer sm90 GPU machine code wrapped in ELF cubins, linked into
cooperative megakernels, then boots the GPU and submits work -- all from a
single native process with zero dependencies.

---

## 1. File Structure

One file: `compiler.ls`. Not multiple files with includes.

Rationale: the `.ls` language has no include mechanism today, and adding one
creates complexity (search paths, circular dependencies, build ordering).
A single file is consistent with Lithos philosophy -- the compiler is one
function that transforms input to output.

The file is organized into sections by comment blocks. Logical module
boundaries exist as groups of related functions, but there is no separate
compilation or linking. The entire compiler is one compilation unit.

Estimated structure:

```
compiler.ls
  |-- Section 0: Constants and data tables         (~300 lines)
  |-- Section 1: Syscall wrappers                  (~100 lines)
  |-- Section 2: Bump allocator + string ops        (~80 lines)
  |-- Section 3: Lexer                             (~150 lines)
  |-- Section 4: sm90 opcode tables                (~200 lines)
  |-- Section 5: sm90 instruction emitter          (~800 lines)
  |-- Section 6: Parser + direct emission          (~1200 lines)
  |-- Section 7: Safetensors reader                (~200 lines)
  |-- Section 8: GPU ELF writer                    (~400 lines)
  |-- Section 9: Megakernel linker                 (~300 lines)
  |-- Section 10: GSP boot                         (~600 lines)
  |-- Section 11: GPFIFO + work submission         (~200 lines)
  |-- Section 12: Main / CLI driver                (~100 lines)
  Total: ~4700 lines
```

---

## 2. The Bootstrap Problem

The compiler is written in `.ls`. The compiler compiles `.ls` to ARM64.
Who compiles the compiler?

**Answer: the Forth bootstrap compiles it exactly once.**

```
Phase 1 (bootstrap compile):
  forth-bootstrap lithos.fs compiler.ls --emit arm64 -o lithos-stage1

Phase 2 (self-compile):
  ./lithos-stage1 compiler.ls -o lithos

Phase 3 (verify):
  ./lithos compiler.ls -o lithos-verify
  diff lithos lithos-verify   # must be identical (fixed point)
```

After Phase 2 succeeds, the Forth bootstrap is never needed again. The
`lithos` binary is the compiler, the kernel builder, and the GPU driver
in one executable.

The Forth bootstrap already has the machinery for this:
- `parser.fs` parses `.ls` syntax and emits GPU machine code
- `emit-arm64.fs` has a code buffer and ARM64 instruction writer
- `arm64-wrap.fs` wraps ARM64 code in an ELF executable

What's missing is the ARM64 instruction encoder (currently a stub that emits
`mov x0, #0; ret`). The bootstrap needs a real ARM64 emitter before it can
compile `compiler.ls`. This encoder is ~400 lines of Forth added to
`emit-arm64.fs`, covering the ~30 ARM64 instructions the compiler uses.

### Bootstrap ARM64 instruction set (minimum viable)

```
Arithmetic:  add, sub, madd, mul (aliases)
Logic:       and, orr, eor, lsl, lsr, asr
Memory:      ldr, str, ldrb, strb (immediate offset, register offset)
Branch:      b, b.cond, bl, blr, ret, cbz, cbnz
Compare:     cmp, tst
Move:        mov, movz, movk, movn
System:      svc #0 (Linux syscalls)
```

All operate on 64-bit (X) registers. The compiler itself never uses SIMD,
FP, or 32-bit (W) mode except inside the sm90 emitter where W-register
stores build GPU instruction words byte-by-byte.

---

## 3. Memory Model

The compiler is a batch program. It runs, produces output files, and exits.
This allows the simplest possible memory model.

### 3.1 Layout

```
+---------------------------+ 0x0000_0000_0040_0078  (entry point)
| .text (compiler code)     |  ~200 KB
+---------------------------+
| .rodata (string constants,|  ~50 KB
|  opcode tables, ELF       |
|  templates)               |
+---------------------------+
| BSS / zero-init globals   |  ~4 KB (variables, small arrays)
+---------------------------+

  ... gap ...

+---------------------------+  mmap'd at runtime:
| Heap: bump allocator      |  64 MB (single mmap, PROT_READ|PROT_WRITE)
+---------------------------+

  ... gap ...

+---------------------------+  mmap'd at runtime (per input file):
| Input .ls file            |  mmap'd read-only (MAP_PRIVATE)
+---------------------------+

+---------------------------+  mmap'd at runtime:
| safetensors file          |  mmap'd read-only (MAP_PRIVATE)
+---------------------------+
```

### 3.2 Bump allocator

```
host fn bump_init
    \ mmap 64 MB anonymous region
    heap_base = mmap(0, 0x4000000, PROT_RW, MAP_ANON|MAP_PRIVATE, -1, 0)
    heap_ptr = heap_base

host fn bump_alloc size -> ptr
    ptr = heap_ptr
    heap_ptr = heap_ptr + size
    \ align to 8 bytes
    heap_ptr = (heap_ptr + 7) & ~7
```

No free. No garbage collection. The compiler runs, allocates forward, then
exits. 64 MB is generous -- the actual working set is ~2 MB for a typical
compilation (token arrays, symbol tables, output buffers).

### 3.3 String handling

All strings are `(addr, len)` pairs -- a 64-bit pointer and a 64-bit length,
stored together as a 16-byte struct. There is no null termination internally.
Strings from input files point directly into the mmap'd file buffer (zero
copy). Strings that need construction (error messages, output paths) are
bump-allocated.

This matches the Forth model: `( addr u )` on the stack becomes two
registers or two struct fields.

---

## 4. The Lexer

### 4.1 Input

The `.ls` source file is mmap'd read-only. The lexer walks the byte buffer
with a position cursor, identical to `lexer.fs` but producing a token array
instead of yielding one token at a time.

### 4.2 Token representation

```
struct Token        \ 16 bytes
    u32 type        \ TOKEN_IDENT, TOKEN_NUMBER, TOKEN_FLOAT, TOKEN_STRING,
                    \ TOKEN_OP, TOKEN_NEWLINE, TOKEN_EOF
    u32 offset      \ byte offset into source buffer
    u32 length      \ byte length of token text
    u32 line        \ line number (for error messages)
```

Tokens are bump-allocated as a contiguous array. The parser reads them
sequentially via an index.

### 4.3 Token types

| Type | Examples | Detection |
|------|----------|-----------|
| IDENT | `fn`, `x`, `hidden_dim`, `rmsnorm` | Starts with letter or `_` |
| NUMBER | `0`, `256`, `0x1F` | Starts with digit; hex if `0x` prefix |
| FLOAT | `1.0`, `-8.0`, `1.442695` | Contains `.`, starts with digit or `-` |
| STRING | `"hello"` | Starts with `"` |
| OP | `+`, `-`, `*`, `/`, `=`, `[`, `]`, `->`, `&`, `>>`, `<<` | Punctuation characters |
| NEWLINE | `\n` | Explicit; needed for line-oriented parsing |
| COMMENT | `\ ...` | Backslash to end of line; stripped by lexer |

### 4.4 Tokenization algorithm

Direct port of `lexer.fs` `src-token`:

```
host fn tokenize src src_len -> tokens n_tokens
    pos = 0
    n_tokens = 0
    loop:
        skip whitespace (spaces, tabs; NOT newlines)
        if pos >= src_len: emit TOKEN_EOF, return
        c = src[pos]
        if c == '\n': emit TOKEN_NEWLINE, pos++, continue
        if c == '\\': skip to next '\n', continue
        start = pos
        if c == '"': scan to closing '"', emit TOKEN_STRING
        elif is_punct(c): emit TOKEN_OP (handle 2-char ops: ->, >>, <<, !=, ==, >=, <=)
        else: scan while not whitespace/punct, classify as IDENT/NUMBER/FLOAT
        n_tokens++
```

Newlines are significant because `.ls` uses indentation for scoping (like
Python). The current Forth lexer discards newlines (treating them as
whitespace) and relies on `each`/`endfor` keywords for structure. The
self-hosting compiler preserves newlines to enable proper indentation-based
block detection in the future. For now, the parser still uses explicit
keywords (`fn`, `each`, `stride`, `endfor`, `barrier`) as block delimiters
-- the newline tokens are available but not required for correct parsing.

---

## 5. The Parser

### 5.1 Architecture: single-pass parse-and-emit

The parser does NOT build an AST. It parses and emits in one pass, identical
to the current Forth parser. When parsing a GPU kernel function, each
statement directly calls the sm90 instruction emitter. When parsing a host
function, each statement directly calls the ARM64 emitter.

This is the right design for Lithos:
- `.ls` is a simple language with no forward references within a function
- Expressions are small (no deeply nested trees)
- One-pass keeps memory usage minimal
- It matches the proven Forth implementation

### 5.2 Parser entry point

```
host fn parse_file tokens n_tokens
    i = 0
    while i < n_tokens:
        tok = tokens[i]
        if tok == "fn":
            parse_fn(tokens, &i)        \ GPU kernel function
        elif tok == "host":
            next; expect "fn"
            parse_host_fn(tokens, &i)   \ ARM64 host function
        else:
            error("expected fn or host fn")
```

### 5.3 Function parsing

```
parse_fn:
    name = next_ident()
    params = []
    while peek() != "->":
        params.append(next_ident())
    consume("->")
    outputs = []
    while peek() is ident (not newline):
        outputs.append(next_ident())
    \ Register symbol table entries
    for p in params: sym_add(p, KIND_INPUT_PTR, alloc_rd())
    for o in outputs: sym_add(o, KIND_OUTPUT_PTR, alloc_rd())
    \ Parse body
    parse_body()
    \ Emit EXIT instruction
    emit_exit()
```

### 5.4 Statement dispatch

The body parser recognizes these statement forms:

| Pattern | Action |
|---------|--------|
| `each VAR` | Emit thread-index prologue (S2R tid, ctaid; IMAD global_idx) |
| `stride VAR BOUND` | Emit stride loop setup (bounds check + forward branch) |
| `for VAR START BOUND STEP` | Emit counted loop (mov, isetp, bra) |
| `endfor` / `endstride` | Emit loop back-edge, patch forward branch |
| `IDENT = EXPR` | Parse expression, emit result to register, update sym table |
| `IDENT [ IDX ] = EXPR` | Parse expression, emit STG to array[idx] |
| `if< A B` / `if>= A B` / `if== A B` | Emit ISETP + predicated BRA |
| `barrier` | Emit BAR.SYNC |
| `shared NAME SIZE TYPE` | Declare shared memory, update shmem-size |
| `param NAME TYPE` | Declare scalar parameter, track in param table |
| `shfl.bfly DST SRC DELTA` | Emit SHFL.BFLY instruction |
| `exp DST SRC` | Emit FMUL(log2e) + MUFU.EX2 |
| `rcp DST SRC` | Emit MUFU.RCP |
| `rsqrt DST SRC` | Emit MUFU.RSQ |
| `neg DST SRC` | Emit FMUL(src, -1.0) |

### 5.5 Expression parser

Recursive descent with two levels of precedence (matching `parser.fs`):

```
parse_expr -> parse_term (('+' | '-') parse_term)*
parse_term -> parse_atom (('*' | '/') parse_atom)*
parse_atom -> NUMBER | FLOAT | IDENT | IDENT '[' idx_expr ']' | '(' expr ')'
```

Each binary operation allocates a fresh register and emits the corresponding
GPU instruction (FADD, FMUL, etc.) or ARM64 instruction (add, mul, etc.)
depending on which backend is active.

---

## 6. Dual Backend

The compiler contains two independent code emitters. The active emitter is
selected by the function declaration keyword.

### 6.1 Dispatch rule

| Declaration | Backend | Output buffer |
|-------------|---------|---------------|
| `fn NAME ...` | sm90 GPU | `sass_buf` (64 KB) |
| `host fn NAME ...` | ARM64 | `code_buf` (64 KB) |
| `kernel NAME ...` | sm90 GPU (alias for `fn`) | `sass_buf` |

The parser sets a mode flag when entering a function. All emit calls within
that function body go to the selected backend. The mode flag is checked at
emit time, not parse time -- the parser logic is shared.

### 6.2 How this works in compiler.ls

The compiler itself is written as `host fn` declarations: `host fn main`,
`host fn parse_file`, `host fn emit_fadd`, etc. These compile to ARM64.

The `.ls` template files it reads as input contain `fn` declarations (GPU
kernels). When the compiler parses those, it calls the sm90 emitter to
write GPU instructions into `sass_buf`.

The compiler binary therefore contains only ARM64 code. The sm90 emitter
is ARM64 code that *writes* sm90 bytes into a buffer -- it does not contain
any sm90 code itself.

### 6.3 Calling convention (ARM64 host functions)

Standard AAPCS64 (ARM64 Procedure Call Standard):

| Register | Purpose |
|----------|---------|
| x0-x7 | Arguments and return values |
| x8 | Indirect result location (struct return) |
| x9-x15 | Scratch (caller-saved) |
| x16-x17 | Intra-procedure scratch |
| x19-x28 | Callee-saved |
| x29 | Frame pointer |
| x30 | Link register (return address) |
| sp | Stack pointer (16-byte aligned) |

Functions with <= 8 arguments pass all in registers. Functions with <= 2
return values return in x0, x1. For functions returning more data (e.g.,
safetensors_read returning a tensor list), the caller passes a pointer to
a bump-allocated output struct in x0.

All values are 64-bit (pointers, sizes, offsets). The compiler does not
use floating-point registers for host-side computation.

---

## 7. The sm90 Emitter (Ported from Forth)

### 7.1 Data structures

The opcode table from `opcodes-sm90.fs` becomes a read-only data section
in the ARM64 binary:

```
\ In compiler.ls, expressed as host-visible constants:
const OP_FMUL       = 0x7220
const OP_FADD       = 0x7221
const OP_FFMA       = 0x7223
const OP_IMAD       = 0x7224
const OP_IADD3      = 0x7210
const OP_LDG        = 0x7981
const OP_STG        = 0x7986
const OP_S2R        = 0x7919
const OP_ISETP      = 0x720c
const OP_BRA        = 0x7947
const OP_EXIT       = 0x794d
const OP_MUFU       = 0x7308
const OP_SHFL       = 0x7b22
const OP_STS        = 0x7988
const OP_LDS        = 0x7984
const OP_BAR        = 0x7b1d
const OP_LOP3_IMM   = 0x7812
const OP_SHF        = 0x7819
const OP_I2FP       = 0x7245
const OP_ATOM_ADD   = 0x79a8
const OP_MEMBAR     = 0x7992
\ ... etc (all from opcodes-sm90.fs)
```

### 7.2 Code buffer

```
host fn sass_buf   -> addr      \ returns pointer to 64KB buffer
host fn sass_pos   -> pos       \ current write position
host fn sass_reset              \ reset position to 0
```

The buffer is a 64 KB region in BSS (or bump-allocated). Each sm90
instruction is 16 bytes written sequentially.

### 7.3 sinst -- the core emit word

The Forth `sinst,` writes 16 bytes (inst64 low, inst64 high = ctrl64).
The ARM64 version:

```
host fn sinst inst64 ctrl64
    \ Write inst64 as 8 bytes little-endian at sass_buf[sass_pos]
    addr = sass_buf_ptr + sass_pos_val
    store_u64(addr, inst64)
    store_u64(addr + 8, ctrl64)
    sass_pos_val = sass_pos_val + 16
```

This is 4 ARM64 instructions: two `str x, [base, offset]` plus an add
and a store to update the position.

### 7.4 make_ctrl

The Forth `make-ctrl` builds a 64-bit control word from 7 fields via
bit-shifting and OR. The ARM64 version is a direct translation:

```
host fn make_ctrl stall yield wbar rbar wait reuse extra41 -> ctrl64
    ctrl64 = extra41
    ctrl64 = ctrl64 | (stall << 41)
    ctrl64 = ctrl64 | (yield << 45)
    ctrl64 = ctrl64 | (wbar << 46)
    ctrl64 = ctrl64 | (rbar << 49)
    ctrl64 = ctrl64 | (wait << 52)
    ctrl64 = ctrl64 | (reuse << 58)
```

### 7.5 Instruction emitters

Each Forth emit word (`fadd,`, `ldg,`, `imad,`, etc.) becomes an ARM64
function. Example -- `fadd,`:

```
host fn emit_fadd rd ra rb
    inst = 0x7221
    inst = inst | (rd << 16)
    inst = inst | (ra << 24)
    inst = inst | (rb << 32)
    ctrl = make_ctrl(5, 0, 7, 7, 0, 0, 0)    \ ctrl-fadd
    sinst(inst, ctrl)
    track_rd(rd)
```

The current Forth source defines ~40 instruction emitters in `gpu/emit.fs`.
Each is 5-15 lines of Forth. The ARM64 translations are slightly longer
(explicit variable names instead of stack juggling) but structurally
identical. Total: ~800 lines for all instruction emitters.

### 7.6 Register allocator

Same bump allocator as parser.fs:

```
host fn rreg_alloc -> reg
    reg = next_rreg
    next_rreg = next_rreg + 1

host fn freg_alloc -> reg
    reg = next_freg
    next_freg = next_freg + 1

host fn preg_alloc -> reg
    reg = next_preg
    next_preg = next_preg + 1
```

Reset at the start of each kernel function. No register reuse within a
kernel (same as current Forth: monotone allocation, ~20-30 registers per
kernel, well within Hopper's 255 architectural limit).

---

## 8. Safetensors Reader

### 8.1 File format

A safetensors file is:
```
[8 bytes: header_size as u64 LE]
[header_size bytes: JSON header]
[remaining bytes: raw tensor data, aligned]
```

The JSON header maps tensor names to `{dtype, shape, data_offsets}`.

### 8.2 Implementation

```
host fn safetensors_open path path_len -> st
    fd = syscall_open(path, O_RDONLY, 0)
    file_size = syscall_lseek(fd, 0, SEEK_END)
    base = syscall_mmap(0, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    syscall_close(fd)

    st = bump_alloc(sizeof_safetensors_handle)
    st.base = base
    st.file_size = file_size
    st.header_size = load_u64(base)
    st.header = base + 8
    st.data_start = base + 8 + st.header_size
```

### 8.3 JSON parsing (minimal)

The compiler does NOT need a general JSON parser. The safetensors header
has a flat structure:

```json
{"tensor_name": {"dtype": "F16", "shape": [4096, 4096], "data_offsets": [0, 33554432]}, ...}
```

The parser needs to:
1. Find a tensor by name (scan for `"name":`)
2. Extract `dtype` string
3. Extract `shape` array of integers
4. Extract `data_offsets` pair `[start, end]`

This is ~100 lines of string scanning with no recursion, no object model,
no heap allocation. All results are `(offset, length)` pairs pointing into
the mmap'd header.

```
host fn st_find_tensor st name name_len -> tensor_info
    \ Scan JSON header for "name":
    pos = 0
    while pos < st.header_size:
        if match_at(st.header, pos, name, name_len):
            \ Found; parse dtype, shape, data_offsets
            tensor_info = parse_tensor_entry(st, pos)
            return
        pos = pos + 1
    tensor_info.base = 0    \ not found

host fn st_tensor_ptr st tensor_info -> ptr
    ptr = st.data_start + tensor_info.offset_start
```

### 8.4 What the compiler does with tensors

For each layer, the compiler reads weight tensor metadata (shape, dtype,
offset) to determine:
- Matrix dimensions (N, K) for GEMV kernels
- Quantization group size for dequant parameters
- Memory layout for parameter struct construction

The raw weight bytes are NOT processed by the compiler -- they are copied
directly into the megakernel's constant memory region or referenced by
GPU VA. The compiler just needs to know shapes and offsets.

---

## 9. ELF Writer (GPU cubins)

### 9.1 Structure

Port of `gpu/emit.fs` build-cubin (the ELF builder embedded in emit.fs,
~300 lines) plus `elf-wrap.fs` (the file I/O wrapper, ~90 lines).

The GPU ELF layout:
```
ELF64 header (EM_CUDA = 190)
Section headers:
  [0] NULL
  [1] .shstrtab
  [2] .strtab
  [3] .symtab         (4 symbols: UND, .text section, .nv.constant0 section, kernel FUNC)
  [4] .nv.info        (EIATTR records: REGCOUNT, MAX_THREADS, etc.)
  [5] .text.<kernel>  (sm90 machine code from sass_buf)
  [6] .nv.info.<kernel> (per-kernel EIATTR: PARAM_CBANK, COOP_GROUP if cooperative)
  [7] .nv.shared.reserved.0 (NOBITS, size = shmem_size)
  [8] .nv.constant0.<kernel> (0x210 header + param_bytes)
```

### 9.2 Implementation strategy

The ELF is built into a bump-allocated output buffer. Each section is
written sequentially with offset tracking. The section header table is
written last (after all section sizes are known).

```
host fn build_gpu_elf kernel_name sass_buf sass_size params shmem_size coop -> elf_buf elf_size
    buf = bump_alloc(sass_size + 8192)    \ generous: code + headers + metadata
    pos = 0

    \ Phase 1: compute sizes
    shstrtab_size = compute_shstrtab(kernel_name)
    strtab_size = compute_strtab(kernel_name)
    symtab_size = 4 * 24                  \ 4 symbols, 24 bytes each
    nvinfo_size = compute_nvinfo(params, shmem_size, coop)
    cbuf0_size = 0x210 + params * 8

    \ Phase 2: write ELF header
    write_elf64_header(buf, &pos, ...)

    \ Phase 3: write sections in order
    write_shstrtab(buf, &pos, kernel_name)
    write_strtab(buf, &pos, kernel_name)
    write_symtab(buf, &pos, kernel_name, text_offset, cbuf0_offset)
    write_nvinfo(buf, &pos, regcount, max_threads, params, shmem_size, coop)
    text_offset = pos
    memcpy(buf + pos, sass_buf, sass_size); pos += sass_size
    write_nvinfo_kernel(buf, &pos, ...)
    \ .nv.shared.reserved.0 is NOBITS (no file content)
    write_cbuf0(buf, &pos, params)

    \ Phase 4: write section header table
    shdr_offset = pos
    write_section_headers(buf, &pos, ...)

    \ Phase 5: patch ELF header with shdr offset
    store_u64(buf + 0x28, shdr_offset)

    elf_buf = buf
    elf_size = pos
```

### 9.3 Cooperative kernel support

For megakernels, the ELF writer emits additional EIATTR records:
- `EIATTR_COOP_GROUP_INSTR_OFFSETS`: byte offsets of grid-sync instructions
- `EIATTR_COOP_GROUP_MASK_REGIDS`: register IDs used by the sync protocol
- `EIATTR_COOPERATIVE_FLAG`: marks the kernel as requiring cooperative launch

These are the same records built by the current Forth `build-elf` when
`cooperative? = 1`.

---

## 10. Megakernel Linker

### 10.1 What it does

A megakernel is one GPU kernel that executes all N layers of the model
sequentially, with grid-sync between layers. The linker:

1. Parses each layer's `.ls` template
2. For each layer, emits the instruction sequence into `sass_buf`
3. Between layers, emits grid-sync instructions
4. Tracks grid-sync offsets for the ELF COOP_GROUP records
5. Wraps the entire buffer in one cooperative GPU ELF

### 10.2 Per-layer emission

```
host fn emit_megakernel model_config layers
    sass_reset()
    for layer_idx in 0 .. n_layers:
        \ Each layer block: rmsnorm -> attention/deltanet -> rmsnorm -> ffn
        emit_rmsnorm(layer_idx, ...)
        emit_grid_sync()
        emit_attention_or_deltanet(layer_idx, ...)
        emit_grid_sync()
        emit_rmsnorm(layer_idx, ...)
        emit_grid_sync()
        emit_ffn(layer_idx, ...)
        if layer_idx < n_layers - 1:
            emit_grid_sync()
    emit_exit()
```

### 10.3 Grid-sync emission

Port of `gpu/emit.fs` `grid-sync,` (~50 lines):

```
host fn emit_grid_sync
    record_gridsync_offset(sass_pos)
    \ Thread 0 of each CTA: atomic increment sync counter
    \ If last CTA: release-store done_flag = 1
    \ Else: acquire-load poll done_flag until == 1
    \ All threads: BAR.SYNC (intra-CTA sync after grid sync)
    emit_bar_sync(0)
    \ ... (exact instruction sequence from gpu/emit.fs grid-sync,)
```

### 10.4 Parameter layout

The megakernel parameter struct is built by the compiler based on model
config. For Qwen 3.5 27B (64 layers: 48 DeltaNet + 16 full-attention):

```
Offset  Content
0x000   Global params (grid sync counter ptr, done flag ptr, n_layers, ...)
0x048   Layer 0 params (W_q ptr, W_k ptr, W_v ptr, W_o ptr, scales, x_ptr, ...)
0x118   Layer 1 params
...
0x2800  End of param region
```

The compiler emits `n_kparams` and param-region size into the ELF's
`.nv.constant0` section and `EIATTR_PARAM_CBANK` record.

---

## 11. Language Extensions for Host Code

The current `.ls` syntax handles GPU math (f32/u32 operations, array
indexing, shared memory, warp shuffles). Host-side code needs additional
capabilities.

### 11.1 Byte and pointer operations

New types beyond `f32` and `u32`:

| Type | Width | Usage |
|------|-------|-------|
| `u8` | 8-bit | Byte manipulation (ELF headers, JSON parsing) |
| `u64` | 64-bit | Pointers, file sizes, mmap addresses |
| `ptr` | 64-bit | Alias for u64, semantic clarity |

Syntax for typed operations:

```
host fn example
    \ Load byte from pointer
    byte = load8(base + offset)

    \ Store 64-bit value
    store64(addr, value)

    \ Pointer arithmetic
    next = ptr + 8
```

The type system is minimal: the compiler tracks whether each variable holds
a 64-bit value (pointer/integer) or a 32-bit value. In ARM64 emission,
64-bit values use X registers and 64-bit load/store instructions; 32-bit
values use W registers and 32-bit instructions.

### 11.2 String comparison

```
host fn str_eq a a_len b b_len -> result
    if a_len != b_len:
        result = 0
        return
    i = 0
    while i < a_len:
        if load8(a + i) != load8(b + i):
            result = 0
            return
        i = i + 1
    result = 1
```

This is a host function written in `.ls` -- it compiles to ARM64 `ldrb`,
`cmp`, `b.ne` instructions.

### 11.3 Syscall wrappers

Linux AArch64 syscall convention: syscall number in x8, arguments in
x0-x5, return value in x0.

```
host fn sys_open path flags mode -> fd
    \ svc #0 with x8 = 56 (openat), x0 = AT_FDCWD, x1 = path, x2 = flags, x3 = mode
    fd = syscall(56, -100, path, flags, mode)

host fn sys_mmap addr length prot flags fd offset -> ptr
    ptr = syscall(222, addr, length, prot, flags, fd, offset)

host fn sys_write fd buf count -> written
    written = syscall(64, fd, buf, count)

host fn sys_exit code
    syscall(93, code)
```

The `syscall` keyword is a compiler intrinsic. It emits:
```arm64
    mov x8, #SYSNO
    mov x0, arg0
    mov x1, arg1
    ...
    svc #0
    ; result in x0
```

### 11.4 Multiple return values

Limited to 2 values (returned in x0, x1):

```
host fn divmod a b -> quotient remainder
    quotient = a / b
    remainder = a - quotient * b
```

For functions needing more outputs, pass a pointer to a bump-allocated
struct:

```
host fn parse_tensor_entry st pos result_ptr
    \ writes dtype, shape, offsets into result_ptr
    store64(result_ptr + 0, dtype_offset)
    store64(result_ptr + 8, dtype_len)
    store64(result_ptr + 16, shape_0)
    store64(result_ptr + 24, shape_1)
    store64(result_ptr + 32, data_offset_start)
    store64(result_ptr + 40, data_offset_end)
```

### 11.5 Control flow additions

The GPU `.ls` has `each`, `stride`, `for`, `if<`, `if>=`, `if==`.
Host code adds:

| Construct | Syntax | Compiles to |
|-----------|--------|-------------|
| While loop | `while COND:` ... `endwhile` | `cmp + b.cond` back-edge |
| Return | `return` or `return VALUE` | `mov x0, val; ret` |
| Early return | `if COND: return VALUE` | `cmp + b.cond` to epilogue |

No `switch`, no `match`, no closures, no exceptions. The language stays
minimal.

### 11.6 Design principle

Every extension is a thin syntax over ARM64 instructions. No hidden
allocation, no implicit copies, no runtime. If you can not see what ARM64
instructions a line generates, the syntax is too complex.

---

## 12. GSP Boot + GPFIFO

### 12.1 BAR mapping

```
host fn map_bars -> bar0 bar4
    \ Open VFIO group (or sysfs resource files)
    fd0 = sys_open("/sys/bus/pci/devices/0000:dd:00.0/resource0", O_RDWR | O_SYNC, 0)
    bar0 = sys_mmap(0, 0x1000000, PROT_RW, MAP_SHARED, fd0, 0)

    fd4 = sys_open("/sys/bus/pci/devices/0000:dd:00.0/resource4", O_RDWR | O_SYNC, 0)
    bar4 = sys_mmap(0, 0x2000000000, PROT_RW, MAP_SHARED, fd4, 0)
```

### 12.2 GSP boot sequence

Direct port of the register sequence from `docs/gsp-native.md` sections
3.2-3.8. Each phase is a host function:

```
host fn gsp_phase1_pmc_check bar0
    boot0 = load32(bar0)
    arch = (boot0 >> 20) & 0xF
    if arch != 0xA:
        sys_write(2, "not Hopper\n", 11)
        sys_exit(1)

host fn gsp_phase2_falcon_reset bar0
    store32(bar0 + 0x1103C0, 1)        \ assert reset
    poll_bits(bar0 + 0x1103C0, 0x700, 0x000)  \ wait for ASSERTED
    store32(bar0 + 0x1103C0, 0)        \ deassert reset
    poll_bits(bar0 + 0x1103C0, 0x700, 0x200)  \ wait for DEASSERTED

host fn gsp_phase3_alloc_structures bar4 bump_ptr -> fmc_params wpr_meta gsp_args
    fmc_params = bar4_bump_alloc(bump_ptr, 4096)
    wpr_meta = bar4_bump_alloc(bump_ptr, 4096)
    gsp_args = bar4_bump_alloc(bump_ptr, 4096)
    \ ... populate structure fields with store32/store64

host fn gsp_phase4_load_firmware bar4 bump_ptr -> fw_addr fw_size
    fd = sys_open("/lib/firmware/nvidia/580.105.08/gsp_ga10x.bin", O_RDONLY, 0)
    fw_size = sys_lseek(fd, 0, SEEK_END)
    sys_lseek(fd, 0, SEEK_SET)
    fw_addr = bar4_bump_alloc(bump_ptr, fw_size)
    sys_read(fd, fw_addr, fw_size)
    sys_close(fd)
    \ Parse ELF64 header, find .fwimage section
    \ Copy payload to WPR region in BAR4

host fn gsp_phase5_start_fmc bar0 fmc_params_pa
    store32(bar0 + 0x110040, fmc_params_pa & 0xFFFFFFFF)
    store32(bar0 + 0x110044, fmc_params_pa >> 32)
    \ ... BCR register writes (from gsp-native.md section 3.6)
    store32(bar0 + 0x111388, 1)        \ CPUCTL: STARTCPU
```

Total GSP boot: ~600 lines of `.ls` host functions. Each function is a
sequence of `store32`, `load32`, `poll_bits`, and `bar4_bump_alloc` calls.
Structurally identical to the C in `lithos_gsp.c` but without any kernel
API -- pure memory-mapped I/O.

### 12.3 GPFIFO channel creation

After GSP boots, channel allocation uses RPC over shared-memory queues:

```
host fn create_channel bar0 bar4 bump_ptr -> channel
    \ Allocate GPFIFO ring buffer (8 KB, 512 entries)
    gpfifo_ring = bar4_bump_alloc(bump_ptr, 8192)
    gpfifo_ring_pa = bar4_to_pa(gpfifo_ring)

    \ Allocate USERD page
    userd = bar4_bump_alloc(bump_ptr, 4096)

    \ Send RPC to GSP: ALLOC_CHANNEL
    rpc_buf = bar4_bump_alloc(bump_ptr, 256)
    populate_alloc_channel_rpc(rpc_buf, gpfifo_ring_pa, ...)
    gsp_rpc_send(bar0, bar4, rpc_buf)
    gsp_rpc_wait(bar0, bar4)

    channel.gpfifo = gpfifo_ring
    channel.userd = userd
    channel.gp_put = 0
```

### 12.4 Work submission (QMD + GPFIFO + doorbell)

```
host fn submit_kernel channel cubin_gpu_va grid_dim block_dim params_ptr
    \ Build QMD (Queue Method Descriptor) in BAR4
    qmd = bar4_bump_alloc(bump_ptr, 256)
    populate_qmd(qmd, cubin_gpu_va, grid_dim, block_dim, params_ptr, ...)

    \ Write GPFIFO entry
    gp_entry = (qmd_gpu_va << 2) | GPFIFO_ENTRY_TYPE_QMD
    store64(channel.gpfifo + channel.gp_put * 8, gp_entry)
    channel.gp_put = (channel.gp_put + 1) & 511

    \ Ring doorbell: write GP_PUT to USERD
    store32(channel.userd + 0x8C, channel.gp_put)
```

---

## 13. Size Estimate

### 13.1 Current Forth codebase

| File | Lines | Purpose |
|------|-------|---------|
| `parser.fs` | 1730 | Lexer, parser, expression evaluation |
| `gpu/emit.fs` | 1490 | sm90 instruction encoders, ELF builder, grid-sync |
| `emit-elementwise.fs` | 552 | Element-wise kernel patterns |
| `emit-reduce.fs` | 330 | Reduction kernel patterns |
| `emit-gemv.fs` | 493 | GEMV kernel patterns |
| `opcodes-sm90.fs` | 355 | Opcode constants |
| `elf-wrap.fs` | 89 | ELF file I/O wrapper |
| `emit-sass.fs` | 13 | raw binary file writer |
| `lexer.fs` | 58 | Tokenizer |
| `emit-arm64.fs` | 41 | ARM64 emitter (stub) |
| `arm64-wrap.fs` | 66 | ARM64 ELF wrapper |
| `inline.fs` | 90 | Function inlining |
| `megakernel-params.fs` | 382 | Megakernel parameter layout |
| `lithos.fs` | 120 | Driver / main |
| **Total** | **5809** | |

### 13.2 Estimated .ls self-hosting compiler

| Section | Lines | Notes |
|---------|-------|-------|
| Constants + opcode tables | 300 | opcodes-sm90.fs (355) but denser (no comment blocks) |
| Syscall wrappers | 100 | open, close, read, write, mmap, munmap, lseek, exit |
| Bump allocator + string ops | 80 | bump_alloc, str_eq, memcpy, memset |
| Lexer | 150 | Cleaner than lexer.fs + parser.fs tokenizer combined |
| sm90 instruction emitter | 800 | gpu/emit.fs (1490) minus ELF builder minus comments |
| Parser + direct emission | 1200 | parser.fs (1730) minus comments, more readable |
| Kernel patterns | 600 | emit-elementwise + emit-reduce + emit-gemv condensed |
| Safetensors reader | 200 | mmap + minimal JSON scan |
| GPU ELF writer | 400 | ELF builder from gpu/emit.fs + elf-wrap.fs |
| Megakernel linker | 300 | megakernel-params.fs + grid-sync orchestration |
| GSP boot | 600 | gsp-native.md register sequences as host functions |
| GPFIFO + work submission | 200 | Channel creation, QMD build, doorbell |
| ARM64 self-emit stubs | 200 | Enough ARM64 encoder for bootstrap |
| Main / CLI driver | 100 | Argument parsing, file I/O, orchestration |
| **Total** | **~5200** | |

### 13.3 Analysis

The `.ls` version is ~5200 lines vs ~5800 lines of Forth. The `.ls` code
is longer per-function (explicit variable names, no stack tricks) but shorter
overall because:
- No Forth boilerplate (variable declarations, create/allot patterns)
- No duplicated stub definitions (parser.fs defines stubs for every emit
  word that gpu/emit.fs later overrides)
- Denser expression syntax (`result = a + b * c` vs. `a b c fmul, swap fadd,`)
- Single file eliminates load-order dependencies and include management

The GSP boot code (~600 lines) is new -- it does not exist in the current
Forth compiler. It is ported from the C kernel module and gsp-native.md.
Without GSP boot, the compiler proper is ~4600 lines.

---

## 14. Data Flow Summary

```
compiler.ls
    |
    v  (bootstrap compile via Forth, or self-compile)
lithos (ARM64 ELF binary)
    |
    |-- reads: model.ls (template describing model architecture)
    |-- reads: model.safetensors (quantized weights)
    |
    |-- Phase 1: Parse model.ls
    |       tokenize -> parse -> emit sm90 per kernel function
    |
    |-- Phase 2: Read safetensors metadata
    |       mmap -> parse JSON header -> extract tensor shapes/offsets
    |
    |-- Phase 3: Build megakernel
    |       for each layer:
    |           emit layer kernels (rmsnorm, attn, ffn) with layer-specific constants
    |           emit grid-sync between stages
    |       wrap in cooperative GPU ELF
    |
    |-- Phase 4: Boot GPU
    |       map BAR0, BAR4
    |       GSP boot sequence (phases 1-8)
    |       create GPFIFO channel
    |
    |-- Phase 5: Launch
    |       copy megakernel ELF to BAR4
    |       build parameter struct (weight pointers, KV cache pointers, ...)
    |       write QMD to GPFIFO ring
    |       ring doorbell
    |       poll for completion
    |
    v
inference output (token IDs)
```

---

## 15. Open Questions

1. **ARM64 encoder completeness.** The Forth bootstrap needs a real ARM64
   encoder before it can compile `compiler.ls`. This is ~400 lines of new
   Forth in `emit-arm64.fs`. Priority: this is the critical path for
   bootstrap.

2. **Self-compile verification.** The fixed-point test (Phase 3 in section
   2) requires bit-identical output. This means the ARM64 encoder must be
   deterministic (no address-dependent decisions) and the ELF layout must
   be fully specified.

3. **Error handling.** The current Forth compiler aborts on errors. The
   self-hosting compiler should at minimum print the file, line number, and
   error message before calling `sys_exit(1)`. No exception mechanism --
   just formatted error output.

4. **PCI device discovery.** The BAR paths in section 12.1 are hardcoded
   to `0000:dd:00.0`. The compiler should scan `/sys/bus/pci/devices/` for
   NVIDIA vendor ID 0x10de and Hopper device IDs, or accept a `--device`
   flag.

5. **FSP boot complexity.** Section 3.7 of gsp-native.md notes that FSP
   communication on production GH200 is ~2000 lines of logic. This is the
   largest single porting effort and may be done incrementally (development
   on pre-production hardware with FSP bypass, then adding FSP support).

6. **Multi-model support.** The current architecture compiles one model
   at a time. Supporting multiple model architectures (DeltaNet vs.
   full-attention) requires either separate `.ls` templates or a template
   parameter mechanism. Current approach: separate templates per
   architecture, compiler selects based on CLI flag.
