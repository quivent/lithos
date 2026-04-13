# Compiler Construct Audit: compiler.ls vs bootstrap lithos-parser.s

Audit date: 2026-04-13
Source: `/home/ubuntu/lithos/compiler/compiler.ls` (5467 lines)
Parser: `/home/ubuntu/lithos/bootstrap/lithos-parser.s` + `lithos-compose.s`

## Complete Construct Table

| Construct | Example from compiler.ls | Line(s) | Parser handles? | Fix needed? |
|---|---|---|---|---|
| **Binding with +** | `val val * 10 + d` | 2522 | YES — parse_additive handles TOK_PLUS | No |
| **Binding with -** | `val 0 - val` | 2524 | YES — parse_additive handles TOK_MINUS | No |
| **Binding with \*** | `idx token_count * 3` | 1768 | YES — parse_multiplicative handles TOK_STAR | No |
| **Binding with /** | `val 0xF9400000 \| ((imm / 8) << 10)` | 502 | YES — parse_multiplicative handles TOK_SLASH | No |
| **Binding with \|** | `result upper \| lower \| under` | 1785 | YES — parse_bitwise handles TOK_PIPE | No |
| **Binding with &** | `result c >= 48 & c <= 57` | 1787 | YES — parse_bitwise handles TOK_AMP | No |
| **Binding with ^** | `inv cond ^ 1` | 450 | YES — parse_bitwise handles TOK_CARET | No |
| **Binding with <<** | `val 0x8B000000 \| (rm << 16)` | 248 | YES — parse_shift handles TOK_SHL | No |
| **Binding with >>** | `chunk1 (imm >> 16) & 0xFFFF` | 485 | YES — parse_shift handles TOK_SHR | No |
| **Nested parentheses** | `val 0xF9400000 \| ((imm / 8) << 10) \| (rn << 5) \| rd` | 502 | YES — parse_atom handles TOK_LPAREN/RPAREN recursively | No |
| **Unary negation (literal)** | `-16` (in `emit_stp_pre FP LR SP -16`) | 825 | YES — parse_int_literal handles leading '-' | No |
| **Unary negation (expression)** | `-100` in `sys_openat -100 path 0 0` | 4364 | YES — parse_atom .Latom_unary_neg emits SUB XZR | No |
| **Hex literals** | `0xF9000000` | 507 | YES — parse_int_literal handles 0x prefix | No |
| **Large hex literals** | `0x000FC00000000000` | 1234 | YES — hex parsing in bootstrap is 64-bit clean | No |
| **Composition def (0 args)** | `arm64_reset :` | 180 | YES — parse_composition parses 0..N params before colon | No |
| **Composition def (1 arg)** | `arm64_emit32 val :` | 175 | YES | No |
| **Composition def (2 args)** | `emit_add_reg rd rn rm :` | 247 | YES | No |
| **Composition def (3 args)** | `emit_add_reg_lsl rd rn rm amount :` | 252 | YES | No |
| **Composition def (4+ args)** | `make_ctrl stall yield wbar rbar wait reuse extra41 :` (7 args) | 1020 | YES | No |
| **Composition def (10+ args)** | `elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size elf_cooperative gridsync_off_ptr gridsync_cnt :` (10 args) | 4928 | YES — params parsed in loop; limit is X0-X7 (8 regs) | **POSSIBLE** — 10 args exceeds X0-X7 ABI |
| **Composition calls** | `arm64_emit32 val` | 249 | YES — parse_comp_call / parse_binding_compose | No |
| **Composition calls (multi-arg)** | `emit_stp_pre FP LR SP -16` | 825 | YES | No |
| **Store: <- width addr val** | `<- 32 arm64_buf + arm64_pos val` | 177 | YES — parse_mem_store handles TOK_STORE | No |
| **Store: <- 8** | `<- 8 gpu_buf + gpu_pos b` | 995 | YES | No |
| **Store: <- 16** | `cubin_emit_u16` calls `cubin_emit_byte` (indirect) | 1722 | YES (via 8-bit stores) | No |
| **Store: <- 32** | `<- 32 tokens + idx type` | 1769 | YES | No |
| **Store: <- 64** | `<- 64 tbl_name_ptr + count * 8 tname_start` | 4688 | YES | No |
| **Load: -> width addr** | `-> 32 tokens idx` | 2411 | YES — parse_mem_load handles TOK_LOAD | No |
| **Load: -> 8** | `c0 -> 8 ptr 0` | 2457 | YES | No |
| **Load: -> 32** | `-> 32 sym_lens i * 4` | 2573 | YES | No |
| **Load: -> 64** | `-> 64 argv 8` | 5433 | YES | No |
| **Load: -> 16** | (not used in compiler.ls) | — | YES — width-dispatch logic present | No |
| **Register write: down-arrow $N val** | `down-arrow $8 56` | 5334 | YES — parse_reg_write + parse_dollar_reg | No |
| **Register read: up-arrow $N** | `fd up-arrow $0` | 5340 | YES — parse_reg_read + parse_dollar_reg | No |
| **for/endfor** | `for i 0 len 1` ... `endfor` | 2519/4727 | **PARTIAL** — parse_for handles `for i start end step` but does NOT handle `endfor` keyword | **YES** |
| **for (3-arg implicit step)** | `for j 0 clen 1` | 2565 | YES — step is parsed as expression | No |
| **if==** | `if== t 50` | 2771 | **NO** — bootstrap parser has `if` (TOK_IF=13) which uses CBZ on condition expression; does NOT parse fused comparison keywords `if==`, `if>=`, etc. | **YES** |
| **if>=** | `if>= gridsync_count 256` | 979 | **NO** — same as above | **YES** |
| **if<** | `if< count ST_MAX_TENSORS` | 4687 | **NO** | **YES** |
| **if>** | `if> gpu_pos 0` | 5454 | **NO** | **YES** |
| **if<=** | `if<= remaining 0` | 5210 | **NO** | **YES** |
| **if!=** | `if!= a b` | 3379 | **NO** | **YES** |
| **elif** | `elif ic == 9` | 2096 | YES — parse_if handles TOK_ELIF | No |
| **else** | `else` | 2099, 5160 | YES — parse_if handles TOK_ELSE | No |
| **while** | `while new_pos < end` | 1816 | YES — parse_while | No |
| **trap (bare keyword)** | `trap` | 5339 | YES — parse_ident_stmt intercepts "trap" text, emits SVC #0 | No |
| **trap (Forth-style: trap result syscall args...)** | `trap fd 56 dirfd path flags mode` | 4348 | **NO** — bootstrap only emits bare SVC #0; does NOT parse result-binding or syscall-number or arg dispatch | **YES** |
| **var declaration** | `var token_count 0` | 144 | YES — parse_var_decl | No |
| **const declaration** | `const SYS_READ 63` | 150 | YES — parse_const_decl | No |
| **buf declaration** | `buf tokens 262144` | 143 | YES — parse_buf_decl | No |
| **\\\\ comments (end-of-line)** | `\\ Write a 32-bit word` | 176 | YES — lexer skips `\\` to EOL | No |
| **\\\\ comment-only lines** | `\\ ============...` | 167 | YES | No |
| **Blank lines** | (throughout) | passim | YES — skip_newlines | No |
| **Indentation (4-space body)** | Standard throughout | passim | YES — parse_body checks indent level | No |
| **return** | `return` | 1826 | YES — parse_return handles bare and expr return | No |
| **continue** | `continue` | 2116 | **NO** — str_continue string is declared but no parse_continue handler exists in bootstrap parser | **YES** |
| **goto** | `goto indent_done` | 2100 | **NO** — str_goto string is declared but no parse_goto handler exists in bootstrap parser | **YES** |
| **label (keyword)** | `label indent_done` | 2101 | YES — parse_label handles TOK_LABEL | No |
| **Inline labels (name:)** | `loop_skip:` | 4385 | **NO** — bootstrap parser does not detect or handle colon-terminated inline labels within body context | **YES** |
| **host composition** | `host mmap_file path :` | 5332 | **NO** — TOK_HOST (35) is defined in token constants but parse_toplevel does NOT dispatch on it; no parse_host_composition exists | **YES** |
| **Forth-style constant: VALUE constant NAME** | `128 constant ST_MAX_TENSORS` | 4331 | **NO** — bootstrap parse_toplevel does not handle TOK_INT at file scope followed by `constant` keyword | **YES** |
| **Bare constant assignment (NAME VALUE)** | `OP_FMUL 0x7220` | 1033 | **PARTIAL** — parse_binding_compose in lithos-compose.s handles `name expr` form; works if NAME is TOK_IDENT and VALUE parses as expression | Likely OK |
| **Array indexing: src [ pos ]** | `c src [ new_pos ]` | 1817 | **NO** — bootstrap expression parser has no handler for TOK_LBRACK following an identifier; this byte-load syntax is not implemented | **YES** |
| **Multi-value return binding** | `base file_size st_open path` | 4741 | **NO** — bootstrap parser binds a single return (X0) per call; multi-value destructuring (2+ names before call) not implemented | **YES** |
| **Comparison operators in expressions** | `c >= 48 & c <= 57` | 1787 | YES — parse_expr comparison level handles ==, !=, <, >, <=, >= | No |
| **Chained & in expressions** | `b0 == 102 & b1 == 111 & b2 == 114` | 1882 | YES — parse_bitwise loops on TOK_AMP | No |
| **Chained \| in expressions** | `ok alpha \| dot \| angle_l` | 1824 | YES — parse_bitwise loops on TOK_PIPE | No |
| **max built-in** | `max_reg max rd max_reg` | 967 | **NO** — `max` is not a keyword or built-in in the bootstrap parser; would be treated as composition call | **YES** |
| **Σ (sum reduction)** | `parse_reduction_sum src` | 2859 | YES (stub) — parse_atom handles TOK_SUM | No |
| **Triangle (max reduction)** | `parse_reduction_max src` | 2865 | YES (stub) — parse_atom handles TOK_MAX | No |
| **Nabla (min reduction)** | `parse_reduction_min src` | 2871 | YES (stub) — parse_atom handles TOK_MIN | No |
| **Sqrt** | `emit_p_mufu rd src 0x2000` | 2837 | YES (stub) — parse_atom handles TOK_SQRT | No |
| **Sin / Cos** | `emit_p_mufu rd src 0x0400` / `0x0000` | 2845/2853 | YES (stub) — parse_atom handles TOK_SIN/TOK_COS | No |
| **# (index/argmax)** | `parse_index_reduction` | 2877 | YES (stub) — parse_atom handles TOK_INDEX | No |
| **\*\* (elementwise)** | `parse_elementwise` | 2881 | **NO** — TOK_STARSTAR (91) is not in bootstrap token constants or parse_atom dispatch | **YES** |
| **\*\*\* (matrix)** | (not used in compiler.ls body) | — | **NO** — TOK_STARSTARSTAR (92) not in bootstrap | N/A |
| **$ (dollar sigil)** | `$8`, `$0`, `$TID_X` | 5334-5340 | YES — parse_dollar_reg handles $N numeric form | No |
| **$ with named registers** | `$TID_X` | (via parse_regread) | **PARTIAL** — bootstrap parse_dollar_reg only handles $N (numeric); named special regs like $TID_X require dict_lookup which is in compiler.ls only | **YES** |
| **shared declaration** | `shared smem 1024 f32` | (via parse_shared) | **NO** — TOK_SHARED not dispatched by bootstrap parse_statement | **YES** |
| **barrier** | `emit_p_bar_sync` (via TOK_BARRIER=32) | 3254 | **NO** — TOK_BARRIER not in bootstrap token constants or dispatch | **YES** |
| **kernel keyword** | `kernel` at top level | 4261 | **NO** — bootstrap parse_toplevel does not dispatch TOK_KERNEL | **YES** |
| **param declaration** | `param name type` | 3304 | **NO** — TOK_PARAM not dispatched by bootstrap parse_statement | **YES** |
| **each** | `each i` | 3243 | YES — parse_each in bootstrap | No |
| **stride** | `stride i dim` | 3247 | **NO** — TOK_STRIDE (19) is defined but not dispatched in bootstrap parse_statement | **YES** |
| **exit keyword** | `exit` | 3261 | **NO** — TOK_EXIT_KW (34) is defined but treated as unknown; no handler for exit as standalone statement | **YES** |
| **Composition as expression** | `val parse_int_token` | 2806 | YES — parse_atom falls through to parse_comp_call for unknown idents | No |
| **Recursive composition calls in expressions** | `emit_p_mov_imm rd (parse_int_token)` | passim | YES — expressions can contain calls | No |
| **1/ (reciprocal prefix)** | `1/ not used at top level` | (via parse_ident_expr) | **NO** — bootstrap has no special ident-prefix parsing for `1/`, `2^`, `e^`, `ln` | N/A (self-hosting only) |
| **2^ (exp2 prefix)** | same | — | **NO** | N/A |
| **e^ / ln prefixes** | same | — | **NO** | N/A |
| **load_32 / store_32 etc.** | `parse_mem_load 32` | 3677 | **NO** — magic identifier dispatch (load_8, load_16, store_32, etc.) not in bootstrap | N/A (self-hosting only) |
| **3-op keywords: and, or, xor, shl, shr, mul, add, sub** | `parse_3op_and` etc. | 3525-3614 | **NO** — these are compiler.ls self-hosting constructs not in bootstrap | N/A (self-hosting only) |
| **weight keyword** | `weight` (TOK_WEIGHT=25) | token enum | **NO** — not dispatched | N/A |
| **project keyword** | `project` (TOK_PROJECT=30) | token enum | **NO** — not dispatched | N/A |

## Summary of Constructs the Bootstrap Parser Does NOT Handle

### Critical (used in compiler.ls, blocks self-compilation):

1. **`if==`, `if>=`, `if<`, `if>`, `if<=`, `if!=`** — Fused comparison-conditionals (lines 979, 2501, 2771, 3379, 4687, 5210, 5454, etc. — 412 total uses). The bootstrap parser only handles `if` as a keyword (TOK_IF=13) followed by an expression and CBZ. The compiler.ls uses fused forms like `if== t 50` where the comparison operator is part of the keyword token. **This is the single largest gap — these appear 412 times in compiler.ls.**

2. **`endfor`** — Loop terminator keyword (lines 4450, 4727, 4835, 5093, 5120). Bootstrap parse_for expects indentation-based loop bodies, not explicit `endfor`. compiler.ls uses both patterns (indentation-based `for` with implicit end AND `endfor`-terminated loops).

3. **`goto label_name`** — Unconditional branch to label (lines 2100, 4393, 4415, 4440, 4530, 4550, etc.). String `str_goto` declared but no parse handler.

4. **`continue`** — Loop continuation (line 2116, 2130, etc.). String `str_continue` declared but no parse handler.

5. **Inline labels (`name:`)** — Labels defined as `identifier:` within body (lines 4385, 4406, 4426, 4518, 4593, 4631, 4663, 4798, 5209, 5220). No detection in bootstrap parser.

6. **`host` composition modifier** — `host mmap_file path :` (lines 5332, 5367, 5420). TOK_HOST defined but not dispatched at top level.

7. **`trap` with result/args (Forth-style)** — `trap fd 56 dirfd path flags mode` (lines 4348-4357, 5205-5221, 5334-5399). Bootstrap only handles bare `trap` as SVC #0; does not parse syscall number, arguments, or result binding.

8. **Forth-style `VALUE constant NAME`** — `128 constant ST_MAX_TENSORS` (lines 4331-4342, 4759-4768). Not parsed at top level.

9. **Multi-value return binding** — `base file_size st_open path` (lines 4536, 4562, 4569, 4581, 4586, 4652, 4685, 4741, 4742, 5437). Bootstrap binds only X0 from calls.

10. **Array byte-load syntax `src [ pos ]`** — (lines 1817, 1831, 1835, 1839, 1847, etc.). No LBRACK handler in expression parser.

11. **`**` (elementwise operator)** — TOK_STARSTAR (91) used at line 2881. Not in bootstrap token set.

12. **`shared` declaration** — Not dispatched.

13. **`barrier`** — Not dispatched.

14. **`kernel` keyword at top level** — Not dispatched (only TOK_IDENT triggers composition parsing).

15. **`param` declaration** — Not dispatched.

16. **`stride` loop form** — Not dispatched.

17. **`exit` as statement** — Not dispatched.

18. **`max` built-in** — `max_reg max rd max_reg` (line 967). Not a recognized keyword or built-in.

19. **Named special registers (`$TID_X`)** — parse_dollar_reg only handles numeric `$N` form.

### Non-critical (self-hosting-only constructs, not needed for bootstrap):

- `1/`, `2^`, `e^`, `ln` prefix operators
- `load_8`, `load_16`, `load_32`, `load_64`, `store_8`, etc. magic identifiers
- 3-operand keywords: `and`, `or`, `xor`, `shl`, `shr`, `mul`, `add`, `sub`
- `weight`, `project`, `template`, `bind`, `runtime` keywords

## Verdict

The bootstrap parser handles the basic structural skeleton of Lithos: composition definitions, parameter binding, expression parsing with all operators (+, -, *, /, |, &, ^, <<, >>), parenthesized expressions, unary negation, hex literals, var/const/buf declarations, if/elif/else, while, for, each, return, label, memory load/store arrows, register read/write with $N, and comments.

However, **19 constructs used in compiler.ls are missing from the bootstrap parser**. The most impactful gap is the fused conditional keywords (`if==`, `if>=`, etc.) which appear 412 times. Without these, compiler.ls cannot self-compile through the bootstrap path. The Forth-style `trap` with result binding, `goto`/`continue`/inline labels, `endfor`, `host` compositions, multi-value returns, and array indexing are also critical for the later sections (safetensors reader, ELF writer, main entry point).
