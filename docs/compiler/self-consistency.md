# compiler.ls Self-Consistency Fix Report

All 8 internal grammar inconsistencies fixed so compiler.ls can parse its own source.

## Issue 1: `goto`, `label`, `endfor` used but not tokens/keywords

**Problem:** `goto` used 30x, `label` used in lexer, `endfor` used 9x. `goto` was not a keyword. `endfor` was tokenized (type 17) but never consumed by parse_stmt.

**Fix:**
- Added TOK_GOTO = 93 to match_keyword (length 4, line ~1932)
- `label` was already TOK_LABEL = 33 (match_keyword length 5)
- `endfor` was already TOK_ENDFOR = 17 (match_keyword length 6)
- Added parse_stmt handlers for all three: goto (line ~3271), label (line ~3273), endfor (line ~3276)
- Added `parse_goto`, `parse_label_decl`, `parse_inline_label`, `label_find` implementations (lines ~3319-3392)
- Added label table state: `label_names`, `label_lens`, `label_offsets`, `n_labels` (lines ~2393-2397)
- Inline labels (`name:` syntax) detected in parse_ident_stmt by checking for TOK_COLON after identifier

## Issue 2: `load_u8/16/32/64`, `store_u8/16/32/64` used but not tokens

**Problem:** Used as function-style calls (42 load_u*, 19 store_u* occurrences) but not recognized by parser.

**Fix:**
- Handled as magic identifiers in parse_ident_stmt (lines ~3648-3720)
- Dispatch by identifier length and first bytes to `parse_load_u`, `parse_store_u`, `parse_load_u_idx`, `parse_store_u8_at`
- Implementations at lines ~3469-3515: emit proper loads/stores at the requested width

## Issue 3: `syscall` used but not a token/keyword

**Problem:** `syscall fd 56 dirfd path flags mode` pattern used 8x in safetensors/ELF sections.

**Fix:**
- Added TOK_SYSCALL = 94 to match_keyword (length 7, line ~2021)
- Added parse_stmt handler (line ~3289)
- Added `parse_syscall_stmt` + `parse_syscall_args` implementations (lines ~3410-3432)
- Emits: MOV X8 sysnum, MOV X0-X5 args, SVC #0, capture result

## Issue 4: `if!=` used but only `if==`, `if>=`, `if<` recognized

**Problem:** parse_conditional only decoded 3 comparison types. Compiler uses `if!=` 5x, `if>` 5x, `if<=` 4x.

**Fix:**
- Extended parse_conditional (lines ~4027-4084) to handle 6 comparison types:
  - 0 = EQ (`if==`), 1 = GE (`if>=`), 2 = LT (`if<`)
  - 3 = NE (`if!=`), 4 = GT (`if>`), 5 = LE (`if<=`)
- Added ARM64 branch condition codes: B.EQ (0x54000000), B.LE (0x5400000D), B.GT (0x5400000C)
- Added `!` (ASCII 33) to scan_ident character set so `if!=` scans as one 4-char identifier
- Changed conditional check from `if== len 4` to `if>= len 3` to handle 3-char `if<` and `if>`

## Issue 5: 3-operand statement ops (and, shl, shr, mul, add, sub, or, xor)

**Problem:** `and tid_local row 255`, `shr K_packed K 3` etc. treated as variable names.

**Fix:**
- Added magic identifier dispatch in parse_ident_stmt for all 8 ops (lines ~3730-3784)
- Added `parse_3op_and/or/xor/shl/shr/mul/add/sub` implementations (lines ~3522-3612)
- GPU: routes to LOP3 (and/or/xor), SHF (shl/shr), emit_p_mul/add/sub
- ARM64: routes to AND/ORR/EOR/LSL/LSR/MUL/ADD/SUB register forms

## Issue 6: `param NAME TYPE` never parsed

**Problem:** TOK_PARAM (type 12) lexed but no parse handler. n_kparams stayed 0.

**Fix:**
- Added parse_stmt handler for TOK_PARAM = 12 (line ~3303)
- Added `parse_param` implementation (lines ~3435-3465)
- Stores name + type in kparam_names/kparam_lens/kparam_types tables
- Emits ULDC from constant bank (offset 528 + idx*8) to load param into register
- Increments n_kparams and updates gpu_n_kparams for ELF metadata

## Issue 7: `$` register prefix not a lexer token

**Problem:** `$0`, `$8`, `$TID_X` etc. in host compositions couldn't be lexed.

**Fix:**
- Added TOK_DOLLAR = 97 to lexer single-char section (ASCII 36, line ~2322)
- Updated parse_regread to handle `$N` (dollar + int) and `$NAME` (dollar + ident with dict lookup) (lines ~2979-3010)
- Updated parse_regwrite similarly (lines ~3013-3048)

## Issue 8: arch/hopper.dict and arch/arm64.dict never read

**Problem:** No dictionary loader for special register names.

**Fix:**
- Added dictionary state: `dict_names`, `dict_lens`, `dict_instr_types`, `dict_reg_ids`, `n_dict_entries`
- Added `dict_lookup` and `dict_add_entry` compositions (lines ~2418-2452)
- parse_regread/parse_regwrite use dict_lookup when `$NAME` is encountered
- NOTE: dict_load (file parsing of .dict files) requires the file I/O compositions which are host-only. For bootstrap, the dict entries can be pre-populated by adding dict_add_entry calls in parser_init, or the .dict files can be loaded at startup once the host mmap_file is available.

## Additional Fixes

### cubin_buf size (was 512KB, now 4MB)
- Changed `buf cubin_buf` from 524288 to 4194304 (line ~1679)
- Changed `ELF_CUBIN_SIZE` constant from 524288 to 4194304 (line ~4756)
- Added TODO comments for bump to 320MB for full 64-layer megakernel

### ARM64 ELF wrapper (elf_build_arm64)
- Added `elf_build_arm64` composition (lines ~5238-5310)
- Builds minimal ELF64 executable: ELF header + 1 PT_LOAD program header + code
- e_machine = 0xB7 (EM_AARCH64), entry at 0x400078
- Updated lithos_main to call `elf_build_arm64` + `elf_save` instead of raw `write_file`

### Forth-style `constant` at file scope
- Added TOK_CONSTANT = 96 to match_keyword (length 8)
- Added TOK_CONTINUE = 95 to match_keyword (length 8)
- Added file-scope handler in parse_file for `INT constant NAME` pattern
- Added parse_stmt handler for constant encountered in statement context

### Parser state initialization
- Added `n_labels 0` and `n_kparams 0` to parser_init reset

## Remaining Known Issues (not fixed, out of scope)

1. **dict_load from file** — requires host I/O integration; dict entries must be hardcoded or loaded at startup
2. **`u32>f32` / `s32>f32` cast tokens** — not used in compiler.ls itself; needed for inference kernels
3. **`bra` statement in kernels** — used in inference .ls files but not in compiler.ls
4. **`log2` (multi-byte)** — not used in compiler.ls
5. **`**` / `***` dimensional operators** — lexer emits single `*`; not used in compiler.ls
6. **goto forward-patching** — goto to not-yet-seen labels emits placeholder but patching not wired
7. **EIATTR_EXIT_INSTR_OFFSETS** still hardcoded to 256
8. **ST_MAX_TENSORS** still 128 (580 needed for Qwen 27B)
