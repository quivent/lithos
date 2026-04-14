# Bootstrap Link Report -- Three New Parser Modules

Date: 2026-04-13

## Summary

The three new ARM64 assembly modules (lithos-expr-eval.s, lithos-compose.s,
lithos-control.s) plus the existing lithos-glue.s stubs file now link
successfully into the lithos-bootstrap binary.

## What was done

### Problem

The 43 undefined symbols fell into three categories:

1. **Symbols defined in lithos-parser.s but not exported** (34 symbols).
   The build.sh `globals_for()` regex only matched prefixes like `code_`,
   `emit_`, `parse_`, `sym_`, etc. It missed symbols with prefixes:
   `alloc_`, `reset_`, `skip_`, `scope_`, `patch_`, `tok_`, `free_`,
   `next_`, `compose_`, and bare names `advance` and `expect`.

2. **Symbols defined in BOTH lithos-parser.s AND the new modules** (7 symbols):
   `parse_atom`, `parse_expr`, `parse_for`, `parse_mem_load`,
   `parse_mem_store`, `parse_reg_read`, `parse_reg_write`.
   The new modules provide replacement implementations.

3. **Truly missing symbols** (9 symbols) provided by lithos-glue.s:
   `emit_ldrb_zero`, `emit_strb_zero`, `emit_ldrh_zero`, `emit_strh_zero`,
   `emit_ldr_w_zero`, `emit_str_w_zero`, `emit_ldr_x_zero`,
   `emit_str_x_zero`, `parse_data_decl`.

### Changes to build.sh

1. **SRCS array**: Added `lithos-expr-eval.s`, `lithos-compose.s`,
   `lithos-control.s`, and `lithos-glue.s`.

2. **globals_for() regex**: Extended the prefix match to include:
   `alloc_|reset_|skip_|scope_|patch_|tok_|free_|next_|compose_`
   and added `advance` and `expect` to the exact-match list.

3. **globals_for_filtered()**: New function that suppresses specific
   symbols from global promotion. Used for lithos-parser.s to keep
   the 7 duplicate symbols local (so the new modules' versions win).

4. **PARSER_SUPPRESS**: Lists the 7 symbols to suppress:
   `parse_atom parse_expr parse_for parse_mem_load parse_mem_store
   parse_reg_read parse_reg_write`

5. **LINK_ORDER**: Added the 4 new .o files.

### No changes to .s source files

All existing .s files were left unmodified. The lithos-glue.s file
(already present on disk) provides the 9 stub implementations.

## Build result

```
built: /home/ubuntu/lithos/bootstrap/lithos-bootstrap
-rwxrwxr-x 1 ubuntu ubuntu 122872 Apr 13 20:42 lithos-bootstrap
```

No duplicate global symbols. No assembly warnings. Clean link.

## Test results

### Test 1: Simple binding -- `x 42`

```
$ printf 'x 42\n' > /tmp/bind.ls
$ ./lithos-bootstrap /tmp/bind.ls -o /tmp/bind.elf
lithos: wrote /tmp/bind.elf
```

Result: SUCCESS. Produces a 132-byte ARM64 ELF. The binary crashes
at runtime with SIGILL (expected -- the emitted code for a bare binding
has no exit syscall).

### Test 2: Composition -- `add a b :`

```
$ printf 'add a b :\n    c a + b\n' > /tmp/comp.ls
$ ./lithos-bootstrap /tmp/comp.ls -o /tmp/comp.elf
parse error
```

Result: PARSE ERROR. The composition syntax `name arg1 arg2 :` followed
by an indented body is not yet handled end-to-end by the current parser
dispatch. The new lithos-compose.s module defines `parse_body_compose`
and related entry points, but the top-level parse loop in lithos-parser.s
does not yet call into them for composition lines. Wiring the dispatch
is the next step.

### Test 3: compiler.ls self-compile

```
$ ./lithos-bootstrap compiler/compiler.ls -o compiler/lithos-stage1
parse error
```

Result: PARSE ERROR. compiler.ls uses the full language (compositions,
control flow, memory ops, etc.) which requires the complete dispatch
integration. The bootstrap parser recognizes tokens but the top-level
`parse_body` in lithos-parser.s does not yet dispatch to the new
`parse_composition_compose`, `parse_if_eq`, `parse_for`, etc.

## Next steps

1. Wire the top-level dispatch in lithos-parser.s to call into:
   - `parse_composition_compose` (lithos-compose.s) for composition lines
   - `parse_if_eq`, `parse_if_ge`, `parse_if_lt` (lithos-control.s) for conditionals
   - `parse_for` (lithos-control.s) for loops
   - `parse_expr` (lithos-expr-eval.s) for expression evaluation

2. The new parse_body in lithos-parser.s needs to detect the `:` token
   to distinguish compositions from bindings, then dispatch accordingly.

3. Runtime testing of emitted code (the simple `x 42` test produces an
   ELF but crashes -- needs an exit syscall to be useful).
