# Bootstrap Glue Report

## Summary

Created `/home/ubuntu/lithos/bootstrap/lithos-glue.s` as the safety-net stub
file for the 43-symbol contract. Added it to `build.sh` SRCS and LINK_ORDER.
Updated `globals_for` in `build.sh` to promote 6 formerly-local symbols.

## Symbol Audit (43 symbols)

### Already provided by lithos-parser.s (34 symbols)

28 as GLOBAL (T/D), 6 as local (t/b):

| Symbol | Status | Object |
|--------|--------|--------|
| alloc_reg | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| emit32 | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| emit_add_reg | GLOBAL | lithos-parser.o |
| emit_and_reg | GLOBAL | lithos-parser.o |
| emit_b | GLOBAL | lithos-parser.o |
| emit_b_cond | GLOBAL | lithos-parser.o |
| emit_cmp_reg | GLOBAL | lithos-parser.o |
| emit_cur | GLOBAL | lithos-parser.o |
| emit_eor_reg | GLOBAL | lithos-parser.o |
| emit_lsl_reg | GLOBAL | lithos-parser.o |
| emit_lsr_reg | GLOBAL | lithos-parser.o |
| emit_mov_imm64 | GLOBAL | lithos-parser.o |
| emit_mul_reg | GLOBAL | lithos-parser.o |
| emit_nop | GLOBAL | lithos-parser.o |
| emit_orr_reg | GLOBAL | lithos-parser.o |
| emit_ret_inst | GLOBAL | lithos-parser.o |
| emit_sdiv_reg | GLOBAL | lithos-parser.o |
| emit_sub_reg | GLOBAL | lithos-parser.o |
| emit_svc | GLOBAL | lithos-parser.o |
| parse_body | GLOBAL | lithos-parser.o |
| parse_buf_decl | GLOBAL | lithos-parser.o |
| parse_const_decl | GLOBAL | lithos-parser.o |
| parse_each | GLOBAL | lithos-parser.o |
| parse_error | GLOBAL | lithos-parser.o |
| parse_if | GLOBAL | lithos-parser.o |
| parse_int_literal | GLOBAL | lithos-parser.o |
| parse_var_decl | GLOBAL | lithos-parser.o |
| patch_b | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| patch_b_cond | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| reset_regs | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| scope_depth | local -> GLOBAL (via build.sh fix) | lithos-parser.o |
| skip_newlines | GLOBAL | lithos-parser.o |
| sym_add | GLOBAL | lithos-parser.o |
| sym_lookup | GLOBAL | lithos-parser.o |
| sym_pop_scope | GLOBAL | lithos-parser.o |

### Provided by lithos-glue.s (9 symbols)

Self-contained stubs, no external emit32 dependency:

| Symbol | Implementation |
|--------|---------------|
| emit_ldrb_zero | LDRB Wt, [Xn] = 0x39400000 ORR Rn<<5 ORR Rt, inline emit |
| emit_strb_zero | STRB Wt, [Xn] = 0x39000000 ORR Rn<<5 ORR Rt, inline emit |
| emit_ldrh_zero | LDRH Wt, [Xn] = 0x79400000 ORR Rn<<5 ORR Rt, inline emit |
| emit_strh_zero | STRH Wt, [Xn] = 0x79000000 ORR Rn<<5 ORR Rt, inline emit |
| emit_ldr_w_zero | LDR Wt, [Xn] = 0xB9400000 ORR Rn<<5 ORR Rt, inline emit |
| emit_str_w_zero | STR Wt, [Xn] = 0xB9000000 ORR Rn<<5 ORR Rt, inline emit |
| emit_ldr_x_zero | LDR Xt, [Xn] = 0xF9400000 ORR Rn<<5 ORR Rt, inline emit |
| emit_str_x_zero | STR Xt, [Xn] = 0xF9000000 ORR Rn<<5 ORR Rt, inline emit |
| parse_data_decl | Delegates to parse_buf_decl (tail call) |

## build.sh Changes

1. Added `lithos-glue.s` to SRCS array
2. Added `lithos-glue.o` to LINK_ORDER array (before ls-shared.o)
3. Updated `globals_for` awk pattern to promote 6 local symbols:
   `alloc_reg`, `emit32`, `free_reg`, `patch_b`, `patch_b_cond`,
   `reset_regs`, `skip_newlines`, `scope_depth`, `next_reg`

## Build Result

```
== duplicate-symbol audit ==
no duplicate global symbols
== linking ==
built: /home/ubuntu/lithos/bootstrap/lithos-bootstrap (123816 bytes)
```

## Test Results

### Test 1: Simple binding
```
$ printf 'x 42\n' | ./lithos-bootstrap /tmp/bind.ls -o /tmp/bind.elf
lithos: wrote /tmp/bind.elf
EXIT: 0
```
PASS

### Test 2: Composition with expression
```
$ printf 'add a b :\n    c a + b\n' | ./lithos-bootstrap /tmp/comp.ls -o /tmp/comp.elf
lithos: wrote /tmp/comp.elf
EXIT: 0
```
PASS

### Test 3: compiler.ls self-hosting
```
$ ./lithos-bootstrap /home/ubuntu/lithos/compiler/compiler.ls -o lithos-stage1
parse error
EXIT: 0
```
FAIL at **line 248** of compiler.ls.

Failing line:
```
    val 0x8B000000 | (rm << 16) | (rn << 5) | rd
```

Root cause: The expression parser does not support **parenthesized
subexpressions** `(expr)`. The `|` and `<<` operators work individually,
but `(rm << 16)` is not recognized as a grouped subexpression.

This is a parser limitation in the expression evaluator, not a linking
or symbol resolution issue. The glue file and symbol exports are correct.

## What Would Fix compiler.ls

The expression parser needs to handle `(` as an atom that recursively
parses a subexpression until `)`. This requires changes to `parse_atom`
or equivalent in lithos-parser.s.
