# Bootstrap Parser Fix Report

## Bug 1: Assignment segfault (`x = 5` -> SIGSEGV)

**Root cause:** In `parse_assignment` (lithos-parser.s), the symbol table entry
pointer (`x5`) and name token pointer were stored in caller-saved registers
(x5, x6, x7). The call to `bl parse_expr` clobbered these registers. When the
code later dereferenced x5 at `.Lassign_existing` via `ldr w1, [x5, #SYM_REG]`,
x5 contained garbage, causing the segfault.

**Fix (already in HEAD):** The HEAD commit already contains the fix: save the
sym entry and name token pointer on the stack (`stp x0, x19, [sp, #-16]!`)
before `parse_expr`, and restore them after (`ldp x5, x6, [sp], #16`). The
result register is stashed in w8 which survives the stack restore. No further
changes needed for this bug.

**Verified:** `var x 42\nx = 5\n` produces valid ELF.

## Bug 2: Composition parse error (`name args :` -> "parse error")

**Root cause:** Two separate bugs combined to cause this.

### Bug 2a: parse_body indent reload (line 1255)

`parse_body` saves the expected indent level with `str w9, [sp, #-16]!`
(pre-decrement store), which places w9 at the new `[sp]`. But on subsequent
loop iterations, the reload used `ldr w9, [sp, #-16]` -- reading 16 bytes
*below* the current sp, which is uninitialized memory. This caused the indent
comparison to use a garbage value, potentially exiting the body loop
prematurely or not at all.

**Fix:** Changed line 1255 from `ldr w9, [sp, #-16]` to `ldr w9, [sp]`.

### Bug 2b: parse_composition restoring x19 (token cursor) on exit (line 1225)

`parse_composition` saved x19 (the token cursor) at entry with
`stp x19, x20, [sp, #-16]!` and restored it at exit with
`ldp x19, x20, [sp], #16`. This rewound the token cursor to the *beginning*
of the composition definition (the name token), causing `parse_toplevel` to
re-parse the same composition infinitely -- until the scope cleanup removed
all symbols, making the body tokens unresolvable and triggering
`parse_comp_call` -> `parse_error`.

**Fix:** Changed the save/restore to only preserve x20 (the HERE pointer),
not x19:
- Line 1126: `stp x19, x20, [sp, #-16]!` -> `stp x20, xzr, [sp, #-16]!`
- Line 1225: `ldp x19, x20, [sp], #16` -> `ldp x20, xzr, [sp], #16`

This allows x19 to remain advanced past the entire composition definition
after `parse_body` returns, so `parse_toplevel` continues to the next
top-level construct.

**Verified:** `add a b :\n    c = a + b\n` produces valid ELF.

## compiler.ls status

compiler.ls fails at line 248 with "parse error" on the construct:

```
emit_add_reg rd rn rm :
    val 0x8B000000 | (rm << 16) | (rn << 5) | rd
```

The token `val` is an identifier followed by an expression (implicit
assignment: `val EXPR` means "create local variable val = EXPR"). The parser
only handles explicit assignment (`name = EXPR`) via the `TOK_EQ` check in
`parse_ident_stmt`. When it sees `IDENT` followed by anything other than `=`
or `[`, it falls through to `parse_comp_call`, treating `val` as a composition
name and the rest as arguments.

This is a missing language feature: **implicit variable binding** (assigning
the result of an expression to a new variable without `=`). The pattern is
used 131 times in compiler.ls. Implementing this requires `parse_ident_stmt`
to detect when the next token starts an expression (INT, IDENT, LPAREN, etc.)
and treat the line as an implicit assignment rather than a composition call.

## Summary of changes

File: `/home/ubuntu/lithos/bootstrap/lithos-parser.s`

| Line | Change | Bug |
|------|--------|-----|
| 1126 | `stp x19, x20` -> `stp x20, xzr` | Composition x19 rewind |
| 1225 | `ldp x19, x20` -> `ldp x20, xzr` | Composition x19 rewind |
| 1255 | `ldr w9, [sp, #-16]` -> `ldr w9, [sp]` | Body indent reload |

## Test results

| Test | Result |
|------|--------|
| `var x 42` | OK (was already working) |
| `var x 42\nx = 5` | OK (assignment to existing var) |
| `x = 5` | OK (assignment creating new var) |
| `add a b :\n    c = a + b\n` | OK (composition with body) |
| compiler.ls (full) | FAIL at line 248: implicit assignment `val EXPR` not supported |
| compiler.ls (lines 1-247) | OK |
