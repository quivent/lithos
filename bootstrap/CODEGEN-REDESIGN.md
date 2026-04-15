# Bootstrap codegen redesign — stack-resident bindings

## What the language is

compiler.ls has exactly three kinds of things:

- **Compositions** — named sequences of statements, arguments in, value out
- **Statements** — binding (`name expr`), memory op (`→`/`←`), register op
  (`↓`/`↑`), conditional, goto, label, composition call, return
- **Expressions** — atoms (literal, ident, buf-name) combined by operators;
  a composition call is just an atom

There is no notion of "register" or "temporary" at the language level.
There is "the result of this expression" and "this binding's value."

## What the bootstrap currently is

Three parallel codegens (`lithos-table.s`, `lithos-expr.s`, parts of
`lithos-bootstrap.s`), each incomplete. The primary one invented
registers because ARM64 has them, then invented spilling because
registers run out, then invented six variants of save/restore because
spilling is broken. Every layer is a workaround for a choice the
language never asked it to make.

Current mechanism tracked per composition:

- `next_reg`
- `reg_floor`
- `spill_count`
- which registers hold bindings vs. temporaries
- which registers to save around `bl`
- a cap at `REG_LAST` replicated in half the emitters
- a call-result move in `handle_binding`
- `emit_save_bindings` / `emit_restore_bindings` with two encodings
- pair of LDR encodings (pre- vs. post-index), fixed three times this
  session in different places

When I trace the bug I'm currently chasing back to its root, it is
always: the allocator doesn't understand that some register still holds
a value someone further up the stack cares about. That's not a bug you
fix. It is the absence of liveness analysis, in a model that needs it.

## What the bootstrap should be

One codegen. Stack-resident bindings. Registers are purely transient.

- Every binding is a stack slot, from the moment it is declared. No
  "try registers first." No threshold. No hybrid.
- Expression evaluation uses a tiny fixed set of scratch registers
  (X9..X12). Never more. The emitter is recursive descent; at each
  step the result is in one scratch, and the next step reads operands
  into other scratches.
- Function calls: args into X0..X7, BL, result in X0. No save/restore
  of bindings around calls because bindings live on the frame, not
  in caller-saved regs.

State tracked per composition: **`frame_size`**. One integer.

Everything else is mechanical emission with no state:

- Binding `name expr` → parse_expr into X9, emit `STR X9, [FP, -slot]`
- Ident use → emit `LDR X9, [FP, -slot]`
- Param `arg` → in prologue, emit `STR X<n>, [FP, -slot]`
- Return → `LDR X0, [FP, -result_slot]; RET`

Prologue: `STP X29, X30, [SP, #-16]!; MOV X29, SP; SUB SP, SP, #<size>`.
Epilogue: `ADD SP, SP, #<size>; LDP X29, X30, [SP], #16; RET`.
Frame size is known only after parsing the body, so emit the prologue's
`SUB` as a placeholder and patch it at function end (same technique
the existing bootstrap already uses for forward branches).

## What gets deleted

- `alloc_reg` / `free_reg` / `reg_floor` / `next_reg` / `spill_count`
- `emit_save_bindings` / `emit_restore_bindings` and every caller
- Spill path in `alloc_reg` (`.Lalloc_spill`) and fill path in `free_reg`
- The reg_floor-aware left-operand-reuse in binary op emitters
- `handle_binding`'s X0-move workaround
- `handle_composition`'s param-move
- The `cap at REG_LAST` guards sprinkled across multiple emitters
- `lithos-expr.s` entirely, once nothing reaches it
- Parallel allocator in `lithos-bootstrap.s`, once nothing reaches it

My estimate: `lithos-table.s` drops from 4439 lines to 1500-2000.
Total bootstrap source shrinks significantly.

## What does NOT get added

- No liveness analysis
- No `let` blocks or any language change
- No new keyword, token, or parser state
- No scope-aware binding lifetime
- No hybrid register-vs-slot allocator

## Commit plan

1. This document. Waypoint before any code changes.
2. Rip: delete `alloc_reg` / `free_reg` / `reg_floor` / spill / save-
   restore-bindings. Build breaks. Errors point at exactly what needs
   rewriting.
3. Add `frame_size` and slot allocation helpers. Rewrite
   `handle_binding` to `STR` into slot.
4. Rewrite parse_atom's ident lookup to `LDR` from slot.
5. Rewrite `handle_composition`'s prologue/epilogue with placeholder
   frame-size patch.
6. Rewrite param handling: at prologue, `STR X<n>` into slot.
7. Rewrite binary op emitters (add/sub/mul/div/shift/bitwise/compare)
   to use a 4-register scratch rotation.
8. Rewrite `handle_store`, `handle_load`, `handle_reg_write`,
   `handle_reg_read` to use scratches.
9. Delete the remaining references to the old machinery.
10. Rebuild. Run the test corpus in order:
    - `bootstrap/test-exit42.ls` (trivial)
    - minimal `host main` tests
    - stage1 on `/tmp/host_main.ls`
    - stage1 on `inference/elementwise.ls`

    First time stage1 produces a correct cubin is when we know we're
    done.

## Risk

The real risk is not performance or frame size. It is missing a corner
case in one of the rewritten emitters and producing incorrect code.
Mitigation: commit after each step, run the smallest relevant test,
revert if the test regresses.

The psychological risk is sunk-cost: 17 commits landed this session
patching the current allocator. Throwing them out is the right move.
This document is the commitment.

## Next

Start from step 2 (rip). Do not write the replacement first. Delete
the broken code, let the build errors surface exactly what needs
rewriting, then rewrite the minimum to make those errors go away.
