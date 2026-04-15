# Plan: BL-based composition calls in the walker

## Goal

Replace the walker's inline expansion of composition references with
ARM64 BL (Branch-with-Link) function calls.  This unblocks self-hosting:
compiler.ls compiling compiler.ls currently infinite-loops because inline
expansion recurses into compositions that reference each other.

## Current state (what works)

- walk_top_level emits prologue/epilogue per composition (STP/SUB SP/...body.../ADD SP/LDP/RET)
- Bindings are stack-resident (STUR/LDUR to [FP, #-slot])
- Scratch registers rotate X9..X14
- Infix expressions work (pending_op + vpush_with_op)
- Reassignment works (if/else dispatch in newline handler)
- Simple programs compile correctly (arith test, host_main, test-exit42)

## What changes

### Step 1: Composition code-offset table

Add `buf comp_code_off_v 256` (u32 per composition, stores byte offset
in arm64_buf where each composition's code starts).

In walk_top_level, before each composition's walk_body call:
```
_ap → 64 arm64_pos_v
← 32 comp_code_off_v + i * 4 _ap
```

This records where composition i's first instruction lands.

### Step 2: Return value convention

Every composition's last expression result must end up in X0 before
the epilogue.  Add to walk_top_level, after walk_body returns:

```
if== is_host 1
    \\ Pop vstack top (if any) into X0 for caller.
    sp → 64 vstack_sp_v
    if> sp 0
        ret_reg vpop
        \\ MOV X0, Xret_reg  (if ret_reg != 0)
        if!= ret_reg 0
            movw 0xAA0003E0 | (ret_reg << 16)
            arm64_emit32 movw
```

### Step 3: BL emission helper

```
emit_a64_bl offset :
    \\ BL imm26: 0x94000000 | (offset/4 & 0x3FFFFFF)
    imm26 (offset >> 2) & 0x3FFFFFF
    val 0x94000000 | imm26
    arm64_emit32 val
```

### Step 4: Forward-reference fixup table

Compositions are emitted top-to-bottom.  A call to a composition
defined LATER in the file needs a placeholder BL that gets patched
after the target is emitted.

```
buf bl_fixup_comp_v   256   \\ u32: composition index being called
buf bl_fixup_site_v   256   \\ u32: byte offset of the placeholder BL
buf bl_fixup_count_v    8
```

When emitting a BL to composition j:
- If `comp_code_off_v[j]` is already known (>= 0): emit BL with
  correct offset.
- Otherwise: record (j, current arm64_pos) in fixup table, emit
  BL #0 as placeholder.

After walk_top_level finishes all compositions, patch loop:
```
for k in 0..bl_fixup_count:
    j    = bl_fixup_comp_v[k]
    site = bl_fixup_site_v[k]
    target = comp_code_off_v[j]
    offset = target - site
    imm26  = (offset >> 2) & 0x3FFFFFF
    patch arm64_buf[site] = 0x94000000 | imm26
```

### Step 5: Replace inline with BL in walk_body

In walk_body's IDENT handler, where it currently does:
```
cidx comp_find off len
if>= cidx 0
    cbs → 32 comp_body_start_v + cidx * 4
    cbe → 32 comp_body_end_v + cidx * 4
    saved → 64 walk_pos_v
    walk_body cbs cbe
    ← 64 walk_pos_v saved
    goto wb_loop
```

Replace with:
```
cidx comp_find off len
if>= cidx 0
    \\ Parse arguments: walk tokens until NEWLINE/operator, evaluate
    \\ each as an expression, MOV result into X0..X7.
    arg_idx 0
    label bl_arg_loop
    np → 64 walk_pos_v
    nt tok_type np
    if== nt 1
        goto bl_arg_done
    if== nt 0
        goto bl_arg_done
    if== nt 2
        goto bl_arg_done
    \\ Stop at operators (they belong to the OUTER expression, not args)
    ... operator checks ...
    \\ Evaluate one arg expression
    \\ (use the existing walk_body token walking for one atom)
    ... evaluate atom, result in scratch reg ...
    \\ MOV X<arg_idx>, Xscratch
    emit MOV X(arg_idx), Xresult
    arg_idx arg_idx + 1
    goto bl_arg_loop
    label bl_arg_done

    \\ Emit BL to composition
    target_off → 32 comp_code_off_v + cidx * 4
    if> target_off 0
        \\ Known offset: emit BL with correct delta
        current → 64 arm64_pos_v
        delta target_off - current
        emit_a64_bl delta
    else
        \\ Forward reference: record fixup, emit placeholder
        fc → 64 bl_fixup_count_v
        ← 32 bl_fixup_comp_v + fc * 4 cidx
        current → 64 arm64_pos_v
        ← 32 bl_fixup_site_v + fc * 4 current
        ← 64 bl_fixup_count_v fc + 1
        arm64_emit32 0x94000000

    \\ Result is in X0; push onto vstack
    scratch alloc_scratch
    \\ MOV Xscratch, X0
    emit MOV Xscratch, X0
    vpush_with_op scratch
    goto wb_loop
```

### Step 6: Argument passing in callee

bind_args already emits STUR Xi, [FP, #-slot] for each param.
This stores X0..X7 to frame slots.  No change needed — BL callers
put args in X0..X7, callees store them.

### Step 7: Patch fixups after all compositions emitted

At the end of walk_top_level, before returning:
```
fc → 64 bl_fixup_count_v
k 0
label patch_loop
if>= k fc
    goto patch_done
j → 32 bl_fixup_comp_v + k * 4
site → 32 bl_fixup_site_v + k * 4
target → 32 comp_code_off_v + j * 4
delta target - site
imm26 (delta >> 2) & 0x3FFFFFF
val 0x94000000 | imm26
← 32 arm64_buf + site val
k k + 1
goto patch_loop
label patch_done
```

## Test plan

1. host_main.ls (no calls) — should still work unchanged
2. arith test (bindings, no calls) — still works
3. Two-composition test:
   ```
   helper x :
       return x + 1

   host main :
       r helper 41
       ↓ $8 93
       ↓ $0 r
       trap
   ```
   Expected: BL to helper, helper returns 42 in X0, main exits 42.
4. compiler.ls on test-exit42.ls — end-to-end
5. compiler.ls on compiler.ls — self-hosting (THE test)

## What gets deleted

- The `walk_body cbs cbe` inline expansion path in the IDENT handler
- The `saved → 64 walk_pos_v` / `← 64 walk_pos_v saved` save/restore
  around inline calls

## Risks

- BL offset calculation off-by-one (byte vs instruction addressing)
- Forward reference patching writing to wrong position
- Argument evaluation consuming too many tokens (eating the operator
  after the last arg)
- Scratch register holding a call result getting clobbered by the
  next alloc_scratch before it's consumed

Each is testable with the two-composition test (#3 above).
