#!/bin/bash
# Apply all hex literal parsing fixes atomically
set -euo pipefail
cd "$(dirname "$0")"

# ============================================================
# 1. lithos-lexer.s: Add TOK_TRAP constant + keyword table entry
# ============================================================
sed -i '/^\.equ TOK_COS,.*81.*E2 89 A1$/a\.equ TOK_TRAP,      89      // trap (syscall)' lithos-lexer.s
sed -i '/^    \.ascii "host"$/a\    .byte 4, TOK_TRAP\n    .ascii "trap"' lithos-lexer.s
echo "Lexer: TOK_TRAP constant and keyword added"

# ============================================================
# 2-7. lithos-parser.s: All changes via Python (atomic)
# ============================================================
python3 << 'PYEOF'
with open('lithos-parser.s', 'r') as f:
    content = f.read()

# --- 2. Add TOK_TRAP constant ---
content = content.replace(
    '.equ TOK_COS,       81\n\n.equ TOK_STRIDE_SZ',
    '.equ TOK_COS,       81\n.equ TOK_TRAP,      89     // trap (syscall)\n\n.equ TOK_STRIDE_SZ'
)

# --- 3. Add TOK_TRAP dispatch in parse_statement ---
content = content.replace(
    '    cmp     w0, #TOK_REG_WRITE\n    b.eq    .Lstmt_reg_write\n    cmp     w0, #TOK_IDENT',
    '    cmp     w0, #TOK_REG_WRITE\n    b.eq    .Lstmt_reg_write\n    cmp     w0, #TOK_TRAP\n    b.eq    .Lstmt_trap\n    cmp     w0, #TOK_IDENT'
)

# --- 4. Add .Lstmt_trap handler before .Lstmt_ident ---
content = content.replace(
    '.Lstmt_ident:\n    bl      parse_ident_stmt',
    '''.Lstmt_trap:
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc
    ldp     x29, x30, [sp], #16
    ret

.Lstmt_ident:
    bl      parse_ident_stmt'''
)

# --- 5. Add parse_dollar_reg before parse_reg_read ---
dollar_reg_func = '''// ============================================================
// parse_dollar_reg -- parse "$N" identifier, return N in w0.
//   Expects TOK_IDENT starting with '$'. Advances x19.
// ============================================================
parse_dollar_reg:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error
    ldr     w1, [x19, #8]
    cmp     w1, #2
    b.lt    parse_error
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #36             // '$'
    b.ne    parse_error
    mov     w0, #0
    mov     w4, #1
.Ldollar_loop:
    cmp     w4, w1
    b.ge    .Ldollar_done
    ldrb    w5, [x2, x4]
    sub     w6, w5, #48         // '0'
    cmp     w6, #9
    b.hi    parse_error
    mov     w7, #10
    mul     w0, w0, w7
    add     w0, w0, w6
    add     w4, w4, #1
    b       .Ldollar_loop
.Ldollar_done:
    cmp     w0, #30
    b.hi    parse_error
    add     x19, x19, #TOK_STRIDE_SZ
    ldp     x29, x30, [sp], #16
    ret

'''

content = content.replace(
    '// ============================================================\n// parse_reg_read',
    dollar_reg_func + '// ============================================================\n// parse_reg_read',
    1  # only first occurrence
)

# --- 6. Rewrite parse_reg_read ---
old_reg_read = '''parse_reg_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '\xe2\x86\x91'

    // Expect register reference \xe2\x80\x94 could be $N (ident starting with $)
    // or a bare number. Parse as expression for flexibility.
    bl      parse_expr          // reg number in w0
    // Result register already holds the value \xe2\x80\x94 this is a conceptual
    // "read register N". In practice, for syscalls, the value is
    // already in XN after SVC. We emit a MOV to capture it.
    mov     w4, w0

    bl      alloc_reg
    mov     w5, w0              // destination reg

    // Emit: MOV Xdest, X<regnum>
    // But regnum is dynamic \xe2\x80\x94 need indirect. For static $N:
    // we'd emit MOV Xdest, XN directly. For now, treat as identity.
    // The composition system will map $0-$7 to x0-x7.
    mov     w0, w5
    mov     w1, w4
    lsl     w2, w1, #16
    mov     w3, #31
    lsl     w3, w3, #5
    orr     w4, w0, w3
    orr     w4, w4, w2
    ORRIMM    w4, 0xAA000000, w16
    mov     w0, w4
    bl      emit32

    ldp     x29, x30, [sp], #16
    ret'''

new_reg_read = '''parse_reg_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '\xe2\x86\x91'

    bl      parse_dollar_reg    // w0 = source register N
    mov     w4, w0

    bl      alloc_reg
    mov     w5, w0              // destination reg

    // Emit: MOV Xdest, XN  (ORR Xd, XZR, Xm)
    lsl     w1, w4, #16        // Rm = source hardware reg
    mov     w2, #31
    lsl     w2, w2, #5         // Rn = XZR
    orr     w3, w5, w2         // Rd = dest scratch reg
    orr     w3, w3, w1
    ORRIMM    w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

    mov     w0, w5
    ldp     x29, x30, [sp], #16
    ret'''

if old_reg_read not in content:
    # Try to find what's actually there
    import re
    m = re.search(r'parse_reg_read:.*?    ret', content, re.DOTALL)
    if m:
        print(f"parse_reg_read found at offset {m.start()}, length {len(m.group())}")
        print("First 200 chars:", repr(m.group()[:200]))
    else:
        print("parse_reg_read: NOT FOUND AT ALL")
    exit(1)

content = content.replace(old_reg_read, new_reg_read)

# --- 7. Rewrite parse_reg_write ---
old_reg_write = '''parse_reg_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '\xe2\x86\x93'

    // Parse register number
    bl      parse_expr
    mov     w4, w0              // target register

    // Parse value
    bl      parse_expr
    mov     w5, w0              // value register

    // Emit: MOV X<target>, Xvalue
    // target reg is in w4 (may be literal or resolved)
    lsl     w1, w5, #16        // Rm = value
    mov     w2, #31
    lsl     w2, w2, #5         // Rn = XZR
    orr     w3, w4, w2
    orr     w3, w3, w1
    ORRIMM    w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

    ldp     x29, x30, [sp], #16
    ret'''

new_reg_write = '''parse_reg_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '\xe2\x86\x93'

    // Parse register number ($N)
    bl      parse_dollar_reg    // w0 = target hardware reg N
    str     w0, [sp, #-16]!    // save target reg across parse_expr

    // Parse value expression
    bl      parse_expr
    mov     w5, w0              // value register (from allocator)
    ldr     w4, [sp], #16      // target hardware reg N (restored)

    // Emit: MOV X<target>, Xvalue  (ORR Xd, XZR, Xm)
    lsl     w1, w5, #16        // Rm = value
    mov     w2, #31
    lsl     w2, w2, #5         // Rn = XZR
    orr     w3, w4, w2
    orr     w3, w3, w1
    ORRIMM    w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

    ldp     x29, x30, [sp], #16
    ret'''

if old_reg_write not in content:
    import re
    m = re.search(r'parse_reg_write:.*?    ret', content, re.DOTALL)
    if m:
        print(f"parse_reg_write found at offset {m.start()}, length {len(m.group())}")
        print("First 200 chars:", repr(m.group()[:200]))
    else:
        print("parse_reg_write: NOT FOUND AT ALL")
    exit(1)

content = content.replace(old_reg_write, new_reg_write)

with open('lithos-parser.s', 'w') as f:
    f.write(content)

print("Parser: all changes applied")
PYEOF

echo "=== Verification ==="
grep -c "parse_dollar_reg" lithos-parser.s
grep -c "TOK_TRAP" lithos-parser.s
grep -c "TOK_TRAP" lithos-lexer.s
echo "Done"
