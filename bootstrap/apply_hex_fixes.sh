#!/bin/bash
# Apply all hex literal parsing fixes atomically
set -euo pipefail
cd "$(dirname "$0")"

# ============================================================
# 1. lithos-lexer.s: Add TOK_TRAP constant + keyword table entry
# ============================================================
# Add TOK_TRAP constant after TOK_COS line
sed -i '/^\.equ TOK_COS,.*81.*E2 89 A1$/a\.equ TOK_TRAP,      89      // trap (syscall)' lithos-lexer.s

# Add "trap" keyword to the keyword table (after "host")
sed -i '/^    \.ascii "host"$/a\    .byte 4, TOK_TRAP\n    .ascii "trap"' lithos-lexer.s

echo "Lexer: TOK_TRAP constant and keyword added"

# ============================================================
# 2. lithos-parser.s: Add TOK_TRAP constant
# ============================================================
sed -i '/^\.equ TOK_COS,       81$/a\.equ TOK_TRAP,      89     // trap (syscall)' lithos-parser.s
echo "Parser: TOK_TRAP constant added"

# ============================================================
# 3. lithos-parser.s: Add TOK_TRAP dispatch in parse_statement
# ============================================================
# Insert after the "b.eq .Lstmt_reg_write" line
sed -i '/^    b\.eq    \.Lstmt_reg_write$/a\    cmp     w0, #TOK_TRAP\n    b.eq    .Lstmt_trap' lithos-parser.s
echo "Parser: TOK_TRAP dispatch added"

# ============================================================
# 4. lithos-parser.s: Add .Lstmt_trap handler (before .Lstmt_ident)
# ============================================================
sed -i '/^\.Lstmt_ident:$/i\
.Lstmt_trap:\
    add     x19, x19, #TOK_STRIDE_SZ\
    bl      emit_svc\
    ldp     x29, x30, [sp], #16\
    ret\
' lithos-parser.s
echo "Parser: .Lstmt_trap handler added"

# ============================================================
# 5. lithos-parser.s: Add parse_dollar_reg before parse_reg_read
# ============================================================
sed -i '/^\/\/ parse_reg_read/i\
\/\/ ============================================================\
\/\/ parse_dollar_reg — parse "$N" identifier, return N in w0.\
\/\/   Expects TOK_IDENT starting with '"'"'$'"'"'. Advances x19.\
\/\/ ============================================================\
parse_dollar_reg:\
    stp     x29, x30, [sp, #-16]!\
    mov     x29, sp\
    ldr     w0, [x19]\
    cmp     w0, #TOK_IDENT\
    b.ne    parse_error\
    ldr     w1, [x19, #8]\
    cmp     w1, #2\
    b.lt    parse_error\
    ldr     w2, [x19, #4]\
    add     x2, x28, x2\
    ldrb    w3, [x2]\
    cmp     w3, #36\
    b.ne    parse_error\
    mov     w0, #0\
    mov     w4, #1\
.Ldollar_loop:\
    cmp     w4, w1\
    b.ge    .Ldollar_done\
    ldrb    w5, [x2, x4]\
    sub     w6, w5, #48\
    cmp     w6, #9\
    b.hi    parse_error\
    mov     w7, #10\
    mul     w0, w0, w7\
    add     w0, w0, w6\
    add     w4, w4, #1\
    b       .Ldollar_loop\
.Ldollar_done:\
    cmp     w0, #30\
    b.hi    parse_error\
    add     x19, x19, #TOK_STRIDE_SZ\
    ldp     x29, x30, [sp], #16\
    ret\
' lithos-parser.s
echo "Parser: parse_dollar_reg function added"

# ============================================================
# 6. lithos-parser.s: Rewrite parse_reg_read to use parse_dollar_reg
# ============================================================
# Replace the parse_expr call + surrounding comments in parse_reg_read
sed -i '/^parse_reg_read:/,/^    ldp     x29, x30, \[sp\], #16$/{
  /Expect register reference/,/already in XN after SVC/{
    /bl      parse_expr/c\    bl      parse_dollar_reg    // w0 = source reg N
    /Expect register reference/d
    /or a bare number/d
    /Result register already/d
    /read register N/d
    /already in XN after SVC/d
  }
  /But regnum is dynamic/,/The composition system will/{
    /But regnum is dynamic/d
    /we.*d emit MOV Xdest/d
    /The composition system/d
  }
  s/    mov     w0, w5$/    \/\/ Emit: MOV Xdest, XN  (ORR Xd, XZR, Xm)/
  s/    mov     w1, w4/    lsl     w1, w4, #16/
  s/    lsl     w2, w1, #16/    mov     w2, #31/
  s/    mov     w3, #31/    lsl     w2, w2, #5/
  s/    lsl     w3, w3, #5/    orr     w3, w5, w2/
  s/    orr     w4, w0, w3/    orr     w3, w3, w1/
  s/    orr     w4, w4, w2/    ORRIMM    w3, 0xAA000000, w16/
  s/    ORRIMM    w4, 0xAA000000, w16/    mov     w0, w3/
  s/    mov     w0, w4$/    bl      emit32/
}' lithos-parser.s
echo "Parser: parse_reg_read rewritten (attempted)"

# ============================================================
# 7. lithos-parser.s: Rewrite parse_reg_write to use parse_dollar_reg
# ============================================================
# This is complex with sed ranges. Use a Python helper.
python3 << 'PYEOF'
import re

with open('lithos-parser.s', 'r') as f:
    content = f.read()

# Replace parse_reg_read body
old_reg_read = '''parse_reg_read:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '↑'

    // Expect register reference — could be $N (ident starting with $)
    // or a bare number. Parse as expression for flexibility.
    bl      parse_expr          // reg number in w0
    // Result register already holds the value — this is a conceptual
    // "read register N". In practice, for syscalls, the value is
    // already in XN after SVC. We emit a MOV to capture it.
    mov     w4, w0

    bl      alloc_reg
    mov     w5, w0              // destination reg

    // Emit: MOV Xdest, X<regnum>
    // But regnum is dynamic — need indirect. For static $N:
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

    add     x19, x19, #TOK_STRIDE_SZ  // skip '↑'

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

old_reg_write = '''parse_reg_write:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    add     x19, x19, #TOK_STRIDE_SZ  // skip '↓'

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

    add     x19, x19, #TOK_STRIDE_SZ  // skip '↓'

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

if old_reg_read not in content:
    print("ERROR: old parse_reg_read not found")
    exit(1)
if old_reg_write not in content:
    print("ERROR: old parse_reg_write not found")
    exit(1)

content = content.replace(old_reg_read, new_reg_read)
content = content.replace(old_reg_write, new_reg_write)

with open('lithos-parser.s', 'w') as f:
    f.write(content)

print("Parser: parse_reg_read and parse_reg_write rewritten")
PYEOF

echo "All patches applied"
grep -c "parse_dollar_reg" lithos-parser.s
grep -c "TOK_TRAP" lithos-parser.s lithos-lexer.s
