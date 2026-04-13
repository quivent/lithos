// lithos-compose.s — Composition and binding parsing for Lithos bootstrap
//
// Parses the two core statement forms of Lithos:
//   1. Composition: name arg1 arg2 ... : (followed by indented body)
//   2. Binding:     name expr           (first token = name, rest = expression)
//
// No `=` sign. No `fn` keyword. No `->` arrow.
//
// Designed to be linked alongside lithos-parser.s (which another worker
// is rewriting). This file does NOT modify lithos-parser.s.
//
// Build:
//   aarch64-linux-gnu-as -o /tmp/compose.o lithos-compose.s
//
// Register conventions (inherited from lithos-bootstrap.s / lithos-parser.s):
//   X19 = TOKP   — pointer to current token triple (type, offset, len)
//   X27 = TOKEND — pointer past last token
//   X28 = SRC    — pointer to source buffer (for extracting text)
//   X20 = HERE   — dictionary/code-space pointer
//
// Token triple layout: [+0] u32 type, [+4] u32 offset, [+8] u32 length
//   Stride = 12 bytes.
//
// ARM64 calling convention for compositions:
//   Args in X0-X7, return in X0
//   Callee-saved: X19-X28
//   Frame: X29 (FP), X30 (LR)

// ============================================================
// Constants — must match lithos-parser.s
// ============================================================

.equ TOK_EOF,        0
.equ TOK_NEWLINE,    1
.equ TOK_INDENT,     2
.equ TOK_INT,        3
.equ TOK_FLOAT,      4
.equ TOK_IDENT,      5
.equ TOK_STRING,     6
.equ TOK_KERNEL,    11
.equ TOK_IF,        13
.equ TOK_ELSE,      14
.equ TOK_ELIF,      15
.equ TOK_FOR,       16
.equ TOK_ENDFOR,    17
.equ TOK_EACH,      18
.equ TOK_STRIDE,    19
.equ TOK_WHILE,     20
.equ TOK_RETURN,    21
.equ TOK_CONST,     22
.equ TOK_VAR,       23
.equ TOK_BUF,       24
.equ TOK_LABEL,     33
.equ TOK_HOST,      35
.equ TOK_LOAD,      36
.equ TOK_STORE,     37
.equ TOK_REG_READ,  38
.equ TOK_REG_WRITE, 39
.equ TOK_PLUS,      50
.equ TOK_MINUS,     51
.equ TOK_STAR,      52
.equ TOK_SLASH,     53
.equ TOK_EQ,        54
.equ TOK_AMP,       61
.equ TOK_PIPE,      62
.equ TOK_CARET,     63
.equ TOK_SHL,       64
.equ TOK_SHR,       65
.equ TOK_COLON,     72
.equ TOK_DATA,      82

.equ TOK_STRIDE_SZ, 12

// Symbol table entry layout
.equ SYM_SIZE,       24
.equ SYM_NAME_OFF,   0
.equ SYM_NAME_LEN,   4
.equ SYM_KIND,       8
.equ SYM_REG,       12
.equ SYM_DEPTH,     16

// Symbol kinds
.equ KIND_LOCAL_REG, 0
.equ KIND_PARAM,     1
.equ KIND_VAR,       2
.equ KIND_BUF,       3
.equ KIND_COMP,      4
.equ KIND_DATA,      5
.equ KIND_CONST,     6

// ARM64 instruction constants
.equ ARM64_NOP,      0xD503201F
.equ ARM64_RET,      0xD65F03C0
.equ ARM64_SVC_0,    0xD4000001
.equ ARM64_STP_FP_LR, 0xA9BF7BFD  // STP X29, X30, [SP, #-16]!
.equ ARM64_MOV_FP_SP,  0x910003FD  // ADD X29, SP, #0
.equ ARM64_LDP_FP_LR, 0xA8C17BFD  // LDP X29, X30, [SP], #16

.equ REG_FIRST,      9
.equ REG_LAST,       15

// ============================================================
// ORRIMM macro — OR a 32-bit immediate via scratch register
// ============================================================
.macro ORRIMM Wd, imm, Wtmp
    mov     \Wtmp, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wtmp, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
    orr     \Wd, \Wd, \Wtmp
.endm

.macro MOVI32 Wd, imm
    mov     \Wd, #((\imm) & 0xFFFF)
    .if (((\imm) >> 16) & 0xFFFF) != 0
    movk    \Wd, #(((\imm) >> 16) & 0xFFFF), lsl #16
    .endif
.endm

// ============================================================
// External symbols (from lithos-parser.s and ls-shared.s)
// ============================================================
.extern ls_sym_count
.extern ls_sym_table
.extern ls_code_buf

// These are defined in lithos-parser.s (or its rewrite) and we
// reference them. When lithos-parser.s is rewritten, these must
// remain available for linking.
.extern emit_ptr
.extern scope_depth
.extern next_reg

// Functions from lithos-parser.s we call
.extern advance
.extern skip_newlines
.extern expect
.extern tok_type
.extern tok_text
.extern tok_match_str
.extern sym_lookup
.extern sym_add
.extern sym_pop_scope
.extern alloc_reg
.extern free_reg
.extern reset_regs
.extern emit32
.extern emit_cur
.extern emit_nop
.extern emit_svc
.extern emit_ret_inst
.extern emit_mov_imm16
.extern emit_mov_imm64
.extern emit_add_reg
.extern emit_sub_reg
.extern emit_mul_reg
.extern emit_sdiv_reg
.extern parse_error
.extern parse_expr
.extern parse_if
.extern parse_for
.extern parse_each
.extern parse_var_decl
.extern parse_buf_decl
.extern parse_const_decl
.extern parse_data_decl
.extern parse_mem_load
.extern parse_mem_store
.extern parse_reg_read
.extern parse_reg_write

// ============================================================
// .text
// ============================================================

.text
.align 4

// ============================================================
// 1. parse_toplevel — main entry: parse one top-level declaration
//
//   Peek at first token:
//   - If IDENT followed by COLON on same line → composition
//   - var / const / buf / data → declaration
//   - Otherwise → binding at file scope
// ============================================================
.globl parse_toplevel_compose
parse_toplevel_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

.Ltlc_loop:
    cmp     x19, x27
    b.hs    .Ltlc_done

    bl      skip_newlines
    cmp     x19, x27
    b.hs    .Ltlc_done

    ldr     w0, [x19]

    // EOF — done
    cmp     w0, #TOK_EOF
    b.eq    .Ltlc_done

    // Declaration keywords
    cmp     w0, #TOK_VAR
    b.eq    .Ltlc_var
    cmp     w0, #TOK_CONST
    b.eq    .Ltlc_const
    cmp     w0, #TOK_BUF
    b.eq    .Ltlc_buf
    cmp     w0, #TOK_DATA
    b.eq    .Ltlc_data

    // IDENT — might be composition or binding
    cmp     w0, #TOK_IDENT
    b.eq    .Ltlc_ident

    // Skip unrecognized token
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Ltlc_loop

.Ltlc_var:
    bl      parse_var_decl
    b       .Ltlc_loop

.Ltlc_const:
    bl      parse_const_decl
    b       .Ltlc_loop

.Ltlc_buf:
    bl      parse_buf_decl
    b       .Ltlc_loop

.Ltlc_data:
    bl      parse_data_decl
    b       .Ltlc_loop

.Ltlc_ident:
    // Disambiguate: scan forward for COLON before NEWLINE/EOF
    bl      compose_scan_for_colon     // w0 = 1 if composition, 0 if not
    cbnz    w0, .Ltlc_comp
    // No colon → binding at file scope
    bl      parse_binding_compose
    b       .Ltlc_loop

.Ltlc_comp:
    bl      parse_composition_compose
    b       .Ltlc_loop

.Ltlc_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// compose_scan_for_colon — lookahead on same line for TOK_COLON
//   Returns: w0 = 1 if COLON found before NEWLINE/EOF, else 0
//   Does not advance x19.
// ============================================================
.align 4
compose_scan_for_colon:
    mov     x4, x19
.Lscan_next:
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lscan_no
    ldr     w0, [x4]
    cmp     w0, #TOK_COLON
    b.eq    .Lscan_yes
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lscan_no
    cmp     w0, #TOK_EOF
    b.eq    .Lscan_no
    b       .Lscan_next
.Lscan_yes:
    mov     w0, #1
    ret
.Lscan_no:
    mov     w0, #0
    ret

// ============================================================
// 2. parse_composition_compose — handles: name arg1 arg2 ... :
//
//   - Record composition name in symbol table (kind=COMPOSITION)
//   - Parse arg list (IDENTs before the colon)
//   - Register each arg as KIND_PARAM
//   - Emit ARM64 function prologue
//   - Parse indented body
//   - Emit epilogue
// ============================================================
.globl parse_composition_compose
.align 4
parse_composition_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!

    // Save ls_sym_count for scope cleanup
    adrp    x0, ls_sym_count
    add     x0, x0, :lo12:ls_sym_count
    ldr     w4, [x0]
    stp     x4, xzr, [sp, #-16]!

    // Increment scope depth
    adrp    x0, scope_depth
    add     x0, x0, :lo12:scope_depth
    ldr     w5, [x0]
    add     w6, w5, #1
    str     w6, [x0]

    // Current token = composition name (TOK_IDENT)
    // Get current emit address — this is where the composition code starts
    bl      emit_cur
    mov     x7, x0

    // Add composition to symbol table (KIND_COMP)
    mov     w1, #KIND_COMP
    mov     w2, #0                  // placeholder, patched below
    mov     w3, w6                  // scope_depth
    bl      sym_add
    // Store the code address in the sym entry's reg field
    str     w7, [x0, #SYM_REG]

    add     x19, x19, #TOK_STRIDE_SZ   // skip composition name

    // Emit function prologue: STP X29, X30, [SP, #-16]!
    MOVI32  w0, ARM64_STP_FP_LR
    bl      emit32
    // MOV X29, SP  (ADD X29, SP, #0)
    MOVI32  w0, ARM64_MOV_FP_SP
    bl      emit32

    // Reset register allocator for this composition
    bl      reset_regs

    // Parse parameters — IDENTs before the colon
    mov     w8, #0                  // param index (maps to X0, X1, ...)
.Lcc_params:
    ldr     w0, [x19]
    cmp     w0, #TOK_COLON
    b.eq    .Lcc_colon

    // Must be an IDENT
    cmp     w0, #TOK_IDENT
    b.ne    .Lcc_param_err

    // Register this param in symbol table
    mov     w1, #KIND_PARAM
    mov     w2, w8                  // register = param index (X0-X7)
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add

    add     w8, w8, #1
    add     x19, x19, #TOK_STRIDE_SZ   // skip param name
    b       .Lcc_params

.Lcc_param_err:
    b       parse_error

.Lcc_colon:
    add     x19, x19, #TOK_STRIDE_SZ   // skip ':'

    // Skip newline(s) after colon
    bl      skip_newlines

    // Determine body indent level
    ldr     w0, [x19]
    cmp     w0, #TOK_INDENT
    b.ne    .Lcc_indent_default
    ldr     w9, [x19, #8]          // indent count from INDENT token
    b       .Lcc_body

.Lcc_indent_default:
    mov     w9, #4                  // default indent

.Lcc_body:
    bl      parse_body_compose

    // Emit function epilogue: LDP X29, X30, [SP], #16
    MOVI32  w0, ARM64_LDP_FP_LR
    bl      emit32
    // RET
    bl      emit_ret_inst

    // Pop scope
    adrp    x0, scope_depth
    add     x0, x0, :lo12:scope_depth
    ldr     w1, [x0]
    sub     w1, w1, #1
    str     w1, [x0]
    mov     w0, w1
    bl      sym_pop_scope

    // Drop saved old sym count
    add     sp, sp, #16

    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 3. parse_binding_compose — handles: name expr (single line)
//
//   First token = name (IDENT)
//   Rest of line = expression
//   Call parse_expr to evaluate expression
//   Register the name in symbol table mapped to result register
//   No code emission for the binding itself — just the expression
// ============================================================
.globl parse_binding_compose
.align 4
parse_binding_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Current token must be IDENT
    ldr     w0, [x19]
    cmp     w0, #TOK_IDENT
    b.ne    parse_error

    // Save pointer to the name token (x19 points into token buffer)
    // We need this after parse_expr to call sym_add.
    str     x19, [sp, #-16]!       // save name token pointer

    // Skip the name — advance to expression
    add     x19, x19, #TOK_STRIDE_SZ

    // Parse the expression (rest of line)
    // Returns result register number in w0
    bl      parse_expr

    // w0 = register holding expression result
    // x19 is now PAST the expression tokens — do NOT rewind it
    mov     w4, w0                  // stash result reg

    // Save post-expr token position, restore x19 to name token for sym_add
    ldr     x5, [sp]               // x5 = saved name token pointer
    str     x19, [sp]              // overwrite with post-expr position
    mov     x19, x5                // x19 = name token (sym_add reads from x19)

    // Register the binding: name → result register
    mov     w1, #KIND_LOCAL_REG
    mov     w2, w4                  // register number
    adrp    x3, scope_depth
    add     x3, x3, :lo12:scope_depth
    ldr     w3, [x3]
    bl      sym_add

    // Restore x19 to post-expr position (past the expression)
    ldr     x19, [sp], #16

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 4. parse_body_compose — parse an indented block
//
//   w9 = minimum indent level for body membership
//   Loop: parse statements until indent decreases or EOF
// ============================================================
.globl parse_body_compose
.align 4
parse_body_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    str     w9, [sp, #-4]!         // save expected indent (4 bytes)
    sub     sp, sp, #12             // align to 16

.Lbc_loop:
    cmp     x19, x27
    b.hs    .Lbc_done

    ldr     w0, [x19]
    cmp     w0, #TOK_EOF
    b.eq    .Lbc_done

    // Skip bare newlines
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lbc_skip_nl

    // Check indent
    cmp     w0, #TOK_INDENT
    b.ne    .Lbc_stmt

    // Read indent level and compare
    ldr     w1, [x19, #8]
    ldr     w9, [sp, #12]          // reload expected indent
    cmp     w1, w9
    b.lt    .Lbc_done               // dedent — body is over

    // Consume the INDENT token
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lbc_loop

.Lbc_skip_nl:
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lbc_loop

.Lbc_stmt:
    bl      parse_stmt_compose
    bl      reset_regs
    b       .Lbc_loop

.Lbc_done:
    add     sp, sp, #16             // drop saved indent + alignment
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// 5. parse_stmt_compose — dispatch one statement within a body
//
//   Check first token:
//     TOK_STORE (←)    → parse memory store
//     TOK_REG_WRITE (↓) → parse register write
//     TOK_FOR          → parse for loop
//     TOK_IF           → parse conditional
//     TOK_EACH         → parse each
//     TOK_WHILE        → parse while
//     TOK_RETURN       → parse return
//     TOK_VAR          → var declaration
//     TOK_BUF          → buf declaration
//     TOK_LOAD (→)     → parse memory load  (at stmt level: load into binding)
//     TOK_REG_READ (↑) → parse register read
//     TOK_IDENT        → binding or composition call (disambiguate)
//     Unknown          → skip
// ============================================================
.globl parse_stmt_compose
.align 4
parse_stmt_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    ldr     w0, [x19]

    cmp     w0, #TOK_STORE
    b.eq    .Lsc_store
    cmp     w0, #TOK_REG_WRITE
    b.eq    .Lsc_reg_write
    cmp     w0, #TOK_FOR
    b.eq    .Lsc_for
    cmp     w0, #TOK_IF
    b.eq    .Lsc_if
    cmp     w0, #TOK_EACH
    b.eq    .Lsc_each
    cmp     w0, #TOK_WHILE
    b.eq    .Lsc_while
    cmp     w0, #TOK_RETURN
    b.eq    .Lsc_return
    cmp     w0, #TOK_VAR
    b.eq    .Lsc_var
    cmp     w0, #TOK_BUF
    b.eq    .Lsc_buf
    cmp     w0, #TOK_LOAD
    b.eq    .Lsc_load
    cmp     w0, #TOK_REG_READ
    b.eq    .Lsc_reg_read
    cmp     w0, #TOK_IDENT
    b.eq    .Lsc_ident

    // Unknown — skip one token
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lsc_done

.Lsc_store:
    bl      parse_mem_store
    b       .Lsc_done
.Lsc_reg_write:
    bl      parse_reg_write
    b       .Lsc_done
.Lsc_for:
    bl      parse_for
    b       .Lsc_done
.Lsc_if:
    bl      parse_if
    b       .Lsc_done
.Lsc_each:
    bl      parse_each
    b       .Lsc_done
.Lsc_while:
    // For now, delegate (external)
    add     x19, x19, #TOK_STRIDE_SZ
    b       .Lsc_done
.Lsc_return:
    // Return statement — evaluate expr, move to X0, emit epilogue
    add     x19, x19, #TOK_STRIDE_SZ   // skip 'return' keyword
    bl      parse_return_compose
    b       .Lsc_done
.Lsc_var:
    bl      parse_var_decl
    b       .Lsc_done
.Lsc_buf:
    bl      parse_buf_decl
    b       .Lsc_done
.Lsc_load:
    bl      parse_mem_load
    b       .Lsc_done
.Lsc_reg_read:
    bl      parse_reg_read
    b       .Lsc_done
.Lsc_ident:
    bl      parse_ident_compose
    b       .Lsc_done

.Lsc_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_return_compose — return expr
//   Evaluate expression, move result to X0
// ============================================================
.align 4
parse_return_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Check if there's an expression or bare return
    cmp     x19, x27
    b.hs    .Lret_bare
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lret_bare
    cmp     w0, #TOK_EOF
    b.eq    .Lret_bare

    // Parse expression
    bl      parse_expr              // result in w0

    // If result is not in X0, emit MOV X0, Xresult
    cbz     w0, .Lret_emit_ep      // already in X0
    // Emit: ORR X0, XZR, Xresult
    lsl     w1, w0, #16            // Rm
    mov     w2, #31
    lsl     w2, w2, #5             // Rn = XZR
    orr     w3, wzr, w2            // Rd = X0 (0)
    orr     w3, w3, w1
    ORRIMM  w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

.Lret_emit_ep:
    // Emit epilogue: LDP + RET
    MOVI32  w0, ARM64_LDP_FP_LR
    bl      emit32
    bl      emit_ret_inst
    b       .Lret_done

.Lret_bare:
    // Bare return — just emit epilogue
    MOVI32  w0, ARM64_LDP_FP_LR
    bl      emit32
    bl      emit_ret_inst

.Lret_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// parse_ident_compose — IDENT at start of statement
//
//   Disambiguation between binding and call:
//   - If second token is an operator (+, -, *, /, |, &, ^, <<, >>),
//     it's a binding: name = expr
//   - If second token is just IDENT/INT/FLOAT with no operator
//     following it on the same line, it's a call
//   - Special case: "trap" → emit SVC #0
// ============================================================
.align 4
parse_ident_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Check for "trap" (4 bytes: t r a p)
    ldr     w1, [x19, #8]
    cmp     w1, #4
    b.ne    .Lpic_not_trap
    ldr     w2, [x19, #4]
    add     x2, x28, x2
    ldrb    w3, [x2]
    cmp     w3, #'t'
    b.ne    .Lpic_not_trap
    ldrb    w3, [x2, #1]
    cmp     w3, #'r'
    b.ne    .Lpic_not_trap
    ldrb    w3, [x2, #2]
    cmp     w3, #'a'
    b.ne    .Lpic_not_trap
    ldrb    w3, [x2, #3]
    cmp     w3, #'p'
    b.ne    .Lpic_not_trap
    // It's "trap"
    add     x19, x19, #TOK_STRIDE_SZ
    bl      emit_svc
    ldp     x29, x30, [sp], #16
    ret

.Lpic_not_trap:
    // Peek at second token
    mov     x4, x19
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lpic_call              // no second token → bare call/no-op

    ldr     w0, [x4]

    // If second token is on next line, this is a bare name (call with 0 args)
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lpic_call
    cmp     w0, #TOK_EOF
    b.eq    .Lpic_call

    // Check if second token is an operator → binding
    bl      compose_is_operator     // w0 = token type to check; returns w0=1 if op
    cbnz    w0, .Lpic_binding

    // Second token is not an operator.
    // Now check if there's an operator AFTER the second token (peek third)
    mov     x4, x19
    add     x4, x4, #TOK_STRIDE_SZ
    add     x4, x4, #TOK_STRIDE_SZ
    cmp     x4, x27
    b.hs    .Lpic_call              // only 2 tokens on line → call

    ldr     w0, [x4]
    // If third token is operator, it's a binding: "x y + z"
    bl      compose_is_operator
    cbnz    w0, .Lpic_binding

    // No operator → it's a composition call
    b       .Lpic_call

.Lpic_binding:
    bl      parse_binding_compose
    ldp     x29, x30, [sp], #16
    ret

.Lpic_call:
    bl      parse_call_compose
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// compose_is_operator — check if token type in w0 is an operator
//   w0 = token type on entry
//   Returns w0 = 1 if operator, 0 otherwise
// ============================================================
.align 4
compose_is_operator:
    // Operators: TOK_PLUS(50) through TOK_SHR(65) covers +,-,*,/,==,!=,<,>,<=,>=,&,|,^,<<,>>
    cmp     w0, #TOK_PLUS
    b.lt    .Lcio_no
    cmp     w0, #TOK_SHR
    b.gt    .Lcio_no
    mov     w0, #1
    ret
.Lcio_no:
    mov     w0, #0
    ret

// ============================================================
// 6. parse_call_compose — composition call: name arg1 arg2 ...
//
//   Look up name in symbol table → get code address
//   Evaluate each arg → put in X0, X1, X2, ...
//   Emit BL <address>
// ============================================================
.globl parse_call_compose
.align 4
parse_call_compose:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Look up composition name
    bl      sym_lookup
    mov     x5, x0                  // sym entry (0 if not found)
    add     x19, x19, #TOK_STRIDE_SZ   // skip name

    // Parse arguments until end of line
    mov     w8, #0                  // arg index

.Lccc_args:
    cmp     x19, x27
    b.hs    .Lccc_emit
    ldr     w0, [x19]
    cmp     w0, #TOK_NEWLINE
    b.eq    .Lccc_emit
    cmp     w0, #TOK_EOF
    b.eq    .Lccc_emit
    cmp     w0, #TOK_INDENT
    b.eq    .Lccc_emit

    // Parse one arg expression
    stp     x8, x5, [sp, #-16]!
    bl      parse_expr              // result reg in w0
    ldp     x8, x5, [sp], #16

    // Move result into arg register Xn (n = w8)
    cmp     w0, w8
    b.eq    .Lccc_arg_next

    // Emit MOV Xarg, Xresult  (ORR Xd, XZR, Xm)
    lsl     w1, w0, #16            // Rm = result reg
    mov     w2, #31
    lsl     w2, w2, #5             // Rn = XZR
    orr     w3, w8, w2             // Rd = arg index
    orr     w3, w3, w1
    ORRIMM  w3, 0xAA000000, w16
    mov     w0, w3
    bl      emit32

.Lccc_arg_next:
    add     w8, w8, #1
    b       .Lccc_args

.Lccc_emit:
    // Emit BL to the composition
    cbz     x5, .Lccc_unknown

    // Known composition — compute BL offset
    bl      emit_cur
    mov     x1, x0                  // current emit address
    ldr     w0, [x5, #SYM_REG]     // target code address
    sub     x2, x0, x1             // byte offset
    asr     x2, x2, #2             // instruction offset (÷4)
    and     w2, w2, #0x3FFFFFF      // 26-bit immediate
    ORRIMM  w2, 0x94000000, w16    // BL opcode
    mov     w0, w2
    bl      emit32
    b       .Lccc_done

.Lccc_unknown:
    // Unknown composition — emit NOP placeholder for late binding / second pass
    bl      emit_nop

.Lccc_done:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Lithos-compose .bss section
// ============================================================
// (No private .bss needed — all state is in lithos-parser.s globals)

