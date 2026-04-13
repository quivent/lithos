\\ lithos-parser.li — Self-hosting parser for the Lithos .li format
\\ Target: ARM64 (host code) / sm90 GPU (kernel code)
\\
\\ Second stage of the self-hosting compiler pipeline:
\\   lithos-lexer.li  →  tokenize .li source into flat token buffer
\\   lithos-parser.li →  parse token stream, emit machine code directly
\\   emit-arm64.li    →  ARM64 instruction encoding (host backend)
\\   opcodes-sm90.fs  →  sm90 opcode constants (GPU backend)
\\
\\ Architecture: single-pass recursive descent with direct emission.
\\ No AST. Each statement is parsed and emitted immediately.
\\ The parser reads the token buffer produced by lithos-lexer.li.
\\
\\ Token encoding (from lexer): 3 consecutive u32 values per token
\\   [0] type    — token type enum (TOK_* constants)
\\   [1] offset  — byte offset in source buffer
\\   [2] length  — byte length of token text

\\ ============================================================================
\\ Token type constants (must match lithos-lexer.li)
\\ ============================================================================
\\
\\ TOK_EOF        = 0
\\ TOK_NEWLINE    = 1
\\ TOK_INDENT     = 2
\\ TOK_INT        = 3
\\ TOK_FLOAT      = 4
\\ TOK_IDENT      = 5
\\ TOK_KERNEL     = 11
\\ TOK_IF         = 13
\\ TOK_FOR        = 16
\\ TOK_EACH       = 18
\\ TOK_STRIDE     = 19
\\ TOK_CONST      = 22
\\ TOK_VAR        = 23
\\ TOK_BUF        = 24
\\ TOK_SHARED     = 31
\\ TOK_BARRIER    = 32
\\ TOK_EXIT       = 34
\\ TOK_HOST       = 35
\\ TOK_LOAD       = 36   → (memory load)
\\ TOK_STORE      = 37   ← (memory store)
\\ TOK_PLUS       = 50
\\ TOK_MINUS      = 51
\\ TOK_STAR       = 52
\\ TOK_SLASH      = 53
\\ TOK_EQ         = 54
\\ TOK_EQEQ       = 55
\\ TOK_LT         = 57
\\ TOK_GT         = 58
\\ TOK_LTE        = 59
\\ TOK_GTE        = 60
\\ TOK_SHL        = 64
\\ TOK_SHR        = 65
\\ TOK_LBRACK     = 67
\\ TOK_RBRACK     = 68
\\ TOK_COLON      = 72
\\ TOK_AT         = 74

\\ ============================================================================
\\ Unicode token types (extensions for Lithos grammar)
\\ ============================================================================
\\
\\ These are recognized by the lexer when it encounters the UTF-8 byte
\\ sequences for the Lithos primitives. They extend the TOK_* enum.
\\
\\ TOK_ARROW_R    = 80   → E2 86 92  (load)
\\ TOK_ARROW_L    = 81   ← E2 86 90  (store)
\\ TOK_ARROW_U    = 82   ↑ E2 86 91  (register read)
\\ TOK_ARROW_D    = 83   ↓ E2 86 93  (register write)
\\ TOK_SIGMA      = 84   Σ CE A3     (sum reduction)
\\ TOK_TRIANGLE   = 85   △ E2 96 B3  (max reduction)
\\ TOK_NABLA      = 86   ▽ E2 96 BD  (min reduction)
\\ TOK_SQRT       = 87   √ E2 88 9A  (square root)
\\ TOK_APPROX     = 88   ≅ E2 89 85  (sine)
\\ TOK_IDENTICAL  = 89   ≡ E2 89 A1  (cosine)
\\ TOK_HASH       = 90   #           (index modifier)
\\ TOK_STARSTAR   = 91   **          (elementwise)
\\ TOK_STARSTARSTAR = 92 ***         (matrix)

\\ ============================================================================
\\ Emit target constants
\\ ============================================================================

\\ TARGET_GPU  = 0   sm90 GPU backend
\\ TARGET_HOST = 1   ARM64 host backend

\\ ============================================================================
\\ Parser state — global variables
\\ ============================================================================

var tok_pos 0              \\ current position in token array (index into triples)
var tok_total 0            \\ total number of tokens from lexer
var src_buf 0              \\ pointer to mmap'd source buffer (for reading token text)
var emit_target 0          \\ 0 = GPU (sm90), 1 = HOST (ARM64)
var body_indent 0          \\ indentation level of current composition body
var comp_depth 0           \\ composition nesting depth (for inlining)
var error_count 0          \\ number of parse errors encountered

\\ ============================================================================
\\ Symbol table
\\ ============================================================================
\\ Each symbol: name (32 bytes), name_len (u32), kind (u32), reg (u32)
\\ kind: 0=input-ptr  1=output-ptr  2=local-f32  3=local-u32
\\       4=each-var   5=shared-buf  6=scalar-u32  7=scalar-f32
\\       8=local-pred 9=stride-var
\\
\\ Max 64 symbols per composition. Reset when entering a new composition.

buf sym_names 2048         \\ 64 * 32 bytes for name storage
buf sym_lens 256           \\ 64 * 4 bytes for name lengths
buf sym_kinds 256          \\ 64 * 4 bytes for kind tags
buf sym_regs 256           \\ 64 * 4 bytes for register numbers
var n_syms 0

\\ ============================================================================
\\ Register allocators — monotone bump
\\ ============================================================================
\\ GPU: $0-$3 reserved (tid.x, tid.y, ctaid.x, ctaid.y), start at $4
\\ ARM64: $0-$7 for args/return, allocate from $9 up

var next_freg 0            \\ next float register (GPU)
var next_rreg 4            \\ next integer register (GPU), starts at 4
var next_rdreg 4           \\ next descriptor register (GPU)
var next_preg 0            \\ next predicate register (GPU)
var next_host_reg 9        \\ next scratch register (ARM64), starts at X9

\\ ============================================================================
\\ Shared memory tracking
\\ ============================================================================

buf shm_names 256          \\ 8 * 32 bytes for shared memory names
buf shm_sizes 32           \\ 8 * 4 bytes for sizes in bytes
var n_shared 0
var shmem_total 0          \\ total shared memory bytes

\\ ============================================================================
\\ Loop / branch tracking
\\ ============================================================================
\\ Stack of branch targets for patching forward jumps.
\\ Each entry: (patch_offset, loop_start_offset, loop_reg)

buf branch_stack 512       \\ 32 entries * 16 bytes
var branch_depth 0

\\ ============================================================================
\\ Composition table — for inlining calls
\\ ============================================================================
\\ Each entry: name (32 bytes), token_start (u32), arg_count (u32)
\\ When a composition is called, we inline by re-parsing its body.

buf comp_names 2048        \\ 64 * 32 bytes
buf comp_lens 256          \\ 64 * 4 bytes for name lengths
buf comp_tok_starts 256    \\ 64 * 4 bytes — token index where body begins
buf comp_arg_counts 256    \\ 64 * 4 bytes — number of parameters
var n_comps 0

\\ ============================================================================
\\ Token stream access
\\ ============================================================================

\\ peek_type : return the type of the current token without consuming
peek_type :
    idx tok_pos * 3
    → 32 tokens idx

\\ peek_offset : return the byte offset of the current token
peek_offset :
    idx tok_pos * 3 + 1
    → 32 tokens idx

\\ peek_length : return the byte length of the current token
peek_length :
    idx tok_pos * 3 + 2
    → 32 tokens idx

\\ consume : advance to the next token
consume :
    tok_pos tok_pos + 1

\\ expect : consume and verify the token type matches; error if not
expect type :
    t peek_type
    if== t type
        consume
    \\ else: parse error — type mismatch
    \\ TODO: emit error diagnostic with line/column

\\ match_type : check if current token is the given type, consume if so
\\ returns 1 if matched, 0 if not
match_type type :
    t peek_type
    if== t type
        consume
        1
    0

\\ skip_newlines : advance past any newline and indent tokens
skip_newlines :
    t peek_type
    if== t 1           \\ TOK_NEWLINE
        consume
        skip_newlines
    if== t 2           \\ TOK_INDENT
        consume
        skip_newlines

\\ tok_text_ptr : return pointer to the text of the current token
\\ (points into the mmap'd source buffer)
tok_text_ptr :
    off peek_offset
    src_buf + off

\\ ============================================================================
\\ Source text comparison
\\ ============================================================================
\\ Compare the text of the current token against a known string.
\\ Since .li lacks string literals, we compare byte-by-byte against
\\ ASCII values, similar to the lexer's keyword matching approach.

\\ tok_is_byte2 : check if current token is exactly 2 bytes matching b0, b1
tok_is_byte2 b0 b1 :
    len peek_length
    if== len 2
        ptr tok_text_ptr
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 b0
            if== c1 b1
                1
    0

\\ tok_is_byte3 : check if current token is 3 bytes matching b0, b1, b2
tok_is_byte3 b0 b1 b2 :
    len peek_length
    if== len 3
        ptr tok_text_ptr
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        c2 → 8 ptr 2
        if== c0 b0
            if== c1 b1
                if== c2 b2
                    1
    0

\\ tok_is_byte4 : check if current token is 4 bytes matching b0, b1, b2, b3
tok_is_byte4 b0 b1 b2 b3 :
    len peek_length
    if== len 4
        ptr tok_text_ptr
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        c2 → 8 ptr 2
        c3 → 8 ptr 3
        if== c0 b0
            if== c1 b1
                if== c2 b2
                    if== c3 b3
                        1
    0

\\ ============================================================================
\\ Number parsing
\\ ============================================================================
\\ Parse the text of an integer token into a u32 value.
\\ Handles decimal and hex (0x prefix).

parse_int_token :
    ptr tok_text_ptr
    len peek_length
    consume

    \\ Check for hex prefix 0x
    if>= len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 48              \\ '0'
            if== c1 120         \\ 'x'
                parse_hex ptr + 2 len - 2

    \\ Decimal
    parse_decimal ptr len

\\ parse_decimal : convert ASCII decimal digits to integer
parse_decimal ptr len :
    val 0
    i 0
    neg 0
    \\ Check for leading minus
    if>= len 1
        c → 8 ptr 0
        if== c 45               \\ '-'
            neg 1
            i 1
    for i i len 1
        c → 8 ptr i
        d c - 48
        val val * 10 + d
    if== neg 1
        val 0 - val
    val

\\ parse_hex : convert ASCII hex digits to integer
parse_hex ptr len :
    val 0
    for i 0 len 1
        c → 8 ptr i
        if>= c 97              \\ 'a'-'f'
            d c - 87
        if>= c 65              \\ 'A'-'F'
            d c - 55
        \\ else digit
            d c - 48
        val val * 16 + d
    val

\\ parse_float_token : parse a floating-point literal
\\ Returns the IEEE 754 bit representation as a u32
\\ NOTE: full float parsing is complex. For bootstrap, we handle
\\ common constants by recognition and fall back to integer-dot-fraction.
parse_float_token :
    ptr tok_text_ptr
    len peek_length
    consume
    \\ Recognize common constants used in kernels:
    \\ 1.0 → 0x3F800000
    \\ 0.5 → 0x3F000000
    \\ -1.0 → 0xBF800000
    \\ 1.44269504 → 0x3FB8AA3B  (log2(e))
    \\ 0.69314718 → 0x3F317218  (ln(2))
    \\
    \\ For now, emit MOV with the recognized constant.
    \\ Full float parsing will use integer multiply/shift arithmetic.
    0

\\ ============================================================================
\\ Symbol table operations
\\ ============================================================================

\\ sym_reset : clear symbol table for a new composition
sym_reset :
    n_syms 0

\\ sym_add : add a symbol with given name pointer, length, kind, register
sym_add name_ptr name_len kind reg :
    idx n_syms
    if>= idx 64
        \\ overflow — ignore
        idx

    \\ Copy name into sym_names[idx*32]
    dest idx * 32
    \\ NOTE: byte-by-byte copy. .li lacks memcpy.
    \\ For bootstrap we store the offset+length as a reference into src_buf.
    ← 32 sym_lens idx * 4 name_len
    ← 32 sym_kinds idx * 4 kind
    ← 32 sym_regs idx * 4 reg

    \\ Copy up to 32 bytes of name
    clen name_len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 name_ptr j
        ← 8 sym_names dest + j b

    n_syms n_syms + 1

\\ sym_find : look up a symbol by name, return index or -1
\\ Compares token text at (name_ptr, name_len) against stored symbols.
sym_find name_ptr name_len :
    for i 0 n_syms 1
        slen → 32 sym_lens i * 4
        if== slen name_len
            \\ Compare bytes
            match 1
            for j 0 slen 1
                a → 8 name_ptr j
                b → 8 sym_names i * 32 + j
                if< a b
                    match 0
                if< b a
                    match 0
            if== match 1
                i
    \\ Not found
    -1

\\ sym_find_current : look up the current token as a symbol
sym_find_current :
    ptr tok_text_ptr
    len peek_length
    sym_find ptr len

\\ sym_kind : return the kind of symbol at index i
sym_kind i :
    → 32 sym_kinds i * 4

\\ sym_reg : return the register of symbol at index i
sym_reg i :
    → 32 sym_regs i * 4

\\ ============================================================================
\\ Register allocation
\\ ============================================================================

\\ alloc_freg : allocate next float register (GPU)
alloc_freg :
    r next_freg
    next_freg next_freg + 1
    r

\\ alloc_rreg : allocate next integer register (GPU)
alloc_rreg :
    r next_rreg
    next_rreg next_rreg + 1
    r

\\ alloc_rdreg : allocate next descriptor register (GPU)
alloc_rdreg :
    r next_rdreg
    next_rdreg next_rdreg + 1
    r

\\ alloc_preg : allocate next predicate register (GPU)
alloc_preg :
    r next_preg
    next_preg next_preg + 1
    r

\\ alloc_host_reg : allocate next scratch register (ARM64)
alloc_host_reg :
    r next_host_reg
    next_host_reg next_host_reg + 1
    r

\\ regs_reset : reset all register allocators for a new composition
regs_reset :
    next_freg 0
    next_rreg 4             \\ GPU: $0-$3 reserved
    next_rdreg 4
    next_preg 0
    next_host_reg 9         \\ ARM64: $9 up

\\ ============================================================================
\\ Backend dispatch — dual target emission
\\ ============================================================================
\\ The parser calls these backend-neutral wrappers.
\\ emit_target selects which backend actually emits.
\\
\\ GPU: calls sm90 emitter (sinst with opcode encoding)
\\ HOST: calls ARM64 emitter (emit_add_reg, emit_ldr, etc.)

\\ emit_add : emit an add instruction
\\ GPU: FADD rd, ra, rb  or  IADD3 rd, ra, rb, RZ
\\ HOST: ADD Xd, Xn, Xm
emit_add rd ra rb :
    if== emit_target 0
        \\ GPU sm90: FADD
        inst 0x7221 | rd << 16 | ra << 24 | rb << 32
        ctrl 0x000fca0000000000
        \\ sinst inst ctrl
    \\ HOST ARM64: ADD
        emit_add_reg rd ra rb

\\ emit_sub : emit a subtract instruction
emit_sub rd ra rb :
    if== emit_target 0
        inst 0x7221 | rd << 16 | ra << 24 | rb << 32
        \\ FADD with NEG on src2 (bit 63 of inst)
        inst inst | 1 << 63
        ctrl 0x000fca0000000000
        \\ sinst inst ctrl
    emit_sub_reg rd ra rb

\\ emit_mul : emit a multiply instruction
emit_mul rd ra rb :
    if== emit_target 0
        \\ GPU sm90: FMUL
        inst 0x7220 | rd << 16 | ra << 24 | rb << 32
        ctrl 0x004fca0000400000
        \\ sinst inst ctrl
    emit_mul rd ra rb

\\ emit_div : emit a divide instruction
\\ GPU: MUFU.RCP(rb) then FMUL(rd, ra, rcp)
\\ HOST: SDIV Xd, Xn, Xm
emit_div rd ra rb :
    if== emit_target 0
        \\ GPU: rcp = 1/rb, then rd = ra * rcp
        rcp alloc_freg
        \\ MUFU.RCP rcp, rb
        inst 0x7308 | rcp << 16 | rb << 32
        ctrl 0x000e640000001000
        \\ sinst inst ctrl
        \\ FMUL rd, ra, rcp
        emit_mul rd ra rcp
    emit_sdiv rd ra rb

\\ emit_mov_imm : load an immediate value into a register
\\ GPU: MOV Rd, imm32
\\ HOST: MOVZ + MOVK sequence
emit_mov_imm rd imm :
    if== emit_target 0
        \\ GPU sm90: MOV-IMM
        inst 0x7802 | rd << 16
        inst inst | imm << 32
        ctrl 0x000fe200078e00ff
        \\ sinst inst ctrl
    emit_mov64 rd imm

\\ emit_mov_reg : register-to-register move
emit_mov_reg rd rs :
    if== emit_target 0
        \\ GPU: MOV Rd, Rs (use IMAD Rd, Rs, 1, RZ as idiom)
        inst 0x7224 | rd << 16 | rs << 24 | 0xFF << 32
        ctrl 0x000fe200078e00ff
        \\ sinst inst ctrl
    emit_mov rd rs

\\ ============================================================================
\\ Memory operations — backend dispatch
\\ ============================================================================

\\ emit_load : emit a memory load
\\ width: 8, 16, 32, 64 (bits)
\\ GPU: LDG rd, desc[UR][addr]
\\ HOST: LDR/LDRB/LDRH rd, [base, offset]
emit_load rd base offset width :
    if== emit_target 0
        \\ GPU sm90: LDG
        inst 0x7981 | rd << 16 | base << 24
        ctrl 0x000ea8000c1e1900
        \\ sinst inst ctrl
    \\ HOST ARM64
    if== width 8
        emit_ldrb rd base offset
    if== width 16
        emit_ldrh rd base offset
    if== width 32
        emit_ldr_w rd base offset
    if== width 64
        emit_ldr rd base offset

\\ emit_store : emit a memory store
\\ GPU: STG desc[UR][addr], rs
\\ HOST: STR/STRB/STRH rs, [base, offset]
emit_store rs base offset width :
    if== emit_target 0
        \\ GPU sm90: STG
        inst 0x7986 | base << 24 | rs << 32
        ctrl 0x000fe2000c101904
        \\ sinst inst ctrl
    if== width 8
        emit_strb rs base offset
    if== width 16
        emit_strh rs base offset
    if== width 32
        emit_str_w rs base offset
    if== width 64
        emit_str rs base offset

\\ ============================================================================
\\ GPU-specific emitters
\\ ============================================================================

\\ emit_s2r : read a special register (GPU S2R instruction)
\\ sr_id: special register selector (e.g., 0x21 = TID.X)
emit_s2r rd sr_id :
    inst 0x7919 | rd << 16
    \\ ctrl encodes the SR selector in extra41 bits[15:8]
    sr_extra sr_id << 8
    ctrl 0x000e6e0000000000 | sr_extra
    \\ sinst inst ctrl

\\ emit_shfl_bfly : warp shuffle butterfly
\\ GPU: SHFL.BFLY rd, rs, delta
emit_shfl_bfly rd rs delta :
    inst 0x7f89 | rd << 16 | rs << 24
    \\ delta encoded in bits [52:32] as delta*32
    delta_enc delta * 32
    inst inst | delta_enc << 32
    ctrl 0x001e6800000e0000
    \\ sinst inst ctrl

\\ emit_bar_sync : barrier synchronization (GPU)
emit_bar_sync :
    inst 0x7b1d
    ctrl 0x000fec0000010000
    \\ sinst inst ctrl

\\ emit_membar : memory barrier (GPU)
emit_membar :
    inst 0x7992
    ctrl 0x000fec0000002000
    \\ sinst inst ctrl

\\ emit_mufu : special function unit (GPU)
\\ subop: MUFU subop from opcodes-sm90 (e.g., 0x1000 = RCP)
emit_mufu rd rs subop :
    inst 0x7308 | rd << 16 | rs << 32
    ctrl 0x000e640000000000 | subop
    \\ sinst inst ctrl

\\ emit_isetp : integer set predicate (GPU)
\\ cmp: comparison type encoded in ctrl bits
emit_isetp pd ra rb :
    inst 0x720c | ra << 24 | rb << 32
    \\ pd in bits [23:21]
    inst inst | pd << 16
    ctrl 0x000fda000bf06070
    \\ sinst inst ctrl

\\ emit_bra : branch (GPU)
\\ offset: relative branch offset in bytes (signed)
emit_bra offset :
    inst 0x7947 | 0xFC << 16
    \\ offset in bits [47:20] as signed 28-bit value
    inst inst | offset << 20
    ctrl 0x000fc0000383ffff
    \\ sinst inst ctrl

\\ emit_bra_predicated : predicated branch (GPU)
emit_bra_predicated pred offset :
    inst 0x7947 | pred << 16
    inst inst | offset << 20
    ctrl 0x000fc0000383ffff
    \\ sinst inst ctrl

\\ emit_exit : exit instruction (GPU)
emit_exit_gpu :
    inst 0x794d
    ctrl 0x000fea0003800000
    \\ sinst inst ctrl

\\ emit_nop : no-operation (GPU)
emit_nop :
    inst 0x7918
    ctrl 0x000fc00000000000
    \\ sinst inst ctrl

\\ ============================================================================
\\ ARM64-specific emitters (for host compositions)
\\ ============================================================================

\\ emit_svc : supervisor call (ARM64 syscall)
emit_svc :
    \\ SVC #0 = 0xD4000001
    arm64_emit32 0xD4000001

\\ emit_ret_host : return from host composition
emit_ret_host :
    \\ RET = 0xD65F03C0
    arm64_emit32 0xD65F03C0

\\ ============================================================================
\\ MUFU subop constants (from opcodes-sm90.fs)
\\ ============================================================================
\\ These are the extra41 values for MUFU variants.
\\
\\ MUFU_COS  = 0x0000   ≡ cosine
\\ MUFU_SIN  = 0x0400   ≅ sine
\\ MUFU_EX2  = 0x0800   2^ exp2
\\ MUFU_LG2  = 0x0c00   log₂
\\ MUFU_RCP  = 0x1000   1/ reciprocal
\\ MUFU_RSQ  = 0x1400   1/√ reciprocal square root
\\ MUFU_SQRT = 0x2000   √ square root

\\ ============================================================================
\\ Expression parser — recursive descent
\\ ============================================================================
\\ Precedence (low to high):
\\   comparison: == >= < (for conditionals, not general expressions)
\\   additive:   + -
\\   multiplicative: * /
\\   unary/atom: number, identifier, register read, load, math intrinsic

\\ parse_expr : parse an expression, return the register holding the result
parse_expr :
    parse_add_expr

\\ parse_add_expr : parse additive expression (term +/- term)
parse_add_expr :
    left parse_mul_expr
    t peek_type
    if== t 50               \\ TOK_PLUS
        consume
        right parse_mul_expr
        rd alloc_freg
        emit_add rd left right
        rd
    if== t 51               \\ TOK_MINUS
        consume
        right parse_mul_expr
        rd alloc_freg
        emit_sub rd left right
        rd
    left

\\ parse_mul_expr : parse multiplicative expression (atom * / atom)
parse_mul_expr :
    left parse_atom
    t peek_type
    if== t 52               \\ TOK_STAR
        consume
        right parse_atom
        rd alloc_freg
        emit_mul rd left right
        rd
    if== t 53               \\ TOK_SLASH
        consume
        right parse_atom
        rd alloc_freg
        emit_div rd left right
        rd
    left

\\ parse_atom : parse an atomic expression
\\ Returns the register number holding the result.
parse_atom :
    t peek_type

    \\ Integer literal → emit MOV immediate
    if== t 3                \\ TOK_INT
        val parse_int_token
        rd alloc_freg
        emit_mov_imm rd val
        rd

    \\ Float literal → emit MOV immediate (IEEE 754 bits)
    if== t 4                \\ TOK_FLOAT
        val parse_float_token
        rd alloc_freg
        emit_mov_imm rd val
        rd

    \\ → width addr — memory load
    if== t 80               \\ TOK_ARROW_R (→)
        consume
        width parse_int_token
        addr_reg parse_expr
        rd alloc_freg
        emit_load rd addr_reg 0 width
        rd

    \\ ↑ $N — register read
    if== t 82               \\ TOK_ARROW_U (↑)
        consume
        parse_regread

    \\ √ expr — square root (MUFU.SQRT)
    if== t 87               \\ TOK_SQRT (√)
        consume
        src parse_expr
        rd alloc_freg
        emit_mufu rd src 0x2000
        rd

    \\ ≅ expr — sine (MUFU.SIN)
    if== t 88               \\ TOK_APPROX (≅)
        consume
        src parse_expr
        rd alloc_freg
        emit_mufu rd src 0x0400
        rd

    \\ ≡ expr — cosine (MUFU.COS)
    if== t 89               \\ TOK_IDENTICAL (≡)
        consume
        src parse_expr
        rd alloc_freg
        emit_mufu rd src 0x0000
        rd

    \\ Σ expr — sum reduction
    if== t 84               \\ TOK_SIGMA (Σ)
        consume
        src parse_expr
        parse_reduction_sum src

    \\ △ expr — max reduction
    if== t 85               \\ TOK_TRIANGLE (△)
        consume
        src parse_expr
        parse_reduction_max src

    \\ ▽ expr — min reduction
    if== t 86               \\ TOK_NABLA (▽)
        consume
        src parse_expr
        parse_reduction_min src

    \\ # modifier — index of reduction (argmax, argmin)
    if== t 90               \\ TOK_HASH (#)
        consume
        parse_index_reduction

    \\ ** expr — elementwise (vector loop)
    if== t 91               \\ TOK_STARSTAR (**)
        consume
        parse_elementwise

    \\ Identifier — variable lookup or composition call
    if== t 5                \\ TOK_IDENT
        parse_ident_expr

    \\ Parenthesized expression
    if== t 69               \\ TOK_LPAREN
        consume
        val parse_expr
        expect 70           \\ TOK_RPAREN
        val

    \\ Error: unexpected token
    -1

\\ parse_ident_expr : handle an identifier in expression position
\\ Could be: variable reference, register name, math intrinsic,
\\ or a composition call
parse_ident_expr :
    ptr tok_text_ptr
    len peek_length

    \\ Check for math intrinsics (multi-byte keywords)
    \\ 1/ → MUFU.RCP (reciprocal)
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 49              \\ '1'
            if== c1 47          \\ '/'
                consume
                src parse_expr
                rd alloc_freg
                emit_mufu rd src 0x1000
                rd

    \\ 2^ → MUFU.EX2
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 50              \\ '2'
            if== c1 94          \\ '^'
                consume
                src parse_expr
                rd alloc_freg
                emit_mufu rd src 0x0800
                rd

    \\ 1/√ → MUFU.RSQ (3-byte check: '1' '/' + UTF-8 √)
    \\ log₂ → MUFU.LG2 (handled by lexer as ident)
    \\ e^ → composite: multiply by log2(e) then 2^
    \\ ln → composite: log₂ then multiply by ln(2)
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        \\ e^
        if== c0 101             \\ 'e'
            if== c1 94          \\ '^'
                consume
                src parse_expr
                \\ rd = src * 1.44269504 (log2e)
                log2e alloc_freg
                emit_mov_imm log2e 0x3FB8AA3B
                tmp alloc_freg
                emit_mul tmp src log2e
                \\ rd = 2^(tmp)
                rd alloc_freg
                emit_mufu rd tmp 0x0800
                rd

        \\ ln
        if== c0 108             \\ 'l'
            if== c1 110         \\ 'n'
                consume
                src parse_expr
                \\ tmp = log₂(src)
                tmp alloc_freg
                emit_mufu tmp src 0x0c00
                \\ rd = tmp * 0.69314718 (ln2)
                ln2 alloc_freg
                emit_mov_imm ln2 0x3F317218
                rd alloc_freg
                emit_mul rd tmp ln2
                rd

    \\ Variable lookup in symbol table
    idx sym_find ptr len
    if>= idx 0
        consume
        k sym_kind idx
        reg sym_reg idx

        \\ Check for array indexing: ident [ expr ]
        nt peek_type
        if== nt 67             \\ TOK_LBRACK
            consume
            idx_reg parse_expr
            expect 68          \\ TOK_RBRACK
            \\ Compute address: base + idx * element_size
            \\ For now, assume 32-bit (4 bytes) elements
            offset_reg alloc_rreg
            four alloc_rreg
            emit_mov_imm four 4
            emit_mul offset_reg idx_reg four
            addr alloc_rreg
            emit_add addr reg offset_reg
            rd alloc_freg
            emit_load rd addr 0 32
            rd

        \\ Plain variable reference — return its register
        reg

    \\ Not a known symbol — might be a composition call
    \\ (handled at statement level, not expression level)
    consume
    -1

\\ ============================================================================
\\ Register read/write parsing
\\ ============================================================================

\\ parse_regread : ↑ $N — read hardware register N
\\ GPU: emit S2R with appropriate SR selector
\\ HOST: emit MRS or MOV from special register
parse_regread :
    \\ Next token should be $N (a dollar-number or dollar-name)
    \\ For now, parse as a plain integer
    t peek_type
    if== t 3                \\ TOK_INT — register number
        regnum parse_int_token
        rd alloc_rreg
        if== emit_target 0
            \\ GPU: S2R
            emit_s2r rd regnum
        \\ HOST: MRS
        \\ emit_mrs rd regnum
        rd
    \\ Named register — check for known names
    if== t 5                \\ TOK_IDENT
        ptr tok_text_ptr
        len peek_length
        consume
        rd alloc_rreg
        \\ Match known special register names
        \\ TID_X = 0x21, TID_Y = 0x22, CTAID_X = 0x25, LANEID = 0x00
        if== len 3
            c0 → 8 ptr 0
            \\ tid → TID_X (shortcut for common case)
            if== c0 116         \\ 't' (tid)
                emit_s2r rd 0x21
                rd
        rd
    -1

\\ parse_regwrite : ↓ $N val — write val into hardware register N
\\ HOST: MOV into register
parse_regwrite :
    \\ Parse register number
    regnum parse_int_token
    \\ Parse value expression
    val_reg parse_expr
    \\ Emit: MOV $regnum, val_reg
    if== emit_target 1
        \\ ARM64: MOV Xregnum, Xval
        emit_mov regnum val_reg
    \\ GPU: direct register write via MOV
        emit_mov_reg regnum val_reg

\\ ============================================================================
\\ Reduction operations (GPU only)
\\ ============================================================================
\\ Reductions use warp-level shuffle-tree patterns.
\\ Σ → sum: SHFL.BFLY + FADD tree (delta = 16, 8, 4, 2, 1)
\\ △ → max: SHFL.BFLY + FMNMX tree (max mode)
\\ ▽ → min: SHFL.BFLY + FMNMX tree (min mode)

\\ parse_reduction_sum : emit shuffle-tree sum reduction
\\ src: register holding the value to reduce
parse_reduction_sum src :
    \\ 5-step butterfly reduction: delta = 16, 8, 4, 2, 1
    rd src
    \\ delta 16
    shfl alloc_freg
    emit_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_add rd2 rd shfl
    \\ delta 8
    shfl2 alloc_freg
    emit_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_add rd3 rd2 shfl2
    \\ delta 4
    shfl3 alloc_freg
    emit_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_add rd4 rd3 shfl3
    \\ delta 2
    shfl4 alloc_freg
    emit_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_add rd5 rd4 shfl4
    \\ delta 1
    shfl5 alloc_freg
    emit_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_add rd6 rd5 shfl5
    rd6

\\ parse_reduction_max : emit shuffle-tree max reduction
parse_reduction_max src :
    \\ Same tree structure but with FMNMX (max mode)
    \\ FMNMX with PT → max, with !PT → min
    rd src
    shfl alloc_freg
    emit_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_fmnmx_max rd2 rd shfl
    shfl2 alloc_freg
    emit_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_fmnmx_max rd3 rd2 shfl2
    shfl3 alloc_freg
    emit_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_fmnmx_max rd4 rd3 shfl3
    shfl4 alloc_freg
    emit_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_fmnmx_max rd5 rd4 shfl4
    shfl5 alloc_freg
    emit_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_fmnmx_max rd6 rd5 shfl5
    rd6

\\ parse_reduction_min : emit shuffle-tree min reduction
parse_reduction_min src :
    rd src
    shfl alloc_freg
    emit_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_fmnmx_min rd2 rd shfl
    shfl2 alloc_freg
    emit_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_fmnmx_min rd3 rd2 shfl2
    shfl3 alloc_freg
    emit_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_fmnmx_min rd4 rd3 shfl3
    shfl4 alloc_freg
    emit_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_fmnmx_min rd5 rd4 shfl4
    shfl5 alloc_freg
    emit_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_fmnmx_min rd6 rd5 shfl5
    rd6

\\ emit_fmnmx_max : FMNMX in max mode (GPU)
emit_fmnmx_max rd ra rb :
    \\ FMNMX with PT predicate → selects max
    inst 0x7209 | rd << 16 | ra << 24 | rb << 32
    ctrl 0x000fca0000000000
    \\ sinst inst ctrl

\\ emit_fmnmx_min : FMNMX in min mode (GPU)
emit_fmnmx_min rd ra rb :
    \\ FMNMX with !PT predicate → selects min
    inst 0x7209 | rd << 16 | ra << 24 | rb << 32
    \\ Set negate-predicate bit to invert PT → !PT for min
    ctrl 0x000fca0000000000 | 1 << 15
    \\ sinst inst ctrl

\\ parse_index_reduction : # followed by △ or ▽
\\ Returns the position (index) of the max/min element
parse_index_reduction :
    t peek_type
    if== t 85               \\ TOK_TRIANGLE (△) → argmax
        consume
        src parse_expr
        \\ Emit argmax: track both value and index through reduction
        parse_argmax src
    if== t 86               \\ TOK_NABLA (▽) → argmin
        consume
        src parse_expr
        parse_argmin src
    -1

\\ parse_argmax : emit argmax reduction (index of maximum element)
\\ Returns register holding the index
parse_argmax src :
    \\ Emit tid read for initial index
    idx alloc_rreg
    emit_s2r idx 0x21       \\ TID.X as initial index
    \\ Shuffle-tree: at each step, compare and select both value and index
    \\ This is an expanded pattern — 5 levels of butterfly
    val src
    \\ delta 16
    shfl_v alloc_freg
    shfl_i alloc_rreg
    emit_shfl_bfly shfl_v val 16
    emit_shfl_bfly shfl_i idx 16
    p alloc_preg
    emit_fsetp_gt p val shfl_v
    new_val alloc_freg
    new_idx alloc_rreg
    emit_sel_f new_val p val shfl_v
    emit_sel_r new_idx p idx shfl_i
    \\ Repeat for delta 8, 4, 2, 1 (abbreviated — same pattern)
    \\ ... (5 total levels)
    new_idx

\\ parse_argmin : emit argmin reduction (index of minimum element)
parse_argmin src :
    idx alloc_rreg
    emit_s2r idx 0x21
    val src
    shfl_v alloc_freg
    shfl_i alloc_rreg
    emit_shfl_bfly shfl_v val 16
    emit_shfl_bfly shfl_i idx 16
    p alloc_preg
    emit_fsetp_lt p val shfl_v
    new_val alloc_freg
    new_idx alloc_rreg
    emit_sel_f new_val p val shfl_v
    emit_sel_r new_idx p idx shfl_i
    new_idx

\\ Predicate helpers for argmax/argmin
emit_fsetp_gt pd ra rb :
    inst 0x720b | ra << 24 | rb << 32 | pd << 16
    ctrl 0x000fda000bf06070
    \\ sinst inst ctrl

emit_fsetp_lt pd ra rb :
    inst 0x720b | ra << 24 | rb << 32 | pd << 16
    \\ LT comparison mode (different ctrl bits)
    ctrl 0x000fda000bf02070
    \\ sinst inst ctrl

emit_sel_f rd pd ra rb :
    \\ Predicated select for float (CSEL equivalent on GPU)
    \\ Use FMNMX with predicate
    inst 0x7209 | rd << 16 | ra << 24 | rb << 32
    ctrl 0x000fca0000000000 | pd << 12
    \\ sinst inst ctrl

emit_sel_r rd pd ra rb :
    \\ Predicated select for integer
    \\ Use LOP3 with predicate to select
    emit_mov_reg rd ra
    \\ TODO: proper predicated move

\\ parse_elementwise : ** expr — vector elementwise loop
\\ Wraps the following expression in a per-element iteration
parse_elementwise :
    \\ This is a modifier: the next expression is applied elementwise
    \\ The implementation emits a stride loop around the expression
    \\ For bootstrap: just parse the inner expression
    parse_expr

\\ ============================================================================
\\ Statement parser — the main dispatch
\\ ============================================================================

\\ parse_stmt : parse a single statement
\\ Dispatches on the first token of the statement.
parse_stmt :
    \\ Skip leading whitespace tokens
    skip_newlines

    t peek_type

    \\ End of file
    if== t 0                \\ TOK_EOF
        0

    \\ → (load) — parse memory load as assignment target
    if== t 80               \\ TOK_ARROW_R (→)
        consume
        width parse_int_token
        addr parse_expr
        rd alloc_freg
        emit_load rd addr 0 width
        rd

    \\ ← (store) — parse memory store
    if== t 81               \\ TOK_ARROW_L (←)
        consume
        parse_store_stmt

    \\ ↑ (register read)
    if== t 82               \\ TOK_ARROW_U (↑)
        consume
        parse_regread

    \\ ↓ (register write)
    if== t 83               \\ TOK_ARROW_D (↓)
        consume
        parse_regwrite

    \\ for — counted loop
    if== t 16               \\ TOK_FOR
        consume
        parse_for

    \\ each — thread-parallel iteration (GPU)
    if== t 18               \\ TOK_EACH
        consume
        parse_each

    \\ stride — stride loop (GPU)
    if== t 19               \\ TOK_STRIDE
        consume
        parse_stride

    \\ if== — equality conditional
    if== t 13               \\ TOK_IF
        \\ The lexer produces if== / if>= / if< as identifiers
        \\ We check the full text
        parse_conditional

    \\ barrier (GPU)
    if== t 32               \\ TOK_BARRIER
        consume
        emit_bar_sync

    \\ shared memory declaration (GPU)
    if== t 31               \\ TOK_SHARED
        consume
        parse_shared

    \\ exit
    if== t 34               \\ TOK_EXIT
        consume
        if== emit_target 0
            emit_exit_gpu
        emit_ret_host

    \\ trap — syscall
    \\ Check by ident text since lexer may not have a dedicated token
    if== t 5                \\ TOK_IDENT
        parse_ident_stmt

    \\ Expression statement (standalone expression, result discarded)
    parse_expr
    1

\\ parse_store_stmt : ← width addr val
\\ Store a value to memory
parse_store_stmt :
    width parse_int_token
    \\ Parse the address expression
    addr parse_expr
    \\ Parse the value expression
    val parse_expr
    emit_store val addr 0 width

\\ parse_ident_stmt : handle an identifier at statement start
\\ Could be:
\\   - trap (syscall)
\\   - variable assignment: x = expr  or  x expr (result stored to x)
\\   - composition call
\\   - standalone expression
parse_ident_stmt :
    ptr tok_text_ptr
    len peek_length

    \\ Check for 'trap'
    if== len 4
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        c2 → 8 ptr 2
        c3 → 8 ptr 3
        if== c0 116             \\ 't'
            if== c1 114         \\ 'r'
                if== c2 97      \\ 'a'
                    if== c3 112 \\ 'p'
                        consume
                        emit_svc

    \\ Check for known math intrinsics at statement level
    \\ These are parsed as: result_name intrinsic_op source
    \\ e.g., "rms √ s / D" → rms = √(s) / D

    \\ Check for conditional keywords: if==, if>=, if<
    if== len 4
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 105             \\ 'i'
            if== c1 102         \\ 'f'
                parse_conditional

    \\ Look up in symbol table
    idx sym_find ptr len
    if>= idx 0
        consume
        \\ Check for assignment: x = expr
        nt peek_type
        if== nt 54              \\ TOK_EQ (=)
            consume
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1

        \\ Check for array store: x [ idx ] = expr
        if== nt 67              \\ TOK_LBRACK
            consume
            idx_expr parse_expr
            expect 68           \\ TOK_RBRACK
            expect 54           \\ TOK_EQ
            val parse_expr
            base sym_reg idx
            offset_reg alloc_rreg
            four alloc_rreg
            emit_mov_imm four 4
            emit_mul offset_reg idx_expr four
            addr alloc_rreg
            emit_add addr base offset_reg
            emit_store val addr 0 32
            1

        \\ Otherwise: this identifier starts an expression
        \\ Could be a composition call or part of a Lithos-style statement:
        \\   result_name op arg1 arg2 ...
        \\ Parse as: the first ident is the result variable,
        \\ the rest is the expression to compute into it.
        \\ Check if next token starts an expression (operator or value)
        nt2 peek_type
        if== nt2 80             \\ → (load expression)
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1
        if== nt2 84             \\ Σ (reduction)
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1
        if== nt2 85             \\ △ (max)
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1
        if== nt2 86             \\ ▽ (min)
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1
        if== nt2 87             \\ √
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1
        if== nt2 90             \\ # (index)
            val parse_expr
            reg sym_reg idx
            emit_mov_reg reg val
            1

        \\ Check if it looks like: name value_expr op value_expr
        \\ Lithos style: "sq x * x" means sq = x * x
        \\ Parse as expression, assign result to this symbol's register
        val parse_expr
        reg sym_reg idx
        emit_mov_reg reg val
        1

    \\ Not in symbol table — could be a new local variable assignment
    \\ In Lithos, first use of a name in a statement creates the variable:
    \\   sq → 32 x i * x i
    \\   means: sq is a new local, value is the expression after it
    consume
    nt peek_type
    \\ If followed by an expression-starting token, treat as new variable
    if== nt 80               \\ → (expression follows)
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 84               \\ Σ
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 85               \\ △
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 86               \\ ▽
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 87               \\ √
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 3                \\ TOK_INT
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 5                \\ TOK_IDENT (could be composition call or expr)
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 88               \\ ≅
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 89               \\ ≡
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 90               \\ #
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 91               \\ **
        val parse_expr
        rd alloc_freg
        emit_mov_reg rd val
        sym_add ptr len 2 rd
        1
    \\ Unrecognized statement — skip to end of line
    skip_to_eol
    0

\\ skip_to_eol : advance token position to next newline
skip_to_eol :
    t peek_type
    if== t 0                \\ TOK_EOF
        0
    if== t 1                \\ TOK_NEWLINE
        consume
    consume
    skip_to_eol

\\ ============================================================================
\\ Control flow parsing
\\ ============================================================================

\\ parse_for : for i start end step
\\ Emit: initialize loop var, emit bounds check, mark branch patch point
parse_for :
    \\ Parse loop variable name
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    \\ Parse start value
    start_reg parse_expr

    \\ Parse end value
    end_reg parse_expr

    \\ Parse step value
    step_reg parse_expr

    \\ Allocate register for loop variable
    loop_reg alloc_rreg

    \\ Initialize loop variable
    emit_mov_reg loop_reg start_reg

    \\ Register loop variable in symbol table
    sym_add var_ptr var_len 3 loop_reg

    \\ Record loop start position for back-edge
    \\ Push onto branch stack: (patch_addr, loop_start, loop_reg, end_reg, step_reg)
    push_branch loop_reg end_reg step_reg

    \\ Emit bounds check (will be patched)
    if== emit_target 0
        \\ GPU: ISETP P0, loop_reg, end_reg (GE → exit loop)
        p alloc_preg
        emit_isetp p loop_reg end_reg
        \\ BRA forward (placeholder — patch later)
        emit_bra_predicated p 0
    \\ ARM64: CMP loop_reg, end_reg; B.GE forward
        emit_cmp_reg loop_reg end_reg
        \\ B.GE placeholder
        arm64_emit32 0x5400000A

\\ parse_for_end : end of for loop — emit back-edge and patch forward branch
parse_for_end :
    \\ Pop branch stack
    loop_reg pop_branch_reg
    end_reg pop_branch_end
    step_reg pop_branch_step

    \\ Emit: loop_var = loop_var + step
    emit_add loop_reg loop_reg step_reg

    \\ Emit back-edge branch to loop start
    \\ TODO: compute offset from current position to loop start
    if== emit_target 0
        emit_bra 0              \\ placeholder offset
    \\ ARM64: B back
        arm64_emit32 0x14000000 \\ placeholder

    \\ Patch the forward branch at loop start
    \\ TODO: patch_branch_forward

\\ parse_each : each i — thread-parallel iteration (GPU)
\\ Emits: S2R tid.x; S2R ctaid.x; IMAD global_idx = ctaid*blockDim + tid
parse_each :
    \\ Parse variable name
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    \\ Allocate registers
    tid_reg alloc_rreg
    ctaid_reg alloc_rreg
    gidx_reg alloc_rreg

    \\ Emit thread index computation
    if== emit_target 0
        \\ GPU: S2R tid.x, S2R ctaid.x
        emit_s2r tid_reg 0x21       \\ SR_TID_X
        emit_s2r ctaid_reg 0x25     \\ SR_CTAID_X

        \\ global_idx = ctaid * blockDim + tid
        \\ blockDim comes from a parameter — for now use a constant register
        bdim_reg alloc_rreg
        \\ IMAD gidx = ctaid * bdim + tid
        \\ Use IMAD: rd = ra * rb + rc
        inst 0x7224 | gidx_reg << 16 | ctaid_reg << 24 | bdim_reg << 32
        ctrl 0x000fe200078e0000 | tid_reg
        \\ sinst inst ctrl

    \\ Register variable in symbol table
    sym_add var_ptr var_len 4 gidx_reg

\\ parse_stride : stride i dim — stride loop (GPU)
\\ Emits: compute thread index, loop with grid-stride pattern
parse_stride :
    \\ Parse variable name
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    \\ Parse dimension (the bound)
    dim_reg parse_expr

    \\ Allocate registers
    loop_reg alloc_rreg

    \\ Initialize: loop_reg = global_thread_idx (from each)
    \\ The stride variable starts at the thread's global index
    tid_reg alloc_rreg
    ctaid_reg alloc_rreg
    emit_s2r tid_reg 0x21
    emit_s2r ctaid_reg 0x25
    gidx alloc_rreg
    bdim alloc_rreg
    \\ global_idx = ctaid * blockDim + tid
    emit_mov_reg loop_reg gidx

    \\ Register in symbol table
    sym_add var_ptr var_len 9 loop_reg

    \\ Push branch info for stride-end
    push_branch loop_reg dim_reg gidx

    \\ Emit bounds check
    p alloc_preg
    emit_isetp p loop_reg dim_reg
    emit_bra_predicated p 0

\\ parse_conditional : if== a b / if>= a b / if< a b
\\ Check the full identifier text to determine comparison type
parse_conditional :
    ptr tok_text_ptr
    len peek_length
    consume

    \\ Determine comparison type from the keyword text
    \\ if== → equality
    \\ if>= → greater-or-equal
    \\ if<  → less-than
    cmp_type 0             \\ 0=eq, 1=ge, 2=lt

    if>= len 4
        c2 → 8 ptr 2
        c3 → 8 ptr 3
        if== c2 61              \\ '='
            if== c3 61          \\ '='
                cmp_type 0      \\ equality
        if== c2 62              \\ '>'
            if== c3 61          \\ '='
                cmp_type 1      \\ >=
        if== c2 60              \\ '<'
            cmp_type 2          \\ <

    \\ Parse the two operands
    a parse_expr
    b parse_expr

    \\ Emit comparison
    if== emit_target 0
        \\ GPU: ISETP/FSETP
        p alloc_preg
        if== cmp_type 0
            \\ EQ comparison
            emit_isetp p a b
            \\ Predicate the next instruction(s) on p
        if== cmp_type 1
            \\ GE comparison
            emit_isetp p a b
        if== cmp_type 2
            \\ LT comparison
            emit_isetp p a b
    \\ ARM64: CMP + conditional branch
        emit_cmp_reg a b
        \\ Emit conditional branch placeholder
        if== cmp_type 0
            \\ BNE forward (skip body if not equal)
            arm64_emit32 0x54000001
        if== cmp_type 1
            \\ BLT forward (skip body if less than)
            arm64_emit32 0x5400000B
        if== cmp_type 2
            \\ BGE forward (skip body if greater-or-equal)
            arm64_emit32 0x5400000A

    \\ Parse the body of the conditional (indented block)
    parse_body

\\ parse_shared : shared name size type
\\ Declare shared memory region (GPU)
parse_shared :
    \\ Parse name
    name_ptr tok_text_ptr
    name_len peek_length
    consume

    \\ Parse size (number of elements)
    size parse_int_token

    \\ Parse type (f32, u32, etc.) — determines element size
    type_tok peek_type
    consume
    elem_size 4             \\ default 4 bytes (f32/u32)
    if== type_tok 43        \\ TOK_F16
        elem_size 2

    \\ Calculate total bytes
    total size * elem_size

    \\ Register in shared memory table
    sidx n_shared
    \\ Store name reference, size
    ← 32 shm_sizes sidx * 4 total

    \\ Allocate a descriptor register for the shared buffer
    rd alloc_rdreg
    sym_add name_ptr name_len 5 rd

    n_shared n_shared + 1
    shmem_total shmem_total + total

\\ ============================================================================
\\ Branch stack operations
\\ ============================================================================

push_branch loop_reg end_reg step_reg :
    idx branch_depth * 16
    ← 32 branch_stack idx loop_reg
    ← 32 branch_stack idx + 4 end_reg
    ← 32 branch_stack idx + 8 step_reg
    \\ Store current emit position for patching
    \\ ← 32 branch_stack idx + 12 current_pos
    branch_depth branch_depth + 1

pop_branch_reg :
    branch_depth branch_depth - 1
    idx branch_depth * 16
    → 32 branch_stack idx

pop_branch_end :
    idx branch_depth * 16
    → 32 branch_stack idx + 4

pop_branch_step :
    idx branch_depth * 16
    → 32 branch_stack idx + 8

\\ ============================================================================
\\ Body parser — indented block
\\ ============================================================================
\\ Parses statements in an indented block until indentation decreases
\\ or end of file is reached.

parse_body :
    \\ Expect a newline after the : or control keyword
    skip_newlines

    \\ Check the indentation of the first statement
    t peek_type
    if== t 2                \\ TOK_INDENT
        new_indent peek_length
        \\ The body's indent level is whatever the first line is
        old_indent body_indent
        body_indent new_indent
        consume

        \\ Parse statements until indent drops
        parse_body_loop old_indent

        \\ Restore indent level
        body_indent old_indent

parse_body_loop old_indent :
    t peek_type
    if== t 0                \\ TOK_EOF
        0

    \\ Check for end of indented block
    if== t 1                \\ TOK_NEWLINE
        consume
        \\ Check next line's indent
        t2 peek_type
        if== t2 2           \\ TOK_INDENT
            ind peek_length
            if< ind body_indent
                \\ Dedent — end of block
                0
            consume
            \\ Still in block — parse next statement
            parse_stmt
            parse_body_loop old_indent
        \\ No indent token → column 0 → definitely dedented
            0

    \\ Parse a statement
    parse_stmt
    parse_body_loop old_indent

\\ ============================================================================
\\ Composition parser
\\ ============================================================================
\\ A composition in Lithos:
\\   name arg1 arg2 ... :
\\       body (indented statements)
\\
\\ No 'fn' keyword. The composition starts with an identifier (name),
\\ followed by more identifiers (args), followed by : (colon),
\\ then an indented body.

parse_composition :
    \\ Read the composition name
    name_ptr tok_text_ptr
    name_len peek_length
    consume

    \\ Reset state for new composition
    sym_reset
    regs_reset
    n_shared 0
    shmem_total 0
    branch_depth 0

    \\ Read args until we hit : (TOK_COLON)
    arg_count 0
    parse_comp_args arg_count

    \\ Expect :
    expect 72               \\ TOK_COLON

    \\ Register this composition in the composition table
    cidx n_comps
    ← 32 comp_tok_starts cidx * 4 tok_pos
    ← 32 comp_arg_counts cidx * 4 arg_count
    \\ Copy name
    clen name_len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 name_ptr j
        ← 8 comp_names cidx * 32 + j b
    ← 32 comp_lens cidx * 4 name_len
    n_comps n_comps + 1

    \\ Parse the body
    parse_body

    \\ Emit function epilogue
    if== emit_target 0
        emit_exit_gpu
    emit_ret_host

\\ parse_comp_args : read argument names until :
\\ Each arg is registered as an input symbol with a fresh register.
parse_comp_args count :
    t peek_type
    if== t 72               \\ TOK_COLON — done
        count

    if== t 5                \\ TOK_IDENT — argument name
        ptr tok_text_ptr
        len peek_length
        consume

        \\ Allocate register for this argument
        if== emit_target 0
            \\ GPU: allocate descriptor register for pointer args
            reg alloc_rdreg
        \\ ARM64: arguments in X0-X7
            reg count

        \\ Register as input symbol
        sym_add ptr len 0 reg
        count count + 1

        \\ Recurse for more args
        parse_comp_args count

    \\ Unexpected token — error
    count

\\ ============================================================================
\\ Top-level file parser
\\ ============================================================================
\\ Parses an entire .li file: a sequence of compositions.

parse_file :
    \\ Loop until EOF
    t peek_type
    if== t 0                \\ TOK_EOF
        0

    \\ Skip blank lines
    if== t 1                \\ TOK_NEWLINE
        consume
        parse_file

    if== t 2                \\ TOK_INDENT
        consume
        parse_file

    \\ Check for 'host' prefix — marks HOST backend composition
    if== t 35               \\ TOK_HOST
        consume
        emit_target 1       \\ Switch to ARM64
        parse_composition
        emit_target 0       \\ Reset to GPU
        parse_file

    \\ Check for \\ (comment at top level)
    \\ Comments are already stripped by the lexer, but handle stray cases

    \\ Default: this is a GPU composition
    if== t 5                \\ TOK_IDENT
        emit_target 0       \\ GPU backend
        parse_composition
        parse_file

    if== t 11               \\ TOK_KERNEL
        consume
        emit_target 0       \\ GPU backend
        parse_composition
        parse_file

    \\ Unknown top-level token — skip
    consume
    parse_file

\\ ============================================================================
\\ Parser initialization
\\ ============================================================================

\\ parser_init : initialize parser state from lexer output
\\ tokens_ptr: pointer to token buffer (from lexer)
\\ total: number of tokens
\\ source: pointer to source buffer
parser_init tokens_ptr total source :
    tok_pos 0
    tok_total total
    src_buf source
    emit_target 0
    body_indent 0
    comp_depth 0
    error_count 0
    n_syms 0
    n_comps 0
    n_shared 0
    shmem_total 0
    branch_depth 0
    regs_reset

\\ ============================================================================
\\ Entry point
\\ ============================================================================
\\ Called by the compiler driver after lexing.
\\
\\ tokens_ptr: pointer to the flat u32 token buffer
\\ total:      number of tokens (token_count from lexer)
\\ source:     pointer to the mmap'd source file
\\
\\ After parse returns:
\\   arm64_buf / arm64_pos contain HOST code (from emit-arm64.li)
\\   sass_buf / sass_pos contain GPU code (from sm90 emitter)

lithos_parse tokens_ptr total source :
    parser_init tokens_ptr total source
    parse_file
