\\ compiler.ls — Lithos self-hosting compiler, single compilation unit
\\
\\ Compiles .ls source files to:
\\   - ARM64 ELF executables (host code)
\\   - sm90 GPU cubins (kernel code)
\\
\\ Architecture: single-pass recursive descent with direct emission.
\\ No AST. No IR. Compositions are parsed and emitted immediately.
\\
\\ Pipeline:
\\   1. mmap source file
\\   2. Lexer → flat token buffer
\\   3. Parser → direct machine code emission (ARM64 or sm90)
\\   4. ELF writer → wrap output in ELF binary
\\   5. Write output file
\\
\\ Six modules combined in order:
\\   Section 0: Constants, buffers, and globals
\\   Section 1: ARM64 backend (emit-arm64)
\\   Section 2: GPU backend (emit-gpu / emit-sm90)
\\   Section 3: Lexer (lithos-lexer)
\\   Section 4: Parser (lithos-parser)
\\   Section 5: Safetensors reader (lithos-safetensors)
\\   Section 6: ELF writer (lithos-elf)
\\   Section 7: Main entry point

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 0 — CONSTANTS, BUFFERS, AND GLOBALS                          ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================================
\\ Token type constants
\\ ============================================================================
\\
\\ TOK_EOF        = 0    end of input
\\ TOK_NEWLINE    = 1    newline character (LF)
\\ TOK_INDENT     = 2    leading whitespace at start of line (length = space count)
\\
\\ TOK_INT        = 3    decimal or hex integer literal
\\ TOK_FLOAT      = 4    floating-point literal (has decimal point)
\\ TOK_IDENT      = 5    identifier (not a keyword)
\\
\\ TOK_KERNEL     = 11
\\ TOK_PARAM      = 12
\\ TOK_IF         = 13
\\ TOK_ELSE       = 14
\\ TOK_ELIF       = 15
\\ TOK_FOR        = 16
\\ TOK_ENDFOR     = 17
\\ TOK_EACH       = 18
\\ TOK_STRIDE     = 19
\\ TOK_WHILE      = 20
\\ TOK_RETURN     = 21
\\ TOK_CONST      = 22
\\ TOK_VAR        = 23
\\ TOK_BUF        = 24
\\ TOK_WEIGHT     = 25
\\ TOK_LAYER      = 26
\\ TOK_BIND       = 27
\\ TOK_RUNTIME    = 28
\\ TOK_TEMPLATE   = 29
\\ TOK_PROJECT    = 30
\\ TOK_SHARED     = 31
\\ TOK_BARRIER    = 32
\\ TOK_LABEL      = 33
\\ TOK_EXIT       = 34
\\ TOK_HOST       = 35   host (marks a host composition vs gpu composition)
\\ TOK_LOAD       = 36   →   (memory load: → width addr)
\\ TOK_STORE      = 37   ←   (memory store: ← width addr val)
\\ TOK_REG_READ   = 38   ↑   (register read: ↑ $N)
\\ TOK_REG_WRITE  = 39   ↓   (register write: ↓ $N val)
\\
\\ TOK_F32        = 40
\\ TOK_U32        = 41
\\ TOK_S32        = 42
\\ TOK_F16        = 43
\\ TOK_PTR        = 44
\\ TOK_VOID       = 45
\\
\\ TOK_PLUS       = 50   +
\\ TOK_MINUS      = 51   -
\\ TOK_STAR       = 52   *
\\ TOK_SLASH      = 53   /
\\ TOK_EQ         = 54   =
\\ TOK_EQEQ       = 55   ==
\\ TOK_NEQ        = 56   !=
\\ TOK_LT         = 57   <
\\ TOK_GT         = 58   >
\\ TOK_LTE        = 59   <=
\\ TOK_GTE        = 60   >=
\\ TOK_AMP        = 61   &
\\ TOK_PIPE       = 62   |
\\ TOK_CARET      = 63   ^
\\ TOK_SHL        = 64   <<
\\ TOK_SHR        = 65   >>
\\
\\ TOK_LBRACK     = 67   [
\\ TOK_RBRACK     = 68   ]
\\ TOK_LPAREN     = 69   (
\\ TOK_RPAREN     = 70   )
\\ TOK_COMMA      = 71   ,
\\ TOK_COLON      = 72   :
\\ TOK_DOT        = 73   .
\\ TOK_AT         = 74   @
\\
\\ TOK_SUM        = 75   Σ   (sum reduction)
\\ TOK_MAX        = 76   △   (max reduction)
\\ TOK_MIN        = 77   ▽   (min reduction)
\\ TOK_INDEX      = 78   #   (index operator: # △ x = argmax)
\\ TOK_SQRT       = 79   √   (square root)
\\ TOK_SIN        = 80   ≅   (MUFU.SIN)
\\ TOK_COS        = 81   ≡   (MUFU.COS)
\\
\\ Parser Unicode token types (extended enum):
\\ TOK_ARROW_R    = 80   → (load)
\\ TOK_ARROW_L    = 81   ← (store)
\\ TOK_ARROW_U    = 82   ↑ (register read)
\\ TOK_ARROW_D    = 83   ↓ (register write)
\\ TOK_SIGMA      = 84   Σ (sum reduction)
\\ TOK_TRIANGLE   = 85   △ (max reduction)
\\ TOK_NABLA      = 86   ▽ (min reduction)
\\ TOK_P_SQRT     = 87   √ (square root)
\\ TOK_APPROX     = 88   ≅ (sine)
\\ TOK_IDENTICAL  = 89   ≡ (cosine)
\\ TOK_HASH       = 90   # (index modifier)
\\ TOK_STARSTAR   = 91   ** (elementwise)
\\ TOK_STARSTARSTAR = 92 *** (matrix)
\\
\\ TOK_GOTO       = 93   goto (unconditional branch to label)
\\ TOK_TRAP    = 94   trap (ARM64 SVC #0 with register setup)
\\ TOK_CONTINUE   = 95   continue (loop continuation)
\\ TOK_CONSTANT   = 96   constant (Forth-style: VALUE constant NAME)
\\ TOK_DOLLAR     = 97   $ (register sigil: $0, $8, $TID_X)

\\ ============================================================================
\\ Lexer token output buffer (shared by lexer and parser)
\\ ============================================================================
\\ Each token is 3 u32 values: type, offset, length.
\\ 262144 u32 slots = room for 87381 tokens.

\\ ============================================================================
\\ ENTRY POINT — must be FIRST composition (ELF entry = start of code buffer)
\\ ============================================================================
\\ Linux ARM64: argc at [SP], argv at SP+8
main :
    argc → 64 $31 0
    argv $31 + 8
    lithos_main argc argv
    ↓ $8 93
    ↓ $0 0
    trap

\\ ============================================================================
\\ Lexer token buffer + syscall constants
\\ ============================================================================

buf tokens 262144
var token_count 0

const SYS_READ      63
const SYS_WRITE     64
const SYS_OPENAT    56
const SYS_CLOSE     57
const SYS_LSEEK     62
const SYS_MMAP      222
const SYS_MUNMAP    215
const SYS_MPROTECT  226
const SYS_EXIT      93
const SYS_BRK       214

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 1 — ARM64 BACKEND (emit-arm64)                               ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================
\\ CODE BUFFER
\\ ============================================================
\\ 1MB buffer for ARM64 machine code.

buf arm64_buf 1048576
var arm64_pos 0

arm64_emit32 val :
    \\ Write a 32-bit little-endian word at current position and advance by 4.
    ← 32 arm64_buf + arm64_pos val
    arm64_pos arm64_pos + 4

arm64_reset :
    arm64_pos 0

\\ ============================================================
\\ REGISTER CONSTANTS
\\ ============================================================

const X0   0
const X1   1
const X2   2
const X3   3
const X4   4
const X5   5
const X6   6
const X7   7
const X8   8
const X9   9
const X10  10
const X11  11
const X12  12
const X13  13
const X14  14
const X15  15
const X16  16
const X17  17
const X18  18
const X19  19
const X20  20
const X21  21
const X22  22
const X23  23
const X24  24
const X25  25
const X26  26
const X27  27
const X28  28
const X29  29
const X30  30
const XZR  31
const SP   31

\\ Aliases for calling convention
const FP   29     \\ frame pointer = X29
const LR   30     \\ link register = X30

\\ Condition codes (4-bit encoding for B.cond, CSEL, CSET)
const COND_EQ  0
const COND_NE  1
const COND_CS  2
const COND_CC  3
const COND_MI  4
const COND_PL  5
const COND_VS  6
const COND_VC  7
const COND_HI  8
const COND_LS  9
const COND_GE  10
const COND_LT  11
const COND_GT  12
const COND_LE  13
const COND_AL  14

\\ ============================================================
\\ DATA PROCESSING — REGISTER (Arithmetic)
\\ ============================================================

\\ ADD Xd, Xn, Xm — 64-bit register add
emit_add_reg rd rn rm :
    val 0x8B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ADD Xd, Xn, Xm, LSL #amount — shifted register add
emit_add_reg_lsl rd rn rm amount :
    val 0x8B000000 | (rm << 16) | (amount << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Xd, Xn, Xm — 64-bit register subtract
emit_sub_reg rd rn rm :
    val 0xCB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ADDS Xd, Xn, Xm — add and set flags
emit_adds_reg rd rn rm :
    val 0xAB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ SUBS Xd, Xn, Xm — subtract and set flags
emit_subs_reg rd rn rm :
    val 0xEB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ CMP Xn, Xm — alias for SUBS XZR, Xn, Xm
emit_cmp_reg rn rm :
    emit_subs_reg XZR rn rm

\\ NEG Xd, Xm — alias for SUB Xd, XZR, Xm
emit_neg rd rm :
    emit_sub_reg rd XZR rm

\\ MUL Xd, Xn, Xm — 64-bit multiply
emit_mul_reg rd rn rm :
    val 0x9B007C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ MADD Xd, Xn, Xm, Xa — multiply-add: Xd = Xa + Xn*Xm
emit_madd rd rn rm ra :
    val 0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ MSUB Xd, Xn, Xm, Xa — multiply-subtract: Xd = Xa - Xn*Xm
emit_msub rd rn rm ra :
    val 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SDIV Xd, Xn, Xm — signed divide
emit_sdiv rd rn rm :
    val 0x9AC00C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ UDIV Xd, Xn, Xm — unsigned divide
emit_udiv rd rn rm :
    val 0x9AC00800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — IMMEDIATE (Arithmetic)
\\ ============================================================

\\ ADD Xd, Xn, #imm12
emit_add_imm rd rn imm :
    val 0x91000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ADD Xd, Xn, #imm12, LSL #12
emit_add_imm_lsl12 rd rn imm :
    val 0x91400000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Xd, Xn, #imm12
emit_sub_imm rd rn imm :
    val 0xD1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ADDS Xd, Xn, #imm12
emit_adds_imm rd rn imm :
    val 0xB1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUBS Xd, Xn, #imm12
emit_subs_imm rd rn imm :
    val 0xF1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ CMP Xn, #imm12
emit_cmp_imm rn imm :
    emit_subs_imm XZR rn imm

\\ CMN Xn, #imm12
emit_cmn_imm rn imm :
    emit_adds_imm XZR rn imm

\\ MOV Xd, Xn — register move
emit_mov rd rn :
    val 0xAA0003E0 | (rn << 16) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — LOGICAL (Register)
\\ ============================================================

\\ AND Xd, Xn, Xm
emit_and_reg rd rn rm :
    val 0x8A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ORR Xd, Xn, Xm
emit_orr_reg rd rn rm :
    val 0xAA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ EOR Xd, Xn, Xm
emit_eor_reg rd rn rm :
    val 0xCA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ANDS Xd, Xn, Xm
emit_ands_reg rd rn rm :
    val 0xEA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ TST Xn, Xm
emit_tst_reg rn rm :
    emit_ands_reg XZR rn rm

\\ MVN Xd, Xm
emit_mvn rd rm :
    val 0xAA2003E0 | (rm << 16) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — LOGICAL (Immediate)
\\ ============================================================

\\ AND Xd, Xn, #bitmask_imm — raw encoding
emit_and_imm rd rn n immr imms :
    val 0x92000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ORR Xd, Xn, #bitmask_imm — raw encoding
emit_orr_imm rd rn n immr imms :
    val 0xB2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ EOR Xd, Xn, #bitmask_imm — raw encoding
emit_eor_imm rd rn n immr imms :
    val 0xD2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ SHIFT / BITFIELD OPERATIONS
\\ ============================================================

\\ LSL Xd, Xn, Xm — variable
emit_lsl_reg rd rn rm :
    val 0x9AC02000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ LSR Xd, Xn, Xm — variable
emit_lsr_reg rd rn rm :
    val 0x9AC02400 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ASR Xd, Xn, Xm — variable
emit_asr_reg rd rn rm :
    val 0x9AC02800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ LSL Xd, Xn, #shift — immediate
emit_lsl_imm rd rn shift :
    immr (64 - shift) & 63
    imms 63 - shift
    val 0xD3400000 | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LSR Xd, Xn, #shift — immediate
emit_lsr_imm rd rn shift :
    val 0xD340FC00 | (shift << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ASR Xd, Xn, #shift — immediate
emit_asr_imm rd rn shift :
    val 0x9340FC00 | (shift << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ CONDITIONAL SELECT
\\ ============================================================

\\ CSEL Xd, Xn, Xm, cond
emit_csel rd rn rm cond :
    val 0x9A800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CSINC Xd, Xn, Xm, cond
emit_csinc rd rn rm cond :
    val 0x9A800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CSET Xd, cond
emit_cset rd cond :
    inv cond ^ 1
    emit_csinc rd XZR XZR inv

\\ CSNEG Xd, Xn, Xm, cond
emit_csneg rd rn rm cond :
    val 0xDA800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CNEG Xd, Xn, cond
emit_cneg rd rn cond :
    inv cond ^ 1
    emit_csneg rd rn rn inv

\\ ============================================================
\\ MOVE WIDE IMMEDIATE
\\ ============================================================

\\ MOVZ Xd, #imm16, LSL #(hw*16)
emit_movz rd imm16 hw :
    val 0xD2800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ MOVK Xd, #imm16, LSL #(hw*16)
emit_movk rd imm16 hw :
    val 0xF2800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ MOVN Xd, #imm16, LSL #(hw*16)
emit_movn rd imm16 hw :
    val 0x92800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ Helper: load a full 64-bit immediate into Xd
emit_mov64 rd imm :
    chunk0 imm & 0xFFFF
    chunk1 (imm >> 16) & 0xFFFF
    chunk2 (imm >> 32) & 0xFFFF
    chunk3 (imm >> 48) & 0xFFFF
    emit_movz rd chunk0 0
    if chunk1 > 0
        emit_movk rd chunk1 1
    if chunk2 > 0
        emit_movk rd chunk2 2
    if chunk3 > 0
        emit_movk rd chunk3 3

\\ ============================================================
\\ LOADS AND STORES
\\ ============================================================

\\ LDR Xt, [Xn, #imm] — 64-bit unsigned offset (scaled by 8)
emit_ldr rd rn imm :
    val 0xF9400000 | ((imm / 8) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Xt, [Xn, #imm] — 64-bit unsigned offset (scaled by 8)
emit_str rt rn imm :
    val 0xF9000000 | ((imm / 8) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDR Xt, [Xn, Xm] — 64-bit register offset
emit_ldr_reg rd rn rm :
    val 0xF8606800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Xt, [Xn, Xm] — 64-bit register offset
emit_str_reg rt rn rm :
    val 0xF8206800 | (rm << 16) | (rn << 5) | rt
    arm64_emit32 val

\\ LDR Wt, [Xn, #imm] — 32-bit unsigned offset (scaled by 4)
emit_ldr_w rd rn imm :
    val 0xB9400000 | ((imm / 4) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Wt, [Xn, #imm] — 32-bit unsigned offset (scaled by 4)
emit_str_w rt rn imm :
    val 0xB9000000 | ((imm / 4) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDRB Wt, [Xn, #imm] — byte load
emit_ldrb rd rn imm :
    val 0x39400000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STRB Wt, [Xn, #imm] — byte store
emit_strb rt rn imm :
    val 0x39000000 | (imm << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDRH Wt, [Xn, #imm] — halfword load (scaled by 2)
emit_ldrh rd rn imm :
    val 0x79400000 | ((imm / 2) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STRH Wt, [Xn, #imm] — halfword store (scaled by 2)
emit_strh rt rn imm :
    val 0x79000000 | ((imm / 2) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDRSB Xt, [Xn, #imm] — signed byte
emit_ldrsb rd rn imm :
    val 0x39800000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LDRSH Xt, [Xn, #imm] — signed halfword
emit_ldrsh rd rn imm :
    val 0x79800000 | ((imm / 2) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LDRSW Xt, [Xn, #imm] — signed word
emit_ldrsw rd rn imm :
    val 0xB9800000 | ((imm / 4) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LDP Xt1, Xt2, [Xn, #imm] — load pair
emit_ldp rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA9400000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP Xt1, Xt2, [Xn, #imm] — store pair
emit_stp rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA9000000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ LDP pre-indexed
emit_ldp_pre rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA9C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP pre-indexed
emit_stp_pre rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ LDP post-indexed
emit_ldp_post rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP post-indexed
emit_stp_post rt1 rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA8800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ LDR Xt, [Xn, #simm9]! — pre-indexed load
emit_ldr_pre rt rn simm :
    val 0xF8400C00 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ STR Xt, [Xn, #simm9]! — pre-indexed store
emit_str_pre rt rn simm :
    val 0xF8000C00 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ LDR Xt, [Xn], #simm9 — post-indexed load
emit_ldr_post rt rn simm :
    val 0xF8400400 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ STR Xt, [Xn], #simm9 — post-indexed store
emit_str_post rt rn simm :
    val 0xF8000400 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ ============================================================
\\ BRANCHES
\\ ============================================================

\\ B offset — unconditional branch
emit_b offset :
    val 0x14000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ BL offset — branch with link
emit_bl offset :
    val 0x94000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ B.cond offset — conditional branch
emit_bcond cond offset :
    val 0x54000000 | (((offset / 4) & 0x7FFFF) << 5) | cond
    arm64_emit32 val

\\ CBZ Xt, offset — branch if zero
emit_cbz rt offset :
    val 0xB4000000 | (((offset / 4) & 0x7FFFF) << 5) | rt
    arm64_emit32 val

\\ CBNZ Xt, offset — branch if not zero
emit_cbnz rt offset :
    val 0xB5000000 | (((offset / 4) & 0x7FFFF) << 5) | rt
    arm64_emit32 val

\\ TBZ Xt, #bit, offset — test bit and branch if zero
emit_tbz rt bit offset :
    b5 (bit >> 5) & 1
    b40 bit & 0x1F
    val 0x36000000 | (b5 << 31) | (b40 << 19) | (((offset / 4) & 0x3FFF) << 5) | rt
    arm64_emit32 val

\\ TBNZ Xt, #bit, offset — test bit and branch if not zero
emit_tbnz rt bit offset :
    b5 (bit >> 5) & 1
    b40 bit & 0x1F
    val 0x37000000 | (b5 << 31) | (b40 << 19) | (((offset / 4) & 0x3FFF) << 5) | rt
    arm64_emit32 val

\\ BR Xn — indirect jump
emit_br rn :
    val 0xD61F0000 | (rn << 5)
    arm64_emit32 val

\\ BLR Xn — indirect call
emit_blr rn :
    val 0xD63F0000 | (rn << 5)
    arm64_emit32 val

\\ RET {Xn}
emit_ret rn :
    val 0xD65F0000 | (rn << 5)
    arm64_emit32 val

\\ RET — via LR
emit_ret_lr :
    emit_ret LR

\\ ============================================================
\\ PC-RELATIVE ADDRESS GENERATION
\\ ============================================================

\\ ADR Xd, offset
emit_adr rd offset :
    immlo offset & 3
    immhi (offset >> 2) & 0x7FFFF
    val (immlo << 29) | 0x10000000 | (immhi << 5) | rd
    arm64_emit32 val

\\ ADRP Xd, offset
emit_adrp rd offset :
    immlo offset & 3
    immhi (offset >> 2) & 0x7FFFF
    val (immlo << 29) | 0x90000000 | (immhi << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ SYSTEM INSTRUCTIONS
\\ ============================================================

\\ SVC #imm16
emit_svc imm16 :
    val 0xD4000001 | (imm16 << 5)
    arm64_emit32 val

\\ SVC #0 — Linux trap
emit_trap :
    emit_svc 0

\\ NOP (ARM64)
arm64_emit_nop :
    arm64_emit32 0xD503201F

\\ DSB ISH
emit_dsb_ish :
    arm64_emit32 0xD5033B9F

\\ DSB SY
emit_dsb_sy :
    arm64_emit32 0xD5033F9F

\\ DMB ISH
emit_dmb_ish :
    arm64_emit32 0xD5033BBF

\\ ISB
emit_isb :
    arm64_emit32 0xD5033FDF

\\ BRK #imm16
emit_brk imm16 :
    val 0xD4200000 | (imm16 << 5)
    arm64_emit32 val

\\ ============================================================
\\ BRANCH PATCHING
\\ ============================================================

arm64_mark :
    pos arm64_pos

arm64_patch_bcond mark_pos :
    offset arm64_pos - mark_pos
    imm19 ((offset / 4) & 0x7FFFF) << 5
    old → 32 arm64_buf + mark_pos
    ← 32 arm64_buf + mark_pos old | imm19

arm64_patch_b mark_pos :
    offset arm64_pos - mark_pos
    imm26 (offset / 4) & 0x3FFFFFF
    old → 32 arm64_buf + mark_pos
    ← 32 arm64_buf + mark_pos old | imm26

arm64_patch_cbz mark_pos :
    offset arm64_pos - mark_pos
    imm19 ((offset / 4) & 0x7FFFF) << 5
    old → 32 arm64_buf + mark_pos
    ← 32 arm64_buf + mark_pos old | imm19

emit_bcond_fwd cond :
    mark arm64_pos
    emit_bcond cond 0

emit_b_fwd :
    mark arm64_pos
    emit_b 0

emit_cbz_fwd rt :
    mark arm64_pos
    emit_cbz rt 0

emit_cbnz_fwd rt :
    mark arm64_pos
    emit_cbnz rt 0

\\ ============================================================
\\ SYSCALL WRAPPERS
\\ ============================================================

emit_trap_exit :
    emit_movz X8 SYS_EXIT 0
    emit_trap

emit_trap_read :
    emit_movz X8 SYS_READ 0
    emit_trap

emit_trap_write :
    emit_movz X8 SYS_WRITE 0
    emit_trap

emit_trap_openat :
    emit_movz X8 SYS_OPENAT 0
    emit_trap

emit_trap_close :
    emit_movz X8 SYS_CLOSE 0
    emit_trap

emit_trap_mmap :
    emit_movz X8 SYS_MMAP 0
    emit_trap

emit_trap_munmap :
    emit_movz X8 SYS_MUNMAP 0
    emit_trap

emit_trap_mprotect :
    emit_movz X8 SYS_MPROTECT 0
    emit_trap

emit_trap_brk :
    emit_movz X8 SYS_BRK 0
    emit_trap

\\ ============================================================
\\ FUNCTION PROLOGUE / EPILOGUE
\\ ============================================================

emit_prologue :
    emit_stp_pre FP LR SP -16
    emit_mov FP SP

emit_epilogue :
    emit_ldp_post FP LR SP 16
    emit_ret_lr

emit_prologue_frame framesize :
    emit_stp_pre FP LR SP (0 - framesize)
    emit_mov FP SP

emit_epilogue_frame framesize :
    emit_ldp_post FP LR SP framesize
    emit_ret_lr

emit_save_pair r1 r2 offset :
    emit_stp r1 r2 SP offset

emit_restore_pair r1 r2 offset :
    emit_ldp r1 r2 SP offset

\\ ============================================================
\\ 32-BIT (W-REGISTER) ARITHMETIC
\\ ============================================================

emit_add_w rd rn rm :
    val 0x0B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_sub_w rd rn rm :
    val 0x4B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_add_imm_w rd rn imm :
    val 0x11000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

emit_sub_imm_w rd rn imm :
    val 0x51000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

emit_subs_imm_w rd rn imm :
    val 0x71000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

emit_cmp_imm_w rn imm :
    emit_subs_imm_w XZR rn imm

emit_cmp_w rn rm :
    val 0x6B000000 | (rm << 16) | (rn << 5) | XZR
    arm64_emit32 val

emit_mul_w rd rn rm :
    val 0x1B007C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_sdiv_w rd rn rm :
    val 0x1AC00C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_udiv_w rd rn rm :
    val 0x1AC00800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_and_w rd rn rm :
    val 0x0A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_orr_w rd rn rm :
    val 0x2A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_eor_w rd rn rm :
    val 0x4A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

emit_movz_w rd imm16 hw :
    val 0x52800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

emit_movk_w rd imm16 hw :
    val 0x72800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

emit_mov32 rd imm :
    chunk0 imm & 0xFFFF
    chunk1 (imm >> 16) & 0xFFFF
    emit_movz_w rd chunk0 0
    if chunk1 > 0
        emit_movk_w rd chunk1 1

\\ ============================================================
\\ EXTEND / EXTRACT
\\ ============================================================

emit_sxtw rd rn :
    val 0x93407C00 | (rn << 5) | rd
    arm64_emit32 val

emit_sxtb rd rn :
    val 0x93401C00 | (rn << 5) | rd
    arm64_emit32 val

emit_sxth rd rn :
    val 0x93403C00 | (rn << 5) | rd
    arm64_emit32 val

emit_uxtb rd rn :
    val 0x53001C00 | (rn << 5) | rd
    arm64_emit32 val

emit_uxth rd rn :
    val 0x53003C00 | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ ATOMIC / EXCLUSIVE
\\ ============================================================

emit_ldxr rt rn :
    val 0xC85F7C00 | (rn << 5) | rt
    arm64_emit32 val

emit_stxr rs rt rn :
    val 0xC8007C00 | (rs << 16) | (rn << 5) | rt
    arm64_emit32 val

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 2 — GPU BACKEND (emit-sm90)                                  ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================
\\ CODE BUFFER
\\ ============================================================

buf gpu_buf 524288
var gpu_pos 0
var max_reg 0

track_rd rd :
    max_reg max rd max_reg

\\ ============================================================
\\ COOPERATIVE GRID-SYNC STATE
\\ ============================================================

var gpu_cooperative 0

buf gridsync_offsets 1024
var gridsync_count 0

record_gridsync :
    if>= gridsync_count 256
        \\ silently clamp at 256 sites
    ← 32 gridsync_offsets + gridsync_count * 4 gpu_pos
    gridsync_count + 1

buf exit_offsets 1024
var exit_count 0

record_exit :
    if>= exit_count 256
        \\ silently clamp at 256 sites
    ← 32 exit_offsets + exit_count * 4 gpu_pos
    exit_count + 1

gpu_reset :
    gpu_pos 0
    max_reg 0
    gridsync_count 0
    exit_count 0
    gpu_cooperative 0

\\ ============================================================
\\ RAW BYTE EMITTERS
\\ ============================================================

gpu_emit_byte b :
    ← 8 gpu_buf + gpu_pos b
    gpu_pos + 1

gpu_emit_u32 val :
    gpu_emit_byte val & 0xFF
    gpu_emit_byte (val >> 8) & 0xFF
    gpu_emit_byte (val >> 16) & 0xFF
    gpu_emit_byte (val >> 24) & 0xFF

gpu_emit_u64 val :
    gpu_emit_u32 val & 0xFFFFFFFF
    gpu_emit_u32 val >> 32

\\ ============================================================
\\ sinst — CORE 16-BYTE INSTRUCTION EMITTER
\\ ============================================================

sinst iword ctrl :
    gpu_emit_u64 iword
    gpu_emit_u64 ctrl

\\ ============================================================
\\ CONTROL WORD CONSTRUCTOR
\\ ============================================================

make_ctrl stall yield wbar rbar wait reuse extra41 :
    extra41
    | (stall << 41)
    | (yield << 45)
    | (wbar << 46)
    | (rbar << 49)
    | (wait << 52)
    | (reuse << 58)

\\ ============================================================
\\ FP32 ARITHMETIC OPCODES
\\ ============================================================

OP_FMUL      0x7220
OP_FADD      0x7221
OP_FFMA      0x7223
OP_FMNMX     0x7209
OP_FSETP     0x720B
OP_FADD_IMM  0x7421
OP_FMUL_IMM  0x7820

\\ ============================================================
\\ INTEGER ARITHMETIC OPCODES
\\ ============================================================

OP_IMAD      0x7224
OP_IMAD_WIDE 0x7225
OP_IADD3     0x7210
OP_IADD3_IMM 0x7810
OP_SHF       0x7819

\\ ============================================================
\\ BIT OPERATION OPCODES
\\ ============================================================

OP_LOP3      0x7212
OP_LOP3_IMM  0x7812
OP_ISETP     0x720C

\\ ============================================================
\\ CONVERSION OPCODES
\\ ============================================================

OP_I2FP      0x7245
OP_F2I       0x7305

\\ ============================================================
\\ SFU (MUFU) OPCODE
\\ ============================================================

OP_MUFU      0x7308

\\ ============================================================
\\ MEMORY OPCODES
\\ ============================================================

OP_LDG       0x7981
OP_STG       0x7986
OP_LDS       0x7984
OP_STS       0x7988
OP_STS_REG   0x7388
OP_LDC       0x7B82
OP_ULDC      0x7AB9

\\ ============================================================
\\ ATOMIC / REDUCTION OPCODES
\\ ============================================================

OP_ATOMG_U32 0x79A8
OP_ATOMG_F32 0x79A3
OP_REDG_F32  0x79A6

\\ ============================================================
\\ WARP / CONTROL FLOW OPCODES
\\ ============================================================

OP_SHFL      0x7F89
OP_SHFL_IDX  0x7589
OP_SHFL_IDX_RR 0x7389
OP_BAR_SYNC  0x7B1D
OP_MEMBAR    0x7992
OP_BRA       0x7947
OP_EXIT      0x794D
OP_S2R       0x7919
OP_S2UR      0x79C3
OP_NOP       0x7918
OP_MOV_IMM   0x7802
OP_HFMA2     0x7235

\\ ============================================================
\\ MUFU SUBOP CONSTANTS
\\ ============================================================

MUFU_COS     0x0000
MUFU_SIN     0x0400
MUFU_EX2     0x0800
MUFU_LG2     0x0C00
MUFU_RCP     0x1000
MUFU_RSQ     0x1400
MUFU_SQRT    0x2000

\\ ============================================================
\\ LOP3 TRUTH TABLE CONSTANTS
\\ ============================================================

LOP3_AND     0xC0
LOP3_OR      0xFC
LOP3_XOR     0x3C

\\ ============================================================
\\ SPECIAL REGISTER IDs
\\ ============================================================

SR_TID_X     0x21
SR_TID_Y     0x22
SR_TID_Z     0x23
SR_CTAID_X   0x25
SR_CTAID_Y   0x26
SR_CTAID_Z   0x27
SR_LANEID    0x00

\\ ============================================================
\\ REGISTER ENCODING CONSTANTS
\\ ============================================================

RZ           0xFF
PT           0x07

\\ ============================================================
\\ PER-INSTRUCTION CONTROL WORD CONSTRUCTORS
\\ ============================================================

ctrl_nop :
    make_ctrl 0 0 7 7 0 0 0

ctrl_s2r sr_id :
    make_ctrl 7 1 1 7 0 0 (sr_id << 8)

ctrl_ldg :
    make_ctrl 4 1 2 7 0 0 0x0C1E1900

ctrl_ldg_wait0 :
    make_ctrl 4 1 2 7 1 0 0x0C1E1900

ctrl_stg :
    make_ctrl 1 1 7 7 0 0 0x0C101904

ctrl_stg_wait2 :
    make_ctrl 4 1 7 7 1 0 0x0C101904

ctrl_ffma rs3 :
    make_ctrl 5 0 7 7 0 0 rs3

ctrl_fadd :
    make_ctrl 5 0 7 7 0 0 0

ctrl_fadd_wait2 :
    make_ctrl 5 0 7 7 4 0 0

ctrl_fmul :
    make_ctrl 5 0 7 7 4 0 0x400000

ctrl_imad rs3 :
    make_ctrl 1 1 7 7 0 0 (0x0F8E0200 | rs3)

ctrl_imad_imm :
    make_ctrl 1 1 7 7 0 0 0x078E00FF

ctrl_isetp :
    make_ctrl 13 0 7 7 0 0 0x0BF06070

ctrl_bra :
    make_ctrl 0 0 7 7 0 0 0x0383FFFF

ctrl_exit :
    make_ctrl 5 1 7 7 0 0 0x03800000

ctrl_uldc :
    make_ctrl 1 1 7 7 0 0 0x0A00

ctrl_ldc :
    make_ctrl 7 1 1 7 0 0 0x0800

ctrl_shfl :
    make_ctrl 14 0 3 7 0 0 0x000E0000

ctrl_sts :
    make_ctrl 1 1 7 7 0 0 0x0800

ctrl_lds :
    make_ctrl 7 1 1 7 0 0 0x0800

ctrl_bar :
    make_ctrl 5 1 7 7 0 0 0x00010000

ctrl_shf :
    make_ctrl 1 0 7 7 0 0 0

ctrl_i2f :
    make_ctrl 5 0 7 7 4 0 0x00201400

ctrl_f2i :
    make_ctrl 2 1 0 7 4 0 0x0020F100

ctrl_lop3 lut :
    make_ctrl 1 0 7 7 0 0 (lut << 8)

ctrl_mufu subfn wbar :
    make_ctrl 8 1 wbar 7 0 0 subfn

\\ ============================================================
\\ PRE-COMPUTED CONTROL WORD CONSTANTS
\\ ============================================================

CTRL_NOP          0x000FC00000000000
CTRL_LDG          0x000EA8000C1E1900
CTRL_LDG_WAIT0    0x001EA8000C1E1900
CTRL_STG          0x000FE2000C101904
CTRL_STG_WAIT2    0x001FE8000C101904
CTRL_LDS          0x000E280008000800
CTRL_STS          0x000FE80008000804
CTRL_LDC          0x000E220000000800
CTRL_ULDC         0x000FE20000000A00
CTRL_FADD         0x000FCA0000000000
CTRL_FADD_WAIT2   0x004FCA0000000000
CTRL_FMUL         0x004FCA0000400000
CTRL_IMAD_IMM     0x000FE200078E00FF
CTRL_ISETP        0x000FDA000BF06070
CTRL_I2FP_S32     0x004FCA0000201400
CTRL_I2FP_U32     0x004FCA0000201000
CTRL_F2I          0x004E24000020F100
CTRL_SHFL         0x001E6800000E0000
CTRL_BAR          0x000FEC0000010000
CTRL_MEMBAR_GPU   0x000FEC0000002000
CTRL_MEMBAR_SYS   0x000FEC0000003000
CTRL_MEMBAR_CTA   0x0003EC0000000000
CTRL_BRA          0x000FC0000383FFFF
CTRL_EXIT         0x000FEA0003800000
CTRL_ATOMG_U32    0x004E2800081EE1C4
CTRL_ATOMG_F32    0x004E2800081EF3C4
CTRL_ATOMG_EXCH   0x004E28000C1EE1C4
CTRL_REDG_F32     0x004FE2000C10F384

\\ ============================================================
\\ INSTRUCTION BUILDERS — ARITHMETIC
\\ ============================================================

\\ NOP — GPU no operation
gpu_emit_nop :
    sinst 0x0000000000007918 ctrl_nop

\\ FADD Rd, Ra, Rb — float32 add
emit_fadd rd ra rb :
    iword OP_FADD | (track_rd rd << 16) | (ra << 24) | (rb << 32)
    sinst iword ctrl_fadd

\\ FFMA Rd, Rs1, Rs2, Rs3 — float fused multiply-add
emit_ffma rd rs1 rs2 rs3 :
    iword OP_FFMA | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_ffma rs3)

\\ FMUL Rd, Rs1, Rs2 — float multiply
emit_fmul rd rs1 rs2 :
    emit_ffma rd rs1 rs2 RZ

\\ FMNMX Rd, Ra, Rb — float min/max
emit_fmnmx rd ra rb :
    iword OP_FMNMX | (track_rd rd << 16) | (ra << 24) | (rb << 32)
    sinst iword ctrl_fadd

\\ FADD.IMM Rd, Ra, imm32
emit_fadd_imm rd ra imm32 :
    iword OP_FADD_IMM | (track_rd rd << 16) | (ra << 24) | (imm32 << 32)
    sinst iword ctrl_fadd

\\ FMUL.IMM Rd, Ra, imm32
emit_fmul_imm rd ra imm32 :
    iword OP_FMUL_IMM | (track_rd rd << 16) | (ra << 24) | (imm32 << 32)
    sinst iword ctrl_fmul

\\ FSETP Pd, Ra, Rb — float set predicate
emit_fsetp pd ra rb :
    iword OP_FSETP | (pd << 16) | (ra << 24) | (rb << 32)
    sinst iword ctrl_isetp

\\ ============================================================
\\ INSTRUCTION BUILDERS — INTEGER
\\ ============================================================

\\ IMAD Rd, Rs1, Rs2, Rs3
emit_imad rd rs1 rs2 rs3 :
    iword OP_IMAD | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_imad rs3)

\\ IMAD.MOV.U32 Rd, Rs1, imm32
emit_imad_imm rd rs1 imm32 :
    iword 0x7424 | (track_rd rd << 16) | (rs1 << 24) | (imm32 << 32)
    sinst iword ctrl_imad_imm

\\ IADD3 Rd, Rs1, Rs2, Rs3
emit_iadd3 rd rs1 rs2 rs3 :
    iword 0x7C10 | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    ctrl make_ctrl 1 1 7 7 0 0 rs3
    sinst iword ctrl

\\ ISETP.GE.U32.AND
emit_isetp_ge pd rs1 rs2 :
    iword OP_ISETP | (pd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword ctrl_isetp

\\ ISETP.LT.U32.AND
emit_isetp_lt pd rs1 rs2 :
    iword OP_ISETP | (pd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword ctrl_isetp

\\ SHF.R.U32 Rd, Rs, shamt
emit_shf_r rd rs shamt :
    iword OP_SHF | (track_rd rd << 16) | (rs << 24) | (shamt << 32) | (RZ << 48)
    sinst iword ctrl_shf

\\ ============================================================
\\ INSTRUCTION BUILDERS — LOGICAL (LOP3)
\\ ============================================================

emit_lop3_and rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_AND)

emit_lop3_or rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_OR)

emit_lop3_xor rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_XOR)

\\ ============================================================
\\ INSTRUCTION BUILDERS — CONVERSION
\\ ============================================================

\\ I2FP.F32.S32 Rd, Rs
emit_i2f rd rs :
    iword OP_I2FP | (track_rd rd << 16) | (rs << 32)
    sinst iword ctrl_i2f

\\ F2I.TRUNC.NTZ Rd, Rs
emit_f2i rd rs :
    iword OP_F2I | (track_rd rd << 16) | (rs << 32)
    sinst iword ctrl_f2i

\\ ============================================================
\\ INSTRUCTION BUILDERS — MUFU (Special Function Unit)
\\ ============================================================

gpu_emit_mufu rd rs subfn wbar :
    iword OP_MUFU | (track_rd rd << 16) | (rs << 32)
    sinst iword (ctrl_mufu subfn wbar)

emit_ex2 rd rs :
    gpu_emit_mufu rd rs MUFU_EX2 7

emit_rcp rd rs :
    gpu_emit_mufu rd rs MUFU_RCP 1

emit_rsq rd rs :
    gpu_emit_mufu rd rs MUFU_RSQ 2

emit_lg2 rd rs :
    gpu_emit_mufu rd rs MUFU_LG2 3

emit_sqrt rd rs :
    gpu_emit_mufu rd rs MUFU_SQRT 4

emit_sin rd rs :
    gpu_emit_mufu rd rs MUFU_SIN 5

emit_cos rd rs :
    gpu_emit_mufu rd rs MUFU_COS 5

\\ ============================================================
\\ INSTRUCTION BUILDERS — MEMORY
\\ ============================================================

\\ LDG.E Rd, [Ra]
emit_ldg rd ra :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24)
    sinst iword ctrl_ldg

\\ LDG.E Rd, [Ra+off]
emit_ldg_off rd ra off :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24) | (off << 32)
    sinst iword ctrl_ldg

\\ LDG.128 Rd, [Ra]
emit_ldg128 rd ra :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24)
    ctrl make_ctrl 4 1 2 7 0 0 0x0C1E1F00
    sinst iword ctrl

\\ STG.E [Ra], Rs
emit_stg ra rs :
    iword OP_STG | (ra << 24) | (rs << 32)
    sinst iword ctrl_stg

\\ STG.E [Ra+off], Rs
emit_stg_off ra rs off :
    iword OP_STG | (ra << 24) | (rs << 32) | (off << 32)
    sinst iword ctrl_stg

\\ LDS.32 Rd, [Ra+off]
emit_lds rd ra off :
    iword OP_LDS | (track_rd rd << 16) | (ra << 24) | (off << 32)
    sinst iword ctrl_lds

\\ STS.32 [Ra+off], Rs
emit_sts ra rs off :
    iword OP_STS | (ra << 24) | (rs << 32) | (off << 32)
    sinst iword ctrl_sts

\\ LDC Rd, c[cbuf][offset]
emit_ldc rd cbuf offset :
    iword OP_LDC | (track_rd rd << 16) | (RZ << 24) | (offset << 32) | (cbuf << 48)
    sinst iword CTRL_LDC

\\ ULDC URn, c[cbuf][offset]
emit_uldc rd cbuf offset :
    iword OP_ULDC | (track_rd rd << 16) | (RZ << 24) | (offset << 32) | (cbuf << 48)
    sinst iword CTRL_ULDC

\\ ============================================================
\\ INSTRUCTION BUILDERS — SPECIAL REGISTER
\\ ============================================================

\\ S2R Rd, sr_id
emit_s2r rd sr_id :
    iword OP_S2R | (track_rd rd << 16)
    sinst iword (ctrl_s2r sr_id)

\\ MOV Rd, imm32 — GPU immediate move
gpu_emit_mov_imm rd imm32 :
    iword OP_MOV_IMM | (track_rd rd << 16) | (imm32 << 32)
    sinst iword ctrl_nop

\\ ============================================================
\\ INSTRUCTION BUILDERS — HFMA2 (packed FP16x2)
\\ ============================================================

emit_hfma2 rd rs1 rs2 rs3 :
    iword OP_HFMA2 | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_ffma rs3)

\\ ============================================================
\\ INSTRUCTION BUILDERS — WARP SHUFFLES
\\ ============================================================

emit_shfl_bfly rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x0C | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

emit_shfl_down rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x08 | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

emit_shfl_up rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x04 | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

emit_shfl_idx rd rs r_lane :
    iword OP_SHFL_IDX_RR | (track_rd rd << 16) | (rs << 24) | (r_lane << 32)
    sinst iword ctrl_shfl

\\ ============================================================
\\ INSTRUCTION BUILDERS — BARRIERS
\\ ============================================================

emit_bar_sync bar_id :
    iword OP_BAR_SYNC | (bar_id << 54)
    sinst iword ctrl_bar

\\ ============================================================
\\ INSTRUCTION BUILDERS — MEMORY BARRIERS
\\ ============================================================

emit_membar_gpu :
    sinst 0x0000000000007992 CTRL_MEMBAR_GPU

emit_membar_sys :
    sinst 0x0000000000007992 CTRL_MEMBAR_SYS

emit_membar_cta :
    sinst 0x0000000000007992 CTRL_MEMBAR_CTA

\\ ============================================================
\\ INSTRUCTION BUILDERS — ATOMICS
\\ ============================================================

emit_atom_add rd ra rs :
    iword OP_ATOMG_U32 | (track_rd rd << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_ATOMG_U32

emit_atom_add_f32 rd ra rs :
    iword OP_ATOMG_F32 | (track_rd rd << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_ATOMG_F32

emit_redg_f32 ra rs :
    iword OP_REDG_F32 | (RZ << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_REDG_F32

\\ ============================================================
\\ INSTRUCTION BUILDERS — BRANCH / CONTROL FLOW
\\ ============================================================

emit_bra byte_offset :
    offset32 byte_offset / 4
    iword 0x00FC7947 | (offset32 << 32)
    sinst iword ctrl_bra

emit_bra_pred byte_offset pred :
    offset32 byte_offset / 4
    iword 0x00FC7947 | (offset32 << 32)
    iword_masked iword & 0xFFFFFFFFFFFF0FFF | (pred << 12)
    sinst iword_masked ctrl_bra

emit_exit :
    record_exit
    sinst 0x000000000000794D ctrl_exit

\\ ============================================================
\\ BRANCH PATCHING HELPERS
\\ ============================================================

gpu_mark :
    gpu_pos

gpu_patch saved_pos :
    byte_offset gpu_pos - saved_pos - 16
    offset32 byte_offset / 4
    ← 32 gpu_buf + saved_pos + 4 offset32

\\ ============================================================
\\ ACQUIRE / RELEASE MEMORY PROTOCOL
\\ ============================================================

emit_stg_release ra rs :
    emit_membar_gpu
    emit_stg ra rs

emit_ldg_acquire rd ra :
    emit_ldg rd ra
    emit_membar_gpu

\\ ============================================================
\\ GPTQ DEQUANTIZATION PRIMITIVES
\\ ============================================================

emit_dequant_nibble rd rsrc nib_idx rscale rzero :
    shamt nib_idx * 4
    emit_shf_r rd rsrc shamt
    emit_lop3_and rd rd 0xF
    emit_i2f rd rd
    emit_ffma rd rd rscale rzero

\\ ============================================================
\\ WARP SUM-REDUCE (Σ)
\\ ============================================================

emit_warp_reduce acc tmp :
    emit_shfl_bfly tmp acc 16
    emit_fadd acc acc tmp
    emit_shfl_bfly tmp acc 8
    emit_fadd acc acc tmp
    emit_shfl_bfly tmp acc 4
    emit_fadd acc acc tmp
    emit_shfl_bfly tmp acc 2
    emit_fadd acc acc tmp
    emit_shfl_bfly tmp acc 1
    emit_fadd acc acc tmp

\\ ============================================================
\\ COOPERATIVE GRID-SYNC
\\ ============================================================

var _gs_r 4
var _gs_p 0

gs_rreg :
    r _gs_r
    _gs_r + 1

gs_preg :
    p _gs_p
    _gs_p + 1

emit_grid_sync ctr_reg flag_reg grid_size :
    gpu_cooperative 1
    record_gridsync

    gs_old gs_rreg
    gs_exp gs_rreg
    gs_poll gs_rreg
    gs_pp gs_preg
    gs_pp2 gs_preg
    gs_tid gs_rreg

    emit_s2r gs_tid SR_TID_X
    gpu_emit_mov_imm gs_exp (grid_size - 1)

    tmp gs_rreg
    gpu_emit_mov_imm tmp 1

    emit_isetp_ge gs_pp2 gs_tid tmp
    emit_bra_pred 208 gs_pp2

    atom_one gs_rreg
    gpu_emit_mov_imm atom_one 1
    emit_atom_add gs_old ctr_reg atom_one
    emit_isetp_ge gs_pp gs_old gs_exp
    emit_bra_pred 64 (gs_pp | 8)

    emit_membar_gpu
    flag_one gs_rreg
    gpu_emit_mov_imm flag_one 1
    emit_stg flag_reg flag_one
    emit_bra 80

    emit_ldg gs_poll flag_reg
    emit_membar_gpu
    tmp2 gs_rreg
    gpu_emit_mov_imm tmp2 1
    emit_isetp_ge gs_pp2 gs_poll tmp2
    emit_bra_pred -80 (gs_pp2 | 8)

    emit_bar_sync 0

\\ ============================================================
\\ MEGAKERNEL PARAM OFFSET HELPERS
\\ ============================================================

sync_counter_param_offset n_data_params :
    n_data_params * 8 + 0x210

done_flag_param_offset n_data_params :
    n_data_params * 8 + 0x218

N_SYNC_PARAMS 2

total_kparams n_data_params :
    n_data_params + N_SYNC_PARAMS

\\ ============================================================
\\ CUBIN BUILDER STATE
\\ ============================================================

\\ TODO: bump to 335544320 (320MB) for full 64-layer megakernel
buf cubin_buf 4194304
var cubin_pos 0
var gpu_shmem_size 0
var gpu_n_kparams 0

buf li_name_buf 64
var li_name_len 0

var shstrtab_off 0
var shstrtab_size 0
var strtab_off 0
var strtab_size 0
var strsym_kernel 0
var symtab_off 0
var symtab_size 0
var sym_kernel_off 0
var nvinfo_off 0
var nvinfo_size 0
var nvinfo_k_off 0
var nvinfo_k_size 0
var text_off 0
var text_size 0
var const0_off 0
var const0_size 0
var shdrs_off 0

var SN_shstrtab 0
var SN_strtab 0
var SN_symtab 0
var SN_nvinfo 0
var SN_nvinfo_k 0
var SN_text 0
var SN_const0 0
var SN_shared 0

cubin_emit_byte b :
    ← 8 cubin_buf + cubin_pos b
    cubin_pos + 1

cubin_emit_u16 val :
    cubin_emit_byte val & 0xFF
    cubin_emit_byte (val >> 8) & 0xFF

cubin_emit_u32 val :
    cubin_emit_u16 val & 0xFFFF
    cubin_emit_u16 val >> 16

cubin_emit_u64 val :
    cubin_emit_u32 val & 0xFFFFFFFF
    cubin_emit_u32 val >> 32

cubin_reset :
    cubin_pos 0

\\ ============================================================
\\ EIATTR CONSTANTS
\\ ============================================================

NVI_FMT_U32              0x04
NVI_FMT_FLAG             0x03
EIATTR_REGCOUNT          0x2F
EIATTR_FRAME_SIZE        0x11
EIATTR_MIN_STACK_SIZE    0x12
EIATTR_CUDA_API_VERSION  0x37
EIATTR_KPARAM_INFO       0x17
EIATTR_PARAM_CBANK       0x0A
EIATTR_EXIT_INSTR_OFFSETS 0x1C
EIATTR_CBANK_PARAM_SIZE  0x19
EIATTR_MAXREG_COUNT      0x1B
EIATTR_SPARSE_MMA_MASK   0x50
EIATTR_SW_WAR            0x36
EIATTR_CRS_STACK_SIZE    0x23
EIATTR_COOP_GROUP_INSTR_OFFSETS 0x28
EIATTR_COOP_GROUP_MASK_REGIDS   0x29

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 3 — LEXER (lithos-lexer)                                     ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================================
\\ Emit a token into the output buffer
\\ ============================================================================

emit_token type offset length :
    idx token_count * 3
    ← 32 tokens + idx type
    idx1 idx + 1
    ← 32 tokens + idx1 offset
    idx2 idx + 2
    ← 32 tokens + idx2 length
    token_count token_count + 1

\\ ============================================================================
\\ Character classification
\\ ============================================================================

is_alpha c :
    upper c >= 65 & c <= 90
    lower c >= 97 & c <= 122
    under c == 95
    result upper | lower | under

is_digit c :
    result c >= 48 & c <= 57

is_hex_digit c :
    digit c >= 48 & c <= 57
    upper c >= 65 & c <= 70
    lower c >= 97 & c <= 102
    result digit | upper | lower

is_alnum c :
    alpha is_alpha c
    digit is_digit c
    result alpha | digit

is_space c :
    sp c == 32
    tb c == 9
    result sp | tb

is_newline c :
    lf c == 10
    cr c == 13
    result lf | cr

\\ ============================================================================
\\ Token scanning helpers
\\ ============================================================================

scan_ident src pos end :
    new_pos pos
    while new_pos < end
        c src [ new_pos ]          \\ BYTE LOAD: ldrb
        alpha is_alnum c
        dot c == 46
        angle_l c == 60
        angle_r c == 62
        eq c == 61
        bang c == 33
        ok alpha | dot | angle_l | angle_r | eq | bang
        if ok == 0
            return
        new_pos new_pos + 1

scan_number src pos end :
    new_pos pos
    c src [ new_pos ]

    if c == 48
        if new_pos + 1 < end
            c2 src [ new_pos + 1 ]
            if c2 == 120 | c2 == 88
                new_pos new_pos + 2
                while new_pos < end
                    ch src [ new_pos ]
                    ok is_hex_digit ch
                    if ok == 0
                        return
                    new_pos new_pos + 1
                return

    while new_pos < end
        ch src [ new_pos ]
        digit is_digit ch
        dot ch == 46
        ok digit | dot
        if ok == 0
            return
        new_pos new_pos + 1

scan_to_eol src pos end :
    new_pos pos
    while new_pos < end
        c src [ new_pos ]
        nl is_newline c
        if nl
            return
        new_pos new_pos + 1

\\ ============================================================================
\\ Keyword matching
\\ ============================================================================

match_keyword src offset length :
    tok_type 5

    if length == 2
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        if b0 == 111 & b1 == 114         \\ 'or'
            tok_type 5
            return

    if length == 3
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        b2 src [ offset + 2 ]
        if b0 == 102 & b1 == 111 & b2 == 114
            tok_type 16
            return
        if b0 == 118 & b1 == 97 & b2 == 114
            tok_type 23
            return
        if b0 == 98 & b1 == 117 & b2 == 102
            tok_type 24
            return
        if b0 == 102 & b1 == 51 & b2 == 50
            tok_type 40
            return
        if b0 == 117 & b1 == 51 & b2 == 50
            tok_type 41
            return
        if b0 == 115 & b1 == 51 & b2 == 50
            tok_type 42
            return
        if b0 == 102 & b1 == 49 & b2 == 54
            tok_type 43
            return
        if b0 == 112 & b1 == 116 & b2 == 114
            tok_type 44
            return

    if length == 4
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        b2 src [ offset + 2 ]
        b3 src [ offset + 3 ]
        if b0 == 101 & b1 == 97 & b2 == 99 & b3 == 104
            tok_type 18
            return
        if b0 == 101 & b1 == 108 & b2 == 115 & b3 == 101
            tok_type 14
            return
        if b0 == 101 & b1 == 108 & b2 == 105 & b3 == 102
            tok_type 15
            return
        if b0 == 118 & b1 == 111 & b2 == 105 & b3 == 100
            tok_type 45
            return
        if b0 == 98 & b1 == 105 & b2 == 110 & b3 == 100
            tok_type 27
            return
        if b0 == 101 & b1 == 120 & b2 == 105 & b3 == 116
            tok_type 34
            return
        if b0 == 104 & b1 == 111 & b2 == 115 & b3 == 116
            tok_type 35
            return
        \\ 'goto' -> 93
        if b0 == 103 & b1 == 111 & b2 == 116 & b3 == 111
            tok_type 93
            return

    if length == 5
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        b2 src [ offset + 2 ]
        b3 src [ offset + 3 ]
        b4 src [ offset + 4 ]
        if b0 == 112 & b1 == 97 & b2 == 114 & b3 == 97 & b4 == 109
            tok_type 12
            return
        if b0 == 119 & b1 == 104 & b2 == 105 & b3 == 108 & b4 == 101
            tok_type 20
            return
        if b0 == 99 & b1 == 111 & b2 == 110 & b3 == 115 & b4 == 116
            tok_type 22
            return
        if b0 == 108 & b1 == 97 & b2 == 121 & b3 == 101 & b4 == 114
            tok_type 26
            return
        if b0 == 108 & b1 == 97 & b2 == 98 & b3 == 101 & b4 == 108
            tok_type 33
            return

    if length == 6
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        b2 src [ offset + 2 ]
        b3 src [ offset + 3 ]
        b4 src [ offset + 4 ]
        b5 src [ offset + 5 ]
        if b0 == 107 & b1 == 101 & b2 == 114 & b3 == 110 & b4 == 101 & b5 == 108
            tok_type 11
            return
        if b0 == 115 & b1 == 116 & b2 == 114 & b3 == 105 & b4 == 100 & b5 == 101
            tok_type 19
            return
        if b0 == 114 & b1 == 101 & b2 == 116 & b3 == 117 & b4 == 114 & b5 == 110
            tok_type 21
            return
        if b0 == 119 & b1 == 101 & b2 == 105 & b3 == 103 & b4 == 104 & b5 == 116
            tok_type 25
            return
        if b0 == 115 & b1 == 104 & b2 == 97 & b3 == 114 & b4 == 101 & b5 == 100
            tok_type 31
            return
        if b0 == 101 & b1 == 110 & b2 == 100 & b3 == 102 & b4 == 111 & b5 == 114
            tok_type 17
            return

    if length == 7
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        if b0 == 114 & b1 == 117
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            if b2 == 110 & b3 == 116 & b4 == 105 & b5 == 109 & b6 == 101
                tok_type 28
                return
        if b0 == 98 & b1 == 97
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            if b2 == 114 & b3 == 114 & b4 == 105 & b5 == 101 & b6 == 114
                tok_type 32
                return
        if b0 == 112 & b1 == 114
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            if b2 == 111 & b3 == 106 & b4 == 101 & b5 == 99 & b6 == 116
                tok_type 30
                return
        \\ 'trap' -> 94
        if b0 == 115 & b1 == 121
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            if b2 == 115 & b3 == 99 & b4 == 97 & b5 == 108 & b6 == 108
                tok_type 94
                return

    if length == 8
        b0 src [ offset ]
        b1 src [ offset + 1 ]
        if b0 == 116 & b1 == 101
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            b7 src [ offset + 7 ]
            if b2 == 109 & b3 == 112 & b4 == 108 & b5 == 97 & b6 == 116 & b7 == 101
                tok_type 29
                return
        \\ 'constant' -> 96
        if b0 == 99 & b1 == 111
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            b7 src [ offset + 7 ]
            if b2 == 110 & b3 == 115 & b4 == 116 & b5 == 97 & b6 == 110 & b7 == 116
                tok_type 96
                return
        \\ 'continue' -> 95
        if b0 == 99 & b1 == 111
            b2 src [ offset + 2 ]
            b3 src [ offset + 3 ]
            b4 src [ offset + 4 ]
            b5 src [ offset + 5 ]
            b6 src [ offset + 6 ]
            b7 src [ offset + 7 ]
            if b2 == 110 & b3 == 116 & b4 == 105 & b5 == 110 & b6 == 117 & b7 == 101
                tok_type 95
                return

\\ ============================================================================
\\ Number type classification
\\ ============================================================================

classify_number src offset length :
    tok_type 3
    i 0
    while i < length
        c src [ offset + i ]
        if c == 46
            tok_type 4
            return
        i i + 1

\\ ============================================================================
\\ Main lexer
\\ ============================================================================

lex src src_len :
    pos 0
    token_count 0
    line_start 1

    while pos < src_len
        c src [ pos ]

        if line_start
            indent 0
            while pos < src_len
                ic src [ pos ]
                if ic == 32
                    indent indent + 1
                    pos pos + 1
                elif ic == 9
                    indent indent + 4
                    pos pos + 1
                else
                    goto indent_done
            label indent_done
            emit_token 2 pos indent
            line_start 0
            if pos >= src_len
                return
            c src [ pos ]

        if c == 10
            emit_token 1 pos 1
            pos pos + 1
            line_start 1
            if pos < src_len
                c2 src [ pos ]
                if c2 == 13
                    pos pos + 1
            continue

        if c == 13
            emit_token 1 pos 1
            pos pos + 1
            line_start 1
            if pos < src_len
                c2 src [ pos ]
                if c2 == 10
                    pos pos + 1
            continue

        if c == 32 | c == 9
            pos pos + 1
            continue

        if c == 35
            emit_token 78 pos 1
            pos pos + 1
            continue

        if c == 92
            if pos + 1 < src_len
                c2 src [ pos + 1 ]
                if c2 == 92
                    pos scan_to_eol src pos src_len
                    continue

        digit is_digit c
        if digit
            start pos
            pos scan_number src pos src_len
            length pos - start
            num_type classify_number src start length
            emit_token num_type start length
            continue

        if c == 45
            if pos + 1 < src_len
                cnext src [ pos + 1 ]
                ndig is_digit cnext
                if ndig
                    start pos
                    pos pos + 1
                    pos scan_number src pos src_len
                    length pos - start
                    num_type classify_number src start length
                    emit_token num_type start length
                    continue

        alpha is_alpha c
        if alpha
            start pos
            pos scan_ident src pos src_len
            length pos - start
            kw match_keyword src start length
            emit_token kw start length
            continue

        if pos + 1 < src_len
            cnext src [ pos + 1 ]
            if c == 61 & cnext == 61
                emit_token 55 pos 2
                pos pos + 2
                continue
            if c == 33 & cnext == 61
                emit_token 56 pos 2
                pos pos + 2
                continue
            if c == 60 & cnext == 61
                emit_token 59 pos 2
                pos pos + 2
                continue
            if c == 62 & cnext == 61
                emit_token 60 pos 2
                pos pos + 2
                continue
            if c == 60 & cnext == 60
                emit_token 64 pos 2
                pos pos + 2
                continue
            if c == 62 & cnext == 62
                emit_token 65 pos 2
                pos pos + 2
                continue

        \\ Multi-byte UTF-8 tokens
        if c == 0xE2
            if pos + 2 < src_len
                b1 src [ pos + 1 ]
                b2 src [ pos + 2 ]
                if b1 == 0x86 & b2 == 0x92
                    emit_token 36 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x86 & b2 == 0x90
                    emit_token 37 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x86 & b2 == 0x91
                    emit_token 38 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x86 & b2 == 0x93
                    emit_token 39 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x96 & b2 == 0xB3
                    emit_token 76 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x96 & b2 == 0xBD
                    emit_token 77 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x88 & b2 == 0x9A
                    emit_token 79 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x89 & b2 == 0x85
                    emit_token 80 pos 3
                    pos pos + 3
                    continue
                if b1 == 0x89 & b2 == 0xA1
                    emit_token 81 pos 3
                    pos pos + 3
                    continue
        if c == 0xCE
            if pos + 1 < src_len
                b1 src [ pos + 1 ]
                if b1 == 0xA3
                    emit_token 75 pos 2
                    pos pos + 2
                    continue

        \\ Single-character operators and punctuation
        if c == 43
            emit_token 50 pos 1
            pos pos + 1
            continue
        if c == 45
            emit_token 51 pos 1
            pos pos + 1
            continue
        if c == 42
            emit_token 52 pos 1
            pos pos + 1
            continue
        if c == 47
            emit_token 53 pos 1
            pos pos + 1
            continue
        if c == 61
            emit_token 54 pos 1
            pos pos + 1
            continue
        if c == 60
            emit_token 57 pos 1
            pos pos + 1
            continue
        if c == 62
            emit_token 58 pos 1
            pos pos + 1
            continue
        if c == 38
            emit_token 61 pos 1
            pos pos + 1
            continue
        if c == 124
            emit_token 62 pos 1
            pos pos + 1
            continue
        if c == 94
            emit_token 63 pos 1
            pos pos + 1
            continue
        if c == 91
            emit_token 67 pos 1
            pos pos + 1
            continue
        if c == 93
            emit_token 68 pos 1
            pos pos + 1
            continue
        if c == 40
            emit_token 69 pos 1
            pos pos + 1
            continue
        if c == 41
            emit_token 70 pos 1
            pos pos + 1
            continue
        if c == 44
            emit_token 71 pos 1
            pos pos + 1
            continue
        if c == 58
            emit_token 72 pos 1
            pos pos + 1
            continue
        if c == 46
            emit_token 73 pos 1
            pos pos + 1
            continue
        if c == 64
            emit_token 74 pos 1
            pos pos + 1
            continue
        \\ $ -> TOK_DOLLAR (97)
        if c == 36
            emit_token 97 pos 1
            pos pos + 1
            continue

        pos pos + 1

    emit_token 0 pos 0

\\ ============================================================================
\\ Lexer entry point
\\ ============================================================================

lithos_lex src src_len :
    lex src src_len

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 4 — PARSER (lithos-parser)                                   ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================================
\\ Parser state — global variables
\\ ============================================================================

var tok_pos 0
var tok_total 0
var src_buf 0
var emit_target 0          \\ 0 = GPU (sm90), 1 = HOST (ARM64)
var body_indent 0
var comp_depth 0
var error_count 0

\\ ============================================================================
\\ Symbol table
\\ ============================================================================

buf sym_names 16384
buf sym_lens 1024
buf sym_kinds 1024
buf sym_regs 1024
var n_syms 0

\\ ============================================================================
\\ Register allocators
\\ ============================================================================

var next_freg 0
var next_rreg 4
var next_rdreg 4
var next_preg 0
var next_host_reg 9

\\ ============================================================================
\\ Shared memory tracking
\\ ============================================================================

buf shm_names 2048
buf shm_sizes 32
var n_shared 0
var shmem_total 0

\\ ============================================================================
\\ Loop / branch tracking
\\ ============================================================================

buf branch_stack 2048
var branch_depth 0

\\ ============================================================================
\\ Composition table
\\ ============================================================================

buf comp_names 16384
buf comp_lens 1024
buf comp_tok_starts 1024
buf comp_arg_counts 1024
var n_comps 0

\\ ============================================================================
\\ Token stream access
\\ ============================================================================

peek_type :
    idx tok_pos * 3
    → 32 tokens idx

peek_offset :
    idx tok_pos * 3 + 1
    → 32 tokens idx

peek_length :
    idx tok_pos * 3 + 2
    → 32 tokens idx

consume :
    tok_pos tok_pos + 1

expect type :
    t peek_type
    if== t type
        consume

match_type type :
    t peek_type
    if== t type
        consume
        1
    0

skip_newlines :
    t peek_type
    if== t 1
        consume
        skip_newlines
    if== t 2
        consume
        skip_newlines

tok_text_ptr :
    off peek_offset
    src_buf + off

\\ ============================================================================
\\ Source text comparison
\\ ============================================================================

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

parse_int_token :
    ptr tok_text_ptr
    len peek_length
    consume

    if>= len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 48
            if== c1 120
                parse_hex ptr + 2 len - 2

    parse_decimal ptr len

parse_decimal ptr len :
    val 0
    i 0
    neg 0
    if>= len 1
        c → 8 ptr 0
        if== c 45
            neg 1
            i 1
    for i i len 1
        c → 8 ptr i
        d c - 48
        val val * 10 + d
    if== neg 1
        val 0 - val
    val

parse_hex ptr len :
    val 0
    for i 0 len 1
        c → 8 ptr i
        if>= c 97
            d c - 87
        if>= c 65
            d c - 55
            d c - 48
        val val * 16 + d
    val

parse_float_token :
    ptr tok_text_ptr
    len peek_length
    consume
    0

\\ ============================================================================
\\ Symbol table operations
\\ ============================================================================

sym_reset :
    n_syms 0

sym_add name_ptr name_len kind reg :
    idx n_syms
    if>= idx 64
        idx

    dest idx * 32
    ← 32 sym_lens idx * 4 name_len
    ← 32 sym_kinds idx * 4 kind
    ← 32 sym_regs idx * 4 reg

    clen name_len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 name_ptr j
        ← 8 sym_names dest + j b

    n_syms n_syms + 1

sym_find name_ptr name_len :
    for i 0 n_syms 1
        slen → 32 sym_lens i * 4
        if== slen name_len
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
    -1

sym_find_current :
    ptr tok_text_ptr
    len peek_length
    sym_find ptr len

sym_kind i :
    → 32 sym_kinds i * 4

sym_reg i :
    → 32 sym_regs i * 4

\\ ============================================================================
\\ Register allocation
\\ ============================================================================

alloc_freg :
    r next_freg
    next_freg next_freg + 1
    r

alloc_rreg :
    r next_rreg
    next_rreg next_rreg + 1
    r

alloc_rdreg :
    r next_rdreg
    next_rdreg next_rdreg + 1
    r

alloc_preg :
    r next_preg
    next_preg next_preg + 1
    r

alloc_host_reg :
    r next_host_reg
    next_host_reg next_host_reg + 1
    r

regs_reset :
    next_freg 0
    next_rreg 4
    next_rdreg 4
    next_preg 0
    next_host_reg 9

\\ ============================================================================
\\ Backend dispatch — dual target emission
\\ ============================================================================
\\ The parser calls these backend-neutral wrappers.
\\ emit_target selects which backend actually emits.

\\ emit_p_add : emit an add instruction (parser dispatch)
emit_p_add rd ra rb :
    if== emit_target 0
        \\ GPU sm90: FADD
        emit_fadd rd ra rb
    \\ HOST ARM64: ADD
        emit_add_reg rd ra rb

\\ emit_p_sub : emit a subtract instruction
emit_p_sub rd ra rb :
    if== emit_target 0
        emit_fadd rd ra rb       \\ TODO: FADD with NEG on src2
    emit_sub_reg rd ra rb

\\ emit_p_mul : emit a multiply instruction
emit_p_mul rd ra rb :
    if== emit_target 0
        emit_fmul rd ra rb
    emit_mul_reg rd ra rb

\\ emit_p_div : emit a divide instruction
emit_p_div rd ra rb :
    if== emit_target 0
        \\ GPU: rcp = 1/rb, then rd = ra * rcp
        rcp alloc_freg
        emit_rcp rcp rb
        emit_fmul rd ra rcp
    emit_sdiv rd ra rb

\\ emit_p_mov_imm : load an immediate value into a register
emit_p_mov_imm rd imm :
    if== emit_target 0
        gpu_emit_mov_imm rd imm
    emit_mov64 rd imm

\\ emit_p_mov_reg : register-to-register move
emit_p_mov_reg rd rs :
    if== emit_target 0
        \\ GPU: MOV via IMAD idiom
        emit_imad rd rs RZ RZ
    emit_mov rd rs

\\ ============================================================================
\\ Memory operations — backend dispatch
\\ ============================================================================

emit_p_load rd base offset width :
    if== emit_target 0
        emit_ldg rd base
    if== width 8
        emit_ldrb rd base offset
    if== width 16
        emit_ldrh rd base offset
    if== width 32
        emit_ldr_w rd base offset
    if== width 64
        emit_ldr rd base offset

emit_p_store rs base offset width :
    if== emit_target 0
        emit_stg base rs
    if== width 8
        emit_strb rs base offset
    if== width 16
        emit_strh rs base offset
    if== width 32
        emit_str_w rs base offset
    if== width 64
        emit_str rs base offset

\\ ============================================================================
\\ GPU-specific emitter wrappers (parser)
\\ ============================================================================

emit_p_s2r rd sr_id :
    emit_s2r rd sr_id

emit_p_shfl_bfly rd rs delta :
    emit_shfl_bfly rd rs delta

emit_p_bar_sync :
    emit_bar_sync 0

emit_p_membar :
    emit_membar_gpu

emit_p_mufu rd rs subop :
    \\ Map parser subop constants to MUFU subfn + wbar
    gpu_emit_mufu rd rs subop 7

emit_p_isetp pd ra rb :
    emit_isetp_ge pd ra rb

emit_p_bra offset :
    emit_bra offset

emit_p_bra_predicated pred offset :
    emit_bra_pred offset pred

emit_p_exit_gpu :
    emit_exit

emit_p_nop :
    gpu_emit_nop

\\ ARM64-specific emitters (parser)
emit_p_svc :
    emit_svc 0

emit_p_ret_host :
    emit_ret_lr

\\ MUFU subop constants (for parser use)
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

parse_expr :
    parse_add_expr

parse_add_expr :
    left parse_mul_expr
    t peek_type
    if== t 50
        consume
        right parse_mul_expr
        rd alloc_freg
        emit_p_add rd left right
        rd
    if== t 51
        consume
        right parse_mul_expr
        rd alloc_freg
        emit_p_sub rd left right
        rd
    left

parse_mul_expr :
    left parse_atom
    t peek_type
    if== t 52
        consume
        right parse_atom
        rd alloc_freg
        emit_p_mul rd left right
        rd
    if== t 53
        consume
        right parse_atom
        rd alloc_freg
        emit_p_div rd left right
        rd
    left

parse_atom :
    t peek_type

    if== t 3
        val parse_int_token
        rd alloc_freg
        emit_p_mov_imm rd val
        rd

    if== t 4
        val parse_float_token
        rd alloc_freg
        emit_p_mov_imm rd val
        rd

    \\ → width addr — memory load
    if== t 80
        consume
        width parse_int_token
        addr_reg parse_expr
        rd alloc_freg
        emit_p_load rd addr_reg 0 width
        rd

    \\ ↑ $N — register read
    if== t 82
        consume
        parse_regread

    \\ √ expr
    if== t 87
        consume
        src parse_expr
        rd alloc_freg
        emit_p_mufu rd src 0x2000
        rd

    \\ ≅ expr — sine
    if== t 88
        consume
        src parse_expr
        rd alloc_freg
        emit_p_mufu rd src 0x0400
        rd

    \\ ≡ expr — cosine
    if== t 89
        consume
        src parse_expr
        rd alloc_freg
        emit_p_mufu rd src 0x0000
        rd

    \\ Σ expr — sum reduction
    if== t 84
        consume
        src parse_expr
        parse_reduction_sum src

    \\ △ expr — max reduction
    if== t 85
        consume
        src parse_expr
        parse_reduction_max src

    \\ ▽ expr — min reduction
    if== t 86
        consume
        src parse_expr
        parse_reduction_min src

    \\ # modifier
    if== t 90
        consume
        parse_index_reduction

    \\ ** elementwise
    if== t 91
        consume
        parse_elementwise

    \\ Identifier
    if== t 5
        parse_ident_expr

    \\ Parenthesized expression
    if== t 69
        consume
        val parse_expr
        expect 70
        val

    -1

parse_ident_expr :
    ptr tok_text_ptr
    len peek_length

    \\ 1/ → MUFU.RCP
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 49
            if== c1 47
                consume
                src parse_expr
                rd alloc_freg
                emit_p_mufu rd src 0x1000
                rd

    \\ 2^ → MUFU.EX2
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 50
            if== c1 94
                consume
                src parse_expr
                rd alloc_freg
                emit_p_mufu rd src 0x0800
                rd

    \\ e^ → composite: multiply by log2(e) then 2^
    \\ ln → composite: log₂ then multiply by ln(2)
    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 101
            if== c1 94
                consume
                src parse_expr
                log2e alloc_freg
                emit_p_mov_imm log2e 0x3FB8AA3B
                tmp alloc_freg
                emit_p_mul tmp src log2e
                rd alloc_freg
                emit_p_mufu rd tmp 0x0800
                rd

        if== c0 108
            if== c1 110
                consume
                src parse_expr
                tmp alloc_freg
                emit_p_mufu tmp src 0x0c00
                ln2 alloc_freg
                emit_p_mov_imm ln2 0x3F317218
                rd alloc_freg
                emit_p_mul rd tmp ln2
                rd

    \\ Variable lookup
    idx sym_find ptr len
    if>= idx 0
        consume
        k sym_kind idx
        reg sym_reg idx

        nt peek_type
        if== nt 67
            consume
            idx_reg parse_expr
            expect 68
            offset_reg alloc_rreg
            four alloc_rreg
            emit_p_mov_imm four 4
            emit_p_mul offset_reg idx_reg four
            addr alloc_rreg
            emit_p_add addr reg offset_reg
            rd alloc_freg
            emit_p_load rd addr 0 32
            rd

        reg

    consume
    -1

\\ ============================================================================
\\ Register read/write parsing
\\ ============================================================================

parse_regread :
    t peek_type
    \\ Handle $N or $NAME (dollar-prefixed register)
    if== t 97
        consume
        t2 peek_type
        if== t2 3
            regnum parse_int_token
            rd alloc_rreg
            if== emit_target 1
                emit_mov rd regnum
            if== emit_target 0
                emit_p_s2r rd regnum
            rd
        if== t2 5
            ptr tok_text_ptr
            len peek_length
            consume
            rd alloc_rreg
            did dict_lookup ptr len
            if>= did 0
                sr_id → 32 dict_reg_ids did * 4
                if== emit_target 0
                    emit_p_s2r rd sr_id
                rd
            rd
        -1
    if== t 3
        regnum parse_int_token
        rd alloc_rreg
        if== emit_target 0
            emit_p_s2r rd regnum
        rd
    if== t 5
        ptr tok_text_ptr
        len peek_length
        consume
        rd alloc_rreg
        if== len 3
            c0 → 8 ptr 0
            if== c0 116
                emit_p_s2r rd 0x21
                rd
        rd
    -1

parse_regwrite :
    t peek_type
    \\ Handle $N or $NAME
    if== t 97
        consume
        t2 peek_type
        if== t2 3
            regnum parse_int_token
            val_reg parse_expr
            if== emit_target 1
                emit_mov regnum val_reg
            emit_p_mov_reg regnum val_reg
        if== t2 5
            ptr tok_text_ptr
            len peek_length
            consume
            val_reg parse_expr
            did dict_lookup ptr len
            if>= did 0
                sr_id → 32 dict_reg_ids did * 4
                if== emit_target 1
                    emit_mov sr_id val_reg
                emit_p_mov_reg sr_id val_reg
    regnum parse_int_token
    val_reg parse_expr
    if== emit_target 1
        emit_mov regnum val_reg
    emit_p_mov_reg regnum val_reg

\\ ============================================================================
\\ Reduction operations (GPU only)
\\ ============================================================================

parse_reduction_sum src :
    rd src
    shfl alloc_freg
    emit_p_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_p_add rd2 rd shfl
    shfl2 alloc_freg
    emit_p_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_p_add rd3 rd2 shfl2
    shfl3 alloc_freg
    emit_p_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_p_add rd4 rd3 shfl3
    shfl4 alloc_freg
    emit_p_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_p_add rd5 rd4 shfl4
    shfl5 alloc_freg
    emit_p_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_p_add rd6 rd5 shfl5
    rd6

parse_reduction_max src :
    rd src
    shfl alloc_freg
    emit_p_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_fmnmx_max rd2 rd shfl
    shfl2 alloc_freg
    emit_p_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_fmnmx_max rd3 rd2 shfl2
    shfl3 alloc_freg
    emit_p_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_fmnmx_max rd4 rd3 shfl3
    shfl4 alloc_freg
    emit_p_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_fmnmx_max rd5 rd4 shfl4
    shfl5 alloc_freg
    emit_p_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_fmnmx_max rd6 rd5 shfl5
    rd6

parse_reduction_min src :
    rd src
    shfl alloc_freg
    emit_p_shfl_bfly shfl rd 16
    rd2 alloc_freg
    emit_fmnmx_min rd2 rd shfl
    shfl2 alloc_freg
    emit_p_shfl_bfly shfl2 rd2 8
    rd3 alloc_freg
    emit_fmnmx_min rd3 rd2 shfl2
    shfl3 alloc_freg
    emit_p_shfl_bfly shfl3 rd3 4
    rd4 alloc_freg
    emit_fmnmx_min rd4 rd3 shfl3
    shfl4 alloc_freg
    emit_p_shfl_bfly shfl4 rd4 2
    rd5 alloc_freg
    emit_fmnmx_min rd5 rd4 shfl4
    shfl5 alloc_freg
    emit_p_shfl_bfly shfl5 rd5 1
    rd6 alloc_freg
    emit_fmnmx_min rd6 rd5 shfl5
    rd6

emit_fmnmx_max rd ra rb :
    iword OP_FMNMX | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword 0x000fca0000000000

emit_fmnmx_min rd ra rb :
    iword OP_FMNMX | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword (0x000fca0000000000 | 1 << 15)

parse_index_reduction :
    t peek_type
    if== t 85
        consume
        src parse_expr
        parse_argmax src
    if== t 86
        consume
        src parse_expr
        parse_argmin src
    -1

parse_argmax src :
    idx alloc_rreg
    emit_p_s2r idx 0x21
    val src
    shfl_v alloc_freg
    shfl_i alloc_rreg
    emit_p_shfl_bfly shfl_v val 16
    emit_p_shfl_bfly shfl_i idx 16
    p alloc_preg
    emit_fsetp_gt p val shfl_v
    new_val alloc_freg
    new_idx alloc_rreg
    emit_sel_f new_val p val shfl_v
    emit_sel_r new_idx p idx shfl_i
    new_idx

parse_argmin src :
    idx alloc_rreg
    emit_p_s2r idx 0x21
    val src
    shfl_v alloc_freg
    shfl_i alloc_rreg
    emit_p_shfl_bfly shfl_v val 16
    emit_p_shfl_bfly shfl_i idx 16
    p alloc_preg
    emit_fsetp_lt p val shfl_v
    new_val alloc_freg
    new_idx alloc_rreg
    emit_sel_f new_val p val shfl_v
    emit_sel_r new_idx p idx shfl_i
    new_idx

emit_fsetp_gt pd ra rb :
    iword 0x720b | ra << 24 | rb << 32 | pd << 16
    sinst iword 0x000fda000bf06070

emit_fsetp_lt pd ra rb :
    iword 0x720b | ra << 24 | rb << 32 | pd << 16
    sinst iword 0x000fda000bf02070

emit_sel_f rd pd ra rb :
    iword 0x7209 | rd << 16 | ra << 24 | rb << 32
    sinst iword (0x000fca0000000000 | pd << 12)

emit_sel_r rd pd ra rb :
    emit_p_mov_reg rd ra

parse_elementwise :
    parse_expr

\\ ============================================================================
\\ Statement parser
\\ ============================================================================

parse_stmt :
    skip_newlines

    t peek_type

    if== t 0
        0

    if== t 80
        consume
        width parse_int_token
        addr parse_expr
        rd alloc_freg
        emit_p_load rd addr 0 width
        rd

    if== t 81
        consume
        parse_store_stmt

    if== t 82
        consume
        parse_regread

    if== t 83
        consume
        parse_regwrite

    if== t 16
        consume
        parse_for

    if== t 18
        consume
        parse_each

    if== t 19
        consume
        parse_stride

    if== t 13
        parse_conditional

    if== t 32
        consume
        emit_p_bar_sync

    if== t 31
        consume
        parse_shared

    if== t 34
        consume
        if== emit_target 0
            emit_p_exit_gpu
        emit_p_ret_host

    \\ TOK_GOTO (93)
    if== t 93
        consume
        parse_goto

    \\ TOK_LABEL (33)
    if== t 33
        consume
        parse_label_decl

    \\ TOK_ENDFOR (17)
    if== t 17
        consume
        parse_for_end

    \\ TOK_RETURN (21)
    if== t 21
        consume
        if== emit_target 0
            emit_p_exit_gpu
        emit_p_ret_host

    \\ TOK_TRAP (94)
    if== t 94
        consume
        parse_trap_stmt

    \\ TOK_CONTINUE (95)
    if== t 95
        consume
        parse_continue_stmt

    \\ TOK_CONSTANT (96)
    if== t 96
        consume
        parse_constant_decl

    \\ TOK_PARAM (12)
    if== t 12
        consume
        parse_param

    if== t 5
        parse_ident_stmt

    parse_expr
    1

parse_store_stmt :
    width parse_int_token
    addr parse_expr
    val parse_expr
    emit_p_store val addr 0 width

\\ ============================================================================
\\ goto / label / continue handlers
\\ ============================================================================

parse_goto :
    ptr tok_text_ptr
    len peek_length
    consume
    lidx label_find ptr len
    if>= lidx 0
        target → 32 label_offsets lidx * 4
        if== emit_target 0
            offset target - gpu_pos
            emit_bra offset
        offset target - arm64_pos
            emit_b offset
    if== emit_target 0
        emit_bra 0
    emit_b 0

parse_label_decl :
    ptr tok_text_ptr
    len peek_length
    consume
    lidx n_labels
    clen len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 ptr j
        ← 8 label_names lidx * 32 + j b
    ← 32 label_lens lidx * 4 len
    if== emit_target 0
        ← 32 label_offsets lidx * 4 gpu_pos
    ← 32 label_offsets lidx * 4 arm64_pos
    n_labels n_labels + 1

parse_inline_label ptr len :
    lidx n_labels
    clen len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 ptr j
        ← 8 label_names lidx * 32 + j b
    ← 32 label_lens lidx * 4 len
    if== emit_target 0
        ← 32 label_offsets lidx * 4 gpu_pos
    ← 32 label_offsets lidx * 4 arm64_pos
    n_labels n_labels + 1

label_find name_ptr name_len :
    for i 0 n_labels 1
        slen → 32 label_lens i * 4
        if== slen name_len
            match 1
            for j 0 slen 1
                a → 8 name_ptr j
                b → 8 label_names i * 32 + j
                if!= a b
                    match 0
            if== match 1
                i
    -1

parse_continue_stmt :
    if== branch_depth 0
        0
    if== emit_target 0
        emit_p_bra 0
    arm64_emit32 0x14000000

\\ ============================================================================
\\ trap handler
\\ ============================================================================

parse_trap_stmt :
    result_ptr tok_text_ptr
    result_len peek_length
    consume
    sysnum parse_expr
    emit_mov64 X8 0
    emit_p_mov_reg X8 sysnum
    arg_idx 0
    parse_trap_args arg_idx
    emit_svc 0
    rd alloc_rreg
    emit_mov rd X0
    sym_add result_ptr result_len 2 rd

parse_trap_args idx :
    t peek_type
    if== t 1
        idx
    if== t 0
        idx
    if>= idx 6
        idx
    val parse_expr
    emit_p_mov_reg idx val
    parse_trap_args idx + 1

\\ ============================================================================
\\ constant handler (Forth-style: VALUE constant NAME)
\\ ============================================================================

parse_constant_decl :
    name_ptr tok_text_ptr
    name_len peek_length
    consume
    rd alloc_rreg
    emit_p_mov_imm rd 0
    sym_add name_ptr name_len 1 rd

\\ ============================================================================
\\ param handler (param NAME TYPE)
\\ ============================================================================

parse_param :
    name_ptr tok_text_ptr
    name_len peek_length
    consume
    type_tok peek_type
    consume
    param_type 0
    if== type_tok 41
        param_type 1
    if== type_tok 44
        param_type 2
    if== type_tok 40
        param_type 3
    pidx n_kparams
    clen name_len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 name_ptr j
        ← 8 kparam_names pidx * 32 + j b
    ← 32 kparam_lens pidx * 4 name_len
    ← 32 kparam_types pidx * 4 param_type
    rd alloc_rdreg
    if== emit_target 0
        param_offset 528 + pidx * 8
        emit_uldc rd 0 param_offset
    sym_add name_ptr name_len 6 rd
    n_kparams n_kparams + 1
    gpu_n_kparams n_kparams

\\ ============================================================================
\\ memory load/store magic identifier dispatch
\\ ============================================================================

parse_mem_load width :
    result_ptr tok_text_ptr
    result_len peek_length
    consume
    addr parse_expr
    rd alloc_rreg
    emit_p_load rd addr 0 width
    sym_add result_ptr result_len 2 rd

parse_mem_store width :
    dest_ptr tok_text_ptr
    dest_len peek_length
    consume
    idx_expr parse_expr
    val_expr parse_expr
    base_idx sym_find dest_ptr dest_len
    if>= base_idx 0
        base_reg sym_reg base_idx
        offset_reg alloc_rreg
        elem_bytes width / 8
        scale alloc_rreg
        emit_p_mov_imm scale elem_bytes
        emit_p_mul offset_reg idx_expr scale
        addr alloc_rreg
        emit_p_add addr base_reg offset_reg
        emit_p_store val_expr addr 0 width

parse_mem_load_idx width :
    base parse_expr
    idx_expr parse_expr
    elem_bytes width / 8
    offset alloc_rreg
    scale alloc_rreg
    emit_p_mov_imm scale elem_bytes
    emit_p_mul offset idx_expr scale
    addr alloc_rreg
    emit_p_add addr base offset
    rd alloc_rreg
    emit_p_load rd addr 0 width
    rd

parse_mem_store_at :
    buf_expr parse_expr
    off_expr parse_expr
    val_expr parse_expr
    addr alloc_rreg
    emit_p_add addr buf_expr off_expr
    emit_p_store val_expr addr 0 8

\\ ============================================================================
\\ 3-operand integer statement ops
\\ ============================================================================

parse_3op_and :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    if== emit_target 0
        emit_lop3_and rd a b
    emit_and_reg rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_or :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    if== emit_target 0
        emit_lop3_or rd a b
    emit_orr_reg rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_xor :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    if== emit_target 0
        emit_lop3_xor rd a b
    emit_eor_reg rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_shl :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    if== emit_target 0
        emit_shf_r rd a b
    emit_lsl_reg rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_shr :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    if== emit_target 0
        emit_shf_r rd a b
    emit_lsr_reg rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_mul :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    emit_p_mul rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_add :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    emit_p_add rd a b
    sym_add dst_ptr dst_len 2 rd

parse_3op_sub :
    dst_ptr tok_text_ptr
    dst_len peek_length
    consume
    a parse_expr
    b parse_expr
    rd alloc_rreg
    emit_p_sub rd a b
    sym_add dst_ptr dst_len 2 rd

parse_ident_stmt :
    ptr tok_text_ptr
    len peek_length

    \\ Check for 'trap'
    if== len 4
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        c2 → 8 ptr 2
        c3 → 8 ptr 3
        if== c0 116
            if== c1 114
                if== c2 97
                    if== c3 112
                        consume
                        emit_p_svc

    \\ Check for conditional keywords: if==, if>=, if<, if!=, if>, if<=
    if>= len 3
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 105
            if== c1 102
                parse_conditional

    \\ Check for inline label (ident followed by TOK_COLON)
    consume
    nt_check peek_type
    if== nt_check 72
        consume
        parse_inline_label ptr len
        1
    \\ Put token back
    tok_pos tok_pos - 1

    \\ Check for mem_load 8 (7 chars)
    if== len 7
        c0 → 8 ptr 0
        if== c0 108
            c4 → 8 ptr 4
            if== c4 95
                c6 → 8 ptr 6
                if== c6 56
                    consume
                    parse_mem_load 8
                    1

    \\ Check for mem_load 16/32/64 (8 chars)
    if== len 8
        c0 → 8 ptr 0
        if== c0 108
            c4 → 8 ptr 4
            if== c4 95
                c6 → 8 ptr 6
                c7 → 8 ptr 7
                consume
                if== c6 49
                    if== c7 54
                        parse_mem_load 16
                        1
                if== c6 51
                    if== c7 50
                        parse_mem_load 32
                        1
                if== c6 54
                    if== c7 52
                        parse_mem_load 64
                        1

    \\ Check for mem_store 8 (8 chars)
    if== len 8
        c0 → 8 ptr 0
        if== c0 115
            c5 → 8 ptr 5
            if== c5 95
                c7 → 8 ptr 7
                if== c7 56
                    consume
                    parse_mem_store 8
                    1

    \\ Check for mem_store 16/32/64 (9 chars)
    if== len 9
        c0 → 8 ptr 0
        if== c0 115
            c5 → 8 ptr 5
            if== c5 95
                c7 → 8 ptr 7
                c8 → 8 ptr 8
                consume
                if== c7 49
                    if== c8 54
                        parse_mem_store 16
                        1
                if== c7 51
                    if== c8 50
                        parse_mem_store 32
                        1
                if== c7 54
                    if== c8 52
                        parse_mem_store 64
                        1

    \\ Check for mem_load 64_idx / mem_load 32_idx (12 chars)
    if== len 12
        c0 → 8 ptr 0
        if== c0 108
            c5 → 8 ptr 5
            if== c5 117
                c7 → 8 ptr 7
                c8 → 8 ptr 8
                consume
                if== c7 54
                    if== c8 52
                        parse_mem_load_idx 64
                if== c7 51
                    if== c8 50
                        parse_mem_load_idx 32

    \\ Check for mem_store 8_at (11 chars)
    if== len 11
        c0 → 8 ptr 0
        if== c0 115
            c8 → 8 ptr 8
            if== c8 97
                consume
                parse_mem_store_at
                1

    \\ Check for 3-operand ops
    if== len 3
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        c2 → 8 ptr 2
        if== c0 97
            if== c1 110
                if== c2 100
                    consume
                    parse_3op_and
                    1
        if== c0 115
            if== c1 104
                if== c2 108
                    consume
                    parse_3op_shl
                    1
        if== c0 115
            if== c1 104
                if== c2 114
                    consume
                    parse_3op_shr
                    1
        if== c0 109
            if== c1 117
                if== c2 108
                    consume
                    parse_3op_mul
                    1
        if== c0 97
            if== c1 100
                if== c2 100
                    consume
                    parse_3op_add
                    1
        if== c0 115
            if== c1 117
                if== c2 98
                    consume
                    parse_3op_sub
                    1
        if== c0 120
            if== c1 111
                if== c2 114
                    consume
                    parse_3op_xor
                    1

    if== len 2
        c0 → 8 ptr 0
        c1 → 8 ptr 1
        if== c0 111
            if== c1 114
                consume
                parse_3op_or
                1

    \\ Look up in symbol table
    idx sym_find ptr len
    if>= idx 0
        consume
        nt peek_type
        if== nt 54
            consume
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1

        if== nt 67
            consume
            idx_expr parse_expr
            expect 68
            expect 54
            val parse_expr
            base sym_reg idx
            offset_reg alloc_rreg
            four alloc_rreg
            emit_p_mov_imm four 4
            emit_p_mul offset_reg idx_expr four
            addr alloc_rreg
            emit_p_add addr base offset_reg
            emit_p_store val addr 0 32
            1

        nt2 peek_type
        if== nt2 80
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1
        if== nt2 84
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1
        if== nt2 85
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1
        if== nt2 86
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1
        if== nt2 87
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1
        if== nt2 90
            val parse_expr
            reg sym_reg idx
            emit_p_mov_reg reg val
            1

        val parse_expr
        reg sym_reg idx
        emit_p_mov_reg reg val
        1

    \\ New variable assignment
    consume
    nt peek_type
    if== nt 80
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 84
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 85
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 86
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 87
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 3
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 5
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 88
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 89
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 90
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    if== nt 91
        val parse_expr
        rd alloc_freg
        emit_p_mov_reg rd val
        sym_add ptr len 2 rd
        1
    skip_to_eol
    0

skip_to_eol :
    t peek_type
    if== t 0
        0
    if== t 1
        consume
    consume
    skip_to_eol

\\ ============================================================================
\\ Control flow parsing
\\ ============================================================================

parse_for :
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    start_reg parse_expr
    end_reg parse_expr
    step_reg parse_expr

    loop_reg alloc_rreg
    emit_p_mov_reg loop_reg start_reg
    sym_add var_ptr var_len 3 loop_reg

    push_branch loop_reg end_reg step_reg

    if== emit_target 0
        p alloc_preg
        emit_p_isetp p loop_reg end_reg
        emit_p_bra_predicated p 0
    emit_cmp_reg loop_reg end_reg
        arm64_emit32 0x5400000A

parse_for_end :
    loop_reg pop_branch_reg
    end_reg pop_branch_end
    step_reg pop_branch_step

    emit_p_add loop_reg loop_reg step_reg

    if== emit_target 0
        emit_p_bra 0
    arm64_emit32 0x14000000

parse_each :
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    tid_reg alloc_rreg
    ctaid_reg alloc_rreg
    gidx_reg alloc_rreg

    if== emit_target 0
        emit_p_s2r tid_reg 0x21
        emit_p_s2r ctaid_reg 0x25

        bdim_reg alloc_rreg
        emit_imad gidx_reg ctaid_reg bdim_reg tid_reg

    sym_add var_ptr var_len 4 gidx_reg

parse_stride :
    var_ptr tok_text_ptr
    var_len peek_length
    consume

    dim_reg parse_expr

    loop_reg alloc_rreg
    tid_reg alloc_rreg
    ctaid_reg alloc_rreg
    emit_p_s2r tid_reg 0x21
    emit_p_s2r ctaid_reg 0x25
    gidx alloc_rreg
    bdim alloc_rreg
    emit_p_mov_reg loop_reg gidx

    sym_add var_ptr var_len 9 loop_reg

    push_branch loop_reg dim_reg gidx

    p alloc_preg
    emit_p_isetp p loop_reg dim_reg
    emit_p_bra_predicated p 0

parse_conditional :
    ptr tok_text_ptr
    len peek_length
    consume

    \\ Decode comparison: 0=EQ 1=GE 2=LT 3=NE 4=GT 5=LE
    cmp_type 0

    if== len 3
        c2 → 8 ptr 2
        if== c2 60
            cmp_type 2
        if== c2 62
            cmp_type 4

    if>= len 4
        c2 → 8 ptr 2
        c3 → 8 ptr 3
        if== c2 61
            if== c3 61
                cmp_type 0
        if== c2 62
            if== c3 61
                cmp_type 1
        if== c2 60
            if== c3 61
                cmp_type 5
        if== c2 33
            if== c3 61
                cmp_type 3

    a parse_expr
    b parse_expr

    if== emit_target 0
        p alloc_preg
        if== cmp_type 0
            emit_p_isetp p a b
        if== cmp_type 1
            emit_p_isetp p a b
        if== cmp_type 2
            emit_p_isetp p a b
        if== cmp_type 3
            emit_p_isetp p a b
        if== cmp_type 4
            emit_p_isetp p a b
        if== cmp_type 5
            emit_p_isetp p a b
    emit_cmp_reg a b
        if== cmp_type 0
            arm64_emit32 0x54000001
        if== cmp_type 1
            arm64_emit32 0x5400000B
        if== cmp_type 2
            arm64_emit32 0x5400000A
        if== cmp_type 3
            arm64_emit32 0x54000000
        if== cmp_type 4
            arm64_emit32 0x5400000D
        if== cmp_type 5
            arm64_emit32 0x5400000C

    parse_body

parse_shared :
    name_ptr tok_text_ptr
    name_len peek_length
    consume

    size parse_int_token

    type_tok peek_type
    consume
    elem_size 4
    if== type_tok 43
        elem_size 2

    total size * elem_size

    sidx n_shared
    ← 32 shm_sizes sidx * 4 total

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
\\ Body parser
\\ ============================================================================

parse_body :
    skip_newlines
    t peek_type
    if== t 2
        new_indent peek_length
        old_indent body_indent
        body_indent new_indent
        consume
        parse_body_loop old_indent
        body_indent old_indent

parse_body_loop old_indent :
    t peek_type
    if== t 0
        0

    if== t 1
        consume
        t2 peek_type
        if== t2 2
            ind peek_length
            if< ind body_indent
                0
            consume
            parse_stmt
            parse_body_loop old_indent
            0

    parse_stmt
    parse_body_loop old_indent

\\ ============================================================================
\\ Composition parser
\\ ============================================================================

parse_composition :
    name_ptr tok_text_ptr
    name_len peek_length
    consume

    sym_reset
    regs_reset
    n_shared 0
    shmem_total 0
    branch_depth 0

    arg_count 0
    parse_comp_args arg_count

    expect 72

    cidx n_comps
    ← 32 comp_tok_starts cidx * 4 tok_pos
    ← 32 comp_arg_counts cidx * 4 arg_count
    clen name_len
    if>= clen 32
        clen 32
    for j 0 clen 1
        b → 8 name_ptr j
        ← 8 comp_names cidx * 32 + j b
    ← 32 comp_lens cidx * 4 name_len
    n_comps n_comps + 1

    parse_body

    if== emit_target 0
        emit_p_exit_gpu
    emit_p_ret_host

parse_comp_args count :
    t peek_type
    if== t 72
        count

    if== t 5
        ptr tok_text_ptr
        len peek_length
        consume

        if== emit_target 0
            reg alloc_rdreg
        reg count

        sym_add ptr len 0 reg
        count count + 1

        parse_comp_args count

    count

\\ ============================================================================
\\ Top-level file parser
\\ ============================================================================

parse_file :
    t peek_type
    if== t 0
        0

    if== t 1
        consume
        parse_file

    if== t 2
        consume
        parse_file

    if== t 35
        consume
        emit_target 1
        parse_composition
        emit_target 0
        parse_file

    if== t 5
        emit_target 0
        parse_composition
        parse_file

    if== t 11
        consume
        emit_target 0
        parse_composition
        parse_file

    \\ Forth-style: VALUE constant NAME (at file scope)
    if== t 3
        val parse_int_token
        t2 peek_type
        if== t2 96
            consume
            name_ptr tok_text_ptr
            name_len peek_length
            consume
            rd alloc_rreg
            emit_p_mov_imm rd val
            sym_add name_ptr name_len 1 rd
            parse_file
        \\ Not a constant decl, put token back
        tok_pos tok_pos - 1

    \\ TOK_CONSTANT at file scope
    if== t 96
        consume
        parse_constant_decl
        parse_file

    consume
    parse_file

\\ ============================================================================
\\ Parser initialization
\\ ============================================================================

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
    n_labels 0
    n_kparams 0
    regs_reset

\\ ============================================================================
\\ Parser entry point
\\ ============================================================================

lithos_parse tokens_ptr total source :
    parser_init tokens_ptr total source
    parse_file

\\ ############################################################################
\\ ##                                                                        ##

\ Section 5 (safetensors reader) moved to compiler/safetensors.ls

\\ ##  SECTION 6 — ELF WRITER (lithos-elf)                                  ##
\\ ##                                                                        ##
\\ ############################################################################

\\ Note: The ELF writer uses the cubin_buf/cubin_pos state from Section 2
\\ (GPU backend). This is intentional — the GPU backend's cubin_* state
\\ holds the ELF output buffer, and the ELF writer populates it.

\\ ============================================================
\\ ELF CONSTANTS
\\ ============================================================

\\ TODO: bump for full megakernel
4194304 constant ELF_CUBIN_SIZE
64 constant EHDR_SIZE
64 constant SHDR_SIZE
24 constant SYM_SIZE

2 constant ET_EXEC
190 constant EM_CUDA
128 constant EV_CUDA

0 constant SHT_NULL
1 constant SHT_PROGBITS
2 constant SHT_SYMTAB
3 constant SHT_STRTAB
8 constant SHT_NOBITS
1879048192 constant SHT_LOPROC

2 constant SHF_ALLOC
4 constant SHF_EXECINSTR
64 constant SHF_INFO_LINK

94396762 constant ELF_FLAGS_CUDA

\\ ============================================================
\\ BYTE EMIT PRIMITIVES (ELF writer uses cubin_buf/cubin_pos)
\\ ============================================================

cb_emit byte :
    cubin_emit_byte byte

cw_emit val_u16 :
    cubin_emit_u16 val_u16

cd_emit val_u32 :
    cubin_emit_u32 val_u32

cq_emit val_u64 :
    cubin_emit_u64 val_u64

cpad target :
    loop_pad:
        if>= cubin_pos target
            return
        cb_emit 0
        goto loop_pad

calign n :
    mask n - 1
    target (cubin_pos + mask) & (mask ^ 4294967295)
    cpad target

\\ ============================================================
\\ PATCH HELPERS
\\ ============================================================

elf_put_u16 val off :
    ← 8 cubin_buf + off (val & 255)
    ← 8 cubin_buf + (off + 1) ((val >> 8) & 255)

elf_put_u32 val off :
    ← 8 cubin_buf + off (val & 255)
    ← 8 cubin_buf + (off + 1) ((val >> 8) & 255)
    ← 8 cubin_buf + (off + 2) ((val >> 16) & 255)
    ← 8 cubin_buf + (off + 3) ((val >> 24) & 255)

elf_put_u64 val off :
    elf_put_u32 (val & 4294967295) off
    elf_put_u32 (val >> 32) (off + 4)

\\ ============================================================
\\ STRING EMIT
\\ ============================================================

elf_emit_str src len :
    for i 0 len 1
        byte → 8 (src + i)
        cb_emit byte
    endfor

\\ ============================================================
\\ elf_init
\\ ============================================================

elf_init :
    cubin_pos 0

\\ ============================================================
\\ elf_write_header
\\ ============================================================

elf_write_header :
    cb_emit 127
    cb_emit 69
    cb_emit 76
    cb_emit 70
    cb_emit 2
    cb_emit 1
    cb_emit 1
    cb_emit 51
    cb_emit 7
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cw_emit 2
    cw_emit 190
    cd_emit 128
    cq_emit 0
    cq_emit 0
    cq_emit 0
    cd_emit ELF_FLAGS_CUDA
    cw_emit 64
    cw_emit 56
    cw_emit 0
    cw_emit 64
    cw_emit 0
    cw_emit 0

\\ ============================================================
\\ SECTION HEADER EMIT
\\ ============================================================

shdr64_a sh_name sh_type sh_flags sh_addr sh_offset :
    cd_emit sh_name
    cd_emit sh_type
    cq_emit sh_flags
    cq_emit sh_addr
    cq_emit sh_offset

shdr64_b sh_size sh_link sh_info sh_addralign sh_entsize :
    cq_emit sh_size
    cd_emit sh_link
    cd_emit sh_info
    cq_emit sh_addralign
    cq_emit sh_entsize

\\ ============================================================
\\ SYMBOL TABLE ENTRY
\\ ============================================================

sym64_emit st_name st_info st_other st_shndx st_value st_size :
    cd_emit st_name
    cb_emit st_info
    cb_emit st_other
    cw_emit st_shndx
    cq_emit st_value
    cq_emit st_size

\\ ============================================================
\\ NV.INFO RECORD EMITTERS
\\ ============================================================

nvi_u32_emit val attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 4
    cd_emit val

nvi_sval_emit val sym_idx attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 8
    cd_emit sym_idx
    cd_emit val

\\ ============================================================
\\ elf_build — Build complete 9-section GPU ELF
\\ Split into sub-compositions to stay within the bootstrap's
\\ 20-register window. Section metadata passed via globals.
\\ ============================================================

\\ Globals for section offsets/sizes (shared between sub-compositions)
var elf_shstrtab_off 0
var elf_shstrtab_size 0
var elf_strtab_off 0
var elf_strtab_size 0
var elf_symtab_off 0
var elf_symtab_size 0
var elf_nvinfo_off 0
var elf_nvinfo_size 0
var elf_nvinfo_k_off 0
var elf_nvinfo_k_size 0
var elf_text_off 0
var elf_text_size 0
var elf_const0_off 0
var elf_const0_size 0
var elf_shdrs_off 0
var elf_sym_kernel_off 0
var elf_param_bytes 0

\\ Section name offsets within .shstrtab
var elf_SN_shstrtab 0
var elf_SN_strtab 0
var elf_SN_symtab 0
var elf_SN_nvinfo 0
var elf_SN_nvinfo_k 0
var elf_SN_text 0
var elf_SN_const0 0
var elf_SN_shared 0

\\ ---- Sub-composition: emit .shstrtab section names ----
elf_emit_shstrtab kernel_name kernel_nlen :
    elf_shstrtab_off cubin_pos
    cb_emit 0

    elf_SN_shstrtab cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 115
    cb_emit 104
    cb_emit 115
    cb_emit 116
    cb_emit 114
    cb_emit 116
    cb_emit 97
    cb_emit 98
    cb_emit 0

    elf_SN_strtab cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 115
    cb_emit 116
    cb_emit 114
    cb_emit 116
    cb_emit 97
    cb_emit 98
    cb_emit 0

    elf_SN_symtab cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 115
    cb_emit 121
    cb_emit 109
    cb_emit 116
    cb_emit 97
    cb_emit 98
    cb_emit 0

    elf_SN_nvinfo cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 110
    cb_emit 118
    cb_emit 46
    cb_emit 105
    cb_emit 110
    cb_emit 102
    cb_emit 111
    cb_emit 0

    elf_SN_nvinfo_k cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 110
    cb_emit 118
    cb_emit 46
    cb_emit 105
    cb_emit 110
    cb_emit 102
    cb_emit 111
    cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

    elf_SN_text cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 116
    cb_emit 101
    cb_emit 120
    cb_emit 116
    cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

    elf_SN_const0 cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 110
    cb_emit 118
    cb_emit 46
    cb_emit 99
    cb_emit 111
    cb_emit 110
    cb_emit 115
    cb_emit 116
    cb_emit 97
    cb_emit 110
    cb_emit 116
    cb_emit 48
    cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

    elf_SN_shared cubin_pos - elf_shstrtab_off
    cb_emit 46
    cb_emit 110
    cb_emit 118
    cb_emit 46
    cb_emit 115
    cb_emit 104
    cb_emit 97
    cb_emit 114
    cb_emit 101
    cb_emit 100
    cb_emit 46
    cb_emit 114
    cb_emit 101
    cb_emit 115
    cb_emit 101
    cb_emit 114
    cb_emit 118
    cb_emit 101
    cb_emit 100
    cb_emit 46
    cb_emit 48
    cb_emit 0

    calign 4
    elf_shstrtab_size cubin_pos - elf_shstrtab_off

\\ ---- Sub-composition: emit .strtab + .symtab ----
elf_emit_strtab_symtab kernel_name kernel_nlen :
    elf_strtab_off cubin_pos
    cb_emit 0
    cb_emit 0
    strsym_kernel cubin_pos - elf_strtab_off
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0
    elf_strtab_size cubin_pos - elf_strtab_off

    calign 8
    elf_symtab_off cubin_pos
    sym64_emit 0 0 0 0 0 0
    sym64_emit 0 3 0 5 0 0
    sym64_emit 0 3 0 8 0 0
    elf_sym_kernel_off cubin_pos
    sym64_emit strsym_kernel 18 16 5 0 0
    elf_symtab_size cubin_pos - elf_symtab_off

\\ ---- Sub-composition: emit .nv.info (global attrs) ----
elf_emit_nvinfo reg_count :
    calign 4
    elf_nvinfo_off cubin_pos
    reg_val reg_count + 1
    if< reg_val 8
        reg_val 8
    nvi_sval_emit reg_val 3 EIATTR_REGCOUNT NVI_FMT_U32
    nvi_sval_emit 0 3 EIATTR_FRAME_SIZE NVI_FMT_U32
    nvi_sval_emit 0 3 EIATTR_MIN_STACK_SIZE NVI_FMT_U32
    elf_nvinfo_size cubin_pos - elf_nvinfo_off

\\ ---- Sub-composition: emit .nv.info.<kernel> (per-kernel attrs) ----
elf_emit_nvinfo_k n_kparams smem_size code_size :
    calign 4
    elf_nvinfo_k_off cubin_pos
    nvi_u32_emit 128 EIATTR_CUDA_API_VERSION NVI_FMT_U32

    for pi 0 n_kparams 1
        ordinal n_kparams - 1 - pi
        cb_emit 4
        cb_emit EIATTR_KPARAM_INFO
        cw_emit 12
        cd_emit 0
        offset_ord (ordinal * 8) << 16
        offset_ord offset_ord | ordinal
        cd_emit offset_ord
        cd_emit 2171904
    endfor

    cb_emit 3
    cb_emit EIATTR_SPARSE_MMA_MASK
    cb_emit 0
    cb_emit 0

    cb_emit 3
    cb_emit EIATTR_MAXREG_COUNT
    cb_emit 255
    cb_emit 0

    if== gpu_cooperative 1
        cb_emit 4
        cb_emit EIATTR_COOP_GROUP_MASK_REGIDS
        cw_emit 16
        cd_emit 4294967295
        cd_emit 4294967295
        cd_emit 4294967295
        cd_emit 4294967295

        cb_emit 4
        cb_emit EIATTR_COOP_GROUP_INSTR_OFFSETS
        cw_emit (gridsync_count * 4)
        for gi 0 gridsync_count 1
            gs_off → 32 (gridsync_offsets + gi * 4)
            cd_emit gs_off
        endfor

    cb_emit 4
    cb_emit EIATTR_EXIT_INSTR_OFFSETS
    if> exit_count 0
        cw_emit (exit_count * 4)
        for ei 0 exit_count 1
            ex_off → 32 (exit_offsets + ei * 4)
            cd_emit ex_off
        endfor
    if== exit_count 0
        cw_emit 4
        cd_emit code_size - 16

    elf_param_bytes n_kparams * 8
    cb_emit 3
    cb_emit EIATTR_CBANK_PARAM_SIZE
    cw_emit elf_param_bytes

    cb_emit 4
    cb_emit EIATTR_PARAM_CBANK
    cw_emit 8
    cd_emit 2
    cbank_val (elf_param_bytes << 16) | 528
    cd_emit cbank_val

    cb_emit 4
    cb_emit EIATTR_SW_WAR
    cw_emit 4
    cd_emit 8

    if> smem_size 0
        cb_emit 4
        cb_emit 51
        cw_emit 4
        cd_emit smem_size

    elf_nvinfo_k_size cubin_pos - elf_nvinfo_k_off

\\ ---- Sub-composition: emit .text + .nv.constant0 ----
elf_emit_text_const0 code_buf code_size :
    calign 128
    elf_text_off cubin_pos

    if== code_size 0
        for zb 0 48 1
            cb_emit 0
        endfor
    else
        for ci 0 code_size 1
            byte → 8 (code_buf + ci)
            cb_emit byte
        endfor

    elf_text_size cubin_pos - elf_text_off
    elf_put_u64 elf_text_size (elf_sym_kernel_off + 16)

    calign 4
    elf_const0_off cubin_pos
    const0_total 528 + elf_param_bytes
    for c0i 0 const0_total 1
        cb_emit 0
    endfor
    elf_const0_size cubin_pos - elf_const0_off

\\ ---- Sub-composition: emit section headers + patch ELF header ----
elf_emit_shdrs smem_size :
    calign 64
    elf_shdrs_off cubin_pos

    shdr64_a 0 0 0 0 0
    shdr64_b 0 0 0 0 0
    shdr64_a elf_SN_shstrtab SHT_STRTAB 0 0 elf_shstrtab_off
    shdr64_b elf_shstrtab_size 0 0 1 0
    shdr64_a elf_SN_strtab SHT_STRTAB 0 0 elf_strtab_off
    shdr64_b elf_strtab_size 0 0 1 0
    shdr64_a elf_SN_symtab SHT_SYMTAB 0 0 elf_symtab_off
    shdr64_b elf_symtab_size 2 3 8 24
    shdr64_a elf_SN_nvinfo SHT_LOPROC 0 0 elf_nvinfo_off
    shdr64_b elf_nvinfo_size 3 0 4 0
    shdr64_a elf_SN_text SHT_PROGBITS 6 0 elf_text_off
    shdr64_b elf_text_size 3 3 128 0
    shdr64_a elf_SN_nvinfo_k SHT_LOPROC SHF_INFO_LINK 0 elf_nvinfo_k_off
    shdr64_b elf_nvinfo_k_size 3 5 4 0
    shdr64_a elf_SN_shared SHT_NOBITS 3 0 elf_const0_off
    shdr64_b smem_size 0 0 16 0
    shdr64_a elf_SN_const0 SHT_PROGBITS 66 0 elf_const0_off
    shdr64_b elf_const0_size 0 5 4 0

    elf_put_u64 0 32
    elf_put_u64 elf_shdrs_off 40
    elf_put_u16 0 54
    elf_put_u16 0 56
    elf_put_u16 9 60
    elf_put_u16 1 62

\\ ---- Orchestrator: calls sub-compositions in order ----
elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size :
    elf_init
    elf_write_header
    elf_emit_shstrtab kernel_name kernel_nlen
    elf_emit_strtab_symtab kernel_name kernel_nlen
    elf_emit_nvinfo reg_count
    elf_emit_nvinfo_k n_kparams smem_size code_size
    elf_emit_text_const0 code_buf code_size
    elf_emit_shdrs smem_size

\\ ============================================================
\\ elf_save — Write cubin_buf to file
\\ ============================================================

elf_save path :
    trap fd SYS_OPENAT -100 path 577 420

    written 0
    remaining cubin_pos
    loop_write:
        if<= remaining 0
            goto done_write
        write_ptr cubin_buf + written
        trap n SYS_WRITE fd write_ptr remaining
        if<= n 0
            goto done_write
        written written + n
        remaining remaining - n
        goto loop_write

    done_write:
    trap ret SYS_CLOSE fd

\\ ============================================================
\\ elf_write_cubin — Convenience: build ELF and write to file
\\ ============================================================

elf_write_cubin kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size out_path :
    elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size
    elf_save out_path

\\ ============================================================
\\ elf_write_simple — Non-cooperative kernel (clears cooperative/gridsync state)
\\ ============================================================

elf_write_simple kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size out_path :
    gpu_cooperative 0
    gridsync_count 0
    exit_count 0
    elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size
    elf_save out_path


\\ ============================================================================
\\ ARM64 ELF WRAPPER (elf_build_arm64)
\\ ============================================================================
\\ Builds an ELF64 executable wrapping ARM64 code in arm64_buf.
\\ e_machine = 0xB7 (EM_AARCH64), entry at text start.
\\ Uses cubin_buf/cubin_pos as output buffer (same as GPU ELF writer).

elf_build_arm64 code_buf code_size :
    cubin_pos 0

    \\ ELF header (64 bytes)
    \\ e_ident: 7f 45 4c 46 02 01 01 00 (ELF64 LE SYSV)
    cb_emit 127
    cb_emit 69
    cb_emit 76
    cb_emit 70
    cb_emit 2
    cb_emit 1
    cb_emit 1
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    \\ e_type = ET_EXEC (2)
    cw_emit 2
    \\ e_machine = EM_AARCH64 (0xB7 = 183)
    cw_emit 183
    \\ e_version = 1
    cd_emit 1
    \\ e_entry = 0x400078 (text start after ELF hdr + phdr)
    cq_emit 0x400078
    \\ e_phoff = 64 (right after ELF header)
    cq_emit 64
    \\ e_shoff = 0 (no section headers for minimal exec)
    cq_emit 0
    \\ e_flags = 0
    cd_emit 0
    \\ e_ehsize = 64
    cw_emit 64
    \\ e_phentsize = 56
    cw_emit 56
    \\ e_phnum = 1
    cw_emit 1
    \\ e_shentsize = 64
    cw_emit 64
    \\ e_shnum = 0
    cw_emit 0
    \\ e_shstrndx = 0
    cw_emit 0

    \\ Program header (56 bytes) — PT_LOAD
    \\ p_type = PT_LOAD (1)
    cd_emit 1
    \\ p_flags = PF_R | PF_X (5)
    cd_emit 5
    \\ p_offset = 0
    cq_emit 0
    \\ p_vaddr = 0x400000
    cq_emit 0x400000
    \\ p_paddr = 0x400000
    cq_emit 0x400000
    \\ p_filesz = 120 + code_size (ehdr + phdr + code)
    total_file 120 + code_size
    cq_emit total_file
    \\ p_memsz = total_file
    cq_emit total_file
    \\ p_align = 0x1000
    cq_emit 0x1000

    \\ Code section starts at offset 120 (0x78)
    for ci 0 code_size 1
        byte → 8 (code_buf + ci)
        cb_emit byte
    endfor

\\ ############################################################################
\\ ##                                                                        ##
\\ ##  SECTION 7 — MAIN ENTRY POINT                                         ##
\\ ##                                                                        ##
\\ ############################################################################

\\ ============================================================
\\ Host trap helpers for main
\\ ============================================================

\\ mmap_file — open, get size via lseek, mmap, close fd
\\ Returns base pointer and file size.
host mmap_file path :
    \\ openat(AT_FDCWD, path, O_RDONLY, 0)
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 0
    ↓ $3 0
    trap
    fd ↑ $0

    \\ lseek(fd, 0, SEEK_END)
    ↓ $8 62
    ↓ $0 fd
    ↓ $1 0
    ↓ $2 2
    trap
    file_size ↑ $0

    \\ mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    ↓ $8 222
    ↓ $0 0
    ↓ $1 file_size
    ↓ $2 1
    ↓ $3 2
    ↓ $4 fd
    ↓ $5 0
    trap
    base ↑ $0

    \\ close(fd)
    ↓ $8 57
    ↓ $0 fd
    trap

\\ write_file — open (create/trunc), write buffer, close
host write_file path buf buf_len :
    \\ openat(AT_FDCWD, path, O_WRONLY|O_CREAT|O_TRUNC, 0755)
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 577
    ↓ $3 493
    trap
    fd ↑ $0

    \\ write(fd, buf, buf_len)
    written 0
    remaining buf_len
    write_loop:
        if<= remaining 0
            goto write_done
        ↓ $8 64
        ↓ $0 fd
        ↓ $1 buf + written
        ↓ $2 remaining
        trap
        n ↑ $0
        if<= n 0
            goto write_done
        written written + n
        remaining remaining - n
        goto write_loop

    write_done:
    \\ close(fd)
    ↓ $8 57
    ↓ $0 fd
    trap

\\ ============================================================
\\ lithos_main — Compiler entry point
\\ ============================================================
\\
\\ Usage: lithos <source.ls> <output> [weights.safetensors]
\\
\\ Arguments (via ARM64 calling convention):
\\   X0 = argc (argument count)
\\   X1 = argv (pointer to argument string pointers)
\\
\\ Pipeline:
\\   1. Parse command line arguments
\\   2. mmap the source file
\\   3. Call lexer to tokenize
\\   4. Call parser to parse and emit machine code
\\   5. Call ELF writer to wrap output
\\   6. Write output file
\\   7. Exit

host lithos_main argc argv :
    \\ Validate argument count (need at least source and output)
    if< argc 3
        \\ Too few arguments — exit with error
        ↓ $8 93
        ↓ $0 1
        trap

    \\ argv[0] = program name (skip)
    \\ argv[1] = source file path
    \\ argv[2] = output file path
    \\ argv[3] = optional safetensors path

    src_path → 64 argv 8          \\ argv[1]
    out_path → 64 argv 16         \\ argv[2]

    \\ Step 1: mmap the source file
    src_base src_size mmap_file src_path

    \\ Step 2: Lex the source
    lithos_lex src_base src_size

    \\ Step 3: Parse and emit machine code
    lithos_parse tokens token_count src_base

    \\ Step 4: Safetensors weights (loaded by safetensors.ls, linked separately)

    \\ Step 5: Build ELF output
    \\ For GPU kernels: wrap gpu_buf in cubin ELF
    \\ For host code: wrap arm64_buf in ARM64 ELF
    \\ Choose based on what was emitted
    if> gpu_pos 0
        \\ GPU output — build cubin
        elf_build li_name_buf li_name_len gpu_buf gpu_pos gpu_n_kparams max_reg gpu_shmem_size gpu_cooperative gridsync_offsets gridsync_count
        elf_save out_path
    if> arm64_pos 0
        \\ Host output — wrap in ARM64 ELF
        elf_build_arm64 arm64_buf arm64_pos
        elf_save out_path

    \\ Step 6: Exit successfully
    ↓ $8 93
    ↓ $0 0
    trap

