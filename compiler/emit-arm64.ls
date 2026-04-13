\\ emit-arm64.li — Lithos ARM64 machine code emitter
\\ Emits raw ARM64 instructions (32-bit fixed-width) into a code buffer.
\\ This is the host-side counterpart of gpu/emit.fs (which emits sm90 GPU code).
\\
\\ ARM64 instructions are always 4 bytes, little-endian.
\\ Encoding reference: Arm Architecture Reference Manual for A-profile (DDI 0487)
\\
\\ NOTE: The .li language currently targets GPU kernels. This file is ASPIRATIONAL —
\\ it shows what the .li syntax looks like when extended for host-side ARM64 code.
\\ Comments marked [EXT] indicate where new language features would be needed.
\\
\\ Usage: the self-hosting Lithos compiler calls these functions to emit native
\\ ARM64 code. Each fn encodes one ARM64 instruction and appends it to the buffer.

\\ ============================================================
\\ CODE BUFFER
\\ ============================================================
\\ Same concept as sass-buf / sass-pos in gpu/emit.fs, but for ARM64.
\\ 1MB is enough for the entire self-hosting compiler binary.

buf arm64_buf 1048576
var arm64_pos 0

\\ [EXT] These fns operate on host memory, not GPU shared/global memory.
\\ The .li compiler would need a "host" execution context for this to work.

arm64_emit32 val :
    \\ Write a 32-bit little-endian word at current position and advance by 4.
    arm64_buf [ arm64_pos ] = val
    arm64_pos = arm64_pos + 4

arm64_reset :
    arm64_pos = 0

\\ ============================================================
\\ REGISTER CONSTANTS
\\ ============================================================
\\ ARM64 has 31 general-purpose 64-bit registers (X0-X30) plus SP/XZR.
\\ W0-W30 are the 32-bit views of the same registers.
\\ Register encoding is 5 bits (0-31) in all instruction fields.

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
const COND_EQ  0     \\ equal (Z=1)
const COND_NE  1     \\ not equal (Z=0)
const COND_CS  2     \\ carry set / unsigned >= (C=1)
const COND_CC  3     \\ carry clear / unsigned < (C=0)
const COND_MI  4     \\ minus / negative (N=1)
const COND_PL  5     \\ plus / positive or zero (N=0)
const COND_VS  6     \\ overflow (V=1)
const COND_VC  7     \\ no overflow (V=0)
const COND_HI  8     \\ unsigned > (C=1 && Z=0)
const COND_LS  9     \\ unsigned <= (C=0 || Z=1)
const COND_GE  10    \\ signed >=  (N=V)
const COND_LT  11    \\ signed <   (N!=V)
const COND_GT  12    \\ signed >   (Z=0 && N=V)
const COND_LE  13    \\ signed <=  (Z=1 || N!=V)
const COND_AL  14    \\ always

\\ ============================================================
\\ LINUX AARCH64 SYSCALL NUMBERS
\\ ============================================================
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

\\ ============================================================
\\ DATA PROCESSING — REGISTER (Arithmetic)
\\ ============================================================
\\ ARM64 encoding: sf=1 (64-bit) in bit 31 for X registers.
\\ Data processing (register): bits [28:25] = 0101 for logical,
\\ 1011 for add/sub shifted register.

\\ ADD Xd, Xn, Xm — 64-bit register add
\\ Encoding: sf=1 op=0 S=0 | 01011 shift=00 0 | Rm(5) imm6=000000 | Rn(5) Rd(5)
\\ = 0x8B000000 | Rm<<16 | Rn<<5 | Rd
emit_add_reg rd rn rm :
    val = 0x8B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ADD Xd, Xn, Xm, LSL #amount — shifted register add
\\ Encoding: 0x8B000000 | Rm<<16 | amount<<10 | Rn<<5 | Rd
emit_add_reg_lsl rd rn rm amount :
    val = 0x8B000000 | (rm << 16) | (amount << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Xd, Xn, Xm — 64-bit register subtract
\\ Encoding: sf=1 op=1 S=0 | 01011 shift=00 0 | Rm(5) imm6=000000 | Rn(5) Rd(5)
\\ = 0xCB000000 | Rm<<16 | Rn<<5 | Rd
emit_sub_reg rd rn rm :
    val = 0xCB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ADDS Xd, Xn, Xm — add and set flags (for CMP alias when Rd=XZR)
\\ Encoding: 0xAB000000 | Rm<<16 | Rn<<5 | Rd
emit_adds_reg rd rn rm :
    val = 0xAB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ SUBS Xd, Xn, Xm — subtract and set flags
\\ Encoding: 0xEB000000 | Rm<<16 | Rn<<5 | Rd
emit_subs_reg rd rn rm :
    val = 0xEB000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ CMP Xn, Xm — alias for SUBS XZR, Xn, Xm
emit_cmp_reg rn rm :
    emit_subs_reg XZR rn rm

\\ NEG Xd, Xm — alias for SUB Xd, XZR, Xm
emit_neg rd rm :
    emit_sub_reg rd XZR rm

\\ MUL Xd, Xn, Xm — 64-bit multiply (alias for MADD Xd, Xn, Xm, XZR)
\\ MADD encoding: 0x9B000000 | Rm<<16 | Ra<<10 | Rn<<5 | Rd
\\ MUL = MADD with Ra=XZR(31): 0x9B007C00 | Rm<<16 | Rn<<5 | Rd
emit_mul rd rn rm :
    val = 0x9B007C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ MADD Xd, Xn, Xm, Xa — multiply-add: Xd = Xa + Xn*Xm
\\ Encoding: 0x9B000000 | Rm<<16 | Ra<<10 | Rn<<5 | Rd
emit_madd rd rn rm ra :
    val = 0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ MSUB Xd, Xn, Xm, Xa — multiply-subtract: Xd = Xa - Xn*Xm
\\ Encoding: 0x9B008000 | Rm<<16 | Ra<<10 | Rn<<5 | Rd
emit_msub rd rn rm ra :
    val = 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SDIV Xd, Xn, Xm — signed divide
\\ Encoding: 0x9AC00C00 | Rm<<16 | Rn<<5 | Rd
emit_sdiv rd rn rm :
    val = 0x9AC00C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ UDIV Xd, Xn, Xm — unsigned divide
\\ Encoding: 0x9AC00800 | Rm<<16 | Rn<<5 | Rd
emit_udiv rd rn rm :
    val = 0x9AC00800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — IMMEDIATE (Arithmetic)
\\ ============================================================

\\ ADD Xd, Xn, #imm12 — add immediate (64-bit)
\\ Encoding: sf=1 op=0 S=0 | 100010 sh=0 | imm12(12) | Rn(5) | Rd(5)
\\ = 0x91000000 | imm12<<10 | Rn<<5 | Rd
emit_add_imm rd rn imm :
    val = 0x91000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ADD Xd, Xn, #imm12, LSL #12 — add immediate shifted
\\ Encoding: 0x91400000 | imm12<<10 | Rn<<5 | Rd
emit_add_imm_lsl12 rd rn imm :
    val = 0x91400000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Xd, Xn, #imm12 — subtract immediate (64-bit)
\\ Encoding: 0xD1000000 | imm12<<10 | Rn<<5 | Rd
emit_sub_imm rd rn imm :
    val = 0xD1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ADDS Xd, Xn, #imm12 — add immediate and set flags
\\ Encoding: 0xB1000000 | imm12<<10 | Rn<<5 | Rd
emit_adds_imm rd rn imm :
    val = 0xB1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUBS Xd, Xn, #imm12 — subtract immediate and set flags
\\ Encoding: 0xF1000000 | imm12<<10 | Rn<<5 | Rd
emit_subs_imm rd rn imm :
    val = 0xF1000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ CMP Xn, #imm12 — alias for SUBS XZR, Xn, #imm12
emit_cmp_imm rn imm :
    emit_subs_imm XZR rn imm

\\ CMN Xn, #imm12 — alias for ADDS XZR, Xn, #imm12
emit_cmn_imm rn imm :
    emit_adds_imm XZR rn imm

\\ MOV Xd, Xn — alias for ORR Xd, XZR, Xn (register move)
emit_mov rd rn :
    val = 0xAA0003E0 | (rn << 16) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — LOGICAL (Register)
\\ ============================================================
\\ Logical shifted register:
\\ sf=1 opc(2) 01010 shift(2) N Rm(5) imm6(6) Rn(5) Rd(5)

\\ AND Xd, Xn, Xm
\\ opc=00 N=0: 0x8A000000 | Rm<<16 | Rn<<5 | Rd
emit_and_reg rd rn rm :
    val = 0x8A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ORR Xd, Xn, Xm
\\ opc=01 N=0: 0xAA000000 | Rm<<16 | Rn<<5 | Rd
emit_orr_reg rd rn rm :
    val = 0xAA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ EOR Xd, Xn, Xm (exclusive OR)
\\ opc=10 N=0: 0xCA000000 | Rm<<16 | Rn<<5 | Rd
emit_eor_reg rd rn rm :
    val = 0xCA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ANDS Xd, Xn, Xm (AND and set flags; TST when Rd=XZR)
\\ opc=11 N=0: 0xEA000000 | Rm<<16 | Rn<<5 | Rd
emit_ands_reg rd rn rm :
    val = 0xEA000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ TST Xn, Xm — alias for ANDS XZR, Xn, Xm
emit_tst_reg rn rm :
    emit_ands_reg XZR rn rm

\\ MVN Xd, Xm — alias for ORN Xd, XZR, Xm
\\ ORN: 0xAA200000 | Rm<<16 | Rn<<5 | Rd, with Rn=XZR(31)
emit_mvn rd rm :
    val = 0xAA2003E0 | (rm << 16) | rd
    arm64_emit32 val

\\ ============================================================
\\ DATA PROCESSING — LOGICAL (Immediate)
\\ ============================================================
\\ Logical immediate uses a bitmask encoding (N:immr:imms).
\\ These are complex to encode in general. We provide helpers for
\\ common masks and an escape hatch for the raw encoding.

\\ AND Xd, Xn, #bitmask_imm — raw encoding
\\ 0x92000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
emit_and_imm rd rn n immr imms :
    val = 0x92000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ORR Xd, Xn, #bitmask_imm — raw encoding
\\ 0xB2000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
emit_orr_imm rd rn n immr imms :
    val = 0xB2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ EOR Xd, Xn, #bitmask_imm — raw encoding
\\ 0xD2000000 | N<<22 | immr<<16 | imms<<10 | Rn<<5 | Rd
emit_eor_imm rd rn n immr imms :
    val = 0xD2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ SHIFT / BITFIELD OPERATIONS
\\ ============================================================

\\ LSL Xd, Xn, Xm — logical shift left (variable)
\\ Alias for LSLV: 0x9AC02000 | Rm<<16 | Rn<<5 | Rd
emit_lsl_reg rd rn rm :
    val = 0x9AC02000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ LSR Xd, Xn, Xm — logical shift right (variable)
\\ Alias for LSRV: 0x9AC02400 | Rm<<16 | Rn<<5 | Rd
emit_lsr_reg rd rn rm :
    val = 0x9AC02400 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ASR Xd, Xn, Xm — arithmetic shift right (variable)
\\ Alias for ASRV: 0x9AC02800 | Rm<<16 | Rn<<5 | Rd
emit_asr_reg rd rn rm :
    val = 0x9AC02800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ LSL Xd, Xn, #shift — immediate left shift
\\ Alias for UBFM Xd, Xn, #(64-shift), #(63-shift)
\\ UBFM: 0xD3400000 | immr<<16 | imms<<10 | Rn<<5 | Rd
\\ For LSL #shift: immr = (64 - shift) mod 64, imms = 63 - shift
emit_lsl_imm rd rn shift :
    immr = (64 - shift) & 63
    imms = 63 - shift
    val = 0xD3400000 | (immr << 16) | (imms << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LSR Xd, Xn, #shift — immediate right shift
\\ Alias for UBFM Xd, Xn, #shift, #63
\\ UBFM: 0xD3400000 | immr<<16 | imms<<10 | Rn<<5 | Rd
emit_lsr_imm rd rn shift :
    val = 0xD340FC00 | (shift << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ASR Xd, Xn, #shift — arithmetic shift right immediate
\\ Alias for SBFM Xd, Xn, #shift, #63
\\ SBFM: 0x93400000 | immr<<16 | imms<<10 | Rn<<5 | Rd
emit_asr_imm rd rn shift :
    val = 0x9340FC00 | (shift << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ CONDITIONAL SELECT
\\ ============================================================

\\ CSEL Xd, Xn, Xm, cond — conditional select
\\ Encoding: 0x9A800000 | Rm<<16 | cond<<12 | Rn<<5 | Rd
emit_csel rd rn rm cond :
    val = 0x9A800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CSINC Xd, Xn, Xm, cond — conditional select increment
\\ Encoding: 0x9A800400 | Rm<<16 | cond<<12 | Rn<<5 | Rd
emit_csinc rd rn rm cond :
    val = 0x9A800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CSET Xd, cond — alias for CSINC Xd, XZR, XZR, invert(cond)
\\ Sets Xd to 1 if cond is true, 0 otherwise.
\\ invert(cond) = cond XOR 1
emit_cset rd cond :
    inv = cond ^ 1
    emit_csinc rd XZR XZR inv

\\ CSNEG Xd, Xn, Xm, cond — conditional select negate
\\ Encoding: 0xDA800400 | Rm<<16 | cond<<12 | Rn<<5 | Rd
emit_csneg rd rn rm cond :
    val = 0xDA800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd
    arm64_emit32 val

\\ CNEG Xd, Xn, cond — alias for CSNEG Xd, Xn, Xn, invert(cond)
emit_cneg rd rn cond :
    inv = cond ^ 1
    emit_csneg rd rn rn inv

\\ ============================================================
\\ MOVE WIDE IMMEDIATE
\\ ============================================================
\\ Used to load immediate values into registers.
\\ hw field: 0=bits[15:0], 1=bits[31:16], 2=bits[47:32], 3=bits[63:48]

\\ MOVZ Xd, #imm16, LSL #(hw*16) — move wide with zero
\\ Encoding: sf=1 opc=10 100101 hw(2) imm16(16) Rd(5)
\\ = 0xD2800000 | hw<<21 | imm16<<5 | Rd
emit_movz rd imm16 hw :
    val = 0xD2800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ MOVK Xd, #imm16, LSL #(hw*16) — move wide with keep
\\ Encoding: 0xF2800000 | hw<<21 | imm16<<5 | Rd
emit_movk rd imm16 hw :
    val = 0xF2800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ MOVN Xd, #imm16, LSL #(hw*16) — move wide with NOT
\\ Encoding: 0x92800000 | hw<<21 | imm16<<5 | Rd
emit_movn rd imm16 hw :
    val = 0x92800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ Helper: load a full 64-bit immediate into Xd using MOVZ + MOVK sequence.
\\ Emits 1-4 instructions depending on which 16-bit chunks are non-zero.
emit_mov64 rd imm :
    chunk0 = imm & 0xFFFF
    chunk1 = (imm >> 16) & 0xFFFF
    chunk2 = (imm >> 32) & 0xFFFF
    chunk3 = (imm >> 48) & 0xFFFF
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
\\ ARM64 load/store use Rn as base register. Offset variants:
\\   - Unsigned immediate (scaled by access size)
\\   - Register offset (Rm, optionally shifted/extended)
\\   - Pre/post-indexed (writeback)

\\ --- 64-bit loads/stores (X registers) ---

\\ LDR Xt, [Xn, #imm] — load 64-bit, unsigned offset (scaled by 8)
\\ Encoding: 11 111 0 01 01 imm12(12) Rn(5) Rt(5)
\\ = 0xF9400000 | (imm/8)<<10 | Rn<<5 | Rt
emit_ldr rd rn imm :
    val = 0xF9400000 | ((imm / 8) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Xt, [Xn, #imm] — store 64-bit, unsigned offset (scaled by 8)
\\ Encoding: 0xF9000000 | (imm/8)<<10 | Rn<<5 | Rt
emit_str rt rn imm :
    val = 0xF9000000 | ((imm / 8) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDR Xt, [Xn, Xm] — load 64-bit, register offset
\\ Encoding: 0xF8606800 | Rm<<16 | Rn<<5 | Rt
emit_ldr_reg rd rn rm :
    val = 0xF8606800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Xt, [Xn, Xm] — store 64-bit, register offset
\\ Encoding: 0xF8206800 | Rm<<16 | Rn<<5 | Rt
emit_str_reg rt rn rm :
    val = 0xF8206800 | (rm << 16) | (rn << 5) | rt
    arm64_emit32 val

\\ --- 32-bit loads/stores (W registers) ---

\\ LDR Wt, [Xn, #imm] — load 32-bit, unsigned offset (scaled by 4)
\\ Encoding: 0xB9400000 | (imm/4)<<10 | Rn<<5 | Rt
emit_ldr_w rd rn imm :
    val = 0xB9400000 | ((imm / 4) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Wt, [Xn, #imm] — store 32-bit, unsigned offset (scaled by 4)
\\ Encoding: 0xB9000000 | (imm/4)<<10 | Rn<<5 | Rt
emit_str_w rt rn imm :
    val = 0xB9000000 | ((imm / 4) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ --- Byte loads/stores ---

\\ LDRB Wt, [Xn, #imm] — load byte, unsigned offset (no scaling)
\\ Encoding: 0x39400000 | imm12<<10 | Rn<<5 | Rt
emit_ldrb rd rn imm :
    val = 0x39400000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STRB Wt, [Xn, #imm] — store byte, unsigned offset
\\ Encoding: 0x39000000 | imm12<<10 | Rn<<5 | Rt
emit_strb rt rn imm :
    val = 0x39000000 | (imm << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ --- Halfword loads/stores ---

\\ LDRH Wt, [Xn, #imm] — load halfword, unsigned offset (scaled by 2)
\\ Encoding: 0x79400000 | (imm/2)<<10 | Rn<<5 | Rt
emit_ldrh rd rn imm :
    val = 0x79400000 | ((imm / 2) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STRH Wt, [Xn, #imm] — store halfword, unsigned offset (scaled by 2)
\\ Encoding: 0x79000000 | (imm/2)<<10 | Rn<<5 | Rt
emit_strh rt rn imm :
    val = 0x79000000 | ((imm / 2) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ --- Sign-extending loads ---

\\ LDRSB Xt, [Xn, #imm] — load signed byte into 64-bit register
\\ Encoding: 0x39800000 | imm12<<10 | Rn<<5 | Rt
emit_ldrsb rd rn imm :
    val = 0x39800000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LDRSH Xt, [Xn, #imm] — load signed halfword into 64-bit register
\\ Encoding: 0x79800000 | (imm/2)<<10 | Rn<<5 | Rt
emit_ldrsh rd rn imm :
    val = 0x79800000 | ((imm / 2) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ LDRSW Xt, [Xn, #imm] — load signed word into 64-bit register
\\ Encoding: 0xB9800000 | (imm/4)<<10 | Rn<<5 | Rt
emit_ldrsw rd rn imm :
    val = 0xB9800000 | ((imm / 4) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ --- Load/Store Pair (for prologue/epilogue) ---

\\ LDP Xt1, Xt2, [Xn, #imm] — load pair, signed offset (scaled by 8)
\\ Encoding: opc=10 1 0 1 0 0 1 01 imm7(7) Rt2(5) Rn(5) Rt1(5)
\\ = 0xA9400000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
\\ imm7 is signed, range -512 to +504 (in multiples of 8)
emit_ldp rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA9400000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP Xt1, Xt2, [Xn, #imm] — store pair, signed offset (scaled by 8)
\\ Encoding: 0xA9000000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
emit_stp rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA9000000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ LDP pre-indexed: LDP Xt1, Xt2, [Xn, #imm]!
\\ Encoding: 0xA9C00000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
emit_ldp_pre rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA9C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP pre-indexed: STP Xt1, Xt2, [Xn, #imm]!
\\ Encoding: 0xA9800000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
emit_stp_pre rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ LDP post-indexed: LDP Xt1, Xt2, [Xn], #imm
\\ Encoding: 0xA8C00000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
emit_ldp_post rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ STP post-indexed: STP Xt1, Xt2, [Xn], #imm
\\ Encoding: 0xA8800000 | (imm/8)<<15 | Rt2<<10 | Rn<<5 | Rt1
emit_stp_post rt1 rt2 rn imm :
    imm7 = (imm / 8) & 0x7F
    val = 0xA8800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1
    arm64_emit32 val

\\ --- Pre/Post-indexed single register ---

\\ LDR Xt, [Xn, #simm9]! — pre-indexed load
\\ Encoding: 0xF8400C00 | (simm9 & 0x1FF)<<12 | Rn<<5 | Rt
emit_ldr_pre rt rn simm :
    val = 0xF8400C00 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ STR Xt, [Xn, #simm9]! — pre-indexed store
\\ Encoding: 0xF8000C00 | (simm9 & 0x1FF)<<12 | Rn<<5 | Rt
emit_str_pre rt rn simm :
    val = 0xF8000C00 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ LDR Xt, [Xn], #simm9 — post-indexed load
\\ Encoding: 0xF8400400 | (simm9 & 0x1FF)<<12 | Rn<<5 | Rt
emit_ldr_post rt rn simm :
    val = 0xF8400400 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ STR Xt, [Xn], #simm9 — post-indexed store
\\ Encoding: 0xF8000400 | (simm9 & 0x1FF)<<12 | Rn<<5 | Rt
emit_str_post rt rn simm :
    val = 0xF8000400 | ((simm & 0x1FF) << 12) | (rn << 5) | rt
    arm64_emit32 val

\\ ============================================================
\\ BRANCHES
\\ ============================================================

\\ B offset — unconditional branch (PC-relative)
\\ Encoding: 000101 imm26(26)
\\ = 0x14000000 | (offset/4 & 0x3FFFFFF)
\\ offset is signed byte offset from this instruction.
emit_b offset :
    val = 0x14000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ BL offset — branch with link (call)
\\ Encoding: 100101 imm26(26)
\\ = 0x94000000 | (offset/4 & 0x3FFFFFF)
emit_bl offset :
    val = 0x94000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ B.cond offset — conditional branch
\\ Encoding: 01010100 imm19(19) 0 cond(4)
\\ = 0x54000000 | ((offset/4 & 0x7FFFF) << 5) | cond
emit_bcond cond offset :
    val = 0x54000000 | (((offset / 4) & 0x7FFFF) << 5) | cond
    arm64_emit32 val

\\ CBZ Xt, offset — compare and branch if zero (64-bit)
\\ Encoding: sf=1 011010 0 imm19(19) Rt(5)
\\ = 0xB4000000 | ((offset/4 & 0x7FFFF) << 5) | Rt
emit_cbz rt offset :
    val = 0xB4000000 | (((offset / 4) & 0x7FFFF) << 5) | rt
    arm64_emit32 val

\\ CBNZ Xt, offset — compare and branch if not zero (64-bit)
\\ Encoding: sf=1 011010 1 imm19(19) Rt(5)
\\ = 0xB5000000 | ((offset/4 & 0x7FFFF) << 5) | Rt
emit_cbnz rt offset :
    val = 0xB5000000 | (((offset / 4) & 0x7FFFF) << 5) | rt
    arm64_emit32 val

\\ TBZ Xt, #bit, offset — test bit and branch if zero
\\ Encoding: b5 011011 0 b40(5) imm14(14) Rt(5)
\\ = 0x36000000 | (bit>>5)<<31 | (bit&0x1F)<<19 | ((offset/4)&0x3FFF)<<5 | Rt
emit_tbz rt bit offset :
    b5 = (bit >> 5) & 1
    b40 = bit & 0x1F
    val = 0x36000000 | (b5 << 31) | (b40 << 19) | (((offset / 4) & 0x3FFF) << 5) | rt
    arm64_emit32 val

\\ TBNZ Xt, #bit, offset — test bit and branch if not zero
\\ Encoding: same as TBZ but bit 24 = 1
\\ = 0x37000000 | (bit>>5)<<31 | (bit&0x1F)<<19 | ((offset/4)&0x3FFF)<<5 | Rt
emit_tbnz rt bit offset :
    b5 = (bit >> 5) & 1
    b40 = bit & 0x1F
    val = 0x37000000 | (b5 << 31) | (b40 << 19) | (((offset / 4) & 0x3FFF) << 5) | rt
    arm64_emit32 val

\\ BR Xn — branch to register (indirect jump)
\\ Encoding: 0xD61F0000 | Rn<<5
emit_br rn :
    val = 0xD61F0000 | (rn << 5)
    arm64_emit32 val

\\ BLR Xn — branch with link to register (indirect call)
\\ Encoding: 0xD63F0000 | Rn<<5
emit_blr rn :
    val = 0xD63F0000 | (rn << 5)
    arm64_emit32 val

\\ RET {Xn} — return (defaults to X30/LR)
\\ Encoding: 0xD65F0000 | Rn<<5
emit_ret rn :
    val = 0xD65F0000 | (rn << 5)
    arm64_emit32 val

\\ RET — return via LR (most common form)
emit_ret_lr :
    emit_ret LR

\\ ============================================================
\\ PC-RELATIVE ADDRESS GENERATION
\\ ============================================================

\\ ADR Xd, offset — form PC-relative address (+-1MB range)
\\ Encoding: immlo(2) 10000 immhi(19) Rd(5)
\\ = ((offset&3)<<29) | 0x10000000 | ((offset>>2)&0x7FFFF)<<5 | Rd
emit_adr rd offset :
    immlo = offset & 3
    immhi = (offset >> 2) & 0x7FFFF
    val = (immlo << 29) | 0x10000000 | (immhi << 5) | rd
    arm64_emit32 val

\\ ADRP Xd, offset — form PC-relative page address (+-4GB range)
\\ Encoding: immlo(2) 10000 immhi(19) Rd(5)  with bit 31 = 1
\\ offset is in units of 4KB pages.
\\ = ((offset&3)<<29) | 0x90000000 | ((offset>>2)&0x7FFFF)<<5 | Rd
emit_adrp rd offset :
    immlo = offset & 3
    immhi = (offset >> 2) & 0x7FFFF
    val = (immlo << 29) | 0x90000000 | (immhi << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ SYSTEM INSTRUCTIONS
\\ ============================================================

\\ SVC #imm16 — supervisor call (syscall)
\\ Encoding: 0xD4000001 | imm16<<5
emit_svc imm16 :
    val = 0xD4000001 | (imm16 << 5)
    arm64_emit32 val

\\ SVC #0 — the standard Linux syscall instruction
emit_syscall :
    emit_svc 0

\\ NOP
\\ Encoding: 0xD503201F
emit_nop :
    arm64_emit32 0xD503201F

\\ DSB ISH — data synchronization barrier (inner shareable)
\\ Encoding: 0xD5033B9F
emit_dsb_ish :
    arm64_emit32 0xD5033B9F

\\ DSB SY — data synchronization barrier (full system)
\\ Encoding: 0xD5033F9F
emit_dsb_sy :
    arm64_emit32 0xD5033F9F

\\ DMB ISH — data memory barrier (inner shareable)
\\ Encoding: 0xD5033BBF
emit_dmb_ish :
    arm64_emit32 0xD5033BBF

\\ ISB — instruction synchronization barrier
\\ Encoding: 0xD5033FDF
emit_isb :
    arm64_emit32 0xD5033FDF

\\ BRK #imm16 — breakpoint
\\ Encoding: 0xD4200000 | imm16<<5
emit_brk imm16 :
    val = 0xD4200000 | (imm16 << 5)
    arm64_emit32 val

\\ ============================================================
\\ BRANCH PATCHING (forward reference resolution)
\\ ============================================================
\\ Same concept as BRA patching in gpu/emit.fs.
\\ arm64_mark returns the current position so we can patch later.

arm64_mark :
    pos = arm64_pos

\\ Patch a B.cond at mark_pos to branch to current position.
\\ Reads the existing instruction, ORs in the computed offset.
arm64_patch_bcond mark_pos :
    offset = arm64_pos - mark_pos
    imm19 = ((offset / 4) & 0x7FFFF) << 5
    \\ [EXT] Need host memory read: existing = arm64_buf[mark_pos]
    \\ existing = existing | imm19
    \\ arm64_buf[mark_pos] = existing
    arm64_buf [ mark_pos ] = arm64_buf [ mark_pos ] | imm19

\\ Patch a B (unconditional) at mark_pos to branch to current position.
arm64_patch_b mark_pos :
    offset = arm64_pos - mark_pos
    imm26 = (offset / 4) & 0x3FFFFFF
    arm64_buf [ mark_pos ] = arm64_buf [ mark_pos ] | imm26

\\ Patch a CBZ/CBNZ at mark_pos to branch to current position.
arm64_patch_cbz mark_pos :
    offset = arm64_pos - mark_pos
    imm19 = ((offset / 4) & 0x7FFFF) << 5
    arm64_buf [ mark_pos ] = arm64_buf [ mark_pos ] | imm19

\\ Emit a placeholder B.cond (offset=0) and return mark for later patching.
emit_bcond_fwd cond :
    mark = arm64_pos
    emit_bcond cond 0

\\ Emit a placeholder B (offset=0) and return mark for later patching.
emit_b_fwd :
    mark = arm64_pos
    emit_b 0

\\ Emit a placeholder CBZ and return mark for later patching.
emit_cbz_fwd rt :
    mark = arm64_pos
    emit_cbz rt 0

\\ Emit a placeholder CBNZ and return mark for later patching.
emit_cbnz_fwd rt :
    mark = arm64_pos
    emit_cbnz rt 0

\\ ============================================================
\\ SYSCALL WRAPPERS
\\ ============================================================
\\ Linux AArch64 syscall convention:
\\   X8  = syscall number
\\   X0-X5 = arguments (set by caller before calling these)
\\   X0  = return value
\\ These helpers just load X8 and emit SVC #0.

emit_syscall_exit :
    emit_movz X8 SYS_EXIT 0
    emit_syscall

emit_syscall_read :
    emit_movz X8 SYS_READ 0
    emit_syscall

emit_syscall_write :
    emit_movz X8 SYS_WRITE 0
    emit_syscall

emit_syscall_openat :
    emit_movz X8 SYS_OPENAT 0
    emit_syscall

emit_syscall_close :
    emit_movz X8 SYS_CLOSE 0
    emit_syscall

emit_syscall_mmap :
    emit_movz X8 SYS_MMAP 0
    emit_syscall

emit_syscall_munmap :
    emit_movz X8 SYS_MUNMAP 0
    emit_syscall

emit_syscall_mprotect :
    emit_movz X8 SYS_MPROTECT 0
    emit_syscall

emit_syscall_brk :
    emit_movz X8 SYS_BRK 0
    emit_syscall

\\ ============================================================
\\ FUNCTION PROLOGUE / EPILOGUE HELPERS
\\ ============================================================
\\ Standard ARM64 calling convention: callee saves X19-X28, FP, LR.
\\ Stack must be 16-byte aligned.

\\ Minimal prologue: save FP and LR, set up frame pointer
\\ STP X29, X30, [SP, #-16]!
\\ MOV X29, SP
emit_prologue :
    emit_stp_pre FP LR SP -16
    emit_mov FP SP

\\ Minimal epilogue: restore FP and LR, return
\\ LDP X29, X30, [SP], #16
\\ RET
emit_epilogue :
    emit_ldp_post FP LR SP 16
    emit_ret_lr

\\ Prologue with local stack space: save FP/LR and allocate N bytes
\\ STP X29, X30, [SP, #-framesize]!
\\ MOV X29, SP
emit_prologue_frame framesize :
    emit_stp_pre FP LR SP (0 - framesize)
    emit_mov FP SP

\\ Epilogue with frame deallocation
\\ LDP X29, X30, [SP], #framesize
\\ RET
emit_epilogue_frame framesize :
    emit_ldp_post FP LR SP framesize
    emit_ret_lr

\\ Save a callee-saved register pair (e.g., X19/X20) at stack offset
emit_save_pair r1 r2 offset :
    emit_stp r1 r2 SP offset

\\ Restore a callee-saved register pair from stack offset
emit_restore_pair r1 r2 offset :
    emit_ldp r1 r2 SP offset

\\ ============================================================
\\ 32-BIT (W-REGISTER) ARITHMETIC
\\ ============================================================
\\ The compiler works with 32-bit values for token IDs, array indices, etc.
\\ W-register variants clear the upper 32 bits of the X register.

\\ ADD Wd, Wn, Wm — 32-bit register add
\\ Encoding: sf=0 → 0x0B000000 | Rm<<16 | Rn<<5 | Rd
emit_add_w rd rn rm :
    val = 0x0B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Wd, Wn, Wm — 32-bit register subtract
\\ Encoding: 0x4B000000 | Rm<<16 | Rn<<5 | Rd
emit_sub_w rd rn rm :
    val = 0x4B000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ADD Wd, Wn, #imm12 — 32-bit add immediate
\\ Encoding: 0x11000000 | imm12<<10 | Rn<<5 | Rd
emit_add_imm_w rd rn imm :
    val = 0x11000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUB Wd, Wn, #imm12 — 32-bit subtract immediate
\\ Encoding: 0x51000000 | imm12<<10 | Rn<<5 | Rd
emit_sub_imm_w rd rn imm :
    val = 0x51000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ SUBS Wd, Wn, #imm12 — 32-bit subtract immediate, set flags
\\ Encoding: 0x71000000 | imm12<<10 | Rn<<5 | Rd
emit_subs_imm_w rd rn imm :
    val = 0x71000000 | (imm << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ CMP Wn, #imm12 — 32-bit compare immediate
emit_cmp_imm_w rn imm :
    emit_subs_imm_w XZR rn imm

\\ CMP Wn, Wm — 32-bit compare register
emit_cmp_w rn rm :
    val = 0x6B000000 | (rm << 16) | (rn << 5) | XZR
    arm64_emit32 val

\\ MUL Wd, Wn, Wm — 32-bit multiply
\\ MADD with Ra=WZR: 0x1B007C00 | Rm<<16 | Rn<<5 | Rd
emit_mul_w rd rn rm :
    val = 0x1B007C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ SDIV Wd, Wn, Wm — 32-bit signed divide
\\ Encoding: 0x1AC00C00 | Rm<<16 | Rn<<5 | Rd
emit_sdiv_w rd rn rm :
    val = 0x1AC00C00 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ UDIV Wd, Wn, Wm — 32-bit unsigned divide
\\ Encoding: 0x1AC00800 | Rm<<16 | Rn<<5 | Rd
emit_udiv_w rd rn rm :
    val = 0x1AC00800 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ AND Wd, Wn, Wm — 32-bit bitwise AND
\\ Encoding: 0x0A000000 | Rm<<16 | Rn<<5 | Rd
emit_and_w rd rn rm :
    val = 0x0A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ ORR Wd, Wn, Wm — 32-bit bitwise OR
\\ Encoding: 0x2A000000 | Rm<<16 | Rn<<5 | Rd
emit_orr_w rd rn rm :
    val = 0x2A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ EOR Wd, Wn, Wm — 32-bit bitwise XOR
\\ Encoding: 0x4A000000 | Rm<<16 | Rn<<5 | Rd
emit_eor_w rd rn rm :
    val = 0x4A000000 | (rm << 16) | (rn << 5) | rd
    arm64_emit32 val

\\ MOVZ Wd, #imm16 — 32-bit move wide with zero
\\ Encoding: 0x52800000 | hw<<21 | imm16<<5 | Rd
emit_movz_w rd imm16 hw :
    val = 0x52800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ MOVK Wd, #imm16, LSL #(hw*16) — 32-bit move wide with keep
\\ Encoding: 0x72800000 | hw<<21 | imm16<<5 | Rd
emit_movk_w rd imm16 hw :
    val = 0x72800000 | (hw << 21) | (imm16 << 5) | rd
    arm64_emit32 val

\\ Helper: load 32-bit immediate into Wd
emit_mov32 rd imm :
    chunk0 = imm & 0xFFFF
    chunk1 = (imm >> 16) & 0xFFFF
    emit_movz_w rd chunk0 0
    if chunk1 > 0
        emit_movk_w rd chunk1 1

\\ ============================================================
\\ EXTEND / EXTRACT OPERATIONS
\\ ============================================================

\\ SXTW Xd, Wn — sign-extend word to 64-bit
\\ Alias for SBFM Xd, Xn, #0, #31
\\ Encoding: 0x93407C00 | Rn<<5 | Rd
emit_sxtw rd rn :
    val = 0x93407C00 | (rn << 5) | rd
    arm64_emit32 val

\\ SXTB Xd, Wn — sign-extend byte to 64-bit
\\ Alias for SBFM Xd, Xn, #0, #7
\\ Encoding: 0x93401C00 | Rn<<5 | Rd
emit_sxtb rd rn :
    val = 0x93401C00 | (rn << 5) | rd
    arm64_emit32 val

\\ SXTH Xd, Wn — sign-extend halfword to 64-bit
\\ Alias for SBFM Xd, Xn, #0, #15
\\ Encoding: 0x93403C00 | Rn<<5 | Rd
emit_sxth rd rn :
    val = 0x93403C00 | (rn << 5) | rd
    arm64_emit32 val

\\ UXTB Wd, Wn — zero-extend byte (clear bits 31:8)
\\ Alias for UBFM Wd, Wn, #0, #7
\\ Encoding: 0x53001C00 | Rn<<5 | Rd
emit_uxtb rd rn :
    val = 0x53001C00 | (rn << 5) | rd
    arm64_emit32 val

\\ UXTH Wd, Wn — zero-extend halfword (clear bits 31:16)
\\ Alias for UBFM Wd, Wn, #0, #15
\\ Encoding: 0x53003C00 | Rn<<5 | Rd
emit_uxth rd rn :
    val = 0x53003C00 | (rn << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ ATOMIC / EXCLUSIVE (for future multi-threaded support)
\\ ============================================================

\\ LDXR Xt, [Xn] — load exclusive register
\\ Encoding: 0xC85F7C00 | Rn<<5 | Rt
emit_ldxr rt rn :
    val = 0xC85F7C00 | (rn << 5) | rt
    arm64_emit32 val

\\ STXR Ws, Xt, [Xn] — store exclusive register
\\ Encoding: 0xC8007C00 | Rs<<16 | Rn<<5 | Rt
emit_stxr rs rt rn :
    val = 0xC8007C00 | (rs << 16) | (rn << 5) | rt
    arm64_emit32 val

\\ ============================================================
\\ COMPLETE INSTRUCTION COUNT SUMMARY
\\ ============================================================
\\ This file provides emitters for the following ARM64 instructions:
\\
\\ Arithmetic (64-bit):  ADD, SUB, ADDS, SUBS, CMP, CMN, NEG, MUL, MADD, MSUB, SDIV, UDIV
\\ Arithmetic (32-bit):  ADD, SUB, SUBS, CMP, MUL, SDIV, UDIV
\\ Logical (register):   AND, ORR, EOR, ANDS, TST, MVN
\\ Logical (32-bit):     AND, ORR, EOR
\\ Logical (immediate):  AND, ORR, EOR  (raw bitmask encoding)
\\ Shift (register):     LSL, LSR, ASR
\\ Shift (immediate):    LSL, LSR, ASR
\\ Move:                 MOV (reg), MOVZ, MOVK, MOVN, mov64 helper, mov32 helper
\\ Move (32-bit):        MOVZ, MOVK
\\ Conditional:          CSEL, CSINC, CSET, CSNEG, CNEG
\\ Branch:               B, BL, B.cond, CBZ, CBNZ, TBZ, TBNZ, BR, BLR, RET
\\ Load/Store (64-bit):  LDR, STR (imm + reg), LDP, STP (signed/pre/post)
\\ Load/Store (32-bit):  LDR, STR (imm)
\\ Load/Store (byte):    LDRB, STRB
\\ Load/Store (half):    LDRH, STRH
\\ Load/Store (signed):  LDRSB, LDRSH, LDRSW
\\ Pre/Post indexed:     LDR, STR, LDP, STP (pre + post)
\\ PC-relative:          ADR, ADRP
\\ Extend:               SXTW, SXTB, SXTH, UXTB, UXTH
\\ System:               SVC, NOP, DSB, DMB, ISB, BRK
\\ Atomic:               LDXR, STXR
\\
\\ Helpers:              arm64_emit32, arm64_reset, arm64_mark
\\                       arm64_patch_bcond, arm64_patch_b, arm64_patch_cbz
\\                       emit_bcond_fwd, emit_b_fwd, emit_cbz_fwd, emit_cbnz_fwd
\\                       emit_prologue, emit_epilogue, emit_prologue_frame, emit_epilogue_frame
\\                       emit_save_pair, emit_restore_pair
\\                       emit_syscall_* (exit, read, write, openat, close, mmap, munmap, mprotect, brk)
\\
\\ Total: 90+ instruction emitters covering everything a self-hosting compiler needs.
