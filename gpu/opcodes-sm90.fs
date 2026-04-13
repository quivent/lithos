\ opcodes-sm90.fs — Empirically verified sm90a (Hopper) opcode constants
\
\ All constants in this file were derived by:
\   1. Writing minimal PTX kernels for each instruction
\   2. Compiling with: ptxas -arch sm_90a
\   3. Disassembling with: nvdisasm --print-instruction-encoding
\
\ Hardware: NVIDIA GH200 (sm_90a)
\ Date verified: 2026-04-13
\ Tool versions: ptxas 12.8.93, nvdisasm 12.8.90
\
\ This file is constants-only — no emit words, no make-ctrl, no control flow.
\ Include this file BEFORE gpu/emit.fs.

\ ============================================================
\ 128-BIT INSTRUCTION FORMAT (informational)
\ ============================================================
\
\ Every Hopper (sm_90a) instruction is 16 bytes = 128 bits:
\   bits  [63:0]   — instruction word  (opcode, registers, immediates)
\   bits [127:64]  — control word      (scheduler: stall, barriers, source mods)
\
\ Uniform field layout for FP32/integer arithmetic ops:
\   bits[15:0]  = opcode
\   bits[23:16] = Rd  (destination register; 8-bit index; RZ=0xFF)
\   bits[31:24] = Ra  (source 1)
\   bits[39:32] = Rb  (source 2)
\   ctrl[7:0]   = Rc  (third source for FFMA/IMAD/IADD3/SHF; RZ=0xFF)
\
\ Control word layout (ctrl = instruction bits [127:64]):
\   ctrl bits [40:0]   — extra41: opaque barrier/descriptor + source modifiers
\   ctrl bits [44:41]  — stall count (0–15 cycles)
\   ctrl bit  45       — yield hint (1 = scheduler may switch warp)
\   ctrl bits [48:46]  — write barrier slot (0–6; 7=none)
\   ctrl bits [51:49]  — read barrier slot  (0–6; 7=none)
\   ctrl bits [57:52]  — wait barriers mask (bit N = stall until slot N clears)
\   ctrl bits [62:58]  — reuse cache flags
\
\ Source modifier encoding (across inst+ctrl boundary):
\   inst bit 63        = NEG on src2  (1 = negate)
\   ctrl extra41 bit 8 = NEG on src1  (1 = negate)
\   ctrl extra41 bit 9 = ABS on src1  (1 = abs)

\ ============================================================
\ FP32 ARITHMETIC OPCODES
\ ============================================================
\
\ All register-register FP32 ops share bits[15:8] = 0x72.
\ Immediate forms use 0x74 (FADD-imm) or 0x78 (FMUL-imm).

$7220 constant OP-FMUL       \ mul.f32       — verified: FMUL R9,R2,R5   → 0x0000000502097220
$7221 constant OP-FADD       \ add.f32       — verified: FADD R9,R2,R5   → 0x0000000502097221
$7223 constant OP-FFMA       \ fma.rn.f32    — verified: FFMA R9,R2,R5,R9→ 0x0000000502097223
$7209 constant OP-FMNMX      \ min/max.f32   — verified: FMNMX R9,R2,R5,PT→0x0000000502097209
$720b constant OP-FSETP      \ setp.*.f32    — verified: FSETP.GTU.AND P3 → 0x000000050200720b
$7421 constant OP-FADD-IMM   \ add.f32 imm   — verified: FADD R7,R2,1   → 0x3f80000002077421
$7820 constant OP-FMUL-IMM   \ mul.f32 imm   — verified: FMUL R7,R2,2   → 0x4000000002077820

\ ============================================================
\ INTEGER ARITHMETIC OPCODES
\ ============================================================

$7224 constant OP-IMAD       \ mul.lo.s32 / mad.lo.s32 — verified: IMAD R9,R2,R5,RZ → 0x0000000502097224
$7225 constant OP-IMAD-WIDE  \ mul.wide.u32            — verified: IMAD.WIDE.U32 R6,R2,R5 → opcode 0x7225
$7210 constant OP-IADD3      \ add.s32 (reg form)      — verified: IADD3 R9,R2,R5,RZ → 0x0000000502097210
$7810 constant OP-IADD3-IMM  \ add.s32 (imm form)      — verified: IADD3 R7,R2,42,RZ → 0x0000002a02077810
$7819 constant OP-SHF        \ shl/shr.b32             — verified: SHF.L.U32 R7,R2,7 → 0x0000000702077819

\ ============================================================
\ BIT OPERATION OPCODES
\ ============================================================

$7212 constant OP-LOP3       \ and/or/xor.b32 (reg form) — verified: LOP3.LUT R9,R2,R5,RZ → 0x0000000502097212
$7812 constant OP-LOP3-IMM   \ and.b32 (imm form)        — verified: LOP3.LUT R7,R2,0xf,RZ→ 0x0000000f02077812
$720c constant OP-ISETP      \ setp.lt/ge/eq.s32          — verified: ISETP.GE.AND P0,PT,R2,R5,PT → 0x000000050200720c

\ ============================================================
\ CONVERSION OPCODES
\ ============================================================
\
\ CORRECTION: OP-I2F was previously 0x7306 (marked "unverified").
\ Actual sm_90a opcode confirmed by probe as 0x7245 (SASS mnemonic I2FP, not I2F).

$7245 constant OP-I2FP       \ cvt.rn.f32.s32/u32 — verified: I2FP.F32.S32 R7,R2 → 0x0000000200077245
$7305 constant OP-F2I        \ cvt.rzi.s32.f32    — verified: F2I.TRUNC.NTZ R7,R2 → 0x0000000200077305

\ ============================================================
\ SFU (SPECIAL FUNCTION UNIT) OPCODE
\ ============================================================
\
\ All MUFU variants share opcode 0x7308. Sub-function is selected by
\ ctrl extra41 bits[13:8] (NOT in the instruction word). See MUFU subop
\ table below.

$7308 constant OP-MUFU       \ all MUFU variants — verified: MUFU.RCP R0,R0 → 0x0000000000007308

\ ============================================================
\ MEMORY OPCODES
\ ============================================================

$7981 constant OP-LDG        \ ld.global.f32  — verified: LDG.E R3,desc[UR4][R2.64] → 0x0000000402037981
$7986 constant OP-STG        \ st.global.f32  — verified: STG.E desc[UR4][R4.64],R3 → 0x0000000304007986
$7984 constant OP-LDS        \ ld.shared.f32  — verified: LDS R9,[UR4+0x0]          → 0x00000004ff097984
$7988 constant OP-STS        \ st.shared.f32  — verified: STS [UR4+0x0],R0          → 0x00000000ff007988
$7388 constant OP-STS-REG    \ st.shared (reg-mode) — verified: STS [R6+0x0],R0    → 0x0000000006007388
$7b82 constant OP-LDC        \ load from cbuf to GPR      — verified: LDC R1,c[0x0][0x28] → 0x00000a00ff017b82
$7ab9 constant OP-ULDC       \ load from cbuf to UR       — verified: ULDC.64 UR4,c[0x0][0x208] → 0x0000820000047ab9

\ ============================================================
\ ATOMIC / REDUCTION OPCODES  (Hopper: ATOMG not ATOM)
\ ============================================================
\
\ CORRECTION: OP-ATOM-ADD was previously 0x798b (a guess).
\ Hopper uses ATOMG (global atomic, descriptor-based), not ATOM.
\ Verified by probe: atom.global.add.u32 → ATOMG.E.ADD.STRONG.GPU
\
\ Field layout for ATOMG (same as other ALU ops):
\   bits[23:16] = Rd  (destination; use RZ=0xFF to discard return value)
\   bits[31:24] = Ra  (address offset register, 64-bit pair)
\   bits[39:32] = Rs  (source data register)
\   UR descriptor register: NOT in word0; encoded in ctrl word

$79a8 constant OP-ATOMG-U32  \ atom.global.add.u32  — verified: ATOMG.E.ADD.STRONG.GPU   → 0x...079a8
$79a3 constant OP-ATOMG-F32  \ atom.global.add.f32  — verified: ATOMG.E.ADD.F32.FTZ.RN   → 0x...079a3
$79a6 constant OP-REDG-F32   \ red.global.add.f32   — verified: REDG.E.ADD.F32.FTZ.RN    → 0x...079a6
\ Note: ATOMG.EXCH also uses 0x79a8 base; distinguished from ADD.U32 entirely by ctrl word.

\ Backward-compat alias (was 0x798b — WRONG; now corrected)
$79a8 constant OP-ATOM-ADD

\ ============================================================
\ WARP / CONTROL FLOW OPCODES
\ ============================================================

$7f89 constant OP-SHFL       \ shfl.sync.bfly/down/up.b32 (imm-delta form) — verified: SHFL.BFLY delta=1 → 0x0c201f0000057f89
$7589 constant OP-SHFL-IDX   \ shfl.sync.idx.b32 (reg-delta, imm-clamp)   — verified: SHFL.IDX RZ,0x1f → 0x00001fff000b7589
$7389 constant OP-SHFL-IDX-RR \ shfl.sync.idx.b32 (full register form)
$7b1d constant OP-BAR-SYNC   \ bar.sync N — verified: BAR.SYNC.DEFER_BLOCKING 0 → 0x0000000000007b1d
$7992 constant OP-MEMBAR     \ membar.gl/sys/cta — verified: MEMBAR.SC.GPU       → 0x0000000000007992
$7947 constant OP-BRA        \ bra TARGET (PT = always taken)              — verified: BRA self-loop  → 0xfffffffc00fc7947
$794d constant OP-EXIT       \ exit — verified: EXIT → 0x000000000000794d
$7919 constant OP-S2R        \ mov.u32 r, %tid.x etc. — verified: S2R SR_TID.X → 0x00000000000r7919
$79c3 constant OP-S2UR       \ store-to-uniform-register (S2UR)
$7918 constant OP-NOP        \ NOP — verified: 0x0000000000007918
$7802 constant OP-MOV-IMM    \ MOV Rd, imm32 — opcode 0x802; imm32 in bits[63:32]
$7235 constant OP-HFMA2      \ HFMA2.MMA (FP16x2 fused multiply-add)

\ ============================================================
\ CTRL WORD CONSTANTS — empirically verified literal values
\ ============================================================
\
\ These replace runtime make-ctrl computation with pre-computed constants.
\ Each was derived from nvdisasm probe output for sm_90a.
\
\ Format: 0x<ctrl64> where ctrl64 = the 64-bit control word (bits [127:64]
\ of the 128-bit instruction, stored as w1 in nvdisasm output).

\ NOP: stall=0 yield=0 wbar=7 rbar=7
$000fc00000000000 constant CTRL-NOP           \ verified: 0x000fc00000000000

\ S2R: stall=7 yield=1 wbar=1 rbar=7; extra41[15:8] = SR-ID (parameterized — use ctrl-s2r)
\ Example: SR_TID.X → ctrl=0x000e6e0000002100, SR_CTAID.X → 0x000eaa0000002500

\ LDG.E: stall=4 yield=1 wbar=2 rbar=7; extra41=0x0c1e1900
$000ea8000c1e1900 constant CTRL-LDG           \ verified: LDG.E R3,desc[UR4][R2.64]
$001ea8000c1e1900 constant CTRL-LDG-WAIT0     \ LDG with wait on barrier 0 (wait=1)

\ STG.E: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0c101904
$000fe2000c101904 constant CTRL-STG           \ verified: STG.E desc[UR4][R4.64],R3
$001fe8000c101904 constant CTRL-STG-WAIT2     \ STG with wait on barrier 2

\ LDS.32: stall=7 yield=1 wbar=1 rbar=7; extra41=0x0800
$000e280008000800 constant CTRL-LDS           \ verified: LDS R9,[UR4+0x0]

\ STS.32: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0800
$000fe80008000804 constant CTRL-STS           \ verified: STS [UR4+0x0],R0

\ LDC: stall=7 yield=1 wbar=1 rbar=7; extra41=0x800
$000e220000000800 constant CTRL-LDC           \ verified: LDC.64 R2,c[0x0][0x210]

\ ULDC.32: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0a00 (bit9=1 for 64-bit width)
$000fe20000000a00 constant CTRL-ULDC          \ verified: ULDC.64 UR4,c[0x0][0x208]

\ FADD: stall=5 yield=0 wbar=7 rbar=7
$000fca0000000000 constant CTRL-FADD          \ verified: FADD R9,R2,R5
$004fca0000000000 constant CTRL-FADD-WAIT2    \ FADD after LDG (wait=4 = barrier 2)

\ FMUL: stall=5 yield=0 wbar=7 rbar=7; extra41=0x400000 (reuse cache)
$004fca0000400000 constant CTRL-FMUL          \ verified: FMUL R9,R2,R5

\ FFMA: Rs3 (accumulator register index) lives in ctrl extra41 bits[7:0].
\   This is inherently parameterized — use ctrl-ffma( rs3 -- ctrl64 ) in emit.fs.
\   Example: FFMA R9,R2,R5,R9 → ctrl=0x008fca0000000009  (Rs3=R9=0x09)

\ IMAD: stall=1 yield=1 wbar=7 rbar=7; extra41 base=0x0f8e0200; Rs3 in bits[7:0]
\   Parameterized — use ctrl-imad( rs3 -- ctrl64 ) in emit.fs.

\ IMAD-IMM: stall=1 yield=1 wbar=7 rbar=7; extra41=0x78e00ff
$000fe200078e00ff constant CTRL-IMAD-IMM      \ verified: IMAD.MOV.U32 pattern

\ ISETP: stall=13 yield=0 wbar=7 rbar=7; extra41=0x0bf06070
$000fda000bf06070 constant CTRL-ISETP         \ verified: ISETP.GE.U32.AND P0

\ I2FP.F32.S32: stall=5 yield=0 wbar=7 rbar=7 wait=4; extra41=0x00201400
$004fca0000201400 constant CTRL-I2FP-S32      \ verified: I2FP.F32.S32 R7,R2

\ I2FP.F32.U32: same as S32 but different type bits in extra41
$004fca0000201000 constant CTRL-I2FP-U32      \ verified: I2FP.F32.U32 R7,R2

\ F2I.TRUNC.NTZ: stall=2 yield=1 wbar=0 rbar=7 wait=4; extra41=0x0020f100
$004e24000020f100 constant CTRL-F2I           \ verified: F2I.TRUNC.NTZ R7,R2

\ SHFL.BFLY/DOWN/UP: stall=14 yield=0 wbar=3 rbar=7; extra41=0x000e0000
$001e6800000e0000 constant CTRL-SHFL          \ verified: SHFL.BFLY delta=1, R5,R0

\ BAR.SYNC: stall=5 yield=1 wbar=7 rbar=7; extra41=0x00010000
$000fec0000010000 constant CTRL-BAR           \ verified: BAR.SYNC.DEFER_BLOCKING 0

\ MEMBAR — scope variants (instruction word is always 0x0000000000007992):
$000fec0000002000 constant CTRL-MEMBAR-GPU    \ verified: MEMBAR.SC.GPU (extra41=0x2000)
$000fec0000003000 constant CTRL-MEMBAR-SYS    \ verified: MEMBAR.SC.SYS (extra41=0x3000)
$0003ec0000000000 constant CTRL-MEMBAR-CTA    \ verified: MEMBAR.SC.CTA (extra41=0x0000)

\ BRA: stall=0 yield=0 wbar=7 rbar=7; extra41=0x0383ffff
$000fc0000383ffff constant CTRL-BRA           \ verified: BRA self-loop

\ EXIT: stall=5 yield=1 wbar=7 rbar=7; extra41=0x03800000
$000fea0003800000 constant CTRL-EXIT          \ verified: EXIT

\ ============================================================
\ ATOMG / REDG CTRL WORD CONSTANTS
\ ============================================================
\
\ CORRECTION: The old ctrl-atom (computed via make-ctrl with extra41=0x00100000)
\ produced 0x001ffc0000100000 — completely wrong.
\ Correct values read directly from nvdisasm probe output:

$004e2800081ee1c4 constant CTRL-ATOMG-U32     \ verified: ATOMG.E.ADD.STRONG.GPU (integer counter)
$004e2800081ef3c4 constant CTRL-ATOMG-F32     \ verified: ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU
$004e28000c1ee1c4 constant CTRL-ATOMG-EXCH    \ verified: ATOMG.E.EXCH.STRONG.GPU
$004fe2000c10f384 constant CTRL-REDG-F32      \ verified: REDG.E.ADD.F32.FTZ.RN.STRONG.GPU (no return)

\ Backward-compat aliases used by emit.fs grid-sync code
$004e2800081ee1c4 constant CTRL-ATOM-U32      \ = CTRL-ATOMG-U32
$004e2800081ef3c4 constant CTRL-ATOM-F32      \ = CTRL-ATOMG-F32

\ ============================================================
\ MUFU SUBOP TABLE  (ctrl extra41 bits[13:8])
\ ============================================================
\
\ All MUFU variants share OP-MUFU = 0x7308 in the instruction word.
\ The sub-function is selected ONLY by ctrl extra41 bits[13:8].
\ These constants are the extra41 values (full field, not just the 6-bit index).
\
\ Verified from probe cubins compiled with ptxas sm_90a / nvdisasm 12.8.90:
\   MUFU.RCP  ctrl=0x000e640000001000  extra41=0x001000  bits[13:8]=0x10
\   MUFU.EX2  ctrl=0x000e640000000800  extra41=0x000800  bits[13:8]=0x08
\   MUFU.RSQ  ctrl=0x000e640000001400  extra41=0x001400  bits[13:8]=0x14
\   MUFU.LG2  ctrl=0x000e640000000c00  extra41=0x000c00  bits[13:8]=0x0c
\   MUFU.SIN  ctrl=0x000e240000000400  extra41=0x000400  bits[13:8]=0x04
\   MUFU.COS  ctrl=0x000e240000000000  extra41=0x000000  bits[13:8]=0x00
\   MUFU.SQRT ctrl=0x000e640000002000  extra41=0x002000  bits[13:8]=0x20

$0000 constant MUFU-COS      \ cos.approx.f32  — extra41 bits[13:8]=0x00
$0400 constant MUFU-SIN      \ sin.approx.f32  — extra41 bits[13:8]=0x04
$0800 constant MUFU-EX2      \ ex2.approx.f32  — extra41 bits[13:8]=0x08
$0c00 constant MUFU-LG2      \ lg2.approx.f32  — extra41 bits[13:8]=0x0c
$1000 constant MUFU-RCP      \ rcp.approx.f32  — extra41 bits[13:8]=0x10
$1400 constant MUFU-RSQ      \ rsqrt.approx.f32 — extra41 bits[13:8]=0x14
$2000 constant MUFU-SQRT     \ sqrt.approx.f32  — extra41 bits[13:8]=0x20

\ ============================================================
\ LOP3 TRUTH TABLE CONSTANTS  (ctrl extra41 bits[15:8])
\ ============================================================
\
\ LOP3.LUT implements AND/OR/XOR via an 8-bit lookup table.
\ The LUT byte lives in ctrl extra41 bits[15:8].
\ Verified from probe: and.b32 → LOP3.LUT ctrl=0x004fca00078ec0ff
\   bits[15:8] = 0xc0 (AND table)
\
\ Truth table semantics (A=src0, B=src1, C=src2; result bit = LUT[C,B,A]):
\   AND: result = A & B        → LUT = 0b11000000 = 0xc0
\   OR:  result = A | B        → LUT = 0b11111100 = 0xfc
\   XOR: result = A ^ B        → LUT = 0b00111100 = 0x3c

$c0 constant LOP3-AND         \ 0xc0 = 11000000b — AND truth table
$fc constant LOP3-OR          \ 0xfc = 11111100b — OR  truth table
$3c constant LOP3-XOR         \ 0x3c = 00111100b — XOR truth table

\ ============================================================
\ SPECIAL REGISTER IDs  (S2R ctrl extra41 bits[15:8])
\ ============================================================
\
\ For S2R, the inst word is always: 0x00000000000<Rd>7919
\ The SR selector lives ONLY in ctrl extra41[15:8], not the inst word.
\ Formula: ctrl extra41 = SR-ID << 8
\
\ Verified from probe_s2r2 and probe_s2r3 (nvdisasm -O0):
\   SR_TID.X   extra41=0x2100  sr-id=0x21
\   SR_TID.Y   extra41=0x2200  sr-id=0x22
\   SR_TID.Z   extra41=0x2300  sr-id=0x23
\   SR_CTAID.X extra41=0x2500  sr-id=0x25
\   SR_CTAID.Y extra41=0x2600  sr-id=0x26
\   SR_CTAID.Z extra41=0x2700  sr-id=0x27
\   SR_LANEID  extra41=0x0000  sr-id=0x00
\
\ Note: %nctaid.x / %warpid are lowered to LDC (constant-bank loads) by ptxas.

$21 constant SR-TID-X         \ %tid.x   — lane index within thread block (x dim)
$22 constant SR-TID-Y         \ %tid.y
$23 constant SR-TID-Z         \ %tid.z
$25 constant SR-CTAID-X       \ %ctaid.x — block index within grid (x dim)
$26 constant SR-CTAID-Y       \ %ctaid.y
$27 constant SR-CTAID-Z       \ %ctaid.z
$00 constant SR-LANEID        \ %laneid  — lane within warp (0..31)

\ ============================================================
\ REGISTER ENCODING NOTES
\ ============================================================
\
\ RZ (zero register): reads as 0, writes discarded.
\   In inst word register fields (bits[23:16], [31:24], [39:32]): 0xFF = 255
\   In ctrl word Rc field (bits[7:0]): 0xFF = RZ
\
\ PT (always-true predicate): encoded as 0x7 in predicate fields.
\   BRA uses bits[23:16] = 0xfc for PT (always taken).
\
\ Width bit: bit 9 of ctrl w1[15:8] promotes 32-bit memory ops to 64-bit.
\   LDG/STG 32-bit: w1[15:8] = 0x19  |  64-bit: w1[15:8] = 0x1b
\   LDS/STS/LDC/ULDC 32-bit: 0x08   |  64-bit: 0x0a

\ ============================================================
\ CORRECTIONS vs PRIOR EMIT-SASS.FS VERSIONS
\ ============================================================
\
\ 1. OP-I2F: was 0x7306 ("unverified") → now 0x7245 (I2FP.F32.S32, verified)
\    SASS mnemonic is I2FP, not I2F. Source register at bits[39:32], not [31:24].
\
\ 2. OP-ATOM-ADD: was 0x798b (guessed) → now 0x79a8 (ATOMG.ADD.U32, verified)
\    Hopper uses ATOMG (descriptor-based global atomic), not ATOM.
\    ATOMG.ADD.F32 is a separate opcode: 0x79a3.
\
\ 3. ctrl-atom computation (make-ctrl with extra41=0x00100000):
\    Computed: 0x001ffc0000100000 — WRONG.
\    Actual CTRL-ATOMG-U32: 0x004e2800081ee1c4 (verified from nvdisasm).
\
\ 4. SHFL encoding (warp_control.md):
\    Old: mode_byte = (log2(delta)<<4)|0x0c, clamp=0x20 (only correct for delta=1)
\    New: delta_enc = delta*32; clamp = delta_enc & 0xff; mode = 0x0c|(delta_enc>>8)
\    Delta=16 old: top16=0x4c20 (WRONG) → new: top16=0x0e00 (verified)
\
\ 5. bar-sync barrier ID field position:
\    Old: bar_id << 16 → placed ID in bits[23:16] (WRONG)
\    New: bar_id << 54 → places ID in bits[55:48] (verified: BAR.SYNC 1 = 0x0040000000007b1d)
\    Only bar-id=0 was unaffected (shifting 0 anywhere gives 0).
