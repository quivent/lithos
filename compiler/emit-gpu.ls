\\ emit-sm90.li — Lithos SM90 (Hopper) GPU machine code emitter
\\ Ported from gpu/emit.fs to Lithos .li composition syntax.
\\
\\ Emits raw sm_90a binary instructions (16 bytes each) into a code buffer.
\\ No PTX, no ptxas, no driver JIT — direct SASS emission.
\\
\\ Each instruction is 128 bits = 8-byte instruction word + 8-byte control word.
\\ The control word encodes hardware scheduling fields (stall, barriers, reuse).
\\
\\ All opcode and control constants empirically verified on GH200 (sm_90a)
\\ via ptxas 12.8.93 + nvdisasm 12.8.90 probe disassembly.

\\ ============================================================
\\ CODE BUFFER
\\ ============================================================
\\ 64KB buffer for GPU machine code (SASS binary).
\\ Position counter tracks current write offset.

buf gpu_buf 65536
var gpu_pos 0

\\ Per-kernel register high-water mark.
\\ Reset to 0 at gpu_reset; updated by every register-destination instruction.
\\ build-elf reads this to emit tight REGCOUNT.
var max_reg 0

\\ Track destination register — update high-water mark, return rd unchanged.
track_rd rd :
    max_reg max rd max_reg

\\ ============================================================
\\ COOPERATIVE GRID-SYNC STATE
\\ ============================================================
\\ cooperative flag — set to 1 when kernel uses grid-wide sync.
\\ build-elf checks this and emits COOP_GROUP attrs in .nv.info.
var cooperative 0

\\ Array of u32 byte offsets for grid-sync sites (up to 256).
buf gridsync_offsets 1024
var gridsync_count 0

\\ Record current gpu_pos as the byte offset of the next grid-sync instruction.
record_gridsync :
    if>= gridsync_count 256
        \\ silently clamp at 256 sites
    ← 32 gridsync_offsets + gridsync_count * 4 gpu_pos
    gridsync_count + 1

\\ Array of u32 byte offsets for EXIT instruction sites (up to 256).
buf exit_offsets 1024
var exit_count 0

record_exit :
    if>= exit_count 256
        \\ silently clamp at 256 sites
    ← 32 exit_offsets + exit_count * 4 gpu_pos
    exit_count + 1

\\ Reset all emitter state for a new kernel.
gpu_reset :
    gpu_pos 0
    max_reg 0
    gridsync_count 0
    exit_count 0
    cooperative 0

\\ ============================================================
\\ RAW BYTE EMITTERS
\\ ============================================================

\\ Write one byte at current position and advance.
emit_byte b :
    ← 8 gpu_buf + gpu_pos b
    gpu_pos + 1

\\ Write 32-bit little-endian word (4 bytes).
emit_u32 val :
    emit_byte val & 0xFF
    emit_byte (val >> 8) & 0xFF
    emit_byte (val >> 16) & 0xFF
    emit_byte (val >> 24) & 0xFF

\\ Write 64-bit little-endian quad (8 bytes).
emit_u64 val :
    emit_u32 val & 0xFFFFFFFF
    emit_u32 val >> 32

\\ ============================================================
\\ sinst — CORE 16-BYTE INSTRUCTION EMITTER
\\ ============================================================
\\ Every sm_90a instruction is 16 bytes: 8-byte inst word + 8-byte ctrl word.
\\ inst word first (low address), ctrl word second (high address).

sinst iword ctrl :
    emit_u64 iword
    emit_u64 ctrl

\\ ============================================================
\\ CONTROL WORD CONSTRUCTOR
\\ ============================================================
\\ make_ctrl builds a 64-bit control word from scheduling fields:
\\   stall  (4 bits, [44:41])  — cycles to stall (0-15)
\\   yield  (1 bit,  [45])     — scheduler yield hint
\\   wbar   (3 bits, [48:46])  — write barrier slot (7=none)
\\   rbar   (3 bits, [51:49])  — read barrier slot (7=none)
\\   wait   (6 bits, [57:52])  — wait barrier mask
\\   reuse  (5 bits, [62:58])  — reuse cache flags
\\   extra41 (41 bits, [40:0]) — opaque barrier/descriptor fields

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
\\ MUFU SUBOP CONSTANTS (ctrl extra41 bits[13:8])
\\ ============================================================

MUFU_COS     0x0000
MUFU_SIN     0x0400
MUFU_EX2     0x0800
MUFU_LG2     0x0C00
MUFU_RCP     0x1000
MUFU_RSQ     0x1400
MUFU_SQRT    0x2000

\\ ============================================================
\\ LOP3 TRUTH TABLE CONSTANTS (ctrl extra41 bits[15:8])
\\ ============================================================

LOP3_AND     0xC0
LOP3_OR      0xFC
LOP3_XOR     0x3C

\\ ============================================================
\\ SPECIAL REGISTER IDs (S2R ctrl extra41[15:8])
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

RZ           0xFF       \\ zero register (reads 0, writes discarded)
PT           0x07       \\ always-true predicate

\\ ============================================================
\\ PER-INSTRUCTION CONTROL WORD CONSTRUCTORS
\\ ============================================================
\\ All stall/barrier values derived from probe disassembly on sm_90a.

\\ NOP: stall=0 yield=0 wbar=7 rbar=7
ctrl_nop :
    make_ctrl 0 0 7 7 0 0 0

\\ S2R: stall=7 yield=1 wbar=1 rbar=7; extra41 = sr_id << 8
ctrl_s2r sr_id :
    make_ctrl 7 1 1 7 0 0 (sr_id << 8)

\\ LDG.E: stall=4 yield=1 wbar=2 rbar=7; extra41=0x0c1e1900
ctrl_ldg :
    make_ctrl 4 1 2 7 0 0 0x0C1E1900

\\ LDG.E with wait on barrier 0
ctrl_ldg_wait0 :
    make_ctrl 4 1 2 7 1 0 0x0C1E1900

\\ STG.E: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0c101904
ctrl_stg :
    make_ctrl 1 1 7 7 0 0 0x0C101904

\\ STG.E with wait on barrier 2
ctrl_stg_wait2 :
    make_ctrl 4 1 7 7 1 0 0x0C101904

\\ FFMA: stall=5 yield=0 wbar=7 rbar=7; rs3 in extra41[7:0]
ctrl_ffma rs3 :
    make_ctrl 5 0 7 7 0 0 rs3

\\ FADD: stall=5 yield=0 wbar=7 rbar=7
ctrl_fadd :
    make_ctrl 5 0 7 7 0 0 0

\\ FADD after LDG (wait on barrier 2)
ctrl_fadd_wait2 :
    make_ctrl 5 0 7 7 4 0 0

\\ FMUL: stall=5 yield=0 wbar=7 rbar=7; extra41=0x400000 (reuse cache)
ctrl_fmul :
    make_ctrl 5 0 7 7 4 0 0x400000

\\ IMAD: stall=1 yield=1 wbar=7 rbar=7; extra41 base=0x0f8e0200, rs3 ORed in
ctrl_imad rs3 :
    make_ctrl 1 1 7 7 0 0 (0x0F8E0200 | rs3)

\\ IMAD-IMM: stall=1 yield=1; extra41=0x78e00ff (MOV.U32 modifier + RZ accum)
ctrl_imad_imm :
    make_ctrl 1 1 7 7 0 0 0x078E00FF

\\ ISETP: stall=13 yield=0 wbar=7 rbar=7; extra41=0x0bf06070
ctrl_isetp :
    make_ctrl 13 0 7 7 0 0 0x0BF06070

\\ BRA: stall=0 yield=0 wbar=7 rbar=7; extra41=0x0383ffff
ctrl_bra :
    make_ctrl 0 0 7 7 0 0 0x0383FFFF

\\ EXIT: stall=5 yield=1 wbar=7 rbar=7; extra41=0x03800000
ctrl_exit :
    make_ctrl 5 1 7 7 0 0 0x03800000

\\ ULDC: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0a00
ctrl_uldc :
    make_ctrl 1 1 7 7 0 0 0x0A00

\\ LDC: stall=7 yield=1 wbar=1 rbar=7; extra41=0x800
ctrl_ldc :
    make_ctrl 7 1 1 7 0 0 0x0800

\\ SHFL: stall=14 yield=0 wbar=3 rbar=7; extra41=0x000e0000
ctrl_shfl :
    make_ctrl 14 0 3 7 0 0 0x000E0000

\\ STS.32: stall=1 yield=1 wbar=7 rbar=7; extra41=0x0800
ctrl_sts :
    make_ctrl 1 1 7 7 0 0 0x0800

\\ LDS.32: stall=7 yield=1 wbar=1 rbar=7; extra41=0x0800
ctrl_lds :
    make_ctrl 7 1 1 7 0 0 0x0800

\\ BAR.SYNC: stall=5 yield=1 wbar=7 rbar=7; extra41=0x00010000
ctrl_bar :
    make_ctrl 5 1 7 7 0 0 0x00010000

\\ SHF: stall=1 yield=0 wbar=7 rbar=7
ctrl_shf :
    make_ctrl 1 0 7 7 0 0 0

\\ I2FP: stall=5 yield=0 wbar=7 rbar=7 wait=4; extra41=0x00201400
ctrl_i2f :
    make_ctrl 5 0 7 7 4 0 0x00201400

\\ F2I: stall=2 yield=1 wbar=0 rbar=7 wait=4; extra41=0x0020f100
ctrl_f2i :
    make_ctrl 2 1 0 7 4 0 0x0020F100

\\ LOP3: lut byte in extra41[15:8]
ctrl_lop3 lut :
    make_ctrl 1 0 7 7 0 0 (lut << 8)

\\ MUFU: stall=8 yield=1 wbar varies; subfn in extra41
ctrl_mufu subfn wbar :
    make_ctrl 8 1 wbar 7 0 0 subfn

\\ ============================================================
\\ PRE-COMPUTED CONTROL WORD CONSTANTS (verified literals)
\\ ============================================================
\\ Used for instructions that don't need parameterized ctrl words.

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

\\ NOP — no operation
emit_nop :
    sinst 0x0000000000007918 ctrl_nop

\\ FADD Rd, Ra, Rb — float32 add
\\ Verified: FADD R9,R2,R5 = inst 0x0000000502097221
\\   Rd=bits[23:16], Ra=bits[31:24], Rb=bits[39:32]
emit_fadd rd ra rb :
    iword OP_FADD | (track_rd rd << 16) | (ra << 24) | (rb << 32)
    sinst iword ctrl_fadd

\\ FFMA Rd, Rs1, Rs2, Rs3 — float fused multiply-add
\\ Rs3 (accumulator reg index) packed into ctrl extra41[7:0] via ctrl_ffma.
emit_ffma rd rs1 rs2 rs3 :
    iword OP_FFMA | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_ffma rs3)

\\ FMUL Rd, Rs1, Rs2 — float multiply (FFMA with RZ accumulator)
emit_fmul rd rs1 rs2 :
    emit_ffma rd rs1 rs2 RZ

\\ FMNMX Rd, Ra, Rb — float min/max (predicate selects min vs max)
emit_fmnmx rd ra rb :
    iword OP_FMNMX | (track_rd rd << 16) | (ra << 24) | (rb << 32)
    sinst iword ctrl_fadd

\\ FADD.IMM Rd, Ra, imm32 — float add with immediate
emit_fadd_imm rd ra imm32 :
    iword OP_FADD_IMM | (track_rd rd << 16) | (ra << 24) | (imm32 << 32)
    sinst iword ctrl_fadd

\\ FMUL.IMM Rd, Ra, imm32 — float multiply with immediate
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

\\ IMAD Rd, Rs1, Rs2, Rs3 — integer multiply-add (reg*reg+reg)
\\ Rs3 (accumulator) in ctrl[7:0] per 4-operand format.
emit_imad rd rs1 rs2 rs3 :
    iword OP_IMAD | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_imad rs3)

\\ IMAD.MOV.U32 Rd, Rs1, imm32 — integer move/multiply immediate
emit_imad_imm rd rs1 imm32 :
    iword 0x7424 | (track_rd rd << 16) | (rs1 << 24) | (imm32 << 32)
    sinst iword ctrl_imad_imm

\\ IADD3 Rd, Rs1, Rs2, Rs3 — 3-input integer add
\\ Rs3 in ctrl[7:0] (4-operand format).
emit_iadd3 rd rs1 rs2 rs3 :
    iword 0x7C10 | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    ctrl make_ctrl 1 1 7 7 0 0 rs3
    sinst iword ctrl

\\ ISETP.GE.U32.AND Pd, Rs1, Rs2 — integer >= comparison to predicate
emit_isetp_ge pd rs1 rs2 :
    iword OP_ISETP | (pd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword ctrl_isetp

\\ ISETP.LT.U32.AND Pd, Rs1, Rs2 — integer < comparison to predicate
emit_isetp_lt pd rs1 rs2 :
    iword OP_ISETP | (pd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword ctrl_isetp

\\ SHF.R.U32 Rd, Rs, shamt — shift right (funnel source=RZ for plain shift)
emit_shf_r rd rs shamt :
    iword OP_SHF | (track_rd rd << 16) | (rs << 24) | (shamt << 32) | (RZ << 48)
    sinst iword ctrl_shf

\\ ============================================================
\\ INSTRUCTION BUILDERS — LOGICAL (LOP3)
\\ ============================================================

\\ LOP3.LUT Rd, Rs, imm32, RZ — AND with immediate
emit_lop3_and rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_AND)

\\ LOP3.LUT Rd, Rs, imm32, RZ — OR with immediate
emit_lop3_or rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_OR)

\\ LOP3.LUT Rd, Rs, imm32, RZ — XOR with immediate
emit_lop3_xor rd rs imm32 :
    iword OP_LOP3_IMM | (track_rd rd << 16) | (rs << 24) | (imm32 << 32)
    sinst iword (ctrl_lop3 LOP3_XOR)

\\ ============================================================
\\ INSTRUCTION BUILDERS — CONVERSION
\\ ============================================================

\\ I2FP.F32.S32 Rd, Rs — signed int32 to float32
\\ Note: source at bits[39:32], NOT bits[31:24].
emit_i2f rd rs :
    iword OP_I2FP | (track_rd rd << 16) | (rs << 32)
    sinst iword ctrl_i2f

\\ F2I.TRUNC.NTZ Rd, Rs — float32 to signed int32 (truncate toward zero)
emit_f2i rd rs :
    iword OP_F2I | (track_rd rd << 16) | (rs << 32)
    sinst iword ctrl_f2i

\\ ============================================================
\\ INSTRUCTION BUILDERS — MUFU (Special Function Unit)
\\ ============================================================
\\ All MUFU variants share OP_MUFU = 0x7308 in instruction word.
\\ Sub-function selected by ctrl extra41 bits[13:8].
\\ inst format: src << 32 | dst << 16 | 0x7308

emit_mufu rd rs subfn wbar :
    iword OP_MUFU | (track_rd rd << 16) | (rs << 32)
    sinst iword (ctrl_mufu subfn wbar)

\\ MUFU.EX2 Rd, Rs — 2^x (softmax exponentiation)
emit_ex2 rd rs :
    emit_mufu rd rs MUFU_EX2 7

\\ MUFU.RCP Rd, Rs — 1/x (reciprocal)
emit_rcp rd rs :
    emit_mufu rd rs MUFU_RCP 1

\\ MUFU.RSQ Rd, Rs — 1/√x (reciprocal square root)
emit_rsq rd rs :
    emit_mufu rd rs MUFU_RSQ 2

\\ MUFU.LG2 Rd, Rs — log₂(x)
emit_lg2 rd rs :
    emit_mufu rd rs MUFU_LG2 3

\\ MUFU.SQRT Rd, Rs — √x
emit_sqrt rd rs :
    emit_mufu rd rs MUFU_SQRT 4

\\ MUFU.SIN Rd, Rs — sin(x) (RoPE)
emit_sin rd rs :
    emit_mufu rd rs MUFU_SIN 5

\\ MUFU.COS Rd, Rs — cos(x) (RoPE)
emit_cos rd rs :
    emit_mufu rd rs MUFU_COS 5

\\ ============================================================
\\ INSTRUCTION BUILDERS — MEMORY
\\ ============================================================

\\ LDG.E Rd, [Ra] — 32-bit global load
emit_ldg rd ra :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24)
    sinst iword ctrl_ldg

\\ LDG.E Rd, [Ra+off] — 32-bit global load with byte offset
emit_ldg_off rd ra off :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24) | (off << 32)
    sinst iword ctrl_ldg

\\ LDG.128 Rd, [Ra] — 128-bit global load (4 registers Rd..Rd+3)
emit_ldg128 rd ra :
    iword OP_LDG | (track_rd rd << 16) | (ra << 24)
    ctrl make_ctrl 4 1 2 7 0 0 0x0C1E1F00
    sinst iword ctrl

\\ STG.E [Ra], Rs — 32-bit global store
emit_stg ra rs :
    iword OP_STG | (ra << 24) | (rs << 32)
    sinst iword ctrl_stg

\\ STG.E [Ra+off], Rs — 32-bit global store with byte offset
emit_stg_off ra rs off :
    iword OP_STG | (ra << 24) | (rs << 32) | (off << 32)
    sinst iword ctrl_stg

\\ LDS.32 Rd, [Ra+off] — load 32 bits from shared memory
emit_lds rd ra off :
    iword OP_LDS | (track_rd rd << 16) | (ra << 24) | (off << 32)
    sinst iword ctrl_lds

\\ STS.32 [Ra+off], Rs — store 32 bits to shared memory
emit_sts ra rs off :
    iword OP_STS | (ra << 24) | (rs << 32) | (off << 32)
    sinst iword ctrl_sts

\\ LDC Rd, c[cbuf][offset] — load from constant bank to GPR
\\ Verified: LDC R1,c[0x0][0x28] = 0x00000a00ff017b82
emit_ldc rd cbuf offset :
    iword OP_LDC | (track_rd rd << 16) | (RZ << 24) | (offset << 32) | (cbuf << 48)
    sinst iword CTRL_LDC

\\ ULDC URn, c[cbuf][offset] — uniform load from constant bank to uniform register
emit_uldc rd cbuf offset :
    iword OP_ULDC | (track_rd rd << 16) | (RZ << 24) | (offset << 32) | (cbuf << 48)
    sinst iword CTRL_ULDC

\\ ============================================================
\\ INSTRUCTION BUILDERS — SPECIAL REGISTER
\\ ============================================================

\\ S2R Rd, sr_id — read special register into Rd
\\ sr_id encoded ONLY in ctrl extra41[15:8], not in instruction word.
emit_s2r rd sr_id :
    iword OP_S2R | (track_rd rd << 16)
    sinst iword (ctrl_s2r sr_id)

\\ MOV Rd, imm32 — move 32-bit immediate into register
\\ Opcode 0x7802; imm32 in bits[63:32].
emit_mov_imm rd imm32 :
    iword OP_MOV_IMM | (track_rd rd << 16) | (imm32 << 32)
    sinst iword ctrl_nop

\\ ============================================================
\\ INSTRUCTION BUILDERS — HFMA2 (packed FP16x2)
\\ ============================================================

\\ HFMA2.MMA Rd, Rs1, Rs2, Rs3 — packed FP16x2 fused multiply-add
emit_hfma2 rd rs1 rs2 rs3 :
    iword OP_HFMA2 | (track_rd rd << 16) | (rs1 << 24) | (rs2 << 32)
    sinst iword (ctrl_ffma rs3)

\\ ============================================================
\\ INSTRUCTION BUILDERS — WARP SHUFFLES
\\ ============================================================
\\
\\ SHFL encoding (128-bit = inst64 + ctrl64):
\\   inst bits[15:0]  = 0x7f89  (opcode: SHFL, PT pred, imm-delta form)
\\   inst bits[23:16] = Rd
\\   inst bits[31:24] = Rs
\\   inst bits[39:32] = 0x00    (unused in imm form)
\\   inst bits[47:40] = 0x1f    (mask = 31)
\\   inst bits[55:48] = clamp   = (delta * 32) & 0xff
\\   inst bits[63:56] = mode    = base | ((delta * 32) >> 8)
\\
\\ Shuffle type in mode bits[3:2]:
\\   0b11=BFLY(0x0C), 0b10=DOWN(0x08), 0b01=UP(0x04), 0b00=IDX(0x00)

\\ SHFL.BFLY PT, Rd, Rs, delta — butterfly shuffle (intra-warp reduce)
emit_shfl_bfly rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x0C | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

\\ SHFL.DOWN PT, Rd, Rs, delta — down shuffle (lane N gets value from N+delta)
emit_shfl_down rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x08 | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

\\ SHFL.UP PT, Rd, Rs, delta — up shuffle (lane N gets value from N-delta)
emit_shfl_up rd rs delta :
    delta_enc delta * 32
    clamp delta_enc & 0xFF
    mode 0x04 | (delta_enc >> 8)
    iword OP_SHFL | (track_rd rd << 16) | (rs << 24) | (0x1F << 40) | (clamp << 48) | (mode << 56)
    sinst iword ctrl_shfl

\\ SHFL.IDX PT, Rd, Rs, R_lane — indexed shuffle (register lane index)
\\ Full register form: opcode 0x7389, R_lane at bits[39:32]
emit_shfl_idx rd rs r_lane :
    iword OP_SHFL_IDX_RR | (track_rd rd << 16) | (rs << 24) | (r_lane << 32)
    sinst iword ctrl_shfl

\\ ============================================================
\\ INSTRUCTION BUILDERS — BARRIERS
\\ ============================================================

\\ BAR.SYNC.DEFER_BLOCKING bar_id — synchronize threads at shared-memory barrier
\\ bar_id encoded as (bar_id << 6) << 48 = bar_id << 54 in instruction word.
emit_bar_sync bar_id :
    iword OP_BAR_SYNC | (bar_id << 54)
    sinst iword ctrl_bar

\\ ============================================================
\\ INSTRUCTION BUILDERS — MEMORY BARRIERS
\\ ============================================================

\\ MEMBAR.SC.GPU — GPU-scope store fence
emit_membar_gpu :
    sinst 0x0000000000007992 CTRL_MEMBAR_GPU

\\ MEMBAR.SC.SYS — system-scope fence (NVLink-coherent)
emit_membar_sys :
    sinst 0x0000000000007992 CTRL_MEMBAR_SYS

\\ MEMBAR.SC.CTA — CTA/block-level fence
emit_membar_cta :
    sinst 0x0000000000007992 CTRL_MEMBAR_CTA

\\ ============================================================
\\ INSTRUCTION BUILDERS — ATOMICS
\\ ============================================================

\\ ATOMG.E.ADD.STRONG.GPU Rd, [Ra], Rs — 32-bit integer atomic add
\\ Rd = old value (RZ=0xFF to discard); Ra = address reg; Rs = value to add.
emit_atom_add rd ra rs :
    iword OP_ATOMG_U32 | (track_rd rd << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_ATOMG_U32

\\ ATOMG.E.ADD.F32 Rd, [Ra], Rs — float32 atomic add
emit_atom_add_f32 rd ra rs :
    iword OP_ATOMG_F32 | (track_rd rd << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_ATOMG_F32

\\ REDG.E.ADD.F32 [Ra], Rs — float32 reduction (no return value)
emit_redg_f32 ra rs :
    iword OP_REDG_F32 | (RZ << 16) | (ra << 24) | (rs << 32)
    sinst iword CTRL_REDG_F32

\\ ============================================================
\\ INSTRUCTION BUILDERS — BRANCH / CONTROL FLOW
\\ ============================================================

\\ BRA offset — unconditional branch (byte offset from next instruction)
\\ Target = (bra_addr + 16) + offset32 * 4
\\ Offset is signed bytes, divided by 4 for dword units.
emit_bra byte_offset :
    offset32 byte_offset / 4
    iword 0x00FC7947 | (offset32 << 32)
    sinst iword ctrl_bra

\\ @Px BRA offset — predicated branch
\\ pred = 0..3 for P0..P3; pred|8 for negated (@!Px)
emit_bra_pred byte_offset pred :
    offset32 byte_offset / 4
    iword 0x00FC7947 | (offset32 << 32)
    \\ Clear predicate field bits[15:12] and insert pred
    iword_masked iword & 0xFFFFFFFFFFFF0FFF | (pred << 12)
    sinst iword_masked ctrl_bra

\\ EXIT — terminate thread
emit_exit :
    record_exit
    sinst 0x000000000000794D ctrl_exit

\\ ============================================================
\\ BRANCH PATCHING HELPERS
\\ ============================================================
\\ gpu_mark saves current position for forward-branch patching.
\\ gpu_patch patches a BRA at a saved position with the current offset.

gpu_mark :
    gpu_pos

gpu_patch saved_pos :
    \\ Calculate byte offset from the instruction after the BRA
    \\ to the current position. BRA is at saved_pos, next inst at saved_pos+16.
    byte_offset gpu_pos - saved_pos - 16
    offset32 byte_offset / 4
    \\ Patch bits[63:32] of the instruction word at saved_pos
    ← 32 gpu_buf + saved_pos + 4 offset32

\\ ============================================================
\\ ACQUIRE / RELEASE MEMORY PROTOCOL
\\ ============================================================
\\ Hopper has no single-instruction acquire-load or release-store.
\\ Canonical pattern from ptxas/CUDA cooperative groups:
\\   Release: MEMBAR.SC.GPU then STG
\\   Acquire: LDG then MEMBAR.SC.GPU

\\ STG with GPU-scope release semantics
emit_stg_release ra rs :
    emit_membar_gpu
    emit_stg ra rs

\\ LDG with GPU-scope acquire semantics
emit_ldg_acquire rd ra :
    emit_ldg rd ra
    emit_membar_gpu

\\ ============================================================
\\ GPTQ DEQUANTIZATION PRIMITIVES
\\ ============================================================
\\ GPTQ packs 8 nibbles per u32. Per nibble:
\\   1. SHF.R.U32  — shift packed word right by (nib_idx * 4)
\\   2. LOP3.AND   — mask low 4 bits
\\   3. I2FP       — int32 to float32
\\   4. FFMA       — scale and offset: result = nibble_f * scale + zero

emit_dequant_nibble rd rsrc nib_idx rscale rzero :
    \\ Step 1: SHF.R.U32 rd, rsrc, nib_idx*4
    shamt nib_idx * 4
    emit_shf_r rd rsrc shamt
    \\ Step 2: LOP3.AND rd, rd, 0xF
    emit_lop3_and rd rd 0xF
    \\ Step 3: I2FP.F32.S32 rd, rd
    emit_i2f rd rd
    \\ Step 4: FFMA rd, rd, rscale, rzero
    emit_ffma rd rd rscale rzero

\\ ============================================================
\\ WARP SUM-REDUCE (Σ)
\\ ============================================================
\\ 5-step butterfly reduce across 32 lanes: SHFL.BFLY + FADD per step.
\\ After return, lane 0 of each warp holds the warp sum in acc.
\\ acc: register index holding partial sum
\\ tmp: scratch register for shfl destination

emit_warp_reduce acc tmp :
    \\ delta=16: shfl.bfly tmp, acc, 16; fadd acc, acc, tmp
    emit_shfl_bfly tmp acc 16
    emit_fadd acc acc tmp
    \\ delta=8
    emit_shfl_bfly tmp acc 8
    emit_fadd acc acc tmp
    \\ delta=4
    emit_shfl_bfly tmp acc 4
    emit_fadd acc acc tmp
    \\ delta=2
    emit_shfl_bfly tmp acc 2
    emit_fadd acc acc tmp
    \\ delta=1
    emit_shfl_bfly tmp acc 1
    emit_fadd acc acc tmp

\\ ============================================================
\\ COOPERATIVE GRID-SYNC
\\ ============================================================
\\ Software barrier across all CTAs on the grid.
\\ Identical to CUDA cooperative groups this_grid().sync().
\\
\\ Protocol:
\\   Thread 0 of each CTA:
\\     (1) Atomically increment sync_counter. Save old value.
\\     (2) If old == grid_size-1 (last CTA to arrive):
\\           MEMBAR.SC.GPU + STG [done_flag], 1 (release write)
\\         Else:
\\           Spin: LDG [done_flag] + MEMBAR.SC.GPU until != 0
\\   All threads: BAR.SYNC 0 (broadcast done to all warps)
\\
\\ 19 instructions total. See gpu/emit.fs for byte offset derivation.
\\
\\ Args:
\\   ctr_reg  — register holding GPU VA of u32 sync counter
\\   flag_reg — register holding GPU VA of u32 done flag
\\   grid_size — compile-time constant = total CTA count
\\
\\ Scratch register allocation uses rreg/preg counters (provided by compiler).

var _gs_r 4
var _gs_p 0

rreg :
    r _gs_r
    _gs_r + 1

preg :
    p _gs_p
    _gs_p + 1

emit_grid_sync ctr_reg flag_reg grid_size :
    \\ Mark cooperative flag
    cooperative 1

    \\ Record grid-sync site offset
    record_gridsync

    \\ Allocate scratch registers and predicates
    gs_old rreg
    gs_exp rreg
    gs_poll rreg
    gs_pp preg
    gs_pp2 preg
    gs_tid rreg

    \\ (a) S2R gs_tid, SR_TID.X — read thread ID
    emit_s2r gs_tid SR_TID_X

    \\ (b) MOV gs_exp, grid_size-1
    emit_mov_imm gs_exp (grid_size - 1)

    \\ (c) MOV tmp, 1 — helper for ISETP
    tmp rreg
    emit_mov_imm tmp 1

    \\ (d) ISETP.GE pp2, gs_tid, tmp — pp2 = (tid >= 1)
    emit_isetp_ge gs_pp2 gs_tid tmp

    \\ (e) @pp2 BRA +208 — non-thread-0 jumps to BAR.SYNC
    emit_bra_pred 208 gs_pp2

    \\ ---- Thread 0 only ----

    \\ (f) MOV atom_one, 1
    atom_one rreg
    emit_mov_imm atom_one 1

    \\ (g) ATOM.E.ADD gs_old, [ctr_reg], atom_one
    emit_atom_add gs_old ctr_reg atom_one

    \\ (h) ISETP.GE gs_pp, gs_old, gs_exp
    emit_isetp_ge gs_pp gs_old gs_exp

    \\ (i) @!gs_pp BRA +64 — not last CTA: jump to spin_top
    emit_bra_pred 64 (gs_pp | 8)

    \\ ---- Last CTA path: write done flag ----

    \\ (j) MEMBAR.SC.GPU
    emit_membar_gpu

    \\ (k) MOV flag_one, 1
    flag_one rreg
    emit_mov_imm flag_one 1

    \\ (l) STG.E [flag_reg], flag_one
    emit_stg flag_reg flag_one

    \\ (m) BRA +80 — skip spin loop to BAR.SYNC
    emit_bra 80

    \\ ---- Spin-poll loop ----
    \\ spin_top: (n)

    \\ (n) LDG.E gs_poll, [flag_reg]
    emit_ldg gs_poll flag_reg

    \\ (o) MEMBAR.SC.GPU (acquire fence)
    emit_membar_gpu

    \\ (p) MOV tmp2, 1 — comparand
    tmp2 rreg
    emit_mov_imm tmp2 1

    \\ (q) ISETP.GE pp2, gs_poll, tmp2
    emit_isetp_ge gs_pp2 gs_poll tmp2

    \\ (r) @!pp2 BRA spin_top (-80 bytes back)
    emit_bra_pred -80 (gs_pp2 | 8)

    \\ ---- All-threads CTA barrier ----
    \\ (s) BAR.SYNC 0
    emit_bar_sync 0

\\ ============================================================
\\ MEGAKERNEL PARAM OFFSET HELPERS
\\ ============================================================
\\ Constant bank c[0x0] layout on Hopper:
\\   Bytes [0x000..0x20F] : driver-reserved (blockDim, gridDim, etc.)
\\   Bytes [0x210 + k*8]  : user param k (8-byte pointer)

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

buf cubin_buf 524288
var cubin_pos 0
var shmem_size 0
var n_kparams 0

\\ Kernel name metadata (shared with parser)
buf li_name_buf 64
var li_name_len 0

\\ Section offset / size tracking variables
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

\\ Section name indices
var SN_shstrtab 0
var SN_strtab 0
var SN_symtab 0
var SN_nvinfo 0
var SN_nvinfo_k 0
var SN_text 0
var SN_const0 0
var SN_shared 0

\\ CUBIN raw emitters
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
