"""
SM90 (Hopper) GPU code generation backend.

Emits raw 128-bit SASS instructions for sm_90a (GH200).
Each instruction is 16 bytes: 8-byte instruction word (lower) + 8-byte control word (upper).

Faithfully ported from compiler/emit-gpu.ls.
"""

import struct


# ============================================================
# OPCODE CONSTANTS
# ============================================================

# FP32 arithmetic
OP_FMUL      = 0x7220
OP_FADD      = 0x7221
OP_FFMA      = 0x7223
OP_FMNMX     = 0x7209
OP_FSETP     = 0x720B
OP_FADD_IMM  = 0x7421
OP_FMUL_IMM  = 0x7820

# Integer arithmetic
OP_IMAD      = 0x7224
OP_IMAD_WIDE = 0x7225
OP_IADD3     = 0x7210
OP_IADD3_IMM = 0x7810
OP_SHF       = 0x7819

# Bit operations
OP_LOP3      = 0x7212
OP_LOP3_IMM  = 0x7812
OP_ISETP     = 0x720C

# Conversion
OP_I2FP      = 0x7245
OP_F2I       = 0x7305

# SFU (MUFU)
OP_MUFU      = 0x7308

# Memory
OP_LDG       = 0x7981
OP_STG       = 0x7986
OP_LDS       = 0x7984
OP_STS       = 0x7988
OP_STS_REG   = 0x7388
OP_LDC       = 0x7B82
OP_ULDC      = 0x7AB9

# Atomic / reduction
OP_ATOMG_U32 = 0x79A8
OP_ATOMG_F32 = 0x79A3
OP_REDG_F32  = 0x79A6

# Warp / control flow
OP_SHFL        = 0x7F89
OP_SHFL_IDX    = 0x7589
OP_SHFL_IDX_RR = 0x7389
OP_BAR_SYNC    = 0x7B1D
OP_MEMBAR      = 0x7992
OP_BRA         = 0x7947
OP_EXIT        = 0x794D
OP_S2R         = 0x7919
OP_S2UR        = 0x79C3
OP_NOP         = 0x7918
OP_MOV_IMM     = 0x7802
OP_HFMA2       = 0x7235

# ============================================================
# MUFU SUBOP CONSTANTS (ctrl extra41 bits[13:8])
# ============================================================

MUFU_COS  = 0x0000
MUFU_SIN  = 0x0400
MUFU_EX2  = 0x0800
MUFU_LG2  = 0x0C00
MUFU_RCP  = 0x1000
MUFU_RSQ  = 0x1400
MUFU_SQRT = 0x2000

# ============================================================
# LOP3 TRUTH TABLE CONSTANTS (ctrl extra41 bits[15:8])
# ============================================================

LOP3_AND = 0xC0
LOP3_OR  = 0xFC
LOP3_XOR = 0x3C

# ============================================================
# SPECIAL REGISTER IDs (S2R ctrl extra41[15:8])
# ============================================================

SR_TID_X   = 0x21
SR_TID_Y   = 0x22
SR_TID_Z   = 0x23
SR_CTAID_X = 0x25
SR_CTAID_Y = 0x26
SR_CTAID_Z = 0x27
SR_LANEID  = 0x00

# ============================================================
# REGISTER ENCODING CONSTANTS
# ============================================================

RZ = 0xFF   # zero register (reads 0, writes discarded)
PT = 0x07   # always-true predicate

# ============================================================
# PRE-COMPUTED CONTROL WORD CONSTANTS (verified literals)
# ============================================================

CTRL_NOP          = 0x000FC00000000000
CTRL_LDG          = 0x000EA8000C1E1900
CTRL_LDG_WAIT0    = 0x001EA8000C1E1900
CTRL_STG          = 0x000FE2000C101904
CTRL_STG_WAIT2    = 0x001FE8000C101904
CTRL_LDS          = 0x000E280008000800
CTRL_STS          = 0x000FE80008000804
CTRL_LDC          = 0x000E220000000800
CTRL_ULDC         = 0x000FE20000000A00
CTRL_FADD         = 0x000FCA0000000000
CTRL_FADD_WAIT2   = 0x004FCA0000000000
CTRL_FMUL         = 0x004FCA0000400000
CTRL_IMAD_IMM     = 0x000FE200078E00FF
CTRL_ISETP        = 0x000FDA000BF06070
CTRL_I2FP_S32     = 0x004FCA0000201400
CTRL_I2FP_U32     = 0x004FCA0000201000
CTRL_F2I          = 0x004E24000020F100
CTRL_SHFL         = 0x001E6800000E0000
CTRL_BAR          = 0x000FEC0000010000
CTRL_MEMBAR_GPU   = 0x000FEC0000002000
CTRL_MEMBAR_SYS   = 0x000FEC0000003000
CTRL_MEMBAR_CTA   = 0x0003EC0000000000
CTRL_BRA          = 0x000FC0000383FFFF
CTRL_EXIT         = 0x000FEA0003800000
CTRL_ATOMG_U32    = 0x004E2800081EE1C4
CTRL_ATOMG_F32    = 0x004E2800081EF3C4
CTRL_ATOMG_EXCH   = 0x004E28000C1EE1C4
CTRL_REDG_F32     = 0x004FE2000C10F384

# Mask for 64-bit values
_MASK64 = (1 << 64) - 1


def make_ctrl(stall: int, yield_flag: int, wbar: int, rbar: int,
              wait: int, reuse: int, extra41: int) -> int:
    """
    Build a 64-bit control word from scheduling fields.

    Layout (bit positions within the 64-bit control word):
      extra41  [40:0]   - opaque barrier/descriptor fields (41 bits)
      stall    [44:41]  - cycles to stall (0-15, 4 bits)
      yield    [45]     - scheduler yield hint (1 bit)
      wbar     [48:46]  - write barrier slot, 7=none (3 bits)
      rbar     [51:49]  - read barrier slot, 7=none (3 bits)
      wait     [57:52]  - wait barrier mask (6 bits)
      reuse    [62:58]  - reuse cache flags (5 bits)
    """
    return (
        (extra41 & ((1 << 41) - 1))
        | ((stall & 0xF) << 41)
        | ((yield_flag & 0x1) << 45)
        | ((wbar & 0x7) << 46)
        | ((rbar & 0x7) << 49)
        | ((wait & 0x3F) << 52)
        | ((reuse & 0x1F) << 58)
    )


# ============================================================
# PER-INSTRUCTION CONTROL WORD CONSTRUCTORS
# ============================================================

def _ctrl_nop() -> int:
    return make_ctrl(0, 0, 7, 7, 0, 0, 0)

def _ctrl_s2r(sr_id: int) -> int:
    return make_ctrl(7, 1, 1, 7, 0, 0, sr_id << 8)

def _ctrl_ldg() -> int:
    return make_ctrl(4, 1, 2, 7, 0, 0, 0x0C1E1900)

def _ctrl_ldg_wait0() -> int:
    return make_ctrl(4, 1, 2, 7, 1, 0, 0x0C1E1900)

def _ctrl_stg() -> int:
    return make_ctrl(1, 1, 7, 7, 0, 0, 0x0C101904)

def _ctrl_stg_wait2() -> int:
    return make_ctrl(4, 1, 7, 7, 1, 0, 0x0C101904)

def _ctrl_ffma(rs3: int) -> int:
    return make_ctrl(5, 0, 7, 7, 0, 0, rs3)

def _ctrl_fadd() -> int:
    return make_ctrl(5, 0, 7, 7, 0, 0, 0)

def _ctrl_fadd_wait2() -> int:
    return make_ctrl(5, 0, 7, 7, 4, 0, 0)

def _ctrl_fmul() -> int:
    return make_ctrl(5, 0, 7, 7, 4, 0, 0x400000)

def _ctrl_imad(rs3: int) -> int:
    return make_ctrl(1, 1, 7, 7, 0, 0, 0x0F8E0200 | rs3)

def _ctrl_imad_imm() -> int:
    return make_ctrl(1, 1, 7, 7, 0, 0, 0x078E00FF)

def _ctrl_isetp() -> int:
    return make_ctrl(13, 0, 7, 7, 0, 0, 0x0BF06070)

def _ctrl_bra() -> int:
    return make_ctrl(0, 0, 7, 7, 0, 0, 0x0383FFFF)

def _ctrl_exit() -> int:
    return make_ctrl(5, 1, 7, 7, 0, 0, 0x03800000)

def _ctrl_uldc() -> int:
    return make_ctrl(1, 1, 7, 7, 0, 0, 0x0A00)

def _ctrl_ldc() -> int:
    return make_ctrl(7, 1, 1, 7, 0, 0, 0x0800)

def _ctrl_shfl() -> int:
    return make_ctrl(14, 0, 3, 7, 0, 0, 0x000E0000)

def _ctrl_sts() -> int:
    return make_ctrl(1, 1, 7, 7, 0, 0, 0x0800)

def _ctrl_lds() -> int:
    return make_ctrl(7, 1, 1, 7, 0, 0, 0x0800)

def _ctrl_bar() -> int:
    return make_ctrl(5, 1, 7, 7, 0, 0, 0x00010000)

def _ctrl_shf() -> int:
    return make_ctrl(1, 0, 7, 7, 0, 0, 0)

def _ctrl_i2f() -> int:
    return make_ctrl(5, 0, 7, 7, 4, 0, 0x00201400)

def _ctrl_f2i() -> int:
    return make_ctrl(2, 1, 0, 7, 4, 0, 0x0020F100)

def _ctrl_lop3(lut: int) -> int:
    return make_ctrl(1, 0, 7, 7, 0, 0, lut << 8)

def _ctrl_mufu(subfn: int, wbar: int) -> int:
    return make_ctrl(8, 1, wbar, 7, 0, 0, subfn)


class SM90Emitter:
    """
    SM90 (Hopper) SASS binary emitter.

    Emits raw 128-bit instructions into an internal byte buffer.
    Each instruction: lower 64 bits = instruction word, upper 64 bits = control word.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._max_reg = 0

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _track_rd(self, rd: int) -> int:
        """Update register high-water mark and return rd unchanged."""
        if rd != RZ and rd > self._max_reg:
            self._max_reg = rd
        return rd

    def _sinst(self, iword: int, ctrl: int) -> None:
        """Emit one 128-bit instruction (inst word + ctrl word), little-endian."""
        self._buf += struct.pack('<QQ', iword & _MASK64, ctrl & _MASK64)

    def _pos(self) -> int:
        """Current byte offset in the code buffer."""
        return len(self._buf)

    # ----------------------------------------------------------
    # Public query
    # ----------------------------------------------------------

    def get_code(self) -> bytes:
        """Return the emitted SASS binary as bytes."""
        return bytes(self._buf)

    def get_register_count(self) -> int:
        """
        Return register count for the SPD header.
        This is max_reg + 1 (registers are 0-indexed), rounded up to
        a multiple of 8 as required by Hopper, minimum 8.
        """
        count = self._max_reg + 1
        count = ((count + 7) // 8) * 8
        return max(count, 8)

    def reset(self) -> None:
        """Reset emitter state for a new kernel."""
        self._buf = bytearray()
        self._max_reg = 0

    # ===========================================================
    # NOP
    # ===========================================================

    def emit_nop(self) -> None:
        """NOP -- no operation."""
        self._sinst(0x0000000000007918, _ctrl_nop())

    # ===========================================================
    # FP32 ARITHMETIC
    # ===========================================================

    def emit_fadd(self, rd: int, ra: int, rb: int) -> None:
        """FADD Rd, Ra, Rb -- float32 add."""
        iword = OP_FADD | (self._track_rd(rd) << 16) | (ra << 24) | (rb << 32)
        self._sinst(iword, _ctrl_fadd())

    def emit_ffma(self, rd: int, rs1: int, rs2: int, rs3: int) -> None:
        """FFMA Rd, Rs1, Rs2, Rs3 -- float fused multiply-add.
        Rs3 (accumulator) packed into ctrl extra41[7:0]."""
        iword = OP_FFMA | (self._track_rd(rd) << 16) | (rs1 << 24) | (rs2 << 32)
        self._sinst(iword, _ctrl_ffma(rs3))

    def emit_fmul(self, rd: int, rs1: int, rs2: int) -> None:
        """FMUL Rd, Rs1, Rs2 -- float multiply (FFMA with RZ accumulator)."""
        self.emit_ffma(rd, rs1, rs2, RZ)

    def emit_fmnmx(self, rd: int, ra: int, rb: int) -> None:
        """FMNMX Rd, Ra, Rb -- float min/max (predicate selects min vs max)."""
        iword = OP_FMNMX | (self._track_rd(rd) << 16) | (ra << 24) | (rb << 32)
        self._sinst(iword, _ctrl_fadd())

    def emit_fsetp(self, pd: int, ra: int, rb: int) -> None:
        """FSETP Pd, Ra, Rb -- float set predicate."""
        iword = OP_FSETP | (pd << 16) | (ra << 24) | (rb << 32)
        self._sinst(iword, _ctrl_isetp())

    def emit_fadd_imm(self, rd: int, ra: int, imm32: int) -> None:
        """FADD.IMM Rd, Ra, imm32 -- float add with immediate."""
        iword = OP_FADD_IMM | (self._track_rd(rd) << 16) | (ra << 24) | (imm32 << 32)
        self._sinst(iword, _ctrl_fadd())

    def emit_fmul_imm(self, rd: int, ra: int, imm32: int) -> None:
        """FMUL.IMM Rd, Ra, imm32 -- float multiply with immediate."""
        iword = OP_FMUL_IMM | (self._track_rd(rd) << 16) | (ra << 24) | (imm32 << 32)
        self._sinst(iword, _ctrl_fmul())

    # ===========================================================
    # HFMA2 (packed FP16x2)
    # ===========================================================

    def emit_hfma2(self, rd: int, rs1: int, rs2: int, rs3: int) -> None:
        """HFMA2.MMA Rd, Rs1, Rs2, Rs3 -- packed FP16x2 fused multiply-add."""
        iword = OP_HFMA2 | (self._track_rd(rd) << 16) | (rs1 << 24) | (rs2 << 32)
        self._sinst(iword, _ctrl_ffma(rs3))

    # ===========================================================
    # INTEGER ARITHMETIC
    # ===========================================================

    def emit_imad(self, rd: int, rs1: int, rs2: int, rs3: int) -> None:
        """IMAD Rd, Rs1, Rs2, Rs3 -- integer multiply-add (reg*reg+reg).
        Rs3 (accumulator) in ctrl[7:0]."""
        iword = OP_IMAD | (self._track_rd(rd) << 16) | (rs1 << 24) | (rs2 << 32)
        self._sinst(iword, _ctrl_imad(rs3))

    def emit_iadd3(self, rd: int, rs1: int, rs2: int, rs3: int) -> None:
        """IADD3 Rd, Rs1, Rs2, Rs3 -- 3-input integer add.
        Rs3 in ctrl[7:0] (4-operand format).
        Note: uses opcode 0x7C10 per emit-gpu.ls."""
        iword = 0x7C10 | (self._track_rd(rd) << 16) | (rs1 << 24) | (rs2 << 32)
        ctrl = make_ctrl(1, 1, 7, 7, 0, 0, rs3)
        self._sinst(iword, ctrl)

    # ===========================================================
    # MEMORY
    # ===========================================================

    def emit_ldg(self, rd: int, ra: int) -> None:
        """LDG.E Rd, [Ra] -- 32-bit global load."""
        iword = OP_LDG | (self._track_rd(rd) << 16) | (ra << 24)
        self._sinst(iword, _ctrl_ldg())

    def emit_ldg_off(self, rd: int, ra: int, off: int) -> None:
        """LDG.E Rd, [Ra+off] -- 32-bit global load with byte offset."""
        iword = OP_LDG | (self._track_rd(rd) << 16) | (ra << 24) | (off << 32)
        self._sinst(iword, _ctrl_ldg())

    def emit_stg(self, ra: int, rs: int) -> None:
        """STG.E [Ra], Rs -- 32-bit global store."""
        iword = OP_STG | (ra << 24) | (rs << 32)
        self._sinst(iword, _ctrl_stg())

    def emit_stg_off(self, ra: int, rs: int, off: int) -> None:
        """STG.E [Ra+off], Rs -- 32-bit global store with byte offset."""
        iword = OP_STG | (ra << 24) | (rs << 32) | (off << 32)
        self._sinst(iword, _ctrl_stg())

    def emit_lds(self, rd: int, ra: int, off: int) -> None:
        """LDS.32 Rd, [Ra+off] -- load 32 bits from shared memory."""
        iword = OP_LDS | (self._track_rd(rd) << 16) | (ra << 24) | (off << 32)
        self._sinst(iword, _ctrl_lds())

    def emit_sts(self, ra: int, rs: int, off: int) -> None:
        """STS.32 [Ra+off], Rs -- store 32 bits to shared memory."""
        iword = OP_STS | (ra << 24) | (rs << 32) | (off << 32)
        self._sinst(iword, _ctrl_sts())

    def emit_ldc(self, rd: int, cbuf: int, offset: int) -> None:
        """LDC Rd, c[cbuf][offset] -- load from constant bank to GPR."""
        iword = (OP_LDC | (self._track_rd(rd) << 16) | (RZ << 24)
                 | (offset << 32) | (cbuf << 48))
        self._sinst(iword, CTRL_LDC)

    def emit_uldc(self, rd: int, cbuf: int, offset: int) -> None:
        """ULDC URn, c[cbuf][offset] -- uniform load from constant bank."""
        iword = (OP_ULDC | (self._track_rd(rd) << 16) | (RZ << 24)
                 | (offset << 32) | (cbuf << 48))
        self._sinst(iword, CTRL_ULDC)

    # ===========================================================
    # ATOMICS / REDUCTIONS
    # ===========================================================

    def emit_atom_add(self, rd: int, ra: int, rs: int) -> None:
        """ATOMG.E.ADD.STRONG.GPU Rd, [Ra], Rs -- 32-bit integer atomic add."""
        iword = OP_ATOMG_U32 | (self._track_rd(rd) << 16) | (ra << 24) | (rs << 32)
        self._sinst(iword, CTRL_ATOMG_U32)

    def emit_atom_add_f32(self, rd: int, ra: int, rs: int) -> None:
        """ATOMG.E.ADD.F32 Rd, [Ra], Rs -- float32 atomic add."""
        iword = OP_ATOMG_F32 | (self._track_rd(rd) << 16) | (ra << 24) | (rs << 32)
        self._sinst(iword, CTRL_ATOMG_F32)

    def emit_red_add_f32(self, ra: int, rs: int) -> None:
        """REDG.E.ADD.F32 [Ra], Rs -- float32 reduction (no return value)."""
        iword = OP_REDG_F32 | (RZ << 16) | (ra << 24) | (rs << 32)
        self._sinst(iword, CTRL_REDG_F32)

    # ===========================================================
    # WARP SHUFFLES
    # ===========================================================

    def emit_shfl_bfly(self, rd: int, rs: int, delta: int) -> None:
        """SHFL.BFLY PT, Rd, Rs, delta -- butterfly shuffle (intra-warp reduce).

        Encoding:
          inst[15:0]  = 0x7F89 (opcode)
          inst[23:16] = Rd
          inst[31:24] = Rs
          inst[47:40] = 0x1F (mask=31)
          inst[55:48] = clamp = (delta*32) & 0xFF
          inst[63:56] = mode  = 0x0C | ((delta*32) >> 8)
        """
        delta_enc = delta * 32
        clamp = delta_enc & 0xFF
        mode = 0x0C | (delta_enc >> 8)
        iword = (OP_SHFL | (self._track_rd(rd) << 16) | (rs << 24)
                 | (0x1F << 40) | (clamp << 48) | (mode << 56))
        self._sinst(iword, _ctrl_shfl())

    def emit_shfl_idx(self, rd: int, rs: int, r_lane: int) -> None:
        """SHFL.IDX PT, Rd, Rs, R_lane -- indexed shuffle (register lane index).
        Full register form: opcode 0x7389, R_lane at bits[39:32]."""
        iword = (OP_SHFL_IDX_RR | (self._track_rd(rd) << 16)
                 | (rs << 24) | (r_lane << 32))
        self._sinst(iword, _ctrl_shfl())

    # ===========================================================
    # BARRIERS / SYNC
    # ===========================================================

    def emit_bar_sync(self, bar_id: int) -> None:
        """BAR.SYNC.DEFER_BLOCKING bar_id -- synchronize threads at barrier.
        bar_id encoded as bar_id << 54 in instruction word."""
        iword = OP_BAR_SYNC | (bar_id << 54)
        self._sinst(iword, _ctrl_bar())

    def emit_membar(self, scope: str = "gpu") -> None:
        """MEMBAR.SC -- memory barrier.

        Args:
            scope: "gpu", "sys", or "cta"
        """
        base = 0x0000000000007992
        if scope == "gpu":
            self._sinst(base, CTRL_MEMBAR_GPU)
        elif scope == "sys":
            self._sinst(base, CTRL_MEMBAR_SYS)
        elif scope == "cta":
            self._sinst(base, CTRL_MEMBAR_CTA)
        else:
            raise ValueError(f"Unknown membar scope: {scope!r}")

    # ===========================================================
    # BRANCH / CONTROL FLOW
    # ===========================================================

    def emit_bra(self, byte_offset: int) -> None:
        """BRA offset -- unconditional branch.
        Target = (bra_addr + 16) + offset32 * 4."""
        offset32 = byte_offset // 4
        iword = 0x00FC7947 | (offset32 << 32)
        self._sinst(iword, _ctrl_bra())

    def emit_bra_pred(self, byte_offset: int, pred: int) -> None:
        """@Px BRA offset -- predicated branch.
        pred = 0..3 for P0..P3; pred|8 for negated (@!Px)."""
        offset32 = byte_offset // 4
        iword = 0x00FC7947 | (offset32 << 32)
        # Clear predicate field bits[15:12] and insert pred
        iword = (iword & ~(0xF << 12)) | (pred << 12)
        self._sinst(iword, _ctrl_bra())

    def emit_exit(self) -> None:
        """EXIT -- terminate thread."""
        self._sinst(0x000000000000794D, _ctrl_exit())

    # ===========================================================
    # BRANCH PATCHING HELPERS
    # ===========================================================

    def mark(self) -> int:
        """Save current position for forward-branch patching."""
        return self._pos()

    def patch(self, saved_pos: int) -> None:
        """Patch a BRA at saved_pos with offset to current position."""
        byte_offset = self._pos() - saved_pos - 16
        offset32 = byte_offset // 4
        # Patch bits[63:32] of the instruction word at saved_pos
        # That's bytes [saved_pos+4 .. saved_pos+7]
        struct.pack_into('<i', self._buf, saved_pos + 4, offset32)

    # ===========================================================
    # SPECIAL REGISTER
    # ===========================================================

    def emit_s2r(self, rd: int, sr_id: int) -> None:
        """S2R Rd, sr_id -- read special register into Rd.
        sr_id encoded ONLY in ctrl extra41[15:8]."""
        iword = OP_S2R | (self._track_rd(rd) << 16)
        self._sinst(iword, _ctrl_s2r(sr_id))

    def emit_mov_imm(self, rd: int, imm32: int) -> None:
        """MOV Rd, imm32 -- move 32-bit immediate into register."""
        iword = OP_MOV_IMM | (self._track_rd(rd) << 16) | (imm32 << 32)
        self._sinst(iword, _ctrl_nop())

    # ===========================================================
    # CONVERSION
    # ===========================================================

    def emit_i2f(self, rd: int, rs: int) -> None:
        """I2FP.F32.S32 Rd, Rs -- signed int32 to float32.
        Note: source at bits[39:32], NOT bits[31:24]."""
        iword = OP_I2FP | (self._track_rd(rd) << 16) | (rs << 32)
        self._sinst(iword, _ctrl_i2f())

    def emit_f2i(self, rd: int, rs: int) -> None:
        """F2I.TRUNC.NTZ Rd, Rs -- float32 to signed int32 (truncate toward zero)."""
        iword = OP_F2I | (self._track_rd(rd) << 16) | (rs << 32)
        self._sinst(iword, _ctrl_f2i())

    # ===========================================================
    # COMPOSITE: WARP SUM-REDUCE
    # ===========================================================

    def emit_warp_reduce(self, acc: int, tmp: int) -> None:
        """5-step butterfly reduce across 32 lanes.
        After return, lane 0 holds the warp sum in acc.

        Args:
            acc: register index holding partial sum
            tmp: scratch register for shfl destination
        """
        for delta in (16, 8, 4, 2, 1):
            self.emit_shfl_bfly(tmp, acc, delta)
            self.emit_fadd(acc, acc, tmp)

    # ===========================================================
    # COMPOSITE: GRID SYNC
    # ===========================================================

    def emit_grid_sync(self, ctr_reg: int, flag_reg: int, grid_size: int) -> None:
        """Cooperative grid-wide synchronization.

        Software barrier across all CTAs on the grid.
        Identical to CUDA cooperative groups this_grid().sync().

        Protocol:
          Thread 0 of each CTA:
            (1) Atomically increment sync_counter. Save old value.
            (2) If old == grid_size-1 (last CTA to arrive):
                  MEMBAR.SC.GPU + STG [done_flag], 1
                Else:
                  Spin: LDG [done_flag] + MEMBAR.SC.GPU until != 0
          All threads: BAR.SYNC 0

        Args:
            ctr_reg: register holding GPU VA of u32 sync counter
            flag_reg: register holding GPU VA of u32 done flag
            grid_size: compile-time constant = total CTA count
        """
        # Allocate scratch registers (starting from R4 per emit-gpu.ls)
        _r = [4]

        def rreg() -> int:
            r = _r[0]
            _r[0] += 1
            return r

        _p = [0]

        def preg() -> int:
            p = _p[0]
            _p[0] += 1
            return p

        gs_old = rreg()
        gs_exp = rreg()
        gs_poll = rreg()
        gs_pp = preg()
        gs_pp2 = preg()
        gs_tid = rreg()

        # (a) S2R gs_tid, SR_TID.X
        self.emit_s2r(gs_tid, SR_TID_X)

        # (b) MOV gs_exp, grid_size-1
        self.emit_mov_imm(gs_exp, grid_size - 1)

        # (c) MOV tmp, 1
        tmp = rreg()
        self.emit_mov_imm(tmp, 1)

        # (d) ISETP.GE pp2, gs_tid, tmp
        self._emit_isetp_ge(gs_pp2, gs_tid, tmp)

        # (e) @pp2 BRA +208 -- non-thread-0 jumps to BAR.SYNC
        self.emit_bra_pred(208, gs_pp2)

        # ---- Thread 0 only ----

        # (f) MOV atom_one, 1
        atom_one = rreg()
        self.emit_mov_imm(atom_one, 1)

        # (g) ATOM.E.ADD gs_old, [ctr_reg], atom_one
        self.emit_atom_add(gs_old, ctr_reg, atom_one)

        # (h) ISETP.GE gs_pp, gs_old, gs_exp
        self._emit_isetp_ge(gs_pp, gs_old, gs_exp)

        # (i) @!gs_pp BRA +64 -- not last CTA: jump to spin_top
        self.emit_bra_pred(64, gs_pp | 8)

        # ---- Last CTA path: write done flag ----

        # (j) MEMBAR.SC.GPU
        self.emit_membar("gpu")

        # (k) MOV flag_one, 1
        flag_one = rreg()
        self.emit_mov_imm(flag_one, 1)

        # (l) STG.E [flag_reg], flag_one
        self.emit_stg(flag_reg, flag_one)

        # (m) BRA +80 -- skip spin loop to BAR.SYNC
        self.emit_bra(80)

        # ---- Spin-poll loop ----

        # (n) LDG.E gs_poll, [flag_reg]
        self.emit_ldg(gs_poll, flag_reg)

        # (o) MEMBAR.SC.GPU (acquire fence)
        self.emit_membar("gpu")

        # (p) MOV tmp2, 1
        tmp2 = rreg()
        self.emit_mov_imm(tmp2, 1)

        # (q) ISETP.GE pp2, gs_poll, tmp2
        self._emit_isetp_ge(gs_pp2, gs_poll, tmp2)

        # (r) @!pp2 BRA spin_top (-80 bytes back)
        self.emit_bra_pred(-80, gs_pp2 | 8)

        # ---- All-threads CTA barrier ----
        # (s) BAR.SYNC 0
        self.emit_bar_sync(0)

    def _emit_isetp_ge(self, pd: int, rs1: int, rs2: int) -> None:
        """ISETP.GE.U32.AND Pd, Rs1, Rs2 -- integer >= comparison to predicate."""
        iword = OP_ISETP | (pd << 16) | (rs1 << 24) | (rs2 << 32)
        self._sinst(iword, _ctrl_isetp())

    # ===========================================================
    # COMPOSITE: GPTQ DEQUANTIZATION
    # ===========================================================

    def emit_dequant_nibble(self, rd: int, rsrc: int, nib_idx: int,
                            rscale: int, rzero: int) -> None:
        """GPTQ nibble dequantization.

        Unpacks one 4-bit nibble from a packed u32 and converts to float:
          1. SHF.R.U32  rd, rsrc, nib_idx*4   -- shift right
          2. LOP3.AND   rd, rd, 0xF            -- mask low 4 bits
          3. I2FP       rd, rd                 -- int32 to float32
          4. FFMA       rd, rd, rscale, rzero  -- scale and offset

        Args:
            rd: destination register
            rsrc: source register with packed nibbles
            nib_idx: nibble index (0-7)
            rscale: register holding scale factor
            rzero: register holding zero point
        """
        shamt = nib_idx * 4
        self._emit_shf_r(rd, rsrc, shamt)
        self._emit_lop3_and(rd, rd, 0xF)
        self.emit_i2f(rd, rd)
        self.emit_ffma(rd, rd, rscale, rzero)

    def _emit_shf_r(self, rd: int, rs: int, shamt: int) -> None:
        """SHF.R.U32 Rd, Rs, shamt -- shift right (funnel source=RZ)."""
        iword = (OP_SHF | (self._track_rd(rd) << 16) | (rs << 24)
                 | (shamt << 32) | (RZ << 48))
        self._sinst(iword, _ctrl_shf())

    def _emit_lop3_and(self, rd: int, rs: int, imm32: int) -> None:
        """LOP3.LUT Rd, Rs, imm32, RZ -- AND with immediate."""
        iword = OP_LOP3_IMM | (self._track_rd(rd) << 16) | (rs << 24) | (imm32 << 32)
        self._sinst(iword, _ctrl_lop3(LOP3_AND))
