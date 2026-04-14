"""
ARM64 machine code emitter for the Lithos Python bootstrapper.

Emits raw ARM64 (AArch64) instructions into an in-memory byte buffer.
Each method appends exactly one 32-bit little-endian instruction word.

Encoding reference: ARM Architecture Reference Manual (ARMv8-A)
Design reference:   compiler/compiler.ls  Section 1 (ARM64 backend)
                    bootstrap/emit-arm64.s
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Register constants
# ---------------------------------------------------------------------------

X0, X1, X2, X3, X4, X5, X6, X7 = range(8)
X8, X9, X10, X11, X12, X13, X14, X15 = range(8, 16)
X16, X17, X18, X19, X20, X21, X22, X23 = range(16, 24)
X24, X25, X26, X27, X28, X29, X30 = range(24, 31)
XZR = 31
SP = 31  # context-dependent: SP in load/store, XZR elsewhere

FP = 29   # frame pointer
LR = 30   # link register

# ---------------------------------------------------------------------------
# Condition codes (4-bit, used by B.cond, CSEL, CSET, etc.)
# ---------------------------------------------------------------------------

COND_EQ = 0   # equal / zero
COND_NE = 1   # not equal / nonzero
COND_CS = 2   # carry set / unsigned >=  (alias HS)
COND_HS = 2
COND_CC = 3   # carry clear / unsigned <  (alias LO)
COND_LO = 3
COND_MI = 4   # minus / negative
COND_PL = 5   # plus / positive or zero
COND_VS = 6   # overflow
COND_VC = 7   # no overflow
COND_HI = 8   # unsigned >
COND_LS = 9   # unsigned <=
COND_GE = 10  # signed >=
COND_LT = 11  # signed <
COND_GT = 12  # signed >
COND_LE = 13  # signed <=
COND_AL = 14  # always

# ---------------------------------------------------------------------------
# Linux AArch64 syscall numbers
# ---------------------------------------------------------------------------

SYS_READ      = 63
SYS_WRITE     = 64
SYS_OPENAT    = 56
SYS_CLOSE     = 57
SYS_MMAP      = 222
SYS_MUNMAP    = 215
SYS_MPROTECT  = 226
SYS_EXIT      = 93
SYS_BRK       = 214


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _u32(value: int) -> int:
    """Mask to unsigned 32-bit."""
    return value & 0xFFFF_FFFF


def _signed_bits(value: int, bits: int) -> int:
    """Extract the low *bits* of a signed value, as unsigned."""
    mask = (1 << bits) - 1
    return value & mask


# ---------------------------------------------------------------------------
# ARM64Emitter
# ---------------------------------------------------------------------------

class ARM64Emitter:
    """
    Emits ARM64 machine code into an in-memory buffer.

    AAPCS64 calling convention notes (for reference by higher layers):
      - x0-x7   : argument / result registers
      - x8      : indirect result / syscall number on Linux
      - x9-x15  : caller-saved temporaries
      - x16-x17 : intra-procedure-call scratch (IP0/IP1)
      - x18     : platform register (reserved on some OSes)
      - x19-x28 : callee-saved
      - x29     : frame pointer (FP)
      - x30     : link register (LR)
      - SP      : stack pointer (16-byte aligned at public interfaces)
    """

    def __init__(self, capacity: int = 1 << 20) -> None:
        self._buf = bytearray(capacity)
        self._pos = 0
        # Forward-reference patches: list of (buf_offset, patch_kind)
        self._patches: List[Tuple[int, str]] = []
        # Label support: name -> buf_offset
        self._labels: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Buffer primitives
    # ------------------------------------------------------------------

    def _emit32(self, word: int) -> None:
        """Write one 32-bit LE instruction at the current position."""
        w = _u32(word)
        struct.pack_into("<I", self._buf, self._pos, w)
        self._pos += 4

    def pos(self) -> int:
        """Current emission offset (byte address within buffer)."""
        return self._pos

    def get_code(self) -> bytes:
        """Return the emitted code as an immutable *bytes* object."""
        return bytes(self._buf[: self._pos])

    def reset(self) -> None:
        """Reset the emission cursor to the beginning."""
        self._pos = 0

    def patch32(self, offset: int, word: int) -> None:
        """Overwrite the 32-bit instruction at *offset*."""
        struct.pack_into("<I", self._buf, offset, _u32(word))

    def read32(self, offset: int) -> int:
        """Read the 32-bit instruction previously written at *offset*."""
        return struct.unpack_from("<I", self._buf, offset)[0]

    # ------------------------------------------------------------------
    # Labels and forward-branch patching
    # ------------------------------------------------------------------

    def mark(self) -> int:
        """Return the current position (for later patching)."""
        return self._pos

    def define_label(self, name: str) -> None:
        self._labels[name] = self._pos

    def label_offset(self, name: str) -> int:
        return self._labels[name]

    def patch_b(self, mark_pos: int, target: Optional[int] = None) -> None:
        """Patch a B instruction at *mark_pos* to branch to *target* (default: current pos)."""
        if target is None:
            target = self._pos
        offset = target - mark_pos
        imm26 = _signed_bits(offset >> 2, 26)
        old = self.read32(mark_pos)
        self.patch32(mark_pos, (old & 0xFC000000) | imm26)

    def patch_bl(self, mark_pos: int, target: Optional[int] = None) -> None:
        """Patch a BL instruction at *mark_pos*."""
        # Same bit layout as B for the imm26 field.
        self.patch_b(mark_pos, target)

    def patch_bcond(self, mark_pos: int, target: Optional[int] = None) -> None:
        """Patch a B.cond (or CBZ/CBNZ) instruction at *mark_pos*."""
        if target is None:
            target = self._pos
        offset = target - mark_pos
        imm19 = _signed_bits(offset >> 2, 19)
        old = self.read32(mark_pos)
        self.patch32(mark_pos, (old & ~(0x7FFFF << 5)) | (imm19 << 5))

    patch_cbz = patch_bcond   # CBZ/CBNZ share the imm19<<5 layout
    patch_cbnz = patch_bcond

    # ------------------------------------------------------------------
    # DATA PROCESSING -- REGISTER (64-bit)
    # ------------------------------------------------------------------

    def emit_add_reg(self, rd: int, rn: int, rm: int) -> None:
        """ADD Xd, Xn, Xm"""
        self._emit32(0x8B000000 | (rm << 16) | (rn << 5) | rd)

    def emit_add_reg_lsl(self, rd: int, rn: int, rm: int, amount: int) -> None:
        """ADD Xd, Xn, Xm, LSL #amount"""
        self._emit32(0x8B000000 | (rm << 16) | ((amount & 0x3F) << 10) | (rn << 5) | rd)

    def emit_sub_reg(self, rd: int, rn: int, rm: int) -> None:
        """SUB Xd, Xn, Xm"""
        self._emit32(0xCB000000 | (rm << 16) | (rn << 5) | rd)

    def emit_adds_reg(self, rd: int, rn: int, rm: int) -> None:
        """ADDS Xd, Xn, Xm"""
        self._emit32(0xAB000000 | (rm << 16) | (rn << 5) | rd)

    def emit_subs_reg(self, rd: int, rn: int, rm: int) -> None:
        """SUBS Xd, Xn, Xm"""
        self._emit32(0xEB000000 | (rm << 16) | (rn << 5) | rd)

    def emit_cmp_reg(self, rn: int, rm: int) -> None:
        """CMP Xn, Xm  (alias: SUBS XZR, Xn, Xm)"""
        self.emit_subs_reg(XZR, rn, rm)

    def emit_neg(self, rd: int, rm: int) -> None:
        """NEG Xd, Xm  (alias: SUB Xd, XZR, Xm)"""
        self.emit_sub_reg(rd, XZR, rm)

    def emit_mul(self, rd: int, rn: int, rm: int) -> None:
        """MUL Xd, Xn, Xm  (MADD Xd, Xn, Xm, XZR)"""
        self._emit32(0x9B007C00 | (rm << 16) | (rn << 5) | rd)

    def emit_madd(self, rd: int, rn: int, rm: int, ra: int) -> None:
        """MADD Xd, Xn, Xm, Xa"""
        self._emit32(0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd)

    def emit_msub(self, rd: int, rn: int, rm: int, ra: int) -> None:
        """MSUB Xd, Xn, Xm, Xa"""
        self._emit32(0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd)

    def emit_sdiv(self, rd: int, rn: int, rm: int) -> None:
        """SDIV Xd, Xn, Xm"""
        self._emit32(0x9AC00C00 | (rm << 16) | (rn << 5) | rd)

    def emit_udiv(self, rd: int, rn: int, rm: int) -> None:
        """UDIV Xd, Xn, Xm"""
        self._emit32(0x9AC00800 | (rm << 16) | (rn << 5) | rd)

    # ------------------------------------------------------------------
    # DATA PROCESSING -- IMMEDIATE (Arithmetic)
    # ------------------------------------------------------------------

    def emit_add_imm(self, rd: int, rn: int, imm12: int) -> None:
        """ADD Xd, Xn, #imm12"""
        self._emit32(0x91000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_add_imm_lsl12(self, rd: int, rn: int, imm12: int) -> None:
        """ADD Xd, Xn, #imm12, LSL #12"""
        self._emit32(0x91400000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_sub_imm(self, rd: int, rn: int, imm12: int) -> None:
        """SUB Xd, Xn, #imm12"""
        self._emit32(0xD1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_adds_imm(self, rd: int, rn: int, imm12: int) -> None:
        """ADDS Xd, Xn, #imm12"""
        self._emit32(0xB1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_subs_imm(self, rd: int, rn: int, imm12: int) -> None:
        """SUBS Xd, Xn, #imm12"""
        self._emit32(0xF1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_cmp_imm(self, rn: int, imm12: int) -> None:
        """CMP Xn, #imm12"""
        self.emit_subs_imm(XZR, rn, imm12)

    def emit_cmn_imm(self, rn: int, imm12: int) -> None:
        """CMN Xn, #imm12"""
        self.emit_adds_imm(XZR, rn, imm12)

    # ------------------------------------------------------------------
    # DATA PROCESSING -- REGISTER MOVE
    # ------------------------------------------------------------------

    def emit_mov_reg(self, rd: int, rn: int) -> None:
        """MOV Xd, Xn  (alias: ORR Xd, XZR, Xn)"""
        self._emit32(0xAA0003E0 | (rn << 16) | rd)

    # ------------------------------------------------------------------
    # DATA PROCESSING -- LOGICAL (Register)
    # ------------------------------------------------------------------

    def emit_and_reg(self, rd: int, rn: int, rm: int) -> None:
        """AND Xd, Xn, Xm"""
        self._emit32(0x8A000000 | (rm << 16) | (rn << 5) | rd)

    def emit_orr_reg(self, rd: int, rn: int, rm: int) -> None:
        """ORR Xd, Xn, Xm"""
        self._emit32(0xAA000000 | (rm << 16) | (rn << 5) | rd)

    def emit_eor_reg(self, rd: int, rn: int, rm: int) -> None:
        """EOR Xd, Xn, Xm"""
        self._emit32(0xCA000000 | (rm << 16) | (rn << 5) | rd)

    def emit_ands_reg(self, rd: int, rn: int, rm: int) -> None:
        """ANDS Xd, Xn, Xm"""
        self._emit32(0xEA000000 | (rm << 16) | (rn << 5) | rd)

    def emit_tst_reg(self, rn: int, rm: int) -> None:
        """TST Xn, Xm  (alias: ANDS XZR, Xn, Xm)"""
        self.emit_ands_reg(XZR, rn, rm)

    def emit_mvn(self, rd: int, rm: int) -> None:
        """MVN Xd, Xm  (alias: ORN Xd, XZR, Xm)"""
        self._emit32(0xAA2003E0 | (rm << 16) | rd)

    # ------------------------------------------------------------------
    # DATA PROCESSING -- LOGICAL (Immediate, raw bitmask encoding)
    # ------------------------------------------------------------------

    def emit_and_imm(self, rd: int, rn: int, n: int, immr: int, imms: int) -> None:
        """AND Xd, Xn, #bitmask  (raw N/immr/imms encoding)"""
        self._emit32(0x92000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)

    def emit_orr_imm(self, rd: int, rn: int, n: int, immr: int, imms: int) -> None:
        """ORR Xd, Xn, #bitmask"""
        self._emit32(0xB2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)

    def emit_eor_imm(self, rd: int, rn: int, n: int, immr: int, imms: int) -> None:
        """EOR Xd, Xn, #bitmask"""
        self._emit32(0xD2000000 | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)

    # ------------------------------------------------------------------
    # SHIFT / BITFIELD (Register & Immediate)
    # ------------------------------------------------------------------

    def emit_lsl_reg(self, rd: int, rn: int, rm: int) -> None:
        """LSL Xd, Xn, Xm"""
        self._emit32(0x9AC02000 | (rm << 16) | (rn << 5) | rd)

    def emit_lsr_reg(self, rd: int, rn: int, rm: int) -> None:
        """LSR Xd, Xn, Xm"""
        self._emit32(0x9AC02400 | (rm << 16) | (rn << 5) | rd)

    def emit_asr_reg(self, rd: int, rn: int, rm: int) -> None:
        """ASR Xd, Xn, Xm"""
        self._emit32(0x9AC02800 | (rm << 16) | (rn << 5) | rd)

    def emit_lsl_imm(self, rd: int, rn: int, shift: int) -> None:
        """LSL Xd, Xn, #shift  (UBFM alias)"""
        immr = (64 - shift) & 63
        imms = 63 - shift
        self._emit32(0xD3400000 | (immr << 16) | (imms << 10) | (rn << 5) | rd)

    def emit_lsr_imm(self, rd: int, rn: int, shift: int) -> None:
        """LSR Xd, Xn, #shift  (UBFM alias)"""
        self._emit32(0xD340FC00 | (shift << 16) | (rn << 5) | rd)

    def emit_asr_imm(self, rd: int, rn: int, shift: int) -> None:
        """ASR Xd, Xn, #shift  (SBFM alias)"""
        self._emit32(0x9340FC00 | (shift << 16) | (rn << 5) | rd)

    # ------------------------------------------------------------------
    # CONDITIONAL SELECT
    # ------------------------------------------------------------------

    def emit_csel(self, rd: int, rn: int, rm: int, cond: int) -> None:
        """CSEL Xd, Xn, Xm, cond"""
        self._emit32(0x9A800000 | (rm << 16) | (cond << 12) | (rn << 5) | rd)

    def emit_csinc(self, rd: int, rn: int, rm: int, cond: int) -> None:
        """CSINC Xd, Xn, Xm, cond"""
        self._emit32(0x9A800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd)

    def emit_cset(self, rd: int, cond: int) -> None:
        """CSET Xd, cond  (alias: CSINC Xd, XZR, XZR, invert(cond))"""
        inv = cond ^ 1
        self.emit_csinc(rd, XZR, XZR, inv)

    def emit_csneg(self, rd: int, rn: int, rm: int, cond: int) -> None:
        """CSNEG Xd, Xn, Xm, cond"""
        self._emit32(0xDA800400 | (rm << 16) | (cond << 12) | (rn << 5) | rd)

    def emit_cneg(self, rd: int, rn: int, cond: int) -> None:
        """CNEG Xd, Xn, cond"""
        inv = cond ^ 1
        self.emit_csneg(rd, rn, rn, inv)

    # ------------------------------------------------------------------
    # MOVE WIDE IMMEDIATE
    # ------------------------------------------------------------------

    def emit_movz(self, rd: int, imm16: int, hw: int = 0) -> None:
        """MOVZ Xd, #imm16, LSL #(hw*16)"""
        self._emit32(0xD2800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd)

    def emit_movk(self, rd: int, imm16: int, hw: int = 0) -> None:
        """MOVK Xd, #imm16, LSL #(hw*16)"""
        self._emit32(0xF2800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd)

    def emit_movn(self, rd: int, imm16: int, hw: int = 0) -> None:
        """MOVN Xd, #imm16, LSL #(hw*16)"""
        self._emit32(0x92800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd)

    # Convenience: load a full 64-bit immediate --------------------------

    def emit_mov_imm(self, rd: int, imm: int) -> None:
        """
        Load an arbitrary 64-bit immediate into Xd.

        Uses MOVZ for the lowest non-zero 16-bit chunk, then MOVK for any
        remaining non-zero chunks.  Produces 1-4 instructions.
        """
        imm = imm & 0xFFFF_FFFF_FFFF_FFFF
        chunks = [
            (imm >>  0) & 0xFFFF,
            (imm >> 16) & 0xFFFF,
            (imm >> 32) & 0xFFFF,
            (imm >> 48) & 0xFFFF,
        ]
        # Find the first non-zero chunk (or use chunk 0 if all zero).
        first = 0
        for i, c in enumerate(chunks):
            if c != 0:
                first = i
                break
        self.emit_movz(rd, chunks[first], first)
        for i, c in enumerate(chunks):
            if i != first and c != 0:
                self.emit_movk(rd, c, i)

    # Alias used by the task description
    emit_mov = emit_mov_imm

    # ------------------------------------------------------------------
    # LOADS AND STORES -- 64-bit unsigned offset
    # ------------------------------------------------------------------

    def emit_ldr(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDR Xt, [Xn, #offset]  (unsigned, scaled by 8)"""
        imm12 = (offset // 8) & 0xFFF
        self._emit32(0xF9400000 | (imm12 << 10) | (rn << 5) | rt)

    def emit_str(self, rt: int, rn: int, offset: int = 0) -> None:
        """STR Xt, [Xn, #offset]  (unsigned, scaled by 8)"""
        imm12 = (offset // 8) & 0xFFF
        self._emit32(0xF9000000 | (imm12 << 10) | (rn << 5) | rt)

    # -- Register offset -------------------------------------------------

    def emit_ldr_reg(self, rt: int, rn: int, rm: int) -> None:
        """LDR Xt, [Xn, Xm]"""
        self._emit32(0xF8606800 | (rm << 16) | (rn << 5) | rt)

    def emit_str_reg(self, rt: int, rn: int, rm: int) -> None:
        """STR Xt, [Xn, Xm]"""
        self._emit32(0xF8206800 | (rm << 16) | (rn << 5) | rt)

    # -- 32-bit loads/stores ---------------------------------------------

    def emit_ldr_w(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDR Wt, [Xn, #offset]  (32-bit, scaled by 4)"""
        imm12 = (offset // 4) & 0xFFF
        self._emit32(0xB9400000 | (imm12 << 10) | (rn << 5) | rt)

    def emit_str_w(self, rt: int, rn: int, offset: int = 0) -> None:
        """STR Wt, [Xn, #offset]  (32-bit, scaled by 4)"""
        imm12 = (offset // 4) & 0xFFF
        self._emit32(0xB9000000 | (imm12 << 10) | (rn << 5) | rt)

    # -- Byte loads/stores -----------------------------------------------

    def emit_ldrb(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDRB Wt, [Xn, #offset]"""
        self._emit32(0x39400000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt)

    def emit_strb(self, rt: int, rn: int, offset: int = 0) -> None:
        """STRB Wt, [Xn, #offset]"""
        self._emit32(0x39000000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt)

    # -- Halfword loads/stores -------------------------------------------

    def emit_ldrh(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDRH Wt, [Xn, #offset]  (scaled by 2)"""
        imm12 = (offset // 2) & 0xFFF
        self._emit32(0x79400000 | (imm12 << 10) | (rn << 5) | rt)

    def emit_strh(self, rt: int, rn: int, offset: int = 0) -> None:
        """STRH Wt, [Xn, #offset]  (scaled by 2)"""
        imm12 = (offset // 2) & 0xFFF
        self._emit32(0x79000000 | (imm12 << 10) | (rn << 5) | rt)

    # -- Signed loads ----------------------------------------------------

    def emit_ldrsb(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDRSB Xt, [Xn, #offset]"""
        self._emit32(0x39800000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt)

    def emit_ldrsh(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDRSH Xt, [Xn, #offset]  (scaled by 2)"""
        imm12 = (offset // 2) & 0xFFF
        self._emit32(0x79800000 | (imm12 << 10) | (rn << 5) | rt)

    def emit_ldrsw(self, rt: int, rn: int, offset: int = 0) -> None:
        """LDRSW Xt, [Xn, #offset]  (scaled by 4)"""
        imm12 = (offset // 4) & 0xFFF
        self._emit32(0xB9800000 | (imm12 << 10) | (rn << 5) | rt)

    # -- Load/Store Pair (signed offset) ---------------------------------

    def emit_ldp(self, rt1: int, rt2: int, rn: int, offset: int = 0) -> None:
        """LDP Xt1, Xt2, [Xn, #offset]  (signed, scaled by 8)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA9400000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    def emit_stp(self, rt1: int, rt2: int, rn: int, offset: int = 0) -> None:
        """STP Xt1, Xt2, [Xn, #offset]  (signed, scaled by 8)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA9000000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    def emit_ldp_pre(self, rt1: int, rt2: int, rn: int, offset: int) -> None:
        """LDP Xt1, Xt2, [Xn, #offset]!  (pre-indexed)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA9C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    def emit_stp_pre(self, rt1: int, rt2: int, rn: int, offset: int) -> None:
        """STP Xt1, Xt2, [Xn, #offset]!  (pre-indexed)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    def emit_ldp_post(self, rt1: int, rt2: int, rn: int, offset: int) -> None:
        """LDP Xt1, Xt2, [Xn], #offset  (post-indexed)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    def emit_stp_post(self, rt1: int, rt2: int, rn: int, offset: int) -> None:
        """STP Xt1, Xt2, [Xn], #offset  (post-indexed)"""
        imm7 = _signed_bits(offset // 8, 7)
        self._emit32(0xA8800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)

    # -- Pre/Post-indexed single register --------------------------------

    def emit_ldr_pre(self, rt: int, rn: int, simm9: int) -> None:
        """LDR Xt, [Xn, #simm9]!  (pre-indexed)"""
        self._emit32(0xF8400C00 | (_signed_bits(simm9, 9) << 12) | (rn << 5) | rt)

    def emit_str_pre(self, rt: int, rn: int, simm9: int) -> None:
        """STR Xt, [Xn, #simm9]!  (pre-indexed)"""
        self._emit32(0xF8000C00 | (_signed_bits(simm9, 9) << 12) | (rn << 5) | rt)

    def emit_ldr_post(self, rt: int, rn: int, simm9: int) -> None:
        """LDR Xt, [Xn], #simm9  (post-indexed)"""
        self._emit32(0xF8400400 | (_signed_bits(simm9, 9) << 12) | (rn << 5) | rt)

    def emit_str_post(self, rt: int, rn: int, simm9: int) -> None:
        """STR Xt, [Xn], #simm9  (post-indexed)"""
        self._emit32(0xF8000400 | (_signed_bits(simm9, 9) << 12) | (rn << 5) | rt)

    # ------------------------------------------------------------------
    # BRANCHES
    # ------------------------------------------------------------------

    def emit_b(self, offset: int = 0) -> None:
        """B offset  (unconditional, PC-relative, byte offset)"""
        imm26 = _signed_bits(offset >> 2, 26)
        self._emit32(0x14000000 | imm26)

    def emit_bl(self, offset: int = 0) -> None:
        """BL offset  (branch with link, byte offset)"""
        imm26 = _signed_bits(offset >> 2, 26)
        self._emit32(0x94000000 | imm26)

    def emit_b_cond(self, cond: int, offset: int = 0) -> None:
        """B.cond offset  (conditional branch, byte offset)"""
        imm19 = _signed_bits(offset >> 2, 19)
        self._emit32(0x54000000 | (imm19 << 5) | (cond & 0xF))

    def emit_cbz(self, rt: int, offset: int = 0) -> None:
        """CBZ Xt, offset"""
        imm19 = _signed_bits(offset >> 2, 19)
        self._emit32(0xB4000000 | (imm19 << 5) | rt)

    def emit_cbnz(self, rt: int, offset: int = 0) -> None:
        """CBNZ Xt, offset"""
        imm19 = _signed_bits(offset >> 2, 19)
        self._emit32(0xB5000000 | (imm19 << 5) | rt)

    def emit_tbz(self, rt: int, bit: int, offset: int = 0) -> None:
        """TBZ Xt, #bit, offset"""
        b5 = (bit >> 5) & 1
        b40 = bit & 0x1F
        imm14 = _signed_bits(offset >> 2, 14)
        self._emit32(0x36000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt)

    def emit_tbnz(self, rt: int, bit: int, offset: int = 0) -> None:
        """TBNZ Xt, #bit, offset"""
        b5 = (bit >> 5) & 1
        b40 = bit & 0x1F
        imm14 = _signed_bits(offset >> 2, 14)
        self._emit32(0x37000000 | (b5 << 31) | (b40 << 19) | (imm14 << 5) | rt)

    # -- Indirect branches -----------------------------------------------

    def emit_br(self, rn: int) -> None:
        """BR Xn"""
        self._emit32(0xD61F0000 | (rn << 5))

    def emit_blr(self, rn: int) -> None:
        """BLR Xn"""
        self._emit32(0xD63F0000 | (rn << 5))

    def emit_ret(self, rn: int = LR) -> None:
        """RET {Xn}  (default: X30 / LR)"""
        self._emit32(0xD65F0000 | (rn << 5))

    # -- Forward-branch helpers (emit placeholder, patch later) ----------

    def emit_b_fwd(self) -> int:
        """Emit a B with offset=0; return mark for later patch_b()."""
        mark = self._pos
        self.emit_b(0)
        return mark

    def emit_bl_fwd(self) -> int:
        """Emit a BL with offset=0; return mark for later patch_bl()."""
        mark = self._pos
        self.emit_bl(0)
        return mark

    def emit_b_cond_fwd(self, cond: int) -> int:
        """Emit a B.cond with offset=0; return mark for later patch_bcond()."""
        mark = self._pos
        self.emit_b_cond(cond, 0)
        return mark

    def emit_cbz_fwd(self, rt: int) -> int:
        mark = self._pos
        self.emit_cbz(rt, 0)
        return mark

    def emit_cbnz_fwd(self, rt: int) -> int:
        mark = self._pos
        self.emit_cbnz(rt, 0)
        return mark

    # ------------------------------------------------------------------
    # PC-RELATIVE ADDRESS GENERATION
    # ------------------------------------------------------------------

    def emit_adr(self, rd: int, offset: int) -> None:
        """ADR Xd, offset"""
        immlo = offset & 3
        immhi = (offset >> 2) & 0x7FFFF
        self._emit32((immlo << 29) | 0x10000000 | (immhi << 5) | rd)

    def emit_adrp(self, rd: int, offset: int) -> None:
        """ADRP Xd, offset"""
        immlo = offset & 3
        immhi = (offset >> 2) & 0x7FFFF
        self._emit32((immlo << 29) | 0x90000000 | (immhi << 5) | rd)

    # ------------------------------------------------------------------
    # SYSTEM INSTRUCTIONS
    # ------------------------------------------------------------------

    def emit_svc(self, imm16: int = 0) -> None:
        """SVC #imm16  (supervisor call / Linux syscall trap)"""
        self._emit32(0xD4000001 | ((imm16 & 0xFFFF) << 5))

    def emit_nop(self) -> None:
        """NOP"""
        self._emit32(0xD503201F)

    def emit_brk(self, imm16: int = 0) -> None:
        """BRK #imm16  (software breakpoint)"""
        self._emit32(0xD4200000 | ((imm16 & 0xFFFF) << 5))

    def emit_dsb_ish(self) -> None:
        """DSB ISH"""
        self._emit32(0xD5033B9F)

    def emit_dsb_sy(self) -> None:
        """DSB SY"""
        self._emit32(0xD5033F9F)

    def emit_dmb_ish(self) -> None:
        """DMB ISH"""
        self._emit32(0xD5033BBF)

    def emit_isb(self) -> None:
        """ISB"""
        self._emit32(0xD5033FDF)

    # ------------------------------------------------------------------
    # EXTEND / EXTRACT
    # ------------------------------------------------------------------

    def emit_sxtw(self, rd: int, rn: int) -> None:
        """SXTW Xd, Wn"""
        self._emit32(0x93407C00 | (rn << 5) | rd)

    def emit_sxtb(self, rd: int, rn: int) -> None:
        """SXTB Xd, Wn"""
        self._emit32(0x93401C00 | (rn << 5) | rd)

    def emit_sxth(self, rd: int, rn: int) -> None:
        """SXTH Xd, Wn"""
        self._emit32(0x93403C00 | (rn << 5) | rd)

    def emit_uxtb(self, rd: int, rn: int) -> None:
        """UXTB Wd, Wn"""
        self._emit32(0x53001C00 | (rn << 5) | rd)

    def emit_uxth(self, rd: int, rn: int) -> None:
        """UXTH Wd, Wn"""
        self._emit32(0x53003C00 | (rn << 5) | rd)

    # ------------------------------------------------------------------
    # ATOMIC / EXCLUSIVE
    # ------------------------------------------------------------------

    def emit_ldxr(self, rt: int, rn: int) -> None:
        """LDXR Xt, [Xn]"""
        self._emit32(0xC85F7C00 | (rn << 5) | rt)

    def emit_stxr(self, rs: int, rt: int, rn: int) -> None:
        """STXR Ws, Xt, [Xn]"""
        self._emit32(0xC8007C00 | (rs << 16) | (rn << 5) | rt)

    # ------------------------------------------------------------------
    # 32-BIT (W-register) ARITHMETIC
    # ------------------------------------------------------------------

    def emit_add_w(self, rd: int, rn: int, rm: int) -> None:
        """ADD Wd, Wn, Wm"""
        self._emit32(0x0B000000 | (rm << 16) | (rn << 5) | rd)

    def emit_sub_w(self, rd: int, rn: int, rm: int) -> None:
        """SUB Wd, Wn, Wm"""
        self._emit32(0x4B000000 | (rm << 16) | (rn << 5) | rd)

    def emit_add_imm_w(self, rd: int, rn: int, imm12: int) -> None:
        """ADD Wd, Wn, #imm12"""
        self._emit32(0x11000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_sub_imm_w(self, rd: int, rn: int, imm12: int) -> None:
        """SUB Wd, Wn, #imm12"""
        self._emit32(0x51000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd)

    def emit_mul_w(self, rd: int, rn: int, rm: int) -> None:
        """MUL Wd, Wn, Wm"""
        self._emit32(0x1B007C00 | (rm << 16) | (rn << 5) | rd)

    def emit_sdiv_w(self, rd: int, rn: int, rm: int) -> None:
        """SDIV Wd, Wn, Wm"""
        self._emit32(0x1AC00C00 | (rm << 16) | (rn << 5) | rd)

    def emit_udiv_w(self, rd: int, rn: int, rm: int) -> None:
        """UDIV Wd, Wn, Wm"""
        self._emit32(0x1AC00800 | (rm << 16) | (rn << 5) | rd)

    def emit_cmp_w(self, rn: int, rm: int) -> None:
        """CMP Wn, Wm"""
        self._emit32(0x6B000000 | (rm << 16) | (rn << 5) | XZR)

    def emit_cmp_imm_w(self, rn: int, imm12: int) -> None:
        """CMP Wn, #imm12"""
        self._emit32(0x71000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | XZR)

    def emit_movz_w(self, rd: int, imm16: int, hw: int = 0) -> None:
        """MOVZ Wd, #imm16, LSL #(hw*16)"""
        self._emit32(0x52800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd)

    def emit_movk_w(self, rd: int, imm16: int, hw: int = 0) -> None:
        """MOVK Wd, #imm16, LSL #(hw*16)"""
        self._emit32(0x72800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd)

    # ------------------------------------------------------------------
    # SYSCALL WRAPPERS
    # ------------------------------------------------------------------

    def emit_syscall(self, sysno: int) -> None:
        """Load syscall number into X8 and execute SVC #0."""
        self.emit_movz(X8, sysno & 0xFFFF, 0)
        if sysno > 0xFFFF:
            self.emit_movk(X8, (sysno >> 16) & 0xFFFF, 1)
        self.emit_svc(0)

    def emit_exit(self) -> None:
        self.emit_syscall(SYS_EXIT)

    def emit_write(self) -> None:
        self.emit_syscall(SYS_WRITE)

    def emit_read(self) -> None:
        self.emit_syscall(SYS_READ)

    # ------------------------------------------------------------------
    # FUNCTION PROLOGUE / EPILOGUE (AAPCS64)
    # ------------------------------------------------------------------

    def emit_prologue(self, frame_size: int = 16) -> None:
        """
        Standard function prologue: save FP/LR, set up frame pointer.
        *frame_size* must be a multiple of 16 (AAPCS64 stack alignment).
        """
        self.emit_stp_pre(FP, LR, SP, -frame_size)
        self.emit_mov_reg(FP, SP)

    def emit_epilogue(self, frame_size: int = 16) -> None:
        """Restore FP/LR and return."""
        self.emit_ldp_post(FP, LR, SP, frame_size)
        self.emit_ret()

    def emit_save_callee_saved(self, pairs: List[Tuple[int, int]], base_offset: int = 16) -> None:
        """Save pairs of callee-saved registers at [SP, #offset]."""
        off = base_offset
        for r1, r2 in pairs:
            self.emit_stp(r1, r2, SP, off)
            off += 16

    def emit_restore_callee_saved(self, pairs: List[Tuple[int, int]], base_offset: int = 16) -> None:
        """Restore pairs of callee-saved registers."""
        off = base_offset
        for r1, r2 in pairs:
            self.emit_ldp(r1, r2, SP, off)
            off += 16


# ---------------------------------------------------------------------------
# Simple register allocator
# ---------------------------------------------------------------------------

class SimpleRegAlloc:
    """
    Trivial register allocator: maps variable names to ARM64 registers.

    Uses x9-x15 as temporaries (caller-saved) and x19-x28 as long-lived
    (callee-saved).  x0-x7 are reserved for arguments/results and x8 for
    syscalls.
    """

    TEMP_REGS = list(range(X9, X16))        # x9..x15
    SAVED_REGS = list(range(X19, X29))       # x19..x28

    def __init__(self) -> None:
        self._map: Dict[str, int] = {}
        self._free_temp = list(reversed(self.TEMP_REGS))
        self._free_saved = list(reversed(self.SAVED_REGS))
        self._used_saved: List[int] = []

    def alloc_temp(self, name: str) -> int:
        """Allocate a caller-saved temp for *name*."""
        if name in self._map:
            return self._map[name]
        if not self._free_temp:
            raise RuntimeError(f"out of temp registers for '{name}'")
        reg = self._free_temp.pop()
        self._map[name] = reg
        return reg

    def alloc_saved(self, name: str) -> int:
        """Allocate a callee-saved register for *name*."""
        if name in self._map:
            return self._map[name]
        if not self._free_saved:
            raise RuntimeError(f"out of callee-saved registers for '{name}'")
        reg = self._free_saved.pop()
        self._map[name] = reg
        self._used_saved.append(reg)
        return reg

    def alloc(self, name: str, prefer_saved: bool = False) -> int:
        """Allocate a register, preferring temp unless *prefer_saved*."""
        if name in self._map:
            return self._map[name]
        if prefer_saved:
            return self.alloc_saved(name)
        return self.alloc_temp(name)

    def free(self, name: str) -> None:
        """Return a register to the free pool."""
        if name not in self._map:
            return
        reg = self._map.pop(name)
        if reg in self.TEMP_REGS:
            self._free_temp.append(reg)
        elif reg in self.SAVED_REGS:
            self._free_saved.append(reg)
            if reg in self._used_saved:
                self._used_saved.remove(reg)

    def lookup(self, name: str) -> Optional[int]:
        return self._map.get(name)

    def used_callee_saved_pairs(self) -> List[Tuple[int, int]]:
        """
        Return (r1, r2) pairs of callee-saved registers that must be
        preserved in the prologue/epilogue.  Unpaired trailing register
        is paired with XZR.
        """
        regs = sorted(self._used_saved)
        pairs = []
        i = 0
        while i < len(regs):
            r1 = regs[i]
            r2 = regs[i + 1] if i + 1 < len(regs) else XZR
            pairs.append((r1, r2))
            i += 2
        return pairs


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke test: encode a few instructions and verify bytes."""
    e = ARM64Emitter()

    # RET (x30) = 0xD65F03C0
    e.emit_ret()
    assert e.read32(0) == 0xD65F03C0, f"RET: got {e.read32(0):#010x}"

    # NOP = 0xD503201F
    e.emit_nop()
    assert e.read32(4) == 0xD503201F, f"NOP: got {e.read32(4):#010x}"

    # SVC #0 = 0xD4000001
    e.emit_svc(0)
    assert e.read32(8) == 0xD4000001, f"SVC: got {e.read32(8):#010x}"

    # ADD X0, X1, X2 = 0x8B020020
    e.emit_add_reg(X0, X1, X2)
    assert e.read32(12) == 0x8B020020, f"ADD reg: got {e.read32(12):#010x}"

    # SUB X3, X4, X5 = 0xCB050083
    e.emit_sub_reg(X3, X4, X5)
    assert e.read32(16) == 0xCB050083, f"SUB reg: got {e.read32(16):#010x}"

    # MOVZ X0, #42, LSL #0 = 0xD2800540
    e.emit_movz(X0, 42, 0)
    assert e.read32(20) == 0xD2800540, f"MOVZ: got {e.read32(20):#010x}"

    # MOVK X0, #0x1234, LSL #16 = 0xF2A24680
    e.emit_movk(X0, 0x1234, 1)
    assert e.read32(24) == 0xF2A24680, f"MOVK: got {e.read32(24):#010x}"

    # MUL X0, X1, X2 = 0x9B027C20
    e.emit_mul(X0, X1, X2)
    assert e.read32(28) == 0x9B027C20, f"MUL: got {e.read32(28):#010x}"

    # SDIV X0, X1, X2 = 0x9AC20C20
    e.emit_sdiv(X0, X1, X2)
    assert e.read32(32) == 0x9AC20C20, f"SDIV: got {e.read32(32):#010x}"

    # STR X0, [X1, #0] = 0xF9000020
    e.emit_str(X0, X1, 0)
    assert e.read32(36) == 0xF9000020, f"STR: got {e.read32(36):#010x}"

    # LDR X0, [X1, #8] = 0xF9400420
    e.emit_ldr(X0, X1, 8)
    assert e.read32(40) == 0xF9400420, f"LDR: got {e.read32(40):#010x}"

    # STP X29, X30, [SP, #-16]! = pre-indexed
    e.emit_stp_pre(FP, LR, SP, -16)
    expected_stp = 0xA9800000 | (_signed_bits(-16 // 8, 7) << 15) | (LR << 10) | (SP << 5) | FP
    assert e.read32(44) == _u32(expected_stp), f"STP pre: got {e.read32(44):#010x}, expected {_u32(expected_stp):#010x}"

    # LDP X29, X30, [SP], #16 = post-indexed
    e.emit_ldp_post(FP, LR, SP, 16)
    expected_ldp = 0xA8C00000 | (_signed_bits(16 // 8, 7) << 15) | (LR << 10) | (SP << 5) | FP
    assert e.read32(48) == _u32(expected_ldp), f"LDP post: got {e.read32(48):#010x}, expected {_u32(expected_ldp):#010x}"

    # B.EQ +8  = 0x54000040
    e.emit_b_cond(COND_EQ, 8)
    assert e.read32(52) == 0x54000040, f"B.EQ: got {e.read32(52):#010x}"

    # CMP X0, X1 => SUBS XZR, X0, X1 = 0xEB01001F
    e.emit_cmp_reg(X0, X1)
    assert e.read32(56) == 0xEB01001F, f"CMP reg: got {e.read32(56):#010x}"

    # Forward branch patching
    pos_before = e.pos()
    mark = e.emit_b_fwd()
    e.emit_nop()
    e.emit_nop()
    e.patch_b(mark)
    patched = e.read32(mark)
    # offset = 12 bytes (3 instructions * 4), imm26 = 3
    assert patched == 0x14000003, f"patch_b: got {patched:#010x}"

    # Verify total length
    code = e.get_code()
    assert len(code) == e.pos()

    # Register allocator smoke test
    ra = SimpleRegAlloc()
    r = ra.alloc_temp("a")
    assert r in SimpleRegAlloc.TEMP_REGS
    r2 = ra.alloc_saved("b")
    assert r2 in SimpleRegAlloc.SAVED_REGS
    assert ra.lookup("a") == r
    ra.free("a")
    assert ra.lookup("a") is None

    print("emit_arm64: all self-tests passed")


if __name__ == "__main__":
    _self_test()
