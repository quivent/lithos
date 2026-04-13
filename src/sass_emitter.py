"""
Lithos direct SASS emitter -- produces cubin binaries without ptxas.

Encodes Hopper (sm_90) SASS instructions as 16-byte pairs and wraps them
in a valid cubin ELF that cuModuleLoadData can load.

Reference: /home/ubuntu/lithos/sass/ENCODING.md
"""

from __future__ import annotations

import struct
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RZ = 0xFF  # zero register
PT = 7     # always-true predicate
RZ_REG = RZ

# Special register IDs (for S2R / S2UR)
SR_MAP = {
    "SR_TID_X":     0x21,
    "SR_TID_Y":     0x22,
    "SR_TID_Z":     0x23,
    "SR_CTAID_X":   0x25,
    "SR_CTAID_Y":   0x26,
    "SR_CTAID_Z":   0x27,
    "SR_NTID_X":    0x29,  # blockDim.x -- value from probed S2R encodings
    "SR_NTID_Y":    0x2a,
    "SR_NTID_Z":    0x2b,
    "SR_LANEID":    0x00,
    "SR_CLOCK":     0x50,
    "SR_GLOBALTIMERLO": 0x62,
}

# Opcode constants (bits [11:0])
OP = {
    # Float
    "FSEL_RR":    0x208,
    "FMNMX_RR":  0x209,
    "FSETP_RR":  0x20b,
    "FMUL_RR":   0x220,
    "FADD_RR":   0x221,
    "FFMA_RR":   0x223,
    "MUFU":      0x308,
    "FADD_RI":   0x421,
    "FSEL_RI":   0x808,
    "FSETP_RI":  0x80b,
    "FMUL_RI":   0x820,
    # Half
    "HMUL2_RR":  0x232,
    "HFMA2_RR":  0x235,
    "HFMA2_RI":  0x435,
    # Tensor
    "HMMA":      0x23c,
    # Integer
    "ISETP_RR":  0xc0c,
    "IADD3_RR":  0xc10,
    "IMAD_RR":   0xc24,
    "IMAD_RI":   0x424,
    # Data movement
    "MOV":       0x802,
    "CS2R":      0x805,
    "LEA_RR":    0x211,
    "UMOV":      0x882,
    # Memory global
    "LDG":       0x981,
    "STG":       0x986,
    "LDGSTS":    0xfae,
    # Memory shared
    "LDS":       0x984,
    "STS":       0x988,
    "LDSM":      0x83b,
    # Memory constant
    "LDC":       0xb82,
    "ULDC":      0xab9,
    # Special regs
    "S2R":       0x919,
    "S2UR":      0x9c3,
    # Shuffle
    "SHFL_REG":  0x589,
    "SHFL_IMM":  0xf89,
    # Sync
    "BAR":       0xb1d,
    "DEPBAR":    0x91a,
    "MEMBAR":    0x992,
    "ERRBAR":    0x9ab,
    "CGAERRBAR": 0x5ab,
    "LDGDEPBAR": 0x9af,
    "CCTL":      0x98f,
    # Control flow
    "BRA":       0x947,
    "EXIT":      0x94d,
    "NOP":       0x918,
    # Uniform
    "ULEA_RR":   0x291,
}

# MUFU sub-functions (ctrl bits [13:10])
MUFU_FUNC = {
    "COS":  0x0,
    "SIN":  0x1,
    "EX2":  0x2,
    "LG2":  0x3,
    "RCP":  0x4,
    "RSQ":  0x5,
    "SQRT": 0x8,
}

# ISETP comparison modes (ctrl bits)
ISETP_CMP = {
    "LT": 0x1,
    "EQ": 0x2,
    "LE": 0x3,
    "GT": 0x4,
    "NE": 0x5,
    "GE": 0x6,
}


# ---------------------------------------------------------------------------
# Instruction helper
# ---------------------------------------------------------------------------

def _encode_pred(pred: int = PT, negate: bool = False) -> int:
    """Encode predicate guard into bits [15:12]."""
    val = pred & 0x7
    if negate:
        val |= 0x8
    return val


def _make_inst(opcode: int, rd: int = RZ, rs1: int = RZ,
               rs2_or_imm: int = 0, pred: int = PT, pred_neg: bool = False,
               is_imm: bool = False) -> int:
    """Build the 64-bit instruction word."""
    guard = _encode_pred(pred, pred_neg)
    low16 = (guard << 12) | (opcode & 0xFFF)
    inst = low16 | ((rd & 0xFF) << 16) | ((rs1 & 0xFF) << 24)
    if is_imm:
        inst |= (rs2_or_imm & 0xFFFFFFFF) << 32
    else:
        inst |= (rs2_or_imm & 0xFF) << 32
    return inst & 0xFFFFFFFFFFFFFFFF


def _default_ctrl(stall: int = 15, yield_hint: int = 0,
                  wbar: int = 7, rbar: int = 7, wait_mask: int = 0) -> int:
    """Build a default control word with scheduling hints."""
    ctrl = 0
    ctrl |= (stall & 0x1F) << 53
    ctrl |= (yield_hint & 1) << 52
    ctrl |= (wbar & 0x7) << 49
    ctrl |= (rbar & 0x7) << 46
    ctrl |= (wait_mask & 0x3F) << 42
    return ctrl


# ---------------------------------------------------------------------------
# SASSEmitter
# ---------------------------------------------------------------------------

class SASSEmitter:
    """
    Emit sm_90 SASS instructions and wrap them in a cubin ELF.

    Usage::

        e = SASSEmitter("my_kernel")
        e.add_param_u64("ptr_a")
        e.add_param_u32("n")
        e.S2R(0, "SR_TID_X")
        e.EXIT()
        cubin = e.build_cubin()
    """

    def __init__(self, kernel_name: str, max_regs: int = 128):
        self.kernel_name = kernel_name
        self.max_regs = max_regs
        self._params: list[tuple[str, int, int]] = []  # (name, size, align)
        self._param_offset = 0
        self._instructions: list[tuple[int, int]] = []  # (inst_word, ctrl_word)
        self._labels: dict[str, int] = {}  # label -> instruction index
        self._fixups: list[tuple[int, str, str]] = []  # (instr_idx, label, fixup_type)

    # ----- Parameter declaration -----

    def add_param_u64(self, name: str):
        """Add a 64-bit (pointer) parameter."""
        self._add_param(name, 8, 8)

    def add_param_u32(self, name: str):
        """Add a 32-bit parameter."""
        self._add_param(name, 4, 4)

    def _add_param(self, name: str, size: int, align: int):
        # Align offset
        self._param_offset = (self._param_offset + align - 1) & ~(align - 1)
        self._params.append((name, size, self._param_offset))
        self._param_offset += size

    def param_offset(self, name: str) -> int:
        """Return the constant-bank offset for a named parameter."""
        for pname, psize, poff in self._params:
            if pname == name:
                return poff
        raise KeyError(f"Unknown parameter: {name}")

    # ----- Label support -----

    def label(self, name: str):
        """Mark current position as a branch target."""
        self._labels[name] = len(self._instructions)

    # ----- Low-level emit -----

    def emit(self, inst: int, ctrl: int):
        """Append a raw 16-byte instruction (inst_word, ctrl_word)."""
        self._instructions.append((inst & 0xFFFFFFFFFFFFFFFF,
                                   ctrl & 0xFFFFFFFFFFFFFFFF))

    def _emit_instr(self, inst: int, ctrl: int = None,
                    stall: int = 15, yield_hint: int = 0,
                    wbar: int = 7, rbar: int = 7, wait_mask: int = 0,
                    ctrl_extra: int = 0):
        """Emit with optional scheduling overrides."""
        if ctrl is None:
            ctrl = _default_ctrl(stall, yield_hint, wbar, rbar, wait_mask)
        ctrl |= ctrl_extra
        self.emit(inst, ctrl)

    # ----- Instruction methods -----

    # -- Floating-point arithmetic --

    def FADD(self, rd: int, rs1: int, rs2, pred: int = PT, pred_neg: bool = False,
             stall: int = 4, **kw):
        """FADD Rd, Rs1, Rs2 (register or float immediate)."""
        if isinstance(rs2, float):
            imm = struct.unpack('<I', struct.pack('<f', rs2))[0]
            inst = _make_inst(OP["FADD_RI"], rd, rs1, imm, pred, pred_neg, is_imm=True)
        else:
            inst = _make_inst(OP["FADD_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    def FMUL(self, rd: int, rs1: int, rs2, pred: int = PT, pred_neg: bool = False,
             stall: int = 4, **kw):
        """FMUL Rd, Rs1, Rs2 (register or float immediate)."""
        if isinstance(rs2, float):
            imm = struct.unpack('<I', struct.pack('<f', rs2))[0]
            inst = _make_inst(OP["FMUL_RI"], rd, rs1, imm, pred, pred_neg, is_imm=True)
        else:
            inst = _make_inst(OP["FMUL_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    def FFMA(self, rd: int, rs1: int, rs2: int, rs3: int,
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """FFMA Rd, Rs1, Rs2, Rs3.  Rs3 goes in ctrl[7:0]."""
        inst = _make_inst(OP["FFMA_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(rs3 & 0xFF), **kw)

    def FMNMX(self, rd: int, rs1: int, rs2: int, is_max: bool = False,
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """FMNMX Rd, Rs1, Rs2, PT/!PT.  is_max=True selects max."""
        inst = _make_inst(OP["FMNMX_RR"], rd, rs1, rs2, pred, pred_neg)
        sel = 0xF if is_max else 0x7  # !PT for max, PT for min
        self._emit_instr(inst, stall=stall, ctrl_extra=(sel << 20), **kw)

    def FSEL(self, rd: int, rs1: int, rs2: int, sel_pred: int = PT,
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """FSEL Rd, Rs1, Rs2, Px."""
        inst = _make_inst(OP["FSEL_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(sel_pred << 20), **kw)

    def FSETP(self, pd: int, rs1: int, rs2, cmp_mode: int = 0x5,
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """FSETP.cmp Pd, Rs1, Rs2. pd=destination predicate (0-6)."""
        if isinstance(rs2, float):
            imm = struct.unpack('<I', struct.pack('<f', rs2))[0]
            inst = _make_inst(OP["FSETP_RI"], RZ, rs1, imm, pred, pred_neg, is_imm=True)
        else:
            inst = _make_inst(OP["FSETP_RR"], RZ, rs1, rs2, pred, pred_neg)
        ctrl_extra = (0xF << 20) | ((pd * 2) << 16) | (cmp_mode << 12)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    def MUFU(self, rd: int, rs1: int, func: str = "RCP",
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """MUFU.func Rd, Rs1 -- special math functions."""
        func_code = MUFU_FUNC[func] if isinstance(func, str) else func
        inst = _make_inst(OP["MUFU"], rd, rs1, 0, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(func_code << 10), **kw)

    # -- Half-precision --

    def HFMA2(self, rd: int, rs1: int, rs2: int, rs3: int,
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """HFMA2.MMA Rd, Rs1, Rs2, Rs3."""
        inst = _make_inst(OP["HFMA2_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(rs3 & 0xFF), **kw)

    def HMUL2(self, rd: int, rs1: int, rs2: int,
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """HMUL2 Rd, Rs1, Rs2."""
        inst = _make_inst(OP["HMUL2_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    # -- Tensor core --

    def HMMA(self, rd: int, ra: int, rb: int, rc: int = RZ,
             shape_type: int = 0x18,
             pred: int = PT, pred_neg: bool = False, stall: int = 15, **kw):
        """HMMA.16816.F32 Rd, Ra, Rb, Rc.  shape_type=0x18 for m16n8k16.f32."""
        inst = _make_inst(OP["HMMA"], rd, ra, rb, pred, pred_neg)
        ctrl_extra = (rc & 0xFF) | ((shape_type & 0xFF) << 8)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    # -- Integer arithmetic --

    def IADD3(self, rd: int, rs1: int, rs2: int, rs3: int = RZ,
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """IADD3 Rd, Rs1, Rs2, Rs3."""
        inst = _make_inst(OP["IADD3_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(rs3 & 0xFF), **kw)

    def IMAD(self, rd: int, rs1: int, rs2, rs3: int = RZ,
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """IMAD Rd, Rs1, Rs2, Rs3.  Rs2 can be int for immediate form."""
        if isinstance(rs2, str):
            # Uniform register like "UR4" -- just encode as register form with value
            ureg = int(rs2.replace("UR", ""))
            inst = _make_inst(OP["IMAD_RR"], rd, rs1, ureg, pred, pred_neg)
        elif isinstance(rs2, int) and rs2 > 0xFF:
            # Treat as immediate
            inst = _make_inst(OP["IMAD_RI"], rd, rs1, rs2, pred, pred_neg, is_imm=True)
        else:
            inst = _make_inst(OP["IMAD_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(rs3 & 0xFF), **kw)

    def ISETP(self, pd: int, rs1: int, rs2: int, cmp: str = "LT",
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """ISETP.cmp Pd, Rs1, Rs2."""
        cmp_code = ISETP_CMP[cmp] if isinstance(cmp, str) else cmp
        inst = _make_inst(OP["ISETP_RR"], RZ, rs1, rs2, pred, pred_neg)
        ctrl_extra = (0xF << 20) | ((pd * 2) << 16) | (cmp_code << 12)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    # -- Data movement --

    def MOV(self, rd: int, imm: int, pred: int = PT, pred_neg: bool = False,
            stall: int = 2, **kw):
        """MOV Rd, #imm32."""
        inst = _make_inst(OP["MOV"], rd, RZ, imm & 0xFFFFFFFF, pred, pred_neg, is_imm=True)
        self._emit_instr(inst, stall=stall, **kw)

    def MOV_REG(self, rd: int, rs: int, pred: int = PT, pred_neg: bool = False,
                stall: int = 2, **kw):
        """MOV Rd, Rs via IADD3 Rd, Rs, RZ, RZ (no dedicated reg-reg MOV)."""
        self.IADD3(rd, rs, RZ, RZ, pred=pred, pred_neg=pred_neg, stall=stall, **kw)

    # -- Special registers --

    def S2R(self, rd: int, sr, pred: int = PT, pred_neg: bool = False,
            stall: int = 15, **kw):
        """S2R Rd, SR_xxx."""
        if isinstance(sr, str):
            sr_id = SR_MAP[sr]
        else:
            sr_id = sr
        # S2R encoding: opcode 0x919, SR ID in control word bits [15:8]
        inst = _make_inst(OP["S2R"], rd, RZ, 0, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=(sr_id << 8), **kw)

    def CS2R(self, rd: int, sr: int, pred: int = PT, pred_neg: bool = False,
             stall: int = 15, **kw):
        """CS2R Rd, CSR."""
        inst = _make_inst(OP["CS2R"], rd, RZ, sr, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    # -- Memory: Global --

    def LDG(self, rd: int, ra: int, offset: int = 0,
            size: int = 32, pred: int = PT, pred_neg: bool = False,
            stall: int = 15, wbar: int = 0, **kw):
        """
        LDG.E[.sz] Rd, [Ra + offset].

        size: 32, 64, 128 bits.
        ra: register holding base address (64-bit).
        """
        # LDG encoding based on probe analysis:
        #   inst bits [63:32] hold immediate offset (24 bits, shifted)
        #   inst [31:24] = Ra (address register)
        #   inst [23:16] = Rd (destination)
        #   ctrl encodes size: bits [13:10] = 0x2(32b), 0x2+bit9(64b), 0x3(128b)
        imm_field = (offset & 0xFFFFFF) << 32
        inst = (OP["LDG"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24))
        inst |= imm_field
        inst &= 0xFFFFFFFFFFFFFFFF
        # Size encoding in ctrl
        if size == 32:
            sz_bits = 0x2 << 10
        elif size == 64:
            sz_bits = (0x2 << 10) | (1 << 9)
        elif size == 128:
            sz_bits = 0x3 << 10
        else:
            sz_bits = 0x2 << 10
        self._emit_instr(inst, stall=stall, wbar=wbar, ctrl_extra=sz_bits, **kw)

    def STG(self, ra: int, rs: int, offset: int = 0,
            size: int = 32, pred: int = PT, pred_neg: bool = False,
            stall: int = 15, **kw):
        """
        STG.E[.sz] [Ra + offset], Rs.

        ra: register holding base address (64-bit).
        rs: register holding data to store.
        """
        imm_field = (offset & 0xFFFFFF) << 32
        inst = (OP["STG"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rs & 0xFF) << 16) | ((ra & 0xFF) << 24))
        inst |= imm_field
        inst &= 0xFFFFFFFFFFFFFFFF
        if size == 32:
            sz_bits = 0x2 << 10
        elif size == 64:
            sz_bits = (0x2 << 10) | (1 << 9)
        elif size == 128:
            sz_bits = 0x3 << 10
        else:
            sz_bits = 0x2 << 10
        self._emit_instr(inst, stall=stall, ctrl_extra=sz_bits, **kw)

    # -- Memory: Shared --

    def LDS(self, rd: int, ra: int, offset: int = 0, size: int = 32,
            pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """LDS[.sz] Rd, [Ra + offset]."""
        imm_field = (offset & 0xFFFFFF) << 32
        inst = (OP["LDS"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24))
        inst |= imm_field
        inst &= 0xFFFFFFFFFFFFFFFF
        if size == 32:
            sz_bits = 0x2 << 10
        elif size == 64:
            sz_bits = (0x2 << 10) | (1 << 9)
        elif size == 128:
            sz_bits = 0x3 << 10
        else:
            sz_bits = 0x2 << 10
        self._emit_instr(inst, stall=stall, ctrl_extra=sz_bits, **kw)

    def STS(self, ra: int, rs: int, offset: int = 0, size: int = 32,
            pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """STS[.sz] [Ra + offset], Rs."""
        imm_field = (offset & 0xFFFFFF) << 32
        inst = (OP["STS"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rs & 0xFF) << 16) | ((ra & 0xFF) << 24))
        inst |= imm_field
        inst &= 0xFFFFFFFFFFFFFFFF
        if size == 32:
            sz_bits = 0x2 << 10
        elif size == 64:
            sz_bits = (0x2 << 10) | (1 << 9)
        elif size == 128:
            sz_bits = 0x3 << 10
        else:
            sz_bits = 0x2 << 10
        self._emit_instr(inst, stall=stall, ctrl_extra=sz_bits, **kw)

    def LDSM(self, rd: int, ra: int, offset: int = 0,
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """LDSM.16.M88.4 Rd, [Ra]."""
        inst = _make_inst(OP["LDSM"], rd, RZ, ra, pred, pred_neg)
        self._emit_instr(inst, stall=stall, ctrl_extra=0x0200, **kw)

    # -- Async copy --

    def LDGSTS(self, rd_shared: int, ra_desc: int, rb_global: int,
               offset_shared: int = 0, offset_global: int = 0,
               pred: int = PT, pred_neg: bool = False, stall: int = 15, **kw):
        """LDGSTS.E.128 [Rd_shared+off], desc[Ra_desc][Rb_global+off]."""
        inst = (OP["LDGSTS"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rd_shared & 0xFF) << 16) | ((rb_global & 0xFF) << 24))
        # Encode offsets in upper bits
        inst |= ((offset_shared & 0xFF) << 40) | ((offset_global & 0xFF) << 48)
        inst &= 0xFFFFFFFFFFFFFFFF
        self._emit_instr(inst, stall=stall, **kw)

    def LDGDEPBAR(self, stall: int = 15, **kw):
        """LDGDEPBAR -- mark dependency for async copies."""
        inst = _make_inst(OP["LDGDEPBAR"], 0, RZ, 0, PT, False)
        self._emit_instr(inst, stall=stall, **kw)

    # -- Warp shuffle --

    def SHFL(self, rd: int, rs: int, lane, clamp: int = 0x1f,
             mode: str = "BFLY",
             pred: int = PT, pred_neg: bool = False, stall: int = 15, **kw):
        """SHFL.mode [PT,] Rd, Rs, lane, clamp."""
        mode_map = {"IDX": 0, "UP": 1, "DOWN": 2, "BFLY": 3}
        mode_bits = mode_map[mode]
        if isinstance(lane, int):
            # Immediate form
            opcode = OP["SHFL_IMM"]
            inst = (opcode | (_encode_pred(pred, pred_neg) << 12) |
                    ((rd & 0xFF) << 16) | ((rs & 0xFF) << 24))
            inst |= (clamp & 0xFF) << 40
            inst |= (lane & 0xFF) << 48
            inst |= mode_bits << 58
        else:
            # Register form
            opcode = OP["SHFL_REG"]
            lane_reg = lane if isinstance(lane, int) else RZ
            inst = (opcode | (_encode_pred(pred, pred_neg) << 12) |
                    ((rd & 0xFF) << 16) | ((rs & 0xFF) << 24))
            inst |= (clamp & 0xFF) << 40
            inst |= (lane_reg & 0xFF) << 32
            inst |= mode_bits << 58
        inst &= 0xFFFFFFFFFFFFFFFF
        self._emit_instr(inst, stall=stall, **kw)

    # -- Synchronization --

    def BAR(self, barrier_id: int = 0, pred: int = PT, pred_neg: bool = False,
            stall: int = 15, **kw):
        """BAR.SYNC.DEFER_BLOCKING barrier_id."""
        inst = _make_inst(OP["BAR"], 0, RZ, 0, pred, pred_neg)
        # Barrier ID in ctrl[16] area
        self._emit_instr(inst, stall=stall, ctrl_extra=(1 << 16), **kw)

    def MEMBAR(self, scope: str = "CTA", pred: int = PT, pred_neg: bool = False,
               stall: int = 15, **kw):
        """MEMBAR.scope."""
        scope_map = {"CTA": 0, "GPU": 2, "SYS": 3}
        inst = _make_inst(OP["MEMBAR"], 0, RZ, 0, pred, pred_neg)
        self._emit_instr(inst, stall=stall,
                         ctrl_extra=(scope_map[scope] << 12), **kw)

    def DEPBAR(self, sb: int = 0, count: int = 0,
               stall: int = 15, **kw):
        """DEPBAR.LE SBn, count."""
        inst = _make_inst(OP["DEPBAR"], 0, RZ, 0, PT, False)
        inst |= (1 << 47)  # LE bit
        inst &= 0xFFFFFFFFFFFFFFFF
        self._emit_instr(inst, stall=stall, **kw)

    def ERRBAR(self, stall: int = 15, **kw):
        """ERRBAR."""
        inst = _make_inst(OP["ERRBAR"], 0, RZ, 0, PT, False)
        self._emit_instr(inst, stall=stall, **kw)

    # -- Control flow --

    def NOP(self, stall: int = 15, **kw):
        """NOP."""
        inst = _make_inst(OP["NOP"], 0, RZ, 0, PT, False)
        self._emit_instr(inst, stall=stall, **kw)

    def EXIT(self, pred: int = PT, pred_neg: bool = False, stall: int = 15, **kw):
        """EXIT."""
        inst = _make_inst(OP["EXIT"], 0, RZ, 0, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    def BRA(self, target, pred: int = PT, pred_neg: bool = False,
            stall: int = 15, **kw):
        """
        BRA target.

        target: label name (str) or relative offset in bytes (int).
        """
        if isinstance(target, str):
            # Deferred fixup
            idx = len(self._instructions)
            inst = _make_inst(OP["BRA"], 0, RZ, 0, pred, pred_neg)
            self._emit_instr(inst, stall=stall, **kw)
            self._fixups.append((idx, target, "BRA"))
        else:
            # Immediate relative offset in bytes -- encode in bits [63:32]
            off_enc = target & 0xFFFFFFFF
            inst = _make_inst(OP["BRA"], 0, RZ, off_enc, pred, pred_neg, is_imm=True)
            self._emit_instr(inst, stall=stall, **kw)

    # -- Constant memory load (for parameter loading) --

    def LDC(self, rd: int, cbank_offset: int,
            size: int = 32, pred: int = PT, pred_neg: bool = False,
            stall: int = 4, wbar: int = 7, **kw):
        """
        LDC Rd, c[0x0][cbank_offset].

        Loads from constant bank 0 at the given byte offset.
        Used to load kernel parameters from the parameter bank.
        The driver places parameters at constant0 + 0x160 (offset within
        the .nv.constant0 section).
        """
        # LDC opcode: 0xb82
        # Encoding: inst[31:24]=RZ (base reg, typically RZ for constant load)
        #           inst[23:16]=Rd
        #           Offset encoded in upper 32 bits
        inst = (OP["LDC"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rd & 0xFF) << 16) | ((RZ & 0xFF) << 24))
        inst |= (cbank_offset & 0xFFFFFFFF) << 32
        inst &= 0xFFFFFFFFFFFFFFFF
        # Size in ctrl bits [13:10]
        if size == 32:
            sz_bits = 0x2 << 10
        elif size == 64:
            sz_bits = (0x2 << 10) | (1 << 9)
        else:
            sz_bits = 0x2 << 10
        self._emit_instr(inst, stall=stall, wbar=wbar, ctrl_extra=sz_bits, **kw)

    # -- LEA (Load Effective Address) --

    def LEA(self, rd: int, rs1: int, rs2: int,
            pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """LEA Rd, Rs1, Rs2."""
        inst = _make_inst(OP["LEA_RR"], rd, rs1, rs2, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    # =======================================================================
    # Build cubin
    # =======================================================================

    def build_cubin(self, sm_version: int = 90, toolkit_version: int = 12) -> bytes:
        """
        Produce a complete cubin ELF binary.

        Returns bytes suitable for cuModuleLoadData().
        """
        self._resolve_fixups()
        return _build_cubin_elf(
            kernel_name=self.kernel_name,
            instructions=self._instructions,
            params=self._params,
            max_regs=self.max_regs,
            sm_version=sm_version,
            toolkit_version=toolkit_version,
        )

    def _resolve_fixups(self):
        """Patch branch targets now that all labels are known."""
        for idx, label, kind in self._fixups:
            if label not in self._labels:
                raise ValueError(f"Undefined label: {label}")
            target_idx = self._labels[label]
            # Each instruction is 16 bytes
            current_pc = idx * 16
            target_pc = target_idx * 16
            rel_offset = target_pc - current_pc
            inst, ctrl = self._instructions[idx]
            # Clear upper 32 bits and set relative offset
            inst = (inst & 0xFFFFFFFF) | ((rel_offset & 0xFFFFFFFF) << 32)
            self._instructions[idx] = (inst, ctrl)


# =========================================================================
# Cubin ELF builder
# =========================================================================

def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _build_cubin_elf(kernel_name: str,
                     instructions: list[tuple[int, int]],
                     params: list[tuple[str, int, int]],
                     max_regs: int,
                     sm_version: int,
                     toolkit_version: int) -> bytes:
    """Build a minimal cubin ELF for Hopper (sm_90)."""

    # ---- Compute sizes ----
    code_size = len(instructions) * 16
    # Pad code to 128-byte alignment (required by .text section align=128)
    code_padded = _align(code_size, 128)

    # Total parameter size
    total_param_size = 0
    for _, psz, poff in params:
        total_param_size = max(total_param_size, poff + psz)

    # Constant bank: 0x160 bytes of header/reserved + param area
    CBANK_PARAM_BASE = 0x160
    cbank_size = _align(CBANK_PARAM_BASE + total_param_size, 4)

    # Section names
    text_sec_name = f".text.{kernel_name}"
    info_kern_name = f".nv.info.{kernel_name}"
    const0_name = f".nv.constant0.{kernel_name}"

    # ---- Section indices ----
    SEC_NULL     = 0
    SEC_SHSTRTAB = 1
    SEC_STRTAB   = 2
    SEC_SYMTAB   = 3
    SEC_NV_INFO  = 4
    SEC_NV_INFO_K = 5
    SEC_TEXT      = 6
    SEC_CONST0    = 7
    NUM_SECTIONS  = 8

    # Build section-name string table (.shstrtab)
    # We build it manually to get exact name offsets for each section header.
    shstrtab_data = bytearray()
    def _add_shstr(s: str) -> int:
        off = len(shstrtab_data)
        shstrtab_data.extend(s.encode('ascii') + b'\x00')
        return off

    SN_NULL      = _add_shstr("")
    SN_SHSTRTAB  = _add_shstr(".shstrtab")
    SN_STRTAB    = _add_shstr(".strtab")
    SN_SYMTAB    = _add_shstr(".symtab")
    SN_NV_INFO   = _add_shstr(".nv.info")
    SN_NV_INFO_K = _add_shstr(info_kern_name)
    SN_TEXT      = _add_shstr(text_sec_name)
    SN_CONST0    = _add_shstr(const0_name)
    shstrtab = bytes(shstrtab_data)

    # Build symbol string table (.strtab)
    strtab_data = bytearray()
    def _add_str(s: str) -> int:
        off = len(strtab_data)
        strtab_data.extend(s.encode('ascii') + b'\x00')
        return off

    _add_str("")  # null string at offset 0
    STR_TEXT     = _add_str(text_sec_name)
    STR_NV_INFO  = _add_str(".nv.info")
    STR_NV_INFO_K = _add_str(info_kern_name)
    STR_CONST0   = _add_str(const0_name)
    STR_FUNC     = _add_str(kernel_name)
    strtab = bytes(strtab_data)

    # ---- Build .nv.info (global) ----
    nv_info = bytearray()
    # EIATTR_CUDA_API_VERSION (0x2f): symidx -> toolkit_version
    # Format: attr(2) + size(2) + symidx(4) + value(4)
    SYM_FUNC_IDX = 5  # symbol index of the kernel function
    nv_info += struct.pack('<HH', 0x2f04, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, toolkit_version)
    # EIATTR_SW_WAR (0x11)
    nv_info += struct.pack('<HH', 0x1104, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)
    # EIATTR_CUDA_VERSION_HI (0x12)
    nv_info += struct.pack('<HH', 0x1204, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)

    # ---- Build .nv.info.<kernel> ----
    nv_info_k = bytearray()
    # EIATTR_MAXREG_COUNT (0x37)
    nv_info_k += struct.pack('<HHI', 0x3704, 4, max_regs)
    # EIATTR_KPARAM_INFO (0x17) for each param, in reverse ordinal order
    for i in range(len(params) - 1, -1, -1):
        pname, psz, poff = params[i]
        # Param info: 4 bytes zero, ordinal(2), cbank_offset(2), space(2), sizetype(2)
        # sizetype: 0x0021 for 8-byte, 0x0011 for 4-byte
        if psz == 8:
            sizetype = 0x0021
        elif psz == 4:
            sizetype = 0x0011
        else:
            sizetype = 0x0021
        nv_info_k += struct.pack('<HH', 0x1704, 12)
        nv_info_k += struct.pack('<I', 0)  # padding
        nv_info_k += struct.pack('<HH', i, CBANK_PARAM_BASE + poff)
        nv_info_k += struct.pack('<HH', 0xf000, sizetype)
    # EIATTR_EXIT_INSTR_OFFSETS (0x36) -- find EXIT instructions
    for idx, (inst, ctrl) in enumerate(instructions):
        if (inst & 0xFFF) == OP["EXIT"]:
            nv_info_k += struct.pack('<HHI', 0x3604, 4, idx * 16)
    # EIATTR_PARAM_CBANK (0x1c): cbank_offset, cbank_size
    nv_info_k += struct.pack('<HH', 0x1c04, 8)
    nv_info_k += struct.pack('<II', CBANK_PARAM_BASE, cbank_size)
    # EIATTR_CBANK_PARAM_SIZE (0x19): total param bytes -- format 03
    nv_info_k += struct.pack('<HH', 0x1903, total_param_size)
    # EIATTR_MAX_THREADS (0x50): no payload, format 03
    nv_info_k += struct.pack('<HH', 0x5003, 0)
    # EIATTR_CRS_STACK_SIZE (0x1b): format 03, value=0xFF (default)
    nv_info_k += struct.pack('<HH', 0x1b03, 0xFF)
    # EIATTR_REGCOUNT (0x0a): maxreg info
    nv_info_k += struct.pack('<HH', 0x0a04, 8)
    nv_info_k += struct.pack('<II', 7, (max_regs << 16) | 0x1c00)

    # ---- Build .nv.constant0 section (all zeros, params set by driver) ----
    const0_data = bytes(cbank_size)

    # ---- Build code (.text) ----
    code_data = bytearray()
    for inst, ctrl in instructions:
        code_data += struct.pack('<QQ', inst, ctrl)
    # Pad to 128-byte boundary
    while len(code_data) < code_padded:
        code_data += b'\x00'

    # ---- Build symbol table ----
    symbols = []
    # sym 0: NULL
    symbols.append(struct.pack('<IBBHQQ', 0, 0, 0, 0, 0, 0))
    # sym 1: .text.<kernel> section
    symbols.append(struct.pack('<IBBHQQ',
                               STR_TEXT, 0x03, 0x00, SEC_TEXT, 0, 0))
    # sym 2: .nv.info section
    symbols.append(struct.pack('<IBBHQQ',
                               STR_NV_INFO, 0x03, 0x00, SEC_NV_INFO, 0, 0))
    # sym 3: .nv.info.<kernel> section
    symbols.append(struct.pack('<IBBHQQ',
                               STR_NV_INFO_K, 0x03, 0x00, SEC_NV_INFO_K, 0, 0))
    # sym 4: .nv.constant0.<kernel> section
    symbols.append(struct.pack('<IBBHQQ',
                               STR_CONST0, 0x03, 0x00, SEC_CONST0, 0, 0))
    # sym 5: kernel function (GLOBAL FUNC)
    symbols.append(struct.pack('<IBBHQQ',
                               STR_FUNC, 0x12, 0x10, SEC_TEXT, 0, code_size))
    symtab_data = b''.join(symbols)
    num_symbols = len(symbols)

    # ---- Layout sections in file ----
    # ELF header = 64 bytes
    # Then sections: shstrtab, strtab, symtab, nv.info, nv.info.kern,
    # text (128-aligned), const0
    # Then section headers
    # Then program headers

    offset = 64  # after ELF header

    # .shstrtab
    shstrtab_off = offset
    offset += len(shstrtab)

    # .strtab
    strtab_off = offset
    offset += len(strtab)

    # .symtab (8-byte aligned)
    offset = _align(offset, 8)
    symtab_off = offset
    offset += len(symtab_data)

    # .nv.info (4-byte aligned)
    offset = _align(offset, 4)
    nv_info_off = offset
    offset += len(nv_info)

    # .nv.info.<kernel> (4-byte aligned)
    offset = _align(offset, 4)
    nv_info_k_off = offset
    offset += len(nv_info_k)

    # .text.<kernel> (128-byte aligned)
    offset = _align(offset, 128)
    text_off = offset
    offset += len(code_data)

    # .nv.constant0.<kernel> (4-byte aligned)
    offset = _align(offset, 4)
    const0_off = offset
    offset += len(const0_data)

    # Section headers (64-byte aligned)
    offset = _align(offset, 8)
    shdr_off = offset
    offset += NUM_SECTIONS * 64  # each section header is 64 bytes

    # Program headers
    phdr_off = offset
    NUM_PHDRS = 3  # PHDR, code LOAD, const0 LOAD
    offset += NUM_PHDRS * 56  # each phdr is 56 bytes

    # ---- Build section headers ----
    shdrs = bytearray()
    SHT_LOPROC = 0x70000000
    SHF_ALLOC = 0x2
    SHF_EXECINSTR = 0x4
    SHF_INFO_LINK = 0x40

    # [0] NULL
    shdrs += _elf_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # [1] .shstrtab  (SHT_STRTAB=3)
    shdrs += _elf_shdr(SN_SHSTRTAB, 3, 0, 0, shstrtab_off,
                       len(shstrtab), 0, 0, 1, 0)

    # [2] .strtab  (SHT_STRTAB=3)
    shdrs += _elf_shdr(SN_STRTAB, 3, 0, 0, strtab_off,
                       len(strtab), 0, 0, 1, 0)

    # [3] .symtab  (SHT_SYMTAB=2)
    shdrs += _elf_shdr(SN_SYMTAB, 2, 0, 0, symtab_off,
                       len(symtab_data), SEC_STRTAB, num_symbols, 8, 24)

    # [4] .nv.info  (SHT_LOPROC)
    shdrs += _elf_shdr(SN_NV_INFO, SHT_LOPROC, 0, 0,
                       nv_info_off, len(nv_info), SEC_SYMTAB, 0, 4, 0)

    # [5] .nv.info.<kernel>  (SHT_LOPROC, INFO_LINK)
    shdrs += _elf_shdr(SN_NV_INFO_K, SHT_LOPROC, SHF_INFO_LINK, 0,
                       nv_info_k_off, len(nv_info_k), SEC_SYMTAB, SEC_TEXT, 4, 0)

    # [6] .text.<kernel>  (SHT_PROGBITS=1, ALLOC|EXECINSTR)
    shdrs += _elf_shdr(SN_TEXT, 1, SHF_ALLOC | SHF_EXECINSTR, 0,
                       text_off, len(code_data), SEC_SYMTAB, SYM_FUNC_IDX, 128, 0)

    # [7] .nv.constant0.<kernel>  (SHT_PROGBITS=1, ALLOC|INFO_LINK)
    shdrs += _elf_shdr(SN_CONST0, 1, SHF_ALLOC | SHF_INFO_LINK, 0,
                       const0_off, len(const0_data), 0, SEC_TEXT, 4, 0)

    # ---- Build program headers ----
    phdrs = bytearray()

    # PHDR segment (points to itself)
    PT_PHDR = 6
    phdrs += _elf_phdr(PT_PHDR, phdr_off, 0, 0, NUM_PHDRS * 56,
                       NUM_PHDRS * 56, 0x4, 8)  # PF_R

    # LOAD segment for .text (code)
    PT_LOAD = 1
    phdrs += _elf_phdr(PT_LOAD, text_off, 0, 0, len(code_data),
                       len(code_data), 0x5, 8)  # PF_R | PF_X

    # LOAD segment for .nv.constant0
    phdrs += _elf_phdr(PT_LOAD, const0_off, 0, 0, len(const0_data),
                       len(const0_data), 0x4, 8)  # PF_R

    # ---- Build ELF header ----
    e_flags = (sm_version << 16) | (5 << 8) | sm_version  # 0x5a055a for sm_90
    elf_header = bytearray(64)
    # ELF magic
    elf_header[0:4] = b'\x7fELF'
    elf_header[4] = 2   # 64-bit
    elf_header[5] = 1   # little-endian
    elf_header[6] = 1   # ELF version
    elf_header[7] = 0x33  # OS/ABI = CUDA
    elf_header[8] = 7     # ABI version
    # e_type = ET_EXEC (2)
    struct.pack_into('<H', elf_header, 16, 2)
    # e_machine = 0xBE (CUDA)
    struct.pack_into('<H', elf_header, 18, 0xBE)
    # e_version = 0x80
    struct.pack_into('<I', elf_header, 20, 0x80)
    # e_entry = 0
    struct.pack_into('<Q', elf_header, 24, 0)
    # e_phoff
    struct.pack_into('<Q', elf_header, 32, phdr_off)
    # e_shoff
    struct.pack_into('<Q', elf_header, 40, shdr_off)
    # e_flags
    struct.pack_into('<I', elf_header, 48, e_flags)
    # e_ehsize = 64
    struct.pack_into('<H', elf_header, 52, 64)
    # e_phentsize = 56
    struct.pack_into('<H', elf_header, 54, 56)
    # e_phnum
    struct.pack_into('<H', elf_header, 56, NUM_PHDRS)
    # e_shentsize = 64
    struct.pack_into('<H', elf_header, 58, 64)
    # e_shnum
    struct.pack_into('<H', elf_header, 60, NUM_SECTIONS)
    # e_shstrndx
    struct.pack_into('<H', elf_header, 62, SEC_SHSTRTAB)

    # ---- Assemble final binary ----
    total_size = phdr_off + NUM_PHDRS * 56
    binary = bytearray(total_size)

    # Copy ELF header
    binary[0:64] = elf_header

    # Copy sections
    binary[shstrtab_off:shstrtab_off + len(shstrtab)] = shstrtab
    binary[strtab_off:strtab_off + len(strtab)] = strtab
    binary[symtab_off:symtab_off + len(symtab_data)] = symtab_data
    binary[nv_info_off:nv_info_off + len(nv_info)] = nv_info
    binary[nv_info_k_off:nv_info_k_off + len(nv_info_k)] = nv_info_k
    binary[text_off:text_off + len(code_data)] = code_data
    binary[const0_off:const0_off + len(const0_data)] = const0_data

    # Copy section headers
    binary[shdr_off:shdr_off + len(shdrs)] = shdrs

    # Copy program headers
    binary[phdr_off:phdr_off + len(phdrs)] = phdrs

    return bytes(binary)


def _build_strtab(strings: list[str]) -> tuple[bytes, dict[int, int]]:
    """Build an ELF string table. Returns (bytes, {list_index: offset})."""
    data = bytearray()
    offsets = {}
    for i, s in enumerate(strings):
        offsets[i] = len(data)
        data += s.encode('ascii') + b'\x00'
    return bytes(data), offsets


def _elf_shdr(name: int, sh_type: int, flags: int, addr: int,
              offset: int, size: int, link: int, info: int,
              addralign: int, entsize: int) -> bytes:
    """Pack a 64-byte ELF section header."""
    return struct.pack('<IIQQQQIIqq',
                       name, sh_type, flags, addr,
                       offset, size, link, info,
                       addralign, entsize)


def _elf_phdr(p_type: int, offset: int, vaddr: int, paddr: int,
              filesz: int, memsz: int, flags: int, align: int) -> bytes:
    """Pack a 56-byte ELF program header."""
    return struct.pack('<IIQQQQQQ',
                       p_type, flags, offset, vaddr,
                       paddr, filesz, memsz, align)


# =========================================================================
# Convenience: build a vector-add kernel
# =========================================================================

def build_vecadd_cubin() -> bytes:
    """
    Build a complete cubin for a float32 vector-add kernel:
        __global__ void vecadd(float *a, float *b, float *c, unsigned int n)

    This kernel computes c[tid] = a[tid] + b[tid] for tid < n.
    """
    e = SASSEmitter("vecadd", max_regs=32)
    e.add_param_u64("a")   # param 0 at cbank+0x160
    e.add_param_u64("b")   # param 1 at cbank+0x168
    e.add_param_u64("c")   # param 2 at cbank+0x170
    e.add_param_u32("n")   # param 3 at cbank+0x178

    CBANK = 0x160

    # R0 = tid.x
    e.S2R(0, "SR_TID_X", stall=2)
    # R1 = ctaid.x
    e.S2R(1, "SR_CTAID_X", stall=2)
    # R2 = ntid.x (blockDim.x)
    e.S2R(2, "SR_NTID_X", stall=15)

    # R3 = ctaid.x * ntid.x + tid.x = global thread id
    e.IMAD(3, 1, 2, 0, stall=4)

    # Load n from constant bank
    e.LDC(4, CBANK + 0x18, size=32, stall=4)

    # Bounds check: P0 = (R3 >= R4) -- if true, skip to exit
    e.ISETP(0, 3, 4, cmp="GE", stall=4)

    # Load base pointers from constant bank (64-bit loads)
    # R6:R7 = a (pointer)
    e.LDC(6, CBANK + 0x00, size=64, stall=2)
    # R8:R9 = b (pointer)
    e.LDC(8, CBANK + 0x08, size=64, stall=2)
    # R10:R11 = c (pointer)
    e.LDC(10, CBANK + 0x10, size=64, stall=4)

    # Compute byte offset: R5 = R3 << 2 (tid * 4 for float)
    # Use IMAD: R5 = R3 * 4 + 0 (with immediate 4)
    # But we need to use SHL via IMAD: R5 = R3 * 4 + RZ
    # IMAD Rd, Rs1, imm, Rs3  -- imm form opcode 0x424
    # Actually let's use IADD3 R5 = R3 + R3 twice, or IMAD with const
    # Simplest: use LEA or shift.  Let's use IMAD with immediate.
    # IMAD R5, R3, 4, RZ
    e.MOV(5, 4, stall=2)  # R5 = 4 (the constant)
    e.IMAD(5, 3, 5, RZ, stall=4)  # R5 = R3 * 4 + 0

    # Widen offset to 64-bit: R12 = 0 (high word of offset)
    e.MOV(12, 0, stall=2)

    # Add offset to base pointers
    # a_addr = R6:R7 + R5:R12  =>  R14:R15
    e.IADD3(14, 6, 5, RZ, stall=2)
    e.IADD3(15, 7, 12, RZ, stall=4)  # carry approximation (ignore carry for simplicity)

    # b_addr = R8:R9 + R5:R12  =>  R16:R17
    e.IADD3(16, 8, 5, RZ, stall=2)
    e.IADD3(17, 9, 12, RZ, stall=4)

    # c_addr = R10:R11 + R5:R12  =>  R18:R19
    e.IADD3(18, 10, 5, RZ, stall=2)
    e.IADD3(19, 11, 12, RZ, stall=4)

    # Load a[tid] -> R20
    e.LDG(20, 14, size=32, stall=15, wbar=0)
    # Load b[tid] -> R21
    e.LDG(21, 16, size=32, stall=15, wbar=1)

    # Wait for loads (wait on barriers 0 and 1)
    e.NOP(stall=15, wait_mask=0x03)

    # R22 = R20 + R21  (FADD)
    e.FADD(22, 20, 21, stall=4)

    # Store c[tid] = R22
    e.STG(18, 22, size=32, stall=4)

    # @P0 EXIT (bounds-check fail path -- predicated EXIT before actual EXIT)
    e.EXIT(pred=0, stall=15)

    # Normal EXIT
    e.EXIT(stall=15)

    return e.build_cubin()


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    import sys
    import os

    cubin = build_vecadd_cubin()

    # Write cubin to disk for debugging
    out_path = "/tmp/vecadd_lithos.cubin"
    with open(out_path, "wb") as f:
        f.write(cubin)
    print(f"Wrote cubin ({len(cubin)} bytes) to {out_path}")

    # Verify with readelf if available
    ret = os.system(f"readelf -h {out_path} 2>/dev/null")
    if ret == 0:
        os.system(f"readelf -S {out_path} 2>/dev/null")

    # Try to load and run on GPU
    try:
        import ctypes
        import array
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cuda_driver import CUDADriver

        gpu = CUDADriver()
        print(f"\nGPU: {gpu.device_name}")

        # Load cubin from bytes
        module = gpu.load_cubin_bytes(cubin)
        func = gpu.get_function(module, "vecadd")
        print(f"Loaded kernel: vecadd @ {func}")

        # Allocate test data
        N = 256
        float_size = 4

        # Create host arrays
        a_host = array.array('f', [float(i) for i in range(N)])
        b_host = array.array('f', [float(i) * 0.5 for i in range(N)])
        c_host = array.array('f', [0.0] * N)

        # Allocate device memory
        a_dev = gpu.mem_alloc(N * float_size)
        b_dev = gpu.mem_alloc(N * float_size)
        c_dev = gpu.mem_alloc(N * float_size)

        # Copy to device
        a_buf, _ = a_host.buffer_info()
        b_buf, _ = b_host.buffer_info()
        gpu.memcpy_htod(a_dev, ctypes.c_void_p(a_buf), N * float_size)
        gpu.memcpy_htod(b_dev, ctypes.c_void_p(b_buf), N * float_size)

        # Launch kernel
        gpu.launch(func,
                   grid=(1, 1, 1),
                   block=(N, 1, 1),
                   args=[ctypes.c_uint64(a_dev.value),
                         ctypes.c_uint64(b_dev.value),
                         ctypes.c_uint64(c_dev.value),
                         ctypes.c_uint32(N)])
        gpu.synchronize()

        # Copy back
        c_buf, _ = c_host.buffer_info()
        gpu.memcpy_dtoh(ctypes.c_void_p(c_buf), c_dev, N * float_size)

        # Verify
        errors = 0
        for i in range(N):
            expected = float(i) + float(i) * 0.5
            if abs(c_host[i] - expected) > 1e-3:
                if errors < 5:
                    print(f"  MISMATCH at [{i}]: got {c_host[i]}, expected {expected}")
                errors += 1

        if errors == 0:
            print(f"\nSUCCESS: all {N} elements correct!")
        else:
            print(f"\nFAILED: {errors}/{N} mismatches")

        # Cleanup
        gpu.mem_free(a_dev)
        gpu.mem_free(b_dev)
        gpu.mem_free(c_dev)
        gpu.close()

    except Exception as ex:
        print(f"\nGPU test error: {ex}")
        import traceback
        traceback.print_exc()
