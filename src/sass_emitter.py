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
    "ISETP_RR":  0x20c,   # ISETP Pd, Rs1, Rs2 (reg-reg)
    "ISETP_RU":  0xc0c,   # ISETP Pd, Rs1, URs2 (rs2 is uniform)
    "IADD3_RR":  0x210,   # IADD3 Rd, Rs1, Rs2, Rs3   (all regular regs)
    "IADD3_RU":  0xc10,   # IADD3 Rd, Rs1, URs2, Rs3  (rs2 is uniform reg)
    "IMAD_RR":   0x224,   # IMAD Rd, Rs1, Rs2, Rs3
    "IMAD_RU":   0xc24,   # IMAD Rd, Rs1, URs2, Rs3
    "IMAD_RI":   0x824,
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
        """IADD3 Rd, Rs1, Rs2, Rs3.

        ctrl bits [23:5] encode the predicate-out (Pcarry, Pcarry2) and
        carry-in predicate fields.  When all outputs are unused we set them
        to PT/disabled, which matches ptxas's ``0xf3e0`` pattern at bits
        [23:5]:  output-pred=PT, dual-pred=PT, carry-in=PT, no negate.
        """
        inst = _make_inst(OP["IADD3_RR"], rd, rs1, rs2, pred, pred_neg)
        # Default: all predicate slots disabled. Reference ctrl pattern for
        # the RRR form (opcode 0x210) with no Pcarry output:
        #   ctrl[31:24]=0x07, ctrl[23:16]=0xff, ctrl[15:8]=0xe0, ctrl[7:0]=rs3.
        ctrl_extra = 0x07ffe000 | (rs3 & 0xFF)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    def IMAD(self, rd: int, rs1: int, rs2, rs3: int = RZ,
             pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """IMAD Rd, Rs1, Rs2, Rs3.  Rs2 can be int for immediate form.

        Reference ctrl bits [23:0]: ``0x8e02`` in ctrl[23:8], Rs3 in [7:0].
        These encode the signed/unsigned flags, output predicate=PT,
        etc. -- values observed across all ptxas IMAD encodings we probed.
        """
        if isinstance(rs2, str):
            # Uniform register like "UR4" -- just encode as register form with value
            ureg = int(rs2.replace("UR", ""))
            inst = _make_inst(OP["IMAD_RR"], rd, rs1, ureg, pred, pred_neg)
        elif isinstance(rs2, int) and rs2 > 0xFF:
            # Treat as immediate
            inst = _make_inst(OP["IMAD_RI"], rd, rs1, rs2, pred, pred_neg, is_imm=True)
        else:
            inst = _make_inst(OP["IMAD_RR"], rd, rs1, rs2, pred, pred_neg)
        ctrl_extra = 0x078e0200 | (rs3 & 0xFF)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    def ISETP(self, pd: int, rs1: int, rs2: int, cmp: str = "LT",
              pred: int = PT, pred_neg: bool = False, stall: int = 4, **kw):
        """ISETP.cmp Pd, Rs1, Rs2."""
        cmp_code = ISETP_CMP[cmp] if isinstance(cmp, str) else cmp
        inst = _make_inst(OP["ISETP_RR"], RZ, rs1, rs2, pred, pred_neg)
        # Reference ISETP.<cmp>.U32.AND ctrl[27:0]:
        #   bits [31:24]=0x0b (signed/unsigned + predicate-combine-AND flags)
        #   bits [23:20]=0xf  (Pnot output = PT, unused)
        #   bits [19:16]=Pd*2 (output predicate register)
        #   bits [15:8]=0x60|(cmp_code<<4)  -> cmp_code goes in bits [15:12]
        #   bits [7:0]=0x70   (src predicate PT, no negate)
        # Reference ISETP.<cmp>.U32.AND (RR form) ctrl[31:0]:
        #   [31:24] = 0x03  (ISETP flags: U32 + AND combine, no negate)
        #   [23:20] = 0xf   (Pnot output disabled)
        #   [19:16] = Pd*2  (output predicate)
        #   [15:12] = cmp_code
        #   [11:8]  = 0x60  (boolean combine mode = AND)
        #   [7:0]   = 0x70  (src predicate PT, no negate)
        ctrl_extra = (0x03 << 24) | (0xF << 20) | ((pd * 2) << 16) \
                   | (cmp_code << 12) | 0x6070
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

    # -- Data movement --

    def MOV(self, rd: int, imm: int, pred: int = PT, pred_neg: bool = False,
            stall: int = 2, **kw):
        """MOV Rd, #imm32.

        ctrl[11:8] = 0xf is the 4-bit byte write-mask (all 4 bytes live).
        ptxas always sets it for a full 32-bit MOV.
        """
        inst = _make_inst(OP["MOV"], rd, RZ, imm & 0xFFFFFFFF, pred, pred_neg, is_imm=True)
        self._emit_instr(inst, stall=stall, ctrl_extra=0x0f00, **kw)

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
        # S2R encoding: opcode 0x919.
        #   inst[23:16] = Rd
        #   inst[31:24] = 0 (not RZ -- must be zero per ptxas encodings)
        #   ctrl[15:8]  = SR ID
        guard = _encode_pred(pred, pred_neg)
        inst = (OP["S2R"] | (guard << 12) | ((rd & 0xFF) << 16))
        self._emit_instr(inst, stall=stall, ctrl_extra=(sr_id << 8), **kw)

    def CS2R(self, rd: int, sr: int, pred: int = PT, pred_neg: bool = False,
             stall: int = 15, **kw):
        """CS2R Rd, CSR."""
        inst = _make_inst(OP["CS2R"], rd, RZ, sr, pred, pred_neg)
        self._emit_instr(inst, stall=stall, **kw)

    # -- Memory: Global --

    def LDG(self, rd: int, ra: int, offset: int = 0,
            size: int = 32, ur_desc: int = 4,
            pred: int = PT, pred_neg: bool = False,
            stall: int = 15, wbar: int = 0, **kw):
        """
        LDG.E[.sz] Rd, desc[URx][Ra.64 + offset].

        Hopper (sm_90) mandates descriptor-based global addressing: the
        uniform register ``ur_desc`` must already contain the generic-global
        memory descriptor (typically loaded from ``c[0x0][0x208]`` into UR4).

        Layout of the 128-bit instruction:
          inst[11:0]   = 0x981 (LDG)
          inst[15:12]  = predicate guard
          inst[23:16]  = Rd           (destination; register pair for .64, quad for .128)
          inst[31:24]  = Ra           (64-bit address register pair: Ra, Ra+1)
          inst[39:32]  = URx          (memory descriptor uniform register)
          inst[63:40]  = 24-bit signed byte offset
          ctrl[11:8]   = 0x9/0xb/0xd  (32/64/128-bit size)
          ctrl[15:12]  = 0x1          (fixed: LDG variant flags)
          ctrl[23:16]  = 0x1e         (cache op .EF + mem-space = global)
          ctrl[31:24]  = 0x0c         (.E extended addressing + desc mode)
        """
        if size == 32:
            sz_nibble = 0x9
        elif size == 64:
            sz_nibble = 0xb
        elif size == 128:
            sz_nibble = 0xd
        else:
            raise ValueError(f"LDG size must be 32/64/128, got {size}")
        inst = (OP["LDG"] | (_encode_pred(pred, pred_neg) << 12) |
                ((rd & 0xFF) << 16) | ((ra & 0xFF) << 24) |
                ((ur_desc & 0xFF) << 32) |
                ((offset & 0xFFFFFF) << 40))
        inst &= 0xFFFFFFFFFFFFFFFF
        ctrl_extra = (0x0c << 24) | (0x1e << 16) | (0x10 << 8) | (sz_nibble << 8)
        # 0x10 << 8 | 0x09 << 8 = 0x1900 (size=32). For 64: 0x1b00, 128: 0x1d00.
        # Resolved explicitly:
        ctrl_extra = 0x0c1e0000 | ((0x10 | sz_nibble) << 8)
        self._emit_instr(inst, stall=stall, wbar=wbar, ctrl_extra=ctrl_extra, **kw)

    def STG(self, ra: int, rs: int, offset: int = 0,
            size: int = 32, ur_desc: int = 4,
            pred: int = PT, pred_neg: bool = False,
            stall: int = 15, **kw):
        """
        STG.E[.sz] desc[URx][Ra.64 + offset], Rs.

        Like LDG, sm_90 requires a descriptor. ``ur_desc`` picks the uniform
        register holding it (UR4 = c[0x0][0x208] generic-global descriptor).

        Layout of the 128-bit instruction:
          inst[11:0]   = 0x986 (STG)
          inst[15:12]  = predicate guard
          inst[23:16]  = 0x00          (reserved; STG has no Rd)
          inst[31:24]  = Ra            (64-bit address register pair)
          inst[39:32]  = Rs            (data register; pair/quad for .64/.128)
          inst[63:40]  = 24-bit signed byte offset
          ctrl[7:0]    = URx           (descriptor encoded in the 3rd-source slot)
          ctrl[11:8]   = 0x9/0xb/0xd   (size nibble)
          ctrl[15:12]  = 0x1           (fixed)
          ctrl[23:16]  = 0x10          (.EF / store, no cache read)
          ctrl[31:24]  = 0x0c          (.E + desc mode)
        """
        if size == 32:
            sz_nibble = 0x9
        elif size == 64:
            sz_nibble = 0xb
        elif size == 128:
            sz_nibble = 0xd
        else:
            raise ValueError(f"STG size must be 32/64/128, got {size}")
        inst = (OP["STG"] | (_encode_pred(pred, pred_neg) << 12) |
                ((ra & 0xFF) << 24) | ((rs & 0xFF) << 32) |
                ((offset & 0xFFFFFF) << 40))
        inst &= 0xFFFFFFFFFFFFFFFF
        ctrl_extra = (0x0c << 24) | (0x10 << 16) | ((0x10 | sz_nibble) << 8) | (ur_desc & 0xFF)
        self._emit_instr(inst, stall=stall, ctrl_extra=ctrl_extra, **kw)

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
    """Build a minimal cubin ELF for Hopper (sm_90).

    The layout mirrors ptxas output so that the CUDA driver recognises
    the kernel entry point.  Required sections:

        .shstrtab, .strtab, .symtab, .nv.info, .nv.info.<kernel>,
        .nv.callgraph, .text.<kernel>, .nv.shared.reserved.0,
        .nv.constant0.<kernel>

    Program headers: PHDR, LOAD(phdr), LOAD(.text + shared), LOAD(shared),
    LOAD(constant0).
    """

    # ---- Compute sizes ----
    code_size = len(instructions) * 16
    code_padded = _align(code_size, 128)

    total_param_size = 0
    for _, psz, poff in params:
        total_param_size = max(total_param_size, poff + psz)

    # Constant bank layout (matching ptxas output):
    #   Bytes 0x00..0x6f  = system constants (gridDim, blockDim, etc.)
    #   Bytes 0x70..0x70+param_window = kernel parameters
    # PARAM_CBANK offset tells the driver where params start.
    # The constant0 section must be at least 0x22c bytes for sm_90
    # or the CUDA driver will reject the launch.
    # On sm_90 the ABI places kernel params at c[0x0][0x210].
    # Bytes 0x200..0x20f hold system-provided memory descriptors
    # (UR4 = generic-global descriptor at 0x208) required by LDG.E/STG.E.
    CBANK_PARAM_BASE = 0x210
    cbank_param_window = 0x180  # standard param window
    cbank_size = CBANK_PARAM_BASE + cbank_param_window  # enough for all params

    text_sec_name = f".text.{kernel_name}"
    info_kern_name = f".nv.info.{kernel_name}"
    const0_name = f".nv.constant0.{kernel_name}"
    shared_name = f".nv.shared.{kernel_name}"

    # ---- Section indices ----
    SEC_NULL       = 0
    SEC_SHSTRTAB   = 1
    SEC_STRTAB     = 2
    SEC_SYMTAB     = 3
    SEC_NV_INFO    = 4
    SEC_NV_INFO_K  = 5
    SEC_NV_CALLGR  = 6
    SEC_TEXT        = 7
    SEC_NV_SHARED  = 8
    SEC_CONST0     = 9
    NUM_SECTIONS   = 10

    # ---- String tables ----
    shstrtab_data = bytearray()
    def _add_shstr(s: str) -> int:
        off = len(shstrtab_data)
        shstrtab_data.extend(s.encode('ascii') + b'\x00')
        return off

    SN_NULL       = _add_shstr("")
    SN_SHSTRTAB   = _add_shstr(".shstrtab")
    SN_STRTAB     = _add_shstr(".strtab")
    SN_SYMTAB     = _add_shstr(".symtab")
    SN_NV_INFO    = _add_shstr(".nv.info")
    SN_NV_INFO_K  = _add_shstr(info_kern_name)
    SN_NV_CALLGR  = _add_shstr(".nv.callgraph")
    SN_TEXT        = _add_shstr(text_sec_name)
    SN_NV_SHARED  = _add_shstr(".nv.shared.reserved.0")
    SN_CONST0     = _add_shstr(const0_name)
    shstrtab = bytes(shstrtab_data)

    strtab_data = bytearray()
    def _add_str(s: str) -> int:
        off = len(strtab_data)
        strtab_data.extend(s.encode('ascii') + b'\x00')
        return off

    _add_str("")  # null at offset 0
    STR_TEXT       = _add_str(text_sec_name)
    STR_NV_INFO    = _add_str(".nv.info")
    STR_NV_INFO_K  = _add_str(info_kern_name)
    STR_NV_CALLGR  = _add_str(".nv.callgraph")
    STR_NV_SHARED_OBJ = _add_str(".nv.reservedSmem.offset0")
    STR_NV_SHARED_ALIAS = _add_str("__nv_reservedSMEM_offset_0_alias")
    STR_CONST0     = _add_str(const0_name)
    STR_FUNC       = _add_str(kernel_name)
    strtab = bytes(strtab_data)

    # ---- .nv.info (global) ----
    SYM_FUNC_IDX = 7  # will be set correctly below based on symbol layout
    nv_info = bytearray()
    nv_info += struct.pack('<HH', 0x2f04, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, toolkit_version)
    nv_info += struct.pack('<HH', 0x1104, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)
    nv_info += struct.pack('<HH', 0x1204, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)

    # ---- .nv.info.<kernel> ----
    nv_info_k = bytearray()
    # MAXREG_COUNT
    nv_info_k += struct.pack('<HHI', 0x3704, 4, max_regs)
    # KPARAM_INFO -- offsets relative to PARAM_CBANK base
    for i in range(len(params) - 1, -1, -1):
        pname, psz, poff = params[i]
        sizetype = 0x0021 if psz == 8 else 0x0011
        nv_info_k += struct.pack('<HH', 0x1704, 12)
        nv_info_k += struct.pack('<I', 0)
        nv_info_k += struct.pack('<HH', i, poff)
        nv_info_k += struct.pack('<HH', 0xf000, sizetype)
    # MAX_THREADS
    nv_info_k += struct.pack('<HH', 0x5003, 0)
    # CRS_STACK_SIZE
    nv_info_k += struct.pack('<HH', 0x1b03, 0xFF)
    # PARAM_CBANK
    nv_info_k += struct.pack('<HH', 0x1c04, 8)
    nv_info_k += struct.pack('<II', CBANK_PARAM_BASE, cbank_param_window)
    # CBANK_PARAM_SIZE
    nv_info_k += struct.pack('<HH', 0x1903, total_param_size)
    # REGCOUNT -- second word encodes actual register allocation:
    #   bits [7:0]   = 0x10 (flags)
    #   bits [15:8]  = 0x02 (bar alloc + flags)
    #   bits [23:16] = actual register count for the kernel
    #   bits [31:24] = 0x00
    # Compute actual register count from instructions (only count true
    # register operands, skip fields that encode immediates/SR IDs).
    actual_regs = 8  # minimum allocation
    # Opcodes where bits [39:32] are NOT a src2 register:
    NON_REG_SRC2 = {
        0x919, 0x9c3,  # S2R, S2UR -- SR ID in ctrl, src2 unused
        0x802, 0x882,  # MOV, UMOV -- immediate in upper 32 bits
        0x94d, 0x947, 0x918,  # EXIT, BRA, NOP
        0xb82, 0xab9,  # LDC, ULDC -- cbank offset encoded
        0xb1d, 0x91a, 0x992, 0x9ab, 0x5ab, 0x9af, 0x98f,  # barriers
    }
    # Opcodes with immediate forms (upper 32 bits = immediate, not register)
    IMM_OPCODES = {
        0x421, 0x820, 0x808, 0x80b,  # FADD_I, FMUL_I, FSEL_I, FSETP_I
        0x424, 0x435, 0x835,  # IMAD_I, HFMA2_I
        0xf89,  # SHFL_IMM
    }
    for inst, ctrl in instructions:
        opcode = inst & 0xFFF
        rd = (inst >> 16) & 0xFF
        rs1 = (inst >> 24) & 0xFF
        # Count destination register (skip for opcodes with no Rd: EXIT/NOP/
        # barriers *and* stores which put 0x00 in the Rd slot).
        NO_RD = (0x94d, 0x918, 0xb1d, 0x91a, 0x992, 0x9ab,
                 0x5ab, 0x9af, 0x986, 0x988, 0x947)  # + STG, STS, BRA
        # rd < 0xFF to skip the RZ marker
        if rd < 0xFF and opcode not in NO_RD:
            actual_regs = max(actual_regs, rd + 1)
        # Count src1 register
        if rs1 != 0xFF:
            actual_regs = max(actual_regs, rs1 + 1)
            # LDG/STG/LDS/STS use a 64-bit address register pair
            if opcode in (0x981, 0x986, 0x984, 0x988):
                actual_regs = max(actual_regs, rs1 + 2)
        # For LDG, inst[39:32] is the UR descriptor -- not a regular reg.
        # For STG, inst[39:32] is the data register (Rs), which we count below.
        # For other reg-reg forms inst[39:32] is src2.
        if opcode in (0x981,):
            pass  # UR descriptor, skip
        elif opcode == 0x986:  # STG: data reg Rs at [39:32]
            rs_data = (inst >> 32) & 0xFF
            if rs_data != 0xFF:
                actual_regs = max(actual_regs, rs_data + 1)
                # Account for 64/128-bit stores (register pair/quad)
                sz_nibble = (ctrl >> 8) & 0xF
                if sz_nibble == 0xb:   # 64-bit
                    actual_regs = max(actual_regs, rs_data + 2)
                elif sz_nibble == 0xd:  # 128-bit
                    actual_regs = max(actual_regs, rs_data + 4)
        elif opcode == 0x988:  # STS
            rs_data = (inst >> 32) & 0xFF
            if rs_data != 0xFF:
                actual_regs = max(actual_regs, rs_data + 1)
        elif opcode not in NON_REG_SRC2 and opcode not in IMM_OPCODES:
            rs2 = (inst >> 32) & 0xFF
            if rs2 != 0xFF:
                actual_regs = max(actual_regs, rs2 + 1)
        # Count src3 in ctrl[7:0] for 4-operand instructions
        if opcode in (0x223, 0x235, 0x23c, 0xc10, 0xc24):  # FFMA, HFMA2, HMMA, IADD3, IMAD
            rs3 = ctrl & 0xFF
            if rs3 != 0xFF:
                actual_regs = max(actual_regs, rs3 + 1)
        # For LDG/LDS 64/128-bit loads, the destination is a register pair/quad.
        if opcode in (0x981, 0x984):
            sz_nibble = (ctrl >> 8) & 0xF
            if sz_nibble == 0xb:   # 64-bit
                actual_regs = max(actual_regs, rd + 2)
            elif sz_nibble == 0xd:  # 128-bit
                actual_regs = max(actual_regs, rd + 4)
    # Hopper requires register-count to be at least (max_used + 2) aligned
    # to 8, because LDG/LDS may write (rd, rd+1) pairs and the scheduler
    # reserves two spill slots behind the high-water mark.
    actual_regs = actual_regs + 2
    actual_regs = (actual_regs + 7) & ~7
    if actual_regs < 24:
        actual_regs = 24  # CUDA minimum for most kernels
    regcount_val = (actual_regs << 16) | 0x0210
    nv_info_k += struct.pack('<HH', 0x0a04, 8)
    nv_info_k += struct.pack('<II', 7, regcount_val)
    # EXIT_INSTR_OFFSETS
    for idx, (inst, ctrl) in enumerate(instructions):
        if (inst & 0xFFF) == OP["EXIT"]:
            nv_info_k += struct.pack('<HHI', 0x3604, 4, idx * 16)

    # ---- .nv.callgraph ----
    callgraph = bytearray()
    callgraph += struct.pack('<II', 0x00000000, 0xFFFFFFFF)
    callgraph += struct.pack('<II', 0x00000000, 0xFFFFFFFE)
    callgraph += struct.pack('<II', 0x00000000, 0xFFFFFFFD)
    callgraph += struct.pack('<II', 0x00000000, 0xFFFFFFFC)

    # ---- .nv.constant0 ----
    const0_data = bytes(cbank_size)

    # ---- .text code ----
    code_data = bytearray()
    for inst, ctrl in instructions:
        code_data += struct.pack('<QQ', inst, ctrl)
    # Pad with NOPs (0x7918) like ptxas does.  Zero-fill would decode as an
    # illegal opcode if the hardware prefetches past the BRA trailer and
    # trigger CUDA_ERROR_ILLEGAL_INSTRUCTION on otherwise-valid kernels.
    NOP_INST = 0x0000000000007918
    NOP_CTRL = 0x000fc00000000000
    while len(code_data) < code_padded:
        code_data += struct.pack('<QQ', NOP_INST, NOP_CTRL)

    # ---- Symbol table ----
    # Match reference layout:
    # 0: NULL
    # 1: .text section sym
    # 2: .nv.reservedSmem.offset0 (WEAK OBJECT, UND)
    # 3: __nv_reservedSMEM_offset_0_alias (WEAK NOTYPE, shndx=SEC_NV_SHARED, other=0xa0)
    # 4: .nv.callgraph section sym
    # 5: .nv.info section sym  -- NOT present in ref (ref has debug_frame there)
    # Actually reference sym order:
    #   0: NULL, 1: .text.vecadd section, 2: .nv.reservedSmem.offset0,
    #   3: __nv_reserved..., 4: .debug_frame section, 5: .nv.callgraph section,
    #   6: vecadd (FUNC), 7: .nv.constant0.vecadd section
    # We'll skip debug_frame but keep the rest:
    #   0: NULL, 1: .text section, 2: .nv.reservedSmem weak,
    #   3: __nv_reserved alias, 4: .nv.info section, 5: .nv.callgraph section,
    #   6: .nv.constant0 section, 7: kernel func
    SYM_FUNC_IDX = 7

    symbols = []
    symbols.append(struct.pack('<IBBHQQ', 0, 0, 0, 0, 0, 0))  # 0: NULL
    symbols.append(struct.pack('<IBBHQQ',  # 1: .text section
                               STR_TEXT, 0x03, 0x00, SEC_TEXT, 0, 0))
    symbols.append(struct.pack('<IBBHQQ',  # 2: .nv.reservedSmem.offset0 (WEAK OBJECT, UND)
                               STR_NV_SHARED_OBJ, 0x21, 0x00, 0, 0, 4))
    symbols.append(struct.pack('<IBBHQQ',  # 3: __nv_reservedSMEM_offset_0_alias (WEAK, other=0xa0)
                               STR_NV_SHARED_ALIAS, 0x20, 0xa0, SEC_NV_SHARED, 0, 0))
    symbols.append(struct.pack('<IBBHQQ',  # 4: .nv.info section
                               STR_NV_INFO, 0x03, 0x00, SEC_NV_INFO, 0, 0))
    symbols.append(struct.pack('<IBBHQQ',  # 5: .nv.callgraph section
                               STR_NV_CALLGR, 0x03, 0x00, SEC_NV_CALLGR, 0, 0))
    symbols.append(struct.pack('<IBBHQQ',  # 6: .nv.constant0 section
                               STR_CONST0, 0x03, 0x00, SEC_CONST0, 0, 0))
    symbols.append(struct.pack('<IBBHQQ',  # 7: kernel function
                               STR_FUNC, 0x12, 0x10, SEC_TEXT, 0, code_size))
    symtab_data = b''.join(symbols)
    num_symbols = len(symbols)

    # Update nv.info to reference correct SYM_FUNC_IDX.
    # 0x2f04 = EIATTR_REGCOUNT (global form): value is the actual register
    # count used by the kernel.  Must match the per-kernel 0x0a04 entry,
    # otherwise the driver rejects launches that touch "out of range" regs
    # with CUDA_ERROR_ILLEGAL_INSTRUCTION.
    nv_info = bytearray()
    nv_info += struct.pack('<HH', 0x2f04, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, actual_regs)
    nv_info += struct.pack('<HH', 0x1104, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)
    nv_info += struct.pack('<HH', 0x1204, 8)
    nv_info += struct.pack('<II', SYM_FUNC_IDX, 0)

    # ---- File layout ----
    offset = 64

    shstrtab_off = offset
    offset += len(shstrtab)

    strtab_off = offset
    offset += len(strtab)

    offset = _align(offset, 8)
    symtab_off = offset
    offset += len(symtab_data)

    offset = _align(offset, 4)
    nv_info_off = offset
    offset += len(nv_info)

    offset = _align(offset, 4)
    nv_info_k_off = offset
    offset += len(nv_info_k)

    offset = _align(offset, 4)
    callgraph_off = offset
    offset += len(callgraph)

    # .text (128-byte aligned)
    offset = _align(offset, 128)
    text_off = offset
    offset += len(code_data)

    # .nv.shared.reserved.0 is NOBITS -- shares offset with next section
    shared_off = offset  # same offset, zero size

    # .nv.constant0
    offset = _align(offset, 4)
    const0_off = offset
    offset += len(const0_data)

    # Section headers
    offset = _align(offset, 8)
    shdr_off = offset
    offset += NUM_SECTIONS * 64

    # Program headers
    phdr_off = offset
    NUM_PHDRS = 5  # PHDR, LOAD(phdr), LOAD(.text+shared), LOAD(shared), LOAD(const0)
    offset += NUM_PHDRS * 56

    # ---- Section headers ----
    shdrs = bytearray()
    SHT_LOPROC = 0x70000000
    SHT_LOPROC1 = 0x70000001
    SHT_NOBITS = 8
    SHF_ALLOC = 0x2
    SHF_WRITE = 0x1
    SHF_EXECINSTR = 0x4
    SHF_INFO_LINK = 0x40

    shdrs += _elf_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # [0] NULL
    shdrs += _elf_shdr(SN_SHSTRTAB, 3, 0, 0, shstrtab_off,
                       len(shstrtab), 0, 0, 1, 0)  # [1]
    shdrs += _elf_shdr(SN_STRTAB, 3, 0, 0, strtab_off,
                       len(strtab), 0, 0, 1, 0)  # [2]
    shdrs += _elf_shdr(SN_SYMTAB, 2, 0, 0, symtab_off,
                       len(symtab_data), SEC_STRTAB, SYM_FUNC_IDX, 8, 24)  # [3]
    shdrs += _elf_shdr(SN_NV_INFO, SHT_LOPROC, 0, 0,
                       nv_info_off, len(nv_info), SEC_SYMTAB, 0, 4, 0)  # [4]
    shdrs += _elf_shdr(SN_NV_INFO_K, SHT_LOPROC, SHF_INFO_LINK, 0,
                       nv_info_k_off, len(nv_info_k), SEC_SYMTAB, SEC_TEXT, 4, 0)  # [5]
    shdrs += _elf_shdr(SN_NV_CALLGR, SHT_LOPROC1, 0, 0,
                       callgraph_off, len(callgraph), SEC_SYMTAB, 0, 4, 8)  # [6]
    shdrs += _elf_shdr(SN_TEXT, 1, SHF_ALLOC | SHF_EXECINSTR, 0,
                       text_off, len(code_data), SEC_SYMTAB, SYM_FUNC_IDX, 128, 0)  # [7]
    shdrs += _elf_shdr(SN_NV_SHARED, SHT_NOBITS, SHF_WRITE | SHF_ALLOC, 0,
                       shared_off, 0, 0, 0, 1, 0)  # [8]
    shdrs += _elf_shdr(SN_CONST0, 1, SHF_ALLOC | SHF_INFO_LINK, 0,
                       const0_off, len(const0_data), 0, SEC_TEXT, 4, 0)  # [9]

    # ---- Program headers ----
    phdrs = bytearray()
    PT_PHDR = 6
    PT_LOAD = 1

    # PHDR (self-referential)
    phdrs += _elf_phdr(PT_PHDR, phdr_off, 0, 0,
                       NUM_PHDRS * 56, NUM_PHDRS * 56, 0x4, 8)
    # LOAD for PHDR
    phdrs += _elf_phdr(PT_LOAD, phdr_off, 0, 0,
                       NUM_PHDRS * 56, NUM_PHDRS * 56, 0x4, 8)
    # LOAD for .text + .nv.shared
    phdrs += _elf_phdr(PT_LOAD, text_off, 0, 0,
                       len(code_data), len(code_data), 0x5, 8)  # R|X
    # LOAD for .nv.shared (RW, zero size)
    phdrs += _elf_phdr(PT_LOAD, shared_off, 0, 0, 0, 0, 0x6, 8)  # R|W
    # LOAD for .nv.constant0
    phdrs += _elf_phdr(PT_LOAD, const0_off, 0, 0,
                       len(const0_data), len(const0_data), 0x4, 8)  # R

    # ---- ELF header ----
    e_flags = (sm_version << 16) | (5 << 8) | sm_version
    elf_header = bytearray(64)
    elf_header[0:4] = b'\x7fELF'
    elf_header[4] = 2   # 64-bit
    elf_header[5] = 1   # little-endian
    elf_header[6] = 1   # version
    elf_header[7] = 0x33  # CUDA ABI
    elf_header[8] = 7     # ABI version
    struct.pack_into('<H', elf_header, 16, 2)      # ET_EXEC
    struct.pack_into('<H', elf_header, 18, 0xBE)   # CUDA
    struct.pack_into('<I', elf_header, 20, 0x80)   # version
    struct.pack_into('<Q', elf_header, 24, 0)      # entry
    struct.pack_into('<Q', elf_header, 32, phdr_off)
    struct.pack_into('<Q', elf_header, 40, shdr_off)
    struct.pack_into('<I', elf_header, 48, e_flags)
    struct.pack_into('<H', elf_header, 52, 64)     # ehsize
    struct.pack_into('<H', elf_header, 54, 56)     # phentsize
    struct.pack_into('<H', elf_header, 56, NUM_PHDRS)
    struct.pack_into('<H', elf_header, 58, 64)     # shentsize
    struct.pack_into('<H', elf_header, 60, NUM_SECTIONS)
    struct.pack_into('<H', elf_header, 62, SEC_SHSTRTAB)

    # ---- Assemble ----
    total_size = phdr_off + NUM_PHDRS * 56
    binary = bytearray(total_size)
    binary[0:64] = elf_header
    binary[shstrtab_off:shstrtab_off + len(shstrtab)] = shstrtab
    binary[strtab_off:strtab_off + len(strtab)] = strtab
    binary[symtab_off:symtab_off + len(symtab_data)] = symtab_data
    binary[nv_info_off:nv_info_off + len(nv_info)] = nv_info
    binary[nv_info_k_off:nv_info_k_off + len(nv_info_k)] = nv_info_k
    binary[callgraph_off:callgraph_off + len(callgraph)] = callgraph
    binary[text_off:text_off + len(code_data)] = code_data
    binary[const0_off:const0_off + len(const0_data)] = const0_data
    binary[shdr_off:shdr_off + len(shdrs)] = shdrs
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

def build_store_const_cubin() -> bytes:
    """
    Build a kernel that stores a constant value to c[0]:
        __global__ void store_const(float *c) { c[0] = 42.0f; }

    This is the simplest "real" kernel that proves our emitter can
    produce functional SASS with memory access and parameter loading.
    """
    e = SASSEmitter("store_const", max_regs=128)
    e.add_param_u64("c")

    # ULDC helper: inst[63:40] = byte_addr/4.
    def uldc(rd, byte_addr, is64=True, stall=0, rbar=7, wait=0x38):
        enc = (byte_addr // 4) & 0xFFFF
        inst = 0xab9 | (0x7 << 12) | ((rd & 0xFF) << 16) | (enc << 40)
        sz = 0x0a00 if is64 else 0x0800
        ctrl = _default_ctrl(stall=stall, rbar=rbar, wait_mask=wait)
        ctrl |= sz
        e.emit(inst, ctrl)

    def ldc(rd, byte_addr, is64=True, stall=0, rbar=7, wait=0x38):
        enc = (byte_addr // 4) & 0xFFFF
        inst = (0xb82 | (0x7 << 12) | ((rd & 0xFF) << 16)
                | (0xFF << 24) | (enc << 40))
        sz = 0x0a00 if is64 else 0x0800
        ctrl = _default_ctrl(stall=stall, rbar=rbar, wait_mask=wait)
        ctrl |= sz
        e.emit(inst, ctrl)

    # Load generic-global memory descriptor into UR4
    uldc(4, 0x208, is64=True, stall=0, wait=0x00)

    # Load c pointer (64-bit) from cbank[0x210] into R2:R3
    ldc(2, 0x210, is64=True, stall=4, wait=0x00)

    # MOV R4, 42.0f = 0x42280000
    e.MOV(4, 0x42280000, stall=2)

    # STG.E desc[UR4][R2.64], R4
    e.STG(2, 4, size=32, ur_desc=4, stall=4)

    # EXIT
    e.emit(0x000000000000794d, 0x000fea0003800000)

    return e.build_cubin()


def build_vecadd_cubin() -> bytes:
    """
    Build a complete cubin for a float32 vector-add kernel:
        __global__ void vecadd(float *a, float *b, float *c, unsigned int n)

    This kernel computes c[tid] = a[tid] + b[tid] for tid < n.

    The instructions are modelled after ptxas output for this kernel,
    using raw emit() for ULDC param loads (which have a complex encoding
    that differs from plain LDC).
    """
    e = SASSEmitter("vecadd", max_regs=128)
    e.add_param_u64("a")   # param 0 at cbank+0x70, ordinal offset 0
    e.add_param_u64("b")   # param 1 at cbank+0x78, ordinal offset 8
    e.add_param_u64("c")   # param 2 at cbank+0x80, ordinal offset 16
    e.add_param_u32("n")   # param 3 at cbank+0x88, ordinal offset 24

    # ULDC helper: load from constant bank 0 at byte address into a uniform reg.
    #   inst[11:0]   = 0xab9 (ULDC)
    #   inst[23:16]  = URd
    #   inst[63:40]  = byte_addr / 4  (the word index within the cbank)
    # ctrl size nibble: 0x0800 for 32-bit, 0x0a00 for 64-bit.
    def uldc(rd, byte_addr, is64=True, stall=0, rbar=7, wait=0x38):
        enc = (byte_addr // 4) & 0xFFFF
        inst = 0xab9 | (0x7 << 12) | ((rd & 0xFF) << 16) | (enc << 40)
        sz = 0x0a00 if is64 else 0x0800
        ctrl = _default_ctrl(stall=stall, rbar=rbar, wait_mask=wait)
        ctrl |= sz
        e.emit(inst, ctrl)

    # LDC helper: load from cbank into a regular register.
    #   inst[11:0]   = 0xb82 (LDC)
    #   inst[23:16]  = Rd
    #   inst[31:24]  = 0xFF (RZ base)
    #   inst[63:40]  = byte_addr / 4
    def ldc(rd, byte_addr, is64=True, stall=0, rbar=7, wait=0x38):
        enc = (byte_addr // 4) & 0xFFFF
        inst = (0xb82 | (0x7 << 12) | ((rd & 0xFF) << 16)
                | (0xFF << 24) | (enc << 40))
        sz = 0x0a00 if is64 else 0x0800
        ctrl = _default_ctrl(stall=stall, rbar=rbar, wait_mask=wait)
        ctrl |= sz
        e.emit(inst, ctrl)

    # -- Kernel body --
    # Load the generic-global memory descriptor into UR4 (mandatory on sm_90
    # for LDG.E/STG.E addressing).
    uldc(4, 0x208, is64=True, stall=15)

    # S2R is variable-latency; use a large stall so the value has time to
    # land before the first consumer reads it.  (A proper implementation
    # would use write-barriers + wait_mask bits, but these stall counts
    # give correct semantics without a full scheduler.)
    e.S2R(3, "SR_TID_X", stall=15, rbar=7)
    e.NOP(stall=15)
    e.NOP(stall=15)

    # Load pointers (param 0/1/2 at 0x210, 0x218, 0x220):
    ldc(8,  0x210, is64=True, stall=15)   # a -> R8:R9
    ldc(10, 0x218, is64=True, stall=15)   # b -> R10:R11
    ldc(12, 0x220, is64=True, stall=15)   # c -> R12:R13

    # R3 <<= 2  (tid * sizeof(float))
    e.IADD3(3, 3, 3, RZ, stall=15)
    e.IADD3(3, 3, 3, RZ, stall=15)
    # zero-extend offset high word
    e.MOV(4, 0, stall=15)

    # a_addr = R8:R9 + R3:R4 -> R16:R17
    e.IADD3(16, 8, 3, RZ, stall=15)
    e.IADD3(17, 9, 4, RZ, stall=15)
    # b_addr -> R18:R19
    e.IADD3(18, 10, 3, RZ, stall=15)
    e.IADD3(19, 11, 4, RZ, stall=15)
    # c_addr -> R20:R21
    e.IADD3(20, 12, 3, RZ, stall=15)
    e.IADD3(21, 13, 4, RZ, stall=15)

    # LDG.E R22, desc[UR4][R16.64]  (a[tid])
    e.LDG(22, 16, size=32, ur_desc=4, stall=15, wbar=0)
    # LDG.E R23, desc[UR4][R18.64]  (b[tid])
    e.LDG(23, 18, size=32, ur_desc=4, stall=15, wbar=1)

    # Wait for both loads
    e.NOP(stall=15, wait_mask=0x03)
    e.NOP(stall=15, wait_mask=0x03)

    # FADD R24 = R22 + R23
    e.FADD(24, 22, 23, stall=15)
    e.NOP(stall=15)

    # STG.E desc[UR4][R20.64], R24
    e.STG(20, 24, size=32, ur_desc=4, stall=15)

    # EXIT
    e.emit(0x000000000000794d, 0x000fea0003800000)

    # Trailing BRA -4 + NOPs (standard postamble)
    e.emit(0xfffffffc00fc7947, 0x000fc0000383ffff)  # BRA -4

    return e.build_cubin()


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    import sys
    import os

    # First test: trivial EXIT-only kernel that loads, launches, and executes.
    print("=" * 60)
    print("Test 1: minimal EXIT kernel")
    print("=" * 60)

    e = SASSEmitter("noop_kernel", max_regs=128)
    e.add_param_u64("ptr")
    e.emit(0x000000000000794d, 0x000fea0003800000)  # EXIT
    noop_cubin = e.build_cubin()
    print(f"Built noop cubin: {len(noop_cubin)} bytes")

    try:
        import ctypes
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cuda_driver import CUDADriver
        gpu = CUDADriver()
        print(f"GPU: {gpu.device_name}")
        mod = gpu.load_cubin_bytes(noop_cubin)
        func = gpu.get_function(mod, "noop_kernel")
        print(f"Loaded kernel: noop_kernel @ {func}")
        gpu.launch(func, grid=(1, 1, 1), block=(32, 1, 1),
                   args=[ctypes.c_uint64(0)])
        gpu.synchronize()
        print("Kernel executed: PASS")
        gpu.close()
    except Exception as ex:
        print(f"noop test error: {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Second test: full vector-add kernel
    print()
    print("=" * 60)
    print("Test 2: vector-add kernel (work-in-progress)")
    print("=" * 60)
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

        gpu = CUDADriver()

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
