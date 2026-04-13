#!/usr/bin/env python3
"""
sass_probe.py -- Lithos SASS Reverse Engineering Tool

Automated protocol for mapping GPU instruction encodings:
  1. Generate minimal PTX kernels containing target instructions
  2. Compile with ptxas -> cubin
  3. Disassemble with nvdisasm -hex -> SASS with encoding
  4. Vary operands systematically to isolate bit fields
  5. Output structured encoding tables

Usage:
    from sass_probe import SASSProbe
    probe = SASSProbe(arch="sm_90")
    result = probe.probe_instruction("add.f32 %f0, %f1, %f2")
    results = probe.probe_all_inference()
    probe.write_encoding_table("ENCODING_AUTO.md")
"""

import os
import re
import struct
import subprocess
import tempfile
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BitField:
    """A named bit range inside a 128-bit SASS instruction."""
    name: str
    hi: int          # inclusive high bit
    lo: int          # inclusive low bit
    value: int = 0   # observed value

    @property
    def width(self):
        return self.hi - self.lo + 1

    def __repr__(self):
        return f"{self.name}[{self.hi}:{self.lo}]={self.value:#x}"


@dataclass
class InstructionEncoding:
    """Decoded encoding for one SASS instruction."""
    ptx_pattern: str = ""
    sass_mnemonic: str = ""
    opcode: int = 0
    opcode_bits: str = "[11:0]"
    form: str = ""           # "reg-reg", "reg-imm", "special"
    inst_word: int = 0       # full 64-bit instruction word
    ctrl_word: int = 0       # full 64-bit control word
    fields: dict = field(default_factory=dict)     # name -> BitField
    modifiers: dict = field(default_factory=dict)   # modifier name -> ctrl bits
    variants: list = field(default_factory=list)    # list of (description, inst, ctrl)
    notes: str = ""

    def to_dict(self):
        d = {
            "ptx_pattern": self.ptx_pattern,
            "sass_mnemonic": self.sass_mnemonic,
            "opcode": f"0x{self.opcode:03x}",
            "form": self.form,
            "inst_word": f"0x{self.inst_word:016x}",
            "ctrl_word": f"0x{self.ctrl_word:016x}",
            "fields": {k: repr(v) for k, v in self.fields.items()},
            "modifiers": {k: f"0x{v:x}" for k, v in self.modifiers.items()},
            "notes": self.notes,
        }
        if self.variants:
            d["variants"] = [
                {"desc": desc, "inst": f"0x{inst:016x}", "ctrl": f"0x{ctrl:016x}"}
                for desc, inst, ctrl in self.variants
            ]
        return d


# ---------------------------------------------------------------------------
# PTX templates
# ---------------------------------------------------------------------------

PTX_HEADER = """\
.version 8.0
.target {arch}
.address_size 64
"""

PTX_KERNEL_WRAP = """\
.visible .entry {name}(
    .param .u64 param_a,
    .param .u64 param_b
)
{{
{regs}
    // Load pointers so ptxas cannot constant-fold
    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
{body}
    ret;
}}
"""


def _ptx_file(arch, kernel_name, reg_decls, body_lines):
    """Build a complete PTX string."""
    regs = "\n".join(f"    {r}" for r in reg_decls)
    body = "\n".join(f"    {l}" for l in body_lines)
    return PTX_HEADER.format(arch=arch) + "\n" + PTX_KERNEL_WRAP.format(
        name=kernel_name, regs=regs, body=body)


# ---------------------------------------------------------------------------
# Core probe engine
# ---------------------------------------------------------------------------

class SASSProbe:
    """Systematic SASS reverse engineering probe."""

    def __init__(self, arch="sm_90", ptxas="ptxas", nvdisasm="nvdisasm",
                 verbose=False, tmpdir=None):
        self.arch = arch
        self.ptxas = ptxas
        self.nvdisasm = nvdisasm
        self.verbose = verbose
        self._tmpdir = tmpdir
        self.results = {}   # category -> list of InstructionEncoding

    # ------------------------------------------------------------------
    # Low-level: compile + disassemble
    # ------------------------------------------------------------------

    def _compile_ptx(self, ptx_source, label="probe"):
        """Compile PTX source -> cubin bytes. Returns (cubin_path, success)."""
        d = self._tmpdir or tempfile.mkdtemp(prefix="sass_probe_")
        ptx_path = os.path.join(d, f"{label}.ptx")
        cubin_path = os.path.join(d, f"{label}.cubin")
        with open(ptx_path, "w") as f:
            f.write(ptx_source)
        cmd = [self.ptxas, "-arch", self.arch, "-o", cubin_path, ptx_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if self.verbose:
                print(f"[ptxas FAIL] {r.stderr.strip()}")
            return None, r.stderr
        return cubin_path, ""

    def _disassemble(self, cubin_path):
        """Disassemble cubin -> list of (offset, mnemonic_line, inst_hex, ctrl_hex)."""
        cmd = [self.nvdisasm, "-hex", cubin_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if self.verbose:
                print(f"[nvdisasm FAIL] {r.stderr.strip()}")
            return []
        return self._parse_sass(r.stdout)

    @staticmethod
    def _parse_sass(sass_text):
        """Parse nvdisasm -hex output into structured records.

        Each SASS instruction in Hopper is two lines:
          /*0050*/   FADD R5, R0, R5 ;    /* 0x0000000500057221 */
                                          /* 0x004fca0000000000 */
        Returns list of dict with keys: offset, mnemonic, full_line, inst, ctrl
        """
        lines = sass_text.split("\n")
        records = []
        i = 0
        # Pattern for instruction line: offset, mnemonic, hex
        inst_pat = re.compile(
            r'/\*([0-9a-fA-F]+)\*/\s+(.+?)\s*;\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/')
        ctrl_pat = re.compile(r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/')

        while i < len(lines):
            m = inst_pat.search(lines[i])
            if m:
                offset = int(m.group(1), 16)
                mnemonic_line = m.group(2).strip()
                inst_hex = int(m.group(3), 16)
                ctrl_hex = 0
                # Next line should have the control word
                if i + 1 < len(lines):
                    cm = ctrl_pat.search(lines[i + 1])
                    if cm:
                        ctrl_hex = int(cm.group(1), 16)
                        i += 1
                records.append({
                    "offset": offset,
                    "mnemonic": mnemonic_line.split()[0] if mnemonic_line else "",
                    "full_line": mnemonic_line,
                    "inst": inst_hex,
                    "ctrl": ctrl_hex,
                })
            i += 1
        return records

    def _compile_and_find(self, ptx_source, target_mnemonic, label="probe"):
        """Compile PTX, disassemble, find all instructions matching mnemonic.

        target_mnemonic can be a string prefix (e.g. "FADD") or a list.
        Returns list of dicts from _parse_sass filtered to target.
        """
        cubin, err = self._compile_ptx(ptx_source, label=label)
        if cubin is None:
            if self.verbose:
                print(f"[compile_and_find] compilation failed: {err}")
            return []
        records = self._disassemble(cubin)
        if isinstance(target_mnemonic, str):
            targets = [target_mnemonic]
        else:
            targets = list(target_mnemonic)
        matches = []
        for rec in records:
            for t in targets:
                if rec["mnemonic"].startswith(t):
                    matches.append(rec)
                    break
        return matches

    # ------------------------------------------------------------------
    # Bit-field extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_bits(value, hi, lo):
        """Extract bits [hi:lo] from an integer (inclusive)."""
        mask = (1 << (hi - lo + 1)) - 1
        return (value >> lo) & mask

    @staticmethod
    def diff_bits(a, b):
        """Return list of bit positions that differ between a and b."""
        xor = a ^ b
        positions = []
        pos = 0
        while xor:
            if xor & 1:
                positions.append(pos)
            xor >>= 1
            pos += 1
        return positions

    def _decode_standard_fields(self, inst, ctrl):
        """Decode standard Hopper register/opcode fields from 128-bit instruction."""
        opcode = inst & 0xFFF
        pred = self.extract_bits(inst, 15, 12)
        rd = self.extract_bits(inst, 23, 16)
        rs1 = self.extract_bits(inst, 31, 24)
        rs2 = self.extract_bits(inst, 39, 32)
        imm32 = self.extract_bits(inst, 63, 32)
        rs3_ctrl = ctrl & 0xFF
        stall = self.extract_bits(ctrl, 57, 53)
        yield_hint = self.extract_bits(ctrl, 52, 52)
        wr_bar = self.extract_bits(ctrl, 51, 49)
        rd_bar = self.extract_bits(ctrl, 48, 46)
        wait_mask = self.extract_bits(ctrl, 45, 42)
        reuse = self.extract_bits(ctrl, 63, 58)
        return {
            "opcode": BitField("opcode", 11, 0, opcode),
            "pred": BitField("pred", 15, 12, pred),
            "rd": BitField("rd", 23, 16, rd),
            "rs1": BitField("rs1", 31, 24, rs1),
            "rs2": BitField("rs2", 39, 32, rs2),
            "imm32": BitField("imm32", 63, 32, imm32),
            "rs3_ctrl": BitField("rs3_ctrl", 71, 64, rs3_ctrl),   # ctrl[7:0]
            "stall": BitField("stall", 121, 117, stall),
            "yield": BitField("yield", 116, 116, yield_hint),
            "wr_bar": BitField("wr_bar", 115, 113, wr_bar),
            "rd_bar": BitField("rd_bar", 112, 110, rd_bar),
            "wait_mask": BitField("wait_mask", 109, 106, wait_mask),
            "reuse": BitField("reuse", 127, 122, reuse),
        }

    # ------------------------------------------------------------------
    # Probe: single arbitrary PTX instruction
    # ------------------------------------------------------------------

    def probe_instruction(self, ptx_instr, target_sass=None):
        """Probe a single PTX instruction.

        ptx_instr: e.g. "add.f32 %f0, %f1, %f2"
        target_sass: expected SASS mnemonic (e.g. "FADD"). Auto-detected if None.

        Returns InstructionEncoding or None.
        """
        # Auto-detect target SASS mnemonic from PTX
        if target_sass is None:
            target_sass = self._guess_sass_mnemonic(ptx_instr)

        # We need to prevent constant folding -- load operands from memory
        reg_decls, setup_lines, instr_line, store_lines = \
            self._wrap_instruction(ptx_instr)

        ptx = _ptx_file(
            self.arch, "sass_probe_single", reg_decls,
            setup_lines + [instr_line] + store_lines)

        matches = self._compile_and_find(ptx, target_sass, label="single")
        if not matches:
            if self.verbose:
                print(f"[probe_instruction] no {target_sass} found for: {ptx_instr}")
            return None

        rec = matches[0]
        fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
        enc = InstructionEncoding(
            ptx_pattern=ptx_instr,
            sass_mnemonic=rec["mnemonic"],
            opcode=fields["opcode"].value,
            inst_word=rec["inst"],
            ctrl_word=rec["ctrl"],
            fields=fields,
        )
        # Determine form
        if enc.opcode & 0xE00 in (0x400, 0x800):
            enc.form = "reg-imm"
        elif enc.opcode < 0x400:
            enc.form = "reg-reg"
        else:
            enc.form = "special"
        return enc

    def _guess_sass_mnemonic(self, ptx_instr):
        """Heuristic map from PTX mnemonic to SASS mnemonic."""
        ptx_op = ptx_instr.strip().split()[0].split(".")[0]
        mapping = {
            "add": "FADD", "mul": "FMUL", "fma": "FFMA",
            "min": "FMNMX", "max": "FMNMX", "neg": "FADD", "abs": "FADD",
            "rcp": "MUFU", "rsqrt": "MUFU", "sqrt": "MUFU",
            "sin": "MUFU", "cos": "MUFU", "ex2": "MUFU", "lg2": "MUFU",
            "setp": "FSETP", "selp": "FSEL",
            "ld": "LDG", "st": "STG",
            "mov": "MOV", "cvt": "F2F",
            "mad": "IMAD", "shl": "SHF", "shr": "SHF",
            "bar": "BAR", "membar": "MEMBAR",
            "shfl": "SHFL",
            "mma": "HMMA",
            "cp": "LDGSTS",
        }
        return mapping.get(ptx_op, ptx_op.upper())

    def _wrap_instruction(self, ptx_instr):
        """Wrap a PTX instruction with loads/stores to prevent constant folding.

        Returns (reg_decls, setup_lines, instruction, store_lines).
        """
        # Detect register types from the instruction
        has_f32 = ".f32" in ptx_instr or "%f" in ptx_instr
        has_f16 = ".f16" in ptx_instr or "%h" in ptx_instr
        has_pred = "%p" in ptx_instr or "setp" in ptx_instr
        has_u32 = ".u32" in ptx_instr or ".s32" in ptx_instr or "%r" in ptx_instr

        reg_decls = [
            ".reg .f32 %f<20>;",
            ".reg .f16 %h<8>;",
            ".reg .b16 %rh<8>;",
            ".reg .pred %p<8>;",
            ".reg .b32 %r<20>;",
            ".reg .b64 %rd<8>;",
        ]
        setup = [
            "// Load from memory to prevent constant folding",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "ld.global.f32 %f2, [%rd0+8];",
            "ld.global.f32 %f3, [%rd0+12];",
            "ld.global.b32 %r0, [%rd0+16];",
            "ld.global.b32 %r1, [%rd0+20];",
        ]
        stores = [
            "// Store results to prevent dead-code elimination",
        ]
        # Detect destination register
        parts = ptx_instr.split(",")
        first_tok = parts[0].strip().split()
        if len(first_tok) >= 2:
            dest = first_tok[-1].strip()
            if dest.startswith("%f"):
                stores.append(f"st.global.f32 [%rd1], {dest};")
            elif dest.startswith("%p"):
                stores.append(f"@{dest} st.global.f32 [%rd1], %f0;")
            elif dest.startswith("%r") and not dest.startswith("%rd"):
                stores.append(f"st.global.b32 [%rd1], {dest};")
            else:
                stores.append(f"st.global.f32 [%rd1], %f0;")
        else:
            stores.append("st.global.f32 [%rd1], %f0;")

        return reg_decls, setup, ptx_instr + ";", stores

    # ------------------------------------------------------------------
    # Systematic operand variation
    # ------------------------------------------------------------------

    def _vary_registers(self, ptx_template, target_sass, variations, label="vary"):
        """Vary register operands to confirm bit positions.

        ptx_template: format string with {rd}, {rs1}, {rs2} placeholders
        variations: list of dicts like {"rd": 5, "rs1": 6, "rs2": 9}
        Returns list of (description, inst, ctrl) tuples.
        """
        results = []
        for i, var in enumerate(variations):
            ptx_instr = ptx_template.format(**var)
            reg_decls, setup, instr, stores = self._wrap_instruction(ptx_instr)
            ptx = _ptx_file(self.arch, f"vary_{label}_{i}", reg_decls,
                            setup + [instr] + stores)
            matches = self._compile_and_find(ptx, target_sass, label=f"{label}_{i}")
            if matches:
                rec = matches[0]
                desc = ", ".join(f"{k}={v}" for k, v in var.items())
                results.append((desc, rec["inst"], rec["ctrl"]))
                if self.verbose:
                    print(f"  [{label}] {desc}: inst=0x{rec['inst']:016x} "
                          f"ctrl=0x{rec['ctrl']:016x}")
        return results

    # ------------------------------------------------------------------
    # Category 1: Float Arithmetic (FADD, FMUL, FFMA, FMNMX, FSETP)
    # ------------------------------------------------------------------

    def probe_float_arithmetic(self):
        """Probe category 1: floating-point arithmetic instructions."""
        results = []

        # --- FADD reg-reg ---
        ptx = _ptx_file(self.arch, "probe_fadd_rr", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "add.f32 %f2, %f0, %f1;",
            "st.global.f32 [%rd1], %f2;",
        ])
        matches = self._compile_and_find(ptx, "FADD", label="fadd_rr")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="add.f32 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # --- FADD reg-imm ---
        ptx = _ptx_file(self.arch, "probe_fadd_ri", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "add.f32 %f1, %f0, 0f3F800000;",   # + 1.0
            "st.global.f32 [%rd1], %f1;",
        ])
        matches = self._compile_and_find(ptx, "FADD", label="fadd_ri")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="add.f32 rd, rs1, 1.0",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-imm", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # --- FMUL reg-reg ---
        ptx = _ptx_file(self.arch, "probe_fmul_rr", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "mul.f32 %f2, %f0, %f1;",
            "st.global.f32 [%rd1], %f2;",
        ])
        matches = self._compile_and_find(ptx, "FMUL", label="fmul_rr")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="mul.f32 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # --- FFMA ---
        ptx = _ptx_file(self.arch, "probe_ffma", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "ld.global.f32 %f2, [%rd0+8];",
            "fma.rn.f32 %f3, %f0, %f1, %f2;",
            "st.global.f32 [%rd1], %f3;",
        ])
        matches = self._compile_and_find(ptx, "FFMA", label="ffma")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="fma.rn.f32 rd, rs1, rs2, rs3",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="rs3 in ctrl[7:0]")
            results.append(enc)

        # --- FMNMX (min) ---
        ptx = _ptx_file(self.arch, "probe_fmin", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "min.f32 %f2, %f0, %f1;",
            "st.global.f32 [%rd1], %f2;",
        ])
        matches = self._compile_and_find(ptx, "FMNMX", label="fmin")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="min.f32 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="FMNMX with PT -> min, !PT -> max. Select pred in ctrl[23:20]")
            results.append(enc)

        # --- FMNMX (max) ---
        ptx = _ptx_file(self.arch, "probe_fmax", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "max.f32 %f2, %f0, %f1;",
            "st.global.f32 [%rd1], %f2;",
        ])
        matches = self._compile_and_find(ptx, "FMNMX", label="fmax")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="max.f32 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="FMNMX with !PT in ctrl[23:20] -> max")
            results.append(enc)

        # --- FSETP ---
        ptx = _ptx_file(self.arch, "probe_fsetp", [
            ".reg .f32 %f<20>;", ".reg .pred %p<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "setp.gt.f32 %p0, %f0, %f1;",
            "@%p0 st.global.f32 [%rd1], %f0;",
        ])
        matches = self._compile_and_find(ptx, "FSETP", label="fsetp")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="setp.gt.f32 pd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="Dest pred in ctrl[19:16], comparison type in ctrl[15:12]")
            results.append(enc)

        self.results["1_float_arithmetic"] = results
        return results

    # ------------------------------------------------------------------
    # Category 2: Special Functions (MUFU)
    # ------------------------------------------------------------------

    def probe_special_functions(self):
        """Probe category 2: MUFU sub-functions."""
        results = []
        mufu_funcs = [
            ("cos.approx.f32", "MUFU.COS"),
            ("sin.approx.f32", "MUFU.SIN"),
            ("ex2.approx.f32", "MUFU.EX2"),
            ("lg2.approx.f32", "MUFU.LG2"),
            ("rcp.approx.f32", "MUFU.RCP"),
            ("rsqrt.approx.f32", "MUFU.RSQ"),
            ("sqrt.approx.f32", "MUFU.SQRT"),
        ]
        for ptx_func, sass_name in mufu_funcs:
            ptx_op = ptx_func.split(".")[0]
            ptx = _ptx_file(self.arch, f"probe_mufu_{ptx_op}", [
                ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
            ], [
                "ld.param.u64 %rd0, [param_a];",
                "ld.param.u64 %rd1, [param_b];",
                "ld.global.f32 %f0, [%rd0];",
                f"{ptx_func} %f1, %f0;",
                "st.global.f32 [%rd1], %f1;",
            ])
            matches = self._compile_and_find(ptx, "MUFU", label=f"mufu_{ptx_op}")
            if matches:
                rec = matches[0]
                fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
                subfunc = self.extract_bits(rec["ctrl"], 13, 10)
                enc = InstructionEncoding(
                    ptx_pattern=f"{ptx_func} rd, rs1",
                    sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                    form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                    fields=fields,
                    modifiers={"subfunc_ctrl_13_10": subfunc},
                    notes=f"{sass_name}: ctrl[13:10]=0x{subfunc:x}")
                results.append(enc)

        self.results["2_special_functions"] = results
        return results

    # ------------------------------------------------------------------
    # Category 3: Half Precision (HADD2, HMUL2, HFMA2)
    # ------------------------------------------------------------------

    def probe_half_precision(self):
        """Probe category 3: FP16x2 packed operations."""
        results = []

        # HADD2
        ptx = _ptx_file(self.arch, "probe_hadd2", [
            ".reg .f16x2 %hh<8>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.b32 %r0, [%rd0];",
            "ld.global.b32 %r1, [%rd0+4];",
            "mov.b32 %hh0, %r0;",
            "mov.b32 %hh1, %r1;",
            "add.rn.f16x2 %hh2, %hh0, %hh1;",
            "mov.b32 %r2, %hh2;",
            "st.global.b32 [%rd1], %r2;",
        ])
        matches = self._compile_and_find(ptx, ["HADD2", "HFMA2"], label="hadd2")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="add.rn.f16x2 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # HMUL2
        ptx = _ptx_file(self.arch, "probe_hmul2", [
            ".reg .f16x2 %hh<8>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.b32 %r0, [%rd0];",
            "ld.global.b32 %r1, [%rd0+4];",
            "mov.b32 %hh0, %r0;",
            "mov.b32 %hh1, %r1;",
            "mul.rn.f16x2 %hh2, %hh0, %hh1;",
            "mov.b32 %r2, %hh2;",
            "st.global.b32 [%rd1], %r2;",
        ])
        matches = self._compile_and_find(ptx, ["HMUL2", "HFMA2"], label="hmul2")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="mul.rn.f16x2 rd, rs1, rs2",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # HFMA2
        ptx = _ptx_file(self.arch, "probe_hfma2", [
            ".reg .f16x2 %hh<8>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.b32 %r0, [%rd0];",
            "ld.global.b32 %r1, [%rd0+4];",
            "ld.global.b32 %r2, [%rd0+8];",
            "mov.b32 %hh0, %r0;",
            "mov.b32 %hh1, %r1;",
            "mov.b32 %hh2, %r2;",
            "fma.rn.f16x2 %hh3, %hh0, %hh1, %hh2;",
            "mov.b32 %r3, %hh3;",
            "st.global.b32 [%rd1], %r3;",
        ])
        matches = self._compile_and_find(ptx, "HFMA2", label="hfma2")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="fma.rn.f16x2 rd, rs1, rs2, rs3",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="reg-reg", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields, notes="rs3 in ctrl[7:0]")
            results.append(enc)

        self.results["3_half_precision"] = results
        return results

    # ------------------------------------------------------------------
    # Category 4: Tensor Core (HMMA, LDMATRIX)
    # ------------------------------------------------------------------

    def probe_tensor_core(self):
        """Probe category 4: tensor core instructions."""
        results = []

        # HMMA.16816.F32 via mma.sync
        ptx = _ptx_file(self.arch, "probe_hmma", [
            ".reg .f32 %f<20>;",
            ".reg .b32 %r<20>;",
            ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.b32 %r0, [%rd0];",
            "ld.global.b32 %r1, [%rd0+4];",
            "ld.global.b32 %r2, [%rd0+8];",
            "ld.global.b32 %r3, [%rd0+12];",
            "ld.global.b32 %r4, [%rd0+16];",
            "ld.global.b32 %r5, [%rd0+20];",
            "// Zero accumulators",
            "mov.f32 %f0, 0f00000000;",
            "mov.f32 %f1, 0f00000000;",
            "mov.f32 %f2, 0f00000000;",
            "mov.f32 %f3, 0f00000000;",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "    {%f0, %f1, %f2, %f3},",
            "    {%r0, %r1, %r2, %r3},",
            "    {%r4, %r5},",
            "    {%f0, %f1, %f2, %f3};",
            "st.global.f32 [%rd1], %f0;",
            "st.global.f32 [%rd1+4], %f1;",
            "st.global.f32 [%rd1+8], %f2;",
            "st.global.f32 [%rd1+12], %f3;",
        ])
        matches = self._compile_and_find(ptx, "HMMA", label="hmma")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="Shape/type in ctrl[15:8], accumulator in ctrl[7:0]")
            results.append(enc)

        # LDMATRIX (LDSM) via ldmatrix.sync
        ptx = _ptx_file(self.arch, "probe_ldsm", [
            ".reg .b32 %r<20>;",
            ".reg .b64 %rd<4>;",
            ".shared .align 16 .b8 smem[1024];",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "mov.u64 %rd2, smem;",
            "// Use tid to compute smem address",
            "mov.u32 %r0, %tid.x;",
            "shl.b32 %r1, %r0, 4;",
            "cvt.u64.u32 %rd3, %r1;",
            "add.u64 %rd2, %rd2, %rd3;",
            "// Cast to .shared address",
            "cvta.to.shared.u64 %rd2, %rd2;",
            "cvt.u32.u64 %r2, %rd2;",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r3, %r4, %r5, %r6}, [%r2];",
            "st.global.b32 [%rd1], %r3;",
            "st.global.b32 [%rd1+4], %r4;",
        ])
        matches = self._compile_and_find(ptx, "LDSM", label="ldsm")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        self.results["4_tensor_core"] = results
        return results

    # ------------------------------------------------------------------
    # Category 5: Shared Memory (LDS, STS, LDSM)
    # ------------------------------------------------------------------

    def probe_shared_memory(self):
        """Probe category 5: shared memory instructions."""
        results = []

        for size_suffix, ptx_type, sass_suffix in [
            ("", ".f32", "32"), (".64", ".v2.f32", "64"), (".128", ".v4.f32", "128")
        ]:
            ptx = _ptx_file(self.arch, f"probe_lds{sass_suffix}", [
                ".reg .f32 %f<20>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
                ".shared .align 16 .b8 smem[4096];",
            ], [
                "ld.param.u64 %rd0, [param_a];",
                "ld.param.u64 %rd1, [param_b];",
                "// Compute shared memory address",
                "mov.u32 %r0, %tid.x;",
                "shl.b32 %r1, %r0, 4;",
                f"ld.shared{ptx_type} %f0, [%r1];",
                "st.global.f32 [%rd1], %f0;",
            ])
            # LDS.64 and LDS.128 load multiple floats, so adjust
            if sass_suffix == "64":
                ptx = _ptx_file(self.arch, f"probe_lds{sass_suffix}", [
                    ".reg .f32 %f<20>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
                    ".shared .align 16 .b8 smem[4096];",
                ], [
                    "ld.param.u64 %rd0, [param_a];",
                    "ld.param.u64 %rd1, [param_b];",
                    "mov.u32 %r0, %tid.x;",
                    "shl.b32 %r1, %r0, 4;",
                    "ld.shared.v2.f32 {%f0, %f1}, [%r1];",
                    "st.global.f32 [%rd1], %f0;",
                    "st.global.f32 [%rd1+4], %f1;",
                ])
            elif sass_suffix == "128":
                ptx = _ptx_file(self.arch, f"probe_lds{sass_suffix}", [
                    ".reg .f32 %f<20>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
                    ".shared .align 16 .b8 smem[4096];",
                ], [
                    "ld.param.u64 %rd0, [param_a];",
                    "ld.param.u64 %rd1, [param_b];",
                    "mov.u32 %r0, %tid.x;",
                    "shl.b32 %r1, %r0, 4;",
                    "ld.shared.v4.f32 {%f0, %f1, %f2, %f3}, [%r1];",
                    "st.global.f32 [%rd1], %f0;",
                    "st.global.f32 [%rd1+4], %f1;",
                ])
            matches = self._compile_and_find(ptx, "LDS", label=f"lds{sass_suffix}")
            if matches:
                rec = matches[0]
                fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
                enc = InstructionEncoding(
                    ptx_pattern=f"ld.shared{size_suffix} (32/64/128-bit)",
                    sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                    form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                    fields=fields,
                    notes=f"LDS {sass_suffix}-bit, size in ctrl[13:10]")
                results.append(enc)

        # STS
        ptx = _ptx_file(self.arch, "probe_sts", [
            ".reg .f32 %f<20>;", ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
            ".shared .align 16 .b8 smem[4096];",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.global.f32 %f0, [%rd0];",
            "mov.u32 %r0, %tid.x;",
            "shl.b32 %r1, %r0, 4;",
            "st.shared.f32 [%r1], %f0;",
        ])
        matches = self._compile_and_find(ptx, "STS", label="sts")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="st.shared.f32 [addr], rs",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        self.results["5_shared_memory"] = results
        return results

    # ------------------------------------------------------------------
    # Category 6: Warp Shuffle (SHFL modes)
    # ------------------------------------------------------------------

    def probe_warp_shuffle(self):
        """Probe category 6: SHFL instruction modes.

        SHFL requires non-constant-foldable source (e.g. %tid.x) and uses
        the membermask syntax: shfl.sync.MODE.b32 dst|pred, src, lane, clamp, mask
        """
        results = []

        # Build one kernel with all four modes using %tid.x to prevent folding
        shfl_modes = [
            ("bfly", ".BFLY", "1"),
            ("down", ".DOWN", "1"),
            ("up", ".UP", "1"),
            ("idx", ".IDX", "0"),
        ]

        ptx = _ptx_file(self.arch, "probe_shfl_all", [
            ".reg .b32 %r<20>;",
            ".reg .pred %p<8>;",
            ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "// Use tid.x as source to prevent constant folding",
            "mov.u32 %r0, %tid.x;",
            "// Four SHFL modes",
            "shfl.sync.bfly.b32 %r1|%p0, %r0, 1, 0x1f, 0xFFFFFFFF;",
            "shfl.sync.down.b32 %r2|%p1, %r0, 1, 0x1f, 0xFFFFFFFF;",
            "shfl.sync.up.b32   %r3|%p2, %r0, 1, 0x1f, 0xFFFFFFFF;",
            "shfl.sync.idx.b32  %r4|%p3, %r0, 0, 0x1f, 0xFFFFFFFF;",
            "// Store all results to prevent DCE",
            "st.global.u32 [%rd1], %r1;",
            "st.global.u32 [%rd1+4], %r2;",
            "st.global.u32 [%rd1+8], %r3;",
            "st.global.u32 [%rd1+12], %r4;",
        ])
        matches = self._compile_and_find(ptx, "SHFL", label="shfl_all")

        for rec in matches:
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            mode_bits = self.extract_bits(rec["inst"], 59, 58)
            mode_names = {0: ".IDX", 1: ".UP", 2: ".DOWN", 3: ".BFLY"}
            sass_mode = mode_names.get(mode_bits, f".?{mode_bits}")
            opcode = fields["opcode"].value
            # Determine form from opcode
            form = "imm" if opcode == 0xf89 else "reg"
            enc = InstructionEncoding(
                ptx_pattern=f"shfl.sync{sass_mode.lower()}.b32",
                sass_mnemonic=rec["mnemonic"], opcode=opcode,
                form=form,
                inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                modifiers={"mode_bits_59_58": mode_bits},
                notes=f"SHFL{sass_mode}: inst[59:58]=0b{mode_bits:02b}, "
                      f"opcode 0x{opcode:03x} ({form})")
            results.append(enc)

        self.results["6_warp_shuffle"] = results
        return results

    # ------------------------------------------------------------------
    # Category 7: Synchronization (BAR, DEPBAR, MEMBAR)
    # ------------------------------------------------------------------

    def probe_synchronization(self):
        """Probe category 7: synchronization instructions."""
        results = []

        # BAR.SYNC
        ptx = _ptx_file(self.arch, "probe_bar", [
            ".reg .b32 %r<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "bar.sync 0;",
            "ld.global.b32 %r0, [%rd0];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "BAR", label="bar")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="bar.sync 0",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        # MEMBAR.CTA
        ptx = _ptx_file(self.arch, "probe_membar_cta", [
            ".reg .b32 %r<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "membar.cta;",
            "ld.global.b32 %r0, [%rd0];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "MEMBAR", label="membar_cta")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            scope = self.extract_bits(rec["ctrl"], 13, 12)
            enc = InstructionEncoding(
                ptx_pattern="membar.cta",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                modifiers={"scope_ctrl_13_12": scope},
                notes=f"MEMBAR.CTA: ctrl[13:12]=0x{scope:x}")
            results.append(enc)

        # MEMBAR.GL (GPU-level)
        ptx = _ptx_file(self.arch, "probe_membar_gl", [
            ".reg .b32 %r<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "membar.gl;",
            "ld.global.b32 %r0, [%rd0];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "MEMBAR", label="membar_gl")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            scope = self.extract_bits(rec["ctrl"], 13, 12)
            enc = InstructionEncoding(
                ptx_pattern="membar.gl",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                modifiers={"scope_ctrl_13_12": scope},
                notes=f"MEMBAR.GPU: ctrl[13:12]=0x{scope:x}")
            results.append(enc)

        # MEMBAR.SYS
        ptx = _ptx_file(self.arch, "probe_membar_sys", [
            ".reg .b32 %r<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "membar.sys;",
            "ld.global.b32 %r0, [%rd0];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "MEMBAR", label="membar_sys")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            scope = self.extract_bits(rec["ctrl"], 13, 12)
            enc = InstructionEncoding(
                ptx_pattern="membar.sys",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                modifiers={"scope_ctrl_13_12": scope},
                notes=f"MEMBAR.SYS: ctrl[13:12]=0x{scope:x}")
            results.append(enc)

        # DEPBAR
        ptx = _ptx_file(self.arch, "probe_depbar", [
            ".reg .b32 %r<4>;", ".reg .b64 %rd<4>;",
            ".shared .align 16 .b8 smem[256];",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "// cp.async to trigger DEPBAR emission",
            "cp.async.ca.shared.global [smem], [%rd0], 4;",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            "ld.shared.b32 %r0, [smem];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "DEPBAR", label="depbar")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="cp.async.wait_group -> DEPBAR",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        self.results["7_synchronization"] = results
        return results

    # ------------------------------------------------------------------
    # Category 8: Async Copy (LDGSTS)
    # ------------------------------------------------------------------

    def probe_async_copy(self):
        """Probe category 8: asynchronous copy instructions."""
        results = []

        ptx = _ptx_file(self.arch, "probe_ldgsts", [
            ".reg .b32 %r<8>;", ".reg .b64 %rd<4>;",
            ".shared .align 16 .b8 smem[1024];",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "cp.async.ca.shared.global [smem], [%rd0], 16;",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            "ld.shared.v4.b32 {%r0, %r1, %r2, %r3}, [smem];",
            "st.global.b32 [%rd1], %r0;",
        ])
        matches = self._compile_and_find(ptx, "LDGSTS", label="ldgsts")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="cp.async.ca.shared.global [smem], [global], 16",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields,
                notes="LDGSTS: combined global load + shared store")
            results.append(enc)

        # Also look for LDGDEPBAR
        matches = self._compile_and_find(ptx, "LDGDEPBAR", label="ldgdepbar")
        if matches:
            rec = matches[0]
            fields = self._decode_standard_fields(rec["inst"], rec["ctrl"])
            enc = InstructionEncoding(
                ptx_pattern="(compiler-generated after cp.async)",
                sass_mnemonic=rec["mnemonic"], opcode=fields["opcode"].value,
                form="special", inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields=fields)
            results.append(enc)

        self.results["8_async_copy"] = results
        return results

    # ------------------------------------------------------------------
    # Category 9: Register Field Mapping
    # ------------------------------------------------------------------

    def probe_register_fields(self):
        """Probe category 9: systematically verify register bit positions.

        Since ptxas performs register allocation (PTX virtual regs != SASS physical regs),
        we use multiple distinct FADD instructions and read the physical register numbers
        from the SASS encoding to confirm the field layout. We also use a kernel with
        many independent operations to force distinct physical registers.
        """
        results = []

        # Build a kernel with many independent FADDs to get distinct register assignments
        ptx = _ptx_file(self.arch, "probe_regmap", [
            ".reg .f32 %f<40>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "// Load many distinct values to force distinct physical registers",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "ld.global.f32 %f2, [%rd0+8];",
            "ld.global.f32 %f3, [%rd0+12];",
            "ld.global.f32 %f4, [%rd0+16];",
            "ld.global.f32 %f5, [%rd0+20];",
            "// Several FADDs with different register combos",
            "add.f32 %f10, %f0, %f1;",
            "add.f32 %f11, %f2, %f3;",
            "add.f32 %f12, %f4, %f5;",
            "add.f32 %f13, %f10, %f11;",
            "add.f32 %f14, %f12, %f13;",
            "// Store all to prevent DCE",
            "st.global.f32 [%rd1], %f10;",
            "st.global.f32 [%rd1+4], %f11;",
            "st.global.f32 [%rd1+8], %f12;",
            "st.global.f32 [%rd1+12], %f13;",
            "st.global.f32 [%rd1+16], %f14;",
        ])
        matches = self._compile_and_find(ptx, "FADD", label="regmap")

        variants = []
        for rec in matches:
            rd = self.extract_bits(rec["inst"], 23, 16)
            rs1 = self.extract_bits(rec["inst"], 31, 24)
            rs2 = self.extract_bits(rec["inst"], 39, 32)
            desc = f"{rec['full_line']}"
            variants.append((desc, rec["inst"], rec["ctrl"]))
            if self.verbose:
                print(f"  {rec['full_line']}: Rd=R{rd} Rs1=R{rs1} Rs2=R{rs2}")

        if variants:
            enc = InstructionEncoding(
                ptx_pattern="add.f32 (multiple, varied register assignments)",
                sass_mnemonic="FADD",
                opcode=0x221,
                form="reg-reg",
                fields={"rd": BitField("rd", 23, 16),
                        "rs1": BitField("rs1", 31, 24),
                        "rs2": BitField("rs2", 39, 32)},
                variants=variants,
                notes="Rd in inst[23:16], Rs1 in inst[31:24], Rs2 in inst[39:32]. "
                      "8-bit fields, R255 (0xFF) = RZ. "
                      "Physical register numbers assigned by ptxas register allocator.")
            results.append(enc)

        # Verify RZ = 0xFF via abs.f32 -> FADD Rd, |Rs|, -RZ
        ptx = _ptx_file(self.arch, "probe_rz_check", [
            ".reg .f32 %f<4>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "abs.f32 %f1, %f0;",
            "st.global.f32 [%rd1], %f1;",
        ])
        matches = self._compile_and_find(ptx, "FADD", label="rz_check")
        if matches:
            rec = matches[0]
            rs2 = self.extract_bits(rec["inst"], 39, 32)
            inst_bit63 = self.extract_bits(rec["inst"], 63, 63)
            ctrl_bit9 = self.extract_bits(rec["ctrl"], 9, 9)
            enc = InstructionEncoding(
                ptx_pattern="abs.f32 -> FADD Rd, |Rs1|, -RZ",
                sass_mnemonic=rec["mnemonic"],
                opcode=0x221,
                inst_word=rec["inst"], ctrl_word=rec["ctrl"],
                fields={"rs2_RZ": BitField("rs2", 39, 32, rs2)},
                notes=f"RZ=0x{rs2:02x} (expect 0xFF), "
                      f"inst[63]=negate_src2={inst_bit63}, "
                      f"ctrl[9]=abs_src1={ctrl_bit9}")
            results.append(enc)

        # Verify 4-operand format (FFMA): rs3 in ctrl[7:0]
        ptx = _ptx_file(self.arch, "probe_rs3_check", [
            ".reg .f32 %f<20>;", ".reg .b64 %rd<4>;",
        ], [
            "ld.param.u64 %rd0, [param_a];",
            "ld.param.u64 %rd1, [param_b];",
            "ld.global.f32 %f0, [%rd0];",
            "ld.global.f32 %f1, [%rd0+4];",
            "ld.global.f32 %f2, [%rd0+8];",
            "ld.global.f32 %f3, [%rd0+12];",
            "fma.rn.f32 %f4, %f0, %f1, %f2;",
            "fma.rn.f32 %f5, %f2, %f3, %f0;",
            "st.global.f32 [%rd1], %f4;",
            "st.global.f32 [%rd1+4], %f5;",
        ])
        matches = self._compile_and_find(ptx, "FFMA", label="rs3_check")
        ffma_variants = []
        for rec in matches:
            rd = self.extract_bits(rec["inst"], 23, 16)
            rs1 = self.extract_bits(rec["inst"], 31, 24)
            rs2 = self.extract_bits(rec["inst"], 39, 32)
            rs3 = rec["ctrl"] & 0xFF
            desc = f"{rec['full_line']} (rs3=ctrl[7:0]=R{rs3})"
            ffma_variants.append((desc, rec["inst"], rec["ctrl"]))
            if self.verbose:
                print(f"  FFMA: Rd=R{rd} Rs1=R{rs1} Rs2=R{rs2} Rs3=R{rs3}")

        if ffma_variants:
            enc = InstructionEncoding(
                ptx_pattern="fma.rn.f32 (rs3 location check)",
                sass_mnemonic="FFMA",
                opcode=0x223,
                form="reg-reg",
                fields={"rd": BitField("rd", 23, 16),
                        "rs1": BitField("rs1", 31, 24),
                        "rs2": BitField("rs2", 39, 32),
                        "rs3": BitField("rs3_ctrl", 71, 64)},
                variants=ffma_variants,
                notes="4-operand: rs3 in ctrl[7:0] (bits 71:64 of 128-bit instruction)")
            results.append(enc)

        self.results["9_register_fields"] = results
        return results

    # ------------------------------------------------------------------
    # Category 10: Immediate Encoding
    # ------------------------------------------------------------------

    def probe_immediate_encoding(self):
        """Probe category 10: immediate value encoding in inst[63:32]."""
        results = []

        test_values = [
            ("1.0", "0f3F800000", 0x3F800000),
            ("2.0", "0f40000000", 0x40000000),
            ("0.5", "0f3F000000", 0x3F000000),
            ("-1.0", "0fBF800000", 0xBF800000),
            ("0.25", "0f3E800000", 0x3E800000),
        ]
        for name, ptx_hex, expected in test_values:
            ptx = _ptx_file(self.arch, f"probe_imm_{name.replace('.','_').replace('-','neg')}", [
                ".reg .f32 %f<4>;", ".reg .b64 %rd<4>;",
            ], [
                "ld.param.u64 %rd0, [param_a];",
                "ld.param.u64 %rd1, [param_b];",
                "ld.global.f32 %f0, [%rd0];",
                f"add.f32 %f1, %f0, {ptx_hex};",
                "st.global.f32 [%rd1], %f1;",
            ])
            matches = self._compile_and_find(ptx, "FADD", label=f"imm_{name}")
            if matches:
                rec = matches[0]
                imm = self.extract_bits(rec["inst"], 63, 32)
                results.append((
                    f"FADD Rd, Rs, {name}",
                    rec["inst"], rec["ctrl"],
                    f"inst[63:32]=0x{imm:08X} (expected 0x{expected:08X}, "
                    f"{'MATCH' if imm == expected else 'MISMATCH'})"
                ))

        # Wrap in encoding
        enc = InstructionEncoding(
            ptx_pattern="add.f32 rd, rs, <float_imm>",
            sass_mnemonic="FADD",
            form="reg-imm",
            fields={"imm32": BitField("imm32", 63, 32)},
            variants=[(desc, inst, ctrl) for desc, inst, ctrl, _ in results],
            notes="IEEE 754 float immediate in inst[63:32]. " +
                  "; ".join(note for _, _, _, note in results))
        self.results["10_immediate_encoding"] = [enc]
        return [enc]

    # ------------------------------------------------------------------
    # Category 11: Predication
    # ------------------------------------------------------------------

    def probe_predication(self):
        """Probe category 11: predicate guard encoding in inst[15:12].

        We compile one kernel per predicate variant, each with a single
        conditional EXIT instruction guarded by the predicate. EXIT cannot
        be optimized away by ptxas, and the predicate value is directly
        observable in the instruction encoding.
        """
        results = []

        pred_tests = [
            ("@%p0", "P0", 0x0),
            ("@%p1", "P1", 0x1),
            ("@%p2", "P2", 0x2),
            ("@%p3", "P3", 0x3),
            ("@!%p0", "!P0", 0x8),
            ("@!%p1", "!P1", 0x9),
            ("@!%p2", "!P2", 0xA),
        ]
        for ptx_guard, name, expected in pred_tests:
            # Use conditional EXIT -- compiler cannot remove it
            ptx = _ptx_file(self.arch, f"probe_pred_{name.replace('!','not')}", [
                ".reg .f32 %f<8>;", ".reg .pred %p<8>;", ".reg .b64 %rd<4>;",
            ], [
                "ld.param.u64 %rd0, [param_a];",
                "ld.param.u64 %rd1, [param_b];",
                "ld.global.f32 %f0, [%rd0];",
                "ld.global.f32 %f1, [%rd0+4];",
                "setp.gt.f32 %p0, %f0, %f1;",
                "setp.lt.f32 %p1, %f0, %f1;",
                "setp.eq.f32 %p2, %f0, %f1;",
                "setp.ne.f32 %p3, %f0, %f1;",
                f"{ptx_guard} bra SKIP;",
                "st.global.f32 [%rd1], %f0;",
                "SKIP:",
                "st.global.f32 [%rd1+4], %f1;",
            ])
            # Look for the predicated BRA instruction
            cubin, err = self._compile_ptx(ptx, label=f"pred_{name}")
            if cubin is None:
                continue
            records = self._disassemble(cubin)
            # Find BRA (or EXIT) with non-PT predicate
            for rec in records:
                pred = self.extract_bits(rec["inst"], 15, 12)
                # Look for instructions with our expected predicate value
                if pred == expected and rec["mnemonic"] in ("BRA", "EXIT", "BRX"):
                    results.append((
                        f"{ptx_guard} {rec['mnemonic']}",
                        rec["inst"], rec["ctrl"],
                        f"inst[15:12]=0x{pred:X} (expected 0x{expected:X}, MATCH)"
                    ))
                    break
            else:
                # Fallback: find any instruction with non-PT (0x7) predicate
                for rec in records:
                    pred = self.extract_bits(rec["inst"], 15, 12)
                    if pred != 0x7:  # Not PT
                        results.append((
                            f"{ptx_guard} {rec['mnemonic']}",
                            rec["inst"], rec["ctrl"],
                            f"inst[15:12]=0x{pred:X} (expected 0x{expected:X}, "
                            f"{'MATCH' if pred == expected else 'MAPPED_DIFF'})"
                        ))
                        break

        # Collect observed predicate values to confirm bit field layout
        observed_preds = set()
        for _, inst, _, _ in results:
            observed_preds.add(self.extract_bits(inst, 15, 12))

        enc = InstructionEncoding(
            ptx_pattern="@Px / @!Px guard on instructions",
            sass_mnemonic="(any)",
            fields={"pred_guard": BitField("pred", 15, 12)},
            variants=[(desc, inst, ctrl) for desc, inst, ctrl, _ in results],
            notes="Pred guard in inst[15:12]: bit[15]=negate, bits[14:12]=pred reg index, "
                  "PT=0x7 (always true, unpredicated). "
                  "Note: ptxas remaps PTX predicate registers to physical SASS predicates. "
                  f"Observed physical pred values: {sorted(observed_preds)}. "
                  "Format: 0x0-0x6 = @P0-@P6, 0x8-0xE = @!P0-@!P6, 0x7 = PT, 0xF = !PT.")
        self.results["11_predication"] = [enc]
        return [enc]

    # ------------------------------------------------------------------
    # Probe all 11 categories
    # ------------------------------------------------------------------

    def probe_all_inference(self):
        """Run all 11 probe categories needed for inference kernels."""
        print(f"=== SASS Probe: {self.arch} ===")
        print()

        categories = [
            ("1. Float Arithmetic", self.probe_float_arithmetic),
            ("2. Special Functions (MUFU)", self.probe_special_functions),
            ("3. Half Precision", self.probe_half_precision),
            ("4. Tensor Core", self.probe_tensor_core),
            ("5. Shared Memory", self.probe_shared_memory),
            ("6. Warp Shuffle", self.probe_warp_shuffle),
            ("7. Synchronization", self.probe_synchronization),
            ("8. Async Copy", self.probe_async_copy),
            ("9. Register Field Mapping", self.probe_register_fields),
            ("10. Immediate Encoding", self.probe_immediate_encoding),
            ("11. Predication", self.probe_predication),
        ]

        all_results = {}
        for name, func in categories:
            print(f"--- {name} ---")
            try:
                res = func()
                count = len(res) if res else 0
                print(f"    Found {count} encoding(s)")
            except Exception as e:
                print(f"    ERROR: {e}")
                res = []
            all_results[name] = res
            print()

        return all_results

    # ------------------------------------------------------------------
    # Architecture comparison
    # ------------------------------------------------------------------

    def compare_architectures(self, other_arch):
        """Compare encodings between this arch and another.

        Returns dict of differences.
        """
        other = SASSProbe(arch=other_arch, ptxas=self.ptxas,
                          nvdisasm=self.nvdisasm, verbose=self.verbose)
        print(f"=== Comparing {self.arch} vs {other_arch} ===")
        print()

        # Run both
        if not self.results:
            self.probe_all_inference()
        other.probe_all_inference()

        diffs = {}
        for cat_key in self.results:
            self_encs = self.results.get(cat_key, [])
            other_encs = other.results.get(cat_key, [])

            # Match by sass_mnemonic
            self_by_mn = {e.sass_mnemonic + "_" + e.form: e for e in self_encs
                          if isinstance(e, InstructionEncoding)}
            other_by_mn = {e.sass_mnemonic + "_" + e.form: e for e in other_encs
                           if isinstance(e, InstructionEncoding)}

            cat_diffs = []
            all_keys = set(self_by_mn.keys()) | set(other_by_mn.keys())
            for k in sorted(all_keys):
                s = self_by_mn.get(k)
                o = other_by_mn.get(k)
                if s and o:
                    if s.opcode != o.opcode:
                        cat_diffs.append({
                            "mnemonic": k,
                            "change": "opcode",
                            "old": f"0x{s.opcode:03x}",
                            "new": f"0x{o.opcode:03x}",
                        })
                    # Check inst word structure differences
                    inst_diff_bits = self.diff_bits(
                        s.inst_word & 0xFFFF,   # just opcode+pred area
                        o.inst_word & 0xFFFF)
                    if inst_diff_bits:
                        cat_diffs.append({
                            "mnemonic": k,
                            "change": "inst_format",
                            "diff_bits": inst_diff_bits,
                            "old_inst": f"0x{s.inst_word:016x}",
                            "new_inst": f"0x{o.inst_word:016x}",
                        })
                elif s and not o:
                    cat_diffs.append({"mnemonic": k, "change": "removed_in_new"})
                elif o and not s:
                    cat_diffs.append({"mnemonic": k, "change": "added_in_new"})

            if cat_diffs:
                diffs[cat_key] = cat_diffs
                print(f"[{cat_key}] {len(cat_diffs)} difference(s)")
                for d in cat_diffs:
                    print(f"  {d}")
            else:
                print(f"[{cat_key}] identical")

        return diffs

    # ------------------------------------------------------------------
    # Output: encoding table (Markdown)
    # ------------------------------------------------------------------

    def write_encoding_table(self, path):
        """Write a structured encoding table in Markdown."""
        lines = []
        lines.append(f"# SASS Encoding Table -- {self.arch}")
        lines.append(f"# Auto-generated by sass_probe.py")
        lines.append("")
        lines.append("---")
        lines.append("")

        for cat_key, encs in sorted(self.results.items()):
            lines.append(f"## {cat_key}")
            lines.append("")

            if not encs:
                lines.append("(no encodings captured)")
                lines.append("")
                continue

            for enc in encs:
                if isinstance(enc, InstructionEncoding):
                    lines.append(f"### {enc.sass_mnemonic} ({enc.form})")
                    lines.append("")
                    lines.append(f"- PTX: `{enc.ptx_pattern}`")
                    if enc.opcode:
                        lines.append(f"- Opcode: `0x{enc.opcode:03x}` (bits [11:0])")
                    lines.append(f"- Instruction word: `0x{enc.inst_word:016x}`")
                    lines.append(f"- Control word:     `0x{enc.ctrl_word:016x}`")

                    if enc.fields:
                        lines.append("")
                        lines.append("| Field | Bits | Value |")
                        lines.append("|-------|------|-------|")
                        for fname, bf in sorted(enc.fields.items(),
                                                key=lambda x: -x[1].hi):
                            lines.append(
                                f"| {bf.name} | [{bf.hi}:{bf.lo}] | "
                                f"0x{bf.value:x} |")

                    if enc.modifiers:
                        lines.append("")
                        lines.append("**Modifiers:**")
                        for mname, mval in enc.modifiers.items():
                            lines.append(f"- {mname} = 0x{mval:x}")

                    if enc.variants:
                        lines.append("")
                        lines.append("**Variants:**")
                        lines.append("")
                        lines.append("| Description | Inst Word | Ctrl Word |")
                        lines.append("|-------------|-----------|-----------|")
                        for desc, inst, ctrl in enc.variants:
                            lines.append(
                                f"| {desc} | `0x{inst:016x}` | `0x{ctrl:016x}` |")

                    if enc.notes:
                        lines.append("")
                        lines.append(f"*Notes:* {enc.notes}")

                    lines.append("")
                    lines.append("---")
                    lines.append("")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Encoding table written to {path}")

    # ------------------------------------------------------------------
    # Output: JSON
    # ------------------------------------------------------------------

    def write_json(self, path):
        """Write all results as JSON."""
        data = {"arch": self.arch, "categories": {}}
        for cat_key, encs in self.results.items():
            data["categories"][cat_key] = []
            for enc in encs:
                if isinstance(enc, InstructionEncoding):
                    data["categories"][cat_key].append(enc.to_dict())
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON written to {path}")

    # ------------------------------------------------------------------
    # Output: Forth / Lithos vocabulary
    # ------------------------------------------------------------------

    def write_forth_vocabulary(self, path):
        """Generate a Forth/Lithos vocabulary file from the encoding table.

        Produces words like:
            : FADD,  ( rd rs1 rs2 -- ) ... sinst, ;
        """
        lines = []
        lines.append(f"\\ Auto-generated SASS vocabulary for {self.arch}")
        lines.append(f"\\ Generated by sass_probe.py")
        lines.append("")
        lines.append("\\ --- Opcode constants ---")
        lines.append("")

        # Collect all unique opcodes
        seen_opcodes = {}  # mnemonic -> opcode
        all_encs = []
        for cat_key, encs in self.results.items():
            for enc in encs:
                if isinstance(enc, InstructionEncoding) and enc.opcode:
                    key = enc.sass_mnemonic.split(".")[0]
                    if enc.form == "reg-imm":
                        key += "-IMM"
                    if key not in seen_opcodes:
                        seen_opcodes[key] = enc.opcode
                        all_encs.append((key, enc))

        for name, enc in sorted(all_encs, key=lambda x: x[1].opcode):
            pred_default = 0x7  # PT
            opcode_with_pred = enc.opcode | (pred_default << 12)
            lines.append(f"${'%04x' % opcode_with_pred} constant OP-{name}")

        lines.append("")
        lines.append("\\ --- Instruction emitter words ---")
        lines.append("")

        # Generate emitter words for key instructions
        for name, enc in sorted(all_encs, key=lambda x: x[1].opcode):
            pred_default = 0x7
            opcode_with_pred = enc.opcode | (pred_default << 12)

            if enc.form == "reg-reg" and "FMA" not in name and "HMMA" not in name:
                lines.append(f"\\ {name}: {enc.ptx_pattern}")
                lines.append(f": {name.lower()},  ( rd rs1 rs2 -- )")
                lines.append(f"  32 lshift >r          \\ rs2 -> bits [39:32]")
                lines.append(f"  24 lshift >r          \\ rs1 -> bits [31:24]")
                lines.append(f"  16 lshift             \\ rd  -> bits [23:16]")
                lines.append(f"  ${opcode_with_pred:04x} or r> or r> or")
                lines.append(f"  $000fc00000000000 sinst, ;")
                lines.append("")
            elif "FMA" in name and enc.form == "reg-reg":
                lines.append(f"\\ {name}: {enc.ptx_pattern}")
                lines.append(f": {name.lower()},  ( rd rs1 rs2 rs3 -- )")
                lines.append(f"  >r                    \\ rs3 saved for ctrl word")
                lines.append(f"  32 lshift >r          \\ rs2 -> bits [39:32]")
                lines.append(f"  24 lshift >r          \\ rs1 -> bits [31:24]")
                lines.append(f"  16 lshift             \\ rd  -> bits [23:16]")
                lines.append(f"  ${opcode_with_pred:04x} or r> or r> or")
                lines.append(f"  $000fca0000000000 r> or sinst, ;")
                lines.append(f"  \\ NOTE: rs3 goes into ctrl[7:0]")
                lines.append("")
            elif enc.form == "reg-imm":
                lines.append(f"\\ {name}: {enc.ptx_pattern}")
                lines.append(f": {name.lower()},  ( rd rs1 imm32 -- )")
                lines.append(f"  32 lshift >r          \\ imm32 -> bits [63:32]")
                lines.append(f"  24 lshift >r          \\ rs1   -> bits [31:24]")
                lines.append(f"  16 lshift             \\ rd    -> bits [23:16]")
                lines.append(f"  ${opcode_with_pred:04x} or r> or r> or")
                lines.append(f"  $000fc00000000000 sinst, ;")
                lines.append("")

        # Special: MUFU sub-functions
        lines.append("\\ --- MUFU sub-functions ---")
        lines.append("\\ All share opcode 0x308, sub-function in ctrl[13:10]")
        lines.append("")
        mufu_map = {}
        for enc in self.results.get("2_special_functions", []):
            if isinstance(enc, InstructionEncoding):
                subfunc = enc.modifiers.get("subfunc_ctrl_13_10", 0)
                fname = enc.sass_mnemonic.replace(".", "_")
                mufu_map[fname] = subfunc

        for fname, subfunc in sorted(mufu_map.items(), key=lambda x: x[1]):
            lines.append(f": {fname.lower()},  ( rd rs1 -- )")
            ctrl_val = 0x000fc00000000000 | (subfunc << 10)
            lines.append(f"  24 lshift >r          \\ rs1 -> bits [31:24]")
            lines.append(f"  16 lshift             \\ rd  -> bits [23:16]")
            lines.append(f"  $7308 or r> or        \\ opcode + pred PT")
            lines.append(f"  ${'%016x' % ctrl_val} sinst, ;")
            lines.append("")

        # Predicate helpers
        lines.append("\\ --- Predicate guard helpers ---")
        lines.append("\\ Apply predicate to an instruction word")
        lines.append(": @p0  ( inst -- inst' )  $FFFFF0FF and $00000000 or ;  \\ @P0")
        lines.append(": @p1  ( inst -- inst' )  $FFFFF0FF and $00001000 or ;  \\ @P1")
        lines.append(": @p2  ( inst -- inst' )  $FFFFF0FF and $00002000 or ;  \\ @P2")
        lines.append(": @p3  ( inst -- inst' )  $FFFFF0FF and $00003000 or ;  \\ @P3")
        lines.append(": @!p0 ( inst -- inst' )  $FFFFF0FF and $00008000 or ;  \\ @!P0")
        lines.append(": @!p1 ( inst -- inst' )  $FFFFF0FF and $00009000 or ;  \\ @!P1")
        lines.append(": @!p2 ( inst -- inst' )  $FFFFF0FF and $0000A000 or ;  \\ @!P2")
        lines.append(": @!p3 ( inst -- inst' )  $FFFFF0FF and $0000B000 or ;  \\ @!P3")
        lines.append("")

        # Constants
        lines.append("\\ --- Register constants ---")
        lines.append("$FF constant RZ  \\ Zero register (R255)")
        lines.append("")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Forth vocabulary written to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Lithos SASS Reverse Engineering Probe")
    parser.add_argument("--arch", default="sm_90",
                        help="Target GPU architecture (default: sm_90)")
    parser.add_argument("--compare", default=None,
                        help="Compare against another architecture (e.g. sm_100)")
    parser.add_argument("--output-md", default=None,
                        help="Write encoding table as Markdown")
    parser.add_argument("--output-json", default=None,
                        help="Write encoding table as JSON")
    parser.add_argument("--output-forth", default=None,
                        help="Write Forth/Lithos vocabulary file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--category", default=None, type=int,
                        help="Probe only this category (1-11)")
    args = parser.parse_args()

    probe = SASSProbe(arch=args.arch, verbose=args.verbose)

    if args.category:
        cat_map = {
            1: probe.probe_float_arithmetic,
            2: probe.probe_special_functions,
            3: probe.probe_half_precision,
            4: probe.probe_tensor_core,
            5: probe.probe_shared_memory,
            6: probe.probe_warp_shuffle,
            7: probe.probe_synchronization,
            8: probe.probe_async_copy,
            9: probe.probe_register_fields,
            10: probe.probe_immediate_encoding,
            11: probe.probe_predication,
        }
        func = cat_map.get(args.category)
        if func:
            results = func()
            print(f"\nCategory {args.category}: {len(results)} encoding(s)")
        else:
            print(f"Unknown category: {args.category}")
            return
    elif args.compare:
        probe.probe_all_inference()
        probe.compare_architectures(args.compare)
    else:
        probe.probe_all_inference()

    if args.output_md:
        probe.write_encoding_table(args.output_md)
    if args.output_json:
        probe.write_json(args.output_json)
    if args.output_forth:
        probe.write_forth_vocabulary(args.output_forth)

    # Default outputs if nothing specified
    if not any([args.output_md, args.output_json, args.output_forth]):
        # Print summary to stdout
        print("\n=== OPCODE SUMMARY ===")
        for cat_key, encs in sorted(probe.results.items()):
            print(f"\n{cat_key}:")
            for enc in encs:
                if isinstance(enc, InstructionEncoding):
                    print(f"  {enc.sass_mnemonic:20s} opcode=0x{enc.opcode:03x}  "
                          f"form={enc.form:10s}  "
                          f"inst=0x{enc.inst_word:016x}")


if __name__ == "__main__":
    main()
