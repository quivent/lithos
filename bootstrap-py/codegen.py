"""
Lithos code generator — AST walker that emits machine code.

Takes an AST (from parser.py) and an emitter backend (ARM64 or SM90),
walks the tree, and emits target-specific machine code.

Register allocation: simple linear allocation from a pool.
  - ARM64: X9..X28 for general purpose (X0..X8 reserved for syscall ABI,
    X29=FP, X30=LR, X31=SP/XZR)
  - SM90: R0..R254 general purpose

Handles:
  - Function definitions (prologue/epilogue)
  - Composition calls (BL for arm64, inline for gpu)
  - Memory ops: <- (store), -> (load), up-arrow (reg read), down-arrow (reg write)
  - Arithmetic: + - * /
  - Comparisons: == != < > <= >=
  - Bitwise: & | ^ << >>
  - Control flow: if, for
  - Syscall trap
  - Constants, variables, buffers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ============================================================
# Emitter protocol — backends must implement this
# ============================================================

class Emitter(Protocol):
    """Interface that ARM64Emitter and SM90Emitter must satisfy."""

    def emit_prologue(self, name: str, n_locals: int) -> None: ...
    def emit_epilogue(self) -> None: ...
    def emit_ret(self) -> None: ...
    def emit_mov_imm(self, rd: int, imm: int) -> None: ...
    def emit_mov_reg(self, rd: int, rs: int) -> None: ...
    def emit_add(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_sub(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_mul(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_div(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_and(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_or(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_xor(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_shl(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_shr(self, rd: int, ra: int, rb: int) -> None: ...
    def emit_add_imm(self, rd: int, ra: int, imm: int) -> None: ...
    def emit_load(self, rd: int, base: int, offset: int, width: int) -> None: ...
    def emit_store(self, rs: int, base: int, offset: int, width: int) -> None: ...
    def emit_cmp(self, ra: int, rb: int) -> None: ...
    def emit_branch_eq(self, label: str) -> None: ...
    def emit_branch_ne(self, label: str) -> None: ...
    def emit_branch_lt(self, label: str) -> None: ...
    def emit_branch_ge(self, label: str) -> None: ...
    def emit_branch_gt(self, label: str) -> None: ...
    def emit_branch_le(self, label: str) -> None: ...
    def emit_branch(self, label: str) -> None: ...
    def emit_label(self, label: str) -> None: ...
    def emit_bl(self, target: str) -> None: ...
    def emit_syscall(self) -> None: ...
    def emit_nop(self) -> None: ...
    def get_code(self) -> bytes: ...
    def get_code_size(self) -> int: ...


# ============================================================
# Register allocator
# ============================================================

@dataclass
class RegisterAllocator:
    """Simple linear register allocator.

    ARM64 convention:
      X0-X7:  argument / result registers (managed by ABI, not allocated)
      X8:     syscall number (managed explicitly)
      X9-X28: general purpose (we allocate from this pool)
      X29:    frame pointer
      X30:    link register
      X31:    SP / XZR
    """

    # Pool of allocatable registers
    pool: list[int] = field(default_factory=lambda: list(range(9, 29)))
    next_idx: int = 0

    # Map from variable name to allocated register
    bindings: dict[str, int] = field(default_factory=dict)

    def alloc(self, name: str) -> int:
        """Allocate a register for a variable name."""
        if name in self.bindings:
            return self.bindings[name]
        if self.next_idx >= len(self.pool):
            raise RuntimeError(f"register allocator exhausted (need reg for '{name}')")
        reg = self.pool[self.next_idx]
        self.next_idx += 1
        self.bindings[name] = reg
        return reg

    def lookup(self, name: str) -> int | None:
        """Look up a variable's register, or None if not allocated."""
        return self.bindings.get(name)

    def get_or_alloc(self, name: str) -> int:
        """Return existing register or allocate a new one."""
        return self.alloc(name)

    def reset(self) -> None:
        """Reset all allocations (e.g. at function boundary)."""
        self.next_idx = 0
        self.bindings.clear()

    def temp(self) -> int:
        """Allocate an anonymous temporary register."""
        return self.alloc(f"__tmp_{self.next_idx}")


# ============================================================
# Symbol table
# ============================================================

@dataclass
class Symbol:
    kind: str          # "const", "var", "buf", "func", "param"
    value: Any = None  # for const: integer value; for buf: size; for func: param list
    name: str = ""


@dataclass
class SymbolTable:
    symbols: dict[str, Symbol] = field(default_factory=dict)

    def define(self, name: str, kind: str, value: Any = None) -> None:
        self.symbols[name] = Symbol(kind=kind, value=value, name=name)

    def lookup(self, name: str) -> Symbol | None:
        return self.symbols.get(name)


# ============================================================
# Code generator
# ============================================================

@dataclass
class CodeGenerator:
    emitter: Emitter
    regs: RegisterAllocator = field(default_factory=RegisterAllocator)
    symtab: SymbolTable = field(default_factory=SymbolTable)
    label_counter: int = 0
    current_func: str | None = None

    def fresh_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f".{prefix}_{self.label_counter}"

    def generate(self, ast: list) -> None:
        """Walk the top-level AST and emit code for each node."""
        for node in ast:
            self._emit_node(node)

    def _emit_node(self, node: dict) -> int | None:
        """Emit code for a single AST node. Returns register holding result, or None."""
        kind = node.get("kind")
        handler = getattr(self, f"_emit_{kind}", None)
        if handler is None:
            raise ValueError(f"unknown AST node kind: {kind!r}")
        return handler(node)

    # ----------------------------------------------------------
    # Top-level declarations
    # ----------------------------------------------------------

    def _emit_const(self, node: dict) -> None:
        """const NAME VALUE"""
        name = node["name"]
        value = node["value"]
        self.symtab.define(name, "const", value)

    def _emit_var(self, node: dict) -> None:
        """var NAME INIT"""
        name = node["name"]
        init = node.get("value", 0)
        self.symtab.define(name, "var", init)
        # Allocate a register and load initial value
        reg = self.regs.get_or_alloc(name)
        self.emitter.emit_mov_imm(reg, init)

    def _emit_buf(self, node: dict) -> None:
        """buf NAME SIZE — reserve a buffer label (address resolved at link time)."""
        name = node["name"]
        size = node["size"]
        self.symtab.define(name, "buf", size)

    def _emit_func(self, node: dict) -> None:
        """Composition (function) definition.

        node = {kind: "func", name: str, params: [str, ...], body: [node, ...],
                is_host: bool, is_kernel: bool}
        """
        name = node["name"]
        params = node.get("params", [])
        body = node.get("body", [])
        is_kernel = node.get("is_kernel", False)

        self.symtab.define(name, "func", params)
        self.current_func = name

        # Reset register allocator for this function scope
        self.regs.reset()

        # Bind parameters to registers (ARM64: X0..X7 for first 8 params)
        for i, pname in enumerate(params):
            if i < 8:
                self.regs.bindings[pname] = i  # X0..X7
            else:
                # Spill to stack — not implemented in bootstrap
                reg = self.regs.alloc(pname)
                # Would need stack load here

        # Emit function label and prologue
        self.emitter.emit_label(name)
        n_locals = max(0, len(params) - 8)
        self.emitter.emit_prologue(name, n_locals)

        # Emit body
        for stmt in body:
            self._emit_node(stmt)

        # Emit epilogue and return
        self.emitter.emit_epilogue()
        self.emitter.emit_ret()

        self.current_func = None

    # ----------------------------------------------------------
    # Statements
    # ----------------------------------------------------------

    def _emit_assign(self, node: dict) -> None:
        """target = expr"""
        target_name = node["target"]
        val_reg = self._emit_expr(node["value"])
        dst_reg = self.regs.get_or_alloc(target_name)
        if dst_reg != val_reg:
            self.emitter.emit_mov_reg(dst_reg, val_reg)

    def _emit_call(self, node: dict) -> None:
        """Composition call: name arg1 arg2 ..."""
        name = node["name"]
        args = node.get("args", [])

        # Load arguments into X0..X7
        for i, arg in enumerate(args):
            if i >= 8:
                break
            arg_reg = self._emit_expr(arg)
            if arg_reg != i:
                self.emitter.emit_mov_reg(i, arg_reg)

        # Branch-and-link
        self.emitter.emit_bl(name)

    def _emit_assign_call(self, node: dict) -> None:
        """result = call(args) — call then move X0 to result register."""
        self._emit_call(node["call"])
        target_name = node["target"]
        dst_reg = self.regs.get_or_alloc(target_name)
        if dst_reg != 0:
            self.emitter.emit_mov_reg(dst_reg, 0)  # result in X0

    def _emit_store(self, node: dict) -> None:
        """<- width addr value — memory store.

        ← 32 base + offset value
        """
        width = node["width"]
        base_reg = self._emit_expr(node["addr"])
        val_reg = self._emit_expr(node["value"])
        offset = node.get("offset", 0)
        self.emitter.emit_store(val_reg, base_reg, offset, width)

    def _emit_load(self, node: dict) -> int:
        """-> width addr — memory load, returns register with loaded value.

        result → 64 base offset
        """
        width = node["width"]
        base_reg = self._emit_expr(node["addr"])
        offset = node.get("offset", 0)
        dst_reg = self.regs.temp()
        self.emitter.emit_load(dst_reg, base_reg, offset, width)
        return dst_reg

    def _emit_reg_write(self, node: dict) -> None:
        """↓ $N value — write a value to hardware register $N."""
        hw_reg = node["reg"]  # integer: 0, 1, ..., 8, etc.
        val_reg = self._emit_expr(node["value"])
        if val_reg != hw_reg:
            self.emitter.emit_mov_reg(hw_reg, val_reg)

    def _emit_reg_read(self, node: dict) -> int:
        """↑ $N — read hardware register $N into a temp."""
        hw_reg = node["reg"]
        return hw_reg  # register already holds the value

    def _emit_trap(self, node: dict) -> None:
        """trap — emit SVC #0 syscall."""
        self.emitter.emit_syscall()

    def _emit_if(self, node: dict) -> None:
        """if<cond> lhs rhs / body / else body"""
        cond = node["cond"]   # "<", ">", "==", "!=", "<=", ">="
        lhs_reg = self._emit_expr(node["lhs"])
        rhs_reg = self._emit_expr(node["rhs"])

        else_label = self.fresh_label("else")
        end_label = self.fresh_label("endif")

        self.emitter.emit_cmp(lhs_reg, rhs_reg)

        # Branch to else on OPPOSITE condition
        inv = {"<": "emit_branch_ge", ">": "emit_branch_le",
               "==": "emit_branch_ne", "!=": "emit_branch_eq",
               "<=": "emit_branch_gt", ">=": "emit_branch_lt"}
        branch_fn = getattr(self.emitter, inv[cond])
        has_else = bool(node.get("else_body"))

        branch_fn(else_label if has_else else end_label)

        # Then body
        for stmt in node.get("body", []):
            self._emit_node(stmt)

        if has_else:
            self.emitter.emit_branch(end_label)
            self.emitter.emit_label(else_label)
            for stmt in node["else_body"]:
                self._emit_node(stmt)

        self.emitter.emit_label(end_label)

    def _emit_for(self, node: dict) -> None:
        """for var start end / body / endfor"""
        var_name = node["var"]
        start_reg = self._emit_expr(node["start"])
        end_reg = self._emit_expr(node["end"])
        loop_reg = self.regs.get_or_alloc(var_name)

        self.emitter.emit_mov_reg(loop_reg, start_reg)

        top_label = self.fresh_label("for_top")
        end_label = self.fresh_label("for_end")

        self.emitter.emit_label(top_label)
        self.emitter.emit_cmp(loop_reg, end_reg)
        self.emitter.emit_branch_ge(end_label)

        for stmt in node.get("body", []):
            self._emit_node(stmt)

        self.emitter.emit_add_imm(loop_reg, loop_reg, 1)
        self.emitter.emit_branch(top_label)
        self.emitter.emit_label(end_label)

    def _emit_label(self, node: dict) -> None:
        """label NAME"""
        self.emitter.emit_label(node["name"])

    def _emit_goto(self, node: dict) -> None:
        """goto LABEL"""
        self.emitter.emit_branch(node["target"])

    def _emit_return(self, node: dict) -> None:
        """return [expr]"""
        if "value" in node:
            val_reg = self._emit_expr(node["value"])
            if val_reg != 0:
                self.emitter.emit_mov_reg(0, val_reg)
        self.emitter.emit_epilogue()
        self.emitter.emit_ret()

    def _emit_barrier(self, node: dict) -> None:
        """barrier — emit DSB SY / membar."""
        self.emitter.emit_nop()  # placeholder; real emitter would emit DSB

    # ----------------------------------------------------------
    # Expression evaluator — returns register number
    # ----------------------------------------------------------

    def _emit_expr(self, node: dict | int | str) -> int:
        """Emit code for an expression, return the register holding the result."""
        if isinstance(node, int):
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, node)
            return reg

        if isinstance(node, str):
            # Variable reference or constant
            sym = self.symtab.lookup(node)
            if sym and sym.kind == "const":
                reg = self.regs.temp()
                self.emitter.emit_mov_imm(reg, sym.value)
                return reg
            r = self.regs.lookup(node)
            if r is not None:
                return r
            # Unknown name — allocate
            return self.regs.get_or_alloc(node)

        if isinstance(node, dict):
            kind = node.get("kind")

            if kind == "int_literal":
                reg = self.regs.temp()
                self.emitter.emit_mov_imm(reg, node["value"])
                return reg

            if kind == "ident":
                return self._emit_expr(node["name"])

            if kind == "binop":
                lhs = self._emit_expr(node["lhs"])
                rhs = self._emit_expr(node["rhs"])
                dst = self.regs.temp()
                op = node["op"]
                ops = {
                    "+": self.emitter.emit_add,
                    "-": self.emitter.emit_sub,
                    "*": self.emitter.emit_mul,
                    "/": self.emitter.emit_div,
                    "&": self.emitter.emit_and,
                    "|": self.emitter.emit_or,
                    "^": self.emitter.emit_xor,
                    "<<": self.emitter.emit_shl,
                    ">>": self.emitter.emit_shr,
                }
                emit_fn = ops.get(op)
                if emit_fn is None:
                    raise ValueError(f"unknown binop: {op!r}")
                emit_fn(dst, lhs, rhs)
                return dst

            if kind == "load":
                return self._emit_load(node)

            if kind == "reg_read":
                return self._emit_reg_read(node)

            if kind == "call_expr":
                self._emit_call(node)
                return 0  # result in X0

            raise ValueError(f"unknown expr kind: {kind!r}")

        raise TypeError(f"cannot emit expression: {node!r}")


# ============================================================
# Public entry point
# ============================================================

def generate(ast: list, emitter: Emitter) -> None:
    """Walk the AST and emit code using the given emitter backend."""
    gen = CodeGenerator(emitter=emitter)
    gen.generate(ast)
