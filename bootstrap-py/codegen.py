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

# Import parser AST node types so we can isinstance-check in _emit_expr
from parser import (
    IntLit, FloatLit, Ident, RegRef, BinOp, UnaryOp, ArrayIndex,
    FuncCall, FuncDef, Assignment, ArrayStore, Store, Load, RegWrite,
    RegRead, For, EndFor, If, Each, Stride, Shared, Param, Label, Goto,
    Barrier, MembarSys, Exit, Trap, Comment, VarDecl, BufDecl, ConstDecl,
    PrefixOp, TypeCast, Reduction,
)


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
    def emit_syscall(self, sysno: int = 0) -> None: ...
    def emit_nop(self) -> None: ...
    def emit_dsb_sy(self) -> None: ...
    def get_code(self) -> bytes: ...


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

    @staticmethod
    def _get(node, key, default=None):
        """Access a field from either a dict or dataclass node."""
        if isinstance(node, dict):
            return node.get(key, default)
        return getattr(node, key, default)

    def generate(self, ast: list) -> None:
        """Walk the top-level AST and emit code for each node."""
        for node in ast:
            self._emit_node(node)

    def _emit_node(self, node) -> int | None:
        """Emit code for a single AST node. Returns register holding result, or None."""
        # Support both dict-based and dataclass-based AST nodes
        if isinstance(node, dict):
            kind = node.get("kind")
        else:
            # Dataclass: map class name to handler
            # FuncDef -> func, Store -> store, Load -> load, etc.
            cls_name = type(node).__name__
            kind_map = {
                'FuncDef': 'func', 'Store': 'store', 'Load': 'load',
                'RegWrite': 'regwrite', 'RegRead': 'regread',
                'Assignment': 'assign', 'FuncCall': 'call',
                'If': 'if', 'For': 'for', 'BufDecl': 'buf',
                'ConstDecl': 'const', 'VarDecl': 'var',
                'Param': 'param', 'Trap': 'trap', 'Exit': 'exit_node',
                'MembarSys': 'membar', 'Barrier': 'barrier',
                'BinOp': 'binop', 'IntLit': 'intlit', 'Ident': 'ident',
                'Label': 'label', 'Goto': 'goto', 'Each': 'each',
                'Shared': 'shared', 'Stride': 'stride',
                'Comment': 'comment', 'EndFor': 'endfor',
            }
            kind = kind_map.get(cls_name, cls_name.lower())
        handler = getattr(self, f"_emit_{kind}", None)
        if handler is None:
            # Skip unknown node types silently for now
            return None
        return handler(node)

    # ----------------------------------------------------------
    # Top-level declarations
    # ----------------------------------------------------------

    def _emit_const(self, node) -> None:
        """const NAME VALUE"""
        name = self._get(node, "name")
        value = self._get(node, "value")
        # value may be an AST node (IntLit), extract raw int
        if isinstance(value, IntLit):
            value = value.value
        self.symtab.define(name, "const", value)

    def _emit_var(self, node) -> None:
        """var NAME INIT"""
        name = self._get(node, "name")
        init = self._get(node, "value", 0)
        if isinstance(init, IntLit):
            init = init.value
        elif not isinstance(init, int):
            init = 0
        self.symtab.define(name, "var", init)
        # Allocate a register and load initial value
        reg = self.regs.get_or_alloc(name)
        self.emitter.emit_mov_imm(reg, init)

    def _emit_buf(self, node) -> None:
        """buf NAME SIZE — reserve a buffer label (address resolved at link time)."""
        name = self._get(node, "name")
        size = self._get(node, "size")
        if isinstance(size, IntLit):
            size = size.value
        self.symtab.define(name, "buf", size)

    def _emit_func(self, node) -> None:
        """Composition (function) definition."""
        name = node.name if hasattr(node, 'name') else node["name"]
        params = node.params if hasattr(node, 'params') else node.get("params", [])
        body = node.body if hasattr(node, 'body') else node.get("body", [])
        is_kernel = getattr(node, 'is_kernel', False)

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

    def _emit_assign(self, node) -> None:
        """target = expr"""
        target_name = self._get(node, "target")
        val_reg = self._emit_expr(self._get(node, "value"))
        dst_reg = self.regs.get_or_alloc(target_name)
        if dst_reg != val_reg:
            self.emitter.emit_mov_reg(dst_reg, val_reg)

    def _emit_call(self, node) -> None:
        """Composition call: name arg1 arg2 ..."""
        name = self._get(node, "name")
        args = self._get(node, "args", [])

        # Load arguments into X0..X7
        for i, arg in enumerate(args):
            if i >= 8:
                break
            arg_reg = self._emit_expr(arg)
            if arg_reg != i:
                self.emitter.emit_mov_reg(i, arg_reg)

        # Branch-and-link (emit BL placeholder, patch later when label known)
        self.emitter.emit_bl(0)  # offset 0 placeholder

    def _emit_assign_call(self, node) -> None:
        """result = call(args) — call then move X0 to result register."""
        call_node = self._get(node, "call")
        self._emit_call(call_node if call_node else node)
        target_name = self._get(node, "target")
        dst_reg = self.regs.get_or_alloc(target_name)
        if dst_reg != 0:
            self.emitter.emit_mov_reg(dst_reg, 0)  # result in X0

    def _emit_store(self, node) -> None:
        """<- width addr value — memory store."""
        # Evaluate address expression fully (includes any offset arithmetic)
        addr_reg = self._emit_expr(self._get(node, "addr"))
        val_reg = self._emit_expr(self._get(node, "value"))
        # Store at [addr_reg, #0] — offset is already folded into addr via BinOp
        self.emitter.emit_store(val_reg, addr_reg, 0)

    def _emit_load(self, node) -> int:
        """-> width addr — memory load, returns register with loaded value."""
        addr_reg = self._emit_expr(self._get(node, "addr"))
        dst_reg = self.regs.temp()
        self.emitter.emit_load(dst_reg, addr_reg, 0)
        return dst_reg

    def _emit_regwrite(self, node) -> None:
        """↓ $N value — write a value to hardware register $N."""
        reg_node = self._get(node, "reg")
        # reg may be a RegRef or int
        if isinstance(reg_node, RegRef):
            hw_reg = int(reg_node.name) if reg_node.name.isdigit() else self.regs.get_or_alloc(reg_node.name)
        elif isinstance(reg_node, int):
            hw_reg = reg_node
        else:
            hw_reg = int(str(reg_node))
        val_reg = self._emit_expr(self._get(node, "value"))
        if val_reg != hw_reg:
            self.emitter.emit_mov_reg(hw_reg, val_reg)

    def _emit_regread(self, node) -> int:
        """↑ $N — read hardware register $N into a temp."""
        reg_node = self._get(node, "reg")
        if isinstance(reg_node, RegRef):
            hw_reg = int(reg_node.name) if reg_node.name.isdigit() else 0
        elif isinstance(reg_node, int):
            hw_reg = reg_node
        else:
            hw_reg = int(str(reg_node))
        return hw_reg

    def _emit_trap(self, node) -> None:
        """trap — emit SVC #0 syscall."""
        sysnum = self._get(node, "sysnum", None)
        if isinstance(sysnum, IntLit):
            sysnum = sysnum.value
        elif sysnum is not None and not isinstance(sysnum, int):
            sysnum = 0
        self.emitter.emit_syscall(sysnum if sysnum is not None else 0)

    def _emit_if(self, node) -> None:
        """if<cond> lhs rhs / body / else body"""
        # Dataclass If: op, left, right, body, label
        # Dict-based: cond, lhs, rhs, body, else_body
        cond = self._get(node, "op") or self._get(node, "cond")
        lhs = self._get(node, "left") or self._get(node, "lhs")
        rhs = self._get(node, "right") or self._get(node, "rhs")
        body = self._get(node, "body", [])

        lhs_reg = self._emit_expr(lhs)
        rhs_reg = self._emit_expr(rhs)

        end_label = self.fresh_label("endif")

        self.emitter.emit_cmp(lhs_reg, rhs_reg)

        # Branch to end on OPPOSITE condition
        inv = {"<": "emit_branch_ge", ">": "emit_branch_le",
               "==": "emit_branch_ne", "!=": "emit_branch_eq",
               "<=": "emit_branch_gt", ">=": "emit_branch_lt"}
        branch_method = inv.get(cond, "emit_branch_ne")
        branch_fn = getattr(self.emitter, branch_method)
        branch_fn(end_label)

        # Then body
        for stmt in body:
            self._emit_node(stmt)

        self.emitter.emit_label(end_label)

    def _emit_for(self, node) -> None:
        """for var start end / body / endfor"""
        var_name = self._get(node, "var")
        start_reg = self._emit_expr(self._get(node, "start"))
        end_reg = self._emit_expr(self._get(node, "end"))
        loop_reg = self.regs.get_or_alloc(var_name)

        self.emitter.emit_mov_reg(loop_reg, start_reg)

        top_label = self.fresh_label("for_top")
        end_label = self.fresh_label("for_end")

        self.emitter.emit_label(top_label)
        self.emitter.emit_cmp(loop_reg, end_reg)
        self.emitter.emit_branch_ge(end_label)

        for stmt in self._get(node, "body", []):
            self._emit_node(stmt)

        self.emitter.emit_add_imm(loop_reg, loop_reg, 1)
        self.emitter.emit_branch(top_label)
        self.emitter.emit_label(end_label)

    def _emit_label(self, node) -> None:
        """label NAME"""
        self.emitter.emit_label(self._get(node, "name"))

    def _emit_goto(self, node) -> None:
        """goto LABEL"""
        target = self._get(node, "target") or self._get(node, "name")
        self.emitter.emit_branch(target)

    def _emit_return(self, node) -> None:
        """return [expr]"""
        val = self._get(node, "value", None)
        if val is not None:
            val_reg = self._emit_expr(val)
            if val_reg != 0:
                self.emitter.emit_mov_reg(0, val_reg)
        self.emitter.emit_epilogue()
        self.emitter.emit_ret()

    def _emit_barrier(self, node) -> None:
        """barrier — emit DSB SY / membar."""
        self.emitter.emit_nop()  # placeholder; real emitter would emit DSB

    def _emit_membar(self, node) -> None:
        """membar_sys — emit DSB SY."""
        self.emitter.emit_dsb_sy()

    def _emit_exit_node(self, node) -> None:
        """exit — emit return/exit."""
        self.emitter.emit_ret()

    def _emit_comment(self, node) -> None:
        """comment — skip."""
        pass

    def _emit_endfor(self, node) -> None:
        """endfor — handled by for body, skip."""
        pass

    def _emit_each(self, node) -> None:
        """each — GPU-specific, emit NOP placeholder."""
        self.emitter.emit_nop()

    def _emit_shared(self, node) -> None:
        """shared — GPU-specific, skip."""
        pass

    def _emit_stride(self, node) -> None:
        """stride — GPU-specific, skip."""
        pass

    def _emit_param(self, node) -> None:
        """param — handled at function level, skip."""
        pass

    # ----------------------------------------------------------
    # Expression evaluator — returns register number
    # ----------------------------------------------------------

    def _emit_expr(self, node) -> int:
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

        # -- Dataclass AST nodes --
        if isinstance(node, IntLit):
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, node.value)
            return reg

        if isinstance(node, FloatLit):
            # Treat as integer bits for now (bootstrap)
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, int(node.value))
            return reg

        if isinstance(node, Ident):
            return self._emit_expr(node.name)

        if isinstance(node, RegRef):
            name = node.name
            if name.isdigit():
                return int(name)
            return self.regs.get_or_alloc(name)

        if isinstance(node, BinOp):
            lhs = self._emit_expr(node.left)
            rhs = self._emit_expr(node.right)
            dst = self.regs.temp()
            op = node.op
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

        if isinstance(node, Load):
            return self._emit_load(node)

        if isinstance(node, RegRead):
            return self._emit_regread(node)

        if isinstance(node, FuncCall):
            self._emit_call(node)
            return 0  # result in X0

        if isinstance(node, ArrayIndex):
            base_reg = self._emit_expr(node.base)
            idx_reg = self._emit_expr(node.index)
            # Compute address = base + index*8, load from it
            addr_reg = self.regs.temp()
            self.emitter.emit_add(addr_reg, base_reg, idx_reg)
            dst_reg = self.regs.temp()
            self.emitter.emit_load(dst_reg, addr_reg, 0)
            return dst_reg

        if isinstance(node, Reduction):
            # Placeholder: just evaluate the operand
            return self._emit_expr(node.operand)

        if isinstance(node, UnaryOp):
            # Placeholder: just evaluate the operand
            return self._emit_expr(node.operand)

        # -- Dict-based fallback (for compatibility) --
        if isinstance(node, dict):
            kind = node.get("kind")

            if kind == "int_literal":
                reg = self.regs.temp()
                self.emitter.emit_mov_imm(reg, node["value"])
                return reg

            if kind == "ident":
                return self._emit_expr(node["name"])

            if kind == "binop":
                lhs = self._emit_expr(node.get("lhs") or node.get("left"))
                rhs = self._emit_expr(node.get("rhs") or node.get("right"))
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
                return self._emit_regread(node)

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
