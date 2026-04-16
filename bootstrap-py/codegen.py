"""
Lithos code generator — AST walker that emits machine code.

Takes an AST (from parser.py) and an emitter backend (ARM64 or SM90),
walks the tree, and emits target-specific machine code.

Register allocation: linear allocation from a pool with LRU spilling.
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
    """Register allocator with temporary recycling and LRU spilling.

    ARM64 convention:
      X0-X7:  argument / result registers (managed by ABI, not allocated)
      X8:     syscall number (managed explicitly)
      X9-X28: general purpose (we allocate from this pool)
      X29:    frame pointer
      X30:    link register
      X31:    SP / XZR

    Temporaries (__tmp_*) are expression intermediates that only live within
    a single statement.  free_temps() releases them back to the pool so they
    can be reused by subsequent statements.

    When the pool is exhausted and a new register is needed, the
    least-recently-used variable is spilled to a stack slot (STR to
    [X29, #-offset]) and its register is reused.  When a spilled variable
    is accessed again, it is reloaded from its stack slot.
    """

    # Pool of allocatable registers
    pool: list[int] = field(default_factory=lambda: list(range(9, 29)))

    # Map from variable name to allocated register
    bindings: dict[str, int] = field(default_factory=dict)

    # Set of registers currently available for allocation
    free_set: list[int] = field(default_factory=list)

    # LRU tracking: list of variable names in access order (most recent last)
    access_order: list[str] = field(default_factory=list)

    # Spill state: variable name -> stack offset (negative, relative to FP)
    spilled: dict[str, int] = field(default_factory=dict)
    next_spill_slot: int = 0  # count of spill slots allocated

    # Emitter reference for spill/reload (set by CodeGenerator before use)
    emitter: Any = None

    def __post_init__(self):
        # Initially all pool registers are free
        self.free_set = list(reversed(self.pool))  # pop from end for efficiency

    def _touch(self, name: str) -> None:
        """Mark a variable as most-recently-used."""
        try:
            self.access_order.remove(name)
        except ValueError:
            pass
        self.access_order.append(name)

    def _spill_lru(self) -> int:
        """Spill the least-recently-used pool variable to the stack."""
        pool_set = set(self.pool)
        for name in list(self.access_order):
            if name in self.bindings and self.bindings[name] in pool_set:
                reg = self.bindings.pop(name)
                self.access_order.remove(name)
                # Assign a spill slot if not already spilled before
                if name not in self.spilled:
                    self.next_spill_slot += 1
                    # Slot N at [FP, #-(16 + N*8)]: below saved FP/LR pair
                    self.spilled[name] = -(16 + self.next_spill_slot * 8)
                # Emit STR to save the register value
                if self.emitter is not None:
                    self.emitter.emit_store(reg, 29, self.spilled[name])
                return reg
        raise RuntimeError("register allocator exhausted: no spillable variables")

    def alloc(self, name: str) -> int:
        """Allocate a register for a variable name."""
        if name in self.bindings:
            self._touch(name)
            return self.bindings[name]
        # Check if this variable was previously spilled -- reload it
        if name in self.spilled:
            if not self.free_set:
                reg = self._spill_lru()
            else:
                reg = self.free_set.pop()
            self.bindings[name] = reg
            self._touch(name)
            # Emit LDR to reload from stack slot
            if self.emitter is not None:
                self.emitter.emit_load(reg, 29, self.spilled[name])
            return reg
        # Fresh allocation
        if not self.free_set:
            reg = self._spill_lru()
        else:
            reg = self.free_set.pop()
        self.bindings[name] = reg
        self._touch(name)
        return reg

    def lookup(self, name: str) -> int | None:
        """Look up a variable's register, or None if not allocated."""
        r = self.bindings.get(name)
        if r is not None:
            self._touch(name)
        return r

    def get_or_alloc(self, name: str) -> int:
        """Return existing register or allocate a new one."""
        return self.alloc(name)

    def reset(self) -> None:
        """Reset all allocations (e.g. at function boundary)."""
        self.bindings.clear()
        self.free_set = list(reversed(self.pool))
        self.access_order.clear()
        self.spilled.clear()
        self.next_spill_slot = 0

    @property
    def spill_area_bytes(self) -> int:
        """Total bytes needed for spill slots (16-byte aligned)."""
        raw = self.next_spill_slot * 8
        return (raw + 15) & ~15

    _temp_counter: int = 0

    def temp(self) -> int:
        """Allocate an anonymous temporary register.
        Uses a monotonic counter — id(object()) returned duplicate IDs because
        Python immediately GC'd the new object, giving all temps the same name
        and therefore the same register."""
        RegisterAllocator._temp_counter += 1
        name = f"__tmp_{RegisterAllocator._temp_counter}"
        return self.alloc(name)

    def free_temps(self) -> None:
        """Release all __tmp_* temporaries back to the free pool.

        Called at statement boundaries so expression intermediates are recycled.
        """
        to_free = [name for name in self.bindings if name.startswith("__tmp_")]
        for name in to_free:
            reg = self.bindings.pop(name)
            self.free_set.append(reg)
            try:
                self.access_order.remove(name)
            except ValueError:
                pass
            self.spilled.pop(name, None)
        # Keep free_set sorted (highest first) so we pop lowest first -- deterministic
        self.free_set.sort(reverse=True)


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

    def __post_init__(self):
        # Wire the emitter into the register allocator so it can emit
        # spill (STR) and reload (LDR) instructions when needed.
        self.regs.emitter = self.emitter

    @property
    def is_gpu(self) -> bool:
        """True if emitting for SM90 GPU target."""
        return hasattr(self.emitter, 'emit_s2r')

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
                'ArrayStore': 'arraystore',
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
        """buf NAME SIZE -- reserve a buffer label (address resolved at link time)."""
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

        if self.is_gpu:
            # GPU: parameters come from constant bank 0 (cbuf0).
            from emit_sm90 import SR_TID_X, SR_CTAID_X
            for i, pname in enumerate(params):
                reg = self.regs.alloc(pname)
                cbuf_offset = 0x210 + i * 8
                self.emitter.emit_ldc(reg, 0, cbuf_offset)
        else:
            # ARM64: parameters in X0..X7 for first 8 params
            for i, pname in enumerate(params):
                if i < 8:
                    self.regs.bindings[pname] = i  # X0..X7
                else:
                    reg = self.regs.alloc(pname)

        # Emit function label and prologue.
        # Reserve generous stack space for spill slots (32 extra slots).
        self.emitter.emit_label(name)
        n_locals = max(0, len(params) - 8)
        spill_headroom = 32
        self.emitter.emit_prologue(name, n_locals + spill_headroom)

        # Emit body -- free temporaries after each statement so registers recycle
        for stmt in body:
            self._emit_node(stmt)
            self.regs.free_temps()

        # Emit epilogue and return.
        # The epilogue must use the same frame_size as the prologue.
        if not self.is_gpu:
            frame_size = max(16, ((n_locals + spill_headroom + 2) * 8 + 15) & ~15)
            self.emitter.emit_epilogue(frame_size)
        else:
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
        """Composition call: name arg1 arg2 ...

        On GPU, certain intrinsic names map to MUFU or SASS instructions.
        """
        name = self._get(node, "name")
        args = self._get(node, "args", [])

        if self.is_gpu:
            from emit_sm90 import RZ
            if name == "neg" and len(args) == 2:
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_sub(dst_reg, RZ, src_reg)
                return
            elif name == "exp" and len(args) == 2:
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_ex2(dst_reg, src_reg)
                return
            elif name == "rcp" and len(args) == 2:
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_rcp(dst_reg, src_reg)
                return
            elif name == "lg" and len(args) == 2:
                # log2(x) -- used by decay_gate.ls for softplus
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_lg2(dst_reg, src_reg)
                return
            elif name == "sin" and len(args) == 2:
                # sine -- used by attend.ls for RoPE
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_sin(dst_reg, src_reg)
                return
            elif name == "cos" and len(args) == 2:
                # cosine -- used by attend.ls for RoPE
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_cos(dst_reg, src_reg)
                return
            elif name == "rsqrt" and len(args) == 2:
                # 1/sqrt(x) -- used by reduce.ls for RMSNorm, attend.ls for scale
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_rsq(dst_reg, src_reg)
                return
            elif name == "sqrt" and len(args) == 2:
                dst_reg = self._emit_expr(args[0])
                src_reg = self._emit_expr(args[1])
                self.emitter.emit_mufu_sqrt(dst_reg, src_reg)
                return

        # Default: ARM64-style call via registers + BL
        for i, arg in enumerate(args):
            if i >= 8:
                break
            arg_reg = self._emit_expr(arg)
            if arg_reg != i:
                self.emitter.emit_mov_reg(i, arg_reg)

        # Branch-and-link (emit BL placeholder, patch later when label known)
        self.emitter.emit_bl(0)  # offset 0 placeholder

    def _emit_assign_call(self, node) -> None:
        """result = call(args) -- call then move X0 to result register."""
        call_node = self._get(node, "call")
        self._emit_call(call_node if call_node else node)
        target_name = self._get(node, "target")
        dst_reg = self.regs.get_or_alloc(target_name)
        if dst_reg != 0:
            self.emitter.emit_mov_reg(dst_reg, 0)  # result in X0

    def _emit_store(self, node) -> None:
        """<- width addr value -- memory store."""
        addr_reg = self._emit_expr(self._get(node, "addr"))
        val_reg = self._emit_expr(self._get(node, "value"))
        self.emitter.emit_store(val_reg, addr_reg, 0)

    def _emit_load(self, node) -> int:
        """-> width addr -- memory load, returns register with loaded value."""
        addr_reg = self._emit_expr(self._get(node, "addr"))
        dst_reg = self.regs.temp()
        self.emitter.emit_load(dst_reg, addr_reg, 0)
        return dst_reg

    def _emit_regwrite(self, node) -> None:
        """down-arrow $N value -- write a value to hardware register $N."""
        reg_node = self._get(node, "reg")
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
        """up-arrow $N -- read hardware register $N into a temp."""
        reg_node = self._get(node, "reg")
        if isinstance(reg_node, RegRef):
            hw_reg = int(reg_node.name) if reg_node.name.isdigit() else 0
        elif isinstance(reg_node, int):
            hw_reg = reg_node
        else:
            hw_reg = int(str(reg_node))
        return hw_reg

    def _emit_trap(self, node) -> None:
        """trap -- emit SVC #0 syscall."""
        sysnum = self._get(node, "sysnum", None)
        if isinstance(sysnum, IntLit):
            sysnum = sysnum.value
        elif sysnum is not None and not isinstance(sysnum, int):
            sysnum = 0
        self.emitter.emit_syscall(sysnum if sysnum is not None else 0)

    def _emit_if(self, node) -> None:
        """if<cond> lhs rhs / body / else body"""
        cond = self._get(node, "op") or self._get(node, "cond")
        lhs = self._get(node, "left") or self._get(node, "lhs")
        rhs = self._get(node, "right") or self._get(node, "rhs")
        body = self._get(node, "body", [])

        lhs_reg = self._emit_expr(lhs)
        rhs_reg = self._emit_expr(rhs)

        end_label = self.fresh_label("endif")

        self.emitter.emit_cmp(lhs_reg, rhs_reg)
        self.regs.free_temps()  # condition operands no longer needed

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
            self.regs.free_temps()

        self.emitter.emit_label(end_label)

    def _emit_for(self, node) -> None:
        """for var start end / body / endfor"""
        var_name = self._get(node, "var")
        start_reg = self._emit_expr(self._get(node, "start"))
        end_reg = self._emit_expr(self._get(node, "end"))
        loop_reg = self.regs.get_or_alloc(var_name)

        self.emitter.emit_mov_reg(loop_reg, start_reg)

        # end_reg must stay live through the loop (used in cmp each iteration),
        # so give it a stable non-temp name if it was a temporary.
        end_binding = [k for k, v in self.regs.bindings.items() if v == end_reg and k.startswith("__tmp_")]
        if end_binding:
            old_name = end_binding[0]
            del self.regs.bindings[old_name]
            self.regs.bindings[f"__for_end_{self.label_counter}"] = end_reg
        self.regs.free_temps()

        top_label = self.fresh_label("for_top")
        end_label = self.fresh_label("for_end")

        self.emitter.emit_label(top_label)
        self.emitter.emit_cmp(loop_reg, end_reg)
        self.emitter.emit_branch_ge(end_label)

        for stmt in self._get(node, "body", []):
            self._emit_node(stmt)
            self.regs.free_temps()

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
        """barrier -- emit DSB SY / membar."""
        self.emitter.emit_nop()

    def _emit_membar(self, node) -> None:
        """membar_sys -- emit DSB SY."""
        self.emitter.emit_dsb_sy()

    def _emit_exit_node(self, node) -> None:
        """exit -- emit return/exit."""
        self.emitter.emit_ret()

    def _emit_comment(self, node) -> None:
        """comment -- skip."""
        pass

    def _emit_endfor(self, node) -> None:
        """endfor -- handled by for body, skip."""
        pass

    def _emit_each(self, node) -> None:
        """each VAR -- GPU: compute global thread index."""
        var_name = self._get(node, "var")
        if self.is_gpu:
            from emit_sm90 import SR_TID_X, SR_CTAID_X, RZ
            idx_reg = self.regs.get_or_alloc(var_name)
            tid_reg = self.regs.temp()
            ctaid_reg = self.regs.temp()
            self.emitter.emit_s2r(tid_reg, SR_TID_X)
            self.emitter.emit_s2r(ctaid_reg, SR_CTAID_X)
            blockdim_reg = self.regs.temp()
            self.emitter.emit_mov_imm(blockdim_reg, 256)
            self.emitter.emit_imad(idx_reg, ctaid_reg, blockdim_reg, tid_reg)
        else:
            self.emitter.emit_nop()

    def _emit_shared(self, node) -> None:
        """shared -- GPU-specific, skip."""
        pass

    def _emit_stride(self, node) -> None:
        """stride -- GPU-specific, skip."""
        pass

    def _emit_param(self, node) -> None:
        """param -- handled at function level, skip."""
        pass

    def _emit_arraystore(self, node) -> None:
        """target [ index ] = value -- array element store."""
        target_name = self._get(node, "target")
        index_node = self._get(node, "index")
        value_node = self._get(node, "value")

        base_reg = self._emit_expr(Ident(name=target_name) if isinstance(target_name, str) else target_name)
        idx_reg = self._emit_expr(index_node)
        val_reg = self._emit_expr(value_node)

        if self.is_gpu:
            offset_reg = self.regs.temp()
            four_reg = self.regs.temp()
            self.emitter.emit_mov_imm(four_reg, 4)
            from emit_sm90 import RZ
            self.emitter.emit_imad(offset_reg, idx_reg, four_reg, RZ)
            addr_reg = self.regs.temp()
            self.emitter.emit_iadd3(addr_reg, base_reg, offset_reg, RZ)
            self.emitter.emit_store(val_reg, addr_reg, 0)
        else:
            addr_reg = self.regs.temp()
            self.emitter.emit_add(addr_reg, base_reg, idx_reg)
            self.emitter.emit_store(val_reg, addr_reg, 0)

    # ----------------------------------------------------------
    # Expression evaluator -- returns register number
    # ----------------------------------------------------------

    def _emit_expr(self, node) -> int:
        """Emit code for an expression, return the register holding the result."""
        if isinstance(node, int):
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, node)
            return reg

        if isinstance(node, str):
            sym = self.symtab.lookup(node)
            if sym and sym.kind == "const":
                reg = self.regs.temp()
                self.emitter.emit_mov_imm(reg, sym.value)
                return reg
            r = self.regs.lookup(node)
            if r is not None:
                return r
            return self.regs.get_or_alloc(node)

        if isinstance(node, IntLit):
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, node.value)
            return reg

        if isinstance(node, FloatLit):
            import struct as _struct
            f32_bits = _struct.unpack('<I', _struct.pack('<f', node.value))[0]
            reg = self.regs.temp()
            self.emitter.emit_mov_imm(reg, f32_bits)
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
            if self.is_gpu:
                from emit_sm90 import RZ
                offset_reg = self.regs.temp()
                four_reg = self.regs.temp()
                self.emitter.emit_mov_imm(four_reg, 4)
                self.emitter.emit_imad(offset_reg, idx_reg, four_reg, RZ)
                addr_reg = self.regs.temp()
                self.emitter.emit_iadd3(addr_reg, base_reg, offset_reg, RZ)
                dst_reg = self.regs.temp()
                self.emitter.emit_load(dst_reg, addr_reg, 0)
                return dst_reg
            else:
                addr_reg = self.regs.temp()
                self.emitter.emit_add(addr_reg, base_reg, idx_reg)
                dst_reg = self.regs.temp()
                self.emitter.emit_load(dst_reg, addr_reg, 0)
                return dst_reg

        if isinstance(node, Reduction):
            return self._emit_expr(node.operand)

        if isinstance(node, UnaryOp):
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
