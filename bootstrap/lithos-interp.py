#!/usr/bin/env python3
"""lithos-interp.py — Reference interpreter for Lithos.

Runs .ls source directly in Python, no codegen. Purpose: give us a
verified-correct execution of Lithos programs so we can diff wirth's
compiled output against what the program should do.

Usage: python3 lithos-interp.py source.ls [args...]
Exits with whatever status the program passes to the exit syscall.
"""

import sys
import os
import struct

sys.setrecursionlimit(500000)
import threading
# Run main logic in a thread with a larger stack to survive deep recursion
# seen in the reference interpretation of compiler-darwin.ls.

(TOK_EOF, TOK_NEWLINE, TOK_INDENT, TOK_INT, TOK_IDENT,
 TOK_IF, TOK_ELIF, TOK_ELSE, TOK_FOR, TOK_WHILE,
 TOK_EACH, TOK_VAR, TOK_RETURN, TOK_TRAP, TOK_CONST,
 TOK_BUF, TOK_LOAD, TOK_REG_READ, TOK_MEM_STORE, TOK_MEM_LOAD,
 TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
 TOK_AMP, TOK_PIPE, TOK_CARET, TOK_SHL, TOK_SHR,
 TOK_EQ, TOK_EQEQ, TOK_NEQ, TOK_LT, TOK_GT, TOK_LTE, TOK_GTE,
 TOK_LPAREN, TOK_RPAREN, TOK_LBRACK, TOK_RBRACK,
 TOK_COLON, TOK_DOLLAR, TOK_HASH,
 TOK_GOTO, TOK_LABEL, TOK_CONTINUE, TOK_BREAK) = range(47)

KEYWORDS = {
    'if': TOK_IF, 'elif': TOK_ELIF, 'else': TOK_ELSE,
    'for': TOK_FOR, 'while': TOK_WHILE, 'each': TOK_EACH,
    'var': TOK_VAR, 'return': TOK_RETURN, 'trap': TOK_TRAP,
    'const': TOK_CONST, 'buf': TOK_BUF,
    'goto': TOK_GOTO, 'label': TOK_LABEL,
    'continue': TOK_CONTINUE, 'break': TOK_BREAK,
}

OP_TOKENS = {TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
             TOK_AMP, TOK_PIPE, TOK_CARET, TOK_SHL, TOK_SHR}

CMP_TOKENS = {TOK_LT, TOK_GT, TOK_LTE, TOK_GTE, TOK_EQEQ, TOK_NEQ}

EXPR_STARTERS = {TOK_INT, TOK_IDENT, TOK_LPAREN, TOK_MINUS,
                 TOK_REG_READ, TOK_MEM_LOAD}

def lex(src: bytes):
    toks = []
    i = 0
    line = 1
    at_line_start = True

    def emit(tp, off, length):
        toks.append((tp, off, length, line))

    while i < len(src):
        if at_line_start:
            indent = 0
            while i < len(src) and src[i:i+1] == b' ':
                indent += 1
                i += 1
            if i >= len(src):
                break
            if src[i:i+1] == b'\n':
                at_line_start = False
                continue
            if src[i:i+2] == b'\\\\':
                at_line_start = False
                continue
            emit(TOK_INDENT, i - indent, indent)
            at_line_start = False

        if i >= len(src):
            break
        c = src[i:i+1]

        if c in (b' ', b'\t'):
            i += 1
            continue

        if c == b'\n':
            emit(TOK_NEWLINE, i, 1)
            i += 1
            line += 1
            at_line_start = True
            continue

        if src[i:i+2] == b'\\\\':
            while i < len(src) and src[i:i+1] != b'\n':
                i += 1
            continue

        # identifier or keyword
        if c.isalpha() or c == b'_':
            start = i
            while i < len(src):
                ch = src[i:i+1]
                if ch.isalnum() or ch == b'_':
                    i += 1
                else:
                    break
            name = src[start:i].decode()
            emit(KEYWORDS.get(name, TOK_IDENT), start, i - start)
            continue

        # $N
        if c == b'$':
            start = i
            i += 1
            while i < len(src) and src[i:i+1].isdigit():
                i += 1
            emit(TOK_IDENT, start, i - start)
            continue

        # integer
        if c.isdigit():
            start = i
            if src[i:i+2] in (b'0x', b'0X'):
                i += 2
                while i < len(src):
                    ch = src[i:i+1]
                    if ch.isdigit() or ch in b'abcdefABCDEF':
                        i += 1
                    else:
                        break
            else:
                while i < len(src) and src[i:i+1].isdigit():
                    i += 1
            emit(TOK_INT, start, i - start)
            continue

        two = src[i:i+2]
        if two == b'==': emit(TOK_EQEQ, i, 2); i += 2; continue
        if two == b'!=': emit(TOK_NEQ, i, 2); i += 2; continue
        if two == b'<=': emit(TOK_LTE, i, 2); i += 2; continue
        if two == b'>=': emit(TOK_GTE, i, 2); i += 2; continue
        if two == b'<<': emit(TOK_SHL, i, 2); i += 2; continue
        if two == b'>>': emit(TOK_SHR, i, 2); i += 2; continue

        three = src[i:i+3]
        if three == b'\xe2\x86\x93': emit(TOK_LOAD, i, 3); i += 3; continue
        if three == b'\xe2\x86\x91': emit(TOK_REG_READ, i, 3); i += 3; continue
        if three == b'\xe2\x86\x90': emit(TOK_MEM_STORE, i, 3); i += 3; continue
        if three == b'\xe2\x86\x92': emit(TOK_MEM_LOAD, i, 3); i += 3; continue

        single_map = {
            b'+': TOK_PLUS, b'-': TOK_MINUS, b'*': TOK_STAR, b'/': TOK_SLASH,
            b'=': TOK_EQ, b'<': TOK_LT, b'>': TOK_GT,
            b'&': TOK_AMP, b'|': TOK_PIPE, b'^': TOK_CARET,
            b'(': TOK_LPAREN, b')': TOK_RPAREN,
            b'[': TOK_LBRACK, b']': TOK_RBRACK,
            b':': TOK_COLON, b'#': TOK_HASH,
        }
        if c in single_map:
            emit(single_map[c], i, 1)
            i += 1
            continue

        i += 1  # unknown

    emit(TOK_EOF, i, 0)
    return toks

class ReturnVal(Exception):
    def __init__(self, v): self.value = v

class BreakLoop(Exception):  pass
class ContinueLoop(Exception): pass
class GotoLabel(Exception):
    def __init__(self, name): self.name = name
class ExitProgram(Exception):
    def __init__(self, code): self.code = code

MASK64 = (1 << 64) - 1

def m(v): return v & MASK64

def sx(v, bits=64):
    """sign-extend from 64 bits to Python int"""
    sign = 1 << (bits - 1)
    return (v & (sign - 1)) - (v & sign)

class Interp:
    def __init__(self, src, argv):
        self.src = src
        self.toks = lex(src)
        self.tk = 0
        self.argv = argv

        self.compositions = {}   # name -> {params, body_tk, body_indent}
        self.consts = {}         # name -> int
        self.bufs = {}           # name -> (mem_off, size)
        self.globals = {}        # name -> int
        self.frames = []         # stack of {name: value}

        # 1 GiB virtual memory. Bufs and globals live here at offset 0+.
        # Addresses returned to the program ARE direct offsets.
        self.mem = bytearray(1 << 30)
        self.mem_top = 8         # leave 0 as NULL sentinel

        self.regs = [0] * 32     # X0..X31

        # Syscall result tracking
        self.open_fds = {}       # "internal fd" -> real os fd
        self.next_fake_fd = 100

        self.last_call_extras = []
        self.frame_param_counts = []
    def cur(self): return self.toks[self.tk]
    def ty(self): return self.toks[self.tk][0]
    def tok_text(self, t):
        return self.src[t[1]:t[1] + t[2]].decode('utf-8', errors='replace')
    def die(self, msg):
        t = self.cur()
        raise RuntimeError(f"interp line {t[3]}: {msg}")
    def lookup(self, name):
        if self.frames and name in self.frames[-1]:
            return self.frames[-1][name]
        if name in self.consts: return self.consts[name]
        if name in self.globals: return self.globals[name]
        if name in self.bufs:
            off, sz = self.bufs[name]
            return off
        return None

    def store(self, name, value):
        v = m(value)
        if self.frames and name in self.frames[-1]:
            self.frames[-1][name] = v
            return True
        if name in self.globals:
            self.globals[name] = v
            return True
        return False

    def new_local(self, name, value):
        self.frames[-1][name] = m(value)
    def mem_read(self, addr, width):
        a = m(addr)
        w = width // 8 if width >= 8 else 1
        if a + w > len(self.mem): return 0
        if width == 8:  return self.mem[a]
        if width == 16: return struct.unpack_from('<H', self.mem, a)[0]
        if width == 32: return struct.unpack_from('<I', self.mem, a)[0]
        if width == 64: return struct.unpack_from('<Q', self.mem, a)[0]
        return 0

    def mem_write(self, addr, width, val):
        a = m(addr)
        v = m(val)
        w = width // 8 if width >= 8 else 1
        if a + w > len(self.mem): return
        if width == 8:  self.mem[a] = v & 0xFF
        elif width == 16: struct.pack_into('<H', self.mem, a, v & 0xFFFF)
        elif width == 32: struct.pack_into('<I', self.mem, a, v & 0xFFFFFFFF)
        elif width == 64: struct.pack_into('<Q', self.mem, a, v)
    def parse_int(self, t):
        s = self.tok_text(t)
        neg = 1
        if s.startswith('-'):
            neg = -1
            s = s[1:]
        if s.startswith(('0x', '0X')):
            return m(neg * int(s, 16))
        return m(neg * int(s))

    def dollar_reg(self, t):
        if t[2] >= 2 and self.src[t[1]:t[1]+1] == b'$':
            try:
                return int(self.src[t[1]+1:t[1]+t[2]])
            except ValueError:
                return -1
        return -1

    def maybe_eat_cont(self, op_types):
        """If at NEWLINE, peek past it + INDENT for a continuation operator."""
        if self.ty() != TOK_NEWLINE:
            return
        i = self.tk + 1
        while i < len(self.toks) and self.toks[i][0] == TOK_NEWLINE:
            i += 1
        if i >= len(self.toks) or self.toks[i][0] != TOK_INDENT:
            return
        if self.toks[i][2] <= 0:
            return
        j = i + 1
        if j >= len(self.toks) or self.toks[j][0] not in op_types:
            return
        self.tk = i + 1

    def parse_primary(self):
        t = self.cur()

        if t[0] == TOK_INT:
            self.tk += 1
            return self.parse_int(t)

        if t[0] == TOK_LPAREN:
            self.tk += 1
            v = self.parse_expr()
            if self.ty() == TOK_RPAREN:
                self.tk += 1
            return v

        if t[0] == TOK_MINUS:
            self.tk += 1
            return m(-sx(self.parse_primary()))

        # $N
        if t[0] == TOK_IDENT and t[2] >= 2 and self.src[t[1]:t[1]+1] == b'$':
            n = self.dollar_reg(t)
            self.tk += 1
            if 0 <= n < 32:
                return self.regs[n]
            return 0

        if t[0] == TOK_REG_READ:
            self.tk += 1
            rt = self.cur()
            n = self.dollar_reg(rt) if rt[0] == TOK_IDENT else -1
            if n >= 0:
                self.tk += 1
                return self.regs[n]
            return 0

        if t[0] == TOK_MEM_LOAD:
            self.tk += 1
            wt = self.cur()
            if wt[0] != TOK_INT:
                self.die("expected width after →")
            width = int(self.tok_text(wt))
            self.tk += 1
            base = self.parse_primary()
            addr = base
            nx = self.ty()
            if nx in (TOK_INT, TOK_IDENT):
                off = self.parse_primary()
                addr = m(base + off)
            return self.mem_read(addr, width)

        # Keywords used as identifiers
        if t[0] in (TOK_BUF, TOK_CONST, TOK_VAR, TOK_LABEL):
            t = (TOK_IDENT, t[1], t[2], t[3])

        if t[0] == TOK_IDENT:
            name = self.tok_text(t)

            # Intrinsics
            if name == 'max' or name == 'min':
                self.tk += 1
                a = self.parse_primary()
                b = self.parse_primary()
                sa = sx(a); sb = sx(b)
                r = max(sa, sb) if name == 'max' else min(sa, sb)
                return m(r)

            self.tk += 1

            # Array subscript `name[idx]` — byte load
            if self.ty() == TOK_LBRACK:
                self.tk += 1
                idx = self.parse_expr()
                if self.ty() == TOK_RBRACK:
                    self.tk += 1
                base = self.lookup(name)
                if base is None: return 0
                return self.mem_read(m(base + idx), 8)

            # Composition call?
            if name in self.compositions:
                return self.call_compo(name, atom_mode=True)

            v = self.lookup(name)
            if v is None: return 0  # tolerance
            return v

        # tolerance for unexpected
        return 0

    def parse_mul(self):
        a = self.parse_primary()
        while True:
            self.maybe_eat_cont((TOK_STAR, TOK_SLASH))
            op = self.ty()
            if op != TOK_STAR and op != TOK_SLASH:
                break
            self.tk += 1
            b = self.parse_primary()
            if op == TOK_STAR:
                a = m(sx(a) * sx(b))
            else:
                sb = sx(b)
                a = 0 if sb == 0 else m(int(sx(a) / sb))
        return a

    def parse_add(self):
        a = self.parse_mul()
        while True:
            self.maybe_eat_cont((TOK_PLUS, TOK_MINUS))
            op = self.ty()
            if op != TOK_PLUS and op != TOK_MINUS:
                break
            self.tk += 1
            b = self.parse_mul()
            a = m(a + b) if op == TOK_PLUS else m(a - b)
        return a

    def parse_shift(self):
        a = self.parse_add()
        while True:
            self.maybe_eat_cont((TOK_SHL, TOK_SHR))
            op = self.ty()
            if op != TOK_SHL and op != TOK_SHR:
                break
            self.tk += 1
            b = self.parse_add()
            sh = b & 63
            a = m(a << sh) if op == TOK_SHL else (a >> sh)
        return a

    def parse_cmp(self):
        a = self.parse_shift()
        op = self.ty()
        if op not in CMP_TOKENS:
            return a
        self.tk += 1
        b = self.parse_shift()
        sa, sb = sx(a), sx(b)
        if op == TOK_LT:   return 1 if sa < sb else 0
        if op == TOK_GT:   return 1 if sa > sb else 0
        if op == TOK_LTE:  return 1 if sa <= sb else 0
        if op == TOK_GTE:  return 1 if sa >= sb else 0
        if op == TOK_EQEQ: return 1 if a == b else 0
        if op == TOK_NEQ:  return 1 if a != b else 0
        return 0

    def parse_bits(self):
        a = self.parse_cmp()
        while True:
            self.maybe_eat_cont((TOK_AMP, TOK_PIPE, TOK_CARET))
            op = self.ty()
            if op not in (TOK_AMP, TOK_PIPE, TOK_CARET):
                break
            self.tk += 1
            b = self.parse_cmp()
            if op == TOK_AMP:   a = a & b
            elif op == TOK_PIPE: a = a | b
            else:                a = a ^ b
        return a

    def parse_expr(self):
        return self.parse_bits()
    def call_compo(self, name, atom_mode=False):
        # Instrumentation
        if not hasattr(self, '_call_counts'):
            self._call_counts = {}
            self._call_depth = 0
            self._max_depth = 0
            self._trace_stack = []
        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        self._call_depth += 1
        if self._call_depth > self._max_depth:
            self._max_depth = self._call_depth
        self._trace_stack.append(name)
        # Trace first 10 parse_file calls + periodic
        if name == 'parse_file':
            cnt = self._call_counts.get(name, 0)
            if cnt <= 15 or cnt % 500 == 0:
                tp = self.globals.get('tok_pos', '<none>')
                tt = self.globals.get('tok_total', '<none>')
                import sys as _sys
                tokens_addr = self.bufs.get('tokens', (0, 0))[0]
                ty_val = 'N/A'
                if isinstance(tp, int):
                    idx = tp * 3
                    try:
                        ty_val = self.mem_read(tokens_addr + idx*4, 32)
                    except Exception:
                        ty_val = 'err'
                print(f"[trace] parse_file #{cnt} depth={self._call_depth} tok_pos={tp}/{tt} mem_type_at_pos={ty_val}", file=_sys.stderr)
                _sys.stderr.flush()
        # Only hard-cap if truly pathological.
        if self._call_depth > 2000 and name == 'parse_file':
            import sys as _sys
            tp = self.globals.get('tok_pos', '<none>')
            tt = self.globals.get('tok_total', '<none>')
            print(f"DEPTH={self._call_depth} parse_file tok_pos={tp} tok_total={tt}", file=_sys.stderr)
            _sys.stderr.flush()
            raise RuntimeError(f"call depth exceeded at {name}, tok_pos={tp}/{tt}")
        compo = self.compositions[name]
        params = compo['params']
        args = []

        # Collect args (atom-level in expression context, expr-level in stmt)
        while len(args) < len(params):
            t = self.ty()
            if t in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
                break
            if t not in EXPR_STARTERS:
                break
            if atom_mode:
                # Stop at operators so caller's expression parser can combine
                if t in OP_TOKENS or t in CMP_TOKENS or t == TOK_RPAREN:
                    break
                args.append(self.parse_primary())
            else:
                args.append(self.parse_expr())

        # Top-level entry (no active caller frame): pull remaining params
        # from X0.. — matches wirth's _start trampoline which loads argc
        # into X0 and argv into X1 before branching to main.  Without this,
        # `main argc argv :` sees argc=0 and instantly goto'es main_exit_err.
        if not self.frames:
            while len(args) < len(params):
                i = len(args)
                args.append(self.regs[i] if i < 32 else 0)

        while len(args) < len(params):
            args.append(0)

        # New frame
        frame = dict(zip(params, args))
        self.frames.append(frame)
        self.frame_param_counts.append(len(params))

        saved_tk = self.tk
        self.tk = compo['body_tk']

        ret = 0
        explicit_return = False
        try:
            ret = self.exec_loop(compo['body_indent'])
        except ReturnVal as r:
            ret = r.value
            explicit_return = True
        # Save the callee's locals so a multi-return caller can read extra
        # values. Order: reversed (last declared first), so callers like
        # `a b c foo args` naturally bind to the most-recently-captured values.
        last_frame = self.frames[-1]
        param_count = len(compo['params'])
        locals_in_order = list(last_frame.values())[param_count:]
        extras = list(reversed(locals_in_order))
        self.last_call_extras = extras
        self.frames.pop()
        self.frame_param_counts.pop()
        self.tk = saved_tk

        # Removed: previous fallback that returned the first non-zero
        # local when ret == 0 was overriding LEGITIMATE 0 returns.
        # E.g. peek_type returning 0 at EOF was being clobbered with the
        # local `idx` value. Source code should explicitly marshal
        # multi-return values into globals (mmap_file does this now).
        self._call_depth -= 1
        self._trace_stack.pop()
        if name == 'match_keyword' and os.environ.get('LITHOS_TRACE_MATCH_KEYWORD'):
            import sys as _sys
            try:
                src_arg = args[0]
                off_arg = args[1]
                ln_arg = args[2]
                text = bytes(self.mem[src_arg + off_arg:src_arg + off_arg + ln_arg]).decode('utf-8', errors='replace')
            except Exception:
                text = '<unk>'
            print(f"TRACE match_keyword({text!r}, len={args[2] if len(args)>=3 else '?'}) -> {ret} (explicit={explicit_return})", file=_sys.stderr)
            _sys.stderr.flush()
        return ret
    def exec_loop(self, body_indent):
        """Execute statements at body_indent until dedent."""
        last = 0
        loop_start = self.tk
        while True:
            if self.ty() == TOK_EOF:
                break
            if self.ty() == TOK_NEWLINE:
                self.tk += 1
                continue
            if self.ty() == TOK_INDENT:
                if self.tk + 1 < len(self.toks) and self.toks[self.tk+1][0] == TOK_NEWLINE:
                    self.tk += 2
                    continue
                if self.toks[self.tk][2] < body_indent:
                    break
                # Check if this is a continuation line (starts with binary op)
                nxt_ty = self.toks[self.tk+1][0] if self.tk + 1 < len(self.toks) else TOK_EOF
                if nxt_ty in OP_TOKENS:
                    self.tk += 1
                    last = self._eval_op_chain(last)
                    continue
                self.tk += 1
                # Fall through to exec statement
            else:
                break
            try:
                last = self.exec_stmt()
            except GotoLabel as g:
                target_tk = self._find_label(loop_start, body_indent, g.name)
                if target_tk < 0:
                    # not found in this body — propagate to outer loop
                    raise
                self.tk = target_tk
        return last

    def _find_label(self, start_tk, body_indent, name):
        """Scan body tokens at body_indent from start_tk, find `label NAME`.
        Returns position after the `label NAME` pair, or -1."""
        i = start_tk
        n = len(self.toks)
        while i < n:
            t = self.toks[i]
            if t[0] == TOK_EOF:
                return -1
            if t[0] == TOK_INDENT:
                if t[2] < body_indent:
                    return -1
                # statement at this indent?
                if t[2] == body_indent and i + 1 < n:
                    nt = self.toks[i+1]
                    if nt[0] == TOK_LABEL and i + 2 < n:
                        nm = self.toks[i+2]
                        if nm[0] == TOK_IDENT and self.tok_text(nm) == name:
                            return i + 3
                    # Python-style `NAME :` label
                    if nt[0] == TOK_IDENT and i + 2 < n and \
                       self.toks[i+2][0] == TOK_COLON and \
                       self.tok_text(nt) == name:
                        return i + 3
            i += 1
        return -1

    def skip_body(self, body_indent):
        """Skip tokens for a body at body_indent."""
        while self.ty() != TOK_EOF:
            if self.ty() == TOK_NEWLINE:
                self.tk += 1
                continue
            if self.ty() == TOK_INDENT:
                if self.toks[self.tk][2] < body_indent:
                    return
                self.tk += 1
                # skip to newline
                while self.ty() not in (TOK_NEWLINE, TOK_EOF):
                    self.tk += 1
            else:
                return

    def exec_stmt(self):
        # Called with tk at a statement start token (past INDENT)
        t = self.cur()

        if t[0] == TOK_TRAP:
            self.tk += 1
            nx = self.ty()
            if nx in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
                return self.do_syscall()
            # `trap name num [args...]`
            if nx == TOK_IDENT:
                name_tok = self.cur()
                name = self.tok_text(name_tok)
                self.tk += 1
                num = self.parse_primary()
                self.regs[16] = num
                args = []
                while len(args) < 8:
                    if self.ty() in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
                        break
                    if self.ty() not in EXPR_STARTERS:
                        break
                    args.append(self.parse_expr())
                for i, v in enumerate(args):
                    self.regs[i] = v
                result = self.do_syscall()
                if self.frames:
                    self.new_local(name, result)
                else:
                    self.globals[name] = result
                return result
            return 0

        if t[0] == TOK_CONST:
            self.tk += 1
            nt = self.cur()
            if nt[0] not in (TOK_IDENT, TOK_BUF, TOK_VAR, TOK_LABEL, TOK_CONST):
                self.die("expected const name")
            name = self.tok_text(nt)
            self.tk += 1
            if self.ty() != TOK_INT:
                self.die("const must be int literal")
            val = self.parse_primary()
            self.consts[name] = val
            return 0

        if t[0] == TOK_BUF:
            self.tk += 1
            nt = self.cur()
            if nt[0] not in (TOK_IDENT, TOK_BUF, TOK_VAR, TOK_LABEL, TOK_CONST):
                self.die("expected buf name")
            name = self.tok_text(nt)
            self.tk += 1
            if self.ty() != TOK_INT:
                self.die("buf size must be int")
            size = self.parse_primary()
            off = self.mem_top
            self.mem_top = (off + size + 7) & ~7
            if self.mem_top > len(self.mem):
                self.die("buf overflow")
            self.bufs[name] = (off, size)
            return 0

        if t[0] == TOK_VAR:
            self.tk += 1
            nt = self.cur()
            if nt[0] not in (TOK_IDENT, TOK_BUF, TOK_VAR, TOK_LABEL, TOK_CONST):
                self.die("expected var name")
            name = self.tok_text(nt)
            self.tk += 1
            nx = self.ty()
            val = self.parse_expr() if nx in EXPR_STARTERS else 0
            if self.frames:
                self.new_local(name, val)
            else:
                self.globals[name] = val
            return val

        if t[0] == TOK_RETURN:
            self.tk += 1
            nx = self.ty()
            if nx in EXPR_STARTERS:
                v = self.parse_expr()
            else:
                # Bare `return` — compiler emits raw `ret` with whatever is
                # in X0. Register allocator keeps the FIRST non-parameter
                # local in X0 (the primary working variable). Emulate that.
                v = 0
                if self.frames:
                    last_frame = self.frames[-1]
                    pcount = self.frame_param_counts[-1] if self.frame_param_counts else 0
                    vals = list(last_frame.values())
                    # First local after the parameters
                    if len(vals) > pcount:
                        v = vals[pcount]
                    elif vals:
                        v = vals[-1]
            raise ReturnVal(v)

        if t[0] == TOK_LOAD:  # ↓
            self.tk += 1
            dt = self.cur()
            n = self.dollar_reg(dt) if dt[0] == TOK_IDENT else -1
            if n >= 0:
                self.tk += 1
                v = self.parse_expr()
                self.regs[n] = m(v)
                return v
            # ↓ width addr val (rarely used in compiler.ls)
            width = self.parse_primary()
            addr = self.parse_expr()
            val = self.parse_expr()
            self.mem_write(addr, width, val)
            return val

        if t[0] == TOK_MEM_STORE:  # ← width addr [off] val
            self.tk += 1
            wt = self.cur()
            if wt[0] != TOK_INT:
                self.die("expected width after ←")
            width = int(self.tok_text(wt))
            self.tk += 1
            base = self.parse_expr()
            middle = self.parse_expr()
            nx = self.ty()
            if nx in EXPR_STARTERS:
                addr = m(base + middle)
                val = self.parse_expr()
            else:
                addr = base
                val = middle
            self.mem_write(addr, width, val)
            return val

        if t[0] == TOK_REG_READ:  # ↑ $N bare
            v = self.parse_expr()
            self.regs[0] = m(v)
            return v

        if t[0] == TOK_MEM_LOAD:  # → width addr  as expression statement
            # The last expression in a composition body is its return value,
            # by convention kept in X0. Without this case, `→ 32 tokens idx`
            # on its own line is dropped and the function returns garbage.
            v = self.parse_expr()
            self.regs[0] = m(v)
            return v

        if t[0] == TOK_IF:
            return self.exec_if_chain()

        if t[0] == TOK_WHILE:
            return self.exec_while()

        if t[0] == TOK_FOR:
            return self.exec_for()

        if t[0] == TOK_EACH:
            return self.exec_each()

        if t[0] == TOK_CONTINUE:
            self.tk += 1
            raise ContinueLoop()
        if t[0] == TOK_BREAK:
            self.tk += 1
            raise BreakLoop()

        if t[0] == TOK_GOTO:
            self.tk += 1
            if self.ty() == TOK_IDENT:
                name = self.tok_text(self.cur())
                self.tk += 1
                raise GotoLabel(name)
            return 0
        if t[0] == TOK_LABEL:
            self.tk += 1
            if self.ty() == TOK_IDENT:
                self.tk += 1
            return 0

        # Python-style `name:` label
        if t[0] == TOK_IDENT and self.tk + 1 < len(self.toks) and \
           self.toks[self.tk+1][0] == TOK_COLON:
            self.tk += 2
            return 0

        # IDENT statement: binding, reassignment, or call
        if t[0] in (TOK_IDENT, TOK_BUF, TOK_CONST, TOK_VAR, TOK_LABEL):
            return self.exec_ident_stmt()

        # Unknown — skip to newline
        while self.ty() not in (TOK_NEWLINE, TOK_EOF):
            self.tk += 1
        return 0

    def exec_ident_stmt(self):
        t = self.cur()
        name = self.tok_text(t)
        self.tk += 1

        # Known composition → call
        if name in self.compositions:
            r = self.call_compo(name, atom_mode=False)
            self.regs[0] = r
            return r

        # Known local/param → reassignment (possibly short-form)
        if self.frames and name in self.frames[-1]:
            return self.do_assign_local(name)
        # Known global → reassignment
        if name in self.globals:
            return self.do_assign_global(name)
        # Known const → can't reassign, treat as read
        if name in self.consts:
            r = self.consts[name]
            self.regs[0] = r
            return r
        # Known buf → read address (address of buf)
        if name in self.bufs:
            off, sz = self.bufs[name]
            self.regs[0] = off
            return off

        # New name: binding
        if self.ty() == TOK_EQ:
            self.tk += 1
        if self.ty() in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
            # bare call to unknown — treat as no-op
            return 0
        # Multi-assign pattern: `a b c foo args` — collect unknown IDENTs
        # until we hit a known composition.
        extra_names = []
        while self.ty() == TOK_IDENT:
            tcur = self.cur()
            nm = self.tok_text(tcur)
            if nm in self.compositions:
                break
            # Already-defined symbols just get read as part of expr, stop here
            if (self.frames and nm in self.frames[-1]) or \
               nm in self.globals or nm in self.consts or nm in self.bufs:
                break
            # Peek: is the NEXT token something that would make this current
            # ident a standalone expression (operator)? If so, it's a real
            # value read (unknown -> 0 fallback). Otherwise treat as extra
            # assignment target.
            nxt_ty = self.toks[self.tk+1][0] if self.tk + 1 < len(self.toks) else TOK_EOF
            if nxt_ty in OP_TOKENS or nxt_ty in CMP_TOKENS or \
               nxt_ty == TOK_LBRACK or nxt_ty in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
                break
            extra_names.append(nm)
            self.tk += 1
        # If we now sit on a known composition, call it for the value.
        # Extra return values come from the callee's last-declared locals.
        if self.ty() == TOK_IDENT:
            nm = self.tok_text(self.cur())
            if nm in self.compositions:
                self.tk += 1  # consume the composition name
                val = self.call_compo(nm, atom_mode=False)
                extras = self.last_call_extras  # reversed locals
                # `name` gets the return value (first slot of reversed locals)
                # `extra_names[i]` gets extras[i+1]
                if self.frames:
                    self.new_local(name, val)
                    for i, en in enumerate(extra_names):
                        ev = extras[i+1] if i+1 < len(extras) else 0
                        self.new_local(en, ev)
                else:
                    self.globals[name] = m(val)
                    for i, en in enumerate(extra_names):
                        ev = extras[i+1] if i+1 < len(extras) else 0
                        self.globals[en] = m(ev)
                return val
        val = self.parse_expr()
        if self.frames:
            self.new_local(name, val)
            for en in extra_names:
                self.new_local(en, val)
        else:
            self.globals[name] = m(val)
            for en in extra_names:
                self.globals[en] = m(val)
        return val

    def do_assign_local(self, name):
        if self.ty() == TOK_EQ:
            self.tk += 1
        # Short-form `name OP expr`?
        cur_ty = self.ty()
        if cur_ty in OP_TOKENS:
            cur_val = self.frames[-1][name]
            val = self._eval_op_chain(cur_val)
            self.frames[-1][name] = m(val)
            self.regs[0] = m(val)
            return val
        # Bare read
        if cur_ty in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
            v = self.frames[-1][name]
            self.regs[0] = v
            return v
        # name expr
        val = self.parse_expr()
        self.frames[-1][name] = m(val)
        self.regs[0] = m(val)
        return val

    def do_assign_global(self, name):
        if self.ty() == TOK_EQ:
            self.tk += 1
        cur_ty = self.ty()
        if cur_ty in OP_TOKENS:
            cur_val = self.globals[name]
            val = self._eval_op_chain(cur_val)
            self.globals[name] = m(val)
            self.regs[0] = m(val)
            return val
        if cur_ty in (TOK_NEWLINE, TOK_EOF, TOK_INDENT):
            v = self.globals[name]
            self.regs[0] = v
            return v
        val = self.parse_expr()
        self.globals[name] = m(val)
        self.regs[0] = m(val)
        return val

    def _eval_op_chain(self, start):
        """Evaluate `OP expr OP expr ...` starting from accumulator `start`."""
        a = m(start)
        while True:
            self.maybe_eat_cont(tuple(OP_TOKENS))
            op = self.ty()
            if op not in OP_TOKENS:
                break
            self.tk += 1
            b = self.parse_primary()
            if op == TOK_PLUS:  a = m(a + b)
            elif op == TOK_MINUS: a = m(a - b)
            elif op == TOK_STAR:  a = m(sx(a) * sx(b))
            elif op == TOK_SLASH:
                a = 0 if sx(b) == 0 else m(int(sx(a) / sx(b)))
            elif op == TOK_AMP:   a = a & b
            elif op == TOK_PIPE:  a = a | b
            elif op == TOK_CARET: a = a ^ b
            elif op == TOK_SHL:   a = m(a << (b & 63))
            elif op == TOK_SHR:   a = a >> (b & 63)
        return a
    def _skip_empty_lines(self):
        """Skip TOK_NEWLINE and empty TOK_INDENT+NEWLINE pairs."""
        while True:
            if self.ty() == TOK_NEWLINE:
                self.tk += 1
                continue
            if self.ty() == TOK_INDENT and self.tk + 1 < len(self.toks) \
               and self.toks[self.tk+1][0] == TOK_NEWLINE:
                self.tk += 2
                continue
            break

    def _peek_body_indent(self):
        """Peek the indent width of the body following the current position."""
        i = self.tk
        while i < len(self.toks):
            t = self.toks[i]
            if t[0] == TOK_NEWLINE:
                i += 1
                continue
            if t[0] == TOK_INDENT:
                if i + 1 < len(self.toks) and self.toks[i+1][0] == TOK_NEWLINE:
                    i += 2
                    continue
                return t[2], i
            return 0, i
        return 0, i

    def exec_if_chain(self):
        """Handle `if [compound] cond : body [elif ...] [else ...]`."""
        took_branch = False
        while True:
            kw = self.ty()
            if kw == TOK_IF or kw == TOK_ELIF:
                self.tk += 1
                cond = self.eval_if_cond()
                if self.ty() == TOK_COLON:
                    self.tk += 1
                self._skip_empty_lines()
                body_indent = self.toks[self.tk][2] if self.ty() == TOK_INDENT else 0
                if cond and not took_branch:
                    self.exec_loop(body_indent)
                    took_branch = True
                else:
                    self.skip_body(body_indent)
            elif kw == TOK_ELSE:
                self.tk += 1
                if self.ty() == TOK_COLON:
                    self.tk += 1
                self._skip_empty_lines()
                body_indent = self.toks[self.tk][2] if self.ty() == TOK_INDENT else 0
                if not took_branch:
                    self.exec_loop(body_indent)
                    took_branch = True
                else:
                    self.skip_body(body_indent)
                break  # no more chain
            else:
                break
            # Consume trailing newlines to reach next potential elif/else
            self._skip_empty_lines()
            if self.ty() == TOK_INDENT:
                # elif/else at current body-indent — advance past the INDENT
                # only if next is elif/else
                if self.tk + 1 < len(self.toks) and \
                   self.toks[self.tk+1][0] in (TOK_ELIF, TOK_ELSE):
                    self.tk += 1
                else:
                    break
            elif self.ty() in (TOK_ELIF, TOK_ELSE):
                pass
            else:
                break
        return 0

    def eval_if_cond(self):
        """After consuming `if` or `elif`, evaluate the condition up to `:` or body start."""
        ct = self.ty()
        if ct in CMP_TOKENS:
            self.tk += 1
            a = self.parse_expr()
            b = self.parse_expr()
            sa, sb = sx(a), sx(b)
            if ct == TOK_LT: return 1 if sa < sb else 0
            if ct == TOK_GT: return 1 if sa > sb else 0
            if ct == TOK_LTE: return 1 if sa <= sb else 0
            if ct == TOK_GTE: return 1 if sa >= sb else 0
            if ct == TOK_EQEQ: return 1 if a == b else 0
            if ct == TOK_NEQ: return 1 if a != b else 0
        a = self.parse_expr()
        # Inline cmp form
        rel = self.ty()
        if rel in CMP_TOKENS:
            self.tk += 1
            b = self.parse_expr()
            sa, sb = sx(a), sx(b)
            if rel == TOK_LT: return 1 if sa < sb else 0
            if rel == TOK_GT: return 1 if sa > sb else 0
            if rel == TOK_LTE: return 1 if sa <= sb else 0
            if rel == TOK_GTE: return 1 if sa >= sb else 0
            if rel == TOK_EQEQ: return 1 if a == b else 0
            if rel == TOK_NEQ: return 1 if a != b else 0
        return 1 if a != 0 else 0
    def exec_while(self):
        self.tk += 1
        cond_tk = self.tk
        # Body indent detection after first execution
        body_indent = 0
        first = True
        while True:
            self.tk = cond_tk
            cond = self.eval_if_cond()
            if self.ty() == TOK_COLON:
                self.tk += 1
            self._skip_empty_lines()
            if first:
                body_indent = self.toks[self.tk][2] if self.ty() == TOK_INDENT else 0
                first = False
            if not cond:
                # Skip body to exit
                self.skip_body(body_indent)
                return 0
            try:
                self.exec_loop(body_indent)
            except BreakLoop:
                self.skip_body(body_indent)
                return 0
            except ContinueLoop:
                pass
    def exec_for(self):
        self.tk += 1
        nt = self.cur()
        if nt[0] != TOK_IDENT:
            self.die("expected for-var")
        var_name = self.tok_text(nt)
        self.tk += 1
        start = self.parse_expr()
        end = self.parse_expr()
        step = 1
        if self.ty() in EXPR_STARTERS:
            step = self.parse_expr()
        if self.ty() == TOK_COLON: self.tk += 1
        self._skip_empty_lines()
        body_indent = self.toks[self.tk][2] if self.ty() == TOK_INDENT else 0
        body_tk = self.tk
        self.new_local(var_name, start)
        i = sx(start)
        e = sx(end)
        s = sx(step)
        if s == 0: s = 1
        while (s > 0 and i < e) or (s < 0 and i > e):
            self.frames[-1][var_name] = m(i)
            self.tk = body_tk
            try:
                self.exec_loop(body_indent)
            except BreakLoop:
                self.skip_body(body_indent)
                return 0
            except ContinueLoop:
                pass
            i += s
        self.tk = body_tk
        self.skip_body(body_indent)
        return 0
    def exec_each(self):
        self.tk += 1
        nt = self.cur()
        if nt[0] == TOK_IDENT:
            self.new_local(self.tok_text(nt), 0)
            self.tk += 1
        if self.ty() == TOK_COLON: self.tk += 1
        self._skip_empty_lines()
        body_indent = self.toks[self.tk][2] if self.ty() == TOK_INDENT else 0
        self.exec_loop(body_indent)
        return 0
    def do_syscall(self):
        """Execute syscall based on X16 (Darwin convention) or X8 (Linux)."""
        num = self.regs[16] if self.regs[16] else self.regs[8]
        x0 = self.regs[0]
        x1 = self.regs[1]
        x2 = self.regs[2]
        x3 = self.regs[3]
        x4 = self.regs[4]
        x5 = self.regs[5]

        # macOS syscalls
        if num == 1:    # exit
            raise ExitProgram(x0 & 0xFF)
        if num == 4:    # write
            # x0=fd, x1=buf, x2=count
            fd = x0
            buf = x1
            count = x2
            if count > 0 and buf < len(self.mem):
                data = bytes(self.mem[buf:buf+count])
                try:
                    n = os.write(fd if fd < 100 else self.open_fds.get(fd, fd),
                                 data)
                    self.regs[0] = n
                    return n
                except OSError as e:
                    self.regs[0] = -1
                    return -1
            self.regs[0] = 0
            return 0
        def _resolve_fd(fd):
            fv = sx(fd)
            if fv < 0 or fv > 10**9: return -1
            return fv if fv < 100 else self.open_fds.get(fv, -1)
        if num == 3:    # read
            try:
                rfd = _resolve_fd(x0)
                if rfd < 0:
                    self.regs[0] = -1; return -1
                data = os.read(rfd, x2)
                self.mem[x1:x1+len(data)] = data
                self.regs[0] = len(data); return len(data)
            except OSError:
                self.regs[0] = -1; return -1
        if num == 6:    # close
            try:
                rfd = _resolve_fd(x0)
                if rfd < 0:
                    self.regs[0] = -1; return -1
                os.close(rfd)
                fv = sx(x0)
                if fv >= 100 and fv in self.open_fds: del self.open_fds[fv]
                self.regs[0] = 0; return 0
            except OSError:
                self.regs[0] = -1; return -1
        if num == 197:  # mmap — emulate by reading the fd into our mem
            length = x1
            rfd = _resolve_fd(x4)
            if rfd < 0:
                self.regs[0] = 0; return 0
            try:
                os.lseek(rfd, x5, 0)
                data = os.read(rfd, length)
                alloc = max((length + 7) & ~7, 8)
                base = self.mem_top
                if base + alloc > len(self.mem):
                    self.regs[0] = 0; return 0
                self.mem[base:base+len(data)] = data
                self.mem_top += alloc
                self.regs[0] = base
                self.regs[1] = len(data)  # Lithos convention: size in X1
                return base
            except OSError:
                self.regs[0] = 0; return 0
        if num == 199:  # lseek
            try:
                rfd = _resolve_fd(x0)
                if rfd < 0:
                    self.regs[0] = -1; return -1
                r = os.lseek(rfd, sx(x1), x2)
                self.regs[0] = r; return r
            except OSError:
                self.regs[0] = -1; return -1
        if num in (73, 74):  # munmap, mprotect
            self.regs[0] = 0; return 0
        if num == 12:   # brk
            self.regs[0] = self.mem_top; return self.mem_top
        if num == 463:  # openat: x0=dirfd, x1=path, x2=flags, x3=mode
            path_bytes = bytearray()
            p = x1
            while p < len(self.mem) and self.mem[p] != 0:
                path_bytes.append(self.mem[p]); p += 1
            path = bytes(path_bytes).decode(errors='replace')
            flags_v = x2 & 0xFFFFFFFF
            # Lithos 1537 = O_WRONLY|O_CREAT|O_TRUNC
            try:
                py_flags = [os.O_RDONLY, os.O_WRONLY, os.O_RDWR, 0][flags_v & 3]
                if flags_v & 0x200: py_flags |= os.O_CREAT
                if flags_v & 0x400: py_flags |= os.O_TRUNC
                mode = x3 & 0o7777 if x3 else 0o644
                real_fd = os.open(path, py_flags, mode)
                fake = self.next_fake_fd
                self.next_fake_fd += 1
                self.open_fds[fake] = real_fd
                self.regs[0] = fake; return fake
            except OSError:
                self.regs[0] = -1; return -1

        # Unknown syscall — no-op, return 0
        self.regs[0] = 0
        return 0
    def collect(self):
        i = 0
        at_line_start = True
        cur_indent = 0
        n = len(self.toks)
        while i < n:
            t = self.toks[i]
            if t[0] == TOK_EOF: break
            if t[0] == TOK_NEWLINE:
                at_line_start = True
                cur_indent = 0
                i += 1; continue
            if t[0] == TOK_INDENT:
                cur_indent = t[2]
                at_line_start = True
                i += 1; continue
            if not at_line_start:
                i += 1; continue
            at_line_start = False
            if cur_indent > 0:
                i += 1; continue
            if t[0] != TOK_IDENT:
                i += 1; continue

            # Skip host/kernel prefix
            name_idx = i
            nm = self.toks[name_idx]
            name_str = self.tok_text(nm)
            if name_str in ('host', 'kernel'):
                name_idx += 1
                if name_idx >= n:
                    break
                nm = self.toks[name_idx]
                if nm[0] != TOK_IDENT:
                    i += 1; continue

            # Walk to colon, collecting params
            j = name_idx
            params = []
            saw_colon = False
            while j < n:
                tj = self.toks[j]
                if tj[0] == TOK_NEWLINE or tj[0] == TOK_EOF:
                    break
                if tj[0] == TOK_COLON:
                    saw_colon = True
                    break
                if j > name_idx and tj[0] in (TOK_IDENT, TOK_BUF, TOK_CONST,
                                               TOK_VAR, TOK_LABEL):
                    params.append(self.tok_text(tj))
                j += 1
            if not saw_colon:
                i += 1; continue

            # Body starts after `:` + newline
            k = j + 1
            while k < n and self.toks[k][0] != TOK_NEWLINE:
                k += 1
            body_tk = k + 1

            name = self.tok_text(nm)
            # Body indent: peek first INDENT in body
            body_indent = 0
            bk = body_tk
            while bk < n:
                bt = self.toks[bk]
                if bt[0] == TOK_NEWLINE:
                    bk += 1; continue
                if bt[0] == TOK_INDENT:
                    body_indent = bt[2]
                    break
                break

            self.compositions[name] = {
                'params': params,
                'body_tk': body_tk,
                'body_indent': body_indent,
            }
            i = body_tk
    def run(self):
        self.collect()

        # Set up argv: push argv strings to memory, build argv array
        argv_ptrs = []
        for s in self.argv:
            sb = s.encode() + b'\0'
            off = self.mem_top
            self.mem[off:off + len(sb)] = sb
            self.mem_top = (self.mem_top + len(sb) + 7) & ~7
            argv_ptrs.append(off)
        argv_array_off = self.mem_top
        for idx, p in enumerate(argv_ptrs):
            struct.pack_into('<Q', self.mem, argv_array_off + idx * 8, p)
        self.mem_top += 8 * (len(argv_ptrs) + 1)
        self.regs[0] = len(self.argv)
        self.regs[1] = argv_array_off

        # First pass: execute top-level statements (const/buf/var decls)
        # by scanning from the top of the token stream at indent 0
        self.tk = 0
        while self.ty() != TOK_EOF:
            t = self.cur()
            if t[0] == TOK_NEWLINE:
                self.tk += 1; continue
            if t[0] == TOK_INDENT:
                if t[2] == 0:
                    # top-level indent 0 — skip and inspect next token
                    self.tk += 1
                    continue
                # indented content inside a composition we've already recorded
                while self.ty() not in (TOK_NEWLINE, TOK_EOF):
                    self.tk += 1
                continue
            # top-level non-indented statement
            if t[0] in (TOK_CONST, TOK_BUF, TOK_VAR):
                self.exec_stmt()
                continue
            if t[0] == TOK_IDENT:
                # Could be composition header (skip) or shorthand const
                # Look ahead: if `NAME INT NEWLINE`, it's shorthand const
                if self.tk + 2 < len(self.toks) and \
                   self.toks[self.tk+1][0] == TOK_INT and \
                   self.toks[self.tk+2][0] in (TOK_NEWLINE, TOK_EOF):
                    name = self.tok_text(t)
                    self.tk += 1
                    val = self.parse_primary()
                    self.consts[name] = val
                    continue
                # Composition header — skip to its body_tk, then skip body
                name = self.tok_text(t)
                if name in ('host', 'kernel'):
                    if self.tk + 1 < len(self.toks) and self.toks[self.tk+1][0] == TOK_IDENT:
                        name = self.tok_text(self.toks[self.tk+1])
                if name in self.compositions:
                    compo = self.compositions[name]
                    self.tk = compo['body_tk']
                    self.skip_body(compo['body_indent'])
                    continue
                # unknown top-level — skip line
                while self.ty() not in (TOK_NEWLINE, TOK_EOF):
                    self.tk += 1
                continue
            # Anything else — skip line
            while self.ty() not in (TOK_NEWLINE, TOK_EOF):
                self.tk += 1

        # Now run main
        if 'main' not in self.compositions:
            raise RuntimeError("no main composition")

        ret = 0
        try:
            ret = self.call_compo('main', atom_mode=False)
        except ExitProgram as e:
            return e.code
        except ReturnVal as r:
            return r.value & 0xFF
        return ret & 0xFF

def main():
    if len(sys.argv) < 2:
        print("usage: lithos-interp.py source.ls [args...]", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        src = f.read()

    interp = Interp(src, sys.argv[1:])
    # Run in a thread with a much larger stack to avoid overflowing on the
    # compiler's deep recursion.
    result = {'code': 0, 'error': None}
    def runner():
        try:
            result['code'] = interp.run()
        except RuntimeError as e:
            import traceback as _tb
            result['error'] = str(e) + "\n" + _tb.format_exc()
        except RecursionError as e:
            import traceback as _tb
            result['error'] = "recursion: " + str(e) + "\n" + _tb.format_exc()
        finally:
            # Dump instrumentation
            import sys as _sys
            if hasattr(interp, '_call_counts'):
                counts = interp._call_counts
                print(f"=== INSTRUMENTATION ===", file=_sys.stderr)
                print(f"max_depth={interp._max_depth}", file=_sys.stderr)
                # Globals
                g = getattr(interp, 'globals', {}) or {}
                for gk in ('arm64_pos', 'gpu_pos', 'token_count'):
                    print(f"global[{gk}]={g.get(gk, '<missing>')}", file=_sys.stderr)
                # Top-called functions
                emit_calls = {k: v for k, v in counts.items() if k.startswith('emit_')}
                parse_calls = {k: v for k, v in counts.items() if k.startswith('parse_')}
                elf_calls = {k: v for k, v in counts.items() if 'elf' in k or 'mach' in k or 'build' in k}
                print(f"emit_* calls: {sorted(emit_calls.items(), key=lambda x:-x[1])[:10]}", file=_sys.stderr)
                print(f"parse_* calls: {sorted(parse_calls.items(), key=lambda x:-x[1])[:10]}", file=_sys.stderr)
                print(f"output_writer calls: {sorted(elf_calls.items(), key=lambda x:-x[1])[:10]}", file=_sys.stderr)
                top10 = sorted(counts.items(), key=lambda x:-x[1])[:15]
                print(f"top calls: {top10}", file=_sys.stderr)
                _sys.stderr.flush()
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if result['error']:
        print(f"interp error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    sys.exit(result['code'])

if __name__ == '__main__':
    main()

