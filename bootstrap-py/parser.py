"""
parser.py -- Recursive-descent parser for Lithos .ls source files.

Takes a flat token list (from the lexer) and produces an AST.

Lithos grammar summary (indentation-based blocks, no braces):

  file       = { composition | comment_line | blank }
  composition = IDENT { IDENT } COLON NEWLINE INDENT body DEDENT
  body       = { statement NEWLINE }
  statement  = store_stmt | load_stmt | for_stmt | if_stmt | each_stmt
             | shared_decl | param_decl | label_stmt | membar_stmt
             | barrier_stmt | exit_stmt | return_stmt | endfor_stmt
             | regwrite_stmt | regread_stmt | prefix_op_stmt
             | ident_stmt | expr_stmt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Token types -- must match the Lithos lexer
# ---------------------------------------------------------------------------

class TT(IntEnum):
    EOF       = 0
    NEWLINE   = 1
    INDENT    = 2
    INT       = 3
    FLOAT     = 4
    IDENT     = 5
    STRING    = 6
    KERNEL    = 11
    PARAM     = 12
    IF        = 13
    ELSE      = 14
    ELIF      = 15
    FOR       = 16
    ENDFOR    = 17
    EACH      = 18
    STRIDE    = 19
    WHILE     = 20
    RETURN    = 21
    CONST     = 22
    VAR       = 23
    BUF       = 24
    WEIGHT    = 25
    LAYER     = 26
    BIND      = 27
    RUNTIME   = 28
    TEMPLATE  = 29
    PROJECT   = 30
    SHARED    = 31
    BARRIER   = 32
    LABEL     = 33
    EXIT      = 34
    HOST      = 35
    LOAD      = 36       # -> (right arrow, memory load)
    STORE     = 37       # <- (left arrow, memory store)
    REG_READ  = 38       # up arrow
    REG_WRITE = 39       # down arrow
    F32       = 40
    U32       = 41
    S32       = 42
    F16       = 43
    PTR       = 44
    VOID      = 45
    PLUS      = 50
    MINUS     = 51
    STAR      = 52
    SLASH     = 53
    EQ        = 54
    EQEQ      = 55
    NEQ       = 56
    LT        = 57
    GT        = 58
    LTE       = 59
    GTE       = 60
    AMP       = 61
    PIPE      = 62
    CARET     = 63
    SHL       = 64
    SHR       = 65
    LBRACK    = 67
    RBRACK    = 68
    LPAREN    = 69
    RPAREN    = 70
    COMMA     = 71
    COLON     = 72
    DOT       = 73
    AT        = 74
    SIGMA     = 75       # Sigma sum
    TRIANGLE  = 76       # triangle max
    NABLA     = 77       # nabla min
    INDEX     = 78       # hash
    SQRT      = 79
    SIN       = 80
    COS       = 81
    TRAP      = 89
    GOTO      = 93
    SYSCALL   = 94
    CONTINUE  = 95
    CONSTANT  = 96
    DOLLAR    = 97


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: int           # TT value
    text: str           # source text of the token
    line: int = 0       # 1-based line number (informational)
    col: int = 0        # 0-based column (informational)
    indent: int = 0     # for INDENT tokens: the indentation level

    @property
    def value(self) -> str:
        """Alias for text, for convenience."""
        return self.text


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class IntLit:
    value: int

@dataclass
class FloatLit:
    value: float

@dataclass
class Ident:
    name: str

@dataclass
class RegRef:
    """Register reference like $r0 or $tid."""
    name: str

@dataclass
class BinOp:
    op: str             # '+', '-', '*', '/', '&', '|', '^', '<<', '>>'
    left: object
    right: object

@dataclass
class UnaryOp:
    """Unary math intrinsic (sqrt, sin, cos, rcp, exp2, ln, etc.)."""
    op: str
    operand: object

@dataclass
class ArrayIndex:
    """array [ index ]"""
    base: object
    index: object

@dataclass
class FuncCall:
    """Composition (function) call: name arg1 arg2 ..."""
    name: str
    args: list

@dataclass
class ShflBfly:
    """shfl.bfly dest src delta"""
    dest: str
    src: object
    delta: object

@dataclass
class Fma:
    """fma acc a b c  =>  acc = a*b + c"""
    dest: str
    a: object
    b: object
    c: object


# -- Statement nodes --------------------------------------------------------

@dataclass
class FuncDef:
    """Composition definition: name arg1 arg2 ... : body"""
    name: str
    params: List[str]
    body: list              # list of statement AST nodes
    is_host: bool = False   # True if prefixed with 'host'

@dataclass
class Assignment:
    """target = expr  OR  target expr (Lithos implicit assignment)"""
    target: str
    value: object

@dataclass
class ArrayStore:
    """target [ index ] = value"""
    target: str
    index: object
    value: object

@dataclass
class Store:
    """<- width addr value  (memory store)"""
    width: object
    addr: object
    value: object

@dataclass
class Load:
    """result -> width addr  (memory load, used as expression)"""
    width: object
    addr: object

@dataclass
class RegWrite:
    """down-arrow $reg value"""
    reg: object
    value: object

@dataclass
class RegRead:
    """up-arrow $reg  (expression)"""
    reg: object

@dataclass
class For:
    """for var start end step : body"""
    var: str
    start: object
    end: object
    step: object
    body: list

@dataclass
class EndFor:
    """Explicit endfor token (flat style)."""
    pass

@dataclass
class If:
    """if== a b  /  if>= a b  /  if< a b ... : body"""
    op: str                 # '==', '>=', '<', '!=', '>', '<='
    left: object
    right: object
    body: list
    label: Optional[str] = None   # optional branch-target label (for flat if)

@dataclass
class Each:
    """each var  -- GPU thread-parallel iteration"""
    var: str

@dataclass
class Stride:
    """stride var dim"""
    var: str
    dim: object

@dataclass
class Shared:
    """shared name count type"""
    name: str
    count: object
    dtype: str

@dataclass
class Param:
    """param name type"""
    name: str
    dtype: str

@dataclass
class Label:
    """label name"""
    name: str

@dataclass
class Goto:
    """goto name"""
    name: str

@dataclass
class Barrier:
    """bar.sync"""
    pass

@dataclass
class MembarSys:
    """membar_sys"""
    pass

@dataclass
class Exit:
    """exit / return"""
    pass

@dataclass
class Trap:
    """trap sysnum arg1 arg2 ..."""
    sysnum: object
    args: list

@dataclass
class Comment:
    text: str

@dataclass
class VarDecl:
    """var name value"""
    name: str
    value: object

@dataclass
class BufDecl:
    """buf name size"""
    name: str
    size: object

@dataclass
class ConstDecl:
    """const name value"""
    name: str
    value: object

@dataclass
class PrefixOp:
    """3-operand prefix forms: add dest a b, shl dest a b, etc."""
    op: str
    dest: str
    a: object
    b: object

@dataclass
class TypeCast:
    """u32>f32 dest src"""
    from_type: str
    to_type: str
    dest: str
    src: object

@dataclass
class Reduction:
    """Sigma/Triangle/Nabla reduction."""
    kind: str       # 'sum', 'max', 'min'
    operand: object


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Prefix-style 3-operand keywords recognised at statement start
_PREFIX_OPS = frozenset({
    'and', 'or', 'xor', 'shl', 'shr', 'add', 'sub', 'mul',
})

# Type cast keywords
_TYPE_CASTS = frozenset({
    'u32>f32', 'f32>u32', 'u32>f16', 'f16>u32', 'f32>f16', 'f16>f32',
    's32>f32', 'f32>s32',
})

# Keywords that are NOT identifiers (should not be treated as variable names
# or call targets when they appear at the start of a line).
_STMT_KEYWORDS = frozenset({
    'for', 'endfor', 'if', 'each', 'stride', 'shared', 'param',
    'label', 'goto', 'exit', 'return', 'trap', 'var', 'buf', 'const',
    'membar_sys', 'bar.sync', 'continue', 'host', 'kernel',
    'shfl.bfly', 'fma',
})


class ParseError(Exception):
    def __init__(self, msg: str, token: Optional[Token] = None):
        self.token = token
        loc = ''
        if token:
            loc = f' at line {token.line} col {token.col}'
        super().__init__(f'{msg}{loc}')


class Parser:
    """Recursive-descent parser for Lithos .ls files.

    Usage::

        tokens = [...]       # list of Token from the lexer
        parser = Parser(tokens)
        ast = parser.parse()   # returns list[FuncDef | ConstDecl | ...]
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token helpers ------------------------------------------------------

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(type=TT.EOF, text='', line=0, col=0)

    def _peek_type(self) -> int:
        return self._peek().type

    def _advance(self) -> Token:
        tok = self._peek()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def _expect(self, tt: int, text: Optional[str] = None) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise ParseError(
                f'expected token type {tt}, got {tok.type} ({tok.text!r})', tok)
        if text is not None and tok.text != text:
            raise ParseError(
                f'expected {text!r}, got {tok.text!r}', tok)
        return self._advance()

    def _match(self, tt: int, text: Optional[str] = None) -> Optional[Token]:
        tok = self._peek()
        if tok.type != tt:
            return None
        if text is not None and tok.text != text:
            return None
        return self._advance()

    def _at_end(self) -> bool:
        return self._peek_type() == TT.EOF

    # -- whitespace / newline helpers ---------------------------------------

    def _skip_newlines(self):
        """Skip NEWLINE and INDENT tokens (blank lines)."""
        while self._peek_type() in (TT.NEWLINE, TT.INDENT):
            self._advance()

    def _skip_to_eol(self):
        """Advance past everything until NEWLINE or EOF."""
        while self._peek_type() not in (TT.NEWLINE, TT.EOF):
            self._advance()

    def _current_indent(self) -> int:
        """Return the indent level of the most-recently-seen INDENT token,
        or 0 if none."""
        # Walk backwards from pos to find the last INDENT on this line.
        i = self.pos - 1
        while i >= 0:
            if self.tokens[i].type == TT.INDENT:
                return self.tokens[i].indent
            if self.tokens[i].type == TT.NEWLINE:
                return 0
            i -= 1
        return 0

    # -- indentation-based block parsing ------------------------------------

    def _parse_block(self, parent_indent: int) -> list:
        """Parse an indented block of statements.

        `parent_indent` is the indentation of the line containing the colon
        that opens this block.  The body must be indented deeper.
        """
        stmts: list = []

        # Consume the NEWLINE after the colon (if any)
        while self._peek_type() == TT.NEWLINE:
            self._advance()

        # Determine the body indentation from the first INDENT token
        if self._peek_type() == TT.INDENT:
            body_indent = self._peek().indent
            if body_indent <= parent_indent:
                return stmts           # empty body
        else:
            # No indent token -> code at column 0 -> not a block body
            return stmts

        while not self._at_end():
            # At the start of a line we expect INDENT or NEWLINE
            tok = self._peek()

            if tok.type == TT.NEWLINE:
                self._advance()
                continue

            if tok.type == TT.INDENT:
                if tok.indent < body_indent:
                    # Check if this is just a blank/comment line (indent 0
                    # followed by NEWLINE).  If so, skip it rather than
                    # treating it as a dedent — the real body may continue
                    # after intervening blank/comment lines.
                    saved = self.pos
                    self._advance()  # consume INDENT
                    if self._peek_type() == TT.NEWLINE:
                        # Blank/comment line — skip and keep parsing the block
                        continue
                    # Real dedent — restore position and break
                    self.pos = saved
                    break
                # Consume the INDENT token
                self._advance()
                # If followed by NEWLINE, it was a blank/comment line -- skip
                if self._peek_type() == TT.NEWLINE:
                    continue
                s = self._parse_statement()
                if s is not None:
                    stmts.append(s)
                continue

            if tok.type == TT.EOF:
                break

            # Non-INDENT, non-NEWLINE token at start of iteration means
            # we're at a line with no indentation (column 0), which is
            # always a dedent from any body. Break out.
            break

        return stmts

    # -- top-level ----------------------------------------------------------

    def parse(self) -> list:
        """Parse an entire .ls file.  Returns a list of top-level AST nodes
        (mostly FuncDef, plus possible ConstDecl / VarDecl / BufDecl)."""
        top: list = []
        while not self._at_end():
            self._skip_newlines()
            if self._at_end():
                break

            tok = self._peek()

            # Comment lines are already stripped by the lexer; if we see
            # a stray INDENT at top level, skip it.
            if tok.type == TT.INDENT:
                self._advance()
                continue

            # host prefix
            is_host = False
            if tok.type == TT.HOST or (tok.type == TT.IDENT and tok.text == 'host'):
                self._advance()
                is_host = True
                self._skip_newlines()
                tok = self._peek()

            # kernel prefix (optional, ignore)
            if tok.type == TT.KERNEL or (tok.type == TT.IDENT and tok.text == 'kernel'):
                self._advance()
                self._skip_newlines()
                tok = self._peek()

            # Top-level declarations: var, buf, const
            if tok.type == TT.VAR or (tok.type == TT.IDENT and tok.text == 'var'):
                top.append(self._parse_var_decl())
                continue
            if tok.type == TT.BUF or (tok.type == TT.IDENT and tok.text == 'buf'):
                top.append(self._parse_buf_decl())
                continue
            if tok.type == TT.CONST or (tok.type == TT.IDENT and tok.text == 'const'):
                top.append(self._parse_const_decl())
                continue
            if tok.type == TT.CONSTANT or (tok.type == TT.IDENT and tok.text == 'constant'):
                top.append(self._parse_const_decl())
                continue

            # Composition (function) definition
            if tok.type == TT.IDENT:
                node = self._parse_composition(is_host)
                top.append(node)
                continue

            # Skip unknown top-level tokens
            self._advance()

        return top

    # -- composition (function definition) ----------------------------------

    def _parse_composition(self, is_host: bool = False) -> FuncDef:
        indent = self._current_indent()
        name_tok = self._expect(TT.IDENT)
        name = name_tok.text

        # Parse parameter names until we see ':'
        params: list[str] = []
        while self._peek_type() != TT.COLON and self._peek_type() != TT.EOF:
            if self._peek_type() == TT.NEWLINE:
                # colon might be missing; abort
                break
            ptok = self._advance()
            if ptok.type == TT.IDENT:
                params.append(ptok.text)
            elif ptok.type == TT.INDENT:
                continue  # skip stray indent
            else:
                # Non-ident in param list -- might be an operator token used
                # as a name, just record its text
                params.append(ptok.text)

        self._expect(TT.COLON)
        body = self._parse_block(indent)
        return FuncDef(name=name, params=params, body=body, is_host=is_host)

    # -- statement dispatch -------------------------------------------------

    def _parse_statement(self) -> object:
        """Parse a single statement and return an AST node."""
        tok = self._peek()
        tt = tok.type
        text = tok.text

        # ---- Memory store: <- width addr value ----------------------------
        if tt == TT.STORE:
            return self._parse_store_stmt()

        # ---- Memory load as statement: result -> width addr ---------------
        # (This is an assignment form; handled in ident_stmt via lookahead)

        # ---- Register write: down-arrow $reg value ------------------------
        if tt == TT.REG_WRITE:
            return self._parse_regwrite()

        # ---- for loop -----------------------------------------------------
        if tt == TT.FOR or (tt == TT.IDENT and text == 'for'):
            return self._parse_for()

        # ---- endfor (explicit, flat style) --------------------------------
        if tt == TT.ENDFOR or (tt == TT.IDENT and text == 'endfor'):
            self._advance()
            return EndFor()

        # ---- each ---------------------------------------------------------
        if tt == TT.EACH or (tt == TT.IDENT and text == 'each'):
            return self._parse_each()

        # ---- stride -------------------------------------------------------
        if tt == TT.STRIDE or (tt == TT.IDENT and text == 'stride'):
            return self._parse_stride()

        # ---- conditionals: if== if>= if< etc. ----------------------------
        if tt == TT.IF or (tt == TT.IDENT and text.startswith('if')):
            return self._parse_if()

        # ---- shared -------------------------------------------------------
        if tt == TT.SHARED or (tt == TT.IDENT and text == 'shared'):
            return self._parse_shared()

        # ---- param --------------------------------------------------------
        if tt == TT.PARAM or (tt == TT.IDENT and text == 'param'):
            return self._parse_param()

        # ---- label --------------------------------------------------------
        if tt == TT.LABEL or (tt == TT.IDENT and text == 'label'):
            return self._parse_label()

        # ---- goto ---------------------------------------------------------
        if tt == TT.GOTO or (tt == TT.IDENT and text == 'goto'):
            return self._parse_goto()

        # ---- barrier / bar.sync -------------------------------------------
        if tt == TT.BARRIER or (tt == TT.IDENT and text in ('bar.sync', 'barrier')):
            self._advance()
            return Barrier()

        # ---- membar_sys ---------------------------------------------------
        if tt == TT.IDENT and text == 'membar_sys':
            self._advance()
            return MembarSys()

        # ---- exit / return ------------------------------------------------
        if tt == TT.EXIT or (tt == TT.IDENT and text == 'exit'):
            self._advance()
            return Exit()
        if tt == TT.RETURN or (tt == TT.IDENT and text == 'return'):
            self._advance()
            return Exit()

        # ---- continue -----------------------------------------------------
        if tt == TT.CONTINUE or (tt == TT.IDENT and text == 'continue'):
            self._advance()
            return Goto(name='__continue__')

        # ---- var / buf / const at local scope -----------------------------
        if tt == TT.VAR or (tt == TT.IDENT and text == 'var'):
            return self._parse_var_decl()
        if tt == TT.BUF or (tt == TT.IDENT and text == 'buf'):
            return self._parse_buf_decl()
        if tt == TT.CONST or (tt == TT.IDENT and text == 'const'):
            return self._parse_const_decl()
        if tt == TT.CONSTANT or (tt == TT.IDENT and text == 'constant'):
            return self._parse_const_decl()

        # ---- trap ---------------------------------------------------------
        if tt == TT.TRAP or (tt == TT.IDENT and text == 'trap'):
            return self._parse_trap()

        # ---- syscall (same as trap) ---------------------------------------
        if tt == TT.SYSCALL or (tt == TT.IDENT and text == 'syscall'):
            return self._parse_trap()

        # ---- identifier at statement start --------------------------------
        if tt == TT.IDENT:
            return self._parse_ident_stmt()

        # ---- register read at statement start (unlikely but valid) --------
        if tt == TT.REG_READ:
            return self._parse_expr()

        # ---- dollar-prefixed call: $NAME arg1 arg2 ... -------------------
        if tt == TT.DOLLAR:
            return self._parse_dollar_call()

        # ---- expression statement (fallback) ------------------------------
        expr = self._parse_expr()
        return expr

    # -- specific statement parsers -----------------------------------------

    def _parse_store_stmt(self):
        """<- width addr value"""
        self._advance()  # consume STORE token
        width = self._parse_expr()
        addr = self._parse_expr()
        value = self._parse_expr()
        return Store(width=width, addr=addr, value=value)

    def _parse_regwrite(self):
        """down-arrow $reg value"""
        self._advance()  # consume REG_WRITE
        reg = self._parse_reg_name()
        value = self._parse_expr()
        return RegWrite(reg=reg, value=value)

    def _parse_reg_name(self):
        """Parse a register reference: $N or $name or just an ident."""
        if self._match(TT.DOLLAR):
            tok = self._advance()
            return RegRef(name='$' + tok.text)
        tok = self._advance()
        return RegRef(name=tok.text)

    def _parse_for(self):
        """for var start end step"""
        indent = self._current_indent()
        self._advance()  # consume 'for'
        var_tok = self._expect(TT.IDENT)
        start = self._parse_expr()
        end = self._parse_expr()
        step = self._parse_expr()
        # Body: either indented block or flat (until endfor)
        body = self._parse_block(indent)
        if not body:
            # Flat style: collect statements until 'endfor'
            body = self._parse_flat_for_body()
        return For(var=var_tok.text, start=start, end=end, step=step, body=body)

    def _parse_flat_for_body(self) -> list:
        """Parse statements until we see 'endfor' (flat / legacy style)."""
        stmts = []
        while not self._at_end():
            self._skip_newlines()
            tok = self._peek()
            if tok.type == TT.INDENT:
                self._advance()
                continue
            if tok.type == TT.ENDFOR or (tok.type == TT.IDENT and tok.text == 'endfor'):
                self._advance()
                break
            s = self._parse_statement()
            if s is not None:
                stmts.append(s)
        return stmts

    def _parse_each(self):
        """each var"""
        self._advance()  # consume 'each'
        var_tok = self._expect(TT.IDENT)
        return Each(var=var_tok.text)

    def _parse_stride(self):
        """stride var dim"""
        self._advance()  # consume 'stride'
        var_tok = self._expect(TT.IDENT)
        dim = self._parse_expr()
        return Stride(var=var_tok.text, dim=dim)

    def _parse_if(self):
        """if== a b [label] / if>= a b [label] ... : body"""
        indent = self._current_indent()
        tok = self._advance()
        text = tok.text

        # Determine comparison operator from the keyword text
        # Forms: if==, if>=, if<, if!=, if>, if<=
        op = '=='
        if text.startswith('if'):
            rest = text[2:]
            if rest in ('==', '>=', '<', '!=', '>', '<='):
                op = rest
            elif rest == '':
                # Bare 'if' -- the comparison op may be the next token
                # Check for ==, >=, <, etc.
                nxt = self._peek()
                if nxt.type == TT.EQEQ:
                    self._advance()
                    op = '=='
                elif nxt.type == TT.GTE:
                    self._advance()
                    op = '>='
                elif nxt.type == TT.LT:
                    self._advance()
                    op = '<'
                elif nxt.type == TT.GT:
                    self._advance()
                    op = '>'
                elif nxt.type == TT.LTE:
                    self._advance()
                    op = '<='
                elif nxt.type == TT.NEQ:
                    self._advance()
                    op = '!='

        left = self._parse_expr()
        right = self._parse_expr()

        # Optional label target (for flat-style: if>= a b skip_label)
        # The label can be an IDENT or a keyword like 'exit'.
        label = None
        nxt = self._peek()
        if nxt.type in (TT.IDENT, TT.EXIT, TT.RETURN) or \
           (nxt.type == TT.IDENT and nxt.text not in _STMT_KEYWORDS):
            # Check if there's a NEWLINE soon -- if the token is followed by
            # NEWLINE, it's a branch-target label (flat if form).
            saved = self.pos
            self._advance()
            after = self._peek()
            if after.type in (TT.NEWLINE, TT.EOF, TT.INDENT):
                label = nxt.text
            else:
                self.pos = saved  # put it back, it's part of body

        body = self._parse_block(indent)
        return If(op=op, left=left, right=right, body=body, label=label)

    def _parse_shared(self):
        """shared name count type"""
        self._advance()  # consume 'shared'
        name_tok = self._expect(TT.IDENT)
        count = self._parse_expr()
        # Type token (f32, u32, f16, etc.)
        dtype = 'f32'
        tok = self._peek()
        if tok.type == TT.IDENT or tok.type in (TT.F32, TT.U32, TT.S32, TT.F16):
            dtype = self._advance().text
        return Shared(name=name_tok.text, count=count, dtype=dtype)

    def _parse_param(self):
        """param name type"""
        self._advance()  # consume 'param'
        name_tok = self._expect(TT.IDENT)
        dtype = 'u32'
        tok = self._peek()
        if tok.type == TT.IDENT or tok.type in (TT.F32, TT.U32, TT.S32, TT.F16, TT.PTR):
            dtype = self._advance().text
        return Param(name=name_tok.text, dtype=dtype)

    def _parse_label(self):
        """label name"""
        self._advance()  # consume 'label'
        name_tok = self._expect(TT.IDENT)
        return Label(name=name_tok.text)

    def _parse_goto(self):
        """goto name"""
        self._advance()  # consume 'goto'
        name_tok = self._expect(TT.IDENT)
        return Goto(name=name_tok.text)

    def _parse_var_decl(self):
        """var name value"""
        self._advance()  # consume 'var'
        name_tok = self._expect(TT.IDENT)
        val = self._parse_expr()
        return VarDecl(name=name_tok.text, value=val)

    def _parse_buf_decl(self):
        """buf name size"""
        self._advance()  # consume 'buf'
        name_tok = self._expect(TT.IDENT)
        size = self._parse_expr()
        return BufDecl(name=name_tok.text, size=size)

    def _parse_const_decl(self):
        """const name value  OR  constant name"""
        self._advance()  # consume 'const' / 'constant'
        name_tok = self._expect(TT.IDENT)
        # Value is optional for 'constant' (Forth-style may have value before keyword)
        nxt = self._peek()
        if nxt.type in (TT.INT, TT.FLOAT, TT.IDENT, TT.MINUS):
            val = self._parse_expr()
        else:
            val = IntLit(value=0)
        return ConstDecl(name=name_tok.text, value=val)

    def _parse_dollar_call(self):
        """$NAME arg1 arg2 ... — call via dollar-prefixed variable."""
        self._advance()  # consume '$'
        name_tok = self._advance()  # consume the name
        name = '$' + name_tok.text
        args = []
        while self._peek_type() not in (TT.NEWLINE, TT.EOF, TT.INDENT) and \
              self._is_expr_start(self._peek()):
            args.append(self._parse_expr())
        return FuncCall(name=name, args=args)

    def _parse_trap(self):
        """trap [sysnum arg1 arg2 ...]  -- bare trap (no args) is valid."""
        self._advance()  # consume 'trap' / 'syscall'
        # Bare trap: no arguments means SVC #0 with no register setup
        if self._peek_type() in (TT.NEWLINE, TT.EOF, TT.INDENT):
            return Trap(sysnum=None, args=[])
        sysnum = self._parse_expr()
        args = []
        while self._peek_type() not in (TT.NEWLINE, TT.EOF, TT.INDENT):
            args.append(self._parse_expr())
        return Trap(sysnum=sysnum, args=args)

    # -- identifier at statement start --------------------------------------

    def _parse_ident_stmt(self):
        """Handle an identifier at the start of a statement.

        Could be:
        - Prefix 3-operand op: ``and dest a b``, ``shl dest a b``
        - Type cast: ``u32>f32 dest src``
        - shfl.bfly: ``shfl.bfly dest src delta``
        - fma: ``fma dest a b c``
        - Function call / composition: ``funcname arg1 arg2``
        - Assignment: ``name expr`` (implicit) or ``name = expr`` (explicit)
        - Assignment via memory load: ``name -> width addr``
        - Array store: ``name [ idx ] = value``
        - Conditional: ``if== a b`` (ident starting with 'if')
        """
        tok = self._peek()
        text = tok.text

        # Conditional keywords embedded in identifiers: if==, if>=, if<
        if text.startswith('if') and len(text) > 2:
            return self._parse_if()

        # membar_sys
        if text == 'membar_sys':
            self._advance()
            return MembarSys()

        # bar.sync
        if text == 'bar.sync':
            self._advance()
            return Barrier()

        # Prefix 3-operand ops: and dest a b
        if text in _PREFIX_OPS:
            return self._parse_prefix_op()

        # Type casts: u32>f32 dest src
        if text in _TYPE_CASTS:
            return self._parse_type_cast()

        # shfl.bfly dest src delta
        if text == 'shfl.bfly':
            return self._parse_shfl_bfly()

        # fma dest a b c
        if text == 'fma':
            return self._parse_fma()

        # Consume the identifier
        self._advance()
        name = text

        nxt = self._peek()

        # Explicit assignment: name = expr
        if nxt.type == TT.EQ:
            self._advance()
            val = self._parse_expr()
            return Assignment(target=name, value=val)

        # Array store: name [ idx ] = value
        if nxt.type == TT.LBRACK:
            self._advance()
            idx = self._parse_expr()
            self._expect(TT.RBRACK)
            self._expect(TT.EQ)
            val = self._parse_expr()
            return ArrayStore(target=name, index=idx, value=val)

        # Check if this looks like a composition header (name args : body)
        # at statement level this shouldn't happen, but guard against it.

        # Memory load form: name -> width addr  (result = load)
        if nxt.type == TT.LOAD:
            self._advance()  # consume ->
            width = self._parse_expr()
            addr = self._parse_expr()
            return Assignment(target=name, value=Load(width=width, addr=addr))

        # If next token starts an expression, it could be:
        #   name expr              -> implicit assignment (name = expr)
        #   name expr expr ...     -> function call (name(expr, expr, ...))
        # Parse the first expression, then check if more follow on the line.
        if self._is_expr_start(nxt):
            first = self._parse_expr()
            # Check if more expression-start tokens follow on the same line
            nxt2 = self._peek()
            if self._is_expr_start(nxt2):
                # Multi-argument: this is a function call
                args = [first]
                while self._peek_type() not in (TT.NEWLINE, TT.EOF, TT.INDENT) and \
                      self._is_expr_start(self._peek()):
                    args.append(self._parse_expr())
                return FuncCall(name=name, args=args)
            return Assignment(target=name, value=first)

        # Bare identifier on a line -- could be a nullary function call
        return FuncCall(name=name, args=[])

    def _parse_prefix_op(self):
        """Parse 3-operand prefix form: op dest a b"""
        tok = self._advance()
        op = tok.text
        dest_tok = self._expect(TT.IDENT)
        a = self._parse_expr()
        b = self._parse_expr()
        return PrefixOp(op=op, dest=dest_tok.text, a=a, b=b)

    def _parse_type_cast(self):
        """Parse type cast: u32>f32 dest src"""
        tok = self._advance()
        parts = tok.text.split('>')
        from_type = parts[0]
        to_type = parts[1] if len(parts) > 1 else ''
        dest_tok = self._expect(TT.IDENT)
        src = self._parse_expr()
        return TypeCast(from_type=from_type, to_type=to_type,
                        dest=dest_tok.text, src=src)

    def _parse_shfl_bfly(self):
        """shfl.bfly dest src delta"""
        self._advance()  # consume 'shfl.bfly'
        dest_tok = self._expect(TT.IDENT)
        src = self._parse_expr()
        delta = self._parse_expr()
        return ShflBfly(dest=dest_tok.text, src=src, delta=delta)

    def _parse_fma(self):
        """fma dest a b c"""
        self._advance()  # consume 'fma'
        dest_tok = self._expect(TT.IDENT)
        a = self._parse_expr()
        b = self._parse_expr()
        c = self._parse_expr()
        return Fma(dest=dest_tok.text, a=a, b=b, c=c)

    # -- expression parser (recursive descent) ------------------------------

    def _is_expr_start(self, tok: Token) -> bool:
        """Return True if `tok` can start an expression."""
        return tok.type in (
            TT.INT, TT.FLOAT, TT.IDENT, TT.LPAREN,
            TT.LOAD, TT.REG_READ, TT.MINUS,
            TT.SQRT, TT.SIN, TT.COS,
            TT.SIGMA, TT.TRIANGLE, TT.NABLA, TT.INDEX,
            TT.DOLLAR,
        )

    def _parse_expr(self):
        """Parse an expression.  Precedence (low to high):
           additive:  + -
           multiplicative:  * /
           atom
        """
        return self._parse_add_expr()

    def _parse_add_expr(self):
        left = self._parse_mul_expr()
        while True:
            tok = self._peek()
            if tok.type == TT.PLUS:
                self._advance()
                right = self._parse_mul_expr()
                left = BinOp(op='+', left=left, right=right)
            elif tok.type == TT.MINUS:
                # Disambiguate: is this subtraction or a negative literal on
                # the next line?  Only treat as minus if on the same line.
                self._advance()
                right = self._parse_mul_expr()
                left = BinOp(op='-', left=left, right=right)
            else:
                break
        return left

    def _parse_mul_expr(self):
        left = self._parse_atom()
        while True:
            tok = self._peek()
            if tok.type == TT.STAR:
                self._advance()
                right = self._parse_atom()
                left = BinOp(op='*', left=left, right=right)
            elif tok.type == TT.SLASH:
                self._advance()
                right = self._parse_atom()
                left = BinOp(op='/', left=left, right=right)
            else:
                break
        return left

    def _parse_atom(self):
        tok = self._peek()

        # Integer literal
        if tok.type == TT.INT:
            self._advance()
            return IntLit(value=_parse_int(tok.text))

        # Float literal
        if tok.type == TT.FLOAT:
            self._advance()
            return FloatLit(value=float(tok.text))

        # Negative number: MINUS followed by number
        if tok.type == TT.MINUS:
            nxt_idx = self.pos + 1
            if nxt_idx < len(self.tokens):
                nxt = self.tokens[nxt_idx]
                if nxt.type == TT.INT:
                    self._advance()  # minus
                    self._advance()  # number
                    return IntLit(value=-_parse_int(nxt.text))
                if nxt.type == TT.FLOAT:
                    self._advance()
                    self._advance()
                    return FloatLit(value=-float(nxt.text))

        # Memory load expression: -> width addr
        if tok.type == TT.LOAD:
            self._advance()
            width = self._parse_expr()
            addr = self._parse_expr()
            return Load(width=width, addr=addr)

        # Register read: up-arrow [$reg | ident]
        if tok.type == TT.REG_READ:
            self._advance()
            reg = self._parse_reg_name()
            return RegRead(reg=reg)

        # sqrt
        if tok.type == TT.SQRT:
            self._advance()
            operand = self._parse_expr()
            return UnaryOp(op='sqrt', operand=operand)

        # sin
        if tok.type == TT.SIN:
            self._advance()
            operand = self._parse_expr()
            return UnaryOp(op='sin', operand=operand)

        # cos
        if tok.type == TT.COS:
            self._advance()
            operand = self._parse_expr()
            return UnaryOp(op='cos', operand=operand)

        # Sigma reduction
        if tok.type == TT.SIGMA:
            self._advance()
            operand = self._parse_expr()
            return Reduction(kind='sum', operand=operand)

        # Triangle reduction (max)
        if tok.type == TT.TRIANGLE:
            self._advance()
            operand = self._parse_expr()
            return Reduction(kind='max', operand=operand)

        # Nabla reduction (min)
        if tok.type == TT.NABLA:
            self._advance()
            operand = self._parse_expr()
            return Reduction(kind='min', operand=operand)

        # Dollar register ref: $N or $name
        if tok.type == TT.DOLLAR:
            self._advance()
            ntok = self._advance()
            return RegRef(name='$' + ntok.text)

        # Parenthesized expression
        if tok.type == TT.LPAREN:
            self._advance()
            val = self._parse_expr()
            self._expect(TT.RPAREN)
            return val

        # Identifier: variable reference, possibly with array indexing
        if tok.type == TT.IDENT:
            self._advance()
            name = tok.text

            # Array indexing: name [ expr ]
            if self._peek_type() == TT.LBRACK:
                self._advance()
                idx = self._parse_expr()
                self._expect(TT.RBRACK)
                return ArrayIndex(base=Ident(name=name), index=idx)

            return Ident(name=name)

        # Types used as identifiers (f32, u32, etc.)
        if tok.type in (TT.F32, TT.U32, TT.S32, TT.F16, TT.PTR):
            self._advance()
            return Ident(name=tok.text)

        # If nothing matches, raise an error
        raise ParseError(f'unexpected token in expression: {tok.text!r} (type={tok.type})', tok)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int(text: str) -> int:
    """Parse an integer literal (decimal or 0x hex)."""
    text = text.strip()
    if text.startswith('0x') or text.startswith('0X'):
        return int(text, 16)
    if text.startswith('-0x') or text.startswith('-0X'):
        return -int(text[1:], 16)
    return int(text)


# ---------------------------------------------------------------------------
# Convenience: pretty-print an AST
# ---------------------------------------------------------------------------

def dump_ast(nodes, indent=0):
    """Print an AST (list of nodes) in a human-readable form."""
    prefix = '  ' * indent
    for node in nodes:
        if isinstance(node, FuncDef):
            host_tag = ' [host]' if node.is_host else ''
            print(f'{prefix}FuncDef {node.name}({", ".join(node.params)}){host_tag}:')
            dump_ast(node.body, indent + 1)
        elif isinstance(node, For):
            print(f'{prefix}For {node.var} = {node.start} .. {node.end} step {node.step}:')
            dump_ast(node.body, indent + 1)
        elif isinstance(node, If):
            lbl = f' -> {node.label}' if node.label else ''
            print(f'{prefix}If {node.op} {node.left} {node.right}{lbl}:')
            dump_ast(node.body, indent + 1)
        elif isinstance(node, Assignment):
            print(f'{prefix}{node.target} = {node.value}')
        elif isinstance(node, Store):
            print(f'{prefix}Store({node.width}) [{node.addr}] = {node.value}')
        elif isinstance(node, list):
            dump_ast(node, indent)
        else:
            print(f'{prefix}{node}')


# ---------------------------------------------------------------------------
# Self-test: if run directly, parse files given as arguments
# ---------------------------------------------------------------------------

def trivial_tokenize(source: str) -> List[Token]:
        """A minimal tokenizer sufficient to exercise the parser on .ls files.

        This is NOT the real lexer -- just enough to bootstrap testing.
        """
        tokens: List[Token] = []
        lines = source.split('\n')

        # Unicode arrow map
        _UNICODE_MAP = {
            '\u2192': (TT.LOAD,      '\u2192'),   # ->
            '\u2190': (TT.STORE,     '\u2190'),    # <-
            '\u2191': (TT.REG_READ,  '\u2191'),    # up
            '\u2193': (TT.REG_WRITE, '\u2193'),    # down
        }

        # Keyword map
        _KW = {
            'for':       TT.FOR,
            'endfor':    TT.ENDFOR,
            'each':      TT.EACH,
            'stride':    TT.STRIDE,
            'if':        TT.IF,
            'else':      TT.ELSE,
            'elif':      TT.ELIF,
            'while':     TT.WHILE,
            'return':    TT.RETURN,
            'const':     TT.CONST,
            'var':       TT.VAR,
            'buf':       TT.BUF,
            'shared':    TT.SHARED,
            'barrier':   TT.BARRIER,
            'label':     TT.LABEL,
            'exit':      TT.EXIT,
            'host':      TT.HOST,
            'kernel':    TT.KERNEL,
            'param':     TT.PARAM,
            'trap':      TT.TRAP,
            'goto':      TT.GOTO,
            'constant':  TT.CONSTANT,
            'continue':  TT.CONTINUE,
        }

        _OPS = {
            '+':  TT.PLUS,
            '-':  TT.MINUS,
            '*':  TT.STAR,
            '/':  TT.SLASH,
            '=':  TT.EQ,
            '<':  TT.LT,
            '>':  TT.GT,
            '[':  TT.LBRACK,
            ']':  TT.RBRACK,
            '(':  TT.LPAREN,
            ')':  TT.RPAREN,
            ',':  TT.COMMA,
            ':':  TT.COLON,
            '.':  TT.DOT,
            '@':  TT.AT,
            '&':  TT.AMP,
            '|':  TT.PIPE,
            '^':  TT.CARET,
            '$':  TT.DOLLAR,
            '#':  TT.INDEX,
        }

        _DIGRAPHS = {
            '==': TT.EQEQ,
            '!=': TT.NEQ,
            '<=': TT.LTE,
            '>=': TT.GTE,
            '<<': TT.SHL,
            '>>': TT.SHR,
        }

        for line_no, line in enumerate(lines, 1):
            # Emit NEWLINE for previous line (except first)
            if line_no > 1:
                tokens.append(Token(type=TT.NEWLINE, text='\n',
                                    line=line_no - 1, col=0))

            # Measure indentation
            stripped = line.lstrip(' \t')
            indent_level = len(line) - len(stripped)
            if indent_level > 0:
                tokens.append(Token(type=TT.INDENT, text=' ' * indent_level,
                                    line=line_no, col=0, indent=indent_level))

            # Skip comment lines
            if stripped.startswith('\\\\') or stripped.startswith('\\'):
                # Check for double backslash (Lithos comment)
                rest = stripped
                if rest.startswith('\\\\'):
                    continue
                # Single backslash that's part of a comment
                if rest.startswith('\\ ') or rest == '\\':
                    continue

            col = indent_level
            text = line
            i = col

            while i < len(text):
                ch = text[i]

                # Skip whitespace
                if ch in (' ', '\t'):
                    i += 1
                    continue

                # Comment: \\ to end of line
                if ch == '\\' and i + 1 < len(text) and text[i + 1] == '\\':
                    break

                # Check for unicode arrows
                if ch in _UNICODE_MAP:
                    tt_val, txt = _UNICODE_MAP[ch]
                    tokens.append(Token(type=tt_val, text=txt,
                                        line=line_no, col=i))
                    i += 1
                    continue

                # Check for digraphs
                if i + 1 < len(text):
                    pair = text[i:i+2]
                    if pair in _DIGRAPHS:
                        tokens.append(Token(type=_DIGRAPHS[pair], text=pair,
                                            line=line_no, col=i))
                        i += 2
                        continue

                # Numbers (including negative numbers handled separately)
                if ch.isdigit() or (ch == '-' and i + 1 < len(text) and text[i+1].isdigit()):
                    start = i
                    if ch == '-':
                        i += 1
                    # Hex prefix
                    if i + 1 < len(text) and text[i] == '0' and text[i+1] in 'xX':
                        i += 2
                        while i < len(text) and text[i] in '0123456789abcdefABCDEF':
                            i += 1
                        tokens.append(Token(type=TT.INT, text=text[start:i],
                                            line=line_no, col=start))
                        continue
                    is_float = False
                    while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                        if text[i] == '.':
                            is_float = True
                        i += 1
                    tt_val = TT.FLOAT if is_float else TT.INT
                    tokens.append(Token(type=tt_val, text=text[start:i],
                                        line=line_no, col=start))
                    continue

                # Single-char operators
                if ch in _OPS:
                    tokens.append(Token(type=_OPS[ch], text=ch,
                                        line=line_no, col=i))
                    i += 1
                    continue

                # Identifiers / keywords (including things like if==, u32>f32)
                if ch.isalpha() or ch == '_':
                    start = i
                    i += 1
                    while i < len(text) and (text[i].isalnum() or text[i] in '_.<>!='):
                        # Allow dots for bar.sync, shfl.bfly
                        # Allow < > = ! for if==, if>=, u32>f32, etc.
                        if text[i] in '<>!=':
                            rest = text[start:i]
                            if rest.startswith('if'):
                                i += 1
                                continue
                            # Type cast: u32>f32, f32>u32, etc.
                            if text[i] == '>' and rest in (
                                'u32', 'f32', 's32', 'f16', 'u16'):
                                i += 1
                                continue
                            # Already past a '>' in a type cast
                            if '>' in rest:
                                i += 1
                                continue
                            break
                        if text[i] == '.':
                            rest = text[start:i]
                            if rest in ('bar', 'shfl'):
                                i += 1
                                continue
                            break
                        i += 1
                    word = text[start:i]
                    if word in _KW:
                        tokens.append(Token(type=_KW[word], text=word,
                                            line=line_no, col=start))
                    else:
                        tokens.append(Token(type=TT.IDENT, text=word,
                                            line=line_no, col=start))
                    continue

                # Unknown char -- skip
                i += 1

        tokens.append(Token(type=TT.EOF, text='', line=len(lines), col=0))
        return tokens



# ---------------------------------------------------------------------------
# Self-test: if run directly, parse files given as arguments
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python parser.py <file.ls> [...]')
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f'\n===== {path} =====')
        with open(path, 'r') as f:
            source = f.read()
        tokens = trivial_tokenize(source)
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            dump_ast(ast)
        except ParseError as e:
            print(f'Parse error: {e}')
