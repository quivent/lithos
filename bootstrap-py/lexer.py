"""
Lithos lexer — Python bootstrap implementation.

Tokenizes .ls source files into a flat list of Token objects.
Faithful port of compiler/lithos-lexer.ls and bootstrap/lithos-lexer.s.

Token encoding mirrors the assembly: each token carries a type ID (matching
the TOK_* constants), the source text value, and line/column info.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


# ============================================================
# Token type constants  (from bootstrap/lithos-table.s)
# ============================================================

class Tok(IntEnum):
    # Structural
    EOF        = 0
    NEWLINE    = 1
    INDENT     = 2

    # Literals
    INT        = 3
    FLOAT      = 4
    IDENT      = 5
    STRING     = 6

    # Keywords
    KERNEL     = 11
    PARAM      = 12
    IF         = 13
    ELSE       = 14
    ELIF       = 15
    FOR        = 16
    ENDFOR     = 17
    EACH       = 18
    STRIDE     = 19
    WHILE      = 20
    RETURN     = 21
    CONST      = 22
    VAR        = 23
    BUF        = 24
    WEIGHT     = 25
    LAYER      = 26
    BIND       = 27
    RUNTIME    = 28
    TEMPLATE   = 29
    PROJECT    = 30
    SHARED     = 31
    BARRIER    = 32
    LABEL      = 33
    EXIT       = 34
    HOST       = 35

    # Memory / register arrows (UTF-8 multi-byte)
    LOAD       = 36   # →
    STORE      = 37   # ←
    REG_READ   = 38   # ↑
    REG_WRITE  = 39   # ↓

    # Type keywords
    F32        = 40
    U32        = 41
    S32        = 42
    F16        = 43
    PTR        = 44
    VOID       = 45

    # Operators
    PLUS       = 50   # +
    MINUS      = 51   # -
    STAR       = 52   # *
    SLASH      = 53   # /
    EQ         = 54   # =
    EQEQ       = 55   # ==
    NEQ        = 56   # !=
    LT         = 57   # <
    GT         = 58   # >
    LTE        = 59   # <=
    GTE        = 60   # >=
    AMP        = 61   # &
    PIPE       = 62   # |
    CARET      = 63   # ^
    SHL        = 64   # <<
    SHR        = 65   # >>

    # Brackets / punctuation
    LBRACK     = 67   # [
    RBRACK     = 68   # ]
    LPAREN     = 69   # (
    RPAREN     = 70   # )
    COMMA      = 71   # ,
    COLON      = 72   # :
    DOT        = 73   # .
    AT         = 74   # @

    # Reduction / math unicode
    SUM        = 75   # Σ
    MAX        = 76   # △
    MIN        = 77   # ▽
    INDEX      = 78   # #
    SQRT       = 79   # √
    SIN        = 80   # ≅
    COS        = 81   # ≡

    # Misc
    TRAP       = 89   # trap
    GOTO       = 93   # goto
    CONTINUE   = 95   # continue
    CONSTANT   = 96   # constant
    DOLLAR     = 97   # $


# ============================================================
# Token
# ============================================================

class Token(NamedTuple):
    type: Tok
    value: str
    line: int
    col: int


# ============================================================
# Keyword table
# ============================================================
# Maps identifier text -> Tok for all keywords.
# Matches the keyword table in lithos-lexer.s exactly.

KEYWORDS: dict[str, Tok] = {
    "if":       Tok.IF,
    "for":      Tok.FOR,
    "var":      Tok.VAR,
    "buf":      Tok.BUF,
    "f32":      Tok.F32,
    "u32":      Tok.U32,
    "s32":      Tok.S32,
    "f16":      Tok.F16,
    "ptr":      Tok.PTR,
    "each":     Tok.EACH,
    "else":     Tok.ELSE,
    "elif":     Tok.ELIF,
    "void":     Tok.VOID,
    "bind":     Tok.BIND,
    "exit":     Tok.EXIT,
    "host":     Tok.HOST,
    "goto":     Tok.GOTO,
    "trap":     Tok.TRAP,
    "param":    Tok.PARAM,
    "while":    Tok.WHILE,
    "const":    Tok.CONST,
    "layer":    Tok.LAYER,
    "label":    Tok.LABEL,
    "kernel":   Tok.KERNEL,
    "stride":   Tok.STRIDE,
    "return":   Tok.RETURN,
    "weight":   Tok.WEIGHT,
    "shared":   Tok.SHARED,
    "endfor":   Tok.ENDFOR,
    "runtime":  Tok.RUNTIME,
    "barrier":  Tok.BARRIER,
    "project":  Tok.PROJECT,
    "template": Tok.TEMPLATE,
    "continue": Tok.CONTINUE,
    "constant": Tok.CONSTANT,
}

# Single-char operators/punctuation -> Tok
SINGLE_CHAR_OPS: dict[str, Tok] = {
    "+": Tok.PLUS,
    "-": Tok.MINUS,
    "*": Tok.STAR,
    "/": Tok.SLASH,
    "=": Tok.EQ,
    "<": Tok.LT,
    ">": Tok.GT,
    "&": Tok.AMP,
    "|": Tok.PIPE,
    "^": Tok.CARET,
    "[": Tok.LBRACK,
    "]": Tok.RBRACK,
    "(": Tok.LPAREN,
    ")": Tok.RPAREN,
    ",": Tok.COMMA,
    ":": Tok.COLON,
    ".": Tok.DOT,
    "@": Tok.AT,
    "#": Tok.INDEX,
    "$": Tok.DOLLAR,
}

# Two-char operators -> Tok
TWO_CHAR_OPS: dict[str, Tok] = {
    "==": Tok.EQEQ,
    "!=": Tok.NEQ,
    "<=": Tok.LTE,
    ">=": Tok.GTE,
    "<<": Tok.SHL,
    ">>": Tok.SHR,
}

# Unicode operators (multi-byte UTF-8)
UNICODE_OPS: dict[str, Tok] = {
    "\u2192": Tok.LOAD,       # → (E2 86 92)
    "\u2190": Tok.STORE,      # ← (E2 86 90)
    "\u2191": Tok.REG_READ,   # ↑ (E2 86 91)
    "\u2193": Tok.REG_WRITE,  # ↓ (E2 86 93)
    "\u03A3": Tok.SUM,        # Σ (CE A3)
    "\u25B3": Tok.MAX,        # △ (E2 96 B3)
    "\u25BD": Tok.MIN,        # ▽ (E2 96 BD)
    "\u221A": Tok.SQRT,       # √ (E2 88 9A)
    "\u2245": Tok.SIN,        # ≅ (E2 89 85)
    "\u2261": Tok.COS,        # ≡ (E2 89 A1)
}


# ============================================================
# Character classification helpers
# ============================================================

def _is_alpha(c: str) -> bool:
    return c.isalpha() or c == "_"


def _is_alnum(c: str) -> bool:
    return c.isalnum() or c == "_"


def _is_ident_char(c: str) -> bool:
    """Characters that can appear inside an identifier.

    The Lithos lexer (scan_ident) accepts alphanumerics, underscore,
    dot, angle brackets, '=', and '!' inside identifiers.  This allows
    compound identifiers like ``shfl.bfly``, ``u32>f32``, ``if>=``, etc.
    """
    return _is_alnum(c) or c in (".", "<", ">", "=", "!")


def _is_hex_digit(c: str) -> bool:
    return c in "0123456789abcdefABCDEF"


# ============================================================
# Core lexer
# ============================================================

def lex(source: str) -> list[Token]:
    """Tokenize a Lithos .ls source string.

    Returns a list of Token objects.  The final token is always TOK_EOF.
    """
    tokens: list[Token] = []
    src = source
    length = len(src)
    pos = 0
    line = 1
    line_start_offset = 0   # offset of current line start (for column calc)
    at_line_start = True     # next non-consumed char is beginning of line

    def col() -> int:
        return pos - line_start_offset + 1

    def emit(tok_type: Tok, value: str, tok_line: int, tok_col: int) -> None:
        tokens.append(Token(tok_type, value, tok_line, tok_col))

    while pos < length:
        # ---- Start of line: measure indentation ----
        if at_line_start:
            indent = 0
            indent_col = col()
            while pos < length:
                c = src[pos]
                if c == " ":
                    indent += 1
                    pos += 1
                elif c == "\t":
                    indent += 4
                    pos += 1
                else:
                    break
            # Emit indent token even if 0 — parser needs line boundaries
            emit(Tok.INDENT, str(indent), line, indent_col)
            at_line_start = False
            if pos >= length:
                break
            # Fall through to process the first non-whitespace char

        c = src[pos]

        # ---- Newline ----
        if c == "\n":
            emit(Tok.NEWLINE, "\\n", line, col())
            pos += 1
            # Skip CR after LF (Windows)
            if pos < length and src[pos] == "\r":
                pos += 1
            line += 1
            line_start_offset = pos
            at_line_start = True
            continue

        if c == "\r":
            emit(Tok.NEWLINE, "\\n", line, col())
            pos += 1
            # Skip LF after CR (Windows)
            if pos < length and src[pos] == "\n":
                pos += 1
            line += 1
            line_start_offset = pos
            at_line_start = True
            continue

        # ---- Inline whitespace (space, tab) ----
        if c in (" ", "\t"):
            pos += 1
            continue

        # ---- Comment: \\ to end of line ----
        if c == "\\" and pos + 1 < length and src[pos + 1] == "\\":
            # Consume to end of line
            pos += 2
            while pos < length and src[pos] not in ("\n", "\r"):
                pos += 1
            continue

        # ---- Number literals ----
        if c.isdigit():
            start = pos
            start_col = col()
            # Hex?
            if c == "0" and pos + 1 < length and src[pos + 1] in ("x", "X"):
                pos += 2
                while pos < length and _is_hex_digit(src[pos]):
                    pos += 1
                emit(Tok.INT, src[start:pos], line, start_col)
                continue
            # Decimal / float
            is_float = False
            while pos < length and (src[pos].isdigit() or src[pos] == "."):
                if src[pos] == ".":
                    is_float = True
                pos += 1
            tok_type = Tok.FLOAT if is_float else Tok.INT
            emit(tok_type, src[start:pos], line, start_col)
            continue

        # Negative number: '-' followed immediately by digit
        if c == "-" and pos + 1 < length and src[pos + 1].isdigit():
            start = pos
            start_col = col()
            pos += 1  # skip '-'
            is_float = False
            while pos < length and (src[pos].isdigit() or src[pos] == "."):
                if src[pos] == ".":
                    is_float = True
                pos += 1
            tok_type = Tok.FLOAT if is_float else Tok.INT
            emit(tok_type, src[start:pos], line, start_col)
            continue

        # ---- String literals (TOK_STRING = 6) ----
        # The .ls lexer source doesn't show string handling (it notes the
        # language lacks string literals), but the assembly token table has
        # TOK_STRING=6 and the parser references it.  We handle double-quoted
        # strings here for completeness.
        if c == '"':
            start = pos
            start_col = col()
            pos += 1
            while pos < length and src[pos] != '"':
                if src[pos] == "\\":
                    pos += 1  # skip escaped char
                pos += 1
            if pos < length:
                pos += 1  # skip closing quote
            emit(Tok.STRING, src[start:pos], line, start_col)
            continue

        # ---- Identifiers and keywords ----
        if _is_alpha(c):
            start = pos
            start_col = col()
            pos += 1
            while pos < length and _is_ident_char(src[pos]):
                pos += 1
            text = src[start:pos]
            tok_type = KEYWORDS.get(text, Tok.IDENT)
            emit(tok_type, text, line, start_col)
            continue

        # ---- Two-character operators (must check before single-char) ----
        if pos + 1 < length:
            two = src[pos:pos + 2]
            if two in TWO_CHAR_OPS:
                start_col = col()
                emit(TWO_CHAR_OPS[two], two, line, start_col)
                pos += 2
                continue

        # ---- Unicode operators ----
        if c in UNICODE_OPS:
            start_col = col()
            emit(UNICODE_OPS[c], c, line, start_col)
            pos += 1
            continue

        # ---- Single-character operators and punctuation ----
        if c in SINGLE_CHAR_OPS:
            start_col = col()
            emit(SINGLE_CHAR_OPS[c], c, line, start_col)
            pos += 1
            continue

        # ---- Unknown character: skip silently ----
        pos += 1

    # ---- Final EOF token ----
    emit(Tok.EOF, "", line, col())
    return tokens


# ============================================================
# Pretty-print helper
# ============================================================

def tok_name(t: Tok) -> str:
    """Return a human-friendly token type name like 'TOK_IDENT'."""
    return f"TOK_{t.name}"


def dump_tokens(tokens: list[Token]) -> None:
    """Print a token list in a readable tabular format."""
    for tok in tokens:
        name = tok_name(tok.type).ljust(16)
        loc = f"{tok.line}:{tok.col}".ljust(8)
        val = repr(tok.value)
        print(f"  {loc} {name} {val}")


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    sample = r"""
\\ Example Lithos snippet
buf weights 1024
var total 0

add x y :
    result = x + y

kernel matmul param A ptr param B ptr param C ptr :
    idx = each 0 1024
    sum = 0
    for i 0 1024
        a = A [ idx * 1024 + i ]
        b = B [ i * 1024 + idx ]
        sum = sum + a * b
    endfor
    C [ idx ] = sum

if total == 0
    total = 42
elif total >= 100
    total = total - 1
else
    total = total + 1

\\ Unicode operators
← 32 addr val
→ 32 addr
↑ $5
↓ $7 val

\\ Hex literal and negative
x = 0xFF00
y = -3
z = 3.14
trap 0
"""

    print("Lithos Python Lexer — self-test")
    print("=" * 50)
    toks = lex(sample)
    dump_tokens(toks)
    print(f"\nTotal tokens: {len(toks)}")

    # Sanity checks
    types = [t.type for t in toks]
    assert Tok.EOF in types, "missing EOF"
    assert Tok.BUF in types, "missing BUF keyword"
    assert Tok.KERNEL in types, "missing KERNEL keyword"
    assert Tok.STORE in types, "missing STORE (←)"
    assert Tok.LOAD in types, "missing LOAD (→)"
    assert Tok.REG_READ in types, "missing REG_READ (↑)"
    assert Tok.REG_WRITE in types, "missing REG_WRITE (↓)"
    assert Tok.EQEQ in types, "missing EQEQ (==)"
    assert Tok.GTE in types, "missing GTE (>=)"
    assert Tok.TRAP in types, "missing TRAP keyword"
    assert Tok.DOLLAR in types, "missing DOLLAR ($)"

    # Check a hex literal was lexed
    hex_toks = [t for t in toks if t.type == Tok.INT and t.value.startswith("0x")]
    assert hex_toks, "missing hex literal"
    assert hex_toks[0].value == "0xFF00"

    # Check negative number
    neg_toks = [t for t in toks if t.type == Tok.INT and t.value.startswith("-")]
    assert neg_toks, "missing negative integer"

    # Check float
    float_toks = [t for t in toks if t.type == Tok.FLOAT]
    assert float_toks, "missing float literal"
    assert float_toks[0].value == "3.14"

    print("\nAll sanity checks passed.")
