"""
Lithos lexer.

See LANGUAGE.md for the full language spec. In short: C-like surface syntax,
explicit memory spaces (@global, @shared, ...), explicit register
declarations, labels + goto, compile-time const, and an asm { ... } escape
hatch for raw PTX.

The lexer is hand-written: no regex, no dependencies, one pass over the
source. Whitespace and // line comments are dropped; asm { ... } blocks are
captured verbatim as a single ASM token.
"""

from dataclasses import dataclass
from typing import List


# Keywords that are not identifiers. `return`, `if`, `else`, `goto` are
# control flow; `kernel`, `reg`, `shared`, `const` are declarations.
KEYWORDS = {
    "kernel", "reg", "shared", "const", "if", "else", "goto", "return",
    "cast_u64", "cast_u32", "cast_f32", "cast_s32",
}

# Single-character punctuation tokens.
PUNCT1 = set("(){}[],;:+-*/%&|^~")

# Two-character tokens we need to recognise before falling back to single.
PUNCT2 = {"==", "!=", "<=", ">=", "<<", ">>", "&&", "||", "->"}


@dataclass
class Tok:
    kind: str       # 'ident' | 'int' | 'float' | 'punct' | 'kw' | 'space' | 'asm'
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Tok({self.kind}, {self.value!r}, {self.line}:{self.col})"


class LexError(Exception):
    pass


def tokenize(src: str) -> List[Tok]:
    toks: List[Tok] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

    def advance(k: int = 1) -> None:
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]

        # Whitespace.
        if c in " \t\r\n":
            advance()
            continue

        # Line comment.
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                advance()
            continue

        # Memory-space annotation: @global, @shared, @const, @local, @param.
        if c == "@":
            start_line, start_col = line, col
            advance()
            j = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                advance()
            name = src[j:i]
            if name not in {"global", "shared", "const", "local", "param"}:
                raise LexError(f"{start_line}:{start_col}: unknown memory space @{name}")
            toks.append(Tok("space", name, start_line, start_col))
            continue

        # asm { ... } — captured as a single token, payload is the body.
        if c == "a" and src[i:i + 3] == "asm" and i + 3 < n and src[i + 3] in " \t\r\n{":
            start_line, start_col = line, col
            advance(3)
            while i < n and src[i] in " \t\r\n":
                advance()
            if i >= n or src[i] != "{":
                raise LexError(f"{start_line}:{start_col}: expected {{ after asm")
            advance()  # consume {
            depth = 1
            body_start = i
            while i < n and depth > 0:
                if src[i] == "{":
                    depth += 1
                elif src[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                advance()
            if i >= n:
                raise LexError(f"{start_line}:{start_col}: unterminated asm block")
            body = src[body_start:i]
            advance()  # consume }
            toks.append(Tok("asm", body, start_line, start_col))
            continue

        # Identifier / keyword. Allow dots inside for things like tid.x,
        # shfl.down, ld.global.f32 — the parser treats these as single names
        # when the context wants an intrinsic, and splits on '.' only for
        # member-style access (which we don't use).
        if c.isalpha() or c == "_":
            start_line, start_col = line, col
            j = i
            while i < n and (src[i].isalnum() or src[i] == "_" or src[i] == "."):
                advance()
            name = src[j:i]
            # Trailing dot is not allowed.
            if name.endswith("."):
                raise LexError(f"{start_line}:{start_col}: trailing '.' in {name}")
            if name in KEYWORDS:
                toks.append(Tok("kw", name, start_line, start_col))
            else:
                toks.append(Tok("ident", name, start_line, start_col))
            continue

        # Numbers. Hex (0x...), floats (1.5, 1e3, 0f41000000 PTX-style), int.
        if c.isdigit():
            start_line, start_col = line, col
            j = i
            # PTX float literal: 0f followed by 8 hex digits — we keep it as
            # a float token with the raw text.
            if c == "0" and i + 1 < n and src[i + 1] in "fF":
                advance(2)
                while i < n and src[i] in "0123456789abcdefABCDEF":
                    advance()
                toks.append(Tok("float", src[j:i], start_line, start_col))
                continue
            # Hex int.
            if c == "0" and i + 1 < n and src[i + 1] in "xX":
                advance(2)
                while i < n and src[i] in "0123456789abcdefABCDEF":
                    advance()
                toks.append(Tok("int", src[j:i], start_line, start_col))
                continue
            # Decimal (int or float).
            is_float = False
            while i < n and src[i].isdigit():
                advance()
            if i < n and src[i] == "." and i + 1 < n and src[i + 1].isdigit():
                is_float = True
                advance()
                while i < n and src[i].isdigit():
                    advance()
            if i < n and src[i] in "eE":
                is_float = True
                advance()
                if i < n and src[i] in "+-":
                    advance()
                while i < n and src[i].isdigit():
                    advance()
            kind = "float" if is_float else "int"
            toks.append(Tok(kind, src[j:i], start_line, start_col))
            continue

        # Two-character punctuation.
        if i + 1 < n and src[i:i + 2] in PUNCT2:
            toks.append(Tok("punct", src[i:i + 2], line, col))
            advance(2)
            continue

        # Single-character punctuation.
        if c in PUNCT1:
            toks.append(Tok("punct", c, line, col))
            advance()
            continue

        # '<' '>' '!' '=' singletons (if not matched above as ==/!= etc).
        if c in "<>!=":
            toks.append(Tok("punct", c, line, col))
            advance()
            continue

        raise LexError(f"{line}:{col}: unexpected character {c!r}")

    return toks


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        for t in tokenize(f.read()):
            print(t)
