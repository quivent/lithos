"""
bridge.py -- Convert lexer tokens to parser tokens.

The lexer (lexer.py) emits Token namedtuples: (type, value, line, col).
The parser (parser.py) expects Token dataclasses: (type, value, line, col, indent).

This module bridges the two.
"""

from lexer import lex as _lex, Token as LexToken, Tok
from parser import Token as ParseToken, TT


def convert_tokens(lex_tokens: list[LexToken]) -> list[ParseToken]:
    """Convert lexer tokens to parser tokens."""
    result = []
    current_indent = 0

    for lt in lex_tokens:
        # Map lexer token type to parser TT
        # They use the same integer IDs, so direct mapping works
        tt = int(lt.type)

        if lt.type == Tok.INDENT:
            # Lexer emits INDENT with value = indent level as string
            current_indent = int(lt.value)
            result.append(ParseToken(
                type=TT.INDENT,
                text=lt.value,
                line=lt.line,
                col=lt.col,
                indent=current_indent,
            ))
        else:
            result.append(ParseToken(
                type=tt,
                text=lt.value,
                line=lt.line,
                col=lt.col,
                indent=current_indent,
            ))

    return result


def lex_and_convert(source: str) -> list[ParseToken]:
    """Lex source and return parser-compatible tokens."""
    return convert_tokens(_lex(source))
