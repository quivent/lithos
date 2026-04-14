#!/usr/bin/env python3
"""Lithos bootstrap compiler (Python implementation).

Usage:
    python main.py input.ls -o output
    python main.py input.ls -o output.cubin --target gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -- Bootstrap modules (same package) --
from bridge import lex_and_convert
from lexer import lex, dump_tokens
from parser import Parser, dump_ast
from codegen import CodeGenerator
from emit_arm64 import ARM64Emitter
from emit_sm90 import SM90Emitter
from elf_writer import ELFWriter, CubinWriter


# ============================================================
# Command-line argument parsing
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lithos-bootstrap",
        description="Lithos bootstrap compiler (Python implementation)",
    )
    p.add_argument(
        "input",
        help="Input .ls source file",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (ELF binary or .cubin)",
    )
    p.add_argument(
        "--target",
        choices=["arm64", "gpu"],
        default="arm64",
        help="Target architecture: arm64 (default) or gpu (SM90 cubin)",
    )
    p.add_argument(
        "--dump-tokens",
        action="store_true",
        help="Print lexer tokens and exit",
    )
    p.add_argument(
        "--dump-ast",
        action="store_true",
        help="Print parser AST and exit",
    )
    return p


# ============================================================
# Pipeline stages
# ============================================================

def read_source(path: str) -> str:
    """Read a .ls source file and return its contents as a string."""
    p = Path(path)
    if not p.exists():
        print(f"lithos: cannot open source: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def compile_arm64(ast: list, output_path: str) -> None:
    """Emit ARM64 machine code from AST and write an ELF binary."""
    emitter = ARM64Emitter()
    cg = CodeGenerator(emitter)
    cg.generate(ast)

    code = emitter.get_code()

    writer = ELFWriter()
    writer.add_text(".text", code)
    writer.write(output_path)
    print(f"lithos: wrote {output_path} ({Path(output_path).stat().st_size} bytes, ARM64 ELF)")


def compile_gpu(ast: list, output_path: str) -> None:
    """Emit SM90 GPU code from AST and write a cubin."""
    emitter = SM90Emitter()
    cg = CodeGenerator(emitter)
    cg.generate(ast)

    code = emitter.get_code()
    regcount = emitter.get_register_count()

    writer = CubinWriter()
    writer.add_text("kernel", code)
    writer.add_nv_info(regcount=regcount)
    writer.write(output_path)
    print(f"lithos: wrote {output_path} ({Path(output_path).stat().st_size} bytes, SM90 cubin)")


# ============================================================
# Main entry point
# ============================================================

def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # Step 1: Read source
    source = read_source(args.input)

    # Step 2: Lex
    raw_tokens = lex(source)

    if args.dump_tokens:
        dump_tokens(raw_tokens)
        print(f"\nTotal tokens: {len(raw_tokens)}")
        return 0

    # Step 3: Parse
    tokens = lex_and_convert(source)
    parser = Parser(tokens)
    ast = parser.parse()

    if args.dump_ast:
        print(dump_ast(ast))
        return 0

    # Step 4+5: Emit code and write output
    if args.target == "arm64":
        compile_arm64(ast, args.output)
    elif args.target == "gpu":
        compile_gpu(ast, args.output)
    else:
        print(f"lithos: unknown target: {args.target}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
