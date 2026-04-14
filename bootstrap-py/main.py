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
from lexer import lex, dump_tokens
from parser import parse
from codegen import generate
from emit_arm64 import ARM64Emitter
from emit_sm90 import SM90Emitter
from elf_writer import ELFWriter
from cubin_writer import CubinWriter


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
    generate(ast, emitter)

    code = emitter.get_code()
    code_size = emitter.get_code_size()

    writer = ELFWriter()
    elf_bytes = writer.build(
        text=code,
        text_size=code_size,
        data=b"",
        data_size=0,
        bss_size=0,
    )

    out = Path(output_path)
    out.write_bytes(elf_bytes)
    print(f"lithos: wrote {out} ({len(elf_bytes)} bytes, ARM64 ELF)")


def compile_gpu(ast: list, output_path: str) -> None:
    """Emit SM90 GPU code from AST and write a cubin."""
    emitter = SM90Emitter()
    generate(ast, emitter)

    code = emitter.get_code()
    code_size = emitter.get_code_size()

    writer = CubinWriter()
    cubin_bytes = writer.build(
        text=code,
        text_size=code_size,
        num_registers=emitter.max_registers_used(),
        shared_mem_size=emitter.shared_mem_size(),
        num_params=emitter.num_params(),
    )

    out = Path(output_path)
    out.write_bytes(cubin_bytes)
    print(f"lithos: wrote {out} ({len(cubin_bytes)} bytes, SM90 cubin)")


# ============================================================
# Main entry point
# ============================================================

def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # Step 1: Read source
    source = read_source(args.input)

    # Step 2: Lex
    tokens = lex(source)

    if args.dump_tokens:
        dump_tokens(tokens)
        print(f"\nTotal tokens: {len(tokens)}")
        return 0

    # Step 3: Parse
    ast = parse(tokens)

    if args.dump_ast:
        import json
        print(json.dumps(ast, indent=2, default=str))
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
