#!/usr/bin/env python3
"""Test harness for the Lithos Python bootstrap.

Reads runtime/doorbell.ls (the simplest .ls file), lexes it, parses it,
prints the AST, and optionally generates ARM64 code + writes a test ELF.

Usage:
    python test_bootstrap.py
    python test_bootstrap.py --emit   # also generate ARM64 code
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOORBELL_LS = PROJECT_ROOT / "runtime" / "doorbell.ls"

# Ensure bootstrap-py is on the path
sys.path.insert(0, str(SCRIPT_DIR))

from lexer import lex, dump_tokens, Tok
from parser import parse


def test_lex_doorbell() -> None:
    """Lex doorbell.ls and verify basic token properties."""
    source = DOORBELL_LS.read_text(encoding="utf-8")
    tokens = lex(source)

    print("=" * 60)
    print("LEXER OUTPUT — doorbell.ls")
    print("=" * 60)
    dump_tokens(tokens)
    print(f"\nTotal tokens: {len(tokens)}")

    # Sanity checks
    types = [t.type for t in tokens]
    assert Tok.EOF in types, "missing EOF"
    assert Tok.IDENT in types, "missing identifiers"
    assert Tok.COLON in types, "missing colon (composition separator)"
    assert Tok.STORE in types, "missing STORE arrow (doorbell uses <- for MMIO writes)"
    assert Tok.INT in types, "missing integer literals"

    # doorbell.ls defines two compositions: doorbell_write_gpput and doorbell_ring
    ident_values = [t.value for t in tokens if t.type == Tok.IDENT]
    assert "doorbell_write_gpput" in ident_values, "missing doorbell_write_gpput"
    assert "doorbell_ring" in ident_values, "missing doorbell_ring"
    assert "membar_sys" in ident_values, "missing membar_sys call"

    print("\nLexer checks passed.")


def test_parse_doorbell() -> None:
    """Parse doorbell.ls and print the AST."""
    source = DOORBELL_LS.read_text(encoding="utf-8")
    tokens = lex(source)
    ast = parse(tokens)

    print("\n" + "=" * 60)
    print("PARSER OUTPUT — doorbell.ls AST")
    print("=" * 60)
    print(json.dumps(ast, indent=2, default=str))

    # Sanity checks on the AST
    assert isinstance(ast, list), "AST should be a list of top-level nodes"
    assert len(ast) >= 2, f"expected at least 2 compositions, got {len(ast)}"

    # Check that we got function definitions
    func_nodes = [n for n in ast if n.get("kind") == "func"]
    func_names = [n["name"] for n in func_nodes]
    print(f"\nCompositions found: {func_names}")
    assert "doorbell_write_gpput" in func_names, "missing doorbell_write_gpput"
    assert "doorbell_ring" in func_names, "missing doorbell_ring"

    # doorbell_write_gpput takes (userd_va, new_gppos) -> 2 params
    dwg = next(n for n in func_nodes if n["name"] == "doorbell_write_gpput")
    assert len(dwg["params"]) == 2, f"expected 2 params, got {dwg['params']}"

    # doorbell_ring takes (userd_va) -> 1 param
    dr = next(n for n in func_nodes if n["name"] == "doorbell_ring")
    assert len(dr["params"]) == 1, f"expected 1 param, got {dr['params']}"

    print("\nParser checks passed.")


def test_codegen_doorbell() -> None:
    """Generate ARM64 code for doorbell.ls and optionally write a test ELF."""
    from codegen import generate
    from emit_arm64 import ARM64Emitter

    source = DOORBELL_LS.read_text(encoding="utf-8")
    tokens = lex(source)
    ast = parse(tokens)

    emitter = ARM64Emitter()
    generate(ast, emitter)

    code = emitter.get_code()
    code_size = emitter.get_code_size()

    print("\n" + "=" * 60)
    print("CODEGEN OUTPUT — doorbell.ls")
    print("=" * 60)
    print(f"Code size: {code_size} bytes")
    if code_size > 0:
        # Hex dump first 64 bytes
        hex_line = " ".join(f"{b:02x}" for b in code[:min(64, code_size)])
        print(f"First bytes: {hex_line}")

    # Try writing a test ELF
    try:
        from elf_writer import ELFWriter
        writer = ELFWriter()
        elf_bytes = writer.build(
            text=code,
            text_size=code_size,
            data=b"",
            data_size=0,
            bss_size=0,
        )
        out_path = SCRIPT_DIR / "test_doorbell.elf"
        out_path.write_bytes(elf_bytes)
        print(f"Wrote test ELF: {out_path} ({len(elf_bytes)} bytes)")
    except ImportError:
        print("(elf_writer not yet available — skipping ELF output)")

    print("\nCodegen checks passed.")


# ============================================================
# Main
# ============================================================

def main() -> int:
    if not DOORBELL_LS.exists():
        print(f"ERROR: cannot find {DOORBELL_LS}", file=sys.stderr)
        return 1

    emit = "--emit" in sys.argv

    test_lex_doorbell()
    test_parse_doorbell()

    if emit:
        test_codegen_doorbell()
    else:
        print("\n(pass --emit to also test ARM64 codegen + ELF writing)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
