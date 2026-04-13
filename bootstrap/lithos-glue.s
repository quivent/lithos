// lithos-glue.s — Safety-net stubs for symbols not yet defined elsewhere
//
// This file provides stub implementations for symbols that are referenced
// by other modules but not yet defined in the current build.
//
// Currently provides:
//   - parse_data_decl  (referenced by lithos-compose.s)
//
// All other symbols from the 43-symbol list are provided by:
//   - lithos-parser.s  (parse_*, sym_*, alloc_reg, reset_regs, etc.)
//   - emit-arm64.s     (emit_*, emit32, patch_b, patch_b_cond, etc.)

// ============================================================
// Global exports
// ============================================================
.global parse_data_decl

// ============================================================
// parse_data_decl — parse "data name size" declaration
//
// Stub: currently a no-op that returns immediately.
// The full implementation will parse a data-section declaration
// and allocate space in the output binary's .data segment.
//
// Once the full parser supports "data" declarations, this stub
// should be replaced with the real implementation.
// ============================================================
.text
.align 4
parse_data_decl:
    ret
