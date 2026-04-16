#!/usr/bin/env python3
"""lint-ls.py -- Lint .ls files for invented English operation names.

Lithos has a specific primitive vocabulary (25 symbols across 12 families).
This linter catches identifiers that look like someone wrote English
operation names (e.g. "reduce_sum", "multiply_by_weight", "compute_output")
instead of using the actual Lithos primitives.

Usage:
    tools/lint-ls.py file1.ls [file2.ls ...]
    tools/lint-ls.py --strict file1.ls       # flag ALL unknown snake_case
"""

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# Whitelist: everything that legitimately appears in .ls source
# ---------------------------------------------------------------------------

# Spec section 2.6 -- keywords
KEYWORDS = {
    "for", "each", "stride", "endfor",
    "if", "else",
    "if==", "if>=", "if<", "if>", "if!=",
    "trap",
    "param", "shared", "const",
    "label", "bra", "exit",
    "barrier",
}

# Type names and width specifiers
TYPES = {
    "u8", "u16", "u32", "u64",
    "f16", "f32", "f64",
    "ptr",
}

# Spec section 3.10 -- named primitives
NAMED_PRIMITIVES = {
    "project", "matvec",
}

# Spec section 6 -- standard compositions
STANDARD_COMPOSITIONS = {
    "sigmoid", "silu", "softplus", "rmsnorm", "l2norm",
}

# Low-level SASS-like instructions that appear in handwritten .ls
SASS_LIKE_OPS = {
    "fma", "neg", "rcp", "rsqrt", "exp", "lg", "exp2", "cos", "sin",
    "shl", "shr", "and", "or", "xor", "not",
    "add", "sub", "mul", "div", "mad",
    "setp", "cvt",
    # Shuffle instructions
    "shfl",
}

# Shuffle sub-types that appear as compound tokens
SHUFFLE_TYPES = {
    "shfl.down", "shfl.bfly", "shfl.up", "shfl.idx",
    "bar.sync",
}

# Type conversion ops
CONVERSION_OPS = {
    "u32>f32", "f32>u32", "f16>f32", "f32>f16",
    "cvt.f32.f16", "cvt.f16.f32",
}

# Inference kernel composition names (from inference/*.ls)
INFERENCE_COMPOSITIONS = {
    # attend.ls
    "rope", "attention_score",
    # attend_full.ls
    "softmax", "mrope_apply", "attention_score_full", "attention_output_full",
    # decay_gate.ls
    "decay_gate",
    # delta_update.ls
    "delta_update",
    # deltanet_fused.ls
    "deltanet_fused",
    # elementwise.ls
    "residual_add", "elemwise_mul", "activate_silu", "scale",
    # embed.ls
    "token_embed", "token_embed_f32",
    # gemv.ls
    "gptq_gemv",
    # recur.ls
    "conv1d_infer", "gate_sigmoid", "deltanet_step", "state_rollback",
    # reduce.ls
    "rmsnorm_residual", "sample_argmax",
    # tid_write.ls
    "kernel",
}

# Runtime composition names (from runtime/*.ls)
RUNTIME_COMPOSITIONS = {
    "doorbell_write_gpput", "doorbell_ring",
    "elf_read_header", "elf_shdr_offset", "elf_shdr_name",
    "elf_section_data", "elf_streq", "elf_find_section",
    "elf_gpu_alloc", "elf_copy_to_gpu", "elf_section_to_gpu",
    "elf_find_kernel_symbol", "elf_load",
    "openat", "mmap", "pci_device_open", "vfio_open",
    "bar0_map", "bar4_map", "pmc_sanity_check", "gsp_boot",
    "runtime_init", "runtime_teardown", "runtime_exit",
    "sync_allocate_flag", "sync_reset_flag", "sync_wait_flag",
    "sync_wait_flag_timeout", "sync_wait_counter",
    "sync_wait_gpget", "sync_drain",
    "launch_kernel", "launch_kernel_sync",
    "munmap", "close_fd", "gsp_shutdown",
    "dispatch_kernel", "wait_for_completion",
    "cbuf0_alloc", "cbuf0_set_register_count",
    "cbuf0_set_param_block", "cbuf0_finalize",
    "qmd_init", "qmd_set_block_dim", "qmd_set_grid_dim",
    "qmd_set_shared_mem", "qmd_set_entry_pc",
    "mem_init", "mem_alloc", "mem_alloc_aligned", "mem_reset",
    "mem_copy", "mem_set32",
    "pb_init", "pb_emit_header_inc", "pb_emit_header_noninc",
    "pb_emit_word", "pb_emit_header", "pb_emit_cb_load",
    "spd_init", "spd_build",
    "pb_emit_launch", "pb_emit_qmd",
}

# Compiler composition names (from compiler/*.ls)
COMPILER_COMPOSITIONS = {
    "emit_token", "is_alpha", "is_digit", "is_hex_digit",
    "is_alnum", "is_space", "is_newline",
    "scan_ident", "scan_number", "scan_to_eol",
    "match_keyword", "classify_number",
    "lex_match_twochar", "lex_match_e2",
    "lex", "lithos_lex",
    # Emitters
    "sinst", "emit_fadd", "emit_fsub", "emit_fmul", "emit_fdiv",
    "emit_sqrt", "emit_sin", "emit_cos",
    "emit_shfl_bfly", "emit_ldg", "emit_stg",
    "emit_imad", "emit_isetp_ge", "emit_isetp_eq",
    "emit_bra_pred", "emit_s2r", "emit_mov_imm", "emit_exit",
    # ARM64 emitters
    "arm64_emit32",
    "emit_a64_fadd", "emit_a64_fsub", "emit_a64_fmul", "emit_a64_fdiv",
    "emit_a64_sin", "emit_a64_cos", "emit_a64_mov_imm",
    "emit_a64_add", "emit_a64_sub", "emit_a64_mul", "emit_a64_sdiv",
    "emit_a64_and", "emit_a64_orr", "emit_a64_eor",
    "emit_a64_lsl", "emit_a64_lsr", "emit_a64_ret",
    "emit_a64_stur", "emit_a64_ldur",
    "emit_a64_sub_imm", "emit_a64_add_imm",
    "emit_a64_bl", "emit_a64_b", "emit_a64_b_cond", "emit_a64_cmp",
    "emit_a64_stp_fp_lr", "emit_a64_mov_fp_sp", "emit_a64_ldp_fp_lr",
    "emit_a64_ldr64", "emit_a64_ldr32", "emit_a64_ldrh", "emit_a64_ldrb",
    "emit_a64_str64", "emit_a64_str32", "emit_a64_strh", "emit_a64_strb",
    "emit_a64_mov_reg", "emit_a64_cmp_reg",
    "emit_a64_bcond_ph", "emit_a64_b_ph",
    # Code buffer ops
    "cb_emit", "cw_emit", "cd_emit", "cq_emit", "cpad", "calign",
    # ELF ops
    "elf_put_u16", "elf_put_u32", "elf_put_u64", "elf_emit_str",
    "shdr64_a", "shdr64_b", "sym64_emit",
    "nvi_u32_emit", "nvi_sval_emit",
    "elf_init", "elf_write_header",
    "c_sn_set", "c_emit_n_shstrtab", "c_emit_n_strtab",
    "c_emit_n_symtab", "c_emit_n_nvinfo", "c_emit_n_nvinfo_k",
    "c_emit_n_text", "c_emit_n_const0", "c_emit_n_shared",
    "elf_emit_shstrtab", "elf_emit_strtab_symtab",
    "elf_emit_nvinfo", "nvi_kparam_emit", "nvi_kparams_all",
    "nvi_emit_fixed_a", "nvi_emit_fixed_b", "elf_emit_nvinfo_k",
    "c_emit_code_bytes", "c_emit_zeros",
    "elf_emit_text", "elf_emit_const0", "elf_emit_text_const0",
    "elf_emit_shdrs_head", "elf_emit_shdrs_tail", "elf_emit_shdrs",
    "elf_build", "elf_save",
    "cubin_write",
    "a64e_b", "a64e_w", "a64e_d", "a64e_q",
    "arm64_elf_build", "arm64_elf_save", "arm64_elf_write",
    # Walker / parser
    "tok_type", "tok_offset", "tok_length",
    "sym_reset", "sym_add", "sym_find",
    "comp_reset", "comp_add", "comp_find",
    "buf_collect", "buf_find", "emit_buf_addr",
    "const_collect", "const_find",
    "reg_reset", "alloc_scratch", "alloc_slot", "alloc_reg",
    "vpush", "vpop", "vpush_with_op",
    "op_apply_hi", "op_apply_lo", "op_flush",
    "wb_set_op", "wb_emit_return", "wb_eval_one_atom",
    "wb_parse_cond", "wb_emit_if", "wb_emit_regread", "wb_emit_regwrite",
    "walk_collect", "bind_args", "emit_primitive",
    "emit_subscript", "emit_assign", "emit_load", "emit_store",
    "emit_reg_read", "emit_reg_write",
    "parse_int_tok",
    "label_reset", "label_add", "label_lookup",
    "goto_fix_add", "goto_fix_patch",
    "load_atom_to_reg", "walk_body",
    "walk_for_kw", "walk_each_kw", "walk_stride_kw", "walk_if_kw",
    "consume_operands", "walk_top_level",
    "mmap_file", "write_file", "main",
    # Safetensors (compiler/safetensors.ls)
    "sys_openat", "sys_close", "sys_lseek", "sys_mmap",
    "st_open", "read_u64_le", "read_u32_le",
    "st_skip_ws", "st_find_char", "st_parse_quoted",
    "st_parse_u64_dec", "st_match_bytes", "st_parse_dtype",
    "st_parse_descriptor", "st_parse_header", "st_find_tensor",
    "st_close", "st_load",
    # Config reader (compiler/config-reader.ls)
    "cfg_sys_openat", "cfg_sys_close", "cfg_sys_lseek", "cfg_sys_mmap",
    "cfg_skip_ws", "cfg_find_char",
    # Mini compiler (compiler/mini.ls)
    "mmap_rw", "st8", "st32", "st64", "ld8", "ld64",
    "open_ro", "open_wr", "fsize", "fseek0", "fclose", "fread", "fwrite",
    "emit32", "enc_movz", "enc_movk", "emit_mov_imm16", "emit_svc80",
    "wu32", "wu64", "memcpy_bytes", "write_macho",
    "skip_ws", "parse_uint", "compile_src", "lithos_main",
}

# Combine all known composition/function names
ALL_KNOWN_IDENTIFIERS = (
    KEYWORDS | TYPES | NAMED_PRIMITIVES | STANDARD_COMPOSITIONS |
    SASS_LIKE_OPS | INFERENCE_COMPOSITIONS | RUNTIME_COMPOSITIONS |
    COMPILER_COMPOSITIONS
)

# Also allow the "host" modifier keyword
ALL_KNOWN_IDENTIFIERS.add("host")
ALL_KNOWN_IDENTIFIERS.add("blockDim")
ALL_KNOWN_IDENTIFIERS.add("gridDim")

# Level-1 plain-English verbs used in unroll.ls files (layer decomposition format)
UNROLL_VERBS = {
    "normalize", "project", "convolve", "activate", "slice", "unitize",
    "rescale", "squash", "decay", "recur", "gate", "add", "rotate",
    "append", "score", "weight",
}
ALL_KNOWN_IDENTIFIERS |= UNROLL_VERBS

# ---------------------------------------------------------------------------
# Suspicious substrings -- English verbs/nouns that signal invented ops
# ---------------------------------------------------------------------------

SUSPICIOUS_SUBSTRINGS = [
    "multiply", "compute", "calculate", "subtract", "divide",
    "accumulate", "extract", "normalize", "process",
    "transform", "convert", "aggregate", "concatenate",
    "initialize", "finalize_output", "apply_weight",
    "reduce_sum", "reduce_max", "reduce_min",
    "load_weight", "load_input", "load_data",
    "store_output", "store_result",
    "square_root", "reciprocal",
    "elementwise_add", "elementwise_mul", "elementwise_sub",
    "matrix_multiply", "dot_product", "inner_product", "outer_product",
    "forward_pass", "backward_pass",
    "attention_compute", "softmax_compute",
    "weight_update", "gradient",
    "broadcast", "scatter", "gather",
    "activation_fn", "nonlinear",
]

# Simpler substring patterns for partial matches within identifiers
SUSPICIOUS_PARTS = [
    "multiply", "compute", "calculate", "subtract", "divide",
    "accumulate", "extract", "normalize", "process",
    "transform", "aggregate", "concatenate",
    "square_root", "reciprocal", "dot_product",
    "inner_product", "outer_product", "matrix_mul",
    "forward_pass", "backward_pass", "weight_update",
    "gradient", "nonlinear", "activation_fn",
    "reduce_sum", "reduce_max", "reduce_min",
    "load_weight", "load_input", "load_data",
    "store_output", "store_result",
]

# Regex for multi-word snake_case: at least two segments joined by underscore
SNAKE_CASE_RE = re.compile(r'^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+$')

# Regex for identifiers on a line (alphanumeric + underscore sequences)
IDENT_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')


def strip_comment(line):
    """Remove \\\\ comment from a line."""
    # Lithos uses \\ for comments
    idx = line.find('\\\\')
    if idx == -1:
        # Also handle single backslash comments (common shorthand)
        idx = line.find('\\')
        if idx >= 0:
            return line[:idx]
        return line
    return line[:idx]


def is_suspicious_substring(ident):
    """Check if an identifier contains suspicious English operation substrings."""
    lower = ident.lower()
    for part in SUSPICIOUS_PARTS:
        if part in lower:
            return True
    return False


def is_strict_suspicious(ident):
    """In strict mode, flag any multi-word snake_case not in the whitelist."""
    return SNAKE_CASE_RE.match(ident) is not None


def lint_line(line, lineno, filename, strict=False):
    """Lint a single line. Returns list of (lineno, ident, reason) warnings."""
    warnings = []

    # Strip comment portion
    code = strip_comment(line)
    if not code.strip():
        return warnings

    # Extract all identifiers from the code portion
    idents = IDENT_RE.findall(code)

    for ident in idents:
        # Skip single-character or very short identifiers (loop vars, etc.)
        if len(ident) <= 2:
            continue

        # Skip pure numeric-looking things that matched (shouldn't, but safety)
        if ident[0].isdigit():
            continue

        # Skip known identifiers
        if ident in ALL_KNOWN_IDENTIFIERS:
            continue

        # Check for suspicious English operation names
        if is_suspicious_substring(ident):
            warnings.append((lineno, ident,
                "looks like an invented English operation name"))
            continue

        # In strict mode, flag any multi-word snake_case not in whitelist
        if strict and is_strict_suspicious(ident):
            warnings.append((lineno, ident,
                "unknown snake_case identifier (--strict)"))

    return warnings


def lint_file(filepath, strict=False):
    """Lint a single .ls file. Returns list of warnings."""
    warnings = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line_warnings = lint_line(line, lineno, filepath, strict)
                for w in line_warnings:
                    warnings.append((filepath, w[0], w[1], w[2]))
    except (OSError, UnicodeDecodeError) as e:
        print(f"ERROR: {filepath}: {e}", file=sys.stderr)
    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Lint .ls files for invented English operation names")
    parser.add_argument("files", nargs="+", help=".ls file paths to lint")
    parser.add_argument("--strict", action="store_true",
        help="Flag ALL multi-word snake_case identifiers not in the whitelist")
    args = parser.parse_args()

    all_warnings = []
    for filepath in args.files:
        if not os.path.isfile(filepath):
            print(f"ERROR: {filepath}: not found", file=sys.stderr)
            continue
        all_warnings.extend(lint_file(filepath, strict=args.strict))

    if all_warnings:
        for filepath, lineno, ident, reason in all_warnings:
            print(f"{filepath}:{lineno}: WARNING: `{ident}` -- {reason}")
        print(f"\n{len(all_warnings)} warning(s) in {len(args.files)} file(s).")
        sys.exit(1)
    else:
        print(f"OK: {len(args.files)} file(s) clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
