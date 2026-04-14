// ============================================================
// ls-shared.s — Canonical shared state for the Lithos bootstrap
// ============================================================
// This file is the single authoritative declaration of every
// piece of shared state used across the Lithos bootstrap stack:
//
//     driver.s → lithos-lexer.s → lithos-parser.s →
//     lithos-expr.s / emit-arm64.s → lithos-elf-writer.s
//
// Before unification, multiple worker .s files each declared
// their own "code_buf", "sym_table", "token buffer", etc.  When
// assembled and linked together those symbols collided (or, worse,
// stayed locally-scoped and silently diverged — every writer
// wrote into its own private copy).
//
// Every symbol declared below is `.globl`.  Every worker file
// `.extern`s the name it needs and writes into the one shared
// buffer.  Private per-pass scratch (parser patch_stack, expr
// register allocator, elf section metadata, etc.) stays private
// to the file that owns it.
//
// Size rationale (matches the largest prior allocation so no
// worker regresses):
//
//   ls_code_buf        1 MB   — ARM64 and SASS emission target
//   ls_token_buf       1 MB   — lexer output, 87381 triples × 12 B
//   ls_sym_table       24 KB  — 512 entries × 48 B  (covers
//                                bootstrap 256×48, expr 128×48,
//                                parser 512×24)
//   ls_comp_table      12 KB  — 256 entries × 48 B (composition defs)
//   ls_elf_buf         4 MB   — output ELF / cubin (megakernels)
//   ls_source_buf_ptr  8 B    — pointer to mmap'd source
//
// NB: ls_source_buf itself is allocated at run-time via mmap
// (16 MB in driver.s).  The canonical *pointer* to that mmap'd
// region lives here as ls_source_buf_ptr.
// ============================================================

.equ LS_CODE_BUF_SIZE,   1048576          // 1 MB
.equ LS_TOKEN_BUF_SIZE,  1048572          // 87381 × 12 B (lexer cap)
.equ LS_SYM_ENTRY_SIZE,  48
.equ LS_SYM_MAX,         1024
.equ LS_SYM_TABLE_SIZE,  (LS_SYM_ENTRY_SIZE * LS_SYM_MAX)   // 24 576
.equ LS_COMP_ENTRY_SIZE, 48
.equ LS_COMP_MAX,        256
.equ LS_COMP_TABLE_SIZE, (LS_COMP_ENTRY_SIZE * LS_COMP_MAX) // 12288
.equ LS_ELF_BUF_SIZE,    4194304          // 4 MB (megakernel cubins)
.equ LS_SOURCE_BUF_SIZE, 16777216         // 16 MB (allocated via mmap)

// ------------------------------------------------------------
// .data — cursors and pointers that must be zero-initialised in
// a known section.  (Some assemblers place .bss "quad 0" in an
// implicit .data; we keep cursors in .data for clarity.)
// ------------------------------------------------------------
.data
.align 3

.globl ls_code_pos
ls_code_pos:        .quad 0           // byte offset into ls_code_buf

.globl ls_token_count
ls_token_count:     .quad 0           // number of tokens in ls_token_buf

.globl ls_sym_count
ls_sym_count:       .quad 0           // number of live entries in ls_sym_table

.globl ls_comp_count
ls_comp_count:      .quad 0           // number of live entries in ls_comp_table

.globl ls_elf_pos
ls_elf_pos:         .quad 0           // byte offset into ls_elf_buf

.globl ls_source_buf_ptr
ls_source_buf_ptr:  .quad 0           // base of mmap'd source buffer

.globl ls_source_len
ls_source_len:      .quad 0           // total source bytes

.globl ls_data_pos
ls_data_pos:        .quad 0           // byte offset into ls_data_buf

.globl ls_last_comp_addr
ls_last_comp_addr:  .quad 0           // code offset of last composition (entry point)

// ------------------------------------------------------------
// .bss — the big shared buffers
// ------------------------------------------------------------
.bss
.align 12                               // page-align the big buffers

.globl ls_code_buf
ls_code_buf:        .space LS_CODE_BUF_SIZE

.align 3
.globl ls_token_buf
ls_token_buf:       .space LS_TOKEN_BUF_SIZE

.align 3
.globl ls_sym_table
ls_sym_table:       .space LS_SYM_TABLE_SIZE

.align 3
.globl ls_comp_table
ls_comp_table:      .space LS_COMP_TABLE_SIZE

.align 12
.globl ls_elf_buf
ls_elf_buf:         .space LS_ELF_BUF_SIZE

.align 12
.globl ls_data_buf
ls_data_buf:        .space 65536      // 64 KB for data segment (string literals etc.)
