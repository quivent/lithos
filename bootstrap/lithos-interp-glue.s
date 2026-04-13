// lithos-interp-glue.s — satisfy lexer dictionary link references
//
// The lexer's DTC dictionary entries reference entry_ls_loop_peek
// which lives in another bootstrap file we don't link.  This stub
// provides the symbol so the linker is happy.

.data
.align 3
.globl entry_ls_loop_peek
entry_ls_loop_peek:
    .quad 0
    .quad 0
    .quad 0
