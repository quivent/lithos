.global cfa_CODE_BUF
.global cfa_CODE_POS
.global cfa_CUBIN_PARAMS
.global cfa_ELF_WRITE_ARM64
.global cfa_ELF_WRITE_CUBIN
.global cfa_LEX_COUNT
.global cfa_LEX_TOKENS
.global cfa_LITHOS_LEX
.global cfa_PARSE_TOKENS
.global cfa_drv_RET
.global code_drv_RET
.global drv_default_output
.global drv_err_mmap_fail
.global drv_err_open_fail
.global drv_forth_file
.global drv_forth_repl
.global drv_include_lens
.global drv_include_ptrs
.global drv_input_len
.global drv_input_path
.global drv_lithos_pipeline
.global drv_msg_mmap
.global drv_msg_open
.global drv_msg_wrote
.global drv_n_includes
.global drv_newline
.global drv_output_len
.global drv_output_path
.global drv_read_file_into
.global drv_run_xt_list
.global drv_src_addr
.global drv_src_len
.global drv_str_target
.global drv_target
.global drv_xt_list_return
.global xtl_cubin_params
.global xtl_get_code
.global xtl_get_tokens
.global xtl_lex
.global xtl_parse
.global xtl_write_arm64
.global xtl_write_cubin

// driver.s — Lithos compilation pipeline driver
//
// Ties together the lexer (lithos-lexer.s), parser (lithos-parser.s),
// emitters (emit-arm64.s / emit-sass.s), and ELF writer
// (lithos-elf-writer.s) into a complete .ls compilation pipeline.
//
// Companion to lithos-bootstrap.s. Shares the same DTC register
// conventions, dictionary entry layout, and macros.
//
// Build: assemble alongside lithos-bootstrap.s + lexer + parser + emitter
//        + elf-writer. The driver provides _start (the program entry),
//        replacing lithos-bootstrap.s's _start when this object file is
//        linked first.
//
//   as -o lithos-bootstrap.o lithos-bootstrap.s
//   as -o lithos-lexer.o     lithos-lexer.s
//   as -o lithos-parser.o    lithos-parser.s
//   as -o emit-arm64.o       emit-arm64.s
//   as -o lithos-elf-writer.o lithos-elf-writer.s
//   as -o driver.o           driver.s
//   ld -o lithos driver.o lithos-bootstrap.o lithos-lexer.o \
//                lithos-parser.o emit-arm64.o lithos-elf-writer.o
//
// Usage:
//   lithos compiler.ls --target arm64 -o output
//   lithos kernel.ls   --target gpu   -o output.cubin
//   lithos compiler.ls extra1.ls extra2.ls --target arm64 -o output
//   lithos boot.fs                                     # legacy Forth mode
//
// Dispatch:
//   argv[1] ends in ".ls" → Lithos pipeline (this driver)
//   argv[1] ends in ".fs" → Forth interpreter (jump to main_loop)
//   no argv[1]            → Forth REPL on stdin
//
// Pipeline stages (Lithos mode):
//   1. Parse argv: input file, --target {arm64|gpu}, -o output, extra .ls
//   2. mmap and read each source .ls file
//   3. Tokenize via LITHOS-LEX
//   4. Parse + emit via PARSE-TOKENS (calls emit-arm64 or emit-sass)
//   5. Write output via ELF-WRITE-ARM64 or ELF-WRITE-CUBIN

// ============================================================
// Externs from the rest of the bootstrap
// ============================================================
.extern _start_forth_main_loop  // alias for main_loop in lithos-bootstrap.s
                                // (see "Integration" comment at end of file)
.extern main_loop
.extern code_INTERPRET
.extern code_LITHOS_LEX           // ( src len -- )
.extern code_LEX_TOKENS           // ( -- token-buf )
.extern code_LEX_COUNT            // ( -- n )
.extern code_PARSE_TOKENS         // ( tok-buf tok-count src-buf -- )
.extern code_CODE_BUF             // ( -- addr )    emit-arm64
.extern code_CODE_POS             // ( -- u )       emit-arm64
.extern code_CODE_RESET           // ( -- )         emit-arm64
.extern code_ELF_WRITE_ARM64      // ( text-buf text-len data-buf data-len bss-len path -- )
.extern code_ELF_WRITE_CUBIN      // ( params-ptr path -- )
.extern code_CUBIN_PARAMS         // ( -- addr )

// Variables and buffers from lithos-bootstrap.s
.extern saved_argc
.extern saved_argv
.extern data_stack_top
.extern ret_stack_top
.extern mem_space
.extern var_state
.extern var_latest
.extern var_source_addr
.extern var_source_len
.extern var_to_in
.extern var_input_fd
.extern last_entry

// ============================================================
// Syscall numbers
// ============================================================
.equ SYS_READ,    63
.equ SYS_WRITE,   64
.equ SYS_OPENAT,  56
.equ SYS_CLOSE,   57
.equ SYS_LSEEK,   62
.equ SYS_EXIT,    93
.equ SYS_MMAP,    222
.equ AT_FDCWD,    -100

.equ SRC_BUF_SIZE, 16777216    // 16 MB per source file

// ============================================================
// DTC NEXT macro
// ============================================================
.ifndef NEXT_DEFINED
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.set NEXT_DEFINED, 1
.endif

// ============================================================
// Entry point — replaces lithos-bootstrap.s's _start
// ============================================================
.global _start
.text
.align 4

_start:
    // Save argc, argv
    ldr     x15, [sp]
    add     x16, sp, #8
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    str     x15, [x0]
    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    str     x16, [x0]

    // Initialize DTC stacks and dictionary pointer
    adrp    x24, data_stack_top
    add     x24, x24, :lo12:data_stack_top
    adrp    x23, ret_stack_top
    add     x23, x23, :lo12:ret_stack_top
    adrp    x20, mem_space
    add     x20, x20, :lo12:mem_space
    mov     x21, #16             // BASE = 16 (Lithos default)

    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    str     xzr, [x0]
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    adrp    x1, last_entry
    add     x1, x1, :lo12:last_entry
    str     x1, [x0]
    adrp    x0, var_source_addr
    add     x0, x0, :lo12:var_source_addr
    str     xzr, [x0]
    adrp    x0, var_source_len
    add     x0, x0, :lo12:var_source_len
    str     xzr, [x0]
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    str     xzr, [x0]
    mov     x22, #0

    // -------- Dispatch on file extension --------
    cmp     x15, #2
    b.lt    drv_forth_repl

    ldr     x1, [x16, #8]        // argv[1]
    mov     x2, x1
    mov     x3, #0
.Lstrlen1:
    ldrb    w4, [x2], #1
    cbz     w4, .Lstrlen1_done
    add     x3, x3, #1
    b       .Lstrlen1
.Lstrlen1_done:
    cmp     x3, #3
    b.lt    drv_forth_file       // too short to have an extension

    sub     x4, x3, #3
    add     x5, x1, x4
    ldrb    w6, [x5]             // expect '.'
    ldrb    w7, [x5, #1]         // 'l' or 'f'
    ldrb    w8, [x5, #2]         // 's'
    cmp     w6, #'.'
    b.ne    drv_forth_file
    cmp     w8, #'s'
    b.ne    drv_forth_file
    cmp     w7, #'l'
    b.eq    drv_lithos_pipeline
    // .fs (or anything ending ".?s" other than .ls) → Forth interpreter

drv_forth_file:
    // Open argv[1] as Forth input
    ldr     x1, [x16, #8]
    mov     x0, #AT_FDCWD
    mov     x2, #0
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    drv_forth_repl
    adrp    x1, var_input_fd
    add     x1, x1, :lo12:var_input_fd
    str     x0, [x1]
    b       main_loop

drv_forth_repl:
    adrp    x1, var_input_fd
    add     x1, x1, :lo12:var_input_fd
    str     xzr, [x1]
    b       main_loop

// ============================================================
// Lithos pipeline
// ============================================================
drv_lithos_pipeline:
    // Save input path (argv[1])
    adrp    x0, drv_input_path
    add     x0, x0, :lo12:drv_input_path
    str     x1, [x0]
    adrp    x0, drv_input_len
    add     x0, x0, :lo12:drv_input_len
    str     x3, [x0]

    // Defaults: target=arm64, output="a.out", no extra includes
    adrp    x0, drv_target
    add     x0, x0, :lo12:drv_target
    mov     x4, #1               // 1=arm64, 2=gpu
    str     x4, [x0]
    adrp    x0, drv_output_path
    add     x0, x0, :lo12:drv_output_path
    adrp    x4, drv_default_output
    add     x4, x4, :lo12:drv_default_output
    str     x4, [x0]
    adrp    x0, drv_output_len
    add     x0, x0, :lo12:drv_output_len
    mov     x4, #5               // strlen("a.out")
    str     x4, [x0]
    adrp    x0, drv_n_includes
    add     x0, x0, :lo12:drv_n_includes
    str     xzr, [x0]

    // Walk argv[2..argc-1]
    adrp    x9, saved_argv
    add     x9, x9, :lo12:saved_argv
    ldr     x9, [x9]
    mov     x10, #2

.Larg_loop:
    cmp     x10, x15
    b.ge    .Larg_done
    ldr     x0, [x9, x10, lsl #3]
    mov     x1, x0
    mov     x2, #0
.Larg_slen:
    ldrb    w3, [x1], #1
    cbz     w3, .Larg_slen_done
    add     x2, x2, #1
    b       .Larg_slen
.Larg_slen_done:
    // x0 = arg ptr, x2 = strlen

    // Match "--target"
    cmp     x2, #8
    b.ne    .Larg_try_o
    adrp    x3, drv_str_target
    add     x3, x3, :lo12:drv_str_target
    mov     x4, #0
.Lcmp_target:
    cmp     x4, #8
    b.ge    .Larg_is_target
    ldrb    w5, [x0, x4]
    ldrb    w6, [x3, x4]
    cmp     w5, w6
    b.ne    .Larg_try_o
    add     x4, x4, #1
    b       .Lcmp_target
.Larg_is_target:
    add     x10, x10, #1
    cmp     x10, x15
    b.ge    .Larg_done
    ldr     x0, [x9, x10, lsl #3]
    ldrb    w1, [x0]
    adrp    x3, drv_target
    add     x3, x3, :lo12:drv_target
    mov     x4, #1               // arm64 (default)
    cmp     w1, #'g'
    b.ne    1f
    mov     x4, #2               // gpu
1:  str     x4, [x3]
    b       .Larg_next

.Larg_try_o:
    // Match "-o"
    cmp     x2, #2
    b.ne    .Larg_try_src
    ldrb    w3, [x0]
    cmp     w3, #'-'
    b.ne    .Larg_try_src
    ldrb    w3, [x0, #1]
    cmp     w3, #'o'
    b.ne    .Larg_try_src
    add     x10, x10, #1
    cmp     x10, x15
    b.ge    .Larg_done
    ldr     x0, [x9, x10, lsl #3]
    mov     x1, x0
    mov     x2, #0
.Lo_slen:
    ldrb    w3, [x1], #1
    cbz     w3, .Lo_slen_done
    add     x2, x2, #1
    b       .Lo_slen
.Lo_slen_done:
    adrp    x3, drv_output_path
    add     x3, x3, :lo12:drv_output_path
    str     x0, [x3]
    adrp    x3, drv_output_len
    add     x3, x3, :lo12:drv_output_len
    str     x2, [x3]
    b       .Larg_next

.Larg_try_src:
    // Anything else: if it starts with '-', skip (unknown flag).
    // Otherwise treat as an additional .ls source file.
    ldrb    w3, [x0]
    cmp     w3, #'-'
    b.eq    .Larg_next
    adrp    x3, drv_n_includes
    add     x3, x3, :lo12:drv_n_includes
    ldr     x4, [x3]
    cmp     x4, #15              // hard cap: 16 sources total
    b.ge    .Larg_next
    adrp    x5, drv_include_ptrs
    add     x5, x5, :lo12:drv_include_ptrs
    str     x0, [x5, x4, lsl #3]
    adrp    x5, drv_include_lens
    add     x5, x5, :lo12:drv_include_lens
    str     x2, [x5, x4, lsl #3]
    add     x4, x4, #1
    str     x4, [x3]

.Larg_next:
    add     x10, x10, #1
    b       .Larg_loop

.Larg_done:
    // -------- Stage 2: Slurp the primary source --------
    // Slurp input.ls into a single contiguous mmap buffer.
    // Additional source files are appended (with a newline between)
    // so that the lexer sees one logical stream.
    adrp    x0, drv_src_addr
    add     x0, x0, :lo12:drv_src_addr
    str     xzr, [x0]
    adrp    x0, drv_src_len
    add     x0, x0, :lo12:drv_src_len
    str     xzr, [x0]

    // mmap a single large buffer for all sources
    mov     x0, #0
    mov     x1, #SRC_BUF_SIZE
    mov     x2, #3                  // PROT_READ|PROT_WRITE
    mov     x3, #0x22               // MAP_PRIVATE|MAP_ANONYMOUS
    mov     x4, #-1
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    drv_err_mmap_fail
    mov     x19, x0                 // x19 = base of source buffer
    adrp    x1, drv_src_addr
    add     x1, x1, :lo12:drv_src_addr
    str     x19, [x1]

    // Read primary input.ls into [x19, ..)
    adrp    x0, drv_input_path
    add     x0, x0, :lo12:drv_input_path
    ldr     x0, [x0]                // path ptr (null-terminated argv)
    mov     x1, x19
    bl      drv_read_file_into      // returns bytes-read in x0
    mov     x18, x0                 // x18 = total source bytes used

    // Append a newline separator
    mov     w0, #'\n'
    add     x1, x19, x18
    strb    w0, [x1]
    add     x18, x18, #1

    // Concatenate any additional .ls files
    adrp    x0, drv_n_includes
    add     x0, x0, :lo12:drv_n_includes
    ldr     x13, [x0]
    cbz     x13, .Lconcat_done
    mov     x14, #0
.Lconcat_loop:
    cmp     x14, x13
    b.ge    .Lconcat_done
    adrp    x0, drv_include_ptrs
    add     x0, x0, :lo12:drv_include_ptrs
    ldr     x0, [x0, x14, lsl #3]
    add     x1, x19, x18
    stp     x13, x14, [sp, #-16]!
    stp     x18, xzr, [sp, #-16]!
    bl      drv_read_file_into
    ldp     x18, xzr, [sp], #16
    ldp     x13, x14, [sp], #16
    add     x18, x18, x0
    mov     w0, #'\n'
    add     x1, x19, x18
    strb    w0, [x1]
    add     x18, x18, #1
    add     x14, x14, #1
    b       .Lconcat_loop
.Lconcat_done:

    adrp    x0, drv_src_len
    add     x0, x0, :lo12:drv_src_len
    str     x18, [x0]

    // -------- Stage 3-5: Run the pipeline via DTC trampoline --------
    // Build a small xt list in drv_pipeline_thread that:
    //   1. Pushes ( src len ) and calls LITHOS-LEX
    //   2. Calls LEX-TOKENS, LEX-COUNT to retrieve token buffer
    //   3. Pushes src and calls PARSE-TOKENS (which emits machine code)
    //   4. Calls CODE-BUF, CODE-POS to retrieve emitted code
    //   5. Pushes (text-buf text-len data-buf=0 data-len=0 bss-len=0 path)
    //      and calls ELF-WRITE-ARM64 (or ELF-WRITE-CUBIN for gpu)
    //   6. EXIT

    // Push src-addr, src-len onto data stack
    str     x22, [x24, #-8]!
    str     x19, [x24, #-8]!
    mov     x22, x18

    // Run: LITHOS-LEX  (consumes src len, populates lex_tokens)
    adrp    x0, xtl_lex
    add     x0, x0, :lo12:xtl_lex
    bl      drv_run_xt_list

    // Run: LEX-TOKENS LEX-COUNT  → leaves ( tok-buf tok-count ) on stack
    adrp    x0, xtl_get_tokens
    add     x0, x0, :lo12:xtl_get_tokens
    bl      drv_run_xt_list

    // Push src-addr (PARSE-TOKENS expects ( tok-buf tok-count src-buf ))
    str     x22, [x24, #-8]!
    mov     x22, x19

    // Run: PARSE-TOKENS
    adrp    x0, xtl_parse
    add     x0, x0, :lo12:xtl_parse
    bl      drv_run_xt_list

    // Code is now in emit-arm64's code_buf at code_pos bytes.
    // Run: CODE-BUF CODE-POS  → leaves ( text-buf text-len ) on stack
    adrp    x0, xtl_get_code
    add     x0, x0, :lo12:xtl_get_code
    bl      drv_run_xt_list

    // Branch on target for write
    adrp    x0, drv_target
    add     x0, x0, :lo12:drv_target
    ldr     x0, [x0]
    cmp     x0, #2
    b.eq    .Lwrite_cubin

    // ARM64: push (data-buf=0 data-len=0 bss-len path), then ELF-WRITE-ARM64
    // bss-len: sum all buf declarations from the symbol table.
    // Walk ls_sym_table[0..ls_sym_count), accumulate sizes for KIND_BUF entries.
    str     x22, [x24, #-8]!     // text-len → NOS
    mov     x22, #0              // data-buf
    str     x22, [x24, #-8]!
    str     x22, [x24, #-8]!     // data-len

    // BSS size is the running total kept in ls_bss_offset, which
    // parse_buf_decl advances on every `buf NAME SIZE` declaration.
    // (The sym table's SYM_REG slot holds each buf's *offset*, not its
    // size, so walking the sym table here is wrong — use the counter.)
    adrp    x0, ls_bss_offset
    add     x0, x0, :lo12:ls_bss_offset
    ldr     x3, [x0]             // x3 = bss_total
    // Align BSS to 16 bytes
    add     x3, x3, #15
    bic     x3, x3, #15
    mov     x22, x3
    str     x22, [x24, #-8]!     // bss-len
    adrp    x0, drv_output_path
    add     x0, x0, :lo12:drv_output_path
    ldr     x22, [x0]            // path

    adrp    x0, xtl_write_arm64
    add     x0, x0, :lo12:xtl_write_arm64
    bl      drv_run_xt_list
    b       .Lpipeline_done

.Lwrite_cubin:
    // GPU: write text-buf and text-len into the cubin parameter block,
    // then call ELF-WRITE-CUBIN with (params-ptr, path).
    // For now, just push (params, path) — assume cubin_params has been
    // populated by emit-sass during parse.
    // Drop the (text-buf text-len) we previously pushed.
    ldr     x22, [x24], #8       // drop text-len
    ldr     x22, [x24], #8       // drop text-buf (now garbage TOS)

    // Run CUBIN-PARAMS to get params pointer
    adrp    x0, xtl_cubin_params
    add     x0, x0, :lo12:xtl_cubin_params
    bl      drv_run_xt_list

    // Push output path
    str     x22, [x24, #-8]!
    adrp    x0, drv_output_path
    add     x0, x0, :lo12:drv_output_path
    ldr     x22, [x0]

    adrp    x0, xtl_write_cubin
    add     x0, x0, :lo12:xtl_write_cubin
    bl      drv_run_xt_list

.Lpipeline_done:
    // Print "lithos: wrote <path>\n"
    adrp    x1, drv_msg_wrote
    add     x1, x1, :lo12:drv_msg_wrote
    mov     x2, #14              // "lithos: wrote "
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x0, drv_output_path
    add     x0, x0, :lo12:drv_output_path
    ldr     x1, [x0]
    adrp    x0, drv_output_len
    add     x0, x0, :lo12:drv_output_len
    ldr     x2, [x0]
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, drv_newline
    add     x1, x1, :lo12:drv_newline
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0

    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// drv_run_xt_list — execute a NULL-terminated xt list as DTC code
// ============================================================
//
// Entry: x0 = pointer to a list of CFAs ending with cfa_drv_RET.
//        Forth data-stack state: as required by the words being run.
// Exit: returns to caller as a normal subroutine.
//
// The trick: cfa_drv_RET's body is a tiny native stub that pops
// the saved x30 from the C-stack and `ret`s. Combined with the
// preserved DTC IP/return-stack, this acts as a clean trampoline.

.align 4
drv_run_xt_list:
    stp     x29, x30, [sp, #-16]!
    str     x26, [sp, #-16]!     // save DTC IP
    mov     x26, x0              // IP = our xt list
    NEXT                          // dispatch first xt

// Re-entry point after xtl_RET fires
.global drv_xt_list_return
drv_xt_list_return:
    ldr     x26, [sp], #16       // restore DTC IP
    ldp     x29, x30, [sp], #16
    ret

// drv_RET — terminates an xt list and returns to drv_run_xt_list's caller
.align 4
code_drv_RET:
    b       drv_xt_list_return

// ============================================================
// drv_read_file_into(x0 = path ptr, x1 = dest buffer)
// Returns bytes-read in x0. Aborts on error.
// ============================================================
.align 4
drv_read_file_into:
    stp     x29, x30, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    mov     x19, x1                 // save dest

    // open(path, O_RDONLY)
    mov     x1, x0
    mov     x0, #AT_FDCWD
    mov     x2, #0
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    drv_err_open_fail
    mov     x20, x0                 // fd

    // lseek(fd, 0, SEEK_END) → size
    mov     x0, x20
    mov     x1, #0
    mov     x2, #2
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x9, x0                  // file size

    // lseek(fd, 0, SEEK_SET)
    mov     x0, x20
    mov     x1, #0
    mov     x2, #0
    mov     x8, #SYS_LSEEK
    svc     #0

    // read(fd, dest, size)
    mov     x0, x20
    mov     x1, x19
    mov     x2, x9
    mov     x8, #SYS_READ
    svc     #0
    mov     x10, x0                 // bytes read

    // close(fd)
    mov     x0, x20
    mov     x8, #SYS_CLOSE
    svc     #0

    mov     x0, x10
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Error handlers
// ============================================================
drv_err_mmap_fail:
    adrp    x1, drv_msg_mmap
    add     x1, x1, :lo12:drv_msg_mmap
    mov     x2, #28
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

drv_err_open_fail:
    adrp    x1, drv_msg_open
    add     x1, x1, :lo12:drv_msg_open
    mov     x2, #28
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// .data — strings, paths, runtime state
// ============================================================
.data
.align 3

drv_default_output:   .ascii "a.out"
                      .byte 0
drv_str_target:       .ascii "--target"
drv_newline:          .byte 10
.align 3

drv_msg_wrote:        .ascii "lithos: wrote "
drv_msg_open:         .ascii "lithos: cannot open source\n"
drv_msg_mmap:         .ascii "lithos: cannot mmap buffer\n"
.align 3

// Driver runtime state
drv_input_path:       .quad 0       // → argv string (null-terminated)
drv_input_len:        .quad 0
drv_output_path:      .quad 0
drv_output_len:       .quad 0
drv_target:           .quad 1       // 1=arm64 2=gpu
drv_src_addr:         .quad 0       // mmap'd source buffer base
drv_src_len:          .quad 0       // total source bytes (incl. concatenated)

// Additional .ls source files passed on the command line
.equ DRV_MAX_INCLUDES, 16
drv_n_includes:       .quad 0
drv_include_ptrs:     .space (DRV_MAX_INCLUDES * 8)
drv_include_lens:     .space (DRV_MAX_INCLUDES * 8)

// ============================================================
// XT lists — DTC sequences invoked from native code
// ============================================================
//
// Each list is a sequence of CFA pointers. drv_run_xt_list runs
// them via NEXT, terminated by cfa_drv_RET which returns to native.
.align 3

cfa_drv_RET:              .quad code_drv_RET

// CFA cells — each holds a pointer to native code. The "execution token"
// (xt) placed in an xtl_* list is the ADDRESS of the CFA cell, not the
// native entry point itself, because NEXT dereferences twice.
cfa_LITHOS_LEX:           .quad code_LITHOS_LEX
cfa_LEX_TOKENS:           .quad code_LEX_TOKENS
cfa_LEX_COUNT:            .quad code_LEX_COUNT
cfa_PARSE_TOKENS:         .quad code_PARSE_TOKENS
cfa_CODE_BUF:             .quad code_CODE_BUF
cfa_CODE_POS:             .quad code_CODE_POS
cfa_ELF_WRITE_ARM64:      .quad code_ELF_WRITE_ARM64
cfa_ELF_WRITE_CUBIN:      .quad code_ELF_WRITE_CUBIN
cfa_CUBIN_PARAMS:         .quad code_CUBIN_PARAMS

// LITHOS-LEX  ( src len -- )
xtl_lex:
    .quad   cfa_LITHOS_LEX
    .quad   cfa_drv_RET

// LEX-TOKENS LEX-COUNT  ( -- tok-buf n )
xtl_get_tokens:
    .quad   cfa_LEX_TOKENS
    .quad   cfa_LEX_COUNT
    .quad   cfa_drv_RET

// PARSE-TOKENS  ( tok-buf n src -- )
xtl_parse:
    .quad   cfa_PARSE_TOKENS
    .quad   cfa_drv_RET

// CODE-BUF CODE-POS  ( -- text-buf text-len )
xtl_get_code:
    .quad   cfa_CODE_BUF
    .quad   cfa_CODE_POS
    .quad   cfa_drv_RET

// ELF-WRITE-ARM64  ( text-buf text-len 0 0 0 path -- )
xtl_write_arm64:
    .quad   cfa_ELF_WRITE_ARM64
    .quad   cfa_drv_RET

// ELF-WRITE-CUBIN  ( params path -- )
xtl_write_cubin:
    .quad   cfa_ELF_WRITE_CUBIN
    .quad   cfa_drv_RET

// CUBIN-PARAMS  ( -- addr )
xtl_cubin_params:
    .quad   cfa_CUBIN_PARAMS
    .quad   cfa_drv_RET

// ============================================================
// Integration notes
// ============================================================
//
// 1. _start in lithos-bootstrap.s must be removed (or made weak)
//    so this driver's _start wins at link time. Symbols `main_loop`
//    and the variables it references stay in lithos-bootstrap.s.
//
// 2. The component object files (lithos-lexer.o, lithos-parser.o,
//    emit-arm64.o, lithos-elf-writer.o) must be linked together.
//    Their `last_entry` chains should be spliced in this order:
//
//        last_entry → entry_lex_token_fetch (lexer)
//        lexer chain ends → entry_pad (bootstrap)
//        + parser, emit-arm64, elf-writer entries linked similarly
//
//    For the driver's purposes, only the `code_*` entry points
//    are referenced (not dictionary entries), so the dictionary
//    chain is irrelevant to this file.
//
// 3. Once the .li self-hosting compiler exists (TODO S2/S3), the
//    PARSE-TOKENS word will be replaced by a native implementation
//    written in .li and compiled by this very driver — closing
//    the bootstrap loop.

.end
