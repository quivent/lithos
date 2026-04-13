// lithos-bootstrap.s — Lithos language bootstrap for ARM64 Linux
//
// Purpose: Host the Lithos compiler. Provides the 87 words the compiler
//          actually uses, nothing more. Derived from the same DTC pattern
//          as forth-bootstrap.s, but this is Lithos's own foundation.
//
// Build: as -o lithos-bootstrap.o lithos-bootstrap.s
//        ld -o lithos-bootstrap lithos-bootstrap.o
//
// Threading: Direct Threaded Code (DTC)
// Register map:
//   X26 = IP   (instruction pointer — next CFA to execute)
//   X25 = W    (working register — current CFA)
//   X24 = DSP  (data stack pointer, full descending)
//   X23 = RSP  (return stack pointer, full descending)
//   X22 = TOS  (top of data stack, cached in register)
//   X20 = HERE (dictionary/data-space pointer)
//   X21 = BASE (number base — 16 by default for Lithos)
//
// Dictionary entry layout:
//   [link]   8 bytes — pointer to previous entry (0 = end)
//   [flags]  1 byte  — bit 7 = immediate, bit 6 = hidden
//   [nlen]   1 byte  — name length (max 31)
//   [name]   nlen bytes, padded to 8-byte alignment
//   [CFA]    8 bytes — code field: pointer to machine code
//   [PFA]    ...     — parameter field

.equ SYS_READ,   63
.equ SYS_WRITE,  64
.equ SYS_OPENAT, 56
.equ SYS_CLOSE,  57
.equ SYS_LSEEK,  62
.equ SYS_EXIT,   93
.equ SYS_BRK,    214
.equ SYS_MMAP,   222
.equ SYS_MUNMAP, 215

.equ AT_FDCWD,   -100

.equ DSTACK_SIZE, 8388608
.equ RSTACK_SIZE, 8388608
.equ MEM_SIZE,    536870912
.equ TIB_SIZE,    4096
.equ WORD_BUF_SIZE, 256
.equ PAD_SIZE,    256
.equ FILE_BUF_SIZE, 524288

.equ F_IMMEDIATE, 0x80
.equ F_HIDDEN,    0x40

// ============================================================
// Macros
// ============================================================

.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm

.macro PUSH reg
    str     x22, [x24, #-8]!
    mov     x22, \reg
.endm

.macro POP reg
    mov     \reg, x22
    ldr     x22, [x24], #8
.endm

.macro RPUSH reg
    str     \reg, [x23, #-8]!
.endm

.macro RPOP reg
    ldr     \reg, [x23], #8
.endm

// ============================================================
// Entry point
// ============================================================

.global _start
.text
.align 4

_start:
    ldr     x15, [sp]
    add     x16, sp, #8
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    str     x15, [x0]
    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    str     x16, [x0]

    adrp    x24, data_stack_top
    add     x24, x24, :lo12:data_stack_top
    adrp    x23, ret_stack_top
    add     x23, x23, :lo12:ret_stack_top
    adrp    x20, mem_space
    add     x20, x20, :lo12:mem_space
    mov     x21, #16             // Lithos defaults to hex

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

    cmp     x15, #2
    b.lt    1f
    ldr     x1, [x16, #8]
    mov     x0, #AT_FDCWD
    mov     x2, #0
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    1f
    adrp    x1, var_input_fd
    add     x1, x1, :lo12:var_input_fd
    str     x0, [x1]
    b       2f
1:
    adrp    x1, var_input_fd
    add     x1, x1, :lo12:var_input_fd
    str     xzr, [x1]
2:
    b       main_loop

// ============================================================
// DOCOL — enter colon definition
// ============================================================
.align 4
code_DOCOL:
    RPUSH   x26
    add     x26, x25, #8
    NEXT

// ============================================================
// Core threaded-code primitives
// ============================================================

.align 4
code_EXIT:
    RPOP    x26
    NEXT

.align 4
code_LIT:
    str     x22, [x24, #-8]!
    ldr     x22, [x26], #8
    NEXT

.align 4
code_LATEBIND:
    ldrb    w0, [x26]
    add     x1, x26, #1
    add     x26, x26, x0
    add     x26, x26, #1
    add     x26, x26, #7
    and     x26, x26, #~7
    stp     x26, x30, [sp, #-16]!
    mov     x2, x0
    mov     x0, x1
    mov     x1, x2
    stp     x0, x1, [sp, #-16]!
    bl      do_find_word
    cbz     x0, latebind_fail
    add     sp, sp, #16
    ldp     x26, x30, [sp], #16
    mov     x25, x0
    ldr     x16, [x25]
    br      x16

latebind_fail:
    ldp     x1, x2, [sp], #16
    ldp     x26, x30, [sp], #16
    stp     x1, x2, [sp, #-16]!
    adrp    x1, err_latebind
    add     x1, x1, :lo12:err_latebind
    mov     x2, #12
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x1, x2, [sp], #16
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, newline
    add     x1, x1, :lo12:newline
    mov     x2, #1
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.align 4
code_LITSTRING:
    str     x22, [x24, #-8]!
    ldr     x0, [x26], #8
    str     x26, [x24, #-8]!
    mov     x22, x0
    add     x26, x26, x0
    add     x26, x26, #7
    and     x26, x26, #~7
    NEXT

.align 4
code_BRANCH:
    ldr     x0, [x26]
    add     x26, x26, x0
    NEXT

.align 4
code_0BRANCH:
    ldr     x0, [x26], #8
    cmp     x22, #0
    ldr     x22, [x24], #8
    b.eq    1f
    NEXT
1:  sub     x26, x26, #8
    ldr     x0, [x26]
    add     x26, x26, x0
    NEXT

// ============================================================
// Arithmetic
// ============================================================

.align 4
code_PLUS:
    ldr     x0, [x24], #8
    add     x22, x0, x22
    NEXT

.align 4
code_MINUS:
    ldr     x0, [x24], #8
    sub     x22, x0, x22
    NEXT

.align 4
code_STAR:
    ldr     x0, [x24], #8
    mul     x22, x0, x22
    NEXT

.align 4
code_SLASH:
    ldr     x0, [x24], #8
    sdiv    x22, x0, x22
    NEXT

.align 4
code_MOD:
    ldr     x0, [x24], #8
    sdiv    x1, x0, x22
    msub    x22, x1, x22, x0
    NEXT

.align 4
code_AND:
    ldr     x0, [x24], #8
    and     x22, x0, x22
    NEXT

.align 4
code_OR:
    ldr     x0, [x24], #8
    orr     x22, x0, x22
    NEXT

.align 4
code_XOR:
    ldr     x0, [x24], #8
    eor     x22, x0, x22
    NEXT

.align 4
code_LSHIFT:
    ldr     x0, [x24], #8
    lsl     x22, x0, x22
    NEXT

.align 4
code_RSHIFT:
    ldr     x0, [x24], #8
    lsr     x22, x0, x22
    NEXT

.align 4
code_NEGATE:
    neg     x22, x22
    NEXT

.align 4
code_INVERT:
    mvn     x22, x22
    NEXT

.align 4
code_1PLUS:
    add     x22, x22, #1
    NEXT

.align 4
code_1MINUS:
    sub     x22, x22, #1
    NEXT

.align 4
code_2STAR:
    lsl     x22, x22, #1
    NEXT

.align 4
code_CELLS:
    lsl     x22, x22, #3
    NEXT

// ============================================================
// Stack
// ============================================================

.align 4
code_DUP:
    str     x22, [x24, #-8]!
    NEXT

.align 4
code_DROP:
    ldr     x22, [x24], #8
    NEXT

.align 4
code_SWAP:
    ldr     x0, [x24]
    str     x22, [x24]
    mov     x22, x0
    NEXT

.align 4
code_OVER:
    str     x22, [x24, #-8]!
    ldr     x22, [x24, #8]
    NEXT

.align 4
code_ROT:
    ldr     x0, [x24]
    ldr     x1, [x24, #8]
    str     x0, [x24, #8]
    str     x22, [x24]
    mov     x22, x1
    NEXT

.align 4
code_NROT:
    ldr     x0, [x24]
    ldr     x1, [x24, #8]
    str     x22, [x24, #8]
    str     x1, [x24]
    mov     x22, x0
    NEXT

.align 4
code_NIP:
    add     x24, x24, #8
    NEXT

.align 4
code_PICK:
    ldr     x22, [x24, x22, lsl #3]
    NEXT

.align 4
code_DEPTH:
    str     x22, [x24, #-8]!
    adrp    x0, data_stack_top
    add     x0, x0, :lo12:data_stack_top
    sub     x22, x0, x24
    asr     x22, x22, #3
    NEXT

.align 4
code_2DUP:
    ldr     x0, [x24]
    str     x22, [x24, #-8]!
    str     x0, [x24, #-8]!
    NEXT

.align 4
code_2DROP:
    ldr     x22, [x24, #8]
    add     x24, x24, #16
    NEXT

.align 4
code_2SWAP:
    ldr     x0, [x24]
    ldr     x1, [x24, #8]
    ldr     x2, [x24, #16]
    str     x22, [x24, #8]
    str     x0, [x24, #16]
    mov     x22, x1
    str     x2, [x24]
    NEXT

// ============================================================
// Return stack
// ============================================================

.align 4
code_TOR:
    RPUSH   x22
    ldr     x22, [x24], #8
    NEXT

.align 4
code_RFROM:
    str     x22, [x24, #-8]!
    RPOP    x22
    NEXT

.align 4
code_RFETCH:
    str     x22, [x24, #-8]!
    ldr     x22, [x23]
    NEXT

// ============================================================
// Memory
// ============================================================

.align 4
code_FETCH:
    ldr     x22, [x22]
    NEXT

.align 4
code_STORE:
    ldr     x0, [x24], #8
    str     x0, [x22]
    ldr     x22, [x24], #8
    NEXT

.align 4
code_CFETCH:
    ldrb    w22, [x22]
    NEXT

.align 4
code_CSTORE:
    ldr     x0, [x24], #8
    strb    w0, [x22]
    ldr     x22, [x24], #8
    NEXT

.align 4
code_PLUSSTORE:
    ldr     x0, [x24], #8
    ldr     x1, [x22]
    add     x1, x1, x0
    str     x1, [x22]
    ldr     x22, [x24], #8
    NEXT

.align 4
code_MOVE:
    POP     x2
    POP     x1
    POP     x0
    cmp     x2, #0
    b.le    2f
1:  ldrb    w3, [x0], #1
    strb    w3, [x1], #1
    subs    x2, x2, #1
    b.ne    1b
2:  NEXT

.align 4
code_FILL:
    POP     x2          // char
    POP     x1          // u
    POP     x0          // addr
    cmp     x1, #0
    b.le    2f
1:  strb    w2, [x0], #1
    subs    x1, x1, #1
    b.ne    1b
2:  NEXT

.align 4
code_HERE:
    str     x22, [x24, #-8]!
    mov     x22, x20
    NEXT

.align 4
code_ALLOT:
    add     x20, x20, x22
    ldr     x22, [x24], #8
    NEXT

.align 4
code_ALIGN:
    add     x20, x20, #7
    and     x20, x20, #~7
    NEXT

.align 4
code_COMMA:
    str     x22, [x20], #8
    ldr     x22, [x24], #8
    NEXT

.align 4
code_CCOMMA:
    strb    w22, [x20], #1
    ldr     x22, [x24], #8
    NEXT

// ============================================================
// Comparison
// ============================================================

.align 4
code_EQ:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csetm   x22, eq
    NEXT

.align 4
code_NEQ:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csetm   x22, ne
    NEXT

.align 4
code_LT:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csetm   x22, lt
    NEXT

.align 4
code_GT:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csetm   x22, gt
    NEXT

.align 4
code_ZEROEQ:
    cmp     x22, #0
    csetm   x22, eq
    NEXT

.align 4
code_ZEROGT:
    cmp     x22, #0
    csetm   x22, gt
    NEXT

.align 4
code_ZEROLT:
    cmp     x22, #0
    csetm   x22, lt
    NEXT

.align 4
code_ZERONE:
    cmp     x22, #0
    csetm   x22, ne
    NEXT

.align 4
code_MIN:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csel    x22, x0, x22, lt
    NEXT

.align 4
code_MAX:
    ldr     x0, [x24], #8
    cmp     x0, x22
    csel    x22, x0, x22, gt
    NEXT

.align 4
code_WITHIN:
    // ( n lo hi -- flag ) true if lo <= n < hi
    // TOS=hi, [sp]=lo, [sp+8]=n
    POP     x2          // hi
    POP     x1          // lo
    POP     x0          // n
    sub     x3, x0, x1
    sub     x4, x2, x1
    cmp     x3, x4
    csetm   x22, lo
    NEXT

// ============================================================
// I/O
// ============================================================

.align 4
code_EMIT:
    adrp    x0, emit_buf
    add     x0, x0, :lo12:emit_buf
    strb    w22, [x0]
    mov     x2, #1
    mov     x1, x0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x22, [x24], #8
    NEXT

.align 4
code_TYPE:
    POP     x2
    POP     x1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    NEXT

.align 4
code_CR:
    adrp    x0, newline
    add     x0, x0, :lo12:newline
    mov     x1, x0
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    NEXT

.align 4
code_SPACE:
    adrp    x1, space_char
    add     x1, x1, :lo12:space_char
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    NEXT

.align 4
code_DOT:
    mov     x0, x22
    ldr     x22, [x24], #8
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    adrp    x4, num_buf
    add     x4, x4, :lo12:num_buf
    add     x4, x4, #32
    mov     x5, x4
    mov     x6, #0
    cmp     x0, #0
    b.ge    2f
    mov     x6, #1
    neg     x0, x0
2:  mov     x1, x21
3:  udiv    x2, x0, x1
    msub    x3, x2, x1, x0
    cmp     x3, #10
    b.lt    4f
    add     x3, x3, #('a' - 10)
    b       5f
4:  add     x3, x3, #'0'
5:  sub     x4, x4, #1
    strb    w3, [x4]
    mov     x0, x2
    cbnz    x0, 3b
    cbz     x6, 6f
    sub     x4, x4, #1
    mov     x3, #'-'
    strb    w3, [x4]
6:  sub     x2, x5, x4
    mov     x1, x4
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, space_char
    add     x1, x1, :lo12:space_char
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

.align 4
code_UDOT:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    mov     x0, x22
    ldr     x22, [x24], #8
    adrp    x4, num_buf
    add     x4, x4, :lo12:num_buf
    add     x4, x4, #32
    mov     x5, x4
    mov     x1, x21
1:  udiv    x2, x0, x1
    msub    x3, x2, x1, x0
    cmp     x3, #10
    b.lt    2f
    add     x3, x3, #('a' - 10)
    b       3f
2:  add     x3, x3, #'0'
3:  sub     x4, x4, #1
    strb    w3, [x4]
    mov     x0, x2
    cbnz    x0, 1b
    sub     x2, x5, x4
    mov     x1, x4
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, space_char
    add     x1, x1, :lo12:space_char
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

.align 4
code_DOTS:
    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    adrp    x8, data_stack_top
    add     x8, x8, :lo12:data_stack_top
    mov     x9, x8
2:  sub     x9, x9, #8
    cmp     x9, x24
    b.lo    3f
    ldr     x0, [x9]
    stp     x8, x9, [sp, #-16]!
    bl      print_number
    ldp     x8, x9, [sp], #16
    b       2b
3:  mov     x0, x22
    bl      print_number
    adrp    x1, newline
    add     x1, x1, :lo12:newline
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

print_number:
    stp     x30, xzr, [sp, #-16]!
    mov     x10, x0
    adrp    x4, num_buf
    add     x4, x4, :lo12:num_buf
    add     x4, x4, #32
    mov     x5, x4
    mov     x6, #0
    cmp     x10, #0
    b.ge    1f
    mov     x6, #1
    neg     x10, x10
1:  mov     x1, #10
2:  udiv    x2, x10, x1
    msub    x3, x2, x1, x10
    add     x3, x3, #'0'
    sub     x4, x4, #1
    strb    w3, [x4]
    mov     x10, x2
    cbnz    x10, 2b
    cbz     x6, 3f
    sub     x4, x4, #1
    mov     x3, #'-'
    strb    w3, [x4]
3:  sub     x2, x5, x4
    mov     x1, x4
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, space_char
    add     x1, x1, :lo12:space_char
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// File I/O
// ============================================================

.align 4
code_OPEN_FILE:
    POP     x3
    POP     x2
    POP     x1
    adrp    x4, pad_buf
    add     x4, x4, :lo12:pad_buf
    mov     x5, x1
    mov     x6, x2
    cbz     x6, 2f
1:  ldrb    w7, [x5], #1
    strb    w7, [x4], #1
    subs    x6, x6, #1
    b.ne    1b
2:  strb    wzr, [x4]
    adrp    x1, pad_buf
    add     x1, x1, :lo12:pad_buf
    mov     x0, #AT_FDCWD
    mov     x2, x3
    mov     x3, #0644
    movk    x3, #0, lsl #16
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    3f
    str     x0, [x24, #-8]!
    mov     x22, #0
    NEXT
3:  neg     x0, x0
    str     xzr, [x24, #-8]!
    mov     x22, x0
    NEXT

.align 4
code_CLOSE_FILE:
    mov     x0, x22
    mov     x8, #SYS_CLOSE
    svc     #0
    cmp     x0, #0
    b.lt    1f
    mov     x22, #0
    NEXT
1:  neg     x22, x0
    NEXT

.align 4
code_READ_FILE:
    POP     x0
    POP     x2
    POP     x1
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #0
    b.lt    1f
    str     x0, [x24, #-8]!
    mov     x22, #0
    NEXT
1:  neg     x1, x0
    str     xzr, [x24, #-8]!
    mov     x22, x1
    NEXT

.align 4
code_WRITE_FILE:
    POP     x0
    POP     x2
    POP     x1
    mov     x8, #SYS_WRITE
    svc     #0
    cmp     x0, #0
    b.lt    1f
    mov     x22, #0
    NEXT
1:  neg     x22, x0
    NEXT

.align 4
code_RO:
    str     x22, [x24, #-8]!
    mov     x22, #0
    NEXT

.align 4
code_WO:
    str     x22, [x24, #-8]!
    mov     x22, #0x241
    NEXT

// ============================================================
// String
// ============================================================

.align 4
code_COUNT:
    ldrb    w0, [x22]
    add     x22, x22, #1
    str     x22, [x24, #-8]!
    mov     x22, x0
    NEXT

.align 4
code_COMPARE:
    // compare ( addr1 u1 addr2 u2 -- n )
    // FIXED: the original forth-bootstrap had stack consumption bugs.
    // This version is clean.
    POP     x3          // u2
    POP     x2          // addr2
    POP     x1          // u1
    POP     x0          // addr1
    cmp     x1, x3
    csel    x4, x1, x3, lt
    mov     x5, #0
    b       2f
1:  ldrb    w6, [x0, x5]
    ldrb    w7, [x2, x5]
    cmp     w6, w7
    b.lt    cmp_lt
    b.gt    cmp_gt
    add     x5, x5, #1
2:  cmp     x5, x4
    b.lt    1b
    cmp     x1, x3
    b.lt    cmp_lt
    b.gt    cmp_gt
    mov     x22, #0
    NEXT
cmp_lt:
    mov     x22, #-1
    NEXT
cmp_gt:
    mov     x22, #1
    NEXT

// ============================================================
// Parsing helpers (called from code, not threaded)
// ============================================================

do_parse_name:
    stp     x30, xzr, [sp, #-16]!
    adrp    x10, var_source_addr
    add     x10, x10, :lo12:var_source_addr
    ldr     x10, [x10]
    adrp    x11, var_source_len
    add     x11, x11, :lo12:var_source_len
    ldr     x11, [x11]
    adrp    x12, var_to_in
    add     x12, x12, :lo12:var_to_in
    ldr     x13, [x12]
1:  cmp     x13, x11
    b.ge    3f
    ldrb    w14, [x10, x13]
    cmp     w14, #32
    b.gt    2f
    add     x13, x13, #1
    b       1b
2:  add     x0, x10, x13
    mov     x1, #0
4:  cmp     x13, x11
    b.ge    5f
    ldrb    w14, [x10, x13]
    cmp     w14, #32
    b.le    5f
    add     x13, x13, #1
    add     x1, x1, #1
    b       4b
5:  str     x13, [x12]
    ldp     x30, xzr, [sp], #16
    ret
3:  str     x13, [x12]
    mov     x0, #0
    mov     x1, #0
    ldp     x30, xzr, [sp], #16
    ret

do_parse:
    stp     x30, xzr, [sp, #-16]!
    adrp    x10, var_source_addr
    add     x10, x10, :lo12:var_source_addr
    ldr     x10, [x10]
    adrp    x11, var_source_len
    add     x11, x11, :lo12:var_source_len
    ldr     x11, [x11]
    adrp    x12, var_to_in
    add     x12, x12, :lo12:var_to_in
    ldr     x13, [x12]
    add     x0, x10, x13
    mov     x1, #0
1:  cmp     x13, x11
    b.ge    2f
    ldrb    w14, [x10, x13]
    add     x13, x13, #1
    cmp     w14, w15
    b.eq    2f
    add     x1, x1, #1
    b       1b
2:  str     x13, [x12]
    ldp     x30, xzr, [sp], #16
    ret

do_find_word:
    stp     x30, xzr, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    mov     x2, x0
    mov     x3, x1
    adrp    x4, var_latest
    add     x4, x4, :lo12:var_latest
    ldr     x4, [x4]
1:  cbz     x4, 4f
    ldrb    w5, [x4, #8]
    tbnz    w5, #6, 3f
    ldrb    w6, [x4, #9]
    cmp     x3, x6
    b.ne    3f
    mov     x7, #0
2:  cmp     x7, x3
    b.ge    5f
    ldrb    w8, [x2, x7]
    add     x9, x4, #10
    ldrb    w10, [x9, x7]
    cmp     w8, #'A'
    b.lt    6f
    cmp     w8, #'Z'
    b.gt    6f
    orr     w8, w8, #0x20
6:  cmp     w10, #'A'
    b.lt    7f
    cmp     w10, #'Z'
    b.gt    7f
    orr     w10, w10, #0x20
7:  cmp     w8, w10
    b.ne    3f
    add     x7, x7, #1
    b       2b
5:  add     x0, x4, #10
    add     x0, x0, x6
    add     x0, x0, #7
    and     x0, x0, #~7
    ldrb    w1, [x4, #8]
    ldp     x19, x20, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret
3:  ldr     x4, [x4]
    b       1b
4:  mov     x0, #0
    mov     x1, #0
    ldp     x19, x20, [sp], #16
    ldp     x30, xzr, [sp], #16
    ret

do_number:
    stp     x30, xzr, [sp, #-16]!
    mov     x2, x0
    mov     x3, x1
    cbz     x3, num_fail
    mov     x4, #0
    mov     x5, #0
    ldrb    w6, [x2]
    cmp     w6, #'-'
    b.ne    1f
    mov     x4, #1
    add     x2, x2, #1
    sub     x3, x3, #1
    cbz     x3, num_fail
1:  ldrb    w6, [x2]
    cmp     w6, #'$'
    b.eq    hex_num
    cmp     w6, #'0'
    b.ne    dec_loop
    cmp     x3, #2
    b.lt    dec_loop
    ldrb    w7, [x2, #1]
    cmp     w7, #'x'
    b.eq    hex_0x
    cmp     w7, #'X'
    b.eq    hex_0x
    b       dec_loop
hex_num:
    add     x2, x2, #1
    sub     x3, x3, #1
    cbz     x3, num_fail
hex_loop:
    ldrb    w6, [x2]
    sub     w7, w6, #'0'
    cmp     w7, #9
    b.ls    hex_digit
    orr     w6, w6, #0x20
    sub     w7, w6, #'a'
    cmp     w7, #5
    b.hi    num_fail
    add     w7, w7, #10
hex_digit:
    lsl     x5, x5, #4
    add     x5, x5, x7
    add     x2, x2, #1
    subs    x3, x3, #1
    b.ne    hex_loop
    b       num_done
hex_0x:
    add     x2, x2, #2
    sub     x3, x3, #2
    cbz     x3, num_fail
    b       hex_loop
dec_loop:
    ldrb    w6, [x2]
    sub     w7, w6, #'0'
    cmp     w7, #9
    b.hi    num_fail
    mov     x8, #10
    mul     x5, x5, x8
    add     x5, x5, x7
    add     x2, x2, #1
    subs    x3, x3, #1
    b.ne    dec_loop
num_done:
    cbz     x4, 2f
    neg     x5, x5
2:  mov     x0, x5
    mov     x1, #1
    ldp     x30, xzr, [sp], #16
    ret
num_fail:
    mov     x0, #0
    mov     x1, #0
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Outer interpreter
// ============================================================

.align 4
code_INTERPRET:
    stp     x30, xzr, [sp, #-16]!

interp_loop:
    bl      do_parse_name
    cbz     x1, interp_done
    stp     x0, x1, [sp, #-16]!
    bl      do_find_word
    cmp     x0, #0
    b.eq    try_number
    ldr     x2, [sp]
    add     sp, sp, #16
    adrp    x3, var_state
    add     x3, x3, :lo12:var_state
    ldr     x3, [x3]
    cbz     x3, exec_word
    tst     x1, #F_IMMEDIATE
    b.ne    exec_word
    str     x0, [x20], #8
    b       interp_loop

exec_word:
    adrp    x1, exec_pad
    add     x1, x1, :lo12:exec_pad
    str     x0, [x1]
    adrp    x2, interp_return_xt
    add     x2, x2, :lo12:interp_return_xt
    str     x2, [x1, #8]
    RPUSH   x26
    mov     x26, x1
    NEXT

.align 4
interp_return_xt:
    .quad   interp_return_code
interp_return_code:
    RPOP    x26
    b       interp_loop

try_number:
    ldp     x0, x1, [sp], #16
    stp     x0, x1, [sp, #-16]!
    bl      do_number
    cbz     x1, not_found
    add     sp, sp, #16
    adrp    x3, var_state
    add     x3, x3, :lo12:var_state
    ldr     x3, [x3]
    cbz     x3, push_number
    adrp    x1, cfa_LIT
    add     x1, x1, :lo12:cfa_LIT
    str     x1, [x20], #8
    str     x0, [x20], #8
    b       interp_loop

push_number:
    str     x22, [x24, #-8]!
    mov     x22, x0
    b       interp_loop

not_found:
    ldp     x0, x1, [sp], #16
    adrp    x3, var_state
    add     x3, x3, :lo12:var_state
    ldr     x3, [x3]
    cbnz    x3, compile_latebind
    // Fatal: undefined word
    stp     x0, x1, [sp, #-16]!
    adrp    x1, err_undef
    add     x1, x1, :lo12:err_undef
    mov     x2, #2
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x0, x1, [sp], #16
    mov     x2, x1
    mov     x1, x0
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, newline
    add     x1, x1, :lo12:newline
    mov     x2, #1
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

compile_latebind:
    adrp    x2, cfa_LATEBIND
    add     x2, x2, :lo12:cfa_LATEBIND
    str     x2, [x20], #8
    strb    w1, [x20], #1
    mov     x3, x0
    mov     x4, x1
1:  cbz     x4, 2f
    ldrb    w5, [x3], #1
    strb    w5, [x20], #1
    sub     x4, x4, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    b       interp_loop

interp_done:
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// Main loop
// ============================================================
.align 4
main_loop:
    bl      do_refill
    cmp     x22, #0
    ldr     x22, [x24], #8
    b.eq    do_quit
    bl      code_INTERPRET
    b       main_loop

do_quit:
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

do_refill:
    stp     x30, xzr, [sp, #-16]!
    str     x22, [x24, #-8]!
    adrp    x10, var_input_fd
    add     x10, x10, :lo12:var_input_fd
    ldr     x0, [x10]
    adrp    x1, tib
    add     x1, x1, :lo12:tib
    mov     x2, #TIB_SIZE
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #0
    b.le    1f
    adrp    x10, var_source_addr
    add     x10, x10, :lo12:var_source_addr
    adrp    x1, tib
    add     x1, x1, :lo12:tib
    str     x1, [x10]
    adrp    x10, var_source_len
    add     x10, x10, :lo12:var_source_len
    str     x0, [x10]
    adrp    x10, var_to_in
    add     x10, x10, :lo12:var_to_in
    str     xzr, [x10]
    mov     x22, #-1
    ldp     x30, xzr, [sp], #16
    ret
1:  mov     x22, #0
    ldp     x30, xzr, [sp], #16
    ret

// ============================================================
// CREATE, VARIABLE, CONSTANT, COLON, SEMICOLON, IMMEDIATE
// ============================================================

.align 4
code_CREATE:
    bl      do_parse_name
    cbz     x1, create_done
    mov     x2, x0
    mov     x3, x1
    add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x4, [x0]
    str     x4, [x20], #8
    sub     x4, x20, #8
    str     x4, [x0]
    strb    wzr, [x20], #1
    and     w3, w3, #0x1F
    strb    w3, [x20], #1
    mov     x5, x3
1:  cbz     x5, 2f
    ldrb    w6, [x2], #1
    strb    w6, [x20], #1
    sub     x5, x5, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, code_DOCREATE
    add     x0, x0, :lo12:code_DOCREATE
    str     x0, [x20], #8
create_done:
    NEXT

.align 4
code_DOCREATE:
    str     x22, [x24, #-8]!
    add     x22, x25, #8
    NEXT

.align 4
code_VARIABLE:
    stp     x26, x30, [sp, #-16]!
    bl      do_parse_name
    cbz     x1, var_done
    mov     x2, x0
    mov     x3, x1
    add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x4, [x0]
    str     x4, [x20], #8
    sub     x4, x20, #8
    str     x4, [x0]
    strb    wzr, [x20], #1
    and     w3, w3, #0x1F
    strb    w3, [x20], #1
    mov     x5, x3
1:  cbz     x5, 2f
    ldrb    w6, [x2], #1
    strb    w6, [x20], #1
    sub     x5, x5, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, code_DOCREATE
    add     x0, x0, :lo12:code_DOCREATE
    str     x0, [x20], #8
    str     xzr, [x20], #8
var_done:
    ldp     x26, x30, [sp], #16
    NEXT

.align 4
code_CONSTANT:
    stp     x26, x30, [sp, #-16]!
    mov     x17, x22
    ldr     x22, [x24], #8
    bl      do_parse_name
    cbz     x1, const_done
    mov     x2, x0
    mov     x3, x1
    add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x4, [x0]
    str     x4, [x20], #8
    sub     x4, x20, #8
    str     x4, [x0]
    strb    wzr, [x20], #1
    and     w3, w3, #0x1F
    strb    w3, [x20], #1
    mov     x5, x3
1:  cbz     x5, 2f
    ldrb    w6, [x2], #1
    strb    w6, [x20], #1
    sub     x5, x5, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, code_DOCONST
    add     x0, x0, :lo12:code_DOCONST
    str     x0, [x20], #8
    str     x17, [x20], #8
const_done:
    ldp     x26, x30, [sp], #16
    NEXT

.align 4
code_DOCONST:
    str     x22, [x24, #-8]!
    ldr     x22, [x25, #8]
    NEXT

.align 4
code_COLON:
    stp     x26, x30, [sp, #-16]!
    bl      do_parse_name
    cbz     x1, colon_done
    mov     x2, x0
    mov     x3, x1
    add     x20, x20, #7
    and     x20, x20, #~7
    mov     x17, x20
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x4, [x0]
    str     x4, [x20], #8
    sub     x4, x20, #8
    str     x4, [x0]
    mov     w5, #F_HIDDEN
    strb    w5, [x20], #1
    and     w3, w3, #0x1F
    strb    w3, [x20], #1
    mov     x5, x3
1:  cbz     x5, 2f
    ldrb    w6, [x2], #1
    strb    w6, [x20], #1
    sub     x5, x5, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, code_DOCOL
    add     x0, x0, :lo12:code_DOCOL
    str     x0, [x20], #8
    adrp    x0, var_colon_entry
    add     x0, x0, :lo12:var_colon_entry
    str     x17, [x0]
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    mov     x1, #-1
    str     x1, [x0]
colon_done:
    ldp     x26, x30, [sp], #16
    NEXT

.align 4
code_SEMICOLON:
    adrp    x0, cfa_EXIT
    add     x0, x0, :lo12:cfa_EXIT
    str     x0, [x20], #8
    adrp    x0, var_colon_entry
    add     x0, x0, :lo12:var_colon_entry
    ldr     x0, [x0]
    ldrb    w1, [x0, #8]
    and     w1, w1, #~F_HIDDEN
    strb    w1, [x0, #8]
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    str     xzr, [x0]
    NEXT

.align 4
code_IMMEDIATE:
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x0, [x0]
    ldrb    w1, [x0, #8]
    orr     w1, w1, #F_IMMEDIATE
    strb    w1, [x0, #8]
    NEXT

// ============================================================
// Control flow compilation (all IMMEDIATE)
// ============================================================

.align 4
code_IF:
    adrp    x0, cfa_0BRANCH
    add     x0, x0, :lo12:cfa_0BRANCH
    str     x0, [x20], #8
    str     x22, [x24, #-8]!
    mov     x22, x20
    str     xzr, [x20], #8
    NEXT

.align 4
code_ELSE:
    adrp    x0, cfa_BRANCH
    add     x0, x0, :lo12:cfa_BRANCH
    str     x0, [x20], #8
    mov     x0, x20
    str     xzr, [x20], #8
    sub     x1, x20, x22
    str     x1, [x22]
    mov     x22, x0
    NEXT

.align 4
code_THEN:
    sub     x0, x20, x22
    str     x0, [x22]
    ldr     x22, [x24], #8
    NEXT

.align 4
code_BEGIN:
    str     x22, [x24, #-8]!
    mov     x22, x20
    NEXT

.align 4
code_WHILE:
    adrp    x0, cfa_0BRANCH
    add     x0, x0, :lo12:cfa_0BRANCH
    str     x0, [x20], #8
    str     x22, [x24, #-8]!
    mov     x22, x20
    str     xzr, [x20], #8
    ldr     x0, [x24]
    str     x22, [x24]
    mov     x22, x0
    NEXT

.align 4
code_REPEAT:
    adrp    x0, cfa_BRANCH
    add     x0, x0, :lo12:cfa_BRANCH
    str     x0, [x20], #8
    sub     x0, x22, x20
    str     x0, [x20], #8
    ldr     x0, [x24], #8
    sub     x1, x20, x0
    str     x1, [x0]
    ldr     x22, [x24], #8
    NEXT

.align 4
code_UNTIL:
    adrp    x0, cfa_0BRANCH
    add     x0, x0, :lo12:cfa_0BRANCH
    str     x0, [x20], #8
    sub     x0, x22, x20
    str     x0, [x20], #8
    ldr     x22, [x24], #8
    NEXT

.align 4
code_AGAIN:
    adrp    x0, cfa_BRANCH
    add     x0, x0, :lo12:cfa_BRANCH
    str     x0, [x20], #8
    sub     x0, x22, x20
    str     x0, [x20], #8
    ldr     x22, [x24], #8
    NEXT

// DO/LOOP runtime
.align 4
code_DO:
    ldr     x0, [x24], #8
    RPUSH   x0
    RPUSH   x22
    ldr     x22, [x24], #8
    NEXT

.align 4
code_QDO:
    ldr     x0, [x24], #8
    cmp     x0, x22
    b.eq    1f
    RPUSH   x0
    RPUSH   x22
    ldr     x22, [x24], #8
    ldr     x0, [x26], #8
    NEXT
1:  ldr     x22, [x24], #8
    ldr     x0, [x26]
    add     x26, x26, x0
    NEXT

.align 4
code_LOOP:
    ldr     x0, [x23]
    add     x0, x0, #1
    ldr     x1, [x23, #8]
    cmp     x0, x1
    b.ge    1f
    str     x0, [x23]
    ldr     x0, [x26]
    add     x26, x26, x0
    NEXT
1:  add     x23, x23, #16
    add     x26, x26, #8
    NEXT

.align 4
code_I:
    str     x22, [x24, #-8]!
    ldr     x22, [x23]
    NEXT

.align 4
code_J:
    str     x22, [x24, #-8]!
    ldr     x22, [x23, #16]
    NEXT

.align 4
code_LEAVE:
    ldr     x0, [x23, #8]
    str     x0, [x23]
    NEXT

.align 4
code_UNLOOP:
    add     x23, x23, #16
    NEXT

// DO/LOOP compilation (IMMEDIATE)
.align 4
code_COMP_DO:
    adrp    x0, cfa_DO
    add     x0, x0, :lo12:cfa_DO
    str     x0, [x20], #8
    str     x22, [x24, #-8]!
    mov     x22, #0
    str     x22, [x24, #-8]!
    mov     x22, x20
    NEXT

.align 4
code_COMP_QDO:
    adrp    x0, cfa_QDO
    add     x0, x0, :lo12:cfa_QDO
    str     x0, [x20], #8
    str     x22, [x24, #-8]!
    mov     x22, x20
    str     xzr, [x20], #8
    str     x22, [x24, #-8]!
    mov     x22, x20
    NEXT

.align 4
code_COMP_LOOP:
    adrp    x0, cfa_LOOP
    add     x0, x0, :lo12:cfa_LOOP
    str     x0, [x20], #8
    sub     x0, x22, x20
    str     x0, [x20], #8
    ldr     x22, [x24], #8
    cbz     x22, 1f
    sub     x0, x20, x22
    str     x0, [x22]
1:  ldr     x22, [x24], #8
    NEXT

.align 4
code_COMP_I:
    adrp    x0, cfa_I
    add     x0, x0, :lo12:cfa_I
    str     x0, [x20], #8
    NEXT

.align 4
code_COMP_J:
    adrp    x0, cfa_J
    add     x0, x0, :lo12:cfa_J
    str     x0, [x20], #8
    NEXT

.align 4
code_COMP_LEAVE:
    adrp    x0, cfa_LEAVE
    add     x0, x0, :lo12:cfa_LEAVE
    str     x0, [x20], #8
    NEXT

.align 4
code_COMP_UNLOOP:
    adrp    x0, cfa_UNLOOP
    add     x0, x0, :lo12:cfa_UNLOOP
    str     x0, [x20], #8
    NEXT

// ============================================================
// String compilation
// ============================================================

.align 4
code_SQUOTE:
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]
    mov     w15, #'"'
    bl      do_parse
    adrp    x3, var_state
    add     x3, x3, :lo12:var_state
    ldr     x3, [x3]
    cbz     x3, squote_interp
    adrp    x2, cfa_LITSTRING
    add     x2, x2, :lo12:cfa_LITSTRING
    str     x2, [x20], #8
    str     x1, [x20], #8
    mov     x3, x0
    mov     x4, x1
1:  cbz     x4, 2f
    ldrb    w5, [x3], #1
    strb    w5, [x20], #1
    sub     x4, x4, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    NEXT
squote_interp:
    str     x22, [x24, #-8]!
    str     x0, [x24, #-8]!
    mov     x22, x1
    NEXT

.align 4
code_DOTQUOTE:
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]
    mov     w15, #'"'
    bl      do_parse
    adrp    x3, var_state
    add     x3, x3, :lo12:var_state
    ldr     x3, [x3]
    cbz     x3, dotquote_interp
    adrp    x2, cfa_LITSTRING
    add     x2, x2, :lo12:cfa_LITSTRING
    str     x2, [x20], #8
    str     x1, [x20], #8
    mov     x3, x0
    mov     x4, x1
1:  cbz     x4, 2f
    ldrb    w5, [x3], #1
    strb    w5, [x20], #1
    sub     x4, x4, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x2, cfa_TYPE
    add     x2, x2, :lo12:cfa_TYPE
    str     x2, [x20], #8
    NEXT
dotquote_interp:
    mov     x2, x1
    mov     x1, x0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    NEXT

// ============================================================
// INCLUDED / INCLUDE
// ============================================================

.align 4
code_INCLUDED:
    stp     x26, x30, [sp, #-16]!
    POP     x1
    POP     x0
    adrp    x4, pad_buf
    add     x4, x4, :lo12:pad_buf
    mov     x5, x0
    mov     x6, x1
    cbz     x6, incl_fail
1:  ldrb    w7, [x5], #1
    strb    w7, [x4], #1
    subs    x6, x6, #1
    b.ne    1b
    strb    wzr, [x4]
    adrp    x1, pad_buf
    add     x1, x1, :lo12:pad_buf
    mov     x0, #AT_FDCWD
    mov     x2, #0
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    incl_open_fail
    mov     x17, x0
    // mmap buffer for this include
    mov     x0, #0
    mov     x1, #FILE_BUF_SIZE
    mov     x2, #3
    mov     x3, #0x22
    mov     x4, #-1
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmp     x0, #0
    b.lt    incl_fail
    mov     x16, x0
    mov     x1, x16
    mov     x0, x17
    mov     x2, #FILE_BUF_SIZE
    mov     x8, #SYS_READ
    svc     #0
    mov     x18, x0
    mov     x0, x17
    mov     x8, #SYS_CLOSE
    svc     #0
    cmp     x18, #0
    b.le    incl_mmap_fail
    // Save source state
    adrp    x0, var_source_addr
    add     x0, x0, :lo12:var_source_addr
    ldr     x4, [x0]
    adrp    x0, var_source_len
    add     x0, x0, :lo12:var_source_len
    ldr     x5, [x0]
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    ldr     x6, [x0]
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    ldr     x7, [x0]
    stp     x4, x5, [sp, #-16]!
    stp     x6, x18, [sp, #-16]!
    stp     x16, x7, [sp, #-16]!
    // Set new source
    adrp    x0, var_source_addr
    add     x0, x0, :lo12:var_source_addr
    str     x16, [x0]
    adrp    x0, var_source_len
    add     x0, x0, :lo12:var_source_len
    str     x18, [x0]
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    str     xzr, [x0]
    bl      code_INTERPRET
    // Cleanup
    ldp     x16, x7, [sp], #16
    mov     x0, x16
    mov     x1, #FILE_BUF_SIZE
    mov     x8, #SYS_MUNMAP
    svc     #0
    ldp     x6, x18, [sp], #16
    ldp     x4, x5, [sp], #16
    adrp    x0, var_to_in
    add     x0, x0, :lo12:var_to_in
    str     x6, [x0]
    adrp    x0, var_source_len
    add     x0, x0, :lo12:var_source_len
    str     x5, [x0]
    adrp    x0, var_source_addr
    add     x0, x0, :lo12:var_source_addr
    str     x4, [x0]
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    str     x7, [x0]
incl_done:
    ldp     x26, x30, [sp], #16
    NEXT
incl_open_fail:
    adrp    x1, err_open
    add     x1, x1, :lo12:err_open
    mov     x2, #18
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, pad_buf
    add     x1, x1, :lo12:pad_buf
    mov     x2, #0
4:  ldrb    w3, [x1, x2]
    cbz     w3, 5f
    add     x2, x2, #1
    b       4b
5:  mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    adrp    x1, newline
    add     x1, x1, :lo12:newline
    mov     x2, #1
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
incl_mmap_fail:
    mov     x0, x16
    mov     x1, #FILE_BUF_SIZE
    mov     x8, #SYS_MUNMAP
    svc     #0
incl_fail:
    ldp     x26, x30, [sp], #16
    NEXT

.align 4
code_INCLUDE:
    bl      do_parse_name
    cbz     x1, 1f
    str     x22, [x24, #-8]!
    str     x0, [x24, #-8]!
    mov     x22, x1
    b       code_INCLUDED
1:  NEXT

// ============================================================
// Miscellaneous words
// ============================================================

.align 4
code_BYE:
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

.align 4
code_EXECUTE:
    mov     x25, x22
    ldr     x22, [x24], #8
    ldr     x16, [x25]
    br      x16

.align 4
code_TICK:
    bl      do_parse_name
    cbz     x1, tick_fail
    bl      do_find_word
    cbz     x0, tick_fail
    str     x22, [x24, #-8]!
    mov     x22, x0
    NEXT
tick_fail:
    adrp    x1, err_undef
    add     x1, x1, :lo12:err_undef
    mov     x2, #2
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.align 4
code_STATE:
    str     x22, [x24, #-8]!
    adrp    x22, var_state
    add     x22, x22, :lo12:var_state
    NEXT

.align 4
code_BASE:
    str     x22, [x24, #-8]!
    mov     x22, x21
    NEXT

.align 4
code_LATEST:
    str     x22, [x24, #-8]!
    adrp    x22, var_latest
    add     x22, x22, :lo12:var_latest
    NEXT

.align 4
code_TOIN:
    str     x22, [x24, #-8]!
    adrp    x22, var_to_in
    add     x22, x22, :lo12:var_to_in
    NEXT

.align 4
code_SOURCE:
    str     x22, [x24, #-8]!
    adrp    x0, var_source_addr
    add     x0, x0, :lo12:var_source_addr
    ldr     x0, [x0]
    str     x0, [x24, #-8]!
    adrp    x0, var_source_len
    add     x0, x0, :lo12:var_source_len
    ldr     x22, [x0]
    NEXT

.align 4
code_LBRACKET:
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    str     xzr, [x0]
    NEXT

.align 4
code_RBRACKET:
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    mov     x1, #-1
    str     x1, [x0]
    NEXT

.align 4
code_CHAR:
    bl      do_parse_name
    cbz     x1, 1f
    ldrb    w0, [x0]
    str     x22, [x24, #-8]!
    mov     x22, x0
    NEXT
1:  str     x22, [x24, #-8]!
    mov     x22, #0
    NEXT

.align 4
code_BRACKETCHAR:
    bl      do_parse_name
    cbz     x1, 1f
    ldrb    w0, [x0]
    adrp    x1, cfa_LIT
    add     x1, x1, :lo12:cfa_LIT
    str     x1, [x20], #8
    str     x0, [x20], #8
1:  NEXT

.align 4
code_LITERAL:
    adrp    x0, cfa_LIT
    add     x0, x0, :lo12:cfa_LIT
    str     x0, [x20], #8
    str     x22, [x20], #8
    ldr     x22, [x24], #8
    NEXT

.align 4
code_HEX:
    mov     x21, #16
    NEXT

.align 4
code_DECIMAL:
    mov     x21, #10
    NEXT

.align 4
code_PARSE:
    mov     w15, w22
    ldr     x22, [x24], #8
    bl      do_parse
    str     x22, [x24, #-8]!
    str     x0, [x24, #-8]!
    mov     x22, x1
    NEXT

.align 4
code_PARSE_NAME:
    bl      do_parse_name
    str     x22, [x24, #-8]!
    str     x0, [x24, #-8]!
    mov     x22, x1
    NEXT

.align 4
code_WORD:
    mov     w15, w22
    bl      do_parse_name
    adrp    x2, word_buf
    add     x2, x2, :lo12:word_buf
    strb    w1, [x2]
    mov     x3, #0
1:  cmp     x3, x1
    b.ge    2f
    ldrb    w4, [x0, x3]
    add     x5, x3, #1
    strb    w4, [x2, x5]
    add     x3, x3, #1
    b       1b
2:  mov     x22, x2
    NEXT

.align 4
code_ACCEPT:
    POP     x2
    mov     x1, x22
    adrp    x0, var_input_fd
    add     x0, x0, :lo12:var_input_fd
    ldr     x0, [x0]
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #0
    b.lt    1f
    mov     x22, x0
    NEXT
1:  mov     x22, #0
    NEXT

.align 4
code_BACKSLASH:
    adrp    x10, var_source_addr
    add     x10, x10, :lo12:var_source_addr
    ldr     x10, [x10]
    adrp    x11, var_source_len
    add     x11, x11, :lo12:var_source_len
    ldr     x11, [x11]
    adrp    x12, var_to_in
    add     x12, x12, :lo12:var_to_in
    ldr     x13, [x12]
1:  cmp     x13, x11
    b.ge    2f
    ldrb    w14, [x10, x13]
    add     x13, x13, #1
    cmp     w14, #10
    b.ne    1b
2:  str     x13, [x12]
    NEXT

.align 4
code_PAREN:
    mov     w15, #')'
    bl      do_parse
    NEXT

.align 4
code_ABORT:
    adrp    x24, data_stack_top
    add     x24, x24, :lo12:data_stack_top
    adrp    x23, ret_stack_top
    add     x23, x23, :lo12:ret_stack_top
    mov     x22, #0
    adrp    x0, var_state
    add     x0, x0, :lo12:var_state
    str     xzr, [x0]
    b       main_loop

.align 4
code_THROW:
    cmp     x22, #0
    b.eq    1f
    b       code_ABORT
1:  ldr     x22, [x24], #8
    NEXT

.align 4
code_BL:
    str     x22, [x24, #-8]!
    mov     x22, #32
    NEXT

.align 4
code_FIND:
    // find ( c-addr -- xt 1 | xt -1 | c-addr 0 )
    ldrb    w1, [x22]
    add     x0, x22, #1
    bl      do_find_word
    cbz     x0, 1f
    str     x0, [x24, #-8]!
    tst     x1, #F_IMMEDIATE
    b.eq    2f
    mov     x22, #1
    NEXT
2:  mov     x22, #-1
    NEXT
1:  str     x22, [x24, #-8]!
    mov     x22, #0
    NEXT

// Conditional compilation
.align 4
code_BRACKETDEFINED:
    bl      do_parse_name
    cbz     x1, 1f
    bl      do_find_word
    str     x22, [x24, #-8]!
    cmp     x0, #0
    csetm   x22, ne
    NEXT
1:  str     x22, [x24, #-8]!
    mov     x22, #0
    NEXT

.align 4
code_BRACKETUNDEFINED:
    bl      do_parse_name
    cbz     x1, 1f
    bl      do_find_word
    str     x22, [x24, #-8]!
    cmp     x0, #0
    csetm   x22, eq
    NEXT
1:  str     x22, [x24, #-8]!
    mov     x22, #-1
    NEXT

.align 4
code_BRACKETIF:
    cmp     x22, #0
    ldr     x22, [x24], #8
    b.ne    1f
    mov     x17, #1
2:  bl      do_parse_name
    cbz     x1, 3f
    cmp     x1, #4
    b.ne    4f
    ldrb    w2, [x0]
    cmp     w2, #'['
    b.ne    4f
    adrp    x3, str_bracket_if
    add     x3, x3, :lo12:str_bracket_if
    ldrb    w4, [x0, #1]
    ldrb    w5, [x3, #1]
    orr     w4, w4, #0x20
    cmp     w4, w5
    b.ne    4f
    ldrb    w4, [x0, #2]
    ldrb    w5, [x3, #2]
    orr     w4, w4, #0x20
    cmp     w4, w5
    b.ne    4f
    ldrb    w4, [x0, #3]
    cmp     w4, #']'
    b.ne    4f
    add     x17, x17, #1
    b       2b
4:  cmp     x17, #1
    b.ne    5f
    cmp     x1, #6
    b.ne    5f
    adrp    x3, str_bracket_else
    add     x3, x3, :lo12:str_bracket_else
    mov     x4, #0
6:  cmp     x4, x1
    b.ge    7f
    ldrb    w5, [x0, x4]
    ldrb    w6, [x3, x4]
    orr     w5, w5, #0x20
    cmp     w5, w6
    b.ne    5f
    add     x4, x4, #1
    b       6b
7:  b       1f
5:  cmp     x1, #6
    b.ne    2b
    adrp    x3, str_bracket_then
    add     x3, x3, :lo12:str_bracket_then
    mov     x4, #0
8:  cmp     x4, x1
    b.ge    9f
    ldrb    w5, [x0, x4]
    ldrb    w6, [x3, x4]
    orr     w5, w5, #0x20
    cmp     w5, w6
    b.ne    2b
    add     x4, x4, #1
    b       8b
9:  sub     x17, x17, #1
    cbz     x17, 1f
    b       2b
3:
1:  NEXT

.align 4
code_BRACKETELSE:
    mov     x17, #1
1:  bl      do_parse_name
    cbz     x1, 2f
    cmp     x1, #6
    b.ne    1b
    adrp    x3, str_bracket_then
    add     x3, x3, :lo12:str_bracket_then
    mov     x4, #0
3:  cmp     x4, x1
    b.ge    4f
    ldrb    w5, [x0, x4]
    ldrb    w6, [x3, x4]
    orr     w5, w5, #0x20
    cmp     w5, w6
    b.ne    1b
    add     x4, x4, #1
    b       3b
4:  sub     x17, x17, #1
    cbz     x17, 2f
    b       1b
2:  NEXT

.align 4
code_BRACKETTHEN:
    NEXT

// DEFER / IS
.align 4
code_DODEFER:
    ldr     x25, [x25, #8]
    ldr     x16, [x25]
    br      x16

.align 4
code_DEFER_ABORT:
    adrp    x1, err_defer_unresolved
    add     x1, x1, :lo12:err_defer_unresolved
    mov     x2, #18
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    b       code_ABORT

.align 4
code_DEFER:
    stp     x26, x30, [sp, #-16]!
    bl      do_parse_name
    cbz     x1, defer_done
    mov     x2, x0
    mov     x3, x1
    add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, var_latest
    add     x0, x0, :lo12:var_latest
    ldr     x4, [x0]
    str     x4, [x20], #8
    sub     x4, x20, #8
    str     x4, [x0]
    strb    wzr, [x20], #1
    and     w3, w3, #0x1F
    strb    w3, [x20], #1
    mov     x5, x3
1:  cbz     x5, 2f
    ldrb    w6, [x2], #1
    strb    w6, [x20], #1
    sub     x5, x5, #1
    b       1b
2:  add     x20, x20, #7
    and     x20, x20, #~7
    adrp    x0, code_DODEFER
    add     x0, x0, :lo12:code_DODEFER
    str     x0, [x20], #8
    adrp    x0, cfa_DEFER_ABORT
    add     x0, x0, :lo12:cfa_DEFER_ABORT
    str     x0, [x20], #8
defer_done:
    ldp     x26, x30, [sp], #16
    NEXT

.align 4
code_IS:
    stp     x26, x30, [sp, #-16]!
    bl      do_parse_name
    cbz     x1, is_done
    bl      do_find_word
    cbz     x0, is_done
    str     x22, [x0, #8]
    ldr     x22, [x24], #8
is_done:
    ldp     x26, x30, [sp], #16
    NEXT

// ARGC / ARGV / SLURP-FILE
.align 4
code_ARGC:
    str     x22, [x24, #-8]!
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    ldr     x22, [x0]
    NEXT

.align 4
code_ARGV:
    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    ldr     x0, [x0]
    ldr     x1, [x0, x22, lsl #3]
    mov     x2, x1
1:  ldrb    w3, [x2], #1
    cbnz    w3, 1b
    sub     x2, x2, x1
    sub     x2, x2, #1
    str     x22, [x24, #-8]!   // push old TOS (index)
    str     x1, [x24]          // replace with c-addr
    mov     x22, x2
    NEXT

.align 4
code_SLURP_FILE:
    ldr     x1, [x24]
    mov     x4, x22
    adrp    x3, path_buf
    add     x3, x3, :lo12:path_buf
    cbz     x4, .Lslurp_copy_done
.Lslurp_copy_loop:
    ldrb    w5, [x1], #1
    strb    w5, [x3], #1
    subs    x4, x4, #1
    b.ne    .Lslurp_copy_loop
.Lslurp_copy_done:
    strb    wzr, [x3]
    adrp    x1, path_buf
    add     x1, x1, :lo12:path_buf
    mov     x0, #-100
    mov     x2, #0
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    slurp_fail
    mov     x9, x0
    mov     x0, x9
    mov     x1, #0
    mov     x2, #2
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x10, x0
    mov     x0, x9
    mov     x1, #0
    mov     x2, #0
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x11, x20
    add     x20, x20, x10
    mov     x0, x9
    mov     x1, x11
    mov     x2, x10
    mov     x8, #SYS_READ
    svc     #0
    mov     x12, x0
    mov     x0, x9
    mov     x8, #SYS_CLOSE
    svc     #0
    str     x11, [x24]
    mov     x22, x12
    NEXT
slurp_fail:
    str     xzr, [x24]
    mov     x22, #0
    NEXT

.align 4
code_PAD:
    str     x22, [x24, #-8]!
    adrp    x22, pad_buf
    add     x22, x22, :lo12:pad_buf
    NEXT

// ============================================================
// Data section
// ============================================================

.data
.align 3

newline:    .byte 10
space_char: .byte 32
emit_buf:   .byte 0
.align 3
num_buf:    .space 64
err_undef:  .ascii "? "
err_open:   .ascii "Cannot open file: "
err_latebind: .ascii "Late-bind: "
err_defer_unresolved: .ascii "Unresolved defer!\n"
    .byte 0
.align 3

str_bracket_if:   .ascii "[if]"
str_bracket_else: .ascii "[else]"
str_bracket_then: .ascii "[then]"
.align 3

.align 3
var_state:      .quad 0
var_latest:     .quad 0
var_to_in:      .quad 0
var_source_addr: .quad 0
var_source_len: .quad 0
var_input_fd:   .quad 0
var_colon_entry: .quad 0
.align 3

word_buf:   .space WORD_BUF_SIZE
pad_buf:    .space PAD_SIZE
.align 3
exec_pad:   .space 16
.align 3

saved_argc: .quad 0
saved_argv: .quad 0
path_buf:   .space 4096

// Internal CFA cells
.align 3
cfa_LIT:        .quad code_LIT
cfa_LATEBIND:   .quad code_LATEBIND
cfa_LITSTRING:  .quad code_LITSTRING
cfa_BRANCH:     .quad code_BRANCH
cfa_0BRANCH:    .quad code_0BRANCH
cfa_DO:         .quad code_DO
cfa_QDO:        .quad code_QDO
cfa_LOOP:       .quad code_LOOP
cfa_I:          .quad code_I
cfa_J:          .quad code_J
cfa_LEAVE:      .quad code_LEAVE
cfa_UNLOOP:     .quad code_UNLOOP
cfa_TYPE:       .quad code_TYPE
cfa_DEFER_ABORT: .quad code_DEFER_ABORT

// Stacks
.bss
.align 4
data_stack: .space DSTACK_SIZE
data_stack_top:
.space 4096
.align 4
ret_stack:  .space RSTACK_SIZE
ret_stack_top:
.align 4
tib:        .space TIB_SIZE
.align 4
mem_space:  .space MEM_SIZE

// ============================================================
// Dictionary — only words Lithos uses
// ============================================================
.data
.align 3

entry_exit:
    .quad   0
    .byte   0
    .byte   4
    .ascii  "exit"
    .align  3
cfa_EXIT:
    .quad   code_EXIT

entry_dup:
    .quad   entry_exit
    .byte   0
    .byte   3
    .ascii  "dup"
    .align  3
    .quad   code_DUP

entry_drop:
    .quad   entry_dup
    .byte   0
    .byte   4
    .ascii  "drop"
    .align  3
    .quad   code_DROP

entry_swap:
    .quad   entry_drop
    .byte   0
    .byte   4
    .ascii  "swap"
    .align  3
    .quad   code_SWAP

entry_over:
    .quad   entry_swap
    .byte   0
    .byte   4
    .ascii  "over"
    .align  3
    .quad   code_OVER

entry_rot:
    .quad   entry_over
    .byte   0
    .byte   3
    .ascii  "rot"
    .align  3
    .quad   code_ROT

entry_nrot:
    .quad   entry_rot
    .byte   0
    .byte   4
    .ascii  "-rot"
    .align  3
    .quad   code_NROT

entry_nip:
    .quad   entry_nrot
    .byte   0
    .byte   3
    .ascii  "nip"
    .align  3
    .quad   code_NIP

entry_pick:
    .quad   entry_nip
    .byte   0
    .byte   4
    .ascii  "pick"
    .align  3
    .quad   code_PICK

entry_depth:
    .quad   entry_pick
    .byte   0
    .byte   5
    .ascii  "depth"
    .align  3
    .quad   code_DEPTH

entry_2dup:
    .quad   entry_depth
    .byte   0
    .byte   4
    .ascii  "2dup"
    .align  3
    .quad   code_2DUP

entry_2drop:
    .quad   entry_2dup
    .byte   0
    .byte   5
    .ascii  "2drop"
    .align  3
    .quad   code_2DROP

entry_2swap:
    .quad   entry_2drop
    .byte   0
    .byte   5
    .ascii  "2swap"
    .align  3
    .quad   code_2SWAP

entry_tor:
    .quad   entry_2swap
    .byte   0
    .byte   2
    .ascii  ">r"
    .align  3
    .quad   code_TOR

entry_rfrom:
    .quad   entry_tor
    .byte   0
    .byte   2
    .ascii  "r>"
    .align  3
    .quad   code_RFROM

entry_rfetch:
    .quad   entry_rfrom
    .byte   0
    .byte   2
    .ascii  "r@"
    .align  3
    .quad   code_RFETCH

entry_plus:
    .quad   entry_rfetch
    .byte   0
    .byte   1
    .ascii  "+"
    .align  3
    .quad   code_PLUS

entry_minus:
    .quad   entry_plus
    .byte   0
    .byte   1
    .ascii  "-"
    .align  3
    .quad   code_MINUS

entry_star:
    .quad   entry_minus
    .byte   0
    .byte   1
    .ascii  "*"
    .align  3
    .quad   code_STAR

entry_slash:
    .quad   entry_star
    .byte   0
    .byte   1
    .ascii  "/"
    .align  3
    .quad   code_SLASH

entry_mod:
    .quad   entry_slash
    .byte   0
    .byte   3
    .ascii  "mod"
    .align  3
    .quad   code_MOD

entry_and:
    .quad   entry_mod
    .byte   0
    .byte   3
    .ascii  "and"
    .align  3
    .quad   code_AND

entry_or:
    .quad   entry_and
    .byte   0
    .byte   2
    .ascii  "or"
    .align  3
    .quad   code_OR

entry_xor:
    .quad   entry_or
    .byte   0
    .byte   3
    .ascii  "xor"
    .align  3
    .quad   code_XOR

entry_lshift:
    .quad   entry_xor
    .byte   0
    .byte   6
    .ascii  "lshift"
    .align  3
    .quad   code_LSHIFT

entry_rshift:
    .quad   entry_lshift
    .byte   0
    .byte   6
    .ascii  "rshift"
    .align  3
    .quad   code_RSHIFT

entry_negate:
    .quad   entry_rshift
    .byte   0
    .byte   6
    .ascii  "negate"
    .align  3
    .quad   code_NEGATE

entry_invert:
    .quad   entry_negate
    .byte   0
    .byte   6
    .ascii  "invert"
    .align  3
    .quad   code_INVERT

entry_1plus:
    .quad   entry_invert
    .byte   0
    .byte   2
    .ascii  "1+"
    .align  3
    .quad   code_1PLUS

entry_1minus:
    .quad   entry_1plus
    .byte   0
    .byte   2
    .ascii  "1-"
    .align  3
    .quad   code_1MINUS

entry_2star:
    .quad   entry_1minus
    .byte   0
    .byte   2
    .ascii  "2*"
    .align  3
    .quad   code_2STAR

entry_cells:
    .quad   entry_2star
    .byte   0
    .byte   5
    .ascii  "cells"
    .align  3
    .quad   code_CELLS

entry_fetch:
    .quad   entry_cells
    .byte   0
    .byte   1
    .ascii  "@"
    .align  3
    .quad   code_FETCH

entry_store:
    .quad   entry_fetch
    .byte   0
    .byte   1
    .ascii  "!"
    .align  3
    .quad   code_STORE

entry_cfetch:
    .quad   entry_store
    .byte   0
    .byte   2
    .ascii  "c@"
    .align  3
    .quad   code_CFETCH

entry_cstore:
    .quad   entry_cfetch
    .byte   0
    .byte   2
    .ascii  "c!"
    .align  3
    .quad   code_CSTORE

entry_plusstore:
    .quad   entry_cstore
    .byte   0
    .byte   2
    .ascii  "+!"
    .align  3
    .quad   code_PLUSSTORE

entry_fill:
    .quad   entry_plusstore
    .byte   0
    .byte   4
    .ascii  "fill"
    .align  3
    .quad   code_FILL

entry_move:
    .quad   entry_fill
    .byte   0
    .byte   4
    .ascii  "move"
    .align  3
    .quad   code_MOVE

entry_here:
    .quad   entry_move
    .byte   0
    .byte   4
    .ascii  "here"
    .align  3
    .quad   code_HERE

entry_allot:
    .quad   entry_here
    .byte   0
    .byte   5
    .ascii  "allot"
    .align  3
    .quad   code_ALLOT

entry_align:
    .quad   entry_allot
    .byte   0
    .byte   5
    .ascii  "align"
    .align  3
    .quad   code_ALIGN

entry_comma:
    .quad   entry_align
    .byte   0
    .byte   1
    .ascii  ","
    .align  3
    .quad   code_COMMA

entry_ccomma:
    .quad   entry_comma
    .byte   0
    .byte   2
    .ascii  "c,"
    .align  3
    .quad   code_CCOMMA

entry_eq:
    .quad   entry_ccomma
    .byte   0
    .byte   1
    .ascii  "="
    .align  3
    .quad   code_EQ

entry_neq:
    .quad   entry_eq
    .byte   0
    .byte   2
    .ascii  "<>"
    .align  3
    .quad   code_NEQ

entry_lt:
    .quad   entry_neq
    .byte   0
    .byte   1
    .ascii  "<"
    .align  3
    .quad   code_LT

entry_gt:
    .quad   entry_lt
    .byte   0
    .byte   1
    .ascii  ">"
    .align  3
    .quad   code_GT

entry_zeroeq:
    .quad   entry_gt
    .byte   0
    .byte   2
    .ascii  "0="
    .align  3
    .quad   code_ZEROEQ

entry_zerogt:
    .quad   entry_zeroeq
    .byte   0
    .byte   2
    .ascii  "0>"
    .align  3
    .quad   code_ZEROGT

entry_zerolt:
    .quad   entry_zerogt
    .byte   0
    .byte   2
    .ascii  "0<"
    .align  3
    .quad   code_ZEROLT

entry_zerone:
    .quad   entry_zerolt
    .byte   0
    .byte   3
    .ascii  "0<>"
    .align  3
    .quad   code_ZERONE

entry_min:
    .quad   entry_zerone
    .byte   0
    .byte   3
    .ascii  "min"
    .align  3
    .quad   code_MIN

entry_max:
    .quad   entry_min
    .byte   0
    .byte   3
    .ascii  "max"
    .align  3
    .quad   code_MAX

entry_within:
    .quad   entry_max
    .byte   0
    .byte   6
    .ascii  "within"
    .align  3
    .quad   code_WITHIN

entry_emit:
    .quad   entry_within
    .byte   0
    .byte   4
    .ascii  "emit"
    .align  3
    .quad   code_EMIT

entry_type:
    .quad   entry_emit
    .byte   0
    .byte   4
    .ascii  "type"
    .align  3
    .quad   code_TYPE

entry_cr:
    .quad   entry_type
    .byte   0
    .byte   2
    .ascii  "cr"
    .align  3
    .quad   code_CR

entry_space:
    .quad   entry_cr
    .byte   0
    .byte   5
    .ascii  "space"
    .align  3
    .quad   code_SPACE

entry_dot:
    .quad   entry_space
    .byte   0
    .byte   1
    .ascii  "."
    .align  3
    .quad   code_DOT

entry_udot:
    .quad   entry_dot
    .byte   0
    .byte   2
    .ascii  "u."
    .align  3
    .quad   code_UDOT

entry_dots:
    .quad   entry_udot
    .byte   0
    .byte   2
    .ascii  ".s"
    .align  3
    .quad   code_DOTS

entry_open_file:
    .quad   entry_dots
    .byte   0
    .byte   9
    .ascii  "open-file"
    .align  3
    .quad   code_OPEN_FILE

entry_close_file:
    .quad   entry_open_file
    .byte   0
    .byte   10
    .ascii  "close-file"
    .align  3
    .quad   code_CLOSE_FILE

entry_read_file:
    .quad   entry_close_file
    .byte   0
    .byte   9
    .ascii  "read-file"
    .align  3
    .quad   code_READ_FILE

entry_write_file:
    .quad   entry_read_file
    .byte   0
    .byte   10
    .ascii  "write-file"
    .align  3
    .quad   code_WRITE_FILE

entry_ro:
    .quad   entry_write_file
    .byte   0
    .byte   3
    .ascii  "r/o"
    .align  3
    .quad   code_RO

entry_wo:
    .quad   entry_ro
    .byte   0
    .byte   3
    .ascii  "w/o"
    .align  3
    .quad   code_WO

entry_count:
    .quad   entry_wo
    .byte   0
    .byte   5
    .ascii  "count"
    .align  3
    .quad   code_COUNT

entry_compare:
    .quad   entry_count
    .byte   0
    .byte   7
    .ascii  "compare"
    .align  3
    .quad   code_COMPARE

entry_squote:
    .quad   entry_compare
    .byte   F_IMMEDIATE
    .byte   2
    .ascii  "s\""
    .align  3
    .quad   code_SQUOTE

entry_dotquote:
    .quad   entry_squote
    .byte   F_IMMEDIATE
    .byte   2
    .ascii  ".\""
    .align  3
    .quad   code_DOTQUOTE

entry_create:
    .quad   entry_dotquote
    .byte   0
    .byte   6
    .ascii  "create"
    .align  3
    .quad   code_CREATE

entry_variable:
    .quad   entry_create
    .byte   0
    .byte   8
    .ascii  "variable"
    .align  3
    .quad   code_VARIABLE

entry_constant:
    .quad   entry_variable
    .byte   0
    .byte   8
    .ascii  "constant"
    .align  3
    .quad   code_CONSTANT

entry_colon:
    .quad   entry_constant
    .byte   0
    .byte   1
    .ascii  ":"
    .align  3
    .quad   code_COLON

entry_semicolon:
    .quad   entry_colon
    .byte   F_IMMEDIATE
    .byte   1
    .ascii  ";"
    .align  3
    .quad   code_SEMICOLON

entry_immediate:
    .quad   entry_semicolon
    .byte   0
    .byte   9
    .ascii  "immediate"
    .align  3
    .quad   code_IMMEDIATE

entry_if:
    .quad   entry_immediate
    .byte   F_IMMEDIATE
    .byte   2
    .ascii  "if"
    .align  3
    .quad   code_IF

entry_else:
    .quad   entry_if
    .byte   F_IMMEDIATE
    .byte   4
    .ascii  "else"
    .align  3
    .quad   code_ELSE

entry_then:
    .quad   entry_else
    .byte   F_IMMEDIATE
    .byte   4
    .ascii  "then"
    .align  3
    .quad   code_THEN

entry_begin:
    .quad   entry_then
    .byte   F_IMMEDIATE
    .byte   5
    .ascii  "begin"
    .align  3
    .quad   code_BEGIN

entry_while:
    .quad   entry_begin
    .byte   F_IMMEDIATE
    .byte   5
    .ascii  "while"
    .align  3
    .quad   code_WHILE

entry_repeat:
    .quad   entry_while
    .byte   F_IMMEDIATE
    .byte   6
    .ascii  "repeat"
    .align  3
    .quad   code_REPEAT

entry_until:
    .quad   entry_repeat
    .byte   F_IMMEDIATE
    .byte   5
    .ascii  "until"
    .align  3
    .quad   code_UNTIL

entry_again:
    .quad   entry_until
    .byte   F_IMMEDIATE
    .byte   5
    .ascii  "again"
    .align  3
    .quad   code_AGAIN

entry_do:
    .quad   entry_again
    .byte   F_IMMEDIATE
    .byte   2
    .ascii  "do"
    .align  3
    .quad   code_COMP_DO

entry_qdo:
    .quad   entry_do
    .byte   F_IMMEDIATE
    .byte   3
    .ascii  "?do"
    .align  3
    .quad   code_COMP_QDO

entry_loop:
    .quad   entry_qdo
    .byte   F_IMMEDIATE
    .byte   4
    .ascii  "loop"
    .align  3
    .quad   code_COMP_LOOP

entry_comp_i:
    .quad   entry_loop
    .byte   F_IMMEDIATE
    .byte   1
    .ascii  "i"
    .align  3
    .quad   code_COMP_I

entry_comp_j:
    .quad   entry_comp_i
    .byte   F_IMMEDIATE
    .byte   1
    .ascii  "j"
    .align  3
    .quad   code_COMP_J

entry_leave:
    .quad   entry_comp_j
    .byte   F_IMMEDIATE
    .byte   5
    .ascii  "leave"
    .align  3
    .quad   code_COMP_LEAVE

entry_unloop:
    .quad   entry_leave
    .byte   F_IMMEDIATE
    .byte   6
    .ascii  "unloop"
    .align  3
    .quad   code_COMP_UNLOOP

entry_state:
    .quad   entry_unloop
    .byte   0
    .byte   5
    .ascii  "state"
    .align  3
    .quad   code_STATE

entry_base:
    .quad   entry_state
    .byte   0
    .byte   4
    .ascii  "base"
    .align  3
    .quad   code_BASE

entry_rbracket:
    .quad   entry_base
    .byte   0
    .byte   1
    .ascii  "]"
    .align  3
    .quad   code_RBRACKET

entry_hex:
    .quad   entry_rbracket
    .byte   0
    .byte   3
    .ascii  "hex"
    .align  3
    .quad   code_HEX

entry_decimal:
    .quad   entry_hex
    .byte   0
    .byte   7
    .ascii  "decimal"
    .align  3
    .quad   code_DECIMAL

entry_bl:
    .quad   entry_decimal
    .byte   0
    .byte   2
    .ascii  "bl"
    .align  3
    .quad   code_BL

entry_find:
    .quad   entry_bl
    .byte   0
    .byte   4
    .ascii  "find"
    .align  3
    .quad   code_FIND

entry_bracketchar:
    .quad   entry_find
    .byte   F_IMMEDIATE
    .byte   6
    .ascii  "[char]"
    .align  3
    .quad   code_BRACKETCHAR

entry_char:
    .quad   entry_bracketchar
    .byte   0
    .byte   4
    .ascii  "char"
    .align  3
    .quad   code_CHAR

entry_literal:
    .quad   entry_char
    .byte   F_IMMEDIATE
    .byte   7
    .ascii  "literal"
    .align  3
    .quad   code_LITERAL

entry_execute:
    .quad   entry_literal
    .byte   0
    .byte   7
    .ascii  "execute"
    .align  3
    .quad   code_EXECUTE

entry_tick:
    .quad   entry_execute
    .byte   0
    .byte   1
    .ascii  "'"
    .align  3
    .quad   code_TICK

entry_bye:
    .quad   entry_tick
    .byte   0
    .byte   3
    .ascii  "bye"
    .align  3
    .quad   code_BYE

entry_bracketdefined:
    .quad   entry_bye
    .byte   F_IMMEDIATE
    .byte   9
    .ascii  "[defined]"
    .align  3
    .quad   code_BRACKETDEFINED

entry_bracketundefined:
    .quad   entry_bracketdefined
    .byte   F_IMMEDIATE
    .byte   11
    .ascii  "[undefined]"
    .align  3
    .quad   code_BRACKETUNDEFINED

entry_bracketif:
    .quad   entry_bracketundefined
    .byte   F_IMMEDIATE
    .byte   4
    .ascii  "[if]"
    .align  3
    .quad   code_BRACKETIF

entry_bracketelse:
    .quad   entry_bracketif
    .byte   F_IMMEDIATE
    .byte   6
    .ascii  "[else]"
    .align  3
    .quad   code_BRACKETELSE

entry_bracketthen:
    .quad   entry_bracketelse
    .byte   F_IMMEDIATE
    .byte   6
    .ascii  "[then]"
    .align  3
    .quad   code_BRACKETTHEN

entry_parse:
    .quad   entry_bracketthen
    .byte   0
    .byte   5
    .ascii  "parse"
    .align  3
    .quad   code_PARSE

entry_parse_name:
    .quad   entry_parse
    .byte   0
    .byte   10
    .ascii  "parse-name"
    .align  3
    .quad   code_PARSE_NAME

entry_word:
    .quad   entry_parse_name
    .byte   0
    .byte   4
    .ascii  "word"
    .align  3
    .quad   code_WORD

entry_accept:
    .quad   entry_word
    .byte   0
    .byte   6
    .ascii  "accept"
    .align  3
    .quad   code_ACCEPT

entry_source:
    .quad   entry_accept
    .byte   0
    .byte   6
    .ascii  "source"
    .align  3
    .quad   code_SOURCE

entry_toin:
    .quad   entry_source
    .byte   0
    .byte   3
    .ascii  ">in"
    .align  3
    .quad   code_TOIN

entry_backslash:
    .quad   entry_toin
    .byte   F_IMMEDIATE
    .byte   1
    .ascii  "\\"
    .align  3
    .quad   code_BACKSLASH

entry_paren:
    .quad   entry_backslash
    .byte   F_IMMEDIATE
    .byte   1
    .ascii  "("
    .align  3
    .quad   code_PAREN

entry_included:
    .quad   entry_paren
    .byte   0
    .byte   8
    .ascii  "included"
    .align  3
    .quad   code_INCLUDED

entry_include:
    .quad   entry_included
    .byte   0
    .byte   7
    .ascii  "include"
    .align  3
    .quad   code_INCLUDE

entry_abort:
    .quad   entry_include
    .byte   0
    .byte   5
    .ascii  "abort"
    .align  3
    .quad   code_ABORT

entry_throw:
    .quad   entry_abort
    .byte   0
    .byte   5
    .ascii  "throw"
    .align  3
    .quad   code_THROW

entry_defer:
    .quad   entry_throw
    .byte   0
    .byte   5
    .ascii  "defer"
    .align  3
    .quad   code_DEFER

entry_is:
    .quad   entry_defer
    .byte   0
    .byte   2
    .ascii  "is"
    .align  3
    .quad   code_IS

entry_argc:
    .quad   entry_is
    .byte   0
    .byte   4
    .ascii  "argc"
    .align  3
    .quad   code_ARGC

entry_argv:
    .quad   entry_argc
    .byte   0
    .byte   4
    .ascii  "argv"
    .align  3
    .quad   code_ARGV

entry_slurp_file:
    .quad   entry_argv
    .byte   0
    .byte   10
    .ascii  "slurp-file"
    .align  3
    .quad   code_SLURP_FILE

// last entry — var_latest points here at boot
last_entry:
entry_pad:
    .quad   entry_slurp_file
    .byte   0
    .byte   3
    .ascii  "pad"
    .align  3
    .quad   code_PAD

.end
