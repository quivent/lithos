// test_weight_load.s -- Minimal test: load and mmap model weights
// Exercises the weight loading path without requiring CUDA context.
//
// Usage: ./test_weight_load <model_dir>

.equ SYS_READ,      63
.equ SYS_WRITE,     64
.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_LSEEK,     62
.equ SYS_MMAP,      222
.equ SYS_EXIT,      93

.equ AT_FDCWD,      -100
.equ O_RDONLY,       0
.equ SEEK_SET,       0
.equ SEEK_END,       2

.equ PROT_READ,      1
.equ MAP_PRIVATE,    2

.equ WEIGHT_SLOTS_PER_LAYER, 26

.data
msg_start:    .asciz "test_weight_load: starting\n"
msg_start_len = . - msg_start - 1
msg_weights:  .asciz "test_weight_load: loading model weights...\n"
msg_weights_len = . - msg_weights - 1
msg_shard:    .asciz "test_weight_load: mmap shard "
msg_shard_len = . - msg_shard - 1
msg_bytes:    .asciz " bytes\n"
msg_bytes_len = . - msg_bytes - 1
msg_wdone:    .asciz "test_weight_load: weight table built (1648 tensors from 4 shards)\n"
msg_wdone_len = . - msg_wdone - 1
msg_werr:     .asciz "test_weight_load: ERROR: failed to mmap shard\n"
msg_werr_len = . - msg_werr - 1
msg_idxerr:   .asciz "test_weight_load: ERROR: weight_index.bin not found or invalid\n"
msg_idxerr_len = . - msg_idxerr - 1
msg_narg:     .asciz "Usage: test_weight_load <model_dir>\n"
msg_narg_len = . - msg_narg - 1
msg_ok:       .asciz "test_weight_load: SUCCESS - all weights loaded and accessible\n"
msg_ok_len = . - msg_ok - 1
msg_embed:    .asciz "  embed_tokens: "
msg_embed_len = . - msg_embed - 1
msg_lmhead:   .asciz "  lm_head:      "
msg_lmhead_len = . - msg_lmhead - 1
msg_fnorm:    .asciz "  final_norm:   "
msg_fnorm_len = . - msg_fnorm - 1
msg_nl:       .asciz "\n"
msg_layer:    .asciz "  layer "
msg_layer_len = . - msg_layer - 1
msg_colon:    .asciz ": "
msg_colon_len = . - msg_colon - 1
msg_slot:     .asciz " slots resolved ("
msg_slot_len = . - msg_slot - 1
msg_null:     .asciz " null)\n"
msg_null_len = . - msg_null - 1
msg_verify:   .asciz "test_weight_load: verifying pointer accessibility...\n"
msg_verify_len = . - msg_verify - 1
msg_readok:   .asciz "test_weight_load: read first byte from each tensor - OK\n"
msg_readok_len = . - msg_readok - 1

widx_filename: .asciz "weight_index.bin"

.align 3
path_buf:     .space 512

.align 3
saved_argc:   .quad 0
saved_argv:   .quad 0

// Shard mmap base pointers
.align 3
shard_bases:   .space 64
shard_sizes:   .space 64
num_shards_loaded: .quad 0

// Weight index mmap
.align 3
widx_base:     .quad 0
widx_size:     .quad 0

// Global weight pointers
.align 3
global_weight_ptrs: .space 24

// Per-layer weight pointer table
.align 3
weight_ptrs:   .space 13312  // 64 * 26 * 8


.text
.global _start
.align 4

_start:
    // Save argc, argv
    ldr     x0, [sp]
    add     x1, sp, #8
    adrp    x2, saved_argc
    add     x2, x2, :lo12:saved_argc
    stp     x0, x1, [x2]

    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!

    // Check argc >= 2
    cmp     x0, #2
    b.lt    .no_args

    // Print start
    adrp    x1, msg_start
    add     x1, x1, :lo12:msg_start
    mov     x2, msg_start_len
    bl      print_msg

    // Print loading message
    adrp    x1, msg_weights
    add     x1, x1, :lo12:msg_weights
    mov     x2, msg_weights_len
    bl      print_msg

    // Call load_weights
    bl      load_weights

    // Print global pointers
    adrp    x1, msg_embed
    add     x1, x1, :lo12:msg_embed
    mov     x2, msg_embed_len
    bl      print_msg
    adrp    x0, global_weight_ptrs
    add     x0, x0, :lo12:global_weight_ptrs
    ldr     x0, [x0, #0]
    bl      print_hex
    bl      print_newline

    adrp    x1, msg_lmhead
    add     x1, x1, :lo12:msg_lmhead
    mov     x2, msg_lmhead_len
    bl      print_msg
    adrp    x0, global_weight_ptrs
    add     x0, x0, :lo12:global_weight_ptrs
    ldr     x0, [x0, #8]
    bl      print_hex
    bl      print_newline

    adrp    x1, msg_fnorm
    add     x1, x1, :lo12:msg_fnorm
    mov     x2, msg_fnorm_len
    bl      print_msg
    adrp    x0, global_weight_ptrs
    add     x0, x0, :lo12:global_weight_ptrs
    ldr     x0, [x0, #16]
    bl      print_hex
    bl      print_newline

    // Print summary per layer: count resolved slots
    mov     x19, #0               // layer_idx
.layer_summary_loop:
    cmp     x19, #64
    b.ge    .layer_summary_done

    // Only print every 8th layer to keep output compact
    tst     x19, #7
    b.ne    .layer_summary_next

    adrp    x1, msg_layer
    add     x1, x1, :lo12:msg_layer
    mov     x2, msg_layer_len
    bl      print_msg

    mov     x0, x19
    bl      print_int

    adrp    x1, msg_colon
    add     x1, x1, :lo12:msg_colon
    mov     x2, msg_colon_len
    bl      print_msg

    // Count non-null slots for this layer
    mov     x20, #0               // slot_idx
    mov     x9, #0                // resolved_count
    mov     x10, #0               // null_count
.count_slots:
    cmp     x20, #WEIGHT_SLOTS_PER_LAYER
    b.ge    .count_done

    adrp    x0, weight_ptrs
    add     x0, x0, :lo12:weight_ptrs
    mov     x1, #WEIGHT_SLOTS_PER_LAYER
    madd    x1, x19, x1, x20
    ldr     x0, [x0, x1, lsl #3]
    cbz     x0, .count_null
    add     x9, x9, #1
    b       .count_next
.count_null:
    add     x10, x10, #1
.count_next:
    add     x20, x20, #1
    b       .count_slots
.count_done:

    // Save counts and print
    stp     x19, x20, [sp, #-16]!
    stp     x9, x10, [sp, #-16]!

    mov     x0, x9
    bl      print_int

    adrp    x1, msg_slot
    add     x1, x1, :lo12:msg_slot
    mov     x2, msg_slot_len
    bl      print_msg

    ldp     x9, x10, [sp], #16
    ldp     x19, x20, [sp], #16

    stp     x19, x20, [sp, #-16]!
    mov     x0, x10
    bl      print_int

    adrp    x1, msg_null
    add     x1, x1, :lo12:msg_null
    mov     x2, msg_null_len
    bl      print_msg

    ldp     x19, x20, [sp], #16

.layer_summary_next:
    add     x19, x19, #1
    b       .layer_summary_loop

.layer_summary_done:

    // ---- Verify accessibility: touch first byte of every non-null pointer ----
    adrp    x1, msg_verify
    add     x1, x1, :lo12:msg_verify
    mov     x2, msg_verify_len
    bl      print_msg

    // Touch globals
    adrp    x0, global_weight_ptrs
    add     x0, x0, :lo12:global_weight_ptrs
    mov     x19, #0
.touch_globals:
    cmp     x19, #3
    b.ge    .touch_globals_done
    ldr     x1, [x0, x19, lsl #3]
    cbz     x1, .touch_global_next
    ldrb    w2, [x1]              // read first byte -- will segfault if bad
.touch_global_next:
    add     x19, x19, #1
    b       .touch_globals
.touch_globals_done:

    // Touch all layer weight pointers
    mov     x19, #0               // layer
.touch_layer_loop:
    cmp     x19, #64
    b.ge    .touch_done

    mov     x20, #0               // slot
.touch_slot_loop:
    cmp     x20, #WEIGHT_SLOTS_PER_LAYER
    b.ge    .touch_slot_done

    adrp    x0, weight_ptrs
    add     x0, x0, :lo12:weight_ptrs
    mov     x1, #WEIGHT_SLOTS_PER_LAYER
    madd    x1, x19, x1, x20
    ldr     x0, [x0, x1, lsl #3]
    cbz     x0, .touch_next
    ldrb    w2, [x0]              // read first byte
.touch_next:
    add     x20, x20, #1
    b       .touch_slot_loop
.touch_slot_done:
    add     x19, x19, #1
    b       .touch_layer_loop
.touch_done:

    adrp    x1, msg_readok
    add     x1, x1, :lo12:msg_readok
    mov     x2, msg_readok_len
    bl      print_msg

    // Final success
    adrp    x1, msg_ok
    add     x1, x1, :lo12:msg_ok
    mov     x2, msg_ok_len
    bl      print_msg

    // Exit 0
    mov     x0, #0
    mov     x8, #93
    svc     #0

.no_args:
    adrp    x1, msg_narg
    add     x1, x1, :lo12:msg_narg
    mov     x2, msg_narg_len
    bl      print_msg
    mov     x0, #1
    mov     x8, #93
    svc     #0


// ============================================================
// load_weights -- identical to the one in launcher.s
// ============================================================
.align 4
load_weights:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    ldr     x19, [x0]
    cmp     x19, #2
    b.lt    .w_done

    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    ldr     x0, [x0]
    ldr     x20, [x0, #8]         // argv[1] = model_dir

    // Build path: model_dir / weight_index.bin
    adrp    x4, path_buf
    add     x4, x4, :lo12:path_buf
    mov     x5, x4
    mov     x0, x20
.w_copy_dir:
    ldrb    w6, [x0], #1
    cbz     w6, .w_dir_done
    strb    w6, [x4], #1
    b       .w_copy_dir
.w_dir_done:
    mov     w6, #'/'
    strb    w6, [x4], #1
    adrp    x0, widx_filename
    add     x0, x0, :lo12:widx_filename
.w_copy_name:
    ldrb    w6, [x0], #1
    strb    w6, [x4], #1
    cbnz    w6, .w_copy_name

    // Open weight_index.bin
    mov     x0, #AT_FDCWD
    mov     x1, x5
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .w_idx_not_found
    mov     x21, x0

    // lseek to get size
    mov     x1, #0
    mov     x2, #SEEK_END
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x22, x0

    // mmap index
    mov     x0, #0
    mov     x1, x22
    mov     x2, #PROT_READ
    mov     x3, #MAP_PRIVATE
    mov     x4, x21
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #1
    b.eq    .w_idx_mmap_fail
    mov     x23, x0

    adrp    x1, widx_base
    add     x1, x1, :lo12:widx_base
    stp     x23, x22, [x1]

    // Close index fd
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0

    // Validate magic = 0x4C575449
    ldr     w0, [x23]
    mov     w1, #0x5449
    movk    w1, #0x4C57, lsl #16
    cmp     w0, w1
    b.ne    .w_idx_not_found

    ldr     w24, [x23, #8]         // num_shards

    // Shard table at offset 32
    add     x25, x23, #32
    mov     x26, #0

.w_shard_loop:
    cmp     w26, w24
    b.ge    .w_shards_done

    // Build shard path
    adrp    x4, path_buf
    add     x4, x4, :lo12:path_buf
    mov     x5, x4
    mov     x0, x20
.ws_copy_dir:
    ldrb    w6, [x0], #1
    cbz     w6, .ws_dir_done
    strb    w6, [x4], #1
    b       .ws_copy_dir
.ws_dir_done:
    mov     w6, #'/'
    strb    w6, [x4], #1
    mov     x0, #264
    mul     x0, x26, x0
    add     x0, x25, x0
    mov     x27, x0
.ws_copy_name:
    ldrb    w6, [x0], #1
    strb    w6, [x4], #1
    cbnz    w6, .ws_copy_name

    ldr     x28, [x27, #256]      // file_size

    // Print shard info
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    adrp    x1, msg_shard
    add     x1, x1, :lo12:msg_shard
    mov     x2, msg_shard_len
    bl      print_msg

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    mov     x0, x26
    bl      print_int
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    sub     sp, sp, #16
    mov     w0, #' '
    strb    w0, [sp]
    mov     x1, sp
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    mov     x0, x28
    bl      print_int
    adrp    x1, msg_bytes
    add     x1, x1, :lo12:msg_bytes
    mov     x2, msg_bytes_len
    bl      print_msg
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    // Open shard (reload path_buf, x5 clobbered by prints)
    mov     x0, #AT_FDCWD
    adrp    x1, path_buf
    add     x1, x1, :lo12:path_buf
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .w_shard_fail
    mov     x21, x0

    // mmap shard
    mov     x0, #0
    mov     x1, x28
    mov     x2, #PROT_READ
    mov     x3, #MAP_PRIVATE
    mov     x4, x21
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #1
    b.eq    .w_shard_fail

    adrp    x1, shard_bases
    add     x1, x1, :lo12:shard_bases
    str     x0, [x1, x26, lsl #3]

    adrp    x1, shard_sizes
    add     x1, x1, :lo12:shard_sizes
    str     x28, [x1, x26, lsl #3]

    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0

    add     x26, x26, #1
    b       .w_shard_loop

.w_shards_done:
    adrp    x0, num_shards_loaded
    add     x0, x0, :lo12:num_shards_loaded
    str     x26, [x0]

    // Resolve globals
    mov     x0, #264
    umull   x0, w24, w0
    add     x0, x0, #32
    add     x25, x23, x0          // global_table_ptr

    ldr     w26, [x23, #20]       // num_globals
    adrp    x27, global_weight_ptrs
    add     x27, x27, :lo12:global_weight_ptrs

    mov     x21, #0
.w_global_loop:
    cmp     w21, w26
    b.ge    .w_globals_done
    mov     x0, #24
    mul     x0, x21, x0
    add     x0, x25, x0
    ldr     w1, [x0]
    ldr     x2, [x0, #8]
    ldr     x3, [x0, #16]
    orr     x4, x2, x3
    cbz     x4, .w_global_null
    adrp    x4, shard_bases
    add     x4, x4, :lo12:shard_bases
    ldr     x4, [x4, x1, lsl #3]
    add     x4, x4, x2
    str     x4, [x27, x21, lsl #3]
    b       .w_global_next
.w_global_null:
    str     xzr, [x27, x21, lsl #3]
.w_global_next:
    add     x21, x21, #1
    b       .w_global_loop

.w_globals_done:
    // Resolve per-layer
    mov     x0, #24
    umull   x0, w26, w0
    add     x25, x25, x0          // layer_table_ptr

    ldr     w21, [x23, #12]       // num_layers
    ldr     w22, [x23, #16]       // slots_per_layer

    adrp    x27, weight_ptrs
    add     x27, x27, :lo12:weight_ptrs

    mov     x19, #0
.w_layer_loop:
    cmp     w19, w21
    b.ge    .w_layers_done
    mov     x28, #0

.w_slot_loop:
    cmp     w28, w22
    b.ge    .w_slots_done

    umull   x0, w19, w22
    add     x0, x0, x28
    mov     x1, #24
    mul     x0, x0, x1
    add     x0, x25, x0

    ldr     w1, [x0]
    ldr     x2, [x0, #8]
    ldr     x3, [x0, #16]
    orr     x4, x2, x3
    cbz     x4, .w_slot_null

    adrp    x4, shard_bases
    add     x4, x4, :lo12:shard_bases
    ldr     x4, [x4, x1, lsl #3]
    add     x4, x4, x2

    mov     x5, #WEIGHT_SLOTS_PER_LAYER
    umull   x5, w19, w5
    add     x5, x5, x28
    str     x4, [x27, x5, lsl #3]
    b       .w_slot_next

.w_slot_null:
    mov     x5, #WEIGHT_SLOTS_PER_LAYER
    umull   x5, w19, w5
    add     x5, x5, x28
    str     xzr, [x27, x5, lsl #3]

.w_slot_next:
    add     x28, x28, #1
    b       .w_slot_loop

.w_slots_done:
    add     x19, x19, #1
    b       .w_layer_loop

.w_layers_done:
    adrp    x1, msg_wdone
    add     x1, x1, :lo12:msg_wdone
    mov     x2, msg_wdone_len
    bl      print_msg
    b       .w_done

.w_idx_not_found:
    adrp    x1, msg_idxerr
    add     x1, x1, :lo12:msg_idxerr
    mov     x2, msg_idxerr_len
    bl      print_msg
    b       .w_done

.w_idx_mmap_fail:
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0
    b       .w_idx_not_found

.w_shard_fail:
    adrp    x1, msg_werr
    add     x1, x1, :lo12:msg_werr
    mov     x2, msg_werr_len
    bl      print_msg
    add     x26, x26, #1
    b       .w_shard_loop

.w_done:
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret


// ============================================================
// Utility functions
// ============================================================

print_msg:
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ret

print_newline:
    stp     x29, x30, [sp, #-16]!
    adrp    x1, msg_nl
    add     x1, x1, :lo12:msg_nl
    mov     x2, #1
    bl      print_msg
    ldp     x29, x30, [sp], #16
    ret

print_int:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32
    mov     x1, sp
    add     x1, x1, #31
    mov     x2, #0
    mov     x3, #10
    cbz     x0, .pi_zero
.pi_loop:
    cbz     x0, .pi_digits
    udiv    x4, x0, x3
    msub    x5, x4, x3, x0
    add     x5, x5, #'0'
    strb    w5, [x1], #-1
    add     x2, x2, #1
    mov     x0, x4
    b       .pi_loop
.pi_digits:
    add     x1, x1, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret
.pi_zero:
    mov     w5, #'0'
    strb    w5, [x1]
    mov     x2, #1
    b       .pi_digits

// print_hex -- print x0 as 0x... hex
print_hex:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32
    // Print "0x" prefix
    mov     w1, #'0'
    strb    w1, [sp]
    mov     w1, #'x'
    strb    w1, [sp, #1]
    mov     x1, sp
    mov     x2, #2
    stp     x0, xzr, [sp, #16]    // save x0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #16]         // restore

    // Convert to hex digits (16 nibbles)
    mov     x1, sp
    add     x1, x1, #15           // end of 16-char buffer
    mov     x2, #16               // always 16 hex digits
    mov     x3, #0
.ph_loop:
    cmp     x3, #16
    b.ge    .ph_print
    and     x4, x0, #0xF
    cmp     x4, #10
    b.lt    .ph_digit
    add     x4, x4, #('a' - 10)
    b       .ph_store
.ph_digit:
    add     x4, x4, #'0'
.ph_store:
    strb    w4, [x1], #-1
    lsr     x0, x0, #4
    add     x3, x3, #1
    b       .ph_loop
.ph_print:
    add     x1, x1, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret
