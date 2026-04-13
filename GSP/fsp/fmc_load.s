// fmc_load.s -- Load extracted GSP-FMC bindata files into BAR4
//
// Opens four pre-extracted FMC blob files from disk, allocates BAR4
// regions via hbm_alloc, reads file contents into the allocations,
// and exports CPU VA / GPU PA / size for each blob.
//
// No libc -- raw syscalls only.
//
// Build:
//   as -I /home/ubuntu/lithos/GSP/fsp -o fmc_load.o fmc_load.s

.equ SYS_READ,      63
.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_LSEEK,     62
.equ SYS_WRITE,     64

.equ AT_FDCWD,      -100
.equ O_RDONLY,       0
.equ SEEK_SET,       0
.equ SEEK_END,       2

// Minimum expected sizes for validation
.equ FMC_IMAGE_MIN_SIZE,    165448      // gsp_fmc_image.bin
.equ FMC_HASH_MIN_SIZE,     48          // SHA-384 = 48 bytes
.equ FMC_PUBKEY_MIN_SIZE,   384         // RSA-3072 = 384 bytes
.equ FMC_SIG_MIN_SIZE,      384         // RSA-3072 = 384 bytes

// ============================================================
// Data section
// ============================================================

.data

.align 3

// --- File paths (relative, run from lithos directory) ---
fmc_image_path:     .asciz "GSP/fw/gsp_fmc_image.bin"
fmc_hash_path:      .asciz "GSP/fw/gsp_fmc_hash.bin"
fmc_pubkey_path:    .asciz "GSP/fw/gsp_fmc_pubkey.bin"
fmc_sig_path:       .asciz "GSP/fw/gsp_fmc_sig.bin"

// --- Exported result variables ---

.align 3
.globl fmc_image_cpu
fmc_image_cpu:      .quad 0
.globl fmc_image_phys
fmc_image_phys:     .quad 0
.globl fmc_image_size
fmc_image_size:     .quad 0

.globl fmc_hash_cpu
fmc_hash_cpu:       .quad 0
.globl fmc_hash_phys
fmc_hash_phys:      .quad 0
.globl fmc_hash_size
fmc_hash_size:      .quad 0

.globl fmc_pubkey_cpu
fmc_pubkey_cpu:     .quad 0
.globl fmc_pubkey_phys
fmc_pubkey_phys:    .quad 0
.globl fmc_pubkey_size
fmc_pubkey_size:    .quad 0

.globl fmc_sig_cpu
fmc_sig_cpu:        .quad 0
.globl fmc_sig_phys
fmc_sig_phys:       .quad 0
.globl fmc_sig_size
fmc_sig_size:       .quad 0

// --- Error / status messages ---
msg_fmc_open_err:   .asciz "fmc: failed to open "
msg_fmc_open_err_len = . - msg_fmc_open_err - 1
msg_fmc_lseek_err:  .asciz "fmc: lseek failed on "
msg_fmc_lseek_err_len = . - msg_fmc_lseek_err - 1
msg_fmc_small_err:  .asciz "fmc: file too small: "
msg_fmc_small_err_len = . - msg_fmc_small_err - 1
msg_fmc_read_err:   .asciz "fmc: read failed on "
msg_fmc_read_err_len = . - msg_fmc_read_err - 1
msg_newline:        .asciz "\n"
msg_fmc_ok:         .asciz "fmc: all FMC blobs loaded to BAR4\n"
msg_fmc_ok_len = . - msg_fmc_ok - 1

// ============================================================
// Text section
// ============================================================

.text

// ---------------------------------------------------------------
// _load_one_blob -- load a single file into a BAR4 allocation
//
// Input:
//   x0 = pointer to null-terminated file path
//   x1 = minimum expected size (bytes)
//
// Output (on success):
//   x0 = cpu_va   (BAR4 mmap'd virtual address)
//   x1 = phys_va  (BAR4 physical address = GPU VA)
//   x2 = file size (bytes actually read)
//
// Output (on failure):
//   x0 = negative errno
//
// Clobbers: x0-x15, x30 (calls hbm_alloc)
// ---------------------------------------------------------------
_load_one_blob:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]

    mov     x19, x0                 // x19 = path pointer
    mov     x20, x1                 // x20 = min_size

    // ---- 1. Open file ----
    mov     x8, SYS_OPENAT
    mov     x0, AT_FDCWD
    mov     x1, x19
    mov     x2, O_RDONLY
    mov     x3, #0
    svc     #0
    cmp     x0, #0
    b.lt    .lob_open_fail
    mov     x21, x0                 // x21 = fd

    // ---- 2. Get file size via lseek(fd, 0, SEEK_END) ----
    mov     x8, SYS_LSEEK
    mov     x0, x21
    mov     x1, #0
    mov     x2, SEEK_END
    svc     #0
    cmp     x0, #0
    b.le    .lob_lseek_fail
    mov     x22, x0                 // x22 = file_size

    // ---- 3. Validate size >= min_size ----
    cmp     x22, x20
    b.lt    .lob_too_small

    // ---- 4. Seek back to start ----
    mov     x8, SYS_LSEEK
    mov     x0, x21
    mov     x1, #0
    mov     x2, SEEK_SET
    svc     #0

    // ---- 5. Allocate BAR4 region via hbm_alloc ----
    mov     x0, x22                 // size = file_size
    bl      hbm_alloc               // returns cpu_addr in x0, gpu_va in x1
    mov     x23, x0                 // x23 = cpu_va (BAR4 mmap'd)
    mov     x24, x1                 // x24 = phys_va (GPU VA)

    // ---- 6. Read file contents into BAR4 allocation ----
    // Read in a loop to handle partial reads
    mov     x25, #0                 // x25 = total bytes read
.lob_read_loop:
    mov     x8, SYS_READ
    mov     x0, x21                 // fd
    add     x1, x23, x25           // buf = cpu_va + bytes_read_so_far
    sub     x2, x22, x25           // count = remaining
    svc     #0
    cmp     x0, #0
    b.lt    .lob_read_fail
    cbz     x0, .lob_read_done     // EOF
    add     x25, x25, x0           // total += bytes_read
    cmp     x25, x22
    b.lt    .lob_read_loop

.lob_read_done:
    // ---- 7. Close fd ----
    mov     x8, SYS_CLOSE
    mov     x0, x21
    svc     #0

    // ---- 8. Return success ----
    mov     x0, x23                 // cpu_va
    mov     x1, x24                 // phys_va
    mov     x2, x22                 // size

    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

// ---- Error paths ----

.lob_open_fail:
    mov     x25, x0                 // save errno
    // Print: "fmc: failed to open <path>\n"
    mov     x8, SYS_WRITE
    mov     x0, #2                  // stderr
    adrp    x1, msg_fmc_open_err
    add     x1, x1, :lo12:msg_fmc_open_err
    mov     x2, msg_fmc_open_err_len
    svc     #0
    bl      .lob_print_path
    mov     x0, x25
    b       .lob_epilogue

.lob_lseek_fail:
    mov     x25, x0
    mov     x8, SYS_WRITE
    mov     x0, #2
    adrp    x1, msg_fmc_lseek_err
    add     x1, x1, :lo12:msg_fmc_lseek_err
    mov     x2, msg_fmc_lseek_err_len
    svc     #0
    bl      .lob_print_path
    // Close fd before returning
    mov     x8, SYS_CLOSE
    mov     x0, x21
    svc     #0
    mov     x0, x25
    cmp     x0, #0
    csel    x0, x0, x25, lt         // ensure negative
    mov     x0, #-1                 // force negative
    b       .lob_epilogue

.lob_too_small:
    mov     x8, SYS_WRITE
    mov     x0, #2
    adrp    x1, msg_fmc_small_err
    add     x1, x1, :lo12:msg_fmc_small_err
    mov     x2, msg_fmc_small_err_len
    svc     #0
    bl      .lob_print_path
    mov     x8, SYS_CLOSE
    mov     x0, x21
    svc     #0
    mov     x0, #-1
    b       .lob_epilogue

.lob_read_fail:
    mov     x25, x0                 // save errno
    mov     x8, SYS_WRITE
    mov     x0, #2
    adrp    x1, msg_fmc_read_err
    add     x1, x1, :lo12:msg_fmc_read_err
    mov     x2, msg_fmc_read_err_len
    svc     #0
    bl      .lob_print_path
    mov     x8, SYS_CLOSE
    mov     x0, x21
    svc     #0
    mov     x0, x25
    b       .lob_epilogue

.lob_epilogue:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

// ---------------------------------------------------------------
// .lob_print_path -- print x19 (path) + newline to stderr
// Clobbers: x0-x3, x8
// ---------------------------------------------------------------
.lob_print_path:
    // Compute strlen of x19
    mov     x2, #0
    mov     x3, x19
.lob_strlen:
    ldrb    w0, [x3, x2]
    cbz     w0, .lob_strlen_done
    add     x2, x2, #1
    b       .lob_strlen
.lob_strlen_done:
    mov     x8, SYS_WRITE
    mov     x0, #2
    mov     x1, x19
    // x2 = length already set
    svc     #0
    // newline
    mov     x8, SYS_WRITE
    mov     x0, #2
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    svc     #0
    ret

// ---------------------------------------------------------------
// fmc_load_blobs -- load all four FMC blobs into BAR4
//
// Requires: hbm_alloc_init has been called (BAR4 allocator ready)
//
// Output:
//   x0 = 0 on success, negative on failure
//
// Side effects: populates fmc_{image,hash,pubkey,sig}_{cpu,phys,size}
//
// Clobbers: x0-x15, x30
// ---------------------------------------------------------------
.globl fmc_load_blobs
fmc_load_blobs:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // ---- 1. Load gsp_fmc_image.bin ----
    adrp    x0, fmc_image_path
    add     x0, x0, :lo12:fmc_image_path
    mov     x1, #(FMC_IMAGE_MIN_SIZE & 0xFFFF)
    movk    x1, #(FMC_IMAGE_MIN_SIZE >> 16), lsl #16
    bl      _load_one_blob
    cmp     x0, #0
    b.lt    .fmc_fail

    adrp    x3, fmc_image_cpu
    add     x3, x3, :lo12:fmc_image_cpu
    str     x0, [x3]
    adrp    x3, fmc_image_phys
    add     x3, x3, :lo12:fmc_image_phys
    str     x1, [x3]
    adrp    x3, fmc_image_size
    add     x3, x3, :lo12:fmc_image_size
    str     x2, [x3]

    // ---- 2. Load gsp_fmc_hash.bin ----
    adrp    x0, fmc_hash_path
    add     x0, x0, :lo12:fmc_hash_path
    mov     x1, FMC_HASH_MIN_SIZE
    bl      _load_one_blob
    cmp     x0, #0
    b.lt    .fmc_fail

    adrp    x3, fmc_hash_cpu
    add     x3, x3, :lo12:fmc_hash_cpu
    str     x0, [x3]
    adrp    x3, fmc_hash_phys
    add     x3, x3, :lo12:fmc_hash_phys
    str     x1, [x3]
    adrp    x3, fmc_hash_size
    add     x3, x3, :lo12:fmc_hash_size
    str     x2, [x3]

    // ---- 3. Load gsp_fmc_pubkey.bin ----
    adrp    x0, fmc_pubkey_path
    add     x0, x0, :lo12:fmc_pubkey_path
    mov     x1, FMC_PUBKEY_MIN_SIZE
    bl      _load_one_blob
    cmp     x0, #0
    b.lt    .fmc_fail

    adrp    x3, fmc_pubkey_cpu
    add     x3, x3, :lo12:fmc_pubkey_cpu
    str     x0, [x3]
    adrp    x3, fmc_pubkey_phys
    add     x3, x3, :lo12:fmc_pubkey_phys
    str     x1, [x3]
    adrp    x3, fmc_pubkey_size
    add     x3, x3, :lo12:fmc_pubkey_size
    str     x2, [x3]

    // ---- 4. Load gsp_fmc_sig.bin ----
    adrp    x0, fmc_sig_path
    add     x0, x0, :lo12:fmc_sig_path
    mov     x1, FMC_SIG_MIN_SIZE
    bl      _load_one_blob
    cmp     x0, #0
    b.lt    .fmc_fail

    adrp    x3, fmc_sig_cpu
    add     x3, x3, :lo12:fmc_sig_cpu
    str     x0, [x3]
    adrp    x3, fmc_sig_phys
    add     x3, x3, :lo12:fmc_sig_phys
    str     x1, [x3]
    adrp    x3, fmc_sig_size
    add     x3, x3, :lo12:fmc_sig_size
    str     x2, [x3]

    // ---- Success: print message, return 0 ----
    mov     x8, SYS_WRITE
    mov     x0, #2                  // stderr
    adrp    x1, msg_fmc_ok
    add     x1, x1, :lo12:msg_fmc_ok
    mov     x2, msg_fmc_ok_len
    svc     #0

    mov     x0, #0
    ldp     x29, x30, [sp], #16
    ret

.fmc_fail:
    // x0 already negative
    ldp     x29, x30, [sp], #16
    ret
