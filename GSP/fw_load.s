// fw_load.s -- Step 5: Load and parse GSP firmware ELF
//
// Opens the GSP firmware binary, parses ELF64 headers to find the
// .fwimage section, and copies the payload into a BAR4 allocation
// (the WPR region).  Returns offsets and physical address needed
// by the FMC bootstrap (Step 6).
//
// No libc -- raw syscalls only.
//
// Build:
//   as -o fw_load.o fw_load.s

.equ SYS_READ,      63
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

// ELF64 header offsets
.equ E_SHOFF,       0x28    // section header table file offset (8 bytes)
.equ E_SHENTSIZE,   0x3A    // section header entry size (2 bytes)
.equ E_SHNUM,       0x3C    // number of section headers (2 bytes)
.equ E_SHSTRNDX,    0x3E    // section name string table index (2 bytes)

// ELF64 section header offsets (within each Elf64_Shdr)
.equ SH_NAME,       0x00    // offset into shstrtab (4 bytes)
.equ SH_OFFSET,     0x18    // section file offset (8 bytes)
.equ SH_SIZE,       0x20    // section size (8 bytes)

// ============================================================
// Data section
// ============================================================

.data

.align 3
fw_path:    .asciz "/lib/firmware/nvidia/580.105.08/gsp_ga10x.bin"
fwimage_name: .asciz ".fwimage"

// Results -- filled by gsp_fw_load, read by caller
.align 3
.globl fw_bar4_cpu          // cpu addr of firmware copy in BAR4
fw_bar4_cpu:    .quad 0
.globl fw_bar4_phys         // physical addr (== GPU VA) of firmware in BAR4
fw_bar4_phys:   .quad 0
.globl fw_image_size        // total size of .fwimage section
fw_image_size:  .quad 0
.globl fw_code_offset       // code segment offset within .fwimage
fw_code_offset: .quad 0
.globl fw_code_size         // code segment size
fw_code_size:   .quad 0
.globl fw_data_offset       // data segment offset within .fwimage
fw_data_offset: .quad 0
.globl fw_data_size         // data segment size
fw_data_size:   .quad 0
.globl fw_manifest_offset   // manifest offset within .fwimage
fw_manifest_offset: .quad 0

// Error messages
msg_fw_open_err:  .asciz "gsp: failed to open firmware file\n"
msg_fw_open_err_len = . - msg_fw_open_err - 1
msg_fw_elf_err:   .asciz "gsp: bad ELF magic\n"
msg_fw_elf_err_len = . - msg_fw_elf_err - 1
msg_fw_sec_err:   .asciz "gsp: .fwimage section not found\n"
msg_fw_sec_err_len = . - msg_fw_sec_err - 1
msg_fw_lseek_err: .asciz "gsp: lseek(SEEK_END) failed or firmware file is empty\n"
msg_fw_lseek_err_len = . - msg_fw_lseek_err - 1
msg_fw_mmap_err:  .asciz "gsp: mmap of firmware file failed (MAP_FAILED)\n"
msg_fw_mmap_err_len = . - msg_fw_mmap_err - 1
msg_fw_too_small: .asciz "gsp: firmware file smaller than ELF64 header (64 bytes)\n"
msg_fw_too_small_len = . - msg_fw_too_small - 1
msg_fw_ok:        .asciz "gsp: firmware loaded to BAR4\n"
msg_fw_ok_len = . - msg_fw_ok - 1

// ============================================================
// Text section
// ============================================================

.text

// ---------------------------------------------------------------
// gsp_fw_load -- load GSP firmware into BAR4
//
// Requires: hbm_alloc_init has been called (BAR4 allocator ready)
//
// Sequence:
//   1. Open firmware ELF file
//   2. mmap the file into process memory
//   3. Verify ELF64 magic
//   4. Parse section headers, find .fwimage by name
//   5. Allocate BAR4 region via hbm_alloc
//   6. Copy .fwimage payload to BAR4
//   7. Extract code/data/manifest offsets from .fwimage header
//   8. Store results, close file
//
// Output:
//   x0 = BAR4 physical address of firmware image
//   x1 = total .fwimage size
//   (detailed results in fw_* globals)
//
// Clobbers: x0-x15, x30 (calls hbm_alloc)
// ---------------------------------------------------------------
.globl gsp_fw_load
gsp_fw_load:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    // ---- 1. Open firmware file ----
    mov     x8, SYS_OPENAT
    mov     x0, AT_FDCWD
    adrp    x1, fw_path
    add     x1, x1, :lo12:fw_path
    mov     x2, O_RDONLY
    mov     x3, #0
    svc     #0
    cmp     x0, #0
    b.lt    .fw_open_fail
    mov     x19, x0                 // x19 = firmware fd

    // ---- 2. Get file size via lseek(fd, 0, SEEK_END) ----
    mov     x8, SYS_LSEEK
    mov     x0, x19
    mov     x1, #0
    mov     x2, SEEK_END
    svc     #0
    // Check lseek result: negative = error, zero = empty file.
    // Treat x0 <= 0 as fatal.
    cmp     x0, #0
    b.le    .fw_lseek_fail
    mov     x20, x0                 // x20 = file size

    // Sanity: file must be at least an ELF64 header (64 bytes).
    cmp     x20, #64
    b.lt    .fw_elf_too_small

    // ---- 3. mmap the firmware file ----
    mov     x8, SYS_MMAP
    mov     x0, #0                  // addr = NULL (kernel chooses)
    mov     x1, x20                 // length = file size
    mov     x2, PROT_READ           // prot = PROT_READ
    mov     x3, MAP_PRIVATE         // flags = MAP_PRIVATE
    mov     x4, x19                 // fd
    mov     x5, #0                  // offset = 0
    svc     #0
    // Check for MAP_FAILED: mmap returns a negative errno in [-4095, -1]
    // on failure.  Idiom from launcher.s: cmn x0, #4096 ; b.hi .fail
    // catches x0 in the error range (unsigned-above comparison).
    cmn     x0, #4096
    b.hi    .fw_mmap_fail
    mov     x21, x0                 // x21 = mmap'd file base

    // ---- 4. Verify ELF64 magic: 0x7f 'E' 'L' 'F' ----
    ldr     w0, [x21]
    mov     w1, #0x457F             // low 16 bits: 0x7f 'E'
    movk    w1, #0x464C, lsl #16   // high 16 bits: 'L' 'F'
    cmp     w0, w1                  // compare against 0x464C457F
    b.ne    .fw_elf_bad

    // ---- 5. Read ELF header fields ----
    ldr     x22, [x21, #E_SHOFF]       // x22 = e_shoff (section hdr table offset)
    ldrh    w23, [x21, #E_SHENTSIZE]   // x23 = e_shentsize (size of each shdr)
    ldrh    w24, [x21, #E_SHNUM]       // x24 = e_shnum (number of sections)
    ldrh    w25, [x21, #E_SHSTRNDX]    // x25 = e_shstrndx (index of shstrtab section)

    // ---- 6. Locate shstrtab ----
    // shstrtab_shdr = file_base + e_shoff + (e_shstrndx * e_shentsize)
    mov     x0, x25
    mul     x0, x0, x23                // shstrndx * shentsize
    add     x0, x0, x22                // + e_shoff
    add     x0, x0, x21                // + file base => shstrtab section header
    ldr     x26, [x0, #SH_OFFSET]      // x26 = shstrtab file offset
    add     x26, x26, x21              // x26 = shstrtab pointer in mmap'd file

    // ---- 7. Iterate sections, find .fwimage ----
    mov     x27, #0                     // x27 = section index
    add     x28, x21, x22              // x28 = pointer to first section header

.fw_scan_loop:
    cmp     x27, x24
    b.ge    .fw_section_not_found

    // Get section name: sh_name is a 4-byte offset into shstrtab
    ldr     w0, [x28, #SH_NAME]
    add     x0, x26, x0                // x0 = pointer to section name string

    // Compare with ".fwimage" (8 chars + null = 9 bytes)
    // Use a byte-by-byte compare loop
    adrp    x1, fwimage_name
    add     x1, x1, :lo12:fwimage_name
    bl      .strcmp8
    cbz     x0, .fw_found              // match => x0 == 0

    // Advance to next section header
    add     x28, x28, x23              // shdr += e_shentsize
    add     x27, x27, #1
    b       .fw_scan_loop

.fw_found:
    // x28 points to the matching section header
    ldr     x22, [x28, #SH_OFFSET]     // fwimage file offset
    ldr     x23, [x28, #SH_SIZE]       // fwimage size

    // Store image size
    adrp    x0, fw_image_size
    add     x0, x0, :lo12:fw_image_size
    str     x23, [x0]

    // ---- 8. Allocate BAR4 region for firmware ----
    mov     x0, x23                     // size = fwimage size
    // Preserve x22, x23 across call (callee-saved via frame)
    stp     x22, x23, [sp, #-16]!
    bl      hbm_alloc                   // returns cpu_addr in x0, gpu_va in x1
    ldp     x22, x23, [sp], #16

    mov     x24, x0                     // x24 = BAR4 cpu addr
    mov     x25, x1                     // x25 = BAR4 phys addr (gpu_va)

    // Store BAR4 addresses
    adrp    x0, fw_bar4_cpu
    add     x0, x0, :lo12:fw_bar4_cpu
    str     x24, [x0]

    adrp    x0, fw_bar4_phys
    add     x0, x0, :lo12:fw_bar4_phys
    str     x25, [x0]

    // ---- 9. Copy .fwimage from mmap'd file to BAR4 ----
    add     x0, x21, x22               // src = file_base + fwimage offset
    mov     x1, x24                     // dst = BAR4 cpu addr
    mov     x2, x23                     // len = fwimage size
    bl      .memcpy64

    // ---- 10. Extract code/data/manifest offsets from .fwimage ----
    //
    // The .fwimage section contains an NVIDIA firmware image with a
    // header structure.  The layout (from gsp_elf64_fwimage_header):
    //   +0x00:  manifest_offset  (8 bytes)
    //   +0x08:  manifest_size    (8 bytes, unused here)
    //   +0x10:  code_offset      (8 bytes)
    //   +0x18:  code_size        (8 bytes)
    //   +0x20:  data_offset      (8 bytes)
    //   +0x28:  data_size        (8 bytes)
    //
    // These offsets are relative to the start of .fwimage.
    add     x0, x21, x22               // pointer to .fwimage in mmap'd file

    ldr     x1, [x0, #0x00]            // manifest_offset
    adrp    x2, fw_manifest_offset
    add     x2, x2, :lo12:fw_manifest_offset
    str     x1, [x2]

    ldr     x1, [x0, #0x10]            // code_offset
    adrp    x2, fw_code_offset
    add     x2, x2, :lo12:fw_code_offset
    str     x1, [x2]

    ldr     x1, [x0, #0x18]            // code_size
    adrp    x2, fw_code_size
    add     x2, x2, :lo12:fw_code_size
    str     x1, [x2]

    ldr     x1, [x0, #0x20]            // data_offset
    adrp    x2, fw_data_offset
    add     x2, x2, :lo12:fw_data_offset
    str     x1, [x2]

    ldr     x1, [x0, #0x28]            // data_size
    adrp    x2, fw_data_size
    add     x2, x2, :lo12:fw_data_size
    str     x1, [x2]

    // ---- 11. Close firmware fd ----
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0

    // ---- 12. Print success ----
    mov     x8, #64                     // SYS_WRITE
    mov     x0, #2                      // stderr
    adrp    x1, msg_fw_ok
    add     x1, x1, :lo12:msg_fw_ok
    mov     x2, msg_fw_ok_len
    svc     #0

    // Return: x0 = BAR4 phys addr, x1 = fwimage size
    mov     x0, x25
    mov     x1, x23

    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// ---------------------------------------------------------------
// Error handlers
// ---------------------------------------------------------------

.fw_open_fail:
    mov     x8, #64
    mov     x0, #2
    adrp    x1, msg_fw_open_err
    add     x1, x1, :lo12:msg_fw_open_err
    mov     x2, msg_fw_open_err_len
    svc     #0
    mov     x0, #1
    mov     x8, SYS_EXIT
    svc     #0

.fw_elf_bad:
    // Print error, close fd, exit.  fd is in x19.
    // (munmap of x21 omitted: single-shot init, kernel reaps on SYS_EXIT.)
    mov     x8, #64
    mov     x0, #2
    adrp    x1, msg_fw_elf_err
    add     x1, x1, :lo12:msg_fw_elf_err
    mov     x2, msg_fw_elf_err_len
    svc     #0
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0
    mov     x0, #2
    mov     x8, SYS_EXIT
    svc     #0

.fw_section_not_found:
    // Print error, close fd, exit.  fd is in x19.
    // (munmap of x21 omitted: single-shot init, kernel reaps on SYS_EXIT.)
    mov     x8, #64
    mov     x0, #2
    adrp    x1, msg_fw_sec_err
    add     x1, x1, :lo12:msg_fw_sec_err
    mov     x2, msg_fw_sec_err_len
    svc     #0
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0
    mov     x0, #3
    mov     x8, SYS_EXIT
    svc     #0

.fw_elf_too_small:
    // Print error, close fd, exit.  fd is in x19, no mmap yet.
    mov     x8, #64
    mov     x0, #2
    adrp    x1, msg_fw_too_small
    add     x1, x1, :lo12:msg_fw_too_small
    mov     x2, msg_fw_too_small_len
    svc     #0
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0
    mov     x0, #6
    mov     x8, SYS_EXIT
    svc     #0

.fw_lseek_fail:
    // Print error, close fd, exit.  fd is in x19.
    mov     x8, #64                     // SYS_WRITE
    mov     x0, #2                      // stderr
    adrp    x1, msg_fw_lseek_err
    add     x1, x1, :lo12:msg_fw_lseek_err
    mov     x2, msg_fw_lseek_err_len
    svc     #0
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0
    mov     x0, #4
    mov     x8, SYS_EXIT
    svc     #0

.fw_mmap_fail:
    // Print error, close fd, exit.  fd is in x19.
    mov     x8, #64                     // SYS_WRITE
    mov     x0, #2                      // stderr
    adrp    x1, msg_fw_mmap_err
    add     x1, x1, :lo12:msg_fw_mmap_err
    mov     x2, msg_fw_mmap_err_len
    svc     #0
    mov     x8, SYS_CLOSE
    mov     x0, x19
    svc     #0
    mov     x0, #5
    mov     x8, SYS_EXIT
    svc     #0

// ---------------------------------------------------------------
// .strcmp8 -- compare two null-terminated strings (short strings)
//
// x0 = string A pointer
// x1 = string B pointer
//
// Returns: x0 = 0 if equal, nonzero otherwise
// Clobbers: x2, x3, x4
// ---------------------------------------------------------------
.strcmp8:
    mov     x4, #0
.strcmp8_loop:
    ldrb    w2, [x0, x4]
    ldrb    w3, [x1, x4]
    cmp     w2, w3
    b.ne    .strcmp8_ne
    cbz     w2, .strcmp8_eq          // both null => match
    add     x4, x4, #1
    b       .strcmp8_loop
.strcmp8_eq:
    mov     x0, #0
    ret
.strcmp8_ne:
    mov     x0, #1
    ret

// ---------------------------------------------------------------
// .memcpy64 -- copy x2 bytes from x0 to x1
//
// Uses 64-byte (4x16B) unrolled loop for bulk, byte loop for tail.
// x0 = src, x1 = dst, x2 = count
// Clobbers: x0, x1, x2, x3, x4, x5, x6
// ---------------------------------------------------------------
.memcpy64:
    // 64-byte bulk loop
    cmp     x2, #64
    b.lt    .memcpy_tail
.memcpy_bulk:
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    sub     x2, x2, #64
    cmp     x2, #64
    b.ge    .memcpy_bulk
.memcpy_tail:
    cbz     x2, .memcpy_done
.memcpy_byte:
    ldrb    w3, [x0], #1
    strb    w3, [x1], #1
    subs    x2, x2, #1
    b.ne    .memcpy_byte
.memcpy_done:
    ret
