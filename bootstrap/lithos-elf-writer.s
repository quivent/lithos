// lithos-elf-writer.s — ELF writer for the Lithos bootstrap compiler
//
// Produces two kinds of ELF64 binaries:
//   1. ARM64 Linux executable (ET_EXEC, EM_AARCH64, PT_LOAD at 0x400000)
//   2. GPU cubin (ET_EXEC, EM_CUDA, 9-section layout for cuModuleLoadData)
//
// Designed to be included into lithos-bootstrap.s or assembled standalone.
// Uses the same DTC register conventions:
//   X26=IP  X25=W  X24=DSP  X23=RSP  X22=TOS  X20=HERE  X21=BASE
//
// Build (standalone test):
//   as -o lithos-elf-writer.o lithos-elf-writer.s
//   ld -o lithos-elf-writer lithos-elf-writer.o
//
// All numeric constants are decimal unless noted.

// ============================================================
// DTC NEXT macro (must be defined before use)
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
// Syscall numbers (ARM64 Linux)
// ============================================================
.equ SYS_READ,    63
.equ SYS_WRITE,   64
.equ SYS_OPENAT,  56
.equ SYS_CLOSE,   57
.equ SYS_EXIT,    93
.equ AT_FDCWD,    -100

// open flags: O_WRONLY|O_CREAT|O_TRUNC = 0x241 = 577
.equ O_WCT,       577
// mode 0644 = 420
.equ MODE_644,    420

// ============================================================
// ELF constants
// ============================================================
.equ ELFCLASS64,      2
.equ ELFDATA2LSB,     1
.equ EV_CURRENT,      1
.equ ELFOSABI_NONE,   0
.equ ELFOSABI_CUDA,   51       // 0x33

.equ ET_EXEC,         2
.equ EM_AARCH64,      183      // 0xB7
.equ EM_CUDA,         190      // 0xBE
.equ EV_CUDA,         128      // 0x80

.equ EHDR_SIZE,       64
.equ PHDR_SIZE,       56
.equ SHDR_SIZE,       64
.equ SYM_SIZE,        24

.equ PT_LOAD,         1
.equ PF_X,            1
.equ PF_W,            2
.equ PF_R,            4

.equ SHT_NULL,        0
.equ SHT_PROGBITS,    1
.equ SHT_SYMTAB,      2
.equ SHT_STRTAB,      3
.equ SHT_NOBITS,      8
.equ SHT_LOPROC,      0x70000000

.equ SHF_WRITE,       1
.equ SHF_ALLOC,       2
.equ SHF_EXECINSTR,   4
.equ SHF_INFO_LINK,   0x40

.equ STB_LOCAL,       0
.equ STB_GLOBAL,      1
.equ STT_NOTYPE,      0
.equ STT_FUNC,        2
.equ STT_SECTION,     3
.equ STO_CUDA_ENTRY,  0x10

// CUDA ELF flags (from probe.cubin)
.equ ELF_FLAGS_CUDA,  0x5a055a

// nv.info attribute IDs
.equ EIATTR_REGCOUNT,              0x2f
.equ EIATTR_FRAME_SIZE,            0x11
.equ EIATTR_MIN_STACK_SIZE,        0x12
.equ EIATTR_CUDA_API_VERSION,      0x37
.equ EIATTR_KPARAM_INFO,           0x17
.equ EIATTR_PARAM_CBANK,           0x0a
.equ EIATTR_EXIT_INSTR_OFFSETS,    0x1c
.equ EIATTR_CBANK_PARAM_SIZE,      0x19
.equ EIATTR_MAXREG_COUNT,          0x1b
.equ EIATTR_SPARSE_MMA_MASK,       0x50
.equ EIATTR_SW_WAR,                0x36
.equ EIATTR_CRS_STACK_SIZE,        0x33
.equ EIATTR_COOP_GROUP_INSTR_OFF,  0x28
.equ EIATTR_COOP_GROUP_MASK_REG,   0x29

.equ NVI_FMT_FLAG,    3
.equ NVI_FMT_U32,     4

// ARM64 executable load address
.equ BASE_ADDR,       0x400000
.equ PAGE_ALIGN,      0x10000

// Buffer sizes
.equ ELF_BUF_SIZE,    524288   // 512KB output buffer

// ============================================================
// BSS — ls_elf_buf / ls_elf_pos live in ls-shared.s; section
// metadata below is writer-private scratch.
// ============================================================
.extern ls_elf_buf
.extern ls_elf_pos

.bss
.align 4

// Section offset/size pairs (ARM64 mode)
arm_text_off:   .space 8
arm_text_size:  .space 8
arm_data_off:   .space 8
arm_data_size:  .space 8
arm_bss_off:    .space 8
arm_bss_size:   .space 8

// Section offset/size pairs (cubin mode)
cu_shstrtab_off:   .space 8
cu_shstrtab_size:  .space 8
cu_strtab_off:     .space 8
cu_strtab_size:    .space 8
cu_symtab_off:     .space 8
cu_symtab_size:    .space 8
cu_nvinfo_off:     .space 8
cu_nvinfo_size:    .space 8
cu_nvinfo_k_off:   .space 8
cu_nvinfo_k_size:  .space 8
cu_text_off:       .space 8
cu_text_size:      .space 8
cu_const0_off:     .space 8
cu_const0_size:    .space 8
cu_shdrs_off:      .space 8
cu_sym_kernel_off: .space 8
cu_strsym_kernel:  .space 8

// Section name offsets within .shstrtab (cubin)
cu_SN_shstrtab:    .space 8
cu_SN_strtab:      .space 8
cu_SN_symtab:      .space 8
cu_SN_nvinfo:      .space 8
cu_SN_nvinfo_k:    .space 8
cu_SN_text:        .space 8
cu_SN_const0:      .space 8
cu_SN_shared:      .space 8

// Temporary: file descriptor for write
elf_fd:         .space 8

// ============================================================
// TEXT — code
// ============================================================
.text
.align 4

// ============================================================
// Byte emit primitives
// ============================================================
// All primitives use ls_elf_buf / ls_elf_pos as implicit state.
// Callee-saved registers preserved; scratch in x0-x5, x8-x9.

// eb_emit: store one byte at ls_elf_buf[ls_elf_pos], increment ls_elf_pos
// x0 = byte value (low 8 bits)
.align 4
eb_emit:
    adrp    x1, ls_elf_pos
    add     x1, x1, :lo12:ls_elf_pos
    ldr     x2, [x1]               // x2 = pos
    adrp    x3, ls_elf_buf
    add     x3, x3, :lo12:ls_elf_buf
    strb    w0, [x3, x2]
    add     x2, x2, #1
    str     x2, [x1]
    ret

// ew_emit: store u16 little-endian
// x0 = u16 value
.align 4
ew_emit:
    stp     x30, x0, [sp, #-16]!
    and     x0, x0, #0xff
    ldr     x5, [sp, #8]
    and     x0, x5, #0xff
    bl      eb_emit
    ldr     x5, [sp, #8]
    lsr     x0, x5, #8
    and     x0, x0, #0xff
    bl      eb_emit
    ldp     x30, x0, [sp], #16
    ret

// ed_emit: store u32 little-endian
// x0 = u32 value
.align 4
ed_emit:
    stp     x30, x0, [sp, #-16]!
    // byte 0
    and     x0, x0, #0xff
    ldr     x5, [sp, #8]
    and     x0, x5, #0xff
    bl      eb_emit
    ldr     x5, [sp, #8]
    lsr     x0, x5, #8
    and     x0, x0, #0xff
    bl      eb_emit
    ldr     x5, [sp, #8]
    lsr     x0, x5, #16
    and     x0, x0, #0xff
    bl      eb_emit
    ldr     x5, [sp, #8]
    lsr     x0, x5, #24
    and     x0, x0, #0xff
    bl      eb_emit
    ldp     x30, x0, [sp], #16
    ret

// eq_emit: store u64 little-endian
// x0 = u64 value
.align 4
eq_emit:
    stp     x30, x0, [sp, #-16]!
    bl      ed_emit                 // low 32 bits
    ldr     x0, [sp, #8]
    lsr     x0, x0, #32
    bl      ed_emit                 // high 32 bits
    ldp     x30, x0, [sp], #16
    ret

// epad_to: zero-fill ls_elf_buf from ls_elf_pos to target offset
// x0 = target offset
.align 4
epad_to:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x0                 // target
1:  adrp    x1, ls_elf_pos
    add     x1, x1, :lo12:ls_elf_pos
    ldr     x2, [x1]
    cmp     x2, x19
    b.ge    2f
    mov     x0, #0
    bl      eb_emit
    b       1b
2:  ldp     x30, x19, [sp], #16
    ret

// ealign: align ls_elf_pos to n-byte boundary
// x0 = alignment (must be power of 2)
.align 4
ealign:
    stp     x30, x19, [sp, #-16]!
    sub     x19, x0, #1            // mask = n-1
    adrp    x1, ls_elf_pos
    add     x1, x1, :lo12:ls_elf_pos
    ldr     x2, [x1]
    add     x2, x2, x19
    mvn     x3, x19
    and     x0, x2, x3             // target = (pos + mask) & ~mask
    bl      epad_to
    ldp     x30, x19, [sp], #16
    ret

// emit_bytes: copy len bytes from src into ls_elf_buf
// x0 = src pointer, x1 = len
.align 4
emit_bytes:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x19, x0                 // src
    mov     x20, x1                 // len
    mov     x21, #0                 // i
1:  cmp     x21, x20
    b.ge    2f
    ldrb    w0, [x19, x21]
    bl      eb_emit
    add     x21, x21, #1
    b       1b
2:  ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// Patch helpers — write values at absolute buffer offsets
// ============================================================

// put_u16: store u16 at ls_elf_buf[off]
// x0 = val, x1 = off
.align 4
put_u16:
    adrp    x2, ls_elf_buf
    add     x2, x2, :lo12:ls_elf_buf
    strb    w0, [x2, x1]
    lsr     x3, x0, #8
    add     x4, x1, #1
    strb    w3, [x2, x4]
    ret

// put_u32: store u32 at ls_elf_buf[off]
// x0 = val, x1 = off
.align 4
put_u32:
    adrp    x2, ls_elf_buf
    add     x2, x2, :lo12:ls_elf_buf
    strb    w0, [x2, x1]
    lsr     x3, x0, #8
    add     x4, x1, #1
    strb    w3, [x2, x4]
    lsr     x3, x0, #16
    add     x4, x1, #2
    strb    w3, [x2, x4]
    lsr     x3, x0, #24
    add     x4, x1, #3
    strb    w3, [x2, x4]
    ret

// put_u64: store u64 at ls_elf_buf[off]
// x0 = val, x1 = off
.align 4
put_u64:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!
    bl      put_u32                 // low 32
    ldp     x0, x1, [sp], #16
    lsr     x0, x0, #32
    add     x1, x1, #4
    bl      put_u32                 // high 32
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// get_pos: return current ls_elf_pos in x0
// ============================================================
.align 4
get_pos:
    adrp    x0, ls_elf_pos
    add     x0, x0, :lo12:ls_elf_pos
    ldr     x0, [x0]
    ret

// ============================================================
// elf_init: reset output buffer position to 0
// ============================================================
.align 4
elf_init:
    adrp    x0, ls_elf_pos
    add     x0, x0, :lo12:ls_elf_pos
    str     xzr, [x0]
    ret


// ############################################################
// PART 1: ARM64 LINUX EXECUTABLE ELF WRITER
// ############################################################
//
// Layout:
//   [0x000] ELF header        (64 bytes)
//   [0x040] Program header     (56 bytes)
//   [0x078] .text section      (emitted ARM64 code)
//   [aligned] .data section    (variables, constants)
//   <no file content for .bss — just memsz in phdr>
//
// Single PT_LOAD segment covers header + text + data.
// Entry point = BASE_ADDR + EHDR_SIZE + PHDR_SIZE (start of .text).
// No section headers needed for execution (minimal binary).

// elf_arm64_header: emit 64-byte ELF64 header for ARM64 exec
// x0 = text_size, x1 = data_size
.align 4
elf_arm64_header:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x19, x0                 // text_size
    mov     x20, x1                 // data_size

    // e_ident[0..3]: magic
    mov     x0, #0x7f
    bl      eb_emit
    mov     x0, #'E'
    bl      eb_emit
    mov     x0, #'L'
    bl      eb_emit
    mov     x0, #'F'
    bl      eb_emit
    // e_ident[4]: ELFCLASS64
    mov     x0, #ELFCLASS64
    bl      eb_emit
    // e_ident[5]: ELFDATA2LSB
    mov     x0, #ELFDATA2LSB
    bl      eb_emit
    // e_ident[6]: EV_CURRENT
    mov     x0, #EV_CURRENT
    bl      eb_emit
    // e_ident[7]: ELFOSABI_NONE (Linux)
    mov     x0, #ELFOSABI_NONE
    bl      eb_emit
    // e_ident[8..15]: padding zeros
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit

    // e_type = ET_EXEC (2)
    mov     x0, #ET_EXEC
    bl      ew_emit
    // e_machine = EM_AARCH64 (183)
    mov     x0, #EM_AARCH64
    bl      ew_emit
    // e_version = EV_CURRENT (1)
    mov     x0, #EV_CURRENT
    bl      ed_emit
    // e_entry = BASE_ADDR + EHDR_SIZE + PHDR_SIZE (start of _start stub)
    mov     x0, #BASE_ADDR
    add     x0, x0, #EHDR_SIZE
    add     x0, x0, #PHDR_SIZE
    bl      eq_emit
    // e_phoff = EHDR_SIZE (64)
    mov     x0, #EHDR_SIZE
    bl      eq_emit
    // e_shoff = 0 (no section headers)
    mov     x0, #0
    bl      eq_emit
    // e_flags = 0
    mov     x0, #0
    bl      ed_emit
    // e_ehsize = 64
    mov     x0, #EHDR_SIZE
    bl      ew_emit
    // e_phentsize = 56
    mov     x0, #PHDR_SIZE
    bl      ew_emit
    // e_phnum = 1
    mov     x0, #1
    bl      ew_emit
    // e_shentsize = 0 (no section headers)
    mov     x0, #0
    bl      ew_emit
    // e_shnum = 0
    mov     x0, #0
    bl      ew_emit
    // e_shstrndx = 0
    mov     x0, #0
    bl      ew_emit

    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// elf_arm64_phdr: emit 56-byte program header
// x0 = text_size, x1 = data_size, x2 = bss_size
.align 4
elf_arm64_phdr:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    stp     x22, xzr, [sp, #-16]!
    mov     x19, x0                 // text_size
    mov     x20, x1                 // data_size
    mov     x21, x2                 // bss_size

    // Compute file size = EHDR + PHDR + text + data
    mov     x22, #EHDR_SIZE
    add     x22, x22, #PHDR_SIZE
    add     x22, x22, x19
    add     x22, x22, x20          // x22 = filesz

    // p_type = PT_LOAD (1)
    mov     x0, #PT_LOAD
    bl      ed_emit
    // p_flags = PF_R | PF_X | PF_W (7 — read/write/exec for simplicity)
    mov     x0, #(PF_R | PF_X | PF_W)
    bl      ed_emit
    // p_offset = 0
    mov     x0, #0
    bl      eq_emit
    // p_vaddr = BASE_ADDR
    mov     x0, #BASE_ADDR
    bl      eq_emit
    // p_paddr = BASE_ADDR
    mov     x0, #BASE_ADDR
    bl      eq_emit
    // p_filesz = EHDR + PHDR + text + data
    mov     x0, x22
    bl      eq_emit
    // p_memsz = filesz + bss_size
    add     x0, x22, x21
    bl      eq_emit
    // p_align = PAGE_ALIGN (0x10000)
    mov     x0, #PAGE_ALIGN
    bl      eq_emit

    ldp     x22, xzr, [sp], #16
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// elf_build_arm64: Build complete ARM64 executable ELF
// ============================================================
// x0 = text_buf   (pointer to ARM64 machine code)
// x1 = text_len   (byte length of .text)
// x2 = data_buf   (pointer to .data content, or 0)
// x3 = data_len   (byte length of .data, or 0)
// x4 = bss_len    (byte length of .bss, or 0)
//
// After return, ls_elf_buf[0..ls_elf_pos) = complete ELF binary.
.align 4
.global elf_build_arm64
elf_build_arm64:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    stp     x22, x23, [sp, #-16]!
    mov     x19, x0                 // text_buf
    mov     x20, x1                 // text_len
    mov     x21, x2                 // data_buf
    mov     x22, x3                 // data_len
    mov     x23, x4                 // bss_len

    // Step 1: Reset buffer
    bl      elf_init

    // Account for _start stub (12 bytes) in text size
    add     x20, x20, #12

    // Step 2: ELF header (64 bytes)
    mov     x0, x20                 // text_size (includes stub)
    mov     x1, x22                 // data_size
    bl      elf_arm64_header

    // Step 3: Program header (56 bytes) — now at offset 64
    mov     x0, x20
    mov     x1, x22
    mov     x2, x23
    bl      elf_arm64_phdr

    // Step 4: .text section — at offset 120 (0x78)
    // Record offset
    bl      get_pos
    adrp    x1, arm_text_off
    add     x1, x1, :lo12:arm_text_off
    str     x0, [x1]

    // Emit _start stub: 3 instructions (12 bytes) before user code.
    // LDR X0, [SP]        = 0xF94003E0  (argc)
    // ADD X1, SP, #8      = 0x910023E1  (argv)
    // B   <main_offset>   = 0x14000000 | ((last_comp_addr/4) + 1)
    //   +1 because B is relative to stub[2], main is at stub_end + last_comp_addr
    mov     w0, #0x03E0
    movk    w0, #0xF940, lsl #16    // LDR X0, [SP]
    bl      ed_emit
    mov     w0, #0x23E1
    movk    w0, #0x9100, lsl #16    // ADD X1, SP, #8
    bl      ed_emit
    // B <main>: offset = (last_comp_addr - stub_B_position) / 4
    // stub_B_position = 8 (3rd instruction, byte offset 8 from text start)
    // main is at last_comp_addr bytes into user code, which starts 12 bytes
    // after text start. So target = 12 + last_comp_addr, source = 8.
    // imm26 = (12 + last_comp_addr - 8) / 4 = (last_comp_addr + 4) / 4
    adrp    x1, ls_last_comp_addr
    add     x1, x1, :lo12:ls_last_comp_addr
    ldr     x1, [x1]
    adrp    x2, ls_code_buf
    add     x2, x2, :lo12:ls_code_buf
    sub     x1, x1, x2             // relative offset of main in user code
    add     x1, x1, #4             // +4 for stub-to-code gap adjustment
    lsr     w1, w1, #2             // convert to instruction count
    and     w1, w1, #0x3FFFFFF     // mask to 26-bit immediate
    mov     w0, #0x14000000         // B opcode
    orr     w0, w0, w1
    bl      ed_emit

    // Copy user text (x20 already includes stub size from step 2)
    mov     x0, x19
    sub     x1, x20, #12           // original text_len without stub
    bl      emit_bytes
    // Record total text size (stub + user code)
    adrp    x1, arm_text_size
    add     x1, x1, :lo12:arm_text_size
    str     x20, [x1]

    // Step 5: .data section (if any)
    cbz     x22, .Larm_no_data
    // Align to 8 bytes for data
    mov     x0, #8
    bl      ealign
    // Record offset
    bl      get_pos
    adrp    x1, arm_data_off
    add     x1, x1, :lo12:arm_data_off
    str     x0, [x1]
    // Copy data
    mov     x0, x21
    mov     x1, x22
    bl      emit_bytes
    // Record size
    adrp    x1, arm_data_size
    add     x1, x1, :lo12:arm_data_size
    str     x22, [x1]
.Larm_no_data:

    // .bss has no file content — accounted for in p_memsz only
    adrp    x1, arm_bss_size
    add     x1, x1, :lo12:arm_bss_size
    str     x23, [x1]

    ldp     x22, x23, [sp], #16
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// elf_save: Write ls_elf_buf[0..ls_elf_pos) to a file
// ============================================================
// x0 = path (null-terminated string pointer)
.align 4
.global elf_save
elf_save:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x19, x0                 // path

    // openat(AT_FDCWD, path, O_WRONLY|O_CREAT|O_TRUNC, 0644)
    mov     x0, #AT_FDCWD
    mov     x1, x19
    mov     x2, #O_WCT
    mov     x3, #MODE_644
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .Lsave_fail
    mov     x19, x0                 // fd

    // Write loop: ls_elf_buf, ls_elf_pos bytes
    adrp    x20, ls_elf_buf
    add     x20, x20, :lo12:ls_elf_buf
    adrp    x1, ls_elf_pos
    add     x1, x1, :lo12:ls_elf_pos
    ldr     x21, [x1]              // remaining = ls_elf_pos
    mov     x1, x20                // write_ptr = ls_elf_buf

.Lsave_loop:
    cbz     x21, .Lsave_close
    mov     x0, x19                // fd
    // x1 = write_ptr (maintained across iterations)
    mov     x2, x21                // count
    mov     x8, #SYS_WRITE
    svc     #0
    cmp     x0, #0
    b.le    .Lsave_close           // error or 0
    add     x1, x1, x0            // advance write_ptr
    sub     x21, x21, x0          // remaining -= written
    b       .Lsave_loop

.Lsave_close:
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

.Lsave_fail:
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// elf_write_arm64: One-call interface — build + save
// ============================================================
// x0 = text_buf, x1 = text_len, x2 = data_buf, x3 = data_len,
// x4 = bss_len, x5 = path (null-terminated)
.align 4
.global elf_write_arm64
elf_write_arm64:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x5                 // save path
    bl      elf_build_arm64
    mov     x0, x19
    bl      elf_save
    ldp     x30, x19, [sp], #16
    ret


// ############################################################
// PART 2: GPU CUBIN ELF WRITER
// ############################################################
//
// 9-section layout:
//   [0] NULL
//   [1] .shstrtab
//   [2] .strtab
//   [3] .symtab              (4 entries)
//   [4] .nv.info             (global kernel attributes)
//   [5] .text.<kernel>       (128-byte aligned SASS)
//   [6] .nv.info.<kernel>    (per-kernel attributes)
//   [7] .nv.shared.reserved.0 (NOBITS)
//   [8] .nv.constant0.<kernel>
//
// No program headers. Section headers at end of file.

// ---- String data for cubin section names ----
.data
.align 3

// Pre-built section name fragments (kernel name appended at runtime)
sn_shstrtab_str:  .asciz ".shstrtab"
sn_strtab_str:    .asciz ".strtab"
sn_symtab_str:    .asciz ".symtab"
sn_nvinfo_str:    .asciz ".nv.info"
sn_nvinfo_k_pfx:  .asciz ".nv.info."
sn_text_k_pfx:    .asciz ".text."
sn_const0_k_pfx:  .asciz ".nv.constant0."
sn_shared_str:    .asciz ".nv.shared.reserved.0"

.text

// emit_asciz: emit null-terminated string from x0; returns bytes emitted in x0
.align 4
emit_asciz:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x0
    mov     x1, #0
1:  ldrb    w0, [x19, x1]
    bl      eb_emit
    ldrb    w0, [x19, x1]
    add     x1, x1, #1
    cbnz    w0, 1b
    mov     x0, x1                  // bytes emitted (including NUL)
    ldp     x30, x19, [sp], #16
    ret

// emit_prefix_kernel: emit prefix string (x0) without NUL, then kernel name + NUL
// x0 = prefix (null-terminated), x1 = kernel_name, x2 = kernel_nlen
.align 4
emit_prefix_kernel:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    mov     x19, x0                 // prefix
    mov     x20, x1                 // kernel_name
    mov     x21, x2                 // kernel_nlen
    // emit prefix bytes (skip trailing NUL)
    mov     x1, #0
1:  ldrb    w0, [x19, x1]
    cbz     w0, 2f
    bl      eb_emit
    ldrb    w0, [x19, x1]          // reload after call
    add     x1, x1, #1
    b       1b
2:  // emit kernel name
    mov     x0, x20
    mov     x1, x21
    bl      emit_bytes
    // emit NUL terminator
    mov     x0, #0
    bl      eb_emit
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// cubin_shdr: emit one 64-byte section header
// Arguments passed on stack (10 u64 values pushed before call)
// For simplicity, we use a register-based interface:
// x0=sh_name, x1=sh_type, x2=sh_flags, x3=sh_addr,
// x4=sh_offset, x5=sh_size, x6=sh_link, x7=sh_info
// [sp+0]=sh_addralign, [sp+8]=sh_entsize
.align 4
cubin_shdr:
    stp     x30, x19, [sp, #-16]!
    // Save all args
    stp     x0, x1, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    // Load addralign, entsize from original stack position
    // They're at [sp + 64 + 16] and [sp + 64 + 24] (4 pushes of 16 + 1 stp x30)
    ldr     x19, [sp, #80]         // sh_addralign
    ldr     x9, [sp, #88]          // sh_entsize

    // sh_name (u32)
    ldr     x0, [sp, #48]
    bl      ed_emit
    // sh_type (u32)
    ldr     x0, [sp, #40]
    bl      ed_emit
    // sh_flags (u64)
    ldr     x0, [sp, #32]
    bl      eq_emit
    // sh_addr (u64)
    ldr     x0, [sp, #24]
    bl      eq_emit
    // sh_offset (u64)
    ldr     x0, [sp, #16]
    bl      eq_emit
    // sh_size (u64)
    ldr     x0, [sp, #8]
    bl      eq_emit
    // sh_link (u32)
    ldr     x0, [sp, #0]
    bl      ed_emit
    // sh_info (u32)
    ldp     x6, x7, [sp], #16
    bl      ed_emit
    mov     x0, x7                  // sh_info already consumed, use saved
    // Actually let me redo this more cleanly

    // Pop saved regs and redo with simpler approach
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x0, x1, [sp], #16
    ldp     x30, x19, [sp], #16
    // Fall through to cubin_shdr_v2
    b       cubin_shdr_v2

// Cleaner section header emitter using a parameter block in memory
// x0 = pointer to 10-element u64 array:
//   [0]=sh_name [1]=sh_type [2]=sh_flags [3]=sh_addr [4]=sh_offset
//   [5]=sh_size [6]=sh_link [7]=sh_info [8]=sh_addralign [9]=sh_entsize
.align 4
cubin_shdr_v2:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x0
    // sh_name (u32)
    ldr     x0, [x19, #0]
    bl      ed_emit
    // sh_type (u32)
    ldr     x0, [x19, #8]
    bl      ed_emit
    // sh_flags (u64)
    ldr     x0, [x19, #16]
    bl      eq_emit
    // sh_addr (u64)
    ldr     x0, [x19, #24]
    bl      eq_emit
    // sh_offset (u64)
    ldr     x0, [x19, #32]
    bl      eq_emit
    // sh_size (u64)
    ldr     x0, [x19, #40]
    bl      eq_emit
    // sh_link (u32)
    ldr     x0, [x19, #48]
    bl      ed_emit
    // sh_info (u32)
    ldr     x0, [x19, #56]
    bl      ed_emit
    // sh_addralign (u64)
    ldr     x0, [x19, #64]
    bl      eq_emit
    // sh_entsize (u64)
    ldr     x0, [x19, #72]
    bl      eq_emit
    ldp     x30, x19, [sp], #16
    ret

// cubin_sym: emit one 24-byte ELF64 symbol
// x0=st_name, x1=st_info, x2=st_other, x3=st_shndx, x4=st_value, x5=st_size
.align 4
cubin_sym:
    stp     x30, x19, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x1, xzr, [sp, #-16]!

    // st_name (u32)
    bl      ed_emit
    // st_info (u8)
    ldr     x0, [sp]
    bl      eb_emit
    // st_other (u8)
    ldr     x0, [sp, #16]
    bl      eb_emit
    // st_shndx (u16)
    ldr     x0, [sp, #24]
    bl      ew_emit
    // st_value (u64)
    ldr     x0, [sp, #32]
    bl      eq_emit
    // st_size (u64)
    ldr     x0, [sp, #40]
    bl      eq_emit

    ldp     x1, xzr, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// nvi_u32: emit plain u32 nv.info record
// x0=val, x1=attr, x2=fmt
.align 4
nvi_u32:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!
    // fmt
    mov     x0, x2
    bl      eb_emit
    // attr
    ldr     x0, [sp, #8]
    bl      eb_emit
    // size = 4
    mov     x0, #4
    bl      ew_emit
    // val
    ldr     x0, [sp]
    bl      ed_emit
    ldp     x0, x1, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// nvi_sval: emit sym+val nv.info record
// x0=val, x1=sym_idx, x2=attr, x3=fmt
.align 4
nvi_sval:
    stp     x30, x19, [sp, #-16]!
    stp     x0, x1, [sp, #-16]!
    // fmt
    mov     x0, x3
    bl      eb_emit
    // attr
    mov     x0, x2
    bl      eb_emit
    // size = 8
    mov     x0, #8
    bl      ew_emit
    // sym_idx
    ldr     x0, [sp, #8]
    bl      ed_emit
    // val
    ldr     x0, [sp]
    bl      ed_emit
    ldp     x0, x1, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// cubin parameter block (passed to elf_build_cubin)
// ============================================================
// For clean register management, the cubin builder takes a
// pointer to a parameter struct:
.bss
.align 3
cubin_params:
    .space 80      // 10 x 8 bytes
// Offsets:
//   [0]  kernel_name    (ptr)
//   [8]  kernel_nlen    (u64)
//   [16] code_buf       (ptr)
//   [24] code_size      (u64)
//   [32] n_kparams      (u64)
//   [40] reg_count      (u64)
//   [48] smem_size      (u64)
//   [56] cooperative    (u64, 0 or 1)
//   [64] gridsync_offsets (ptr)
//   [72] gridsync_count (u64)

// Scratch space for section header parameter blocks
shdr_scratch:   .space 80     // 10 x u64

.text

// ============================================================
// elf_build_cubin: Build complete 9-section GPU ELF
// ============================================================
// x0 = pointer to cubin_params struct (see layout above)
//
// After return, ls_elf_buf[0..ls_elf_pos) = complete cubin ELF.
.align 4
.global elf_build_cubin
elf_build_cubin:
    stp     x30, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    stp     x22, x23, [sp, #-16]!
    stp     x24, x25, [sp, #-16]!
    stp     x26, x27, [sp, #-16]!
    stp     x28, x29, [sp, #-16]!
    // Use frame pointer
    mov     x29, sp
    // Allocate local storage (256 bytes for section metadata)
    sub     sp, sp, #256

    mov     x19, x0                 // params ptr
    // Load frequently-used params
    ldr     x20, [x19, #0]         // kernel_name
    ldr     x21, [x19, #8]         // kernel_nlen
    ldr     x22, [x19, #16]        // code_buf
    ldr     x23, [x19, #24]        // code_size
    ldr     x24, [x19, #32]        // n_kparams
    ldr     x25, [x19, #40]        // reg_count
    ldr     x26, [x19, #48]        // smem_size
    ldr     x27, [x19, #56]        // cooperative
    ldr     x28, [x19, #64]        // gridsync_offsets

    // ---- Step 1: Reset and emit ELF header placeholder ----
    bl      elf_init
    bl      cubin_elf_header

    // ---- Step 2: .shstrtab ----
    bl      get_pos
    str     x0, [sp, #0]           // [sp+0] = shstrtab_off

    // NUL byte at offset 0
    mov     x0, #0
    bl      eb_emit

    // .shstrtab
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #8]           // [sp+8] = SN_shstrtab
    adrp    x0, sn_shstrtab_str
    add     x0, x0, :lo12:sn_shstrtab_str
    bl      emit_asciz

    // .strtab
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #16]          // [sp+16] = SN_strtab
    adrp    x0, sn_strtab_str
    add     x0, x0, :lo12:sn_strtab_str
    bl      emit_asciz

    // .symtab
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #24]          // [sp+24] = SN_symtab
    adrp    x0, sn_symtab_str
    add     x0, x0, :lo12:sn_symtab_str
    bl      emit_asciz

    // .nv.info
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #32]          // [sp+32] = SN_nvinfo
    adrp    x0, sn_nvinfo_str
    add     x0, x0, :lo12:sn_nvinfo_str
    bl      emit_asciz

    // .nv.info.<kernel>
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #40]          // [sp+40] = SN_nvinfo_k
    adrp    x0, sn_nvinfo_k_pfx
    add     x0, x0, :lo12:sn_nvinfo_k_pfx
    mov     x1, x20
    mov     x2, x21
    bl      emit_prefix_kernel

    // .text.<kernel>
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #48]          // [sp+48] = SN_text
    adrp    x0, sn_text_k_pfx
    add     x0, x0, :lo12:sn_text_k_pfx
    mov     x1, x20
    mov     x2, x21
    bl      emit_prefix_kernel

    // .nv.constant0.<kernel>
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #56]          // [sp+56] = SN_const0
    adrp    x0, sn_const0_k_pfx
    add     x0, x0, :lo12:sn_const0_k_pfx
    mov     x1, x20
    mov     x2, x21
    bl      emit_prefix_kernel

    // .nv.shared.reserved.0
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #64]          // [sp+64] = SN_shared
    adrp    x0, sn_shared_str
    add     x0, x0, :lo12:sn_shared_str
    bl      emit_asciz

    // Align to 4
    mov     x0, #4
    bl      ealign

    // shstrtab_size
    bl      get_pos
    ldr     x1, [sp, #0]
    sub     x0, x0, x1
    str     x0, [sp, #72]          // [sp+72] = shstrtab_size

    // ---- Step 3: .strtab ----
    bl      get_pos
    str     x0, [sp, #80]          // [sp+80] = strtab_off

    // sym0: empty name (NUL)
    mov     x0, #0
    bl      eb_emit
    // sym1: empty name (NUL)
    mov     x0, #0
    bl      eb_emit

    // strsym_kernel = pos - strtab_off
    bl      get_pos
    ldr     x1, [sp, #80]
    sub     x0, x0, x1
    str     x0, [sp, #88]          // [sp+88] = strsym_kernel

    // kernel name + NUL
    mov     x0, x20
    mov     x1, x21
    bl      emit_bytes
    mov     x0, #0
    bl      eb_emit

    // strtab_size
    bl      get_pos
    ldr     x1, [sp, #80]
    sub     x0, x0, x1
    str     x0, [sp, #96]          // [sp+96] = strtab_size

    // ---- Step 4: .symtab (4 entries x 24 = 96 bytes) ----
    mov     x0, #8
    bl      ealign

    bl      get_pos
    str     x0, [sp, #104]         // [sp+104] = symtab_off

    // sym0: UNDEF
    mov     x0, #0
    mov     x1, #0
    mov     x2, #0
    mov     x3, #0
    mov     x4, #0
    mov     x5, #0
    bl      cubin_sym

    // sym1: SECTION LOCAL, shndx=5 (.text.<kernel>)
    mov     x0, #0
    mov     x1, #STT_SECTION       // st_info = 3 (STB_LOCAL|STT_SECTION)
    mov     x2, #0
    mov     x3, #5
    mov     x4, #0
    mov     x5, #0
    bl      cubin_sym

    // sym2: SECTION LOCAL, shndx=8 (.nv.constant0.<kernel>)
    mov     x0, #0
    mov     x1, #STT_SECTION
    mov     x2, #0
    mov     x3, #8
    mov     x4, #0
    mov     x5, #0
    bl      cubin_sym

    // sym3: FUNC GLOBAL STO_CUDA_ENTRY shndx=5
    // st_info = 0x12 (STB_GLOBAL<<4 | STT_FUNC)
    // st_other = 0x10 (STO_CUDA_ENTRY)
    // st_size patched later
    bl      get_pos
    str     x0, [sp, #112]         // [sp+112] = sym_kernel_off

    ldr     x0, [sp, #88]          // strsym_kernel
    mov     x1, #0x12              // STB_GLOBAL|STT_FUNC
    mov     x2, #STO_CUDA_ENTRY
    mov     x3, #5                 // .text.<kernel> section
    mov     x4, #0                 // st_value
    mov     x5, #0                 // st_size (patched later)
    bl      cubin_sym

    // symtab_size
    bl      get_pos
    ldr     x1, [sp, #104]
    sub     x0, x0, x1
    str     x0, [sp, #120]         // [sp+120] = symtab_size

    // ---- Step 5: .nv.info (global attributes) ----
    mov     x0, #4
    bl      ealign

    bl      get_pos
    str     x0, [sp, #128]         // [sp+128] = nvinfo_off

    // REGCOUNT (sym_idx=3)
    mov     x0, x25                // reg_count
    cmp     x0, #8
    b.ge    1f
    mov     x0, #8                 // minimum 8
1:  mov     x1, #3                 // sym_idx
    mov     x2, #EIATTR_REGCOUNT
    mov     x3, #NVI_FMT_U32
    bl      nvi_sval

    // FRAME_SIZE = 0
    mov     x0, #0
    mov     x1, #3
    mov     x2, #EIATTR_FRAME_SIZE
    mov     x3, #NVI_FMT_U32
    bl      nvi_sval

    // MIN_STACK_SIZE = 0
    mov     x0, #0
    mov     x1, #3
    mov     x2, #EIATTR_MIN_STACK_SIZE
    mov     x3, #NVI_FMT_U32
    bl      nvi_sval

    // nvinfo_size
    bl      get_pos
    ldr     x1, [sp, #128]
    sub     x0, x0, x1
    str     x0, [sp, #136]         // [sp+136] = nvinfo_size

    // ---- Step 6: .nv.info.<kernel> (per-kernel attributes) ----
    mov     x0, #4
    bl      ealign

    bl      get_pos
    str     x0, [sp, #144]         // [sp+144] = nvinfo_k_off

    // CUDA_API_VERSION = 0x80
    mov     x0, #128
    mov     x1, #EIATTR_CUDA_API_VERSION
    mov     x2, #NVI_FMT_U32
    bl      nvi_u32

    // KPARAM_INFO: one 12-byte record per param, reverse ordinal
    mov     x9, #0                 // pi = 0
.Lkparam_loop:
    cmp     x9, x24                // pi < n_kparams?
    b.ge    .Lkparam_done
    // ordinal = n_kparams - 1 - pi
    sub     x10, x24, #1
    sub     x10, x10, x9

    // fmt = NVI_FMT_U32
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    // attr = EIATTR_KPARAM_INFO
    mov     x0, #EIATTR_KPARAM_INFO
    bl      eb_emit
    // size = 12
    mov     x0, #12
    bl      ew_emit
    // index = 0
    mov     x0, #0
    bl      ed_emit
    // (offset<<16)|ordinal: offset = ordinal*8
    sub     x10, x24, #1
    sub     x10, x10, x9           // reload ordinal
    lsl     x11, x10, #3           // offset = ordinal * 8
    lsl     x11, x11, #16          // offset << 16
    orr     x0, x11, x10           // | ordinal
    bl      ed_emit
    // flags: 0x0021f000 (size=8 ptr)
    mov     x0, #0x0021
    lsl     x0, x0, #16
    orr     x0, x0, #0xf000
    bl      ed_emit

    add     x9, x9, #1
    b       .Lkparam_loop
.Lkparam_done:

    // SPARSE_MMA_MASK (FLAG: fmt=03 attr=0x50 val=0)
    mov     x0, #NVI_FMT_FLAG
    bl      eb_emit
    mov     x0, #EIATTR_SPARSE_MMA_MASK
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit

    // MAXREG_COUNT (HVAL: fmt=03 attr=0x1b val=0x00ff)
    mov     x0, #NVI_FMT_FLAG
    bl      eb_emit
    mov     x0, #EIATTR_MAXREG_COUNT
    bl      eb_emit
    mov     x0, #255
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit

    // Cooperative attributes (if cooperative == 1)
    cmp     x27, #1
    b.ne    .Lno_coop

    // COOP_GROUP_MASK_REGIDS: fmt=04 attr=0x29 size=16; 4x 0xffffffff
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_COOP_GROUP_MASK_REG
    bl      eb_emit
    mov     x0, #16
    bl      ew_emit
    mov     w0, #-1                // 0xffffffff
    bl      ed_emit
    mov     w0, #-1
    bl      ed_emit
    mov     w0, #-1
    bl      ed_emit
    mov     w0, #-1
    bl      ed_emit

    // COOP_GROUP_INSTR_OFFSETS: fmt=04 attr=0x28 size=N*4
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_COOP_GROUP_INSTR_OFF
    bl      eb_emit
    ldr     x9, [x19, #72]         // gridsync_count
    lsl     x0, x9, #2             // size = count * 4
    bl      ew_emit
    // Emit each offset
    mov     x10, #0
.Lgs_loop:
    cmp     x10, x9
    b.ge    .Lgs_done
    ldr     w0, [x28, x10, lsl #2]
    bl      ed_emit
    add     x10, x10, #1
    b       .Lgs_loop
.Lgs_done:

.Lno_coop:
    // EXIT_INSTR_OFFSETS: fmt=04 attr=0x1c size=4 val=0x100
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_EXIT_INSTR_OFFSETS
    bl      eb_emit
    mov     x0, #4
    bl      ew_emit
    mov     x0, #256               // 0x100
    bl      ed_emit

    // CBANK_PARAM_SIZE (HVAL: param_bytes = n_kparams * 8)
    lsl     x9, x24, #3            // param_bytes
    mov     x0, #NVI_FMT_FLAG
    bl      eb_emit
    mov     x0, #EIATTR_CBANK_PARAM_SIZE
    bl      eb_emit
    and     x0, x9, #0xff
    bl      eb_emit
    lsr     x0, x9, #8
    and     x0, x0, #0xff
    bl      eb_emit

    // PARAM_CBANK (SVAL: sym_idx=2, cbank_offset=0x210, param_bytes in high 16)
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_PARAM_CBANK
    bl      eb_emit
    mov     x0, #8
    bl      ew_emit
    mov     x0, #2                 // sym_idx for constant0 section
    bl      ed_emit
    // (param_bytes << 16) | 0x210
    lsl     x0, x9, #16
    mov     x11, #0x210
    orr     x0, x0, x11
    bl      ed_emit

    // SW_WAR workaround flag
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_SW_WAR
    bl      eb_emit
    mov     x0, #4
    bl      ew_emit
    mov     x0, #8
    bl      ed_emit

    // CRS_STACK_SIZE (if smem_size > 0)
    cbz     x26, .Lno_crs
    mov     x0, #NVI_FMT_U32
    bl      eb_emit
    mov     x0, #EIATTR_CRS_STACK_SIZE
    bl      eb_emit
    mov     x0, #4
    bl      ew_emit
    mov     x0, x26                // smem_size
    bl      ed_emit
.Lno_crs:

    // nvinfo_k_size
    bl      get_pos
    ldr     x1, [sp, #144]
    sub     x0, x0, x1
    str     x0, [sp, #152]         // [sp+152] = nvinfo_k_size

    // ---- Step 7: .text.<kernel> (128-byte aligned SASS) ----
    mov     x0, #128
    bl      ealign

    bl      get_pos
    str     x0, [sp, #160]         // [sp+160] = text_off

    // Copy SASS code (or 48 zero bytes as fallback)
    cbz     x23, .Lcu_text_fallback
    mov     x0, x22                // code_buf
    mov     x1, x23                // code_size
    bl      emit_bytes
    b       .Lcu_text_done
.Lcu_text_fallback:
    mov     x9, #0
1:  cmp     x9, #48
    b.ge    .Lcu_text_done
    mov     x0, #0
    bl      eb_emit
    add     x9, x9, #1
    b       1b
.Lcu_text_done:

    // text_size
    bl      get_pos
    ldr     x1, [sp, #160]
    sub     x0, x0, x1
    str     x0, [sp, #168]         // [sp+168] = text_size

    // Patch kernel symbol st_size (u64 at sym_kernel_off + 16)
    ldr     x0, [sp, #168]         // text_size
    ldr     x1, [sp, #112]         // sym_kernel_off
    add     x1, x1, #16
    bl      put_u64

    // ---- Step 8: .nv.constant0.<kernel> ----
    // 0x210 = 528 bytes reserved + param_bytes
    mov     x0, #4
    bl      ealign

    bl      get_pos
    str     x0, [sp, #176]         // [sp+176] = const0_off

    lsl     x9, x24, #3            // param_bytes
    add     x10, x9, #528          // const0_total = 528 + param_bytes
    mov     x11, #0
1:  cmp     x11, x10
    b.ge    2f
    mov     x0, #0
    bl      eb_emit
    add     x11, x11, #1
    b       1b
2:
    // const0_size
    bl      get_pos
    ldr     x1, [sp, #176]
    sub     x0, x0, x1
    str     x0, [sp, #184]         // [sp+184] = const0_size

    // ---- Step 9: Section headers (9 x 64 = 576 bytes) ----
    mov     x0, #64
    bl      ealign

    bl      get_pos
    str     x0, [sp, #192]         // [sp+192] = shdrs_off

    // Helper: use shdr_scratch as parameter block for cubin_shdr_v2
    // [0] NULL section
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    stp     xzr, xzr, [x0, #0]
    stp     xzr, xzr, [x0, #16]
    stp     xzr, xzr, [x0, #32]
    stp     xzr, xzr, [x0, #48]
    stp     xzr, xzr, [x0, #64]
    bl      cubin_shdr_v2

    // [1] .shstrtab (SHT_STRTAB)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #8]           // SN_shstrtab
    str     x1, [x0, #0]           // sh_name
    mov     x1, #SHT_STRTAB
    str     x1, [x0, #8]           // sh_type
    str     xzr, [x0, #16]         // sh_flags
    str     xzr, [x0, #24]         // sh_addr
    ldr     x1, [sp, #0]           // shstrtab_off
    str     x1, [x0, #32]          // sh_offset
    ldr     x1, [sp, #72]          // shstrtab_size
    str     x1, [x0, #40]          // sh_size
    str     xzr, [x0, #48]         // sh_link
    str     xzr, [x0, #56]         // sh_info
    mov     x1, #1
    str     x1, [x0, #64]          // sh_addralign
    str     xzr, [x0, #72]         // sh_entsize
    bl      cubin_shdr_v2

    // [2] .strtab (SHT_STRTAB)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #16]          // SN_strtab
    str     x1, [x0, #0]
    mov     x1, #SHT_STRTAB
    str     x1, [x0, #8]
    str     xzr, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #80]          // strtab_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #96]          // strtab_size
    str     x1, [x0, #40]
    str     xzr, [x0, #48]
    str     xzr, [x0, #56]
    mov     x1, #1
    str     x1, [x0, #64]
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // [3] .symtab (SHT_SYMTAB, link=2, info=3)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #24]          // SN_symtab
    str     x1, [x0, #0]
    mov     x1, #SHT_SYMTAB
    str     x1, [x0, #8]
    str     xzr, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #104]         // symtab_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #120]         // symtab_size
    str     x1, [x0, #40]
    mov     x1, #2                 // sh_link = .strtab
    str     x1, [x0, #48]
    mov     x1, #3                 // sh_info = first_global
    str     x1, [x0, #56]
    mov     x1, #8
    str     x1, [x0, #64]          // sh_addralign
    mov     x1, #SYM_SIZE
    str     x1, [x0, #72]          // sh_entsize
    bl      cubin_shdr_v2

    // [4] .nv.info (SHT_LOPROC, link=3=.symtab)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #32]          // SN_nvinfo
    str     x1, [x0, #0]
    mov     x1, #SHT_LOPROC
    str     x1, [x0, #8]
    str     xzr, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #128]         // nvinfo_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #136]         // nvinfo_size
    str     x1, [x0, #40]
    mov     x1, #3                 // sh_link = .symtab
    str     x1, [x0, #48]
    str     xzr, [x0, #56]
    mov     x1, #4
    str     x1, [x0, #64]
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // [5] .text.<kernel> (SHT_PROGBITS, flags=ALLOC|EXECINSTR=6, link=3, info=3)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #48]          // SN_text
    str     x1, [x0, #0]
    mov     x1, #SHT_PROGBITS
    str     x1, [x0, #8]
    mov     x1, #(SHF_ALLOC | SHF_EXECINSTR)
    str     x1, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #160]         // text_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #168]         // text_size
    str     x1, [x0, #40]
    mov     x1, #3                 // sh_link = .symtab
    str     x1, [x0, #48]
    mov     x1, #3                 // sh_info = kernel sym
    str     x1, [x0, #56]
    mov     x1, #128
    str     x1, [x0, #64]          // sh_addralign
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // [6] .nv.info.<kernel> (SHT_LOPROC, flags=INFO_LINK=0x40, link=3, info=5)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #40]          // SN_nvinfo_k
    str     x1, [x0, #0]
    mov     x1, #SHT_LOPROC
    str     x1, [x0, #8]
    mov     x1, #SHF_INFO_LINK
    str     x1, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #144]         // nvinfo_k_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #152]         // nvinfo_k_size
    str     x1, [x0, #40]
    mov     x1, #3
    str     x1, [x0, #48]
    mov     x1, #5                 // info = .text section
    str     x1, [x0, #56]
    mov     x1, #4
    str     x1, [x0, #64]
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // [7] .nv.shared.reserved.0 (SHT_NOBITS, flags=ALLOC|WRITE=3)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #64]          // SN_shared
    str     x1, [x0, #0]
    mov     x1, #SHT_NOBITS
    str     x1, [x0, #8]
    mov     x1, #(SHF_WRITE | SHF_ALLOC)
    str     x1, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #176]         // const0_off (file position placeholder)
    str     x1, [x0, #32]
    str     x26, [x0, #40]         // sh_size = smem_size
    str     xzr, [x0, #48]
    str     xzr, [x0, #56]
    mov     x1, #16
    str     x1, [x0, #64]
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // [8] .nv.constant0.<kernel> (SHT_PROGBITS, flags=ALLOC|INFO_LINK=0x42, info=5)
    adrp    x0, shdr_scratch
    add     x0, x0, :lo12:shdr_scratch
    ldr     x1, [sp, #56]          // SN_const0
    str     x1, [x0, #0]
    mov     x1, #SHT_PROGBITS
    str     x1, [x0, #8]
    mov     x1, #(SHF_ALLOC | SHF_INFO_LINK)
    str     x1, [x0, #16]
    str     xzr, [x0, #24]
    ldr     x1, [sp, #176]         // const0_off
    str     x1, [x0, #32]
    ldr     x1, [sp, #184]         // const0_size
    str     x1, [x0, #40]
    str     xzr, [x0, #48]
    mov     x1, #5                 // info = .text section
    str     x1, [x0, #56]
    mov     x1, #4
    str     x1, [x0, #64]
    str     xzr, [x0, #72]
    bl      cubin_shdr_v2

    // ---- Step 10: Patch ELF header ----
    // e_phoff = 0 (no program headers)
    mov     x0, #0
    mov     x1, #32
    bl      put_u64
    // e_shoff = shdrs_off
    ldr     x0, [sp, #192]
    mov     x1, #40
    bl      put_u64
    // e_phentsize = 0
    mov     x0, #0
    mov     x1, #54
    bl      put_u16
    // e_phnum = 0
    mov     x0, #0
    mov     x1, #56
    bl      put_u16
    // e_shnum = 9
    mov     x0, #9
    mov     x1, #60
    bl      put_u16
    // e_shstrndx = 1
    mov     x0, #1
    mov     x1, #62
    bl      put_u16

    // Restore stack
    add     sp, sp, #256
    ldp     x28, x29, [sp], #16
    ldp     x26, x27, [sp], #16
    ldp     x24, x25, [sp], #16
    ldp     x22, x23, [sp], #16
    ldp     x20, x21, [sp], #16
    ldp     x30, x19, [sp], #16
    ret

// cubin_elf_header: emit 64-byte ELF64 header for CUDA cubin
// (fields patched later by elf_build_cubin)
.align 4
cubin_elf_header:
    str     x30, [sp, #-16]!

    // e_ident magic
    mov     x0, #0x7f
    bl      eb_emit
    mov     x0, #'E'
    bl      eb_emit
    mov     x0, #'L'
    bl      eb_emit
    mov     x0, #'F'
    bl      eb_emit
    // ELFCLASS64
    mov     x0, #ELFCLASS64
    bl      eb_emit
    // ELFDATA2LSB
    mov     x0, #ELFDATA2LSB
    bl      eb_emit
    // EV_CURRENT
    mov     x0, #EV_CURRENT
    bl      eb_emit
    // ELFOSABI_CUDA = 0x33
    mov     x0, #ELFOSABI_CUDA
    bl      eb_emit
    // ABI version = 7
    mov     x0, #7
    bl      eb_emit
    // padding (7 bytes)
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit
    mov     x0, #0
    bl      eb_emit

    // e_type = ET_EXEC
    mov     x0, #ET_EXEC
    bl      ew_emit
    // e_machine = EM_CUDA
    mov     x0, #EM_CUDA
    bl      ew_emit
    // e_version = EV_CUDA (0x80)
    mov     x0, #EV_CUDA
    bl      ed_emit
    // e_entry = 0
    mov     x0, #0
    bl      eq_emit
    // e_phoff = 0 (patched)
    mov     x0, #0
    bl      eq_emit
    // e_shoff = 0 (patched)
    mov     x0, #0
    bl      eq_emit
    // e_flags = ELF_FLAGS_CUDA (0x5a055a — needs movz+movk)
    movz    x0, #0x055a
    movk    x0, #0x005a, lsl #16
    bl      ed_emit
    // e_ehsize = 64
    mov     x0, #EHDR_SIZE
    bl      ew_emit
    // e_phentsize = 56 (patched to 0 if no phdrs)
    mov     x0, #PHDR_SIZE
    bl      ew_emit
    // e_phnum = 0 (patched)
    mov     x0, #0
    bl      ew_emit
    // e_shentsize = 64
    mov     x0, #SHDR_SIZE
    bl      ew_emit
    // e_shnum = 0 (patched)
    mov     x0, #0
    bl      ew_emit
    // e_shstrndx = 0 (patched)
    mov     x0, #0
    bl      ew_emit

    ldr     x30, [sp], #16
    ret

// ============================================================
// elf_write_cubin: One-call interface — build cubin + save
// ============================================================
// x0 = cubin_params pointer, x1 = output path (null-terminated)
.align 4
.global elf_write_cubin
elf_write_cubin:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x1                // save path
    bl      elf_build_cubin
    mov     x0, x19
    bl      elf_save
    ldp     x30, x19, [sp], #16
    ret

// ============================================================
// elf_write_cubin_simple: Non-cooperative kernel convenience
// ============================================================
// x0=kernel_name, x1=kernel_nlen, x2=code_buf, x3=code_size,
// x4=n_kparams, x5=reg_count, x6=smem_size, x7=out_path
.align 4
.global elf_write_cubin_simple
elf_write_cubin_simple:
    stp     x30, x19, [sp, #-16]!
    mov     x19, x7                // save path

    // Fill cubin_params
    adrp    x8, cubin_params
    add     x8, x8, :lo12:cubin_params
    str     x0, [x8, #0]          // kernel_name
    str     x1, [x8, #8]          // kernel_nlen
    str     x2, [x8, #16]         // code_buf
    str     x3, [x8, #24]         // code_size
    str     x4, [x8, #32]         // n_kparams
    str     x5, [x8, #40]         // reg_count
    str     x6, [x8, #48]         // smem_size
    str     xzr, [x8, #56]        // cooperative = 0
    str     xzr, [x8, #64]        // gridsync_offsets = NULL
    str     xzr, [x8, #72]        // gridsync_count = 0

    mov     x0, x8
    mov     x1, x19
    bl      elf_write_cubin

    ldp     x30, x19, [sp], #16
    ret


// ############################################################
// PART 3: DTC WORD WRAPPERS
// ############################################################
// These wrap the above functions as DTC words callable from
// the Lithos threaded interpreter. They bridge the register
// conventions: DTC stack (X24=DSP, X22=TOS) <-> ABI calls.

// elf-build-arm64 ( text-buf text-len data-buf data-len bss-len -- )
.align 4
code_ELF_BUILD_ARM64:
    // TOS = bss_len
    mov     x4, x22                // bss_len
    ldr     x3, [x24], #8          // data_len
    ldr     x2, [x24], #8          // data_buf
    ldr     x1, [x24], #8          // text_len
    ldr     x0, [x24], #8          // text_buf
    ldr     x22, [x24], #8         // restore TOS

    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x21, [sp, #-16]!
    stp     x20, xzr, [sp, #-16]!

    bl      elf_build_arm64

    ldp     x20, xzr, [sp], #16
    ldp     x22, x21, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// elf-save ( path-addr -- )
.align 4
code_ELF_SAVE:
    mov     x0, x22
    ldr     x22, [x24], #8

    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x21, [sp, #-16]!
    stp     x20, xzr, [sp, #-16]!

    bl      elf_save

    ldp     x20, xzr, [sp], #16
    ldp     x22, x21, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// elf-write-arm64 ( text-buf text-len data-buf data-len bss-len path-addr -- )
.align 4
code_ELF_WRITE_ARM64:
    mov     x5, x22                // path
    ldr     x4, [x24], #8          // bss_len
    ldr     x3, [x24], #8          // data_len
    ldr     x2, [x24], #8          // data_buf
    ldr     x1, [x24], #8          // text_len
    ldr     x0, [x24], #8          // text_buf
    ldr     x22, [x24], #8

    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x21, [sp, #-16]!
    stp     x20, xzr, [sp, #-16]!

    bl      elf_write_arm64

    ldp     x20, xzr, [sp], #16
    ldp     x22, x21, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// elf-build-cubin ( params-ptr -- )
.align 4
code_ELF_BUILD_CUBIN:
    mov     x0, x22
    ldr     x22, [x24], #8

    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x21, [sp, #-16]!
    stp     x20, xzr, [sp, #-16]!

    bl      elf_build_cubin

    ldp     x20, xzr, [sp], #16
    ldp     x22, x21, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// elf-write-cubin ( params-ptr path-addr -- )
.align 4
code_ELF_WRITE_CUBIN:
    mov     x1, x22                // path
    ldr     x0, [x24], #8          // params
    ldr     x22, [x24], #8

    stp     x26, x25, [sp, #-16]!
    stp     x24, x23, [sp, #-16]!
    stp     x22, x21, [sp, #-16]!
    stp     x20, xzr, [sp, #-16]!

    bl      elf_write_cubin

    ldp     x20, xzr, [sp], #16
    ldp     x22, x21, [sp], #16
    ldp     x24, x23, [sp], #16
    ldp     x26, x25, [sp], #16
    NEXT

// elf-buf ( -- addr )   Push address of output buffer
.align 4
code_ELF_BUF:
    str     x22, [x24, #-8]!
    adrp    x22, ls_elf_buf
    add     x22, x22, :lo12:ls_elf_buf
    NEXT

// elf-pos ( -- addr )   Push address of position variable
.align 4
code_ELF_POS:
    str     x22, [x24, #-8]!
    adrp    x22, ls_elf_pos
    add     x22, x22, :lo12:ls_elf_pos
    NEXT

// cubin-params ( -- addr )   Push address of cubin parameter block
.align 4
code_CUBIN_PARAMS:
    str     x22, [x24, #-8]!
    adrp    x22, cubin_params
    add     x22, x22, :lo12:cubin_params
    NEXT


// ############################################################
// PART 4: MACROS FOR DTC THREADING
// ############################################################
// These are duplicated from lithos-bootstrap.s so this file
// can assemble standalone. When included into lithos-bootstrap.s,
// remove this section and use the existing NEXT/PUSH/POP macros.

.ifndef NEXT_DEFINED
.macro NEXT
    ldr     x25, [x26], #8
    ldr     x16, [x25]
    br      x16
.endm
.set NEXT_DEFINED, 1
.endif

// ============================================================
// Dictionary entries — extends the chain past entry_p_parse_tokens
// (the tail of lithos-parser.s). Tail of this file: entry_ew_cubin_params
// ============================================================
.data
.align 3

entry_ew_elf_build_arm64:
    .quad   entry_p_parse_tokens
    .byte   0
    .byte   15
    .ascii  "elf-build-arm64"
    .align  3
    .quad   code_ELF_BUILD_ARM64

entry_ew_elf_save:
    .quad   entry_ew_elf_build_arm64
    .byte   0
    .byte   8
    .ascii  "elf-save"
    .align  3
    .quad   code_ELF_SAVE

entry_ew_elf_write_arm64:
    .quad   entry_ew_elf_save
    .byte   0
    .byte   15
    .ascii  "elf-write-arm64"
    .align  3
    .quad   code_ELF_WRITE_ARM64

entry_ew_elf_build_cubin:
    .quad   entry_ew_elf_write_arm64
    .byte   0
    .byte   15
    .ascii  "elf-build-cubin"
    .align  3
    .quad   code_ELF_BUILD_CUBIN

entry_ew_elf_write_cubin:
    .quad   entry_ew_elf_build_cubin
    .byte   0
    .byte   15
    .ascii  "elf-write-cubin"
    .align  3
    .quad   code_ELF_WRITE_CUBIN

entry_ew_elf_buf:
    .quad   entry_ew_elf_write_cubin
    .byte   0
    .byte   7
    .ascii  "elf-buf"
    .align  3
    .quad   code_ELF_BUF

entry_ew_elf_pos:
    .quad   entry_ew_elf_buf
    .byte   0
    .byte   7
    .ascii  "elf-pos"
    .align  3
    .quad   code_ELF_POS

entry_ew_cubin_params:
    .quad   entry_ew_elf_pos
    .byte   0
    .byte   12
    .ascii  "cubin-params"
    .align  3
    .quad   code_CUBIN_PARAMS
