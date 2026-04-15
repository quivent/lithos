\\ ============================================================================
\\ parts/05-arm64-elf.ls — ARM64 ELF executable writer
\\ ============================================================================
\\
\\ Wraps a block of ARM64 machine code (already emitted into some caller
\\ buffer) into a statically linked ELF64 AArch64 Linux executable and
\\ writes it to disk.
\\
\\ Entry point:
\\
\\     arm64_elf_write code_buf code_size out_path
\\
\\ where:
\\   code_buf  — pointer to ARM64 machine code bytes (little-endian u32s)
\\   code_size — length of code in bytes (must be multiple of 4)
\\   out_path  — NUL-terminated path for the output executable
\\
\\ Layout of the produced file:
\\
\\     0x0000  ELF64 header         (64 bytes)
\\     0x0040  one program header   (56 bytes)  — PT_LOAD, PF_R|PF_W|PF_X
\\     0x0078  _start stub          (12 bytes)  — 3 instructions
\\     0x0084  user code            (code_size bytes, starts with `main`)
\\
\\ e_entry = 0x400078.  At the entry the stub runs:
\\
\\     LDR   X0, [SP]                ; argc
\\     ADD   X1, SP, #8              ; argv
\\     MOVZ  X28, #0x50, LSL #16     ; X28 = 0x500000 (BSS base)
\\     <user main falls through>
\\
\\ The program header advertises p_memsz large enough to cover the 1 MiB of
\\ LOAD space plus a 16 MiB BSS region starting at 0x500000 — matching the
\\ X28 base the stub loads.
\\
\\ All output bytes accumulate in a64_elf_buf; a64_elf_pos_v tracks the
\\ write cursor. File I/O is via direct syscalls in a `host` composition.

\\ ============================================================================
\\ OUTPUT BUFFER
\\ ============================================================================
\\ 2 MiB is plenty for any minimal-compiler program: ELF header + PHDR is
\\ fixed at 0x78 bytes and the code section from the minimal compiler is
\\ bounded by arm64_buf (1 MiB in 03-arm64-emit side).

buf a64_elf_buf 2097152
buf a64_elf_pos_v 8

\\ ============================================================================
\\ LOW-LEVEL EMIT HELPERS  (little-endian, into a64_elf_buf)
\\ ============================================================================

a64e_b byte :
    \\ Append one byte, advance cursor.
    _p → 64 a64_elf_pos_v
    ← 8 a64_elf_buf + _p byte
    ← 64 a64_elf_pos_v _p + 1

a64e_w val :
    \\ Append a little-endian 16-bit word.
    a64e_b val & 0xFF
    a64e_b (val >> 8) & 0xFF

a64e_d val :
    \\ Append a little-endian 32-bit word.
    a64e_w val & 0xFFFF
    a64e_w (val >> 16) & 0xFFFF

a64e_q val :
    \\ Append a little-endian 64-bit word.
    a64e_d val & 0xFFFFFFFF
    a64e_d (val >> 32) & 0xFFFFFFFF

\\ ============================================================================
\\ arm64_elf_build — build the in-memory ELF image in a64_elf_buf
\\ ============================================================================
\\ Fills a64_elf_buf with EHDR(64) + PHDR(56) + stub(12) + code(code_size).
\\ Leaves a64_elf_pos_v pointing at one-past-the-last written byte — i.e.
\\ equal to the final file size.

arm64_elf_build code_buf code_size :
    ← 64 a64_elf_pos_v 0

    \\ ------------------------------------------------------------------
    \\ ELF64 header (64 bytes)
    \\ ------------------------------------------------------------------
    \\ e_ident: 7F 45 4C 46 02 01 01 00 00 00 00 00 00 00 00 00
    a64e_b 0x7F
    a64e_b 0x45
    a64e_b 0x4C
    a64e_b 0x46
    a64e_b 2                \\ EI_CLASS = ELFCLASS64
    a64e_b 1                \\ EI_DATA  = ELFDATA2LSB
    a64e_b 1                \\ EI_VERSION = EV_CURRENT
    a64e_b 0                \\ EI_OSABI = SYSV
    a64e_b 0
    a64e_b 0
    a64e_b 0
    a64e_b 0
    a64e_b 0
    a64e_b 0
    a64e_b 0
    a64e_b 0

    a64e_w 2                \\ e_type    = ET_EXEC
    a64e_w 183              \\ e_machine = EM_AARCH64
    a64e_d 1                \\ e_version = 1
    a64e_q 0x400078         \\ e_entry   = BASE + EHDR + PHDR
    a64e_q 64               \\ e_phoff   = 0x40
    a64e_q 0                \\ e_shoff   = 0
    a64e_d 0                \\ e_flags
    a64e_w 64               \\ e_ehsize
    a64e_w 56               \\ e_phentsize
    a64e_w 1                \\ e_phnum
    a64e_w 64               \\ e_shentsize
    a64e_w 0                \\ e_shnum
    a64e_w 0                \\ e_shstrndx

    \\ ------------------------------------------------------------------
    \\ Program header (56 bytes) — one PT_LOAD segment, RWX
    \\ ------------------------------------------------------------------
    \\ filesz  = 0x78 (EHDR+PHDR) + 12 (stub) + code_size
    \\ memsz   = 0x100000 (1 MiB) + 0x1000000 (16 MiB BSS) = 0x1100000
    \\ X28 is loaded with 0x500000 by the stub, so the segment must span
    \\ from p_vaddr (0x400000) through 0x500000 + BSS.

    filesz    0x78 + 12 + code_size
    memsz_val 0x1100000

    a64e_d 1                \\ p_type  = PT_LOAD
    a64e_d 7                \\ p_flags = PF_X | PF_W | PF_R
    a64e_q 0                \\ p_offset
    a64e_q 0x400000         \\ p_vaddr
    a64e_q 0x400000         \\ p_paddr
    a64e_q filesz           \\ p_filesz
    a64e_q memsz_val        \\ p_memsz
    a64e_q 0x10000          \\ p_align

    \\ ------------------------------------------------------------------
    \\ _start stub (12 bytes, 3 instructions)
    \\ ------------------------------------------------------------------
    \\ LDR  X0, [SP]              — argc = *SP
    \\ ADD  X1, SP, #8            — argv = SP + 8
    \\ MOVZ X28, #0x50, LSL #16   — X28 = 0x500000 (BSS base)
    \\ Execution then falls through into the caller's first instruction,
    \\ which the minimal compiler arranges to be `main`.

    a64e_d 0xF94003E0       \\ LDR  X0, [SP, #0]
    a64e_d 0x910023E1       \\ ADD  X1, SP, #8
    a64e_d 0xD2A00A1C       \\ MOVZ X28, #0x50, LSL #16

    \\ ------------------------------------------------------------------
    \\ User code (code_size bytes copied verbatim)
    \\ ------------------------------------------------------------------
    for ci 0 code_size 1
        b0 → 8 code_buf + ci
        a64e_b b0
    endfor

\\ ============================================================================
\\ arm64_elf_save — flush a64_elf_buf to disk at `path`
\\ ============================================================================
\\ Writes a64_elf_pos_v bytes starting at a64_elf_buf to `path`,
\\ creating the file (mode 0755) and truncating any existing content.

host arm64_elf_save path :
    \\ openat(AT_FDCWD, path, O_WRONLY|O_CREAT|O_TRUNC, 0755)
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 577
    ↓ $3 493
    trap
    fd ↑ $0

    \\ write(fd, a64_elf_buf + written, remaining)
    written   0
    remaining → 64 a64_elf_pos_v
    a64s_write_loop:
        if<= remaining 0
            goto a64s_write_done
        ↓ $8 64
        ↓ $0 fd
        ↓ $1 a64_elf_buf + written
        ↓ $2 remaining
        trap
        n ↑ $0
        if<= n 0
            goto a64s_write_done
        written   written + n
        remaining remaining - n
        goto a64s_write_loop

    a64s_write_done:
    \\ close(fd)
    ↓ $8 57
    ↓ $0 fd
    trap

\\ ============================================================================
\\ arm64_elf_write — public entry point: build + save
\\ ============================================================================
\\ Builds the ELF image in a64_elf_buf and then writes it to out_path.

arm64_elf_write code_buf code_size out_path :
    arm64_elf_build code_buf code_size
    arm64_elf_save  out_path
