\\ elf_load.ls — Load a compiled GPU ELF into BAR4 HBM and extract entry_pc.
\\
\\ Replaces cuModuleLoadData + cuModuleGetFunction.
\\
\\ Input: a GPU ELF produced by compiler/lithos-elf.ls, resident in host memory
\\ at cpu_va with byte length elf_size.
\\
\\ Output: the GPU VA of the kernel's .text section (which becomes the base for
\\ entry_pc in the QMD) plus the symbol's offset within .text. The caller adds
\\ the two to form the 40-bit entry_pc written into QMD byte 0x0118.
\\
\\ ELF64 layout we consume (produced by lithos-elf.ls):
\\   Ehdr at byte 0 (64 bytes)
\\       e_shoff at 0x28 (64-bit)
\\       e_shnum at 0x3c (16-bit)
\\       e_shstrndx at 0x3e (16-bit)
\\   Shdr entries at e_shoff (64 bytes each)
\\       sh_name   at 0x00 (u32, offset into .shstrtab)
\\       sh_type   at 0x04 (u32)
\\       sh_offset at 0x18 (u64, byte offset in ELF file)
\\       sh_size   at 0x20 (u64)
\\   Symtab entries (24 bytes each)
\\       st_name  at 0x00 (u32)
\\       st_value at 0x08 (u64)
\\       st_size  at 0x10 (u64)

\\ ============================================================
\\ ELF HEADER READ — returns e_shoff, e_shnum, e_shstrndx
\\ ============================================================

elf_read_header elf_va :
    e_shoff → 64 elf_va + 0x28
    e_shnum → 16 elf_va + 0x3c
    e_shstrndx → 16 elf_va + 0x3e

\\ ============================================================
\\ SHDR HELPERS
\\ ============================================================

\\ Absolute VA of section header with index shdr_idx.
elf_shdr_offset elf_va e_shoff shdr_idx :
    shdr_va elf_va + e_shoff + shdr_idx * 64

\\ VA of sh_name string (pointer into shstrtab).
elf_shdr_name shdr_va shstrtab_base :
    sh_name → 32 shdr_va + 0x00
    name_ptr shstrtab_base + sh_name

\\ Section raw data pointer (CPU VA) and byte size.
elf_section_data elf_va shdr_va :
    sh_offset → 64 shdr_va + 0x18
    sh_size → 64 shdr_va + 0x20
    data_ptr elf_va + sh_offset
    size sh_size

\\ ============================================================
\\ STRING COMPARISON — needle is length-prefixed by nlen
\\ ============================================================
\\ Returns match = 1 if the C string at hay_ptr equals the nlen bytes at
\\ needle_ptr (and hay has a terminating NUL immediately after). Else 0.

elf_streq hay_ptr needle_ptr nlen :
    match 1
    for i 0 nlen 1
        h → 8 hay_ptr + i
        n → 8 needle_ptr + i
        if== h n
            match match
        if< h n
            match 0
        if< n h
            match 0
    tail → 8 hay_ptr + nlen
    if< 0 tail
        match 0

\\ ============================================================
\\ FIND SECTION BY NAME — iterate shdr table, match against needle bytes
\\ ============================================================
\\ Returns shdr_va = 0 if no match found.

elf_find_section elf_va needle_ptr nlen :
    elf_read_header elf_va
    shstrtab_shdr elf_va + e_shoff + e_shstrndx * 64
    shstrtab_off → 64 shstrtab_shdr + 0x18
    shstrtab_base elf_va + shstrtab_off
    shdr_va 0
    for i 0 e_shnum 1
        cand elf_va + e_shoff + i * 64
        sh_name → 32 cand + 0x00
        name_ptr shstrtab_base + sh_name
        elf_streq name_ptr needle_ptr nlen
        if== match 1
            shdr_va cand

\\ ============================================================
\\ GPU MEMORY COPY — allocate BAR4 HBM, copy CPU bytes to it
\\ ============================================================
\\ hbm_alloc is a bump allocator over the BAR4-mapped HBM region; its state
\\ lives at hbm_alloc_state (u64 cursor, aligned up to 256 bytes per call).

elf_gpu_alloc size :
    cur → 64 hbm_alloc_state
    gpu_va cur
    next cur + size
    next next + 255
    mask next / 256 * 256
    ← 64 hbm_alloc_state mask

\\ Copy nbytes from CPU VA src to GPU VA dst via BAR4 window. BAR4 is identity-
\\ mapped into our ARM64 VA space at bar4_host_base, so a GPU VA gpu_va maps to
\\ host VA (bar4_host_base + (gpu_va - bar4_gpu_base)).
elf_copy_to_gpu src dst nbytes :
    host_base → 64 bar4_host_base
    gpu_base → 64 bar4_gpu_base
    host_dst host_base + dst - gpu_base
    for i 0 nbytes 8
        w → 64 src + i
        ← 64 host_dst + i w

\\ Allocate GPU memory for a section and copy its bytes; returns gpu_va.
elf_section_to_gpu elf_va shdr_va :
    elf_section_data elf_va shdr_va
    gpu_va elf_gpu_alloc size
    elf_copy_to_gpu data_ptr gpu_va size
    text_gpu_va gpu_va
    text_size size

\\ ============================================================
\\ SYMBOL TABLE LOOKUP
\\ ============================================================
\\ Walk .symtab; for each entry, resolve st_name through the .strtab linked
\\ by .symtab's sh_link (always 2 in our cubin writer). Match against
\\ kernel_name/kernel_nlen; return st_value (offset within .text).

\\ Needle literal ".symtab\0" lives in .rodata. Callers pass a pointer.
elf_find_kernel_symbol elf_va symtab_name_ptr strtab_name_ptr kernel_name kernel_nlen :
    elf_find_section elf_va symtab_name_ptr 7
    symtab_shdr shdr_va
    elf_find_section elf_va strtab_name_ptr 7
    strtab_shdr shdr_va
    elf_section_data elf_va symtab_shdr
    symtab_base data_ptr
    symtab_bytes size
    elf_section_data elf_va strtab_shdr
    strtab_base data_ptr
    entry_offset 0
    n_syms symtab_bytes / 24
    for i 0 n_syms 1
        sym symtab_base + i * 24
        st_name → 32 sym + 0x00
        st_value → 64 sym + 0x08
        name_ptr strtab_base + st_name
        elf_streq name_ptr kernel_name kernel_nlen
        if== match 1
            entry_offset st_value

\\ ============================================================
\\ TOP-LEVEL: elf_load
\\ ============================================================
\\ Given an ELF resident at CPU VA elf_cpu_va (elf_size bytes) plus the
\\ kernel name and length, this:
\\   1. Finds .text.<kernel>
\\   2. Allocates GPU HBM, copies .text bytes over BAR4
\\   3. Finds the kernel symbol and reads st_value
\\
\\ Output slots (passed by caller as pointers):
\\   out_text_gpu_va  — 64-bit GPU VA of .text section base
\\   out_text_size    — 64-bit byte length of .text
\\   out_entry_off    — 64-bit offset of kernel entry within .text
\\
\\ Caller forms entry_pc = out_text_gpu_va + out_entry_off and writes it to
\\ QMD byte 0x0118 via qmd_set_entry_pc.
\\
\\ text_needle_ptr/text_needle_nlen is the prebuilt ".text.<kernel>" string
\\ that matches the section the compiler emits. The caller is expected to
\\ build this needle (".text." + kernel_name + NUL) in a scratch buffer; the
\\ rest of the symbol table probe reuses kernel_name directly.

elf_load elf_cpu_va elf_size text_needle_ptr text_needle_nlen symtab_name_ptr strtab_name_ptr kernel_name kernel_nlen out_text_gpu_va out_text_size out_entry_off :
    elf_find_section elf_cpu_va text_needle_ptr text_needle_nlen
    text_shdr shdr_va
    elf_section_to_gpu elf_cpu_va text_shdr
    ← 64 out_text_gpu_va text_gpu_va
    ← 64 out_text_size text_size
    elf_find_kernel_symbol elf_cpu_va symtab_name_ptr strtab_name_ptr kernel_name kernel_nlen
    ← 64 out_entry_off entry_offset
