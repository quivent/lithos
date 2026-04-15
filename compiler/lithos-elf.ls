\\ lithos-elf.li — GPU ELF writer for Lithos self-hosting compiler
\\
\\ Ported from gpu/emit.fs (build-cubin) and compiler/elf-wrap.fs.
\\ Takes a GPU code buffer (sm90 instruction stream) and wraps it
\\ in a valid ELF64 binary that cuModuleLoadData can load.
\\
\\ ELF layout (9 sections):
\\   [0] NULL
\\   [1] .shstrtab           — section name string table
\\   [2] .strtab             — symbol name string table
\\   [3] .symtab             — symbol table (4 entries)
\\   [4] .nv.info            — global kernel attributes
\\   [5] .text.<kernel>      — GPU machine code (128-byte aligned)
\\   [6] .nv.info.<kernel>   — per-kernel attributes
\\   [7] .nv.shared.reserved.0  — shared memory reservation (NOBITS)
\\   [8] .nv.constant0.<kernel> — constant bank 0 (driver reserved + params)
\\
\\ This is HOST code (ARM64 target), not GPU kernel code.
\\ Uses Linux syscalls: open, write, close.

\\ ============================================================
\\ CONSTANTS
\\ ============================================================

524288 constant CUBIN_SIZE       \\ 512KB output buffer
64 constant EHDR_SIZE            \\ ELF64 header size
64 constant SHDR_SIZE            \\ section header entry size
24 constant SYM_SIZE             \\ ELF64 symbol table entry size

\\ ELF constants
2 constant ET_EXEC
190 constant EM_CUDA             \\ 0xBE
128 constant EV_CUDA             \\ 0x80 — CUDA ELF version

\\ Section types
0 constant SHT_NULL
1 constant SHT_PROGBITS
2 constant SHT_SYMTAB
3 constant SHT_STRTAB
8 constant SHT_NOBITS
1879048192 constant SHT_LOPROC   \\ 0x70000000 — NVIDIA .nv.info sections

\\ Section flags
2 constant SHF_ALLOC
4 constant SHF_EXECINSTR
64 constant SHF_INFO_LINK        \\ 0x40

\\ EIATTR constants (from probe.cubin decode)
4 constant NVI_FMT_U32
3 constant NVI_FMT_FLAG
47 constant EIATTR_REGCOUNT             \\ 0x2f
17 constant EIATTR_FRAME_SIZE           \\ 0x11
18 constant EIATTR_MIN_STACK_SIZE       \\ 0x12
55 constant EIATTR_CUDA_API_VERSION     \\ 0x37
23 constant EIATTR_KPARAM_INFO          \\ 0x17
10 constant EIATTR_PARAM_CBANK          \\ 0x0a
28 constant EIATTR_EXIT_INSTR_OFFSETS   \\ 0x1c
25 constant EIATTR_CBANK_PARAM_SIZE     \\ 0x19
27 constant EIATTR_MAXREG_COUNT         \\ 0x1b
80 constant EIATTR_SPARSE_MMA_MASK      \\ 0x50
54 constant EIATTR_SW_WAR               \\ 0x36
51 constant EIATTR_CRS_STACK_SIZE       \\ 0x33

\\ Cooperative grid-sync attributes
40 constant EIATTR_COOP_GROUP_INSTR_OFFSETS  \\ 0x28
41 constant EIATTR_COOP_GROUP_MASK_REGIDS    \\ 0x29

\\ CUDA ELF flags (from probe cubin)
94396762 constant ELF_FLAGS_CUDA   \\ 0x5a055a

\\ ARM64 Linux syscall numbers
56 constant NR_OPENAT
57 constant NR_CLOSE
64 constant NR_WRITE

\\ ============================================================
\\ OUTPUT BUFFER STATE
\\ ============================================================
\\ [HOST] These are global variables in the ARM64 .bss/.data sections.
\\ global cubin_buf    CUBIN_SIZE u8     — output buffer
\\ global cubin_pos    1 u32             — current write position
\\
\\ Section metadata (filled during build, used for fixups):
\\ global shstrtab_off   1 u32    global shstrtab_size  1 u32
\\ global strtab_off     1 u32    global strtab_size    1 u32
\\ global symtab_off     1 u32    global symtab_size    1 u32
\\ global nvinfo_off     1 u32    global nvinfo_size    1 u32
\\ global nvinfo_k_off   1 u32    global nvinfo_k_size  1 u32
\\ global text_off       1 u32    global text_size      1 u32
\\ global const0_off     1 u32    global const0_size    1 u32
\\ global shdrs_off      1 u32
\\ global sym_kernel_off 1 u32
\\ global strsym_kernel  1 u32
\\
\\ Section name offsets within .shstrtab:
\\ global SN_shstrtab 1 u32   global SN_strtab  1 u32
\\ global SN_symtab   1 u32   global SN_nvinfo  1 u32
\\ global SN_nvinfo_k 1 u32   global SN_text    1 u32
\\ global SN_const0   1 u32   global SN_shared  1 u32

\\ ============================================================
\\ BYTE EMIT PRIMITIVES
\\ ============================================================
\\ [NOTE: .li host extension needed] Byte-level buffer writes.
\\ cubin_buf and cubin_pos are global state.

cb_emit byte :
    \\ cubin_buf[cubin_pos] = byte; cubin_pos++
    \\ [HOST INTRINSIC] store_u8 cubin_buf cubin_pos byte
    store_u8_at cubin_buf cubin_pos byte
    cubin_pos = cubin_pos + 1

cw_emit val_u16 :
    \\ Emit u16 little-endian (2 bytes)
    lo = val_u16 & 255
    cb_emit lo
    hi = val_u16 >> 8
    cb_emit hi

cd_emit val_u32 :
    \\ Emit u32 little-endian (4 bytes)
    cw_emit (val_u32 & 65535)
    cw_emit (val_u32 >> 16)

cq_emit val_u64 :
    \\ Emit u64 little-endian (8 bytes)
    cd_emit (val_u64 & 4294967295)
    cd_emit (val_u64 >> 32)

cpad target :
    \\ Zero-fill cubin_buf from cubin_pos to target
    loop_pad:
        if>= cubin_pos target
            return
        cb_emit 0
        goto loop_pad

calign n :
    \\ Align cubin_pos to n-byte boundary (n must be power of 2)
    mask = n - 1
    target = (cubin_pos + mask) & (mask ^ 4294967295)    \\ ~mask via XOR with all-ones
    cpad target

\\ ============================================================
\\ PATCH HELPERS — write values at absolute buffer offsets
\\ ============================================================

put_u16 val off :
    \\ Store u16 little-endian at cubin_buf[off]
    store_u8_at cubin_buf off (val & 255)
    store_u8_at cubin_buf (off + 1) ((val >> 8) & 255)

put_u32 val off :
    \\ Store u32 little-endian at cubin_buf[off..off+3]
    store_u8_at cubin_buf off (val & 255)
    store_u8_at cubin_buf (off + 1) ((val >> 8) & 255)
    store_u8_at cubin_buf (off + 2) ((val >> 16) & 255)
    store_u8_at cubin_buf (off + 3) ((val >> 24) & 255)

put_u64 val off :
    \\ Store u64 little-endian at cubin_buf[off..off+7]
    put_u32 (val & 4294967295) off
    put_u32 (val >> 32) (off + 4)

\\ ============================================================
\\ STRING EMIT — write raw bytes from a string into cubin_buf
\\ ============================================================

emit_str src len :
    for i 0 len 1
        byte = load_u8 (src + i)
        cb_emit byte
    endfor

\\ ============================================================
\\ elf_init — Reset output buffer
\\ ============================================================

elf_init :
    cubin_pos = 0

\\ ============================================================
\\ elf_write_header — Emit 64-byte ELF64 header (fields patched later)
\\ ============================================================

elf_write_header :
    \\ e_ident[0..3]: magic number
    cb_emit 127       \\ 0x7F
    cb_emit 69        \\ 'E'
    cb_emit 76        \\ 'L'
    cb_emit 70        \\ 'F'
    cb_emit 2         \\ ELFCLASS64
    cb_emit 1         \\ ELFDATA2LSB
    cb_emit 1         \\ EV_CURRENT
    cb_emit 51        \\ 0x33 — NVIDIA CUDA OS/ABI
    cb_emit 7         \\ ABI version
    cb_emit 0         \\ padding
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    \\ e_type = ET_EXEC (2)
    cw_emit 2
    \\ e_machine = EM_CUDA (0xBE = 190)
    cw_emit 190
    \\ e_version = 0x80 (CUDA)
    cd_emit 128
    \\ e_entry = 0
    cq_emit 0
    \\ e_phoff = 0 (patched later; no program headers)
    cq_emit 0
    \\ e_shoff = 0 (patched later)
    cq_emit 0
    \\ e_flags (CUDA flags from probe)
    cd_emit ELF_FLAGS_CUDA
    \\ e_ehsize = 64
    cw_emit 64
    \\ e_phentsize = 56 (patched to 0 later if no program headers)
    cw_emit 56
    \\ e_phnum = 0 (patched later)
    cw_emit 0
    \\ e_shentsize = 64
    cw_emit 64
    \\ e_shnum = 0 (patched later)
    cw_emit 0
    \\ e_shstrndx = 0 (patched later)
    cw_emit 0

\\ ============================================================
\\ SECTION HEADER EMIT — one 64-byte ELF64 section header
\\ ============================================================

shdr64_a sh_name sh_type sh_flags sh_addr sh_offset :
    cd_emit sh_name
    cd_emit sh_type
    cq_emit sh_flags
    cq_emit sh_addr
    cq_emit sh_offset

shdr64_b sh_size sh_link sh_info sh_addralign sh_entsize :
    cq_emit sh_size
    cd_emit sh_link
    cd_emit sh_info
    cq_emit sh_addralign
    cq_emit sh_entsize

\\ ============================================================
\\ SYMBOL TABLE ENTRY — 24-byte ELF64 Sym
\\ ============================================================

sym64_emit st_name st_info st_other st_shndx st_value st_size :
    cd_emit st_name
    cb_emit st_info
    cb_emit st_other
    cw_emit st_shndx
    cq_emit st_value
    cq_emit st_size

\\ ============================================================
\\ NV.INFO RECORD EMITTERS
\\ ============================================================

\\ Plain u32 record: [fmt u8][attr u8][size=4 u16][val u32]
nvi_u32_emit val attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 4
    cd_emit val

\\ Sym+val record: [fmt u8][attr u8][size=8 u16][sym_idx u32][val u32]
nvi_sval_emit val sym_idx attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 8
    cd_emit sym_idx
    cd_emit val

\\ ============================================================
\\ elf_build — Build complete 9-section GPU ELF
\\ ============================================================
\\ kernel_name:  pointer to kernel name string
\\ kernel_nlen:  byte length of kernel name
\\ code_buf:     pointer to sm90 SASS instruction buffer
\\ code_size:    byte length of SASS code
\\ n_kparams:    number of kernel parameters (8-byte pointers each)
\\ reg_count:    highest register index used by kernel
\\ smem_size:    bytes of static shared memory (0 if none)
\\
\\ Reads globals: cooperative, gridsync_offsets, gridsync_count,
\\                exit_offsets, exit_count
\\
\\ After return, cubin_buf[0..cubin_pos) contains the complete ELF.

elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size :

    \\ ---- Step 1: ELF header placeholder (64 bytes) ----
    elf_init
    elf_write_header

    \\ ---- Step 2: .shstrtab (section 1) ----
    shstrtab_off = cubin_pos
    cb_emit 0                                 \\ null byte at offset 0

    SN_shstrtab = cubin_pos - shstrtab_off
    \\ ".shstrtab\0"
    cb_emit 46    \\ '.'
    cb_emit 115   \\ 's'
    cb_emit 104   \\ 'h'
    cb_emit 115   \\ 's'
    cb_emit 116   \\ 't'
    cb_emit 114   \\ 'r'
    cb_emit 116   \\ 't'
    cb_emit 97    \\ 'a'
    cb_emit 98    \\ 'b'
    cb_emit 0

    SN_strtab = cubin_pos - shstrtab_off
    \\ ".strtab\0"
    cb_emit 46    \\ '.'
    cb_emit 115   \\ 's'
    cb_emit 116   \\ 't'
    cb_emit 114   \\ 'r'
    cb_emit 116   \\ 't'
    cb_emit 97    \\ 'a'
    cb_emit 98    \\ 'b'
    cb_emit 0

    SN_symtab = cubin_pos - shstrtab_off
    \\ ".symtab\0"
    cb_emit 46    \\ '.'
    cb_emit 115   \\ 's'
    cb_emit 121   \\ 'y'
    cb_emit 109   \\ 'm'
    cb_emit 116   \\ 't'
    cb_emit 97    \\ 'a'
    cb_emit 98    \\ 'b'
    cb_emit 0

    SN_nvinfo = cubin_pos - shstrtab_off
    \\ ".nv.info\0"
    cb_emit 46    \\ '.'
    cb_emit 110   \\ 'n'
    cb_emit 118   \\ 'v'
    cb_emit 46    \\ '.'
    cb_emit 105   \\ 'i'
    cb_emit 110   \\ 'n'
    cb_emit 102   \\ 'f'
    cb_emit 111   \\ 'o'
    cb_emit 0

    SN_nvinfo_k = cubin_pos - shstrtab_off
    \\ ".nv.info.<kernel>\0"
    cb_emit 46    \\ '.'
    cb_emit 110   \\ 'n'
    cb_emit 118   \\ 'v'
    cb_emit 46    \\ '.'
    cb_emit 105   \\ 'i'
    cb_emit 110   \\ 'n'
    cb_emit 102   \\ 'f'
    cb_emit 111   \\ 'o'
    cb_emit 46    \\ '.'
    emit_str kernel_name kernel_nlen
    cb_emit 0

    SN_text = cubin_pos - shstrtab_off
    \\ ".text.<kernel>\0"
    cb_emit 46    \\ '.'
    cb_emit 116   \\ 't'
    cb_emit 101   \\ 'e'
    cb_emit 120   \\ 'x'
    cb_emit 116   \\ 't'
    cb_emit 46    \\ '.'
    emit_str kernel_name kernel_nlen
    cb_emit 0

    SN_const0 = cubin_pos - shstrtab_off
    \\ ".nv.constant0.<kernel>\0"
    cb_emit 46    \\ '.'
    cb_emit 110   \\ 'n'
    cb_emit 118   \\ 'v'
    cb_emit 46    \\ '.'
    cb_emit 99    \\ 'c'
    cb_emit 111   \\ 'o'
    cb_emit 110   \\ 'n'
    cb_emit 115   \\ 's'
    cb_emit 116   \\ 't'
    cb_emit 97    \\ 'a'
    cb_emit 110   \\ 'n'
    cb_emit 116   \\ 't'
    cb_emit 48    \\ '0'
    cb_emit 46    \\ '.'
    emit_str kernel_name kernel_nlen
    cb_emit 0

    SN_shared = cubin_pos - shstrtab_off
    \\ ".nv.shared.reserved.0\0"
    cb_emit 46    \\ '.'
    cb_emit 110   \\ 'n'
    cb_emit 118   \\ 'v'
    cb_emit 46    \\ '.'
    cb_emit 115   \\ 's'
    cb_emit 104   \\ 'h'
    cb_emit 97    \\ 'a'
    cb_emit 114   \\ 'r'
    cb_emit 101   \\ 'e'
    cb_emit 100   \\ 'd'
    cb_emit 46    \\ '.'
    cb_emit 114   \\ 'r'
    cb_emit 101   \\ 'e'
    cb_emit 115   \\ 's'
    cb_emit 101   \\ 'e'
    cb_emit 114   \\ 'r'
    cb_emit 118   \\ 'v'
    cb_emit 101   \\ 'e'
    cb_emit 100   \\ 'd'
    cb_emit 46    \\ '.'
    cb_emit 48    \\ '0'
    cb_emit 0

    calign 4
    shstrtab_size = cubin_pos - shstrtab_off

    \\ ---- Step 3: .strtab (section 2) ----
    strtab_off = cubin_pos
    cb_emit 0     \\ sym0 empty name
    cb_emit 0     \\ sym1 (SECTION) empty name
    strsym_kernel = cubin_pos - strtab_off
    emit_str kernel_name kernel_nlen
    cb_emit 0
    strtab_size = cubin_pos - strtab_off

    \\ ---- Step 4: .symtab (section 3; 4 entries x 24 = 96 bytes) ----
    calign 8
    symtab_off = cubin_pos

    \\ sym0: UNDEF
    sym64_emit 0 0 0 0 0 0

    \\ sym1: SECTION LOCAL, shndx=5 (.text.<kernel>)
    sym64_emit 0 3 0 5 0 0

    \\ sym2: SECTION LOCAL, shndx=8 (.nv.constant0.<kernel>)
    sym64_emit 0 3 0 8 0 0

    \\ sym3: FUNC GLOBAL STO_CUDA_ENTRY shndx=5; st_size patched after .text
    sym_kernel_off = cubin_pos
    sym64_emit strsym_kernel 18 16 5 0 0
    \\ st_info=0x12 (STB_GLOBAL|STT_FUNC), st_other=0x10 (STO_CUDA_ENTRY)

    symtab_size = cubin_pos - symtab_off

    \\ ---- Step 5: .nv.info (section 4; global attributes) ----
    calign 4
    nvinfo_off = cubin_pos

    \\ REGCOUNT (sym_idx=3 = kernel symbol)
    reg_val = reg_count + 1
    if< reg_val 8
        reg_val = 8
    nvi_sval_emit reg_val 3 EIATTR_REGCOUNT NVI_FMT_U32
    \\ FRAME_SIZE = 0
    nvi_sval_emit 0 3 EIATTR_FRAME_SIZE NVI_FMT_U32
    \\ MIN_STACK_SIZE = 0
    nvi_sval_emit 0 3 EIATTR_MIN_STACK_SIZE NVI_FMT_U32

    nvinfo_size = cubin_pos - nvinfo_off

    \\ ---- Step 6: .nv.info.<kernel> (section 6; per-kernel attributes) ----
    calign 4
    nvinfo_k_off = cubin_pos

    \\ CUDA_API_VERSION = 0x80
    nvi_u32_emit 128 EIATTR_CUDA_API_VERSION NVI_FMT_U32

    \\ KPARAM_INFO: one 12-byte record per param, in REVERSE ordinal order
    \\ Record: [fmt=0x04][attr=0x17][size=0x000c][index=0][offset<<16|ordinal][0x0021f000]
    for pi 0 n_kparams 1
        ordinal = n_kparams - 1 - pi
        cb_emit 4                                  \\ fmt = NVI_FMT_U32
        cb_emit EIATTR_KPARAM_INFO
        cw_emit 12                                 \\ size = 12 bytes
        cd_emit 0                                  \\ index
        offset_ord = (ordinal * 8) << 16
        offset_ord = offset_ord | ordinal
        cd_emit offset_ord                         \\ (offset<<16)|ordinal
        cd_emit 2171904                            \\ 0x0021f000 — flags=0xf000, size=8 ptr
    endfor

    \\ SPARSE_MMA_MASK (FLAG format: fmt=03 attr=0x50 val=0)
    cb_emit 3
    cb_emit EIATTR_SPARSE_MMA_MASK
    cb_emit 0
    cb_emit 0

    \\ MAXREG_COUNT = 0xff (HVAL format: fmt=03 attr=0x1b val_u16=0x00ff)
    cb_emit 3
    cb_emit EIATTR_MAXREG_COUNT
    cb_emit 255
    cb_emit 0

    \\ ---- Cooperative grid-sync attributes (if cooperative=1) ----
    if== cooperative 1
        \\ COOP_GROUP_MASK_REGIDS: fmt=0x04 attr=0x29 size=16; four 0xffffffff words
        cb_emit 4
        cb_emit EIATTR_COOP_GROUP_MASK_REGIDS
        cw_emit 16
        cd_emit 4294967295     \\ 0xffffffff
        cd_emit 4294967295
        cd_emit 4294967295
        cd_emit 4294967295

        \\ COOP_GROUP_INSTR_OFFSETS: fmt=0x04 attr=0x28 size=N*4
        cb_emit 4
        cb_emit EIATTR_COOP_GROUP_INSTR_OFFSETS
        cw_emit (gridsync_count * 4)
        for gi 0 gridsync_count 1
            gs_off = load_u32 (gridsync_offsets + gi * 4)
            cd_emit gs_off
        endfor

    \\ EXIT_INSTR_OFFSETS: emit actual EXIT byte offsets tracked during emission
    cb_emit 4
    cb_emit EIATTR_EXIT_INSTR_OFFSETS
    if> exit_count 0
        cw_emit (exit_count * 4)
        for ei 0 exit_count 1
            ex_off = load_u32 (exit_offsets + ei * 4)
            cd_emit ex_off
        endfor
    if== exit_count 0
        cw_emit 4
        cd_emit code_size - 16

    \\ CBANK_PARAM_SIZE (HVAL; total param bytes)
    param_bytes = n_kparams * 8
    cb_emit 3
    cb_emit EIATTR_CBANK_PARAM_SIZE
    cw_emit param_bytes

    \\ PARAM_CBANK (SVAL: sym_idx=2, cbank_offset=0x210, param_bytes in high 16)
    cb_emit 4
    cb_emit EIATTR_PARAM_CBANK
    cw_emit 8
    cd_emit 2                                  \\ sym_idx = constant0 section symbol
    cbank_val = (param_bytes << 16) | 528      \\ 528 = 0x210
    cd_emit cbank_val

    \\ SW_WAR workaround flag
    cb_emit 4
    cb_emit EIATTR_SW_WAR
    cw_emit 4
    cd_emit 8

    \\ MAX_SHARED_MEM_PER_BLOCK_OPTIN (attr=0x33; only if smem_size > 0)
    if> smem_size 0
        cb_emit 4
        cb_emit 51     \\ 0x33
        cw_emit 4
        cd_emit smem_size

    nvinfo_k_size = cubin_pos - nvinfo_k_off

    \\ ---- Step 7: .text.<kernel> (section 5; 128-byte aligned SASS code) ----
    calign 128
    text_off = cubin_pos

    \\ Copy code_buf[0..code_size) into cubin_buf
    \\ If code_size is 0, emit minimal NOP+NOP+EXIT (48 bytes placeholder)
    if== code_size 0
        \\ [NOTE: caller should provide valid SASS; this is a safety fallback]
        for zb 0 48 1
            cb_emit 0
        endfor
    else
        for ci 0 code_size 1
            byte = load_u8 (code_buf + ci)
            cb_emit byte
        endfor

    text_size = cubin_pos - text_off

    \\ Patch kernel symbol st_size (u64 at sym_kernel_off + 16)
    put_u64 text_size (sym_kernel_off + 16)

    \\ ---- Step 8: .nv.constant0.<kernel> ----
    \\ 0x210 bytes reserved for driver (blockDim, gridDim, etc.)
    \\ + param_bytes for user kernel parameters
    calign 4
    const0_off = cubin_pos
    const0_total = 528 + param_bytes    \\ 528 = 0x210
    for c0i 0 const0_total 1
        cb_emit 0
    endfor
    const0_size = cubin_pos - const0_off

    \\ ---- Step 9: Section headers (9 x 64 = 576 bytes) ----
    calign 64
    shdrs_off = cubin_pos

    \\ [0] NULL
    shdr64_a 0 0 0 0 0
    shdr64_b 0 0 0 0 0

    \\ [1] .shstrtab (SHT_STRTAB)
    shdr64_a SN_shstrtab SHT_STRTAB 0 0 shstrtab_off
    shdr64_b shstrtab_size 0 0 1 0

    \\ [2] .strtab (SHT_STRTAB)
    shdr64_a SN_strtab SHT_STRTAB 0 0 strtab_off
    shdr64_b strtab_size 0 0 1 0

    \\ [3] .symtab (SHT_SYMTAB, link=2=.strtab, info=3=first_global)
    shdr64_a SN_symtab SHT_SYMTAB 0 0 symtab_off
    shdr64_b symtab_size 2 3 8 24

    \\ [4] .nv.info (SHT_LOPROC, link=3=.symtab)
    shdr64_a SN_nvinfo SHT_LOPROC 0 0 nvinfo_off
    shdr64_b nvinfo_size 3 0 4 0

    \\ [5] .text.<kernel> (SHT_PROGBITS, flags=ALLOC|EXECINSTR, link=3, info=3)
    shdr64_a SN_text SHT_PROGBITS 6 0 text_off
    shdr64_b text_size 3 3 128 0

    \\ [6] .nv.info.<kernel> (SHT_LOPROC, flags=0x40=INFO_LINK, link=3, info=5=.text)
    shdr64_a SN_nvinfo_k SHT_LOPROC SHF_INFO_LINK 0 nvinfo_k_off
    shdr64_b nvinfo_k_size 3 5 4 0

    \\ [7] .nv.shared.reserved.0 (SHT_NOBITS, flags=ALLOC|EXECINSTR)
    \\ sh_size = smem_size (NOBITS — no file content)
    shdr64_a SN_shared SHT_NOBITS 3 0 const0_off
    shdr64_b smem_size 0 0 16 0

    \\ [8] .nv.constant0.<kernel> (SHT_PROGBITS, flags=0x42=ALLOC|INFO_LINK, info=5=.text)
    shdr64_a SN_const0 SHT_PROGBITS 66 0 const0_off
    shdr64_b const0_size 0 5 4 0

    \\ ---- Step 10: Patch ELF header ----
    put_u64 0 32           \\ e_phoff = 0 (no program headers)
    put_u64 shdrs_off 40   \\ e_shoff
    put_u16 0 54           \\ e_phentsize = 0
    put_u16 0 56           \\ e_phnum = 0
    put_u16 9 60           \\ e_shnum = 9
    put_u16 1 62           \\ e_shstrndx = 1

\\ ============================================================
\\ elf_save — Write cubin_buf to a file
\\ ============================================================
\\ path: null-terminated file path string
\\ Uses ARM64 Linux syscalls: openat, write, close.
\\ open flags: O_WRONLY|O_CREAT|O_TRUNC = 0x241 = 577
\\ mode: 0644 = 420

elf_save path :
    \\ openat(AT_FDCWD, path, O_WRONLY|O_CREAT|O_TRUNC, 0644)
    syscall fd NR_OPENAT -100 path 577 420

    \\ write(fd, cubin_buf, cubin_pos)
    \\ [NOTE: large writes may need a loop; Linux write() can return short]
    written = 0
    remaining = cubin_pos
    loop_write:
        if<= remaining 0
            goto done_write
        write_ptr = cubin_buf + written
        syscall n NR_WRITE fd write_ptr remaining
        if<= n 0
            goto done_write    \\ error or EOF
        written = written + n
        remaining = remaining - n
        goto loop_write

    done_write:
    \\ close(fd)
    syscall ret NR_CLOSE fd

\\ ============================================================
\\ elf_write_cubin — Convenience: build ELF and write to file
\\ ============================================================
\\ One-call interface matching elf-wrap.fs write-elf.

elf_write_cubin kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size out_path :
    elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size
    elf_save out_path

\\ ============================================================
\\ elf_write_simple — Non-cooperative kernel (clears cooperative/gridsync state)
\\ ============================================================

elf_write_simple kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size out_path :
    cooperative 0
    gridsync_count 0
    exit_count 0
    elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size
    elf_save out_path
