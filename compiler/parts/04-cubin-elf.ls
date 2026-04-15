\\ ============================================================================
\\ 04-cubin-elf.ls — NVIDIA SM90 cubin (ELF64) writer
\\ ============================================================================
\\
\\ Entry point:
\\   cubin_write kernel_name kernel_nlen code_buf code_size
\\               n_kparams reg_count smem_size out_path
\\
\\ Produces a 9-section cubin ELF64:
\\   [0] NULL
\\   [1] .shstrtab                      (section name strings)
\\   [2] .strtab                        (symbol name strings)
\\   [3] .symtab                        (4 symbols: null, strtab, text, kernel)
\\   [4] .nv.info                       (REGCOUNT, FRAME_SIZE, MIN_STACK_SIZE)
\\   [5] .text.<kernel>                 (SASS bytes, 128-byte aligned)
\\   [6] .nv.info.<kernel>              (per-kernel EIATTR records)
\\   [7] .nv.shared.reserved.0          (NOBITS; size = smem_size)
\\   [8] .nv.constant0.<kernel>         (528 + n_kparams*8 zero bytes)
\\
\\ Then a 9-entry section header table; ELF header e_shoff/e_shnum patched in.

\\ ============================================================================
\\ ELF / EIATTR constants
\\ ============================================================================

0          constant SHT_NULL
1          constant SHT_PROGBITS
2          constant SHT_SYMTAB
3          constant SHT_STRTAB
8          constant SHT_NOBITS
1879048192 constant SHT_LOPROC

2  constant SHF_ALLOC
4  constant SHF_EXECINSTR
64 constant SHF_INFO_LINK

94396762 constant ELF_FLAGS_CUDA

0x04 constant NVI_FMT_U32
0x03 constant NVI_FMT_FLAG

0x2F constant EIATTR_REGCOUNT
0x11 constant EIATTR_FRAME_SIZE
0x12 constant EIATTR_MIN_STACK_SIZE
0x37 constant EIATTR_CUDA_API_VERSION
0x17 constant EIATTR_KPARAM_INFO
0x0A constant EIATTR_PARAM_CBANK
0x1C constant EIATTR_EXIT_INSTR_OFFSETS
0x19 constant EIATTR_CBANK_PARAM_SIZE
0x1B constant EIATTR_MAXREG_COUNT
0x50 constant EIATTR_SPARSE_MMA_MASK
0x36 constant EIATTR_SW_WAR

\\ Syscall numbers (Linux aarch64 host)
56  constant SYS_OPENAT
64  constant SYS_WRITE
57  constant SYS_CLOSE

\\ ============================================================================
\\ Output buffer + position
\\ ============================================================================

buf cubin_buf 4194304
buf cubin_pos_v 8

\\ Section offsets / sizes (globals; shared across sub-compositions)
buf c_shstrtab_off_v  8
buf c_shstrtab_size_v 8
buf c_strtab_off_v    8
buf c_strtab_size_v   8
buf c_symtab_off_v    8
buf c_symtab_size_v   8
buf c_nvinfo_off_v    8
buf c_nvinfo_size_v   8
buf c_nvinfok_off_v   8
buf c_nvinfok_size_v  8
buf c_text_off_v      8
buf c_text_size_v     8
buf c_const0_off_v    8
buf c_const0_size_v   8
buf c_shdrs_off_v     8
buf c_sym_kernel_off_v 8
buf c_strsym_kernel_v 8
buf c_param_bytes_v   8

\\ Section-name offsets within .shstrtab
buf c_SN_shstrtab_v  8
buf c_SN_strtab_v    8
buf c_SN_symtab_v    8
buf c_SN_nvinfo_v    8
buf c_SN_nvinfo_k_v  8
buf c_SN_text_v      8
buf c_SN_const0_v    8
buf c_SN_shared_v    8

\\ ============================================================================
\\ Byte emit primitives (write into cubin_buf at cubin_pos_v, bump position)
\\ ============================================================================

cb_emit b :
    _cp → 64 cubin_pos_v
    ← 8 cubin_buf + _cp b
    ← 64 cubin_pos_v _cp + 1

cw_emit v :
    cb_emit v & 0xFF
    cb_emit (v >> 8) & 0xFF

cd_emit v :
    cw_emit v & 0xFFFF
    cw_emit (v >> 16) & 0xFFFF

cq_emit v :
    cd_emit v & 0xFFFFFFFF
    cd_emit (v >> 32) & 0xFFFFFFFF

\\ Pad with zeros until cubin_pos_v == target
cpad target :
    label cpad_loop
    _cp → 64 cubin_pos_v
    if>= _cp target
        goto cpad_done
    cb_emit 0
    goto cpad_loop
    label cpad_done

\\ Align cubin_pos_v up to a multiple of n (n must be a power of two)
calign n :
    mask n - 1
    _cp → 64 cubin_pos_v
    target (_cp + mask) & (mask ^ 4294967295)
    cpad target

\\ ============================================================================
\\ In-place patch helpers (overwrite bytes at a fixed file offset)
\\ ============================================================================

elf_put_u16 val off :
    ← 8 cubin_buf + off val & 0xFF
    ← 8 cubin_buf + (off + 1) (val >> 8) & 0xFF

elf_put_u32 val off :
    ← 8 cubin_buf + off val & 0xFF
    ← 8 cubin_buf + (off + 1) (val >> 8) & 0xFF
    ← 8 cubin_buf + (off + 2) (val >> 16) & 0xFF
    ← 8 cubin_buf + (off + 3) (val >> 24) & 0xFF

elf_put_u64 val off :
    elf_put_u32 val & 0xFFFFFFFF off
    elf_put_u32 (val >> 32) & 0xFFFFFFFF off + 4

\\ ============================================================================
\\ String emit (copy len bytes from src into cubin buffer)
\\ ============================================================================

elf_emit_str src len :
    i 0
    label es_loop
    if>= i len
        goto es_done
    byte → 8 src + i
    cb_emit byte
    i i + 1
    goto es_loop
    label es_done

\\ ============================================================================
\\ Section-header / symbol / nv.info record emitters
\\ ============================================================================

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

sym64_emit st_name st_info st_other st_shndx st_value st_size :
    cd_emit st_name
    cb_emit st_info
    cb_emit st_other
    cw_emit st_shndx
    cq_emit st_value
    cq_emit st_size

nvi_u32_emit val attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 4
    cd_emit val

nvi_sval_emit val sym_idx attr fmt :
    cb_emit fmt
    cb_emit attr
    cw_emit 8
    cd_emit sym_idx
    cd_emit val

\\ ============================================================================
\\ elf_init / elf_write_header
\\ ============================================================================

elf_init :
    ← 64 cubin_pos_v 0

\\ ELF64 header: e_ident[16], e_type, e_machine, e_version, e_entry,
\\ e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum,
\\ e_shentsize, e_shnum, e_shstrndx.  e_shoff / e_shnum are patched later.
elf_write_header :
    cb_emit 127
    cb_emit 69
    cb_emit 76
    cb_emit 70
    cb_emit 2
    cb_emit 1
    cb_emit 1
    cb_emit 51
    cb_emit 7
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cb_emit 0
    cw_emit 2
    cw_emit 190
    cd_emit 128
    cq_emit 0
    cq_emit 0
    cq_emit 0
    cd_emit ELF_FLAGS_CUDA
    cw_emit 64
    cw_emit 56
    cw_emit 0
    cw_emit 64
    cw_emit 0
    cw_emit 0

\\ ============================================================================
\\ Sub-composition: emit .shstrtab (section name strings)
\\
\\ Records each section-name offset (relative to .shstrtab start) into the
\\ c_SN_* globals so the section headers can reference them later.
\\ ============================================================================

\\ Helper: record (cubin_pos - shstrtab_off) into the named SN slot
c_sn_set slot_v :
    _cp → 64 cubin_pos_v
    _eso → 64 c_shstrtab_off_v
    ← 64 slot_v _cp - _eso

\\ Per-section literal-name emitters. Each writes the section name plus a
\\ trailing NUL into cubin_buf (kernel-suffixed names also append the kernel).
c_emit_n_shstrtab :
    c_sn_set c_SN_shstrtab_v
    cb_emit 46 cb_emit 115 cb_emit 104 cb_emit 115
    cb_emit 116 cb_emit 114 cb_emit 116 cb_emit 97
    cb_emit 98 cb_emit 0

c_emit_n_strtab :
    c_sn_set c_SN_strtab_v
    cb_emit 46 cb_emit 115 cb_emit 116 cb_emit 114
    cb_emit 116 cb_emit 97 cb_emit 98 cb_emit 0

c_emit_n_symtab :
    c_sn_set c_SN_symtab_v
    cb_emit 46 cb_emit 115 cb_emit 121 cb_emit 109
    cb_emit 116 cb_emit 97 cb_emit 98 cb_emit 0

c_emit_n_nvinfo :
    c_sn_set c_SN_nvinfo_v
    cb_emit 46 cb_emit 110 cb_emit 118 cb_emit 46
    cb_emit 105 cb_emit 110 cb_emit 102 cb_emit 111
    cb_emit 0

c_emit_n_nvinfo_k kernel_name kernel_nlen :
    c_sn_set c_SN_nvinfo_k_v
    cb_emit 46 cb_emit 110 cb_emit 118 cb_emit 46
    cb_emit 105 cb_emit 110 cb_emit 102 cb_emit 111
    cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

c_emit_n_text kernel_name kernel_nlen :
    c_sn_set c_SN_text_v
    cb_emit 46 cb_emit 116 cb_emit 101 cb_emit 120
    cb_emit 116 cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

c_emit_n_const0 kernel_name kernel_nlen :
    c_sn_set c_SN_const0_v
    cb_emit 46 cb_emit 110 cb_emit 118 cb_emit 46
    cb_emit 99 cb_emit 111 cb_emit 110 cb_emit 115
    cb_emit 116 cb_emit 97 cb_emit 110 cb_emit 116
    cb_emit 48 cb_emit 46
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0

c_emit_n_shared :
    c_sn_set c_SN_shared_v
    cb_emit 46 cb_emit 110 cb_emit 118 cb_emit 46
    cb_emit 115 cb_emit 104 cb_emit 97 cb_emit 114
    cb_emit 101 cb_emit 100 cb_emit 46 cb_emit 114
    cb_emit 101 cb_emit 115 cb_emit 101 cb_emit 114
    cb_emit 118 cb_emit 101 cb_emit 100 cb_emit 46
    cb_emit 48 cb_emit 0

elf_emit_shstrtab kernel_name kernel_nlen :
    _cp → 64 cubin_pos_v
    ← 64 c_shstrtab_off_v _cp
    cb_emit 0
    c_emit_n_shstrtab
    c_emit_n_strtab
    c_emit_n_symtab
    c_emit_n_nvinfo
    c_emit_n_nvinfo_k kernel_name kernel_nlen
    c_emit_n_text kernel_name kernel_nlen
    c_emit_n_const0 kernel_name kernel_nlen
    c_emit_n_shared
    calign 4
    _cp2 → 64 cubin_pos_v
    _eso → 64 c_shstrtab_off_v
    ← 64 c_shstrtab_size_v _cp2 - _eso

\\ ============================================================================
\\ Sub-composition: emit .strtab and .symtab
\\
\\ .strtab: [""][kernel_name]
\\ .symtab: 4 entries — NULL, strtab section, text section, kernel (GLOBAL FUNC)
\\ The kernel symbol's st_value (offset 0 in .text) and st_size (code length)
\\ are patched in elf_emit_text_const0 once .text has been written.
\\ ============================================================================

elf_emit_strtab_symtab kernel_name kernel_nlen :
    _cp → 64 cubin_pos_v
    ← 64 c_strtab_off_v _cp
    cb_emit 0
    cb_emit 0
    _cp2 → 64 cubin_pos_v
    _sto → 64 c_strtab_off_v
    ← 64 c_strsym_kernel_v _cp2 - _sto
    elf_emit_str kernel_name kernel_nlen
    cb_emit 0
    _cp3 → 64 cubin_pos_v
    ← 64 c_strtab_size_v _cp3 - _sto

    calign 8
    _sym_start → 64 cubin_pos_v
    ← 64 c_symtab_off_v _sym_start
    sym64_emit 0 0 0 0 0 0
    sym64_emit 0 3 0 5 0 0
    sym64_emit 0 3 0 8 0 0
    _cp4 → 64 cubin_pos_v
    ← 64 c_sym_kernel_off_v _cp4
    _sk → 64 c_strsym_kernel_v
    sym64_emit _sk 18 16 5 0 0
    _cp5 → 64 cubin_pos_v
    ← 64 c_symtab_size_v _cp5 - _sym_start

\\ ============================================================================
\\ Sub-composition: emit .nv.info (global per-module attrs)
\\
\\ Records: REGCOUNT, FRAME_SIZE=0, MIN_STACK_SIZE=0.
\\ REGCOUNT is reg_count+1, with a minimum floor of 8.
\\ ============================================================================

elf_emit_nvinfo reg_count :
    calign 4
    _cp → 64 cubin_pos_v
    ← 64 c_nvinfo_off_v _cp
    reg_val reg_count + 1
    if< reg_val 8
        reg_val 8
    nvi_sval_emit reg_val 3 EIATTR_REGCOUNT NVI_FMT_U32
    nvi_sval_emit 0 3 EIATTR_FRAME_SIZE NVI_FMT_U32
    nvi_sval_emit 0 3 EIATTR_MIN_STACK_SIZE NVI_FMT_U32
    _cp2 → 64 cubin_pos_v
    _no → 64 c_nvinfo_off_v
    ← 64 c_nvinfo_size_v _cp2 - _no

\\ ============================================================================
\\ Sub-composition: emit .nv.info.<kernel> (per-kernel attrs)
\\
\\ Records: CUDA_API_VERSION, per-param KPARAM_INFO, SPARSE_MMA_MASK,
\\          MAXREG_COUNT=255, EXIT_INSTR_OFFSETS (single offset at code_size-16),
\\          CBANK_PARAM_SIZE, PARAM_CBANK, SW_WAR=8, and — if smem_size>0 —
\\          the static shared memory size attr (0x33).
\\ ============================================================================

\\ Emit a single KPARAM_INFO (12-byte payload) record for parameter `ordinal`.
nvi_kparam_emit ordinal :
    cb_emit 4
    cb_emit EIATTR_KPARAM_INFO
    cw_emit 12
    cd_emit 0
    offset_ord (ordinal * 8) << 16
    offset_ord offset_ord | ordinal
    cd_emit offset_ord
    cd_emit 2171904

\\ Emit all KPARAM_INFO records (in reverse-ordinal order, matching nvcc).
nvi_kparams_all n_kparams :
    pi 0
    label kp_loop
    if>= pi n_kparams
        goto kp_done
    ordinal n_kparams - 1 - pi
    nvi_kparam_emit ordinal
    pi pi + 1
    goto kp_loop
    label kp_done

\\ Trailing fixed records: SPARSE_MMA_MASK, MAXREG_COUNT, EXIT_INSTR_OFFSETS.
nvi_emit_fixed_a code_size :
    cb_emit 3
    cb_emit EIATTR_SPARSE_MMA_MASK
    cb_emit 0
    cb_emit 0
    cb_emit 3
    cb_emit EIATTR_MAXREG_COUNT
    cb_emit 255
    cb_emit 0
    cb_emit 4
    cb_emit EIATTR_EXIT_INSTR_OFFSETS
    cw_emit 4
    cd_emit code_size - 16

\\ Trailing fixed records: CBANK_PARAM_SIZE, PARAM_CBANK, SW_WAR, optional smem.
nvi_emit_fixed_b n_kparams smem_size :
    ← 64 c_param_bytes_v n_kparams * 8
    cb_emit 3
    cb_emit EIATTR_CBANK_PARAM_SIZE
    _ep → 64 c_param_bytes_v
    cw_emit _ep
    cb_emit 4
    cb_emit EIATTR_PARAM_CBANK
    cw_emit 8
    cd_emit 2
    cbank_val (_ep << 16) | 528
    cd_emit cbank_val
    cb_emit 4
    cb_emit EIATTR_SW_WAR
    cw_emit 4
    cd_emit 8
    if> smem_size 0
        cb_emit 4
        cb_emit 51
        cw_emit 4
        cd_emit smem_size

elf_emit_nvinfo_k n_kparams smem_size code_size :
    calign 4
    _cp → 64 cubin_pos_v
    ← 64 c_nvinfok_off_v _cp
    nvi_u32_emit 128 EIATTR_CUDA_API_VERSION NVI_FMT_U32
    nvi_kparams_all n_kparams
    nvi_emit_fixed_a code_size
    nvi_emit_fixed_b n_kparams smem_size
    _cp2 → 64 cubin_pos_v
    _ko → 64 c_nvinfok_off_v
    ← 64 c_nvinfok_size_v _cp2 - _ko

\\ ============================================================================
\\ Sub-composition: emit .text.<kernel> and .nv.constant0.<kernel>
\\
\\ .text is 128-byte aligned; if code_size==0 we emit 48 zero bytes so the
\\ section is non-empty. After emitting .text we patch the kernel symbol's
\\ st_size field (symbol entry + 16) with the actual code length.
\\ .nv.constant0 is (528 + n_kparams*8) zero bytes.
\\ ============================================================================

\\ Copy code_size bytes from code_buf into the cubin buffer.
c_emit_code_bytes code_buf code_size :
    ci 0
    label tc_loop
    if>= ci code_size
        goto tc_done
    byte → 8 code_buf + ci
    cb_emit byte
    ci ci + 1
    goto tc_loop
    label tc_done

\\ Emit n zero bytes at the current cubin position.
c_emit_zeros n :
    zi 0
    label cz_loop
    if>= zi n
        goto cz_done
    cb_emit 0
    zi zi + 1
    goto cz_loop
    label cz_done

\\ Emit .text.<kernel> (128-byte aligned). For empty code emit 48 zero bytes
\\ so the section is non-empty. Patches the kernel symbol's st_size.
elf_emit_text code_buf code_size :
    calign 128
    _cp → 64 cubin_pos_v
    ← 64 c_text_off_v _cp
    if== code_size 0
        c_emit_zeros 48
    if> code_size 0
        c_emit_code_bytes code_buf code_size
    _cp2 → 64 cubin_pos_v
    _to → 64 c_text_off_v
    ← 64 c_text_size_v _cp2 - _to
    _ts → 64 c_text_size_v
    _ko → 64 c_sym_kernel_off_v
    elf_put_u64 _ts _ko + 16

\\ Emit .nv.constant0.<kernel> (4-byte aligned, 528 + n_kparams*8 zero bytes).
elf_emit_const0 :
    calign 4
    _cp → 64 cubin_pos_v
    ← 64 c_const0_off_v _cp
    _ep → 64 c_param_bytes_v
    const0_total 528 + _ep
    c_emit_zeros const0_total
    _cp2 → 64 cubin_pos_v
    _co → 64 c_const0_off_v
    ← 64 c_const0_size_v _cp2 - _co

elf_emit_text_const0 code_buf code_size :
    elf_emit_text code_buf code_size
    elf_emit_const0

\\ ============================================================================
\\ Sub-composition: emit section headers, patch ELF header e_shoff/e_shnum
\\
\\ 9 section headers in order: NULL, .shstrtab, .strtab, .symtab,
\\ .nv.info, .text, .nv.info.<kernel>, .nv.shared.reserved.0, .nv.constant0.
\\ Splits shdr into two emitters (shdr64_a/shdr64_b) to stay within the
\\ bootstrap's per-composition register window.
\\ ============================================================================

elf_emit_shdrs_head :
    _sn1 → 64 c_SN_shstrtab_v
    _sn2 → 64 c_SN_strtab_v
    _sn3 → 64 c_SN_symtab_v
    _o1 → 64 c_shstrtab_off_v
    _s1 → 64 c_shstrtab_size_v
    _o2 → 64 c_strtab_off_v
    _s2 → 64 c_strtab_size_v
    _o3 → 64 c_symtab_off_v
    _s3 → 64 c_symtab_size_v
    shdr64_a 0 0 0 0 0
    shdr64_b 0 0 0 0 0
    shdr64_a _sn1 SHT_STRTAB 0 0 _o1
    shdr64_b _s1 0 0 1 0
    shdr64_a _sn2 SHT_STRTAB 0 0 _o2
    shdr64_b _s2 0 0 1 0
    shdr64_a _sn3 SHT_SYMTAB 0 0 _o3
    shdr64_b _s3 2 3 8 24

elf_emit_shdrs_tail smem_size :
    _sn4 → 64 c_SN_nvinfo_v
    _sn5 → 64 c_SN_text_v
    _sn6 → 64 c_SN_nvinfo_k_v
    _sn7 → 64 c_SN_shared_v
    _sn8 → 64 c_SN_const0_v
    _o4 → 64 c_nvinfo_off_v
    _s4 → 64 c_nvinfo_size_v
    _o5 → 64 c_text_off_v
    _s5 → 64 c_text_size_v
    _o6 → 64 c_nvinfok_off_v
    _s6 → 64 c_nvinfok_size_v
    _o7 → 64 c_const0_off_v
    _s7 → 64 c_const0_size_v
    shdr64_a _sn4 SHT_LOPROC 0 0 _o4
    shdr64_b _s4 3 0 4 0
    shdr64_a _sn5 SHT_PROGBITS 6 0 _o5
    shdr64_b _s5 3 3 128 0
    shdr64_a _sn6 SHT_LOPROC SHF_INFO_LINK 0 _o6
    shdr64_b _s6 3 5 4 0
    shdr64_a _sn7 SHT_NOBITS 3 0 _o7
    shdr64_b smem_size 0 0 16 0
    shdr64_a _sn8 SHT_PROGBITS 66 0 _o7
    shdr64_b _s7 0 5 4 0

elf_emit_shdrs smem_size :
    calign 64
    _cp → 64 cubin_pos_v
    ← 64 c_shdrs_off_v _cp
    elf_emit_shdrs_head
    elf_emit_shdrs_tail smem_size
    elf_put_u64 0 32
    _so → 64 c_shdrs_off_v
    elf_put_u64 _so 40
    elf_put_u16 0 54
    elf_put_u16 0 56
    elf_put_u16 9 60
    elf_put_u16 1 62

\\ ============================================================================
\\ Orchestrator: build complete 9-section cubin into cubin_buf
\\ ============================================================================

elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size :
    elf_init
    elf_write_header
    elf_emit_shstrtab kernel_name kernel_nlen
    elf_emit_strtab_symtab kernel_name kernel_nlen
    elf_emit_nvinfo reg_count
    elf_emit_nvinfo_k n_kparams smem_size code_size
    elf_emit_text_const0 code_buf code_size
    elf_emit_shdrs smem_size

\\ ============================================================================
\\ elf_save — write cubin_buf[0 .. cubin_pos_v) to `path`
\\ ============================================================================

elf_save path :
    trap fd SYS_OPENAT -100 path 577 420

    written 0
    remaining → 64 cubin_pos_v
    label ew_loop
    if<= remaining 0
        goto ew_done
    write_ptr cubin_buf + written
    trap n SYS_WRITE fd write_ptr remaining
    if<= n 0
        goto ew_done
    written written + n
    remaining remaining - n
    goto ew_loop
    label ew_done
    trap ret SYS_CLOSE fd

\\ ============================================================================
\\ cubin_write — single entry point: build cubin and write to file
\\ ============================================================================

cubin_write kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size out_path :
    elf_build kernel_name kernel_nlen code_buf code_size n_kparams reg_count smem_size
    elf_save out_path
