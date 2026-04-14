"""
ELF64 binary writer for the Lithos compiler bootstrap.

Produces two types of output:
  1. ARM64 ELF executable — static, no libc, raw syscalls
  2. CUDA cubin ELF — for GPU kernels (SM90, cuModuleLoadData compatible)

Ported from compiler/lithos-elf.ls and compiler/compiler.ls (section 6).
"""

import struct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EHDR_SIZE = 64          # ELF64 header size
SHDR_SIZE = 64          # Section header entry size
SYM_SIZE  = 24          # ELF64 symbol table entry size
PHDR_SIZE = 56          # Program header entry size

# e_type
ET_EXEC = 2

# e_machine
EM_AARCH64 = 0xB7       # 183
EM_CUDA    = 0xBE        # 190

# e_version
EV_CURRENT = 1
EV_CUDA    = 0x80        # 128

# Section types
SHT_NULL     = 0
SHT_PROGBITS = 1
SHT_SYMTAB   = 2
SHT_STRTAB   = 3
SHT_NOBITS   = 8
SHT_LOPROC   = 0x70000000  # NVIDIA .nv.info sections

# Section flags
SHF_WRITE     = 1
SHF_ALLOC     = 2
SHF_EXECINSTR = 4
SHF_INFO_LINK = 0x40

# Program header types
PT_LOAD = 1

# Program header flags
PF_X = 1
PF_W = 2
PF_R = 4

# CUDA ELF flags (from probe cubin: 0x5a055a)
ELF_FLAGS_CUDA = 0x5A055A

# EIATTR constants
EIATTR_PARAM_CBANK            = 0x0A
EIATTR_FRAME_SIZE             = 0x11
EIATTR_MIN_STACK_SIZE         = 0x12
EIATTR_KPARAM_INFO            = 0x17
EIATTR_CBANK_PARAM_SIZE       = 0x19
EIATTR_MAXREG_COUNT           = 0x1B
EIATTR_EXIT_INSTR_OFFSETS     = 0x1C
EIATTR_REGCOUNT               = 0x2F
EIATTR_COOP_GROUP_INSTR_OFFSETS = 0x28
EIATTR_COOP_GROUP_MASK_REGIDS   = 0x29
EIATTR_CRS_STACK_SIZE         = 0x33
EIATTR_SW_WAR                 = 0x36
EIATTR_CUDA_API_VERSION       = 0x37
EIATTR_SPARSE_MMA_MASK        = 0x50

# NV info format codes
NVI_FMT_FLAG = 3
NVI_FMT_U32  = 4

# ARM64 load address
ARM64_BASE_ADDR = 0x400000


# ---------------------------------------------------------------------------
# Helper: align to boundary
# ---------------------------------------------------------------------------

def _align(pos: int, n: int) -> int:
    """Return pos rounded up to the next multiple of n."""
    mask = n - 1
    return (pos + mask) & ~mask


# ---------------------------------------------------------------------------
# ELF64 struct packers (all little-endian)
# ---------------------------------------------------------------------------

def _pack_ehdr(e_type, e_machine, e_version, e_entry, e_phoff, e_shoff,
               e_flags, e_phentsize, e_phnum, e_shnum, e_shstrndx,
               ei_osabi=0, ei_abiversion=0):
    """Pack a 64-byte ELF64 header."""
    e_ident = bytes([
        0x7F, 0x45, 0x4C, 0x46,  # magic
        2,                        # ELFCLASS64
        1,                        # ELFDATA2LSB
        1,                        # EV_CURRENT
        ei_osabi,
        ei_abiversion,
        0, 0, 0, 0, 0, 0, 0      # padding
    ])
    return e_ident + struct.pack('<HHIQQQIHHHHHH',
        e_type, e_machine, e_version,
        e_entry, e_phoff, e_shoff,
        e_flags,
        EHDR_SIZE,     # e_ehsize
        e_phentsize,
        e_phnum,
        SHDR_SIZE,     # e_shentsize
        e_shnum,
        e_shstrndx)


def _pack_shdr(sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
               sh_link, sh_info, sh_addralign, sh_entsize):
    """Pack a 64-byte ELF64 section header."""
    return struct.pack('<IIQQQQIIQQ',
        sh_name, sh_type, sh_flags, sh_addr,
        sh_offset, sh_size, sh_link, sh_info,
        sh_addralign, sh_entsize)


def _pack_sym(st_name, st_info, st_other, st_shndx, st_value, st_size):
    """Pack a 24-byte ELF64 symbol table entry."""
    return struct.pack('<IBBHQQ',
        st_name, st_info, st_other, st_shndx,
        st_value, st_size)


def _pack_phdr(p_type, p_flags, p_offset, p_vaddr, p_paddr,
               p_filesz, p_memsz, p_align):
    """Pack a 56-byte ELF64 program header."""
    return struct.pack('<IIQQQQQQ',
        p_type, p_flags, p_offset, p_vaddr, p_paddr,
        p_filesz, p_memsz, p_align)


# ---------------------------------------------------------------------------
# String table builder
# ---------------------------------------------------------------------------

class _StringTable:
    """Accumulate null-terminated strings, tracking their offsets."""

    def __init__(self):
        self._data = bytearray(b'\x00')  # starts with null byte at offset 0

    def add(self, s: str) -> int:
        """Add string, return its offset within the table."""
        off = len(self._data)
        self._data.extend(s.encode('ascii'))
        self._data.append(0)
        return off

    def data(self) -> bytes:
        return bytes(self._data)

    def __len__(self):
        return len(self._data)


# ===========================================================================
# ELFWriter — ARM64 static executable
# ===========================================================================

class ELFWriter:
    """
    Build an ARM64 ELF64 static executable.

    Layout:
      [ELF header 64B]
      [Program header 56B]  (PT_LOAD, R+X)
      [.text code]
      [.data]  (if any)

    Entry point = BASE + 0x78 (right after ehdr + phdr = 120 bytes).
    No section headers needed for a minimal static binary, but we include
    .text, .data, .shstrtab for completeness.
    """

    def __init__(self, base_addr: int = ARM64_BASE_ADDR):
        self._base = base_addr
        self._text_sections: list[tuple[str, bytes]] = []
        self._data_sections: list[tuple[str, bytes]] = []
        self._entry: int | None = None

    # -- public API ----------------------------------------------------------

    def add_text(self, name: str, code_bytes: bytes):
        """Append executable code."""
        self._text_sections.append((name, code_bytes))

    def add_data(self, name: str, data_bytes: bytes):
        """Append read/write data."""
        self._data_sections.append((name, data_bytes))

    def set_entry(self, addr: int):
        """Set the virtual entry point address."""
        self._entry = addr

    def write(self, filename: str):
        """Serialize the ELF to *filename*."""
        blob = self._build()
        with open(filename, 'wb') as f:
            f.write(blob)

    def build(self) -> bytes:
        """Return the ELF as bytes (for in-memory use)."""
        return self._build()

    # -- internals -----------------------------------------------------------

    def _build(self) -> bytes:
        # Concatenate all text
        all_text = b''.join(data for _, data in self._text_sections)
        all_data = b''.join(data for _, data in self._data_sections)

        has_data = len(all_data) > 0

        # Number of program headers: 1 (RX segment for text) + optionally 1 (RW for data)
        n_phdr = 2 if has_data else 1
        phdr_total = n_phdr * PHDR_SIZE

        # Code starts right after ehdr + phdrs
        code_offset = EHDR_SIZE + phdr_total
        code_vaddr = self._base + code_offset
        code_end = code_offset + len(all_text)

        # Data starts after text, page-aligned in memory
        if has_data:
            data_offset = _align(code_end, 0x1000)
            data_vaddr = self._base + data_offset
        else:
            data_offset = code_end
            data_vaddr = 0

        # Entry point
        entry = self._entry if self._entry is not None else code_vaddr

        # -- build section headers & shstrtab for a proper ELF --
        shstrtab = _StringTable()
        sn_shstrtab = shstrtab.add('.shstrtab')
        sn_text = shstrtab.add('.text')
        sn_data = shstrtab.add('.data') if has_data else 0
        shstrtab_bytes = shstrtab.data()
        # Align shstrtab size to 4
        if len(shstrtab_bytes) % 4:
            shstrtab_bytes += b'\x00' * (4 - len(shstrtab_bytes) % 4)

        # Total file content before section headers:
        #   ehdr + phdrs + text + (padding) + data + shstrtab
        # Place shstrtab after data
        if has_data:
            shstrtab_file_off = data_offset + len(all_data)
        else:
            shstrtab_file_off = code_end

        shdrs_file_off = _align(shstrtab_file_off + len(shstrtab_bytes), 8)

        # Section count: NULL + .shstrtab + .text + (optionally .data)
        n_sections = 3 + (1 if has_data else 0)
        shstrndx = 1  # .shstrtab is section 1

        # -- Total RX segment size (file + mem) --
        rx_filesz = code_end   # ehdr + phdrs + text
        rx_memsz = rx_filesz

        # -- Build output buffer --
        buf = bytearray()

        # ELF header
        buf += _pack_ehdr(
            e_type=ET_EXEC,
            e_machine=EM_AARCH64,
            e_version=EV_CURRENT,
            e_entry=entry,
            e_phoff=EHDR_SIZE,
            e_shoff=shdrs_file_off,
            e_flags=0,
            e_phentsize=PHDR_SIZE,
            e_phnum=n_phdr,
            e_shnum=n_sections,
            e_shstrndx=shstrndx,
            ei_osabi=0,
            ei_abiversion=0,
        )

        # Program header 0: PT_LOAD R+X (covers ehdr + phdr + .text)
        buf += _pack_phdr(
            p_type=PT_LOAD,
            p_flags=PF_R | PF_X,
            p_offset=0,
            p_vaddr=self._base,
            p_paddr=self._base,
            p_filesz=rx_filesz,
            p_memsz=rx_memsz,
            p_align=0x1000,
        )

        # Program header 1: PT_LOAD R+W (data segment)
        if has_data:
            buf += _pack_phdr(
                p_type=PT_LOAD,
                p_flags=PF_R | PF_W,
                p_offset=data_offset,
                p_vaddr=data_vaddr,
                p_paddr=data_vaddr,
                p_filesz=len(all_data),
                p_memsz=len(all_data),
                p_align=0x1000,
            )

        # .text
        buf += all_text

        # Pad to data offset if needed
        if has_data:
            buf += b'\x00' * (data_offset - len(buf))
            buf += all_data

        # Pad to shstrtab offset
        if len(buf) < shstrtab_file_off:
            buf += b'\x00' * (shstrtab_file_off - len(buf))
        buf += shstrtab_bytes

        # Pad to section header offset
        if len(buf) < shdrs_file_off:
            buf += b'\x00' * (shdrs_file_off - len(buf))

        # Section headers
        # [0] NULL
        buf += _pack_shdr(0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0)

        # [1] .shstrtab
        buf += _pack_shdr(sn_shstrtab, SHT_STRTAB, 0, 0,
                          shstrtab_file_off, len(shstrtab_bytes), 0, 0, 1, 0)

        # [2] .text
        buf += _pack_shdr(sn_text, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                          code_vaddr, code_offset, len(all_text), 0, 0, 4, 0)

        # [3] .data (optional)
        if has_data:
            buf += _pack_shdr(sn_data, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                              data_vaddr, data_offset, len(all_data), 0, 0, 8, 0)

        return bytes(buf)


# ===========================================================================
# CubinWriter — CUDA cubin ELF (SM90)
# ===========================================================================

class CubinWriter(ELFWriter):
    """
    Build a CUDA cubin ELF64 for GPU kernels.

    ELF layout (9 sections, matching lithos-elf.ls):
      [0] NULL
      [1] .shstrtab
      [2] .strtab
      [3] .symtab   (4 entries: UNDEF, SECTION .text, SECTION .const0, kernel FUNC)
      [4] .nv.info  (global: REGCOUNT, FRAME_SIZE, MIN_STACK_SIZE)
      [5] .text.<kernel>      (SM90 SASS, 128-byte aligned)
      [6] .nv.info.<kernel>   (per-kernel attrs)
      [7] .nv.shared.reserved.0  (NOBITS)
      [8] .nv.constant0.<kernel> (cbuf0: 0x210 driver-reserved + params)
    """

    def __init__(self):
        # Don't call super().__init__(); cubin is entirely different layout
        self._kernel_name: str | None = None
        self._code: bytes = b''
        self._n_params: int = 0
        self._reg_count: int = 8
        self._smem_size: int = 0
        self._cooperative: bool = False
        self._gridsync_offsets: list[int] = []
        self._exit_offsets: list[int] = [0x100]

    # -- public API ----------------------------------------------------------

    def add_text(self, name: str, code_bytes: bytes):
        """Set the kernel name and its SASS machine code."""
        self._kernel_name = name
        self._code = code_bytes

    def add_data(self, name: str, data_bytes: bytes):
        """Not used for cubin; ignored."""
        pass

    def set_entry(self, addr: int):
        """Not used for cubin; ignored."""
        pass

    def add_nv_info(self, regcount: int = 8, n_params: int = 0,
                    smem_size: int = 0, cooperative: bool = False,
                    gridsync_offsets: list[int] | None = None,
                    exit_offsets: list[int] | None = None):
        """Configure CUDA kernel attributes."""
        self._reg_count = max(regcount, 8)
        self._n_params = n_params
        self._smem_size = smem_size
        self._cooperative = cooperative
        self._gridsync_offsets = gridsync_offsets or []
        if exit_offsets is not None:
            self._exit_offsets = exit_offsets

    def write(self, filename: str):
        blob = self._build_cubin()
        with open(filename, 'wb') as f:
            f.write(blob)

    def build(self) -> bytes:
        return self._build_cubin()

    # -- internals -----------------------------------------------------------

    def _build_cubin(self) -> bytes:
        kname = self._kernel_name or 'kernel'
        code = self._code
        n_params = self._n_params
        reg_count = self._reg_count
        smem_size = self._smem_size
        param_bytes = n_params * 8

        buf = bytearray()

        # ---- ELF header placeholder (64 bytes, patched at end) ----
        buf += b'\x00' * EHDR_SIZE

        # ---- .shstrtab (section 1) ----
        shstrtab_off = len(buf)
        shstrtab = bytearray(b'\x00')  # null byte at offset 0

        sn_shstrtab = len(shstrtab)
        shstrtab += b'.shstrtab\x00'

        sn_strtab = len(shstrtab)
        shstrtab += b'.strtab\x00'

        sn_symtab = len(shstrtab)
        shstrtab += b'.symtab\x00'

        sn_nvinfo = len(shstrtab)
        shstrtab += b'.nv.info\x00'

        sn_nvinfo_k = len(shstrtab)
        shstrtab += f'.nv.info.{kname}\x00'.encode('ascii')

        sn_text = len(shstrtab)
        shstrtab += f'.text.{kname}\x00'.encode('ascii')

        sn_const0 = len(shstrtab)
        shstrtab += f'.nv.constant0.{kname}\x00'.encode('ascii')

        sn_shared = len(shstrtab)
        shstrtab += b'.nv.shared.reserved.0\x00'

        # Align to 4 bytes
        while len(shstrtab) % 4:
            shstrtab += b'\x00'
        shstrtab_size = len(shstrtab)
        buf += shstrtab

        # ---- .strtab (section 2) ----
        strtab_off = len(buf)
        strtab = bytearray(b'\x00\x00')  # sym0 + sym1 empty names
        strsym_kernel = len(strtab)
        strtab += kname.encode('ascii') + b'\x00'
        strtab_size = len(strtab)
        buf += strtab

        # ---- .symtab (section 3; 4 entries x 24 = 96 bytes) ----
        # Align to 8
        pad = _align(len(buf), 8) - len(buf)
        buf += b'\x00' * pad
        symtab_off = len(buf)

        # sym0: UNDEF
        buf += _pack_sym(0, 0, 0, 0, 0, 0)
        # sym1: SECTION LOCAL, shndx=5 (.text.<kernel>)
        buf += _pack_sym(0, 3, 0, 5, 0, 0)   # st_info=3 => STB_LOCAL|STT_SECTION
        # sym2: SECTION LOCAL, shndx=8 (.nv.constant0.<kernel>)
        buf += _pack_sym(0, 3, 0, 8, 0, 0)
        # sym3: FUNC GLOBAL STO_CUDA_ENTRY shndx=5; st_size patched after .text
        sym_kernel_buf_off = len(buf)
        buf += _pack_sym(strsym_kernel, 0x12, 0x10, 5, 0, 0)
        # st_info=0x12 (STB_GLOBAL|STT_FUNC), st_other=0x10 (STO_CUDA_ENTRY)

        symtab_size = len(buf) - symtab_off

        # ---- .nv.info (section 4; global attributes) ----
        pad = _align(len(buf), 4) - len(buf)
        buf += b'\x00' * pad
        nvinfo_off = len(buf)

        # REGCOUNT (sym_idx=3)
        buf += self._nvi_sval(reg_count, 3, EIATTR_REGCOUNT, NVI_FMT_U32)
        # FRAME_SIZE = 0
        buf += self._nvi_sval(0, 3, EIATTR_FRAME_SIZE, NVI_FMT_U32)
        # MIN_STACK_SIZE = 0
        buf += self._nvi_sval(0, 3, EIATTR_MIN_STACK_SIZE, NVI_FMT_U32)

        nvinfo_size = len(buf) - nvinfo_off

        # ---- .nv.info.<kernel> (section 6; per-kernel attributes) ----
        pad = _align(len(buf), 4) - len(buf)
        buf += b'\x00' * pad
        nvinfo_k_off = len(buf)

        # CUDA_API_VERSION = 0x80 (128)
        buf += self._nvi_u32(128, EIATTR_CUDA_API_VERSION, NVI_FMT_U32)

        # KPARAM_INFO: one 12-byte record per param, reverse ordinal order
        for pi in range(n_params):
            ordinal = n_params - 1 - pi
            rec = bytearray()
            rec.append(NVI_FMT_U32)          # fmt
            rec.append(EIATTR_KPARAM_INFO)   # attr
            rec += struct.pack('<H', 12)     # size = 12
            rec += struct.pack('<I', 0)      # index
            offset_ord = ((ordinal * 8) << 16) | ordinal
            rec += struct.pack('<I', offset_ord)
            rec += struct.pack('<I', 0x0021F000)  # flags=0xf000, size=8 ptr
            buf += rec

        # SPARSE_MMA_MASK (FLAG: fmt=03 attr=0x50 val=0)
        buf += bytes([NVI_FMT_FLAG, EIATTR_SPARSE_MMA_MASK, 0, 0])

        # MAXREG_COUNT = 0xff (HVAL: fmt=03 attr=0x1b val_u16=0x00ff)
        buf += bytes([NVI_FMT_FLAG, EIATTR_MAXREG_COUNT, 0xFF, 0x00])

        # Cooperative grid-sync attributes
        if self._cooperative:
            # COOP_GROUP_MASK_REGIDS: fmt=04 attr=0x29 size=16; four 0xffffffff
            buf += struct.pack('<BBH', NVI_FMT_U32, EIATTR_COOP_GROUP_MASK_REGIDS, 16)
            buf += struct.pack('<IIII', 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

            # COOP_GROUP_INSTR_OFFSETS
            gs_count = len(self._gridsync_offsets)
            buf += struct.pack('<BBH', NVI_FMT_U32, EIATTR_COOP_GROUP_INSTR_OFFSETS,
                               gs_count * 4)
            for gs_off in self._gridsync_offsets:
                buf += struct.pack('<I', gs_off)

        # EXIT_INSTR_OFFSETS: fmt=04 attr=0x1c size=N*4
        exit_count = len(self._exit_offsets)
        buf += struct.pack('<BBH', NVI_FMT_U32, EIATTR_EXIT_INSTR_OFFSETS,
                           exit_count * 4)
        for ex_off in self._exit_offsets:
            buf += struct.pack('<I', ex_off)

        # CBANK_PARAM_SIZE (HVAL; total param bytes)
        buf += bytes([NVI_FMT_FLAG, EIATTR_CBANK_PARAM_SIZE,
                      param_bytes & 0xFF, (param_bytes >> 8) & 0xFF])

        # PARAM_CBANK (SVAL: sym_idx=2, cbank_offset=0x210 | param_bytes<<16)
        buf += struct.pack('<BBH', NVI_FMT_U32, EIATTR_PARAM_CBANK, 8)
        buf += struct.pack('<I', 2)   # sym_idx = constant0 section symbol
        cbank_val = (param_bytes << 16) | 0x210
        buf += struct.pack('<I', cbank_val)

        # SW_WAR workaround flag
        buf += self._nvi_u32(8, EIATTR_SW_WAR, NVI_FMT_U32)

        # CRS_STACK_SIZE / MAX_SHARED_MEM_PER_BLOCK_OPTIN (attr=0x33)
        if smem_size > 0:
            buf += self._nvi_u32(smem_size, EIATTR_CRS_STACK_SIZE, NVI_FMT_U32)

        nvinfo_k_size = len(buf) - nvinfo_k_off

        # ---- .text.<kernel> (section 5; 128-byte aligned SASS code) ----
        pad = _align(len(buf), 128) - len(buf)
        buf += b'\x00' * pad
        text_off = len(buf)

        if len(code) == 0:
            # Safety fallback: 48 bytes of zeros (NOP+NOP+EXIT placeholder)
            buf += b'\x00' * 48
        else:
            buf += code
        text_size = len(buf) - text_off

        # Patch kernel symbol st_size (u64 at sym_kernel_buf_off + 16)
        struct.pack_into('<Q', buf, sym_kernel_buf_off + 16, text_size)

        # ---- .nv.constant0.<kernel> ----
        pad = _align(len(buf), 4) - len(buf)
        buf += b'\x00' * pad
        const0_off = len(buf)
        const0_total = 0x210 + param_bytes  # 528 bytes driver-reserved + params
        buf += b'\x00' * const0_total
        const0_size = len(buf) - const0_off

        # ---- Section headers (9 x 64 = 576 bytes) ----
        pad = _align(len(buf), 64) - len(buf)
        buf += b'\x00' * pad
        shdrs_off = len(buf)

        # [0] NULL
        buf += _pack_shdr(0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0)

        # [1] .shstrtab (SHT_STRTAB)
        buf += _pack_shdr(sn_shstrtab, SHT_STRTAB, 0, 0,
                          shstrtab_off, shstrtab_size, 0, 0, 1, 0)

        # [2] .strtab (SHT_STRTAB)
        buf += _pack_shdr(sn_strtab, SHT_STRTAB, 0, 0,
                          strtab_off, strtab_size, 0, 0, 1, 0)

        # [3] .symtab (SHT_SYMTAB, link=2=.strtab, info=3=first_global)
        buf += _pack_shdr(sn_symtab, SHT_SYMTAB, 0, 0,
                          symtab_off, symtab_size, 2, 3, 8, SYM_SIZE)

        # [4] .nv.info (SHT_LOPROC, link=3=.symtab)
        buf += _pack_shdr(sn_nvinfo, SHT_LOPROC, 0, 0,
                          nvinfo_off, nvinfo_size, 3, 0, 4, 0)

        # [5] .text.<kernel> (SHT_PROGBITS, flags=ALLOC|EXECINSTR, link=3, info=3)
        buf += _pack_shdr(sn_text, SHT_PROGBITS,
                          SHF_ALLOC | SHF_EXECINSTR, 0,
                          text_off, text_size, 3, 3, 128, 0)

        # [6] .nv.info.<kernel> (SHT_LOPROC, flags=INFO_LINK, link=3, info=5=.text)
        buf += _pack_shdr(sn_nvinfo_k, SHT_LOPROC, SHF_INFO_LINK, 0,
                          nvinfo_k_off, nvinfo_k_size, 3, 5, 4, 0)

        # [7] .nv.shared.reserved.0 (SHT_NOBITS, flags=ALLOC|WRITE(3))
        # sh_offset = const0_off (matches Lithos source), sh_size = smem_size
        buf += _pack_shdr(sn_shared, SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0,
                          const0_off, smem_size, 0, 0, 16, 0)

        # [8] .nv.constant0.<kernel> (SHT_PROGBITS, flags=0x42=ALLOC|INFO_LINK, info=5)
        buf += _pack_shdr(sn_const0, SHT_PROGBITS,
                          SHF_ALLOC | SHF_INFO_LINK, 0,
                          const0_off, const0_size, 0, 5, 4, 0)

        # ---- Patch ELF header ----
        ehdr = _pack_ehdr(
            e_type=ET_EXEC,
            e_machine=EM_CUDA,
            e_version=EV_CUDA,
            e_entry=0,
            e_phoff=0,
            e_shoff=shdrs_off,
            e_flags=ELF_FLAGS_CUDA,
            e_phentsize=0,
            e_phnum=0,
            e_shnum=9,
            e_shstrndx=1,
            ei_osabi=0x33,       # NVIDIA CUDA OS/ABI
            ei_abiversion=7,
        )
        buf[0:EHDR_SIZE] = ehdr

        return bytes(buf)

    # -- NV info record helpers ----------------------------------------------

    @staticmethod
    def _nvi_u32(val: int, attr: int, fmt: int) -> bytes:
        """Plain u32 record: [fmt][attr][size=4][val]."""
        return struct.pack('<BBH I', fmt, attr, 4, val)

    @staticmethod
    def _nvi_sval(val: int, sym_idx: int, attr: int, fmt: int) -> bytes:
        """Sym+val record: [fmt][attr][size=8][sym_idx][val]."""
        return struct.pack('<BBH II', fmt, attr, 8, sym_idx, val)


# ===========================================================================
# Convenience: one-call cubin builder (matches elf_write_simple)
# ===========================================================================

def write_cubin(kernel_name: str, code: bytes, n_params: int,
                reg_count: int = 8, smem_size: int = 0,
                filename: str | None = None,
                cooperative: bool = False,
                gridsync_offsets: list[int] | None = None) -> bytes:
    """
    Build a CUDA cubin ELF and optionally write it to a file.

    Returns the cubin bytes.
    """
    w = CubinWriter()
    w.add_text(kernel_name, code)
    w.add_nv_info(
        regcount=reg_count,
        n_params=n_params,
        smem_size=smem_size,
        cooperative=cooperative,
        gridsync_offsets=gridsync_offsets,
    )
    data = w.build()
    if filename:
        with open(filename, 'wb') as f:
            f.write(data)
    return data


def write_arm64(code: bytes, filename: str | None = None,
                data: bytes = b'',
                base_addr: int = ARM64_BASE_ADDR) -> bytes:
    """
    Build an ARM64 static ELF executable and optionally write it to a file.

    Returns the ELF bytes.
    """
    w = ELFWriter(base_addr=base_addr)
    w.add_text('.text', code)
    if data:
        w.add_data('.data', data)
    # Entry point defaults to start of .text
    elf = w.build()
    if filename:
        with open(filename, 'wb') as f:
            f.write(elf)
    return elf


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == '__main__':
    import sys
    import os

    print('ELF writer self-test...')

    # --- Test 1: ARM64 ELF ---
    # Minimal ARM64: mov x0, #0; mov x8, #93; svc #0  (exit(0))
    arm64_exit0 = bytes([
        0x00, 0x00, 0x80, 0xD2,  # mov x0, #0
        0xA8, 0x0B, 0x80, 0xD2,  # mov x8, #93
        0x01, 0x00, 0x00, 0xD4,  # svc #0
    ])
    arm64_elf = write_arm64(arm64_exit0)

    # Verify ELF magic
    assert arm64_elf[0:4] == b'\x7fELF', 'ARM64: bad magic'
    assert arm64_elf[4] == 2, 'ARM64: not ELFCLASS64'
    assert arm64_elf[5] == 1, 'ARM64: not little-endian'
    # e_machine at offset 18 (u16 LE)
    e_machine = struct.unpack_from('<H', arm64_elf, 18)[0]
    assert e_machine == EM_AARCH64, f'ARM64: e_machine={e_machine:#x}'
    # e_type at offset 16
    e_type = struct.unpack_from('<H', arm64_elf, 16)[0]
    assert e_type == ET_EXEC, f'ARM64: e_type={e_type}'
    print(f'  ARM64 ELF: {len(arm64_elf)} bytes, OK')

    # --- Test 2: CUDA cubin ---
    fake_sass = b'\x00' * 256  # placeholder SASS
    cubin = write_cubin('matmul', fake_sass, n_params=3, reg_count=32)

    assert cubin[0:4] == b'\x7fELF', 'CUBIN: bad magic'
    assert cubin[7] == 0x33, 'CUBIN: bad OS/ABI (not CUDA)'
    e_machine = struct.unpack_from('<H', cubin, 18)[0]
    assert e_machine == EM_CUDA, f'CUBIN: e_machine={e_machine:#x}'
    e_flags = struct.unpack_from('<I', cubin, 48)[0]
    assert e_flags == ELF_FLAGS_CUDA, f'CUBIN: e_flags={e_flags:#x}'
    # Verify section count = 9
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    assert e_shnum == 9, f'CUBIN: e_shnum={e_shnum}'
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    assert e_shstrndx == 1, f'CUBIN: e_shstrndx={e_shstrndx}'
    print(f'  CUDA cubin: {len(cubin)} bytes, 9 sections, OK')

    # --- Test 3: Write to disk and verify ---
    test_dir = '/tmp/lithos_elf_test'
    os.makedirs(test_dir, exist_ok=True)
    arm64_path = os.path.join(test_dir, 'test_arm64')
    cubin_path = os.path.join(test_dir, 'test.cubin')
    write_arm64(arm64_exit0, filename=arm64_path)
    write_cubin('matmul', fake_sass, n_params=3, reg_count=32, filename=cubin_path)
    assert os.path.getsize(arm64_path) == len(arm64_elf)
    assert os.path.getsize(cubin_path) == len(cubin)
    print(f'  File write: OK ({arm64_path}, {cubin_path})')

    print('All tests passed.')
