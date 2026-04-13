\ elf-wrap.fs — Wrap ARM64 machine code into a minimal Linux executable.
\
\ Produces an ET_EXEC ELF64 ARM64 binary with a single PT_LOAD segment
\ covering the .text payload. Loaded at 0x400000; entry is the start of
\ the payload. This mirrors the layout used by sixth/eighth for its
\ standalone binaries.

variable elf-fd
variable elf-text-size

: ef,  ( byte -- )
  elf-fd @ >r
  pad over c!  pad 1 r> write-file drop drop ;

: ef-w32,  ( u32 -- )
  dup 255 and ef, 8 rshift
  dup 255 and ef, 8 rshift
  dup 255 and ef, 8 rshift
  255 and ef, ;

: ef-w64,  ( u64 -- )  dup ef-w32,  32 rshift ef-w32, ;

$400000 constant BASE-ADDR
64      constant EHDR-SIZE
56      constant PHDR-SIZE

: elf-header  ( text-len -- )
  elf-text-size !
  \ e_ident
  $7f ef,  [char] E ef,  [char] L ef,  [char] F ef,
  2 ef, 1 ef, 1 ef, 0 ef,
  0 ef, 0 ef, 0 ef, 0 ef, 0 ef, 0 ef, 0 ef, 0 ef,
  \ e_type = ET_EXEC(2)
  2 ef, 0 ef,
  \ e_machine = EM_AARCH64(183)
  183 ef, 0 ef,
  \ e_version
  1 ef-w32,
  \ e_entry = BASE + EHDR + PHDR
  BASE-ADDR EHDR-SIZE + PHDR-SIZE + ef-w64,
  \ e_phoff = EHDR-SIZE
  EHDR-SIZE ef-w64,
  \ e_shoff = 0 (no section headers)
  0 ef-w64,
  \ e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx
  0 ef-w32,
  EHDR-SIZE ef, 0 ef,  PHDR-SIZE ef, 0 ef,  1 ef, 0 ef,
  0 ef, 0 ef,  0 ef, 0 ef,  0 ef, 0 ef, ;

: elf-phdr  ( -- )
  1 ef-w32,                     \ PT_LOAD
  5 ef-w32,                     \ R|X
  0 ef-w64,                     \ p_offset
  BASE-ADDR ef-w64,             \ p_vaddr
  BASE-ADDR ef-w64,             \ p_paddr
  elf-text-size @ EHDR-SIZE PHDR-SIZE + + ef-w64,   \ p_filesz
  elf-text-size @ EHDR-SIZE PHDR-SIZE + + ef-w64,   \ p_memsz
  $10000 ef-w64, ;              \ align

: write-elf  ( code-addr code-u out-addr out-u -- )
  577 open-file drop elf-fd !
  over elf-header
  elf-phdr
  \ Now write the code payload
  elf-fd @ write-file drop
  elf-fd @ close-file drop ;
