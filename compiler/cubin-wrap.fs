\ cubin-wrap.fs — Wrap raw SASS into a minimal ELF64 cubin for sm_90.
\
\ A full cubin has .text.<kernel>, .nv.info, .nv.info.<kernel>, .symtab,
\ .strtab, .shstrtab, and NVIDIA-specific segments. For now we emit just
\ enough structure that nvdisasm can see the kernel bytes; a loader-ready
\ cubin requires the NVIDIA proprietary .nv.info records, which we treat
\ as a TODO.
\
\ Layout produced:
\   ELF64 header (64 bytes)
\   Program header 0: PT_LOAD for .text (56 bytes)
\   Section headers: NULL, .text, .shstrtab
\   Raw SASS bytes
\   .shstrtab

variable cubin-fd

: cb,  ( byte -- )   cubin-fd @ >r
  pad over c! pad 1 r> write-file drop drop ;

: cb-w32,  ( u32 -- )
  dup 255 and cb, 8 rshift
  dup 255 and cb, 8 rshift
  dup 255 and cb, 8 rshift
  255 and cb, ;

: cb-w64,  ( u64 -- )
  dup cb-w32,  32 rshift cb-w32, ;

\ Emit a minimal ELF64 cubin stub. Not a valid loader target — see TODO
\ above — but a real file with the SASS bytes visible.
: write-cubin  ( addr u -- )
  577 open-file drop cubin-fd !
  \ ELF64 magic + class64 + little-endian + version1 + OSABI=CUDA(51)
  $7f cb, [char] E cb, [char] L cb, [char] F cb,
  2 cb, 1 cb, 1 cb, 51 cb,
  \ ABI version + padding
  7 cb, 0 cb, 0 cb, 0 cb, 0 cb, 0 cb, 0 cb, 0 cb,
  \ e_type = ET_EXEC(2), e_machine = EM_CUDA(190)
  2 cb, 0 cb,  190 cb, 0 cb,
  \ e_version = 1
  1 cb-w32,
  \ e_entry, e_phoff, e_shoff — we fill with plausible placeholders
  0 cb-w64,
  64 cb-w64,        \ phoff just past header
  0 cb-w64,         \ no section headers yet (TODO)
  \ e_flags — sm_90 in low 8 bits
  $00520000 cb-w32,
  \ e_ehsize, e_phentsize, e_phnum
  64 cb, 0 cb,  56 cb, 0 cb,  1 cb, 0 cb,
  \ e_shentsize, e_shnum, e_shstrndx
  64 cb, 0 cb,  0 cb, 0 cb,  0 cb, 0 cb,
  \ Program header 0: PT_LOAD
  1 cb-w32,  5 cb-w32,        \ type, flags = R|X
  $1000 cb-w64,              \ offset (data starts at 0x1000)
  0 cb-w64,                  \ vaddr
  0 cb-w64,                  \ paddr
  over cb-w64,               \ filesz = payload length
  dup cb-w64,                \ memsz = payload length
  $1000 cb-w64,              \ align
  \ Pad to 0x1000
  $1000 120 - 0 ?do 0 cb, loop
  \ Payload
  cubin-fd @ write-file drop
  cubin-fd @ close-file drop ;
