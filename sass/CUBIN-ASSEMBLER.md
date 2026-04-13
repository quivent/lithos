# Cubin Assembler — Current State and GEMV Skeleton

## Tool

`/home/ubuntu/lithos/gpu/emit.fs` — GPU machine code emitter source.
(emit-sass-auto.fs deleted — opcode constants are a subset of gpu/emit.fs)

No ptxas, no PTX. SASS opcodes are emitted as raw 16-byte instruction+control word pairs directly into a buffer, then wrapped in a minimal ELF.

## What Exists

- **Instruction emitter core**: `sass,` / `sinst,` emit 16-byte pairs into `sass-buf`.
- **47 sm_90 opcodes** mapped: all arithmetic (FADD, FFMA, FMUL, FMNMX, MUFU, IMAD, IADD3, ISETP, HMMA, HFMA2), memory (LDG, STG, LDS, STS, LDSM, LDGSTS), control (BRA, EXIT, NOP, S2R, S2UR, DEPBAR, BAR, MEMBAR), and shuffle (SHFL, ULDC, LDC).
- **Full predicate guard encoding** (`@p0`..`@!p3` helpers).
- **ELF header writer**: `cubin-elf-header` emits the 64-byte ELF64 header with correct e_flags (`0x5a055a`), EM_CUDA (`0xBE`), NVIDIA ABI byte (`0x33`).
- **Vector-add cubin** (`sass/probe.cubin`) was emitted by ptxas from probe.ptx and verified to execute on GH200. Its structure is the reference skeleton.

## What `build-cubin` Does Today

Resets the buffer, calls `cubin-elf-header`, and returns. The remaining 9 steps are stubs with `\ TODO` comments. No section headers, no `.nv.info`, no `.text.<kernel>` stitching. The emitter can produce correct instruction bytes but cannot yet produce a loadable ELF.

## Minimum Cubin Skeleton for GEMV

From `probe.cubin` (nvdisasm + readelf verified):

| Field | Value | Source |
|---|---|---|
| Register count (EIATTR_REGCOUNT 0x2f04) | 12 for probe; GEMV needs ~16 (addr regs + accum) | `.nv.info` |
| Frame size (EIATTR_FRAME_SIZE 0x1104) | 0 (no stack) | `.nv.info` |
| Min stack size (EIATTR_MIN_STACK_SIZE 0x1204) | 0 | `.nv.info` |
| Shared memory | `.nv.shared.reserved.*` NOBITS section, size = 0 for pure-global GEMV | `.nv.info` |
| Param bank base | `c[0x0][0x210]` — constant bank 0 at offset 0x210 | EIATTR_PARAM_CBANK |
| Param bank size (EIATTR_CBANK_PARAM_SIZE 0x1903) | 0x1c for 4 params (each 8-byte pointer = 3 pointers + 1 int = 28 bytes) | `.nv.info.probe` |
| Thread block size | declared at launch, not in cubin; no field needed | — |
| CUDA API version (EIATTR_CUDA_API_VERSION 0x3704) | 0x80 | `.nv.info.<kernel>` |
| `.nv.callgraph` | 4 entries × 8 bytes; terminal entries use 0xffffffff/0xfffffffe/... | `.nv.callgraph` |
| `.text.<kernel>` align | 128 bytes | section header sh_addralign |

GEMV-specific register budget: R0=tid, R2-R3=input ptr, R4-R5=weight ptr, R6-R7=output ptr, R8=accumulator, R9-R10=scratch, UR4-UR5=uniform ctaid — ~12-16 registers total, matching probe.

## What Is Missing for a Full GEMV Cubin

1. **`build-cubin` body** — section layout: shstrtab, strtab, symtab, `.nv.info`, `.nv.info.<kernel>`, `.nv.constant0.<kernel>`, `.text.<kernel>`, `.nv.shared.*`, `.nv.callgraph`.
2. **Section header array** — 12 entries at known offset; `e_shoff`, `e_shnum`, `e_shstrndx` patched into ELF header after layout.
3. **Symbol table entry** for the kernel with `STO_CUDA_ENTRY` flag and `st_size = .text size`.
4. **`.nv.info` key-value serialization** — format: `u8 fmt | u8 attr`, `u16 size`, data. Attrs needed: REGCOUNT, FRAME_SIZE, MIN_STACK_SIZE, CUDA_API_VERSION, KPARAM_INFO (one entry per param), CBANK_PARAM_SIZE, PARAM_CBANK, EXIT_INSTR_OFFSETS.
5. **Correct scheduling (control words)** — the auto-generated emitters use a stub `$000fc00000000000`. Stall counts, write/read barrier indices, and reuse flags must be set correctly for LDG/STG latency hiding. Probe cubin has the reference patterns.
6. **`load-cubin` FFI call** — `cuModuleLoadData` binding is a stub returning 0.

---

`dance: [R0→tid] [UR4→ctaid] [IMAD→offset] [LDG→a[i]] [LDG→w[i]] [FFMA→acc] [STG→out]`
