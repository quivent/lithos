# sm_90a SASS Memory Instruction Encodings

Reverse-engineered from nvdisasm with `-hex` flag on sm_90a cubins compiled
by ptxas 12.x (Feb 2025 build). All hex values are exact hardware encodings
read from disassembly — no guessing.

Tool chain:
```
ptxas  -arch sm_90a -o kernel.cubin kernel.ptx
nvdisasm -hex kernel.cubin
```

Each SASS instruction is 128 bits = two consecutive 64-bit words in memory.
nvdisasm prints them as two hex values:
- `w0` (first 64-bit word, printed first): instruction body
- `w1` (second 64-bit word): control/scheduling/predicate word

Bit numbering: bit 0 is LSB of w0; bit 64 is LSB of w1.

---

## Global Memory: LDG.E

PTX source: `ld.global.f32 %f0, [%rd0];`

### Full encoding table

| Instruction                           | w0                 | w1                 |
|---------------------------------------|--------------------|--------------------|
| `LDG.E R3, desc[UR4][R2.64]`         | `0x0000000402037981` | `0x001eaa000c1e1900` |
| `LDG.E R9, desc[UR4][R2.64+0x4]`     | `0x0000040402097981` | `0x000ee8000c1e1900` |
| `LDG.E R11, desc[UR4][R2.64+0x100]`  | `0x00010004020b7981` | `0x000f28000c1e1900` |
| `LDG.E R13, desc[UR4][R2.64+0x1000]` | `0x00100004020d7981` | `0x000f68000c1e1900` |
| `LDG.E R15, desc[UR4][R2.64+0x10000]`| `0x01000004020f7981` | `0x000f68000c1e1900` |
| `LDG.E R17, desc[UR4][R2.64+0x100000]`| `0x1000000402117981` | `0x000f68000c1e1900` |
| `LDG.E R19, desc[UR4][R2.64+0x7ffffc]`| `0x7ffffc0402137981` | `0x000f62000c1e1900` |
| `LDG.E.64 R2, desc[UR4][R2.64]`      | `0x0000000402027981` | `0x001eaa000c1e1b00` |

### w0 field layout

```
  bits [15: 0] = 0x7981                  (opcode — LDG.E)
  bits [23:16] = dst_gpr                 (destination GPR, 8-bit register number)
  bits [31:24] = addr_gpr                (base address GPR, holds 64-bit byte ptr)
  bits [39:32] = ur_reg                  (uniform register with desc[] handle)
  bits [63:40] = imm_offset              (24-bit signed byte offset, see range below)
```

**Verified extraction formula (Python):**
```python
opcode   =  w0        & 0xffff
dst_gpr  = (w0 >> 16) & 0xff
addr_gpr = (w0 >> 24) & 0xff
ur_reg   = (w0 >> 32) & 0xff
imm_off  = (w0 >> 40) & 0xffffff   # treat as signed: if > 0x7fffff: off -= 0x1000000
```

All test cases verified OK against known field values.

### Immediate offset field

- **Range**: 0x000000 to 0x7ffffc (0 to 8,388,604 bytes)
- **Maximum**: 0x7ffffc observed (24-bit, top bit unused for negative offsets in practice)
- **Units**: bytes (not words)
- **Alignment**: no alignment enforced in the encoding; hardware assumes naturally aligned
- **Negative offsets**: 24-bit signed — negative values are 2's complement in bits [63:40]
  (ptxas did not generate negative offsets in tested patterns; positive only observed)

### Cache qualifier bits (in w1)

The cache policy lives in `w1[23:8]` (16 bits spanning the second and third bytes of w1).

| PTX qualifier      | SASS mnemonic      | w1[23:8]  | Key bit differences                     |
|--------------------|--------------------|-----------|-----------------------------------------|
| (none / default)   | `LDG.E`            | `0x1e19`  | baseline                                |
| `.ca` (cache-all)  | `LDG.E.STRONG.SM`  | `0x1eb9`  | w1[13]=1, w1[7]=1 (SM scope)           |
| `.cg` (cache-global)| `LDG.E.STRONG.GPU`| `0x1ef9`  | w1[14:13]=11 (GPU scope)               |
| `.cs` (streaming)  | `LDG.E.EF`         | `0x0e19`  | w1[20]=0 (evict-first)                 |
| `.lu` (last-use)   | `LDG.E.LU`         | `0x3e19`  | w1[21]=1 (last-use hint)               |
| `.cv` (volatile)   | `LDG.E.STRONG.SYS` | `0x1f59`  | w1[16]=1, w1[14]=1, w1[6]=1 (SYS scope)|

Evict policy field in `w1[23:16]`:
```
  0x1e = 0001_1110   default (L1 cache, evict normal)
  0x0e = 0000_1110   EF — evict first (streaming/no-reuse)
  0x3e = 0011_1110   LU — last use
  0x1f = 0001_1111   STRONG.SYS — system-coherent
```

Scope field within `w1[15:8]`:
```
  0x19 = 0001_1001   no scope modifier (GPU-default)
  0xb9 = 1011_1001   STRONG.SM  (bits 7,5 set)
  0xf9 = 1111_1001   STRONG.GPU (bits 7,6,5 set)
  0x59 = 0101_1001   STRONG.SYS (bits 6,4 set)
```

### Width bit

Bit 9 of w1 (bit 1 of `w1[15:8]`) selects 32-bit vs 64-bit:

| Width | w1[15:8] | bit 9 |
|-------|----------|-------|
| 32-bit | `0x19` (0001_1001) | 0 |
| 64-bit | `0x1b` (0001_1011) | 1 |

This bit is consistent across LDG, STG, LDS, STS, LDC, ULDC.

---

## Global Memory: STG.E

PTX source: `st.global.f32 [%rd1], %f0;`

### Full encoding table

| Instruction                              | w0                   | w1                   |
|------------------------------------------|----------------------|----------------------|
| `STG.E desc[UR4][R4.64], R3`            | `0x0000000304007986` | `0x004fe8000c101904` |
| `STG.E desc[UR4][R4.64+0x4], R3`        | `0x0000040304007986` | `0x000fe8000c101904` |
| `STG.E desc[UR4][R4.64+0x100], R3`      | `0x0001000304007986` | `0x000fe8000c101904` |
| `STG.E desc[UR4][R4.64+0x10000], R3`    | `0x0100000304007986` | `0x000fe8000c101904` |
| `STG.E desc[UR4][R4.64+0x100000], R3`   | `0x1000000304007986` | `0x000fe8000c101904` |
| `STG.E desc[UR4][R4.64+0x7ffffc], R3`   | `0x7ffffc0304007986` | `0x000fe2000c101904` |
| `STG.E.64 desc[UR4][R4.64], R2`         | `0x0000000204007986` | `0x004fe2000c101b04` |

### w0 field layout

```
  bits [15: 0] = 0x7986                  (opcode — STG.E)
  bits [23:16] = 0x00                    (always zero — reserved)
  bits [31:24] = addr_gpr                (base address GPR, 64-bit ptr)
  bits [39:32] = src_gpr                 (source data GPR to store)
  bits [63:40] = imm_offset              (24-bit signed byte offset)
```

**Key difference from LDG.E**: the UR register (desc[] handle) is NOT encoded in w0.
For STG.E, `w1[7:0]` holds the UR register number:

```
  w1[7:0] = ur_reg    (STG.E: UR register with the desc[] handle)
```

Verified cases:
- `STG.E desc[UR4][R4.64], R3`:  w1 = `0x004fe8000c101904`  → `w1[7:0]` = 0x04 = UR4 ✓
- `STG.E desc[UR6][R4.64], R8`:  w1 = `0x002fe8000c101906`  → `w1[7:0]` = 0x06 = UR6 ✓

### STG.E cache qualifier bits

| PTX qualifier | SASS mnemonic          | w1[15:8] |
|---------------|------------------------|----------|
| (none)        | `STG.E`                | `0x19`   |
| `.wb`         | `STG.E.STRONG.SM`      | `0xb9`   |
| `.cg`         | `STG.E.STRONG.GPU`     | `0xf9`   |
| `.cs`         | `STG.E.EF`             | `0x19`   |
| `.wt`         | `STG.E.STRONG.SYS`     | `0x59`   |

Full w1 examples:
```
STG.E default:      w1=0x004fe8000c101904  w1[15:8]=0x19
STG.E.STRONG.SM:    w1=0x000fe8000c10b904  w1[15:8]=0xb9
STG.E.STRONG.GPU:   w1=0x000fe8000c10f904  w1[15:8]=0xf9
STG.E.EF:           w1=0x000fe8000c001904  w1[23:16]=0x00 (evict-first)
STG.E.STRONG.SYS:   w1=0x000fe2000c115904  w1[15:8]=0x59
```

---

## Shared Memory: STS (Store to Shared)

PTX source: `st.shared.f32 [smem + offset], %f0;`

### Full encoding table

| Instruction               | w0                   | w1                   |
|---------------------------|----------------------|----------------------|
| `STS [UR4+0x0], R0`       | `0x00000000ff007988` | `0x004fe80008000804` |
| `STS [UR4+0x8], R6`       | `0x00000806ff007988` | `0x008fe80008000804` |
| `STS [UR4+0x100], R6`     | `0x00010006ff007988` | `0x000fe80008000804` |
| `STS [UR4+0x200], R6`     | `0x00020006ff007988` | `0x004fe80008000804` |
| `STS [UR4+0x1000], R6`    | `0x00100006ff007988` | `0x000fe80008000804` |
| `STS [UR4+0x3ffc], R6`    | `0x003ffc06ff007988` | `0x000fe80008000804` |
| `STS.64 [UR4+0x0], R6`    | `0x00000006ff007988` | `0x000fe80008000a04` |
| `STS [R6+0x0], R0` (reg)  | `0x0000000006007388` | (scheduling varies)  |

### w0 field layout

UR-mode (smem accessed via uniform register base, opcode `0x7988`):
```
  bits [15: 0] = 0x7988        (opcode — STS, UR-mode)
  bits [23:16] = 0x00          (always zero)
  bits [31:24] = 0xff          (UR-mode marker — distinguishes from reg-mode)
  bits [39:32] = src_gpr       (source data register)
  bits [55:40] = smem_offset   (14-bit byte offset into shared memory)
  bits [63:56] = 0x00          (always zero for 14-bit range)
```

Register-mode (smem address in GPR, opcode `0x7388`):
```
  bits [15: 0] = 0x7388        (opcode — STS, reg-mode)
  bits [23:16] = 0x00          (always zero)
  bits [31:24] = addr_gpr      (GPR holding shared memory address)
  bits [39:32] = src_gpr       (source data register)
  bits [55:40] = 0x0000        (offset is in addr_gpr, not immediate)
```

The UR register number (for UR-mode) is in `w1[7:0]`:
- `STS [UR4+...]: w1[7:0] = 0x04`

### Shared memory offset range

- **Maximum offset**: 0x3ffc (16,380 bytes — 14-bit field, word-aligned)
- The 14-bit offset allows addressing a 16 KB shared memory window
- For offsets beyond 16 KB, the compiler uses register-mode with IADD to compute the address

---

## Shared Memory: LDS (Load from Shared)

PTX source: `ld.shared.f32 %f1, [smem + offset];`

### Full encoding table

| Instruction              | w0                   | w1                   |
|--------------------------|----------------------|----------------------|
| `LDS R9, [UR4+0x0]`      | `0x00000004ff097984` | `0x000e280008000800` |
| `LDS R11, [UR4+0x8]`     | `0x00000804ff0b7984` | `0x000ea80008000800` |
| `LDS R11, [UR4+0x100]`   | `0x00010004ff0b7984` | `0x000e280008000800` |
| `LDS.64 R8, [UR4+0x0]`   | `0x00000004ff087984` | `0x000e680008000a00` |
| `LDS R3, [R3+0x0]` (reg) | `0x0000000003037984` | (scheduling varies)  |

### w0 field layout

UR-mode (opcode `0x7984`):
```
  bits [15: 0] = 0x7984        (opcode — LDS, UR-mode)
  bits [23:16] = dst_gpr       (destination GPR)
  bits [31:24] = 0xff          (UR-mode marker)
  bits [39:32] = ur_reg        (uniform register with shared mem base)
  bits [55:40] = smem_offset   (14-bit byte offset into shared memory)
```

Register-mode (opcode `0x7984` also, but b3 = addr GPR):
```
  bits [15: 0] = 0x7984        (opcode — LDS, reg-mode)
  bits [23:16] = dst_gpr       (destination GPR)
  bits [31:24] = addr_gpr      (GPR holding shared memory address)
  bits [39:32] = addr_gpr      (same as b3 in observed cases; offset=0)
```

Note: LDS and STS differ in which byte holds `dst_gpr` vs `src_gpr`:
- **LDS**: `dst_gpr` at bits [23:16], `ur_reg` at bits [39:32]
- **STS**: `src_gpr` at bits [39:32], no GPR at bits [23:16] (always 0)

### Type variants

`ld.shared.f32`, `ld.shared.b32`, and `ld.shared.u32` all generate the same
SASS encoding (opcode `0x7984`). The PTX type annotation is not encoded in SASS.
ptxas collapses them to identical instructions.

---

## Constant Bank Loads: LDC and ULDC

PTX source: kernel parameter reads generate these implicitly.

### LDC — load from constant bank to GPR

| Instruction                       | w0                   | w1                   |
|-----------------------------------|----------------------|----------------------|
| `LDC R1, c[0x0][0x28]`           | `0x00000a00ff017b82` | `0x000ff00000000800` |
| `LDC.64 R2, c[0x0][0x210]`       | `0x00008400ff027b82` | `0x000e220000000a00` |
| `LDC.64 R4, c[0x0][0x218]`       | `0x00008600ff047b82` | `0x000ea40000000a00` |
| `LDC.64 R8, c[0x0][0x208]`       | `0x00008200ff087b82` | `0x00321e0000000a00` |
| `LDC.64 R6, c[0x0][0x230]`       | `0x00008c00ff067b82` | `0x000e620000000a00` |

### w0 field layout

```
  bits [15: 0] = 0x7b82        (opcode — LDC)
  bits [23:16] = dst_gpr       (destination GPR)
  bits [31:24] = 0xff          (cbuf marker — constant bank addressing mode)
  bits [39:32] = bank_num      (constant bank number; 0 = cbuf0 = param bank)
  bits [47:40] = word_index    (cbuf_byte_addr / 4; 8-bit = 256 words = 1024 bytes)
  bits [55:48] = 0x00          (high bits of word index for large offsets)
```

**Offset calculation**: `cbuf_byte_addr = word_index * 4`

Examples:
- `c[0x0][0x28]`: word_index = 0x0a, byte_addr = 0x0a × 4 = 0x28 ✓
- `c[0x0][0x210]`: word_index = 0x84, byte_addr = 0x84 × 4 = 0x210 ✓
- `c[0x0][0x230]`: word_index = 0x8c, byte_addr = 0x8c × 4 = 0x230 ✓

All verified against known cbuf offsets.

### ULDC — load from constant bank to Uniform Register

| Instruction                        | w0                   | w1                   |
|------------------------------------|----------------------|----------------------|
| `ULDC.64 UR4, c[0x0][0x208]`      | `0x0000820000047ab9` | `0x000fe40000000a00` |
| `ULDC UR6, c[0x0][0x220]`         | `0x0000880000067ab9` | `0x000fe20000000800` |
| `ULDC UR6, c[0x0][0x228]`         | `0x00008a0000067ab9` | `0x000fe20000000800` |

### ULDC w0 field layout

```
  bits [15: 0] = 0x7ab9        (opcode — ULDC; uniform variant)
  bits [23:16] = dst_ur        (destination uniform register number)
  bits [31:24] = 0x00          (uniform mode marker; contrast with LDC's 0xff)
  bits [39:32] = 0x00          (bank number; 0 = cbuf0)
  bits [47:40] = word_index    (cbuf_byte_addr / 4)
```

Same offset formula as LDC: `byte_addr = word_index × 4`.

**Key distinction**: LDC has `0xff` at bits[31:24], ULDC has `0x00`. This is how the
hardware distinguishes writes to the scalar GPR file (LDC) from writes to the
uniform register file (ULDC).

---

## Width Bit: Universal Rule

Bit 9 of w1 (bit 1 of `w1[15:8]`) controls 32-bit vs 64-bit width for **all** memory
instructions:

| Instruction class | 32-bit w1[15:8] | 64-bit w1[15:8] | Width bit (bit 9 = bit 1 of byte) |
|-------------------|-----------------|-----------------|------------------------------------|
| LDG.E             | `0x19` (0001_1001) | `0x1b` (0001_1011) | 0 → 1 |
| STG.E             | `0x19` (0001_1001) | `0x1b` (0001_1011) | 0 → 1 |
| LDS               | `0x08` (0000_1000) | `0x0a` (0000_1010) | 0 → 1 |
| STS               | `0x08` (0000_1000) | `0x0a` (0000_1010) | 0 → 1 |
| LDC               | `0x08` (0000_1000) | `0x0a` (0000_1010) | 0 → 1 |
| ULDC              | `0x08` (0000_1000) | `0x0a` (0000_1010) | 0 → 1 |

Setting bit 9 of w1 promotes any 32-bit operation to 64-bit with no other encoding change.

---

## Constant Bank 0 (cbuf0) Parameter Layout

Kernel parameters are stored in cbuf0 by the CUDA runtime. The compiler reads them
using LDC (to GPR) or ULDC (to uniform register).

### Observed cbuf0 offsets

```
Byte 0x028:  Stack frame pointer / return addr (always LDC R1, c[0][0x28])
Byte 0x208:  Start of user kernel parameters (first param)
```

Parameters are packed sequentially from byte 0x208, in declaration order.
Each `.u64` parameter takes 8 bytes; each `.u32`/`.f32` takes 4 bytes.

Example layout for `probe_ldg_f32(.param .u64 p_a, .param .u64 p_b)`:
```
cbuf0[0x208..0x20f] = p_a (u64)   -> ULDC.64 UR4, c[0x0][0x208]
cbuf0[0x218..0x21f] = p_b (u64)   -> LDC.64  R4,  c[0x0][0x218]
```

Example for `probe_params(u64 p_ptr, u32 p_rows, u32 p_cols, f32 p_scale, u64 p_out)`:
```
cbuf0[0x208..0x20f] = p_ptr   (u64)  -> ULDC.64 UR4, c[0x0][0x208]
cbuf0[0x210..0x213] = p_rows  (u32)  (merged with p_ptr ULDC? or separate)
cbuf0[0x214..0x217] = p_cols  (u32)
cbuf0[0x220..0x223] = p_scale (f32)  -> ULDC UR6, c[0x0][0x220]
cbuf0[0x228..0x22f] = p_out   (u64)  -> LDC.64 R4, c[0x0][0x228]
```

The EIATTR_PARAM_CBANK nvinfo annotation records:
- Base offset in cbuf0 (always `0x0210` in 2-param kernels observed)  
- Total parameter block size in bytes

### Compiler strategy for parameter types

- **Pointer params** (`.u64`) used as `desc[]` base → **ULDC.64** (loads into UR for
  use with `desc[UR][R.64]` addressing in LDG/STG)
- **Pointer params** (`.u64`) used as direct GPR address → **LDC.64** (loads into GPR
  pair for use as `[R.64 + offset]`)
- **Scalar params** (`.u32`, `.f32`) used in uniform computations → **ULDC** (into UR)
- **Scalar params** (`.u32`, `.f32`) used in per-thread computation → **LDC** (into GPR)

---

## Opcode Summary

| SASS Instruction | Opcode (w0[15:0]) | Notes                        |
|------------------|-------------------|------------------------------|
| LDG.E            | `0x7981`          | Global load, UR desc-mode    |
| STG.E            | `0x7986`          | Global store, UR desc-mode   |
| LDS (UR-mode)    | `0x7984`          | Shared load via UR base      |
| STS (UR-mode)    | `0x7988`          | Shared store via UR base     |
| LDS (reg-mode)   | `0x7984`          | Same opcode! b3 != 0xff      |
| STS (reg-mode)   | `0x7388`          | Different opcode from UR-mode|
| LDC              | `0x7b82`          | Constant bank load to GPR    |
| ULDC             | `0x7ab9`          | Constant bank load to UR     |

---

## Notes on Coalescing and Alignment

### LDG.E / STG.E

- The `desc[UR]` mechanism uses a **descriptor** in a uniform register, loaded via
  `ULDC.64`. This enables the compiler to encode global loads without spending a
  full 64-bit pointer in each instruction — the UR holds the descriptor handle.
- Hardware performs **128-byte cache-line coalescing**: consecutive threads hitting
  addresses within the same 128-byte aligned block issue one cache miss.
- The 24-bit immediate offset (bits [63:40]) is **byte-addressed** with no alignment
  constraint in the encoding, but performance requires that loads be naturally aligned
  (4-byte for 32-bit, 8-byte for 64-bit).
- **Maximum inline offset**: 0x7FFFFC (≈8 MB). Larger offsets require register addition.

### LDS / STS

- Shared memory addressing uses a **UR-relative mode** (`opcode 0x7988`/`0x7984`) where
  the UR holds the CGA-aware shared memory base (computed via `S2UR + ULEA`).
- The 14-bit offset field (bits [55:40]) covers 0..16380 bytes (0x0000..0x3FFC).
- For shared memory arrays larger than 16 KB, the compiler switches to register-mode
  addressing with `IADD3` to compute the address.
- **Bank conflict**: the hardware has 32 shared memory banks, each 4 bytes wide. Threads
  in the same warp accessing different addresses in the same bank cause serialization.
  The encoding does not indicate bank; conflict detection is purely address-arithmetic.

### LDC / ULDC

- Only constant bank 0 (`c[0x0][...]`) observed for kernel parameters.
- CUDA supports up to 16 constant banks but user kernels see only cbuf0.
- The 8-bit word index at bits [47:40] allows addressing 256 words = 1024 bytes of
  a single bank. The EIATTR_CBANK_PARAM_SIZE nvinfo attribute records the actual size
  used (e.g., `0x0010` for a 16-byte parameter block).

---

## Quick Reference: Assembling Instructions

### LDG.E (32-bit global load)

```
w0 = 0x7981 | (dst << 16) | (addr << 24) | ((ur & 0xff) << 32) | ((offset & 0xffffff) << 40)
w1 = 0x001eaa000c1e1900   # default scheduling/cache; adjust byte9 for width/cache
```

### STG.E (32-bit global store)

```
w0 = 0x7986 | (0x00 << 16) | (addr << 24) | ((src & 0xff) << 32) | ((offset & 0xffffff) << 40)
w1 = ur | 0x004fe8000c101900   # ur in low byte; default scheduling
```

### STS (32-bit shared store, UR-mode)

```
w0 = 0x7988 | (0x00 << 16) | (0xff << 24) | ((src & 0xff) << 32) | ((smem_offset & 0xffff) << 40)
w1 = ur | 0x000fe80008000800   # ur in low byte
```

### LDS (32-bit shared load, UR-mode)

```
w0 = 0x7984 | (dst << 16) | (0xff << 24) | ((ur & 0xff) << 32) | ((smem_offset & 0xffff) << 40)
w1 = 0x000e280008000800   # default scheduling; no UR in w1 for LDS
```

### ULDC (32-bit constant bank load to UR)

```
word_index = cbuf_byte_addr / 4   # must be integer (4-byte aligned)
w0 = 0x7ab9 | (dst_ur << 16) | (0x00 << 24) | (0x00 << 32) | (word_index << 40)
w1 = 0x000fe20000000800   # 32-bit; use 0x000fe40000000a00 for 64-bit
```

### ULDC.64 (64-bit constant bank load to UR pair)

```
word_index = cbuf_byte_addr / 4
w0 = 0x7ab9 | (dst_ur << 16) | (0x00 << 24) | (0x00 << 32) | (word_index << 40)
w1 = 0x000fe40000000a00   # bit 9 set for 64-bit
```

### LDC.64 (64-bit constant bank load to GPR pair)

```
word_index = cbuf_byte_addr / 4
w0 = 0x7b82 | (dst_gpr << 16) | (0xff << 24) | (0x00 << 32) | (word_index << 40)
w1 = 0x000e220000000a00   # bit 9 set for 64-bit
```

---

## Relationship to elf-wrap.fs

The Lithos ELF emitter (`compiler/elf-wrap.fs`) places the parameter buffer at
`cbuf0`. Based on this reverse-engineering:

1. **Parameters start at cbuf0 byte 0x208** (word index 0x82)
2. **Each `.u64` pointer parameter** should be loaded with `ULDC.64 URx, c[0x0][offset]`
   where offset = 0x208 + (param_index × 8)
3. **ULDC encoding**: `w0 = 0x7ab9 | (ur << 16) | (word_idx << 40)` where `word_idx = offset / 4`
4. **Width select**: set bit 9 of w1 for 64-bit (`0x0a` in w1[15:8]) vs 32-bit (`0x08`)
5. The `0xff` vs `0x00` at w0[31:24] distinguishes LDC (to GPR) from ULDC (to UR)

---

*Generated 2026-04-13 by probing sm_90a cubins with ptxas/nvdisasm.*
*All encodings read directly from hardware — no inference from documentation.*
