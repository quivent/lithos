# Hopper sm_90a SASS: Integer, Atomic, and Bit Operations
## Empirically verified from nvdisasm --print-instruction-encoding

All encodings compiled with `ptxas -arch sm_90a` and disassembled with
`nvdisasm --print-instruction-encoding`. Each 128-bit instruction is two
64-bit words: `word0` (instruction) and `word1` (control / scheduling).

Register field layout in `word0` is uniform across all arithmetic/logic ops:

```
bits[15: 0]  base opcode (instruction identity + immediate flag)
bits[23:16]  Rd  destination register
bits[31:24]  Ra  first source register (or address offset)
bits[39:32]  Rb  second source register (or immediate for imm-form)
bits[63:40]  upper modifier bits (usually 0 for 32-bit ops)
```

The **third source (Rc / addend)** for three-source instructions (IMAD, IADD3, SHF)
is encoded in `ctrl word bits[7:0]`. `RZ = 0xff`.

---

## Integer Arithmetic

### 1. IMAD — Integer Multiply-Accumulate

PTX `mul.lo.s32` and `mad.lo.s32` both emit IMAD.

```
IMAD Rd, Ra, Rb, RZ         ; mul.lo.s32  (addend = RZ)
  word0 = 0x0000000502097224  [R9,R2,R5,RZ example]
  ctrl  = 0x004fca00078e02ff

IMAD Rd, Ra, Rb, Rc         ; mad.lo.s32  (addend = Rc)
  word0 = 0x0000000502097224  [same word0 — Rc is in ctrl!]
  ctrl  = 0x004fca00078e0202  [ctrl bits[7:0] = Rc register number]
```

**Opcode**: `0x7224`

**Key finding**: `mul.lo.s32` and `mad.lo.s32` produce **identical word0**.
The addend register `Rc` lives in `ctrl bits[7:0]`. `RZ = 0xff`.

`mul.lo.u32` emits the same IMAD opcode — unsigned vs signed is not
distinguished by ptxas in the opcode (both are 32-bit multiply with truncation).

Field template:
```
word0 = 0x7224 | (Rd << 16) | (Ra << 24) | (Rb << 32)
ctrl[7:0] = Rc   (0xff = RZ)
```

---

### 2. IADD3 — Three-Input Integer Add

PTX `add.s32` emits IADD3 (Ampere/Hopper uses IADD3 even for two-operand add).

**Register form** (`add.s32 Rd, Ra, Rb`):
```
IADD3 Rd, Ra, Rb, RZ
  word0 = 0x0000000502097210  [R9,R2,R5,RZ]
  ctrl  = 0x004fca0007ffe0ff

  Opcode [15:0] = 0x7210
  word0 = 0x7210 | (Rd << 16) | (Ra << 24) | (Rb << 32)
```

**Immediate form** (`add.s32 Rd, Ra, imm`):
```
IADD3 Rd, Ra, imm, RZ
  word0 = 0x0000002a02077810  [R7, R2, 42, RZ]
  ctrl  = 0x004fca0007ffe0ff

  Opcode [15:0] = 0x7810  (bit 11 set = immediate form)
  Immediate = bits[39:32] = 0x2a = 42
  word0 = 0x7810 | (Rd << 16) | (Ra << 24) | (imm << 32)
```

Immediate flag: bit 11 of the opcode word distinguishes `0x7210` (reg) from `0x7810` (imm).

---

### 3. SHF — Shift (Funnel Shift)

All shifts emit SHF (funnel shift). Shift amount is always in `word0 bits[39:32]`.
The data source register for right-shifts is in `ctrl bits[7:0]`.

**Left shift** (`shl.b32 Rd, Ra, imm`):
```
SHF.L.U32 Rd, Ra, imm, RZ
  word0 = 0x0000000702077819  [R7, R2, 7, RZ]
  ctrl  = 0x004fca00000006ff

  Opcode = 0x7819
  word0 = 0x7819 | (Rd << 16) | (Ra << 24) | (imm << 32)
  ctrl[7:0] = 0xff (RZ — upper input for funnel shift is zero)
```

**Arithmetic right shift** (`shr.s32 Rd, Ra, imm`):
```
SHF.R.S32.HI Rd, RZ, imm, Rs     ; note: Ra=RZ, Rs is in ctrl
  word0 = 0x00000003ff077819  [R7, RZ, 3, R2]
  ctrl  = 0x004fca0000011402

  word0 bits[31:24] = 0xff = RZ (funnel upper half = RZ)
  word0 bits[39:32] = shift amount (imm=3)
  ctrl bits[7:0]    = Rs = source register (R2=0x02)
```

**Logical right shift** (`shr.u32 Rd, Ra, imm`):
```
SHF.R.U32.HI Rd, RZ, imm, Rs
  word0 = 0x00000003ff077819  [same word0 as SHF.R.S32!]
  ctrl  = 0x004fca0000011602

  SHF.R.S32 ctrl bits[9:8] = 0b00 (signed)
  SHF.R.U32 ctrl bits[9:8] = 0b10 (unsigned)
  XOR diff = 0x0000000000000200
```

---

## Compare and Set Predicate

### 4. ISETP — Integer Set Predicate

**Opcode**: `0x720c`

```
ISETP.GE.AND P0, PT, Ra, Rb, PT
  word0 = 0x000000050200720c  [Ra=R2, Rb=R5]
  ctrl  = 0x004fc80003f06270
```

Predicate destination registers are encoded in the **ctrl word**, not word0.

**PTX-to-SASS mapping** (ptxas optimizes by inverting condition):

| PTX | SASS | SEL modifier |
|-----|------|--------------|
| `setp.lt.s32` | `ISETP.GE.AND` + `SEL Rd, RZ, 1, P0` | `P0` (inverted) |
| `setp.ge.s32` | `ISETP.GE.AND` + `SEL Rd, RZ, 1, !P0` | `!P0` |
| `setp.eq.s32` | `ISETP.NE.AND` + `SEL Rd, RZ, 1, P0` | `P0` |

Condition code is in `ctrl bits[13:12]`:
```
ctrl GE (0x62..): bits[13:12] = 0b10
ctrl NE (0x52..): bits[13:12] = 0b01
```

---

## Bit Operations — LOP3

Ampere/Hopper unifies AND/OR/XOR into `LOP3.LUT` (logical op with 8-bit truth table).

**Opcode**: `0x7212` (register form), `0x7812` (immediate form)

The 8-bit truth table lives in **ctrl word bits[15:8]**.

### Truth Tables

| Operation | Truth Table | Binary |
|-----------|------------|--------|
| AND | `0xc0` | `11000000` |
| OR  | `0xfc` | `11111100` |
| XOR | `0x3c` | `00111100` |

**Register form** (`and.b32 Rd, Ra, Rb`):
```
LOP3.LUT Rd, Ra, Rb, RZ, 0xc0, !PT
  word0 = 0x0000000502097212  [R9, R2, R5, RZ]
  ctrl  = 0x004fca00078ec0ff  (truth table 0xc0 in bits[15:8])

  word0 = 0x7212 | (Rd << 16) | (Ra << 24) | (Rb << 32)
  ctrl  = base_ctrl | (truth_table << 8)
  ctrl bits[7:0] = 0xff (Rc=RZ)
```

**Immediate form** (`and.b32 Rd, Ra, imm` — critical for W4A16 nibble extract):
```
LOP3.LUT Rd, Ra, 0xf, RZ, 0xc0, !PT    ; and.b32 Rd, Ra, 0xf
  word0 = 0x0000000f02077812  [R7, R2, imm=0xf, RZ]
  ctrl  = 0x004fca00078ec0ff

  Opcode = 0x7812  (bit 11 set = immediate form)
  Immediate = bits[39:32] (here imm=0xf for nibble mask)
  word0 = 0x7812 | (Rd << 16) | (Ra << 24) | (imm << 32)
```

### Encoding Template

```
LOP3 register form:
  word0 = 0x7212 | (Rd << 16) | (Ra << 24) | (Rb << 32)
  ctrl  = 0x004fca00078e0000 | (lut << 8) | 0xff   [Rc=RZ in bits[7:0]]

LOP3 immediate form:
  word0 = 0x7812 | (Rd << 16) | (Ra << 24) | (imm << 32)
  ctrl  = 0x004fca00078e0000 | (lut << 8) | 0xff
```

---

## Conversions

### 5. I2FP — Integer to Float

PTX `cvt.rn.f32.s32` and `cvt.rn.f32.u32` emit `I2FP`.

**Opcode**: `0x7245` (same word0 for both — type in ctrl word)

```
I2FP.F32.S32 Rd, Rs     ; cvt.rn.f32.s32
  word0 = 0x0000000200077245  [R7, R2]
  ctrl  = 0x004fca0000201400

I2FP.F32.U32 Rd, Rs     ; cvt.rn.f32.u32
  word0 = 0x0000000200077245  [same word0!]
  ctrl  = 0x004fca0000201000

  S32 vs U32 XOR in ctrl = 0x0000000000000400
  Type encoded in ctrl bits[10:8] (approximately)
```

Field template:
```
word0 = 0x7245 | (Rd << 16) | (Rs << 24)
```

---

## Atomics — ATOMG / REDG (HIGHEST PRIORITY)

On Hopper, global atomic ops are `ATOMG` (not `ATOM`). The instruction
uses **descriptor-based addressing**: `desc[URa][Ra]` where `URa` is a
uniform register (not encoded in `word0`) and `Ra` is the per-thread
address offset.

### Field Layout for ATOMG

```
word0 field layout:
  bits[15: 0]  base opcode
  bits[23:16]  Rd  destination register (RZ=0 for no-return)
  bits[31:24]  Ra  address offset register (64-bit pair)
  bits[39:32]  Rs  source data register

The uniform descriptor register URa is NOT in word0.
It is encoded in the ctrl word.
```

### 6. ATOMG.E.ADD.F32 — Floating-Point Atomic Add

```
ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU PT, Rd, desc[UR4][Ra], Rs
  word0 = 0x79a3 | (Rd << 16) | (Ra << 24) | (Rs << 32)
  ctrl  = 0x004e2800081ef3c4

Example (Rd=R5, Ra=R4, Rs=R3):
  word0 = 0x00000003040579a3
  ctrl  = 0x004e2800081ef3c4
```

**Opcode**: `0x79a3`  
**Sub-opcode bits[3:0]**: `0x3` = ADD.F32

### 7. ATOMG.E.ADD.U32 — Unsigned Integer Atomic Add (Grid Sync Counter)

```
ATOMG.E.ADD.STRONG.GPU PT, Rd, desc[UR4][Ra], Rs
  word0 = 0x79a8 | (Rd << 16) | (Ra << 24) | (Rs << 32)
  ctrl  = 0x004e2800081ee1c4

Example (Rd=R5, Ra=R4, Rs=R3):
  word0 = 0x00000003040579a8
  ctrl  = 0x004e2800081ee1c4
```

**Opcode**: `0x79a8`

Difference from F32: ctrl `bits[15:8]` = `0xe1` (U32) vs `0xf3` (F32 with FTZ+RN).

> **Note**: ptxas automatically optimizes `atom.global.add.u32` into a
> warp-reduction pattern using `@P0 ATOMG`. For cooperative grid sync
> emit the instruction directly without the warp reduction wrapper.

### 8. ATOMG.E.EXCH — Atomic Exchange

```
ATOMG.E.EXCH.STRONG.GPU PT, Rd, desc[UR4][Ra], Rs
  word0 = 0x79a8 | (Rd << 16) | (Ra << 24) | (Rs << 32)
  ctrl  = 0x004e28000c1ee1c4

Example (Rd=R5, Ra=R4, Rs=R3):
  word0 = 0x00000003040579a8
  ctrl  = 0x004e28000c1ee1c4
```

**EXCH shares opcode `0x79a8` with ADD.U32**. Distinction is entirely in ctrl word.

### 9. REDG.E.ADD.F32 — Reduction (No Return Value)

```
REDG.E.ADD.F32.FTZ.RN.STRONG.GPU desc[UR4][Ra], Rs
  word0 = 0x79a6 | (0 << 16) | (Ra << 24) | (Rs << 32)
  ctrl  = 0x004fe2000c10f384

Example (Ra=R4, Rs=R3):
  word0 = 0x00000003040079a6
  ctrl  = 0x004fe2000c10f384

  Note: Rd field = 0 (no destination register)
```

**Opcode**: `0x79a6`

---

## Address Computation

### 10. IMAD.WIDE.U32 — Wide Multiply for Pointer Arithmetic

PTX `mad.wide.u32` and `mul.wide.u32` emit `IMAD.WIDE.U32` (64-bit result).

```
IMAD.WIDE.U32 Rd_pair, Ra, Rb, RZ     ; mul.wide.u32
  word0 = 0x0000000502067225  [Rd_pair=R6:R7, Ra=R2, Rb=R5]
  ctrl  = ...

mad.wide.u32 expands to:
  IMAD.WIDE.U32 R8, R2, R5, RZ        ; 64-bit multiply
  IADD3 R8, P0, R8, UR6, RZ           ; add base pointer (low word)
  IADD3.X R9, R9, UR7, RZ, P0, !PT   ; add base pointer (high word + carry)
```

**IMAD.WIDE opcode**: `0x7225` (vs `0x7224` for 32-bit IMAD)

---

## Summary Opcode Table

| Instruction | PTX | Opcode [15:0] | Notes |
|-------------|-----|----------------|-------|
| IMAD | `mul.lo.s32`, `mad.lo.s32` | `0x7224` | Rc in ctrl[7:0] |
| IMAD.WIDE | `mul.wide.u32`, `mad.wide.u32` | `0x7225` | 64-bit result |
| IADD3 | `add.s32` (reg) | `0x7210` | Rc=RZ in ctrl[7:0] |
| IADD3 | `add.s32` (imm) | `0x7810` | bit11=imm flag |
| SHF | `shl.b32`, `shr.s32`, `shr.u32` | `0x7819` | Rs in ctrl[7:0] |
| LOP3 | `and/or/xor.b32` (reg) | `0x7212` | lut in ctrl[15:8] |
| LOP3 | `and.b32` (imm) | `0x7812` | bit11=imm flag |
| ISETP | `setp.lt/ge/eq.s32` | `0x720c` | cond in ctrl[13:12] |
| I2FP | `cvt.rn.f32.s32` | `0x7245` | type in ctrl |
| I2FP | `cvt.rn.f32.u32` | `0x7245` | type in ctrl |
| ATOMG.ADD.F32 | `atom.global.add.f32` | `0x79a3` | |
| ATOMG.ADD.U32 | `atom.global.add.u32` | `0x79a8` | |
| ATOMG.EXCH | `atom.global.exch.b32` | `0x79a8` | same as ADD.U32 base |
| REDG.ADD.F32 | `red.global.add.f32` | `0x79a6` | no Rd |

---

## ATOM Stub Correction

The `emit-sass.fs` stub had `OP-ATOM-ADD = $798b`. This value was a guess.

**Empirically verified corrections:**

```
; OLD (wrong):
$798b constant OP-ATOM-ADD

; NEW (verified from nvdisasm):
$79a8 constant OP-ATOM-ADD-U32   \ ATOMG.E.ADD.STRONG.GPU — integer counter
$79a3 constant OP-ATOM-ADD-F32   \ ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU

; Hardcoded ctrl words (replace make-ctrl speculation):
$004e2800081ee1c4 constant CTRL-ATOM-U32   \ verified from nvdisasm
$004e2800081ef3c4 constant CTRL-ATOM-F32   \ verified from nvdisasm
$004fe2000c10f384 constant CTRL-REDG-F32   \ no-return reduction
```

The `ctrl-atom` function (`make-ctrl` with `extra41=0x00100000`) computed
`0x001ffc0000100000` — which is **completely wrong**. The actual ctrl word
for a U32 atomic is `0x004e2800081ee1c4`.

For the cooperative grid sync counter (U32 ADD):
```
: atom-add,  ( rd ra rs -- )
  32 lshift >r              \ rs -> bits[39:32]
  24 lshift >r              \ ra -> bits[31:24]
  16 lshift                 \ rd -> bits[23:16]
  $79a8 or r> or r> or      \ opcode | Rd | Ra | Rs
  $004e2800081ee1c4 sinst, ; \ verified ctrl word
```

---

## RZ Register Encoding

`RZ` (the always-zero register) is encoded as:
- `0xff` in instruction word register fields (bits[23:16], [31:24], [39:32])
- `0xff` in ctrl word `bits[7:0]` (Rc/addend position)

Confirmed by IMAD with `RZ` addend: `ctrl bits[7:0] = 0xff`.
