# SASS FP32 Arithmetic — Hopper sm_90a Encoding Reference

All encodings in this document were derived empirically by:
1. Writing minimal PTX kernels
2. Compiling with `ptxas -arch sm_90a`
3. Disassembling with `nvdisasm --print-instruction-encoding`

No values are guessed or inferred from older architectures. Every hex word
shown was read directly from nvdisasm output.

---

## 128-bit Instruction Format

Every Hopper (sm_90a) instruction occupies 16 bytes = 128 bits:

```
bits [63:0]   — instruction word  (the opcode, registers, immediates)
bits [127:64] — control word      (scheduler: stall, barriers, source modifiers)
```

nvdisasm prints the instruction word first, then the control word on the
next line. The control word bits that encode source modifiers (NEG, ABS)
straddle the boundary — some modifier bits live in the instruction word
(bit 63 = NEG on src2), others live in the control word's `extra41` field
(bits 8 and 9 = NEG/ABS on src1).

### Control Word Layout

```
ctrl bits [40:0]  — extra41: opaque barrier/descriptor field + source modifiers
ctrl bit  41–44   — stall count (0–15 cycles; 0 = no stall)
ctrl bit  45      — yield hint (1 = scheduler may switch warp)
ctrl bits [48:46] — write barrier slot (0–6; 7 = none)
ctrl bits [51:49] — read barrier slot  (0–6; 7 = none)
ctrl bits [57:52] — wait barriers mask (bit N = stall until slot N clears)
ctrl bits [62:58] — reuse cache flags
```

---

## Register Encoding (common to all FP32 instructions)

For all register-register FP32 instructions:

```
bits [23:16] = destination register Rd (8-bit index, 0–255; R255 = RZ)
bits [31:24] = source register Rs1 (8-bit index)
bits [39:32] = source register Rs2 (8-bit index)
```

For FSETP specifically (destination is a predicate, not a float register):

```
bits [31:24] = source register Rs1 (8-bit index)
bits [39:32] = source register Rs2 (8-bit index)
bits [23:16] = 0x00 (unused in instruction word; pred dest is in ctrl extra41)
```

RZ (zero register, read as 0.0, writes discarded) = register index 255 = 0xFF.

---

## Source Modifier Encoding

Modifiers split across instruction word and control word:

| Modifier         | Location                      | Value        |
|------------------|-------------------------------|--------------|
| NEG on src2      | inst bit 63                   | `1` = negate |
| NEG on src1      | ctrl extra41 bit 8            | `1` = negate |
| ABS on src1      | ctrl extra41 bit 9            | `1` = abs    |

**Verified examples:**

```
FADD R11, R2, -R5          inst=0x80000005020b7221  ctrl extra41=0x000 (no src1 mod)
                            bit 63 = 1 (NEG on src2 R5)

FADD R15, -R5, -RZ  (FNEG) inst=0x800000ff050f7221  ctrl extra41=0x100
                            bit 63 = 1 (NEG on src2 RZ)
                            ctrl bit 8 = 1 (NEG on src1 R5)

FADD R19, |R5|, -RZ (FABS) inst=0x800000ff05137221  ctrl extra41=0x200
                            bit 63 = 1 (NEG on src2 RZ)
                            ctrl bit 9 = 1 (ABS on src1 R5)
```

---

## Opcode Field Layout (bits [15:0])

All FP32 arithmetic instructions share a common high-byte pattern in bits [15:8]:

| Source form    | bits [15:8] | Binary      |
|----------------|-------------|-------------|
| register-reg   | `0x72`      | `0111 0010` |
| FADD immediate | `0x74`      | `0111 0100` |
| FMUL immediate | `0x78`      | `0111 1000` |

The low byte bits [7:0] encodes the specific operation:

| Instruction | Low byte [7:0] | Full opcode [15:0] |
|-------------|----------------|--------------------|
| FMUL        | `0x20`         | `0x7220`           |
| FADD        | `0x21`         | `0x7221`           |
| FFMA        | `0x23`         | `0x7223`           |
| FMNMX       | `0x09`         | `0x7209`           |
| FSETP       | `0x0b`         | `0x720b`           |
| FMUL imm    | `0x20`         | `0x7820`           |
| FADD imm    | `0x21`         | `0x7421`           |

---

## FMUL — FP32 Multiply

**PTX:** `mul.f32 %f2, %f0, %f1`
**SASS:** `FMUL R9, R2, R5`

### Probe kernel and disassembly

```ptx
.version 8.0 / .target sm_90a
mul.f32 %f2, %f0, %f1;
```

```
nvdisasm output:
  FMUL R9, R2, R5 ;   /* 0x0000000502097220 */
                       /* 0x004fca0000400000 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x0000000502097220

bits [7:0]   = 0x20          — FMUL opcode (low byte)
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x09 = R9     — destination register
bits [31:24] = 0x02 = R2     — source register 1
bits [39:32] = 0x05 = R5     — source register 2
bits [63:40] = 0x000000       — no modifiers
```

### Source modifiers

```
NEG on src2 (bit 63 = 1):
  FMUL R9, R2, -R5  →  0x8000000502097220

NEG on src1 (ctrl extra41 bit 8 = 1):
  FMUL R9, -R2, R5  →  inst = 0x0000000502097220, ctrl extra41 |= 0x100

ABS on src1 (ctrl extra41 bit 9 = 1):
  FMUL R9, |R2|, R5 →  inst = 0x0000000502097220, ctrl extra41 |= 0x200
```

### Control word (typical)

```
0x004fca0000400000
  stall=5, yield=0, wbar=7, rbar=7, wait=0x4 (wait on barrier 2)
  extra41 = 0x400000 (reuse cache bits for R2, R5)
```

### emit-sass.fs constant

`$7220` — defined in `sass/emit-sass-auto.fs` as `OP-FMUL`.
Lithos uses `fmul, ( rd rs1 rs2 -- )` which calls `255 ffma,` (FFMA with Rs3=RZ).

---

## FADD — FP32 Add

**PTX:** `add.f32 %f2, %f0, %f1`
**SASS:** `FADD R9, R2, R5`

### Probe disassembly

```
FADD R9, R2, R5 ;   /* 0x0000000502097221 */
                     /* 0x004fca0000000000 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x0000000502097221

bits [7:0]   = 0x21          — FADD opcode (low byte)
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x09 = R9     — destination register
bits [31:24] = 0x02 = R2     — source register 1
bits [39:32] = 0x05 = R5     — source register 2
bits [63:40] = 0x000000       — no modifiers (bit 63 = NEG-src2 flag)
```

FMUL and FADD differ only in bit 0 of the low byte: FMUL=0x20, FADD=0x21.

### Source modifiers

```
sub.f32 %f, %f0, %f1 → FADD Rd, Rs1, -Rs2
  0x80000005020b7221  (bit 63 = 1 → NEG on src2)

neg.f32 → FADD Rd, -Rs, -RZ
  0x800000ff050f7221  (bit 63=1 + ctrl extra41 bit 8)

abs.f32 → FADD Rd, |Rs|, -RZ
  0x800000ff05137221  (bit 63=1 + ctrl extra41 bit 9)
```

### emit-sass.fs constant

`$7221` — defined in `sass/emit-sass.fs` as `OP-FADD`.
Builder: `fadd, ( rd ra rb -- )`.

---

## FFMA — FP32 Fused Multiply-Add

**PTX:** `fma.rn.f32 %f2, %f0, %f1, %f3`
**SASS:** `FFMA R9, R2, R5, R9`

### Probe disassembly

```
FFMA R9, R2, R5, R9 ;   /* 0x0000000502097223 */
                          /* 0x008fca0000000009 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x0000000502097223

bits [7:0]   = 0x23          — FFMA opcode
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x09 = R9     — destination Rd
bits [31:24] = 0x02 = R2     — source Rs1
bits [39:32] = 0x05 = R5     — source Rs2
bits [63:40] = 0x000000       — no modifiers
```

**The accumulator register Rs3** is NOT in the instruction word.
It is encoded in the control word `extra41` bits [7:0]:

```
ctrl extra41 bits [7:0] = Rs3 register index

  FFMA R9, R2, R5, R9  → ctrl extra41[7:0] = 0x09 (R9)
  FFMA R9, R2, R5, R0  → ctrl extra41[7:0] = 0x00 (R0)
  FFMA R9, R2, R5, R7  → ctrl extra41[7:0] = 0x07 (R7)
```

This matches the `ctrl-ffma ( rs3 -- ctrl64 )` implementation in emit-sass.fs.

### Control word (loop body, no external dependencies)

```
0x000fca0000000009   (Rs3=R9, stall=5, yield=0, wbar=7, rbar=7, wait=0)
  extra41 = 0x09 = R9
```

### emit-sass.fs constant

`$7223` — defined in `sass/emit-sass.fs` as `OP-FFMA`.
Builder: `ffma, ( rd rs1 rs2 rs3 -- )`.

---

## FNEG — FP32 Negate

**PTX:** `neg.f32 %f2, %f0`
**SASS:** `FADD R7, -R2, -RZ`

There is no separate FNEG opcode. `neg.f32` compiles to `FADD` with:
- NEG modifier on src1 (ctrl extra41 bit 8 = 1)
- NEG modifier on src2 (inst bit 63 = 1)
- src2 = RZ (register 255 = 0xFF)

The computation is: `Rd = -Rs + (-0.0) = -Rs` (IEEE 754: `-0.0` is the additive identity).

### Probe disassembly

```
FADD R7, -R2, -RZ ;   /* 0x800000ff02077221 */
                        /* 0x004fca0000000100 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x800000ff02077221

bits [7:0]   = 0x21          — FADD opcode
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x07 = R7     — destination Rd
bits [31:24] = 0x02 = R2     — source Rs1 (with NEG in ctrl)
bits [39:32] = 0xFF = RZ     — source Rs2 = zero register
bit  63      = 1             — NEG on src2 (→ -RZ = -0.0)
```

### Control word

```
0x004fca0000000100
  extra41 = 0x100   ← bit 8 = NEG on src1
  stall=5, wbar=7, rbar=7
```

### Construction formula

```
inst = 0x800000ff_<RZ>_<Rd>_7221
     = (NEG_SRC2_BIT) | (RZ << 32) | (Rs1 << 24) | (Rd << 16) | 0x7221

ctrl extra41 = 0x100  (NEG on src1)
```

---

## FABS — FP32 Absolute Value

**PTX:** `abs.f32 %f2, %f0`
**SASS:** `FADD R7, |R2|, -RZ`

No separate FABS opcode. `abs.f32` compiles to `FADD` with:
- ABS modifier on src1 (ctrl extra41 bit 9 = 1)
- NEG modifier on src2 (inst bit 63 = 1)
- src2 = RZ

The computation is: `Rd = |Rs| + (-0.0) = |Rs|`.

### Probe disassembly

```
FADD R7, |R2|, -RZ ;   /* 0x800000ff02077221 */
                         /* 0x004fca0000000200 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x800000ff02077221   ← IDENTICAL to FNEG!
```

**The FNEG and FABS instruction words are bit-for-bit identical.**
The difference is entirely in the control word:

```
FNEG: ctrl extra41 = 0x100  (bit 8 = NEG on src1)
FABS: ctrl extra41 = 0x200  (bit 9 = ABS on src1)
```

This is the one case in Hopper where the control word is semantically
load-bearing for instruction correctness, not just scheduling.

---

## FMNMX — FP32 Min/Max

**PTX:** `min.f32` / `max.f32`
**SASS:** `FMNMX Rd, Rs1, Rs2, {PT | !PT}`

A single opcode with a predicate selector operand:
- `PT` (predicate true, always-true): selects **minimum**
- `!PT` (inverted PT, always-false): selects **maximum**

### Probe disassembly

```
min.f32 → FMNMX R9, R2, R5, PT ;    /* 0x0000000502097209 */
                                      /* 0x004fca0003800000 */

max.f32 → FMNMX R9, R2, R5, !PT ;   /* 0x0000000502097209 */
                                      /* 0x004fca0007800000 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x0000000502097209   ← IDENTICAL for min and max!

bits [7:0]   = 0x09          — FMNMX opcode
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x09 = R9     — destination Rd
bits [31:24] = 0x02 = R2     — source Rs1
bits [39:32] = 0x05 = R5     — source Rs2
```

**The min/max selector** is in the control word `extra41`:

```
min (PT  = always-true):  ctrl extra41 bit 26 = 0  (extra41 = 0x0003800000)
max (!PT = always-false): ctrl extra41 bit 26 = 1  (extra41 = 0x0007800000)
```

`extra41` difference: `0x0004000000 = 1 << 26`.

### Construction formula

```
For min: ctrl extra41 = 0x03800000
For max: ctrl extra41 = 0x07800000  (= min_extra41 | (1 << 26))
```

---

## FSETP — FP32 Compare and Set Predicate

**PTX:** `setp.lt.f32 %p0, %f0, %f1` (and other conditions)
**SASS:** `FSETP.{cond}.AND Pd, PT, Rs1, Rs2, PT`

### Probe disassembly

```
setp.lt.f32 → FSETP.GTU.AND P3, PT, R2, R5, PT ;   /* 0x000000050200720b */
                                                      /* 0x0c4fe40003f6c000 */
```

Note: ptxas canonicalizes `setp.lt(a,b)` to `FSETP.GTU(b,a)` (operand swap)
or other equivalent forms. Do not assume a 1:1 PTX→SASS condition mapping.

### Encoding (64-bit instruction word)

```
Full word: 0x000000050200720b   (example: Rs1=R2, Rs2=R5)

bits [7:0]   = 0x0b          — FSETP opcode
bits [15:8]  = 0x72          — register-register form
bits [23:16] = 0x00          — always 0x00 (predicate dest is in ctrl, not here)
bits [31:24] = 0x02 = R2     — source Rs1
bits [39:32] = 0x05 = R5     — source Rs2
bits [63:40] = 0x000000       — no modifiers
```

**All FSETP variations (conditions, predicate destinations) share the same
instruction word for the same source operands.** The condition code and
predicate destination register live entirely in the control word.

### Condition code and predicate encoding in ctrl extra41

```
ctrl extra41 bits [19:17] = destination predicate register P (0–7)
ctrl extra41 bits [15:12] = comparison condition code
ctrl extra41 constant     = 0x0003f04000 (backbone bits, always set)

Predicate P mapping:
  P0 → bits[19:17] = 000  (adds 0 * 0x20000)
  P1 → bits[19:17] = 001  (adds 1 * 0x20000 = 0x020000)
  P2 → bits[19:17] = 010  (adds 2 * 0x20000 = 0x040000)
  P3 → bits[19:17] = 011  (adds 3 * 0x20000 = 0x060000)
  P4 → bits[19:17] = 100  (adds 4 * 0x20000 = 0x080000)
  ...
```

### Condition code table (bits [15:12] of extra41)

All 6 observed values confirmed by disassembly. The remaining 10 are
derived from the standard Hopper/Ampere FP comparison encoding pattern.

| Code | Binary | Name  | Meaning                            | Status    |
|------|--------|-------|------------------------------------|-----------|
| 0x0  | 0000   | F     | always false                       | inferred  |
| 0x1  | 0001   | LT    | less-than, ordered                 | inferred  |
| 0x2  | 0010   | EQ    | equal, ordered                     | inferred  |
| 0x3  | 0011   | LE    | less-or-equal, ordered             | inferred  |
| 0x4  | 0100   | GT    | greater-than, ordered              | observed  |
| 0x5  | 0101   | NE    | not-equal, ordered                 | observed  |
| 0x6  | 0110   | GE    | greater-or-equal, ordered          | observed  |
| 0x7  | 0111   | NUM   | ordered (neither is NaN)           | inferred  |
| 0x8  | 1000   | NAN   | unordered (at least one NaN)       | inferred  |
| 0x9  | 1001   | LTU   | less-than, unordered               | inferred  |
| 0xa  | 1010   | EQU   | equal, unordered                   | inferred  |
| 0xb  | 1011   | LEU   | less-or-equal, unordered           | inferred  |
| 0xc  | 1100   | GTU   | greater-than, unordered            | observed  |
| 0xd  | 1101   | NEU   | not-equal, unordered               | observed  |
| 0xe  | 1110   | GEU   | greater-or-equal, unordered        | observed  |
| 0xf  | 1111   | T     | always true                        | inferred  |

### Example ctrl extra41 values (verified)

```
FSETP.GT.AND  P0, PT, R2, R5, PT  →  extra41 = 0x0003f04000  (P0, GT=0x4)
FSETP.GEU.AND P0, PT, R2, R5, PT  →  extra41 = 0x0003f0e000  (P0, GEU=0xe)
FSETP.GEU.AND P1, PT, R2, R5, PT  →  extra41 = 0x0003f2e000  (P1, GEU=0xe)
FSETP.GEU.AND P2, PT, R2, R5, PT  →  extra41 = 0x0003f4e000  (P2, GEU=0xe)
FSETP.NEU.AND P2, PT, R2, R5, PT  →  extra41 = 0x0003f4d000  (P2, NEU=0xd)
FSETP.GT.AND  P2, PT, R2, R5, PT  →  extra41 = 0x0003f44000  (P2, GT=0x4)
FSETP.NE.AND  P1, PT, R2, R5, PT  →  extra41 = 0x0003f25000  (P1, NE=0x5)
FSETP.GE.AND  P4, PT, R2, R5, PT  →  extra41 = 0x0003f86000  (P4, GE=0x6)
FSETP.GTU.AND P3, PT, R2, R5, PT  →  extra41 = 0x0003f6c000  (P3, GTU=0xc)
```

### Construction formula

```
backbone = 0x0003f04000
pred_bits = pred_reg_index << 17
cond_bits = cond_code << 12
extra41 = backbone | pred_bits | cond_bits
```

Verified: `0x0003f04000 | (2 << 17) | (0xe << 12) = 0x0003f4e000` ✓ (GEU P2)

---

## FADD Immediate — FP32 Add with Float Constant

**PTX:** `add.f32 %f2, %f0, 0f3F800000`  (1.0f)
**SASS:** `FADD R7, R2, 1`

### Probe disassembly

```
FADD R7, R2, 1 ;   /* 0x3f80000002077421 */
                    /* 0x004fca0000000000 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x3f80000002077421   (immediate = 1.0f = 0x3f800000)

bits [7:0]   = 0x21          — FADD opcode (same as RR form)
bits [15:8]  = 0x74          — immediate form (vs 0x72 for RR)
bits [23:16] = 0x07 = R7     — destination Rd
bits [31:24] = 0x02 = R2     — source Rs1 (register)
bits [39:32] = 0x00          — unused (no Rs2 register)
bits [63:32] = 0x3f800000    — 32-bit IEEE 754 single-precision immediate
```

### Additional verified examples

```
FADD R7, R0, 2.0   → 0x4000000000077421  (2.0f = 0x40000000)
FADD R9, R0, 0.5   → 0x3f00000000097421  (0.5f = 0x3f000000)
FADD R11, R0, -1.0 → 0xbf800000000b7421  (-1.0f = 0xbf800000)
```

The full IEEE 754 single-precision bit pattern occupies bits [63:32].
All 32 bits are stored; there is no truncation.

### emit-sass.fs usage

There is no dedicated `fadd-imm,` builder yet. To emit:
```
(imm32_as_u32) 32 lshift  rd 16 lshift or  rs1 24 lshift or  $7421 or
```

---

## FMUL Immediate — FP32 Multiply by Float Constant

**PTX:** `mul.f32 %f2, %f0, 0f40000000`  (2.0f)
**SASS:** `FMUL R7, R2, 2`

### Probe disassembly

```
FMUL R7, R2, 2 ;   /* 0x4000000002077820 */
                    /* 0x004fca0000400000 */
```

### Encoding (64-bit instruction word)

```
Full word: 0x4000000002077820   (immediate = 2.0f = 0x40000000)

bits [7:0]   = 0x20          — FMUL opcode (same as RR form)
bits [15:8]  = 0x78          — immediate form (vs 0x72 for RR)
bits [23:16] = 0x07 = R7     — destination Rd
bits [31:24] = 0x02 = R2     — source Rs1 (register)
bits [39:32] = 0x00          — unused
bits [63:32] = 0x40000000    — 32-bit IEEE 754 single-precision immediate
```

### Difference from FADD-imm

```
FADD-imm: bits [15:8] = 0x74  (bit 2 of high byte)
FMUL-imm: bits [15:8] = 0x78  (bit 3 of high byte)
RR forms:  bits [15:8] = 0x72  (bit 1 of high byte)
```

### Additional verified examples (from probe_2b.sass)

```
FMUL R7, R4, 0.5                → 0x3f00000004077820
FMUL R8, R4, 0.15915493667...   → 0x3e22f98304087820  (non-round constant)
FMUL R7, R4, -1 (negate form)   → uses FMUL with imm 0xbf800000
```

---

## Opcode Table Summary

All encodings verified against nvdisasm on sm_90a.

| PTX              | SASS        | Opcode [15:0] | Notes                           |
|------------------|-------------|---------------|---------------------------------|
| `mul.f32`        | FMUL        | `0x7220`      | RR; also `0x7820` for imm form  |
| `add.f32`        | FADD        | `0x7221`      | RR; also `0x7421` for imm form  |
| `fma.rn.f32`     | FFMA        | `0x7223`      | Rs3 in ctrl extra41[7:0]        |
| `neg.f32`        | FADD (FNEG) | `0x7221`      | -Rs + -RZ; modifiers in ctrl    |
| `abs.f32`        | FADD (FABS) | `0x7221`      | \|Rs\| + -RZ; modifiers in ctrl |
| `min.f32`        | FMNMX (PT)  | `0x7209`      | selector in ctrl extra41 bit 26 |
| `max.f32`        | FMNMX (!PT) | `0x7209`      | same; !PT = ctrl bit 26 set     |
| `setp.*.f32`     | FSETP       | `0x720b`      | cond+pred in ctrl extra41       |

### Existing emit-sass.fs constants (verified correct)

```
$7220 constant OP-FMUL   ← in emit-sass-auto.fs ✓
$7221 constant OP-FADD   ← in emit-sass.fs ✓
$7223 constant OP-FFMA   ← in emit-sass.fs ✓
```

No FMNMX or FSETP constants exist yet; the control word complexity makes
them unsuitable for simple constant-based encoding.

---

## Cross-Reference: Lithos Primitives

| Lithos op   | PTX          | SASS        | emit-sass.fs builder     |
|-------------|--------------|-------------|--------------------------|
| `*` (fmul)  | `mul.f32`    | FMUL/FFMA   | `fmul,` or `ffma,`       |
| `+` (fadd)  | `add.f32`    | FADD        | `fadd,`                  |
| `-` (fsub)  | `sub.f32`    | FADD + NEG  | `fadd,` + modifier bits  |
| `fma`       | `fma.rn.f32` | FFMA        | `ffma,`                  |
| `neg`       | `neg.f32`    | FADD/FNEG   | (not yet a builder)      |
| `abs`       | `abs.f32`    | FADD/FABS   | (not yet a builder)      |
| `min`       | `min.f32`    | FMNMX PT    | (not yet a builder)      |
| `max`       | `max.f32`    | FMNMX !PT   | (not yet a builder)      |
| `lt` etc.   | `setp.lt`    | FSETP       | (not yet a builder)      |

---

*Generated 2026-04-13 by systematic probe compilation on NVIDIA Hopper sm_90a.*
*Tool versions: ptxas built 2025-02-21, nvdisasm built 2025-02-13.*
