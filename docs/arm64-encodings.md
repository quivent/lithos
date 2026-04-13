# ARM64 Instruction Encodings for the Lithos Self-Hosting Compiler

Reference for the ARM64 emitter team. Every instruction the Lithos compiler
binary needs to emit, with exact 32-bit encoding templates.

All encodings are little-endian as written to the ELF `.text` segment.
Register fields: `Rd` = destination (bits 4:0), `Rn` = first source (bits 9:5),
`Rm` = second source (bits 20:16). `XZR`/`WZR` = register 31 when used as
a source; `SP` = register 31 when used by `ADD`/`SUB` (immediate) and
`LDR`/`STR` (base register).

---

## Register Convention

| Register | Role | Callee-saved? |
|----------|------|---------------|
| X0-X7 | Arguments / return values | No |
| X8 | Syscall number (Linux) / indirect result | No |
| X9-X15 | Temporaries | No |
| X16-X17 | Intra-procedure-call scratch (IP0/IP1) | No |
| X18 | Platform register (reserved on Linux) | -- |
| X19-X28 | Callee-saved registers | Yes |
| X29 | Frame pointer (FP) | Yes |
| X30 | Link register (LR) | Yes |
| SP (31) | Stack pointer (not a GPR in most contexts) | -- |
| XZR (31) | Zero register (reads as 0, writes discarded) | -- |

W-registers (W0-W30, WZR) are the low 32 bits of the corresponding X-register.
Operations on W-registers zero-extend the result into the full 64-bit register.

---

## 1. Integer Arithmetic

### ADD (shifted register, 64-bit)

`ADD Xd, Xn, Xm` -- emit `0x8B000000 | (Rm << 16) | (Rn << 5) | Rd`

```
31  30 29 28       24 23 22 21 20    16 15      10 9     5 4    0
 1   0  0  0 1 0 1 1  0  0  0  Rm[4:0]  imm6[5:0] Rn[4:0] Rd[4:0]
```

- `sf=1` (bit 31): 64-bit. Set to 0 for 32-bit (`ADD Wd, Wn, Wm` = `0x0B000000`).
- `shift` (bits 23:22): 00=LSL, 01=LSR, 10=ASR. `imm6` = shift amount.
- `ADD Xd, Xn, Xm, LSL #n` = `0x8B000000 | (Rm << 16) | (n << 10) | (Rn << 5) | Rd`

### ADD (immediate, 64-bit)

`ADD Xd, Xn, #imm12` -- emit `0x91000000 | (imm12 << 10) | (Rn << 5) | Rd`

```
31  30 29 28       24 23 22 21                10 9     5 4    0
 1   0  0  1 0 0 0 1  sh  0   imm12[11:0]      Rn[4:0] Rd[4:0]
```

- `sh` (bit 22): 0 = no shift, 1 = LSL #12.
- `ADD Xd, Xn, #imm12, LSL #12` = `0x91400000 | (imm12 << 10) | (Rn << 5) | Rd`
- 32-bit form: `0x11000000` base.

### SUB (shifted register, 64-bit)

`SUB Xd, Xn, Xm` -- emit `0xCB000000 | (Rm << 16) | (Rn << 5) | Rd`

```
31  30 29 28       24 23 22 21 20    16 15      10 9     5 4    0
 1   1  0  0 1 0 1 1  0  0  0  Rm[4:0]  imm6[5:0] Rn[4:0] Rd[4:0]
```

- 32-bit form: `0x4B000000`.
- With shift: same encoding, fill `shift` and `imm6`.

### SUB (immediate, 64-bit)

`SUB Xd, Xn, #imm12` -- emit `0xD1000000 | (imm12 << 10) | (Rn << 5) | Rd`

- 32-bit form: `0x51000000`.

### SUBS (register) -- sets flags

`SUBS Xd, Xn, Xm` -- emit `0xEB000000 | (Rm << 16) | (Rn << 5) | Rd`

- **CMP Xn, Xm** is an alias: `SUBS XZR, Xn, Xm` = `0xEB000000 | (Rm << 16) | (Rn << 5) | 0x1F`
- 32-bit: `0x6B000000`.

### SUBS (immediate)

`SUBS Xd, Xn, #imm12` -- emit `0xF1000000 | (imm12 << 10) | (Rn << 5) | Rd`

- **CMP Xn, #imm** = `0xF1000000 | (imm12 << 10) | (Rn << 5) | 0x1F`
- 32-bit: `0x71000000`.

### MUL (64-bit)

`MUL Xd, Xn, Xm` -- emit `0x9B007C00 | (Rm << 16) | (Rn << 5) | Rd`

```
31  30 29 28    24 23 21 20    16 15 14      10 9     5 4    0
 1   0  0  1 1 0 1 1 0 0 0 Rm[4:0] 0 Ra=11111  Rn[4:0] Rd[4:0]
```

- This is actually `MADD Xd, Xn, Xm, XZR` (Ra=31 means add zero).
- 32-bit: `0x1B007C00`.

### MADD (multiply-add, 64-bit)

`MADD Xd, Xn, Xm, Xa` -- emit `0x9B000000 | (Rm << 16) | (Ra << 10) | (Rn << 5) | Rd`

- Result = `Xn * Xm + Xa`.

### MSUB (multiply-subtract, 64-bit)

`MSUB Xd, Xn, Xm, Xa` -- emit `0x9B008000 | (Rm << 16) | (Ra << 10) | (Rn << 5) | Rd`

- Result = `Xa - Xn * Xm`. Used by the bootstrap for modulo: `sdiv` then `msub`.

### SDIV (64-bit)

`SDIV Xd, Xn, Xm` -- emit `0x9AC00C00 | (Rm << 16) | (Rn << 5) | Rd`

```
31  30 29 28    21 20    16 15      10 9     5 4    0
 1   0  0  1 1 0 1 0 1 1 0 0 0 0 0 0 Rm[4:0] 0 0 0 0 1 1 Rn[4:0] Rd[4:0]
```

- 32-bit: `0x1AC00C00`.

### UDIV (64-bit)

`UDIV Xd, Xn, Xm` -- emit `0x9AC00800 | (Rm << 16) | (Rn << 5) | Rd`

- 32-bit: `0x1AC00800`.

### NEG (alias)

`NEG Xd, Xm` = `SUB Xd, XZR, Xm` = `0xCB000000 | (Rm << 16) | (31 << 5) | Rd`

---

## 2. Logical / Shift

### AND (shifted register, 64-bit)

`AND Xd, Xn, Xm` -- emit `0x8A000000 | (Rm << 16) | (Rn << 5) | Rd`

```
31  30 29 28       24 23 22 21 20    16 15      10 9     5 4    0
 1   0  0  0 1 0 1 0  shift  0  Rm[4:0]  imm6      Rn[4:0] Rd[4:0]
```

- 32-bit: `0x0A000000`.

### ANDS (shifted register) -- sets flags

`ANDS Xd, Xn, Xm` -- emit `0xEA000000 | (Rm << 16) | (Rn << 5) | Rd`

- **TST Xn, Xm** = `ANDS XZR, Xn, Xm` = `0xEA000000 | (Rm << 16) | (Rn << 5) | 0x1F`

### AND (immediate, 64-bit)

`AND Xd, Xn, #bitmask` -- emit `0x92000000 | (bitmask_encoding << 10) | (Rn << 5) | Rd`

```
31  30 29 28       23 22 21        16 15      10 9     5 4    0
 1   0  0  1 0 0 1 0 0  N  immr[5:0]  imms[5:0] Rn[4:0] Rd[4:0]
```

The bitmask immediate uses the ARM64 logical immediate encoding scheme:
- `N`, `immr`, `imms` encode a repeating bit pattern.
- Common patterns:
  - `#~7` (align to 8): N=1, immr=0, imms=0x3C -> full encoding `0x9240F800` base.
  - `#0xFF` (byte mask): N=1, immr=0, imms=7 -> `0x92401C00` base.
  - `#0xFFFF`: N=1, immr=0, imms=0xF -> `0x92403C00` base.

For the compiler emitter, pre-compute a lookup table for commonly needed bitmask values. The general encoder is complex (see helper function below).

### ORR (shifted register, 64-bit)

`ORR Xd, Xn, Xm` -- emit `0xAA000000 | (Rm << 16) | (Rn << 5) | Rd`

- **MOV Xd, Xm** = `ORR Xd, XZR, Xm` = `0xAA000000 | (Rm << 16) | (31 << 5) | Rd`
- 32-bit: `0x2A000000`.

### EOR (shifted register, 64-bit)

`EOR Xd, Xn, Xm` -- emit `0xCA000000 | (Rm << 16) | (Rn << 5) | Rd`

- 32-bit: `0x4A000000`.

### MVN (bitwise NOT, alias)

`MVN Xd, Xm` = `ORN Xd, XZR, Xm` -- emit `0xAA200000 | (Rm << 16) | (31 << 5) | Rd`

### ORN (OR NOT, shifted register, 64-bit)

`ORN Xd, Xn, Xm` -- emit `0xAA200000 | (Rm << 16) | (Rn << 5) | Rd`

### LSL (register, 64-bit)

`LSL Xd, Xn, Xm` = `LSLV Xd, Xn, Xm` -- emit `0x9AC02000 | (Rm << 16) | (Rn << 5) | Rd`

- 32-bit: `0x1AC02000`.

### LSR (register, 64-bit)

`LSR Xd, Xn, Xm` = `LSRV Xd, Xn, Xm` -- emit `0x9AC02400 | (Rm << 16) | (Rn << 5) | Rd`

- 32-bit: `0x1AC02400`.

### ASR (register, 64-bit)

`ASR Xd, Xn, Xm` = `ASRV Xd, Xn, Xm` -- emit `0x9AC02800 | (Rm << 16) | (Rn << 5) | Rd`

- 32-bit: `0x1AC02800`.

### LSL / LSR / ASR (immediate, via UBFM/SBFM aliases)

`LSL Xd, Xn, #shift` = `UBFM Xd, Xn, #(64-shift), #(63-shift)`
- emit `0xD3400000 | ((64-shift) << 16) | ((63-shift) << 10) | (Rn << 5) | Rd`

`LSR Xd, Xn, #shift` = `UBFM Xd, Xn, #shift, #63`
- emit `0xD340FC00 | (shift << 16) | (Rn << 5) | Rd`

`ASR Xd, Xn, #shift` = `SBFM Xd, Xn, #shift, #63`
- emit `0x9340FC00 | (shift << 16) | (Rn << 5) | Rd`

32-bit variants: replace `0xD3` with `0x53`, `0x93` with `0x13`, and use modulo 32.

---

## 3. Load / Store

### LDR (immediate offset, unsigned, 64-bit)

`LDR Xt, [Xn, #imm]` -- emit `0xF9400000 | ((imm/8) << 10) | (Rn << 5) | Rt`

```
31 30 29 28     24 23 22 21          10 9     5 4    0
 1  1  1  1 1 0 0 1 0  1   imm12[11:0] Rn[4:0] Rt[4:0]
```

- `imm` must be a multiple of 8, range 0..32760.
- 32-bit (`LDR Wt`): `0xB9400000 | ((imm/4) << 10) | ...`

### LDR (register offset, 64-bit)

`LDR Xt, [Xn, Xm]` -- emit `0xF8606800 | (Rm << 16) | (Rn << 5) | Rt`

```
31 30 29 28     24 23 22 21 20    16 15 14 13 12 11 10 9     5 4    0
 1  1  1  1 1 0 0 0 0  1  1  Rm     0  1  1  S  1  0  Rn      Rt
```

- Option=011 (LSL), S=0 (no shift). For `LSL #3`: set S=1 -> `0xF8607800`.
- `LDR Xt, [Xn, Xm, LSL #3]` = `0xF8607800 | (Rm << 16) | (Rn << 5) | Rt`
  (Used in bootstrap: `ldr x22, [x24, x22, lsl #3]` for PICK.)

### LDR (pre-index, 64-bit)

`LDR Xt, [Xn, #simm9]!` -- emit `0xF8400C00 | (simm9 << 12) | (Rn << 5) | Rt`

```
31 30 29 28     24 23 22 21 20      12 11 10 9     5 4    0
 1  1  1  1 1 0 0 0 0  1  0  simm9[8:0] 1  1  Rn[4:0] Rt[4:0]
```

- `simm9` is a 9-bit signed immediate (-256 to +255).
- Encode as: `(simm9 & 0x1FF) << 12`.

### LDR (post-index, 64-bit)

`LDR Xt, [Xn], #simm9` -- emit `0xF8400400 | ((simm9 & 0x1FF) << 12) | (Rn << 5) | Rt`

```
31 30 29 28     24 23 22 21 20      12 11 10 9     5 4    0
 1  1  1  1 1 0 0 0 0  1  0  simm9[8:0] 0  1  Rn[4:0] Rt[4:0]
```

- Heavily used in bootstrap: `ldr x22, [x24], #8` = post-increment pop.

### STR (immediate offset, unsigned, 64-bit)

`STR Xt, [Xn, #imm]` -- emit `0xF9000000 | ((imm/8) << 10) | (Rn << 5) | Rt`

- `imm` must be a multiple of 8, range 0..32760.
- 32-bit: `0xB9000000 | ((imm/4) << 10) | ...`

### STR (pre-index, 64-bit)

`STR Xt, [Xn, #simm9]!` -- emit `0xF8000C00 | ((simm9 & 0x1FF) << 12) | (Rn << 5) | Rt`

- Bootstrap pattern: `str x22, [x24, #-8]!` = push onto data stack.
  - simm9 = -8 = 0x1F8. Encoded: `0xF81F8C00 | (24 << 5) | 22` = `0xF81F8316`.

### STR (post-index, 64-bit)

`STR Xt, [Xn], #simm9` -- emit `0xF8000400 | ((simm9 & 0x1FF) << 12) | (Rn << 5) | Rt`

### LDRB (byte load, unsigned offset)

`LDRB Wt, [Xn, #imm12]` -- emit `0x39400000 | (imm12 << 10) | (Rn << 5) | Rt`

- `imm12` range 0..4095 (no scaling, byte granularity).

### LDRB (post-index)

`LDRB Wt, [Xn], #simm9` -- emit `0x38400400 | ((simm9 & 0x1FF) << 12) | (Rn << 5) | Rt`

### STRB (byte store, unsigned offset)

`STRB Wt, [Xn, #imm12]` -- emit `0x39000000 | (imm12 << 10) | (Rn << 5) | Rt`

### STRB (post-index)

`STRB Wt, [Xn], #simm9` -- emit `0x38000400 | ((simm9 & 0x1FF) << 12) | (Rn << 5) | Rt`

### LDRH (halfword load, unsigned offset)

`LDRH Wt, [Xn, #imm]` -- emit `0x79400000 | ((imm/2) << 10) | (Rn << 5) | Rt`

- `imm` must be a multiple of 2, range 0..8190.

### STRH (halfword store, unsigned offset)

`STRH Wt, [Xn, #imm]` -- emit `0x79000000 | ((imm/2) << 10) | (Rn << 5) | Rt`

### LDP (load pair, 64-bit)

`LDP Xt1, Xt2, [Xn, #simm7*8]` -- emit `0xA9400000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

```
31 30 29 28 27 26 25 24 23 22 21     15 14    10 9     5 4    0
 1  0  1  0  1  0  0  1  0  1 simm7[6:0] Rt2[4:0] Rn[4:0] Rt1[4:0]
```

- `simm7` is signed, range -64 to +63. Actual byte offset = simm7 * 8.

### LDP (post-index, 64-bit)

`LDP Xt1, Xt2, [Xn], #simm7*8` -- emit `0xA8C00000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

- Launcher pattern: `ldp x26, x30, [sp], #16` -> simm7 = 2 (16/8).

### LDP (pre-index, 64-bit)

`LDP Xt1, Xt2, [Xn, #simm7*8]!` -- emit `0xA9C00000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

### STP (store pair, 64-bit)

`STP Xt1, Xt2, [Xn, #simm7*8]` -- emit `0xA9000000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

### STP (pre-index, 64-bit)

`STP Xt1, Xt2, [Xn, #simm7*8]!` -- emit `0xA9800000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

- Launcher prologue: `stp x29, x30, [sp, #-16]!` -> simm7 = -2 (-16/8).
  - `0xA9800000 | (0x7E << 15) | (30 << 10) | (31 << 5) | 29` = `0xA9BF7BFD`.

### STP (post-index, 64-bit)

`STP Xt1, Xt2, [Xn], #simm7*8` -- emit `0xA8800000 | ((simm7 & 0x7F) << 15) | (Rt2 << 10) | (Rn << 5) | Rt1`

### 32-bit pair variants

- `LDP Wt1, Wt2, [Xn, #simm7*4]`: `0x29400000` base (scale factor = 4).
- `STP Wt1, Wt2, [Xn, #simm7*4]`: `0x29000000` base.

---

## 4. Branch

### B (unconditional)

`B label` -- emit `0x14000000 | (imm26 & 0x03FFFFFF)`

```
31 30      26 25                                  0
 0  0 0 1 0 1  imm26[25:0]
```

- `imm26` = (target - current_pc) / 4. Signed, range +/-128 MB.

### BL (branch with link / call)

`BL label` -- emit `0x94000000 | (imm26 & 0x03FFFFFF)`

- Saves return address in X30 (LR).

### B.cond (conditional branch)

`B.cond label` -- emit `0x54000000 | ((imm19 & 0x7FFFF) << 5) | cond`

```
31 30      25 24 23                  5 4 3    0
 0  1 0 1 0 1 0 0 imm19[18:0]         0 cond[3:0]
```

- `imm19` = (target - current_pc) / 4. Signed, range +/-1 MB.

Condition codes:

| Mnemonic | cond | Meaning |
|----------|------|---------|
| EQ | 0x0 | Equal (Z=1) |
| NE | 0x1 | Not equal (Z=0) |
| CS/HS | 0x2 | Carry set / unsigned >= |
| CC/LO | 0x3 | Carry clear / unsigned < |
| MI | 0x4 | Minus / negative |
| PL | 0x5 | Plus / positive or zero |
| VS | 0x6 | Overflow |
| VC | 0x7 | No overflow |
| HI | 0x8 | Unsigned > |
| LS | 0x9 | Unsigned <= |
| GE | 0xA | Signed >= |
| LT | 0xB | Signed < |
| GT | 0xC | Signed > |
| LE | 0xD | Signed <= |
| AL | 0xE | Always |

- `B.EQ label` = `0x54000000 | (imm19 << 5) | 0x0`
- `B.NE label` = `0x54000000 | (imm19 << 5) | 0x1`
- `B.LT label` = `0x54000000 | (imm19 << 5) | 0xB`
- `B.GE label` = `0x54000000 | (imm19 << 5) | 0xA`
- `B.GT label` = `0x54000000 | (imm19 << 5) | 0xC`
- `B.LE label` = `0x54000000 | (imm19 << 5) | 0xD`

### BR (branch to register)

`BR Xn` -- emit `0xD61F0000 | (Rn << 5)`

```
31                  10 9     5 4    0
 1 1 0 1 0 1 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 Rn[4:0] 0 0 0 0 0
```

### BLR (branch with link to register)

`BLR Xn` -- emit `0xD63F0000 | (Rn << 5)`

- Used in launcher for all CUDA driver API calls: `bl cuInit`, etc.

### RET

`RET` (return to X30) -- emit `0xD65F03C0`

- Actually `RET X30`. For arbitrary register: `0xD65F0000 | (Rn << 5)`.

### CBZ (compare and branch if zero)

`CBZ Xt, label` -- emit `0xB4000000 | ((imm19 & 0x7FFFF) << 5) | Rt`

```
31  30      25 24 23                  5 4    0
sf  0 1 1 0 1 0 0 imm19[18:0]          Rt[4:0]
```

- `sf=1` (bit 31): 64-bit. `sf=0`: 32-bit (`0x34000000`).
- Range: +/-1 MB.
- Launcher pattern: `cbnz x0, cuda_error` to check CUDA return codes.

### CBNZ (compare and branch if not zero)

`CBNZ Xt, label` -- emit `0xB5000000 | ((imm19 & 0x7FFFF) << 5) | Rt`

- 32-bit form: `0x35000000`.

### TBZ (test bit and branch if zero)

`TBZ Xt, #bit, label` -- emit `0x36000000 | (b5 << 31) | (b40 << 19) | ((imm14 & 0x3FFF) << 5) | Rt`

```
31   30      25 24 23    19 18                5 4    0
b5   0 1 1 0 1 1 0 b40[4:0] imm14[13:0]        Rt[4:0]
```

- `b5` = bit 5 of the bit number (0 for bits 0-31, 1 for bits 32-63).
- `b40` = bits 4:0 of the bit number.
- Range: +/-32 KB.

### TBNZ (test bit and branch if not zero)

`TBNZ Xt, #bit, label` -- emit `0x37000000 | (b5 << 31) | (b40 << 19) | ((imm14 & 0x3FFF) << 5) | Rt`

---

## 5. Compare and Conditional Select

### CMP (alias for SUBS with Rd=XZR)

`CMP Xn, Xm` = `SUBS XZR, Xn, Xm` -- emit `0xEB00001F | (Rm << 16) | (Rn << 5)`
`CMP Xn, #imm12` = `SUBS XZR, Xn, #imm12` -- emit `0xF100001F | (imm12 << 10) | (Rn << 5)`

- 32-bit: `0x6B00001F` (register), `0x7100001F` (immediate).

### TST (alias for ANDS with Rd=XZR)

`TST Xn, Xm` = `ANDS XZR, Xn, Xm` -- emit `0xEA00001F | (Rm << 16) | (Rn << 5)`

### CSEL (conditional select)

`CSEL Xd, Xn, Xm, cond` -- emit `0x9A800000 | (Rm << 16) | (cond << 12) | (Rn << 5) | Rd`

```
31  30 29 28       21 20    16 15 14   12 11 10 9     5 4    0
 1   0  0  1 1 0 1 0 1 0 0 Rm[4:0] cond[3:0] 0  0 Rn[4:0] Rd[4:0]
```

- `Xd = cond_true ? Xn : Xm`
- 32-bit: `0x1A800000`.

### CSET (conditional set, alias)

`CSET Xd, cond` = `CSINC Xd, XZR, XZR, invert(cond)`
- `CSINC Xd, Xn, Xm, cond` -- emit `0x9A800400 | (Rm << 16) | (cond << 12) | (Rn << 5) | Rd`
- For CSET: `0x9A9F07E0 | (invert(cond) << 12) | Rd`
  - Invert cond: flip bit 0 (EQ->NE, LT->GE, etc.)
  - `CSET Xd, EQ` = `CSINC Xd, XZR, XZR, NE` = `0x9A9F17E0 | Rd`

### CSINC (conditional select increment)

`CSINC Xd, Xn, Xm, cond` -- emit `0x9A800400 | (Rm << 16) | (cond << 12) | (Rn << 5) | Rd`

- `Xd = cond_true ? Xn : (Xm + 1)`

---

## 6. Move

### MOV (register, alias for ORR)

`MOV Xd, Xm` = `ORR Xd, XZR, Xm` -- emit `0xAA0003E0 | (Rm << 16) | Rd`

- 32-bit: `0x2A0003E0 | (Rm << 16) | Rd`.

### MOVZ (move wide with zero)

`MOVZ Xd, #imm16` -- emit `0xD2800000 | (imm16 << 5) | Rd`

```
31  30 29 28       23 22 21 20                  5 4    0
sf   1  0  1 0 0 1 0 1  hw     imm16[15:0]       Rd[4:0]
```

- `hw` (bits 22:21): shift amount = hw * 16. 00=bits 0:15, 01=bits 16:31, 10=bits 32:47, 11=bits 48:63.
- `MOVZ Xd, #imm16, LSL #16` = `0xD2A00000 | (imm16 << 5) | Rd`
- `MOVZ Xd, #imm16, LSL #32` = `0xD2C00000 | (imm16 << 5) | Rd`
- `MOVZ Xd, #imm16, LSL #48` = `0xD2E00000 | (imm16 << 5) | Rd`
- 32-bit (sf=0): `0x52800000` (only hw=00 and hw=01 valid).

### MOVK (move wide with keep)

`MOVK Xd, #imm16` -- emit `0xF2800000 | (imm16 << 5) | Rd`

- `MOVK Xd, #imm16, LSL #16` = `0xF2A00000 | (imm16 << 5) | Rd`
- `MOVK Xd, #imm16, LSL #32` = `0xF2C00000 | (imm16 << 5) | Rd`
- `MOVK Xd, #imm16, LSL #48` = `0xF2E00000 | (imm16 << 5) | Rd`

**Building a 64-bit constant** (e.g., loading address 0x0000_DEAD_BEEF_CAFE):

```
MOVZ Xd, #0xCAFE                // bits 15:0
MOVK Xd, #0xBEEF, LSL #16      // bits 31:16
MOVK Xd, #0xDEAD, LSL #32      // bits 47:32
                                 // bits 63:48 are 0 from MOVZ
```

Emit sequence:
```
0xD280_0000 | (0xCAFE << 5) | Rd      // MOVZ Xd, #0xCAFE
0xF2A0_0000 | (0xBEEF << 5) | Rd      // MOVK Xd, #0xBEEF, LSL #16
0xF2C0_0000 | (0xDEAD << 5) | Rd      // MOVK Xd, #0xDEAD, LSL #32
```

Optimize: skip MOVK for zero chunks; use MOVZ at the highest non-zero chunk.

### MOVN (move wide with NOT)

`MOVN Xd, #imm16` -- emit `0x92800000 | (imm16 << 5) | Rd`

- Result = `~(imm16 << (hw * 16))`.
- `MOV Xd, #-1` = `MOVN Xd, #0` = `0x92800000 | Rd`.
- `MOV Xd, #-100` = `MOVN Xd, #99` = `0x92800000 | (99 << 5) | Rd`.
- Useful for small negative constants without needing SUB from XZR.

### MOV (to/from SP, alias for ADD)

`MOV Xd, SP` = `ADD Xd, SP, #0` = `0x910003E0 | Rd`
`MOV SP, Xn` = `ADD SP, Xn, #0` = `0x9100001F | (Rn << 5)`

---

## 7. Address Generation (PC-relative)

### ADR (PC-relative, +/-1 MB)

`ADR Xd, label` -- emit `(immlo << 29) | 0x10000000 | (immhi << 5) | Rd`

```
31  30 29 28       24 23                  5 4    0
op  immlo  1 0 0 0 0  immhi[18:0]          Rd[4:0]
```

- `op=0` for ADR. `imm21` = target - PC. Split: `immlo` = imm21[1:0], `immhi` = imm21[20:2].
- Range: +/-1 MB, byte resolution.

### ADRP (PC-relative, page, +/-4 GB)

`ADRP Xd, label` -- emit `(immlo << 29) | 0x90000000 | (immhi << 5) | Rd`

- `op=1` for ADRP. `imm21` = (page_of_target - page_of_PC) / 4096.
- Result = `(PC & ~0xFFF) + (imm21 << 12)`.
- Always paired with `ADD Xd, Xd, :lo12:symbol` to get the full address.

**Standard ADRP+ADD pattern** (used throughout launcher and bootstrap):

```
adrp    x0, symbol          // load page address
add     x0, x0, :lo12:symbol  // add page offset (low 12 bits)
```

Emit:
```
ADRP with relocation        // linker fills imm21
ADD  x0, x0, #(symbol & 0xFFF)  // 0x91000000 | (lo12 << 10) | (Rn << 5) | Rd
```

For a self-hosting compiler emitting position-dependent code at a known base
address, you can compute the offsets directly instead of using relocations.

---

## 8. System Instructions

### SVC (supervisor call / syscall)

`SVC #0` -- emit `0xD4000001`

```
31                  5 4    0
 1 1 0 1 0 1 0 0 0 0 0 imm16[15:0] 0 0 0 0 1
```

- `SVC #imm16`: `0xD4000001 | (imm16 << 5)`. Linux always uses `SVC #0`.

### MRS (read system register)

`MRS Xt, <sysreg>` -- emit `0xD5300000 | (sysreg_encoding << 5) | Rt`

Common system registers:

| Register | op0 | op1 | CRn | CRm | op2 | Encoding (bits 19:5) | Use |
|----------|-----|-----|-----|-----|-----|---------------------|-----|
| CNTVCT_EL0 | 3 | 3 | 14 | 0 | 2 | 0xDF02 | Virtual timer count |
| CNTFRQ_EL0 | 3 | 3 | 14 | 0 | 0 | 0xDF00 | Timer frequency |
| FPCR | 3 | 3 | 4 | 4 | 0 | 0xDA20 | FP control |
| FPSR | 3 | 3 | 4 | 4 | 1 | 0xDA21 | FP status |
| CurrentEL | 3 | 0 | 4 | 2 | 2 | 0xC212 | Current exception level |

- `MRS X0, CNTVCT_EL0` = `0xD53BE040` (for cycle counting / timestamps).

### MSR (write system register)

`MSR <sysreg>, Xt` -- emit `0xD5100000 | (sysreg_encoding << 5) | Rt`

### DSB (data synchronization barrier)

`DSB SY` -- emit `0xD5033F9F`

| Variant | CRm | Encoding |
|---------|-----|----------|
| DSB SY | 0xF | `0xD5033F9F` |
| DSB ISH | 0xB | `0xD5033B9F` |
| DSB ISHST | 0xA | `0xD5033A9F` |

### DMB (data memory barrier)

`DMB SY` -- emit `0xD5033FBF`
`DMB ISH` -- emit `0xD5033BBF`

### ISB (instruction synchronization barrier)

`ISB` -- emit `0xD5033FDF`

- Required after writing code to memory (self-modifying / JIT) before executing it.

---

## 9. Linux ARM64 Syscall Numbers

Syscall convention: number in X8, args in X0-X5, return in X0. Use `SVC #0`.

| Syscall | Number | Signature |
|---------|--------|-----------|
| ioctl | 29 | ioctl(fd, request, ...) |
| openat | 56 | openat(dirfd, path, flags, mode) |
| close | 57 | close(fd) |
| lseek | 62 | lseek(fd, offset, whence) |
| read | 63 | read(fd, buf, count) |
| write | 64 | write(fd, buf, count) |
| pread64 | 67 | pread64(fd, buf, count, offset) |
| fstat | 80 | fstat(fd, statbuf) |
| exit | 93 | exit(status) |
| exit_group | 94 | exit_group(status) |
| brk | 214 | brk(addr) |
| munmap | 215 | munmap(addr, length) |
| mmap | 222 | mmap(addr, length, prot, flags, fd, offset) |
| mprotect | 226 | mprotect(addr, len, prot) |
| madvise | 233 | madvise(addr, length, advice) |

Key constants:

| Name | Value | Notes |
|------|-------|-------|
| AT_FDCWD | -100 | Use with openat for relative paths |
| O_RDONLY | 0 | |
| O_WRONLY | 1 | |
| O_RDWR | 2 | |
| O_CREAT | 64 | |
| O_TRUNC | 512 | |
| O_CREAT\|O_WRONLY\|O_TRUNC | 577 | Used by elf-wrap.fs |
| SEEK_SET | 0 | |
| SEEK_CUR | 1 | |
| SEEK_END | 2 | |
| PROT_READ | 1 | |
| PROT_WRITE | 2 | |
| PROT_EXEC | 4 | |
| MAP_PRIVATE | 2 | |
| MAP_ANONYMOUS | 0x20 | |
| MADV_WILLNEED | 3 | |

---

## 10. ELF Header Emission

The self-hosting binary must emit ELF64 ARM64 executables (e_machine = 183 / EM_AARCH64).
See `/home/ubuntu/lithos/compiler/arm64-wrap.fs` (aliased as `elf-wrap.fs`) for the
existing layout:

- ET_EXEC, single PT_LOAD segment at `0x400000`.
- ELF header = 64 bytes, program header = 56 bytes, then `.text` payload.
- Entry point = `0x400000 + 64 + 56 = 0x400078`.
- Flags: PF_R | PF_X (5) for code segment.

For a writable data segment, add a second PT_LOAD with PF_R | PF_W (6).

---

## 11. Quick Encoding Reference (Emitter Lookup Table)

Copy-paste table for the emitter. Each line: what to emit, base opcode, field packing.

```
// ---- Arithmetic (64-bit X registers) ----
ADD  Xd, Xn, Xm          0x8B000000 | (Rm<<16) | (Rn<<5) | Rd
ADD  Xd, Xn, #imm12      0x91000000 | (imm12<<10) | (Rn<<5) | Rd
SUB  Xd, Xn, Xm          0xCB000000 | (Rm<<16) | (Rn<<5) | Rd
SUB  Xd, Xn, #imm12      0xD1000000 | (imm12<<10) | (Rn<<5) | Rd
SUBS Xd, Xn, Xm          0xEB000000 | (Rm<<16) | (Rn<<5) | Rd
SUBS Xd, Xn, #imm12      0xF1000000 | (imm12<<10) | (Rn<<5) | Rd
MUL  Xd, Xn, Xm          0x9B007C00 | (Rm<<16) | (Rn<<5) | Rd
MADD Xd, Xn, Xm, Xa      0x9B000000 | (Rm<<16) | (Ra<<10) | (Rn<<5) | Rd
MSUB Xd, Xn, Xm, Xa      0x9B008000 | (Rm<<16) | (Ra<<10) | (Rn<<5) | Rd
SDIV Xd, Xn, Xm          0x9AC00C00 | (Rm<<16) | (Rn<<5) | Rd
UDIV Xd, Xn, Xm          0x9AC00800 | (Rm<<16) | (Rn<<5) | Rd
NEG  Xd, Xm               0xCB0003E0 | (Rm<<16) | Rd

// ---- Arithmetic (32-bit W registers) ----
ADD  Wd, Wn, Wm          0x0B000000 | (Rm<<16) | (Rn<<5) | Rd
ADD  Wd, Wn, #imm12      0x11000000 | (imm12<<10) | (Rn<<5) | Rd
SUB  Wd, Wn, Wm          0x4B000000 | (Rm<<16) | (Rn<<5) | Rd
SUB  Wd, Wn, #imm12      0x51000000 | (imm12<<10) | (Rn<<5) | Rd
SUBS Wd, Wn, Wm          0x6B000000 | (Rm<<16) | (Rn<<5) | Rd
SUBS Wd, Wn, #imm12      0x71000000 | (imm12<<10) | (Rn<<5) | Rd
MUL  Wd, Wn, Wm          0x1B007C00 | (Rm<<16) | (Rn<<5) | Rd
SDIV Wd, Wn, Wm          0x1AC00C00 | (Rm<<16) | (Rn<<5) | Rd
UDIV Wd, Wn, Wm          0x1AC00800 | (Rm<<16) | (Rn<<5) | Rd

// ---- Logical (64-bit) ----
AND  Xd, Xn, Xm          0x8A000000 | (Rm<<16) | (Rn<<5) | Rd
ORR  Xd, Xn, Xm          0xAA000000 | (Rm<<16) | (Rn<<5) | Rd
EOR  Xd, Xn, Xm          0xCA000000 | (Rm<<16) | (Rn<<5) | Rd
ANDS Xd, Xn, Xm          0xEA000000 | (Rm<<16) | (Rn<<5) | Rd
ORN  Xd, Xn, Xm          0xAA200000 | (Rm<<16) | (Rn<<5) | Rd
MVN  Xd, Xm               0xAA2003E0 | (Rm<<16) | Rd

// ---- Logical (32-bit) ----
AND  Wd, Wn, Wm          0x0A000000 | (Rm<<16) | (Rn<<5) | Rd
ORR  Wd, Wn, Wm          0x2A000000 | (Rm<<16) | (Rn<<5) | Rd
EOR  Wd, Wn, Wm          0x4A000000 | (Rm<<16) | (Rn<<5) | Rd

// ---- Shift (register, 64-bit) ----
LSL  Xd, Xn, Xm          0x9AC02000 | (Rm<<16) | (Rn<<5) | Rd
LSR  Xd, Xn, Xm          0x9AC02400 | (Rm<<16) | (Rn<<5) | Rd
ASR  Xd, Xn, Xm          0x9AC02800 | (Rm<<16) | (Rn<<5) | Rd

// ---- Shift (register, 32-bit) ----
LSL  Wd, Wn, Wm          0x1AC02000 | (Rm<<16) | (Rn<<5) | Rd
LSR  Wd, Wn, Wm          0x1AC02400 | (Rm<<16) | (Rn<<5) | Rd
ASR  Wd, Wn, Wm          0x1AC02800 | (Rm<<16) | (Rn<<5) | Rd

// ---- Shift (immediate, 64-bit) ----
LSL  Xd, Xn, #n          0xD3400000 | ((64-n)<<16) | ((63-n)<<10) | (Rn<<5) | Rd
LSR  Xd, Xn, #n          0xD340FC00 | (n<<16) | (Rn<<5) | Rd
ASR  Xd, Xn, #n          0x9340FC00 | (n<<16) | (Rn<<5) | Rd

// ---- Move ----
MOV  Xd, Xm               0xAA0003E0 | (Rm<<16) | Rd
MOV  Wd, Wm               0x2A0003E0 | (Rm<<16) | Rd
MOVZ Xd, #imm16           0xD2800000 | (imm16<<5) | Rd
MOVZ Xd, #imm16, LSL#16  0xD2A00000 | (imm16<<5) | Rd
MOVZ Xd, #imm16, LSL#32  0xD2C00000 | (imm16<<5) | Rd
MOVZ Xd, #imm16, LSL#48  0xD2E00000 | (imm16<<5) | Rd
MOVK Xd, #imm16           0xF2800000 | (imm16<<5) | Rd
MOVK Xd, #imm16, LSL#16  0xF2A00000 | (imm16<<5) | Rd
MOVK Xd, #imm16, LSL#32  0xF2C00000 | (imm16<<5) | Rd
MOVK Xd, #imm16, LSL#48  0xF2E00000 | (imm16<<5) | Rd
MOVN Xd, #imm16           0x92800000 | (imm16<<5) | Rd
MOVZ Wd, #imm16           0x52800000 | (imm16<<5) | Rd
MOVZ Wd, #imm16, LSL#16  0x52A00000 | (imm16<<5) | Rd
MOVK Wd, #imm16           0x72800000 | (imm16<<5) | Rd
MOVK Wd, #imm16, LSL#16  0x72A00000 | (imm16<<5) | Rd

// ---- Load (64-bit) ----
LDR  Xt, [Xn, #imm]      0xF9400000 | ((imm/8)<<10) | (Rn<<5) | Rt
LDR  Xt, [Xn, Xm]        0xF8606800 | (Rm<<16) | (Rn<<5) | Rt
LDR  Xt, [Xn, Xm, LSL#3] 0xF8607800 | (Rm<<16) | (Rn<<5) | Rt
LDR  Xt, [Xn, #s9]!      0xF8400C00 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt
LDR  Xt, [Xn], #s9       0xF8400400 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt

// ---- Load (32-bit) ----
LDR  Wt, [Xn, #imm]      0xB9400000 | ((imm/4)<<10) | (Rn<<5) | Rt
LDR  Wt, [Xn], #s9       0xB8400400 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt

// ---- Load byte/half ----
LDRB Wt, [Xn, #imm12]    0x39400000 | (imm12<<10) | (Rn<<5) | Rt
LDRB Wt, [Xn], #s9       0x38400400 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt
LDRH Wt, [Xn, #imm]      0x79400000 | ((imm/2)<<10) | (Rn<<5) | Rt

// ---- Store (64-bit) ----
STR  Xt, [Xn, #imm]      0xF9000000 | ((imm/8)<<10) | (Rn<<5) | Rt
STR  Xt, [Xn, #s9]!      0xF8000C00 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt
STR  Xt, [Xn], #s9       0xF8000400 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt

// ---- Store (32-bit) ----
STR  Wt, [Xn, #imm]      0xB9000000 | ((imm/4)<<10) | (Rn<<5) | Rt

// ---- Store byte/half ----
STRB Wt, [Xn, #imm12]    0x39000000 | (imm12<<10) | (Rn<<5) | Rt
STRB Wt, [Xn], #s9       0x38000400 | ((s9&0x1FF)<<12) | (Rn<<5) | Rt
STRH Wt, [Xn, #imm]      0x79000000 | ((imm/2)<<10) | (Rn<<5) | Rt

// ---- Load/Store pair (64-bit) ----
LDP  Xt1,Xt2,[Xn,#s7*8]  0xA9400000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1
LDP  Xt1,Xt2,[Xn],#s7*8  0xA8C00000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1
LDP  Xt1,Xt2,[Xn,#s7*8]! 0xA9C00000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1
STP  Xt1,Xt2,[Xn,#s7*8]  0xA9000000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1
STP  Xt1,Xt2,[Xn,#s7*8]! 0xA9800000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1
STP  Xt1,Xt2,[Xn],#s7*8  0xA8800000 | ((s7&0x7F)<<15) | (Rt2<<10) | (Rn<<5) | Rt1

// ---- Branch ----
B    label                 0x14000000 | (imm26 & 0x03FFFFFF)
BL   label                 0x94000000 | (imm26 & 0x03FFFFFF)
BR   Xn                    0xD61F0000 | (Rn<<5)
BLR  Xn                    0xD63F0000 | (Rn<<5)
RET                        0xD65F03C0
RET  Xn                    0xD65F0000 | (Rn<<5)
B.EQ label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0x0
B.NE label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0x1
B.CS label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0x2
B.CC label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0x3
B.GE label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0xA
B.LT label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0xB
B.GT label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0xC
B.LE label                 0x54000000 | ((imm19&0x7FFFF)<<5) | 0xD
CBZ  Xt, label             0xB4000000 | ((imm19&0x7FFFF)<<5) | Rt
CBNZ Xt, label             0xB5000000 | ((imm19&0x7FFFF)<<5) | Rt
CBZ  Wt, label             0x34000000 | ((imm19&0x7FFFF)<<5) | Rt
CBNZ Wt, label             0x35000000 | ((imm19&0x7FFFF)<<5) | Rt
TBZ  Xt, #bit, label      0x36000000 | (b5<<31) | (b40<<19) | ((imm14&0x3FFF)<<5) | Rt
TBNZ Xt, #bit, label      0x37000000 | (b5<<31) | (b40<<19) | ((imm14&0x3FFF)<<5) | Rt

// ---- Compare / Conditional ----
CMP  Xn, Xm               0xEB00001F | (Rm<<16) | (Rn<<5)
CMP  Xn, #imm12           0xF100001F | (imm12<<10) | (Rn<<5)
CMP  Wn, Wm               0x6B00001F | (Rm<<16) | (Rn<<5)
CMP  Wn, #imm12           0x7100001F | (imm12<<10) | (Rn<<5)
TST  Xn, Xm               0xEA00001F | (Rm<<16) | (Rn<<5)
CSEL Xd, Xn, Xm, cond    0x9A800000 | (Rm<<16) | (cond<<12) | (Rn<<5) | Rd
CSEL Wd, Wn, Wm, cond    0x1A800000 | (Rm<<16) | (cond<<12) | (Rn<<5) | Rd
CSINC Xd, Xn, Xm, cond   0x9A800400 | (Rm<<16) | (cond<<12) | (Rn<<5) | Rd
CSET Xd, cond              0x9A9F07E0 | (inv_cond<<12) | Rd

// ---- Address generation ----
ADR  Xd, label             (immlo<<29) | 0x10000000 | (immhi<<5) | Rd
ADRP Xd, label             (immlo<<29) | 0x90000000 | (immhi<<5) | Rd

// ---- System ----
SVC  #0                    0xD4000001
SVC  #imm16               0xD4000001 | (imm16<<5)
MRS  Xt, CNTVCT_EL0       0xD53BE040 | Rt
DSB  SY                    0xD5033F9F
DSB  ISH                   0xD5033B9F
DMB  SY                    0xD5033FBF
DMB  ISH                   0xD5033BBF
ISB                        0xD5033FDF
```

---

## 12. Emitter Helper: Load 64-bit Immediate

The most common multi-instruction sequence. Given a 64-bit value `v`,
emit the minimal MOVZ/MOVK sequence:

```
emit_mov_imm64(Rd, v):
    chunks[0] = v & 0xFFFF
    chunks[1] = (v >> 16) & 0xFFFF
    chunks[2] = (v >> 32) & 0xFFFF
    chunks[3] = (v >> 48) & 0xFFFF

    // Find first non-zero chunk for MOVZ (or chunk 0 if all zero)
    first = lowest i where chunks[i] != 0 (default 0)

    emit MOVZ Xd, #chunks[first], LSL #(first*16)
    for i = first+1 to 3:
        if chunks[i] != 0:
            emit MOVK Xd, #chunks[i], LSL #(i*16)
```

For negative values near zero, consider MOVN:
```
    if v is negative and only one 16-bit chunk differs from 0xFFFF:
        emit MOVN Xd, #(~chunks[i] & 0xFFFF), LSL #(i*16)
```

---

## 13. Emitter Helper: Syscall Sequence

Pattern for all Linux syscalls:

```
emit_syscall(nr, arg0, arg1, arg2, arg3, arg4, arg5):
    // Load arguments into X0-X5 as needed (often already in place)
    emit MOV X8, #nr        // MOVZ X8, #nr
    emit SVC #0             // 0xD4000001
    // Result in X0 (negative = -errno on error)
```

Example -- `write(1, buf, len)`:
```
    0xD2800020              // MOVZ X0, #1         (fd = stdout)
    // X1 = buf address (load with ADRP+ADD or MOV_IMM64)
    // X2 = len
    0xD2800808              // MOVZ X8, #64        (SYS_WRITE)
    0xD4000001              // SVC #0
```

---

## 14. Emitter Helper: Function Prologue / Epilogue

Standard callee-saved frame for functions that call other functions:

```
// Prologue
STP X29, X30, [SP, #-16]!    // 0xA9BF7BFD
MOV X29, SP                    // 0x910003FD

// ... save any X19-X28 used ...
STP X19, X20, [SP, #-16]!    // 0xA9BF53F3

// Epilogue
LDP X19, X20, [SP], #16      // 0xA8C153F3
LDP X29, X30, [SP], #16      // 0xA8C17BFD
RET                            // 0xD65F03C0
```

Leaf functions (no calls, no stack frame needed):
```
// Just use X0-X15 freely
RET                            // 0xD65F03C0
```

---

## 15. Existing Lithos Encodings Already Verified

From `emit-arm64.fs`:
- `MOV X0, #0` = `0xD2800000` (MOVZ X0, #0)
- `RET` = `0xD65F03C0`

From `elf-wrap.fs`:
- ELF64 ARM64 header generation at base address `0x400000`
- e_machine = 183 (EM_AARCH64)
- Single PT_LOAD, R|X, entry at offset 120

From `launcher.s` and `lithos-bootstrap.s`:
- Full usage of ADRP+ADD, STP/LDP pre/post-index, SVC, CBZ/CBNZ
- Register offset LDR with LSL (PICK operation)
- MSUB for modulo computation
- All syscalls listed in this document
