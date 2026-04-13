# SASS SFU Encodings — Hopper (sm_90a)

Verified by compiling minimal PTX kernels with `ptxas -arch sm_90a` and
disassembling with `nvdisasm -hex`. All encodings are exact 64-bit words
from actual cubin output. No guesses.

Tool versions: ptxas 12.8.93, nvdisasm 12.8.90.

---

## MUFU — Special Function Unit

All MUFU variants share **opcode `0x7308`** in instruction word bits [15:0].
The sub-function is selected by **`extra41` bits [13:8]** in the control word.
This field passes through `make-ctrl` as the `extra41` argument; it does not
appear in the instruction word at all.

### Instruction word format

```
bits [15:0]  = 0x7308  (opcode, constant for all MUFU)
bits [23:16] = Rd      (destination register index, 8 bits)
bits [31:24] = 0x00    (unused in MUFU)
bits [39:32] = Rs      (source register index, 8 bits)
bits [63:40] = 0x00    (unused)
```

Template (with Rd=R7, Rs=R0):
```
0x0000000000077308
      ^^^^^^^^ Rs=R0 at bits[39:32]
          ^^^^  Rd=R7 at bits[23:16]
```

### Sub-function encoding (in control word `extra41` field)

The `extra41` value is placed in ctrl bits [40:0]. Only bits [13:8] of that
field vary between MUFU sub-functions; bits [7:0] are always 0x00.

| Operation | PTX                | extra41 subop | ctrl bits [13:8] |
|-----------|-------------------|---------------|-----------------|
| COS       | `cos.approx.f32`  | `0x0000`      | `0x00`          |
| SIN       | `sin.approx.f32`  | `0x0400`      | `0x04`          |
| EX2       | `ex2.approx.f32`  | `0x0800`      | `0x08`          |
| LG2       | `lg2.approx.f32`  | `0x0c00`      | `0x0c`          |
| RCP       | `rcp.approx.f32`  | `0x1000`      | `0x10`          |
| RSQ       | `rsqrt.approx.f32`| `0x1400`      | `0x14`          |
| SQRT      | `sqrt.approx.f32` | `0x2000`      | `0x20`          |

Pattern: subop field increments in steps of 4 (i.e., step of 1 in bits [11:10]).
COS=0, SIN=1, EX2=2, LG2=3, RCP=4, RSQ=5, SQRT=8 when counted in units of 4.

### Full 128-bit encodings (control | instruction)

Observed from ptxas-generated cubins with minimal scheduling context.
The control word scheduling fields (stall, wbar) will differ in a real
kernel — the `extra41` subop field is the only functionally significant
part of the control word.

```
MUFU.RCP  R0, R0:  ctrl=0x000e640000001000  inst=0x0000000000007308
MUFU.EX2  R7, R0:  ctrl=0x000e640000000800  inst=0x0000000000077308
MUFU.RSQ  R7, R0:  ctrl=0x000e640000001400  inst=0x0000000000077308
MUFU.LG2  R7, R0:  ctrl=0x000e640000000c00  inst=0x0000000000077308
MUFU.SIN  R7, R7:  ctrl=0x000e240000000400  inst=0x0000000700077308
MUFU.COS  R7, R7:  ctrl=0x000e240000000000  inst=0x0000000700077308
MUFU.SQRT R7, R0:  ctrl=0x000e640000002000  inst=0x0000000000077308
```

Control word scheduling for these probe kernels (decoded from ctrl):

| Op   | stall | yield | wbar | rbar | extra41 (subop) |
|------|-------|-------|------|------|-----------------|
| RCP  | 2     | 1     | 1    | 7    | 0x1000          |
| EX2  | 2     | 1     | 1    | 7    | 0x0800          |
| RSQ  | 2     | 1     | 1    | 7    | 0x1400          |
| LG2  | 2     | 1     | 1    | 7    | 0x0c00          |
| SIN  | 2     | 1     | 0    | 7    | 0x0400          |
| COS  | 2     | 1     | 0    | 7    | 0x0000          |
| SQRT | 2     | 1     | 1    | 7    | 0x2000          |

Note: SIN and COS have `ctrl[47:40] = 0x24` (trig pipeline), while all
other MUFU variants have `ctrl[47:40] = 0x64` (math pipeline). Both
field values encode the same op-class; the difference is likely a pipeline
hint or latency code baked in by ptxas.

Cross-check: previously probed sm_90 data (from `sass/probe_2b.sass`) used
`stall=8` and varied wbars, but the `extra41` subop values are **identical**,
confirming the subop field is scheduling-independent.

### `emit-sass.fs` verification

`$7308 constant OP-MUFU` — **confirmed correct**. All MUFU variants share
this opcode in bits [15:0] of the instruction word.

The `ctrl-mufu` constructor in `emit-sass.fs`:
```forth
: ctrl-mufu  ( subfn-extra41 wbar -- ctrl64 )
  swap >r
  >r 8 1 r> 7 0 0 r>
  make-ctrl ;
```
Uses `stall=8 yield=1 rbar=7` — appropriate for a real kernel context
where MUFU results are consumed by the next instruction.

---

## Two-instruction sequences for exp and log

PTX `ex2.approx.f32` and `lg2.approx.f32` implement 2-input math functions,
but natural `exp(x)` and `log(x)` require an additional FMUL.

### exp(x) = 2^(x · log₂e)

```
ptxas expansion of exp(x):
  FMUL   Rt, Rx, 1.4426950216293335   ; log2(e) = 0x3fb8aa3b
  MUFU.EX2 Rd, Rt                      ; 2^Rt
```

From probe kernel `probe_exp`:
```
/*0050*/ FMUL R0, R2, 1.4426950216293334961  /* inst=0x3fb8aa3b02007820 ctrl=0x004fca0000400000 */
/*0080*/ MUFU.EX2 R7, R0                     /* inst=0x0000000000077308 ctrl=0x000e640000000800 */
```

FMUL-immediate opcode: `0x7820`. Immediate is the upper 32 bits of the
instruction word, stored as a raw IEEE-754 float32 bit pattern.

### log(x) = log₂(x) · ln(2)

```
ptxas expansion of log(x):
  MUFU.LG2 Rt, Rx                     ; log2(x)
  FMUL     Rd, Rt, 0.6931471824645996 ; ln(2) = 0x3f317218
```

From probe kernel `probe_log`:
```
/*0070*/ MUFU.LG2 R6, R0              /* inst=0x0000000000067308 ctrl=0x000e640000000c00 */
/*0090*/ FMUL R7, R6, 0.69314718246459960938  /* inst=0x3f31721806077820 ctrl=0x000fca0000400000 */
```

### FMUL-immediate instruction format

```
bits [15:0]  = 0x7820  (FMUL-immediate opcode)
bits [23:16] = Rd      (destination)
bits [31:24] = Rs      (source register)
bits [63:32] = imm32   (IEEE-754 float32 immediate, big-endian bit position)
```

---

## Integer/Float Conversions

### I2FP.F32.S32 — signed int32 to float32

PTX: `cvt.rn.f32.s32`

```
Instruction word: 0x0000000200077245
  bits [15:0]  = 0x7245  (I2FP opcode)
  bits [23:16] = Rd      (float32 destination, e.g. R7)
  bits [39:32] = Rs      (int32 source, e.g. R2)

Control word:     0x004fca0000201400
  stall   = 5
  yield   = 0
  wbar    = 7
  rbar    = 7
  wait    = 4
  extra41 = 0x00201400
```

**Correction to `emit-sass.fs`**: the file previously claimed opcode `0x7306`
(marked "unverified"). The actual sm_90a opcode is **`0x7245`**. The SASS
mnemonic is `I2FP.F32.S32`, not `I2F.F32.S32`.

### F2I.TRUNC.NTZ — float32 to signed int32 (truncate toward zero)

PTX: `cvt.rzi.s32.f32`

```
Instruction word: 0x0000000200077305
  bits [15:0]  = 0x7305  (F2I opcode)
  bits [23:16] = Rd      (int32 destination, e.g. R7)
  bits [39:32] = Rs      (float32 source, e.g. R2)

Control word:     0x004e24000020f100
  stall   = 2
  yield   = 1
  wbar    = 0
  rbar    = 7
  wait    = 4
  extra41 = 0x0020f100
```

Note: `F2I` opcode `0x7305` is adjacent to `MUFU` opcode `0x7308` — these
are distinct instructions despite the similar opcode values.

---

## Opcode summary table

| SASS mnemonic      | Opcode (bits[15:0]) | Notes                      |
|--------------------|--------------------|-----------------------------|
| MUFU.* (all)       | `0x7308`           | Subop in ctrl extra41[13:8] |
| I2FP.F32.S32       | `0x7245`           | int32→float32               |
| F2I.TRUNC.NTZ      | `0x7305`           | float32→int32 (truncate)    |
| FMUL (reg)         | `0x7220`           | FMUL Rd, Ra, Rb             |
| FMUL (immediate)   | `0x7820`           | FMUL Rd, Ra, imm32          |

---

## Source PTX probes

All probes compiled and verified:

```
/tmp/sass-probe/rcp.ptx    → rcp.approx.f32  → MUFU.RCP
/tmp/sass-probe/ex2.ptx    → ex2.approx.f32  → MUFU.EX2
/tmp/sass-probe/rsqrt.ptx  → rsqrt.approx.f32→ MUFU.RSQ
/tmp/sass-probe/sqrt.ptx   → sqrt.approx.f32 → MUFU.SQRT
/tmp/sass-probe/lg2.ptx    → lg2.approx.f32  → MUFU.LG2
/tmp/sass-probe/sin.ptx    → sin.approx.f32  → MUFU.SIN
/tmp/sass-probe/cos.ptx    → cos.approx.f32  → MUFU.COS
/tmp/sass-probe/exp.ptx    → FMUL(log2e) + MUFU.EX2
/tmp/sass-probe/log.ptx    → MUFU.LG2 + FMUL(ln2)
/tmp/sass-probe/i2f.ptx    → cvt.rn.f32.s32  → I2FP.F32.S32
/tmp/sass-probe/f2i.ptx    → cvt.rzi.s32.f32 → F2I.TRUNC.NTZ
```

Compilation: `ptxas -arch sm_90a`
Disassembly: `nvdisasm -hex -c`
