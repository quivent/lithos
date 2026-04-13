# Hopper (sm_90) SASS Encoding -- Reverse Engineering Notes

Derived from systematic probing with ptxas 12.8 / nvdisasm 12.8.
All hex values verified by varying one operand at a time across 11 probe kernels.

---

## 1. Instruction Format

Every SASS instruction is **16 bytes** (128 bits), stored little-endian:

| Bytes   | Name             | Contents                                          |
|---------|------------------|---------------------------------------------------|
| 0--7    | Instruction word | Opcode, predicate, register operands, immediates  |
| 8--15   | Control word     | Scheduling, barriers, reuse flags, modifiers      |

### Instruction Word Layout (64 bits)

```
[63:32] - Source2 register (reg-reg) OR 32-bit immediate (reg-imm forms)
[31:24] - Source1 register (8-bit register index)
[23:16] - Destination register (8-bit register index)
[15:12] - Predicate guard (see section 4)
[11:0]  - Base opcode (12 bits)
```

For 3-source instructions (FFMA, HFMA2, HMMA): src3 is in ctrl word bits [7:0].

### Encoding Forms

Most instructions have two opcode variants:

| Form         | Opcode range | bits [63:32]          | bits [31:24] |
|--------------|-------------|------------------------|--------------|
| Register-Register | 0x2xx  | src2 register (8-bit, zero-extended) | src1 reg |
| Register-Immediate | 0x4xx/0x8xx | 32-bit IEEE754 float immediate | src1 reg |

---

## 2. Complete Opcode Map

### Base Opcode (bits [11:0])

#### Floating-Point Arithmetic

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x208  | FSEL            | reg-reg  | Float select (conditional move) |
| 0x209  | FMNMX           | reg-reg  | Float min/max (predicate selects min vs max) |
| 0x20b  | FSETP           | reg-reg  | Float set predicate             |
| 0x220  | FMUL            | reg-reg  | Float multiply                  |
| 0x221  | FADD            | reg-reg  | Float add                       |
| 0x223  | FFMA            | reg-reg  | Float fused multiply-add        |
| 0x308  | MUFU            | reg-reg  | Multi-function unit (special functions) |
| 0x421  | FADD            | reg-imm  | Float add (immediate src2)      |
| 0x808  | FSEL            | reg-imm  | Float select (immediate src2)   |
| 0x80b  | FSETP           | reg-imm  | Float set predicate (immediate) |
| 0x820  | FMUL            | reg-imm  | Float multiply (immediate src2) |

#### Half-Precision (FP16x2 packed)

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x232  | HMUL2           | reg-reg  | Packed f16x2 multiply           |
| 0x235  | HFMA2.MMA       | reg-reg  | Packed f16x2 fused multiply-add |
| 0x435  | HFMA2.MMA       | reg-imm  | Packed f16x2 FMA (2x f16 imm)  |
| 0x835  | HFMA2.MMA       | reg-imm  | Packed f16x2 FMA (alternate)    |

#### Tensor Core

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x23c  | HMMA            | reg-reg  | Half-precision matrix multiply-accumulate |

#### Integer Arithmetic

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0xc0c  | ISETP           | reg-reg  | Integer set predicate           |
| 0xc10  | IADD3           | reg-reg  | Integer 3-input add             |
| 0xc24  | IMAD            | reg-reg  | Integer multiply-add            |
| 0x424  | IMAD            | reg-imm  | Integer multiply-add (immediate)|

#### Data Movement

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x802  | MOV             | imm      | Move immediate to register      |
| 0x805  | CS2R            |          | Control/status register to register |
| 0x211  | LEA             | reg-reg  | Load effective address          |
| 0x882  | UMOV            | imm      | Move immediate to uniform register |

#### Memory -- Global

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x981  | LDG             |          | Load global memory              |
| 0x986  | STG             |          | Store global memory             |
| 0xfae  | LDGSTS          |          | Load global, store shared (async copy) |

#### Memory -- Shared

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x984  | LDS             |          | Load shared memory              |
| 0x988  | STS             |          | Store shared memory             |
| 0x83b  | LDSM            |          | Load shared matrix (ldmatrix)   |

#### Memory -- Constant

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0xb82  | LDC             |          | Load constant memory            |
| 0xab9  | ULDC            |          | Uniform load constant           |

#### Special Registers

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x919  | S2R             |          | Special register to register    |
| 0x9c3  | S2UR            |          | Special register to uniform register |

#### Warp Communication

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x589  | SHFL            | reg      | Warp shuffle (register lane src)|
| 0xf89  | SHFL            | imm      | Warp shuffle (immediate lane src)|

#### Synchronization and Barriers

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0xb1d  | BAR             |          | Barrier synchronization         |
| 0x91a  | DEPBAR          |          | Dependency barrier              |
| 0x992  | MEMBAR          |          | Memory barrier                  |
| 0x9ab  | ERRBAR          |          | Error barrier                   |
| 0x5ab  | CGAERRBAR       |          | CGA error barrier               |
| 0x9af  | LDGDEPBAR       |          | Load global dependency barrier  |
| 0x98f  | CCTL            |          | Cache control                   |

#### Control Flow

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x947  | BRA             |          | Branch                          |
| 0x94d  | EXIT            |          | Exit thread                     |
| 0x918  | NOP             |          | No operation                    |

#### Uniform Arithmetic

| Opcode | Mnemonic        | Form     | Description                     |
|--------|-----------------|----------|---------------------------------|
| 0x291  | ULEA            | reg-reg  | Uniform load effective address  |

---

## 3. Register Field Bit Positions

### Standard 3-Operand Format (FADD, FMUL, FMNMX, etc.)

```
Instruction word:
  [23:16] = Destination register (Rd)     -- 8-bit index
  [31:24] = Source 1 register (Rs1)       -- 8-bit index
  [39:32] = Source 2 register (Rs2)       -- 8-bit index (reg-reg form)
  [63:32] = 32-bit immediate             -- (reg-imm form, replaces Rs2)
```

### 4-Operand Format (FFMA, HFMA2.MMA, HMMA)

```
Instruction word:
  [23:16] = Destination register (Rd)
  [31:24] = Source 1 register (Rs1)
  [39:32] = Source 2 register (Rs2)
Control word:
  [7:0]   = Source 3 register (Rs3)
```

### Special Register Values

| Value | Meaning |
|-------|---------|
| 0xFF  | RZ (zero register, reads as 0, writes discarded) |

### Verified Register Encoding Examples (FADD)

```
FADD R5,  R6, R9   -> inst = 0x0000000906057221  (Rd=0x05, Rs1=0x06, Rs2=0x09)
FADD R13, R0, R5   -> inst = 0x00000005000d7221  (Rd=0x0d, Rs1=0x00, Rs2=0x05)
FADD R19, R6, R11  -> inst = 0x0000000b06137221  (Rd=0x13, Rs1=0x06, Rs2=0x0b)
FADD R11, R11, R8  -> inst = 0x000000080b0b7221  (Rd=0x0b, Rs1=0x0b, Rs2=0x08)
```

---

## 4. Predicate Encoding

### Predicate Guard (instruction word bits [15:12])

Every instruction has a 4-bit predicate guard field:

```
bit [15]    = Negate flag (0 = @Px, 1 = @!Px)
bits [14:12] = Predicate register index (0-6 = P0-P6, 7 = PT)
```

| bits [15:12] | Meaning      |
|-------------|--------------|
| 0x0         | @P0          |
| 0x1         | @P1          |
| 0x2         | @P2          |
| 0x3         | @P3          |
| 0x7         | PT (always true, unpredicated) |
| 0x8         | @!P0         |
| 0x9         | @!P1         |
| 0xA         | @!P2         |
| 0xB         | @!P3         |
| 0xF         | @!PT (never executes) |

### Verified Examples

```
EXIT (PT):          inst[15:12] = 0x7  -> 0x...794d
@P0 STG:            inst[15:12] = 0x0  -> 0x...0986
@P2 EXIT:           inst[15:12] = 0x2  -> 0x...294d
@!P1 STG:           inst[15:12] = 0x9  -> 0x...9986
@!P2 FMUL:          inst[15:12] = 0xA  -> 0x...a220
```

### FSETP Destination Predicate Encoding

FSETP writes comparison results to a predicate register. The destination
predicate and comparison mode are encoded in the **control word**:

```
ctrl [23:20] = Second predicate input (PT = 0xF)
ctrl [19:16] = Destination predicate * 2 (P0=0x0, P1=0x2, P2=0x4, P3=0x6)
ctrl [15:12] = Comparison type (see table below)
```

| ctrl[15:12] | Comparison   |
|-------------|-------------|
| 0x4         | .GT         |
| 0x5         | .NE         |
| 0xD         | .NEU        |
| 0xE         | .GEU        |

### FSEL Predicate Select

FSEL (float select) uses a predicate to choose between two sources.
The select predicate is encoded in control word bits [23:20].

---

## 5. Immediate Encoding Format

### 32-bit Float Immediate

In reg-imm instruction forms, bits [63:32] hold a full IEEE 754 single-precision
float value:

```
FADD R5,  R0, 1.0    -> inst[63:32] = 0x3F800000  (1.0f)
FADD R7,  R0, 2.0    -> inst[63:32] = 0x40000000  (2.0f)
FADD R9,  R0, 0.5    -> inst[63:32] = 0x3F000000  (0.5f)
FADD R11, R0, -1.0   -> inst[63:32] = 0xBF800000  (-1.0f)
FADD R15, R0, 50.0   -> inst[63:32] = 0x42480000  (50.0f)
FADD R17, R0, 0.25   -> inst[63:32] = 0x3E800000  (0.25f)
```

The immediate replaces source 2. Source 1 remains a register in bits [31:24].

### Integer Immediate (MOV)

MOV uses opcode 0x802 with the 32-bit immediate in bits [63:32]:

```
MOV R9, 0xbf800000   -> inst = 0xbf80000000097802
MOV R7, 0x3f3504f3   -> inst = 0x3f3504f300077802
```

---

## 6. Control Word Format (64 bits)

```
[63:58] - Register reuse cache flags (6 bits)
          Encodes which operand register values to cache for reuse.
          Bit 58 = src1 reuse, higher bits for src2/src3.
[57:53] - Stall count (5 bits, 0-31 cycles)
          Minimum cycles to wait before issuing this instruction.
[52:52] - Yield hint
[51:49] - Write barrier index (0-5, 7=none)
          Scoreboard slot for tracking when this instruction's write completes.
[48:46] - Read barrier index (0-5, 7=none)
          Scoreboard slot for tracking when this instruction's read completes.
[45:42] - Wait barrier mask (4-6 bits)
          Which scoreboard barriers must complete before this instruction issues.
[41:0]  - Instruction-specific modifiers (varies by opcode)
```

### Instruction-Specific Modifier Fields (ctrl bits [41:0])

#### FADD/FMUL Modifiers

| ctrl bits | Meaning          | Verified |
|-----------|------------------|----------|
| bit 13    | .SAT (saturate to [0,1]) | FADD.SAT -> ctrl bit 13 = 1 |
| bit 9     | abs(src1) modifier | FADD R3, abs(R9), -RZ -> ctrl bit 9 = 1 |
| bit 8     | neg(src1) modifier | FADD R19, -R9, -RZ -> ctrl bit 8 = 1 |

Note: neg(src2) is encoded in instruction word bit [63] = 1 (for reg-reg forms).

#### FMUL Rounding Mode

| ctrl bits [15:14] | Rounding     |
|-------------------|-------------|
| 0b00              | .RN (default, round nearest) |
| 0b11              | .RZ (round toward zero) |

#### MUFU Sub-Function Selector

All special math functions share opcode 0x308 (MUFU). The function
is selected by control word bits [13:10]:

| ctrl[13:10] | Sub-function | Description          |
|-------------|-------------|----------------------|
| 0x0         | MUFU.COS    | Cosine               |
| 0x1         | MUFU.SIN    | Sine                 |
| 0x2         | MUFU.EX2    | 2^x (base-2 exp)    |
| 0x3         | MUFU.LG2    | log2(x)              |
| 0x4         | MUFU.RCP    | 1/x (reciprocal)     |
| 0x5         | MUFU.RSQ    | 1/sqrt(x) (reciprocal sqrt) |
| 0x8         | MUFU.SQRT   | sqrt(x)              |

#### MEMBAR Scope

All memory barriers share opcode 0x992. Scope is in control word bits [13:12]:

| ctrl[13:12] | Scope        |
|-------------|-------------|
| 0x0         | .CTA (block-level) |
| 0x2         | .GPU (device-level) |
| 0x3         | .SYS (system-level) |

#### FMNMX (Float Min/Max)

FMNMX uses opcode 0x209 for both min and max. A predicate operand
selects the operation:

```
FMNMX R13, R7, R0, PT   -> min (PT selects min)
FMNMX R15, R7, R0, !PT  -> max (!PT selects max)
```

The select predicate is encoded in control word bits [23:20]:
- PT (0x7) in those bits -> min
- !PT (0xF) in those bits -> max

---

## 7. SHFL (Warp Shuffle) Encoding

Two opcode forms depending on whether the lane source is register or immediate:

| Opcode | Form |
|--------|------|
| 0x589  | Register lane source (e.g., SHFL.IDX with RZ) |
| 0xf89  | Immediate lane source (e.g., SHFL.BFLY/DOWN/UP with literal) |

Shuffle mode is in instruction word bits [59:58]:

| bits [59:58] | Mode       |
|-------------|-----------|
| 0b00        | .IDX      |
| 0b01        | .UP       |
| 0b10        | .DOWN     |
| 0b11        | .BFLY     |

Instruction word layout:
```
[63:60] - Mode and flags
[59:58] - Shuffle mode
[55:48] - Clamp/mask value
[47:40] - Lane width mask (0x1f = full warp)
[31:24] - Source register
[23:16] - Destination register
```

### Verified Examples

```
SHFL.BFLY PT, R5, R0, 0x1, 0x1f  -> 0x0c201f0000057f89
SHFL.DOWN PT, R7, R0, 0x1, 0x1f  -> 0x08201f0000077f89
SHFL.UP   PT, R9, R0, 0x1, 0x1f  -> 0x04201f0000097f89
SHFL.IDX  PT, R11, R0, RZ, 0x1f  -> 0x00001fff000b7589
```

---

## 8. Tensor Core (HMMA) Encoding

### HMMA.16816.F32 (mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32)

```
Opcode: 0x23c
Instruction word:
  [23:16] = Destination start register (R4)
  [31:24] = Source A start register (R4, 4 consecutive regs)
  [39:32] = Source B start register (R8, 2 consecutive regs)
Control word:
  [7:0]   = Source C/Accumulator start register (RZ=0xFF for zero init)
  [15:8]  = Shape/type encoding (0x18 for m16n8k16.f32.f16.f16.f32)
```

Verified encoding:
```
HMMA.16816.F32 R4, R4, R8, RZ
  inst = 0x000000080404723c
  ctrl = 0x001fde00000018ff
```

---

## 9. Shared Memory Instruction Encoding

### LDS (Load Shared)

Opcode: 0x984. Size encoding in control word bits [13:10]:

| ctrl[13:10] | Size   |
|-------------|--------|
| 0x2         | 32-bit |
| 0x2 + ctrl[9]=1 | 64-bit |
| 0x3         | 128-bit |

```
LDS R7, [R5]       -> inst=0x0000000005077984  ctrl bits[13:10]=0x2 (32-bit)
LDS.64 R4, [UR4]   -> inst=0x00000004ff047984  ctrl=...0x0a00 (64-bit)
```

### STS (Store Shared)

Opcode: 0x988. Size in control word bits [13:10] (0x3 = 128-bit).

```
STS.128 [UR4], RZ -> inst=0x000000ffff007988  ctrl=...0x0c04
```

### LDSM (Load Shared Matrix)

Opcode: 0x83b. Used for ldmatrix.

```
LDSM.16.M88.4 R4, [UR4] -> inst=0x00000004ff04783b  ctrl=0x000e220008000200
```

---

## 10. Async Copy (LDGSTS) Encoding

### LDGSTS (Load Global Store Shared)

Opcode: 0xfae. Combines global load + shared store in one instruction.

```
LDGSTS.E.128 [R5], desc[UR4][R2.64]            -> inst=0x0000000002057fae
LDGSTS.E.128 [R5+0x10], desc[UR4][R2.64+0x10]  -> inst=0x0001001002057fae
```

### LDGDEPBAR

Opcode: 0x9af. Marks dependency for async copies.

```
LDGDEPBAR -> inst=0x00000000000079af  ctrl=0x000e220000000000
```

### DEPBAR

Opcode: 0x91a. Waits on scoreboard barrier.

```
DEPBAR.LE SB0, 0x0 -> inst=0x000080000000791a  ctrl=0x000fc80000000000
```

---

## 11. Synchronization Encoding

### BAR (Barrier)

Opcode: 0xb1d.

```
BAR.SYNC.DEFER_BLOCKING 0x0 -> inst=0x0000000000007b1d  ctrl=0x000fec0000010000
```

### MEMBAR

Opcode: 0x992. Scope in ctrl[13:12] (see section 6).

### ERRBAR / CGAERRBAR

ERRBAR opcode: 0x9ab. CGAERRBAR opcode: 0x5ab.
These are emitted by the compiler after MEMBAR.SC.GPU and MEMBAR.SC.SYS.

---

## 12. Predicated Instruction Examples (Encoding Cross-Reference)

```
  (no pred)  EXIT           -> 0x000000000000794d  (bits[15:12]=0x7=PT)
  @P0        STG.E          -> 0x0000180704000986  (bits[15:12]=0x0=P0)
  @P2        EXIT           -> 0x000000000000294d  (bits[15:12]=0x2=P2)
  @!P1       STG.E          -> 0x00001c0004009986  (bits[15:12]=0x9=!P1)
  @!P1       FMUL (imm)     -> 0x4b80000000009820  (bits[15:12]=0x9=!P1)
  @!P2       FMUL (reg-reg) -> 0x000000090909a220  (bits[15:12]=0xA=!P2)
  @!P2       FSEL           -> 0x000000010409a208  (bits[15:12]=0xA=!P2)
```

---

## 13. NEG/ABS Modifier Encoding

Float negation and absolute value are NOT separate instructions.
They are modifiers on FADD (and other float instructions):

```
abs(x) = FADD Rd, |Rs|, -RZ
  -> inst bit [63] = 1 (negate src2, i.e., -RZ)
  -> ctrl bit [9] = 1 (absolute value on src1)
  Example: FADD R3, |R9|, -RZ = inst 0x800000ff09037221, ctrl ...0200

neg(x) = FADD Rd, -Rs, -RZ
  -> inst bit [63] = 1 (negate src2)
  -> ctrl bit [8] = 1 (negate src1)
  Example: FADD R19, -R9, -RZ = inst 0x800000ff09137221, ctrl ...0100
```

---

## 14. FMIN/FMAX Encoding

There is no separate FMIN or FMAX opcode. Both map to **FMNMX** (opcode 0x209):

```
FMIN = FMNMX Rd, Rs1, Rs2, PT    (select predicate = PT -> min)
FMAX = FMNMX Rd, Rs1, Rs2, !PT   (select predicate = !PT -> max)
```

The select predicate is in control word bits [23:20]:
- 0x3 or 0x7 (PT) -> min
- 0xF (!PT) -> max

```
FMNMX R13, R7, R0, PT  -> inst=0x00000000070d7209  ctrl=...03800000 (min)
FMNMX R15, R7, R0, !PT -> inst=0x00000000070f7209  ctrl=...07800000 (max)
```

---

## 15. Summary: Opcode Map (Sorted by Opcode)

| Opcode [11:0] | Mnemonic    | Category         |
|---------------|-------------|------------------|
| 0x208         | FSEL        | Float select     |
| 0x209         | FMNMX       | Float min/max    |
| 0x20b         | FSETP       | Float set pred   |
| 0x211         | LEA         | Address calc     |
| 0x220         | FMUL        | Float multiply   |
| 0x221         | FADD        | Float add        |
| 0x223         | FFMA        | Float FMA        |
| 0x232         | HMUL2       | Half multiply    |
| 0x235         | HFMA2       | Half FMA         |
| 0x23c         | HMMA        | Tensor core MMA  |
| 0x291         | ULEA        | Uniform LEA      |
| 0x308         | MUFU        | Special functions|
| 0x421         | FADD (imm)  | Float add        |
| 0x424         | IMAD (imm)  | Int multiply-add |
| 0x435         | HFMA2 (imm) | Half FMA         |
| 0x589         | SHFL (reg)  | Warp shuffle     |
| 0x5ab         | CGAERRBAR   | Sync             |
| 0x802         | MOV         | Data movement    |
| 0x805         | CS2R        | Ctrl reg read    |
| 0x808         | FSEL (imm)  | Float select     |
| 0x80b         | FSETP (imm) | Float set pred   |
| 0x820         | FMUL (imm)  | Float multiply   |
| 0x835         | HFMA2 (imm) | Half FMA         |
| 0x83b         | LDSM        | Shared matrix ld |
| 0x882         | UMOV        | Uniform move     |
| 0x918         | NOP         | No operation     |
| 0x919         | S2R         | Special reg read |
| 0x91a         | DEPBAR      | Dep barrier      |
| 0x947         | BRA         | Branch           |
| 0x94d         | EXIT        | Exit thread      |
| 0x981         | LDG         | Global load      |
| 0x984         | LDS         | Shared load      |
| 0x986         | STG         | Global store     |
| 0x988         | STS         | Shared store     |
| 0x98f         | CCTL        | Cache control    |
| 0x992         | MEMBAR      | Memory barrier   |
| 0x9ab         | ERRBAR      | Error barrier    |
| 0x9af         | LDGDEPBAR   | Async dep barrier|
| 0x9c3         | S2UR        | Spec reg to unif |
| 0xab9         | ULDC        | Uniform const ld |
| 0xb1d         | BAR         | Barrier sync     |
| 0xb82         | LDC         | Constant load    |
| 0xc0c         | ISETP       | Int set pred     |
| 0xc10         | IADD3       | Int 3-input add  |
| 0xc24         | IMAD        | Int multiply-add |
| 0xf89         | SHFL (imm)  | Warp shuffle     |
| 0xfae         | LDGSTS      | Async copy       |

---

## 16. Key Observations for Code Generation

1. **Opcode is 12 bits** (bits [11:0]), not 16 as initially thought. Bits [15:12]
   are always the predicate guard.

2. **Two instruction forms** share the same mnemonic but have different opcodes
   (e.g., FADD reg-reg = 0x221, FADD reg-imm = 0x421).

3. **FMIN/FMAX** do not exist as separate opcodes. Use FMNMX with PT/!PT.

4. **FNEG/FABS** do not exist as separate opcodes. Use FADD with modifier bits.

5. **MUFU** is a single opcode (0x308) with sub-function in ctrl[13:10].
   This covers RCP, RSQ, SIN, COS, EX2, LG2, and SQRT.

6. **src3** for 4-operand instructions (FFMA, HFMA2, HMMA) is in control
   word bits [7:0], NOT in the instruction word.

7. **Scheduling is per-instruction** -- each instruction has its own 64-bit
   control word with stall counts, barrier indices, and reuse flags.

8. **RZ (R255 = 0xFF)** serves as both a zero source and a null destination.

9. **Async copy** (cp.async in PTX) maps to LDGSTS, not a separate CP.ASYNC
   instruction. The compiler emits LDGSTS + LDGDEPBAR + DEPBAR.

10. **The compiler aggressively constant-folds** at ptxas time. To get actual
    arithmetic instructions emitted, source operands must come from memory loads
    or other non-constant sources.
