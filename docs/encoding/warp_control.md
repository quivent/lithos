# sm_90a SASS: Warp Control, Shuffles, Barriers, Special Registers

All encodings verified via `ptxas -arch sm_90a` + `nvdisasm --print-instruction-encoding`.
Probe files: `probe_6b.sass` (SHFL), `probe_7.sass` (MEMBAR), `probe_reduce` (5-step reduction),
`probe_s2r2/s2r3` (S2R all variants), `probe_bar` (BAR.SYNC), `probe_bra3` (BRA).

Each instruction is 128 bits = 64-bit `inst` word + 64-bit `ctrl` word, emitted little-endian.

---

## 1. Warp Shuffle Instructions

### Instruction word layout (immediate-delta form, opcode 0x7f89)

```
bits[15:0]  = 0x7f89  opcode: SHFL, PT predicate, immediate mode
bits[23:16] = Rd      destination register
bits[31:24] = Rs      source register (whose value is shuffled)
bits[39:32] = 0x00    unused in immediate form
bits[47:40] = 0x1f    warp mask (31 = all 32 lanes)
bits[55:48] = clamp   = (delta * 32) & 0xff
bits[63:56] = mode    = type_base | ((delta * 32) >> 8)
```

**Shuffle type** in `mode[3:2]`:

| mode[3:2] | type_base | shuffle | description |
|-----------|-----------|---------|-------------|
| 0b11 | 0x0c | BFLY | butterfly XOR — lane N XORs with lane (N ^ delta) |
| 0b10 | 0x08 | DOWN | lane N reads from lane N+delta |
| 0b01 | 0x04 | UP   | lane N reads from lane N-delta |
| 0b00 | 0x00 | IDX  | all lanes read from absolute lane index |

**Delta encoding** — both `mode` and `clamp` together carry a 10-bit value `delta * 32`:

| delta | delta×32 | mode byte | clamp byte | top-16 of inst |
|-------|----------|-----------|------------|----------------|
| 1     | 0x020    | 0x0c      | 0x20       | 0x0c20         |
| 2     | 0x040    | 0x0c      | 0x40       | 0x0c40         |
| 4     | 0x080    | 0x0c      | 0x80       | 0x0c80         |
| 8     | 0x100    | 0x0d      | 0x00       | 0x0d00         |
| 16    | 0x200    | 0x0e      | 0x00       | 0x0e00         |

Formula: `delta_enc = delta * 32; clamp = delta_enc & 0xff; mode = type_base | (delta_enc >> 8)`

### 1a. SHFL.BFLY — butterfly XOR shuffle

PTX: `shfl.sync.bfly.b32 %dst|%p, %src, delta, 0x1f, 0xffffffff`

Full verified encodings (rd=R5, rs=R0):

```
delta= 1:  inst=0x0c201f0000057f89  ctrl=0x001e6800000e0000
delta= 2:  inst=0x0c401f0000057f89  ctrl=0x001e6800000e0000
delta= 4:  inst=0x0c801f0000057f89  ctrl=0x001e6800000e0000
delta= 8:  inst=0x0d001f0000057f89  ctrl=0x001e6800000e0000
delta=16:  inst=0x0e001f0000057f89  ctrl=0x001e6800000e0000
```

Ctrl word: stall=14 yield=0 wbar=3 rbar=7.

**BUG FOUND AND FIXED**: The original `shfl-bfly,` implementation in `gpu/emit.fs` used the
formula `mode_byte = (log2(delta)<<4)|0x0c` with `clamp` hardcoded to `0x20`. This only
produces the correct encoding for `delta=1`. For deltas 2, 4, 8, 16 the emitted instructions
are invalid SASS. The fix: `mode = 0x0c | ((delta*32)>>8)`, `clamp = (delta*32) & 0xff`.

### 1b. SHFL.DOWN — down shuffle

PTX: `shfl.sync.down.b32 %dst|%p, %src, delta, 0x1f, 0xffffffff`

```
delta= 1:  inst=0x08201f0000077f89  ctrl=0x000e2800000e0000  (probe_6b)
delta= 2:  inst=0x08401f0000077f89
delta= 4:  inst=0x08801f0000077f89
delta= 8:  inst=0x09001f0000077f89
delta=16:  inst=0x0a001f0000077f89
```

Same ctrl scheduling as BFLY.

### 1c. SHFL.UP — up shuffle

PTX: `shfl.sync.up.b32 %dst|%p, %src, delta, 0x0, 0xffffffff`

Note: for UP-shuffle the mask argument in PTX is the lower-clamp boundary (not the same semantic as BFLY/DOWN).

```
delta= 1:  inst=0x04201f0000097f89  ctrl=0x000ea800000e0000  (probe_6b)
delta= 2:  inst=0x04401f0000097f89
delta= 4:  inst=0x04801f0000097f89
delta= 8:  inst=0x05001f0000097f89
delta=16:  inst=0x06001f0000097f89
```

### 1d. SHFL.IDX — indexed (absolute lane) shuffle

PTX: `shfl.sync.idx.b32 %dst|%p, %src, lane_idx, 0x1f, 0xffffffff`

Two opcode variants exist:

**Register-index form** (opcode 0x7389, both delta and clamp in registers):
```
SHFL.IDX PT, R11, R0, RZ, 0x1f  ->  inst=0x00001fff000b7589  ctrl=0x000ee800000e0000
  bits[15:0]  = 0x7589  (IDX, reg-delta, imm-clamp)
  bits[23:16] = Rd
  bits[31:24] = Rs
  bits[39:32] = lane_reg  (RZ=0xff means lane 0)
  bits[47:40] = 0x1f  mask
  bits[55:48] = 0x00  clamp
  bits[63:56] = 0x00  mode (IDX type)
```

Full register form (opcode 0x7389):
```
SHFL.IDX PT, R4, R3, R5, R8  ->  inst=0x0000000503047389
  bits[39:32] = R5  (lane index register)
  bits[47:40+] encodes R8 (clamp register) via different field layout
```

---

## 2. Five-Step Warp Reduction Sequence

The Σ (reduction) primitive for a 32-lane warp sum, `acc` accumulates result, `tmp` is scratch:

```
Step 1 (delta=16):  SHFL.BFLY PT, tmp, acc, 0x10, 0x1f  = 0x0e001f[acc][tmp]7f89
                    FADD acc, acc, tmp
Step 2 (delta=8):   SHFL.BFLY PT, tmp, acc, 0x08, 0x1f  = 0x0d001f[acc][tmp]7f89
                    FADD acc, acc, tmp
Step 3 (delta=4):   SHFL.BFLY PT, tmp, acc, 0x04, 0x1f  = 0x0c801f[acc][tmp]7f89
                    FADD acc, acc, tmp
Step 4 (delta=2):   SHFL.BFLY PT, tmp, acc, 0x02, 0x1f  = 0x0c401f[acc][tmp]7f89
                    FADD acc, acc, tmp
Step 5 (delta=1):   SHFL.BFLY PT, tmp, acc, 0x01, 0x1f  = 0x0c201f[acc][tmp]7f89
                    FADD acc, acc, tmp
```

Replace `[acc]` with register byte at bits[31:24] and `[tmp]` at bits[23:16].

Example with acc=R0=0x00, tmp=R1=0x01:
```
SHFL.BFLY PT, R1, R0, 0x10, 0x1f:  0x0e001f0000017f89
SHFL.BFLY PT, R1, R0, 0x08, 0x1f:  0x0d001f0000017f89
SHFL.BFLY PT, R1, R0, 0x04, 0x1f:  0x0c801f0000017f89
SHFL.BFLY PT, R1, R0, 0x02, 0x1f:  0x0c401f0000017f89
SHFL.BFLY PT, R1, R0, 0x01, 0x1f:  0x0c201f0000017f89
```

After all 5 steps, lane 0 holds the full 32-lane sum.

---

## 3. BAR.SYNC — Block Barrier

PTX: `bar.sync N`  → SASS: `BAR.SYNC.DEFER_BLOCKING N`

```
inst word:  0x00[barid*0x40]000000007b1d
ctrl word:  0x000fec0000010000
```

Field layout:
- `bits[15:0]  = 0x7b1d`  opcode
- `bits[55:48] = barrier_id * 0x40`  (barrier ID 0-15, shifted by 6 into bits[55:48])
- ctrl `extra41 = 0x00010000` (constant for all BAR.SYNC)

In 64-bit encoding: `inst = (bar_id << 54) | 0x7b1d`

Verified encodings:
```
BAR.SYNC 0:  inst=0x0000000000007b1d  ctrl=0x000fec0000010000
BAR.SYNC 1:  inst=0x0040000000007b1d  ctrl=0x000fec0000010000
             (0x40 = 64 = 1 * 64 at bits[55:48])
```

Scheduling: stall=5, yield=1, wbar=7, rbar=7.

**BUG FIXED**: The previous `bar-sync,` implementation used `bar-id 16 lshift` placing the ID
in bits[23:16]. The correct position is bits[55:48] (= `bar_id 54 lshift`). For barrier ID 0
both encodings happen to produce the same result (0 shifted anywhere is 0), so single-barrier
kernels worked but any kernel using two or more barriers would have silently emitted wrong SASS.

BAR.SYNC is distinct from DEPBAR:
- `BAR.SYNC` synchronizes all threads in the CTA at a named barrier
- `DEPBAR.LE` is a scoreboard stall for instruction-level dependency tracking

---

## 4. MEMBAR — Memory Fences

All MEMBAR variants share the same instruction word. Scope is in ctrl `extra41[13:12]`:

```
MEMBAR.SC.GPU  (membar.gl in PTX):
  inst=0x0000000000007992  ctrl=0x000fec0000002000
  extra41[13:12] = 0b10 = scope GPU

MEMBAR.SC.SYS  (membar.sys in PTX):
  inst=0x0000000000007992  ctrl=0x000fec0000003000
  extra41[13:12] = 0b11 = scope SYS

MEMBAR.SC.CTA  (membar.cta in PTX):
  inst=0x0000000000007992  ctrl=0x0003ec0000000000
  extra41[13:12] = 0b00 = scope CTA
  Note: CTA scope uses different sched word (sched=0x0003ec00, lower stall)
```

Scope encoding summary:

| PTX | SASS | extra41[13:12] | extra41 value |
|-----|------|----------------|---------------|
| membar.cta | MEMBAR.SC.CTA | 00 | 0x00000000 |
| membar.gl  | MEMBAR.SC.GPU | 10 | 0x00002000 |
| membar.sys | MEMBAR.SC.SYS | 11 | 0x00003000 |

---

## 5. S2R — Special Register Reads

PTX: `mov.u32 %r, %tid.x` → SASS: `S2R Rd, SR_TID.X`

The instruction word carries only the destination register. The SR selector lives
**entirely in the ctrl word's extra41 field, bits[15:8]**.

```
inst = 0x00000000000r7919   (r = Rd in bits[23:16])
ctrl = ctrl-s2r-base | (sr_id << 8)
```

### SR selector values (verified via probe_s2r2, probe_s2r3 -O0)

| PTX special reg | SASS name | SR id (hex) | ctrl extra41 |
|-----------------|-----------|-------------|--------------|
| %tid.x          | SR_TID.X  | 0x21 | 0x2100 |
| %tid.y          | SR_TID.Y  | 0x22 | 0x2200 |
| %tid.z          | SR_TID.Z  | 0x23 | 0x2300 |
| %ctaid.x        | SR_CTAID.X | 0x25 | 0x2500 |
| %ctaid.y        | SR_CTAID.Y | 0x26 | 0x2600 |
| %ctaid.z        | SR_CTAID.Z | 0x27 | 0x2700 |
| %laneid         | SR_LANEID | 0x00 | 0x0000 |

**Note**: `%nctaid.x`, `%warpid`, `%nwarpid` are **not** emitted as S2R by ptxas — they become
constant-bank loads (`LDC Rd, c[0x0][offset]`) because ptxas treats grid dimensions as static.

### Verified ctrl words (from probe_s2r2 with -O2):

```
S2R Rd, SR_TID.X:   inst=0x00000000000r7919  ctrl=0x000e2e0000002100
S2R Rd, SR_TID.Y:   inst=0x00000000000r7919  ctrl=0x000e620000002200
S2R Rd, SR_CTAID.X: inst=0x00000000000r7919  ctrl=0x000eaa0000002500
S2R Rd, SR_CTAID.Y: inst=0x00000000000r7919  ctrl=0x000f220000002600
S2R Rd, SR_LANEID:  inst=0x00000000000r7919  ctrl=0x000f620000000000
```

The sched portion of ctrl varies by instruction position (write-barrier 1, stall=7, yield=1, rbar=7).
The key invariant is that `extra41[15:8] = sr_id`.

---

## 6. Control Flow

### 6a. BRA — Unconditional Branch

PTX: `bra TARGET`

```
inst = (offset32 << 32) | 0x00fc7947
ctrl = 0x000fc0000383ffff

bits[15:0]  = 0x7947  opcode
bits[23:16] = 0xfc    PT predicate (always-taken)
bits[31:24] = 0x00
bits[63:32] = signed 32-bit offset in 4-byte units from PC_next
```

**Target formula**: `target = (bra_addr + 16) + offset32 * 4`

For a self-loop (target = current instruction):
`offset32 = (bra_addr - (bra_addr + 16)) / 4 = -16/4 = -4 = 0xfffffffc`

Verified: `BRA loop` → `inst=0xfffffffc00fc7947  ctrl=0x000fc0000383ffff`

### 6b. @Pred BRA — Conditional Branch

```
inst = (offset32 << 32) | (pred_field << 12) | 0x00fc7947 & ~(0x7 << 12)
     = (offset32 << 32) | (pred_id << 12) | 0x0000f047
```

where pred_id: P0=0, P1=1, P2=2, P3=3, PT=7.

For `@P0 BRA`: opcode byte `0x0947` (P0 in bits[14:12] = 0b000).
For negated `@!P0 BRA`: `0x8947` (negate bit set in bits[15]).

Example verified from probe_6b:
```
BRA .loop:  inst=0xfffffffc00fc7947  (self-loop, PT, offset=-4)
```

### 6c. EXIT — Thread Termination

```
inst = 0x000000000000794d
ctrl = 0x000fea0003800000

Scheduling: stall=5, yield=1, wbar=7, rbar=7, extra41=0x03800000
```

For predicated exit `@!P0 EXIT`:
```
inst = 0x000000000000894d  (P0 negated)
ctrl = 0x001fea0003800000
```

---

## 7. Summary Table

| PTX instruction | SASS mnemonic | Opcode | Key fields |
|----------------|---------------|--------|-----------|
| shfl.sync.bfly.b32 | SHFL.BFLY | 0x7f89 | mode=0x0c\|(d\*32>>8), clamp=d\*32&0xff |
| shfl.sync.down.b32 | SHFL.DOWN | 0x7f89 | mode=0x08\|(d\*32>>8), clamp=d\*32&0xff |
| shfl.sync.up.b32   | SHFL.UP   | 0x7f89 | mode=0x04\|(d\*32>>8), clamp=d\*32&0xff |
| shfl.sync.idx.b32  | SHFL.IDX  | 0x7589/0x7389 | mode=0x00, lane in bits[39:32] |
| bar.sync N         | BAR.SYNC.DEFER_BLOCKING | 0x7b1d | barrier_id in bits[55:48]=(N<<6) |
| membar.gl          | MEMBAR.SC.GPU | 0x7992 | ctrl extra41=0x2000 |
| membar.sys         | MEMBAR.SC.SYS | 0x7992 | ctrl extra41=0x3000 |
| membar.cta         | MEMBAR.SC.CTA | 0x7992 | ctrl extra41=0x0000 |
| mov.u32 r, %tid.x  | S2R SR_TID.X  | 0x7919 | SR id=0x21 in ctrl extra41[15:8] |
| mov.u32 r, %tid.y  | S2R SR_TID.Y  | 0x7919 | SR id=0x22 |
| mov.u32 r, %ctaid.x| S2R SR_CTAID.X| 0x7919 | SR id=0x25 |
| mov.u32 r, %ctaid.y| S2R SR_CTAID.Y| 0x7919 | SR id=0x26 |
| mov.u32 r, %laneid | S2R SR_LANEID | 0x7919 | SR id=0x00 |
| bra TARGET         | BRA           | 0x7947 | offset in bits[63:32], 4-byte units |
| @%p bra TARGET     | @Px BRA       | 0x_947 | pred in bits[14:12] |
| exit / ret         | EXIT          | 0x794d | ctrl extra41=0x03800000 |

---

## 8. Bugs Found and Fixed

The previous implementation used `mode_byte = (log2(delta) << 4) | 0x0c` with `clamp` hardcoded
to `0x20`. This was only correct for `delta=1`. The `warp-reduce,` word called `shfl-bfly,` with
deltas 16, 8, 4, 2, 1, meaning the first four reduction steps emitted garbage instructions.

**Old (wrong)**:
```
delta=16: mode=0x4c, clamp=0x20  -> inst top 16 bits = 0x4c20  (WRONG)
delta=8:  mode=0x3c, clamp=0x20  -> inst top 16 bits = 0x3c20  (WRONG)
```

**New (correct)**:
```
delta=16: mode=0x0e, clamp=0x00  -> inst top 16 bits = 0x0e00  (verified)
delta=8:  mode=0x0d, clamp=0x00  -> inst top 16 bits = 0x0d00  (verified)
```

The fix is applied to both `shfl-bfly,` and `shfl-down,` in `gpu/emit.fs`.

### Bug 2: bar-sync, wrong barrier ID field position

The previous `bar-sync,` word used `bar-id 16 lshift` placing the barrier ID in bits[23:16]
of the instruction word. The correct field is bits[55:48], requiring `bar-id 54 lshift`
(equivalently `bar_id * 64` at byte position 6).

**Old (wrong)**:
```
bar-id=1: 0x000000000001 | 0x7b1d = 0x000000000017b1d  (bit 16 set, WRONG)
```

**New (correct)**:
```
bar-id=1: (1 << 54) | 0x7b1d = 0x0040000000007b1d  (verified ✓)
```

Only kernels using two or more distinct barrier IDs (e.g., nested synchronization) were affected.
The current codebase uses only `bar-sync 0` everywhere, which is correct under both encodings.
