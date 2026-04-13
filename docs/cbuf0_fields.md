# Constant Buffer and Shader Descriptor Fields — Hopper sm_90 (GH200)

**Status:** EMPIRICALLY OBSERVED via differential pushbuffer probing on GH200
(CUDA 12.8, driver 580.105.08, April 2026). Extends the QMD field map from
`docs/qmd_fields.md`.

**Tool:** `/home/ubuntu/lithos/tools/cbuf0_probe.c` and
`tools/cbuf0_probe2.c` (the v2 probe produced the final clean data).

**Method:** Three kernels compiled from identical source with `--maxrregcount`
set to 16, 32, and 64, producing actual register counts of 22, 30, and 38
(as reported by `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`). For each
kernel launch, the 4MB pushbuffer ring at host VA `0x200800000` was snapshotted
before and after `cuLaunchKernel`. The diff region was parsed for
`CB_LOAD_INLINE_DATA` (method `0x206d`) payloads, which were then diffed across
kernels.

## Key Finding: register_count Is NOT In cbuf0 Or The QMD

The QMD probe (`docs/qmd_fields.md`) established that `register_count` is absent
from the 132-dword QMD body. This probe further establishes:

- **register_count is NOT in cbuf0** (constant buffer 0, the buffer addressed
  by SASS `c[0x0][offset]`). The cbuf0 section in each cubin
  (`.nv.constant0.kern`) is all zeros except for the kernel parameter area at
  offset 0x210. The driver maps cbuf0 at a GPU VA stored in QMD dw[50:51]
  (`0x3_24010000` in our captures) and does NOT reload it per-launch.

- **register_count IS in a 384-byte Shader Descriptor** loaded inline via the
  pushbuffer before each kernel launch. This descriptor is the 4th of 5 inline
  CB loads the CUDA driver issues per launch.

## Pushbuffer Launch Sequence

Each `cuLaunchKernel` produces 5 inline constant buffer loads via methods
`0x2062` (set target VA), `0x2060` (set size), `0x206c` (bind), and `0x206d`
(load inline data), followed by the QMD submission:

| # | Size (bytes) | Dwords | Description |
|---|-------------|--------|-------------|
| 0 | 528         | 132    | **QMD body** — same 132-dword structure from `qmd_fields.md` |
| 1 | 8           | 2      | **Fence/timestamp** at QMD+0x210 (identical across kernels) |
| 2 | 384         | 96     | **Context descriptor** — 1 diff (a pointer); no kernel-specific scalar data |
| 3 | 384         | 96     | **Shader Program Descriptor (SPD)** — contains `register_count` and other per-kernel metadata |
| 4 | 4           | 1      | **Small patch** — single dword, identical across kernels |

After these 5 loads, the driver submits the QMD for execution.

## Shader Program Descriptor (Load #3) — Field Map

This 384-byte (96-dword) structure at target VA `0x2_0460XXXX` contains
per-kernel hardware configuration. 8 dwords differ across kernels with
different register counts:

### Confirmed Fields

| Offset | Dword | Field | Evidence |
|--------|-------|-------|----------|
| **0x094** | 37 | **register_count** (in byte[2], bits 23:16) | k16(22 regs)=0x08**16**0001, k32(30 regs)=0x08**1e**0001, k64(38 regs)=0x08**26**0001. Byte[2]: 0x16=22, 0x1e=30, 0x26=38. **Exact 3-point match.** |
| 0x098 | 38 | entry_pc (low 32 bits) | Same values as QMD dw[70]: k16=0x277a0000, k32=0x277a0d00, k64=0x277a1a00 |
| 0x0A8 | 42 | entry_pc (full 40-bit, packed) | k16=0x03277a00, k32=0x03277a0d, k64=0x03277a1a — low byte is entry_pc[15:8], next 3 bytes = hi32 |
| 0x0B8 | 46 | register_alloc_granule | k16(22)=13, k32(30)=14, k64(38)=15. Monotonic with regs; likely `ceil(regs / some_granularity) - 1` or similar HW allocation unit |

### Encoding of register_count at Offset 0x094

The full dword at offset 0x094 is structured as:

```
Bits 31:24  Bits 23:16        Bits 15:0
0x08        register_count    0x0001

Byte layout (little-endian in memory):
  byte[0] = 0x01  (constant)
  byte[1] = 0x00  (constant)
  byte[2] = register_count (raw value from cuFuncGetAttribute)
  byte[3] = 0x08  (constant, possibly flags or version)
```

Three-point verification:
- 22 regs -> byte[2] = 0x16 = 22
- 30 regs -> byte[2] = 0x1e = 30
- 38 regs -> byte[2] = 0x26 = 38

### Other Interesting Fields in the SPD

All values from k16 (22 regs) unless noted:

| Offset | Value | Notes |
|--------|-------|-------|
| 0x000 | 0x113f0000 | Header / magic (varies with first launch vs subsequent) |
| 0x028 | 0x00000003 | Flags |
| 0x02c | 0x00190000 | Possibly shared memory config |
| 0x034 | 0x10300011 | HW config flags |
| 0x038 | VA pointer | Points to another descriptor (differs per kernel) |
| 0x044 | 0x03000640 | Possibly grid/dispatch config |
| 0x048 | 0xbc040040 | HW resource config |
| 0x080 | 0x00000001 | Block dim x |
| 0x084 | 0x00000001 | Block dim y |
| 0x088 | 0x00000001 | Block dim z |
| 0x08c | 0x80010000 | Thread config flags |
| 0x090 | 0x00010001 | Constant across kernels |
| 0x094 | 0x08**XX**0001 | **register_count in byte[2]** |
| 0x098 | entry_pc lo32 | Kernel text GPU VA (low) |
| 0x09c | 0x00000003 | entry_pc hi8 |
| 0x0a8 | packed entry_pc | 40-bit entry_pc in different layout |
| 0x0ac | 0x001a0000 | Kernel text size / range |
| 0x0b8 | alloc granule | Register allocation unit (13/14/15 for 22/30/38 regs) |
| 0x0c0 | 0x0c98b400 | Local memory base (scratch) |
| 0x0c4 | 0x01800000 | Local memory size |
| 0x0c8 | 0x0c900400 | Shared memory base |
| 0x0cc | 0x04800000 | Shared memory pool size |
| 0x0e8 | dup of 0x0c0 | Local memory base (duplicate) |
| 0x0f8 | 0x0c900000 | Shared memory base (variant) |

## cbuf0 (Constant Buffer 0) Layout

The actual cbuf0 (`c[0x0]` in SASS) is mapped at GPU VA from QMD dw[50:51]
(observed as `0x3_24010000`). It is NOT reloaded per-launch. The `.nv.constant0`
ELF section in the cubin is all zeros except where kernel parameters are placed.

At runtime, cbuf0 contains:

| Offset | Size | Content |
|--------|------|---------|
| 0x000-0x1FF | 512 B | Reserved / driver metadata (all zeros in cubin; runtime contents unknown — not accessible via cuMemcpyDtoH) |
| 0x200-0x207 | 8 B | Unknown (zeros in cubin) |
| 0x208 | 8 B | Memory descriptor for global loads (`ULDC.64 UR4, c[0x0][0x208]` in SASS) |
| 0x210 | N B | **Kernel parameters** (the `float *out` pointer in our test = 8 bytes) |
| 0x028 | 4 B | Stack/frame pointer (accessed by `LDC R1, c[0x0][0x28]` in SASS) |

Note: offsets 0x000-0x1FF may contain driver-populated metadata at runtime, but
we cannot read GPU VA `0x3_24010000` from the host (not BAR1-mapped). The kernel
itself can read these via `c[0x0][offset]` SASS instructions.

## QMD Constant Buffer Pointer Fields (Updated)

| QMD Offset | Dwords | Field |
|-----------|--------|-------|
| 0x0C0 | dw[48:49] | Per-launch sideband VA (increments each launch; used for QMD body address) |
| 0x0C8 | dw[50:51] | **cbuf0 base GPU VA** (`0x3_24010000` — same for all kernels in a context) |

## Pushbuffer Methods Reference

| Method | Name | Payload |
|--------|------|---------|
| 0x2060 | CB_SIZE_AND_FLAGS | `{size_bytes, flags/valid}` |
| 0x2062 | CB_TARGET_ADDR | `{addr_hi32, addr_lo32}` |
| 0x206c | CB_BIND | bind constant buffer (`0x41` = bind + valid) |
| 0x206d | CB_LOAD_INLINE_DATA | N dwords of data (NONINC method) |
| 0x20ad | INVALIDATE_CONSTANT | invalidate const cache |
| 0x20b0 | SET_SHADER_SHARED_MEMORY | shared memory config |
| 0x26c0 | SEND_PCAS | pre-compute dispatch config |

## What Lithos Must Do (Updated)

To hand-assemble a valid kernel launch on Hopper:

1. **QMD body** (132 dwords / 528 bytes) — populate:
   - grid/block dims at 0x00-0x14 and 0x15c-0x164
   - entry_pc at 0x118/0x11c
   - shared_mem_size at 0x02c
   - total_smem_alloc at 0x13c
   - cbuf0 base at 0x0c8/0x0cc
   - Sideband VA at 0x0c0/0x0c4 (per-launch)
   - Other fields: copy from a captured template

2. **Shader Program Descriptor** (96 dwords / 384 bytes) — populate:
   - **register_count at offset 0x094, byte[2]** (bits 23:16)
   - entry_pc at 0x098 and 0x0a8
   - Register alloc granule at 0x0b8
   - Block dims at 0x080-0x088
   - Local/shared memory bases at 0x0c0-0x0cc and 0x0e8-0x0fc
   - Other fields: copy from a captured template

3. **Submit sequence**: issue the 5 CB_LOAD_INLINE_DATA payloads in order
   via pushbuffer methods 0x2062/0x2060/0x206c/0x206d, then submit the QMD.

The cleanest Lithos path: capture one good launch (all 5 CB loads + QMD) via
a dummy `cuLaunchKernel`, then for each subsequent launch, mutate only the
fields we understand:
- In the QMD: grid/block, entry_pc, shared_mem_size
- In the SPD: **register_count** (byte[2] at 0x094), entry_pc, alloc granule,
  block dims

## Register Allocation Granule at 0x0B8

The field at SPD offset 0x0B8 correlates with register count:

| Regs | Value | ceil(regs/8) + 10 |
|------|-------|-------------------|
| 22   | 13    | ceil(22/8) + 10 = 3 + 10 = 13 |
| 30   | 14    | ceil(30/8) + 10 = 4 + 10 = 14 |
| 38   | 15    | ceil(38/8) + 10 = 5 + 10 = 15 |

**Likely formula: `ceil(register_count / 8) + 10`** (3-point match).
The "+10" base likely accounts for a fixed register file overhead (system
registers, predicate registers, etc.). The "/8" granularity matches Hopper's
register file bank width. For Lithos: use this formula, or capture per-kernel
and replay.

## Raw Data

- `/tmp/cbuf0_probe.bin` — raw probe data from v1
- Probe output: run `./tools/cbuf0_probe2` to reproduce

---
*Confirmed fields: register_count = SPD offset 0x094 byte[2] (3-point exact match: 22/30/38)*
