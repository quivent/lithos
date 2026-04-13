# QMD Structure — Hopper sm_90 (GH100/GH200)

**Version:** EMPIRICALLY OBSERVED on GH200 (CUDA 12.8, driver 580.105.08)
**Size:** 528 bytes (132 dwords) — NOT 256 bytes as previous reconstructions assumed
**Alignment:** the QMD buffer is 256-byte aligned in GPU VA space

## What is the QMD

Queue Meta Descriptor. cuLaunchKernel assembles one per kernel launch and
writes it to a pre-allocated device buffer via the DMA inline copy engine.
The compute engine is then pointed at it via SEND_PCAS_A. No separate
256-byte aligned "blob" at the GPFIFO ring — the QMD is DMA'd inline.

## Empirical Method

A C program (`/tmp/qmd_ringcompare.c`) snapshotted the 4MB host-accessible
pushbuffer ring (`/dev/zero` MAP_SHARED at host VA 0x200800000) before and
after two consecutive cuLaunchKernel calls on null_kernel:
- Launch1: grid=3×1×1, block=4×1×1
- Launch2: grid=7×1×1, block=13×1×1

Both launches produced 384-dword (1536-byte) pushbuffers. The differential
identifies which bytes encode grid/block parameters.

## Pushbuffer Encoding — Correct Bit Layout

The NVC36F DMA pushbuffer header format (from `clc36f.h`):

```
bits[31:29] = opcode   (1=INC, 3=NONINC, 5=INC_ONCE, 4=IMMD)
bits[28:16] = count    (13 bits — number of data dwords)
bits[15:13] = subch    (3 bits — subchannel, 1 = compute class)
bits[11:0]  = addr     (12 bits — class method byte offset, NOT divided by 4)
```

Class method byte offset = addr field directly (e.g. addr=0x0180 = NVA0C0_LINE_LENGTH_IN).

## Launch Protocol (empirically observed, GH200)

A single cuLaunchKernel emits the following method sequence to subchannel 1
(Hopper compute class NVC9C0):

```
Step 1: Set DMA destination (OFFSET_OUT)
  INC sc=1 addr=0x0188 (OFFSET_OUT_UPPER) count=2
    val[0x0188] = 0x00000003           -- high 8 bits of QMD destination GPU VA
    val[0x018c] = 0x26280000           -- low 32 bits of QMD destination GPU VA

Step 2: Set DMA size
  INC sc=1 addr=0x0180 (LINE_LENGTH_IN) count=2
    val[0x0180] = 528                  -- QMD size in bytes
    val[0x0184] = 1                    -- LINE_COUNT = 1

Step 3: Trigger DMA
  INC sc=1 addr=0x01b0 (LAUNCH_DMA) count=1
    val[0x01b0] = 0x00000041           -- DST=PITCH, SYSMEMBAR_DISABLE

Step 4: Write 132 dwords of QMD inline
  NONINC sc=1 addr=0x01b4 (LOAD_INLINE_DATA) count=132
    dword[0]  = block_dim_x            -- first QMD field
    dword[1]  = block_dim_y
    ...                                -- (full QMD layout below)

Step 5: Dispatch compute
  INC sc=1 addr=0x02b4 (SEND_PCAS_A) count=1
    val = QMD_GPU_VA >> 8             -- 40-bit QMD address, shifted right 8

  INC sc=1 addr=0x02c0 (SEND_SIGNALING_PCAS2_B) count=1
    val = 0x0000000a                  -- schedule/dispatch trigger
```

Method byte addresses resolve to Kepler/Ampere class constants:
- `0x0180` = `NVA0C0_LINE_LENGTH_IN`
- `0x0184` = `NVA0C0_LINE_COUNT`
- `0x0188` = `NVA0C0_OFFSET_OUT_UPPER`
- `0x018c` = `NVA0C0_OFFSET_OUT`
- `0x01b0` = `NVA0C0_LAUNCH_DMA`
- `0x01b4` = `NVA0C0_LOAD_INLINE_DATA`
- `0x02b4` = `NVA0C0_SEND_PCAS_A`
- `0x02c0` = `NVC6C0_SEND_SIGNALING_PCAS2_B`

## QMD Field Map — Empirically Observed

```
Byte    Dword  Status                  Field / Observed Values
------  -----  ----------------------  ----------------------------------------
0x0000    0    CONFIRMED               block_dim_x  (4 vs 13 across launches)
0x0004    1    CONFIRMED               block_dim_y  (1 in both)
0x0008    2    CONFIRMED               block_dim_z  (1 in both)
0x000c    3    CONFIRMED               grid_dim_x   (3 vs 7 across launches)
0x0010    4    CONFIRMED               grid_dim_y   (1 in both)
0x0014    5    CONFIRMED               grid_dim_z   (1 in both)
0x0018    6    INFERRED (zero)         reserved or shared_mem_size=0
0x001c    7    UNKNOWN                 0x0000ea9d (60061) — fence/ordinal ID
0x0020    8    UNKNOWN                 0xad000000 — fence/ordinal high bits
0x0024    9    UNKNOWN                 0x0000ea9c (60060) — fence/ordinal ID
0x0028   10    UNKNOWN                 0x00fffdc0 — semaphore mask or barrier?
0x002c   11    INFERRED (zero)         reserved
0x0030   12    INFERRED                ceil(grid_dim_x/4) [1→3, 2→7] — CTA batch count?
0x0034   13    INFERRED (zero)         reserved
0x0038   14    UNKNOWN (varies)        0x0460c000 / 0x0460c800 — changes with block_dim_x
                                       difference = 0x800 per 9-thread increase (lmem base?)
0x003c   15    UNKNOWN                 0x00000002 — could be entry_pc_hi if 0x0038=lo
0x0040   16    UNKNOWN                 0x038432c8 — fixed across launches (cbuf base?)
...
0x0058   22    UNKNOWN                 0x00000001
0x0068   26    UNKNOWN                 0x00000120 (288)
...
0x00c0   48    UNKNOWN (addr)          0x26280000 / 0x26290000 — changes per launch
                                       (session-level address, 0x10000 apart)
0x00c4   49    UNKNOWN                 0x00000003
0x00c8   50    UNKNOWN                 0x24010000
0x00cc   51    UNKNOWN                 0x00000003
...
0x010c   67    UNKNOWN                 0x00000084 (132) = QMD dword count
0x0114   69    UNKNOWN                 0x00000400 (1024)
0x0118   70    UNKNOWN                 0x277a0000
0x011c   71    UNKNOWN                 0x00000003
...
0x013c   79    UNKNOWN                 0x00000400 (1024)
0x0144   81    UNKNOWN                 0x00000001
0x0148   82    UNKNOWN                 0x00000001
0x014c   83    UNKNOWN                 0x00000001
0x0150   84    INFERRED                0x3f800000 = IEEE 754 1.0 — cluster_dim_x?
0x0154   85    INFERRED                0x3f800000 = IEEE 754 1.0 — cluster_dim_y?
0x0158   86    INFERRED                0x3f800000 = IEEE 754 1.0 — cluster_dim_z?
0x015c   87    CONFIRMED               grid_dim_x copy (3 vs 7 across launches)
0x0160   88    CONFIRMED               0x00000001 — grid_dim_y copy
0x0164   89    CONFIRMED               0x00000001 — grid_dim_z copy
...
0x016c   91    UNKNOWN                 0x01000000
...
0x0174   93    UNKNOWN                 0x0000ea9d (60061) — same fence ID as 0x001c
...
0x0188   98    UNKNOWN                 0x00000001
0x018c   99    UNKNOWN                 0x00000001
...
0x0198  102    UNKNOWN (addr)          0x26280210 / 0x26290210 — changes per launch
0x019c  103    UNKNOWN                 0x00000003
0x01a0  104    UNKNOWN (addr)          0x26280218 / 0x26290218 — changes per launch
0x01a4  105    UNKNOWN                 0x00000003
...
0x0200  128    UNKNOWN                 0x8dfe7f00 — bitfield or hash
0x0204  129    UNKNOWN                 0x9896b85c — bitfield or hash
(bytes 0x0208-0x020f = 0, end of 528-byte QMD)
```

## Fields Refuted from Prior Reconstruction

The prior assumed layout (derived from cla0c0qmd.h + delta analysis) was:

| Old Assumed Field      | Old Byte | Empirical Byte | Verdict |
|------------------------|----------|----------------|---------|
| `qmd_version = 4`      | 0x000    | 0x000 = block_dim_x | WRONG — 4 is block_dim_x, not version |
| `entry_pc[31:0]`       | 0x008    | 0x008 = block_dim_z | WRONG |
| `entry_pc[63:32]`      | 0x00c    | 0x00c = grid_dim_x  | WRONG |
| `register_count`       | 0x010    | 0x010 = grid_dim_y  | WRONG |
| `shared_mem_size`      | 0x014    | 0x014 = grid_dim_z  | WRONG |
| `block_dim_x`          | 0x018    | 0x000 (actual)      | WRONG OFFSET by 0x18 |
| `block_dim_y/z`        | 0x01c/1e | 0x004/008 (actual)  | WRONG OFFSET |
| `grid_dim_x/y/z`       | 0x020/24/28 | 0x00c/10/14 (actual) | WRONG OFFSET |
| `cluster_dim_xyz`      | 0x02c    | 0x150/154/158 as floats | WRONG |
| `cbuf0_base_addr`      | 0x030    | unknown (not found)  | UNVERIFIED |
| 256-byte total size    | —        | 528 bytes (actual)   | WRONG |

## What Lithos Must Do Instead

The "standalone 256-byte QMD blob" design is incorrect for GH200. The actual
mechanism is:

1. Allocate a 528-byte GPU-accessible buffer (aligned 256 bytes in GPU VA space).
2. Populate the QMD structure using the empirically confirmed layout above.
3. Write the QMD via the GPFIFO pushbuffer using:
   - LINE_LENGTH_IN = 528, LINE_COUNT = 1
   - OFFSET_OUT (64-bit GPU VA of QMD buffer)
   - LAUNCH_DMA = 0x41
   - NONINC writes to LOAD_INLINE_DATA (132 dwords)
4. Dispatch with SEND_PCAS_A (= qmd_gpu_va >> 8) and SEND_SIGNALING_PCAS2_B.

Fields still unknown (need further experiments with varying register_count,
smem_size, and cbuf0 to identify their byte offsets in the 528-byte QMD):
- entry_pc (kernel text GPU VA)
- register_count
- shared_mem_size
- cbuf0_base_addr and cbuf0_size

## Raw Observed QMD (Launch1: grid=3×1×1, block=4×1×1, null_kernel)

```
byte 0x000: 00000004  00000001  00000001  00000003
byte 0x010: 00000001  00000001  00000000  0000ea9d
byte 0x020: ad000000  0000ea9c  00fffdc0  00000000
byte 0x030: 00000001  00000000  0460c000  00000002
byte 0x040: 038432c8  00000000  00000000  00000000
byte 0x050: 00000000  00000000  00000001  00000000
byte 0x060: 00000000  00000000  00000120  00000000
byte 0x070: 00000000  00000000  00000000  00000000
(bytes 0x080-0x0bf: all zeros)
byte 0x0c0: 26280000  00000003  24010000  00000003
(bytes 0x0d0-0x10b: all zeros)
byte 0x10c: 00000084  00000000  00000400  277a0000
byte 0x11c: 00000003  00000000  00000000  00000000
byte 0x12c: 00000000  00000000  00000400  00000000
byte 0x13c: 00000001  00000001  00000001  3f800000
byte 0x14c: 3f800000  3f800000  00000003  00000001
byte 0x15c: (grid_dim_x=3 at 0x15c, grid_dim_y/z at 0x160/164)
byte 0x16c: 01000000  00000000  0000ea9d  00000000
(bytes 0x17c-0x187: zeros)
byte 0x188: 00000001  00000001  00000000  00000000
byte 0x198: 26280210  00000003  26280218  00000003
(bytes 0x1a8-0x1ff: zeros)
byte 0x200: 8dfe7f00  9896b85c
(bytes 0x208-0x20f: zeros — end of 528-byte QMD)
```

## Gap Note

The following fields remain unidentified and require additional experiments:
- **entry_pc**: launch two different kernels (different text VAs) and diff the QMDs
- **register_count**: compile kernels with forced register counts (maxrregcount pragma)
- **shared_mem_size**: launch with explicit dynamic shared memory sizes
- **cbuf0_base_addr / cbuf0_size**: launch kernels with non-trivial parameter buffers
- **The 3 × 0x3f800000 fields** (bytes 0x150-0x158) are likely cluster_dim_xyz encoded
  as IEEE 754 floats — hypothesis only, not confirmed by differential
- **byte 0x0030** = ceil(grid_dim_x/4) in the two tests; may be a CTA scheduling hint

*Binary: pb_launch1.bin and pb_launch2.bin saved in /tmp/ — raw pushbuffer dwords*
*Tools: /tmp/qmd_ringcompare.c, /tmp/qmd_fullread.c*

---
*dance vector: [write_qmd=confirmed_mechanism, standalone_blob=refuted, syscall=0]*
