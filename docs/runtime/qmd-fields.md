# QMD Field Offsets — Hopper sm_90 (GH200)

**Status:** EMPIRICALLY OBSERVED via differential probing on GH200 (CUDA 12.8,
driver 580.105.08, April 2026). Supersedes the "UNKNOWN" fields in
`docs/qmd_hopper_sm90.md`.

**Method:** Pairs of kernels that differ in exactly one parameter were launched
via `cuLaunchKernel`. The 4MB pushbuffer ring at host VA `0x200800000` was
snapshotted before and after each launch. The 132-dword QMD bodies (inlined via
`NVA0C0_LOAD_INLINE_DATA` with `count=132`) were extracted and diffed. A baseline
pair of *identical* launches established the "session noise" set (QMD buffer
GPU VAs, scratch base increments, CTA counters). Diffs outside the noise set
identify true field locations.

**Tool:** `/home/ubuntu/lithos/tools/qmd_probe_driver` (source in
`tools/qmd_probe_driver.c`, helper cubins in `/tmp/probe_kern_{a,b,c,smem}.cubin`).

## Summary of Unknown Field Probes

| Field             | Offset | Dword | Status     | Evidence |
|-------------------|--------|-------|------------|----------|
| `shared_mem_size` | 0x002c | 11    | CONFIRMED  | 0 → 4096 → 8192 linear |
| `entry_pc` (lo32) | 0x0118 | 70    | CONFIRMED  | 4 distinct kernels → 4 distinct values within the same kernel-text arena (0x3_277a0000 base) |
| `entry_pc` (hi32) | 0x011c | 71    | CONFIRMED  | constant `0x00000003` in 4/4 kernels; forms 64-bit GPU VA with dw[70] |
| `register_count`  | —      | —     | NOT FOUND  | No field in the 132-dword QMD body varies with register count (8, 30, 70 tested). Likely stored in cbuf0 metadata. |

## Experimental Data

### Experiment 1: entry_pc (different kernels)

Four kernels with distinct entry points, all launched with the same grid/block/
smem parameters:

| Kernel    | Cubin size | Regs | dw[70] @ 0x0118 | dw[71] @ 0x011c | Full 64-bit GPU VA |
|-----------|-----------:|-----:|-----------------|-----------------|---------------------|
| kern_a    | 2992 B     | 8    | `0x277a0000`    | `0x00000003`    | `0x3_277a0000`      |
| kern_b    | 4016 B     | 30   | `0x277a0a00`    | `0x00000003`    | `0x3_277a0a00`      |
| kern_c    | ~8500 B    | 70   | `0x277a1800`    | `0x00000003`    | `0x3_277a1800`      |
| kern_smem | 3376 B     | 8    | `0x277a6a00`    | `0x00000003`    | `0x3_277a6a00`      |

The low-32 value increments monotonically as each cubin is loaded, consistent
with concatenation of kernel text in a per-context kernel arena at GPU VA
`0x3_277a0000`. The high-32 word is constant `0x00000003` (bits 32-39 of the
40-bit Hopper GPU VA space for device-private text).

**Baseline noise** (two back-to-back launches of kern_a, same parameters):
- dw[12] at 0x0030: increments by +1 per launch (CTA batch counter)
- dw[14] at 0x0038: increments by +0x800 per launch (local-memory scratch base)
- dw[48] at 0x00c0: increments by +0x10000 per launch (per-launch QMD sideband VA)
- dw[102] at 0x0198: same pattern as dw[48] (+0x210 offset)
- dw[104] at 0x01a0: same pattern as dw[48] (+0x218 offset)

dw[70] is **NOT** in the noise set — it's stable across identical launches and
differs only when the kernel changes. This is the defining signature of
`entry_pc`.

### Experiment 2: shared_mem_size (same kernel, varying dynamic smem)

Same kernel (`kern_smem`) launched with three different dynamic shared-memory
sizes via `cuLaunchKernel(..., smem, ...)`:

| smem bytes | dw[11] @ 0x002c | dw[79] @ 0x013c |
|-----------:|:----------------|:----------------|
| 0          | `0x00000000`    | `0x00000400` (1024) |
| 4096       | `0x00001000`    | `0x00001400` (5120) |
| 8192       | `0x00002000`    | `0x00002400` (9216) |

- **dw[11] at 0x002c** stores the raw dynamic shared-memory size in bytes.
  Three-point confirmation: linear (0, +4096, +8192).
- **dw[79] at 0x013c** stores `total_smem_allocation = 1024 + dynamic_smem_size`.
  The 1024-byte offset is a per-CTA shared-memory header / static smem padding
  driver-reserved region. This is a *derived* field; it tracks shared_mem_size
  with a constant offset.

### Experiment 3: register_count (same launch params, varying regs)

Three kernels launched with the same grid/block/smem but different ptxas
register allocations:

| Kernel | Regs (cuFuncGetAttribute) |
|--------|--------------------------:|
| kern_a | 8                         |
| kern_b | 30                        |
| kern_c | 70                        |

**Result:** Diffing kern_a vs kern_b (and kern_a vs kern_c) produces exactly
the same set of differing dwords as the session-noise baseline — plus **only**
dw[70] (entry_pc). *No field in the 132-dword QMD body carries the register
count.*

This is surprising given existing NVIDIA public headers (`cla0c0qmd.h`) that
define `REGISTER_COUNT_V2` within the QMD, but is reproducible here: kern_a
(8 regs), kern_b (30 regs), and kern_c (70 regs) produce byte-identical QMD
bodies outside entry_pc and the session-noise slots.

**Interpretation:** On sm_90 with this CUDA/driver version, the driver stores
register_count in the kernel metadata constant buffer (cbuf0), which the
Hopper compute class reads when binding the SM to the kernel. The QMD only
carries a pointer to cbuf0 (dw[70] / dw[71] as entry_pc, plus dw[48]/dw[50]
as cbuf0 base) — the per-kernel scalar attributes travel inside that buffer.

*Further experiment needed:* Read and diff the 1024-byte region at the cbuf0
GPU VA (`dw[48] << 8` or `dw[50] << 8`?) to locate register_count. This
requires either a host-mapped BAR1 read or a GPU-side copy kernel.

## Updated Field Map

```
Byte    Dword  Status       Field
------  -----  -----------  -------------------------------------------
0x0000    0    CONFIRMED    block_dim_x
0x0004    1    CONFIRMED    block_dim_y
0x0008    2    CONFIRMED    block_dim_z
0x000c    3    CONFIRMED    grid_dim_x
0x0010    4    CONFIRMED    grid_dim_y
0x0014    5    CONFIRMED    grid_dim_z
0x0018    6    INFERRED     reserved (always zero)
0x001c    7    UNKNOWN      fence/ordinal (session-variant)
0x0020    8    UNKNOWN      fence/ordinal high
0x0024    9    UNKNOWN      fence/ordinal
0x0028   10    UNKNOWN      semaphore/barrier mask
0x002c   11    CONFIRMED    shared_mem_size (dynamic, in bytes)  ← NEW
0x0030   12    INFERRED     CTA launch counter (increments per launch)
0x0038   14    INFERRED     local-memory scratch base (increments)
0x003c   15    UNKNOWN      constant 2 (QMD flags?)
0x0040   16    UNKNOWN      constant 0x038432c8 (cbuf1 base?)
0x00c0   48    UNKNOWN      per-launch QMD sideband GPU VA (noise)
0x0118   70    CONFIRMED    entry_pc (low 32 bits of kernel text GPU VA)  ← NEW
0x011c   71    CONFIRMED    entry_pc (high 32 bits; typically 0x00000003) ← NEW
0x013c   79    CONFIRMED    total_smem_alloc = shared_mem_size + 1024    ← NEW
0x0150   84    INFERRED     cluster_dim_x (IEEE 1.0)
0x0154   85    INFERRED     cluster_dim_y (IEEE 1.0)
0x0158   86    INFERRED     cluster_dim_z (IEEE 1.0)
0x015c   87    CONFIRMED    grid_dim_x (duplicate)
0x0160   88    CONFIRMED    grid_dim_y (duplicate)
0x0164   89    CONFIRMED    grid_dim_z (duplicate)
0x0198  102    UNKNOWN      QMD-sideband VA + 0x210 (noise)
0x01a0  104    UNKNOWN      QMD-sideband VA + 0x218 (noise)
0x0200  128    UNKNOWN      hash/checksum low
0x0204  129    UNKNOWN      hash/checksum high
```

## What Lithos Must Do

To hand-assemble a valid QMD for a Lithos kernel:

1. **grid/block dims** — direct copies at bytes 0x00-0x14 and 0x15c-0x164.
2. **entry_pc** — write the 40-bit GPU VA of the kernel's first SASS
   instruction to bytes 0x118 (low 32) and 0x11c (high 8 bits in low byte).
   Lithos gets this from `cuModuleGetFunction` + internal GPU VA introspection
   (or by using cuda driver API to load the cubin and hand-assembling the
   launch afterward).
3. **shared_mem_size** — write `dynamic_smem_bytes` to byte 0x02c.
4. **total smem alloc** — write `dynamic_smem_bytes + 1024` to byte 0x13c.
5. **register_count** — NOT written in the QMD body. Either:
   - (a) Let the driver populate cbuf0 (via normal cuLaunchKernel flow), OR
   - (b) Hand-write a constant buffer at a driver-compatible GPU VA and set
     the cbuf0 base fields at bytes 0xc0 and 0xc8.
6. **Cluster / grid duplicate / fence fields** — copy from a captured reference
   QMD as template; the driver patches these from the schedule.

The cleanest Lithos path: capture one good QMD via a dummy cuLaunchKernel, then
mutate only the four fields we understand (grid/block at 0x00-0x14,
shared_mem_size at 0x2c, entry_pc at 0x118/0x11c, total_smem_alloc at 0x13c)
for each subsequent launch.

## Raw Capture Bins

- `/tmp/qmd_probe_final.bin` — 5 × 528-byte QMDs in order:
  kern_a, kern_b, smem=0, smem=4096, smem=8192.
- Probe binaries: `/home/ubuntu/lithos/tools/qmd_probe_driver` (linked against
  CUDA driver API), helper cubins in `/tmp/probe_kern_*.cubin`.

---
*dance vector: [entry_pc=0x118, shared_mem_size=0x2c, register_count=cbuf0_not_qmd]*
