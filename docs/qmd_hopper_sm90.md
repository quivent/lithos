# QMD Structure — Hopper sm_90 (GH100/GH200)

**Version:** 4 (GV100=2, GA100=3, GH100=4)  
**Size:** 256 bytes (0x100), 64 dwords, packed bitfields  
**Alignment:** 256-byte aligned in GPU VA space

## What is the QMD

Queue Meta Descriptor. The GPU CE reads it before dispatching a compute
grid. Lithos writes one per kernel launch directly to the GPFIFO ring;
no syscall involved. `SET_INLINE_QMD` or `SEND_PCAS_A` hands the host
engine the QMD address (`qmd_va >> 8`).

## Field Map (derived from Kepler `cla0c0qmd.h` + `.nv.info` EIATTR analysis)

```
Offset  Width  Field
0x000   32     qmd_version       = 4 (Hopper)
0x004   32     sm_global_caching_enable / misc flags
0x008   32     entry_pc[31:0]    — low 32 bits of kernel .text VA
0x00C   32     entry_pc[63:32]   — high bits; full 64-bit PC
0x010   16     register_count    — EIATTR_MAXREG_COUNT from .nv.info
0x012   16     barrier_count     — EIATTR_EXPL_CACHING / barrier fields
0x014   32     shared_mem_size   — smem bytes (dynamic + static, 128B aligned)
0x018   32     block_dim_x       — blockDim.x
0x01C   16     block_dim_y
0x01E   16     block_dim_z
0x020   32     grid_dim_x        — gridDim.x
0x024   32     grid_dim_y
0x028   32     grid_dim_z
0x02C   32     cluster_dim_xyz   — Hopper-new; [7:0]=x [15:8]=y [23:16]=z; set 1,1,1 if unused
0x030   64     cbuf0_base_addr   — constant bank 0 VA (kernel param buffer)
0x038   32     cbuf0_size        — size in bytes of param buffer (EIATTR_PARAM_CBANK)
0x03C   32     cbuf0_flags       — valid + size encoding per Kepler convention
0x040   64     cbuf1_base_addr   — constant bank 1 (driver constants, set 0 if unused)
...
0x0A0   32     tma_descriptor_ptr[31:0]   — Hopper TMA; 0 for GEMV
0x0A4   32     tma_descriptor_ptr[63:32]
0x0F0   32     qmd_reserved_end
```

Fields above 0x040 not needed by GEMV: zero them.

## GEMV Launch — Fields Lithos Must Fill

For a GEMV kernel (one CTA per output row, or BLOCK_M rows, 128–256
threads):

| Field | Source | Notes |
|---|---|---|
| `qmd_version` | constant 4 | Hopper identifier |
| `entry_pc` | cubin base VA + `.text.gemv_kernel` offset | from ELF symbol |
| `register_count` | `EIATTR_MAXREG_COUNT` in `.nv.info` | typically 64–128 for GEMV |
| `shared_mem_size` | `EIATTR_MAX_THREADS + smem` from `.nv.info` | tile buffer |
| `block_dim_x/y/z` | launch config | e.g. 256, 1, 1 |
| `grid_dim_x/y/z` | `ceil(M / BLOCK_M)`, 1, 1 | M = output rows |
| `cluster_dim_xyz` | 1, 1, 1 | no cluster launch for GEMV |
| `cbuf0_base_addr` | param buffer GPU VA | holds {A_ptr, x_ptr, y_ptr, M, N, lda} |
| `cbuf0_size` | sizeof(KernelParams) | EIATTR_PARAM_CBANK gives exact bytes |
| everything else | 0 | zero-fill the 256-byte block first |

## Gap Note

Hopper QMD field layout is **not in open-gpu-kernel-modules**. Only
`cla0c0qmd.h` (Kepler) is shipped. The offsets above are reconstructed
from: Kepler header structure, GV100→GA100→GH100 delta analysis via
`nvdisasm -qmd` output patterns, and `.nv.info` EIATTR cross-reference.
Empirical validation against a running GH200 workload (strace of
pushbuffer writes) is required before production use. See
`nvidia_driver_abi.md` §Unknowns item 1.

---
*dance vector: [write_qmd=1, doorbell=1, syscall=0, latency=min]*
