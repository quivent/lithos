# Lithos Runtime — QMD / cbuf0 / Doorbell dispatch

This directory contains the native Lithos kernel launch path: everything needed
to execute a compiled GPU ELF on a GH200 without going through CUDA/libcuda.

All files are `.ls` Lithos source (compositions, flattened by the compiler).

## Files

| File | Purpose |
| --- | --- |
| `qmd.ls` | Build a 528-byte QMD (block/grid/smem/entry_pc). Field offsets from `docs/qmd_fields.md`. |
| `pushbuffer.ls` | Emit the 6-method inline-DMA sequence that submits a QMD via the Compute subchannel. |
| `doorbell.ls` | Publish GPPut at USERD+0x8c and ring the doorbell at USERD+0x90 (with DSB SY). |
| `cbuf0.ls` | Allocate/populate constant buffer 0 — the driver-managed constant block kernels read via LDC. |
| `dispatch.ls` | Top-level `dispatch_kernel` composition that chains QMD → cbuf0 → pushbuffer → doorbell → wait. |

## Status

Working / specified from known offsets:
- QMD block_dim, grid_dim (+ cluster copies), shared_mem (+ derived 0x013c), entry_pc (40-bit split).
- Pushbuffer method-header encoding (sc / opcode / count / subchannel / method).
- Inline-DMA QMD submission (OFFSET_OUT, LINE_LENGTH/COUNT, LAUNCH_DMA=0x41, LOAD_INLINE_DATA x132, SEND_PCAS_A, SEND_SIGNALING_PCAS2_B=0x0a).
- USERD GPPut + doorbell with DSB SY.

Unknown — must be probed before first launch succeeds:
- **register_count offset inside cbuf0.** The QMD probe confirmed it is NOT in
  the QMD body. We have not yet diffed cbuf0 contents between two kernels with
  different `.reg` counts. `cbuf0_set_register_count` is a sentinel that writes
  offset 0 and is marked `TODO(probe):` — do not rely on it until the probe
  experiment runs. See the header comment in `cbuf0.ls`.

## Related references

- `docs/qmd_hopper_sm90.md` — QMD format specification.
- `docs/qmd_fields.md` — empirically-probed byte offsets.
- `docs/language-primitives.md` — Lithos grammar.
- `GSP/rpc_channel.s` — GPFIFO channel creation (`USERD_GPPUT_OFF=0x08C`).
- `tools/qmd_probe_driver.c` — CUDA-based QMD probe (reference; not on the
  native execution path).
