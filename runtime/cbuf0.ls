\\ cbuf0.ls — Constant buffer 0 construction
\\
\\ cbuf0 is a GPU-resident constant buffer the driver populates alongside every
\\ QMD. The kernel reads it via LDC c[0x0][offset] instructions. It contains:
\\   - kernel parameters (the "param block") starting at offset 0x210
\\   - driver-populated metadata including the register count
\\   - other launch-time scalars (grid dims, block dims, nctaid, etc.)
\\
\\ PROBE COMPLETE (April 2026):
\\   register_count is NOT in cbuf0. The cbuf0 section (.nv.constant0) is all
\\   zeros except for kernel parameters at offset 0x210, and cbuf0 is not
\\   reloaded per-launch. register_count belongs in the Shader Program Descriptor
\\   (SPD), which is the 4th of 5 inline CB loads the driver issues per launch.
\\   SPD offset 0x094, bits 23:16. See docs/cbuf0_fields.md for the full
\\   5-part pushbuffer sequence and 3-point probe verification.

\\ Bump allocator state lives at cbuf0_alloc_state (a 64-bit GPU VA cursor into
\\ our BAR4 HBM pool, set up at runtime init). Caller passes an output slot.
cbuf0_alloc size out_gpu_va_ptr :
    cur → 64 cbuf0_alloc_state
    ← 64 out_gpu_va_ptr cur
    next cur + size
    \\ 256-byte align the next allocation (cbuf0 must be 256B-aligned for LDC)
    next next + 255
    mask next / 256 * 256
    ← 64 cbuf0_alloc_state mask

\\ OBSOLETE: register_count belongs in SPD offset 0x094, not cbuf0. See pushbuffer.ls pb_emit_spd.
\\ Formula: (0x08 << 24) | (reg_count << 16) | 0x0001. Ref: docs/cbuf0_fields.md.
cbuf0_set_register_count cbuf0_ptr count :
    \\ OBSOLETE: register_count belongs in SPD offset 0x094, not cbuf0. See pushbuffer.ls pb_emit_spd.
    ← 32 cbuf0_ptr + 0 count

\\ Param block copy. NVIDIA convention: user params start at cbuf0 offset 0x210.
\\ data is a host-visible source pointer; we stream nbytes as 32-bit words.
cbuf0_set_param_block cbuf0_ptr offset data nbytes :
    for i 0 nbytes 4
        w → 32 data + i
        ← 32 cbuf0_ptr + offset + i w

\\ Ensure writes land in HBM before QMD submission. Host-side: DSB SY.
cbuf0_finalize cbuf0_ptr :
    membar_sys
