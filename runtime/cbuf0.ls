\\ cbuf0.ls — Constant buffer 0 construction
\\
\\ cbuf0 is a GPU-resident constant buffer the driver populates alongside every
\\ QMD. The kernel reads it via LDC c[0x0][offset] instructions. It contains:
\\   - kernel parameters (the "param block") starting at offset 0x210
\\   - driver-populated metadata including the register count
\\   - other launch-time scalars (grid dims, block dims, nctaid, etc.)
\\
\\ CRITICAL UNKNOWN:
\\   register_count offset inside cbuf0 is NOT YET PROBED.
\\   The QMD probe (tools/qmd_probe_driver.c) confirmed register_count is NOT
\\   in the QMD body — but we did not diff cbuf0 contents between kernels with
\\   different register counts. TODO(probe): launch two kernels — one with
\\   .reg .b32 r<8> and one with .reg .b32 r<64> — dump cbuf0 for each, diff.
\\   The differing dword is the register_count offset. Until then, this field
\\   is left unwritten and the GPU will fault or use a stale value.

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

\\ TODO(probe): replace REGISTER_COUNT_OFFSET with the real value once diffed.
\\ Writing to cbuf0_ptr + 0 until known — this is INTENTIONALLY a sentinel so a
\\ grep on "TODO(probe): register_count" finds every consumer that needs updating.
cbuf0_set_register_count cbuf0_ptr count :
    \\ TODO(probe): register_count offset — see cbuf0.ls header comment.
    \\ DO NOT USE this composition until probe completes.
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
