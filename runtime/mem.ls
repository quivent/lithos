\\ mem.ls — BAR4 bump allocator
\\
\\ On GH200, BAR4 IS coherent HBM. CPU VA == GPU VA via NVLink-C2C ATS.
\\ No dma_alloc_coherent, no kernel module, no DMA mapping, no copies.
\\ The CPU writes a byte at VA X; the GPU reads that same byte at GPU VA X.
\\
\\ This file replaces the CUDA driver memory surface:
\\   cuMemAlloc       → mem_alloc
\\   cuMemFree        → no-op (bump allocator; reset at shutdown)
\\   cuMemcpyHtoD/DtoH → mem_copy (plain memcpy, shared memory)
\\   cuMemsetD32      → mem_set32 (straight store loop)
\\
\\ Allocator state lives in three 64-bit cells in CPU memory, tracking the
\\ bump pointer into BAR4. Runtime init writes mem_bar4_base and mem_bar4_size
\\ once, then mem_bar4_next advances on each allocation.
\\
\\   mem_bar4_base — start of BAR4 mapping (GPU/CPU VA; they match)
\\   mem_bar4_size — total BAR4 size in bytes (128 GB on GH200)
\\   mem_bar4_next — next free offset within BAR4 (relative to base)
\\
\\ SYNTAX NOTE: Other runtime/*.ls files reference state cells by name and
\\ access them via `→ 64 name` / `← 64 name val`. We follow the same pattern
\\ here. The cells themselves are expected to be provisioned by the compiler
\\ or by an init stub (see cbuf0.ls's cbuf0_alloc_state for precedent).

\\ One-time init: record the BAR4 mapping and reset the bump cursor.
mem_init bar4_base size :
    ← 64 mem_bar4_base bar4_base
    ← 64 mem_bar4_size size
    ← 64 mem_bar4_next 0

\\ Core allocator. Aligns the cursor up to `align`, bumps by `size`, returns
\\ the resulting GPU VA (same value is valid as CPU VA on GH200).
\\
\\ SYNTAX NOTE: alignment uses division+multiply rather than bitwise AND/NOT,
\\ mirroring cbuf0.ls's `next / 256 * 256` idiom. If the grammar supports
\\ bitmask operators (`and`, `xor`) more directly, this can be tightened.
mem_alloc size align out_gpu_va :
    cur → 64 mem_bar4_next
    bumped cur + align
    bumped bumped - 1
    aligned bumped / align * align
    new_next aligned + size
    limit → 64 mem_bar4_size
    if>= new_next limit
        \\ Out of BAR4 — no fallback (we deliberately have no kernel module).
        trap
    ← 64 mem_bar4_next new_next
    base → 64 mem_bar4_base
    gpu_va base + aligned
    ← 64 out_gpu_va gpu_va

\\ Default 256-byte alignment (matches GPU LDG/LDC natural alignment).
mem_alloc_aligned size out_gpu_va :
    mem_alloc size 256 out_gpu_va

\\ Reset the bump pointer. Equivalent to freeing every live allocation at once.
\\ Callers must have already drained the GPU (all QMDs completed) before this.
mem_reset :
    ← 64 mem_bar4_next 0

\\ Byte-granularity copy. Both dst and src are BAR4 VAs (or CPU-owned RAM —
\\ BAR4 is coherent with CPU caches on GH200, so a plain byte loop suffices).
\\ SYNTAX NOTE: a tighter 64-bit-stride loop with tail handling would be
\\ faster; kept simple here to stay within the grammar's `for` + `→/←` surface.
mem_copy dst src nbytes :
    for i 0 nbytes 1
        b → 8 src + i
        ← 8 dst + i b

\\ Fill `count` 32-bit slots at `addr` with `value`. Replaces cuMemsetD32.
mem_set32 addr value count :
    for i 0 count 1
        off i * 4
        ← 32 addr + off value
