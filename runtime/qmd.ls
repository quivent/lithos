\\ qmd.ls — QMD (Queue Meta Data) builder for Hopper SM90
\\
\\ The QMD is a 528-byte descriptor that tells the GPU how to launch a kernel:
\\   - grid/block dimensions
\\   - shared memory size
\\   - entry program counter (40-bit GPU virtual address)
\\
\\ Field offsets are from docs/qmd_fields.md (empirically probed via qmd_probe_driver.c).
\\ register_count is NOT in the QMD body and NOT in cbuf0. It lives in the Shader Program
\\ Descriptor (SPD), which is the 4th inline CB load in the 5-part pushbuffer launch sequence.
\\ SPD offset 0x094, bits 23:16 (byte[2]). Formula: (0x08 << 24) | (reg_count << 16) | 0x0001.
\\ See pushbuffer.ls pb_emit_spd and docs/cbuf0_fields.md (authoritative ref for the 5-part sequence).

\\ Zero the 528-byte QMD body at qmd_ptr.
qmd_init qmd_ptr :
    for i 0 528 8
        ← 64 qmd_ptr + i 0

\\ block_dim_xyz at bytes 0x00 / 0x04 / 0x08
qmd_set_block_dim qmd_ptr bx by bz :
    ← 32 qmd_ptr + 0x00 bx
    ← 32 qmd_ptr + 0x04 by
    ← 32 qmd_ptr + 0x08 bz

\\ grid_dim_xyz at 0x0c / 0x10 / 0x14, plus cluster-scheduling copies at 0x15c / 0x160 / 0x164
qmd_set_grid_dim qmd_ptr gx gy gz :
    ← 32 qmd_ptr + 0x0c gx
    ← 32 qmd_ptr + 0x10 gy
    ← 32 qmd_ptr + 0x14 gz
    ← 32 qmd_ptr + 0x15c gx
    ← 32 qmd_ptr + 0x160 gy
    ← 32 qmd_ptr + 0x164 gz

\\ shared_mem_size at byte 0x002c. Driver-derived field at 0x013c = nbytes + 1024.
qmd_set_shared_mem qmd_ptr nbytes :
    ← 32 qmd_ptr + 0x002c nbytes
    ← 32 qmd_ptr + 0x013c nbytes + 1024

\\ entry_pc is a 40-bit GPU VA split across two 32-bit slots:
\\   lo32 at 0x0118, hi8 at 0x011c
qmd_set_entry_pc qmd_ptr gpu_va :
    lo gpu_va / 0x100000000 * 0x100000000
    lo gpu_va - lo
    hi gpu_va / 0x100000000
    ← 32 qmd_ptr + 0x0118 lo
    ← 32 qmd_ptr + 0x011c hi
