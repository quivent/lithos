\\ launch.ls — kernel dispatch composition (replaces cuLaunchKernel)
\\
\\ One QMD, one doorbell, zero syscalls at steady state. Ties together the
\\ ELF entry point (from elf_load.ls), the BAR4 bump allocator (mem.ls), the
\\ 528-byte descriptor builder (qmd.ls), the constant-buffer packer (cbuf0.ls),
\\ the GPFIFO method emitter (pushbuffer.ls), and the USERD doorbell (doorbell.ls).
\\
\\ Shared runtime state cells (provisioned once by runtime_init, referenced by
\\ name — same convention as mem_bar4_next / cbuf0_alloc_state):
\\
\\   current_pushbuffer — GPU VA of the pushbuffer ring control cell (pb_ptr;
\\                        first 8 bytes hold the write cursor, stream at +8)
\\   current_gpput      — 32-bit dword cursor into the GPFIFO (producer index)
\\   userd_address      — GPU VA of the USERD page (GPPut=+0x8c, doorbell=+0x90)
\\
\\ See runtime/init.ls for the stores that populate these cells.

\\ Allocate, build, and submit one QMD. Non-blocking — caller uses sync.ls to
\\ wait. Returns nothing; on BAR4 exhaustion mem_alloc traps.
launch_kernel elf_entry_gpu_va reg_count grid_x grid_y grid_z block_x block_y block_z shmem_size params_ptr params_size :
    \\ --- 1. Allocate 528-byte QMD in HBM (256-byte aligned for SEND_PCAS_A) ---
    mem_alloc 528 256 qmd_gpu_va_slot
    qmd_gpu_va → 64 qmd_gpu_va_slot

    \\ --- 2. Fill QMD body ---
    qmd_init qmd_gpu_va
    qmd_set_block_dim qmd_gpu_va block_x block_y block_z
    qmd_set_grid_dim qmd_gpu_va grid_x grid_y grid_z
    qmd_set_shared_mem qmd_gpu_va shmem_size
    qmd_set_entry_pc qmd_gpu_va elf_entry_gpu_va

    \\ --- 3. Allocate cbuf0 (0x210 driver block + params) ---
    cbuf0_size 0x210 + params_size
    mem_alloc cbuf0_size 256 cbuf0_gpu_va_slot
    cbuf0_gpu_va → 64 cbuf0_gpu_va_slot
    cbuf0_set_param_block cbuf0_gpu_va 0x210 params_ptr params_size
    cbuf0_finalize cbuf0_gpu_va

    \\ --- 4. Build SPD (384 bytes, 256-byte aligned) ---
    mem_alloc 384 256 spd_gpu_va_slot
    spd_gpu_va → 64 spd_gpu_va_slot
    spd_build spd_gpu_va elf_entry_gpu_va reg_count block_x block_y block_z shmem_size

    \\ --- 5. Load current pushbuffer + GPPut ---
    pb_ptr → 64 current_pushbuffer
    gpput_before → 32 current_gpput

    \\ Note cursor before emission so we can measure how many dwords we appended.
    cur_before → 64 pb_ptr

    \\ --- 6. Emit the full 5-part CB load + QMD submission sequence ---
    pb_emit_launch pb_ptr qmd_gpu_va qmd_gpu_va spd_gpu_va spd_gpu_va cbuf0_gpu_va

    \\ --- 7. Compute advance. pb_emit_launch advances pb_ptr's internal cursor
    \\        by exactly the number of dwords we produced. GPPut is measured
    \\        in dwords, so (cur_after - cur_before) / 4 is our delta. ---
    cur_after → 64 pb_ptr
    entry_bytes cur_after - cur_before
    entry_dwords entry_bytes / 4
    new_gpput gpput_before + entry_dwords
    ← 32 current_gpput new_gpput

    \\ --- 8. Publish GPPut and ring the doorbell ---
    userd_va → 64 userd_address
    doorbell_write_gpput userd_va new_gpput
    doorbell_ring userd_va

\\ Blocking launch: submit, then spin on done_flag until the kernel retires.
\\ Uses the symmetric-grid / symmetric-block convenience shape (grid and block
\\ each take one scalar that's broadcast to x/y/z). Wire into sync.ls for the
\\ actual poll — we just forward the GPU VA of the completion dword.
launch_kernel_sync elf_entry_gpu_va reg_count grid block shmem_size params_ptr params_size done_flag_gpu_va :
    launch_kernel elf_entry_gpu_va reg_count grid grid grid block block block shmem_size params_ptr params_size
    wait_for_completion done_flag_gpu_va
