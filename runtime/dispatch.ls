\\ dispatch.ls — top-level kernel launch
\\
\\ Ties together QMD construction, cbuf0 population, pushbuffer emission, and
\\ the USERD doorbell ring. This is the Lithos equivalent of cuLaunchKernel —
\\ but with zero CUDA/libcuda involvement: we go straight from ELF+params to
\\ GPU execution through the driver-allocated GPFIFO channel.
\\
\\ Prerequisites (set up once at runtime init, not here):
\\   - GPFIFO channel allocated (see GSP/rpc_channel.s)
\\   - pushbuffer BAR4 mapping, userd_va, qmd_alloc_state, cbuf0_alloc_state
\\   - current_gppos: producer cursor for GPFIFO ring
\\   - done_flag: GPU-writable scratch polled for completion

\\ Launch one kernel. Blocks until the completion flag fires.
dispatch_kernel elf_gpu_va grid_x grid_y grid_z block_x block_y block_z smem_size params params_size :
    \\ --- QMD ---
    qmd_alloc qmd_ptr_slot qmd_gpu_va_slot
    qmd_ptr → 64 qmd_ptr_slot
    qmd_gpu_va → 64 qmd_gpu_va_slot
    qmd_init qmd_ptr
    qmd_set_block_dim qmd_ptr block_x block_y block_z
    qmd_set_grid_dim qmd_ptr grid_x grid_y grid_z
    qmd_set_shared_mem qmd_ptr smem_size
    qmd_set_entry_pc qmd_ptr elf_gpu_va

    \\ --- cbuf0 ---
    \\ Param block lives at cbuf0+0x210; total size = 0x210 + params_size.
    cbuf0_size 0x210 + params_size
    cbuf0_alloc cbuf0_size cbuf0_ptr_slot
    cbuf0_ptr → 64 cbuf0_ptr_slot
    cbuf0_set_param_block cbuf0_ptr 0x210 params params_size
    \\ TODO(probe): cbuf0_set_register_count cbuf0_ptr <kernel_regcount>
    \\ — offset unknown; see cbuf0.ls. Launch will fail or mis-schedule until probed.
    cbuf0_finalize cbuf0_ptr

    \\ --- pushbuffer ---
    pb_emit_qmd pushbuffer qmd_gpu_va qmd_ptr

    \\ --- doorbell ---
    pb_cur → 64 pushbuffer
    pb_base pushbuffer + 8
    entry_bytes pb_cur - pb_base
    gppos_now → 32 current_gppos
    new_gppos gppos_now + entry_bytes / 4
    ← 32 current_gppos new_gppos
    doorbell_write_gpput userd_va new_gppos
    doorbell_ring userd_va

    \\ --- wait ---
    wait_for_completion done_flag_addr

\\ Spin-poll the completion flag. GPU writes non-zero when kernel retires.
wait_for_completion done_flag_addr :
    for i 0 1 0
        v → 32 done_flag_addr
        if== v 0
            \\ keep spinning
            v v
        if>= v 1
            ← 32 done_flag_addr 0
            \\ break: set loop guard so we exit
            v v
