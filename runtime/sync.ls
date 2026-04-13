\\ sync.ls — kernel completion polling
\\
\\ Replaces CUDA's async-ordering primitives:
\\   cuStreamSynchronize  → sync_wait_flag / sync_wait_gpget
\\   cuEventRecord        → (kernel EXIT + SEND_SIGNALING_PCAS2_B=0x0a in pushbuffer)
\\   cuEventSynchronize   → sync_wait_counter
\\
\\ Lithos model: each kernel's last instruction is EXIT. The pushbuffer emits
\\ SEND_SIGNALING_PCAS2_B with value 0x0a after SEND_PCAS_A (see pushbuffer.ls),
\\ which makes the host channel front-end advance USERD GPGet when the QMD
\\ retires. BAR4 is coherent via NVLink-C2C, so the CPU sees the update the
\\ instant the cache line is invalidated — no MMIO, no DMA fence.
\\
\\ Two complementary completion paths:
\\   1. Kernel-authored flag. Caller threads a GPU VA through cbuf0 as a
\\      param; the kernel does `← 32 flag 1` right before EXIT. CPU spins on
\\      that word. Finest granularity — per-kernel, per-stage.
\\   2. Channel GPGet. Host front-end bumps USERD+0x88 as it consumes GPFIFO
\\      entries. Cheapest — no extra kernel stores, no param plumbing —
\\      but granularity is one whole GPFIFO entry (a full QMD submission).
\\
\\ USERD layout (from GSP/rpc_channel.s, mirrors doorbell.ls):
\\   USERD + 0x88 : GPGet  (consumer index; GPU advances as QMDs retire)
\\   USERD + 0x8c : GPPut  (producer index; CPU advances on submission)
\\   USERD + 0x90 : Doorbell

\\ Allocate one 32-bit completion flag in BAR4, zero-initialized.
\\ Caller passes the resulting GPU VA into the kernel via cbuf0 so the kernel
\\ can store 1 there just before EXIT.
sync_allocate_flag out_flag_gpu_va :
    mem_alloc 4 4 out_flag_gpu_va
    flag → 64 out_flag_gpu_va
    ← 32 flag 0

\\ Reset a completion flag to 0 so it can be reused for the next launch.
sync_reset_flag flag_gpu_va :
    ← 32 flag_gpu_va 0

\\ Spin-poll a 32-bit word until it is non-zero.
\\ BAR4 is CPU-cache-coherent with the GPU on GH200, so a plain load sees the
\\ GPU's store without any explicit invalidation.
sync_wait_flag flag_gpu_va :
    for i 0 1 0
        v → 32 flag_gpu_va
        if>= v 1
            ← 32 flag_gpu_va 0
            \\ Observed: clear and exit by collapsing the open-ended loop.
            v v

\\ Bounded-iteration version: traps after max_iter reads if the flag never
\\ fires. Use in tests / bring-up to surface hangs instead of spinning forever.
sync_wait_flag_timeout flag_gpu_va max_iter :
    for i 0 max_iter 1
        v → 32 flag_gpu_va
        if>= v 1
            ← 32 flag_gpu_va 0
            \\ Flag observed; exit early by skipping the trap below.
            ret ret
    \\ Loop exhausted without seeing the flag — fail loudly.
    trap

\\ Generic 32-bit counter poll. Blocks until *counter_addr == expected_value.
\\ Used for GPGet / custom monotonic counters placed anywhere in BAR4.
sync_wait_counter counter_addr expected_value :
    for i 0 1 0
        v → 32 counter_addr
        if== v expected_value
            v v

\\ Spin on the channel's GPGet until it reaches `expected_gpget`. The GPU host
\\ front-end bumps this as GPFIFO entries retire. Pair with dispatch.ls's
\\ current_gppos tracking: after submit, pass that value here to wait.
sync_wait_gpget userd_va expected_gpget :
    for i 0 1 0
        v → 32 userd_va + 0x88
        if== v expected_gpget
            v v

\\ Drain the entire channel — block until GPGet catches up to GPPut. Equivalent
\\ to cuCtxSynchronize: all outstanding work on this channel has retired.
sync_drain userd_va :
    gpput → 32 userd_va + 0x8c
    sync_wait_gpget userd_va gpput
