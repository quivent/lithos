\\ doorbell.ls — USERD doorbell ring for submitting pushbuffer work to the GPU
\\
\\ The USERD (User Doorbell) region is a mapped slice of the channel's state
\\ where the CPU publishes GPPut and rings the doorbell. The GPU host interface
\\ snoops this region and kicks off work.
\\
\\ Layout (from GSP/rpc_channel.s, USERD_GPPUT_OFF=0x08C):
\\   USERD + 0x8c : GPPut  (producer index into GPFIFO ring)
\\   USERD + 0x90 : Doorbell (any write rings it)
\\
\\ On ARM64 we need a DSB SY after each store so the GPU sees our writes in
\\ order. The primitive `membar_sys` lowers to `DSB SY` on ARM64.

\\ Publish new GPPut value so the GPU knows more work is queued.
doorbell_write_gpput userd_va new_gppos :
    ← 32 userd_va + 0x8c new_gppos
    membar_sys

\\ Ring the doorbell at USERD+0x90. Any value works; GPU only cares about the write.
doorbell_ring userd_va :
    ← 32 userd_va + 0x90 0
    membar_sys
