// GSP/rpc_channel.s -- Step 9: Message queue RPC for channel allocation
//
// Translates lithos_gsp_alloc_channel() and the underlying message queue
// protocol from kernel/lithos_gsp.c and kernel/lithos_channel.c to raw ARM64.
// No libc.
//
// Source references:
//   GspMsgQueueSendCommand()   -- message_queue_cpu.c
//   GspMsgQueueReceiveStatus() -- message_queue_cpu.c
//   _kchannelSendChannelAllocRpc() -- kernel_channel.c
//   rpc_global_enums.h         -- RPC function codes
//   g_rpc-structures.h         -- RPC payload structures
//
// Architecture:
//   The GSP RPC protocol uses two shared-memory ring buffers in BAR4:
//     Command queue: CPU writes RPC requests, GSP reads
//     Status queue:  GSP writes RPC responses, CPU reads
//   Queue head/tail are tracked via BAR0 registers at NV_PGSP_QUEUE_HEAD/TAIL.
//
// Inputs (gsp_rpc_alloc_channel):
//   x0 = bar0 base (mmap'd BAR0 virtual address)
//   x1 = bar4 base (mmap'd BAR4 virtual address -- coherent HBM)
//   x2 = cmd_queue_offset (offset from bar4 to command queue buffer)
//   x3 = stat_queue_offset (offset from bar4 to status queue buffer)
//   x4 = gpfifo_gpu_va (physical address of GPFIFO ring, or 0 to bump-alloc)
//   x5 = gpfifo_entries (number of GPFIFO entries, e.g. 1024)
//   x6 = bump_ptr (pointer to uint64 bump allocator offset in BAR4)
//
// Returns:
//   x0 = 0 on success, negative on error
//   x1 = hardware channel_id (on success)
//   The GPFIFO ring (8KB) is allocated from BAR4 via bump allocator.
//   USERD base address stored in .data section (userd_base).
//
// Build:
//   as -o rpc_channel.o rpc_channel.s

// Shared constants (RPC codes, handles, classes, queue offsets, etc.)
.include "gsp_common.s"

// ---- File-specific constants ----

// Queue buffer size: each RPC queue (cmd and status) is a 256 KB ring in BAR4.
// head/tail are byte offsets that wrap modulo this size.
.equ QUEUE_BUF_SIZE,       0x40000     // 256 KB

.data
.balign 8

// Sequence counter for RPC messages (monotonically increasing)
rpc_sequence:   .quad 0

// Stored results after channel allocation
.globl userd_base
.globl gpfifo_base
.globl gpfifo_put
.globl channel_id_out
userd_base:     .quad 0     // BAR0 VA of USERD page for allocated channel
gpfifo_base:    .quad 0     // BAR4 VA of GPFIFO ring buffer
gpfifo_put:     .quad 0     // current GPFIFO put index (starts at 0)
channel_id_out: .quad 0     // hardware channel ID returned by GSP

.text

// ====================================================================
// Helper: load_qhead_offset -- compute BAR0 offset for QUEUE_HEAD(idx)
//   NV_PGSP_QUEUE_HEAD(i) = 0x110C00 + i*8
// Input:  w0 = queue index
// Output: x0 = BAR0 offset
// ====================================================================
.macro load_qhead_off dst, idx
    mov \dst, #0x0C00
    movk \dst, #0x0011, lsl #16         // \dst = 0x110C00
    .if \idx > 0
    add \dst, \dst, #(\idx * 8)         // + idx*8
    .endif
.endm

// Same for QUEUE_TAIL
.macro load_qtail_off dst, idx
    mov \dst, #0x0C04
    movk \dst, #0x0011, lsl #16         // \dst = 0x110C04
    .if \idx > 0
    add \dst, \dst, #(\idx * 8)         // + idx*8
    .endif
.endm


// ====================================================================
// mq_write -- Write an RPC message to the command queue
//
// Inputs:
//   x0 = bar0 base
//   x1 = cmd_queue_va (bar4 + cmd_queue_offset)
//   x2 = pointer to RPC message buffer (header + payload, already formatted)
//   x3 = total message length (header + payload, not including MQ wrapper)
//
// Protocol:
//   1. Read current queue head from BAR0 register
//   2. Write MQ element header (length + flags) at queue[head]
//   3. Copy RPC message after MQ element header
//   4. Advance head register at BAR0 to notify GSP
//
// Returns:
//   x0 = 0 on success
// ====================================================================
.globl mq_write
.type  mq_write, %function
.balign 4

mq_write:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]

    mov x19, x0                         // x19 = bar0
    mov x20, x1                         // x20 = cmd_queue_va
    mov x21, x2                         // x21 = message buffer ptr
    mov x22, x3                         // x22 = message length

    // Validate message length: must not exceed QUEUE_BUF_SIZE minus element overhead.
    // This prevents buffer overrun that would corrupt the ring.
    add w9, w22, #MQ_ELEM_HDR_SIZE      // total element size
    add w9, w9, #7
    and w9, w9, #0xFFFFFFF8             // aligned element size
    mov w10, #QUEUE_BUF_SIZE             // 0x40000 -- too large for cmp immediate
    cmp w9, w10
    b.hs .mq_write_fail                 // element too large for queue

    // Read current command queue head from BAR0 register
    // NV_PGSP_QUEUE_HEAD(0) = BAR0+0x110C00
    load_qhead_off x8, 0
    ldr w0, [x19, x8]                   // w0 = current head offset
    mov w9, w0                          // w9 = saved original head offset

    // Compute write position in queue buffer
    add x1, x20, x0                     // x1 = queue_va + head_offset

    // Write MQ element header: length and flags
    add w2, w22, #MQ_ELEM_HDR_SIZE      // total element size = msg + wrapper
    str w2, [x1]                        // element_length
    str wzr, [x1, #4]                   // flags = 0

    // Copy RPC message (header + payload) into queue after MQ wrapper
    add x1, x1, #MQ_ELEM_HDR_SIZE      // x1 = write position for RPC data
    mov x3, x22                         // x3 = bytes to copy
.mq_copy:
    cbz x3, .mq_copy_done
    ldrb w4, [x21], #1
    strb w4, [x1], #1
    sub x3, x3, #1
    b .mq_copy
.mq_copy_done:

    // Memory barrier: ensure all stores to BAR4 (queue data) are visible
    // before we update the head register.
    dsb st

    // Advance head by total element size (aligned to 8 bytes)
    add w2, w22, #MQ_ELEM_HDR_SIZE      // total element size
    add w2, w2, #7
    and w2, w2, #0xFFFFFFF8             // align to 8 bytes
    load_qhead_off x8, 0
    add w0, w9, w2                      // new head = saved_head + aligned_size
    // Wrap modulo QUEUE_BUF_SIZE (power of two)
    mov w3, #(QUEUE_BUF_SIZE - 1)
    and w0, w0, w3
    str w0, [x19, x8]                   // BAR0+0x110C00 <- new head

    // Memory barrier: head write must be visible to GSP
    dsb sy

    mov x0, #0                          // success
    b .mq_write_epilog

.mq_write_fail:
    mov x0, #-1                         // error: message too large for queue

.mq_write_epilog:
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.size mq_write, . - mq_write


// ====================================================================
// mq_poll_response -- Poll status queue for RPC response
//
// Inputs:
//   x0 = bar0 base
//   x1 = stat_queue_va (bar4 + stat_queue_offset)
//   x2 = expected sequence number
//
// Protocol:
//   Poll NV_PGSP_QUEUE_HEAD(1) until it advances past our read position.
//   Then read the response from the status queue buffer.
//
// Returns:
//   x0 = 0 on success, -1 on timeout
//   x1 = rpc_result from response header (on success)
//   x2 = pointer to response payload (on success, within stat_queue_va)
// ====================================================================
.globl mq_poll_response
.type  mq_poll_response, %function
.balign 4

mq_poll_response:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]

    mov x19, x0                         // x19 = bar0
    mov x20, x1                         // x20 = stat_queue_va
    mov x21, x2                         // x21 = expected sequence

    // Read current status queue tail (our read position)
    // NV_PGSP_QUEUE_TAIL(1) = BAR0+0x110C04 + 1*8 = BAR0+0x110C0C
    load_qtail_off x8, 1
    ldr w23, [x19, x8]                  // w23 = current tail (our read offset)

    // Timeout counter: ~10M iterations with MMIO reads = ~10s
    movz w24, #0x9680
    movk w24, #0x0098, lsl #16          // w24 = 0x00989680 = 10000000

.poll_stat_loop:
    // Read status queue head from BAR0
    // NV_PGSP_QUEUE_HEAD(1) = BAR0+0x110C00 + 1*8 = BAR0+0x110C08
    load_qhead_off x8, 1
    ldr w1, [x19, x8]                   // w1 = status queue head

    // If head != tail, there is a new message (handles wrap-around correctly,
    // since both head and tail are wrapped modulo QUEUE_BUF_SIZE)
    cmp w1, w23
    b.ne .poll_got_response

    // Decrement timeout counter
    subs x24, x24, #1
    b.eq .poll_timeout

    // Small spin delay (~100ns)
    isb
    b .poll_stat_loop

.poll_got_response:
    // Read response from stat_queue_va + tail offset
    add x1, x20, x23                    // x1 = response element address

    // Skip MQ element header (8 bytes) to get to RPC header
    add x2, x1, #MQ_ELEM_HDR_SIZE      // x2 = RPC header base

    // Read rpc_result from response header (offset 0x10 in RPC header)
    ldr w3, [x2, #0x10]                 // rpc_result
    // Read sequence from response header (offset 0x18)
    ldr w4, [x2, #0x18]                 // sequence
    // Verify sequence matches
    cmp w4, w21
    b.ne .poll_seq_mismatch             // wrong sequence, skip this element

    // Advance tail past this element
    ldr w5, [x1]                        // read element_length from MQ header
    add w5, w5, #7
    and w5, w5, #0xFFFFFFF8             // align to 8
    add w23, w23, w5                    // new tail
    // Wrap modulo QUEUE_BUF_SIZE (power of two)
    mov w6, #(QUEUE_BUF_SIZE - 1)
    and w23, w23, w6
    load_qtail_off x8, 1
    str w23, [x19, x8]                  // update tail register

    // Return: x0=0, x1=rpc_result, x2=payload ptr
    mov x0, #0
    mov w1, w3                          // rpc_result
    add x2, x2, #RPC_HDR_SIZE           // payload starts after RPC header
    b .poll_done

.poll_seq_mismatch:
    // Consume the mismatched element: advance tail past it so we don't
    // re-read the same element forever (which would infinite-loop).
    ldr w5, [x1]                        // element_length from MQ header
    add w5, w5, #7
    and w5, w5, #0xFFFFFFF8             // align to 8
    add w23, w23, w5                    // advance tail
    mov w6, #(QUEUE_BUF_SIZE - 1)
    and w23, w23, w6                    // wrap modulo QUEUE_BUF_SIZE
    load_qtail_off x8, 1
    str w23, [x19, x8]                  // update tail register
    // Decrement timeout so we cannot spin forever
    subs x24, x24, #1
    b.eq .poll_timeout
    b .poll_stat_loop

.poll_timeout:
    mov x0, #-1
    mov x1, #0
    mov x2, #0

.poll_done:
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.size mq_poll_response, . - mq_poll_response


// ====================================================================
// format_rpc_header -- Format an RPC header in a scratch buffer
//
// Inputs:
//   x0 = buffer pointer (at least RPC_HDR_SIZE + payload_size bytes)
//   w1 = function code
//   w2 = payload size (bytes after header)
//
// Writes RPC header at [x0], returns x0 unchanged.
// Increments rpc_sequence.
// ====================================================================
.type  format_rpc_header, %function
.balign 4

format_rpc_header:
    // header_version = 0x02000000
    mov w3, #0x0000
    movk w3, #0x0200, lsl #16           // w3 = 0x02000000
    str w3, [x0, #0x00]                 // header_version

    // signature = 0x43505246 ("FRPC")
    mov w3, #0x5246
    movk w3, #0x4350, lsl #16           // w3 = 0x43505246
    str w3, [x0, #0x04]                 // signature

    // length = RPC_HDR_SIZE + payload_size
    add w3, w2, #RPC_HDR_SIZE
    str w3, [x0, #0x08]                 // length

    // function code
    str w1, [x0, #0x0C]                 // function

    // rpc_result = 0 (filled by GSP on reply)
    str wzr, [x0, #0x10]                // rpc_result
    str wzr, [x0, #0x14]                // rpc_result_private

    // sequence number (atomic increment via load-exclusive/store-exclusive
    // to prevent duplicate sequence numbers under concurrent access)
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
1:  ldxr x4, [x3]
    add x5, x4, #1
    stxr w6, x5, [x3]
    cbnz w6, 1b                          // retry if exclusive store failed
    str w4, [x0, #0x18]                 // sequence

    // spare = 0
    str wzr, [x0, #0x1C]

    ret

.size format_rpc_header, . - format_rpc_header


// ====================================================================
// gsp_rpc_alloc_channel -- Full channel allocation via GSP RPC
//
// Sends three RPCs in sequence:
//   1. NV_RM_RPC_ALLOC_ROOT     -> allocate client handle
//   2. NV_RM_RPC_ALLOC_DEVICE   -> allocate device object
//   3. NV_RM_RPC_ALLOC_CHANNEL  -> allocate GPFIFO channel (class 0xC86F)
//
// After channel allocation:
//   - Bump-allocates 8KB GPFIFO ring from BAR4
//   - Stores USERD base address (BAR0+0xFC0000 + chid*0x200)
//   - Initializes gpfifo_put = 0
//
// Inputs:
//   x0 = bar0 base
//   x1 = bar4 base
//   x2 = cmd_queue_offset (from bar4)
//   x3 = stat_queue_offset (from bar4)
//   x4 = gpfifo_gpu_va (physical address of GPFIFO ring, or 0 to bump-alloc)
//   x5 = gpfifo_entries
//   x6 = bump_ptr (pointer to u64 bump offset variable)
//
// Returns:
//   x0 = 0 on success, negative on error
//   x1 = hardware channel_id
// ====================================================================
.globl gsp_rpc_alloc_channel
.type  gsp_rpc_alloc_channel, %function
.balign 4

gsp_rpc_alloc_channel:
    stp x29, x30, [sp, #-176]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]

    mov x19, x0                         // x19 = bar0
    mov x20, x1                         // x20 = bar4
    add x21, x1, x2                     // x21 = cmd_queue_va
    add x22, x1, x3                     // x22 = stat_queue_va
    mov x23, x4                         // x23 = gpfifo_gpu_va (or 0)
    mov x24, x5                         // x24 = gpfifo_entries
    mov x25, x6                         // x25 = bump_ptr

    // ================================================================
    // CONTRACT: gpfifo_gpu_va (x4/x23) MUST be zero on entry.
    // Callers must use the bump-allocation path so we can derive a
    // valid CPU VA (bar4_mmap + offset) for pushbuffer writes. On
    // GH200 bar4_phys != bar4_mmap, so a caller-supplied GPU VA
    // cannot be reused as a CPU VA -- doing so would fault.
    // Reject non-zero with -EINVAL (-22) via .rpc_einval.
    // ================================================================
    cbnz x23, .rpc_einval

    // RPC scratch buffer at sp+96 (80 bytes, enough for header + largest payload)
    // sp+96 .. sp+175 = 80 bytes scratch

    // ================================================================
    // RPC 1: NV_RM_RPC_ALLOC_ROOT (function=2)
    // Payload: { u32 hClient }
    // Source: rpc_global_enums.h, kernel_channel.c
    // ================================================================
    add x0, sp, #96                     // scratch buffer
    mov w1, #NV_VGPU_MSG_FUNCTION_ALLOC_ROOT    // 2
    mov w2, #RPC_ALLOC_ROOT_PAYLOAD     // 4 bytes
    bl format_rpc_header

    // Write payload: hClient = 0x01000000
    add x0, sp, #96
    mov w1, #0x0000
    movk w1, #0x0100, lsl #16           // w1 = 0x01000000
    str w1, [x0, #RPC_HDR_SIZE]         // payload[0] = hClient

    // Send RPC via command queue
    mov x0, x19                         // bar0
    mov x1, x21                         // cmd_queue_va
    add x2, sp, #96                     // message buffer
    mov x3, #(RPC_HDR_SIZE + RPC_ALLOC_ROOT_PAYLOAD)
    bl mq_write
    cbnz x0, .rpc_fail                  // check mq_write result

    // Poll for response
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1                      // sequence we just sent
    mov x0, x19                         // bar0
    mov x1, x22                         // stat_queue_va
    bl mq_poll_response
    cbnz x0, .rpc_fail
    cbnz w1, .rpc_fail                  // check rpc_result != 0 => GSP error

    // ================================================================
    // RPC 2: NV_RM_RPC_ALLOC_DEVICE (function=3)
    // Payload: { u32 hClient, u32 hDevice, u32 hClass }
    // Source: rpc_global_enums.h
    // ================================================================
    add x0, sp, #96
    mov w1, #NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE  // 3
    mov w2, #RPC_ALLOC_DEVICE_PAYLOAD   // 12 bytes
    bl format_rpc_header

    add x0, sp, #96
    mov w1, #0x0000
    movk w1, #0x0100, lsl #16           // w1 = 0x01000000 (hClient)
    str w1, [x0, #RPC_HDR_SIZE]
    mov w1, #0x0001
    movk w1, #0x0100, lsl #16           // w1 = 0x01000001 (hDevice)
    str w1, [x0, #(RPC_HDR_SIZE + 4)]
    mov w1, #NV01_DEVICE_0              // 0x0080
    str w1, [x0, #(RPC_HDR_SIZE + 8)]   // hClass

    mov x0, x19
    mov x1, x21
    add x2, sp, #96
    mov x3, #(RPC_HDR_SIZE + RPC_ALLOC_DEVICE_PAYLOAD)
    bl mq_write
    cbnz x0, .rpc_fail                  // check mq_write result

    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1
    mov x0, x19
    mov x1, x22
    bl mq_poll_response
    cbnz x0, .rpc_fail
    cbnz w1, .rpc_fail                  // check rpc_result != 0 => GSP error

    // ================================================================
    // RPC 3: NV_RM_RPC_ALLOC_SUBDEVICE (function=19)
    // Payload: { u32 hClient, u32 hParent (=hDevice), u32 hSubdevice, u32 hClass }
    // Source: rpc_global_enums.h -- NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE
    // ================================================================
    add x0, sp, #96
    mov w1, #NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE   // 19
    mov w2, #RPC_ALLOC_SUBDEVICE_PAYLOAD            // 16 bytes
    bl format_rpc_header

    add x0, sp, #96
    // hClient = 0x01000000
    mov w1, #0x0000
    movk w1, #0x0100, lsl #16
    str w1, [x0, #RPC_HDR_SIZE]
    // hParent = hDevice = 0x01000001
    mov w1, #0x0001
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 4)]
    // hSubdevice = 0x01000002
    mov w1, #0x0002
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 8)]
    // hClass = NV20_SUBDEVICE_0 = 0x20E0
    mov w1, #NV20_SUBDEVICE_0
    str w1, [x0, #(RPC_HDR_SIZE + 12)]

    mov x0, x19
    mov x1, x21
    add x2, sp, #96
    mov x3, #(RPC_HDR_SIZE + RPC_ALLOC_SUBDEVICE_PAYLOAD)
    bl mq_write
    cbnz x0, .rpc_fail                  // check mq_write result

    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1
    mov x0, x19
    mov x1, x22
    bl mq_poll_response
    cbnz x0, .rpc_fail
    cbnz w1, .rpc_fail                  // check rpc_result != 0 => GSP error

    // ================================================================
    // Bump-allocate GPFIFO ring from BAR4 (8KB, page-aligned)
    // If gpfifo_gpu_va was provided (non-zero), skip allocation.
    // On GH200: BAR4 PA == GPU VA, so the bump offset + bar4_phys
    // gives the GPU VA. The caller provides gpfifo_gpu_va or we
    // use the bump-allocated BAR4 offset.
    // ================================================================
    // x28 holds GPFIFO CPU VA (for zeroing / storing as cpu-accessible base).
    // x23 holds GPFIFO GPU VA (== bar4_phys + offset on GH200); this is what
    // gets sent to GSP in the RPC. CPU VA and GPU VA are two different values.
    mov x28, x23                        // if caller provided, treat as both
    cbnz x23, .gpfifo_ready

    // Read current bump offset
    ldr x0, [x25]                       // current bump offset
    // Align up to GPFIFO_ALIGN (4096)
    add x0, x0, #(GPFIFO_ALIGN - 1)
    mov x1, #~(GPFIFO_ALIGN - 1)
    and x0, x0, x1
    // x0 = aligned offset for GPFIFO within BAR4
    add x28, x20, x0                    // x28 = bar4_mmap + offset = GPFIFO CPU VA
    // GPU VA = bar4_phys + offset
    adrp x2, bar4_phys
    add x2, x2, :lo12:bar4_phys
    ldr x2, [x2]                        // x2 = bar4_phys
    add x23, x2, x0                     // x23 = bar4_phys + offset = GPFIFO GPU VA
    // Update bump pointer past GPFIFO
    add x1, x0, #GPFIFO_SIZE
    str x1, [x25]                       // bump_ptr += alignment + GPFIFO_SIZE

    // Zero the GPFIFO ring (8KB) using CPU VA
    mov x1, x28                         // GPFIFO CPU VA
    mov x2, #(GPFIFO_SIZE / 8)          // iterations (8 bytes each)
.zero_gpfifo:
    cbz x2, .gpfifo_ready
    str xzr, [x1], #8
    sub x2, x2, #1
    b .zero_gpfifo

.gpfifo_ready:
    // Store GPFIFO CPU base in .data (kernel reads via CPU VA)
    adrp x0, gpfifo_base
    add x0, x0, :lo12:gpfifo_base
    str x28, [x0]

    // Zero gpfifo_put
    adrp x0, gpfifo_put
    add x0, x0, :lo12:gpfifo_put
    str xzr, [x0]

    // ================================================================
    // RPC 4: NV_RM_RPC_ALLOC_CHANNEL (function=6)
    // Payload (from g_rpc-structures.h / _kchannelSendChannelAllocRpc):
    //   offset 0x00: u32 hClient
    //   offset 0x04: u32 hParent (= hDevice)
    //   offset 0x08: u32 hChannel
    //   offset 0x0C: u32 hClass        (= 0xC86F)
    //   offset 0x10: u64 gpFifoBase    (GPU VA of GPFIFO ring)
    //   offset 0x18: u32 gpFifoEntries
    //   offset 0x1C: u32 hVeAddr       (= 0)
    //   offset 0x20: u32 veaSpace      (= 0)
    //   offset 0x24: u32 tsgId         (= 0)
    //
    // Source: kernel_channel.c:_kchannelSendChannelAllocRpc
    // ================================================================
    add x0, sp, #96
    mov w1, #NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA    // 6
    mov w2, #RPC_ALLOC_CHANNEL_PAYLOAD  // 40 bytes
    bl format_rpc_header

    add x0, sp, #96
    // hClient = 0x01000000
    mov w1, #0x0000
    movk w1, #0x0100, lsl #16
    str w1, [x0, #RPC_HDR_SIZE]         // offset 0x00: hClient
    // hParent = hSubdevice = 0x01000002
    mov w1, #0x0002
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 4)]   // offset 0x04: hParent
    // hChannel = 0x01000003
    mov w1, #0x0003
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 8)]   // offset 0x08: hChannel
    // hClass = HOPPER_CHANNEL_GPFIFO_A = 0xC86F
    mov w1, #HOPPER_CHANNEL_GPFIFO
    str w1, [x0, #(RPC_HDR_SIZE + 12)]  // offset 0x0C: hClass = 0xC86F
    // gpFifoBase (64-bit GPU VA) -- x23 holds GPFIFO GPU VA (bar4_phys + off)
    // On GH200: BAR4 PA == GPU VA; CPU VA is separately held in x28
    str x23, [x0, #(RPC_HDR_SIZE + 16)] // offset 0x10: gpFifoBase
    // gpFifoEntries
    str w24, [x0, #(RPC_HDR_SIZE + 24)] // offset 0x18: gpFifoEntries
    // Zero remaining fields (hVeAddr, veaSpace, tsgId)
    str wzr, [x0, #(RPC_HDR_SIZE + 28)] // offset 0x1C: hVeAddr = 0
    str wzr, [x0, #(RPC_HDR_SIZE + 32)] // offset 0x20: veaSpace = 0
    str wzr, [x0, #(RPC_HDR_SIZE + 36)] // offset 0x24: tsgId = 0

    // Send channel alloc RPC
    mov x0, x19
    mov x1, x21
    add x2, sp, #96
    mov x3, #(RPC_HDR_SIZE + RPC_ALLOC_CHANNEL_PAYLOAD)
    bl mq_write
    cbnz x0, .rpc_fail                  // check mq_write result

    // Poll for response
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1
    mov x0, x19
    mov x1, x22
    bl mq_poll_response
    cbnz x0, .rpc_fail
    cbnz w1, .rpc_fail                  // check rpc_result != 0 => GSP error

    // ================================================================
    // Extract channel_id from response
    // mq_poll_response returns x2 = pointer to response payload.
    // The channel_id is the first u32 of the response payload.
    // ================================================================
    ldr w26, [x2]                       // w26 = channel_id

    // Store channel_id
    adrp x0, channel_id_out
    add x0, x0, :lo12:channel_id_out
    // w26 was loaded via `ldr w26,...` which zero-extends into x26.
    // channel_id_out is declared .quad (64-bit); store full x26.
    str x26, [x0]

    // ================================================================
    // Set up USERD base address
    // USERD for channel i = BAR0 + 0xFC0000 + channel_id * 0x200
    // Source: clc86f.h Nvc86fControl structure
    // GPPut is at USERD + 0x08C
    // ================================================================
    mov w0, #USERD_STRIDE               // 0x200
    mul w0, w26, w0                     // chid * 0x200
    // USERD_BAR0_BASE = 0xFC0000 (too large for single mov)
    mov w1, #0x0000
    movk w1, #0x00FC, lsl #16           // w1 = 0x00FC0000
    add w0, w0, w1                      // BAR0 offset to USERD page
    // Extend to 64-bit and add bar0
    mov w7, w0                          // zero-extend w0 into x7
    add x7, x19, x7                     // x7 = bar0 + USERD offset = USERD VA
    adrp x1, userd_base
    add x1, x1, :lo12:userd_base
    str x7, [x1]                        // store USERD VA

    // Return success
    mov x0, #0
    mov w1, w26                         // channel_id in x1 (zero-extended)
    b .rpc_done

.rpc_fail:
    mov x0, #-1
    mov x1, #0
    b .rpc_done

.rpc_einval:
    // Caller passed non-zero gpfifo_gpu_va; this entry is reserved
    // for the bump-allocation path that derives the matching CPU VA.
    // Distinct return code (-22 = -EINVAL) so misuse is identifiable.
    mov x0, #-22
    mov x1, #0

.rpc_done:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

.size gsp_rpc_alloc_channel, . - gsp_rpc_alloc_channel
