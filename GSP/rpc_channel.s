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

// ---- BAR0 register offsets ----
// Source: dev_gsp.h -- NV_PGSP_QUEUE_HEAD/TAIL
// NV_PGSP_QUEUE_HEAD(i) = 0x110C00 + i*8
// NV_PGSP_QUEUE_TAIL(i) = 0x110C04 + i*8
// These are loaded into registers because they exceed str immediate range.

// Queue indices (from message_queue_cpu.c)
.equ RPC_CMD_QUEUE_IDX,     0           // command queue = index 0
.equ RPC_STAT_QUEUE_IDX,    1           // status queue = index 1

// ---- USERD layout (BAR0 offset, from clc86f.h) ----
// USERD base for channel i = BAR0+0xFC0000 + chid*0x200
.equ USERD_STRIDE,          0x200
.equ USERD_GPPUT_OFF,       0x08C       // GPPut offset within USERD page

// ---- GPFIFO sizing ----
.equ GPFIFO_SIZE,           8192        // 8KB = 1024 entries * 8 bytes each
.equ GPFIFO_ALIGN,          4096        // page-aligned

// ---- RPC function codes (from rpc_global_enums.h) ----
.equ NV_VGPU_MSG_FUNCTION_ALLOC_ROOT,        2
.equ NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE,      3
.equ NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA, 6

// ---- RPC object classes ----
.equ NV01_DEVICE_0,         0x0080      // device class
.equ HOPPER_CHANNEL_GPFIFO, 0xC86F      // Hopper GPFIFO channel class

// ---- RPC object handles (arbitrary, must be unique per client) ----
.equ HANDLE_CLIENT,         0x01000000
.equ HANDLE_DEVICE,         0x01000001
.equ HANDLE_CHANNEL,        0x01000003

// ---- RPC message header layout (from rpc_headers.h / message_queue_cpu.c) ----
// Each RPC message in the command queue has:
//   offset 0x00: u32 header_version (= 0x02000000)
//   offset 0x04: u32 signature      (= 0x43505246 "FRPC")
//   offset 0x08: u32 length         (total message length in bytes)
//   offset 0x0C: u32 function       (RPC function code)
//   offset 0x10: u32 rpc_result     (0 on send, filled by GSP on reply)
//   offset 0x14: u32 rpc_result_private (reserved)
//   offset 0x18: u32 sequence       (monotonic sequence number)
//   offset 0x1C: u32 spare          (0)
//   offset 0x20: ... payload (function-specific)
.equ RPC_HDR_SIZE,          0x20        // 32 bytes header

// ---- RPC payload sizes ----
.equ RPC_ALLOC_ROOT_PAYLOAD,    4       // u32 hClient
.equ RPC_ALLOC_DEVICE_PAYLOAD,  12      // u32 hClient, u32 hDevice, u32 hClass
.equ RPC_ALLOC_CHANNEL_PAYLOAD, 40      // full channel alloc params

// ---- Message queue element wrapper ----
// Source: msgq library -- each element has an 8-byte wrapper:
//   offset 0x00: u32 element_length
//   offset 0x04: u32 flags (0 = normal)
//   offset 0x08: ... RPC header + payload
.equ MQ_ELEM_HDR_SIZE,     8

// ---- Timeout ----
.equ RPC_POLL_LIMIT,       10000000    // ~10M iterations with MMIO = ~10s

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

    // Read current command queue head from BAR0 register
    // NV_PGSP_QUEUE_HEAD(0) = BAR0+0x110C00
    load_qhead_off x8, 0
    ldr w0, [x19, x8]                   // w0 = current head offset

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
    ldr w0, [x19, x8]                   // re-read current head
    add w0, w0, w2                      // new head
    str w0, [x19, x8]                   // BAR0+0x110C00 <- new head

    // Memory barrier: head write must be visible to GSP
    dsb sy

    mov x0, #0                          // success
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
    movz x24, #0x0098
    movk x24, #0x9680, lsl #0           // x24 = 0x989680 = 10000000
    // (movz sets upper, movk patches lower -- but actually:
    //  10000000 = 0x989680, so movz w24, #0x9680; movk w24, #0x98, lsl #16)
    movz w24, #0x9680
    movk w24, #0x0098, lsl #16          // w24 = 0x00989680 = 10000000

.poll_stat_loop:
    // Read status queue head from BAR0
    // NV_PGSP_QUEUE_HEAD(1) = BAR0+0x110C00 + 1*8 = BAR0+0x110C08
    load_qhead_off x8, 1
    ldr w1, [x19, x8]                   // w1 = status queue head

    // If head > tail, there is a new message
    cmp w1, w23
    b.hi .poll_got_response

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
    b.ne .poll_stat_loop                // wrong sequence, keep polling

    // Advance tail past this element
    ldr w5, [x1]                        // read element_length from MQ header
    add w5, w5, #7
    and w5, w5, #0xFFFFFFF8             // align to 8
    add w23, w23, w5                    // new tail
    load_qtail_off x8, 1
    str w23, [x19, x8]                  // update tail register

    // Return: x0=0, x1=rpc_result, x2=payload ptr
    mov x0, #0
    mov w1, w3                          // rpc_result
    add x2, x2, #RPC_HDR_SIZE           // payload starts after RPC header
    b .poll_done

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

    // sequence number (load, increment, store)
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x4, [x3]
    add x5, x4, #1
    str x5, [x3]
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

    // Poll for response
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1                      // sequence we just sent
    mov x0, x19                         // bar0
    mov x1, x22                         // stat_queue_va
    bl mq_poll_response
    cbnz x0, .rpc_fail

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

    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1
    mov x0, x19
    mov x1, x22
    bl mq_poll_response
    cbnz x0, .rpc_fail

    // ================================================================
    // Bump-allocate GPFIFO ring from BAR4 (8KB, page-aligned)
    // If gpfifo_gpu_va was provided (non-zero), skip allocation.
    // On GH200: BAR4 PA == GPU VA, so the bump offset + bar4_phys
    // gives the GPU VA. The caller provides gpfifo_gpu_va or we
    // use the bump-allocated BAR4 offset.
    // ================================================================
    cbnz x23, .gpfifo_ready

    // Read current bump offset
    ldr x0, [x25]                       // current bump offset
    // Align up to GPFIFO_ALIGN (4096)
    add x0, x0, #(GPFIFO_ALIGN - 1)
    mov x1, #~(GPFIFO_ALIGN - 1)
    and x0, x0, x1
    // x0 = aligned offset for GPFIFO within BAR4
    add x23, x20, x0                    // x23 = bar4 + offset = GPFIFO CPU VA
    // Update bump pointer past GPFIFO
    add x1, x0, #GPFIFO_SIZE
    str x1, [x25]                       // bump_ptr += alignment + GPFIFO_SIZE

    // Zero the GPFIFO ring (8KB)
    mov x1, x23                         // GPFIFO VA
    mov x2, #(GPFIFO_SIZE / 8)          // iterations (8 bytes each)
.zero_gpfifo:
    cbz x2, .gpfifo_ready
    str xzr, [x1], #8
    sub x2, x2, #1
    b .zero_gpfifo

.gpfifo_ready:
    // Store GPFIFO base in .data
    adrp x0, gpfifo_base
    add x0, x0, :lo12:gpfifo_base
    str x23, [x0]

    // Zero gpfifo_put
    adrp x0, gpfifo_put
    add x0, x0, :lo12:gpfifo_put
    str xzr, [x0]

    // ================================================================
    // RPC 3: NV_RM_RPC_ALLOC_CHANNEL (function=6)
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
    // hParent = hDevice = 0x01000001
    mov w1, #0x0001
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 4)]   // offset 0x04: hParent
    // hChannel = 0x01000003
    mov w1, #0x0003
    movk w1, #0x0100, lsl #16
    str w1, [x0, #(RPC_HDR_SIZE + 8)]   // offset 0x08: hChannel
    // hClass = HOPPER_CHANNEL_GPFIFO_A = 0xC86F
    mov w1, #HOPPER_CHANNEL_GPFIFO
    str w1, [x0, #(RPC_HDR_SIZE + 12)]  // offset 0x0C: hClass = 0xC86F
    // gpFifoBase (64-bit GPU VA) -- x23 holds GPFIFO cpu VA
    // On GH200: PA == GPU VA, so this is correct if bar4 was identity-mapped
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

    // Poll for response
    adrp x3, rpc_sequence
    add x3, x3, :lo12:rpc_sequence
    ldr x2, [x3]
    sub w2, w2, #1
    mov x0, x19
    mov x1, x22
    bl mq_poll_response
    cbnz x0, .rpc_fail

    // ================================================================
    // Extract channel_id from response
    // mq_poll_response returns x2 = pointer to response payload.
    // The channel_id is the first u32 of the response payload.
    // ================================================================
    ldr w26, [x2]                       // w26 = channel_id

    // Store channel_id
    adrp x0, channel_id_out
    add x0, x0, :lo12:channel_id_out
    str w26, [x0]

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

.rpc_done:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

.size gsp_rpc_alloc_channel, . - gsp_rpc_alloc_channel
