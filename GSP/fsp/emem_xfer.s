// GSP/fsp/emem_xfer.s -- Step 7a: FSP EMEM transfer primitives
//
// Low-level word-stream engine over the FSP EMEMC/EMEMD window.  EMEM is a
// ~4 KB scratch SRAM inside the FSP Falcon, carved into 8 independent
// channels.  Access is paging-style:
//
//   1. Program NV_PFSP_EMEMC(ch) with a byte offset and the AINCW (write)
//      or AINCR (read) auto-increment bit.  Low 2 bits of the offset are
//      ignored by hardware (words are 4-byte aligned).  The offset field
//      uses bits[15:2] (BLK[15:8] + OFFS[7:2]); because the byte offset
//      already sits in bits[15:2], we simply OR in the AINC* bit.
//   2. Each read/write of NV_PFSP_EMEMD(ch) transfers one 32-bit dword and
//      auto-advances the internal pointer.
//
// No MCTP, no COT framing -- this file is strictly the dword pump.  See
// fsp/mctp.s for packet framing and fsp/bootcmd.s for the top-level flow.
//
// Register constants (from dev_fsp_pri.h, GH100):
//   NV_PFSP_EMEMC(i)        = 0x008F2AC0 + i*8
//   NV_PFSP_EMEMD(i)        = 0x008F2AC4 + i*8
//   NV_PFSP_QUEUE_HEAD(i)   = 0x008F2C00 + i*8
//   NV_PFSP_QUEUE_TAIL(i)   = 0x008F2C04 + i*8
//   NV_PFSP_EMEMC_AINCW     = bit 24
//   NV_PFSP_EMEMC_AINCR     = bit 25
//
// Exports:
//   fsp_emem_write(bar0, channel, offset, src_buf, len_bytes) -> int
//   fsp_emem_read (bar0, channel, offset, dst_buf, len_bytes) -> int
//   fsp_queue_advance_head(bar0, channel, new_head)              -> int
//
// AAPCS64.  No libc.  Clobbers x0-x9 only; all callee-saves preserved.
//
// Build: as -o emem_xfer.o emem_xfer.s

// ---- EMEMC offset field mask (bits[15:2]) ----
// Byte offset within EMEM -- we OR this directly into EMEMC.
.equ EMEMC_OFFS_MASK,           0xFFFC

// Per-channel EMEM size (bytes).  Channel 0 command queue is 512 B;
// we use this as the upper bound for offset+len validation.
.equ FSP_EMEM_CHANNEL_SIZE,     512

.text

// --------------------------------------------------------------------
// Helper macro: materialize BAR0 + (REGBASE + channel*8) into Xdst.
// REGBASE is a 32-bit constant (e.g. 0x008F2AC0); channel_w is a w-reg
// holding the channel index (0..7); xbar0 holds the BAR0 base VA.
// Uses wtmp (a w scratch reg) internally.
// --------------------------------------------------------------------
.macro bar0_reg_addr xdst, xbar0, channel_w, wtmp, regbase
    // wtmp = regbase (32-bit).  Use movz+movk so any value works.
    movz    \wtmp, #((\regbase) & 0xFFFF)
    movk    \wtmp, #(((\regbase) >> 16) & 0xFFFF), lsl #16
    // wtmp += channel * 8
    add     \wtmp, \wtmp, \channel_w, lsl #3
    // xdst = xbar0 + (uxtw wtmp)
    add     \xdst, \xbar0, \wtmp, uxtw
.endm


// ====================================================================
// fsp_emem_write -- push bytes to FSP EMEM via the EMEMC/EMEMD window
//
// Rounds len_bytes up to a 4-byte multiple; any trailing bytes in the
// final dword are read from src_buf (caller is expected to have padded
// with zeros or don't-cares up to the next 4-byte boundary, as is the
// case for MCTP packets which are always 256-byte framed).
//
// Inputs:
//   x0 = bar0 base (mmap'd BAR0 VA)
//   w1 = channel index (0..7)
//   w2 = byte offset into EMEM (must be 4-byte aligned; low 2 bits ignored)
//   x3 = source buffer pointer
//   w4 = length in bytes
//
// Returns:
//   x0 = 0 on success
// ====================================================================
.globl fsp_emem_write
.type  fsp_emem_write, %function
.balign 4
fsp_emem_write:
    // Validate channel index (must be 0..7)
    cmp     w1, #7
    b.hi    .Lwrite_bad_channel

    // Bounds check: offset + len_bytes must not exceed channel size.
    add     w8, w2, w4                          // w8 = offset + len_bytes
    cmp     w8, w2                              // check for u32 wraparound
    b.lo    .Lwrite_bad_bounds
    cmp     w8, #FSP_EMEM_CHANNEL_SIZE
    b.hi    .Lwrite_bad_bounds

    // Compute EMEMC / EMEMD MMIO addresses for this channel.
    //   x7  = &EMEMC(ch)
    //   x9  = &EMEMD(ch) = &EMEMC(ch) + 4
    bar0_reg_addr x7, x0, w1, w8, 0x008F2AC0
    add     x9, x7, #4

    // Build EMEMC value: (offset & 0xFFFC) | AINCW (bit 24 = 0x01000000).
    mov     w8, #EMEMC_OFFS_MASK
    and     w6, w2, w8
    movk    w6, #0x0100, lsl #16            // OR in AINCW bit

    // Program EMEMC: set byte offset and arm auto-increment on write.
    str     w6, [x7]
    dsb     st                              // commit EMEMC cursor before data writes

    // Convert len_bytes -> len_dwords, rounding up: (len + 3) >> 2
    add     w5, w4, #3
    lsr     w5, w5, #2

    cbz     w5, .Lwrite_done

    mov     x6, x3                          // src cursor
.Lwrite_loop:
    ldr     w8, [x6], #4                    // load next 32-bit word
    str     w8, [x9]                        // write to EMEMD (auto-advances)
    sub     w5, w5, #1
    cbnz    w5, .Lwrite_loop

.Lwrite_done:
    dsb     st                              // fence all EMEMD writes before return
    mov     x0, #0
    ret

.Lwrite_bad_channel:
    mov     x0, #-1                         // invalid channel index
    ret

.Lwrite_bad_bounds:
    mov     x0, #-2                         // offset + len exceeds channel size
    ret


// ====================================================================
// fsp_emem_read -- drain bytes from FSP EMEM via the EMEMC/EMEMD window
//
// Rounds len_bytes up to a 4-byte multiple; the caller's buffer must
// have room for the rounded-up size.
//
// Inputs:
//   x0 = bar0 base
//   w1 = channel index
//   w2 = byte offset into EMEM (4-byte aligned)
//   x3 = destination buffer
//   w4 = length in bytes
//
// Returns:
//   x0 = 0 on success
// ====================================================================
.globl fsp_emem_read
.type  fsp_emem_read, %function
.balign 4
fsp_emem_read:
    // Validate channel index (must be 0..7)
    cmp     w1, #7
    b.hi    .Lread_bad_channel

    // Bounds check: offset + len_bytes must not exceed channel size.
    add     w8, w2, w4                          // w8 = offset + len_bytes
    cmp     w8, w2                              // check for u32 wraparound
    b.lo    .Lread_bad_bounds
    cmp     w8, #FSP_EMEM_CHANNEL_SIZE
    b.hi    .Lread_bad_bounds

    bar0_reg_addr x7, x0, w1, w8, 0x008F2AC0
    add     x9, x7, #4                      // EMEMD addr

    // Build EMEMC value: (offset & 0xFFFC) | AINCR (bit 25 = 0x02000000).
    mov     w8, #EMEMC_OFFS_MASK
    and     w6, w2, w8
    movk    w6, #0x0200, lsl #16            // OR in AINCR bit

    // Program EMEMC: set byte offset and arm auto-increment on read.
    str     w6, [x7]
    dsb     st                              // commit EMEMC cursor before data reads
    isb                                     // serialize before EMEMD load sequence

    // Convert len_bytes -> len_dwords, rounding up.
    add     w5, w4, #3
    lsr     w5, w5, #2

    cbz     w5, .Lread_done

    mov     x6, x3                          // dst cursor
.Lread_loop:
    ldr     w8, [x9]                        // read dword from EMEMD (auto-advances)
    str     w8, [x6], #4
    sub     w5, w5, #1
    cbnz    w5, .Lread_loop

.Lread_done:
    mov     x0, #0
    ret

.Lread_bad_channel:
    mov     x0, #-1                         // invalid channel index
    ret

.Lread_bad_bounds:
    mov     x0, #-2                         // offset + len exceeds channel size
    ret


// ====================================================================
// fsp_queue_advance_head -- publish a new CPU->FSP queue head to signal FSP
//
// Writes the new byte-offset head into NV_PFSP_QUEUE_HEAD(channel).
// A DSB SY barrier before the store flushes all prior EMEM dword writes;
// a second DSB SY after the store fences subsequent host activity from
// racing the doorbell observed by the FSP.
//
// Inputs:
//   x0 = bar0 base
//   w1 = channel index
//   w2 = new head value (byte offset into the command ring)
//
// Returns:
//   x0 = 0
// ====================================================================
.globl fsp_queue_advance_head
.type  fsp_queue_advance_head, %function
.balign 4
fsp_queue_advance_head:
    // Validate channel index (must be 0..7)
    cmp     w1, #7
    b.hi    .Lhead_bad_channel

    // x7 = &QUEUE_HEAD(ch)
    bar0_reg_addr x7, x0, w1, w8, 0x008F2C00

    // Ensure all prior EMEM dword writes are globally observable before
    // we bump the head register (the FSP's doorbell).
    dsb     sy
    str     w2, [x7]
    dsb     sy

    mov     x0, #0
    ret

.Lhead_bad_channel:
    mov     x0, #-1                         // invalid channel index
    ret
