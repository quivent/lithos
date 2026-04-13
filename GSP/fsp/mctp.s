// GSP/fsp/mctp.s -- MCTP + NVDM framing for FSP COT exchange.
//
// Wire format (little-endian, packed):
//   [ MCTP_HEADER  : 7 bytes ] [ NVDM_HEADER : 4 bytes ] [ payload : <=245 bytes ]
//   Total per-packet overhead = 11 bytes; max EMEM packet = 256 bytes.
//
// MCTP header layout (7 bytes, packed, byte-addressed, little endian):
//   byte 0 : reserved/version   (0x01   -- MCTP base protocol revision)
//   byte 1 : destination EID    (0x00   -- FSP)
//   byte 2 : source EID         (0x21   -- host RM / lithos)
//   byte 3 : flags/tag          SOM[7] | EOM[6] | PKT_SEQ[5:4] | TO[3] | MSG_TAG[2:0]
//   byte 4 : msg type           0x7E   (vendor-defined PCI)
//   byte 5 : vendor id low      0xDE
//   byte 6 : vendor id high     0x10   -- 0x10DE = NVIDIA
//
// NVDM header (4 bytes):
//   byte 0 : NVDM type (0x13 for COT)
//   byte 1 : reserved / version
//   byte 2 : reserved
//   byte 3 : reserved
//
// som_eom_flags argument to mctp_packetize(): low 2 bits encode packet state
//   0 = START      (SOM=1, EOM=0)
//   1 = MIDDLE     (SOM=0, EOM=0)
//   2 = END        (SOM=0, EOM=1)
//   3 = SINGLE     (SOM=1, EOM=1)
// Upper bits carry the running 2-bit packet sequence number (bits 4:5 of byte 3).
//
// AAPCS64, no libc, no frame pointer in leaf helpers, stp/ldp for spills.
//
// Exports:
//   mctp_packetize       -- build one MCTP+NVDM packet in a caller buffer
//   mctp_send_payload    -- chunk + ship a full payload via fsp_emem_write
//   mctp_depacketize     -- strip headers, copy payload to caller buffer
//
// External symbol (from fsp/emem_xfer.s):
//   fsp_emem_write(bar0=x0, channel=w1, src=x2, len_bytes=w3) -> w0
//   fsp_queue_advance_head(bar0=x0, channel=w1, n_bytes=w2)   -> w0
//
// Build/verify:
//   as -o /dev/null /home/ubuntu/lithos/GSP/fsp/mctp.s

    .arch armv8-a
    .text

// ---- constants ----
    .equ MCTP_HEADER_SIZE,      7
    .equ NVDM_HEADER_SIZE,      4
    .equ MCTP_PKT_OVERHEAD,     11          // 7 + 4
    .equ MCTP_PKT_MAX,          256
    .equ MCTP_PAYLOAD_MAX,      245         // 256 - 11

    .equ MCTP_REV,              0x01
    .equ MCTP_DST_EID,          0x00        // FSP endpoint
    .equ MCTP_SRC_EID,          0x21        // host RM
    .equ MCTP_MSGTYPE_VDM,      0x7E        // vendor-defined PCI
    .equ MCTP_VID_LO,           0xDE
    .equ MCTP_VID_HI,           0x10        // 0x10DE = NVIDIA
    .equ NVDM_TYPE_COT,         0x13

    .equ FLAG_SOM_BIT,          7
    .equ FLAG_EOM_BIT,          6

    .equ STATE_START,           0
    .equ STATE_MIDDLE,          1
    .equ STATE_END,             2
    .equ STATE_SINGLE,          3

    .globl mctp_packetize
    .globl mctp_send_payload
    .globl mctp_depacketize
    .extern fsp_emem_write
    .extern fsp_queue_advance_head

// ============================================================================
// mctp_packetize(src_payload, payload_len, dst_buf, som_eom_flags)
//   x0 = src_payload      (input; may be NULL if payload_len == 0)
//   w1 = payload_len      (0..245)
//   x2 = dst_buf          (writable, >= payload_len + 11 bytes)
//   w3 = som_eom_flags    (bits 0..1 = state; bits 4..5 = pkt seq)
// Returns:
//   w0 = total packet length in bytes (payload_len + 11)
//
// Writes the 7-byte MCTP header, then the 4-byte NVDM header (COT type),
// then copies payload_len bytes from x0 to x2+11 byte-by-byte (src buffer
// may be unaligned; total byte count is small and bounded by 245).
// ============================================================================
mctp_packetize:
    // Clamp payload_len to MCTP_PAYLOAD_MAX defensively (no error; caller
    // is responsible, we just prevent buffer overflow on the dst_buf).
    mov     w9, #MCTP_PAYLOAD_MAX
    cmp     w1, w9
    csel    w1, w9, w1, hi              // w1 = min(w1, 245)

    // ---- decode state / seq into a single flag byte ----
    and     w10, w3, #0x3               // state in w10
    and     w11, w3, #0x30              // pkt_seq<<4 preserved in w11 (bits 4..5)
    // Default: SOM=0, EOM=0 (middle).
    mov     w12, #0
    cmp     w10, #STATE_START
    b.ne    1f
    orr     w12, w12, #(1 << FLAG_SOM_BIT)
    b       4f
1:
    cmp     w10, #STATE_END
    b.ne    2f
    orr     w12, w12, #(1 << FLAG_EOM_BIT)
    b       4f
2:
    cmp     w10, #STATE_SINGLE
    b.ne    3f
    orr     w12, w12, #(1 << FLAG_SOM_BIT)
    orr     w12, w12, #(1 << FLAG_EOM_BIT)
    b       4f
3:
    // STATE_MIDDLE / unknown -- leave SOM=EOM=0.
4:
    orr     w12, w12, w11               // fold packet sequence into bits 4..5

    // ---- emit MCTP header (7 bytes) ----
    mov     w13, #MCTP_REV
    strb    w13, [x2, #0]
    mov     w13, #MCTP_DST_EID
    strb    w13, [x2, #1]
    mov     w13, #MCTP_SRC_EID
    strb    w13, [x2, #2]
    strb    w12, [x2, #3]
    mov     w13, #MCTP_MSGTYPE_VDM
    strb    w13, [x2, #4]
    mov     w13, #MCTP_VID_LO
    strb    w13, [x2, #5]
    mov     w13, #MCTP_VID_HI
    strb    w13, [x2, #6]

    // ---- emit NVDM header (4 bytes) ----
    mov     w13, #NVDM_TYPE_COT
    strb    w13, [x2, #7]
    strb    wzr, [x2, #8]
    strb    wzr, [x2, #9]
    strb    wzr, [x2, #10]

    // ---- copy payload bytes: x0 -> x2 + 11, count = w1 ----
    add     x5, x2, #MCTP_PKT_OVERHEAD  // dst cursor
    mov     x6, x0                      // src cursor
    mov     w7, w1                      // byte counter

    // Fast path: 8-byte chunks while >=8 bytes remain.
    cmp     w7, #8
    b.lo    .Lcopy_tail
.Lcopy_qword:
    ldr     x8, [x6], #8
    str     x8, [x5], #8
    sub     w7, w7, #8
    cmp     w7, #8
    b.hs    .Lcopy_qword
.Lcopy_tail:
    cbz     w7, .Lcopy_done
.Lcopy_byte:
    ldrb    w8, [x6], #1
    strb    w8, [x5], #1
    subs    w7, w7, #1
    b.ne    .Lcopy_byte
.Lcopy_done:

    // ---- return total packet length = payload_len + 11 ----
    add     w0, w1, #MCTP_PKT_OVERHEAD
    ret

// ============================================================================
// mctp_send_payload(bar0, channel, payload, payload_len)
//   x0 = bar0
//   w1 = channel
//   x2 = payload
//   w3 = payload_len
// Returns:
//   w0 = 0 on success, negative errno from fsp_emem_write on failure
//
// Splits [payload, payload_len) into 245-byte MCTP packets, sets SOM on
// first, EOM on last, increments a 2-bit packet-sequence counter modulo 4,
// and for each packet:
//   1. Formats into a stack-resident 256-byte scratch (mctp_packetize).
//   2. Calls fsp_emem_write(bar0, channel, scratch, pkt_len).
//   3. On any non-zero return, aborts and propagates that status.
// After all packets are written, advances QUEUE_HEAD by total bytes sent
// via fsp_queue_advance_head.
//
// Stack frame (64-byte aligned):
//   [sp+0..255]  scratch packet buffer (MCTP_PKT_MAX)
//   [sp+256..]   callee-save spills
// Total frame = 256 + 96 = 352 bytes (16-byte aligned).
// ============================================================================
    .equ SEND_FRAME,            352
    .equ SEND_BUF_OFF,          0
    .equ SEND_SAVE_OFF,         256

mctp_send_payload:
    stp     x29, x30, [sp, #-SEND_FRAME]!
    mov     x29, sp
    stp     x19, x20, [sp, #SEND_SAVE_OFF + 0]
    stp     x21, x22, [sp, #SEND_SAVE_OFF + 16]
    stp     x23, x24, [sp, #SEND_SAVE_OFF + 32]
    stp     x25, x26, [sp, #SEND_SAVE_OFF + 48]
    stp     x27, x28, [sp, #SEND_SAVE_OFF + 64]

    // Preserve inputs in callee-saved regs.
    mov     x19, x0                     // bar0
    mov     w20, w1                     // channel
    mov     x21, x2                     // payload cursor
    mov     w22, w3                     // bytes remaining
    mov     w23, w3                     // total bytes (for head advance)
    mov     w24, #0                     // packet sequence counter (2 bits)
    mov     w25, #0                     // total bytes transmitted on the wire

    add     x26, sp, #SEND_BUF_OFF      // scratch buffer base

    // Handle zero-length payload: send one SINGLE packet with empty payload.
    cbnz    w22, .Lsend_loop
    mov     w3, #STATE_SINGLE           // flags = SINGLE, seq=0
    mov     w1, #0                      // payload_len = 0
    mov     x0, xzr                     // src = NULL (len=0 so not dereferenced)
    mov     x2, x26
    bl      mctp_packetize
    // w0 = packet length (= 11)
    mov     w27, w0                     // save pkt_len
    mov     x0, x19
    mov     w1, w20
    mov     x2, x26
    mov     w3, w27
    bl      fsp_emem_write
    cbnz    w0, .Lsend_fail
    add     w25, w25, w27
    b       .Lsend_done

.Lsend_loop:
    // Determine this packet's payload chunk size (min(remaining, 245)).
    mov     w9, #MCTP_PAYLOAD_MAX
    cmp     w22, w9
    csel    w28, w22, w9, ls            // w28 = chunk_len

    // Compute state flags.
    //   first = (bytes_remaining == total)        -> SOM
    //   last  = (chunk_len == bytes_remaining)    -> EOM
    cmp     w22, w23
    cset    w10, eq                     // w10 = first?
    cmp     w28, w22
    cset    w11, eq                     // w11 = last?

    // state = (first<<1)|last ... but we want explicit states:
    //   first && last  -> SINGLE  (3)
    //   first && !last -> START   (0)
    //   !first && last -> END     (2)
    //   else           -> MIDDLE  (1)
    // Build via compare tree.
    cbz     w10, .Lnot_first
    cbz     w11, .Lset_start
    mov     w12, #STATE_SINGLE
    b       .Lstate_done
.Lset_start:
    mov     w12, #STATE_START
    b       .Lstate_done
.Lnot_first:
    cbz     w11, .Lset_mid
    mov     w12, #STATE_END
    b       .Lstate_done
.Lset_mid:
    mov     w12, #STATE_MIDDLE
.Lstate_done:

    // Combine state with packet-sequence counter (bits 4..5).
    and     w13, w24, #0x3
    lsl     w13, w13, #4
    orr     w12, w12, w13

    // mctp_packetize(src=x21, len=w28, dst=x26, flags=w12)
    mov     x0, x21
    mov     w1, w28
    mov     x2, x26
    mov     w3, w12
    bl      mctp_packetize
    mov     w27, w0                     // w27 = pkt_len (<=256)

    // fsp_emem_write(bar0=x19, channel=w20, src=x26, len=w27)
    mov     x0, x19
    mov     w1, w20
    mov     x2, x26
    mov     w3, w27
    bl      fsp_emem_write
    cbnz    w0, .Lsend_fail

    // Advance counters / cursors.
    add     w25, w25, w27
    add     x21, x21, x28, uxtw         // payload cursor += chunk_len
    sub     w22, w22, w28
    add     w24, w24, #1                // seq++ (used modulo 4 next iter)

    cbnz    w22, .Lsend_loop

.Lsend_done:
    // Advance QUEUE_HEAD by total wire bytes. fsp_queue_advance_head takes
    // (bar0=x0, channel=w1, n_bytes=w2).
    mov     x0, x19
    mov     w1, w20
    mov     w2, w25
    bl      fsp_queue_advance_head
    // Treat its return as the final status. Zero = success.

.Lsend_ret:
    ldp     x19, x20, [sp, #SEND_SAVE_OFF + 0]
    ldp     x21, x22, [sp, #SEND_SAVE_OFF + 16]
    ldp     x23, x24, [sp, #SEND_SAVE_OFF + 32]
    ldp     x25, x26, [sp, #SEND_SAVE_OFF + 48]
    ldp     x27, x28, [sp, #SEND_SAVE_OFF + 64]
    ldp     x29, x30, [sp], #SEND_FRAME
    ret

.Lsend_fail:
    // w0 already carries the nonzero error status from fsp_emem_write.
    b       .Lsend_ret

// ============================================================================
// mctp_depacketize(src_buf, src_len, dst_buf, dst_max)
//   x0 = src_buf       (MCTP+NVDM framed packet, incoming from EMEM)
//   w1 = src_len       (bytes in src_buf, must be >= 11)
//   x2 = dst_buf       (output buffer for payload)
//   w3 = dst_max       (maximum bytes to write into dst_buf)
// Returns:
//   w0 = payload length on success
//   w0 = -1 (0xFFFFFFFF) on framing error (too short, bad msgtype, bad VID)
//   w0 = -2 if payload would exceed dst_max
//
// Validates:
//   - src_len >= 11
//   - msg_type byte (offset 4) == 0x7E
//   - vendor id (bytes 5..6) == 0x10DE
//   - NVDM type (byte 7) == 0x13 (COT)
// Then copies (src_len - 11) bytes from src_buf+11 to dst_buf.
// ============================================================================
mctp_depacketize:
    cmp     w1, #MCTP_PKT_OVERHEAD
    b.lo    .Ldepk_eframe

    // Validate msg type.
    ldrb    w9, [x0, #4]
    cmp     w9, #MCTP_MSGTYPE_VDM
    b.ne    .Ldepk_eframe

    // Validate vendor id (little endian, bytes 5..6 = 0xDE, 0x10).
    ldrb    w9, [x0, #5]
    cmp     w9, #MCTP_VID_LO
    b.ne    .Ldepk_eframe
    ldrb    w9, [x0, #6]
    cmp     w9, #MCTP_VID_HI
    b.ne    .Ldepk_eframe

    // Validate NVDM type.
    ldrb    w9, [x0, #7]
    cmp     w9, #NVDM_TYPE_COT
    b.ne    .Ldepk_eframe

    // payload_len = src_len - 11.
    sub     w10, w1, #MCTP_PKT_OVERHEAD

    // Bound-check against dst_max.
    cmp     w10, w3
    b.hi    .Ldepk_eoverflow

    // Copy payload_len bytes from src+11 to dst.
    add     x5, x0, #MCTP_PKT_OVERHEAD  // src cursor
    mov     x6, x2                      // dst cursor
    mov     w7, w10                     // byte counter
    cbz     w7, .Ldepk_ok

    cmp     w7, #8
    b.lo    .Ldepk_tail
.Ldepk_qword:
    ldr     x8, [x5], #8
    str     x8, [x6], #8
    sub     w7, w7, #8
    cmp     w7, #8
    b.hs    .Ldepk_qword
.Ldepk_tail:
    cbz     w7, .Ldepk_ok
.Ldepk_byte:
    ldrb    w8, [x5], #1
    strb    w8, [x6], #1
    subs    w7, w7, #1
    b.ne    .Ldepk_byte

.Ldepk_ok:
    mov     w0, w10
    ret

.Ldepk_eframe:
    mov     w0, #-1
    ret

.Ldepk_eoverflow:
    mov     w0, #-2
    ret

    .section .note.GNU-stack,"",%progbits
