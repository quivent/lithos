// GSP/fsp/mctp.s -- MCTP + NVDM framing for FSP COT exchange.
//
// Wire format (little-endian, packed):
//   SOM packets:  [ MCTP transport DW : 4 B ] [ NVDM/msg DW : 4 B ] [ payload : <=248 B ]
//   CONT packets: [ MCTP transport DW : 4 B ] [ payload : <=252 B ]
//   SOM overhead = 8 bytes (2 DWs); continuation overhead = 4 bytes (1 DW).
//   Max EMEM packet = 256 bytes.
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
// NVDM header (4 bytes, SOM packets only):
//   byte 0 : NVDM type (0x14 for COT)
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
//   fsp_emem_write(bar0=x0, channel=w1, offset=w2, src=x3, len_bytes=w4) -> w0
//   fsp_queue_advance_head(bar0=x0, channel=w1, new_head=w2)              -> w0
//
// Build/verify:
//   as -o /dev/null /home/ubuntu/lithos/GSP/fsp/mctp.s

    .arch armv8-a
    .text

// ---- constants ----
    .equ MCTP_HEADER_SIZE,      7
    .equ NVDM_HEADER_SIZE,      4
    .equ MCTP_PKT_OVERHEAD_SOM,  8          // 2 DWs: MCTP transport + NVDM type
    .equ MCTP_PKT_OVERHEAD_CONT, 4         // 1 DW:  MCTP transport only
    .equ MCTP_PKT_MAX,          256
    .equ MCTP_PAYLOAD_MAX_SOM,  248         // 256 - 8
    .equ MCTP_PAYLOAD_MAX_CONT, 252         // 256 - 4
    .equ FSP_PACKET_SIZE_BYTES, 256         // each packet occupies a full 256-B slot

    .equ MCTP_REV,              0x01
    .equ MCTP_DST_EID,          0x00        // FSP endpoint
    .equ MCTP_SRC_EID,          0x21        // host RM
    .equ MCTP_MSGTYPE_VDM,      0x7E        // vendor-defined PCI
    .equ MCTP_VID_LO,           0xDE
    .equ MCTP_VID_HI,           0x10        // 0x10DE = NVIDIA
    .equ NVDM_TYPE_COT,         0x14

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
//   w1 = payload_len      (0..248 for SOM, 0..252 for continuation)
//   x2 = dst_buf          (writable, >= payload_len + overhead bytes)
//   w3 = som_eom_flags    (bits 0..1 = state; bits 4..5 = pkt seq)
// Returns:
//   w0 = total packet length in bytes
//
// SOM packets (SOM=1): writes 4-byte MCTP transport DW, then 4-byte NVDM
// header DW (8 bytes total overhead), then copies payload.
// Continuation/END packets (SOM=0): writes 4-byte MCTP transport DW only
// (4 bytes overhead), then copies payload.
// ============================================================================
mctp_packetize:
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

    // ---- determine if SOM: check bit 7 of w12 ----
    tst     w12, #(1 << FLAG_SOM_BIT)
    b.eq    .Lpkt_cont

    // ---- SOM path: clamp payload to MCTP_PAYLOAD_MAX_SOM ----
    mov     w9, #MCTP_PAYLOAD_MAX_SOM
    cmp     w1, w9
    csel    w1, w9, w1, hi              // w1 = min(w1, 248)

    // ---- emit MCTP transport header (4 bytes / 1 DW) ----
    mov     w13, #MCTP_REV
    strb    w13, [x2, #0]
    mov     w13, #MCTP_DST_EID
    strb    w13, [x2, #1]
    mov     w13, #MCTP_SRC_EID
    strb    w13, [x2, #2]
    strb    w12, [x2, #3]

    // ---- emit NVDM / message-type header (4 bytes / 1 DW, SOM only) ----
    mov     w13, #MCTP_MSGTYPE_VDM
    strb    w13, [x2, #4]
    mov     w13, #MCTP_VID_LO
    strb    w13, [x2, #5]
    mov     w13, #MCTP_VID_HI
    strb    w13, [x2, #6]
    mov     w13, #NVDM_TYPE_COT
    strb    w13, [x2, #7]

    // ---- copy payload: x0 -> x2 + 8, count = w1 ----
    add     x5, x2, #MCTP_PKT_OVERHEAD_SOM
    mov     w14, #MCTP_PKT_OVERHEAD_SOM  // save overhead for return calc
    b       .Lpkt_copy

.Lpkt_cont:
    // ---- continuation/END path: clamp payload to MCTP_PAYLOAD_MAX_CONT ----
    mov     w9, #MCTP_PAYLOAD_MAX_CONT
    cmp     w1, w9
    csel    w1, w9, w1, hi              // w1 = min(w1, 252)

    // ---- emit MCTP transport header only (4 bytes / 1 DW) ----
    mov     w13, #MCTP_REV
    strb    w13, [x2, #0]
    mov     w13, #MCTP_DST_EID
    strb    w13, [x2, #1]
    mov     w13, #MCTP_SRC_EID
    strb    w13, [x2, #2]
    strb    w12, [x2, #3]

    // ---- copy payload: x0 -> x2 + 4, count = w1 ----
    add     x5, x2, #MCTP_PKT_OVERHEAD_CONT
    mov     w14, #MCTP_PKT_OVERHEAD_CONT // save overhead for return calc

.Lpkt_copy:
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

    // ---- return total packet length = payload_len + overhead ----
    add     w0, w1, w14
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
// Splits [payload, payload_len) into MCTP packets (248-byte payload on the
// SOM packet, 252-byte payload on continuation packets), sets SOM on first,
// EOM on last, increments a 2-bit packet-sequence counter modulo 4, and for
// each packet:
//   1. Formats into a stack-resident 256-byte scratch (mctp_packetize).
//   2. Calls fsp_emem_write(bar0, channel, offset, scratch, pkt_len).
//   3. On any non-zero return, aborts and propagates that status.
// After all packets are written, advances QUEUE_HEAD by
// num_packets * FSP_PACKET_SIZE_BYTES (256-byte aligned slots).
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
    mov     w23, w3                     // total bytes (for first-packet check)
    mov     w24, #0                     // packet sequence counter (2 bits)
    mov     w25, #0                     // packet index (for EMEM offset calc)

    add     x26, sp, #SEND_BUF_OFF      // scratch buffer base

    // Handle zero-length payload: send one SINGLE packet with empty payload.
    cbnz    w22, .Lsend_loop
    mov     w3, #STATE_SINGLE           // flags = SINGLE, seq=0
    mov     w1, #0                      // payload_len = 0
    mov     x0, xzr                     // src = NULL (len=0 so not dereferenced)
    mov     x2, x26
    bl      mctp_packetize
    mov     w27, w0                     // save pkt_len
    // fsp_emem_write(bar0, channel, offset=0, src, len)
    mov     x0, x19
    mov     w1, w20
    mov     w2, #0                      // byte_offset = pkt_index(0) * 256
    mov     x3, x26
    mov     w4, w27
    bl      fsp_emem_write
    cbnz    w0, .Lsend_fail
    mov     w25, #1                     // 1 packet sent
    b       .Lsend_done

.Lsend_loop:
    // Determine this packet's max payload based on SOM vs continuation.
    //   first = (bytes_remaining == total)        -> SOM  (max 248)
    //   !first                                    -> CONT (max 252)
    cmp     w22, w23
    b.ne    .Lsend_cont_max
    mov     w9, #MCTP_PAYLOAD_MAX_SOM
    b       .Lsend_clamp
.Lsend_cont_max:
    mov     w9, #MCTP_PAYLOAD_MAX_CONT
.Lsend_clamp:
    cmp     w22, w9
    csel    w28, w22, w9, ls            // w28 = chunk_len

    // Compute state flags.
    //   first = (bytes_remaining == total)        -> SOM
    //   last  = (chunk_len == bytes_remaining)    -> EOM
    cmp     w22, w23
    cset    w10, eq                     // w10 = first?
    cmp     w28, w22
    cset    w11, eq                     // w11 = last?

    // state = explicit states:
    //   first && last  -> SINGLE  (3)
    //   first && !last -> START   (0)
    //   !first && last -> END     (2)
    //   else           -> MIDDLE  (1)
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

    // fsp_emem_write(bar0, channel, offset, src, len)
    //   offset = pkt_index * 256 (each packet occupies a 256-B EMEM slot)
    mov     x0, x19
    mov     w1, w20
    lsl     w2, w25, #8                 // byte_offset = pkt_index * 256
    mov     x3, x26
    mov     w4, w27
    bl      fsp_emem_write
    cbnz    w0, .Lsend_fail

    // Advance counters / cursors.
    add     w25, w25, #1                // pkt_index++
    add     x21, x21, x28, uxtw         // payload cursor += chunk_len
    sub     w22, w22, w28
    add     w24, w24, #1                // seq++ (used modulo 4 next iter)

    cbnz    w22, .Lsend_loop

.Lsend_done:
    // Advance QUEUE_HEAD by num_packets * 256 (each packet is a full
    // 256-byte EMEM slot). fsp_queue_advance_head(bar0, channel, new_head).
    mov     x0, x19
    mov     w1, w20
    lsl     w2, w25, #8                 // new_head = num_packets * 256
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
//   w1 = src_len       (bytes in src_buf, must be >= 4)
//   x2 = dst_buf       (output buffer for payload)
//   w3 = dst_max       (maximum bytes to write into dst_buf)
// Returns:
//   w0 = payload length on success
//   w0 = -1 (0xFFFFFFFF) on framing error (too short, bad msgtype, bad VID)
//   w0 = -2 if payload would exceed dst_max
//
// Checks SOM bit (byte 3, bit 7) to determine header size:
//   SOM packets: validates msg_type, vendor id, NVDM type in DW1 (bytes 4..7),
//                overhead = 8 bytes, payload starts at offset 8.
//   Continuation packets: overhead = 4 bytes, payload starts at offset 4.
// ============================================================================
mctp_depacketize:
    // Minimum packet size is 4 bytes (transport header only).
    cmp     w1, #MCTP_PKT_OVERHEAD_CONT
    b.lo    .Ldepk_eframe

    // Check SOM bit in byte 3 (flags/tag) to determine header size.
    ldrb    w9, [x0, #3]
    tst     w9, #(1 << FLAG_SOM_BIT)
    b.eq    .Ldepk_cont

    // ---- SOM packet: 8 bytes overhead, validate DW1 fields ----
    cmp     w1, #MCTP_PKT_OVERHEAD_SOM
    b.lo    .Ldepk_eframe

    // Validate msg type (byte 4).
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

    // Validate NVDM type (byte 7).
    ldrb    w9, [x0, #7]
    cmp     w9, #NVDM_TYPE_COT
    b.ne    .Ldepk_eframe

    // payload_len = src_len - 8.
    sub     w10, w1, #MCTP_PKT_OVERHEAD_SOM
    add     x5, x0, #MCTP_PKT_OVERHEAD_SOM  // src cursor at payload start
    b       .Ldepk_check_bounds

.Ldepk_cont:
    // ---- Continuation packet: 4 bytes overhead, no NVDM header ----
    sub     w10, w1, #MCTP_PKT_OVERHEAD_CONT
    add     x5, x0, #MCTP_PKT_OVERHEAD_CONT // src cursor at payload start

.Ldepk_check_bounds:
    // Bound-check against dst_max.
    cmp     w10, w3
    b.hi    .Ldepk_eoverflow

    // Copy payload_len bytes to dst.
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
