// fsp/bootcmd.s -- Step 7: Top-level FSP Chain-of-Trust boot command
//
// Sends the FSP the signed GSP-FMC Chain-of-Trust (COT) boot payload,
// waits for its verdict, and acknowledges the response.  On success,
// FSP releases PRIV_LOCKDOWN internally so poll_lockdown (Step 8) can
// observe the bit drop.
//
// This file is the only externally-callable entry point under GSP/fsp/.
// It orchestrates the following helpers (each in its own .s file):
//
//   fsp_init            -- verify FSP is ready (scratch group 2 magic,
//                          queue/msgq empty).  bar0 in x0.
//   cot_build_payload   -- populate an NVDM_PAYLOAD_COT struct.
//                          x0 = fw_image_bar4_cpu, x1 = manifest_off,
//                          x2 = dst buffer.  (See fsp/cot_payload.s.)
//   mctp_send_payload   -- frame + write packets to CPU->FSP EMEM queue.
//                          x0 = bar0, x1 = channel (0), x2 = payload ptr,
//                          x3 = payload len.
//   fsp_response_poll   -- wait for MSGQ_HEAD != MSGQ_TAIL, with timeout.
//                          x0 = bar0, x1 = channel, x2 = timeout_secs.
//                          Returns 0 / -1 on timeout.
//   fsp_emem_read       -- drain one response packet from EMEM into buffer.
//                          x0 = bar0, x1 = channel, x2 = byte_offset,
//                          x3 = dst, x4 = len_bytes.
//   fsp_response_parse  -- validate MCTP+NVDM hdrs, extract status dword.
//                          x0 = buf, x1 = len.  Returns FSP status (0 ok).
//
// Calling convention: AAPCS64.  x0 = bar0_base.  Returns x0 = 0 on
// success, negative on failure.  Does not SYS_EXIT; caller (boot.s)
// prints a banner and decides how to abort.

.equ SYS_WRITE,     64

.equ COT_PAYLOAD_SIZE,      860      // NVDM_PAYLOAD_COT, byte-exact
.equ COT_SCRATCH_STACK,     896      // 860 bumped to 16-byte alignment
.equ RESP_BUF_SIZE,         256      // one EMEM packet
.equ FSP_CHANNEL,           0
.equ FSP_EMEM_CH0_MSGQ_OFFSET, 512   // FSP->CPU response queue starts at EMEM byte 512
.equ NVDM_TYPE_COT,         0x14     // per kern_fsp.h (see fsp_plan.md)

// ---- Error codes returned in x0 ----
.equ FSP_ERR_INIT,       -1
.equ FSP_ERR_BUILD,      -2
.equ FSP_ERR_SEND,       -3
.equ FSP_ERR_POLL,       -4
.equ FSP_ERR_EMEM_READ,  -5
.equ FSP_ERR_REJECTED,   -6

// ============================================================
// External symbols
// ============================================================
.extern fsp_init
.extern cot_build_payload
.extern mctp_send_payload
.extern fsp_emem_read
.extern fsp_response_poll
.extern fsp_response_parse
.extern fsp_diag_print          // weakly-used; see .weak below

.extern fw_bar4_cpu
.extern fw_manifest_offset

.weak  fsp_diag_print

// ============================================================
// Data -- progress messages (stderr)
// ============================================================
.data
.align 3

msg_fsp_begin:       .asciz "gsp[fsp]: begin Chain-of-Trust bootcmd\n"
msg_fsp_begin_len    = . - msg_fsp_begin - 1

msg_fsp_init:        .asciz "gsp[fsp]: step 1/7 -- fsp_init (check FSP ready)...\n"
msg_fsp_init_len     = . - msg_fsp_init - 1
msg_fsp_init_fail:   .asciz "gsp[fsp]: fsp_init FAILED -- FSP not ready\n"
msg_fsp_init_fail_len= . - msg_fsp_init_fail - 1

msg_fsp_build:       .asciz "gsp[fsp]: step 2/7 -- cot_build_payload (860 B COT struct)...\n"
msg_fsp_build_len    = . - msg_fsp_build - 1
msg_fsp_build_fail:  .asciz "gsp[fsp]: cot_build_payload FAILED\n"
msg_fsp_build_fail_len= . - msg_fsp_build_fail - 1

msg_fsp_send:        .asciz "gsp[fsp]: step 3/7 -- mctp_send_payload (packetize + QUEUE_HEAD advance)...\n"
msg_fsp_send_len     = . - msg_fsp_send - 1
msg_fsp_send_fail:   .asciz "gsp[fsp]: mctp_send_payload FAILED\n"
msg_fsp_send_fail_len= . - msg_fsp_send_fail - 1

msg_fsp_poll:        .asciz "gsp[fsp]: step 4/7 -- fsp_response_poll (await MSGQ_HEAD)...\n"
msg_fsp_poll_len     = . - msg_fsp_poll - 1
msg_fsp_poll_fail:   .asciz "gsp[fsp]: fsp_response_poll FAILED (timeout)\n"
msg_fsp_poll_fail_len= . - msg_fsp_poll_fail - 1

msg_fsp_read:        .asciz "gsp[fsp]: step 5/7 -- fsp_emem_read (drain response packet)...\n"
msg_fsp_read_len     = . - msg_fsp_read - 1
msg_fsp_read_fail:   .asciz "gsp[fsp]: fsp_emem_read FAILED\n"
msg_fsp_read_fail_len= . - msg_fsp_read_fail - 1

msg_fsp_parse:       .asciz "gsp[fsp]: step 6/7 -- fsp_response_parse (decode NVDM status)...\n"
msg_fsp_parse_len    = . - msg_fsp_parse - 1
msg_fsp_reject:      .asciz "gsp[fsp]: FSP REJECTED COT -- nonzero NVDM status\n"
msg_fsp_reject_len   = . - msg_fsp_reject - 1

msg_fsp_ack:         .asciz "gsp[fsp]: step 7/7 -- ack response (advance MSGQ_TAIL)...\n"
msg_fsp_ack_len      = . - msg_fsp_ack - 1

msg_fsp_done:        .asciz "gsp[fsp]: Chain-of-Trust accepted -- FMC authorized\n"
msg_fsp_done_len     = . - msg_fsp_done - 1

// ============================================================
// MSGQ_TAIL advance constants (shared with emem_xfer.s register map)
// Kept local here so this file assembles standalone; real driver
// should call a helper once one exists.
// ============================================================
.equ NV_PFSP_MSGQ_HEAD_BASE, 0x008F2C80
.equ NV_PFSP_MSGQ_TAIL_BASE, 0x008F2C84
.equ NV_PFSP_MSGQ_STRIDE,    8

// ============================================================
// Text
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// void fsp_print(x1 = msg, x2 = len)
//   raw SYS_WRITE to stderr.  Clobbers x0, x8.
// ------------------------------------------------------------
.align 4
fsp_print:
    mov     x0, #2                  // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// int fsp_send_boot_commands(bar0_base)
//
// Stack frame (32 + 896 + 256 = 1184 bytes, 16B aligned):
//
//   [sp +    0] saved x29, x30   (stp/ldp reach this)
//   [sp +   16] saved bar0       (callee-preserved scratch)
//   [sp +   32] cot payload   (896 B reserved, 860 B used)
//   [sp +  928] response buf  (256 B)
// ------------------------------------------------------------
.equ FRAME_SIZE,        1184
.equ OFF_X29,           0
.equ OFF_BAR0,          16
.equ OFF_COT_BUF,       32
.equ OFF_RESP_BUF,      928

.global fsp_send_boot_commands
.align 4
fsp_send_boot_commands:
    // ---- Prologue ----
    sub     sp, sp, #FRAME_SIZE
    stp     x29, x30, [sp, #OFF_X29]
    add     x29, sp, #OFF_X29
    str     x0, [sp, #OFF_BAR0]     // save bar0_base

    // Banner
    adrp    x1, msg_fsp_begin
    add     x1, x1, :lo12:msg_fsp_begin
    mov     x2, #msg_fsp_begin_len
    bl      fsp_print

    // =============================================================
    // Step 1: fsp_init(bar0) -- verify FSP ready, queues empty.
    // =============================================================
    adrp    x1, msg_fsp_init
    add     x1, x1, :lo12:msg_fsp_init
    mov     x2, #msg_fsp_init_len
    bl      fsp_print

    ldr     x0, [sp, #OFF_BAR0]
    bl      fsp_init
    cmp     w0, #0
    b.lt    .Lfail_init

    // =============================================================
    // Step 2: cot_build_payload(cot_buf, fmc_phys, manifest_off, boot_args)
    //
    //   fmc_phys     = fw_bar4_cpu[fw_manifest_offset]... actually the
    //                  caller here supplies the BAR4 PA directly;
    //                  cot_payload.s knows how to slice hash/key/sig out
    //                  of the signed FMC at the manifest offset.
    //
    //   For now we pass:
    //     x0 = &cot_buf (stack)
    //     x1 = fw_bar4_cpu   (CPU-side pointer into the FMC blob;
    //                         cot_payload.s will translate to BAR4 PA
    //                         via its own captured bar4_phys)
    //     x2 = fw_manifest_offset
    //     x3 = 0   (boot-args sysmem offset -- populated later once
    //               the GSP_FMC_BOOT_PARAMS page is bump-allocated)
    // =============================================================
    adrp    x1, msg_fsp_build
    add     x1, x1, :lo12:msg_fsp_build
    mov     x2, #msg_fsp_build_len
    bl      fsp_print

    adrp    x4, fw_bar4_cpu
    add     x4, x4, :lo12:fw_bar4_cpu
    ldr     x0, [x4]                    // x0 = fw_bar4_cpu (u64)

    adrp    x4, fw_manifest_offset
    add     x4, x4, :lo12:fw_manifest_offset
    ldr     x1, [x4]                    // x1 = fw_manifest_offset (u64)

    add     x2, sp, #OFF_COT_BUF        // x2 = dst scratch buffer
    bl      cot_build_payload
    cmp     w0, #0
    b.lt    .Lfail_build

    // =============================================================
    // Step 3: mctp_send_payload(bar0, ch=0, nvdm=COT, buf, len)
    // =============================================================
    adrp    x1, msg_fsp_send
    add     x1, x1, :lo12:msg_fsp_send
    mov     x2, #msg_fsp_send_len
    bl      fsp_print

    ldr     x0, [sp, #OFF_BAR0]
    mov     x1, #FSP_CHANNEL
    add     x2, sp, #OFF_COT_BUF
    mov     w3, #COT_PAYLOAD_SIZE
    bl      mctp_send_payload
    cmp     w0, #0
    b.lt    .Lfail_send

    // =============================================================
    // Step 4: fsp_response_poll(bar0, ch) -- await response.
    // =============================================================
    adrp    x1, msg_fsp_poll
    add     x1, x1, :lo12:msg_fsp_poll
    mov     x2, #msg_fsp_poll_len
    bl      fsp_print

    ldr     x0, [sp, #OFF_BAR0]
    mov     x1, #FSP_CHANNEL
    mov     w2, #5                      // timeout: 5 seconds
    bl      fsp_response_poll
    cmp     w0, #0
    b.lt    .Lfail_poll

    // =============================================================
    // Step 5: fsp_emem_read(bar0, ch, dst, max_dw) -- drain one packet.
    //   256-byte packet = 64 dwords.
    // =============================================================
    adrp    x1, msg_fsp_read
    add     x1, x1, :lo12:msg_fsp_read
    mov     x2, #msg_fsp_read_len
    bl      fsp_print

    ldr     x0, [sp, #OFF_BAR0]
    mov     x1, #FSP_CHANNEL
    mov     w2, #FSP_EMEM_CH0_MSGQ_OFFSET // byte offset: FSP->CPU response queue
    add     x3, sp, #OFF_RESP_BUF
    mov     w4, #RESP_BUF_SIZE          // 256 bytes
    bl      fsp_emem_read
    cmp     w0, #0
    b.lt    .Lfail_read

    // =============================================================
    // Step 6: fsp_response_parse(buf, len) -- decode NVDM status.
    //   Returns x0 = FSP status code.  0 = accepted, non-zero = reject.
    // =============================================================
    adrp    x1, msg_fsp_parse
    add     x1, x1, :lo12:msg_fsp_parse
    mov     x2, #msg_fsp_parse_len
    bl      fsp_print

    add     x0, sp, #OFF_RESP_BUF
    mov     x1, #RESP_BUF_SIZE
    bl      fsp_response_parse
    cmp     w0, #0
    b.ne    .Lfail_reject

    // =============================================================
    // Step 7: ACK the response -- advance MSGQ_TAIL to MSGQ_HEAD so
    // FSP may reuse the message buffer.  Inlined MMIO:
    //   head = ldr bar0 + MSGQ_HEAD_BASE + ch*8
    //   str  head -> bar0 + MSGQ_TAIL_BASE + ch*8
    // =============================================================
    adrp    x1, msg_fsp_ack
    add     x1, x1, :lo12:msg_fsp_ack
    mov     x2, #msg_fsp_ack_len
    bl      fsp_print

    ldr     x0, [sp, #OFF_BAR0]         // x0 = bar0 base
    // ch = 0 so no channel-stride math needed; keep generic via immediate.
    movz    w4, #(NV_PFSP_MSGQ_HEAD_BASE & 0xFFFF)
    movk    w4, #((NV_PFSP_MSGQ_HEAD_BASE >> 16) & 0xFFFF), lsl #16
    add     x4, x0, x4                  // x4 = &MSGQ_HEAD(0)
    ldr     w5, [x4]                    // w5 = current head
    movz    w6, #(NV_PFSP_MSGQ_TAIL_BASE & 0xFFFF)
    movk    w6, #((NV_PFSP_MSGQ_TAIL_BASE >> 16) & 0xFFFF), lsl #16
    add     x6, x0, x6                  // x6 = &MSGQ_TAIL(0)
    str     w5, [x6]                    // tail := head  (ACK)
    dsb     sy

    // =============================================================
    // Success
    // =============================================================
    adrp    x1, msg_fsp_done
    add     x1, x1, :lo12:msg_fsp_done
    mov     x2, #msg_fsp_done_len
    bl      fsp_print

    mov     x0, #0
    b       .Lepilogue

// ============================================================
// Failure paths -- print, set retcode, fall through to epilogue.
// ============================================================
.Lfail_init:
    adrp    x1, msg_fsp_init_fail
    add     x1, x1, :lo12:msg_fsp_init_fail
    mov     x2, #msg_fsp_init_fail_len
    bl      fsp_print
    mov     x0, #FSP_ERR_INIT
    b       .Lepilogue

.Lfail_build:
    adrp    x1, msg_fsp_build_fail
    add     x1, x1, :lo12:msg_fsp_build_fail
    mov     x2, #msg_fsp_build_fail_len
    bl      fsp_print
    mov     x0, #FSP_ERR_BUILD
    b       .Lepilogue

.Lfail_send:
    adrp    x1, msg_fsp_send_fail
    add     x1, x1, :lo12:msg_fsp_send_fail
    mov     x2, #msg_fsp_send_fail_len
    bl      fsp_print
    mov     x0, #FSP_ERR_SEND
    b       .Lepilogue

.Lfail_poll:
    adrp    x1, msg_fsp_poll_fail
    add     x1, x1, :lo12:msg_fsp_poll_fail
    mov     x2, #msg_fsp_poll_fail_len
    bl      fsp_print
    mov     x0, #FSP_ERR_POLL
    b       .Lepilogue

.Lfail_read:
    adrp    x1, msg_fsp_read_fail
    add     x1, x1, :lo12:msg_fsp_read_fail
    mov     x2, #msg_fsp_read_fail_len
    bl      fsp_print
    mov     x0, #FSP_ERR_EMEM_READ
    b       .Lepilogue

.Lfail_reject:
    adrp    x1, msg_fsp_reject
    add     x1, x1, :lo12:msg_fsp_reject
    mov     x2, #msg_fsp_reject_len
    bl      fsp_print
    mov     x0, #FSP_ERR_REJECTED
    // fall through

.Lepilogue:
    ldp     x29, x30, [sp, #OFF_X29]
    add     sp, sp, #FRAME_SIZE
    ret
