// probe_fsp.s -- Read-only FSP state probe for GH200
//
// Maps BAR0 via sysfs, reads FSP-related registers:
//   - THERM_I2CS_SCRATCH (FSP boot-complete sentinel)
//   - SCRATCH_GROUP_2 (8 dwords of FSP diagnostic state)
//   - SCRATCH_GROUP_3 (8 dwords)
//   - FSP EMEM queue heads/tails (4 channels)
//   - FSP MSGQ heads/tails (4 channels)
//
// Does NOT modify GPU state. Safe with nvidia.ko loaded.
// Requires root.
//
// Build:  as -o probe_fsp.o probe_fsp.s && ld -o probe_fsp probe_fsp.o
// Run:    sudo ./probe_fsp

.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ SYS_NANOSLEEP, 101
.equ AT_FDCWD,      -100
.equ PROT_READ,     1
.equ MAP_SHARED,    1
.equ BAR0_SIZE,     0x1000000

// FSP register offsets
.equ THERM_I2CS_SCRATCH,     0x0200BC
.equ SCRATCH_GROUP_2_BASE,   0x8F0320
.equ SCRATCH_GROUP_3_BASE,   0x8F0330
.equ QUEUE_HEAD_BASE,        0x8F2C00     // +i*8
.equ QUEUE_TAIL_BASE,        0x8F2C04     // +i*8
.equ MSGQ_HEAD_BASE,         0x8F2C80     // +i*8
.equ MSGQ_TAIL_BASE,         0x8F2C84     // +i*8

.data
.align 3

bar0_path:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource0\0"

msg_header:     .ascii "=== Lithos FSP State Probe (read-only) ===\n\n"
msg_header_len = . - msg_header

msg_fsp_boot:   .ascii "THERM_I2CS_SCRATCH = 0x"
msg_fsp_blen  = . - msg_fsp_boot

msg_fsp_ok:     .ascii "  -> FSP boot COMPLETE (low byte = 0xFF)\n"
msg_fsp_ok_len= . - msg_fsp_ok

msg_fsp_no:     .ascii "  -> FSP boot NOT complete (low byte != 0xFF)\n"
msg_fsp_no_len= . - msg_fsp_no

msg_sg2:        .ascii "\nSCRATCH_GROUP_2:\n"
msg_sg2_len   = . - msg_sg2

msg_sg3:        .ascii "\nSCRATCH_GROUP_3:\n"
msg_sg3_len   = . - msg_sg3

msg_queues:     .ascii "\nFSP EMEM Queues:\n"
msg_q_len     = . - msg_queues

msg_msgq:       .ascii "\nFSP Message Queues:\n"
msg_mq_len    = . - msg_msgq

msg_bracket_l:  .ascii "  ["
msg_bl_len    = . - msg_bracket_l

msg_bracket_r:  .ascii "] = 0x"
msg_br_len    = . - msg_bracket_r

msg_head:       .ascii "  CH"
msg_head_len  = . - msg_head

msg_h_eq:       .ascii " H=0x"
msg_h_eq_len  = . - msg_h_eq

msg_t_eq:       .ascii " T=0x"
msg_t_eq_len  = . - msg_t_eq

msg_newline:    .ascii "\n"

msg_scratch2:   .ascii "SCRATCH_GROUP_2[0] re-read = 0x"
msg_s2_len    = . - msg_scratch2

msg_fsp_alive:  .ascii "  -> FSP ALIVE (scratch changed between reads)\n"
msg_alive_len = . - msg_fsp_alive

msg_fsp_stall:  .ascii "  WARNING: FSP may be STALLED (scratch static, all queues idle)\n"
msg_stall_len = . - msg_fsp_stall

msg_fsp_qact:   .ascii "  -> Queue activity detected (head != tail on some channel)\n"
msg_qact_len  = . - msg_fsp_qact

msg_open_fail:  .ascii "probe_fsp: failed to open resource0\n"
msg_opf_len   = . - msg_open_fail

msg_mmap_fail:  .ascii "probe_fsp: mmap failed\n"
msg_mmf_len   = . - msg_mmap_fail

// timespec for nanosleep: 0 seconds, 100000000 nanoseconds (100ms)
.align 3
sleep_req:
    .quad   0                   // tv_sec  = 0
    .quad   100000000           // tv_nsec = 100ms
sleep_rem:
    .quad   0                   // tv_sec  (remainder)
    .quad   0                   // tv_nsec (remainder)

.text
.align 4

.global _start
_start:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]

    // Print header
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Open BAR0
    mov     x0, #AT_FDCWD
    adrp    x1, bar0_path
    add     x1, x1, :lo12:bar0_path
    mov     x2, #0                  // O_RDONLY (read-only probe)
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0

    // mmap BAR0 read-only
    mov     x0, #0
    mov     x1, #BAR0_SIZE
    mov     x2, #PROT_READ
    mov     x3, #MAP_SHARED
    mov     x4, x19
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .mmap_fail
    mov     x20, x0             // x20 = bar0

    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // ---- THERM_I2CS_SCRATCH (first read) ----
    movz    x1, #0x00BC
    movk    x1, #0x0002, lsl #16
    ldr     w21, [x20, x1]

    mov     x0, #1
    adrp    x1, msg_fsp_boot
    add     x1, x1, :lo12:msg_fsp_boot
    mov     x2, #msg_fsp_blen
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w21
    bl      .print_hex32_nl

    and     w0, w21, #0xFF
    cmp     w0, #0xFF
    b.ne    .fsp_not_ok

    // Low byte is 0xFF -- but could be stale.  Double-read with delay.

    // Also read SCRATCH_GROUP_2[0] before the sleep
    movz    x1, #0x0320
    movk    x1, #0x008F, lsl #16   // 0x8F0320
    ldr     w25, [x20, x1]         // w25 = scratch_group_2[0] first read

    // Sleep ~100ms via nanosleep syscall
    adrp    x0, sleep_req
    add     x0, x0, :lo12:sleep_req
    adrp    x1, sleep_rem
    add     x1, x1, :lo12:sleep_rem
    mov     x8, #SYS_NANOSLEEP
    svc     #0

    // Re-read SCRATCH_GROUP_2[0] after sleep
    movz    x1, #0x0320
    movk    x1, #0x008F, lsl #16
    ldr     w26, [x20, x1]         // w26 = scratch_group_2[0] second read

    // Print second scratch value
    mov     x0, #1
    adrp    x1, msg_scratch2
    add     x1, x1, :lo12:msg_scratch2
    mov     x2, #msg_s2_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w26
    bl      .print_hex32_nl

    // Check if scratch changed between reads
    cmp     w25, w26
    b.ne    .fsp_confirmed_alive

    // Scratch is static.  Check all queue head/tail pairs for activity.
    // w27 = activity flag (0 = all idle, 1 = some channel has head != tail)
    mov     w27, #0

    // Check EMEM queues (4 channels)
    mov     w23, #0
.dbl_q_loop:
    cmp     w23, #4
    b.ge    .dbl_q_done
    movz    x22, #0x2C00
    movk    x22, #0x008F, lsl #16
    lsl     w24, w23, #3
    add     x22, x22, x24, uxtw
    ldr     w0, [x20, x22]         // HEAD
    add     x22, x22, #4
    ldr     w1, [x20, x22]         // TAIL
    cmp     w0, w1
    b.eq    .dbl_q_next
    mov     w27, #1                 // activity detected
.dbl_q_next:
    add     w23, w23, #1
    b       .dbl_q_loop
.dbl_q_done:

    // Check MSGQ queues (4 channels)
    mov     w23, #0
.dbl_mq_loop:
    cmp     w23, #4
    b.ge    .dbl_mq_done
    movz    x22, #0x2C80
    movk    x22, #0x008F, lsl #16
    lsl     w24, w23, #3
    add     x22, x22, x24, uxtw
    ldr     w0, [x20, x22]         // HEAD
    add     x22, x22, #4
    ldr     w1, [x20, x22]         // TAIL
    cmp     w0, w1
    b.eq    .dbl_mq_next
    mov     w27, #1                 // activity detected
.dbl_mq_next:
    add     w23, w23, #1
    b       .dbl_mq_loop
.dbl_mq_done:

    // Evaluate: scratch static + all queues idle => likely stalled
    cbz     w27, .fsp_likely_stalled

    // Scratch static but queues have pending work -- some life
    mov     x0, #1
    adrp    x1, msg_fsp_qact
    add     x1, x1, :lo12:msg_fsp_qact
    mov     x2, #msg_qact_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    adrp    x1, msg_fsp_ok
    add     x1, x1, :lo12:msg_fsp_ok
    mov     x2, #msg_fsp_ok_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .fsp_chk_done

.fsp_confirmed_alive:
    mov     x0, #1
    adrp    x1, msg_fsp_alive
    add     x1, x1, :lo12:msg_fsp_alive
    mov     x2, #msg_alive_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    adrp    x1, msg_fsp_ok
    add     x1, x1, :lo12:msg_fsp_ok
    mov     x2, #msg_fsp_ok_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .fsp_chk_done

.fsp_likely_stalled:
    // scratch 0xFF but static, all queues idle => dead-after-boot
    mov     x0, #1
    adrp    x1, msg_fsp_stall
    add     x1, x1, :lo12:msg_fsp_stall
    mov     x2, #msg_stall_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w28, #1
    b       .fsp_chk_done_noset

.fsp_not_ok:
    mov     x0, #1
    adrp    x1, msg_fsp_no
    add     x1, x1, :lo12:msg_fsp_no
    mov     x2, #msg_fsp_no_len
    mov     x8, #SYS_WRITE
    svc     #0
    // FSP not booted -- exit with error after dumping diagnostics
    // Set flag in x28 so we exit non-zero at the end
    mov     w28, #1
    b       .fsp_chk_done_noset
.fsp_chk_done:
    mov     w28, #0               // FSP OK, will exit 0
.fsp_chk_done_noset:

    // ---- SCRATCH_GROUP_2 (8 dwords at stride 4) ----
    mov     x0, #1
    adrp    x1, msg_sg2
    add     x1, x1, :lo12:msg_sg2
    mov     x2, #msg_sg2_len
    mov     x8, #SYS_WRITE
    svc     #0

    movz    x22, #0x0320
    movk    x22, #0x008F, lsl #16   // 0x8F0320
    mov     w23, #0                 // index
.sg2_loop:
    cmp     w23, #8
    b.ge    .sg2_done
    ldr     w21, [x20, x22]
    mov     w0, w23
    mov     w1, w21
    bl      .print_indexed_reg
    add     x22, x22, #4
    add     w23, w23, #1
    b       .sg2_loop
.sg2_done:

    // ---- SCRATCH_GROUP_3 (8 dwords at stride 4) ----
    mov     x0, #1
    adrp    x1, msg_sg3
    add     x1, x1, :lo12:msg_sg3
    mov     x2, #msg_sg3_len
    mov     x8, #SYS_WRITE
    svc     #0

    movz    x22, #0x0330
    movk    x22, #0x008F, lsl #16   // 0x8F0330
    mov     w23, #0
.sg3_loop:
    cmp     w23, #8
    b.ge    .sg3_done
    ldr     w21, [x20, x22]
    mov     w0, w23
    mov     w1, w21
    bl      .print_indexed_reg
    add     x22, x22, #4
    add     w23, w23, #1
    b       .sg3_loop
.sg3_done:

    // ---- EMEM Queues (4 channels, HEAD/TAIL at stride 8) ----
    mov     x0, #1
    adrp    x1, msg_queues
    add     x1, x1, :lo12:msg_queues
    mov     x2, #msg_q_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     w23, #0
.q_loop:
    cmp     w23, #4
    b.ge    .q_done
    // HEAD
    movz    x22, #0x2C00
    movk    x22, #0x008F, lsl #16
    lsl     w24, w23, #3            // channel * 8
    add     x22, x22, x24, uxtw
    ldr     w21, [x20, x22]        // HEAD
    add     x22, x22, #4
    ldr     w0, [x20, x22]         // TAIL
    mov     w1, w23                 // channel
    mov     w2, w21                 // head
    mov     w3, w0                  // tail
    bl      .print_ht_line
    add     w23, w23, #1
    b       .q_loop
.q_done:

    // ---- Message Queues (4 channels) ----
    mov     x0, #1
    adrp    x1, msg_msgq
    add     x1, x1, :lo12:msg_msgq
    mov     x2, #msg_mq_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     w23, #0
.mq_loop:
    cmp     w23, #4
    b.ge    .mq_done
    movz    x22, #0x2C80
    movk    x22, #0x008F, lsl #16
    lsl     w24, w23, #3
    add     x22, x22, x24, uxtw
    ldr     w21, [x20, x22]
    add     x22, x22, #4
    ldr     w0, [x20, x22]
    mov     w1, w23
    mov     w2, w21
    mov     w3, w0
    bl      .print_ht_line
    add     w23, w23, #1
    b       .mq_loop
.mq_done:

    // ---- Exit with status based on FSP boot check ----
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    mov     x0, x28              // 0 if FSP OK, 1 if not booted
    mov     x8, #SYS_EXIT
    svc     #0

// ---- Error paths ----
.open_fail:
    mov     x0, #2
    adrp    x1, msg_open_fail
    add     x1, x1, :lo12:msg_open_fail
    mov     x2, #msg_opf_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.mmap_fail:
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
    mov     x0, #2
    adrp    x1, msg_mmap_fail
    add     x1, x1, :lo12:msg_mmap_fail
    mov     x2, #msg_mmf_len
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// .print_indexed_reg -- print "  [N] = 0xXXXXXXXX\n"
//   w0 = index, w1 = value
// ============================================================
.print_indexed_reg:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    mov     w19, w0             // index
    mov     w20, w1             // value

    // "  ["
    mov     x0, #1
    adrp    x1, msg_bracket_l
    add     x1, x1, :lo12:msg_bracket_l
    mov     x2, #msg_bl_len
    mov     x8, #SYS_WRITE
    svc     #0

    // index digit
    add     w0, w19, #'0'
    sub     sp, sp, #16
    strb    w0, [sp]
    mov     x0, #1
    mov     x1, sp
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16

    // "] = 0x"
    mov     x0, #1
    adrp    x1, msg_bracket_r
    add     x1, x1, :lo12:msg_bracket_r
    mov     x2, #msg_br_len
    mov     x8, #SYS_WRITE
    svc     #0

    // hex value + newline
    mov     w0, w20
    bl      .print_hex32_nl

    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ============================================================
// .print_ht_line -- print "  CHn H=0x... T=0x...\n"
//   w1 = channel, w2 = head, w3 = tail
// ============================================================
.print_ht_line:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     w19, w1             // channel
    mov     w20, w2             // head
    mov     w21, w3             // tail

    // "  CH"
    mov     x0, #1
    adrp    x1, msg_head
    add     x1, x1, :lo12:msg_head
    mov     x2, #msg_head_len
    mov     x8, #SYS_WRITE
    svc     #0

    // channel digit
    add     w0, w19, #'0'
    sub     sp, sp, #16
    strb    w0, [sp]
    mov     x0, #1
    mov     x1, sp
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16

    // " H=0x"
    mov     x0, #1
    adrp    x1, msg_h_eq
    add     x1, x1, :lo12:msg_h_eq
    mov     x2, #msg_h_eq_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     w0, w20
    bl      .print_hex32

    // " T=0x"
    mov     x0, #1
    adrp    x1, msg_t_eq
    add     x1, x1, :lo12:msg_t_eq
    mov     x2, #msg_t_eq_len
    mov     x8, #SYS_WRITE
    svc     #0

    mov     w0, w21
    bl      .print_hex32_nl

    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ============================================================
// .print_hex32 -- print 8 hex digits to stdout (no newline)
// .print_hex32_nl -- same but with trailing newline
// ============================================================
.print_hex32_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      .print_hex32
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #16
    ret

.print_hex32:
    sub     sp, sp, #16
    mov     w3, #28
    add     x4, sp, #0
.hex_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex_digit
    add     w5, w5, #('a' - 10)
    b       .hex_store
.hex_digit:
    add     w5, w5, #'0'
.hex_store:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .hex_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret
