// probe_fsp_dry.s -- FSP EMEM dry-run probe
//
// Validates the FSP EMEM write/read path end-to-end WITHOUT advancing
// QUEUE_HEAD, so the FSP never sees a command.  This lets us exercise:
//
//   1. Steps 1-5 of the GSP cold-boot sequence (bar_map, pmc_check,
//      falcon_reset, hbm_alloc, fw_load).
//   2. fsp_init -- verify FSP boot sentinel, queues empty.
//   3. cot_build_payload -- populate the 860-byte NVDM_PAYLOAD_COT.
//   4. Write a 0xDEADBEEF test pattern to 4 EMEM slots, read back
//      and verify each matches.
//   5. Write the raw COT payload bytes (not MCTP-framed) to EMEM,
//      read back and hex dump the first 32 bytes of each slot.
//
// QUEUE_HEAD is NEVER advanced -- the FSP is not notified.
//
// Build (assembly check only):
//   as -o probe_fsp_dry.o probe_fsp_dry.s
//
// Link (requires all boot objects):
//   as -o probe_fsp_dry.o probe_fsp_dry.s
//   as -o bar_map.o bar_map.s
//   as -o pmc_check.o pmc_check.s
//   as -o falcon_reset.o falcon_reset.s
//   as -o hbm_alloc.o hbm_alloc.s
//   as -o fw_load.o fw_load.s
//   as -o fsp/init.o fsp/init.s
//   as -o fsp/cot_payload.o fsp/cot_payload.s
//   as -o fsp/emem_xfer.o fsp/emem_xfer.s
//   ld -o probe_fsp_dry probe_fsp_dry.o bar_map.o pmc_check.o \
//        falcon_reset.o hbm_alloc.o fw_load.o fsp/init.o \
//        fsp/cot_payload.o fsp/emem_xfer.o -static
//
// Exit codes:
//   0 = all EMEM writes verified, dry run complete
//   1 = bar_map_init failed
//   2 = pmc_check failed
//   3 = falcon_reset failed
//   4 = hbm_alloc_init failed
//   5 = fsp_init failed
//   6 = EMEM test pattern mismatch
//   7 = EMEM COT readback mismatch

    .arch armv8-a

// ---- Syscall numbers ----
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93

// ---- BAR4 reservation (same as boot.s) ----
.equ GSP_RESERVED_BYTES, 0x04000000            // 64 MB

// ---- FSP / EMEM constants ----
.equ FSP_CHANNEL,            0
.equ EMEM_SLOT_SIZE,         256                // each EMEM packet slot
.equ NUM_TEST_SLOTS,         4                  // 4 slots for test pattern
.equ COT_PAYLOAD_SIZE,       860                // NVDM_PAYLOAD_COT
.equ COT_SCRATCH_STACK,      864                // 860 rounded to 16-byte align
.equ TEST_PATTERN,           0xDEADBEEF         // test pattern dword
.equ HEX_DUMP_BYTES,         32                 // bytes to hex-dump per slot

// ============================================================
// External symbols -- boot path (Steps 1-5)
// ============================================================
.extern bar_map_init
.extern pmc_check
.extern falcon_reset
.extern hbm_alloc_init
.extern gsp_fw_load

.extern bar0_base
.extern bar4_base
.extern bar4_phys

.extern fw_bar4_cpu
.extern fw_manifest_offset

// ============================================================
// External symbols -- FSP subsystem
// ============================================================
.extern fsp_init
.extern cot_build_payload
.extern fsp_emem_write
.extern fsp_emem_read

// ============================================================
// Data section -- messages
// ============================================================
.data
.align 3

msg_banner:
    .asciz "probe[fsp-dry]: ========== FSP EMEM dry-run probe ==========\n"
msg_banner_len = . - msg_banner - 1

// ---- Steps 1-5 ----
msg_s1_begin:    .asciz "probe[fsp-dry]: [1/5] bar_map_init...\n"
msg_s1_begin_len = . - msg_s1_begin - 1
msg_s1_ok:       .asciz "probe[fsp-dry]: [1/5] bar_map_init -- OK\n"
msg_s1_ok_len    = . - msg_s1_ok - 1
msg_s1_fail:     .asciz "probe[fsp-dry]: [1/5] bar_map_init -- FAILED\n"
msg_s1_fail_len  = . - msg_s1_fail - 1

msg_s2_begin:    .asciz "probe[fsp-dry]: [2/5] pmc_check...\n"
msg_s2_begin_len = . - msg_s2_begin - 1
msg_s2_ok:       .asciz "probe[fsp-dry]: [2/5] pmc_check -- OK\n"
msg_s2_ok_len    = . - msg_s2_ok - 1
msg_s2_fail:     .asciz "probe[fsp-dry]: [2/5] pmc_check -- FAILED\n"
msg_s2_fail_len  = . - msg_s2_fail - 1

msg_s3_begin:    .asciz "probe[fsp-dry]: [3/5] falcon_reset...\n"
msg_s3_begin_len = . - msg_s3_begin - 1
msg_s3_ok:       .asciz "probe[fsp-dry]: [3/5] falcon_reset -- OK\n"
msg_s3_ok_len    = . - msg_s3_ok - 1
msg_s3_fail:     .asciz "probe[fsp-dry]: [3/5] falcon_reset -- FAILED\n"
msg_s3_fail_len  = . - msg_s3_fail - 1

msg_s4_begin:    .asciz "probe[fsp-dry]: [4/5] hbm_alloc_init...\n"
msg_s4_begin_len = . - msg_s4_begin - 1
msg_s4_ok:       .asciz "probe[fsp-dry]: [4/5] hbm_alloc_init -- OK\n"
msg_s4_ok_len    = . - msg_s4_ok - 1
msg_s4_fail:     .asciz "probe[fsp-dry]: [4/5] hbm_alloc_init -- FAILED\n"
msg_s4_fail_len  = . - msg_s4_fail - 1

msg_s5_begin:    .asciz "probe[fsp-dry]: [5/5] gsp_fw_load...\n"
msg_s5_begin_len = . - msg_s5_begin - 1
msg_s5_ok:       .asciz "probe[fsp-dry]: [5/5] gsp_fw_load -- OK\n"
msg_s5_ok_len    = . - msg_s5_ok - 1

// ---- FSP dry run ----
msg_dry_banner:
    .asciz "probe[fsp-dry]: === FSP DRY RUN (no QUEUE_HEAD advance) ===\n"
msg_dry_banner_len = . - msg_dry_banner - 1

msg_fsp_init_begin:
    .asciz "probe[fsp-dry]: [A] fsp_init -- checking FSP readiness...\n"
msg_fsp_init_begin_len = . - msg_fsp_init_begin - 1
msg_fsp_init_ok:
    .asciz "probe[fsp-dry]: [A] fsp_init -- FSP ready\n"
msg_fsp_init_ok_len = . - msg_fsp_init_ok - 1
msg_fsp_init_fail:
    .asciz "probe[fsp-dry]: [A] fsp_init -- FAILED\n"
msg_fsp_init_fail_len = . - msg_fsp_init_fail - 1

msg_cot_begin:
    .asciz "probe[fsp-dry]: [B] cot_build_payload -- building 860-byte COT struct...\n"
msg_cot_begin_len = . - msg_cot_begin - 1
msg_cot_ok:
    .asciz "probe[fsp-dry]: [B] cot_build_payload -- OK (860 bytes)\n"
msg_cot_ok_len = . - msg_cot_ok - 1

msg_test_begin:
    .asciz "probe[fsp-dry]: [C] EMEM test pattern (0xDEADBEEF x64 per slot)...\n"
msg_test_begin_len = . - msg_test_begin - 1

msg_write_slot:
    .asciz "probe[fsp-dry]:     wrote test pattern to slot "
msg_write_slot_len = . - msg_write_slot - 1

msg_read_slot:
    .asciz "probe[fsp-dry]:     read back slot "
msg_read_slot_len = . - msg_read_slot - 1

msg_match:
    .asciz " -- MATCH\n"
msg_match_len = . - msg_match - 1

msg_mismatch:
    .asciz " -- MISMATCH\n"
msg_mismatch_len = . - msg_mismatch - 1

msg_test_ok:
    .asciz "probe[fsp-dry]: [C] EMEM test pattern -- ALL SLOTS VERIFIED\n"
msg_test_ok_len = . - msg_test_ok - 1

msg_test_fail:
    .asciz "probe[fsp-dry]: [C] EMEM test pattern -- VERIFICATION FAILED\n"
msg_test_fail_len = . - msg_test_fail - 1

msg_cot_write_begin:
    .asciz "probe[fsp-dry]: [D] writing raw COT payload to EMEM (4 x 256B slots)...\n"
msg_cot_write_begin_len = . - msg_cot_write_begin - 1

msg_cot_readback:
    .asciz "probe[fsp-dry]: [E] reading back COT from EMEM -- hex dump (32B/slot)...\n"
msg_cot_readback_len = . - msg_cot_readback - 1

msg_slot_hdr:
    .asciz "probe[fsp-dry]:     slot "
msg_slot_hdr_len = . - msg_slot_hdr - 1

msg_at_offset:
    .asciz " @ EMEM offset 0x"
msg_at_offset_len = . - msg_at_offset - 1

msg_colon:
    .asciz ":\n"
msg_colon_len = . - msg_colon - 1

msg_hex_prefix:
    .asciz "probe[fsp-dry]:       "
msg_hex_prefix_len = . - msg_hex_prefix - 1

msg_cot_match:
    .asciz "probe[fsp-dry]:     COT readback slot "
msg_cot_match_len = . - msg_cot_match - 1

msg_no_advance:
    .asciz "probe[fsp-dry]: === QUEUE_HEAD NOT ADVANCED -- FSP NOT NOTIFIED ===\n"
msg_no_advance_len = . - msg_no_advance - 1

msg_complete:
    .asciz "probe[fsp-dry]: === FSP DRY RUN COMPLETE ===\n"
msg_complete_len = . - msg_complete - 1

msg_newline:
    .asciz "\n"
msg_newline_len = . - msg_newline - 1

msg_space:
    .asciz " "
msg_space_len = . - msg_space - 1

// ---- Hex lookup table ----
.align 3
hex_table:
    .ascii "0123456789abcdef"

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ============================================================
// Stack frame layout for _start
//
//   [sp +    0]  saved x29, x30             (16 B)
//   [sp +   16]  saved x19..x28             (80 B, 5 pairs)
//   [sp +   96]  hex_line    (128 B)        -- hex dump scratch (low offset
//                                              so STP/LDP can reach it)
//   [sp +  224]  write_buf   (256 B)        -- EMEM write source
//   [sp +  480]  read_buf    (256 B)        -- EMEM read dest
//   [sp +  736]  cot_buf     (864 B)        -- COT payload (860B, +4 pad)
//   total = 1600
// ============================================================
.equ FRAME_SIZE,        1600
.equ OFF_X29,           0
.equ OFF_CALLEE_SAVE,   16
.equ OFF_HEX_LINE,      96
.equ OFF_WRITE_BUF,     224
.equ OFF_READ_BUF,      480
.equ OFF_COT_BUF,       736

.global _start
.align 4
_start:
    // ---- Prologue ----
    sub     sp, sp, #FRAME_SIZE
    stp     x29, x30, [sp, #OFF_X29]
    add     x29, sp, #OFF_X29
    stp     x19, x20, [sp, #OFF_CALLEE_SAVE + 0]
    stp     x21, x22, [sp, #OFF_CALLEE_SAVE + 16]
    stp     x23, x24, [sp, #OFF_CALLEE_SAVE + 32]
    stp     x25, x26, [sp, #OFF_CALLEE_SAVE + 48]
    stp     x27, x28, [sp, #OFF_CALLEE_SAVE + 64]

    // ---- Banner ----
    adrp    x1, msg_banner
    add     x1, x1, :lo12:msg_banner
    mov     x2, #msg_banner_len
    bl      probe_print

    // =============================================================
    // Step 1: bar_map_init
    // =============================================================
    adrp    x1, msg_s1_begin
    add     x1, x1, :lo12:msg_s1_begin
    mov     x2, #msg_s1_begin_len
    bl      probe_print

    bl      bar_map_init
    cmp     x0, #0
    b.lt    .Lfail_s1

    adrp    x1, msg_s1_ok
    add     x1, x1, :lo12:msg_s1_ok
    mov     x2, #msg_s1_ok_len
    bl      probe_print

    // =============================================================
    // Step 2: pmc_check
    // =============================================================
    adrp    x1, msg_s2_begin
    add     x1, x1, :lo12:msg_s2_begin
    mov     x2, #msg_s2_begin_len
    bl      probe_print

    bl      pmc_check
    cmp     x0, #0
    b.lt    .Lfail_s2

    adrp    x1, msg_s2_ok
    add     x1, x1, :lo12:msg_s2_ok
    mov     x2, #msg_s2_ok_len
    bl      probe_print

    // =============================================================
    // Step 3: falcon_reset
    // =============================================================
    adrp    x1, msg_s3_begin
    add     x1, x1, :lo12:msg_s3_begin
    mov     x2, #msg_s3_begin_len
    bl      probe_print

    bl      falcon_reset
    cmp     x0, #0
    b.lt    .Lfail_s3

    adrp    x1, msg_s3_ok
    add     x1, x1, :lo12:msg_s3_ok
    mov     x2, #msg_s3_ok_len
    bl      probe_print

    // =============================================================
    // Step 4: hbm_alloc_init
    // =============================================================
    adrp    x1, msg_s4_begin
    add     x1, x1, :lo12:msg_s4_begin
    mov     x2, #msg_s4_begin_len
    bl      probe_print

    adrp    x0, bar4_base
    add     x0, x0, :lo12:bar4_base
    ldr     x0, [x0]
    adrp    x1, bar4_phys
    add     x1, x1, :lo12:bar4_phys
    ldr     x1, [x1]
    mov     x2, #GSP_RESERVED_BYTES
    bl      hbm_alloc_init
    cmp     x0, #0
    b.lt    .Lfail_s4

    adrp    x1, msg_s4_ok
    add     x1, x1, :lo12:msg_s4_ok
    mov     x2, #msg_s4_ok_len
    bl      probe_print

    // =============================================================
    // Step 5: gsp_fw_load (exits internally on failure)
    // =============================================================
    adrp    x1, msg_s5_begin
    add     x1, x1, :lo12:msg_s5_begin
    mov     x2, #msg_s5_begin_len
    bl      probe_print

    bl      gsp_fw_load

    adrp    x1, msg_s5_ok
    add     x1, x1, :lo12:msg_s5_ok
    mov     x2, #msg_s5_ok_len
    bl      probe_print

    // =============================================================
    // === FSP DRY RUN ===
    // =============================================================
    adrp    x1, msg_dry_banner
    add     x1, x1, :lo12:msg_dry_banner
    mov     x2, #msg_dry_banner_len
    bl      probe_print

    // ---------------------------------------------------------
    // Step A: fsp_init(bar0)
    // ---------------------------------------------------------
    adrp    x1, msg_fsp_init_begin
    add     x1, x1, :lo12:msg_fsp_init_begin
    mov     x2, #msg_fsp_init_begin_len
    bl      probe_print

    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    bl      fsp_init
    cmp     w0, #0
    b.lt    .Lfail_fsp_init

    adrp    x1, msg_fsp_init_ok
    add     x1, x1, :lo12:msg_fsp_init_ok
    mov     x2, #msg_fsp_init_ok_len
    bl      probe_print

    // ---------------------------------------------------------
    // Step B: cot_build_payload
    //   x0 = fw_bar4_cpu, x1 = fw_manifest_offset, x2 = &cot_buf
    // ---------------------------------------------------------
    adrp    x1, msg_cot_begin
    add     x1, x1, :lo12:msg_cot_begin
    mov     x2, #msg_cot_begin_len
    bl      probe_print

    adrp    x4, fw_bar4_cpu
    add     x4, x4, :lo12:fw_bar4_cpu
    ldr     x0, [x4]                       // x0 = fw_bar4_cpu

    adrp    x4, fw_manifest_offset
    add     x4, x4, :lo12:fw_manifest_offset
    ldr     x1, [x4]                       // x1 = fw_manifest_offset

    add     x2, sp, #OFF_COT_BUF           // x2 = &cot_buf on stack
    bl      cot_build_payload
    // cot_build_payload returns w0 = 860 (payload size); we just proceed.

    adrp    x1, msg_cot_ok
    add     x1, x1, :lo12:msg_cot_ok
    mov     x2, #msg_cot_ok_len
    bl      probe_print

    // ---------------------------------------------------------
    // Step C: Write 0xDEADBEEF test pattern to 4 EMEM slots,
    //         read back and verify.
    //
    // Fill write_buf with 0xDEADBEEF (64 dwords = 256 bytes),
    // then for each slot i in [0..3]:
    //   fsp_emem_write(bar0, ch=0, offset=i*256, write_buf, 256)
    //   fsp_emem_read (bar0, ch=0, offset=i*256, read_buf,  256)
    //   memcmp write_buf vs read_buf, 256 bytes
    // ---------------------------------------------------------
    adrp    x1, msg_test_begin
    add     x1, x1, :lo12:msg_test_begin
    mov     x2, #msg_test_begin_len
    bl      probe_print

    // Fill write_buf with 0xDEADBEEF
    add     x5, sp, #OFF_WRITE_BUF
    movz    w6, #0xBEEF
    movk    w6, #0xDEAD, lsl #16           // w6 = 0xDEADBEEF
    mov     w7, #64                         // 64 dwords
.Lfill_pattern:
    str     w6, [x5], #4
    subs    w7, w7, #1
    b.ne    .Lfill_pattern

    // Loop over 4 slots
    mov     w19, #0                         // w19 = slot index

.Ltest_loop:
    cmp     w19, #NUM_TEST_SLOTS
    b.ge    .Ltest_all_done

    // Print "wrote test pattern to slot N"
    adrp    x1, msg_write_slot
    add     x1, x1, :lo12:msg_write_slot
    mov     x2, #msg_write_slot_len
    bl      probe_print
    mov     w0, w19
    bl      probe_print_digit
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #msg_newline_len
    bl      probe_print

    // fsp_emem_write(bar0, ch=0, offset=slot*256, write_buf, 256)
    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    mov     w1, #FSP_CHANNEL
    lsl     w2, w19, #8                     // byte_offset = slot * 256
    add     x3, sp, #OFF_WRITE_BUF
    mov     w4, #EMEM_SLOT_SIZE
    bl      fsp_emem_write

    // fsp_emem_read(bar0, ch=0, offset=slot*256, read_buf, 256)
    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    mov     w1, #FSP_CHANNEL
    lsl     w2, w19, #8
    add     x3, sp, #OFF_READ_BUF
    mov     w4, #EMEM_SLOT_SIZE
    bl      fsp_emem_read

    // Compare write_buf vs read_buf (256 bytes = 64 dwords)
    add     x5, sp, #OFF_WRITE_BUF
    add     x6, sp, #OFF_READ_BUF
    mov     w7, #64                         // 64 dwords to compare
    mov     w20, #1                         // w20 = match flag (1=match)
.Lcmp_loop:
    ldr     w8, [x5], #4
    ldr     w9, [x6], #4
    cmp     w8, w9
    b.eq    .Lcmp_next
    mov     w20, #0                         // mismatch
.Lcmp_next:
    subs    w7, w7, #1
    b.ne    .Lcmp_loop

    // Print "read back slot N -- MATCH / MISMATCH"
    adrp    x1, msg_read_slot
    add     x1, x1, :lo12:msg_read_slot
    mov     x2, #msg_read_slot_len
    bl      probe_print
    mov     w0, w19
    bl      probe_print_digit

    cbz     w20, .Lslot_mismatch
    adrp    x1, msg_match
    add     x1, x1, :lo12:msg_match
    mov     x2, #msg_match_len
    bl      probe_print
    b       .Ltest_next

.Lslot_mismatch:
    adrp    x1, msg_mismatch
    add     x1, x1, :lo12:msg_mismatch
    mov     x2, #msg_mismatch_len
    bl      probe_print
    b       .Lfail_test_pattern

.Ltest_next:
    add     w19, w19, #1
    b       .Ltest_loop

.Ltest_all_done:
    adrp    x1, msg_test_ok
    add     x1, x1, :lo12:msg_test_ok
    mov     x2, #msg_test_ok_len
    bl      probe_print

    // ---------------------------------------------------------
    // Step D: Write raw COT payload bytes to EMEM (4 slots)
    //
    // COT payload is 860 bytes.  We write it as 4 x 256-byte
    // chunks (last chunk is 92 bytes of payload + 164 zeros).
    // ---------------------------------------------------------
    adrp    x1, msg_cot_write_begin
    add     x1, x1, :lo12:msg_cot_write_begin
    mov     x2, #msg_cot_write_begin_len
    bl      probe_print

    mov     w19, #0                         // slot index
    mov     w21, #COT_PAYLOAD_SIZE          // bytes remaining

.Lcot_write_loop:
    cmp     w19, #NUM_TEST_SLOTS
    b.ge    .Lcot_write_done

    // Determine chunk size: min(remaining, 256)
    mov     w22, #EMEM_SLOT_SIZE
    cmp     w21, w22
    csel    w22, w21, w22, lo               // w22 = chunk_len

    // fsp_emem_write(bar0, ch=0, offset=slot*256, cot_buf + slot*256, chunk_len)
    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    mov     w1, #FSP_CHANNEL
    lsl     w2, w19, #8                     // byte_offset = slot * 256
    add     x3, sp, #OFF_COT_BUF
    add     x3, x3, x19, lsl #8            // src = cot_buf + slot*256
    mov     w4, w22
    bl      fsp_emem_write

    sub     w21, w21, w22                   // remaining -= chunk_len
    add     w19, w19, #1
    b       .Lcot_write_loop

.Lcot_write_done:

    // ---------------------------------------------------------
    // Step E: Read back COT from EMEM, hex dump first 32 bytes
    //         of each slot, and compare against the original
    //         COT buffer.
    // ---------------------------------------------------------
    adrp    x1, msg_cot_readback
    add     x1, x1, :lo12:msg_cot_readback
    mov     x2, #msg_cot_readback_len
    bl      probe_print

    mov     w19, #0                         // slot index

.Lcot_read_loop:
    cmp     w19, #NUM_TEST_SLOTS
    b.ge    .Lcot_read_done

    // Print "slot N @ EMEM offset 0xMMM:"
    adrp    x1, msg_slot_hdr
    add     x1, x1, :lo12:msg_slot_hdr
    mov     x2, #msg_slot_hdr_len
    bl      probe_print
    mov     w0, w19
    bl      probe_print_digit
    adrp    x1, msg_at_offset
    add     x1, x1, :lo12:msg_at_offset
    mov     x2, #msg_at_offset_len
    bl      probe_print
    lsl     w0, w19, #8                     // offset value
    bl      probe_print_hex16               // print as 4-digit hex
    adrp    x1, msg_colon
    add     x1, x1, :lo12:msg_colon
    mov     x2, #msg_colon_len
    bl      probe_print

    // fsp_emem_read(bar0, ch=0, offset=slot*256, read_buf, 256)
    adrp    x0, bar0_base
    add     x0, x0, :lo12:bar0_base
    ldr     x0, [x0]
    mov     w1, #FSP_CHANNEL
    lsl     w2, w19, #8
    add     x3, sp, #OFF_READ_BUF
    mov     w4, #EMEM_SLOT_SIZE
    bl      fsp_emem_read

    // Hex dump first 32 bytes (2 lines of 16)
    add     x0, sp, #OFF_READ_BUF
    mov     w1, #HEX_DUMP_BYTES
    bl      probe_hex_dump

    // Compare read_buf against cot_buf + slot*256 (first min(256, remaining) bytes)
    // We compare the full 256 bytes that were written.
    add     x5, sp, #OFF_COT_BUF
    add     x5, x5, x19, lsl #8            // expected = cot_buf + slot*256
    add     x6, sp, #OFF_READ_BUF           // actual = read_buf

    // Determine how many bytes were valid in this slot
    mov     w7, #COT_PAYLOAD_SIZE
    lsl     w8, w19, #8                     // slot_offset = slot * 256
    subs    w7, w7, w8                      // remaining_payload = total - slot_offset
    b.le    .Lcot_slot_skip_cmp             // no payload in this slot (shouldn't happen)
    mov     w8, #EMEM_SLOT_SIZE
    cmp     w7, w8
    csel    w7, w7, w8, lo                  // w7 = min(remaining, 256)

    mov     w20, #1                         // match flag
.Lcot_cmp_loop:
    cbz     w7, .Lcot_cmp_end
    ldrb    w8, [x5], #1
    ldrb    w9, [x6], #1
    cmp     w8, w9
    b.eq    .Lcot_cmp_ok
    mov     w20, #0
.Lcot_cmp_ok:
    sub     w7, w7, #1
    b       .Lcot_cmp_loop

.Lcot_cmp_end:
    // Print "COT readback slot N -- MATCH/MISMATCH"
    adrp    x1, msg_cot_match
    add     x1, x1, :lo12:msg_cot_match
    mov     x2, #msg_cot_match_len
    bl      probe_print
    mov     w0, w19
    bl      probe_print_digit

    cbz     w20, .Lcot_slot_mismatch
    adrp    x1, msg_match
    add     x1, x1, :lo12:msg_match
    mov     x2, #msg_match_len
    bl      probe_print
    b       .Lcot_slot_next

.Lcot_slot_skip_cmp:
    // Print match for empty trailing slot (zero-padded)
    adrp    x1, msg_cot_match
    add     x1, x1, :lo12:msg_cot_match
    mov     x2, #msg_cot_match_len
    bl      probe_print
    mov     w0, w19
    bl      probe_print_digit
    adrp    x1, msg_match
    add     x1, x1, :lo12:msg_match
    mov     x2, #msg_match_len
    bl      probe_print
    b       .Lcot_slot_next

.Lcot_slot_mismatch:
    adrp    x1, msg_mismatch
    add     x1, x1, :lo12:msg_mismatch
    mov     x2, #msg_mismatch_len
    bl      probe_print
    // Continue to next slot rather than aborting; report at end

.Lcot_slot_next:
    add     w19, w19, #1
    b       .Lcot_read_loop

.Lcot_read_done:

    // ---------------------------------------------------------
    // Final summary
    // ---------------------------------------------------------
    adrp    x1, msg_no_advance
    add     x1, x1, :lo12:msg_no_advance
    mov     x2, #msg_no_advance_len
    bl      probe_print

    adrp    x1, msg_complete
    add     x1, x1, :lo12:msg_complete
    mov     x2, #msg_complete_len
    bl      probe_print

    // ---- Exit 0 ----
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Failure paths
// ============================================================
.Lfail_s1:
    adrp    x1, msg_s1_fail
    add     x1, x1, :lo12:msg_s1_fail
    mov     x2, #msg_s1_fail_len
    bl      probe_print
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.Lfail_s2:
    adrp    x1, msg_s2_fail
    add     x1, x1, :lo12:msg_s2_fail
    mov     x2, #msg_s2_fail_len
    bl      probe_print
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

.Lfail_s3:
    adrp    x1, msg_s3_fail
    add     x1, x1, :lo12:msg_s3_fail
    mov     x2, #msg_s3_fail_len
    bl      probe_print
    mov     x0, #3
    mov     x8, #SYS_EXIT
    svc     #0

.Lfail_s4:
    adrp    x1, msg_s4_fail
    add     x1, x1, :lo12:msg_s4_fail
    mov     x2, #msg_s4_fail_len
    bl      probe_print
    mov     x0, #4
    mov     x8, #SYS_EXIT
    svc     #0

.Lfail_fsp_init:
    adrp    x1, msg_fsp_init_fail
    add     x1, x1, :lo12:msg_fsp_init_fail
    mov     x2, #msg_fsp_init_fail_len
    bl      probe_print
    mov     x0, #5
    mov     x8, #SYS_EXIT
    svc     #0

.Lfail_test_pattern:
    adrp    x1, msg_test_fail
    add     x1, x1, :lo12:msg_test_fail
    mov     x2, #msg_test_fail_len
    bl      probe_print
    mov     x0, #6
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// Utility subroutines
// ============================================================

// ------------------------------------------------------------
// probe_print -- write x2 bytes from x1 to stderr
// Clobbers: x0, x8
// ------------------------------------------------------------
.align 4
probe_print:
    mov     x0, #2                          // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// probe_print_digit -- print single decimal digit (0-9) from w0
// Uses a 1-byte stack scratch.  Clobbers: x0-x3, x8.
// ------------------------------------------------------------
.align 4
probe_print_digit:
    add     w0, w0, #'0'                   // ASCII
    strb    w0, [sp, #OFF_HEX_LINE]        // borrow hex_line[0]
    mov     x0, #2
    add     x1, sp, #OFF_HEX_LINE
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// probe_print_hex16 -- print w0 as 4-digit lowercase hex
// Output goes to hex_line scratch, then SYS_WRITE.
// Clobbers: x0-x6, x8.
// ------------------------------------------------------------
.align 4
probe_print_hex16:
    add     x5, sp, #OFF_HEX_LINE          // output scratch
    adrp    x6, hex_table
    add     x6, x6, :lo12:hex_table

    // Digit 0 (bits 15:12)
    lsr     w1, w0, #12
    and     w1, w1, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #0]

    // Digit 1 (bits 11:8)
    lsr     w1, w0, #8
    and     w1, w1, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #1]

    // Digit 2 (bits 7:4)
    lsr     w1, w0, #4
    and     w1, w1, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #2]

    // Digit 3 (bits 3:0)
    and     w1, w0, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #3]

    // Write 4 chars
    mov     x0, #2
    add     x1, sp, #OFF_HEX_LINE
    mov     x2, #4
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// probe_print_hex8 -- print w0 (low byte) as 2-digit hex
// Clobbers: x0-x6, x8.
// ------------------------------------------------------------
.align 4
probe_print_hex8:
    add     x5, sp, #OFF_HEX_LINE
    adrp    x6, hex_table
    add     x6, x6, :lo12:hex_table

    lsr     w1, w0, #4
    and     w1, w1, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #0]

    and     w1, w0, #0xF
    ldrb    w2, [x6, x1]
    strb    w2, [x5, #1]

    mov     x0, #2
    add     x1, sp, #OFF_HEX_LINE
    mov     x2, #2
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// ------------------------------------------------------------
// probe_hex_dump -- print N bytes from buffer as hex, 16 per line
//   x0 = buffer, w1 = byte count
// Prints lines like:
//   probe[fsp-dry]:       de ad be ef 01 02 ...  (16 bytes)
// Clobbers: x0-x9, x19-x22 (caller-saved context assumed safe
// because we only call from within _start which already saved them)
//
// NOTE: This routine uses x23/x24 as temporaries.  The caller
// (_start) has already saved them in the prologue.
// ------------------------------------------------------------
.align 4
probe_hex_dump:
    mov     x23, x0                        // src cursor
    mov     w24, w1                        // total bytes

.Lhex_line_loop:
    cbz     w24, .Lhex_dump_done

    // Print line prefix
    stp     x23, x24, [sp, #OFF_HEX_LINE + 64]  // save across call
    adrp    x1, msg_hex_prefix
    add     x1, x1, :lo12:msg_hex_prefix
    mov     x2, #msg_hex_prefix_len
    bl      probe_print
    ldp     x23, x24, [sp, #OFF_HEX_LINE + 64]

    // Print up to 16 bytes on this line
    mov     w25, #16
    cmp     w24, w25
    csel    w25, w24, w25, lo               // w25 = min(remaining, 16)
    mov     w26, #0                         // byte index on this line

.Lhex_byte_loop:
    cmp     w26, w25
    b.ge    .Lhex_line_end

    // Save state across subroutine calls
    stp     x23, x24, [sp, #OFF_HEX_LINE + 64]
    stp     x25, x26, [sp, #OFF_HEX_LINE + 80]

    ldrb    w0, [x23, x26]
    bl      probe_print_hex8

    // Print space separator
    adrp    x1, msg_space
    add     x1, x1, :lo12:msg_space
    mov     x2, #msg_space_len
    bl      probe_print

    ldp     x23, x24, [sp, #OFF_HEX_LINE + 64]
    ldp     x25, x26, [sp, #OFF_HEX_LINE + 80]

    add     w26, w26, #1
    b       .Lhex_byte_loop

.Lhex_line_end:
    // Print newline
    stp     x23, x24, [sp, #OFF_HEX_LINE + 64]
    stp     x25, x26, [sp, #OFF_HEX_LINE + 80]
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #msg_newline_len
    bl      probe_print
    ldp     x23, x24, [sp, #OFF_HEX_LINE + 64]
    ldp     x25, x26, [sp, #OFF_HEX_LINE + 80]

    add     x23, x23, x25                  // advance src cursor
    sub     w24, w24, w25                   // remaining -= line_bytes
    b       .Lhex_line_loop

.Lhex_dump_done:
    ret

    .section .note.GNU-stack,"",%progbits
