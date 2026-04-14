// pmc_check.s -- Step 2: PMC identity check
//
// Reads PMC_BOOT_0 register at BAR0+0x000000 and verifies this is a
// Hopper-architecture GPU (bits[23:20] == 0xA).
//
// Exports:
//   pmc_check   -- verify GPU identity, return 0 on Hopper, -1 otherwise
//
// Requires: bar0_base must be populated (call bar_map_init first).
//
// Calling convention: ARM64 AAPCS.
// Returns: x0 = 0 if Hopper, -1 if not (or BAR0 not mapped).
//          x1 = raw PMC_BOOT_0 value (for diagnostics).
//
// Register map (BAR0 offsets from gsp-native.md):
//   0x000000 = PMC_BOOT_0 -- GPU identity register
//              bits[23:20] = architecture:
//                0x6 = Pascal, 0x7 = Volta, 0x8 = Turing,
//                0x9 = Ampere, 0xA = Hopper, 0xB = Blackwell

// Shared constants (syscalls, PMC offsets, etc.)
.include "gsp_common.s"

// ============================================================
// Data section
// ============================================================
.data
.align 3

msg_hopper_ok:    .asciz "gsp: PMC_BOOT_0 verified -- Hopper architecture\n"
msg_hopper_ok_len = . - msg_hopper_ok - 1
msg_not_hopper:   .asciz "gsp: ERROR: not Hopper (bits[23:20] != 0xA)\n"
msg_not_hopper_len = . - msg_not_hopper - 1
pmc_msg_bar0_null:    .asciz "gsp: ERROR: BAR0 not mapped (bar0_base == 0)\n"
pmc_msg_bar0_null_len = . - pmc_msg_bar0_null - 1
msg_boot0_is:     .asciz "gsp: PMC_BOOT_0 = 0x"
msg_boot0_is_len = . - msg_boot0_is - 1

// Hex digit table for printing
hex_chars:        .ascii "0123456789abcdef"

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// pmc_check -- read PMC_BOOT_0 and verify Hopper
//
// Returns: x0 = 0 if Hopper, -1 if not
//          x1 = raw PMC_BOOT_0 value
// Clobbers: x0-x8, x16-x17
// ------------------------------------------------------------
.global pmc_check
pmc_check:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    // Load BAR0 base address
    adrp    x19, bar0_base
    add     x19, x19, :lo12:bar0_base
    ldr     x19, [x19]

    // Check BAR0 is mapped
    cbz     x19, .pmc_bar0_null

    // ---- Read PMC_BOOT_0 (BAR0 + 0x000000) ----
    // On MMIO mappings, ldr performs an uncacheable device read.
    ldr     w20, [x19, #PMC_BOOT_0]  // w20 = PMC_BOOT_0 value

    // Print "gsp: PMC_BOOT_0 = 0x"
    adrp    x1, msg_boot0_is
    add     x1, x1, :lo12:msg_boot0_is
    mov     x2, #msg_boot0_is_len
    bl      pmc_print_msg

    // Print the 32-bit hex value
    mov     w0, w20
    bl      pmc_print_hex32

    // Print newline
    sub     sp, sp, #16
    mov     w0, #'\n'
    strb    w0, [sp]
    mov     x0, #2                    // stderr
    mov     x1, sp
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16

    // ---- Extract architecture field: bits[23:20] ----
    // ubfx w1, w20, #20, #4  -- extract 4 bits starting at bit 20
    ubfx    w1, w20, #20, #4

    // Compare against Hopper (0xA)
    cmp     w1, #ARCH_HOPPER
    b.ne    .pmc_not_hopper

    // ---- Hopper confirmed ----
    adrp    x1, msg_hopper_ok
    add     x1, x1, :lo12:msg_hopper_ok
    mov     x2, #msg_hopper_ok_len
    bl      pmc_print_msg

    mov     x0, #0                    // success
    uxtw    x1, w20            // return raw PMC_BOOT_0 in x1
    b       .pmc_return

.pmc_not_hopper:
    adrp    x1, msg_not_hopper
    add     x1, x1, :lo12:msg_not_hopper
    mov     x2, #msg_not_hopper_len
    bl      pmc_print_msg
    mov     x0, #-1
    uxtw    x1, w20
    b       .pmc_return

.pmc_bar0_null:
    adrp    x1, pmc_msg_bar0_null
    add     x1, x1, :lo12:pmc_msg_bar0_null
    mov     x2, #pmc_msg_bar0_null_len
    bl      pmc_print_msg
    mov     x0, #-1
    mov     x1, #0

.pmc_return:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ------------------------------------------------------------
// pmc_print_hex32 -- print w0 as 8-digit hex to stderr
// Clobbers: x0-x8
// ------------------------------------------------------------
.align 4
pmc_print_hex32:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16               // 8-byte hex buffer on stack
    mov     w3, w0                    // save value
    adrp    x4, hex_chars
    add     x4, x4, :lo12:hex_chars
    mov     x5, sp                    // buffer pointer

    // Emit 8 hex digits, MSB first
    mov     w6, #28                   // shift count (28, 24, 20, ... 0)
.hex32_loop:
    lsr     w7, w3, w6
    and     w7, w7, #0xF
    ldrb    w7, [x4, w7, uxtw]
    strb    w7, [x5], #1
    subs    w6, w6, #4
    b.ge    .hex32_loop

    // Write 8 chars to stderr
    mov     x0, #2
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0

    add     sp, sp, #16
    ldp     x29, x30, [sp], #16
    ret

// ------------------------------------------------------------
// pmc_print_msg -- write string to stderr
// x1 = string address, x2 = length
// ------------------------------------------------------------
.align 4
pmc_print_msg:
    mov     x0, #2
    mov     x8, #SYS_WRITE
    svc     #0
    ret
