// probe_pci.s -- Read-only PCI config space / BAR / topology probe for GH200
//
// Reads PCI config header via sysfs, parses BARs from resource file,
// prints NUMA node, link speed, link width.
// Does NOT modify GPU state. Safe to run with nvidia.ko loaded.
//
// Build:  as -o probe_pci.o probe_pci.s && ld -o probe_pci probe_pci.o
// Run:    sudo ./probe_pci
//
// PCI BDF is hardcoded below -- patch if needed.

.equ SYS_OPENAT,    56
.equ SYS_READ,      63
.equ SYS_WRITE,     64
.equ SYS_CLOSE,     57
.equ SYS_EXIT,      93
.equ AT_FDCWD,      -100
.equ O_RDONLY,       0

.data
.align 3

// --- sysfs paths ---
path_config:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/config\0"
path_resource:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/resource\0"
path_numa:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/numa_node\0"
path_link_speed:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/current_link_speed\0"
path_link_width:
    .ascii "/sys/bus/pci/devices/0000:dd:00.0/current_link_width\0"

// --- messages ---
msg_header:     .ascii "=== Lithos PCI Config Probe (0000:dd:00.0) ===\n"
msg_header_len = . - msg_header

msg_sect1:      .ascii "\n--- PCI Config Header ---\n"
msg_sect1_len = . - msg_sect1

msg_vendor:     .ascii "Vendor ID         = 0x"
msg_vendor_len = . - msg_vendor

msg_device:     .ascii "Device ID         = 0x"
msg_device_len = . - msg_device

msg_command:    .ascii "Command           = 0x"
msg_command_len = . - msg_command

msg_status:     .ascii "Status            = 0x"
msg_status_len = . - msg_status

msg_revision:   .ascii "Revision          = 0x"
msg_revision_len = . - msg_revision

msg_class:      .ascii "Class Code        = 0x"
msg_class_len = . - msg_class

msg_bar0:       .ascii "BAR0 (raw)        = 0x"
msg_bar0_len  = . - msg_bar0

msg_bar1:       .ascii "BAR1 (raw)        = 0x"
msg_bar1_len  = . - msg_bar1

msg_bar23:      .ascii "BAR2+3 (raw 64b)  = 0x"
msg_bar23_len = . - msg_bar23

msg_bar45:      .ascii "BAR4+5 (raw 64b)  = 0x"
msg_bar45_len = . - msg_bar45

msg_subsys_vid: .ascii "Subsys Vendor ID  = 0x"
msg_subsys_vid_len = . - msg_subsys_vid

msg_subsys_did: .ascii "Subsys Device ID  = 0x"
msg_subsys_did_len = . - msg_subsys_did

msg_nvidia_ok:  .ascii "  -> Vendor         = NVIDIA (0x10DE)\n"
msg_nvidia_ok_len = . - msg_nvidia_ok

msg_nvidia_bad: .ascii "  -> Vendor         = UNKNOWN (expected 0x10DE)\n"
msg_nvidia_bad_len = . - msg_nvidia_bad

msg_gh200_ok:   .ascii "  -> Device         = GH200 (0x2342)\n"
msg_gh200_ok_len = . - msg_gh200_ok

msg_gh200_bad:  .ascii "  -> Device         = UNKNOWN (expected 0x2342)\n"
msg_gh200_bad_len = . - msg_gh200_bad

msg_sect2:      .ascii "\n--- BAR Layout (from sysfs resource) ---\n"
msg_sect2_len = . - msg_sect2

msg_rbar0:      .ascii "BAR0  start=0x"
msg_rbar0_len = . - msg_rbar0
msg_rbar2:      .ascii "BAR2  start=0x"
msg_rbar2_len = . - msg_rbar2
msg_rbar4:      .ascii "BAR4  start=0x"
msg_rbar4_len = . - msg_rbar4

msg_end:        .ascii "  end=0x"
msg_end_len   = . - msg_end

msg_size:       .ascii "  size=0x"
msg_size_len  = . - msg_size

msg_flags:      .ascii "  flags=0x"
msg_flags_len = . - msg_flags

msg_sect3:      .ascii "\n--- Topology ---\n"
msg_sect3_len = . - msg_sect3

msg_numa:       .ascii "NUMA node         = "
msg_numa_len  = . - msg_numa

msg_lspeed:     .ascii "Link Speed        = "
msg_lspeed_len = . - msg_lspeed

msg_lwidth:     .ascii "Link Width        = "
msg_lwidth_len = . - msg_lwidth

msg_newline:    .ascii "\n"

msg_open_fail:  .ascii "probe_pci: failed to open sysfs file\n"
msg_open_flen = . - msg_open_fail

msg_read_fail:  .ascii "probe_pci: read failed\n"
msg_read_flen = . - msg_read_fail

.bss
.align 3
config_buf:     .skip 256           // PCI config header (need 64 bytes min)
resource_buf:   .skip 1024          // sysfs resource file text
sysfs_buf:      .skip 256           // generic small sysfs read buffer
hex_out:        .skip 20            // scratch for hex output

.text
.align 4

.global _start
_start:
    // ---- Print header ----
    mov     x0, #1
    adrp    x1, msg_header
    add     x1, x1, :lo12:msg_header
    mov     x2, #msg_header_len
    mov     x8, #SYS_WRITE
    svc     #0

    // ============================================================
    // SECTION 1: PCI Config Header
    // ============================================================
    mov     x0, #1
    adrp    x1, msg_sect1
    add     x1, x1, :lo12:msg_sect1
    mov     x2, #msg_sect1_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Open config
    mov     x0, #AT_FDCWD
    adrp    x1, path_config
    add     x1, x1, :lo12:path_config
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0             // x19 = fd

    // Read 64 bytes
    mov     x0, x19
    adrp    x1, config_buf
    add     x1, x1, :lo12:config_buf
    mov     x2, #64
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #64
    b.lt    .read_fail

    // Close
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Load base of config_buf into x20
    adrp    x20, config_buf
    add     x20, x20, :lo12:config_buf

    // -- Vendor ID (offset 0, u16) --
    ldrh    w21, [x20, #0]
    adrp    x1, msg_vendor
    add     x1, x1, :lo12:msg_vendor
    mov     w2, #msg_vendor_len
    mov     w0, w21
    bl      .print_hex16_nl

    // Check NVIDIA
    mov     w0, #0x10DE
    cmp     w21, w0
    b.ne    .not_nvidia
    mov     x0, #1
    adrp    x1, msg_nvidia_ok
    add     x1, x1, :lo12:msg_nvidia_ok
    mov     x2, #msg_nvidia_ok_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .vendor_done
.not_nvidia:
    mov     x0, #1
    adrp    x1, msg_nvidia_bad
    add     x1, x1, :lo12:msg_nvidia_bad
    mov     x2, #msg_nvidia_bad_len
    mov     x8, #SYS_WRITE
    svc     #0
.vendor_done:

    // -- Device ID (offset 2, u16) --
    ldrh    w21, [x20, #2]
    adrp    x1, msg_device
    add     x1, x1, :lo12:msg_device
    mov     w2, #msg_device_len
    mov     w0, w21
    bl      .print_hex16_nl

    // Check GH200
    movz    w0, #0x2342
    cmp     w21, w0
    b.ne    .not_gh200
    mov     x0, #1
    adrp    x1, msg_gh200_ok
    add     x1, x1, :lo12:msg_gh200_ok
    mov     x2, #msg_gh200_ok_len
    mov     x8, #SYS_WRITE
    svc     #0
    b       .device_done
.not_gh200:
    mov     x0, #1
    adrp    x1, msg_gh200_bad
    add     x1, x1, :lo12:msg_gh200_bad
    mov     x2, #msg_gh200_bad_len
    mov     x8, #SYS_WRITE
    svc     #0
.device_done:

    // -- Command register (offset 4, u16) --
    ldrh    w21, [x20, #4]
    adrp    x1, msg_command
    add     x1, x1, :lo12:msg_command
    mov     w2, #msg_command_len
    mov     w0, w21
    bl      .print_hex16_nl

    // -- Status register (offset 6, u16) --
    ldrh    w21, [x20, #6]
    adrp    x1, msg_status
    add     x1, x1, :lo12:msg_status
    mov     w2, #msg_status_len
    mov     w0, w21
    bl      .print_hex16_nl

    // -- Revision (offset 8, u8) --
    ldrb    w21, [x20, #8]
    adrp    x1, msg_revision
    add     x1, x1, :lo12:msg_revision
    mov     w2, #msg_revision_len
    mov     w0, w21
    bl      .print_hex8_nl

    // -- Class code (offset 9, 3 bytes: prog_if, subclass, class) --
    // Stored little-endian: offset 9 = prog_if, 10 = subclass, 11 = base class
    // We want to display as base_class:subclass:prog_if = XX:XX:XX
    ldrb    w23, [x20, #11]     // base class
    ldrb    w22, [x20, #10]     // subclass
    ldrb    w21, [x20, #9]      // prog_if
    // Combine into 24-bit: (base<<16)|(sub<<8)|prog
    orr     w0, w21, w22, lsl #8
    orr     w0, w0, w23, lsl #16
    adrp    x1, msg_class
    add     x1, x1, :lo12:msg_class
    mov     w2, #msg_class_len
    bl      .print_hex24_nl

    // -- BAR0 (offset 0x10, u32) --
    ldr     w21, [x20, #0x10]
    adrp    x1, msg_bar0
    add     x1, x1, :lo12:msg_bar0
    mov     w2, #msg_bar0_len
    mov     w0, w21
    bl      .print_reg

    // -- BAR1 (offset 0x14, u32) --
    ldr     w21, [x20, #0x14]
    adrp    x1, msg_bar1
    add     x1, x1, :lo12:msg_bar1
    mov     w2, #msg_bar1_len
    mov     w0, w21
    bl      .print_reg

    // -- BAR2+3 (offset 0x18-0x1C, u64 little-endian) --
    ldr     w21, [x20, #0x18]   // BAR2 low
    ldr     w22, [x20, #0x1C]   // BAR3 high
    adrp    x1, msg_bar23
    add     x1, x1, :lo12:msg_bar23
    mov     w2, #msg_bar23_len
    // Combine: x0 = (BAR3 << 32) | BAR2
    mov     x0, x21
    orr     x0, x0, x22, lsl #32
    bl      .print_hex64_with_prefix_nl

    // -- BAR4+5 (offset 0x20-0x24, u64 little-endian) --
    ldr     w21, [x20, #0x20]   // BAR4 low
    ldr     w22, [x20, #0x24]   // BAR5 high
    adrp    x1, msg_bar45
    add     x1, x1, :lo12:msg_bar45
    mov     w2, #msg_bar45_len
    mov     x0, x21
    orr     x0, x0, x22, lsl #32
    bl      .print_hex64_with_prefix_nl

    // -- Subsystem Vendor ID (offset 0x2C, u16) --
    ldrh    w21, [x20, #0x2C]
    adrp    x1, msg_subsys_vid
    add     x1, x1, :lo12:msg_subsys_vid
    mov     w2, #msg_subsys_vid_len
    mov     w0, w21
    bl      .print_hex16_nl

    // -- Subsystem Device ID (offset 0x2E, u16) --
    ldrh    w21, [x20, #0x2E]
    adrp    x1, msg_subsys_did
    add     x1, x1, :lo12:msg_subsys_did
    mov     w2, #msg_subsys_did_len
    mov     w0, w21
    bl      .print_hex16_nl

    // ============================================================
    // SECTION 2: BAR Layout from sysfs resource
    // ============================================================
    mov     x0, #1
    adrp    x1, msg_sect2
    add     x1, x1, :lo12:msg_sect2
    mov     x2, #msg_sect2_len
    mov     x8, #SYS_WRITE
    svc     #0

    // Open resource
    mov     x0, #AT_FDCWD
    adrp    x1, path_resource
    add     x1, x1, :lo12:path_resource
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .open_fail
    mov     x19, x0

    // Read resource file
    mov     x0, x19
    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    mov     x2, #1023
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #0
    b.le    .read_fail
    mov     x24, x0             // x24 = bytes read

    // NUL-terminate
    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    strb    wzr, [x1, x24]

    // Close
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Parse resource file: each line is "start end flags\n"
    // We want lines 0 (BAR0), 2 (BAR2), 4 (BAR4)
    adrp    x25, resource_buf
    add     x25, x25, :lo12:resource_buf

    // --- BAR0 (line 0) ---
    mov     x0, x25
    bl      .parse_resource_line    // returns x0=start, x1=end, x2=flags, x3=ptr to next line
    mov     x26, x3             // save next-line pointer
    // Print BAR0
    stp     x0, x1, [sp, #-32]!
    str     x2, [sp, #16]
    // "BAR0  start=0x"
    mov     x0, #1
    adrp    x1, msg_rbar0
    add     x1, x1, :lo12:msg_rbar0
    mov     x2, #msg_rbar0_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #0]       // start
    bl      .print_hex64
    // "  end=0x"
    mov     x0, #1
    adrp    x1, msg_end
    add     x1, x1, :lo12:msg_end
    mov     x2, #msg_end_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #8]       // end
    bl      .print_hex64
    // "  size=0x"
    mov     x0, #1
    adrp    x1, msg_size
    add     x1, x1, :lo12:msg_size
    mov     x2, #msg_size_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x0, x1, [sp, #0]   // start, end
    sub     x0, x1, x0
    add     x0, x0, #1         // size = end - start + 1
    bl      .print_hex64
    // "  flags=0x"
    mov     x0, #1
    adrp    x1, msg_flags
    add     x1, x1, :lo12:msg_flags
    mov     x2, #msg_flags_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #16]      // flags
    bl      .print_hex64
    add     sp, sp, #32
    bl      .print_nl

    // Skip line 1 (BAR1) to get to line 2 (BAR2)
    mov     x0, x26
    bl      .skip_line          // skip BAR1
    mov     x26, x0

    // --- BAR2 (line 2) ---
    mov     x0, x26
    bl      .parse_resource_line
    mov     x26, x3
    stp     x0, x1, [sp, #-16]!
    // "BAR2  start=0x"
    mov     x0, #1
    adrp    x1, msg_rbar2
    add     x1, x1, :lo12:msg_rbar2
    mov     x2, #msg_rbar2_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #0]
    bl      .print_hex64
    mov     x0, #1
    adrp    x1, msg_end
    add     x1, x1, :lo12:msg_end
    mov     x2, #msg_end_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #8]
    bl      .print_hex64
    mov     x0, #1
    adrp    x1, msg_size
    add     x1, x1, :lo12:msg_size
    mov     x2, #msg_size_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x0, x1, [sp, #0]
    sub     x0, x1, x0
    add     x0, x0, #1
    bl      .print_hex64
    add     sp, sp, #16
    bl      .print_nl

    // Skip line 3 (BAR3) to get to line 4 (BAR4)
    mov     x0, x26
    bl      .skip_line
    mov     x26, x0

    // --- BAR4 (line 4) ---
    mov     x0, x26
    bl      .parse_resource_line
    stp     x0, x1, [sp, #-16]!
    // "BAR4  start=0x"
    mov     x0, #1
    adrp    x1, msg_rbar4
    add     x1, x1, :lo12:msg_rbar4
    mov     x2, #msg_rbar4_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #0]
    bl      .print_hex64
    mov     x0, #1
    adrp    x1, msg_end
    add     x1, x1, :lo12:msg_end
    mov     x2, #msg_end_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldr     x0, [sp, #8]
    bl      .print_hex64
    mov     x0, #1
    adrp    x1, msg_size
    add     x1, x1, :lo12:msg_size
    mov     x2, #msg_size_len
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x0, x1, [sp, #0]
    sub     x0, x1, x0
    add     x0, x0, #1
    bl      .print_hex64
    add     sp, sp, #16
    bl      .print_nl

    // ============================================================
    // SECTION 3: Topology (sysfs text files)
    // ============================================================
    mov     x0, #1
    adrp    x1, msg_sect3
    add     x1, x1, :lo12:msg_sect3
    mov     x2, #msg_sect3_len
    mov     x8, #SYS_WRITE
    svc     #0

    // NUMA node
    adrp    x0, msg_numa
    add     x0, x0, :lo12:msg_numa
    mov     w1, #msg_numa_len
    adrp    x2, path_numa
    add     x2, x2, :lo12:path_numa
    bl      .print_sysfs_value

    // Link speed
    adrp    x0, msg_lspeed
    add     x0, x0, :lo12:msg_lspeed
    mov     w1, #msg_lspeed_len
    adrp    x2, path_link_speed
    add     x2, x2, :lo12:path_link_speed
    bl      .print_sysfs_value

    // Link width
    adrp    x0, msg_lwidth
    add     x0, x0, :lo12:msg_lwidth
    mov     w1, #msg_lwidth_len
    adrp    x2, path_link_width
    add     x2, x2, :lo12:path_link_width
    bl      .print_sysfs_value

    // ---- Exit 0 ----
    mov     x0, #0
    mov     x8, #SYS_EXIT
    svc     #0

// ---- Error paths ----
.open_fail:
    mov     x0, #2
    adrp    x1, msg_open_fail
    add     x1, x1, :lo12:msg_open_fail
    mov     x2, #msg_open_flen
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

.read_fail:
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
    mov     x0, #2
    adrp    x1, msg_read_fail
    add     x1, x1, :lo12:msg_read_fail
    mov     x2, #msg_read_flen
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, #2
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// .print_sysfs_value -- print "LABEL" then contents of sysfs file
//   x0 = label addr, w1 = label len, x2 = path addr
// ============================================================
.print_sysfs_value:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    str     x2, [sp, #16]      // save path
    // Print label
    mov     x2, x1
    mov     x1, x0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    // Open file
    mov     x0, #AT_FDCWD
    ldr     x1, [sp, #16]
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .sysfs_na
    mov     w19, w0
    // Read
    mov     x0, x19
    adrp    x1, sysfs_buf
    add     x1, x1, :lo12:sysfs_buf
    mov     x2, #255
    mov     x8, #SYS_READ
    svc     #0
    mov     x23, x0             // bytes read
    // Close
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
    // Write contents (includes trailing newline from sysfs)
    cmp     x23, #0
    b.le    .sysfs_na
    mov     x0, #1
    adrp    x1, sysfs_buf
    add     x1, x1, :lo12:sysfs_buf
    mov     x2, x23
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #48
    ret
.sysfs_na:
    // Print "N/A\n"
    mov     x0, #1
    adr     x1, .sysfs_na_str
    mov     x2, #4
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #48
    ret
.sysfs_na_str:
    .ascii "N/A\n"

// ============================================================
// .print_reg -- print prefix + 8-digit hex + newline
//   w0 = value, x1 = prefix addr, w2 = prefix len
// ============================================================
.print_reg:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     w22, w0
    // Print prefix
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    // Print hex
    mov     w0, w22
    bl      .print_hex32
    bl      .print_nl
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_hex16_nl -- print prefix + 4-digit hex + newline
//   w0 = value (low 16 bits), x1 = prefix addr, w2 = prefix len
// ============================================================
.print_hex16_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     w22, w0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w22
    bl      .print_hex16
    bl      .print_nl
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_hex8_nl -- print prefix + 2-digit hex + newline
//   w0 = value (low 8 bits), x1 = prefix addr, w2 = prefix len
// ============================================================
.print_hex8_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     w22, w0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w22
    bl      .print_hex8
    bl      .print_nl
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_hex24_nl -- print prefix + 6-digit hex + newline
//   w0 = value (low 24 bits), x1 = prefix addr, w2 = prefix len
// ============================================================
.print_hex24_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     w22, w0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     w0, w22
    bl      .print_hex24
    bl      .print_nl
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_hex64_with_prefix_nl -- print prefix + 16-digit hex + newline
//   x0 = 64-bit value, x1 = prefix addr, w2 = prefix len
// ============================================================
.print_hex64_with_prefix_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x22, x0
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    mov     x0, x22
    bl      .print_hex64
    bl      .print_nl
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_nl -- print newline
// ============================================================
.print_nl:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x0, #1
    adrp    x1, msg_newline
    add     x1, x1, :lo12:msg_newline
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// .print_hex64 -- print 16 hex digits to stdout
//   x0 = 64-bit value
// ============================================================
.print_hex64:
    sub     sp, sp, #32
    mov     x3, #60             // shift start
    add     x4, sp, #0
.hex64_loop:
    lsr     x5, x0, x3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex64_digit
    add     w5, w5, #('a' - 10)
    b       .hex64_store
.hex64_digit:
    add     w5, w5, #'0'
.hex64_store:
    strb    w5, [x4], #1
    subs    x3, x3, #4
    b.ge    .hex64_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #16
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ret

// ============================================================
// .print_hex32 -- print 8 hex digits to stdout
//   w0 = 32-bit value
// ============================================================
.print_hex32:
    sub     sp, sp, #16
    mov     w3, #28
    add     x4, sp, #0
.hex32_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex32_digit
    add     w5, w5, #('a' - 10)
    b       .hex32_store
.hex32_digit:
    add     w5, w5, #'0'
.hex32_store:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .hex32_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #8
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret

// ============================================================
// .print_hex16 -- print 4 hex digits
//   w0 = value (low 16 bits)
// ============================================================
.print_hex16:
    sub     sp, sp, #16
    mov     w3, #12
    add     x4, sp, #0
.hex16_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex16_digit
    add     w5, w5, #('a' - 10)
    b       .hex16_store
.hex16_digit:
    add     w5, w5, #'0'
.hex16_store:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .hex16_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #4
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret

// ============================================================
// .print_hex8 -- print 2 hex digits
//   w0 = value (low 8 bits)
// ============================================================
.print_hex8:
    sub     sp, sp, #16
    // High nibble
    lsr     w5, w0, #4
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex8_hi_digit
    add     w5, w5, #('a' - 10)
    b       .hex8_hi_store
.hex8_hi_digit:
    add     w5, w5, #'0'
.hex8_hi_store:
    strb    w5, [sp, #0]
    // Low nibble
    and     w5, w0, #0xF
    cmp     w5, #10
    b.lt    .hex8_lo_digit
    add     w5, w5, #('a' - 10)
    b       .hex8_lo_store
.hex8_lo_digit:
    add     w5, w5, #'0'
.hex8_lo_store:
    strb    w5, [sp, #1]
    mov     x0, #1
    mov     x1, sp
    mov     x2, #2
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret

// ============================================================
// .print_hex24 -- print 6 hex digits
//   w0 = value (low 24 bits)
// ============================================================
.print_hex24:
    sub     sp, sp, #16
    mov     w3, #20
    add     x4, sp, #0
.hex24_loop:
    lsr     w5, w0, w3
    and     w5, w5, #0xF
    cmp     w5, #10
    b.lt    .hex24_digit
    add     w5, w5, #('a' - 10)
    b       .hex24_store
.hex24_digit:
    add     w5, w5, #'0'
.hex24_store:
    strb    w5, [x4], #1
    subs    w3, w3, #4
    b.ge    .hex24_loop
    mov     x0, #1
    mov     x1, sp
    mov     x2, #6
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16
    ret

// ============================================================
// .parse_resource_line -- parse "0xSTART 0xEND 0xFLAGS\n"
//   x0 = pointer to line start
//   Returns: x0 = start, x1 = end, x2 = flags, x3 = pointer past newline
// ============================================================
.parse_resource_line:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    mov     x10, x0             // x10 = cursor

    // Parse start (skip "0x" prefix)
    bl      .parse_hex_field    // x10 in, returns x0 = value, x10 = past delimiter
    str     x0, [sp, #16]      // save start

    // Parse end
    bl      .parse_hex_field
    str     x0, [sp, #24]      // save end

    // Parse flags (until newline or NUL)
    bl      .parse_hex_field
    str     x0, [sp, #32]      // save flags

    // x10 now points past the newline (or at the delimiter)
    ldr     x0, [sp, #16]
    ldr     x1, [sp, #24]
    ldr     x2, [sp, #32]
    mov     x3, x10
    ldp     x29, x30, [sp], #48
    ret

// ============================================================
// .parse_hex_field -- parse one hex number from string at x10
//   Skips leading spaces and "0x" prefix
//   Returns: x0 = parsed value, x10 = pointer past trailing space/newline
// ============================================================
.parse_hex_field:
    // Skip spaces
.phf_skip_space:
    ldrb    w11, [x10]
    cmp     w11, #' '
    b.ne    .phf_check_0x
    add     x10, x10, #1
    b       .phf_skip_space

.phf_check_0x:
    // Check for "0x" prefix
    ldrb    w11, [x10]
    cmp     w11, #'0'
    b.ne    .phf_parse
    ldrb    w11, [x10, #1]
    cmp     w11, #'x'
    b.ne    .phf_parse
    add     x10, x10, #2       // skip "0x"

.phf_parse:
    mov     x0, #0
.phf_loop:
    ldrb    w11, [x10]
    // Check if hex digit
    cmp     w11, #'0'
    b.lt    .phf_done
    cmp     w11, #'9'
    b.le    .phf_digit09
    cmp     w11, #'a'
    b.lt    .phf_check_upper
    cmp     w11, #'f'
    b.le    .phf_digit_af
    b       .phf_check_upper
.phf_digit09:
    sub     w11, w11, #'0'
    b       .phf_accum
.phf_digit_af:
    sub     w11, w11, #('a' - 10)
    b       .phf_accum
.phf_check_upper:
    cmp     w11, #'A'
    b.lt    .phf_done
    cmp     w11, #'F'
    b.gt    .phf_done
    sub     w11, w11, #('A' - 10)
.phf_accum:
    lsl     x0, x0, #4
    orr     x0, x0, x11
    add     x10, x10, #1
    b       .phf_loop
.phf_done:
    // Skip trailing space or newline
    ldrb    w11, [x10]
    cmp     w11, #' '
    b.eq    .phf_skip_trail
    cmp     w11, #'\n'
    b.eq    .phf_skip_trail
    cmp     w11, #'\t'
    b.eq    .phf_skip_trail
    ret
.phf_skip_trail:
    add     x10, x10, #1
    ret

// ============================================================
// .skip_line -- advance past next newline
//   x0 = pointer, returns x0 = pointer past newline
// ============================================================
.skip_line:
.sl_loop:
    ldrb    w1, [x0]
    cbz     w1, .sl_done
    add     x0, x0, #1
    cmp     w1, #'\n'
    b.ne    .sl_loop
.sl_done:
    ret
