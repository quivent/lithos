// bar_map.s -- Step 1: mmap BAR0 (GPU registers) and BAR4 (coherent HBM)
//
// Opens PCI sysfs resource files and mmaps them into the process address
// space.  No libc, raw Linux syscalls only.
//
// Exports:
//   bar_map_init   -- maps BAR0 + BAR4, stores base pointers in globals
//   bar0_base      -- .quad, virtual address of BAR0 mmap (16MB GPU MMIO)
//   bar4_base      -- .quad, virtual address of BAR4 mmap (HBM window)
//   bar4_phys      -- .quad, physical address of BAR4 (== GPU VA on GH200)
//
// PCI BDF is set by pci_bdf_slot below.  Default 0000:dd:00.0 (this host).
// To change, patch the ASCII string before calling bar_map_init.
//
// !!! WARNING: THE BDF STRING "dd:00" IS DUPLICATED IN THREE PLACES !!!
//     pci_bdf_slot   (bar0_path)
//     pci_bdf_slot4  (bar4_path)
//     pci_bdf_slot_r (resource_path)
// ALL THREE MUST BE PATCHED TOGETHER.  Patching only one will cause
// BAR0/BAR4/phys-parse to reference different devices.  See the
// comments next to each copy below.
//
// Calling convention: ARM64 AAPCS.  Clobbers x0-x8, x16-x17.
// Returns: x0 = 0 on success, -1 on failure.
// On success, bar0_base and bar4_base are populated.
//
// Requires: root (or udev ACL on sysfs resource files).
//           nvidia.ko must be unbound from the device.

// ---- Syscall numbers (aarch64) ----
.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_MMAP,      222
.equ SYS_READ,      63
.equ SYS_WRITE,     64
.equ SYS_EXIT,      93
.equ SYS_FLOCK,     32

// ---- flock constants ----
.equ LOCK_EX,       2
.equ LOCK_NB,       4

// ---- Constants ----
.equ AT_FDCWD,      -100
.equ PROT_READ,     1
.equ PROT_WRITE,    2
.equ PROT_RW,       3
.equ MAP_SHARED,    1

// BAR0 = 16 MB = 0x1000000
.equ BAR0_SIZE,     0x1000000

// BAR4 = 128 GB = 0x2000000000
// For initial bring-up, map a 256 MB window (sufficient for GSP boot
// structures).  Full 128 GB can be mapped later if needed.
.equ BAR4_SIZE_LO,  0x10000000        // 256 MB

// ============================================================
// Data section
// ============================================================
.data
.align 3

bar0_base:      .quad 0               // mmap'd BAR0 virtual address
bar4_base:      .quad 0               // mmap'd BAR4 virtual address
bar4_phys:      .quad 0               // BAR4 physical base (parsed from sysfs)
bar0_lock_fd:   .quad -1              // BAR0 resource fd kept open for flock

.global bar0_base
.global bar4_base
.global bar4_phys

// PCI sysfs paths -- BDF slot is embedded in the string.
// Default: 0000:dd:00.0 (observed on this GH200; typical integrated-GPU BDF
// on older GH200 units is 0000:09:00.0 — patch all three .ascii sites below
// to match `lspci -D | grep NVIDIA`).
// Patch pci_bdf_slot (5 bytes "dd:00") to match your system.
bar0_path:
    .ascii "/sys/bus/pci/devices/0000:"
pci_bdf_slot:                                // !!! BDF COPY 1/3 -- MUST MATCH pci_bdf_slot4 AND pci_bdf_slot_r !!!
    .ascii "dd:00"
    .ascii ".0/resource0"
    .byte 0

bar4_path:
    .ascii "/sys/bus/pci/devices/0000:"
pci_bdf_slot4:                               // !!! BDF COPY 2/3 -- MUST MATCH pci_bdf_slot AND pci_bdf_slot_r !!!
    .ascii "dd:00"
    .ascii ".0/resource4"
    .byte 0

// Physical address resource file (for parsing BAR4 phys addr)
resource_path:
    .ascii "/sys/bus/pci/devices/0000:"
pci_bdf_slot_r:                              // !!! BDF COPY 3/3 -- MUST MATCH pci_bdf_slot AND pci_bdf_slot4 !!!
    .ascii "dd:00"
    .ascii ".0/resource"
    .byte 0

// Status messages
msg_bar0_ok:    .asciz "gsp: BAR0 mapped (16MB GPU registers)\n"
msg_bar0_ok_len = . - msg_bar0_ok - 1
msg_bar4_ok:    .asciz "gsp: BAR4 mapped (HBM coherent window)\n"
msg_bar4_ok_len = . - msg_bar4_ok - 1
msg_bar0_fail:  .asciz "gsp: ERROR: failed to open/mmap BAR0\n"
msg_bar0_fail_len = . - msg_bar0_fail - 1
msg_bar4_fail:  .asciz "gsp: ERROR: failed to open/mmap BAR4\n"
msg_bar4_fail_len = . - msg_bar4_fail - 1
msg_bar0_locked: .asciz "bar_map: GPU locked by another process\n"
msg_bar0_locked_len = . - msg_bar0_locked - 1

// Scratch buffer for reading /resource file (to parse BAR4 phys)
.align 3
resource_buf:   .space 2048

// ============================================================
// Text section
// ============================================================
.text
.align 4

// ------------------------------------------------------------
// bar_map_init -- map BAR0 and BAR4 from sysfs resource files
//
// Returns: x0 = 0 success, x0 = -1 failure
// Side effects: populates bar0_base, bar4_base, bar4_phys
// ------------------------------------------------------------
.global bar_map_init
bar_map_init:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]

    // ---- Parse BAR4 physical address from /resource file ----
    // The sysfs "resource" file has one line per BAR.
    // Line format: 0xSTART 0xEND 0xFLAGS
    // BAR4 is on line index 4 (fifth line, 0-indexed).
    bl      parse_bar4_phys
    cmp     x0, #0
    b.lt    .bar_map_fail_early

    // ---- Open BAR0 resource file ----
    mov     x0, #AT_FDCWD
    adrp    x1, bar0_path
    add     x1, x1, :lo12:bar0_path
    // O_RDWR | O_SYNC = 0x101002 on aarch64
    movz    x2, #0x1002               // low 16 bits
    movk    x2, #0x0010, lsl #16      // bits[31:16]
    mov     x3, #0                    // mode (ignored for open)
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .bar0_open_fail
    mov     x19, x0                   // x19 = bar0_fd

    // Exclusive lock on BAR0 resource file -- prevents concurrent gsp_boot
    mov     x0, x19                   // fd
    mov     x1, #(LOCK_EX | LOCK_NB)  // non-blocking exclusive lock
    mov     x8, #SYS_FLOCK
    svc     #0
    cmp     x0, #0
    b.lt    .bar0_flock_fail          // another process holds the lock

    // ---- mmap BAR0: 16 MB, PROT_READ|PROT_WRITE, MAP_SHARED ----
    mov     x0, #0                    // addr = NULL (kernel chooses)
    mov     x1, #BAR0_SIZE            // length = 16 MB
    mov     x2, #PROT_RW             // prot = PROT_READ | PROT_WRITE
    mov     x3, #MAP_SHARED           // flags = MAP_SHARED (MMIO)
    mov     x4, x19                   // fd = bar0_fd
    mov     x5, #0                    // offset = 0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096                 // MAP_FAILED check (error if >= -4095)
    b.hi    .bar0_mmap_fail

    // Store BAR0 base
    adrp    x1, bar0_base
    add     x1, x1, :lo12:bar0_base
    str     x0, [x1]

    // Keep BAR0 fd open -- flock is released on close, and we need
    // the lock to persist for the lifetime of the boot sequence.
    // The mmap persists independently of the fd.
    adrp    x1, bar0_lock_fd
    add     x1, x1, :lo12:bar0_lock_fd
    str     x19, [x1]              // store fd for later cleanup

    // Print success
    adrp    x1, msg_bar0_ok
    add     x1, x1, :lo12:msg_bar0_ok
    mov     x2, #msg_bar0_ok_len
    bl      bar_print_msg

    // ---- Open BAR4 resource file ----
    mov     x0, #AT_FDCWD
    adrp    x1, bar4_path
    add     x1, x1, :lo12:bar4_path
    movz    x2, #0x1002
    movk    x2, #0x0010, lsl #16
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .bar4_open_fail
    mov     x19, x0                   // x19 = bar4_fd

    // ---- mmap BAR4: 256 MB initial window, MAP_SHARED ----
    mov     x0, #0                    // addr = NULL
    mov     x1, #BAR4_SIZE_LO         // length = 256 MB (initial window)
    mov     x2, #PROT_RW
    mov     x3, #MAP_SHARED
    mov     x4, x19                   // fd = bar4_fd
    mov     x5, #0                    // offset = 0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #4096
    b.hi    .bar4_mmap_fail

    // Store BAR4 base
    adrp    x1, bar4_base
    add     x1, x1, :lo12:bar4_base
    str     x0, [x1]

    // Close BAR4 fd
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Print success
    adrp    x1, msg_bar4_ok
    add     x1, x1, :lo12:msg_bar4_ok
    mov     x2, #msg_bar4_ok_len
    bl      bar_print_msg

    // ---- Success ----
    mov     x0, #0
    b       .bar_map_return

.bar0_flock_fail:
    // flock failed -- close the fd and return error
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
    adrp    x1, msg_bar0_locked
    add     x1, x1, :lo12:msg_bar0_locked
    mov     x2, #msg_bar0_locked_len
    bl      bar_print_msg
    mov     x0, #-2                   // -2 = GPU locked by another process
    b       .bar_map_return

.bar0_mmap_fail:
    // mmap failed but fd in x19 is still open -- close it before erroring
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
.bar0_open_fail:
    adrp    x1, msg_bar0_fail
    add     x1, x1, :lo12:msg_bar0_fail
    mov     x2, #msg_bar0_fail_len
    bl      bar_print_msg
    mov     x0, #-1
    b       .bar_map_return

.bar4_mmap_fail:
    // mmap failed but fd in x19 is still open -- close it before erroring
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
.bar4_open_fail:
    adrp    x1, msg_bar4_fail
    add     x1, x1, :lo12:msg_bar4_fail
    mov     x2, #msg_bar4_fail_len
    bl      bar_print_msg
    mov     x0, #-1
    b       .bar_map_return

.bar_map_fail_early:
    mov     x0, #-1

.bar_map_return:
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// ------------------------------------------------------------
// parse_bar4_phys -- read sysfs "resource" file, extract BAR4
//                    physical base address
//
// The file has 13 lines, one per resource.  Each line:
//   0xSTART 0xEND 0xFLAGS
// BAR4 is resource index 4 (line 5, 0-indexed from 0).
// We skip 4 newlines, then parse the hex start address.
//
// Returns: x0 = 0 on success (bar4_phys populated), -1 on error
// ------------------------------------------------------------
.align 4
parse_bar4_phys:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // Open resource file
    mov     x0, #AT_FDCWD
    adrp    x1, resource_path
    add     x1, x1, :lo12:resource_path
    mov     x2, #0                    // O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .parse_phys_fail
    mov     x19, x0                   // fd

    // Read up to 2048 bytes
    mov     x0, x19
    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    mov     x2, #2048
    mov     x8, #SYS_READ
    svc     #0
    cmp     x0, #0
    b.le    .parse_phys_close_fail
    mov     x20, x0                   // bytes_read

    // Close fd
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0

    // Parse: skip 4 newlines to reach line 4 (BAR4)
    adrp    x1, resource_buf
    add     x1, x1, :lo12:resource_buf
    mov     x2, #4                    // skip 4 lines
    add     x3, x1, x20              // end pointer

.skip_line:
    cbz     x2, .parse_hex
    cmp     x1, x3
    b.ge    .parse_phys_fail_noclose
.skip_char:
    cmp     x1, x3
    b.ge    .parse_phys_fail_noclose
    ldrb    w4, [x1], #1
    cmp     w4, #'\n'
    b.ne    .skip_char
    sub     x2, x2, #1
    b       .skip_line

.parse_hex:
    // x1 now points to start of line 4.  Format: "0xHHHH..."
    // Skip optional leading whitespace
.skip_ws:
    cmp     x1, x3
    b.ge    .parse_phys_fail_noclose
    ldrb    w4, [x1]
    cmp     w4, #' '
    b.ne    .check_0x
    add     x1, x1, #1
    b       .skip_ws

.check_0x:
    // Expect "0x" prefix
    cmp     x1, x3
    b.ge    .parse_phys_fail_noclose
    ldrb    w4, [x1], #1
    cmp     w4, #'0'
    b.ne    .parse_phys_fail_noclose
    cmp     x1, x3
    b.ge    .parse_phys_fail_noclose
    ldrb    w4, [x1], #1
    cmp     w4, #'x'
    b.ne    .parse_phys_fail_noclose

    // Accumulate hex digits into x21
    mov     x21, #0
.hex_loop:
    cmp     x1, x3
    b.ge    .hex_done
    ldrb    w4, [x1], #1
    // Check 0-9
    sub     w5, w4, #'0'
    cmp     w5, #9
    b.ls    .hex_digit
    // Check a-f
    sub     w5, w4, #'a'
    cmp     w5, #5
    b.ls    .hex_alpha_lower
    // Check A-F
    sub     w5, w4, #'A'
    cmp     w5, #5
    b.ls    .hex_alpha_upper
    // Not a hex char -- done
    b       .hex_done

.hex_digit:
    sub     w5, w4, #'0'
    lsl     x21, x21, #4
    orr     x21, x21, x5
    b       .hex_loop

.hex_alpha_lower:
    sub     w5, w4, #'a'
    add     w5, w5, #10
    lsl     x21, x21, #4
    orr     x21, x21, x5
    b       .hex_loop

.hex_alpha_upper:
    sub     w5, w4, #'A'
    add     w5, w5, #10
    lsl     x21, x21, #4
    orr     x21, x21, x5
    b       .hex_loop

.hex_done:
    // x21 = BAR4 physical start address
    adrp    x1, bar4_phys
    add     x1, x1, :lo12:bar4_phys
    str     x21, [x1]

    mov     x0, #0
    b       .parse_phys_return

.parse_phys_close_fail:
    mov     x0, x19
    mov     x8, #SYS_CLOSE
    svc     #0
.parse_phys_fail_noclose:
.parse_phys_fail:
    mov     x0, #-1

.parse_phys_return:
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ------------------------------------------------------------
// bar_print_msg -- write string to stderr (fd 2)
// x1 = string address, x2 = length
// ------------------------------------------------------------
.align 4
bar_print_msg:
    mov     x0, #2                    // stderr
    mov     x8, #SYS_WRITE
    svc     #0
    ret
