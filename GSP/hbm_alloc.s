// hbm_alloc.s -- Step 4: BAR4 bump allocator for GSP boot
//
// Allocates coherent HBM regions from BAR4 for GSP firmware structures.
// On GH200 with C2C/ATS: CPU PA == GPU VA for BAR4 addresses.
// CPU stores to BAR4 are GPU-visible without any cache flush.
//
// Single-threaded boot -- no locks, no free.  Bump only.
//
// Build:
//   as -o hbm_alloc.o hbm_alloc.s

.equ ALIGN_2MB,      0x200000          // 2MB alignment boundary
.equ ALIGN_2MB_MASK, 0x1FFFFF          // mask for 2MB alignment

// ============================================================
// Data section
// ============================================================

.data

.align 3
hbm_base:   .quad 0     // BAR4 mmap virtual address (set by caller)
hbm_phys:   .quad 0     // BAR4 physical base, e.g. 0x42000000000
hbm_bump:   .quad 0     // current offset from base (starts at 0)
hbm_limit:  .quad 0     // BAR4 upper bound (base + 256MB)

// ============================================================
// Text section
// ============================================================

.text

// ---------------------------------------------------------------
// hbm_alloc_init -- initialize the bump allocator
//
// x0 = BAR4 mmap address (cpu virtual)
// x1 = BAR4 physical base address
// x2 = initial offset (typically 0, or skip reserved region)
//
// Clobbers: none beyond storing to data section
// ---------------------------------------------------------------
.globl hbm_alloc_init
hbm_alloc_init:
    adrp    x3, hbm_base
    add     x3, x3, :lo12:hbm_base
    str     x0, [x3]               // hbm_base = mmap addr

    adrp    x3, hbm_phys
    add     x3, x3, :lo12:hbm_phys
    str     x1, [x3]               // hbm_phys = physical base

    adrp    x3, hbm_bump
    add     x3, x3, :lo12:hbm_bump
    str     x2, [x3]               // hbm_bump = initial offset

    // Compute and store BAR4 limit = bar4_base + 256MB
    mov     x4, #0x10000000
    add     x4, x0, x4
    adrp    x3, hbm_limit
    add     x3, x3, :lo12:hbm_limit
    str     x4, [x3]               // hbm_limit = bar4_base + 256MB

    mov     x0, #0                  // return 0 = success
    ret

// ---------------------------------------------------------------
// hbm_alloc -- bump-allocate from BAR4
//
// Input:
//   x0 = requested size in bytes
//
// Output:
//   x0 = cpu_addr  (virtual address in mmap'd BAR4 region)
//   x1 = gpu_va    (physical address = GPU virtual address)
//
// The allocation is aligned UP to a 2MB boundary so it can be
// used for WPR regions, page tables, and DMA structures that
// require large-page alignment.
//
// Clobbers: x2, x3, x4, x5
// ---------------------------------------------------------------
.globl hbm_alloc
hbm_alloc:
    // Align size up to 2MB: size = (size + 0x1FFFFF) & ~0x1FFFFF
    mov     x2, ALIGN_2MB_MASK
    add     x0, x0, x2              // size + 0x1FFFFF
    bic     x0, x0, x2              // clear low 21 bits => aligned size
    // x0 = aligned_size

    // Load current bump offset
    adrp    x3, hbm_bump
    add     x3, x3, :lo12:hbm_bump
    ldr     x4, [x3]               // x4 = old bump offset

    // Also align the bump offset itself to 2MB (defensive)
    add     x4, x4, x2
    bic     x4, x4, x2             // x4 = aligned old bump

    // Compute new bump = aligned old bump + aligned size
    add     x5, x4, x0

    // bounds check: new_bump (cpu_va) must not exceed limit
    adrp    x6, hbm_limit
    add     x6, x6, :lo12:hbm_limit
    ldr     x6, [x6]
    // compute absolute address
    adrp    x7, hbm_base
    add     x7, x7, :lo12:hbm_base
    ldr     x7, [x7]
    add     x8, x7, x5             // absolute new_bump = base + offset
    cmp     x8, x6
    b.hi    .alloc_oom

    str     x5, [x3]               // store new bump offset
    dsb     st                      // ensure store visible before caller writes to BAR4

    // cpu_addr = hbm_base + aligned old bump
    adrp    x3, hbm_base
    add     x3, x3, :lo12:hbm_base
    ldr     x3, [x3]
    add     x0, x3, x4             // x0 = cpu_addr

    // gpu_va = hbm_phys + aligned old bump
    adrp    x3, hbm_phys
    add     x3, x3, :lo12:hbm_phys
    ldr     x3, [x3]
    add     x1, x3, x4             // x1 = gpu_va (== physical addr)

    ret

.alloc_oom:
    mov     x0, #-1                // x0 = -1 indicates OOM
    mov     x1, #0                  // x1 = 0 (no valid gpu_va)
    ret

// ---------------------------------------------------------------
// hbm_pos -- return current allocator position (for bookkeeping)
//
// Output:
//   x0 = current bump offset (bytes allocated so far)
// ---------------------------------------------------------------
.globl hbm_pos
hbm_pos:
    adrp    x3, hbm_bump
    add     x3, x3, :lo12:hbm_bump
    ldr     x0, [x3]
    ret
