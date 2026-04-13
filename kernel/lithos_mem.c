// SPDX-License-Identifier: GPL-2.0-only
/*
 * lithos_mem.c — VRAM allocator for the Lithos GPU inference engine
 *
 * Two allocation paths for GH200:
 *
 * Path A — Coherent HBM (BAR4, 128 GB):
 *   On GH200 the Grace CPU and H100 GPU share HBM via NVLink-C2C (ATS).
 *   BAR4 physical base 0x42000000000 is simultaneously:
 *     - GPU-accessible via its native GPU VA space
 *     - CPU-accessible via ioremap_wc (write-combining) or ioremap_cache
 *   We maintain a simple bump allocator over this pool.  The GPU VA equals
 *   the physical address on GH200 (identity mapping through C2C/ATS).
 *   This is the primary path for weights, KV cache, and activation buffers.
 *
 * Path B — System RAM (vmalloc):
 *   If BAR4 is unavailable (module loaded alongside nvidia.ko, or on a
 *   non-GH200 Hopper variant), we fall back to vmalloc'd pages.  On GH200
 *   with ATS enabled, the GPU MMU can see host pages without DMA mapping;
 *   the GPU VA is the physical address of the vmalloc page.
 *   On non-ATS systems this path would require a separate GPU MMU mapping
 *   (tracked by the RM); that case is not implemented here.
 *
 * Allocator design:
 *   A bump allocator is sufficient for an inference engine:
 *   - Model weights are loaded once at startup and never freed
 *   - KV cache is a single large allocation
 *   - Activation buffers are preallocated in the megakernel
 *   For finer-grained allocation add a free-list on top.
 *
 * Thread safety:
 *   A single spinlock protects the bump pointer.  Allocations from
 *   ioctl handlers are atomic (no sleeping, no waiting).
 */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/io.h>
#include <linux/spinlock.h>
#include <linux/mm.h>
#include <linux/pci.h>

#include "lithos.h"
#include "lithos_dev.h"
#include "lithos_mem.h"

/* Per-free-entry bookkeeping (path B only, no-op for bump allocator) */
struct lithos_free_entry {
    uint64_t gpu_va;
    uint64_t size;
    void    *cpu_va;
    bool     is_vmalloc;
};

#define LITHOS_MAX_ALLOCS 4096

/*
 * Allocator state embedded in the device structure.
 * We keep it in a separately kzalloc'd block so the main device struct
 * stays small.
 */
struct lithos_mem_state {
    /* Path A: coherent HBM bump allocator */
    resource_size_t  hbm_phys_base;   /* BAR4 physical start */
    resource_size_t  hbm_size;        /* BAR4 total size */
    uint64_t         hbm_bump;        /* current bump pointer (offset from base) */

    /* Path B: vmalloc fallback tracking */
    bool             use_vmalloc;

    /* Allocation tracking table (for free) */
    struct lithos_free_entry  allocs[LITHOS_MAX_ALLOCS];
    int                       n_allocs;

    spinlock_t  lock;
};

/* Stashed on the device; accessed via a global here for simplicity.
 * In a multi-GPU driver this would be per-device. */
static struct lithos_mem_state *g_mem;

/*
 * lithos_mem_init — called from probe() after BAR4 physical address is known.
 *
 * pool_phys: physical base of BAR4 (0 if not available)
 * pool_size: size of BAR4 (0 if not available)
 */
int lithos_mem_init(struct lithos_device *ldev,
                    resource_size_t pool_phys, resource_size_t pool_size)
{
    struct lithos_mem_state *mem;

    mem = kzalloc(sizeof(*mem), GFP_KERNEL);
    if (!mem)
        return -ENOMEM;

    spin_lock_init(&mem->lock);

    if (pool_phys && pool_size) {
        mem->hbm_phys_base = pool_phys;
        mem->hbm_size      = pool_size;
        mem->hbm_bump      = 0;
        mem->use_vmalloc   = false;
        pr_info("lithos/mem: coherent HBM pool phys=0x%llx size=%llu GB\n",
                (unsigned long long)pool_phys,
                (unsigned long long)(pool_size >> 30));
    } else {
        mem->use_vmalloc = true;
        pr_warn("lithos/mem: BAR4 not available — using vmalloc fallback\n");
        pr_warn("lithos/mem: GPU VA will be physical address of vmalloc pages; "
                "requires ATS to be enabled on GH200\n");
    }

    g_mem = mem;
    return 0;
}

void lithos_mem_fini(struct lithos_device *ldev)
{
    int i;

    if (!g_mem)
        return;

    /* Free any vmalloc'd allocations still alive */
    for (i = 0; i < g_mem->n_allocs; i++) {
        if (g_mem->allocs[i].is_vmalloc && g_mem->allocs[i].cpu_va) {
            vfree(g_mem->allocs[i].cpu_va);
            g_mem->allocs[i].cpu_va = NULL;
        }
    }

    kfree(g_mem);
    g_mem = NULL;
}

/*
 * lithos_alloc_vram — allocate GPU-accessible memory.
 *
 * Path A (coherent HBM):
 *   Bump the allocator, return gpu_va = hbm_phys_base + offset.
 *   cpu_va is NOT returned here; callers that need CPU access to HBM
 *   should ioremap_wc the physical address themselves or use BAR4 directly.
 *   For the inference engine this is fine: the CPU writes weights during
 *   load (via a temporary ioremap), then lets the GPU own the region.
 *
 * Path B (vmalloc fallback):
 *   Allocate page-aligned memory with vmalloc_32 (stays below 4 GB for
 *   easier GPU VA handling on older HW; on GH200 with ATS the full PA
 *   range is accessible so regular vmalloc is fine).
 *   gpu_va = virt_to_phys(cpu_va) — valid only with ATS.
 */
int lithos_alloc_vram(struct lithos_device *ldev, uint64_t size,
                       uint64_t *gpu_va_out, uint64_t *cpu_va_out)
{
    struct lithos_free_entry *entry;
    unsigned long flags;
    uint64_t aligned_size;
    int ret = 0;

    if (!g_mem)
        return -ENODEV;

    /* Align to 2 MB for GPU TLB efficiency (large page boundary) */
    aligned_size = ALIGN(size, SZ_2M);

    spin_lock_irqsave(&g_mem->lock, flags);

    if (g_mem->n_allocs >= LITHOS_MAX_ALLOCS) {
        spin_unlock_irqrestore(&g_mem->lock, flags);
        return -ENOMEM;
    }

    entry = &g_mem->allocs[g_mem->n_allocs];

    if (!g_mem->use_vmalloc) {
        /* Path A: coherent HBM bump allocator */
        if (g_mem->hbm_bump + aligned_size > g_mem->hbm_size) {
            pr_err("lithos/mem: HBM pool exhausted "
                   "(bump=%llu size=%llu pool=%llu)\n",
                   (unsigned long long)g_mem->hbm_bump,
                   (unsigned long long)aligned_size,
                   (unsigned long long)g_mem->hbm_size);
            ret = -ENOMEM;
            goto out;
        }

        entry->gpu_va    = g_mem->hbm_phys_base + g_mem->hbm_bump;
        entry->cpu_va    = NULL;   /* caller ioremaps if needed */
        entry->size      = aligned_size;
        entry->is_vmalloc = false;

        g_mem->hbm_bump += aligned_size;

        *gpu_va_out = entry->gpu_va;
        *cpu_va_out = 0;   /* not directly CPU-mapped by default */

        pr_info("lithos/mem: HBM alloc size=%llu gpu_va=0x%llx bump=%llu MB\n",
                (unsigned long long)aligned_size,
                (unsigned long long)entry->gpu_va,
                (unsigned long long)(g_mem->hbm_bump >> 20));

    } else {
        /* Path B: vmalloc fallback */
        void *kva;

        spin_unlock_irqrestore(&g_mem->lock, flags);

        kva = vmalloc(aligned_size);
        if (!kva)
            return -ENOMEM;

        spin_lock_irqsave(&g_mem->lock, flags);

        if (g_mem->n_allocs >= LITHOS_MAX_ALLOCS) {
            spin_unlock_irqrestore(&g_mem->lock, flags);
            vfree(kva);
            return -ENOMEM;
        }

        entry = &g_mem->allocs[g_mem->n_allocs];
        entry->cpu_va    = kva;
        entry->gpu_va    = (uint64_t)virt_to_phys(kva);
        entry->size      = aligned_size;
        entry->is_vmalloc = true;

        *gpu_va_out = entry->gpu_va;
        *cpu_va_out = (uint64_t)(uintptr_t)kva;

        pr_info("lithos/mem: vmalloc alloc size=%llu gpu_va=0x%llx cpu_va=0x%llx\n",
                (unsigned long long)aligned_size,
                (unsigned long long)entry->gpu_va,
                (unsigned long long)*cpu_va_out);
    }

    g_mem->n_allocs++;

out:
    spin_unlock_irqrestore(&g_mem->lock, flags);
    return ret;
}

/*
 * lithos_free_vram — release a previous allocation.
 *
 * For Path A (HBM): bump allocator doesn't actually free; we just mark
 * the entry invalid.  A proper free-list can be added later but the
 * inference engine's static allocation pattern doesn't require it.
 *
 * For Path B (vmalloc): actually calls vfree().
 */
int lithos_free_vram(struct lithos_device *ldev,
                      uint64_t gpu_va, uint64_t size)
{
    struct lithos_free_entry *entry;
    unsigned long flags;
    int i;

    if (!g_mem)
        return -ENODEV;

    spin_lock_irqsave(&g_mem->lock, flags);

    for (i = 0; i < g_mem->n_allocs; i++) {
        entry = &g_mem->allocs[i];
        if (entry->gpu_va == gpu_va && entry->size == ALIGN(size, SZ_2M)) {
            if (entry->is_vmalloc && entry->cpu_va) {
                spin_unlock_irqrestore(&g_mem->lock, flags);
                vfree(entry->cpu_va);
                spin_lock_irqsave(&g_mem->lock, flags);
                entry->cpu_va = NULL;
            }
            entry->gpu_va = 0;
            entry->size   = 0;
            spin_unlock_irqrestore(&g_mem->lock, flags);
            pr_info("lithos/mem: freed gpu_va=0x%llx\n",
                    (unsigned long long)gpu_va);
            return 0;
        }
    }

    spin_unlock_irqrestore(&g_mem->lock, flags);
    pr_warn("lithos/mem: free: gpu_va=0x%llx not found\n",
            (unsigned long long)gpu_va);
    return -EINVAL;
}
