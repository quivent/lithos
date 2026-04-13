/*
 * qmd_probe.cu — Differential QMD probing for Hopper sm_90 (GH200)
 *
 * Finds the byte offsets of entry_pc, register_count, and shared_mem_size
 * within the 528-byte QMD by launching pairs of kernels that differ in
 * exactly one parameter, snapshotting the GPFIFO pushbuffer ring, and
 * diffing the inline QMD bodies.
 *
 * Build:
 *   nvcc -o /home/ubuntu/lithos/tools/qmd_probe \
 *        /home/ubuntu/lithos/tools/qmd_probe.cu \
 *        -arch=sm_90 -lcuda
 *
 * Run:
 *   /home/ubuntu/lithos/tools/qmd_probe
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <setjmp.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Safe memory copy (handles SIGSEGV from unmapped ring pages)         */
/* ------------------------------------------------------------------ */

static sigjmp_buf jmp_env;
static void segv_handler(int s) { siglongjmp(jmp_env, 1); }

static int safe_memcpy(void *dst, const void *src, size_t n) {
    struct sigaction sa = {0}, old_segv, old_bus;
    sa.sa_handler = segv_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);
    int ok = (sigsetjmp(jmp_env, 1) == 0);
    if (ok) memcpy(dst, src, n);
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);
    return ok;
}

/* ------------------------------------------------------------------ */
/* GPFIFO ring scanner                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    uintptr_t start;
    uintptr_t end;
    char perms[8];
    char path[256];
} MapEntry;

static int read_maps(MapEntry *out, int max) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    int n = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) && n < max) {
        uintptr_t s, e;
        char p[8] = {0}, o[16], d[16], ino[16], path[256] = {0};
        if (sscanf(line, "%lx-%lx %7s %15s %15s %15s %255[^\n]",
                   &s, &e, p, o, d, ino, path) >= 3) {
            out[n].start = s;
            out[n].end = e;
            memcpy(out[n].perms, p, 7);
            strncpy(out[n].path, path, 255);
            n++;
        }
    }
    fclose(f);
    return n;
}

/* Find GPFIFO ring regions: MAP_SHARED, 2MB or 4MB, from /dev/zero */
#define MAX_RINGS 32
static int find_rings(uintptr_t *addrs, size_t *sizes) {
    MapEntry maps[4096];
    int nm = read_maps(maps, 4096);
    int nr = 0;
    for (int i = 0; i < nm && nr < MAX_RINGS; i++) {
        size_t sz = maps[i].end - maps[i].start;
        if (sz < 0x100000) continue;  /* at least 1MB */
        if (sz > 0x1000000) continue; /* at most 16MB */
        if (maps[i].perms[0] != 'r') continue;
        if (maps[i].perms[3] != 's') continue;  /* must be shared */
        /* Try to read a byte to verify accessibility */
        uint8_t tmp;
        if (safe_memcpy(&tmp, (void *)maps[i].start, 1)) {
            addrs[nr] = maps[i].start;
            sizes[nr] = sz;
            nr++;
        }
    }
    return nr;
}

/* ------------------------------------------------------------------ */
/* Ring snapshot and diff                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t size;
    uintptr_t base;
} RingSnap;

static RingSnap *snap_rings(uintptr_t *addrs, size_t *sizes, int nrings) {
    RingSnap *snaps = (RingSnap *)calloc(nrings, sizeof(RingSnap));
    for (int i = 0; i < nrings; i++) {
        snaps[i].data = (uint8_t *)malloc(sizes[i]);
        snaps[i].size = sizes[i];
        snaps[i].base = addrs[i];
        if (!safe_memcpy(snaps[i].data, (void *)addrs[i], sizes[i])) {
            memset(snaps[i].data, 0, sizes[i]);
        }
    }
    return snaps;
}

static void free_snaps(RingSnap *s, int n) {
    for (int i = 0; i < n; i++) free(s[i].data);
    free(s);
}

/*
 * Find the QMD inline data by looking for the NONINC header with count=132.
 * Pushbuffer format: bits[31:29]=opcode, bits[28:16]=count, bits[15:13]=subch
 * NONINC opcode = 3 (0x60000000), count=132 (0x00840000), subch=1 (0x2000)
 * addr=0x01b4 (LOAD_INLINE_DATA)
 * Full header: 0x60862000 | 0x01b4 = ...
 *
 * Actually let's just search for the changed region and extract 132 dwords
 * after the NONINC header.
 */

#define QMD_DWORDS 132
#define QMD_BYTES  (QMD_DWORDS * 4)

/* Extract QMD from a ring snapshot by finding the NONINC sc=1 addr=0x01b4 count=132 header.
 * Returns pointer to static buffer with the 132 dwords, or NULL. */
static int extract_qmd_from_diff(RingSnap *before, RingSnap *after,
                                  uint32_t *qmd_out)
{
    /* Find the region that changed */
    if (before->size != after->size) return 0;
    size_t sz = before->size;
    int first_diff = -1, last_diff = -1;
    for (int i = 0; i < (int)sz; i++) {
        if (before->data[i] != after->data[i]) {
            if (first_diff < 0) first_diff = i;
            last_diff = i;
        }
    }
    if (first_diff < 0) return 0;

    /* Scan the changed region for the NONINC header to LOAD_INLINE_DATA */
    /* Align to dword boundary */
    int start = first_diff & ~3;
    int end = (last_diff + 4) & ~3;
    const uint32_t *dw = (const uint32_t *)(after->data + start);
    int ndw = (end - start) / 4;

    for (int i = 0; i < ndw - QMD_DWORDS; i++) {
        uint32_t hdr = dw[i];
        int opcode = (hdr >> 29) & 7;
        int count = (hdr >> 16) & 0x1FFF;
        int subch = (hdr >> 13) & 7;
        int addr = hdr & 0x1FFF;

        /* NONINC=3, subch=1, addr=0x01b4, count=132 */
        if (opcode == 3 && subch == 1 && addr == 0x01b4 && count == 132) {
            memcpy(qmd_out, &dw[i + 1], QMD_BYTES);
            return 1;
        }
    }

    /* Fallback: look for any NONINC with count=132 */
    for (int i = 0; i < ndw - QMD_DWORDS; i++) {
        uint32_t hdr = dw[i];
        int opcode = (hdr >> 29) & 7;
        int count = (hdr >> 16) & 0x1FFF;
        if (opcode == 3 && count == 132) {
            fprintf(stderr, "  [fallback] Found NONINC count=132 at ring+0x%x\n",
                    start + i * 4);
            memcpy(qmd_out, &dw[i + 1], QMD_BYTES);
            return 1;
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Kernels for probing                                                  */
/* ------------------------------------------------------------------ */

/* Kernel A: minimal register usage */
__global__ void kern_few_regs(float *out) {
    out[0] = 1.0f;
}

/* Kernel B: heavy register usage (volatile arrays force register spill/usage) */
__global__ void __launch_bounds__(256, 1) kern_many_regs(float *out) {
    volatile float a0  = 1.0f,  a1  = 2.0f,  a2  = 3.0f,  a3  = 4.0f;
    volatile float a4  = 5.0f,  a5  = 6.0f,  a6  = 7.0f,  a7  = 8.0f;
    volatile float a8  = 9.0f,  a9  = 10.0f, a10 = 11.0f, a11 = 12.0f;
    volatile float a12 = 13.0f, a13 = 14.0f, a14 = 15.0f, a15 = 16.0f;
    volatile float a16 = 17.0f, a17 = 18.0f, a18 = 19.0f, a19 = 20.0f;
    volatile float a20 = 21.0f, a21 = 22.0f, a22 = 23.0f, a23 = 24.0f;
    volatile float a24 = 25.0f, a25 = 26.0f, a26 = 27.0f, a27 = 28.0f;
    volatile float a28 = 29.0f, a29 = 30.0f, a30 = 31.0f, a31 = 32.0f;
    volatile float a32 = 33.0f, a33 = 34.0f, a34 = 35.0f, a35 = 36.0f;
    volatile float a36 = 37.0f, a37 = 38.0f, a38 = 39.0f, a39 = 40.0f;
    volatile float a40 = 41.0f, a41 = 42.0f, a42 = 43.0f, a43 = 44.0f;
    volatile float a44 = 45.0f, a45 = 46.0f, a46 = 47.0f, a47 = 48.0f;
    volatile float a48 = 49.0f, a49 = 50.0f, a50 = 51.0f, a51 = 52.0f;
    volatile float a52 = 53.0f, a53 = 54.0f, a54 = 55.0f, a55 = 56.0f;
    volatile float a56 = 57.0f, a57 = 58.0f, a58 = 59.0f, a59 = 60.0f;
    volatile float a60 = 61.0f, a61 = 62.0f, a62 = 63.0f, a63 = 64.0f;
    out[0] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
             a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 +
             a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 +
             a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 +
             a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 +
             a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47 +
             a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55 +
             a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63;
}

/* Simple kernel for shared_mem_size experiment (same kernel, different smem) */
extern "C" __global__ void kern_smem(float *out) {
    extern __shared__ float smem[];
    smem[threadIdx.x] = 1.0f;
    __syncthreads();
    out[0] = smem[0];
}

/* ------------------------------------------------------------------ */
/* Diff and report                                                      */
/* ------------------------------------------------------------------ */

static void diff_qmds(const char *label,
                       const uint32_t *qmd_a, const uint32_t *qmd_b,
                       const char *desc_a, const char *desc_b)
{
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "EXPERIMENT: %s\n", label);
    fprintf(stderr, "  A: %s\n  B: %s\n", desc_a, desc_b);
    fprintf(stderr, "========================================\n");

    int ndiff = 0;
    for (int i = 0; i < QMD_DWORDS; i++) {
        if (qmd_a[i] != qmd_b[i]) {
            fprintf(stderr, "  DIFF dword[%3d] byte 0x%04x: A=0x%08x (%u)  B=0x%08x (%u)\n",
                    i, i * 4, qmd_a[i], qmd_a[i], qmd_b[i], qmd_b[i]);
            ndiff++;
        }
    }
    if (ndiff == 0) {
        fprintf(stderr, "  NO DIFFERENCES FOUND (unexpected!)\n");
    }
    fprintf(stderr, "  Total differing dwords: %d\n", ndiff);
}

static void dump_qmd(const char *label, const uint32_t *qmd) {
    fprintf(stderr, "\n--- Full QMD: %s ---\n", label);
    for (int i = 0; i < QMD_DWORDS; i++) {
        if (qmd[i] != 0 || (i < 10)) {
            fprintf(stderr, "  dw[%3d] 0x%04x = 0x%08x (%u)\n",
                    i, i * 4, qmd[i], qmd[i]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Main: Run 3 experiments                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    fprintf(stderr, "=== QMD Differential Probe Tool (Hopper sm_90) ===\n\n");

    /* Discover GPFIFO rings */
    uintptr_t ring_addrs[MAX_RINGS];
    size_t ring_sizes[MAX_RINGS];

    /* Force CUDA init first to get the rings mapped */
    float *d_out;
    cudaError_t ce = cudaMalloc(&d_out, 4096);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(ce));
        return 1;
    }

    /* Warm up: launch a dummy kernel so pushbuffer ring is established */
    kern_few_regs<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    int nrings = find_rings(ring_addrs, ring_sizes);
    fprintf(stderr, "Found %d candidate GPFIFO ring(s):\n", nrings);
    for (int i = 0; i < nrings; i++) {
        fprintf(stderr, "  ring[%d]: 0x%lx size=0x%lx (%lu KB)\n",
                i, ring_addrs[i], ring_sizes[i], ring_sizes[i] / 1024);
    }
    if (nrings == 0) {
        fprintf(stderr, "ERROR: No GPFIFO rings found!\n");
        return 1;
    }

    /* Query register counts via CUDA API */
    {
        cudaFuncAttributes attr_a, attr_b;
        cudaFuncGetAttributes(&attr_a, (const void *)kern_few_regs);
        cudaFuncGetAttributes(&attr_b, (const void *)kern_many_regs);
        fprintf(stderr, "\nKernel register counts (from CUDA API):\n");
        fprintf(stderr, "  kern_few_regs:  %d registers\n", attr_a.numRegs);
        fprintf(stderr, "  kern_many_regs: %d registers\n", attr_b.numRegs);
    }

    /* ============================================================== */
    /* EXPERIMENT 1: entry_pc — two different kernels                    */
    /* ============================================================== */
    fprintf(stderr, "\n\n*** EXPERIMENT 1: entry_pc ***\n");

    uint32_t qmd_entry_a[QMD_DWORDS], qmd_entry_b[QMD_DWORDS];
    int got_entry_a = 0, got_entry_b = 0;

    {
        RingSnap *snap_before = snap_rings(ring_addrs, ring_sizes, nrings);
        kern_few_regs<<<1, 1>>>(d_out);
        cudaDeviceSynchronize();
        RingSnap *snap_after = snap_rings(ring_addrs, ring_sizes, nrings);

        for (int r = 0; r < nrings && !got_entry_a; r++) {
            got_entry_a = extract_qmd_from_diff(&snap_before[r], &snap_after[r], qmd_entry_a);
            if (got_entry_a) fprintf(stderr, "  Got QMD for kern_few_regs from ring[%d]\n", r);
        }
        free_snaps(snap_before, nrings);
        free_snaps(snap_after, nrings);
    }

    /* Small delay to ensure pushbuffer advances */
    cudaDeviceSynchronize();

    {
        RingSnap *snap_before = snap_rings(ring_addrs, ring_sizes, nrings);
        kern_many_regs<<<1, 1>>>(d_out);
        cudaDeviceSynchronize();
        RingSnap *snap_after = snap_rings(ring_addrs, ring_sizes, nrings);

        for (int r = 0; r < nrings && !got_entry_b; r++) {
            got_entry_b = extract_qmd_from_diff(&snap_before[r], &snap_after[r], qmd_entry_b);
            if (got_entry_b) fprintf(stderr, "  Got QMD for kern_many_regs from ring[%d]\n", r);
        }
        free_snaps(snap_before, nrings);
        free_snaps(snap_after, nrings);
    }

    if (got_entry_a && got_entry_b) {
        dump_qmd("kern_few_regs", qmd_entry_a);
        dump_qmd("kern_many_regs", qmd_entry_b);
        diff_qmds("entry_pc + register_count (two different kernels)",
                   qmd_entry_a, qmd_entry_b,
                   "kern_few_regs (few regs, small code)",
                   "kern_many_regs (many regs, large code)");
    } else {
        fprintf(stderr, "  FAILED: could not extract QMDs (a=%d b=%d)\n",
                got_entry_a, got_entry_b);
    }

    /* ============================================================== */
    /* EXPERIMENT 2: register_count — same kernel with maxrregcount     */
    /* We already get this from Experiment 1 since kern_few_regs and    */
    /* kern_many_regs differ in register count.                         */
    /* But we also note it explicitly.                                  */
    /* ============================================================== */
    fprintf(stderr, "\n\n*** EXPERIMENT 2: register_count ***\n");
    fprintf(stderr, "  (Covered by Experiment 1 — kern_few_regs vs kern_many_regs)\n");
    fprintf(stderr, "  Diffs that match the register count delta are register_count field.\n");

    /* ============================================================== */
    /* EXPERIMENT 3: shared_mem_size — same kernel, different smem       */
    /* ============================================================== */
    fprintf(stderr, "\n\n*** EXPERIMENT 3: shared_mem_size ***\n");

    uint32_t qmd_smem_a[QMD_DWORDS], qmd_smem_b[QMD_DWORDS];
    int got_smem_a = 0, got_smem_b = 0;

    /* Launch with 0 dynamic shared memory */
    {
        RingSnap *snap_before = snap_rings(ring_addrs, ring_sizes, nrings);
        kern_smem<<<1, 32, 0>>>(d_out);
        cudaDeviceSynchronize();
        RingSnap *snap_after = snap_rings(ring_addrs, ring_sizes, nrings);

        for (int r = 0; r < nrings && !got_smem_a; r++) {
            got_smem_a = extract_qmd_from_diff(&snap_before[r], &snap_after[r], qmd_smem_a);
            if (got_smem_a) fprintf(stderr, "  Got QMD for smem=0 from ring[%d]\n", r);
        }
        free_snaps(snap_before, nrings);
        free_snaps(snap_after, nrings);
    }

    cudaDeviceSynchronize();

    /* Launch with 4096 bytes dynamic shared memory */
    {
        RingSnap *snap_before = snap_rings(ring_addrs, ring_sizes, nrings);
        kern_smem<<<1, 32, 4096>>>(d_out);
        cudaDeviceSynchronize();
        RingSnap *snap_after = snap_rings(ring_addrs, ring_sizes, nrings);

        for (int r = 0; r < nrings && !got_smem_b; r++) {
            got_smem_b = extract_qmd_from_diff(&snap_before[r], &snap_after[r], qmd_smem_b);
            if (got_smem_b) fprintf(stderr, "  Got QMD for smem=4096 from ring[%d]\n", r);
        }
        free_snaps(snap_before, nrings);
        free_snaps(snap_after, nrings);
    }

    if (got_smem_a && got_smem_b) {
        dump_qmd("smem=0", qmd_smem_a);
        dump_qmd("smem=4096", qmd_smem_b);
        diff_qmds("shared_mem_size (same kernel, 0 vs 4096 dynamic smem)",
                   qmd_smem_a, qmd_smem_b,
                   "kern_smem with 0 bytes dynamic shared",
                   "kern_smem with 4096 bytes dynamic shared");
    } else {
        fprintf(stderr, "  FAILED: could not extract QMDs (a=%d b=%d)\n",
                got_smem_a, got_smem_b);
    }

    /* ============================================================== */
    /* EXPERIMENT 4 (bonus): shared_mem_size with 8192 for confirmation */
    /* ============================================================== */
    fprintf(stderr, "\n\n*** EXPERIMENT 4: shared_mem_size confirmation (8192) ***\n");

    uint32_t qmd_smem_c[QMD_DWORDS];
    int got_smem_c = 0;

    {
        RingSnap *snap_before = snap_rings(ring_addrs, ring_sizes, nrings);
        kern_smem<<<1, 32, 8192>>>(d_out);
        cudaDeviceSynchronize();
        RingSnap *snap_after = snap_rings(ring_addrs, ring_sizes, nrings);

        for (int r = 0; r < nrings && !got_smem_c; r++) {
            got_smem_c = extract_qmd_from_diff(&snap_before[r], &snap_after[r], qmd_smem_c);
            if (got_smem_c) fprintf(stderr, "  Got QMD for smem=8192 from ring[%d]\n", r);
        }
        free_snaps(snap_before, nrings);
        free_snaps(snap_after, nrings);
    }

    if (got_smem_a && got_smem_c) {
        diff_qmds("shared_mem_size confirmation (0 vs 8192)",
                   qmd_smem_a, qmd_smem_c,
                   "kern_smem with 0 bytes dynamic shared",
                   "kern_smem with 8192 bytes dynamic shared");
    }

    /* ============================================================== */
    /* Summary                                                          */
    /* ============================================================== */
    fprintf(stderr, "\n\n========================================\n");
    fprintf(stderr, "SUMMARY OF FINDINGS\n");
    fprintf(stderr, "========================================\n");

    if (got_entry_a && got_entry_b) {
        fprintf(stderr, "\nEntry PC + Register Count (Experiment 1):\n");
        cudaFuncAttributes attr_a, attr_b;
        cudaFuncGetAttributes(&attr_a, (const void *)kern_few_regs);
        cudaFuncGetAttributes(&attr_b, (const void *)kern_many_regs);

        for (int i = 0; i < QMD_DWORDS; i++) {
            if (qmd_entry_a[i] != qmd_entry_b[i]) {
                uint32_t a = qmd_entry_a[i], b = qmd_entry_b[i];
                fprintf(stderr, "  dw[%3d] 0x%04x: 0x%08x vs 0x%08x",
                        i, i * 4, a, b);

                /* Check if this is register count */
                if ((a & 0xFF) == (uint32_t)attr_a.numRegs ||
                    (a & 0xFF) == (uint32_t)attr_b.numRegs)
                    fprintf(stderr, "  <-- LIKELY register_count (lo byte matches %d or %d)",
                            attr_a.numRegs, attr_b.numRegs);
                if (((a >> 16) & 0xFF) == (uint32_t)attr_a.numRegs)
                    fprintf(stderr, "  <-- LIKELY register_count (bits[23:16] = %d)",
                            attr_a.numRegs);

                /* Check if large difference (entry_pc changes a lot) */
                int64_t delta = (int64_t)b - (int64_t)a;
                if (delta > 0x100 || delta < -0x100)
                    fprintf(stderr, "  <-- LIKELY entry_pc (large delta: %ld)", delta);

                fprintf(stderr, "\n");
            }
        }
    }

    if (got_smem_a && got_smem_b) {
        fprintf(stderr, "\nShared Memory Size (Experiment 3):\n");
        for (int i = 0; i < QMD_DWORDS; i++) {
            if (qmd_smem_a[i] != qmd_smem_b[i]) {
                uint32_t a = qmd_smem_a[i], b = qmd_smem_b[i];
                fprintf(stderr, "  dw[%3d] 0x%04x: 0x%08x (%u) vs 0x%08x (%u)",
                        i, i * 4, a, a, b, b);
                if (b - a == 4096)
                    fprintf(stderr, "  <-- CONFIRMED shared_mem_size (delta = 4096)");
                else if (b == 4096 && a == 0)
                    fprintf(stderr, "  <-- CONFIRMED shared_mem_size (0 -> 4096)");
                fprintf(stderr, "\n");
            }
        }
    }

    if (got_smem_a && got_smem_c) {
        fprintf(stderr, "\nShared Memory Size Confirmation (Experiment 4):\n");
        for (int i = 0; i < QMD_DWORDS; i++) {
            if (qmd_smem_a[i] != qmd_smem_c[i]) {
                uint32_t a = qmd_smem_a[i], c = qmd_smem_c[i];
                fprintf(stderr, "  dw[%3d] 0x%04x: 0x%08x (%u) vs 0x%08x (%u)",
                        i, i * 4, a, a, c, c);
                if (c - a == 8192)
                    fprintf(stderr, "  <-- CONFIRMED (delta = 8192)");
                else if (c == 8192 && a == 0)
                    fprintf(stderr, "  <-- CONFIRMED (0 -> 8192)");
                fprintf(stderr, "\n");
            }
        }
    }

    /* Save raw QMDs to file for archiving */
    FILE *fout = fopen("/tmp/qmd_probe_results.bin", "wb");
    if (fout) {
        if (got_entry_a) fwrite(qmd_entry_a, 4, QMD_DWORDS, fout);
        if (got_entry_b) fwrite(qmd_entry_b, 4, QMD_DWORDS, fout);
        if (got_smem_a)  fwrite(qmd_smem_a,  4, QMD_DWORDS, fout);
        if (got_smem_b)  fwrite(qmd_smem_b,  4, QMD_DWORDS, fout);
        if (got_smem_c)  fwrite(qmd_smem_c,  4, QMD_DWORDS, fout);
        fclose(fout);
        fprintf(stderr, "\nRaw QMDs saved to /tmp/qmd_probe_results.bin\n");
    }

    cudaFree(d_out);
    return 0;
}
