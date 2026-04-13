/*
 * qmd_probe_driver.c — Differential QMD probing using CUDA driver API.
 *
 * Proven approach (matches /tmp/qmd_ringcompare.c): cuLaunchKernel on cubins
 * with ring snapshot at 0x200800000. Finds entry_pc, register_count, and
 * shared_mem_size by launching kernel pairs that differ in exactly one
 * parameter.
 *
 * Build:
 *   gcc -O0 -g -o /home/ubuntu/lithos/tools/qmd_probe_driver \
 *       /home/ubuntu/lithos/tools/qmd_probe_driver.c \
 *       -I/usr/local/cuda/include -lcuda -ldl -lpthread
 *
 * Run:
 *   /home/ubuntu/lithos/tools/qmd_probe_driver
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <setjmp.h>
#include <signal.h>
#include <cuda.h>

#define CHECK(call) do { \
    CUresult r = (call); \
    if (r != CUDA_SUCCESS) { \
        const char *s = NULL; cuGetErrorString(r, &s); \
        fprintf(stderr, "CUDA err %d (%s) L%d\n", (int)r, s?s:"?", __LINE__); \
        exit(1); \
    } \
} while(0)

/* Safe memcpy */
static sigjmp_buf jmp_env;
static void segv(int s) { siglongjmp(jmp_env, 1); }
static int scopy(void *dst, const void *src, size_t n) {
    struct sigaction sa = {0}, o_sv, o_bus;
    sa.sa_handler = segv; sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o_sv); sigaction(SIGBUS, &sa, &o_bus);
    int ok = (sigsetjmp(jmp_env, 1) == 0);
    if (ok) memcpy(dst, src, n);
    sigaction(SIGSEGV, &o_sv, NULL); sigaction(SIGBUS, &o_bus, NULL);
    return ok;
}

/* Primary ring at 0x200800000, size 4MB */
#define RING_BASE 0x200800000UL
#define RING_SIZE 0x400000UL

#define QMD_DW 132
#define QMD_SZ (QMD_DW * 4)

/* Extract QMD from ring diff — find NONINC count=132 header in the changed region */
static int extract_qmd(const uint8_t *before, const uint8_t *after,
                       size_t sz, uint32_t *qmd_out)
{
    const uint32_t *dw_b = (const uint32_t *)before;
    const uint32_t *dw_a = (const uint32_t *)after;
    int ndw = (int)(sz / 4);

    /* Find first diff byte */
    int first = -1, last = -1;
    for (size_t k = 0; k < sz; k++) {
        if (before[k] != after[k]) {
            if (first < 0) first = (int)k;
            last = (int)k;
        }
    }
    if (first < 0) return 0;

    /* Search the changed dword range for NONINC count=132 */
    int start_dw = (first & ~3) / 4;
    int end_dw = ((last + 4) & ~3) / 4;
    if (start_dw < 0) start_dw = 0;
    if (end_dw > ndw - QMD_DW - 1) end_dw = ndw - QMD_DW - 1;

    /* Take the LAST NONINC count=132 in the diff region (most recent launch) */
    int best = -1;
    for (int i = start_dw; i < end_dw; i++) {
        uint32_t hdr = dw_a[i];
        int opcode = (hdr >> 29) & 7;
        int count  = (hdr >> 16) & 0x1FFF;
        if (opcode == 3 && count == 132) {
            best = i;
        }
    }
    if (best >= 0) {
        memcpy(qmd_out, &dw_a[best + 1], QMD_SZ);
        return 1;
    }
    return 0;
}

/* Load a cubin, return CUfunction */
static CUfunction load_func(const char *cubin_path, const char *kname, CUmodule *mod_out) {
    FILE *f = fopen(cubin_path, "rb");
    if (!f) { perror(cubin_path); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *data = malloc(sz); fread(data, 1, sz, f); fclose(f);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, data));
    CUfunction func; CHECK(cuModuleGetFunction(&func, mod, kname));
    free(data);
    *mod_out = mod;
    return func;
}

/* Get register count via driver API */
static int reg_count(CUfunction f) {
    int n = 0;
    cuFuncGetAttribute(&n, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    return n;
}

/* Launch a kernel and capture its QMD */
static int capture_qmd(CUfunction f, unsigned smem, void **params, uint32_t *qmd_out) {
    uint8_t *snap1 = malloc(RING_SIZE);
    uint8_t *snap2 = malloc(RING_SIZE);

    scopy(snap1, (const void *)RING_BASE, RING_SIZE);
    CHECK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, smem, NULL, params, NULL));
    cuCtxSynchronize();
    scopy(snap2, (const void *)RING_BASE, RING_SIZE);

    int ok = extract_qmd(snap1, snap2, RING_SIZE, qmd_out);
    free(snap1); free(snap2);
    return ok;
}

static void diff_report(const char *label, const uint32_t *a, const uint32_t *b,
                         const char *da, const char *db) {
    fprintf(stdout, "\n======== %s ========\n", label);
    fprintf(stdout, "  A: %s\n  B: %s\n", da, db);
    int nd = 0;
    for (int i = 0; i < QMD_DW; i++) {
        if (a[i] != b[i]) {
            int64_t d = (int64_t)b[i] - (int64_t)a[i];
            fprintf(stdout, "  dw[%3d] 0x%04x: A=0x%08x (%u)  B=0x%08x (%u)  delta=%ld\n",
                    i, i*4, a[i], a[i], b[i], b[i], (long)d);
            nd++;
        }
    }
    fprintf(stdout, "  Total diffs: %d\n", nd);
}

int main(void) {
    fprintf(stdout, "=== QMD Differential Probe (driver API) ===\n\n");

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod_a, mod_b, mod_c, mod_s;
    CUfunction f_a = load_func("/tmp/probe_kern_a.cubin", "kern_a", &mod_a);
    CUfunction f_b = load_func("/tmp/probe_kern_b.cubin", "kern_b", &mod_b);
    CUfunction f_c = load_func("/tmp/probe_kern_c.cubin", "kern_c", &mod_c);
    CUfunction f_s = load_func("/tmp/probe_kern_smem.cubin", "kern_smem", &mod_s);

    int r_a = reg_count(f_a);
    int r_b = reg_count(f_b);
    int r_c = reg_count(f_c);
    int r_s = reg_count(f_s);
    fprintf(stdout, "Register counts:\n");
    fprintf(stdout, "  kern_a:    %d regs\n", r_a);
    fprintf(stdout, "  kern_b:    %d regs\n", r_b);
    fprintf(stdout, "  kern_c:    %d regs\n", r_c);
    fprintf(stdout, "  kern_smem: %d regs\n", r_s);

    /* Query entry PC (function GPU VA) */
    CUdeviceptr ent_a = 0, ent_b = 0;
    /* Note: driver API doesn't directly expose entry_pc; we infer it from QMD */

    /* Allocate device memory for params */
    CUdeviceptr d_out; CHECK(cuMemAlloc(&d_out, 4096));
    void *params[] = { &d_out };

    /* Warmup (one launch to establish ring) */
    CHECK(cuLaunchKernel(f_a, 1,1,1, 1,1,1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    /* ---- EXP0: baseline noise (same kernel twice) ---- */
    uint32_t q_a1[QMD_DW] = {0}, q_a2[QMD_DW] = {0};
    int ok_a1 = capture_qmd(f_a, 0, params, q_a1);
    int ok_a2 = capture_qmd(f_a, 0, params, q_a2);
    fprintf(stdout, "\n[EXP0] baseline (kern_a x 2): %s %s\n",
            ok_a1?"OK":"FAIL", ok_a2?"OK":"FAIL");

    /* Identify the "noise" dwords that change between identical launches */
    int noise[QMD_DW] = {0};
    if (ok_a1 && ok_a2) {
        diff_report("EXP0 baseline (kern_a vs kern_a)",
                    q_a1, q_a2, "kern_a (first)", "kern_a (second)");
        for (int i = 0; i < QMD_DW; i++) if (q_a1[i] != q_a2[i]) noise[i] = 1;
    }

    /* ---- EXP1: entry_pc + register_count ---- */
    uint32_t q_a[QMD_DW] = {0}, q_b[QMD_DW] = {0};
    int ok_a = capture_qmd(f_a, 0, params, q_a);
    int ok_b = capture_qmd(f_b, 0, params, q_b);
    fprintf(stdout, "\n[EXP1] capture kern_a: %s  kern_b: %s\n",
            ok_a?"OK":"FAIL", ok_b?"OK":"FAIL");
    if (ok_a && ok_b) {
        diff_report("EXP1: kern_a vs kern_b (entry_pc + reg_count)",
                    q_a, q_b,
                    "kern_a (8 regs)", "kern_b (30 regs)");

        fprintf(stdout, "\nEXP1 diffs NOT in baseline noise set:\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_a[i] != q_b[i] && !noise[i]) {
                fprintf(stdout, "  dw[%3d] 0x%04x: 0x%08x -> 0x%08x  (SIGNAL: not noise)\n",
                        i, i*4, q_a[i], q_b[i]);
            }
        }
    }

    /* ---- EXP1b: kern_c (70 regs) — triple-point for register_count ---- */
    uint32_t q_c[QMD_DW] = {0};
    int ok_c = capture_qmd(f_c, 0, params, q_c);
    fprintf(stdout, "\n[EXP1b] capture kern_c (70 regs): %s\n", ok_c?"OK":"FAIL");
    if (ok_a && ok_c) {
        diff_report("EXP1b: kern_a vs kern_c", q_a, q_c,
                    "kern_a (8 regs)", "kern_c (70 regs)");
        fprintf(stdout, "\nEXP1b diffs NOT in baseline noise:\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_a[i] != q_c[i] && !noise[i]) {
                fprintf(stdout, "  dw[%3d] 0x%04x: 0x%08x -> 0x%08x\n",
                        i, i*4, q_a[i], q_c[i]);
            }
        }
    }
    if (ok_b && ok_c) {
        fprintf(stdout, "\n  kern_b vs kern_c non-noise diffs:\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_b[i] != q_c[i] && !noise[i]) {
                fprintf(stdout, "  dw[%3d] 0x%04x: 0x%08x -> 0x%08x\n",
                        i, i*4, q_b[i], q_c[i]);
            }
        }
    }

    /* ---- EXP2: shared_mem_size (0 vs 4096) ---- */
    uint32_t q_s0[QMD_DW] = {0}, q_s4[QMD_DW] = {0}, q_s8[QMD_DW] = {0};
    int ok_s0 = capture_qmd(f_s, 0, params, q_s0);
    int ok_s4 = capture_qmd(f_s, 4096, params, q_s4);
    int ok_s8 = capture_qmd(f_s, 8192, params, q_s8);
    fprintf(stdout, "\n[EXP2] smem=0: %s  smem=4096: %s  smem=8192: %s\n",
            ok_s0?"OK":"FAIL", ok_s4?"OK":"FAIL", ok_s8?"OK":"FAIL");
    if (ok_s0 && ok_s4) {
        diff_report("EXP2: smem=0 vs smem=4096",
                    q_s0, q_s4, "dynamic smem 0", "dynamic smem 4096");
    }
    if (ok_s0 && ok_s8) {
        diff_report("EXP3: smem=0 vs smem=8192",
                    q_s0, q_s8, "dynamic smem 0", "dynamic smem 8192");
    }

    /* ---- Dumps ---- */
    fprintf(stdout, "\n--- Full QMD: kern_a (8 regs) ---\n");
    for (int i = 0; i < QMD_DW; i++)
        if (q_a[i]) fprintf(stdout, "  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4, q_a[i], q_a[i]);
    fprintf(stdout, "\n--- Full QMD: kern_b (30 regs) ---\n");
    for (int i = 0; i < QMD_DW; i++)
        if (q_b[i]) fprintf(stdout, "  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4, q_b[i], q_b[i]);
    fprintf(stdout, "\n--- Full QMD: kern_c (70 regs) ---\n");
    for (int i = 0; i < QMD_DW; i++)
        if (q_c[i]) fprintf(stdout, "  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4, q_c[i], q_c[i]);
    fprintf(stdout, "\n--- Full QMD: kern_smem (smem=0) ---\n");
    for (int i = 0; i < QMD_DW; i++)
        if (q_s0[i]) fprintf(stdout, "  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4, q_s0[i], q_s0[i]);
    fprintf(stdout, "\n--- Full QMD: kern_smem (smem=4096) ---\n");
    for (int i = 0; i < QMD_DW; i++)
        if (q_s4[i]) fprintf(stdout, "  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4, q_s4[i], q_s4[i]);

    /* ---- Analysis ---- */
    fprintf(stdout, "\n\n######## FINAL ANALYSIS ########\n");

    /* Filter out known-variable fields: session/allocation addresses */
    /* These change every launch regardless: dw[48] (0xc0), dw[70] (0x118),
     * dw[102] (0x198), dw[104] (0x1a0), and often dw[12] (0x30).
     * They're QMD buffer VAs not entry_pc. */

    if (ok_a && ok_b) {
        fprintf(stdout, "\nEntry PC candidates (large delta, not a session addr):\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_a[i] == q_b[i]) continue;
            int64_t d = (int64_t)q_b[i] - (int64_t)q_a[i];
            fprintf(stdout, "  dw[%3d] 0x%04x: A=0x%08x B=0x%08x delta=%ld",
                    i, i*4, q_a[i], q_b[i], (long)d);
            /* Check if adjacent dword could be the hi32 of a 64-bit GPU VA */
            if (i + 1 < QMD_DW && q_a[i+1] == q_b[i+1] && q_a[i+1] > 0 && q_a[i+1] < 0x100) {
                uint64_t va_a = ((uint64_t)q_a[i+1] << 32) | q_a[i];
                uint64_t va_b = ((uint64_t)q_b[i+1] << 32) | q_b[i];
                fprintf(stdout, "  [possibly 64-bit VA: A=0x%lx B=0x%lx]",
                        va_a, va_b);
            }
            fprintf(stdout, "\n");
        }

        fprintf(stdout, "\nRegister count candidates (match %d / %d):\n", r_a, r_b);
        for (int i = 0; i < QMD_DW; i++) {
            if (q_a[i] == q_b[i]) continue;
            uint32_t va = q_a[i], vb = q_b[i];
            /* Check various byte positions */
            for (int byte = 0; byte < 4; byte++) {
                uint32_t fa = (va >> (byte*8)) & 0xFF;
                uint32_t fb = (vb >> (byte*8)) & 0xFF;
                if (fa == (uint32_t)r_a && fb == (uint32_t)r_b) {
                    fprintf(stdout, "  dw[%3d] 0x%04x byte[%d]: %d -> %d  <-- MATCHES register count\n",
                            i, i*4, byte, fa, fb);
                }
            }
            /* Also check bitfield match at other positions */
            /* Register count is often encoded as ceil(regs/8) or similar */
            int ra_quant = (r_a + 7) & ~7;
            int rb_quant = (r_b + 7) & ~7;
            for (int byte = 0; byte < 4; byte++) {
                uint32_t fa = (va >> (byte*8)) & 0xFF;
                uint32_t fb = (vb >> (byte*8)) & 0xFF;
                if (fa == (uint32_t)ra_quant && fb == (uint32_t)rb_quant && ra_quant != r_a) {
                    fprintf(stdout, "  dw[%3d] 0x%04x byte[%d]: %d -> %d  <-- MATCHES rounded regcount (%d / %d)\n",
                            i, i*4, byte, fa, fb, ra_quant, rb_quant);
                }
            }
        }
    }

    if (ok_s0 && ok_s4) {
        fprintf(stdout, "\nShared memory size candidates (noise-filtered):\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_s0[i] == q_s4[i]) continue;
            if (noise[i]) continue;  /* filter session noise */
            int64_t d4 = (int64_t)q_s4[i] - (int64_t)q_s0[i];
            fprintf(stdout, "  dw[%3d] 0x%04x: s0=0x%08x s4k=0x%08x delta=%ld",
                    i, i*4, q_s0[i], q_s4[i], (long)d4);
            if (ok_s8) {
                int64_t d8 = (int64_t)q_s8[i] - (int64_t)q_s0[i];
                fprintf(stdout, "  s8k=0x%08x delta8k=%ld", q_s8[i], (long)d8);
                if (d4 == 4096 && d8 == 8192)
                    fprintf(stdout, "  *** CONFIRMED shared_mem_size (linear delta) ***");
                if (q_s0[i] == 0 && q_s4[i] == 4096 && q_s8[i] == 8192)
                    fprintf(stdout, "  *** CONFIRMED shared_mem_size (raw) ***");
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\nAll shared memory diffs (incl noise, for reference):\n");
        for (int i = 0; i < QMD_DW; i++) {
            if (q_s0[i] == q_s4[i]) continue;
            fprintf(stdout, "  dw[%3d] 0x%04x: s0=0x%08x s4k=0x%08x%s",
                    i, i*4, q_s0[i], q_s4[i], noise[i]?"  [NOISE]":"");
            if (ok_s8) fprintf(stdout, "  s8k=0x%08x", q_s8[i]);
            fprintf(stdout, "\n");
        }
    }

    /* Save raw QMDs */
    FILE *out = fopen("/tmp/qmd_probe_final.bin", "wb");
    if (out) {
        fwrite(q_a,  4, QMD_DW, out);
        fwrite(q_b,  4, QMD_DW, out);
        fwrite(q_s0, 4, QMD_DW, out);
        fwrite(q_s4, 4, QMD_DW, out);
        fwrite(q_s8, 4, QMD_DW, out);
        fclose(out);
    }

    cuMemFree(d_out);
    cuModuleUnload(mod_a);
    cuModuleUnload(mod_b);
    cuModuleUnload(mod_c);
    cuModuleUnload(mod_s);
    cuCtxDestroy(ctx);
    return 0;
}
