/*
 * cbuf0_probe.c — Find register_count and other fields in cbuf0 (constant buffer 0).
 *
 * The QMD probe (docs/qmd_fields.md) confirmed register_count is NOT in the
 * 528-byte QMD body on sm_90. The CUDA driver stores it in cbuf0.
 *
 * Approach:
 *   1. Build test cubins with nvcc using --maxrregcount to force different register
 *      allocations (volatile arrays go to local memory, not registers).
 *   2. Capture QMDs from pushbuffer ring.
 *   3. Identify cbuf0 base VA in QMD.
 *   4. Use cuMemcpyDtoH to read cbuf0 contents (it's device-accessible memory).
 *   5. Diff across kernels to find register_count.
 *
 * Build:
 *   gcc -O0 -g -o /home/ubuntu/lithos/tools/cbuf0_probe \
 *       /home/ubuntu/lithos/tools/cbuf0_probe.c \
 *       -I/usr/local/cuda/include -lcuda -ldl -lpthread
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

static sigjmp_buf jmp_env;
static volatile int in_scopy = 0;
static void segv(int s) { if (in_scopy) siglongjmp(jmp_env, 1); _exit(139); }
static int scopy(void *dst, const void *src, size_t n) {
    struct sigaction sa = {0}, o_sv, o_bus;
    sa.sa_handler = segv; sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o_sv); sigaction(SIGBUS, &sa, &o_bus);
    in_scopy = 1;
    int ok = (sigsetjmp(jmp_env, 1) == 0);
    if (ok) memcpy(dst, src, n);
    in_scopy = 0;
    sigaction(SIGSEGV, &o_sv, NULL); sigaction(SIGBUS, &o_bus, NULL);
    return ok;
}

#define RING_BASE 0x200800000UL
#define RING_SIZE 0x400000UL
#define QMD_DW 132
#define QMD_SZ (QMD_DW * 4)

static int extract_qmd(const uint8_t *before, const uint8_t *after,
                       size_t sz, uint32_t *qmd_out) {
    const uint32_t *dw_a = (const uint32_t *)after;
    int ndw = (int)(sz / 4);
    int first = -1, last = -1;
    for (size_t k = 0; k < sz; k++) {
        if (before[k] != after[k]) {
            if (first < 0) first = (int)k;
            last = (int)k;
        }
    }
    if (first < 0) return 0;
    int start_dw = (first & ~3) / 4;
    int end_dw = ((last + 4) & ~3) / 4;
    if (start_dw < 0) start_dw = 0;
    if (end_dw > ndw - QMD_DW - 1) end_dw = ndw - QMD_DW - 1;
    int best = -1;
    for (int i = start_dw; i < end_dw; i++) {
        uint32_t hdr = dw_a[i];
        int opcode = (hdr >> 29) & 7;
        int count  = (hdr >> 16) & 0x1FFF;
        if (opcode == 3 && count == 132) best = i;
    }
    if (best >= 0) {
        memcpy(qmd_out, &dw_a[best + 1], QMD_SZ);
        return 1;
    }
    return 0;
}

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

static CUfunction load_cubin_func(const char *path, const char *name, CUmodule *mod) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *data = malloc(sz); fread(data, 1, sz, f); fclose(f);
    CHECK(cuModuleLoadData(mod, data));
    CUfunction func; CHECK(cuModuleGetFunction(&func, *mod, name));
    free(data);
    return func;
}

static int get_regs(CUfunction f) {
    int n = 0; cuFuncGetAttribute(&n, CU_FUNC_ATTRIBUTE_NUM_REGS, f); return n;
}

static void build_cubin_maxreg(const char *src, const char *cubin_path, int maxreg) {
    char src_path[256];
    snprintf(src_path, sizeof(src_path), "%s.cu", cubin_path);
    FILE *f = fopen(src_path, "w");
    fprintf(f, "%s", src);
    fclose(f);
    char cmd[1024];
    if (maxreg > 0) {
        snprintf(cmd, sizeof(cmd),
                 "nvcc -cubin -arch=sm_90a --maxrregcount=%d -o %s %s 2>&1",
                 maxreg, cubin_path, src_path);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "nvcc -cubin -arch=sm_90a -o %s %s 2>&1", cubin_path, src_path);
    }
    printf("  %s\n", cmd);
    if (system(cmd) != 0) { fprintf(stderr, "nvcc failed\n"); exit(1); }
}

int main(void) {
    printf("=== cbuf0 Probe: Finding register_count in constant buffer 0 ===\n\n");

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    printf("GPU: sm_%d%d\n\n", major, minor);

    /* Build kernels. Use a kernel with enough register-resident computation
     * that --maxrregcount will produce different actual counts.
     * Key: the computation must use many LIVE values simultaneously. */

    /* This kernel keeps 'n' float values live simultaneously via a reduction tree */
    const char *heavy_src =
        "extern \"C\" __global__ void kern(float *out) {\n"
        "    float v0 = out[0], v1 = out[1], v2 = out[2], v3 = out[3];\n"
        "    float v4 = out[4], v5 = out[5], v6 = out[6], v7 = out[7];\n"
        "    float v8 = out[8], v9 = out[9], v10 = out[10], v11 = out[11];\n"
        "    float v12 = out[12], v13 = out[13], v14 = out[14], v15 = out[15];\n"
        "    float v16 = out[16], v17 = out[17], v18 = out[18], v19 = out[19];\n"
        "    float v20 = out[20], v21 = out[21], v22 = out[22], v23 = out[23];\n"
        "    float v24 = out[24], v25 = out[25], v26 = out[26], v27 = out[27];\n"
        "    float v28 = out[28], v29 = out[29], v30 = out[30], v31 = out[31];\n"
        "    v0 = v0*v1 + v2*v3; v4 = v4*v5 + v6*v7;\n"
        "    v8 = v8*v9 + v10*v11; v12 = v12*v13 + v14*v15;\n"
        "    v16 = v16*v17 + v18*v19; v20 = v20*v21 + v22*v23;\n"
        "    v24 = v24*v25 + v26*v27; v28 = v28*v29 + v30*v31;\n"
        "    v0 = v0 + v4 + v8 + v12 + v16 + v20 + v24 + v28;\n"
        "    out[0] = v0;\n"
        "}\n";

    printf("Building test kernels with --maxrregcount...\n");
    build_cubin_maxreg(heavy_src, "/tmp/cbuf0_k16.cubin",  16);
    build_cubin_maxreg(heavy_src, "/tmp/cbuf0_k32.cubin",  32);
    build_cubin_maxreg(heavy_src, "/tmp/cbuf0_k64.cubin",  64);
    build_cubin_maxreg(heavy_src, "/tmp/cbuf0_k128.cubin", 128);
    build_cubin_maxreg(heavy_src, "/tmp/cbuf0_k255.cubin", 255);

    CUmodule m16, m32, m64, m128, m255;
    CUfunction f16  = load_cubin_func("/tmp/cbuf0_k16.cubin",  "kern", &m16);
    CUfunction f32  = load_cubin_func("/tmp/cbuf0_k32.cubin",  "kern", &m32);
    CUfunction f64  = load_cubin_func("/tmp/cbuf0_k64.cubin",  "kern", &m64);
    CUfunction f128 = load_cubin_func("/tmp/cbuf0_k128.cubin", "kern", &m128);
    CUfunction f255 = load_cubin_func("/tmp/cbuf0_k255.cubin", "kern", &m255);

    int r16  = get_regs(f16);
    int r32  = get_regs(f32);
    int r64  = get_regs(f64);
    int r128 = get_regs(f128);
    int r255 = get_regs(f255);

    printf("\nRegister counts (cuFuncGetAttribute):\n");
    printf("  maxreg=16:  %d regs\n",  r16);
    printf("  maxreg=32:  %d regs\n",  r32);
    printf("  maxreg=64:  %d regs\n",  r64);
    printf("  maxreg=128: %d regs\n", r128);
    printf("  maxreg=255: %d regs\n", r255);

    /* Pick the ones that differ */
    struct {
        const char *name;
        CUfunction func;
        int regs;
        uint32_t qmd[QMD_DW];
    } kerns[5];
    int nk = 0;

    struct { const char *name; CUfunction func; int regs; } all[] = {
        { "k16",  f16,  r16  },
        { "k32",  f32,  r32  },
        { "k64",  f64,  r64  },
        { "k128", f128, r128 },
        { "k255", f255, r255 },
    };

    /* Add all with unique register counts */
    for (int i = 0; i < 5; i++) {
        int dup = 0;
        for (int j = 0; j < nk; j++) {
            if (kerns[j].regs == all[i].regs) { dup = 1; break; }
        }
        if (!dup) {
            kerns[nk].name = all[i].name;
            kerns[nk].func = all[i].func;
            kerns[nk].regs = all[i].regs;
            memset(kerns[nk].qmd, 0, QMD_SZ);
            nk++;
        }
    }

    printf("\n%d kernels with distinct register counts selected.\n", nk);

    if (nk < 2) {
        printf("ERROR: Need at least 2 different register counts to diff. Aborting.\n");
        return 1;
    }

    /* Device memory */
    CUdeviceptr d_out;
    CHECK(cuMemAlloc(&d_out, 4096));
    void *params[] = { &d_out };

    /* Warmup */
    CHECK(cuLaunchKernel(kerns[0].func, 1,1,1, 1,1,1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    /* Capture QMDs */
    printf("\nCapturing QMDs...\n");
    for (int k = 0; k < nk; k++) {
        int ok = capture_qmd(kerns[k].func, 0, params, kerns[k].qmd);
        printf("  %s (%d regs): %s\n", kerns[k].name, kerns[k].regs, ok ? "OK" : "FAIL");
    }

    /* Print QMD for first kernel */
    printf("\nFull QMD (%s, %d regs) — non-zero:\n", kerns[0].name, kerns[0].regs);
    for (int i = 0; i < QMD_DW; i++) {
        if (kerns[0].qmd[i] != 0)
            printf("  dw[%3d] 0x%04x = 0x%08x (%u)\n", i, i*4,
                   kerns[0].qmd[i], kerns[0].qmd[i]);
    }

    /* Establish noise */
    uint32_t q_noise[QMD_DW];
    capture_qmd(kerns[0].func, 0, params, q_noise);
    int noise[QMD_DW] = {0};
    for (int i = 0; i < QMD_DW; i++)
        if (kerns[0].qmd[i] != q_noise[i]) noise[i] = 1;

    /* QMD diffs */
    printf("\nQMD diffs (non-noise):\n");
    for (int i = 0; i < QMD_DW; i++) {
        if (noise[i]) continue;
        int differs = 0;
        for (int k = 1; k < nk; k++)
            if (kerns[k].qmd[i] != kerns[0].qmd[i]) { differs = 1; break; }
        if (!differs) continue;
        printf("  dw[%3d] 0x%04x:", i, i*4);
        for (int k = 0; k < nk; k++)
            printf("  %s(%d)=0x%08x", kerns[k].name, kerns[k].regs, kerns[k].qmd[i]);
        printf("\n");
    }

    /* Now find GPU VA candidates in the QMD for cbuf0.
     * Key QMD fields from previous run:
     *   dw[48:49] @ 0xC0 = per-launch VA (varies per kernel AND per launch)
     *   dw[50:51] @ 0xC8 = constant VA (same across all kernels: 0x3_24010000)
     *   dw[70:71] @ 0x118 = entry_pc (varies per kernel)
     *
     * cbuf0 is likely at dw[48:49] since it's per-kernel. But it's also in the
     * noise set (changes per launch). Let's look at dw[50:51] too — it might be
     * the constant buffer BASE that the driver pre-populates.
     *
     * Actually on Hopper, cbuf0 address is SET in the QMD by the driver. Looking
     * at the pushbuffer methods, there's a pattern with method 0x2062 and 0x2060
     * that sets up the constant buffer BEFORE the QMD is submitted. */

    printf("\n\n######## PROBING CANDIDATE GPU VAs ########\n\n");

    /* From the pushbuffer scan, we saw:
     *   [-8] method 0x2062: data 0x00000003, 0x26290000  -> this is the VA at dw[48:49]!
     *   [-5] method 0x2060: data 0x00000210, 0x00000001  -> size=0x210, index=1 (cbuf0?)
     *   Earlier: method 0x2062: 0x00000003, 0x26280210 -> another VA
     *            method 0x2060: 0x00000008, 0x00000001 -> size=8, index=1
     *   And two large (96-dword) loads at method 0x206d
     *
     * Method 0x2062 sets a VA, 0x2060 sets size + index.
     * The 96-dword load at 0x206d is likely writing cbuf0 contents!
     * That's 384 bytes of constant buffer data loaded inline.
     *
     * So cbuf0 is loaded inline into the pushbuffer, not via a GPU VA read.
     * We need to extract those 96-dword payloads and diff them! */

    printf("Strategy: cbuf0 is loaded via inline pushbuffer methods (0x206d),\n");
    printf("not via a GPU VA. Extracting inline constant buffer data...\n\n");

    /* Re-capture with full ring for each kernel to get the inline data */
    for (int ki = 0; ki < nk; ki++) {
        uint8_t *snap1 = malloc(RING_SIZE);
        uint8_t *snap2 = malloc(RING_SIZE);
        scopy(snap1, (const void *)RING_BASE, RING_SIZE);
        CHECK(cuLaunchKernel(kerns[ki].func, 1,1,1, 1,1,1, 0, NULL, params, NULL));
        cuCtxSynchronize();
        scopy(snap2, (const void *)RING_BASE, RING_SIZE);

        /* Find QMD position */
        const uint32_t *ring1 = (const uint32_t *)snap1;
        const uint32_t *ring2 = (const uint32_t *)snap2;
        int ndw = RING_SIZE / 4;
        int first = -1, last = -1;
        for (int i = 0; i < ndw; i++) {
            if (ring1[i] != ring2[i]) {
                if (first < 0) first = i;
                last = i;
            }
        }

        int qmd_pos = -1;
        for (int i = first; i <= last; i++) {
            uint32_t hdr = ring2[i];
            if (((hdr >> 29) & 7) == 3 && ((hdr >> 16) & 0x1FFF) == 132)
                qmd_pos = i;
        }

        if (qmd_pos < 0) {
            printf("  %s: QMD not found in ring diff\n", kerns[ki].name);
            free(snap1); free(snap2);
            continue;
        }

        printf("=== %s (%d regs) — pushbuffer before QMD (pos %d) ===\n",
               kerns[ki].name, kerns[ki].regs, qmd_pos);

        /* Scan backwards for method 0x206d (NONINC data loads) and 0x2062/0x2060 */
        int scan_start = qmd_pos - 400;
        if (scan_start < 0) scan_start = 0;

        /* Track constant buffer loads */
        int cb_load_count = 0;

        for (int i = scan_start; i < qmd_pos; ) {
            uint32_t w = ring2[i];
            int opcode = (w >> 29) & 7;
            int count = (w >> 16) & 0x1FFF;
            int method = w & 0xFFFF;

            if (opcode < 1 || opcode > 4 || count < 1 || count > 200) { i++; continue; }

            if (method == 0x2062 || method == 0x2060 || method == 0x206c ||
                method == 0x206d || method == 0x20ad || method == 0x20b0 ||
                method == 0x26c0) {

                printf("  [%+4d] op=%d cnt=%3d meth=0x%04x", i - qmd_pos, opcode, count, method);

                if (method == 0x2062) printf(" (CB_ADDR_HI+LO)");
                if (method == 0x2060) printf(" (CB_SIZE+INDEX)");
                if (method == 0x206c) printf(" (CB_BIND)");
                if (method == 0x206d) printf(" (CB_LOAD_INLINE_DATA)");
                if (method == 0x20ad) printf(" (INVALIDATE_CONSTANT)");
                if (method == 0x20b0) printf(" (SET_SHADER_SHARED_MEMORY)");
                if (method == 0x26c0) printf(" (SEND_PCAS?)");

                if (count <= 8) {
                    printf("  data:");
                    for (int j = 1; j <= count && i+j < ndw; j++)
                        printf(" 0x%08x", ring2[i + j]);
                }
                printf("\n");

                /* If this is a CB_LOAD_INLINE_DATA (0x206d), dump the full payload */
                if (method == 0x206d && count > 8) {
                    printf("    CB_LOAD_INLINE_DATA payload (%d dwords = %d bytes):\n",
                           count, count * 4);
                    for (int j = 1; j <= count && i+j < ndw; j++) {
                        printf("    [0x%03x] dw[%3d] = 0x%08x (%u)\n",
                               (j-1)*4, j-1, ring2[i + j], ring2[i + j]);
                    }
                    cb_load_count++;
                }
            }

            i += 1 + count;
        }

        printf("\n");
        free(snap1); free(snap2);
    }

    /* Now the real analysis: extract the inline CB data for each kernel and diff.
     * We need to specifically capture the 96-dword inline loads and compare them. */

    printf("\n######## EXTRACTING AND DIFFING INLINE CB DATA ########\n\n");

    /* For each kernel, extract ALL inline CB loads (method 0x206d) from the
     * pushbuffer region before the QMD */
    #define MAX_CB_LOADS 8
    #define MAX_CB_DW 256

    struct cb_load {
        uint64_t target_va;  /* from preceding 0x2062 method */
        int size_bytes;      /* from preceding 0x2060 method */
        int cb_index;        /* from preceding 0x2060 method */
        int n_dwords;
        uint32_t data[MAX_CB_DW];
    };

    struct kern_cb_data {
        int n_loads;
        struct cb_load loads[MAX_CB_LOADS];
    } kern_cbs[5];

    for (int ki = 0; ki < nk; ki++) {
        kern_cbs[ki].n_loads = 0;

        uint8_t *snap1 = malloc(RING_SIZE);
        uint8_t *snap2 = malloc(RING_SIZE);
        scopy(snap1, (const void *)RING_BASE, RING_SIZE);
        CHECK(cuLaunchKernel(kerns[ki].func, 1,1,1, 1,1,1, 0, NULL, params, NULL));
        cuCtxSynchronize();
        scopy(snap2, (const void *)RING_BASE, RING_SIZE);

        const uint32_t *ring1 = (const uint32_t *)snap1;
        const uint32_t *ring2 = (const uint32_t *)snap2;
        int ndw = RING_SIZE / 4;

        /* Find diff bounds and QMD position */
        int first = -1, last = -1;
        for (int i = 0; i < ndw; i++) {
            if (ring1[i] != ring2[i]) {
                if (first < 0) first = i;
                last = i;
            }
        }
        int qmd_pos = -1;
        for (int i = first; i <= last; i++) {
            uint32_t hdr = ring2[i];
            if (((hdr >> 29) & 7) == 3 && ((hdr >> 16) & 0x1FFF) == 132)
                qmd_pos = i;
        }

        if (qmd_pos < 0) { free(snap1); free(snap2); continue; }

        /* Parse methods before QMD */
        int scan_start = qmd_pos - 400;
        if (scan_start < 0) scan_start = 0;

        uint64_t cur_va = 0;
        int cur_size = 0, cur_index = 0;

        for (int i = scan_start; i < qmd_pos; ) {
            uint32_t w = ring2[i];
            int opcode = (w >> 29) & 7;
            int count = (w >> 16) & 0x1FFF;
            int method = w & 0xFFFF;

            if (opcode < 1 || opcode > 4 || count < 1 || count > 200) { i++; continue; }

            if (method == 0x2062 && count >= 2 && i + 2 < ndw) {
                /* CB address: hi, lo */
                uint32_t hi = ring2[i + 1];
                uint32_t lo = ring2[i + 2];
                cur_va = ((uint64_t)hi << 32) | lo;
            }
            if (method == 0x2060 && count >= 2 && i + 2 < ndw) {
                /* CB size and index */
                cur_size = ring2[i + 1];
                cur_index = ring2[i + 2];
            }
            if (method == 0x206d && count > 0) {
                /* Inline data load */
                int li = kern_cbs[ki].n_loads;
                if (li < MAX_CB_LOADS) {
                    kern_cbs[ki].loads[li].target_va = cur_va;
                    kern_cbs[ki].loads[li].size_bytes = cur_size;
                    kern_cbs[ki].loads[li].cb_index = cur_index;
                    kern_cbs[ki].loads[li].n_dwords = count < MAX_CB_DW ? count : MAX_CB_DW;
                    for (int j = 0; j < kern_cbs[ki].loads[li].n_dwords; j++)
                        kern_cbs[ki].loads[li].data[j] = ring2[i + 1 + j];
                    kern_cbs[ki].n_loads++;
                }
            }

            i += 1 + count;
        }

        free(snap1); free(snap2);

        printf("%s (%d regs): %d CB loads found\n", kerns[ki].name, kerns[ki].regs,
               kern_cbs[ki].n_loads);
        for (int l = 0; l < kern_cbs[ki].n_loads; l++) {
            struct cb_load *cb = &kern_cbs[ki].loads[l];
            printf("  load[%d]: VA=0x%016lx size=%d index=%d dwords=%d\n",
                   l, cb->target_va, cb->size_bytes, cb->cb_index, cb->n_dwords);
        }
    }

    /* Now diff corresponding CB loads across kernels */
    printf("\n######## CB LOAD DIFFS ########\n\n");

    /* Match loads by their index within the sequence (assume same ordering) */
    int max_loads = kern_cbs[0].n_loads;
    for (int l = 0; l < max_loads; l++) {
        /* Check if all kernels have this load */
        int all_have = 1;
        for (int k = 0; k < nk; k++) {
            if (l >= kern_cbs[k].n_loads) { all_have = 0; break; }
        }
        if (!all_have) continue;

        struct cb_load *ref = &kern_cbs[0].loads[l];
        int min_dw = ref->n_dwords;
        for (int k = 1; k < nk; k++) {
            if (kern_cbs[k].loads[l].n_dwords < min_dw)
                min_dw = kern_cbs[k].loads[l].n_dwords;
        }

        /* Count diffs */
        int n_diffs = 0;
        for (int d = 0; d < min_dw; d++) {
            int differs = 0;
            for (int k = 1; k < nk; k++) {
                if (kern_cbs[k].loads[l].data[d] != ref->data[d]) { differs = 1; break; }
            }
            if (differs) n_diffs++;
        }

        printf("CB load[%d]: size=%d index=%d, %d diffs in %d dwords\n",
               l, ref->size_bytes, ref->cb_index, n_diffs, min_dw);

        if (n_diffs == 0) {
            printf("  (identical across all kernels — not cbuf0 metadata)\n\n");
            continue;
        }

        /* Show all diffs */
        for (int d = 0; d < min_dw; d++) {
            int differs = 0;
            for (int k = 1; k < nk; k++) {
                if (kern_cbs[k].loads[l].data[d] != ref->data[d]) { differs = 1; break; }
            }
            if (!differs) continue;

            printf("  [0x%03x] dw[%3d]:", d*4, d);
            for (int k = 0; k < nk; k++) {
                uint32_t v = kern_cbs[k].loads[l].data[d];
                printf("  %s(%d)=0x%08x(%u)", kerns[k].name, kerns[k].regs, v, v);
            }

            /* Check for register count matches */
            int exact = 1, rnd2 = 1, rnd4 = 1, rnd8 = 1;
            for (int k = 0; k < nk; k++) {
                uint32_t v = kern_cbs[k].loads[l].data[d];
                int r = kerns[k].regs;
                if (v != (uint32_t)r) exact = 0;
                if (v != (uint32_t)((r+1)&~1)) rnd2 = 0;
                if (v != (uint32_t)((r+3)&~3)) rnd4 = 0;
                if (v != (uint32_t)((r+7)&~7)) rnd8 = 0;
            }
            if (exact) printf("  *** REGISTER_COUNT (exact) ***");
            if (rnd2)  printf("  *** REGISTER_COUNT (round2) ***");
            if (rnd4)  printf("  *** REGISTER_COUNT (round4) ***");
            if (rnd8)  printf("  *** REGISTER_COUNT (round8) ***");

            /* Check bitfield encodings */
            for (int bp = 0; bp < 4; bp++) {
                int byte_match = 1;
                for (int k = 0; k < nk; k++) {
                    uint32_t v = (kern_cbs[k].loads[l].data[d] >> (bp*8)) & 0xFF;
                    if (v != (uint32_t)kerns[k].regs) { byte_match = 0; break; }
                }
                if (byte_match)
                    printf("  *** REG_COUNT byte[%d] ***", bp);
            }

            /* Check monotonic */
            int mono = 1;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].regs > kerns[k-1].regs &&
                    kern_cbs[k].loads[l].data[d] <= kern_cbs[k-1].loads[l].data[d]) mono = 0;
            }
            if (mono) printf("  (monotonic)");

            printf("\n");
        }

        /* Full dump of this CB load for all kernels */
        printf("\n  Full CB load[%d] dump:\n", l);
        for (int d = 0; d < min_dw; d++) {
            int differs = 0;
            for (int k = 1; k < nk; k++) {
                if (kern_cbs[k].loads[l].data[d] != ref->data[d]) { differs = 1; break; }
            }
            printf("  [0x%03x] %s", d*4, differs ? "*** " : "    ");
            for (int k = 0; k < nk; k++) {
                if (k > 0 && kern_cbs[k].loads[l].data[d] == ref->data[d])
                    printf("  (same)        ");
                else
                    printf("  0x%08x(%3u)", kern_cbs[k].loads[l].data[d],
                           kern_cbs[k].loads[l].data[d]);
            }
            printf("\n");
        }

        printf("\n");
    }

    /* Also try reading via cuMemcpyDtoH at the VA from dw[50:51] (0xC8)
     * which was the same across all kernels: 0x3_24010000.
     * This might be a shared constant buffer template. */
    printf("\n######## TRYING cuMemcpyDtoH ON KNOWN QMD VAs ########\n\n");

    /* dw[50:51] — same across kernels */
    {
        uint64_t va = ((uint64_t)kerns[0].qmd[51] << 32) | kerns[0].qmd[50];
        printf("QMD dw[50:51] @ 0xC8: VA = 0x%016lx\n", va);
        uint32_t buf[256] = {0};
        CUresult r = cuMemcpyDtoH(buf, (CUdeviceptr)va, 1024);
        if (r == CUDA_SUCCESS) {
            printf("  READABLE! First 64 dwords:\n");
            for (int d = 0; d < 64; d++)
                printf("  [0x%03x] = 0x%08x (%u)\n", d*4, buf[d], buf[d]);
        } else {
            const char *s = NULL; cuGetErrorString(r, &s);
            printf("  Not readable: %s\n", s?s:"?");
        }
    }

    /* dw[48:49] — per-kernel (use kern[0]'s) */
    {
        uint64_t va = ((uint64_t)kerns[0].qmd[49] << 32) | kerns[0].qmd[48];
        printf("\nQMD dw[48:49] @ 0xC0: VA = 0x%016lx\n", va);
        uint32_t buf[256] = {0};
        CUresult r = cuMemcpyDtoH(buf, (CUdeviceptr)va, 1024);
        if (r == CUDA_SUCCESS) {
            printf("  READABLE! First 64 dwords:\n");
            for (int d = 0; d < 64; d++)
                printf("  [0x%03x] = 0x%08x (%u)\n", d*4, buf[d], buf[d]);
        } else {
            const char *s = NULL; cuGetErrorString(r, &s);
            printf("  Not readable: %s\n", s?s:"?");
        }
    }

    cuMemFree(d_out);
    cuModuleUnload(m16); cuModuleUnload(m32);
    cuModuleUnload(m64); cuModuleUnload(m128);
    cuModuleUnload(m255);
    cuCtxDestroy(ctx);

    printf("\nDone.\n");
    return 0;
}
