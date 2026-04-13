/*
 * cbuf0_probe2.c — Extract per-kernel inline constant buffer loads from pushbuffer.
 *
 * Focused probe: for each kernel launch, capture the full pushbuffer diff region
 * and extract ALL inline CB_LOAD_INLINE_DATA (method 0x206d) payloads. Then diff
 * corresponding payloads across kernels with different register counts.
 *
 * Build:
 *   gcc -O0 -g -o tools/cbuf0_probe2 tools/cbuf0_probe2.c \
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

static void build_cubin(const char *src, const char *cubin_path, int maxreg) {
    char src_path[256];
    snprintf(src_path, sizeof(src_path), "%s.cu", cubin_path);
    FILE *f = fopen(src_path, "w");
    fprintf(f, "%s", src);
    fclose(f);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "nvcc -cubin -arch=sm_90a --maxrregcount=%d -o %s %s 2>&1",
             maxreg, cubin_path, src_path);
    if (system(cmd) != 0) { fprintf(stderr, "nvcc failed\n"); exit(1); }
}

#define MAX_LOADS 16
#define MAX_DW 256

struct cb_load {
    uint64_t addr;
    int size;
    int index;
    int n_dwords;
    uint32_t data[MAX_DW];
};

struct launch_capture {
    int n_loads;
    struct cb_load loads[MAX_LOADS];
};

/* Parse pushbuffer diff to extract all CB_LOAD_INLINE_DATA payloads
 * from a SINGLE kernel launch. The diff region contains exactly one launch. */
static void extract_cb_loads(const uint8_t *before, const uint8_t *after,
                              size_t sz, struct launch_capture *cap) {
    const uint32_t *ring = (const uint32_t *)after;
    int ndw = sz / 4;
    cap->n_loads = 0;

    /* Find diff bounds */
    int first_byte = -1, last_byte = -1;
    for (size_t k = 0; k < sz; k++) {
        if (before[k] != after[k]) {
            if (first_byte < 0) first_byte = k;
            last_byte = k;
        }
    }
    if (first_byte < 0) return;

    int start_dw = first_byte / 4;
    int end_dw = (last_byte + 4) / 4;
    if (end_dw > ndw) end_dw = ndw;

    /* Parse methods in the diff region */
    uint64_t cur_addr = 0;
    int cur_size = 0, cur_index = 0;

    for (int i = start_dw; i < end_dw; ) {
        uint32_t w = ring[i];
        int opcode = (w >> 29) & 7;
        int count = (w >> 16) & 0x1FFF;
        int method = w & 0xFFFF;

        if (opcode < 1 || opcode > 4 || count < 1 || count > 200) { i++; continue; }

        if (method == 0x2062 && count >= 2 && i + 2 < ndw) {
            /* CB address: hi dword, lo dword */
            cur_addr = ((uint64_t)ring[i+1] << 32) | ring[i+2];
        }
        if (method == 0x2060 && count >= 2 && i + 2 < ndw) {
            cur_size = ring[i+1];
            cur_index = ring[i+2];
        }
        if (method == 0x206d && count > 0) {
            int li = cap->n_loads;
            if (li < MAX_LOADS) {
                cap->loads[li].addr = cur_addr;
                cap->loads[li].size = cur_size;
                cap->loads[li].index = cur_index;
                int n = count < MAX_DW ? count : MAX_DW;
                cap->loads[li].n_dwords = n;
                for (int j = 0; j < n && i+1+j < ndw; j++)
                    cap->loads[li].data[j] = ring[i+1+j];
                cap->n_loads++;
            }
        }
        i += 1 + count;
    }
}

int main(void) {
    printf("=== cbuf0 Probe v2: Inline CB Load Differential Analysis ===\n\n");

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    /* Use the same heavy kernel source from before */
    const char *src =
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

    build_cubin(src, "/tmp/cbuf0_k16.cubin", 16);
    build_cubin(src, "/tmp/cbuf0_k32.cubin", 32);
    build_cubin(src, "/tmp/cbuf0_k64.cubin", 64);

    CUmodule m16, m32, m64;
    CUfunction f16 = load_cubin_func("/tmp/cbuf0_k16.cubin", "kern", &m16);
    CUfunction f32 = load_cubin_func("/tmp/cbuf0_k32.cubin", "kern", &m32);
    CUfunction f64 = load_cubin_func("/tmp/cbuf0_k64.cubin", "kern", &m64);

    int r16 = get_regs(f16);
    int r32 = get_regs(f32);
    int r64 = get_regs(f64);

    printf("Register counts: k16=%d  k32=%d  k64=%d\n\n", r16, r32, r64);

    CUdeviceptr d_out;
    CHECK(cuMemAlloc(&d_out, 4096));
    void *params[] = { &d_out };

    /* Warmup — several launches to stabilize ring */
    for (int w = 0; w < 5; w++) {
        CHECK(cuLaunchKernel(f16, 1,1,1, 1,1,1, 0, NULL, params, NULL));
        cuCtxSynchronize();
    }

    /* Capture each kernel's launch with full pushbuffer parsing.
     * Critical: we capture IMMEDIATELY around each launch to avoid
     * mixing with other launches' data. */

    struct {
        const char *name;
        CUfunction func;
        int regs;
        struct launch_capture cap;
    } kerns[] = {
        { "k16", f16, r16, {0} },
        { "k32", f32, r32, {0} },
        { "k64", f64, r64, {0} },
    };
    int nk = 3;

    for (int k = 0; k < nk; k++) {
        uint8_t *snap1 = malloc(RING_SIZE);
        uint8_t *snap2 = malloc(RING_SIZE);

        scopy(snap1, (const void *)RING_BASE, RING_SIZE);
        CHECK(cuLaunchKernel(kerns[k].func, 1,1,1, 1,1,1, 0, NULL, params, NULL));
        cuCtxSynchronize();
        scopy(snap2, (const void *)RING_BASE, RING_SIZE);

        extract_cb_loads(snap1, snap2, RING_SIZE, &kerns[k].cap);

        printf("%s (%d regs): %d CB loads\n", kerns[k].name, kerns[k].regs,
               kerns[k].cap.n_loads);
        for (int l = 0; l < kerns[k].cap.n_loads; l++) {
            struct cb_load *cb = &kerns[k].cap.loads[l];
            printf("  [%d] addr=0x%016lx size=%d idx=%d dwords=%d\n",
                   l, cb->addr, cb->size, cb->index, cb->n_dwords);
        }
        printf("\n");

        free(snap1); free(snap2);
    }

    /* Match CB loads by size (they should correspond across kernels) */
    printf("\n######## DIFFING CORRESPONDING CB LOADS ########\n\n");

    /* Strategy: match by (size, index) tuple */
    for (int l0 = 0; l0 < kerns[0].cap.n_loads; l0++) {
        struct cb_load *ref = &kerns[0].cap.loads[l0];

        /* Find matching loads in other kernels */
        int matches[3] = { l0, -1, -1 };
        for (int k = 1; k < nk; k++) {
            for (int l = 0; l < kerns[k].cap.n_loads; l++) {
                if (kerns[k].cap.loads[l].size == ref->size &&
                    kerns[k].cap.loads[l].n_dwords == ref->n_dwords) {
                    matches[k] = l;
                    break;
                }
            }
        }

        /* Check all kernels have a match */
        int all_match = 1;
        for (int k = 0; k < nk; k++) {
            if (matches[k] < 0) { all_match = 0; break; }
        }
        if (!all_match) continue;

        /* Count diffs */
        int n_diffs = 0;
        int min_dw = ref->n_dwords;
        for (int k = 1; k < nk; k++) {
            int nd = kerns[k].cap.loads[matches[k]].n_dwords;
            if (nd < min_dw) min_dw = nd;
        }
        for (int d = 0; d < min_dw; d++) {
            int diff = 0;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].cap.loads[matches[k]].data[d] != ref->data[d]) { diff = 1; break; }
            }
            if (diff) n_diffs++;
        }

        printf("CB load size=%d idx=%d: %d diffs in %d dwords\n",
               ref->size, ref->index, n_diffs, min_dw);

        if (n_diffs == 0) {
            printf("  (identical — not kernel-specific)\n\n");
            continue;
        }

        /* Show diffs with register count analysis */
        for (int d = 0; d < min_dw; d++) {
            int diff = 0;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].cap.loads[matches[k]].data[d] != ref->data[d]) { diff = 1; break; }
            }
            if (!diff) continue;

            printf("  [0x%03x] dw[%d]:", d*4, d);
            for (int k = 0; k < nk; k++) {
                uint32_t v = kerns[k].cap.loads[matches[k]].data[d];
                printf("  %s(%d)=0x%08x(%u)", kerns[k].name, kerns[k].regs, v, v);
            }

            /* Register count checks */
            int exact = 1, rnd2 = 1, rnd4 = 1, rnd8 = 1;
            for (int k = 0; k < nk; k++) {
                uint32_t v = kerns[k].cap.loads[matches[k]].data[d];
                int r = kerns[k].regs;
                if (v != (uint32_t)r) exact = 0;
                if (v != (uint32_t)((r+1)&~1)) rnd2 = 0;
                if (v != (uint32_t)((r+3)&~3)) rnd4 = 0;
                if (v != (uint32_t)((r+7)&~7)) rnd8 = 0;
            }
            if (exact) printf("  *** REGISTER_COUNT (exact) ***");
            if (rnd2) printf("  *** REG_COUNT (round2) ***");
            if (rnd4) printf("  *** REG_COUNT (round4) ***");
            if (rnd8) printf("  *** REG_COUNT (round8) ***");

            /* Byte-level checks */
            for (int bp = 0; bp < 4; bp++) {
                int bmatch = 1;
                for (int k = 0; k < nk; k++) {
                    uint32_t v = (kerns[k].cap.loads[matches[k]].data[d] >> (bp*8)) & 0xFF;
                    if (v != (uint32_t)kerns[k].regs) { bmatch = 0; break; }
                }
                if (bmatch) printf("  *** REG_COUNT byte[%d] ***", bp);
            }

            /* Check reg_count / 2 (Hopper register granularity is 2) */
            {
                int match_half = 1;
                for (int k = 0; k < nk; k++) {
                    uint32_t v = kerns[k].cap.loads[matches[k]].data[d];
                    /* Register allocation on Hopper: round up to multiple of 2 */
                    int alloc = (kerns[k].regs + 1) & ~1; /* round to 2 */
                    if (v != (uint32_t)(alloc / 2)) { match_half = 0; break; }
                }
                if (match_half) printf("  *** REG_COUNT/2 ***");
            }

            printf("\n");
        }

        /* Full dump */
        printf("\n  Full dump (size=%d):\n", ref->size);
        for (int d = 0; d < min_dw; d++) {
            int diff = 0;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].cap.loads[matches[k]].data[d] != ref->data[d]) { diff = 1; break; }
            }
            printf("  [0x%03x] %s", d*4, diff ? "***" : "   ");
            for (int k = 0; k < nk; k++) {
                uint32_t v = kerns[k].cap.loads[matches[k]].data[d];
                if (k > 0 && !diff)
                    printf("  --------");
                else
                    printf("  0x%08x", v);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Also try matching by position (load index in sequence) */
    printf("\n######## POSITIONAL CB LOAD DIFF (by index) ########\n\n");
    int max_loads = kerns[0].cap.n_loads;
    for (int k = 1; k < nk; k++) {
        if (kerns[k].cap.n_loads < max_loads) max_loads = kerns[k].cap.n_loads;
    }

    for (int l = 0; l < max_loads; l++) {
        int min_dw = kerns[0].cap.loads[l].n_dwords;
        for (int k = 1; k < nk; k++) {
            int nd = kerns[k].cap.loads[l].n_dwords;
            if (nd < min_dw) min_dw = nd;
        }

        int n_diffs = 0;
        for (int d = 0; d < min_dw; d++) {
            int diff = 0;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].cap.loads[l].data[d] != kerns[0].cap.loads[l].data[d]) { diff = 1; break; }
            }
            if (diff) n_diffs++;
        }

        printf("Load[%d]: size=%d/%d/%d dwords=%d, %d diffs\n", l,
               kerns[0].cap.loads[l].size, kerns[1].cap.loads[l].size,
               kerns[2].cap.loads[l].size, min_dw, n_diffs);

        if (n_diffs == 0) {
            printf("  (identical)\n");
            continue;
        }

        for (int d = 0; d < min_dw; d++) {
            int diff = 0;
            for (int k = 1; k < nk; k++) {
                if (kerns[k].cap.loads[l].data[d] != kerns[0].cap.loads[l].data[d]) { diff = 1; break; }
            }
            if (!diff) continue;

            printf("  [0x%03x]:", d*4);
            for (int k = 0; k < nk; k++) {
                uint32_t v = kerns[k].cap.loads[l].data[d];
                printf("  %s(%d)=0x%08x(%u)", kerns[k].name, kerns[k].regs, v, v);
            }

            /* Comprehensive register count search */
            for (int k = 0; k < nk; k++) {
                uint32_t v = kerns[k].cap.loads[l].data[d];
                int r = kerns[k].regs;
                /* Check if any byte/nibble/field matches the register count */
                for (int bp = 0; bp < 4; bp++) {
                    int byte_val = (v >> (bp*8)) & 0xFF;
                    if (byte_val == r && k == 0) {
                        /* Check all kernels */
                        int all_ok = 1;
                        for (int kk = 1; kk < nk; kk++) {
                            int bv = (kerns[kk].cap.loads[l].data[d] >> (bp*8)) & 0xFF;
                            if (bv != kerns[kk].regs) { all_ok = 0; break; }
                        }
                        if (all_ok) printf("  *** REG byte[%d] ***", bp);
                    }
                }
            }

            printf("\n");
        }
        printf("\n");
    }

    cuMemFree(d_out);
    cuModuleUnload(m16); cuModuleUnload(m32); cuModuleUnload(m64);
    cuCtxDestroy(ctx);
    printf("Done.\n");
    return 0;
}
