/*
 * dump_qmd.c — QMD pushbuffer capture tool for Lithos
 *
 * Strategy:
 *   1. cuInit + cuCtxCreate on device 0
 *   2. Record all MAP_SHARED mappings from /proc/self/maps BEFORE module load
 *   3. Load a minimal cubin (gemv.cubin), get function handle
 *   4. Allocate device memory for dummy args
 *   5. Record MAP_SHARED mappings AFTER cuModuleLoadData
 *   6. Launch kernel (1 block, 1 thread, 0 smem)
 *   7. cuCtxSynchronize
 *   8. Scan all MAP_SHARED pages (both new and pre-existing) for QMD signature:
 *        - qmd_version == 4 at dword[0] (Hopper)
 *        - Scan backwards from current pushbuffer write pointer if we can find it
 *   9. Dump all matching 256-byte regions to /tmp/qmd_dump.bin and stderr
 *
 * Build:
 *   gcc -O0 -g -o /home/ubuntu/lithos/tools/dump_qmd \
 *       /home/ubuntu/lithos/tools/dump_qmd.c \
 *       -I/usr/local/cuda/include \
 *       -L/usr/lib/aarch64-linux-gnu \
 *       -lcuda -ldl -lpthread
 *
 * Run:
 *   /home/ubuntu/lithos/tools/dump_qmd
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <cuda.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

#define CHECK_CU(call) do {                                             \
    CUresult _r = (call);                                               \
    if (_r != CUDA_SUCCESS) {                                           \
        const char *_s = NULL;                                          \
        cuGetErrorString(_r, &_s);                                      \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",               \
                (int)_r, _s ? _s : "?", __FILE__, __LINE__);           \
        exit(1);                                                        \
    }                                                                   \
} while(0)

/* ------------------------------------------------------------------ */
/* /proc/self/maps scanner                                              */
/* ------------------------------------------------------------------ */

#define MAX_MAPS 4096

typedef struct {
    uintptr_t start;
    uintptr_t end;
    char perms[8];
    char path[256];
} MapEntry;

static int read_maps(MapEntry *out, int max_count) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) { perror("fopen /proc/self/maps"); return 0; }
    int n = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) && n < max_count) {
        uintptr_t s, e;
        char perms[8], offs[16], dev[16], inode_s[16], path[256];
        path[0] = '\0';
        int r = sscanf(line, "%lx-%lx %7s %15s %15s %15s %255[^\n]",
                       &s, &e, perms, offs, dev, inode_s, path);
        if (r >= 3) {
            out[n].start = s;
            out[n].end   = e;
            strncpy(out[n].perms, perms, 7); out[n].perms[7] = '\0';
            strncpy(out[n].path,  path,  255); out[n].path[255] = '\0';
            n++;
        }
    }
    fclose(f);
    return n;
}

static void print_maps(const MapEntry *maps, int n, const char *tag) {
    fprintf(stderr, "\n=== /proc/self/maps [%s] — %d entries ===\n", tag, n);
    for (int i = 0; i < n; i++) {
        size_t sz = maps[i].end - maps[i].start;
        /* Only print SHARED mappings or large anonymous ones */
        if (strchr(maps[i].perms, 's') ||
            (sz >= 1024*1024 && maps[i].path[0] == '\0')) {
            fprintf(stderr, "  %016lx-%016lx  %s  sz=0x%lx  %s\n",
                    maps[i].start, maps[i].end,
                    maps[i].perms, (unsigned long)sz,
                    maps[i].path[0] ? maps[i].path : "(anon)");
        }
    }
}

/* ------------------------------------------------------------------ */
/* QMD search                                                           */
/* ------------------------------------------------------------------ */

/*
 * Search a mapped region for 256-byte blocks that look like a Hopper QMD.
 * Hopper QMD version == 4 at bytes [0:4], and has a non-zero entry_pc
 * at bytes [8:16].
 *
 * Also try version==4 at bytes [0:4] even if entry_pc is zero (for
 * zeroed-out but version-stamped QMDs).
 */
static int search_qmd_in_region(const uint8_t *base, size_t sz,
                                uintptr_t va_base,
                                uint8_t *found_out, /* 256 bytes out */
                                uintptr_t *found_va)
{
    if (sz < 256) return 0;

    /* Walk 256-byte aligned blocks */
    for (size_t off = 0; off + 256 <= sz; off += 256) {
        const uint32_t *dw = (const uint32_t *)(base + off);
        uint32_t ver = dw[0];
        if (ver == 4) {
            /* Candidate: check that at least one non-zero field exists */
            int nonzero = 0;
            for (int k = 1; k < 64; k++) {
                if (dw[k]) { nonzero = 1; break; }
            }
            if (nonzero) {
                memcpy(found_out, base + off, 256);
                *found_va = va_base + off;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Also try 4-byte alignment (not just 256-byte) for the pushbuffer scan,
 * since the QMD might not be 256-byte aligned in the ring.
 */
static int search_qmd_fine(const uint8_t *base, size_t sz,
                           uintptr_t va_base,
                           uint8_t found_out[][256],
                           uintptr_t found_va[],
                           int max_found)
{
    int nfound = 0;
    if (sz < 256) return 0;

    for (size_t off = 0; off + 256 <= sz && nfound < max_found; off += 4) {
        const uint32_t *dw = (const uint32_t *)(base + off);
        uint32_t ver = dw[0];
        if (ver == 4) {
            int nonzero = 0;
            for (int k = 1; k < 64; k++) {
                if (dw[k]) { nonzero = 1; break; }
            }
            if (nonzero) {
                memcpy(found_out[nfound], base + off, 256);
                found_va[nfound] = va_base + off;
                nfound++;
                off += 252; /* skip to after this candidate */
            }
        }
    }
    return nfound;
}

/* ------------------------------------------------------------------ */
/* Hex dump                                                             */
/* ------------------------------------------------------------------ */

static void hexdump(FILE *f, const uint8_t *data, size_t sz, const char *label) {
    fprintf(f, "\n=== %s ===\n", label);
    for (size_t i = 0; i < sz; i += 16) {
        fprintf(f, "  %04zx: ", i);
        for (size_t j = i; j < i+16 && j < sz; j++)
            fprintf(f, "%02x ", data[j]);
        fprintf(f, "\n");
    }
}

/* ------------------------------------------------------------------ */
/* QMD field decoder for Hopper (our reconstructed layout)             */
/* ------------------------------------------------------------------ */

static void decode_qmd(FILE *f, const uint8_t *q, uintptr_t va) {
    const uint32_t *dw = (const uint32_t *)q;

    fprintf(f, "\n=== QMD Decode (va=0x%lx) ===\n", va);
    fprintf(f, "  [0x000] qmd_version      = %u\n",  dw[0]);
    fprintf(f, "  [0x004] dword[1]         = 0x%08x\n", dw[1]);
    fprintf(f, "  [0x008] entry_pc_lo      = 0x%08x\n", dw[2]);
    fprintf(f, "  [0x00C] entry_pc_hi      = 0x%08x\n", dw[3]);
    uint64_t entry_pc = ((uint64_t)dw[3] << 32) | dw[2];
    fprintf(f, "         => entry_pc       = 0x%016lx\n", entry_pc);
    fprintf(f, "  [0x010] dword[4]         = 0x%08x  (expected: register_count lo16 + barrier_count hi16)\n", dw[4]);
    fprintf(f, "         => register_count = %u (lo16)\n", dw[4] & 0xffff);
    fprintf(f, "         => barrier_count  = %u (hi16)\n", (dw[4] >> 16) & 0xffff);
    fprintf(f, "  [0x014] dword[5]         = 0x%08x  (expected: shared_mem_size)\n", dw[5]);
    fprintf(f, "  [0x018] dword[6]         = 0x%08x  (expected: block_dim_x)\n", dw[6]);
    fprintf(f, "  [0x01C] dword[7]         = 0x%08x  (expected: block_dim_y<<16 | block_dim_z or split)\n", dw[7]);
    fprintf(f, "         => lo16           = %u\n", dw[7] & 0xffff);
    fprintf(f, "         => hi16           = %u\n", (dw[7] >> 16) & 0xffff);
    fprintf(f, "  [0x020] dword[8]         = 0x%08x  (expected: grid_dim_x)\n", dw[8]);
    fprintf(f, "  [0x024] dword[9]         = 0x%08x  (expected: grid_dim_y)\n", dw[9]);
    fprintf(f, "  [0x028] dword[10]        = 0x%08x  (expected: grid_dim_z)\n", dw[10]);
    fprintf(f, "  [0x02C] dword[11]        = 0x%08x  (expected: cluster_dim_xyz)\n", dw[11]);
    fprintf(f, "  [0x030] dword[12]        = 0x%08x  (expected: cbuf0_base_addr_lo)\n", dw[12]);
    fprintf(f, "  [0x034] dword[13]        = 0x%08x  (expected: cbuf0_base_addr_hi)\n", dw[13]);
    uint64_t cbuf0 = ((uint64_t)dw[13] << 32) | dw[12];
    fprintf(f, "         => cbuf0_addr     = 0x%016lx\n", cbuf0);
    fprintf(f, "  [0x038] dword[14]        = 0x%08x  (expected: cbuf0_size)\n", dw[14]);
    fprintf(f, "  [0x03C] dword[15]        = 0x%08x  (expected: cbuf0_flags)\n", dw[15]);

    /* Print all dwords */
    fprintf(f, "\n  --- All 64 dwords ---\n");
    for (int i = 0; i < 64; i++) {
        if (dw[i] != 0) {
            fprintf(f, "  [0x%03x] dword[%2d] = 0x%08x (%u)\n",
                    i*4, i, dw[i], dw[i]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Read file into malloc'd buffer                                       */
/* ------------------------------------------------------------------ */

static uint8_t *read_file(const char *path, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    fclose(f);
    *out_sz = sz;
    return buf;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const char *cubin_path = argc > 1 ? argv[1] :
        "/home/ubuntu/lithos/inference/gemv.cubin";
    const char *kernel_name = argc > 2 ? argv[2] : "gptq_gemv";

    fprintf(stderr, "QMD dump tool — cubin: %s  kernel: %s\n",
            cubin_path, kernel_name);

    /* Step 1: CUDA init */
    fprintf(stderr, "\n[1] cuInit...\n");
    CHECK_CU(cuInit(0));

    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    {
        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        fprintf(stderr, "    Device: %s\n", name);
    }

    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    fprintf(stderr, "    Context created.\n");

    /* Step 2: maps before module load */
    MapEntry maps_before[MAX_MAPS];
    int nb = read_maps(maps_before, MAX_MAPS);
    print_maps(maps_before, nb, "before module load");

    /* Step 3: Load module */
    fprintf(stderr, "\n[2] Loading cubin...\n");
    size_t cubin_sz;
    uint8_t *cubin_data = read_file(cubin_path, &cubin_sz);
    if (!cubin_data) { fprintf(stderr, "Failed to read cubin\n"); return 1; }

    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, cubin_data));
    fprintf(stderr, "    Module loaded.\n");

    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, kernel_name));
    fprintf(stderr, "    Function '%s' obtained.\n", kernel_name);

    /* Step 4: maps after module load */
    MapEntry maps_after[MAX_MAPS];
    int na = read_maps(maps_after, MAX_MAPS);
    print_maps(maps_after, na, "after module load");

    /* Find new MAP_SHARED or large anonymous mappings */
    fprintf(stderr, "\n=== New mappings after module load ===\n");
    for (int i = 0; i < na; i++) {
        int found_before = 0;
        for (int j = 0; j < nb; j++) {
            if (maps_after[i].start == maps_before[j].start &&
                maps_after[i].end   == maps_before[j].end) {
                found_before = 1;
                break;
            }
        }
        if (!found_before) {
            size_t sz = maps_after[i].end - maps_after[i].start;
            fprintf(stderr, "  NEW: %016lx-%016lx  %s  sz=0x%lx  %s\n",
                    maps_after[i].start, maps_after[i].end,
                    maps_after[i].perms, (unsigned long)sz,
                    maps_after[i].path[0] ? maps_after[i].path : "(anon)");
        }
    }

    /* Step 5: Allocate device memory (dummy 4KB) */
    CUdeviceptr d_mem;
    CHECK_CU(cuMemAlloc(&d_mem, 4096));
    CHECK_CU(cuMemsetD32(d_mem, 0, 1024));
    fprintf(stderr, "\n[3] Device memory allocated at 0x%lx\n", (unsigned long)d_mem);

    /* Step 6: Launch kernel
     * We use a tiny grid: 1 block of 1 thread (it will likely crash/bail
     * inside the kernel, but the QMD gets written to the pushbuffer first).
     * Pass dummy pointer as single param. */
    void *params[1] = { &d_mem };

    /* Capture maps just before launch */
    MapEntry maps_prelaunch[MAX_MAPS];
    int np = read_maps(maps_prelaunch, MAX_MAPS);

    fprintf(stderr, "\n[4] Launching kernel (1x1x1 grid, 1x1x1 block)...\n");
    CUresult launch_result = cuLaunchKernel(
        func,
        1, 1, 1,   /* grid */
        1, 1, 1,   /* block */
        0,         /* shared mem */
        NULL,      /* stream */
        params,
        NULL
    );
    fprintf(stderr, "    cuLaunchKernel returned: %d\n", (int)launch_result);

    /* Step 7: Capture maps immediately after launch, before sync */
    MapEntry maps_postlaunch[MAX_MAPS];
    int npl = read_maps(maps_postlaunch, MAX_MAPS);

    /* Step 8: Sync */
    fprintf(stderr, "\n[5] Syncing...\n");
    CUresult sync_result = cuCtxSynchronize();
    fprintf(stderr, "    cuCtxSynchronize returned: %d\n", (int)sync_result);

    /* Step 9: Scan all readable shared mappings for QMD */
    fprintf(stderr, "\n[6] Scanning for QMD (version==4 dword)...\n");

    #define MAX_QMD_FOUND 32
    uint8_t qmd_candidates[MAX_QMD_FOUND][256];
    uintptr_t qmd_vas[MAX_QMD_FOUND];
    int total_found = 0;

    for (int i = 0; i < npl && total_found < MAX_QMD_FOUND; i++) {
        const MapEntry *me = &maps_postlaunch[i];
        size_t sz = me->end - me->start;

        /* Only scan readable maps */
        if (me->perms[0] != 'r') continue;

        /* Skip very small or very large (>128MB) maps to stay fast */
        if (sz < 256) continue;
        if (sz > 128UL * 1024 * 1024) {
            fprintf(stderr, "  Skipping large map 0x%lx sz=0x%lx %s\n",
                    me->start, (unsigned long)sz, me->path);
            continue;
        }

        /* Try to access (some maps may fault) */
        const uint8_t *base = (const uint8_t *)me->start;

        /* Quick check: is memory accessible? Read first byte. */
        volatile uint8_t probe;
        /* Use mincore to check if pages are present */
        /* Actually just try - we'll get a signal if it faults */
        /* For safety use a pipe read trick or just try with error recovery */

        /* Scan the region */
        int n = search_qmd_fine(base, sz, me->start,
                                qmd_candidates + total_found,
                                qmd_vas + total_found,
                                MAX_QMD_FOUND - total_found);
        if (n > 0) {
            fprintf(stderr, "  Found %d QMD candidate(s) in [%lx-%lx] %s %s\n",
                    n, me->start, me->end, me->perms,
                    me->path[0] ? me->path : "(anon)");
            total_found += n;
        }
    }

    fprintf(stderr, "\n[7] Total QMD candidates found: %d\n", total_found);

    /* Step 10: Dump to file and decode */
    if (total_found == 0) {
        fprintf(stderr, "  No QMD found. The pushbuffer may not be readable from host.\n");
        fprintf(stderr, "  This is expected if CUDA uses non-MAP_SHARED GPU memory.\n");
        fprintf(stderr, "  Try the LD_PRELOAD wrapper instead.\n");
    }

    FILE *bin_out = fopen("/tmp/qmd_dump.bin", "wb");
    for (int i = 0; i < total_found; i++) {
        char label[64];
        snprintf(label, sizeof(label), "QMD candidate %d (va=0x%lx)", i, qmd_vas[i]);
        hexdump(stderr, qmd_candidates[i], 256, label);
        decode_qmd(stderr, qmd_candidates[i], qmd_vas[i]);
        if (bin_out) {
            fwrite(qmd_candidates[i], 1, 256, bin_out);
        }
    }
    if (bin_out) {
        fclose(bin_out);
        fprintf(stderr, "\n[8] Wrote %d QMD(s) to /tmp/qmd_dump.bin\n", total_found);
    }

    /* Cleanup */
    cuMemFree(d_mem);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(cubin_data);

    return 0;
}
