# cuBLAS Kernel Library Catalog for Lithos

Machine: aarch64 (GH200), CUDA 12.8, cuBLAS 12.8.4.1
Date: 2026-04-12

---

## 1. Summary Statistics

### Library Sizes
| Library | Size | ELF Cubins | PTX Modules |
|---------|------|-----------|-------------|
| libcublas.so.12.8.4.1 | 110 MB | 1,610 | 192 |
| libcublasLt.so.12.8.4.1 | 687 MB | 5,732 | 136 |
| **Total** | **797 MB** | **7,342** | **328** |

### Kernel Function Counts (sm_90 only)
| Library | Unique Functions (sm_90) |
|---------|------------------------|
| libcublas | 200+ |
| libcublasLt | 10,525 |
| **Total sm_90 kernels** | **~10,725** |

Extrapolating across all architectures: **~60,000-80,000 total kernel variants** across both libraries.

### GPU Architecture Targets

**libcublas** (10 targets):
| Arch | Cubins | Generation |
|------|--------|-----------|
| sm_50 | 192 | Maxwell |
| sm_60 | 193 | Pascal |
| sm_61 | 118 | Pascal (consumer) |
| sm_70 | 194 | Volta |
| sm_75 | 14 | Turing |
| sm_80 | 195 | Ampere |
| sm_86 | 118 | Ampere (GA10x) |
| sm_90 | 195 | Hopper |
| sm_100 | 195 | Blackwell |
| sm_120 | 196 | Next-gen |

**libcublasLt** (11 targets):
| Arch | Cubins | Generation |
|------|--------|-----------|
| sm_50 | 90 | Maxwell |
| sm_60 | 91 | Pascal |
| sm_61 | 82 | Pascal (consumer) |
| sm_70 | 144 | Volta |
| sm_75 | 116 | Turing |
| sm_80 | 477 | Ampere |
| sm_86 | 100 | Ampere (GA10x) |
| sm_89 | 247 | Ada Lovelace |
| sm_90 | 1,591 | Hopper |
| sm_100 | 1,362 | Blackwell |
| sm_120 | 1,432 | Next-gen |

Key observation: sm_90 (Hopper) has the most cubins in cublasLt -- NVIDIA clearly prioritizes
Hopper optimization. sm_89 (Ada) only appears in cublasLt, not cublas.

### PTX (Forward Compatibility)
- All PTX targets sm_120 only (highest arch), providing JIT forward-compatibility
- 192 PTX modules in cublas, 136 in cublasLt (1 targeting sm_52)
- PTX is fallback for future GPUs; all real performance comes from pre-compiled SASS cubins

---

## 2. Data Type Variants

Counted in cublasLt sm_90 kernels:

| Data Type | Mangled Name | Kernel Count | Notes |
|-----------|-------------|-------------|-------|
| FP16 (half) | `6__half` | 1,258 | Primary LLM inference type |
| BF16 | `13__nv_bfloat16` | 1,035 | Primary LLM training type |
| FP8 E4M3 | `13__nv_fp8_e4m3` | 37 | Hopper FP8 (inference) |
| FP8 E5M2 | `13__nv_fp8_e5m2` | 37 | Hopper FP8 (training gradients) |
| FP32 | `f` | extensive | Accumulator + legacy |
| FP64 | `d` | ~12 | Scientific computing |
| TF32 | `tf32` | 288 | Ampere+ tensor core mode |
| INT8 (s8) | `a` (signed char) | 626 | Quantized inference |
| Complex (float2) | `6float2` | present | Not relevant for LLM |
| Complex (double2) | `7double2` | present | Not relevant for LLM |

**Notable absence: No INT4 (W4A16) kernels.** cuBLAS does not ship W4A16 GEMM.
Quantized int4 inference must use cutlass, marlin, or custom kernels.

---

## 3. Kernel Categories

### cublasLt sm_90 Kernel Class Breakdown

| Kernel Class | Count | Description |
|-------------|-------|-------------|
| x_cublas (matmul) | 1,407 | CUTLASS-based warp-specialized GEMM |
| gemvx | 1,056 | Extended GEMV with epilogue fusion |
| gemv | 824 | Standard GEMV |
| gemmk1_kernel | 632 | Single-K-slice GEMM (small N, thin GEMM) |
| gemmSN_NN_kernel | 456 | Small-N NN-layout GEMM (GEMV-like) |
| splitKreduce_kernel | 314 | Split-K reduction epilogue |
| gemvNSP_kernel | 288 | Non-split-pipelined GEMV |
| gemmSN_TN_kernel | 182 | Small-N TN-layout GEMM |
| epilogue | 147 | Standalone epilogue kernels |
| gemmSN_kernel_int | 64 | Integer small-N GEMM |
| sgemm_largek | 8 | Large-K single-precision |
| dgemm_largek | 12 | Large-K double-precision |
| cgemm_largek | 9 | Complex large-K |
| zgemm_largek | 18 | Double-complex large-K |
| syrk/symm/her2k | 256 | BLAS-3 routines |

### Tensor Core Usage
- **1,536 HMMA/WGMMA instructions** found in libcublas sm_90 SASS
- **5,828 WGMMA-related functions** in libcublasLt sm_90
- Hopper uses WGMMA (Warpgroup MMA) for all tensor-core GEMM operations

---

## 4. Naming Convention Decoded

### C++ Mangled Name Structure

Example: `_ZN8cublasLt19splitKreduce_kernelILi8ELi32Eif13__nv_fp8_e4m3fS1_Lb1Ef6__halfS2_Lb0ELb0ELb0EEEv...`

Decoded:
```
_ZN8cublasLt                    -> namespace cublasLt
19splitKreduce_kernel           -> function: splitKreduce_kernel
I                               -> template parameters begin
  Li8E                          -> int literal 8 (tile dim or warp count)
  Li32E                         -> int literal 32 (tile dim or threads)
  i                             -> int (accumulator type)
  f                             -> float (scale type)
  13__nv_fp8_e4m3               -> input A type: FP8 E4M3
  f                             -> float (intermediate type)
  S1_                           -> same as input A (input B type)
  Lb1E                          -> bool true (transpose flag)
  f                             -> float (output scale type)
  6__half                       -> output type: FP16
  S2_                           -> same as __half (bias type)
  Lb0ELb0ELb0E                  -> three bool flags (epilogue options)
EE                              -> end template params
v                               -> void return
...                             -> parameter types
```

### Key Kernel Families

**1. gemmk1_kernel** - Single-slice thin GEMM for batch=1 / small-N
```
Template: <accum_type, data_type, threads=256, unroll=5, flags..., TensorLayout, bias_mode>
Uses: cublasGemvTensorStridedBatched / cublasGemvTensorBatched
```

**2. gemmSN_{NN,TN}_kernel** - Small-N optimized GEMM (GEMV territory)
```
Template: <data_type, threads, N_tile, unroll, M_tile, K_tile, flags, Layout>
Variants: _64addr (64-bit addressing), _half (FP16 specialization)
```

**3. splitKreduce_kernel** - Split-K reduction pass
```
Template: <warp_m, warp_n, accum, scale, A_type, B_type, transpose, out_scale, C_type, bias_type, flags...>
Always runs after split-K GEMM main body
```

**4. x_cublas matmul** - Primary CUTLASS/warp-specialized GEMM
```
The bulk of cublasLt. These are NVIDIA's hand-tuned warp-specialized GEMM kernels
using WGMMA instructions on Hopper. Tile sizes embedded in template parameters.
```

### Tile Size Variants

Common tile configurations found in template parameters:
- Thread counts: 128, 256
- M tiles: 4, 8, 16, 32, 128, 256
- N tiles: 2, 4, 8, 16
- K tiles: 4, 8, 16, 32
- Unroll factors: 2, 4, 5, 6
- Shared memory sizes: 32KB, 64KB, 128KB, 256KB

---

## 5. Representative sm_90 SASS Disassembly

This is the `splitKreduce_kernel<8,32,int,float,fp8_e4m3,...>` from libcublas.
This kernel performs the reduction phase after a split-K GEMM and shows NVIDIA's
hand-tuned Hopper SASS including descriptor-based loads and FP8 conversion.

```sass
// Prologue: Load parameters from constant memory, check thread bounds
/*0000*/  LDC R1, c[0x0][0x28] ;                    // Stack pointer
/*0010*/  LDC R0, c[0x0][0x248] ;                    // Load operation mode
/*0020*/  ULDC.64 UR4, c[0x0][0x2c8] ;               // Uniform load: pointer A
/*0030*/  ULDC.64 UR6, c[0x0][0x2d8] ;               // Uniform load: pointer B
/*0040*/  ISETP.NE.U32.AND P0, PT, RZ, UR4, PT ;     // Check pointer != null
/*0060*/  ISETP.NE.U32.AND P2, PT, RZ, UR6, PT ;     // Check pointer != null
/*0070*/  ULDC.64 UR10, c[0x0][0x208] ;              // Load TMA descriptor base

// Scale factor loading via TMA descriptors
/*01c0*/  @P5 LDG.E.CONSTANT R0, desc[UR10][R6.64] ; // Descriptor-based global load
/*01e0*/  @P2 LDG.E.CONSTANT R19, desc[UR10][R14.64] ;
/*0200*/  @P1 LDG.E.CONSTANT R13, desc[UR10][R10.64] ;

// Thread ID computation
/*0220*/  S2R R12, SR_CTAID.X ;                       // CTA (block) ID
/*0250*/  S2R R5, SR_TID.X ;                          // Thread ID within block
/*0280*/  IMAD R12, R12, UR4, R5 ;                    // Global thread index

// FP8 E4M3 conversion in the epilogue
/*04d0*/  F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R11, RZ, R2, RZ ;  // FP32 -> FP8 E4M3

// Inner reduction loop: 8-element unrolled FADD chain
/*10b0*/  FADD R11, R10, R11 ;     // sum += val[0]
/*10c0*/  FADD R28, R11, R28 ;     // sum += val[1]
/*10d0*/  FADD R28, R28, R27 ;     // sum += val[2]
/*10e0*/  FADD R25, R28, R25 ;     // sum += val[3]
/*10f0*/  FADD R25, R25, R22 ;     // sum += val[4]
/*1100*/  FADD R10, R25, R18 ;     // sum += val[5]
/*1110*/  FADD R10, R10, R23 ;     // sum += val[6]
/*1130*/  FADD R10, R10, R29 ;     // sum += val[7]

// FP8 output conversion + store
/*14f0*/  F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R19, RZ, R14, RZ ;  // FP32 -> FP8
/*1510*/  STG.E.U8 desc[UR10][R10.64], R19 ;          // Store 1-byte FP8 result

// Cross-warp reduction via REDUX + ATOMG
/*15e0*/  REDUX.MAX.S32 UR5, R5 ;                     // Warp-level max reduction
/*1640*/  ATOMG.E.MAX.S32.STRONG.GPU PT, RZ, desc[UR10][R2.64], R7 ; // Global atomic max
```

### Key SASS Observations for Lithos:
1. **Descriptor-based loads** (`LDG.E.CONSTANT desc[UR][R.64]`): Hopper TMA-style memory access
2. **F2FP instructions**: Hardware FP8 conversion (E4M3/E5M2), saturation + NaN handling built-in
3. **REDUX instructions**: Warp-level reductions without shared memory
4. **ATOMG**: Global atomics with strong GPU ordering for split-K writeback
5. **8-wide FADD chain**: No ILP between additions -- purely sequential accumulation
6. **No WGMMA in this kernel**: splitKreduce is a simple epilogue; the WGMMA instructions are in the main GEMM body (x_cublas matmul kernels)

---

## 6. Qwen 3.5-27B Inference (batch=1) Kernel Mapping

### Model Parameters (Qwen 3.5-27B)
- hidden_dim = 5120
- intermediate_dim = 13824 (MLP)
- num_heads = 40, head_dim = 128
- num_kv_heads = 8 (GQA 5:1)
- vocab_size = 151,936

### Kernels Hit at batch=1 (decode phase)

**Linear projections (all are GEMV at batch=1):**

| Layer | Shape (M x N x K) | cuBLAS Kernel | Type |
|-------|-------------------|---------------|------|
| QKV projection | 1 x 7680 x 5120 | gemmk1_kernel / gemmSN_NN_kernel | bf16 GEMV |
| O projection | 1 x 5120 x 5120 | gemmk1_kernel / gemmSN_NN_kernel | bf16 GEMV |
| Gate+Up projection | 1 x 27648 x 5120 | gemmk1_kernel / gemmSN_NN_kernel | bf16 GEMV |
| Down projection | 1 x 5120 x 13824 | gemmk1_kernel / gemmSN_NN_kernel | bf16 GEMV |
| LM head | 1 x 248320 x 5120 | gemmk1_kernel | bf16 GEMV |

At batch=1, cuBLAS dispatches to the **gemmk1_kernel** and **gemmSN** families,
not the large tile WGMMA GEMM kernels. These are specialized GEMV kernels using
thread-block-level reduction rather than tensor cores.

Relevant cublasLt kernels (sm_90):
- `gemmk1_kernel<l, __nv_bfloat16, 256, 5, ...>` -- 256 threads, unroll 5
- `gemmSN_NN_kernel<__nv_bfloat16, 256, 4, 2, 8, ...>` -- small-N specialized
- `gemvx` / `gemv` variants

### Quantized Variants

**INT8 GEMM (W8A8):**
- 626 kernels with signed char (`a` in mangling) in cublasLt sm_90
- `gemmk1_kernel<l,f,256,5,...,cublasGemvTensorStridedBatched<Ka>,...>` -- int8 input, float accumulator
- These use IMMA (integer tensor core MMA) on Hopper

**INT4/W4A16:** NOT present in cuBLAS. Must use:
- CUTLASS W4A16 kernels
- Marlin kernels (in vLLM)
- AWQ/GPTQ custom kernels

**FP8 GEMM:**
- `splitKreduce_kernel<..., fp8_e4m3, ...>` -- FP8 inference split-K
- Only 37 kernels each for E4M3 and E5M2 (limited coverage)
- FP8 GEMV not well-supported at batch=1

---

## 7. FlashInfer / FlashAttention Analysis

### FlashInfer (v0.6.6)
- **9,734 pre-compiled cubins** in flashinfer_cubin package
- **1.1 GB** total on disk
- Location: `/home/ubuntu/vllm-env/lib/python3.10/site-packages/flashinfer_cubin/`

**Breakdown by category:**
| Category | Count | Description |
|----------|-------|-------------|
| FMHA (attention) | 7,994 | Fused multi-head attention cubins |
| Deep-GEMM | 317 | FP8 grouped GEMM (MoE) |
| GEMM | 54 | CUTLASS-based GEMM |
| **Total** | **8,365** | (some overlap in directory counting) |

**SM Architecture Targets:**
| Target | Cubin Count |
|--------|------------|
| Sm100f (Blackwell) | 6,630 |
| Sm103a (Blackwell variant) | 1,114 |
| Sm100a (Blackwell variant) | 250 |

**Critical finding: FlashInfer 0.6.6 ships NO sm_90 (Hopper) pre-compiled cubins.**
All cubins target sm_100/sm_103 (Blackwell). On this GH200 (sm_90), FlashInfer
must JIT-compile at runtime or fall back to a different codepath.

**FlashInfer FMHA naming convention:**
```
fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128StaticKeepsAbForGen.cubin
                  ^input    ^output  ^head ^KV layout ^mask ^page ^seq mode ^tile  ^scheduling ^phase
```

### Flash-Attention (v2.7.4.post1)
- Installed system-wide at `/usr/lib/python3/dist-packages`
- Uses compiled .so extensions (not cubin files)

### FlashMLA (in vLLM)
- `_flashmla_C.abi3.so` and `_flashmla_extension_C.abi3.so`
- Multi-latent attention for DeepSeek-style models

---

## 8. Lithos Replacement Strategy

### What Lithos Should Replace and Why

#### Priority 1: GEMV Kernels (batch=1 decode) -- HIGHEST IMPACT
cuBLAS ships generic GEMV kernels (gemmk1, gemmSN) that:
- Cannot fuse with RMSNorm, RoPE, SiLU, or residual add
- Require separate kernel launches for each linear layer
- Do not exploit weight layout optimization for specific hidden_dim values

**Fusion opportunities:**
- RMSNorm + QKV projection (eliminates one global memory round-trip of 5120 * 2 bytes)
- Gate + Up projection + SiLU activation (eliminates 27648 * 2 byte intermediate)
- Down projection + residual add (eliminates 5120 * 2 byte store+load)

#### Priority 2: Attention Kernels -- ALREADY HANDLED BY FLASHINFER
FlashInfer/FlashAttention already replaces cuBLAS for attention. However:
- FlashInfer ships NO sm_90 cubins (only sm_100), requiring JIT on GH200
- Lithos could ship pre-compiled sm_90 attention kernels for faster cold-start

#### Priority 3: Quantized GEMM (W4A16, W8A8) -- MISSING FROM CUBLAS
cuBLAS has NO int4 kernels. Current W4A16 goes through Marlin/CUTLASS.
Lithos opportunity: fused dequant + GEMM + epilogue for:
- GPTQ-int4 quantized Qwen 3.5-27B
- AWQ-int4 quantized variants
- FP8 quantized variants with custom scaling

#### Priority 4: Prefill GEMM (batch > 1) -- LOWER PRIORITY
For large batch prefill, cuBLAS WGMMA kernels are already near-optimal (90%+ of
peak FLOPS). Lithos should only replace these if:
- Custom epilogue fusion is needed (bias + activation + quantize)
- Non-standard layouts are required
- Specific tile sizes for model dimensions would improve utilization

### Size Comparison
| Component | Size | Kernels |
|-----------|------|---------|
| cuBLAS | 110 MB | ~10,725 (sm_90) |
| cuBLASLt | 687 MB | ~10,525 (sm_90) |
| FlashInfer cubins | 1.1 GB | 9,734 |
| **Lithos target** | **< 10 MB** | **~20 kernels** |

Lithos needs only ~20 purpose-built kernels for Qwen 3.5-27B on GH200, versus
NVIDIA shipping 80,000+ generic variants. This is the fundamental advantage:
model-specific kernels eliminate the combinatorial explosion.

---

## Raw Data Files

- `cublas-elf-list.txt` -- All 1,610 ELF cubins in libcublas
- `cublasLt-elf-list.txt` -- All 5,732 ELF cubins in libcublasLt
- `cublas-ptx-list.txt` -- 192 PTX modules in libcublas
- `cublasLt-ptx-list.txt` -- 136 PTX modules in libcublasLt
- `cublas-sm90-functions.txt` -- 200 sm_90 kernel functions in libcublas
- `cublasLt-sm90-functions.txt` -- 10,525 sm_90 kernel functions in libcublasLt
- `cublas-sm90-sass-sample.txt` -- First 500 lines of sm_90 SASS from libcublas
- `sass-sample-detail.txt` -- Detailed SASS of splitKreduce kernel body
- `wgmma-grep.txt` -- WGMMA/HMMA function hits in cublasLt
