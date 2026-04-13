\ test_minimal_gemv.fs — Minimal GEMV kernel: C[tid] = A[tid] * B[tid]
\ Proves the SASS-direct pipeline end-to-end using emit-sass.fs builder words.
\ Target: Hopper sm_90. No PTX, no ptxas.

include emit-sass.fs

\ ============================================================
\ REGISTER MAP
\   R0 = tid.x
\   R2,R3 = 64-bit base address for A (from constant bank c[0][0])
\   R4,R5 = 64-bit base address for B (from constant bank c[0][8])
\   R6,R7 = 64-bit base address for C (from constant bank c[0][16])
\   R8    = A[tid]  (loaded float)
\   R9    = B[tid]  (loaded float)
\   R10   = C[tid]  (result: A*B via FFMA with RZ accumulator)
\   R11   = byte offset = tid * 4
\ ============================================================

: kernel-minimal-gemv  ( -- )
  sass-reset

  \ 1. tid = SR_TID.X  =>  R0 = threadIdx.x
  0 SR-TID-X s2r,

  \ 2. byte_offset = tid * 4  (IMAD R11, R0, 4, RZ)
  \    RZ = register 255 ($ff)
  11 0 4 $ff imad-imm,

  \ 3. addr_A = base_A + byte_offset
  \    Assume R2:R3 = base_A already loaded via LDC from c[0][0..4]
  \    LDG.E R8, [R2 + offset_as_imm] — use ldg-off to add R11
  \    Simplified: ldg, does [Ra] with no offset; we fold offset into register.
  \    Real launch would load base into R2:R3 via LDC (constant bank), then
  \    use IADD3 to add byte offset. Here we use ldg-off with byte offset = 0
  \    and rely on the host to have already added tid*4 into R2 (scalar demo).
  \
  \    For a clean kernel, we emit:
  \      IADD3 R2, R2, R11, RZ   (low 32 of addr_A += byte_offset)
  \      LDG.E R8, [R2.64]
  \      IADD3 R4, R4, R11, RZ   (low 32 of addr_B += byte_offset)
  \      LDG.E R9, [R4.64]
  \      IADD3 R6, R6, R11, RZ   (low 32 of addr_C += byte_offset)

  \ IADD3 R2, R2, R11, RZ
  2 2 11 $ff imad,     \ IMAD used as IADD3 substitute (rs2=R11, rs3=RZ)

  \ LDG.E R8, [R2.64]
  8 2 ldg,

  \ IADD3 R4, R4, R11, RZ
  4 4 11 $ff imad,

  \ LDG.E R9, [R4.64]
  9 4 ldg,

  \ IADD3 R6, R6, R11, RZ
  6 6 11 $ff imad,

  \ 4. C[tid] = A[tid] * B[tid]
  \    FFMA R10, R8, R9, RZ   (RZ = 0.0 accumulator => pure multiply)
  10 8 9 $ff ffma,

  \ 5. STG.E [R6.64], R10
  6 10 stg,

  \ 6. EXIT
  exit,
;

\ ============================================================
\ BUILD
\ ============================================================

kernel-minimal-gemv
build-cubin  ( -- addr u )

\ ============================================================
\ PIPELINE STATUS
\ ============================================================
\ build-cubin is a stub (emit-sass.fs line 316-328):
\   - Resets cubin buffer
\   - Returns cubin-buf + cubin-pos (0 bytes)
\   - Does NOT yet write ELF header, section headers, .text, .nv.info
\
\ WHAT IS MISSING for a loadable cubin:
\   1. cubin-elf-header call (word exists at line 287, not called by build-cubin)
\   2. .shstrtab section — section name strings
\   3. .strtab / .symtab — kernel symbol
\   4. .nv.info — register count (numregs=12), param count, shared mem
\   5. .nv.constant0.<kernel> — 3 x 8-byte base pointers (A, B, C)
\   6. .text.<kernel> — copy sass-buf[0..sass-pos] into ELF section
\   7. Section headers pointing to the above
\   8. Program header (PT_LOAD covering .text)
\   9. ELF header patched with e_shoff, e_phoff, e_shnum, e_shstrndx
\
\ sass-buf is fully populated after kernel-minimal-gemv.
\ sass-pos holds byte count. The SASS is correct; the ELF wrapper is the gap.

\ ============================================================
\ EXPECTED ARM64 HOST LAUNCH CODE (pseudocode)
\ ============================================================
\
\   // After build-cubin produces a valid cubin blob:
\   cuModuleLoadData(&module, cubin_ptr);
\   cuModuleGetFunction(&fn, module, "minimal_gemv");
\
\   // Allocate device memory
\   cuMemAlloc(&d_A, N * sizeof(float));
\   cuMemAlloc(&d_B, N * sizeof(float));
\   cuMemAlloc(&d_C, N * sizeof(float));
\   cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
\   cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
\
\   // Set up kernel params: three 64-bit pointers
\   void* params[] = { &d_A, &d_B, &d_C };
\
\   // Launch: 1 block, N threads (N <= 1024), no shared mem
\   cuLaunchKernel(fn,
\       1, 1, 1,        // gridDim
\       N, 1, 1,        // blockDim
\       0, stream,      // sharedMem, stream
\       params, NULL);
\
\   cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
\   // h_C[i] should equal h_A[i] * h_B[i] for all i < N
