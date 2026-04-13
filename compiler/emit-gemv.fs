\ emit-gemv.fs — GEMV/project GPU instruction pattern emitter for Lithos compiler
\
\ Emits the complete GPTQ W4A16 matrix-vector multiply kernel body.
\ This is the heaviest primitive: ~40 sm90 instructions per inner loop
\ iteration (8 nibbles x 5 ops), memory-bandwidth bound on Hopper.
\
\ Pattern: y[row] = sum_k( dequant(W_packed[row,k]) * x[k] )
\   W is 4-bit quantized (8 nibbles per u32), one scale per 128 weights
\   (= 16 packed u32s). Zero point = 8 (subtracted as -8.0 in dequant).
\
\ Launch config: gridDim.x = N (one block per output row)
\                blockDim.x = 256 (threads per block)
\
\ Register budget (inner loop): ~14 registers
\   R_row(1) R_tid(1) R_kp(1) R_bound(1) R_rowbase(1) R_packed(1)
\   R_scale(1) R_xbase(1) R_acc(1) R_nib(1) R_xval(1) R_neg8s(1)
\   R_tmp(1) R_addr(1)
\ Total kernel: ~20 registers (adds N, K_packed, warp/lane, smem scratch)

\ ============================================================
\ DEPENDENCIES
\ ============================================================
\ gpu/emit.fs provides:
\   fadd, ffma, s2r, imad, imad-imm, ldg, ldg-off, stg, exit,
\   bra, bra-pred, isetp-ge, isetp-lt, iadd3, mov-imm,
\   shf-r-imm, lop3-and-imm, i2f-s32-f32, dequant-nibble,
\   warp-reduce, shfl-bfly,
\   sts, lds, bar-sync,
\   sass-pos, sass-buf, sinst, nop,
\   track-rd, max-reg-used,
\   SR-TID-X, SR-CTAID-X
\
\ emit-reduce.fs (parallel worker — not yet written) expected interface:
\   emit-warp-reduce  ( partial-reg tmp-reg -- )
\     Emits 5x SHFL.BFLY + FADD butterfly reduction within a warp.
\     After return, lane 0 holds the warp sum in partial-reg.
\   emit-cross-warp-reduce  ( partial-reg warp-id-reg smem-off-reg tmp-reg -- )
\     Emits STS partial to smem, BAR.SYNC, first-warp tree reduce via LDS,
\     and broadcast of final result. After return, thread 0 holds the sum.
\
\ parser.fs (expected) provides:
\   rreg+ — allocate next general register, return its index

s" /home/ubuntu/lithos/gpu/emit.fs" included

\ ============================================================
\ CONSTANTS
\ ============================================================

\ IEEE 754 bit patterns
$00000000 constant FP32-ZERO      \ 0.0
$c1000000 constant FP32-NEG8      \ -8.0  (GPTQ W4A16 zero point)
$ff        constant RZ-G          \ zero register = R255

\ Block dimension (threads per block) for GEMV kernel
256 constant GEMV-BLOCKDIM

\ ============================================================
\ REGISTER ALLOCATOR (simple bump allocator)
\ ============================================================
\ The parser's rreg+ is not yet available. Provide a local version
\ so this file is self-contained for testing. If rreg+ is already
\ defined (from parser.fs), these will harmlessly shadow it.

variable gemv-next-reg   0 gemv-next-reg !

: gemv-reg-reset  ( -- )  0 gemv-next-reg ! ;

: gemv-reg+  ( -- reg )
  gemv-next-reg @ dup 1+ gemv-next-reg ! ;

\ ============================================================
\ STATE VARIABLES  (module-level, shared across emit words)
\ ============================================================
\ Declared once at compile time. Each emit-gemv call writes fresh values.

\ Dequant-8 scratch
variable dq-nib      \ nibble extraction scratch register
variable dq-xval     \ x element value register
variable dq-neg8s    \ precomputed -8.0 * scale register
variable dq-acc      \ accumulator register
variable dq-xaddr    \ x base address register
variable dq-scale    \ scale register
variable dq-packed   \ packed u32 register

\ Cross-warp reduction scratch
variable xw-tid      variable xw-partial
variable xw-tmp      variable xw-smoff
variable xw-laneid   variable xw-warpid

\ Main emit-gemv state
variable gv-N        variable gv-K
variable gv-y        variable gv-x
variable gv-scales   variable gv-W
variable gv-tid      variable gv-row
variable gv-Kpacked  variable gv-rowbase
variable gv-acc      variable gv-kp
variable gv-wpidx    variable gv-waddr
variable gv-packed   variable gv-scale
variable gv-xaddr

\ Branch patching addresses
variable gv-exit-bra   \ sass-pos of the bounds-check forward branch
variable gv-loop-top   \ sass-pos of the K-loop top
variable gv-loop-bra   \ sass-pos of the loop-exit forward branch

\ ============================================================
\ BRANCH PATCHING HELPER
\ ============================================================
\ patch-bra-offset ( target-pos bra-pos -- )
\ Patches the 32-bit offset field at bytes [4..7] of the instruction
\ at bra-pos. Offset = target_pos - (bra_pos + 16).

: patch-bra-offset  ( target-pos bra-pos -- )
  \ Compute byte offset = target - (bra_pos + 16), then convert to dwords
  dup >r 16 + - r>               \ stack: byte-offset bra_pos
  swap 4 / swap                  \ stack: dword-offset bra_pos
  \ Address of byte 4 in the instruction at bra_pos
  4 + sass-buf + >r              \ stack: dword-offset;  R: addr
  dup        r@ c!               \ byte 0
  8 rshift
  dup        r@ 1+ c!            \ byte 1
  8 rshift
  dup        r@ 2 + c!           \ byte 2
  8 rshift   r> 3 + c! ;         \ byte 3

\ ============================================================
\ INTEGER MULTIPLY BY IMMEDIATE HELPER
\ ============================================================
\ imul-imm, ( rd rs1 imm32 -- )
\ Emits: MOV Rtmp, imm32; IMAD Rd, Rs1, Rtmp, RZ
\ Uses a scratch register from the bump allocator.
\ This is needed because imad-imm, in gpu/emit.fs is actually
\ IMAD.MOV.U32 (a move-immediate), not a multiply.

: imul-imm,  ( rd rs1 imm32 -- )
  gemv-reg+ >r                  \ allocate Rtmp
  r@ swap mov-imm,              \ MOV Rtmp, imm32
  r> RZ-G imad, ;               \ IMAD Rd, Rs1, Rtmp, RZ

\ ============================================================
\ GEMV PROLOGUE — row index + thread ID setup
\ ============================================================
\ emit-gemv-prologue ( -- row-reg tid-reg )
\   S2R  Rrow, SR_CTAID.X     ; row = blockIdx.x (one block per row)
\   S2R  Rtid, SR_TID.X       ; tid = threadIdx.x (0..255)
\ Returns register indices for row and tid.

: emit-gemv-prologue  ( -- row-reg tid-reg )
  gemv-reg+                     \ Rrow
  dup SR-CTAID-X s2r,           \ S2R Rrow, SR_CTAID.X
  gemv-reg+                     \ Rtid
  dup SR-TID-X s2r, ;           \ S2R Rtid, SR_TID.X

\ ============================================================
\ DEQUANT-8 — 8-nibble dequant+FMA unroll (the hot inner loop)
\ ============================================================
\ emit-dequant-8 ( packed-reg scale-reg xaddr-reg acc-reg -- acc-reg )
\
\ For each nibble i (0..7) of the packed u32, emits:
\   If i > 0: SHF.R.U32 Rnib, Rpacked, i*4   ; shift to nibble
\   LOP3      Rnib, Rsrc, 0xF                 ; mask low 4 bits
\   I2F.F32   Rnib, Rnib                      ; int -> float
\   FFMA      Rnib, Rnib, Rscale, Rneg8s      ; (nibble-8)*scale
\   LDG       Rxval, [Rxaddr + i*4]           ; load x element
\   FFMA      Racc, Rnib, Rxval, Racc         ; accumulate
\
\ Optimization: pre-compute neg8_scale = -8.0 * scale once per packed
\ word, then FFMA(nib_f, nib_f, scale, neg8_scale) folds the zero-point
\ subtract and scale multiply into one instruction per nibble.
\
\ Total: 1 (neg8s setup) + 8 * (1 shift + 1 mask + 1 i2f + 1 ffma_dq
\         + 1 ldg + 1 ffma_acc) = 1 + 8*6 = 49 instructions.
\ Nibble 0 skips the shift: 1 + 47 + 5 = 48 instructions.

: emit-dequant-8  ( packed-reg scale-reg xaddr-reg acc-reg -- acc-reg )
  \ Store arguments into module variables
  dq-acc !  dq-xaddr !  dq-scale !  dq-packed !

  \ Allocate scratch registers (reused across all 8 nibbles)
  gemv-reg+ dq-nib !
  gemv-reg+ dq-xval !
  gemv-reg+ dq-neg8s !

  \ Pre-compute neg8_scale = -8.0 * scale + 0.0
  \ MOV Rneg8s, 0xC1000000 (-8.0 in IEEE 754)
  dq-neg8s @ FP32-NEG8 mov-imm,
  \ FFMA Rneg8s, Rneg8s, Rscale, RZ  =>  neg8s = (-8.0) * scale
  dq-neg8s @ dq-neg8s @ dq-scale @ RZ-G ffma,

  \ 8-nibble compile-time unrolled loop
  8 0 do
    \ --- Extract nibble i ---
    i 0 = if
      \ Nibble 0: bits [3:0] already in low position, just mask
      dq-nib @ dq-packed @ $f lop3-and-imm,
    else
      \ SHF.R Rnib, Rpacked, i*4  (shift right to bring nibble i to low bits)
      dq-nib @ dq-packed @ i 4 * shf-r-imm,
      \ LOP3 Rnib, Rnib, 0xF  (mask to 4 bits)
      dq-nib @ dq-nib @ $f lop3-and-imm,
    then

    \ --- I2F Rnib, Rnib (signed int to float) ---
    dq-nib @ dq-nib @ i2f-s32-f32,

    \ --- FFMA Rnib, Rnib, Rscale, Rneg8s ---
    \ Computes: float(nibble) * scale + (-8.0 * scale) = (nibble - 8) * scale
    dq-nib @ dq-nib @ dq-scale @ dq-neg8s @ ffma,

    \ --- LDG Rxval, [Rxaddr + i*4] ---
    \ Load x[k_p*8 + i] at byte offset i*4 from x base address
    i 0 = if
      dq-xval @ dq-xaddr @ ldg,
    else
      dq-xval @ dq-xaddr @ i 4 * ldg-off,
    then

    \ --- FFMA acc, dequant_val, x_val, acc ---
    dq-acc @ dq-nib @ dq-xval @ dq-acc @ ffma,
  loop

  \ Return accumulator register index
  dq-acc @ ;

\ ============================================================
\ WARP REDUCTION (wraps gpu/emit.fs warp-reduce,)
\ ============================================================
\ emit-warp-reduce ( partial-reg tmp-reg -- )
\ 5x SHFL.BFLY + FADD butterfly reduction across 32 lanes.
\ Lane 0 of each warp holds the warp sum afterward.

: emit-warp-reduce  ( partial-reg tmp-reg -- )
  warp-reduce, ;

\ ============================================================
\ CROSS-WARP REDUCTION via shared memory
\ ============================================================
\ emit-cross-warp-reduce ( partial-reg tid-reg -- )
\
\ After intra-warp reduction, lane 0 of each warp has a partial sum.
\ 256 threads / 32 lanes = 8 warps. Reduces to a single value in
\ thread 0's partial-reg.
\
\ Algorithm:
\   1. warp_id = tid >> 5;  lane_id = tid & 0x1F
\   2. if lane_id == 0: STS smem[warp_id*4], partial
\   3. BAR.SYNC 0
\   4. if tid < 8: LDS partial, smem[tid*4]  else: MOV partial, 0.0
\   5. Warp butterfly reduction (32 lanes; 8 real + 24 zeros)
\   6. Thread 0 has final sum in partial-reg.

: emit-cross-warp-reduce  ( partial-reg tid-reg -- )
  xw-tid !  xw-partial !

  \ Allocate scratch registers
  gemv-reg+ xw-warpid !
  gemv-reg+ xw-laneid !
  gemv-reg+ xw-smoff !
  gemv-reg+ xw-tmp !

  \ warp_id = tid >> 5
  xw-warpid @ xw-tid @ 5 shf-r-imm,
  \ lane_id = tid & 0x1F
  xw-laneid @ xw-tid @ $1f lop3-and-imm,
  \ smem_off = warp_id * 4  (byte offset for f32 store)
  xw-smoff @ xw-warpid @ 4 imul-imm,

  \ --- Lane 0 of each warp stores to shared memory ---
  \ ISETP.GE P0, lane_id, 1  =>  P0 = (lane_id >= 1) = not lane 0
  xw-tmp @ 1 mov-imm,
  0 xw-laneid @ xw-tmp @ isetp-ge,
  \ @P0 BRA +16  (skip one instruction: the STS)
  16 0 bra-pred,
  \ STS [smem_off], partial
  xw-smoff @ xw-partial @ 0 sts,

  \ --- Barrier: all threads in block synchronize ---
  0 bar-sync,

  \ --- Thread tid < 8 loads from smem; tid >= 8 gets 0.0 ---
  \ Compute load_off = tid * 4
  xw-smoff @ xw-tid @ 4 imul-imm,
  \ ISETP.GE P0, tid, 8
  xw-tmp @ 8 mov-imm,
  0 xw-tid @ xw-tmp @ isetp-ge,
  \ @P0 BRA +32  (skip LDS + BRA = 2 instructions if tid >= 8)
  32 0 bra-pred,
  \ LDS partial, smem[tid*4]  (tid < 8 only)
  xw-partial @ xw-smoff @ 0 lds,
  \ Unconditional BRA +16 to skip MOV (tid < 8 already loaded via LDS)
  16 bra,
  \ MOV partial, 0.0  (reached only by tid >= 8 that branched over LDS+BRA)
  xw-partial @ FP32-ZERO mov-imm,

  \ --- Final warp reduction (all 32 lanes of warp 0 participate) ---
  \ Lanes 8-31 hold 0.0, contributing nothing to the sum.
  xw-partial @ xw-tmp @ emit-warp-reduce

  \ Result: thread 0 holds final sum in xw-partial register.
  ;

\ ============================================================
\ MAIN EMITTER: emit-gemv
\ ============================================================
\ emit-gemv ( W-reg scales-reg x-reg y-reg K-val N-val -- )
\
\ Emits the complete GEMV kernel body for GPTQ W4A16 projection.
\ W-reg, scales-reg, x-reg, y-reg: register indices holding base pointers
\   (loaded from kernel params by the caller / compiler driver).
\ K-val, N-val: compile-time integer constants for matrix dimensions.
\   K = number of input elements (must be multiple of 8).
\   N = number of output rows.

: emit-gemv  ( W-reg scales-reg x-reg y-reg K-val N-val -- )
  gv-N !  gv-K !
  gv-y !  gv-x !  gv-scales !  gv-W !

  \ Reset register allocator
  gemv-reg-reset

  \ ==== Step 1: Prologue — row index + thread ID ====
  emit-gemv-prologue             \ ( -- row-reg tid-reg )
  gv-tid !  gv-row !

  \ ==== Step 2: Bounds check — if row >= N, branch to EXIT ====
  gemv-reg+ >r
  r@ gv-N @ mov-imm,            \ Rtmp = N  (compile-time constant)
  0 gv-row @ r> isetp-ge,       \ P0 = (row >= N)
  sass-pos @ gv-exit-bra !      \ save BRA position for later patching
  0 0 bra-pred,                  \ @P0 BRA +0  (placeholder offset)

  \ ==== Step 3: K_packed = K / 8  (compile-time constant) ====
  gemv-reg+
  dup gv-K @ 8 / mov-imm,       \ R_Kpacked = K/8
  gv-Kpacked !

  \ ==== Step 4: row_base = row * K_packed ====
  \ IMAD row_base, row, K_packed, RZ  =>  row_base = row * K_packed + 0
  gemv-reg+
  dup gv-row @ gv-Kpacked @ RZ-G imad,
  gv-rowbase !

  \ ==== Step 5: Initialize accumulator to 0.0 ====
  gemv-reg+
  dup FP32-ZERO mov-imm,
  gv-acc !

  \ ==== Step 6: Copy tid into loop counter k_p ====
  \ IADD3 k_p, tid, RZ, RZ  (k_p = tid + 0 + 0 = tid)
  gemv-reg+
  dup gv-tid @ RZ-G RZ-G iadd3,
  gv-kp !

  \ ==== Step 7: K-loop top ====
  sass-pos @ gv-loop-top !       \ record loop-top position

  \ Bounds check: P0 = (k_p >= K_packed) => exit loop
  0 gv-kp @ gv-Kpacked @ isetp-ge,
  sass-pos @ gv-loop-bra !      \ save forward-branch for patching
  0 0 bra-pred,                  \ @P0 BRA loop_exit (placeholder)

  \ ==== Step 8: Address computation for this iteration ====

  \ wp_idx = row_base + k_p  (packed word index into weight matrix)
  gemv-reg+
  dup gv-rowbase @ gv-kp @ RZ-G iadd3,
  gv-wpidx !

  \ W_addr = W_ptr + wp_idx * 4  (byte address)
  gemv-reg+ >r
  r@ gv-wpidx @ 4 imul-imm,     \ Rwoff = wp_idx * 4
  gemv-reg+
  dup r> gv-W @ RZ-G iadd3,     \ Rwaddr = W + woff
  \ IADD3 args: (rd rs1 rs2 rs3) => rd = rs1 + rs2 + rs3
  \ Here: Rwaddr = Rwoff + W_ptr + RZ
  \ Note: iadd3 takes ( rd rs1 rs2 rs3 -- ), result = rs1+rs2+rs3
  gv-waddr !

  \ LDG packed, [Rwaddr]
  gemv-reg+
  dup gv-waddr @ ldg,
  gv-packed !

  \ Scale address: scales_ptr + (wp_idx >> 4) * 4
  \ Group index = wp_idx / 16 = wp_idx >> 4
  gemv-reg+                      \ Rgrp
  dup gv-wpidx @ 4 shf-r-imm,  \ Rgrp = wp_idx >> 4
  gemv-reg+ >r                  \ Rgrpoff on r-stack
  r@ swap 4 imul-imm,           \ Rgrpoff = Rgrp * 4; IMAD(Rgrpoff, Rgrp, 4, RZ)
  gemv-reg+
  dup r> gv-scales @ RZ-G iadd3, \ Rsaddr = grpoff + scales + 0
  gemv-reg+
  dup over ldg,                  \ Rscale = LDG [Rsaddr]
  nip
  gv-scale !

  \ x_addr = x_ptr + k_p * 32  (byte address: k_p packs 8 floats, 8*4=32 bytes)
  gemv-reg+ >r
  r@ gv-kp @ 32 imul-imm,       \ Rxoff = k_p * 32
  gemv-reg+
  dup r> gv-x @ RZ-G iadd3,     \ Rxaddr = x + xoff
  gv-xaddr !

  \ ==== Step 9: 8x dequant + FMA unroll ====
  gv-packed @ gv-scale @ gv-xaddr @ gv-acc @
  emit-dequant-8
  gv-acc !

  \ ==== Step 10: Loop increment + branch back ====
  \ k_p += 256 (GEMV-BLOCKDIM)
  gemv-reg+ >r
  r@ GEMV-BLOCKDIM mov-imm,
  gv-kp @ gv-kp @ r> RZ-G iadd3,

  \ BRA loop_top  (backwards branch)
  \ offset = loop_top - (current_pos + 16)
  gv-loop-top @ sass-pos @ 16 + -
  bra,

  \ ==== Patch loop-exit forward branch ====
  sass-pos @ gv-loop-bra @ patch-bra-offset

  \ ==== Step 11: Warp reduction ====
  gemv-reg+ >r
  gv-acc @ r> emit-warp-reduce

  \ ==== Step 12: Cross-warp reduction via shared memory ====
  gv-acc @ gv-tid @ emit-cross-warp-reduce

  \ ==== Step 13: Thread 0 stores y[row] ====
  \ Compute y address unconditionally (harmless for non-storing threads).
  \ y_addr = y_ptr + row * 4
  gemv-reg+ >r
  r@ gv-row @ 4 imul-imm,       \ Ryoff = row * 4
  gemv-reg+
  dup r> gv-y @ RZ-G iadd3,     \ Ryaddr = y + yoff

  \ Guard the store: if tid >= 1, skip STG
  gemv-reg+ >r
  r@ 1 mov-imm,
  0 gv-tid @ r> isetp-ge,       \ P0 = (tid >= 1)
  16 0 bra-pred,                 \ @P0 BRA +16 (skip STG)
  \ STG [Ryaddr], acc  (only thread 0 executes this)
  gv-acc @ stg,

  \ ==== Patch bounds-check exit branch ====
  sass-pos @ gv-exit-bra @ patch-bra-offset

  \ ==== EXIT ====
  exit, ;

\ ============================================================
\ CONVENIENCE: emit-gemv-kernel
\ ============================================================
\ Full kernel emission with buffer reset.
\ emit-gemv-kernel ( W-reg scales-reg x-reg y-reg K N -- )

: emit-gemv-kernel  ( W-reg scales-reg x-reg y-reg K N -- )
  sass-reset
  gemv-reg-reset
  emit-gemv ;

\ ============================================================
\ REGISTER BUDGET SUMMARY
\ ============================================================
\ After emit-gemv-kernel completes, max-reg-used @ holds the high-water
\ mark. Expected allocation (bump allocator, no reuse):
\
\   Prologue:      R0 (row), R1 (tid)                           = 2
\   Bounds:        R2 (N constant)                               = 1
\   Loop setup:    R3 (K_packed), R4 (row_base), R5 (acc), R6 (k_p)  = 4
\   Addresses:     R7 (wp_idx), R8 (w_off), R9 (w_addr),
\                  R10 (packed), R11 (grp), R12 (grp_off),
\                  R13 (scale_addr), R14 (scale),
\                  R15 (x_off), R16 (x_addr)                    = 10
\   Dequant:       R17 (nib), R18 (xval), R19 (neg8_scale)      = 3
\   Loop stride:   R20 (step)                                    = 1
\   Warp reduce:   R21 (tmp)                                     = 1
\   Cross-warp:    R22 (warp_id), R23 (lane_id), R24 (smem_off),
\                  R25 (xw_tmp)                                  = 4
\   Store:         R26 (one), R27 (y_off), R28 (y_addr)         = 3
\   ---------------------------------------------------------------
\   Total:         ~29 registers (bump, no reuse)
\
\ With register reuse (rewriting dead regs), this compresses to ~18-20.
\ Hopper budget: 255 architectural registers. Occupancy target: <=32
\ registers per thread for 8 warps/SM.
\
\ Missing encoders (documented for future work):
\   - LDC / ULDC for kernel parameter loading (params loaded by caller)
\   - SHF.L (left shift) — available in emit-elementwise.fs if included
\   - FMUL standalone — available in emit-elementwise.fs if included
