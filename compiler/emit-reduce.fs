\ emit-reduce.fs — Reduction (sum) SASS pattern emitter for Lithos compiler
\
\ Emits the SASS instruction sequence for the `+` (reduce) primitive.
\ This is the ONLY primitive that needs shared memory, barriers, and
\ warp shuffles.  Every other use of shuffles/smem/barriers in DeltaNet
\ comes through this pattern.
\
\ Public interface (reusable by emit-gemv.fs and emit-linalg.fs):
\   emit-declare-smem      ( n-words -- smem-base )
\   emit-warp-reduce       ( partial-reg tmp-reg -- )
\   emit-cross-warp-reduce ( partial-reg smem-base tid-reg -- result-reg )
\   emit-full-reduce       ( array-base-reg array-len idx-reg -- result-reg )
\   emit-isetp-ge-imm      ( pp rs1 imm -- )
\   emit-isetp-lt-imm      ( pp rs1 imm -- )
\
\ Load order: parser.fs must be loaded before this file (provides
\ freg+, rreg+, preg+, shmem-size).  This file includes the SASS
\ instruction encoders.

\ Pull in instruction encoders from the SASS backend.
\ Provides: shfl-bfly, fadd, sts, lds, bar-sync, isetp-ge, isetp-lt,
\           bra-pred, bra, mov-imm, iadd3, imad, ldg, s2r, lop3-and-imm,
\           shf-r-imm, warp-reduce, sinst, sass-pos, track-rd,
\           SR-TID-X, shmem-size.
s" /home/ubuntu/lithos/compiler/emit-sass.fs" included

\ ============================================================
\ SHARED MEMORY DECLARATION
\ ============================================================

\ Byte offset counter for shared memory.  Multiple allocations accumulate.
variable smem-next-offset  0 smem-next-offset !

: smem-offset-reset  ( -- )  0 smem-next-offset ! ;

\ emit-declare-smem ( n-words -- smem-base )
\ Reserves n-words * 4 bytes of shared memory.  Returns the byte offset
\ of this allocation.  Updates the cubin shmem-size high-water mark.
: emit-declare-smem  ( n-words -- smem-base )
  smem-next-offset @           \ -- n-words base
  swap 4 *                     \ -- base n-bytes
  smem-next-offset @ +         \ -- base new-offset
  dup smem-next-offset !
  shmem-size @ max shmem-size !
  ;                            \ -- base

\ ============================================================
\ PREDICATE HELPERS
\ ============================================================

\ emit-isetp-ge-imm ( pp rs1 imm -- )
\ Sets predicate pp when rs1 >= imm.  Loads imm into a temp register,
\ then emits ISETP.GE.  Allocates one scratch register.
: emit-isetp-ge-imm  ( pp rs1 imm -- )
  rreg+ dup >r                 \ -- pp rs1 imm tmp ; R: tmp
  swap mov-imm,                \ MOV tmp, imm ; -- pp rs1 ; R: tmp
  r> isetp-ge, ;               \ ISETP.GE pp, rs1, tmp

\ emit-isetp-lt-imm ( pp rs1 imm -- )
\ Sets predicate pp when rs1 < imm.  Same pattern with ISETP.LT.
: emit-isetp-lt-imm  ( pp rs1 imm -- )
  rreg+ dup >r
  swap mov-imm,
  r> isetp-lt, ;

\ ============================================================
\ INTRA-WARP REDUCTION  (5x SHFL.BFLY + FADD)
\ ============================================================

\ emit-warp-reduce ( partial-reg tmp-reg -- )
\ Emits 10 instructions: for each delta in {16,8,4,2,1}:
\   SHFL.BFLY tmp, partial, delta, 0x1f1f
\   FADD      partial, partial, tmp
\ After return partial-reg holds the warp sum.  Lane 0 is guaranteed
\ correct; all lanes hold the full sum due to butterfly topology.
\ Delegates to warp-reduce, from sass/emit-sass.fs.
: emit-warp-reduce  ( partial-reg tmp-reg -- )
  warp-reduce, ;

\ ============================================================
\ CROSS-WARP REDUCTION VIA SHARED MEMORY
\ ============================================================
\
\ Strategy (matches reduce.li lines 44-78):
\   1. lane 0 of each warp -> STS [smem + warp_id*4]
\   2. BAR.SYNC 0
\   3. First warp (tid<32): LDS from smem, butterfly reduce
\   4. Thread 0: STS result to smem[0]
\   5. BAR.SYNC 0
\   6. All threads: LDS broadcast from smem[0]

\ Scratch variables (avoid deep return-stack gymnastics)
variable xw-partial   variable xw-smem   variable xw-tid
variable xw-lane      variable xw-warp   variable xw-addr
variable xw-total     variable xw-tmp    variable xw-addr2
variable xw-result    variable xw-smreg

\ emit-cross-warp-reduce ( partial-reg smem-base tid-reg -- result-reg )
\ smem-base is a byte offset (immediate, not register).  tid-reg holds
\ the thread index register.  Returns a freshly allocated register that
\ holds the broadcast sum in every thread after the two barriers.
: emit-cross-warp-reduce  ( partial-reg smem-base tid-reg -- result-reg )
  xw-tid !  xw-smem !  xw-partial !

  \ Allocate scratch registers
  rreg+ xw-lane !              \ lane_id
  rreg+ xw-warp !              \ warp_id
  rreg+ xw-addr !              \ smem store address
  rreg+ xw-tmp !               \ general scratch
  rreg+ xw-addr2 !             \ smem load address (first warp)
  freg+ xw-total !             \ reduction accumulator for first warp
  freg+ xw-result !            \ broadcast result (all threads)

  \ ---- Compute lane_id and warp_id ----
  xw-lane @  xw-tid @  $1f  lop3-and-imm,    \ lane_id = tid & 31
  xw-warp @  xw-tid @  5  shf-r-imm,         \ warp_id = tid >> 5

  \ ---- addr = smem_base + warp_id * 4 ----
  xw-tmp @  4  mov-imm,
  rreg+ xw-smreg !
  xw-smreg @  xw-smem @  mov-imm,             \ smem_base_reg
  xw-addr @  xw-warp @  xw-tmp @  xw-smreg @  imad,

  \ ---- Step 1: lane 0 stores partial to smem[warp_id] ----
  preg+
  dup xw-lane @  1  emit-isetp-ge-imm          \ P = (lane_id >= 1)
  16  swap bra-pred,                            \ @P BRA +16 (skip STS)
  xw-addr @  xw-partial @  0  sts,             \ STS [addr+0], partial

  \ ---- Step 2: barrier ----
  0 bar-sync,

  \ ---- Step 3: first warp loads and reduces ----
  \ Instructions in first-warp section (for forward-branch offset):
  \   1  MOV total, 0.0
  \   2  ISETP.GE P, tid, 8  (via emit-isetp-ge-imm: MOV+ISETP = 2 insts)
  \   3  (MOV inside helper)
  \   4  @P BRA +64
  \   5  MOV tmp, 4
  \   6  MOV smreg, smem_base
  \   7  IMAD addr2, tid, tmp, smreg
  \   8  LDS total, [addr2+0]
  \   9-18  warp-reduce (10 instructions)
  \   19  ISETP.GE P, tid, 1  (MOV+ISETP = 2 insts)
  \   20  (MOV inside helper)
  \   21  @P BRA +16
  \   22  MOV smreg, smem_base
  \   23  STS [smreg+0], total
  \ = 23 instructions = 368 bytes
  preg+
  dup xw-tid @  32  emit-isetp-ge-imm           \ P = (tid >= 32)
  368 swap bra-pred,                             \ @P BRA +368

  \ Default: total = 0.0 (for warp0 threads with tid >= num_warps)
  xw-total @  0  mov-imm,

  \ Guard LDS: skip if tid >= 8 (8 warps for blockDim=256)
  preg+
  dup xw-tid @  8  emit-isetp-ge-imm            \ P = (tid >= 8)
  64 swap bra-pred,                              \ @P BRA +64 (skip 4 insts)

  \ Compute load address: addr2 = tid * 4 + smem_base
  xw-tmp @  4  mov-imm,
  xw-smreg @  xw-smem @  mov-imm,
  xw-addr2 @  xw-tid @  xw-tmp @  xw-smreg @  imad,
  \ LDS total, [addr2+0]
  xw-total @  xw-addr2 @  0  lds,

  \ Intra-warp reduce within first warp
  xw-total @  xw-tmp @  emit-warp-reduce

  \ Thread 0 stores final sum to smem[0]
  preg+
  dup xw-tid @  1  emit-isetp-ge-imm            \ P = (tid >= 1)
  32 swap bra-pred,                              \ @P BRA +32 (skip MOV+STS)
  xw-smreg @  xw-smem @  mov-imm,
  xw-smreg @  xw-total @  0  sts,               \ STS [smem+0], total

  \ ---- Step 4: second barrier ----
  0 bar-sync,

  \ ---- Step 5: all threads load broadcast result ----
  xw-smreg @  xw-smem @  mov-imm,
  xw-result @  xw-smreg @  0  lds,              \ LDS result, [smem+0]

  xw-result @ ;                                  \ -- result-reg

\ ============================================================
\ STRIDE LOOP ACCUMULATION
\ ============================================================

\ Scratch variables for stride loop
variable sl-base   variable sl-len    variable sl-idx
variable sl-part   variable sl-i      variable sl-tmp
variable sl-addr   variable sl-stride variable sl-lenreg
variable sl-pred

\ emit-stride-loop-accum ( array-base-reg array-len idx-reg -- partial-reg )
\ Emits a stride loop that accumulates elements from global memory.
\ array-base-reg: register holding base pointer
\ array-len: immediate (number of elements)
\ idx-reg: register holding thread's starting index (tid)
\ Returns register holding the per-thread partial sum.
\
\ Generated pattern (8 instructions per iteration):
\   MOV partial, 0.0
\   IADD3 i, idx, RZ, RZ        ; i = idx (register copy)
\   MOV stride, 256
\   MOV lenreg, array-len
\   loop_top:
\     ISETP.GE P, i, lenreg     ; bounds check
\     @P BRA loop_end            ; exit if i >= len
\     MOV tmp4, 4
\     IMAD addr, i, tmp4, base  ; addr = base + i*4
\     LDG val, [addr]           ; load element
\     FADD partial, partial, val ; accumulate
\     IADD3 i, i, stride, RZ   ; i += blockDim
\     BRA loop_top              ; back to top
\   loop_end:
: emit-stride-loop-accum  ( array-base-reg array-len idx-reg -- partial-reg )
  sl-idx !  sl-len !  sl-base !

  \ Allocate registers
  freg+ sl-part !              \ FP32 accumulator
  rreg+ sl-i !                 \ integer loop index
  freg+ sl-tmp !               \ loaded FP32 value
  rreg+ sl-addr !              \ computed address
  rreg+ sl-stride !            \ stride register (256)
  rreg+ sl-lenreg !            \ length register
  preg+ sl-pred !              \ loop predicate

  \ Initialize
  sl-part @  0  mov-imm,                  \ partial = 0.0
  sl-i @  sl-idx @  $ff  $ff  iadd3,     \ i = idx (RZ=0xff)
  sl-stride @  256  mov-imm,             \ stride = 256 (blockDim)
  sl-lenreg @  sl-len @  mov-imm,        \ lenreg = array_len

  \ ---- Loop top (record position for back-edge) ----
  sass-pos @                               \ -- loop-top-pos

  \ Bounds check: P = (i >= len)
  sl-pred @  sl-i @  sl-lenreg @  isetp-ge,

  \ @P BRA loop_end: skip 6 instructions (MOV+IMAD+LDG+FADD+IADD3+BRA = 96 bytes)
  96  sl-pred @  bra-pred,

  \ Compute address: addr = i * 4 + base
  rreg+ dup >r  4  mov-imm,               \ tmp4 = 4
  sl-addr @  sl-i @  r>  sl-base @  imad,

  \ Load element from global memory
  sl-tmp @  sl-addr @  ldg,

  \ Accumulate: partial += val
  sl-part @  sl-part @  sl-tmp @  fadd,

  \ Advance: i += stride
  sl-i @  sl-i @  sl-stride @  $ff  iadd3,

  \ Branch back to loop top (negative offset)
  sass-pos @ 16 +  swap -                 \ offset = loop_top - (here + 16)
  bra,

  sl-part @ ;                              \ -- partial-reg

\ ============================================================
\ FULL REDUCTION  (the `+` reduce primitive)
\ ============================================================

\ emit-full-reduce ( array-base-reg array-len idx-reg -- result-reg )
\ The all-in-one word that `+` (reduce) maps to.
\ Emits:
\   1. Stride loop accumulating partial sums per thread
\   2. Intra-warp butterfly reduction (32 lanes -> 1)
\   3. Cross-warp shared-memory reduction + broadcast
\ Returns a register holding the final sum in all threads.
: emit-full-reduce  ( array-base-reg array-len idx-reg -- result-reg )
  \ Declare 32 words of shared memory (supports up to 32 warps / 1024 threads)
  32 emit-declare-smem         \ -- base len idx smem-offset
  >r                           \ R: smem-offset

  \ Step 1: stride loop -> per-thread partial sum
  emit-stride-loop-accum       \ -- partial ; R: smem-offset

  \ Step 2: intra-warp butterfly reduction
  rreg+                        \ -- partial shfl-tmp
  over swap emit-warp-reduce   \ -- partial

  \ Step 3: cross-warp reduction (needs a fresh tid register)
  rreg+ dup SR-TID-X s2r,     \ -- partial tid
  r>                           \ -- partial tid smem-offset
  swap                         \ -- partial smem-offset tid
  emit-cross-warp-reduce       \ -- result-reg
  ;
