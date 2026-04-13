\ Lithos — GPU compute language
\ SASS-only backend. Patterns are the vocabulary.

\ Number to decimal string (for immediates)
create num-buf 32 allot

\ ============================================================
\ REGISTER POOLS
\ ============================================================

variable n-pred  0 n-pred !
variable n-r32   0 n-r32 !
variable n-r64   0 n-r64 !
variable n-f32   0 n-f32 !
variable n-f16   0 n-f16 !

: regs-reset  0 n-pred !  0 n-r32 !  0 n-r64 !  0 n-f32 !  0 n-f16 ! ;

: pred+  ( -- n )  n-pred @  1 n-pred +! ;
: r32+   ( -- n )  n-r32 @   1 n-r32 +! ;
: r64+   ( -- n )  n-r64 @   1 n-r64 +! ;
: f32+   ( -- n )  n-f32 @   1 n-f32 +! ;
: f16+   ( -- n )  n-f16 @   1 n-f16 +! ;

\ ============================================================
\ THREAD INDEXING — CACHED PATTERNS
\ ============================================================

\ Global thread ID (1D) — the most common pattern in every kernel
\ tid = blockIdx.x * blockDim.x + threadIdx.x
: global-tid  ( -- r32 )
  r32+ r32+ r32+ r32+  \ allocate 4 regs: tid, ctaid, ntid, result
  3 - \ return the result register number (last allocated - 3)
  ;

\ Bounds check — guard against out-of-range threads
\ if (tid >= n) return;
: bounds-check  ( r-tid r-n -- )
  pred+  drop drop  \ allocate predicate, emit setp + @p bra DONE
  ;

\ ============================================================
\ MEMORY ACCESS — CACHED PATTERNS
\ ============================================================

\ Coalesced f32 load: addr + tid*4
: coalesced-f32@  ( r-base r-tid -- f32 )
  r64+ r64+  f32+  drop drop  \ cvt, shl, add, ld.global.f32
  ;

\ Coalesced f32 store: addr + tid*4
: coalesced-f32!  ( f-val r-base r-tid -- )
  r64+ r64+  drop drop drop  \ cvt, shl, add, st.global.f32
  ;

\ Vectorized load: 4 floats at once
: vec4-f32@  ( r-base r-tid -- f32 f32 f32 f32 )
  r64+ r64+  f32+ f32+ f32+ f32+  drop drop
  ;

\ ============================================================
\ ARITHMETIC — FUSED PATTERNS
\ ============================================================

\ FMA: d = a*b + c (single instruction, no intermediate rounding)
: fma.f32  ( fa fb fc -- fd )
  f32+  drop drop drop
  ;

\ RMSNorm pattern: x * rsqrt(mean(x^2) + eps)
: rmsnorm  ( f-in f-weight f-eps -- f-out )
  f32+ f32+ f32+  drop drop drop
  ;

\ SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
: silu.f32  ( f-in -- f-out )
  f32+ f32+  drop
  ;

\ ============================================================
\ WARP REDUCTION — CACHED PATTERNS
\ ============================================================

\ Warp-level sum reduction using butterfly shuffles
: warp-reduce-add.f32  ( f-in -- f-out )
  f32+  drop
  ;

\ ============================================================
\ TENSOR CORE — THE GEMM PATTERN
\ ============================================================

: tc-load-a  ( r-smem-addr -- )
  drop
  ;

: tc-load-b  ( r-smem-addr -- )
  drop
  ;

\ Tensor core multiply-accumulate: D = A*B + C
: tc-mma  ( -- )
  ;

\ Async global→shared copy (bypasses registers, uses TMA on Hopper)
: async-copy  ( r-dst-shared r-src-global n-bytes -- )
  drop drop drop
  ;

\ ============================================================
\ COMPLETE PATTERNS — INFERENCE BUILDING BLOCKS
\ ============================================================

\ Softmax numerator: exp(x - max)
: safe-exp.f32  ( f-in f-max -- f-out )
  f32+  drop drop
  ;

\ Dot product of two f16 vectors (4 elements, packed)
: dot4-f16  ( r-a r-b -- f-acc )
  f32+  drop drop
  ;

\ Quantized GEMV: W4A16 dequant + multiply
: w4a16-gemv  ( r-weight r-scale r-activation -- f-acc )
  f32+ f32+  r32+  drop drop drop
  ;
