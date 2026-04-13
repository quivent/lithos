\ Lithos — GPU compute language
\ Emits PTX text into a buffer. Patterns are the vocabulary.

\ ============================================================
\ PTX TEXT BUFFER
\ ============================================================

262144 constant PTX-SIZE
create ptx-buf PTX-SIZE allot
variable ptx-pos  0 ptx-pos !

: ptx-reset  0 ptx-pos ! ;
: ptx+  ( addr u -- )  \ append string to ptx buffer
  dup >r  ptx-buf ptx-pos @ + swap move  r> ptx-pos +! ;
: ptx-c  ( c -- )  ptx-buf ptx-pos @ + c!  1 ptx-pos +! ;
: ptx-nl  10 ptx-c ;
: ptx$  ( -- addr u )  ptx-buf ptx-pos @ ;

\ Number to decimal string (for immediates)
create num-buf 32 allot
\ Convert non-negative integer to decimal string in num-buf.
\ Returns ( addr u ) pointing into num-buf. Writes at num-buf[31] downward.
variable ptx-num-len
: ptx-num  ( n -- addr u )
  0 ptx-num-len !
  dup 0= if
    drop
    [char] 0 num-buf 31 + c!
    num-buf 31 + 1 exit
  then
  num-buf 32 +   \ pointer after last slot
  swap           ( ptr n )
  begin dup 0> while
    10 /mod                     ( ptr rem quot )
    swap [char] 0 +             ( ptr quot digit )
    rot 1- tuck c!              ( quot ptr' )   \ ptr decremented, digit stored
    1 ptx-num-len +!
    swap                        ( ptr' quot )
  repeat
  drop                          ( ptr-first-digit )
  ptx-num-len @ ;

\ ============================================================
\ PTX HEADER
\ ============================================================

: ptx-header  ( -- )
  s" .version 8.0" ptx+ ptx-nl
  s" .target sm_90" ptx+ ptx-nl
  s" .address_size 64" ptx+ ptx-nl
  ptx-nl ;

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

: emit-regs  ( -- )
  n-pred @ 0> if  s" .reg .pred %p<" ptx+  n-pred @ ptx-num ptx+  s" >;" ptx+ ptx-nl  then
  n-r32 @  0> if  s" .reg .b32 %r<"  ptx+  n-r32 @  ptx-num ptx+  s" >;" ptx+ ptx-nl  then
  n-r64 @  0> if  s" .reg .b64 %rd<" ptx+  n-r64 @  ptx-num ptx+  s" >;" ptx+ ptx-nl  then
  n-f32 @  0> if  s" .reg .f32 %f<"  ptx+  n-f32 @  ptx-num ptx+  s" >;" ptx+ ptx-nl  then
  n-f16 @  0> if  s" .reg .b16 %h<"  ptx+  n-f16 @  ptx-num ptx+  s" >;" ptx+ ptx-nl  then ;

\ ============================================================
\ KERNEL STRUCTURE
\ ============================================================

: kernel{  ( addr u -- )  \ start kernel, name on stack as string
  s" .visible .entry " ptx+ ptx+ s"  (" ptx+ ptx-nl
  regs-reset ;

: param-u64  ( addr u -- )  s" .param .u64 " ptx+ ptx+ ;
: param-u32  ( addr u -- )  s" .param .u32 " ptx+ ptx+ ;
: param-f32  ( addr u -- )  s" .param .f32 " ptx+ ptx+ ;
: ,param  s" ," ptx+ ptx-nl ;
: )params  ptx-nl s" )" ptx+ ptx-nl s" {" ptx+ ptx-nl ;

: }kernel  s" exit;" ptx+ ptx-nl s" }" ptx+ ptx-nl ;

\ ============================================================
\ THREAD INDEXING — CACHED PATTERNS
\ ============================================================

\ Global thread ID (1D) — the most common pattern in every kernel
\ tid = blockIdx.x * blockDim.x + threadIdx.x
: global-tid  ( -- r32 )
  r32+ r32+ r32+ r32+  \ allocate 4 regs: tid, ctaid, ntid, result
  \ Emit after regs are declared:
  \ mov.u32 %r{tid}, %tid.x;
  \ mov.u32 %r{ctaid}, %ctaid.x;
  \ mov.u32 %r{ntid}, %ntid.x;
  \ mad.lo.s32 %r{result}, %r{ctaid}, %r{ntid}, %r{tid};
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
  \ ld.global.v4.f32 {%f...}, [addr]
  ;

\ ============================================================
\ ARITHMETIC — FUSED PATTERNS
\ ============================================================

\ FMA: d = a*b + c (single instruction, no intermediate rounding)
: fma.f32  ( fa fb fc -- fd )
  f32+  drop drop drop
  \ fma.rn.f32 %f{d}, %f{a}, %f{b}, %f{c}
  ;

\ RMSNorm pattern: x * rsqrt(mean(x^2) + eps)
\ This is multiple instructions fused into one word
: rmsnorm  ( f-in f-weight f-eps -- f-out )
  f32+ f32+ f32+  drop drop drop
  \ mul.f32 sq, in, in          (x^2)
  \ add.f32 acc, acc, sq        (accumulate)
  \ ... warp reduce ...
  \ rcp.approx.f32 inv, acc     (1/mean)
  \ sqrt.approx.f32 scale, inv  (rsqrt)
  \ mul.f32 out, in, scale
  \ mul.f32 out, out, weight
  ;

\ SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
: silu.f32  ( f-in -- f-out )
  f32+ f32+  drop
  \ neg.f32 t, in
  \ ex2.approx.f32 t, t    (2^(-x) ≈ e^(-x) with scale)
  \ add.f32 t, t, 1.0
  \ rcp.approx.f32 t, t
  \ mul.f32 out, in, t
  ;

\ ============================================================
\ WARP REDUCTION — CACHED PATTERNS
\ ============================================================

\ Warp-level sum reduction using butterfly shuffles
\ Takes a value in every lane, returns sum in lane 0
: warp-reduce-add.f32  ( f-in -- f-out )
  f32+  drop
  \ shfl.sync.bfly.b32 t, in, 16, 0x1f, 0xffffffff
  \ add.f32 in, in, t
  \ shfl.sync.bfly.b32 t, in, 8, 0x1f, 0xffffffff
  \ add.f32 in, in, t
  \ shfl.sync.bfly.b32 t, in, 4, 0x1f, 0xffffffff
  \ add.f32 in, in, t
  \ shfl.sync.bfly.b32 t, in, 2, 0x1f, 0xffffffff
  \ add.f32 in, in, t
  \ shfl.sync.bfly.b32 t, in, 1, 0x1f, 0xffffffff
  \ add.f32 in, in, t
  ;

\ ============================================================
\ TENSOR CORE — THE GEMM PATTERN
\ ============================================================

\ Load matrix tile from shared memory into tensor core registers
: tc-load-a  ( r-smem-addr -- )
  drop
  \ ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r...}, [addr]
  ;

: tc-load-b  ( r-smem-addr -- )
  drop
  \ ldmatrix.sync.aligned.x2.m8n8.shared.trans.b16 {%r...}, [addr]
  ;

\ Tensor core multiply-accumulate: D = A*B + C
\ m16n8k16, f16 inputs, f32 accumulator
: tc-mma  ( -- )
  \ mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  \   {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3}
  ;

\ Async global→shared copy (bypasses registers, uses TMA on Hopper)
: async-copy  ( r-dst-shared r-src-global n-bytes -- )
  drop drop drop
  \ cp.async.cg.shared.global [dst], [src], bytes
  ;

: async-commit  ( -- )
  s" cp.async.commit_group;" ptx+ ptx-nl ;

: async-wait  ( n -- )
  s" cp.async.wait_group " ptx+ ptx-num ptx+ s" ;" ptx+ ptx-nl  drop ;

\ ============================================================
\ SYNCHRONIZATION
\ ============================================================

: barrier  ( -- )  s" bar.sync 0;" ptx+ ptx-nl ;
: membar-gpu  ( -- )  s" membar.gl;" ptx+ ptx-nl ;
: membar-cta  ( -- )  s" membar.cta;" ptx+ ptx-nl ;

\ ============================================================
\ SHARED MEMORY
\ ============================================================

: shared-decl  ( n-bytes -- )
  s" .shared .align 16 .b8 smem[" ptx+ ptx-num ptx+ s" ];" ptx+ ptx-nl ;

: dyn-shared-decl  ( -- )
  s" .extern .shared .align 16 .b8 dyn_smem[];" ptx+ ptx-nl ;

\ ============================================================
\ COMPLETE PATTERNS — INFERENCE BUILDING BLOCKS
\ ============================================================

\ Softmax numerator: exp(x - max)
: safe-exp.f32  ( f-in f-max -- f-out )
  f32+  drop drop
  \ sub.f32 t, in, max
  \ ex2.approx.f32 out, t  (with ln2 scale factor)
  ;

\ Dot product of two f16 vectors (4 elements, packed)
: dot4-f16  ( r-a r-b -- f-acc )
  f32+  drop drop
  \ dp4a or fma chain on packed f16x2
  ;

\ Quantized GEMV: W4A16 dequant + multiply
\ The core of inference — weight is 4-bit, activation is f16
: w4a16-gemv  ( r-weight r-scale r-activation -- f-acc )
  f32+ f32+  r32+  drop drop drop
  \ Extract 4-bit nibbles from packed weight
  \ Dequantize: (nibble - 8) * scale
  \ FMA with activation
  ;
