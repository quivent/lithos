\ ls-dispatch.fs — Target dispatch layer for the .ls parser
\
\ Bridges the .ls parser to either ARM64 (host) or SASS (GPU) emitters.
\ The parser calls ls-emit-* words; this layer checks ls-target and
\ dispatches to the appropriate backend.
\
\ Required load order:
\   gpu/emit.fs                — SASS emitter (fadd, ldg, stg, etc.)
\   compiler/emit-elementwise.fs — fmul,, fadd-imm,, etc.
\   compiler/emit-arm64.fs     — ARM64 code buffer (a,, emit-host-stub)
\   (asm.fs loaded separately) — ARM64 instruction encoders (arm-add-reg, etc.)
\   compiler/ls-dispatch.fs    — this file
\
\ Public interface:
\   ls-target                  — variable: 0=ARM64, 1=GPU/SASS
\   ls-emit-add ... ls-emit-store etc. — target-dispatched emitters
\   ls-emit-each, ls-emit-barrier, ls-emit-reduce, etc. — GPU parallel ops

\ ============================================================
\ TARGET SELECTOR
\ ============================================================
\ 0 = ARM64 (host CPU)
\ 1 = GPU / SASS (sm_90 Hopper)

variable ls-target  1 ls-target !   \ default GPU — Lithos is GPU-first

: gpu?  ( -- flag )  ls-target @ 1 = ;
: arm?  ( -- flag )  ls-target @ 0= ;

\ ============================================================
\ RESET — clear the active backend's code buffer
\ ============================================================

: ls-reset  ( -- )
  gpu? if sass-reset else code-reset then ;

\ ============================================================
\ ARITHMETIC DISPATCH
\ ============================================================
\ All arithmetic dispatchers take register indices on the Forth
\ data stack and route to the correct backend emitter.
\
\ GPU:   SASS register indices (R0-R255), emitted as 128-bit instructions
\ ARM64: ARM64 register indices (X0-X30), emitted as 32-bit instructions

\ ls-emit-add  ( rd ra rb -- )
\ GPU:  FADD Rd, Ra, Rb      (FP32 register add)
\ ARM64: add Xd, Xn, Xm      (64-bit integer add)
: ls-emit-add  ( rd ra rb -- )
  gpu? if fadd, else arm-add-reg emit32 then ;

\ ls-emit-add-imm  ( rd ra imm -- )
\ GPU:  FADD Rd, Ra, imm32   (FP32 immediate add)
\ ARM64: add Xd, Xn, #imm12  (64-bit immediate add)
: ls-emit-add-imm  ( rd ra imm -- )
  gpu? if fadd-imm, else arm-add-imm emit32 then ;

\ ls-emit-sub  ( rd ra rb -- )
\ GPU:  FADD Rd, Ra, -Rb     (FP32 subtract via negated source)
\ ARM64: sub Xd, Xn, Xm
\
\ Note: SASS has no FSUB opcode. Subtraction uses FADD with the NEG
\ modifier on src2 (inst bit 63 = 1). We set that bit here.
variable _sub-rb  variable _sub-ra  variable _sub-rd
: ls-emit-sub  ( rd ra rb -- )
  gpu? if
    _sub-rb !  _sub-ra !  _sub-rd !
    \ Emit FADD with NEG on src2: set inst bit 63
    _sub-rb @ 32 lshift >r
    _sub-ra @ 24 lshift >r
    _sub-rd @ track-rd 16 lshift
    $7221 or r> or r> or
    1 63 lshift or               \ NEG src2 (bit 63 of inst word)
    ctrl-fadd sinst,
  else
    arm-sub-reg emit32
  then ;

\ ls-emit-mul  ( rd ra rb -- )
\ GPU:  FMUL Rd, Ra, Rb      (FP32 register multiply)
\ ARM64: mul Xd, Xn, Xm
: ls-emit-mul  ( rd ra rb -- )
  gpu? if fmul, else arm-mul emit32 then ;

\ ls-emit-fma  ( rd ra rb rc -- )
\ GPU:  FFMA Rd, Ra, Rb, Rc  (FP32 fused multiply-add: Rd = Ra*Rb + Rc)
\ ARM64: madd Xd, Xn, Xm, Xa (Xd = Xn*Xm + Xa)
: ls-emit-fma  ( rd ra rb rc -- )
  gpu? if ffma, else arm-madd emit32 then ;

\ ls-emit-div  ( rd ra rb -- )
\ GPU:  MUFU.RCP tmp, Rb; FMUL Rd, Ra, tmp  (reciprocal then multiply)
\ ARM64: sdiv Xd, Xn, Xm
variable _div-rd  variable _div-ra  variable _div-rb  variable _div-tmp
: ls-emit-div  ( rd ra rb -- )
  gpu? if
    _div-rb !  _div-ra !  _div-rd !
    rreg+ _div-tmp !
    _div-tmp @  _div-rb @  rcp,        \ tmp = 1/rb
    _div-rd @  _div-ra @  _div-tmp @  fmul,   \ rd = ra * tmp
  else
    arm-sdiv emit32
  then ;

\ ls-emit-neg  ( rd rs -- )
\ GPU:  FADD Rd, RZ, -Rs (negate via FADD with zero + NEG modifier)
\ ARM64: neg Xd, Xn  (= sub Xd, XZR, Xn)
: ls-emit-neg  ( rd rs -- )
  gpu? if
    swap track-rd 16 lshift
    swap 32 lshift >r
    $ff 24 lshift >r          \ Ra = RZ = 0xFF
    $7221 or r> or r> or
    1 63 lshift or             \ NEG src2
    ctrl-fadd sinst,
  else
    \ ARM64: sub Xd, XZR(=31), Xn
    swap 31 rot arm-sub-reg emit32
  then ;

\ ls-emit-mov-imm  ( rd imm32 -- )
\ GPU:  MOV Rd, imm32
\ ARM64: movz Xd, imm16, LSL#0  (lower 16 bits only — caller chains movk for >16bit)
: ls-emit-mov-imm  ( rd imm32 -- )
  gpu? if mov-imm, else swap 0 arm-movz emit32 then ;

\ ============================================================
\ MEMORY DISPATCH
\ ============================================================
\ The .ls language uses arrow notation:
\   →  (right arrow) = load  = read from memory
\   ←  (left arrow)  = store = write to memory

\ ls-emit-load  ( rd ra -- )
\ GPU:  LDG.E Rd, [Ra.64]    (global memory load, 32-bit)
\ ARM64: ldr Xd, [Xn]        (64-bit load, unsigned offset 0)
: ls-emit-load  ( rd ra -- )
  gpu? if ldg, else swap 0 arm-ldr-off emit32 then ;

\ ls-emit-load-off  ( rd ra off -- )
\ GPU:  LDG.E Rd, [Ra.64+off]
\ ARM64: ldr Xd, [Xn, #off]
: ls-emit-load-off  ( rd ra off -- )
  gpu? if ldg-off, else arm-ldr-off emit32 then ;

\ ls-emit-store  ( ra rs -- )
\ GPU:  STG.E [Ra.64], Rs    (global memory store, 32-bit)
\ ARM64: str Xd, [Xn]        (64-bit store, unsigned offset 0)
: ls-emit-store  ( ra rs -- )
  gpu? if stg, else swap 0 arm-str-off emit32 then ;

\ ls-emit-store-off  ( ra rs off -- )
\ GPU:  STG.E [Ra.64+off], Rs
\ ARM64: str Xt, [Xn, #off]
: ls-emit-store-off  ( ra rs off -- )
  gpu? if stg-off, else arm-str-off emit32 then ;

\ ls-emit-load-shared  ( rd ra off -- )
\ GPU:  LDS.32 Rd, [Ra+off]  (shared memory load)
\ ARM64: no-op / error — shared memory is GPU-only
: ls-emit-load-shared  ( rd ra off -- )
  gpu? if lds, else drop drop drop then ;

\ ls-emit-store-shared  ( ra rs off -- )
\ GPU:  STS.32 [Ra+off], Rs  (shared memory store)
\ ARM64: no-op — shared memory is GPU-only
: ls-emit-store-shared  ( ra rs off -- )
  gpu? if sts, else drop drop drop then ;

\ ============================================================
\ INTEGER ARITHMETIC DISPATCH
\ ============================================================

\ ls-emit-iadd  ( rd rs1 rs2 rs3 -- )
\ GPU:  IADD3 Rd, Rs1, Rs2, Rs3  (3-input integer add; use RZ=255 for 2-input)
\ ARM64: add Xd, Xn, Xm  (ignores rs3)
: ls-emit-iadd  ( rd rs1 rs2 rs3 -- )
  gpu? if iadd3, else drop arm-add-reg emit32 then ;

\ ls-emit-imul  ( rd rs1 rs2 rs3 -- )
\ GPU:  IMAD Rd, Rs1, Rs2, Rs3  (integer multiply-add; use RZ for pure multiply)
\ ARM64: mul Xd, Xn, Xm  (ignores rs3)
: ls-emit-imul  ( rd rs1 rs2 rs3 -- )
  gpu? if imad, else drop arm-mul emit32 then ;

\ ls-emit-shl  ( rd rs shamt -- )
\ GPU:  SHF.L.U32 Rd, Rs, shamt
\ ARM64: lsl Xd, Xn, #shamt
: ls-emit-shl  ( rd rs shamt -- )
  gpu? if shf-r-imm, else arm-lsl-imm emit32 then ;

\ ls-emit-and  ( rd ra rb -- )
\ GPU:  LOP3 Rd, Ra, Rb, RZ with AND truth table
\ ARM64: and Xd, Xn, Xm
: ls-emit-and  ( rd ra rb -- )
  gpu? if
    \ LOP3.LUT register form: rd ra rb with LUT=0xC0 (AND)
    32 lshift >r  24 lshift >r  track-rd 16 lshift
    $7212 or r> or r> or
    $c0 ctrl-lop3 sinst,
  else
    arm-and-reg emit32
  then ;

\ ls-emit-or  ( rd ra rb -- )
\ GPU:  LOP3 Rd, Ra, Rb, RZ with OR truth table
\ ARM64: orr Xd, Xn, Xm
: ls-emit-or  ( rd ra rb -- )
  gpu? if
    32 lshift >r  24 lshift >r  track-rd 16 lshift
    $7212 or r> or r> or
    $fc ctrl-lop3 sinst,
  else
    arm-orr-reg emit32
  then ;

\ ls-emit-xor  ( rd ra rb -- )
\ GPU:  LOP3 Rd, Ra, Rb, RZ with XOR truth table
\ ARM64: eor Xd, Xn, Xm
: ls-emit-xor  ( rd ra rb -- )
  gpu? if
    32 lshift >r  24 lshift >r  track-rd 16 lshift
    $7212 or r> or r> or
    $3c ctrl-lop3 sinst,
  else
    arm-eor-reg emit32
  then ;

\ ============================================================
\ COMPARISON / BRANCH DISPATCH
\ ============================================================

\ ls-emit-cmp-ge  ( pd rs1 rs2 -- )
\ GPU:  ISETP.GE.U32.AND Pd, PT, Rs1, Rs2, PT
\ ARM64: cmp Xn, Xm (sets flags; pd ignored)
: ls-emit-cmp-ge  ( pd rs1 rs2 -- )
  gpu? if isetp-ge, else drop arm-sub-reg emit32 drop then ;

\ ls-emit-bra  ( byte-offset -- )
\ GPU:  BRA offset  (PC-relative, signed)
\ ARM64: b offset   (PC-relative, signed, 26-bit imm)
: ls-emit-bra  ( byte-offset -- )
  gpu? if bra, else 4 / $14000000 or emit32 then ;

\ ls-emit-exit  ( -- )
\ GPU:  EXIT (terminate thread)
\ ARM64: ret (return to caller)
: ls-emit-exit  ( -- )
  gpu? if exit, else $d65f03c0 emit32 then ;

\ ls-emit-nop  ( -- )
: ls-emit-nop  ( -- )
  gpu? if nop, else $d503201f emit32 then ;

\ ============================================================
\ TYPE CONVERSION DISPATCH
\ ============================================================

\ ls-emit-i2f  ( rd rs -- )
\ GPU:  I2FP.F32.S32 Rd, Rs
\ ARM64: scvtf Dd, Xn  (integer to float — caller handles FP regs)
: ls-emit-i2f  ( rd rs -- )
  gpu? if i2f-s32-f32, else drop drop then ;

\ ls-emit-f2i  ( rd rs -- )
\ GPU:  F2I.TRUNC.NTZ Rd, Rs
\ ARM64: fcvtzs Xd, Dn  (float to integer — caller handles FP regs)
: ls-emit-f2i  ( rd rs -- )
  gpu? if f2i-f32-s32, else drop drop then ;

\ ============================================================
\ MATH INTRINSIC DISPATCH (GPU-only; ARM64 falls back to libm calls)
\ ============================================================
\ These are SFU (Special Function Unit) operations on GPU.
\ On ARM64 they would need libm calls — stubbed for now.

: ls-emit-rcp   ( rd rs -- )  gpu? if rcp,   else 2drop then ;
: ls-emit-rsqrt ( rd rs -- )  gpu? if rsqrt, else 2drop then ;
: ls-emit-sqrt  ( rd rs -- )  gpu? if sqrt,  else 2drop then ;
: ls-emit-exp2  ( rd rs -- )  gpu? if ex2,   else 2drop then ;
: ls-emit-log2  ( rd rs -- )  gpu? if lg2,   else 2drop then ;
: ls-emit-sin   ( rd rs -- )  gpu? if sin,   else 2drop then ;
: ls-emit-cos   ( rd rs -- )  gpu? if cos,   else 2drop then ;

\ ============================================================
\ GPU PARALLEL OPERATIONS
\ ============================================================
\ These have no ARM64 equivalent — they are GPU execution model
\ primitives. On ARM64 they either no-op or map to serial loops.

\ ls-emit-s2r  ( rd sr-id -- )
\ Read a special register (thread ID, block ID, lane ID, etc.)
\ ARM64: no-op (no thread model on host)
: ls-emit-s2r  ( rd sr-id -- )
  gpu? if s2r, else 2drop then ;

\ ls-emit-each  ( -- )
\ Thread-parallel iteration. On GPU, this sets up the thread index
\ computation: tid = blockIdx.x * blockDim.x + threadIdx.x
\ On ARM64, this would begin a serial for-loop (stubbed).
\
\ Emits the standard GPU thread-index preamble:
\   S2R  r_tid, SR_TID.X          — threadIdx.x
\   S2R  r_bid, SR_CTAID.X        — blockIdx.x
\   IMAD r_gid, r_bid, blockDim, r_tid  — global index
\
\ Returns: ( r_gid -- ) the register holding the global thread index
variable _each-tid  variable _each-bid
: ls-emit-each  ( blockDim -- r_gid )
  gpu? if
    rreg+ _each-tid !
    rreg+ _each-bid !
    _each-tid @  SR-TID-X   s2r,          \ r_tid = threadIdx.x
    _each-bid @  SR-CTAID-X s2r,          \ r_bid = blockIdx.x
    rreg+ dup >r
    _each-bid @  swap  mov-imm,            \ r_bdim = blockDim (immediate)
    rreg+ dup >r
    _each-bid @  r>  _each-tid @  imad,    \ r_gid = bid * bdim + tid
    r>
  else
    drop 0   \ ARM64: return register 0 as placeholder
  then ;

\ ls-emit-stride  ( rd r_idx stride -- )
\ Compute strided address: rd = base + idx * stride
\ GPU:  IMAD Rd, Ridx, stride_reg, Rbase  (or shift for power-of-2)
\ ARM64: madd or shift sequence
: ls-emit-stride  ( rd r_idx r_stride r_base -- )
  gpu? if imad, else drop arm-mul emit32 drop then ;

\ ls-emit-barrier  ( bar-id -- )
\ GPU:  BAR.SYNC bar-id  (synchronize all threads in thread block)
\ ARM64: no-op (no thread barrier on host)
: ls-emit-barrier  ( bar-id -- )
  gpu? if bar-sync, else drop then ;

\ ls-emit-membar-gpu  ( -- )
\ GPU:  MEMBAR.SC.GPU  (device-scope memory fence)
\ ARM64: dmb ish  (inner-shareable data memory barrier)
: ls-emit-membar-gpu  ( -- )
  gpu? if membar-gpu, else $d5033bbf emit32 then ;

\ ls-emit-warp-reduce  ( acc tmp -- )
\ GPU:  5-step butterfly SHFL.BFLY + FADD reduction across 32 lanes
\ ARM64: no-op (single-threaded; value already in acc)
: ls-emit-warp-reduce  ( acc tmp -- )
  gpu? if warp-reduce, else 2drop then ;

\ ls-emit-shfl-bfly  ( rd rs delta -- )
\ GPU:  SHFL.BFLY PT, Rd, Rs, delta, 0x1f
\ ARM64: mov Rd, Rs  (no shuffle on host — identity)
: ls-emit-shfl-bfly  ( rd rs delta -- )
  gpu? if shfl-bfly, else drop arm-mov-reg emit32 then ;

\ ls-emit-shfl-down  ( rd rs delta -- )
\ GPU:  SHFL.DOWN PT, Rd, Rs, delta, 0x1f
\ ARM64: mov Rd, Rs  (identity fallback)
: ls-emit-shfl-down  ( rd rs delta -- )
  gpu? if shfl-down, else drop arm-mov-reg emit32 then ;

\ ============================================================
\ GPU KERNEL OPERATIONS  (** / *** / grid-sync)
\ ============================================================
\ These map .ls syntax to the appropriate emit pattern.

\ ls-emit-elementwise  ( -- )
\ Marker: the current kernel body is elementwise (**).
\ Sets up the per-thread index and bounds check pattern.
\ GPU: emits S2R + IMAD + ISETP + BRA preamble
\ ARM64: emits loop header
variable ls-ew-gid
: ls-emit-elementwise  ( n-elems blockDim -- )
  gpu? if
    ls-emit-each ls-ew-gid !
    \ Bounds check: if gid >= n_elems, exit
    rreg+ dup >r  swap  mov-imm,         \ r_n = n_elems
    preg+ dup >r  ls-ew-gid @  swap  isetp-ge,  \ p = (gid >= n_elems)
    0 r> bra-pred,                        \ placeholder branch (patched later)
    r> drop                               \ clean up
  else
    2drop
  then ;

\ ls-emit-matrix  ( -- )
\ Marker: the current kernel body is matrix (***).
\ GPU: emits 2D thread index computation (row + col)
\ ARM64: emits nested loop
variable ls-mx-row  variable ls-mx-col
: ls-emit-matrix  ( rows cols blockDim -- )
  gpu? if
    \ 2D indexing: row from blockIdx.y * blockDim + threadIdx.y
    \              col from blockIdx.x * blockDim + threadIdx.x
    rreg+ dup ls-mx-col !  SR-TID-X  s2r,
    rreg+ dup >r  SR-CTAID-X  s2r,
    rreg+ dup >r  swap  mov-imm,          \ r_bdim = blockDim
    ls-mx-col @  r>  r>  ls-mx-col @  imad,  \ col = ctaid.x * bdim + tid.x
    rreg+ dup ls-mx-row !  SR-TID-Y  s2r,
    rreg+ dup >r  SR-CTAID-Y  s2r,
    rreg+ dup >r  rot  mov-imm,
    ls-mx-row @  r>  r>  ls-mx-row @  imad,  \ row = ctaid.y * bdim + tid.y
    drop drop                              \ consume rows, cols
  else
    drop drop drop
  then ;

\ ls-emit-grid-sync  ( sync-ctr-reg flag-reg grid-size -- )
\ GPU:  Full cooperative grid synchronization (19-instruction sequence)
\ ARM64: no-op (single process)
: ls-emit-grid-sync  ( sync-ctr-reg flag-reg grid-size -- )
  gpu? if emit-grid-sync else drop drop drop then ;

\ ls-emit-atom-add  ( rd ra rs -- )
\ GPU:  ATOMG.E.ADD.STRONG.GPU Rd, [Ra], Rs
\ ARM64: ldaxr/stlxr loop (stubbed)
: ls-emit-atom-add  ( rd ra rs -- )
  gpu? if atom-add, else drop drop drop then ;

\ ============================================================
\ SHARED MEMORY DECLARATION (GPU-only)
\ ============================================================
\ Declares a shared memory buffer. On GPU, this reserves space in
\ the .nv.shared section. On ARM64, allocates stack space.

\ ls-emit-shared  ( bytes -- )
\ GPU: records shared memory size for ELF builder
\ ARM64: sub sp, sp, #bytes
: ls-emit-shared  ( bytes -- )
  gpu? if
    drop   \ size tracked by parser's shm-bytes table, not emitted as instructions
  else
    \ ARM64: allocate on stack
    31 31 rot arm-sub-imm emit32
  then ;

\ ============================================================
\ CONVENIENCE ALIASES for .ls arrow notation
\ ============================================================
\ The parser maps → to load and ← to store.

: ls-emit-->  ( rd ra -- )       ls-emit-load ;       \ →
: ls-emit-<-  ( ra rs -- )       ls-emit-store ;      \ ←
: ls-emit-->off  ( rd ra off -- ) ls-emit-load-off ;   \ → with offset
: ls-emit-<-off  ( ra rs off -- ) ls-emit-store-off ;  \ ← with offset
