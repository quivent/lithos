\ emit-elementwise.fs — Element-wise GPU instruction pattern emitters for Lithos compiler
\
\ Pattern layer: Forth words that call sequences of sm90 instruction encoders
\ (from gpu/emit.fs) to implement each element-wise primitive's instruction pattern.
\
\ These primitives operate per-element on vectors and are trivially parallel.
\ The compiler calls these patterns when lowering Lithos primitives to GPU machine code.

\ ============================================================
\ DEPENDENCIES
\ ============================================================
\ gpu/emit.fs provides: fadd, ffma, s2r, imad, ldg, stg, exit,
\   bra, bra-pred, isetp-ge, isetp-lt, iadd3, mov-imm, shf-r-imm,
\   rcp, rsqrt, lg2, ex2, sqrt, sin, cos, nop,
\   sass-pos, sass-buf, sinst,
\   SR-TID-X, SR-CTAID-X
\ parser.fs provides: rreg+, freg+, preg+, rdreg+, next-rreg, next-freg

s" /home/ubuntu/lithos/gpu/emit.fs" included

\ ============================================================
\ CONSTANTS
\ ============================================================

\ IEEE 754 single-precision constants (as 32-bit hex integers)
$3fb8aa3b constant FP32-LOG2E     \ 1.442695 = log2(e)
$3f317218 constant FP32-LN2       \ 0.693147 = ln(2)
$bf800000 constant FP32-NEG1      \ -1.0
$3f800000 constant FP32-ONE       \ 1.0

\ RZ (zero register) = register 255
$ff constant RZ

\ ============================================================
\ FMUL — MISSING FROM gpu/emit.fs
\ ============================================================
\ FMUL Rd, Ra, Rb — FP32 multiply (register form)
\ Opcode: 0x220 -> $7220 (FADD is $7221; FMUL is adjacent)
\ Encoding mirrors FADD: Rd bits[23:16], Ra bits[31:24], Rb bits[39:32]
\ Scheduling: same as FADD (stall=5, yield=0, wbar=7, rbar=7)
\
\ NOTE: FMUL opcode unverified against probe disassembly.
\ If incorrect, replace $7220 with probed value.

: fmul,  ( rd ra rb -- )
  32 lshift >r              \ rb -> bits[39:32]
  24 lshift >r              \ ra -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7220 or r> or r> or
  ctrl-fadd sinst, ;

\ FMUL Rd, Ra, imm32 — FP32 multiply with 32-bit immediate
\ imm32 in bits[63:32], Ra in bits[31:24], Rd in bits[23:16]
\ Uses ctrl-fadd (same scheduling as FADD-imm).
\
\ Opcode: OP-FMUL-IMM = $7820 (probe-verified in opcodes-sm90.fs).

: fmul-imm,  ( rd ra imm32 -- )
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ ra    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  OP-FMUL-IMM or r> or r> or
  ctrl-fadd sinst, ;

\ FADD Rd, Ra, imm32 — FP32 add with 32-bit immediate
\ Same encoding pattern as FMUL-imm but with FADD opcode family.
\ Opcode: OP-FADD-IMM = $7421 (probe-verified in opcodes-sm90.fs).

: fadd-imm,  ( rd ra imm32 -- )
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ ra    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  OP-FADD-IMM or r> or r> or
  ctrl-fadd sinst, ;

\ SHF.L.U32 Rd, Rs, shamt, RZ — shift left by immediate
\ Mirrors shf-r-imm, but with left-shift modifier.
\ shf-r-imm, uses opcode $7819. Left vs right is encoded in ctrl
\ extra41 bits. Using same opcode with bit 19 toggled for direction.
\ NOTE: SHF.L direction bit unverified. Placeholder — may need
\ probe verification for the direction modifier encoding.

: shf-l-imm,  ( rd rs shamt -- )
  32 lshift >r              \ shamt -> bits[47:32]
  24 lshift >r              \ rs    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7819 or r> or            \ rs
  $ff 48 lshift or          \ RZ funnel source at bits[55:48]
  r> or                     \ shamt
  \ SHF.L needs direction modifier in ctrl extra41
  \ Using ctrl with extra41 bit set for left shift
  1 0 7 7 0 0 $10 make-ctrl sinst, ;

\ ============================================================
\ KERNEL PROLOGUE
\ ============================================================

\ emit-prologue ( -- )
\ Emits the standard thread-indexing sequence:
\   S2R  R0, SR_TID.X       ; threadIdx.x
\   S2R  R1, SR_CTAID.X     ; blockIdx.x
\   IMAD R0, R1, blockDim, R0  ; global index = blockIdx.x * blockDim + threadIdx.x
\ R0 holds global thread index afterward. R1 is scratch.
\ blockDim is loaded as a constant (set by emit-load-blockdim or via IMAD immediate).
\ Here we use IMAD with R2 as blockDim register — caller must load blockDim into R2
\ before the kernel launch, or use a uniform register.
\
\ Convention: R0 = tid, R1 = ctaid, R2 = blockDim (loaded separately)
\ Returns with global index in R0.

: emit-prologue  ( -- )
  0 SR-TID-X   s2r,          \ S2R R0, SR_TID.X
  1 SR-CTAID-X s2r,          \ S2R R1, SR_CTAID.X
  0 1 2 0      imad, ;       \ IMAD R0, R1, R2, R0 -> R0 = R1*R2 + R0

\ ============================================================
\ KERNEL EPILOGUE
\ ============================================================

\ emit-epilogue ( -- )
\ Emits EXIT instruction to terminate thread.

: emit-epilogue  ( -- )
  exit, ;

\ ============================================================
\ PARAMETER LOADING
\ ============================================================

\ emit-load-param ( param-idx -- reg )
\ Load a 64-bit pointer parameter from constant bank c[0x0][offset].
\ Offset = 0x160 + param_idx * 8 (CUDA kernel param base on Hopper).
\
\ MISSING ENCODER: Neither uldc, nor ldc, exists in gpu/emit.fs.
\ Workaround: use two MOV-IMM instructions to construct the address,
\ or use IMAD-IMM as a register move. For now, emit placeholder NOPs
\ and document the gap.
\
\ TODO: Add LDC or ULDC encoder to gpu/emit.fs:
\   ULDC URn, c[0x0][offset]  — opcode $7ab9
\   LDC  Rn, c[0x0][offset]   — opcode $7b82

: emit-load-param  ( param-idx -- reg )
  8 * $160 +                  \ offset = 0x160 + param_idx * 8
  drop                        \ cannot encode LDC yet
  rreg+                       \ allocate destination register
  dup 0 mov-imm,              \ placeholder: MOV Rd, 0 (will be replaced by LDC)
  \ TODO: replace with: dup offset ldc, (when encoder exists)
  ;

\ ============================================================
\ ARRAY LOAD / STORE
\ ============================================================

\ emit-array-load ( base-reg idx-reg -- dst-reg )
\ Load a single f32 element from array[idx]:
\   SHF.L Rt, idx, 2, RZ     ; byte offset = idx * 4 (f32 = 4 bytes)
\   LDG.E Rd, [base + Rt]    ; load from base + byte_offset
\
\ Note: LDG with offset addressing — we compute byte offset into Rt,
\ then use IADD3 to form full address in a temp 64-bit pair, then LDG.
\ Simplified: use shf-l for multiply-by-4, then iadd3 for addr, then ldg.

: emit-array-load  ( base-reg idx-reg -- dst-reg )
  \ SHF.L Rt, idx, 2       -> byte offset = idx * 4
  \ IADD3 Ra, base, Rt, RZ -> address = base + offset
  \ LDG.E Rd, [Ra]         -> load value
  rreg+ >r                    \ R: rt; stack: base idx
  r@ swap 2 shf-l-imm,        \ shf-l-imm, ( rd rs shamt -- ); stack: base; R: rt
  r>                           \ stack: base rt
  rreg+ >r                    \ R: ra; stack: base rt
  r@ -rot RZ iadd3,           \ iadd3, ( rd rs1 rs2 rs3 -- ) = ra base rt RZ; R: ra
  r>                           \ stack: ra
  rreg+ swap                   \ stack: rd ra
  over swap ldg,               \ ldg, ( rd ra -- ); stack: rd
  ;

\ emit-array-store ( base-reg idx-reg val-reg -- )
\ Store a single f32 element to array[idx]:
\   SHF.L Rt, idx, 2, RZ     ; byte offset = idx * 4
\   IADD3 Ra, base, Rt, RZ   ; address = base + offset
\   STG.E [Ra], val           ; store

: emit-array-store  ( base-reg idx-reg val-reg -- )
  \ SHF.L Rt, idx, 2       -> byte offset = idx * 4
  \ IADD3 Ra, base, Rt, RZ -> address = base + offset
  \ STG.E [Ra], val        -> store
  >r                          \ R: val; stack: base idx
  rreg+ >r                    \ R: val rt; stack: base idx
  r@ swap 2 shf-l-imm,        \ shf-l-imm, ( rd rs shamt -- ); stack: base; R: val rt
  r>                           \ stack: base rt; R: val
  rreg+ >r                    \ R: val ra; stack: base rt
  r@ -rot RZ iadd3,           \ iadd3, ( rd rs1 rs2 rs3 -- ) = ra base rt RZ; R: val ra
  r> r> stg, ;                \ stg, ( ra rs -- ); ra from R, val from R

\ ============================================================
\ ELEMENT-WISE PRIMITIVES
\ ============================================================

\ 1. emit-mul ( ra rb -- rd )
\    FMUL Rd, Ra, Rb

: emit-mul  ( ra rb -- rd )
  rreg+ >r
  r@ -rot fmul,              \ FMUL Rd, Ra, Rb
  r> ;

\ 2. emit-mul-scalar ( ra imm -- rd )
\    FMUL Rd, Ra, imm32  (for * -1, * 1.442695, etc.)

: emit-mul-scalar  ( ra imm -- rd )
  rreg+ >r                    \ R: rd; stack: ra imm
  r@ -rot                     \ stack: rd ra imm
  fmul-imm,                   \ fmul-imm, ( rd ra imm -- )
  r> ;

\ 3. emit-add ( ra rb -- rd )
\    FADD Rd, Ra, Rb

: emit-add  ( ra rb -- rd )
  rreg+ >r
  r@ -rot fadd,              \ FADD Rd, Ra, Rb
  r> ;

\ 4. emit-add-scalar ( ra imm -- rd )
\    FADD Rd, Ra, imm32  (for + 1, etc.)

: emit-add-scalar  ( ra imm -- rd )
  rreg+ >r                    \ R: rd; stack: ra imm
  r@ -rot                     \ stack: rd ra imm
  fadd-imm,                   \ fadd-imm, ( rd ra imm -- )
  r> ;

\ 5. emit-sub ( ra rb -- rd )
\    FADD Rd, Ra, -Rb  (negate source modifier on Rb)
\    Since we lack a negate-modifier encoding, use:
\    FMUL Rt, Rb, -1.0 then FADD Rd, Ra, Rt

: emit-sub  ( ra rb -- rd )
  \ Negate rb: FMUL Rt, Rb, -1.0  then FADD Rd, Ra, Rt
  swap >r                      \ R: ra; stack: rb
  rreg+ >r                    \ R: ra rt; stack: rb
  r@ swap FP32-NEG1            \ stack: rt rb neg1; R: ra rt
  fmul-imm,                   \ FMUL Rt, Rb, -1.0; stack: empty; R: ra rt
  r> r>                        \ stack: rt ra
  swap                         \ stack: ra rt
  rreg+ >r                    \ R: rd; stack: ra rt
  r@ -rot fadd,               \ FADD Rd, Ra, Rt; stack: empty; R: rd
  r> ;

\ 6. emit-div ( ra rb -- rd )
\    MUFU.RCP Rt, Rb  then  FMUL Rd, Ra, Rt

: emit-div  ( ra rb -- rd )
  \ MUFU.RCP Rt, Rb  then  FMUL Rd, Ra, Rt
  rreg+ >r                    \ R: rt; stack: ra rb
  r@ swap rcp,                \ rcp, ( rd rs -- ) -> MUFU.RCP Rt, Rb; stack: ra; R: rt
  r>                           \ stack: ra rt
  rreg+ >r                    \ R: rd; stack: ra rt
  r@ -rot fmul,               \ fmul, ( rd ra rb -- ) -> FMUL Rd, Ra, Rt; R: rd
  r> ;

\ 7. emit-exp ( ra -- rd )
\    FMUL Rt, Ra, 1.442695  then  MUFU.EX2 Rd, Rt

: emit-exp  ( ra -- rd )
  \ FMUL Rt, Ra, log2(e)  then  MUFU.EX2 Rd, Rt
  rreg+ >r                    \ R: rt; stack: ra
  r@ swap FP32-LOG2E           \ stack: rt ra log2e; R: rt
  fmul-imm,                   \ fmul-imm, ( rd ra imm -- ); stack: empty; R: rt
  r>                           \ stack: rt
  rreg+ swap                   \ stack: rd rt
  over swap ex2,               \ ex2, ( rd rs -- ); stack: rd
  ;

\ 8. emit-log ( ra -- rd )
\    MUFU.LG2 Rt, Ra  then  FMUL Rd, Rt, 0.693147

: emit-log  ( ra -- rd )
  \ MUFU.LG2 Rt, Ra  then  FMUL Rd, Rt, ln(2)
  rreg+ >r                    \ R: rt; stack: ra
  r@ swap lg2,                \ lg2, ( rd rs -- ); stack: empty; R: rt
  r>                           \ stack: rt
  rreg+ >r                    \ R: rd; stack: rt
  r@ swap FP32-LN2             \ stack: rd rt ln2; R: rd
  fmul-imm,                   \ fmul-imm, ( rd ra imm -- ); R: rd
  r> ;

\ 9. emit-sqrt ( ra -- rd )
\    MUFU.RSQ Rt, Ra  then  MUFU.RCP Rd, Rt

: emit-sqrt  ( ra -- rd )
  \ MUFU.RSQ Rt, Ra  then  MUFU.RCP Rd, Rt
  rreg+ >r                    \ R: rt; stack: ra
  r@ swap rsqrt,              \ rsqrt, ( rd rs -- ); stack: empty; R: rt
  r>                           \ stack: rt
  rreg+ swap                   \ stack: rd rt
  over swap rcp,               \ rcp, ( rd rs -- ); stack: rd
  ;

\ 10. emit-rcp ( ra -- rd )
\     MUFU.RCP Rd, Ra

: emit-rcp  ( ra -- rd )
  rreg+ >r
  r@ swap rcp,                \ MUFU.RCP Rd, Ra
  r> ;

\ 11. emit-rsqrt ( ra -- rd )
\     MUFU.RSQ Rd, Ra

: emit-rsqrt  ( ra -- rd )
  rreg+ >r
  r@ swap rsqrt,              \ MUFU.RSQ Rd, Ra
  r> ;

\ ============================================================
\ STRIDE LOOP
\ ============================================================
\ Standard GPU stride loop: each thread handles elements
\   idx, idx+blockDim, idx+2*blockDim, ...
\ until idx >= bound.
\
\ emit-stride-begin saves the current sass-pos for the branch-back target.
\ emit-stride-end emits the increment and backward branch.

variable stride-loop-top    0 stride-loop-top !
variable stride-loop-step   0 stride-loop-step !
variable stride-loop-idx    0 stride-loop-idx !
variable stride-loop-bound  0 stride-loop-bound !
variable stride-loop-bra    0 stride-loop-bra !   \ sass-pos of forward BRA (for patching)

\ emit-stride-begin ( idx-reg bound-reg step-reg -- )
\ Emits:
\   ISETP.GE P0, idx, bound   ; if idx >= bound, skip loop
\   @P0 BRA <forward>         ; branch past loop end (patched by emit-stride-end)
\ Saves loop state for emit-stride-end.

: emit-stride-begin  ( idx-reg bound-reg step-reg -- )
  stride-loop-step !
  stride-loop-bound !
  stride-loop-idx !
  \ Emit comparison: ISETP.GE P0, idx, bound
  0 stride-loop-idx @ stride-loop-bound @ isetp-ge,
  \ Save position of forward BRA for patching, then emit placeholder
  sass-pos @ stride-loop-bra !
  0 0 bra-pred,               \ @P0 BRA +0 (placeholder — patched below)
  \ Record loop body start (branch-back target for emit-stride-end)
  sass-pos @ stride-loop-top ! ;

\ patch-forward-bra ( -- )
\ Patches the forward BRA emitted by emit-stride-begin to jump to current pos.
\ BRA encoding: signed offset in bits[63:32] of inst word (bytes 4-7 of 16-byte slot).
variable patch-addr   variable patch-off
: patch-forward-bra  ( -- )
  sass-pos @  stride-loop-bra @  -  16 -  4 /  patch-off !
  stride-loop-bra @ 4 + sass-buf +              patch-addr !
  patch-off @           patch-addr @ c!
  patch-off @  8 rshift patch-addr @ 1+ c!
  patch-off @ 16 rshift patch-addr @ 2 + c!
  patch-off @ 24 rshift patch-addr @ 3 + c! ;

\ emit-stride-end ( -- )
\ Emits:
\   IADD3 idx, idx, step, RZ  ; idx += step
\   BRA loop_top               ; branch back to loop body
\ Then patches the forward BRA to skip past this point.

: emit-stride-end  ( -- )
  \ IADD3 idx, idx, step, RZ  -> idx += step
  stride-loop-idx @ dup stride-loop-step @ RZ iadd3,
  \ BRA back to loop top (negative offset from next instruction)
  \ offset = loop_top - (current_pos + 16)
  stride-loop-top @  sass-pos @  -  16 -
  bra,
  \ Patch the forward branch to land here (past the loop)
  patch-forward-bra ;

\ ============================================================
\ DIMENSIONAL VECTOR EMITTERS  (** / ++ / -- / //)
\ ============================================================
\ Each "vec" emitter wraps one stride loop around a scalar op.
\ Calling convention: value stack carries pointer registers for
\ the input arrays and the output array.  The emitter consumes
\ those registers and leaves the output-array register on the stack.
\
\ Stack for binary ops: ( a-base b-base result-base bound step -- )
\ (bound = element count, step = gridDim.x * blockDim.x)
\
\ For the dimensional emitters the caller is expected to push:
\   a-ptr b-ptr out-ptr n step
\ where n is the element count and step is the grid stride.
\ The loop index register is allocated internally.

\ emit-vec-binary ( a-ptr b-ptr out-ptr n step op-xt -- )
\ Generic one-loop stride vector operation.
\ op-xt is the scalar emitter word to call: ( ra rb -- rd )
\ op-xt is passed on the data stack (execution token).

variable evb-a    variable evb-b    variable evb-out
variable evb-n    variable evb-step variable evb-op
variable evb-idx  variable evb-va   variable evb-vb   variable evb-rd
\ evb-op holds the execution token for the scalar op; stored via variable
\ to avoid return-stack lifetime issues across nested calls.

: emit-vec-binary  ( a-ptr b-ptr out-ptr n step op-xt -- )
  evb-op !  evb-step !  evb-n !
  evb-out !  evb-b !  evb-a !
  rreg+ evb-idx !                        \ allocate loop index register
  \ Copy global thread index (R0) into evb-idx as starting index
  evb-idx @  0  mov-imm,                 \ idx = 0 (placeholder; real: MOV idx, R0)
  evb-idx @  evb-n @  evb-step @         \ ( idx-reg bound-reg step-reg )
  emit-stride-begin
    evb-a @  evb-idx @  emit-array-load  evb-va !
    evb-b @  evb-idx @  emit-array-load  evb-vb !
    evb-va @  evb-vb @  evb-op @ execute evb-rd !
    evb-out @  evb-idx @  evb-rd @  emit-array-store
  emit-stride-end ;

\ emit-mul-vec ( a-ptr b-ptr out-ptr n step -- )
: emit-mul-vec  ( a-ptr b-ptr out-ptr n step -- )
  ' emit-mul  emit-vec-binary ;

\ emit-add-vec ( a-ptr b-ptr out-ptr n step -- )
: emit-add-vec  ( a-ptr b-ptr out-ptr n step -- )
  ' emit-add  emit-vec-binary ;

\ emit-sub-vec ( a-ptr b-ptr out-ptr n step -- )
: emit-sub-vec  ( a-ptr b-ptr out-ptr n step -- )
  ' emit-sub  emit-vec-binary ;

\ emit-div-vec ( a-ptr b-ptr out-ptr n step -- )
: emit-div-vec  ( a-ptr b-ptr out-ptr n step -- )
  ' emit-div  emit-vec-binary ;

\ ============================================================
\ DIMENSIONAL MATRIX EMITTERS  (*** / +++ / --- / ///)
\ ============================================================
\ Outer-product style: for each (i, j) pair:
\   result[i*N + j] = a[i] op b[j]
\
\ Stack: ( a-ptr b-ptr out-ptr M N step -- )
\ M = length of a (rows), N = length of b (cols)
\ step = gridDim.x * blockDim.x (shared for both loop levels)
\
\ Two nested stride loops — outer over i, inner over j.
\ We save/restore stride-loop state across nesting manually
\ because emit-stride-begin/end use single shared variables.

\ Saved outer-loop state
variable slx-top    variable slx-step
variable slx-idx    variable slx-bound  variable slx-bra

: save-stride-state  ( -- )
  stride-loop-top @   slx-top !
  stride-loop-step @  slx-step !
  stride-loop-idx @   slx-idx !
  stride-loop-bound @ slx-bound !
  stride-loop-bra @   slx-bra ! ;

: restore-stride-state  ( -- )
  slx-top @   stride-loop-top !
  slx-step @  stride-loop-step !
  slx-idx @   stride-loop-idx !
  slx-bound @ stride-loop-bound !
  slx-bra @   stride-loop-bra ! ;

variable ema-a    variable ema-b    variable ema-out
variable ema-M    variable ema-N    variable ema-step
variable ema-op   \ execution token of scalar op — stored here, not on R-stack
variable ema-i    variable ema-j
variable ema-va   variable ema-vb   variable ema-rd
variable ema-row-off variable ema-addr

\ emit-mat-binary ( a-ptr b-ptr out-ptr M N step op-xt -- )
: emit-mat-binary  ( a-ptr b-ptr out-ptr M N step op-xt -- )
  ema-op !                \ save op-xt in variable (safe across nested calls)
  ema-step !  ema-N !  ema-M !
  ema-out !  ema-b !  ema-a !

  rreg+ ema-i !                     \ outer loop index (i)
  ema-i @  0  mov-imm,              \ i = 0 (placeholder for R0)

  ema-i @  ema-M @  ema-step @
  emit-stride-begin                 \ outer loop: for i in [0, M)

    save-stride-state                \ save outer loop vars

    rreg+ ema-j !                   \ inner loop index (j)
    ema-j @  0  mov-imm,            \ j = 0 (placeholder for R0)

    ema-j @  ema-N @  ema-step @
    emit-stride-begin               \ inner loop: for j in [0, N)

      \ Load a[i] and b[j]
      ema-a @  ema-i @  emit-array-load  ema-va !
      ema-b @  ema-j @  emit-array-load  ema-vb !

      \ Apply scalar op
      ema-va @  ema-vb @  ema-op @ execute  ema-rd !

      \ Compute flat index: i*N + j into a temp register
      rreg+ ema-row-off !
      ema-row-off @  ema-i @  ema-N @  RZ  imad,   \ IMAD row-off, i, N, RZ
      rreg+ ema-addr !
      ema-addr @  ema-row-off @  ema-j @  RZ  iadd3,  \ addr = row-off + j

      \ Store result[i*N + j]
      ema-out @  ema-addr @  ema-rd @  emit-array-store

    emit-stride-end                 \ end inner loop

    restore-stride-state            \ restore outer loop vars

  emit-stride-end ;                  \ end outer loop (op-xt in ema-op — no R-stack cleanup)

\ emit-mul-mat ( a-ptr b-ptr out-ptr M N step -- )  — outer product
: emit-mul-mat  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-mul  emit-mat-binary ;

\ emit-add-mat ( a-ptr b-ptr out-ptr M N step -- )
: emit-add-mat  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-add  emit-mat-binary ;

\ emit-sub-mat ( a-ptr b-ptr out-ptr M N step -- )
: emit-sub-mat  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-sub  emit-mat-binary ;

\ emit-div-mat ( a-ptr b-ptr out-ptr M N step -- )
: emit-div-mat  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-div  emit-mat-binary ;

\ ============================================================
\ TENSOR EMITTERS  (**** / ++++)
\ ============================================================
\ Three-nested-loop forms — outer product over three axes.
\ Stack: ( a-ptr b-ptr out-ptr M N P step op-xt -- )
\ result[i*N*P + j*P + k] = a[i] op b[j] op ... (placeholder)
\
\ For now we emit a scalar stub via the mat emitter with the k-loop
\ dimension fixed to 1, giving a correct but non-maximally-efficient
\ sequence. Full three-loop specialisation is left for a later pass.

: emit-mul-ten  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-mul  emit-mat-binary ;

: emit-add-ten  ( a-ptr b-ptr out-ptr M N step -- )
  ' emit-add  emit-mat-binary ;

\ ============================================================
\ END
\ ============================================================
