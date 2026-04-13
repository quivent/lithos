\ ls-compiler.fs — Main compiler dispatcher for the Lithos .ls stack language.
\
\ Loads a .ls source, tokenizes it (ls-tokenizer.fs), builds a definition
\ table (pass 1), then compiles named definitions to Hopper SASS (pass 2),
\ wraps the result in an ELF cubin, and writes to disk.
\
\ Required load order (include before this file):
\   compiler/ls-tokenizer.fs      — tokenizer + token stream
\   compiler/emit-elementwise.fs  — element-wise GPU instruction emitters
\     (transitively includes gpu/emit.fs)
\   compiler/emit-reduce.fs       — reduction emitters
\   compiler/emit-gemv.fs         — GEMV / project emitters
\   compiler/parser.fs            — provides li-set-name, li-name$,
\                                   n-kparams, freg+, rreg+, preg+
\   compiler/elf-wrap.fs          — write-cubin / write-elf
\
\ Public interface:
\   ls-compile-kernel ( name-addr name-len -- )
\   ls-compile-file   ( filename-addr filename-len -- )
\   ls-test-vadd      ( -- )   built-in smoke test

\ ============================================================
\ REGISTER ALLOCATORS
\ ============================================================
\ Shadow parser.fs rreg+/freg+/preg+ with local versions using our
\ own counters.  If parser.fs is loaded first its definitions are
\ overridden (Forth resolves to the most-recently-defined word).
\ R0-R3 reserved (prologue). P6/P7 reserved by SASS hardware.

variable lsc-rreg-ctr
variable lsc-freg-ctr
variable lsc-preg-ctr

: lsc-regs-reset  ( -- )
  4 lsc-rreg-ctr !
  0 lsc-freg-ctr !
  0 lsc-preg-ctr ! ;

lsc-regs-reset

: rreg+  ( -- n )  lsc-rreg-ctr @ dup 1+ lsc-rreg-ctr ! ;
: freg+  ( -- n )  lsc-freg-ctr @ dup 1+ lsc-freg-ctr ! ;

: preg+  ( -- n )
  lsc-preg-ctr @
  dup 5 > if         \ counter > 5 -> next would be P6 or P7 (reserved)
    drop 8           \ skip to P8
    9 lsc-preg-ctr ! \ advance past P8 for next call
    exit
  then
  dup 1+ lsc-preg-ctr ! ;

\ ============================================================
\ DEFINITION TABLE
\ ============================================================
\ Maps definition names -> body token slices [first-idx, count).
\ Names are interned into defpool to survive tokenizer buffer reuse.

64    constant MAX-DEFS
2048  constant DEFPOOL-SZ

create defpool   DEFPOOL-SZ allot
variable defpool-pos  0 defpool-pos !

create dt-nptr  MAX-DEFS cells allot   \ pointer into defpool
create dt-nlen  MAX-DEFS cells allot   \ name byte count
create dt-ftok  MAX-DEFS cells allot   \ first body token index (in tok-buf)
create dt-tcnt  MAX-DEFS cells allot   \ body token count
variable dt-n   0 dt-n !

\ Scratch variables for dt-name-copy
variable dtnc-src  variable dtnc-len  variable dtnc-dst

\ dt-name-copy ( src-addr src-len -- dst-addr )
\ Appends src-len bytes from src-addr to defpool; returns dst-addr.
\ MOVE ( c-addr1 c-addr2 u -- ) copies u bytes from c-addr1 to c-addr2.
: dt-name-copy  ( src-addr src-len -- dst-addr )
  dtnc-len !  dtnc-src !                      \ stash args
  defpool-pos @ defpool + dtnc-dst !           \ compute dst
  dtnc-len @ defpool-pos +!                    \ advance pool write cursor
  dtnc-src @  dtnc-dst @  dtnc-len @  move    \ MOVE ( src dst u )
  dtnc-dst @ ;                                 \ return dst-addr

\ Scratch variables for dt-register
variable dtr-slot  variable dtr-na  variable dtr-nu

: dt-register  ( name-addr name-len first-tok body-count -- )
  dt-n @ MAX-DEFS >= if 2drop 2drop exit then
  dt-n @ dtr-slot !
  dtr-slot @ cells dt-tcnt + !            \ body-count
  dtr-slot @ cells dt-ftok + !            \ first-tok
  dtr-nu !  dtr-na !                      \ save name-len and name-addr
  dtr-nu @  dtr-slot @ cells dt-nlen + !  \ store length
  dtr-na @  dtr-nu @  dt-name-copy        \ intern name; returns dst-addr
  dtr-slot @ cells dt-nptr + !             \ store pool pointer
  1 dt-n +! ;

\ Scratch variables for dt-find
variable dtf-src  variable dtf-u

\ dt-find ( addr u -- idx true | false )
\ ls-str= ( a1 u1 a2 u2 -- flag ) both strings with lengths.
: dt-find  ( addr u -- idx true | false )
  dtf-u !  dtf-src !
  dt-n @ 0= if 0 exit then
  dt-n @ 0 do
    i cells dt-nlen + @ dtf-u @ =  if    \ quick length pre-check
      dtf-src @  dtf-u @                  \ a1 u1
      i cells dt-nptr + @  dtf-u @        \ a2 u2
      ls-str=  if
        i -1 unloop exit
      then
    then
  loop
  0 ;

\ ============================================================
\ PASS 1 - CATALOGUE DEFINITIONS
\ ============================================================

variable p1-open   0 p1-open !
variable p1-na     0 p1-na !     \ open def: name addr (in tokenizer pool)
variable p1-nu     0 p1-nu !     \ open def: name len
variable p1-bst    0 p1-bst !   \ open def: first body token index

: p1-close  ( -- )
  p1-na @  p1-nu @
  p1-bst @
  tok-cursor @ 1-  p1-bst @ -   \ body count (exclude current token)
  dt-register
  0 p1-open ! ;

: ls-pass1  ( -- )
  0 dt-n !  0 defpool-pos !
  0 p1-open !
  ls-reset-cursor
  begin ls-tokens-done? 0= while
    ls-next-token
    dup DEF_START = if
      drop
      p1-open @ if p1-close then
      ls-token-name  p1-nu !  p1-na !
      tok-cursor @ p1-bst !
      -1 p1-open !
    else
    dup DEF_END = if
      drop
      p1-open @ if p1-close then
    else
      drop
    then then
  repeat
  p1-open @ if
    p1-na @  p1-nu @
    p1-bst @
    tok-cursor @  p1-bst @ -
    dt-register
    0 p1-open !
  then ;

\ ============================================================
\ PEEK
\ ============================================================

: ls-peek-type  ( -- type | -1 )
  ls-tokens-done? if -1 exit then
  tok-cursor @ tok-record @ ;    \ field 0 of the record = token type

\ ============================================================
\ PARAM NAME TABLE
\ ============================================================
\ scan-params records each param name and its register ID.
\ compile-token uses param-find to push the correct reg on demand.

16 constant MAX-PARAMS
20 constant PNAME-MAX      \ max bytes per param name

create param-regs   MAX-PARAMS cells allot
create param-nbufs  MAX-PARAMS PNAME-MAX * allot
create param-nlens  MAX-PARAMS cells allot
variable n-loaded-params  0 n-loaded-params !

\ Scratch for param-find
variable pf-src  variable pf-u

: param-find  ( addr u -- reg | -1 )
  pf-u !  pf-src !
  n-loaded-params @ 0= if -1 exit then
  n-loaded-params @ 0 do
    i cells param-nlens + @ pf-u @ =  if
      pf-src @  pf-u @                    \ a1 u1
      i PNAME-MAX * param-nbufs +  pf-u @ \ a2 u2
      ls-str=  if
        i cells param-regs + @ unloop exit
      then
    then
  loop
  -1 ;

\ Scratch for scan-params
variable scp-end   variable scp-saved
variable scp-pa    variable scp-pu

: scan-params  ( def-idx -- first-body-cursor )
  dup  cells dt-ftok + @        \ ( def-idx first-tok )
  swap cells dt-tcnt + @        \ ( first-tok body-count )
  over + scp-end !              \ end = first + count; stack: ( first-tok )
  tok-cursor @ scp-saved !      \ save outer cursor
  tok-cursor !                  \ cursor = first-tok
  0 n-loaded-params !

  begin
    tok-cursor @ scp-end @ <
    ls-peek-type NAME_REF = and
  while
    ls-next-token drop              \ consume NAME_REF
    ls-token-name  scp-pu !  scp-pa !

    \ Store param length
    scp-pu @  n-loaded-params @ cells param-nlens + !
    \ Copy param name bytes: MOVE ( src dst u )
    scp-pa @
    n-loaded-params @ PNAME-MAX * param-nbufs +
    scp-pu @
    move

    \ Emit param load; store result register
    n-loaded-params @  emit-load-param
    n-loaded-params @ cells param-regs + !
    1 n-loaded-params +!
  repeat

  tok-cursor @                   \ return: first real body token cursor
  scp-saved @ tok-cursor ! ;    \ restore outer cursor

\ ============================================================
\ REPEAT COUNTER  (used by TOK_REPEAT dispatch in compile-token)
\ ============================================================

variable lsc-rep-n  0 lsc-rep-n !

\ ============================================================
\ COOPERATIVE GRID-SYNC BARRIER STUB
\ ============================================================
\ emit-grid-sync ( -- )
\ Placed here (before compile-token) because compile-token references it.
\ Placeholder: emits a NOP.  Replace with the real SM90 cooperative
\ grid-sync sequence when the encoder is available:
\   MEMBAR.SC.CLUSTER ; BAR.SYNC ; MEMBAR.SC.CLUSTER

: emit-grid-sync  ( -- )  nop, ;

\ ============================================================
\ FORWARD DECLARATION
\ ============================================================
\ compile-body-tokens calls compile-token-impl, and the NAME_REF
\ handler in compile-token calls compile-body-tokens to inline
\ definitions.  Break the cycle with a deferred word.

defer compile-token-impl

\ ============================================================
\ COMPILE A BODY SLICE
\ ============================================================

variable cbs-saved

: compile-body-tokens  ( first-idx tok-count -- )
  over + >r                       \ R: end-idx;  stack: ( first-idx )
  tok-cursor @ cbs-saved !        \ save outer cursor
  tok-cursor !                    \ cursor = first-idx
  begin tok-cursor @ r@ < while
    ls-next-token  compile-token-impl
  repeat
  r> drop
  cbs-saved @ tok-cursor ! ;

\ ============================================================
\ COMPILE ONE TOKEN
\ ============================================================

: compile-token  ( type -- )

  \ Structural tokens: no code emitted
  dup OPERAND   = if drop exit then
  dup DEF_START = if drop exit then
  dup DEF_END   = if drop exit then

  \ Literals: push value onto the value stack for the next emitter
  dup LIT_INT   = if drop  ls-token-val  exit then
  dup LIT_FLOAT = if drop  ls-token-val  exit then

  \ NAME_REF: param push -> definition inline -> placeholder
  dup NAME_REF = if
    drop
    ls-token-name                        \ ( -- addr u )
    \ 1. Is it a kernel param?
    2dup param-find dup -1 <> if         \ found: ( addr u reg )
      -rot 2drop                          \ drop addr/u, leave reg
      exit
    then
    drop                                  \ discard -1
    \ 2. Is it a user definition?
    dt-find if                            \ ( -- idx )
      dup  cells dt-ftok + @
      swap cells dt-tcnt + @
      compile-body-tokens                 \ inline body (no value left on stack)
    else
      0 emit-load-param                   \ placeholder param load
    then
    exit
  then

  \ PRIM_MUL (0): scalar multiply
  dup PRIM_MUL = if drop  emit-mul  exit then

  \ PRIM_ADD (1): scalar add
  dup PRIM_ADD = if drop  emit-add  exit then

  dup PRIM_SUB   = if drop  emit-sub    exit then   \ (2)
  dup PRIM_DIV   = if drop  emit-div    exit then   \ (3)
  dup PRIM_EXP   = if drop  emit-exp    exit then   \ (4)
  dup PRIM_LOG   = if drop  emit-log    exit then   \ (5)
  dup PRIM_SQRT  = if drop  emit-sqrt   exit then   \ (6)
  dup PRIM_RCP   = if drop  emit-rcp    exit then   \ (7)
  dup PRIM_RSQRT = if drop  emit-rsqrt  exit then   \ (8)

  \ PRIM_OUTER (9): legacy — maps to PRIM_MUL_MAT (outer product)
  dup PRIM_OUTER = if drop  emit-mul-mat  exit then

  \ Dimensional operators — repeated symbol forms (20-29)
  dup PRIM_MUL_VEC = if drop  emit-mul-vec  exit then   \ **
  dup PRIM_MUL_MAT = if drop  emit-mul-mat  exit then   \ ***
  dup PRIM_MUL_TEN = if drop  emit-mul-ten  exit then   \ ****
  dup PRIM_ADD_VEC = if drop  emit-add-vec  exit then   \ ++
  dup PRIM_ADD_MAT = if drop  emit-add-mat  exit then   \ +++
  dup PRIM_ADD_TEN = if drop  emit-add-ten  exit then   \ ++++
  dup PRIM_SUB_VEC = if drop  emit-sub-vec  exit then   \ --
  dup PRIM_SUB_MAT = if drop  emit-sub-mat  exit then   \ ---
  dup PRIM_DIV_VEC = if drop  emit-div-vec  exit then   \ //
  dup PRIM_DIV_MAT = if drop  emit-div-mat  exit then   \ ///

  \ PRIM_PROJECT (10): GPTQ W4A16 quantized GEMV
  \ Value stack: ( W-reg scales-reg x-reg y-reg K N )
  dup PRIM_PROJECT = if drop  emit-gemv-kernel  exit then

  \ PRIM_MATVEC (11): full-precision state-matrix x vector
  \ Same hardware path; caller provides scale=1.0, zero=0.
  dup PRIM_MATVEC = if drop  emit-gemv-kernel  exit then

  \ PRIM_SUM (12): Sigma — cross-thread reduction to a scalar.
  \ Value stack: ( partial-reg -- result-reg )
  \ emit-Sigma from emit-reduce.fs: warp butterfly + cross-warp smem.
  dup PRIM_SUM = if drop  emit-Σ  exit then

  \ TOK_REPEAT (19): NAME x N — inline a definition N times.
  \ Token fields: ls-token-val=N, ls-token-name=def-name.
  \ Emits a grid-sync barrier between iterations (not after the last).
  \ The Forth value stack threads the hidden-state register(s) across
  \ iterations automatically — no separate bookkeeping needed.
  dup TOK_REPEAT = if
    drop
    ls-token-val                     \ ( -- N )
    ls-token-name                    \ ( -- N name-addr name-len )
    dt-find 0= if                    \ not found -> skip
      drop exit                      \ drop N (0= already consumed the false flag)
    then                             \ ( -- N def-idx )
    dup  cells dt-ftok + @           \ ( N def-idx first-tok )
    swap cells dt-tcnt + @           \ ( N first-tok body-count )
    rot                              \ ( first-tok body-count N )
    lsc-rep-n !                      \ save N; stack: ( first-tok body-count )
    lsc-rep-n @ 0 do
      2dup compile-body-tokens        \ inline body for iteration i
      i lsc-rep-n @ 1- < if
        emit-grid-sync               \ barrier between iterations only
      then
    loop
    2drop                            \ discard first-tok / body-count
    exit
  then

  drop ;   \ unknown token type: silently discard

' compile-token is compile-token-impl

\ ============================================================
\ ls-compile-kernel
\ ============================================================
\ Compiles one named definition to SASS, wraps in ELF cubin,
\ and writes "<kernel-name>.cubin" to the current directory.

variable lk-def-idx
variable lk-first-body
variable lkn-len

create lk-outbuf  128 allot
variable lk-outlen

\ lk-mk-outname ( name-addr name-len -- )
\ Builds "<name>.cubin" in lk-outbuf / lk-outlen.
: lk-mk-outname  ( name-addr name-len -- )
  lkn-len !                          \ save name length; stack: ( name-addr )
  lk-outbuf  lkn-len @  move         \ MOVE ( src dst u ): name-addr lk-outbuf lkn-len@
  lkn-len @ lk-outlen !              \ length = name-len so far
  s" .cubin"                          \ ( ".cubin"-addr 6 )
  lk-outbuf lk-outlen @ + swap        \ ( ".cubin"-addr lk-outbuf+len 6 ) after swap
  \ swap makes TOS the u=6, second=dst: MOVE ( src dst u ) needs 3 items.
  \ Stack before swap: ( ".cubin"-addr 6 dst ), swap: ( ".cubin"-addr dst 6 )
  move
  6 lk-outlen +! ;

: ls-compile-kernel  ( name-addr name-len -- )
  2dup dt-find 0= if
    cr ." ls-compiler: definition not found: " type cr  exit
  then
  lk-def-idx !
  2dup lk-mk-outname
  2drop

  sass-reset
  lsc-regs-reset

  \ Set kernel name for cubin ELF metadata
  \ li-set-name ( addr u -- ) is provided by parser.fs
  lk-outbuf  lk-outlen @ 6 -
  li-set-name

  \ Scan leading param NAME_REFs: emit-load-param for each
  lk-def-idx @ scan-params lk-first-body !
  n-loaded-params @ n-kparams !

  emit-prologue

  \ Compile body tokens.  Param NAME_REFs are resolved via param-find
  \ which pushes the pre-loaded register ID.
  lk-first-body @                         \ first real body token cursor
  lk-def-idx @ cells dt-ftok + @
  lk-def-idx @ cells dt-tcnt + @ +       \ end cursor
  lk-first-body @ -                       \ body count from first-body to end
  compile-body-tokens

  emit-epilogue

  lk-outbuf  lk-outlen @
  write-cubin
  cr ." ls-compiler: wrote " lk-outbuf lk-outlen @ type cr ;

\ ============================================================
\ ls-compile-file
\ ============================================================

65536 constant LSF-MAX
create lsf-buf  LSF-MAX allot
variable lsf-len  0 lsf-len !
variable lsf-fd

: ls-compile-file  ( filename-addr filename-len -- )
  r/o open-file if
    drop  cr ." ls-compiler: cannot open file" cr  exit
  then
  lsf-fd !
  lsf-buf LSF-MAX lsf-fd @ read-file drop  lsf-len !
  lsf-fd @ close-file drop
  lsf-len @ 0= if  cr ." ls-compiler: empty file" cr  exit  then

  lsf-buf lsf-len @ ls-tokenize
  ls-pass1

  cr ." ls-compiler: " dt-n @ . ." definitions found" cr

  dt-n @ 0 do
    i cells dt-nptr + @
    i cells dt-nlen + @
    ls-compile-kernel
  loop ;

\ ============================================================
\ SMOKE TEST: vadd
\ ============================================================
\ Source text (embedded in memory, no file I/O):
\
\   vadd a b c n
\     a              <- NAME_REF: push param-reg for a
\     b              <- NAME_REF: push param-reg for b
\     +              <- PRIM_ADD: emit-add -> result reg
\     c              <- NAME_REF: push param-reg for c
\     n              <- NAME_REF: push param-reg for n
\
\ Expected SASS sequence:
\   S2R R0, SR_TID.X ; S2R R1, SR_CTAID.X ; IMAD R0, R1, R2, R0  (prologue)
\   MOV R4, 0        (emit-load-param placeholder for a -> R4)
\   MOV R5, 0        (emit-load-param placeholder for b -> R5)
\   MOV R6, 0        (emit-load-param placeholder for c -> R6)
\   MOV R7, 0        (emit-load-param placeholder for n -> R7)
\   FADD R8, R4, R5  (emit-add; R4 and R5 pushed by NAME_REF "a"/"b")
\   EXIT             (epilogue)

create lsc-tsrc  512 allot
variable lsc-tsrc-len  0 lsc-tsrc-len !

: lsct+c  ( c -- )  lsc-tsrc lsc-tsrc-len @ + c!  1 lsc-tsrc-len +! ;
: lsct+s  ( addr u -- )  0 do dup i + c@ lsct+c loop drop ;

: lsc-build-vadd-src  ( -- addr u )
  0 lsc-tsrc-len !
  s" vadd a b c n" lsct+s  10 lsct+c
  s"   a"          lsct+s  10 lsct+c
  s"   b"          lsct+s  10 lsct+c
  s"   +"          lsct+s  10 lsct+c
  s"   c"          lsct+s  10 lsct+c
  s"   n"          lsct+s  10 lsct+c
  lsc-tsrc lsc-tsrc-len @ ;

: ls-test-vadd  ( -- )
  cr ." === ls-compiler vadd smoke test ===" cr
  lsc-build-vadd-src ls-tokenize
  ls-pass1
  cr ." Definitions: " dt-n @ . cr
  dt-n @ 0> if
    0 cells dt-nptr + @
    0 cells dt-nlen + @
    2dup ." Compiling: " type cr
    ls-compile-kernel
  then
  cr ." === done ===" cr ;
