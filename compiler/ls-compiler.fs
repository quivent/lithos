\ ls-compiler.fs — Main compiler dispatcher for the Lithos .ls stack language.
\
\ Loads a .ls source, tokenizes it (ls-tokenizer.fs), builds a definition
\ table (pass 1), then compiles named definitions to Hopper SASS (pass 2),
\ wraps the result in an ELF cubin, and writes to disk.
\
\ Required load order (include before this file):
\   compiler/ls-tokenizer.fs      — tokenizer + token stream
\   compiler/emit-elementwise.fs  — element-wise SASS emitters
\     (transitively includes sass/emit-sass.fs)
\   compiler/emit-reduce.fs       — reduction emitters
\   compiler/emit-gemv.fs         — GEMV / project emitters
\   compiler/parser.fs            — provides li-set-name, li-name$,
\                                   n-kparams, freg+, rreg+, preg+
\   compiler/cubin-wrap.fs        — write-cubin
\
\ Public interface:
\   ls-compile-kernel ( name-addr name-len -- )
\   ls-compile-file   ( filename-addr filename-len -- )
\   ls-test-vadd      ( -- )   built-in smoke test

\ ============================================================
\ REGISTER ALLOCATORS
\ ============================================================
\ Shadow parser.fs's rreg+/freg+/preg+ with local versions that
\ use our own counter variables.  If parser.fs was loaded first its
\ definitions are overridden — behaviour is identical either way.
\ R0-R3 reserved (prologue: tid, ctaid, blockDim, scratch).
\ P6/P7 reserved by SASS hardware.

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
  dup 5 > if               \ counter > 5 means next would be P6 or P7
    drop 8                 \ skip to P8
    9 lsc-preg-ctr !       \ next call returns P9
    exit
  then
  dup 1+ lsc-preg-ctr ! ;

\ ============================================================
\ DEFINITION TABLE
\ ============================================================
\ Maps definition names to body token slices [first-idx, count).
\ Names are interned into defpool to survive tokenizer reuse.

variable lsc-rep-n   0 lsc-rep-n !   \ repeat count for current TOK_REPEAT unroll

64    constant MAX-DEFS
2048  constant DEFPOOL-SZ

create defpool   DEFPOOL-SZ allot
variable defpool-pos  0 defpool-pos !

create dt-nptr  MAX-DEFS cells allot   \ pointer into defpool
create dt-nlen  MAX-DEFS cells allot   \ name byte count
create dt-ftok  MAX-DEFS cells allot   \ first body token index
create dt-tcnt  MAX-DEFS cells allot   \ body token count
variable dt-n   0 dt-n !

\ dtnc-* scratch for dt-name-copy
variable dtnc-src  variable dtnc-len  variable dtnc-dst

\ dt-name-copy ( src-addr src-len -- dst-addr )
\ Copies src-len bytes from src-addr into defpool.  Returns dst-addr.
: dt-name-copy  ( src-addr src-len -- dst-addr )
  dtnc-len !  dtnc-src !                      \ save args
  defpool-pos @ defpool + dtnc-dst !           \ compute dst
  dtnc-len @ defpool-pos +!                    \ advance pool pointer
  dtnc-src @  dtnc-dst @  dtnc-len @  move    \ MOVE ( src dst u )
  dtnc-dst @ ;                                 \ return dst-addr

\ dtr-* scratch for dt-register
variable dtr-slot  variable dtr-na  variable dtr-nu

: dt-register  ( name-addr name-len first-tok body-count -- )
  dt-n @ MAX-DEFS >= if 2drop 2drop exit then
  dt-n @ dtr-slot !
  dtr-slot @ cells dt-tcnt + !           \ body-count
  dtr-slot @ cells dt-ftok + !           \ first-tok
  dtr-nu !  dtr-na !                     \ name-len name-addr
  dtr-nu @  dtr-slot @ cells dt-nlen + ! \ store name length
  dtr-na @  dtr-nu @  dt-name-copy       \ intern; ( -- dst-addr )
  dtr-slot @ cells dt-nptr + !            \ store pool pointer
  1 dt-n +! ;

\ dt-find scratch
variable dtf-src  variable dtf-u

\ dt-find ( addr u -- idx true | false )
\ ls-str= ( a1 u1 a2 u2 -- flag ) — both strings with their lengths.
: dt-find  ( addr u -- idx true | false )
  dtf-u !  dtf-src !
  dt-n @ 0= if 0 exit then
  dt-n @ 0 do
    i cells dt-nlen + @ dtf-u @ =  if         \ quick length pre-check
      dtf-src @  dtf-u @                       \ a1 u1
      i cells dt-nptr + @  dtf-u @             \ a2 u2
      ls-str=  if
        i -1 unloop exit
      then
    then
  loop
  0 ;

\ ============================================================
\ PASS 1 — CATALOGUE DEFINITIONS
\ ============================================================

variable p1-open   0 p1-open !
variable p1-na     0 p1-na !
variable p1-nu     0 p1-nu !
variable p1-bst    0 p1-bst !   \ first body token index

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
  tok-cursor @ tok-record @ ;

\ ============================================================
\ PARAM NAME TABLE
\ ============================================================
\ scan-params records each param's name and register ID.
\ compile-token uses param-find to push the right reg on demand.

16 constant MAX-PARAMS
20 constant PNAME-MAX

create param-regs   MAX-PARAMS cells allot
create param-nbufs  MAX-PARAMS PNAME-MAX * allot
create param-nlens  MAX-PARAMS cells allot
variable n-loaded-params  0 n-loaded-params !

\ pf-* scratch for param-find
variable pf-src  variable pf-u

: param-find  ( addr u -- reg | -1 )
  pf-u !  pf-src !
  n-loaded-params @ 0= if -1 exit then
  n-loaded-params @ 0 do
    i cells param-nlens + @ pf-u @ =  if         \ quick length check
      pf-src @  pf-u @                            \ a1 u1
      i PNAME-MAX * param-nbufs +  pf-u @         \ a2 u2
      ls-str=  if
        i cells param-regs + @ unloop exit
      then
    then
  loop
  -1 ;

\ scp-* scratch for scan-params
variable scp-end   variable scp-saved
variable scp-pa    variable scp-pu    \ current param name

: scan-params  ( def-idx -- first-body-cursor )
  dup  cells dt-ftok + @              \ ( def-idx first )
  swap cells dt-tcnt + @              \ ( first count )
  over + scp-end !                   \ end = first + count; stack: ( first )
  tok-cursor @ scp-saved !           \ save outer cursor
  tok-cursor !                       \ cursor = first
  0 n-loaded-params !

  begin
    tok-cursor @ scp-end @ <
    ls-peek-type NAME_REF = and
  while
    ls-next-token drop
    ls-token-name  scp-pu !  scp-pa !

    scp-pu @  n-loaded-params @ cells param-nlens + !
    scp-pa @  n-loaded-params @ PNAME-MAX * param-nbufs +  scp-pu @  move

    n-loaded-params @  emit-load-param
    n-loaded-params @ cells param-regs + !
    1 n-loaded-params +!
  repeat

  tok-cursor @
  scp-saved @ tok-cursor ! ;

\ ============================================================
\ FORWARD DECLARATION (compile-body-tokens ↔ compile-token)
\ ============================================================

defer compile-token-impl

\ ============================================================
\ COMPILE A BODY SLICE
\ ============================================================

variable cbs-saved

: compile-body-tokens  ( first-idx tok-count -- )
  over + >r
  tok-cursor @ cbs-saved !
  tok-cursor !
  begin tok-cursor @ r@ < while
    ls-next-token  compile-token-impl
  repeat
  r> drop
  cbs-saved @ tok-cursor ! ;

\ ============================================================
\ COMPILE ONE TOKEN
\ ============================================================

\ ============================================================
\ COOPERATIVE GRID-SYNC BARRIER STUB
\ ============================================================
\ emit-grid-sync  ( -- )
\ Emitted between successive unrolled iterations of a NAME x N
\ body (e.g. Layer x 64).  The Forth value stack at the call site
\ holds live registers carrying the inter-layer hidden state.
\ The barrier instruction is a pure synchronisation side-effect
\ and must not modify any GP register.
\
\ STUB: replace nop, with the real SM90 cooperative grid-sync
\ SASS sequence when ready:
\   MEMBAR.SC.CLUSTER ; BAR.SYNC ; MEMBAR.SC.CLUSTER
\ (or the SM90a grid-scope variant).
\
: emit-grid-sync  ( -- )
  nop, ;                      \ placeholder -- real barrier goes here

: compile-token  ( type -- )

  dup OPERAND   = if drop exit then
  dup DEF_START = if drop exit then
  dup DEF_END   = if drop exit then

  dup LIT_INT   = if drop  ls-token-val  exit then
  dup LIT_FLOAT = if drop  ls-token-val  exit then

  dup NAME_REF = if
    drop
    ls-token-name                       \ ( -- addr u )
    \ 1. Param name?
    2dup param-find dup -1 <> if        \ ( addr u reg ) if found
      -rot 2drop                         \ drop addr/u, keep reg
      exit
    then
    drop                                 \ discard the -1
    \ 2. User definition?
    dt-find if                           \ ( -- idx )
      dup  cells dt-ftok + @
      swap cells dt-tcnt + @
      compile-body-tokens
    else
      0 emit-load-param                  \ placeholder
    then
    exit
  then

  dup PRIM_MUL = if
    drop
    ls-peek-type dup LIT_INT = swap LIT_FLOAT = or if
      ls-next-token drop  ls-token-val
      emit-mul-scalar
    else
      emit-mul
    then
    exit
  then

  dup PRIM_ADD = if
    drop
    ls-peek-type dup LIT_INT = swap LIT_FLOAT = or if
      ls-next-token drop  ls-token-val
      emit-add-scalar
    else
      emit-add
    then
    exit
  then

  dup PRIM_SUB   = if drop  emit-sub    exit then
  dup PRIM_DIV   = if drop  emit-div    exit then
  dup PRIM_EXP   = if drop  emit-exp    exit then
  dup PRIM_LOG   = if drop  emit-log    exit then
  dup PRIM_SQRT  = if drop  emit-sqrt   exit then
  dup PRIM_RCP   = if drop  emit-rcp    exit then
  dup PRIM_RSQRT = if drop  emit-rsqrt  exit then

  \ PRIM_OUTER (9): emit the per-element multiply of the outer product body
  dup PRIM_OUTER = if
    drop  emit-mul  exit
  then

  \ PRIM_PROJECT (10): GPTQ W4A16 quantized GEMV
  \ Value stack: ( W-reg scales-reg x-reg y-reg K N )
  dup PRIM_PROJECT = if
    drop  emit-gemv-kernel  exit
  then

  \ PRIM_MATVEC (11): full-precision state-matrix × vector
  \ Same hardware path; caller uses scale=1.0, zero=0.
  dup PRIM_MATVEC = if
    drop  emit-gemv-kernel  exit
  then

  \ PRIM_SUM (12): Σ — cross-thread warp+smem reduction
  \ Each thread holds a partial scalar (result of per-element multiply).
  \ Emits intra-warp butterfly + cross-warp smem broadcast.
  \ Value stack: ( partial-reg -- result-reg )
  dup PRIM_SUM = if
    drop  emit-Σ  exit
  then

  \ ---- TOK_REPEAT (19): NAME x N  inline N times with grid-sync barriers ----
  \ Token layout (set by tokenizer parse-body-line):
  \   ls-token-val      = N  (repeat count)
  \   ls-token-name     = name of definition to inline
  \
  \ Register threading:
  \   The Forth value stack carries the model state across iterations.
  \   After each compile-body-tokens call the stack holds the output
  \   register(s) produced by that iteration.  Those registers are
  \   exactly the inputs consumed by the next iteration.  The stack
  \   threads state automatically -- no separate bookkeeping needed.
  \
  \ emit-grid-sync contract:
  \   Called BETWEEN iterations (not after the last one).
  \   On entry the value stack holds live registers carrying the
  \   inter-layer hidden state.  The barrier is a pure side-effect
  \   and must leave all GP registers undisturbed.
  dup TOK_REPEAT = if
    drop
    ls-token-val              \ ( -- N )
    ls-token-name             \ ( -- N name-addr name-len )
    dt-find 0= if             \ definition not in table -- skip
      drop exit               \ pop N; 0= consumed the false flag
    then                      \ ( -- N def-idx )
    dup  cells dt-ftok + @    \ ( N def-idx first-tok )
    swap cells dt-tcnt + @    \ ( N first-tok body-count )
    rot                       \ ( first-tok body-count N )
    lsc-rep-n !               \ save N; stack: ( first-tok body-count )
    lsc-rep-n @ 0 do
      \ Replay the body token slice for iteration i.
      \ 2dup keeps first-tok/body-count for subsequent iterations.
      2dup compile-body-tokens
      \ Grid-sync barrier between iterations, not after the last.
      i lsc-rep-n @ 1- < if emit-grid-sync then
    loop
    2drop                     \ discard first-tok / body-count
    exit
  then

  drop ;

' compile-token is compile-token-impl

\ ============================================================
\ ls-compile-kernel
\ ============================================================

variable lk-def-idx
variable lk-first-body

create lk-outbuf  128 allot
variable lk-outlen

\ lkn-* scratch for lk-mk-outname
variable lkn-len

: lk-mk-outname  ( name-addr name-len -- )
  lkn-len !                              \ save name length
  lk-outbuf  lkn-len @  move            \ MOVE ( src dst u ): name-addr lk-outbuf lkn-len@
  \ Fix: MOVE ( src dst u ) = ( name-addr lk-outbuf lkn-len@ )
  \ After "lkn-len !": stack = ( name-addr )
  \ "lk-outbuf" -> ( name-addr lk-outbuf )
  \ "lkn-len @" -> ( name-addr lk-outbuf lkn-len@ )
  \ "move" consumes ( src dst u ) = ( name-addr lk-outbuf lkn-len@ ) ✓
  lkn-len @  lk-outlen !                \ name part of output length
  s" .cubin"
  lk-outbuf lk-outlen @ + swap          \ dst=lk-outbuf+name-len; swap with u=6
  move                                   \ MOVE ( ".cubin"-addr dst 6 ) ✓
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

  \ Set kernel name for cubin ELF metadata (li-set-name from parser.fs)
  lk-outbuf  lk-outlen @ 6 -
  li-set-name

  \ Scan leading param NAME_REFs; emit-load-param for each
  lk-def-idx @ scan-params lk-first-body !
  n-loaded-params @ n-kparams !

  emit-prologue

  \ Compile body tokens (params are pushed on demand by NAME_REF dispatch)
  lk-first-body @
  lk-def-idx @ cells dt-ftok + @
  lk-def-idx @ cells dt-tcnt + @ +
  lk-first-body @ -
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
\ Expected compilation flow:
\   emit-prologue
\   emit-load-param(0..3) -> R4..R7 (pointers: a b c n)
\   NAME_REF "a" -> param-find -> push R4
\   NAME_REF "b" -> param-find -> push R5
\   PRIM_ADD     -> emit-add(R4,R5) -> R8
\   NAME_REF "c" -> param-find -> push R6
\   NAME_REF "n" -> param-find -> push R7
\   emit-epilogue

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
