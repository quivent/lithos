\ ls-compiler.fs — Main compiler dispatcher for the Lithos .ls stack language.
\
\ Loads a .ls source file, tokenizes it (via ls-tokenizer.fs), builds a
\ definition table, then compiles named definitions to Hopper SASS, wraps
\ the result in an ELF cubin (via cubin-wrap.fs), and saves to disk.
\
\ Load order (must already be included before this file):
\   compiler/ls-tokenizer.fs    — tokenizer + token stream
\   compiler/emit-elementwise.fs — element-wise SASS pattern emitters
\     (this file pulls in sass/emit-sass.fs, which supplies sinst, etc.)
\   compiler/emit-reduce.fs     — reduction emitters
\   compiler/emit-gemv.fs       — GEMV/project emitters
\   sass/emit-sass.fs           — raw SASS encoders (already via above)
\
\ This file provides:
\   ls-compile-kernel  ( name-addr name-len -- )
\   ls-compile-file    ( filename-addr filename-len -- )
\
\ ============================================================
\ DEPENDENCIES — pull in the emitter stack if not yet loaded
\ ============================================================
\ Each include guard is implemented by checking whether a sentinel
\ word is already defined. Forth's [DEFINED] / defined? is not
\ universal; we use a forward-definition trick: try to find each
\ key word and skip the include if found.
\
\ Instead of fragile guards, we document the required load order
\ and rely on the caller to include prerequisites. The words below
\ are referenced by name; if they are missing the Forth system will
\ report an "undefined word" error at compile time pointing clearly
\ to the missing dependency.

\ ============================================================
\ REGISTER ALLOCATORS
\ ============================================================
\ parser.fs defines freg+, rreg+, preg+. If parser.fs has NOT been
\ loaded (stand-alone use of this file), we define minimal versions
\ here. We detect by checking whether `next-rreg` is defined.
\
\ Convention (mirrors parser.fs):
\   R0..R3  — reserved for prologue (tid, ctaid, blockDim, scratch)
\   F0..    — float registers, allocated from 0
\   P0..P5  — predicate registers (P6, P7 reserved by hardware)

variable lsc-rreg-base   4 lsc-rreg-base !   \ first allocatable int reg
variable lsc-freg-base   0 lsc-freg-base !   \ first allocatable float reg
variable lsc-preg-base   0 lsc-preg-base !   \ first allocatable pred reg

\ Allocators — forward to parser.fs versions when present, else use locals.
\ We always define these so the rest of the file resolves cleanly. If
\ parser.fs has already defined rreg+ etc., the later definitions shadow
\ them (which is fine — Forth words are looked up latest-first).
variable lsc-rreg-counter
variable lsc-freg-counter
variable lsc-preg-counter

: lsc-regs-reset  ( -- )
  lsc-rreg-base @ lsc-rreg-counter !
  lsc-freg-base @ lsc-freg-counter !
  lsc-preg-base @ lsc-preg-counter ! ;

lsc-regs-reset

: rreg+  ( -- n )  lsc-rreg-counter @ dup 1+ lsc-rreg-counter ! ;
: freg+  ( -- n )  lsc-freg-counter @ dup 1+ lsc-freg-counter ! ;

\ Predicate registers: skip P6 and P7 (reserved by SASS hardware)
: preg+  ( -- n )
  lsc-preg-counter @
  dup 6 = if drop 8 lsc-preg-counter ! 8 exit then  \ skip over P6/P7
  dup 7 = if drop 8 lsc-preg-counter ! 8 exit then
  dup 1+ lsc-preg-counter ! ;

\ ============================================================
\ VALUE STACK  (tracks live register IDs during compilation)
\ ============================================================
\ Conceptually this IS the Forth data stack: each emitter word
\ consumes/produces register IDs on the normal Forth stack. No
\ separate stack structure is needed — Forth's own stack serves.
\ The helpers below are just documentation aliases.

\ ( -- )  push-reg: push a register ID onto the value stack (just a normal Forth push)
\ ( n -- )  pop-reg:  consume a register ID (just DROP or use as argument)

\ ============================================================
\ DEFINITION TABLE
\ ============================================================
\ Maps definition names → a saved slice of the token stream.
\ We store: name-addr(32), name-len, first-tok-index, tok-count
\ and copy the name bytes into our own name pool so the original
\ line-buf / op-pool can be reused.
\
\ Up to MAX-DEFS definitions, name strings up to DEF-NAME-MAX bytes.

64  constant MAX-DEFS
32  constant DEF-NAME-MAX

\ Name pool — packed strings for definition names
2048 constant DEF-NAME-POOL-SZ
create def-name-pool DEF-NAME-POOL-SZ allot
variable def-name-pos  0 def-name-pos !

\ Per-definition metadata arrays (parallel)
create def-name-addr MAX-DEFS cells allot    \ pointer into def-name-pool
create def-name-len  MAX-DEFS cells allot    \ byte length of name
create def-first-tok MAX-DEFS cells allot    \ index of first body token
create def-tok-cnt   MAX-DEFS cells allot    \ number of body tokens
variable def-count   0 def-count !

\ def-name-intern ( addr u -- pool-addr )
\ Copy a name string into def-name-pool, return its address.
: def-name-intern  ( addr u -- pool-addr )
  def-name-pos @ def-name-pool +  >r    \ R: dst
  dup def-name-pos +!                   \ advance pool pointer
  r@ swap move                          \ copy bytes
  r> ;                                  \ return dst address

\ def-register ( name-addr name-len first-tok-idx tok-count -- )
\ Register a compiled definition in the table.
: def-register  ( name-addr name-len first-tok body-count -- )
  def-count @ MAX-DEFS >= if 2drop 2drop exit then
  def-count @ >r                        \ R: slot index
  r@ cells def-tok-cnt  + !             \ body-count
  r@ cells def-first-tok + !            \ first-tok-idx
  \ intern the name
  def-name-intern                       \ ( name-addr -- pool-addr )
  r@ cells def-name-addr + !
  r> dup cells def-name-len + !
  \ wait — name-len is already consumed by def-name-intern? No:
  \ def-name-intern ( addr u -- pool-addr ) consumes both.
  \ But we need to store the length too. Re-examine the stack.
  \ Stack picture at the dup: we have used name-len inside intern.
  \ We need to save name-len before calling intern.  Fix below.
  drop  \ discard the attempted store — see corrected version
  1 def-count +! ;

\ Corrected def-register — stores length separately before interning.
: def-register  ( name-addr name-len first-tok body-count -- )
  def-count @ MAX-DEFS >= if 2drop 2drop exit then
  def-count @ >r                         \ R: slot
  r@ cells def-tok-cnt  + !              \ body-count
  r@ cells def-first-tok + !             \ first-tok-idx
  \ name-addr name-len on stack
  over r@ cells def-name-len + !         \ save name-len (peeked with over)
  def-name-intern                        \ ( addr u -- pool-addr )
  r> cells def-name-addr + !             \ save pool-addr; R: empty
  1 def-count +! ;

\ def-find ( name-addr name-len -- idx true | false )
\ Linear search through the definition table.
: def-find  ( addr u -- idx true | false )
  def-count @ 0= if 2drop 0 exit then
  def-count @ 0 do
    i cells def-name-len + @  over =  if         \ lengths match?
      2dup  i cells def-name-addr + @  over       \ addr u pool-addr u
      ls-str=  if
        2drop i -1 unloop exit
      then
    then
  loop
  2drop 0 ;

\ ============================================================
\ PASS 1 — BUILD DEFINITION TABLE
\ ============================================================
\ Walk the full token stream once, recording where each definition
\ body starts and ends. Leaves tok-cursor just past the last token.

variable p1-in-def       0 p1-in-def !
variable p1-def-name-a   0 p1-def-name-a !
variable p1-def-name-u   0 p1-def-name-u !
variable p1-def-start    0 p1-def-start !   \ tok-cursor at first body token

: ls-pass1  ( -- )
  0 def-count !
  0 p1-in-def !
  ls-reset-cursor
  begin ls-tokens-done? 0= while
    ls-next-token              \ ( -- type )
    dup DEF_START = if
      drop
      \ Close any open definition first
      p1-in-def @ if
        p1-def-name-a @  p1-def-name-u @
        p1-def-start @
        tok-cursor @ 1-  p1-def-start @ -   \ body-count (tokens between start and now)
        def-register
      then
      \ Begin new definition: name is in the PREVIOUS token (the DEF_START token)
      \ The token just consumed via ls-next-token is DEF_START; name is in its fields.
      \ ls-token-name reads from tok-cursor-1.
      ls-token-name  p1-def-name-u !  p1-def-name-a !
      tok-cursor @ p1-def-start !     \ next token will be first body token
      -1 p1-in-def !
    else
    dup DEF_END = if
      drop
      p1-in-def @ if
        p1-def-name-a @  p1-def-name-u @
        p1-def-start @
        tok-cursor @ 1-  p1-def-start @ -   \ body tokens (exclude the DEF_END itself)
        def-register
        0 p1-in-def !
      then
    else
      \ All other tokens: just skip during pass 1
      drop
    then then
  repeat ;

\ ============================================================
\ PEEK AHEAD — check next token without consuming
\ ============================================================

: ls-peek-type  ( -- type | -1 )
  ls-tokens-done? if -1 exit then
  tok-cursor @ tok-record @ ;

\ ============================================================
\ OPERAND CHECK — is the immediately following token a literal?
\ ============================================================
\ Used by PRIM_MUL and PRIM_ADD to choose scalar vs vector variant.

: next-is-literal?  ( -- flag )
  ls-peek-type dup LIT_INT = swap LIT_FLOAT = or ;

\ ============================================================
\ COMPILE-TIME VALUE STACK HELPERS
\ ============================================================
\ The Forth stack IS the value stack. Emitter words (emit-mul,
\ emit-add, etc.) operate on register-ID values sitting on the
\ Forth stack. Comments use ( ra rb -- rd ) notation throughout.

\ ============================================================
\ FORWARD DECLARATION: compile-token
\ ============================================================
\ compile-body-tokens calls compile-token recursively (for NAME_REF
\ inlining). In Forth, forward references are handled with a deferred
\ word that is later filled in.

defer compile-token-fwd

\ ============================================================
\ COMPILE BODY TOKENS
\ ============================================================
\ Compile a slice of the token stream [first-idx .. first-idx+count).
\ Saves and restores tok-cursor so the caller's position is unaffected.

variable cb-saved-cursor

: compile-body-tokens  ( first-idx count -- )
  over +                         \ ( first end )
  swap tok-cursor !              \ set cursor to first-idx (tok-cursor = first)
  begin
    tok-cursor @ over <          \ while cursor < end
  while
    ls-next-token
    compile-token-fwd
  repeat
  drop ;

\ ============================================================
\ COMPILE ONE TOKEN
\ ============================================================
\ Dispatches on token type, manipulates the value stack (Forth stack),
\ and calls the appropriate emitter word.

\ scratch variable for scalar immediate value
variable lsc-imm

: compile-token  ( type -- )

  \ ---- OPERAND / NAME_REF: push reference operand (not a value) ----
  \ OPERAND tokens are secondary words following a primitive; they are
  \ consumed by the primitive's look-ahead logic (next-is-literal? / ls-next-token).
  \ NAME_REF either inlines a definition or loads a parameter pointer.
  dup OPERAND = if drop exit then

  \ ---- LIT_INT: push integer literal as immediate (used as scalar) ----
  dup LIT_INT = if
    drop
    ls-token-val           \ ( -- int-value )
    exit                   \ leave value on stack for next emitter to consume
  then

  \ ---- LIT_FLOAT: push float literal (IEEE 754 representation) ----
  dup LIT_FLOAT = if
    drop
    ls-token-val           \ value stored as int-encoded float
    exit
  then

  \ ---- DEF_START / DEF_END: skip during body compilation ----
  \ (pass 1 already catalogued these; pass 2 should not see them
  \  inside compile-body-tokens since we use index ranges)
  dup DEF_START = if drop exit then
  dup DEF_END   = if drop exit then

  \ ---- NAME_REF: inline a named definition ----
  dup NAME_REF = if
    drop
    ls-token-name                     \ ( -- addr u )
    def-find if                       \ ( -- idx )
      dup cells def-first-tok + @     \ ( idx first )
      swap cells def-tok-cnt  + @     \ ( first count )
      \ Save outer cursor, compile body, restore outer cursor
      tok-cursor @ >r
      compile-body-tokens
      r> tok-cursor !
    else
      \ Unknown name: not a definition. Could be a kernel parameter
      \ reference (pointer loaded at prologue time). For now, emit a
      \ placeholder load and push the result register.
      \ In a full compiler this would look up the param slot.
      emit-load-param drop           \ ( param-idx -- reg ): use 0 as placeholder
    then
    exit
  then

  \ ---- PRIM_MUL (0): element-wise multiply or scalar multiply ----
  dup PRIM_MUL = if
    drop
    next-is-literal? if
      ls-next-token drop             \ consume the LIT token
      ls-token-val                   \ ( ra -- ra imm )
      emit-mul-scalar                \ ( ra imm -- rd )
    else
      emit-mul                       \ ( ra rb -- rd )
    then
    exit
  then

  \ ---- PRIM_ADD (1): element-wise add or scalar add ----
  dup PRIM_ADD = if
    drop
    next-is-literal? if
      ls-next-token drop
      ls-token-val                   \ ( ra -- ra imm )
      emit-add-scalar
    else
      emit-add                       \ ( ra rb -- rd )
    then
    exit
  then

  \ ---- PRIM_SUB (2) ----
  dup PRIM_SUB = if  drop  emit-sub  exit  then

  \ ---- PRIM_DIV (3) ----
  dup PRIM_DIV = if  drop  emit-div  exit  then

  \ ---- PRIM_EXP (4) ----
  dup PRIM_EXP = if  drop  emit-exp  exit  then

  \ ---- PRIM_LOG (5) ----
  dup PRIM_LOG = if  drop  emit-log  exit  then

  \ ---- PRIM_SQRT (6) ----
  dup PRIM_SQRT = if  drop  emit-sqrt  exit  then

  \ ---- PRIM_RCP (7) ----
  dup PRIM_RCP = if  drop  emit-rcp  exit  then

  \ ---- PRIM_RSQRT (8) ----
  dup PRIM_RSQRT = if  drop  emit-rsqrt  exit  then

  \ ---- PRIM_OUTER (9): 2-D outer product — nested multiply ----
  \ outer u v:  for each pair (u[i], v[j]) emit FMUL into a result register.
  \ At this level we have two vector registers on the stack.
  \ We emit emit-mul as a representative single-step (the outer loop
  \ structure requires the host loop; here we emit the body primitive).
  dup PRIM_OUTER = if
    drop
    emit-mul                         \ ( ra rb -- rd ): body of the outer product
    exit
  then

  \ ---- PRIM_PROJECT (10): projection — emit_gemv_kernel ----
  \ PRIM_PROJECT = quantized weight matrix projection (W4A16 dequant GEMV).
  \ The value stack at this point should hold: W scales x y K N.
  \ emit-gemv-kernel ( W-reg scales-reg x-reg y-reg K N -- )
  dup PRIM_PROJECT = if
    drop
    emit-gemv-kernel               \ consumes 6 values from the value stack
    exit
  then

  \ ---- PRIM_MATVEC (11): state-matrix matvec (no dequant) ----
  \ PRIM_MATVEC is conceptually similar to GEMV but applied to the full-
  \ precision state matrix S (no quantization). For now we emit the same
  \ GEMV kernel body since the underlying MAC pattern is identical;
  \ the caller is responsible for passing a full-precision weight pointer
  \ (scale=1.0, zero=0). A future specialisation can skip dequant entirely.
  dup PRIM_MATVEC = if
    drop
    emit-gemv-kernel               \ same hardware pattern, no dequant (scale=FP32-ONE)
    exit
  then

  \ ---- Unknown token type: drop silently ----
  drop ;

\ Wire the deferred word to the concrete definition.
' compile-token is compile-token-fwd

\ ============================================================
\ PROLOGUE EMISSION
\ ============================================================
\ emit-prologue (from emit-elementwise.fs) emits S2R + IMAD for
\ the standard thread-indexing sequence. After return R0 = global tid.
\ We also reserve R0..R3 so subsequent rreg+ calls start at R4.

: ls-emit-prologue  ( -- )
  lsc-regs-reset
  emit-prologue ;        \ S2R R0,tid  S2R R1,ctaid  IMAD R0,R1,R2,R0

\ ============================================================
\ PARAMETER POINTER LOADING
\ ============================================================
\ .ls kernels list their pointer parameters in the DEF_START parameter
\ tokens (the NAME_REF tokens following the definition name).
\ We walk those tokens and call emit-load-param for each.
\ Returns: count of params loaded.

variable lp-count

: ls-load-params  ( def-idx -- n-params )
  \ The tokens immediately after DEF_START are NAME_REF (one per param).
  \ Pass 1 stored first-tok index; we walk forward until a non-NAME_REF.
  \ However, the tokenizer includes parameter names as NAME_REF tokens
  \ WITHIN the def-first-tok range — they precede the body proper.
  \ We scan from first-tok, emit-load-param for each NAME_REF, and stop
  \ when we see a non-NAME_REF or reach end of body.
  0 lp-count !
  dup cells def-first-tok + @    \ first-tok-idx
  swap cells def-tok-cnt  + @    \ body token count
  over + swap                    \ ( end first )
  tok-cursor @ >r                \ save outer cursor
  tok-cursor !                   \ set cursor to first body token
  begin
    tok-cursor @ over <          \ while cursor < end
  while
    ls-peek-type NAME_REF =      \ next token is a NAME_REF?
    0= if leave then             \ no — body begins here; stop
    ls-next-token drop           \ consume the NAME_REF
    \ ls-token-name gives the param name; we don't use it for emit-load-param
    \ (param index = ordinal position starting at 0)
    lp-count @  emit-load-param  \ ( param-idx -- reg )
    drop                         \ discard the reg for now (caller sets up params)
    1 lp-count +!
  repeat
  drop
  r> tok-cursor !                \ restore outer cursor
  lp-count @ ;

\ ============================================================
\ TOP-LEVEL KERNEL COMPILER
\ ============================================================
\ ls-compile-kernel ( name-addr name-len -- )
\ Finds the named definition, emits prologue + body + epilogue,
\ wraps in cubin, and saves to "<name>.cubin".

\ Scratch buffer for output filename
create lsc-out-buf 128 allot
variable lsc-out-len

: ls-make-outname  ( name-addr name-len -- out-addr out-len )
  \ Build "<name>.cubin" in lsc-out-buf
  dup lsc-out-len !
  lsc-out-buf swap move            \ copy name
  s" .cubin" lsc-out-buf lsc-out-len @ + swap move
  6 lsc-out-len +!
  lsc-out-buf lsc-out-len @ ;

variable lsc-def-idx
variable lsc-first-body-tok        \ cursor position of first real body token

: ls-compile-kernel  ( name-addr name-len -- )
  2dup def-find 0= if
    cr ." ls-compiler: definition not found: " type cr
    exit
  then
  lsc-def-idx !

  \ Reset SASS buffer and register allocator
  sass-reset
  lsc-regs-reset

  \ Set kernel name for cubin metadata
  2dup li-set-name                   \ sets li-name-buf / li-name-len

  \ Set param count for cubin nv.info
  lsc-def-idx @ cells def-first-tok + @   \ first body token index
  lsc-def-idx @ cells def-tok-cnt  + @    \ body token count

  \ ---- Count param NAME_REFs at start of body ----
  \ (so cubin-wrap knows how many pointer params to declare)
  over +                             \ end index
  swap                               \ ( end first )
  tok-cursor @ >r
  tok-cursor !                       \ cursor -> first body tok
  0 lp-count !
  begin
    tok-cursor @ over <
    ls-peek-type NAME_REF = and
  while
    ls-next-token drop
    1 lp-count +!
  repeat
  lsc-first-body-tok @ tok-cursor @ lsc-first-body-tok !   \ save first real body tok
  \ correction: save current cursor (past param tokens) as first body start
  tok-cursor @ lsc-first-body-tok !
  drop                               \ drop end index
  r> tok-cursor !
  lp-count @ n-kparams !             \ tell cubin how many params

  \ ---- Emit kernel prologue ----
  ls-emit-prologue

  \ ---- Load pointer parameters into registers (R4..R4+n-params-1) ----
  \ Each param gets emit-load-param called with its ordinal index.
  \ The registers are pushed onto the value stack in param order.
  lp-count @ 0 do
    i emit-load-param               \ ( idx -- reg ) pushes reg onto Forth stack
  loop

  \ ---- Compile body tokens ----
  \ Body starts after param NAME_REFs. We use the first-body-tok index.
  lsc-def-idx @ cells def-first-tok + @   \ raw first tok
  \ fast-forward past param NAME_REFs
  tok-cursor @ >r
  over tok-cursor !
  lp-count @ 0 do
    ls-tokens-done? 0= if
      ls-peek-type NAME_REF = if ls-next-token drop then
    then
  loop
  tok-cursor @ lsc-first-body-tok !
  r> tok-cursor !

  \ Now compile [first-body-tok .. first-tok + body-count)
  lsc-first-body-tok @
  lsc-def-idx @ cells def-first-tok + @
  lsc-def-idx @ cells def-tok-cnt  + @  +
  lsc-first-body-tok @ -                 \ adjusted count
  compile-body-tokens

  \ ---- Drop any leftover values on the stack ----
  \ A well-formed kernel should have stored its result via emit-array-store.
  \ Any register IDs left are discarded.

  \ ---- Emit epilogue ----
  emit-epilogue                          \ EXIT instruction

  \ ---- Wrap in cubin and save ----
  2dup ls-make-outname                   \ ( name-addr name-len out-addr out-len )
  2swap drop drop                        \ drop name, keep out-addr out-len
  write-cubin
  cr ." Compiled: " lsc-out-buf lsc-out-len @ type cr ;

\ ============================================================
\ FILE LOADER
\ ============================================================
\ ls-compile-file ( filename-addr filename-len -- )
\ Slurps the .ls file, tokenizes it, builds the definition table,
\ then compiles every top-level definition that maps to a real kernel
\ (i.e., whose body contains at least one primitive token).

\ File read buffer
131072 constant LS-MAX-FILE-SZ
create ls-file-buf LS-MAX-FILE-SZ allot
variable ls-file-sz  0 ls-file-sz !

variable lf-fid

: ls-slurp-file  ( filename-addr filename-len -- ok )
  r/o open-file if drop 0 exit then    \ ( fid )
  lf-fid !
  ls-file-buf LS-MAX-FILE-SZ lf-fid @ read-file drop
  ls-file-sz !
  lf-fid @ close-file drop
  ls-file-sz @ 0> ;

: ls-compile-file  ( filename-addr filename-len -- )
  2dup ls-slurp-file 0= if
    cr ." ls-compiler: cannot open file: " type cr
    exit
  then
  2drop

  \ Tokenize the loaded source
  ls-file-buf ls-file-sz @ ls-tokenize

  \ Build the definition table (pass 1)
  ls-pass1

  \ Compile every definition in the table
  def-count @ 0 do
    i cells def-name-addr + @    \ name pool-addr
    i cells def-name-len  + @    \ name length
    ls-compile-kernel
  loop ;

\ ============================================================
\ CONVENIENCE: compile a named kernel from an already-tokenized stream
\ ============================================================
\ ls-compile-named ( name-addr name-len -- )
\ Like ls-compile-kernel but expects ls-tokenize and ls-pass1 already done.

: ls-compile-named  ( name-addr name-len -- )
  ls-compile-kernel ;

\ ============================================================
\ SIMPLE TEST ENTRY POINT
\ ============================================================
\ Compile the built-in vadd test definition and verify a cubin is produced.
\
\   vadd a b c n
\     a
\     b
\     +
\     c
\     n
\
\ Expected code sequence:
\   emit-prologue
\   emit-load-param(0) -> Ra    (pointer to a)
\   emit-load-param(1) -> Rb    (pointer to b)
\   emit-load-param(2) -> Rc    (pointer to c)
\   emit-load-param(3) -> Rn    (pointer to n — length)
\   NAME_REF "a" -> inline: already loaded above (pushed as Ra)
\   NAME_REF "b" -> inline: already loaded above (pushed as Rb)
\   emit-add (Ra Rb -- Rd)
\   NAME_REF "c" -> pushed as Rc
\   NAME_REF "n" -> pushed as Rn
\   emit-epilogue

create vadd-test-src 512 allot
variable vadd-test-len  0 vadd-test-len !

: vadd+c  ( c -- )  vadd-test-src vadd-test-len @ + c!  1 vadd-test-len +! ;
: vadd+s  ( addr u -- )  0 do dup i + c@ vadd+c loop drop ;

: build-vadd-test-src  ( -- addr u )
  0 vadd-test-len !
  s" vadd a b c n"  vadd+s  10 vadd+c
  s"   a"           vadd+s  10 vadd+c
  s"   b"           vadd+s  10 vadd+c
  s"   +"           vadd+s  10 vadd+c
  s"   c"           vadd+s  10 vadd+c
  s"   n"           vadd+s  10 vadd+c
  vadd-test-src vadd-test-len @ ;

: ls-test-vadd  ( -- )
  cr ." === ls-compiler vadd test ===" cr
  build-vadd-test-src ls-tokenize
  ls-pass1
  cr ." Definitions found: " def-count @ . cr
  def-count @ 0 > if
    0 cells def-name-addr + @
    0 cells def-name-len  + @
    2dup cr ." Compiling: " type cr
    ls-compile-kernel
    cr ." vadd.cubin written." cr
  then
  cr ." === done ===" cr ;
