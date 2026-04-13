\ parser.fs — Lithos math-function parser (extended)
\
\ A Lithos program is a collection of math function definitions.
\ Syntax:
\
\   fn NAME param1 param2 ... -> output1 output2 ...
\       each VAR
\           output [ VAR ] = param1 [ VAR ] + param2 [ VAR ]
\
\ Extended features:
\   for k 0 K 16          — loop with counter, bound, step
\   shr, and, or, xor     — integer/bitwise ops
\   shfl.bfly              — warp shuffle (butterfly)
\   shared buf 5120 f32    — shared memory declaration
\   param n u32            — scalar parameter (u32 or f32)
\   exp, rcp, rsqrt, sqrt, sin, cos — math intrinsics
\   1.0, 0.5, -8.0        — float constants (IEEE 754 hex)
\   f32>s32, u32>f32 etc   — type conversions
\   @p0 bra DONE           — predicated branch
\   if>= tid n exit        — bounds check / early exit

\ Load core vocabulary (register pools, regs-reset).
s" /home/ubuntu/lithos/core.fs" included

variable li-mode     0 li-mode !
variable li-defs     0 li-defs !
variable li-kernels  0 li-kernels !
variable li-hosts    0 li-hosts !

\ ---- Current function name ---------------------------------------------------

create li-name-buf 64 allot
variable li-name-len  0 li-name-len !
: li-set-name  ( addr u -- )  dup li-name-len ! li-name-buf swap move ;
: li-name$  ( -- addr u )  li-name-buf li-name-len @ ;

\ ---- Symbol table: up to 64 variables ----
\ kind: 0=input-ptr 1=output-ptr 2=local-f32 3=each-var 4=local-u32
\       5=scalar-u32 6=scalar-f32 7=local-pred 8=shared-buf

64 constant MAX-SYMS
create sym-bufs  MAX-SYMS 32 * allot
create sym-lens  MAX-SYMS cells allot
create sym-kinds MAX-SYMS cells allot
create sym-regs  MAX-SYMS cells allot     \ rd# for ptrs, f# for locals, r# for each
variable n-syms     0 n-syms !
variable n-inputs   0 n-inputs !
variable n-outputs  0 n-outputs !

: sym-name@  ( i -- addr u )  dup 32 * sym-bufs + swap cells sym-lens + @ ;
: sym-kind@  ( i -- k )       cells sym-kinds + @ ;
: sym-reg@   ( i -- r )       cells sym-regs + @ ;

: sym-find  ( addr u -- i | -1 )
    n-syms @ 0 ?do
        2dup i sym-name@ li-tok= if 2drop i unloop exit then
    loop
    2drop -1 ;

: sym-add  ( addr u kind reg -- i )
    n-syms @ MAX-SYMS < 0= if 2drop 2drop -1 exit then
    n-syms @ >r
    r@ cells sym-regs + !
    r@ cells sym-kinds + !
    dup 32 > if drop 32 then
    dup r@ cells sym-lens + !
    r@ 32 * sym-bufs + swap move
    1 n-syms +!
    r> ;

: sym-reset  0 n-syms !  0 n-inputs !  0 n-outputs ! ;

\ ---- Register allocators (monotone) -----------------------------------------

variable next-freg   0 next-freg !
variable next-rreg   4 next-rreg !     \ r0-r3 reserved for tid
variable next-rdreg  4 next-rdreg !    \ rd0-rd3 reserved for params
variable next-preg   0 next-preg !

: freg+  ( -- n )  next-freg @  1 next-freg +! ;
: rreg+  ( -- n )  next-rreg @  1 next-rreg +! ;
: rdreg+ ( -- n )  next-rdreg @ 1 next-rdreg +! ;
: preg+  ( -- n )  next-preg @  1 next-preg +! ;

\ ---- Shared memory tracking --------------------------------------------------
\ Up to 8 shared memory declarations
8 constant MAX-SHARED
create shm-names MAX-SHARED 32 * allot
create shm-nlens MAX-SHARED cells allot
create shm-bytes MAX-SHARED cells allot
variable n-shared  0 n-shared !

: shm-name@ ( i -- addr u ) dup 32 * shm-names + swap cells shm-nlens + @ ;

\ ---- Scalar param tracking ---------------------------------------------------
\ Track scalar params separately for the header
16 constant MAX-SPARAMS
create sparam-names MAX-SPARAMS 32 * allot
create sparam-nlens MAX-SPARAMS cells allot
create sparam-types MAX-SPARAMS cells allot  \ 0=u32 1=f32
variable n-sparams  0 n-sparams !

: sparam-name@ ( i -- addr u ) dup 32 * sparam-names + swap cells sparam-nlens + @ ;
: sparam-type@ ( i -- t ) cells sparam-types + @ ;

\ ---- Label counter -----------------------------------------------------------
variable next-label  0 next-label !
: label+ ( -- n ) next-label @ 1 next-label +! ;

\ ---- For-loop stack (nesting up to 8 deep) -----------------------------------
8 constant MAX-FOR
create for-labels MAX-FOR cells allot    \ label number
create for-counters MAX-FOR cells allot  \ sym index of counter
variable for-depth  0 for-depth !

\ ---- Keyword string constants ------------------------------------------------

create k-fn       2 allot   s" fn"    k-fn    swap move
create k-arrow    2 allot   s" ->"    k-arrow swap move
create k-each     4 allot   s" each"  k-each  swap move
create k-lbrack   1 allot   s" ["     k-lbrack swap move
create k-rbrack   1 allot   s" ]"     k-rbrack swap move
create k-eq       1 allot   s" ="     k-eq     swap move
create k-plus     1 allot   s" +"     k-plus   swap move
create k-minus    1 allot   s" -"     k-minus  swap move
create k-star     1 allot   s" *"     k-star   swap move
create k-slash    1 allot   s" /"     k-slash  swap move

create k-for      3 allot   s" for"     k-for    swap move
create k-endfor   6 allot   s" endfor"  k-endfor swap move
create k-stride    6 allot   s" stride"    k-stride    swap move
create k-endstride 9 allot   s" endstride" k-endstride swap move
create k-shr      3 allot   s" shr"     k-shr    swap move
create k-shl      3 allot   s" shl"     k-shl    swap move
create k-and      3 allot   s" and"     k-and    swap move
create k-or       2 allot   s" or"      k-or     swap move
create k-xor      3 allot   s" xor"     k-xor    swap move
create k-shfl     9 allot   s" shfl.bfly" k-shfl swap move
create k-shared   6 allot   s" shared"  k-shared swap move
create k-param    5 allot   s" param"   k-param  swap move
create k-u32      3 allot   s" u32"     k-u32    swap move
create k-f32      3 allot   s" f32"     k-f32    swap move
create k-s32      3 allot   s" s32"     k-s32    swap move
create k-exp      3 allot   s" exp"     k-exp    swap move
create k-rcp      3 allot   s" rcp"     k-rcp    swap move
create k-rsqrt    5 allot   s" rsqrt"   k-rsqrt  swap move
create k-sqrt     4 allot   s" sqrt"    k-sqrt   swap move
create k-sin      3 allot   s" sin"     k-sin    swap move
create k-cos      3 allot   s" cos"     k-cos    swap move
create k-neg      3 allot   s" neg"     k-neg    swap move
create k-fma      3 allot   s" fma"     k-fma    swap move
create k-bar      7 allot   s" barrier" k-bar    swap move
create k-ld       2 allot   s" ld"      k-ld     swap move
create k-st       2 allot   s" st"      k-st     swap move
create k-mov      3 allot   s" mov"     k-mov    swap move
create k-setp     4 allot   s" setp"    k-setp   swap move
create k-ifge     4 allot   s" if>="    k-ifge   swap move
create k-iflt     4 allot   s" if<"     k-iflt   swap move
create k-exit     4 allot   s" exit"    k-exit   swap move
create k-bra      3 allot   s" bra"     k-bra    swap move
create k-label    5 allot   s" label"   k-label  swap move
create k-at       1 allot   s" @"       k-at     swap move
create k-ret      3 allot   s" ret"     k-ret    swap move
create k-add      3 allot   s" add"     k-add    swap move
create k-sub      3 allot   s" sub"     k-sub    swap move
create k-mul      3 allot   s" mul"     k-mul    swap move
create k-mad      3 allot   s" mad"     k-mad    swap move
create k-cvt      3 allot   s" cvt"     k-cvt    swap move
create k-ldg      8 allot   s" ld.global" k-ldg  swap move
create k-stg      8 allot   s" st.global" k-stg  swap move
create k-lds      9 allot   s" ld.shared" k-lds  swap move
create k-sts      9 allot   s" st.shared" k-sts  swap move

\ Type conversion keywords
create k-f32s32   7 allot   s" f32>s32" k-f32s32 swap move
create k-f32u32   7 allot   s" f32>u32" k-f32u32 swap move
create k-u32f32   7 allot   s" u32>f32" k-u32f32 swap move
create k-s32f32   7 allot   s" s32>f32" k-s32f32 swap move

\ ---- Param counting (used by cubin-wrap.fs) ----------------------------------
: count-ptr-params ( -- n )
    0 n-syms @ 0 ?do
        i sym-kind@ dup 0 = swap 1 = or if 1+ then
    loop ;

: count-all-params ( -- n )
    count-ptr-params n-sparams @ + ;

\ ---- SASS emitter stubs (overridden when emit-sass.fs is loaded) -------------
\ These let the parser compile even without the SASS backend.
: s2r,       ( rd sr -- )     2drop ;
: imad,      ( rd r1 r2 r3 -- ) 2drop 2drop ;
: imad-imm,  ( rd r1 imm -- ) drop 2drop ;
: fadd,      ( rd ra rb -- )  drop 2drop ;
: mov-imm,   ( rd imm -- )   2drop ;
: isetp-ge,  ( pd r1 r2 -- ) drop 2drop ;
: isetp-lt,  ( pd r1 r2 -- ) drop 2drop ;
: bra-pred,  ( off pred -- ) 2drop ;
: ldg-off,   ( rd ra off -- ) drop 2drop ;
: stg-off,   ( ra rs off -- ) drop 2drop ;
: exit,      ( -- ) ;
: nop,       ( -- ) ;
variable sass-pos  0 sass-pos !

\ ---- IEEE 754 float encoding -------------------------------------------------
\ Convert a float string like "1.0" or "0.5" to IEEE 754 hex "0f3F800000"
\ We use a lookup table for common constants to avoid needing FP in Forth.

\ Known float constants table: string -> IEEE754 hex
\ We store these as pairs. The parser checks the table.
32 constant MAX-FCONST
create fconst-strs  MAX-FCONST 16 * allot
create fconst-lens  MAX-FCONST cells allot
create fconst-vals  MAX-FCONST cells allot
variable n-fconst  0 n-fconst !

: fconst-add ( addr u hex-val -- )
    n-fconst @ MAX-FCONST < 0= if drop 2drop exit then
    n-fconst @ >r
    r@ cells fconst-vals + !
    dup r@ cells fconst-lens + !
    r@ 16 * fconst-strs + swap move
    1 n-fconst +!
    r> drop ;

: fconst-find ( addr u -- hex-val -1 | 0 )
    n-fconst @ 0 ?do
        i cells fconst-lens + @ over = if
            2dup i 16 * fconst-strs + over li-tok= if
                2drop i cells fconst-vals + @ -1 unloop exit
            then
        then
    loop
    2drop 0 ;

\ Register common float constants
: init-fconsts
    s" 0.0"     1078530048 fconst-add   \ placeholder, will use 0x00000000
    s" 1.0"     1065353216 fconst-add   \ 0x3F800000
    s" -1.0"    3212836864 fconst-add   \ 0xBF800000
    s" 0.5"     1056964608 fconst-add   \ 0x3F000000
    s" -0.5"    3204448256 fconst-add   \ 0xBF000000
    s" 2.0"     1073741824 fconst-add   \ 0x40000000
    s" -2.0"    3221225472 fconst-add   \ 0xC0000000
    s" 0.25"    1048576000 fconst-add   \ 0x3E800000
    s" 3.0"     1077936128 fconst-add   \ 0x40400000
    s" 4.0"     1082130432 fconst-add   \ 0x40800000
    s" 8.0"     1090519040 fconst-add   \ 0x41000000
    s" -8.0"    3238002688 fconst-add   \ 0xC1000000
    s" 16.0"    1098907648 fconst-add   \ 0x41800000
    s" 0.125"   1040187392 fconst-add   \ 0x3E000000
    s" 10.0"    1092616192 fconst-add   \ 0x41200000
    s" 0.693147" 1060205080 fconst-add  \ 0x3F317218 (ln2)
    s" 1.442695" 1069066811 fconst-add  \ 0x3FB8AA3B (1/ln2 = log2(e))
    s" 0.00001" 931135283 fconst-add    \ ~1e-5: 0x3727C5AC
    s" 1e-5"    931135283 fconst-add    \ same
    s" 1e-6"    897988541 fconst-add    \ 0x358637BD
    s" 0.000001" 897988541 fconst-add
;
\ Fix 0.0 entry: it should be 0x00000000
: fix-fconst-zero  0 0 cells fconst-vals + ! ;

init-fconsts
fix-fconst-zero

\ ---- Emit "each" (thread indexing) -------------------------------------------
\ %r0=ctaid.x  %r1=ntid.x  %r2=tid.x  %r3=global_tid

: emit-each-tid  ( -- )
    0 $25 s2r,
    1 $29 s2r,
    2 $21 s2r,
    3 0 1 2 imad, ;

\ ---- Emit indexed load/store with variable index register --------------------
\ idx-reg holds the r32 register number to use as the array index.
\ Default is 3 (%r3 = each-var global tid), but for-loop counters override it.

variable idx-reg   3 idx-reg !

: resolve-idx ( addr u -- rreg )
    \ Look up index variable name and return its r32 register number.
    \ If it's the each-var (kind=3), return 3 (%r3).
    \ If it's a local-u32 (kind=4, e.g. for counter), return its reg.
    \ If it's a scalar-u32 (kind=5), return its reg.
    sym-find dup -1 = if drop 3 exit then
    dup sym-kind@ 3 = if drop 3 exit then
    sym-reg@ ;

\ ---- resolve-idx-expr: parse computed index expression inside [ ] -------------
\ Handles three forms:
\   simple:      var              -> resolve-idx -> rreg
\   mul-add:     var * iconst + var  -> mad.lo.u32 %rN, rA, iconst, rB  -> rreg
\   add-only:    var + var           -> add.u32 %rN, rA, rB             -> rreg
\ Consumes tokens up to but not including the closing "]".
\ After return the caller must consume "]".

variable rie-pos0
variable rie-rA
variable rie-rB
variable rie-rC
variable rie-imm
variable rie-result

: resolve-sym-rreg ( addr u -- rreg )
    \ resolve a single token that must be a u32/each symbol; return r-reg number
    sym-find dup -1 = if drop 3 exit then
    dup sym-kind@ 3 = if drop 3 exit then
    sym-reg@ ;

: resolve-idx-expr ( -- rreg )
    src-pos @ rie-pos0 !
    src-token resolve-sym-rreg rie-rA !
    src-pos @ >r  src-token  r> swap >r src-pos !  r>
    2dup k-star 1 li-tok= if
        2drop
        src-token 2dup is-number? if parse-uint rie-imm ! else 2drop 1 rie-imm ! then
        src-token 2drop
        src-token resolve-sym-rreg rie-rB !
        rreg+ rie-result !
        rie-result @ rie-rA @ rie-imm @ rie-rB @ imad,
        rie-result @
    else 2dup k-plus 1 li-tok= if
        2drop
        src-token resolve-sym-rreg rie-rB !
        rreg+ rie-result !
        rie-result @ rie-rA @ 1 rie-rB @ imad,
        rie-result @
    else
        2drop rie-rA @
    then then ;

: emit-iload-r  ( param-rd idx-rreg -- freg )
    freg+ -rot
    rreg+ >r  r@ swap 4 0 imad,
    rot dup >r swap r> r> ldg-off, ;

: emit-iload  ( param-rd -- freg )
    3 emit-iload-r ;

: emit-istore-r  ( param-rd idx-rreg freg -- )
    >r  rreg+ >r  r@ swap 4 0 imad,
    swap r> r> stg-off, ;

: emit-istore  ( param-rd freg -- )
    swap 3 swap emit-istore-r ;

\ ---- Helper: check if token looks like a number (starts with digit or '-' followed by digit)
: is-digit? ( c -- flag ) dup [char] 0 < 0= swap [char] 9 > 0= and ;
: is-number? ( addr u -- flag )
    dup 0= if 2drop 0 exit then
    over c@                         \ ( addr u c )
    dup [char] 0 < 0= over [char] 9 > 0= and if
        drop 2drop -1 exit
    then
    [char] - = if
        dup 1 > if
            over 1+ c@ dup [char] 0 < 0= swap [char] 9 > 0= and
            if 2drop -1 exit then
        then
        2drop 0 exit
    then
    2drop 0 ;

\ Check if token looks like a float (contains '.')
: has-dot ( addr u -- flag )
    dup 0= if 2drop 0 exit then
    0 ?do
        dup i + c@ [char] . = if drop -1 unloop exit then
    loop
    drop 0 ;
: is-float? ( addr u -- flag )
    2dup has-dot 0= if 2drop 0 exit then
    \ Also check first char is digit or '-' (not just any word with a dot)
    over c@ dup [char] 0 < 0= over [char] 9 > 0= and if drop 2drop -1 exit then
    [char] - = if 2drop -1 exit then
    2drop 0 ;

\ ---- Parse integer from string -----------------------------------------------
variable pi-neg
: parse-int ( addr u -- n )
    0 pi-neg !
    over c@ [char] - = if 1- swap 1+ swap -1 pi-neg ! then
    parse-uint
    pi-neg @ if negate then ;

\ Simpler version: unsigned parse
variable pu-acc
: parse-uint ( addr u -- n )
    0 pu-acc !
    0 ?do
        dup i + c@ [char] 0 -
        pu-acc @ 10 * + pu-acc !
    loop
    drop pu-acc @ ;

\ Parse hex number (after 0x prefix)
: hex-val ( c -- n )
    dup [char] 0 < 0= over [char] 9 > 0= and if [char] 0 - exit then
    dup [char] a < 0= over [char] f > 0= and if [char] a - 10 + exit then
    dup [char] A < 0= over [char] F > 0= and if [char] A - 10 + exit then
    drop 0 ;

variable ph-acc
: parse-hex ( addr u -- n )
    0 ph-acc !
    0 ?do
        dup i + c@ hex-val
        ph-acc @ 16 * + ph-acc !
    loop
    drop ph-acc @ ;

\ Check if token starts with 0x
: is-hex? ( addr u -- flag )
    dup 2 < if 2drop 0 exit then
    drop dup c@ [char] 0 = swap 1 + c@ [char] x = and ;

\ ---- Resolve operand: look up symbol, return kind and reg --------------------
\ For use in direct PTX-like statements

\ ---- Expression parser (returns freg holding result) -------------------------
variable 'parse-expr
variable 'parse-fn-body

: peek-tok  ( -- addr u )
    src-pos @ >r  src-token  r> swap >r src-pos !  r> ;

\ Emit a float constant, returns freg
variable fc-freg
variable fc-hex
: emit-fconst ( addr u -- freg )
    2dup fconst-find if
        fc-hex ! 2drop freg+ fc-freg !
        fc-freg @ fc-hex @ mov-imm,
        fc-freg @
    else
        2drop freg+ fc-freg !
        fc-freg @ 0 mov-imm,
        fc-freg @
    then ;
            drop

\ Emit an integer constant into an r32 register
variable ic-rreg
: emit-iconst ( n -- rreg )
    rreg+ ic-rreg !
    ic-rreg @ swap mov-imm,
    ic-rreg @ ;

\ ---- Inline function call ----------------------------------------------------
\ When parse-atom encounters an unknown identifier that is a registered
\ function name, we inline its body into the caller.
\
\ Inline mode: during inline replay, the callee's `each` is skipped (the
\ caller already has thread indexing), and the callee's output store
\ y[i] = expr  becomes a register capture: the expr result freg is saved
\ to inline-result-freg, and the store is suppressed.
\
\ inline-call ( fn-idx -- freg )

variable inline-mode       0 inline-mode !
variable inline-result-freg  -1 inline-result-freg !

variable ic-fn-idx
variable ic-saved-src-pos
variable ic-saved-nsyms
8 constant MAX-INLINE-ARGS
create ic-arg-regs MAX-INLINE-ARGS cells allot
create ic-arg-kinds MAX-INLINE-ARGS cells allot

: inline-call  ( fn-idx -- freg )
    ic-fn-idx !
    inline-depth @ MAX-INLINE-DEPTH < 0= if
        freg+ dup >r
        r@ 0 mov-imm,
        r> exit
    then
    1 inline-depth +!

    \ Parse caller arguments: one token per callee param
    ic-fn-idx @ cells fndef-nparams + @ 0 ?do
        src-token dup 0= if 2drop 0 i cells ic-arg-regs + ! 0 i cells ic-arg-kinds + ! else
            sym-find dup -1 = if
                drop 0 i cells ic-arg-regs + ! 0 i cells ic-arg-kinds + !
            else
                dup sym-reg@ i cells ic-arg-regs + !
                sym-kind@ i cells ic-arg-kinds + !
            then
        then
    loop

    \ Save parser state
    src-pos @ ic-saved-src-pos !
    n-syms @ ic-saved-nsyms !

    \ Add callee's input params to sym table mapped to caller's arg registers
    ic-fn-idx @ cells fndef-nparams + @ 0 ?do
        ic-fn-idx @ i fndef-param-name@
        i cells ic-arg-kinds + @
        i cells ic-arg-regs + @
        sym-add drop
    loop

    \ Add callee's outputs as local-f32 (kind=2) with fresh fregs.
    \ When the callee does y[i] = expr, inline mode intercepts the store:
    \ instead of emitting st.global, it captures the expr freg.
    ic-fn-idx @ cells fndef-nouts + @ 0 ?do
        ic-fn-idx @ i fndef-out-name@
        2                          \ kind = local-f32
        freg+                      \ fresh freg for output
        sym-add drop
    loop

    \ Enter inline mode, jump to callee body
    -1 inline-result-freg !
    inline-mode @ >r
    -1 inline-mode !
    ic-fn-idx @ cells fndef-src-start + @ src-pos !

    \ Parse the callee's body
    'parse-fn-body @ execute

    \ Restore parser state
    r> inline-mode !
    ic-saved-src-pos @ src-pos !
    ic-saved-nsyms @ n-syms !

    -1 inline-depth +!

    \ Return the captured result freg
    inline-result-freg @ ;

variable pa-saved-pos
: parse-atom  ( -- freg )
    src-token dup 0= if 2drop -1 exit then

    \ Check for float constant (contains '.')
    2dup is-float? if
        emit-fconst exit
    then

    \ Check for hex literal 0x...
    2dup is-hex? if
        2 - swap 2 + swap parse-hex emit-iconst exit
    then

    \ Check for plain integer
    2dup is-number? if
        parse-uint emit-iconst exit
    then

    \ Look up identifier
    2dup sym-find dup -1 = if
        drop
        \ Check if it's a registered function name (inline call)
        2dup fndef-find dup -1 <> if
            nip nip                 \ drop token addr/u, keep fn-idx
            inline-call exit
        then
        drop
        \ Unknown — emit zero
        2drop
        freg+ dup pa-saved-pos !
        pa-saved-pos @ 0 mov-imm,
        pa-saved-pos @ exit
    then
    nip nip               \ ( sym-idx )
    \ Peek next token for "["
    src-pos @ pa-saved-pos !
    src-token dup 0= if
        2drop pa-saved-pos @ src-pos !
        dup sym-kind@ 2 = if  sym-reg@  else
        dup sym-kind@ 3 = if  drop -1  else
        dup sym-kind@ 4 = if  sym-reg@  else
        dup sym-kind@ 5 = if  sym-reg@  else
        dup sym-kind@ 6 = if  sym-reg@  else
        sym-reg@
        then then then then then exit
    then
    2dup k-lbrack 1 li-tok= if
        2drop
        \ If this is a local-f32 (kind=2), e.g. from inline result,
        \ just consume the index tokens and return the freg directly.
        dup sym-kind@ 2 = if
            resolve-idx-expr drop  \ parse and discard index rreg
            src-token 2drop        \ consume "]"
            sym-reg@ exit
        then
        resolve-idx-expr         \ parse simple var or A*C+B expr -> rreg
        src-token 2drop          \ consume "]"
        swap sym-reg@ swap       \ ( param-rd idx-rreg )
        emit-iload-r             \ emit load with correct index register
        exit
    then
    \ Not "[" — put token back
    2drop pa-saved-pos @ src-pos !
    dup sym-kind@ 2 = if  sym-reg@  else
    dup sym-kind@ 3 = if  drop -1  else
    dup sym-kind@ 4 = if  sym-reg@  else
    dup sym-kind@ 5 = if  sym-reg@  else
    dup sym-kind@ 6 = if  sym-reg@  else
    sym-reg@
    then then then then then ;

variable pt-saved-pos
variable pt-result-freg

: parse-term  ( -- freg )
    parse-atom
    begin
        src-pos @ pt-saved-pos !
        src-token dup 0= if 2drop pt-saved-pos @ src-pos ! exit then
        2dup k-star 1 li-tok= if
            2drop
            parse-atom swap
            freg+ pt-result-freg !
            pt-result-freg @ rot rot fmul,
            pt-result-freg @
        else 2dup k-slash 1 li-tok= if
            2drop
            parse-atom swap
            freg+ pt-result-freg !
            pt-result-freg @ rot rot fmul,  \ div placeholder
            pt-result-freg @
        else
            2drop pt-saved-pos @ src-pos ! exit
        then then
    again ;

variable pe-saved-pos
variable pe-result-freg

: parse-expr  ( -- freg )
    parse-term
    begin
        src-pos @ pe-saved-pos !
        src-token dup 0= if 2drop pe-saved-pos @ src-pos ! exit then
        2dup k-plus 1 li-tok= if
            2drop
            parse-term swap
            freg+ pe-result-freg !
            pe-result-freg @ swap over fadd,
            pe-result-freg @
        else 2dup k-minus 1 li-tok= if
            2drop
            parse-term swap
            freg+ pe-result-freg !
            pe-result-freg @ rot rot fadd,  \ sub placeholder
            pe-result-freg @
        else
            2drop pe-saved-pos @ src-pos ! exit
        then then
    again ;

' parse-expr 'parse-expr !

\ ==============================================================================
\ FEATURE 1: FOR LOOP
\ Syntax: for COUNTER START BOUND STEP ... endfor
\ Emits: mov counter, start; LOOP_label: setp.ge counter, bound; @p bra END; ... add counter, step; bra LOOP; END:
\ ==============================================================================

\ Scratch storage for for-loop args
variable for-counter-reg
variable for-start-val
variable for-start-is-reg   \ -1 if start is a register, 0 if literal
variable for-bound-val
variable for-bound-is-reg
variable for-step-val
variable for-label-num

: emit-for-v2 ( -- )
    \ Parse counter name
    src-token
    2dup sym-find dup -1 = if
        drop rreg+ dup for-counter-reg !
        4 swap sym-add drop
    else
        nip nip sym-reg@ for-counter-reg !
    then

    \ Parse start
    src-token 2dup is-number? if
        parse-uint for-start-val ! 0 for-start-is-reg !
    else
        2dup sym-find dup -1 <> if
            nip nip sym-reg@ for-start-val ! -1 for-start-is-reg !
        else
            drop 2drop 0 for-start-val ! 0 for-start-is-reg !
        then
    then

    \ Parse bound
    src-token 2dup sym-find dup -1 <> if
        nip nip sym-reg@ for-bound-val ! -1 for-bound-is-reg !
    else
        drop   \ drop the -1 from sym-find
        2dup is-number? if
            parse-uint for-bound-val ! 0 for-bound-is-reg !
        else
            2drop 0 for-bound-val ! 0 for-bound-is-reg !
        then
    then

    \ Parse step
    src-token 2dup is-number? if
        parse-uint for-step-val !
    else
        sym-find dup -1 <> if sym-reg@ else drop 1 then for-step-val !
    then

    \ Allocate label
    label+ for-label-num !

    \ Push to for-stack
    for-depth @ MAX-FOR < if
        for-label-num @ for-depth @ cells for-labels + !
        for-counter-reg @ for-depth @ cells for-counters + !
        1 for-depth +!
    then

    \ SASS: mov-imm counter, start;  record loop-top sass-pos;
    \       isetp-ge pred, counter, bound;  bra-pred end_offset pred
    for-start-is-reg @ if
        \ counter = start-reg: use imad-imm with imm=0 to copy
        for-counter-reg @ for-start-val @ 0 imad-imm,
    else
        for-counter-reg @ for-start-val @ mov-imm,
    then
    \ Record top-of-loop position (in sass-pos units = bytes)
    sass-pos @ for-depth @ 1- cells for-labels + !   \ reuse label slot for sass-pos
    preg+ >r
    \ Bound: SASS isetp-ge needs a register; use imad-imm to materialize if literal
    for-bound-is-reg @ 0= if
        rreg+ dup for-bound-val @ mov-imm,
        r@ for-counter-reg @ swap isetp-ge,
    else
        r@ for-counter-reg @ for-bound-val @ isetp-ge,
    then
    \ Emit placeholder bra-pred (offset 0 for now; endfor will patch)
    0 r> bra-pred, ;

: emit-endfor ( -- )
    for-depth @ 0= if exit then
    -1 for-depth +!
    for-depth @ cells for-labels + @ >r
    for-depth @ cells for-counters + @ >r
    r@ r@ for-step-val @ 0 imad,
    r> drop r> drop ;

\ ==============================================================================
\ FEATURE: STRIDE LOOP
\ Syntax: stride I BOUND ... endstride
\ Semantics: i = tid.x; while i < bound: body; i += blockDim.x
\ ==============================================================================

variable stride-counter-reg
variable stride-bound-val
variable stride-bound-is-reg
variable stride-label-num

: emit-stride ( -- )
    \ Parse loop variable name
    src-token
    2dup sym-find dup -1 = if
        drop rreg+ dup stride-counter-reg !
        4 swap sym-add drop
    else
        nip nip sym-reg@ stride-counter-reg !
    then

    \ Parse bound (register or literal)
    src-token 2dup sym-find dup -1 <> if
        nip nip sym-reg@ stride-bound-val ! -1 stride-bound-is-reg !
    else
        drop
        2dup is-number? if
            parse-uint stride-bound-val ! 0 stride-bound-is-reg !
        else
            2drop 0 stride-bound-val ! 0 stride-bound-is-reg !
        then
    then

    \ Allocate label
    label+ stride-label-num !

    \ Push to for-stack (reuse for-stack for nesting)
    for-depth @ MAX-FOR < if
        stride-label-num @ for-depth @ cells for-labels + !
        stride-counter-reg @ for-depth @ cells for-counters + !
        1 for-depth +!
    then

    stride-counter-reg @ 2 0 imad-imm,
    sass-pos @ for-depth @ 1- cells for-labels + !
    preg+ >r
    stride-bound-is-reg @ 0= if
        rreg+ dup stride-bound-val @ mov-imm,
        r@ stride-counter-reg @ swap isetp-ge,
    else
        r@ stride-counter-reg @ stride-bound-val @ isetp-ge,
    then
    0 r> bra-pred, ;

: emit-endstride ( -- )
    for-depth @ 0= if exit then
    -1 for-depth +!
    for-depth @ cells for-labels + @ >r
    for-depth @ cells for-counters + @ >r
    r@ r@ 1 0 imad,
    r> drop r> drop ;

\ ==============================================================================
\ FEATURE 2: INTEGER/BITWISE OPS
\ Syntax: shr DST SRC AMT  (also and, or, xor, shl)
\ ==============================================================================

variable bw-dst
variable bw-src1
variable bw-src2
variable bw-src2-imm     \ -1 if src2 is immediate
create bw-op-buf 16 allot
variable bw-op-len

: parse-bw-args ( -- )
    \ Parse dst
    src-token 2dup sym-find dup -1 = if
        drop rreg+ dup bw-dst !
        >r 4 r> sym-add drop
    else nip nip sym-reg@ bw-dst ! then

    \ Parse src1
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst bw-src1 !
        else 2drop 0 bw-src1 ! then
    else nip nip sym-reg@ bw-src1 ! then

    \ Parse src2
    0 bw-src2-imm !
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint bw-src2 ! -1 bw-src2-imm !
        else 2dup is-hex? if
            2 - swap 2 + swap parse-hex bw-src2 ! -1 bw-src2-imm !
        else 2drop 0 bw-src2 ! then then
    else nip nip sym-reg@ bw-src2 ! then ;

: emit-bw-instr ( op-addr op-u -- )
    dup bw-op-len ! bw-op-buf swap move
    parse-bw-args
    ;

: emit-shr  s" shr" emit-bw-instr ;
: emit-shl  s" shl" emit-bw-instr ;
: emit-and  s" and" emit-bw-instr ;
: emit-or   s" or"  emit-bw-instr ;
: emit-xor  s" xor" emit-bw-instr ;

\ ==============================================================================
\ FEATURE 3: WARP SHUFFLE
\ Syntax: shfl.bfly DST SRC OFFSET
\ Emits: shfl.sync.bfly.b32 DST, SRC, OFFSET, 0x1f; (mask 0xffffffff)
\ ==============================================================================

variable shfl-dst
variable shfl-src
variable shfl-off

variable shfl-dst-is-float
variable shfl-src-is-float
variable shfl-dst-freg
variable shfl-src-freg

: emit-shfl-bfly ( -- )
    0 shfl-dst-is-float !
    0 shfl-src-is-float !

    \ Parse DST
    src-token 2dup sym-find dup -1 = if
        drop
        \ New variable: create as float, use temp r32 for shuffle
        freg+ shfl-dst-freg !
        2 shfl-dst-freg @ sym-add drop
        rreg+ shfl-dst !
        1 shfl-dst-is-float !
    else
        nip nip
        dup sym-kind@ 2 = if
            \ Float variable: allocate temp r32 for shuffle, remember freg
            sym-reg@ shfl-dst-freg !
            rreg+ shfl-dst !
            1 shfl-dst-is-float !
        else sym-reg@ shfl-dst ! then
    then

    \ Parse SRC
    src-token 2dup sym-find dup -1 = if
        drop 2drop 0 shfl-src !
    else
        nip nip
        dup sym-kind@ 2 = if
            \ Float variable: move to temp r32 before shuffle
            sym-reg@ shfl-src-freg !
            rreg+ dup shfl-src !
            1 shfl-src-is-float !
            \ float src: handled by SASS shfl
            \ float src: no PTX emission needed
        else sym-reg@ shfl-src ! then
    then

    \ Parse OFFSET
    src-token 2dup is-number? if
        parse-uint shfl-off !
    else
        sym-find dup -1 <> if sym-reg@ shfl-off ! else drop 0 shfl-off ! then
    then

    \ SASS: shfl-bfly
    shfl-dst @ shfl-src @ shfl-off @ shfl-bfly, ;

\ ==============================================================================
\ FEATURE 4: SHARED MEMORY
\ Syntax: shared NAME BYTES TYPE
\ Emits declaration in header, registers name as shared buffer
\ ==============================================================================

: parse-shared-v2 ( -- )
    src-token                         \ ( name-addr name-u )
    \ Save name into shared table
    n-shared @ >r
    2dup dup r@ cells shm-nlens + !
    r@ 32 * shm-names + swap move

    \ Parse element count
    src-token 2dup is-number? if parse-uint else 2drop 0 then

    \ Parse type -> compute bytes
    src-token 2dup k-f32 3 li-tok= if
        2drop 4 *
    else 2dup k-u32 3 li-tok= if
        2drop 4 *
    else 2dup k-s32 3 li-tok= if
        2drop 4 *
    else
        2drop    \ assume bytes already
    then then then

    r@ cells shm-bytes + !

    \ Register symbol: kind=8, reg=shared-index
    8 r> sym-add drop
    1 n-shared +! ;

\ ==============================================================================
\ FEATURE 5: SCALAR PARAMS
\ Syntax: param NAME TYPE (u32 or f32)
\ ==============================================================================

: parse-scalar-param ( -- )
    src-token                         \ name
    2dup                              \ save copy

    \ Save to sparam table
    n-sparams @ >r
    2dup dup r@ cells sparam-nlens + !
    r@ 32 * sparam-names + swap move

    \ Parse type
    src-token 2dup k-u32 3 li-tok= if
        2drop 0 r@ cells sparam-types + !
        \ Register as symbol kind=5, allocate r32
        rreg+ >r 5 r> sym-add drop
    else 2dup k-f32 3 li-tok= if
        2drop 1 r@ cells sparam-types + !
        \ Register as symbol kind=6, allocate freg
        freg+ >r 6 r> sym-add drop
    else
        2drop 0 r@ cells sparam-types + !
        rreg+ >r 5 r> sym-add drop
    then then

    r> drop
    1 n-sparams +! ;

\ ==============================================================================
\ FEATURE 6: MATH INTRINSICS
\ Syntax: exp DST SRC  (also rcp, rsqrt, sqrt, sin, cos, neg)
\ ==============================================================================

create mu-op-buf 32 allot
variable mu-op-len
variable mu-dst
variable mu-src

: emit-math-unary ( op-addr op-u -- )
    dup mu-op-len ! mu-op-buf swap move
    src-token 2dup sym-find dup -1 = if
        drop freg+ dup mu-dst !
        >r 2 r> sym-add drop
    else nip nip sym-reg@ mu-dst ! then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if
            emit-fconst mu-src !
        else 2drop 0 mu-src ! then
    else nip nip sym-reg@ mu-src ! then ;

: emit-exp    s" ex2.approx.f32"   emit-math-unary ;
: emit-rcp    s" rcp.approx.f32"   emit-math-unary ;
: emit-rsqrt  s" rsqrt.approx.f32" emit-math-unary ;
: emit-sqrt   s" sqrt.approx.f32"  emit-math-unary ;
: emit-sin    s" sin.approx.f32"   emit-math-unary ;
: emit-cos    s" cos.approx.f32"   emit-math-unary ;
: emit-neg    s" neg.f32"          emit-math-unary ;

\ FMA: fma DST A B C -> fma.rn.f32 DST, A, B, C
variable fma-d
variable fma-a
variable fma-b
variable fma-c

: parse-fma-arg ( -- freg )
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst nip nip
        else 2drop 0 then
    else nip nip sym-reg@ then ;

: emit-fma ( -- )
    src-token 2dup sym-find dup -1 = if
        drop freg+ dup fma-d !
        >r 2 r> sym-add drop
    else nip nip sym-reg@ fma-d ! then
    parse-fma-arg fma-a !
    parse-fma-arg fma-b !
    parse-fma-arg fma-c !
    fma-d @ fma-a @ fma-b @ fma-c @ ffma, ;

\ ==============================================================================
\ FEATURE 8: TYPE CONVERSIONS
\ Syntax: f32>s32 DST SRC   (also u32>f32, s32>f32, f32>u32)
\ ==============================================================================

\ Type conversion using variables
variable cvt-dst-reg
variable cvt-src-reg
variable cvt-dst-float   \ -1=float, 0=int
variable cvt-src-float
create cvt-suffix-buf 32 allot
variable cvt-suffix-len

: emit-cvt-v2 ( suffix-addr suffix-u dst-float src-float -- )
    cvt-src-float ! cvt-dst-float !
    dup cvt-suffix-len ! cvt-suffix-buf swap move
    src-token 2dup sym-find dup -1 = if
        drop cvt-dst-float @ if
            freg+ dup cvt-dst-reg ! >r 2 r> sym-add drop
        else rreg+ dup cvt-dst-reg ! >r 4 r> sym-add drop then
    else nip nip sym-reg@ cvt-dst-reg ! then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst cvt-src-reg !
        else 2dup is-number? if parse-uint emit-iconst cvt-src-reg !
        else 2drop 0 cvt-src-reg ! then then
    else nip nip sym-reg@ cvt-src-reg ! then ;

: emit-f32-to-s32  s" rzi.s32.f32" 0 -1 emit-cvt-v2 ;
: emit-f32-to-u32  s" rzi.u32.f32" 0 -1 emit-cvt-v2 ;
: emit-u32-to-f32  s" rn.f32.u32"  -1 0 emit-cvt-v2 ;
: emit-s32-to-f32  s" rn.f32.s32"  -1 0 emit-cvt-v2 ;

\ ==============================================================================
\ FEATURE 9: PREDICATION
\ Syntax: @p0 INSTRUCTION...  or  setp.ge DST A B
\ ==============================================================================

create setp-cmp-buf 8 allot
variable setp-cmp-len
create setp-type-buf 8 allot
variable setp-type-len
variable setp-pred
variable setp-src1
variable setp-src2

: emit-setp ( -- )
    src-token dup setp-cmp-len ! setp-cmp-buf swap move
    src-token dup setp-type-len ! setp-type-buf swap move
    src-token 2dup sym-find dup -1 = if
        drop preg+ dup setp-pred ! >r 7 r> sym-add drop
    else nip nip sym-reg@ setp-pred ! then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst setp-src1 !
        else 2drop 0 setp-src1 ! then
    else nip nip sym-reg@ setp-src1 ! then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst setp-src2 !
        else 2drop 0 setp-src2 ! then
    else nip nip sym-reg@ setp-src2 ! then ;

\ ==============================================================================
\ FEATURE 10: BOUNDS CHECK / EARLY EXIT
\ Syntax: if>= A B exit  ->  setp.ge.u32 %pN, A, B; @pN bra $L_exit
\ ==============================================================================

variable ifge-src1
variable ifge-src2
variable ifge-pred
create ifge-target 32 allot
variable ifge-target-len

: parse-ifge-args ( -- )
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst ifge-src1 !
        else 2drop 0 ifge-src1 ! then
    else nip nip sym-reg@ ifge-src1 ! then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst ifge-src2 !
        else 2drop 0 ifge-src2 ! then
    else nip nip sym-reg@ ifge-src2 ! then

    src-token dup ifge-target-len ! ifge-target swap move ;

: emit-ifge-exit ( -- )
    parse-ifge-args
    preg+ ifge-pred !
    ifge-pred @ ifge-src1 @ ifge-src2 @ isetp-ge,
    0 ifge-pred @ bra-pred, ;   \ offset 0 placeholder; target resolves at link time

: emit-iflt-exit ( -- )
    parse-ifge-args
    preg+ ifge-pred !
    ifge-pred @ ifge-src1 @ ifge-src2 @ isetp-lt,
    0 ifge-pred @ bra-pred, ;

\ ==============================================================================
\ EXTENDED STATEMENT: Direct PTX-like instructions
\ ==============================================================================

\ Emit label: label NAME -> $L_NAME:
: emit-label ( -- )  src-token 2drop ;

\ Emit branch: bra LABEL -> bra $L_LABEL
: emit-bra ( -- )  src-token 2drop ;

\ Emit barrier
: emit-barrier ( -- )  bar-sync, ;

\ Emit mov: mov DST SRC (u32 move between registers)
: emit-mov ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint swap mov-imm, exit
        else 2drop 0 then
    else nip nip sym-reg@ then
    swap >r >r  r> r@ 0 imad-imm, r> drop ;

\ Emit add.u32: add DST SRC1 SRC2
: emit-add-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    >r >r >r  r> r> 1 r> imad, ;

\ Emit sub.u32
: emit-sub-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    drop 2drop ;

\ mul.lo.u32
: emit-mul-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    drop 2drop ;

\ mad.lo.u32
: emit-mad ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else nip nip sym-reg@ then
    >r >r >r >r  r> r> r> r> imad, ;

\ ---- ld.global / st.global with computed address -----------------------------
\ ld.global.TYPE DST BASE OFFSET
\ Syntax: ld.global DST BASE OFFSET TYPE

variable ld-scratch-rd

\ Global load: ld.global DST BASE OFFSET_REG
\ Uses %rd30/%rd31 as scratch
variable ldst-dst
variable ldst-base
variable ldst-offset
variable ldst-type    \ 0=f32 1=u32

: emit-ld-global-v2 ( -- )
    \ Parse dst
    src-token 2dup sym-find dup -1 = if
        drop
        \ Check next token to determine type
        freg+ dup ldst-dst !
        >r 2 r> sym-add drop
    else nip nip sym-reg@ ldst-dst ! then

    \ Parse base (pointer / rd reg)
    src-token 2dup sym-find dup -1 = if
        drop 2drop 0 ldst-base !
    else nip nip sym-reg@ ldst-base ! then

    \ Parse offset (r32 reg)
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst ldst-offset !
        else 2drop 0 ldst-offset ! then
    else nip nip sym-reg@ ldst-offset ! then

    \ Optional type
    src-pos @ >r
    src-token 2dup k-u32 3 li-tok= if
        2drop r> drop 1 ldst-type !
    else 2dup k-f32 3 li-tok= if
        2drop r> drop 0 ldst-type !
    else
        2drop r> src-pos ! 0 ldst-type !
    then then

    \ SASS: ldg-off rd, ra, byte_offset (offset=idx*4)
    \ We use imad to compute byte offset into a temp reg, then ldg-off
    rreg+ >r   \ temp reg for byte offset
    r@ ldst-offset @ 4 0 imad,   \ temp = idx * 4 + 0
    ldst-dst @ ldst-base @ r> ldg-off, ;

: emit-st-global-v2 ( -- )
    \ st.global BASE OFFSET SRC
    \ Parse base
    src-token 2dup sym-find dup -1 = if
        drop 2drop 0 ldst-base !
    else nip nip sym-reg@ ldst-base ! then

    \ Parse offset (r32 reg)
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst ldst-offset !
        else 2drop 0 ldst-offset ! then
    else nip nip sym-reg@ ldst-offset ! then

    \ Parse src
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if
            emit-fconst ldst-dst !
        else 2dup is-number? if
            parse-uint emit-iconst ldst-dst !
        else 2drop 0 ldst-dst ! then then
    else nip nip sym-reg@ ldst-dst ! then

    \ Optional type
    src-pos @ >r
    src-token 2dup k-u32 3 li-tok= if
        2drop r> drop 1 ldst-type !
    else 2dup k-f32 3 li-tok= if
        2drop r> drop 0 ldst-type !
    else
        2drop r> src-pos ! 0 ldst-type !
    then then

    \ SASS: stg-off ra, rs, byte_offset
    rreg+ >r   \ temp reg for byte offset
    r@ ldst-offset @ 4 0 imad,   \ temp = idx * 4 + 0
    ldst-base @ ldst-dst @ r> stg-off, ;

\ ---- Shared memory load/store ------------------------------------------------
\ ld.shared DST SHMEM_NAME OFFSET
\ st.shared SHMEM_NAME OFFSET SRC

: emit-ld-shared ( -- )
    src-token 2dup sym-find dup -1 = if
        drop freg+ dup ldst-dst ! >r 2 r> sym-add drop
    else nip nip sym-reg@ ldst-dst ! then
    src-token 2drop
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint drop
        else 2drop then
    else >r 2drop r> sym-reg@ drop then ;

: emit-st-shared ( -- )
    src-token 2drop
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint drop
        else 2drop then
    else nip nip sym-reg@ drop then
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst drop
        else 2drop then
    else nip nip sym-reg@ drop then ;

\ ---- Predicated instruction emission ----------------------------------------
\ @pN ... -- reads @pN token, then dispatches the next instruction with predicate prefix

variable pred-reg-num
: emit-predicated ( addr u -- )
    1- swap 1+ swap
    2dup sym-find dup -1 <> if
        nip nip sym-reg@ pred-reg-num !
    else
        drop 2drop
        src-token 2dup k-bra 3 li-tok= if 2drop src-token 2drop
        else 2drop then  exit
    then
    src-token
    2dup k-bra 3 li-tok= if
        2drop src-token 2drop  0 pred-reg-num @ bra-pred, exit
    then
    2dup s" mov" 3 li-tok= if
        2drop
        src-token 2dup sym-find dup -1 <> if nip nip sym-reg@ drop
        else drop 2drop then
        src-token 2dup sym-find dup -1 <> if nip nip sym-reg@ drop
        else 2dup is-float? if emit-fconst drop else 2drop then then
        exit
    then
    2drop ;

\ ---- Statement parser --------------------------------------------------------
\ Extended to handle all new features

variable stmt-matched   \ -1 if a keyword matched

\ Dispatch keyword group 1: control flow + bitwise
: stmt-dispatch1 ( addr u -- result | addr u )
    2dup k-fn 2 li-tok= if 2drop 0 -1 stmt-matched ! exit then
    2dup k-each 4 li-tok= if
        2drop src-token
        inline-mode @ if
            \ In inline mode, skip the each (caller already set up thread indexing).
            \ Just consume the variable name token without re-emitting tid code.
            2drop
        else
            3 3 sym-add drop emit-each-tid
        then
        -1 -1 stmt-matched ! exit
    then
    2dup k-endfor 6 li-tok= if 2drop emit-endfor -1 -1 stmt-matched ! exit then
    2dup k-for 3 li-tok= if 2drop emit-for-v2 -1 -1 stmt-matched ! exit then
    2dup k-endstride 9 li-tok= if 2drop emit-endstride -1 -1 stmt-matched ! exit then
    2dup k-stride 6 li-tok= if 2drop emit-stride -1 -1 stmt-matched ! exit then
    2dup k-shr 3 li-tok= if 2drop emit-shr -1 -1 stmt-matched ! exit then
    2dup k-shl 3 li-tok= if 2drop emit-shl -1 -1 stmt-matched ! exit then
    2dup k-and 3 li-tok= if 2drop emit-and -1 -1 stmt-matched ! exit then
    2dup k-or  2 li-tok= if 2drop emit-or  -1 -1 stmt-matched ! exit then
    2dup k-xor 3 li-tok= if 2drop emit-xor -1 -1 stmt-matched ! exit then
    2dup k-shfl 9 li-tok= if 2drop emit-shfl-bfly -1 -1 stmt-matched ! exit then
    2dup k-shared 6 li-tok= if 2drop parse-shared-v2 -1 -1 stmt-matched ! exit then
    2dup k-param 5 li-tok= if 2drop parse-scalar-param -1 -1 stmt-matched ! exit then
    ;

\ Dispatch keyword group 2: math + conversions
: stmt-dispatch2 ( addr u -- result | addr u )
    2dup k-exp   3 li-tok= if 2drop emit-exp   -1 -1 stmt-matched ! exit then
    2dup k-rcp   3 li-tok= if 2drop emit-rcp   -1 -1 stmt-matched ! exit then
    2dup k-rsqrt 5 li-tok= if 2drop emit-rsqrt -1 -1 stmt-matched ! exit then
    2dup k-sqrt  4 li-tok= if 2drop emit-sqrt  -1 -1 stmt-matched ! exit then
    2dup k-sin   3 li-tok= if 2drop emit-sin   -1 -1 stmt-matched ! exit then
    2dup k-cos   3 li-tok= if 2drop emit-cos   -1 -1 stmt-matched ! exit then
    2dup k-neg   3 li-tok= if 2drop emit-neg   -1 -1 stmt-matched ! exit then
    2dup k-fma   3 li-tok= if 2drop emit-fma   -1 -1 stmt-matched ! exit then
    2dup k-f32s32 7 li-tok= if 2drop emit-f32-to-s32 -1 -1 stmt-matched ! exit then
    2dup k-f32u32 7 li-tok= if 2drop emit-f32-to-u32 -1 -1 stmt-matched ! exit then
    2dup k-u32f32 7 li-tok= if 2drop emit-u32-to-f32 -1 -1 stmt-matched ! exit then
    2dup k-s32f32 7 li-tok= if 2drop emit-s32-to-f32 -1 -1 stmt-matched ! exit then
    ;

\ Dispatch keyword group 3: control + memory + misc
: stmt-dispatch3 ( addr u -- result | addr u )
    2dup k-setp 4 li-tok= if 2drop emit-setp -1 -1 stmt-matched ! exit then
    2dup k-ifge 4 li-tok= if 2drop emit-ifge-exit -1 -1 stmt-matched ! exit then
    2dup k-iflt 4 li-tok= if 2drop emit-iflt-exit -1 -1 stmt-matched ! exit then
    2dup k-label 5 li-tok= if 2drop emit-label -1 -1 stmt-matched ! exit then
    2dup k-bra 3 li-tok= if 2drop emit-bra -1 -1 stmt-matched ! exit then
    2dup k-bar 7 li-tok= if 2drop emit-barrier -1 -1 stmt-matched ! exit then
    2dup k-mov 3 li-tok= if 2drop emit-mov -1 -1 stmt-matched ! exit then
    2dup k-add 3 li-tok= if 2drop emit-add-u32 -1 -1 stmt-matched ! exit then
    2dup k-sub 3 li-tok= if 2drop emit-sub-u32 -1 -1 stmt-matched ! exit then
    2dup k-mul 3 li-tok= if 2drop emit-mul-u32 -1 -1 stmt-matched ! exit then
    2dup k-mad 3 li-tok= if 2drop emit-mad     -1 -1 stmt-matched ! exit then
    2dup k-ldg 8 li-tok= if 2drop emit-ld-global-v2 -1 -1 stmt-matched ! exit then
    2dup k-stg 8 li-tok= if 2drop emit-st-global-v2 -1 -1 stmt-matched ! exit then
    2dup k-lds 9 li-tok= if 2drop emit-ld-shared -1 -1 stmt-matched ! exit then
    2dup k-sts 9 li-tok= if 2drop emit-st-shared -1 -1 stmt-matched ! exit then
    ;

: stmt-dispatch4 ( addr u -- result | addr u )
    2dup k-cvt 3 li-tok= if 2drop
        src-token
        src-token 2dup sym-find dup -1 = if
            drop freg+ >r 2 r@ sym-add >r r> drop r>
        else nip nip sym-reg@ then
        src-token 2dup sym-find dup -1 = if
            drop 2dup is-number? if parse-uint emit-iconst
            else 2drop 0 then
        else nip nip sym-reg@ then
        >r >r
        r> drop r> drop 2drop
        -1 -1 stmt-matched ! exit
    then
    over c@ [char] @ = if emit-predicated -1 -1 stmt-matched ! exit then
    ;

variable ps-dst-freg
variable ps-saved-pos

: parse-stmt  ( addr u -- continue? )
    0 stmt-matched !
    stmt-dispatch1  stmt-matched @ if exit then
    stmt-dispatch2  stmt-matched @ if exit then
    stmt-dispatch3  stmt-matched @ if exit then
    stmt-dispatch4  stmt-matched @ if exit then

    \ ---- Original expression-based statements ----
    2dup sym-find dup -1 = if
        \ New name — peek for "=" (local assignment)
        drop
        src-pos @ ps-saved-pos !
        src-token dup 0= if 2drop ps-saved-pos @ src-pos ! 2drop -1 exit then
        2dup k-eq 1 li-tok= if
            2drop
            freg+ ps-dst-freg !
            2 ps-dst-freg @ sym-add drop
            parse-expr
            dup ps-dst-freg @ = if
                drop
            else
                drop
            then
            -1 exit
        then
        2dup k-lbrack 1 li-tok= if
            2drop ps-saved-pos @ src-pos ! 2drop -1 exit
        then
        2drop ps-saved-pos @ src-pos ! 2drop -1 exit
    then
    nip nip                         \ ( sym-idx ) drop addr u
    \ Known symbol — peek for "[" or "="
    src-pos @ ps-saved-pos !
    src-token dup 0= if 2drop ps-saved-pos @ src-pos ! drop -1 exit then
    2dup k-lbrack 1 li-tok= if
        2drop                       ( sym-idx )
        \ In inline mode with a local-f32 output, capture result instead of storing
        inline-mode @ over sym-kind@ 2 = and if
            >r
            resolve-idx-expr drop   \ consume index, discard rreg
            src-token 2drop         \ consume "]"
            src-token 2drop         \ consume "="
            parse-expr              \ ( freg ) the computed value
            dup inline-result-freg !
            \ Update the output sym's register to point to this freg
            r> cells sym-regs + !
            -1 exit
        then
        resolve-idx-expr            \ parse simple var or A*C+B expr -> rreg
        src-token 2drop             \ consume "]"
        src-token 2drop             \ consume "="
        swap                        ( idx-rreg sym-idx )
        >r parse-expr r>            ( idx-rreg freg sym-idx )
        sym-reg@ -rot               ( param-rd idx-rreg freg )
        emit-istore-r
        -1 exit
    then
    2dup k-eq 1 li-tok= if
        2drop                       ( sym-idx )
        parse-expr                  ( sym-idx freg )
        over sym-kind@ 2 = if
            drop
            drop
        else 2drop then
        -1 exit
    then
    2drop ps-saved-pos @ src-pos ! drop -1 ;

\ ---- Parse a complete function -----------------------------------------------

: parse-fn-params  ( -- )
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then
        2dup k-arrow 2 li-tok= if 2drop r> drop exit then
        \ Check if this is "param" (scalar param before ->)
        2dup k-param 5 li-tok= if
            2drop r> drop
            parse-scalar-param
        else
            r> drop
            0 n-inputs @ sym-add drop
            1 n-inputs +!
        then
    again ;

variable pfb-saved-pos
: parse-fn-body  ( -- )
    begin
        src-pos @ pfb-saved-pos !
        src-token dup 0= if 2drop exit then
        parse-stmt
        0= if pfb-saved-pos @ src-pos ! exit then
    again ;

\ Check if token is a body keyword (returns -1 if yes, 0 if no)
: is-body-keyword? ( addr u -- flag )
    2dup k-each   4 li-tok= if 2drop -1 exit then
    2dup k-fn     2 li-tok= if 2drop -1 exit then
    2dup k-for    3 li-tok= if 2drop -1 exit then
    2dup k-ifge   4 li-tok= if 2drop -1 exit then
    2dup k-iflt   4 li-tok= if 2drop -1 exit then
    2dup k-label  5 li-tok= if 2drop -1 exit then
    2dup k-shr    3 li-tok= if 2drop -1 exit then
    2dup k-shl    3 li-tok= if 2drop -1 exit then
    2dup k-and    3 li-tok= if 2drop -1 exit then
    2dup k-or     2 li-tok= if 2drop -1 exit then
    2dup k-xor    3 li-tok= if 2drop -1 exit then
    2dup k-mov    3 li-tok= if 2drop -1 exit then
    2dup k-add    3 li-tok= if 2drop -1 exit then
    2dup k-sub    3 li-tok= if 2drop -1 exit then
    2dup k-mul    3 li-tok= if 2drop -1 exit then
    2dup k-mad    3 li-tok= if 2drop -1 exit then
    2dup k-exp    3 li-tok= if 2drop -1 exit then
    2dup k-rcp    3 li-tok= if 2drop -1 exit then
    2drop 0 ;

: is-body-keyword2? ( addr u -- flag )
    2dup k-rsqrt  5 li-tok= if 2drop -1 exit then
    2dup k-sqrt   4 li-tok= if 2drop -1 exit then
    2dup k-neg    3 li-tok= if 2drop -1 exit then
    2dup k-fma    3 li-tok= if 2drop -1 exit then
    2dup k-sin    3 li-tok= if 2drop -1 exit then
    2dup k-cos    3 li-tok= if 2drop -1 exit then
    2dup k-setp   4 li-tok= if 2drop -1 exit then
    2dup k-bar    7 li-tok= if 2drop -1 exit then
    2dup k-bra    3 li-tok= if 2drop -1 exit then
    2dup k-shfl   9 li-tok= if 2drop -1 exit then
    2dup k-cvt    3 li-tok= if 2drop -1 exit then
    2dup k-f32s32 7 li-tok= if 2drop -1 exit then
    2dup k-f32u32 7 li-tok= if 2drop -1 exit then
    2dup k-u32f32 7 li-tok= if 2drop -1 exit then
    2dup k-s32f32 7 li-tok= if 2drop -1 exit then
    2dup k-endfor 6 li-tok= if 2drop -1 exit then
    2dup k-stride    6 li-tok= if 2drop -1 exit then
    2dup k-endstride 9 li-tok= if 2drop -1 exit then
    2dup k-ldg    8 li-tok= if 2drop -1 exit then
    2dup k-stg    8 li-tok= if 2drop -1 exit then
    2dup k-lds    9 li-tok= if 2drop -1 exit then
    2dup k-sts    9 li-tok= if 2drop -1 exit then
    over c@ [char] @ = if 2drop -1 exit then
    2drop 0 ;

: parse-fn-decls-and-outputs  ( -- )
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then

        \ Check body keywords (split into two words to avoid Forth limits)
        2dup is-body-keyword? if 2drop r> src-pos ! exit then
        2dup is-body-keyword2? if 2drop r> src-pos ! exit then

        \ Declarations
        2dup k-param 5 li-tok= if
            2drop r> drop
            parse-scalar-param
        else 2dup k-shared 6 li-tok= if
            2drop r> drop
            parse-shared-v2
        else
            \ Must be an output name
            r> drop
            1 n-inputs @ n-outputs @ + sym-add drop
            1 n-outputs +!
        then then
    again ;

\ ---- Variables for function registration after parsing -----------------------
variable fn-body-start
variable fn-body-end

\ Redefine parse-fn with the cleaner structure
: parse-fn  ( -- )
    sym-reset
    0 next-freg !
    4 next-rreg !
    4 next-rdreg !
    0 next-preg !
    0 n-shared !
    0 n-sparams !
    0 for-depth !
    0 next-label !
    regs-reset
    1 li-defs +!  1 li-kernels +!
    src-token dup 0= if 2drop exit then
    li-set-name
    parse-fn-params
    parse-fn-decls-and-outputs
    src-pos @ fn-body-start !
    parse-fn-body
    src-pos @ fn-body-end !
    \ Register this function for inlining
    li-name$ n-inputs @ n-outputs @
    fn-body-start @ fn-body-end @
    fndef-register
    \ Store param and output names from the symbol table
    n-fndefs @ 1- >r
    n-inputs @ 0 ?do
        i sym-name@  r@ i fndef-set-param-name
    loop
    n-outputs @ 0 ?do
        n-inputs @ i + sym-name@  r@ i fndef-set-out-name
    loop
    r> drop ;

\ ---- Top-level token handler -------------------------------------------------

: li-body-word  ( addr u -- )
    2dup k-fn 2 li-tok= if
        2drop
        parse-fn
        exit
    then
    2drop ;
