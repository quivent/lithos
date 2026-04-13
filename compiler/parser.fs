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
\   for k 0 K 16          — PTX loop with counter, bound, step
\   shr, and, or, xor     — integer/bitwise ops
\   shfl.bfly              — warp shuffle (butterfly)
\   shared buf 5120 f32    — shared memory declaration
\   param n u32            — scalar parameter (u32 or f32)
\   exp, rcp, rsqrt, sqrt, sin, cos — math intrinsics
\   1.0, 0.5, -8.0        — float constants (IEEE 754 hex)
\   f32>s32, u32>f32 etc   — type conversions
\   @p0 bra DONE           — predicated branch
\   if>= tid n exit        — bounds check / early exit

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

\ ---- PTX emission helpers ----------------------------------------------------

: ptx-indent  s"     " ptx+ ;
: ptx-freg  ( n -- )   s" %f"  ptx+  ptx-num ptx+ ;
: ptx-r32   ( n -- )   s" %r"  ptx+  ptx-num ptx+ ;
: ptx-r64   ( n -- )   s" %rd" ptx+  ptx-num ptx+ ;
: ptx-preg  ( n -- )   s" %p"  ptx+  ptx-num ptx+ ;

\ ---- IEEE 754 float encoding -------------------------------------------------
\ Convert a float string like "1.0" or "0.5" to IEEE 754 hex "0f3F800000"
\ We use a lookup table for common constants to avoid needing FP in Forth.

\ Helper: emit 4-bit hex digit
: hex-digit ( n -- c )
    dup 10 < if [char] 0 + else 10 - [char] A + then ;

\ Emit a 32-bit value as 8 hex digits
: emit-hex32 ( n -- )
    8 0 do
        dup 28 rshift hex-digit ptx-c
        4 lshift
    loop drop ;

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
        2dup i cells fconst-lens + @ over = if
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

\ ---- PTX header for a function -----------------------------------------------

variable header-emitted  0 header-emitted !

\ Count pointer params (kinds 0 and 1)
: count-ptr-params ( -- n )
    0 n-syms @ 0 ?do
        i sym-kind@ dup 0 = swap 1 = or if 1+ then
    loop ;

\ Count all entry params (ptr + scalar)
: count-all-params ( -- n )
    count-ptr-params n-sparams @ + ;

: ptx-emit-header  ( -- )
    header-emitted @ if exit then  1 header-emitted !
    ptx-header
    s" .visible .entry " ptx+  li-name$ ptx+  s" (" ptx+ ptx-nl

    \ Emit pointer params
    0  \ param counter
    n-syms @ 0 ?do
        i sym-kind@ dup 0 = swap 1 = or if
            ptx-indent s" .param .u64 " ptx+
            i sym-name@ ptx+
            1+
            dup count-all-params < if s" ," ptx+ then
            ptx-nl
        then
    loop

    \ Emit scalar params
    n-sparams @ 0 ?do
        ptx-indent
        i sparam-type@ 0= if s" .param .u32 " ptx+ else s" .param .f32 " ptx+ then
        s" param_" ptx+
        i sparam-name@ ptx+
        1+
        dup count-all-params < if s" ," ptx+ then
        ptx-nl
    loop
    drop

    s" )" ptx+ ptx-nl
    s" {" ptx+ ptx-nl

    \ Shared memory declarations
    n-shared @ 0 ?do
        ptx-indent s" .shared .align 16 .b8 " ptx+
        i shm-name@ ptx+ s" [" ptx+
        i cells shm-bytes + @ ptx-num ptx+ s" ];" ptx+ ptx-nl
    loop

    \ Register pools (generous sizes)
    ptx-indent s" .reg .pred %p<32>;"  ptx+ ptx-nl
    ptx-indent s" .reg .b32  %r<128>;" ptx+ ptx-nl
    ptx-indent s" .reg .b64  %rd<64>;" ptx+ ptx-nl
    ptx-indent s" .reg .f32  %f<128>;" ptx+ ptx-nl
    ptx-nl

    \ Load pointer params into rd registers
    n-syms @ 0 ?do
        i sym-kind@ dup 0 = swap 1 = or if
            ptx-indent s" ld.param.u64 " ptx+  i sym-reg@ ptx-r64
            s" , [" ptx+  i sym-name@ ptx+  s" ];" ptx+ ptx-nl
        then
    loop

    \ Load scalar params into r/f registers
    n-sparams @ 0 ?do
        i sparam-type@ 0= if
            ptx-indent s" ld.param.u32 " ptx+
            \ Find the sym for this scalar param
            i sparam-name@ sym-find dup -1 <> if
                dup sym-reg@ ptx-r32
            else drop then
            s" , [param_" ptx+  i sparam-name@ ptx+  s" ];" ptx+ ptx-nl
        else
            ptx-indent s" ld.param.f32 " ptx+
            i sparam-name@ sym-find dup -1 <> if
                dup sym-reg@ ptx-freg
            else drop then
            s" , [param_" ptx+  i sparam-name@ ptx+  s" ];" ptx+ ptx-nl
        then
    loop
    ptx-nl ;

: ptx-emit-footer  ( -- )
    ptx-nl
    s" $L_exit:" ptx+ ptx-nl
    ptx-indent s" ret;" ptx+ ptx-nl
    s" }" ptx+ ptx-nl ;

\ ---- Emit "each" (thread indexing) -------------------------------------------
\ %r0=ctaid.x  %r1=ntid.x  %r2=tid.x  %r3=global_tid

: emit-each-tid  ( -- )
    ptx-indent s" mov.u32 %r0, %ctaid.x;" ptx+ ptx-nl
    ptx-indent s" mov.u32 %r1, %ntid.x;"  ptx+ ptx-nl
    ptx-indent s" mov.u32 %r2, %tid.x;"   ptx+ ptx-nl
    ptx-indent s" mad.lo.s32 %r3, %r0, %r1, %r2;" ptx+ ptx-nl
    ptx-nl ;

\ ---- Emit indexed load: freg = param_base_rd[%r3] (f32) --------------------

: emit-iload  ( param-rd -- freg )
    freg+ swap
    ptx-indent s" mul.wide.u32 %rd22, %r3, 4;" ptx+ ptx-nl
    ptx-indent s" add.u64 %rd23, " ptx+  ptx-r64  s" , %rd22;" ptx+ ptx-nl
    ptx-indent s" ld.global.f32 " ptx+  dup ptx-freg  s" , [%rd23];" ptx+ ptx-nl ;

\ ---- Emit indexed store: param_base_rd[%r3] = freg --------------------------

: emit-istore  ( param-rd freg -- )
    swap
    ptx-indent s" mul.wide.u32 %rd22, %r3, 4;" ptx+ ptx-nl
    ptx-indent s" add.u64 %rd23, " ptx+  ptx-r64  s" , %rd22;" ptx+ ptx-nl
    ptx-indent s" st.global.f32 [%rd23], " ptx+  ptx-freg  s" ;" ptx+ ptx-nl ;

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
    over c@ [char] 0 = swap 1 + c@ [char] x = and ;

\ ---- Resolve operand: look up symbol, return kind and reg --------------------
\ For use in direct PTX-like statements

\ ---- Expression parser (returns freg holding result) -------------------------
variable 'parse-expr

: peek-tok  ( -- addr u )
    src-pos @ >r  src-token  r> swap >r src-pos !  r> ;

\ Emit a float constant, returns freg
: emit-fconst ( addr u -- freg )
    2dup fconst-find if
        >r 2drop freg+ dup >r
        ptx-indent s" mov.f32 " ptx+  r@ ptx-freg  s" , 0f" ptx+
        r> drop
        r> emit-hex32
        s" ;" ptx+ ptx-nl
    else
        \ Unknown float — emit as 0.0 and print warning
        2drop freg+ dup >r
        ptx-indent s" mov.f32 " ptx+  r@ ptx-freg  s" , 0f00000000;" ptx+ ptx-nl
        r>
    then ;

\ Emit an integer constant into an r32 register
: emit-iconst ( n -- rreg )
    rreg+ dup >r
    ptx-indent s" mov.u32 " ptx+  r@ ptx-r32  s" , " ptx+
    dup 0< if
        \ Negative: emit as unsigned representation
        s" 0x" ptx+ emit-hex32
    else
        ptx-num ptx+
    then
    s" ;" ptx+ ptx-nl
    r> ;

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
        \ Unknown — emit zero
        drop 2drop
        freg+ dup >r
        ptx-indent s" mov.f32 " ptx+  r@ ptx-freg  s" , 0f00000000;" ptx+ ptx-nl
        r> exit
    then
    >r 2drop r>           \ ( sym-idx )
    \ Peek next token for "["
    src-pos @ >r
    src-token dup 0= if
        2drop r> src-pos !
        dup sym-kind@ 2 = if  sym-reg@  else
        dup sym-kind@ 3 = if  drop -1  else  \ each-var shouldn't appear as atom value
        dup sym-kind@ 4 = if  sym-reg@  else  \ local-u32 returns rreg
        dup sym-kind@ 5 = if  sym-reg@  else  \ scalar-u32 returns rreg
        dup sym-kind@ 6 = if  sym-reg@  else  \ scalar-f32 returns freg
        sym-reg@
        then then then then then exit
    then
    2dup k-lbrack 1 li-tok= if
        2drop r> drop
        src-token 2drop          \ consume index var (the "i")
        src-token 2drop          \ consume "]"
        sym-reg@ emit-iload      \ emit load, returns freg
        exit
    then
    \ Not "[" — put token back
    2drop r> src-pos !
    dup sym-kind@ 2 = if  sym-reg@  else
    dup sym-kind@ 3 = if  drop -1  else
    dup sym-kind@ 4 = if  sym-reg@  else
    dup sym-kind@ 5 = if  sym-reg@  else
    dup sym-kind@ 6 = if  sym-reg@  else
    sym-reg@
    then then then then then ;

: parse-term  ( -- freg )
    parse-atom
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then
        2dup k-star 1 li-tok= if
            2drop r> drop
            parse-atom swap       ( freg2 freg1 )
            freg+ >r
            ptx-indent s" mul.f32 " ptx+  r@ ptx-freg  s" , " ptx+
            ptx-freg  s" , " ptx+  ptx-freg  s" ;" ptx+ ptx-nl
            r>
        else 2dup k-slash 1 li-tok= if
            2drop r> drop
            parse-atom swap
            freg+ >r
            ptx-indent s" div.approx.f32 " ptx+  r@ ptx-freg  s" , " ptx+
            ptx-freg  s" , " ptx+  ptx-freg  s" ;" ptx+ ptx-nl
            r>
        else
            2drop r> src-pos ! exit
        then then
    again ;

: parse-expr  ( -- freg )
    parse-term
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then
        2dup k-plus 1 li-tok= if
            2drop r> drop
            parse-term swap
            freg+ >r
            ptx-indent s" add.f32 " ptx+  r@ ptx-freg  s" , " ptx+
            ptx-freg  s" , " ptx+  ptx-freg  s" ;" ptx+ ptx-nl
            r>
        else 2dup k-minus 1 li-tok= if
            2drop r> drop
            parse-term swap
            freg+ >r
            ptx-indent s" sub.f32 " ptx+  r@ ptx-freg  s" , " ptx+
            ptx-freg  s" , " ptx+  ptx-freg  s" ;" ptx+ ptx-nl
            r>
        else
            2drop r> src-pos ! exit
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
        >r 2drop r> sym-reg@ for-counter-reg !
    then

    \ Parse start
    src-token 2dup is-number? if
        parse-uint for-start-val !
    else
        sym-find dup -1 <> if sym-reg@ else drop 0 then for-start-val !
    then

    \ Parse bound
    src-token 2dup sym-find dup -1 <> if
        >r 2drop r> sym-reg@ for-bound-val ! -1 for-bound-is-reg !
    else
        2dup is-number? if
            parse-uint for-bound-val ! 0 for-bound-is-reg !
        else
            sym-find dup -1 <> if sym-reg@ for-bound-val ! -1 for-bound-is-reg !
            else drop 0 for-bound-val ! 0 for-bound-is-reg ! then
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

    \ Emit: mov counter, start
    ptx-indent s" mov.u32 " ptx+
    for-counter-reg @ ptx-r32 s" , " ptx+
    for-start-val @ ptx-num ptx+ s" ;" ptx+ ptx-nl

    \ Emit loop label
    s" $L_for_" ptx+ for-label-num @ ptx-num ptx+ s" :" ptx+ ptx-nl

    \ Emit: setp.ge.u32 %pN, counter, bound
    preg+ >r
    ptx-indent s" setp.ge.u32 " ptx+ r@ ptx-preg s" , " ptx+
    for-counter-reg @ ptx-r32 s" , " ptx+
    for-bound-is-reg @ if
        for-bound-val @ ptx-r32
    else
        for-bound-val @ ptx-num ptx+
    then
    s" ;" ptx+ ptx-nl

    \ Emit: @pN bra $L_endfor_X
    ptx-indent s" @" ptx+ r> ptx-preg s"  bra $L_endfor_" ptx+
    for-label-num @ ptx-num ptx+ s" ;" ptx+ ptx-nl
    ptx-nl ;

: emit-endfor ( -- )
    for-depth @ 0= if exit then
    -1 for-depth +!

    \ Get loop info
    for-depth @ cells for-labels + @ >r
    for-depth @ cells for-counters + @ >r

    \ Emit: add.u32 counter, counter, step
    ptx-indent s" add.u32 " ptx+
    r@ ptx-r32 s" , " ptx+
    r@ ptx-r32 s" , " ptx+
    for-step-val @ ptx-num ptx+ s" ;" ptx+ ptx-nl

    \ Emit: bra $L_for_X
    ptx-indent s" bra $L_for_" ptx+
    r> drop r> ptx-num ptx+ s" ;" ptx+ ptx-nl

    \ Emit: $L_endfor_X:
    s" $L_endfor_" ptx+ for-depth @ cells for-labels + @ ptx-num ptx+ s" :" ptx+ ptx-nl
    ptx-nl ;

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
    else >r 2drop r> sym-reg@ bw-dst ! then

    \ Parse src1
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst bw-src1 !
        else 2drop 0 bw-src1 ! then
    else >r 2drop r> sym-reg@ bw-src1 ! then

    \ Parse src2
    0 bw-src2-imm !
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint bw-src2 ! -1 bw-src2-imm !
        else 2dup is-hex? if
            2 - swap 2 + swap parse-hex bw-src2 ! -1 bw-src2-imm !
        else 2drop 0 bw-src2 ! then then
    else >r 2drop r> sym-reg@ bw-src2 ! then ;

: emit-bw-instr ( op-addr op-u -- )
    dup bw-op-len ! bw-op-buf swap move
    parse-bw-args
    ptx-indent bw-op-buf bw-op-len @ ptx+
    s" .b32 " ptx+
    bw-dst @ ptx-r32 s" , " ptx+
    bw-src1 @ ptx-r32 s" , " ptx+
    bw-src2-imm @ if
        bw-src2 @ ptx-num ptx+
    else
        bw-src2 @ ptx-r32
    then
    s" ;" ptx+ ptx-nl ;

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

: emit-shfl-bfly ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then   \ dst

    src-token 2dup sym-find dup -1 = if
        drop 2drop 0
    else >r 2drop r> sym-reg@ then   \ src

    src-token 2dup is-number? if
        parse-uint
    else
        sym-find dup -1 <> if sym-reg@ else drop 0 then
    then                              \ offset

    >r >r >r
    ptx-indent s" shfl.sync.bfly.b32 " ptx+
    r> ptx-r32 s" , " ptx+           \ dst
    r> ptx-r32 s" , " ptx+           \ src
    r> ptx-num ptx+ s" , 31, -1;" ptx+ ptx-nl ;

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

: emit-math-unary ( ptx-op-addr ptx-op-u -- )
    src-token 2dup sym-find dup -1 = if
        drop freg+ >r 2 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then   \ dst

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if
            emit-fconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ src

    >r >r
    ptx-indent 2swap ptx+ s"  " ptx+
    r> ptx-freg s" , " ptx+
    r> ptx-freg s" ;" ptx+ ptx-nl ;

: emit-exp    s" ex2.approx.f32"   emit-math-unary ;
: emit-rcp    s" rcp.approx.f32"   emit-math-unary ;
: emit-rsqrt  s" rsqrt.approx.f32" emit-math-unary ;
: emit-sqrt   s" sqrt.approx.f32"  emit-math-unary ;
: emit-sin    s" sin.approx.f32"   emit-math-unary ;
: emit-cos    s" cos.approx.f32"   emit-math-unary ;
: emit-neg    s" neg.f32"          emit-math-unary ;

\ FMA: fma DST A B C -> fma.rn.f32 DST, A, B, C
: emit-fma ( -- )
    src-token 2dup sym-find dup -1 = if
        drop freg+ >r 2 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then   \ dst

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ a

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ b

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ c

    >r >r >r >r
    ptx-indent s" fma.rn.f32 " ptx+
    r> ptx-freg s" , " ptx+
    r> ptx-freg s" , " ptx+
    r> ptx-freg s" , " ptx+
    r> ptx-freg s" ;" ptx+ ptx-nl ;

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

: emit-cvt-v2 ( ptx-suffix-addr ptx-suffix-u dst-float src-float -- )
    cvt-src-float ! cvt-dst-float !
    \ Save suffix string
    dup cvt-suffix-len ! cvt-suffix-buf swap move

    \ Parse DST
    src-token 2dup sym-find dup -1 = if
        drop
        cvt-dst-float @ if
            freg+ dup cvt-dst-reg !
            >r 2 r> sym-add drop
        else
            rreg+ dup cvt-dst-reg !
            >r 4 r> sym-add drop
        then
    else
        >r 2drop r> sym-reg@ cvt-dst-reg !
    then

    \ Parse SRC
    src-token 2dup sym-find dup -1 = if
        drop
        2dup is-float? if
            emit-fconst cvt-src-reg !
        else
            2dup is-number? if
                parse-uint emit-iconst cvt-src-reg !
            else
                2drop 0 cvt-src-reg !
            then
        then
    else
        >r 2drop r> sym-reg@ cvt-src-reg !
    then

    \ Emit: cvt.SUFFIX dst, src
    ptx-indent s" cvt." ptx+ cvt-suffix-buf cvt-suffix-len @ ptx+

    s"  " ptx+
    cvt-dst-float @ if cvt-dst-reg @ ptx-freg else cvt-dst-reg @ ptx-r32 then
    s" , " ptx+
    cvt-src-float @ if cvt-src-reg @ ptx-freg else cvt-src-reg @ ptx-r32 then
    s" ;" ptx+ ptx-nl ;

: emit-f32-to-s32  s" rzi.s32.f32" 0 -1 emit-cvt-v2 ;
: emit-f32-to-u32  s" rzi.u32.f32" 0 -1 emit-cvt-v2 ;
: emit-u32-to-f32  s" rn.f32.u32"  -1 0 emit-cvt-v2 ;
: emit-s32-to-f32  s" rn.f32.s32"  -1 0 emit-cvt-v2 ;

\ ==============================================================================
\ FEATURE 9: PREDICATION
\ Syntax: @p0 INSTRUCTION...  or  setp.ge DST A B
\ ==============================================================================

: emit-setp ( -- )
    \ setp.CMP.TYPE pred src1 src2
    \ Parse comparison: ge, lt, eq, ne, gt, le
    src-token                         \ cmp string (e.g., "ge")
    2dup                              \ save

    \ Parse type
    src-token                         \ type string
    2dup                              \ save

    \ Parse pred dst
    src-token 2dup sym-find dup -1 = if
        drop preg+ >r 7 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then

    \ Parse src1
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    \ Parse src2
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    \ ( cmp-addr cmp-u type-addr type-u pred src1 src2 )
    >r >r >r
    ptx-indent s" setp." ptx+
    \ emit cmp
    2swap ptx+
    s" ." ptx+ ptx+       \ type
    s"  " ptx+
    r> ptx-preg s" , " ptx+
    r> ptx-r32 s" , " ptx+
    r> ptx-r32 s" ;" ptx+ ptx-nl ;

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
    else >r 2drop r> sym-reg@ ifge-src1 ! then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst ifge-src2 !
        else 2drop 0 ifge-src2 ! then
    else >r 2drop r> sym-reg@ ifge-src2 ! then

    src-token dup ifge-target-len ! ifge-target swap move ;

: emit-ifge-exit ( -- )
    parse-ifge-args
    preg+ ifge-pred !
    ptx-indent s" setp.ge.u32 " ptx+ ifge-pred @ ptx-preg s" , " ptx+
    ifge-src1 @ ptx-r32 s" , " ptx+
    ifge-src2 @ ptx-r32 s" ;" ptx+ ptx-nl
    ptx-indent s" @" ptx+ ifge-pred @ ptx-preg s"  bra $L_" ptx+
    ifge-target ifge-target-len @ ptx+ s" ;" ptx+ ptx-nl ;

: emit-iflt-exit ( -- )
    parse-ifge-args
    preg+ ifge-pred !
    ptx-indent s" setp.lt.u32 " ptx+ ifge-pred @ ptx-preg s" , " ptx+
    ifge-src1 @ ptx-r32 s" , " ptx+
    ifge-src2 @ ptx-r32 s" ;" ptx+ ptx-nl
    ptx-indent s" @" ptx+ ifge-pred @ ptx-preg s"  bra $L_" ptx+
    ifge-target ifge-target-len @ ptx+ s" ;" ptx+ ptx-nl ;

\ ==============================================================================
\ EXTENDED STATEMENT: Direct PTX-like instructions
\ ==============================================================================

\ Emit label: label NAME -> $L_NAME:
: emit-label ( -- )
    src-token
    s" $L_" ptx+ ptx+ s" :" ptx+ ptx-nl ;

\ Emit branch: bra LABEL -> bra $L_LABEL
: emit-bra ( -- )
    src-token
    ptx-indent s" bra $L_" ptx+ ptx+ s" ;" ptx+ ptx-nl ;

\ Emit barrier
: emit-barrier ( -- )
    ptx-indent s" bar.sync 0;" ptx+ ptx-nl ;

\ Emit mov: mov DST SRC (u32 move between registers)
: emit-mov ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
            >r >r
            ptx-indent s" mov.u32 " ptx+ r> ptx-r32 s" , " ptx+
            r> ptx-num ptx+ s" ;" ptx+ ptx-nl exit
        else
            2drop 0
        then
    else >r 2drop r> sym-reg@ then

    swap >r >r
    ptx-indent s" mov.u32 " ptx+ r> ptx-r32 s" , " ptx+ r> ptx-r32 s" ;" ptx+ ptx-nl ;

\ Emit add.u32: add DST SRC1 SRC2
: emit-add-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
            >r >r >r
            ptx-indent s" add.u32 " ptx+
            r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+
            r> ptx-num ptx+ s" ;" ptx+ ptx-nl exit
        else
            2drop 0
        then
    else >r 2drop r> sym-reg@ then

    >r >r >r
    ptx-indent s" add.u32 " ptx+
    r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+ r> ptx-r32 s" ;" ptx+ ptx-nl ;

\ Emit sub.u32
: emit-sub-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
            >r >r >r
            ptx-indent s" sub.u32 " ptx+
            r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+
            r> ptx-num ptx+ s" ;" ptx+ ptx-nl exit
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    >r >r >r
    ptx-indent s" sub.u32 " ptx+
    r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+ r> ptx-r32 s" ;" ptx+ ptx-nl ;

\ mul.lo.u32
: emit-mul-u32 ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
            >r >r >r
            ptx-indent s" mul.lo.u32 " ptx+
            r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+
            r> ptx-num ptx+ s" ;" ptx+ ptx-nl exit
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then

    >r >r >r
    ptx-indent s" mul.lo.u32 " ptx+
    r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+ r> ptx-r32 s" ;" ptx+ ptx-nl ;

\ mad.lo.u32
: emit-mad ( -- )
    src-token 2dup sym-find dup -1 = if
        drop rreg+ >r 4 r@ sym-add >r r> drop r>
    else >r 2drop r> sym-reg@ then   \ dst

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ a

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ b

    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if parse-uint emit-iconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ c

    >r >r >r >r
    ptx-indent s" mad.lo.u32 " ptx+
    r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+ r> ptx-r32 s" , " ptx+
    r> ptx-r32 s" ;" ptx+ ptx-nl ;

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
    else >r 2drop r> sym-reg@ ldst-dst ! then

    \ Parse base (pointer / rd reg)
    src-token 2dup sym-find dup -1 = if
        drop 2drop 0 ldst-base !
    else >r 2drop r> sym-reg@ ldst-base ! then

    \ Parse offset (r32 reg)
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst ldst-offset !
        else 2drop 0 ldst-offset ! then
    else >r 2drop r> sym-reg@ ldst-offset ! then

    \ Optional type
    src-pos @ >r
    src-token 2dup k-u32 3 li-tok= if
        2drop r> drop 1 ldst-type !
    else 2dup k-f32 3 li-tok= if
        2drop r> drop 0 ldst-type !
    else
        2drop r> src-pos ! 0 ldst-type !
    then then

    \ Emit address calc
    ptx-indent s" mul.wide.u32 %rd30, " ptx+
    ldst-offset @ ptx-r32 s" , 4;" ptx+ ptx-nl
    ptx-indent s" add.u64 %rd31, " ptx+
    ldst-base @ ptx-r64 s" , %rd30;" ptx+ ptx-nl

    \ Emit load
    ldst-type @ 0= if
        ptx-indent s" ld.global.f32 " ptx+
        ldst-dst @ ptx-freg s" , [%rd31];" ptx+ ptx-nl
    else
        ptx-indent s" ld.global.u32 " ptx+
        ldst-dst @ ptx-r32 s" , [%rd31];" ptx+ ptx-nl
    then ;

: emit-st-global-v2 ( -- )
    \ st.global BASE OFFSET SRC
    \ Parse base
    src-token 2dup sym-find dup -1 = if
        drop 2drop 0 ldst-base !
    else >r 2drop r> sym-reg@ ldst-base ! then

    \ Parse offset (r32 reg)
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint emit-iconst ldst-offset !
        else 2drop 0 ldst-offset ! then
    else >r 2drop r> sym-reg@ ldst-offset ! then

    \ Parse src
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if
            emit-fconst ldst-dst !
        else 2dup is-number? if
            parse-uint emit-iconst ldst-dst !
        else 2drop 0 ldst-dst ! then then
    else >r 2drop r> sym-reg@ ldst-dst ! then

    \ Optional type
    src-pos @ >r
    src-token 2dup k-u32 3 li-tok= if
        2drop r> drop 1 ldst-type !
    else 2dup k-f32 3 li-tok= if
        2drop r> drop 0 ldst-type !
    else
        2drop r> src-pos ! 0 ldst-type !
    then then

    \ Emit address calc
    ptx-indent s" mul.wide.u32 %rd30, " ptx+
    ldst-offset @ ptx-r32 s" , 4;" ptx+ ptx-nl
    ptx-indent s" add.u64 %rd31, " ptx+
    ldst-base @ ptx-r64 s" , %rd30;" ptx+ ptx-nl

    \ Emit store
    ldst-type @ 0= if
        ptx-indent s" st.global.f32 [%rd31], " ptx+
        ldst-dst @ ptx-freg s" ;" ptx+ ptx-nl
    else
        ptx-indent s" st.global.u32 [%rd31], " ptx+
        ldst-dst @ ptx-r32 s" ;" ptx+ ptx-nl
    then ;

\ ---- Shared memory load/store ------------------------------------------------
\ ld.shared DST SHMEM_NAME OFFSET
\ st.shared SHMEM_NAME OFFSET SRC

: emit-ld-shared ( -- )
    \ dst
    src-token 2dup sym-find dup -1 = if
        drop freg+ dup ldst-dst !
        >r 2 r> sym-add drop
    else >r 2drop r> sym-reg@ ldst-dst ! then

    \ shared mem name
    src-token ptx-indent s" ld.shared.f32 " ptx+
    ldst-dst @ ptx-freg s" , [" ptx+
    ptx+  \ emit shared name

    \ offset
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
            s" +" ptx+ ptx-num ptx+
        else 2drop then
    else >r 2drop r>
        sym-reg@
        \ Can't do register offset in shared addressing easily, use separate add
        \ For now emit as immediate 0
        drop
    then
    s" ];" ptx+ ptx-nl ;

: emit-st-shared ( -- )
    \ shared mem name
    src-token 2dup   \ save name

    \ offset
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-number? if
            parse-uint
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ offset val

    \ src
    src-token 2dup sym-find dup -1 = if
        drop 2dup is-float? if emit-fconst
        else 2drop 0 then
    else >r 2drop r> sym-reg@ then   \ src reg

    >r >r
    ptx-indent s" st.shared.f32 [" ptx+
    ptx+   \ shared name
    s" +" ptx+ r> ptx-num ptx+ s" ], " ptx+
    r> ptx-freg s" ;" ptx+ ptx-nl ;

\ ---- Predicated instruction emission ----------------------------------------
\ @pN ... -- reads @pN token, then dispatches the next instruction with predicate prefix

: emit-predicated ( addr u -- )
    \ The token starts with '@' — it's like @p0, @p1 etc
    \ Extract predicate register number
    \ Token format: @pN or @!pN
    ptx-indent ptx+   \ emit the @pN as-is
    s"  " ptx+

    \ Now parse the instruction after it
    src-token
    2dup k-bra 3 li-tok= if
        2drop
        src-token   \ label
        s" bra $L_" ptx+ ptx+ s" ;" ptx+ ptx-nl
        exit
    then
    2dup s" mov" 3 li-tok= if
        2drop
        s" mov.f32 " ptx+
        src-token 2dup sym-find dup -1 <> if
            >r 2drop r> sym-reg@ ptx-freg
        else drop 2drop then
        s" , " ptx+
        src-token 2dup sym-find dup -1 <> if
            >r 2drop r> sym-reg@ ptx-freg
        else 2dup is-float? if emit-fconst ptx-freg
        else 2drop then then
        s" ;" ptx+ ptx-nl
        exit
    then
    \ Unknown predicated instruction — emit as raw text
    ptx+ ptx-nl ;

\ ---- Raw PTX emission --------------------------------------------------------
\ ptx "any raw ptx text here"
\ Allows embedding arbitrary PTX when needed

create k-ptx 3 allot  s" ptx" k-ptx swap move
create k-raw 3 allot  s" raw" k-raw swap move

\ ---- Statement parser --------------------------------------------------------
\ Extended to handle all new features

: parse-stmt  ( addr u -- continue? )
    \ Check for "fn" (starts a new function, so current one is done)
    2dup k-fn 2 li-tok= if 2drop 0 exit then

    \ Check for "each"
    2dup k-each 4 li-tok= if
        2drop
        src-token                    \ variable name
        3 3 sym-add drop            \ kind=3 (each-var), reg=%r3
        emit-each-tid
        -1 exit
    then

    \ Check for "endfor"
    2dup k-endfor 6 li-tok= if
        2drop emit-endfor -1 exit
    then

    \ Check for "for"
    2dup k-for 3 li-tok= if
        2drop emit-for-v2 -1 exit
    then

    \ Check for bitwise ops
    2dup k-shr 3 li-tok= if 2drop emit-shr -1 exit then
    2dup k-shl 3 li-tok= if 2drop emit-shl -1 exit then
    2dup k-and 3 li-tok= if 2drop emit-and -1 exit then
    2dup k-or  2 li-tok= if 2drop emit-or  -1 exit then
    2dup k-xor 3 li-tok= if 2drop emit-xor -1 exit then

    \ Check for warp shuffle
    2dup k-shfl 9 li-tok= if 2drop emit-shfl-bfly -1 exit then

    \ Check for shared memory
    2dup k-shared 6 li-tok= if 2drop parse-shared-v2 -1 exit then

    \ Check for scalar params
    2dup k-param 5 li-tok= if 2drop parse-scalar-param -1 exit then

    \ Check for math intrinsics
    2dup k-exp   3 li-tok= if 2drop emit-exp   -1 exit then
    2dup k-rcp   3 li-tok= if 2drop emit-rcp   -1 exit then
    2dup k-rsqrt 5 li-tok= if 2drop emit-rsqrt -1 exit then
    2dup k-sqrt  4 li-tok= if 2drop emit-sqrt  -1 exit then
    2dup k-sin   3 li-tok= if 2drop emit-sin   -1 exit then
    2dup k-cos   3 li-tok= if 2drop emit-cos   -1 exit then
    2dup k-neg   3 li-tok= if 2drop emit-neg   -1 exit then
    2dup k-fma   3 li-tok= if 2drop emit-fma   -1 exit then

    \ Check for type conversions
    2dup k-f32s32 7 li-tok= if 2drop emit-f32-to-s32 -1 exit then
    2dup k-f32u32 7 li-tok= if 2drop emit-f32-to-u32 -1 exit then
    2dup k-u32f32 7 li-tok= if 2drop emit-u32-to-f32 -1 exit then
    2dup k-s32f32 7 li-tok= if 2drop emit-s32-to-f32 -1 exit then

    \ Check for setp
    2dup k-setp 4 li-tok= if 2drop emit-setp -1 exit then

    \ Check for bounds check / early exit
    2dup k-ifge 4 li-tok= if 2drop emit-ifge-exit -1 exit then
    2dup k-iflt 4 li-tok= if 2drop emit-iflt-exit -1 exit then

    \ Check for label
    2dup k-label 5 li-tok= if 2drop emit-label -1 exit then

    \ Check for bra
    2dup k-bra 3 li-tok= if 2drop emit-bra -1 exit then

    \ Check for barrier
    2dup k-bar 7 li-tok= if 2drop emit-barrier -1 exit then

    \ Check for mov
    2dup k-mov 3 li-tok= if 2drop emit-mov -1 exit then

    \ Check for integer add/sub/mul/mad
    2dup k-add 3 li-tok= if 2drop emit-add-u32 -1 exit then
    2dup k-sub 3 li-tok= if 2drop emit-sub-u32 -1 exit then
    2dup k-mul 3 li-tok= if 2drop emit-mul-u32 -1 exit then
    2dup k-mad 3 li-tok= if 2drop emit-mad     -1 exit then

    \ Check for ld.global / st.global
    2dup k-ldg 8 li-tok= if 2drop emit-ld-global-v2 -1 exit then
    2dup k-stg 8 li-tok= if 2drop emit-st-global-v2 -1 exit then

    \ Check for ld.shared / st.shared
    2dup k-lds 9 li-tok= if 2drop emit-ld-shared -1 exit then
    2dup k-sts 9 li-tok= if 2drop emit-st-shared -1 exit then

    \ Check for predicated instruction (@pN ...)
    over c@ [char] @ = if
        emit-predicated -1 exit
    then

    \ Check for cvt (direct)
    2dup k-cvt 3 li-tok= if 2drop
        \ cvt SUFFIX DST SRC
        src-token   \ suffix like "rn.f32.u32"
        src-token 2dup sym-find dup -1 = if
            drop freg+ >r 2 r@ sym-add >r r> drop r>
        else >r 2drop r> sym-reg@ then
        src-token 2dup sym-find dup -1 = if
            drop 2dup is-number? if parse-uint emit-iconst
            else 2drop 0 then
        else >r 2drop r> sym-reg@ then
        \ ( suffix-addr suffix-u dst src )
        >r >r
        ptx-indent s" cvt." ptx+ ptx+ s"  " ptx+
        r> ptx-freg s" , " ptx+ r> ptx-r32 s" ;" ptx+ ptx-nl
        -1 exit
    then

    \ ---- Original expression-based statements ----
    \ Must be NAME ... = expr  or  NAME [ ... ] = expr
    2dup sym-find dup -1 = if
        \ New name — peek for "=" (local assignment)
        drop
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! 2drop -1 exit then
        2dup k-eq 1 li-tok= if
            2drop r> drop
            \ New local variable — allocate freg, parse expr, copy
            freg+ >r
            2 r@ sym-add drop       \ kind=2 (local), reg=freg
            parse-expr               ( src-freg ) R: dst-freg
            dup r@ = if
                drop r> drop
            else
                ptx-indent s" mov.f32 " ptx+  r@ ptx-freg  s" , " ptx+
                ptx-freg s" ;" ptx+ ptx-nl
                r> drop
            then
            -1 exit
        then
        2dup k-lbrack 1 li-tok= if
            2drop r> src-pos ! 2drop -1 exit
        then
        2drop r> src-pos ! 2drop -1 exit
    then
    >r 2drop r>                     \ ( sym-idx )
    \ Known symbol — peek for "[" or "="
    src-pos @ >r
    src-token dup 0= if 2drop r> src-pos ! drop -1 exit then
    2dup k-lbrack 1 li-tok= if
        2drop r> drop               ( sym-idx )
        src-token 2drop             \ consume index var
        src-token 2drop             \ consume "]"
        src-token 2drop             \ consume "="
        parse-expr                  ( sym-idx freg )
        swap sym-reg@ swap          ( param-rd freg )
        emit-istore
        -1 exit
    then
    2dup k-eq 1 li-tok= if
        2drop r> drop               ( sym-idx )
        parse-expr                  ( sym-idx freg )
        over sym-kind@ 2 = if
            \ Reassign local
            ptx-indent s" mov.f32 " ptx+
            over sym-reg@ ptx-freg  s" , " ptx+  ptx-freg  s" ;" ptx+ ptx-nl
            drop
        else 2drop then
        -1 exit
    then
    2drop r> src-pos ! drop -1 ;

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

: parse-fn-body  ( -- )
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> drop exit then
        parse-stmt
        0= if r> src-pos ! exit then
        r> drop
    again ;

: parse-fn-decls-and-outputs  ( -- )
    \ After parsing input params and "->", read outputs and declarations
    \ until we hit a body keyword (each, for, if>=, label, or known body start)
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then

        \ Body-starting keywords — stop collecting
        2dup k-each   4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-fn     2 li-tok= if 2drop r> src-pos ! exit then
        2dup k-for    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-ifge   4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-iflt   4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-label  5 li-tok= if 2drop r> src-pos ! exit then
        2dup k-shr    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-shl    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-and    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-or     2 li-tok= if 2drop r> src-pos ! exit then
        2dup k-xor    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-mov    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-add    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-sub    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-mul    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-mad    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-exp    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-rcp    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-rsqrt  5 li-tok= if 2drop r> src-pos ! exit then
        2dup k-sqrt   4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-neg    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-fma    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-sin    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-cos    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-setp   4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-ldg    8 li-tok= if 2drop r> src-pos ! exit then
        2dup k-stg    8 li-tok= if 2drop r> src-pos ! exit then
        2dup k-lds    9 li-tok= if 2drop r> src-pos ! exit then
        2dup k-sts    9 li-tok= if 2drop r> src-pos ! exit then
        2dup k-bar    7 li-tok= if 2drop r> src-pos ! exit then
        2dup k-bra    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-shfl   9 li-tok= if 2drop r> src-pos ! exit then
        2dup k-cvt    3 li-tok= if 2drop r> src-pos ! exit then
        2dup k-f32s32 7 li-tok= if 2drop r> src-pos ! exit then
        2dup k-f32u32 7 li-tok= if 2drop r> src-pos ! exit then
        2dup k-u32f32 7 li-tok= if 2drop r> src-pos ! exit then
        2dup k-s32f32 7 li-tok= if 2drop r> src-pos ! exit then
        2dup k-endfor 6 li-tok= if 2drop r> src-pos ! exit then

        \ Check if @ predicated
        over c@ [char] @ = if 2drop r> src-pos ! exit then

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

\ Redefine parse-fn with the cleaner structure
: parse-fn  ( -- )
    sym-reset
    0 header-emitted !
    0 next-freg !
    4 next-rreg !
    4 next-rdreg !
    0 next-preg !
    0 n-shared !
    0 n-sparams !
    0 for-depth !
    0 next-label !
    ptx-reset
    regs-reset
    1 li-defs +!  1 li-kernels +!
    src-token dup 0= if 2drop exit then
    li-set-name
    parse-fn-params
    parse-fn-decls-and-outputs
    ptx-emit-header
    parse-fn-body
    ptx-emit-footer ;

\ ---- Top-level token handler -------------------------------------------------

: li-body-word  ( addr u -- )
    2dup k-fn 2 li-tok= if
        2drop
        parse-fn
        exit
    then
    2drop ;
