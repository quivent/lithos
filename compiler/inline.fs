\ inline.fs — Function inlining for Lithos (this IS fusion)
\
\ In Lithos, every function call is inlined: the callee's body is emitted
\ directly into the caller's PTX stream. This means composing functions
\ produces a single GPU kernel — no call overhead, no separate launches.
\ That is what "fusion" means in Lithos: composed math functions become
\ one GPU-executable body.
\
\ Implementation: when the parser encounters a function call (NAME arg1 arg2)
\ inside a body, it looks up the callee in the function table. If found,
\ it replays the callee's body tokens with parameter substitution: callee
\ params are bound to the caller's arguments (register numbers).

\ ---- Function source registry ------------------------------------------------
\ We save each function's source token range so we can replay it.

16 constant MAX-FN-DEFS
8 constant MAX-FN-PARAMS
create fndef-names     MAX-FN-DEFS 32 * allot
create fndef-nlens     MAX-FN-DEFS cells allot
create fndef-src-start MAX-FN-DEFS cells allot  \ src-pos at start of body
create fndef-src-end   MAX-FN-DEFS cells allot  \ src-pos at end of body
create fndef-nparams   MAX-FN-DEFS cells allot
create fndef-nouts     MAX-FN-DEFS cells allot
variable n-fndefs  0 n-fndefs !

\ Parameter names for each function (up to MAX-FN-PARAMS params per fn)
create fndef-param-names  MAX-FN-DEFS MAX-FN-PARAMS * 32 * allot
create fndef-param-nlens  MAX-FN-DEFS MAX-FN-PARAMS * cells allot

\ Output names for each function (up to MAX-FN-PARAMS outputs per fn)
create fndef-out-names  MAX-FN-DEFS MAX-FN-PARAMS * 32 * allot
create fndef-out-nlens  MAX-FN-DEFS MAX-FN-PARAMS * cells allot

: fndef-name@ ( i -- addr u )
    dup 32 * fndef-names + swap cells fndef-nlens + @ ;

: fndef-find  ( addr u -- i | -1 )
    n-fndefs @ 0 ?do
        2dup i fndef-name@ li-tok= if 2drop i unloop exit then
    loop
    2drop -1 ;

: fndef-param-name@ ( fn-idx param-idx -- addr u )
    swap MAX-FN-PARAMS * + dup
    32 * fndef-param-names + swap cells fndef-param-nlens + @ ;

: fndef-out-name@ ( fn-idx out-idx -- addr u )
    swap MAX-FN-PARAMS * + dup
    32 * fndef-out-names + swap cells fndef-out-nlens + @ ;

: fndef-set-param-name ( addr u fn-idx param-idx -- )
    swap MAX-FN-PARAMS * + >r
    dup 32 > if drop 32 then
    dup r@ cells fndef-param-nlens + !
    r> 32 * fndef-param-names + swap move ;

: fndef-set-out-name ( addr u fn-idx out-idx -- )
    swap MAX-FN-PARAMS * + >r
    dup 32 > if drop 32 then
    dup r@ cells fndef-out-nlens + !
    r> 32 * fndef-out-names + swap move ;

: fndef-register  ( name-addr name-u nparams nouts src-start src-end -- )
    n-fndefs @ MAX-FN-DEFS < 0= if 2drop 2drop 2drop exit then
    n-fndefs @ >r
    r@ cells fndef-src-end + !
    r@ cells fndef-src-start + !
    r@ cells fndef-nouts + !
    r@ cells fndef-nparams + !
    dup 32 > if drop 32 then
    dup r@ cells fndef-nlens + !
    r@ 32 * fndef-names + swap move
    1 n-fndefs +!
    r> drop ;

\ ---- Inline expansion --------------------------------------------------------
\ To inline function F into the current context:
\ 1. Save current src-pos
\ 2. Set src-pos to fndef-src-start[F]
\ 3. For each callee param, bind to caller's register
\ 4. Parse the body (emitting PTX into the current buffer)
\ 5. Restore src-pos
\
\ This is the mechanism by which "fn scale x factor -> y" composed with
\ "fn add a b -> c" produces a single kernel body.

\ Inline depth counter (to prevent infinite recursion)
variable inline-depth  0 inline-depth !
4 constant MAX-INLINE-DEPTH
