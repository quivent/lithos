\ parser.fs — Lithos math-function parser
\
\ A Lithos program is a collection of math function definitions.
\ Syntax:
\
\   fn NAME param1 param2 ... -> output1 output2 ...
\       each VAR
\           output [ VAR ] = param1 [ VAR ] + param2 [ VAR ]
\
\ "fn" introduces a function.  Parameters before "->" are inputs (pointers).
\ Names after "->" are outputs (pointers).  "each VAR" introduces parallel
\ iteration (maps to global thread index).  Infix expressions use
\ whitespace-separated operators: + - * /
\
\ Token dispatch: the driver (lithos.fs) calls li-body-word with each
\ whitespace-delimited token.  When we see "fn" we consume the entire
\ function definition (signature + body) from the source buffer.

variable li-mode     0 li-mode !
variable li-defs     0 li-defs !
variable li-kernels  0 li-kernels !
variable li-hosts    0 li-hosts !

\ ---- Current function name ---------------------------------------------------

create li-name-buf 64 allot
variable li-name-len  0 li-name-len !
: li-set-name  ( addr u -- )  dup li-name-len ! li-name-buf swap move ;
: li-name$  ( -- addr u )  li-name-buf li-name-len @ ;

\ ---- Symbol table: up to 32 variables (params, outputs, locals, each-var) ----
\ kind: 0=input-ptr 1=output-ptr 2=local-f32 3=each-var 4=local-u32

32 constant MAX-SYMS
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
: freg+  ( -- n )  next-freg @  1 next-freg +! ;

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

\ ---- PTX emission helpers ----------------------------------------------------

: ptx-indent  s"     " ptx+ ;
: ptx-freg  ( n -- )   s" %f"  ptx+  ptx-num ptx+ ;
: ptx-r32   ( n -- )   s" %r"  ptx+  ptx-num ptx+ ;
: ptx-r64   ( n -- )   s" %rd" ptx+  ptx-num ptx+ ;

\ ---- PTX header for a function -----------------------------------------------

variable header-emitted  0 header-emitted !

: ptx-emit-header  ( -- )
    header-emitted @ if exit then  1 header-emitted !
    ptx-header
    s" .visible .entry " ptx+  li-name$ ptx+  s" (" ptx+ ptx-nl
    \ All params are .u64 pointers
    n-inputs @ n-outputs @ + 0 ?do
        ptx-indent s" .param .u64 " ptx+
        i sym-name@ ptx+
        i n-inputs @ n-outputs @ + 1- < if s" ," ptx+ then
        ptx-nl
    loop
    s" )" ptx+ ptx-nl
    s" {" ptx+ ptx-nl
    \ Register pools
    ptx-indent s" .reg .pred %p<8>;"   ptx+ ptx-nl
    ptx-indent s" .reg .b32  %r<64>;"  ptx+ ptx-nl
    ptx-indent s" .reg .b64  %rd<32>;" ptx+ ptx-nl
    ptx-indent s" .reg .f32  %f<64>;"  ptx+ ptx-nl
    ptx-nl
    \ Load params into rd registers
    n-inputs @ n-outputs @ + 0 ?do
        ptx-indent s" ld.param.u64 " ptx+  i sym-reg@ ptx-r64
        s" , [" ptx+  i sym-name@ ptx+  s" ];" ptx+ ptx-nl
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
\ Uses %rd22/%rd23 as scratch for address math.

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

\ ---- Expression parser (returns freg holding result) -------------------------
\ Uses save/restore of src-pos for lookahead.
\
\ Grammar:
\   expr  = term { ("+" | "-") term }
\   term  = atom { "*" atom }
\   atom  = NAME "[" VAR "]"     (indexed load)
\         | NAME                 (local var reference — its freg)

\ Forward declaration for mutual recursion.
variable 'parse-expr

: peek-tok  ( -- addr u )
    src-pos @ >r  src-token  r> swap >r src-pos !  r> ;

: parse-atom  ( -- freg )
    src-token dup 0= if 2drop -1 exit then
    \ Look up identifier
    2dup sym-find dup -1 = if
        \ Unknown — try as number literal? For now emit zero.
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
        sym-reg@
        then then exit
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
    sym-reg@
    then then ;

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

\ ---- Statement parser --------------------------------------------------------
\ Reads one statement:
\   "each" VAR           -> emit thread index, register VAR
\   NAME "[" VAR "]" "=" expr   -> parse expr, emit indexed store
\   NAME "=" expr                -> parse expr, register local

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
            \ NAME [ i ] = expr  where NAME is unknown — shouldn't happen
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
\ Called when "fn" is seen. Consumes the rest of the function from src-*.

: parse-fn-params  ( -- )
    \ Read input params until "->" or EOF
    begin
        src-token dup 0= if 2drop exit then
        2dup k-arrow 2 li-tok= if 2drop exit then
        0 n-inputs @ sym-add drop
        1 n-inputs +!
    again ;

: parse-fn-outputs  ( -- )
    \ Read output names until body keyword or EOF
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> src-pos ! exit then
        2dup k-each 4 li-tok= if 2drop r> src-pos ! exit then
        2dup k-fn   2 li-tok= if 2drop r> src-pos ! exit then
        r> drop
        1 n-inputs @ n-outputs @ + sym-add drop
        1 n-outputs +!
    again ;

: parse-fn-body  ( -- )
    begin
        src-pos @ >r
        src-token dup 0= if 2drop r> drop exit then
        parse-stmt
        0= if r> src-pos ! exit then
        r> drop
    again ;

: parse-fn  ( -- )
    sym-reset
    0 header-emitted !
    0 next-freg !
    ptx-reset
    regs-reset
    1 li-defs +!  1 li-kernels +!
    src-token dup 0= if 2drop exit then
    li-set-name
    parse-fn-params
    parse-fn-outputs
    ptx-emit-header
    parse-fn-body
    ptx-emit-footer ;

\ ---- Top-level token handler -------------------------------------------------
\ Called by lithos-compile for each token.

: li-body-word  ( addr u -- )
    2dup k-fn 2 li-tok= if
        2drop
        parse-fn
        exit
    then
    \ Legacy keywords for backward compat
    \ (skip unknown top-level tokens)
    2drop ;
