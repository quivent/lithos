\ lithos.fs — Lithos compiler driver
\
\ Lithos: a language where everything is a math function.
\ Functions compose. Function calls inline (that is fusion).
\
\ Usage (from the forth-bootstrap CLI):
\   forth-bootstrap lithos.fs <input.li> --emit sass  -o out.cubin
\   forth-bootstrap lithos.fs <input.li> --emit cubin -o out.cubin
\   forth-bootstrap lithos.fs <input.li> --emit arm64 -o out
\
\ The driver reads argv, slurps the source file, evaluates it token by
\ token through the parser, and dumps the selected backend buffer to disk.

\ Load sub-modules (order matters).
s" /home/ubuntu/lithos/compiler/lexer.fs"     included
\ SASS + ARM64 emitters are optional; include only when needed.

s" /home/ubuntu/lithos/compiler/parser.fs"    included
s" /home/ubuntu/lithos/compiler/inline.fs"   included

\ ---- Argument parsing ------------------------------------------------------
variable arg-input-set  0 arg-input-set !
create arg-input-buf  256 allot  variable arg-input-len  0 arg-input-len !
create arg-output-buf 256 allot  variable arg-output-len 0 arg-output-len !
variable arg-emit  1 arg-emit !   \ 1=sass 2=arm64 3=cubin (default: sass)

: str-copy  ( src u dst -- )
  \ Copy u bytes from src to dst. Forth's MOVE is ( src dst u -- ).
  swap move ;

: arg-set-input  ( addr u -- )
  dup arg-input-len !  arg-input-buf str-copy
  1 arg-input-set ! ;

: arg-set-output ( addr u -- )
  dup arg-output-len !  arg-output-buf str-copy ;

variable pa-skip  0 pa-skip !
create pa-emit-kw 6 allot   s" --emit" pa-emit-kw swap move
create pa-o-kw    2 allot   s" -o"     pa-o-kw    swap move
create pa-sass-kw 4 allot   s" sass"   pa-sass-kw swap move
create pa-arm-kw   5 allot   s" arm64"  pa-arm-kw  swap move
create pa-cubin-kw 5 allot   s" cubin"  pa-cubin-kw swap move
create pa-ptx-kw   3 allot   s" ptx"    pa-ptx-kw  swap move

: pa-try-emit-val  ( addr u -- )
  2dup pa-sass-kw  4 li-tok= if 2drop 1 arg-emit ! exit then
  2dup pa-arm-kw   5 li-tok= if 2drop 2 arg-emit ! exit then
  2dup pa-cubin-kw 5 li-tok= if 2drop 3 arg-emit ! exit then
  2dup pa-ptx-kw   3 li-tok= if 2drop 4 arg-emit ! exit then
  2drop ;

\ Handle one argv[i]; i on stack coming in, consumed on exit.
: pa-maybe-emit  ( i addr u -- i 0 | 0 )
  2dup pa-emit-kw 6 li-tok= 0= if exit then
  2drop
  dup 1+ dup argc < if argv pa-try-emit-val else drop then
  drop 0 ;

: pa-maybe-o  ( i addr u -- i 0 | 0 )
  2dup pa-o-kw 2 li-tok= 0= if exit then
  2drop
  dup 1+ dup argc < if argv arg-set-output else drop then
  drop 0 ;

: pa-one  ( i -- )
  pa-skip @ 0> if drop -1 pa-skip +! exit then
  dup argv                             ( i addr u )
  pa-maybe-emit  dup 0= if drop 1 pa-skip ! exit then
  pa-maybe-o     dup 0= if drop 1 pa-skip ! exit then
  rot drop                             ( addr u )
  over c@ [char] - = if 2drop exit then
  arg-input-set @ if 2drop else arg-set-input then ;

: parse-args  ( -- )
  argc 2 ?do i pa-one loop ;

\ Drive the parser over the whole source (src-* words come from lexer.fs).
: lithos-compile  ( src-addr src-len -- )
  src-load
  begin
    src-token dup 0= if 2drop exit then
    li-body-word
  again ;

\ ---- Main ------------------------------------------------------------------
: usage  ( -- )
  s" usage: lithos.fs <input.li> --emit {sass|arm64|cubin} -o <output>" type cr ;

: lithos-main  ( -- )
  parse-args
  arg-input-set @ 0= if usage exit then
  arg-output-len @ 0= if usage exit then

  arg-input-buf arg-input-len @ slurp-file
  dup 0= if drop 2drop s" ERROR: cannot read input" type cr exit then
  arg-emit @ 1 = arg-emit @ 3 = or if
    s" /home/ubuntu/lithos/compiler/emit-sass.fs" included
  then
  lithos-compile

  arg-emit @ 1 = if
    \ SASS raw bytes
    arg-output-buf arg-output-len @ write-sass-raw
  else arg-emit @ 2 = if
    s" /home/ubuntu/lithos/compiler/emit-arm64.fs" included
    emit-host-stub
    arg-output-buf arg-output-len @ write-arm64-raw
  else arg-emit @ 3 = if
    \ cubin: SASS wrapped in a complete ELF64 cubin for cuModuleLoadData
    s" /home/ubuntu/lithos/compiler/cubin-wrap.fs" included
    arg-output-buf arg-output-len @ write-cubin
  else arg-emit @ 4 = if
    \ PTX text output
    arg-output-buf arg-output-len @
    577 open-file drop >r
    ptx-buf ptx-pos @ r@ write-file drop
    r> close-file drop
  then then then then

  s" lithos: wrote " type arg-output-buf arg-output-len @ type
  s"  (defs=" type li-defs @ .
  s" kernels=" type li-kernels @ .
  s" hosts=" type li-hosts @ . s" )" type cr ;

lithos-main
bye
