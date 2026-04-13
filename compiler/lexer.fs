\ lexer.fs — Lithos tokenizer (thin layer over Forth's native parser)
\
\ Lithos source is whitespace-delimited, just like Forth.
\ This file provides source-buffer management (src-load, src-token) and
\ the string comparison word li-tok= used by the parser dispatch table.

\ ---- Source buffer tokenizer -----------------------------------------------
\ We maintain our own source pointer so we can read tokens from a slurped
\ .li file without disturbing the host Forth's input buffer.

variable src-addr
variable src-len
variable src-pos   0 src-pos !

: src-load  ( addr u -- )  src-len ! src-addr !  0 src-pos ! ;
: src-eof?  ( -- flag )    src-pos @ src-len @ < 0= ;
: src-peek  ( -- c )       src-addr @ src-pos @ + c@ ;

: src-skip-ws  ( -- )
  begin src-eof? 0= while
    src-peek dup 33 < if drop 1 src-pos +!
    else drop exit then
  repeat ;

: src-skip-line  ( -- )
  begin src-eof? 0= while
    src-peek 1 src-pos +!
    10 = if exit then
  repeat ;

\ src-token ( -- addr u ; 0 0 at EOF )
: src-token
  begin
    src-skip-ws
    src-eof? if 0 0 exit then
    src-peek [char] \ = if src-skip-line 0 else -1 then
  until
  src-addr @ src-pos @ +       \ addr
  0                            \ u (length accumulator)
  begin
    src-eof? if exit then
    src-peek 33 < if exit then
    1+ 1 src-pos +!
  again ;

\ ---- String comparison (with bootstrap COMPARE workaround) -----------------
\ The bootstrap's COMPARE primitive has an off-by-one and consumes five
\ stack cells instead of four (net -4 rather than -3). We compensate by
\ pushing a dummy cell below the four arguments before calling COMPARE.

: compare*  ( a1 u1 a2 u2 -- n )
  >r >r >r >r 0 r> r> r> r> compare ;

: li-tok=  ( addr1 u1 addr2 u2 -- flag )
  compare* 0= ;

\ Print a token (for debug / error messages).
: li-.tok  ( addr u -- )  type ;
