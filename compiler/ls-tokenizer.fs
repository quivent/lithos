\ ls-tokenizer.fs — Tokenizer for the Lithos .ls stack language
\
\ The .ls format is line-oriented:
\   - Lines starting with # are comments
\   - Blank lines are skipped
\   - Non-indented lines start a definition (name + params)
\   - Indented lines are body tokens (one per line)
\
\ Produces a flat stream of token records for downstream consumption.

\ ---- Token type constants ----------------------------------------------------

 0 constant PRIM_MUL
 1 constant PRIM_ADD
 2 constant PRIM_SUB
 3 constant PRIM_DIV
 4 constant PRIM_EXP
 5 constant PRIM_LOG
 6 constant PRIM_SQRT
 7 constant PRIM_RCP
 8 constant PRIM_RSQRT
 9 constant PRIM_OUTER
10 constant PRIM_PROJECT
11 constant PRIM_MATVEC
12 constant LIT_INT
13 constant LIT_FLOAT
14 constant NAME_REF
15 constant DEF_START
16 constant DEF_END
17 constant OPERAND

\ ---- Token buffer ------------------------------------------------------------
\ Each token record: 5 cells = type, value, name-addr, name-len, operand-offset
\ Operand stored in a side string buffer.

4096 constant MAX-TOKENS
create tok-buf MAX-TOKENS 5 * cells allot
variable tok-count   0 tok-count !
variable tok-cursor  0 tok-cursor !

\ Operand string pool (operand texts packed sequentially)
4096 constant OP-POOL-SZ
create op-pool OP-POOL-SZ allot
variable op-pool-pos  0 op-pool-pos !

: tok-record  ( i -- addr )  5 * cells tok-buf + ;
: tok-emit-raw  ( type val name-addr name-len op-off -- )
  tok-count @ MAX-TOKENS < 0= if 2drop 2drop drop exit then
  tok-count @ tok-record
  >r  r@ 4 cells + !  r@ 3 cells + !  r@ 2 cells + !
       r@ 1 cells + !  r> !
  1 tok-count +! ;

: tok-simple  ( type -- )  0 0 0 0 tok-emit-raw ;
: tok-with-name  ( type addr u -- )  0 -rot 0 tok-emit-raw ;
: tok-with-val   ( type n -- )  0 0 0 tok-emit-raw ;

\ Store operand string in pool, return offset
: op-store  ( addr u -- off )
  op-pool-pos @ >r
  dup op-pool-pos @ + OP-POOL-SZ < 0= if 2drop r> exit then
  op-pool r@ + swap dup op-pool-pos +! move
  r> ;

\ ---- Line buffer and source management ---------------------------------------

create line-buf 1024 allot
variable line-len   0 line-len !

variable ls-src-addr
variable ls-src-len
variable ls-src-pos  0 ls-src-pos !

: ls-eof?  ( -- flag )  ls-src-pos @ ls-src-len @ >= ;

\ Read one line into line-buf. Returns false if EOF.
: ls-read-line  ( -- flag )
  ls-eof? if 0 exit then
  0 line-len !
  begin
    ls-eof? 0= while
    ls-src-addr @ ls-src-pos @ + c@
    1 ls-src-pos +!
    dup 10 = if drop -1 exit then
    line-buf line-len @ + c!
    1 line-len +!
  repeat
  line-len @ 0> ;

\ ---- String utilities --------------------------------------------------------

\ Compare two strings (reuse li-tok= pattern from lexer.fs)
: ls-str=  ( a1 u1 a2 u2 -- flag )
  >r >r >r >r 0 r> r> r> r> compare 0= ;

\ Check if character is a digit or minus-then-digit (for literal detection)
: is-digit?  ( c -- flag )  dup [char] 0 >= swap [char] 9 <= and ;
: is-numeric-start?  ( c -- flag )
  dup is-digit? if drop -1 exit then
  [char] - = ;

\ Check if a string looks like a number (integer or float)
: is-number?  ( addr u -- flag )
  dup 0= if 2drop 0 exit then
  over c@ is-numeric-start? 0= if 2drop 0 exit then
  \ Scan rest for digits or '.'
  1 ?do
    dup i + c@ dup is-digit? swap [char] . = or 0= if drop 0 unloop exit then
  loop
  drop -1 ;

: has-dot?  ( addr u -- flag )
  0 ?do dup i + c@ [char] . = if drop -1 unloop exit then loop
  drop 0 ;

\ Parse integer from string (simple decimal)
: ls-parse-int  ( addr u -- n )
  0 0 2swap >number 2drop drop ;

\ ---- Primitive matching ------------------------------------------------------
\ Match first word of body line against known primitives.
\ UTF-8 aware: sqrt is U+221A = E2 88 9A (3 bytes).

create p-mul     1 allot  [char] * p-mul c!
create p-add     1 allot  [char] + p-add c!
create p-sub     1 allot  [char] - p-sub c!
create p-div     1 allot  [char] / p-div c!
create p-exp     3 allot  s" exp"     p-exp     swap move
create p-log     3 allot  s" log"     p-log     swap move
create p-outer   5 allot  s" outer"   p-outer   swap move
create p-project 7 allot  s" project" p-project swap move
create p-matvec  6 allot  s" matvec"  p-matvec  swap move

\ UTF-8 sqrt: 0xE2 0x88 0x9A (3 bytes for U+221A)
create p-sqrt 3 allot
226 p-sqrt c!  136 p-sqrt 1+ c!  154 p-sqrt 2 + c!

\ 1/ (2 bytes)
create p-rcp 2 allot  [char] 1 p-rcp c!  [char] / p-rcp 1+ c!

\ 1/sqrt: 1/ + sqrt = "1/" + 3-byte-sqrt = 5 bytes
create p-rsqrt 5 allot
[char] 1 p-rsqrt c!  [char] / p-rsqrt 1+ c!
226 p-rsqrt 2 + c!  136 p-rsqrt 3 + c!  154 p-rsqrt 4 + c!

\ UTF-8 multiply sign: U+00D7 = C3 97 (2 bytes)
create p-times 2 allot
195 p-times c!  151 p-times 1+ c!

: match-prim  ( addr u -- type true | false )
  2dup p-mul     1 ls-str= if 2drop PRIM_MUL   -1 exit then
  2dup p-times   2 ls-str= if 2drop PRIM_MUL   -1 exit then
  2dup p-add     1 ls-str= if 2drop PRIM_ADD   -1 exit then
  2dup p-sub     1 ls-str= if 2drop PRIM_SUB   -1 exit then
  2dup p-div     1 ls-str= if 2drop PRIM_DIV   -1 exit then
  2dup p-exp     3 ls-str= if 2drop PRIM_EXP   -1 exit then
  2dup p-log     3 ls-str= if 2drop PRIM_LOG   -1 exit then
  2dup p-rsqrt   5 ls-str= if 2drop PRIM_RSQRT -1 exit then
  2dup p-rcp     2 ls-str= if 2drop PRIM_RCP   -1 exit then
  2dup p-sqrt    3 ls-str= if 2drop PRIM_SQRT  -1 exit then
  2dup p-outer   5 ls-str= if 2drop PRIM_OUTER -1 exit then
  2dup p-project 7 ls-str= if 2drop PRIM_PROJECT -1 exit then
  2dup p-matvec  6 ls-str= if 2drop PRIM_MATVEC  -1 exit then
  2drop 0 ;

\ ---- Line parsing ------------------------------------------------------------

\ Skip leading whitespace, return offset to first non-space char
: line-indent  ( -- offset )
  0
  begin dup line-len @ < while
    line-buf over + c@ 32 > if exit then
    1+
  repeat ;

\ Find first whitespace after position 'off', return end offset
: find-ws  ( off -- end )
  begin dup line-len @ < while
    line-buf over + c@ 33 < if exit then
    1+
  repeat ;

\ Skip whitespace starting at 'off'
: skip-ws  ( off -- off' )
  begin dup line-len @ < while
    line-buf over + c@ 32 > if exit then
    1+
  repeat ;

\ Parse body line (indented): extract first word, optional second word
: parse-body-line  ( -- )
  line-indent dup line-len @ >= if drop exit then
  dup find-ws  \ stack: start end
  2dup = if 2drop exit then
  \ first word: addr = line-buf+start, len = end-start
  over line-buf + swap over -   \ addr1 u1  (first word)
  \ Check for second word (operand)
  2dup 2>r
  find-ws skip-ws              \ wait -- need to recalculate from line
  2r> 2drop
  \ Redo: recompute from scratch
  drop  \ clean up

  \ Simpler approach: get first word boundaries
  line-indent
  dup find-ws    \ s1 e1
  over line-buf + over 2 pick -   \ s1 e1 addr1 u1
  2>r                              \ save first word
  skip-ws                          \ e1 -> s2
  dup line-len @ < if              \ there's a second word
    dup find-ws over line-buf + over 2 pick -  \ s2 e2 addr2 u2
    2>r
    2drop                          \ drop s2 e2
    \ We have first word (2r@ from outer) and operand
    2r> 2r>                        \ op-addr op-u word-addr word-u
    2swap                          \ word-addr word-u op-addr op-u
    \ Try to match first word as primitive
    2swap match-prim if            \ prim-type ; operand on stack
      \ Emit primitive with operand
      >r                           \ save type
      dup op-store                 \ op-addr op-u -> off
      -rot                         \ off op-addr op-u
      r> 0 2swap tok-emit-raw     \ type=prim val=0 name-addr=op-addr name-len=op-u op-off=off
      drop                         \ drop the extra off -- wait, rethink
    else                           \ not a prim, it's name_ref + operand... just emit name
      2drop                        \ drop operand, just emit as name_ref
      NAME_REF -rot tok-with-name
    then
  else
    drop                           \ drop s2
    2r>                            \ retrieve first word
    \ Single word: primitive, literal, or name-ref
    2dup match-prim if
      2drop tok-simple
    else
      2dup is-number? if
        2dup has-dot? if
          2drop LIT_FLOAT 0 tok-with-val  \ float (no real parse)
        else
          ls-parse-int LIT_INT swap tok-with-val
        then
      else
        NAME_REF -rot tok-with-name
      then
    then
  then ;

\ Parse definition header: name at column 0, then params
: parse-def-line  ( -- )
  \ Count params: each whitespace-separated word after the first
  0 find-ws   \ end of name
  0 line-buf + over    \ addr u (def name)
  DEF_START -rot tok-with-name

  \ Count and skip params
  begin
    skip-ws dup line-len @ < while
    \ This is a param — emit as NAME_REF so downstream knows param names
    dup find-ws over line-buf + over 2 pick -
    NAME_REF -rot tok-with-name
    nip  \ drop start, keep end
  repeat
  drop ;

\ ---- Main tokenizer ----------------------------------------------------------

variable in-def  0 in-def !

: ls-process-line  ( -- )
  line-len @ 0= if exit then                  \ blank line
  line-buf c@ [char] # = if exit then          \ comment
  line-indent 0= if
    \ Definition header
    in-def @ if DEF_END tok-simple then
    parse-def-line
    -1 in-def !
  else
    \ Body line
    parse-body-line
  then ;

: ls-tokenize  ( addr u -- )
  \ Reset state
  0 tok-count !  0 tok-cursor !
  0 op-pool-pos !  0 in-def !
  ls-src-len !  ls-src-addr !  0 ls-src-pos !
  begin ls-read-line while
    ls-process-line
  repeat
  \ Close last definition if open
  in-def @ if DEF_END tok-simple  0 in-def ! then ;

\ ---- Iterator API ------------------------------------------------------------

: ls-tokens-done?  ( -- flag )  tok-cursor @ tok-count @ >= ;

: ls-cur  ( -- addr )  tok-cursor @ tok-record ;

: ls-next-token  ( -- type )
  ls-tokens-done? if -1 exit then
  ls-cur @
  1 tok-cursor +! ;

: ls-token-val  ( -- n )
  tok-cursor @ 1- tok-record 1 cells + @ ;

: ls-token-name  ( -- addr u )
  tok-cursor @ 1- tok-record
  dup 2 cells + @  swap 3 cells + @ ;

: ls-token-operand  ( -- addr u )
  tok-cursor @ 1- tok-record
  dup 2 cells + @  swap 3 cells + @ ;

: ls-reset-cursor  ( -- )  0 tok-cursor ! ;

\ ---- Type name printing (debug) ---------------------------------------------

: .tok-type  ( type -- )
  dup PRIM_MUL     = if drop ." PRIM_MUL"     exit then
  dup PRIM_ADD     = if drop ." PRIM_ADD"     exit then
  dup PRIM_SUB     = if drop ." PRIM_SUB"     exit then
  dup PRIM_DIV     = if drop ." PRIM_DIV"     exit then
  dup PRIM_EXP     = if drop ." PRIM_EXP"     exit then
  dup PRIM_LOG     = if drop ." PRIM_LOG"     exit then
  dup PRIM_SQRT    = if drop ." PRIM_SQRT"    exit then
  dup PRIM_RCP     = if drop ." PRIM_RCP"     exit then
  dup PRIM_RSQRT   = if drop ." PRIM_RSQRT"  exit then
  dup PRIM_OUTER   = if drop ." PRIM_OUTER"   exit then
  dup PRIM_PROJECT = if drop ." PRIM_PROJECT" exit then
  dup PRIM_MATVEC  = if drop ." PRIM_MATVEC"  exit then
  dup LIT_INT      = if drop ." LIT_INT"      exit then
  dup LIT_FLOAT    = if drop ." LIT_FLOAT"    exit then
  dup NAME_REF     = if drop ." NAME_REF"     exit then
  dup DEF_START    = if drop ." DEF_START"    exit then
  dup DEF_END      = if drop ." DEF_END"      exit then
  dup OPERAND      = if drop ." OPERAND"      exit then
  . ;

\ ---- Test --------------------------------------------------------------------

\ Build test string with embedded newlines
create test-src 128 allot
variable test-len  0 test-len !

: t+c  ( c -- )  test-src test-len @ + c!  1 test-len +! ;
: t+s  ( addr u -- )
  0 ?do dup i + c@ t+c loop drop ;

: build-test-src  ( -- addr u )
  0 test-len !
  s" test_add a b" t+s 10 t+c
  s"   a"          t+s 10 t+c
  s"   + b"        t+s 10 t+c
  test-src test-len @ ;

: ls-test
  build-test-src ls-tokenize
  ls-reset-cursor
  cr ." === ls-tokenizer test ===" cr
  begin ls-tokens-done? 0= while
    ls-next-token
    dup .tok-type
    dup DEF_START = over NAME_REF = or if
      ."  " ls-token-name type
    then
    dup PRIM_ADD = over PRIM_MUL = or over PRIM_SUB = or over PRIM_DIV = or if
      ls-token-name dup 0> if
        ."  operand=" type
      else
        2drop
      then
    then
    drop
    cr
  repeat
  ." === done ===" cr ;

ls-test
