\ ls-tokenizer.fs — Tokenizer for the Lithos .ls stack language
\
\ The .ls format is line-oriented:
\   - Lines starting with # are comments
\   - Blank lines are skipped
\   - Non-indented lines start a definition (name + params)
\   - Indented lines are body tokens (one per line)
\
\ Produces a flat stream of token records for downstream consumption.

\ ---- Bootstrap compatibility words ------------------------------------------
\ forth-bootstrap provides <, >, =, 0<, 0>, 0= but not the composite forms.
: >=  ( a b -- flag )  < 0= ;
: <=  ( a b -- flag )  > 0= ;

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
 9 constant PRIM_OUTER    \ (retained for compatibility — compiler maps to PRIM_MUL_MAT)
10 constant PRIM_PROJECT
11 constant PRIM_MATVEC
12 constant PRIM_SUM      \ Σ — reduce-sum over a vector (full warp+cross-warp reduce)
13 constant LIT_INT
14 constant LIT_FLOAT
15 constant NAME_REF
16 constant DEF_START
17 constant DEF_END
18 constant OPERAND
19 constant TOK_REPEAT   \ NAME × N — inline a definition N times with grid-sync barriers
                         \ token.val      = N (repeat count)
                         \ token.name-addr/len = name of definition to repeat

\ ---- Dimensional operator tokens ---------------------------------------------
\ The number of repeated operator symbols encodes the loop nesting depth.
\ Single-symbol forms (scalar) use the existing PRIM_* constants above.
20 constant PRIM_MUL_VEC  \ **   — element-wise multiply over a vector (one stride loop)
21 constant PRIM_MUL_MAT  \ ***  — outer product (nested stride loops; replaces PRIM_OUTER)
22 constant PRIM_MUL_TEN  \ **** — tensor outer product (three nested stride loops)
23 constant PRIM_ADD_VEC  \ ++   — element-wise add over a vector
24 constant PRIM_ADD_MAT  \ +++  — outer add (nested stride loops)
25 constant PRIM_ADD_TEN  \ ++++ — tensor outer add
26 constant PRIM_SUB_VEC  \ --   — element-wise subtract over a vector
27 constant PRIM_SUB_MAT  \ ---  — outer subtract (nested stride loops)
28 constant PRIM_DIV_VEC  \ //   — element-wise divide over a vector
29 constant PRIM_DIV_MAT  \ ///  — outer divide (nested stride loops)

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

\ name-intern ( src u -- dst u )
\ Copies src string into op-pool so the name survives line-buf rewrites.
variable ni-src  variable ni-u  variable ni-dst
: name-intern  ( src u -- dst u )
  ni-u !  ni-src !
  op-pool  op-pool-pos @  +  ni-dst !          \ dst = op-pool + pos
  ni-src @  ni-dst @  ni-u @  move             \ copy bytes
  ni-u @  op-pool-pos  +!                      \ advance pool cursor
  ni-dst @  ni-u @ ;                           \ return ( dst u )

: tok-with-name  ( type addr u -- )  name-intern  0 -rot 0 tok-emit-raw ;
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

create p-mul     1 allot  char * p-mul c!
create p-add     1 allot  char + p-add c!
create p-sub     1 allot  char - p-sub c!
create p-div     1 allot  char / p-div c!
create p-exp     3 allot  s" exp"     p-exp     swap move
create p-log     3 allot  s" log"     p-log     swap move
create p-project 7 allot  s" project" p-project swap move
create p-matvec  6 allot  s" matvec"  p-matvec  swap move

\ UTF-8 Σ: 0xCE 0xA3 (2 bytes for U+03A3)
create p-sum 2 allot
206 p-sum c!  163 p-sum 1+ c!

\ UTF-8 sqrt: 0xE2 0x88 0x9A (3 bytes for U+221A)
create p-sqrt 3 allot
226 p-sqrt c!  136 p-sqrt 1+ c!  154 p-sqrt 2 + c!

\ 1/ (2 bytes)
create p-rcp 2 allot  char 1 p-rcp c!  char / p-rcp 1+ c!

\ 1/sqrt: 1/ + sqrt = "1/" + 3-byte-sqrt = 5 bytes
create p-rsqrt 5 allot
char 1 p-rsqrt c!  char / p-rsqrt 1+ c!
226 p-rsqrt 2 + c!  136 p-rsqrt 3 + c!  154 p-rsqrt 4 + c!

\ UTF-8 multiply sign: U+00D7 = C3 97 (2 bytes)
create p-times 2 allot
195 p-times c!  151 p-times 1+ c!

\ is-times? ( addr u -- flag )  True when the string is exactly the × glyph.
: is-times?  ( addr u -- flag )  p-times 2 ls-str= ;

\ count-leading ( addr u ch -- n )
\ Count how many leading bytes in the string equal ch.
variable cl-addr  variable cl-u  variable cl-ch
: count-leading  ( addr u ch -- n )
  cl-ch !  cl-u !  cl-addr !
  0                              \ n = 0
  begin
    dup cl-u @ <                 \ n < length
    over cl-addr @ + c@ cl-ch @ = and   \ and char[n] = ch
  while
    1+                           \ n++
  repeat ;

\ match-dim-op ( addr u first-ch scalar-tok vec-tok mat-tok ten-tok -- type true | false )
\ If every byte in addr/u equals first-ch, emit the token for the run length.
\   length 1 -> scalar-tok
\   length 2 -> vec-tok
\   length 3 -> mat-tok
\   length 4 -> ten-tok
\   else     -> false
variable mdo-s  variable mdo-v  variable mdo-m  variable mdo-t
: match-dim-op  ( addr u ch s v m t -- type true | false )
  mdo-t !  mdo-m !  mdo-v !  mdo-s !   \ save tok types; stack: addr u ch
  >r 2dup r> count-leading              \ ( addr u n )
  over = 0= if 2drop 0 exit then       \ not all same char -> no match
  drop                                  \ drop addr; stack: ( n )
  dup 1 = if drop mdo-s @ -1 exit then
  dup 2 = if drop mdo-v @ -1 exit then
  dup 3 = if drop mdo-m @ -1 exit then
      4 = if      mdo-t @ -1 exit then
  0 ;

: match-prim  ( addr u -- type true | false )
  \ Named keywords first (before single-char tests)
  2dup p-exp     3 ls-str= if 2drop PRIM_EXP     -1 exit then
  2dup p-log     3 ls-str= if 2drop PRIM_LOG     -1 exit then
  2dup p-rsqrt   5 ls-str= if 2drop PRIM_RSQRT   -1 exit then
  2dup p-rcp     2 ls-str= if 2drop PRIM_RCP     -1 exit then
  2dup p-sqrt    3 ls-str= if 2drop PRIM_SQRT    -1 exit then
  2dup p-project 7 ls-str= if 2drop PRIM_PROJECT -1 exit then
  2dup p-matvec  6 ls-str= if 2drop PRIM_MATVEC  -1 exit then
  2dup p-sum     2 ls-str= if 2drop PRIM_SUM     -1 exit then
  2dup p-times   2 ls-str= if 2drop PRIM_MUL     -1 exit then
  \ Dimensional symbol operators: repeated *, +, -, /
  2dup [char] * PRIM_MUL PRIM_MUL_VEC PRIM_MUL_MAT PRIM_MUL_TEN match-dim-op if exit then
  2dup [char] + PRIM_ADD PRIM_ADD_VEC PRIM_ADD_MAT PRIM_ADD_TEN match-dim-op if exit then
  2dup [char] - PRIM_SUB PRIM_SUB_VEC PRIM_SUB_MAT 0           match-dim-op if exit then
  2dup [char] / PRIM_DIV PRIM_DIV_VEC PRIM_DIV_MAT 0           match-dim-op if exit then
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

\ ---- parse-body-line variables (avoid deep stack juggling) ------------------
variable pbl-w1a   \ first word: address in line-buf
variable pbl-w1u   \ first word: byte length
variable pbl-w2a   \ second word: address
variable pbl-w2u   \ second word: byte length
variable pbl-w3a   \ third word: address
variable pbl-w3u   \ third word: byte length
variable pbl-nw    \ number of words found (1, 2, or 3+)

\ pbl-extract-word ( off -- off' )
\ Starting at byte offset 'off' in line-buf, skip any leading whitespace,
\ then read one whitespace-delimited word and write it into the pbl-wNa/u
\ pair whose index equals the current value of pbl-nw (1, 2, or 3).
\ Returns the offset after the word.
\ Caller increments pbl-nw after calling.
variable pbl-ws   variable pbl-we

: pbl-extract-word  ( off -- off' )
  skip-ws                               \ ( s )
  dup line-len @ >= if exit then        \ at end of line — no word
  dup pbl-ws !                          \ save start
  dup find-ws pbl-we !                  \ save end
  pbl-ws @ line-buf + pbl-nw @ 1 = if pbl-w1a ! else
                       pbl-nw @ 2 = if pbl-w2a ! else
                                        pbl-w3a ! then then
  pbl-we @ pbl-ws @ -  pbl-nw @ 1 = if pbl-w1u ! else
                        pbl-nw @ 2 = if pbl-w2u ! else
                                        pbl-w3u ! then then
  1 pbl-nw +!
  pbl-we @ ;

\ parse-body-line  ( -- )
\ Parses one indented body line.  Handles three patterns:
\   TOKEN                      -> NAME_REF, primitive, or literal (1 word)
\   PRIM operand               -> primitive with operand (2 words)
\   NAME × N                   -> TOK_REPEAT  inline N times (3 words)

: parse-body-line  ( -- )
  \ ---- guard: blank/all-whitespace line ----
  line-indent dup line-len @ >= if drop exit then
  \ ---- initialise word slots ----
  1 pbl-nw !
  0 pbl-w1a !  0 pbl-w1u !
  0 pbl-w2a !  0 pbl-w2u !
  0 pbl-w3a !  0 pbl-w3u !
  \ ---- extract up to three words ----
  \ Each pbl-extract-word consumes the incoming offset and returns the
  \ offset after the word (or returns unchanged if at end-of-line).
  \ The final returned offset is dropped after extraction is complete.
  pbl-extract-word     \ word 1 (pbl-nw was 1, now 2); stack: ( off1 )
  pbl-nw @ 2 = if     \ only if word 1 was found
    pbl-extract-word   \ word 2 (pbl-nw was 2, now 3); stack: ( off2 )
  then
  pbl-nw @ 3 = if     \ only if word 2 was found
    pbl-extract-word   \ word 3 (pbl-nw was 3, now 4); stack: ( off3 )
  then
  drop                             \ discard the last offset (not needed further)
  \ pbl-nw is now 1+number-of-words-found (1 means 0 found, 2 means 1, etc.)
  \ Adjust: actual word count = pbl-nw @ 1-
  pbl-w1u @ 0= if exit then             \ no first word at all — skip

  \ ---- three-word case: NAME × N ----
  pbl-nw @ 1- 3 >= if
    pbl-w2a @  pbl-w2u @  is-times? if
      \ Emit TOK_REPEAT  val=N  name=w1
      \ tok-emit-raw ( type val name-addr name-len op-off -- )
      TOK_REPEAT
      pbl-w3a @  pbl-w3u @  ls-parse-int   \ N
      pbl-w1a @  pbl-w1u @                  \ name
      0 tok-emit-raw
      exit
    then
    \ Three words, second is not ×: fall through to single-word emit of w1.
  then

  \ ---- two-word case: PRIM operand ----
  pbl-nw @ 1- 2 = if
    pbl-w1a @  pbl-w1u @  match-prim if   \ ( -- prim-type )
      \ Emit primitive token — tok-emit-raw ( type val name-addr name-len op-off -- )
      pbl-w2a @  pbl-w2u @  op-store      \ ( prim-type op-off )
      swap                                 \ ( op-off prim-type )
      0                                    \ ( op-off prim-type 0=val )
      pbl-w2a @  pbl-w2u @                 \ ( op-off prim-type 0 name-addr name-u )
      \ tok-emit-raw wants ( type val name-addr name-len op-off )
      \ rearrange: ( op-off prim-type 0 w2a w2u ) -> ( prim-type 0 w2a w2u op-off )
      4 roll                               \ ( prim-type 0 w2a w2u op-off )
      tok-emit-raw
      exit
    then
    \ w1 not a primitive but has a second word: emit w1 as NAME_REF, ignore w2.
    pbl-w1a @  pbl-w1u @  NAME_REF -rot tok-with-name
    exit
  then

  \ ---- single-word case ----
  pbl-w1a @  pbl-w1u @
  2dup match-prim if
    2drop tok-simple
  else
    2dup is-number? if
      2dup has-dot? if
        2drop LIT_FLOAT 0 tok-with-val
      else
        ls-parse-int LIT_INT swap tok-with-val
      then
    else
      NAME_REF -rot tok-with-name
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
  dup PRIM_RSQRT   = if drop ." PRIM_RSQRT"   exit then
  dup PRIM_PROJECT = if drop ." PRIM_PROJECT" exit then
  dup PRIM_MATVEC  = if drop ." PRIM_MATVEC"  exit then
  dup PRIM_SUM     = if drop ." PRIM_SUM"     exit then
  dup LIT_INT      = if drop ." LIT_INT"      exit then
  dup LIT_FLOAT    = if drop ." LIT_FLOAT"    exit then
  dup NAME_REF     = if drop ." NAME_REF"     exit then
  dup DEF_START    = if drop ." DEF_START"    exit then
  dup DEF_END      = if drop ." DEF_END"      exit then
  dup OPERAND      = if drop ." OPERAND"      exit then
  dup TOK_REPEAT   = if drop ." TOK_REPEAT"   exit then
  dup PRIM_MUL_VEC = if drop ." PRIM_MUL_VEC" exit then
  dup PRIM_MUL_MAT = if drop ." PRIM_MUL_MAT" exit then
  dup PRIM_MUL_TEN = if drop ." PRIM_MUL_TEN" exit then
  dup PRIM_ADD_VEC = if drop ." PRIM_ADD_VEC" exit then
  dup PRIM_ADD_MAT = if drop ." PRIM_ADD_MAT" exit then
  dup PRIM_ADD_TEN = if drop ." PRIM_ADD_TEN" exit then
  dup PRIM_SUB_VEC = if drop ." PRIM_SUB_VEC" exit then
  dup PRIM_SUB_MAT = if drop ." PRIM_SUB_MAT" exit then
  dup PRIM_DIV_VEC = if drop ." PRIM_DIV_VEC" exit then
  dup PRIM_DIV_MAT = if drop ." PRIM_DIV_MAT" exit then
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
