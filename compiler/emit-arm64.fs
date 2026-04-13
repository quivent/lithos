\ emit-arm64.fs — ARM64 code emitter for Lithos host functions
\
\ The existing /home/ubuntu/sixth/eighth/src/asm.fs provides a full ARM64
\ instruction encoder (mov, add, ldr, str, bl, ret, branch, ...). We reuse
\ it verbatim. This file adds a small code buffer and a writer.

\ ---- Code buffer -----------------------------------------------------------
65536 constant CODE-SIZE
create code-buf CODE-SIZE allot
variable code-pos  0 code-pos !

: code-reset  0 code-pos ! ;

\ Append a little-endian 32-bit instruction word.
: a,  ( u32 -- )
  code-buf code-pos @ +
  >r
  dup r@ c!
  8 rshift dup r@ 1+ c!
  8 rshift dup r@ 2 + c!
  8 rshift     r> 3 + c!
  4 code-pos +! ;

\ ---- Minimal host-word emission --------------------------------------------
\ For the first pass we emit just a canonical "return 0" stub per host word.
\ A real backend would parse the body and emit instructions; that hookup
\ lives where parser.fs currently routes to a // unknown comment when mode=2.
\
\ ARM64 encodings used:
\   mov x0, #0     → 0xd2800000
\   ret            → 0xd65f03c0

: emit-host-stub  ( -- )
  $d2800000 a,     \ mov x0, #0
  $d65f03c0 a,  ;  \ ret

\ ---- Writer ---------------------------------------------------------------
: write-arm64-raw  ( addr u -- )
  577 open-file drop >r
  code-buf code-pos @ r@ write-file drop
  r> close-file drop ;
