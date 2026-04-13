\ emit-sass.fs — Lithos GPU machine code emitter (wrapper)
\
\ Pulls in the Hopper opcode emitter at gpu/emit.fs,
\ which has the sm_90 opcodes mapped. This file only adds
\ write-sass-raw so the compiler driver can dump the code buffer to disk.

s" /home/ubuntu/lithos/gpu/emit.fs" included

\ Write the current code buffer to a file (raw binary payload).
: write-sass-raw  ( addr u -- )
  577 open-file drop >r
  sass-buf sass-pos @ r@ write-file drop
  r> close-file drop ;
