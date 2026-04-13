\ emit-sass.fs — Lithos SASS binary emitter (wrapper)
\
\ Pulls in the existing Hopper opcode emitter at /home/ubuntu/lithos/sass/
\ emit-sass.fs, which has the 47 sm_90 opcodes already mapped. This file
\ only adds the write-sass word so the compiler driver can dump the cubin
\ payload to disk.

s" /home/ubuntu/lithos/sass/emit-sass.fs" included

\ Write the current SASS buffer to a file (pre-cubin-wrap payload).
: write-sass-raw  ( addr u -- )
  577 open-file drop >r
  sass-buf sass-pos @ r@ write-file drop
  r> close-file drop ;
