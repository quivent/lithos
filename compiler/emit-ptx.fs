\ emit-ptx.fs — PTX text emitter for Lithos
\
\ This file simply pulls in the existing Lithos PTX vocabulary (core.fs and
\ patterns.fs) and exposes a single write-ptx word that dumps the PTX buffer
\ to a file. The per-kernel header/body/footer emission lives in parser.fs;
\ the 120 pattern words live in ../patterns.fs.

s" /home/ubuntu/lithos/core.fs"     included
s" /home/ubuntu/lithos/patterns.fs" included

\ Dump ptx-buf to the path in ( addr u ).
\ File-access vocabulary: open-file, write-file, close-file, w/o.
2 constant W/O-MODE    \ O_WRONLY
1 constant O-CREAT     \ actually ignored; open-file uses passed flag

: create-out  ( addr u -- fd )
  \ Use 577 = O_WRONLY | O_CREAT | O_TRUNC on Linux
  577 open-file drop ;

: write-ptx  ( addr u -- )
  create-out >r
  ptx-buf ptx-pos @ r@ write-file drop
  r> close-file drop ;
