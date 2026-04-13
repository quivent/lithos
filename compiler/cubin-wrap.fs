\ cubin-wrap.fs — Build a complete sm_90 cubin and write it to disk.
\
\ The heavy lifting lives in /home/ubuntu/lithos/sass/emit-sass.fs which
\ provides build-cubin ( -- addr u ). This file only glues parser state
\ (kernel name, param count) into the builder and handles file I/O.
\
\ Cubin layout produced by build-cubin:
\   [0] NULL
\   [1] .shstrtab
\   [2] .strtab
\   [3] .symtab   (4 entries: UND, SECTION(.text), SECTION(.nv.constant0),
\                  FUNC GLOBAL kernel STO_CUDA_ENTRY)
\   [4] .nv.info
\   [5] .text.<kernel>
\   [6] .nv.info.<kernel>
\   [7] .nv.shared.reserved.0   (NOBITS)
\   [8] .nv.constant0.<kernel>  (0x210 reserved + param_bytes)
\ No program headers: cuModuleLoadData tolerates shdr-only cubins for a
\ single kernel without shared-mem dependencies.

variable cw-fd

\ write-cubin ( outpath outlen -- )
\ Builds the cubin from the current sass-buf + parser state and writes it.
: write-cubin  ( outpath outlen -- )
  \ Propagate parser's param count (ptr + scalar) to the cubin builder.
  count-all-params  n-kparams !

  \ Open output file from the two TOS items (outpath outlen), then build cubin
  \ and fetch addr/len from cubin-buf / cubin-pos directly — the bootstrap
  \ appears to leave extra ephemeral items on the param stack through
  \ lithos-compile, so we avoid relying on stack order.
  577 open-file drop cw-fd !     \ ( --  )  consumes outpath outlen mode
  build-cubin 2drop              \ discard addr/u — we read them via globals
  cubin-buf cubin-pos @  cw-fd @  write-file drop
  cw-fd @ close-file drop ;
