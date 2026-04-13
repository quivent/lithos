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
\
\ Cooperative kernels:
\   Set  1 cooperative? !  (or call build-cubin-coop) before write-cubin.
\   emit-grid-sync (written separately) calls record-gridsync-offset before
\   emitting each grid-sync instruction so offsets are tracked automatically.
\   The driver requires EIATTR_COOP_GROUP_INSTR_OFFSETS and
\   EIATTR_COOP_GROUP_MASK_REGIDS in .nv.info.<kernel> to accept
\   cuLaunchCooperativeKernel.  build-cubin emits them when cooperative?=1.
\
\ ---- Megakernel n-kparams values (from compiler/megakernel-params.fs) ----
\
\ For megakernel cubins, bypass count-all-params and set n-kparams directly:
\
\   DeltaNet megakernel (48 layers, cooperative):
\     n-kparams = DN-NKPARAMS = 1280
\     cbuf0 param region = 10240 bytes  (0x2800)
\     Layout: 48 × 208-byte layer blocks + 72 bytes global + 184 bytes pad
\     To build: DN-NKPARAMS n-kparams !  1 cooperative? !
\               s" deltanet_mega" li-set-name  write-cubin
\
\   Full-attention megakernel (16 layers, cooperative):
\     n-kparams = FA-NKPARAMS = 448
\     cbuf0 param region = 3584 bytes  (0xE00)
\     Layout: 16 × 208-byte layer blocks + 56 bytes global + 200 bytes pad
\     To build: FA-NKPARAMS n-kparams !  1 cooperative? !
\               s" attention_mega" li-set-name  write-cubin
\
\ EIATTR_PARAM_CBANK in .nv.info.<kernel> encodes the param-region byte count
\ in bits [31:16] of the second u32 (see emit-sass.fs line ~850):
\   DeltaNet: ( 10240 << 16 ) | 0x210  = 0x28000210
\   FA:       (  3584 << 16 ) | 0x210  = 0x0E000210
\
\ Grid-sync counter and flag GPU addresses are written into the global-params
\ struct by launcher.s at each token step (cuMemAlloc once at startup, then
\ their device VAs are stored at fixed offsets in the global-ptrs array that
\ build-deltanet-params / build-attention-params reads).  No cbuf0 patching
\ is needed at launch time; the entire param buffer is rebuilt each step via
\ build-deltanet-params / build-attention-params.

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

\ write-cubin-coop ( outpath outlen -- )
\ Like write-cubin but marks the kernel cooperative before building.
\ Assumes emit-grid-sync has called record-gridsync-offset for each sync site.
: write-cubin-coop  ( outpath outlen -- )
  1 cooperative? !
  write-cubin ;
