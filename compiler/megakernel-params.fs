\ megakernel-params.fs — Megakernel parameter buffer layout for Lithos.
\
\ Architecture: two cooperative grid-sync megakernels per model.
\   - DeltaNet megakernel: 48 layers, one launch, cooperative grid-sync
\   - Full-attention megakernel: 16 layers, one launch, cooperative grid-sync
\
\ A CUDA kernel's parameters live in constant bank 0 (cbuf0), starting at
\ offset 0x210.  The entire cbuf0 section must be 256-byte aligned per
\ Hopper constant-bank alignment requirements (QMD cbuf0_size field).
\
\ Layout (DeltaNet):
\   [0 .. 9983]   Per-layer weight pointers: 48 × 208 bytes = 9984 bytes
\   [9984 .. 10055]  Global runtime pointers: 72 bytes
\   [10056 .. 10239] Padding to 256-byte boundary: 184 bytes
\   Total: 10240 bytes  → n-kparams = 10240 / 8 = 1280
\
\ Layout (Full-attention):
\   [0 .. 3327]    Per-layer weight pointers: 16 × 208 bytes = 3328 bytes
\   [3328 .. 3383]  Global runtime pointers: 56 bytes
\   [3384 .. 3583]  Padding to 256-byte boundary: 200 bytes
\   Total: 3584 bytes  → n-kparams = 3584 / 8 = 448
\
\ Per-layer slot layout (identical for both megakernels; slot indices match
\ WS_DN_* / WS_FA_* / WS_GATE_* / WS_INPUT_NORM / WS_POST_NORM in launcher.s):
\   Slot  Byte-off  Name
\    0      0       qkv_qweight  / q_qweight
\    1      8       qkv_scales   / q_scales
\    2     16       qkv_qzeros   / q_qzeros
\    3     24       z_qweight    / k_qweight
\    4     32       z_scales     / k_scales
\    5     40       z_qzeros     / k_qzeros
\    6     48       out_qweight  / v_qweight
\    7     56       out_scales   / v_scales
\    8     64       out_qzeros   / v_qzeros
\    9     72       proj_a       / o_qweight
\   10     80       proj_b       / o_scales
\   11     88       A_log        / o_qzeros
\   12     96       dt_bias      / q_norm
\   13    104       conv1d       / k_norm
\   14    112       norm         / (unused, zero)
\   15    120       mlp_gate_qweight
\   16    128       mlp_gate_scales
\   17    136       mlp_gate_qzeros
\   18    144       mlp_up_qweight
\   19    152       mlp_up_scales
\   20    160       mlp_up_qzeros
\   21    168       mlp_down_qweight
\   22    176       mlp_down_scales
\   23    184       mlp_down_qzeros
\   24    192       input_norm
\   25    200       post_norm
\    — 26 pointers × 8 bytes = 208 bytes per layer —

\ ============================================================
\ Shared per-layer constants
\ ============================================================

26 constant LAYER-PTRS          \ pointers per layer
208 constant LAYER-STRIDE       \ bytes per layer slot (26 × 8)

\ ============================================================
\ DeltaNet megakernel — 48 layers
\ ============================================================

48 constant DN-LAYERS

\ Per-layer byte-offset constants (relative to layer base address).
\ Layer base for layer L = DN-LAYER-BASE(L) = L * LAYER-STRIDE
0   constant DN-QKV-QWEIGHT-OFF
8   constant DN-QKV-SCALES-OFF
16  constant DN-QKV-QZEROS-OFF
24  constant DN-Z-QWEIGHT-OFF
32  constant DN-Z-SCALES-OFF
40  constant DN-Z-QZEROS-OFF
48  constant DN-OUT-QWEIGHT-OFF
56  constant DN-OUT-SCALES-OFF
64  constant DN-OUT-QZEROS-OFF
72  constant DN-PROJ-A-OFF
80  constant DN-PROJ-B-OFF
88  constant DN-A-LOG-OFF
96  constant DN-DT-BIAS-OFF
104 constant DN-CONV1D-OFF
112 constant DN-NORM-OFF
120 constant DN-GATE-QWEIGHT-OFF
128 constant DN-GATE-SCALES-OFF
136 constant DN-GATE-QZEROS-OFF
144 constant DN-UP-QWEIGHT-OFF
152 constant DN-UP-SCALES-OFF
160 constant DN-UP-QZEROS-OFF
168 constant DN-DOWN-QWEIGHT-OFF
176 constant DN-DOWN-SCALES-OFF
184 constant DN-DOWN-QZEROS-OFF
192 constant DN-INPUT-NORM-OFF
200 constant DN-POST-NORM-OFF

\ Global parameter block base offset (immediately after per-layer block).
DN-LAYERS LAYER-STRIDE * constant DN-GLOBAL-BASE   \ = 9984

\ Global parameter offsets (absolute byte offset into param buffer).
DN-GLOBAL-BASE  0 + constant DN-ACT-A-OFF          \ activation_buffer_a  (u64)
DN-GLOBAL-BASE  8 + constant DN-ACT-B-OFF          \ activation_buffer_b  (u64)
DN-GLOBAL-BASE 16 + constant DN-RESIDUAL-OFF       \ residual_buffer      (u64)
DN-GLOBAL-BASE 24 + constant DN-STATE-BASE-OFF     \ deltanet_state_base  (u64) — ptr to 48-entry array
DN-GLOBAL-BASE 32 + constant DN-CONV1D-STATE-OFF   \ conv1d_state_base    (u64) — ptr to 48-entry array
DN-GLOBAL-BASE 40 + constant DN-Z-BUF-OFF          \ z_buffer             (u64)
DN-GLOBAL-BASE 48 + constant DN-SEQPOS-OFF         \ sequence_position    (u32)
DN-GLOBAL-BASE 52 + constant DN-HIDDENDIM-OFF      \ hidden_dim           (u32)
DN-GLOBAL-BASE 56 + constant DN-GSYNC-CTR-OFF      \ grid_sync_counter    (u64) — allocated at launch
DN-GLOBAL-BASE 64 + constant DN-GSYNC-FLAG-OFF     \ grid_sync_flag       (u64) — allocated at launch

\ Raw buffer sizes before and after 256-byte padding.
10056 constant DN-PARAMS-RAW     \ 9984 + 72
10240 constant DN-PARAMS-BYTES   \ padded to 256-byte boundary (next multiple of 256)
1280  constant DN-NKPARAMS       \ DN-PARAMS-BYTES / 8  (for EIATTR_PARAM_CBANK)

\ ============================================================
\ Full-attention megakernel — 16 layers
\ ============================================================

16 constant FA-LAYERS

\ Per-layer byte-offset constants (relative to layer base).
\ Slot indices intentionally match WS_FA_* in launcher.s.
0   constant FA-Q-QWEIGHT-OFF
8   constant FA-Q-SCALES-OFF
16  constant FA-Q-QZEROS-OFF
24  constant FA-K-QWEIGHT-OFF
32  constant FA-K-SCALES-OFF
40  constant FA-K-QZEROS-OFF
48  constant FA-V-QWEIGHT-OFF
56  constant FA-V-SCALES-OFF
64  constant FA-V-QZEROS-OFF
72  constant FA-O-QWEIGHT-OFF
80  constant FA-O-SCALES-OFF
88  constant FA-O-QZEROS-OFF
96  constant FA-Q-NORM-OFF
104 constant FA-K-NORM-OFF
\ slot 14 (offset 112) is unused for FA; zero-filled by builder
120 constant FA-GATE-QWEIGHT-OFF
128 constant FA-GATE-SCALES-OFF
136 constant FA-GATE-QZEROS-OFF
144 constant FA-UP-QWEIGHT-OFF
152 constant FA-UP-SCALES-OFF
160 constant FA-UP-QZEROS-OFF
168 constant FA-DOWN-QWEIGHT-OFF
176 constant FA-DOWN-SCALES-OFF
184 constant FA-DOWN-QZEROS-OFF
192 constant FA-INPUT-NORM-OFF
200 constant FA-POST-NORM-OFF

\ Global parameter block base offset (after per-layer block).
FA-LAYERS LAYER-STRIDE * constant FA-GLOBAL-BASE   \ = 3328

\ Global parameter offsets (absolute byte offset into param buffer).
FA-GLOBAL-BASE  0 + constant FA-ACT-A-OFF          \ activation_buffer_a  (u64)
FA-GLOBAL-BASE  8 + constant FA-ACT-B-OFF          \ activation_buffer_b  (u64)
FA-GLOBAL-BASE 16 + constant FA-RESIDUAL-OFF       \ residual_buffer      (u64)
FA-GLOBAL-BASE 24 + constant FA-KV-CACHE-OFF       \ kv_cache_base        (u64)
FA-GLOBAL-BASE 32 + constant FA-SEQPOS-OFF         \ sequence_position    (u32)
FA-GLOBAL-BASE 36 + constant FA-HIDDENDIM-OFF      \ hidden_dim           (u32)
FA-GLOBAL-BASE 40 + constant FA-GSYNC-CTR-OFF      \ grid_sync_counter    (u64)
FA-GLOBAL-BASE 48 + constant FA-GSYNC-FLAG-OFF     \ grid_sync_flag       (u64)

\ Raw buffer sizes before and after 256-byte padding.
3384 constant FA-PARAMS-RAW     \ 3328 + 56
3584 constant FA-PARAMS-BYTES   \ padded to 256-byte boundary
448  constant FA-NKPARAMS       \ FA-PARAMS-BYTES / 8

\ ============================================================
\ Param buffer allocator
\ ============================================================
\ Simple heap bump-allocator for the param buffers.  Caller must
\ ensure the backing store is large enough.  We allocate from the
\ ARM64 process heap via brk.
\
\ param-buf-alloc ( nbytes -- addr )
\   Allocate nbytes from the system heap (brk), return start address.
\   Rounds up to 256-byte boundary before and after to satisfy cbuf0
\   alignment requirements.

: param-buf-alloc  ( nbytes -- addr )
  \ Round nbytes up to next 256-byte multiple first.
  255 + 255 invert and     ( rounded-nbytes )
  here ;                   \ Return HERE as the base address.
  \ In a real Forth with brk we would: dup allot — but this Forth
  \ already manages its own dictionary space with allot.
  \ Caller should: nbytes allot  after calling this to advance HERE.

\ ============================================================
\ DeltaNet param buffer builder
\ ============================================================
\
\ build-deltanet-params ( weight-table-ptr global-ptrs-ptr -- buf-addr buf-len )
\
\ weight-table-ptr : pointer to a flat u64 array of
\       64 × 26 × 8 bytes (all 64 model layers, 26 slots each),
\       laid out as weight_table[layer_idx][slot_idx].
\       Only the 48 DeltaNet layers are used here (those where
\       layer_idx % 4 != 3).
\
\ global-ptrs-ptr  : pointer to a packed struct of the 10 global
\       runtime values in the order:
\         u64 activation_buffer_a
\         u64 activation_buffer_b
\         u64 residual_buffer
\         u64 deltanet_state_base
\         u64 conv1d_state_base
\         u64 z_buffer
\         u32 sequence_position
\         u32 hidden_dim
\         u64 grid_sync_counter   (allocated at launch by launcher.s)
\         u64 grid_sync_flag      (allocated at launch by launcher.s)
\
\ Returns: (buf-addr buf-len) ready to hand to cbuf0.

\ Global variable holding the param buffer base.  Allocated once on
\ first call (or pre-allocated by the caller and poked via
\ dn-param-buf !).
variable dn-param-buf   0 dn-param-buf !

\ Scratch variables used by the builder.
variable dpb-wt      \ current weight-table base
variable dpb-gp      \ global-ptrs base
variable dpb-base    \ param buffer base address
variable dpb-off     \ current write offset into param buffer

\ Store a u64 into the param buffer at byte offset dpb-off, advance.
: dpb-u64!  ( u64 -- )
  dpb-base @ dpb-off @ +  !    \ store 8 bytes (Forth cell = 8 bytes on 64-bit)
  8 dpb-off +! ;

\ Store a u32 into the param buffer at byte offset dpb-off, advance.
: dpb-u32!  ( u32 -- )
  dpb-base @ dpb-off @ +  l!   \ 32-bit store
  4 dpb-off +! ;

\ Zero-fill from current dpb-off to end of buffer.
: dpb-zero-pad  ( total-bytes -- )
  dpb-off @ ?do  0 dpb-base @ i + c!  loop ;

\ Read u64 from weight table: layer-index(0-63) slot-index(0-25) -- u64
\ weight_table[layer][slot] = *(wt_base + (layer*26 + slot)*8)
: wt-slot@  ( layer slot -- u64 )
  swap 26 * +  8 *  dpb-wt @ + @ ;

\ Read u64 from global-ptrs at byte offset.
: gp-u64@   ( byte-off -- u64 )  dpb-gp @ + @ ;
\ Read u32 from global-ptrs at byte offset.
: gp-u32@   ( byte-off -- u32 )  dpb-gp @ + l@ ;

\ Emit all 26 weight slots for one layer.
\ layer-model-idx is the actual model layer index (0-63).
: dpb-emit-layer  ( layer-model-idx -- )
  26 0 do
    dup i wt-slot@  dpb-u64!
  loop
  drop ;

: build-deltanet-params  ( weight-table-ptr global-ptrs-ptr -- buf-addr buf-len )
  dpb-gp !
  dpb-wt !

  \ Allocate param buffer if not yet done.
  dn-param-buf @ 0= if
    DN-PARAMS-BYTES allot  here DN-PARAMS-BYTES -  dn-param-buf !
  then
  dn-param-buf @ dpb-base !
  0 dpb-off !

  \ --- Per-layer block (48 DeltaNet layers) ---
  \ DeltaNet layers are model layers where layer_idx % 4 != 3.
  \ In a 64-layer model (0-63) the DeltaNet layers are:
  \   0,1,2, 4,5,6, 8,9,10, ..., 60,61,62  (48 total)
  \ We walk 0..63 and skip every 4th (the FA layers at idx%4==3).
  64 0 do
    i 4 mod 3 = if  \ FA layer — skip
    else
      i dpb-emit-layer
    then
  loop

  \ --- Global params ---
  0  gp-u64@ dpb-u64!   \ activation_buffer_a
  8  gp-u64@ dpb-u64!   \ activation_buffer_b
  16 gp-u64@ dpb-u64!   \ residual_buffer
  24 gp-u64@ dpb-u64!   \ deltanet_state_base
  32 gp-u64@ dpb-u64!   \ conv1d_state_base
  40 gp-u64@ dpb-u64!   \ z_buffer
  48 gp-u32@ dpb-u32!   \ sequence_position (u32)
  52 gp-u32@ dpb-u32!   \ hidden_dim (u32)
  56 gp-u64@ dpb-u64!   \ grid_sync_counter
  64 gp-u64@ dpb-u64!   \ grid_sync_flag

  \ --- Pad to 256-byte boundary ---
  DN-PARAMS-BYTES dpb-zero-pad

  dpb-base @ DN-PARAMS-BYTES ;

\ ============================================================
\ Full-attention param buffer builder (stub)
\ ============================================================
\
\ build-attention-params ( weight-table-ptr global-ptrs-ptr -- buf-addr buf-len )
\
\ weight-table-ptr : same 64-layer weight table, only 16 FA layers used
\   (those where layer_idx % 4 == 3: model layers 3,7,11,...,63).
\
\ global-ptrs-ptr  : packed struct in order:
\         u64 activation_buffer_a
\         u64 activation_buffer_b
\         u64 residual_buffer
\         u64 kv_cache_base
\         u32 sequence_position
\         u32 hidden_dim
\         u64 grid_sync_counter   (allocated at launch)
\         u64 grid_sync_flag      (allocated at launch)
\
\ Returns: (buf-addr buf-len) ready for cbuf0.

variable fa-param-buf   0 fa-param-buf !

variable fab-wt
variable fab-gp
variable fab-base
variable fab-off

: fab-u64!  ( u64 -- )
  fab-base @ fab-off @ +  !
  8 fab-off +! ;

: fab-u32!  ( u32 -- )
  fab-base @ fab-off @ +  l!
  4 fab-off +! ;

: fab-zero-pad  ( total-bytes -- )
  fab-off @ ?do  0 fab-base @ i + c!  loop ;

: fab-wt-slot@  ( layer slot -- u64 )
  swap 26 * +  8 *  fab-wt @ + @ ;

: fab-gp-u64@  ( byte-off -- u64 )  fab-gp @ + @ ;
: fab-gp-u32@  ( byte-off -- u32 )  fab-gp @ + l@ ;

: fab-emit-layer  ( layer-model-idx -- )
  26 0 do
    dup i fab-wt-slot@  fab-u64!
  loop
  drop ;

: build-attention-params  ( weight-table-ptr global-ptrs-ptr -- buf-addr buf-len )
  fab-gp !
  fab-wt !

  fa-param-buf @ 0= if
    FA-PARAMS-BYTES allot  here FA-PARAMS-BYTES -  fa-param-buf !
  then
  fa-param-buf @ fab-base !
  0 fab-off !

  \ --- Per-layer block (16 FA layers: model layers 3,7,11,...,63) ---
  \ FA layers are where layer_idx % 4 == 3.
  \ Slot 14 (offset 112) is unused for FA; the weight table entry is 0.
  64 0 do
    i 4 mod 3 = if
      i fab-emit-layer
    then
  loop

  \ --- Global params ---
  0  fab-gp-u64@ fab-u64!   \ activation_buffer_a
  8  fab-gp-u64@ fab-u64!   \ activation_buffer_b
  16 fab-gp-u64@ fab-u64!   \ residual_buffer
  24 fab-gp-u64@ fab-u64!   \ kv_cache_base
  32 fab-gp-u32@ fab-u32!   \ sequence_position (u32)
  36 fab-gp-u32@ fab-u32!   \ hidden_dim (u32)
  40 fab-gp-u64@ fab-u64!   \ grid_sync_counter
  48 fab-gp-u64@ fab-u64!   \ grid_sync_flag

  \ --- Pad to 256-byte boundary ---
  FA-PARAMS-BYTES fab-zero-pad

  fab-base @ FA-PARAMS-BYTES ;
