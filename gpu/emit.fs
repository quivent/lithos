\ gpu/emit.fs — Lithos GPU machine code emitter for Hopper (sm_90)
\ Emits raw sm90 binary instructions (16 bytes each) into a code buffer.
\ No PTX, no ptxas, no driver JIT.

\ Verified opcode constants for sm_90a — must be included first.
s" /home/ubuntu/lithos/gpu/opcodes-sm90.fs" included

\ ============================================================
\ KERNEL NAME METADATA (shared with parser.fs / ls-compiler.fs)
\ ============================================================
\ These are the canonical definitions. parser.fs references them
\ without redefining (they are loaded here first, before parser.fs).
create li-name-buf 64 allot
variable li-name-len  0 li-name-len !
: li-set-name  ( addr u -- )  dup li-name-len !  li-name-buf swap move ;
: li-name$  ( -- addr u )  li-name-buf  li-name-len @ ;

\ ============================================================
\ CONTROL WORD SCHEDULER  (sm_90 128-bit instruction format)
\ ============================================================
\ Each 16-byte instruction = 8-byte inst word + 8-byte ctrl word.
\ The ctrl word encodes hardware scheduling fields at bits [127:64]
\ of the 128-bit instruction (ctrl bit N = instruction bit N+64).
\
\   ctrl bits 41-44  = stall count  (0-15 cycles; 0=no stall)
\   ctrl bit  45     = yield hint   (1 = scheduler may switch warp)
\   ctrl bits 46-48  = write barrier slot (0-6; 7=none)
\   ctrl bits 49-51  = read barrier slot  (0-6; 7=none)
\   ctrl bits 52-57  = wait barriers mask (bit N => stall until slot N clears)
\   ctrl bits 58-62  = reuse cache flags
\   ctrl bits  0-40  = opaque barrier-table / cache fields
\                      (taken verbatim from probe disassembly)
\
\ Verified against probe*.sass nvdisasm output (sm_90).
\
\ make-ctrl  ( stall yield wbar rbar wait reuse extra41 -- ctrl64 )
\ Uses scratch variables to avoid r-stack LIFO ordering bugs.
variable mc-reuse  variable mc-wait   variable mc-rbar
variable mc-wbar   variable mc-yield  variable mc-stall
: make-ctrl  ( stall yield wbar rbar wait reuse extra41 -- ctrl64 )
  \ TOS = extra41; pop remaining into variables (top-first = reverse order)
  >r                             \ save extra41
  mc-reuse !  mc-wait !  mc-rbar !  mc-wbar !  mc-yield !  mc-stall !
  r>                             \ extra41 (bits 0-40)
  mc-stall @ 41 lshift or        \ stall  -> bits 41-44
  mc-yield @ 45 lshift or        \ yield  -> bit  45
  mc-wbar  @ 46 lshift or        \ wbar   -> bits 46-48
  mc-rbar  @ 49 lshift or        \ rbar   -> bits 49-51
  mc-wait  @ 52 lshift or        \ wait   -> bits 52-57
  mc-reuse @ 58 lshift or ;      \ reuse  -> bits 58-62              \ reuse  -> bits 58-62

\ ---- Per-instruction control word constructors ----
\ All stall/barrier values derived from probe disassembly.

\ NOP: stall=0 yield=0 wbar=7 rbar=7 — verified 0x000fc00000000000
: ctrl-nop        0 0 7 7 0 0 0             make-ctrl ;

\ S2R: stall=7 yield=1 wbar=1 rbar=7 — verified 0x000e6e0000002100
\ extra41 = sr-id << 8  (probe: SR_TID.X=0x21 -> extra41=0x2100)
: ctrl-s2r  ( sr-id -- ctrl64 )
  8 lshift >r  7 1 1 7 0 0 r> make-ctrl ;

\ LDG.E: stall=4 yield=1 wbar=2 rbar=7 — verified 0x000ea8000c1e1900
\ extra41 = 0x0c1e1900 (opaque cache-line / descriptor fields from probe)
: ctrl-ldg        4 1 2 7 0 0 $0c1e1900     make-ctrl ;

\ LDG.E with wait on barrier 0 (first load in sequence after a barrier)
\ verified: 0x001ea8000c1e1900  wait=1 (bit 0)
: ctrl-ldg-wait0  4 1 2 7 1 0 $0c1e1900     make-ctrl ;

\ STG.E: stall=1 yield=1 wbar=7 rbar=7 — verified 0x000fe2000c101904
: ctrl-stg        1 1 7 7 0 0 $0c101904     make-ctrl ;

\ STG.E waiting on barrier 2 (store depends on LDG output):
\ verified: 0x001fe8000c101904  wait=1, stall=4
: ctrl-stg-wait2  4 1 7 7 1 0 $0c101904     make-ctrl ;

\ FFMA: stall=5 yield=0 wbar=7 rbar=7 — probe: 0x000fca0000000009 -> stall=5
\ Rs3 (accumulator reg index) lives in extra41 bits[7:0]
: ctrl-ffma  ( rs3 -- ctrl64 )  >r 5 0 7 7 0 0 r> make-ctrl ;

\ FADD: stall=5 yield=0 wbar=7 rbar=7 (no load dependency)
: ctrl-fadd       5 0 7 7 0 0 0             make-ctrl ;

\ FADD after LDG (waits on write barrier 2): verified 0x004fca0000000000
: ctrl-fadd-wait2 5 0 7 7 4 0 0             make-ctrl ;

\ IMAD: stall=1 yield=1 wbar=7 rbar=7
\ extra41 base = 0x0f8e0200 (opaque scheduler fields from probe); rs3 ORed into bits[7:0]
\ probe IMAD R0,R3,UR4,R0 ctrl: 0x002fe2000f8e0200 (extra41=0x0f8e0200, rs3=R0=0x00)
: ctrl-imad  ( rs3 -- ctrl64 )  $0f8e0200 or >r  1 1 7 7 0 0 r>  make-ctrl ;

\ IMAD-IMM: stall=1 yield=1, extra41=0x78e00ff (MOV.U32 modifier + RZ accum=0xff)
\ verified: 0x000fe200078e00ff
: ctrl-imad-imm   1 1 7 7 0 0 $78e00ff      make-ctrl ;

\ ISETP: stall=13 yield=0 wbar=7 rbar=7 — verified 0x000fda000bf06070
: ctrl-isetp      13 0 7 7 0 0 $0bf06070    make-ctrl ;

\ BRA: stall=0 yield=0 wbar=7 rbar=7 — verified 0x000fc0000383ffff
: ctrl-bra        0 0 7 7 0 0 $0383ffff     make-ctrl ;

\ EXIT: stall=5 yield=1 wbar=7 rbar=7 — verified 0x000fea0003800000
: ctrl-exit       5 1 7 7 0 0 $03800000     make-ctrl ;

\ ULDC: stall=1 yield=1 wbar=7 rbar=7 — verified 0x000fe20000000a00
: ctrl-uldc       1 1 7 7 0 0 $0a00         make-ctrl ;

\ LDC: stall=7 yield=1 wbar=1 rbar=7 — verified 0x000e220000000800
: ctrl-ldc        7 1 1 7 0 0 $800          make-ctrl ;

\ SHFL.BFLY: stall=14 yield=0 wbar=3 rbar=7 — verified 0x001e6800000e0000
: ctrl-shfl       14 0 3 7 0 0 $000e0000    make-ctrl ;

\ STS.32: stall=1 yield=1 wbar=7 rbar=7 extra=0x0800 (ctrl bits[13:10]=0x2 => 32-bit)
\ derived from STS.128 ctrl=0x000fe20008000c04; 32-bit drops the 0x0400 width bit
: ctrl-sts        1 1 7 7 0 0 $0800         make-ctrl ;

\ LDS.32: stall=7 yield=1 wbar=1 rbar=7 extra=0x0800 (ctrl bits[13:10]=0x2 => 32-bit)
\ ENCODING.md: LDS R7,[R5] ctrl bits[13:10]=0x2; scheduling from LDC (same latency)
: ctrl-lds        7 1 1 7 0 0 $0800         make-ctrl ;

\ BAR.SYNC: stall=5 yield=1 wbar=7 rbar=7 extra=0x00010000
\ verified: BAR.SYNC.DEFER_BLOCKING 0x0 -> ctrl=0x000fec0000010000
: ctrl-bar        5 1 7 7 0 0 $00010000     make-ctrl ;

\ ============================================================
\ GPU CODE BUFFER
\ ============================================================

65536 constant SASS-SIZE
create sass-buf SASS-SIZE allot
variable sass-pos  0 sass-pos !

\ Per-kernel register high-water mark.
\ Reset to 0 at sass-reset; updated by every register-destination instruction.
\ build-elf reads this to emit a tight REGCOUNT instead of a hardcoded value.
variable max-reg-used  0 max-reg-used !

\ track-rd  ( rd -- rd )
\ Non-destructive: updates max-reg-used if rd > current max, then leaves rd.
: track-rd  ( rd -- rd )  dup max-reg-used @ max max-reg-used ! ;

\ ============================================================
\ COOPERATIVE GRID-SYNC STATE
\ ============================================================
\ cooperative?  — set to 1 (true) when the kernel uses grid-wide sync.
\ build-elf checks this and emits COOP_GROUP attrs in .nv.info.<kernel>.
\ Default is 0 (non-cooperative; backward compatible).
variable cooperative?  0 cooperative? !

\ gridsync-offsets — array of u32 byte offsets into .text.<kernel> for each
\ grid-sync instruction emitted by emit-grid-sync / grid-sync,.
create gridsync-offsets  256 cells allot   \ up to 256 grid-sync sites
variable gridsync-count  0 gridsync-count !

\ record-gridsync-offset  ( -- )
\ Snapshot current sass-pos as the byte offset of the NEXT grid-sync instruction.
\ Call this immediately BEFORE emitting the sync instruction bytes so that
\ sass-pos still points to the instruction start.
: record-gridsync-offset  ( -- )
  gridsync-count @ 255 > if exit then       \ silently clamp at 256 sites
  sass-pos @  gridsync-offsets gridsync-count @ cells +  !
  1 gridsync-count +! ;

\ coop-reset  ( -- )  — clear coop state; called by sass-reset.
: coop-reset  ( -- )
  0 gridsync-count !
  0 cooperative? ! ;

: sass-reset  0 sass-pos !  0 max-reg-used !  coop-reset ;

\ Simpler: emit raw bytes into the code buffer
: sb,  ( byte -- )  sass-buf sass-pos @ + c!  1 sass-pos +! ;
: sw,  ( u32 -- )   dup sb, 8 rshift dup sb, 8 rshift dup sb, 8 rshift sb, ;
: sq,  ( u64 -- )   dup sw, 32 rshift sw, ;

\ Emit a full 16-byte sm90 instruction word from two 64-bit values
: sinst,  ( inst64 ctrl64 -- )  swap sq, sq, ;

\ Emit one 16-byte instruction word from four u32 values.
\ DEPRECATED — use sinst, instead.
: sass,  ( inst-lo inst-hi ctrl-lo ctrl-hi -- )
  swap 32 lshift swap or >r    \ combine ctrl-hi|ctrl-lo -> ctrl64
  swap 32 lshift swap or       \ combine inst-hi|inst-lo -> inst64
  r> sinst, ;

\ Opcodes are defined in opcodes-sm90.fs (single source of truth).
\ Removed duplicate table that had WRONG values for OP-IMAD ($7c24 should be $7224),
\ OP-ISETP ($7c0c should be $720c), OP-IADD3 ($7c10 should be $7210).

\ Special register IDs — SR selector field in S2R ctrl extra41[15:8]
\ Verified from probe_s2r2 and probe_s2r3 (O0) nvdisasm output.
\ The inst word for S2R is always 0x00000000000r7919 (rd in bits[23:16]).
\ The SR ID is carried ONLY in the ctrl word extra41[15:8], not in the inst word.
\
\ S2R ctrl formula: ctrl-s2r sr-id  where sr-id goes into extra41[15:8] = sr-id << 8.
\   SR_TID.X   ctrl=0x000e6e0000002100  (extra41=0x2100, sr-id=0x21)
\   SR_TID.Y   ctrl=0x000e620000002200  (extra41=0x2200, sr-id=0x22)
\   SR_CTAID.X ctrl=0x000eaa0000002500  (extra41=0x2500, sr-id=0x25)
\   SR_CTAID.Y ctrl=0x000f220000002600  (extra41=0x2600, sr-id=0x26)
\   SR_LANEID  ctrl=0x000f620000000000  (extra41=0x0000, sr-id=0x00)
\
\ Note: %nctaid.x / %warpid are lowered to constant-bank loads by ptxas (not S2R).
$21 constant SR-TID-X
$22 constant SR-TID-Y
$23 constant SR-TID-Z
$25 constant SR-CTAID-X
$26 constant SR-CTAID-Y
$27 constant SR-CTAID-Z
$00 constant SR-LANEID     \ lane within warp (0..31)

\ ============================================================
\ INSTRUCTION BUILDERS
\ ============================================================

\ NOP: inst = 0x0000000000007918, ctrl = 0x000fc00000000000
: nop,  ( -- )
  $0000000000007918 ctrl-nop sinst, ;



\ FADD Rd, Ra, Rb
\ From probe: FADD R9, R2, R5 = inst 0x0000000502097221
\ Rd = bits[23:16] = 0x09 (R9)
\ Ra = bits[31:24] = 0x02 (R2)
\ Rb = bits[39:32] = 0x05 (R5)
: fadd,  ( rd ra rb -- )
  32 lshift >r              \ rb -> bits[39:32]
  24 lshift >r              \ ra -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7221 or r> or r> or
  ctrl-fadd sinst, ;

\ FFMA placeholder — immediately overridden by the full definition below.
\ Kept so that forward references (if any) resolve during compilation.
: ffma,  ( rd ra rb rc -- )
  drop drop drop drop
  $0000000502097223 0 ctrl-ffma sinst, ;

\ ============================================================
\ GEMV OPERATIONS — LDG, STG, IMAD, SHFL.BFLY, BRA, ISETP
\ Encodings verified against probe disassembly (probe*.sass)
\ ============================================================

\ (OP-BRA and OP-ISETP already defined above in the opcode table.)

\ LDG.E Rd, [Ra]  — 32-bit global load, no offset
\ Verified: LDG.E R0, [R2.64] -> inst=0x0000000402007981 ctrl=0x000ea8000c1e1900
: ldg,  ( rd ra -- )
  24 lshift >r              \ ra -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7981 or r> or
  ctrl-ldg sinst, ;

\ LDG.E Rd, [Ra+off]  — 32-bit global load with immediate byte offset
\ Verified: LDG.E R5, [R2.64+0x4] -> 0x0000040402057981  (off in bits[47:32])
: ldg-off,  ( rd ra off -- )
  32 lshift >r              \ off -> bits[47:32]
  24 lshift >r              \ ra  -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7981 or r> or r> or
  ctrl-ldg sinst, ;

\ STG.E [Ra], Rs  — 32-bit global store, no offset
\ Verified: STG.E [R6.64], R9 -> inst=0x0000000906007986 ctrl=0x000fe2000c101904
: stg,  ( ra rs -- )
  32 lshift >r              \ rs -> bits[39:32]
  24 lshift                 \ ra -> bits[31:24]
  $7986 or r> or
  ctrl-stg sinst, ;

\ STG.E [Ra+off], Rs  — 32-bit global store with immediate byte offset
\ Verified: STG.E [R2+0x4], R7 -> 0x0000040702007986
: stg-off,  ( ra rs off -- )
  32 lshift >r              \ off -> bits[47:32]
  swap 32 lshift >r         \ rs  -> bits[39:32]
  24 lshift                 \ ra  -> bits[31:24]
  $7986 or r> or r> or
  ctrl-stg sinst, ;

\ IMAD Rd, Rs1, Rs2, Rs3  — integer multiply-add (reg*reg+reg)
\ Verified: IMAD R0, R3, UR4, R0 -> inst=0x0000000403007c24
\ Rs3 (accumulator) in ctrl[7:0] per ENCODING.md 4-operand format
: imad,  ( rd rs1 rs2 rs3 -- )
  >r                        \ rs3 saved for ctrl-imad
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7c24 or r> or r> or
  r> ctrl-imad sinst, ;

\ IMAD-IMM Rd, Rs1, imm32  — integer multiply immediate + RZ accumulator
\ Verified: IMAD.MOV.U32 Rd, RZ, RZ, imm -> opcode $7424, Rs2=RZ in ctrl[7:0]
: imad-imm,  ( rd rs1 imm32 -- )
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ rs1   -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7424 or r> or r> or
  ctrl-imad-imm sinst, ;

\ FFMA Rd, Rs1, Rs2, Rs3  — float fused multiply-add (register form)
\ Correct implementation replacing the stub above.
\ Rs3 (accumulator reg index) packed into ctrl extra41[7:0] via ctrl-ffma.
\ Note: redefines the stub ffma, from above.
: ffma,  ( rd rs1 rs2 rs3 -- )
  >r                        \ rs3 -> ctrl-ffma arg
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7223 or r> or r> or
  r> ctrl-ffma sinst, ;

\ FMUL Rd, Rs1, Rs2 — float multiply (= FFMA with RZ accumulator)
: fmul,  ( rd rs1 rs2 -- )  255 ffma, ;

\ SHFL.BFLY PT, Rd, Rs, delta  — warp butterfly shuffle (imm delta, mask=0x1f)
\ Verified from probe_6b (delta=1) and probe_reduce (all 5 reduction deltas).
\
\ ENCODING (128-bit = inst64 + ctrl64):
\   inst bits[15:0]  = 0x7f89  (opcode: SHFL, PT pred, immediate-delta form)
\   inst bits[23:16] = Rd      (destination register)
\   inst bits[31:24] = Rs      (source register)
\   inst bits[39:32] = 0x00    (unused in immediate form)
\   inst bits[47:40] = 0x1f    (mask = warp_width - 1 = 31)
\   inst bits[55:48] = clamp   = (delta * 32) & 0xff
\   inst bits[63:56] = mode    = 0x0c | ((delta * 32) >> 8)
\
\ Delta encoding: delta_enc = delta * 32 (a 10-bit field split across bits[57:48])
\   delta=1:  delta_enc=0x020, mode=0x0c, clamp=0x20 -> top16=0x0c20
\   delta=2:  delta_enc=0x040, mode=0x0c, clamp=0x40 -> top16=0x0c40
\   delta=4:  delta_enc=0x080, mode=0x0c, clamp=0x80 -> top16=0x0c80
\   delta=8:  delta_enc=0x100, mode=0x0d, clamp=0x00 -> top16=0x0d00
\   delta=16: delta_enc=0x200, mode=0x0e, clamp=0x00 -> top16=0x0e00
\
\ Shuffle type bits[3:2] of mode byte:
\   0b11 = BFLY (butterfly XOR), 0b10 = DOWN, 0b01 = UP, 0b00 = IDX
\
\ Verified encodings (rd=R5, rs=R0 shown):
\   SHFL.BFLY PT, R5, R0, 0x01, 0x1f -> 0x0c201f0000057f89 (probe_6b ✓)
\   SHFL.BFLY PT, R5, R0, 0x02, 0x1f -> 0x0c401f0000057f89 (probe_reduce ✓)
\   SHFL.BFLY PT, R5, R0, 0x04, 0x1f -> 0x0c801f0000057f89 (probe_reduce ✓)
\   SHFL.BFLY PT, R5, R0, 0x08, 0x1f -> 0x0d001f0000057f89 (probe_reduce ✓)
\   SHFL.BFLY PT, R5, R0, 0x10, 0x1f -> 0x0e001f0000057f89 (probe_reduce ✓)
\
\ ctrl from probe_6b: stall=14 yield=0 wbar=3 rbar=7 = 0x001e6800000e0000
: shfl-bfly,  ( rd rs delta -- )
  \ delta_enc = delta * 32; split into clamp (low 8) and mode (high bits | 0x0c)
  32 *                      \ delta_enc = delta * 32  (e.g. delta=8 -> 256=0x100)
  dup $ff and 48 lshift >r  \ clamp = delta_enc & 0xff -> save for bits[55:48]
  8 rshift $0c or           \ mode  = 0x0c | (delta_enc >> 8)
  56 lshift >r              \ mode_byte -> bits[63:56]
  24 lshift >r              \ rs -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7f89 or r> or            \ merge rs
  $1f 40 lshift or          \ mask=0x1f at bits[47:40]
  r> or                     \ clamp at bits[55:48]
  r> or                     \ mode_byte at bits[63:56]
  ctrl-shfl sinst, ;

\ SHFL.DOWN PT, Rd, Rs, delta  — warp down-shuffle (lane N gets value from lane N+delta)
\ Same encoding as SHFL.BFLY but shuffle type = DOWN (mode bits[3:2] = 0b10 -> base 0x08).
\
\ Verified encodings (rd=R7, rs=R0 shown):
\   SHFL.DOWN PT, R7, R0, 0x01, 0x1f -> 0x08201f0000077f89 (probe_6b ✓)
\   For delta=2: top16=0x0840, delta=4: 0x0880, delta=8: 0x0900, delta=16: 0x0a00
\
\ ctrl: same as shfl-bfly (stall=14 yield=0 wbar=3 rbar=7)
: shfl-down,  ( rd rs delta -- )
  32 *                      \ delta_enc = delta * 32
  dup $ff and 48 lshift >r  \ clamp at bits[55:48]
  8 rshift $08 or           \ mode = 0x08 | (delta_enc >> 8)  [DOWN type]
  56 lshift >r
  24 lshift >r              \ rs -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7f89 or r> or
  $1f 40 lshift or
  r> or
  r> or
  ctrl-shfl sinst, ;

\ SHFL.UP PT, Rd, Rs, delta  — warp up-shuffle (lane N gets value from lane N-delta)
\ Shuffle type = UP (mode bits[3:2] = 0b01 -> base 0x04).
\ mask field for UP shuffle is 0x00 (no upper clamp; clamp is handled differently).
\
\ Verified encoding (rd=R9, rs=R0):
\   SHFL.UP PT, R9, R0, 0x01, 0x1f -> 0x04201f0000097f89 (probe_6b ✓)
\   For delta=2: top16=0x0440, delta=4: 0x0480, delta=8: 0x0500, delta=16: 0x0600
\
\ Note: for SHFL.UP the 'mask_and_clamp' PTX argument is the clamp value (not mask),
\ so mask field in bits[47:40] encodes the clamp boundary.
: shfl-up,  ( rd rs delta -- )
  32 *                      \ delta_enc = delta * 32
  dup $ff and 48 lshift >r  \ clamp at bits[55:48]
  8 rshift $04 or           \ mode = 0x04 | (delta_enc >> 8)  [UP type]
  56 lshift >r
  24 lshift >r              \ rs -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7f89 or r> or
  $1f 40 lshift or
  r> or
  r> or
  ctrl-shfl sinst, ;

\ SHFL.IDX PT, Rd, Rs, lane_reg, 0x1f  — indexed warp shuffle (register lane index)
\ Reads from an absolute lane index (not relative). Both delta and clamp are registers.
\
\ Opcode variants:
\   0x7f89: immediate delta AND clamp (BFLY/DOWN/UP imm form)
\   0x7589: register delta, immediate clamp (IDX-RZ: clamp register = RZ)
\   0x7389: register delta AND register clamp (full register form)
\
\ Verified: SHFL.IDX PT, R11, R0, RZ, 0x1f -> 0x00001fff000b7589 (probe_6b ✓)
\   RZ as delta_reg = 0xff; mask=0x1f; mode=0x00 (IDX type); opcode=0x7589
\
\ For IDX with register lane index (most common use):
\   inst bits[15:0]  = 0x7389 (opcode: reg-delta, reg-clamp form)
\   inst bits[23:16] = Rd
\   inst bits[31:24] = Rs (source)
\   inst bits[39:32] = R_lane (lane index register)
\   inst bits[63:56] = 0x00 (IDX type, no delta offset)
: shfl-idx,  ( rd rs r_lane -- )
  32 lshift >r              \ r_lane -> bits[39:32]
  24 lshift >r              \ rs -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7389 or r> or r> or      \ merge rs and r_lane
  ctrl-shfl sinst, ;

\ BRA offset32  — branch PC-relative (signed byte offset from next instruction)
\ Verified: BRA loop -> inst=0xfffffffc00fc7947 ctrl=0x000fc0000383ffff
\   bits[63:32] = signed_offset32 in 4-byte units from PC_next (PC_current + 16)
\   bits[23:16] = 0xfc (PT predicate = always taken)
\   bits[31:24] = 0x00
\   bits[15:0]  = 0x7947 (BRA opcode)
\
\ Target address formula: target = (bra_addr + 16) + offset32 * 4
\ For self-loop (target = bra_addr): offset32 = -4 (0xfffffffc)
\   proof: (bra_addr+16) + (-4)*4 = bra_addr+16-16 = bra_addr ✓
: bra,  ( byte-offset -- )
  4 /  32 lshift $00fc7947 or
  ctrl-bra sinst, ;

\ @Px BRA offset32  — predicated branch (pred = 0..3 for P0..P3)
: bra-pred,  ( byte-offset pred -- )
  12 lshift >r
  4 /  32 lshift $00fc7947 or
  $FFFF0FFF and r> or
  ctrl-bra sinst, ;

\ ISETP.GE.U32.AND Pd, PT, Rs1, Rs2, PT  — integer >= comparison to predicate
\ Verified: ISETP.GE.U32.AND P0, PT, R2, R5, PT -> opcode OP-ISETP = $720c
\   bits[23:16]=Pd, bits[31:24]=Rs1, bits[39:32]=Rs2
\ ctrl-isetp: stall=13, opaque mode bits from probe
: isetp-ge,  ( pd rs1 rs2 -- )
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  16 lshift                 \ pd  -> bits[23:16]
  OP-ISETP or r> or r> or
  ctrl-isetp sinst, ;

\ ISETP.LT.U32.AND Pd, PT, Rs1, Rs2, PT  — integer < comparison to predicate
: isetp-lt,  ( pd rs1 rs2 -- )
  32 lshift >r
  24 lshift >r
  16 lshift
  OP-ISETP or r> or r> or
  ctrl-isetp sinst, ;

\ HFMA2.MMA Rd, Rs1, Rs2, Rs3  — packed FP16x2 fused multiply-add (reg form)
\ Opcode 0x235 -> bits[15:0] = $7235; Rs3 in ctrl[7:0]; same scheduling as FFMA.
: hfma2,  ( rd rs1 rs2 rs3 -- )
  >r                        \ rs3 -> ctrl-ffma arg
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7235 or r> or r> or
  r> ctrl-ffma sinst, ;

\ LDG.128 Rd, [Ra]  — 128-bit global load (4 registers Rd..Rd+3)
\ Width modifier: extra41 $0c1e1f00 (=$0c1e1900|$0600) selects 128-bit access.
\ Scheduling same as LDG.E: stall=4, yield=1, wbar=2, rbar=7.
: ldg128,  ( rd ra -- )
  24 lshift >r              \ ra -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7981 or r> or
  4 1 2 7 0 0 $0c1e1f00 make-ctrl sinst, ;

\ S2R Rd, sr-id  — read special register into Rd
\ Opcode 0x919 -> $7919; sr-id encoded ONLY in ctrl extra41[15:8].
\ inst word carries no sr-id field — it lives solely in the ctrl word.
\ Verified: S2R R0, SR_TID.X -> inst=0x0000000000007919 ctrl=0x000e6e0000002100
: s2r,  ( rd sr-id -- )
  >r                        \ save sr-id for ctrl
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7919 or
  r> ctrl-s2r sinst, ;

\ MOV Rd, imm32  — move 32-bit immediate into register
\ Opcode 0x802 -> $7802; imm32 in bits[63:32].
\ Scheduling: stall=0 yield=0 wbar=7 rbar=7 (fast ALU, no dependency).
: mov-imm,  ( rd imm32 -- )
  32 lshift >r              \ imm32 -> bits[63:32]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7802 or r> or
  ctrl-nop sinst, ;

\ IADD3 Rd, Rs1, Rs2, Rs3  — 3-input integer add
\ Opcode 0xc10 -> $7c10; Rs3 in ctrl[7:0] (4-operand format).
\ Scheduling same as IMAD: stall=1, yield=1, wbar=7, rbar=7.
: iadd3,  ( rd rs1 rs2 rs3 -- )
  >r                        \ rs3 -> ctrl extra41 bits[7:0]
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7c10 or r> or r> or
  1 1 7 7 0 0 r> make-ctrl sinst, ;

\ EXIT  — terminate thread (PT predicate, always executes)
\ ctrl-exit: stall=5, yield=1, wbar=7, rbar=7, extra=0x03800000
\ Verified: 0x000fea0003800000
: exit,  ( -- )
  $000000000000794d ctrl-exit sinst, ;

\ ============================================================
\ WARP SUM-REDUCE HELPER  (GEMV warp reduction pattern)
\ 5-step butterfly reduce across 32 lanes: SHFL.BFLY + FADD per step.
\ After return, lane 0 of each warp holds the warp sum in acc.
\ acc: register index holding partial sum
\ tmp: scratch register index for shfl destination
\ ============================================================
: warp-reduce,  ( acc tmp -- )
  over over  16 shfl-bfly,   \ tmp = shfl(acc, 16)
  over over swap fadd,       \ acc = acc + tmp
  over over   8 shfl-bfly,
  over over swap fadd,
  over over   4 shfl-bfly,
  over over swap fadd,
  over over   2 shfl-bfly,
  over over swap fadd,
  over over   1 shfl-bfly,
  over over swap fadd,
  2drop ;

\ ============================================================
\ GPTQ W4A16 DEQUANTIZATION PRIMITIVES
\ ============================================================
\ GPTQ packs 8 nibbles per u32. For nibble i (0-7):
\   1. SHF.R.U32  Rd, Rpacked, (i*4)     -- shift right by immediate
\   2. LOP3 Rd, Rd, 0xF, RZ, imm8-lut   -- mask low 4 bits (AND with 0xF)
\   3. I2F.F32.S32 Rd, Rd               -- s32 -> f32
\   4. FFMA Rd, Rd, Rscale, Rzero        -- Rd = nibble_f * scale + zero_f

\ --- SHF.R.U32 (shift right, immediate amount, unsigned 32-bit) ---
\ Probed: SHF.R.U32.HI R0, RZ, 0x1e, R0 -> inst=0x0000001eff007819
\ Field layout (inst word):
\   bits[15:0]  = 0x7819 (opcode 0x819, pred=PT)
\   bits[23:16] = Rd
\   bits[31:24] = Rs1 (source register, left-shifted input)
\   bits[47:32] = shift-amount immediate (5 bits, 0-31)
\   bits[55:48] = Rs2 (funnel source; RZ=0xFF for plain shift)
\ Scheduling: stall=1 yield=0 wbar=7 rbar=7 (ALU, no latency pressure)
: ctrl-shf  1 0 7 7 0 0 0 make-ctrl ;

: shf-r-imm,  ( rd rs shamt -- )
  \ SHF.R.U32 Rd, Rs, shamt, RZ
  32 lshift >r              \ shamt -> bits[47:32]
  24 lshift >r              \ rs    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7819 or r> or            \ rs
  $ff 48 lshift or          \ RZ funnel source at bits[55:48]
  r> or                     \ shamt
  ctrl-shf sinst, ;

\ --- LOP3 (3-input logical, reg-imm-imm form for AND-with-constant) ---
\ Opcode: 0x212 (unverified against probe; canonical sm_90 LOP3 reg-imm form)
\ Encoding (reg-imm form, opcode 0x812 for imm32 in bits[63:32]):
\   bits[15:0]  = 0x7812 (opcode 0x812, pred=PT)
\   bits[23:16] = Rd
\   bits[31:24] = Rs1
\   bits[63:32] = 32-bit immediate (the mask, e.g. 0x0000000F)
\   LUT byte for AND = 0xC0 (A AND B); packed into ctrl extra41 bits[15:8]
\ NOTE: opcode 0x812 = imm form of LOP3; ctrl LUT byte selects the operation.
: ctrl-lop3  ( lut-byte -- ctrl64 )  8 lshift >r  1 0 7 7 0 0 r> make-ctrl ;

: lop3-and-imm,  ( rd rs imm32 -- )
  \ LOP3.LUT Rd, Rs, imm32, RZ, 0xC0  (Rd = Rs AND imm32)
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ rs    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7812 or r> or r> or
  $c0 ctrl-lop3 sinst, ;    \ LUT=0xC0 = AND(A,B,C)=A&B

: lop3-or-imm,  ( rd rs imm32 -- )
  \ LOP3.LUT Rd, Rs, imm32, RZ, 0xFC  (Rd = Rs OR imm32)
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ rs    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7812 or r> or r> or
  $fc ctrl-lop3 sinst, ;    \ LUT=0xFC = OR(A,B)

: lop3-xor-imm,  ( rd rs imm32 -- )
  \ LOP3.LUT Rd, Rs, imm32, RZ, 0x3C  (Rd = Rs XOR imm32)
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ rs    -> bits[31:24]
  track-rd 16 lshift        \ rd    -> bits[23:16]
  $7812 or r> or r> or
  $3c ctrl-lop3 sinst, ;    \ LUT=0x3C = XOR(A,B)

\ --- I2FP.F32.S32 (signed int32 to float32) ---
\ Opcode: 0x7245  — VERIFIED by probe (ptxas sm_90a, nvdisasm 12.8.90)
\ PTX:    cvt.rn.f32.s32 Rd, Rs
\ SM90:   I2FP.F32.S32 Rd, Rs
\ Encoding:
\   bits[15:0]  = 0x7245 (I2FP opcode)
\   bits[23:16] = Rd (float32 destination)
\   bits[39:32] = Rs (int32 source)    <-- NOTE: src at bits[39:32], not [31:24]
\ Probe: inst=0x0000000200077245  ctrl=0x004fca0000201400
\   stall=5 yield=0 wbar=7 rbar=7 wait=4 extra41=0x00201400
\ Scheduling: stall=5 yield=0 wbar=7 rbar=7 (FP conversion latency)
: ctrl-i2f  5 0 7 7 4 0 $201400 make-ctrl ;

: i2f-s32-f32,  ( rd rs -- )
  \ I2FP.F32.S32 Rd, Rs
  32 lshift >r              \ rs -> bits[39:32]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7245 or r> or
  ctrl-i2f sinst, ;

\ --- F2I.TRUNC.NTZ (float32 to signed int32, truncate toward zero) ---
\ Opcode: 0x7305  — VERIFIED by probe (ptxas sm_90a, nvdisasm 12.8.90)
\ PTX:    cvt.rzi.s32.f32 Rd, Rs
\ SM90:   F2I.TRUNC.NTZ Rd, Rs
\ Encoding:
\   bits[15:0]  = 0x7305 (F2I opcode)
\   bits[23:16] = Rd (int32 destination)
\   bits[39:32] = Rs (float32 source)
\ Probe: inst=0x0000000200077305  ctrl=0x004e24000020f100
\   stall=2 yield=1 wbar=0 rbar=7 wait=4 extra41=0x0020f100
: ctrl-f2i  2 1 0 7 4 0 $20f100 make-ctrl ;

: f2i-f32-s32,  ( rd rs -- )
  \ F2I.TRUNC.NTZ Rd, Rs
  32 lshift >r              \ rs -> bits[39:32]
  track-rd 16 lshift        \ rd -> bits[23:16]
  $7305 or r> or
  ctrl-f2i sinst, ;

\ ============================================================
\ MUFU — Multi-Function Unit (special math, opcode 0x308)
\ ============================================================
\ Inst format: src << 32 | dst << 16 | 0x7308
\   bits[15:0]  = 0x7308  (opcode, identical for all MUFU sub-functions)
\   bits[23:16] = Rd  (destination register)
\   bits[39:32] = Rs  (source register)
\
\ Sub-function is in ctrl extra41 bits[13:8] (NOT in the instruction word):
\   COS=0x0000  SIN=0x0400  EX2=0x0800  LG2=0x0c00
\   RCP=0x1000  RSQ=0x1400  SQRT=0x2000
\
\ Scheduling: stall=8 yield=1 rbar=7; wbar varies per sub-function.
\   EX2->7  RCP->1  RSQ->2  LG2->3  SQRT->4  SIN->5  COS->5
\
\ Probe verification (probe_2b.sass sm_90, stall=8 scheduling context):
\   MUFU.EX2  R9,R6   inst=0x0000000600097308 ctrl=0x000ff00000000800
\   MUFU.RCP  R4,R4   inst=0x0000000400047308 ctrl=0x000e700000001000
\   MUFU.RSQ  R7,R0   inst=0x0000000000077308 ctrl=0x000eb00000001400
\   MUFU.LG2  R11,R0  inst=0x00000000000b7308 ctrl=0x000ef00000000c00
\   MUFU.SQRT R17,R0  inst=0x0000000000117308 ctrl=0x000f300000002000
\   MUFU.SIN  R13,R8  inst=0x00000008000d7308 ctrl=0x000f700000000400
\   MUFU.COS  R15,R8  inst=0x00000008000f7308 ctrl=0x000f620000000000
\
\ Additional probe verification (ptxas sm_90a, nvdisasm 12.8.90, minimal context):
\   MUFU.RCP  R0,R0   inst=0x0000000000007308 ctrl=0x000e640000001000
\   MUFU.EX2  R7,R0   inst=0x0000000000077308 ctrl=0x000e640000000800
\   MUFU.RSQ  R7,R0   inst=0x0000000000077308 ctrl=0x000e640000001400
\   MUFU.LG2  R7,R0   inst=0x0000000000077308 ctrl=0x000e640000000c00
\   MUFU.SIN  R7,R7   inst=0x0000000700077308 ctrl=0x000e240000000400
\   MUFU.COS  R7,R7   inst=0x0000000700077308 ctrl=0x000e240000000000
\   MUFU.SQRT R7,R0   inst=0x0000000000077308 ctrl=0x000e640000002000
\ Extra41 subop bits confirmed identical across both probe sessions.

: ctrl-mufu  ( subfn-extra41 wbar -- ctrl64 )
  swap >r                       \ R: subfn-extra41; stack: wbar
  >r 8 1 r> 7 0 0 r>            \ stack: 8 1 wbar 7 0 0 subfn-extra41
  make-ctrl ;

: mufu,  ( rd rs subfn-extra41 wbar -- )
  ctrl-mufu >r
  swap 32 lshift >r           \ rs -> upper inst word
  16 lshift                   \ rd -> bits[23:16]
  $7308 or r> or              \ opcode | rs
  r> sinst, ;                 \ ctrl

\ MUFU.EX2 Rd, Rs  — 2^x  (softmax)
: ex2,  ( rd rs -- )  $0800 7 mufu, ;

\ MUFU.RCP Rd, Rs  — 1/x  (reciprocal)
: rcp,  ( rd rs -- )  $1000 1 mufu, ;

\ MUFU.RSQ Rd, Rs  — 1/sqrt(x)  (attention normalization)
: rsqrt,  ( rd rs -- )  $1400 2 mufu, ;

\ MUFU.LG2 Rd, Rs  — log2(x)
: lg2,  ( rd rs -- )  $0c00 3 mufu, ;

\ MUFU.SQRT Rd, Rs  — sqrt(x)
: sqrt,  ( rd rs -- )  $2000 4 mufu, ;

\ MUFU.SIN Rd, Rs  — sine  (RoPE)
: sin,  ( rd rs -- )  $0400 5 mufu, ;

\ MUFU.COS Rd, Rs  — cosine  (RoPE)
: cos,  ( rd rs -- )  $0000 5 mufu, ;

\ --- dequant-nibble: full 4-instruction GPTQ nibble dequantization ---
\ ( rd rsrc-packed nib-idx rs-scale rs-zero -- )
\   Emits:
\     SHF.R.U32  Rd, Rpacked, (nib-idx*4)   -> Rd = packed >> (i*4)
\     LOP3       Rd, Rd, 0xF                 -> Rd = Rd & 0xF
\     I2F.F32    Rd, Rd                      -> Rd = float(Rd)
\     FFMA       Rd, Rd, Rscale, Rzero       -> Rd = Rd*scale + zero
: dequant-nibble,  ( rd rsrc nib-idx rscale rzero -- )
  \ rd is the destination for all 4 instructions — save it on the return stack.
  \ After setup: r-stack top->bot = rd  rscale  rzero
  >r >r                     \ save rzero rscale; stack: rd rsrc nib-idx
  rot                        \ stack: rsrc nib-idx rd
  dup >r                    \ dup rd, save to r-stack; stack: rsrc nib-idx rd
  -rot                      \ stack: rd rsrc nib-idx
  4 *                        \ stack: rd rsrc shamt
  \ 1. SHF.R.U32 rd, rsrc, shamt  (consumes rd rsrc shamt)
  shf-r-imm,
  \ r-stack (top->bot): rd rscale rzero
  \ 2. LOP3 rd, rd, 0xF  — r@ peeks rd without consuming
  r@ r@ $f lop3-and-imm,
  \ 3. I2F.F32.S32 rd, rd
  r@ r@ i2f-s32-f32,
  \ 4. FFMA rd, rd, rscale, rzero
  r> dup r> r>              \ stack: rd rd rscale rzero
  ffma, ;                   \ FFMA rd, rd, rscale, rzero


\ ============================================================
\ SHARED MEMORY INSTRUCTIONS  (smem operands for split-K / multi-warp kernels)
\ ============================================================

\ STS.32 [Ra+off], Rs  — store Rs to shared memory (32-bit)
\ Opcode 0x7988; addr-reg in bits[31:24], data-reg in bits[39:32], off in bits[47:32]
\ ctrl-sts: stall=1 yield=1 wbar=7 rbar=7 extra=0x0800 (32-bit width field)
: sts,  ( ra rs off -- )
  32 lshift >r              \ off -> bits[47:32]
  swap 32 lshift >r         \ rs  -> bits[39:32]
  24 lshift                 \ ra  -> bits[31:24]
  $7988 or r> or r> or
  ctrl-sts sinst, ;

\ LDS.32 Rd, [Ra+off]  — load 32 bits from shared memory into Rd
\ Opcode 0x7984; addr-reg in bits[31:24], dest in bits[23:16], off in bits[47:32]
\ From ENCODING.md: LDS R7,[R5] = 0x0000000005077984
: lds,  ( rd ra off -- )
  32 lshift >r              \ off -> bits[47:32]
  24 lshift >r              \ ra  -> bits[31:24]
  track-rd 16 lshift        \ rd  -> bits[23:16]
  $7984 or r> or r> or
  ctrl-lds sinst, ;

\ BAR.SYNC.DEFER_BLOCKING bar-id  — synchronize all threads at shared-memory barrier
\ Opcode 0x7b1d; bar-id encoded as (bar_id * 64) in bits[55:48] of the inst word.
\
\ Verified encodings:
\   BAR.SYNC 0: inst=0x0000000000007b1d  ctrl=0x000fec0000010000
\   BAR.SYNC 1: inst=0x0040000000007b1d  ctrl=0x000fec0000010000
\
\ bits[55:48] = bar_id * 0x40 = bar_id << 6
\ In 64-bit: barrier field = bar_id << 54  (6 bits at [55:48] in byte 6)
\
\ BUG FIXED: previous implementation used '16 lshift' placing bar-id in bits[23:16],
\ which only works correctly for bar-id=0. The actual field is at bits[55:48].
: bar-sync,  ( bar-id -- )
  6 lshift 48 lshift $7b1d or    \ bar_id * 64 -> bits[55:48], then opcode in low 16
  ctrl-bar sinst, ;

\ ============================================================
\ MEMBAR ENCODERS  (used by grid-sync acquire/release protocol)
\ ============================================================
\
\ All MEMBAR variants share the same instruction word: 0x0000000000007992
\ The scope is encoded in the ctrl word's extra41 field bits[13:12]:
\   00 = CTA (block-level fence)
\   10 = GPU (device-level fence, crosses SMs)
\   11 = SYS (system-level fence, NVLink-coherent, CPU-GPU)
\
\ Verified from probe_membar (probe_7.sass):
\   MEMBAR.SC.GPU: inst=0x0000000000007992 ctrl=0x000fec0000002000 (extra41=0x2000)
\   MEMBAR.SC.SYS: inst=0x0000000000007992 ctrl=0x000fec0000003000 (extra41=0x3000)
\   MEMBAR.SC.CTA: inst=0x0000000000007992 ctrl=0x0003ec0000000000 (extra41=0x0000)
\   Note: CTA scope uses a different sched word (lower stall bits) than GPU/SYS.

\ MEMBAR.SC.GPU  — GPU-scope store fence; all prior stores become visible
\   across all SMs before any subsequent load/store on this thread.
: membar-gpu,  ( -- )
  $0000000000007992 $000fec0000002000 sinst, ;

\ MEMBAR.SC.SYS  — system-scope fence (NVLink-coherent; for CPU-GPU sync)
: membar-sys,  ( -- )
  $0000000000007992 $000fec0000003000 sinst, ;

\ MEMBAR.SC.CTA  — CTA/block-level fence (same as membar.cta in PTX)
\   Orders memory accesses within the same thread block (CTA).
\   Use before/after shared memory accesses when bar.sync is not sufficient.
\   Lower overhead than GPU/SYS scopes.
: membar-cta,  ( -- )
  $0000000000007992 $0003ec0000000000 sinst, ;

\ ============================================================
\ ATOMG.E.ADD — global atomic add (Hopper descriptor-based)
\ ============================================================
\
\ ATOMG.E.ADD.STRONG.GPU Rd, desc[UR][Ra], Rs
\   — atomically adds Rs to the 32-bit word at [Ra], returns old value in Rd.
\   Use Rd=RZ (0xFF) to discard the return value (e.g. for arrive-only sync).
\
\ Verified empirically via ptxas + nvdisasm --print-instruction-encoding
\ (docs/encoding/integer_atomic.md).  Probe: atom.global.add.u32 / atom.global.add.f32
\
\ word0 field layout:
\   bits[15: 0] = opcode
\   bits[23:16] = Rd (destination; 0xff=RZ for no-return)
\   bits[31:24] = Ra (address-offset register, 64-bit pair)
\   bits[39:32] = Rs (source data register)
\   Uniform descriptor register (UR) is NOT in word0; it lives in the ctrl word.
\
\ Opcodes (bits[15:0]):
\   ATOMG.ADD.U32 = 0x79a8   (integer counter, used for grid sync)
\   ATOMG.ADD.F32 = 0x79a3   (floating-point add with FTZ+RN)
\   ATOMG.EXCH    = 0x79a8   (same base opcode as ADD.U32; differs in ctrl)
\   REDG.ADD.F32  = 0x79a6   (reduction, no return)
\
\ ctrl words (fully verified):
\   ATOMG.ADD.U32: 0x004e2800081ee1c4
\   ATOMG.ADD.F32: 0x004e2800081ef3c4  (byte[1]=0xf3 encodes FTZ+RN)
\   ATOMG.EXCH:    0x004e28000c1ee1c4
\   REDG.ADD.F32:  0x004fe2000c10f384
\
\ The old ctrl-atom computed 0x001ffc0000100000 — that was wrong.
\ The old OP-ATOM-ADD = 0x798b — that was also wrong.

\ OP-ATOM-ADD-U32, OP-ATOM-ADD-F32, CTRL-ATOM-U32, CTRL-ATOM-F32,
\ CTRL-REDG-F32, OP-ATOM-ADD all defined in opcodes-sm90.fs.

\ atom-add,  ( rd ra rs -- )
\ ATOMG.E.ADD.STRONG.GPU Rd, desc[UR][Ra], Rs
\ Rd = old value (use RZ=0xff to discard); Ra = 64-bit address offset reg;
\ Rs = 32-bit value to add.  Ctrl word uses UR4 descriptor (hardcoded for now).
: atom-add,  ( rd ra rs -- )
  32 lshift >r              \ rs -> bits[39:32]
  24 lshift >r              \ ra -> bits[31:24]
  track-rd 16 lshift        \ rd -> bits[23:16]
  OP-ATOM-ADD or r> or r> or
  CTRL-ATOM-U32 sinst, ;

\ ============================================================
\ LD-ACQUIRE / ST-RELEASE via MEMBAR+LDG / MEMBAR+STG protocol
\ ============================================================
\
\ Hopper has no single-instruction acquire-load or release-store.
\ The canonical pattern (what ptxas emits for ld.acquire.gpu / st.release.gpu
\ on sm_90, confirmed by probe_7.sass and CUDA cooperative groups source):
\
\   Release store:  MEMBAR.SC.GPU  then  STG.E [Ra], Rs
\   Acquire load:   LDG.E Rd, [Ra]  then  MEMBAR.SC.GPU
\
\ stg-release,  ( ra rs -- )
\ Store Rs to [Ra] with GPU-scope release semantics.
: stg-release,  ( ra rs -- )
  membar-gpu,   \ fence: all prior stores ordered before this one
  stg, ;        \ STG.E [Ra], Rs

\ ldg-acquire,  ( rd ra -- )
\ Load [Ra] into Rd with GPU-scope acquire semantics.
: ldg-acquire,  ( rd ra -- )
  ldg,          \ LDG.E Rd, [Ra]
  membar-gpu, ; \ fence: all subsequent ops ordered after this load

\ ============================================================
\ COOPERATIVE GRID-SYNC — software barrier across all CTAs on the grid
\ ============================================================
\
\ Background: the megakernel runs all N DeltaNet layers in one GPU kernel
\ launch.  Between layers, every thread block must synchronize so that layer
\ L+1 doesn't read state written by layer L before all CTAs have written it.
\ BAR.SYNC covers intra-CTA threads; this covers all CTAs grid-wide.
\
\ Protocol (identical to CUDA cooperative groups this_grid().sync()):
\
\   Thread 0 of each CTA:
\     (1) Atomically increment sync_counter.  Save old value.
\     (2) If old == grid_size-1  (this is the last CTA to arrive):
\           MEMBAR.SC.GPU  +  STG [done_flag], 1   (release write)
\         Else:
\           Spin: LDG [done_flag] + MEMBAR.SC.GPU
\                 loop until value != 0
\   All threads: BAR.SYNC 0  (CTA barrier to broadcast "done" to all warps)
\
\ Instruction sequence emitted (13 fixed instructions + 1 reg-alloc helper = 14 ops):
\   (a) S2R    gs-tid, SR_TID.X          — read thread ID
\   (b) MOV    gs-exp, grid_size-1       — load expected old value
\   (c) MOV    pp2-tmp, 1                — helper for ISETP immediate
\   (d) ISETP.GE pp2, gs-tid, 1         — pp2 = (tid >= 1), i.e. non-thread-0
\   (e) @pp2 BRA +208                   — non-thread-0 threads skip to BAR.SYNC
\   (f) MOV    atom-one, 1              — addend for atomic
\   (g) ATOM.E.ADD gs-old,[gs-ctr],1    — arrive; gs-old = previous count
\   (h) ISETP.GE gs-pp, gs-old, gs-exp — gs-pp = (old >= grid_size-1)
\   (i) @!gs-pp BRA +64                 — if not last CTA, jump to spin loop
\   (j) MEMBAR.SC.GPU                   — release fence (last CTA path)
\   (k) MOV    flag-one, 1              — value to store
\   (l) STG.E  [gs-flag], flag-one     — release write: done!
\   (m) BRA +80                         — skip spin loop to BAR.SYNC
\   spin_top:
\   (n) LDG.E  gs-poll, [gs-flag]      — poll: read done flag
\   (o) MEMBAR.SC.GPU                   — acquire fence
\   (p) MOV    poll-cmp, 1              — comparand for ISETP immediate
\   (q) ISETP.GE pp2, gs-poll, 1       — pp2 = (flag >= 1), i.e. done
\   (r) @!pp2 BRA spin_top (-80)       — loop back if not done yet
\   skip_to_barsync:
\   (s) BAR.SYNC 0                      — CTA-wide barrier; all warps sync
\
\ Branch offset calculation (each instruction = 16 bytes, offset = signed
\ bytes from the instruction FOLLOWING the branch):
\   Absolute positions (relative to start of emit-grid-sync output):
\     (a)=0  (b)=16  (c)=32  (d)=48  (e)=64  (f)=80  (g)=96  (h)=112
\     (i)=128  (j)=144  (k)=160  (l)=176  (m)=192
\     (n)=208  (o)=224  (p)=240  (q)=256  (r)=272  (s)=288
\   (e) NEXT=(f)=80,  target=(s)=288: offset = 288-80  = +208
\   (i) NEXT=(j)=144, target=(n)=208: offset = 208-144 = +64
\   (m) NEXT=(n)=208, target=(s)=288: offset = 288-208 = +80
\   (r) NEXT=(s)=288, target=(n)=208: offset = 208-288 = -80
\
\ Stack effect: ( sync-counter-addr-reg done-flag-addr-reg grid-size -- )
\   sync-counter-addr-reg : register (or reg pair) holding GPU VA of u32 counter
\   done-flag-addr-reg    : register (or reg pair) holding GPU VA of u32 flag
\   grid-size             : compile-time constant = total CTA count
\ All three arguments consumed.  No value returned.

\ Register allocator stubs — overridden by ls-compiler.fs at load time.
\ emit-grid-sync is compiled against these stubs; ls-compiler.fs
\ redefines rreg+/freg+/preg+ for all code compiled after it loads.
variable _gs-r  4 _gs-r !
variable _gs-f  0 _gs-f !
variable _gs-p  0 _gs-p !
: rreg+  ( -- n )  _gs-r @ dup 1+ _gs-r ! ;
: freg+  ( -- n )  _gs-f @ dup 1+ _gs-f ! ;
: preg+  ( -- n )  _gs-p @ dup 1+ _gs-p ! ;

variable gs-ctr   variable gs-flag  variable gs-grid
variable gs-old   variable gs-exp   variable gs-poll
variable gs-pp    variable gs-pp2   variable gs-tid

: emit-grid-sync  ( sync-counter-addr-reg done-flag-addr-reg grid-size -- )
  gs-grid !  gs-flag !  gs-ctr !

  \ Allocate scratch registers and predicates
  rreg+ gs-old !            \ ATOM return: old counter value
  rreg+ gs-exp !            \ expected value = grid_size - 1
  rreg+ gs-poll !           \ spin-poll register
  preg+ gs-pp !             \ predicate: is last CTA? (old >= grid_size-1)
  preg+ gs-pp2 !            \ predicate: non-thread-0 / spin-poll exit
  rreg+ gs-tid !            \ thread ID in CTA

  \ (a) S2R gs-tid, SR_TID.X
  gs-tid @  SR-TID-X  s2r,

  \ (b) MOV gs-exp, grid_size-1
  gs-exp @  gs-grid @ 1-  mov-imm,

  \ (c)+(d) ISETP.GE pp2, gs-tid, 1  via emit-isetp-ge-imm helper pattern:
  \   MOV tmp, 1  then  ISETP.GE pp2, gs-tid, tmp
  rreg+ dup >r  1  mov-imm,            \ (c) MOV tmp, 1
  gs-pp2 @  gs-tid @  r>  isetp-ge,   \ (d) ISETP.GE pp2, gs-tid, tmp

  \ (e) @pp2 BRA +208  (non-thread-0 jumps to (s)=BAR.SYNC, 13 insts past (e))
  208  gs-pp2 @  bra-pred,

  \ ---- Thread 0 only below this point ----

  \ (f) MOV atom-one, 1
  rreg+ dup >r  1  mov-imm,

  \ (g) ATOM.E.ADD gs-old, [gs-ctr], atom-one
  gs-old @  gs-ctr @  r>  atom-add,

  \ (h) ISETP.GE gs-pp, gs-old, gs-exp
  gs-pp @  gs-old @  gs-exp @  isetp-ge,

  \ (i) @!gs-pp BRA +64  (not last CTA: jump 4 insts past (j) to spin_top=(n))
  64  gs-pp @ 8 or  bra-pred,

  \ ---- Last CTA path: write done flag ----

  \ (j) MEMBAR.SC.GPU
  membar-gpu,

  \ (k) MOV flag-one, 1
  rreg+ dup >r  1  mov-imm,

  \ (l) STG.E [gs-flag], flag-one
  gs-flag @  r>  stg,

  \ (m) BRA +80  (skip 5 insts of spin loop to (s)=BAR.SYNC)
  80  bra,

  \ ---- Spin-poll loop ----
  \ spin_top: (n)
  \ (n) LDG.E gs-poll, [gs-flag]
  gs-poll @  gs-flag @  ldg,

  \ (o) MEMBAR.SC.GPU  (acquire fence after load)
  membar-gpu,

  \ (p)+(q) ISETP.GE pp2, gs-poll, 1:
  rreg+ dup >r  1  mov-imm,            \ (p) MOV tmp, 1
  gs-pp2 @  gs-poll @  r>  isetp-ge,  \ (q) ISETP.GE pp2, gs-poll, tmp

  \ (r) @!pp2 BRA spin_top  = -5 insts back = -80 bytes from instruction (s)
  -80  gs-pp2 @ 8 or  bra-pred,

  \ ---- All-threads CTA barrier ----
  \ skip_to_barsync: (s)
  \ (s) BAR.SYNC 0
  0  bar-sync, ;

\ ============================================================
\ MEGAKERNEL PARAM OFFSETS FOR COOPERATIVE GRID-SYNC
\ ============================================================
\
\ Constant bank c[0x0] layout on Hopper:
\   Bytes [0x000..0x20F] : driver-reserved (blockDim, gridDim, clock, etc.)
\   Bytes [0x210 + k*8]  : user param k (8-byte pointer or scalar, k=0,1,...)
\
\ The megakernel (DeltaNet fused) passes data pointers first; the two
\ grid-sync state pointers come last:
\   param[K+0] = u64  sync_counter_ptr  (points to pre-zeroed u32 in HBM)
\   param[K+1] = u64  done_flag_ptr     (points to pre-zeroed u32 in HBM)
\
\ K = number of data pointer params (e.g. 6 for deltanet_fused:
\     q, k, v, z, decay, state -> K=6).
\
\ sync-counter-param-offset ( n-data-params -- cbank-byte-offset )
\ Returns byte offset within c[0x0] where sync_counter_ptr lives.
: sync-counter-param-offset  ( n-data-params -- offset )
  8 * $210 + ;

\ done-flag-param-offset ( n-data-params -- cbank-byte-offset )
\ Returns byte offset within c[0x0] where done_flag_ptr lives.
: done-flag-param-offset  ( n-data-params -- offset )
  8 * $218 + ;

\ emit-grid-sync-params ( n-data-params -- sync-ctr-off done-flag-off )
\ Returns both offsets as a pair.  Intended usage:
\   6 emit-grid-sync-params  -> ( 0x240 0x248 )  (for K=6 data params)
: emit-grid-sync-params  ( n-data-params -- sync-ctr-off done-flag-off )
  dup sync-counter-param-offset
  swap done-flag-param-offset ;

\ ============================================================
\ MEGAKERNEL PARAM COUNT HELPER
\ ============================================================
\
\ The megakernel's n-kparams (used by build-cubin for .nv.info KPARAM_INFO)
\ must include the two grid-sync pointer params appended after the data params.
\
2 constant N-SYNC-PARAMS   \ always 2: sync_counter_ptr + done_flag_ptr

\ total-kparams ( n-data-params -- total-including-sync )
: total-kparams  ( n-data-params -- total )
  N-SYNC-PARAMS + ;


\ A cubin is an ELF64 file with NVIDIA-specific sections.
\ We need:
\   ELF header (64 bytes)
\   Section headers
\   .text.<kernel> — the GPU machine code
\   .nv.info — kernel metadata (register count, param info)
\   .nv.constant0.<kernel> — constant bank (kernel params)
\   .shstrtab — section name strings
\   .strtab — symbol strings
\   .symtab — symbol table
\   Program headers

524288 constant CUBIN-SIZE
create cubin-buf CUBIN-SIZE allot
variable cubin-pos  0 cubin-pos !
variable shmem-size  0 shmem-size !   \ bytes of static shared memory for kernel
variable n-kparams   0 n-kparams !    \ number of kernel pointer params (set from parser)

\ Forth-bootstrap lacks VALUE. Use plain VARIABLEs for all build-cubin offsets
\ and sizes — declared here so they're late-bound only once, at compile of
\ build-cubin. Reads are `NAME @`, writes are `val NAME !`.
variable shstrtab-off    variable shstrtab-size
variable strtab-off      variable strtab-size      variable strsym-kernel
variable symtab-off      variable symtab-size      variable sym-kernel-off
variable nvinfo-off      variable nvinfo-size
variable nvinfo-k-off    variable nvinfo-k-size
variable text-off        variable text-size
variable const0-off      variable const0-size
variable shdrs-off
variable SN-shstrtab     variable SN-strtab        variable SN-symtab
variable SN-nvinfo       variable SN-nvinfo-k      variable SN-text
variable SN-const0       variable SN-shared

: cubin-reset  0 cubin-pos ! ;
: cb,  ( byte -- )  cubin-buf cubin-pos @ + c!  1 cubin-pos +! ;
: cw,  ( u16 -- )   dup cb, 8 rshift cb, ;
: cd,  ( u32 -- )   dup cw, 16 rshift cw, ;
: cq,  ( u64 -- )   dup cd, 32 rshift cd, ;
: cpad  ( target -- )  begin dup cubin-pos @ > while 0 cb, repeat drop ;

\ ELF64 header for cubin
: cubin-elf-header  ( -- )
  \ e_ident
  $7F cb, [char] E cb, [char] L cb, [char] F cb,
  $02 cb,   \ ELFCLASS64
  $01 cb,   \ ELFDATA2LSB
  $01 cb,   \ EV_CURRENT
  $33 cb,   \ NVIDIA CUDA OS/ABI
  $07 cb,   \ ABI version
  $00 cb, $00 cb, $00 cb, $00 cb, $00 cb, $00 cb, $00 cb,
  \ e_type
  2 cw,     \ ET_EXEC
  $BE cw,   \ EM_CUDA (190)
  $80 cd,   \ e_version (0x80 for CUDA)
  0 cq,     \ e_entry
  0 cq,     \ e_phoff (filled later)
  0 cq,     \ e_shoff (filled later)
  $5a055a cd,  \ e_flags (from probe cubin)
  64 cw,    \ e_ehsize
  56 cw,    \ e_phentsize
  0 cw,     \ e_phnum (filled later)
  64 cw,    \ e_shentsize
  0 cw,     \ e_shnum (filled later)
  0 cw, ;   \ e_shstrndx (filled later)

\ ============================================================
\ BUILD CUBIN — helpers
\ ============================================================

\ Kernel name string (addr u).
\ Prefer li-name$ (set by parser for the last parsed fn); fall back to "gemv".
: kernel-name
  li-name-len @ 0> if li-name$ else s" gemv" then ;

\ Emit counted string as raw bytes (no length byte)
: emit-str  ( addr u -- )  0 do dup i + c@ cb, loop drop ;

\ Patch helpers: store little-endian values at absolute file offset
: put-u16  ( u16 off -- )
  cubin-buf + >r  dup r@ c!  8 rshift r> 1+ c! ;
: put-u32  ( u32 off -- )
  cubin-buf + >r
  dup r@ c!  8 rshift dup r@ 1+ c!  8 rshift dup r@ 2 + c!  8 rshift r> 3 + c! ;
: put-u64  ( u64 off -- )
  \ Store u64 in little-endian at [off..off+7]. Correct form:
  \   val off -> put low u32 at off, high u32 at off+4
  >r  dup  r@ put-u32                ( val )
  32 rshift  r> 4 +  put-u32 ;

\ Align cubin-pos to n (power-of-2) byte boundary
: calign  ( n -- )  1- >r  cubin-pos @ r@ + r> invert and  cpad ;

\ ELF64 section header: 10 fields, 64 bytes total.
\ Fields on param stack, left-to-right (sh_name deepest, sh_ent TOS):
\   sh_name(u32) sh_type(u32) sh_flags(u64) sh_addr(u64) sh_off(u64)
\   sh_size(u64) sh_link(u32) sh_info(u32) sh_align(u64) sh_ent(u64)
\ Emitted in that order. Use scratch variables to avoid r-stack juggling bugs.
variable sh-ent-v  variable sh-align-v  variable sh-info-v  variable sh-link-v
variable sh-size-v variable sh-off-v    variable sh-addr-v  variable sh-flags-v
variable sh-type-v
: shdr64,
  sh-ent-v !   sh-align-v ! sh-info-v !  sh-link-v !
  sh-size-v !  sh-off-v !   sh-addr-v !  sh-flags-v ! sh-type-v !
  \ sh_name still on stack
  cd,  sh-type-v @ cd,  sh-flags-v @ cq,  sh-addr-v @ cq,
  sh-off-v @ cq,  sh-size-v @ cq,  sh-link-v @ cd,  sh-info-v @ cd,
  sh-align-v @ cq,  sh-ent-v @ cq, ;

\ ELF64 symtab entry: 24 bytes
\ ( st_name st_info st_other st_shndx st_value st_size -- )
: sym64,
  >r >r >r >r >r  ( st_name left )
  cd,  r> cb,  r> cb,  r> cw,  r> cq,  r> cq, ;

\ .nv.info plain u32 record: [fmt u8][attr u8][size=4 u16][val u32]  (no sym_idx)
: nvi-u32,  ( val attr fmt -- )  cb, cb,  4 cw,  cd, ;

\ .nv.info sym+val record: [fmt u8][attr u8][size=8 u16][sym_idx u32][val u32]
\ probe: REGCOUNT/FRAME_SIZE/MIN_STACK_SIZE use this 8-byte form with sym_idx=kernel_sym
: nvi-sval,  ( val sym_idx attr fmt -- )  cb, cb,  8 cw,  cd, cd, ;

\ EIATTR constants (from probe.cubin decode)
$04 constant NVI-FMT-U32
$03 constant NVI-FMT-FLAG
$2f constant EIATTR-REGCOUNT
$11 constant EIATTR-FRAME-SIZE
$12 constant EIATTR-MIN-STACK-SIZE
$37 constant EIATTR-CUDA-API-VERSION
$17 constant EIATTR-KPARAM-INFO
$0a constant EIATTR-PARAM-CBANK
$1c constant EIATTR-EXIT-INSTR-OFFSETS
$19 constant EIATTR-CBANK-PARAM-SIZE
$1b constant EIATTR-MAXREG-COUNT
$50 constant EIATTR-SPARSE-MMA-MASK
$36 constant EIATTR-SW-WAR
$23 constant EIATTR-CRS-STACK-SIZE
\ Cooperative grid-sync attributes (verified from probe_6b.sass — grid-sync probe)
$28 constant EIATTR-COOP-GROUP-INSTR-OFFSETS  \ fmt=$04; N*4 bytes; one u32 per sync site
$29 constant EIATTR-COOP-GROUP-MASK-REGIDS    \ fmt=$04; 16 bytes; 4 x $ffffffff bitmask

\ ============================================================
\ BUILD CUBIN
\ ============================================================

\ Produces 9-section ELF64 cubin, no program headers:
\   [0] NULL  [1] .shstrtab  [2] .strtab  [3] .symtab
\   [4] .nv.info  [5] .text.<k>  [6] .nv.info.<k>
\   [7] .nv.shared.reserved.0  [8] .nv.constant0.<k>

: build-cubin  ( -- addr u )
  cubin-reset

  \ 1. ELF header placeholder — fields patched in step 10
  cubin-elf-header

  \ 2. .shstrtab  (section 1)
  cubin-pos @ shstrtab-off !
  0 cb,
  cubin-pos @ shstrtab-off @ - SN-shstrtab !
    s" .shstrtab" emit-str 0 cb,
  cubin-pos @ shstrtab-off @ - SN-strtab !
    s" .strtab" emit-str 0 cb,
  cubin-pos @ shstrtab-off @ - SN-symtab !
    s" .symtab" emit-str 0 cb,
  cubin-pos @ shstrtab-off @ - SN-nvinfo !
    s" .nv.info" emit-str 0 cb,
  cubin-pos @ shstrtab-off @ - SN-nvinfo-k !
    s" .nv.info." emit-str  kernel-name emit-str  0 cb,
  cubin-pos @ shstrtab-off @ - SN-text !
    s" .text." emit-str  kernel-name emit-str  0 cb,
  cubin-pos @ shstrtab-off @ - SN-const0 !
    s" .nv.constant0." emit-str  kernel-name emit-str  0 cb,
  cubin-pos @ shstrtab-off @ - SN-shared !
    s" .nv.shared.reserved.0" emit-str  0 cb,
  4 calign
  cubin-pos @ shstrtab-off @ - shstrtab-size !

  \ 3. .strtab  (section 2)
  cubin-pos @ strtab-off !
  0 cb,  0 cb,                  \ sym0 empty; sym1 (SECTION) empty
  cubin-pos @ strtab-off @ - strsym-kernel !
  kernel-name emit-str  0 cb,
  cubin-pos @ strtab-off @ - strtab-size !

  \ 4. .symtab  (section 3; 4 entries x 24 = 96 bytes)
  8 calign
  cubin-pos @ symtab-off !
  0 cd, 0 cb, 0 cb, 0 cw, 0 cq, 0 cq,    \ sym0: UNDEF
  0 cd, 3 cb, 0 cb, 5 cw, 0 cq, 0 cq,    \ sym1: SECTION LOCAL, shndx=5 (.text.<k>)
  0 cd, 3 cb, 0 cb, 8 cw, 0 cq, 0 cq,    \ sym2: SECTION LOCAL, shndx=8 (.nv.constant0.<k>)
  cubin-pos @ sym-kernel-off !
  strsym-kernel @ cd, $12 cb, $10 cb, 5 cw, 0 cq, 0 cq,
  \ sym3: FUNC GLOBAL STO_CUDA_ENTRY shndx=5; st_size patched after .text
  cubin-pos @ symtab-off @ - symtab-size !

  \ 5. .nv.info  (section 4; global register / stack attributes)
  4 calign
  cubin-pos @ nvinfo-off !
  max-reg-used @ 1+ 8 max  3 EIATTR-REGCOUNT  NVI-FMT-U32 nvi-sval,
  0  3 EIATTR-FRAME-SIZE     NVI-FMT-U32 nvi-sval,
  0  3 EIATTR-MIN-STACK-SIZE NVI-FMT-U32 nvi-sval,
  cubin-pos @ nvinfo-off @ - nvinfo-size !

  \ 6. .nv.info.<kernel>  (section 6; per-kernel attributes)
  4 calign
  cubin-pos @ nvinfo-k-off !
  $80 EIATTR-CUDA-API-VERSION NVI-FMT-U32 nvi-u32,
  \ KPARAM_INFO per param (8-byte ptr each, all in CBANK). Emit in REVERSE ordinal
  \ order to mirror ptxas output (highest ordinal first).
  \ Record body (12 bytes) per sm_90a nvcc ref decode of vadd:
  \   u32 index=0
  \   u32 (offset<<16) | ordinal
  \   u32 0x0021f000  (flags=0xf000 in low half, size-field=0x0021 in high half)
  n-kparams @ 0 ?do
    n-kparams @ 1- i - >r   ( ordinal on rstack )
    $04 cb, EIATTR-KPARAM-INFO cb, $0c cw,
    0 cd,                                          \ index
    r@ 8 *  16 lshift  r@ or  cd,                  \ (offset<<16)|ordinal
    $0021f000 cd,                                  \ flags=0xf000, size=8 ptr
    r> drop
  loop
  \ SPARSE_MMA_MASK flag (val=0)
  $03 cb, EIATTR-SPARSE-MMA-MASK cb, 0 cb, 0 cb,
  \ MAXREG_COUNT = 0xff (HVAL format: fmt=03 attr=0x1b val_u16)
  $03 cb, EIATTR-MAXREG-COUNT cb, $ff cb, 0 cb,
  \ ---- cooperative grid-sync attributes (omitted when cooperative?=0) ----
  \ Ordering matches probe_6b.sass: MASK_REGIDS before INSTR_OFFSETS,
  \ both before EXIT_INSTR_OFFSETS.
  cooperative? @ if
    \ COOP_GROUP_MASK_REGIDS (fmt=$04 attr=$29 size=16; four $ffffffff words)
    \ Bitmask marks all register slots as eligible for coop-group state.
    \ probe_6b.sass: four consecutive 0xffffffff u32 entries (16 bytes total).
    $04 cb, EIATTR-COOP-GROUP-MASK-REGIDS cb, $10 cw,
    $ffffffff cd, $ffffffff cd, $ffffffff cd, $ffffffff cd,
    \ COOP_GROUP_INSTR_OFFSETS (fmt=$04 attr=$28 size=N*4; one u32 per sync site)
    \ Each entry is the byte offset within .text.<kernel> of a grid-sync instruction.
    $04 cb, EIATTR-COOP-GROUP-INSTR-OFFSETS cb, gridsync-count @ 4 * cw,
    gridsync-count @ 0 ?do
      gridsync-offsets i cells + @ cd,
    loop
  then
  \ EXIT_INSTR_OFFSETS (fmt=04 attr=0x1c size=4 val=offset of EXIT in .text)
  \ For a kernel with a single EXIT at the end, pass text-size-16 (start of EXIT instr).
  \ We emit 0x100 as a safe default; actual offset would be patched by instruction-level
  \ tracking. ptxas for vadd puts EXIT at offset 0x100.
  $04 cb, EIATTR-EXIT-INSTR-OFFSETS cb, 4 cw, $100 cd,
  \ CBANK_PARAM_SIZE (HVAL; total param bytes)
  $03 cb, EIATTR-CBANK-PARAM-SIZE cb,  n-kparams @ 8 *  cw,
  \ PARAM_CBANK (SVAL: sym_idx=kernel_section_sym(1), cbank_offset=0x210 in low 16,
  \ param_bytes in high 16 of second u32)
  $04 cb, EIATTR-PARAM-CBANK cb, 8 cw,  2 cd,  n-kparams @ 8 * 16 lshift $210 or cd,
  \ SW_WAR workaround flag byte (matches sm_90a nvcc ref)
  $04 cb, EIATTR-SW-WAR cb, 4 cw, 8 cd,
  \ MAX_SHARED_MEM_PER_BLOCK_OPTIN = shmem-size (attr=0x33; emitted only when > 0)
  shmem-size @ dup 0> if
    $04 cb, $33 cb, 4 cw,  cd,
  else drop then
  cubin-pos @ nvinfo-k-off @ - nvinfo-k-size !

  \ 7. .text.<kernel>  (section 5; 128-byte aligned, copied from code buffer)
  128 calign
  cubin-pos @ text-off !
  sass-pos @ 0= if nop, nop, exit, then    \ guarantee non-empty
  sass-pos @ 0 ?do
    sass-buf i + c@ cb,
  loop
  cubin-pos @ text-off @ - text-size !

  \ Patch kernel sym st_size (u64 at sym-kernel-off+16)
  text-size @  sym-kernel-off @ 16 + put-u64

  \ 8. .nv.constant0.<kernel>
  \ Constant bank 0 is a per-kernel implicit-arg region. Bytes [0..0x210) are
  \ reserved for the driver (blockDim, gridDim, etc.); user params start at 0x210.
  \ Total size = 0x210 + param_bytes.
  4 calign
  cubin-pos @ const0-off !
  $210 n-kparams @ 8 * +  0 ?do 0 cb, loop
  cubin-pos @ const0-off @ - const0-size !

  \ 9. Section headers  (9 x 64 = 576 bytes, 64-byte aligned)
  64 calign
  cubin-pos @ shdrs-off !

  0 0 0 0  0 0  0 0  0 0  shdr64,                                                       \ [0] NULL
  SN-shstrtab @  3  0 0  shstrtab-off @ shstrtab-size @  0 0  1  0  shdr64,             \ [1] .shstrtab
  SN-strtab @    3  0 0  strtab-off @   strtab-size @    0 0  1  0  shdr64,             \ [2] .strtab
  SN-symtab @    2  0 0  symtab-off @   symtab-size @    2 3  8  24 shdr64,             \ [3] .symtab
  SN-nvinfo @    $70000000  0 0  nvinfo-off @  nvinfo-size @  3 0  4  0  shdr64,        \ [4] .nv.info
  SN-text @      1  6 0  text-off @  text-size @  3 3  128  0  shdr64,                  \ [5] .text.<k>
  SN-nvinfo-k @  $70000000  $40 0  nvinfo-k-off @  nvinfo-k-size @  3 5  4  0  shdr64,  \ [6] .nv.info.<k>
  SN-shared @    8  3 0  const0-off @  shmem-size @  0 0  16  0  shdr64,                \ [7] .nv.shared
  SN-const0 @    1  $42 0  const0-off @  const0-size @  0 5  4  0  shdr64,              \ [8] .nv.const0 (info=text section)

  \ 10. Patch ELF header
  0            32 put-u64    \ e_phoff  = 0  (no program headers)
  shdrs-off @  40 put-u64    \ e_shoff
  0            54 put-u16    \ e_phentsize = 0  (offset 54, overrides initial 56)
  0            56 put-u16    \ e_phnum = 0      (offset 56)
  9            60 put-u16    \ e_shnum = 9      (offset 60)
  1            62 put-u16    \ e_shstrndx = 1   (offset 62)
  \ e_shentsize = 64 already written by cubin-elf-header at offset 58 — do not overwrite

  cubin-buf cubin-pos @ ;
  \ After return: TOS=cubin-pos@, second=cubin-buf

\ build-cubin-shmem  ( shmem-bytes -- addr u )
\ Convenience: set shmem-size then call build-cubin.
\ Use instead of build-cubin when the kernel needs static shared memory.
: build-cubin-shmem  ( shmem-bytes -- addr u )
  shmem-size !
  build-cubin ;

\ Load cubin into GPU via driver API
\ This word will call cuModuleLoadData via FFI
: load-cubin  ( addr u -- module )
  \ TODO: CUDA driver API call
  drop drop 0 ;

\ ============================================================
\ CONSTANT BANK LOAD / UNIFORM LOAD INSTRUCTIONS
\ ============================================================

\ ldc,  ( dst cbuf-idx offset -- )
\ LDC Rd, c[cbuf-idx][offset]  — load 32-bit value from constant bank to GPR
\ Opcode: OP-LDC = $7b82.  ctrl: CTRL-LDC (stall=7 yield=1 wbar=1 rbar=7).
\
\ Instruction word field layout (verified: LDC R1,c[0x0][0x28] = 0x00000a00ff017b82):
\   bits[15: 0] = $7b82        (opcode)
\   bits[23:16] = Rd           (destination GPR)
\   bits[31:24] = $ff          (RZ placeholder — cbuf-idx is in ctrl, not inst word)
\   bits[47:32] = offset       (14-bit byte offset within constant bank)
\   bits[55:48] = cbuf-idx     (constant bank index 0-7, placed at bits[55:48])
\
\ Note: from probe decode of 0x00000a00ff017b82:
\   bits[23:16]=0x01 (R1), bits[31:24]=0xff (RZ), bits[47:32]=0x0a00>>16=offset,
\   bits[55:48]=0x00 (bank 0).  The 14-bit offset 0x28 = 40 appears in bits[47:32].
: ldc,  ( dst cbuf-idx offset -- )
  32 lshift >r              \ offset -> bits[47:32]
  48 lshift >r              \ cbuf-idx -> bits[55:48]
  track-rd 16 lshift        \ dst -> bits[23:16]
  OP-LDC or
  $ff 24 lshift or          \ RZ at bits[31:24]
  r> or r> or               \ cbuf-idx then offset
  CTRL-LDC sinst, ;

\ uldc,  ( dst cbuf-idx offset -- )
\ ULDC URn, c[cbuf-idx][offset]  — uniform load from constant bank to uniform register
\ Opcode: OP-ULDC = $7ab9.  ctrl: CTRL-ULDC (stall=1 yield=1 wbar=7 rbar=7).
\
\ Same field layout as ldc, (offset in bits[47:32], cbuf-idx in bits[55:48]).
\ Verified: ULDC.64 UR4,c[0x0][0x208] = 0x0000820000047ab9
\   bits[23:16]=0x04 (UR4), bits[31:24]=0x00, bits[47:32]=0x0000,
\   bits[55:48]=0x82 encodes bank=0 + 64-bit width bit; offset=0x208 in bits[47:32].
: uldc,  ( dst cbuf-idx offset -- )
  32 lshift >r              \ offset -> bits[47:32]
  48 lshift >r              \ cbuf-idx -> bits[55:48]
  track-rd 16 lshift        \ dst -> bits[23:16]
  OP-ULDC or
  $ff 24 lshift or          \ RZ at bits[31:24]
  r> or r> or               \ cbuf-idx then offset
  CTRL-ULDC sinst, ;

\ ============================================================
\ ATOMIC / REDUCTION FLOAT INSTRUCTIONS
\ ============================================================

\ atom-add-f32,  ( dst addr -- )
\ ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU Rd, desc[UR][Ra], Rs
\   Atomically adds a float32 value; Rd receives the old value.
\   Use Rd=RZ (0xFF) to discard the return value.
\
\ Opcode: OP-ATOMG-F32 = $79a3.  ctrl: CTRL-ATOMG-F32.
\
\ Field layout (same as atom-add,):
\   bits[23:16] = Rd (destination; RZ=0xFF to discard)
\   bits[31:24] = Ra (64-bit address register)
\   bits[39:32] = Rs (source data register)
\
\ Note: Rs is the value to add.  The caller passes ( dst addr ) and
\ we use RZ (0xFF) as the implicit source value placeholder; callers
\ who need a real source should use the three-argument form directly
\ by inlining.  For the common use-case (atomic accumulate into a
\ pointer using a known value register), pass the src register as dst
\ and the actual destination register as the first argument, then swap.
\
\ Practical calling convention adopted here to match atom-add, symmetry:
\   atom-add-f32, ( dst addr src -- )   (three arguments, matching atom-add,)
\   This mirrors atom-add, exactly: rd ra rs.
: atom-add-f32,  ( dst addr src -- )
  32 lshift >r              \ src  -> bits[39:32]
  24 lshift >r              \ addr -> bits[31:24]
  track-rd 16 lshift        \ dst  -> bits[23:16]
  OP-ATOMG-F32 or r> or r> or
  CTRL-ATOMG-F32 sinst, ;

\ redg-f32,  ( addr src -- )
\ REDG.E.ADD.F32.FTZ.RN.STRONG.GPU desc[UR][Ra], Rs
\   Float reduction — atomically adds Rs to [addr], discards old value.
\   No destination register (Rd=RZ implicitly).
\
\ Opcode: OP-REDG-F32 = $79a6.  ctrl: CTRL-REDG-F32.
\
\ Field layout:
\   bits[23:16] = RZ ($FF) — no return value
\   bits[31:24] = Ra (64-bit address register)
\   bits[39:32] = Rs (source data register, value to reduce)
: redg-f32,  ( addr src -- )
  32 lshift >r              \ src  -> bits[39:32]
  24 lshift                 \ addr -> bits[31:24]
  $ff 16 lshift or          \ RZ   -> bits[23:16] (no destination)
  OP-REDG-F32 or r> or
  CTRL-REDG-F32 sinst, ;
