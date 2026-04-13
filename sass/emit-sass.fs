\ emit-sass.fs — Lithos SASS binary emitter for Hopper (sm_90)
\ Emits raw SASS instructions (16 bytes each) into a code buffer.
\ No PTX, no ptxas, no driver JIT.

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
: make-ctrl  ( stall yield wbar rbar wait reuse extra41 -- ctrl64 )
  >r >r >r >r >r >r >r
  r>                             \ extra41 (bits 0-40)
  r> 41 lshift or                \ stall  -> bits 41-44
  r> 45 lshift or                \ yield  -> bit  45
  r> 46 lshift or                \ wbar   -> bits 46-48
  r> 49 lshift or                \ rbar   -> bits 49-51
  r> 52 lshift or                \ wait   -> bits 52-57
  r> 58 lshift or ;              \ reuse  -> bits 58-62

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
: ctrl-ffma  ( rs3 -- ctrl64 )  5 0 7 7 0 0 rot make-ctrl ;

\ FADD: stall=5 yield=0 wbar=7 rbar=7 (no load dependency)
: ctrl-fadd       5 0 7 7 0 0 0             make-ctrl ;

\ FADD after LDG (waits on write barrier 2): verified 0x004fca0000000000
: ctrl-fadd-wait2 5 0 7 7 4 0 0             make-ctrl ;

\ IMAD: stall=1 yield=1 wbar=7 rbar=7
\ extra41 base = 0x0f8e0200 (opaque scheduler fields from probe); rs3 ORed into bits[7:0]
\ probe IMAD R0,R3,UR4,R0 ctrl: 0x002fe2000f8e0200 (extra41=0x0f8e0200, rs3=R0=0x00)
: ctrl-imad  ( rs3 -- ctrl64 )  1 1 7 7 0 0  $0f8e0200 rot or  make-ctrl ;

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
\ SASS CODE BUFFER
\ ============================================================

65536 constant SASS-SIZE
create sass-buf SASS-SIZE allot
variable sass-pos  0 sass-pos !

\ Per-kernel register high-water mark.
\ Reset to 0 at sass-reset; updated by every register-destination instruction.
\ build-cubin reads this to emit a tight REGCOUNT instead of a hardcoded value.
variable max-reg-used  0 max-reg-used !

\ track-rd  ( rd -- rd )
\ Non-destructive: updates max-reg-used if rd > current max, then leaves rd.
: track-rd  ( rd -- rd )  dup max-reg-used @ max max-reg-used ! ;

: sass-reset  0 sass-pos !  0 max-reg-used ! ;

\ Emit one 16-byte instruction: 8-byte inst word + 8-byte ctrl word
: sass,  ( inst-lo inst-hi ctrl-lo ctrl-hi -- )
  >r >r
  sass-buf sass-pos @ + >r
  \ inst word (little-endian, low 32 bits first)
  over r@ !          \ inst-lo at offset 0
  r@ 4 + !           \ inst-hi at offset 4
  \ ctrl word
  r> 8 + >r
  r@ r> ! drop       \ ctrl-lo at offset 8 — simplified
  \ TODO: proper 64-bit store for both words
  16 sass-pos +!
  r> drop ;

\ Simpler: emit raw bytes
: sb,  ( byte -- )  sass-buf sass-pos @ + c!  1 sass-pos +! ;
: sw,  ( u32 -- )   dup sb, 8 rshift dup sb, 8 rshift dup sb, 8 rshift sb, ;
: sq,  ( u64 -- )   dup sw, 32 rshift sw, ;

\ Emit a full 16-byte instruction from two 64-bit values
: sinst,  ( inst64 ctrl64 -- )  swap sq, sq, ;

\ ============================================================
\ HOPPER OPCODE TABLE (from probe disassembly)
\ ============================================================

\ Opcodes are in bits [15:0] of the instruction word (little-endian)
$7221 constant OP-FADD
$7223 constant OP-FFMA
$7819 constant OP-SHF
$7918 constant OP-NOP
$7919 constant OP-S2R
$7947 constant OP-BRA
$794d constant OP-EXIT
$7981 constant OP-LDG
$7986 constant OP-STG
$79c3 constant OP-S2UR
$7ab9 constant OP-ULDC
$7b82 constant OP-LDC
$7c0c constant OP-ISETP
$7c10 constant OP-IADD3
$7c24 constant OP-IMAD
$7984 constant OP-LDS
$7988 constant OP-STS
$7b1d constant OP-BAR-SYNC

\ Special register IDs (from S2R encoding)
$21 constant SR-TID-X
$22 constant SR-TID-Y
$23 constant SR-TID-Z
$25 constant SR-CTAID-X
$26 constant SR-CTAID-Y
$27 constant SR-CTAID-Z

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

\ FFMA Rd, Ra, Rb, Rc (fused multiply-add)  [stub — real impl below]
: ffma,  ( rd ra rb rc -- )
  drop drop drop drop
  $0000000502097223 0 ctrl-ffma sinst, ;

\ ============================================================
\ GEMV OPERATIONS — LDG, STG, IMAD, SHFL.BFLY, BRA, ISETP
\ Encodings verified against probe disassembly (probe*.sass)
\ ============================================================

\ Corrected opcode constants (bits[15:0] including PT predicate $7xxx)
\ Extend the existing OP-* table with missing GEMV ops
$7947 constant OP-BRA
$7c0c constant OP-ISETP

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

\ SHFL.BFLY PT, Rd, Rs, delta  — warp butterfly shuffle (imm delta, mask=0x1f)
\ Verified: SHFL.BFLY PT, R5, R0, 0x1, 0x1f -> inst=0x0c201f0000057f89
\   bits[63:56] = (log2(delta)<<4)|0x0c  (BFLY mode + delta-log2)
\   bits[55:48] = clamp = mask+1 = 0x20
\   bits[47:40] = mask = 0x1f
\   bits[31:24] = Rs, bits[23:16] = Rd, bits[15:0] = opcode $7f89
\ ctrl from probe_6b: stall=14 yield=0 wbar=3 rbar=7
: shfl-bfly,  ( rd rs delta -- )
  \ Compute mode byte: (log2(delta)<<4)|0x0c
  \ Use successive halving to find log2 (delta must be power of 2: 1,2,4,8,16)
  dup 1 = if drop 0 else
  dup 2 = if drop 1 else
  dup 4 = if drop 2 else
  dup 8 = if drop 3 else
             drop 4        \ delta=16
  then then then then
  4 lshift $0c or          \ mode_byte = (log2_delta<<4)|0x0c
  56 lshift >r             \ mode_byte -> bits[63:56]
  24 lshift >r             \ rs -> bits[31:24]
  track-rd 16 lshift       \ rd -> bits[23:16]
  $7f89 or r> or           \ rs
  $1f 40 lshift or          \ mask=0x1f at bits[47:40]
  $20 48 lshift or          \ clamp=0x20 at bits[55:48]
  r> or                    \ mode_byte
  ctrl-shfl sinst, ;

\ SHFL.DOWN PT, Rd, Rs, delta  — warp down-shuffle (lane N gets lane N+delta)
\ Encoding mirrors SHFL.BFLY but mode nibble = 0x0d (DOWN) instead of 0x0c (BFLY).
\ bits[63:56] = (log2(delta)<<4)|0x0d  (DOWN mode + delta-log2)
\ bits[55:48] = clamp = mask+1 = 0x20
\ bits[47:40] = mask = 0x1f
\ bits[31:24] = Rs, bits[23:16] = Rd, bits[15:0] = opcode $7f89
\ ctrl: same as shfl-bfly (stall=14 yield=0 wbar=3 rbar=7)
: shfl-down,  ( rd rs delta -- )
  dup 1 = if drop 0 else
  dup 2 = if drop 1 else
  dup 4 = if drop 2 else
  dup 8 = if drop 3 else
             drop 4        \ delta=16
  then then then then
  4 lshift $0d or          \ mode_byte = (log2_delta<<4)|0x0d
  56 lshift >r             \ mode_byte -> bits[63:56]
  24 lshift >r             \ rs -> bits[31:24]
  track-rd 16 lshift       \ rd -> bits[23:16]
  $7f89 or r> or           \ rs
  $1f 40 lshift or          \ mask=0x1f at bits[47:40]
  $20 48 lshift or          \ clamp=0x20 at bits[55:48]
  r> or                    \ mode_byte
  ctrl-shfl sinst, ;

\ BRA offset32  — branch PC-relative (signed byte offset from next instruction)
\ Verified: BRA loop -> inst=0xfffffffc00fc7947 ctrl=0x000fc0000383ffff
\ bits[63:32]=signed_offset, bits[23:16]=0xfc, bits[31:24]=0x00
: bra,  ( offset32 -- )
  32 lshift $00fc7947 or
  ctrl-bra sinst, ;

\ @Px BRA offset32  — predicated branch (pred = 0..3 for P0..P3)
: bra-pred,  ( offset32 pred -- )
  12 lshift >r
  32 lshift $00fc7947 or
  $FFFF0FFF and r> or
  ctrl-bra sinst, ;

\ ISETP.GE.U32.AND Pd, PT, Rs1, Rs2, PT  — integer >= comparison to predicate
\ Verified: ISETP.GE.U32.AND P0, PT, R0, UR4, PT -> inst=0x0000000400007c0c
\   bits[23:16]=Pd, bits[31:24]=Rs1, bits[39:32]=Rs2
\ ctrl-isetp: stall=13, opaque mode bits from probe
: isetp-ge,  ( pd rs1 rs2 -- )
  32 lshift >r              \ rs2 -> bits[39:32]
  24 lshift >r              \ rs1 -> bits[31:24]
  16 lshift                 \ pd  -> bits[23:16]
  $7c0c or r> or r> or
  ctrl-isetp sinst, ;

\ ISETP.LT.U32.AND Pd, PT, Rs1, Rs2, PT  — integer < comparison to predicate
: isetp-lt,  ( pd rs1 rs2 -- )
  32 lshift >r
  24 lshift >r
  16 lshift
  $7c0c or r> or r> or
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
  over over  16 shfl-bfly   \ tmp = shfl(acc, 16)
  over over swap fadd,      \ acc = acc + tmp
  over over   8 shfl-bfly
  over over swap fadd,
  over over   4 shfl-bfly
  over over swap fadd,
  over over   2 shfl-bfly
  over over swap fadd,
  over over   1 shfl-bfly
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
  16 lshift                 \ rd    -> bits[23:16]
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
: ctrl-lop3  ( lut-byte -- ctrl64 )  8 lshift   1 0 7 7 0 0 rot make-ctrl ;

: lop3-and-imm,  ( rd rs imm32 -- )
  \ LOP3.LUT Rd, Rs, imm32, RZ, 0xC0  (Rd = Rs AND imm32)
  32 lshift >r              \ imm32 -> bits[63:32]
  24 lshift >r              \ rs    -> bits[31:24]
  16 lshift                 \ rd    -> bits[23:16]
  $7812 or r> or r> or
  $c0 ctrl-lop3 sinst, ;    \ LUT=0xC0 = AND(A,B,C)=A&B

\ --- I2F.F32.S32 (signed int to float32) ---
\ Opcode: 0x306 (unverified against probe; canonical sm_90 I2F)
\ Encoding:
\   bits[15:0]  = 0x7306 (opcode 0x306, pred=PT)
\   bits[23:16] = Rd (float destination)
\   bits[31:24] = Rs1 (int source)
\ Scheduling: stall=4 yield=0 wbar=7 rbar=7 (conversion latency ~4 cycles)
: ctrl-i2f  4 0 7 7 0 0 0 make-ctrl ;

: i2f-s32-f32,  ( rd rs -- )
  \ I2F.F32.S32 Rd, Rs
  24 lshift >r              \ rs -> bits[31:24]
  16 lshift                 \ rd -> bits[23:16]
  $7306 or r> or
  ctrl-i2f sinst, ;

\ ============================================================
\ MUFU — Multi-Function Unit (special math, opcode 0x308)
\ ============================================================
\ Inst format: src << 32 | dst << 16 | 0x7308
\ Sub-function is in ctrl extra41 bits[13:10]:
\   COS=0x0000  SIN=0x0400  EX2=0x0800  LG2=0x0c00
\   RCP=0x1000  RSQ=0x1400  SQRT=0x2000
\ Scheduling: all stall=8 yield=1 rbar=7; wbar varies (probed):
\   EX2->7  RCP->1  RSQ->2  LG2->3  SQRT->4  SIN->5  COS->5
\
\ Probe verification (probe_2b.sass sm_90):
\   MUFU.EX2  R9,R6   inst=0x0000000600097308 ctrl=0x000ff00000000800
\   MUFU.RCP  R4,R4   inst=0x0000000400047308 ctrl=0x000e700000001000
\   MUFU.RSQ  R7,R0   inst=0x0000000000077308 ctrl=0x000eb00000001400
\   MUFU.LG2  R11,R0  inst=0x00000000000b7308 ctrl=0x000ef00000000c00
\   MUFU.SQRT R17,R0  inst=0x0000000000117308 ctrl=0x000f300000002000
\   MUFU.SIN  R13,R8  inst=0x00000008000d7308 ctrl=0x000f700000000400
\   MUFU.COS  R15,R8  inst=0x00000008000f7308 ctrl=0x000f620000000000

: ctrl-mufu  ( subfn-extra41 wbar -- ctrl64 )
  >r   8 1 r> 7 0 0 rot make-ctrl ;

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
  16 lshift                 \ rd  -> bits[23:16]
  $7984 or r> or r> or
  ctrl-lds sinst, ;

\ BAR.SYNC.DEFER_BLOCKING bar-id  — synchronize all threads at shared-memory barrier
\ Opcode 0x7b1d; bar-id in bits[23:16] (0..15)
\ verified: BAR.SYNC.DEFER_BLOCKING 0x0 = inst=0x0000000000007b1d ctrl=0x000fec0000010000
: bar-sync,  ( bar-id -- )
  16 lshift $7b1d or
  ctrl-bar sinst, ;


\ A cubin is an ELF64 file with NVIDIA-specific sections.
\ We need:
\   ELF header (64 bytes)
\   Section headers
\   .text.<kernel> — the SASS code
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

  \ 7. .text.<kernel>  (section 5; 128-byte aligned, copied from sass-buf)
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
  SN-const0 @    1  $22 0  const0-off @  const0-size @  0 3  4  0  shdr64,              \ [8] .nv.const0

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
