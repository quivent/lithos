\\ ============================================================
\\ parts/03-arm64-emit.ls — ARM64 host opcode emitters
\\ ============================================================
\\ One composition per Lithos primitive that targets the Grace
\\ (ARM64) host CPU. Each emit fn builds a 32-bit instruction
\\ word from its operands and appends it via `arm64_emit32`
\\ (defined alongside arm64_buf / arm64_pos_v in the runtime).
\\
\\ Register naming convention:
\\   rd / rt  = destination / transfer target  (Xd or Sd)
\\   ra / rn  = first source / base             (Xn or Sn)
\\   rb / rm  = second source                   (Xm or Sm)
\\   rc       = third source / accumulator      (Xa or Sa)
\\
\\ Encodings follow ARM ARM (DDI0596). 64-bit integer forms and
\\ 32-bit single-precision FP forms are used throughout.

\\ ============================================================
\\ FLOATING-POINT (FP32 — Sd, Sn, Sm)
\\ ============================================================

\\ FADD Sd, Sn, Sm
emit_a64_fadd rd ra rb :
    val 0x1E202800 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ FSUB Sd, Sn, Sm
emit_a64_fsub rd ra rb :
    val 0x1E203800 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ FMUL Sd, Sn, Sm
emit_a64_fmul rd ra rb :
    val 0x1E200800 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ FDIV Sd, Sn, Sm  (single instruction on ARM64)
emit_a64_fdiv rd ra rb :
    val 0x1E201800 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ FMADD Sd, Sn, Sm, Sa  (fused multiply-add: Sd = Sa + Sn*Sm)
emit_a64_fmadd rd ra rb rc :
    val 0x1F000000 | (rb << 16) | (rc << 10) | (ra << 5) | rd
    arm64_emit32 val

\\ FSQRT Sd, Sn  (single instruction — no MUFU.RSQ + RCP dance)
emit_a64_fsqrt rd ra :
    val 0x1E21C000 | (ra << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ MATH APPROXIMATIONS (ex2 / lg2 / sin / cos)
\\ ============================================================
\\ ARM64 has no MUFU equivalent. Host-side inference is rare,
\\ so emit a NOP placeholder. Real deployments route these to
\\ a polynomial expansion or `libm` call via BL.

\\ 2^x — NOP placeholder (see note above)
emit_a64_ex2 rd ra :
    arm64_emit32 0xD503201F

\\ log2 x — NOP placeholder
emit_a64_lg2 rd ra :
    arm64_emit32 0xD503201F

\\ sin x — NOP placeholder
emit_a64_sin rd ra :
    arm64_emit32 0xD503201F

\\ cos x — NOP placeholder
emit_a64_cos rd ra :
    arm64_emit32 0xD503201F

\\ ============================================================
\\ INTEGER (64-bit — Xd, Xn, Xm)
\\ ============================================================

\\ ADD Xd, Xn, Xm
emit_a64_add rd ra rb :
    val 0x8B000000 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ ADD Xd, Xn, #imm12
emit_a64_add_imm rd ra imm :
    val 0x91000000 | (imm << 10) | (ra << 5) | rd
    arm64_emit32 val

\\ SUB Xd, Xn, Xm
emit_a64_sub rd ra rb :
    val 0xCB000000 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ MUL Xd, Xn, Xm  (alias for MADD with Xa=XZR)
emit_a64_mul rd ra rb :
    val 0x9B007C00 | (rb << 16) | (ra << 5) | rd
    arm64_emit32 val

\\ MADD Xd, Xn, Xm, Xa  (Xd = Xa + Xn*Xm)
emit_a64_madd rd ra rb rc :
    val 0x9B000000 | (rb << 16) | (rc << 10) | (ra << 5) | rd
    arm64_emit32 val

\\ CMP Xn, Xm  (alias for SUBS XZR, Xn, Xm)
emit_a64_cmp ra rb :
    val 0xEB00001F | (rb << 16) | (ra << 5)
    arm64_emit32 val

\\ MOV Xd, Xn  (alias for ORR Xd, XZR, Xm)
emit_a64_mov rd ra :
    val 0xAA0003E0 | (ra << 16) | rd
    arm64_emit32 val

\\ MOVZ Xd, #imm16, LSL #(shift*16)
emit_a64_movz rd imm shift :
    val 0xD2800000 | (shift << 21) | (imm << 5) | rd
    arm64_emit32 val

\\ MOVK Xd, #imm16, LSL #(shift*16)
emit_a64_movk rd imm shift :
    val 0xF2800000 | (shift << 21) | (imm << 5) | rd
    arm64_emit32 val

\\ ============================================================
\\ BRANCHES
\\ ============================================================

\\ B.cond offset  (offset in bytes, scaled /4 into imm19)
emit_a64_b_cond cond offset :
    val 0x54000000 | (((offset / 4) & 0x7FFFF) << 5) | cond
    arm64_emit32 val

\\ B offset  (unconditional, imm26 scaled by 4)
emit_a64_b offset :
    val 0x14000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ BL offset  (branch with link — function call)
emit_a64_bl offset :
    val 0x94000000 | ((offset / 4) & 0x3FFFFFF)
    arm64_emit32 val

\\ RET  (return via X30/LR)
emit_a64_ret :
    arm64_emit32 0xD65F03C0

\\ ============================================================
\\ MEMORY
\\ ============================================================

\\ LDR Xt, [Xn, #imm]  (64-bit, unsigned offset scaled by 8)
emit_a64_ldr rd rn offset :
    val 0xF9400000 | ((offset / 8) << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STR Xt, [Xn, #imm]  (64-bit, unsigned offset scaled by 8)
emit_a64_str rt rn offset :
    val 0xF9000000 | ((offset / 8) << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDRB Wt, [Xn, #imm]  (byte load, unscaled offset)
emit_a64_ldrb rd rn offset :
    val 0x39400000 | (offset << 10) | (rn << 5) | rd
    arm64_emit32 val

\\ STRB Wt, [Xn, #imm]  (byte store, unscaled offset)
emit_a64_strb rt rn offset :
    val 0x39000000 | (offset << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ STP Xt, Xt2, [Xn, #imm]!  (pre-indexed pair store — prologue)
emit_a64_stp_pre rt rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ LDP Xt, Xt2, [Xn], #imm  (post-indexed pair load — epilogue)
emit_a64_ldp_post rt rt2 rn imm :
    imm7 (imm / 8) & 0x7F
    val 0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt
    arm64_emit32 val

\\ ============================================================
\\ SYSTEM
\\ ============================================================

\\ SVC #imm16  (syscall)
emit_a64_svc imm :
    val 0xD4000001 | (imm << 5)
    arm64_emit32 val
