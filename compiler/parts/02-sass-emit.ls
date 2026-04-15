\\ ============================================================================
\\ parts/02-sass-emit.ls — SASS (NVIDIA Hopper sm_90a) opcode emitters
\\ ============================================================================
\\
\\ One emit composition per Lithos primitive. Each builds a 64-bit instruction
\\ word + 64-bit control word and calls `sinst iword ctrl` to write both
\\ little-endian to gpu_buf at gpu_pos_v.
\\
\\ All encodings verified empirically against ptxas + nvdisasm on sm_90a.
\\ See docs/encoding/*.md for the derivation of every hex constant used here.
\\
\\ Assumes (declared elsewhere):
\\   gpu_buf, gpu_pos_v         — code buffer + position
\\   sinst iword ctrl           — writes two 64-bit words
\\
\\ Register conventions:
\\   RZ = 0xFF (always-zero register, also reads-as-zero)
\\   PT = 0x07 (always-true predicate)
\\
\\ ============================================================================

RZ  0xFF
PT  0x07

\\ ============================================================
\\ ARITHMETIC (§3.4) — FADD, FSUB, FMUL, FDIV, FFMA
\\ ============================================================

\\ FADD Rd, Ra, Rb                 — float add (opcode 0x7221, ctrl=0x000FCA0000000000)
emit_fadd rd ra rb :
    iword 0x7221 | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword 0x000FCA0000000000

\\ FADD Rd, Ra, -Rb                — float sub (opcode 0x7221 + NEG-src2 bit 63)
emit_fsub rd ra rb :
    iword 0x7221 | (rd << 16) | (ra << 24) | (rb << 32) | (1 << 63)
    sinst iword 0x000FCA0000000000

\\ FMUL Rd, Ra, Rb                 — float multiply (opcode 0x7220)
emit_fmul rd ra rb :
    iword 0x7220 | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword 0x004FCA0000400000

\\ FFMA Rd, Ra, Rb, Rc             — fused multiply-add (opcode 0x7223, Rc in ctrl[7:0])
emit_ffma rd ra rb rc :
    iword 0x7223 | (rd << 16) | (ra << 24) | (rb << 32)
    ctrl 0x000FCA0000000000 | rc
    sinst iword ctrl

\\ FDIV Rd, Ra, Rb                 — two-instruction sequence: MUFU.RCP + FMUL
\\ rt is a caller-provided scratch reg (same register index as rd is fine)
emit_fdiv rd ra rb rt :
    \\ MUFU.RCP Rt, Rb           (opcode 0x7308, subop 0x1000)
    rcp_iword 0x7308 | (rt << 16) | (rb << 32)
    sinst rcp_iword 0x001E640000001000
    \\ FMUL Rd, Ra, Rt
    mul_iword 0x7220 | (rd << 16) | (ra << 24) | (rt << 32)
    sinst mul_iword 0x004FCA0000400000

\\ ============================================================
\\ MATH — SFU (§3.7) — all MUFU variants (opcode 0x7308)
\\ subop lives in ctrl extra41 bits [13:8]; packed as value << 0 into extra41
\\ ============================================================

\\ MUFU.RCP Rd, Ra                 — reciprocal
emit_rcp rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E640000001000

\\ MUFU.RSQ Rd, Ra                 — reciprocal sqrt
emit_rsq rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E640000001400

\\ MUFU.EX2 Rd, Ra                 — 2^x
emit_ex2 rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E640000000800

\\ MUFU.LG2 Rd, Ra                 — log2(x)
emit_lg2 rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E640000000C00

\\ MUFU.SQRT Rd, Ra                — sqrt (single-op form, 0x2000 subop)
\\ Spec mandates RSQ+RCP; emitted as two instructions for numerical-identity match
emit_sqrt rd ra :
    \\ MUFU.RSQ Rd, Ra  (1/sqrt)
    iword_rsq 0x7308 | (rd << 16) | (ra << 32)
    sinst iword_rsq 0x001E640000001400
    \\ MUFU.RCP Rd, Rd  (1 / (1/sqrt) = sqrt)
    iword_rcp 0x7308 | (rd << 16) | (rd << 32)
    sinst iword_rcp 0x001E640000001000

\\ MUFU.SIN Rd, Ra
emit_sin rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E240000000400

\\ MUFU.COS Rd, Ra
emit_cos rd ra :
    iword 0x7308 | (rd << 16) | (ra << 32)
    sinst iword 0x001E240000000000

\\ ============================================================
\\ REDUCTIONS (§3.6) — SHFL.BFLY, FMNMX, BAR.SYNC
\\ ============================================================

\\ SHFL.BFLY Rd, Ra, delta, 0x1f   — butterfly warp shuffle (opcode 0x7F89)
\\ delta is the XOR lane offset (1,2,4,8,16 for a 5-step warp reduction).
\\ mask is the 8-bit warp mask (typically 0x1f = all 32 lanes active).
emit_shfl_bfly rd ra delta mask :
    denc  delta * 32
    clamp denc & 0xFF
    mode  0x0C | (denc >> 8)
    iword 0x7F89 | (rd << 16) | (ra << 24) | (mask << 40) | (clamp << 48) | (mode << 56)
    sinst iword 0x001E6800000E0000

\\ FMNMX Rd, Ra, Rb, !PT           — max (opcode 0x7209, !PT selector in ctrl bit 26)
emit_fmnmx_max rd ra rb :
    iword 0x7209 | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword 0x004FCA0007800000

\\ FMNMX Rd, Ra, Rb, PT            — min (opcode 0x7209, PT selector)
emit_fmnmx_min rd ra rb :
    iword 0x7209 | (rd << 16) | (ra << 24) | (rb << 32)
    sinst iword 0x004FCA0003800000

\\ BAR.SYNC 0                      — CTA barrier (opcode 0x7B1D, bar_id in bits[55:48])
emit_bar_sync :
    sinst 0x0000000000007B1D 0x000FEC0000010000

\\ ============================================================
\\ MEMORY (§3.2) — LDG, STG, LDS, STS (32-bit variants)
\\ ============================================================

\\ LDG.E Rd, [Ra]                  — global load, 32-bit (opcode 0x7981)
emit_ldg rd raddr :
    iword 0x7981 | (rd << 16) | (raddr << 24)
    sinst iword 0x000EA8000C1E1900

\\ STG.E [Ra], Rv                  — global store, 32-bit (opcode 0x7986)
emit_stg raddr rval :
    iword 0x7986 | (raddr << 24) | (rval << 32)
    sinst iword 0x000FE2000C101904

\\ LDS Rd, [Ra]                    — shared load (UR-mode opcode 0x7984 with 0xFF marker)
emit_lds rd raddr :
    iword 0x7984 | (rd << 16) | (RZ << 24) | (raddr << 32)
    sinst iword 0x000E280008000800

\\ STS [Ra], Rv                    — shared store (UR-mode opcode 0x7988)
emit_sts raddr rval :
    iword 0x7988 | (RZ << 24) | (rval << 32) | (raddr << 40)
    sinst iword 0x000FE80008000804

\\ ============================================================
\\ INTEGER — IADD3, IMAD, ISETP, BRA, S2R, MOV, EXIT
\\ ============================================================

\\ IADD3 Rd, Ra, Rb, Rc            — 3-input int add (opcode 0x7210, Rc in ctrl[7:0])
emit_iadd3 rd ra rb rc :
    iword 0x7210 | (rd << 16) | (ra << 24) | (rb << 32)
    ctrl 0x000FCA0007FFE000 | rc
    sinst iword ctrl

\\ IMAD Rd, Ra, Rb, Rc             — int multiply-add (opcode 0x7224, Rc in ctrl[7:0])
emit_imad rd ra rb rc :
    iword 0x7224 | (rd << 16) | (ra << 24) | (rb << 32)
    ctrl 0x000FCA00078E0200 | rc
    sinst iword ctrl

\\ ISETP.GE P, Ra, Rb              — int set-predicate ≥ (opcode 0x720C, cond=0x6)
emit_isetp_ge p ra rb :
    iword 0x720C | (ra << 24) | (rb << 32)
    ctrl 0x000FDA0003F06000 | (p << 17)
    sinst iword ctrl

\\ ISETP.LT P, Ra, Rb              — int set-predicate < (cond=0x1)
emit_isetp_lt p ra rb :
    iword 0x720C | (ra << 24) | (rb << 32)
    ctrl 0x000FDA0003F01000 | (p << 17)
    sinst iword ctrl

\\ ISETP.EQ P, Ra, Rb              — int set-predicate == (cond=0x2)
emit_isetp_eq p ra rb :
    iword 0x720C | (ra << 24) | (rb << 32)
    ctrl 0x000FDA0003F02000 | (p << 17)
    sinst iword ctrl

\\ @P BRA offset                   — predicated branch (opcode 0x7947, pred in bits[14:12])
\\ offset is the signed byte offset from PC_next; scaled to 4-byte units
emit_bra_pred p offset :
    off32 offset / 4
    iword 0x0000F047 | (p << 12) | (off32 << 32)
    sinst iword 0x000FC0000383FFFF

\\ BRA offset                      — unconditional branch (PT = 7 in bits[14:12])
emit_bra offset :
    off32 offset / 4
    iword 0x00FC7947 | (off32 << 32)
    sinst iword 0x000FC0000383FFFF

\\ S2R Rd, SR_*                    — system-register read (opcode 0x7919, sr_id in ctrl[15:8])
emit_s2r rd sr_id :
    iword 0x7919 | (rd << 16)
    sinst iword 0x000E2E0000000000 | (sr_id << 8)

\\ MOV Rd, Ra                      — register move via IMAD idiom (Rd = Ra * RZ + RZ)
\\ There is no standalone MOV on Hopper; ptxas emits IMAD.MOV.U32.
emit_mov rd ra :
    iword 0x7224 | (rd << 16) | (ra << 24) | (RZ << 32)
    ctrl 0x000FCA00078E0200 | RZ
    sinst iword ctrl

\\ MOV Rd, imm32                   — load 32-bit immediate (opcode 0x7802)
emit_mov_imm rd imm :
    iword 0x7802 | (rd << 16) | (imm << 32)
    sinst iword 0x000FC00000000F00

\\ EXIT                            — kernel terminator (opcode 0x794D)
emit_exit :
    sinst 0x000000000000794D 0x000FEA0003800000
