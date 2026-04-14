# Bootstrap Emit Layer Extension Report

## Emitters Added

All additions are in `/home/ubuntu/lithos/bootstrap/emit-arm64.s`, sections 13-14, with corresponding dictionary entries chained after `entry_e_cbnz_fwd`.

### New Code Words (Section 13)

| Word | Stack Effect | Description | Encoding |
|------|-------------|-------------|----------|
| `emit-mov-imm` | ( rd imm16 -- ) | MOVZ Xd, #imm16 | 0xD2800000 \| imm16<<5 \| Rd |
| `emit-prologue` | ( -- ) | STP x29,x30,[sp,#-16]! + MOV x29,sp | 0xA9BF7BFD, 0x910003FD |
| `emit-epilogue` | ( -- ) | LDP x29,x30,[sp],#16 + RET | 0xA8C17BFD, 0xD65F03C0 |
| `emit-stp-imm` | ( rt1 rt2 rn imm -- ) | STP pair with signed offset | 0xA9000000 base |
| `emit-ldp-imm` | ( rt1 rt2 rn imm -- ) | LDP pair with signed offset | 0xA9400000 base |
| `emit-dmb-sy` | ( -- ) | DMB SY (full system barrier) | 0xD5033FBF |
| `emit-patch-cbnz` | ( mark_pos -- ) | Patch CBNZ offset at mark_pos | 19-bit imm19<<5 |

### Condition Code Pushers (Section 13)

| Word | Value | ARM64 Condition |
|------|-------|----------------|
| `cond-eq` | 0 | Equal (Z=1) |
| `cond-ne` | 1 | Not equal (Z=0) |
| `cond-hs` | 2 | Unsigned higher or same (C=1) |
| `cond-lo` | 3 | Unsigned lower (C=0) |
| `cond-mi` | 4 | Negative (N=1) |
| `cond-pl` | 5 | Positive or zero (N=0) |
| `cond-vs` | 6 | Overflow (V=1) |
| `cond-vc` | 7 | No overflow (V=0) |
| `cond-hi` | 8 | Unsigned higher |
| `cond-ls` | 9 | Unsigned lower or same |
| `cond-ge` | 10 | Signed greater or equal |
| `cond-lt` | 11 | Signed less than |
| `cond-gt` | 12 | Signed greater than |
| `cond-le` | 13 | Signed less or equal |

### Composite Emitters (Section 14)

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `emit-array-load` | ( rd rb ri log2size -- ) | LSL X16,Xri,#log2size; ADD X16,Xrb,X16; LDR Xrd,[X16] |
| `emit-array-store` | ( rs rb ri log2size -- ) | LSL X16,Xri,#log2size; ADD X16,Xrb,X16; STR Xrs,[X16] |

Array emitters use X16 (IP0) as scratch, which is the standard ARM64 intra-procedure-call scratch register.

## Pre-Existing Coverage

The following words requested by the parser spec were already present:

- `emit-add-reg`, `emit-sub-reg`, `emit-mul`, `emit-udiv`, `emit-sdiv` (category 1)
- `emit-add-imm`, `emit-sub-imm` (category 2)
- `emit-and-reg`, `emit-orr-reg`, `emit-eor-reg`, `emit-lsl-imm`, `emit-lsr-imm` (category 3)
- `emit-ldr`, `emit-str`, `emit-ldrb`, `emit-strb`, `emit-ldrh`, `emit-strh`, `emit-ldr-w`, `emit-str-w`, `emit-ldr-reg`, `emit-str-reg` (category 4)
- `emit-b`, `emit-bl`, `emit-br`, `emit-ret`, `emit-cbz`, `emit-cbnz`, `emit-bcond` (category 6)
- `emit-cmp-reg`, `emit-cmp-imm` (category 7)
- `emit-mov`, `emit-mov64` (category 8)
- `emit-ldp`, `emit-stp`, `emit-ldp-pre`, `emit-stp-pre` (category 9)
- `emit-svc`, `emit-nop`, `emit-dsb-sy` (category 10)
- `emit-mark`, `emit-patch-bcond`, `emit-patch-b` (category 11)

## Sample Encodings Verified

Cross-checked against `aarch64-linux-gnu-as` + `objdump -d`:

| Instruction | Expected | Objdump | Match |
|-------------|----------|---------|-------|
| STP x29,x30,[sp,#-16]! | 0xA9BF7BFD | 0xa9bf7bfd | YES |
| MOV x29, sp | 0x910003FD | 0x910003fd | YES |
| LDP x29,x30,[sp],#16 | 0xA8C17BFD | 0xa8c17bfd | YES |
| RET | 0xD65F03C0 | 0xd65f03c0 | YES |
| MOVZ x5, #0x1234 | 0xD2824685 | 0xd2824685 | YES |
| DMB SY | 0xD5033FBF | 0xd5033fbf | YES |
| DSB SY | 0xD5033F9F | 0xd5033f9f | YES |
| NOP | 0xD503201F | 0xd503201f | YES |
| SVC #0 | 0xD4000001 | 0xd4000001 | YES |
| LSL x16,x2,#3 | 0xD37DF050 | 0xd37df050 | YES |
| ADD x16,x1,x16 | 0x8B100030 | 0x8b100030 | YES |
| LDR x5,[x16] | 0xF9400205 | 0xf9400205 | YES |

## Assembly Verification

`emit-arm64.s` assembles cleanly with zero warnings:
```
aarch64-linux-gnu-as -o emit-arm64-test.o emit-arm64.s
```

## Notes

- Dictionary chain: new entries chain from `entry_e_cbnz_fwd` through to `entry_e_patch_cbnz` (new tail). The `emit_last_entry` label is updated.
- No modifications to `lithos-lexer.s` or `lithos-parser.s`.
- The array composite emitters hardcode X16 as scratch. If the parser needs a different temp, this can be parameterized by adding a temp-reg argument.
