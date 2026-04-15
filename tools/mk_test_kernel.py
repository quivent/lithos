#!/usr/bin/env python3
"""Generate a minimal SM90 test kernel that writes thread IDs to output.

Each thread writes its global thread ID (as float) to output[tid].
256 threads, 1 block → output[0..255] = 0.0, 1.0, 2.0, ... 255.0

SASS sequence:
  S2R R0, SR_TID_X           # R0 = threadIdx.x
  I2F.F32.U32 R1, R0         # R1 = float(R0)
  LDC R2, c[0x0][0x210]      # R2 = output pointer (first kernel param)
  IMAD R3, R0, 4, R2         # R3 = &output[tid] (4 bytes per float)
  STG.E [R3], R1             # *R3 = R1
  EXIT
"""

import sys
sys.path.insert(0, '/home/ubuntu/quivent/lithos/bootstrap-py')

from emit_sm90 import SM90Emitter
from elf_writer import CubinWriter

e = SM90Emitter()

# Mirror NVIDIA's exact instruction sequence from the reference kernel.
# Reference (nvdisasm of nvcc-compiled kernel):
#   [0] LDC R1, c[0x0][0xa00]        — load driver metadata (we skip)
#   [1] S2R R5, SR_TID_X
#   [2] LDC R2, c[0x2][0x84]         — param from cbuf2 (we use cbuf0+0x210)
#   [3] ULDC UR4, c[0x0][0x82]       — uniform load (we skip)
#   [4] IMAD R2, R5, 0x4, R2         — byte offset (uses immediate form 0x7825)
#   [5] I2F R5, R5                   — int to float
#   [6] STG [R2], R5                 — store
#   [7] EXIT

# We emit the same logic but using only opcodes we know are correct.
# Use the exact same bit patterns from the reference where possible.

# S2R R5, SR_TID_X — copy exact encoding from reference
# ref: i=0x0000000000057919 c=0x000e2e0000002100
import struct
def raw(inst, ctrl):
    e._buf.extend(struct.pack('<QQ', inst, ctrl))

# [0] S2R R5, SR_TID_X (exact copy from reference)
raw(0x0000000000057919, 0x000e2e0000002100)

# [1] I2F.F32.U32 R5, R5 (exact copy from reference)
raw(0x0000000500057245, 0x000fca0000201400)

# [2] LDC R2, c[0x2][0x84] — load output ptr from cbuf2 (where CUDA puts params)
# Reference: i=0x00008400ff027b82 c=0x000e220000000a00
raw(0x00008400ff027b82, 0x000e220000000a00)

# [3] IMAD R2, R5, 0x4, R2 — use the immediate form like NVIDIA does
# Reference IMAD.IMM: opcode 0x7825, imm in bits[53:32]
# Encoding: Rd[23:16]=2, Rs1[31:24]=5, Rs3 in ctrl extra41[7:0]=2, imm[53:32]=4
# We build this manually since our emitter only has the reg form
imad_iw = 0x7825 | (2 << 16) | (5 << 24) | (4 << 32)
imad_cw = 0x000fe200078e0202  # from reference, with rs3=R2 in extra41[7:0]
raw(imad_iw, imad_cw)

# [4] STG.E [R2], R5 — exact copy from reference
raw(0x0000000502007986, 0x000fe2000c101904)

# [5] EXIT — exact copy from reference
raw(0x000000000000794d, 0x000fea0003800000)

code = e.get_code()
print(f"Code: {len(code)} bytes, {len(code)//16} instructions")
print(f"Registers: {e.get_register_count()}")

w = CubinWriter()
w.add_text('kernel', code)
w.add_nv_info(regcount=e.get_register_count(), n_params=1)
w.write('/tmp/tid_write.cubin')
print(f"Wrote /tmp/tid_write.cubin")
