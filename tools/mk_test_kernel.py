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

# S2R R0, SR_TID_X (sr_id = 0x21 = 33)
e.emit_s2r(0, 0x21)

# I2F.F32.U32 R1, R0 — convert int to float
e.emit_i2f(1, 0)

# LDC R2, c[0x0][0x210] — load output pointer (first kernel param, 8 bytes)
e.emit_ldc(2, 0, 0x210)

# MOV R4, 4 — load stride constant
e.emit_mov_imm(4, 4)

# IMAD R3, R0, R4, R2 — R3 = tid * 4 + output_ptr
e.emit_imad(3, 0, 4, 2)

# STG.E [R3], R1 — store float to global memory
e.emit_stg(3, 1)

# EXIT
e.emit_exit()

code = e.get_code()
print(f"Code: {len(code)} bytes, {len(code)//16} instructions")
print(f"Registers: {e.get_register_count()}")

w = CubinWriter()
w.add_text('kernel', code)
w.add_nv_info(regcount=e.get_register_count(), n_params=1)
w.write('/tmp/tid_write.cubin')
print(f"Wrote /tmp/tid_write.cubin")
