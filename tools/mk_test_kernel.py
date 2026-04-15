#!/usr/bin/env python3
"""Generate a minimal SM90 test kernel that writes thread IDs to output.

Each thread writes its global thread ID (as float) to output[tid].
256 threads, 1 block -> output[0..255] = 0.0, 1.0, 2.0, ... 255.0

SASS sequence (matches nvcc reference byte-exact):
  S2R R5, SR_TID_X           # R5 = threadIdx.x
  I2F.F32.U32 R5, R5         # R5 = float(R5)
  LDC R2, c[0x2][0x84]       # R2 = output pointer (kernel param from cbuf 2)
  IMAD R2, R5, 0x4, R2       # R2 = &output[tid] (4 bytes per float)
  STG.E [R2], R5             # *R2 = R5
  EXIT
"""

import sys
sys.path.insert(0, '/home/ubuntu/quivent/lithos/bootstrap-py')

from emit_sm90 import SM90Emitter, SR_TID_X
from elf_writer import CubinWriter

e = SM90Emitter()

# Emit using the structured emitter API -- all encodings verified against
# nvcc reference (nvdisasm output, byte-exact match).

e.emit_s2r(5, SR_TID_X)       # S2R R5, SR_TID_X
e.emit_i2f(5, 5)              # I2F.F32.U32 R5, R5
e.emit_ldc(2, 2, 0x84)        # LDC R2, c[0x2][0x84]
e.emit_imad_imm(2, 5, 4, 2)   # IMAD R2, R5, 0x4, R2
e.emit_stg(2, 5)              # STG.E [R2], R5
e.emit_exit()                  # EXIT

code = e.get_code()
print(f"Code: {len(code)} bytes, {len(code)//16} instructions")
print(f"Registers: {e.get_register_count()}")

w = CubinWriter()
w.add_text('kernel', code)
w.add_nv_info(regcount=e.get_register_count(), n_params=1)
w.write('/tmp/tid_write.cubin')
print(f"Wrote /tmp/tid_write.cubin")
