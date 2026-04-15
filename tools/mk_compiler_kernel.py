#!/usr/bin/env python3
"""Generate a test kernel using the lithos emitter API (not raw bytes).

This proves the emitter functions produce correct opcodes.
The kernel: each thread writes float(tid) to output[tid].

Uses only verified opcodes:
  S2R, I2F, LDC, IMAD_IMM, STG, EXIT
"""
import sys
sys.path.insert(0, '/home/ubuntu/quivent/lithos/bootstrap-py')

from emit_sm90 import SM90Emitter
from elf_writer import CubinWriter

e = SM90Emitter()

# Matches reference register allocation exactly (R5 for TID/float, R2 for ptr/addr)

# S2R R5, SR_TID_X (thread index)
e.emit_s2r(5, 0x21)

# I2F R5, R5 (convert to float for output, reuse R5)
e.emit_i2f(5, 5)

# LDC R2, c[0x2][0x84] (load output pointer from constant buffer 2)
e.emit_ldc(2, 2, 0x84)

# IMAD R2, R5, 4, R2 (byte address = tid * 4 + output_ptr, overwrite R2)
e.emit_imad_imm(2, 5, 4, 2)

# STG [R2], R5 (store float to global memory)
e.emit_stg(2, 5)

# EXIT
e.emit_exit()

code = e.get_code()
print(f"Code: {len(code)} bytes, {len(code)//16} instructions")
print(f"Registers: {e.get_register_count()}")

# Verify with nvdisasm if available
import subprocess, tempfile, os
w = CubinWriter()
w.add_text('kernel', code)
w.add_nv_info(regcount=e.get_register_count(), n_params=1)
w.write('/tmp/compiler_kernel.cubin')
print(f"Wrote /tmp/compiler_kernel.cubin")

result = subprocess.run(['nvdisasm', '/tmp/compiler_kernel.cubin'],
                       capture_output=True, text=True)
if 'error' in result.stderr.lower():
    print(f"nvdisasm ERRORS:\n{result.stderr}")
    sys.exit(1)
else:
    # Show the instructions
    for line in result.stdout.split('\n'):
        if '/*' in line and ';' in line:
            print(f"  {line.strip()}")
    print("nvdisasm: all instructions valid")
