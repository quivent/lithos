#!/bin/bash
# Build reference cubin from CUDA, extract SASS, compare with ours
set -e

cd ~/lithos/tools

echo "=== Building reference kernel ==="
nvcc -cubin -arch=sm_90a -o /tmp/ref.cubin mk_reference.cu 2>&1

echo "=== Reference cubin ==="
readelf -S /tmp/ref.cubin 2>&1 | grep -E "text|info|constant"

echo "=== Disassemble reference ==="
nvdisasm /tmp/ref.cubin 2>&1 | head -30

echo "=== Disassemble ours ==="
nvdisasm /tmp/tid_write.cubin 2>&1 | head -30

echo "=== Hex dump reference .text (first 128 bytes) ==="
python3 -c "
import struct
with open('/tmp/ref.cubin', 'rb') as f:
    data = f.read()
# Find .text section
ehdr_shoff = struct.unpack_from('<Q', data, 40)[0]
ehdr_shnum = struct.unpack_from('<H', data, 60)[0]
ehdr_shstrndx = struct.unpack_from('<H', data, 62)[0]
shstr_off = struct.unpack_from('<Q', data, ehdr_shoff + ehdr_shstrndx * 64 + 24)[0]
shstrtab = data[shstr_off:]
for i in range(ehdr_shnum):
    sh = ehdr_shoff + i * 64
    name_off = struct.unpack_from('<I', data, sh)[0]
    name = shstrtab[name_off:shstrtab.index(0, name_off)].decode()
    if '.text' in name:
        off = struct.unpack_from('<Q', data, sh + 24)[0]
        sz = struct.unpack_from('<Q', data, sh + 32)[0]
        print(f'REF {name}: offset={off} size={sz}')
        for j in range(0, min(sz, 128), 16):
            iw = struct.unpack_from('<Q', data, off+j)[0]
            cw = struct.unpack_from('<Q', data, off+j+8)[0]
            print(f'  [{j//16:2d}] i=0x{iw:016x} c=0x{cw:016x}')

with open('/tmp/tid_write.cubin', 'rb') as f:
    data2 = f.read()
ehdr_shoff = struct.unpack_from('<Q', data2, 40)[0]
ehdr_shnum = struct.unpack_from('<H', data2, 60)[0]
ehdr_shstrndx = struct.unpack_from('<H', data2, 62)[0]
shstr_off = struct.unpack_from('<Q', data2, ehdr_shoff + ehdr_shstrndx * 64 + 24)[0]
shstrtab = data2[shstr_off:]
for i in range(ehdr_shnum):
    sh = ehdr_shoff + i * 64
    name_off = struct.unpack_from('<I', data2, sh)[0]
    name = shstrtab[name_off:shstrtab.index(0, name_off)].decode()
    if '.text' in name:
        off = struct.unpack_from('<Q', data2, sh + 24)[0]
        sz = struct.unpack_from('<Q', data2, sh + 32)[0]
        print(f'OURS {name}: offset={off} size={sz}')
        for j in range(0, min(sz, 128), 16):
            iw = struct.unpack_from('<Q', data2, off+j)[0]
            cw = struct.unpack_from('<Q', data2, off+j+8)[0]
            print(f'  [{j//16:2d}] i=0x{iw:016x} c=0x{cw:016x}')
"
