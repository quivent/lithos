#!/bin/bash
# build.sh -- Build the native Lithos launcher
# Links against libcuda.so.1 for direct CUDA driver API calls.
set -e
cd /home/ubuntu/lithos

echo '[lithos] Assembling launcher.s...'
as -o bin/launcher.o src/launcher.s

echo '[lithos] Linking against libcuda.so.1...'
ld -dynamic-linker /lib/ld-linux-aarch64.so.1 \
   -o bin/lithos-launch \
   bin/launcher.o \
   -lcuda \
   -L/usr/lib/aarch64-linux-gnu

chmod +x bin/lithos-launch

echo '[lithos] Built: bin/lithos-launch'
echo '[lithos] Size:' $(stat -c %s bin/lithos-launch) 'bytes'
echo '[lithos] Usage: bin/lithos-launch <model_dir> <kernel_dir> [max_tokens]'
echo ''
echo '[lithos] Before first run, generate weight index:'
echo '  python3 bin/gen_weight_index.py <model_dir>'
