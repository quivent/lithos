#!/usr/bin/env python3
"""gen_weight_index.py -- Generate binary weight index for the Lithos native launcher.

Reads safetensors shard headers and produces a compact binary file that the
ARM64 launcher can load without parsing JSON.

Output format (weight_index.bin):
  Header:
    u32  magic          = 0x4C575449 ('LTWI' = Lithos Weight Index)
    u32  version        = 1
    u32  num_shards     = N (4 for this model)
    u32  num_layers     = 64
    u32  slots_per_layer = 26
    u32  num_globals    = 3 (embed, lm_head, final_norm)
    u32  reserved[2]

  Shard table (num_shards entries, 264 bytes each):
    char filename[256]  = null-terminated shard filename (basename only)
    u64  file_size      = size of the shard file

  Global tensor table (num_globals entries, 24 bytes each):
    u32  shard_idx      = which shard (0-based)
    u32  reserved
    u64  file_offset    = byte offset in shard file
    u64  size           = tensor size in bytes

  Per-layer weight table (num_layers * slots_per_layer entries, 24 bytes each):
    u32  shard_idx      = which shard (0-based)
    u32  reserved
    u64  file_offset    = byte offset in shard file
    u64  size           = tensor size in bytes

Total: 32 + 4*264 + 3*24 + 64*26*24 = 32 + 1056 + 72 + 39936 = 41096 bytes

Usage:
  python3 gen_weight_index.py <model_dir> [output_file]
"""

import struct
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [output_file]", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "weight_index.bin")

    # Discover shard files
    shard_files = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith('.safetensors') and f.startswith('model-')
    ])
    num_shards = len(shard_files)
    print(f"Found {num_shards} shard files")

    # Load all shard headers
    shard_info = []  # (filename, file_size, header_size, header_dict)
    tensor_index = {}  # full_name -> (shard_idx, file_offset, size, dtype, shape)

    for shard_idx, fname in enumerate(shard_files):
        fpath = os.path.join(model_dir, fname)
        fsize = os.path.getsize(fpath)
        with open(fpath, 'rb') as f:
            hdr_size = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_size))

        data_base = 8 + hdr_size
        shard_info.append((fname, fsize, hdr_size, hdr))

        for name, info in hdr.items():
            if not isinstance(info, dict) or 'data_offsets' not in info:
                continue
            start, end = info['data_offsets']
            tensor_index[name] = (shard_idx, data_base + start, end - start, info['dtype'], info.get('shape', []))

    # Define per-layer weight slots
    # DeltaNet layers (layer_idx % 4 != 3)
    DN_SLOTS = [
        'linear_attn.in_proj_qkv.qweight',     # 0
        'linear_attn.in_proj_qkv.scales',       # 1
        'linear_attn.in_proj_qkv.qzeros',       # 2
        'linear_attn.in_proj_z.qweight',         # 3
        'linear_attn.in_proj_z.scales',          # 4
        'linear_attn.in_proj_z.qzeros',          # 5
        'linear_attn.out_proj.qweight',          # 6
        'linear_attn.out_proj.scales',           # 7
        'linear_attn.out_proj.qzeros',           # 8
        'linear_attn.in_proj_a.weight',          # 9
        'linear_attn.in_proj_b.weight',          # 10
        'linear_attn.A_log',                     # 11
        'linear_attn.dt_bias',                   # 12
        'linear_attn.conv1d.weight',             # 13
        'linear_attn.norm.weight',               # 14
        'mlp.gate_proj.qweight',                 # 15
        'mlp.gate_proj.scales',                  # 16
        'mlp.gate_proj.qzeros',                  # 17
        'mlp.up_proj.qweight',                   # 18
        'mlp.up_proj.scales',                    # 19
        'mlp.up_proj.qzeros',                    # 20
        'mlp.down_proj.qweight',                 # 21
        'mlp.down_proj.scales',                  # 22
        'mlp.down_proj.qzeros',                  # 23
        'input_layernorm.weight',                # 24
        'post_attention_layernorm.weight',       # 25
    ]

    # Full-attention layers (layer_idx % 4 == 3)
    FA_SLOTS = [
        'self_attn.q_proj.qweight',              # 0
        'self_attn.q_proj.scales',               # 1
        'self_attn.q_proj.qzeros',               # 2
        'self_attn.k_proj.qweight',              # 3
        'self_attn.k_proj.scales',               # 4
        'self_attn.k_proj.qzeros',               # 5
        'self_attn.v_proj.qweight',              # 6
        'self_attn.v_proj.scales',               # 7
        'self_attn.v_proj.qzeros',               # 8
        'self_attn.o_proj.qweight',              # 9
        'self_attn.o_proj.scales',               # 10
        'self_attn.o_proj.qzeros',               # 11
        'self_attn.q_norm.weight',               # 12
        'self_attn.k_norm.weight',               # 13
        '',                                       # 14 (unused, pad)
        'mlp.gate_proj.qweight',                 # 15
        'mlp.gate_proj.scales',                  # 16
        'mlp.gate_proj.qzeros',                  # 17
        'mlp.up_proj.qweight',                   # 18
        'mlp.up_proj.scales',                    # 19
        'mlp.up_proj.qzeros',                    # 20
        'mlp.down_proj.qweight',                 # 21
        'mlp.down_proj.scales',                  # 22
        'mlp.down_proj.qzeros',                  # 23
        'input_layernorm.weight',                # 24
        'post_attention_layernorm.weight',       # 25
    ]

    NUM_LAYERS = 64
    SLOTS_PER_LAYER = len(DN_SLOTS)
    assert SLOTS_PER_LAYER == len(FA_SLOTS)

    # Global tensors
    GLOBALS = [
        'model.language_model.embed_tokens.weight',
        'lm_head.weight',
        'model.language_model.norm.weight',
    ]
    NUM_GLOBALS = len(GLOBALS)

    # Build output
    out = bytearray()

    # Header (32 bytes)
    out += struct.pack('<I', 0x4C575449)  # magic
    out += struct.pack('<I', 1)            # version
    out += struct.pack('<I', num_shards)
    out += struct.pack('<I', NUM_LAYERS)
    out += struct.pack('<I', SLOTS_PER_LAYER)
    out += struct.pack('<I', NUM_GLOBALS)
    out += struct.pack('<I', 0)            # reserved
    out += struct.pack('<I', 0)            # reserved

    # Shard table
    for fname, fsize, _, _ in shard_info:
        name_bytes = fname.encode('utf-8')
        padded = name_bytes[:255] + b'\x00' * (256 - min(len(name_bytes), 255))
        out += padded
        out += struct.pack('<Q', fsize)

    # Global tensor table
    for gname in GLOBALS:
        if gname in tensor_index:
            shard_idx, file_offset, size, dtype, shape = tensor_index[gname]
            out += struct.pack('<IIQ Q', shard_idx, 0, file_offset, size)
            print(f"  global {gname}: shard={shard_idx}, offset={file_offset}, size={size}, dtype={dtype}")
        else:
            out += struct.pack('<IIQ Q', 0, 0, 0, 0)
            print(f"  WARNING: global {gname} not found!")

    # Per-layer weight table
    missing = []
    found = 0
    for layer_idx in range(NUM_LAYERS):
        is_fa = (layer_idx % 4 == 3)
        slots = FA_SLOTS if is_fa else DN_SLOTS
        prefix = f'model.language_model.layers.{layer_idx}.'

        for slot_idx, slot_name in enumerate(slots):
            if slot_name == '':
                out += struct.pack('<IIQ Q', 0, 0, 0, 0)
                continue

            full_name = prefix + slot_name
            if full_name in tensor_index:
                shard_idx, file_offset, size, dtype, shape = tensor_index[full_name]
                out += struct.pack('<IIQ Q', shard_idx, 0, file_offset, size)
                found += 1
            else:
                out += struct.pack('<IIQ Q', 0, 0, 0, 0)
                missing.append(full_name)

    print(f"\nResolved {found} layer tensors, {len(missing)} missing")
    if missing:
        print("Missing tensors:")
        for m in missing:
            print(f"  {m}")

    # Write output
    with open(output_file, 'wb') as f:
        f.write(out)

    print(f"\nWrote {len(out)} bytes to {output_file}")
    print(f"  Header: 32 bytes")
    print(f"  Shard table: {num_shards * 264} bytes ({num_shards} shards)")
    print(f"  Global table: {NUM_GLOBALS * 24} bytes")
    print(f"  Layer table: {NUM_LAYERS * SLOTS_PER_LAYER * 24} bytes ({NUM_LAYERS} layers x {SLOTS_PER_LAYER} slots)")


if __name__ == '__main__':
    main()
