#!/usr/bin/env python3
"""
Build a binary weight index for the native ARM64 launcher.

Reads safetensors shard headers and emits a flat binary file that the launcher
can mmap directly -- no JSON parsing needed in assembly.

Binary format:
  Header (64 bytes):
    magic           u32   0x4C574958  ("LWIX" = Lithos Weight IndeX)
    version         u32   2
    num_shards      u32
    num_layers      u32   64
    tensors_per_layer u32 40 (union of linear_attn + self_attn + mlp + norms)
    entry_size      u32   32 (bytes per tensor entry)
    shard_table_off u64   offset to shard filename table
    layer_table_off u64   offset to layer records
    reserved        24 bytes (pad to 64)

  Shard filename table (starts at shard_table_off):
    For each shard: 256-byte null-padded filename (basename only)

  Layer records (starts at layer_table_off):
    For each layer (0..63), a fixed-size block of tensor entries.
    Each tensor entry (32 bytes):
      shard_index   u32   which shard file (0-based)
      _pad0         u32   alignment padding
      byte_offset   u64   absolute file offset to tensor data
      byte_size     u64   size of tensor data in bytes
      num_elements  u64   total number of elements

    Tensors stored in fixed order (see TENSOR_SLOTS).
    Hybrid architecture: layers 0,1,2,4,5,6,8,... use linear_attn (DeltaNet),
    every 4th layer (3,7,11,...,63) uses standard self_attn.
    Unused slots for a given layer type are filled with sentinel 0xFF..FF.

    The launcher checks shard_index != 0xFFFFFFFF to know if a slot is populated.

Usage:
    python3 build_weight_index.py /path/to/model/dir [output.bin]
"""

import json
import struct
import sys
from pathlib import Path

# --- Fixed tensor slot ordering (40 slots) ---
# Slots 0-14:  linear_attn tensors (populated for DeltaNet layers)
# Slots 15-28: self_attn tensors (populated for standard attention layers)
# Slots 29-37: mlp tensors (always populated)
# Slots 38-39: layernorms (always populated)

TENSOR_SLOTS = [
    # -- linear_attn block (slots 0-14) --
    "linear_attn.in_proj_qkv.qweight",        #  0
    "linear_attn.in_proj_qkv.qzeros",         #  1
    "linear_attn.in_proj_qkv.scales",          #  2
    "linear_attn.in_proj_b.weight",            #  3
    "linear_attn.in_proj_a.weight",            #  4
    "linear_attn.in_proj_z.qweight",           #  5
    "linear_attn.in_proj_z.qzeros",            #  6
    "linear_attn.in_proj_z.scales",            #  7
    "linear_attn.out_proj.qweight",            #  8
    "linear_attn.out_proj.qzeros",             #  9
    "linear_attn.out_proj.scales",             # 10
    "linear_attn.conv1d.weight",               # 11
    "linear_attn.A_log",                       # 12
    "linear_attn.dt_bias",                     # 13
    "linear_attn.norm.weight",                 # 14
    # -- self_attn block (slots 15-28) --
    "self_attn.q_proj.qweight",                # 15
    "self_attn.q_proj.qzeros",                 # 16
    "self_attn.q_proj.scales",                 # 17
    "self_attn.k_proj.qweight",                # 18
    "self_attn.k_proj.qzeros",                 # 19
    "self_attn.k_proj.scales",                 # 20
    "self_attn.v_proj.qweight",                # 21
    "self_attn.v_proj.qzeros",                 # 22
    "self_attn.v_proj.scales",                 # 23
    "self_attn.o_proj.qweight",                # 24
    "self_attn.o_proj.qzeros",                 # 25
    "self_attn.o_proj.scales",                 # 26
    "self_attn.q_norm.weight",                 # 27
    "self_attn.k_norm.weight",                 # 28
    # -- mlp block (slots 29-37) --
    "mlp.gate_proj.qweight",                   # 29
    "mlp.gate_proj.qzeros",                    # 30
    "mlp.gate_proj.scales",                    # 31
    "mlp.up_proj.qweight",                     # 32
    "mlp.up_proj.qzeros",                      # 33
    "mlp.up_proj.scales",                      # 34
    "mlp.down_proj.qweight",                   # 35
    "mlp.down_proj.qzeros",                    # 36
    "mlp.down_proj.scales",                    # 37
    # -- layernorms (slots 38-39) --
    "input_layernorm.weight",                  # 38
    "post_attention_layernorm.weight",         # 39
]

MAGIC = 0x4C574958  # "LWIX"
VERSION = 2
SHARD_NAME_SIZE = 256
TENSOR_ENTRY_SIZE = 32  # 4 + 4 + 8 + 8 + 8
HEADER_SIZE = 64
NUM_LAYERS = 64
TENSORS_PER_LAYER = len(TENSOR_SLOTS)

# Sentinel for missing tensors
MISSING_U32 = 0xFFFFFFFF
MISSING_U64 = 0xFFFFFFFFFFFFFFFF


def read_safetensors_header(path):
    """Read and parse the JSON header from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
    header = json.loads(header_json)
    data_start = 8 + header_size
    return header, data_start


def compute_num_elements(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [output.bin]", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(model_dir / "weight_index.bin")

    # Find shard files
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if not shard_files:
        print(f"ERROR: No model-*.safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    num_shards = len(shard_files)
    print(f"Found {num_shards} shard files")

    # Parse all shard headers
    tensor_map = {}  # full_name -> (shard_idx, data_start, info_dict)
    for shard_idx, shard_path in enumerate(shard_files):
        header, data_start = read_safetensors_header(shard_path)
        count = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            tensor_map[name] = (shard_idx, data_start, info)
            count += 1
        print(f"  Shard {shard_idx} ({shard_path.name}): {count} tensors, data @ byte {data_start}")

    layer_prefix = "model.language_model.layers."

    # Detect layer types
    linear_attn_layers = set()
    self_attn_layers = set()
    for name in tensor_map:
        if not name.startswith(layer_prefix):
            continue
        rest = name[len(layer_prefix):]
        layer_num = int(rest.split(".")[0])
        if ".linear_attn." in name:
            linear_attn_layers.add(layer_num)
        elif ".self_attn." in name:
            self_attn_layers.add(layer_num)

    print(f"\n  DeltaNet (linear_attn) layers: {len(linear_attn_layers)} "
          f"({min(linear_attn_layers)}..{max(linear_attn_layers)} excl every 4th)")
    print(f"  Standard (self_attn) layers:   {len(self_attn_layers)} "
          f"({sorted(self_attn_layers)[:4]}...{sorted(self_attn_layers)[-2:]})")

    # Build binary output
    shard_table_off = HEADER_SIZE
    shard_table_size = num_shards * SHARD_NAME_SIZE
    layer_table_off = shard_table_off + shard_table_size
    record_size = TENSORS_PER_LAYER * TENSOR_ENTRY_SIZE
    total_size = layer_table_off + NUM_LAYERS * record_size

    buf = bytearray(total_size)

    # Write header
    struct.pack_into("<IIIIII", buf, 0,
                     MAGIC, VERSION, num_shards, NUM_LAYERS, TENSORS_PER_LAYER, TENSOR_ENTRY_SIZE)
    struct.pack_into("<QQ", buf, 24, shard_table_off, layer_table_off)

    # Write shard filename table
    for i, shard_path in enumerate(shard_files):
        name_bytes = shard_path.name.encode("utf-8")
        offset = shard_table_off + i * SHARD_NAME_SIZE
        buf[offset:offset + len(name_bytes)] = name_bytes

    # Write layer records
    total_found = 0
    total_missing = 0

    for layer in range(NUM_LAYERS):
        layer_off = layer_table_off + layer * record_size

        for slot_idx, slot_name in enumerate(TENSOR_SLOTS):
            entry_off = layer_off + slot_idx * TENSOR_ENTRY_SIZE
            full_name = f"{layer_prefix}{layer}.{slot_name}"

            if full_name in tensor_map:
                shard_idx, data_start, info = tensor_map[full_name]
                shape = info["shape"]
                data_offsets = info["data_offsets"]

                byte_offset = data_start + data_offsets[0]
                byte_size = data_offsets[1] - data_offsets[0]
                num_elements = compute_num_elements(shape)

                struct.pack_into("<IIQQQ", buf, entry_off,
                                 shard_idx, 0, byte_offset, byte_size, num_elements)
                total_found += 1
            else:
                struct.pack_into("<IIQQQ", buf, entry_off,
                                 MISSING_U32, MISSING_U32,
                                 MISSING_U64, MISSING_U64, MISSING_U64)
                total_missing += 1

    # Write output
    with open(output_path, "wb") as f:
        f.write(buf)

    print(f"\nWrote {output_path}")
    print(f"  Total size: {total_size} bytes ({total_size / 1024:.1f} KB)")
    print(f"  {num_shards} shards, {NUM_LAYERS} layers, {TENSORS_PER_LAYER} slots/layer")
    print(f"  {total_found} tensors indexed, {total_missing} slots unused (expected for hybrid arch)")
    print(f"  Shard table: offset {shard_table_off}, {shard_table_size} bytes")
    print(f"  Layer table: offset {layer_table_off}, {record_size} bytes/layer")

    # Sanity check: verify a DeltaNet layer and a self_attn layer
    print(f"\n--- Sanity check ---")
    for layer in [0, 3, 63]:
        is_selfattn = layer in self_attn_layers
        tag = "self_attn" if is_selfattn else "linear_attn"
        print(f"\n  Layer {layer} ({tag}):")
        for slot_idx, slot_name in enumerate(TENSOR_SLOTS):
            entry_off = layer_table_off + layer * record_size + slot_idx * TENSOR_ENTRY_SIZE
            si, _, bo, bs, ne = struct.unpack_from("<IIQQQ", buf, entry_off)
            if si == MISSING_U32:
                status = "  --"
            else:
                status = f"shard={si} off=0x{bo:012x} size={bs:>10,} elems={ne:>10,}"
            print(f"    [{slot_idx:2d}] {slot_name:45s} {status}")

    # Cross-verify: read back and check a known tensor
    print(f"\n--- Cross-verification ---")
    test_name = f"{layer_prefix}0.linear_attn.A_log"
    shard_idx, data_start, info = tensor_map[test_name]
    expected_off = data_start + info["data_offsets"][0]
    expected_size = info["data_offsets"][1] - info["data_offsets"][0]

    entry_off = layer_table_off + 0 * record_size + 12 * TENSOR_ENTRY_SIZE  # slot 12 = A_log
    si, _, bo, bs, ne = struct.unpack_from("<IIQQQ", buf, entry_off)
    ok = si == shard_idx and bo == expected_off and bs == expected_size
    print(f"  layer0.A_log: index says shard={si} off={bo} size={bs} | {'OK' if ok else 'MISMATCH'}")

    # Verify actual bytes from shard match expected F32 values
    shard_path = shard_files[si]
    import numpy as np
    with open(shard_path, "rb") as f:
        f.seek(bo)
        data = np.frombuffer(f.read(bs), dtype=np.float32)
    print(f"  A_log values (first 8 of {len(data)}): {data[:8]}")
    print(f"  A_log range: [{data.min():.4f}, {data.max():.4f}]")


if __name__ == "__main__":
    main()
