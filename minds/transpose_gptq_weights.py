#!/usr/bin/env python3
"""
Transpose GPTQ qweight tensors from [K/8, N] row-major to [N, K/8] column-major.

The original layout stores weights as qweight[k_packed, col] with row-major C order,
meaning iterating along the K dimension (inner GEMV loop) strides by N*4 bytes.
For N=12288 that's ~49KB per K step -- terrible for cache/coalescing.

The transposed layout stores qweight[col, k_packed], so iterating along K is
stride-1 (4 bytes between consecutive K entries for the same column). This makes
the GEMV inner loop sequential in memory.

Scales are similarly transposed from [n_groups, N] to [N, n_groups].
qzeros from [n_groups, N/8] to [N/8, n_groups].

Usage:
    python3 transpose_gptq_weights.py /path/to/model /path/to/output
"""

import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def transpose_safetensors_file(src_path, dst_path):
    """Transpose all qweight/scales/qzeros tensors in a safetensors file."""
    f = safe_open(src_path, framework='pt')
    keys = list(f.keys())

    tensors = {}
    n_transposed = 0

    for key in keys:
        tensor = f.get_tensor(key)

        if key.endswith('.qweight'):
            # [K/8, N] -> [N, K/8]
            assert tensor.ndim == 2, f"{key}: expected 2D, got {tensor.ndim}D"
            K_packed, N = tensor.shape
            tensor_t = tensor.T.contiguous()
            assert tensor_t.shape == (N, K_packed), f"{key}: transpose failed"
            tensors[key] = tensor_t
            n_transposed += 1

        elif key.endswith('.scales'):
            # [n_groups, N] -> [N, n_groups]
            assert tensor.ndim == 2, f"{key}: expected 2D, got {tensor.ndim}D"
            tensor_t = tensor.T.contiguous()
            tensors[key] = tensor_t
            n_transposed += 1

        elif key.endswith('.qzeros'):
            # [n_groups, N/8] -> [N/8, n_groups]
            assert tensor.ndim == 2, f"{key}: expected 2D, got {tensor.ndim}D"
            tensor_t = tensor.T.contiguous()
            tensors[key] = tensor_t
            n_transposed += 1

        else:
            # Non-quantized tensors: copy as-is
            tensors[key] = tensor

    save_file(tensors, dst_path)
    return n_transposed, len(keys)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <src_model_dir> <dst_model_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    os.makedirs(dst_dir, exist_ok=True)

    # Find all safetensors files
    st_files = sorted(f for f in os.listdir(src_dir) if f.endswith('.safetensors'))
    if not st_files:
        print("No safetensors files found!")
        sys.exit(1)

    print(f"Transposing GPTQ weights: [K/8, N] -> [N, K/8]")
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Files: {len(st_files)}")
    print()

    total_transposed = 0
    total_tensors = 0
    t0 = time.time()

    for st_file in st_files:
        src_path = os.path.join(src_dir, st_file)
        dst_path = os.path.join(dst_dir, st_file)

        print(f"  {st_file}...", end=' ', flush=True)
        t1 = time.time()
        n_trans, n_total = transpose_safetensors_file(src_path, dst_path)
        dt = time.time() - t1
        print(f"{n_trans}/{n_total} tensors transposed in {dt:.1f}s")

        total_transposed += n_trans
        total_tensors += n_total

    elapsed = time.time() - t0
    print(f"\nDone: {total_transposed}/{total_tensors} tensors transposed in {elapsed:.1f}s")

    # Copy non-safetensors files (config, tokenizer, etc.)
    for fname in os.listdir(src_dir):
        if fname.endswith('.safetensors'):
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"  Copied {fname}")

    # Update index file to note transposition
    idx_path = os.path.join(dst_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        if 'metadata' not in idx:
            idx['metadata'] = {}
        idx['metadata']['qweight_layout'] = 'transposed_NK'
        idx['metadata']['original_layout'] = 'KN_row_major'
        with open(idx_path, 'w') as f:
            json.dump(idx, f, indent=2)
        print("  Updated index metadata with layout info")

    print(f"\nTransposed model saved to: {dst_dir}")


if __name__ == '__main__':
    main()
