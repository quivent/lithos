#!/usr/bin/env python3
"""
Extract the four GSP-FMC blobs (hash, signature, publickey, image) for GH100
from the open-gpu-kernel-modules generated bindata C file.

Source:  /tmp/ogkm/src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmFmcGfwProdSigned_GH100.c
Output:  /home/ubuntu/lithos/GSP/fw/gsp_fmc_{image,hash,pubkey,sig}.bin

All four blobs are uncompressed in this OGKM tree.
Expected sizes: image=165448, hash=48, sig=384, pubkey=384.
"""
import re
import os
import sys

SRC = "/tmp/ogkm/src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmFmcGfwProdSigned_GH100.c"
OUT = "/home/ubuntu/lithos/GSP/fw"

LABELS = {
    "image":  ("BINDATA_LABEL_UCODE_IMAGE", "gsp_fmc_image.bin",  165448),
    "hash":   ("BINDATA_LABEL_UCODE_HASH",  "gsp_fmc_hash.bin",       48),
    "sig":    ("BINDATA_LABEL_UCODE_SIG",   "gsp_fmc_sig.bin",       384),
    "pubkey": ("BINDATA_LABEL_UCODE_PKEY",  "gsp_fmc_pubkey.bin",    384),
}

ARRAY_PREFIX = "kgspBinArchiveGspRmFmcGfwProdSigned_GH100_"

def extract(text, label):
    var = ARRAY_PREFIX + label + "_data"
    # find the line declaring this array
    m = re.search(r"\b" + re.escape(var) + r"\b\s*\[\s*\]\s*=\s*\{", text)
    if not m:
        raise SystemExit(f"array {var} not found")
    # walk until matching closing brace
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        i += 1
    body = text[start:i-1]
    # parse hex bytes
    bytes_ = bytearray()
    for tok in re.findall(r"0x[0-9a-fA-F]+", body):
        bytes_.append(int(tok, 16))
    return bytes(bytes_)

def main():
    with open(SRC) as f:
        text = f.read()
    os.makedirs(OUT, exist_ok=True)
    for kind, (label, fname, expect_size) in LABELS.items():
        data = extract(text, label)
        if len(data) != expect_size:
            print(f"WARN {kind}: got {len(data)} bytes, expected {expect_size}", file=sys.stderr)
        outpath = os.path.join(OUT, fname)
        with open(outpath, "wb") as f:
            f.write(data)
        print(f"{outpath}: {len(data)} bytes")

if __name__ == "__main__":
    main()
