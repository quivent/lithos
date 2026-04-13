# GSP-FMC bindata for GH100 (GH200)

Four binary blobs that the FSP Chain-of-Trust (COT) requires to launch the
GSP firmware microcontroller on a Hopper-class GPU:

| File                  | Size (B) | Purpose                                                |
|-----------------------|---------:|--------------------------------------------------------|
| `gsp_fmc_image.bin`   |  165 448 | The GSP-FMC ucode image (RISC-V) loaded into WPR2     |
| `gsp_fmc_hash.bin`    |       48 | SHA-384 of the image (matches `sha384sum image`)      |
| `gsp_fmc_pubkey.bin`  |      384 | RSA-3072 public key (modulus, big-endian)             |
| `gsp_fmc_sig.bin`     |      384 | RSA-3072 PKCS#1 v1.5 signature over the hash          |

## Provenance

Extracted from the open-gpu-kernel-modules (OGKM) source tree on this host:

```
/tmp/ogkm/src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmFmcGfwProdSigned_GH100.c
```

Variant: **`Prod`-signed** (production fuse), GH100 (Hopper, GH200 480GB
silicon — confirmed by `nvidia-smi -L`).  The bindata file declares four
arrays, all uncompressed in this OGKM revision:

| Bindata label              | C array suffix                                                | Size  |
|----------------------------|---------------------------------------------------------------|------:|
| `BINDATA_LABEL_UCODE_IMAGE`| `kgspBinArchiveGspRmFmcGfwProdSigned_GH100_..._IMAGE_data`   | 165448|
| `BINDATA_LABEL_UCODE_HASH` | `..._HASH_data`                                              |     48|
| `BINDATA_LABEL_UCODE_SIG`  | `..._SIG_data`                                               |    384|
| `BINDATA_LABEL_UCODE_PKEY` | `..._PKEY_data`                                              |    384|

The OGKM array naming differs from what
`/tmp/ogkm/nouveau/extract-firmware-nouveau.py::fmc()` assumes
(`*_ucode_{hash,sig,pkey,image}_data`); the Nouveau script targets an older
OGKM revision and would not match these arrays.  We therefore extract
directly with `extract_gh100.py` (a small, self-contained brace-walking
parser — no ELF wrapper, just raw blobs).

## Verification

`sha384(gsp_fmc_image.bin)` was computed and matches `gsp_fmc_hash.bin`
byte-for-byte:

```
568d1f6c0fecc2fadcd5a6cb42a66da02b80fc87f24f3048b0e8330cafbf6231
024e58dc94bffc5a5b7d223adb985258
```

This proves the four blobs are mutually consistent: the public key in
`gsp_fmc_pubkey.bin` signed `gsp_fmc_hash.bin` to produce
`gsp_fmc_sig.bin`, and that hash is the SHA-384 of the image we extracted.

## Re-running the extractor

```
python3 /home/ubuntu/lithos/GSP/fw/extract_gh100.py
```

It re-reads the OGKM C file in place; no other inputs.

## Chips covered

GH100 only.  The same OGKM tree also ships:

- `..._GspRmFmcGfwProdSigned_GB100.c`   (Blackwell B100 prod)
- `..._GspRmFmcGfwProdSigned_GB10B.c`   (Blackwell GB10 prod, e.g. GB200)
- `..._GspRmFmcGfwProdSigned_GB202.c`   (Blackwell B200 prod)
- `..._GspRmFmcGfwProdSigned_GB20B.c`
- `..._GspRmCcFmcGfwProdSigned_GH100.c` (Confidential-Compute variant)
- Debug-signed counterparts for each.

Lithos targets GH200 (= GH100 silicon) per `README.md`, so the Prod-signed
GH100 blobs are the canonical pick.  For other targets, change
`SRC` and `ARRAY_PREFIX` in `extract_gh100.py`.

## How `fw_load.s` and `fsp/cot_payload.s` should consume these

The current `fw_load.s` parses an ELF file (the GSP-RM firmware ELF that
ships in `/lib/firmware/nvidia/.../gsp_*.bin` — that's a separate artefact
covering `.fwimage`/`.fwsignature_*`).  The FMC blobs in this directory
are a *different* artefact and feed the **FSP COT payload**, not
`fw_load.s`.  Suggested integration:

1. **Init-time load.** Add a small helper (e.g. `fmc_load.s`) that, at
   startup, opens each of the four files via `openat`/`read`, mmaps or
   copies them into the BAR4 / WPR2 staging buffer, and records:

   - `fmc_image_cpu`, `fmc_image_phys`, `fmc_image_size  = 165448`
   - `fmc_hash_cpu`,  `fmc_hash_phys`,  `fmc_hash_size   = 48`
   - `fmc_pubkey_cpu`,`fmc_pubkey_phys`,`fmc_pubkey_size = 384`
   - `fmc_sig_cpu`,   `fmc_sig_phys`,   `fmc_sig_size    = 384`

   The image must be placed at the WPR2 base that the COT payload
   advertises to FSP; the hash/pubkey/sig get embedded in the COT struct
   itself (see `fsp/registers.s` and `fsp_plan.md` lines 194-231).

2. **COT struct fill-in.** `cot_build_payload` (per `fsp_plan.md`)
   accepts `fmc_phys` and the byte offsets of hash/pubkey/sig within the
   payload.  Concretely it should:

   - Copy the 48-byte hash into the COT `fmc.hash384` field.
   - Copy the 384-byte pubkey modulus into the COT `fmc.pubkey` field.
   - Copy the 384-byte signature into the COT `fmc.signature` field.
   - Write `fmc_phys` and `fmc_size = 165448` into the
     `fmc.image_pa` / `fmc.image_size` fields.

3. **Hand-off.** `bcr_start.s` then sets BCR registers per
   `boot.s:276-303` (`x2 = fmc_image_pa = fw_bar4_phys`) and the FSP
   verifies the COT before releasing the GSP from reset.

## Status

- [x] All four blobs extracted, sizes match spec exactly.
- [x] SHA-384 of image matches the extracted hash blob (sanity check).
- [ ] Wire `fmc_load.s` and `cot_build_payload` to consume these files
      (next step; tracked in `fsp_plan.md`).
- [ ] Optional: add a build-time check that the on-disk
      `gsp_fmc_hash.bin` still matches `sha384sum gsp_fmc_image.bin`
      after any re-extraction.

## If the OGKM tree disappears

If `/tmp/ogkm` is wiped, re-clone it:

```
git clone --depth 1 https://github.com/NVIDIA/open-gpu-kernel-modules /tmp/ogkm
```

The bindata C files are checked in (auto-generated from the upstream
proprietary RM tree) and will be at the same paths.  `extract_gh100.py`
will then work unchanged.
