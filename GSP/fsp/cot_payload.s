// cot_payload.s -- Chain-of-Trust (COT) payload population for FSP boot
//
// Builds NVDM_PAYLOAD_COT (860 bytes) by extracting hash/pubkey/signature
// from the GSP firmware manifest (the "manifest" section carved out of
// .fwimage by Step 5 / fw_load.s).
//
// Struct layout (from /tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/
// kern_fsp_cot_payload.h, #pragma pack(1)):
//
//   offset  size  field
//    0      2     version
//    2      2     size
//    4      8     gspFmcSysmemOffset
//   12      8     frtsSysmemOffset
//   20      4     frtsSysmemSize
//   24      8     frtsVidmemOffset
//   32      4     frtsVidmemSize
//   36     48     hash384[12]          (SHA-384)
//   84    384     publicKey[96]        (RSA-3072 public key)
//  468    384     signature[96]        (RSA-3072 signature)
//  852      8     gspBootArgsSysmemOffset
//   ---------
//  860 total
//
// The C reference (kern_fsp_gh100.c::kfspSetupGspImages) pulls hash,
// pubkey, and signature from *separate* bindata blobs labelled
// UCODE_HASH / UCODE_SIG / UCODE_PKEY.  In the GSP firmware ELF used by
// Lithos (gsp_ga10x.bin / gsp_gh100.bin), these three blobs are
// concatenated into the single "manifest" region that fw_load.s locates
// via fw_manifest_offset.  The byte layout within that manifest is:
//
//     +MANIFEST_HASH_OFFSET        48 bytes  SHA-384 digest
//     +MANIFEST_PUBKEY_OFFSET     384 bytes  RSA-3072 public key
//     +MANIFEST_SIG_OFFSET        384 bytes  RSA-3072 signature
//
// The offsets below match the signed-FMC manifest layout documented in
// the OpenRM bindata archive headers for GH100.  They are HARDWARE /
// FIRMWARE-VERSION SPECIFIC and must be re-verified when switching to a
// different gsp_*.bin revision.  See the comment block at the end of
// this file for the verification procedure.
//
// Build:
//   as -o cot_payload.o /home/ubuntu/lithos/GSP/fsp/cot_payload.s

// ============================================================
// Hardware/firmware-version specific constants
// --- MUST BE VERIFIED AGAINST THE INSTALLED GSP FIRMWARE ---
// ============================================================

// Offsets within the manifest blob (relative to fw_manifest_offset
// within .fwimage, i.e. relative to (fw_bar4_cpu + fw_manifest_offset)).
.equ MANIFEST_HASH_OFFSET,    0x000    // SHA-384 digest  (48 B)
.equ MANIFEST_PUBKEY_OFFSET,  0x030    // 48
.equ MANIFEST_SIG_OFFSET,     0x1B0    // 48 + 384 = 432 = 0x1B0

// COT payload field sizes
.equ COT_HASH_BYTES,      48
.equ COT_PUBKEY_BYTES,    384
.equ COT_SIG_BYTES,       384
.equ COT_PAYLOAD_BYTES,   860

// Field offsets within NVDM_PAYLOAD_COT
.equ COT_OFF_VERSION,     0
.equ COT_OFF_SIZE,        2
.equ COT_OFF_GSP_FMC,     4
.equ COT_OFF_FRTS_SYS,    12
.equ COT_OFF_FRTS_SYSSZ,  20
.equ COT_OFF_FRTS_VID,    24
.equ COT_OFF_FRTS_VIDSZ,  32
.equ COT_OFF_HASH,        36
.equ COT_OFF_PUBKEY,      84
.equ COT_OFF_SIG,         468
.equ COT_OFF_BOOT_ARGS,   852

// Version constant (GH100 / GA10x: 1)
.equ COT_VERSION,         1


// ============================================================
// Text
// ============================================================

.text

// ---------------------------------------------------------------
// cot_extract_hash(manifest, dst)
//   x0 = manifest base pointer (CPU VA)
//   x1 = destination (>= 48 bytes)
// Copies 48 bytes of SHA-384 hash out of the manifest.
// Clobbers: x0, x1, x2, x3, x4
// ---------------------------------------------------------------
.globl cot_extract_hash
cot_extract_hash:
    add     x0, x0, #MANIFEST_HASH_OFFSET
    mov     x2, #COT_HASH_BYTES
    b       .cot_memcpy

// ---------------------------------------------------------------
// cot_extract_pubkey(manifest, dst)
//   x0 = manifest base pointer
//   x1 = destination (>= 384 bytes)
// Copies 384 bytes of RSA-3072 public key.
// ---------------------------------------------------------------
.globl cot_extract_pubkey
cot_extract_pubkey:
    add     x0, x0, #MANIFEST_PUBKEY_OFFSET
    mov     x2, #COT_PUBKEY_BYTES
    b       .cot_memcpy

// ---------------------------------------------------------------
// cot_extract_signature(manifest, dst)
//   x0 = manifest base pointer
//   x1 = destination (>= 384 bytes)
// Copies 384 bytes of RSA-3072 signature.
// ---------------------------------------------------------------
.globl cot_extract_signature
cot_extract_signature:
    mov     x2, #MANIFEST_SIG_OFFSET
    add     x0, x0, x2
    mov     x2, #COT_SIG_BYTES
    b       .cot_memcpy

// ---------------------------------------------------------------
// cot_build_payload(fw_image_bar4_cpu, fw_manifest_offset, dst_buf)
//   x0 = firmware image CPU VA  (fw_bar4_cpu)
//   x1 = manifest offset within image  (fw_manifest_offset)
//   x2 = destination buffer  (>= 860 bytes)
//
// Zero-fills dst, then populates version/size and copies hash, pubkey,
// and signature from manifest at the documented offsets.  Leaves the
// sysmem / vidmem offset fields at zero; the orchestrator (bootcmd.s)
// patches them in from Step 4 / Step 5 outputs before MCTP send.
//
// Returns: w0 = COT_PAYLOAD_BYTES (860)
// Clobbers: x0-x7, x30
// ---------------------------------------------------------------
.globl cot_build_payload
cot_build_payload:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]

    // --- Validate inputs before trusting them ---
    // x0 = fw_image_bar4_cpu, x1 = manifest_offset, x2 = dst_buf
    cbz     x0, .cot_build_fail        // NULL firmware image pointer
    cbz     x2, .cot_build_fail        // NULL destination buffer

    // x19 = manifest base = fw_image_base + manifest_offset
    add     x19, x0, x1
    mov     x20, x2                    // x20 = dst buffer base

    // ---- 1. Zero-fill dst [0 .. 860) ----
    mov     x3, x20
    mov     x4, #COT_PAYLOAD_BYTES
    // Bulk 32-byte stp xzr,xzr (2 pairs) loop
.cot_zero_bulk:
    cmp     x4, #32
    b.lt    .cot_zero_tail8
    stp     xzr, xzr, [x3], #16
    stp     xzr, xzr, [x3], #16
    sub     x4, x4, #32
    b       .cot_zero_bulk
.cot_zero_tail8:
    cmp     x4, #8
    b.lt    .cot_zero_tail1
    str     xzr, [x3], #8
    sub     x4, x4, #8
    b       .cot_zero_tail8
.cot_zero_tail1:
    cbz     x4, .cot_zero_done
.cot_zero_byte:
    strb    wzr, [x3], #1
    subs    x4, x4, #1
    b.ne    .cot_zero_byte
.cot_zero_done:

    // ---- 2. Populate version/size ----
    mov     w3, #COT_VERSION
    strh    w3, [x20, #COT_OFF_VERSION]
    mov     w3, #COT_PAYLOAD_BYTES
    strh    w3, [x20, #COT_OFF_SIZE]

    // sysmem / vidmem / gspFmcSysmemOffset / gspBootArgsSysmemOffset
    // are left zero -- bootcmd.s patches them in after this returns.

    // ---- 3. Copy hash384 (48 bytes) from manifest+HASH_OFFSET to
    //         dst + COT_OFF_HASH ----
    mov     x0, x19
    add     x1, x20, #COT_OFF_HASH
    bl      cot_extract_hash

    // ---- 4. Copy publicKey (384 bytes) ----
    mov     x0, x19
    add     x1, x20, #COT_OFF_PUBKEY
    bl      cot_extract_pubkey

    // ---- 5. Copy signature (384 bytes) ----
    mov     x0, x19
    add     x1, x20, #COT_OFF_SIG
    bl      cot_extract_signature

    // ---- 6. Sanity-check: hash must not be all-zero ----
    // An all-zero SHA-384 means the manifest region was unmapped/corrupt.
    // Check first 8 bytes; if zero, spot-check bytes 8..15 and 16..23.
    ldr     x3, [x20, #COT_OFF_HASH]
    cbnz    x3, .cot_hash_ok
    ldr     x3, [x20, #(COT_OFF_HASH + 8)]
    cbnz    x3, .cot_hash_ok
    ldr     x3, [x20, #(COT_OFF_HASH + 16)]
    cbnz    x3, .cot_hash_ok
    // All three 8-byte chunks are zero -- almost certainly invalid
    b       .cot_build_fail

.cot_hash_ok:
    // ---- 7. Return payload length ----
    mov     w0, #COT_PAYLOAD_BYTES

    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

.cot_build_fail:
    mov     x0, #-1                    // return negative = failure
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ---------------------------------------------------------------
// .cot_memcpy -- copy x2 bytes from x0 to x1 (internal helper)
//
// Style matches fw_load.s .memcpy64: 64-byte unrolled bulk, byte tail.
// Arguments are consumed (x0/x1 are advanced, x2 decremented).
// Clobbers: x0, x1, x2, x3, x4
// ---------------------------------------------------------------
.cot_memcpy:
    cmp     x2, #64
    b.lt    .cot_memcpy_tail
.cot_memcpy_bulk:
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    ldp     x3, x4, [x0], #16
    stp     x3, x4, [x1], #16
    sub     x2, x2, #64
    cmp     x2, #64
    b.ge    .cot_memcpy_bulk
.cot_memcpy_tail:
    cbz     x2, .cot_memcpy_done
.cot_memcpy_byte:
    ldrb    w3, [x0], #1
    strb    w3, [x1], #1
    subs    x2, x2, #1
    b.ne    .cot_memcpy_byte
.cot_memcpy_done:
    ret

// ============================================================
// VERIFICATION NOTE -- manifest offset constants
// ============================================================
//
// MANIFEST_HASH_OFFSET / MANIFEST_PUBKEY_OFFSET / MANIFEST_SIG_OFFSET
// above assume the GH100 signed-FMC manifest layout:
//
//     [0x000 .. 0x030)  SHA-384 hash          (48 B)
//     [0x030 .. 0x1B0)  RSA-3072 public key   (384 B)
//     [0x1B0 .. 0x330)  RSA-3072 signature    (384 B)
//
// This matches the concatenation order of the BINDATA_LABEL_UCODE_HASH
// / UCODE_PKEY / UCODE_SIG storage blobs emitted by NVIDIA's bindata
// tool when building gsp_*.bin.  However, some firmware revisions pad
// or reorder these fields.  To re-verify against a particular
// gsp_*.bin installed under /lib/firmware/nvidia/<ver>/:
//
//   1. Locate the .fwimage ELF section, read the fwimage header to
//      obtain manifest_offset / manifest_size (fw_load.s already does
//      this; values land in fw_manifest_offset / via the .fwimage
//      header at +0x00 / +0x08).
//   2. Hex-dump the manifest region:
//        tools/dump_qmd <gsp_*.bin> <manifest_offset> <manifest_size>
//      Expected first 48 B = high-entropy SHA-384 digest; next 384 B
//      leading with 0x00 0x00 0x00 0xC0 (RSA modulus length prefix);
//      trailing 384 B likewise.
//   3. If the layout does not match, update the three .equ lines at
//      the top of this file.  Sizes (48 / 384 / 384) are fixed by the
//      NVDM_PAYLOAD_COT ABI and must not change.
//
// The reference driver avoids this entirely by pulling each blob
// from a separately-labelled bindata storage entry; we cannot use that
// path because the shipping .bin concatenates them before install.
