# Step 7: FSP (Firmware Security Processor) Communication — Implementation Plan

Status: UNSTARTED. Steps 1-6 of the GSP boot sequence already live in
`/home/ubuntu/lithos/GSP/*.s` (bar_map, pmc_check, falcon_reset, hbm_alloc,
fw_load, bcr_start, poll_lockdown, rpc_channel). This document plans the
remaining ~2,000-line wall: the FSP boot-command exchange that authorizes the
GSP's FMC before PRIV_LOCKDOWN releases.

Reference source (for porting, not linking):
- `/tmp/ogkm/src/nvidia/src/kernel/gpu/fsp/kern_fsp.c`                (988 LOC — core queue/MCTP)
- `/tmp/ogkm/src/nvidia/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c` (1725 LOC — GH100/GH200 HAL)
- `/tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h`
- `/tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_mctp_pkt_state.h`
- `/tmp/ogkm/src/common/inc/swref/published/hopper/gh100/dev_fsp_pri.h`

Total reference: ~2,713 LOC of C. Target: ~1,800-2,200 LOC of ARM64 asm
(assembly is less dense for struct population but denser for register pokes).

---

## 1. Register Layout (BAR0 + 0x8F2000 region)

All are 32-bit MMIO registers. `str w?, [x_bar0, #offset]` / `ldr w?, [...]`.

| Symbol                         | Offset                | Dir | Purpose                                                                      |
|--------------------------------|-----------------------|-----|------------------------------------------------------------------------------|
| `NV_PFSP_EMEMC(i)`             | `0x008F2AC0 + i*8`    | RW  | EMEM control: sets BLK[15:8] (256B block), OFFS[7:2] (DW offset), AINCW/AINCR auto-increment |
| `NV_PFSP_EMEMD(i)`             | `0x008F2AC4 + i*8`    | RW  | EMEM data: reads/writes one 32-bit word at the address programmed by EMEMC; pointer advances if AINCW/R set |
| `NV_PFSP_QUEUE_HEAD(i)`        | `0x008F2C00 + i*8`    | RW  | CPU->FSP command queue head (bytes). CPU writes to push a message.            |
| `NV_PFSP_QUEUE_TAIL(i)`        | `0x008F2C04 + i*8`    | RW  | CPU->FSP command queue tail. FSP advances as it consumes.                     |
| `NV_PFSP_MSGQ_HEAD(i)`         | `0x008F2C80 + i*8`    | RW  | FSP->CPU response queue head. FSP advances to push a response.                |
| `NV_PFSP_MSGQ_TAIL(i)`         | `0x008F2C84 + i*8`    | RW  | FSP->CPU response queue tail. CPU advances after consuming.                   |
| `NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2(i)` | `0x008F0320 + i*4` | RW | Boot status scratch words. Polled for FSP readiness / error codes.          |
| `NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_3(i)` | `0x008F0330 + i*4` | RW | Secondary boot status scratch.                                               |
| (GSP target mask release)      | query via scratch     | R   | FSP clears this once COT is accepted; gates poll_lockdown (Step 8).           |

`__SIZE_1 = 8` for EMEMC/EMEMD/QUEUE_HEAD/MSGQ_HEAD — there are 8 channels. GH100 uses channel index 0 for FSP's RM/ucode task and occasionally others for secondary services; our port only needs channel 0.

EMEM layout: EMEM is a 4 KB scratch SRAM inside the FSP Falcon. It is
logically divided into 256-byte blocks (BLK) of 64 DWs each (OFFS). Writing
a word is a three-step dance:
1. Program EMEMC(ch) with BLK, OFFS=0, AINCW=1.
2. Stream N DW to EMEMD(ch); pointer auto-advances.
3. For reads, program AINCR=1 and stream from EMEMD(ch).

Per-channel partition: each channel owns a contiguous slice of EMEM. The
reference driver uses `cmdQPhysicalOffset = 0` and `msgQPhysicalOffset = 2 KB`
for channel 0. Sizes are also fixed (see kern_fsp_gh100.c — queue geometry is
GPU-hardcoded, no runtime negotiation).

---

## 2. COT / MCTP Data Structures

All are `#pragma pack(1)` — no padding. Little-endian wire order.

### 2.1 MCTP transport envelope (per-packet, 4 bytes)
Each 256-byte EMEM packet begins with:
```
struct MCTP_HEADER {        // 7 bytes, but first 4 are a single DW "constBlob"
    u32 constBlob;          // magic + SOM/EOM flags + seq + dst/src EID (see kern_fsp.c)
    u8  msgType;            // 0x7E = vendor-defined PCI (NVDM)
    u16 vendorId;           // 0x10DE (NVIDIA)
};
```
Plus a 4-byte NVDM type header preceding payload:
```
struct NVDM_HEADER {
    u32 nvdmType;           // e.g. NVDM_TYPE_COT = 0x15 (exact value — see kern_fsp.h)
};
```
The first packet of a message carries both headers; intermediate/end packets
only carry the MCTP header with SOM=0.

Packet state enum (for SOM/EOM flag derivation):
```
MCTP_PACKET_STATE_START        = 0   // SOM=1, EOM=0
MCTP_PACKET_STATE_INTERMEDIATE = 1   // SOM=0, EOM=0
MCTP_PACKET_STATE_END          = 2   // SOM=0, EOM=1
MCTP_PACKET_STATE_SINGLE_PACKET= 3   // SOM=1, EOM=1
```

### 2.2 COT boot payload — `NVDM_PAYLOAD_COT` (792 bytes)
```
u16 version;                 //   2  = 1 for GH100
u16 size;                    //   2  = sizeof(struct)
u64 gspFmcSysmemOffset;      //   8  BAR4-visible PA of FMC image
u64 frtsSysmemOffset;        //   8  FRTS (Faulting-Related-Tasks Subsystem) sysmem
u32 frtsSysmemSize;          //   4
u64 frtsVidmemOffset;        //   8  offset FROM END of FB (negative from top)
u32 frtsVidmemSize;          //   4
u32 hash384[12];             //  48  SHA-384 hash of FMC
u32 publicKey[96];           // 384  RSA-3072 public key
u32 signature[96];           // 384  RSA-3072 signature over FMC
u64 gspBootArgsSysmemOffset; //   8  BAR4 PA of GSP_FMC_BOOT_PARAMS
// total: 860 bytes  (2+2+8+8+4+8+4+48+384+384+8)
```
The FMC hash, public key, and signature are embedded in the signed FMC
blob (`gsp_fmc.bin`) at well-known offsets. We copy them — we do not compute
them. Step 5 (`fw_load.s`) already locates the FMC image; it must additionally
expose these byte ranges to Step 7.

### 2.3 GSP_FMC_BOOT_PARAMS (~128 bytes — pointed to by `gspBootArgsSysmemOffset`)
Referenced but not defined in the MCTP header. Source:
`src/common/sdk/nvidia/inc/dev_gsp.h::GSP_FMC_BOOT_PARAMS`. Fields:
```
GSP_ACR_BOOT_GSP_RM_PARAMS bootGspRmParams;  // WPR1 range, GSP img offset/size
GSP_RM_PARAMS              gspRmParams;       // GSP RM args ptr (BAR4 PA)
```
We already allocate a BAR4 page for this in Step 5; Step 7 fills it.

### 2.4 Response packet
Response fits in a single 256-byte EMEM packet: MCTP+NVDM headers followed
by `NVDM_PAYLOAD_COT_RESPONSE` — a 4-byte status code. Non-zero = FSP rejected
the COT; abort with diagnostic via scratch group 2/3.

---

## 3. Protocol Sequence

All steps run from ARM64 userspace via BAR0 mmap (already established by
Step 1 `bar_map.s`). No kernel involvement.

### 3.1 Prerequisites (verify before first EMEMC write)
1. Read `NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2(0)` — wait for FSP-ready
   sentinel (value defined in kern_fsp_gh100.c, typically a specific magic).
2. Read `QUEUE_HEAD(0) == QUEUE_TAIL(0)` — queue empty.
3. Read `MSGQ_HEAD(0) == MSGQ_TAIL(0)` — response queue empty.

### 3.2 Build COT payload in a scratch buffer (HBM or CPU stack)
4. Populate `NVDM_PAYLOAD_COT` in a 1 KB scratch (860-byte payload + 32-byte
   header overhead rounded up). Fields drawn from:
   - `gspFmcSysmemOffset`:     from Step 5 (fw_load.s) output
   - `frtsSysmemOffset/Size`:  from Step 4 (hbm_alloc.s) output — FRTS BAR4 region
   - `frtsVidmemOffset/Size`:  computed as (FB_end - FRTS_size), 1 MB aligned
   - `hash384/publicKey/signature`: sliced from signed FMC blob
   - `gspBootArgsSysmemOffset`: BAR4 PA of the boot-params page

### 3.3 Packetize + write to CPU->FSP command queue
5. Chunk the 860-byte payload into ceil((860+hdrs)/248) = 4 packets of
   256 bytes each (256 - 8-byte headers = 248 payload bytes per packet, first
   packet loses an additional 4 bytes for NVDM header).
6. For each packet, pick a free EMEM block (ch0 cmd queue partition). Program
   `EMEMC(0) = (BLK<<8) | (OFFS<<2) | AINCW=1`.
7. Stream 64 DW to `EMEMD(0)` — MCTP header DW, then NVDM-type DW (first
   packet only), then payload DW, zero-padded to 64 DW.
8. After all packets written, advance `QUEUE_HEAD(0) += 256 * packet_count`
   (modulo queue size) to signal FSP.

### 3.4 Wait for response
9. Poll `MSGQ_HEAD(0) != MSGQ_TAIL(0)` with 5 s timeout (FSP's worst-case
   COT verify is <1 s but GH200 can stall on PLL lock).
10. Read response block via `EMEMC(0)` with AINCR=1, streaming 64 DW from
    `EMEMD(0)`.
11. Validate MCTP header (single-packet, matching NVDM type, non-null src EID).
12. Extract status DW. Nonzero -> fatal abort path (print scratch group 2/3
    error code via the Forth diagnostic channel).

### 3.5 Lockdown release handshake
13. Advance `MSGQ_TAIL(0) = MSGQ_HEAD(0)` to acknowledge consumption.
14. Now drop to Step 8 (`poll_lockdown.s`) — polls BAR0+0x1100F4 bit 13
    clear. FSP has released the target mask internally.

### 3.6 Error taxonomy
- FSP not ready (scratch group 2 != ready magic) -> bail before writing EMEM.
- EMEM write busy (EMEMC AINCW latched but BUSY bit set) -> retry with MMIO read-back barrier.
- Response timeout -> dump scratch group 2/3, return -ETIMEDOUT.
- Response status nonzero -> map FSP error code (see kern_fsp_retval.h) to a human tag.

---

## 4. File Breakdown (ARM64 asm)

Proposed layout under `/home/ubuntu/lithos/GSP/fsp/`:

### `fsp/emem_xfer.s` — EMEM DMA primitives (~350 LOC)
Low-level window-paging over EMEMC/EMEMD. No MCTP, no COT. Pure
word-stream engine. Exports:
- `fsp_emem_write_block(ch, blk_offset_bytes, src_ptr, n_dwords)`
- `fsp_emem_read_block(ch, blk_offset_bytes, dst_ptr, n_dwords)`
- `fsp_queue_advance_head(ch, n_bytes)` / `fsp_msgq_advance_tail(ch, n_bytes)`
- `fsp_queue_is_empty(ch)` / `fsp_msgq_has_data(ch)`
Plus the register-offset constants (`.equ NV_PFSP_EMEMC_BASE, 0x8F2AC0` etc.).

### `fsp/mctp.s` — MCTP + NVDM packetizer/depacketizer (~400 LOC)
Translates a byte buffer into 256-byte framed packets with correct SOM/EOM
flags. Exports:
- `fsp_mctp_send(ch, nvdm_type, buf, len)` -> packetizes, calls emem_xfer,
  advances queue head.
- `fsp_mctp_recv(ch, nvdm_type_expected, buf, max_len)` -> depacketizes,
  returns payload length or error.
- constBlob bitfield packing helpers.

### `fsp/cot_payload.s` — COT struct assembly (~350 LOC)
Populates `NVDM_PAYLOAD_COT` from the outputs of earlier steps. Also locates
hash/key/signature inside the FMC blob (constant offsets for GH100 FMC
layout — document in comments). Exports:
- `fsp_build_cot_payload(dst_buf, fmc_phys, frts_sys, frts_vid, boot_args_phys)`

### `fsp/bootcmd.s` — Top-level orchestration (~300 LOC)
The only externally-callable entry point (`fsp_send_boot_commands`). Does:
1. Precondition polls (scratch group 2, queue empty).
2. Allocates 1 KB scratch (BAR4 bump-alloc or stack).
3. Calls `fsp_build_cot_payload`.
4. Calls `fsp_mctp_send` with NVDM type = COT.
5. Polls + `fsp_mctp_recv`.
6. Validates response status, ack via MSGQ_TAIL.
7. Returns to caller (drops into Step 8).

### `fsp/fsp_diag.s` — Error reporting / scratch decode (~200 LOC)
Not strictly required for the happy path, but essential for bringup. Decodes
`NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2/3` values and kern_fsp_retval.h status
codes into printable tags via the existing diag UART/serial path that
`rpc_channel.s` already uses. Keep this file ruthlessly trimmed if LOC budget
tightens — drop to 100 LOC of raw hex-dump if needed.

### Total estimate: ~1,600 LOC
Leaves ~400 LOC headroom vs. the 2,000-LOC C budget for comments,
register constants, and alignment padding macros.

---

## 5. Dependencies

READY — Step 7 can start immediately. Prerequisites (all complete):
- Step 1 `bar_map.s`           — BAR0/BAR4 mmap handles
- Step 2 `pmc_check.s`         — PMC enable confirmed
- Step 3 `falcon_reset.s`      — GSP Falcon reset done
- Step 4 `hbm_alloc.s`         — FRTS, WPR, boot-params pages carved from BAR4
- Step 5 `fw_load.s`           — FMC loaded; must additionally expose:
  - `fmc_phys` (BAR4 PA), `fmc_hash_offset`, `fmc_pubkey_offset`, `fmc_sig_offset`
- Step 6 `bcr_start.s`         — BCR set up (FSP runs before CPUCTL.STARTCPU)

Consumers of Step 7 output:
- Step 8 `poll_lockdown.s`     — can already poll; just needs FSP to have released target mask
- Step 9 `rpc_channel.s`       — unaffected

One small addition to Step 5 is needed: exporting the three byte offsets
into the FMC blob where hash/pubkey/sig live. That is a ~10 LOC change to
`fw_load.s` and is not blocking — it can ship in the same PR as Step 7.

---

## 6. Testing Approach (no-brick path)

The FSP cannot physically brick a GH200 via rejected COT — a failed COT
just leaves PRIV_LOCKDOWN asserted and the GPU inert. Power cycle restores
full state. But wrong EMEM writes into scratch regions *can* arm watchdog
traps, so we gate rollout:

### Tier 0 — software-only validation (~1 hour)
- Unit-test `fsp/mctp.s` against synthesized packets. Feed known-good
  byte streams captured from `nvidia.ko` via `rm_log` tracepoint and
  diff packet framing.
- Unit-test `fsp/cot_payload.s` by comparing its output buffer (for fixed
  inputs) byte-for-byte with what kern_fsp_gh100.c produces under a C
  unit-test harness.

### Tier 1 — dry-run on a sacrificial GH200 (~1 day)
- Boot the machine normally via nvidia.ko first; confirm known-good state.
- `rmmod nvidia`; run lithos launcher with FSP step gated by an env flag
  that skips actual `QUEUE_HEAD` advance — just writes EMEM and reads it
  back via AINCR. Confirms EMEM pathing without involving FSP.
- Check scratch group 2/3 stayed at known-good values.

### Tier 2 — real COT exchange
- Enable QUEUE_HEAD advance; send real COT.
- Poll MSGQ_HEAD with 5 s timeout.
- On success, do NOT yet fall through to Step 8. Abort and reboot. Confirm
  scratch group 2 shows "COT accepted" magic.

### Tier 3 — full end-to-end
- Chain into Step 8 poll_lockdown. On success, FMC runs; GSP boots; Step 9
  RPC proves liveness. This is the first cooperative-megakernel load.

### Instrumentation
- Every EMEM write logged via ring buffer in BAR4 visible to host.
- Scratch group 2/3 sampled every ms for 10 s after QUEUE_HEAD advance,
  streamed via `rpc_channel`'s already-working diag channel.
- If anything fails at Tier 2/3: single-button recovery is `ipmitool power
  cycle` + `nvidia-smi` from another node to confirm GPU re-enumerates.

### What we do NOT test
- Multiple COT rounds. One FMC, one COT, done. If we need re-auth (e.g.
  after GPU reset) we just rerun the full Step-1-through-7 sequence.
- Async NVDM callbacks (kern_fsp.c's `kfspPollForAsyncResponse` path). Lithos
  is single-threaded megakernel; only synchronous COT at boot. Saves ~300 LOC
  vs. reference.

---

## 7. Out-of-Scope (intentional)

Drop from the port to stay under budget:
- `kfspPollForAsyncResponse` / TMR event callbacks — synchronous only.
- Registry override reading (`kfspInitRegistryOverrides`) — compile-time constants.
- Multiple NVDM types (RM command channel, LS_UCODE, etc.) — we only need COT.
- FSP SR-IOV / multi-VF paths — single-tenant bare metal only.
- Blackwell HALs (GB100/GB202) — GH100 only for now; parallel file later.

Keeping these out of scope is what compresses 2,713 LOC of reference C into
~1,600 LOC of focused ARM64 asm.
