// GSP/fsp/registers.s -- FSP (Firmware Security Processor) constants.
//
// Header-like translation unit.  Contributes NO code and NO data symbols.
// Pure `.equ` definitions + comments.  Intended to be `.include`d (or at
// minimum read-alongside) by the other fsp/*.s files:
//
//     fsp/emem_xfer.s   -- EMEM window-paging primitives
//     fsp/mctp.s        -- MCTP/NVDM packetize / depacketize
//     fsp/cot_payload.s -- NVDM_PAYLOAD_COT struct assembly
//     fsp/bootcmd.s     -- top-level orchestration (fsp_send_boot_commands)
//     fsp/fsp_diag.s    -- scratch-group decode / error reporting
//
// The file must assemble cleanly as an empty object:
//     as -o /dev/null GSP/fsp/registers.s
//
// -----------------------------------------------------------------------------
// SOURCES OF TRUTH
//
//   Register offsets / bitfields:
//     /tmp/ogkm/src/common/inc/swref/published/hopper/gh100/dev_fsp_pri.h
//
//   COT payload struct (NVDM_PAYLOAD_COT, MCTP_HEADER):
//     /tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h
//
//   MCTP header bitfields, msg-type / vendor-id constants:
//     /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/fsp/fsp_mctp_format.h
//
//   NVDM message-type enum:
//     /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/fsp/fsp_nvdm_format.h
//
//   EMEM channel allocation (RM = channel 0, size = 1 KB):
//     /tmp/ogkm/src/nvidia/arch/nvalloc/common/inc/fsp/fsp_emem_channels.h
//
//   MCTP packet-state enum (SOM/EOM derivation):
//     /tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_mctp_pkt_state.h
//
//   FSP response status codes:
//     /tmp/ogkm/src/nvidia/inc/kernel/gpu/fsp/kern_fsp_retval.h
//
//   Overall plan (register layout, protocol sequence):
//     /home/ubuntu/lithos/GSP/fsp_plan.md Section 1+2
//
// All register accesses are 32-bit MMIO relative to BAR0.  Use:
//     str w<val>, [x_bar0, #(offset)]      // writes
//     ldr w<val>, [x_bar0, #(offset)]      // reads
// Where offset > 12-bit immediate range (most FSP regs are 0x8F0000-0x8F3000
// i.e. > 4 KB), materialize the offset into a register first.

// =============================================================================
// 1.  FSP register base
// =============================================================================
//
// FSP Falcon PRI block starts at BAR0 + 0x008F2000.  The scratch-group
// registers live slightly below at 0x8F0000-0x8F0400, but the "FSP base"
// per the RM driver is the PRI block origin.  We expose BOTH for clarity:
//
//   - FSP_BASE             = 0x8F2000    (PRI block, home of EMEM/QUEUE/MSGQ)
//   - FSP_FALCON_BASE      = 0x8F0000    (Falcon common register block that
//                                         hosts SCRATCH_GROUP_2 / _3)
//
// The task description mandates NV_PFSP_BASE = 0x8F2000, matching fsp_plan.md
// Section 1.

.equ NV_PFSP_BASE,                  0x008F2000
.equ NV_PFSP_FALCON_BASE,           0x008F0000

// =============================================================================
// 2.  EMEMC / EMEMD  (per-channel windowed access to FSP internal 4 KB SRAM)
// =============================================================================
//
// From dev_fsp_pri.h:
//     #define NV_PFSP_EMEMC(i) (0x008F2AC0 + (i)*8)
//     #define NV_PFSP_EMEMD(i) (0x008F2AC4 + (i)*8)
//     #define NV_PFSP_EMEMC__SIZE_1 8     // 8 channels total (0..7)
//
// EMEMC programs the (BLK, OFFS) cursor and AINCW/AINCR auto-advance bits;
// EMEMD reads or writes one 32-bit DW at the programmed cursor, auto-
// advancing the cursor if AINCW/R is set.
//
// Channel 0 is the RM/boot channel (FSP_EMEM_CHANNEL_RM).  Channel 0 owns
// 1 KB of EMEM -- 512 B for the CPU->FSP command queue and 512 B for the
// FSP->CPU response queue (see FSP_EMEM_CH0_* below).

.equ NV_PFSP_EMEMC_BASE,            0x008F2AC0  // EMEMC(0)
.equ NV_PFSP_EMEMD_BASE,            0x008F2AC4  // EMEMD(0)
.equ NV_PFSP_EMEM_STRIDE,           8           // EMEMC(i) = base + i*8
.equ NV_PFSP_EMEM_NUM_CHANNELS,     8           // __SIZE_1

// Shortcuts for the one channel we use (ch0).  Per-channel call sites
// that hard-code ch0 can load these directly instead of computing
// base + stride*i at runtime.
.equ NV_PFSP_EMEMC_CH0,             0x008F2AC0  // EMEMC(0)
.equ NV_PFSP_EMEMD_CH0,             0x008F2AC4  // EMEMD(0)

// -----------------------------------------------------------------------------
// 2a.  EMEMC bitfield definitions
// -----------------------------------------------------------------------------
//
// From dev_fsp_pri.h:
//     NV_PFSP_EMEMC_OFFS   7:2   -- dword offset within a 256-byte block
//     NV_PFSP_EMEMC_BLK   15:8   -- which 256-byte block (EMEM has 16 blocks
//                                    of 256 B for 4 KB total)
//     NV_PFSP_EMEMC_AINCW 24:24  -- auto-increment on write
//     NV_PFSP_EMEMC_AINCR 25:25  -- auto-increment on read
//
// A typical write-setup value is:
//     EMEMC = (BLK << 8) | (OFFS << 2) | AINCW
// and for read-setup:
//     EMEMC = (BLK << 8) | (OFFS << 2) | AINCR

.equ NV_PFSP_EMEMC_OFFS_SHIFT,      2            // bits 7:2
.equ NV_PFSP_EMEMC_OFFS_WIDTH,      6
.equ NV_PFSP_EMEMC_OFFS_MASK,       0x000000FC   // (0x3F << 2)

.equ NV_PFSP_EMEMC_BLK_SHIFT,       8            // bits 15:8
.equ NV_PFSP_EMEMC_BLK_WIDTH,       8
.equ NV_PFSP_EMEMC_BLK_MASK,        0x0000FF00   // (0xFF << 8)

.equ NV_PFSP_EMEMC_AINCW_SHIFT,     24           // bit 24
.equ NV_PFSP_EMEMC_AINCW,           0x01000000   // (1 << 24), "TRUE"
.equ NV_PFSP_EMEMC_AINCW_FALSE,     0x00000000

.equ NV_PFSP_EMEMC_AINCR_SHIFT,     25           // bit 25
.equ NV_PFSP_EMEMC_AINCR,           0x02000000   // (1 << 25), "TRUE"
.equ NV_PFSP_EMEMC_AINCR_FALSE,     0x00000000

// Convenience: both auto-increment bits set (occasionally useful if a
// caller wants to stream a packet in and then reply in place).
.equ NV_PFSP_EMEMC_AINC_BOTH,       0x03000000

// EMEM geometry constants derivable from the bitfields above.
.equ FSP_EMEM_BLOCK_SIZE,           256          // 64 DW per block
.equ FSP_EMEM_DWORDS_PER_BLOCK,     64
.equ FSP_EMEM_TOTAL_BYTES,          4096         // 16 blocks * 256 B

// =============================================================================
// 3.  QUEUE_HEAD / QUEUE_TAIL  (CPU -> FSP command queue)
// =============================================================================
//
// From dev_fsp_pri.h:
//     NV_PFSP_QUEUE_HEAD(i) = 0x008F2C00 + i*8
//     NV_PFSP_QUEUE_TAIL(i) = 0x008F2C04 + i*8
//
// CPU writes HEAD to push; FSP advances TAIL as it consumes.  Values are
// byte offsets into the channel's command-queue partition of EMEM.

.equ NV_PFSP_QUEUE_HEAD_BASE,       0x008F2C00  // QUEUE_HEAD(0)
.equ NV_PFSP_QUEUE_TAIL_BASE,       0x008F2C04  // QUEUE_TAIL(0)
.equ NV_PFSP_QUEUE_STRIDE,          8

.equ NV_PFSP_QUEUE_HEAD_CH0,        0x008F2C00
.equ NV_PFSP_QUEUE_TAIL_CH0,        0x008F2C04

// =============================================================================
// 4.  MSGQ_HEAD / MSGQ_TAIL  (FSP -> CPU response queue)
// =============================================================================
//
// From dev_fsp_pri.h:
//     NV_PFSP_MSGQ_HEAD(i) = 0x008F2C80 + i*8
//     NV_PFSP_MSGQ_TAIL(i) = 0x008F2C84 + i*8
//
// FSP advances HEAD to publish a response.  CPU advances TAIL after
// consuming.  Queue-empty iff HEAD == TAIL.

.equ NV_PFSP_MSGQ_HEAD_BASE,        0x008F2C80  // MSGQ_HEAD(0)
.equ NV_PFSP_MSGQ_TAIL_BASE,        0x008F2C84  // MSGQ_TAIL(0)
.equ NV_PFSP_MSGQ_STRIDE,           8

.equ NV_PFSP_MSGQ_HEAD_CH0,         0x008F2C80
.equ NV_PFSP_MSGQ_TAIL_CH0,         0x008F2C84

// =============================================================================
// 5.  Falcon Common Scratch Groups  (boot status / error sentinels)
// =============================================================================
//
// From dev_fsp_pri.h:
//     NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2(i) = 0x008F0320 + i*4   (4 words)
//     NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_3(i) = 0x008F0330 + i*4
//
// Group 2 index 0 is the primary FSP-boot-status sentinel the CPU polls
// before the first EMEM write (Section 3.1 step 1 of fsp_plan.md).  Group 3
// carries secondary status / error codes exposed by FSP firmware on abort.
//
// NOTE: scratch groups use a 4-byte stride (not 8) because each is a plain
// 32-bit word, not a head/tail pair.

.equ SCRATCH_GROUP_2,               0x008F0320  // alias per task description
.equ SCRATCH_GROUP_3,               0x008F0330  // alias per task description

.equ NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_2_BASE, 0x008F0320
.equ NV_PFSP_FALCON_COMMON_SCRATCH_GROUP_3_BASE, 0x008F0330
.equ NV_PFSP_FALCON_SCRATCH_STRIDE, 4
.equ NV_PFSP_FALCON_SCRATCH_COUNT,  4           // __SIZE_1 = 4 each

// =============================================================================
// 6.  AINCW / AINCR bit definitions in EMEMC (auto-advance-on-write/read)
// =============================================================================
//
// Already defined above in Section 2a for grouping.  Aliases named to match
// the task description wording for locatability:

.equ FSP_EMEMC_AINCW_BIT,           24
.equ FSP_EMEMC_AINCR_BIT,           25
.equ FSP_EMEMC_AINCW_MASK,          0x01000000  // same as NV_PFSP_EMEMC_AINCW
.equ FSP_EMEMC_AINCR_MASK,          0x02000000  // same as NV_PFSP_EMEMC_AINCR

// =============================================================================
// 7.  MCTP / NVDM header formatting
// =============================================================================
//
// From fsp_mctp_format.h (bitfields in the 32-bit "constBlob" word):
//     MCTP_HEADER_VERSION   3:0
//     MCTP_HEADER_RSVD      7:4
//     MCTP_HEADER_DEID     15:8      -- destination endpoint ID
//     MCTP_HEADER_SEID     23:16     -- source endpoint ID
//     MCTP_HEADER_TAG      26:24
//     MCTP_HEADER_TO       27:27     -- tag owner
//     MCTP_HEADER_SEQ      29:28     -- sequence number mod 4
//     MCTP_HEADER_EOM      30:30     -- end of message
//     MCTP_HEADER_SOM      31:31     -- start of message
//
// The MCTP message-type / vendor-id word that follows the transport header
// packs:
//     MCTP_MSG_HEADER_TYPE        6:0    = 0x7E (vendor-defined PCI)
//     MCTP_MSG_HEADER_IC          7:7    = 0 (no integrity check)
//     MCTP_MSG_HEADER_VENDOR_ID  23:8    = 0x10DE (NVIDIA)
//     MCTP_MSG_HEADER_NVDM_TYPE  31:24   = NVDM_TYPE_* (e.g. 0x14 for COT)

// ---- MCTP transport header (first 32 bits of every packet) ----

.equ MCTP_HEADER_VERSION_SHIFT,     0
.equ MCTP_HEADER_VERSION_MASK,      0x0000000F
.equ MCTP_HEADER_VERSION_VAL,       0x1         // protocol version 1

.equ MCTP_HEADER_DEID_SHIFT,        8
.equ MCTP_HEADER_DEID_MASK,         0x0000FF00

.equ MCTP_HEADER_SEID_SHIFT,        16
.equ MCTP_HEADER_SEID_MASK,         0x00FF0000

.equ MCTP_HEADER_TAG_SHIFT,         24
.equ MCTP_HEADER_TAG_MASK,          0x07000000

.equ MCTP_HEADER_TO_SHIFT,          27
.equ MCTP_HEADER_TO_MASK,           0x08000000

.equ MCTP_HEADER_SEQ_SHIFT,         28
.equ MCTP_HEADER_SEQ_MASK,          0x30000000

.equ MCTP_HEADER_EOM_SHIFT,         30
.equ MCTP_HEADER_EOM_MASK,          0x40000000
.equ MCTP_HEADER_EOM_BIT,           0x40000000

.equ MCTP_HEADER_SOM_SHIFT,         31
.equ MCTP_HEADER_SOM_MASK,          0x80000000
.equ MCTP_HEADER_SOM_BIT,           0x80000000

// ---- MCTP message header (2nd DW on SOM packets; 1st DW of payload) ----

.equ MCTP_MSG_HEADER_TYPE_SHIFT,        0
.equ MCTP_MSG_HEADER_TYPE_MASK,         0x0000007F
.equ MCTP_MSG_HEADER_TYPE_VENDOR_PCI,   0x7E

.equ MCTP_MSG_HEADER_IC_SHIFT,          7
.equ MCTP_MSG_HEADER_IC_MASK,           0x00000080

.equ MCTP_MSG_HEADER_VENDOR_ID_SHIFT,   8
.equ MCTP_MSG_HEADER_VENDOR_ID_MASK,    0x00FFFF00
.equ MCTP_MSG_HEADER_VENDOR_ID_NV,      0x10DE

.equ MCTP_MSG_HEADER_NVDM_TYPE_SHIFT,   24
.equ MCTP_MSG_HEADER_NVDM_TYPE_MASK,    0xFF000000

// ---- Packet-state enum (from kern_fsp_mctp_pkt_state.h) ----
// Determines SOM/EOM bits in the transport header.

.equ MCTP_PACKET_STATE_START,           0   // SOM=1, EOM=0  (first of many)
.equ MCTP_PACKET_STATE_INTERMEDIATE,    1   // SOM=0, EOM=0
.equ MCTP_PACKET_STATE_END,             2   // SOM=0, EOM=1  (last of many)
.equ MCTP_PACKET_STATE_SINGLE_PACKET,   3   // SOM=1, EOM=1  (fits in one pkt)

// ---- Header DWORD indices within a packet (from kern_fsp.c) ----
//     #define HEADER_DWORD_MCTP 0
//     #define HEADER_DWORD_NVDM 1
//     #define HEADER_DWORD_MAX  2
// i.e. DW 0 is the MCTP transport header; DW 1 (on SOM packets only) is
// the MCTP-message / NVDM-type header.

.equ FSP_HEADER_DWORD_MCTP,         0
.equ FSP_HEADER_DWORD_NVDM,         1
.equ FSP_HEADER_DWORD_MAX,          2

// =============================================================================
// 8.  COT payload structure sizes
// =============================================================================
//
// NVDM_PAYLOAD_COT (from kern_fsp_cot_payload.h, #pragma pack(1)):
//     u16 version                         //   2
//     u16 size                            //   2
//     u64 gspFmcSysmemOffset              //   8
//     u64 frtsSysmemOffset                //   8
//     u32 frtsSysmemSize                  //   4
//     u64 frtsVidmemOffset                //   8   (offset from end of FB)
//     u32 frtsVidmemSize                  //   4
//     u32 hash384[12]                     //  48   SHA-384 of FMC
//     u32 publicKey[96]                   // 384   RSA-3072 pubkey
//     u32 signature[96]                   // 384   RSA-3072 signature
//     u64 gspBootArgsSysmemOffset         //   8
// total = 2+2+8+8+4+8+4+48+384+384+8 = 860 bytes.
//
// MCTP_HEADER (transport-header struct, also pack(1)):
//     u32 constBlob                       //   4
//     u8  msgType                         //   1
//     u16 vendorId                        //   2
// total = 7 bytes (but the wire layout aligns constBlob to a full DW and
// packs msgType+vendorId into the low 3 bytes of the next DW alongside
// the NVDM_TYPE byte, so on-wire "MCTP + NVDM" is effectively 8 bytes /
// 2 DW, matching HEADER_DWORD_MAX = 2).

.equ NVDM_PAYLOAD_COT_SIZE,         860
.equ MCTP_HEADER_SIZE,              7       // sizeof(struct MCTP_HEADER)
.equ NVDM_HEADER_SIZE,              4       // 1 NVDM type DW

// Derived byte offsets within NVDM_PAYLOAD_COT.  Useful for cot_payload.s
// when populating fields.  All offsets assume #pragma pack(1).
.equ COT_OFF_VERSION,               0       //  u16
.equ COT_OFF_SIZE,                  2       //  u16
.equ COT_OFF_GSP_FMC_SYSMEM,        4       //  u64
.equ COT_OFF_FRTS_SYSMEM_OFF,      12       //  u64
.equ COT_OFF_FRTS_SYSMEM_SIZE,     20       //  u32
.equ COT_OFF_FRTS_VIDMEM_OFF,      24       //  u64
.equ COT_OFF_FRTS_VIDMEM_SIZE,     32       //  u32
.equ COT_OFF_HASH384,              36       //  u32[12] = 48 bytes
.equ COT_OFF_PUBLIC_KEY,           84       //  u32[96] = 384 bytes
.equ COT_OFF_SIGNATURE,           468       //  u32[96] = 384 bytes
.equ COT_OFF_GSP_BOOT_ARGS,       852       //  u64  (852+8 = 860 = total)

// Values written into the payload.version / payload.size fields.  GH100
// uses version 1; size matches sizeof(NVDM_PAYLOAD_COT) = 860.
.equ COT_VERSION_GH100,             1
.equ COT_PAYLOAD_SIZE_GH100,        NVDM_PAYLOAD_COT_SIZE

// =============================================================================
// 9.  NVDM type codes  (from fsp_nvdm_format.h)
// =============================================================================
//
// The only one Lithos needs is NVDM_TYPE_COT (0x14) and the response type
// NVDM_TYPE_FSP_RESPONSE (0x15).  Others listed for completeness / future use.

.equ NVDM_TYPE_RESET,               0x04
.equ NVDM_TYPE_HULK,                0x11
.equ NVDM_TYPE_FIRMWARE_UPDATE,     0x12
.equ NVDM_TYPE_PRC,                 0x13
.equ NVDM_TYPE_COT,                 0x14    // <<< Lithos boot path uses this
.equ NVDM_TYPE_FSP_RESPONSE,        0x15    // <<< Lithos reads this back
.equ NVDM_TYPE_CAPS_QUERY,          0x16
.equ NVDM_TYPE_INFOROM,             0x17
.equ NVDM_TYPE_SMBPBI,              0x18
.equ NVDM_TYPE_ROMREAD,             0x1A
.equ NVDM_TYPE_UEFI_RM,             0x1C
.equ NVDM_TYPE_UEFI_XTL_DEBUG_INTR, 0x1D
.equ NVDM_TYPE_TNVL,                0x1F
.equ NVDM_TYPE_CLOCK_BOOST,         0x20
.equ NVDM_TYPE_FSP_GSP_COMM,        0x21

// -----------------------------------------------------------------------------
// FSP response status codes (from kern_fsp_retval.h)
// -----------------------------------------------------------------------------
//
// The NVDM_TYPE_FSP_RESPONSE payload begins with a 32-bit status.  Zero means
// success; any nonzero value means the command was rejected.  The specific
// codes below come directly from kern_fsp_retval.h.

.equ FSP_OK,                            0x00
.equ FSP_ERR_IFS_ERR_INVALID_STATE,     0x9E
.equ FSP_ERR_IFR_FILE_NOT_FOUND,        0x9F
.equ FSP_ERR_IFS_ERR_NOT_SUPPORTED,     0xA0
.equ FSP_ERR_IFS_ERR_INVALID_DATA,      0xA1
.equ FSP_ERR_PRC_ERROR_INVALID_KNOB_ID, 0x1E3

// =============================================================================
// 10.  Channel indices and per-channel EMEM geometry
// =============================================================================
//
// GH100 exposes 8 EMEM channels but RM (and Lithos) only uses channel 0.
// fsp_emem_channels.h documents:
//     FSP_EMEM_CHANNEL_RM      = 0
//     FSP_EMEM_CHANNEL_RM_SIZE = 1024        // 1 KB partition for ch0
//
// Within ch0's 1 KB, the reference driver splits:
//     [   0 ..  512)   CPU -> FSP command queue  (cmdQPhysicalOffset = 0)
//     [ 512 .. 1024)   FSP -> CPU response queue (msgQPhysicalOffset = 512)
//
// The fsp_plan.md text uses three symbolic names for the role-specific
// queues.  We define all three against channel 0 so call sites can read
// intentionally (e.g. "advance SUBMISSION_CHANNEL head" rather than
// "advance channel 0 queue head").
//
// NB: in Lithos, SUBMISSION_CHANNEL and COMMAND_CHANNEL are aliases for the
// same CPU->FSP queue on channel 0; they differ only in intent (SUBMISSION
// = the act of pushing a packet; COMMAND = the overall command pathway).
// RESPONSE_CHANNEL is the FSP->CPU direction on the same channel.

.equ FSP_EMEM_CHANNEL_RM,           0
.equ FSP_EMEM_CHANNEL_RM_SIZE,      1024

.equ FSP_COMMAND_CHANNEL,           0       // CPU -> FSP (command path)
.equ FSP_SUBMISSION_CHANNEL,        0       // alias: push direction
.equ FSP_RESPONSE_CHANNEL,          0       // FSP -> CPU (response path)

// Per-channel EMEM partition offsets (in EMEM bytes, not BAR0 bytes).
// These match the reference driver's cmdQPhysicalOffset / msgQPhysicalOffset
// for channel 0 and are used by emem_xfer.s when it translates a queue-
// relative byte offset into a (BLK, OFFS) pair.

.equ FSP_EMEM_CH0_CMDQ_OFFSET,      0       // bytes into EMEM
.equ FSP_EMEM_CH0_CMDQ_SIZE,        512
.equ FSP_EMEM_CH0_MSGQ_OFFSET,      512
.equ FSP_EMEM_CH0_MSGQ_SIZE,        512

// Max packet sizes.  FSP uses 256-byte EMEM packets; per-packet payload
// capacity is 256 minus the header DWs on SOM packets.
.equ FSP_PACKET_SIZE_BYTES,         256
.equ FSP_PACKET_PAYLOAD_SOM_BYTES,  248     // 256 - 8 (MCTP DW + NVDM DW)
.equ FSP_PACKET_PAYLOAD_CONT_BYTES, 252     // 256 - 4 (MCTP DW only)

// =============================================================================
// End of registers.s.  Assemble check:
//     as -o /dev/null /home/ubuntu/lithos/GSP/fsp/registers.s
// Should produce an empty object and exit 0.
// =============================================================================
