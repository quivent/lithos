\\ pushbuffer.ls — GPFIFO pushbuffer builder
\\
\\ The pushbuffer is a ring of 32-bit methods consumed by the host-side channel
\\ front-end (HOST1X/CE/Compute engine). Each entry is a method header + payload.
\\ To launch a kernel we use an inline DMA sequence that copies 528 bytes of QMD
\\ body into the engine's method stream, then issues SEND_PCAS_A to hand the QMD
\\ off to the Compute scheduler.
\\
\\ Method header encoding (one 32-bit word):
\\   bit 31       sc   (single-control)
\\   bits 30..29  opcode: 0=INC, 2=NONINC, 3=IMMD
\\   bits 28..16  count
\\   bits 15..13  subchannel
\\   bits 12..0   method address in dwords (method_byte >> 2)

\\ Write cursor is stored in the first 8 bytes of pb_ptr; actual stream starts at +8.
pb_init pb_ptr :
    ← 64 pb_ptr pb_ptr + 8

\\ Build a method header with opcode=INC (0) and emit it at the write cursor.
pb_emit_header_inc pb_ptr method count subchannel :
    addr method / 4
    hdr addr + subchannel * 0x2000
    hdr hdr + count * 0x10000
    hdr hdr + 0x80000000
    cur → 64 pb_ptr
    ← 32 cur hdr
    ← 64 pb_ptr cur + 4

\\ Same but with opcode=NONINC (2) for bulk data blasts.
pb_emit_header_noninc pb_ptr method count subchannel :
    addr method / 4
    hdr addr + subchannel * 0x2000
    hdr hdr + count * 0x10000
    hdr hdr + 0xc0000000
    cur → 64 pb_ptr
    ← 32 cur hdr
    ← 64 pb_ptr cur + 4

\\ Append one raw 32-bit dword at the cursor.
pb_emit_word pb_ptr w :
    cur → 64 pb_ptr
    ← 32 cur w
    ← 64 pb_ptr cur + 4

\\ Generic header emit — dispatches by opcode (0=INC, 2=NONINC, 3=IMMD).
pb_emit_header pb_ptr method count sub subchannel :
    if== sub 0
        pb_emit_header_inc pb_ptr method count subchannel
    if== sub 2
        pb_emit_header_noninc pb_ptr method count subchannel

\\ ========================================================================
\\ CB load helper — emit one inline constant buffer load via pushbuffer.
\\
\\ The GPU compute engine loads constant buffers through a 4-method sequence:
\\   1. CB_TARGET_ADDR (0x2062)  — set target GPU VA (hi32, lo32)
\\   2. CB_SIZE_AND_FLAGS (0x2060) — set size + valid flag
\\   3. CB_BIND (0x206c)         — bind the constant buffer
\\   4. CB_LOAD_INLINE_DATA (0x206d) — bulk data (NONINC method)
\\
\\ target_gpu_va: GPU VA where this CB will be accessible
\\ data_ptr:      host-visible pointer to the data to inline
\\ nbytes:        size in bytes (must be dword-aligned)
\\ ========================================================================
pb_emit_cb_load pb_ptr target_gpu_va data_ptr nbytes :
    \\ CB_TARGET_ADDR: hi32, lo32
    pb_emit_header_inc pb_ptr 0x2062 2 1
    hi target_gpu_va / 0x100000000
    lo target_gpu_va - hi * 0x100000000
    pb_emit_word pb_ptr hi
    pb_emit_word pb_ptr lo

    \\ CB_SIZE_AND_FLAGS: size in bytes, flags=1 (valid)
    pb_emit_header_inc pb_ptr 0x2060 2 1
    pb_emit_word pb_ptr nbytes
    pb_emit_word pb_ptr 1

    \\ CB_BIND = 0x41 (bind + valid)
    pb_emit_header_inc pb_ptr 0x206c 1 1
    pb_emit_word pb_ptr 0x41

    \\ CB_LOAD_INLINE_DATA: nbytes/4 dwords via NONINC
    ndwords nbytes / 4
    pb_emit_header_noninc pb_ptr 0x206d ndwords 1
    for i 0 nbytes 4
        w → 32 data_ptr + i
        pb_emit_word pb_ptr w

\\ ========================================================================
\\ SPD (Shader Program Descriptor) builder — 384 bytes / 96 dwords.
\\
\\ The SPD is the 4th of 5 inline CB loads the hardware expects before QMD
\\ submission. It carries per-kernel metadata that doesn't fit in the QMD:
\\   - register_count at offset 0x094, byte[2] (bits 23:16)
\\   - entry_pc at 0x098 (lo32) and 0x09c (hi8)
\\   - packed entry_pc at 0x0a8
\\   - register_alloc_granule at 0x0b8 = ceil(reg_count / 8) + 10
\\   - block dims at 0x080/0x084/0x088
\\
\\ Other fields are set to probe-captured constants from a known-good launch.
\\ See docs/cbuf0_fields.md for the full field map.
\\
\\ Inputs:
\\   spd_ptr     — host-visible 384-byte buffer (caller allocates in BAR4)
\\   entry_pc    — 40-bit GPU VA of kernel .text
\\   reg_count   — register count from ELF .nv.info EIATTR_REGCOUNT
\\   block_x/y/z — thread block dimensions
\\   smem_size   — shared memory in bytes
\\ ========================================================================
spd_init spd_ptr :
    for i 0 384 8
        ← 64 spd_ptr + i 0

spd_build spd_ptr entry_pc reg_count block_x block_y block_z smem_size :
    spd_init spd_ptr

    \\ --- Static fields from probe captures ---
    ← 32 spd_ptr + 0x000 0x113f0000      \\ header / magic
    ← 32 spd_ptr + 0x028 0x00000003      \\ flags
    ← 32 spd_ptr + 0x02c 0x00190000      \\ shared memory config
    ← 32 spd_ptr + 0x034 0x10300011      \\ HW config flags
    ← 32 spd_ptr + 0x044 0x03000640      \\ grid/dispatch config
    ← 32 spd_ptr + 0x048 0xbc040040      \\ HW resource config
    ← 32 spd_ptr + 0x08c 0x80010000      \\ thread config flags
    ← 32 spd_ptr + 0x090 0x00010001      \\ constant

    \\ --- Block dimensions ---
    ← 32 spd_ptr + 0x080 block_x
    ← 32 spd_ptr + 0x084 block_y
    ← 32 spd_ptr + 0x088 block_z

    \\ --- register_count at 0x094: (0x08 << 24) | (reg_count << 16) | 0x0001 ---
    regword reg_count * 0x10000
    regword regword + 0x08000000
    regword regword + 0x0001
    ← 32 spd_ptr + 0x094 regword

    \\ --- entry_pc lo32 at 0x098, hi8 at 0x09c ---
    pc_hi entry_pc / 0x100000000
    pc_lo entry_pc - pc_hi * 0x100000000
    ← 32 spd_ptr + 0x098 pc_lo
    ← 32 spd_ptr + 0x09c pc_hi

    \\ --- packed entry_pc at 0x0a8: hi_byte | (lo32 >> 8) ---
    pc_shifted pc_lo / 256
    pc_packed pc_hi * 0x1000000 + pc_shifted
    ← 32 spd_ptr + 0x0a8 pc_packed

    \\ --- register_alloc_granule at 0x0b8 = ceil(reg_count / 8) + 10 ---
    granule reg_count + 7
    granule granule / 8
    granule granule + 10
    ← 32 spd_ptr + 0x0b8 granule

    \\ --- shared memory fields ---
    ← 32 spd_ptr + 0x0c8 0x0c900400      \\ shared memory base (from probe)
    ← 32 spd_ptr + 0x0cc 0x04800000      \\ shared memory pool size
    ← 32 spd_ptr + 0x0f8 0x0c900000      \\ shared memory base (variant)

    \\ --- local memory (scratch) ---
    ← 32 spd_ptr + 0x0c0 0x0c98b400      \\ local memory base
    ← 32 spd_ptr + 0x0c4 0x01800000      \\ local memory size
    ← 32 spd_ptr + 0x0e8 0x0c98b400      \\ local memory base (dup)

    membar_sys

\\ ========================================================================
\\ Full 5-part kernel launch sequence.
\\
\\ The Hopper compute engine expects 5 inline CB loads followed by the QMD
\\ submission. This replaces the old pb_emit_qmd which only did the QMD part.
\\
\\ Sequence (from docs/cbuf0_fields.md probe data):
\\   Load 0: QMD body           (528 bytes / 132 dwords)
\\   Load 1: Fence/timestamp    (8 bytes / 2 dwords)
\\   Load 2: Context descriptor (384 bytes / 96 dwords)
\\   Load 3: SPD                (384 bytes / 96 dwords) ← register_count here
\\   Load 4: Small patch        (4 bytes / 1 dword)
\\   Then:   SEND_PCAS_A + SEND_SIGNALING_PCAS2_B
\\
\\ Inputs:
\\   pb_ptr         — pushbuffer write cursor (first 8 bytes = cursor VA)
\\   qmd_gpu_va     — GPU VA of the 528-byte QMD (256-byte aligned)
\\   qmd_body_ptr   — host-visible pointer to QMD body
\\   spd_gpu_va     — GPU VA of the 384-byte SPD
\\   spd_body_ptr   — host-visible pointer to SPD body
\\   cbuf0_gpu_va   — GPU VA of cbuf0
\\ ========================================================================
pb_emit_launch pb_ptr qmd_gpu_va qmd_body_ptr spd_gpu_va spd_body_ptr cbuf0_gpu_va :
    \\ --- Load 0: QMD body (528 bytes) ---
    pb_emit_cb_load pb_ptr qmd_gpu_va qmd_body_ptr 528

    \\ --- Load 1: Fence/timestamp (8 bytes at QMD+0x210) ---
    fence_va qmd_gpu_va + 0x210
    \\ Build a 2-dword fence inline (zeros — no fence needed for basic launch)
    pb_emit_header_inc pb_ptr 0x2062 2 1
    hi fence_va / 0x100000000
    lo fence_va - hi * 0x100000000
    pb_emit_word pb_ptr hi
    pb_emit_word pb_ptr lo
    pb_emit_header_inc pb_ptr 0x2060 2 1
    pb_emit_word pb_ptr 8
    pb_emit_word pb_ptr 1
    pb_emit_header_inc pb_ptr 0x206c 1 1
    pb_emit_word pb_ptr 0x41
    pb_emit_header_noninc pb_ptr 0x206d 2 1
    pb_emit_word pb_ptr 0
    pb_emit_word pb_ptr 0

    \\ --- Load 2: Context descriptor (384 bytes, zeros for minimal launch) ---
    \\ For basic execution, an all-zero context descriptor is sufficient.
    \\ The driver populates this with context state; we can add fields as needed.
    pb_emit_header_inc pb_ptr 0x2062 2 1
    ctx_va cbuf0_gpu_va + 0x1000
    hi ctx_va / 0x100000000
    lo ctx_va - hi * 0x100000000
    pb_emit_word pb_ptr hi
    pb_emit_word pb_ptr lo
    pb_emit_header_inc pb_ptr 0x2060 2 1
    pb_emit_word pb_ptr 384
    pb_emit_word pb_ptr 1
    pb_emit_header_inc pb_ptr 0x206c 1 1
    pb_emit_word pb_ptr 0x41
    pb_emit_header_noninc pb_ptr 0x206d 96 1
    for i 0 96 1
        pb_emit_word pb_ptr 0

    \\ --- Load 3: SPD (384 bytes) — the one with register_count ---
    pb_emit_cb_load pb_ptr spd_gpu_va spd_body_ptr 384

    \\ --- Load 4: Small patch (4 bytes, zero) ---
    patch_va cbuf0_gpu_va + 0x2000
    pb_emit_header_inc pb_ptr 0x2062 2 1
    hi patch_va / 0x100000000
    lo patch_va - hi * 0x100000000
    pb_emit_word pb_ptr hi
    pb_emit_word pb_ptr lo
    pb_emit_header_inc pb_ptr 0x2060 2 1
    pb_emit_word pb_ptr 4
    pb_emit_word pb_ptr 1
    pb_emit_header_inc pb_ptr 0x206c 1 1
    pb_emit_word pb_ptr 0x41
    pb_emit_header_noninc pb_ptr 0x206d 1 1
    pb_emit_word pb_ptr 0

    \\ --- SEND_PCAS_A = qmd_gpu_va >> 8 ---
    pb_emit_header_inc pb_ptr 0x02b4 1 1
    pb_emit_word pb_ptr qmd_gpu_va / 256

    \\ --- SEND_SIGNALING_PCAS2_B = 0x0a (schedule + invalidate) ---
    pb_emit_header_inc pb_ptr 0x02c0 1 1
    pb_emit_word pb_ptr 0x0a

\\ Legacy wrapper — kept for backward compatibility with dispatch.ls / launch.ls.
\\ Emits only the QMD + SEND_PCAS without the 5-part CB load sequence.
\\ Use pb_emit_launch for the full sequence.
pb_emit_qmd pb_ptr qmd_gpu_va qmd_body_ptr :
    \\ Inline DMA target offset (upper,lower) — where the engine DMAs the QMD to
    pb_emit_header_inc pb_ptr 0x0188 2 1
    hi qmd_gpu_va / 0x100000000
    lo qmd_gpu_va - hi * 0x100000000
    pb_emit_word pb_ptr hi
    pb_emit_word pb_ptr lo

    \\ LINE_LENGTH_IN = 528, LINE_COUNT = 1
    pb_emit_header_inc pb_ptr 0x0180 2 1
    pb_emit_word pb_ptr 528
    pb_emit_word pb_ptr 1

    \\ LAUNCH_DMA = 0x41 (SRC=PIPELINED, DST=PIPELINED, COMPLETION=NONE)
    pb_emit_header_inc pb_ptr 0x01b0 1 1
    pb_emit_word pb_ptr 0x41

    \\ 528 bytes of QMD body = 132 dwords, via NONINC at LOAD_INLINE_DATA (0x01b4)
    pb_emit_header_noninc pb_ptr 0x01b4 132 1
    for i 0 528 4
        w → 32 qmd_body_ptr + i
        pb_emit_word pb_ptr w

    \\ SEND_PCAS_A = qmd_gpu_va >> 8 — hands QMD to Compute scheduler
    pb_emit_header_inc pb_ptr 0x02b4 1 1
    pb_emit_word pb_ptr qmd_gpu_va / 256

    \\ SEND_SIGNALING_PCAS2_B = 0x0a (PCAS2_ACTION = schedule + invalidate)
    pb_emit_header_inc pb_ptr 0x02c0 1 1
    pb_emit_word pb_ptr 0x0a
