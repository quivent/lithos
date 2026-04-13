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

\\ Emit the full QMD submission sequence on subchannel 1 (Compute).
\\ qmd_gpu_va is the GPU VA where the 528-byte QMD lives; qmd_body_ptr is the
\\ host-visible BAR4 pointer we copy the 132 dwords from.
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
