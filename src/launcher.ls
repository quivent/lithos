\\ Lithos inference launcher — end-to-end inference for Qwen 3.5 27B
\\
\\ Replaces src/launcher.s (4096 lines of CUDA-dependent code).
\\ Target: GH200, no CUDA, no Python, no external dependencies.
\\
\\ Architecture:
\\   One cooperative megakernel per model (forward pass).
\\   All state lives in BAR4 coherent HBM.
\\   Weights are baked into the ELF .text as FMUL-IMM / HFMA2 immediates.
\\   One QMD doorbell write per decoded token.

\\ ─────────────────────────────────────────────────────────────────────
\\ Entry point
\\ ─────────────────────────────────────────────────────────────────────

lithos_main argc argv :
    \\ 1. Bring the GPU up. runtime_init opens vfio, maps BAR0 (regs)
    \\    and BAR4 (coherent HBM aperture), boots GSP, initializes the
    \\    host-channel ring, and returns the two base pointers.
    bar0 bar4 runtime_init

    \\ BAR4 heap: 128 GB aperture starting at bar4.
    mem_init bar4 0x2000000000

    \\ 2. Load the compiled forward-pass megakernel ELF.
    \\    compiler.ls produced kernels/forward.elf — a single cubin
    \\    whose .text is all 64 layers of Qwen 3.5 27B flattened.
    forward_elf_cpu forward_elf_size load_file "kernels/forward.elf"
    forward_entry elf_load forward_elf_cpu forward_elf_size

    \\ 3. Allocate runtime buffers in BAR4 HBM.
    \\    Activation scratch (residual + per-layer working set).
    activations mem_alloc 0x200000 256

    \\ KV cache: 16 full-attention layers × 32768 positions × 8 kv-heads
    \\ × 128 head_dim × 2 (K,V) × 2 bytes (f16) = 2 GB.
    kv_cache mem_alloc 0x80000000 256

    \\ DeltaNet recurrent state: 48 layers × 16 key-heads × 3 value-heads
    \\ × 128 × 128 × 4 bytes (f32) = 240 MB.
    dn_state mem_alloc 0x0F000000 256

    \\ Output logits: 248320 × 4 bytes (f32).
    logits mem_alloc 0x00100000 256

    \\ Position counter + done flag (one u32 each, coherent).
    position_buf mem_alloc 64 64
    done_flag mem_alloc 64 64

    \\ Parameter block passed through constant memory to the megakernel.
    params mem_alloc 256 256

    \\ 4. Read prompt tokens.
    prompt prompt_len tokenize argv

    \\ 5. Seed position and copy prompt into activations ring.
    ← 32 position_buf 0

    \\ 6. Decode loop. For each position, dispatch the megakernel once.
    \\    The megakernel walks all 64 layers, the LM head, and argmax
    \\    sampling in a single cooperative grid with 67 internal
    \\    grid.sync barriers. Host sees one dispatch per token.

    for pos 0 262144 1
        \\ Current token: prompt during prefill, prior output otherwise.
        if< pos prompt_len
            tok → 32 prompt + pos * 4
        if>= pos prompt_len
            tok → 32 logits

        \\ Build the kernel parameter block:
        \\   [0]  token id
        \\   [1]  position
        \\   [2]  activations base
        \\   [3]  kv_cache base
        \\   [4]  dn_state base
        \\   [5]  logits base
        \\   [6]  done_flag
        ← 32 params + 0  tok
        ← 32 params + 8  pos
        ← 64 params + 16 activations
        ← 64 params + 24 kv_cache
        ← 64 params + 32 dn_state
        ← 64 params + 40 logits
        ← 64 params + 48 done_flag

        \\ Clear done flag before dispatch.
        ← 32 done_flag 0

        \\ Single dispatch. Grid = (132 cores, 1, 1) × (256, 1, 1) threads,
        \\ 32 KB shared per CTA. launch_kernel_sync writes the QMD,
        \\ rings the doorbell, and spins on done_flag.
        launch_kernel_sync forward_entry 132 1 1 256 1 1 32768 params done_flag

        \\ Emit the sampled token (argmax already written to logits[0]
        \\ by the tail of the megakernel).
        next → 32 logits
        emit_token next

        \\ EOS → stop. Qwen 3.5 uses 151645 (im_end) as the chat EOS.
        if== next 151645
            done_flag ← 8 done_flag 1

    \\ 7. Teardown.
    runtime_teardown bar0 bar4
    runtime_exit 0

\\ ─────────────────────────────────────────────────────────────────────
\\ I/O helpers — ARM64 syscall primitives
\\ ─────────────────────────────────────────────────────────────────────

load_file path :
    \\ openat(AT_FDCWD, path, O_RDONLY)
    ↓ $8 56
    ↓ $0 -100
    ↓ $1 path
    ↓ $2 0
    trap
    fd ↑ $0

    \\ fstat to get size — statbuf on the syscall stack frame.
    ↓ $8 80
    ↓ $0 fd
    ↓ $1 statbuf
    trap
    size → 64 statbuf + 48

    \\ mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)
    ↓ $8 222
    ↓ $0 0
    ↓ $1 size
    ↓ $2 1
    ↓ $3 2
    ↓ $4 fd
    ↓ $5 0
    trap
    buf ↑ $0

    \\ close(fd)
    ↓ $8 57
    ↓ $0 fd
    trap

\\ Placeholder tokenizer. Real BPE comes later; for now read a pre-
\\ tokenized u32 stream from argv[1] and mmap it directly.
tokenize argv :
    path → 64 argv + 8
    buf size load_file path
    prompt buf
    prompt_len size / 4

\\ Write one u32 token as decimal to stdout (fd 1) followed by a space.
\\ Full detokenization happens in a separate pass later.
emit_token tok :
    ← 32 emit_buf tok
    ← 8 emit_buf + 4 32

    ↓ $8 64
    ↓ $0 1
    ↓ $1 emit_buf
    ↓ $2 5
    trap
