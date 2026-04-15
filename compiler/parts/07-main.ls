\\ ============================================================================
\\ parts/07-main.ls — Lithos minimal compiler entry point
\\ ============================================================================
\\
\\ Pipeline:
\\   1. Parse argv: argv[1] = source path, argv[2] = output path.
\\   2. mmap the source file (mmap_file is from the host runtime).
\\   3. lex   — tokenize source bytes into the `tokens` buffer.
\\   4. walk  — phase 1 collects compositions, phase 2 emits SASS / ARM64.
\\   5. wrap  — choose backend based on whether GPU or ARM64 code was emitted.
\\   6. exit.
\\
\\ ARM64 calling convention:
\\   X0 = argc, X1 = argv (pointer to char* array)
\\
\\ The host helpers (mmap_file, write_file) are themselves Lithos compositions
\\ tagged with `host`, so they run on Grace and use SVC #0 for syscalls.

buf src_path_v 8
buf out_path_v 8
buf src_base_v 8
buf src_size_v 8

host main argc argv :
    \\ Validate argument count.
    if< argc 3
        goto main_err

    \\ Pull argv[1] / argv[2] from the argv pointer array.
    sp → 64 (argv + 8)
    op → 64 (argv + 16)
    ← 64 src_path_v sp
    ← 64 out_path_v op

    \\ Step 1: mmap the source file. The host helper leaves base in X6 and
    \\ size in X7 (see compiler-fat's mmap_file convention).
    _sp → 64 src_path_v
    mmap_file _sp
    base ↑ $6
    size ↑ $7
    ← 64 src_base_v base
    ← 64 src_size_v size

    \\ Step 2: tokenize.
    _b → 64 src_base_v
    _s → 64 src_size_v
    lithos_lex _b _s

    \\ Step 3: walk and emit. Sets any_gpu_v / any_arm64_v as a side effect.
    walk_top_level

    \\ Step 4: choose backend and write the output file.
    _gp → 64 gpu_pos_v
    if> _gp 0
        goto main_gpu
    _ap → 64 arm64_pos_v
    if> _ap 0
        goto main_arm64
    goto main_ok

    label main_gpu
    \\ Cubin (NVIDIA ELF) for the GPU side.
    _gp → 64 gpu_pos_v
    _op → 64 out_path_v
    cubin_write gpu_buf _gp _op
    goto main_ok

    label main_arm64
    _ap → 64 arm64_pos_v
    _op → 64 out_path_v
    arm64_elf_write arm64_buf _ap _op
    goto main_ok

    label main_ok
    ↓ $8 93
    ↓ $0 0
    trap

    label main_err
    ↓ $8 93
    ↓ $0 1
    trap
