\\ mini.ls — Minimal Lithos compiler for macOS ARM64.
\\
\\ SCOPE (pragmatic): compiles a minuscule subset of Lithos — just enough
\\ to cycle the simplest test program through the stage-1 tool. Features:
\\
\\   * A single composition named `main` with no params.
\\   * Statements: `↓ $N int_literal` and bare `trap`.
\\
\\ Everything else is out of scope for this tiny proof-of-concept.
\\
\\ Design
\\ ------
\\ Wirth emits __TEXT as read-only, so mutable state lives in mmap'd pages.
\\ We hold only a handful of counters (cpos, tokens count) in stack locals
\\ and pass them explicitly; buffers (source, code, output) are mmap'd.
\\
\\ Output: macOS ARM64 Mach-O — run `codesign -s -` on the result.

\\ =========================================================================
\\ Entry point. dyld gives argc in X0, argv in X1.
\\ =========================================================================

main :
    argc ↑ $0
    argv ↑ $1
    lithos_main argc argv
    ↓ $16 1
    ↓ $0 0
    trap

\\ =========================================================================
\\ Heap: mmap an anonymous R/W page.
\\ =========================================================================

mmap_rw size :
    trap p 197 0 size 3 4098 -1 0
    p

\\ =========================================================================
\\ Memory access helpers (avoid wirth's `→ 64 base + off` bug).
\\ =========================================================================

st8 base off val :
    a base + off
    ← 8 a val

st32 base off val :
    a base + off
    ← 32 a val

st64 base off val :
    a base + off
    ← 64 a val

ld8 base off :
    a base + off
    r → 8 a
    r

ld64 base off :
    a base + off
    r → 64 a
    r

\\ =========================================================================
\\ File IO helpers (macOS Darwin syscalls).
\\ =========================================================================

open_ro path :
    trap fd 463 -2 path 0 0
    fd

open_wr path :
    trap fd 463 -2 path 1537 493
    fd

fsize fd :
    trap n 199 fd 0 2
    n

fseek0 fd :
    trap n 199 fd 0 0
    n

fclose fd :
    trap r 6 fd
    r

fread fd bufp n :
    trap r 3 fd bufp n
    r

fwrite fd bufp n :
    trap r 4 fd bufp n
    r

\\ =========================================================================
\\ ARM64 encoder. `code` is the code buffer; `cpp` points at a u64 cpos.
\\ =========================================================================

emit32 code cpp val :
    p ld64 cpp 0
    ← 32 code + p val
    st64 cpp 0 p + 4

\\ MOVZ rd, imm16 << (hw*16)
enc_movz rd imm hw :
    v 0xD2800000 | hw << 21 | (imm & 0xFFFF) << 5 | (rd & 31)
    v

\\ MOVK rd, imm16 << (hw*16)
enc_movk rd imm hw :
    v 0xF2800000 | hw << 21 | (imm & 0xFFFF) << 5 | (rd & 31)
    v

emit_mov_imm16 code cpp rd imm :
    v enc_movz rd imm 0
    emit32 code cpp v

emit_svc80 code cpp :
    emit32 code cpp 0xD4001001

\\ =========================================================================
\\ Minimal Mach-O writer.
\\ =========================================================================

wu32 obuf off v :
    ← 32 obuf + off v

wu64 obuf off v :
    lo v & 0xFFFFFFFF
    hi v >> 32 & 0xFFFFFFFF
    wu32 obuf off lo
    wu32 obuf off + 4 hi

memcpy_bytes dst doff src soff n :
    mci 0
    while mci < n
        sa soff + mci
        mcb ld8 src sa
        da doff + mci
        st8 dst da mcb
        mci mci + 1

\\ Write the 10 load commands + code + linkedit for a minimal Mach-O.
write_macho out_path code code_len :
    pz_sz 72
    tx_sz 152
    le_sz 72
    cf_sz 16
    sy_sz 24
    ds_sz 80
    dl_sz 32
    bv_sz 24
    mn_sz 24
    ld_sz 56
    lc_total pz_sz + tx_sz + le_sz + cf_sz + sy_sz + ds_sz + dl_sz + bv_sz + mn_sz + ld_sz
    ncmds 10
    hdr_sz 32
    lc_end hdr_sz + lc_total
    raw_code_off lc_end + 64 + 3
    code_off raw_code_off & 0xFFFFFFFC
    text_filesize code_off + code_len
    page 0x4000
    padded_sum text_filesize + page - 1
    text_padded padded_sum / page * page
    le_off text_padded
    cf_data_off le_off
    cf_data_size 56
    symtab_off cf_data_off + cf_data_size
    symtab_size 16
    strtab_off symtab_off + symtab_size
    strtab_size 32
    le_used strtab_off + strtab_size - le_off
    filesize_total le_off + le_used
    text_vmaddr 0x100000000
    le_vmaddr text_vmaddr + text_padded
    obuf mmap_rw filesize_total + 16
    \\ -- mach_header_64 --
    wu32 obuf 0 0xFEEDFACF
    wu32 obuf 4 0x0100000C
    wu32 obuf 8 0
    wu32 obuf 12 2
    wu32 obuf 16 ncmds
    wu32 obuf 20 lc_total
    wu32 obuf 24 0x00200085
    wu32 obuf 28 0
    \\ -- LC_SEGMENT_64 __PAGEZERO --
    p hdr_sz
    wu32 obuf p 0x19
    wu32 obuf p + 4 pz_sz
    st8 obuf p + 8 95
    st8 obuf p + 9 95
    st8 obuf p + 10 80
    st8 obuf p + 11 65
    st8 obuf p + 12 71
    st8 obuf p + 13 69
    st8 obuf p + 14 90
    st8 obuf p + 15 69
    st8 obuf p + 16 82
    st8 obuf p + 17 79
    wu64 obuf p + 24 0
    wu64 obuf p + 32 0x100000000
    wu64 obuf p + 40 0
    wu64 obuf p + 48 0
    wu32 obuf p + 56 0
    wu32 obuf p + 60 0
    wu32 obuf p + 64 0
    wu32 obuf p + 68 0
    p p + pz_sz
    \\ -- LC_SEGMENT_64 __TEXT --
    wu32 obuf p 0x19
    wu32 obuf p + 4 tx_sz
    st8 obuf p + 8 95
    st8 obuf p + 9 95
    st8 obuf p + 10 84
    st8 obuf p + 11 69
    st8 obuf p + 12 88
    st8 obuf p + 13 84
    wu64 obuf p + 24 text_vmaddr
    wu64 obuf p + 32 text_padded
    wu64 obuf p + 40 0
    wu64 obuf p + 48 text_padded
    wu32 obuf p + 56 5
    wu32 obuf p + 60 5
    wu32 obuf p + 64 1
    wu32 obuf p + 68 0
    p p + 72
    \\ section __text / __TEXT
    st8 obuf p 95
    st8 obuf p + 1 95
    st8 obuf p + 2 116
    st8 obuf p + 3 101
    st8 obuf p + 4 120
    st8 obuf p + 5 116
    st8 obuf p + 16 95
    st8 obuf p + 17 95
    st8 obuf p + 18 84
    st8 obuf p + 19 69
    st8 obuf p + 20 88
    st8 obuf p + 21 84
    wu64 obuf p + 32 text_vmaddr + code_off
    wu64 obuf p + 40 code_len
    wu32 obuf p + 48 code_off
    wu32 obuf p + 52 2
    wu32 obuf p + 56 0
    wu32 obuf p + 60 0
    wu32 obuf p + 64 0x80000400
    wu32 obuf p + 68 0
    wu32 obuf p + 72 0
    wu32 obuf p + 76 0
    p p + 80
    \\ -- LC_SEGMENT_64 __LINKEDIT --
    wu32 obuf p 0x19
    wu32 obuf p + 4 le_sz
    st8 obuf p + 8 95
    st8 obuf p + 9 95
    st8 obuf p + 10 76
    st8 obuf p + 11 73
    st8 obuf p + 12 78
    st8 obuf p + 13 75
    st8 obuf p + 14 69
    st8 obuf p + 15 68
    st8 obuf p + 16 73
    st8 obuf p + 17 84
    wu64 obuf p + 24 le_vmaddr
    wu64 obuf p + 32 page
    wu64 obuf p + 40 le_off
    wu64 obuf p + 48 le_used
    wu32 obuf p + 56 1
    wu32 obuf p + 60 1
    wu32 obuf p + 64 0
    wu32 obuf p + 68 0
    p p + 72
    \\ -- LC_DYLD_CHAINED_FIXUPS --
    wu32 obuf p 0x80000034
    wu32 obuf p + 4 cf_sz
    wu32 obuf p + 8 cf_data_off
    wu32 obuf p + 12 cf_data_size
    p p + 16
    \\ -- LC_SYMTAB --
    wu32 obuf p 0x2
    wu32 obuf p + 4 sy_sz
    wu32 obuf p + 8 symtab_off
    wu32 obuf p + 12 1
    wu32 obuf p + 16 strtab_off
    wu32 obuf p + 20 strtab_size
    p p + 24
    \\ -- LC_DYSYMTAB --
    wu32 obuf p 0xB
    wu32 obuf p + 4 ds_sz
    i 8
    while i < ds_sz
        st8 obuf p + i 0
        i i + 4
    wu32 obuf p + 16 1
    p p + ds_sz
    \\ -- LC_LOAD_DYLINKER --
    wu32 obuf p 0xE
    wu32 obuf p + 4 dl_sz
    wu32 obuf p + 8 12
    st8 obuf p + 12 47
    st8 obuf p + 13 117
    st8 obuf p + 14 115
    st8 obuf p + 15 114
    st8 obuf p + 16 47
    st8 obuf p + 17 108
    st8 obuf p + 18 105
    st8 obuf p + 19 98
    st8 obuf p + 20 47
    st8 obuf p + 21 100
    st8 obuf p + 22 121
    st8 obuf p + 23 108
    st8 obuf p + 24 100
    p p + 32
    \\ -- LC_BUILD_VERSION --
    wu32 obuf p 0x32
    wu32 obuf p + 4 bv_sz
    wu32 obuf p + 8 1
    wu32 obuf p + 12 0x000B0000
    wu32 obuf p + 16 0x000B0000
    wu32 obuf p + 20 0
    p p + 24
    \\ -- LC_MAIN --
    wu32 obuf p 0x80000028
    wu32 obuf p + 4 mn_sz
    wu64 obuf p + 8 code_off
    wu64 obuf p + 16 0
    p p + 24
    \\ -- LC_LOAD_DYLIB --
    wu32 obuf p 0xC
    wu32 obuf p + 4 ld_sz
    wu32 obuf p + 8 24
    wu32 obuf p + 12 2
    wu32 obuf p + 16 0x054C0000
    wu32 obuf p + 20 0x00010000
    \\ "/usr/lib/libSystem.B.dylib" — 26 bytes + NUL
    st8 obuf p + 24 47
    st8 obuf p + 25 117
    st8 obuf p + 26 115
    st8 obuf p + 27 114
    st8 obuf p + 28 47
    st8 obuf p + 29 108
    st8 obuf p + 30 105
    st8 obuf p + 31 98
    st8 obuf p + 32 47
    st8 obuf p + 33 108
    st8 obuf p + 34 105
    st8 obuf p + 35 98
    st8 obuf p + 36 83
    st8 obuf p + 37 121
    st8 obuf p + 38 115
    st8 obuf p + 39 116
    st8 obuf p + 40 101
    st8 obuf p + 41 109
    st8 obuf p + 42 46
    st8 obuf p + 43 66
    st8 obuf p + 44 46
    st8 obuf p + 45 100
    st8 obuf p + 46 121
    st8 obuf p + 47 108
    st8 obuf p + 48 105
    st8 obuf p + 49 98
    st8 obuf p + 50 0
    p p + 56
    \\ Copy code
    memcpy_bytes obuf code_off code 0 code_len
    \\ Chained fixups header (empty)
    cfh cf_data_off
    wu32 obuf cfh 0
    wu32 obuf cfh + 4 0x20
    wu32 obuf cfh + 8 0x30
    wu32 obuf cfh + 12 0x30
    wu32 obuf cfh + 16 0
    wu32 obuf cfh + 20 1
    wu32 obuf cfh + 24 0
    wu32 obuf cfh + 28 0
    wu32 obuf cfh + 32 3
    wu32 obuf cfh + 36 0
    wu32 obuf cfh + 40 0
    wu32 obuf cfh + 44 0
    wu32 obuf cfh + 48 0
    wu32 obuf cfh + 52 0
    \\ Symtab: 1 symbol
    wu32 obuf symtab_off 1
    st8 obuf symtab_off + 4 0x0F
    st8 obuf symtab_off + 5 1
    st8 obuf symtab_off + 6 0x10
    st8 obuf symtab_off + 7 0
    wu64 obuf symtab_off + 8 text_vmaddr
    \\ String table
    st8 obuf strtab_off 32
    st8 obuf strtab_off + 1 95
    st8 obuf strtab_off + 2 109
    st8 obuf strtab_off + 3 104
    st8 obuf strtab_off + 4 95
    st8 obuf strtab_off + 5 101
    st8 obuf strtab_off + 6 120
    st8 obuf strtab_off + 7 101
    st8 obuf strtab_off + 8 99
    st8 obuf strtab_off + 9 117
    st8 obuf strtab_off + 10 116
    st8 obuf strtab_off + 11 101
    st8 obuf strtab_off + 12 95
    st8 obuf strtab_off + 13 104
    st8 obuf strtab_off + 14 101
    st8 obuf strtab_off + 15 97
    st8 obuf strtab_off + 16 100
    st8 obuf strtab_off + 17 101
    st8 obuf strtab_off + 18 114
    st8 obuf strtab_off + 19 0
    fd open_wr out_path
    fwrite fd obuf filesize_total
    fclose fd
    0

\\ =========================================================================
\\ Simple scanner: walks the source byte by byte, finds a `↓ $N INT` or
\\ `trap` — enough to handle test-darwin-exit42.ls. No real tokenizer.
\\
\\ Returns nothing; fills code buffer via cpp counter.
\\ =========================================================================

is_digit c :
    r 0
    if>= c 48
        if<= c 57
            r 1
    r

is_space c :
    r 0
    if== c 32
        r 1
    if== c 9
        r 1
    if== c 13
        r 1
    r

\\ Skip spaces/tabs/CR (NOT newlines).
skip_ws src slen start :
    i start
    loop 1
    while loop
        if>= i slen
            loop 0
        if< i slen
            c ld8 src i
            s is_space c
            if== s 1
                i i + 1
            if== s 0
                loop 0
    i

\\ Parse an unsigned decimal integer starting at src+start.
\\ Returns (value, end_offset) via out ptr: out[0]=val, out[1]=end.
parse_uint src slen start out :
    i start
    v 0
    loop 1
    while loop
        if>= i slen
            loop 0
        if< i slen
            c ld8 src i
            d is_digit c
            if== d 0
                loop 0
            if== d 1
                v v * 10 + c - 48
                i i + 1
    st64 out 0 v
    st64 out 8 i

\\ Scan src for `↓ $N INT` patterns and `trap` tokens.
\\ Each `↓ $N INT` emits MOVZ XN, #INT (for INT < 65536).
\\ Each bare `trap` emits: MOVZ X16, #1 ; SVC #0x80.
\\ Composition headers (`main :` ...) emit prologue/epilogue.
\\
\\ Skips `\\...` comments and newlines cleanly.
compile_src src slen code cpp :
    tmp mmap_rw 32
    i 0
    in_header 0
    \\ Emit prologue for implicit main wrapping all top-level statements
    \\ actually we parse compositions explicitly.
    \\
    \\ Pass: we recognise exactly 3 patterns at statement indent > 0:
    \\   ↓ $N INT
    \\   trap
    \\ And at indent 0: `name :` starts a composition, close on next
    \\ non-indent line.
    in_compo 0
    while i < slen
        c ld8 src i
        \\ newline
        if== c 10
            i i + 1
        if!= c 10
            \\ check if we're at line start (previous was newline or i=0)
            at_start 0
            if== i 0
                at_start 1
            if!= i 0
                pv i - 1
                prev ld8 src pv
                if== prev 10
                    at_start 1
            if== at_start 1
                \\ Skip indent on this line
                j skip_ws src slen i
                if>= j slen
                    i slen
                if< j slen
                    cc ld8 src j
                    if== cc 10
                        i j
                    if!= cc 10
                        \\ comment line?
                        if== cc 92
                            j1 j + 1
                            kn j1 < slen
                            if== kn 1
                                cc2 ld8 src j1
                                if== cc2 92
                                    \\ skip to newline
                                    kk j
                                    kloop 1
                                    while kloop
                                        if>= kk slen
                                            kloop 0
                                        if< kk slen
                                            cx ld8 src kk
                                            if== cx 10
                                                kloop 0
                                            if!= cx 10
                                                kk kk + 1
                                    i kk
                        if!= cc 92
                            \\ Is this line a composition header (at col 0)?
                            if== j i
                                \\ Top-level line. Parse `name params : ...`
                                \\ We handle ONLY `main :` (optionally with params but
                                \\ compilable programs here have none).
                                \\ Find colon on this line
                                kk j
                                clook 1
                                found_colon 0
                                while clook
                                    if>= kk slen
                                        clook 0
                                    if< kk slen
                                        cx ld8 src kk
                                        if== cx 10
                                            clook 0
                                        if== cx 58
                                            found_colon 1
                                            clook 0
                                        if!= cx 58
                                            if!= cx 10
                                                kk kk + 1
                                if== found_colon 1
                                    \\ close previous composition if open
                                    if== in_compo 1
                                        \\ ADD SP, X29, #0
                                        emit32 code cpp 0x910003BF
                                        \\ LDP X29,X30,[SP],#16
                                        emit32 code cpp 0xA8C17BFD
                                        \\ RET
                                        emit32 code cpp 0xD65F03C0
                                    \\ Open new composition
                                    \\ STP X29,X30,[SP,#-16]!
                                    emit32 code cpp 0xA9BF7BFD
                                    \\ MOV X29, SP
                                    emit32 code cpp 0x910003FD
                                    \\ SUB SP,SP,#512
                                    emit32 code cpp 0xD1080000 | 512 << 10 | 0x3FF
                                    in_compo 1
                                    \\ advance i to after this line
                                    lk kk
                                    klo 1
                                    while klo
                                        if>= lk slen
                                            klo 0
                                        if< lk slen
                                            cx ld8 src lk
                                            if== cx 10
                                                klo 0
                                            if!= cx 10
                                                lk lk + 1
                                    i lk
                                if== found_colon 0
                                    \\ treat as no-op; skip line
                                    i kk
                            if!= j i
                                \\ Indented line — a statement inside a composition.
                                \\ Attempt to recognize ↓ or trap.
                                cj ld8 src j
                                if== cj 226
                                    \\ Possibly ↓ (0xE2 0x86 0x93)
                                    ja j + 1
                                    jb j + 2
                                    b1 ld8 src ja
                                    b2 ld8 src jb
                                    if== b1 134
                                        if== b2 147
                                            \\ ↓ token. Skip arrow, then $N, then INT.
                                            k j + 3
                                            k skip_ws src slen k
                                            \\ Expect $ — if missing, bail
                                            if< k slen
                                                cd ld8 src k
                                                if== cd 36
                                                    \\ $N
                                                    k k + 1
                                                    parse_uint src slen k tmp
                                                    rn ld64 tmp 0
                                                    kend ld64 tmp 8
                                                    kend skip_ws src slen kend
                                                    \\ parse INT (may start with 0x)
                                                    hex 0
                                                    if< kend slen
                                                        cp ld8 src kend
                                                        if== cp 48
                                                            kend1 kend + 1
                                                            kn2 kend1 < slen
                                                            if== kn2 1
                                                                cp2 ld8 src kend1
                                                                if== cp2 120
                                                                    hex 1
                                                    intv 0
                                                    if== hex 1
                                                        khex kend + 2
                                                        hloop 1
                                                        while hloop
                                                            if>= khex slen
                                                                hloop 0
                                                            if< khex slen
                                                                chh ld8 src khex
                                                                dv 0 - 1
                                                                if>= chh 48
                                                                    if<= chh 57
                                                                        dv chh - 48
                                                                if>= chh 97
                                                                    if<= chh 102
                                                                        dv chh - 87
                                                                if>= chh 65
                                                                    if<= chh 70
                                                                        dv chh - 55
                                                                if< dv 0
                                                                    hloop 0
                                                                if>= dv 0
                                                                    intv intv * 16 + dv
                                                                    khex khex + 1
                                                        kend khex
                                                    if== hex 0
                                                        parse_uint src slen kend tmp
                                                        intv ld64 tmp 0
                                                        kend ld64 tmp 8
                                                    \\ Emit MOVZ Xrn, #intv (low 16 bits)
                                                    lo16 intv & 0xFFFF
                                                    v enc_movz rn lo16 0
                                                    emit32 code cpp v
                                                    \\ Upper bits if needed
                                                    up intv >> 16
                                                    upper up & 0xFFFF
                                                    if> upper 0
                                                        vk enc_movk rn upper 1
                                                        emit32 code cpp vk
                                                    \\ advance to end of line
                                                    lk kend
                                                    klo 1
                                                    while klo
                                                        if>= lk slen
                                                            klo 0
                                                        if< lk slen
                                                            cx ld8 src lk
                                                            if== cx 10
                                                                klo 0
                                                            if!= cx 10
                                                                lk lk + 1
                                                    i lk
                                if!= cj 226
                                    \\ bare `trap` at j? Check 4 chars.
                                    kend j + 4
                                    is_trap 0
                                    if<= kend slen
                                        cc0 ld8 src j
                                        t1 j + 1
                                        t2 j + 2
                                        t3 j + 3
                                        cc1 ld8 src t1
                                        cc2 ld8 src t2
                                        cc3 ld8 src t3
                                        if== cc0 116
                                            if== cc1 114
                                                if== cc2 97
                                                    if== cc3 112
                                                        is_trap 1
                                    if== is_trap 1
                                        \\ MOVZ X16, #1
                                        v enc_movz 16 1 0
                                        emit32 code cpp v
                                        \\ SVC #0x80
                                        emit_svc80 code cpp
                                        \\ advance past line
                                        lk j + 4
                                        klo 1
                                        while klo
                                            if>= lk slen
                                                klo 0
                                            if< lk slen
                                                cx ld8 src lk
                                                if== cx 10
                                                    klo 0
                                                if!= cx 10
                                                    lk lk + 1
                                        i lk
                                    if== is_trap 0
                                        \\ Unknown statement — skip rest of line
                                        lk j
                                        klo 1
                                        while klo
                                            if>= lk slen
                                                klo 0
                                            if< lk slen
                                                cx ld8 src lk
                                                if== cx 10
                                                    klo 0
                                                if!= cx 10
                                                    lk lk + 1
                                        i lk
            if== at_start 0
                \\ not at line start (shouldn't normally happen since we advance
                \\ by full lines); consume char to make progress
                i i + 1
    \\ Close final composition
    if== in_compo 1
        emit32 code cpp 0x910003BF
        emit32 code cpp 0xA8C17BFD
        emit32 code cpp 0xD65F03C0
    0

\\ =========================================================================
\\ lithos_main: argc argv
\\ =========================================================================

lithos_main argc argv :
    if< argc 3
        ↓ $16 1
        ↓ $0 2
        trap
    a1 argv + 8
    src_path → 64 a1
    a2 argv + 16
    out_path → 64 a2
    fd open_ro src_path
    if< fd 0
        ↓ $16 1
        ↓ $0 3
        trap
    slen fsize fd
    src mmap_rw slen + 16
    fseek0 fd
    fread fd src slen
    fclose fd
    code mmap_rw 65536
    cpp mmap_rw 16
    st64 cpp 0 0
    compile_src src slen code cpp
    code_len ld64 cpp 0
    write_macho out_path code code_len
    0
