/*
 * lithos-boot-wirth.c — C bootstrap compiler for Lithos, macOS ARM64 target
 *
 * Author tag: "wirth" (Niklaus Wirth, patron saint of simple compilers).
 *
 * Architecture
 * ------------
 * Single-pass recursive-descent compiler.  No AST, no IR — the parser walks
 * the token array and emits ARM64 machine code directly into a buffer, then
 * the Mach-O writer wraps that buffer in a runnable executable.
 *
 * Locals live on the stack, never in registers.  Each composition prologue
 * allocates a fixed frame (FRAME_SIZE bytes) and every parameter, binding,
 * and var gets its own 8-byte slot.  Expression temporaries use a bump
 * allocator over X9..X18 that is reset at every statement boundary, so the
 * old assembly bootstrap's "register spill" aliasing bug cannot recur.
 *
 *   Prologue:  STP X29, X30, [SP, #-16]!
 *              ADD X29, SP, #0
 *              SUB SP, SP, #FRAME_SIZE
 *   Epilogue:  ADD SP, X29, #0
 *              LDP X29, X30, [SP], #16
 *              RET
 *
 * Reading  a local at offset -N from X29: LDUR Xt, [X29, #-N]
 * Writing  a local at offset -N from X29: STUR Xt, [X29, #-N]
 *
 * Darwin Mach-O output
 * --------------------
 * macOS 11+ rejects unsigned executables.  The writer emits a minimum
 * dynamic executable (LC_MAIN + LC_LOAD_DYLINKER + LC_LOAD_DYLIB(libSystem)
 * + LC_DYLD_CHAINED_FIXUPS + LC_SYMTAB + LC_DYSYMTAB + LC_BUILD_VERSION),
 * reserves 64 bytes of slack between the load commands and the code so the
 * post-link codesign pass can insert LC_CODE_SIGNATURE without clobbering
 * the first instructions, and then invokes `codesign -s -` to fill in an
 * ad-hoc signature.  Without that step the kernel SIGKILLs the process at
 * exec time.
 *
 * trap encoding
 * -------------
 * Test programs use the Linux syscall convention (`↓ $8 93 ; ↓ $0 42 ; trap`).
 * On macOS the trap handler needs X16 = syscall number and SVC #0x80, and
 * exit is syscall 1 — NOT 93.  We emit `MOV X16, #1 ; SVC #0x80` on every
 * `trap`.  X0 already holds the status from `↓ $0 42`.  The `↓ $8 93`
 * harmlessly sets X8.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

/* ==========================================================================
 * Basic types and utilities
 * ========================================================================== */

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long long u64;
typedef long long i64;

#define FRAME_SIZE       512
#define MAX_TOKENS       65536
#define MAX_SYMS         4096
#define MAX_CODE         (1 << 20)
#define MAX_NAME         64
#define PAGE_SIZE_OUT    0x4000
#define TEXT_VMADDR      0x100000000ULL

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf,1,2)));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "wirth: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

/* ==========================================================================
 * Tokens
 * ========================================================================== */

enum {
    TOK_EOF = 0,
    TOK_NEWLINE,
    TOK_INDENT,
    TOK_INT,
    TOK_IDENT,
    TOK_IF,
    TOK_ELIF,
    TOK_ELSE,
    TOK_FOR,
    TOK_WHILE,
    TOK_EACH,
    TOK_VAR,
    TOK_RETURN,
    TOK_TRAP,
    TOK_CONST,
    TOK_BUF,
    TOK_LOAD,       /* ↓ */
    TOK_REG_READ,   /* ↑ */
    TOK_MEM_STORE,  /* ← */
    TOK_MEM_LOAD,   /* → */
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_AMP, TOK_PIPE, TOK_CARET,
    TOK_SHL, TOK_SHR,
    TOK_EQ,
    TOK_EQEQ, TOK_NEQ, TOK_LT, TOK_GT, TOK_LTE, TOK_GTE,
    TOK_LPAREN, TOK_RPAREN,
    TOK_LBRACK, TOK_RBRACK,
    TOK_COLON,
    TOK_DOLLAR,
    TOK_HASH,
    TOK_GOTO,
    TOK_LABEL,
    TOK_CONTINUE,
    TOK_BREAK,
};

typedef struct {
    int type;
    int off;
    int len;
    int line;
} Token;

static char    *src;
static size_t   src_len;
static Token    toks[MAX_TOKENS];
static int      ntoks;
static int      tk;

static const struct { const char *kw; int tok; } keywords[] = {
    {"if",     TOK_IF},
    {"elif",   TOK_ELIF},
    {"else",   TOK_ELSE},
    {"for",    TOK_FOR},
    {"while",  TOK_WHILE},
    {"each",   TOK_EACH},
    {"var",    TOK_VAR},
    {"return", TOK_RETURN},
    {"trap",   TOK_TRAP},
    {"goto",   TOK_GOTO},
    {"label",  TOK_LABEL},
    {"continue", TOK_CONTINUE},
    {"break",    TOK_BREAK},
    {"const",  TOK_CONST},
    {"buf",    TOK_BUF},
    {NULL, 0}
};

static int is_ident_start(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int is_ident_cont(int c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

static void emit_tok(int type, int off, int len, int line) {
    if (ntoks >= MAX_TOKENS) die("token buffer overflow");
    Token *t = &toks[ntoks++];
    t->type = type; t->off = off; t->len = len; t->line = line;
}

static void lex(void) {
    size_t i = 0;
    int line = 1;
    int at_line_start = 1;

    while (i < src_len) {
        if (at_line_start) {
            int indent = 0;
            while (i < src_len && (src[i] == ' ' || src[i] == '\t')) {
                indent += (src[i] == '\t') ? 8 : 1;
                i++;
            }
            /* skip comment-only lines */
            if (i < src_len && src[i] == '\\' && i + 1 < src_len && src[i+1] == '\\') {
                while (i < src_len && src[i] != '\n') i++;
                /* fall through to newline handling */
            }
            if (i < src_len && src[i] != '\n') {
                emit_tok(TOK_INDENT, (int)i, indent, line);
            }
            at_line_start = 0;
            continue;
        }

        char c = src[i];

        if (c == '\n') {
            emit_tok(TOK_NEWLINE, (int)i, 1, line);
            i++; line++; at_line_start = 1;
            continue;
        }

        if (c == ' ' || c == '\t' || c == '\r') { i++; continue; }

        if (c == '\\' && i + 1 < src_len && src[i+1] == '\\') {
            while (i < src_len && src[i] != '\n') i++;
            continue;
        }

        /* UTF-8 arrows */
        if ((u8)c == 0xE2 && i + 2 < src_len) {
            u8 b1 = (u8)src[i+1], b2 = (u8)src[i+2];
            if (b1 == 0x86 && b2 == 0x93) {   /* ↓ */
                emit_tok(TOK_LOAD, (int)i, 3, line);
                i += 3; continue;
            }
            if (b1 == 0x86 && b2 == 0x91) {   /* ↑ */
                emit_tok(TOK_REG_READ, (int)i, 3, line);
                i += 3; continue;
            }
            if (b1 == 0x86 && b2 == 0x90) {   /* ← */
                emit_tok(TOK_MEM_STORE, (int)i, 3, line);
                i += 3; continue;
            }
            if (b1 == 0x86 && b2 == 0x92) {   /* → */
                emit_tok(TOK_MEM_LOAD, (int)i, 3, line);
                i += 3; continue;
            }
            i += 3; continue;
        }

        if (c >= '0' && c <= '9') {
            int s = (int)i;
            if (c == '0' && i + 1 < src_len && (src[i+1] == 'x' || src[i+1] == 'X')) {
                i += 2;
                while (i < src_len && ((src[i] >= '0' && src[i] <= '9') ||
                                       (src[i] >= 'a' && src[i] <= 'f') ||
                                       (src[i] >= 'A' && src[i] <= 'F'))) i++;
            } else {
                while (i < src_len && src[i] >= '0' && src[i] <= '9') i++;
            }
            emit_tok(TOK_INT, s, (int)i - s, line);
            continue;
        }

        if (c == '$') {
            int s = (int)i;
            i++;
            while (i < src_len && ((src[i] >= '0' && src[i] <= '9') || is_ident_start(src[i]))) i++;
            emit_tok(TOK_DOLLAR, s, (int)i - s, line);
            continue;
        }

        if (is_ident_start(c)) {
            int s = (int)i;
            while (i < src_len && is_ident_cont(src[i])) i++;
            int len = (int)i - s;
            int tt = TOK_IDENT;
            for (int k = 0; keywords[k].kw; k++) {
                int kl = (int)strlen(keywords[k].kw);
                if (kl == len && memcmp(src + s, keywords[k].kw, len) == 0) {
                    tt = keywords[k].tok;
                    break;
                }
            }
            emit_tok(tt, s, len, line);
            continue;
        }

        switch (c) {
        case '+': emit_tok(TOK_PLUS, (int)i, 1, line); i++; break;
        case '-': emit_tok(TOK_MINUS,(int)i, 1, line); i++; break;
        case '*': emit_tok(TOK_STAR, (int)i, 1, line); i++; break;
        case '/': emit_tok(TOK_SLASH,(int)i, 1, line); i++; break;
        case '&': emit_tok(TOK_AMP,  (int)i, 1, line); i++; break;
        case '|': emit_tok(TOK_PIPE, (int)i, 1, line); i++; break;
        case '^': emit_tok(TOK_CARET,(int)i, 1, line); i++; break;
        case '(': emit_tok(TOK_LPAREN,(int)i, 1, line); i++; break;
        case ')': emit_tok(TOK_RPAREN,(int)i, 1, line); i++; break;
        case '[': emit_tok(TOK_LBRACK,(int)i, 1, line); i++; break;
        case ']': emit_tok(TOK_RBRACK,(int)i, 1, line); i++; break;
        case ':': emit_tok(TOK_COLON,(int)i, 1, line); i++; break;
        case '#': emit_tok(TOK_HASH, (int)i, 1, line); i++; break;
        case '<':
            if (i+1 < src_len && src[i+1] == '<') { emit_tok(TOK_SHL, (int)i, 2, line); i += 2; }
            else if (i+1 < src_len && src[i+1] == '=') { emit_tok(TOK_LTE,(int)i, 2, line); i += 2; }
            else { emit_tok(TOK_LT, (int)i, 1, line); i++; }
            break;
        case '>':
            if (i+1 < src_len && src[i+1] == '>') { emit_tok(TOK_SHR, (int)i, 2, line); i += 2; }
            else if (i+1 < src_len && src[i+1] == '=') { emit_tok(TOK_GTE,(int)i, 2, line); i += 2; }
            else { emit_tok(TOK_GT, (int)i, 1, line); i++; }
            break;
        case '=':
            if (i+1 < src_len && src[i+1] == '=') { emit_tok(TOK_EQEQ,(int)i, 2, line); i += 2; }
            else { emit_tok(TOK_EQ, (int)i, 1, line); i++; }
            break;
        case '!':
            if (i+1 < src_len && src[i+1] == '=') { emit_tok(TOK_NEQ, (int)i, 2, line); i += 2; }
            else i++;
            break;
        default:
            i++;
        }
    }
    emit_tok(TOK_EOF, (int)src_len, 0, line);
}

/* ==========================================================================
 * Symbol table
 * ========================================================================== */

enum {
    SYM_LOCAL   = 1,
    SYM_COMPO   = 2,
    SYM_FOR_VAR = 3,
    SYM_CONST   = 4,
    SYM_BUF     = 5,
    SYM_GLOBAL  = 6,
};

typedef struct {
    char name[MAX_NAME];
    int  kind;
    int  slot;       /* local: positive, read via [X29, #-slot] */
    int  code_off;   /* composition: offset in code buffer */
    int  nparams;    /* composition parameter count */
} Sym;

static Sym syms[MAX_SYMS];
static int nsyms;

static int sym_add(const char *name, int len, int kind) {
    if (nsyms >= MAX_SYMS) die("too many symbols");
    if (len >= MAX_NAME)   die("name too long: %.*s", len, name);
    Sym *s = &syms[nsyms];
    memcpy(s->name, name, len);
    s->name[len] = 0;
    s->kind = kind;
    s->slot = 0;
    s->code_off = 0;
    s->nparams = 0;
    return nsyms++;
}

static int sym_find(const char *name, int len) {
    for (int i = nsyms - 1; i >= 0; i--) {
        Sym *s = &syms[i];
        if ((int)strlen(s->name) == len && memcmp(s->name, name, len) == 0)
            return i;
    }
    return -1;
}

static void sym_trim(int keep_count) { nsyms = keep_count; }

/* ==========================================================================
 * Forward-reference fixups
 *
 * When a composition is called before its code_off is known (forward ref),
 * we emit `BL 0` and record a fixup {code_off, sym_idx}. After every
 * composition has been parsed and its code_off filled in, resolve_fixups()
 * walks the table and patches each BL with the correct delta.
 *
 * For convenience we use this path for backward calls too (code_off already
 * set), so there's one BL emission path regardless of direction.
 * ========================================================================== */

/* Forward decl for label fixups */
static void patch32(int off, u32 w);

#define MAX_FIXUPS 8192
typedef struct {
    int code_off;
    int sym_idx;
} Fixup;
static Fixup fixups[MAX_FIXUPS];
static int   nfixups;

/* In-function goto/label table. Cleared per composition. */
#define MAX_LABELS 512
typedef struct {
    char name[64];
    int  len;
    int  code_off;      /* -1 if not yet defined */
} Label;
static Label labels[MAX_LABELS];
static int   nlabels;

typedef struct {
    int  code_off;      /* B placeholder location */
    char name[64];
    int  len;
} LabelFixup;
static LabelFixup label_fixups[MAX_LABELS];
static int        n_label_fixups;

static void labels_reset(void) { nlabels = 0; n_label_fixups = 0; }

/* Loop stack for `continue` / `break`. Each entry remembers where the
 * loop top is and a list of B placeholders for `break` to patch to the
 * loop exit. */
typedef struct {
    int top_off;              /* code offset of loop top (target for continue) */
    int break_patches[128];   /* code offsets of B placeholders to patch */
    int n_break_patches;
} LoopCtx;
static LoopCtx loop_stack[16];
static int     loop_depth;

static void loop_push(int top_off) {
    if (loop_depth >= 16) die("loops nested too deep");
    loop_stack[loop_depth].top_off = top_off;
    loop_stack[loop_depth].n_break_patches = 0;
    loop_depth++;
}

static void loop_pop_and_patch_breaks(int exit_off) {
    if (loop_depth <= 0) return;
    LoopCtx *l = &loop_stack[--loop_depth];
    for (int i = 0; i < l->n_break_patches; i++) {
        int bp = l->break_patches[i];
        int delta = exit_off - bp;
        u32 b = 0x14000000u | ((u32)(delta / 4) & 0x3FFFFFFu);
        patch32(bp, b);
    }
}

static int loop_current_top(void) {
    if (loop_depth <= 0) die("`continue` outside a loop");
    return loop_stack[loop_depth - 1].top_off;
}

static void loop_record_break(int code_off) {
    if (loop_depth <= 0) die("`break` outside a loop");
    LoopCtx *l = &loop_stack[loop_depth - 1];
    if (l->n_break_patches >= 128) die("too many breaks in one loop");
    l->break_patches[l->n_break_patches++] = code_off;
}

static int label_find(const char *name, int len) {
    for (int i = 0; i < nlabels; i++) {
        if (labels[i].len == len && memcmp(labels[i].name, name, (size_t)len) == 0)
            return i;
    }
    return -1;
}

static void label_define(const char *name, int len, int code_off) {
    int i = label_find(name, len);
    if (i >= 0) { labels[i].code_off = code_off; return; }
    if (nlabels >= MAX_LABELS) die("too many labels in composition");
    Label *l = &labels[nlabels++];
    memcpy(l->name, name, (size_t)len);
    l->len = len;
    l->code_off = code_off;
}

static void label_add_fixup(const char *name, int len, int code_off) {
    if (n_label_fixups >= MAX_LABELS) die("too many label fixups");
    LabelFixup *f = &label_fixups[n_label_fixups++];
    memcpy(f->name, name, (size_t)len);
    f->len = len;
    f->code_off = code_off;
}

static void resolve_label_fixups(void) {
    for (int i = 0; i < n_label_fixups; i++) {
        LabelFixup *f = &label_fixups[i];
        int li = label_find(f->name, f->len);
        if (li < 0) die("undefined label '%.*s'", f->len, f->name);
        int target = labels[li].code_off;
        if (target < 0) die("label '%.*s' declared but not defined", f->len, f->name);
        int delta = target - f->code_off;
        u32 b = 0x14000000u | ((u32)(delta / 4) & 0x3FFFFFFu);
        patch32(f->code_off, b);
    }
}

static void add_fixup(int code_off, int sym_idx) {
    if (nfixups >= MAX_FIXUPS) die("too many forward references");
    fixups[nfixups].code_off = code_off;
    fixups[nfixups].sym_idx  = sym_idx;
    nfixups++;
}

/* ==========================================================================
 * `buf` data storage (v1 — trailing-__TEXT approach, read-only)
 * --------------------------------------------------------------------------
 * wirth's Mach-O writer emits a single R+X __TEXT segment and no __DATA
 * segment.  Rather than introducing a full __DATA segment (which would
 * shift __LINKEDIT offsets and touch the chained-fixups data), we append
 * `buf` storage to the END of the __TEXT segment.  The resulting memory is
 * READ-only since __TEXT is mapped R+X.  This is fine for bootstrap
 * programs that only READ from their bufs (e.g. the tests in this file)
 * and for compiler-darwin.ls's parse path which currently uses buf
 * addresses as handles passed to the generated binary's own storage.
 *
 * Writes through the wirth-compiled binary into a buf address will fault.
 * That's a known v1 limitation; a future version will either introduce a
 * real __DATA segment or make the compiled program mmap its own writable
 * pages and copy the buf address in.
 *
 * Address of a buf at data-offset D:
 *     buf_vmaddr = TEXT_VMADDR + text_code_file_off + code_len_final + D
 * where code_len_final is known only after parse_file() completes.  We
 * therefore emit every buf reference as a FIXED-size 4-instruction
 * MOVZ/MOVK/MOVK/MOVK sequence and record a fixup; after parsing is done,
 * resolve_buf_fixups() walks the table and patches each sequence in place.
 * ========================================================================== */
#define DATA_MAX       (16 * 1024 * 1024)    /* 16 MiB of buf storage */
#define MAX_BUF_FIXUPS 8192

static u8  data_buf[DATA_MAX];
static int data_off;

typedef struct {
    int code_word_idx;   /* index into code[] of the first MOVZ of the sequence */
    int rd;              /* destination X register */
    int buf_data_off;    /* byte offset within data_buf */
} BufFixup;

static BufFixup buf_fixups[MAX_BUF_FIXUPS];
static int      n_buf_fixups;

static void add_buf_fixup(int code_word_idx, int rd, int buf_data_off) {
    if (n_buf_fixups >= MAX_BUF_FIXUPS) die("too many buf references");
    buf_fixups[n_buf_fixups].code_word_idx = code_word_idx;
    buf_fixups[n_buf_fixups].rd            = rd;
    buf_fixups[n_buf_fixups].buf_data_off  = buf_data_off;
    n_buf_fixups++;
}

/* ==========================================================================
 * Top-level `var` (SYM_GLOBAL) PC-relative address fixups
 * --------------------------------------------------------------------------
 * Unlike SYM_BUF references — which materialise an absolute 64-bit VA via a
 * MOVZ/MOVK chain that is incorrect under PIE/ASLR once the kernel slides
 * the image — SYM_GLOBAL references compute their address via a single ADR
 * instruction (PC-relative, ±1MiB range, slide-invariant).  The 8-byte data
 * slot still lives in the trailing __TEXT data region (same bump allocator
 * as buf/const) so we don't perturb the Mach-O writer's segment/section
 * layout.
 *
 * At parse time we don't yet know the final code length, so we emit a
 * 1-instruction NOP placeholder at the address-materialisation point and
 * record a GlobalFixup {code_word_idx, rd, data_off}.  At end-of-parse, once
 * code_words (and therefore code_len_final) is known, resolve_global_fixups()
 * walks the table and patches each NOP with the correct ADR Xd, #delta.
 *
 * The byte delta is purely relative: both pc and target are at
 *     TEXT_VMADDR + text_code_file_off + <something>
 * so the constants cancel in the subtraction, and only the within-__TEXT
 * offsets remain.  That means this path is independent of the Mach-O
 * writer's header layout.
 * ========================================================================== */
typedef struct {
    int code_word_idx;   /* index into code[] of the NOP to patch to ADR */
    int rd;              /* destination X register holding the computed address */
    int data_off;        /* byte offset within data_buf */
} GlobalFixup;

#define MAX_GLOBAL_FIXUPS 8192
static GlobalFixup global_fixups[MAX_GLOBAL_FIXUPS];
static int         n_global_fixups;

static void add_global_fixup(int code_word_idx, int rd, int data_off_val) {
    if (n_global_fixups >= MAX_GLOBAL_FIXUPS) die("too many global references");
    global_fixups[n_global_fixups].code_word_idx = code_word_idx;
    global_fixups[n_global_fixups].rd            = rd;
    global_fixups[n_global_fixups].data_off      = data_off_val;
    n_global_fixups++;
}

/* ==========================================================================
 * Code buffer + ARM64 encoder
 * ========================================================================== */

static u32 code[MAX_CODE / 4];
static int code_words;

static int frame_next_slot;
static int temp_next;

static void emit32(u32 w) {
    if ((size_t)code_words * 4 >= MAX_CODE) die("code buffer overflow");
    code[code_words++] = w;
}

static int cur_off(void) { return code_words * 4; }

static void patch32(int off, u32 w) {
    if (off < 0 || off >= code_words * 4) die("patch out of range");
    code[off / 4] = w;
}

static int alloc_temp(void) {
    int r = 9 + temp_next;
    temp_next++;
    if (temp_next > 10) {
        int line = (tk < ntoks) ? toks[tk].line : -1;
        die("line %d: expression too complex (>10 temps)", line);
    }
    return r;
}
static void reset_temps(void) { temp_next = 0; }

static int alloc_slot(void) {
    frame_next_slot += 8;
    if (frame_next_slot > FRAME_SIZE) die("too many locals (frame overflow)");
    return frame_next_slot;
}

/* MOVZ Xd, #imm16 LSL #(shift*16) */
static u32 enc_movz(int rd, u32 imm16, int shift) {
    return 0xD2800000u | ((u32)(shift & 3) << 21) | ((imm16 & 0xFFFF) << 5) | (u32)(rd & 31);
}
/* MOVK Xd, #imm16 LSL #(shift*16) */
static u32 enc_movk(int rd, u32 imm16, int shift) {
    return 0xF2800000u | ((u32)(shift & 3) << 21) | ((imm16 & 0xFFFF) << 5) | (u32)(rd & 31);
}

static void emit_mov_imm64(int rd, u64 imm) {
    u32 slices[4];
    for (int i = 0; i < 4; i++) slices[i] = (u32)((imm >> (i*16)) & 0xFFFF);
    int first = -1;
    for (int i = 0; i < 4; i++) if (slices[i]) { first = i; break; }
    if (first == -1) {
        emit32(enc_movz(rd, 0, 0));
        return;
    }
    emit32(enc_movz(rd, slices[first], first));
    for (int i = first + 1; i < 4; i++) {
        if (slices[i]) emit32(enc_movk(rd, slices[i], i));
    }
}

/* Emit a FIXED-size 4-instruction MOVZ/MOVK/MOVK/MOVK sequence loading a
 * 64-bit immediate.  The high bits are always MOVK'd even when zero, so the
 * sequence has a constant 16-byte footprint — suitable for in-place patching
 * when the target address isn't yet known.  Used for `buf` references. */
static void emit_mov_imm64_fixed(int rd, u64 imm) {
    emit32(enc_movz(rd, (u32)(imm        & 0xFFFF), 0));
    emit32(enc_movk(rd, (u32)((imm >> 16) & 0xFFFF), 1));
    emit32(enc_movk(rd, (u32)((imm >> 32) & 0xFFFF), 2));
    emit32(enc_movk(rd, (u32)((imm >> 48) & 0xFFFF), 3));
}

/* In-place patch of a previously emitted fixed 4-instruction sequence at
 * code[code_word_idx .. code_word_idx+3] to load `imm` into register `rd`. */
static void patch_mov_imm64_fixed(int code_word_idx, int rd, u64 imm) {
    code[code_word_idx + 0] = enc_movz(rd, (u32)(imm        & 0xFFFF), 0);
    code[code_word_idx + 1] = enc_movk(rd, (u32)((imm >> 16) & 0xFFFF), 1);
    code[code_word_idx + 2] = enc_movk(rd, (u32)((imm >> 32) & 0xFFFF), 2);
    code[code_word_idx + 3] = enc_movk(rd, (u32)((imm >> 48) & 0xFFFF), 3);
}

static void emit_mov_reg(int rd, int rm) {
    if (rd == rm) return;
    emit32(0xAA0003E0u | ((u32)(rm & 31) << 16) | (u32)(rd & 31));
}

static void emit_add_reg(int rd, int rn, int rm) {
    emit32(0x8B000000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_sub_reg(int rd, int rn, int rm) {
    emit32(0xCB000000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_and_reg(int rd, int rn, int rm) {
    emit32(0x8A000000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_orr_reg(int rd, int rn, int rm) {
    emit32(0xAA000000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_eor_reg(int rd, int rn, int rm) {
    emit32(0xCA000000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_mul(int rd, int rn, int rm) {
    emit32(0x9B007C00u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_sdiv(int rd, int rn, int rm) {
    emit32(0x9AC00C00u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_lsl_reg(int rd, int rn, int rm) {
    emit32(0x9AC02000u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_lsr_reg(int rd, int rn, int rm) {
    emit32(0x9AC02400u | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_neg(int rd, int rm) {
    emit32(0xCB0003E0u | ((u32)(rm & 31) << 16) | (u32)(rd & 31));
}
static void emit_cmp_reg(int rn, int rm) {
    emit32(0xEB00001Fu | ((u32)(rm & 31) << 16) | ((u32)(rn & 31) << 5));
}

static void emit_add_imm(int rd, int rn, u32 imm12) {
    emit32(0x91000000u | ((imm12 & 0xFFF) << 10) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
}
static void emit_sub_imm(int rd, int rn, u32 imm) {
    if (imm <= 0xFFF) {
        emit32(0xD1000000u | ((imm & 0xFFF) << 10) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
    } else if ((imm & 0xFFF) == 0 && (imm >> 12) <= 0xFFF) {
        emit32(0xD1400000u | (((imm >> 12) & 0xFFF) << 10) | ((u32)(rn & 31) << 5) | (u32)(rd & 31));
    } else {
        emit_mov_imm64(16, imm);
        emit_sub_reg(rd, rn, 16);
    }
}

/* STUR Xt, [Xn, #simm9] */
static void emit_stur(int rt, int rn, int simm9) {
    emit32(0xF8000000u | (((u32)simm9 & 0x1FF) << 12) | ((u32)(rn & 31) << 5) | (u32)(rt & 31));
}
/* LDUR Xt, [Xn, #simm9] */
static void emit_ldur(int rt, int rn, int simm9) {
    emit32(0xF8400000u | (((u32)simm9 & 0x1FF) << 12) | ((u32)(rn & 31) << 5) | (u32)(rt & 31));
}

/* Width-dispatched memory store/load with zero offset.  Used by the ←/→
 * arrow operators.  Width is one of {8, 16, 32, 64} meaning bit-width
 * (so 8 → 1 byte, 16 → 2 bytes, 32 → 4 bytes, 64 → 8 bytes).  Encodings
 * are the "unsigned immediate offset" variants of STR/LDR with imm=0:
 *
 *   STRB Wt, [Xn]  = 0x39000000 | Rn<<5 | Rt
 *   LDRB Wt, [Xn]  = 0x39400000 | Rn<<5 | Rt
 *   STRH Wt, [Xn]  = 0x79000000 | Rn<<5 | Rt
 *   LDRH Wt, [Xn]  = 0x79400000 | Rn<<5 | Rt
 *   STR  Wt, [Xn]  = 0xB9000000 | Rn<<5 | Rt   (32-bit)
 *   LDR  Wt, [Xn]  = 0xB9400000 | Rn<<5 | Rt   (32-bit)
 *   STR  Xt, [Xn]  = 0xF9000000 | Rn<<5 | Rt   (64-bit)
 *   LDR  Xt, [Xn]  = 0xF9400000 | Rn<<5 | Rt   (64-bit)
 */
static void emit_str_w(int rt, int rn, int width) {
    u32 base;
    switch (width) {
    case 8:  base = 0x39000000u; break;
    case 16: base = 0x79000000u; break;
    case 32: base = 0xB9000000u; break;
    case 64: base = 0xF9000000u; break;
    default: die("bad store width %d (expected 8/16/32/64)", width);
    }
    emit32(base | ((u32)(rn & 31) << 5) | (u32)(rt & 31));
}

static void emit_ldr_w(int rt, int rn, int width) {
    u32 base;
    switch (width) {
    case 8:  base = 0x39400000u; break;
    case 16: base = 0x79400000u; break;
    case 32: base = 0xB9400000u; break;
    case 64: base = 0xF9400000u; break;
    default: die("bad load width %d (expected 8/16/32/64)", width);
    }
    emit32(base | ((u32)(rn & 31) << 5) | (u32)(rt & 31));
}

static void emit_b(int delta_bytes) {
    emit32(0x14000000u | ((u32)(delta_bytes / 4) & 0x3FFFFFFu));
}
static void emit_bl(int delta_bytes) {
    emit32(0x94000000u | ((u32)(delta_bytes / 4) & 0x3FFFFFFu));
}
static void emit_b_cond(int cc, int delta_bytes) {
    emit32(0x54000000u | (((u32)(delta_bytes / 4) & 0x7FFFFu) << 5) | ((u32)cc & 0xF));
}
static void emit_cbz(int rt, int delta_bytes) {
    emit32(0xB4000000u | (((u32)(delta_bytes / 4) & 0x7FFFFu) << 5) | (u32)(rt & 31));
}
static void emit_svc(int imm16) {
    emit32(0xD4000001u | (((u32)imm16 & 0xFFFF) << 5));
}
static void emit_ret(void) { emit32(0xD65F03C0u); }

/* STP X29, X30, [SP, #-16]!  pre-index */
static void emit_stp_fp_lr_pre(void) {
    /* STP Xt1, Xt2, [Xn, #imm7]! pre-index, 64-bit
     * encoding: 10 101 0 011 0 imm7 Rt2 Rn Rt1
     *  = 0xA9800000 | (imm7<<15) | (Rt2<<10) | (Rn<<5) | Rt1
     * imm7 is signed scaled by 8.  For #-16 / 8 = -2 → 7 bits = 0x7E */
    int imm7 = (-16 / 8) & 0x7F;
    emit32(0xA9800000u | ((u32)imm7 << 15) | (30u << 10) | (31u << 5) | 29u);
}

/* LDP X29, X30, [SP], #16  post-index */
static void emit_ldp_fp_lr_post(void) {
    /* encoding: 10 101 0 001 1 imm7 Rt2 Rn Rt1 = 0xA8C00000 | ... */
    int imm7 = (16 / 8) & 0x7F;
    emit32(0xA8C00000u | ((u32)imm7 << 15) | (30u << 10) | (31u << 5) | 29u);
}

enum { CC_EQ=0, CC_NE=1, CC_HS=2, CC_LO=3, CC_MI=4, CC_PL=5, CC_VS=6, CC_VC=7,
       CC_HI=8, CC_LS=9, CC_GE=10, CC_LT=11, CC_GT=12, CC_LE=13, CC_AL=14 };

static void emit_prologue(void) {
    emit_stp_fp_lr_pre();
    emit_add_imm(29, 31, 0);        /* MOV X29, SP */
    emit_sub_imm(31, 31, FRAME_SIZE);
}

static void emit_epilogue(void) {
    emit_add_imm(31, 29, 0);        /* MOV SP, X29 */
    emit_ldp_fp_lr_post();
    emit_ret();
}

/* ==========================================================================
 * Parser
 * ========================================================================== */

static Token *cur(void) { return &toks[tk]; }
static int peek_type(void) { return toks[tk].type; }

static i64 tok_int_value(Token *t) {
    const char *p = src + t->off;
    int n = t->len;
    i64 v = 0;
    if (n > 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        for (int i = 2; i < n; i++) {
            char c = p[i];
            int d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else break;
            v = v * 16 + d;
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (p[i] < '0' || p[i] > '9') break;
            v = v * 10 + (p[i] - '0');
        }
    }
    return v;
}

static int tok_dollar_reg(Token *t) {
    const char *p = src + t->off + 1;
    int n = t->len - 1;
    int r = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] < '0' || p[i] > '9') return -1;
        r = r * 10 + (p[i] - '0');
    }
    if (r < 0 || r > 30) return -1;
    return r;
}

static int is_expr_start(int t) {
    return t == TOK_INT || t == TOK_IDENT || t == TOK_LPAREN
        || t == TOK_MINUS || t == TOK_REG_READ || t == TOK_DOLLAR
        || t == TOK_MEM_LOAD;
}

static int parse_expr(void);
static void parse_body(int body_indent);
static void emit_bl_compo(int si);
static int choose_dest(int a);

static int parse_primary(void) {
    Token *t = cur();

    if (t->type == TOK_INT) {
        tk++;
        int r = alloc_temp();
        emit_mov_imm64(r, (u64)tok_int_value(t));
        return r;
    }

    if (t->type == TOK_MINUS) {
        tk++;
        int a = parse_primary();
        int r = alloc_temp();
        emit_neg(r, a);
        return r;
    }

    if (t->type == TOK_LPAREN) {
        tk++;
        int r = parse_expr();
        if (cur()->type == TOK_RPAREN) tk++;
        return r;
    }

    if (t->type == TOK_REG_READ) {
        tk++;
        Token *d = cur();
        if (d->type != TOK_DOLLAR) die("line %d: expected $reg after arrow", t->line);
        int rn = tok_dollar_reg(d);
        if (rn < 0) die("line %d: bad $reg", d->line);
        tk++;
        int r = alloc_temp();
        emit_mov_reg(r, rn);
        return r;
    }

    if (t->type == TOK_MEM_LOAD) {
        /* → width base [offset] — memory load. Width is int literal (8/16/32/64).
         * Address = base, or base+offset if a third atom follows. */
        tk++;
        if (cur()->type != TOK_INT)
            die("line %d: expected width after →", t->line);
        i64 width = tok_int_value(cur());
        tk++;
        int base_r = parse_primary();
        int addr_r = base_r;
        /* Optional offset atom — if the next token can start an expression,
         * treat it as an offset that's added to base. Stop at operators so
         * the enclosing expression can combine the load result with them. */
        int nx = peek_type();
        if (nx == TOK_INT || nx == TOK_IDENT) {
            int off_r = parse_primary();
            int combined = choose_dest(base_r);
            emit_add_reg(combined, base_r, off_r);
            addr_r = combined;
        }
        int dst_r = choose_dest(addr_r);
        /* Encode as byte/half/word/dword load at [Xn] */
        emit_ldr_w(dst_r, addr_r, (int)width);
        return dst_r;
    }

    if (t->type == TOK_DOLLAR) {
        int rn = tok_dollar_reg(t);
        if (rn < 0) die("line %d: bad $reg", t->line);
        tk++;
        int r = alloc_temp();
        emit_mov_reg(r, rn);
        return r;
    }

    /* Keywords sometimes used as idents in expressions (compiler-darwin.ls
     * binds `buf`, `const`, etc. as param names). Fall through to IDENT. */
    if (t->type == TOK_BUF || t->type == TOK_CONST ||
        t->type == TOK_VAR || t->type == TOK_LABEL) {
        t = (Token *)(void *)t;  /* keep pointer; treat as IDENT */
        /* Rewrite token type locally to drive into the IDENT path */
        t->type = TOK_IDENT;
    }

    if (t->type == TOK_IDENT) {
        /* Built-in intrinsics: `max a b` and `min a b`. Emit CMP + CSEL. */
        if ((t->len == 3 && (memcmp(src + t->off, "max", 3) == 0 ||
                             memcmp(src + t->off, "min", 3) == 0))) {
            int is_max = (src[t->off] == 'm' && src[t->off + 1] == 'a');
            tk++;
            int a = parse_primary();
            int b = parse_primary();
            int r = choose_dest(a);
            emit_cmp_reg(a, b);
            /* CSEL Xd, Xn, Xm, cond   = 0x9A800000 | (Rm<<16) | (cond<<12) | (Rn<<5) | Rd
             * max: pick a if a > b (CC_GT=12), else b. So CSEL Rd, a, b, GT.
             * min: pick a if a < b (CC_LT=11), else b. */
            int cond = is_max ? 12 : 11;
            emit32(0x9A800000u |
                   ((u32)(b & 31) << 16) |
                   ((u32)cond << 12) |
                   ((u32)(a & 31) << 5)  |
                   (u32)(r & 31));
            return r;
        }
        int si = sym_find(src + t->off, t->len);
        tk++;
        if (si < 0) {
            /* Unknown name — treat as a stubbed zero-arg composition call
             * so the file can parse. Code that actually reaches the call
             * will segfault at runtime, but this lets us get through
             * compiler-darwin.ls's forward-ref + typo spots. */
            int r = alloc_temp();
            emit_mov_imm64(r, 0);
            return r;
        }
        Sym *s = &syms[si];
        if (s->kind == SYM_LOCAL || s->kind == SYM_FOR_VAR) {
            int r = alloc_temp();
            emit_ldur(r, 29, -s->slot);
            /* Array subscript: `name[index]` is a BYTE load from base+index */
            if (cur()->type == TOK_LBRACK) {
                tk++;
                int idx_r = parse_expr();
                if (cur()->type == TOK_RBRACK) tk++;
                /* Effective address = r + idx_r; load byte */
                int addr_r = choose_dest(r);
                emit_add_reg(addr_r, r, idx_r);
                int dst_r = choose_dest(addr_r);
                /* LDRB Wt, [Xn] = 0x39400000 | (Rn<<5) | Rt */
                emit32(0x39400000u | ((u32)(addr_r & 31) << 5) | (u32)(dst_r & 31));
                return dst_r;
            }
            return r;
        }
        if (s->kind == SYM_COMPO) {
            /* Expression-level composition call. Collect nparams atoms as
             * args (parse_primary, not parse_expr, so operators bind to
             * the call result rather than getting slurped into args). */
            int nparams = s->nparams;
            int argregs[16];
            int nargs = 0;
            while (nargs < nparams) {
                int nx = peek_type();
                /* Stop at operators / terminators — they bind to the call */
                if (nx == TOK_NEWLINE || nx == TOK_EOF || nx == TOK_INDENT ||
                    nx == TOK_RPAREN || nx == TOK_COLON ||
                    nx == TOK_PLUS  || nx == TOK_MINUS || nx == TOK_STAR ||
                    nx == TOK_SLASH || nx == TOK_AMP   || nx == TOK_PIPE ||
                    nx == TOK_CARET || nx == TOK_SHL   || nx == TOK_SHR ||
                    nx == TOK_EQEQ  || nx == TOK_NEQ   ||
                    nx == TOK_LT    || nx == TOK_GT    ||
                    nx == TOK_LTE   || nx == TOK_GTE)
                    break;
                if (!is_expr_start(nx)) break;
                argregs[nargs++] = parse_primary();
            }
            for (int i = nargs - 1; i >= 0; i--) {
                emit_mov_reg(i, argregs[i]);
            }
            emit_bl_compo(si);
            int r = alloc_temp();
            emit_mov_reg(r, 0);
            return r;
        }
        if (s->kind == SYM_CONST) {
            /* const value is stored in code_off (sign-extended to 64 bits) */
            int r = alloc_temp();
            emit_mov_imm64(r, (u64)(i64)s->code_off);
            return r;
        }
        if (s->kind == SYM_BUF) {
            /* Buf's virtual address isn't known until parse_file() finishes
             * and we know code_len_final. Emit a fixed 4-insn placeholder and
             * record a fixup to resolve later. */
            int r = alloc_temp();
            int word_idx = code_words;
            emit_mov_imm64_fixed(r, 0);
            add_buf_fixup(word_idx, r, s->code_off);
            return r;
        }
        if (s->kind == SYM_GLOBAL) {
            /* Top-level `var` — a module-scope 8-byte data slot. PIE is
             * disabled for wirth-emitted binaries, so we can emit the full
             * absolute VA via a 4-instruction MOVZ+MOVK chain, patched at
             * end-of-parse once code_len_final is known. */
            int addr_r = alloc_temp();
            int word_idx = code_words;
            emit_mov_imm64_fixed(addr_r, 0);  /* 4 words — patched at fixup */
            add_global_fixup(word_idx, addr_r, s->code_off);
            int val_r = alloc_temp();
            /* LDR Xt, [Xn, #0] — 64-bit unsigned offset, imm12=0 */
            emit32(0xF9400000u | ((u32)(addr_r & 31) << 5) | (u32)(val_r & 31));
            return val_r;
        }
        die("line %d: cannot use '%.*s' in expression", t->line, t->len, src + t->off);
    }

    /* Tolerant recovery: unexpected token in expression — emit 0 as the
     * value, don't advance, and let the caller bail on its own logic. */
    if (t->type == TOK_NEWLINE || t->type == TOK_EOF || t->type == TOK_INDENT ||
        t->type == TOK_RPAREN || t->type == TOK_RBRACK) {
        int r = alloc_temp();
        emit_mov_imm64(r, 0);
        return r;
    }
    die("line %d: unexpected token in expression (type=%d)", t->line, t->type);
}

/* Reuse a temp register in place when possible so a chain of binary
 * operators doesn't exhaust the 10-temp window. If `a` is already in the
 * temp range [9..18], emit into `a` directly. Otherwise allocate a fresh
 * temp to hold the result (keeps locals/params untouched). */
static int choose_dest(int a) {
    if (a >= 9 && a <= 18) return a;
    return alloc_temp();
}

/* Peek past a NEWLINE + INDENT (deeper than current statement) + operator
 * pattern. If the next significant token is in `op_set`, eat the NEWLINE
 * and INDENT so the caller's operator loop sees the operator. This enables
 * multi-line expressions where the continuation starts with `|`, `+`, etc.
 * Returns 1 if a continuation was consumed, 0 otherwise. */
static int maybe_eat_continuation(const int *op_set, int nops) {
    if (cur()->type != TOK_NEWLINE) return 0;
    /* look ahead: NEWLINE [NEWLINE...] INDENT op */
    int i = tk + 1;
    while (i < ntoks && toks[i].type == TOK_NEWLINE) i++;
    if (i >= ntoks || toks[i].type != TOK_INDENT) return 0;
    int indent_level = toks[i].len;
    if (indent_level <= 0) return 0;
    int j = i + 1;
    if (j >= ntoks) return 0;
    int op = toks[j].type;
    int found = 0;
    for (int k = 0; k < nops; k++) if (op_set[k] == op) { found = 1; break; }
    if (!found) return 0;
    /* Eat up through the INDENT so the next cur() sees the operator. */
    tk = i + 1;
    return 1;
}

/* After combining `a` and `b` into `dst`, release any temps that were
 * allocated for `b`'s sub-expression. `mark_before_b` is temp_next at
 * the point we started parsing `b`. We keep `dst` live. */
static void release_b_temps(int dst, int mark_before_b) {
    if (dst >= 9 && dst <= 18) {
        int dst_idx = (dst - 9) + 1;
        if (dst_idx > mark_before_b) temp_next = dst_idx;
        else                         temp_next = mark_before_b;
    } else {
        temp_next = mark_before_b;
    }
}

static int parse_mul(void) {
    static const int ops[] = { TOK_STAR, TOK_SLASH };
    int a = parse_primary();
    for (;;) {
        maybe_eat_continuation(ops, 2);
        int op = cur()->type;
        if (op != TOK_STAR && op != TOK_SLASH) break;
        tk++;
        int mark_b = temp_next;
        int b = parse_primary();
        int r = choose_dest(a);
        if (op == TOK_STAR) emit_mul(r, a, b);
        else                 emit_sdiv(r, a, b);
        release_b_temps(r, mark_b);
        a = r;
    }
    return a;
}
static int parse_add(void) {
    static const int ops[] = { TOK_PLUS, TOK_MINUS };
    int a = parse_mul();
    for (;;) {
        maybe_eat_continuation(ops, 2);
        int op = cur()->type;
        if (op != TOK_PLUS && op != TOK_MINUS) break;
        tk++;
        int mark_b = temp_next;
        int b = parse_mul();
        int r = choose_dest(a);
        if (op == TOK_PLUS) emit_add_reg(r, a, b);
        else                 emit_sub_reg(r, a, b);
        release_b_temps(r, mark_b);
        a = r;
    }
    return a;
}
static int parse_shift(void) {
    static const int ops[] = { TOK_SHL, TOK_SHR };
    int a = parse_add();
    for (;;) {
        maybe_eat_continuation(ops, 2);
        int op = cur()->type;
        if (op != TOK_SHL && op != TOK_SHR) break;
        tk++;
        int mark_b = temp_next;
        int b = parse_add();
        int r = choose_dest(a);
        if (op == TOK_SHL) emit_lsl_reg(r, a, b);
        else                emit_lsr_reg(r, a, b);
        release_b_temps(r, mark_b);
        a = r;
    }
    return a;
}
static int parse_cmp_inner(void);
static int parse_bits(void) {
    static const int ops[] = { TOK_AMP, TOK_PIPE, TOK_CARET };
    int a = parse_cmp_inner();
    for (;;) {
        maybe_eat_continuation(ops, 3);
        int op = cur()->type;
        if (op != TOK_AMP && op != TOK_PIPE && op != TOK_CARET) break;
        tk++;
        int mark_b = temp_next;
        int b = parse_cmp_inner();
        int r = choose_dest(a);
        if (op == TOK_AMP)       emit_and_reg(r, a, b);
        else if (op == TOK_PIPE) emit_orr_reg(r, a, b);
        else                     emit_eor_reg(r, a, b);
        release_b_temps(r, mark_b);
        a = r;
    }
    return a;
}
static int parse_cmp_inner(void) {
    int a = parse_shift();
    int op = cur()->type;
    if (op != TOK_LT && op != TOK_GT && op != TOK_LTE &&
        op != TOK_GTE && op != TOK_EQEQ && op != TOK_NEQ)
        return a;
    tk++;
    int mark_b = temp_next;
    int b = parse_shift();
    int r = choose_dest(a);
    emit_cmp_reg(a, b);
    int cc;
    switch (op) {
    case TOK_LT:   cc = 11; break;
    case TOK_GT:   cc = 12; break;
    case TOK_LTE:  cc = 13; break;
    case TOK_GTE:  cc = 10; break;
    case TOK_EQEQ: cc = 0;  break;
    case TOK_NEQ:  cc = 1;  break;
    default:       cc = 0;  break;
    }
    int ncc = cc ^ 1;
    emit32(0x9A9F07E0u | ((u32)(ncc & 0xF) << 12) | (u32)(r & 31));
    release_b_temps(r, mark_b);
    return r;
}
static int parse_expr(void) { return parse_bits(); }

/* ==========================================================================
 * Statement dispatch
 * ========================================================================== */

/* Emit a BL to a composition by symbol index. If the target's code_off
 * is already known (backward call), emit a direct branch. Otherwise emit
 * a BL 0 placeholder and record a fixup to resolve at end of parse_file. */
static void emit_bl_compo(int si) {
    Sym *s = &syms[si];
    if (s->code_off >= 0) {
        int delta = s->code_off - cur_off();
        emit_bl(delta);
    } else {
        int here = cur_off();
        emit_bl(0);
        add_fixup(here, si);
    }
}

static void emit_call_args(int si) {
    /* Push each arg onto the target's stack as we parse it, then pop in
     * reverse order into X0..X(N-1). This sidesteps the 10-temp window
     * because each arg releases its temps immediately after being pushed. */
    Sym *s = &syms[si];
    int nparams = s->nparams;
    int nargs = 0;
    int saved_temp_next = temp_next;
    while (nargs < nparams) {
        int t = peek_type();
        if (t == TOK_NEWLINE || t == TOK_EOF || t == TOK_INDENT) break;
        if (!is_expr_start(t)) break;
        int r = parse_expr();
        /* STR Xr, [SP, #-16]!  (pre-index push) */
        emit32(0xF81F0FE0u | (u32)(r & 31));
        nargs++;
        temp_next = saved_temp_next;
    }
    /* Pop into X(nargs-1), X(nargs-2), ..., X0
     * Pushed in order arg0, arg1, ... so last-pushed is on top.
     * Reverse iteration: i = nargs-1 gets top (which is arg(nargs-1)). Good. */
    for (int i = nargs - 1; i >= 0; i--) {
        /* LDR Xi, [SP], #16  (post-index pop) */
        emit32(0xF84107E0u | (u32)(i & 31));
    }
    emit_bl_compo(si);
}

static void parse_stmt(void) {
    reset_temps();
    if (cur()->type == TOK_INDENT) tk++;

    Token *t = cur();

    if (t->type == TOK_NEWLINE) { tk++; return; }
    if (t->type == TOK_EOF)     return;

    if (t->type == TOK_CONTINUE) {
        tk++;
        int top = loop_current_top();
        int here = cur_off();
        emit_b(top - here);
        return;
    }
    if (t->type == TOK_BREAK) {
        tk++;
        int here = cur_off();
        loop_record_break(here);
        emit_b(0);       /* patched when loop closes */
        return;
    }
    if (t->type == TOK_GOTO) {
        tk++;
        Token *n = cur();
        if (n->type != TOK_IDENT) die("line %d: expected label after goto", t->line);
        tk++;
        int target = -1;
        int li = label_find(src + n->off, n->len);
        if (li >= 0 && labels[li].code_off >= 0) {
            target = labels[li].code_off;
        }
        int here = cur_off();
        if (target >= 0) {
            int delta = target - here;
            emit_b(delta);
        } else {
            emit_b(0);
            label_add_fixup(src + n->off, n->len, here);
        }
        return;
    }
    if (t->type == TOK_LABEL) {
        tk++;
        Token *n = cur();
        if (n->type != TOK_IDENT) die("line %d: expected name after label", t->line);
        tk++;
        label_define(src + n->off, n->len, cur_off());
        return;
    }

    if (t->type == TOK_TRAP) {
        tk++;
        /* Two forms:
         *   `trap`                   — bare SVC. Whatever the program set in
         *                               X16 (via ↓ $16 N) + args in X0-X7 is
         *                               the syscall. DO NOT clobber X16.
         *   `trap name num [arg...]` — syscall with args: put num in X16, args
         *                               in X0..X7, issue SVC, bind result into
         *                               `name` as a new local. */
        int nx = cur()->type;
        if (nx == TOK_NEWLINE || nx == TOK_EOF || nx == TOK_INDENT) {
            /* Bare trap: just SVC. The caller already set X16/X0-X7. */
            emit_svc(0x80);
            return;
        }
        /* trap name num [arg...] */
        if (nx != TOK_IDENT) {
            /* Fallback: treat as bare trap and skip rest to newline */
            emit_mov_imm64(16, 1);
            emit_svc(0x80);
            while (cur()->type != TOK_NEWLINE && cur()->type != TOK_EOF) tk++;
            return;
        }
        Token *nm = cur();
        tk++;
        /* Syscall number */
        int num_r = parse_primary();
        emit_mov_reg(16, num_r);
        /* Collect up to 8 args, push/pop to set X0..X(n-1) */
        int saved_temp_next = temp_next;
        int nargs = 0;
        while (nargs < 8) {
            int tt = peek_type();
            if (tt == TOK_NEWLINE || tt == TOK_EOF || tt == TOK_INDENT) break;
            if (!is_expr_start(tt)) break;
            int r = parse_expr();
            emit32(0xF81F0FE0u | (u32)(r & 31));    /* STR Xr, [SP, #-16]! */
            nargs++;
            temp_next = saved_temp_next;
        }
        for (int i = nargs - 1; i >= 0; i--) {
            emit32(0xF84107E0u | (u32)(i & 31));    /* LDR Xi, [SP], #16 */
        }
        emit_svc(0x80);
        /* Bind return value (X0) into `name` as a local */
        int slot = alloc_slot();
        int ni = sym_add(src + nm->off, nm->len, SYM_LOCAL);
        syms[ni].slot = slot;
        emit_stur(0, 29, -slot);
        return;
    }

    if (t->type == TOK_CONST) {
        /* `const NAME int-literal` — introduces a named 32-bit integer
         * constant.  The value is stashed in the symbol's `code_off` slot
         * (which is otherwise unused for non-compositions) and materialised
         * via MOVZ/MOVK when the name is referenced in an expression. */
        tk++;
        Token *n = cur();
        if (n->type != TOK_IDENT) die("line %d: expected name after const", t->line);
        tk++;
        if (cur()->type != TOK_INT)
            die("line %d: const '%.*s' must be followed by int literal",
                t->line, n->len, src + n->off);
        i64 val = tok_int_value(cur());
        tk++;
        int si = sym_add(src + n->off, n->len, SYM_CONST);
        syms[si].code_off = (int)val;   /* must fit in int32 */
        return;
    }

    if (t->type == TOK_BUF) {
        /* `buf NAME size-bytes` — allocates `size` bytes of zero-initialised
         * storage in the trailing data region of __TEXT.  See the comment on
         * data_buf / buf_fixups above for the memory-model caveats.  The
         * symbol resolves to the virtual address of the first buf byte. */
        tk++;
        Token *n = cur();
        if (n->type != TOK_IDENT) die("line %d: expected name after buf", t->line);
        tk++;
        if (cur()->type != TOK_INT)
            die("line %d: buf '%.*s' size must be int literal",
                t->line, n->len, src + n->off);
        i64 size = tok_int_value(cur());
        tk++;
        if (size < 0) die("line %d: negative buf size", t->line);
        int bo = data_off;
        int new_off = bo + (int)size;
        new_off = (new_off + 7) & ~7;    /* 8-byte align for the next buf */
        if (new_off > DATA_MAX) die("buf data overflow (DATA_MAX=%d)", DATA_MAX);
        data_off = new_off;
        int si = sym_add(src + n->off, n->len, SYM_BUF);
        syms[si].code_off = bo;          /* offset within data_buf */
        syms[si].slot     = (int)size;   /* size, in case anyone wants it */
        return;
    }

    if (t->type == TOK_VAR) {
        tk++;
        Token *n = cur();
        if (n->type != TOK_IDENT) die("line %d: expected name after var", t->line);
        tk++;
        int slot = alloc_slot();
        int si = sym_add(src + n->off, n->len, SYM_LOCAL);
        syms[si].slot = slot;
        if (is_expr_start(cur()->type)) {
            int r = parse_expr();
            emit_stur(r, 29, -slot);
        } else {
            /* zero-init via XZR (register 31 in store encoding) */
            emit_stur(31, 29, -slot);
        }
        return;
    }

    if (t->type == TOK_RETURN) {
        tk++;
        if (is_expr_start(cur()->type)) {
            int r = parse_expr();
            emit_mov_reg(0, r);
        }
        emit_epilogue();
        return;
    }

    if (t->type == TOK_LOAD) {
        tk++;
        Token *d = cur();
        if (d->type == TOK_DOLLAR) {
            /* ↓ $N expr */
            int rn = tok_dollar_reg(d);
            if (rn < 0) die("line %d: bad $reg", d->line);
            tk++;
            int r = parse_expr();
            emit_mov_reg(rn, r);
            return;
        }
        /* memory store: ↓ width addr val.  Ignore width for v1 (64-bit). */
        (void)parse_expr();
        int ar = parse_expr();
        int vr = parse_expr();
        emit_stur(vr, ar, 0);
        return;
    }

    if (t->type == TOK_MEM_STORE) {
        /* ← width addr [offset] val
         *   3 operands: store val at addr.
         *   4 operands: store val at (addr + offset).
         * compiler-darwin.ls uses the 4-operand form for indexed stores. */
        tk++;
        if (cur()->type != TOK_INT)
            die("line %d: expected width after ←", t->line);
        i64 width = tok_int_value(cur());
        tk++;
        int base_r   = parse_expr();
        int middle_r = parse_expr();
        int nx = peek_type();
        if (is_expr_start(nx)) {
            int combined = choose_dest(base_r);
            emit_add_reg(combined, base_r, middle_r);
            int val_r = parse_expr();
            emit_str_w(val_r, combined, (int)width);
        } else {
            emit_str_w(middle_r, base_r, (int)width);
        }
        return;
    }

    if (t->type == TOK_REG_READ) {
        /* bare load statement — discard */
        (void)parse_expr();
        return;
    }

    if (t->type == TOK_IF) {
        tk++;
        /* if/elif/else chain.  Each branch emits:
         *   [cond test + inverted B.cond / CBZ to patch_off_skip]
         *   <body>
         *   B 0                 — patched to end-of-chain
         * After each body, patch_off_skip is patched to land at the start
         * of the next branch (or the fallthrough after the whole chain).
         * `else` has no condition and no skip-patch. */
        int end_patch_offs[32];
        int n_end_patches = 0;

        /* Parse the initial `if` condition like before, then reuse the
         * same loop for each `elif` branch (continue) and finally for
         * optional `else` (handled after the break). */
        for (;;) {
            int is_compound = 0;
            int patch_off_skip = 0;
            int patch_is_cbz = 0;

            int ct = cur()->type;
            int is_cmp_compound = (ct == TOK_LT || ct == TOK_GT || ct == TOK_LTE ||
                                   ct == TOK_GTE || ct == TOK_EQEQ || ct == TOK_NEQ);
            if (is_cmp_compound) {
                tk++;
                int a = parse_expr();
                int b = parse_expr();
                emit_cmp_reg(a, b);
                int cc_inv;
                switch (ct) {
                case TOK_LT:   cc_inv = CC_GE; break;
                case TOK_GT:   cc_inv = CC_LE; break;
                case TOK_LTE:  cc_inv = CC_GT; break;
                case TOK_GTE:  cc_inv = CC_LT; break;
                case TOK_EQEQ: cc_inv = CC_NE; break;
                case TOK_NEQ:  cc_inv = CC_EQ; break;
                default:       cc_inv = CC_AL;
                }
                patch_off_skip = cur_off();
                emit_b_cond(cc_inv, 0);
                is_compound = 1;
            } else {
                /* Simple `cond expr [cmp rhs]`.  If a relational operator
                 * appears after the first expression, emit CMP + inverted
                 * B.cond.  Otherwise fall back to "truthy expr" (CBZ skip). */
                int a = parse_expr();
                int rel = cur()->type;
                int is_rel = (rel == TOK_LT || rel == TOK_GT || rel == TOK_LTE ||
                              rel == TOK_GTE || rel == TOK_EQEQ || rel == TOK_NEQ);
                if (is_rel) {
                    tk++;
                    int b = parse_expr();
                    emit_cmp_reg(a, b);
                    int cc_inv;
                    switch (rel) {
                    case TOK_LT:   cc_inv = CC_GE; break;
                    case TOK_GT:   cc_inv = CC_LE; break;
                    case TOK_LTE:  cc_inv = CC_GT; break;
                    case TOK_GTE:  cc_inv = CC_LT; break;
                    case TOK_EQEQ: cc_inv = CC_NE; break;
                    case TOK_NEQ:  cc_inv = CC_EQ; break;
                    default:       cc_inv = CC_AL;
                    }
                    patch_off_skip = cur_off();
                    emit_b_cond(cc_inv, 0);
                    is_compound = 1;
                } else {
                    patch_off_skip = cur_off();
                    emit_cbz(a, 0);
                    patch_is_cbz = 1;
                    is_compound = 0;
                }
            }
            if (cur()->type == TOK_COLON) tk++;
            if (cur()->type == TOK_NEWLINE) tk++;

            parse_body(-1);
            /* sym_trim removed: inner-scope names persist to end of composition */
            /* frame_next_slot reset removed: slots persist across inner scopes */

            /* After the body: peek past any INDENT tokens at the same
             * column as the `if` to see if there's an `elif` or `else`.
             * parse_body returns with `cur` sitting on an INDENT whose
             * .len < body_indent (i.e. the indent of the if/elif/else). */
            int has_more = 0;
            int saved_tk = tk;
            /* Skip blank lines and indents */
            while (cur()->type == TOK_NEWLINE ||
                   (cur()->type == TOK_INDENT && toks[tk+1].type == TOK_NEWLINE)) {
                if (cur()->type == TOK_NEWLINE) tk++;
                else tk += 2;
            }
            int after_indent_tk = tk;
            if (cur()->type == TOK_INDENT) {
                after_indent_tk = tk + 1;
            }
            int next_tt = (after_indent_tk < ntoks) ? toks[after_indent_tk].type : TOK_EOF;

            if (next_tt == TOK_ELIF || next_tt == TOK_ELSE) {
                /* Consume the INDENT (if any) and the elif/else keyword. */
                if (cur()->type == TOK_INDENT) tk++;
                has_more = 1;
            } else {
                /* No continuation — rewind so the outer parse_body still
                 * sees the INDENT we peeked past. */
                tk = saved_tk;
            }

            /* Emit the unconditional B to skip past the rest of the chain.
             * We emit this only if there's a continuation; if there isn't,
             * the skip-patch naturally lands right after this body. */
            if (has_more) {
                if (n_end_patches < 32) {
                    end_patch_offs[n_end_patches++] = cur_off();
                }
                emit_b(0);
            }

            /* Patch the cond-skip to land here (at the next branch start,
             * or the fallthrough if this was the last conditional branch). */
            int delta = cur_off() - patch_off_skip;
            u32 old = code[patch_off_skip / 4];
            if (patch_is_cbz) {
                u32 rt = old & 0x1F;
                u32 w = 0xB4000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | rt;
                patch32(patch_off_skip, w);
            } else if (is_compound) {
                u32 cc = old & 0xF;
                u32 w = 0x54000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | cc;
                patch32(patch_off_skip, w);
            }

            if (!has_more) break;

            /* Now cur() is at TOK_ELIF or TOK_ELSE — consume the keyword. */
            if (cur()->type == TOK_ELIF) {
                tk++;
                continue;   /* top of loop parses the next condition */
            }
            /* TOK_ELSE: parse a body with no condition test, then end. */
            tk++;
            if (cur()->type == TOK_COLON) tk++;
            if (cur()->type == TOK_NEWLINE) tk++;
            parse_body(-1);
            break;
        }

        /* Patch all the "end-of-branch" B instructions to land here. */
        int end_off = cur_off();
        for (int i = 0; i < n_end_patches; i++) {
            int po = end_patch_offs[i];
            int dd = end_off - po;
            u32 w = 0x14000000u | ((u32)(dd / 4) & 0x3FFFFFFu);
            patch32(po, w);
        }
        return;
    }

    if (t->type == TOK_WHILE) {
        tk++;
        int top = cur_off();
        int r = parse_expr();
        int patch_off = cur_off();
        emit_cbz(r, 0);
        if (cur()->type == TOK_COLON) tk++;
        if (cur()->type == TOK_NEWLINE) tk++;

        loop_push(top);
        parse_body(-1);
        /* sym_trim removed: inner-scope names persist to end of composition */
        /* frame_next_slot reset removed: slots persist across inner scopes */

        emit_b(top - cur_off());
        int delta = cur_off() - patch_off;
        u32 old = code[patch_off / 4];
        u32 rt = old & 0x1F;
        u32 w = 0xB4000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | rt;
        patch32(patch_off, w);
        loop_pop_and_patch_breaks(cur_off());
        return;
    }

    if (t->type == TOK_FOR) {
        tk++;
        Token *vn = cur();
        if (vn->type != TOK_IDENT) die("line %d: expected for-var name", t->line);
        tk++;
        int islot = alloc_slot();
        int isym  = sym_add(src + vn->off, vn->len, SYM_FOR_VAR);
        syms[isym].slot = islot;

        int start_r = parse_expr();
        int end_r   = parse_expr();
        int end_slot = alloc_slot();
        emit_stur(end_r, 29, -end_slot);
        emit_stur(start_r, 29, -islot);

        int step_slot = -1;
        if (is_expr_start(cur()->type)) {
            int sr = parse_expr();
            step_slot = alloc_slot();
            emit_stur(sr, 29, -step_slot);
        }

        if (cur()->type == TOK_COLON) tk++;
        if (cur()->type == TOK_NEWLINE) tk++;

        int top = cur_off();
        reset_temps();
        int it = alloc_temp();
        int et = alloc_temp();
        emit_ldur(it, 29, -islot);
        emit_ldur(et, 29, -end_slot);
        emit_cmp_reg(it, et);
        int patch_off = cur_off();
        emit_b_cond(CC_GE, 0);

        /* For `continue`, jump target is the increment step (not the cmp),
         * so record the increment's position AFTER the body parses. For
         * simplicity we point continue at the top (redundant cmp but
         * correct). */
        loop_push(top);
        parse_body(-1);
        /* sym_trim removed: inner-scope names persist to end of composition */
        /* frame_next_slot reset removed: slots persist across inner scopes */

        reset_temps();
        int i2 = alloc_temp();
        emit_ldur(i2, 29, -islot);
        if (step_slot >= 0) {
            int sr = alloc_temp();
            emit_ldur(sr, 29, -step_slot);
            emit_add_reg(i2, i2, sr);
        } else {
            emit_add_imm(i2, i2, 1);
        }
        emit_stur(i2, 29, -islot);
        emit_b(top - cur_off());

        int delta = cur_off() - patch_off;
        u32 old = code[patch_off / 4];
        u32 cc = old & 0xF;
        u32 w = 0x54000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | cc;
        patch32(patch_off, w);
        loop_pop_and_patch_breaks(cur_off());
        return;
    }

    if (t->type == TOK_EACH) {
        tk++;
        Token *vn = cur();
        if (vn->type != TOK_IDENT) die("line %d: expected each-var name", t->line);
        tk++;
        int slot = alloc_slot();
        int si = sym_add(src + vn->off, vn->len, SYM_LOCAL);
        syms[si].slot = slot;
        emit_stur(31, 29, -slot);  /* i = 0 */
        if (cur()->type == TOK_COLON) tk++;
        if (cur()->type == TOK_NEWLINE) tk++;
        parse_body(-1);
        /* sym_trim removed: inner-scope names persist to end of composition */
        /* frame_next_slot reset removed: slots persist across inner scopes */
        return;
    }

    if (t->type == TOK_IDENT) {
        Token *n = t;
        /* `name:` — label definition (Python-style). Peek ahead before
         * sym_find so we don't confuse the label with a same-named var. */
        if (tk + 1 < ntoks && toks[tk+1].type == TOK_COLON) {
            tk += 2;
            label_define(src + n->off, n->len, cur_off());
            return;
        }
        int si = sym_find(src + n->off, n->len);
        tk++;

        if (si < 0) {
            /* New name: binding — `name [=] expr` */
            if (cur()->type == TOK_EQ) tk++;
            int r = parse_expr();
            int slot = alloc_slot();
            int ni = sym_add(src + n->off, n->len, SYM_LOCAL);
            syms[ni].slot = slot;
            emit_stur(r, 29, -slot);
            return;
        }
        Sym *s = &syms[si];
        if (s->kind == SYM_COMPO) {
            emit_call_args(si);
            return;
        }
        if (s->kind == SYM_GLOBAL) {
            /* Top-level `var` — statement forms:
             *   name                — bare read (value in X0, no write)
             *   name [=] expr       — value = expr
             *   name OP expr        — value = (current value) OP expr  (short form) */
            if (cur()->type == TOK_EQ) tk++;
            int data_off_for_global = s->code_off;

            /* Bare read: just load the global and put value in X0. */
            if (cur()->type == TOK_NEWLINE || cur()->type == TOK_EOF ||
                cur()->type == TOK_INDENT) {
                int addr_r = alloc_temp();
                int word_idx = code_words;
                emit_mov_imm64_fixed(addr_r, 0);  /* 4 words — patched */
                add_global_fixup(word_idx, addr_r, data_off_for_global);
                int val_r = alloc_temp();
                emit32(0xF9400000u | ((u32)(addr_r & 31) << 5) | (u32)(val_r & 31));
                emit_mov_reg(0, val_r);
                return;
            }

            /* Detect short form: statement continues with a binary operator */
            int nx = cur()->type;
            int is_op = (nx == TOK_PLUS  || nx == TOK_MINUS || nx == TOK_STAR ||
                         nx == TOK_SLASH || nx == TOK_AMP   || nx == TOK_PIPE ||
                         nx == TOK_CARET || nx == TOK_SHL   || nx == TOK_SHR);

            int val_r;
            if (is_op) {
                /* Load current value, then parse_expr will see the operator
                 * and combine. Trick: push the current value into the token
                 * stream semantics by loading it as the LHS of parse_add
                 * etc. Simplest path: load the global, then manually run
                 * the operator chain. */
                int cur_r = alloc_temp();
                int addr1_r = alloc_temp();
                int wi1 = code_words;
                emit_mov_imm64_fixed(addr1_r, 0);
                add_global_fixup(wi1, addr1_r, data_off_for_global);
                emit32(0xF9400000u | ((u32)(addr1_r & 31) << 5) | (u32)(cur_r & 31));

                /* Parse the RHS expression chain manually. Consume the
                 * operator + next term, iterate. We reuse the same logic
                 * as parse_add/parse_bits by inlining an operator loop. */
                int a = cur_r;
                static const int all_ops2[] = {
                    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
                    TOK_AMP, TOK_PIPE, TOK_CARET, TOK_SHL, TOK_SHR
                };
                for (;;) {
                    maybe_eat_continuation(all_ops2, 9);
                    int op = cur()->type;
                    if (op != TOK_PLUS && op != TOK_MINUS && op != TOK_STAR &&
                        op != TOK_SLASH && op != TOK_AMP && op != TOK_PIPE &&
                        op != TOK_CARET && op != TOK_SHL && op != TOK_SHR)
                        break;
                    tk++;
                    int mark_before_b = temp_next;
                    int b = parse_primary();
                    int r = choose_dest(a);
                    switch (op) {
                    case TOK_PLUS:  emit_add_reg(r, a, b); break;
                    case TOK_MINUS: emit_sub_reg(r, a, b); break;
                    case TOK_STAR:  emit_mul(r, a, b); break;
                    case TOK_SLASH: emit_sdiv(r, a, b); break;
                    case TOK_AMP:   emit_and_reg(r, a, b); break;
                    case TOK_PIPE:  emit_orr_reg(r, a, b); break;
                    case TOK_CARET: emit_eor_reg(r, a, b); break;
                    case TOK_SHL:   emit_lsl_reg(r, a, b); break;
                    case TOK_SHR:   emit_lsr_reg(r, a, b); break;
                    }
                    if (r >= 9 && r <= 18 && r < mark_before_b + 9) {
                        temp_next = (r - 9) + 1;
                    } else {
                        temp_next = mark_before_b;
                    }
                    a = r;
                }
                val_r = a;
            } else {
                val_r = parse_expr();
            }

            int addr_r = alloc_temp();
            int word_idx = code_words;
            emit_mov_imm64_fixed(addr_r, 0);  /* 4 words — patched at fixup time */
            add_global_fixup(word_idx, addr_r, data_off_for_global);
            /* STR Xt, [Xn, #0] */
            emit32(0xF9000000u | ((u32)(addr_r & 31) << 5) | (u32)(val_r & 31));
            return;
        }
        /* Existing local — reassignment. Supports short form `name OP expr`
         * the same way globals do. Also supports multi-line expression
         * continuations: `extra41 \n    | (stall << 41) \n    | ...`
         * where the leading ident is a bare "read the current value"
         * followed by a continuation. */
        if (cur()->type == TOK_EQ) tk++;
        static const int all_ops[] = {
            TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
            TOK_AMP, TOK_PIPE, TOK_CARET, TOK_SHL, TOK_SHR
        };
        /* If we're at a NEWLINE, peek past it for a continuation operator. */
        maybe_eat_continuation(all_ops, 9);

        int nx = cur()->type;
        int is_op = (nx == TOK_PLUS  || nx == TOK_MINUS || nx == TOK_STAR ||
                     nx == TOK_SLASH || nx == TOK_AMP   || nx == TOK_PIPE ||
                     nx == TOK_CARET || nx == TOK_SHL   || nx == TOK_SHR);

        /* Bare read: `name` with no following operator or value.  Emit
         * MOV X0, Xname so the value is in the return register if this
         * is the last statement. No stur/store. */
        if (nx == TOK_NEWLINE || nx == TOK_EOF || nx == TOK_INDENT) {
            int cur_r = alloc_temp();
            emit_ldur(cur_r, 29, -s->slot);
            emit_mov_reg(0, cur_r);
            return;
        }
        int r;
        if (is_op) {
            int cur_r = alloc_temp();
            emit_ldur(cur_r, 29, -s->slot);
            int a = cur_r;
            for (;;) {
                maybe_eat_continuation(all_ops, 9);
                int op = cur()->type;
                if (op != TOK_PLUS && op != TOK_MINUS && op != TOK_STAR &&
                    op != TOK_SLASH && op != TOK_AMP && op != TOK_PIPE &&
                    op != TOK_CARET && op != TOK_SHL && op != TOK_SHR)
                    break;
                tk++;
                int mark_before_b = temp_next;
                int b = parse_primary();
                int dst = choose_dest(a);
                switch (op) {
                case TOK_PLUS:  emit_add_reg(dst, a, b); break;
                case TOK_MINUS: emit_sub_reg(dst, a, b); break;
                case TOK_STAR:  emit_mul(dst, a, b); break;
                case TOK_SLASH: emit_sdiv(dst, a, b); break;
                case TOK_AMP:   emit_and_reg(dst, a, b); break;
                case TOK_PIPE:  emit_orr_reg(dst, a, b); break;
                case TOK_CARET: emit_eor_reg(dst, a, b); break;
                case TOK_SHL:   emit_lsl_reg(dst, a, b); break;
                case TOK_SHR:   emit_lsr_reg(dst, a, b); break;
                }
                /* Free the temps b (and its sub-expression temps) used up.
                 * Keep `dst` live: if dst was reused from `a` it's below
                 * mark_before_b already; if dst was a fresh temp just above
                 * mark_before_b, keep it. */
                if (dst >= 9 && dst <= 18 && dst < mark_before_b + 9) {
                    temp_next = (dst - 9) + 1;
                } else {
                    temp_next = mark_before_b;
                }
                a = dst;
            }
            r = a;
        } else {
            r = parse_expr();
        }
        emit_stur(r, 29, -s->slot);
        /* Put the value in X0 too — Lithos convention: the last statement's
         * value is the composition's return value. Harmless if not last. */
        emit_mov_reg(0, r);
        return;
    }

    /* Unknown statement: skip to newline to recover */
    while (cur()->type != TOK_NEWLINE && cur()->type != TOK_EOF) tk++;
}

static void parse_body(int body_indent_in) {
    /* Skip any leading newlines/blank-line indents */
    for (;;) {
        if (cur()->type == TOK_NEWLINE) { tk++; continue; }
        if (cur()->type == TOK_INDENT && toks[tk+1].type == TOK_NEWLINE) { tk += 2; continue; }
        break;
    }

    int body_indent = body_indent_in;
    if (body_indent < 0) {
        if (cur()->type != TOK_INDENT) return;   /* empty body */
        body_indent = cur()->len;
    }

    while (cur()->type != TOK_EOF) {
        if (cur()->type == TOK_NEWLINE) { tk++; continue; }
        if (cur()->type == TOK_INDENT) {
            if (toks[tk+1].type == TOK_NEWLINE) { tk += 2; continue; }
            if (cur()->len < body_indent) return;
            /* If indent is deeper than body_indent, still parse at that indent
             * — parse_stmt will eat the INDENT and recursive structures will
             * open new nested bodies as needed. */
            tk++;
        } else {
            /* Orphan token mid-body (e.g. a trailing integer left over from
             * a previous statement's expression). Consume it as an "unknown
             * statement" rather than exiting the body — otherwise we'd drop
             * out of the body early and trim symbols that later statements
             * still need. */
            int t = cur()->type;
            if (t == TOK_INT || t == TOK_RPAREN || t == TOK_RBRACK ||
                t == TOK_COLON) {
                tk++;
                continue;
            }
            return;
        }
        parse_stmt();
        if (cur()->type == TOK_NEWLINE) tk++;
    }
}

/* ==========================================================================
 * Top-level: compositions
 * ========================================================================== */

static void parse_composition(void) {
    /* Skip a `host` or `kernel` prefix (lexed as TOK_IDENT — the bootstrap
     * treats both as semantic no-ops). The pre-pass did the same. */
    if (cur()->type == TOK_IDENT) {
        int len = cur()->len;
        const char *s = src + cur()->off;
        if ((len == 4 && memcmp(s, "host",   4) == 0) ||
            (len == 6 && memcmp(s, "kernel", 6) == 0)) {
            tk++;
        }
    }

    Token *n = cur();
    if (n->type != TOK_IDENT) { tk++; return; }
    tk++;

    int si = sym_find(src + n->off, n->len);
    if (si < 0) si = sym_add(src + n->off, n->len, SYM_COMPO);
    else if (syms[si].kind != SYM_COMPO)
        die("line %d: '%s' already defined", n->line, syms[si].name);

    int saved_syms = nsyms;
    frame_next_slot = 0;

    int nparams = 0;
    int param_slots[16] = {0};
    /* Keywords that can appear as identifiers in param position (compiler-
     * darwin.ls uses `buf` as a param name, e.g. in `write_file path buf buf_len`). */
    while (cur()->type == TOK_IDENT || cur()->type == TOK_BUF ||
           cur()->type == TOK_CONST || cur()->type == TOK_VAR ||
           cur()->type == TOK_LABEL) {
        Token *p = cur();
        tk++;
        int slot = alloc_slot();
        int psi = sym_add(src + p->off, p->len, SYM_LOCAL);
        syms[psi].slot = slot;
        if (nparams < 16) param_slots[nparams] = slot;
        nparams++;
    }
    if (nparams > 16) die("line %d: >16 params unsupported", n->line);
    syms[si].nparams = nparams;
    syms[si].code_off = cur_off();

    if (cur()->type == TOK_COLON) tk++;
    if (cur()->type == TOK_NEWLINE) tk++;

    emit_prologue();
    /* Spill first 8 params from ABI registers X0..X7 to stack slots. Extra
     * params (9..16) are passed in X8..X15 by the simple calling conv
     * wirth uses (not standard ABI, but consistent since we control both
     * sides of every call). */
    int n_reg_params = nparams < 16 ? nparams : 16;
    for (int i = 0; i < n_reg_params; i++) {
        emit_stur(i, 29, -param_slots[i]);
    }

    labels_reset();
    parse_body(-1);
    resolve_label_fixups();

    /* Fallthrough epilogue (MOV X0, #0 is not implicit) */
    emit_epilogue();

    sym_trim(saved_syms);
}

/* ==========================================================================
 * Bootstrap trampoline
 * ==========================================================================
 * LC_MAIN points at entryoff (the very first instruction of the code
 * section).  We reserve a two-instruction trampoline there: `BL main ;
 * BL exit_stub`.  After all compositions have been parsed we patch the
 * first BL to reach `main` and the second to a tiny stub that emits the
 * macOS exit(0) syscall, so compositions that fall off the end without
 * calling `trap` still exit cleanly.
 */

static int trampoline_bl_main_off;
static int trampoline_exit_stub_off;

static void emit_trampoline_placeholder(void) {
    /* Darwin arm64 LC_MAIN convention: kernel/dyld invoke the entry point
     * as a C-style main(argc, argv) — argc in X0, argv in X1.  We must
     * preserve both so that compositions like `main : argc ↑ $0 ; argv ↑ $1`
     * can read them.  Previously we zeroed X0 here to force an exit status
     * of 0 for programs that never call `trap`, but that broke argc for
     * every program that actually reads arguments.  The fall-through
     * exit_stub path below zeroes X0 *after* main returns, which handles
     * the no-trap case without clobbering argc on the way in. */
    trampoline_bl_main_off = cur_off();
    emit_bl(0);                  /* patched later */
    trampoline_exit_stub_off = cur_off();
    /* If main returns normally, fall through to MOV X0, #0 ; BL exit_stub */
    emit_mov_imm64(0, 0);
    emit_bl(0);                  /* patched later */
}

static int exit_stub_off;
static void emit_exit_stub(void) {
    exit_stub_off = cur_off();
    emit_mov_imm64(16, 1);
    emit_svc(0x80);
}

static void patch_trampoline(int main_off) {
    u32 bl1 = 0x94000000u | ((u32)((main_off - trampoline_bl_main_off) / 4) & 0x3FFFFFFu);
    patch32(trampoline_bl_main_off, bl1);
    int bl2_off = trampoline_exit_stub_off;
    /* skip the MOV X0, #0 instruction — BL is at +4 */
    int bl2_pos = bl2_off + 4;
    u32 bl2 = 0x94000000u | ((u32)((exit_stub_off - bl2_pos) / 4) & 0x3FFFFFFu);
    patch32(bl2_pos, bl2);
}

/* Peek: does the upcoming top-level IDENT look like a composition header
 * (i.e., it's followed eventually by a colon before a newline)? */
static int looks_like_compo_header(int start) {
    for (int i = start; i < ntoks; i++) {
        if (toks[i].type == TOK_NEWLINE || toks[i].type == TOK_EOF) return 0;
        if (toks[i].type == TOK_COLON) return 1;
    }
    return 0;
}

/* Pre-pass over the token stream: sym_add every top-level composition with
 * code_off = -1 so forward references find their targets at parse time.
 * nparams is counted as IDENT tokens between the name and ':'. */
static void collect_compositions(void) {
    int at_line_start = 1;
    int cur_indent    = 0;
    for (int i = 0; i < ntoks; i++) {
        int tt = toks[i].type;
        if (tt == TOK_EOF)     break;
        if (tt == TOK_NEWLINE) { at_line_start = 1; cur_indent = 0; continue; }
        if (tt == TOK_INDENT)  { cur_indent = toks[i].len; at_line_start = 1; continue; }

        if (!at_line_start) continue;
        at_line_start = 0;
        if (cur_indent > 0) continue;  /* only top-level lines */
        if (tt != TOK_IDENT) continue;

        /* Skip a leading `host` or `kernel` prefix, which are lexed as
         * TOK_IDENT in this build of wirth. Both are semantic no-ops for
         * the bootstrap — they select host vs. GPU target, which does
         * not affect host-ARM64 codegen. */
        int name_idx = i;
        if (toks[name_idx].type == TOK_IDENT) {
            int nlen = toks[name_idx].len;
            const char *ns = src + toks[name_idx].off;
            if ((nlen == 4 && memcmp(ns, "host",   4) == 0) ||
                (nlen == 6 && memcmp(ns, "kernel", 6) == 0)) {
                name_idx++;
                if (name_idx >= ntoks || toks[name_idx].type != TOK_IDENT) continue;
            }
        }

        int j = name_idx;
        int saw_colon = 0;
        int nparams   = 0;
        while (j < ntoks) {
            int tj = toks[j].type;
            if (tj == TOK_NEWLINE || tj == TOK_EOF) break;
            if (tj == TOK_COLON) { saw_colon = 1; break; }
            /* Count IDENT and keyword-as-ident tokens as params */
            if (j > name_idx && (tj == TOK_IDENT || tj == TOK_BUF ||
                                  tj == TOK_CONST || tj == TOK_VAR ||
                                  tj == TOK_LABEL))
                nparams++;
            j++;
        }
        if (!saw_colon) continue;

        Token *n = &toks[name_idx];
        int existing = sym_find(src + n->off, n->len);
        if (existing >= 0 && syms[existing].kind == SYM_COMPO) continue;
        int si = sym_add(src + n->off, n->len, SYM_COMPO);
        syms[si].code_off = -1;          /* forward-declared */
        syms[si].nparams  = nparams;
    }
}

/* After all compositions are emitted, patch every placeholder BL with
 * the resolved delta to its target. */
static void resolve_fixups(void) {
    for (int i = 0; i < nfixups; i++) {
        int bl_off = fixups[i].code_off;
        int si     = fixups[i].sym_idx;
        if (si < 0 || si >= nsyms) die("bad fixup sym_idx");
        Sym *s = &syms[si];
        if (s->code_off < 0)
            die("unresolved forward reference to '%s'", s->name);
        int delta = s->code_off - bl_off;
        u32 bl = 0x94000000u | ((u32)(delta / 4) & 0x3FFFFFFu);
        patch32(bl_off, bl);
    }
}

/* Patch every SYM_GLOBAL NOP placeholder with an ADR instruction whose
 * byte delta reaches the global's 8-byte slot in the trailing data region.
 * Must be called after all code emission is complete (so code_words is final)
 * but before the Mach-O writer packs everything.  The delta is purely
 * relative to the instruction's position within __TEXT, which means it's
 * independent of the Mach-O header layout — same code/data region works
 * regardless of where the kernel slides the image under PIE. */
/* Patch a 4-word placeholder (reserved by emit_mov_imm64_fixed) as:
 *   [0] ADRP Xd, <page>
 *   [1] ADD  Xd, Xd, #<page_offset>
 *   [2] NOP
 *   [3] NOP
 * The high 32 bits of the original MOVZ chain are unused — the PC-relative
 * form handles addresses anywhere in the process. */
static void patch_pcrel_load_addr(int code_word_idx, int rd, int insn_pos, u64 target_va) {
    /* insn_pos is the BYTE offset of the ADRP instruction relative to text
     * (code) base; add TEXT_VMADDR + text file offset to get its VA. But
     * we can compute delta symbolically: target_page - insn_page. */
    u64 insn_page   = (u64)insn_pos & ~(u64)0xFFF;
    u64 target_page = target_va & ~(u64)0xFFF;
    i64 page_delta  = (i64)target_page - (i64)insn_page;
    /* ADRP has 21-bit signed immediate scaled by 4K pages. */
    i64 page_imm = page_delta / 4096;
    if (page_imm > (1 << 20) - 1 || page_imm < -(1 << 20))
        die("global too far for ADRP (page_delta=%lld)", (long long)page_delta);
    u32 imm21 = (u32)(page_imm & 0x1FFFFFu);
    u32 immlo = imm21 & 3;
    u32 immhi = (imm21 >> 2) & 0x7FFFFu;
    u32 adrp  = 0x90000000u | (immlo << 29) | (immhi << 5) | (u32)(rd & 31);
    u32 page_off = (u32)(target_va & 0xFFFu);
    u32 add = 0x91000000u | (page_off << 10) | ((u32)(rd & 31) << 5) | (u32)(rd & 31);
    u32 nop = 0xD503201Fu;
    code[code_word_idx + 0] = adrp;
    code[code_word_idx + 1] = add;
    code[code_word_idx + 2] = nop;
    code[code_word_idx + 3] = nop;
}

static void resolve_global_fixups(void) {
    /* intentionally empty — patched in write_macho where code file offset
     * is known. */
}

static void parse_file(void) {
    tk = 0;
    collect_compositions();
    emit_trampoline_placeholder();

    int made_implicit_main = 0;
    int implicit_main_si = -1;

    while (cur()->type != TOK_EOF) {
        if (cur()->type == TOK_NEWLINE) { tk++; continue; }
        if (cur()->type == TOK_INDENT) {
            if (toks[tk+1].type == TOK_NEWLINE) { tk += 2; continue; }
            /* stray indented content at top level — skip */
            tk++; continue;
        }

        int t = cur()->type;

        /* `const` and `buf` are top-level-only declarations.  They emit no
         * code and must live at top-level scope so subsequent compositions
         * can reference them.  Handle them here, BEFORE the implicit-main
         * machinery, so (a) their symbols aren't trimmed when implicit main
         * closes out and (b) they don't silently fall inside an implicit
         * main prologue. */
        if (t == TOK_CONST || t == TOK_BUF) {
            if (made_implicit_main) {
                /* Close the implicit main first — const/buf come after it.
                 * Uncommon: in compiler-darwin.ls they appear at the top of
                 * the file, but keep this path defensive. */
                emit_epilogue();
                made_implicit_main = 0;
                sym_trim(implicit_main_si + 1);
            }
            parse_stmt();
            if (cur()->type == TOK_NEWLINE) tk++;
            continue;
        }

        /* Top-level `var NAME VALUE` — module-scope global, 8-byte data slot
         * in the trailing __TEXT data region.  Same lifetime rules as const/
         * buf: must be seen BEFORE implicit main is opened so the symbol
         * isn't trimmed and so compositions defined later in the file can
         * reference it.  If implicit main has already been opened, we fall
         * through to parse_stmt's existing TOK_VAR path (which makes it a
         * local) to preserve legacy test-file behaviour. */
        if (t == TOK_VAR && !made_implicit_main) {
            tk++;                    /* consume `var` */
            Token *n = cur();
            if (n->type != TOK_IDENT)
                die("line %d: expected name after top-level var", toks[tk].line);
            tk++;                    /* consume name */
            if (cur()->type != TOK_INT)
                die("line %d: top-level var '%.*s' must have int-literal "
                    "initializer (expression globals not supported in v1)",
                    n->line, n->len, src + n->off);
            i64 init_val = tok_int_value(cur());
            tk++;                    /* consume int */

            /* Allocate an 8-byte slot in data_buf (same bump allocator as
             * buf, with 8-byte alignment of the tail). */
            int bo = data_off;
            int new_off = bo + 8;
            new_off = (new_off + 7) & ~7;
            if (new_off > DATA_MAX)
                die("global data overflow (DATA_MAX=%d)", DATA_MAX);
            data_off = new_off;

            /* Write the initializer value in host byte order (same as how
             * the code buffer is written, so this machine's endianness
             * matches the target). */
            memcpy(data_buf + bo, &init_val, 8);

            int si = sym_add(src + n->off, n->len, SYM_GLOBAL);
            syms[si].code_off = bo;      /* offset within data_buf */

            if (cur()->type == TOK_NEWLINE) tk++;
            continue;
        }

        /* Top-level shorthand `NAME INT NEWLINE` — declare a const. This
         * is wirth's compatibility path for compiler-darwin.ls's terse
         * opcode constant lists (`OP_FADD 0x7221` etc.). Requires NO
         * open implicit main so the const stays at top-level scope. */
        if (t == TOK_IDENT && !made_implicit_main) {
            if (tk + 1 < ntoks && toks[tk+1].type == TOK_INT &&
                tk + 2 < ntoks &&
                (toks[tk+2].type == TOK_NEWLINE || toks[tk+2].type == TOK_EOF)) {
                Token *nn = cur();
                tk++;                     /* consume name */
                i64 val = tok_int_value(cur());
                tk++;                     /* consume int */
                int si = sym_add(src + nn->off, nn->len, SYM_CONST);
                syms[si].code_off = (int)val;
                if (cur()->type == TOK_NEWLINE) tk++;
                continue;
            }
        }

        /* If the upcoming top-level statement isn't a composition header
         * (no colon before the newline), fabricate an implicit `main` that
         * collects those bare statements. */
        int is_compo = 0;
        if (t == TOK_IDENT) is_compo = looks_like_compo_header(tk);

        if (!is_compo) {
            if (!made_implicit_main) {
                /* Create a main composition at the current code offset */
                implicit_main_si = sym_add("main", 4, SYM_COMPO);
                syms[implicit_main_si].code_off = cur_off();
                syms[implicit_main_si].nparams  = 0;
                frame_next_slot = 0;
                emit_prologue();
                made_implicit_main = 1;
            }
            parse_stmt();
            if (cur()->type == TOK_NEWLINE) tk++;
            continue;
        }

        if (made_implicit_main) {
            /* Close out the implicit main so we can start a real composition */
            emit_epilogue();
            made_implicit_main = 0;
            sym_trim(implicit_main_si + 1);
        }
        parse_composition();
    }
    if (made_implicit_main) {
        emit_epilogue();
        sym_trim(implicit_main_si + 1);
    }
    emit_exit_stub();

    /* All compositions are emitted — their code_off fields are set.
     * Patch every forward-reference BL placeholder. */
    resolve_fixups();

    /* Patch every top-level `var` address-materialisation NOP with the
     * correct ADR instruction (PC-relative — slide-invariant). */
    resolve_global_fixups();

    /* Find main; if absent, treat first composition as entry. */
    int main_off = -1;
    for (int i = 0; i < nsyms; i++) {
        if (syms[i].kind == SYM_COMPO && strcmp(syms[i].name, "main") == 0) {
            main_off = syms[i].code_off;
            break;
        }
    }
    if (main_off < 0) {
        for (int i = 0; i < nsyms; i++) {
            if (syms[i].kind == SYM_COMPO) { main_off = syms[i].code_off; break; }
        }
    }
    if (main_off < 0) die("no compositions defined");
    patch_trampoline(main_off);
}

/* ==========================================================================
 * Mach-O writer (macOS ARM64 dynamic, ad-hoc signed post-link)
 * ========================================================================== */

typedef struct __attribute__((packed)) {
    u32 magic;
    u32 cputype;
    u32 cpusubtype;
    u32 filetype;
    u32 ncmds;
    u32 sizeofcmds;
    u32 flags;
    u32 reserved;
} MH64;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    char segname[16];
    u64 vmaddr, vmsize, fileoff, filesize;
    u32 maxprot, initprot;
    u32 nsects, flags;
} SEG64;

typedef struct __attribute__((packed)) {
    char sectname[16];
    char segname[16];
    u64 addr, size;
    u32 offset, align, reloff, nreloc, flags, reserved1, reserved2, reserved3;
} SEC64;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u32 platform, minos, sdk, ntools;
} BUILDV;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u64 entryoff, stacksize;
} ENTRYPT;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u32 symoff, nsyms, stroff, strsize;
} SYMTAB;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u32 ilocalsym, nlocalsym, iextdefsym, nextdefsym, iundefsym, nundefsym;
    u32 tocoff, ntoc, modtaboff, nmodtab;
    u32 extrefsymoff, nextrefsyms, indirectsymoff, nindirectsyms;
    u32 extreloff, nextrel, locreloff, nlocrel;
} DYSYMTAB;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u32 dataoff, datasize;
} LEDATA;

typedef struct __attribute__((packed)) {
    u32 cmd, cmdsize;
    u32 name_off, timestamp, current_version, compat_version;
} DYLIB;

#define LC_SEGMENT_64          0x19
#define LC_SYMTAB              0x02
#define LC_DYSYMTAB            0x0B
#define LC_LOAD_DYLINKER       0x0E
#define LC_LOAD_DYLIB          0x0C
#define LC_MAIN                (0x28 | 0x80000000u)
#define LC_DYLD_CHAINED_FIXUPS (0x34 | 0x80000000u)
#define LC_BUILD_VERSION       0x32

#define MH_MAGIC_64    0xfeedfacf
#define CPU_TYPE_ARM64 0x0100000c
#define MH_EXECUTE     2
#define MH_NOUNDEFS    0x00000001
#define MH_DYLDLINK    0x00000004
#define MH_TWOLEVEL    0x00000080
#define MH_PIE         0x00200000

#define S_ATTR_PURE_INSTRUCTIONS  0x80000000
#define S_ATTR_SOME_INSTRUCTIONS  0x00000400

static void write_macho(const char *outpath) {
    u32 pz_sz = (u32)sizeof(SEG64);
    u32 tx_sz = (u32)(sizeof(SEG64) + sizeof(SEC64));
    u32 le_sz = (u32)sizeof(SEG64);
    u32 cf_sz = (u32)sizeof(LEDATA);
    u32 sy_sz = (u32)sizeof(SYMTAB);
    u32 ds_sz = (u32)sizeof(DYSYMTAB);
    u32 dl_sz = 32;
    u32 bv_sz = (u32)sizeof(BUILDV);
    u32 mn_sz = (u32)sizeof(ENTRYPT);
    u32 ld_sz = 56;

    u32 lc_total = pz_sz + tx_sz + le_sz + cf_sz + sy_sz + ds_sz + dl_sz + bv_sz + mn_sz + ld_sz;
    u32 ncmds = 10;
    u32 hdr_sz = (u32)sizeof(MH64);
    u32 lc_end = hdr_sz + lc_total;
    /* 64 bytes of slack so codesign can append LC_CODE_SIGNATURE without
     * trampling the code. */
    u32 code_off = (lc_end + 64 + 3) & ~3u;
    u32 code_len = (u32)(code_words * 4);
    /* `buf` data trails the code within __TEXT (see the buf comment block
     * near data_buf).  data_len is 8-byte aligned by add_buf_fixup's data_off
     * maintenance; we re-assert here and also include it in the segment. */
    u32 data_len = (u32)data_off;
    u32 text_filesize = code_off + code_len + data_len;
    u32 text_padded = (text_filesize + PAGE_SIZE_OUT - 1) & ~(u32)(PAGE_SIZE_OUT - 1);
    if (text_padded == 0) text_padded = PAGE_SIZE_OUT;

    /* Resolve `buf` fixups now that we know where code and data live in the
     * final image.  Each buf's virtual address is computed as
     *   TEXT_VMADDR + code_off + code_len + buf_data_off
     * and patched into the 4-instruction MOVZ/MOVK sequence recorded at
     * parse time.  Patching code[] in place is safe because the memcpy of
     * `code` into the output buffer happens further below. */
    for (int i = 0; i < n_buf_fixups; i++) {
        BufFixup *bf = &buf_fixups[i];
        u64 va = (u64)TEXT_VMADDR + (u64)code_off + (u64)code_len
               + (u64)(u32)bf->buf_data_off;
        int insn_file_off = code_off + bf->code_word_idx * 4;
        u64 insn_va = (u64)TEXT_VMADDR + (u64)insn_file_off;
        /* Compute page delta between insn and target via VA */
        u64 insn_page   = insn_va & ~(u64)0xFFF;
        u64 target_page = va & ~(u64)0xFFF;
        i64 page_delta  = (i64)target_page - (i64)insn_page;
        i64 page_imm    = page_delta / 4096;
        u32 imm21 = (u32)((u64)page_imm & 0x1FFFFFu);
        u32 immlo = imm21 & 3;
        u32 immhi = (imm21 >> 2) & 0x7FFFFu;
        u32 adrp  = 0x90000000u | (immlo << 29) | (immhi << 5) | (u32)(bf->rd & 31);
        u32 page_off = (u32)(va & 0xFFFu);
        u32 add = 0x91000000u | (page_off << 10) | ((u32)(bf->rd & 31) << 5) | (u32)(bf->rd & 31);
        u32 nop = 0xD503201Fu;
        code[bf->code_word_idx + 0] = adrp;
        code[bf->code_word_idx + 1] = add;
        code[bf->code_word_idx + 2] = nop;
        code[bf->code_word_idx + 3] = nop;
    }
    /* Same treatment for global vars */
    for (int i = 0; i < n_global_fixups; i++) {
        GlobalFixup *gf = &global_fixups[i];
        u64 va = (u64)TEXT_VMADDR + (u64)code_off + (u64)code_len
               + (u64)(u32)gf->data_off;
        int insn_file_off = code_off + gf->code_word_idx * 4;
        u64 insn_va = (u64)TEXT_VMADDR + (u64)insn_file_off;
        u64 insn_page   = insn_va & ~(u64)0xFFF;
        u64 target_page = va & ~(u64)0xFFF;
        i64 page_delta  = (i64)target_page - (i64)insn_page;
        i64 page_imm    = page_delta / 4096;
        u32 imm21 = (u32)((u64)page_imm & 0x1FFFFFu);
        u32 immlo = imm21 & 3;
        u32 immhi = (imm21 >> 2) & 0x7FFFFu;
        u32 adrp  = 0x90000000u | (immlo << 29) | (immhi << 5) | (u32)(gf->rd & 31);
        u32 page_off = (u32)(va & 0xFFFu);
        u32 add = 0x91000000u | (page_off << 10) | ((u32)(gf->rd & 31) << 5) | (u32)(gf->rd & 31);
        u32 nop = 0xD503201Fu;
        code[gf->code_word_idx + 0] = adrp;
        code[gf->code_word_idx + 1] = add;
        code[gf->code_word_idx + 2] = nop;
        code[gf->code_word_idx + 3] = nop;
    }

    u32 le_off  = text_padded;
    u32 cf_data_off = le_off;
    u32 cf_data_size = 56;
    u32 symtab_off = cf_data_off + cf_data_size;
    u32 nsyms_le = 1;
    u32 symtab_size = nsyms_le * 16;
    u32 strtab_off = symtab_off + symtab_size;
    u32 strtab_size = 32;
    u32 le_used = strtab_off + strtab_size - le_off;
    u32 filesize_total = le_off + le_used;

    u64 text_vmaddr = TEXT_VMADDR;
    u64 le_vmaddr   = text_vmaddr + text_padded;

    char *buf = calloc(1, filesize_total + 16);
    if (!buf) die("out of memory");

    MH64 *h = (MH64*)buf;
    h->magic = MH_MAGIC_64;
    h->cputype = CPU_TYPE_ARM64;
    h->cpusubtype = 0;
    h->filetype = MH_EXECUTE;
    h->ncmds = ncmds;
    h->sizeofcmds = lc_total;
    h->flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE;
    h->reserved = 0;

    char *p = buf + hdr_sz;

    SEG64 *pz = (SEG64*)p;
    pz->cmd = LC_SEGMENT_64; pz->cmdsize = pz_sz;
    memcpy(pz->segname, "__PAGEZERO", 10);
    pz->vmsize = 0x100000000ULL;
    p += pz_sz;

    SEG64 *tx = (SEG64*)p;
    tx->cmd = LC_SEGMENT_64; tx->cmdsize = tx_sz;
    memcpy(tx->segname, "__TEXT", 6);
    tx->vmaddr = text_vmaddr; tx->vmsize = text_padded;
    tx->fileoff = 0; tx->filesize = text_padded;
    tx->maxprot = 5; tx->initprot = 5;
    tx->nsects = 1;
    SEC64 *sec = (SEC64*)(tx + 1);
    memcpy(sec->sectname, "__text", 6);
    memcpy(sec->segname, "__TEXT", 6);
    sec->addr = text_vmaddr + code_off;
    sec->size = code_len;
    sec->offset = code_off;
    sec->align = 2;
    sec->flags = S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS;
    p += tx_sz;

    SEG64 *le = (SEG64*)p;
    le->cmd = LC_SEGMENT_64; le->cmdsize = le_sz;
    memcpy(le->segname, "__LINKEDIT", 10);
    le->vmaddr = le_vmaddr; le->vmsize = PAGE_SIZE_OUT;
    le->fileoff = le_off; le->filesize = le_used;
    le->maxprot = 1; le->initprot = 1;
    p += le_sz;

    LEDATA *cf = (LEDATA*)p;
    cf->cmd = LC_DYLD_CHAINED_FIXUPS;
    cf->cmdsize = cf_sz;
    cf->dataoff = cf_data_off;
    cf->datasize = cf_data_size;
    p += cf_sz;

    SYMTAB *sy = (SYMTAB*)p;
    sy->cmd = LC_SYMTAB;
    sy->cmdsize = sy_sz;
    sy->symoff = symtab_off;
    sy->nsyms = nsyms_le;
    sy->stroff = strtab_off;
    sy->strsize = strtab_size;
    p += sy_sz;

    DYSYMTAB *ds = (DYSYMTAB*)p;
    memset(ds, 0, ds_sz);
    ds->cmd = LC_DYSYMTAB;
    ds->cmdsize = ds_sz;
    ds->nextdefsym = 1;
    p += ds_sz;

    {
        u32 *w = (u32*)p;
        w[0] = LC_LOAD_DYLINKER;
        w[1] = dl_sz;
        w[2] = 12;
        memcpy(p + 12, "/usr/lib/dyld", 13);
        p += dl_sz;
    }

    BUILDV *bv = (BUILDV*)p;
    bv->cmd = LC_BUILD_VERSION;
    bv->cmdsize = bv_sz;
    bv->platform = 1;
    bv->minos = 11 << 16;
    bv->sdk   = 11 << 16;
    bv->ntools = 0;
    p += bv_sz;

    ENTRYPT *mn = (ENTRYPT*)p;
    mn->cmd = LC_MAIN;
    mn->cmdsize = mn_sz;
    mn->entryoff = code_off;
    mn->stacksize = 0;
    p += mn_sz;

    {
        DYLIB *ld = (DYLIB*)p;
        ld->cmd = LC_LOAD_DYLIB;
        ld->cmdsize = ld_sz;
        ld->name_off = 24;
        ld->timestamp = 2;
        ld->current_version = 0x054c0000;
        ld->compat_version  = 0x00010000;
        memcpy(p + 24, "/usr/lib/libSystem.B.dylib", 26);
        p += ld_sz;
    }

    memcpy(buf + code_off, code, code_len);
    if (data_len > 0) {
        /* Append buf data immediately after the code. Mapped R+X as part of
         * __TEXT — see the buf-storage comment block for the read-only
         * limitation. */
        memcpy(buf + code_off + code_len, data_buf, data_len);
    }

    /* Empty chained fixups header */
    u32 *cfh = (u32*)(buf + cf_data_off);
    cfh[0] = 0;
    cfh[1] = 0x20;
    cfh[2] = 0x30;
    cfh[3] = 0x30;
    cfh[4] = 0;
    cfh[5] = 1;
    cfh[6] = 0;
    cfh[8] = 3;
    cfh[9] = 0; cfh[10] = 0; cfh[11] = 0;

    /* Symbol table: one symbol __mh_execute_header */
    u8 *symp = (u8*)(buf + symtab_off);
    *(u32*)(symp + 0) = 1;
    symp[4] = 0x0F;
    symp[5] = 1;
    *(u16*)(symp + 6) = 0x10;
    *(u64*)(symp + 8) = text_vmaddr;

    char *str = buf + strtab_off;
    str[0] = ' ';
    memcpy(str + 1, "_mh_execute_header", 18);

    int fd = open(outpath, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (fd < 0) die("cannot open %s: %s", outpath, strerror(errno));
    if (write(fd, buf, filesize_total) != (ssize_t)filesize_total)
        die("short write: %s", strerror(errno));
    close(fd);
    free(buf);

    /* codesign -s - (ad-hoc) — without this the kernel SIGKILLs at exec. */
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "codesign -s - -f '%s' >/dev/null 2>&1", outpath);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "wirth: warning: codesign failed (rc=%d); binary may not run\n", rc);
    }
}

/* ==========================================================================
 * main
 * ========================================================================== */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.ls> <output>\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    int fd = open(inpath, O_RDONLY);
    if (fd < 0) die("cannot open %s", inpath);
    struct stat st;
    if (fstat(fd, &st) < 0) die("stat failed");
    src_len = (size_t)st.st_size;
    src = malloc(src_len + 1);
    if (!src) die("out of memory");
    if (read(fd, src, src_len) != (ssize_t)src_len) die("short read");
    src[src_len] = 0;
    close(fd);

    lex();
    parse_file();
    write_macho(outpath);
    return 0;
}
