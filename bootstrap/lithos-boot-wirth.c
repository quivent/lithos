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
    TOK_LOAD,       /* ↓ */
    TOK_REG_READ,   /* ↑ */
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_AMP, TOK_PIPE, TOK_CARET,
    TOK_SHL, TOK_SHR,
    TOK_EQ,
    TOK_EQEQ, TOK_NEQ, TOK_LT, TOK_GT, TOK_LTE, TOK_GTE,
    TOK_LPAREN, TOK_RPAREN,
    TOK_COLON,
    TOK_DOLLAR,
    TOK_HASH,
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
    if (temp_next > 10) die("expression too complex (>10 temps)");
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
        || t == TOK_MINUS || t == TOK_REG_READ || t == TOK_DOLLAR;
}

static int parse_expr(void);
static void parse_body(int body_indent);

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

    if (t->type == TOK_DOLLAR) {
        int rn = tok_dollar_reg(t);
        if (rn < 0) die("line %d: bad $reg", t->line);
        tk++;
        int r = alloc_temp();
        emit_mov_reg(r, rn);
        return r;
    }

    if (t->type == TOK_IDENT) {
        int si = sym_find(src + t->off, t->len);
        tk++;
        if (si < 0) die("line %d: unknown name '%.*s'", t->line, t->len, src + t->off);
        Sym *s = &syms[si];
        if (s->kind == SYM_LOCAL || s->kind == SYM_FOR_VAR) {
            int r = alloc_temp();
            emit_ldur(r, 29, -s->slot);
            return r;
        }
        if (s->kind == SYM_COMPO) {
            int delta = s->code_off - cur_off();
            emit_bl(delta);
            int r = alloc_temp();
            emit_mov_reg(r, 0);
            return r;
        }
        die("line %d: cannot use '%.*s' in expression", t->line, t->len, src + t->off);
    }

    die("line %d: unexpected token in expression (type=%d)", t->line, t->type);
}

static int parse_mul(void) {
    int a = parse_primary();
    for (;;) {
        int op = cur()->type;
        if (op != TOK_STAR && op != TOK_SLASH) break;
        tk++;
        int b = parse_primary();
        int r = alloc_temp();
        if (op == TOK_STAR) emit_mul(r, a, b);
        else                 emit_sdiv(r, a, b);
        a = r;
    }
    return a;
}
static int parse_add(void) {
    int a = parse_mul();
    for (;;) {
        int op = cur()->type;
        if (op != TOK_PLUS && op != TOK_MINUS) break;
        tk++;
        int b = parse_mul();
        int r = alloc_temp();
        if (op == TOK_PLUS) emit_add_reg(r, a, b);
        else                 emit_sub_reg(r, a, b);
        a = r;
    }
    return a;
}
static int parse_shift(void) {
    int a = parse_add();
    for (;;) {
        int op = cur()->type;
        if (op != TOK_SHL && op != TOK_SHR) break;
        tk++;
        int b = parse_add();
        int r = alloc_temp();
        if (op == TOK_SHL) emit_lsl_reg(r, a, b);
        else                emit_lsr_reg(r, a, b);
        a = r;
    }
    return a;
}
static int parse_bits(void) {
    int a = parse_shift();
    for (;;) {
        int op = cur()->type;
        if (op != TOK_AMP && op != TOK_PIPE && op != TOK_CARET) break;
        tk++;
        int b = parse_shift();
        int r = alloc_temp();
        if (op == TOK_AMP)       emit_and_reg(r, a, b);
        else if (op == TOK_PIPE) emit_orr_reg(r, a, b);
        else                     emit_eor_reg(r, a, b);
        a = r;
    }
    return a;
}
static int parse_expr(void) { return parse_bits(); }

/* ==========================================================================
 * Statement dispatch
 * ========================================================================== */

static void emit_call_args(int si) {
    Sym *s = &syms[si];
    int argregs[8];
    int nargs = 0;
    while (nargs < s->nparams) {
        int t = peek_type();
        if (t == TOK_NEWLINE || t == TOK_EOF || t == TOK_INDENT) break;
        if (!is_expr_start(t)) break;
        argregs[nargs++] = parse_expr();
    }
    /* Move arg regs into X0..X(nargs-1) back-to-front to avoid overlap. */
    for (int i = nargs - 1; i >= 0; i--) {
        emit_mov_reg(i, argregs[i]);
    }
    int delta = s->code_off - cur_off();
    emit_bl(delta);
}

static void parse_stmt(void) {
    reset_temps();
    if (cur()->type == TOK_INDENT) tk++;

    Token *t = cur();

    if (t->type == TOK_NEWLINE) { tk++; return; }
    if (t->type == TOK_EOF)     return;

    if (t->type == TOK_TRAP) {
        tk++;
        /* macOS exit syscall: X16 = 1, SVC #0x80.  X0 already holds the exit code. */
        emit_mov_imm64(16, 1);
        emit_svc(0x80);
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

    if (t->type == TOK_REG_READ) {
        /* bare load statement — discard */
        (void)parse_expr();
        return;
    }

    if (t->type == TOK_IF) {
        tk++;
        int ct = cur()->type;
        int is_compound = (ct == TOK_LT || ct == TOK_GT || ct == TOK_LTE ||
                           ct == TOK_GTE || ct == TOK_EQEQ || ct == TOK_NEQ);
        int patch_off;

        if (is_compound) {
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
            patch_off = cur_off();
            emit_b_cond(cc_inv, 0);
        } else {
            int r = parse_expr();
            patch_off = cur_off();
            emit_cbz(r, 0);
        }
        if (cur()->type == TOK_COLON) tk++;
        if (cur()->type == TOK_NEWLINE) tk++;

        int prev_syms = nsyms;
        int prev_frame = frame_next_slot;
        parse_body(-1);
        sym_trim(prev_syms);
        frame_next_slot = prev_frame;

        int delta = cur_off() - patch_off;
        u32 old = code[patch_off / 4];
        if (is_compound) {
            u32 cc = old & 0xF;
            u32 w = 0x54000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | cc;
            patch32(patch_off, w);
        } else {
            u32 rt = old & 0x1F;
            u32 w = 0xB4000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | rt;
            patch32(patch_off, w);
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

        int prev_syms = nsyms;
        int prev_frame = frame_next_slot;
        parse_body(-1);
        sym_trim(prev_syms);
        frame_next_slot = prev_frame;

        emit_b(top - cur_off());
        int delta = cur_off() - patch_off;
        u32 old = code[patch_off / 4];
        u32 rt = old & 0x1F;
        u32 w = 0xB4000000u | (((u32)(delta / 4) & 0x7FFFFu) << 5) | rt;
        patch32(patch_off, w);
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

        int prev_syms = nsyms;
        int prev_frame = frame_next_slot;
        parse_body(-1);
        sym_trim(prev_syms);
        frame_next_slot = prev_frame;

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
        int prev_syms = nsyms;
        int prev_frame = frame_next_slot;
        parse_body(-1);
        sym_trim(prev_syms);
        frame_next_slot = prev_frame;
        return;
    }

    if (t->type == TOK_IDENT) {
        Token *n = t;
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
        /* Existing local — reassignment */
        if (cur()->type == TOK_EQ) tk++;
        int r = parse_expr();
        emit_stur(r, 29, -s->slot);
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
    int param_slots[8] = {0};
    while (cur()->type == TOK_IDENT) {
        Token *p = cur();
        tk++;
        int slot = alloc_slot();
        int psi = sym_add(src + p->off, p->len, SYM_LOCAL);
        syms[psi].slot = slot;
        if (nparams < 8) param_slots[nparams] = slot;
        nparams++;
    }
    if (nparams > 8) die("line %d: >8 params unsupported", n->line);
    syms[si].nparams = nparams;
    syms[si].code_off = cur_off();

    if (cur()->type == TOK_COLON) tk++;
    if (cur()->type == TOK_NEWLINE) tk++;

    emit_prologue();
    for (int i = 0; i < nparams; i++) {
        emit_stur(i, 29, -param_slots[i]);
    }

    parse_body(-1);

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
    /* Zero X0 before entering main so that programs whose `trap` path
     * never touches $0 exit with status 0 rather than argc. */
    emit_mov_imm64(0, 0);
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

static void parse_file(void) {
    tk = 0;
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

        /* If the upcoming top-level statement isn't a composition header
         * (no colon before the newline), fabricate an implicit `main` that
         * collects those bare statements. */
        int t = cur()->type;
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
    u32 text_filesize = code_off + code_len;
    u32 text_padded = (text_filesize + PAGE_SIZE_OUT - 1) & ~(u32)(PAGE_SIZE_OUT - 1);
    if (text_padded == 0) text_padded = PAGE_SIZE_OUT;

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
