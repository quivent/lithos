/* lithos-boot-linus.c — bootstrap compiler for Lithos
 *
 * Written by "linus". One C file. Reads a .ls source, emits a macOS
 * ARM64 Mach-O executable. Stack-resident locals: every binding/var/param
 * gets an offset from SP. Expression temps are a bump allocator over
 * X9..X18 that resets at each statement boundary. No register allocator
 * drama, no fraudulent spills.
 *
 * Build:  cc -O2 -Wall -Wextra -o lithos-boot-linus lithos-boot-linus.c
 * Run:    ./lithos-boot-linus input.ls output
 *         chmod +x output && ./output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* ====================================================================
 * Token type constants — match lithos-lexer.s so we could share .ls files.
 * ==================================================================== */
enum {
    TOK_EOF = 0, TOK_NEWLINE = 1, TOK_INDENT = 2, TOK_INT = 3,
    TOK_FLOAT = 4, TOK_IDENT = 5,
    TOK_KERNEL = 11, TOK_PARAM = 12, TOK_IF = 13, TOK_ELSE = 14,
    TOK_ELIF = 15, TOK_FOR = 16, TOK_ENDFOR = 17, TOK_EACH = 18,
    TOK_STRIDE = 19, TOK_WHILE = 20, TOK_RETURN = 21, TOK_CONST = 22,
    TOK_VAR = 23, TOK_BUF = 24,
    TOK_LOAD = 36, TOK_STORE = 37, TOK_REG_READ = 38, TOK_REG_WRITE = 39,
    TOK_F32 = 40, TOK_U32 = 41, TOK_S32 = 42, TOK_F16 = 43,
    TOK_PTR = 44, TOK_VOID = 45,
    TOK_PLUS = 50, TOK_MINUS = 51, TOK_STAR = 52, TOK_SLASH = 53,
    TOK_EQ = 54, TOK_EQEQ = 55, TOK_NEQ = 56,
    TOK_LT = 57, TOK_GT = 58, TOK_LTE = 59, TOK_GTE = 60,
    TOK_AMP = 61, TOK_PIPE = 62, TOK_CARET = 63,
    TOK_SHL = 64, TOK_SHR = 65,
    TOK_LBRACK = 67, TOK_RBRACK = 68, TOK_LPAREN = 69, TOK_RPAREN = 70,
    TOK_COMMA = 71, TOK_COLON = 72, TOK_DOT = 73, TOK_AT = 74,
    TOK_TRAP = 89
};

/* ====================================================================
 * Register conventions
 * ==================================================================== */
#define R_SP    31
#define R_FP    29
#define R_LR    30
#define TEMP_LO 9
#define TEMP_HI 18    /* X9..X18 — ten expression temporaries */

/* Condition codes for B.cond */
enum { CC_EQ = 0, CC_NE = 1, CC_CS = 2, CC_CC = 3, CC_MI = 4, CC_PL = 5,
       CC_VS = 6, CC_VC = 7, CC_HI = 8, CC_LS = 9, CC_GE = 10, CC_LT = 11,
       CC_GT = 12, CC_LE = 13 };

/* Symbol kinds */
enum { KIND_LOCAL = 1, KIND_PARAM = 2, KIND_COMP = 3, KIND_UNKNOWN = 0 };

/* ====================================================================
 * Global state
 * ==================================================================== */
static const char *src;        /* mmap'd source buffer */
static size_t      src_len;

typedef struct {
    int type;
    int off;
    int len;
} Token;

static Token  tokens[131072];
static int    n_tokens;
static int    tpos;              /* current token index */

typedef struct {
    char name[96];
    int  name_len;
    int  kind;
    int  offset;                 /* stack slot for locals, code-word idx for comps */
    int  scope;
} Sym;

static Sym    syms[8192];
static int    n_syms;
static int    scope_depth;

static uint32_t code[524288];    /* 2 MB of instructions */
static int      code_len;        /* in 32-bit words */

static int    temp_next;         /* next expression temp register (X9..X18) */
static int    frame_next;        /* next stack slot (positive offset from SP) */

static const char *src_path;     /* for error messages */

/* ====================================================================
 * Errors
 * ==================================================================== */
static void die_at(const char *msg, int off) {
    /* Compute line/col from src offset */
    int line = 1, col = 1;
    for (int i = 0; i < off && i < (int)src_len; i++) {
        if (src[i] == '\n') { line++; col = 1; } else col++;
    }
    fprintf(stderr, "%s:%d:%d: %s\n", src_path ? src_path : "?", line, col, msg);
    exit(1);
}

static void die(const char *msg) {
    if (tpos < n_tokens) die_at(msg, tokens[tpos].off);
    fprintf(stderr, "lithos-boot-linus: %s\n", msg);
    exit(1);
}

/* ====================================================================
 * Lexer
 * ==================================================================== */
static int is_ident_start(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int is_ident_cont(int c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}
static int is_digit(int c) { return c >= '0' && c <= '9'; }

/* Keyword table — literal string + token type */
typedef struct { const char *s; int len; int tok; } KW;
static const KW kwtab[] = {
    {"if",     2, TOK_IF},
    {"for",    3, TOK_FOR}, {"var", 3, TOK_VAR}, {"buf", 3, TOK_BUF},
    {"f32",    3, TOK_F32}, {"u32", 3, TOK_U32}, {"s32", 3, TOK_S32},
    {"f16",    3, TOK_F16}, {"ptr", 3, TOK_PTR},
    {"each",   4, TOK_EACH}, {"else", 4, TOK_ELSE}, {"elif", 4, TOK_ELIF},
    {"void",   4, TOK_VOID}, {"trap", 4, TOK_TRAP},
    {"param",  5, TOK_PARAM}, {"while", 5, TOK_WHILE}, {"const", 5, TOK_CONST},
    {"kernel", 6, TOK_KERNEL}, {"stride", 6, TOK_STRIDE}, {"return", 6, TOK_RETURN},
    {"endfor", 6, TOK_ENDFOR},
    {NULL, 0, 0}
};

static int match_kw(const char *s, int len) {
    for (int i = 0; kwtab[i].s; i++) {
        if (kwtab[i].len == len && memcmp(kwtab[i].s, s, (size_t)len) == 0)
            return kwtab[i].tok;
    }
    return TOK_IDENT;
}

static void emit_tok(int type, int off, int len) {
    if (n_tokens >= (int)(sizeof tokens / sizeof tokens[0])) die("too many tokens");
    tokens[n_tokens].type = type;
    tokens[n_tokens].off  = off;
    tokens[n_tokens].len  = len;
    n_tokens++;
}

static void lex(void) {
    size_t i = 0;
    int at_line_start = 1;
    while (i < src_len) {
        unsigned char c;
        if (at_line_start) {
            int indent = 0;
            while (i < src_len && src[i] == ' ') { indent++; i++; }
            /* Blank line or comment-only line: skip without emitting INDENT */
            if (i >= src_len || src[i] == '\n') { at_line_start = 0; continue; }
            if (src[i] == '\\' && i + 1 < src_len && src[i + 1] == '\\') {
                at_line_start = 0;
                continue;
            }
            emit_tok(TOK_INDENT, (int)(i - indent), indent);
            at_line_start = 0;
        }
        if (i >= src_len) break;
        c = (unsigned char)src[i];

        if (c == ' ' || c == '\t') { i++; continue; }

        if (c == '\n') {
            emit_tok(TOK_NEWLINE, (int)i, 1);
            i++;
            at_line_start = 1;
            continue;
        }

        /* line comment */
        if (c == '\\' && i + 1 < src_len && src[i + 1] == '\\') {
            while (i < src_len && src[i] != '\n') i++;
            continue;
        }

        /* identifier / keyword */
        if (is_ident_start(c)) {
            size_t start = i;
            while (i < src_len && is_ident_cont((unsigned char)src[i])) i++;
            int len = (int)(i - start);
            int tok = match_kw(&src[start], len);
            emit_tok(tok, (int)start, len);
            continue;
        }

        /* $N register reference — emit as IDENT, parser recognizes it */
        if (c == '$') {
            size_t start = i;
            i++;
            while (i < src_len && is_digit((unsigned char)src[i])) i++;
            emit_tok(TOK_IDENT, (int)start, (int)(i - start));
            continue;
        }

        /* integer literal (decimal or hex) */
        if (is_digit(c)) {
            size_t start = i;
            if (i + 1 < src_len && src[i] == '0' &&
                (src[i + 1] == 'x' || src[i + 1] == 'X')) {
                i += 2;
                while (i < src_len) {
                    int ch = (unsigned char)src[i];
                    if (!((ch >= '0' && ch <= '9') ||
                          (ch >= 'a' && ch <= 'f') ||
                          (ch >= 'A' && ch <= 'F'))) break;
                    i++;
                }
            } else {
                while (i < src_len && is_digit((unsigned char)src[i])) i++;
            }
            emit_tok(TOK_INT, (int)start, (int)(i - start));
            continue;
        }

        /* two-char operators */
        if (i + 1 < src_len) {
            char a = src[i], b = src[i + 1];
            if (a == '=' && b == '=') { emit_tok(TOK_EQEQ, (int)i, 2); i += 2; continue; }
            if (a == '!' && b == '=') { emit_tok(TOK_NEQ,  (int)i, 2); i += 2; continue; }
            if (a == '<' && b == '=') { emit_tok(TOK_LTE,  (int)i, 2); i += 2; continue; }
            if (a == '>' && b == '=') { emit_tok(TOK_GTE,  (int)i, 2); i += 2; continue; }
            if (a == '<' && b == '<') { emit_tok(TOK_SHL,  (int)i, 2); i += 2; continue; }
            if (a == '>' && b == '>') { emit_tok(TOK_SHR,  (int)i, 2); i += 2; continue; }
        }

        /* UTF-8 arrows: ← → ↑ ↓ */
        if (c == 0xE2 && i + 2 < src_len && (unsigned char)src[i + 1] == 0x86) {
            unsigned char c3 = (unsigned char)src[i + 2];
            int tok = -1;
            if      (c3 == 0x90) tok = TOK_STORE;       /* ← */
            else if (c3 == 0x91) tok = TOK_REG_READ;    /* ↑ */
            else if (c3 == 0x92) tok = TOK_LOAD;        /* → */
            else if (c3 == 0x93) tok = TOK_REG_WRITE;   /* ↓ */
            if (tok >= 0) {
                emit_tok(tok, (int)i, 3);
                i += 3;
                continue;
            }
        }

        /* single-char operators / punctuation */
        int tok = -1;
        switch (c) {
            case '+': tok = TOK_PLUS;   break;
            case '-': tok = TOK_MINUS;  break;
            case '*': tok = TOK_STAR;   break;
            case '/': tok = TOK_SLASH;  break;
            case '=': tok = TOK_EQ;     break;
            case '<': tok = TOK_LT;     break;
            case '>': tok = TOK_GT;     break;
            case '&': tok = TOK_AMP;    break;
            case '|': tok = TOK_PIPE;   break;
            case '^': tok = TOK_CARET;  break;
            case '(': tok = TOK_LPAREN; break;
            case ')': tok = TOK_RPAREN; break;
            case '[': tok = TOK_LBRACK; break;
            case ']': tok = TOK_RBRACK; break;
            case ',': tok = TOK_COMMA;  break;
            case ':': tok = TOK_COLON;  break;
            case '.': tok = TOK_DOT;    break;
            case '@': tok = TOK_AT;     break;
            default: break;
        }
        if (tok >= 0) { emit_tok(tok, (int)i, 1); i++; continue; }

        /* unknown — skip a byte */
        i++;
    }
    emit_tok(TOK_EOF, (int)i, 0);
}

/* ====================================================================
 * Symbol table
 * ==================================================================== */
static Sym *sym_add(const char *name, int len, int kind, int offset) {
    if (n_syms >= (int)(sizeof syms / sizeof syms[0])) die("sym table full");
    Sym *s = &syms[n_syms++];
    int n = len;
    if (n > (int)sizeof(s->name) - 1) n = (int)sizeof(s->name) - 1;
    memcpy(s->name, name, (size_t)n);
    s->name[n] = 0;
    s->name_len = n;
    s->kind = kind;
    s->offset = offset;
    s->scope = scope_depth;
    return s;
}

static Sym *sym_lookup(const char *name, int len) {
    for (int i = n_syms - 1; i >= 0; i--) {
        if (syms[i].name_len == len &&
            memcmp(syms[i].name, name, (size_t)len) == 0)
            return &syms[i];
    }
    return NULL;
}

/* ====================================================================
 * ARM64 instruction emission
 * ==================================================================== */
static void emit_word(uint32_t w) {
    if (code_len >= (int)(sizeof code / sizeof code[0])) die("code buffer full");
    code[code_len++] = w;
}

/* MOVZ Xd, #imm16, LSL #(shift*16)  — shift ∈ {0,1,2,3} */
static void emit_movz(int rd, uint32_t imm16, int shift) {
    emit_word(0xD2800000u | ((uint32_t)shift << 21) |
              ((imm16 & 0xFFFFu) << 5) | (uint32_t)(rd & 31));
}
static void emit_movk(int rd, uint32_t imm16, int shift) {
    emit_word(0xF2800000u | ((uint32_t)shift << 21) |
              ((imm16 & 0xFFFFu) << 5) | (uint32_t)(rd & 31));
}

/* Materialize a 64-bit immediate into Xd using MOVZ + up to 3 MOVKs.
 * Handles negative values by emitting all non-zero 16-bit chunks. */
static void emit_mov_imm64(int rd, uint64_t imm) {
    int emitted = 0;
    for (int shift = 0; shift < 4; shift++) {
        uint32_t chunk = (uint32_t)((imm >> (shift * 16)) & 0xFFFFu);
        if (!emitted) {
            emit_movz(rd, chunk, shift);
            emitted = 1;
        } else if (chunk) {
            emit_movk(rd, chunk, shift);
        }
    }
}

/* MOV Xd, Xm = ORR Xd, XZR, Xm */
static void emit_mov_reg(int rd, int rm) {
    if (rd == rm) return;
    emit_word(0xAA0003E0u | ((uint32_t)(rm & 31) << 16) | (uint32_t)(rd & 31));
}

/* ADD Xd, Xn, Xm (no shift) */
static void emit_add_reg(int rd, int rn, int rm) {
    emit_word(0x8B000000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
static void emit_sub_reg(int rd, int rn, int rm) {
    emit_word(0xCB000000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
static void emit_and_reg(int rd, int rn, int rm) {
    emit_word(0x8A000000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
static void emit_orr_reg(int rd, int rn, int rm) {
    emit_word(0xAA000000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
static void emit_eor_reg(int rd, int rn, int rm) {
    emit_word(0xCA000000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}

/* MUL Xd, Xn, Xm = MADD Xd, Xn, Xm, XZR */
static void emit_mul_reg(int rd, int rn, int rm) {
    emit_word(0x9B007C00u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
/* SDIV Xd, Xn, Xm */
static void emit_sdiv_reg(int rd, int rn, int rm) {
    emit_word(0x9AC00C00u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}

/* LSLV / LSRV / ASRV */
static void emit_lslv(int rd, int rn, int rm) {
    emit_word(0x9AC02000u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}
static void emit_lsrv(int rd, int rn, int rm) {
    emit_word(0x9AC02400u | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5)  | (uint32_t)(rd & 31));
}

/* NEG Xd, Xn = SUB Xd, XZR, Xn */
static void emit_neg(int rd, int rn) {
    emit_word(0xCB0003E0u | ((uint32_t)(rn & 31) << 16) | (uint32_t)(rd & 31));
}

/* CMP Xn, Xm = SUBS XZR, Xn, Xm */
static void emit_cmp_reg(int rn, int rm) {
    emit_word(0xEB00001Fu | ((uint32_t)(rm & 31) << 16) |
              ((uint32_t)(rn & 31) << 5));
}

/* ADD Xd, Xn, #imm12  (64-bit, no shift) */
static void emit_add_imm(int rd, int rn, int imm12) {
    emit_word(0x91000000u | (((uint32_t)imm12 & 0xFFFu) << 10) |
              ((uint32_t)(rn & 31) << 5) | (uint32_t)(rd & 31));
}
static void emit_sub_imm(int rd, int rn, int imm12) {
    emit_word(0xD1000000u | (((uint32_t)imm12 & 0xFFFu) << 10) |
              ((uint32_t)(rn & 31) << 5) | (uint32_t)(rd & 31));
}

/* LDR Xt, [Xn, #imm12] — unsigned offset, must be multiple of 8 */
static void emit_ldr_ofs(int rt, int rn, int byte_off) {
    int imm12 = byte_off / 8;
    emit_word(0xF9400000u | (((uint32_t)imm12 & 0xFFFu) << 10) |
              ((uint32_t)(rn & 31) << 5) | (uint32_t)(rt & 31));
}
static void emit_str_ofs(int rt, int rn, int byte_off) {
    int imm12 = byte_off / 8;
    emit_word(0xF9000000u | (((uint32_t)imm12 & 0xFFFu) << 10) |
              ((uint32_t)(rn & 31) << 5) | (uint32_t)(rt & 31));
}

/* STP Xa, Xb, [SP, #-16]!  (pre-index, 64-bit pair) */
static void emit_stp_pre_sp(int ra, int rb, int simm7) {
    /* STP: 10 101 0 011 0 imm7 Rt2 Rn Rt  with pre-index variant */
    /* 0xA9800000 base, PRE = bit 23? Actually: STP pre-idx = 0xA9800000 | ... let's be precise. */
    /* 64-bit STP pre-index: 1010 1001 100 imm7 Rt2 Rn Rt = 0xA9800000 | ((imm7/8)&0x7F)<<15 | Rt2<<10 | Rn<<5 | Rt */
    int imm7 = (simm7 / 8) & 0x7F;
    emit_word(0xA9800000u | ((uint32_t)imm7 << 15) |
              ((uint32_t)(rb & 31) << 10) | ((uint32_t)(R_SP & 31) << 5) |
              (uint32_t)(ra & 31));
}
/* LDP Xa, Xb, [SP], #16  (post-index) */
static void emit_ldp_post_sp(int ra, int rb, int simm7) {
    int imm7 = (simm7 / 8) & 0x7F;
    /* 64-bit LDP post-index: 1010 1000 110 imm7 Rt2 Rn Rt = 0xA8C00000 | ... */
    emit_word(0xA8C00000u | ((uint32_t)imm7 << 15) |
              ((uint32_t)(rb & 31) << 10) | ((uint32_t)(R_SP & 31) << 5) |
              (uint32_t)(ra & 31));
}

/* MOV Xd, SP = ADD Xd, SP, #0 */
static void emit_mov_from_sp(int rd) { emit_add_imm(rd, R_SP, 0); }

/* RET */
static void emit_ret(void) { emit_word(0xD65F03C0u); }

/* SVC #0x80  (Darwin syscall) */
static void emit_svc_darwin(void) { emit_word(0xD4001001u); }

/* BL imm26 (word offset, signed) */
static void emit_bl(int word_off) {
    emit_word(0x94000000u | ((uint32_t)word_off & 0x3FFFFFFu));
}
/* B imm26 */
static void emit_b(int word_off) {
    emit_word(0x14000000u | ((uint32_t)word_off & 0x3FFFFFFu));
}
/* B.cond imm19 */
static void emit_b_cond(int cond, int word_off) {
    emit_word(0x54000000u | (((uint32_t)word_off & 0x7FFFFu) << 5) |
              (uint32_t)(cond & 0xF));
}
/* CBZ Xt, imm19 */
static void emit_cbz(int rt, int word_off) {
    emit_word(0xB4000000u | (((uint32_t)word_off & 0x7FFFFu) << 5) |
              (uint32_t)(rt & 31));
}

/* Patch a branch at code[idx] so its offset lands at current code_len. */
static void patch_b_here(int idx) {
    int delta = code_len - idx;
    uint32_t w = code[idx];
    w = (w & 0xFC000000u) | ((uint32_t)delta & 0x3FFFFFFu);
    code[idx] = w;
}
static void patch_b_cond_here(int idx) {
    int delta = code_len - idx;
    uint32_t w = code[idx];
    w = (w & 0xFF00001Fu) | (((uint32_t)delta & 0x7FFFFu) << 5);
    code[idx] = w;
}
static void patch_cbz_here(int idx) {
    int delta = code_len - idx;
    uint32_t w = code[idx];
    w = (w & 0xFF00001Fu) | (((uint32_t)delta & 0x7FFFFu) << 5);
    code[idx] = w;
}

/* ====================================================================
 * Allocator helpers
 * ==================================================================== */
static int alloc_temp(void) {
    if (temp_next > TEMP_HI) die("expression too complex");
    return temp_next++;
}
static void reset_temps(void) { temp_next = TEMP_LO; }

static int alloc_slot(void) {
    int off = frame_next;
    frame_next += 8;
    return off;
}

/* ====================================================================
 * Token stream helpers
 * ==================================================================== */
static int peek(void) { return tokens[tpos].type; }
static const Token *peekt(void) { return &tokens[tpos]; }
static void advance(void) {
    if (tokens[tpos].type != TOK_EOF) tpos++;
}

static void skip_newlines(void) {
    while (peek() == TOK_NEWLINE) advance();
}

/* ====================================================================
 * Parser
 * ==================================================================== */
static int  parse_expr(void);
static void parse_body(int body_indent);
static void parse_statement(void);

/* Parse int literal from a TOK_INT, return its value. Advances x. */
static long parse_int_value(const Token *t) {
    long v = 0;
    int i = t->off;
    int end = t->off + t->len;
    int neg = 0;
    if (i < end && src[i] == '-') { neg = 1; i++; }
    if (end - i >= 2 && src[i] == '0' &&
        (src[i + 1] == 'x' || src[i + 1] == 'X')) {
        i += 2;
        while (i < end) {
            int c = (unsigned char)src[i];
            int d;
            if      (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else break;
            v = v * 16 + d;
            i++;
        }
    } else {
        while (i < end && src[i] >= '0' && src[i] <= '9') {
            v = v * 10 + (src[i] - '0');
            i++;
        }
    }
    return neg ? -v : v;
}

/* Return the dollar-register number from a TOK_IDENT starting with '$', or -1 */
static int dollar_reg(const Token *t) {
    if (t->len < 2 || src[t->off] != '$') return -1;
    int n = 0;
    for (int i = 1; i < t->len; i++) {
        if (src[t->off + i] < '0' || src[t->off + i] > '9') return -1;
        n = n * 10 + (src[t->off + i] - '0');
    }
    return n;
}

/* ===== parse_atom =====
 * Consume one atom (literal, ident, paren expr, unary -).
 * Returns the register holding the result — always an expression temp. */
static int parse_atom(void) {
    const Token *t = peekt();

    if (t->type == TOK_INT) {
        long v = parse_int_value(t);
        advance();
        int r = alloc_temp();
        emit_mov_imm64(r, (uint64_t)v);
        return r;
    }

    if (t->type == TOK_LPAREN) {
        advance();
        int r = parse_expr();
        if (peek() != TOK_RPAREN) die("expected )");
        advance();
        return r;
    }

    if (t->type == TOK_MINUS) {
        advance();
        int r = parse_atom();
        emit_neg(r, r);
        return r;
    }

    /* Identifier — local/param read, or zero-arg call, or unknown forward ref */
    if (t->type == TOK_IDENT || (t->type >= 11 && t->type <= 45)) {
        /* Handle $N specially: it's a direct physical register number */
        int dn = dollar_reg(t);
        if (dn >= 0) {
            advance();
            /* Read the physical register into an expression temp */
            int r = alloc_temp();
            emit_mov_reg(r, dn);
            return r;
        }

        const char *name = &src[t->off];
        int len = t->len;
        Sym *s = sym_lookup(name, len);
        advance();

        if (s && (s->kind == KIND_LOCAL || s->kind == KIND_PARAM)) {
            int r = alloc_temp();
            emit_ldr_ofs(r, R_SP, s->offset);
            return r;
        }
        if (s && s->kind == KIND_COMP) {
            /* Expression-level call with NO arguments. For call-with-args
             * the statement parser handles it. */
            int cur = code_len;
            int delta = s->offset - cur;
            emit_bl(delta);
            int r = alloc_temp();
            emit_mov_reg(r, 0);   /* result is in X0 */
            return r;
        }
        /* Unknown — emit a zero placeholder, no-op behaviour */
        int r = alloc_temp();
        emit_mov_imm64(r, 0);
        return r;
    }

    fprintf(stderr, "parse_atom: unexpected token type %d at offset %d\n",
            t->type, t->off);
    exit(1);
}

/* ===== parse_muldiv / parse_addsub / parse_shift / parse_bitwise / parse_expr ===== */
static int parse_muldiv(void) {
    int l = parse_atom();
    for (;;) {
        int op = peek();
        if (op != TOK_STAR && op != TOK_SLASH) break;
        advance();
        int r = parse_atom();
        if      (op == TOK_STAR)  emit_mul_reg(l, l, r);
        else                      emit_sdiv_reg(l, l, r);
    }
    return l;
}

static int parse_addsub(void) {
    int l = parse_muldiv();
    for (;;) {
        int op = peek();
        if (op != TOK_PLUS && op != TOK_MINUS) break;
        advance();
        int r = parse_muldiv();
        if (op == TOK_PLUS) emit_add_reg(l, l, r);
        else                 emit_sub_reg(l, l, r);
    }
    return l;
}

static int parse_shift(void) {
    int l = parse_addsub();
    for (;;) {
        int op = peek();
        if (op != TOK_SHL && op != TOK_SHR) break;
        advance();
        int r = parse_addsub();
        if (op == TOK_SHL) emit_lslv(l, l, r);
        else               emit_lsrv(l, l, r);
    }
    return l;
}

static int parse_bitwise(void) {
    int l = parse_shift();
    for (;;) {
        int op = peek();
        if (op != TOK_AMP && op != TOK_PIPE && op != TOK_CARET) break;
        advance();
        int r = parse_shift();
        if      (op == TOK_AMP)   emit_and_reg(l, l, r);
        else if (op == TOK_PIPE)  emit_orr_reg(l, l, r);
        else                      emit_eor_reg(l, l, r);
    }
    return l;
}

static int parse_expr(void) {
    /* Full expression. We don't do comparisons here — those live in
     * the if/while statement parsers where they are consumed directly. */
    return parse_bitwise();
}

/* Look up "main" in the sym table after parsing.  Returns -1 if absent. */
static int find_main_code_offset(void) {
    Sym *s = sym_lookup("main", 4);
    return (s && s->kind == KIND_COMP) ? s->offset : -1;
}

/* Parse args for a statement-level call to a known composition. */
static void parse_call_and_emit(Sym *comp) {
    int arg_num = 0;
    while (peek() != TOK_NEWLINE && peek() != TOK_EOF &&
           peek() != TOK_INDENT  && arg_num < 8) {
        /* Stop at stray operators/punctuation that cannot start an atom. */
        int p = peek();
        if (p == TOK_RPAREN || p == TOK_COMMA) break;
        int r = parse_expr();
        if (r != arg_num) emit_mov_reg(arg_num, r);
        arg_num++;
    }
    int cur = code_len;
    emit_bl(comp->offset - cur);
}

/* Parse a compound conditional ("<", ">", "==", …) inside if/while.
 * Assumes the caller has already consumed the comparison token.
 * Emits CMP + inverted B.cond placeholder; returns the placeholder idx. */
static int parse_cond_branch(int cmp_tok) {
    int l = parse_expr();
    int r = parse_expr();
    emit_cmp_reg(l, r);
    int inv;
    switch (cmp_tok) {
        case TOK_LT:   inv = CC_GE; break;
        case TOK_GT:   inv = CC_LE; break;
        case TOK_LTE:  inv = CC_GT; break;
        case TOK_GTE:  inv = CC_LT; break;
        case TOK_EQEQ: inv = CC_NE; break;
        case TOK_NEQ:  inv = CC_EQ; break;
        default:       inv = CC_NE; break;
    }
    int idx = code_len;
    emit_b_cond(inv, 0);
    return idx;
}

/* Expect: if [compound] body [else body] */
static void parse_if(void) {
    advance();  /* past 'if' */
    int is_compound = 0;
    int cmp_tok = 0;
    int tk = peek();
    if (tk == TOK_LT || tk == TOK_GT || tk == TOK_LTE || tk == TOK_GTE ||
        tk == TOK_EQEQ || tk == TOK_NEQ) {
        is_compound = 1;
        cmp_tok = tk;
        advance();
    }

    int skip_idx;
    int is_cbz = 0;
    if (is_compound) {
        skip_idx = parse_cond_branch(cmp_tok);
    } else {
        int r = parse_expr();
        skip_idx = code_len;
        emit_cbz(r, 0);
        is_cbz = 1;
    }

    reset_temps();
    skip_newlines();

    /* Determine body indent from the next INDENT token, if any. */
    int body_indent = 0;
    if (peek() == TOK_INDENT) body_indent = peekt()->len;
    parse_body(body_indent);

    /* If there's an else clause, we need a jump past it. */
    int has_else = 0;
    int saved_skip_idx = skip_idx;
    int saved_is_cbz = is_cbz;
    int end_idx = -1;

    skip_newlines();
    if (peek() == TOK_ELSE) {
        has_else = 1;
        /* Emit unconditional jump past the else body */
        end_idx = code_len;
        emit_b(0);
        /* Patch the original skip to land here (start of else) */
        if (saved_is_cbz) patch_cbz_here(saved_skip_idx);
        else              patch_b_cond_here(saved_skip_idx);
        advance();  /* past 'else' */
        skip_newlines();
        int else_indent = 0;
        if (peek() == TOK_INDENT) else_indent = peekt()->len;
        parse_body(else_indent);
        patch_b_here(end_idx);
    } else {
        if (saved_is_cbz) patch_cbz_here(saved_skip_idx);
        else              patch_b_cond_here(saved_skip_idx);
    }
    (void)has_else;
}

/* Expect: while <expr> : body   (simple truthy test) */
static void parse_while(void) {
    advance();  /* past 'while' */
    int loop_top = code_len;

    int tk = peek();
    int skip_idx;
    int is_cbz = 0;
    if (tk == TOK_LT || tk == TOK_GT || tk == TOK_LTE || tk == TOK_GTE ||
        tk == TOK_EQEQ || tk == TOK_NEQ) {
        advance();
        skip_idx = parse_cond_branch(tk);
    } else {
        int r = parse_expr();
        skip_idx = code_len;
        emit_cbz(r, 0);
        is_cbz = 1;
    }

    reset_temps();
    skip_newlines();
    int body_indent = 0;
    if (peek() == TOK_INDENT) body_indent = peekt()->len;
    parse_body(body_indent);

    /* Jump back to top */
    emit_b(loop_top - code_len);
    /* Patch exit branch to here */
    if (is_cbz) patch_cbz_here(skip_idx);
    else        patch_b_cond_here(skip_idx);
}

/* Expect: for i start end [step] : body */
static void parse_for(void) {
    advance();  /* past 'for' */
    const Token *nt = peekt();
    if (nt->type != TOK_IDENT) die("expected loop variable after 'for'");
    const char *name = &src[nt->off];
    int len = nt->len;
    advance();

    int i_slot = alloc_slot();
    sym_add(name, len, KIND_LOCAL, i_slot);

    /* start */
    int start_r = parse_expr();
    emit_str_ofs(start_r, R_SP, i_slot);
    reset_temps();

    /* end */
    int end_r = parse_expr();
    int end_slot = alloc_slot();
    emit_str_ofs(end_r, R_SP, end_slot);
    reset_temps();

    /* optional step */
    int step_slot = -1;
    if (peek() != TOK_NEWLINE && peek() != TOK_INDENT && peek() != TOK_EOF) {
        int step_r = parse_expr();
        step_slot = alloc_slot();
        emit_str_ofs(step_r, R_SP, step_slot);
        reset_temps();
    }

    skip_newlines();

    int loop_top = code_len;
    int iv = alloc_temp();
    int ev = alloc_temp();
    emit_ldr_ofs(iv, R_SP, i_slot);
    emit_ldr_ofs(ev, R_SP, end_slot);
    emit_cmp_reg(iv, ev);
    int exit_idx = code_len;
    emit_b_cond(CC_GE, 0);
    reset_temps();

    int body_indent = 0;
    if (peek() == TOK_INDENT) body_indent = peekt()->len;
    parse_body(body_indent);

    /* increment */
    int iv2 = alloc_temp();
    emit_ldr_ofs(iv2, R_SP, i_slot);
    if (step_slot >= 0) {
        int sv = alloc_temp();
        emit_ldr_ofs(sv, R_SP, step_slot);
        emit_add_reg(iv2, iv2, sv);
    } else {
        emit_add_imm(iv2, iv2, 1);
    }
    emit_str_ofs(iv2, R_SP, i_slot);
    reset_temps();

    emit_b(loop_top - code_len);
    patch_b_cond_here(exit_idx);
}

/* parse_statement: one statement at the current indent level.
 * Consumes everything up to (not including) the next NEWLINE/EOF/INDENT. */
static void parse_statement(void) {
    const Token *t = peekt();

    if (t->type == TOK_TRAP) {
        advance();
        emit_svc_darwin();
        return;
    }

    if (t->type == TOK_REG_WRITE) {
        /* ↓ $N value */
        advance();
        const Token *rt = peekt();
        int dn = (rt->type == TOK_IDENT) ? dollar_reg(rt) : -1;
        if (dn < 0) die("expected $N after ↓");
        advance();
        /* If the value is a literal, emit MOVZ/MOVK directly into the physical reg */
        if (peek() == TOK_INT) {
            long v = parse_int_value(peekt());
            advance();
            emit_mov_imm64(dn, (uint64_t)v);
            return;
        }
        int vr = parse_expr();
        if (vr != dn) emit_mov_reg(dn, vr);
        return;
    }

    if (t->type == TOK_REG_READ) {
        /* ↑ $N — bare read, put value into X0 */
        advance();
        const Token *rt = peekt();
        int dn = (rt->type == TOK_IDENT) ? dollar_reg(rt) : -1;
        if (dn < 0) die("expected $N after ↑");
        advance();
        emit_mov_reg(0, dn);
        return;
    }

    if (t->type == TOK_VAR) {
        advance();
        const Token *nt = peekt();
        if (nt->type != TOK_IDENT && (nt->type < 11 || nt->type > 45))
            die("expected name after 'var'");
        const char *name = &src[nt->off];
        int len = nt->len;
        advance();
        int slot = alloc_slot();
        sym_add(name, len, KIND_LOCAL, slot);
        if (peek() == TOK_NEWLINE || peek() == TOK_EOF || peek() == TOK_INDENT) {
            int r = alloc_temp();
            emit_mov_imm64(r, 0);
            emit_str_ofs(r, R_SP, slot);
        } else {
            int r = parse_expr();
            emit_str_ofs(r, R_SP, slot);
        }
        return;
    }

    if (t->type == TOK_IF)    { parse_if();    return; }
    if (t->type == TOK_WHILE) { parse_while(); return; }
    if (t->type == TOK_FOR)   { parse_for();   return; }

    if (t->type == TOK_RETURN) {
        advance();
        if (peek() != TOK_NEWLINE && peek() != TOK_EOF && peek() != TOK_INDENT) {
            int r = parse_expr();
            if (r != 0) emit_mov_reg(0, r);
        }
        /* We don't emit the epilogue here — composition end does. A RETURN
         * mid-function would need its own epilogue emission. Keep simple for now. */
        return;
    }

    /* Identifier at statement start — binding, reassignment, or call */
    if (t->type == TOK_IDENT || (t->type >= 11 && t->type <= 45)) {
        const char *name = &src[t->off];
        int len = t->len;
        Sym *s = sym_lookup(name, len);

        if (s && s->kind == KIND_COMP) {
            advance();
            parse_call_and_emit(s);
            return;
        }
        if (s && (s->kind == KIND_LOCAL || s->kind == KIND_PARAM)) {
            /* Reassignment: name expr */
            advance();
            if (peek() == TOK_NEWLINE || peek() == TOK_EOF || peek() == TOK_INDENT) {
                /* bare read of a local as a statement — no effect */
                return;
            }
            int vr = parse_expr();
            emit_str_ofs(vr, R_SP, s->offset);
            return;
        }

        /* New name — if something follows, it's a binding: name expr */
        advance();
        if (peek() == TOK_NEWLINE || peek() == TOK_EOF || peek() == TOK_INDENT) {
            /* Bare call to unknown — forward reference or stub, emit nothing */
            return;
        }
        int vr = parse_expr();
        int slot = alloc_slot();
        sym_add(name, len, KIND_LOCAL, slot);
        emit_str_ofs(vr, R_SP, slot);
        return;
    }

    /* Unknown — skip one token */
    advance();
}

/* Parse a body whose statements are at the given indent level.
 * Stops when we see EOF or an INDENT shallower than body_indent. */
static void parse_body(int body_indent) {
    int observed = -1;
    while (peek() != TOK_EOF) {
        if (peek() == TOK_NEWLINE) { advance(); continue; }
        if (peek() == TOK_INDENT) {
            int lvl = peekt()->len;
            if (observed == -1) observed = lvl;
            if (lvl < body_indent || lvl < observed) return;
            advance();
            continue;
        }
        /* First token of a statement — must be after an indent OR at column 0 */
        if (observed == -1 && body_indent > 0) {
            /* We haven't seen any indent yet — if we're a nested body, we're
             * empty. Bail out. */
            return;
        }
        parse_statement();
        reset_temps();
    }
}

/* Emit composition prologue and spill params to stack. */
#define FRAME_BYTES  512

static void parse_composition(void) {
    /* Composition header: name [args]* : */
    const Token *t = peekt();
    if (t->type != TOK_IDENT && (t->type < 11 || t->type > 45))
        die("expected composition name");

    const char *name = &src[t->off];
    int len = t->len;
    int code_addr = code_len;
    sym_add(name, len, KIND_COMP, code_addr);
    advance();

    int saved_syms = n_syms;
    scope_depth++;
    frame_next = 0;
    temp_next = TEMP_LO;

    /* Collect param names until ':' or newline */
    int n_args = 0;
    const char *arg_name[8];
    int         arg_len [8];
    while (peek() != TOK_COLON && peek() != TOK_NEWLINE && peek() != TOK_EOF) {
        const Token *at = peekt();
        if (at->type == TOK_IDENT || (at->type >= 11 && at->type <= 45)) {
            if (n_args < 8) {
                arg_name[n_args] = &src[at->off];
                arg_len [n_args] = at->len;
                n_args++;
            }
            advance();
        } else {
            advance();
        }
    }
    if (peek() == TOK_COLON) advance();
    skip_newlines();

    /* Prologue */
    emit_stp_pre_sp(R_FP, R_LR, -16);
    emit_mov_from_sp(R_FP);
    emit_sub_imm(R_SP, R_SP, FRAME_BYTES);

    /* Spill params X0..X(N-1) into stack slots, add sym entries */
    for (int i = 0; i < n_args; i++) {
        int slot = alloc_slot();
        emit_str_ofs(i, R_SP, slot);
        sym_add(arg_name[i], arg_len[i], KIND_PARAM, slot);
    }

    /* Body */
    int body_indent = 0;
    if (peek() == TOK_INDENT) body_indent = peekt()->len;
    parse_body(body_indent);

    /* Epilogue */
    emit_add_imm(R_SP, R_SP, FRAME_BYTES);
    emit_ldp_post_sp(R_FP, R_LR, 16);
    emit_ret();

    /* Pop locals/params, keep the composition symbol */
    n_syms = saved_syms;
    scope_depth--;
}

static void parse_file(void) {
    while (peek() != TOK_EOF) {
        int p = peek();
        if (p == TOK_NEWLINE || p == TOK_INDENT) { advance(); continue; }
        parse_composition();
    }
}

/* ====================================================================
 * Output: Mach-O via as + ld
 *
 * Modern macOS (arm64) requires executables to link against libSystem
 * and bear a valid code signature. Rather than hand-rolling that
 * complexity (LC_LOAD_DYLINKER, LC_LOAD_DYLIB, LC_MAIN, __LINKEDIT,
 * dyld info, code signature) in a bootstrap, we emit our code bytes
 * as an assembly file and shell out to the system as(1) + ld(1), which
 * handle every wart of the Mach-O format correctly. The machine code
 * we emit is untouched — only the ELF/Mach-O wrapper changes.
 * ==================================================================== */
static int write_executable(const char *out_path, int main_code_idx) {
    char tmp_s[256], tmp_o[256];
    snprintf(tmp_s, sizeof tmp_s, "/tmp/lithos-linus-%d.s", (int)getpid());
    snprintf(tmp_o, sizeof tmp_o, "/tmp/lithos-linus-%d.o", (int)getpid());

    FILE *f = fopen(tmp_s, "w");
    if (!f) { perror(tmp_s); return -1; }
    fprintf(f, "// auto-generated by lithos-boot-linus\n");
    fprintf(f, ".global _start\n");
    fprintf(f, ".text\n");
    fprintf(f, ".align 4\n");
    fprintf(f, "_start:\n");
    /* main composition must live at word main_code_idx. If it isn't word 0,
     * emit a jump at _start to it. In practice the main-first invariant
     * holds (we sym_add 'main' when we encounter it, and parse_file processes
     * compositions top to bottom), but be defensive. */
    if (main_code_idx != 0) {
        /* Emit a branch from _start to main's word offset. We can't know
         * the absolute address here, so we insert a relative B. The assembler
         * will encode it. */
        fprintf(f, "    b main_start\n");
    }
    /* Now dump all our code words. Label 'main_start' at position
     * main_code_idx. */
    for (int i = 0; i < code_len; i++) {
        if (i == main_code_idx && main_code_idx != 0) fprintf(f, "main_start:\n");
        fprintf(f, "    .word 0x%08x\n", code[i]);
    }
    if (main_code_idx == 0) {
        /* Not strictly necessary but helps debuggers find main */
    }
    fclose(f);

    /* Locate SDK */
    FILE *pf = popen("xcrun --sdk macosx --show-sdk-path 2>/dev/null", "r");
    if (!pf) { fprintf(stderr, "popen xcrun failed\n"); return -1; }
    char sdk[1024];
    if (!fgets(sdk, sizeof sdk, pf)) { pclose(pf); fprintf(stderr, "xcrun returned nothing\n"); return -1; }
    pclose(pf);
    /* strip newline */
    size_t sl = strlen(sdk);
    while (sl && (sdk[sl - 1] == '\n' || sdk[sl - 1] == '\r')) sdk[--sl] = 0;

    /* Assemble */
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "as -arch arm64 -o '%s' '%s'", tmp_o, tmp_s);
    if (system(cmd) != 0) { fprintf(stderr, "as failed\n"); return -1; }

    /* Link against libSystem */
    snprintf(cmd, sizeof cmd,
             "ld -arch arm64 -o '%s' -e _start "
             "-platform_version macos 15.0.0 15.0.0 "
             "-lSystem -L'%s/usr/lib' "
             "'%s' 2>/dev/null",
             out_path, sdk, tmp_o);
    if (system(cmd) != 0) { fprintf(stderr, "ld failed\n"); return -1; }

    /* Cleanup */
    unlink(tmp_s);
    unlink(tmp_o);
    return 0;
}

/* ====================================================================
 * main
 * ==================================================================== */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s input.ls output\n", argv[0]);
        return 1;
    }
    src_path = argv[1];
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    src_len = (size_t)st.st_size;
    src = mmap(NULL, src_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (src == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    close(fd);

    lex();
    tpos = 0;
    parse_file();

    int main_idx = find_main_code_offset();
    if (main_idx < 0) {
        fprintf(stderr, "lithos-boot-linus: no 'main' composition found\n");
        return 1;
    }

    if (write_executable(argv[2], main_idx) < 0) {
        fprintf(stderr, "lithos-boot-linus: failed to write output\n");
        return 1;
    }
    fprintf(stderr, "lithos-boot-linus: wrote %s (%d instructions, main @ word %d)\n",
            argv[2], code_len, main_idx);
    return 0;
}
