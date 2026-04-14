/*
 * lithos-boot-ken.c -- a minimal C bootstrap compiler for Lithos.
 *
 * Author: "ken"
 *
 * Reads a .ls source file, tokenizes it, parses top-level compositions,
 * emits ARM64 machine code directly into a buffer, and writes a macOS
 * ARM64 Mach-O executable (LC_UNIXTHREAD-style, no dyld, no dylibs).
 *
 * Architectural notes:
 *   - Single-pass recursive descent. No AST. Emit as we parse.
 *   - All locals (bindings, vars, params) live at fixed offsets from x29.
 *     No register allocator that spills to the target program; expression
 *     temps are a bump allocator over x9..x18 that resets at statement
 *     boundaries.
 *   - Compositions get a fixed-size frame (512 bytes). Slots are assigned
 *     at declaration; deepest slot used determines frame high-water mark
 *     but we just use the full FRAME_SIZE to keep it trivial.
 *   - Parameters (x0..x7 on entry) are spilled to stack slots at the top
 *     of the prologue so the rest of the body treats them like any local.
 *   - The target is macOS syscalls. The test .ls files use Linux exit
 *     (x8 = 93) so the emitter translates: if the user writes `<DOWN> $8 N`
 *     we also load x16 with the Darwin equivalent (93 -> 1 for exit,
 *     otherwise passthrough). `trap` emits SVC #0x80.
 *
 * Supported features (v1):
 *   - Compositions with 0..8 args and nested bodies
 *   - Integer literals (decimal, 0x hex, negative)
 *   - Identifiers (read/write locals, params, bindings)
 *   - Bindings (`name expr`), reassignment, `var name expr`
 *   - Arithmetic: + - * / & | ^ << >> (left-assoc, basic precedence)
 *   - Parens
 *   - `<DOWN> $N val` (register write), `<UP> $N` (register read)
 *   - `trap` (SVC #0x80)
 *   - `if` compound (if< if> if== if!= if<= if>=) with `else`
 *   - Simple if via CBZ
 *   - `for i start end [step] : body`
 *   - `while expr : body`
 *   - Function calls (BL to known compositions)
 *   - Mem load/store (<DOWN>/<UP> with non-$ address)  [minimal]
 *
 * Build:  cc -O2 -Wall -Wextra -o lithos-boot-ken lithos-boot-ken.c
 * Usage:  lithos-boot-ken input.ls output-executable
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* ============================================================ */
/*  Debug                                                        */
/* ============================================================ */
static int g_debug = 0;
static void DBG(const char *fmt, ...) {
    if (!g_debug) return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[ken] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* ============================================================ */
/*  Token types                                                  */
/* ============================================================ */
enum {
    T_EOF = 0,
    T_NEWLINE,
    T_INDENT,
    T_INT,
    T_IDENT,
    T_KW_IF,
    T_KW_ELSE,
    T_KW_ELIF,
    T_KW_FOR,
    T_KW_WHILE,
    T_KW_EACH,
    T_KW_VAR,
    T_KW_RETURN,
    T_KW_TRAP,
    T_REG_WRITE,  /* the down-arrow */
    T_REG_READ,   /* the up-arrow   */
    T_PLUS,
    T_MINUS,
    T_STAR,
    T_SLASH,
    T_AMP,
    T_PIPE,
    T_CARET,
    T_SHL,
    T_SHR,
    T_EQ,
    T_EQEQ,
    T_NEQ,
    T_LT,
    T_GT,
    T_LTE,
    T_GTE,
    T_LPAREN,
    T_RPAREN,
    T_COLON,
    T_DOLLAR,     /* $ — used only in $N register refs */
    T_COMMA
};

typedef struct {
    int type;
    int line;
    int64_t ival;   /* for T_INT     */
    const char *s;  /* for T_IDENT — not NUL-terminated */
    int slen;
} Token;

/* ============================================================ */
/*  Source state                                                 */
/* ============================================================ */
static char *g_src = NULL;
static size_t g_src_len = 0;
static Token *g_tok = NULL;
static int g_ntok = 0;
static int g_cap_tok = 0;
static int g_tk = 0;    /* current token index */

static void die(const char *msg, int line) {
    fprintf(stderr, "lithos-boot-ken: error at line %d: %s\n", line, msg);
    exit(1);
}

/* ============================================================ */
/*  Lexer                                                        */
/* ============================================================ */
static void tok_push(int type, int line, int64_t ival, const char *s, int slen) {
    if (g_ntok >= g_cap_tok) {
        g_cap_tok = g_cap_tok ? g_cap_tok * 2 : 1024;
        g_tok = (Token *)realloc(g_tok, g_cap_tok * sizeof(Token));
    }
    Token *t = &g_tok[g_ntok++];
    t->type = type; t->line = line; t->ival = ival; t->s = s; t->slen = slen;
}

static int kw_match(const char *s, int len, const char *kw) {
    int klen = (int)strlen(kw);
    if (len != klen) return 0;
    return memcmp(s, kw, klen) == 0;
}

static int is_ident_start(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int is_ident_cont(int c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}
static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_hexdigit(int c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static void lex(void) {
    size_t i = 0;
    int line = 1;
    int at_line_start = 1;
    while (i < g_src_len) {
        unsigned char c = (unsigned char)g_src[i];
        if (at_line_start) {
            /* Compute indent */
            int indent = 0;
            while (i < g_src_len && (g_src[i] == ' ' || g_src[i] == '\t')) {
                if (g_src[i] == '\t') indent += 8; else indent += 1;
                i++;
            }
            tok_push(T_INDENT, line, indent, NULL, 0);
            at_line_start = 0;
            continue;
        }
        if (c == '\n') {
            tok_push(T_NEWLINE, line, 0, NULL, 0);
            line++;
            i++;
            at_line_start = 1;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r') { i++; continue; }
        /* Line comment `\\` */
        if (c == '\\' && i + 1 < g_src_len && g_src[i+1] == '\\') {
            while (i < g_src_len && g_src[i] != '\n') i++;
            continue;
        }
        /* UTF-8 arrows: DOWN (E2 86 93) and UP (E2 86 91) */
        if (c == 0xE2 && i + 2 < g_src_len) {
            unsigned char c1 = (unsigned char)g_src[i+1];
            unsigned char c2 = (unsigned char)g_src[i+2];
            if (c1 == 0x86 && c2 == 0x93) {
                tok_push(T_REG_WRITE, line, 0, NULL, 0);
                i += 3; continue;
            }
            if (c1 == 0x86 && c2 == 0x91) {
                tok_push(T_REG_READ, line, 0, NULL, 0);
                i += 3; continue;
            }
            /* Unknown UTF-8 — skip first byte defensively */
            i++; continue;
        }
        /* Numeric literal */
        if (is_digit(c)) {
            int64_t v = 0;
            if (c == '0' && i + 1 < g_src_len && (g_src[i+1] == 'x' || g_src[i+1] == 'X')) {
                i += 2;
                while (i < g_src_len && is_hexdigit((unsigned char)g_src[i])) {
                    unsigned char cc = (unsigned char)g_src[i];
                    int d = is_digit(cc) ? cc - '0' :
                            (cc >= 'a' && cc <= 'f') ? cc - 'a' + 10 : cc - 'A' + 10;
                    v = v * 16 + d;
                    i++;
                }
            } else {
                while (i < g_src_len && is_digit((unsigned char)g_src[i])) {
                    v = v * 10 + (g_src[i] - '0');
                    i++;
                }
            }
            tok_push(T_INT, line, v, NULL, 0);
            continue;
        }
        /* Identifier / keyword */
        if (is_ident_start(c)) {
            size_t start = i;
            while (i < g_src_len && is_ident_cont((unsigned char)g_src[i])) i++;
            int len = (int)(i - start);
            const char *s = &g_src[start];
            int tt = T_IDENT;
            if      (kw_match(s, len, "if"))    tt = T_KW_IF;
            else if (kw_match(s, len, "else"))  tt = T_KW_ELSE;
            else if (kw_match(s, len, "elif"))  tt = T_KW_ELIF;
            else if (kw_match(s, len, "for"))   tt = T_KW_FOR;
            else if (kw_match(s, len, "while")) tt = T_KW_WHILE;
            else if (kw_match(s, len, "each"))  tt = T_KW_EACH;
            else if (kw_match(s, len, "var"))   tt = T_KW_VAR;
            else if (kw_match(s, len, "return"))tt = T_KW_RETURN;
            else if (kw_match(s, len, "trap"))  tt = T_KW_TRAP;
            tok_push(tt, line, 0, s, len);
            continue;
        }
        /* Punctuation / operators */
        switch (c) {
        case '+': tok_push(T_PLUS, line, 0, NULL, 0); i++; continue;
        case '-': tok_push(T_MINUS, line, 0, NULL, 0); i++; continue;
        case '*': tok_push(T_STAR, line, 0, NULL, 0); i++; continue;
        case '/': tok_push(T_SLASH, line, 0, NULL, 0); i++; continue;
        case '&': tok_push(T_AMP, line, 0, NULL, 0); i++; continue;
        case '|': tok_push(T_PIPE, line, 0, NULL, 0); i++; continue;
        case '^': tok_push(T_CARET, line, 0, NULL, 0); i++; continue;
        case '(': tok_push(T_LPAREN, line, 0, NULL, 0); i++; continue;
        case ')': tok_push(T_RPAREN, line, 0, NULL, 0); i++; continue;
        case ':': tok_push(T_COLON, line, 0, NULL, 0); i++; continue;
        case ',': tok_push(T_COMMA, line, 0, NULL, 0); i++; continue;
        case '$': tok_push(T_DOLLAR, line, 0, NULL, 0); i++; continue;
        case '<':
            if (i+1 < g_src_len && g_src[i+1] == '<') { tok_push(T_SHL, line, 0, NULL, 0); i += 2; continue; }
            if (i+1 < g_src_len && g_src[i+1] == '=') { tok_push(T_LTE, line, 0, NULL, 0); i += 2; continue; }
            tok_push(T_LT, line, 0, NULL, 0); i++; continue;
        case '>':
            if (i+1 < g_src_len && g_src[i+1] == '>') { tok_push(T_SHR, line, 0, NULL, 0); i += 2; continue; }
            if (i+1 < g_src_len && g_src[i+1] == '=') { tok_push(T_GTE, line, 0, NULL, 0); i += 2; continue; }
            tok_push(T_GT, line, 0, NULL, 0); i++; continue;
        case '=':
            if (i+1 < g_src_len && g_src[i+1] == '=') { tok_push(T_EQEQ, line, 0, NULL, 0); i += 2; continue; }
            tok_push(T_EQ, line, 0, NULL, 0); i++; continue;
        case '!':
            if (i+1 < g_src_len && g_src[i+1] == '=') { tok_push(T_NEQ, line, 0, NULL, 0); i += 2; continue; }
            i++; continue;
        }
        /* Unknown char — skip */
        i++;
    }
    tok_push(T_EOF, line, 0, NULL, 0);
}

/* ============================================================ */
/*  Code emission                                                */
/* ============================================================ */
#define CODE_MAX (1u << 20)
static uint8_t *g_code = NULL;
static uint32_t g_code_pos = 0;

static void emit32(uint32_t inst) {
    if (g_code_pos + 4 > CODE_MAX) { fprintf(stderr, "code overflow\n"); exit(1); }
    g_code[g_code_pos + 0] = (inst >>  0) & 0xff;
    g_code[g_code_pos + 1] = (inst >>  8) & 0xff;
    g_code[g_code_pos + 2] = (inst >> 16) & 0xff;
    g_code[g_code_pos + 3] = (inst >> 24) & 0xff;
    g_code_pos += 4;
}

static uint32_t code_here(void) { return g_code_pos; }

/* MOVZ Xd, #imm16, LSL #(shift*16)  shift in {0,1,2,3} */
static void emit_movz(int rd, uint32_t imm16, int shift) {
    uint32_t inst = 0xD2800000u | ((uint32_t)(shift & 3) << 21) | ((imm16 & 0xffff) << 5) | (rd & 0x1f);
    emit32(inst);
}

/* MOVK Xd, #imm16, LSL #(shift*16) */
static void emit_movk(int rd, uint32_t imm16, int shift) {
    uint32_t inst = 0xF2800000u | ((uint32_t)(shift & 3) << 21) | ((imm16 & 0xffff) << 5) | (rd & 0x1f);
    emit32(inst);
}

/* MOVN Xd, #imm16 */
static void emit_movn(int rd, uint32_t imm16, int shift) {
    uint32_t inst = 0x92800000u | ((uint32_t)(shift & 3) << 21) | ((imm16 & 0xffff) << 5) | (rd & 0x1f);
    emit32(inst);
}

/* MOV Xd, imm64 via MOVZ/MOVK sequence, handling negatives compactly */
static void emit_mov_imm64(int rd, int64_t v) {
    uint64_t u = (uint64_t)v;
    if (v >= 0) {
        int started = 0;
        for (int s = 0; s < 4; s++) {
            uint32_t chunk = (uint32_t)((u >> (s * 16)) & 0xffff);
            if (chunk == 0 && started) continue;
            if (!started) { emit_movz(rd, chunk, s); started = 1; }
            else emit_movk(rd, chunk, s);
        }
        if (!started) emit_movz(rd, 0, 0);
    } else {
        /* Use MOVN for low chunk that has the most bits set, then MOVK */
        uint64_t nu = ~u;
        /* Simple: MOVN rd, #(~u & 0xffff), shift 0; then MOVK for the rest */
        emit_movn(rd, (uint32_t)(nu & 0xffff), 0);
        for (int s = 1; s < 4; s++) {
            uint32_t chunk = (uint32_t)((u >> (s * 16)) & 0xffff);
            /* If it equals 0xffff the MOVN already set it, skip; else MOVK */
            if (chunk == 0xffff) continue;
            emit_movk(rd, chunk, s);
        }
    }
}

/* MOV Xd, Xm  (ORR Xd, XZR, Xm) */
static void emit_mov_reg(int rd, int rm) {
    if (rd == rm) return;
    uint32_t inst = 0xAA0003E0u | ((rm & 0x1f) << 16) | (rd & 0x1f);
    emit32(inst);
}

/* ADD Xd, Xn, Xm */
static void emit_add_reg(int rd, int rn, int rm) {
    emit32(0x8B000000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* SUB Xd, Xn, Xm */
static void emit_sub_reg(int rd, int rn, int rm) {
    emit32(0xCB000000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* MUL Xd, Xn, Xm */
static void emit_mul_reg(int rd, int rn, int rm) {
    emit32(0x9B007C00u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* SDIV Xd, Xn, Xm */
static void emit_sdiv_reg(int rd, int rn, int rm) {
    emit32(0x9AC00C00u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* AND Xd, Xn, Xm */
static void emit_and_reg(int rd, int rn, int rm) {
    emit32(0x8A000000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* ORR Xd, Xn, Xm */
static void emit_orr_reg(int rd, int rn, int rm) {
    emit32(0xAA000000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* EOR Xd, Xn, Xm */
static void emit_eor_reg(int rd, int rn, int rm) {
    emit32(0xCA000000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* LSL variable: LSL Xd, Xn, Xm  (LSLV) */
static void emit_lslv(int rd, int rn, int rm) {
    emit32(0x9AC02000u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* LSR variable */
static void emit_lsrv(int rd, int rn, int rm) {
    emit32(0x9AC02400u | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* NEG Xd, Xm  (SUB Xd, XZR, Xm) */
static void emit_neg_reg(int rd, int rm) {
    emit_sub_reg(rd, 31, rm);
}

/* ADD Xd, Xn, #imm12 (unsigned, no shift) */
static void emit_add_imm(int rd, int rn, uint32_t imm12) {
    emit32(0x91000000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* SUB Xd, Xn, #imm12 */
static void emit_sub_imm(int rd, int rn, uint32_t imm12) {
    emit32(0xD1000000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* ADD Xd, Xn, #imm12, LSL #12 */
static void emit_add_imm_shift12(int rd, int rn, uint32_t imm12) {
    emit32(0x91400000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}
/* SUB Xd, Xn, #imm12, LSL #12 */
static void emit_sub_imm_shift12(int rd, int rn, uint32_t imm12) {
    emit32(0xD1400000u | ((imm12 & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rd & 0x1f));
}

/* Emit SP := SP - n (n >= 0, < 16 MB). Uses two 12-bit imm fields. */
static void emit_sub_sp(uint32_t n) {
    uint32_t low  = n & 0xfff;
    uint32_t high = (n >> 12) & 0xfff;
    if (high) emit_sub_imm_shift12(31, 31, high);
    if (low)  emit_sub_imm(31, 31, low);
}
/* Emit SP := SP + n */
static void emit_add_sp(uint32_t n) {
    uint32_t low  = n & 0xfff;
    uint32_t high = (n >> 12) & 0xfff;
    if (high) emit_add_imm_shift12(31, 31, high);
    if (low)  emit_add_imm(31, 31, low);
}

/* CMP Xn, Xm  (SUBS XZR, Xn, Xm) */
static void emit_cmp_reg(int rn, int rm) {
    emit32(0xEB00001Fu | ((rm & 0x1f) << 16) | ((rn & 0x1f) << 5));
}

/* STUR Xt, [Xn, #simm9]  (negative offsets from x29) */
static void emit_stur(int rt, int rn, int simm9) {
    uint32_t imm9 = (uint32_t)(simm9 & 0x1ff);
    emit32(0xF8000000u | (imm9 << 12) | ((rn & 0x1f) << 5) | (rt & 0x1f));
}
/* LDUR Xt, [Xn, #simm9] */
static void emit_ldur(int rt, int rn, int simm9) {
    uint32_t imm9 = (uint32_t)(simm9 & 0x1ff);
    emit32(0xF8400000u | (imm9 << 12) | ((rn & 0x1f) << 5) | (rt & 0x1f));
}
/* LDR Xt, [Xn, #imm12*8]  (unsigned-offset) */
static void emit_ldr_uofs(int rt, int rn, uint32_t byte_ofs) {
    uint32_t e = byte_ofs / 8;
    emit32(0xF9400000u | ((e & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f));
}
/* STR Xt, [Xn, #imm12*8]  (unsigned-offset) */
static void emit_str_uofs(int rt, int rn, uint32_t byte_ofs) {
    uint32_t e = byte_ofs / 8;
    emit32(0xF9000000u | ((e & 0xfff) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f));
}
/* STR Xt, [Xn] */
static void emit_str0(int rt, int rn) { emit_str_uofs(rt, rn, 0); }
/* LDR Xt, [Xn] */
static void emit_ldr0(int rt, int rn) { emit_ldr_uofs(rt, rn, 0); }

/* Prologue STP X29, X30, [SP, #-16]!  = 0xA9BF7BFD */
static void emit_prologue_stp(void) { emit32(0xA9BF7BFDu); }
/* MOV X29, SP  = 0x910003FD */
static void emit_prologue_mov_fp(void) { emit32(0x910003FDu); }
/* LDP X29, X30, [SP], #16  = 0xA8C17BFD */
static void emit_epilogue_ldp(void) { emit32(0xA8C17BFDu); }
/* RET = 0xD65F03C0 */
static void emit_ret(void) { emit32(0xD65F03C0u); }
/* SVC #0x80 = 0xD4001001 */
static void emit_svc_0x80(void) { emit32(0xD4001001u); }

/* Unconditional B imm26 (imm26 is instruction-index offset, signed) */
static void emit_b_imm(int32_t imm26) {
    uint32_t f = (uint32_t)imm26 & 0x03ffffff;
    emit32(0x14000000u | f);
}
/* BL imm26 */
static void emit_bl_imm(int32_t imm26) {
    uint32_t f = (uint32_t)imm26 & 0x03ffffff;
    emit32(0x94000000u | f);
}
/* B.cond imm19 (signed) */
static void emit_bcond_imm(int cond, int32_t imm19) {
    uint32_t f = (uint32_t)imm19 & 0x7ffff;
    emit32(0x54000000u | (f << 5) | (cond & 0xf));
}
/* CBZ Xt, imm19 */
static void emit_cbz_imm(int rt, int32_t imm19) {
    uint32_t f = (uint32_t)imm19 & 0x7ffff;
    emit32(0xB4000000u | (f << 5) | (rt & 0x1f));
}

/* Placeholders we'll patch later. Emit with offset 0, return address to patch. */
static uint32_t emit_b_placeholder(void) { uint32_t here = code_here(); emit_b_imm(0); return here; }
static uint32_t emit_bcond_placeholder(int cond) { uint32_t here = code_here(); emit_bcond_imm(cond, 0); return here; }
static uint32_t emit_cbz_placeholder(int rt) { uint32_t here = code_here(); emit_cbz_imm(rt, 0); return here; }
/* emit_bl_placeholder reserved — unused in v1 */

/* Patch a B at `patch_at` (byte offset in code buffer) to branch to `target`. */
static void patch_b(uint32_t patch_at, uint32_t target) {
    int32_t rel = ((int32_t)target - (int32_t)patch_at) / 4;
    uint32_t inst = 0x14000000u | ((uint32_t)rel & 0x03ffffff);
    g_code[patch_at+0] = inst & 0xff;
    g_code[patch_at+1] = (inst >> 8) & 0xff;
    g_code[patch_at+2] = (inst >> 16) & 0xff;
    g_code[patch_at+3] = (inst >> 24) & 0xff;
}
/* patch_bl reserved — unused in v1 */
static void patch_bcond(uint32_t patch_at, int cond, uint32_t target) {
    int32_t rel = ((int32_t)target - (int32_t)patch_at) / 4;
    uint32_t inst = 0x54000000u | (((uint32_t)rel & 0x7ffff) << 5) | (cond & 0xf);
    g_code[patch_at+0] = inst & 0xff;
    g_code[patch_at+1] = (inst >> 8) & 0xff;
    g_code[patch_at+2] = (inst >> 16) & 0xff;
    g_code[patch_at+3] = (inst >> 24) & 0xff;
}
static void patch_cbz(uint32_t patch_at, int rt, uint32_t target) {
    int32_t rel = ((int32_t)target - (int32_t)patch_at) / 4;
    uint32_t inst = 0xB4000000u | (((uint32_t)rel & 0x7ffff) << 5) | (rt & 0x1f);
    g_code[patch_at+0] = inst & 0xff;
    g_code[patch_at+1] = (inst >> 8) & 0xff;
    g_code[patch_at+2] = (inst >> 16) & 0xff;
    g_code[patch_at+3] = (inst >> 24) & 0xff;
}

/* ============================================================ */
/*  Symbol table                                                 */
/* ============================================================ */
enum { SK_COMP, SK_LOCAL, SK_PARAM };
typedef struct {
    char name[64];
    int kind;
    /* For SK_LOCAL / SK_PARAM: offset from x29 (positive number,
       meaning [x29, #-offset]). */
    int stack_off;
    /* For SK_COMP: code offset of entry (byte offset in g_code). */
    uint32_t code_off;
    /* Number of declared params (for SK_COMP). */
    int nparams;
    /* Scope level: 0 = global (comps), 1+ = inside a comp. */
    int scope;
} Sym;

#define SYM_MAX 1024
static Sym g_sym[SYM_MAX];
static int g_nsym = 0;
static int g_scope = 0;

/* Per-composition frame bookkeeping: next free stack slot. Always multiples of 8. */
static int g_frame_next = 0;
#define FRAME_SIZE 512

static Sym *sym_lookup(const char *s, int len) {
    /* Search in reverse so innermost shadows outermost. */
    for (int i = g_nsym - 1; i >= 0; i--) {
        Sym *sy = &g_sym[i];
        if ((int)strlen(sy->name) == len && memcmp(sy->name, s, len) == 0) return sy;
    }
    return NULL;
}

static Sym *sym_add(const char *s, int len, int kind) {
    if (g_nsym >= SYM_MAX) { fprintf(stderr, "symbol table overflow\n"); exit(1); }
    Sym *sy = &g_sym[g_nsym++];
    if (len >= (int)sizeof(sy->name)) len = (int)sizeof(sy->name) - 1;
    memcpy(sy->name, s, len);
    sy->name[len] = 0;
    sy->kind = kind;
    sy->stack_off = 0;
    sy->code_off = 0;
    sy->nparams = 0;
    sy->scope = g_scope;
    return sy;
}

static int frame_alloc_slot(void) {
    g_frame_next += 8;
    if (g_frame_next > FRAME_SIZE) { fprintf(stderr, "frame overflow (>%d bytes)\n", FRAME_SIZE); exit(1); }
    return g_frame_next;
}

/* ============================================================ */
/*  Expression temp register allocator (bump, over x9..x18)       */
/* ============================================================ */
#define TMP_FIRST 9
#define TMP_LAST  18
static int g_next_tmp = TMP_FIRST;

static int tmp_alloc(void) {
    if (g_next_tmp > TMP_LAST) {
        fprintf(stderr, "expression requires too many temps (>%d)\n", TMP_LAST - TMP_FIRST + 1);
        exit(1);
    }
    return g_next_tmp++;
}
static void tmp_reset(void) { g_next_tmp = TMP_FIRST; }

/* ============================================================ */
/*  Parser                                                       */
/* ============================================================ */
static Token *tok(void) { return &g_tok[g_tk]; }
static void next(void) { if (g_tok[g_tk].type != T_EOF) g_tk++; }
static int at(int type) { return g_tok[g_tk].type == type; }

/* Forward decls */
static int parse_expr(void);
static void parse_statement(int min_indent);
static void parse_body(int min_indent);
static int is_stmt_end(int t);

/* Parse an atom: INT, IDENT (read), ( expr ), -atom, UP-arrow $N */
static int parse_atom(void) {
    Token *t = tok();
    int out = tmp_alloc();
    if (t->type == T_INT) {
        emit_mov_imm64(out, t->ival);
        next();
        return out;
    }
    if (t->type == T_MINUS) {
        next();
        /* recursively parse atom, then neg */
        int inner = parse_atom();
        emit_neg_reg(out, inner);
        return out;
    }
    if (t->type == T_LPAREN) {
        next();
        /* Reuse the sub-expression's register directly to avoid wasting a temp. */
        g_next_tmp--; /* give back 'out' because we want the inner to take over */
        int r = parse_expr();
        if (!at(T_RPAREN)) die("expected ')'", t->line);
        next();
        return r;
    }
    if (t->type == T_REG_READ) {
        next();
        if (at(T_DOLLAR)) {
            next();
            if (!at(T_INT)) die("expected register number after $", t->line);
            int regn = (int)tok()->ival;
            next();
            emit_mov_reg(out, regn);
            return out;
        }
        /* Otherwise: memory load form, read width+addr. Minimal: UP width addr. */
        /* parse_expr twice */
        g_next_tmp--;
        int w = parse_expr(); (void)w;
        int a = parse_expr();
        emit_ldr0(out = tmp_alloc(), a);
        return out;
    }
    if (t->type == T_IDENT) {
        Sym *sy = sym_lookup(t->s, t->slen);
        if (sy && (sy->kind == SK_LOCAL || sy->kind == SK_PARAM)) {
            emit_ldur(out, 29, -sy->stack_off);
            next();
            return out;
        }
        /* Known composition — call it with its declared arg count, then
           produce the result (x0) into `out`. */
        if (sy && sy->kind == SK_COMP) {
            int nparams = sy->nparams;
            uint32_t comp_off = sy->code_off;
            next();
            /* Parse `nparams` arguments. Each arg is a full sub-expression
               so we can't just call parse_atom — but we also can't call
               parse_expr because that would consume later args as operands.
               The convention is: args are atoms (literal/ident/paren).
               For more complex expressions in args, the caller must use
               parentheses. */
            int args[8];
            for (int i = 0; i < nparams && i < 8; i++) {
                if (is_stmt_end(tok()->type)) break;
                /* Give back our 'out' so the arg parse can reuse it? no —
                   we need 'out' to persist. Just allocate args via atoms. */
                args[i] = parse_atom();
            }
            for (int i = 0; i < nparams && i < 8; i++) {
                emit_mov_reg(i, args[i]);
            }
            uint32_t here = code_here();
            int32_t rel = ((int32_t)comp_off - (int32_t)here) / 4;
            emit_bl_imm(rel);
            emit_mov_reg(out, 0);
            return out;
        }
        /* Unknown ident — treat as 0 for now */
        emit_mov_imm64(out, 0);
        next();
        return out;
    }
    /* Fallback */
    emit_mov_imm64(out, 0);
    next();
    return out;
}

/* Multiplicative: atom  (( *|/ ) atom)* */
static int parse_mul(void) {
    int l = parse_atom();
    for (;;) {
        int op = tok()->type;
        if (op != T_STAR && op != T_SLASH) break;
        next();
        int r = parse_atom();
        int out = tmp_alloc();
        if (op == T_STAR) emit_mul_reg(out, l, r);
        else              emit_sdiv_reg(out, l, r);
        l = out;
    }
    return l;
}
/* Additive: mul (( +|- ) mul)* */
static int parse_add(void) {
    int l = parse_mul();
    for (;;) {
        int op = tok()->type;
        if (op != T_PLUS && op != T_MINUS) break;
        next();
        int r = parse_mul();
        int out = tmp_alloc();
        if (op == T_PLUS) emit_add_reg(out, l, r);
        else              emit_sub_reg(out, l, r);
        l = out;
    }
    return l;
}
/* Shift: add (( << | >> ) add)* */
static int parse_shift(void) {
    int l = parse_add();
    for (;;) {
        int op = tok()->type;
        if (op != T_SHL && op != T_SHR) break;
        next();
        int r = parse_add();
        int out = tmp_alloc();
        if (op == T_SHL) emit_lslv(out, l, r);
        else             emit_lsrv(out, l, r);
        l = out;
    }
    return l;
}
/* Bitwise: shift (( & | | | ^ ) shift)* */
static int parse_bitwise(void) {
    int l = parse_shift();
    for (;;) {
        int op = tok()->type;
        if (op != T_AMP && op != T_PIPE && op != T_CARET) break;
        next();
        int r = parse_shift();
        int out = tmp_alloc();
        if (op == T_AMP)  emit_and_reg(out, l, r);
        else if (op == T_PIPE) emit_orr_reg(out, l, r);
        else              emit_eor_reg(out, l, r);
        l = out;
    }
    return l;
}

static int parse_expr(void) { return parse_bitwise(); }

/* Parse a $N register reference (after skipping optional whitespace).
   Returns the register number. */
static int parse_dollar_reg(void) {
    int line = tok()->line;
    if (!at(T_DOLLAR)) die("expected $ before register number", line);
    next();
    if (!at(T_INT)) die("expected register number after $", line);
    int n = (int)tok()->ival;
    next();
    return n;
}

/* Pending Linux syscall number (after `<DOWN> $8 N`). -1 = none. */
static int g_pending_linux_sysno = -1;

/* Statement starters that terminate an expression list on the same line. */
static int is_stmt_end(int t) {
    return t == T_NEWLINE || t == T_EOF || t == T_INDENT;
}

/* Handle `<DOWN> $N val` = MOV XN, val */
static void handle_reg_write(void) {
    int line = tok()->line;
    next();  /* skip arrow */
    if (!at(T_DOLLAR)) die("expected $ after <DOWN>", line);
    int regn = parse_dollar_reg();
    /* Fast path: integer literal */
    if (at(T_INT)) {
        int64_t v = tok()->ival;
        next();
        emit_mov_imm64(regn, v);
        /* Remember pending linux syscall number if writing x8 */
        if (regn == 8) g_pending_linux_sysno = (int)v;
        return;
    }
    if (at(T_MINUS) && g_tok[g_tk+1].type == T_INT) {
        next();
        int64_t v = -tok()->ival;
        next();
        emit_mov_imm64(regn, v);
        if (regn == 8) g_pending_linux_sysno = (int)v;
        return;
    }
    /* Expression path */
    int r = parse_expr();
    emit_mov_reg(regn, r);
    if (regn == 8) g_pending_linux_sysno = -1;  /* dynamic, can't translate */
}

/* Handle `<DOWN>` as a memory store. Form: <DOWN> width addr val
   Only reachable when the token after <DOWN> is not $.*/
static void handle_mem_store(void) {
    next(); /* skip arrow */
    int w = parse_expr(); (void)w;  /* ignored — 64-bit only */
    int a = parse_expr();
    int v = parse_expr();
    emit_str0(v, a);
}

/* Handle `<UP> $N` = MOV Xout, XN (returning out as a value in x0).
   When used as a standalone statement, result goes to x0. */
static void handle_reg_read_stmt(void) {
    next(); /* skip arrow */
    if (at(T_DOLLAR)) {
        int regn = parse_dollar_reg();
        emit_mov_reg(0, regn);
        return;
    }
    /* Memory load: <UP> width addr */
    int w = parse_expr(); (void)w;
    int a = parse_expr();
    int out = tmp_alloc();
    emit_ldr0(out, a);
    emit_mov_reg(0, out);
}

static int tok_cond_to_cc(int tt, int *is_compound) {
    *is_compound = 1;
    switch (tt) {
    case T_LT:   return 0xB;  /* LT */
    case T_GT:   return 0xC;  /* GT */
    case T_LTE:  return 0xD;  /* LE */
    case T_GTE:  return 0xA;  /* GE */
    case T_EQEQ: return 0x0;  /* EQ */
    case T_NEQ:  return 0x1;  /* NE */
    }
    *is_compound = 0;
    return 0;
}
/* Inverted condition (for skip-on-false) */
static int cc_inverted(int cc) {
    switch (cc) {
    case 0x0: return 0x1; /* EQ -> NE */
    case 0x1: return 0x0;
    case 0xA: return 0xB; /* GE -> LT */
    case 0xB: return 0xA;
    case 0xC: return 0xD; /* GT -> LE */
    case 0xD: return 0xC;
    }
    return 0x0;
}

/* Process a pending Linux syscall number for `trap`: translate to Darwin.
   Emits MOV X16, #darwin_sysno. */
static void emit_darwin_syscall_prep(void) {
    if (g_pending_linux_sysno < 0) return;
    int darwin;
    switch (g_pending_linux_sysno) {
    case 93: darwin = 1;   break; /* exit */
    case 64: darwin = 4;   break; /* write */
    case 63: darwin = 3;   break; /* read */
    case 57: darwin = 6;   break; /* close */
    case 56: darwin = 5;   break; /* openat ~ open (not exact, but keeps x16 sane) */
    case 214: darwin = 45; break; /* brk ~ sbrk */
    default: darwin = g_pending_linux_sysno; break;
    }
    emit_mov_imm64(16, darwin);
    g_pending_linux_sysno = -1;
}

/* Handle `trap` */
static void handle_trap(void) {
    next(); /* skip 'trap' */
    emit_darwin_syscall_prep();
    emit_svc_0x80();
}

/* Forward declarations for control flow */
static void handle_if(int min_indent);
static void handle_for(int min_indent);
static void handle_while(int min_indent);
static void handle_each(int min_indent);

/* ============================================================ */
/*  Statements                                                   */
/* ============================================================ */

/* Parse a body: all statements whose leading indent >= min_indent.
   Must be called positioned at the INDENT token before the first stmt. */
static void parse_body(int min_indent) {
    while (!at(T_EOF)) {
        /* Blank line: skip */
        if (at(T_NEWLINE)) { next(); continue; }
        if (at(T_INDENT)) {
            int cur = (int)tok()->ival;
            /* Blank line (just INDENT then NEWLINE)? */
            if (g_tok[g_tk+1].type == T_NEWLINE) { next(); next(); continue; }
            if (cur < min_indent) return;   /* dedent — end of body */
            next();  /* consume indent */
            parse_statement(min_indent);
            continue;
        }
        /* Statement without leading INDENT? shouldn't normally happen */
        parse_statement(min_indent);
    }
}

/* Parse one statement. Assumes we are positioned AT the first token of the
   statement (INDENT already consumed by caller).  */
static void parse_statement(int min_indent) {
    tmp_reset();
    Token *t = tok();

    if (t->type == T_KW_TRAP) { handle_trap(); return; }
    if (t->type == T_KW_IF)   { handle_if(min_indent); return; }
    if (t->type == T_KW_FOR)  { handle_for(min_indent); return; }
    if (t->type == T_KW_WHILE){ handle_while(min_indent); return; }
    if (t->type == T_KW_EACH) { handle_each(min_indent); return; }

    if (t->type == T_REG_WRITE) {
        /* Two forms: <DOWN> $N val (reg write) or <DOWN> width addr val (store) */
        if (g_tok[g_tk+1].type == T_DOLLAR) handle_reg_write();
        else handle_mem_store();
        return;
    }
    if (t->type == T_REG_READ) {
        handle_reg_read_stmt();
        return;
    }

    if (t->type == T_KW_VAR) {
        next();
        if (!at(T_IDENT)) die("expected name after 'var'", t->line);
        Token name = *tok();
        next();
        /* Parse value expression */
        int r = parse_expr();
        Sym *sy = sym_add(name.s, name.slen, SK_LOCAL);
        sy->stack_off = frame_alloc_slot();
        emit_stur(r, 29, -sy->stack_off);
        return;
    }

    if (t->type == T_KW_RETURN) {
        next();
        if (!is_stmt_end(tok()->type)) {
            int r = parse_expr();
            emit_mov_reg(0, r);
        }
        /* Emit epilogue */
        emit_add_sp(FRAME_SIZE);
        emit_epilogue_ldp();
        emit_ret();
        return;
    }

    if (t->type == T_IDENT) {
        /* Candidates:
             1. known comp -> call (possibly with args)
             2. known local/param followed by end-of-stmt -> bare read, MOV X0
             3. known local/param followed by operator -> reassignment via expr
             4. known local/param followed by value token -> reassign (skip name)
             5. unknown name followed by value -> binding (introduce local)
             6. unknown name alone -> no-op
        */
        Sym *sy = sym_lookup(t->s, t->slen);
        Token next_tk = g_tok[g_tk+1];

        if (sy && sy->kind == SK_COMP) {
            /* Call */
            Sym *comp = sy;
            next();  /* skip name */
            int argi = 0;
            while (!is_stmt_end(tok()->type)) {
                int r = parse_expr();
                emit_mov_reg(argi, r);
                argi++;
                if (argi >= 8) break;
            }
            uint32_t here = code_here();
            int32_t rel = ((int32_t)comp->code_off - (int32_t)here) / 4;
            emit_bl_imm(rel);
            return;
        }

        if (sy && (sy->kind == SK_LOCAL || sy->kind == SK_PARAM)) {
            /* Reassignment vs bare read */
            if (is_stmt_end(next_tk.type)) {
                /* bare read — emit ldur into x0 */
                emit_ldur(0, 29, -sy->stack_off);
                next();
                return;
            }
            /* If the token right after the name is an arithmetic/bitwise
               operator, we parse the full expression STARTING at the name
               so it's a proper r-value use. Otherwise the name is the lhs
               and the rhs begins AFTER the name. */
            int t1 = next_tk.type;
            int full_rhs = (t1 == T_PLUS || t1 == T_MINUS || t1 == T_STAR || t1 == T_SLASH ||
                            t1 == T_AMP  || t1 == T_PIPE  || t1 == T_CARET || t1 == T_SHL ||
                            t1 == T_SHR);
            int stack_off = sy->stack_off;
            if (!full_rhs) next();  /* skip name */
            if (t1 == T_EQ) next();  /* skip = */
            int r = parse_expr();
            emit_stur(r, 29, -stack_off);
            return;
        }

        /* Unknown name */
        if (is_stmt_end(next_tk.type)) {
            /* No-op */
            next();
            return;
        }
        /* Treat as binding: introduce a local */
        Token name = *tok();
        next();
        /* If the next token is '=', this is `name = expr` form */
        if (at(T_EQ)) next();
        int r = parse_expr();
        Sym *nsy = sym_add(name.s, name.slen, SK_LOCAL);
        nsy->stack_off = frame_alloc_slot();
        emit_stur(r, 29, -nsy->stack_off);
        return;
    }

    /* Bare expression (INT, (...)) — evaluate, result in x0 */
    if (t->type == T_INT || t->type == T_LPAREN || t->type == T_MINUS) {
        int r = parse_expr();
        emit_mov_reg(0, r);
        return;
    }

    /* Unknown token — skip to newline */
    while (!is_stmt_end(tok()->type)) next();
}

/* ============================================================ */
/*  Control flow                                                 */
/* ============================================================ */

static void handle_if(int min_indent) {
    int line = tok()->line;
    next(); /* skip 'if' */
    int is_compound;
    int cc = tok_cond_to_cc(tok()->type, &is_compound);

    uint32_t patch_skip = 0;
    int patch_kind = 0;  /* 0 = CBZ, 1 = B.cond */
    int patch_rt = 0;
    int patch_cond = 0;

    if (is_compound) {
        next();  /* consume compare token */
        int l = parse_expr();
        int r = parse_expr();
        emit_cmp_reg(l, r);
        int inv = cc_inverted(cc);
        patch_skip = emit_bcond_placeholder(inv);
        patch_kind = 1;
        patch_cond = inv;
    } else {
        int r = parse_expr();
        patch_skip = emit_cbz_placeholder(r);
        patch_kind = 0;
        patch_rt = r;
    }

    /* Expect colon optionally */
    if (at(T_COLON)) next();
    if (at(T_NEWLINE)) next();

    /* Parse body — find the indent of the first body line */
    int body_indent = min_indent + 2;
    if (at(T_INDENT)) body_indent = (int)tok()->ival;
    (void)line;
    parse_body(body_indent);

    /* After body, optional else */
    /* Skip blank lines to find the else */
    /* Save position */
    int save = g_tk;
    while (at(T_INDENT) || at(T_NEWLINE)) {
        if (at(T_INDENT) && g_tok[g_tk+1].type == T_NEWLINE) { next(); next(); continue; }
        if (at(T_NEWLINE)) { next(); continue; }
        break;
    }
    /* If next statement is 'else', require it to be at same indent as 'if'. */
    if (at(T_KW_ELSE)) {
        /* emit skip-over-else B */
        uint32_t patch_over = emit_b_placeholder();
        /* Patch the skip to HERE */
        uint32_t here = code_here();
        if (patch_kind == 0) patch_cbz(patch_skip, patch_rt, here);
        else                 patch_bcond(patch_skip, patch_cond, here);
        next(); /* eat 'else' */
        if (at(T_COLON)) next();
        if (at(T_NEWLINE)) next();
        int else_indent = min_indent + 2;
        if (at(T_INDENT)) else_indent = (int)tok()->ival;
        parse_body(else_indent);
        uint32_t end = code_here();
        patch_b(patch_over, end);
    } else {
        g_tk = save;
        uint32_t here = code_here();
        if (patch_kind == 0) patch_cbz(patch_skip, patch_rt, here);
        else                 patch_bcond(patch_skip, patch_cond, here);
    }
}

static void handle_while(int min_indent) {
    next(); /* skip while */
    uint32_t top = code_here();
    int r = parse_expr();
    uint32_t exit_patch = emit_cbz_placeholder(r);
    if (at(T_COLON)) next();
    if (at(T_NEWLINE)) next();
    int body_indent = min_indent + 2;
    if (at(T_INDENT)) body_indent = (int)tok()->ival;
    parse_body(body_indent);
    /* B top */
    uint32_t here = code_here();
    int32_t rel = ((int32_t)top - (int32_t)here) / 4;
    emit_b_imm(rel);
    /* Patch exit */
    uint32_t after = code_here();
    patch_cbz(exit_patch, r, after);
}

/* for i start end [step] : body */
static void handle_for(int min_indent) {
    int line = tok()->line;
    next(); /* skip for */
    if (!at(T_IDENT)) die("expected loop var after 'for'", line);
    Token name = *tok();
    next();
    /* start */
    int rstart = parse_expr();
    /* Introduce loop var local */
    Sym *sy = sym_add(name.s, name.slen, SK_LOCAL);
    sy->stack_off = frame_alloc_slot();
    emit_stur(rstart, 29, -sy->stack_off);

    /* Allocate end and step locals */
    int end_off = frame_alloc_slot();
    int rend = parse_expr();
    emit_stur(rend, 29, -end_off);

    int step_off = frame_alloc_slot();
    int has_step = !is_stmt_end(tok()->type) && !at(T_COLON);
    int rstep;
    if (has_step) {
        rstep = parse_expr();
    } else {
        rstep = tmp_alloc();
        emit_mov_imm64(rstep, 1);
    }
    emit_stur(rstep, 29, -step_off);

    if (at(T_COLON)) next();
    if (at(T_NEWLINE)) next();

    /* Loop top: reload i, cmp with end, branch if >= */
    uint32_t top = code_here();
    tmp_reset();
    int i_reg = tmp_alloc();
    int e_reg = tmp_alloc();
    emit_ldur(i_reg, 29, -sy->stack_off);
    emit_ldur(e_reg, 29, -end_off);
    emit_cmp_reg(i_reg, e_reg);
    /* B.GE exit (imm19 patched later). GE = 0xA */
    uint32_t exit_patch = emit_bcond_placeholder(0xA);

    int body_indent = min_indent + 2;
    if (at(T_INDENT)) body_indent = (int)tok()->ival;
    parse_body(body_indent);

    /* Increment: i = i + step */
    tmp_reset();
    int i2 = tmp_alloc();
    int s2 = tmp_alloc();
    emit_ldur(i2, 29, -sy->stack_off);
    emit_ldur(s2, 29, -step_off);
    emit_add_reg(i2, i2, s2);
    emit_stur(i2, 29, -sy->stack_off);

    /* Branch to top */
    uint32_t here = code_here();
    int32_t rel = ((int32_t)top - (int32_t)here) / 4;
    emit_b_imm(rel);

    uint32_t after = code_here();
    patch_bcond(exit_patch, 0xA, after);
}

/* each i : body  — on host, just set Xi = 0 and run body once. */
static void handle_each(int min_indent) {
    int line = tok()->line;
    next(); /* skip 'each' */
    if (!at(T_IDENT)) die("expected name after 'each'", line);
    Token name = *tok();
    next();
    Sym *sy = sym_add(name.s, name.slen, SK_LOCAL);
    sy->stack_off = frame_alloc_slot();
    /* zero into slot */
    int r = tmp_alloc();
    emit_mov_imm64(r, 0);
    emit_stur(r, 29, -sy->stack_off);

    if (at(T_COLON)) next();
    if (at(T_NEWLINE)) next();
    int body_indent = min_indent + 2;
    if (at(T_INDENT)) body_indent = (int)tok()->ival;
    parse_body(body_indent);
}

/* ============================================================ */
/*  Composition parsing                                          */
/* ============================================================ */

static void parse_composition(void) {
    /* We're at a name token at column 0 — no leading INDENT. */
    Token *nt = tok();
    if (nt->type != T_IDENT) { next(); return; }
    next();

    /* Collect param names until colon */
    Token params[8];
    int nparams = 0;
    while (!at(T_COLON) && !at(T_EOF) && !at(T_NEWLINE)) {
        if (at(T_IDENT)) {
            if (nparams < 8) params[nparams++] = *tok();
            next();
        } else {
            /* skip stray token */
            next();
        }
    }
    if (at(T_COLON)) next();
    if (at(T_NEWLINE)) next();

    /* Register the composition symbol BEFORE emitting body so it's callable
       from within itself and from subsequent comps. */
    Sym *comp = sym_add(nt->s, nt->slen, SK_COMP);
    comp->code_off = code_here();
    comp->nparams = nparams;

    /* New scope */
    g_scope++;
    g_frame_next = 0;
    int saved_nsym = g_nsym;

    /* Prologue */
    emit_prologue_stp();
    emit_prologue_mov_fp();
    emit_sub_sp(FRAME_SIZE);

    /* Spill params to stack slots */
    for (int i = 0; i < nparams; i++) {
        Sym *ps = sym_add(params[i].s, params[i].slen, SK_PARAM);
        ps->stack_off = frame_alloc_slot();
        emit_stur(i, 29, -ps->stack_off);
    }

    /* Parse body */
    int body_indent = 1;  /* any indent > 0 */
    if (at(T_INDENT)) {
        body_indent = (int)tok()->ival;
        if (body_indent == 0) body_indent = 1;  /* nothing indented */
    }
    parse_body(body_indent);

    /* Epilogue: restore SP, LDP fp/lr, RET */
    emit_add_sp(FRAME_SIZE);
    emit_epilogue_ldp();
    emit_ret();

    /* Pop scope: drop all locals introduced inside */
    g_scope--;
    g_nsym = saved_nsym;
    g_frame_next = 0;
}

static void parse_file(void) {
    g_tk = 0;
    /* Accept leading blank lines and indent-zero markers */
    while (!at(T_EOF)) {
        if (at(T_NEWLINE)) { next(); continue; }
        if (at(T_INDENT)) {
            /* Top-level statements must be at indent 0 */
            if (tok()->ival == 0) { next(); continue; }
            /* Anything else is stray */
            next(); continue;
        }
        if (at(T_IDENT)) {
            /* Top-level composition */
            parse_composition();
            continue;
        }
        if (at(T_KW_TRAP)) {
            /* Bare top-level trap: emit as its own tiny function */
            /* We don't wrap it; just pretend it's a composition named __top */
            parse_composition();
            continue;
        }
        next();
    }
}

/* ============================================================ */
/*  Mach-O writer                                                */
/* ============================================================ */

/* Constants (avoid pulling in <mach-o/loader.h> to stay truly minimal) */
#define MH_MAGIC_64     0xFEEDFACFu
#define CPU_TYPE_ARM64  0x0100000Cu
#define MH_EXECUTE      0x2
#define MH_NOUNDEFS     0x1
#define LC_SEGMENT_64   0x19
#define LC_UNIXTHREAD   0x5
#define ARM_THREAD_STATE64_FLAVOR 6
#define VM_PROT_READ    1
#define VM_PROT_WRITE   2
#define VM_PROT_EXECUTE 4

/*
 * Mach-O emission strategy:
 *
 * On Apple Silicon macOS (11+), the AMFI (Apple Mobile File Integrity)
 * daemon refuses to execute static Mach-O binaries that don't use dyld —
 * even if they are ad-hoc signed. A minimal working binary must:
 *
 *   - Link against /usr/lib/dyld (LC_LOAD_DYLINKER)
 *   - Link against /usr/lib/libSystem.B.dylib (LC_LOAD_DYLIB)
 *   - Supply a chained-fixups blob in __LINKEDIT (LC_DYLD_CHAINED_FIXUPS)
 *   - Supply a symbol table, LC_BUILD_VERSION, LC_MAIN, LC_CODE_SIGNATURE
 *   - Carry a valid ad-hoc code-signature (CodeDirectory blob)
 *
 * Writing all of that from scratch is hundreds of lines of fiddly bytes
 * and an SHA-256 implementation. Since the reference bootstrap already
 * shells out to `as` + `ld` (see compile-darwin.sh), this compiler takes
 * the same pragmatic approach: we emit our ARM64 machine code bytes into
 * a temporary assembly file as `.byte` directives wrapped by a `_main:`
 * label, then invoke `as` and `ld` (system tools already present on any
 * macOS dev box) to produce the final Mach-O. The system linker handles
 * dyld-chains, BUILD_VERSION, CodeDirectory, and ad-hoc signing, all
 * correctly.
 *
 * This keeps the C source small, self-contained, and robust across
 * macOS versions. The binary is still a regular Mach-O, just wrapped
 * for us by `ld`.
 */
#define PAGE_SIZE 16384

/* Write our ARM64 code bytes as a `.byte` directive assembly file.
   We place a single global `_main` label at the start of the code. Each
   composition is emitted consecutively, so `_main` maps to the byte 0
   of the code buffer if the first composition in the source is main;
   otherwise we just emit a leading `B <main_offset>` fixup... actually,
   we use the known offset of the main composition and branch to it. */
static void write_macho(const char *out_path) {
    uint32_t code_size = g_code_pos;

    /* Find `main` composition; any other composition is emitted too and
       `main` is the entry. If main is not at offset 0, we prepend a tiny
       trampoline that branches to it. */
    uint32_t main_off = 0;
    int found_main = 0;
    for (int i = 0; i < g_nsym; i++) {
        if (g_sym[i].kind == SK_COMP && strcmp(g_sym[i].name, "main") == 0) {
            main_off = g_sym[i].code_off;
            found_main = 1;
            break;
        }
    }
    if (!found_main) {
        for (int i = g_nsym - 1; i >= 0; i--) {
            if (g_sym[i].kind == SK_COMP) { main_off = g_sym[i].code_off; found_main = 1; break; }
        }
    }
    if (!found_main) {
        fprintf(stderr, "lithos-boot-ken: no composition found\n");
        exit(1);
    }

    /* Create tmpdir */
    char tmpdir[] = "/tmp/lithos-ken-XXXXXX";
    if (!mkdtemp(tmpdir)) { perror("mkdtemp"); exit(1); }
    char asm_path[1024], obj_path[1024];
    snprintf(asm_path, sizeof(asm_path), "%s/wrap.s", tmpdir);
    snprintf(obj_path, sizeof(obj_path), "%s/wrap.o", tmpdir);

    FILE *f = fopen(asm_path, "w");
    if (!f) { perror("fopen asm"); exit(1); }
    fprintf(f, ".text\n");
    fprintf(f, ".global _main\n");
    fprintf(f, ".align 4\n");
    /* Emit the blob as a stream of .byte directives, with an internal
       _blob label at the very start. We then declare _main as an alias
       for (_blob + main_off) via .set so ld points the entry at the
       correct composition without needing label duplication. */
    fprintf(f, "__ken_blob:\n");
    for (uint32_t i = 0; i < code_size; i++) {
        fprintf(f, "    .byte 0x%02x\n", g_code[i]);
    }
    fprintf(f, ".set _main, __ken_blob + %u\n", main_off);
    fclose(f);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "as -arch arm64 -o %s %s", obj_path, asm_path);
    if (system(cmd) != 0) {
        fprintf(stderr, "lithos-boot-ken: 'as' failed\n");
        exit(1);
    }

    const char *sdk_env = getenv("LITHOS_SDK");
    char sdk_buf[1024] = {0};
    if (!sdk_env) {
        FILE *pf = popen("xcrun --sdk macosx --show-sdk-path 2>/dev/null", "r");
        if (pf) {
            if (fgets(sdk_buf, sizeof(sdk_buf) - 1, pf)) {
                size_t n = strlen(sdk_buf);
                while (n > 0 && (sdk_buf[n-1] == '\n' || sdk_buf[n-1] == '\r')) sdk_buf[--n] = 0;
                sdk_env = sdk_buf;
            }
            pclose(pf);
        }
    }

    if (sdk_env && sdk_env[0]) {
        snprintf(cmd, sizeof(cmd),
                 "ld -arch arm64 -platform_version macos 11.0.0 15.0.0 "
                 "-syslibroot %s -lSystem -e _main -o %s %s",
                 sdk_env, out_path, obj_path);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "ld -arch arm64 -platform_version macos 11.0.0 15.0.0 "
                 "-lSystem -e _main -o %s %s",
                 out_path, obj_path);
    }
    if (system(cmd) != 0) {
        fprintf(stderr, "lithos-boot-ken: 'ld' failed\n");
        exit(1);
    }

    /* Cleanup temp files */
    unlink(asm_path);
    unlink(obj_path);
    rmdir(tmpdir);

    chmod(out_path, 0755);
    DBG("wrote Mach-O executable %s", out_path);
}

/* ============================================================ */
/*  main                                                         */
/* ============================================================ */

static void load_source(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); exit(1); }
    g_src_len = (size_t)st.st_size;
    g_src = (char *)malloc(g_src_len + 1);
    if (!g_src) { fprintf(stderr, "oom\n"); exit(1); }
    ssize_t n = read(fd, g_src, g_src_len);
    if (n < 0 || (size_t)n != g_src_len) { perror("read"); exit(1); }
    g_src[g_src_len] = 0;
    close(fd);
}

int main(int argc, char **argv) {
    const char *in = NULL;
    const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) g_debug = 1;
        else if (!in) in = argv[i];
        else if (!out) out = argv[i];
    }
    if (!in || !out) {
        fprintf(stderr, "usage: %s [-v] input.ls output\n", argv[0]);
        return 1;
    }

    load_source(in);
    lex();

    g_code = (uint8_t *)calloc(1, CODE_MAX);
    if (!g_code) { fprintf(stderr, "oom\n"); return 1; }

    parse_file();

    if (g_debug) fprintf(stderr, "[ken] emitted %u bytes of code, %d tokens, %d symbols\n",
                         g_code_pos, g_ntok, g_nsym);

    write_macho(out);
    return 0;
}
