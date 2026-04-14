/* ============================================================================
 * lithos-boot-dennis.c — Lithos bootstrap compiler for macOS ARM64 (Mach-O)
 *
 * Author:  "dennis"  (one of four parallel bootstrap implementations)
 *
 * Architecture
 * ------------
 *   source file
 *      |
 *      |  lex()              :: char* src -> Tok[]
 *      v
 *   token array  (flat, no AST)
 *      |
 *      |  parse_file()       :: Tok[] -> ARM64 code buffer
 *      v
 *   code buffer (uint32_t* of raw ARM64 instructions)
 *      |
 *      |  write_macho()      :: code -> Mach-O file on disk
 *      v
 *   executable
 *
 * Each Lithos composition becomes a function with a real stack frame:
 *
 *     STP X29, X30, [SP, #-16]!     // save caller's FP, LR
 *     MOV X29, SP                   // establish frame
 *     SUB SP, SP, #FRAME_SIZE       // reserve locals (fixed 512B)
 *     STR X0, [X29, #-off_param0]   // spill params into locals
 *     ...
 *     <body>
 *     MOV SP, X29
 *     LDP X29, X30, [SP], #16
 *     RET
 *
 * Every local (parameter, binding, var) lives at a negative offset from X29.
 * Reading a local loads into a fresh expression temp; writing a local stores
 * a result back.  Expression temps are a bump allocator over X9..X18 that
 * resets at every statement boundary.  Because locals never live in
 * caller-saved registers, the allocator never needs to spill — if a single
 * expression demands >10 temps we just error out.  This is the fix for the
 * old assembly bootstrap's fraudulent spill.
 *
 * Only the minimal subset of ARM64 / Mach-O that the test programs need is
 * implemented.  No dynamic linker, no __DATA segment, no symbol table, no
 * LC_MAIN — just __PAGEZERO + __TEXT (containing the header, load commands,
 * and code) + LC_UNIXTHREAD pointing at the code.
 *
 * Syscall ABI: Lithos source uses Linux numbers in X8 (e.g. 93 = exit).
 * On Darwin the number lives in X16 and SVC takes #0x80.  `handle_reg_write`
 * transparently rewrites `↓ $8 <linux_num>` into `MOV X16, #<darwin_num>`
 * for the handful of syscalls we know about (only `exit` really matters for
 * v1).  `trap` emits `SVC #0x80`.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* ============================================================================
 * Tokens
 * ============================================================================ */

enum {
    TK_EOF = 0,
    TK_NEWLINE,
    TK_INDENT,       /* col = leading space count */
    TK_INT,          /* decimal or hex literal */
    TK_IDENT,        /* identifier (may begin with '$') */
    /* keywords */
    TK_IF, TK_ELSE, TK_ELIF,
    TK_FOR, TK_WHILE, TK_EACH,
    TK_VAR, TK_RETURN, TK_TRAP,
    /* operators / punctuation */
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH,
    TK_AMP, TK_PIPE, TK_CARET, TK_SHL, TK_SHR,
    TK_EQ, TK_EQEQ, TK_NEQ, TK_LT, TK_GT, TK_LTE, TK_GTE,
    TK_LPAREN, TK_RPAREN, TK_COLON, TK_COMMA,
    /* compound-if forms: `if<`, `if>`, etc. are folded into TK_IF + the
     * trailing operator token during the lexer's second look. */
    /* memory / register arrows */
    TK_LOAD,         /*  ↑  U+2191 read/load    */
    TK_STORE,        /*  ↓  U+2193 write/store  */
    /* Bonus arrows (for memory ops in the full grammar). */
    TK_ARROW_R,      /*  →  U+2192 load         */
    TK_ARROW_L,      /*  ←  U+2190 store        */
};

typedef struct {
    int type;
    int line;
    int col;           /* for INDENT, this is the indent count */
    int ival;          /* for INT */
    const char* text;  /* for IDENT: pointer into source buffer */
    int len;           /* for IDENT: length in bytes */
} Tok;

static Tok*  g_tokens = NULL;
static int   g_ntok   = 0;
static int   g_tp     = 0;     /* parser index */
static const char* g_src = NULL;

/* ============================================================================
 * Lexer
 * ============================================================================ */

static void die(const char* msg) {
    fprintf(stderr, "lithos-boot-dennis: %s\n", msg);
    exit(1);
}

static void dief(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "lithos-boot-dennis: "); vfprintf(stderr, fmt, ap);
    fputc('\n', stderr); va_end(ap); exit(1);
}

/* Keyword table — order matters only for the length checks in match_kw. */
static int match_kw(const char* s, int n) {
    if (n == 2) {
        if (!memcmp(s, "if", 2)) return TK_IF;
    } else if (n == 3) {
        if (!memcmp(s, "for", 3)) return TK_FOR;
        if (!memcmp(s, "var", 3)) return TK_VAR;
    } else if (n == 4) {
        if (!memcmp(s, "else", 4)) return TK_ELSE;
        if (!memcmp(s, "elif", 4)) return TK_ELIF;
        if (!memcmp(s, "each", 4)) return TK_EACH;
        if (!memcmp(s, "trap", 4)) return TK_TRAP;
    } else if (n == 5) {
        if (!memcmp(s, "while", 5)) return TK_WHILE;
    } else if (n == 6) {
        if (!memcmp(s, "return", 6)) return TK_RETURN;
    }
    return TK_IDENT;
}

static int is_ident_start(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$';
}
static int is_ident_cont(int c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

/* Append a token to the growing array. */
static void push_tok(Tok t) {
    static int cap = 0;
    if (g_ntok >= cap) {
        cap = cap ? cap * 2 : 1024;
        g_tokens = (Tok*)realloc(g_tokens, cap * sizeof(Tok));
        if (!g_tokens) die("out of memory (tokens)");
    }
    g_tokens[g_ntok++] = t;
}

/* Tokenize the entire source.  Indentation-sensitive: emits TK_INDENT at the
 * start of every non-blank line with its column count, and TK_NEWLINE at end.
 * Blank lines and comment-only lines are dropped. */
static void lex(const char* src, int len) {
    int i = 0, line = 1;
    int at_line_start = 1;

    while (i < len) {
        /* Indentation (only at the start of a line). */
        if (at_line_start) {
            int col = 0;
            while (i < len && src[i] == ' ') { col++; i++; }
            /* Skip blank / comment-only / tab-funky lines. */
            if (i >= len) break;
            if (src[i] == '\n') { i++; line++; continue; }
            if (src[i] == '\\' && i+1 < len && src[i+1] == '\\') {
                while (i < len && src[i] != '\n') i++;
                continue;
            }
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_INDENT; t.line = line; t.col = col;
            push_tok(t);
            at_line_start = 0;
        }

        int c = (unsigned char)src[i];

        /* Newline */
        if (c == '\n') {
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_NEWLINE; t.line = line;
            push_tok(t);
            i++; line++; at_line_start = 1;
            continue;
        }

        /* Whitespace (not newline) */
        if (c == ' ' || c == '\t' || c == '\r') { i++; continue; }

        /* Comment: `\\` to end of line */
        if (c == '\\' && i+1 < len && src[i+1] == '\\') {
            while (i < len && src[i] != '\n') i++;
            continue;
        }

        /* UTF-8 arrows. */
        if (c == 0xE2 && i+2 < len) {
            unsigned char b1 = (unsigned char)src[i+1];
            unsigned char b2 = (unsigned char)src[i+2];
            int arrow = -1;
            if (b1 == 0x86) {
                if (b2 == 0x91) arrow = TK_LOAD;      /* ↑ U+2191 */
                else if (b2 == 0x93) arrow = TK_STORE; /* ↓ U+2193 */
                else if (b2 == 0x92) arrow = TK_ARROW_R;/* → U+2192 */
                else if (b2 == 0x90) arrow = TK_ARROW_L;/* ← U+2190 */
            }
            if (arrow >= 0) {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = arrow; t.line = line;
                push_tok(t);
                i += 3;
                continue;
            }
            /* Unknown UTF-8 sequence — skip one byte and let the error
             * surface naturally. */
            i++;
            continue;
        }

        /* Integer literal (decimal or hex) */
        if (c >= '0' && c <= '9') {
            int v = 0;
            if (c == '0' && i+1 < len && (src[i+1] == 'x' || src[i+1] == 'X')) {
                i += 2;
                while (i < len) {
                    int ch = (unsigned char)src[i];
                    int digit;
                    if (ch >= '0' && ch <= '9') digit = ch - '0';
                    else if (ch >= 'a' && ch <= 'f') digit = ch - 'a' + 10;
                    else if (ch >= 'A' && ch <= 'F') digit = ch - 'A' + 10;
                    else break;
                    v = (v << 4) | digit;
                    i++;
                }
            } else {
                while (i < len && src[i] >= '0' && src[i] <= '9') {
                    v = v * 10 + (src[i] - '0');
                    i++;
                }
            }
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_INT; t.ival = v; t.line = line;
            push_tok(t);
            continue;
        }

        /* Identifier (or $-prefixed register name) */
        if (is_ident_start(c)) {
            int start = i;
            i++;
            while (i < len && is_ident_cont((unsigned char)src[i])) i++;
            int n = i - start;
            int kw = (c == '$') ? TK_IDENT : match_kw(src + start, n);
            Tok t; memset(&t, 0, sizeof(t));
            t.type = kw; t.line = line;
            t.text = src + start; t.len = n;
            push_tok(t);
            continue;
        }

        /* Operators / punctuation */
        if (c == '=') {
            if (i+1 < len && src[i+1] == '=') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_EQEQ; t.line = line;
                push_tok(t); i += 2; continue;
            }
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_EQ; t.line = line;
            push_tok(t); i++; continue;
        }
        if (c == '!') {
            if (i+1 < len && src[i+1] == '=') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_NEQ; t.line = line;
                push_tok(t); i += 2; continue;
            }
            i++; continue;
        }
        if (c == '<') {
            if (i+1 < len && src[i+1] == '=') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_LTE; t.line = line;
                push_tok(t); i += 2; continue;
            }
            if (i+1 < len && src[i+1] == '<') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_SHL; t.line = line;
                push_tok(t); i += 2; continue;
            }
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_LT; t.line = line;
            push_tok(t); i++; continue;
        }
        if (c == '>') {
            if (i+1 < len && src[i+1] == '=') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_GTE; t.line = line;
                push_tok(t); i += 2; continue;
            }
            if (i+1 < len && src[i+1] == '>') {
                Tok t; memset(&t, 0, sizeof(t));
                t.type = TK_SHR; t.line = line;
                push_tok(t); i += 2; continue;
            }
            Tok t; memset(&t, 0, sizeof(t));
            t.type = TK_GT; t.line = line;
            push_tok(t); i++; continue;
        }

        int single = -1;
        switch (c) {
            case '+': single = TK_PLUS;  break;
            case '-': single = TK_MINUS; break;
            case '*': single = TK_STAR;  break;
            case '/': single = TK_SLASH; break;
            case '&': single = TK_AMP;   break;
            case '|': single = TK_PIPE;  break;
            case '^': single = TK_CARET; break;
            case '(': single = TK_LPAREN; break;
            case ')': single = TK_RPAREN; break;
            case ':': single = TK_COLON; break;
            case ',': single = TK_COMMA; break;
        }
        if (single >= 0) {
            Tok t; memset(&t, 0, sizeof(t));
            t.type = single; t.line = line;
            push_tok(t); i++; continue;
        }

        /* Skip anything we don't recognize. */
        i++;
    }

    /* Final EOF */
    Tok t; memset(&t, 0, sizeof(t));
    t.type = TK_EOF; t.line = line;
    push_tok(t);
}

/* ============================================================================
 * Symbol table (locals and compositions)
 *
 * Locals live on the stack at a negative offset from X29.  Compositions live
 * at a known code-buffer address (recorded when their prologue is emitted).
 * ============================================================================ */

enum { SYM_LOCAL = 1, SYM_COMP = 2 };

typedef struct {
    char name[64];
    int  kind;         /* SYM_LOCAL | SYM_COMP */
    int  offset;       /* LOCAL: positive byte offset from X29
                         * (actual addr = X29 - offset) */
    int  code_addr;    /* COMP: byte offset into code buffer */
    int  nparams;      /* COMP: number of params */
    int  scope;        /* scope depth at which it was introduced */
} Sym;

#define MAX_SYMS 512
static Sym g_syms[MAX_SYMS];
static int g_nsym = 0;
static int g_scope = 0;

static Sym* sym_lookup(const char* name, int len) {
    for (int i = g_nsym - 1; i >= 0; i--) {
        if ((int)strlen(g_syms[i].name) == len &&
            !memcmp(g_syms[i].name, name, len)) {
            return &g_syms[i];
        }
    }
    return NULL;
}

static Sym* sym_add(const char* name, int len, int kind) {
    if (g_nsym >= MAX_SYMS) die("symbol table overflow");
    if (len >= 64) len = 63;
    Sym* s = &g_syms[g_nsym++];
    memset(s, 0, sizeof(*s));
    memcpy(s->name, name, len);
    s->name[len] = 0;
    s->kind = kind;
    s->scope = g_scope;
    return s;
}

static void sym_pop_scope(int old_nsym) {
    g_nsym = old_nsym;
}

/* ============================================================================
 * ARM64 encoder
 * ============================================================================ */

#define CODE_MAX (1 << 20)
static uint32_t g_code[CODE_MAX / 4];
static int g_code_pos = 0;  /* number of uint32_t written */

static int code_here(void) { return g_code_pos * 4; }

static void emit32(uint32_t w) {
    if (g_code_pos >= CODE_MAX / 4) die("code buffer overflow");
    g_code[g_code_pos++] = w;
}

/* MOVZ Xd, #imm16, LSL #shift */
static void emit_movz(int d, uint32_t imm16, int shift) {
    /* shift: 0/16/32/48 -> hw field 00/01/10/11 */
    uint32_t hw = (shift / 16) & 3;
    emit32(0xD2800000u | (hw << 21) | ((imm16 & 0xFFFF) << 5) | (d & 31));
}

/* MOVK Xd, #imm16, LSL #shift */
static void emit_movk(int d, uint32_t imm16, int shift) {
    uint32_t hw = (shift / 16) & 3;
    emit32(0xF2800000u | (hw << 21) | ((imm16 & 0xFFFF) << 5) | (d & 31));
}

/* Load a 64-bit immediate into Xd using MOVZ + up-to-three MOVKs. */
static void emit_mov_imm64(int d, int64_t val) {
    uint64_t u = (uint64_t)val;
    int placed = 0;
    for (int s = 0; s < 4; s++) {
        uint32_t chunk = (uint32_t)((u >> (s * 16)) & 0xFFFF);
        if (chunk == 0 && placed) continue;
        if (!placed) {
            emit_movz(d, chunk, s * 16);
            placed = 1;
        } else {
            emit_movk(d, chunk, s * 16);
        }
    }
    if (!placed) emit_movz(d, 0, 0); /* explicit zero */
}

/* MOV Xd, Xm == ORR Xd, XZR, Xm */
static void emit_mov_reg(int d, int m) {
    if (d == m) return;
    emit32(0xAA0003E0u | ((m & 31) << 16) | (d & 31));
}

/* ADD Xd, Xn, Xm (shifted register, no shift) */
static void emit_add_reg(int d, int n, int m) {
    emit32(0x8B000000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* SUB Xd, Xn, Xm */
static void emit_sub_reg(int d, int n, int m) {
    emit32(0xCB000000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR) */
static void emit_mul_reg(int d, int n, int m) {
    emit32(0x9B000000u | ((m & 31) << 16) | (31 << 10) | ((n & 31) << 5) | (d & 31));
}
/* SDIV Xd, Xn, Xm */
static void emit_sdiv_reg(int d, int n, int m) {
    emit32(0x9AC00C00u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* AND Xd, Xn, Xm */
static void emit_and_reg(int d, int n, int m) {
    emit32(0x8A000000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* ORR Xd, Xn, Xm */
static void emit_orr_reg(int d, int n, int m) {
    emit32(0xAA000000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* EOR Xd, Xn, Xm */
static void emit_eor_reg(int d, int n, int m) {
    emit32(0xCA000000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* LSLV / LSRV / ASRV aliases */
static void emit_lsl_reg(int d, int n, int m) {
    emit32(0x9AC02000u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
static void emit_lsr_reg(int d, int n, int m) {
    emit32(0x9AC02400u | ((m & 31) << 16) | ((n & 31) << 5) | (d & 31));
}
/* NEG Xd, Xm == SUB Xd, XZR, Xm */
static void emit_neg_reg(int d, int m) {
    emit32(0xCB0003E0u | ((m & 31) << 16) | (d & 31));
}

/* CMP Xn, Xm == SUBS XZR, Xn, Xm */
static void emit_cmp_reg(int n, int m) {
    emit32(0xEB00001Fu | ((m & 31) << 16) | ((n & 31) << 5));
}

/* LDUR Xt, [Xn, #simm9] — unscaled signed 9-bit offset */
static void emit_ldur(int t, int n, int simm9) {
    uint32_t imm = ((uint32_t)simm9) & 0x1FF;
    emit32(0xF8400000u | (imm << 12) | ((n & 31) << 5) | (t & 31));
}
/* STUR Xt, [Xn, #simm9] */
static void emit_stur(int t, int n, int simm9) {
    uint32_t imm = ((uint32_t)simm9) & 0x1FF;
    emit32(0xF8000000u | (imm << 12) | ((n & 31) << 5) | (t & 31));
}

/* STP/LDP X29, X30 with pre/post index for the classic prologue/epilogue */
static void emit_stp_x29_x30_pre(int simm7_bytes) {
    /* STP X29, X30, [SP, #imm]! — 64-bit pre-indexed.
     * Encoding: 1010 1001 10 sss sss ss Rt2 Rn Rt
     *   Template 0xA9800000 with pre-index bits.
     *   Actually STP (pre-indexed) = 0xA9800000 | imm7(signed)<<15 | Rt2<<10 | Rn<<5 | Rt
     * For STP X29, X30, [SP, #-16]! → 0xA9BF7BFD. */
    int imm7 = (simm7_bytes >> 3) & 0x7F;
    uint32_t w = 0xA9800000u
               | ((uint32_t)imm7 << 15)
               | (30u << 10)           /* X30 */
               | (31u << 5)            /* SP  */
               | 29u;                  /* X29 */
    emit32(w);
}
static void emit_ldp_x29_x30_post(int simm7_bytes) {
    /* LDP X29, X30, [SP], #imm — 64-bit post-indexed.
     * Template 0xA8C00000 | imm7<<15 | Rt2<<10 | Rn<<5 | Rt.
     * For LDP X29, X30, [SP], #16 → 0xA8C17BFD. */
    int imm7 = (simm7_bytes >> 3) & 0x7F;
    uint32_t w = 0xA8C00000u
               | ((uint32_t)imm7 << 15)
               | (30u << 10)
               | (31u << 5)
               | 29u;
    emit32(w);
}

/* SUB SP, SP, #imm12 (immediate, no shift) */
static void emit_sub_sp_imm(int imm12) {
    emit32(0xD1000000u | ((uint32_t)(imm12 & 0xFFF) << 10) | (31u << 5) | 31u);
}
/* MOV SP, X29  (ADD SP, X29, #0) */
static void emit_mov_sp_x29(void) {
    emit32(0x910003A0u | (31u << 0));  /* Rd=31(SP), Rn=29(X29), imm=0 */
}

/* RET (X30) */
static void emit_ret(void) { emit32(0xD65F03C0u); }

/* SVC #imm16 */
static void emit_svc_imm(uint32_t imm16) {
    emit32(0xD4000001u | ((imm16 & 0xFFFF) << 5));
}

/* B #imm26 (unconditional) — relative, imm26 is signed 26-bit word offset. */
static void emit_b_rel(int word_off) {
    uint32_t imm = ((uint32_t)word_off) & 0x3FFFFFF;
    emit32(0x14000000u | imm);
}

/* B.cond #imm19 (signed 19-bit word offset) */
static void emit_bcond_rel(int cond, int word_off) {
    uint32_t imm = ((uint32_t)word_off) & 0x7FFFF;
    emit32(0x54000000u | (imm << 5) | (cond & 0xF));
}

/* CBZ Xt, #imm19 */
static void emit_cbz_rel(int t, int word_off) {
    uint32_t imm = ((uint32_t)word_off) & 0x7FFFF;
    emit32(0xB4000000u | (imm << 5) | (t & 31));
}

/* BL #imm26 */
static void emit_bl_rel(int word_off) {
    uint32_t imm = ((uint32_t)word_off) & 0x3FFFFFF;
    emit32(0x94000000u | imm);
}

/* Patch helpers.  All `addr` arguments are byte offsets into g_code. */
static void patch_b(int addr, int target_addr) {
    int word_off = (target_addr - addr) / 4;
    uint32_t w = g_code[addr / 4];
    w = (w & 0xFC000000u) | (((uint32_t)word_off) & 0x3FFFFFF);
    g_code[addr / 4] = w;
}
static void patch_bcond(int addr, int target_addr) {
    int word_off = (target_addr - addr) / 4;
    uint32_t w = g_code[addr / 4];
    w = (w & 0xFF00001Fu) | ((((uint32_t)word_off) & 0x7FFFF) << 5);
    g_code[addr / 4] = w;
}
static void patch_cbz(int addr, int target_addr) {
    int word_off = (target_addr - addr) / 4;
    uint32_t w = g_code[addr / 4];
    w = (w & 0xFF00001Fu) | ((((uint32_t)word_off) & 0x7FFFF) << 5);
    g_code[addr / 4] = w;
}

/* ARM64 condition codes */
enum { CC_EQ=0, CC_NE=1, CC_GE=10, CC_LT=11, CC_GT=12, CC_LE=13 };

/* ============================================================================
 * Expression temp allocator
 *
 * Temps live in X9..X18 (ten registers).  One expression gets up to ten;
 * we reset at every statement boundary.  No spilling.
 * ============================================================================ */

#define TMP_FIRST 9
#define TMP_LAST  18
static int g_next_tmp = TMP_FIRST;

static int tmp_alloc(void) {
    if (g_next_tmp > TMP_LAST)
        die("expression too complex (>10 live temps)");
    return g_next_tmp++;
}
static void tmp_reset(void) { g_next_tmp = TMP_FIRST; }

/* ============================================================================
 * Frame state — per composition
 * ============================================================================ */

#define FRAME_SIZE 512    /* fixed frame bytes for every composition */

static int g_frame_used = 0;  /* how many 8-byte slots currently used */

/* Allocate a stack slot (8 bytes) for a new local.  Returns the positive
 * byte offset from X29 — actual address is [X29, #-offset]. */
static int frame_alloc_slot(void) {
    g_frame_used += 8;
    if (g_frame_used > FRAME_SIZE)
        die("frame overflow (too many locals per composition)");
    return g_frame_used;
}

static void local_load(int dest_reg, int offset) {
    /* LDUR Xt, [X29, #-offset] */
    emit_ldur(dest_reg, 29, -offset);
}
static void local_store(int src_reg, int offset) {
    emit_stur(src_reg, 29, -offset);
}

/* ============================================================================
 * Token cursor helpers
 * ============================================================================ */

static Tok* peek(void)      { return &g_tokens[g_tp]; }
static Tok* peek_at(int k)  { return &g_tokens[g_tp + k]; }
static void advance(void)   { if (g_tokens[g_tp].type != TK_EOF) g_tp++; }

static int at_statement_end(void) {
    int t = peek()->type;
    return t == TK_NEWLINE || t == TK_EOF || t == TK_INDENT;
}

/* ============================================================================
 * Expression parser
 *
 * Precedence (low → high):
 *   bitwise-or  |
 *   bitwise-xor ^
 *   bitwise-and &
 *   shift       << >>
 *   add/sub     + -
 *   mul/div     * /
 *   unary       - (prefix)
 *   atom        INT | ident | ( expr ) | call | $reg | ↑ $reg
 *
 * Every non-terminal returns the ARM64 register holding the result (X9..X18).
 * ============================================================================ */

/* Forward */
static int parse_expr(void);

/* Parse a `$N` identifier — returns N (physical register index). */
static int parse_dollar_reg(void) {
    Tok* t = peek();
    if (t->type != TK_IDENT || t->len < 2 || t->text[0] != '$')
        die("expected $N register name");
    int n = 0;
    for (int i = 1; i < t->len; i++) {
        int c = t->text[i];
        if (c < '0' || c > '9') die("malformed $N register name");
        n = n * 10 + (c - '0');
    }
    if (n > 30) die("$N out of range (0..30)");
    advance();
    return n;
}

/* Emit a function call to a known composition.  Currently handles
 * parameter passing by evaluating each arg to a temp, then moving into
 * X0..X7 right before the BL.  Since arg temps may collide with X0..X7,
 * we materialise them on the caller's stack slots... except we don't
 * have caller's stack slots for arbitrary call sites.  Simpler scheme:
 * collect arg registers first, then move them into X0..X7 in order.
 * This works because the arg registers are in X9+ and X0..X7 are
 * disjoint from them. */
static int parse_call(Sym* comp) {
    /* Advance past the name (caller already matched ident). */
    advance();

    int argregs[8];
    int argcnt = 0;

    /* Collect argument expressions.  We require each argument to be a
     * simple atom on the comp-call line.  This matches Lithos usage:
     * `open 0 0 0`, `exit42`, etc.  Full expression args require
     * parentheses to disambiguate from the next statement. */
    while (!at_statement_end() && argcnt < comp->nparams) {
        int r;
        Tok* t = peek();
        if (t->type == TK_INT) {
            r = tmp_alloc();
            emit_mov_imm64(r, t->ival);
            advance();
        } else if (t->type == TK_IDENT) {
            Sym* s = sym_lookup(t->text, t->len);
            if (!s) dief("unknown identifier '%.*s' in call arg", t->len, t->text);
            if (s->kind != SYM_LOCAL)
                die("call arg must be a local");
            r = tmp_alloc();
            local_load(r, s->offset);
            advance();
        } else if (t->type == TK_LPAREN) {
            r = parse_expr();
        } else {
            break;
        }
        argregs[argcnt++] = r;
    }

    /* Move args into X0..Xn.  Walk in order; since X0..X7 are all
     * distinct from X9..X18, no conflict. */
    for (int i = 0; i < argcnt; i++) {
        emit_mov_reg(i, argregs[i]);
    }

    /* BL to the composition's code address. */
    int here = code_here();
    int word_off = (comp->code_addr - here) / 4;
    emit_bl_rel(word_off);

    /* Result lives in X0.  Move it into a fresh temp so the caller can
     * consume it as an expression. */
    int r = tmp_alloc();
    emit_mov_reg(r, 0);
    return r;
}

static int parse_atom(void) {
    Tok* t = peek();

    if (t->type == TK_INT) {
        int r = tmp_alloc();
        emit_mov_imm64(r, t->ival);
        advance();
        return r;
    }

    if (t->type == TK_MINUS) {
        /* unary minus */
        advance();
        int r = parse_atom();
        emit_neg_reg(r, r);
        return r;
    }

    if (t->type == TK_LPAREN) {
        advance();
        int r = parse_expr();
        if (peek()->type != TK_RPAREN)
            die("missing closing paren in expression");
        advance();
        return r;
    }

    if (t->type == TK_LOAD) {
        /* `↑ $N` — read physical register N into a temp. */
        advance();
        int n = parse_dollar_reg();
        int r = tmp_alloc();
        emit_mov_reg(r, n);
        return r;
    }

    if (t->type == TK_IDENT) {
        /* `$N`: physical register read. */
        if (t->len >= 2 && t->text[0] == '$') {
            int n = parse_dollar_reg();
            int r = tmp_alloc();
            emit_mov_reg(r, n);
            return r;
        }
        /* Identifier lookup. */
        Sym* s = sym_lookup(t->text, t->len);
        if (!s) dief("unknown identifier '%.*s'", t->len, t->text);
        if (s->kind == SYM_COMP) {
            return parse_call(s);
        }
        /* Local read. */
        int r = tmp_alloc();
        local_load(r, s->offset);
        advance();
        return r;
    }

    dief("unexpected token in expression (type=%d)", t->type);
    return 0;
}

static int parse_muldiv(void) {
    int l = parse_atom();
    for (;;) {
        int op = peek()->type;
        if (op != TK_STAR && op != TK_SLASH) break;
        advance();
        int r = parse_atom();
        int d = tmp_alloc();
        if (op == TK_STAR) emit_mul_reg(d, l, r);
        else               emit_sdiv_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_addsub(void) {
    int l = parse_muldiv();
    for (;;) {
        int op = peek()->type;
        if (op != TK_PLUS && op != TK_MINUS) break;
        advance();
        int r = parse_muldiv();
        int d = tmp_alloc();
        if (op == TK_PLUS) emit_add_reg(d, l, r);
        else                emit_sub_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_shift(void) {
    int l = parse_addsub();
    for (;;) {
        int op = peek()->type;
        if (op != TK_SHL && op != TK_SHR) break;
        advance();
        int r = parse_addsub();
        int d = tmp_alloc();
        if (op == TK_SHL) emit_lsl_reg(d, l, r);
        else               emit_lsr_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_bitand(void) {
    int l = parse_shift();
    while (peek()->type == TK_AMP) {
        advance();
        int r = parse_shift();
        int d = tmp_alloc();
        emit_and_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_bitxor(void) {
    int l = parse_bitand();
    while (peek()->type == TK_CARET) {
        advance();
        int r = parse_bitand();
        int d = tmp_alloc();
        emit_eor_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_bitor(void) {
    int l = parse_bitxor();
    while (peek()->type == TK_PIPE) {
        advance();
        int r = parse_bitxor();
        int d = tmp_alloc();
        emit_orr_reg(d, l, r);
        l = d;
    }
    return l;
}

static int parse_expr(void) { return parse_bitor(); }

/* ============================================================================
 * Statement parser
 * ============================================================================ */

/* Skip a run of TK_NEWLINE / TK_INDENT(0) tokens.  Leaves the cursor at the
 * first real token or the first TK_INDENT(>0). */
static void skip_blank_lines(void) {
    for (;;) {
        int t = peek()->type;
        if (t == TK_NEWLINE) { advance(); continue; }
        if (t == TK_INDENT && peek()->col == 0) {
            /* Blank at column 0: only blank if followed by NEWLINE/EOF. */
            int next = peek_at(1)->type;
            if (next == TK_NEWLINE || next == TK_EOF) {
                advance(); continue;
            }
        }
        return;
    }
}

/* Return the indent of the next statement (peek, don't consume). */
static int peek_indent(void) {
    skip_blank_lines();
    if (peek()->type == TK_INDENT) return peek()->col;
    return 0;
}

/* Forward declarations for the mutually-recursive chunks. */
static void parse_body(int min_indent);
static void parse_statement(void);

/* Darwin syscall remapping: translate Linux syscall numbers (in X8) to the
 * corresponding Darwin values for X16.  Unknown numbers pass through. */
static int darwin_syscall_remap(int linux_nr) {
    switch (linux_nr) {
        case  93: return  1;   /* exit    */
        case  64: return  4;   /* write   */
        case  63: return  3;   /* read    */
        case  56: return 463;  /* openat  */
        case  57: return  6;   /* close   */
        case 222: return 197;  /* mmap    */
        case 214: return 12;   /* brk     */
        default:  return linux_nr;
    }
}

/* `↓ $N expr` — write to physical register N.  Special case: if N==8,
 * remap to X16 with Darwin syscall numbers. */
static void handle_reg_write(void) {
    advance(); /* ↓ */
    int n = parse_dollar_reg();

    int dest = n;
    int do_remap = (n == 8);

    /* Fast path: integer literal. */
    if (peek()->type == TK_INT) {
        int64_t v = peek()->ival;
        advance();
        if (do_remap) { dest = 16; v = darwin_syscall_remap((int)v); }
        emit_mov_imm64(dest, v);
        return;
    }
    /* Fast path: negative integer literal. */
    if (peek()->type == TK_MINUS && peek_at(1)->type == TK_INT) {
        advance();
        int64_t v = -peek()->ival;
        advance();
        if (do_remap) { dest = 16; v = darwin_syscall_remap((int)v); }
        emit_mov_imm64(dest, v);
        return;
    }

    /* General expression — evaluate into a temp and MOV. */
    int r = parse_expr();
    if (do_remap) dest = 16;
    emit_mov_reg(dest, r);
}

/* `↑ $N` as a statement: do nothing, it's just a load with no destination. */
static void handle_reg_read_stmt(void) {
    advance();              /* ↑ */
    (void)parse_dollar_reg();
}

/* `var name expr` */
static void handle_var(void) {
    advance(); /* var */
    Tok* t = peek();
    if (t->type != TK_IDENT) die("expected name after 'var'");
    const char* nm = t->text; int nl = t->len;
    advance();

    int r = parse_expr();
    Sym* s = sym_add(nm, nl, SYM_LOCAL);
    s->offset = frame_alloc_slot();
    local_store(r, s->offset);
}

/* `name = expr` or `name expr` binding/reassignment. */
static void handle_ident_stmt(void) {
    Tok* name = peek();
    Sym* s = sym_lookup(name->text, name->len);

    /* If it's a known composition, it's a bare call. */
    if (s && s->kind == SYM_COMP) {
        (void)parse_call(s);
        return;
    }

    /* Peek past the name. */
    int next = peek_at(1)->type;

    /* `name = expr`  reassignment with `=` */
    if (next == TK_EQ) {
        const char* nm = name->text; int nl = name->len;
        advance(); advance(); /* name = */
        int r = parse_expr();
        if (s && s->kind == SYM_LOCAL) {
            local_store(r, s->offset);
        } else {
            Sym* ns = sym_add(nm, nl, SYM_LOCAL);
            ns->offset = frame_alloc_slot();
            local_store(r, ns->offset);
        }
        return;
    }

    /* `name expr` — binding of a new local (no existing sym) or
     * reassignment of an existing one. */
    if (next == TK_INT || next == TK_IDENT || next == TK_MINUS ||
        next == TK_LPAREN || next == TK_LOAD) {
        const char* nm = name->text; int nl = name->len;
        advance();
        int r = parse_expr();
        if (s && s->kind == SYM_LOCAL) {
            local_store(r, s->offset);
        } else {
            Sym* ns = sym_add(nm, nl, SYM_LOCAL);
            ns->offset = frame_alloc_slot();
            local_store(r, ns->offset);
        }
        return;
    }

    /* Bare ident (no args) followed by newline — could be a zero-arg call
     * to a not-yet-seen composition.  We don't support forward refs in v1,
     * so error out clearly. */
    dief("unresolved identifier '%.*s' as bare statement", name->len, name->text);
}

/* Helper: parse the body of an if/for/while/composition — everything at
 * strictly greater indent than `base_indent`. */
static void parse_body(int base_indent) {
    skip_blank_lines();
    int body_indent = peek_indent();
    if (body_indent <= base_indent) return;

    while (peek()->type != TK_EOF) {
        skip_blank_lines();
        if (peek()->type == TK_EOF) break;
        if (peek()->type != TK_INDENT) break;
        int col = peek()->col;
        if (col < body_indent) break;        /* dedent to caller level */
        advance();                            /* consume INDENT */
        tmp_reset();
        parse_statement();
    }
}

/* if-statement.  Supports both `if expr : body` (truthy check) and
 * `if<op> a b` compound forms, plus `elif` / `else`.
 *
 * The grammar in the test files also accepts `if a == b` (operator in the
 * middle) — we recognise that too because the lexer already produces TK_EQEQ
 * for `==` and we can detect it after the first atom. */
static void handle_if(void) {
    advance(); /* if */

    /* Compound-if with operator first: `if< a b`, `if>= a b`, etc. */
    int tt = peek()->type;
    int is_compound_first = (tt == TK_LT || tt == TK_GT || tt == TK_LTE ||
                             tt == TK_GTE || tt == TK_EQEQ || tt == TK_NEQ);

    int cond_addr = 0;      /* address of the branch we'll patch past body */
    int cond_kind = 0;      /* 0 = CBZ, 1 = B.cond */

    if (is_compound_first) {
        int op = tt;
        advance();
        int la = parse_expr();
        int rb = parse_expr();
        emit_cmp_reg(la, rb);
        /* Emit B.(inverted cond) past body. */
        int cc;
        switch (op) {
            case TK_LT:   cc = CC_GE; break;
            case TK_LTE:  cc = CC_GT; break;
            case TK_GT:   cc = CC_LE; break;
            case TK_GTE:  cc = CC_LT; break;
            case TK_EQEQ: cc = CC_NE; break;
            case TK_NEQ:  cc = CC_EQ; break;
            default: cc = CC_NE;
        }
        cond_addr = code_here();
        emit_bcond_rel(cc, 0);
        cond_kind = 1;
    } else {
        /* Parse left expression.  If next is a comparator, fold into
         * compound form (`if a == b`).  Otherwise treat as truthy CBZ. */
        int la = parse_expr();
        int tt2 = peek()->type;
        if (tt2 == TK_EQEQ || tt2 == TK_NEQ || tt2 == TK_LT ||
            tt2 == TK_GT   || tt2 == TK_LTE || tt2 == TK_GTE) {
            int op = tt2; advance();
            int rb = parse_expr();
            emit_cmp_reg(la, rb);
            int cc;
            switch (op) {
                case TK_LT:   cc = CC_GE; break;
                case TK_LTE:  cc = CC_GT; break;
                case TK_GT:   cc = CC_LE; break;
                case TK_GTE:  cc = CC_LT; break;
                case TK_EQEQ: cc = CC_NE; break;
                case TK_NEQ:  cc = CC_EQ; break;
                default: cc = CC_NE;
            }
            cond_addr = code_here();
            emit_bcond_rel(cc, 0);
            cond_kind = 1;
        } else {
            /* truthy check: CBZ la, end_of_then */
            cond_addr = code_here();
            emit_cbz_rel(la, 0);
            cond_kind = 0;
        }
    }

    /* Optional `:` */
    if (peek()->type == TK_COLON) advance();

    /* Body is anything indented strictly beyond the if line.
     * parse_body doesn't know our own indent, so we infer it from the
     * next INDENT token.  We assume the if was at some indent K, and
     * the body is at indent > K.  We pass K = peek_indent_before_body
     * as the minimum the body must exceed.  In practice the body just
     * continues until dedent. */
    /* We use a trick: record our own indent as "the indent of the body
     * minus 1".  Since parse_body only checks `col > base_indent`, any
     * value strictly less than the body's indent works. */
    parse_body(0);   /* 0: any indent = body */

    /* Emit B past else (placeholder). */
    int end_addr = code_here();
    emit_b_rel(0);

    /* Patch cond branch to here (start of else). */
    int else_start = code_here();
    if (cond_kind == 0) patch_cbz(cond_addr, else_start);
    else                patch_bcond(cond_addr, else_start);

    /* Check for elif/else. */
    skip_blank_lines();
    /* elif / else must appear at same indent as the if — since we're
     * after parse_body and peek() may be at INDENT.  Peek past INDENT. */
    int saved_tp = g_tp;
    if (peek()->type == TK_INDENT) advance();
    int nt = peek()->type;
    if (nt == TK_ELIF) {
        /* Recurse: elif = nested if */
        handle_if();
    } else if (nt == TK_ELSE) {
        advance();
        if (peek()->type == TK_COLON) advance();
        parse_body(0);
    } else {
        /* No else — rewind. */
        g_tp = saved_tp;
    }

    /* Patch the end branch. */
    int here = code_here();
    patch_b(end_addr, here);
}

/* `for i start end [step] : body` — counted loop.  `i` becomes a local. */
static void handle_for(void) {
    advance(); /* for */

    Tok* t = peek();
    if (t->type != TK_IDENT) die("expected loop variable after 'for'");
    const char* nm = t->text; int nl = t->len;
    advance();

    int start_r = parse_expr();
    int end_r   = parse_expr();

    /* Step is optional; default 1. */
    int step_r;
    if (!at_statement_end() && peek()->type != TK_COLON) {
        step_r = parse_expr();
    } else {
        step_r = tmp_alloc();
        emit_mov_imm64(step_r, 1);
    }

    /* Bind loop variable. */
    Sym* iv = sym_add(nm, nl, SYM_LOCAL);
    iv->offset = frame_alloc_slot();
    local_store(start_r, iv->offset);

    /* Move end/step into stable temps (in X9..X18 range).  They're
     * already there from parse_expr, but we need to keep them live across
     * the body.  Since parse_body calls tmp_reset() on each statement,
     * the body won't preserve our end/step values in registers.  Store
     * them in locals too. */
    Sym* end_sym  = sym_add("__for_end",  9, SYM_LOCAL);
    end_sym->offset = frame_alloc_slot();
    local_store(end_r, end_sym->offset);

    Sym* step_sym = sym_add("__for_step", 10, SYM_LOCAL);
    step_sym->offset = frame_alloc_slot();
    local_store(step_r, step_sym->offset);

    if (peek()->type == TK_COLON) advance();

    /* loop top */
    int loop_top = code_here();

    /* Load i and end, compare. */
    tmp_reset();
    int i_r = tmp_alloc(); local_load(i_r, iv->offset);
    int e_r = tmp_alloc(); local_load(e_r, end_sym->offset);
    emit_cmp_reg(i_r, e_r);
    int exit_addr = code_here();
    emit_bcond_rel(CC_GE, 0);

    /* Body. */
    parse_body(0);

    /* i += step */
    tmp_reset();
    int i2 = tmp_alloc(); local_load(i2, iv->offset);
    int s2 = tmp_alloc(); local_load(s2, step_sym->offset);
    int sum = tmp_alloc();
    emit_add_reg(sum, i2, s2);
    local_store(sum, iv->offset);

    /* Jump back to top. */
    int here = code_here();
    int word_off = (loop_top - here) / 4;
    emit_b_rel(word_off);

    /* Patch exit. */
    patch_bcond(exit_addr, code_here());
}

static void handle_while(void) {
    advance(); /* while */
    int loop_top = code_here();
    int cond_r = parse_expr();
    if (peek()->type == TK_COLON) advance();
    int exit_addr = code_here();
    emit_cbz_rel(cond_r, 0);
    parse_body(0);
    int here = code_here();
    int word_off = (loop_top - here) / 4;
    emit_b_rel(word_off);
    patch_cbz(exit_addr, code_here());
}

static void handle_return(void) {
    advance(); /* return */
    if (!at_statement_end()) {
        int r = parse_expr();
        emit_mov_reg(0, r);
    }
    emit_mov_sp_x29();
    emit_ldp_x29_x30_post(16);
    emit_ret();
}

static void handle_trap(void) {
    advance();
    emit_svc_imm(0x80);
}

static void parse_statement(void) {
    Tok* t = peek();
    tmp_reset();

    switch (t->type) {
        case TK_STORE:   handle_reg_write(); break;
        case TK_LOAD:    handle_reg_read_stmt(); break;
        case TK_VAR:     handle_var(); break;
        case TK_IF:      handle_if(); break;
        case TK_FOR:     handle_for(); break;
        case TK_WHILE:   handle_while(); break;
        case TK_RETURN:  handle_return(); break;
        case TK_TRAP:    handle_trap(); break;
        case TK_IDENT:   handle_ident_stmt(); break;
        case TK_NEWLINE: advance(); break;
        case TK_INDENT:  advance(); break;
        default:
            /* Skip unknown. */
            advance();
            break;
    }
    /* Consume trailing NEWLINE if any. */
    while (peek()->type == TK_NEWLINE) advance();
}

/* ============================================================================
 * Top-level composition parser
 *
 *   name param1 param2 ... :
 *       body
 *
 * Registers the composition in the symbol table at its code address, emits
 * prologue + body + epilogue, pops composition-local symbols. */
static void parse_composition(void) {
    Tok* name = peek();
    if (name->type != TK_IDENT) die("expected composition name at top level");
    const char* nm = name->text; int nl = name->len;
    advance();

    /* Parameters: identifiers up to `:` (or newline for zero-arg). */
    const char* pnames[8]; int plens[8]; int nparams = 0;
    while (peek()->type == TK_IDENT && nparams < 8) {
        pnames[nparams] = peek()->text;
        plens[nparams]  = peek()->len;
        nparams++;
        advance();
    }

    if (peek()->type != TK_COLON)
        die("expected ':' after composition header");
    advance();

    /* Register composition symbol BEFORE emitting prologue so recursive
     * calls can resolve.  Record current code pos. */
    Sym* comp = sym_add(nm, nl, SYM_COMP);
    comp->code_addr = code_here();
    comp->nparams = nparams;

    /* New scope. */
    int saved_nsym = g_nsym;
    int saved_frame = g_frame_used;
    int saved_scope = g_scope;
    g_scope++;
    g_frame_used = 0;

    /* Prologue.
     *   STP X29, X30, [SP, #-16]!
     *   MOV X29, SP
     *   SUB SP, SP, #FRAME_SIZE
     */
    emit_stp_x29_x30_pre(-16);
    emit_mov_sp_x29();
    emit_sub_sp_imm(FRAME_SIZE);

    /* Spill parameters X0..X(n-1) to locals. */
    for (int i = 0; i < nparams; i++) {
        Sym* p = sym_add(pnames[i], plens[i], SYM_LOCAL);
        p->offset = frame_alloc_slot();
        local_store(i, p->offset);
    }

    /* Body. */
    parse_body(0);

    /* Epilogue.
     *   MOV SP, X29
     *   LDP X29, X30, [SP], #16
     *   RET
     * (If the last statement was `trap`, the RET is unreachable but
     * harmless.) */
    emit_mov_sp_x29();
    emit_ldp_x29_x30_post(16);
    emit_ret();

    /* Pop locals (keep the composition symbol). */
    sym_pop_scope(saved_nsym + 1);
    g_frame_used = saved_frame;
    g_scope = saved_scope;
}

/* ============================================================================
 * Top-level file parser
 * ============================================================================ */

/* Parse a complete file.  Emits a small entry stub at code offset 0 that
 * simply jumps to (or calls) `main`.  If no `main` is defined we call the
 * first top-level composition. */
static void parse_file(void) {
    /* Leave the first instruction as a placeholder branch — we'll patch it
     * to point at main once we know its address.  BL would also work, but
     * an unconditional B is simpler and doesn't require the main to RET
     * (it can trap and exit the process). */
    int entry_b = code_here();
    emit_b_rel(0);

    /* Walk top-level: each line starting with a non-blank column-0 token
     * is a composition header.  Also accept loose statements at column 0
     * (for test-trap.ls which is just `trap`). */
    for (;;) {
        skip_blank_lines();
        if (peek()->type == TK_EOF) break;

        /* Strip a leading INDENT(0). */
        if (peek()->type == TK_INDENT && peek()->col == 0) advance();

        if (peek()->type == TK_EOF) break;

        if (peek()->type == TK_IDENT) {
            /* Could be a composition header or a bare statement.
             * Look ahead for `:` after identifiers — signals composition. */
            int save = g_tp;
            advance();
            while (peek()->type == TK_IDENT) advance();
            int is_comp = (peek()->type == TK_COLON);
            g_tp = save;
            if (is_comp) {
                parse_composition();
                continue;
            }
        }

        /* Fallback: treat as a loose statement.  Wrap it in an implicit
         * `__entry` composition that just runs and traps. */
        if (sym_lookup("main", 4) == NULL) {
            Sym* comp = sym_add("main", 4, SYM_COMP);
            comp->code_addr = code_here();
            comp->nparams = 0;

            int saved_nsym = g_nsym;
            int saved_frame = g_frame_used;
            g_scope++;
            g_frame_used = 0;

            emit_stp_x29_x30_pre(-16);
            emit_mov_sp_x29();
            emit_sub_sp_imm(FRAME_SIZE);

            /* Parse every remaining top-level statement as the main body. */
            while (peek()->type != TK_EOF) {
                skip_blank_lines();
                if (peek()->type == TK_EOF) break;
                if (peek()->type == TK_INDENT) advance();
                tmp_reset();
                parse_statement();
            }

            emit_mov_sp_x29();
            emit_ldp_x29_x30_post(16);
            emit_ret();

            sym_pop_scope(saved_nsym + 1);
            g_frame_used = saved_frame;
            g_scope--;
            continue;
        }

        /* Something else at top level we don't understand — skip. */
        advance();
    }

    /* Patch the entry branch to jump to `main`. */
    Sym* main_sym = sym_lookup("main", 4);
    if (!main_sym || main_sym->kind != SYM_COMP)
        die("no `main` composition defined");
    patch_b(entry_b, main_sym->code_addr);
}

/* ============================================================================
 * SHA-256 (RFC 6234).  Needed to ad-hoc sign the output binary.
 * ============================================================================ */

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    uint8_t  buf[64];
    size_t   len;
} Sha256;

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_compress(Sha256* s, const uint8_t* blk) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i*4] << 24) | ((uint32_t)blk[i*4+1] << 16) |
               ((uint32_t)blk[i*4+2] << 8) | (uint32_t)blk[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19)  ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint32_t e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void sha256_init(Sha256* s) {
    s->h[0]=0x6a09e667; s->h[1]=0xbb67ae85; s->h[2]=0x3c6ef372; s->h[3]=0xa54ff53a;
    s->h[4]=0x510e527f; s->h[5]=0x9b05688c; s->h[6]=0x1f83d9ab; s->h[7]=0x5be0cd19;
    s->bits = 0; s->len = 0;
}
static void sha256_update(Sha256* s, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    s->bits += (uint64_t)n * 8;
    while (n > 0) {
        size_t take = 64 - s->len;
        if (take > n) take = n;
        memcpy(s->buf + s->len, p, take);
        s->len += take; p += take; n -= take;
        if (s->len == 64) {
            sha256_compress(s, s->buf);
            s->len = 0;
        }
    }
}
static void sha256_final(Sha256* s, uint8_t out[32]) {
    uint64_t bits = s->bits;
    uint8_t pad = 0x80;
    sha256_update(s, &pad, 1);
    uint8_t zero = 0;
    while (s->len != 56) sha256_update(s, &zero, 1);
    uint8_t bits_be[8];
    for (int i = 0; i < 8; i++) bits_be[i] = (uint8_t)(bits >> ((7 - i) * 8));
    sha256_update(s, bits_be, 8);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(s->h[i] >> 24);
        out[i*4+1] = (uint8_t)(s->h[i] >> 16);
        out[i*4+2] = (uint8_t)(s->h[i] >> 8);
        out[i*4+3] = (uint8_t)(s->h[i]);
    }
}

static void sha256_buf(const void* data, size_t n, uint8_t out[32]) {
    Sha256 s; sha256_init(&s); sha256_update(&s, data, n); sha256_final(&s, out);
}

/* ============================================================================
 * Mach-O writer
 *
 * We produce a dyld-linked Mach-O executable because modern macOS
 * (Sequoia+) refuses to run pure-static LC_UNIXTHREAD binaries under the
 * AppleSystemPolicy gatekeeper.  The binary is ad-hoc signed in-place.
 *
 * File layout:
 *
 *    0:   mach_header_64           (32 bytes)
 *    32:  load commands            (~624 bytes)
 *    ??:  padding to 16-byte align
 *    code_offset:
 *         [code bytes]                  (our emitted ARM64)
 *    round up to PAGE (16K) = linkedit_off
 *    linkedit_off:
 *         [LC_DYLD_CHAINED_FIXUPS blob]  (minimal — header with 0 fixups)
 *         [LC_SYMTAB string table blob]  (empty)
 *         [LC_CODE_SIGNATURE blob]       (ad-hoc, SHA-256 of file)
 *
 * The code runs under dyld.  dyld calls the entry point as a C-style
 * function: X0=argc, X1=argv, X2=envp, X3=apple, LR = dyld stub that
 * invokes exit() when main() returns.  We ignore the args and rely on
 * `trap` to exit; if the composition returns via RET, dyld exits too.
 *
 * Load commands (12 total):
 *   LC_SEGMENT_64 __PAGEZERO        (72)
 *   LC_SEGMENT_64 __TEXT (1 sect)   (152)
 *   LC_SEGMENT_64 __LINKEDIT        (72)
 *   LC_DYLD_CHAINED_FIXUPS          (16)
 *   LC_SYMTAB                       (24)
 *   LC_DYSYMTAB                     (80)
 *   LC_LOAD_DYLINKER                (32)
 *   LC_BUILD_VERSION                (32)
 *   LC_MAIN                         (24)
 *   LC_LOAD_DYLIB /usr/lib/libSystem.B.dylib  (56)
 *   LC_CODE_SIGNATURE               (16)
 * ============================================================================ */

#define MH_MAGIC_64   0xFEEDFACFu
#define CPU_TYPE_ARM64 0x0100000Cu
#define CPU_SUBTYPE_ARM64_ALL 0u
#define MH_EXECUTE    0x2u

#define MH_NOUNDEFS   0x1u
#define MH_DYLDLINK   0x4u
#define MH_TWOLEVEL   0x80u
#define MH_PIE        0x200000u

#define LC_SEGMENT_64          0x19u
#define LC_SYMTAB              0x2u
#define LC_DYSYMTAB            0xBu
#define LC_LOAD_DYLINKER       0xEu
#define LC_LOAD_DYLIB          0xCu
#define LC_BUILD_VERSION       0x32u
#define LC_MAIN                (0x28u | 0x80000000u)
#define LC_DYLD_CHAINED_FIXUPS (0x34u | 0x80000000u)
#define LC_CODE_SIGNATURE      0x1Du

#define VM_PROT_READ    1u
#define VM_PROT_EXECUTE 4u

#define PLATFORM_MACOS 1u

/* Code signing constants. */
#define CSMAGIC_EMBEDDED_SIGNATURE 0xfade0cc0u
#define CSMAGIC_CODEDIRECTORY      0xfade0c02u
#define CSMAGIC_REQUIREMENTS       0xfade0c01u
#define CS_ADHOC                   0x0002u
#define CS_LINKER_SIGNED           0x20000u
#define CS_EXECSEG_MAIN_BINARY     0x1u
#define CS_HASHTYPE_SHA256         2u
#define CS_SUPPORTSEXECSEG         0x20400u
#define CS_SLOT_CODEDIRECTORY      0u

/* Little-endian scalar writers */
static void w32(uint8_t** p, uint32_t v) {
    (*p)[0] = (uint8_t)(v);
    (*p)[1] = (uint8_t)(v >> 8);
    (*p)[2] = (uint8_t)(v >> 16);
    (*p)[3] = (uint8_t)(v >> 24);
    *p += 4;
}
static void w64(uint8_t** p, uint64_t v) {
    w32(p, (uint32_t)v);
    w32(p, (uint32_t)(v >> 32));
}
static void wstr16(uint8_t** p, const char* s) {
    uint8_t buf[16] = {0};
    size_t n = strlen(s);
    if (n > 16) n = 16;
    memcpy(buf, s, n);
    memcpy(*p, buf, 16);
    *p += 16;
}

/* Big-endian helpers for code signing blobs (all multibyte fields in the
 * signature are stored big-endian, regardless of host endianness). */
static void w32be(uint8_t** p, uint32_t v) {
    (*p)[0] = (uint8_t)(v >> 24);
    (*p)[1] = (uint8_t)(v >> 16);
    (*p)[2] = (uint8_t)(v >> 8);
    (*p)[3] = (uint8_t)(v);
    *p += 4;
}
static void w64be(uint8_t** p, uint64_t v) {
    w32be(p, (uint32_t)(v >> 32));
    w32be(p, (uint32_t)v);
}

/* Build the ad-hoc code signature blob.
 *
 *   identifier:      NUL-terminated ASCII name embedded in the CodeDirectory
 *   signed_data:     pointer to the bytes that should be hashed
 *   signed_len:      number of bytes to hash (must be < 4 GiB)
 *   exec_seg_limit:  size of the executable segment (__TEXT) in bytes
 *   out_size:        on return, total blob size in bytes
 *
 * Returns a freshly malloc'd buffer (caller frees).
 */
static uint8_t* build_signature(const char* identifier,
                                const uint8_t* signed_data,
                                uint32_t signed_len,
                                uint64_t exec_seg_limit,
                                uint32_t* out_size) {
    const uint32_t page_size = 4096;
    const uint32_t hash_size = 32;

    /* Number of hash slots = ceil(signed_len / page_size) */
    uint32_t nslots = (signed_len + page_size - 1) / page_size;

    /* CodeDirectory layout for version 0x20400 (supports execseg):
     *   40-byte fixed header up through pageSize (36 + 4 one-byte fields)
     *   4  spare2
     *   4  scatterOffset   (version >= 0x20100)
     *   4  teamOffset      (version >= 0x20200)
     *   4  spare3          (version >= 0x20300)
     *   8  codeLimit64     (version >= 0x20300)
     *   8  execSegBase     (version >= 0x20400)
     *   8  execSegLimit
     *   8  execSegFlags
     *   N  identifier (NUL-terminated)
     *   N  code hash slots (nslots * hash_size)
     *
     * Total fixed CD body = 40 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 8 = 88 bytes.
     */
    const uint32_t cd_fixed = 88;
    uint32_t ident_len = (uint32_t)strlen(identifier) + 1;  /* include NUL */
    uint32_t cd_len = cd_fixed + ident_len + nslots * hash_size;

    /* SuperBlob header = 12 bytes (magic, length, count).
     * Blob index = count * 8 bytes.
     * For a single CodeDirectory: 12 + 8 = 20 bytes before the CD. */
    const uint32_t sb_header = 12 + 8;   /* one blob index entry */
    uint32_t sb_len = sb_header + cd_len;

    uint8_t* blob = (uint8_t*)calloc(1, sb_len);
    if (!blob) die("out of memory (signature blob)");

    uint8_t* p = blob;
    /* SuperBlob header. */
    w32be(&p, CSMAGIC_EMBEDDED_SIGNATURE);
    w32be(&p, sb_len);
    w32be(&p, 1);                           /* count */
    /* Blob index entry [0]: CodeDirectory */
    w32be(&p, CS_SLOT_CODEDIRECTORY);
    w32be(&p, sb_header);                   /* offset of the CD within blob */

    /* CodeDirectory. */
    uint8_t* cd_start = p;
    w32be(&p, CSMAGIC_CODEDIRECTORY);
    w32be(&p, cd_len);
    w32be(&p, CS_SUPPORTSEXECSEG);          /* version = 0x20400 */
    w32be(&p, CS_ADHOC | CS_LINKER_SIGNED); /* flags */
    w32be(&p, cd_fixed + ident_len);        /* hashOffset (offset to first hash) */
    w32be(&p, cd_fixed);                    /* identOffset */
    w32be(&p, 0);                           /* nSpecialSlots */
    w32be(&p, nslots);                      /* nCodeSlots */
    w32be(&p, signed_len);                  /* codeLimit */
    *p++ = (uint8_t)hash_size;              /* hashSize */
    *p++ = (uint8_t)CS_HASHTYPE_SHA256;     /* hashType */
    *p++ = 0;                                /* platform */
    *p++ = 12;                               /* pageSize = log2(4096) */
    w32be(&p, 0);                           /* spare2 */
    w32be(&p, 0);                           /* scatterOffset */
    w32be(&p, 0);                           /* teamOffset */
    w32be(&p, 0);                           /* spare3 */
    w64be(&p, 0);                           /* codeLimit64 */
    w64be(&p, 0);                           /* execSegBase */
    w64be(&p, exec_seg_limit);              /* execSegLimit */
    w64be(&p, CS_EXECSEG_MAIN_BINARY);      /* execSegFlags */

    /* identifier string */
    memcpy(p, identifier, ident_len);
    p += ident_len;

    /* Code slot hashes: hash each 4K page of the signed data. */
    for (uint32_t i = 0; i < nslots; i++) {
        uint32_t start = i * page_size;
        uint32_t end = start + page_size;
        if (end > signed_len) end = signed_len;
        sha256_buf(signed_data + start, end - start, p);
        p += hash_size;
    }

    /* Sanity. */
    if ((uint32_t)(p - cd_start) != cd_len) {
        die("code signature blob size mismatch");
    }

    *out_size = sb_len;
    return blob;
}

/* Round a uint up to a power-of-two boundary. */
static uint64_t round_up(uint64_t x, uint64_t mul) {
    return ((x + mul - 1) / mul) * mul;
}

static void write_macho(const char* out_path, const uint8_t* code, int code_size) {
    const uint64_t PAGE = 0x4000;     /* 16K, arm64 default */

    /* Load command sizes (deterministic). */
    uint32_t cmdsize_pagezero      = 72;
    uint32_t cmdsize_text          = 152;   /* segment + 1 section */
    uint32_t cmdsize_linkedit      = 72;
    uint32_t cmdsize_symtab        = 24;
    uint32_t cmdsize_dysymtab      = 80;
    /* LC_LOAD_DYLINKER: 12-byte header + "/usr/lib/dyld\0" padded to 4
     * bytes.  Total = 12 + round_up(14, 4) = 12 + 16 = 28.  Clang uses 32. */
    uint32_t cmdsize_loaddylinker  = 32;
    uint32_t cmdsize_buildversion  = 32;   /* no tools */
    uint32_t cmdsize_main          = 24;
    /* LC_LOAD_DYLIB: 24-byte header + dylib name.  For
     * "/usr/lib/libSystem.B.dylib\0" (27 bytes), padded to 8 → 32.
     * Total = 24 + 32 = 56. */
    uint32_t cmdsize_loaddylib     = 56;
    uint32_t cmdsize_codesig       = 16;

    uint32_t ncmds = 10;
    uint32_t sizeof_header = 32;
    uint32_t sizeof_cmds = cmdsize_pagezero + cmdsize_text + cmdsize_linkedit
                         + cmdsize_symtab
                         + cmdsize_dysymtab + cmdsize_loaddylinker
                         + cmdsize_buildversion + cmdsize_main
                         + cmdsize_loaddylib + cmdsize_codesig;

    /* Code offset: right after the load commands, 16-byte aligned. */
    uint32_t code_offset = (uint32_t)round_up(sizeof_header + sizeof_cmds, 16);

    /* __TEXT size: round up code_offset + code_size to PAGE. */
    uint64_t text_end = (uint64_t)code_offset + (uint64_t)code_size;
    uint64_t text_file_size = round_up(text_end, PAGE);

    /* __LINKEDIT starts right after __TEXT. */
    uint32_t linkedit_off = (uint32_t)text_file_size;

    /* Inside __LINKEDIT we place:
     *   [symtab string blob] (empty — zero-length strtab, one NUL)
     *   [code signature blob]
     *
     * We need to know sig size in advance (since it's a deterministic
     * function of the signed bytes and identifier). */
    uint32_t symtab_stroff  = linkedit_off;
    uint32_t symtab_strsize = 8;                        /* minimum aligned */
    uint32_t codesig_off    = round_up(symtab_stroff + symtab_strsize, 16);

    const char* ident = "lithos-boot-dennis";
    uint32_t ident_len = (uint32_t)strlen(ident) + 1;
    uint32_t page4k = 4096;
    /* codeLimit = codesig_off — everything up to the signature is hashed. */
    uint32_t nslots = (codesig_off + page4k - 1) / page4k;
    uint32_t cd_len = 88 + ident_len + nslots * 32;
    uint32_t sig_size = 20 + cd_len;

    uint64_t linkedit_end = (uint64_t)codesig_off + sig_size;
    uint64_t linkedit_size = linkedit_end - linkedit_off;
    uint64_t linkedit_vmsize = round_up(linkedit_size, PAGE);
    uint64_t file_size = linkedit_end;

    uint64_t text_vmaddr = 0x100000000ull;
    uint64_t linkedit_vmaddr = text_vmaddr + text_file_size;

    uint8_t* buf = (uint8_t*)calloc(1, (size_t)file_size);
    if (!buf) die("out of memory (macho buffer)");

    uint8_t* p = buf;

    /* ---- mach_header_64 ---- */
    w32(&p, MH_MAGIC_64);
    w32(&p, CPU_TYPE_ARM64);
    w32(&p, CPU_SUBTYPE_ARM64_ALL);
    w32(&p, MH_EXECUTE);
    w32(&p, ncmds);
    w32(&p, sizeof_cmds);
    w32(&p, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);
    w32(&p, 0);

    /* ---- LC_SEGMENT_64 __PAGEZERO ---- */
    w32(&p, LC_SEGMENT_64);
    w32(&p, cmdsize_pagezero);
    wstr16(&p, "__PAGEZERO");
    w64(&p, 0);
    w64(&p, 0x100000000ull);
    w64(&p, 0);
    w64(&p, 0);
    w32(&p, 0); w32(&p, 0); w32(&p, 0); w32(&p, 0);

    /* ---- LC_SEGMENT_64 __TEXT (with __text section) ---- */
    w32(&p, LC_SEGMENT_64);
    w32(&p, cmdsize_text);
    wstr16(&p, "__TEXT");
    w64(&p, text_vmaddr);
    w64(&p, text_file_size);
    w64(&p, 0);
    w64(&p, text_file_size);
    w32(&p, VM_PROT_READ | VM_PROT_EXECUTE);
    w32(&p, VM_PROT_READ | VM_PROT_EXECUTE);
    w32(&p, 1);
    w32(&p, 0);
    /* section __text */
    wstr16(&p, "__text");
    wstr16(&p, "__TEXT");
    w64(&p, text_vmaddr + code_offset);
    w64(&p, (uint64_t)code_size);
    w32(&p, code_offset);
    w32(&p, 2);
    w32(&p, 0);
    w32(&p, 0);
    w32(&p, 0x80000400u);
    w32(&p, 0);
    w32(&p, 0);
    w32(&p, 0);

    /* ---- LC_SEGMENT_64 __LINKEDIT ---- */
    w32(&p, LC_SEGMENT_64);
    w32(&p, cmdsize_linkedit);
    wstr16(&p, "__LINKEDIT");
    w64(&p, linkedit_vmaddr);
    w64(&p, linkedit_vmsize);
    w64(&p, linkedit_off);
    w64(&p, linkedit_size);
    w32(&p, VM_PROT_READ);
    w32(&p, VM_PROT_READ);
    w32(&p, 0);
    w32(&p, 0);

    /* ---- LC_SYMTAB ---- */
    w32(&p, LC_SYMTAB);
    w32(&p, cmdsize_symtab);
    w32(&p, symtab_stroff);       /* symoff (== stroff since nsyms=0) */
    w32(&p, 0);                    /* nsyms */
    w32(&p, symtab_stroff);       /* stroff */
    w32(&p, symtab_strsize);      /* strsize */

    /* ---- LC_DYSYMTAB ---- */
    w32(&p, LC_DYSYMTAB);
    w32(&p, cmdsize_dysymtab);
    for (int i = 0; i < 18; i++) w32(&p, 0);  /* 72 bytes of zeroed fields */

    /* ---- LC_LOAD_DYLINKER ---- */
    w32(&p, LC_LOAD_DYLINKER);
    w32(&p, cmdsize_loaddylinker);
    w32(&p, 12);                              /* name offset (after header) */
    {
        const char* name = "/usr/lib/dyld";
        size_t n = strlen(name) + 1;
        memcpy(p, name, n);
        memset(p + n, 0, cmdsize_loaddylinker - 12 - n);
        p += cmdsize_loaddylinker - 12;
    }

    /* ---- LC_BUILD_VERSION ---- */
    w32(&p, LC_BUILD_VERSION);
    w32(&p, cmdsize_buildversion);
    w32(&p, PLATFORM_MACOS);       /* platform */
    w32(&p, (15u << 16) | (0u << 8));   /* minos  = 15.0.0 */
    w32(&p, (15u << 16) | (0u << 8));   /* sdk    = 15.0.0 */
    w32(&p, 0);                    /* ntools */
    w32(&p, 0);                    /* pad to 32 bytes */
    w32(&p, 0);

    /* ---- LC_MAIN ---- */
    w32(&p, LC_MAIN);
    w32(&p, cmdsize_main);
    w64(&p, (uint64_t)code_offset);   /* entryoff (file offset from __TEXT) */
    w64(&p, 0);                       /* stacksize */

    /* ---- LC_LOAD_DYLIB libSystem ---- */
    w32(&p, LC_LOAD_DYLIB);
    w32(&p, cmdsize_loaddylib);
    w32(&p, 24);                      /* name offset */
    w32(&p, 2);                       /* timestamp */
    w32(&p, (1356u << 16));           /* current_version = 1356.0.0 */
    w32(&p, (1u << 16));              /* compatibility_version = 1.0.0 */
    {
        const char* name = "/usr/lib/libSystem.B.dylib";
        size_t n = strlen(name) + 1;
        memcpy(p, name, n);
        memset(p + n, 0, cmdsize_loaddylib - 24 - n);
        p += cmdsize_loaddylib - 24;
    }

    /* ---- LC_CODE_SIGNATURE ---- */
    w32(&p, LC_CODE_SIGNATURE);
    w32(&p, cmdsize_codesig);
    w32(&p, codesig_off);
    w32(&p, sig_size);

    /* End of load commands — sanity check. */
    if ((uint32_t)(p - buf) != sizeof_header + sizeof_cmds)
        dief("Mach-O header size mismatch: %u vs %u",
             (uint32_t)(p - buf), sizeof_header + sizeof_cmds);

    /* ---- code at code_offset ---- */
    memcpy(buf + code_offset, code, code_size);

    /* ---- symtab strtab: one NUL byte to start (strtab[0] is unused) ---- */
    buf[symtab_stroff] = 0;

    /* ---- code signature blob ---- */
    uint32_t actual_sig_size;
    uint8_t* sig = build_signature(ident, buf, codesig_off,
                                   text_file_size, &actual_sig_size);
    if (actual_sig_size != sig_size) {
        dief("sig size mismatch: predicted %u, actual %u",
             sig_size, actual_sig_size);
    }
    memcpy(buf + codesig_off, sig, actual_sig_size);
    free(sig);

    /* Write to disk. */
    int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (fd < 0) { perror(out_path); exit(1); }
    ssize_t w = write(fd, buf, (size_t)file_size);
    if (w != (ssize_t)file_size) { perror("write"); exit(1); }
    close(fd);
    chmod(out_path, 0755);

    free(buf);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <source.ls> <output>\n", argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    /* Slurp source. */
    int fd = open(in_path, O_RDONLY);
    if (fd < 0) { perror(in_path); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
    size_t sz = (size_t)st.st_size;
    char* src = (char*)malloc(sz + 1);
    if (!src) die("out of memory (source)");
    ssize_t nread = read(fd, src, sz);
    if (nread < 0 || (size_t)nread != sz) { perror("read"); return 1; }
    src[sz] = 0;
    close(fd);

    g_src = src;
    lex(src, (int)sz);

    parse_file();

    int code_bytes = g_code_pos * 4;
    write_macho(out_path, (const uint8_t*)g_code, code_bytes);

    fprintf(stderr, "lithos-boot-dennis: wrote %s (%d code bytes, %d syms)\n",
            out_path, code_bytes, g_nsym);
    return 0;
}
