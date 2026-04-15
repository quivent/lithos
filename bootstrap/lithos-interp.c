/* lithos-interp.c — C port of the Lithos reference interpreter.
 *
 * Runs .ls source directly, no codegen.  1:1 port of lithos-interp.py.
 * Usage: ./lithos-interp source.ls [args...]
 *
 * Build: cc -O2 -o lithos-interp lithos-interp.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ── Token types ─────────────────────────────────────────────── */
enum {
    TOK_EOF, TOK_NEWLINE, TOK_INDENT, TOK_INT, TOK_IDENT,
    TOK_IF, TOK_ELIF, TOK_ELSE, TOK_FOR, TOK_WHILE,
    TOK_EACH, TOK_VAR, TOK_RETURN, TOK_TRAP, TOK_CONST,
    TOK_BUF, TOK_LOAD, TOK_REG_READ, TOK_MEM_STORE, TOK_MEM_LOAD,
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_AMP, TOK_PIPE, TOK_CARET, TOK_SHL, TOK_SHR,
    TOK_EQ, TOK_EQEQ, TOK_NEQ, TOK_LT, TOK_GT, TOK_LTE, TOK_GTE,
    TOK_LPAREN, TOK_RPAREN, TOK_LBRACK, TOK_RBRACK,
    TOK_COLON, TOK_DOLLAR, TOK_HASH,
    TOK_GOTO, TOK_LABEL, TOK_CONTINUE, TOK_BREAK,
    TOK_COUNT
};

#define is_op(t) ((t)>=TOK_PLUS && (t)<=TOK_SHR)
#define is_cmp(t) ((t)>=TOK_EQEQ && (t)<=TOK_GTE)
#define is_expr_start(t) ((t)==TOK_INT||(t)==TOK_IDENT||(t)==TOK_LPAREN|| \
                          (t)==TOK_MINUS||(t)==TOK_REG_READ||(t)==TOK_MEM_LOAD)

/* ── Token storage ───────────────────────────────────────────── */
typedef struct { int type; int off; int len; } Tok;
#define MAX_TOKS 200000
static Tok toks[MAX_TOKS];
static int ntoks;

/* ── Source ───────────────────────────────────────────────────── */
static uint8_t *src;
static int srclen;

/* ── Lexer ───────────────────────────────────────────────────── */
static int kw(const char *s, int n) {
    if (n==2 && !memcmp(s,"if",2)) return TOK_IF;
    if (n==4 && !memcmp(s,"elif",4)) return TOK_ELIF;
    if (n==4 && !memcmp(s,"else",4)) return TOK_ELSE;
    if (n==3 && !memcmp(s,"for",3)) return TOK_FOR;
    if (n==5 && !memcmp(s,"while",5)) return TOK_WHILE;
    if (n==4 && !memcmp(s,"each",4)) return TOK_EACH;
    if (n==3 && !memcmp(s,"var",3)) return TOK_VAR;
    if (n==6 && !memcmp(s,"return",6)) return TOK_RETURN;
    if (n==4 && !memcmp(s,"trap",4)) return TOK_TRAP;
    if (n==5 && !memcmp(s,"const",5)) return TOK_CONST;
    if (n==8 && !memcmp(s,"constant",8)) return TOK_CONST;
    if (n==3 && !memcmp(s,"buf",3)) return TOK_BUF;
    if (n==4 && !memcmp(s,"goto",4)) return TOK_GOTO;
    if (n==5 && !memcmp(s,"label",5)) return TOK_LABEL;
    if (n==8 && !memcmp(s,"continue",8)) return TOK_CONTINUE;
    if (n==5 && !memcmp(s,"break",5)) return TOK_BREAK;
    return TOK_IDENT;
}

static void emit(int tp, int off, int len) {
    if (ntoks < MAX_TOKS) { toks[ntoks].type=tp; toks[ntoks].off=off; toks[ntoks].len=len; ntoks++; }
}

static void do_lex(void) {
    int i=0, at_line_start=1;
    ntoks=0;
    while (i < srclen) {
        if (at_line_start) {
            int indent=0;
            while (i<srclen && src[i]==' ') { indent++; i++; }
            if (i>=srclen) break;
            if (src[i]=='\n') { at_line_start=0; continue; }
            if (i+1<srclen && src[i]=='\\' && src[i+1]=='\\') { at_line_start=0; continue; }
            emit(TOK_INDENT, i-indent, indent);
            at_line_start=0;
        }
        if (i>=srclen) break;
        uint8_t c = src[i];
        if (c==' '||c=='\t') { i++; continue; }
        if (c=='\n') { emit(TOK_NEWLINE,i,1); i++; at_line_start=1; continue; }
        if (i+1<srclen && c=='\\' && src[i+1]=='\\') {
            while (i<srclen && src[i]!='\n') i++;
            continue;
        }
        /* identifier / keyword */
        if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_') {
            int s=i;
            while (i<srclen && ((src[i]>='a'&&src[i]<='z')||(src[i]>='A'&&src[i]<='Z')||
                   (src[i]>='0'&&src[i]<='9')||src[i]=='_')) i++;
            emit(kw((char*)src+s,i-s), s, i-s);
            continue;
        }
        /* $N */
        if (c=='$') { int s=i; i++; while(i<srclen&&src[i]>='0'&&src[i]<='9')i++; emit(TOK_IDENT,s,i-s); continue; }
        /* integer */
        if (c>='0'&&c<='9') {
            int s=i;
            if (i+1<srclen && src[i]=='0' && (src[i+1]=='x'||src[i+1]=='X')) {
                i+=2;
                while(i<srclen&&((src[i]>='0'&&src[i]<='9')||(src[i]>='a'&&src[i]<='f')||(src[i]>='A'&&src[i]<='F')))i++;
            } else { while(i<srclen&&src[i]>='0'&&src[i]<='9')i++; }
            emit(TOK_INT,s,i-s); continue;
        }
        /* two-char ops */
        if (i+1<srclen) {
            uint8_t n=src[i+1];
            if (c=='='&&n=='=') { emit(TOK_EQEQ,i,2); i+=2; continue; }
            if (c=='!'&&n=='=') { emit(TOK_NEQ,i,2);  i+=2; continue; }
            if (c=='<'&&n=='=') { emit(TOK_LTE,i,2);  i+=2; continue; }
            if (c=='>'&&n=='=') { emit(TOK_GTE,i,2);  i+=2; continue; }
            if (c=='<'&&n=='<') { emit(TOK_SHL,i,2);  i+=2; continue; }
            if (c=='>'&&n=='>') { emit(TOK_SHR,i,2);  i+=2; continue; }
        }
        /* UTF-8 arrows */
        if (i+2<srclen && c==0xe2 && src[i+1]==0x86) {
            uint8_t b2=src[i+2];
            if (b2==0x93) { emit(TOK_LOAD,i,3);     i+=3; continue; }
            if (b2==0x91) { emit(TOK_REG_READ,i,3);  i+=3; continue; }
            if (b2==0x90) { emit(TOK_MEM_STORE,i,3); i+=3; continue; }
            if (b2==0x92) { emit(TOK_MEM_LOAD,i,3);  i+=3; continue; }
        }
        /* single-char ops */
        switch(c) {
        case '+': emit(TOK_PLUS,i,1);break; case '-': emit(TOK_MINUS,i,1);break;
        case '*': emit(TOK_STAR,i,1);break; case '/': emit(TOK_SLASH,i,1);break;
        case '=': emit(TOK_EQ,i,1);break;   case '<': emit(TOK_LT,i,1);break;
        case '>': emit(TOK_GT,i,1);break;   case '&': emit(TOK_AMP,i,1);break;
        case '|': emit(TOK_PIPE,i,1);break;  case '^': emit(TOK_CARET,i,1);break;
        case '(': emit(TOK_LPAREN,i,1);break; case ')': emit(TOK_RPAREN,i,1);break;
        case '[': emit(TOK_LBRACK,i,1);break; case ']': emit(TOK_RBRACK,i,1);break;
        case ':': emit(TOK_COLON,i,1);break;  case '#': emit(TOK_HASH,i,1);break;
        default: break;
        }
        i++;
    }
    emit(TOK_EOF,i,0);
}

/* ── Interpreter state ───────────────────────────────────────── */
#define MEM_SIZE (1<<30)
static uint8_t *mem;
static int64_t mem_top = 8;

static int64_t regs[32];
static int tk; /* current token index */

#define MASK64 0xFFFFFFFFFFFFFFFFULL
static inline uint64_t M(int64_t v) { return (uint64_t)v; }
static inline int64_t sx(uint64_t v) { return (int64_t)v; }

/* ── Memory access ───────────────────────────────────────────── */
static uint64_t mem_read(uint64_t addr, int width) {
    uint64_t a = addr & MASK64;
    if (a + (width/8) > (uint64_t)MEM_SIZE) return 0;
    switch(width) {
    case 8:  return mem[a];
    case 16: { uint16_t v; memcpy(&v,mem+a,2); return v; }
    case 32: { uint32_t v; memcpy(&v,mem+a,4); return v; }
    case 64: { uint64_t v; memcpy(&v,mem+a,8); return v; }
    }
    return 0;
}
static void mem_write(uint64_t addr, int width, uint64_t val) {
    uint64_t a = addr & MASK64;
    if (a + (width/8) > (uint64_t)MEM_SIZE) return;
    switch(width) {
    case 8:  mem[a] = val&0xFF; break;
    case 16: { uint16_t v=val; memcpy(mem+a,&v,2); break; }
    case 32: { uint32_t v=val; memcpy(mem+a,&v,4); break; }
    case 64: { uint64_t v=val; memcpy(mem+a,&v,8); break; }
    }
}

/* ── Symbol tables ───────────────────────────────────────────── */
#define MAX_COMPOS 512
typedef struct { char name[64]; char params[16][32]; int nparams; int body_tk; int body_indent; } Compo;
static Compo compos[MAX_COMPOS];
static int ncompos;

#define MAX_CONSTS 512
static struct { char name[64]; int64_t val; } consts[MAX_CONSTS];
static int nconsts;

#define MAX_BUFS 512
static struct { char name[64]; int64_t off; int64_t size; } bufs[MAX_BUFS];
static int nbufs;

#define MAX_GLOBALS 1024
static struct { char name[64]; int64_t val; } globals[MAX_GLOBALS];
static int nglobals;

/* Frames */
#define MAX_FRAMES 2048
#define MAX_LOCALS 128
#define LOCAL_HT_SIZE 256
#define LOCAL_HT_MASK (LOCAL_HT_SIZE-1)
typedef struct {
    char names[MAX_LOCALS][64];
    int64_t vals[MAX_LOCALS];
    int n; int nparams;
    int ht[LOCAL_HT_SIZE]; /* hash -> local index, -1 = empty */
} Frame;
static Frame frames[MAX_FRAMES];
static int nframes;

/* ── Helpers ─────────────────────────────────────────────────── */
static void tok_text(int idx, char *buf, int bufsz) {
    int n = toks[idx].len;
    if (n >= bufsz) n = bufsz-1;
    memcpy(buf, src+toks[idx].off, n);
    buf[n] = 0;
}

static int ty(void) { return toks[tk].type; }

static int64_t parse_int_tok(int idx) {
    char buf[64]; tok_text(idx, buf, sizeof buf);
    int neg = 0; char *p = buf;
    if (*p=='-') { neg=1; p++; }
    int64_t v;
    if (p[0]=='0' && (p[1]=='x'||p[1]=='X')) v = (int64_t)strtoull(p+2,NULL,16);
    else v = (int64_t)strtoull(p,NULL,10);
    return neg ? M(-v) : M(v);
}

static int dollar_reg(int idx) {
    if (toks[idx].len >= 2 && src[toks[idx].off] == '$') {
        char buf[16]; tok_text(idx, buf, sizeof buf);
        return atoi(buf+1);
    }
    return -1;
}

/* ── Hash table for fast name lookup ──────────────────────────── */
#define HASH_SIZE 4096
#define HASH_MASK (HASH_SIZE-1)
static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

/* Per-table: hash -> index, with linear probing. -1 = empty. */
static int ht_compo[HASH_SIZE], ht_const[HASH_SIZE], ht_buf[HASH_SIZE], ht_global[HASH_SIZE];

static void ht_init(int *ht) { memset(ht, 0xFF, HASH_SIZE*sizeof(int)); }

static void ht_insert(int *ht, const char *name, int idx) {
    uint32_t h = fnv1a(name) & HASH_MASK;
    while (ht[h] >= 0) h = (h+1) & HASH_MASK;
    ht[h] = idx;
}

/* ── Lookup ──────────────────────────────────────────────────── */
static int find_compo(const char *name) {
    uint32_t h = fnv1a(name) & HASH_MASK;
    while (ht_compo[h] >= 0) {
        if (!strcmp(compos[ht_compo[h]].name, name)) return ht_compo[h];
        h = (h+1) & HASH_MASK;
    }
    return -1;
}
static int find_const(const char *name) {
    uint32_t h = fnv1a(name) & HASH_MASK;
    while (ht_const[h] >= 0) {
        if (!strcmp(consts[ht_const[h]].name, name)) return ht_const[h];
        h = (h+1) & HASH_MASK;
    }
    return -1;
}
static int find_buf(const char *name) {
    uint32_t h = fnv1a(name) & HASH_MASK;
    while (ht_buf[h] >= 0) {
        if (!strcmp(bufs[ht_buf[h]].name, name)) return ht_buf[h];
        h = (h+1) & HASH_MASK;
    }
    return -1;
}
static int find_global(const char *name) {
    uint32_t h = fnv1a(name) & HASH_MASK;
    while (ht_global[h] >= 0) {
        if (!strcmp(globals[ht_global[h]].name, name)) return ht_global[h];
        h = (h+1) & HASH_MASK;
    }
    return -1;
}
static int find_local(const char *name) {
    if (nframes==0) return -1;
    Frame *f = &frames[nframes-1];
    uint32_t h = fnv1a(name) & LOCAL_HT_MASK;
    while (f->ht[h] >= 0) {
        if (!strcmp(f->names[f->ht[h]], name)) return f->ht[h];
        h = (h+1) & LOCAL_HT_MASK;
    }
    return -1;
}

static int64_t lookup(const char *name) {
    int i;
    if ((i=find_local(name))>=0) return frames[nframes-1].vals[i];
    if ((i=find_const(name))>=0) return consts[i].val;
    if ((i=find_global(name))>=0) return globals[i].val;
    if ((i=find_buf(name))>=0) return bufs[i].off;
    return 0;
}

static void store_var(const char *name, int64_t val) {
    int i;
    uint64_t v = M(val);
    if ((i=find_local(name))>=0) { frames[nframes-1].vals[i]=v; return; }
    if ((i=find_global(name))>=0) { globals[i].val=v; return; }
}

static void new_local(const char *name, int64_t val) {
    Frame *f = &frames[nframes-1];
    if (f->n < MAX_LOCALS) {
        int idx = f->n;
        strncpy(f->names[idx], name, 63); f->names[idx][63]=0;
        f->vals[idx] = M(val);
        /* insert into frame hash */
        uint32_t h = fnv1a(name) & LOCAL_HT_MASK;
        while (f->ht[h] >= 0) h = (h+1) & LOCAL_HT_MASK;
        f->ht[h] = idx;
        f->n++;
    }
}

static void set_const(const char *name, int64_t val) {
    int i = find_const(name);
    if (i>=0) { consts[i].val=M(val); return; }
    if (nconsts < MAX_CONSTS) {
        strncpy(consts[nconsts].name,name,63); consts[nconsts].val=M(val);
        ht_insert(ht_const, name, nconsts); nconsts++;
    }
}
static void set_buf(const char *name, int64_t size) {
    if (nbufs < MAX_BUFS) {
        strncpy(bufs[nbufs].name,name,63);
        int64_t off = mem_top;
        mem_top = (mem_top + size + 7) & ~7;
        bufs[nbufs].off = off; bufs[nbufs].size = size;
        ht_insert(ht_buf, name, nbufs); nbufs++;
    }
}
static void set_global(const char *name, int64_t val) {
    int i = find_global(name);
    if (i>=0) { globals[i].val = M(val); return; }
    if (nglobals < MAX_GLOBALS) {
        strncpy(globals[nglobals].name,name,63); globals[nglobals].val=M(val);
        ht_insert(ht_global, name, nglobals); nglobals++;
    }
}

/* ── Control flow via longjmp ─────────────────────────────────── */
#include <setjmp.h>

static jmp_buf jmp_return;
static int64_t return_val;
static jmp_buf jmp_exit;
static int exit_code;
static jmp_buf jmp_goto;
static char goto_label[64];
static jmp_buf jmp_break;
static jmp_buf jmp_continue;

/* ── Forward declarations ─────────────────────────────────────── */
static int64_t parse_expr(void);
static int64_t parse_primary(void);
static int64_t parse_shift(void);
static int64_t exec_stmt(void);
static int64_t exec_loop(int body_indent);
static void skip_body(int body_indent);
static int64_t call_compo(const char *name, int atom_mode);

/* ── File descriptors ─────────────────────────────────────────── */
#define MAX_FDS 256
static int open_fds[MAX_FDS]; /* fake_fd -> real_fd, 0=unused */
static int next_fake_fd = 100;

static int resolve_fd(int64_t fd) {
    int fv = (int)sx(fd);
    if (fv < 0 || fv > 1000000) return -1;
    if (fv < 100) return fv;
    if (fv < MAX_FDS+100 && open_fds[fv-100]) return open_fds[fv-100];
    return -1;
}

/* ── Expression parser ───────────────────────────────────────── */
static void skip_empty_lines(void) {
    while (1) {
        if (ty()==TOK_NEWLINE) { tk++; continue; }
        if (ty()==TOK_INDENT && tk+1<ntoks && toks[tk+1].type==TOK_NEWLINE) { tk+=2; continue; }
        break;
    }
}

static void maybe_eat_cont(int allow_plus_minus) {
    /* If at NEWLINE, peek past it + INDENT for a continuation operator */
    if (ty() != TOK_NEWLINE) return;
    int i = tk+1;
    while (i<ntoks && toks[i].type==TOK_NEWLINE) i++;
    if (i>=ntoks || toks[i].type!=TOK_INDENT) return;
    if (toks[i].len<=0) return;
    int j = i+1;
    if (j>=ntoks) return;
    int nt = toks[j].type;
    if (is_op(nt)) {
        if (!allow_plus_minus && (nt==TOK_PLUS||nt==TOK_MINUS)) return;
        tk = i+1;
    }
}

static int64_t parse_primary(void) {
    int t = ty();

    if (t == TOK_INT) { int idx=tk; tk++; return parse_int_tok(idx); }

    if (t == TOK_LPAREN) { tk++; int64_t v=parse_expr(); if(ty()==TOK_RPAREN)tk++; return v; }

    if (t == TOK_MINUS) { tk++; return M(-sx(parse_primary())); }

    if (t == TOK_IDENT) {
        char name[64]; tok_text(tk, name, sizeof name);
        /* $N */
        if (name[0]=='$' && toks[tk].len>=2) {
            int n = dollar_reg(tk); tk++;
            return (n>=0&&n<32) ? regs[n] : 0;
        }
        tk++;
        /* subscript */
        if (ty()==TOK_LBRACK) {
            tk++;
            int64_t idx = parse_expr();
            if (ty()==TOK_RBRACK) tk++;
            int64_t base = lookup(name);
            return mem_read(M(base+idx), 8);
        }
        /* locals/globals/consts/bufs shadow compositions */
        int li;
        if ((li=find_local(name))>=0) return frames[nframes-1].vals[li];
        if ((li=find_const(name))>=0) return consts[li].val;
        if ((li=find_global(name))>=0) return globals[li].val;
        if ((li=find_buf(name))>=0) return bufs[li].off;
        /* composition call */
        if (find_compo(name)>=0) return call_compo(name, 1);
        return 0;
    }

    if (t == TOK_REG_READ) {
        tk++;
        if (ty()==TOK_IDENT) { int n=dollar_reg(tk); tk++; return (n>=0&&n<32)?regs[n]:0; }
        return 0;
    }

    if (t == TOK_MEM_LOAD) {
        tk++;
        if (ty()!=TOK_INT) return 0;
        int width = (int)parse_int_tok(tk); tk++;
        int64_t base = parse_shift();
        int64_t addr = base;
        int nt = ty();
        if (nt==TOK_INT || nt==TOK_IDENT) {
            int64_t off = parse_primary();
            addr = M(base + off);
        }
        return mem_read(addr, width);
    }

    if (t == TOK_BUF || t == TOK_CONST || t == TOK_VAR || t == TOK_LABEL) {
        /* treat as ident */
        char name[64]; tok_text(tk, name, sizeof name);
        tk++;
        int li;
        if ((li=find_local(name))>=0) return frames[nframes-1].vals[li];
        if ((li=find_const(name))>=0) return consts[li].val;
        if ((li=find_global(name))>=0) return globals[li].val;
        if ((li=find_buf(name))>=0) return bufs[li].off;
        if (find_compo(name)>=0) return call_compo(name, 1);
        return 0;
    }

    return 0;
}

static int64_t parse_mul(void) {
    int64_t a = parse_primary();
    while (1) {
        maybe_eat_cont(0);
        int op = ty();
        if (op!=TOK_STAR && op!=TOK_SLASH) break;
        tk++;
        int64_t b = parse_primary();
        if (op==TOK_STAR) a = M(sx(a)*sx(b));
        else { int64_t sb=sx(b); a = sb==0 ? 0 : M(sx(a)/sb); }
    }
    return a;
}

static int64_t parse_add(void) {
    int64_t a = parse_mul();
    while (1) {
        maybe_eat_cont(1);
        int op = ty();
        if (op!=TOK_PLUS && op!=TOK_MINUS) break;
        tk++;
        int64_t b = parse_mul();
        a = op==TOK_PLUS ? M(a+b) : M(a-b);
    }
    return a;
}

static int64_t parse_shift(void) {
    int64_t a = parse_add();
    while (1) {
        maybe_eat_cont(0);
        int op = ty();
        if (op!=TOK_SHL && op!=TOK_SHR) break;
        tk++;
        int64_t b = parse_add();
        int sh = b & 63;
        a = op==TOK_SHL ? M((uint64_t)a << sh) : (int64_t)((uint64_t)a >> sh);
    }
    return a;
}

static int64_t parse_cmp(void) {
    int64_t a = parse_shift();
    int op = ty();
    if (!is_cmp(op)) return a;
    tk++;
    int64_t b = parse_shift();
    int64_t sa=sx(a), sb=sx(b);
    switch(op) {
    case TOK_LT: return sa<sb; case TOK_GT: return sa>sb;
    case TOK_LTE: return sa<=sb; case TOK_GTE: return sa>=sb;
    case TOK_EQEQ: return a==b; case TOK_NEQ: return a!=b;
    }
    return 0;
}

static int64_t parse_bits(void) {
    int64_t a = parse_cmp();
    while (1) {
        maybe_eat_cont(0);
        int op = ty();
        if (op!=TOK_AMP && op!=TOK_PIPE && op!=TOK_CARET) break;
        tk++;
        int64_t b = parse_cmp();
        if (op==TOK_AMP) a=a&b; else if (op==TOK_PIPE) a=a|b; else a=a^b;
    }
    return a;
}

static int64_t parse_expr(void) { return parse_bits(); }

/* ── Condition evaluation ─────────────────────────────────────── */
static int eval_if_cond(void) {
    int ct = ty();
    if (is_cmp(ct)) {
        tk++;
        int64_t a = parse_expr(), b = parse_expr();
        int64_t sa=sx(a),sb=sx(b);
        switch(ct) {
        case TOK_LT: return sa<sb; case TOK_GT: return sa>sb;
        case TOK_LTE: return sa<=sb; case TOK_GTE: return sa>=sb;
        case TOK_EQEQ: return a==b; case TOK_NEQ: return a!=b;
        }
    }
    int64_t a = parse_expr();
    int rel = ty();
    if (is_cmp(rel)) {
        tk++;
        int64_t b = parse_expr();
        int64_t sa=sx(a),sb=sx(b);
        switch(rel) {
        case TOK_LT: return sa<sb; case TOK_GT: return sa>sb;
        case TOK_LTE: return sa<=sb; case TOK_GTE: return sa>=sb;
        case TOK_EQEQ: return a==b; case TOK_NEQ: return a!=b;
        }
    }
    return a != 0;
}

/* ── Op chain (for reassignment: `name OP expr OP expr ...`) ── */
static int64_t eval_op_chain(int64_t start) {
    int64_t a = M(start);
    while (1) {
        maybe_eat_cont(1);
        int op = ty();
        if (!is_op(op)) break;
        tk++;
        int64_t b = parse_primary();
        switch(op) {
        case TOK_PLUS: a=M(a+b);break; case TOK_MINUS: a=M(a-b);break;
        case TOK_STAR: a=M(sx(a)*sx(b));break;
        case TOK_SLASH: { int64_t sb=sx(b); a=sb?M(sx(a)/sb):0; break; }
        case TOK_AMP: a=a&b;break; case TOK_PIPE: a=a|b;break; case TOK_CARET: a=a^b;break;
        case TOK_SHL: a=M((uint64_t)a<<(b&63));break; case TOK_SHR: a=(uint64_t)a>>(b&63);break;
        }
    }
    return a;
}

/* ── Skip body ───────────────────────────────────────────────── */
static void skip_body(int body_indent) {
    while (ty()!=TOK_EOF) {
        if (ty()==TOK_NEWLINE) { tk++; continue; }
        if (ty()==TOK_INDENT) {
            if (toks[tk].len < body_indent) return;
            tk++;
            while (ty()!=TOK_NEWLINE && ty()!=TOK_EOF) tk++;
        } else return;
    }
}

/* ── Find label ──────────────────────────────────────────────── */
static int find_label(int start_tk, int body_indent, const char *name) {
    int i = start_tk;
    while (i < ntoks) {
        if (toks[i].type==TOK_EOF) return -1;
        if (toks[i].type==TOK_INDENT) {
            if (toks[i].len < body_indent) return -1;
            if (toks[i].len == body_indent && i+1<ntoks) {
                if (toks[i+1].type==TOK_LABEL && i+2<ntoks) {
                    char nm[64]; tok_text(i+2,nm,sizeof nm);
                    if (!strcmp(nm,name)) return i+3;
                }
                if (toks[i+1].type==TOK_IDENT && i+2<ntoks && toks[i+2].type==TOK_COLON) {
                    char nm[64]; tok_text(i+1,nm,sizeof nm);
                    if (!strcmp(nm,name)) return i+3;
                }
            }
        }
        i++;
    }
    return -1;
}

/* ── Syscall ─────────────────────────────────────────────────── */
static int64_t do_syscall(void) {
    /* Prefer X8 (Linux convention) when set; fall back to X16 (Darwin). */
    int64_t num = regs[8] ? regs[8] : regs[16];
    regs[8] = 0; regs[16] = 0; /* clear after read */
    int64_t x0=regs[0],x1=regs[1],x2=regs[2],x3=regs[3],x4=regs[4],x5=regs[5];

    /* exit */
    if (num==1 || num==93) { exit_code=x0&0xFF; longjmp(jmp_exit,1); }

    /* write */
    if (num==4 || num==64) {
        int fd=resolve_fd(x0); if(fd<0)fd=x0;
        if (x2>0 && (uint64_t)x1<MEM_SIZE) {
            int n=write(fd, mem+x1, x2);
            regs[0]=n; return n;
        }
        regs[0]=0; return 0;
    }
    /* read */
    if (num==3 || num==63) {
        int fd=resolve_fd(x0); if(fd<0){regs[0]=-1;return -1;}
        char tmp[65536]; int want=x2>65536?65536:x2;
        int n=read(fd,tmp,want);
        if(n>0) memcpy(mem+x1,tmp,n);
        regs[0]=n; return n;
    }
    /* close */
    if (num==6 || num==57) {
        int fd=resolve_fd(x0); if(fd<0){regs[0]=-1;return -1;}
        close(fd);
        int fv=sx(x0); if(fv>=100&&fv<MAX_FDS+100) open_fds[fv-100]=0;
        regs[0]=0; return 0;
    }
    /* lseek */
    if (num==199 || num==62) {
        int fd=resolve_fd(x0); if(fd<0){regs[0]=-1;return -1;}
        off_t r=lseek(fd,sx(x1),x2);
        regs[0]=r; return r;
    }
    /* mmap */
    if (num==197 || num==222) {
        int64_t length=x1;
        int fd=resolve_fd(x4); if(fd<0){regs[0]=0;return 0;}
        lseek(fd,x5,SEEK_SET);
        int64_t base=mem_top;
        int64_t alloc=(length+7)&~7; if(alloc<8)alloc=8;
        if(base+alloc>(int64_t)MEM_SIZE){regs[0]=0;return 0;}
        int n=read(fd,mem+base,length);
        (void)n;
        mem_top+=alloc;
        regs[0]=base; regs[1]=length;
        return base;
    }
    /* openat */
    if (num==463 || num==56) {
        char path[1024]; int pi=0;
        uint64_t p=x1;
        while(p<MEM_SIZE && mem[p]!=0 && pi<1023) path[pi++]=mem[p++];
        path[pi]=0;
        int flags_v = x2 & 0xFFFFFFFF;
        int py_flags;
        if (num==56) {
            /* Linux flags */
            py_flags = flags_v & 3; /* O_RDONLY=0, O_WRONLY=1, O_RDWR=2 */
            if (flags_v & 0x40) py_flags |= O_CREAT;
            if (flags_v & 0x200) py_flags |= O_TRUNC;
        } else {
            /* Darwin flags */
            py_flags = flags_v & 3;
            if (flags_v & 0x200) py_flags |= O_CREAT;
            if (flags_v & 0x400) py_flags |= O_TRUNC;
        }
        int mode = x3 ? (x3 & 07777) : 0644;
        int rfd = open(path, py_flags, mode);
        if (rfd<0) { regs[0]=-1; return -1; }
        int fake = next_fake_fd++;
        if (fake-100 < MAX_FDS) open_fds[fake-100] = rfd;
        regs[0]=fake; return fake;
    }
    /* munmap, mprotect, brk */
    if (num==73||num==74||num==215||num==226) { regs[0]=0; return 0; }
    if (num==12||num==214) { regs[0]=mem_top; return mem_top; }

    regs[0]=0; return 0;
}

/* ── exec_loop ───────────────────────────────────────────────── */
static int64_t exec_loop(int body_indent) {
    int64_t last = 0;
    int loop_start = tk;

    /* Save outer goto context — nested calls must not clobber it */
    jmp_buf saved_goto;
    memcpy(saved_goto, jmp_goto, sizeof(jmp_buf));

    while (1) {
        if (ty()==TOK_EOF) break;
        if (ty()==TOK_NEWLINE) { tk++; continue; }
        if (ty()==TOK_INDENT) {
            if (tk+1<ntoks && toks[tk+1].type==TOK_NEWLINE) { tk+=2; continue; }
            if (toks[tk].len < body_indent) break;
            /* continuation line */
            int nxt = (tk+1<ntoks) ? toks[tk+1].type : TOK_EOF;
            if (is_op(nxt) && nxt!=TOK_PLUS && nxt!=TOK_MINUS) {
                tk++;
                last = eval_op_chain(last);
                continue;
            }
            tk++;
        } else break;

        /* goto handling via setjmp */
        if (setjmp(jmp_goto) != 0) {
            int target = find_label(loop_start, body_indent, goto_label);
            if (target < 0) {
                /* propagate to outer loop */
                memcpy(jmp_goto, saved_goto, sizeof(jmp_buf));
                longjmp(jmp_goto, 1);
            }
            tk = target;
            continue;
        }
        last = exec_stmt();
    }
    memcpy(jmp_goto, saved_goto, sizeof(jmp_buf));
    return last;
}

/* ── exec_if_chain ───────────────────────────────────────────── */
static int64_t exec_if_chain(void) {
    int took = 0;
    while (1) {
        int kw = ty();
        if (kw==TOK_IF || kw==TOK_ELIF) {
            tk++;
            int cond = eval_if_cond();
            if (ty()==TOK_COLON) tk++;
            skip_empty_lines();
            int bi = (ty()==TOK_INDENT) ? toks[tk].len : 0;
            if (cond && !took) { exec_loop(bi); took=1; }
            else skip_body(bi);
        } else if (kw==TOK_ELSE) {
            tk++;
            if (ty()==TOK_COLON) tk++;
            skip_empty_lines();
            int bi = (ty()==TOK_INDENT) ? toks[tk].len : 0;
            if (!took) { exec_loop(bi); took=1; }
            else skip_body(bi);
            break;
        } else break;
        skip_empty_lines();
        if (ty()==TOK_INDENT && tk+1<ntoks &&
            (toks[tk+1].type==TOK_ELIF || toks[tk+1].type==TOK_ELSE)) tk++;
        else if (ty()==TOK_ELIF || ty()==TOK_ELSE) {}
        else break;
    }
    return 0;
}

/* ── exec_for ────────────────────────────────────────────────── */
static int64_t exec_for(void) {
    tk++;
    if (ty()!=TOK_IDENT) return 0;
    char var[64]; tok_text(tk,var,sizeof var); tk++;
    int64_t start=parse_expr(), end=parse_expr(), step=1;
    if (is_expr_start(ty())) step=parse_expr();
    if (ty()==TOK_COLON) tk++;
    skip_empty_lines();
    int bi = (ty()==TOK_INDENT) ? toks[tk].len : 0;
    int body_tk = tk;
    new_local(var, start);
    int64_t i=sx(start), e=sx(end), s=sx(step);
    if (s==0) s=1;
    while ((s>0 && i<e) || (s<0 && i>e)) {
        store_var(var, M(i));
        tk = body_tk;
        exec_loop(bi);
        i += s;
    }
    tk = body_tk;
    skip_body(bi);
    return 0;
}

/* ── call_compo ──────────────────────────────────────────────── */
static int64_t last_call_extras[MAX_LOCALS];
static int n_extras;

static int64_t call_compo(const char *name, int atom_mode) {
    int ci = find_compo(name);
    if (ci < 0) return 0;
    Compo *c = &compos[ci];
    int64_t args[16]; int nargs = 0;

    while (nargs < c->nparams) {
        int t = ty();
        if (t==TOK_NEWLINE||t==TOK_EOF||t==TOK_INDENT) break;
        if (!is_expr_start(t)) break;
        if (atom_mode && (is_op(t)||is_cmp(t)||t==TOK_RPAREN)) break;
        args[nargs++] = atom_mode ? parse_primary() : parse_expr();
    }
    /* top-level: pull from regs */
    if (nframes==0) { while(nargs<c->nparams && nargs<32) { args[nargs]=regs[nargs]; nargs++; } }
    while (nargs<c->nparams) args[nargs++]=0;

    /* push frame */
    if (nframes >= MAX_FRAMES) { fprintf(stderr,"frame overflow\n"); exit(1); }
    Frame *f = &frames[nframes];
    f->n = 0; f->nparams = c->nparams;
    memset(f->ht, 0xFF, sizeof(f->ht));
    for (int i=0;i<c->nparams;i++) {
        strncpy(f->names[i], c->params[i], 63); f->names[i][63]=0;
        f->vals[i] = args[i];
        uint32_t h = fnv1a(c->params[i]) & LOCAL_HT_MASK;
        while (f->ht[h] >= 0) h = (h+1) & LOCAL_HT_MASK;
        f->ht[h] = i;
    }
    f->n = c->nparams;
    nframes++;

    int saved_tk = tk;
    tk = c->body_tk;

    int64_t ret = 0;
    jmp_buf saved_return;
    memcpy(saved_return, jmp_return, sizeof(jmp_buf));
    if (setjmp(jmp_return) == 0) {
        ret = exec_loop(c->body_indent);
    } else {
        ret = return_val;
    }
    memcpy(jmp_return, saved_return, sizeof(jmp_buf));

    /* extras */
    Frame *ff = &frames[nframes-1];
    n_extras = 0;
    for (int i=ff->n-1; i>=ff->nparams; i--) {
        if (n_extras < MAX_LOCALS) last_call_extras[n_extras++] = ff->vals[i];
    }

    nframes--;
    tk = saved_tk;
    return ret;
}

/* ── exec_ident_stmt ─────────────────────────────────────────── */
static int64_t exec_ident_stmt(void) {
    char name[64]; tok_text(tk, name, sizeof name); tk++;

    /* locals shadow compositions */
    int li = find_local(name);
    if (li >= 0) {
        if (ty()==TOK_EQ) tk++;
        int ct = ty();
        if (is_op(ct)) {
            int64_t v = eval_op_chain(frames[nframes-1].vals[li]);
            frames[nframes-1].vals[li] = M(v); regs[0]=M(v); return v;
        }
        if (ct==TOK_NEWLINE||ct==TOK_EOF||ct==TOK_INDENT) {
            int64_t v = frames[nframes-1].vals[li]; regs[0]=v; return v;
        }
        int64_t v = parse_expr();
        frames[nframes-1].vals[li] = M(v); regs[0]=M(v); return v;
    }
    /* composition call */
    if (find_compo(name)>=0) { int64_t r=call_compo(name,0); regs[0]=r; return r; }
    /* global reassign */
    int gi = find_global(name);
    if (gi>=0) {
        if (ty()==TOK_EQ) tk++;
        int ct=ty();
        if (is_op(ct)) { int64_t v=eval_op_chain(globals[gi].val); globals[gi].val=M(v); regs[0]=M(v); return v; }
        if (ct==TOK_NEWLINE||ct==TOK_EOF||ct==TOK_INDENT) { int64_t v=globals[gi].val; regs[0]=v; return v; }
        int64_t v=parse_expr(); globals[gi].val=M(v); regs[0]=M(v); return v;
    }
    /* const */
    int ci2=find_const(name);
    if (ci2>=0) { regs[0]=consts[ci2].val; return consts[ci2].val; }
    /* buf */
    int bi=find_buf(name);
    if (bi>=0) { regs[0]=bufs[bi].off; return bufs[bi].off; }

    /* new binding */
    if (ty()==TOK_EQ) tk++;
    if (ty()==TOK_NEWLINE||ty()==TOK_EOF||ty()==TOK_INDENT) return 0;

    /* multi-assign: collect unknown idents until we hit a known composition */
    char extra_names[16][64]; int nextra=0;
    while (ty()==TOK_IDENT) {
        char nm[64]; tok_text(tk,nm,sizeof nm);
        if (find_compo(nm)>=0) break;
        if (find_local(nm)>=0||find_global(nm)>=0||find_const(nm)>=0||find_buf(nm)>=0) break;
        int nxt = (tk+1<ntoks) ? toks[tk+1].type : TOK_EOF;
        if (is_op(nxt)||is_cmp(nxt)||nxt==TOK_LBRACK||nxt==TOK_NEWLINE||nxt==TOK_EOF||nxt==TOK_INDENT) break;
        if (nextra<16) strncpy(extra_names[nextra++],nm,63);
        tk++;
    }
    /* composition call for value? */
    if (ty()==TOK_IDENT) {
        char nm[64]; tok_text(tk,nm,sizeof nm);
        if (find_compo(nm)>=0) {
            tk++;
            int64_t val = call_compo(nm,0);
            if (nframes) new_local(name,val); else set_global(name,val);
            for (int i=0;i<nextra;i++) {
                int64_t ev = (i+1<n_extras) ? last_call_extras[i+1] : 0;
                if (nframes) new_local(extra_names[i],ev); else set_global(extra_names[i],ev);
            }
            return val;
        }
    }
    int64_t val = parse_expr();
    if (nframes) { new_local(name,val); for(int i=0;i<nextra;i++) new_local(extra_names[i],val); }
    else { set_global(name,val); for(int i=0;i<nextra;i++) set_global(extra_names[i],val); }
    return val;
}

/* ── exec_stmt ───────────────────────────────────────────────── */
static int64_t exec_stmt(void) {
    int t = ty();

    if (t==TOK_TRAP) {
        tk++;
        if (ty()==TOK_NEWLINE||ty()==TOK_EOF||ty()==TOK_INDENT) return do_syscall();
        if (ty()==TOK_IDENT) {
            char nm[64]; tok_text(tk,nm,sizeof nm); tk++;
            int64_t num=parse_primary(); regs[16]=num;
            int na=0;
            while(na<8 && is_expr_start(ty()) && ty()!=TOK_NEWLINE && ty()!=TOK_EOF) { regs[na++]=parse_expr(); }
            int64_t r=do_syscall();
            if(nframes) new_local(nm,r); else set_global(nm,r);
            return r;
        }
        return 0;
    }
    if (t==TOK_CONST) {
        tk++;
        char nm[64]; tok_text(tk,nm,sizeof nm); tk++;
        if(ty()!=TOK_INT) return 0;
        set_const(nm, parse_primary());
        return 0;
    }
    if (t==TOK_BUF) {
        tk++;
        char nm[64]; tok_text(tk,nm,sizeof nm); tk++;
        if(ty()!=TOK_INT) return 0;
        set_buf(nm, parse_primary());
        return 0;
    }
    if (t==TOK_VAR) {
        tk++;
        char nm[64]; tok_text(tk,nm,sizeof nm); tk++;
        int64_t val = is_expr_start(ty()) ? parse_expr() : 0;
        if(nframes) new_local(nm,val); else set_global(nm,val);
        return val;
    }
    if (t==TOK_RETURN) {
        tk++;
        int64_t v = 0;
        if (is_expr_start(ty())) v = parse_expr();
        else if (nframes) {
            Frame *f=&frames[nframes-1];
            if (f->n > f->nparams) v = f->vals[f->nparams];
            else if (f->n > 0) v = f->vals[f->n-1];
        }
        return_val = v;
        longjmp(jmp_return, 1);
    }
    if (t==TOK_LOAD) { /* ↓ $N val */
        tk++;
        if (ty()==TOK_IDENT) {
            int n=dollar_reg(tk);
            if(n>=0) { tk++; int64_t v=parse_expr(); regs[n]=M(v); return v; }
        }
        int64_t width=parse_primary(), addr=parse_expr(), val=parse_expr();
        mem_write(addr,width,val);
        return val;
    }
    if (t==TOK_MEM_STORE) { /* ← W addr val */
        tk++;
        if(ty()!=TOK_INT) return 0;
        int width=(int)parse_int_tok(tk); tk++;
        int64_t base=parse_expr(), middle=parse_expr();
        if (is_expr_start(ty())) {
            int64_t addr=M(base+middle), val=parse_expr();
            mem_write(addr,width,val); return val;
        }
        mem_write(base,width,middle); return middle;
    }
    if (t==TOK_REG_READ) { int64_t v=parse_expr(); regs[0]=M(v); return v; }
    if (t==TOK_MEM_LOAD) { int64_t v=parse_expr(); regs[0]=M(v); return v; }

    if (t==TOK_IF) return exec_if_chain();
    if (t==TOK_WHILE) {
        tk++; int cond_tk=tk;
        int bi=0; int first=1;
        while(1) {
            tk=cond_tk;
            int c=eval_if_cond();
            if(ty()==TOK_COLON) tk++;
            skip_empty_lines();
            if(first) { bi=(ty()==TOK_INDENT)?toks[tk].len:0; first=0; }
            if(!c) { skip_body(bi); return 0; }
            exec_loop(bi);
        }
    }
    if (t==TOK_FOR) return exec_for();
    if (t==TOK_EACH) {
        tk++;
        if(ty()==TOK_IDENT) { char nm[64]; tok_text(tk,nm,sizeof nm); new_local(nm,0); tk++; }
        if(ty()==TOK_COLON)tk++;
        skip_empty_lines();
        int bi=(ty()==TOK_INDENT)?toks[tk].len:0;
        exec_loop(bi);
        return 0;
    }
    if (t==TOK_CONTINUE) { tk++; longjmp(jmp_continue,1); }
    if (t==TOK_BREAK) { tk++; longjmp(jmp_break,1); }
    if (t==TOK_GOTO) {
        tk++;
        if(ty()==TOK_IDENT) { tok_text(tk,goto_label,sizeof goto_label); tk++; longjmp(jmp_goto,1); }
        return 0;
    }
    if (t==TOK_LABEL) { tk++; if(ty()==TOK_IDENT) tk++; return 0; }

    /* name: label */
    if (t==TOK_IDENT && tk+1<ntoks && toks[tk+1].type==TOK_COLON) { tk+=2; return 0; }

    /* ident stmt */
    if (t==TOK_IDENT || t==TOK_BUF || t==TOK_CONST || t==TOK_VAR || t==TOK_LABEL)
        return exec_ident_stmt();

    /* bare expression */
    if (is_expr_start(t)) { int64_t v=parse_expr(); regs[0]=M(v); return v; }

    /* skip unknown */
    while(ty()!=TOK_NEWLINE && ty()!=TOK_EOF) tk++;
    return 0;
}

/* ── Collect compositions ────────────────────────────────────── */
static void collect(void) {
    int i=0, cur_indent=0, at_line_start=1;
    ncompos=0;
    ht_init(ht_compo);
    while (i<ntoks) {
        if (toks[i].type==TOK_EOF) break;
        if (toks[i].type==TOK_NEWLINE) { at_line_start=1; cur_indent=0; i++; continue; }
        if (toks[i].type==TOK_INDENT) { cur_indent=toks[i].len; at_line_start=1; i++; continue; }
        if (!at_line_start) { i++; continue; }
        at_line_start=0;
        if (cur_indent>0) { i++; continue; }
        if (toks[i].type!=TOK_IDENT) { i++; continue; }

        int name_idx = i;
        char nm[64]; tok_text(name_idx,nm,sizeof nm);
        if (!strcmp(nm,"host")||!strcmp(nm,"kernel")) {
            name_idx++;
            if (name_idx>=ntoks || toks[name_idx].type!=TOK_IDENT) { i++; continue; }
        }

        /* collect params until colon */
        int j=name_idx;
        char params[16][32]; int np=0; int saw_colon=0;
        while (j<ntoks) {
            if (toks[j].type==TOK_NEWLINE||toks[j].type==TOK_EOF) break;
            if (toks[j].type==TOK_COLON) { saw_colon=1; break; }
            if (j>name_idx && (toks[j].type==TOK_IDENT||toks[j].type==TOK_BUF||
                toks[j].type==TOK_CONST||toks[j].type==TOK_VAR||toks[j].type==TOK_LABEL)) {
                if (np<16) { tok_text(j,params[np],32); np++; }
            }
            j++;
        }
        if (!saw_colon) { i++; continue; }

        int k = j+1;
        while (k<ntoks && toks[k].type!=TOK_NEWLINE) k++;
        int body_tk = k+1;

        /* body indent */
        int body_indent=0, bk=body_tk;
        while (bk<ntoks) {
            if (toks[bk].type==TOK_NEWLINE) { bk++; continue; }
            if (toks[bk].type==TOK_INDENT) { body_indent=toks[bk].len; break; }
            break;
        }

        if (ncompos<MAX_COMPOS) {
            Compo *c = &compos[ncompos];
            tok_text(name_idx, c->name, sizeof c->name);
            c->nparams = np;
            for (int p=0;p<np;p++) strncpy(c->params[p],params[p],31);
            c->body_tk = body_tk;
            c->body_indent = body_indent;
            ht_insert(ht_compo, c->name, ncompos);
            ncompos++;
        }
        i = body_tk;
    }
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,"usage: lithos-interp source.ls [args...]\n"); return 1; }

    /* Read source */
    int fd = open(argv[1], O_RDONLY);
    if (fd<0) { perror(argv[1]); return 1; }
    struct stat st; fstat(fd, &st);
    srclen = st.st_size;
    src = malloc(srclen+1);
    read(fd, src, srclen); src[srclen]=0;
    close(fd);

    /* Init hash tables */
    ht_init(ht_compo); ht_init(ht_const); ht_init(ht_buf); ht_init(ht_global);

    /* Allocate memory */
    mem = mmap(NULL, MEM_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mem==MAP_FAILED) { perror("mmap"); return 1; }

    /* Lex */
    do_lex();

    /* Collect compositions */
    collect();

    /* Set up argv in interp memory */
    int64_t argv_ptrs[64];
    for (int i=1; i<argc && i<64; i++) {
        int len = strlen(argv[i])+1;
        memcpy(mem+mem_top, argv[i], len);
        argv_ptrs[i-1] = mem_top;
        mem_top = (mem_top+len+7) & ~7;
    }
    int64_t argv_arr = mem_top;
    for (int i=0; i<argc-1; i++) {
        uint64_t p = argv_ptrs[i];
        memcpy(mem+argv_arr+i*8, &p, 8);
    }
    mem_top += 8*(argc);
    regs[0] = argc-1;
    regs[1] = argv_arr;

    /* Run top-level declarations */
    tk = 0;
    while (ty()!=TOK_EOF) {
        int t = toks[tk].type;
        if (t==TOK_NEWLINE) { tk++; continue; }
        if (t==TOK_INDENT) {
            if (toks[tk].len==0) { tk++; continue; }
            while(ty()!=TOK_NEWLINE&&ty()!=TOK_EOF) tk++;
            continue;
        }
        if (t==TOK_CONST||t==TOK_BUF||t==TOK_VAR) { exec_stmt(); continue; }
        /* Forth-style: INT constant NAME */
        if (t==TOK_INT && tk+2<ntoks && toks[tk+1].type==TOK_IDENT) {
            char w[16]; tok_text(tk+1,w,sizeof w);
            if (!strcmp(w,"constant") && toks[tk+2].type==TOK_IDENT) {
                int64_t val = parse_int_tok(tk);
                char nm[64]; tok_text(tk+2,nm,sizeof nm);
                set_const(nm, val);
                tk+=3; continue;
            }
        }
        if (t==TOK_IDENT) {
            char nm[64]; tok_text(tk,nm,sizeof nm);
            /* shorthand const: NAME INT at toplevel */
            if (tk+2<ntoks && toks[tk+1].type==TOK_INT &&
                (toks[tk+2].type==TOK_NEWLINE||toks[tk+2].type==TOK_EOF)) {
                int64_t val=parse_int_tok(tk+1);
                set_const(nm,val); tk+=2; continue;
            }
            if (!strcmp(nm,"host")||!strcmp(nm,"kernel")) {
                if(tk+1<ntoks&&toks[tk+1].type==TOK_IDENT) tok_text(tk+1,nm,sizeof nm);
            }
            int ci=find_compo(nm);
            if (ci>=0) { tk=compos[ci].body_tk; skip_body(compos[ci].body_indent); continue; }
            while(ty()!=TOK_NEWLINE&&ty()!=TOK_EOF) tk++;
            continue;
        }
        while(ty()!=TOK_NEWLINE&&ty()!=TOK_EOF) tk++;
    }

    /* Run main */
    if (find_compo("main")<0) { fprintf(stderr,"no main composition\n"); return 1; }

    int code = 0;
    if (setjmp(jmp_exit)==0) {
        call_compo("main", 0);
    } else {
        code = exit_code;
    }
    return code;
}
