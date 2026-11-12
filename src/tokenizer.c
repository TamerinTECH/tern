// Gemma 4 tokenizer ("gemma4" in GGUF): byte-pair encoding with ranked merges over text whose spaces were
// replaced by U+2581, the same procedure and the same token ids as llama.cpp.
#include "tern.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tokenizer {
    int n_vocab;
    const char **text; int *len; int *type;
    int *table; uint32_t mask;                  // open addressing hash: token text -> id
    const char **mtext; int *mlen; int n_merges;
    int *mtable; uint32_t mmask;                // merge "left right" -> rank
    int bos, eos, unk;
    int max_special_len;
    int byte_tok[256];
};

static uint32_t hash_bytes(const char *s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}
static uint32_t hash_pair(const char *a, int na, const char *b, int nb) {      // == hash_bytes(a + ' ' + b)
    uint32_t h = 2166136261u;
    for (int i = 0; i < na; i++) { h ^= (uint8_t)a[i]; h *= 16777619u; }
    h ^= ' '; h *= 16777619u;
    for (int i = 0; i < nb; i++) { h ^= (uint8_t)b[i]; h *= 16777619u; }
    return h;
}

static int lookup(const tokenizer *t, const char *s, int n) {
    uint32_t i = hash_bytes(s, n) & t->mask;
    for (;;) {
        int id = t->table[i];
        if (id < 0) return -1;
        if (t->len[id] == n && memcmp(t->text[id], s, n) == 0) return id;
        i = (i + 1) & t->mask;
    }
}

// rank of the merge of two adjacent pieces, or -1
static int merge_rank(const tokenizer *t, const char *a, int na, const char *b, int nb) {
    uint32_t i = hash_pair(a, na, b, nb) & t->mmask;
    for (;;) {
        int r = t->mtable[i];
        if (r < 0) return -1;
        const char *m = t->mtext[r];
        if (t->mlen[r] == na + nb + 1 && memcmp(m, a, na) == 0 && m[na] == ' ' && memcmp(m + na + 1, b, nb) == 0) return r;
        i = (i + 1) & t->mmask;
    }
}

tokenizer *tok_load(const gguf *g) {
    uint64_t ml = 0; const char *model = gguf_get_str(g, "tokenizer.ggml.model", &ml);
    if (!model || ml != 6 || memcmp(model, "gemma4", 6)) { fprintf(stderr, "tokenizer: only the gemma4 tokenizer is supported\n"); return NULL; }
    tokenizer *t = calloc(1, sizeof *t);
    uint32_t et; const uint8_t *p;
    int64_t n = gguf_get_array(g, "tokenizer.ggml.tokens", &et, &p);
    if (n <= 0 || et != 8) { fprintf(stderr, "tokenizer: no token list\n"); return NULL; }
    t->n_vocab = (int)n;
    t->text = malloc(n * sizeof *t->text); t->len = malloc(n * sizeof *t->len);
    for (int64_t i = 0; i < n; i++) {
        uint64_t l; memcpy(&l, p, 8); p += 8;
        t->text[i] = (const char *)p; t->len[i] = (int)l; p += l;
    }
    t->type = malloc(n * sizeof(int));
    const uint8_t *tp;
    if (gguf_get_array(g, "tokenizer.ggml.token_type", &et, &tp) == n) memcpy(t->type, tp, n * 4); else for (int64_t i = 0; i < n; i++) t->type[i] = 1;
    uint32_t cap = 1; while (cap < (uint32_t)n * 2) cap <<= 1;
    t->mask = cap - 1; t->table = malloc(cap * sizeof(int));
    for (uint32_t i = 0; i < cap; i++) t->table[i] = -1;
    for (int id = 0; id < t->n_vocab; id++) {
        uint32_t i = hash_bytes(t->text[id], t->len[id]) & t->mask;
        int dup = 0;
        while (t->table[i] >= 0) {
            int o = t->table[i];
            if (t->len[o] == t->len[id] && memcmp(t->text[o], t->text[id], t->len[id]) == 0) { dup = 1; break; }
            i = (i + 1) & t->mask;
        }
        if (!dup) t->table[i] = id;
        // type 3 = control, 4 = user defined: matched as whole strings before BPE
        if ((t->type[id] == 3 || t->type[id] == 4) && t->len[id] > t->max_special_len) t->max_special_len = t->len[id];
    }
    int64_t nm = gguf_get_array(g, "tokenizer.ggml.merges", &et, &p);
    if (nm <= 0 || et != 8) { fprintf(stderr, "tokenizer: no merges\n"); return NULL; }
    t->n_merges = (int)nm; t->mtext = malloc(nm * sizeof *t->mtext); t->mlen = malloc(nm * sizeof *t->mlen);
    cap = 1; while (cap < (uint32_t)nm * 2) cap <<= 1;
    t->mmask = cap - 1; t->mtable = malloc(cap * sizeof(int));
    for (uint32_t i = 0; i < cap; i++) t->mtable[i] = -1;
    for (int r = 0; r < t->n_merges; r++) {
        uint64_t l; memcpy(&l, p, 8); p += 8;
        t->mtext[r] = (const char *)p; t->mlen[r] = (int)l; p += l;
        uint32_t i = hash_bytes(t->mtext[r], t->mlen[r]) & t->mmask;
        int dup = 0;
        while (t->mtable[i] >= 0) {     // the first (lowest) rank of a repeated merge wins
            int o = t->mtable[i];
            if (t->mlen[o] == t->mlen[r] && memcmp(t->mtext[o], t->mtext[r], t->mlen[r]) == 0) { dup = 1; break; }
            i = (i + 1) & t->mmask;
        }
        if (!dup) t->mtable[i] = r;
    }
    uint32_t v;
    t->bos = gguf_get_u32(g, "tokenizer.ggml.bos_token_id", &v) ? (int)v : 2;
    t->eos = gguf_get_u32(g, "tokenizer.ggml.eos_token_id", &v) ? (int)v : 1;
    t->unk = gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", &v) ? (int)v : 3;
    for (int b = 0; b < 256; b++) {
        char buf[8]; int l = snprintf(buf, sizeof buf, "<0x%02X>", b);
        t->byte_tok[b] = lookup(t, buf, l);
    }
    return t;
}

int tok_n_vocab(const tokenizer *t) { return t->n_vocab; }
int tok_bos(const tokenizer *t) { return t->bos; }
int tok_eos(const tokenizer *t) { return t->eos; }
int tok_find(const tokenizer *t, const char *text) { return lookup(t, text, (int)strlen(text)); }
int tok_is_special(const tokenizer *t, int id) { return t->type[id] == 3; }

int tok_piece(const tokenizer *t, int id, char *buf, int max) {
    if (id < 0 || id >= t->n_vocab) return 0;
    if (t->type[id] == 6) {  // byte token "<0xXX>"
        if (max < 1) return 0;
        buf[0] = (char)strtol(t->text[id] + 3, NULL, 16);
        return 1;
    }
    const char *s = t->text[id]; int n = t->len[id], o = 0;
    for (int i = 0; i < n && o < max;) {
        if (i + 2 < n && (uint8_t)s[i] == 0xE2 && (uint8_t)s[i + 1] == 0x96 && (uint8_t)s[i + 2] == 0x81) { buf[o++] = ' '; i += 3; }
        else buf[o++] = s[i++];
    }
    return o;
}

// ---- BPE over one word (a run without newlines, spaces already escaped)
typedef struct { int start, n, prev, next; } sym;
typedef struct { int left, right, size, rank; } bigram;
typedef struct { bigram *a; int n, cap; } heap;

static int bg_before(const bigram *l, const bigram *r) {    // l is merged before r: lower rank, then further left
    return l->rank < r->rank || (l->rank == r->rank && l->left < r->left);
}
static void heap_push(heap *h, bigram b) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap * 2 : 256; h->a = realloc(h->a, h->cap * sizeof(bigram)); }
    int i = h->n++;
    while (i > 0) { int p = (i - 1) / 2; if (!bg_before(&b, &h->a[p])) break; h->a[i] = h->a[p]; i = p; }
    h->a[i] = b;
}
static bigram heap_pop(heap *h) {
    bigram top = h->a[0], last = h->a[--h->n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1; if (c >= h->n) break;
        if (c + 1 < h->n && bg_before(&h->a[c + 1], &h->a[c])) c++;
        if (!bg_before(&h->a[c], &last)) break;
        h->a[i] = h->a[c]; i = c;
    }
    h->a[i] = last;
    return top;
}

static void try_add(const tokenizer *t, const char *s, sym *sy, heap *h, int l, int r) {
    if (l < 0 || r < 0) return;
    int rank = merge_rank(t, s + sy[l].start, sy[l].n, s + sy[r].start, sy[r].n);
    if (rank < 0) return;
    bigram b = {l, r, sy[l].n + sy[r].n, rank};
    heap_push(h, b);
}

static int utf8_len(uint8_t c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

static int emit_piece(const tokenizer *t, const char *s, int n, int *out, int max) {
    int id = lookup(t, s, n), o = 0;
    if (id >= 0) { if (o < max) out[o++] = id; return o; }
    for (int k = 0; k < n; k++) {       // unknown piece: byte tokens
        int bt = t->byte_tok[(uint8_t)s[k]];
        if (bt >= 0 && o < max) out[o++] = bt;
    }
    return o;
}

static int bpe_word(const tokenizer *t, const char *s, int n, int *out, int max) {
    if (n == 0) return 0;
    if (s[0] == '\n') { int id = lookup(t, s, n); if (id >= 0) { if (max > 0) out[0] = id; return max > 0; } }   // a run of newlines that is a token
    sym *sy = malloc((size_t)n * sizeof(sym)); int ns = 0;
    for (int i = 0; i < n;) {
        int l = utf8_len((uint8_t)s[i]); if (i + l > n) l = n - i;
        sy[ns] = (sym){i, l, ns - 1, ns + 1}; ns++; i += l;
    }
    sy[ns - 1].next = -1;
    heap h = {0};
    for (int i = 1; i < ns; i++) try_add(t, s, sy, &h, i - 1, i);
    while (h.n > 0) {
        bigram b = heap_pop(&h);
        sym *L = &sy[b.left], *R = &sy[b.right];
        if (L->n == 0 || R->n == 0 || L->n + R->n != b.size) continue;     // outdated
        L->n += R->n; R->n = 0;
        L->next = R->next;
        if (R->next >= 0) sy[R->next].prev = b.left;
        try_add(t, s, sy, &h, L->prev, b.left);
        try_add(t, s, sy, &h, b.left, L->next);
    }
    int o = 0;
    for (int i = 0; i >= 0; i = sy[i].next) o += emit_piece(t, s + sy[i].start, sy[i].n, out + o, max - o);
    free(sy); free(h.a);
    return o;
}

static int flush_plain(const tokenizer *t, const char *s, int n, int *out, int max) {
    // escape whitespace: ' ' -> U+2581; then words are maximal runs of newlines or of anything else
    char *e = malloc((size_t)n * 3 + 1); int m = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == ' ') { e[m++] = (char)0xE2; e[m++] = (char)0x96; e[m++] = (char)0x81; }
        else e[m++] = s[i];
    }
    int o = 0;
    for (int i = 0; i < m;) {
        int j = i, nl = e[i] == '\n';
        while (j < m && (e[j] == '\n') == nl) j++;
        o += bpe_word(t, e + i, j - i, out + o, max - o);
        i = j;
    }
    free(e);
    return o;
}

int tok_encode(const tokenizer *t, const char *text, int add_bos, int parse_special, int *out, int max) {
    int o = 0, n = (int)strlen(text), plain0 = 0;
    if (add_bos && o < max) out[o++] = t->bos;
    for (int i = 0; i < n;) {
        int hit = -1, hl = 0;
        // control and user-defined tokens are matched as whole strings; try the longest match first
        int lim = t->max_special_len < n - i ? t->max_special_len : n - i;
        for (int l = lim; l >= 1; l--) {
            int id = lookup(t, text + i, l);
            if (id >= 0 && (t->type[id] == 4 || (t->type[id] == 3 && parse_special))) { hit = id; hl = l; break; }
        }
        if (hit >= 0) {
            o += flush_plain(t, text + plain0, i - plain0, out + o, max - o);
            if (o < max) out[o++] = hit;
            i += hl; plain0 = i;
        } else i++;
    }
    o += flush_plain(t, text + plain0, n - plain0, out + o, max - o);
    return o;
}
