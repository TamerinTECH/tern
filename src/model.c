// Gemma 4 forward pass (text). One call = one pass over the weights for up to TERN_MAX_SLOTS tokens.
// Covers the published variants: per-layer token inputs and shared key/value layers (E2B, E4B), grouped
// key/value heads and a dense feed-forward (31B), and a dense feed-forward plus routed experts (26B-A4B).
// Reference for the computation: llama.cpp src/models/gemma4.cpp; tests/ref_gemma4.py is an independent
// float implementation.
#define _GNU_SOURCE
#include "tern.h"
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define MAX_EXP_USED 16

typedef struct {
    pmat qkv, o, gu, down;                  // rows of qkv: queries, then keys and values (layers with their own cache)
    int ple_packed; pmat pg, pp; fmat fg, fp;       // per-layer input: gate (PL x E) and projection (E x PL)
    pmat ex_gu, ex_down;                    // experts: n_expert * 2*ffe rows (E wide), n_expert * E rows (ffe wide)
    const float *router, *router_scale, *ex_scale;
    const float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm, *ple_norm;
    const float *post_ffn_norm_1, *pre_ffn_norm_2, *post_ffn_norm_2;
    float out_scale;
    int hd, nq, nkv, n_ff, local, kv_src, own_kv, has_v, moe;
    uint16_t **K, **V;                      // cache per sequence, half floats: for each of the nkv heads a contiguous run of
                                            // kv_rows rows of hd values (row r of head g at (g * kv_rows + r) * hd);
                                            // rows past the committed length are scratch for the current pass
} layer;

struct model {
    int n_embd, n_layer, n_head, n_vocab, n_ctx, n_swa, n_seq, kv0, pl;
    int n_expert, n_exp_used, ffe;
    int max_hd, max_ff, max_nq, max_kv;
    int hs, hf;                             // head sizes of sliding-window and full-attention layers (0: no such layer)
    size_t kv_rows;                         // rows per head in the caches: n_ctx + TERN_MAX_SLOTS
    float eps, rope_base, rope_base_swa, softcap;
    const float *rope_freqs; int n_rope_freqs;
    int *len;                               // committed tokens per sequence
    int row[TERN_MAX_SLOTS], pos[TERN_MAX_SLOTS];
    layer *layers;
    const float *out_norm;
    const gguf_tensor *embd, *ple;          // token embedding (lookup); per-layer token inputs (lookup)
    pmat head;                              // the token embedding again, packed, as the output matrix
    fmat ple_proj; const float *ple_proj_norm;
    uint8_t *arena; size_t arena_size, arena_used;
    // scratch, per slot
    float *x[TERN_MAX_SLOTS], *sa[TERN_MAX_SLOTS], *t0[TERN_MAX_SLOTS], *t1[TERN_MAX_SLOTS], *pli[TERN_MAX_SLOTS], *mo[TERN_MAX_SLOTS];
    qvec qv[TERN_MAX_SLOTS], qx[TERN_MAX_SLOTS];        // matrix input; expert input
    float *rope[TERN_MAX_SLOTS];            // cos/sin pairs: sliding-window layers, then full-attention layers
    float *scores; int *vis;                // per thread: n_head * (n_ctx + slots) scores, (n_ctx + slots) visible rows
    // experts
    int ex_id[TERN_MAX_SLOTS][MAX_EXP_USED]; float ex_w[TERN_MAX_SLOTS][MAX_EXP_USED];
    float *rin[TERN_MAX_SLOTS], *rlg[TERN_MAX_SLOTS];           // router: input (E), logit per expert
    float *ex_gu[TERN_MAX_SLOTS], *ex_out[TERN_MAX_SLOTS];      // per slot: n_exp_used * 2*ffe, n_exp_used * E
    qvec *ex_h;                             // [slot * n_exp_used + j]: quantized gelu(gate) * up
    float *logits;                          // exact output layer: TERN_MAX_SLOTS * n_vocab
};

static int ra_on = 1;           // read-ahead by idle threads (see below); TERN_RA=0 turns it off
static int ra_idle_mode = 1;    // TERN_RA=1: only inside the parallel regions of the per-slot steps; 2 (default): by waiting workers
static void ra_idle(void *arg, int tid);

int model_n_embd(const model *m) { return m->n_embd; }
int model_n_vocab(const model *m) { return m->n_vocab; }
int model_n_ctx(const model *m) { return m->n_ctx; }
int model_len(const model *m, int seq) { return m->len[seq]; }
void model_truncate(model *m, int seq, int len) { if (len < m->len[seq]) m->len[seq] = len; }
double model_weight_bytes(const model *m) { return (double)m->arena_used; }

static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n + 127) / 64 * 64)) { fprintf(stderr, "out of memory\n"); exit(1); }
    memset(p, 0, (n + 127) / 64 * 64);
    return p;
}

// ---------------------------------------------------------------- loading
typedef struct { uint32_t type; const uint8_t *src; size_t src_stride; uint8_t *dst; size_t dst_stride; int rows, cols; } pack_task;
typedef struct {
    const gguf *g; uint8_t *arena; int dry; size_t off;
    pack_task *tasks; int n_tasks, cap_tasks;
} builder;

static const gguf_tensor *find(const gguf *g, const char *fmt, int il) {
    char name[128]; snprintf(name, sizeof name, fmt, il);
    return gguf_find_tensor(g, name);
}
static const gguf_tensor *need(const gguf *g, const char *fmt, int il) {
    const gguf_tensor *t = find(g, fmt, il);
    if (!t) { char name[128]; snprintf(name, sizeof name, fmt, il); fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    return t;
}
static const float *f32_of(const gguf *g, const char *fmt, int il, int n) {
    const gguf_tensor *t = need(g, fmt, il);
    if (t->type != GT_F32 || (int)t->dims[0] != n) { fprintf(stderr, "%.*s: expected %d floats\n", t->name_len, t->name, n); exit(1); }
    return (const float *)t->data;
}
static int rows_of(const gguf_tensor *t) { int r = 1; for (int d = 1; d < t->n_dims; d++) r *= (int)t->dims[d]; return r; }

static uint8_t *arena_take(builder *b, size_t n) {
    size_t at = b->off; b->off += (n + 63) / 64 * 64;
    return b->dry ? NULL : b->arena + at;
}

// one packed matrix from up to 3 tensors with the same width (their rows follow each other)
static pmat packed(builder *b, const gguf_tensor *t0, const gguf_tensor *t1, const gguf_tensor *t2) {
    const gguf_tensor *ts[3] = {t0, t1, t2};
    int cols = (int)t0->dims[0], kind = p4_kind_of(t0->type), rows = 0;
    if (kind < 0 || cols % QB) { fprintf(stderr, "%.*s: type %s (width %d) has no packed form\n", t0->name_len, t0->name, quant_type_name(t0->type), cols); exit(1); }
    for (int i = 0; i < 3 && ts[i]; i++) {
        if ((int)ts[i]->dims[0] != cols || p4_kind_of(ts[i]->type) != kind) { fprintf(stderr, "%.*s: does not match %.*s\n", ts[i]->name_len, ts[i]->name, t0->name_len, t0->name); exit(1); }
        rows += rows_of(ts[i]);
    }
    pmat M = {NULL, rows, cols, kind, p4_row_bytes(cols, kind)};
    uint8_t *dst = arena_take(b, M.stride * (size_t)rows);
    M.W = dst;
    if (!b->dry) for (int i = 0; i < 3 && ts[i]; i++) {
        if (b->n_tasks == b->cap_tasks) { b->cap_tasks = b->cap_tasks ? b->cap_tasks * 2 : 1024; b->tasks = realloc(b->tasks, sizeof(pack_task) * (size_t)b->cap_tasks); }
        int r = rows_of(ts[i]);
        b->tasks[b->n_tasks++] = (pack_task){ts[i]->type, ts[i]->data, quant_row_bytes(ts[i]->type, cols), dst, M.stride, r, cols};
        dst += M.stride * (size_t)r;
    }
    return M;
}

// a float matrix: F32 stays, F16 stays, BF16 becomes F32
static fmat floats(builder *b, const gguf_tensor *t) {
    int cols = (int)t->dims[0], rows = rows_of(t);
    fmat M = {NULL, rows, cols, t->type == GT_F16};
    if (t->type != GT_F32 && t->type != GT_F16 && t->type != GT_BF16) { fprintf(stderr, "%.*s: expected a float matrix\n", t->name_len, t->name); exit(1); }
    size_t n = (size_t)rows * cols;
    uint8_t *dst = arena_take(b, n * (M.f16 ? 2 : 4));
    M.W = dst;
    if (!b->dry) {
        if (t->type == GT_BF16) for (int r = 0; r < rows; r++) dequant_row(GT_BF16, t->data + (size_t)r * cols * 2, cols, (float *)dst + (size_t)r * cols);
        else memcpy(dst, t->data, n * (M.f16 ? 2 : 4));
    }
    return M;
}

static void pack_run(void *arg, int tid) {
    pack_task *t = arg; int nt = pool_threads();
    int r0 = (int)((int64_t)t->rows * tid / nt), r1 = (int)((int64_t)t->rows * (tid + 1) / nt);
    for (int r = r0; r < r1; r++) p4_pack_row(t->type, t->src + t->src_stride * (size_t)r, t->cols, t->dst + t->dst_stride * (size_t)r);
}

static const char *key_prefix = "gemma4";      // "gemma4-assistant" while a drafter file is read
static int key_u32(const gguf *g, const char *key, int def) {
    char k[128]; uint32_t v; snprintf(k, sizeof k, "%s.%s", key_prefix, key);
    return gguf_get_u32(g, k, &v) ? (int)v : def;
}
static float key_f32(const gguf *g, const char *key, float def) {
    char k[128]; float v; snprintf(k, sizeof k, "%s.%s", key_prefix, key);
    return gguf_get_f32(g, k, &v) ? v : def;
}
// a per-layer value that is stored as one number or as an array (ints or bools)
static int key_layer(const gguf *g, const char *key, int il, int def) {
    char k[128]; snprintf(k, sizeof k, "%s.%s", key_prefix, key);
    uint32_t et; const uint8_t *p; int64_t n = gguf_get_array(g, k, &et, &p);
    if (n < 0) return key_u32(g, key, def);
    if (il >= n) return def;
    if (et == 7 || et == 0 || et == 1) return p[il];                        // bool, u8, i8
    if (et == 4 || et == 5) { uint32_t v; memcpy(&v, p + 4 * (size_t)il, 4); return (int)v; }
    if (et == 10 || et == 11) { uint64_t v; memcpy(&v, p + 8 * (size_t)il, 8); return (int)v; }
    return def;
}

static void build(builder *b, model *m) {
    const gguf *g = b->g;
    for (int il = 0; il < m->n_layer; il++) {
        layer *L = &m->layers[il];
        const gguf_tensor *q = need(g, "blk.%d.attn_q.weight", il), *k = NULL, *v = NULL;
        if (L->own_kv) { k = need(g, "blk.%d.attn_k.weight", il); v = find(g, "blk.%d.attn_v.weight", il); }
        L->has_v = v != NULL;
        L->qkv = packed(b, q, k, v);
        L->o = packed(b, need(g, "blk.%d.attn_output.weight", il), NULL, NULL);
        L->gu = packed(b, need(g, "blk.%d.ffn_gate.weight", il), need(g, "blk.%d.ffn_up.weight", il), NULL);
        L->down = packed(b, need(g, "blk.%d.ffn_down.weight", il), NULL, NULL);
        if (L->moe) {
            const gguf_tensor *gu = find(g, "blk.%d.ffn_gate_up_exps.weight", il);
            if (gu) L->ex_gu = packed(b, gu, NULL, NULL);
            else { fprintf(stderr, "experts with separate gate and up matrices are not supported\n"); exit(1); }
            L->ex_down = packed(b, need(g, "blk.%d.ffn_down_exps.weight", il), NULL, NULL);
        }
        if (m->pl) {
            const gguf_tensor *pg = need(g, "blk.%d.inp_gate.weight", il), *pp = need(g, "blk.%d.proj.weight", il);
            L->ple_packed = p4_kind_of(pg->type) >= 0 && p4_kind_of(pp->type) >= 0;
            if (L->ple_packed) { L->pg = packed(b, pg, NULL, NULL); L->pp = packed(b, pp, NULL, NULL); }
            else { L->fg = floats(b, pg); L->fp = floats(b, pp); }
        }
    }
    if (m->pl) m->ple_proj = floats(b, need(g, "per_layer_model_proj.weight", 0));
    m->head = packed(b, m->embd, NULL, NULL);
}

// sizes first, then one arena (huge pages where the system gives them), then packing on all threads
static void run_packing(builder *b, const uint8_t *keep) {
    for (int i = 0; i < b->n_tasks; i++) {
        pack_task *t = &b->tasks[i];
        pool_run(pack_run, t);
        // the file pages of a repacked tensor are not needed again (except the token embedding, which is looked up)
        uintptr_t a0 = ((uintptr_t)t->src + 4095) & ~(uintptr_t)4095, a1 = ((uintptr_t)t->src + t->src_stride * (size_t)t->rows) & ~(uintptr_t)4095;
        if (a1 > a0 && t->src != keep) madvise((void *)a0, a1 - a0, MADV_DONTNEED);
    }
    free(b->tasks); b->tasks = NULL; b->n_tasks = b->cap_tasks = 0;
}
// Page size for the weights (measurements: docs/DESIGN.md, section 8). On the benchmark VM memory behind
// transparent huge pages is read at 85 to 260 GB/s depending on the blocks an allocation receives, memory behind
// 4 KB pages at 170 - 265 GB/s; but small pages cost a little on every page, which shows when the
// matrices are large or when arithmetic, not memory, is the limit. The rule that was best or equal in every case
// measured: 4 KB pages if the model is small for the machine (less than 512 MB of weights per thread) and at most
// 8 sequences are evaluated per pass, huge pages otherwise. TERN_HUGE=1 or 0 overrides it.
static int huge_default = 1;
int tern_huge_pages(void) { const char *e = getenv("TERN_HUGE"); return e ? atoi(e) != 0 : huge_default; }
static uint8_t *arena_alloc(size_t n) {
    uint8_t *a = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a == MAP_FAILED) { perror("mmap"); return NULL; }
    madvise(a, n, tern_huge_pages() ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return a;
}

model *model_load(const gguf *g, const model_params *p) {
    uint64_t al = 0; const char *arch = gguf_get_str(g, "general.architecture", &al);
    if (!arch || al != 6 || memcmp(arch, "gemma4", 6)) { fprintf(stderr, "not a Gemma 4 model file (general.architecture)\n"); return NULL; }
    model *m = calloc(1, sizeof *m);
    const int E = m->n_embd = key_u32(g, "embedding_length", 0), NL = m->n_layer = key_u32(g, "block_count", 0);
    m->n_head = key_u32(g, "attention.head_count", 0);
    m->n_swa = key_u32(g, "attention.sliding_window", 0);
    m->eps = key_f32(g, "attention.layer_norm_rms_epsilon", 1e-6f);
    m->rope_base = key_f32(g, "rope.freq_base", 1000000.0f); m->rope_base_swa = key_f32(g, "rope.freq_base_swa", 10000.0f);
    m->softcap = key_f32(g, "final_logit_softcapping", 0.0f);
    m->pl = key_u32(g, "embedding_length_per_layer_input", 0);
    m->kv0 = NL - key_u32(g, "attention.shared_kv_layers", 0);
    m->n_expert = key_u32(g, "expert_count", 0); m->n_exp_used = key_u32(g, "expert_used_count", 0); m->ffe = key_u32(g, "expert_feed_forward_length", 0);
    const int hd_full = key_u32(g, "attention.key_length", 256), hd_swa = key_u32(g, "attention.key_length_swa", hd_full);
    if (!E || !NL || !m->n_head || E % QB || m->kv0 < 2 || m->n_exp_used > MAX_EXP_USED) { fprintf(stderr, "unsupported Gemma 4 configuration\n"); return NULL; }
    m->n_ctx = p->n_ctx; m->n_seq = p->n_seq > 0 ? p->n_seq : 1;
    m->len = calloc((size_t)m->n_seq, sizeof(int));
    m->embd = need(g, "token_embd.weight", 0); m->n_vocab = (int)m->embd->dims[1];
    if ((int)m->embd->dims[0] != E || !quant_row_bytes(m->embd->type, E)) { fprintf(stderr, "token_embd: unsupported type %s\n", quant_type_name(m->embd->type)); return NULL; }
    m->out_norm = f32_of(g, "output_norm.weight", 0, E);
    const gguf_tensor *rf = gguf_find_tensor(g, "rope_freqs.weight");
    if (rf && rf->type == GT_F32) { m->rope_freqs = (const float *)rf->data; m->n_rope_freqs = (int)rf->dims[0]; }
    if (m->pl) {
        m->ple = need(g, "per_layer_token_embd.weight", 0);
        if ((int)m->ple->dims[0] != m->pl * NL || !quant_row_bytes(m->ple->type, m->pl * NL)) { fprintf(stderr, "per_layer_token_embd: unsupported shape or type\n"); return NULL; }
        m->ple_proj_norm = f32_of(g, "per_layer_proj_norm.weight", 0, m->pl);
    }

    m->layers = calloc((size_t)NL, sizeof(layer));
    for (int il = 0; il < NL; il++) {
        layer *L = &m->layers[il];
        L->local = key_layer(g, "attention.sliding_window_pattern", il, 0);
        L->hd = L->local ? hd_swa : hd_full;
        L->nkv = key_layer(g, "attention.head_count_kv", il, 1);
        L->nq = m->n_head * L->hd;
        L->n_ff = key_layer(g, "feed_forward_length", il, 0);
        L->own_kv = il < m->kv0;
        // a layer without its own cache reads the last layer below kv0 with the same kind of attention
        L->kv_src = L->own_kv ? il : m->kv0 - (L->local ? 2 : 1);
        L->moe = m->n_expert > 0 && find(g, "blk.%d.ffn_gate_inp.weight", il) != NULL;
        if (m->n_head % L->nkv || L->hd % 16 || L->n_ff % QB) { fprintf(stderr, "layer %d: unsupported attention or feed-forward shape\n", il); return NULL; }
        if (L->hd > m->max_hd) m->max_hd = L->hd;
        if (L->local) m->hs = L->hd; else m->hf = L->hd;
        if (L->n_ff > m->max_ff) m->max_ff = L->n_ff;
        if (L->nq > m->max_nq) m->max_nq = L->nq;
        if (L->nkv * L->hd > m->max_kv) m->max_kv = L->nkv * L->hd;
        L->attn_norm = f32_of(g, "blk.%d.attn_norm.weight", il, E);
        L->q_norm = f32_of(g, "blk.%d.attn_q_norm.weight", il, L->hd);
        if (L->own_kv) L->k_norm = f32_of(g, "blk.%d.attn_k_norm.weight", il, L->hd);
        L->post_attn_norm = f32_of(g, "blk.%d.post_attention_norm.weight", il, E);
        L->ffn_norm = f32_of(g, "blk.%d.ffn_norm.weight", il, E);
        L->post_ffn_norm = f32_of(g, "blk.%d.post_ffw_norm.weight", il, E);
        const gguf_tensor *os = find(g, "blk.%d.layer_output_scale.weight", il);
        L->out_scale = os ? *(const float *)os->data : 1.0f;
        if (m->pl) L->ple_norm = f32_of(g, "blk.%d.post_norm.weight", il, E);
        if (L->moe) {
            const gguf_tensor *r = need(g, "blk.%d.ffn_gate_inp.weight", il);
            if (r->type != GT_F32 || (int)r->dims[0] != E || (int)r->dims[1] != m->n_expert) { fprintf(stderr, "layer %d: unexpected router\n", il); return NULL; }
            L->router = (const float *)r->data;
            L->router_scale = f32_of(g, "blk.%d.ffn_gate_inp.scale", il, E);
            const gguf_tensor *es = find(g, "blk.%d.ffn_down_exps.scale", il);
            L->ex_scale = es ? (const float *)es->data : NULL;
            L->post_ffn_norm_1 = f32_of(g, "blk.%d.post_ffw_norm_1.weight", il, E);
            L->pre_ffn_norm_2 = f32_of(g, "blk.%d.pre_ffw_norm_2.weight", il, E);
            L->post_ffn_norm_2 = f32_of(g, "blk.%d.post_ffw_norm_2.weight", il, E);
        }
    }
    // layers that share keys and values read layer kv0 - 2 (sliding window) or kv0 - 1 (full attention)
    if (m->kv0 < NL && (m->layers[m->kv0 - 1].local || !m->layers[m->kv0 - 2].local)) { fprintf(stderr, "unexpected layer pattern below the shared key/value layers\n"); return NULL; }
    if (m->max_hd > 512 || m->n_expert > 512 || (m->n_swa && m->n_swa <= TERN_MAX_SLOTS)) { fprintf(stderr, "unsupported sizes (head, experts or window)\n"); return NULL; }

    builder b = {.g = g, .dry = 1};
    build(&b, m);
    m->arena_size = b.off;
    huge_default = m->n_seq > 8 || m->arena_size / (size_t)pool_threads() >= ((size_t)512 << 20);
    if (!(m->arena = arena_alloc(m->arena_size))) return NULL;
    b.dry = 0; b.off = 0; b.arena = m->arena;
    build(&b, m);
    m->arena_used = b.off;
    run_packing(&b, m->embd->data);

    // caches and scratch
    size_t kv_rows = m->kv_rows = (size_t)(m->n_ctx + TERN_MAX_SLOTS);
    for (int il = 0; il < NL; il++) {
        layer *L = &m->layers[il];
        if (!L->own_kv) continue;
        L->K = malloc((size_t)m->n_seq * sizeof(void *)); L->V = malloc((size_t)m->n_seq * sizeof(void *));
        for (int q = 0; q < m->n_seq; q++) {
            L->K[q] = xalloc(kv_rows * (size_t)(L->nkv * L->hd) * sizeof(uint16_t) + 64);
            L->V[q] = xalloc(kv_rows * (size_t)(L->nkv * L->hd) * sizeof(uint16_t) + 64);
        }
    }
    int big0 = m->max_nq + 2 * m->max_kv, big1 = m->max_ff > m->max_nq ? m->max_ff : m->max_nq;
    if (2 * m->max_ff > big0) big0 = 2 * m->max_ff;
    if (m->pl * NL > big0) big0 = m->pl * NL;
    if (E > big0) big0 = E;
    if (E > big1) big1 = E;
    for (int s = 0; s < TERN_MAX_SLOTS; s++) {
        m->x[s] = xalloc(sizeof(float) * (size_t)E); m->sa[s] = xalloc(sizeof(float) * (size_t)E); m->mo[s] = xalloc(sizeof(float) * (size_t)E);
        m->t0[s] = xalloc(sizeof(float) * (size_t)big0); m->t1[s] = xalloc(sizeof(float) * (size_t)big1);
        qvec_alloc(&m->qv[s], big1);
        m->rope[s] = xalloc(sizeof(float) * (size_t)(m->hs + m->hf));
        if (m->pl) m->pli[s] = xalloc(sizeof(float) * (size_t)(m->pl * NL));
        if (m->n_expert) {
            qvec_alloc(&m->qx[s], E);
            m->rin[s] = xalloc(sizeof(float) * (size_t)E); m->rlg[s] = xalloc(sizeof(float) * (size_t)m->n_expert);
            m->ex_gu[s] = xalloc(sizeof(float) * (size_t)m->n_exp_used * 2 * m->ffe);
            m->ex_out[s] = xalloc(sizeof(float) * (size_t)m->n_exp_used * E);
        }
    }
    if (m->n_expert) {
        m->ex_h = calloc((size_t)TERN_MAX_SLOTS * m->n_exp_used, sizeof(qvec));
        for (int i = 0; i < TERN_MAX_SLOTS * m->n_exp_used; i++) qvec_alloc(&m->ex_h[i], m->ffe);
    }
    const char *ra = getenv("TERN_RA");
    if (ra && atoi(ra) == 0) ra_on = 0;
    if (!ra_on || (ra && atoi(ra) == 1)) ra_idle_mode = 0;
    if (ra_idle_mode) pool_set_idle(ra_idle);
    m->scores = xalloc((size_t)pool_threads() * m->n_head * kv_rows * sizeof(float));
    m->vis = xalloc((size_t)pool_threads() * kv_rows * sizeof(int));
    return m;
}

// ---------------------------------------------------------------- small float ops
static void rmsnorm(const float *x, const float *w, int n, float eps, float *y) {
    float ss = dot_f32(x, x, n);
    float sc = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i] * sc * w[i];
    else for (int i = 0; i < n; i++) y[i] = x[i] * sc;
}

// llama.cpp: cache[2i] = cos(theta_i / f_i), cache[2i+1] = sin(theta_i / f_i), theta_0 = pos, theta_{i+1} = theta_i * base^(-2/n)
static void rope_table(float *cache, int pos, float base, int n, const float *factors, int nf) {
    float theta_scale = powf(base, -2.0f / (float)n), theta = (float)pos;
    for (int i = 0; i < n; i += 2) {
        float t = factors && i / 2 < nf ? theta / factors[i / 2] : theta;
        cache[i] = cosf(t); cache[i + 1] = sinf(t); theta *= theta_scale;
    }
}
static void rope_apply(float *x, const float *cache, int n) {  // pairs (i, i + n/2)
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        float c = cache[2 * i], s = cache[2 * i + 1], a = x[i], b = x[i + h];
        x[i] = a * c - b * s; x[i + h] = a * s + b * c;
    }
}

// ---------------------------------------------------------------- parallel regions
double prof_mv[8], prof_attn, prof_fwd, prof_head; long prof_passes, prof_slots;
enum { PM_QKV, PM_O, PM_GU, PM_DOWN, PM_PLE, PM_EXP, PM_HEAD };

typedef struct { const pmat *M; int S; const qvec *x; float *const *y; } mv_job;
static void mv_run(void *arg, int tid) {
    mv_job *j = arg; int nt = pool_threads();
    int r0 = (int)((int64_t)j->M->rows * tid / nt), r1 = (int)((int64_t)j->M->rows * (tid + 1) / nt);
    matvec_p4(j->M, r0, r1, j->x, j->S, j->y);
}
static _Atomic unsigned ra_epoch;
static _Atomic(const pmat *) ra_hint;       // the matrix that waiting workers read ahead (see below)
static void matvec(const pmat *M, const qvec *x, int S, float *const *y, int kind) {
    mv_job j = {M, S, x, y};
    double t = now_sec();
    atomic_store_explicit(&ra_hint, NULL, memory_order_relaxed);
    ra_epoch++;
    pool_run(mv_run, &j);
    prof_mv[kind] += now_sec() - t;
}
typedef struct { const fmat *M; int S; const float *const *x; float *const *y; } fv_job;
static void fv_run(void *arg, int tid) {
    fv_job *j = arg; int nt = pool_threads();
    int r0 = (int)((int64_t)j->M->rows * tid / nt), r1 = (int)((int64_t)j->M->rows * (tid + 1) / nt);
    matvec_f(j->M, r0, r1, j->x, j->S, j->y);
}
static void matvec_fl(const fmat *M, float *const *x, int S, float *const *y, int kind) {
    fv_job j = {M, S, (const float *const *)x, y};
    double t = now_sec();
    pool_run(fv_run, &j);
    prof_mv[kind] += now_sec() - t;
}

// ---------------------------------------------------------------- read-ahead
// With one stream most threads have nothing to do during the short per-slot steps between the matrix
// products, and the memory bus idles with them. A thread without slot work therefore reads its share of the
// next matrix (the rows the next matrix product will give it) into its caches, until the slot work of the
// region is done. With one or two slots the per-slot step is too short for a parallel region of its own (a pass
// has hundreds of them, each costs a few microseconds of hand-over): the caller then does the step alone and the
// waiting workers read ahead on their own, a few cache lines between two looks for work (ra_idle).
// TERN_RA=0 turns read-ahead off, TERN_RA=1 keeps it inside parallel regions.
#define RA_MAX ((size_t)1 << 20)       // bytes one thread reads ahead of a matrix at most
static struct { const uint8_t *base; size_t off; unsigned epoch, sink; char pad[40]; } ra_state[64] __attribute__((aligned(64)));
// ra_epoch (declared with the matrix products) is bumped by every one of them: a matrix is read ahead once per use
static void read_ahead(int tid, const pmat *M, _Atomic int *done, int n) {
    int nt = pool_threads();
    int r0 = (int)((int64_t)M->rows * tid / nt), r1 = (int)((int64_t)M->rows * (tid + 1) / nt);
    const uint8_t *base = M->W + (size_t)r0 * M->stride;
    size_t len = (size_t)(r1 - r0) * M->stride;
    if (len > RA_MAX) len = RA_MAX;
    if (ra_state[tid].base != base || ra_state[tid].epoch != ra_epoch) { ra_state[tid].base = base; ra_state[tid].epoch = ra_epoch; ra_state[tid].off = 0; }
    size_t off = ra_state[tid].off; unsigned sum = 0;
    while (off < len && atomic_load_explicit(done, memory_order_relaxed) < n) {
        size_t e = off + 1024 < len ? off + 1024 : len;       // look at the stop condition every 16 cache lines
        for (; off < e; off += 64) sum += base[off];
    }
    ra_state[tid].off = off; ra_state[tid].sink += sum;
}

// one slice of read-ahead by a waiting worker (pool_set_idle)
static void ra_idle(void *arg, int tid) {
    (void)arg;
    const pmat *M = atomic_load_explicit(&ra_hint, memory_order_relaxed);
    if (!M) return;
    const int nt = pool_threads();
    const int r0 = (int)((int64_t)M->rows * tid / nt), r1 = (int)((int64_t)M->rows * (tid + 1) / nt);
    const uint8_t *base = M->W + (size_t)r0 * M->stride;
    size_t len = (size_t)(r1 - r0) * M->stride;
    if (len > RA_MAX) len = RA_MAX;
    const unsigned ep = atomic_load_explicit(&ra_epoch, memory_order_relaxed);
    if (ra_state[tid].base != base || ra_state[tid].epoch != ep) { ra_state[tid].base = base; ra_state[tid].epoch = ep; ra_state[tid].off = 0; }
    size_t off = ra_state[tid].off; unsigned sum = 0;
    if (off >= len) return;
    const size_t e = off + 1024 < len ? off + 1024 : len;
    for (; off < e; off += 64) sum += base[off];
    ra_state[tid].off = off; ra_state[tid].sink += sum;
}

// ---------------------------------------------------------------- attention
// One slot attends over rows [lo, len) of its sequence plus this pass's scratch rows of its ancestors and
// itself, in the cache of layer L->kv_src. The query heads are in t0[s]; the result goes to t1[s].
typedef struct { const layer *L, *KL; const uint16_t *K, *V; size_t kv_rows; const int *rows; int n, n_head; const float *q; float *sc, *out; } att_job;
static void att_scores_part(const att_job *j, int r0, int r1) {
    const layer *L = j->L; const int hd = L->hd, r = j->n_head / L->nkv;
    for (int g = 0; g < L->nkv; g++)
        attn_scores(j->K + (size_t)g * j->kv_rows * hd, hd, j->rows + r0, r1 - r0, j->q + (size_t)g * r * hd, hd, r, j->sc + (size_t)r0 * j->n_head + g * r, j->n_head);
}
static void att_scores_run(void *arg, int tid) {
    att_job *j = arg; int nt = pool_threads();
    att_scores_part(j, (int)((int64_t)j->n * tid / nt), (int)((int64_t)j->n * (tid + 1) / nt));
}
// value mixing: the work units are (value head, block of 16 dims)
static void att_mix_part(const att_job *j, int u0, int u1) {
    const layer *L = j->L; const int hd = L->hd, r = j->n_head / L->nkv, per = hd / 16;
    for (int g = u0 / per; g < L->nkv && g * per < u1; g++) {
        int a = g * per > u0 ? 0 : u0 - g * per, b = (g + 1) * per < u1 ? per : u1 - g * per;
        attn_mix(j->V + (size_t)g * j->kv_rows * hd, hd, j->rows, j->n, hd, r, a * 16, b * 16, j->sc + g * r, j->n_head, j->out + (size_t)g * r * hd);
    }
}
static void att_mix_run(void *arg, int tid) {
    att_job *j = arg; int nt = pool_threads(), units = j->L->nkv * (j->L->hd / 16);
    att_mix_part(j, (int)((int64_t)units * tid / nt), (int)((int64_t)units * (tid + 1) / nt));
}

static void attend(model *m, const layer *L, const slot *sl, int s, float *sc, int *rows, int split) {
    const layer *KL = &m->layers[L->kv_src];
    const int seq = sl[s].seq, len = m->len[seq];
    int lo = L->local && m->pos[s] - m->n_swa + 1 > 0 ? m->pos[s] - m->n_swa + 1 : 0;
    if (lo > len) lo = len;
    int n = 0;
    for (int r = lo; r < len; r++) rows[n++] = r;
    // ancestors oldest first, the slot itself last: the same order as if they had been committed one by one,
    // so that a token evaluated speculatively gets bit-identical results
    int anc[TERN_MAX_SLOTS], na = 0;
    for (int a = s; a >= 0; a = sl[a].parent) anc[na++] = m->row[a];
    while (na > 0) rows[n++] = anc[--na];       // always inside the window: a pass has fewer slots than the window has tokens
    att_job j = {L, KL, KL->K[seq], KL->V[seq], m->kv_rows, rows, n, m->n_head, m->t0[s], sc, m->t1[s]};
    memset(j.out, 0, sizeof(float) * (size_t)L->nq);
    if (split && n >= 64) pool_run(att_scores_run, &j); else att_scores_part(&j, 0, n);
    softmax_heads(sc, n, m->n_head);
    if (split && n >= 64) pool_run(att_mix_run, &j); else att_mix_part(&j, 0, L->nkv * (L->hd / 16));
}

// slots are handed out by thread number, so that a slot's scratch data stays with one core; threads without
// a slot read ahead
typedef struct { model *m; const layer *L; const slot *sl; const int *act; int S; const pmat *next; _Atomic int done; } attn_job;
static void attn_run(void *arg, int tid) {
    attn_job *j = arg; model *m = j->m; int nt = pool_threads();
    float *sc = m->scores + (size_t)tid * m->n_head * m->kv_rows;
    int *rows = m->vis + (size_t)tid * m->kv_rows;
    for (int i = tid; i < j->S; i += nt) { attend(m, j->L, j->sl, j->act[i], sc, rows, 0); atomic_fetch_add(&j->done, 1); }
    if (j->next) read_ahead(tid, j->next, &j->done, j->S);
}

// ---------------------------------------------------------------- per-slot steps between the matrix products
enum { ST_EMBED, ST_PLE_IN, ST_PRE_ATTN, ST_QKV, ST_POST_ATTN, ST_PRE_FFN, ST_FFN_MID, ST_POST_FFN, ST_PLE_PRE, ST_PLE_MID, ST_PLE_POST, ST_FINAL, ST_EXP_MID, ST_EXP_SUM };

// Routing for the experts. The slot step leaves the router's input in rin[s]; the 128 logits (one row of E floats
// each) are computed on all threads, the choice of the K best follows.
static void router_pick(model *m, int s) {
    const int NX = m->n_expert, K = m->n_exp_used;
    float *lg = m->rlg[s], mx = -INFINITY, sum = 0;
    for (int e = 0; e < NX; e++) if (lg[e] > mx) mx = lg[e];
    for (int e = 0; e < NX; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
    float ws = 0;
    for (int j = 0; j < K; j++) {       // the K most likely experts, most likely first
        int best = 0;
        for (int e = 1; e < NX; e++) if (lg[e] > lg[best]) best = e;
        m->ex_id[s][j] = best; m->ex_w[s][j] = lg[best] / sum; ws += m->ex_w[s][j]; lg[best] = -1.0f;
    }
    if (ws < 6.103515625e-5f) ws = 6.103515625e-5f;
    for (int j = 0; j < K; j++) m->ex_w[s][j] /= ws;
}
#define RT_PIECE 8          // experts per work unit
typedef struct { model *m; const layer *L; const int *act; int n; _Atomic int nextu; } rt_job;
static void rt_run(void *arg, int tid) {
    (void)tid;
    rt_job *j = arg; model *m = j->m; const int E = m->n_embd, per = (m->n_expert + RT_PIECE - 1) / RT_PIECE;
    for (;;) {
        int u = atomic_fetch_add(&j->nextu, 1);
        if (u >= j->n * per) break;
        int s = j->act[u / per], e0 = (u % per) * RT_PIECE, e1 = e0 + RT_PIECE < m->n_expert ? e0 + RT_PIECE : m->n_expert;
        for (int e = e0; e < e1; e++) m->rlg[s][e] = dot_f32(j->L->router + (size_t)e * E, m->rin[s], E);
    }
}
static void router_all(model *m, const layer *L, const int *act, int n) {
    rt_job j = {.m = m, .L = L, .act = act, .n = n};
    atomic_store(&j.nextu, 0);
    if (pool_threads() > 1) pool_run(rt_run, &j); else rt_run(&j, 0);
    for (int i = 0; i < n; i++) router_pick(m, act[i]);
}

static void slot_step(model *m, const layer *L, const slot *sl, int s, int step, float *const *hidden) {
    const int E = m->n_embd, NL = m->n_layer, PL = m->pl;
    switch (step) {
    case ST_EMBED: {        // token embedding, rotary tables, per-layer token inputs
        dequant_row(m->embd->type, m->embd->data + (size_t)sl[s].token * quant_row_bytes(m->embd->type, E), E, m->x[s]);
        const float emb_scale = sqrtf((float)E);
        for (int i = 0; i < E; i++) m->x[s][i] *= emb_scale;
        if (m->hs) rope_table(m->rope[s], m->pos[s], m->rope_base_swa, m->hs, NULL, 0);
        if (m->hf) rope_table(m->rope[s] + m->hs, m->pos[s], m->rope_base, m->hf, m->rope_freqs, m->n_rope_freqs);
        if (PL) {
            dequant_row(m->ple->type, m->ple->data + (size_t)sl[s].token * quant_row_bytes(m->ple->type, PL * NL), PL * NL, m->pli[s]);
            const float sc = sqrtf((float)PL) / sqrtf(2.0f);
            for (int i = 0; i < PL * NL; i++) m->pli[s][i] *= sc;
        }
    } break;
    case ST_PLE_IN: {       // t0 = projection of the embedding to NL * PL: normalize per layer, add the table rows
        const float isq2 = 1.0f / sqrtf(2.0f), eps = m->eps * (float)E;
        for (int il = 0; il < NL; il++) {
            float *pp = m->t0[s] + il * PL, *dst = m->pli[s] + il * PL;
            float sc = 1.0f / sqrtf(dot_f32(pp, pp, PL) / (float)PL + eps);
            for (int i = 0; i < PL; i++) dst[i] += pp[i] * sc * m->ple_proj_norm[i] * isq2;
        }
    } break;
    case ST_PRE_ATTN:
        rmsnorm(m->x[s], L->attn_norm, E, m->eps, m->t1[s]);
        quantize_q8(m->t1[s], E, &m->qv[s]);
        break;
    case ST_QKV: {          // query norms and rotation; key and value into the cache
        const int hd = L->hd, nh = m->n_head;
        const float *rc = m->rope[s] + (L->local ? 0 : m->hs);       // the full-attention table follows the sliding-window one
        float *q = m->t0[s];
        for (int h = 0; h < nh; h++) { rmsnorm(q + h * hd, L->q_norm, hd, m->eps, q + h * hd); rope_apply(q + h * hd, rc, hd); }
        if (L->own_kv) {
            float *k = q + L->nq, *v = L->has_v ? k + L->nkv * hd : k, kn[512], vn[512];
            int seq = sl[s].seq;
            for (int g = 0; g < L->nkv; g++) {
                size_t at = ((size_t)g * m->kv_rows + (size_t)m->row[s]) * (size_t)hd;
                rmsnorm(v + g * hd, NULL, hd, m->eps, vn);          // v may be k: the values are normalized without weights
                rmsnorm(k + g * hd, L->k_norm, hd, m->eps, kn);
                rope_apply(kn, rc, hd);
                f32_to_f16_row(kn, L->K[seq] + at, hd);
                f32_to_f16_row(vn, L->V[seq] + at, hd);
            }
        }
    } break;
    case ST_POST_ATTN:
        quantize_q8(m->t1[s], L->nq, &m->qv[s]);
        break;
    case ST_PRE_FFN:        // residual, then the feed-forward input (and the experts' input and routing)
        rmsnorm(m->t0[s], L->post_attn_norm, E, m->eps, m->t0[s]);
        for (int i = 0; i < E; i++) m->sa[s][i] = m->t0[s][i] + m->x[s][i];
        if (L->moe) {
            rmsnorm(m->sa[s], NULL, E, m->eps, m->rin[s]);             // the router's input (router_all follows this step)
            const float rs = 1.0f / sqrtf((float)E);
            for (int i = 0; i < E; i++) m->rin[s][i] *= rs * L->router_scale[i];
            rmsnorm(m->sa[s], L->pre_ffn_norm_2, E, m->eps, m->t1[s]);
            quantize_q8(m->t1[s], E, &m->qx[s]);
        }
        rmsnorm(m->sa[s], L->ffn_norm, E, m->eps, m->t1[s]);
        quantize_q8(m->t1[s], E, &m->qv[s]);
        break;
    case ST_FFN_MID:        // gelu(gate) * up
        gelu_mul(m->t0[s], m->t0[s] + L->n_ff, m->t1[s], L->n_ff);
        quantize_q8(m->t1[s], L->n_ff, &m->qv[s]);
        break;
    case ST_EXP_MID:        // the same for each of the slot's experts
        for (int j = 0; j < m->n_exp_used; j++) {
            float *gu = m->ex_gu[s] + (size_t)j * 2 * m->ffe;
            gelu_mul(gu, gu + m->ffe, gu, m->ffe);
            quantize_q8(gu, m->ffe, &m->ex_h[s * m->n_exp_used + j]);
        }
        break;
    case ST_EXP_SUM: {      // weighted sum of the experts' outputs
        float *mo = m->mo[s];
        memset(mo, 0, sizeof(float) * (size_t)E);
        for (int j = 0; j < m->n_exp_used; j++) {
            const float w = m->ex_w[s][j] * (L->ex_scale ? L->ex_scale[m->ex_id[s][j]] : 1.0f);
            const float *o = m->ex_out[s] + (size_t)j * E;
            for (int i = 0; i < E; i++) mo[i] += w * o[i];
        }
    } break;
    case ST_POST_FFN:       // residual
        if (L->moe) {
            rmsnorm(m->t0[s], L->post_ffn_norm_1, E, m->eps, m->t0[s]);
            rmsnorm(m->mo[s], L->post_ffn_norm_2, E, m->eps, m->mo[s]);
            for (int i = 0; i < E; i++) m->t0[s][i] += m->mo[s][i];
        }
        rmsnorm(m->t0[s], L->post_ffn_norm, E, m->eps, m->t0[s]);
        for (int i = 0; i < E; i++) m->x[s][i] = m->t0[s][i] + m->sa[s][i];
        if (!PL) for (int i = 0; i < E; i++) m->x[s][i] *= L->out_scale;
        break;
    case ST_PLE_PRE:
        quantize_q8(m->x[s], E, &m->qv[s]);
        break;
    case ST_PLE_MID:        // gelu(gate) * per-layer input
        gelu_mul(m->t0[s], m->pli[s] + (size_t)(L - m->layers) * PL, m->t1[s], PL);
        if (L->ple_packed) quantize_q8(m->t1[s], PL, &m->qv[s]);
        break;
    case ST_PLE_POST:
        rmsnorm(m->t0[s], L->ple_norm, E, m->eps, m->t0[s]);
        for (int i = 0; i < E; i++) m->x[s][i] = (m->x[s][i] + m->t0[s][i]) * L->out_scale;
        break;
    case ST_FINAL:
        if (hidden && hidden[s]) rmsnorm(m->x[s], m->out_norm, E, m->eps, hidden[s]);
        break;
    }
}

typedef struct { model *m; const layer *L; const slot *sl; const int *act; int n, step; float *const *hidden; const pmat *next; _Atomic int done; } slot_job;
static void slot_run(void *arg, int tid) {
    slot_job *j = arg; int nt = pool_threads();
    for (int i = tid; i < j->n; i += nt) { slot_step(j->m, j->L, j->sl, j->act[i], j->step, j->hidden); atomic_fetch_add(&j->done, 1); }
    if (j->next) read_ahead(tid, j->next, &j->done, j->n);
}
// The per-slot work is small for one slot but adds up for a batch: it runs on all threads when there are
// several. `next`: the matrix of the following matrix product, read ahead by the threads without a slot.
static void slots_do(model *m, const layer *L, const slot *sl, const int *act, int n, int step, float *const *hidden, const pmat *next) {
    if (n == 0) return;
    if (!ra_on) next = NULL;
    if (pool_threads() == 1 || (n < 3 && (!next || ra_idle_mode))) {
        if (ra_idle_mode && next) atomic_store_explicit(&ra_hint, next, memory_order_relaxed);
        for (int i = 0; i < n; i++) slot_step(m, L, sl, act[i], step, hidden);
        return;
    }
    slot_job j = {.m = m, .L = L, .sl = sl, .act = act, .n = n, .step = step, .hidden = hidden, .next = next};
    atomic_store(&j.done, 0);
    pool_run(slot_run, &j);
}

// The feed-forward middle step (gelu(gate) * up and the quantization of the result) is the longest per-slot
// step: 21,504 values per layer in the 31B, one after the other 4 % of a pass. It is cut into pieces that all
// threads take from a counter. A piece is a multiple of 512 values (the kernel's group of 16 blocks), so every
// value goes through the same instructions as in one piece.
#define MID_PIECE 1024
typedef struct { model *m; const layer *L; const int *act; int per, total; const pmat *next; _Atomic int nextu, done; } mid_job;
static void mid_run(void *arg, int tid) {
    mid_job *j = arg; model *m = j->m; const int F = j->L->n_ff;
    for (;;) {
        int u = atomic_fetch_add(&j->nextu, 1);
        if (u >= j->total) break;
        const int s = j->act[u / j->per], o = (u % j->per) * MID_PIECE, len = F - o < MID_PIECE ? F - o : MID_PIECE;
        gelu_mul(m->t0[s] + o, m->t0[s] + F + o, m->t1[s] + o, len);
        const qvec *v = &m->qv[s]; const int b = o / QB;
        qvec part = {v->q + o, v->p + o, v->d + b, v->k + b, v->c + b, v->ha + b, v->hb + b};
        quantize_q8(m->t1[s] + o, len, &part);
        atomic_fetch_add(&j->done, 1);
    }
    if (j->next) read_ahead(tid, j->next, &j->done, j->total);
}
static void ffn_mid(model *m, const layer *L, const slot *sl, const int *act, int n) {
    const int F = L->n_ff;
    if (pool_threads() == 1 || F % 512 || F < 4 * MID_PIECE) { slots_do(m, L, sl, act, n, ST_FFN_MID, NULL, &L->down); return; }
    mid_job j = {.m = m, .L = L, .act = act, .per = (F + MID_PIECE - 1) / MID_PIECE, .next = ra_on ? &L->down : NULL};
    j.total = n * j.per; atomic_store(&j.nextu, 0); atomic_store(&j.done, 0);
    pool_run(mid_run, &j);
}

// ---------------------------------------------------------------- experts
// The slots of a pass choose their experts independently. Each distinct expert is evaluated once for all the
// slots that chose it (one sweep over its rows); the work units of a parallel region are (expert, row range).
typedef struct { model *m; const int *act; int n; _Atomic int nextu; } xmid_job;
static void xmid_run(void *arg, int tid) {      // gelu(gate) * up for one expert of one slot
    (void)tid;
    xmid_job *j = arg; model *m = j->m; const int K = m->n_exp_used;
    for (;;) {
        int u = atomic_fetch_add(&j->nextu, 1);
        if (u >= j->n * K) break;
        const int s = j->act[u / K], e = u % K;
        float *gu = m->ex_gu[s] + (size_t)e * 2 * m->ffe;
        gelu_mul(gu, gu + m->ffe, gu, m->ffe);
        quantize_q8(gu, m->ffe, &m->ex_h[s * K + e]);
    }
}
typedef struct { int e, n; int slot[TERN_MAX_SLOTS], j[TERN_MAX_SLOTS]; } exp_use;
typedef struct { model *m; const layer *L; exp_use *use; int n_use, phase, parts; _Atomic int next; } exp_job;
static void exp_run(void *arg, int tid) {
    exp_job *J = arg; model *m = J->m; const layer *L = J->L; const int E = m->n_embd, ffe = m->ffe, K = m->n_exp_used;
    for (;;) {
        int u = atomic_fetch_add(&J->next, 1);
        if (u >= J->n_use * J->parts) break;
        const exp_use *x = &J->use[u / J->parts]; int part = u % J->parts;
        qvec in[TERN_MAX_SLOTS]; float *out[TERN_MAX_SLOTS];
        if (J->phase == 0) {        // gate and up: rows e * 2ffe .. of ex_gu
            const int rows = 2 * ffe, r0 = (int)((int64_t)rows * part / J->parts), r1 = (int)((int64_t)rows * (part + 1) / J->parts);
            pmat M = L->ex_gu; M.W += (size_t)x->e * rows * M.stride; M.rows = rows;
            for (int i = 0; i < x->n; i++) { in[i] = m->qx[x->slot[i]]; out[i] = m->ex_gu[x->slot[i]] + (size_t)x->j[i] * rows; }
            matvec_p4(&M, r0, r1, in, x->n, out);
        } else {                    // down: rows e * E .. of ex_down
            const int r0 = (int)((int64_t)E * part / J->parts), r1 = (int)((int64_t)E * (part + 1) / J->parts);
            pmat M = L->ex_down; M.W += (size_t)x->e * E * M.stride; M.rows = E;
            for (int i = 0; i < x->n; i++) { in[i] = m->ex_h[x->slot[i] * K + x->j[i]]; out[i] = m->ex_out[x->slot[i]] + (size_t)x->j[i] * E; }
            matvec_p4(&M, r0, r1, in, x->n, out);
        }
    }
}

static void experts(model *m, const layer *L, const slot *sl, const int *act, int n) {
    const int K = m->n_exp_used;
    exp_use *use = malloc(sizeof(exp_use) * (size_t)(n * K)); int n_use = 0;
    int *where = malloc(sizeof(int) * (size_t)m->n_expert);
    for (int e = 0; e < m->n_expert; e++) where[e] = -1;
    for (int i = 0; i < n; i++) for (int j = 0; j < K; j++) {
        int e = m->ex_id[act[i]][j];
        if (where[e] < 0) { where[e] = n_use; use[n_use].e = e; use[n_use].n = 0; n_use++; }
        exp_use *x = &use[where[e]];
        x->slot[x->n] = act[i]; x->j[x->n] = j; x->n++;
    }
    // split every expert's rows so that all threads have work even when one slot uses only K experts
    int parts = (2 * pool_threads() + n_use - 1) / n_use;
    if (parts < 1) parts = 1;
    if (parts > 16) parts = 16;
    double t = now_sec();
    exp_job J = {.m = m, .L = L, .use = use, .n_use = n_use, .phase = 0, .parts = parts};
    atomic_store(&J.next, 0);
    pool_run(exp_run, &J);
    if (pool_threads() > 1) { xmid_job xj = {.m = m, .act = act, .n = n}; atomic_store(&xj.nextu, 0); pool_run(xmid_run, &xj); }
    else slots_do(m, L, sl, act, n, ST_EXP_MID, NULL, NULL);
    J.phase = 1; atomic_store(&J.next, 0);
    pool_run(exp_run, &J);
    slots_do(m, L, sl, act, n, ST_EXP_SUM, NULL, NULL);
    prof_mv[PM_EXP] += now_sec() - t;
    free(use); free(where);
}

// ---------------------------------------------------------------- forward
void model_forward(model *m, const slot *sl, int n, float *const *hidden) {
    double t_fwd = now_sec();
    int all[TERN_MAX_SLOTS], out[TERN_MAX_SLOTS], n_out = 0;
    for (int s = 0; s < n; s++) {
        int seq = sl[s].seq, k = 0;
        for (int a = 0; a < s; a++) k += sl[a].seq == seq;
        m->row[s] = m->len[seq] + k;
        m->pos[s] = sl[s].parent < 0 ? m->len[seq] : m->pos[sl[s].parent] + 1;
        all[s] = s;
        if (hidden && hidden[s]) out[n_out++] = s;
    }
    qvec qa[TERN_MAX_SLOTS]; float *ya[TERN_MAX_SLOTS], *xa[TERN_MAX_SLOTS];
    slots_do(m, NULL, sl, all, n, ST_EMBED, NULL, m->pl ? NULL : &m->layers[0].qkv);
    if (m->pl) {
        for (int s = 0; s < n; s++) { xa[s] = m->x[s]; ya[s] = m->t0[s]; }
        matvec_fl(&m->ple_proj, xa, n, ya, PM_PLE);
        slots_do(m, NULL, sl, all, n, ST_PLE_IN, NULL, &m->layers[0].qkv);
    }
    for (int il = 0; il < m->n_layer; il++) {
        const layer *L = &m->layers[il];
        // layers without their own cache only matter for the slots whose output is wanted
        const int *act = il < m->kv0 ? all : out; const int S = il < m->kv0 ? n : n_out;
        if (S == 0) break;
        const pmat *after = il + 1 < m->n_layer ? &m->layers[il + 1].qkv : NULL;     // the first matrix after this layer
#define GATHER() for (int i = 0; i < S; i++) { qa[i] = m->qv[act[i]]; ya[i] = m->t0[act[i]]; }
        slots_do(m, L, sl, act, S, ST_PRE_ATTN, NULL, &L->qkv);
        GATHER(); matvec(&L->qkv, qa, S, ya, PM_QKV);
        slots_do(m, L, sl, act, S, ST_QKV, NULL, &L->o);
        {
            double t = now_sec();
            int vis = m->pos[act[0]] + 1;
            if (L->local && vis > m->n_swa) vis = m->n_swa;
            if (S == 1 && (vis >= 64 || !ra_on || ra_idle_mode || pool_threads() == 1)) {      // a long context: scores and mixing on all threads
                if (ra_on && ra_idle_mode) atomic_store_explicit(&ra_hint, &L->o, memory_order_relaxed);
                attend(m, L, sl, act[0], m->scores, m->vis, 1);
            }
            else { attn_job aj = {.m = m, .L = L, .sl = sl, .act = act, .S = S, .next = ra_on ? &L->o : NULL}; atomic_store(&aj.done, 0); pool_run(attn_run, &aj); }
            prof_attn += now_sec() - t;
        }
        slots_do(m, L, sl, act, S, ST_POST_ATTN, NULL, &L->o);
        GATHER(); matvec(&L->o, qa, S, ya, PM_O);
        slots_do(m, L, sl, act, S, ST_PRE_FFN, NULL, &L->gu);
        if (L->moe) router_all(m, L, act, S);
        GATHER(); matvec(&L->gu, qa, S, ya, PM_GU);
        ffn_mid(m, L, sl, act, S);
        GATHER(); matvec(&L->down, qa, S, ya, PM_DOWN);
        if (L->moe) experts(m, L, sl, act, S);
        slots_do(m, L, sl, act, S, ST_POST_FFN, NULL, m->pl ? (L->ple_packed ? &L->pg : NULL) : after);
        if (m->pl) {
            if (L->ple_packed) {
                slots_do(m, L, sl, act, S, ST_PLE_PRE, NULL, &L->pg);
                GATHER(); matvec(&L->pg, qa, S, ya, PM_PLE);
                slots_do(m, L, sl, act, S, ST_PLE_MID, NULL, &L->pp);
                GATHER(); matvec(&L->pp, qa, S, ya, PM_PLE);
            } else {
                for (int i = 0; i < S; i++) { xa[i] = m->x[act[i]]; ya[i] = m->t0[act[i]]; }
                matvec_fl(&L->fg, xa, S, ya, PM_PLE);
                slots_do(m, L, sl, act, S, ST_PLE_MID, NULL, NULL);
                for (int i = 0; i < S; i++) xa[i] = m->t1[act[i]];
                matvec_fl(&L->fp, xa, S, ya, PM_PLE);
            }
            slots_do(m, L, sl, act, S, ST_PLE_POST, NULL, after);
        }
#undef GATHER
    }
    slots_do(m, NULL, sl, out, n_out, ST_FINAL, hidden, NULL);
    prof_fwd += now_sec() - t_fwd; prof_passes++; prof_slots += n;
}

void model_print_profile(void) {
    if (!prof_passes) return;
    double mv = 0; for (int k = 0; k < 6; k++) mv += prof_mv[k];
    fprintf(stderr, "body: %ld passes, %.1f slots per pass | per pass (ms): total %.3f = attention matrices %.3f + feed-forward %.3f + experts %.3f + per-layer inputs %.3f + attention %.3f + other %.3f | output layer %.3f\n",
            prof_passes, (double)prof_slots / prof_passes, 1e3 * prof_fwd / prof_passes, 1e3 * (prof_mv[PM_QKV] + prof_mv[PM_O]) / prof_passes,
            1e3 * (prof_mv[PM_GU] + prof_mv[PM_DOWN]) / prof_passes, 1e3 * prof_mv[PM_EXP] / prof_passes, 1e3 * prof_mv[PM_PLE] / prof_passes,
            1e3 * prof_attn / prof_passes, 1e3 * (prof_fwd - mv - prof_attn) / prof_passes, 1e3 * prof_head / prof_passes);
}

void model_commit(model *m, int seq, const int *slot_idx, int n) {
    int len = m->len[seq];
    for (int j = 0; j < n; j++) {
        size_t src = (size_t)m->row[slot_idx[j]], dst = (size_t)(len + j);   // src >= dst always, and later sources are never overwritten
        if (src == dst) continue;
        for (int il = 0; il < m->kv0; il++) {
            layer *L = &m->layers[il]; size_t w = (size_t)L->hd;
            for (int g = 0; g < L->nkv; g++) {
                size_t base = (size_t)g * m->kv_rows;
                memcpy(L->K[seq] + (base + dst) * w, L->K[seq] + (base + src) * w, w * sizeof(uint16_t));
                memcpy(L->V[seq] + (base + dst) * w, L->V[seq] + (base + src) * w, w * sizeof(uint16_t));
            }
        }
    }
    m->len[seq] = len + n;
}

// ---------------------------------------------------------------- exact output layer
// All rows of an output matrix for n hidden states in one sweep; the best token of each state goes to out.
typedef struct { const pmat *M; const qvec *x; int n; float **y; int best[64][TERN_MAX_SLOTS]; } head_job;
static void head_run(void *arg, int tid) {
    head_job *j = arg; int nt = pool_threads(), V = j->M->rows;
    int r0 = (int)((int64_t)V * tid / nt), r1 = (int)((int64_t)V * (tid + 1) / nt);
    matvec_p4(j->M, r0, r1, j->x, j->n, j->y);
    for (int s = 0; s < j->n; s++) {
        const float *lg = j->y[s]; int best = r0;
        for (int v = r0 + 1; v < r1; v++) if (lg[v] > lg[best]) best = v;
        j->best[tid][s] = best;
    }
}
static void head_sweep(const pmat *M, const qvec *x, int n, float **y, int *out) {
    head_job *j = malloc(sizeof *j);
    j->M = M; j->x = x; j->n = n; j->y = y;
    pool_run(head_run, j);
    for (int s = 0; s < n; s++) {
        int best = j->best[0][s];
        for (int th = 1; th < pool_threads(); th++) if (y[s][j->best[th][s]] > y[s][best]) best = j->best[th][s];
        out[s] = best;
    }
    free(j);
}

void model_head_exact_batch(model *m, float *const *hidden, int n, int *out) {
    double t = now_sec();
    if (!m->logits) m->logits = xalloc(sizeof(float) * (size_t)TERN_MAX_SLOTS * m->n_vocab);
    float *y[TERN_MAX_SLOTS];
    for (int s = 0; s < n; s++) { quantize_q8(hidden[s], m->n_embd, &m->qv[s]); y[s] = m->logits + (size_t)s * m->n_vocab; }
    head_sweep(&m->head, m->qv, n, y, out);
    prof_head += now_sec() - t;
}

// one state; with logits != NULL all n_vocab logits (soft-capped as the model defines) are stored there
int model_head_exact(model *m, const float *hidden, float *logits) {
    float *h = (float *)hidden; int best;
    model_head_exact_batch(m, &h, 1, &best);
    if (logits) {
        memcpy(logits, m->logits, sizeof(float) * (size_t)m->n_vocab);
        if (m->softcap > 0) for (int v = 0; v < m->n_vocab; v++) logits[v] = m->softcap * tanhf(logits[v] / m->softcap);
    }
    return best;
}

// the packed output matrix (n_vocab rows) and the model's logit soft cap (0: none), for the fast output layer
void model_output_matrix(const model *m, pmat *M, float *softcap) { *M = m->head; *softcap = m->softcap; }
void model_output_row(const model *m, int token, float *out) {
    dequant_row(m->embd->type, m->embd->data + (size_t)token * quant_row_bytes(m->embd->type, m->n_embd), m->n_embd, out);
}

// Diagnostic: only the weight matrices of the body, in pass order, with dummy inputs. Returns seconds per pass.
double model_bench_sweep(model *m, int S, int reps) {
    float *y[TERN_MAX_SLOTS]; qvec qa[TERN_MAX_SLOTS];
    int big = m->max_ff > m->max_nq ? m->max_ff : m->max_nq;
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < big; i++) m->t1[s][i] = (float)((i * 37 + s * 11) % 97) - 48.0f;
        quantize_q8(m->t1[s], big, &m->qv[s]);
        y[s] = m->t0[s]; qa[s] = m->qv[s];
    }
    double t = now_sec();
    for (int r = 0; r < reps; r++) for (int il = 0; il < m->n_layer; il++) {
        const layer *L = &m->layers[il];
        matvec(&L->qkv, qa, S, y, PM_QKV); matvec(&L->o, qa, S, y, PM_O); matvec(&L->gu, qa, S, y, PM_GU); matvec(&L->down, qa, S, y, PM_DOWN);
    }
    return (now_sec() - t) / reps;
}

// ---------------------------------------------------------------- drafter ("assistant")
// Gemma 4 ships small companion models that guess the next tokens (llama.cpp: src/models/gemma4-assistant.cpp).
// A drafter has a few layers of its own but no keys and values: it attends over the main model's cache
// (the last sliding-window layer and the last full-attention layer). One step takes a token and the hidden
// state that predicted it, and returns the next token and an estimate of the hidden state after it, which
// feeds the following step. The main model then checks all guesses in one pass.
struct assistant {
    model *tgt;
    int n_embd, n_layer, n_head, n_vocab, hs, hf, max_nq, max_ff;
    float eps, rope_base, rope_base_swa; const float *rope_freqs; int n_rope_freqs;
    layer *layers; const float *out_norm;
    pmat pre, post, head;
    uint8_t *arena; size_t arena_size;
    float *xh, *x, *sa, *t0, *t1, *rope, *logits, *scores; int *vis; qvec qv;
    double t_total; long steps;
};

static void asst_build(builder *b, assistant *a) {
    const gguf *g = b->g;
    for (int il = 0; il < a->n_layer; il++) {
        layer *L = &a->layers[il];
        L->qkv = packed(b, need(g, "blk.%d.attn_q.weight", il), NULL, NULL);
        L->o = packed(b, need(g, "blk.%d.attn_output.weight", il), NULL, NULL);
        L->gu = packed(b, need(g, "blk.%d.ffn_gate.weight", il), need(g, "blk.%d.ffn_up.weight", il), NULL);
        L->down = packed(b, need(g, "blk.%d.ffn_down.weight", il), NULL, NULL);
    }
    a->pre = packed(b, need(g, "nextn.pre_projection.weight", 0), NULL, NULL);
    a->post = packed(b, need(g, "nextn.post_projection.weight", 0), NULL, NULL);
    a->head = packed(b, need(g, "token_embd.weight", 0), NULL, NULL);
}

assistant *assistant_load(const gguf *g, model *tgt) {
    uint64_t al = 0; const char *arch = gguf_get_str(g, "general.architecture", &al);
    if (!arch || al != 16 || memcmp(arch, "gemma4-assistant", 16)) { fprintf(stderr, "not a Gemma 4 drafter file (general.architecture)\n"); return NULL; }
    assistant *a = calloc(1, sizeof *a);
    a->tgt = tgt;
    key_prefix = "gemma4-assistant";
    const int E = a->n_embd = key_u32(g, "embedding_length", 0), NL = a->n_layer = key_u32(g, "block_count", 0);
    a->n_head = key_u32(g, "attention.head_count", 0);
    a->eps = key_f32(g, "attention.layer_norm_rms_epsilon", 1e-6f);
    a->rope_base = key_f32(g, "rope.freq_base", 1000000.0f); a->rope_base_swa = key_f32(g, "rope.freq_base_swa", 10000.0f);
    const int hd_full = key_u32(g, "attention.key_length", 256), hd_swa = key_u32(g, "attention.key_length_swa", hd_full);
    const int e_out = key_u32(g, "embedding_length_out", 0);
    int ok = E && NL && a->n_head && E % QB == 0 && e_out == tgt->n_embd;
    a->layers = calloc((size_t)(NL > 0 ? NL : 1), sizeof(layer));
    for (int il = 0; ok && il < NL; il++) {
        layer *L = &a->layers[il];
        L->local = key_layer(g, "attention.sliding_window_pattern", il, 0);
        L->hd = L->local ? hd_swa : hd_full;
        L->nkv = key_layer(g, "attention.head_count_kv", il, 1);
        L->nq = a->n_head * L->hd;
        L->n_ff = key_layer(g, "feed_forward_length", il, 0);
        // the main model's last layer of the same kind (or the layer it shares keys and values with)
        int tl = tgt->n_layer - (L->local ? 2 : 1);
        if (tl < 0) { ok = 0; break; }
        tl = tgt->layers[tl].kv_src;
        const layer *T = &tgt->layers[tl];
        L->kv_src = tl;
        if (T->hd != L->hd || T->nkv != L->nkv || T->local != L->local || a->n_head % L->nkv || L->n_ff % QB) ok = 0;
        if (L->local) a->hs = L->hd; else a->hf = L->hd;
        if (L->nq > a->max_nq) a->max_nq = L->nq;
        if (L->n_ff > a->max_ff) a->max_ff = L->n_ff;
    }
    if (!ok) { fprintf(stderr, "this drafter does not fit the model (sizes or layer pattern)\n"); key_prefix = "gemma4"; return NULL; }
    for (int il = 0; il < NL; il++) {
        layer *L = &a->layers[il];
        L->attn_norm = f32_of(g, "blk.%d.attn_norm.weight", il, E);
        L->q_norm = f32_of(g, "blk.%d.attn_q_norm.weight", il, L->hd);
        L->post_attn_norm = f32_of(g, "blk.%d.post_attention_norm.weight", il, E);
        L->ffn_norm = f32_of(g, "blk.%d.ffn_norm.weight", il, E);
        L->post_ffn_norm = f32_of(g, "blk.%d.post_ffw_norm.weight", il, E);
        const gguf_tensor *os = find(g, "blk.%d.layer_output_scale.weight", il);
        L->out_scale = os ? *(const float *)os->data : 1.0f;
    }
    a->out_norm = f32_of(g, "output_norm.weight", 0, E);
    const gguf_tensor *rf = gguf_find_tensor(g, "rope_freqs.weight");
    if (rf && rf->type == GT_F32) { a->rope_freqs = (const float *)rf->data; a->n_rope_freqs = (int)rf->dims[0]; }
    key_prefix = "gemma4";

    builder b = {.g = g, .dry = 1};
    asst_build(&b, a);
    a->arena_size = b.off;
    if (!(a->arena = arena_alloc(a->arena_size))) return NULL;
    b.dry = 0; b.off = 0; b.arena = a->arena;
    asst_build(&b, a);
    run_packing(&b, NULL);
    a->n_vocab = a->head.rows;
    if (a->pre.cols != 2 * tgt->n_embd || a->pre.rows != E || a->post.rows != tgt->n_embd || a->post.cols != E || a->head.cols != E || a->n_vocab != tgt->n_vocab) {
        fprintf(stderr, "this drafter does not fit the model (projection sizes)\n"); return NULL;
    }
    int EB = tgt->n_embd, big0 = a->max_nq, big1 = a->max_ff;
    if (2 * a->max_ff > big0) big0 = 2 * a->max_ff;
    if (E > big0) big0 = E;
    if (a->max_nq > big1) big1 = a->max_nq;
    if (2 * EB > big1) big1 = 2 * EB;
    a->xh = xalloc(sizeof(float) * (size_t)2 * EB); a->x = xalloc(sizeof(float) * (size_t)E); a->sa = xalloc(sizeof(float) * (size_t)E);
    a->t0 = xalloc(sizeof(float) * (size_t)big0); a->t1 = xalloc(sizeof(float) * (size_t)big1);
    qvec_alloc(&a->qv, big1);
    a->rope = xalloc(sizeof(float) * (size_t)(a->hs + a->hf));
    a->logits = xalloc(sizeof(float) * (size_t)a->n_vocab);
    a->scores = xalloc((size_t)pool_threads() * a->n_head * tgt->kv_rows * sizeof(float));
    a->vis = xalloc(tgt->kv_rows * sizeof(int));
    return a;
}

// One draft step for a sequence. `token` follows the committed tokens of the sequence; h (the main model's
// width) is the main model's final hidden state that predicted it, or the previous step's h_out.
// Returns the drafted next token; h_out is the state to pass to the next step; *p_top (if wanted) is the
// probability of the drafted token among the ten most likely.
int assistant_step(assistant *a, int seq, int token, const float *h, float *h_out, float *p_top) {
    double t_start = now_sec();
    model *m = a->tgt; const int EB = m->n_embd, E = a->n_embd, len = m->len[seq], pos = len;
    dequant_row(m->embd->type, m->embd->data + (size_t)token * quant_row_bytes(m->embd->type, EB), EB, a->xh);
    const float emb_scale = sqrtf((float)EB);
    for (int i = 0; i < EB; i++) a->xh[i] *= emb_scale;
    memcpy(a->xh + EB, h, sizeof(float) * (size_t)EB);
    quantize_q8(a->xh, 2 * EB, &a->qv);
    float *y = a->x;
    matvec(&a->pre, &a->qv, 1, &y, PM_PLE);
    if (a->hs) rope_table(a->rope, pos, a->rope_base_swa, a->hs, NULL, 0);
    if (a->hf) rope_table(a->rope + a->hs, pos, a->rope_base, a->hf, a->rope_freqs, a->n_rope_freqs);
    y = a->t0;
    for (int il = 0; il < a->n_layer; il++) {
        const layer *L = &a->layers[il], *TL = &m->layers[L->kv_src];
        const int hd = L->hd, F = L->n_ff;
        rmsnorm(a->x, L->attn_norm, E, a->eps, a->t1);
        quantize_q8(a->t1, E, &a->qv);
        matvec(&L->qkv, &a->qv, 1, &y, PM_QKV);
        const float *rc = a->rope + (L->local ? 0 : a->hs);
        for (int hh = 0; hh < a->n_head; hh++) { rmsnorm(a->t0 + hh * hd, L->q_norm, hd, a->eps, a->t0 + hh * hd); rope_apply(a->t0 + hh * hd, rc, hd); }
        int lo = L->local && pos - m->n_swa + 1 > 0 ? pos - m->n_swa + 1 : 0, n = 0;
        for (int r = lo; r < len; r++) a->vis[n++] = r;
        att_job j = {L, TL, TL->K[seq], TL->V[seq], m->kv_rows, a->vis, n, a->n_head, a->t0, a->scores, a->t1};
        memset(a->t1, 0, sizeof(float) * (size_t)L->nq);
        if (n >= 64) pool_run(att_scores_run, &j); else att_scores_part(&j, 0, n);
        softmax_heads(a->scores, n, a->n_head);
        if (n >= 64) pool_run(att_mix_run, &j); else att_mix_part(&j, 0, L->nkv * (hd / 16));
        quantize_q8(a->t1, L->nq, &a->qv);
        matvec(&L->o, &a->qv, 1, &y, PM_O);
        rmsnorm(a->t0, L->post_attn_norm, E, a->eps, a->t0);
        for (int i = 0; i < E; i++) a->sa[i] = a->t0[i] + a->x[i];
        rmsnorm(a->sa, L->ffn_norm, E, a->eps, a->t1);
        quantize_q8(a->t1, E, &a->qv);
        matvec(&L->gu, &a->qv, 1, &y, PM_GU);
        gelu_mul(a->t0, a->t0 + F, a->t1, F);
        quantize_q8(a->t1, F, &a->qv);
        matvec(&L->down, &a->qv, 1, &y, PM_DOWN);
        rmsnorm(a->t0, L->post_ffn_norm, E, a->eps, a->t0);
        for (int i = 0; i < E; i++) a->x[i] = (a->t0[i] + a->sa[i]) * L->out_scale;
    }
    rmsnorm(a->x, a->out_norm, E, a->eps, a->t1);
    quantize_q8(a->t1, E, &a->qv);
    int best; float *lg = a->logits;
    head_sweep(&a->head, &a->qv, 1, &lg, &best);
    y = h_out;
    matvec(&a->post, &a->qv, 1, &y, PM_PLE);
    if (p_top) {        // probability of the best token among the ten most likely
        float top[10]; int nt = 0;
        for (int v = 0; v < a->n_vocab; v++) {
            float x = lg[v];
            if (nt == 10 && x <= top[9]) continue;
            int i = nt < 10 ? nt++ : 9;
            while (i > 0 && top[i - 1] < x) { top[i] = top[i - 1]; i--; }
            top[i] = x;
        }
        float sum = 0; for (int i = 0; i < nt; i++) sum += expf(top[i] - top[0]);
        *p_top = 1.0f / sum;
    }
    a->t_total += now_sec() - t_start; a->steps++;
    return best;
}

void assistant_stats(const assistant *a, double *ms_per_step, long *steps) {
    *steps = a->steps; *ms_per_step = a->steps ? 1e3 * a->t_total / a->steps : 0;
}
