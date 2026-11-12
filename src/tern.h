// tern - a small, fast CPU inference engine for Gemma 4 text models.
// Reads published GGUF files as they are.
#ifndef TERN_H
#define TERN_H
#include <stddef.h>
#include <stdint.h>

#define TERN_MAX_SLOTS 32   // tokens evaluated together in one pass over the weights
#define QB 32               // quantization block (elements)

// ---------------------------------------------------------------- gguf.c
enum {      // GGUF tensor types used here
    GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q4_1 = 3, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14, GT_BF16 = 30
};
typedef struct {
    const char *name; int name_len;
    int n_dims; uint64_t dims[4];
    uint32_t type;
    const uint8_t *data;
} gguf_tensor;

typedef struct gguf gguf;
gguf *gguf_open(const char *path);
const gguf_tensor *gguf_find_tensor(const gguf *g, const char *name);
int gguf_get_u32(const gguf *g, const char *key, uint32_t *out);
int gguf_get_f32(const gguf *g, const char *key, float *out);
// arrays: returns element count, sets *first to the raw first element (strings: u64 len + bytes, packed)
int64_t gguf_get_array(const gguf *g, const char *key, uint32_t *elem_type, const uint8_t **first);
const char *gguf_get_str(const gguf *g, const char *key, uint64_t *len);
void gguf_release_pages(const gguf *g);

// ---------------------------------------------------------------- pool.c
typedef void (*pool_fn)(void *arg, int tid);
void pool_init(int n_threads);
void pool_run(pool_fn fn, void *arg);   // runs fn on all threads (tid 0 = caller) and waits
int  pool_threads(void);
void pool_set_idle(pool_fn fn);         // fn(NULL, tid) is called by waiting workers between two looks for work; it must return within a microsecond
double now_sec(void);

// ---------------------------------------------------------------- quant.c (GGUF row formats)
size_t quant_row_bytes(uint32_t type, int cols);                              // 0: type not supported
int    dequant_row(uint32_t type, const uint8_t *src, int cols, float *out);   // 0 on success
const char *quant_type_name(uint32_t type);
float    f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

// ---------------------------------------------------------------- kernels.c
// An activation vector quantized like llama.cpp's Q8_0: per block of 32 a scale and 32 int8.
typedef struct {
    int8_t  *q;     // n values, natural order
    int8_t  *p;     // the same values in the kernel's interleaved order (see kernels.c)
    float   *d;     // n/32 scales
    int32_t *k;     // n/32: -8 * sum(q in block)   (folds the "-8" offset of 4-bit weights into the integer sum)
    float   *c;     // n/32: d * k                  (the same term in float, for rows that carry their own offset)
    int32_t *ha, *hb;   // n/32 each: -32 * sum of the first / last 16 q of the block (offset of 6-bit weights)
} qvec;
void qvec_alloc(qvec *v, int n);
void quantize_q8(const float *x, int n, qvec *out);

// Packed matrix rows (layout: kernels.c), per block of 32 weights:
//   P4   a scale and 32 nibbles, w = scale * (nibble - 8)                         (GGUF Q4_0)
//   P4M  scale and offset, w = scale * nibble + offset                            (GGUF Q4_1, Q4_K)
//   P8   two scales (one per 16 weights) and 32 bytes, w = scale * (byte - 32)    (GGUF Q6_K)
enum { P4 = 0, P4M = 1, P8 = 2 };
typedef struct { const uint8_t *W; int rows, cols, kind; size_t stride; } pmat;
int    p4_kind_of(uint32_t gguf_type);                 // P4, P4M, P8 or -1 (no packed form)
size_t p4_row_bytes(int cols, int kind);
void   p4_pack_row(uint32_t gguf_type, const uint8_t *src, int cols, uint8_t *dst);
// y[s][r] = sum_j W[r][j] * x[s][j] for rows r0..r1-1
void matvec_p4(const pmat *M, int r0, int r1, const qvec *x, int nslots, float *const *y);
void matvec_p4_list(const pmat *M, const int *rows, int n, const qvec *x, int nslots, float *const *y);   // y[s][i] for rows[i]
// float matrices (rows x cols, row-major; half floats if f16)
typedef struct { const void *W; int rows, cols, f16; } fmat;
void matvec_f(const fmat *M, int r0, int r1, const float *const *x, int nslots, float *const *y);
void gelu_mul(const float *g, const float *u, float *out, int n);   // out = gelu(g) * u
float dot_f32(const float *a, const float *b, int n);
// attention: keys and values are cached as half floats, `stride` values per token (all key/value heads of the
// token side by side). One call handles the nh query heads that share one key/value head (the caller offsets
// K or V to that head); scores go to out[t * so + h].
void attn_scores(const uint16_t *K, int stride, const int *rows, int n, const float *q, int hd, int nh, float *out, int so);
void softmax_heads(float *sc, int n, int nh);       // per head, over the n rows of sc[t * nh + h]
void attn_mix(const uint16_t *V, int stride, const int *rows, int n, int hd, int nh, int i0, int i1, const float *sc, int so, float *out);
void f32_to_f16_row(const float *x, uint16_t *y, int n);
const char *kernels_name(void);

// ---------------------------------------------------------------- tokenizer.c
typedef struct tokenizer tokenizer;
tokenizer *tok_load(const gguf *g);
int  tok_encode(const tokenizer *t, const char *text, int add_bos, int parse_special, int *out, int max);
int  tok_piece(const tokenizer *t, int id, char *buf, int max);   // text for one token (spaces restored)
int  tok_n_vocab(const tokenizer *t);
int  tok_bos(const tokenizer *t);
int  tok_eos(const tokenizer *t);
int  tok_find(const tokenizer *t, const char *text);              // exact token text -> id or -1
int  tok_is_special(const tokenizer *t, int id);

// ---------------------------------------------------------------- model.c
typedef struct model model;
typedef struct {
    int n_threads;
    int n_ctx;      // tokens per sequence
    int n_seq;      // independent sequences (each has its own cache)
} model_params;

// One token to evaluate in a pass. Slots of a pass form a forest: a slot sees the committed tokens of its
// sequence plus its ancestors in this pass. A plain batch has parent = -1 everywhere; a run of consecutive
// tokens is a chain (parent = previous slot); speculative decoding uses a tree of guesses.
typedef struct { int token, seq, parent; } slot;

model *model_load(const gguf *g, const model_params *p);
int  model_n_embd(const model *m);
int  model_n_vocab(const model *m);
int  model_n_ctx(const model *m);
int  model_len(const model *m, int seq);          // committed tokens of a sequence
void model_truncate(model *m, int seq, int len);  // forget committed tokens beyond len
// ONE pass over the weights for n slots (n <= TERN_MAX_SLOTS). hidden[i] (n_embd floats, final norm applied)
// is filled where hidden[i] != NULL. Nothing is committed: the keys/values of the slots stay in scratch rows.
// Slots without a hidden output only run the layers that write keys and values (Gemma 4: the lower layers).
void model_forward(model *m, const slot *sl, int n, float *const *hidden);
// Keep the given slots of the last pass (a path from a root, in order) as the next tokens of the sequence.
void model_commit(model *m, int seq, const int *slot_idx, int n);
// exact output layer: logits for all tokens; returns argmax. logits may be NULL.
int  model_head_exact(model *m, const float *hidden, float *logits);
void model_head_exact_batch(model *m, float *const *hidden, int n, int *out);   // n states, one sweep
void model_output_matrix(const model *m, pmat *M, float *softcap);
void model_output_row(const model *m, int token, float *out);    // one row of the output matrix as floats
double model_weight_bytes(const model *m);
int  tern_huge_pages(void);        // 1: the weights go to huge pages, 0: to 4 KB pages (rule in model.c, or TERN_HUGE)
double model_bench_sweep(model *m, int S, int reps);
void model_print_profile(void);

// Drafter ("assistant"): a small companion model that guesses the next tokens from the main model's cache.
typedef struct assistant assistant;
assistant *assistant_load(const gguf *g, model *target);
// One draft step for a sequence: `token` follows the committed tokens and h is the main model's final hidden
// state that predicted it (or the previous step's h_out). Returns the drafted next token.
int  assistant_step(assistant *a, int seq, int token, const float *h, float *h_out, float *p_top);
void assistant_stats(const assistant *a, double *ms_per_step, long *steps);

// ---------------------------------------------------------------- head.c (fast output layer)
typedef struct fasthead fasthead;
// plan_path: a file written by head_make_plan (tern --make-head). keep: tokens that are scored with full rows.
fasthead *head_load(const char *plan_path, model *m, int keep);
// the best token for each of n hidden states; top_tok/top_logit (optional): 64 best candidates per state
void head_argmax(fasthead *h, float *const *hidden, int n, int *out, int *top_tok, float *top_logit);
void head_stats(const fasthead *h, double *ms_per_call, double *rows_scanned, long *calls);
// writes a plan from n recorded final hidden states (rows of H, E wide): the R directions they vary most in
int  head_make_plan(const char *path, const float *H, int n, int E, int R);

// ---------------------------------------------------------------- draft.c (guesses for speculative decoding)
typedef struct drafter drafter;
drafter *draft_load(const char *ngram_path, float min_conf);   // path may be NULL: copy-from-context only
int draft_next(const drafter *d, const int *ctx, int n);       // guessed next token or -1

#endif
