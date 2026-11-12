// Fast output layer.
//
// The exact output layer multiplies the hidden state h with all 262,144 rows of the output matrix: 0.5 - 1.8 GB
// of memory traffic per token for Gemma 4. This one reads about a seventh of that and returns the same token:
//   1. sketch: every row w is kept a second time in a form that is cheap to scan,
//          w  ~  (w's coordinates along u_1..u_R) + a * sign(rest of w)
//      u_1..u_R are the directions in which final hidden states vary most (the "plan": R x E numbers, written
//      by `tern --make-head` from generated text). The part of a logit that comes from these directions is
//      computed from R bytes per token, the rest from ONE BIT per weight and one scale per row.
//   2. the K tokens with the best sketch scores are scored again with their full rows, by the same kernel as the
//      exact layer: the returned token is the exact argmax whenever it is among those K.
//   3. core: rows that one bit per weight cannot hold are scored with their full rows at every step. Those are
//      the rows where a few large weights carry the row (about 4 % of the rows, among them the most frequent
//      tokens); they are found at load time from the rows themselves (CORE_SHAPE below).
// Why the directions are needed: a 1-bit copy of a row is wrong in proportion to the length of the vector it
// meets, and a fifth of a hidden state's energy is the same vector at every step. Measurements for each of these
// choices: tools/research/head_candidates.py and docs/DESIGN.md.
#define _GNU_SOURCE
#include "tern.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) && !defined(TERN_FORCE_SCALAR)
#include <immintrin.h>
#define TERN_AVX512 1
#define TERN_AVX2 0
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__) && !defined(TERN_FORCE_SCALAR)
#include <immintrin.h>
#define TERN_AVX512 0
#define TERN_AVX2 1
#else
#define TERN_AVX512 0
#define TERN_AVX2 0
#endif

#define KEEP 64             // best candidates per slot reported back (for sampling)
#define PLAN_MAGIC "TERNHD05"
// A row whose rest r (after the directions) has mean(r^2) / mean(|r|)^2 above this goes to the core. The ratio is
// pi/2 = 1.57 when the weights are spread like a bell curve, which holds for 96 % of the rows (their ratios lie
// between 1.52 and 1.60), and grows when few weights dominate.
#define CORE_SHAPE 1.65f

struct fasthead {
    int V, E, R, K, KL, nt; float softcap;
    float *U;                       // R x E: the plan's directions (orthonormal)
    float *cs;                      // R: step of the int8 coordinates along each direction
    uint8_t *sk; size_t stride;     // per token: R int8 | E/8 bytes of signs | padding to a multiple of 64
    float *a;                       // per token: scale of the signs
    pmat rows;                      // the model's output matrix (packed rows)
    int n_core, *core; uint8_t *is_core; pmat core_rows;       // core: token ids (ascending), bitmap, their rows side by side
    // per slot
    qvec qv[TERN_MAX_SLOTS];        // h for the full rows
    float *g[TERN_MAX_SLOTS];       // R coordinates of h along the directions, times cs
    int8_t *q8[TERN_MAX_SLOTS]; float sq[TERN_MAX_SLOTS]; int32_t Q[TERN_MAX_SLOTS];   // rest of h as int8, its step, sum
    float *sc[TERN_MAX_SLOTS], thr[TERN_MAX_SLOTS];     // sketch scores of all tokens; the K-th best
    float *core_sc[TERN_MAX_SLOTS];                     // logits of the core rows
    float *hp, *sel;
    // per thread
    float **scratch; int **cand; float **clog;
    float *tsel; int *tnsel;        // [thread][slot]: the thread's KL best scores
    int *tbest, *tcount; float *tbestv;     // [thread][slot]
    int *ttop; float *ttopv; int want_top;  // [thread][slot][KEEP]
    double t_stage[4], t_build; long calls, slots, cands;
};

static void head_bench(fasthead *h);
static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n + 127) / 64 * 64)) { fprintf(stderr, "out of memory\n"); exit(1); }
    memset(p, 0, (n + 127) / 64 * 64);
    return p;
}

static void axpy(float *restrict y, const float *restrict x, float a, int n) { for (int i = 0; i < n; i++) y[i] += a * x[i]; }

// ---------------------------------------------------------------- building the sketch (at load)
// sign bits of r (bit i set: r[i] >= 0); returns sum |r| and sum r^2
static void signs_of(const float *r, int n, uint8_t *bits, double *sum_abs, double *sum_sq) {
    double s1 = 0, s2 = 0;
#if TERN_AVX512
    __m512 a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        const __m512 v = _mm512_loadu_ps(r + i);
        const uint16_t m = (uint16_t)_mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GE_OQ);
        memcpy(bits + i / 8, &m, 2);
        a1 = _mm512_add_ps(a1, _mm512_abs_ps(v)); a2 = _mm512_fmadd_ps(v, v, a2);
    }
    s1 = _mm512_reduce_add_ps(a1); s2 = _mm512_reduce_add_ps(a2);
#elif TERN_AVX2
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(r + i);
        bits[i / 8] = (uint8_t)_mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ));
        a1 = _mm256_add_ps(a1, _mm256_and_ps(v, absm)); a2 = _mm256_fmadd_ps(v, v, a2);
    }
    float t1[8], t2[8]; _mm256_storeu_ps(t1, a1); _mm256_storeu_ps(t2, a2);
    for (int i = 0; i < 8; i++) { s1 += t1[i]; s2 += t2[i]; }
#else
    memset(bits, 0, (size_t)n / 8);
    for (int i = 0; i < n; i++) { float v = r[i]; s1 += fabsf(v); s2 += v * v; if (v >= 0) bits[i >> 3] |= (uint8_t)(1u << (i & 7)); }
#endif
    *sum_abs = s1; *sum_sq = s2;
}

typedef struct { fasthead *h; const model *m; float *A, *shape; } build_job;
static void build_run(void *arg, int tid) {
    build_job *j = arg; fasthead *h = j->h; const int E = h->E, R = h->R;
    const int r0 = (int)((int64_t)h->V * tid / h->nt), r1 = (int)((int64_t)h->V * (tid + 1) / h->nt);
    float *w = xalloc(sizeof(float) * (size_t)E);
    for (int t = r0; t < r1; t++) {
        float *A = j->A + (size_t)t * R; double s1, s2;
        model_output_row(j->m, t, w);
        for (int k = 0; k < R; k++) A[k] = dot_f32(w, h->U + (size_t)k * E, E);
        for (int k = 0; k < R; k++) axpy(w, h->U + (size_t)k * E, -A[k], E);
        signs_of(w, E, h->sk + (size_t)t * h->stride + R, &s1, &s2);
        // the scale: the root mean square of the rest. (sum|r| / E would fit the row best; sum r^2 / sum|r| keeps
        // the length of a row that meets a vector along its own signs. Measured, the value between them misses least.)
        h->a[t] = (float)sqrt(s2 / E);
        j->shape[t] = s1 > 0 ? (float)(s2 * E / (s1 * s1)) : 0.0f;
    }
    free(w);
}
static void build_quant_run(void *arg, int tid) {
    build_job *j = arg; fasthead *h = j->h; const int R = h->R;
    const int r0 = (int)((int64_t)h->V * tid / h->nt), r1 = (int)((int64_t)h->V * (tid + 1) / h->nt);
    for (int t = r0; t < r1; t++) {
        int8_t *dst = (int8_t *)h->sk + (size_t)t * h->stride; const float *A = j->A + (size_t)t * R;
        for (int k = 0; k < R; k++) { long q = lrintf(A[k] / h->cs[k]); dst[k] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q); }
    }
}

fasthead *head_load(const char *plan_path, model *m, int n_keep) {
    double t0 = now_sec();
    FILE *f = fopen(plan_path, "rb");
    if (!f) { perror(plan_path); return NULL; }
    char magic[8]; int hdr[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, PLAN_MAGIC, 8) || fread(hdr, 4, 4, f) != 4) {
        fprintf(stderr, "%s: not a plan for the fast output layer (make one with: tern -m MODEL -f prompts.txt -np 32 --quiet --make-head FILE)\n", plan_path); fclose(f); return NULL;
    }
    fasthead *h = calloc(1, sizeof *h);
    h->E = hdr[0]; h->R = hdr[1]; h->nt = pool_threads();
    model_output_matrix(m, &h->rows, &h->softcap);
    h->V = h->rows.rows;
    if (h->E != h->rows.cols) { fprintf(stderr, "%s: made for a model of width %d, this one has %d\n", plan_path, h->E, h->rows.cols); fclose(f); return NULL; }
    if (h->E % 256 || h->R % 16 || h->R < 16 || h->R > 64) { fprintf(stderr, "%s: %d directions for width %d are not supported\n", plan_path, h->R, h->E); fclose(f); return NULL; }
    const int E = h->E, R = h->R, V = h->V, nt = h->nt;
    h->U = xalloc(sizeof(float) * (size_t)R * E);
    if (fread(h->U, sizeof(float), (size_t)R * E, f) != (size_t)R * E) { fprintf(stderr, "%s: too short\n", plan_path); fclose(f); return NULL; }
    fclose(f);
    h->K = n_keep < 16 ? 16 : n_keep > V / 16 ? V / 16 : n_keep;
    h->KL = (4 * h->K + nt - 1) / nt; if (h->KL < 64) h->KL = 64;
    h->stride = ((size_t)R + E / 8 + 63) / 64 * 64;
    h->sk = mmap(NULL, (size_t)V * h->stride, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h->sk == MAP_FAILED) { perror("mmap"); return NULL; }
    madvise(h->sk, (size_t)V * h->stride, tern_huge_pages() ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    h->cs = xalloc(sizeof(float) * (size_t)R); h->a = xalloc(sizeof(float) * (size_t)V);
    build_job j = {h, m, malloc(sizeof(float) * (size_t)V * R), malloc(sizeof(float) * (size_t)V)};
    pool_run(build_run, &j);
    for (int k = 0; k < R; k++) {
        float amax = 0;
        for (int t = 0; t < V; t++) { float a = fabsf(j.A[(size_t)t * R + k]); if (a > amax) amax = a; }
        h->cs[k] = amax > 0 ? amax / 127.0f : 1.0f;
    }
    pool_run(build_quant_run, &j);
    // the core: rows that the signs cannot hold, side by side
    h->is_core = calloc((size_t)V / 8 + 1, 1);
    for (int t = 0; t < V; t++) if (j.shape[t] > CORE_SHAPE || j.shape[t] <= 0) h->n_core++;
    h->core = malloc(sizeof(int) * (size_t)(h->n_core + 1));
    h->core_rows = h->rows; h->core_rows.rows = h->n_core;
    uint8_t *cr = xalloc((size_t)(h->n_core + 1) * h->rows.stride);
    for (int t = 0, i = 0; t < V; t++) if (j.shape[t] > CORE_SHAPE || j.shape[t] <= 0) {
        h->core[i] = t; h->is_core[t >> 3] |= (uint8_t)(1u << (t & 7));
        memcpy(cr + (size_t)i++ * h->rows.stride, h->rows.W + (size_t)t * h->rows.stride, h->rows.stride);
    }
    h->core_rows.W = cr;
    free(j.A); free(j.shape);
    h->hp = xalloc(sizeof(float) * (size_t)E);
    h->sel = xalloc(sizeof(float) * (size_t)nt * h->KL);
    h->scratch = calloc((size_t)nt, sizeof(float *)); h->cand = calloc((size_t)nt, sizeof(int *)); h->clog = calloc((size_t)nt, sizeof(float *));
    for (int t = 0; t < nt; t++) {
        size_t len = (size_t)V / nt + 64;
        h->scratch[t] = xalloc(sizeof(float) * 2 * (size_t)h->KL); h->cand[t] = xalloc(sizeof(int) * len); h->clog[t] = xalloc(sizeof(float) * len);
    }
    h->tsel = xalloc(sizeof(float) * (size_t)nt * TERN_MAX_SLOTS * h->KL);
    h->tnsel = xalloc(sizeof(int) * (size_t)nt * TERN_MAX_SLOTS);
    h->tbest = xalloc(sizeof(int) * (size_t)nt * TERN_MAX_SLOTS); h->tcount = xalloc(sizeof(int) * (size_t)nt * TERN_MAX_SLOTS);
    h->tbestv = xalloc(sizeof(float) * (size_t)nt * TERN_MAX_SLOTS);
    h->t_build = now_sec() - t0;
    if (getenv("TERN_HEADBENCH")) head_bench(h);
    return h;
}

// ---------------------------------------------------------------- scan
// Sketch scores of tokens t0..t1-1 for slots s0..s0+ns-1 (ns <= 4):
//   sc[t] = sum_k A8[t][k] * g[k]  +  a[t] * sq * (2 * (sum of q8 where the sign bit is set) - Q)
// The sign bits become a vector of 0/1 bytes once per block of weights, whatever the number of slots.
#if TERN_AVX512
// na accumulators per slot: the multiply-add has 4 cycles of latency, one chain per slot would be bound by it
static inline __attribute__((always_inline)) void scan_n(const fasthead *h, int t0, int t1, int s0, const int ns, const int na) {
    const int nb = h->E / 64, R = h->R;
    const __m512i ones = _mm512_set1_epi8(1);
    for (int t = t0; t < t1; t++) {
        const uint8_t *row = h->sk + (size_t)t * h->stride, *bits = row + R;
        __m512i acc[4][4]; __m512 av[4];
        for (int s = 0; s < ns; s++) for (int u = 0; u < na; u++) acc[s][u] = _mm512_setzero_si512();
        for (int b = 0; b < nb; b += na)
            for (int u = 0; u < na; u++) {
                uint64_t w; memcpy(&w, bits + 8 * (b + u), 8);
                const __m512i sel = _mm512_maskz_mov_epi8(_cvtu64_mask64(w), ones);
                for (int s = 0; s < ns; s++) acc[s][u] = _mm512_dpbusd_epi32(acc[s][u], sel, _mm512_load_si512((const void *)(h->q8[s0 + s] + 64 * (b + u))));
            }
        for (int k = 0; k < R; k += 16) av[k / 16] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const void *)(row + k))));
        const float a = h->a[t];
        for (int s = 0; s < ns; s++) {
            const float as = a * h->sq[s0 + s];
            __m512i sum = acc[s][0];
            for (int u = 1; u < na; u++) sum = _mm512_add_epi32(sum, acc[s][u]);
            __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(sum), _mm512_set1_ps(2.0f * as));
            for (int k = 0; k < R; k += 16) v = _mm512_fmadd_ps(av[k / 16], _mm512_loadu_ps(h->g[s0 + s] + k), v);
            h->sc[s0 + s][t] = _mm512_reduce_add_ps(v) - as * (float)h->Q[s0 + s];
        }
    }
}
#elif TERN_AVX2
static inline float hsum_ps(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_add_ss(x, _mm_shuffle_ps(x, x, 1)));
}
static inline __attribute__((always_inline)) void scan_n(const fasthead *h, int t0, int t1, int s0, const int ns, const int na) {
    (void)na;
    const int nb = h->E / 32, R = h->R;
    // bit j of byte k of a 32-bit word -> byte 8k+j
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (int t = t0; t < t1; t++) {
        const uint8_t *row = h->sk + (size_t)t * h->stride, *bits = row + R;
        __m256i a16[4], a32[4]; __m256 av[8];
        for (int s = 0; s < ns; s++) { a16[s] = _mm256_setzero_si256(); a32[s] = _mm256_setzero_si256(); }
        for (int b = 0; b < nb;) {
            const int e = b + 64 < nb ? b + 64 : nb;        // 64 blocks: the 16-bit sums stay below 64 * 254
            for (; b < e; b++) {
                int32_t w; memcpy(&w, bits + 4 * b, 4);
                const __m256i sel = _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(w), shuf), bit), ones8);
                for (int s = 0; s < ns; s++)
                    a16[s] = _mm256_add_epi16(a16[s], _mm256_maddubs_epi16(sel, _mm256_load_si256((const void *)(h->q8[s0 + s] + 32 * b))));
            }
            for (int s = 0; s < ns; s++) { a32[s] = _mm256_add_epi32(a32[s], _mm256_madd_epi16(a16[s], ones16)); a16[s] = _mm256_setzero_si256(); }
        }
        for (int k = 0; k < R; k += 8) av[k / 8] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const void *)(row + k))));
        const float a = h->a[t];
        for (int s = 0; s < ns; s++) {
            const float as = a * h->sq[s0 + s];
            __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(a32[s]), _mm256_set1_ps(2.0f * as));
            for (int k = 0; k < R; k += 8) v = _mm256_fmadd_ps(av[k / 8], _mm256_loadu_ps(h->g[s0 + s] + k), v);
            h->sc[s0 + s][t] = hsum_ps(v) - as * (float)h->Q[s0 + s];
        }
    }
}
#else
static inline void scan_n(const fasthead *h, int t0, int t1, int s0, const int ns, const int na) {
    (void)na;
    const int E = h->E, R = h->R;
    for (int t = t0; t < t1; t++) {
        const uint8_t *row = h->sk + (size_t)t * h->stride, *bits = row + R; const int8_t *A = (const int8_t *)row;
        const float a = h->a[t];
        for (int s = 0; s < ns; s++) {
            const int8_t *q = h->q8[s0 + s]; int sum = 0; float pc = 0;
            for (int i = 0; i < E; i++) if (bits[i >> 3] >> (i & 7) & 1) sum += q[i];
            for (int k = 0; k < R; k++) pc += (float)A[k] * h->g[s0 + s][k];
            h->sc[s0 + s][t] = pc + a * h->sq[s0 + s] * (float)(2 * sum - h->Q[s0 + s]);
        }
    }
}
#endif
static void scan(const fasthead *h, int t0, int t1, int s0, int ns) {
    switch (ns) {       // a constant slot count per call: the accumulators stay in registers
        case 1: scan_n(h, t0, t1, s0, 1, 4); break;
        case 2: scan_n(h, t0, t1, s0, 2, 4); break;
        case 3: scan_n(h, t0, t1, s0, 3, 2); break;
        default: scan_n(h, t0, t1, s0, 4, 2); break;
    }
}

// TERN_HEADBENCH=1: cost of the scan kernel on one thread (after loading), per token and slot
static void head_bench(fasthead *h) {
    for (int s = 0; s < 4; s++) {
        h->g[s] = xalloc(sizeof(float) * (size_t)h->R); h->q8[s] = xalloc((size_t)h->E); h->sc[s] = xalloc(sizeof(float) * (size_t)h->V);
        h->core_sc[s] = xalloc(sizeof(float) * (size_t)(h->n_core + 16));
        qvec_alloc(&h->qv[s], h->E);
        for (int i = 0; i < h->E; i++) h->q8[s][i] = (int8_t)((i * 37 + s * 11) % 255 - 127);
        h->sq[s] = 0.01f;
    }
    for (int ns = 1; ns <= 4; ns++) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            double t = now_sec();
            for (int t0 = 0; t0 < h->V; t0 += 16) scan(h, t0, t0 + 16 < h->V ? t0 + 16 : h->V, 0, ns);
            t = now_sec() - t; if (t < best) best = t;
        }
        fprintf(stderr, "scan kernel, 1 thread, %d slot%s: %.2f ms per sweep, %.1f ns per token and slot (%.2f GB/s)\n",
                ns, ns > 1 ? "s" : " ", best * 1e3, best * 1e9 / h->V / ns, (double)h->V * h->stride / best / 1e9);
    }
}

// k-th largest of a[0..n) (k = 1: the maximum); reorders a so that a[0..k) are the k largest
static float kth_largest(float *a, int n, int k) {
    int lo = 0, hi = n - 1, target = k - 1;
    while (lo < hi) {
        float pivot = a[(lo + hi) / 2]; int i = lo, j = hi;
        while (i <= j) {
            while (a[i] > pivot) i++;
            while (a[j] < pivot) j--;
            if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (target <= j) hi = j; else if (target >= i) lo = i; else break;
    }
    return a[target];
}

// the k largest of v[0..n) into buf (room for 2k values), unordered; returns how many (k, or n if smaller)
static int largest_of(const float *v, int n, int k, float *buf) {
    float thr = -INFINITY; int m = 0;
    for (int i = 0; i < n; i++)
        if (v[i] > thr) {
            buf[m++] = v[i];
            if (m == 2 * k) { thr = kth_largest(buf, m, k); m = k; }
        }
    if (m > k) { kth_largest(buf, m, k); m = k; }
    return m;
}

// is (logit v, token t) better than (bv, bt)? Ties go to the lower token id, as in the exact layer.
static inline int better(float v, int t, float bv, int bt) { return bt < 0 || v > bv || (v == bv && t < bt); }

typedef struct { fasthead *h; int n, stage; } hjob;
static void head_run(void *arg, int tid) {
    hjob *j = arg; fasthead *h = j->h; const int S = j->n;
    const int r0 = (int)((int64_t)h->V * tid / h->nt), r1 = (int)((int64_t)h->V * (tid + 1) / h->nt), len = r1 - r0;
    const int c0 = (int)((int64_t)h->n_core * tid / h->nt), c1 = (int)((int64_t)h->n_core * (tid + 1) / h->nt);
    if (j->stage == 0) {            // sketch scores of this thread's tokens and the best KL of them per slot; its share of the core rows
        for (int t = r0; t < r1; t += 16) {         // 16 sketch rows stay in the first-level cache across the slot groups
            const int e = t + 16 < r1 ? t + 16 : r1;
            for (int s = 0; s < S; s += 4) scan(h, t, e, s, S - s < 4 ? S - s : 4);
        }
        float *tmp = h->scratch[tid];
        for (int s = 0; s < S; s++) {
            const int kl = largest_of(h->sc[s] + r0, len, h->KL, tmp);
            memcpy(h->tsel + ((size_t)tid * TERN_MAX_SLOTS + s) * h->KL, tmp, sizeof(float) * (size_t)kl);
            h->tnsel[tid * TERN_MAX_SLOTS + s] = kl;
        }
        if (c1 > c0) matvec_p4(&h->core_rows, c0, c1, h->qv, S, h->core_sc);
    } else {                        // full rows of this thread's tokens at or above the slot's threshold; the best of them and of its core rows
        int *c = h->cand[tid]; float *lg = h->clog[tid]; const uint8_t *core = h->is_core;
        for (int s = 0; s < S; s++) {
            const float *sc = h->sc[s]; const float thr = h->thr[s]; int n = 0, best = -1; float bv = 0;
            for (int t = r0; t < r1; t++) if (sc[t] >= thr && !(core[t >> 3] >> (t & 7) & 1)) c[n++] = t;
            if (n) matvec_p4_list(&h->rows, c, n, &h->qv[s], 1, &lg);
            for (int i = 0; i < n; i++) if (better(lg[i], c[i], bv, best)) { best = c[i]; bv = lg[i]; }
            const float *cv = h->core_sc[s];
            for (int i = c0; i < c1; i++) if (better(cv[i], h->core[i], bv, best)) { best = h->core[i]; bv = cv[i]; }
            const int o = tid * TERN_MAX_SLOTS + s;
            h->tbest[o] = best; h->tbestv[o] = bv; h->tcount[o] = n;
            if (h->want_top) {
                int *tt = h->ttop + (size_t)o * KEEP, tk = 0; float *tv = h->ttopv + (size_t)o * KEEP;
                for (int part = 0; part < 2; part++) {
                    const int cnt = part ? c1 - c0 : n;
                    for (int i = 0; i < cnt; i++) {
                        const float v = part ? cv[c0 + i] : lg[i];
                        if (tk == KEEP && v <= tv[KEEP - 1]) continue;
                        int k = tk < KEEP ? tk++ : KEEP - 1;
                        while (k > 0 && tv[k - 1] < v) { tv[k] = tv[k - 1]; tt[k] = tt[k - 1]; k--; }
                        tv[k] = v; tt[k] = part ? h->core[c0 + i] : c[i];
                    }
                }
                for (int k = tk; k < KEEP; k++) tt[k] = -1;
            }
        }
    }
}

// Best token for each of n hidden states. If top_tok/top_logit are given they receive, per state, the 64 best
// candidates and their logits (soft-capped as the model defines), best first.
void head_argmax(fasthead *h, float *const *hidden, int n, int *out, int *top_tok, float *top_logit) {
    double t0 = now_sec(), t1;
    const int E = h->E, R = h->R, nt = h->nt;
    if (top_tok && !h->ttop) {
        h->ttop = xalloc(sizeof(int) * (size_t)nt * TERN_MAX_SLOTS * KEEP); h->ttopv = xalloc(sizeof(float) * (size_t)nt * TERN_MAX_SLOTS * KEEP);
    }
    h->want_top = top_tok != NULL;
    for (int s = 0; s < n; s++) {
        if (!h->g[s]) {
            qvec_alloc(&h->qv[s], E);
            h->g[s] = xalloc(sizeof(float) * (size_t)R); h->q8[s] = xalloc((size_t)E); h->sc[s] = xalloc(sizeof(float) * (size_t)h->V);
            h->core_sc[s] = xalloc(sizeof(float) * (size_t)(h->n_core + 16));
        }
        quantize_q8(hidden[s], E, &h->qv[s]);
        // h = (its part along the directions) + rest; the rest as int8 with one step
        float *hp = h->hp, *g = h->g[s], amax = 0;
        memcpy(hp, hidden[s], sizeof(float) * (size_t)E);
        for (int k = 0; k < R; k++) g[k] = dot_f32(h->U + (size_t)k * E, hidden[s], E);
        for (int k = 0; k < R; k++) { axpy(hp, h->U + (size_t)k * E, -g[k], E); g[k] *= h->cs[k]; }
        for (int i = 0; i < E; i++) { float a = fabsf(hp[i]); if (a > amax) amax = a; }
        const float d = amax > 0 ? amax / 127.0f : 1.0f, id = 1.0f / d; int sum = 0;
        for (int i = 0; i < E; i++) { int q = (int)lrintf(hp[i] * id); h->q8[s][i] = (int8_t)q; sum += q; }
        h->sq[s] = d; h->Q[s] = sum;
    }
    t1 = now_sec(); h->t_stage[0] += t1 - t0; t0 = t1;
    hjob j = {h, n, 0};
    pool_run(head_run, &j);
    t1 = now_sec(); h->t_stage[1] += t1 - t0; t0 = t1;
    for (int s = 0; s < n; s++) {       // the K-th best score overall is among the threads' best
        int m = 0;
        for (int t = 0; t < nt; t++) {
            int k = h->tnsel[t * TERN_MAX_SLOTS + s];
            memcpy(h->sel + m, h->tsel + ((size_t)t * TERN_MAX_SLOTS + s) * h->KL, sizeof(float) * (size_t)k); m += k;
        }
        h->thr[s] = m > h->K ? kth_largest(h->sel, m, h->K) : -INFINITY;
    }
    t1 = now_sec(); h->t_stage[2] += t1 - t0; t0 = t1;
    j.stage = 1;
    pool_run(head_run, &j);
    for (int s = 0; s < n; s++) {
        int best = -1; float bv = 0;
        for (int t = 0; t < nt; t++) {
            const int o = t * TERN_MAX_SLOTS + s;
            h->cands += h->tcount[o];
            if (h->tbest[o] >= 0 && better(h->tbestv[o], h->tbest[o], bv, best)) { best = h->tbest[o]; bv = h->tbestv[o]; }
        }
        out[s] = best < 0 ? 0 : best;
        if (top_tok) {
            int *tt = top_tok + s * KEEP, tk = 0; float *tv = top_logit + s * KEEP;
            for (int t = 0; t < nt; t++) {
                const size_t o = (size_t)(t * TERN_MAX_SLOTS + s) * KEEP;
                for (int i = 0; i < KEEP && h->ttop[o + i] >= 0; i++) {
                    float v = h->ttopv[o + i];
                    if (tk == KEEP && v <= tv[KEEP - 1]) break;         // the thread's list is sorted
                    int k = tk < KEEP ? tk++ : KEEP - 1;
                    while (k > 0 && tv[k - 1] < v) { tv[k] = tv[k - 1]; tt[k] = tt[k - 1]; k--; }
                    tv[k] = v; tt[k] = h->ttop[o + i];
                }
            }
            for (int i = 0; i < KEEP; i++) {
                if (i >= tk) { tt[i] = -1; tv[i] = -INFINITY; }
                else if (h->softcap > 0) tv[i] = h->softcap * tanhf(tv[i] / h->softcap);
            }
        }
    }
    h->cands += (long)n * h->n_core;
    h->t_stage[3] += now_sec() - t0;
    h->calls++; h->slots += n;
}

void head_stats(const fasthead *h, double *ms_per_call, double *cands_per_slot, long *calls) {
    double total = h->t_stage[0] + h->t_stage[1] + h->t_stage[2] + h->t_stage[3];
    *ms_per_call = h->calls ? 1e3 * total / h->calls : 0; *calls = h->calls;
    *cands_per_slot = h->slots ? (double)h->cands / h->slots : 0;
    if (getenv("TERN_PROF") && h->calls)
        fprintf(stderr, "fast output layer: sketch %.0f MB (%d directions + 1 bit per weight) and %d core rows (%.0f MB), built in %.2f s | per call (ms): split of h %.3f  scan + core %.3f  threshold %.3f  full rows %.3f\n",
                (double)h->V * h->stride / 1e6, h->R, h->n_core, (double)h->n_core * h->rows.stride / 1e6, h->t_build, 1e3 * h->t_stage[0] / h->calls,
                1e3 * h->t_stage[1] / h->calls, 1e3 * h->t_stage[2] / h->calls, 1e3 * h->t_stage[3] / h->calls);
}

// ---------------------------------------------------------------- making a plan
// The R directions in which n recorded hidden states (rows of H) vary most: the leading eigenvectors of H^T H,
// by orthogonal iteration (a few rounds are enough: what matters is the space they span, not each vector).
typedef struct { const float *H; int n, E, p; const float *X; float **Y; int nt; } plan_job;
static void plan_run(void *arg, int tid) {
    plan_job *j = arg; const int E = j->E, p = j->p;
    const int i0 = (int)((int64_t)j->n * tid / j->nt), i1 = (int)((int64_t)j->n * (tid + 1) / j->nt);
    float *Y = j->Y[tid];
    memset(Y, 0, sizeof(float) * (size_t)p * E);
    for (int i = i0; i < i1; i++) {
        const float *hrow = j->H + (size_t)i * E;
        for (int k = 0; k < p; k++) axpy(Y + (size_t)k * E, hrow, dot_f32(hrow, j->X + (size_t)k * E, E), E);
    }
}
static void orthonormalize(float *X, int p, int E) {       // modified Gram-Schmidt, rows of X
    for (int k = 0; k < p; k++) {
        float *x = X + (size_t)k * E;
        for (int rep = 0; rep < 2; rep++)
            for (int q = 0; q < k; q++) {
                const float *y = X + (size_t)q * E; double d = 0;
                for (int i = 0; i < E; i++) d += (double)x[i] * y[i];
                for (int i = 0; i < E; i++) x[i] -= (float)d * y[i];
            }
        double nn = 0; for (int i = 0; i < E; i++) nn += (double)x[i] * x[i];
        float inv = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0.0f;
        for (int i = 0; i < E; i++) x[i] *= inv;
    }
}

int head_make_plan(const char *path, const float *H, int n, int E, int R) {
    const int p = R + 8, nt = pool_threads();
    if (n < 4 * p) { fprintf(stderr, "plan: only %d recorded steps; generate more text (a few thousand tokens)\n", n); return 1; }
    float *X = xalloc(sizeof(float) * (size_t)p * E);
    float **Y = calloc((size_t)nt, sizeof(float *));
    for (int t = 0; t < nt; t++) Y[t] = xalloc(sizeof(float) * (size_t)p * E);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < (size_t)p * E; i++) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; X[i] = (float)((double)(rng >> 11) / 9007199254740992.0 - 0.5); }
    orthonormalize(X, p, E);
    plan_job j = {H, n, E, p, X, Y, nt};
    for (int it = 0; it < 24; it++) {
        pool_run(plan_run, &j);
        memset(X, 0, sizeof(float) * (size_t)p * E);
        for (int t = 0; t < nt; t++) for (size_t i = 0; i < (size_t)p * E; i++) X[i] += Y[t][i];
        orthonormalize(X, p, E);
    }
    double total = 0, part = 0;         // share of the hidden states' energy in the R directions
    for (int i = 0; i < n; i++) {
        const float *hrow = H + (size_t)i * E;
        total += dot_f32(hrow, hrow, E);
        for (int k = 0; k < R; k++) { float d = dot_f32(hrow, X + (size_t)k * E, E); part += (double)d * d; }
    }
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    int hdr[4] = {E, R, n, 0};
    fwrite(PLAN_MAGIC, 1, 8, f); fwrite(hdr, 4, 4, f); fwrite(X, sizeof(float), (size_t)R * E, f);
    fclose(f);
    fprintf(stderr, "plan: %d directions from %d recorded steps hold %.1f%% of the hidden states' energy -> %s (%.0f KB)\n",
            R, n, 100 * part / total, path, (24 + 4.0 * R * E) / 1024);
    for (int t = 0; t < nt; t++) free(Y[t]);
    free(Y); free(X);
    return 0;
}
