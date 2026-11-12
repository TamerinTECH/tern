// Hot loops. Three builds of each: AVX-512 with VNNI, AVX2 (+FMA, F16C), and portable C (also with
// -DTERN_FORCE_SCALAR). The build picks one from the compiler flags (-march=native).
//
// Packed rows. A row of nblk blocks (32 weights each) is a sequence of groups of L blocks: as many groups
// of 16 as fit, then at most one group each of 8, 4, 2 and 1 blocks.
//   P4  group := scale[L] (half floats)                      | nib[4][4L bytes]     w = scale * (n - 8)
//   P4M group := scale[L] | offset[L] (half floats)          | nib[4][4L bytes]     w = scale * n + offset * -8
//   P8  group := scaleA[L] | scaleB[L] (floats)              | val[8][4L bytes]     w = scale * (v - 32)
//   nib[u][4l + e] = n(l, 8u + e) | n(l, 8u + 4 + e) << 4          n(l, j): nibble j of block l of the group
//   val[v][4l + e] = v(l, 4v + e); scaleA is for the first 16 values of a block, scaleB for the last 16
// A quantized activation vector is stored in the matching order: for the same group 8 vectors of 4L bytes,
//   act[v][4l + e] = q(l, 4v + e)
// so that a byte dot-product instruction, which adds 4 products into each 32-bit lane, leaves the complete
// integer sum of block l in lane l after 8 instructions: one int-to-float conversion and one multiply by the
// block scales serve 16 blocks (512 weights). The low nibbles of nib[u] meet act[2u], the high ones act[2u+1].
// AVX2 works on half vectors (8 lanes), the portable code on single lanes: the layout is the same for all.
#include "tern.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
#define INL static inline __attribute__((always_inline))

const char *kernels_name(void) { return TERN_AVX512 ? "AVX-512 VNNI" : TERN_AVX2 ? "AVX2" : "portable C"; }

#if TERN_AVX2
static inline float hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(a, _mm_shuffle_ps(a, a, 1)));
}
#endif

static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n + 127) / 64 * 64)) abort();
    memset(p, 0, (n + 127) / 64 * 64);
    return p;
}

void qvec_alloc(qvec *v, int n) {
    size_t nb = (size_t)(n / QB + 16);
    v->q = xalloc((size_t)n + 64); v->p = xalloc((size_t)n + 64);
    v->d = xalloc(sizeof(float) * nb); v->k = xalloc(sizeof(int32_t) * nb); v->c = xalloc(sizeof(float) * nb);
    v->ha = xalloc(sizeof(int32_t) * nb); v->hb = xalloc(sizeof(int32_t) * nb);
}

// ---------------------------------------------------------------- activation quantization
static void quantize_blocks(const float *x, int n, qvec *out) {
    int nb = n / QB;
#if TERN_AVX512
    for (int b = 0; b < nb; b++) {
        __m512 v0 = _mm512_loadu_ps(x + b * QB), v1 = _mm512_loadu_ps(x + b * QB + 16);
        float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        __m512 vid = _mm512_set1_ps(id);
        __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid)), i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));
        _mm_storeu_si128((__m128i *)(out->q + b * QB), _mm512_cvtsepi32_epi8(i0));
        _mm_storeu_si128((__m128i *)(out->q + b * QB + 16), _mm512_cvtsepi32_epi8(i1));
        int sa = _mm512_reduce_add_epi32(i0), sb = _mm512_reduce_add_epi32(i1);
        out->d[b] = d; out->k[b] = -8 * (sa + sb); out->c[b] = d * (float)(-8 * (sa + sb)); out->ha[b] = -32 * sa; out->hb[b] = -32 * sb;
    }
#elif TERN_AVX2
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int b = 0; b < nb; b++) {
        __m256 v[4], mx = _mm256_setzero_ps();
        for (int k = 0; k < 4; k++) { v[k] = _mm256_loadu_ps(x + b * QB + 8 * k); mx = _mm256_max_ps(mx, _mm256_and_ps(v[k], absm)); }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        float amax = _mm_cvtss_f32(m), d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v[0], vid)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v[1], vid));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v[2], vid)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v[3], vid));
        // 32-bit -> 8-bit with saturation; the packs work per 128-bit lane, the permute restores the order
        __m256i p8 = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        p8 = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)(out->q + b * QB), p8);
        __m256i ya = _mm256_add_epi32(i0, i1), yb = _mm256_add_epi32(i2, i3);
        __m128i xa = _mm_add_epi32(_mm256_castsi256_si128(ya), _mm256_extracti128_si256(ya, 1));
        __m128i xb = _mm_add_epi32(_mm256_castsi256_si128(yb), _mm256_extracti128_si256(yb, 1));
        xa = _mm_add_epi32(xa, _mm_shuffle_epi32(xa, 0x4E)); xa = _mm_add_epi32(xa, _mm_shuffle_epi32(xa, 0xB1));
        xb = _mm_add_epi32(xb, _mm_shuffle_epi32(xb, 0x4E)); xb = _mm_add_epi32(xb, _mm_shuffle_epi32(xb, 0xB1));
        int sa = _mm_cvtsi128_si32(xa), sb = _mm_cvtsi128_si32(xb);
        out->d[b] = d; out->k[b] = -8 * (sa + sb); out->c[b] = d * (float)(-8 * (sa + sb)); out->ha[b] = -32 * sa; out->hb[b] = -32 * sb;
    }
#else
    for (int b = 0; b < nb; b++) {
        const float *p = x + b * QB; float amax = 0;
        for (int i = 0; i < QB; i++) { float a = fabsf(p[i]); if (a > amax) amax = a; }
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f; int sa = 0, sb = 0;
        for (int i = 0; i < QB; i++) { int q = (int)nearbyintf(p[i] * id); out->q[b * QB + i] = (int8_t)q; if (i < 16) sa += q; else sb += q; }
        out->d[b] = d; out->k[b] = -8 * (sa + sb); out->c[b] = d * (float)(-8 * (sa + sb)); out->ha[b] = -32 * sa; out->hb[b] = -32 * sb;
    }
#endif
}

typedef uint32_t __attribute__((may_alias)) u32a;

// the group sizes of a row, in order: for (L = 16; L; L >>= 1) while (GROUP_FITS(n - b, L)) ...
#define GROUP_FITS(left, L) ((left) >= (L) && ((L) == 16 || (left) % (2 * (L)) >= (L)))

void quantize_q8(const float *x, int n, qvec *out) {
    quantize_blocks(x, n, out);
    const u32a *src = (const u32a *)out->q; u32a *dst = (u32a *)out->p;
    int nb = n / QB, b = 0;
    for (int L = 16; L; L >>= 1)
        for (; GROUP_FITS(nb - b, L); b += L, src += 8 * L, dst += 8 * L)
            for (int v = 0; v < 8; v++) for (int l = 0; l < L; l++) dst[v * L + l] = src[l * 8 + v];
}

// ---------------------------------------------------------------- packing rows
int p4_kind_of(uint32_t type) { return type == GT_Q4_0 ? P4 : (type == GT_Q4_1 || type == GT_Q4_K) ? P4M : type == GT_Q6_K ? P8 : -1; }
static size_t block_bytes(int kind) { return kind == P4 ? 18 : kind == P4M ? 20 : 40; }
size_t p4_row_bytes(int cols, int kind) { return (size_t)(cols / QB) * block_bytes(kind); }
// offset of the group that starts at block b (b a multiple of 16)
static size_t group_offset(int b, int kind) { return (size_t)b * block_bytes(kind); }

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

// one GGUF row -> per block: 32 values, and two numbers: P4: scale bits; P4M: scale bits, offset / -8 bits (half
// floats); P8: the two half-block scales (floats)
static void decode_row(uint32_t type, const uint8_t *src, int nblk, uint8_t *val, uint16_t *s16, float *s32) {
    if (type == GT_Q4_0 || type == GT_Q4_1) {
        int hdr = type == GT_Q4_0 ? 2 : 4;
        for (int b = 0; b < nblk; b++, src += hdr + 16) {
            s16[2 * b] = rd16(src);
            s16[2 * b + 1] = type == GT_Q4_1 ? f32_to_f16(f16_to_f32(rd16(src + 2)) * -0.125f) : 0;
            for (int j = 0; j < 16; j++) { val[b * 32 + j] = src[hdr + j] & 15; val[b * 32 + j + 16] = src[hdr + j] >> 4; }
        }
    } else if (type == GT_Q4_K) {    // super-blocks of 8 blocks, w = d * sc_j * q - dmin * m_j
        for (int sb = 0; sb < nblk / 8; sb++, src += 144) {
            float d = f16_to_f32(rd16(src)), dmin = f16_to_f32(rd16(src + 2));
            const uint8_t *s = src + 4, *q = src + 16;
            for (int j = 0; j < 8; j++) {
                int ds, ms;
                if (j < 4) { ds = s[j] & 63; ms = s[j + 4] & 63; }
                else { ds = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); ms = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
                int b = sb * 8 + j;
                s16[2 * b] = f32_to_f16(d * (float)ds); s16[2 * b + 1] = f32_to_f16(dmin * (float)ms * 0.125f);
                const uint8_t *qq = q + (j / 2) * 32;
                for (int l = 0; l < 32; l++) val[b * 32 + l] = (j & 1) ? qq[l] >> 4 : qq[l] & 15;
            }
        }
    } else {                         // Q6_K: super-blocks of 8 blocks, w = d * sc_j * (v - 32), a scale per 16 values
        for (int sb = 0; sb < nblk / 8; sb++, src += 210) {
            const uint8_t *ql = src, *qh = src + 128; const int8_t *sc = (const int8_t *)(src + 192);
            float d = f16_to_f32(rd16(src + 208));
            uint8_t *y = val + sb * 256;
            for (int j = 0; j < 16; j++) s32[sb * 16 + j] = d * (float)sc[j];
            for (int n = 0; n < 256; n += 128, ql += 64, qh += 32)
                for (int l = 0; l < 32; l++) {
                    y[n + l] = (uint8_t)((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4));      y[n + l + 32] = (uint8_t)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4));
                    y[n + l + 64] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)); y[n + l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
                }
        }
    }
}

void p4_pack_row(uint32_t type, const uint8_t *src, int cols, uint8_t *dst) {
    int nblk = cols / QB, kind = p4_kind_of(type);
    uint8_t *val = malloc((size_t)cols); uint16_t *s16 = malloc(sizeof(uint16_t) * (size_t)nblk * 2); float *s32 = malloc(sizeof(float) * (size_t)nblk * 2);
    decode_row(type, src, nblk, val, s16, s32);
    int b = 0;
    for (int L = 16; L; L >>= 1)
        for (; GROUP_FITS(nblk - b, L); b += L) {
            if (kind == P8) {
                for (int h = 0; h < 2; h++) for (int l = 0; l < L; l++) memcpy(dst + 4 * (h * L + l), &s32[2 * (b + l) + h], 4);
                dst += 8 * L;
                for (int v = 0; v < 8; v++) for (int l = 0; l < L; l++) for (int e = 0; e < 4; e++) dst[v * 4 * L + 4 * l + e] = val[(b + l) * 32 + 4 * v + e];
                dst += 32 * L;
            } else {
                for (int h = 0; h < (kind == P4M ? 2 : 1); h++) for (int l = 0; l < L; l++) memcpy(dst + 2 * (h * L + l), &s16[2 * (b + l) + h], 2);
                dst += (kind == P4M ? 4 : 2) * L;
                for (int u = 0; u < 4; u++) for (int l = 0; l < L; l++) for (int e = 0; e < 4; e++)
                    dst[u * 4 * L + 4 * l + e] = (uint8_t)(val[(b + l) * 32 + 8 * u + e] | (val[(b + l) * 32 + 8 * u + 4 + e] << 4));
                dst += 16 * L;
            }
        }
    free(val); free(s16); free(s32);
}

// ---------------------------------------------------------------- packed matrix x quantized vectors
// The row kernels handle blocks b0..b1-1 of one packed row (g: the group that starts at b0, a multiple of 16)
// against S quantized vectors; out[s] = partial dot product.
typedef struct { const int8_t *p; const float *d, *c; const int32_t *k, *ha, *hb; } xref;

#if TERN_AVX512
// loads of L lanes (L * 4 bytes of bytes or ints, L floats, L half floats), zero-extended to 512 bits
INL __m512i ld_i(const void *p, const int L) {
    if (L == 16) return _mm512_loadu_si512(p);
    if (L == 8) return _mm512_zextsi256_si512(_mm256_loadu_si256(p));
    if (L == 4) return _mm512_zextsi128_si512(_mm_loadu_si128(p));
    if (L == 2) return _mm512_zextsi128_si512(_mm_loadl_epi64(p));
    int32_t v; memcpy(&v, p, 4); return _mm512_zextsi128_si512(_mm_cvtsi32_si128(v));
}
INL __m512 ld_f(const void *p, const int L) { return _mm512_castsi512_ps(ld_i(p, L)); }
INL __m512 ld_h(const uint8_t *p, const int L) {
    if (L == 16) return _mm512_cvtph_ps(_mm256_loadu_si256((const void *)p));
    if (L == 8) return _mm512_zextps256_ps512(_mm256_cvtph_ps(_mm_loadu_si128((const void *)p)));
    if (L == 4) return _mm512_zextps128_ps512(_mm_cvtph_ps(_mm_loadl_epi64((const void *)p)));
    int32_t v = 0; memcpy(&v, p, 2 * (size_t)L); return _mm512_zextps128_ps512(_mm_cvtph_ps(_mm_cvtsi32_si128(v)));
}

INL const uint8_t *group_p4(const uint8_t *g, const int L, const int has_min, const int b, const xref *x, const int S, __m512 *acc) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const int VW = 4 * L;
    const __m512 dw = ld_h(g, L); g += 2 * L;
    __m512 mw = _mm512_setzero_ps();
    if (has_min) { mw = ld_h(g, L); g += 2 * L; }
    const __m512i w0 = ld_i(g, L), w1 = ld_i(g + VW, L), w2 = ld_i(g + 2 * VW, L), w3 = ld_i(g + 3 * VW, L);
    const __m512i lo0 = _mm512_and_si512(w0, m4), hi0 = _mm512_and_si512(_mm512_srli_epi16(w0, 4), m4);
    const __m512i lo1 = _mm512_and_si512(w1, m4), hi1 = _mm512_and_si512(_mm512_srli_epi16(w1, 4), m4);
    const __m512i lo2 = _mm512_and_si512(w2, m4), hi2 = _mm512_and_si512(_mm512_srli_epi16(w2, 4), m4);
    const __m512i lo3 = _mm512_and_si512(w3, m4), hi3 = _mm512_and_si512(_mm512_srli_epi16(w3, 4), m4);
    for (int s = 0; s < S; s++) {
        const int8_t *a = x[s].p + (size_t)b * QB;
        __m512i i0 = has_min ? _mm512_setzero_si512() : ld_i(x[s].k + b, L), i1 = _mm512_setzero_si512();
        i0 = _mm512_dpbusd_epi32(i0, lo0, ld_i(a, L));          i1 = _mm512_dpbusd_epi32(i1, hi0, ld_i(a + VW, L));
        i0 = _mm512_dpbusd_epi32(i0, lo1, ld_i(a + 2 * VW, L)); i1 = _mm512_dpbusd_epi32(i1, hi1, ld_i(a + 3 * VW, L));
        i0 = _mm512_dpbusd_epi32(i0, lo2, ld_i(a + 4 * VW, L)); i1 = _mm512_dpbusd_epi32(i1, hi2, ld_i(a + 5 * VW, L));
        i0 = _mm512_dpbusd_epi32(i0, lo3, ld_i(a + 6 * VW, L)); i1 = _mm512_dpbusd_epi32(i1, hi3, ld_i(a + 7 * VW, L));
        const __m512 f = _mm512_cvtepi32_ps(_mm512_add_epi32(i0, i1));
        acc[s] = _mm512_fmadd_ps(f, _mm512_mul_ps(dw, ld_f(x[s].d + b, L)), acc[s]);
        if (has_min) acc[s] = _mm512_fmadd_ps(mw, ld_f(x[s].c + b, L), acc[s]);
    }
    return g + 4 * VW;
}

INL const uint8_t *group_p8(const uint8_t *g, const int L, const int b, const xref *x, const int S, __m512 *acc) {
    const int VW = 4 * L;
    const __m512 sa = ld_f(g, L), sb = ld_f(g + 4 * L, L); g += 8 * L;
    __m512i w[8];
    for (int v = 0; v < 8; v++) w[v] = ld_i(g + v * VW, L);
    for (int s = 0; s < S; s++) {
        const int8_t *a = x[s].p + (size_t)b * QB;
        __m512i i0 = ld_i(x[s].ha + b, L), i1 = ld_i(x[s].hb + b, L);
        i0 = _mm512_dpbusd_epi32(i0, w[0], ld_i(a, L));          i1 = _mm512_dpbusd_epi32(i1, w[4], ld_i(a + 4 * VW, L));
        i0 = _mm512_dpbusd_epi32(i0, w[1], ld_i(a + VW, L));     i1 = _mm512_dpbusd_epi32(i1, w[5], ld_i(a + 5 * VW, L));
        i0 = _mm512_dpbusd_epi32(i0, w[2], ld_i(a + 2 * VW, L)); i1 = _mm512_dpbusd_epi32(i1, w[6], ld_i(a + 6 * VW, L));
        i0 = _mm512_dpbusd_epi32(i0, w[3], ld_i(a + 3 * VW, L)); i1 = _mm512_dpbusd_epi32(i1, w[7], ld_i(a + 7 * VW, L));
        const __m512 f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), sa, _mm512_mul_ps(_mm512_cvtepi32_ps(i1), sb));
        acc[s] = _mm512_fmadd_ps(f, ld_f(x[s].d + b, L), acc[s]);
    }
    return g + 8 * VW;
}

INL void row_packed(const uint8_t *g, int b0, int b1, const int kind, const xref *x, const int S, float *out) {
    __m512 acc[S];
    for (int s = 0; s < S; s++) acc[s] = _mm512_setzero_ps();
    int b = b0;
#define GROUP(L) (kind == P8 ? group_p8(g, L, b, x, S, acc) : group_p4(g, L, kind == P4M, b, x, S, acc))
    for (; b + 16 <= b1; b += 16) g = GROUP(16);
    if (b1 - b >= 8) { g = GROUP(8); b += 8; }
    if (b1 - b >= 4) { g = GROUP(4); b += 4; }
    if (b1 - b >= 2) { g = GROUP(2); b += 2; }
    if (b1 - b >= 1) GROUP(1);
#undef GROUP
    for (int s = 0; s < S; s++) out[s] = _mm512_reduce_add_ps(acc[s]);
}
#elif TERN_AVX2
// loads of W lanes (W <= 8), zero-extended to 256 bits
INL __m256i ld_i(const void *p, const int W) {
    if (W == 8) return _mm256_loadu_si256(p);
    if (W == 4) return _mm256_zextsi128_si256(_mm_loadu_si128(p));
    if (W == 2) return _mm256_zextsi128_si256(_mm_loadl_epi64(p));
    int32_t v; memcpy(&v, p, 4); return _mm256_zextsi128_si256(_mm_cvtsi32_si128(v));
}
INL __m256 ld_f(const void *p, const int W) { return _mm256_castsi256_ps(ld_i(p, W)); }
INL __m256 ld_h(const uint8_t *p, const int W) {
    if (W == 8) return _mm256_cvtph_ps(_mm_loadu_si128((const void *)p));
    if (W == 4) return _mm256_zextps128_ps256(_mm_cvtph_ps(_mm_loadl_epi64((const void *)p)));
    int32_t v = 0; memcpy(&v, p, 2 * (size_t)W); return _mm256_zextps128_ps256(_mm_cvtph_ps(_mm_cvtsi32_si128(v)));
}
// sum of the two 16-bit halves of every 32-bit lane (what pmaddwd with ones does), without the multiplier:
// on Zen 2 / Zen 3 integer vector multiplies run on a single port, which is what limits this kernel
INL __m256i widen_add(__m256i x) {
    return _mm256_add_epi32(_mm256_srai_epi32(x, 16), _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16));
}
// W (<= 8) blocks of a group, starting at block b. sc/mn: their scales and offsets; nb: their bytes in the first
// of the 4 nibble vectors (VW bytes apart); ao: offset of their bytes in the first of the 8 activation vectors.
// AVX2 has no byte dot product: pmaddubsw adds 2 products into each 16-bit lane (at most 3810 with nibbles);
// the 8 vectors of a block add up to at most 30480, which still fits, so a single widening add completes the
// integer sums of 8 blocks.
INL void lanes_p4(const uint8_t *sc, const uint8_t *mn, const uint8_t *nb, const int VW, const int W, const int has_min, const int b, const size_t ao,
                  const xref *x, const int S, __m256 *acc) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256 dw = ld_h(sc, W);
    __m256 mw = _mm256_setzero_ps();
    if (has_min) mw = ld_h(mn, W);
    const __m256i w0 = ld_i(nb, W), w1 = ld_i(nb + VW, W), w2 = ld_i(nb + 2 * VW, W), w3 = ld_i(nb + 3 * VW, W);
    const __m256i lo0 = _mm256_and_si256(w0, m4), hi0 = _mm256_and_si256(_mm256_srli_epi16(w0, 4), m4);
    const __m256i lo1 = _mm256_and_si256(w1, m4), hi1 = _mm256_and_si256(_mm256_srli_epi16(w1, 4), m4);
    const __m256i lo2 = _mm256_and_si256(w2, m4), hi2 = _mm256_and_si256(_mm256_srli_epi16(w2, 4), m4);
    const __m256i lo3 = _mm256_and_si256(w3, m4), hi3 = _mm256_and_si256(_mm256_srli_epi16(w3, 4), m4);
    for (int s = 0; s < S; s++) {
        const int8_t *a = x[s].p + ao;
        __m256i t0 = _mm256_add_epi16(_mm256_maddubs_epi16(lo0, ld_i(a, W)), _mm256_maddubs_epi16(hi0, ld_i(a + VW, W)));
        __m256i t1 = _mm256_add_epi16(_mm256_maddubs_epi16(lo1, ld_i(a + 2 * VW, W)), _mm256_maddubs_epi16(hi1, ld_i(a + 3 * VW, W)));
        __m256i t2 = _mm256_add_epi16(_mm256_maddubs_epi16(lo2, ld_i(a + 4 * VW, W)), _mm256_maddubs_epi16(hi2, ld_i(a + 5 * VW, W)));
        __m256i t3 = _mm256_add_epi16(_mm256_maddubs_epi16(lo3, ld_i(a + 6 * VW, W)), _mm256_maddubs_epi16(hi3, ld_i(a + 7 * VW, W)));
        __m256i i32 = widen_add(_mm256_add_epi16(_mm256_add_epi16(t0, t1), _mm256_add_epi16(t2, t3)));
        if (!has_min) i32 = _mm256_add_epi32(i32, ld_i(x[s].k + b, W));
        acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32), _mm256_mul_ps(dw, ld_f(x[s].d + b, W)), acc[s]);
        if (has_min) acc[s] = _mm256_fmadd_ps(mw, ld_f(x[s].c + b, W), acc[s]);
    }
}
// the same for 8-bit rows (values up to 63: two pmaddubsw results may be added in 16 bits, not four)
INL void lanes_p8(const uint8_t *sa_p, const uint8_t *sb_p, const uint8_t *vb, const int VW, const int W, const int b, const size_t ao,
                  const xref *x, const int S, __m256 *acc) {
    const __m256 sa = ld_f(sa_p, W), sb = ld_f(sb_p, W);
    __m256i w[8];
    for (int v = 0; v < 8; v++) w[v] = ld_i(vb + v * VW, W);
    for (int s = 0; s < S; s++) {
        const int8_t *a = x[s].p + ao;
        __m256i t0 = _mm256_add_epi16(_mm256_maddubs_epi16(w[0], ld_i(a, W)), _mm256_maddubs_epi16(w[1], ld_i(a + VW, W)));
        __m256i t1 = _mm256_add_epi16(_mm256_maddubs_epi16(w[2], ld_i(a + 2 * VW, W)), _mm256_maddubs_epi16(w[3], ld_i(a + 3 * VW, W)));
        __m256i t2 = _mm256_add_epi16(_mm256_maddubs_epi16(w[4], ld_i(a + 4 * VW, W)), _mm256_maddubs_epi16(w[5], ld_i(a + 5 * VW, W)));
        __m256i t3 = _mm256_add_epi16(_mm256_maddubs_epi16(w[6], ld_i(a + 6 * VW, W)), _mm256_maddubs_epi16(w[7], ld_i(a + 7 * VW, W)));
        __m256i i0 = _mm256_add_epi32(_mm256_add_epi32(widen_add(t0), widen_add(t1)), ld_i(x[s].ha + b, W));
        __m256i i1 = _mm256_add_epi32(_mm256_add_epi32(widen_add(t2), widen_add(t3)), ld_i(x[s].hb + b, W));
        const __m256 f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), sa, _mm256_mul_ps(_mm256_cvtepi32_ps(i1), sb));
        acc[s] = _mm256_fmadd_ps(f, ld_f(x[s].d + b, W), acc[s]);
    }
}

INL void row_packed(const uint8_t *g, int b0, int b1, const int kind, const xref *x, const int S, float *out) {
    __m256 acc[S];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    const int hs = kind == P4 ? 2 : kind == P4M ? 4 : 8;         // bytes of scales (and offsets) per block
    const int vs = kind == P8 ? 2 : 1;                           // value vectors: 8 (P8) or 4 (nibbles)
    const int sw = kind == P8 ? 4 : 2;                           // bytes per scale
    int b = b0;
    for (; b + 16 <= b1; b += 16, g += (hs + 16 * vs) * 16) {       // a group of 16 is two halves of 8 lanes
        const uint8_t *vb = g + hs * 16;
        for (int h = 0; h < 2; h++) {
            const size_t ao = (size_t)b * QB + (size_t)h * 32;
            if (kind == P8) lanes_p8(g + sw * 8 * h, g + sw * (16 + 8 * h), vb + 32 * h, 64, 8, b + 8 * h, ao, x, S, acc);
            else lanes_p4(g + sw * 8 * h, g + sw * (16 + 8 * h), vb + 32 * h, 64, 8, kind == P4M, b + 8 * h, ao, x, S, acc);
        }
    }
#define TAIL(L) if (b1 - b >= L) { \
        if (kind == P8) lanes_p8(g, g + sw * L, g + hs * L, 4 * L, L, b, (size_t)b * QB, x, S, acc); \
        else lanes_p4(g, g + sw * L, g + hs * L, 4 * L, L, kind == P4M, b, (size_t)b * QB, x, S, acc); \
        g += (hs + 16 * vs) * L; b += L; }
    TAIL(8) TAIL(4) TAIL(2) TAIL(1)
#undef TAIL
    for (int s = 0; s < S; s++) out[s] = hsum256(acc[s]);
}
#else
INL void row_packed(const uint8_t *g, int b0, int b1, const int kind, const xref *x, const int S, float *out) {
    for (int s = 0; s < S; s++) out[s] = 0;
    int b = b0;
    for (int L = 16; L; L >>= 1)
        for (; GROUP_FITS(b1 - b, L); b += L) {
            const uint8_t *vb = g + (kind == P4 ? 2 : kind == P4M ? 4 : 8) * L;
            for (int s = 0; s < S; s++) {
                const int8_t *a = x[s].p + (size_t)b * QB;
                for (int l = 0; l < L; l++) {
                    if (kind == P8) {
                        int s0 = x[s].ha[b + l], s1 = x[s].hb[b + l]; float fa, fb;
                        for (int v = 0; v < 8; v++) for (int e = 0; e < 4; e++) { int t = vb[v * 4 * L + 4 * l + e] * a[v * 4 * L + 4 * l + e]; if (v < 4) s0 += t; else s1 += t; }
                        memcpy(&fa, g + 4 * l, 4); memcpy(&fb, g + 4 * (L + l), 4);
                        out[s] += ((float)s0 * fa + (float)s1 * fb) * x[s].d[b + l];
                    } else {
                        int sum = 0;
                        for (int u = 0; u < 4; u++) for (int e = 0; e < 4; e++) {
                            int w = vb[u * 4 * L + 4 * l + e];
                            sum += (w & 15) * a[(2 * u) * 4 * L + 4 * l + e] + (w >> 4) * a[(2 * u + 1) * 4 * L + 4 * l + e];
                        }
                        float dw = f16_to_f32(rd16(g + 2 * l));
                        if (kind == P4M) out[s] += dw * x[s].d[b + l] * (float)sum + f16_to_f32(rd16(g + 2 * (L + l))) * x[s].c[b + l];
                        else out[s] += dw * x[s].d[b + l] * (float)(sum + x[s].k[b + l]);
                    }
                }
            }
            g += block_bytes(kind) * (size_t)L;
        }
}
#endif

// rows r0..r1-1 (or, with idx, the rows idx[r0..r1-1]), column blocks c0..c1-1, S <= 8 vectors.
// first != 0: store the partial sums, else add them.
INL void sweep_packed(const pmat *M, const int *idx, int r0, int r1, int c0, int c1, const int kind, const qvec *x, const int S, float *const *y, int first) {
    const size_t goff = group_offset(c0, kind);
    xref xr[S]; float out[S];
    for (int s = 0; s < S; s++) xr[s] = (xref){x[s].p, x[s].d, x[s].c, x[s].k, x[s].ha, x[s].hb};
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = M->W + (size_t)(idx ? idx[r] : r) * M->stride;
        if (idx && r + 4 < r1) { const uint8_t *nx = M->W + (size_t)idx[r + 4] * M->stride; for (size_t o = 0; o < M->stride; o += 64) __builtin_prefetch(nx + o); }
        row_packed(row + goff, c0, c1, kind, xr, S, out);
        if (first) for (int s = 0; s < S; s++) y[s][r] = out[s];
        else for (int s = 0; s < S; s++) y[s][r] += out[s];
    }
}

#define SWEEP_CASES(KIND) switch (S) { \
        case 1: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 1, y, first); return; \
        case 2: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 2, y, first); return; \
        case 3: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 3, y, first); return; \
        case 4: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 4, y, first); return; \
        case 5: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 5, y, first); return; \
        case 6: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 6, y, first); return; \
        case 7: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 7, y, first); return; \
        case 8: sweep_packed(M, idx, r0, r1, c0, c1, KIND, x, 8, y, first); return; }
// constant slot counts and row kinds let the compiler keep the accumulators in registers
static void sweep_s(const pmat *M, const int *idx, int r0, int r1, int c0, int c1, const qvec *x, int S, float *const *y, int first) {
    if (M->kind == P4) SWEEP_CASES(P4) else if (M->kind == P4M) SWEEP_CASES(P4M) else SWEEP_CASES(P8)
}

// Cache blocking. A row is evaluated in column chunks of a fixed size; the activations of one group of slots
// for a chunk, plus the weights of one row block for that chunk, stay in the first-level cache. With more
// slots than one group, a row block is reused for every slot group before the next block is read from memory.
// The chunk size does not depend on the number of slots: the result for a slot is the same sequence of float
// operations whether it is evaluated alone or together with others (speculative decoding relies on that).
int tern_chunk_blocks = 96, tern_slot_group = 4;     // tunable (see bench/kernbench.c)

static void matvec_any(const pmat *M, const int *idx, int r0, int r1, const qvec *x, int S, float *const *y) {
    const int nblk = M->cols / QB, sg = S < tern_slot_group ? S : tern_slot_group, cb = tern_chunk_blocks;
    const size_t chunk_bytes = (size_t)(cb < nblk ? cb : nblk) * block_bytes(M->kind);
    int rb = S > sg ? (int)(12 * 1024 / chunk_bytes) : 1;   // rows per block
    if (rb < 1) rb = 1;
    for (int r = r0; r < r1; r += rb) {
        int re = r + rb < r1 ? r + rb : r1;
        for (int c0 = 0; c0 < nblk;) {
            int c1 = nblk - (c0 + cb) < 16 ? nblk : c0 + cb;
            for (int s0 = 0; s0 < S; s0 += sg)
                sweep_s(M, idx, r, re, c0, c1, x + s0, S - s0 < sg ? S - s0 : sg, y + s0, c0 == 0);
            c0 = c1;
        }
    }
}

void matvec_p4(const pmat *M, int r0, int r1, const qvec *x, int nslots, float *const *y) {
    matvec_any(M, NULL, r0, r1, x, nslots, y);
}
void matvec_p4_list(const pmat *M, const int *rows, int n, const qvec *x, int nslots, float *const *y) {
    matvec_any(M, rows, 0, n, x, nslots, y);
}

// ---------------------------------------------------------------- float matrix x float vectors
#if TERN_AVX512
INL void sweep_f(const fmat *M, int r0, int r1, int c0, int c1, const float *const *x, const int S, float *const *y, int first) {
    for (int r = r0; r < r1; r++) {
        __m512 acc[S];
        for (int s = 0; s < S; s++) acc[s] = _mm512_setzero_ps();
        if (M->f16) {
            const uint16_t *w = (const uint16_t *)M->W + (size_t)r * M->cols;
            for (int i = c0; i < c1; i += 16) {
                __m512 wv = _mm512_cvtph_ps(_mm256_loadu_si256((const void *)(w + i)));
                for (int s = 0; s < S; s++) acc[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x[s] + i), acc[s]);
            }
        } else {
            const float *w = (const float *)M->W + (size_t)r * M->cols;
            for (int i = c0; i < c1; i += 16) {
                __m512 wv = _mm512_loadu_ps(w + i);
                for (int s = 0; s < S; s++) acc[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x[s] + i), acc[s]);
            }
        }
        if (first) for (int s = 0; s < S; s++) y[s][r] = _mm512_reduce_add_ps(acc[s]);
        else for (int s = 0; s < S; s++) y[s][r] += _mm512_reduce_add_ps(acc[s]);
    }
}
#elif TERN_AVX2
INL void sweep_f(const fmat *M, int r0, int r1, int c0, int c1, const float *const *x, const int S, float *const *y, int first) {
    for (int r = r0; r < r1; r++) {
        __m256 acc[S];
        for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
        if (M->f16) {
            const uint16_t *w = (const uint16_t *)M->W + (size_t)r * M->cols;
            for (int i = c0; i < c1; i += 8) {
                __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const void *)(w + i)));
                for (int s = 0; s < S; s++) acc[s] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x[s] + i), acc[s]);
            }
        } else {
            const float *w = (const float *)M->W + (size_t)r * M->cols;
            for (int i = c0; i < c1; i += 8) {
                __m256 wv = _mm256_loadu_ps(w + i);
                for (int s = 0; s < S; s++) acc[s] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x[s] + i), acc[s]);
            }
        }
        if (first) for (int s = 0; s < S; s++) y[s][r] = hsum256(acc[s]);
        else for (int s = 0; s < S; s++) y[s][r] += hsum256(acc[s]);
    }
}
#else
static inline void sweep_f(const fmat *M, int r0, int r1, int c0, int c1, const float *const *x, const int S, float *const *y, int first) {
    for (int r = r0; r < r1; r++) for (int s = 0; s < S; s++) {
        float acc = 0;
        if (M->f16) { const uint16_t *w = (const uint16_t *)M->W + (size_t)r * M->cols; for (int i = c0; i < c1; i++) acc += f16_to_f32(w[i]) * x[s][i]; }
        else { const float *w = (const float *)M->W + (size_t)r * M->cols; for (int i = c0; i < c1; i++) acc += w[i] * x[s][i]; }
        if (first) y[s][r] = acc; else y[s][r] += acc;
    }
}
#endif
static void sweep_f_s(const fmat *M, int r0, int r1, int c0, int c1, const float *const *x, int S, float *const *y, int first) {
    switch (S) {
        case 1: sweep_f(M, r0, r1, c0, c1, x, 1, y, first); return;
        case 2: sweep_f(M, r0, r1, c0, c1, x, 2, y, first); return;
        case 3: sweep_f(M, r0, r1, c0, c1, x, 3, y, first); return;
        case 4: sweep_f(M, r0, r1, c0, c1, x, 4, y, first); return;
        case 5: sweep_f(M, r0, r1, c0, c1, x, 5, y, first); return;
        case 6: sweep_f(M, r0, r1, c0, c1, x, 6, y, first); return;
        case 7: sweep_f(M, r0, r1, c0, c1, x, 7, y, first); return;
        case 8: sweep_f(M, r0, r1, c0, c1, x, 8, y, first); return;
    }
}
// cols must be a multiple of 16. Column chunks of a fixed size, as above.
void matvec_f(const fmat *M, int r0, int r1, const float *const *x, int S, float *const *y) {
    const int sg = S < 4 ? S : 4, cc = 768;                 // columns per chunk
    const size_t chunk_bytes = (size_t)(cc < M->cols ? cc : M->cols) * (M->f16 ? 2 : 4);
    int rb = S > sg ? (int)(8 * 1024 / chunk_bytes) : 1;
    if (rb < 1) rb = 1;
    for (int r = r0; r < r1; r += rb) {
        int re = r + rb < r1 ? r + rb : r1;
        for (int c0 = 0; c0 < M->cols; c0 += cc) {
            int c1 = c0 + cc < M->cols ? c0 + cc : M->cols;
            for (int s0 = 0; s0 < S; s0 += sg)
                sweep_f_s(M, r, re, c0, c1, x + s0, S - s0 < sg ? S - s0 : sg, y + s0, c0 == 0);
        }
    }
}

// ---------------------------------------------------------------- float helpers
float dot_f32(const float *a, const float *b, int n) {
#if TERN_AVX512
    __m512 s0 = _mm512_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1u << (n - i < 16 ? n - i : 16)) - 1);
        s0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), s0);
        if (n - i > 16) { m = (__mmask16)((1u << (n - i - 16)) - 1); s1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i + 16), _mm512_maskz_loadu_ps(m, b + i + 16), s1); }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
#elif TERN_AVX2
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
#endif
}

#if TERN_AVX512
// exp(z) for z in [-87, 88]
static inline __m512 exp512(__m512 z) {
    const __m512 fk = _mm512_roundscale_ps(_mm512_mul_ps(z, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fk, _mm512_set1_ps(0.693359375f), z);
    r = _mm512_fnmadd_ps(fk, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f)); p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f)); p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(p, fk);
}
#elif TERN_AVX2
static inline __m256 exp256(__m256 z) {
    const __m256 fk = _mm256_round_ps(_mm256_mul_ps(z, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fk, _mm256_set1_ps(0.693359375f), z);
    r = _mm256_fnmadd_ps(fk, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f)); p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fk), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
#endif

// out[i] = gelu(g[i]) * u[i], gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))  (the form llama.cpp uses)
void gelu_mul(const float *g, const float *u, float *out, int n) {
    int i = 0;
#if TERN_AVX512
    const __m512 c0 = _mm512_set1_ps(0.79788456080286535588f), c1 = _mm512_set1_ps(0.044715f), one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(g + i);
        __m512 t = _mm512_mul_ps(_mm512_mul_ps(c0, x), _mm512_fmadd_ps(c1, _mm512_mul_ps(x, x), one));     // tanh argument
        // 0.5 (1 + tanh t) = 1 / (1 + exp(-2t)); clamp so that exp stays finite
        __m512 z = _mm512_max_ps(_mm512_min_ps(_mm512_mul_ps(t, _mm512_set1_ps(-2.0f)), _mm512_set1_ps(80.0f)), _mm512_set1_ps(-80.0f));
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, exp512(z)));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(x, sig), _mm512_loadu_ps(u + i)));
    }
#elif TERN_AVX2
    const __m256 c0 = _mm256_set1_ps(0.79788456080286535588f), c1 = _mm256_set1_ps(0.044715f), one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(g + i);
        __m256 t = _mm256_mul_ps(_mm256_mul_ps(c0, x), _mm256_fmadd_ps(c1, _mm256_mul_ps(x, x), one));
        __m256 z = _mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(t, _mm256_set1_ps(-2.0f)), _mm256_set1_ps(80.0f)), _mm256_set1_ps(-80.0f));
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256(z)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(x, sig), _mm256_loadu_ps(u + i)));
    }
#endif
    for (; i < n; i++) {
        float x = g[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x))) * u[i];
    }
}

// ---------------------------------------------------------------- attention helpers
// Scores of cached key rows against the nh query heads that share one key head.
// K: half floats, `stride` per token, of which this key head uses the first hd (the caller offsets K);
// rows[t]: index of the t-th visible row; q: nh * hd floats; out[t * so + h] for h < nh.
void attn_scores(const uint16_t *K, int stride, const int *rows, int n, const float *q, int hd, int nh, float *out, int so) {
#if TERN_AVX512 || TERN_AVX2
#if TERN_AVX512
    enum { HMAX = 16, VL = 16 };
#else
    enum { HMAX = 8, VL = 8 };
#endif
    for (int h0 = 0; h0 < nh; h0 += HMAX) {
        const int hn = nh - h0 < HMAX ? nh - h0 : HMAX;
        const float *qq = q + (size_t)h0 * hd;
        for (int t = 0; t < n; t++) {
            const uint16_t *k = K + (size_t)rows[t] * stride;
#if TERN_AVX512
            __m512 acc[HMAX];
            for (int h = 0; h < hn; h++) acc[h] = _mm512_setzero_ps();
            for (int i = 0; i < hd; i += VL) {
                const __m512 kv = _mm512_cvtph_ps(_mm256_loadu_si256((const void *)(k + i)));
                for (int h = 0; h < hn; h++) acc[h] = _mm512_fmadd_ps(kv, _mm512_loadu_ps(qq + (size_t)h * hd + i), acc[h]);
            }
            for (int h = 0; h < hn; h++) out[(size_t)t * so + h0 + h] = _mm512_reduce_add_ps(acc[h]);
#else
            __m256 acc[HMAX];
            for (int h = 0; h < hn; h++) acc[h] = _mm256_setzero_ps();
            for (int i = 0; i < hd; i += VL) {
                const __m256 kv = _mm256_cvtph_ps(_mm_loadu_si128((const void *)(k + i)));
                for (int h = 0; h < hn; h++) acc[h] = _mm256_fmadd_ps(kv, _mm256_loadu_ps(qq + (size_t)h * hd + i), acc[h]);
            }
            for (int h = 0; h < hn; h++) out[(size_t)t * so + h0 + h] = hsum256(acc[h]);
#endif
        }
    }
#else
    for (int t = 0; t < n; t++) {
        const uint16_t *k = K + (size_t)rows[t] * stride;
        for (int h = 0; h < nh; h++) {
            float sum = 0;
            for (int i = 0; i < hd; i++) sum += f16_to_f32(k[i]) * q[(size_t)h * hd + i];
            out[(size_t)t * so + h] = sum;
        }
    }
#endif
}

// softmax over the n rows, separately for each of the nh heads (sc[t * nh + h]), in place
void softmax_heads(float *sc, int n, int nh) {
#if TERN_AVX512 || TERN_AVX2
    if (nh == 1 || nh == 2 || nh == 4 || nh == 8 || nh == 16 || nh == 32) {
        // the head of element i is i mod nh, so every P floats repeat the same pattern
        enum { VL = TERN_AVX512 ? 16 : 8 };
        const int P = nh > 16 ? 32 : 16, NV = P / VL, tot = n * nh, full = tot / P * P;
        float mx[32], sum[32], inv[32];
        for (int j = 0; j < P; j++) { mx[j] = -INFINITY; sum[j] = 0; }
#if TERN_AVX512
#define VEC __m512
#define VLOAD _mm512_loadu_ps
#define VSTORE _mm512_storeu_ps
#define VMAX _mm512_max_ps
#define VADD _mm512_add_ps
#define VSUB _mm512_sub_ps
#define VMUL _mm512_mul_ps
#define VSET1 _mm512_set1_ps
#define VEXP exp512
#else
#define VEC __m256
#define VLOAD _mm256_loadu_ps
#define VSTORE _mm256_storeu_ps
#define VMAX _mm256_max_ps
#define VADD _mm256_add_ps
#define VSUB _mm256_sub_ps
#define VMUL _mm256_mul_ps
#define VSET1 _mm256_set1_ps
#define VEXP exp256
#endif
        VEC vm[4], vs[4], vi[4];
        for (int v = 0; v < NV; v++) vm[v] = VSET1(-INFINITY);
        for (int i = 0; i < full; i += P) for (int v = 0; v < NV; v++) vm[v] = VMAX(vm[v], VLOAD(sc + i + v * VL));
        for (int v = 0; v < NV; v++) VSTORE(mx + v * VL, vm[v]);
        for (int i = full; i < tot; i++) if (sc[i] > mx[i - full]) mx[i - full] = sc[i];
        for (int h = 0; h < nh; h++) for (int j = h + nh; j < P; j += nh) if (mx[j] > mx[h]) mx[h] = mx[j];
        for (int j = nh; j < P; j++) mx[j] = mx[j % nh];
        for (int v = 0; v < NV; v++) { vm[v] = VLOAD(mx + v * VL); vs[v] = VSET1(0.0f); }
        for (int i = 0; i < full; i += P) for (int v = 0; v < NV; v++) {
            VEC e = VEXP(VMAX(VSUB(VLOAD(sc + i + v * VL), vm[v]), VSET1(-87.0f)));
            VSTORE(sc + i + v * VL, e); vs[v] = VADD(vs[v], e);
        }
        for (int v = 0; v < NV; v++) VSTORE(sum + v * VL, vs[v]);
        for (int i = full; i < tot; i++) { float d = sc[i] - mx[i - full]; sc[i] = expf(d < -87.0f ? -87.0f : d); sum[i - full] += sc[i]; }
        for (int h = 0; h < nh; h++) for (int j = h + nh; j < P; j += nh) sum[h] += sum[j];
        for (int j = 0; j < P; j++) inv[j] = 1.0f / sum[j % nh];
        for (int v = 0; v < NV; v++) vi[v] = VLOAD(inv + v * VL);
        for (int i = 0; i < full; i += P) for (int v = 0; v < NV; v++) VSTORE(sc + i + v * VL, VMUL(VLOAD(sc + i + v * VL), vi[v]));
        for (int i = full; i < tot; i++) sc[i] *= inv[i - full];
#undef VEC
#undef VLOAD
#undef VSTORE
#undef VMAX
#undef VADD
#undef VSUB
#undef VMUL
#undef VSET1
#undef VEXP
        return;
    }
#endif
    for (int h = 0; h < nh; h++) {
        float mx = -INFINITY, sum = 0;
        for (int t = 0; t < n; t++) if (sc[(size_t)t * nh + h] > mx) mx = sc[(size_t)t * nh + h];
        for (int t = 0; t < n; t++) { sc[(size_t)t * nh + h] = expf(sc[(size_t)t * nh + h] - mx); sum += sc[(size_t)t * nh + h]; }
        for (int t = 0; t < n; t++) sc[(size_t)t * nh + h] /= sum;
    }
}

// out[h * hd + i] += sum_t sc[t * so + h] * V[rows[t] * stride + i] for the nh heads that share this value head
// and the value dims i0..i1-1 (multiples of 16)
void attn_mix(const uint16_t *V, int stride, const int *rows, int n, int hd, int nh, int i0, int i1, const float *sc, int so, float *out) {
#if TERN_AVX512 || TERN_AVX2
#if TERN_AVX512
    enum { HMAX = 16, VL = 16 };
#else
    enum { HMAX = 8, VL = 8 };
#endif
    for (int h0 = 0; h0 < nh; h0 += HMAX) {
        const int hn = nh - h0 < HMAX ? nh - h0 : HMAX;
        for (int i = i0; i < i1; i += VL) {
#if TERN_AVX512
            __m512 acc[HMAX];
            for (int h = 0; h < hn; h++) acc[h] = _mm512_loadu_ps(out + (size_t)(h0 + h) * hd + i);
            for (int t = 0; t < n; t++) {
                const __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const void *)(V + (size_t)rows[t] * stride + i)));
                const float *w = sc + (size_t)t * so + h0;
                for (int h = 0; h < hn; h++) acc[h] = _mm512_fmadd_ps(v, _mm512_set1_ps(w[h]), acc[h]);
            }
            for (int h = 0; h < hn; h++) _mm512_storeu_ps(out + (size_t)(h0 + h) * hd + i, acc[h]);
#else
            __m256 acc[HMAX];
            for (int h = 0; h < hn; h++) acc[h] = _mm256_loadu_ps(out + (size_t)(h0 + h) * hd + i);
            for (int t = 0; t < n; t++) {
                const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const void *)(V + (size_t)rows[t] * stride + i)));
                const float *w = sc + (size_t)t * so + h0;
                for (int h = 0; h < hn; h++) acc[h] = _mm256_fmadd_ps(v, _mm256_broadcast_ss(w + h), acc[h]);
            }
            for (int h = 0; h < hn; h++) _mm256_storeu_ps(out + (size_t)(h0 + h) * hd + i, acc[h]);
#endif
        }
    }
#else
    for (int h = 0; h < nh; h++) for (int t = 0; t < n; t++) {
        const float w = sc[(size_t)t * so + h]; const uint16_t *v = V + (size_t)rows[t] * stride;
        for (int i = i0; i < i1; i++) out[(size_t)h * hd + i] += w * f16_to_f32(v[i]);
    }
#endif
}

void f32_to_f16_row(const float *x, uint16_t *y, int n) {
    int i = 0;
#if TERN_AVX512
    for (; i + 16 <= n; i += 16) _mm256_storeu_si256((void *)(y + i), _mm512_cvtps_ph(_mm512_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif TERN_AVX2
    for (; i + 8 <= n; i += 8) _mm_storeu_si128((void *)(y + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
    for (; i < n; i++) y[i] = f32_to_f16(x[i]);
}
