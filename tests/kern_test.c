// Checks the kernels against plain double-precision arithmetic on random data.
// build: see Makefile (make test). Run with the SIMD build and with PORTABLE=1.
#include "../src/tern.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float rndf(void) { return (float)(rnd() % 20001) / 10000.0f - 1.0f; }
static float rndg(void) { return (rndf() + rndf() + rndf()) * (rnd() % 7 == 0 ? 4.0f : 1.0f); }   // heavy-ish tails

static int fails;
static void check(const char *what, double got, double want, double tol) {
    double err = fabs(got - want), scale = fabs(want) > 1 ? fabs(want) : 1;
    if (!(err <= tol * scale)) { if (fails < 20) printf("FAIL %s: got %.6f want %.6f\n", what, got, want); fails++; }
}

// a random GGUF row of the given 4-bit type
static void random_row(uint32_t type, int cols, uint8_t *row) {
    size_t n = quant_row_bytes(type, cols);
    for (size_t i = 0; i < n; i++) row[i] = (uint8_t)rnd();
    if (type == GT_Q4_0 || type == GT_Q4_1) {
        int bs = type == GT_Q4_0 ? 18 : 20;
        for (int b = 0; b < cols / 32; b++) {
            uint16_t h = f32_to_f16(0.001f + 0.02f * (float)(rnd() % 100) / 100.0f); memcpy(row + (size_t)b * bs, &h, 2);
            if (type == GT_Q4_1) { h = f32_to_f16(-0.2f * (float)(rnd() % 100) / 100.0f); memcpy(row + (size_t)b * bs + 2, &h, 2); }
        }
    } else if (type == GT_Q4_K) {
        for (int b = 0; b < cols / 256; b++) {
            uint16_t h = f32_to_f16(0.0005f + 0.001f * (float)(rnd() % 100) / 100.0f); memcpy(row + (size_t)b * 144, &h, 2);
            h = f32_to_f16(0.002f * (float)(rnd() % 100) / 100.0f); memcpy(row + (size_t)b * 144 + 2, &h, 2);
        }
    } else {    // Q6_K
        for (int b = 0; b < cols / 256; b++) { uint16_t h = f32_to_f16(0.00005f + 0.0002f * (float)(rnd() % 100) / 100.0f); memcpy(row + (size_t)b * 210 + 208, &h, 2); }
    }
}

static void test_p4(uint32_t type, int rows, int cols, int S) {
    size_t gb = quant_row_bytes(type, cols); int kind = p4_kind_of(type);
    pmat M = {NULL, rows, cols, kind, p4_row_bytes(cols, kind)};
    uint8_t *src = malloc(gb * (size_t)rows), *W = aligned_alloc(64, (M.stride * (size_t)rows + 127) / 64 * 64);
    float *wf = malloc(sizeof(float) * (size_t)rows * cols);
    for (int r = 0; r < rows; r++) {
        random_row(type, cols, src + gb * (size_t)r);
        dequant_row(type, src + gb * (size_t)r, cols, wf + (size_t)r * cols);
        p4_pack_row(type, src + gb * (size_t)r, cols, W + M.stride * (size_t)r);
    }
    M.W = W;
    qvec x[TERN_MAX_SLOTS]; float *y[TERN_MAX_SLOTS], *xf = malloc(sizeof(float) * (size_t)cols);
    double *xr = malloc(sizeof(double) * (size_t)S * cols);
    for (int s = 0; s < S; s++) {
        qvec_alloc(&x[s], cols); y[s] = malloc(sizeof(float) * (size_t)rows);
        for (int i = 0; i < cols; i++) xf[i] = rndg() * (i % 64 < 32 ? 3.0f : 0.2f);
        quantize_q8(xf, cols, &x[s]);
        for (int i = 0; i < cols; i++) xr[(size_t)s * cols + i] = (double)x[s].d[i / 32] * x[s].q[i];   // what the kernel sees
    }
    matvec_p4(&M, 0, rows, x, S, y);
    char what[96];
    for (int s = 0; s < S; s++) for (int r = 0; r < rows; r++) {
        double want = 0, mag = 0;
        for (int i = 0; i < cols; i++) { double t = (double)wf[(size_t)r * cols + i] * xr[(size_t)s * cols + i]; want += t; mag += fabs(t); }
        snprintf(what, sizeof what, "p4 %s %dx%d S=%d slot %d row %d", quant_type_name(type), rows, cols, S, s, r);
        double err = fabs(y[s][r] - want);
        // Q4_K scales pass through half floats (relative 2^-11 per block); the rest is float rounding
        if (!(err <= (type == GT_Q4_K ? 6e-4 : 2e-5) * mag + 1e-4)) { if (fails < 20) printf("FAIL %s: got %.6f want %.6f (sum of magnitudes %.3f)\n", what, y[s][r], want, mag); fails++; }
    }
    // list variant: a few rows in scrambled order
    int idx[16]; float *yl[TERN_MAX_SLOTS];
    for (int i = 0; i < 16; i++) idx[i] = (int)(rnd() % (uint32_t)rows);
    for (int s = 0; s < S; s++) yl[s] = malloc(sizeof(float) * 16);
    matvec_p4_list(&M, idx, 16, x, S, yl);
    for (int s = 0; s < S; s++) for (int i = 0; i < 16; i++) check("p4 list", yl[s][i], y[s][idx[i]], 1e-6);
    free(src); free(W); free(wf); free(xf); free(xr);
}

static void test_f(int rows, int cols, int f16, int S) {
    float *wf = malloc(sizeof(float) * (size_t)rows * cols); uint16_t *wh = malloc(sizeof(uint16_t) * (size_t)rows * cols);
    for (size_t i = 0; i < (size_t)rows * cols; i++) { wh[i] = f32_to_f16(rndg() * 0.05f); wf[i] = f16_to_f32(wh[i]); }
    fmat M = {f16 ? (void *)wh : (void *)wf, rows, cols, f16};
    float *x[TERN_MAX_SLOTS], *y[TERN_MAX_SLOTS];
    for (int s = 0; s < S; s++) { x[s] = malloc(sizeof(float) * (size_t)cols); y[s] = malloc(sizeof(float) * (size_t)rows); for (int i = 0; i < cols; i++) x[s][i] = rndg(); }
    matvec_f(&M, 0, rows, (const float *const *)x, S, y);
    for (int s = 0; s < S; s++) for (int r = 0; r < rows; r++) {
        double want = 0, mag = 0;
        for (int i = 0; i < cols; i++) { double t = (double)wf[(size_t)r * cols + i] * x[s][i]; want += t; mag += fabs(t); }
        if (!(fabs(y[s][r] - want) <= 2e-5 * mag + 1e-6)) { if (fails < 20) printf("FAIL float matrix %dx%d f16=%d S=%d: got %.6f want %.6f\n", rows, cols, f16, S, y[s][r], want); fails++; }
    }
    free(wf); free(wh);
}

static void test_attention(int n, int hd, int nh) {
    int total = n + 7;
    uint16_t *K = malloc(sizeof(uint16_t) * (size_t)total * hd), *V = malloc(sizeof(uint16_t) * (size_t)total * hd);
    float *q = malloc(sizeof(float) * (size_t)nh * hd), *sc = malloc(sizeof(float) * (size_t)n * nh), *out = calloc((size_t)nh * hd, sizeof(float));
    int *rows = malloc(sizeof(int) * (size_t)n);
    for (size_t i = 0; i < (size_t)total * hd; i++) { K[i] = f32_to_f16(rndg()); V[i] = f32_to_f16(rndg()); }
    for (int i = 0; i < nh * hd; i++) q[i] = rndg() * 0.1f;
    for (int t = 0; t < n; t++) rows[t] = (t * 5 + 3) % total;
    attn_scores(K, hd, rows, n, q, hd, nh, sc, nh);
    double *want = malloc(sizeof(double) * (size_t)n * nh);
    for (int t = 0; t < n; t++) for (int h = 0; h < nh; h++) {
        double s = 0; for (int i = 0; i < hd; i++) s += (double)f16_to_f32(K[(size_t)rows[t] * hd + i]) * q[h * hd + i];
        want[t * nh + h] = s; check("attention scores", sc[t * nh + h], s, 1e-4);
    }
    softmax_heads(sc, n, nh);
    for (int h = 0; h < nh; h++) {
        double mx = -1e300, sum = 0;
        for (int t = 0; t < n; t++) if (want[t * nh + h] > mx) mx = want[t * nh + h];
        for (int t = 0; t < n; t++) sum += exp(want[t * nh + h] - mx);
        for (int t = 0; t < n; t++) check("softmax", sc[t * nh + h], exp(want[t * nh + h] - mx) / sum, 2e-4);
    }
    attn_mix(V, hd, rows, n, hd, nh, 0, hd / 2, sc, nh, out);
    attn_mix(V, hd, rows, n, hd, nh, hd / 2, hd, sc, nh, out);
    for (int h = 0; h < nh; h++) for (int i = 0; i < hd; i += 37) {
        double s = 0; for (int t = 0; t < n; t++) s += (double)sc[t * nh + h] * f16_to_f32(V[(size_t)rows[t] * hd + i]);
        check("attention mix", out[h * hd + i], s, 1e-4);
    }
    free(K); free(V); free(q); free(sc); free(out); free(rows); free(want);
}

int main(void) {
    printf("kernels: %s\n", kernels_name());
    static const uint32_t types[] = {GT_Q4_0, GT_Q4_1, GT_Q4_K, GT_Q6_K};
    static const int widths[] = {256, 512, 768, 1536, 2816, 4096, 5376, 6144};      // 768: groups 16+8; 2816: 5x16 + 8; 5376: 10x16 + 8
    for (int t = 0; t < 4; t++) for (int w = 0; w < 8; w++) for (int S = 1; S <= 32; S = S < 9 ? S + 1 : S + 11)
        test_p4(types[t], 37, widths[w], S);
    // widths that are not multiples of 256 (Q4_0 / Q4_1 only): 640 = 16+4, 704 = 16+4+2, 2112 = 4x16+2, 96 = 2+1, 32 = 1
    static const int odd[] = {640, 704, 2112, 96, 32, 992};
    for (int w = 0; w < 6; w++) for (int S = 1; S <= 9; S += 4) { test_p4(GT_Q4_0, 41, odd[w], S); test_p4(GT_Q4_1, 41, odd[w], S); }
    test_p4(GT_Q4_0, 300, 12288, 32);
    for (int f16 = 0; f16 < 2; f16++) { test_f(256, 1536, f16, 1); test_f(1536, 256, f16, 5); test_f(100, 1536, f16, 32); test_f(64, 4096, f16, 20); }
    test_attention(1, 256, 8); test_attention(37, 256, 8); test_attention(600, 512, 8); test_attention(50, 256, 4); test_attention(33, 256, 16); test_attention(20, 256, 6); test_attention(45, 256, 32); test_attention(3, 512, 32); test_attention(45, 256, 32); test_attention(3, 512, 32);
    // gelu
    float g[100], u[100], o[100];
    for (int i = 0; i < 100; i++) { g[i] = rndg() * 3; u[i] = rndg(); }
    gelu_mul(g, u, o, 100);
    for (int i = 0; i < 100; i++) check("gelu", o[i], 0.5 * g[i] * (1 + tanh(0.7978845608028654 * g[i] * (1 + 0.044715 * g[i] * g[i]))) * u[i], 1e-5);
    // half floats
    float hx[40]; uint16_t hy[40];
    for (int i = 0; i < 40; i++) hx[i] = rndg() * 10;
    f32_to_f16_row(hx, hy, 40);
    for (int i = 0; i < 40; i++) check("f16", f16_to_f32(hy[i]), hx[i], 1e-3);
    float a[70], b[70]; double dd = 0;
    for (int i = 0; i < 70; i++) { a[i] = rndg(); b[i] = rndg(); dd += (double)a[i] * b[i]; }
    check("dot_f32", dot_f32(a, b, 70), dd, 1e-5);
    printf(fails ? "%d FAILURES\n" : "all kernel checks passed\n", fails);
    return fails != 0;
}
