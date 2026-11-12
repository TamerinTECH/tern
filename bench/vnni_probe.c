// What does this CPU sustain for byte dot products in the first-level cache? Variants of the inner loop:
// 512-bit vs 256-bit instructions, activation operand from memory vs register.
// build: gcc -O3 -march=native vnni_probe.c -o vnni_probe
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint8_t W[4096] __attribute__((aligned(64)));
static int8_t  X[4096] __attribute__((aligned(64)));

#define REPORT(name, macs) printf("%-58s %7.1f GMAC/s\n", name, (macs) / (now() - t0) / 1e9)

int main(void) {
    for (int i = 0; i < 4096; i++) { W[i] = (uint8_t)(i * 7 & 15); X[i] = (int8_t)(i * 13); }
    const long reps = 4000000;
    volatile int64_t sink = 0;
    double t0;

    {   // A: zmm, weights in registers, activations from memory, 8 per group, 2 chains (the kernel's inner part)
        __m512i w[8]; for (int k = 0; k < 8; k++) w[k] = _mm512_load_si512(W + 64 * k);
        __m512i tot = _mm512_setzero_si512();
        t0 = now();
        for (long r = 0; r < reps; r++) {
            const int8_t *a = X + ((r & 7) << 9);
            __m512i i0 = _mm512_setzero_si512(), i1 = i0;
            i0 = _mm512_dpbusd_epi32(i0, w[0], _mm512_load_si512(a));       i1 = _mm512_dpbusd_epi32(i1, w[1], _mm512_load_si512(a + 64));
            i0 = _mm512_dpbusd_epi32(i0, w[2], _mm512_load_si512(a + 128)); i1 = _mm512_dpbusd_epi32(i1, w[3], _mm512_load_si512(a + 192));
            i0 = _mm512_dpbusd_epi32(i0, w[4], _mm512_load_si512(a + 256)); i1 = _mm512_dpbusd_epi32(i1, w[5], _mm512_load_si512(a + 320));
            i0 = _mm512_dpbusd_epi32(i0, w[6], _mm512_load_si512(a + 384)); i1 = _mm512_dpbusd_epi32(i1, w[7], _mm512_load_si512(a + 448));
            tot = _mm512_add_epi32(tot, _mm512_add_epi32(i0, i1));
        }
        sink += _mm512_reduce_add_epi32(tot);
        REPORT("A zmm: 8 dpbusd (mem operand), 2 chains, per group", 512.0 * reps);
    }
    {   // B: same with 4 independent chains
        __m512i w[8]; for (int k = 0; k < 8; k++) w[k] = _mm512_load_si512(W + 64 * k);
        __m512i tot = _mm512_setzero_si512();
        t0 = now();
        for (long r = 0; r < reps; r++) {
            const int8_t *a = X + ((r & 7) << 9);
            __m512i i0 = _mm512_setzero_si512(), i1 = i0, i2 = i0, i3 = i0;
            i0 = _mm512_dpbusd_epi32(i0, w[0], _mm512_load_si512(a));       i1 = _mm512_dpbusd_epi32(i1, w[1], _mm512_load_si512(a + 64));
            i2 = _mm512_dpbusd_epi32(i2, w[2], _mm512_load_si512(a + 128)); i3 = _mm512_dpbusd_epi32(i3, w[3], _mm512_load_si512(a + 192));
            i0 = _mm512_dpbusd_epi32(i0, w[4], _mm512_load_si512(a + 256)); i1 = _mm512_dpbusd_epi32(i1, w[5], _mm512_load_si512(a + 320));
            i2 = _mm512_dpbusd_epi32(i2, w[6], _mm512_load_si512(a + 384)); i3 = _mm512_dpbusd_epi32(i3, w[7], _mm512_load_si512(a + 448));
            tot = _mm512_add_epi32(tot, _mm512_add_epi32(_mm512_add_epi32(i0, i1), _mm512_add_epi32(i2, i3)));
        }
        sink += _mm512_reduce_add_epi32(tot);
        REPORT("B zmm: 8 dpbusd (mem operand), 4 chains", 512.0 * reps);
    }
    {   // C: zmm, both operands in registers (upper bound of the multiplier ports)
        __m512i w[8], x[8]; for (int k = 0; k < 8; k++) { w[k] = _mm512_load_si512(W + 64 * k); x[k] = _mm512_load_si512(X + 64 * k); }
        __m512i tot = _mm512_setzero_si512();
        t0 = now();
        for (long r = 0; r < reps; r++) {
            __m512i i0 = _mm512_setzero_si512(), i1 = i0, i2 = i0, i3 = i0;
            i0 = _mm512_dpbusd_epi32(i0, w[0], x[0]); i1 = _mm512_dpbusd_epi32(i1, w[1], x[1]);
            i2 = _mm512_dpbusd_epi32(i2, w[2], x[2]); i3 = _mm512_dpbusd_epi32(i3, w[3], x[3]);
            i0 = _mm512_dpbusd_epi32(i0, w[4], x[4]); i1 = _mm512_dpbusd_epi32(i1, w[5], x[5]);
            i2 = _mm512_dpbusd_epi32(i2, w[6], x[6]); i3 = _mm512_dpbusd_epi32(i3, w[7], x[7]);
            tot = _mm512_add_epi32(tot, _mm512_add_epi32(_mm512_add_epi32(i0, i1), _mm512_add_epi32(i2, i3)));
            x[0] = _mm512_xor_si512(x[0], tot);
        }
        sink += _mm512_reduce_add_epi32(tot);
        REPORT("C zmm: 8 dpbusd (registers only), 4 chains", 512.0 * reps);
    }
    {   // D: ymm, 16 dpbusd per 512 weights, activations from memory, 4 chains
        __m256i w[16]; for (int k = 0; k < 16; k++) w[k] = _mm256_load_si256((const void *)(W + 32 * k));
        __m256i tot = _mm256_setzero_si256();
        t0 = now();
        for (long r = 0; r < reps; r++) {
            const int8_t *a = X + ((r & 7) << 9);
            __m256i i0 = _mm256_setzero_si256(), i1 = i0, i2 = i0, i3 = i0;
            for (int k = 0; k < 16; k += 4) {
                i0 = _mm256_dpbusd_epi32(i0, w[k], _mm256_load_si256((const void *)(a + 32 * k)));
                i1 = _mm256_dpbusd_epi32(i1, w[k + 1], _mm256_load_si256((const void *)(a + 32 * k + 32)));
                i2 = _mm256_dpbusd_epi32(i2, w[k + 2], _mm256_load_si256((const void *)(a + 32 * k + 64)));
                i3 = _mm256_dpbusd_epi32(i3, w[k + 3], _mm256_load_si256((const void *)(a + 32 * k + 96)));
            }
            tot = _mm256_add_epi32(tot, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        }
        sink += _mm256_extract_epi32(tot, 0);
        REPORT("D ymm: 16 dpbusd (mem operand), 4 chains", 512.0 * reps);
    }
    {   // E: zmm, full kernel step for one slot: 8 dpbusd + add + cvt + mul + fma (scales from memory)
        __m512i w[8]; for (int k = 0; k < 8; k++) w[k] = _mm512_load_si512(W + 64 * k);
        static float xd[64] __attribute__((aligned(64))); static int32_t xk[64] __attribute__((aligned(64)));
        for (int i = 0; i < 64; i++) { xd[i] = 0.01f * (float)(i + 1); xk[i] = i; }
        __m512 acc = _mm512_setzero_ps(); const __m512 dw = _mm512_set1_ps(0.5f);
        t0 = now();
        for (long r = 0; r < reps; r++) {
            const int8_t *a = X + ((r & 7) << 9);
            __m512i i0 = _mm512_load_si512(xk + (r & 3) * 16), i1 = _mm512_setzero_si512();
            i0 = _mm512_dpbusd_epi32(i0, w[0], _mm512_load_si512(a));       i1 = _mm512_dpbusd_epi32(i1, w[1], _mm512_load_si512(a + 64));
            i0 = _mm512_dpbusd_epi32(i0, w[2], _mm512_load_si512(a + 128)); i1 = _mm512_dpbusd_epi32(i1, w[3], _mm512_load_si512(a + 192));
            i0 = _mm512_dpbusd_epi32(i0, w[4], _mm512_load_si512(a + 256)); i1 = _mm512_dpbusd_epi32(i1, w[5], _mm512_load_si512(a + 320));
            i0 = _mm512_dpbusd_epi32(i0, w[6], _mm512_load_si512(a + 384)); i1 = _mm512_dpbusd_epi32(i1, w[7], _mm512_load_si512(a + 448));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(i0, i1)), _mm512_mul_ps(dw, _mm512_load_ps(xd + (r & 3) * 16)), acc);
        }
        sink += (int64_t)_mm512_reduce_add_ps(acc);
        REPORT("E zmm: full step (8 dpbusd, add, cvt, mul, fma)", 512.0 * reps);
    }
    {   // F: as E for 4 slots interleaved (4 independent accumulators), weights unpacked from nibbles once per group
        static float xd[64] __attribute__((aligned(64))); static int32_t xk[64] __attribute__((aligned(64)));
        for (int i = 0; i < 64; i++) { xd[i] = 0.01f * (float)(i + 1); xk[i] = i; }
        __m512 acc[4]; for (int s = 0; s < 4; s++) acc[s] = _mm512_setzero_ps();
        const __m512 dw = _mm512_set1_ps(0.5f); const __m512i m4 = _mm512_set1_epi8(15);
        t0 = now();
        for (long r = 0; r < reps / 4; r++) {
            const uint8_t *g = W + ((r & 7) << 8);
            __m512i lo[4], hi[4];
            for (int u = 0; u < 4; u++) { __m512i v = _mm512_load_si512(g + 64 * u); lo[u] = _mm512_and_si512(v, m4); hi[u] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4); }
            for (int s = 0; s < 4; s++) {
                const int8_t *a = X + (((r + s) & 7) << 9);
                __m512i i0 = _mm512_load_si512(xk + s * 16), i1 = _mm512_setzero_si512();
                i0 = _mm512_dpbusd_epi32(i0, lo[0], _mm512_load_si512(a));       i1 = _mm512_dpbusd_epi32(i1, hi[0], _mm512_load_si512(a + 64));
                i0 = _mm512_dpbusd_epi32(i0, lo[1], _mm512_load_si512(a + 128)); i1 = _mm512_dpbusd_epi32(i1, hi[1], _mm512_load_si512(a + 192));
                i0 = _mm512_dpbusd_epi32(i0, lo[2], _mm512_load_si512(a + 256)); i1 = _mm512_dpbusd_epi32(i1, hi[2], _mm512_load_si512(a + 320));
                i0 = _mm512_dpbusd_epi32(i0, lo[3], _mm512_load_si512(a + 384)); i1 = _mm512_dpbusd_epi32(i1, hi[3], _mm512_load_si512(a + 448));
                acc[s] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(i0, i1)), _mm512_mul_ps(dw, _mm512_load_ps(xd + s * 16)), acc[s]);
            }
        }
        sink += (int64_t)_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])));
        REPORT("F zmm: 4 slots per group incl. nibble unpack", 512.0 * reps);
    }
    for (int variant = 0; variant < 4; variant++) {
        // G: F with weight vectors 32 bytes off a cache line boundary (every 64-byte load spans two lines)
        // H: F plus a horizontal sum and a store after every 3 groups (a row of 1536 weights)
        // I: F plus the 16 block scales converted from half floats per group
        // J: all three (what the engine's kernel does)
        static float xd[64] __attribute__((aligned(64))); static int32_t xk[64] __attribute__((aligned(64)));
        static uint8_t W2[8192 + 64] __attribute__((aligned(64))); static float y[4][1024];
        for (int i = 0; i < 64; i++) { xd[i] = 0.01f * (float)(i + 1); xk[i] = i; }
        for (int i = 0; i < 8192 + 64; i++) W2[i] = (uint8_t)(i * 7);
        const int off = (variant == 0 || variant == 3) ? 32 : 0, red = variant == 1 || variant == 3, sc = variant == 2 || variant == 3;
        __m512 acc[4]; for (int s = 0; s < 4; s++) acc[s] = _mm512_setzero_ps();
        const __m512i m4 = _mm512_set1_epi8(15);
        __m512 dw = _mm512_set1_ps(0.5f);
        t0 = now();
        for (long r = 0; r < reps / 4; r++) {
            const uint8_t *g = W2 + off + ((r % 24) * 288);
            if (sc) dw = _mm512_cvtph_ps(_mm256_loadu_si256((const void *)(g - 32 + (off ? 0 : 32))));
            __m512i lo[4], hi[4];
            for (int u = 0; u < 4; u++) { __m512i v = _mm512_loadu_si512(g + 64 * u); lo[u] = _mm512_and_si512(v, m4); hi[u] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4); }
            for (int s = 0; s < 4; s++) {
                const int8_t *a = X + (((r + s) & 7) << 9);
                __m512i i0 = _mm512_load_si512(xk + s * 16), i1 = _mm512_setzero_si512();
                i0 = _mm512_dpbusd_epi32(i0, lo[0], _mm512_load_si512(a));       i1 = _mm512_dpbusd_epi32(i1, hi[0], _mm512_load_si512(a + 64));
                i0 = _mm512_dpbusd_epi32(i0, lo[1], _mm512_load_si512(a + 128)); i1 = _mm512_dpbusd_epi32(i1, hi[1], _mm512_load_si512(a + 192));
                i0 = _mm512_dpbusd_epi32(i0, lo[2], _mm512_load_si512(a + 256)); i1 = _mm512_dpbusd_epi32(i1, hi[2], _mm512_load_si512(a + 320));
                i0 = _mm512_dpbusd_epi32(i0, lo[3], _mm512_load_si512(a + 384)); i1 = _mm512_dpbusd_epi32(i1, hi[3], _mm512_load_si512(a + 448));
                acc[s] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(i0, i1)), _mm512_mul_ps(dw, _mm512_load_ps(xd + s * 16)), acc[s]);
            }
            if (red && r % 3 == 2) for (int s = 0; s < 4; s++) { y[s][(r / 3) & 1023] = _mm512_reduce_add_ps(acc[s]); acc[s] = _mm512_setzero_ps(); }
        }
        sink += (int64_t)_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]))) + (int64_t)y[0][5];
        static const char *names[] = {"G F + weights off the line boundary", "H F + sum and store per 3 groups", "I F + scales from half floats", "J F + all three (the engine's kernel)"};
        REPORT(names[variant], 512.0 * reps);
    }
    return (int)(sink & 1);
}
