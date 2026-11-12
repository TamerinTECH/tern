// x86 twin of membw.c: (1) streaming memory bandwidth and (2) int8 dot-product (AVX-512 VNNI or AVX2) throughput
// in-cache, per thread count. Threads are pinned to distinct physical cores first, then to their siblings.
// build: gcc -O3 -march=native -pthread membw_x86.c -o membw_x86
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { uint8_t *w; int8_t *x; size_t n; int reps; int cpu; int64_t out; } job_t;
static pthread_barrier_t bar;

static void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

// read n bytes of w, byte dot product against a small x (in L1), reps times
static void *stream_job(void *p) {
    job_t *j = p; pin(j->cpu);
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    pthread_barrier_wait(&bar);
    for (int r = 0; r < j->reps; r++) {
        const uint8_t *w = j->w; const int8_t *x = j->x;
        for (size_t i = 0; i < j->n; i += 256) {
            size_t k = i & 3840;   // x is 4096 bytes
            a0 = _mm512_dpbusd_epi32(a0, _mm512_load_si512(w + i), _mm512_load_si512(x + k));
            a1 = _mm512_dpbusd_epi32(a1, _mm512_load_si512(w + i + 64), _mm512_load_si512(x + k + 64));
            a2 = _mm512_dpbusd_epi32(a2, _mm512_load_si512(w + i + 128), _mm512_load_si512(x + k + 128));
            a3 = _mm512_dpbusd_epi32(a3, _mm512_load_si512(w + i + 192), _mm512_load_si512(x + k + 192));
        }
    }
    j->out = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
#else
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    pthread_barrier_wait(&bar);
    for (int r = 0; r < j->reps; r++) {
        const uint8_t *w = j->w; const int8_t *x = j->x;
        for (size_t i = 0; i < j->n; i += 128) {
            size_t k = i & 3968;
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const void *)(w + i)), _mm256_load_si256((const void *)(x + k))), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const void *)(w + i + 32)), _mm256_load_si256((const void *)(x + k + 32))), ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const void *)(w + i + 64)), _mm256_load_si256((const void *)(x + k + 64))), ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const void *)(w + i + 96)), _mm256_load_si256((const void *)(x + k + 96))), ones));
        }
    }
    a0 = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
    j->out = _mm256_extract_epi32(a0, 0) + _mm256_extract_epi32(a0, 4);
#endif
    return NULL;
}

static int ncpu, smt;

static void run(const char *name, size_t bytes_per_thread, int reps, int nt) {
    pthread_t th[64]; job_t jb[64];
    for (int t = 0; t < nt; t++) {
        jb[t].n = bytes_per_thread; jb[t].reps = reps;
        // physical cores first (even logical ids when SMT siblings are adjacent), then the siblings
        jb[t].cpu = smt ? (t < ncpu / 2 ? 2 * t : 2 * (t - ncpu / 2) + 1) : t;
        if (posix_memalign((void **)&jb[t].w, 64, bytes_per_thread)) exit(1);
        if (posix_memalign((void **)&jb[t].x, 64, 4096)) exit(1);
        for (size_t i = 0; i < bytes_per_thread; i++) jb[t].w[i] = (uint8_t)((i * 31 + t) & 15);
        memset(jb[t].x, 1, 4096);
    }
    pthread_barrier_init(&bar, NULL, nt + 1);
    for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, stream_job, &jb[t]);
    pthread_barrier_wait(&bar);
    double t0 = now();
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    double dt = now() - t0;
    double total = (double)bytes_per_thread * reps * nt;
    printf("%-26s threads=%2d  %7.2f GB/s  = %7.2f GMAC/s (int8 vpdpbusd)\n", name, nt, total / dt / 1e9, total / dt / 1e9);
    for (int t = 0; t < nt; t++) { free(jb[t].w); free(jb[t].x); }
}

int main(int argc, char **argv) {
    ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    smt = argc > 1 ? atoi(argv[1]) : 1;     // 1: logical cpus 2k and 2k+1 share a core (the usual x86 numbering); 0: every cpu is a core
    int counts[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}, n = 0;
    for (int i = 0; i < 10; i++) if (counts[i] <= ncpu) n = i + 1;
    for (int i = 0; i < n; i++) run("DRAM stream 256MB/thread", 256u << 20, 4, counts[i]);
    for (int i = 0; i < n; i++) run("L2-resident 256KB/thread", 256u << 10, 20000, counts[i]);
    for (int i = 0; i < n; i++) run("L1-resident 16KB/thread", 16u << 10, 300000, counts[i]);
    return 0;
}
