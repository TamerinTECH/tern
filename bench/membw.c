// Measures (1) streaming memory bandwidth and (2) int8 SDOT throughput in-cache, per thread count.
// build: gcc -O3 -mcpu=native -pthread membw.c -o membw
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { int8_t *w; int8_t *x; size_t n; int reps; int nx; int64_t out; } job_t;

// stream: read n bytes of w, sdot against a small x (in L1), reps times
static void *stream_job(void *p) {
    job_t *j = p; int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < j->reps; r++) {
        const int8_t *w = j->w; const int8_t *x = j->x;
        for (size_t i = 0; i < j->n; i += 64) {
            size_t k = i & 4032; // x is 4096 bytes
            a0 = vdotq_s32(a0, vld1q_s8(w + i), vld1q_s8(x + k));
            a1 = vdotq_s32(a1, vld1q_s8(w + i + 16), vld1q_s8(x + k + 16));
            a2 = vdotq_s32(a2, vld1q_s8(w + i + 32), vld1q_s8(x + k + 32));
            a3 = vdotq_s32(a3, vld1q_s8(w + i + 48), vld1q_s8(x + k + 48));
        }
    }
    j->out = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
    return NULL;
}

static void run(const char *name, size_t bytes_per_thread, int reps, int nt) {
    pthread_t th[8]; job_t jb[8];
    for (int t = 0; t < nt; t++) {
        jb[t].n = bytes_per_thread; jb[t].reps = reps;
        if (posix_memalign((void **)&jb[t].w, 64, bytes_per_thread)) exit(1);
        if (posix_memalign((void **)&jb[t].x, 64, 4096)) exit(1);
        for (size_t i = 0; i < bytes_per_thread; i++) jb[t].w[i] = (int8_t)(i * 31 + t);
        memset(jb[t].x, 1, 4096);
    }
    double t0 = now();
    for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, stream_job, &jb[t]);
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    double dt = now() - t0;
    double total = (double)bytes_per_thread * reps * nt;
    printf("%-28s threads=%d  %.2f GB/s  = %.2f GMAC/s (int8 sdot)\n", name, nt, total / dt / 1e9, total / dt / 1e9);
    for (int t = 0; t < nt; t++) { free(jb[t].w); free(jb[t].x); }
}

int main(void) {
    for (int nt = 1; nt <= 4; nt++) run("DRAM stream 256MB/thread", 256u << 20, 4, nt);
    for (int nt = 1; nt <= 4; nt += 3) run("L2-resident 512KB/thread", 512u << 10, 4000, nt);
    for (int nt = 1; nt <= 4; nt += 3) run("L1-resident 32KB/thread", 32u << 10, 60000, nt);
    return 0;
}
