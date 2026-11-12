// How fast do all threads together read memory when the work comes in short parallel regions?
// A model's pass is a few hundred parallel regions; in each one every thread reads its share of one matrix
// (E2B: 2 - 10 MB per region, 31B: 20 - 250 MB). This reads a buffer (2 GB) once per region size, in regions of
// that size, every thread summing the bytes of its share, and prints the rate.
// Where the memory was first written matters on machines with more than one memory node (also inside a virtual
// machine that does not show its nodes): `touch` chooses who writes each page first.
// usage: burst [threads] [touch: 0 = the calling thread, 1 = every thread its share of the buffer,
//                                2 = the 2 MB pages in turn by all threads] [buffer MB, default 2048] [buffers, default 1]
#define _GNU_SOURCE
#include "../src/tern.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static const uint8_t *buf; static size_t r_off, r_len; static int nt; static volatile uint64_t sink[64];
static uint8_t *wbuf; static size_t total; static int touch_mode;

static void touch(void *arg, int tid) {
    (void)arg;
    if (touch_mode == 1) {
        size_t a = total * (size_t)tid / (size_t)nt & ~(size_t)4095, b = total * (size_t)(tid + 1) / (size_t)nt & ~(size_t)4095;
        for (size_t i = a; i < b; i += 4096) wbuf[i] = (uint8_t)i;
    } else {
        const size_t page = (size_t)2 << 20;
        for (size_t pg = (size_t)tid; pg * page < total; pg += (size_t)nt)
            for (size_t i = pg * page; i < (pg + 1) * page && i < total; i += 4096) wbuf[i] = (uint8_t)i;
    }
}

static void run(void *arg, int tid) {
    (void)arg;
    size_t a = r_off + r_len * (size_t)tid / (size_t)nt, b = r_off + r_len * (size_t)(tid + 1) / (size_t)nt;
    a &= ~(size_t)63; b &= ~(size_t)63;
    uint64_t s = 0;
#ifdef __AVX2__
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = a; i < b; i += 64) {
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_load_si256((const void *)(buf + i)), _mm256_setzero_si256()));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_load_si256((const void *)(buf + i + 32)), _mm256_setzero_si256()));
    }
    s = (uint64_t)_mm256_extract_epi64(acc, 0) + (uint64_t)_mm256_extract_epi64(acc, 1) + (uint64_t)_mm256_extract_epi64(acc, 2) + (uint64_t)_mm256_extract_epi64(acc, 3);
#else
    for (size_t i = a; i < b; i += 8) { uint64_t v; memcpy(&v, buf + i, 8); s += v; }
#endif
    sink[tid] += s;
}

// Where in physical memory did the buffer land? (needs root: the kernel hides page frame numbers otherwise)
static void where(const uint8_t *m, size_t n) {
    FILE *f = fopen("/proc/self/pagemap", "rb");
    if (!f) return;
    enum { BINS = 16 };
    long hist[BINS] = {0}, seen = 0; uint64_t maxpfn = 0, e;
    FILE *mi = fopen("/proc/meminfo", "r"); char line[128]; unsigned long long total_kb = 0;
    while (mi && fgets(line, sizeof line, mi)) if (sscanf(line, "MemTotal: %llu kB", &total_kb) == 1) break;
    if (mi) fclose(mi);
    const uint64_t top = (total_kb / 4 + (1 << 20)) | 1;       // page frames, plus 4 GB for the hole below 4 GB
    for (size_t i = 0; i < n; i += (size_t)1 << 21) {            // one page per 2 MB
        if (fseek(f, (long)(((uintptr_t)(m + i) >> 12) * 8), SEEK_SET) || fread(&e, 8, 1, f) != 1) break;
        uint64_t pfn = e & (((uint64_t)1 << 55) - 1);
        if (!(e >> 63) || !pfn) continue;
        if (pfn > maxpfn) maxpfn = pfn;
        hist[pfn * BINS / top < BINS ? pfn * BINS / top : BINS - 1]++; seen++;
    }
    fclose(f);
    if (!seen) { printf("physical placement: not visible (run as root)\n"); return; }
    printf("physical placement, share of the buffer in each sixteenth of the machine's memory (low to high):");
    for (int b = 0; b < BINS; b++) printf(" %.0f%%", 100.0 * hist[b] / seen);
    printf("\n");
}

int main(int argc, char **argv) {
    nt = argc > 1 ? atoi(argv[1]) : 4;
    touch_mode = argc > 2 ? atoi(argv[2]) : 0;
    pool_init(nt);
    total = (size_t)(argc > 3 ? atoi(argv[3]) : 2048) << 20;        // buffer size in MB
    const int buffers = argc > 4 ? atoi(argv[4]) : 1;                // more than one: each is allocated while the earlier ones are kept
    for (int nb = 0; nb < buffers; nb++) {
        uint8_t *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        if (!getenv("BURST_SMALL_PAGES")) madvise(m, total, MADV_HUGEPAGE);        // BURST_SMALL_PAGES=1: 4 KB pages
        buf = m; r_off = 0; r_len = total;
        wbuf = m;
        if (touch_mode == 0) for (size_t i = 0; i < total; i += 4096) m[i] = (uint8_t)i;
        else pool_run(touch, NULL);
        pool_run(run, NULL);
        where(m, total);
        const double mb[] = {1, 2, 5, 10, 20, 70, 250, 2048};
        for (int k = 0; k < 8; k++) {
            r_len = (size_t)(mb[k] * 1048576.0);
            if (r_len > total) continue;
            double best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                double t = now_sec();
                for (r_off = 0; r_off + r_len <= total; r_off += r_len) pool_run(run, NULL);
                t = now_sec() - t; if (t < best) best = t;
            }
            printf("buffer %d: %d threads, touch %d, regions of %6.0f MB (%7.0f KB per thread): %6.1f GB/s, %7.1f microseconds per region\n",
                   nb + 1, nt, touch_mode, mb[k], mb[k] * 1024 / nt, (double)(total / r_len * r_len) / best / 1e9, best / (double)(total / r_len) * 1e6);
        }
    }
    return 0;
}
