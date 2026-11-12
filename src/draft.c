// Cheap guesses of the next token, for speculative decoding.
//
// A pass over the weights costs about the same for 1 token as for 4 (memory-bound), so the engine evaluates the
// token it is sure of together with a few guessed followers; every guess that the model confirms is a token
// that needed no pass of its own. The output is unchanged: a wrong guess is simply thrown away.
// Sources of guesses, in this order:
//   1. copy:   the last few tokens occurred earlier in this context -> what followed them there
//   2. n-gram: table built from text the model generated before (tools/build_draft.py)
#define _GNU_SOURCE
#include "tern.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_ORDER 8

struct drafter {
    int order;                              // longest context in the table
    uint64_t cap[MAX_ORDER + 1];            // slots per order (power of two)
    const uint64_t *key[MAX_ORDER + 1];
    const int32_t *next[MAX_ORDER + 1];
    const float *conf[MAX_ORDER + 1];       // share of the occurrences that continued with `next`
    float min_conf;
    int copy_min;                           // shortest repeated run that counts as a copy signal
};

static uint64_t hash_tokens(const int *t, int k) {       // FNV-1a over the little-endian bytes; same in build_draft.py
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < k; i++) {
        uint32_t v = (uint32_t)t[i];
        for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
    }
    return h ? h : 1;
}

drafter *draft_load(const char *path, float min_conf) {
    drafter *d = calloc(1, sizeof *d);
    d->min_conf = min_conf; d->copy_min = 3;
    if (!path) return d;                    // copy-only drafter
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); free(d); return NULL; }
    struct stat st; fstat(fd, &st);
    const uint8_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED || memcmp(base, "TERNNG01", 8)) { fprintf(stderr, "%s: not an n-gram table\n", path); free(d); return NULL; }
    memcpy(&d->order, base + 8, 4);
    if (d->order > MAX_ORDER) d->order = MAX_ORDER;
    size_t off = 16;
    for (int k = 1; k <= d->order; k++) {
        memcpy(&d->cap[k], base + off, 8); off += 8;
        d->key[k] = (const uint64_t *)(base + off); off += 8 * d->cap[k];
        d->next[k] = (const int32_t *)(base + off); off += 4 * d->cap[k];
        d->conf[k] = (const float *)(base + off); off += 4 * d->cap[k];
    }
    return d;
}

// Guess the token after ctx[0..n). Returns -1 when there is no confident guess.
int draft_next(const drafter *d, const int *ctx, int n) {
    // 1. copy: longest earlier occurrence of the context's tail (at least copy_min tokens, at most 16)
    if (n > d->copy_min) {
        int best_len = 0, best_pos = -1;
        int last = ctx[n - 1], lo = n > 4096 ? n - 4096 : 0;
        for (int i = n - 2; i >= lo; i--) {
            if (ctx[i] != last) continue;
            int l = 1;
            while (l < 16 && i - l >= 0 && ctx[i - l] == ctx[n - 1 - l]) l++;
            if (l > best_len) { best_len = l; best_pos = i; if (l == 16) break; }
        }
        if (best_len >= d->copy_min) return ctx[best_pos + 1];
    }
    // 2. n-gram: longest context first
    for (int k = d->order < n ? d->order : n; k >= 1; k--) {
        uint64_t h = hash_tokens(ctx + n - k, k), mask = d->cap[k] - 1;
        for (uint64_t i = h & mask;; i = (i + 1) & mask) {
            uint64_t key = d->key[k][i];
            if (key == 0) break;
            if (key == h) { if (d->conf[k][i] >= d->min_conf) return d->next[k][i]; else break; }
        }
    }
    return -1;
}
