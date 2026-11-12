// Minimal GGUF (v2/v3) reader over mmap. No copies: tensors point into the file mapping.
#define _GNU_SOURCE
#include "tern.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { const char *key; int key_len; uint32_t type; const uint8_t *val; } gguf_kv;

struct gguf {
    const uint8_t *base; size_t size;
    gguf_kv *kvs; uint64_t n_kv;
    gguf_tensor *tensors; uint64_t n_tensors;
};

static uint32_t rd32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t rd64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }

static const size_t scalar_size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

static void skip_value(const uint8_t **p, uint32_t type) {
    if (type == 8) { uint64_t n = rd64(p); *p += n; }
    else if (type == 9) {
        uint32_t et = rd32(p); uint64_t n = rd64(p);
        if (et == 8) { for (uint64_t i = 0; i < n; i++) { uint64_t l = rd64(p); *p += l; } }
        else *p += n * scalar_size[et];
    } else *p += scalar_size[type];
}

gguf *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st; fstat(fd, &st);
    const uint8_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { perror("mmap"); return NULL; }
    if (memcmp(base, "GGUF", 4) != 0) { fprintf(stderr, "%s: not a GGUF file\n", path); return NULL; }
    gguf *g = calloc(1, sizeof *g);
    g->base = base; g->size = st.st_size;
    const uint8_t *p = base + 4;
    uint32_t version = rd32(&p);
    if (version < 2) { fprintf(stderr, "GGUF version %u not supported\n", version); return NULL; }
    g->n_tensors = rd64(&p); g->n_kv = rd64(&p);
    g->kvs = calloc(g->n_kv, sizeof *g->kvs);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        uint64_t kl = rd64(&p);
        g->kvs[i].key = (const char *)p; g->kvs[i].key_len = (int)kl; p += kl;
        g->kvs[i].type = rd32(&p);
        g->kvs[i].val = p;
        skip_value(&p, g->kvs[i].type);
    }
    g->tensors = calloc(g->n_tensors, sizeof *g->tensors);
    uint64_t *offs = calloc(g->n_tensors, sizeof *offs);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        uint64_t nl = rd64(&p);
        t->name = (const char *)p; t->name_len = (int)nl; p += nl;
        t->n_dims = (int)rd32(&p);
        for (int d = 0; d < 4; d++) t->dims[d] = 1;
        for (int d = 0; d < t->n_dims; d++) t->dims[d] = rd64(&p);
        t->type = rd32(&p);
        offs[i] = rd64(&p);
    }
    uint32_t align = 32; gguf_get_u32(g, "general.alignment", &align);
    size_t data0 = (size_t)(p - base);
    data0 = (data0 + align - 1) / align * align;
    for (uint64_t i = 0; i < g->n_tensors; i++) g->tensors[i].data = base + data0 + offs[i];
    free(offs);
    return g;
}

static const gguf_kv *find_kv(const gguf *g, const char *key) {
    int kl = (int)strlen(key);
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kvs[i].key_len == kl && memcmp(g->kvs[i].key, key, kl) == 0) return &g->kvs[i];
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf *g, const char *name) {
    int nl = (int)strlen(name);
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].name_len == nl && memcmp(g->tensors[i].name, name, nl) == 0) return &g->tensors[i];
    return NULL;
}

int gguf_get_u32(const gguf *g, const char *key, uint32_t *out) {
    const gguf_kv *kv = find_kv(g, key);
    if (!kv) return 0;
    switch (kv->type) {
        case 0: case 7: *out = kv->val[0]; return 1;
        case 1: *out = (uint32_t)(int8_t)kv->val[0]; return 1;
        case 2: { uint16_t v; memcpy(&v, kv->val, 2); *out = v; return 1; }
        case 4: case 5: memcpy(out, kv->val, 4); return 1;
        case 10: case 11: { uint64_t v; memcpy(&v, kv->val, 8); *out = (uint32_t)v; return 1; }
        default: return 0;
    }
}

int gguf_get_f32(const gguf *g, const char *key, float *out) {
    const gguf_kv *kv = find_kv(g, key);
    if (!kv) return 0;
    if (kv->type == 6) { memcpy(out, kv->val, 4); return 1; }
    if (kv->type == 12) { double v; memcpy(&v, kv->val, 8); *out = (float)v; return 1; }
    return 0;
}

int64_t gguf_get_array(const gguf *g, const char *key, uint32_t *elem_type, const uint8_t **first) {
    const gguf_kv *kv = find_kv(g, key);
    if (!kv || kv->type != 9) return -1;
    const uint8_t *p = kv->val;
    *elem_type = rd32(&p);
    uint64_t n = rd64(&p);
    *first = p;
    return (int64_t)n;
}

const char *gguf_get_str(const gguf *g, const char *key, uint64_t *len) {
    const gguf_kv *kv = find_kv(g, key);
    if (!kv || kv->type != 8) return NULL;
    const uint8_t *p = kv->val;
    *len = rd64(&p);
    return (const char *)p;
}

// Drop the file pages read so far from this process (they stay in the system's page cache and come back on
// access). Called after loading: the 4-bit matrices have been repacked and are not read from the file again.
void gguf_release_pages(const gguf *g) {
    madvise((void *)g->base, g->size, MADV_DONTNEED);
}
