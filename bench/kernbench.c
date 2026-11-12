// Speed of the 4-bit matrix kernel on matrices shaped like Gemma 4 E2B's: multiply-accumulates per second
// for 1..32 slots, on 1..N threads. No model needed.
// usage: kernbench [threads] [chunk_blocks] [slot_group]
#include "../src/tern.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int tern_chunk_blocks, tern_slot_group;
typedef struct { const pmat *M; const qvec *x; int S; float *const *y; } job;
static void run(void *arg, int tid) {
    job *j = arg; int nt = pool_threads();
    int r0 = (int)((int64_t)j->M->rows * tid / nt), r1 = (int)((int64_t)j->M->rows * (tid + 1) / nt);
    matvec_p4(j->M, r0, r1, j->x, j->S, j->y);
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 1;
    if (argc > 2) tern_chunk_blocks = atoi(argv[2]);
    if (argc > 3) tern_slot_group = atoi(argv[3]);
    pool_init(nt);
    printf("kernels: %s, %d threads, column chunk %d blocks, slot group %d\n", kernels_name(), nt, tern_chunk_blocks, tern_slot_group);
    static const int shapes[][2] = {{16, 1536}, {12288, 1536}, {1536, 12288}, {1536, 4096}, {24576, 1536}};
    static const char *names[] = {"in cache 16x1536", "up/gate 12288x1536", "down 1536x12288", "attn out 1536x4096", "gate+up 24576x1536"};
    for (int k = 0; k < 5; k++) {
        int rows = shapes[k][0], cols = shapes[k][1];
        pmat M = {NULL, rows, cols, P4, p4_row_bytes(cols, P4)};
        size_t bytes = M.stride * (size_t)rows;
        uint8_t *W = aligned_alloc(64, (bytes + 127) / 64 * 64), *src = malloc(quant_row_bytes(GT_Q4_0, cols));
        for (int r = 0; r < rows; r++) {
            for (size_t i = 0; i < quant_row_bytes(GT_Q4_0, cols); i++) src[i] = (uint8_t)((i * 2654435761u + (unsigned)r * 40503u) >> 13);
            for (int b = 0; b < cols / 32; b++) { uint16_t h = 0x3C00; memcpy(src + (size_t)b * 18, &h, 2); }
            p4_pack_row(GT_Q4_0, src, cols, W + M.stride * (size_t)r);
        }
        M.W = W;
        qvec x[TERN_MAX_SLOTS]; float *y[TERN_MAX_SLOTS]; float *xf = malloc(sizeof(float) * (size_t)cols);
        for (int s = 0; s < TERN_MAX_SLOTS; s++) {
            qvec_alloc(&x[s], cols); y[s] = aligned_alloc(64, sizeof(float) * (size_t)((rows + 15) / 16 * 16));
            for (int i = 0; i < cols; i++) xf[i] = (float)((i * 37 + s * 11) % 97) - 48.0f;
            quantize_q8(xf, cols, &x[s]);
        }
        for (int S = 1; S <= TERN_MAX_SLOTS; S *= 2) {
            job j = {&M, x, S, y};
            int reps = (int)(3e9 * nt / ((double)rows * cols * S)) + 1;
            pool_run(run, &j);
            double t = now_sec();
            for (int r = 0; r < reps; r++) pool_run(run, &j);
            double dt = now_sec() - t;
            printf("%-20s %5.1f MB  slots %2d: %7.1f GMAC/s  %6.2f GB/s of weights  %7.3f ms per pass\n", names[k], bytes / 1e6, S,
                   (double)rows * cols * S * reps / dt / 1e9, (double)bytes * reps / dt / 1e9, dt / reps * 1e3);
        }
        free(W); free(src); free(xf);
        for (int s = 0; s < TERN_MAX_SLOTS; s++) free(y[s]);
    }
    return 0;
}
