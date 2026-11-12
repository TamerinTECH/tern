// Cost of one parallel region of the thread pool with nothing to do (the hand-off overhead).
// usage: poolbench [threads]
#include "../src/tern.h"
#include <stdio.h>
#include <stdlib.h>
static void nop(void *arg, int tid) { (void)arg; (void)tid; }
int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 4;
    pool_init(nt);
    for (int i = 0; i < 1000; i++) pool_run(nop, NULL);
    const int reps = 200000;
    double t = now_sec();
    for (int i = 0; i < reps; i++) pool_run(nop, NULL);
    printf("%d threads: %.2f microseconds per empty parallel region\n", nt, (now_sec() - t) / reps * 1e6);
    return 0;
}
