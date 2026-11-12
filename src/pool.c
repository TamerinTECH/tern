// Tiny thread pool for many very short parallel regions (a pass has a few hundred of them, tens of microseconds
// each): workers spin on a generation counter and fall asleep on a futex only after a while without work.
// Threads are pinned to distinct physical cores first (TERN_PIN=0 turns pinning off).
#define _GNU_SOURCE
#include "tern.h"
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define SPIN_BEFORE_YIELD 4000      // then give the core to whoever else wants it between looks
#define SPIN_BEFORE_SLEEP 200000
#define MAX_THREADS 64

static int n_thr = 1;
static pthread_t threads[MAX_THREADS];
static int cpu_of[MAX_THREADS];
// Each of these is written by one side and polled by the other, thousands of times per second: one cache
// line each, so that a worker reporting completion does not disturb the workers still looking for work.
static struct { _Atomic uint32_t generation; pool_fn fn; void *arg; _Atomic int n_sleeping; } job __attribute__((aligned(64)));
static struct { _Atomic uint32_t done; char pad[60]; } done_of[MAX_THREADS] __attribute__((aligned(64)));

static _Atomic(pool_fn) idle_fn;     // what a waiting worker does between two looks for work (NULL: nothing)
void pool_set_idle(pool_fn fn) { atomic_store(&idle_fn, fn); }

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void pin_self(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

// logical cpus this process may use, one per physical core first, then the remaining hardware threads
static int cpu_order(int *order, int max) {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed)) return 0;
    int n = 0, core_seen[1024] = {0}, later[1024], nl = 0;
    for (int c = 0; c < CPU_SETSIZE && c < 1024; c++) {
        if (!CPU_ISSET(c, &allowed)) continue;
        char path[96]; int core = c, pkg = 0;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        FILE *f = fopen(path, "r"); if (f) { if (fscanf(f, "%d", &core) != 1) core = c; fclose(f); }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        f = fopen(path, "r"); if (f) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        int key = (pkg * 256 + core) % 1024;
        if (!core_seen[key]) { core_seen[key] = 1; if (n < max) order[n++] = c; }
        else if (nl < 1024) later[nl++] = c;
    }
    for (int i = 0; i < nl && n < max; i++) order[n++] = later[i];
    return n;
}

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    pin_self(cpu_of[tid]);
    uint32_t seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&job.generation, memory_order_acquire) == seen) {
            pool_fn idle = atomic_load_explicit(&idle_fn, memory_order_relaxed);
            if (idle) idle(NULL, tid);
            cpu_relax();
            if (++spins > SPIN_BEFORE_YIELD) sched_yield();
            if (spins > SPIN_BEFORE_SLEEP) {
                atomic_fetch_add(&job.n_sleeping, 1);
                syscall(SYS_futex, &job.generation, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
                atomic_fetch_sub(&job.n_sleeping, 1);
                spins = 0;
            }
        }
        seen = atomic_load_explicit(&job.generation, memory_order_acquire);
        job.fn(job.arg, tid);
        atomic_store_explicit(&done_of[tid].done, seen, memory_order_release);
    }
    return NULL;
}

void pool_init(int n) {
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    n_thr = n;
    int order[MAX_THREADS], no = 0;
    const char *pin = getenv("TERN_PIN");
    if (!pin || atoi(pin) != 0) no = cpu_order(order, MAX_THREADS);
    for (int i = 0; i < n; i++) cpu_of[i] = i < no ? order[i] : -1;
    if (no < n) for (int i = 0; i < n; i++) cpu_of[i] = -1;       // more threads than cpus: leave it to the scheduler
    pin_self(cpu_of[0]);
    for (int i = 1; i < n; i++) pthread_create(&threads[i], NULL, worker, (void *)(intptr_t)i);
}

int pool_threads(void) { return n_thr; }

void pool_run(pool_fn fn, void *arg) {
    if (n_thr == 1) { fn(arg, 0); return; }
    job.fn = fn; job.arg = arg;
    uint32_t gen = atomic_fetch_add_explicit(&job.generation, 1, memory_order_release) + 1;
    if (atomic_load(&job.n_sleeping) > 0) syscall(SYS_futex, &job.generation, FUTEX_WAKE_PRIVATE, MAX_THREADS, NULL, NULL, 0);
    fn(arg, 0);
    // a worker that lost its core to another process gets it back sooner if the caller steps aside
    for (int i = 1, spins = 0; i < n_thr; i++)
        while (atomic_load_explicit(&done_of[i].done, memory_order_acquire) != gen) { cpu_relax(); if (++spins > SPIN_BEFORE_YIELD) sched_yield(); }
}
