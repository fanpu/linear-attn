// chase.c - dependent-load (pointer-chasing) latency across working-set sizes, one core, one page policy.
//
//   gcc -O2 -o chase chase.c
//   ./chase --cpu 5 --min 4096 --max 2147483648 --per-octave 8 --loads 20000000 --reps 5 [--huge] [--seq] > out.tsv
//   ./chase --cpu 5 --heartbeat 65536 --samples 1000000 --chunk 2000 > hb.bin      (binary: float64 ns per chunk)
//
// Each node is one 64-byte cache line. The chain is a single random cycle over all lines (Sattolo), so every
// load depends on the previous one and the hardware prefetcher gets no stride to follow. --seq instead chains
// lines in address order (a prefetch-friendly null). Working sets are log-spaced. For each size the chain is
// walked `loads` times, `reps` times, and the minimum ns/load is reported together with the median.
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

static uint64_t NODE = 64;   // bytes per node (--node); 64 = one cache line, 8 = eight nodes per line

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng(void) {  // splitmix64
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

#define NEXT(buf, off) (*(uint64_t *)((uint8_t *)(buf) + (off)))
static void build_chain(void *buf, uint64_t n, int seq) {   // n nodes, byte offsets in the chain
    if (seq) { for (uint64_t i = 0; i < n; i++) NEXT(buf, i * NODE) = ((i + 1) % n) * NODE; return; }
    uint64_t *perm = malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++) perm[i] = i;
    for (uint64_t i = n - 1; i > 0; i--) {            // Sattolo: one cycle of length n
        uint64_t j = rng() % i;
        uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (uint64_t i = 0; i < n; i++) NEXT(buf, perm[i] * NODE) = perm[(i + 1) % n] * NODE;
    free(perm);
}

static uint64_t walk(void *buf, uint64_t start, uint64_t loads) {
    uint64_t p = start;
    for (uint64_t i = 0; i < loads; i += 8) {      // unrolled, still one dependency chain
        p = NEXT(buf, p); p = NEXT(buf, p); p = NEXT(buf, p); p = NEXT(buf, p);
        p = NEXT(buf, p); p = NEXT(buf, p); p = NEXT(buf, p); p = NEXT(buf, p);
    }
    return p;
}

static void *alloc(uint64_t bytes, int huge) {
    uint64_t sz = (bytes + (2u << 20) - 1) & ~((uint64_t)(2u << 20) - 1);   // round to 2 MiB
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, sz, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(p, 0, bytes);     // fault the pages in before timing
    return p;
}

static int cmp_d(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    int cpu = -1, huge = 0, seq = 0, per_oct = 8, reps = 5;
    uint64_t smin = 4096, smax = 1ull << 31, loads = 20000000, hb = 0, samples = 1000000, chunk = 2000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) cpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--huge")) huge = 1;
        else if (!strcmp(argv[i], "--seq")) seq = 1;
        else if (!strcmp(argv[i], "--min")) smin = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max")) smax = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--per-octave")) per_oct = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--loads")) loads = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heartbeat")) hb = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--samples")) samples = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--chunk")) chunk = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--seed")) rng_state = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--node")) NODE = strtoull(argv[++i], 0, 10);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set)) { perror("sched_setaffinity"); return 1; }
    }
    if (hb) {   // heartbeat: fixed working set, `samples` chunks of `chunk` loads, ns per chunk written as float64 + timestamps
        uint64_t n = hb / NODE;
        void *buf = alloc(hb, huge);
        build_chain(buf, n, seq);
        uint64_t p = walk(buf, 0, chunk * 50);   // warm
        double *out = malloc(samples * 2 * sizeof(double));
        double t0 = now_ns();
        for (uint64_t s = 0; s < samples; s++) {
            double a = now_ns();
            p = walk(buf, p, chunk);
            double b = now_ns();
            out[2 * s] = a - t0; out[2 * s + 1] = b - a;
        }
        fwrite(out, sizeof(double), samples * 2, stdout);
        fprintf(stderr, "heartbeat done: %llu samples, sink %llu, %.1f s\n", (unsigned long long)samples, (unsigned long long)p, (now_ns() - t0) / 1e9);
        return 0;
    }
    printf("# cpu=%d huge=%d seq=%d node=%llu loads=%llu reps=%d\n", cpu, huge, seq, (unsigned long long)NODE, (unsigned long long)loads, reps);
    printf("bytes\tlines\tns_min\tns_med\tns_max\n");
    double *ts = malloc(reps * sizeof(double));
    for (double s = (double)smin; s <= (double)smax * 1.0001; s *= pow(2.0, 1.0 / per_oct)) {
        uint64_t bytes = ((uint64_t)(s + 0.5)) & ~63ull;
        uint64_t n = bytes / NODE;
        if (n < 2) continue;
        void *buf = alloc(bytes, huge);
        build_chain(buf, n, seq);
        uint64_t l = loads; if (l < 2 * n) l = 2 * n;        // at least 2 passes over the set
        uint64_t p = walk(buf, 0, l / 4 < 100000 ? 100000 : l / 4);   // warm
        for (int r = 0; r < reps; r++) {
            double a = now_ns(); p = walk(buf, p, l); double b = now_ns();
            ts[r] = (b - a) / (double)l;
        }
        qsort(ts, reps, sizeof(double), cmp_d);
        printf("%llu\t%llu\t%.3f\t%.3f\t%.3f\n", (unsigned long long)bytes, (unsigned long long)n, ts[0], ts[reps / 2], ts[reps - 1]);
        fflush(stdout);
        if (p == 0xFFFFFFFFFFFFull) fprintf(stderr, "sink\n");
        munmap(buf, (bytes + (2u << 20) - 1) & ~((uint64_t)(2u << 20) - 1));
    }
    return 0;
}
