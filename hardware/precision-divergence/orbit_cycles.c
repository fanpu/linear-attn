// Brent cycle detection for the rounded logistic map x -> 4*x*(1-x) in float32 / float64.
// Every orbit of a finite format is eventually periodic; this finds tail length mu and period lambda
// for many seeds, plus the smallest value on the cycle (a cycle fingerprint).
//
//   gcc -O2 -ffp-contract=off -fopenmp orbit_cycles.c -o orbit_cycles
//   ./orbit_cycles f32 2000 1 > cache/cycles_f32.tsv
//   ./orbit_cycles f64 64 1 > cache/cycles_f64.tsv
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#define DEF(T, NAME)                                                              \
  static inline T NAME##_f(T x) { return (T)4 * x * ((T)1 - x); }                 \
  static void NAME##_brent(T x0, uint64_t *mu_out, uint64_t *lam_out, T *cmin) {   \
    uint64_t power = 1, lam = 1;                                                  \
    T tort = x0, hare = NAME##_f(x0);                                             \
    while (tort != hare) {                                                        \
      if (power == lam) { tort = hare; power <<= 1; lam = 0; }                    \
      hare = NAME##_f(hare); lam++;                                               \
    }                                                                             \
    tort = x0; hare = x0;                                                         \
    for (uint64_t i = 0; i < lam; i++) hare = NAME##_f(hare);                     \
    uint64_t mu = 0;                                                              \
    while (tort != hare) { tort = NAME##_f(tort); hare = NAME##_f(hare); mu++; }  \
    T m = tort, y = tort;                                                         \
    for (uint64_t i = 0; i < lam; i++) { y = NAME##_f(y); if (y < m) m = y; }     \
    *mu_out = mu; *lam_out = lam; *cmin = m;                                      \
  }
DEF(float, f32)
DEF(double, f64)

static uint64_t splitmix(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s f32|f64 nseeds seed\n", argv[0]); return 1; }
  int is64 = strcmp(argv[1], "f64") == 0;
  int n = atoi(argv[2]);
  uint64_t base = strtoull(argv[3], 0, 10);
  printf("seed_index\tx0\tmu\tlambda\tcycle_min\n");
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < n; i++) {
    uint64_t s = base * 1000003ULL + (uint64_t)i;
    double u = (double)(splitmix(&s) >> 11) * 0x1.0p-53;   // uniform in [0,1)
    uint64_t mu, lam;
    char line[256];
    if (is64) {
      double cm; f64_brent(u, &mu, &lam, &cm);
      snprintf(line, sizeof line, "%d\t%.17g\t%llu\t%llu\t%.17g\n", i, u, (unsigned long long)mu, (unsigned long long)lam, cm);
    } else {
      float cm; float x0 = (float)u; f32_brent(x0, &mu, &lam, &cm);
      snprintf(line, sizeof line, "%d\t%.9g\t%llu\t%llu\t%.9g\n", i, (double)x0, (unsigned long long)mu, (unsigned long long)lam, (double)cm);
    }
    #pragma omp critical
    { fputs(line, stdout); fflush(stdout); }
  }
  return 0;
}
