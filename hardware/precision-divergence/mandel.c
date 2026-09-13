// Mandelbrot escape counts with the whole computation (pixel coordinates included) in one precision.
//   gcc -O2 -ffp-contract=off -fopenmp mandel.c -o cache/mandel
//   ./cache/mandel f32|f64|f128 cx cy width W H maxiter out.bin
// Output: float32 array H*W of smooth escape count (maxiter+1 = did not escape), row 0 = top.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static void k32(float cx, float cy, float width, int W, int H, int maxit, float *out) {
  float dx = width / (float)W;
  #pragma omp parallel for schedule(dynamic, 4)
  for (int j = 0; j < H; j++) {
    float y0 = cy + ((float)(H / 2 - j) - (float)0.5) * dx;
    for (int i = 0; i < W; i++) {
      float x0 = cx + ((float)(i - W / 2) + (float)0.5) * dx;
      float x = 0, y = 0, x2 = 0, y2 = 0; int n = 0;
      while (n < maxit && x2 + y2 <= (float)1e6) { y = (float)2 * x * y + y0; x = x2 - y2 + x0; x2 = x * x; y2 = y * y; n++; }
      float v;
      if (n >= maxit) v = (float)(maxit + 1);
      else { double m = (double)(x2 + y2); v = (float)(n + 1 - log(log(m) / 2.0 / log(2.0)) / log(2.0)); }
      out[j * W + i] = v;
    }
  }
}
static void k64(double cx, double cy, double width, int W, int H, int maxit, float *out) {
  double dx = width / (double)W;
  #pragma omp parallel for schedule(dynamic, 4)
  for (int j = 0; j < H; j++) {
    double y0 = cy + ((double)(H / 2 - j) - (double)0.5) * dx;
    for (int i = 0; i < W; i++) {
      double x0 = cx + ((double)(i - W / 2) + (double)0.5) * dx;
      double x = 0, y = 0, x2 = 0, y2 = 0; int n = 0;
      while (n < maxit && x2 + y2 <= (double)1e6) { y = (double)2 * x * y + y0; x = x2 - y2 + x0; x2 = x * x; y2 = y * y; n++; }
      float v;
      if (n >= maxit) v = (float)(maxit + 1);
      else { double m = (double)(x2 + y2); v = (float)(n + 1 - log(log(m) / 2.0 / log(2.0)) / log(2.0)); }
      out[j * W + i] = v;
    }
  }
}
static void k128(long double cx, long double cy, long double width, int W, int H, int maxit, float *out) {
  long double dx = width / (long double)W;
  #pragma omp parallel for schedule(dynamic, 4)
  for (int j = 0; j < H; j++) {
    long double y0 = cy + ((long double)(H / 2 - j) - (long double)0.5) * dx;
    for (int i = 0; i < W; i++) {
      long double x0 = cx + ((long double)(i - W / 2) + (long double)0.5) * dx;
      long double x = 0, y = 0, x2 = 0, y2 = 0; int n = 0;
      while (n < maxit && x2 + y2 <= (long double)1e6) { y = (long double)2 * x * y + y0; x = x2 - y2 + x0; x2 = x * x; y2 = y * y; n++; }
      float v;
      if (n >= maxit) v = (float)(maxit + 1);
      else { double m = (double)(x2 + y2); v = (float)(n + 1 - log(log(m) / 2.0 / log(2.0)) / log(2.0)); }
      out[j * W + i] = v;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 9) { fprintf(stderr, "usage\n"); return 1; }
  const char *t = argv[1];
  long double cx = strtold(argv[2], 0), cy = strtold(argv[3], 0), w = strtold(argv[4], 0);
  int W = atoi(argv[5]), H = atoi(argv[6]), maxit = atoi(argv[7]);
  float *out = malloc(sizeof(float) * W * H);
  if (!strcmp(t, "f32")) k32((float)cx, (float)cy, (float)w, W, H, maxit, out);
  else if (!strcmp(t, "f64")) k64((double)cx, (double)cy, (double)w, W, H, maxit, out);
  else k128(cx, cy, w, W, H, maxit, out);
  FILE *f = fopen(argv[8], "wb"); fwrite(out, sizeof(float), W * H, f); fclose(f);
  return 0;
}
