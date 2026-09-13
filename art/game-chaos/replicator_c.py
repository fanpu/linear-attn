"""Continuous-time coupled replicator equations (Sato, Akiyama & Farmer 2002) on generalised
rock-paper-scissors, integrated in logit coordinates with fixed-step RK4 in C (OpenMP over
orbits), float64. Also integrates the variational equation for the largest Lyapunov exponent
and records Poincare-section crossings refined by secant iteration.

  x_i' = x_i [(A y)_i - x.A y],   y_i' = y_i [(B x)_i - y.B x]
  A = [[ex,-1,1],[1,ex,-1],[-1,1,ex]],  B = same with ey.   Zero-sum iff ex = -ey.

Logits u_k = log(x_{k+1}/x_0), v_k = log(y_{k+1}/y_0) (k=1,2) turn this into
  u_k' = (A y)_k - (A y)_0 ,  v_k' = (B x)_k - (B x)_0 ,
whose right-hand side is bounded, so orbits that graze the simplex boundary stay accurate.

Section (SAF Fig. 1): g = x_1 - x_0 + y_1 - y_0 = 0, crossing with g increasing.
(0-indexed here; SAF's x2 - x1 + y2 - y1 = 0 with 1-indexing.)
"""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

C_SRC = r"""
#include <math.h>
#include <string.h>
#include <omp.h>

static inline void probs(const double *u, double *x) {
    double m = u[0] > u[1] ? u[0] : u[1]; if (m < 0) m = 0;
    double e0 = exp(-m), e1 = exp(u[0]-m), e2 = exp(u[1]-m), z = e0+e1+e2;
    x[0]=e0/z; x[1]=e1/z; x[2]=e2/z;
}
static inline void Amul(double e, const double *y, double *r) {
    r[0] = e*y[0] - y[1] + y[2];
    r[1] = y[0] + e*y[1] - y[2];
    r[2] = -y[0] + y[1] + e*y[2];
}
/* state s = (u1,u2,v1,v2, du1,du2,dv1,dv2) */
static void rhs(const double *s, double ex, double ey, int tangent, double *ds) {
    double x[3], y[3], Ay[3], Bx[3];
    probs(s, x); probs(s+2, y);
    Amul(ex, y, Ay); Amul(ey, x, Bx);
    ds[0] = Ay[1]-Ay[0]; ds[1] = Ay[2]-Ay[0];
    ds[2] = Bx[1]-Bx[0]; ds[3] = Bx[2]-Bx[0];
    if (tangent) {
        /* dx = (diag x - x x^T) (0,du1,du2) */
        double a0=0, a1=s[4], a2=s[5], m = x[1]*a1 + x[2]*a2;
        double dx[3] = {x[0]*(a0-m), x[1]*(a1-m), x[2]*(a2-m)};
        double b1=s[6], b2=s[7], n = y[1]*b1 + y[2]*b2;
        double dy[3] = {y[0]*(0-n), y[1]*(b1-n), y[2]*(b2-n)};
        double dAy[3], dBx[3]; Amul(ex, dy, dAy); Amul(ey, dx, dBx);
        ds[4] = dAy[1]-dAy[0]; ds[5] = dAy[2]-dAy[0];
        ds[6] = dBx[1]-dBx[0]; ds[7] = dBx[2]-dBx[0];
    }
}
static void rk4(double *s, double h, double ex, double ey, int nt) {
    int n = nt ? 8 : 4; double k1[8],k2[8],k3[8],k4[8],t[8];
    rhs(s,ex,ey,nt,k1); for(int i=0;i<n;i++) t[i]=s[i]+0.5*h*k1[i];
    rhs(t,ex,ey,nt,k2); for(int i=0;i<n;i++) t[i]=s[i]+0.5*h*k2[i];
    rhs(t,ex,ey,nt,k3); for(int i=0;i<n;i++) t[i]=s[i]+h*k3[i];
    rhs(t,ex,ey,nt,k4); for(int i=0;i<n;i++) s[i]+=h/6.0*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
}
static inline double gsec(const double *s) {
    double x[3], y[3]; probs(s,x); probs(s+2,y); return x[1]-x[0]+y[1]-y[0];
}
/*
 n orbits; s0 (n,4) initial logits; ex, ey (n); h step; nsteps; renorm every `every` steps.
 traj_every: store probs (x0..2,y0..2) every traj_every steps into traj (n, ntraj, 6).
 sec: (n, maxsec, 6) probs at crossings; nsec (n).
 lyap: (n) largest LE (per unit time), lyap_hist (n, nhist) running estimate every hist_every steps.
 Hdrift (n): max |H - H0|.
*/
void integrate(int n, const double *s0, const double *ex, const double *ey, double h, long nsteps,
               int every, int traj_every, int ntraj, double *traj, int maxsec, double *sec, int *nsec,
               double *lyap, int hist_every, int nhist, double *lyap_hist, double *Hdrift) {
    #pragma omp parallel for schedule(dynamic)
    for (int o = 0; o < n; o++) {
        double s[8]; memcpy(s, s0 + 4*o, 4*sizeof(double));
        s[4]=1; s[5]=0.3; s[6]=-0.7; s[7]=0.2;
        double nr = sqrt(s[4]*s[4]+s[5]*s[5]+s[6]*s[6]+s[7]*s[7]); for(int i=4;i<8;i++) s[i]/=nr;
        double sumlog = 0; int ns = 0; int ti = 0; int hi = 0;
        double x[3], y[3]; probs(s,x); probs(s+2,y);
        double H0 = -(log(x[0])+log(x[1])+log(x[2])+log(y[0])+log(y[1])+log(y[2]))/3.0, hd = 0;
        double gprev = gsec(s);
        for (long t = 1; t <= nsteps; t++) {
            double sp[8]; memcpy(sp, s, sizeof sp);
            rk4(s, h, ex[o], ey[o], 1);
            double gnow = gsec(s);
            if (gprev < 0 && gnow >= 0 && ns < maxsec) {
                /* secant refinement on tau in (0,h) from sp */
                double ta = 0, tb = h, ga = gprev, gb = gnow, tt[8];
                for (int it = 0; it < 6; it++) {
                    double tc = ta - ga*(tb-ta)/(gb-ga);
                    memcpy(tt, sp, sizeof tt); rk4(tt, tc, ex[o], ey[o], 0);
                    double gc = gsec(tt);
                    if (fabs(gc) < 1e-15) { ta = tb = tc; ga = gb = gc; break; }
                    ta = tb; ga = gb; tb = tc; gb = gc;
                }
                memcpy(tt, sp, sizeof tt); rk4(tt, tb, ex[o], ey[o], 0);
                probs(tt, sec + 6*((long)o*maxsec+ns)); probs(tt+2, sec + 6*((long)o*maxsec+ns)+3);
                ns++;
            }
            gprev = gnow;
            if (t % every == 0) {
                double m = sqrt(s[4]*s[4]+s[5]*s[5]+s[6]*s[6]+s[7]*s[7]);
                sumlog += log(m); for (int i=4;i<8;i++) s[i]/=m;
                probs(s,x); probs(s+2,y);
                double H = -(log(x[0])+log(x[1])+log(x[2])+log(y[0])+log(y[1])+log(y[2]))/3.0;
                if (fabs(H-H0) > hd) hd = fabs(H-H0);
            }
            if (traj_every > 0 && t % traj_every == 0 && ti < ntraj) {
                probs(s, traj + 6*((long)o*ntraj+ti)); probs(s+2, traj + 6*((long)o*ntraj+ti)+3); ti++;
            }
            if (hist_every > 0 && t % hist_every == 0 && hi < nhist) {
                lyap_hist[(long)o*nhist+hi] = sumlog / (t/every*every*h); hi++;
            }
        }
        nsec[o] = ns; lyap[o] = sumlog / ((nsteps/every)*every*h); Hdrift[o] = hd;
    }
}
"""


def _lib():
    tag = hashlib.md5(C_SRC.encode()).hexdigest()[:10]
    so = os.path.join(HERE, 'cache', f'replicator_{tag}.so')
    if not os.path.exists(so):
        os.makedirs(os.path.dirname(so), exist_ok=True)
        src = so.replace('.so', '.c')
        open(src, 'w').write(C_SRC)
        subprocess.check_call(['gcc', '-O3', '-march=native', '-ffast-math', '-fno-finite-math-only',
                               '-fopenmp', '-shared', '-fPIC', src, '-o', so, '-lm'])
    return ctypes.CDLL(so)


def probs_to_logits(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return np.stack([np.log(x[..., 1] / x[..., 0]), np.log(x[..., 2] / x[..., 0]),
                     np.log(y[..., 1] / y[..., 0]), np.log(y[..., 2] / y[..., 0])], -1)


def energy(x, y):
    return -(np.log(x).sum(-1) + np.log(y).sum(-1)) / 3.0


def integrate(s0, ex, ey, h=0.01, T=1000.0, every=100, traj_every=0, ntraj=0, maxsec=0,
              hist_every=0, nhist=0):
    """s0: (n,4) logits. Returns dict of numpy arrays."""
    lib = _lib()
    s0 = np.ascontiguousarray(s0, np.float64); n = s0.shape[0]
    ex = np.ascontiguousarray(np.broadcast_to(ex, (n,)), np.float64)
    ey = np.ascontiguousarray(np.broadcast_to(ey, (n,)), np.float64)
    nsteps = int(round(T / h))
    traj = np.zeros((n, max(ntraj, 1), 6)); sec = np.zeros((n, max(maxsec, 1), 6))
    nsec = np.zeros(n, np.int32); lyap = np.zeros(n); hist = np.zeros((n, max(nhist, 1)))
    Hd = np.zeros(n)
    P = ctypes.POINTER(ctypes.c_double); I = ctypes.POINTER(ctypes.c_int)
    f = lib.integrate
    f.argtypes = [ctypes.c_int, P, P, P, ctypes.c_double, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, P, ctypes.c_int, P, I, P, ctypes.c_int, ctypes.c_int, P, P]
    f(n, s0.ctypes.data_as(P), ex.ctypes.data_as(P), ey.ctypes.data_as(P), h, nsteps, every,
      traj_every, ntraj, traj.ctypes.data_as(P), maxsec, sec.ctypes.data_as(P), nsec.ctypes.data_as(I),
      lyap.ctypes.data_as(P), hist_every, nhist, hist.ctypes.data_as(P), Hd.ctypes.data_as(P))
    return dict(traj=traj[:, :ntraj], sec=sec, nsec=nsec, lyap=lyap, lyap_hist=hist[:, :nhist], Hdrift=Hd)


if __name__ == '__main__':
    # smoke test: SAF Table I initial conditions
    import time
    for eps in (0.0, 0.25, 0.5):
        k = np.arange(1, 6)
        x = np.stack([np.full(5, 0.5), 0.01 * k, 0.5 - 0.01 * k], -1)
        y = np.tile([0.5, 0.25, 0.25], (5, 1))
        t = time.time()
        r = integrate(probs_to_logits(x, y), eps, -eps, h=0.01, T=20000, every=100, maxsec=20000)
        print(f'eps={eps}: lambda1*1e3 =', np.round(r['lyap'] * 1e3, 2), 'nsec', r['nsec'],
              'Hdrift', r['Hdrift'].max(), f'{time.time()-t:.1f}s')
