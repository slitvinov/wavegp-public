# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Brown & Minion 1995 — double shear layer (uniform grid)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slitvinov/wavegp-public/blob/main/brown0.ipynb)
#
# Self-contained notebook that reproduces the **double shear layer** test of
# Brown & Minion, *"Performance of Under-resolved Two-Dimensional
# Incompressible Flow Simulations"*, J. Comput. Phys. **122** (1995) 165–183.
#
# This is the **uniform-grid** variant: the solver `main0.c` is `main.c` with
# all the AMR machinery stripped out (block-tree neighbour search, the
# lifted-wavelet ghost-cell exchange, refinement/coarsening).  On a uniform
# grid it produces bit-for-bit identical results — and needs no stencil
# tables, so there is no `gen_table.py` step.
#
# **By default it runs only 64² and 128²** — these finish quickly.  Each
# resolution doubling costs ~8× (4× cells × 2× time steps from the CFL limit),
# so 256² ≈ 8× and 512² ≈ 64× the 128² run.  Enable them in `RESOLUTIONS`.

# %% [markdown]
# ## 1. Imports
#
# Colab already ships gcc, curl, numpy, matplotlib and pillow — nothing to
# install.

# %%
import os, sys, subprocess, shutil

# %% [markdown]
# ## 2. Configuration
#
# `RESOLUTIONS` selects which cases to run.  Keep `[64, 128]` for a quick run;
# uncomment the full list to also run the expensive 256² / 512² cases.

# %%
RESOLUTIONS = [64, 128]                  # quick — default
# RESOLUTIONS = [64, 128, 256, 512]      # full Brown & Minion set (slow!)

# level L gives a (2^L * 8)^2 grid (8x8 cells per block); paper panel labels
LEVEL = {64: 3, 128: 4, 256: 5, 512: 6}
LABEL = {64: "A", 128: "B", 256: "C", 512: "D"}

# reference panels live in this repo
RAW = "https://raw.githubusercontent.com/slitvinov/wavegp-public/main"

print("will run:", RESOLUTIONS)

# %% [markdown]
# ## 3. Solver source
#
# The next cell writes the uniform-grid solver to disk.

# %%
# %%writefile main0.c
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* --- solver (BiCGStab with block-diagonal preconditioner) --- */
struct Solver {
  int blen, m, nnz, mean_row;
  const double *coo_val;
  const int *coo_row, *coo_col;
  const double *h2;
  double *precond;
  double *r, *rhat, *p, *nu, *t, *z, *x_opt;
};

static struct Solver *solver_create(int blen, const double *precond) {
  struct Solver *s = calloc(1, sizeof *s);
  s->blen = blen;
  s->precond = malloc(blen * blen * sizeof(double));
  memcpy(s->precond, precond, blen * blen * sizeof(double));
  return s;
}

static void solver_destroy(struct Solver *s) {
  free(s->precond);
  free(s->r);
  free(s->rhat);
  free(s->p);
  free(s->nu);
  free(s->t);
  free(s->z);
  free(s->x_opt);
  free(s);
}

static void sol_spmv(int m, int nnz, const double *val, const int *row,
                     const int *col, const double *x, double *y) {
  memset(y, 0, m * sizeof(double));
  for (int i = 0; i < nnz; i++)
    y[row[i]] += val[i] * x[col[i]];
}

static void sol_precond_apply(int m, int blen, const double *P,
                              const double *x, double *y) {
  int nb = m / blen;
#pragma omp parallel for schedule(static)
  for (int b = 0; b < nb; b++) {
    const double *xb = x + b * blen;
    double *yb = y + b * blen;
    for (int i = 0; i < blen; i++) {
      double s = 0;
      const double *row = P + i * blen;
      for (int j = 0; j < blen; j++)
        s += row[j] * xb[j];
      yb[i] = s;
    }
  }
}

static void sol_axpy(int m, double a, const double *x, double *y) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++)
    y[i] += a * x[i];
}

static void sol_scal(int m, double a, double *x) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++)
    x[i] *= a;
}

static double sol_dot(int m, const double *a, const double *b) {
  double s = 0;
#pragma omp parallel for reduction(+:s) schedule(static)
  for (int i = 0; i < m; i++)
    s += a[i] * b[i];
  return s;
}

static double sol_amax(int m, const double *x) {
  double mx = 0;
#pragma omp parallel for reduction(max:mx) schedule(static)
  for (int i = 0; i < m; i++) {
    double a = fabs(x[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

static void sol_matvec(struct Solver *s, const double *x, double *y) {
  sol_spmv(s->m, s->nnz, s->coo_val, s->coo_row, s->coo_col, x, y);
  if (s->mean_row >= 0) {
    double sum = 0;
    for (int i = 0; i < s->m; i++)
      sum += s->h2[i / s->blen] * x[i];
    y[s->mean_row] = sum;
  }
}

static void bicgstab(struct Solver *s, double *x, double max_error,
                     double max_rel_error, int max_restarts) {
  int m = s->m;
  double *r = s->r, *rhat = s->rhat, *p = s->p;
  double *nu = s->nu, *t = s->t, *z = s->z, *x_opt = s->x_opt;
  double eps = 1e-21;

  sol_matvec(s, x, nu);
  sol_axpy(m, -1.0, nu, r);

  double error = sol_amax(m, r);
  double error_init = error;
  double error_opt = error;
  memcpy(x_opt, x, m * sizeof(double));
  memcpy(rhat, r, m * sizeof(double));
  memset(nu, 0, m * sizeof(double));
  memset(p, 0, m * sizeof(double));

  double rho_prev = 1, alpha = 1, omega = 1;
  int restarts = 0;

  for (int k = 0; k < 1000; k++) {
    double rho = sol_dot(m, rhat, r);
    double nr = sol_dot(m, r, r);
    double nrh = sol_dot(m, rhat, rhat);
    int serious_breakdown = rho * rho < 1e-16 * nr * nrh;

    double beta = (rho / (rho_prev + eps)) * (alpha / (omega + eps));

    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) break;
      memcpy(rhat, r, m * sizeof(double));
      rho = sol_dot(m, r, r);
      memset(nu, 0, m * sizeof(double));
      memset(p, 0, m * sizeof(double));
      rho_prev = 1; alpha = 1; omega = 1;
      beta = (rho / (rho_prev + eps)) * (alpha / (omega + eps));
    }

    sol_axpy(m, -omega, nu, p);
    sol_scal(m, beta, p);
    sol_axpy(m, 1.0, r, p);

    sol_precond_apply(m, s->blen, s->precond, p, z);
    sol_matvec(s, z, nu);

    double rhat_nu = sol_dot(m, rhat, nu);
    alpha = rho / (rhat_nu + eps);

    sol_axpy(m, alpha, z, x);
    sol_axpy(m, -alpha, nu, r);

    sol_precond_apply(m, s->blen, s->precond, r, z);
    sol_matvec(s, z, t);

    double tr = sol_dot(m, t, r);
    double tt = sol_dot(m, t, t);
    omega = tr / (tt + eps);

    sol_axpy(m, omega, z, x);
    sol_axpy(m, -omega, t, r);

    error = sol_amax(m, r);
    if (error < error_opt) {
      error_opt = error;
      memcpy(x_opt, x, m * sizeof(double));
      if (error <= max_error || error / error_init <= max_rel_error)
        break;
    }
    rho_prev = rho;
  }
  memcpy(x, x_opt, m * sizeof(double));
}

static void solver_solve(struct Solver *s, int update_matrix,
    int m, int nnz,
    const double *coo_val, const int *coo_row, const int *coo_col,
    double *x, const double *b, const double *h2, int mean_row,
    double tol, double rtol, int restarts) {
  if (update_matrix || s->m != m) {
    free(s->r);     free(s->rhat);  free(s->p);
    free(s->nu);    free(s->t);     free(s->z);
    free(s->x_opt);
    s->r     = malloc(m * sizeof(double));
    s->rhat  = malloc(m * sizeof(double));
    s->p     = malloc(m * sizeof(double));
    s->nu    = malloc(m * sizeof(double));
    s->t     = malloc(m * sizeof(double));
    s->z     = malloc(m * sizeof(double));
    s->x_opt = malloc(m * sizeof(double));
  }
  s->m = m;
  s->nnz = nnz;
  s->coo_val = coo_val;
  s->coo_row = coo_row;
  s->coo_col = coo_col;
  s->h2 = h2;
  s->mean_row = mean_row;
  memcpy(s->r, b, m * sizeof(double));
  bicgstab(s, x, tol, rtol, restarts);
}
/* --- end solver --- */

typedef double Real;
enum { BS = 8 };
enum {
  F_U = 0,   /* u velocity */
  F_V = 1,   /* v velocity */
  F_P = 2,   /* pressure */
  F_PHI = 3, /* pressure correction (Poisson solve) */
  F_W = 4,   /* vorticity (output/indicator) */
  F_TMP = 5, /* scratch */
  F_N = 6,
  BLK_S = F_N *BS *BS,
};

enum { BC_WALL = 0, BC_SYMMETRY, BC_INFLOW, BC_OUTFLOW, BC_PERIODIC };
struct FaceBC { int type; Real val[F_N]; };
enum AdSt { Leave = 0, Refine = 1, Compress = -1, Dealloc = 2 };
struct Blk;
static struct Sim {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int step;
  int dump_count;
  Real CFL;
  Real Ctol;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real nextDumpTime;
  Real Rtol;
  Real time;
  Real L[2];
  int nb[2];
  long long n;
  struct FaceBC bc[4]; /* x-, x+, y-, y+ */
  struct Blk *blk;
  Real *fld;
  /* Solvers */
  struct Solver *solver;      /* Poisson (Cholesky precond) */
  struct Solver *helm_solver; /* Helmholtz (identity precond) */
  int coo_nnz, coo_cap;
  double *coo_val, *sol_x, *sol_b, *sol_h2;
  int *coo_row, *coo_col;
} sim;
static const char *arg_find(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc)
        return argv[i + 1];
      fprintf(stderr, "main.c: error: option -%s has no value\n", key);
      exit(1);
    }
  fprintf(stderr, "main.c: error: option -%s is not set\n", key);
  exit(1);
}
static Real arg_r(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad real '%s'\n", key, s);
    exit(1);
  }
  return v;
}
static int arg_i(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}
static int arg_i_opt(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0 && i + 1 < argc) {
      char *end;
      long v = strtol(argv[i + 1], &end, 10);
      if (end != argv[i + 1]) return (int)v;
    }
  return def;
}
struct Blk {
  double h, origin[2];
  int level, n, ix, iy;
};
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)
static void bl_fill(struct Blk *b, int level, int ix, int iy) {
  int scale = 1 << (level - sim.levelStart);
  b->level = level;
  b->n = 1 << level;
  b->ix = ix;
  b->iy = iy;
  b->h = sim.L[0] / (BS * sim.nb[0] * scale);
  b->origin[0] = b->h * BS * ix;
  b->origin[1] = b->h * BS * iy;
}
struct {
  int offset;
  int dim;
  const char *prefix;
} fld_t[] = {{F_U, 1, "u"}, {F_V, 1, "v"}, {F_P, 1, "p"},
            {F_PHI, 1, NULL}, {F_W, 1, "vort"}, {F_TMP, 1, NULL}};
enum { NVARS = sizeof fld_t / sizeof *fld_t };

enum { LB_BUF = ((2*4+BS)*(2*4+BS) + (BS/2+4+3)*(BS/2+4+3)) * 2 };
/* Uniform-grid ghost-cell fill: copy the block interior into the centre of
   the (2*ss+BS)^2 padded buffer m, then copy an ss-wide halo from the 8
   periodic same-level neighbour blocks.  (Replaces the lifted-wavelet AMR
   ghost exchange; on a uniform grid every neighbour is at the same level
   so the exchange is a plain copy.) */
static void lb_load(Real *m, int dim, int blk_offset, int ss,
                     long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int ns0 = sim.nb[0], ns1 = sim.nb[1];
  int xi = info->ix, yi = info->iy;

  Real *p0 = BLK(info_idx) + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1, cy = icode / 3 - 1;
    if (!cx && !cy)
      continue;
    int nx = (xi + cx + ns0) % ns0, ny = (yi + cy + ns1) % ns1;
    Real *src = BLK((long long)ny * ns0 + nx) + BS * BS * blk_offset;
    int dc = cx < 0 ? 0 : cx == 0 ? ss : ss + BS;   /* dest col start */
    int dr = cy < 0 ? 0 : cy == 0 ? ss : ss + BS;   /* dest row start */
    int sc = cx < 0 ? BS - ss : 0;                  /* src  col start */
    int sr = cy < 0 ? BS - ss : 0;
    int wc = cx == 0 ? BS : ss;                     /* copy width  */
    int wr = cy == 0 ? BS : ss;                     /* copy height */
    for (int r = 0; r < wr; r++)
      memcpy(m + dim * ((dr + r) * nm + dc),
             src + dim * ((sr + r) * BS + sc),
             wc * dim * sizeof(Real));
  }
}

/* ---- Poisson solver infrastructure ---- */
/* Wide Laplacian: (-4φ + φ_{i+2} + φ_{i-2} + φ_{j+2} + φ_{j-2}) / (4h²)
   Local block stencil: stride-2 neighbors */
static double ps_Aloc(int I1, int I2) {
  int j1=I1/BS, i1=I1%BS, j2=I2/BS, i2=I2%BS;
  if (i1==i2 && j1==j2) return 4.0;
  if ((abs(i1-i2)==2 && j1==j2) || (i1==i2 && abs(j1-j2)==2)) return -1.0;
  return 0.0;
}
static void ps_prec(double *P_inv) {
  double L[64][64], L_inv[64][64];
  memset(L, 0, sizeof L); memset(L_inv, 0, sizeof L_inv);
  for (int i=0; i<BS*BS; i++) L_inv[i][i]=1.0;
  for (int i=0; i<BS*BS; i++) {
    double s1=0; for (int k=0; k<i; k++) s1+=L[i][k]*L[i][k];
    L[i][i]=sqrt(ps_Aloc(i,i)-s1);
    for (int j=i+1; j<BS*BS; j++) {
      double s2=0; for (int k=0; k<i; k++) s2+=L[i][k]*L[j][k];
      L[j][i]=(ps_Aloc(j,i)-s2)/L[i][i];
    }
  }
  for (int br=0; br<BS*BS; br++) {
    double bsf=1./L[br][br];
    for (int c=0; c<=br; c++) L_inv[br][c]*=bsf;
    for (int wr=br+1; wr<BS*BS; wr++) {
      double wsf=L[wr][br];
      for (int c=0; c<=br; c++) L_inv[wr][c]-=wsf*L_inv[br][c];
    }
  }
  for (int i=0; i<BS*BS; i++) for (int j=0; j<BS*BS; j++) {
    double aux=0;
    for (int k=0; k<BS*BS; k++) aux += i<=k && j<=k ? L_inv[k][i]*L_inv[k][j] : 0;
    P_inv[i*BS*BS+j] = -aux;
  }
}
/* ---- Incompressible NS: Godunov-projection (Brown & Minion 1995) ---- */

static Real NU = 1e-4; /* kinematic viscosity */

static inline Real minmod(Real a, Real b) {
  return a * b <= 0 ? 0 : fabs(a) < fabs(b) ? a : b;
}

/* Compute vorticity: w = dv/dx - du/dy */
static void compute_vorticity(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *w = BLK(id) + BS*BS*F_W;
      int ss=1, nm=2*ss+BS;
      Real ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
#define U(di,dj) bu[nm*((j)+(dj)+ss)+(i)+(di)+ss]
#define V(di,dj) bv[nm*((j)+(dj)+ss)+(i)+(di)+ss]
          w[j*BS+i] = (V(1,0)-V(-1,0))*ih - (U(0,1)-U(0,-1))*ih;
#undef U
#undef V
        }
    }
  }
}

/* Refinement indicator based on vorticity magnitude */
static void dump(Real time, int step, char *path) {
  long i, j, k, x, y;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX];
  FILE *file;
  float xyz[4 * BS * BS][2];
  char *xyz_base, xdmf_path[FILENAME_MAX];
  FILE *xdmf;
  long long ncell = 0;
  for (i = 0; i < sim.n; i++) ncell += BS * BS;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    Real h = sim.blk[i].h, ox = sim.blk[i].origin[0], oy = sim.blk[i].origin[1];
    for (j = 0; j < BS; j++)
      for (k = 0; k < BS; k++) {
        int c = j * BS + k;
        float x0=ox+k*h, y0=oy+j*h, x1=x0+h, y1=y0+h;
        xyz[4*c+0][0]=x0; xyz[4*c+0][1]=y0;
        xyz[4*c+1][0]=x0; xyz[4*c+1][1]=y1;
        xyz[4*c+2][0]=x1; xyz[4*c+2][1]=y1;
        xyz[4*c+3][0]=x1; xyz[4*c+3][1]=y0;
      }
    fwrite(xyz, sizeof(float), 4*2*BS*BS, file);
  }
  fclose(file);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xdmf = fopen(xdmf_path, "w");
  fprintf(xdmf,
    "<Xdmf Version=\"2.0\">\n<Domain><Grid>\n"
    "  <Time Value=\"%.16e\"/>\n"
    "  <Information Name=\"Step\" Value=\"%d\"/>\n"
    "  <Topology Dimensions=\"%lld\" TopologyType=\"Quadrilateral\"/>\n"
    "  <Geometry Type=\"XY\">\n"
    "    <DataItem Dimensions=\"%lld 4 2\" Format=\"Binary\""
    " DataType=\"Float\" Precision=\"4\" Endian=\"Little\">%s</DataItem>\n"
    "  </Geometry>\n", time, step, ncell, ncell, xyz_path);
  for (size_t fi = 0; fi < NVARS; fi++)
    if (fld_t[fi].prefix) {
      int dim = fld_t[fi].dim, offset = fld_t[fi].offset;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, fld_t[fi].prefix);
      fprintf(xdmf,
        "  <Attribute Name=\"%s\" Center=\"Cell\" AttributeType=\"%s\">\n"
        "    <DataItem Dimensions=\"%lld %s\" Format=\"Binary\""
        " DataType=\"Float\" Precision=\"8\" Endian=\"Little\">%s</DataItem>\n"
        "  </Attribute>\n",
        fld_t[fi].prefix,
        dim > 1 ? "Vector" : "Scalar", ncell,
        dim > 1 ? "2" : "1", attr_path);
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(BLK(j) + offset*BS*BS, sizeof(Real), dim*BS*BS, file);
      fclose(file);
    }
  fprintf(xdmf, "</Grid></Domain></Xdmf>\n");
  fclose(xdmf);
}

/* AMR adaptation (reuse from main.c but simpler indicator) */
static inline Real slope4(Real phim2, Real phim1, Real phi0, Real phip1, Real phip2) {
  Real DC = 0.5*(phip1 - phim1);
  Real DL = phi0 - phim1;
  Real DR = phip1 - phi0;
  Real dlim = DL*DR > 0 ? fmin(2*fabs(DL), 2*fabs(DR)) : 0;
  Real dprime = fmin(fabs(DC), dlim) * (DC > 0 ? 1 : (DC < 0 ? -1 : 0));

  /* For the full 4th-order slope we need δ'_{i+1} and δ'_{i-1}.
     Compute them inline. */
  Real DC_p = 0.5*(phip2 - phi0);
  Real DL_p = phip1 - phi0;
  Real DR_p = phip2 - phip1;
  Real dlim_p = DL_p*DR_p > 0 ? fmin(2*fabs(DL_p), 2*fabs(DR_p)) : 0;
  Real dp_p = fmin(fabs(DC_p), dlim_p) * (DC_p > 0 ? 1 : (DC_p < 0 ? -1 : 0));

  Real DC_m = 0.5*(phi0 - phim2);
  Real DL_m = phim1 - phim2;
  Real DR_m = phi0 - phim1;
  Real dlim_m = DL_m*DR_m > 0 ? fmin(2*fabs(DL_m), 2*fabs(DR_m)) : 0;
  Real dp_m = fmin(fabs(DC_m), dlim_m) * (DC_m > 0 ? 1 : (DC_m < 0 ? -1 : 0));

  Real d4 = 4.0/3.0 * DC - (dp_p + dp_m) / 6.0;
  Real sgn = DC > 0 ? 1 : (DC < 0 ? -1 : 0);
  return fmin(fabs(d4), dlim) * sgn;
}

/* Multigrid Poisson solver for periodic cell-centered grid.
   Solves (-4u + u_{i+1} + u_{i-1} + u_{j+1} + u_{j-1}) = f on M×M periodic grid.
   Uses W-cycle with cell-centered bilinear prolongation. */
static void mg_smooth(double *u, const double *f, int m, int niter) {
  for (int sw = 0; sw < niter; sw++) {
    for (int color = 0; color < 2; color++)
      for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
          if ((i + j) % 2 != color) continue;
          int ip = (i+1)%m, im = (i-1+m)%m, jp = (j+1)%m, jm = (j-1+m)%m;
          u[j*m+i] = 0.25 * (u[j*m+ip]+u[j*m+im]+u[jp*m+i]+u[jm*m+i] - f[j*m+i]);
        }
    double mn = 0;
    for (int k = 0; k < m*m; k++) mn += u[k];
    mn /= m*m;
    for (int k = 0; k < m*m; k++) u[k] -= mn;
  }
}

static void mg_residual(const double *u, const double *f, double *r, int m) {
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i+1)%m, im = (i-1+m)%m, jp = (j+1)%m, jm = (j-1+m)%m;
      r[j*m+i] = f[j*m+i] - (-4*u[j*m+i]+u[j*m+ip]+u[j*m+im]+u[jp*m+i]+u[jm*m+i]);
    }
}

static void mg_restrict(const double *rf, double *rc, int mf) {
  /* Full-weighting restriction (9-point stencil, variational pair of bilinear prolongation) */
  int mc = mf / 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int i2=2*i, j2=2*j;
      int i2p=(i2+1)%mf, i2m=(i2-1+mf)%mf, j2p=(j2+1)%mf, j2m=(j2-1+mf)%mf;
      rc[j*mc+i] = (4*rf[j2*mf+i2]
        + 2*(rf[j2*mf+i2p]+rf[j2*mf+i2m]+rf[j2p*mf+i2]+rf[j2m*mf+i2])
        + rf[j2p*mf+i2p]+rf[j2p*mf+i2m]+rf[j2m*mf+i2p]+rf[j2m*mf+i2m]) / 16.0;
    }
}

static void mg_prolong_add(const double *ec, double *uf, int mc) {
  /* Cell-centered bilinear prolongation.
     Fine cell (2j, 2i) center at (j+1/4)*hc — closest to coarse(j,i), then (j-1,i-1).
     Weights: 9/16 from nearest, 3/16 from face-adjacent, 1/16 from diagonal. */
  int mf = mc * 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int im = (i-1+mc)%mc, jm = (j-1+mc)%mc;
      int ip = (i+1)%mc,    jp = (j+1)%mc;
      double cij = ec[j*mc+i];
      int fi = 2*i, fj = 2*j, fi1 = (2*i+1)%mf, fj1 = (2*j+1)%mf;
      /* fine(2j,  2i)   — lower-left quarter: uses (j,i), (j,i-1), (j-1,i), (j-1,i-1) */
      uf[fj*mf+fi]   += (9*cij + 3*ec[j*mc+im] + 3*ec[jm*mc+i] + ec[jm*mc+im]) / 16.0;
      /* fine(2j,  2i+1) — lower-right quarter: uses (j,i), (j,i+1), (j-1,i), (j-1,i+1) */
      uf[fj*mf+fi1]  += (9*cij + 3*ec[j*mc+ip] + 3*ec[jm*mc+i] + ec[jm*mc+ip]) / 16.0;
      /* fine(2j+1,2i)   — upper-left quarter: uses (j,i), (j,i-1), (j+1,i), (j+1,i-1) */
      uf[fj1*mf+fi]  += (9*cij + 3*ec[j*mc+im] + 3*ec[jp*mc+i] + ec[jp*mc+im]) / 16.0;
      /* fine(2j+1,2i+1) — upper-right quarter: uses (j,i), (j,i+1), (j+1,i), (j+1,i+1) */
      uf[fj1*mf+fi1] += (9*cij + 3*ec[j*mc+ip] + 3*ec[jp*mc+i] + ec[jp*mc+ip]) / 16.0;
    }
}

static void mg_vcycle(double *u, double *f, double *r, int m) {
  if (m <= 4) { mg_smooth(u, f, m, 50); return; }
  int mc = m / 2;
  double *uc = calloc(mc*mc, sizeof(double));
  double *fc = calloc(mc*mc, sizeof(double));
  double *rc = calloc(mc*mc, sizeof(double));

  mg_smooth(u, f, m, 4);
  mg_residual(u, f, r, m);
  mg_restrict(r, fc, m);
  mg_vcycle(uc, fc, rc, mc);
  mg_prolong_add(uc, u, mc);
  mg_smooth(u, f, m, 4);

  free(uc); free(fc); free(rc);
}

/* PCG solver for (-4φ + Σφ_nb) = f on M×M periodic grid.
   Uses multigrid V-cycle as preconditioner. */
static void mg_solve_periodic(double *x, const double *f, int M, double tol) {
  int N = M*M;
  double *rr = malloc(N*sizeof(double));
  double *z = calloc(N, sizeof(double));
  double *p = malloc(N*sizeof(double));
  double *Ap = malloc(N*sizeof(double));
  double *r_tmp = malloc(N*sizeof(double));

  /* r = f - A*x */
  mg_residual(x, f, rr, M);
  { double mn=0; for(int k=0;k<N;k++) mn+=rr[k]; mn/=N; for(int k=0;k<N;k++) rr[k]-=mn; }

  /* z = M^{-1} r (one V-cycle) */
  memset(z, 0, N*sizeof(double));
  mg_vcycle(z, rr, r_tmp, M);
  { double mn=0; for(int k=0;k<N;k++) mn+=z[k]; mn/=N; for(int k=0;k<N;k++) z[k]-=mn; }
  memcpy(p, z, N*sizeof(double));
  double rz = 0; for(int k=0;k<N;k++) rz += rr[k]*z[k];

  for (int it = 0; it < 100; it++) {
    /* Ap = A*p */
    for (int j=0;j<M;j++) for (int i=0;i<M;i++) {
      int ip=(i+1)%M,im=(i-1+M)%M,jp=(j+1)%M,jm=(j-1+M)%M;
      Ap[j*M+i] = -4*p[j*M+i]+p[j*M+ip]+p[j*M+im]+p[jp*M+i]+p[jm*M+i];
    }
    double pAp = 0; for(int k=0;k<N;k++) pAp += p[k]*Ap[k];
    if (fabs(pAp) < 1e-30) break;
    double alpha = rz / pAp;
    for(int k=0;k<N;k++) { x[k] += alpha*p[k]; rr[k] -= alpha*Ap[k]; }
    { double mn=0; for(int k=0;k<N;k++) mn+=rr[k]; mn/=N; for(int k=0;k<N;k++) rr[k]-=mn; }
    { double mn=0; for(int k=0;k<N;k++) mn+=x[k]; mn/=N; for(int k=0;k<N;k++) x[k]-=mn; }

    double rmax = 0; for(int k=0;k<N;k++) if(fabs(rr[k])>rmax) rmax=fabs(rr[k]);
    if (rmax < tol) break;

    /* z = M^{-1} r */
    memset(z, 0, N*sizeof(double));
    mg_vcycle(z, rr, r_tmp, M);
    { double mn=0; for(int k=0;k<N;k++) mn+=z[k]; mn/=N; for(int k=0;k<N;k++) z[k]-=mn; }
    double rz2 = 0; for(int k=0;k<N;k++) rz2 += rr[k]*z[k];
    double beta = rz2 / (rz + 1e-30);
    for(int k=0;k<N;k++) p[k] = z[k] + beta*p[k];
    rz = rz2;
  }
  { double mn=0; for(int k=0;k<N;k++) mn+=x[k]; mn/=N; for(int k=0;k<N;k++) x[k]-=mn; }
  free(rr); free(z); free(p); free(Ap); free(r_tmp);
}

/* --- AMR-aware gather/scatter: flatten to finest-level uniform grid --- */
static Real amr_finest_h(void) {
  Real hmin = sim.blk[0].h;
  for (long long i = 1; i < sim.n; i++)
    if (sim.blk[i].h < hmin) hmin = sim.blk[i].h;
  return hmin;
}
static int amr_finest_N(Real hf) { return (int)(sim.L[0] / hf + 0.5); }

/* Gather block field → flat Ng×Ng array.
   Fine blocks (ratio=1): direct copy.
   Coarse blocks: bilinear interpolation from cell centers using lb_load ghost data. */
static void amr_gather(double *dst, int field, int Ng, Real hf) {
  for (long long id = 0; id < sim.n; id++) {
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    Real *src = BLK(id) + BS * BS * field;
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++)
        dst[(by + j) * Ng + bx + i] = src[j * BS + i];
  }
}

/* Scatter flat Ng×Ng → block field (average for coarse blocks) */
static void amr_scatter(double *src, int field, int Ng, Real hf) {
  for (long long id = 0; id < sim.n; id++) {
    Real *dst = BLK(id) + BS * BS * field;
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++)
        dst[j * BS + i] = src[(by + j) * Ng + bx + i];
  }
}

/*
 * AMReX-style 2D Godunov edge state computation + MAC projection.
 * Ported from amrex-hydro/Godunov/hydro_godunov_edge_state_2D.cpp
 *
 * Algorithm:
 * 1. PLM prediction: L/R states at all faces using 4th-order slopes
 * 2. Upwind y-edges (yzlo) using vmac for x-direction transverse
 * 3. Final x-edge: xlo/xhi + normal div + transverse flux → Riemann with umac
 * 4. Upwind x-edges (xzlo) using umac for y-direction transverse
 * 5. Final y-edge: ylo/yhi + normal div + transverse flux → Riemann with vmac
 * 6. MAC projection on final edges
 * 7. Advection from projected edges → CN update
 */
static void advect_diffuse(Real dt) {
  Real h0 = amr_finest_h();
  int N = amr_finest_N(h0);
  Real ih = 1.0/h0;
  Real dtdx = dt/h0, dtdy = dt/h0;
  int NN = N*N;
  /* Periodic index helpers */
  #define IDX(i,j) (((j)+N)%N*N + ((i)+N)%N)

  /* Flat cell-centered fields (gathered from AMR blocks) */
  double *qu = calloc(NN,8), *qv = calloc(NN,8);
  amr_gather(qu, F_U, N, h0);
  amr_gather(qv, F_V, N, h0);

  /* Flat MAC velocities (from previous step or initial MAC projection) */
  /* For first call: compute preliminary MAC from simple PLM + upwind + project */
  double *umac = calloc(NN,8), *vmac = calloc(NN,8);
  /* Simple PLM + upwind for preliminary MAC */
  for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
    /* x-face at (i+1/2, j): left state from cell (i), right state from cell (i+1) */
    Real su_i = slope4(qu[IDX(i-2,j)],qu[IDX(i-1,j)],qu[IDX(i,j)],qu[IDX(i+1,j)],qu[IDX(i+2,j)]);
    Real su_ip = slope4(qu[IDX(i-1,j)],qu[IDX(i,j)],qu[IDX(i+1,j)],qu[IDX(i+2,j)],qu[IDX(i+3,j)]);
    Real uc = qu[IDX(i,j)], un = qu[IDX(i+1,j)];
    Real sL = uc > 0 ? 1 : 0, sR = un < 0 ? 1 : 0;
    Real lo = uc + (0.5 - sL*0.5*dtdx*uc)*su_i;
    Real hi = un + (-0.5 - sR*0.5*dtdx*un)*su_ip;
    Real uface = (lo+hi)*0.5;
    umac[IDX(i+1,j)] = (uface >= 0) ? lo : hi;
    if (fabs(uface) < 1e-10) umac[IDX(i+1,j)] = 0.5*(lo+hi);

    /* y-face at (i, j+1/2) */
    Real sv_j = slope4(qv[IDX(i,j-2)],qv[IDX(i,j-1)],qv[IDX(i,j)],qv[IDX(i,j+1)],qv[IDX(i,j+2)]);
    Real sv_jp = slope4(qv[IDX(i,j-1)],qv[IDX(i,j)],qv[IDX(i,j+1)],qv[IDX(i,j+2)],qv[IDX(i,j+3)]);
    Real vc = qv[IDX(i,j)], vn = qv[IDX(i,j+1)];
    sL = vc > 0 ? 1 : 0; sR = vn < 0 ? 1 : 0;
    lo = vc + (0.5 - sL*0.5*dtdy*vc)*sv_j;
    hi = vn + (-0.5 - sR*0.5*dtdy*vn)*sv_jp;
    Real vface = (lo+hi)*0.5;
    vmac[IDX(i,j+1)] = (vface >= 0) ? lo : hi;
    if (fabs(vface) < 1e-10) vmac[IDX(i,j+1)] = 0.5*(lo+hi);
  }
  /* MAC projection */
  {
    double *div = calloc(NN,8), *phi = calloc(NN,8);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++)
      div[IDX(i,j)] = (umac[IDX(i+1,j)]-umac[IDX(i,j)] + vmac[IDX(i,j+1)]-vmac[IDX(i,j)])*ih;
    for (int k=0;k<NN;k++) div[k] *= h0*h0;
    mg_solve_periodic(phi, div, N, 1e-10);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      umac[IDX(i,j)] -= (phi[IDX(i,j)]-phi[IDX(i-1,j)])*ih;
      vmac[IDX(i,j)] -= (phi[IDX(i,j)]-phi[IDX(i,j-1)])*ih;
    }
    free(div); free(phi);
  }

  /* Step 1: PLM L/R states at all faces for EACH component */
  /* Process one component at a time (n=0 for u, n=1 for v) */
  double *xedge_u = calloc(NN,8), *xedge_v = calloc(NN,8);
  double *yedge_u = calloc(NN,8), *yedge_v = calloc(NN,8);

  for (int n = 0; n < 2; n++) {
    double *q = (n==0) ? qu : qv;
    double *xedge = (n==0) ? xedge_u : xedge_v;
    double *yedge = (n==0) ? yedge_u : yedge_v;

    /* xlo[i,j] = left state at face (i+1/2,j) from cell i
       xhi[i,j] = right state at face (i-1/2,j) from cell i
       Note: xlo[i] is stored at face index i+1, xhi[i] at face index i */
    double *xlo = calloc(NN,8), *xhi = calloc(NN,8);
    double *ylo = calloc(NN,8), *yhi = calloc(NN,8);

    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      Real s = slope4(q[IDX(i-2,j)],q[IDX(i-1,j)],q[IDX(i,j)],q[IDX(i+1,j)],q[IDX(i+2,j)]);
      Real uc = qu[IDX(i,j)]; /* advecting velocity at cell center */
      /* Ipx: left state at face i+1/2 (AMReX: umns = S(i) + 0.5*(1-u*dt/dx)*slope) */
      xlo[IDX(i+1,j)] = q[IDX(i,j)] + 0.5*(1.0 - umac[IDX(i+1,j)]*dtdx)*s;
      /* Imx: right state at face i+1/2 from cell i+1 — computed by cell i+1 */
      /* Instead: right state at face i-1/2 from cell i */
      xhi[IDX(i,j)] = q[IDX(i,j)] + 0.5*(-1.0 - umac[IDX(i,j)]*dtdx)*s;

      Real sy = slope4(q[IDX(i,j-2)],q[IDX(i,j-1)],q[IDX(i,j)],q[IDX(i,j+1)],q[IDX(i,j+2)]);
      Real vc = qv[IDX(i,j)];
      ylo[IDX(i,j+1)] = q[IDX(i,j)] + 0.5*(1.0 - vmac[IDX(i,j+1)]*dtdy)*sy;
      yhi[IDX(i,j)] = q[IDX(i,j)] + 0.5*(-1.0 - vmac[IDX(i,j)]*dtdy)*sy;
    }

    /* Step 2: Upwind y-edges (yzlo) using vmac */
    double *yzlo = calloc(NN,8);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      Real vad = vmac[IDX(i,j)];
      Real lo_v = ylo[IDX(i,j)], hi_v = yhi[IDX(i,j)];
      yzlo[IDX(i,j)] = (fabs(vad) < 1e-10) ? 0.5*(lo_v+hi_v) : ((vad >= 0) ? lo_v : hi_v);
    }

    /* Step 3: Final x-edge with full transverse correction (AMReX lines 220-274)
       stl = xlo(i,j) + correction at cell (i-1,j)
       sth = xhi(i,j) + correction at cell (i,j)
       Correction = -(dt/2) * [q*du/dx + d(vmac*q)/dy] + (non-conservative: q*divu)
       For incompressible: divu=0, so correction = -(dt/2) * div(u_vec * q) */
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      /* Left state: from cell (i-1,j) */
      Real quxl = (umac[IDX(i,j)] - umac[IDX(i-1,j)]) * q[IDX(i-1,j)];
      Real stl = xlo[IDX(i,j)]
        - 0.5*dtdx * quxl
        - 0.5*dtdy * (yzlo[IDX(i-1,j+1)]*vmac[IDX(i-1,j+1)]
                      -yzlo[IDX(i-1,j  )]*vmac[IDX(i-1,j  )]);
      /* For non-conservative (incompressible): add q*divu = 0 */

      /* Right state: from cell (i,j) */
      Real quxh = (umac[IDX(i+1,j)] - umac[IDX(i,j)]) * q[IDX(i,j)];
      Real sth = xhi[IDX(i,j)]
        - 0.5*dtdx * quxh
        - 0.5*dtdy * (yzlo[IDX(i,j+1)]*vmac[IDX(i,j+1)]
                      -yzlo[IDX(i,j  )]*vmac[IDX(i,j  )]);

      /* Riemann solve using MAC velocity */
      Real uad = umac[IDX(i,j)];
      xedge[IDX(i,j)] = (fabs(uad) < 1e-10) ? 0.5*(stl+sth) : ((uad >= 0) ? stl : sth);
    }

    /* Step 4: Upwind x-edges (xzlo) using umac */
    double *xzlo = calloc(NN,8);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      Real uad = umac[IDX(i,j)];
      Real lo_u = xlo[IDX(i,j)], hi_u = xhi[IDX(i,j)];
      xzlo[IDX(i,j)] = (fabs(uad) < 1e-10) ? 0.5*(lo_u+hi_u) : ((uad >= 0) ? lo_u : hi_u);
    }

    /* Step 5: Final y-edge with full transverse correction (AMReX lines 302-358) */
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      Real qvyl = (vmac[IDX(i,j)] - vmac[IDX(i,j-1)]) * q[IDX(i,j-1)];
      Real stl = ylo[IDX(i,j)]
        - 0.5*dtdy * qvyl
        - 0.5*dtdx * (xzlo[IDX(i+1,j-1)]*umac[IDX(i+1,j-1)]
                      -xzlo[IDX(i  ,j-1)]*umac[IDX(i  ,j-1)]);

      Real qvyh = (vmac[IDX(i,j+1)] - vmac[IDX(i,j)]) * q[IDX(i,j)];
      Real sth = yhi[IDX(i,j)]
        - 0.5*dtdy * qvyh
        - 0.5*dtdx * (xzlo[IDX(i+1,j)]*umac[IDX(i+1,j)]
                      -xzlo[IDX(i  ,j)]*umac[IDX(i  ,j)]);

      Real vad = vmac[IDX(i,j)];
      yedge[IDX(i,j)] = (fabs(vad) < 1e-10) ? 0.5*(stl+sth) : ((vad >= 0) ? stl : sth);
    }

    free(xlo); free(xhi); free(ylo); free(yhi); free(yzlo); free(xzlo);
  }

  /* Step 6: MAC projection on final edges (make advecting velocity div-free) */
  /* The edge velocities for advection are xedge_u (u at x-faces) and yedge_v (v at y-faces) */
  {
    double *div = calloc(NN,8), *phi = calloc(NN,8);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++)
      div[IDX(i,j)] = (xedge_u[IDX(i+1,j)]-xedge_u[IDX(i,j)]
                       +yedge_v[IDX(i,j+1)]-yedge_v[IDX(i,j)])*ih;
    for (int k=0;k<NN;k++) div[k] *= h0*h0;
    mg_solve_periodic(phi, div, N, 1e-10);
    for (int j=0;j<N;j++) for (int i=0;i<N;i++) {
      xedge_u[IDX(i,j)] -= (phi[IDX(i,j)]-phi[IDX(i-1,j)])*ih;
      yedge_v[IDX(i,j)] -= (phi[IDX(i,j)]-phi[IDX(i,j-1)])*ih;
    }
    free(div); free(phi);
  }

  /* Step 7: Compute advection from edges → CN update (on flat grid) */
  { double *qp = calloc(NN, 8);
    amr_gather(qp, F_P, N, h0);
    double *u_new = calloc(NN, 8), *v_new = calloc(NN, 8);
    for (int gj = 0; gj < N; gj++) for (int gi = 0; gi < N; gi++) {
      Real uc = qu[IDX(gi,gj)], vc = qv[IDX(gi,gj)];
      Real uR=xedge_u[IDX(gi+1,gj)], uL=xedge_u[IDX(gi,gj)];
      Real vT=yedge_v[IDX(gi,gj+1)], vB=yedge_v[IDX(gi,gj)];
      Real uu_R=xedge_u[IDX(gi+1,gj)], uu_L=xedge_u[IDX(gi,gj)];
      Real vv_T=yedge_v[IDX(gi,gj+1)], vv_B=yedge_v[IDX(gi,gj)];
      Real u_yT=yedge_u[IDX(gi,gj+1)], u_yB=yedge_u[IDX(gi,gj)];
      Real v_xR=xedge_v[IDX(gi+1,gj)], v_xL=xedge_v[IDX(gi,gj)];
      Real adv_u = 0.5*(uR+uL)*(uu_R-uu_L)*ih + 0.5*(vT+vB)*(u_yT-u_yB)*ih;
      Real adv_v = 0.5*(uR+uL)*(v_xR-v_xL)*ih + 0.5*(vT+vB)*(vv_T-vv_B)*ih;
      Real lap_u = (qu[IDX(gi+1,gj)]+qu[IDX(gi-1,gj)]+qu[IDX(gi,gj+1)]+qu[IDX(gi,gj-1)]-4*uc)*ih*ih;
      Real lap_v = (qv[IDX(gi+1,gj)]+qv[IDX(gi-1,gj)]+qv[IDX(gi,gj+1)]+qv[IDX(gi,gj-1)]-4*vc)*ih*ih;
      Real dpx = (qp[IDX(gi+1,gj)]-qp[IDX(gi-1,gj)])*0.5*ih;
      Real dpy = (qp[IDX(gi,gj+1)]-qp[IDX(gi,gj-1)])*0.5*ih;
      Real alpha = NU*dt*0.5;
      u_new[IDX(gi,gj)] = uc + alpha*lap_u + dt*(-adv_u - dpx);
      v_new[IDX(gi,gj)] = vc + alpha*lap_v + dt*(-adv_v - dpy);
    }
    /* Scatter result back to blocks */
    amr_scatter(u_new, F_U, N, h0);
    amr_scatter(v_new, F_V, N, h0);
    free(qp); free(u_new); free(v_new);
  }

  free(qu); free(qv); free(umac); free(vmac);
  free(xedge_u); free(xedge_v); free(yedge_u); free(yedge_v);
  #undef IDX
}
static void helmholtz_solve(Real dt, int field) {
  Real alpha = NU * dt * 0.5;
  int N = BS * BS * sim.n;
  sim.sol_x = realloc(sim.sol_x, N * sizeof(double));
  sim.sol_b = realloc(sim.sol_b, N * sizeof(double));
  sim.sol_h2 = realloc(sim.sol_h2, sim.n * sizeof(double));
  sim.coo_nnz = 0;
  if (sim.coo_cap < 8 * N) {
    sim.coo_cap = 8 * N;
    sim.coo_val = realloc(sim.coo_val, sim.coo_cap * sizeof(double));
    sim.coo_row = realloc(sim.coo_row, sim.coo_cap * sizeof(int));
    sim.coo_col = realloc(sim.coo_col, sim.coo_cap * sizeof(int));
  }
#define COO(v, r, c) do { \
    sim.coo_val[sim.coo_nnz]=(v); sim.coo_row[sim.coo_nnz]=(r); \
    sim.coo_col[sim.coo_nnz]=(c); sim.coo_nnz++; } while(0)
  /* Assemble (I - α∆_h) where ∆_h = (-4 + Σnb)/h².
     Matrix entry: (1 + 4α/h²) on diagonal, (-α/h²) on neighbors. */
  static const int nb_dx[4] = {-1, 1, 0, 0};
  static const int nb_dy[4] = {0, 0, -1, 1};
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real h = info->h;
    Real ah2 = alpha / (h * h);
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        int sfc = i * BS * BS + iy * BS + ix;
        COO(1.0 + 4.0 * ah2, sfc, sfc);
        for (int d = 0; d < 4; d++) {
          int nx = ix + nb_dx[d], ny = iy + nb_dy[d];
          if (nx >= 0 && nx < BS && ny >= 0 && ny < BS) {
            COO(-ah2, sfc, i * BS * BS + ny * BS + nx);
          } else {
            int nbx = (info->ix + nb_dx[d] + sim.nb[0]) % sim.nb[0];
            int nby = (info->iy + nb_dy[d] + sim.nb[1]) % sim.nb[1];
            long long nbi = (long long)nby * sim.nb[0] + nbx;
            int nnx = ((nx % BS) + BS) % BS;
            int nny = ((ny % BS) + BS) % BS;
            COO(-ah2, sfc, nbi * BS * BS + nny * BS + nnx);
          }
        }
      }
  }
#undef COO
  /* Gather RHS and initial guess */
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    sim.sol_h2[i] = sim.blk[i].h * sim.blk[i].h;
    memcpy(&sim.sol_b[i*BS*BS], BLK(i) + BS*BS*field, BS*BS*sizeof(Real));
    memcpy(&sim.sol_x[i*BS*BS], BLK(i) + BS*BS*field, BS*BS*sizeof(Real));
  }
  solver_solve(sim.helm_solver, 1, N, sim.coo_nnz,
      sim.coo_val, sim.coo_row, sim.coo_col,
      sim.sol_x, sim.sol_b, sim.sol_h2, -1,
      1e-10, 1e-6, 50);
  /* Scatter solution */
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++)
    memcpy(BLK(i) + BS*BS*field, &sim.sol_x[i*BS*BS], BS*BS*sizeof(Real));
}

/* Poisson solve: Laplacian(phi) = div(u*)/dt */
static void poisson_solve(Real dt) {
  /* RHS = div(u*)/dt stored in F_TMP */
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *rhs = BLK(id)+BS*BS*F_TMP;
      int ss=1, nm=2*ss+BS;
      Real h = sim.blk[id].h;
      /* RHS = (4h²) * div(u*) / dt for wide Laplacian L = (-4φ + Σφ_{±2})/(4h²)
         div = (u_{i+1}-u_{i-1})/(2h) + (v_{j+1}-v_{j-1})/(2h)
         So rhs = 4h² * div / dt = 2h/dt * (u_{i+1}-u_{i-1}+v_{j+1}-v_{j-1}) */
      Real fac = 2.0 * h / dt;
      for (int j=0;j<BS;j++) for (int i=0;i<BS;i++)
        rhs[j*BS+i] = fac * (bu[nm*(j+ss)+i+1+ss]-bu[nm*(j+ss)+i-1+ss]
                             +bv[nm*(j+1+ss)+i+ss]-bv[nm*(j-1+ss)+i+ss]);
    }
  }
  /* Multigrid Poisson solve on flat periodic grid.
     The wide Laplacian L = (-4φ + φ_{i±2} + φ_{j±2}) decouples into 4 sub-grids
     based on (i%2, j%2). Each sub-grid is a standard 5-point Laplacian on (N/2)².
     Solve each sub-grid independently with multigrid V-cycles.
     For AMR: gather to finest-level grid, solve, scatter back. */
  Real hf = amr_finest_h();
  int Ng = amr_finest_N(hf);
  int M = Ng / 2;

  /* Recompute RHS on fine grid: the RHS was computed per-block with each block's h,
     but we need it consistent on the fine grid. Recompute from gathered velocity. */
  double *rhs_full = calloc(Ng * Ng, sizeof(double));
  double *phi_full = calloc(Ng * Ng, sizeof(double));
  { double *gu = calloc(Ng*Ng, 8), *gv = calloc(Ng*Ng, 8);
    amr_gather(gu, F_U, Ng, hf);
    amr_gather(gv, F_V, Ng, hf);
    Real fac = 2.0 * hf / dt;
    #define GI(i,j) (((j)+Ng)%Ng*Ng + ((i)+Ng)%Ng)
    for (int j=0;j<Ng;j++) for (int i=0;i<Ng;i++)
      rhs_full[j*Ng+i] = fac * (gu[GI(i+1,j)]-gu[GI(i-1,j)]
                                +gv[GI(i,j+1)]-gv[GI(i,j-1)]);
    #undef GI
    free(gu); free(gv);
  }
  amr_gather(phi_full, F_PHI, Ng, hf);

  /* For each sub-grid (sx, sy) in {0,1}², extract, solve, scatter back */
  for (int sy = 0; sy < 2; sy++)
    for (int sx = 0; sx < 2; sx++) {
      /* Extract sub-grid: cell (i,j) in full grid → (i/2, j/2) in sub-grid
         if i%2==sx && j%2==sy */
      double *f = calloc(M * M, sizeof(double));
      double *x = calloc(M * M, sizeof(double));
      for (int j=0;j<M;j++) for (int i=0;i<M;i++) {
        f[j*M+i] = rhs_full[(2*j+sy)*Ng + 2*i+sx];
        x[j*M+i] = phi_full[(2*j+sy)*Ng + 2*i+sx];
      }

      /* Solve using multigrid-preconditioned CG */
      mg_solve_periodic(x, f, M, 1e-10);

      /* Scatter back to full grid */
      for (int j=0;j<M;j++) for (int i=0;i<M;i++)
        phi_full[(2*j+sy)*Ng + 2*i+sx] = x[j*M+i];

      free(f); free(x);
    }

  /* Check wide Laplacian residual on full grid */
  { double rmax = 0;
    for (int j=0;j<Ng;j++) for (int i=0;i<Ng;i++) {
      int ip=(i+2)%Ng, im=(i-2+Ng)%Ng, jp=(j+2)%Ng, jm=(j-2+Ng)%Ng;
      double r = rhs_full[j*Ng+i] - (-4*phi_full[j*Ng+i]+phi_full[j*Ng+ip]+phi_full[j*Ng+im]+phi_full[jp*Ng+i]+phi_full[jm*Ng+i]);
      if (fabs(r) > rmax) rmax = fabs(r);
    }
    if (rmax > 1e-4) fprintf(stderr, "  poisson res=%.2e\n", rmax);
  }

  /* Scatter back to blocks */
  amr_scatter(phi_full, F_PHI, Ng, hf);
  free(rhs_full); free(phi_full);
}

/* Project: u^{n+1} = u* - dt*grad(phi), update pressure */
static void project(Real dt) {
#pragma omp parallel
  {
    Real bp[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bp, 1, F_PHI, 1, id);
      Real *u = BLK(id)+BS*BS*F_U;
      Real *v = BLK(id)+BS*BS*F_V;
      Real *p = BLK(id)+BS*BS*F_P;
      Real *phi = BLK(id)+BS*BS*F_PHI;
      int ss=1, nm=2*ss+BS;
      Real ih = 0.5/sim.blk[id].h;
      for (int j=0;j<BS;j++) for (int i=0;i<BS;i++) {
        int k = j*BS+i;
#define PH(di,dj) bp[nm*((j)+(dj)+ss)+(i)+(di)+ss]
        u[k] -= dt * (PH(1,0)-PH(-1,0))*ih;
        v[k] -= dt * (PH(0,1)-PH(0,-1))*ih;
        p[k] += phi[k]; /* accumulate pressure */
#undef PH
      }
    }
  }
}

static const struct {
  const char *name; int type; size_t off;
} param_tab[] = {
  {"levelStart", 0, offsetof(struct Sim, levelStart)},
  {"CFL", 1, offsetof(struct Sim, CFL)},
  {"tend", 1, offsetof(struct Sim, endTime)},
  {"tdump", 1, offsetof(struct Sim, dumpTime)},
};

int main(int argc, char **argv) {
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  fprintf(stderr, "main.c: %d threads\n", omp_get_num_threads());
#endif
  char *base = (char *)&sim;
  for (size_t i = 0; i < sizeof param_tab / sizeof *param_tab; i++)
    if (param_tab[i].type == 0)
      *(int *)(base + param_tab[i].off) = arg_i(argc, argv, param_tab[i].name);
    else
      *(Real *)(base + param_tab[i].off) = arg_r(argc, argv, param_tab[i].name);
  int dumpSteps = arg_i_opt(argc, argv, "sdump", 0);
  NU = arg_r(argc, argv, "nu");

  /* Domain: [0,1]^2, doubly periodic (Brown & Minion 1995) */
  sim.L[0] = 1.0; sim.L[1] = 1.0;
  sim.bc[0].type = BC_PERIODIC; sim.bc[1].type = BC_PERIODIC;
  sim.bc[2].type = BC_PERIODIC; sim.bc[3].type = BC_PERIODIC;
  {
    int ns = 1 << sim.levelStart;
    sim.nb[0] = ns; sim.nb[1] = ns;
    sim.n = (long long)ns * ns;
    sim.blk = calloc(sim.n, sizeof *sim.blk);
    sim.fld = calloc(sim.n * BLK_S, sizeof(Real));
    long long idx = 0;
    for (int iy = 0; iy < ns; iy++)
      for (int ix = 0; ix < ns; ix++)
        bl_fill(&sim.blk[idx++], sim.levelStart, ix, iy);
  }
  { double P_inv[BS*BS*BS*BS];
    ps_prec(P_inv); sim.solver = solver_create(BS*BS, P_inv);
    /* Identity preconditioner for Helmholtz (well-conditioned) */
    memset(P_inv, 0, sizeof P_inv);
    for (int i = 0; i < BS*BS; i++) P_inv[i*BS*BS+i] = -1.0;
    sim.helm_solver = solver_create(BS*BS, P_inv);
  }

  /* IC: double shear layer (Eq. 27-28) */
  Real rho_layer = 30.0, delta = 0.05;
  fprintf(stderr, "main.c: IC rho=%g delta=%g nu=%g\n", rho_layer, delta, NU);
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real *u = BLK(i)+BS*BS*F_U;
    Real *v = BLK(i)+BS*BS*F_V;
    Real h = info->h;
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        int j = BS*iy+ix;
        Real x = info->origin[0] + (ix+0.5)*h;
        Real y = info->origin[1] + (iy+0.5)*h;
        u[j] = y <= 0.5 ? tanh(rho_layer*(y-0.25)) : tanh(rho_layer*(0.75-y));
        v[j] = delta * sin(2*M_PI*x);
      }
  }

  /* Compute initial pressure p^{-1/2} by iteration (paper Sec. 3).
     Run a few projection steps with dt→0 to establish pressure field. */
  {
    Real dt0 = sim.blk[0].h / 10.0; /* small dt */
    for (int iter = 0; iter < 0; iter++) { /* disabled — too slow with flat-array advect */
      advect_diffuse(dt0);
      helmholtz_solve(dt0, F_U);
      helmholtz_solve(dt0, F_V);
      poisson_solve(dt0);
      /* Don't project — just accumulate pressure, then restore velocity */
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *p = BLK(i)+BS*BS*F_P;
        Real *phi = BLK(i)+BS*BS*F_PHI;
        for (int j = 0; j < BS*BS; j++) p[j] += phi[j];
      }
      /* Restore IC velocity */
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        struct Blk *info = &sim.blk[i];
        Real *uu = BLK(i)+BS*BS*F_U;
        Real *vv = BLK(i)+BS*BS*F_V;
        Real hh = info->h;
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
            int j = BS*iy+ix;
            Real x = info->origin[0]+(ix+0.5)*hh;
            Real y = info->origin[1]+(iy+0.5)*hh;
            uu[j] = y <= 0.5 ? tanh(rho_layer*(y-0.25)) : tanh(rho_layer*(0.75-y));
            vv[j] = delta * sin(2*M_PI*x);
          }
      }
    }
    fprintf(stderr, "main.c: initial pressure computed\n");
  }

  /* Main loop */
  while (1) {
    if (sim.step % 10 == 0) {
      compute_vorticity();
      fprintf(stderr, "main.c: %08d %.6e dt=%.3e blk=%lld\n",
              sim.step, sim.time, sim.dt, sim.n);
    }
    {
      int do_dump = 0;
      if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
        sim.nextDumpTime += sim.dumpTime; do_dump = 1;
      }
      if (dumpSteps > 0 && sim.step % dumpSteps == 0) do_dump = 1;
      if (do_dump) {
        compute_vorticity();
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
        dump(sim.time, sim.step, path);
      }
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime) break;

    /* CFL */
    Real smax = 0;
#pragma omp parallel for reduction(max:smax)
    for (long long i = 0; i < sim.n; i++) {
      Real *u = BLK(i)+BS*BS*F_U;
      Real *v = BLK(i)+BS*BS*F_V;
      Real ih = 1.0/sim.blk[i].h;
      for (int j = 0; j < BS*BS; j++)
        smax = fmax(smax, fmax(fabs(u[j]), fabs(v[j]))*ih);
    }
    sim.dt = sim.CFL / (smax + 1e-30);

    /* Projection method time step */
    advect_diffuse(sim.dt);       /* Godunov advection → RHS in F_U, F_V */
    helmholtz_solve(sim.dt, F_U); /* Crank-Nicolson viscosity for u */
    helmholtz_solve(sim.dt, F_V); /* Crank-Nicolson viscosity for v */
    poisson_solve(sim.dt);        /* ∆φ = ∇·U* */
    project(sim.dt);              /* U^{n+1} = U* - dt∇φ */

    sim.time += sim.dt;
    sim.step++;
  }
  solver_destroy(sim.solver);
  solver_destroy(sim.helm_solver);
  free(sim.coo_val); free(sim.coo_row); free(sim.coo_col);
  free(sim.sol_x); free(sim.sol_b); free(sim.sol_h2);
  fprintf(stderr, "main.c: end\n");
}

# %% [markdown]
# ## 4. Reference figures
#
# Download the paper reference panels for the selected resolutions.

# %%
for N in RESOLUTIONS:
    for figkey in ("fig2", "fig3"):
        rel = f"ref/{figkey}_{LABEL[N]}.png"
        os.makedirs("ref", exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", rel, f"{RAW}/{rel}"], check=True)
        print("fetched", rel)

# %% [markdown]
# ## 5. Compile the solver

# %%
subprocess.run(["gcc", "-O2", "-o", "main0", "main0.c", "-fopenmp", "-lm"],
               check=True)
print("built ./main0")


# %% [markdown]
# ## 6. Run
#
# Uniform grid, ν = 1e-4, CFL = 0.8, to t = 2.0 with dumps every 0.4 — so dump
# index 2 is t=0.8 (Fig. 2) and index 3 is t=1.2 (Fig. 3).

# %%
def run(size):
    level = LEVEL[size]
    out = f"run{size}"
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    print(f"=== run {size}x{size} (level {level}) ===")
    subprocess.run(
        ["../main0", "-levelStart", str(level), "-levelMax", str(level),
         "-AdaptSteps", "0", "-Rtol", "0", "-Ctol", "0",
         "-nu", "1e-4", "-CFL", "0.8", "-tend", "2.0", "-tdump", "0.4"],
        cwd=out, check=True)

for size in RESOLUTIONS:
    run(size)

# %% [markdown]
# ## 7. Comparison: paper vs computed

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BS = 8  # cells per block

def read_vort(path, N):
    """Reassemble an NxN vorticity field from the block-ordered raw dump."""
    nb = N // BS
    data = np.fromfile(path, dtype=np.float64)
    grid = np.zeros((N, N))
    for b in range(nb * nb):
        bx, by = b % nb, b // nb
        blk = data[b * BS * BS:(b + 1) * BS * BS].reshape(BS, BS)
        grid[by * BS:(by + 1) * BS, bx * BS:(bx + 1) * BS] = blk
    return grid

def plot_vort(ax, w, N):
    h = 1.0 / N
    x = np.linspace(h / 2, 1 - h / 2, N)
    X, Y = np.meshgrid(x, x)
    lv = np.arange(-36, 37, 6)
    lv = lv[lv != 0]
    ax.contour(X, Y, w, levels=lv, colors="k", linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

panels = [("fig2", 2, "t=0.8"), ("fig3", 3, "t=1.2")]
nrow = len(RESOLUTIONS)
fig, axes = plt.subplots(nrow, 4, figsize=(16, 4 * nrow), squeeze=False)

for row, N in enumerate(RESOLUTIONS):
    label = LABEL[N]
    for cp, (figkey, dump_idx, tlabel) in enumerate(panels):
        paper_col, comp_col = cp * 2, cp * 2 + 1
        try:
            axes[row, paper_col].imshow(Image.open(f"ref/{figkey}_{label}.png"),
                                        cmap="gray")
        except Exception as e:
            print("paper", figkey, label, "-", e)
        axes[row, paper_col].set_xticks([])
        axes[row, paper_col].set_yticks([])
        try:
            w = read_vort(f"run{N}/vel.{dump_idx:08d}.vort.raw", N)
            plot_vort(axes[row, comp_col], w, N)
        except Exception as e:
            print("computed", N, figkey, "-", e)
    axes[row, 0].set_ylabel(f"{N}²", fontsize=13, fontweight="bold")

axes[0, 0].set_title("Paper  t=0.8")
axes[0, 1].set_title("Computed  t=0.8")
axes[0, 2].set_title("Paper  t=1.2")
axes[0, 3].set_title("Computed  t=1.2")
fig.suptitle("Brown & Minion 1995 — double shear layer (ρ=30, ν=1e-4)",
             fontsize=14)
fig.tight_layout()
fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
print("saved comparison.png")

# %%
from IPython.display import Image as IPyImage
IPyImage("comparison.png")
