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
# # Brown & Minion 1995 — double shear layer
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slitvinov/wavegp-public/blob/main/brown.ipynb)
#
# Self-contained notebook that reproduces the **double shear layer** test of
# Brown & Minion, *"Performance of Under-resolved Two-Dimensional
# Incompressible Flow Simulations"*, J. Comput. Phys. **122** (1995) 165–183.
#
# It builds the AMR incompressible projection solver `main.c`, runs it, and
# lays the computed vorticity contours side-by-side with the paper's Fig. 2
# (t=0.8) and Fig. 3 (t=1.2).
#
# The solver source `main.c` and the table generator `gen_table.py` are
# **embedded in this notebook** — the cells below write them out with
# `%%writefile`, so nothing but the paper reference panels is downloaded.
#
# **By default it runs only 64² and 128²** — these finish quickly.  Each
# resolution doubling costs ~8× (4× cells × 2× time steps from the CFL limit),
# so 256² ≈ 8× and 512² ≈ 64× the 128² run.  Enable them in `RESOLUTIONS`.

# %% [markdown]
# ## 1. Dependencies

# %%
import sys, os, subprocess, shutil

IN_COLAB = "google.colab" in sys.modules
print("Colab:", IN_COLAB)

if IN_COLAB:
    subprocess.run("apt-get -qq install -y gcc >/dev/null", shell=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "numpy", "matplotlib", "pillow"], check=True)
    # amriso provides the lifted-wavelet stencils gen_table.py needs
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "git+https://github.com/cselab/amriso"], check=True)

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
# The next two cells write the embedded source files to disk.

# %%
# %%writefile main.c
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
struct HMap {
  long long *keys;
  int *vals;
  int cap;
};
static int hm_slot(const struct HMap *m, long long key) {
  unsigned long long h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  return (int)(h >> 32) & (m->cap - 1);
}
static int hm_get(const struct HMap *m, long long key) {
  int i = hm_slot(m, key);
  while (m->keys[i] >= 0) {
    if (m->keys[i] == key) return m->vals[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}
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
  struct HMap hm;
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
static long long hm_key(int level, int ix, int iy) {
  long long n = 1LL << level;
  return ((n * n) - 1) / 3 + iy * n + ix;
}
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

static inline int nb_skin(int c, int coord, int n) {
  int skin = coord == 0 || coord == n - 1;
  int skip = coord == 0 ? -1 : 1;
  return c == skip && skin;
}
static const int nb_ch_off[9][2][2] = {
  [0] = {{-1, -1}, {0, 0}},
  [1] = {{ 0, -1}, {1, -1}},
  [2] = {{ 2, -1}, {0, 0}},
  [3] = {{-1,  0}, {-1, 1}},
  [4] = {{ 0,  0}, {0, 0}},
  [5] = {{ 2,  0}, {2, 1}},
  [6] = {{-1,  2}, {0, 0}},
  [7] = {{ 0,  2}, {1, 2}},
  [8] = {{ 2,  2}, {0, 0}},
};
static const int nb_ch_n[9] = {1, 2, 1, 2, 0, 2, 1, 2, 1};
static void hm_rebuild(void) {
  int cap = 1;
  while (cap < 4 * sim.n) cap <<= 1;
  if (sim.hm.cap != cap) {
    free(sim.hm.keys);
    free(sim.hm.vals);
    sim.hm.cap = cap;
    sim.hm.keys = malloc(cap * sizeof *sim.hm.keys);
    sim.hm.vals = malloc(cap * sizeof *sim.hm.vals);
  }
  memset(sim.hm.keys, 0xff, cap * sizeof *sim.hm.keys);
  for (long long i = 0; i < sim.n; i++) {
    long long key = hm_key(sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy);
    int s = hm_slot(&sim.hm, key);
    while (sim.hm.keys[s] >= 0 && sim.hm.keys[s] != key)
      s = (s + 1) & (cap - 1);
    sim.hm.keys[s] = key;
    sim.hm.vals[s] = i;
  }
}
struct Nb {
  int8_t s;
  int idx;
  int ch[2];
};
static struct Nb nb_find(int level, int ix, int iy, int icode) {
  struct Nb r = {0, -1, {-1, -1}};
  int cx = icode % 3 - 1, cy = icode / 3 - 1;
  int scale = 1 << (level - sim.levelStart);
  int nd[2] = {sim.nb[0]*scale, sim.nb[1]*scale};
  int pos[2] = {ix, iy};
  int c[2] = {cx, cy};
  int skin[2], nbc = 0;
  for (int d = 0; d < 2; d++) { skin[d] = nb_skin(c[d], pos[d], nd[d]); nbc += skin[d]; }
  /* Periodic BCs: skip boundary handling, use wrapped coordinates */
  if (nbc > 0) {
    int all_periodic = 1;
    for (int d = 0; d < 2; d++) {
      if (!skin[d]) continue;
      int face = 2*d + (pos[d] == nd[d]-1 ? 1 : 0);
      if (sim.bc[face].type != BC_PERIODIC) all_periodic = 0;
    }
    if (all_periodic) { nbc = 0; for (int d = 0; d < 2; d++) skin[d] = 0; }
  }
  if (nbc > 0) {
    if (nbc == 1) {
      for (int d = 0; d < 2; d++) {
        if (!skin[d]) continue;
        int face = 2*d + (pos[d] == nd[d]-1 ? 1 : (pos[d] == 0 && c[d] == -1 ? 0 : 1));
        int bt = sim.bc[face].type;
        if (bt == BC_OUTFLOW) { r.s = 6+d; return r; }
        if (bt == BC_INFLOW) { r.s = 8+d; return r; }
        r.s = 3+d; return r; /* wall/symmetry */
      }
    }
    /* Multi-axis: outflow/inflow dominates */
    for (int d = 0; d < 2; d++) {
      if (!skin[d]) continue;
      int face = 2*d + (pos[d] == nd[d]-1 ? 1 : (pos[d] == 0 && c[d] == -1 ? 0 : 1));
      int bt = sim.bc[face].type;
      if (bt == BC_OUTFLOW) { r.s = 6+d; return r; }
      if (bt == BC_INFLOW) { r.s = 8+d; return r; }
    }
    r.s = 5; return r; /* wall corner */
  }
  int nx = (ix + cx + nd[0]) % nd[0], ny = (iy + cy + nd[1]) % nd[1];
  int idx = hm_get(&sim.hm, hm_key(level, nx, ny));
  if (idx >= 0) {
    r.s = 0;
    r.idx = idx;
    return r;
  }
  if (level > 0) {
    idx = hm_get(&sim.hm, hm_key(level - 1, nx / 2, ny / 2));
    if (idx >= 0) {
      r.s = 2;
      r.idx = idx;
      return r;
    }
  }
  if (level > 1) {
    idx = hm_get(&sim.hm, hm_key(level - 2, nx / 4, ny / 4));
    if (idx >= 0) {
      r.s = 2;
      r.idx = idx;
      return r;
    }
  }
  r.s = 1;
  int L1 = level + 1, nL1 = 1 << L1;
  for (int b = 0; b < nb_ch_n[icode]; b++) {
    int fx = (ix * 2 + nb_ch_off[icode][b][0] + nL1) % nL1;
    int fy = (iy * 2 + nb_ch_off[icode][b][1] + nL1) % nL1;
    r.ch[b] = hm_get(&sim.hm, hm_key(L1, fx, fy));
  }
  return r;
}
enum {
  OP_COPY,
  OP_AVG,
  OP_INTERP9,
  OP_INTERP3,
  OP_LELI,
  OP_BC_SCALAR,
  OP_BC_VECTOR,
  OP_BC_CORNER,
  OP_BC_FIXED,
};
struct LbOp {
  int8_t type;
  int8_t blk_idx;
  int8_t dst_idx;
  int8_t flags;
  int32_t src_off, dst_off, p1, p2;
};
struct LbSrc {
  int8_t level_delta;
  int8_t xi_mul, yi_mul;
  int8_t xi_add, yi_add;
  int8_t xi_shift, yi_shift;
  int8_t is_self;
  int8_t self_idx;
};
enum { MAX_PRE = 32, MAX_POST = 48, MAX_OPS = MAX_PRE + MAX_POST };
struct LbTab {
  int8_t n_blk;
  struct LbSrc blk_src[2];
  int8_t _pad;
  int32_t n_pre;
  int32_t n_post;
  struct LbOp ops[MAX_OPS];
};
static void lb_exec(Real *const blk[], Real *const dst[],
                         const struct LbOp*ops, int n, int dim, int nm, int nc) {
  Real *m = dst[0], *c = dst[1];
  for (int i = 0; i < n; i++) {
    const struct LbOp*o = &ops[i];
    switch (o->type) {
    case OP_COPY:
      memcpy(dst[o->dst_idx] + o->dst_off, blk[o->blk_idx] + o->src_off,
             o->p1 * dim * sizeof(Real));
      break;
    case OP_AVG: {
      Real *src = blk[o->blk_idx] + o->src_off;
      Real *d = dst[o->dst_idx] + o->dst_off;
      Real *q1 = src + o->p2 * dim;
      for (int k = 0; k < o->p1; k++)
        for (int dd = 0; dd < dim; dd++)
          d[k * dim + dd] =
              (src[2 * k * dim + dd] + src[(2 * k + 1) * dim + dd] +
               q1[2 * k * dim + dd] + q1[(2 * k + 1) * dim + dd]) /
              4;
      break;
    }
    case OP_INTERP9: {
      static const int8_t W[4][9] = {
          {1, 10, -1, 10, 56, -6, -1, -6, 1},
          {-1, 10, 1, -6, 56, 10, 1, -6, -1},
          {-1, -6, 1, 10, 56, -6, 1, 10, -1},
          {1, -6, -1, -6, 56, 10, -1, 10, 1},
      };
      const int8_t *w = W[o->flags & 3];
      for (int d = 0; d < dim; d++) {
        Real sum = 0;
        for (int jj = 0; jj < 3; jj++)
          for (int ii = 0; ii < 3; ii++)
            sum += w[3 * jj + ii] * c[o->src_off + d + dim * ((ii - 1) + nc * (jj - 1))];
        m[o->dst_off + d] = sum / 64.0;
      }
      break;
    }
    case OP_INTERP3: {
      for (int d = 0; d < dim; d++)
        m[o->dst_off + d] =
            (o->blk_idx * c[o->src_off + d] + o->dst_idx * c[o->p1 + d] +
             o->flags * c[o->p2 + d]) /
            32.0;
      break;
    }
    case OP_LELI: {
      static const int8_t W[2][3] = {
          {8, 10, -3},
          {24, -15, 6},
      };
      const int8_t *w = W[o->flags & 1];
      for (int d = 0; d < dim; d++) {
        Real a = m[o->src_off + d], b = m[o->dst_off + d], cv = m[o->p1 + d];
        m[o->src_off + d] = (w[0] * a + w[1] * b + w[2] * cv) / 15.0;
      }
      break;
    }
    case OP_BC_SCALAR: {
      Real *buf = dst[o->dst_idx];
      for (int d = 0; d < dim; d++)
        buf[o->dst_off + d] = buf[o->src_off + d];
      break;
    }
    case OP_BC_VECTOR: {
      Real *buf = dst[o->dst_idx];
      int dir = o->flags & 1;
      buf[o->dst_off + dir] = -buf[o->src_off + dir];
      buf[o->dst_off + 1 - dir] = buf[o->src_off + 1 - dir];
      break;
    }
    case OP_BC_CORNER: {
      Real *buf = dst[o->dst_idx];
      buf[o->dst_off] = -buf[o->src_off];
      buf[o->dst_off + 1] = -buf[o->src_off + 1];
      break;
    }
    case OP_BC_FIXED: {
      Real *src = dst[o->blk_idx]; /* bc_const buffer */
      Real *buf = dst[o->dst_idx];
      for (int d = 0; d < dim; d++)
        buf[o->dst_off + d] = src[d];
      break;
    }
    }
  }
}

enum { N_STATUS = 10 };
static const struct LbTab (*lb_tab[5][3])[3][2][2][N_STATUS];
static void lb_init(void) {
  int configs[][2] = {{1,1}, {1,2}, {2,1}, {4,1}};
  for (int ci = 0; ci < 3; ci++) {
    int ss = configs[ci][0], dim = configs[ci][1];
    char fname[64];
    snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "main.c: cannot open %s\n", fname);
      exit(1);
    }
    size_t sz = 3 * 3 * 2 * 2 * N_STATUS * sizeof(struct LbTab);
    struct LbTab *tab = malloc(sz);
    if (fread(tab, 1, sz, fp) != sz) {
      fprintf(stderr, "main.c: short read from %s\n", fname);
      exit(1);
    }
    fclose(fp);
    lb_tab[ss][dim] = (const struct LbTab (*)[3][2][2][N_STATUS])tab;
  }
}
enum { LB_BUF = ((2*4+BS)*(2*4+BS) + (BS/2+4+3)*(BS/2+4+3)) * 2 };
static void lb_load(Real *m, int dim, int blk_offset, int ss, long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int level = info->level;
  int xi = info->ix, yi = info->iy;
  const struct LbTab (*cflb_tab)[3][2][2][N_STATUS] = lb_tab[ss][dim];

  Real *p0 = BLK(info_idx) + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  Real *c = m + nm * nm * dim;
  Real bc_const[2] = {0};
  Real *dst[3] = {m, c, bc_const};
  int scale = 1 << (level - sim.levelStart);
  int nd2[2] = {sim.nb[0]*scale, sim.nb[1]*scale};
  int pos[2] = {xi, yi};

  struct {
    const struct LbTab *e;
    Real *blk[2];
  } dirs[8];
  int nd = 0;
  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1, cy = icode / 3 - 1;
    if (!cx && !cy)
      continue;
    struct Nb nr = nb_find(level, xi, yi, icode);

    /* Fill bc_const for inflow faces */
    if (nr.s == 8 || nr.s == 9) {
      int axis = nr.s - 8;
      int face = 2*axis + (pos[axis] == nd2[axis]-1 ? 1 : 0);
      memcpy(bc_const, &sim.bc[face].val[blk_offset], dim * sizeof(Real));
    }
    const struct LbTab *te =
        &cflb_tab[cx + 1][cy + 1][xi % 2][yi % 2][nr.s];
    Real *blk[2] = {NULL, NULL};
    for (int b = 0; b < te->n_blk; b++) {
      const struct LbSrc *bs = &te->blk_src[b];
      if (bs->is_self) {
        blk[b] = dst[bs->self_idx];
      } else if (bs->level_delta == 1) {
        blk[b] = BLK(nr.ch[b]) + BS * BS * blk_offset;
      } else {
        blk[b] = BLK(nr.idx) + BS * BS * blk_offset;
      }
    }
    dirs[nd].e = te;
    dirs[nd].blk[0] = blk[0];
    dirs[nd].blk[1] = blk[1];
    nd++;
  }

  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre,
                 dim, nm, nc);
  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE,
                 dirs[i].e->n_post, dim, nm, nc);
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
struct PsOp { int8_t blk_ref, cell_ix, cell_iy, _pad; float coeff; };
enum { PS_MAX_OPS = 16 };
struct PsEnt { int32_t n_ops; struct PsOp ops[PS_MAX_OPS]; };
static const struct PsEnt *ps_tab;
static void ps_load(void) {
  FILE *fp = fopen("tab_poisson.bin", "rb");
  if (!fp) { fprintf(stderr, "cannot open tab_poisson.bin\n"); exit(1); }
  size_t sz = 4*BS*2*4*sizeof(struct PsEnt);
  struct PsEnt *tab = malloc(sz);
  if (fread(tab,1,sz,fp) != sz) { fprintf(stderr, "short read tab_poisson.bin\n"); exit(1); }
  fclose(fp); ps_tab = tab;
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
static void compute_indicator(void) {
  compute_vorticity();
#pragma omp parallel for
  for (long long id = 0; id < sim.n; id++) {
    Real *w = BLK(id) + BS*BS*F_W;
    Real *t = BLK(id) + BS*BS*F_TMP;
    Real h = sim.blk[id].h;
    for (int j = 0; j < BS*BS; j++)
      t[j] = fabs(w[j]) * h; /* scale by h for resolution-independent threshold */
  }
}

/* Output dump */
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
static const Real ad_ref_w[4][9] = {
  { 1./64, 10./64, -1./64, 10./64, 56./64, -6./64, -1./64, -6./64,  1./64},
  {-1./64, 10./64,  1./64, -6./64, 56./64, 10./64,  1./64, -6./64, -1./64},
  {-1./64, -6./64,  1./64, 10./64, 56./64, -6./64,  1./64, 10./64, -1./64},
  { 1./64, -6./64, -1./64, -6./64, 56./64, 10./64, -1./64, 10./64,  1./64},
};
static const int ad_sib_ic[4] = {-1, 5, 7, 8};
static int ad_run(void) {
  compute_indicator();
  enum AdSt *state = calloc(sim.n, sizeof *state);
  long long *ref_idx = malloc(sim.n * sizeof *ref_idx);
  long long *com_idx = malloc(sim.n * sizeof *com_idx);
  long long n_ref = 0, n_com = 0;
  int Changed = 0;
#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS*BS*F_TMP;
    double Linf = 0;
    for (int j = 0; j < BS*BS; j++) Linf = fmax(Linf, fabs(b[j]));
    int lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax-1 ? Refine
             : Linf < sim.Ctol && lev > sim.levelStart ? Compress
             : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed) goto done;
  for (int More = 1; More;) {
    More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] != Refine) continue;
      struct Blk *bj = &sim.blk[j];
      for (int ic = 0; ic < 9; ic++) {
        if (ic == 4) continue;
        struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, ic);
        if (nr.s >= 3 || nr.idx < 0) continue;
        if (nr.s == 2 && state[nr.idx] != Refine)
          { state[nr.idx] = Refine; More = 1; }
        else if (nr.s == 0 && state[nr.idx] == Compress)
          state[nr.idx] = Leave;
      }
    }
  }
  for (long long j = 0; j < sim.n; j++) {
    if (state[j] != Compress) continue;
    struct Blk *bj = &sim.blk[j];
    if ((bj->ix | bj->iy) & 1) continue;
    long long sib[4] = {j};
    int ok = 1;
    for (int s = 1; s < 4 && ok; s++) {
      struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, ad_sib_ic[s]);
      ok = nr.s == 0 && nr.idx >= 0 && state[nr.idx] == Compress;
      sib[s] = nr.idx;
    }
    for (int s = 0; s < 4 && ok; s++) {
      struct Blk *bs = &sim.blk[sib[s]];
      for (int ic = 0; ic < 9 && ok; ic++)
        if (ic != 4) ok = nb_find(bs->level, bs->ix, bs->iy, ic).s != 1;
    }
    if (!ok) state[j] = Leave;
  }
  for (long long j = 0; j < sim.n; j++)
    if (state[j] == Refine) ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix|sim.blk[j].iy)&1))
      com_idx[n_com++] = j;
  fprintf(stderr, "  ad: com/ref %lld/%lld\n", n_com, n_ref);
  if (n_ref == 0 && n_com == 0) goto done;
  long long nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4*n_ref*BLK_S*sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++) state[i] = Leave;
#pragma omp parallel
  {
    Real lm[LB_BUF];
#pragma omp for
    for (long long k = 0; k < n_ref; k++) {
      struct Blk *par = &sim.blk[ref_idx[k]];
      int px=par->ix, py=par->iy;
      Real *blks[4];
      for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
        long long ci = nprev + 4*k + 2*J+I;
        bl_fill(&sim.blk[ci], par->level+1, 2*px+I, 2*py+J);
        blks[2*J+I] = BLK(ci);
      }
      int nm = 2 + BS;
      for (size_t m = 0; m < NVARS; m++) {
        int dim=fld_t[m].dim, offset=fld_t[m].offset;
        lb_load(lm, dim, offset, 1, ref_idx[k]);
        for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
          Real *b = blks[J*2+I] + offset*BS*BS;
          for (int j=0;j<BS;j+=2) for (int i=0;i<BS;i+=2) {
            int i0=i/2+I*(BS/2)+1, j0=j/2+J*(BS/2)+1;
            int sub[4]={BS*j+i, BS*j+i+1, BS*(j+1)+i, BS*(j+1)+i+1};
            for (int s=0;s<4;s++) for (int d=0;d<dim;d++) {
              Real val=0;
              for (int kk=0;kk<9;kk++)
                val += ad_ref_w[s][kk]*lm[dim*(nm*(j0+kk/3-1)+i0+kk%3-1)+d];
              b[dim*sub[s]+d] = val;
            }
          }
        }
      }
      state[ref_idx[k]] = Dealloc;
    }
#pragma omp for
    for (long long k = 0; k < n_com; k++) {
      long long ci = com_idx[k];
      struct Blk *p0 = &sim.blk[ci];
      int level=p0->level, x=p0->ix, y=p0->iy;
      Real *blk[4] = {BLK(ci)};
      for (int s=1;s<4;s++) { blk[s]=BLK(nb_find(level,x,y,ad_sib_ic[s]).idx); state[nb_find(level,x,y,ad_sib_ic[s]).idx]=Dealloc; }
      for (size_t v=0;v<NVARS;v++) {
        int dim=fld_t[v].dim, off=fld_t[v].offset;
        Real *dst=blk[0]+off*BS*BS;
        for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
          Real *src=blk[J*2+I]+off*BS*BS;
          for (int j=0;j<BS;j+=2) for (int i=0;i<BS;i+=2) {
            int o=BS*(j/2+J*(BS/2))+i/2+I*(BS/2);
            for (int d=0;d<dim;d++)
              dst[dim*o+d]=(src[dim*(BS*j+i)+d]+src[dim*(BS*j+i+1)+d]+src[dim*(BS*(j+1)+i)+d]+src[dim*(BS*(j+1)+i+1)+d])/4;
          }
        }
      }
      bl_fill(p0, level-1, x/2, y/2);
    }
  }
  long long cnt=0;
  for (long long i=0;i<sim.n;i++) {
    if (state[i]==Dealloc) continue;
    if (cnt!=i) { memmove(BLK(cnt),BLK(i),BLK_S*sizeof(Real)); sim.blk[cnt]=sim.blk[i]; }
    cnt++;
  }
  sim.n=cnt;
  sim.blk=realloc(sim.blk,sim.n*sizeof*sim.blk);
  sim.fld=realloc(sim.fld,sim.n*BLK_S*sizeof(Real));
  hm_rebuild();
done:
  free(state); free(ref_idx); free(com_idx);
  return Changed;
}

/* ---- Incompressible NS time step ---- */

/*
 * 4th-order monotone limited slope (Brown & Minion Eq. 20 right column).
 * D^C = (φ_{i+1} - φ_{i-1}) / 2
 * D^L = φ_i - φ_{i-1}
 * D^R = φ_{i+1} - φ_i
 * δ^lim = { min(2|D^L|, 2|D^R|) if D^L*D^R > 0, else 0 } * sign(D^C)
 * δ' = min(|D^C|, δ^lim * sign(D^C))
 * δ = min(|4*D^C/3 - (δ'_{i+1} + δ'_{i-1})/6|, δ^lim) * sign(D^C)
 */
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
  memset(dst, 0, (size_t)Ng * Ng * sizeof(double));
  Real lb[LB_BUF];
  for (long long id = 0; id < sim.n; id++) {
    Real h = sim.blk[id].h;
    int ratio = (int)(h / hf + 0.5);
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    if (ratio == 1) {
      /* Fine block: direct copy */
      Real *src = BLK(id) + BS * BS * field;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          dst[(by + j) * Ng + bx + i] = src[j * BS + i];
    } else {
      /* Coarse block: bilinear interpolation from cell centers.
         Load with 1-cell ghost layer for interpolation at edges. */
      lb_load(lb, 1, field, 1, id);
      int ss = 1, nm = 2 * ss + BS;
      /* Each coarse cell (i,j) maps to ratio×ratio fine cells.
         Fine cell (di,dj) within coarse cell has fractional position
         fx = (di + 0.5) / ratio - 0.5, fy = (dj + 0.5) / ratio - 0.5
         Bilinear interpolation from 4 nearest coarse cell centers. */
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          for (int dj = 0; dj < ratio; dj++)
            for (int di = 0; di < ratio; di++) {
              double fx = ((double)di + 0.5) / ratio - 0.5;
              double fy = ((double)dj + 0.5) / ratio - 0.5;
              int i0 = (fx < 0) ? -1 : 0, j0 = (fy < 0) ? -1 : 0;
              double wx = fx - i0, wy = fy - j0;
              /* lb indices: cell (i,j) in block → lb[nm*(j+ss)+i+ss] */
              #define LB(ci,cj) lb[nm*((j)+(cj)+ss)+(i)+(ci)+ss]
              double v00 = LB(i0, j0);
              double v10 = LB(i0 + 1, j0);
              double v01 = LB(i0, j0 + 1);
              double v11 = LB(i0 + 1, j0 + 1);
              #undef LB
              double val = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10
                         + (1 - wx) * wy * v01 + wx * wy * v11;
              int gx = bx + i * ratio + di;
              int gy = by + j * ratio + dj;
              if (gx >= 0 && gx < Ng && gy >= 0 && gy < Ng)
                dst[gy * Ng + gx] = val;
            }
    }
  }
}

/* Scatter flat Ng×Ng → block field (average for coarse blocks) */
static void amr_scatter(double *src, int field, int Ng, Real hf) {
  for (long long id = 0; id < sim.n; id++) {
    Real *dst = BLK(id) + BS * BS * field;
    Real h = sim.blk[id].h;
    int ratio = (int)(h / hf + 0.5);
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    Real inv = 1.0 / (ratio * ratio);
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++) {
        double sum = 0;
        for (int dj = 0; dj < ratio; dj++)
          for (int di = 0; di < ratio; di++)
            sum += src[(by + j * ratio + dj) * Ng + bx + i * ratio + di];
        dst[j * BS + i] = sum * inv;
      }
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
  static const int nb_ic[4] = {3, 5, 1, 7};
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
            struct Nb pnr = nb_find(info->level, info->ix, info->iy, nb_ic[d]);
            if (pnr.s != 0 || pnr.idx < 0) continue;
            int nnx = ((nx % BS) + BS) % BS;
            int nny = ((ny % BS) + BS) % BS;
            COO(-ah2, sfc, pnr.idx * BS * BS + nny * BS + nnx);
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
  hm_rebuild();
  lb_init();
  ps_load();
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

    if (sim.step > 0 && sim.step % sim.AdaptSteps == 0) ad_run();

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

# %%
# %%writefile gen_table.py
# #!/usr/bin/env python3
"""
Generate 2D binary dispatch tables for AMR ghost fill/interpolation/BC.

Status codes (N_STATUS=10):
  0: same-level neighbor
  1: finer neighbor (children)
  2: coarser neighbor
  3: x-wall/symmetry   4: y-wall/symmetry   5: xy-corner
  6: x-outflow          7: y-outflow
  8: x-inflow           9: y-inflow

Table layout: [cx+1][cy+1][xparity][yparity][status] -> entry
  3 * 3 * 2 * 2 * 10 = 360 entries per (ss, dim) config.
"""

import struct

BS = 8
MAX_PRE = 32
MAX_POST = 48
MAX_OPS = MAX_PRE + MAX_POST

OP_COPY = 0
OP_AVG = 1
OP_INTERP9 = 2
OP_INTERP3 = 3
OP_LELI = 4
OP_BC_SCALAR = 5
OP_BC_VECTOR = 6
OP_BC_CORNER = 7
OP_BC_FIXED = 8

N_STATUS = 10

# Child neighbor patterns: (cx, cy, row_stride, col_stride, offsets, n_blk)
CHILD_NB = [
    (-1, -1, 3, 1, [(-1, -1)], 1),
    ( 0, -1, 1, 1, [( 0, -1), (1, -1)], 2),
    ( 1, -1, 3, 1, [( 2, -1)], 1),
    (-1,  0, 1, 2, [(-1,  0), (-1, 1)], 2),
    ( 1,  0, 1, 2, [( 2,  0), (2, 1)], 2),
    (-1,  1, 3, 1, [(-1,  2)], 1),
    ( 0,  1, 1, 1, [( 0,  2), (1, 2)], 2),
    ( 1,  1, 3, 1, [( 2,  2)], 1),
]

LI_LE_TAB = [
    (0, +1, [(0, 0, -1, 0, -2), (1, 0, -2, 0, -3)]),
    (0, -1, [(1, 0, +2, 0, +3), (0, 0, +1, 0, +2)]),
    (+1, 0, [(0, -1, 0, -2, 0), (1, -2, 0, -3, 0)]),
    (-1, 0, [(1, +2, 0, +3, 0), (0, +1, 0, +2, 0)]),
]

# 1D Lagrange interpolation weights (×32) from 3 coarse cells at positions pts,
# evaluated at fine subcell offset t = ±1/4.
# branch: 0 = left edge (pts 0,1,2), 1 = right edge (-2,-1,0), 2 = interior (-1,0,1)
_PTS = {0: (0, 1, 2), 1: (-2, -1, 0), 2: (-1, 0, 1)}
def _lagrange3(pts, t):
    """3-point Lagrange weights at t, scaled by 32."""
    p = list(pts)
    return tuple(int(round(32 * (t-p[j])*(t-p[k]) / ((p[i]-p[j])*(p[i]-p[k]))))
                 for i, j, k in [(0,1,2),(1,0,2),(2,0,1)])
FACE_W = {(b, sub): _lagrange3(_PTS[b], 0.25 * (2*sub - 1))
           for b in range(3) for sub in range(2)}  # sub: 0→t=-1/4, 1→t=+1/4
# FACE_DSIGN[dirn][k]: sign pattern for 4 subcells in 2×2 block.
# k = (tangent_sub, normal_sub) ordering depends on dirn.
# dirn=0 (x-face): subcells (ix_sub, iy_sub) → k = iy_sub*2 + ix_sub, sign = (-1)^iy_sub
# dirn=1 (y-face): subcells (ix_sub, iy_sub) → k = iy_sub*2 + ix_sub, sign = (-1)^ix_sub
# k indexes 4 subcells: k=0:(0,0), k=1:(1,0), k=2:(0,1), k=3:(1,1)
# dirn=0 (x-face): tangent=y, sign alternates with k&1 (ix_sub)
# dirn=1 (y-face): tangent=x, sign alternates with k>>1 (iy_sub)
FACE_DSIGN = [[1, -1, 1, -1], [1, 1, -1, -1]]


def get_child_pattern(cx, cy):
    for e in CHILD_NB:
        if e[0] == cx and e[1] == cy:
            return e
    return None


def ghost_bounds(c, ss):
    s = -ss if c < 0 else (0 if c == 0 else BS)
    e = 0 if c < 0 else (BS if c == 0 else BS + ss)
    return s, e


def coarse_bounds(c, ss, coff):
    nc = BS // 2 + ss + 3
    s = coff if c < 0 else (0 if c == 0 else BS // 2)
    e = 0 if c < 0 else (BS // 2 if c == 0 else min(BS // 2 + (ss + 1) // 2 + 1, coff + nc))
    return s, e


def make_blksrc(level_delta=0, xi_mul=0, yi_mul=0, xi_add=0, yi_add=0,
                xi_shift=0, yi_shift=0, is_self=0, self_idx=0):
    return dict(level_delta=level_delta, xi_mul=xi_mul, yi_mul=yi_mul,
                xi_add=xi_add, yi_add=yi_add, xi_shift=xi_shift,
                yi_shift=yi_shift, is_self=is_self, self_idx=self_idx)


def build_entry(cx, cy, xp, yp, s, ss, dim):
    coff = (-ss - 1) // 2 - 1
    nm = 2 * ss + BS
    nc = BS // 2 + ss + 3
    eC = (ss + 1) // 2 + 2

    fs0, fe0 = ghost_bounds(cx, ss)
    fs1, fe1 = ghost_bounds(cy, ss)
    cs0, ce0 = coarse_bounds(cx, ss, coff)
    cs1, ce1 = coarse_bounds(cy, ss, coff)

    cstart = [0, 0]
    for d in range(2):
        cc = cx if d == 0 else cy
        coord = xp if d == 0 else yp
        base_d = (coord + cc + 2) % 2
        ce_d = 1 if (cc != 0) and (coord == (1 if cc < 0 else 0)) else 0
        cstart[d] = (max(cc, 0) * BS // 2 +
                     (1 - abs(cc)) * base_d * BS // 2 - cc * BS +
                     ce_d * cc * BS // 2)

    sC0 = (-ss - 1) // 2 if cx < 0 else (0 if cx == 0 else BS // 2)
    sC1 = (-ss - 1) // 2 if cy < 0 else (0 if cy == 0 else BS // 2)
    is_face = abs(cx) + abs(cy) == 1

    e = dict(cx=cx, cy=cy, n_blk=0, blk_src=[],
             fill=[], cbc=[], interp_ops=[], fbc=[])

    # ---- Outflow (s=6,7): zero-gradient ----
    if s in (6, 7):
        axis = s - 6
        side = 1 if (cx if axis == 0 else cy) > 0 else 0
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(is_self=1, self_idx=0)]
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                interior = [ix, iy]
                interior[axis] = 0 if side == 0 else BS - 1
                src = dim * ((interior[0] + ss) + nm * (interior[1] + ss))
                dst = dim * ((ix + ss) + nm * (iy + ss))
                e['fill'].append((OP_BC_SCALAR, 0, 0, 0, src, dst, 0, 0))
        return e

    # ---- Inflow (s=8,9): constant value ----
    if s in (8, 9):
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(is_self=1, self_idx=2)]
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                dst = dim * ((ix + ss) + nm * (iy + ss))
                e['fill'].append((OP_BC_FIXED, 0, 0, 0, 0, dst, 0, 0))
        return e

    # ---- Block sources ----
    if s == 0:
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(xi_mul=1, yi_mul=1, xi_add=cx, yi_add=cy)]
    elif s == 1:
        pat = get_child_pattern(cx, cy)
        e['n_blk'] = pat[5]
        e['blk_src'] = [make_blksrc(level_delta=1, xi_mul=2, yi_mul=2,
                                     xi_add=pat[4][b][0], yi_add=pat[4][b][1])
                        for b in range(pat[5])]
    elif s == 2:
        e['n_blk'] = 2
        e['blk_src'] = [
            make_blksrc(level_delta=-1, xi_mul=1, yi_mul=1,
                        xi_add=cx, yi_add=cy, xi_shift=1, yi_shift=1),
            make_blksrc(is_self=1, self_idx=0),
        ]

    # ---- Fill ops ----
    if s == 0:
        for iy in range(fs1, fe1):
            src_off = dim * (BS * (iy - cy * BS) + (fs0 - cx * BS))
            dst_off = dim * ((fs0 + ss) + (iy + ss) * nm)
            cols = fe0 - fs0
            e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, cols, 0))
        if dim > 0:
            lcs0, lcs1 = cs0, cs1
            lce0 = ce0 if cx < 1 else min(BS // 2 + eC - 1, coff + nc)
            lce1 = ce1 if cy < 1 else min(BS // 2 + eC - 1, coff + nc)
            cols = lce0 - lcs0
            if cols > 0:
                s0 = lcs0 + max(cx, 0) * (BS // 2) - cx * BS + min(0, cx) * cols
                s1_v = lcs1 + max(cy, 0) * (BS // 2) - cy * BS + min(0, cy) * (lce1 - lcs1)
                di = lcs0 - coff
                for iy in range(lcs1, lce1):
                    y0 = 2 * (iy - lcs1) + s1_v
                    src_off = dim * (BS * y0 + s0)
                    dst_off = dim * (di + (iy - coff) * nc)
                    e['fill'].append((OP_AVG, 0, 1, 0, src_off, dst_off, cols, BS))

    elif s == 1:
        pat = get_child_pattern(cx, cy)
        width = abs(cx) * (fe0 - fs0) + (1 - abs(cx)) * ((fe0 - fs0) // 2)
        B = 0
        for cnt in range(pat[5]):
            aux = (B % 2) if abs(cx) == 1 else (B // 2)
            di = abs(cx) * (fs0 + ss) + (1 - abs(cx)) * (fs0 + ss + (B % 2) * (fe0 - fs0) // 2)
            sx = fs0 - cx * BS + min(0, cx) * (fe0 - fs0)
            iy2 = fs1
            while iy2 < fe1:
                sy = 2 * (iy2 - cy * BS) + min(0, cy) * BS if abs(cy) == 1 else iy2
                dk = di + (abs(cy) * (iy2 + ss) +
                           (1 - abs(cy)) * (iy2 // 2 + ss + aux * (fe1 - fs1) // 2)) * nm
                src_off = dim * (BS * sy + sx)
                dst_off = dim * dk
                e['fill'].append((OP_AVG, cnt, 0, 0, src_off, dst_off, width, BS))
                iy2 += pat[3]
            B += pat[2]

    elif s == 2:
        for iy in range(cs1, ce1):
            src_off = dim * (BS * (iy + cstart[1]) + cs0 + cstart[0])
            dst_off = dim * (cs0 - coff + (iy - coff) * nc)
            cols = ce0 - cs0
            e['fill'].append((OP_COPY, 0, 1, 0, src_off, dst_off, cols, 0))
        for j in range(BS // 2):
            for i in range(BS // 2):
                if 1 < i < BS // 2 - 2 and 2 < j < BS // 2 - 2:
                    continue
                src_off = dim * (2 * i + ss + nm * (2 * j + ss))
                dst_off = dim * (i - coff + (j - coff) * nc)
                e['fill'].append((OP_AVG, 1, 1, 0, src_off, dst_off, 1, nm))

    # ---- Interpolation ops (coarser neighbor) ----
    if s == 2 and is_face:
        s0_i, s1_i = fs0, fs1
        e0, e1 = fe0, fe1
        for iy in range(s1_i, e1, 2):
            YY = (iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) // 2 + sC1 - coff
            y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
            iyp = -1 if abs(iy) % 2 == 1 else 1
            dy_val = 0.25 * (2 * y - 1)
            for ix in range(s0_i, e0, 2):
                XX = (ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) // 2 + sC0 - coff
                x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                ixp = -1 if abs(ix) % 2 == 1 else 1
                dx_val = 0.25 * (2 * x - 1)
                if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                    continue
                i1 = dim * (XX + nc * YY)
                j0 = dim * ((ix + ss) + nm * (iy + ss))
                dirn = 0 if cx != 0 else 1
                stride = nc if dirn == 0 else 1
                CC = YY if dirn == 0 else XX
                t = dy_val if dirn == 0 else dx_val
                branch = 0 if CC + coff == 0 else (1 if CC + coff == BS // 2 - 1 else 2)
                ok1 = s1_i <= iy + iyp < e1
                ok2 = s0_i <= ix + ixp < e0
                cs_off = stride * dim
                pts = _PTS[branch]
                srcs = tuple(i1 + p * cs_off for p in pts)
                # Lagrange weights: sub=1 → t=+1/4, sub=0 → t=-1/4
                wvp = FACE_W[branch, 1]  # t = +1/4
                wvm = FACE_W[branch, 0]  # t = -1/4
                if t < 0:
                    wvp, wvm = wvm, wvp
                sign_mask = sum((1 << k) for k in range(4) if FACE_DSIGN[dirn][k] < 0)
                dests = [j0, j0 + nm * dim * iyp, j0 + dim * ixp,
                         j0 + dim * ixp + nm * dim * iyp]
                ok = [True, ok1, ok2, ok1 and ok2]
                for k in range(4):
                    if not ok[k]:
                        continue
                    w = wvm if (sign_mask & (1 << k)) else wvp
                    e['interp_ops'].append((OP_INTERP3, w[0], w[1], w[2],
                                            srcs[0], dests[k], srcs[1], srcs[2]))

        li = next((j for j in range(4)
                    if LI_LE_TAB[j][0] == cx and LI_LE_TAB[j][1] == cy), -1)
        if li >= 0:
            for iy in range(s1_i, e1):
                for ix in range(s0_i, e0):
                    if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                        continue
                    ka = dim * ((ix + ss) + nm * (iy + ss))
                    x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                    y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
                    p = x if cx != 0 else y
                    is_LE, b_dx, b_dy, c_dx, c_dy = LI_LE_TAB[li][2][p]
                    kb = dim * ((ix + ss + b_dx) + nm * (iy + ss + b_dy))
                    kc = dim * ((ix + ss + c_dx) + nm * (iy + ss + c_dy))
                    e['interp_ops'].append((OP_LELI, 0, 0, is_LE, ka, kb, kc, 0))

    elif s == 2 and not is_face:
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                YY = (iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) // 2 + sC1
                XX = (ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) // 2 + sC0
                src_c = dim * (XX - coff + nc * (YY - coff))
                dst_m = dim * (ix + ss + nm * (iy + ss))
                x = abs(ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) % 2
                y = abs(iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) % 2
                e['interp_ops'].append((OP_INTERP9, 0, 0, x | (y << 1),
                                        src_c, dst_m, 0, 0))

    # ---- Wall/symmetry BC (s=3,4,5) ----
    # Mirror formula: m(i, side, L) = 2*side*L - 1 - i
    # s=3: reflect x, s=4: reflect y, s=5: reflect both
    if s in (3, 4, 5):
        refl = [s != 4, s != 3]  # which axes get reflected
        sd = [(1 + cx) // 2, (1 + cy) // 2]  # side per axis: 0=left/bottom, 1=right/top
        nrefl = refl[0] + refl[1]
        op = OP_BC_SCALAR if dim == 1 else (OP_BC_CORNER if nrefl == 2 else OP_BC_VECTOR)
        flags = 0 if nrefl == 2 else (0 if refl[0] else 1)
        cc = [cx, cy]

        # Ghost ranges: for wall faces, restrict to wall-normal ghost cells
        gs = [fs0, fs1]; ge = [fe0, fe1]
        for k in range(2):
            if refl[k]:
                gs[k] = -ss if sd[k] == 0 else BS
                ge[k] = 0 if sd[k] == 0 else BS + ss

        # Fine buffer: mirror = 2*sd*BS - 1 - i + ss, identity = i + ss
        # Corner (s=5) uses full ghost region; faces use wall-restricted range
        if s == 5:
            fgs = [fs0, fs1]; fge = [fe0, fe1]
        else:
            fgs = [gs[0] if refl[0] else fs0, gs[1] if refl[1] else fs1]
            fge = [ge[0] if refl[0] else fe0, ge[1] if refl[1] else fe1]
        for iy in range(fgs[1], fge[1]):
            for ix in range(fgs[0], fge[0]):
                p = [ix, iy]
                mp = [2*sd[k]*BS - 1 - p[k] + ss if refl[k] else p[k] + ss for k in range(2)]
                i0 = (ix + ss) + nm * (iy + ss)
                i1 = mp[0] + nm * mp[1]
                e['fbc'].append((op, 0, 0, flags, dim * i1, dim * i0, 0, 0))

        # Coarse buffer: mirror = 2*sd*(BS//2) - 1 - i - sI, identity = i - sI
        sI = (-ss - 1) // 2 - 1
        eI0 = (ss + 1) // 2 + 2
        H = BS // 2
        if s == 5:
            cgs = [cs0, cs1]; cge = [ce0, ce1]
        else:
            cgs = [0, 0]; cge = [0, 0]
            for k in range(2):
                cs_k, ce_k = (cs0, ce0) if k == 0 else (cs1, ce1)
                if refl[k]:
                    cgs[k] = sI if sd[k] == 0 else H
                    cge[k] = 0 if sd[k] == 0 else min(H + eI0 - 1, sI + nc)
                else:
                    cgs[k] = cs_k
                    cge[k] = ce_k
        for iy in range(cgs[1], cge[1]):
            for ix in range(cgs[0], cge[0]):
                p = [ix, iy]
                mp = [2*sd[k]*H - 1 - p[k] - sI if refl[k] else p[k] - sI for k in range(2)]
                i0 = (ix - sI) + nc * (iy - sI)
                i1 = mp[0] + nc * mp[1]
                e['cbc'].append((op, 0, 1, flags, dim * i1, dim * i0, 0, 0))

    return e


# ---- Binary serialization ----

ENTRY_SIZE = 20 + 8 + MAX_OPS * 20


def pack_op(op):
    return struct.pack('<bbbb iiii', *op)


def pack_blksrc(b):
    return struct.pack('<bbbbbbb?b',
                       b['level_delta'], b['xi_mul'], b['yi_mul'],
                       b['xi_add'], b['yi_add'], b['xi_shift'], b['yi_shift'],
                       bool(b['is_self']), b['self_idx'])


ZERO_OP = pack_op((0, 0, 0, 0, 0, 0, 0, 0))
ZERO_BLKSRC = make_blksrc()


def pack_entry(e):
    buf = bytearray()
    buf += struct.pack('<b', e['n_blk'])
    blks = list(e['blk_src'])
    while len(blks) < 2:
        blks.append(ZERO_BLKSRC)
    for b in blks:
        buf += pack_blksrc(b)
    buf += b'\x00'
    assert len(buf) == 20

    pre_ops = e['fill'] + e['cbc']
    post_ops = e['interp_ops'] + e['fbc']
    assert len(pre_ops) <= MAX_PRE, f"pre overflow: {len(pre_ops)} > {MAX_PRE}"
    assert len(post_ops) <= MAX_POST, f"post overflow: {len(post_ops)} > {MAX_POST}"

    buf += struct.pack('<ii', len(pre_ops), len(post_ops))
    for op in pre_ops:
        buf += pack_op(op)
    buf += ZERO_OP * (MAX_PRE - len(pre_ops))
    for op in post_ops:
        buf += pack_op(op)
    buf += ZERO_OP * (MAX_POST - len(post_ops))
    assert len(buf) == ENTRY_SIZE
    return bytes(buf)


def build_and_write(fname, ss, dim):
    max_pre = max_post = 0
    with open(fname, 'wb') as f:
        for cx in range(-1, 2):
            for cy in range(-1, 2):
                for xp in range(2):
                    for yp in range(2):
                        for s in range(N_STATUS):
                            if cx == 0 and cy == 0:
                                f.write(b'\x00' * ENTRY_SIZE)
                            else:
                                e = build_entry(cx, cy, xp, yp, s, ss, dim)
                                pre = len(e['fill']) + len(e['cbc'])
                                post = len(e['interp_ops']) + len(e['fbc'])
                                max_pre = max(max_pre, pre)
                                max_post = max(max_post, post)
                                f.write(pack_entry(e))
    n_entries = 3 * 3 * 2 * 2 * N_STATUS
    sz = n_entries * ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz // 1024} KB), max_pre={max_pre}, max_post={max_post}')


# ---- Poisson table ----

MAX_POISSON_OPS = 16
POISSON_ENTRY_SIZE = 4 + MAX_POISSON_OPS * 8

P_INTERP_OFF = [[-2, -1, 0], [2, 1, 0], [-1, 1, 0]]
P_INTERP_D1 = [[1/8, -1/2, 3/8], [-1/8, 1/2, -3/8], [-1/8, 1/8, 0]]
P_INTERP_D2 = [[1/32, -1/16, 1/32], [1/32, -1/16, 1/32], [1/32, 1/32, -1/16]]


def poisson_interp_ops(add, c_blk, cix, ciy, f_blk, fc_ix, fc_iy, ff_ix, ff_iy,
                       signInt, signTaylor, dir):
    add(f_blk, fc_ix, fc_iy, signInt * 2/3)
    add(f_blk, ff_ix, ff_iy, -signInt * 1/5)
    tf = signInt * 8/15
    add(c_blk, cix, ciy, tf)
    tang = ciy if dir == 0 else cix
    c = 0 if tang in (BS - 1, BS // 2 - 1) else (1 if tang in (0, BS // 2) else 2)
    for i in range(3):
        off = P_INTERP_OFF[c][i]
        ox, oy = (cix, ciy + off) if dir == 0 else (cix + off, ciy) if off else (cix, ciy)
        add(c_blk, ox, oy, signTaylor * tf * P_INTERP_D1[c][i])
        add(c_blk, ox, oy, tf * P_INTERP_D2[c][i])


def build_poisson_entry(edge, tc, parity, state):
    """Wide Laplacian (stride 2) for projection method.

    L(φ)_{i,j} = (-4φ_{i,j} + φ_{i+2,j} + φ_{i-2,j} + φ_{i,j+2} + φ_{i,j-2}) / (4h²)

    Each entry encodes ONE neighbor contribution for cell (ix,iy) at block edge.
    edge: 0=left(x-), 1=right(x+), 2=bottom(y-), 3=top(y+)
    tc: tangent coordinate (0..BS-1)
    state: 0=same-level, 1=same-level neighbor, 2=coarser, 3=finer
    """
    dir = edge >> 1   # 0=x, 1=y
    side = edge & 1   # 0=low, 1=high
    # stride-2 step in normal direction
    sign = 2 * side - 1
    dx, dy = 2 * (1 - dir) * sign, 2 * dir * sign
    # cell at the block edge
    ix = (0 if side == 0 else BS - 1) if dir == 0 else tc
    iy = tc if dir == 0 else (0 if side == 0 else BS - 1)

    ops = {}

    def add(blk, cx, cy, coeff):
        key = (blk, cx, cy)
        ops[key] = ops.get(key, 0.0) + coeff

    if state == 0:
        # Interior: stride-2 neighbor is within the block
        add(0, ix + dx, iy + dy, 1.0)
        add(0, ix, iy, -1.0)
    elif state == 1:
        # Same-level neighbor block: stride-2 neighbor crosses into neighbor
        # For side=0 (left edge, ix=0): need cell at ix-2 = neighbor's BS-2
        # For side=1 (right edge, ix=BS-1): need cell at ix+2 = neighbor's 1
        ne_normal = (BS - 2) if side == 0 else 1
        ne_ix = ne_normal if dir == 0 else tc
        ne_iy = tc if dir == 0 else ne_normal
        add(1, ne_ix, ne_iy, 1.0)
        add(0, ix, iy, -1.0)
    elif state == 2:
        # Coarser neighbor: interpolate from coarse grid
        cix = (1 - side) * (BS - 1) if dir == 0 else tc // 2 + parity * (BS // 2)
        ciy = tc // 2 + parity * (BS // 2) if dir == 0 else (1 - side) * (BS - 1)
        signTaylor = -1.0 if tc % 2 == 0 else 1.0
        poisson_interp_ops(add, 2, cix, ciy, 0, ix, iy, ix - dx // 2, iy - dy // 2,
                           1.0, signTaylor, dir)
        add(0, ix, iy, -1.0)
    elif state == 3:
        # Finer neighbor: restrict from fine grid
        fe0 = BS - 2 if side == 0 else 1  # stride-2 position in fine block
        fe1 = BS - 3 if side == 0 else 2  # one more step for gradient
        ft = (tc % (BS // 2)) * 2
        for dt in range(2):
            fc = (fe0, ft + dt) if dir == 0 else (ft + dt, fe0)
            ff = (fe1, ft + dt) if dir == 0 else (ft + dt, fe1)
            add(3, fc[0], fc[1], 1.0)
            signTaylor_f = -1.0 if dt == 0 else 1.0
            poisson_interp_ops(add, 0, ix, iy, 3, fc[0], fc[1], ff[0], ff[1],
                               -1.0, signTaylor_f, dir)

    return [(k[0], k[1], k[2], v) for k, v in ops.items() if abs(v) > 1e-15]


def pack_poisson_entry(ops):
    assert len(ops) <= MAX_POISSON_OPS, f"poisson ops overflow: {len(ops)}"
    buf = struct.pack('<i', len(ops))
    for op in ops:
        buf += struct.pack('<bbbx f', op[0], op[1], op[2], op[3])
    buf += b'\x00' * 8 * (MAX_POISSON_OPS - len(ops))
    assert len(buf) == POISSON_ENTRY_SIZE
    return buf


def build_poisson_table():
    fname = 'tab_poisson.bin'
    max_ops = 0
    with open(fname, 'wb') as f:
        for edge in range(4):
            for tc in range(BS):
                for parity in range(2):
                    for state in range(4):
                        ops = build_poisson_entry(edge, tc, parity, state)
                        max_ops = max(max_ops, len(ops))
                        f.write(pack_poisson_entry(ops))
    sz = 4 * BS * 2 * 4 * POISSON_ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz // 1024} KB), max ops={max_ops}')


if __name__ == '__main__':
    for ss, dim in [(1, 1), (1, 2), (2, 1), (4, 1)]:
        build_and_write(f'tab_ss{ss}_dim{dim}.bin', ss, dim)
    build_poisson_table()

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
# ## 5. Generate the lifted-wavelet stencil tables
#
# `gen_table.py` (uses `amriso`) writes the `tab_*.bin` files the solver loads.

# %%
if not os.path.exists("tab_ss1_dim1.bin"):
    subprocess.run([sys.executable, "gen_table.py"], check=True)
print("tables:", sorted(f for f in os.listdir(".") if f.startswith("tab_")))

# %% [markdown]
# ## 6. Compile the solver

# %%
subprocess.run(["gcc", "-O2", "-o", "main", "main.c", "-fopenmp", "-lm"],
               check=True)
print("built ./main")


# %% [markdown]
# ## 7. Run
#
# Uniform grid (`-AdaptSteps 0`), ν = 1e-4, CFL = 0.8, to t = 2.0 with dumps
# every 0.4 — so dump index 2 is t=0.8 (Fig. 2) and index 3 is t=1.2 (Fig. 3).

# %%
def run(size):
    level = LEVEL[size]
    out = f"run{size}"
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for b in os.listdir("."):
        if b.startswith("tab_") and b.endswith(".bin"):
            shutil.copy(b, out)
    print(f"=== run {size}x{size} (level {level}) ===")
    subprocess.run(
        ["../main", "-levelStart", str(level), "-levelMax", str(level),
         "-AdaptSteps", "0", "-Rtol", "0", "-Ctol", "0",
         "-nu", "1e-4", "-CFL", "0.8", "-tend", "2.0", "-tdump", "0.4"],
        cwd=out, check=True)

for size in RESOLUTIONS:
    run(size)

# %% [markdown]
# ## 8. Comparison: paper vs computed

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
fig.suptitle("Brown & Minion 1995 - double shear layer (rho=30, nu=1e-4)",
             fontsize=14)
fig.tight_layout()
fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
print("saved comparison.png")

# %%
from IPython.display import Image as IPyImage
IPyImage("comparison.png")
