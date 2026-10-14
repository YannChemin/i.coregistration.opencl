/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      2D transformation models fitted to matches: similarity,
 *               affinity, homography (normalized DLT, Hartley and
 *               Zisserman) and polynomials of order 2 and 3, by weighted
 *               least squares, with leave-one-out residuals.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>

#include "models.h"

#define MAXP 20 /* Largest parameter count. */

static const char *names[MODEL_NTYPES] = {"similarity", "affine", "homography",
                                          "poly2", "poly3"};

const char *model_name(enum model_type t) { return names[t]; }

int model_parse(const char *name)
{
    int t;

    for (t = 0; t < MODEL_NTYPES; t++)
        if (strcmp(name, names[t]) == 0)
            return t;
    return -1;
}

/* Terms of the polynomial of each linear model, per axis. */
static int poly_terms(enum model_type t)
{
    switch (t) {
    case MODEL_AFFINE:
        return 3;
    case MODEL_POLY2:
        return 6;
    case MODEL_POLY3:
        return 10;
    default:
        return 0;
    }
}

int model_min_points(enum model_type t)
{
    switch (t) {
    case MODEL_SIMILARITY:
        return 2;
    case MODEL_HOMOGRAPHY:
        return 4;
    default:
        return poly_terms(t);
    }
}

/* Parameters of the linear models (all but the homography). */
static int nparams(enum model_type t)
{
    return t == MODEL_SIMILARITY ? 4 : 2 * poly_terms(t);
}

/* 1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3: the first nt. */
static void monomials(double x, double y, int nt, double *v)
{
    v[0] = 1;
    v[1] = x;
    v[2] = y;
    if (nt > 3) {
        v[3] = x * x;
        v[4] = x * y;
        v[5] = y * y;
    }
    if (nt > 6) {
        v[6] = x * x * x;
        v[7] = x * x * y;
        v[8] = x * y * y;
        v[9] = y * y * y;
    }
}

/* The two rows (X and Y equations) of a linear model's design matrix at
 * the normalized point (x, y). */
static void design_rows(enum model_type t, double x, double y, double *rx,
                        double *ry)
{
    int p = nparams(t), nt;

    memset(rx, 0, p * sizeof(double));
    memset(ry, 0, p * sizeof(double));
    if (t == MODEL_SIMILARITY) {
        /* X = a x - b y + tx, Y = b x + a y + ty. */
        rx[0] = x;
        rx[1] = -y;
        rx[2] = 1;
        ry[0] = y;
        ry[1] = x;
        ry[3] = 1;
        return;
    }
    nt = poly_terms(t);
    monomials(x, y, nt, rx);
    monomials(x, y, nt, ry + nt);
}

/* In-place Cholesky factorization of the p x p matrix a (lower part).
 * Returns 0 if a pivot is not positive or too small next to the
 * diagonal: a singular or badly conditioned system. */
static int cholesky(double *a, int p)
{
    double dmax = 0;
    int i, j, k;

    for (i = 0; i < p; i++)
        dmax = fmax(dmax, a[i * p + i]);
    if (!(dmax > 0))
        return 0;
    for (j = 0; j < p; j++) {
        double d = a[j * p + j];

        for (k = 0; k < j; k++)
            d -= a[j * p + k] * a[j * p + k];
        if (!(d > 1e-12 * dmax))
            return 0;
        a[j * p + j] = sqrt(d);
        for (i = j + 1; i < p; i++) {
            double s = a[i * p + j];

            for (k = 0; k < j; k++)
                s -= a[i * p + k] * a[j * p + k];
            a[i * p + j] = s / a[j * p + j];
        }
    }
    return 1;
}

/* Solve L L^T x = b in place, L from cholesky(). */
static void chol_solve(const double *l, int p, double *b)
{
    int i, k;

    for (i = 0; i < p; i++) {
        for (k = 0; k < i; k++)
            b[i] -= l[i * p + k] * b[k];
        b[i] /= l[i * p + i];
    }
    for (i = p - 1; i >= 0; i--) {
        for (k = i + 1; k < p; k++)
            b[i] -= l[k * p + i] * b[k];
        b[i] /= l[i * p + i];
    }
}

/* Eigenvectors of the symmetric 9 x 9 matrix a (destroyed) by cyclic
 * Jacobi rotations: eigenvalues in ev, eigenvectors in the columns of v. */
static void jacobi9(double *a, double *ev, double *v)
{
    const int n = 9;
    int i, j, k, sweep;

    for (i = 0; i < n * n; i++)
        v[i] = i % (n + 1) == 0;
    for (sweep = 0; sweep < 60; sweep++) {
        double off = 0;

        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++)
                off += a[i * n + j] * a[i * n + j];
        if (off < 1e-30)
            break;
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++) {
                double aij = a[i * n + j], th, t, c, s;

                if (fabs(aij) < 1e-300)
                    continue;
                th = (a[j * n + j] - a[i * n + i]) / (2 * aij);
                t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1));
                c = 1 / sqrt(t * t + 1);
                s = t * c;
                for (k = 0; k < n; k++) {
                    double ki = a[k * n + i], kj = a[k * n + j];

                    a[k * n + i] = c * ki - s * kj;
                    a[k * n + j] = s * ki + c * kj;
                }
                for (k = 0; k < n; k++) {
                    double ik = a[i * n + k], jk = a[j * n + k];

                    a[i * n + k] = c * ik - s * jk;
                    a[j * n + k] = s * ik + c * jk;
                }
                for (k = 0; k < n; k++) {
                    double ki = v[k * n + i], kj = v[k * n + j];

                    v[k * n + i] = c * ki - s * kj;
                    v[k * n + j] = s * ki + c * kj;
                }
            }
    }
    for (i = 0; i < n; i++)
        ev[i] = a[i * n + i];
}

#define PT(i) (pts + 4 * (size_t)(idx ? idx[i] : (i)))
#define WT(i) (w ? w[idx ? idx[i] : (i)] : 1.0)

/* Centre and scale (mean distance to the centre over sqrt 2) of the
 * points with positive weight, coordinates off and off + 1 of a match. */
static int normalization(const double *pts, const int *idx, int n,
                         const double *w, int off, double *nz)
{
    double cx = 0, cy = 0, d = 0, sw = 0;
    int i;

    for (i = 0; i < n; i++)
        if (WT(i) > 0) {
            cx += PT(i)[off];
            cy += PT(i)[off + 1];
            sw++;
        }
    if (sw == 0)
        return 0;
    cx /= sw;
    cy /= sw;
    for (i = 0; i < n; i++)
        if (WT(i) > 0)
            d += hypot(PT(i)[off] - cx, PT(i)[off + 1] - cy);
    d /= sw * M_SQRT2;
    if (!(d > 0))
        return 0;
    nz[0] = cx;
    nz[1] = cy;
    nz[2] = d;
    return 1;
}

static int fit_homography(struct model *m, const double *pts, const int *idx,
                          int n, const double *w)
{
    double a[81] = {0}, ev[9], v[81], emax = 0, e1 = INFINITY, e2 = INFINITY;
    int i, j, k, kmin = 0;

    for (i = 0; i < n; i++) {
        const double *p = PT(i);
        double wi = WT(i), x, y, X, Y, r[2][9];

        if (!(wi > 0))
            continue;
        x = (p[0] - m->sn[0]) / m->sn[2];
        y = (p[1] - m->sn[1]) / m->sn[2];
        X = (p[2] - m->rn[0]) / m->rn[2];
        Y = (p[3] - m->rn[1]) / m->rn[2];
        r[0][0] = x, r[0][1] = y, r[0][2] = 1, r[0][3] = 0, r[0][4] = 0;
        r[0][5] = 0, r[0][6] = -X * x, r[0][7] = -X * y, r[0][8] = -X;
        r[1][0] = 0, r[1][1] = 0, r[1][2] = 0, r[1][3] = x, r[1][4] = y;
        r[1][5] = 1, r[1][6] = -Y * x, r[1][7] = -Y * y, r[1][8] = -Y;
        for (k = 0; k < 2; k++)
            for (j = 0; j < 9; j++) {
                int l;

                for (l = j; l < 9; l++)
                    a[j * 9 + l] += wi * r[k][j] * r[k][l];
            }
    }
    for (j = 0; j < 9; j++)
        for (k = 0; k < j; k++)
            a[j * 9 + k] = a[k * 9 + j];
    jacobi9(a, ev, v);
    for (j = 0; j < 9; j++) {
        emax = fmax(emax, ev[j]);
        if (ev[j] < e1) {
            e2 = e1;
            e1 = ev[j];
            kmin = j;
        }
        else if (ev[j] < e2)
            e2 = ev[j];
    }
    /* A second near-zero eigenvalue: the matches do not determine H. */
    if (!(emax > 0) || e2 < 1e-10 * emax)
        return 0;
    for (j = 0; j < 9; j++)
        m->c[j] = v[j * 9 + kmin];
    return 1;
}

int model_fit(struct model *m, enum model_type t, const double *pts,
              const int *idx, int n, const double *w)
{
    double a[MAXP * MAXP] = {0}, b[MAXP] = {0}, rx[MAXP], ry[MAXP];
    int i, j, k, p, used = 0;

    for (i = 0; i < n; i++)
        used += WT(i) > 0;
    if (used < model_min_points(t))
        return 0;
    memset(m, 0, sizeof(*m));
    m->type = t;
    if (!normalization(pts, idx, n, w, 0, m->sn) ||
        !normalization(pts, idx, n, w, 2, m->rn))
        return 0;
    if (t == MODEL_HOMOGRAPHY)
        return fit_homography(m, pts, idx, n, w);

    p = nparams(t);
    for (i = 0; i < n; i++) {
        const double *q = PT(i);
        double wi = WT(i), X, Y;

        if (!(wi > 0))
            continue;
        design_rows(t, (q[0] - m->sn[0]) / m->sn[2],
                    (q[1] - m->sn[1]) / m->sn[2], rx, ry);
        X = (q[2] - m->rn[0]) / m->rn[2];
        Y = (q[3] - m->rn[1]) / m->rn[2];
        for (j = 0; j < p; j++) {
            b[j] += wi * (rx[j] * X + ry[j] * Y);
            for (k = 0; k <= j; k++)
                a[j * p + k] += wi * (rx[j] * rx[k] + ry[j] * ry[k]);
        }
    }
    if (!cholesky(a, p))
        return 0;
    chol_solve(a, p, b);
    memcpy(m->c, b, p * sizeof(double));
    return 1;
}

void model_apply(const struct model *m, double x, double y, double *X,
                 double *Y)
{
    double xn = (x - m->sn[0]) / m->sn[2], yn = (y - m->sn[1]) / m->sn[2];
    double u, v;

    if (m->type == MODEL_HOMOGRAPHY) {
        const double *h = m->c;
        double d = h[6] * xn + h[7] * yn + h[8];

        u = (h[0] * xn + h[1] * yn + h[2]) / d;
        v = (h[3] * xn + h[4] * yn + h[5]) / d;
    }
    else {
        double rx[MAXP], ry[MAXP];
        int j, p = nparams(m->type);

        design_rows(m->type, xn, yn, rx, ry);
        u = v = 0;
        for (j = 0; j < p; j++) {
            u += rx[j] * m->c[j];
            v += ry[j] * m->c[j];
        }
    }
    *X = u * m->rn[2] + m->rn[0];
    *Y = v * m->rn[2] + m->rn[1];
}

double model_sqres(const struct model *m, const double *p)
{
    double X, Y;

    model_apply(m, p[0], p[1], &X, &Y);
    return (X - p[2]) * (X - p[2]) + (Y - p[3]) * (Y - p[3]);
}

int model_plausible(const struct model *m, double x, double y, double bound)
{
    double h = m->sn[2] * 1e-3, j[4], X0, Y0, X1, Y1, t, d, s1, s2;

    model_apply(m, x - h, y, &X0, &Y0);
    model_apply(m, x + h, y, &X1, &Y1);
    j[0] = (X1 - X0) / (2 * h);
    j[2] = (Y1 - Y0) / (2 * h);
    model_apply(m, x, y - h, &X0, &Y0);
    model_apply(m, x, y + h, &X1, &Y1);
    j[1] = (X1 - X0) / (2 * h);
    j[3] = (Y1 - Y0) / (2 * h);
    d = j[0] * j[3] - j[1] * j[2];
    if (!(d > 0))
        return 0;
    /* Singular values from the trace and determinant of J^T J. */
    t = j[0] * j[0] + j[1] * j[1] + j[2] * j[2] + j[3] * j[3];
    s1 = sqrt(0.5 * (t + sqrt(fmax(t * t - 4 * d * d, 0))));
    s2 = d / s1;
    return s1 <= bound && s2 >= 1 / bound;
}

int model_loo(enum model_type t, const double *pts, const int *idx, int n,
              const double *w, double *loo)
{
    double a[MAXP * MAXP] = {0}, rx[MAXP], ry[MAXP], zx[MAXP], zy[MAXP];
    struct model m;
    int i, j, k, p;

    if (t == MODEL_HOMOGRAPHY || !model_fit(&m, t, pts, idx, n, w))
        return 0;
    p = nparams(t);
    for (i = 0; i < n; i++) {
        const double *q = PT(i);
        double wi = WT(i);

        if (!(wi > 0))
            continue;
        design_rows(t, (q[0] - m.sn[0]) / m.sn[2], (q[1] - m.sn[1]) / m.sn[2],
                    rx, ry);
        for (j = 0; j < p; j++)
            for (k = 0; k <= j; k++)
                a[j * p + k] += wi * (rx[j] * rx[k] + ry[j] * ry[k]);
    }
    cholesky(a, p);
    for (i = 0; i < n; i++) {
        const double *q = PT(i);
        double wi = WT(i), h[4] = {0}, ex, ey, det, X, Y;

        design_rows(t, (q[0] - m.sn[0]) / m.sn[2], (q[1] - m.sn[1]) / m.sn[2],
                    rx, ry);
        model_apply(&m, q[0], q[1], &X, &Y);
        ex = q[2] - X;
        ey = q[3] - Y;
        if (!(wi > 0)) {
            loo[i] = hypot(ex, ey);
            continue;
        }
        /* The 2 x 2 block of the hat matrix, w R N^-1 R^T, and the
         * deleted residual (I - H)^-1 e. */
        memcpy(zx, rx, p * sizeof(double));
        memcpy(zy, ry, p * sizeof(double));
        chol_solve(a, p, zx);
        chol_solve(a, p, zy);
        for (j = 0; j < p; j++) {
            h[0] += rx[j] * zx[j];
            h[1] += rx[j] * zy[j];
            h[2] += ry[j] * zx[j];
            h[3] += ry[j] * zy[j];
        }
        for (j = 0; j < 4; j++)
            h[j] = (j == 0 || j == 3) - wi * h[j];
        det = h[0] * h[3] - h[1] * h[2];
        if (fabs(det) < 1e-12) {
            loo[i] = INFINITY;
            continue;
        }
        loo[i] = hypot((h[3] * ex - h[1] * ey) / det,
                       (-h[2] * ex + h[0] * ey) / det);
    }
    return 1;
}
