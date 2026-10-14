/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Check of the robust estimation stage (models.c,
 *               maxflow.c, robust.c) on synthetic matches: a known model,
 *               Gaussian noise, uniform outliers and coherent clusters
 *               (moved objects). Each estimator must recover the model
 *               within tolerance and label the inliers. The minimum cut
 *               is checked against brute force on small graphs, the
 *               leave-one-out residuals against refits. CPU only. Built
 *               by "make robust_check", not installed; exits 1 on any
 *               failure. With --fit, estimates one model from a file of
 *               matches (see fit_file) for tests/robust_compare.py.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>

#include "maxflow.h"
#include "models.h"
#include "robust.h"

#define SIDE 1000.0
#define NOISE 0.3

static unsigned long long rng = 12345;
static int failures;

static double urand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (rng >> 11) * (1.0 / 9007199254740992.0);
}

static double nrand(void)
{
    double u = urand() + 1e-300, v = urand();

    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

static void check(int ok, const char *fmt, ...)
{
    va_list ap;

    printf("%s ", ok ? "ok  " : "FAIL");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    failures += !ok;
}

/* A true model of each type: close to the identity, as between images
 * already roughly aligned. */
static void truth(enum model_type t, double x, double y, double *X, double *Y)
{
    double u = x / SIDE - 0.5, v = y / SIDE - 0.5;

    switch (t) {
    case MODEL_SIMILARITY:
        *X = 1.001 * (cos(0.004) * x - sin(0.004) * y) + 3.3;
        *Y = 1.001 * (sin(0.004) * x + cos(0.004) * y) - 2.7;
        break;
    case MODEL_AFFINE:
        *X = 1.002 * x + 0.003 * y + 1.5;
        *Y = -0.002 * x + 0.998 * y + 4.0;
        break;
    case MODEL_HOMOGRAPHY: {
        double d = 1 + 2e-5 * x - 1e-5 * y;

        *X = (1.001 * x + 0.002 * y + 2) / d;
        *Y = (-0.001 * x + 0.999 * y - 3) / d;
        break;
    }
    case MODEL_POLY2:
        *X = x + 2 + 3 * u * u - 2 * u * v;
        *Y = y - 1 + 2 * v * v + u;
        break;
    default:
        *X = x + 1 + 2 * u * u * u - v * v + u * v;
        *Y = y + 2 - 3 * v * v * v + u * u * v;
    }
}

/* n matches: inliers with noise, a fraction of uniform outliers, and
 * clusters of matches all shifted by the same 15-30 units (moved
 * objects). truth_in[i] is 1 for the inliers. Cost: the inliers are a
 * bit better on average, as matcher scores are. */
static void make_set(enum model_type t, int n, double outliers, int clusters,
                     double *pts, float *cost, unsigned char *truth_in)
{
    int i, k = 0, per = clusters ? n / 20 : 0;

    for (i = 0; i < n; i++) {
        double x = urand() * SIDE, y = urand() * SIDE, X, Y;

        truth(t, x, y, &X, &Y);
        truth_in[i] = 1;
        cost[i] = (float)urand();
        if (k < clusters * per) {
            /* Cluster c: around a centre, one common shift. */
            int c = k / per;
            double cx = 150 + 700 * fmod(c * 0.618, 1.0);
            double cy = 150 + 700 * fmod(c * 0.414, 1.0);
            double a = c * 2.1, d = 15 + 3 * c;

            x = cx + (urand() - 0.5) * 60;
            y = cy + (urand() - 0.5) * 60;
            truth(t, x, y, &X, &Y);
            X += d * cos(a);
            Y += d * sin(a);
            truth_in[i] = 0;
            cost[i] = (float)(0.2 * urand());
            k++;
        }
        else if (urand() < outliers) {
            X = urand() * SIDE;
            Y = urand() * SIDE;
            truth_in[i] = 0;
            cost[i] = (float)(0.3 + 0.7 * urand());
        }
        else {
            X += NOISE * nrand();
            Y += NOISE * nrand();
            cost[i] = (float)(0.6 * urand());
        }
        pts[4 * i] = x;
        pts[4 * i + 1] = y;
        pts[4 * i + 2] = X;
        pts[4 * i + 3] = Y;
    }
}

/* Largest distance between the model and the truth over a grid. */
static double model_error(const struct model *m, enum model_type t)
{
    double e = 0, x, y;

    for (y = 0; y <= SIDE; y += SIDE / 10)
        for (x = 0; x <= SIDE; x += SIDE / 10) {
            double X, Y, Xt, Yt;

            model_apply(m, x, y, &X, &Y);
            truth(t, x, y, &Xt, &Yt);
            e = fmax(e, hypot(X - Xt, Y - Yt));
        }
    return e;
}

static void check_models(void)
{
    int t, n = 200, i;
    double *pts = G_malloc(4 * n * sizeof(double));
    float *cost = G_malloc(n * sizeof(float));
    unsigned char *tin = G_malloc(n);
    double *loo = G_malloc(n * sizeof(double));

    for (t = 0; t < MODEL_NTYPES; t++) {
        struct model m;
        double e;
        int idx[10];

        /* Exact fit to noise-free minimal samples. */
        make_set(t, n, 0, 0, pts, cost, tin);
        for (i = 0; i < n; i++)
            truth(t, pts[4 * i], pts[4 * i + 1], &pts[4 * i + 2],
                  &pts[4 * i + 3]);
        for (i = 0; i < model_min_points(t); i++)
            idx[i] = i;
        check(model_fit(&m, t, pts, idx, model_min_points(t), NULL) &&
                  (e = model_error(&m, t)) < (t >= MODEL_POLY2 ? 1 : 1e-6),
              "%-10s minimal sample: error %.2e (tolerance %g)", model_name(t),
              model_fit(&m, t, pts, idx, model_min_points(t), NULL)
                  ? model_error(&m, t)
                  : -1,
              t >= MODEL_POLY2 ? 1 : 1e-6);
        /* Least squares on all points recovers the truth exactly. */
        check(model_fit(&m, t, pts, NULL, n, NULL) && model_error(&m, t) < 1e-6,
              "%-10s least squares: error %.2e (tolerance %g)", model_name(t),
              model_error(&m, t), 1e-6);
        /* Collinear points are degenerate for all but the similarity. */
        for (i = 0; i < n; i++) {
            pts[4 * i + 1] = 0.5 * pts[4 * i];
            truth(t, pts[4 * i], pts[4 * i + 1], &pts[4 * i + 2],
                  &pts[4 * i + 3]);
        }
        if (t != MODEL_SIMILARITY)
            check(!model_fit(&m, t, pts, NULL, n, NULL),
                  "%-10s collinear points rejected", model_name(t));
    }

    /* Leave-one-out residuals against explicit refits. */
    make_set(MODEL_POLY2, n, 0, 0, pts, cost, tin);
    for (t = 0; t < MODEL_NTYPES; t++) {
        double worst = 0;
        int j, *idx = G_malloc(n * sizeof(int));

        if (t == MODEL_HOMOGRAPHY)
            continue;
        model_loo(t, pts, NULL, n, NULL, loo);
        for (i = 0; i < n; i += 37) {
            struct model m;
            double X, Y;
            int k = 0;

            for (j = 0; j < n; j++)
                if (j != i)
                    idx[k++] = j;
            model_fit(&m, t, pts, idx, k, NULL);
            model_apply(&m, pts[4 * i], pts[4 * i + 1], &X, &Y);
            worst =
                fmax(worst, fabs(hypot(X - pts[4 * i + 2], Y - pts[4 * i + 3]) -
                                 loo[i]));
        }
        check(worst < 1e-8, "%-10s leave-one-out vs refit: %.2e (%g)",
              model_name(t), worst, 1e-8);
        G_free(idx);
    }
    G_free(pts);
    G_free(cost);
    G_free(tin);
    G_free(loo);
}

/* Minimum cut against brute force on random submodular energies. */
static void check_maxflow(void)
{
    int trial, worst_ok = 1;

    for (trial = 0; trial < 300; trial++) {
        int n = 2 + trial % 11, i, j, lab, best_lab = 0, e;
        double u[12][2], p[12][12][4], best = INFINITY, got;
        int has[12][12];
        struct maxflow *g = maxflow_new(n, n * n);

        memset(has, 0, sizeof(has));
        for (i = 0; i < n; i++) {
            u[i][0] = urand() * 3;
            u[i][1] = urand() * 3;
            maxflow_unary(g, i, u[i][0], u[i][1]);
        }
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++)
                if (urand() < 0.4) {
                    double a = urand(), d = urand(), b = urand() * 2,
                           c = a + d - b + urand() * 2;

                    if (c < 0)
                        c = urand();
                    if (b + c < a + d)
                        b = a + d - c;
                    p[i][j][0] = a;
                    p[i][j][1] = b;
                    p[i][j][2] = c;
                    p[i][j][3] = d;
                    has[i][j] = 1;
                    maxflow_pairwise(g, i, j, a, b, c, d);
                }
        got = maxflow_solve(g);
        for (lab = 0; lab < (1 << n); lab++) {
            double en = 0;

            for (i = 0; i < n; i++)
                en += u[i][(lab >> i) & 1];
            for (i = 0; i < n; i++)
                for (j = i + 1; j < n; j++)
                    if (has[i][j])
                        en += p[i][j][((lab >> i) & 1) * 2 + ((lab >> j) & 1)];
            if (en < best) {
                best = en;
                best_lab = lab;
            }
        }
        /* The cut's labelling must reach the minimum energy. */
        {
            double en = 0;

            for (i = 0; i < n; i++)
                en += u[i][maxflow_label(g, i)];
            for (i = 0; i < n; i++)
                for (j = i + 1; j < n; j++)
                    if (has[i][j])
                        en += p[i][j]
                               [maxflow_label(g, i) * 2 + maxflow_label(g, j)];
            e = fabs(en - best) < 1e-9 && fabs(got - best) < 1e-9;
        }
        (void)best_lab;
        if (!e) {
            worst_ok = 0;
            printf("  maxflow trial %d: min %g, cut %g\n", trial, best, got);
        }
        maxflow_free(g);
    }
    check(worst_ok, "maxflow against brute force on 300 graphs");
}

static void check_robust(void)
{
    static const struct {
        enum model_type t;
        double outliers;
        int clusters;
    } sets[] = {
        {MODEL_SIMILARITY, 0.3, 0}, {MODEL_AFFINE, 0.5, 0},
        {MODEL_AFFINE, 0.3, 4},     {MODEL_HOMOGRAPHY, 0.5, 2},
        {MODEL_POLY2, 0.4, 3},      {MODEL_POLY3, 0.3, 2},
        {MODEL_AFFINE, 0.8, 0},
    };
    int n = 1500, s, k;
    double *pts = G_malloc(4 * n * sizeof(double));
    float *cost = G_malloc(n * sizeof(float));
    unsigned char *tin = G_malloc(n), *in = G_malloc(n);
    double *res = G_malloc(n * sizeof(double)),
           *w = G_malloc(n * sizeof(double));

    for (s = 0; s < (int)(sizeof(sets) / sizeof(sets[0])); s++) {
        make_set(sets[s].t, n, sets[s].outliers, sets[s].clusters, pts, cost,
                 tin);
        for (k = ROBUST_RANSAC; k < ROBUST_NKINDS; k++) {
            struct robust_config cfg;
            struct robust_result r;
            int i, tp = 0, fp = 0, pos = 0;
            double e;
            char what[96];

            robust_defaults(&cfg, k, sets[s].t, 3 * NOISE * 3);
            cfg.spacing = 10;
            robust_fit(pts, n, cost, &cfg, &r, res, w, in);
            for (i = 0; i < n; i++) {
                pos += tin[i];
                tp += in[i] && tin[i];
                fp += in[i] && !tin[i];
            }
            e = r.ok ? model_error(&r.model, sets[s].t) : INFINITY;
            snprintf(what, sizeof(what),
                     "%-10s %-10s outliers %2.0f%% clusters %d (%5d iters):",
                     robust_name(k), model_name(sets[s].t),
                     100 * sets[s].outliers, sets[s].clusters, r.iters);
            /* The model within 1 unit of the truth everywhere (noise 0.3),
             * at least 95 % of the inliers found, at most 2 % of the
             * labelled inliers wrong. */
            check(r.ok && e < 1 && tp >= 0.95 * pos && fp <= 0.02 * (tp + fp),
                  "%s model error %.3f, recall %.3f, precision %.3f", what, e,
                  (double)tp / pos, tp + fp ? (double)tp / (tp + fp) : 0.0);
        }
    }
    G_free(pts);
    G_free(cost);
    G_free(tin);
    G_free(in);
    G_free(res);
    G_free(w);
}

/* --fit mode, for tests/robust_compare.py: read matches (x y X Y cost
 * per line) from a file, estimate, print "inlier <0|1>" per match then
 * "map x y X Y" for the model on an 11 x 11 grid over [0, side]. */
static int fit_file(int argc, char **argv)
{
    struct robust_config cfg;
    struct robust_result r;
    double *pts = NULL, *res, *w, side;
    float *cost = NULL;
    unsigned char *in;
    int n = 0, cap = 0, i, k, t;
    FILE *fp;

    if (argc != 7)
        G_fatal_error("usage: robust_check --fit file kind model threshold "
                      "side");
    k = robust_parse(argv[3]);
    t = model_parse(argv[4]);
    if (k < 0 || t < 0 || !(fp = fopen(argv[2], "r")))
        G_fatal_error("bad --fit arguments");
    side = atof(argv[6]);
    for (;;) {
        double v[5];

        if (fscanf(fp, "%lf %lf %lf %lf %lf", v, v + 1, v + 2, v + 3, v + 4) !=
            5)
            break;
        if (n == cap) {
            cap = cap ? 2 * cap : 1024;
            pts = G_realloc(pts, 4 * cap * sizeof(double));
            cost = G_realloc(cost, cap * sizeof(float));
        }
        memcpy(pts + 4 * n, v, 4 * sizeof(double));
        cost[n++] = (float)v[4];
    }
    fclose(fp);
    res = G_malloc((n + 1) * sizeof(double));
    w = G_malloc((n + 1) * sizeof(double));
    in = G_malloc(n + 1);
    robust_defaults(&cfg, k, t, atof(argv[5]));
    cfg.spacing = 10;
    robust_fit(pts, n, cost, &cfg, &r, res, w, in);
    for (i = 0; i < n; i++)
        printf("inlier %d\n", in[i]);
    if (r.ok) {
        double x, y;

        for (y = 0; y <= side + 1e-9; y += side / 10)
            for (x = 0; x <= side + 1e-9; x += side / 10) {
                double X, Y;

                model_apply(&r.model, x, y, &X, &Y);
                printf("map %.6f %.6f %.6f %.6f\n", x, y, X, Y);
            }
    }
    return 0;
}

int main(int argc, char **argv)
{
    G_no_gisinit();
    if (argc > 1 && strcmp(argv[1], "--fit") == 0)
        return fit_file(argc, argv);
    check_models();
    check_maxflow();
    check_robust();
    printf("%d failure(s)\n", failures);
    return failures > 0;
}
