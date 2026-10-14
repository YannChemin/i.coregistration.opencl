/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Robust model estimation from matches: RANSAC, MAGSAC++
 *               (Barath, Noskova, Ivashechkin and Matas, 2020), Graph-Cut
 *               RANSAC (Barath and Matas, 2018) and their combination as
 *               in SupeRANSAC (Barath, 2025), with PROSAC sampling (Chum
 *               and Matas, 2005) and the independent inliers of VSAC
 *               (Ivashechkin, Barath and Matas, 2021).
 *
 *               The MAGSAC++ quality and weights follow the paper and the
 *               authors' magsac code (new BSD licence, Copyright (C)
 *               Daniel Barath); the PROSAC growth function and the graph-
 *               cut energy follow superansac (MIT licence, Copyright (c)
 *               2024 Daniel Barath). Written anew in C for 2D residuals
 *               (two degrees of freedom), where the incomplete gamma
 *               functions have closed forms.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "maxflow.h"
#include "robust.h"

/* sqrt of the 0.99 quantile of the chi-squared distribution with two
 * degrees of freedom: the largest residual of an inlier is k sigma_max. */
#define K_SIGMA 3.0348542587702925
#define SQRT_PI 1.7724538509055160
#define MAX_SAMPLE 10
#define IRLS_ITERS 10
#define GC_ITERS 10

static const char *kind_names[ROBUST_NKINDS] = {"none", "ransac", "magsac",
                                                "gcransac", "superansac"};

const char *robust_name(enum robust_kind k) { return kind_names[k]; }

int robust_parse(const char *name)
{
    int k;

    for (k = 0; k < ROBUST_NKINDS; k++)
        if (strcmp(name, kind_names[k]) == 0)
            return k;
    return -1;
}

void robust_defaults(struct robust_config *c, enum robust_kind kind,
                     enum model_type model, double threshold)
{
    memset(c, 0, sizeof(*c));
    c->kind = kind;
    c->model = model;
    c->threshold = threshold;
    c->confidence = 0.99;
    c->min_iters = 50;
    c->max_iters = 5000;
    c->spacing = 0;
    c->scale_bound = 2;
    c->gc_lambda = 0.1;
    c->gc_radius = 20;
    c->gc_neighbours = 8;
    c->seed = 0x9e3779b97f4a7c15ULL;
}

/* Upper incomplete gamma Gamma(1/2, x) and lower gamma(3/2, x). */
static double upper_half(double x) { return SQRT_PI * erfc(sqrt(x)); }

static double lower_three_halves(double x)
{
    return 0.5 * SQRT_PI * erf(sqrt(x)) - sqrt(x) * exp(-x);
}

struct ctx {
    const double *pts;
    const struct robust_config *cfg;
    int n, m;
    double t2;           /* Squared threshold. */
    double s2;           /* sigma_max^2 for MAGSAC++. */
    double gk_up;        /* Gamma(1/2, k^2 / 2). */
    double mag_pre;      /* 2^(3/2) / sigma_max. */
    double outlier_loss; /* MAGSAC++ loss of an outlier. */
    double w_zero;       /* MAGSAC++ weight at zero residual. */
    double xc, yc, extent;
    double *mult; /* Multiplicity weights (independent inliers). */
    double *r2;   /* Squared residuals of the current model. */
    double *w;    /* Fit weights. */
    unsigned long long rng;
    int *nb_start, *nb; /* Neighbourhood graph (CSR), graph cut only. */
};

static unsigned long long next_rand(struct ctx *c)
{
    /* xorshift64*. */
    c->rng ^= c->rng >> 12;
    c->rng ^= c->rng << 25;
    c->rng ^= c->rng >> 27;
    return c->rng * 2685821657736338717ULL;
}

static int rand_below(struct ctx *c, int n)
{
    return (int)((next_rand(c) >> 11) % (unsigned long long)n);
}

/* MAGSAC++ loss of a squared residual (to be minimized). */
static double magsac_loss(const struct ctx *c, double r2)
{
    double x;

    if (r2 >= c->t2)
        return c->outlier_loss;
    x = r2 / (2 * c->s2);
    return c->mag_pre * (0.5 * c->s2 * lower_three_halves(x) +
                         0.25 * r2 * (upper_half(x) - c->gk_up));
}

/* MAGSAC++ weight relative to a zero residual. */
static double magsac_weight(const struct ctx *c, double r2)
{
    if (r2 >= c->t2)
        return 0;
    return (upper_half(r2 / (2 * c->s2)) - c->gk_up) / c->w_zero;
}

static int uses_magsac(enum robust_kind k)
{
    return k == ROBUST_MAGSAC || k == ROBUST_SUPERANSAC;
}

static int uses_gc(enum robust_kind k)
{
    return k == ROBUST_GCRANSAC || k == ROBUST_SUPERANSAC;
}

/* Quality of model mo (lower is better), stopping early once worse than
 * best; residuals into c->r2 (complete only when not stopped early). */
static double quality(struct ctx *c, const struct model *mo, double best)
{
    double loss = 0;
    int i;

    for (i = 0; i < c->n; i++) {
        double r2 = model_sqres(mo, c->pts + 4 * (size_t)i), l;

        if (!isfinite(r2))
            r2 = DBL_MAX;
        c->r2[i] = r2;
        switch (c->cfg->kind) {
        case ROBUST_MAGSAC:
        case ROBUST_SUPERANSAC:
            l = magsac_loss(c, r2);
            break;
        case ROBUST_GCRANSAC:
            l = fmin(r2, c->t2) / c->t2;
            break;
        default:
            l = r2 > c->t2;
        }
        loss += c->mult[i] * l;
        if (loss > best)
            return INFINITY;
    }
    return loss;
}

static int plausible(const struct ctx *c, const struct model *mo)
{
    return model_plausible(mo, c->xc, c->yc, c->cfg->scale_bound);
}

/* Least squares on the inliers of the residuals in c->r2. */
static int refit_inliers(struct ctx *c, struct model *mo)
{
    int i;

    for (i = 0; i < c->n; i++)
        c->w[i] = c->r2[i] <= c->t2 ? c->mult[i] : 0;
    return model_fit(mo, c->cfg->model, c->pts, NULL, c->n, c->w);
}

/* Iteratively reweighted least squares with the MAGSAC++ weights,
 * starting from *mo (residuals in c->r2); keeps the best quality. */
static void irls(struct ctx *c, struct model *mo, double *loss)
{
    struct model cand;
    int it, i;

    for (it = 0; it < IRLS_ITERS; it++) {
        double l;

        for (i = 0; i < c->n; i++)
            c->w[i] = c->mult[i] * magsac_weight(c, c->r2[i]);
        if (!model_fit(&cand, c->cfg->model, c->pts, NULL, c->n, c->w) ||
            !plausible(c, &cand))
            break;
        l = quality(c, &cand, INFINITY);
        if (!(l < *loss * (1 - 1e-9))) {
            quality(c, mo, INFINITY); /* Restore the residuals. */
            break;
        }
        *mo = cand;
        *loss = l;
    }
}

/* Neighbours of every match in the image within gc_radius, the
 * gc_neighbours nearest, through a grid of that cell size. */
static void neighbourhood(struct ctx *c)
{
    double r = c->cfg->gc_radius, x0 = INFINITY, y0 = INFINITY;
    int k = c->cfg->gc_neighbours, n = c->n, i, gw, gh, cell;
    int *head, *next;
    double *bestd;
    int *besti;

    for (i = 0; i < n; i++) {
        x0 = fmin(x0, c->pts[4 * (size_t)i]);
        y0 = fmin(y0, c->pts[4 * (size_t)i + 1]);
    }
    gw = gh = 1;
    for (i = 0; i < n; i++) {
        gw = (int)fmax(gw, (c->pts[4 * (size_t)i] - x0) / r + 2);
        gh = (int)fmax(gh, (c->pts[4 * (size_t)i + 1] - y0) / r + 2);
    }
    head = G_malloc((size_t)gw * gh * sizeof(int));
    for (i = 0; i < gw * gh; i++)
        head[i] = -1;
    next = G_malloc(n * sizeof(int));
    for (i = 0; i < n; i++) {
        int gx = (int)((c->pts[4 * (size_t)i] - x0) / r);
        int gy = (int)((c->pts[4 * (size_t)i + 1] - y0) / r);

        cell = gy * gw + gx;
        next[i] = head[cell];
        head[cell] = i;
    }
    c->nb_start = G_malloc((n + 1) * sizeof(int));
    c->nb = G_malloc((size_t)n * k * sizeof(int));
    bestd = G_malloc(k * sizeof(double));
    besti = G_malloc(k * sizeof(int));
    c->nb_start[0] = 0;
    for (i = 0; i < n; i++) {
        double xi = c->pts[4 * (size_t)i], yi = c->pts[4 * (size_t)i + 1];
        int gx = (int)((xi - x0) / r), gy = (int)((yi - y0) / r), nf = 0;
        int dx, dy, j;

        for (dy = -1; dy <= 1; dy++)
            for (dx = -1; dx <= 1; dx++) {
                if (gx + dx < 0 || gx + dx >= gw || gy + dy < 0 ||
                    gy + dy >= gh)
                    continue;
                for (j = head[(gy + dy) * gw + gx + dx]; j >= 0; j = next[j]) {
                    double d = hypot(c->pts[4 * (size_t)j] - xi,
                                     c->pts[4 * (size_t)j + 1] - yi);
                    int p;

                    if (j == i || d > r)
                        continue;
                    /* Insertion into the k nearest so far. */
                    if (nf == k && d >= bestd[k - 1])
                        continue;
                    p = nf < k ? nf++ : k - 1;
                    while (p > 0 && bestd[p - 1] > d) {
                        bestd[p] = bestd[p - 1];
                        besti[p] = besti[p - 1];
                        p--;
                    }
                    bestd[p] = d;
                    besti[p] = j;
                }
            }
        memcpy(c->nb + c->nb_start[i], besti, nf * sizeof(int));
        c->nb_start[i + 1] = c->nb_start[i] + nf;
    }
    G_free(head);
    G_free(next);
    G_free(bestd);
    G_free(besti);
}

/* Inlier labelling of the residuals in c->r2 by a minimum cut
 * (superansac's energy): truncated quadratic unary terms, Potts-like
 * pairwise terms between neighbours. */
static void gc_label(struct ctx *c, unsigned char *lab)
{
    double lam = c->cfg->gc_lambda, *d = c->w;
    struct maxflow *g = maxflow_new(c->n, c->nb_start[c->n]);
    int i, j;

    for (i = 0; i < c->n; i++) {
        d[i] = fmin(c->r2[i] / c->t2, 1);
        /* Label 1 = inlier. */
        if (c->r2[i] <= c->t2)
            maxflow_unary(g, i, (1 - lam) * (1 - d[i]), 0);
        else
            maxflow_unary(g, i, 0, (1 - lam) * d[i]);
    }
    if (lam > 0)
        for (i = 0; i < c->n; i++)
            for (j = c->nb_start[i]; j < c->nb_start[i + 1]; j++) {
                int q = c->nb[j];

                if (q <= i)
                    continue;
                maxflow_pairwise(g, i, q, lam * 0.5 * (d[i] + d[q]), lam, lam,
                                 0);
            }
    maxflow_solve(g);
    for (i = 0; i < c->n; i++)
        lab[i] = maxflow_label(g, i);
    maxflow_free(g);
}

/* Graph-cut local optimization from *mo (residuals in c->r2): label,
 * refit on the labelled inliers, keep while the quality improves. */
static void gc_optimize(struct ctx *c, struct model *mo, double *loss,
                        unsigned char *lab)
{
    struct model cand;
    int it, i;

    for (it = 0; it < GC_ITERS; it++) {
        double l;

        gc_label(c, lab);
        for (i = 0; i < c->n; i++)
            c->w[i] = lab[i] ? c->mult[i] : 0;
        if (!model_fit(&cand, c->cfg->model, c->pts, NULL, c->n, c->w) ||
            !plausible(c, &cand))
            break;
        l = quality(c, &cand, INFINITY);
        if (!(l < *loss * (1 - 1e-9))) {
            quality(c, mo, INFINITY);
            break;
        }
        *mo = cand;
        *loss = l;
    }
}

/* Multiplicity weights: matches sharing a spacing x spacing cell of the
 * image count once together. */
struct keyed {
    long long key;
    int i;
};

static int cmp_keyed(const void *a, const void *b)
{
    const struct keyed *p = a, *q = b;

    return (p->key > q->key) - (p->key < q->key);
}

static void multiplicity(struct ctx *c)
{
    double s = c->cfg->spacing;
    struct keyed *k;
    int i, j;

    for (i = 0; i < c->n; i++)
        c->mult[i] = 1;
    if (!(s > 0))
        return;
    k = G_malloc(c->n * sizeof(*k));
    for (i = 0; i < c->n; i++) {
        long long gx = (long long)floor(c->pts[4 * (size_t)i] / s);
        long long gy = (long long)floor(c->pts[4 * (size_t)i + 1] / s);

        k[i].key = gy * 4194304LL + gx;
        k[i].i = i;
    }
    qsort(k, c->n, sizeof(*k), cmp_keyed);
    for (i = 0; i < c->n; i = j) {
        int t;

        for (j = i + 1; j < c->n && k[j].key == k[i].key; j++)
            ;
        for (t = i; t < j; t++)
            c->mult[k[t].i] = 1.0 / (j - i);
    }
    G_free(k);
}

/* PROSAC (Chum and Matas, 2005): samples from a growing set of the
 * best-ranked matches, the newest always included; growth function as
 * superansac's. */
struct prosac {
    int *order, *growth, subset, n, m;
    long kth, converge;
};

static void prosac_init(struct prosac *p, const float *cost, int n, int m,
                        long converge)
{
    double tn = converge;
    long tnp = 1;
    int i;
    struct keyed *k = G_malloc(n * sizeof(*k));

    p->n = n;
    p->m = m;
    p->kth = 1;
    p->converge = converge;
    p->order = G_malloc(n * sizeof(int));
    p->growth = G_malloc(n * sizeof(int));
    /* Ascending cost; ties keep their order. Costs are scaled to integer
     * keys: only their order matters. */
    for (i = 0; i < n; i++) {
        k[i].key = (long long)(cost[i] * 1e6) * 1048576LL + i;
        k[i].i = i;
    }
    qsort(k, n, sizeof(*k), cmp_keyed);
    for (i = 0; i < n; i++)
        p->order[i] = k[i].i;
    G_free(k);
    for (i = 0; i < m; i++)
        tn *= (double)(m - i) / (n - i);
    for (i = 0; i < n; i++) {
        double tn1;

        if (i + 1 <= m) {
            p->growth[i] = tnp;
            continue;
        }
        tn1 = (double)(i + 1) * tn / (i + 1 - m);
        tnp += (long)ceil(tn1 - tn);
        p->growth[i] = tnp > 2000000000L ? 2000000000 : (int)tnp;
        tn = tn1;
    }
    p->subset = m;
}

/* m distinct indices below range into s. */
static void unique_sample(struct ctx *c, int range, int m, int *s)
{
    int i, j;

    for (i = 0; i < m; i++) {
        int v;

        do {
            v = rand_below(c, range);
            for (j = 0; j < i && s[j] != v; j++)
                ;
        } while (j < i);
        s[i] = v;
    }
}

static void prosac_sample(struct ctx *c, struct prosac *p, int *s)
{
    int i;

    if (p->kth > p->converge || p->subset >= p->n) {
        unique_sample(c, p->n, p->m, s);
    }
    else {
        unique_sample(c, p->subset - 1, p->m - 1, s);
        s[p->m - 1] = p->subset - 1;
    }
    for (i = 0; i < p->m; i++)
        s[i] = p->order[s[i]];
    p->kth++;
    if (p->kth <= p->converge && p->subset < p->n &&
        p->kth > p->growth[p->subset - 1])
        p->subset++;
}

/* Samples whose image points are almost coincident determine nothing. */
static int spread_ok(const struct ctx *c, const int *s, int m)
{
    double tol = 1e-3 * c->extent;
    int i, j;

    for (i = 0; i < m; i++)
        for (j = i + 1; j < m; j++)
            if (hypot(c->pts[4 * (size_t)s[i]] - c->pts[4 * (size_t)s[j]],
                      c->pts[4 * (size_t)s[i] + 1] -
                          c->pts[4 * (size_t)s[j] + 1]) < tol)
                return 0;
    return 1;
}

int robust_fit(const double *pts, int n, const float *cost,
               const struct robust_config *cfg, struct robust_result *res,
               double *residual, double *weight, unsigned char *inlier)
{
    struct ctx c;
    struct prosac pr;
    struct model best, mo;
    unsigned char *lab = NULL;
    double best_loss = INFINITY, xmin = INFINITY, xmax = -INFINITY;
    double ymin = INFINITY, ymax = -INFINITY, sigma;
    int s[MAX_SAMPLE], i, needed, use_prosac;

    memset(res, 0, sizeof(*res));
    for (i = 0; i < n; i++) {
        residual[i] = INFINITY;
        weight[i] = 0;
        inlier[i] = 0;
    }
    memset(&c, 0, sizeof(c));
    c.m = model_min_points(cfg->model);
    if (n < c.m || cfg->kind == ROBUST_NONE || !(cfg->threshold > 0))
        return 0;
    c.pts = pts;
    c.cfg = cfg;
    c.n = n;
    c.t2 = cfg->threshold * cfg->threshold;
    sigma = cfg->threshold / K_SIGMA;
    c.s2 = sigma * sigma;
    c.gk_up = upper_half(K_SIGMA * K_SIGMA / 2);
    c.mag_pre = pow(2, 1.5) / sigma;
    c.outlier_loss =
        sigma * M_SQRT2 * lower_three_halves(K_SIGMA * K_SIGMA / 2);
    c.w_zero = upper_half(0) - c.gk_up;
    c.rng = cfg->seed ? cfg->seed : 1;
    c.mult = G_malloc(n * sizeof(double));
    c.r2 = G_malloc(n * sizeof(double));
    c.w = G_malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        c.xc += pts[4 * (size_t)i] / n;
        c.yc += pts[4 * (size_t)i + 1] / n;
        xmin = fmin(xmin, pts[4 * (size_t)i]);
        xmax = fmax(xmax, pts[4 * (size_t)i]);
        ymin = fmin(ymin, pts[4 * (size_t)i + 1]);
        ymax = fmax(ymax, pts[4 * (size_t)i + 1]);
    }
    c.extent = fmax(xmax - xmin, ymax - ymin);
    multiplicity(&c);
    if (uses_gc(cfg->kind)) {
        neighbourhood(&c);
        lab = G_malloc(n);
    }
    use_prosac = cfg->kind == ROBUST_SUPERANSAC && cost;
    if (use_prosac)
        prosac_init(&pr, cost, n, c.m, 100000);

    needed = cfg->max_iters;
    for (res->iters = 0; res->iters < cfg->max_iters &&
                         (res->iters < cfg->min_iters || res->iters < needed);
         res->iters++) {
        double l;

        if (use_prosac)
            prosac_sample(&c, &pr, s);
        else
            unique_sample(&c, n, c.m, s);
        if (!spread_ok(&c, s, c.m) ||
            !model_fit(&mo, cfg->model, pts, s, c.m, NULL) ||
            !plausible(&c, &mo))
            continue;
        l = quality(&c, &mo, best_loss);
        if (!(l < best_loss))
            continue;
        best = mo;
        best_loss = l;
        /* Local optimization of every new best (LO-RANSAC). */
        if (cfg->kind == ROBUST_RANSAC) {
            if (refit_inliers(&c, &mo) && plausible(&c, &mo) &&
                (l = quality(&c, &mo, INFINITY)) < best_loss) {
                best = mo;
                best_loss = l;
            }
            else
                quality(&c, &best, INFINITY);
        }
        if (uses_gc(cfg->kind))
            gc_optimize(&c, &best, &best_loss, lab);
        if (uses_magsac(cfg->kind))
            irls(&c, &best, &best_loss);
        /* Adaptive termination on the inlier ratio of the best model. */
        {
            int ni = 0;
            double eps, p;

            for (i = 0; i < n; i++)
                ni += c.r2[i] <= c.t2;
            eps = (double)ni / n;
            p = pow(eps, c.m);
            if (p >= 1)
                needed = 0;
            else if (p > 0)
                needed = (int)fmin(cfg->max_iters,
                                   ceil(log(1 - cfg->confidence) / log1p(-p)));
        }
    }

    if (isfinite(best_loss)) {
        res->ok = 1;
        quality(&c, &best, INFINITY);
        if (uses_magsac(cfg->kind))
            irls(&c, &best, &best_loss);
        else if (cfg->kind == ROBUST_RANSAC) {
            if (refit_inliers(&c, &mo) && plausible(&c, &mo))
                best = mo;
            quality(&c, &best, INFINITY);
        }
        if (cfg->kind == ROBUST_GCRANSAC) {
            gc_label(&c, lab);
            for (i = 0; i < n; i++)
                c.w[i] = lab[i] ? c.mult[i] : 0;
            if (model_fit(&mo, cfg->model, pts, NULL, n, c.w) &&
                plausible(&c, &mo))
                best = mo;
            quality(&c, &best, INFINITY);
            gc_label(&c, lab);
        }
        res->model = best;
        for (i = 0; i < n; i++) {
            residual[i] = sqrt(c.r2[i]);
            if (cfg->kind == ROBUST_GCRANSAC)
                inlier[i] = lab[i];
            else
                inlier[i] = c.r2[i] <= c.t2;
            weight[i] =
                uses_magsac(cfg->kind) ? magsac_weight(&c, c.r2[i]) : inlier[i];
            res->ninliers += inlier[i];
        }
    }

    G_free(c.mult);
    G_free(c.r2);
    G_free(c.w);
    G_free(c.nb_start);
    G_free(c.nb);
    G_free(lab);
    if (use_prosac) {
        G_free(pr.order);
        G_free(pr.growth);
    }
    return res->ok;
}
