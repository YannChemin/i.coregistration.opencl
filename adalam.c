/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Nearest-neighbour descriptor matching and AdaLAM outlier
 *               filtering (adaptive locally-affine matching), following
 *               kornia.feature.adalam step by step.
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

#include "adalam.h"

void adalam_defaults(struct adalam_config *c)
{
    c->area_ratio = 100;
    c->search_expansion = 16;
    c->ransac_iters = 256;
    c->min_inliers = 6;
    c->min_confidence = 200;
    c->det_thr = 5;
    c->force_seed_mnn = 0;
}

/* Nearest neighbour of each d1 in d2 by the squared distance
 * |a|^2 + |b|^2 - 2 a.b (kornia dist_matrix), Lowe's ratio of the two
 * smallest distances (the second clamped to 1e-3) and, when nn21 is not
 * NULL, the nearest d1 of each d2. */
static void nearest(const float *d1, int n1, const float *d2, int n2, int dim,
                    int *nn12, float *score, int *nn21)
{
    float *dm = G_malloc((size_t)n1 * n2 * sizeof(float));
    float *q1 = G_malloc(n1 * sizeof(float)), *q2 = G_malloc(n2 * sizeof(float));
    int i, j, k;

    for (i = 0; i < n1; i++) {
        float s = 0;

        for (k = 0; k < dim; k++)
            s += d1[(size_t)i * dim + k] * d1[(size_t)i * dim + k];
        q1[i] = s;
    }
    for (j = 0; j < n2; j++) {
        float s = 0;

        for (k = 0; k < dim; k++)
            s += d2[(size_t)j * dim + k] * d2[(size_t)j * dim + k];
        q2[j] = s;
    }
#pragma omp parallel for private(j, k) schedule(static)
    for (i = 0; i < n1; i++) {
        const float *a = d1 + (size_t)i * dim;
        float best = FLT_MAX, second = FLT_MAX;
        int bj = 0;

        for (j = 0; j < n2; j++) {
            const float *b = d2 + (size_t)j * dim;
            float dot = 0, d;

            for (k = 0; k < dim; k++)
                dot += a[k] * b[k];
            d = q1[i] + q2[j] - 2.0f * dot;
            dm[(size_t)i * n2 + j] = d;
            if (d < best) {
                second = best;
                best = d;
                bj = j;
            }
            else if (d < second)
                second = d;
        }
        nn12[i] = bj;
        score[i] = best / (second > 1e-3f ? second : 1e-3f);
    }
    if (nn21)
        for (j = 0; j < n2; j++) {
            float best = FLT_MAX;
            int bi = 0;

            for (i = 0; i < n1; i++)
                if (dm[(size_t)i * n2 + j] < best) {
                    best = dm[(size_t)i * n2 + j];
                    bi = i;
                }
            nn21[j] = bi;
        }
    G_free(dm);
    G_free(q1);
    G_free(q2);
}

/* The neighbourhoods' samples, concatenated by neighbourhood (RANSAC
 * problem): positions relative to the seed in both images, scaled by
 * the search radius, and the keypoints they come from. */
struct problems {
    int nr, N;
    int *off, *rd; /* Per problem: first sample, number of samples. */
    int *grp;      /* Per sample: its problem. */
    float *x, *y;  /* Per sample: (x, y) in image 1, in image 2. */
    int *kp1, *kp2;
};

struct keyed {
    float v;
    int i;
};

static int cmp_keyed(const void *a, const void *b)
{
    const struct keyed *p = a, *q = b;

    if (p->v < q->v)
        return -1;
    if (p->v > q->v)
        return 1;
    return (p->i > q->i) - (p->i < q->i);
}

/* Round to IEEE half precision and back (torch .half().float()). */
static float to_half(float v) { return (float)(_Float16)v; }

/* Residuals |A x - y| of every sample under its problem's affinity
 * (A: a00, a01, a10, a11 per problem), sorted ascending within each
 * problem: sample indices into idx, squared residuals into sq. */
static void sorted_residuals(const struct problems *p, const float *A,
                             float *sq, int *idx, struct keyed *tmp)
{
    int r, s, k;

    for (s = 0; s < p->N; s++) {
        const float *a = A + 4 * p->grp[s];
        float px = a[0] * p->x[2 * s] + a[1] * p->x[2 * s + 1];
        float py = a[2] * p->x[2 * s] + a[3] * p->x[2 * s + 1];

        tmp[s].v = hypotf(px - p->y[2 * s], py - p->y[2 * s + 1]);
        tmp[s].i = s;
    }
    for (r = 0; r < p->nr; r++)
        qsort(tmp + p->off[r], p->rd[r], sizeof(struct keyed), cmp_keyed);
    for (k = 0; k < p->N; k++) {
        idx[k] = tmp[k].i;
        sq[k] = tmp[k].v * tmp[k].v;
    }
}

/* Weight of each sorted column, 1 / the number of consecutive columns
 * equal (in half precision) over every row, from eq[j] (column j equals
 * column j + 1 in every row so far). */
static void duplicate_weights(const unsigned char *eq, int N, float *w)
{
    int j = 0;

    while (j < N) {
        int e = j, k;

        while (e < N - 1 && eq[e])
            e++;
        for (k = j; k <= e; k++)
            w[k] = 1.0f / (e - j + 1);
        j = e + 1;
    }
}

/* Confidence-based inlier selection on one row of sorted squared
 * residuals: per problem, the (truncated) weighted count of samples
 * whose squared residual times min_conf stays within the progressive
 * inlier rate, the problem's weight sum and the squared residual of the
 * last counted sample. */
static void select_inliers(const struct problems *p, const float *sq,
                           const float *w0, double min_conf, long *cnt,
                           double *bal, float *hi)
{
    int r, k;

    for (r = 0; r < p->nr; r++) {
        int o = p->off[r], n = p->rd[r];
        double b = 0, cum = 0, c = 0;

        for (k = o; k < o + n; k++)
            b += sq[k] <= 1e-8f ? 0.0 : w0[k];
        for (k = o; k < o + n; k++) {
            int perfect = sq[k] <= 1e-8f;
            double w = perfect ? 0.0 : w0[k];

            cum += w;
            if (perfect || sq[k] * min_conf <= cum / b)
                c += w;
        }
        cnt[r] = (long)c;
        bal[r] = b;
        hi[r] = cnt[r] > 0 ? sq[o + cnt[r] - 1] : 0.0f;
    }
}

static float confidence(long cnt, double bal, float hi)
{
    return cnt > 0 ? (float)(cnt / (bal * hi)) : 0.0f;
}

/* Inverse of a 2x2 matrix (a, b, c, d) with |det| < 1e-10 set to 1e-10. */
static void inv2(const double *m, double *o)
{
    double det = m[0] * m[3] - m[1] * m[2];

    if (fabs(det) < 1e-10)
        det = 1e-10;
    o[0] = m[3] / det;
    o[1] = -m[1] / det;
    o[2] = -m[2] / det;
    o[3] = m[0] / det;
}

/* The "first k couples" of kornia's draw_first_k_couples: all pairs
 * (a, b), b < a, of the first samples, exhaustively, then the rest. */
static void first_couples(int k, int *ca, int *cb)
{
    int m = (int)(sqrt(2.0 * k + 0.25) - 0.5);
    int rest = k - m * (m + 1) / 2, t = 0, a, b;

    for (a = 1; a <= m; a++)
        for (b = 0; b < a; b++, t++) {
            ca[t] = a;
            cb[t] = b;
        }
    for (b = 0; b < rest; b++, t++) {
        ca[t] = rest;
        cb[t] = b;
    }
}

struct match {
    int i1, i2, order;
    float dist;
};

static int cmp_match(const void *a, const void *b)
{
    const struct match *p = a, *q = b;

    if (p->i1 != q->i1)
        return p->i1 - q->i1;
    if (p->i2 != q->i2)
        return p->i2 - q->i2;
    return p->order - q->order;
}

/* Sort the matches and keep the first of each identical pair. */
static int unique_matches(struct match *m, int n, int **pairs, float **dist)
{
    int i, k = 0;

    qsort(m, n, sizeof(struct match), cmp_match);
    *pairs = G_malloc((2 * n + 2) * sizeof(int));
    *dist = G_malloc((n + 1) * sizeof(float));
    for (i = 0; i < n; i++) {
        if (k > 0 && (*pairs)[2 * k - 2] == m[i].i1 &&
            (*pairs)[2 * k - 1] == m[i].i2)
            continue;
        (*pairs)[2 * k] = m[i].i1;
        (*pairs)[2 * k + 1] = m[i].i2;
        (*dist)[k++] = m[i].dist;
    }
    return k;
}

/* Squared radius of the seed circles from the keypoint extent (kornia
 * infers the image shape from the keypoints when it is not a tuple, as
 * in the repository). */
static double seed_radius(const float *k, int n, double area_ratio)
{
    float x0 = FLT_MAX, y0 = FLT_MAX, x1 = -FLT_MAX, y1 = -FLT_MAX;
    int i;

    for (i = 0; i < n; i++) {
        x0 = fminf(x0, k[2 * i]);
        x1 = fmaxf(x1, k[2 * i]);
        y0 = fminf(y0, k[2 * i + 1]);
        y1 = fmaxf(y1, k[2 * i + 1]);
    }
    return sqrt((double)(x1 - x0) * (y1 - y0) / area_ratio / M_PI);
}

static float sqdist(const float *a, const float *b)
{
    float dx = a[0] - b[0], dy = a[1] - b[1];

    return dx * dx + dy * dy;
}

int adalam_match(const float *k1, const float *d1, int n1, const float *k2,
                 const float *d2, int n2, int dim,
                 const struct adalam_config *cfg, int **pairs, float **dist)
{
    int *nn12, *nn21 = NULL, *seeds, nseeds = 0, i, j, r, s, k, it, nm = 0;
    float *score, *dist1, *A, *sq, *w0, *hi;
    unsigned char *mnn = NULL, *eq;
    double R1, R2, rr1, se1, se2, *bal;
    long *cnt, *best_cnt;
    struct problems p;
    struct keyed *tmp;
    struct match *mt;
    int *idx, *ca, *cb, *inl;
    double *best_bal;
    float *best_hi;
    int nret;

    *pairs = NULL;
    *dist = NULL;
    if (n1 <= 1 || n2 <= 1)
        return 0;

    nn12 = G_malloc(n1 * sizeof(int));
    score = G_malloc(n1 * sizeof(float));
    if (cfg->force_seed_mnn)
        nn21 = G_malloc(n2 * sizeof(int));
    nearest(d1, n1, d2, n2, dim, nn12, score, nn21);
    if (nn21) {
        mnn = G_malloc(n1);
        for (i = 0; i < n1; i++)
            mnn[i] = nn21[nn12[i]] == i;
    }

    R1 = seed_radius(k1, n1, cfg->area_ratio);
    R2 = seed_radius(k2, n2, cfg->area_ratio);
    rr1 = R1 * R1;
    se1 = cfg->search_expansion * R1;
    se2 = cfg->search_expansion * R2;

    dist1 = G_malloc((size_t)n1 * n1 * sizeof(float));
#pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < n1; i++)
        for (j = 0; j < n1; j++)
            dist1[(size_t)i * n1 + j] = sqdist(k1 + 2 * i, k1 + 2 * j);

    /* Seeds: the best-scored match (lowest ratio) within R1, with a
     * ratio below 0.8^2, among mutual nearest neighbours if asked. */
    seeds = G_malloc(n1 * sizeof(int));
    for (i = 0; i < n1; i++) {
        int seed = score[i] < 0.64f && (!mnn || mnn[i]);

        for (j = 0; j < n1 && seed; j++)
            if (dist1[(size_t)i * n1 + j] < rr1 && score[i] > score[j] &&
                (!mnn || mnn[j]))
                seed = 0;
        if (seed)
            seeds[nseeds++] = i;
    }

    /* Neighbourhoods: matches within the expanded radius of the seed in
     * both images; seeds with fewer than min_inliers are dropped. */
    memset(&p, 0, sizeof(p));
    p.off = G_malloc((nseeds + 1) * sizeof(int));
    p.rd = G_malloc((nseeds + 1) * sizeof(int));
    for (s = 0; s < nseeds; s++) {
        int si = seeds[s], cntn = 0;
        const float *t2 = k2 + 2 * nn12[si];

        for (j = 0; j < n1; j++)
            if (dist1[(size_t)si * n1 + j] < se1 * se1 &&
                sqdist(t2, k2 + 2 * nn12[j]) < se2 * se2)
                cntn++;
        if (cntn < cfg->min_inliers)
            continue;
        seeds[p.nr] = si;
        p.off[p.nr] = p.N;
        p.rd[p.nr++] = cntn;
        p.N += cntn;
    }

    if (p.nr == 0) {
        /* No neighbourhood: every match with a ratio up to 0.95. */
        mt = G_malloc((n1 + 1) * sizeof(struct match));
        for (i = 0; i < n1; i++)
            if (score[i] <= 0.95f) {
                mt[nm].i1 = i;
                mt[nm].i2 = nn12[i];
                mt[nm].order = nm;
                mt[nm++].dist = score[i];
            }
        nret = unique_matches(mt, nm, pairs, dist);
        G_free(mt);
        G_free(p.off);
        G_free(p.rd);
        G_free(seeds);
        G_free(dist1);
        G_free(nn12);
        G_free(nn21);
        G_free(mnn);
        G_free(score);
        return nret;
    }

    /* Samples of each problem, ordered by ratio (best first), relative to
     * the seed and scaled by the search radius. */
    p.grp = G_malloc(p.N * sizeof(int));
    p.x = G_malloc(2 * p.N * sizeof(float));
    p.y = G_malloc(2 * p.N * sizeof(float));
    p.kp1 = G_malloc(p.N * sizeof(int));
    p.kp2 = G_malloc(p.N * sizeof(int));
    tmp = G_malloc(p.N * sizeof(struct keyed));
    for (r = 0; r < p.nr; r++) {
        int si = seeds[r], o = p.off[r], c = 0;
        const float *t1 = k1 + 2 * si, *t2 = k2 + 2 * nn12[si];

        for (j = 0; j < n1; j++)
            if (dist1[(size_t)si * n1 + j] < se1 * se1 &&
                sqdist(t2, k2 + 2 * nn12[j]) < se2 * se2) {
                tmp[o + c].v = score[j];
                tmp[o + c++].i = j;
            }
        qsort(tmp + o, c, sizeof(struct keyed), cmp_keyed);
        for (k = 0; k < c; k++) {
            int jj = tmp[o + k].i;

            s = o + k;
            p.grp[s] = r;
            p.kp1[s] = jj;
            p.kp2[s] = nn12[jj];
            p.x[2 * s] = (k1[2 * jj] - t1[0]) / se1;
            p.x[2 * s + 1] = (k1[2 * jj + 1] - t1[1]) / se1;
            p.y[2 * s] = (k2[2 * nn12[jj]] - t2[0]) / se2;
            p.y[2 * s + 1] = (k2[2 * nn12[jj] + 1] - t2[1]) / se2;
        }
    }
    G_free(dist1);

    /* RANSAC, first stage: affinities through fixed sample couples. */
    ca = G_malloc(cfg->ransac_iters * sizeof(int));
    cb = G_malloc(cfg->ransac_iters * sizeof(int));
    first_couples(cfg->ransac_iters, ca, cb);
    A = G_malloc(4 * p.nr * sizeof(float));
    sq = G_malloc(p.N * sizeof(float));
    idx = G_malloc(p.N * sizeof(int));
    w0 = G_malloc(p.N * sizeof(float));
    eq = G_malloc(p.N);
    cnt = G_malloc(p.nr * sizeof(long));
    bal = G_malloc(p.nr * sizeof(double));
    hi = G_malloc(p.nr * sizeof(float));
    best_cnt = G_malloc(p.nr * sizeof(long));
    best_bal = G_malloc(p.nr * sizeof(double));
    best_hi = G_malloc(p.nr * sizeof(float));
    inl = G_malloc(p.N * sizeof(int));

#define HYPOTHESES(it)                                                         \
    for (r = 0; r < p.nr; r++) {                                               \
        int a = p.off[r] + ca[it] % p.rd[r], b = p.off[r] + cb[it] % p.rd[r]; \
        double X[4] = {p.x[2 * a], p.x[2 * a + 1], p.x[2 * b],                 \
                       p.x[2 * b + 1]};                                        \
        double Y[4] = {p.y[2 * a], p.y[2 * a + 1], p.y[2 * b],                 \
                       p.y[2 * b + 1]};                                        \
        double Xi[4], M[4];                                                    \
                                                                               \
        inv2(X, Xi);                                                           \
        M[0] = Xi[0] * Y[0] + Xi[1] * Y[2];                                    \
        M[1] = Xi[0] * Y[1] + Xi[1] * Y[3];                                    \
        M[2] = Xi[2] * Y[0] + Xi[3] * Y[2];                                    \
        M[3] = Xi[2] * Y[1] + Xi[3] * Y[3];                                    \
        /* The affinity is the transpose of X^-1 Y. */                         \
        A[4 * r] = M[0];                                                       \
        A[4 * r + 1] = M[2];                                                   \
        A[4 * r + 2] = M[1];                                                   \
        A[4 * r + 3] = M[3];                                                   \
    }

    /* Columns of sorted squared residuals equal in every hypothesis count
     * as duplicates (torch.unique_consecutive over dim 1). */
    memset(eq, 1, p.N);
    eq[p.N - 1] = 0;
    for (it = 0; it < cfg->ransac_iters; it++) {
        HYPOTHESES(it);
        sorted_residuals(&p, A, sq, idx, tmp);
        for (k = 0; k < p.N - 1; k++)
            if (eq[k] && to_half(sq[k]) != to_half(sq[k + 1]))
                eq[k] = 0;
    }
    duplicate_weights(eq, p.N, w0);

    /* Best hypothesis of each problem: the highest inlier count. */
    for (r = 0; r < p.nr; r++)
        best_cnt[r] = -1;
    for (it = 0; it < cfg->ransac_iters; it++) {
        HYPOTHESES(it);
        sorted_residuals(&p, A, sq, idx, tmp);
        select_inliers(&p, sq, w0, cfg->min_confidence, cnt, bal, hi);
        for (r = 0; r < p.nr; r++)
            if (cnt[r] > best_cnt[r]) {
                best_cnt[r] = cnt[r];
                best_bal[r] = bal[r];
                best_hi[r] = hi[r];
                memcpy(inl + p.off[r], idx + p.off[r], cnt[r] * sizeof(int));
            }
    }
#undef HYPOTHESES

    {
        long total = 0;

        for (r = 0; r < p.nr; r++)
            total += best_cnt[r];
        if (total > 0) {
            /* Refit: least-squares affinity on each problem's inliers,
             * replaced by the identity when its scale change exceeds
             * det_thr, then a single selection round. */
            double dt2 = cfg->det_thr * cfg->det_thr;

            for (r = 0; r < p.nr; r++) {
                double XtX[4] = {0, 0, 0, 0}, YtX[4] = {0, 0, 0, 0}, Xi[4];
                double a00, a01, a10, a11, ea, eb, ed, trh, df, disc, e1, e2;

                for (k = 0; k < best_cnt[r]; k++) {
                    int t = inl[p.off[r] + k];
                    double x0 = p.x[2 * t], x1 = p.x[2 * t + 1];
                    double y0 = p.y[2 * t], y1 = p.y[2 * t + 1];

                    XtX[0] += x0 * x0;
                    XtX[1] += x0 * x1;
                    XtX[2] += x1 * x0;
                    XtX[3] += x1 * x1;
                    YtX[0] += y0 * x0;
                    YtX[1] += y0 * x1;
                    YtX[2] += y1 * x0;
                    YtX[3] += y1 * x1;
                }
                inv2(XtX, Xi);
                a00 = YtX[0] * Xi[0] + YtX[1] * Xi[2];
                a01 = YtX[0] * Xi[1] + YtX[1] * Xi[3];
                a10 = YtX[2] * Xi[0] + YtX[3] * Xi[2];
                a11 = YtX[2] * Xi[1] + YtX[3] * Xi[3];
                /* Eigenvalues of A A^T (batch_2x2_ellipse). */
                ea = a00 * a00 + a01 * a01;
                eb = a00 * a10 + a01 * a11;
                ed = a10 * a10 + a11 * a11;
                trh = 0.5 * (ea + ed);
                df = 0.5 * (ea - ed);
                disc = hypot(df, eb);
                e1 = fmax(trh + disc, 0.0);
                e2 = fmax(trh - disc, 0.0);
                if (e2 < 1.0 / dt2 || e1 > dt2) {
                    a00 = a11 = 1;
                    a01 = a10 = 0;
                }
                A[4 * r] = a00;
                A[4 * r + 1] = a01;
                A[4 * r + 2] = a10;
                A[4 * r + 3] = a11;
            }
            sorted_residuals(&p, A, sq, idx, tmp);
            for (k = 0; k < p.N - 1; k++)
                eq[k] = to_half(sq[k]) == to_half(sq[k + 1]);
            eq[p.N - 1] = 0;
            duplicate_weights(eq, p.N, w0);
            select_inliers(&p, sq, w0, cfg->min_confidence, best_cnt,
                           best_bal, best_hi);
            memcpy(inl, idx, p.N * sizeof(int));
        }
    }

    /* Accept the inliers of confident problems with enough of them. */
    mt = G_malloc((p.N + 1) * sizeof(struct match));
    for (r = 0; r < p.nr; r++) {
        float conf = confidence(best_cnt[r], best_bal[r], best_hi[r]);
        float ratio = 1.0f / conf;

        if (!(conf >= cfg->min_confidence &&
              (float)best_cnt[r] * (1.0f - ratio) >= cfg->min_inliers))
            continue;
        for (k = 0; k < best_cnt[r]; k++) {
            int t = inl[p.off[r] + k];

            mt[nm].i1 = p.kp1[t];
            mt[nm].i2 = p.kp2[t];
            mt[nm].order = nm;
            mt[nm++].dist = ratio;
        }
    }
    nret = unique_matches(mt, nm, pairs, dist);

    G_free(mt);
    G_free(inl);
    G_free(best_hi);
    G_free(best_bal);
    G_free(best_cnt);
    G_free(hi);
    G_free(bal);
    G_free(cnt);
    G_free(eq);
    G_free(w0);
    G_free(idx);
    G_free(sq);
    G_free(A);
    G_free(cb);
    G_free(ca);
    G_free(tmp);
    G_free(p.kp2);
    G_free(p.kp1);
    G_free(p.y);
    G_free(p.x);
    G_free(p.grp);
    G_free(p.rd);
    G_free(p.off);
    G_free(seeds);
    G_free(nn12);
    G_free(nn21);
    G_free(mnn);
    G_free(score);
    return nret;
}
