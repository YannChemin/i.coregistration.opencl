/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Robust estimation over control points: per tile and
 *               over the region, in map coordinates.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>

#include <grass/gis.h>

#include "estimate.h"

static void map_coords(const struct gcp *g, const struct Cell_head *w,
                       double *p)
{
    p[0] = w->west + (g->xs + 0.5) * w->ew_res;
    p[1] = w->north - (g->ys + 0.5) * w->ns_res;
    p[2] = w->west + (g->xr + 0.5) * w->ew_res;
    p[3] = w->north - (g->yr + 0.5) * w->ns_res;
}

int estimate_points(struct gcp *v, int n, const struct robust_config *cfg,
                    const struct Cell_head *win, int only_inliers,
                    struct model *out)
{
    double *pts = G_malloc(4 * (size_t)(n + 1) * sizeof(double));
    double *res = G_malloc((n + 1) * sizeof(double));
    double *w = G_malloc((n + 1) * sizeof(double));
    float *cost = G_malloc((n + 1) * sizeof(float));
    unsigned char *in = G_malloc(n + 1);
    int *idx = G_malloc((n + 1) * sizeof(int));
    struct robust_result r;
    int i, k = 0;

    for (i = 0; i < n; i++)
        if (!only_inliers || v[i].inlier) {
            map_coords(&v[i], win, pts + 4 * (size_t)k);
            cost[k] = v[i].dist;
            idx[k++] = i;
        }
    robust_fit(pts, k, cost, cfg, &r, res, w, in);
    for (i = 0; i < k; i++) {
        struct gcp *g = &v[idx[i]];

        g->inlier = r.ok && in[i];
        g->weight = r.ok ? w[i] : 0;
    }
    for (i = 0; i < n; i++) {
        double p[4];

        if (!r.ok) {
            v[i].residual = NAN;
            continue;
        }
        map_coords(&v[i], win, p);
        v[i].residual = sqrt(model_sqres(&r.model, p));
        if (only_inliers && !v[i].inlier)
            v[i].weight = 0;
    }
    if (r.ok && out)
        *out = r.model;
    G_free(pts);
    G_free(res);
    G_free(w);
    G_free(cost);
    G_free(in);
    G_free(idx);
    return r.ok;
}

void estimate_rms(const struct gcp *v, int n, enum model_type t,
                  const struct Cell_head *win, double *rms, double *loo_rms)
{
    double *pts = G_malloc(4 * (size_t)(n + 1) * sizeof(double));
    double *loo = G_malloc((n + 1) * sizeof(double));
    double s = 0;
    int i, k = 0;

    for (i = 0; i < n; i++)
        if (v[i].inlier && isfinite(v[i].residual)) {
            map_coords(&v[i], win, pts + 4 * (size_t)k++);
            s += v[i].residual * v[i].residual;
        }
    *rms = k ? sqrt(s / k) : NAN;
    *loo_rms = NAN;
    if (k && model_loo(t, pts, NULL, k, NULL, loo)) {
        s = 0;
        for (i = 0; i < k; i++)
            s += loo[i] * loo[i];
        *loo_rms = sqrt(s / k);
    }
    G_free(pts);
    G_free(loo);
}
