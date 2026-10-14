/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_ROBUST_H
#define I_COREGISTRATION_OPENCL_ROBUST_H

#include "models.h"

/* Robust estimation of a model from matches with outliers.
 *
 * - ransac: uniform samples, inlier count, least squares on the inliers.
 * - magsac: uniform samples, MAGSAC++ quality (noise scale marginalized
 *   up to sigma_max = threshold / k, Barath et al. 2020), final
 *   iteratively reweighted least squares with the MAGSAC++ weights.
 * - gcransac: uniform samples, truncated quadratic quality, graph-cut
 *   local optimization (Barath and Matas, 2018): inliers labelled with a
 *   spatial coherence term between neighbouring matches.
 * - superansac: the combination recommended by Barath (2025): PROSAC
 *   samples in the order of the match costs, MAGSAC++ quality, graph-cut
 *   local optimization, final MAGSAC++ IRLS.
 *
 * Every kind rejects degenerate samples (no unique fit) and implausible
 * models (orientation reversed, scale beyond scale_bound), and with
 * spacing > 0 counts matches lying in the same spacing x spacing cell of
 * the image once (the independent inliers of VSAC, Ivashechkin et al.
 * 2021), so that a dense cluster (one roof, one aircraft) cannot outvote
 * the ground. */
enum robust_kind {
    ROBUST_NONE,
    ROBUST_RANSAC,
    ROBUST_MAGSAC,
    ROBUST_GCRANSAC,
    ROBUST_SUPERANSAC,
    ROBUST_NKINDS
};

struct robust_config {
    enum robust_kind kind;
    enum model_type model;
    double threshold;  /* Largest residual of an inlier, data units. */
    double confidence; /* Of finding an all-inlier sample: 0.99. */
    int min_iters, max_iters;
    double spacing;     /* Cell of the independent inliers (0: off). */
    double scale_bound; /* Model plausibility (2: scale 0.5..2). */
    double gc_lambda;   /* Weight of the spatial coherence term. */
    double gc_radius;   /* Neighbourhood of the graph cut, data units. */
    int gc_neighbours;  /* Neighbours per match in the graph. */
    unsigned long long seed;
};

/* Defaults for a kind and model, threshold in data units. */
void robust_defaults(struct robust_config *c, enum robust_kind kind,
                     enum model_type model, double threshold);

const char *robust_name(enum robust_kind k);
int robust_parse(const char *name);

struct robust_result {
    struct model model;
    int ok;       /* 0: too few matches or no plausible model found. */
    int iters;    /* Hypotheses drawn. */
    int ninliers; /* Matches labelled inliers. */
};

/* Estimate the model from the n matches pts (x, y, X, Y each), cost
 * (lower is better; NULL: none, PROSAC then samples uniformly). Writes,
 * for every match, its residual to the model (data units), its weight
 * (0..1: the MAGSAC++ weight relative to a zero residual for the MAGSAC
 * kinds, 1 or 0 for the others) and whether it is an inlier. When the
 * estimation fails, every match is an outlier with weight 0 and an
 * infinite residual. Returns res->ok. */
int robust_fit(const double *pts, int n, const float *cost,
               const struct robust_config *cfg, struct robust_result *res,
               double *residual, double *weight, unsigned char *inlier);

#endif /* I_COREGISTRATION_OPENCL_ROBUST_H */
