/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MODELS_H
#define I_COREGISTRATION_OPENCL_MODELS_H

/* 2D transformations from the image to co-register (x, y) to the
 * reference (X, Y), fitted to matches by (weighted) least squares. A
 * match is four doubles: x, y, X, Y. Coordinates are normalized
 * (centroid and mean distance) before fitting, so that the systems stay
 * well conditioned whatever the units. */
enum model_type {
    MODEL_SIMILARITY, /* Rotation, scale, shift: 4 parameters. */
    MODEL_AFFINE,     /* 6 parameters (polynomial of order 1). */
    MODEL_HOMOGRAPHY, /* 8 parameters (normalized DLT). */
    MODEL_POLY2,      /* Polynomial of order 2: 12 parameters. */
    MODEL_POLY3,      /* Polynomial of order 3: 20 parameters. */
    MODEL_NTYPES
};

struct model {
    enum model_type type;
    double sn[3], rn[3]; /* Normalization: centre x, y and scale. */
    double c[20];        /* Coefficients in normalized coordinates. */
};

/* Name of a model ("similarity", "affine", "homography", "poly2",
 * "poly3"), and the type of a name (-1 if unknown). */
const char *model_name(enum model_type t);
int model_parse(const char *name);

/* Matches a minimal sample needs. */
int model_min_points(enum model_type t);

/* Fit model m of type t to the n matches pts[idx[i]] (idx NULL: the
 * first n), with weights w[idx[i]] (w NULL: all 1). Returns 0 when the
 * matches do not determine the model (too few, collinear, coincident:
 * a singular or badly conditioned system). */
int model_fit(struct model *m, enum model_type t, const double *pts,
              const int *idx, int n, const double *w);

/* Map (x, y) of the image to the reference. */
void model_apply(const struct model *m, double x, double y, double *X,
                 double *Y);

/* Squared distance between the mapped image point of match p and its
 * reference point. */
double model_sqres(const struct model *m, const double *p);

/* Whether the model is plausible for images already roughly aligned:
 * its Jacobian at (x, y) preserves orientation and its singular values
 * lie within [1 / bound, bound]. */
int model_plausible(const struct model *m, double x, double y, double bound);

/* Leave-one-out residuals of a model fitted by least squares to the n
 * matches pts[idx[i]] with weights w (NULL: 1): the distance of each
 * match to the model fitted without it, from the diagonal of the hat
 * matrix. Linear models only (not the homography); loo[i] is written
 * for i < n. Returns 0 if the fit fails or the model is not linear. */
int model_loo(enum model_type t, const double *pts, const int *idx, int n,
              const double *w, double *loo);

#endif /* I_COREGISTRATION_OPENCL_MODELS_H */
