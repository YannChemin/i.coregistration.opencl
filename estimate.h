/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_ESTIMATE_H
#define I_COREGISTRATION_OPENCL_ESTIMATE_H

#include <grass/gis.h>

#include "gcp.h"
#include "robust.h"

/* Robust estimation over control points, in map coordinates (source
 * east/north to reference east/north). With only_inliers set, the
 * points already rejected take no part in the fit and stay outliers.
 * The others get their inlier flag and weight from the estimation; all
 * get their residual to the fitted model, in map units. When no model
 * can be fitted, the points taking part become outliers. Returns 1 when
 * a model was fitted (into *out when not NULL). */
int estimate_points(struct gcp *v, int n, const struct robust_config *cfg,
                    const struct Cell_head *win, int only_inliers,
                    struct model *out);

/* Root mean square of the inlier residuals, and of their leave-one-out
 * residuals for the linear models (*loo_rms NAN otherwise), map units. */
void estimate_rms(const struct gcp *v, int n, enum model_type t,
                  const struct Cell_head *win, double *rms, double *loo_rms);

#endif /* I_COREGISTRATION_OPENCL_ESTIMATE_H */
