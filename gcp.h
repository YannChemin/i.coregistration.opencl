/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_GCP_H
#define I_COREGISTRATION_OPENCL_GCP_H

#include <grass/gis.h>

#include "image.h"

/* One control point: a cell of the source image and where it lies in
 * the reference, in cells of the current region (column, row; cell
 * centres at integers). */
struct gcp {
    double xs, ys;   /* Source. */
    double xr, yr;   /* Reference. */
    float dist;      /* AdaLAM inverse confidence (lower is better). */
    int tile;        /* Tile of the match, 1-based. */
    int refined;     /* Reference position refined to a fraction of a cell. */
    int n;           /* Matches merged into this point. */
    int inlier;      /* Kept by the robust estimation (status in POINTS). */
    double weight;   /* Robust weight, 0..1 (1 without estimation). */
    double residual; /* Distance to the fitted model, map units (NAN: no
                        model). */
};

struct gcp_list {
    int n, cap;
    struct gcp *v;
};

void gcp_add(struct gcp_list *l, const struct gcp *g);

/* Refine the reference position of g to a fraction of a cell: the peak
 * of the normalized cross-correlation of the grey (mean RGB) source
 * window of radius rad around (xs, ys) with reference windows shifted
 * by up to search cells, fitted by a parabola along each axis. Returns
 * 0 and leaves g unchanged when a window leaves the image or meets an
 * invalid cell, the source window is flat, or the peak lies on the edge
 * of the search area. */
int gcp_refine(struct gcp *g, const struct rgb_image *src,
               const struct rgb_image *ref, int rad, int search);

/* Merge the points that share a reference cell (the repository averages
 * the source positions of identical reference coordinates): positions
 * averaged, best inverse confidence kept. Inliers and outliers of the
 * robust estimation are merged apart; weights and residuals are
 * averaged. */
void gcp_merge(struct gcp_list *l);

/* Write the points as the control points of imagery group (its POINTS
 * file: source east/north as image coordinates, reference east/north as
 * target coordinates, active unless rejected by the robust estimation)
 * and make the current project and mapset its target, for i.rectify. */
void gcp_write_group(const struct gcp_list *l, const char *group,
                     const struct Cell_head *win);

/* Write the points as a vector point map at their reference positions
 * with an attribute table (source and reference coordinates, shift,
 * inverse confidence, tile, refinement, merged matches, robust weight,
 * residual and inlier flag). */
void gcp_write_vector(const struct gcp_list *l, const char *name,
                      const struct Cell_head *win);

#endif /* I_COREGISTRATION_OPENCL_GCP_H */
