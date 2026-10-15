/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_DENSE_H
#define I_COREGISTRATION_OPENCL_DENSE_H

#include <grass/gis.h>

#define DENSE_MAPS 4

/* The dense shift of a dense matcher over the current region, written
 * as rasters <base>_dx (east), <base>_dy (north), in map units,
 * <base>_certainty (0..1) and, for matchers that predict it, <base>_sd
 * (the predicted standard deviation of the shift, map units, blended as
 * the shifts). Tiles are blended in their overlaps,
 * with weights falling linearly to the tile edges over the overlap
 * width, times the certainty for the shifts. Tiles must be added in
 * rows of tiles from the top, each row at most tile rows high, and
 * dense_flush called before each row; rows are written as soon as no
 * later tile reaches them, so only tile rows of the region are held. */
struct dense_out {
    char names[DENSE_MAPS][GNAME_MAX];
    int fd[DENSE_MAPS], nmaps;
    int rows, cols, tile, overlap, next;
    double ew_res, ns_res;
    const unsigned char *valid; /* Source validity, rows x cols. */
    float *sw, *swc, *sx, *sy;  /* Ring of tile rows of the region. */
    float *ss;                  /* Same for the sd (NULL: none). */
    double min[DENSE_MAPS], max[DENSE_MAPS];
};

/* Fatal unless <base>_dx, <base>_dy, <base>_certainty (and <base>_sd
 * with with_sd) are legal names and may be written (absent, or
 * --overwrite). */
void dense_check(const char *base, int with_sd);

/* Open the rasters over region win. valid (win's rows x cols, 0 where
 * the source is invalid: NULL in all of them) must last until
 * dense_close. */
void dense_open(struct dense_out *d, const char *base,
                const struct Cell_head *win, int tile, int overlap,
                const unsigned char *valid, int with_sd);

/* Add one tile's field (h x w cells from (x0, y0), shifts and sd in
 * cells along the columns and rows, NAN where none; sd unused without
 * with_sd). */
void dense_add(struct dense_out *d, int x0, int y0, int w, int h,
               const float *dx, const float *dy, const float *cert,
               const float *sd);

/* Write the rows above row (no later tile reaches them). */
void dense_flush(struct dense_out *d, int row);

/* Write the remaining rows, close the rasters, write their colours,
 * units and history. */
void dense_close(struct dense_out *d);

#endif /* I_COREGISTRATION_OPENCL_DENSE_H */
