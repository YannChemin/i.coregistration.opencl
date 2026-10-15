/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      The dense shift and certainty rasters of a dense matcher,
 *               tiles blended in their overlaps.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "dense.h"

static const char *suffix[DENSE_MAPS] = {"dx", "dy", "certainty", "sd"};

static void make_names(const char *base, int nmaps,
                       char names[DENSE_MAPS][GNAME_MAX])
{
    int k;

    for (k = 0; k < nmaps; k++) {
        if (snprintf(names[k], GNAME_MAX, "%s_%s", base, suffix[k]) >=
            GNAME_MAX)
            G_fatal_error(_("dense=<%s> is too long"), base);
        if (G_legal_filename(names[k]) != 1)
            G_fatal_error(_("<%s> is an illegal file name"), names[k]);
    }
}

void dense_check(const char *base, int with_sd)
{
    char names[DENSE_MAPS][GNAME_MAX];
    int k, nmaps = with_sd ? 4 : 3;

    make_names(base, nmaps, names);
    for (k = 0; k < nmaps; k++)
        if (G_find_raster2(names[k], G_mapset()) && !G_get_overwrite())
            G_fatal_error(_("Raster map <%s> already exists (use "
                            "--overwrite)"),
                          names[k]);
}

void dense_open(struct dense_out *d, const char *base,
                const struct Cell_head *win, int tile, int overlap,
                const unsigned char *valid, int with_sd)
{
    size_t n = (size_t)tile * win->cols;
    int k;

    d->nmaps = with_sd ? 4 : 3;
    make_names(base, d->nmaps, d->names);
    for (k = 0; k < d->nmaps; k++) {
        d->fd[k] = Rast_open_new(d->names[k], FCELL_TYPE);
        d->min[k] = INFINITY;
        d->max[k] = -INFINITY;
    }
    d->rows = win->rows;
    d->cols = win->cols;
    d->tile = tile;
    d->overlap = overlap;
    d->next = 0;
    d->ew_res = win->ew_res;
    d->ns_res = win->ns_res;
    d->valid = valid;
    d->sw = G_calloc(n, sizeof(float));
    d->swc = G_calloc(n, sizeof(float));
    d->sx = G_calloc(n, sizeof(float));
    d->sy = G_calloc(n, sizeof(float));
    d->ss = with_sd ? G_calloc(n, sizeof(float)) : NULL;
}

/* Blending weight of cell i of n along one side of a tile: 1 inside,
 * falling linearly to the edges over the overlap width. */
static float ramp(int i, int n, int width)
{
    float r = 1.0f, a = (i + 0.5f) / width, b = (n - i - 0.5f) / width;

    if (width <= 0)
        return 1.0f;
    if (a < r)
        r = a;
    if (b < r)
        r = b;
    return r;
}

void dense_add(struct dense_out *d, int x0, int y0, int w, int h,
               const float *dx, const float *dy, const float *cert,
               const float *sd)
{
    int x, y;

    if (y0 < d->next || y0 + h > d->next + d->tile)
        G_fatal_error(_("Dense output: tile rows %d..%d outside the held "
                        "rows %d..%d"),
                      y0, y0 + h - 1, d->next, d->next + d->tile - 1);
    for (y = 0; y < h; y++) {
        size_t r = (size_t)((y0 + y) % d->tile) * d->cols;
        float wy = ramp(y, h, d->overlap);

        for (x = 0; x < w; x++) {
            size_t o = (size_t)y * w + x, i = r + x0 + x;
            float wt = wy * ramp(x, w, d->overlap);

            d->sw[i] += wt;
            if (!(cert[o] > 0.0f) || isnan(dx[o]) || isnan(dy[o]))
                continue;
            d->swc[i] += wt * cert[o];
            d->sx[i] += wt * cert[o] * dx[o];
            d->sy[i] += wt * cert[o] * dy[o];
            if (d->ss)
                d->ss[i] += wt * cert[o] * sd[o];
        }
    }
}

static void track(struct dense_out *d, int k, float v)
{
    if (v < d->min[k])
        d->min[k] = v;
    if (v > d->max[k])
        d->max[k] = v;
}

void dense_flush(struct dense_out *d, int row)
{
    FCELL *out[DENSE_MAPS];
    double cell = sqrt(d->ew_res * d->ns_res);
    int k, x;

    if (row > d->rows)
        row = d->rows;
    for (k = 0; k < d->nmaps; k++)
        out[k] = Rast_allocate_f_buf();
    for (; d->next < row; d->next++) {
        size_t r = (size_t)(d->next % d->tile) * d->cols;
        const unsigned char *v = d->valid + (size_t)d->next * d->cols;

        for (k = 0; k < d->nmaps; k++)
            Rast_set_f_null_value(out[k], d->cols);
        for (x = 0; x < d->cols; x++) {
            size_t i = r + x;

            if (!v[x] || !(d->sw[i] > 0.0f))
                continue;
            out[2][x] = d->swc[i] / d->sw[i];
            track(d, 2, out[2][x]);
            if (d->swc[i] > 0.0f) {
                out[0][x] = d->sx[i] / d->swc[i] * d->ew_res;
                out[1][x] = -d->sy[i] / d->swc[i] * d->ns_res;
                track(d, 0, out[0][x]);
                track(d, 1, out[1][x]);
                /* A NAN sd (no positive definite precision) stays NULL. */
                if (d->ss && isfinite(d->ss[i])) {
                    out[3][x] = d->ss[i] / d->swc[i] * cell;
                    track(d, 3, out[3][x]);
                }
            }
        }
        for (k = 0; k < d->nmaps; k++)
            Rast_put_f_row(d->fd[k], out[k]);
        /* Cleared for the tiles below. */
        memset(d->sw + r, 0, d->cols * sizeof(float));
        memset(d->swc + r, 0, d->cols * sizeof(float));
        memset(d->sx + r, 0, d->cols * sizeof(float));
        memset(d->sy + r, 0, d->cols * sizeof(float));
        if (d->ss)
            memset(d->ss + r, 0, d->cols * sizeof(float));
    }
    for (k = 0; k < d->nmaps; k++)
        G_free(out[k]);
}

void dense_close(struct dense_out *d)
{
    struct History hist;
    struct Colors colors;
    int k;

    dense_flush(d, d->rows);
    for (k = 0; k < d->nmaps; k++) {
        Rast_close(d->fd[k]);
        Rast_put_cell_title(
            d->names[k],
            k == 0   ? _("Shift east from the image to the reference")
            : k == 1 ? _("Shift north from the image to the reference")
            : k == 2 ? _("Certainty of the shift")
                     : _("Predicted standard deviation of the shift"));
        if (k == 3) {
            Rast_make_fp_colors(&colors, "viridis", 0,
                                d->max[3] > 0 ? d->max[3] : 1);
            Rast_write_units(d->names[k], G_database_unit_name(TRUE));
        }
        else if (k < 2) {
            double a = fmax(fabs(d->min[k]), fabs(d->max[k]));

            if (!(a > 0))
                a = 1;
            Rast_make_fp_colors(&colors, "differences", -a, a);
            Rast_write_units(d->names[k], G_database_unit_name(TRUE));
        }
        else
            Rast_make_fp_colors(&colors, "viridis", 0, 1);
        Rast_write_colors(d->names[k], G_mapset(), &colors);
        Rast_free_colors(&colors);
        Rast_short_history(d->names[k], "raster", &hist);
        Rast_command_history(&hist);
        Rast_write_history(d->names[k], &hist);
    }
    G_free(d->sw);
    G_free(d->swc);
    G_free(d->sx);
    G_free(d->sy);
    G_free(d->ss);
}
