/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Imagery group input as 8-bit RGB, as is or with a
 *               percentile stretch.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/imagery.h>
#include <grass/raster.h>

#include "image.h"
#include "progress.h"

#define NBINS 65536

static void resolve_group(const char *group, const char *subgroup,
                          struct Ref *ref)
{
    char name[GNAME_MAX], mapset[GMAPSET_MAX];
    const char *found;

    if (!G_name_is_fully_qualified(group, name, mapset)) {
        G_strlcpy(name, group, sizeof(name));
        mapset[0] = '\0';
    }
    found = G_find_file2("group", name, mapset);
    if (!found)
        G_fatal_error(_("Imagery group <%s> not found"), group);

    I_init_group_ref(ref);
    if (subgroup) {
        if (!I_find_subgroup2(name, subgroup, found))
            G_fatal_error(_("Subgroup <%s> not found in group <%s>"), subgroup,
                          group);
        if (!I_get_subgroup_ref2(name, subgroup, found, ref))
            G_fatal_error(_("Unable to read subgroup <%s> of group <%s>"),
                          subgroup, group);
    }
    else if (!I_get_group_ref2(name, found, ref))
        G_fatal_error(_("Unable to read imagery group <%s>"), group);

    if (ref->nfiles != 3)
        G_fatal_error(_("%s <%s> contains %d raster maps; exactly 3 (used "
                        "as R, G, B in group order) are required. Use "
                        "i.group to build a 3-band group or subgroup."),
                      subgroup ? _("Subgroup") : _("Group"),
                      subgroup ? subgroup : group, ref->nfiles);
}

/* Value at percentile pct (0..100) of a histogram over [min, max]. */
static double hist_percentile(const long *hist, long total, double min,
                              double max, double pct)
{
    double target = pct / 100.0 * (total - 1), width = (max - min) / NBINS;
    long cum = 0;
    int b;

    for (b = 0; b < NBINS; b++) {
        if (cum + hist[b] > target) {
            double frac = hist[b] ? (target - cum) / hist[b] : 0.0;

            return min + (b + frac) * width;
        }
        cum += hist[b];
    }
    return max;
}

void image_read_group(const char *group, const char *subgroup, int stretch,
                      double lo, double hi, struct rgb_image *img)
{
    struct Ref ref;
    int fd[3], b, row, col, nrows, ncols;
    DCELL *buf[3];
    double min[3], max[3], vlo[3], vhi[3];
    long *hist, total = 0;
    size_t n;

    resolve_group(group, subgroup, &ref);

    nrows = Rast_window_rows();
    ncols = Rast_window_cols();
    n = (size_t)nrows * ncols;
    img->rows = nrows;
    img->cols = ncols;
    img->rgb = G_calloc(n * 3, 1);
    progress_begin(PG_READ, n / 1e6);
    img->valid = G_calloc(n, 1);

    for (b = 0; b < 3; b++) {
        snprintf(img->names[b], sizeof(img->names[b]), "%s@%s",
                 ref.file[b].name, ref.file[b].mapset);
        fd[b] = Rast_open_old(ref.file[b].name, ref.file[b].mapset);
        buf[b] = Rast_allocate_d_buf();
        min[b] = INFINITY;
        max[b] = -INFINITY;
    }
    G_message(_("Input bands (R, G, B): <%s>, <%s>, <%s>"), img->names[0],
              img->names[1], img->names[2]);

    /* Pass 1: validity mask and per-band range over the region. */
    G_message(_("Reading imagery group <%s>..."), group);
    for (row = 0; row < nrows; row++) {
        progress_part(row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;

            if (Rast_is_d_null_value(&buf[0][col]) ||
                Rast_is_d_null_value(&buf[1][col]) ||
                Rast_is_d_null_value(&buf[2][col]))
                continue;
            img->valid[i] = 1;
            total++;
            for (b = 0; b < 3; b++) {
                if (buf[b][col] < min[b])
                    min[b] = buf[b][col];
                if (buf[b][col] > max[b])
                    max[b] = buf[b][col];
            }
        }
    }
    if (total == 0)
        G_fatal_error(_("No cell has valid values in all three bands in the "
                        "current region"));

    /* Bands already within 0..255 (8-bit basemaps) are taken as they are,
     * as the repository reads its JPEG images, unless a stretch is asked. */
    if (!stretch) {
        for (b = 0; b < 3; b++)
            if (min[b] < 0 || max[b] > 255)
                break;
        if (b == 3) {
            for (b = 0; b < 3; b++) {
                vlo[b] = 0;
                vhi[b] = 255;
            }
            G_verbose_message(_("Bands within 0..255, read without stretch"));
            goto convert;
        }
        lo = 2;
        hi = 98;
        G_message(_("Bands outside 0..255: stretched between their 2 and 98 "
                    "percentiles"));
    }

    /* Pass 2: per-band histograms for the percentile stretch. */
    hist = G_calloc(3L * NBINS, sizeof(long));
    for (row = 0; row < nrows; row++) {
        progress_part(nrows + row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            if (!img->valid[(size_t)row * ncols + col])
                continue;
            for (b = 0; b < 3; b++) {
                double r = max[b] > min[b]
                               ? (buf[b][col] - min[b]) / (max[b] - min[b])
                               : 0.0;
                int k = (int)(r * NBINS);

                hist[(long)b * NBINS + (k >= NBINS ? NBINS - 1 : k)]++;
            }
        }
    }
    for (b = 0; b < 3; b++) {
        vlo[b] =
            hist_percentile(hist + (long)b * NBINS, total, min[b], max[b], lo);
        vhi[b] =
            hist_percentile(hist + (long)b * NBINS, total, min[b], max[b], hi);
        if (!(vhi[b] > vlo[b])) {
            vlo[b] = min[b];
            vhi[b] = max[b];
        }
        if (!(vhi[b] > vlo[b]))
            G_warning(_("Band <%s> is constant in the current region"),
                      img->names[b]);
        G_verbose_message(_("Band <%s>: stretch %g .. %g (range %g .. %g)"),
                          img->names[b], vlo[b], vhi[b], min[b], max[b]);
    }
    G_free(hist);

    /* Pass 3: 8-bit conversion. */
convert:
    for (row = 0; row < nrows; row++) {
        progress_part(2.0 * nrows + row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;

            if (!img->valid[i])
                continue;
            for (b = 0; b < 3; b++) {
                double v = vhi[b] > vlo[b] ? (buf[b][col] - vlo[b]) /
                                                 (vhi[b] - vlo[b]) * 255.0
                                           : 0.0;

                img->rgb[3 * i + b] = v <= 0.0     ? 0
                                      : v >= 255.0 ? 255
                                                   : (unsigned char)(v + 0.5);
            }
        }
    }

    for (b = 0; b < 3; b++) {
        Rast_close(fd[b]);
        G_free(buf[b]);
    }
    I_free_group_ref(&ref);
    progress_end();
}

void image_free(struct rgb_image *img)
{
    G_free(img->rgb);
    G_free(img->valid);
    img->rgb = img->valid = NULL;
}
