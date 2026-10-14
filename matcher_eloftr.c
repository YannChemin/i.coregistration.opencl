/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      The Efficient LoFTR matcher: grey tiles through the
 *               network on the OpenCL device, matches converted to cells
 *               and checked against the valid area of the tile.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "eloftr.h"
#include "matcher_eloftr.h"

/* The reference reports fine positions as the pixel index minus
 * FW / 2 - 0.5 (window offset convention of the original code): added
 * back to get cell indices (centres at integers). */
#define KP_OFFSET 3.5

struct eloftr_state {
    struct eloftr_model net;
    int npe;
    float *grey0, *grey1;
};

/* Grey value of the reference's image processor (0.2989 R + 0.5870 G +
 * 0.1140 B) of a (3, H, W) tile in 0..1. */
static void to_grey(const float *rgb, int H, int W, float *grey)
{
    size_t p = (size_t)H * W, i;

    for (i = 0; i < p; i++)
        grey[i] =
            0.2989f * rgb[i] + 0.5870f * rgb[p + i] + 0.1140f * rgb[2 * p + i];
}

static void eloftr_m_alloc(struct matcher *m, int H, int W)
{
    struct eloftr_state *s = m->state;

    if (s->npe)
        s->net.pos_scale = (float)ELOFTR_TRAIN_RES / (H > W ? H : W);
    eloftr_alloc(&s->net, H, W);
    s->grey0 = G_malloc((size_t)H * W * sizeof(float));
    s->grey1 = G_malloc((size_t)H * W * sizeof(float));
}

static void eloftr_m_run(struct matcher *m, const struct matcher_tile *t)
{
    struct eloftr_state *s = m->state;

    if (t->H != s->net.H || t->W != s->net.W)
        G_fatal_error(_("ELoFTR tile %d x %d differs from its buffers"), t->H,
                      t->W);
    to_grey(t->img1, t->H, t->W, s->grey0);
    to_grey(t->img2, t->H, t->W, s->grey1);
    eloftr_match(&s->net, s->grey0, s->grey1, m->dump_dir, t->index);
}

static int valid_at(const unsigned char *v, int h, int w, double x, double y)
{
    long ix = lround(x), iy = lround(y);

    return ix >= 0 && iy >= 0 && ix < w && iy < h &&
           (!v || v[(size_t)iy * w + ix]);
}

static int eloftr_m_match(struct matcher *m, const struct matcher_tile *t,
                          struct match_list *out)
{
    struct eloftr_state *s = m->state;
    int i, n = 0;

    for (i = 0; i < s->net.nm; i++) {
        const float *k = s->net.kp + 4 * (size_t)i;
        struct match mt;

        mt.x1 = k[0] + KP_OFFSET;
        mt.y1 = k[1] + KP_OFFSET;
        mt.x2 = k[2] + KP_OFFSET;
        mt.y2 = k[3] + KP_OFFSET;
        mt.cost = 1.0f - s->net.mconf[i];
        /* Padding and NULL cells match nothing. */
        if (!valid_at(t->valid1, t->h, t->w, mt.x1, mt.y1) ||
            !valid_at(t->valid2, t->h, t->w, mt.x2, mt.y2))
            continue;
        match_add(out, &mt);
        n++;
    }
    G_debug(1, "tile %d: %d coarse matches, %d in the valid area", t->index,
            s->net.nm, n);
    return n;
}

/* Grey inputs (H x W float32 each) and the matches as text: coarse
 * indices i, j, confidence, then x0 y0 x1 y1 in the reference's
 * convention, for tests/reference_compare_eloftr.py. */
static void eloftr_m_dump(struct matcher *m, const struct matcher_tile *t,
                          const char *dir)
{
    struct eloftr_state *s = m->state;
    size_t bytes = (size_t)t->H * t->W * sizeof(float);
    char path[GPATH_MAX];
    FILE *fp;
    int i;

    matcher_dump_file(dir, t->index, "eloftr_grey_0.f32", s->grey0, bytes);
    matcher_dump_file(dir, t->index, "eloftr_grey_1.f32", s->grey1, bytes);
    snprintf(path, sizeof(path), "%s/tile%03d_eloftr_matches.txt", dir,
             t->index);
    if (!(fp = fopen(path, "w")))
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fprintf(fp, "# x0 %d y0 %d w %d h %d H %d W %d pos_scale %.9g\n", t->x0,
            t->y0, t->w, t->h, t->H, t->W, s->net.pos_scale);
    for (i = 0; i < s->net.nm; i++) {
        const float *k = s->net.kp + 4 * (size_t)i;

        fprintf(fp, "%d %d %.9g %.6f %.6f %.6f %.6f\n", s->net.mi[i],
                s->net.mj[i], s->net.mconf[i], k[0], k[1], k[2], k[3]);
    }
    fclose(fp);
}

static void eloftr_m_destroy(struct matcher *m)
{
    struct eloftr_state *s = m->state;

    eloftr_free(&s->net);
    G_free(s->grey0);
    G_free(s->grey1);
    G_free(s);
    m->state = NULL;
}

void matcher_eloftr_create(struct matcher *m, struct ops *ops,
                           struct pth_file *pf, const char *name, int npe)
{
    struct eloftr_state *s = G_calloc(1, sizeof(*s));

    eloftr_load(&s->net, ops, pf);
    s->npe = npe;
    memset(m, 0, sizeof(*m));
    m->name = name;
    m->multiple = ELOFTR_MULTIPLE;
    m->ops = ops;
    m->state = s;
    m->alloc = eloftr_m_alloc;
    m->run = eloftr_m_run;
    m->match = eloftr_m_match;
    m->dump = eloftr_m_dump;
    m->destroy = eloftr_m_destroy;
}
