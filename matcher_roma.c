/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      The RoMa matcher: tiles through the network on the OpenCL
 *               device, the dense warp sampled into matches by blocks
 *               of cells, checked against the valid area of the tile.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "matcher_roma.h"
#include "roma.h"

struct roma_state {
    struct roma_model net;
    int spacing;
    float min_cert;
};

static void roma_m_alloc(struct matcher *m, int H, int W)
{
    struct roma_state *s = m->state;

    roma_alloc(&s->net, H, W);
}

static void roma_m_run(struct matcher *m, const struct matcher_tile *t)
{
    struct roma_state *s = m->state;

    if (t->H != s->net.H || t->W != s->net.W)
        G_fatal_error(_("RoMa tile %d x %d differs from its buffers"), t->H,
                      t->W);
    roma_match(&s->net, t->img1, t->img2, m->dump_dir, t->index);
}

static int valid_at(const unsigned char *v, int h, int w, double x, double y)
{
    long ix = lround(x), iy = lround(y);

    return ix >= 0 && iy >= 0 && ix < w && iy < h &&
           (!v || v[(size_t)iy * w + ix]);
}

/* In each block of spacing x spacing valid cells of the image, the cell
 * of highest certainty; a match where that certainty reaches min_cert
 * and the warp lands on a valid cell of the reference. */
static int roma_m_match(struct matcher *m, const struct matcher_tile *t,
                        struct match_list *out)
{
    struct roma_state *s = m->state;
    const float *cert = s->net.cert, *flow = s->net.flow;
    size_t hw = (size_t)t->H * t->W;
    int S = s->spacing, bx, by, n = 0;

    for (by = 0; by < t->h; by += S)
        for (bx = 0; bx < t->w; bx += S) {
            int x, y, bxe = bx + S < t->w ? bx + S : t->w;
            int bye = by + S < t->h ? by + S : t->h;
            long best = -1;
            struct match mt;

            for (y = by; y < bye; y++)
                for (x = bx; x < bxe; x++) {
                    size_t i = (size_t)y * t->W + x;

                    if (t->valid1 && !t->valid1[(size_t)y * t->w + x])
                        continue;
                    if (best < 0 || cert[i] > cert[best])
                        best = (long)i;
                }
            if (best < 0 || !(cert[best] >= s->min_cert))
                continue;
            mt.x1 = best % t->W;
            mt.y1 = best / t->W;
            mt.x2 = (flow[best] + 1.0) * 0.5 * t->W - 0.5;
            mt.y2 = (flow[hw + best] + 1.0) * 0.5 * t->H - 0.5;
            mt.cost = 1.0f - cert[best];
            if (!valid_at(t->valid2, t->h, t->w, mt.x2, mt.y2))
                continue;
            match_add(out, &mt);
            n++;
        }
    G_debug(1, "tile %d: %d RoMa matches", t->index, n);
    return n;
}

/* The dense warp: a shift wherever the image is valid, certainty 0 (and
 * no shift) where the image is invalid or the warp lands on an invalid
 * cell of the reference. */
static void roma_m_field(struct matcher *m, const struct matcher_tile *t,
                         float *dx, float *dy, float *cert)
{
    struct roma_state *s = m->state;
    const float *c = s->net.cert, *flow = s->net.flow;
    size_t hw = (size_t)t->H * t->W;
    int x, y;

    for (y = 0; y < t->h; y++)
        for (x = 0; x < t->w; x++) {
            size_t i = (size_t)y * t->W + x, o = (size_t)y * t->w + x;
            double xr = (flow[i] + 1.0) * 0.5 * t->W - 0.5;
            double yr = (flow[hw + i] + 1.0) * 0.5 * t->H - 0.5;

            if ((t->valid1 && !t->valid1[o]) ||
                !valid_at(t->valid2, t->h, t->w, xr, yr)) {
                dx[o] = dy[o] = NAN;
                cert[o] = 0.0f;
                continue;
            }
            dx[o] = (float)(xr - x);
            dy[o] = (float)(yr - y);
            cert[o] = c[i];
        }
}

/* The tile's geometry, for tests/reference_compare_roma.py (the network
 * writes its own dumps during run). */
static void roma_m_dump(struct matcher *m, const struct matcher_tile *t,
                        const char *dir)
{
    char path[GPATH_MAX];
    FILE *fp;

    (void)m;
    snprintf(path, sizeof(path), "%s/tile%03d_roma.txt", dir, t->index);
    if (!(fp = fopen(path, "w")))
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fprintf(fp, "x0 %d y0 %d w %d h %d H %d W %d coarse %d\n", t->x0, t->y0,
            t->w, t->h, t->H, t->W, ROMA_COARSE);
    fclose(fp);
}

static void roma_m_destroy(struct matcher *m)
{
    struct roma_state *s = m->state;

    roma_free(&s->net);
    G_free(s);
    m->state = NULL;
}

void matcher_roma_create(struct matcher *m, struct ops *ops,
                         struct pth_file *pf, struct pth_file *dino,
                         struct pth_file *v2, const char *name, int imagenet,
                         int spacing, float min_cert)
{
    struct roma_state *s = G_calloc(1, sizeof(*s));

    roma_load(&s->net, ops, pf, dino);
    if (v2)
        roma_load_v2(&s->net, v2);
    s->net.imagenet = imagenet;
    s->spacing = spacing;
    s->min_cert = min_cert;
    memset(m, 0, sizeof(*m));
    m->name = name;
    m->multiple = ROMA_MULTIPLE;
    m->ops = ops;
    m->state = s;
    m->alloc = roma_m_alloc;
    m->run = roma_m_run;
    m->match = roma_m_match;
    m->dump = roma_m_dump;
    m->field = roma_m_field;
    m->destroy = roma_m_destroy;
}
