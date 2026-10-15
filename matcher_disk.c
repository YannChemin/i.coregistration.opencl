/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      The DISK + AdaLAM matcher: DISK keypoints and
 *               descriptors on the OpenCL device, nearest-neighbour
 *               matching and AdaLAM filtering on the host.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "disk.h"
#include "matcher_disk.h"

struct disk_state {
    struct disk_model disk;
    struct disk_matcher_config cfg;
    int *xy1, *xy2;
    float *k1, *k2, *s1, *s2, *d1, *d2;
    int n1, n2;
    /* Matches of the last tile, for the dump. */
    int nm, *pairs;
    float *dist;
};

static void disk_m_alloc(struct matcher *m, int H, int W)
{
    struct disk_state *s = m->state;
    int n = s->cfg.features;

    disk_alloc(&s->disk, H, W, n);
    s->xy1 = G_malloc(2 * n * sizeof(int));
    s->xy2 = G_malloc(2 * n * sizeof(int));
    s->k1 = G_malloc(2 * n * sizeof(float));
    s->k2 = G_malloc(2 * n * sizeof(float));
    s->s1 = G_malloc(n * sizeof(float));
    s->s2 = G_malloc(n * sizeof(float));
    s->d1 = G_malloc((size_t)n * DISK_DESC * sizeof(float));
    s->d2 = G_malloc((size_t)n * DISK_DESC * sizeof(float));
}

static void disk_m_run(struct matcher *m, const struct matcher_tile *t)
{
    struct disk_state *s = m->state;
    int n = s->cfg.features;
    float th = s->cfg.threshold;

    s->n1 = disk_detect(&s->disk, t->img1, t->H, t->W, t->h, t->w, n, th,
                        t->valid1, s->xy1, s->s1, s->d1);
    s->n2 = disk_detect(&s->disk, t->img2, t->H, t->W, t->h, t->w, n, th,
                        t->valid2, s->xy2, s->s2, s->d2);
}

static int disk_m_match(struct matcher *m, const struct matcher_tile *t,
                        struct match_list *out)
{
    struct disk_state *s = m->state;
    int i;

    for (i = 0; i < 2 * s->n1; i++)
        s->k1[i] = s->xy1[i];
    for (i = 0; i < 2 * s->n2; i++)
        s->k2[i] = s->xy2[i];
    G_free(s->pairs);
    G_free(s->dist);
    s->nm = adalam_match(s->k1, s->d1, s->n1, s->k2, s->d2, s->n2, DISK_DESC,
                         &s->cfg.adalam, &s->pairs, &s->dist);
    G_debug(1, "tile %d: %d + %d keypoints, %d matches", t->index, s->n1, s->n2,
            s->nm);
    for (i = 0; i < s->nm; i++) {
        struct match mt;
        int a = s->pairs[2 * i], b = s->pairs[2 * i + 1];

        mt.x1 = s->xy1[2 * a];
        mt.y1 = s->xy1[2 * a + 1];
        mt.x2 = s->xy2[2 * b];
        mt.y2 = s->xy2[2 * b + 1];
        mt.cost = s->dist[i];
        mt.sd = NAN;
        mt.cycle = NAN;
        match_add(out, &mt);
    }
    return s->nm;
}

/* Keypoints as float32 rows: x, y, score, then the descriptor. */
static void dump_keypoints(const char *dir, int t, const char *what,
                           const int *xy, const float *score, const float *desc,
                           int n)
{
    size_t row = 3 + DISK_DESC;
    float *v = G_malloc((n + 1) * row * sizeof(float));
    int i;

    for (i = 0; i < n; i++) {
        v[i * row] = xy[2 * i];
        v[i * row + 1] = xy[2 * i + 1];
        v[i * row + 2] = score[i];
        memcpy(v + i * row + 3, desc + (size_t)i * DISK_DESC,
               DISK_DESC * sizeof(float));
    }
    matcher_dump_file(dir, t, what, v, n * row * sizeof(float));
    G_free(v);
}

/* The DISK inputs (3 x H x W float32 per image), keypoints and the
 * AdaLAM matches (i1 i2 dist), for tests/reference_compare.py. */
static void disk_m_dump(struct matcher *m, const struct matcher_tile *t,
                        const char *dir)
{
    struct disk_state *s = m->state;
    size_t bytes = 3 * (size_t)t->H * t->W * sizeof(float);
    char path[GPATH_MAX];
    FILE *fp;
    int i;

    matcher_dump_file(dir, t->index, "src_input.f32", t->img1, bytes);
    matcher_dump_file(dir, t->index, "ref_input.f32", t->img2, bytes);
    dump_keypoints(dir, t->index, "src_kp.f32", s->xy1, s->s1, s->d1, s->n1);
    dump_keypoints(dir, t->index, "ref_kp.f32", s->xy2, s->s2, s->d2, s->n2);
    snprintf(path, sizeof(path), "%s/tile%03d_matches.txt", dir, t->index);
    if (!(fp = fopen(path, "w")))
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fprintf(fp, "# x0 %d y0 %d w %d h %d H %d W %d\n", t->x0, t->y0, t->w, t->h,
            t->H, t->W);
    for (i = 0; i < s->nm; i++)
        fprintf(fp, "%d %d %.9g\n", s->pairs[2 * i], s->pairs[2 * i + 1],
                s->dist[i]);
    fclose(fp);
}

static void disk_m_destroy(struct matcher *m)
{
    struct disk_state *s = m->state;

    disk_free(&s->disk);
    G_free(s->xy1);
    G_free(s->xy2);
    G_free(s->k1);
    G_free(s->k2);
    G_free(s->s1);
    G_free(s->s2);
    G_free(s->d1);
    G_free(s->d2);
    G_free(s->pairs);
    G_free(s->dist);
    G_free(s);
    m->state = NULL;
}

void matcher_disk_create(struct matcher *m, struct ops *ops,
                         struct pth_file *pf,
                         const struct disk_matcher_config *cfg)
{
    struct disk_state *s = G_calloc(1, sizeof(*s));

    s->cfg = *cfg;
    disk_load(&s->disk, ops, pf);
    memset(m, 0, sizeof(*m));
    m->name = "disk";
    m->multiple = DISK_MULTIPLE;
    m->ops = ops;
    m->state = s;
    m->alloc = disk_m_alloc;
    m->run = disk_m_run;
    m->match = disk_m_match;
    m->dump = disk_m_dump;
    m->destroy = disk_m_destroy;
}
