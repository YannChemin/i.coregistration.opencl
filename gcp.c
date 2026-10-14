/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Control points: refinement to a fraction of a cell,
 *               merging, imagery group POINTS file and vector output.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <grass/dbmi.h>
#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/imagery.h>
#include <grass/vector.h>

#include "gcp.h"
#include "progress.h"

void gcp_add(struct gcp_list *l, const struct gcp *g)
{
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 1024;
        l->v = G_realloc(l->v, l->cap * sizeof(struct gcp));
    }
    l->v[l->n++] = *g;
}

static double grey(const struct rgb_image *img, int x, int y)
{
    const unsigned char *p = img->rgb + 3 * ((size_t)y * img->cols + x);

    return (p[0] + p[1] + p[2]) / 3.0;
}

/* Normalized cross-correlation of the source window around (sx, sy)
 * with the reference window around (rx, ry); -2 when a window leaves
 * the image or meets an invalid cell, or a window is flat. */
static double ncc(const struct rgb_image *src, int sx, int sy,
                  const struct rgb_image *ref, int rx, int ry, int rad)
{
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n, va, vb;
    int dx, dy;

    if (sx - rad < 0 || sy - rad < 0 || sx + rad >= src->cols ||
        sy + rad >= src->rows || rx - rad < 0 || ry - rad < 0 ||
        rx + rad >= ref->cols || ry + rad >= ref->rows)
        return -2;
    for (dy = -rad; dy <= rad; dy++)
        for (dx = -rad; dx <= rad; dx++) {
            double a, b;

            if (!src->valid[(size_t)(sy + dy) * src->cols + sx + dx] ||
                !ref->valid[(size_t)(ry + dy) * ref->cols + rx + dx])
                return -2;
            a = grey(src, sx + dx, sy + dy);
            b = grey(ref, rx + dx, ry + dy);
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
    n = (2 * rad + 1) * (2 * rad + 1);
    va = saa - sa * sa / n;
    vb = sbb - sb * sb / n;
    if (va <= 1e-9 || vb <= 1e-9)
        return -2;
    return (sab - sa * sb / n) / sqrt(va * vb);
}

/* Vertex of the parabola through (-1, a), (0, b), (1, c). */
static double vertex(double a, double b, double c)
{
    double den = a - 2 * b + c;

    if (den >= 0)
        return 0;
    return 0.5 * (a - c) / den;
}

int gcp_refine(struct gcp *g, const struct rgb_image *src,
               const struct rgb_image *ref, int rad, int search)
{
    int sx = (int)lround(g->xs), sy = (int)lround(g->ys);
    int rx = (int)lround(g->xr), ry = (int)lround(g->yr);
    int n = 2 * search + 1, dx, dy, bx = 0, by = 0;
    double *c = G_malloc(n * n * sizeof(double)), best = -2;

    for (dy = -search; dy <= search; dy++)
        for (dx = -search; dx <= search; dx++) {
            double v = ncc(src, sx, sy, ref, rx + dx, ry + dy, rad);

            c[(dy + search) * n + dx + search] = v;
            if (v > best) {
                best = v;
                bx = dx;
                by = dy;
            }
        }
    if (best <= -2 || abs(bx) == search || abs(by) == search) {
        G_free(c);
        return 0;
    }
#define C(x, y) c[((y) + search) * n + (x) + search]
    g->xr = rx + bx + vertex(C(bx - 1, by), C(bx, by), C(bx + 1, by));
    g->yr = ry + by + vertex(C(bx, by - 1), C(bx, by), C(bx, by + 1));
#undef C
    /* The source cell may be off its rounded position by a fraction. */
    g->xr += g->xs - sx;
    g->yr += g->ys - sy;
    g->refined = 1;
    G_free(c);
    return 1;
}

struct keyed {
    long key;
    int i;
};

static int cmp_keyed(const void *a, const void *b)
{
    const struct keyed *p = a, *q = b;

    if (p->key != q->key)
        return (p->key > q->key) - (p->key < q->key);
    return p->i - q->i;
}

void gcp_merge(struct gcp_list *l)
{
    struct keyed *k;
    int i, j, out = 0;

    if (l->n == 0)
        return;
    /* Reference cell as key: before refinement the positions are whole
     * cells; refined ones round to their cell. */
    k = G_malloc(l->n * sizeof(struct keyed));
    for (i = 0; i < l->n; i++) {
        k[i].key = lround(l->v[i].yr) * 4194304L + lround(l->v[i].xr);
        k[i].i = i;
    }
    qsort(k, l->n, sizeof(struct keyed), cmp_keyed);
    {
        struct gcp *v = G_malloc(l->n * sizeof(struct gcp));

        for (i = 0; i < l->n; i = j) {
            struct gcp m = l->v[k[i].i];
            int c = 1;

            m.n = l->v[k[i].i].n;
            for (j = i + 1; j < l->n && k[j].key == k[i].key; j++) {
                const struct gcp *g = &l->v[k[j].i];

                m.xs += g->xs;
                m.ys += g->ys;
                m.xr += g->xr;
                m.yr += g->yr;
                m.dist = fminf(m.dist, g->dist);
                m.refined = m.refined && g->refined;
                m.n += g->n;
                c++;
            }
            m.xs /= c;
            m.ys /= c;
            m.xr /= c;
            m.yr /= c;
            v[out++] = m;
        }
        G_free(l->v);
        l->v = v;
        l->n = l->cap = out;
    }
    G_free(k);
}

static double east(const struct Cell_head *w, double col)
{
    return w->west + (col + 0.5) * w->ew_res;
}

static double north(const struct Cell_head *w, double row)
{
    return w->north - (row + 0.5) * w->ns_res;
}

void gcp_write_group(const struct gcp_list *l, const char *group,
                     const struct Cell_head *win)
{
    struct Control_Points cp;
    int i;

    memset(&cp, 0, sizeof(cp));
    for (i = 0; i < l->n; i++) {
        const struct gcp *g = &l->v[i];

        I_new_control_point(&cp, east(win, g->xs), north(win, g->ys),
                            east(win, g->xr), north(win, g->yr), 1);
    }
    if (I_put_control_points(group, &cp) < 0)
        G_fatal_error(_("Unable to write the control points of group <%s>"),
                      group);
    I_put_target(group, G_location(), G_mapset());
    G_message(_("%d control points written to imagery group <%s> (target "
                "<%s@%s>)"),
              l->n, group, G_location(), G_mapset());
    if (cp.count) {
        G_free(cp.e1);
        G_free(cp.n1);
        G_free(cp.e2);
        G_free(cp.n2);
        G_free(cp.status);
    }
}

void gcp_write_vector(const struct gcp_list *l, const char *name,
                      const struct Cell_head *win)
{
    struct Map_info Map;
    struct field_info *Fi;
    struct line_pnts *Points;
    struct line_cats *Cats;
    dbDriver *driver;
    dbString sql;
    char buf[1024];
    int i;

    if (Vect_open_new(&Map, name, WITHOUT_Z) < 0)
        G_fatal_error(_("Unable to create vector map <%s>"), name);
    Vect_hist_command(&Map);

    Fi = Vect_default_field_info(&Map, 1, NULL, GV_1TABLE);
    Vect_map_add_dblink(&Map, 1, NULL, Fi->table, GV_KEY_COLUMN, Fi->database,
                        Fi->driver);
    driver = db_start_driver_open_database(Fi->driver,
                                           Vect_subst_var(Fi->database, &Map));
    if (!driver)
        G_fatal_error(_("Unable to open database <%s> by driver <%s>"),
                      Fi->database, Fi->driver);
    db_set_error_handler_driver(driver);
    db_init_string(&sql);
    snprintf(buf, sizeof(buf),
             "CREATE TABLE %s (%s INTEGER, src_east DOUBLE PRECISION, "
             "src_north DOUBLE PRECISION, ref_east DOUBLE PRECISION, "
             "ref_north DOUBLE PRECISION, dx DOUBLE PRECISION, "
             "dy DOUBLE PRECISION, shift DOUBLE PRECISION, "
             "dist DOUBLE PRECISION, tile INTEGER, refined INTEGER, "
             "matches INTEGER)",
             Fi->table, GV_KEY_COLUMN);
    db_set_string(&sql, buf);
    if (db_execute_immediate(driver, &sql) != DB_OK)
        G_fatal_error(_("Unable to create table: %s"), db_get_string(&sql));
    if (db_create_index2(driver, Fi->table, GV_KEY_COLUMN) != DB_OK)
        G_warning(_("Unable to create index for table <%s>"), Fi->table);
    if (db_grant_on_table(driver, Fi->table, DB_PRIV_SELECT,
                          DB_GROUP | DB_PUBLIC) != DB_OK)
        G_fatal_error(_("Unable to grant privileges on table <%s>"),
                      Fi->table);

    Points = Vect_new_line_struct();
    Cats = Vect_new_cats_struct();
    db_begin_transaction(driver);
    for (i = 0; i < l->n; i++) {
        const struct gcp *g = &l->v[i];
        double e1 = east(win, g->xs), n1 = north(win, g->ys);
        double e2 = east(win, g->xr), n2 = north(win, g->yr);

        Vect_reset_line(Points);
        Vect_reset_cats(Cats);
        Vect_append_point(Points, e2, n2, 0.0);
        Vect_cat_set(Cats, 1, i + 1);
        Vect_write_line(&Map, GV_POINT, Points, Cats);
        snprintf(buf, sizeof(buf),
                 "INSERT INTO %s VALUES (%d, %.4f, %.4f, %.4f, %.4f, %.4f, "
                 "%.4f, %.4f, %.6g, %d, %d, %d)",
                 Fi->table, i + 1, e1, n1, e2, n2, e2 - e1, n2 - n1,
                 hypot(e2 - e1, n2 - n1), g->dist, g->tile, g->refined, g->n);
        db_set_string(&sql, buf);
        if (db_execute_immediate(driver, &sql) != DB_OK)
            G_fatal_error(_("Unable to insert into table: %s"),
                          db_get_string(&sql));
    }
    db_commit_transaction(driver);
    db_close_database_shutdown_driver(driver);
    db_free_string(&sql);
    Vect_destroy_line_struct(Points);
    Vect_destroy_cats_struct(Cats);
    Vect_build(&Map);
    Vect_close(&Map);
}
