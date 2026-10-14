/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Helpers shared by the matchers: match lists and debug
 *               dumps.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdio.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "matcher.h"

void match_add(struct match_list *l, const struct match *m)
{
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 1024;
        l->v = G_realloc(l->v, l->cap * sizeof(*l->v));
    }
    l->v[l->n++] = *m;
}

void matcher_dump_file(const char *dir, int tile, const char *what,
                       const void *data, size_t bytes)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/tile%03d_%s", dir, tile, what);
    if (!(fp = fopen(path, "wb")) || fwrite(data, 1, bytes, fp) != bytes)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
}
