/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MATCHER_H
#define I_COREGISTRATION_OPENCL_MATCHER_H

#include "ops.h"

/* One match between the two images of a tile, in cells of the tile
 * (column, row; cell centres at integers). */
struct match {
    double x1, y1; /* Image to co-register. */
    double x2, y2; /* Reference. */
    float cost;    /* Lower is better (AdaLAM's inverse confidence, or
                      1 - confidence for matchers that give one). */
};

struct match_list {
    int n, cap;
    struct match *v;
};

void match_add(struct match_list *l, const struct match *m);

/* One tile of both images: (3, H, W) floats in 0..1 (R, G, B planes)
 * whose content covers the top-left h x w cells, the rest zero padding,
 * and the validity of those cells (h x w, 0 where a band is NULL). */
struct matcher_tile {
    const float *img1, *img2;
    const unsigned char *valid1, *valid2;
    int H, W, h, w;
    int x0, y0; /* Top-left cell in the region, for diagnostics. */
    int index;  /* 1-based, for diagnostics and dumps. */
};

/* A matcher: a network on the OpenCL device (run) followed by host-side
 * matching (match), so that the two stages can be timed and reported
 * apart. Every matcher uses the generic kernels of ops. */
struct matcher {
    const char *name;
    int multiple; /* Tile sides must be multiples of this. */
    struct ops *ops;
    void *state;
    /* Developer aid: folder of the debug dumps (I_COREGISTRATION_OPENCL_DUMP)
     * or NULL; matchers that dump during run read it there. */
    const char *dump_dir;
    /* Work buffers for tiles of up to H x W cells. */
    void (*alloc)(struct matcher *m, int H, int W);
    /* Device stage on one tile; returns once its results are read back
     * (so that the stage can be timed). */
    void (*run)(struct matcher *m, const struct matcher_tile *t);
    /* Host stage: append the tile's matches to out; returns their
     * number. */
    int (*match)(struct matcher *m, const struct matcher_tile *t,
                 struct match_list *out);
    /* Developer aid: write the tile's inputs and intermediate results
     * into folder dir (I_COREGISTRATION_OPENCL_DUMP), after match. */
    void (*dump)(struct matcher *m, const struct matcher_tile *t,
                 const char *dir);
    void (*destroy)(struct matcher *m);
};

/* Write bytes to <dir>/tile<NNN>_<what>, fatal on failure. */
void matcher_dump_file(const char *dir, int tile, const char *what,
                       const void *data, size_t bytes);

#endif /* I_COREGISTRATION_OPENCL_MATCHER_H */
