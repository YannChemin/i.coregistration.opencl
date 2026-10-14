/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H
#define I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H

#include "matcher.h"
#include "pth_loader.h"

/* Efficient LoFTR: detector-free semi-dense matches on grey images, to a
 * fraction of a cell, with the coarse confidence as cost (1 - conf).
 * The same network runs Efficient LoFTR's and MatchAnything's weights;
 * name is the matcher's name in messages; with npe, the RoPE positions
 * are scaled by ELOFTR_TRAIN_RES over the tile's long side. */
void matcher_eloftr_create(struct matcher *m, struct ops *ops,
                           struct pth_file *pf, const char *name, int npe);

#endif /* I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H */
