/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H
#define I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H

#include "matcher.h"
#include "pth_loader.h"

/* Efficient LoFTR: detector-free semi-dense matches on grey images, to a
 * fraction of a cell, with the coarse confidence as cost (1 - conf). */
void matcher_eloftr_create(struct matcher *m, struct ops *ops,
                           struct pth_file *pf);

#endif /* I_COREGISTRATION_OPENCL_MATCHER_ELOFTR_H */
