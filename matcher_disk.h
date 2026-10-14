/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MATCHER_DISK_H
#define I_COREGISTRATION_OPENCL_MATCHER_DISK_H

#include "adalam.h"
#include "matcher.h"
#include "pth_loader.h"

/* DISK keypoints and descriptors on both images of a tile, matched by
 * nearest neighbour and filtered by AdaLAM (the translation of
 * RS_coreg_LoFTR). Matches lie on whole cells. */
struct disk_matcher_config {
    int features;    /* Keypoints per image and tile. */
    float threshold; /* Heatmap score threshold. */
    struct adalam_config adalam;
};

/* Load the DISK weights from pf (closed by the caller afterwards). */
void matcher_disk_create(struct matcher *m, struct ops *ops,
                         struct pth_file *pf,
                         const struct disk_matcher_config *cfg);

#endif /* I_COREGISTRATION_OPENCL_MATCHER_DISK_H */
