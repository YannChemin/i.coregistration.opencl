/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MATCHER_ROMA_H
#define I_COREGISTRATION_OPENCL_MATCHER_ROMA_H

#include "matcher.h"
#include "pth_loader.h"

/* RoMa: a dense warp of the image onto the reference with its
 * certainty, sampled into matches: in every block of spacing x spacing
 * cells, the cell of highest certainty if that certainty reaches
 * min_cert, with cost 1 - certainty. pf holds RoMa's weights (or
 * MatchAnything's), dino DINOv2 ViT-L/14's; imagenet: normalize the
 * inputs with the ImageNet mean and deviation (RoMa's, not
 * MatchAnything's). name: the matcher's name in messages. */
void matcher_roma_create(struct matcher *m, struct ops *ops,
                         struct pth_file *pf, struct pth_file *dino,
                         const char *name, int imagenet, int spacing,
                         float min_cert);

#endif /* I_COREGISTRATION_OPENCL_MATCHER_ROMA_H */
