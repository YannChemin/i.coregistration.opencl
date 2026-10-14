/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_IMAGE_H
#define I_COREGISTRATION_OPENCL_IMAGE_H

#include <grass/gis.h>

/* The current region as an 8-bit RGB image. */
struct rgb_image {
    int rows, cols;
    unsigned char *rgb;   /* rows * cols * 3, band order of the group. */
    unsigned char *valid; /* rows * cols, 0 where any band is NULL. */
    char names[3][GNAME_MAX + GMAPSET_MAX + 1];
};

/* Read the three raster maps of an imagery group (or of one of its
 * subgroups when subgroup is not NULL) over the current region as 8-bit
 * RGB. With stretch set, each band is stretched linearly between its lo
 * and hi percentiles to 0..255; without, bands within 0..255 are taken
 * as they are and others stretched between their 2 and 98 percentiles.
 * Fatal unless the (sub)group holds exactly three maps. */
void image_read_group(const char *group, const char *subgroup, int stretch,
                      double lo, double hi, struct rgb_image *img);

void image_free(struct rgb_image *img);

#endif /* I_COREGISTRATION_OPENCL_IMAGE_H */
