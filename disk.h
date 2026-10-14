/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_DISK_H
#define I_COREGISTRATION_OPENCL_DISK_H

#include "ocl_backend.h"
#include "pth_loader.h"

#define DISK_DESC 128   /* Descriptor dimension. */
#define DISK_LAYERS 9   /* Five down blocks, four up blocks. */
#define DISK_MULTIPLE 16 /* Input sides must be multiples of this. */

/* DISK (Tyszkiewicz et al., 2020) on an OpenCL device: the U-Net of
 * kornia.feature.DISK (5x5 kernels, down 16-32-64-64-64, up 64-64-64-129)
 * with its pretrained weights, heatmap NMS and descriptor sampling. */
struct disk_model {
    struct ocl_backend *ocl;
    int cin[DISK_LAYERS], cout[DISK_LAYERS];
    cl_mem w[DISK_LAYERS], b[DISK_LAYERS], alpha[DISK_LAYERS];
    cl_kernel k_conv, k_stats, k_pool, k_up, k_nms, k_gather;
    /* Work buffers for tiles of up to H x W cells. */
    int H, W, nmax;
    cl_mem input, f[5], u[3], tmp, cat, out, mean, rstd, mask, xy, desc;
    float *heat;
    unsigned char *maskh;
};

/* Read the DISK weights (the "extractor" state_dict of depth-save.pth or
 * epipolar-save.pth, or a bare state_dict with the same names) and
 * upload them. Fatal on any missing tensor or shape mismatch. */
void disk_load(struct disk_model *m, struct ocl_backend *ocl,
               struct pth_file *pf);

/* Allocate work buffers for tiles of up to H x W cells (multiples of
 * DISK_MULTIPLE) and up to nmax keypoints. */
void disk_alloc(struct disk_model *m, int H, int W, int nmax);

/* Detect keypoints on img, a (3, H, W) float image in 0..1 whose content
 * covers its top-left h x w cells (the rest zero padding, as
 * pad_if_not_divisible): NMS over 5x5 windows on the heatmap cropped to
 * h x w, cells above thresh, then the n highest (kornia
 * heatmap_to_keypoints), skipping cells where valid is 0 when valid is
 * not NULL. Keypoints come in raster order: xy (x, y pairs), score (the
 * heatmap value) and L2-normalized descriptors (n x DISK_DESC). Returns
 * the number of keypoints. */
int disk_detect(struct disk_model *m, const float *img, int H, int W, int h,
                int w, int n, float thresh, const unsigned char *valid,
                int *xy, float *score, float *desc);

void disk_free(struct disk_model *m);

#endif /* I_COREGISTRATION_OPENCL_DISK_H */
