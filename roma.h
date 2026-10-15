/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_ROMA_H
#define I_COREGISTRATION_OPENCL_ROMA_H

#include "ops.h"
#include "pth_loader.h"

#define ROMA_MULTIPLE 8   /* Tile sides: VGG features down to 1/8. */
#define ROMA_COARSE 560   /* Side of the coarse pass (training size). */
#define ROMA_PATCH 14     /* DINOv2 patch size. */
#define ROMA_VIT_DIM 1024 /* DINOv2 ViT-L/14 width. */
#define ROMA_VIT_DEPTH 24 /* DINOv2 blocks. */
#define ROMA_DEC_DIM 1024 /* Coordinate decoder width (GP 512 + 512). */
#define ROMA_DEC_DEPTH 5  /* Coordinate decoder blocks. */
#define ROMA_CLS_RES 64   /* Coarse warp classes: 64 x 64 anchors. */
#define ROMA_SCALES 5     /* Refiners at 1/16, 1/8, 1/4, 1/2, 1/1. */
#define ROMA_HIDDEN 8     /* Hidden blocks per refiner. */

/* One pre-norm transformer block (DINOv2's, or the coordinate
 * decoder's): LayerNorm, attention, LayerNorm, MLP; DINOv2's layer
 * scales are folded into proj and fc2. */
struct roma_block {
    cl_mem n1g, n1b, qkv_w, qkv_b, proj_w, proj_b;
    cl_mem n2g, n2b, fc1_w, fc1_b, fc2_w, fc2_b;
};

/* One ConvRefiner: block1 and ROMA_HIDDEN hidden blocks of a depthwise
 * 5 x 5 convolution with its BatchNorm folded, ReLU and a 1 x 1
 * convolution; out_conv to (dx, dy, certainty). */
struct roma_refiner {
    int scale, feat, emb, radius, cin; /* cin = 2 feat + emb + corr. */
    cl_mem dw_w[ROMA_HIDDEN + 1], dw_b[ROMA_HIDDEN + 1];
    cl_mem pw_w[ROMA_HIDDEN + 1], pw_b[ROMA_HIDDEN + 1];
    cl_mem out_w, out_b, emb_w, emb_b;
};

/* RoMa (Edstedt, Sun, Bökman, Wadenbäck and Felsberg, 2024) on an
 * OpenCL device, outdoor weights, image A warped onto image B: DINOv2
 * ViT-L/14 coarse features and VGG19-BN fine features, Gaussian process
 * match encoder, transformer coordinate decoder (classification over 64
 * x 64 anchors), ConvRefiners with local correlation at 1/16 .. 1/1. As
 * RoMa's match(): a coarse pass at ROMA_COARSE x ROMA_COARSE, then the
 * 1/8 .. 1/1 refiners again at the tile's own size ("upsample_res"). */
struct roma_model {
    struct ops *ops;
    struct ocl_backend *ocl;
    /* VGG19-BN convolutions (BatchNorm folded), the 16 of features[:40]. */
    cl_mem vgg_w[16], vgg_b[16];
    int vgg_cin[16], vgg_cout[16];
    /* DINOv2. */
    cl_mem patch_w, patch_b, cls, pos, norm_g, norm_b;
    struct roma_block vit[ROMA_VIT_DEPTH];
    /* Decoder: projections per scale (BatchNorm folded), GP, coordinate
     * decoder, refiners. */
    cl_mem proj_w[ROMA_SCALES], proj_b[ROMA_SCALES];
    int proj_in[ROMA_SCALES], proj_out[ROMA_SCALES];
    float *gp_basis; /* Host: cos(8 pi pos_conv(grid)), n16 x 512. */
    cl_mem gp_f;
    struct roma_block dec[ROMA_DEC_DEPTH];
    cl_mem to_out_w, to_out_b;
    struct roma_refiner ref[ROMA_SCALES];
    cl_kernel k_norm, k_cos, k_cls, k_warp, k_corr, k_disp, k_update, k_dw;

    /* Work buffers for tiles of H x W (multiples of ROMA_MULTIPLE). */
    int H, W, hm, wm;         /* hm x wm: the larger of the tile and the coarse
                                 pass, for buffers shared by both passes. */
    cl_mem img, v_a, v_b;     /* VGG activations. */
    cl_mem P[ROMA_SCALES][2]; /* Projected features per scale, image. */
    cl_mem t_x, t_n, t_qkv, t_o, t_h, t_out, t_s;       /* Transformers. */
    cl_mem g_y[2], g_nx, g_ny, g_kyy, g_kxy, g_p, g_mu; /* GP. */
    cl_mem d_a, d_b, d_c, fl[2], ce[2], cert16, lowres;
    /* Last run, host side: flow (2, H, W) normalized to -1..1 in image
     * B, certainty (H, W) in 0..1, as RoMa's match() (attenuated by the
     * coarse certainty, zero where the warp leaves the image). */
    float *flow, *cert;
    float *in_c[2], *in_f[2]; /* Normalized inputs of both passes. */
};

/* Read the weights: state_dict of RoMa's RegressionMatcher (roma_outdoor.pth)
 * and DINOv2 ViT-L/14 (dinov2_vitl14_pretrain.pth); fold the BatchNorms
 * and layer scales, upload. Fatal on any missing tensor or shape
 * mismatch. */
void roma_load(struct roma_model *m, struct ops *ops, struct pth_file *pf,
               struct pth_file *dino);

/* Work buffers for tiles of up to H x W cells. */
void roma_alloc(struct roma_model *m, int H, int W);

/* Dense warp of image A onto image B, both (3, H, W) in 0..1 at the
 * alloc size: results in m->flow and m->cert. With dump set (a folder),
 * the inputs and intermediate tensors of tile number tile are written
 * there for tests/reference_compare_roma.py (tileNNN_roma_<what>.f32). */
void roma_match(struct roma_model *m, const float *imgA, const float *imgB,
                const char *dump, int tile);

void roma_free(struct roma_model *m);

#endif /* I_COREGISTRATION_OPENCL_ROMA_H */
