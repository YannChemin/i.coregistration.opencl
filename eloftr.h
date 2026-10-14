/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_ELOFTR_H
#define I_COREGISTRATION_OPENCL_ELOFTR_H

#include "ops.h"
#include "pth_loader.h"

#define ELOFTR_MULTIPLE 32 /* Tile sides: 1/8 features, aggregated by 4. */
#define ELOFTR_DIM 256     /* Coarse feature dimension. */
#define ELOFTR_FINE_DIM 64 /* Fine feature dimension. */
#define ELOFTR_BLOCKS 21   /* RepVGG blocks: 1 + 2 + 4 + 14. */
#define ELOFTR_LAYERS 4    /* Transformer layers (self + cross each). */

/* One aggregated attention block of the coarse transformer. */
struct eloftr_attn {
    cl_mem q_agg;           /* Depthwise 4x4 (256, 1, 4, 4). */
    cl_mem norm_g, norm_b;  /* LayerNorm of the aggregated tokens. */
    cl_mem wq, wk, wv, wo;  /* (256, 256), no bias. */
    cl_mem fc1, fc2;        /* (512, 512), (256, 512), no bias. */
    cl_mem mlp_g, mlp_b;    /* LayerNorm after the MLP. */
};

/* Efficient LoFTR (Wang et al., 2024) on an OpenCL device, with the
 * outdoor weights of zju-community/efficientloftr (Hugging Face,
 * safetensors): RepVGG backbone re-parameterized at load, aggregated
 * attention transformer with 2D RoPE, coarse dual-softmax matching by
 * blocks, two-stage fine matching. */
struct eloftr_model {
    struct ops *ops;
    struct ocl_backend *ocl;
    cl_mem bw[ELOFTR_BLOCKS], bb[ELOFTR_BLOCKS]; /* Folded 3x3 convs. */
    int bcin[ELOFTR_BLOCKS], bcout[ELOFTR_BLOCKS], bstride[ELOFTR_BLOCKS];
    struct eloftr_attn self_attn[ELOFTR_LAYERS], cross_attn[ELOFTR_LAYERS];
    /* Fine fusion: out_conv, then two blocks (out_conv1 1x1, out_conv2
     * 3x3 with its BatchNorm folded, out_conv3 3x3). */
    cl_mem f_out, f1_w[2], f2_w[2], f2_b[2], f3_w[2];
    cl_kernel k_row_lse, k_col_lse, k_conf_row, k_conf_col, k_fine;

    /* Work buffers for tiles of H x W (multiples of ELOFTR_MULTIPLE). */
    int H, W, h, w, ah, aw, nmax;
    cl_mem img, s_a, s_b, f1[2], f2[2], x[2];   /* Backbone, per image. */
    cl_mem t_q, t_kv, t_qn, t_kvn, t_Q, t_K, t_V, t_O, t_O2, t_cat, t_tok,
        t_h1, t_h2, rope; /* Transformer. */
    cl_mem s_blk, row_lse, col_m, col_s, row_max, row_arg, col_max,
        col_arg; /* Coarse matching. */
    cl_mem fu_a, fu_b, fu_c, fine[2]; /* Fine fusion. */
    cl_mem m_idx, m_out;             /* Fine matching. */
    int blk_rows;
    /* Last run, for dumps: coarse matches (i, j, confidence). */
    int nm;
    int *mi, *mj;
    float *mconf, *kp; /* kp: x0, y0, x1, y1 per match (reference
                          convention: pixel index - 3.5). */
};

/* Read the weights (a safetensors file of EfficientLoFTRForKeypointMatching
 * or a state_dict with the same names), fold the BatchNorms and upload.
 * Fatal on any missing tensor or shape mismatch. */
void eloftr_load(struct eloftr_model *m, struct ops *ops, struct pth_file *pf);

/* Work buffers for tiles of up to H x W cells. */
void eloftr_alloc(struct eloftr_model *m, int H, int W);

/* Match two grey images (H x W floats in 0..1, the alloc size): coarse
 * mutual matches above the 0.2 confidence threshold, refined to a
 * fraction of a cell. Results in m->nm, m->kp, m->mconf. With dump set
 * (a folder), the intermediate tensors of tile number tile are written
 * there for tests/reference_compare_eloftr.py: tileNNN_eloftr_<what>.f32
 * for the 1/8 backbone features (backbone_0/1, 256 x H/8 x W/8), the
 * transformer output (coarse_0/1) and the fine features (fine_0/1,
 * 64 x H x W). */
void eloftr_match(struct eloftr_model *m, const float *grey0,
                  const float *grey1, const char *dump, int tile);

void eloftr_free(struct eloftr_model *m);

#endif /* I_COREGISTRATION_OPENCL_ELOFTR_H */
