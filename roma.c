/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      RoMa (Edstedt, Sun, Bökman, Wadenbäck and Felsberg, 2024)
 *               on an OpenCL device: DINOv2 ViT-L/14 and VGG19-BN
 *               features, Gaussian process match encoder, transformer
 *               coordinate decoder, ConvRefiners with local correlation;
 *               a coarse pass at 560 x 560 and a fine pass at the tile's
 *               size, as RoMa's match().
 *
 *               Translated from the authors' code (romatch, commit
 *               77f8d68, MIT License, Copyright (c) 2023 Johan Edstedt),
 *               with DINOv2's vision transformer (Apache License 2.0,
 *               Copyright (c) Meta Platforms, Inc. and affiliates).
 *               Changes: C and OpenCL instead of PyTorch, float32
 *               throughout (no automatic mixed precision); BatchNorms
 *               and layer scales folded into the convolutions and
 *               linear layers at load; the Gaussian process's Cholesky
 *               solve on the host in double precision; one direction
 *               (A to B) instead of the symmetric batch; images resized
 *               for the coarse pass by an antialiased bicubic filter
 *               written after Pillow's (the reference resizes with
 *               Pillow).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "progress.h"
#include "roma.h"

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

#define EPS_BN 1e-5f
#define EPS_LN_VIT 1e-6f /* DINOv2. */
#define EPS_LN_DEC 1e-5f /* Coordinate decoder (nn.LayerNorm default). */
#define VIT_HEADS 16
#define DEC_HEADS 8
#define MLP_RATIO 4
#define GP_DIM 512
#define GP_T 0.2f
#define GP_NOISE 0.1
#define COS_EPS 1e-6f
#define VIT_GRID 37 /* DINOv2's position embeddings: 518 / 14. */
#define NCLS (ROMA_CLS_RES * ROMA_CLS_RES)
#define C16 (ROMA_COARSE / ROMA_PATCH) /* Coarse grid side: 40. */
#define N16 (C16 * C16)
#define NTOK (N16 + 1) /* DINOv2 tokens with the class token. */

/* Refiners, from roma_model() of the outdoor model: scale, features per
 * image, displacement embedding, local correlation radius. */
static const int ref_scale[ROMA_SCALES] = {16, 8, 4, 2, 1};
static const int ref_feat[ROMA_SCALES] = {512, 512, 256, 64, 9};
static const int ref_emb[ROMA_SCALES] = {128, 64, 32, 16, 6};
static const int ref_radius[ROMA_SCALES] = {7, 3, 2, 0, 0};
static const int proj_in[ROMA_SCALES] = {1024, 512, 256, 128, 64};

/* VGG19-BN features[:40]: convolutions per block and their widths; the
 * BatchNorm of the convolution at index i is at i + 1. */
static const int vgg_blocks[4] = {2, 2, 4, 4};
static const int vgg_width[4] = {64, 128, 256, 512};

static const float im_mean[3] = {0.485f, 0.456f, 0.406f};
static const float im_std[3] = {0.229f, 0.224f, 0.225f};

static float *read_host(struct pth_file *pf, const char *name, int ndim,
                        const long *shape)
{
    return pth_read_f32(pf, pth_require(pf, name, ndim, shape));
}

static cl_mem upload(struct roma_model *m, const float *v, size_t n,
                     const char *what)
{
    return ocl_upload(m->ocl, v, n * sizeof(float), what);
}

static cl_mem load_dev(struct roma_model *m, struct pth_file *pf,
                       const char *name, int ndim, const long *shape)
{
    float *v = read_host(pf, name, ndim, shape);
    long n = 1;
    int d;
    cl_mem b;

    for (d = 0; d < ndim; d++)
        n *= shape[d];
    b = upload(m, v, n, name);
    G_free(v);
    return b;
}

/* BatchNorm <prefix>.{weight,bias,running_mean,running_var} of c
 * channels folded into weights w (c x per) and bias b. */
static void fold(struct pth_file *pf, const char *prefix, int c, long per,
                 float *w, float *b)
{
    char name[256];
    long shape[1] = {c};
    float *g, *bt, *mu, *var;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    g = read_host(pf, name, 1, shape);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    bt = read_host(pf, name, 1, shape);
    snprintf(name, sizeof(name), "%s.running_mean", prefix);
    mu = read_host(pf, name, 1, shape);
    snprintf(name, sizeof(name), "%s.running_var", prefix);
    var = read_host(pf, name, 1, shape);
    ops_fold_bn(w, b, c, per, g, bt, mu, var, EPS_BN);
    G_free(g);
    G_free(bt);
    G_free(mu);
    G_free(var);
}

/* Convolution <conv> (co x per weights, bias) followed by BatchNorm
 * <bn>, folded and uploaded. */
static void load_conv_bn(struct roma_model *m, struct pth_file *pf,
                         const char *conv, const char *bn, int co, int ndim,
                         const long *shape, cl_mem *w_out, cl_mem *b_out)
{
    char name[256];
    long per = 1, bs[1] = {co};
    float *w, *b;
    int d;

    for (d = 1; d < ndim; d++)
        per *= shape[d];
    snprintf(name, sizeof(name), "%s.weight", conv);
    w = read_host(pf, name, ndim, shape);
    snprintf(name, sizeof(name), "%s.bias", conv);
    if (pth_find(pf, name))
        b = read_host(pf, name, 1, bs);
    else
        b = G_calloc(co, sizeof(float));
    fold(pf, bn, co, per, w, b);
    *w_out = upload(m, w, (size_t)co * per, conv);
    *b_out = upload(m, b, co, conv);
    G_free(w);
    G_free(b);
}

/* One transformer block <pre>.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,
 * mlp.fc2}; with layer scales (DINOv2's ls1.gamma, ls2.gamma) folded
 * into proj and fc2; with or without qkv bias. */
static void load_block(struct roma_model *m, struct pth_file *pf,
                       const char *pre, int D, int qkv_bias, int ls,
                       struct roma_block *b)
{
    char name[256];
    long v1[1] = {D}, sqkv[2] = {3 * D, D}, s3[1] = {3 * D};
    long sp[2] = {D, D}, s1[2] = {MLP_RATIO * D, D}, sb1[1] = {MLP_RATIO * D};
    long s2[2] = {D, MLP_RATIO * D};
    float *w, *bias, *g;
    int i, j;

#define LD(field, suffix, nd, shape)                                           \
    snprintf(name, sizeof(name), "%s.%s", pre, suffix);                        \
    b->field = load_dev(m, pf, name, nd, shape)
    LD(n1g, "norm1.weight", 1, v1);
    LD(n1b, "norm1.bias", 1, v1);
    LD(qkv_w, "attn.qkv.weight", 2, sqkv);
    if (qkv_bias) {
        LD(qkv_b, "attn.qkv.bias", 1, s3);
    }
    else
        b->qkv_b = NULL;
    LD(n2g, "norm2.weight", 1, v1);
    LD(n2b, "norm2.bias", 1, v1);
    LD(fc1_w, "mlp.fc1.weight", 2, s1);
    LD(fc1_b, "mlp.fc1.bias", 1, sb1);
#undef LD
    /* proj and fc2, times the layer scale per output channel. */
    for (j = 0; j < 2; j++) {
        const char *wn = j ? "mlp.fc2" : "attn.proj";
        int in = j ? MLP_RATIO * D : D;

        snprintf(name, sizeof(name), "%s.%s.weight", pre, wn);
        w = read_host(pf, name, 2, j ? s2 : sp);
        snprintf(name, sizeof(name), "%s.%s.bias", pre, wn);
        bias = read_host(pf, name, 1, v1);
        if (ls) {
            snprintf(name, sizeof(name), "%s.ls%d.gamma", pre, j + 1);
            g = read_host(pf, name, 1, v1);
            for (i = 0; i < D; i++) {
                int c;

                for (c = 0; c < in; c++)
                    w[(size_t)i * in + c] *= g[i];
                bias[i] *= g[i];
            }
            G_free(g);
        }
        if (j) {
            b->fc2_w = upload(m, w, (size_t)D * in, "fc2");
            b->fc2_b = upload(m, bias, D, "fc2");
        }
        else {
            b->proj_w = upload(m, w, (size_t)D * in, "proj");
            b->proj_b = upload(m, bias, D, "proj");
        }
        G_free(w);
        G_free(bias);
    }
}

static void load_vgg(struct roma_model *m, struct pth_file *pf)
{
    int blk, j, l = 0, idx = 0, cin = 3;

    for (blk = 0; blk < 4; blk++) {
        for (j = 0; j < vgg_blocks[blk]; j++, l++) {
            int co = vgg_width[blk];
            long shape[4] = {co, cin, 3, 3};
            char conv[64], bn[64];

            snprintf(conv, sizeof(conv), "encoder.cnn.layers.%d", idx);
            snprintf(bn, sizeof(bn), "encoder.cnn.layers.%d", idx + 1);
            load_conv_bn(m, pf, conv, bn, co, 4, shape, &m->vgg_w[l],
                         &m->vgg_b[l]);
            m->vgg_cin[l] = cin;
            m->vgg_cout[l] = co;
            cin = co;
            idx += 3; /* Convolution, BatchNorm, ReLU. */
        }
        idx++; /* MaxPool. */
    }
}

/* PyTorch's bicubic convolution weights (A = -0.75) at fraction t. */
static void cubic_weights(double t, double w[4])
{
    const double A = -0.75;
    double x;

    x = t + 1;
    w[0] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x = t;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x = 1 - t;
    w[2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x = 2 - t;
    w[3] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

/* DINOv2's interpolate_pos_encoding for a C16 x C16 grid: the 37 x 37
 * patch embeddings resized by F.interpolate(mode="bicubic",
 * scale_factor=(C16 + 0.1) / 37), whose source coordinates use that
 * factor rather than the size ratio; the class embedding first. */
static float *interp_pos(const float *pos)
{
    float *out = G_malloc((size_t)NTOK * ROMA_VIT_DIM * sizeof(float));
    double inv = VIT_GRID / (C16 + 0.1);
    int oy, ox, c, i, j;

    memcpy(out, pos, ROMA_VIT_DIM * sizeof(float));
    for (oy = 0; oy < C16; oy++)
        for (ox = 0; ox < C16; ox++) {
            double ry = inv * (oy + 0.5) - 0.5, rx = inv * (ox + 0.5) - 0.5;
            int iy = (int)floor(ry), ix = (int)floor(rx);
            double wy[4], wx[4];
            float *o = out + (size_t)(1 + oy * C16 + ox) * ROMA_VIT_DIM;

            cubic_weights(ry - iy, wy);
            cubic_weights(rx - ix, wx);
            for (c = 0; c < ROMA_VIT_DIM; c++) {
                double s = 0;

                for (i = 0; i < 4; i++) {
                    int yy = iy - 1 + i;

                    yy = yy < 0 ? 0 : yy > VIT_GRID - 1 ? VIT_GRID - 1 : yy;
                    for (j = 0; j < 4; j++) {
                        int xx = ix - 1 + j;

                        xx = xx < 0 ? 0 : xx > VIT_GRID - 1 ? VIT_GRID - 1 : xx;
                        s += wy[i] * wx[j] *
                             pos[(size_t)(1 + yy * VIT_GRID + xx) *
                                     ROMA_VIT_DIM +
                                 c];
                    }
                }
                o[c] = (float)s;
            }
        }
    return out;
}

static void load_dino(struct roma_model *m, struct pth_file *pf)
{
    long sw[4] = {ROMA_VIT_DIM, 3, ROMA_PATCH, ROMA_PATCH};
    long v1[1] = {ROMA_VIT_DIM}, sc[3] = {1, 1, ROMA_VIT_DIM};
    long sp[3] = {1, VIT_GRID * VIT_GRID + 1, ROMA_VIT_DIM};
    float *pos, *ip;
    int i;

    m->patch_w = load_dev(m, pf, "patch_embed.proj.weight", 4, sw);
    m->patch_b = load_dev(m, pf, "patch_embed.proj.bias", 1, v1);
    m->cls = load_dev(m, pf, "cls_token", 3, sc);
    pos = read_host(pf, "pos_embed", 3, sp);
    ip = interp_pos(pos);
    m->pos = upload(m, ip, (size_t)NTOK * ROMA_VIT_DIM, "pos_embed");
    G_free(pos);
    G_free(ip);
    for (i = 0; i < ROMA_VIT_DEPTH; i++) {
        char pre[32];

        snprintf(pre, sizeof(pre), "blocks.%d", i);
        load_block(m, pf, pre, ROMA_VIT_DIM, 1, 1, &m->vit[i]);
    }
    m->norm_g = load_dev(m, pf, "norm.weight", 1, v1);
    m->norm_b = load_dev(m, pf, "norm.bias", 1, v1);
}

/* GP basis of the coarse grid of image B: cos(8 pi pos_conv(x, y)) at
 * the C16 x C16 cell centres, token-major (N16 x GP_DIM). */
static void load_gp(struct roma_model *m, struct pth_file *pf)
{
    long sw[4] = {GP_DIM, 2, 1, 1}, sb[1] = {GP_DIM};
    float *w = read_host(pf, "decoder.gps.16.pos_conv.weight", 4, sw);
    float *b = read_host(pf, "decoder.gps.16.pos_conv.bias", 1, sb);
    int i, d;

    m->gp_basis = G_malloc((size_t)N16 * GP_DIM * sizeof(float));
    for (i = 0; i < N16; i++) {
        float x =
            -1.0f + 1.0f / C16 + (i % C16) * (2.0f - 2.0f / C16) / (C16 - 1);
        float y =
            -1.0f + 1.0f / C16 + (i / C16) * (2.0f - 2.0f / C16) / (C16 - 1);

        for (d = 0; d < GP_DIM; d++) {
            float v = w[2 * d] * x + w[2 * d + 1] * y + b[d];

            m->gp_basis[(size_t)i * GP_DIM + d] = cosf(8.0f * (float)M_PI * v);
        }
    }
    G_free(w);
    G_free(b);
}

static void load_refiner(struct roma_model *m, struct pth_file *pf, int s)
{
    struct roma_refiner *r = &m->ref[s];
    char pre[64], conv[96], bn[96], name[128];
    int b, corr;

    r->scale = ref_scale[s];
    r->feat = ref_feat[s];
    r->emb = ref_emb[s];
    r->radius = ref_radius[s];
    corr = r->radius ? (2 * r->radius + 1) * (2 * r->radius + 1) : 0;
    r->cin = 2 * r->feat + r->emb + corr;
    snprintf(pre, sizeof(pre), "decoder.conv_refiner.%d", r->scale);
    for (b = 0; b <= ROMA_HIDDEN; b++) {
        long sdw[4] = {r->cin, 1, 5, 5}, spw[4] = {r->cin, r->cin, 1, 1};
        long sb[1] = {r->cin};
        char blk[96];

        if (b == 0)
            snprintf(blk, sizeof(blk), "%s.block1", pre);
        else
            snprintf(blk, sizeof(blk), "%s.hidden_blocks.%d", pre, b - 1);
        snprintf(conv, sizeof(conv), "%s.0", blk);
        snprintf(bn, sizeof(bn), "%s.1", blk);
        load_conv_bn(m, pf, conv, bn, r->cin, 4, sdw, &r->dw_w[b], &r->dw_b[b]);
        snprintf(name, sizeof(name), "%s.3.weight", blk);
        r->pw_w[b] = load_dev(m, pf, name, 4, spw);
        snprintf(name, sizeof(name), "%s.3.bias", blk);
        r->pw_b[b] = load_dev(m, pf, name, 1, sb);
    }
    {
        long so[4] = {3, r->cin, 1, 1}, sob[1] = {3};
        long se[4] = {r->emb, 2, 1, 1}, seb[1] = {r->emb};

        snprintf(name, sizeof(name), "%s.out_conv.weight", pre);
        r->out_w = load_dev(m, pf, name, 4, so);
        snprintf(name, sizeof(name), "%s.out_conv.bias", pre);
        r->out_b = load_dev(m, pf, name, 1, sob);
        snprintf(name, sizeof(name), "%s.disp_emb.weight", pre);
        r->emb_w = load_dev(m, pf, name, 4, se);
        snprintf(name, sizeof(name), "%s.disp_emb.bias", pre);
        r->emb_b = load_dev(m, pf, name, 1, seb);
    }
}

void roma_load(struct roma_model *m, struct ops *ops, struct pth_file *pf,
               struct pth_file *dino)
{
    long so[2] = {NCLS + 1, ROMA_DEC_DIM}, sob[1] = {NCLS + 1};
    int s, i;

    memset(m, 0, sizeof(*m));
    m->ops = ops;
    m->ocl = ops->ocl;
    m->imagenet = 1;
    /* MatchAnything's Lightning checkpoint. */
    pth_strip_prefix(pf, "matcher.model.");
    progress_begin(PG_LOAD,
                   (pth_bytes(pf, "") + pth_bytes(dino, "")) / 1073741824.0);
    load_vgg(m, pf);
    load_dino(m, dino);
    for (s = 0; s < ROMA_SCALES; s++) {
        long shape[4] = {ref_feat[s], proj_in[s], 1, 1};
        char conv[64], bn[64];

        m->proj_in[s] = proj_in[s];
        m->proj_out[s] = ref_feat[s];
        snprintf(conv, sizeof(conv), "decoder.proj.%d.0", ref_scale[s]);
        snprintf(bn, sizeof(bn), "decoder.proj.%d.1", ref_scale[s]);
        load_conv_bn(m, pf, conv, bn, ref_feat[s], 4, shape, &m->proj_w[s],
                     &m->proj_b[s]);
    }
    load_gp(m, pf);
    for (i = 0; i < ROMA_DEC_DEPTH; i++) {
        char pre[64];

        snprintf(pre, sizeof(pre), "decoder.embedding_decoder.blocks.%d", i);
        load_block(m, pf, pre, ROMA_DEC_DIM, 0, 0, &m->dec[i]);
    }
    m->to_out_w =
        load_dev(m, pf, "decoder.embedding_decoder.to_out.weight", 2, so);
    m->to_out_b =
        load_dev(m, pf, "decoder.embedding_decoder.to_out.bias", 1, sob);
    for (s = 0; s < ROMA_SCALES; s++)
        load_refiner(m, pf, s);
    progress_end();
    m->k_norm = ocl_kernel(m->ocl, "roma_row_norm");
    m->k_cos = ocl_kernel(m->ocl, "roma_cos_exp");
    m->k_cls = ocl_kernel(m->ocl, "roma_cls_to_flow");
    m->k_warp = ocl_kernel(m->ocl, "roma_warp_sample");
    m->k_corr = ocl_kernel(m->ocl, "roma_local_corr");
    m->k_disp = ocl_kernel(m->ocl, "roma_disp");
    m->k_update = ocl_kernel(m->ocl, "roma_update");
    m->k_dw = ocl_kernel(m->ocl, "roma_dwconv5");
}

static cl_mem walloc(struct roma_model *m, size_t floats, const char *what)
{
    return ocl_alloc(m->ocl, floats * sizeof(float), what);
}

static size_t max_sz(size_t a, size_t b) { return a > b ? a : b; }

/* Side of scale index s (16, 8, .. 1) for a pass of side n. */
static int side(int n, int s)
{
    return s == 0 ? n / ROMA_PATCH : n / ref_scale[s];
}

void roma_alloc(struct roma_model *m, int H, int W)
{
    size_t hm, dmax = 0, n16 = N16;
    int k, s, p;

    if (H % ROMA_MULTIPLE || W % ROMA_MULTIPLE)
        G_fatal_error(_("RoMa tile %d x %d is not a multiple of %d"), H, W,
                      ROMA_MULTIPLE);
    m->H = H;
    m->W = W;
    m->hm = H > ROMA_COARSE ? H : ROMA_COARSE;
    m->wm = W > ROMA_COARSE ? W : ROMA_COARSE;
    hm = (size_t)m->hm * m->wm;
    m->img = walloc(m, 3 * hm, "image");
    m->v_a = walloc(m, 64 * hm, "VGG");
    m->v_b = walloc(m, 64 * hm, "VGG");
    for (k = 0; k < 2; k++) {
        m->P[0][k] = walloc(m, (size_t)GP_DIM * n16, "1/16 features");
        for (s = 1; s < ROMA_SCALES; s++)
            m->P[s][k] = walloc(m,
                                (size_t)ref_feat[s] * (m->hm / ref_scale[s]) *
                                    (m->wm / ref_scale[s]),
                                "projected features");
        m->g_y[k] = walloc(m, n16 * GP_DIM, "GP tokens");
    }
    m->t_x = walloc(m, (size_t)NTOK * ROMA_VIT_DIM, "tokens");
    m->t_n = walloc(m, (size_t)NTOK * ROMA_VIT_DIM, "tokens");
    m->t_qkv = walloc(m, (size_t)NTOK * 3 * ROMA_VIT_DIM, "qkv");
    m->t_o = walloc(m, (size_t)NTOK * ROMA_VIT_DIM, "attention");
    m->t_h = walloc(m, (size_t)NTOK * MLP_RATIO * ROMA_VIT_DIM, "mlp");
    m->t_out = walloc(m, n16 * (NCLS + 1), "anchor logits");
    m->t_s = walloc(m, (size_t)VIT_HEADS * NTOK * NTOK, "attention scores");
    m->g_nx = walloc(m, n16, "norms");
    m->g_ny = walloc(m, n16, "norms");
    m->g_kyy = walloc(m, n16 * n16, "GP kernel");
    m->g_kxy = walloc(m, n16 * n16, "GP kernel");
    m->g_p = walloc(m, n16 * GP_DIM, "GP solve");
    m->g_mu = walloc(m, n16 * GP_DIM, "GP posterior");
    m->gp_f = upload(m, m->gp_basis, n16 * GP_DIM, "GP basis");
    /* Refiner inputs at every scale of both passes. */
    for (p = 0; p < 2; p++)
        for (s = p; s < ROMA_SCALES; s++) {
            int ph = p ? H : ROMA_COARSE, pw = p ? W : ROMA_COARSE;
            size_t n = (size_t)side(ph, s) * side(pw, s);

            dmax = max_sz(dmax, (size_t)m->ref[s].cin * n);
        }
    m->d_a = walloc(m, dmax, "refiner");
    m->d_b = walloc(m, dmax, "refiner");
    m->d_c = walloc(m, max_sz(3 * hm, (size_t)ROMA_VIT_DIM * n16), "refiner");
    for (k = 0; k < 2; k++) {
        m->fl[k] = walloc(m, 2 * hm, "warp");
        m->ce[k] = walloc(m, hm, "certainty");
    }
    m->cert16 = walloc(m, n16, "coarse certainty");
    m->lowres = walloc(m, (size_t)H * W, "coarse certainty");
    m->flow = G_malloc(2 * (size_t)H * W * sizeof(float));
    m->cert = G_malloc((size_t)H * W * sizeof(float));
    for (k = 0; k < 2; k++) {
        m->in_c[k] =
            G_malloc(3 * (size_t)ROMA_COARSE * ROMA_COARSE * sizeof(float));
        m->in_f[k] = G_malloc(3 * (size_t)H * W * sizeof(float));
    }
}

static void run(struct roma_model *m, cl_kernel k, int dims, const size_t *g,
                const size_t *l, const char *what)
{
    ops_run(m->ocl, k, dims, g, l, what);
}

static void copy(struct roma_model *m, cl_mem src, size_t soff, cl_mem dst,
                 size_t doff, size_t floats)
{
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, src, dst, soff * sizeof(float),
                                  doff * sizeof(float), floats * sizeof(float),
                                  0, NULL, NULL),
              "clEnqueueCopyBuffer");
}

/* Copy rows x cols floats from src (row pitch spitch floats) into dst
 * (row pitch dpitch) at column dcol. */
static void copy_rect(struct roma_model *m, cl_mem src, cl_mem dst, int rows,
                      int cols, int spitch, int dpitch, int dcol)
{
    size_t so[3] = {0, 0, 0}, dor[3] = {(size_t)dcol * sizeof(float), 0, 0};
    size_t reg[3] = {(size_t)cols * sizeof(float), (size_t)rows, 1};

    ocl_check(clEnqueueCopyBufferRect(m->ocl->queue, src, dst, so, dor, reg,
                                      (size_t)spitch * sizeof(float), 0,
                                      (size_t)dpitch * sizeof(float), 0, 0,
                                      NULL, NULL),
              "clEnqueueCopyBufferRect");
}

/* Multi-head attention of the qkv rows (N x 3D: queries, keys, values,
 * head h at columns h * hd of each) into t_o (N x D), through the full
 * score matrices (heads x N x N in t_s): two batched GEMMs around a row
 * softmax, faster here than the blockwise kernel for these N. */
static void attention(struct roma_model *m, int N, int D, int heads)
{
    struct ops *o = m->ops;
    int hd = D / heads;
    long nn = (long)N * N;

    ops_gemm(o, N, N, hd, ops_bmat(m->t_qkv, 0, 3 * D, hd, 0),
             ops_bmat(m->t_qkv, D, 3 * D, hd, 0), 1,
             ops_bmat(m->t_s, 0, N, nn, 0), heads, 1, NULL, 0, NULL, 0,
             1.0f / sqrtf((float)hd), ACT_NONE, 0);
    ops_softmax_rows(o, m->t_s, (long)heads * N, N, 1.0f);
    ops_gemm(o, N, hd, N, ops_bmat(m->t_s, 0, N, nn, 0),
             ops_bmat(m->t_qkv, 2 * D, 3 * D, hd, 0), 0,
             ops_bmat(m->t_o, 0, D, hd, 0), heads, 1, NULL, 0, NULL, 0, 1.0f,
             ACT_NONE, 0);
}

/* Pre-norm transformer block on X (N x D tokens) in place. */
static void block(struct roma_model *m, const struct roma_block *b, int N,
                  int D, int heads, float eps)
{
    struct ops *o = m->ops;

    ops_layernorm(o, N, D, m->t_x, m->t_n, b->n1g, b->n1b, eps, ACT_NONE, 0);
    ops_linear(o, N, D, 3 * D, m->t_n, 0, b->qkv_w, b->qkv_b, m->t_qkv, 0,
               ACT_NONE, 0, 0);
    attention(m, N, D, heads);
    ops_linear(o, N, D, D, m->t_o, 0, b->proj_w, b->proj_b, m->t_x, 0, ACT_NONE,
               0, 1);
    ops_layernorm(o, N, D, m->t_x, m->t_n, b->n2g, b->n2b, eps, ACT_NONE, 0);
    ops_linear(o, N, D, MLP_RATIO * D, m->t_n, 0, b->fc1_w, b->fc1_b, m->t_h, 0,
               ACT_GELU, 0, 0);
    ops_linear(o, N, MLP_RATIO * D, D, m->t_h, 0, b->fc2_w, b->fc2_b, m->t_x, 0,
               ACT_NONE, 0, 1);
}

/* DINOv2 on m->img (3 x 560 x 560, normalized): the normalized patch
 * tokens, projected (decoder.proj.16), into P[0][k]. */
static void dino(struct roma_model *m, int k, cl_mem feat_out)
{
    struct ops *o = m->ops;
    struct conv_geom g = ops_geom(ROMA_PATCH, ROMA_PATCH, 0);
    int D = ROMA_VIT_DIM, i;

    ops_conv2d(o, m->img, 3, ROMA_COARSE, ROMA_COARSE, m->patch_w, m->patch_b,
               D, &g, NULL, NULL, NULL, ACT_NONE, 0, m->t_h);
    ops_transpose(o, m->t_h, m->t_o, 1, D, N16);
    copy(m, m->cls, 0, m->t_x, 0, D);
    copy(m, m->t_o, 0, m->t_x, D, (size_t)N16 * D);
    ops_axpby(o, m->t_x, m->t_x, 1.0f, m->pos, 1.0f, (long)NTOK * D,
              (long)NTOK * D);
    for (i = 0; i < ROMA_VIT_DEPTH; i++)
        block(m, &m->vit[i], NTOK, D, VIT_HEADS, EPS_LN_VIT);
    ops_layernorm(o, NTOK, D, m->t_x, m->t_n, m->norm_g, m->norm_b, EPS_LN_VIT,
                  ACT_NONE, 0);
    copy(m, m->t_n, D, m->t_o, 0, (size_t)N16 * D);
    ops_transpose(o, m->t_o, feat_out, 1, N16, D);
    ops_conv1x1(o, feat_out, 0, D, N16, m->proj_w[0], m->proj_b[0], GP_DIM,
                m->P[0][k], 0, ACT_NONE, 0);
}

/* VGG19-BN on m->img (3 x H x W, normalized) of image k: the features
 * before each pooling, projected, into P[4] (1/1) .. P[1] (1/8). */
static void vgg(struct roma_model *m, int H, int W, int k)
{
    struct ops *o = m->ops;
    struct conv_geom g = ops_geom(3, 1, 1);
    cl_mem in = m->img, out;
    int blk, j, l = 0, h = H, w = W;

    for (blk = 0; blk < 4; blk++) {
        int s = 4 - blk; /* Scale index of 1, 2, 4, 8. */

        for (j = 0; j < vgg_blocks[blk]; j++, l++) {
            out = in == m->v_a ? m->v_b : m->v_a;
            ops_conv2d(o, in, m->vgg_cin[l], h, w, m->vgg_w[l], m->vgg_b[l],
                       m->vgg_cout[l], &g, NULL, NULL, NULL, ACT_RELU, 0, out);
            in = out;
        }
        ops_conv1x1(o, in, 0, m->proj_in[s], h * w, m->proj_w[s], m->proj_b[s],
                    m->proj_out[s], m->P[s][k], 0, ACT_NONE, 0);
        if (blk < 3) {
            out = in == m->v_a ? m->v_b : m->v_a;
            ops_pool2d(o, in, vgg_width[blk], h, w, 2, 2, 0, 1, out);
            in = out;
            h /= 2;
            w /= 2;
        }
    }
}

/* Cholesky solve of (K + noise I) X = F in place on F (n x d), K the
 * n x n kernel matrix (float, symmetric), in double precision. */
static void gp_solve(const float *K, int n, float *F, int d)
{
    double *L = G_malloc((size_t)n * n * sizeof(double));
    double *Z = G_malloc((size_t)n * d * sizeof(double));
    int i, j;

    for (j = 0; j < n; j++) {
        double s = (double)K[(size_t)j * n + j] + GP_NOISE;
        const double *lj = L + (size_t)j * n;
        int k;

        for (k = 0; k < j; k++)
            s -= lj[k] * lj[k];
        if (!(s > 0))
            G_fatal_error(_("RoMa: Gaussian process kernel not positive "
                            "definite"));
        L[(size_t)j * n + j] = sqrt(s);
#pragma omp parallel for schedule(static)
        for (i = j + 1; i < n; i++) {
            const double *li = L + (size_t)i * n;
            double t = K[(size_t)i * n + j];
            int q;

            for (q = 0; q < j; q++)
                t -= li[q] * lj[q];
            L[(size_t)i * n + j] = t / lj[j];
        }
    }
    /* L Z = F, then L^T X = Z, by rows, vectorized over the d columns. */
    for (i = 0; i < n; i++) {
        double *z = Z + (size_t)i * d;
        const double *li = L + (size_t)i * n;
        int q, c;

        for (c = 0; c < d; c++)
            z[c] = F[(size_t)i * d + c];
        for (q = 0; q < i; q++) {
            const double *zq = Z + (size_t)q * d;
            double a = li[q];

            for (c = 0; c < d; c++)
                z[c] -= a * zq[c];
        }
        for (c = 0; c < d; c++)
            z[c] /= li[i];
    }
    for (i = n - 1; i >= 0; i--) {
        double *z = Z + (size_t)i * d;
        int q, c;

        for (q = i + 1; q < n; q++) {
            const double *zq = Z + (size_t)q * d;
            double a = L[(size_t)q * n + i];

            for (c = 0; c < d; c++)
                z[c] -= a * zq[c];
        }
        for (c = 0; c < d; c++) {
            z[c] /= L[(size_t)i * n + i];
            F[(size_t)i * d + c] = (float)z[c];
        }
    }
    G_free(L);
    G_free(Z);
}

/* CosKernel of token sets A (N x d) and B (M x d) into S (N x M). */
static void cos_kernel(struct roma_model *m, cl_mem A, cl_mem na, cl_mem B,
                       cl_mem nb, int N, int M, cl_mem S)
{
    cl_int cN = N, cM = M;
    cl_float T = GP_T, eps = COS_EPS;
    size_t g[2] = {(size_t)M, (size_t)N};

    ops_gemm(m->ops, N, M, GP_DIM, ops_mat(A, 0, GP_DIM), ops_mat(B, 0, GP_DIM),
             1, ops_mat(S, 0, M), 1, 1, NULL, 0, NULL, 0, 1.0f, ACT_NONE, 0);
    SETARG(m->k_cos, 0, S);
    SETARG(m->k_cos, 1, na);
    SETARG(m->k_cos, 2, nb);
    SETARG(m->k_cos, 3, cN);
    SETARG(m->k_cos, 4, cM);
    SETARG(m->k_cos, 5, T);
    SETARG(m->k_cos, 6, eps);
    run(m, m->k_cos, 2, g, NULL, "roma_cos_exp");
}

static void row_norm(struct roma_model *m, cl_mem X, int rows, cl_mem out)
{
    cl_int cr = rows, cc = GP_DIM;
    size_t g = (size_t)rows;

    SETARG(m->k_norm, 0, X);
    SETARG(m->k_norm, 1, cr);
    SETARG(m->k_norm, 2, cc);
    SETARG(m->k_norm, 3, out);
    run(m, m->k_norm, 1, &g, NULL, "roma_row_norm");
}

/* Gaussian process posterior of A's 1/16 features given B's and the
 * position basis of B's grid, into t_x columns 0..511, A's features in
 * columns 512..1023: the coordinate decoder's input tokens. */
static void gp(struct roma_model *m)
{
    struct ops *o = m->ops;
    float *K = G_malloc((size_t)N16 * N16 * sizeof(float));
    float *F = G_malloc((size_t)N16 * GP_DIM * sizeof(float));

    ops_transpose(o, m->P[0][0], m->g_y[0], 1, GP_DIM, N16);
    ops_transpose(o, m->P[0][1], m->g_y[1], 1, GP_DIM, N16);
    row_norm(m, m->g_y[0], N16, m->g_nx);
    row_norm(m, m->g_y[1], N16, m->g_ny);
    cos_kernel(m, m->g_y[1], m->g_ny, m->g_y[1], m->g_ny, N16, N16, m->g_kyy);
    cos_kernel(m, m->g_y[0], m->g_nx, m->g_y[1], m->g_ny, N16, N16, m->g_kxy);
    ocl_read(m->ocl, m->g_kyy, 0, K, (size_t)N16 * N16 * sizeof(float));
    memcpy(F, m->gp_basis, (size_t)N16 * GP_DIM * sizeof(float));
    gp_solve(K, N16, F, GP_DIM);
    ocl_write(m->ocl, m->g_p, 0, F, (size_t)N16 * GP_DIM * sizeof(float));
    ops_gemm(o, N16, GP_DIM, N16, ops_mat(m->g_kxy, 0, N16),
             ops_mat(m->g_p, 0, GP_DIM), 0, ops_mat(m->g_mu, 0, GP_DIM), 1, 1,
             NULL, 0, NULL, 0, 1.0f, ACT_NONE, 0);
    copy_rect(m, m->g_mu, m->t_x, N16, GP_DIM, GP_DIM, ROMA_DEC_DIM, 0);
    copy_rect(m, m->g_y[0], m->t_x, N16, GP_DIM, GP_DIM, ROMA_DEC_DIM, GP_DIM);
    G_free(K);
    G_free(F);
}

/* Coordinate decoder on t_x: the coarse warp (anchor classes) and
 * certainty logits into fl[0], ce[0] (1/16 grid). */
static void decoder(struct roma_model *m)
{
    struct ops *o = m->ops;
    cl_int ld = NCLS + 1, C = NCLS, res = ROMA_CLS_RES, n = N16;
    size_t g = (size_t)N16 * 256, l = 256;
    int i;

    for (i = 0; i < ROMA_DEC_DEPTH; i++)
        block(m, &m->dec[i], N16, ROMA_DEC_DIM, DEC_HEADS, EPS_LN_DEC);
    ops_linear(o, N16, ROMA_DEC_DIM, NCLS + 1, m->t_x, 0, m->to_out_w,
               m->to_out_b, m->t_out, 0, ACT_NONE, 0, 0);
    SETARG(m->k_cls, 0, m->t_out);
    SETARG(m->k_cls, 1, ld);
    SETARG(m->k_cls, 2, C);
    SETARG(m->k_cls, 3, res);
    SETARG(m->k_cls, 4, n);
    SETARG(m->k_cls, 5, m->fl[0]);
    SETARG(m->k_cls, 6, m->ce[0]);
    run(m, m->k_cls, 1, &g, &l, "roma_cls_to_flow");
}

/* ConvRefiner of scale index s on an hs x ws grid of a pass of size
 * Hf x Wf (scale factor sf): updates the warp fl and certainty ce. */
static void refine(struct roma_model *m, int s, int hs, int ws, int Hf, int Wf,
                   float sf, cl_mem fl, cl_mem ce)
{
    struct ops *o = m->ops;
    const struct roma_refiner *r = &m->ref[s];
    cl_int cC = r->feat, cH = hs, cW = ws, coff, cr = r->radius, cn;
    cl_float fac = 40.0f / 32.0f * sf, scale = (float)r->scale;
    cl_float cWf = (float)Wf, cHf = (float)Hf;
    size_t n = (size_t)hs * ws, g2[2] = {n, (size_t)r->feat}, g1 = n;
    int b;

    cn = (cl_int)n;
    /* d = (x, x_hat, displacement embedding, local correlation). */
    copy(m, m->P[s][0], 0, m->d_a, 0, (size_t)r->feat * n);
    coff = r->feat;
    SETARG(m->k_warp, 0, m->P[s][1]);
    SETARG(m->k_warp, 1, cC);
    SETARG(m->k_warp, 2, cH);
    SETARG(m->k_warp, 3, cW);
    SETARG(m->k_warp, 4, fl);
    SETARG(m->k_warp, 5, m->d_a);
    SETARG(m->k_warp, 6, coff);
    run(m, m->k_warp, 2, g2, NULL, "roma_warp_sample");
    SETARG(m->k_disp, 0, fl);
    SETARG(m->k_disp, 1, cH);
    SETARG(m->k_disp, 2, cW);
    SETARG(m->k_disp, 3, fac);
    SETARG(m->k_disp, 4, m->d_c);
    run(m, m->k_disp, 1, &g1, NULL, "roma_disp");
    ops_conv1x1(o, m->d_c, 0, 2, (int)n, r->emb_w, r->emb_b, r->emb, m->d_a,
                (long)2 * r->feat * n, ACT_NONE, 0);
    if (r->radius) {
        size_t gc = n * 64, lc = 64;

        coff = 2 * r->feat + r->emb;
        SETARG(m->k_corr, 0, m->P[s][0]);
        SETARG(m->k_corr, 1, m->P[s][1]);
        SETARG(m->k_corr, 2, cC);
        SETARG(m->k_corr, 3, cH);
        SETARG(m->k_corr, 4, cW);
        SETARG(m->k_corr, 5, cr);
        SETARG(m->k_corr, 6, fl);
        SETARG(m->k_corr, 7, m->d_a);
        SETARG(m->k_corr, 8, coff);
        run(m, m->k_corr, 1, &gc, &lc, "roma_local_corr");
    }
    for (b = 0; b <= ROMA_HIDDEN; b++) {
        size_t gd[3] = {(size_t)(ws + 15) / 16 * 16,
                        (size_t)(hs + 15) / 16 * 16, (size_t)r->cin},
               ld[3] = {16, 16, 1};

        SETARG(m->k_dw, 0, m->d_a);
        SETARG(m->k_dw, 1, cH);
        SETARG(m->k_dw, 2, cW);
        SETARG(m->k_dw, 3, r->dw_w[b]);
        SETARG(m->k_dw, 4, r->dw_b[b]);
        SETARG(m->k_dw, 5, m->d_b);
        run(m, m->k_dw, 3, gd, ld, "roma_dwconv5");
        ops_conv1x1(o, m->d_b, 0, r->cin, (int)n, r->pw_w[b], r->pw_b[b],
                    r->cin, m->d_a, 0, ACT_NONE, 0);
    }
    ops_conv1x1(o, m->d_a, 0, r->cin, (int)n, r->out_w, r->out_b, 3, m->d_c, 0,
                ACT_NONE, 0);
    SETARG(m->k_update, 0, fl);
    SETARG(m->k_update, 1, ce);
    SETARG(m->k_update, 2, m->d_c);
    SETARG(m->k_update, 3, cn);
    SETARG(m->k_update, 4, scale);
    SETARG(m->k_update, 5, cWf);
    SETARG(m->k_update, 6, cHf);
    run(m, m->k_update, 1, &g1, NULL, "roma_update");
}

/* Pillow's antialiased bicubic resampling (a = -0.5, support 2 scaled
 * by the reduction) of one plane, along rows then columns. */
static double pil_cubic(double x)
{
    const double a = -0.5;

    x = fabs(x);
    if (x < 1)
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

static void resize_axis(const float *in, int n_in, long in_stride, float *out,
                        int n_out, long out_stride, int count, long in_step,
                        long out_step)
{
    double scale = (double)n_in / n_out, fs = scale > 1 ? scale : 1;
    double support = 2 * fs;
    int o, c;

    for (o = 0; o < n_out; o++) {
        double center = (o + 0.5) * scale, ww = 0, k[64];
        int xmin = (int)(center - support + 0.5), xmax, x;

        if (xmin < 0)
            xmin = 0;
        xmax = (int)(center + support + 0.5);
        if (xmax > n_in)
            xmax = n_in;
        xmax -= xmin;
        if (xmax > 64)
            G_fatal_error(_("RoMa: resampling ratio too large"));
        for (x = 0; x < xmax; x++) {
            k[x] = pil_cubic((x + xmin - center + 0.5) / fs);
            ww += k[x];
        }
        for (x = 0; x < xmax; x++)
            k[x] = ww != 0 ? k[x] / ww : 0;
        for (c = 0; c < count; c++) {
            const float *src = in + c * in_step + (long)xmin * in_stride;
            double s = 0;

            for (x = 0; x < xmax; x++)
                s += k[x] * src[x * in_stride];
            out[c * out_step + o * out_stride] = (float)s;
        }
    }
}

/* (3, H, W) in 0..1 resized to (3, h, w), normalized with the ImageNet
 * mean and deviation (RoMa's transforms) when imagenet is set. */
static void prepare(const float *in, int H, int W, float *out, int h, int w,
                    int imagenet)
{
    size_t p = (size_t)h * w;
    int c;

    if (H == h && W == w)
        memcpy(out, in, 3 * p * sizeof(float));
    else {
        float *tmp = G_malloc((size_t)H * w * sizeof(float));

        for (c = 0; c < 3; c++) {
            const float *src = in + (size_t)c * H * W;

            resize_axis(src, W, 1, tmp, w, 1, H, W, w);
            resize_axis(tmp, H, w, out + c * p, h, w, w, 1, 1);
        }
        G_free(tmp);
    }
    if (!imagenet)
        return;
    for (c = 0; c < 3; c++) {
        size_t i;

        for (i = 0; i < p; i++)
            out[c * p + i] = (out[c * p + i] - im_mean[c]) / im_std[c];
    }
}

static void dump_dev(struct roma_model *m, const char *dir, int tile,
                     const char *what, cl_mem b, size_t floats)
{
    char path[GPATH_MAX];
    float *v = G_malloc(floats * sizeof(float));
    FILE *fp;

    ocl_read(m->ocl, b, 0, v, floats * sizeof(float));
    snprintf(path, sizeof(path), "%s/tile%03d_roma_%s.f32", dir, tile, what);
    if (!(fp = fopen(path, "wb")) ||
        fwrite(v, sizeof(float), floats, fp) != floats)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
    G_free(v);
}

static void dump_host(const char *dir, int tile, const char *what,
                      const float *v, size_t floats)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/tile%03d_roma_%s.f32", dir, tile, what);
    if (!(fp = fopen(path, "wb")) ||
        fwrite(v, sizeof(float), floats, fp) != floats)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
}

/* Warp and certainty after refiner s of pass p, for the dumps. */
static void dump_scale(struct roma_model *m, const char *dir, int tile, int p,
                       int s, cl_mem fl, cl_mem ce, size_t n)
{
    char what[64];

    snprintf(what, sizeof(what), "p%d_s%d_flow", p, ref_scale[s]);
    dump_dev(m, dir, tile, what, fl, 2 * n);
    snprintf(what, sizeof(what), "p%d_s%d_cert", p, ref_scale[s]);
    dump_dev(m, dir, tile, what, ce, n);
}

/* The refiners from scale index s0 to the finest of a pass of size Hp x
 * Wp, the warp starting in fl[*cur]; resized between scales. */
static void refiners(struct roma_model *m, int p, int s0, int Hp, int Wp,
                     float sf, int *cur, const char *dump, int tile)
{
    int s;

    for (s = s0; s < ROMA_SCALES; s++) {
        int hs = side(Hp, s), ws = side(Wp, s);

        refine(m, s, hs, ws, Hp, Wp, sf, m->fl[*cur], m->ce[*cur]);
        if (p == 0 && s == 0)
            copy(m, m->ce[*cur], 0, m->cert16, 0, (size_t)hs * ws);
        if (dump)
            dump_scale(m, dump, tile, p, s, m->fl[*cur], m->ce[*cur],
                       (size_t)hs * ws);
        if (s < ROMA_SCALES - 1) {
            int hn = side(Hp, s + 1), wn = side(Wp, s + 1);

            ops_resize(m->ops, m->fl[*cur], 2, hs, ws, m->fl[!*cur], hn, wn, 0,
                       0, 0.0f);
            ops_resize(m->ops, m->ce[*cur], 1, hs, ws, m->ce[!*cur], hn, wn, 0,
                       0, 0.0f);
            *cur = !*cur;
        }
    }
}

void roma_match(struct roma_model *m, const float *imgA, const float *imgB,
                const char *dump, int tile)
{
    size_t hw = (size_t)m->H * m->W, i;
    size_t pc = 3 * (size_t)ROMA_COARSE * ROMA_COARSE;
    float sf = sqrtf((float)m->H * m->W) / ROMA_COARSE, *low;
    int k, cur = 0;

    /* Coarse pass. */
    for (k = 0; k < 2; k++) {
        prepare(k ? imgB : imgA, m->H, m->W, m->in_c[k], ROMA_COARSE,
                ROMA_COARSE, m->imagenet);
        ocl_write(m->ocl, m->img, 0, m->in_c[k], pc * sizeof(float));
        vgg(m, ROMA_COARSE, ROMA_COARSE, k);
        dino(m, k, m->d_c);
        if (dump) {
            dump_host(dump, tile, k ? "in_c_1" : "in_c_0", m->in_c[k], pc);
            dump_dev(m, dump, tile, k ? "dino_1" : "dino_0", m->d_c,
                     (size_t)ROMA_VIT_DIM * N16);
        }
    }
    gp(m);
    if (dump)
        dump_dev(m, dump, tile, "gp", m->g_mu, (size_t)N16 * GP_DIM);
    decoder(m);
    if (dump) {
        dump_dev(m, dump, tile, "cls_flow", m->fl[0], 2 * N16);
        dump_dev(m, dump, tile, "cls_cert", m->ce[0], N16);
    }
    refiners(m, 0, 0, ROMA_COARSE, ROMA_COARSE, 1.0f, &cur, dump, tile);

    /* Fine pass at the tile's size, from the coarse pass's finest warp. */
    for (k = 0; k < 2; k++) {
        prepare(k ? imgB : imgA, m->H, m->W, m->in_f[k], m->H, m->W,
                m->imagenet);
        ocl_write(m->ocl, m->img, 0, m->in_f[k], 3 * hw * sizeof(float));
        vgg(m, m->H, m->W, k);
        if (dump)
            dump_host(dump, tile, k ? "in_f_1" : "in_f_0", m->in_f[k], 3 * hw);
    }
    ops_resize(m->ops, m->fl[cur], 2, ROMA_COARSE, ROMA_COARSE, m->fl[!cur],
               side(m->H, 1), side(m->W, 1), 0, 0, 0.0f);
    ops_resize(m->ops, m->ce[cur], 1, ROMA_COARSE, ROMA_COARSE, m->ce[!cur],
               side(m->H, 1), side(m->W, 1), 0, 0, 0.0f);
    cur = !cur;
    refiners(m, 1, 1, m->H, m->W, sf, &cur, dump, tile);

    /* Certainty attenuated where the coarse one is negative, as a
     * probability; none where the warp leaves image B. */
    ops_resize(m->ops, m->cert16, 1, C16, C16, m->lowres, m->H, m->W, 0, 0,
               0.0f);
    ocl_read(m->ocl, m->fl[cur], 0, m->flow, 2 * hw * sizeof(float));
    ocl_read(m->ocl, m->ce[cur], 0, m->cert, hw * sizeof(float));
    low = G_malloc(hw * sizeof(float));
    ocl_read(m->ocl, m->lowres, 0, low, hw * sizeof(float));
    for (i = 0; i < hw; i++) {
        float c = m->cert[i] - (low[i] < 0 ? 0.5f * low[i] : 0.0f);
        float *fx = m->flow + i, *fy = m->flow + hw + i;

        c = 1.0f / (1.0f + expf(-c));
        if (fabsf(*fx) > 1 || fabsf(*fy) > 1)
            c = 0;
        *fx = *fx < -1 ? -1 : *fx > 1 ? 1 : *fx;
        *fy = *fy < -1 ? -1 : *fy > 1 ? 1 : *fy;
        m->cert[i] = c;
    }
    G_free(low);
    if (dump) {
        dump_host(dump, tile, "flow", m->flow, 2 * hw);
        dump_host(dump, tile, "cert", m->cert, hw);
    }
}

static void release_block(struct roma_block *b)
{
    cl_mem *all[] = {&b->n1g,    &b->n1b,    &b->qkv_w, &b->qkv_b,
                     &b->proj_w, &b->proj_b, &b->n2g,   &b->n2b,
                     &b->fc1_w,  &b->fc1_b,  &b->fc2_w, &b->fc2_b};
    int i;

    for (i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
        ocl_release(all[i]);
}

void roma_free(struct roma_model *m)
{
    cl_mem *all[] = {&m->patch_w,  &m->patch_b, &m->cls,    &m->pos,
                     &m->norm_g,   &m->norm_b,  &m->gp_f,   &m->to_out_w,
                     &m->to_out_b, &m->img,     &m->v_a,    &m->v_b,
                     &m->t_x,      &m->t_n,     &m->t_qkv,  &m->t_o,
                     &m->t_h,      &m->t_out,   &m->t_s,    &m->g_y[0],
                     &m->g_y[1],   &m->g_nx,    &m->g_ny,   &m->g_kyy,
                     &m->g_kxy,    &m->g_p,     &m->g_mu,   &m->d_a,
                     &m->d_b,      &m->d_c,     &m->fl[0],  &m->fl[1],
                     &m->ce[0],    &m->ce[1],   &m->cert16, &m->lowres};
    cl_kernel ks[] = {m->k_norm, m->k_cos,  m->k_cls,    m->k_warp,
                      m->k_corr, m->k_disp, m->k_update, m->k_dw};
    int i, s, b;

    for (i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
        ocl_release(all[i]);
    for (i = 0; i < 16; i++) {
        ocl_release(&m->vgg_w[i]);
        ocl_release(&m->vgg_b[i]);
    }
    for (i = 0; i < ROMA_VIT_DEPTH; i++)
        release_block(&m->vit[i]);
    for (i = 0; i < ROMA_DEC_DEPTH; i++)
        release_block(&m->dec[i]);
    for (s = 0; s < ROMA_SCALES; s++) {
        struct roma_refiner *r = &m->ref[s];

        ocl_release(&m->proj_w[s]);
        ocl_release(&m->proj_b[s]);
        ocl_release(&m->P[s][0]);
        ocl_release(&m->P[s][1]);
        for (b = 0; b <= ROMA_HIDDEN; b++) {
            ocl_release(&r->dw_w[b]);
            ocl_release(&r->dw_b[b]);
            ocl_release(&r->pw_w[b]);
            ocl_release(&r->pw_b[b]);
        }
        ocl_release(&r->out_w);
        ocl_release(&r->out_b);
        ocl_release(&r->emb_w);
        ocl_release(&r->emb_b);
    }
    for (i = 0; i < (int)(sizeof(ks) / sizeof(ks[0])); i++)
        if (ks[i])
            clReleaseKernel(ks[i]);
    G_free(m->gp_basis);
    G_free(m->flow);
    G_free(m->cert);
    for (i = 0; i < 2; i++) {
        G_free(m->in_c[i]);
        G_free(m->in_f[i]);
    }
    memset(m, 0, sizeof(*m));
}
