/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Efficient LoFTR (Wang, He, Peng, Tan and Zhou, 2024) on an
 *               OpenCL device: RepVGG backbone, aggregated attention
 *               transformer with 2D RoPE, coarse dual-softmax matching,
 *               fine fusion and two-stage fine matching.
 *
 *               Translated from EfficientLoFTRForKeypointMatching of
 *               Hugging Face Transformers 4.57 (Apache License 2.0,
 *               Copyright 2025 The HuggingFace Team), whose names the
 *               weights use, checked against the authors' code at commit
 *               ffd4a46 (Apache License 2.0, Copyright the authors).
 *               Changes: C and OpenCL instead of PyTorch; the RepVGG
 *               branches and BatchNorms folded into single convolutions
 *               at load; the coarse score matrix processed by blocks of
 *               rows instead of whole; fine windows read in place
 *               instead of unfolded. The fine matching follows the
 *               authors' code where Transformers departs from it (scale
 *               of the first stage scores; Transformers' first stage
 *               always returns window position 0, its gather running
 *               over the wrong axis).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "eloftr.h"
#include "progress.h"

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

/* config.json of zju-community/efficientloftr. */
static const int stage_blocks[4] = {1, 2, 4, 14};
static const int stage_out[4] = {64, 64, 128, 256};
static const int stage_stride[4] = {2, 1, 2, 2};
#define HEADS 8
#define MLP 512
#define EPS_BN 1e-5f
#define EPS_LN 1e-5f
#define LEAKY 0.01f
#define COARSE_T 0.1f
#define COARSE_THR 0.2f
#define BORDER 2
#define FINE_T 10.0f
#define ROPE_BASE 10000.0
#define ROPE_DIM 128 /* head_dim x partial_rotary_factor: 32 x 4. */

static float *read_host(struct pth_file *pf, const char *name, int ndim,
                        const long *shape)
{
    return pth_read_f32(pf, pth_require(pf, name, ndim, shape));
}

static cl_mem upload(struct eloftr_model *m, const float *v, size_t n,
                     const char *what)
{
    return ocl_upload(m->ocl, v, n * sizeof(float), what);
}

static cl_mem load_dev(struct eloftr_model *m, struct pth_file *pf,
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

/* RepVGG block l: conv1 (3x3) + conv2 (1x1) + identity, each with its
 * BatchNorm, folded into one 3x3 convolution with bias. */
static void load_block(struct eloftr_model *m, struct pth_file *pf, int l,
                       int stage, int blk)
{
    int ci = m->bcin[l], co = m->bcout[l], k, c;
    long s3[4] = {co, ci, 3, 3}, s1[4] = {co, ci, 1, 1};
    char pre[160], name[200];
    float *w, *b, *w1, *b1, *wi, *bi;

    snprintf(pre, sizeof(pre), "efficientloftr.backbone.stages.%d.blocks.%d",
             stage, blk);
    snprintf(name, sizeof(name), "%s.conv1.conv.weight", pre);
    w = read_host(pf, name, 4, s3);
    b = G_calloc(co, sizeof(float));
    snprintf(name, sizeof(name), "%s.conv1.norm", pre);
    fold(pf, name, co, (long)ci * 9, w, b);

    snprintf(name, sizeof(name), "%s.conv2.conv.weight", pre);
    w1 = read_host(pf, name, 4, s1);
    b1 = G_calloc(co, sizeof(float));
    snprintf(name, sizeof(name), "%s.conv2.norm", pre);
    fold(pf, name, co, ci, w1, b1);
    for (k = 0; k < co; k++) {
        for (c = 0; c < ci; c++)
            w[((size_t)k * ci + c) * 9 + 4] += w1[(size_t)k * ci + c];
        b[k] += b1[k];
    }
    snprintf(name, sizeof(name), "%s.identity.weight", pre);
    if (pth_find(pf, name)) {
        if (ci != co || m->bstride[l] != 1)
            G_fatal_error(_("ELoFTR: unexpected identity branch in <%s>"), pre);
        wi = G_calloc((size_t)co * ci, sizeof(float));
        bi = G_calloc(co, sizeof(float));
        for (k = 0; k < co; k++)
            wi[(size_t)k * ci + k] = 1;
        snprintf(name, sizeof(name), "%s.identity", pre);
        fold(pf, name, co, ci, wi, bi);
        for (k = 0; k < co; k++) {
            for (c = 0; c < ci; c++)
                w[((size_t)k * ci + c) * 9 + 4] += wi[(size_t)k * ci + c];
            b[k] += bi[k];
        }
        G_free(wi);
        G_free(bi);
    }
    else if (ci == co && m->bstride[l] == 1)
        G_fatal_error(_("ELoFTR: identity branch missing in <%s>"), pre);
    m->bw[l] = upload(m, w, (size_t)co * ci * 9, "backbone weight");
    m->bb[l] = upload(m, b, co, "backbone bias");
    G_free(w);
    G_free(b);
    G_free(w1);
    G_free(b1);
}

static void load_attn(struct eloftr_model *m, struct pth_file *pf, int layer,
                      const char *kind, struct eloftr_attn *a)
{
    char pre[160], name[220];
    long qa[4] = {ELOFTR_DIM, 1, 4, 4}, sq[2] = {ELOFTR_DIM, ELOFTR_DIM};
    long s1[2] = {MLP, 2 * ELOFTR_DIM}, s2[2] = {ELOFTR_DIM, MLP};
    long v1[1] = {ELOFTR_DIM};

    snprintf(pre, sizeof(pre),
             "efficientloftr.local_feature_transformer.layers.%d.%s", layer,
             kind);
#define LD(field, suffix, nd, shape)                                           \
    snprintf(name, sizeof(name), "%s.%s", pre, suffix);                        \
    a->field = load_dev(m, pf, name, nd, shape)
    LD(q_agg, "aggregation.q_aggregation.weight", 4, qa);
    LD(norm_g, "aggregation.norm.weight", 1, v1);
    LD(norm_b, "aggregation.norm.bias", 1, v1);
    LD(wq, "attention.q_proj.weight", 2, sq);
    LD(wk, "attention.k_proj.weight", 2, sq);
    LD(wv, "attention.v_proj.weight", 2, sq);
    LD(wo, "attention.o_proj.weight", 2, sq);
    LD(fc1, "mlp.fc1.weight", 2, s1);
    LD(fc2, "mlp.fc2.weight", 2, s2);
    LD(mlp_g, "mlp.layer_norm.weight", 1, v1);
    LD(mlp_b, "mlp.layer_norm.bias", 1, v1);
#undef LD
}

static void load_fusion(struct eloftr_model *m, struct pth_file *pf)
{
    static const int dims[3] = {256, 128, 64};
    long so[4] = {256, 256, 1, 1};
    float *w;
    int i, k;

    /* The coarse features enter the fusion divided by sqrt(256): folded
     * into out_conv, which has no bias. */
    w = read_host(pf, "refinement_layer.out_conv.weight", 4, so);
    for (k = 0; k < 256 * 256; k++)
        w[k] /= 16.0f;
    m->f_out = upload(m, w, 256 * 256, "out_conv");
    G_free(w);
    for (i = 0; i < 2; i++) {
        int hid = dims[i + 1], inter = dims[i];
        long s1[4] = {inter, hid, 1, 1}, s2[4] = {inter, inter, 3, 3};
        long s3[4] = {hid, inter, 3, 3};
        char name[160];
        float *b = G_calloc(inter, sizeof(float));

        snprintf(name, sizeof(name),
                 "refinement_layer.out_conv_layers.%d.out_conv1.weight", i);
        m->f1_w[i] = load_dev(m, pf, name, 4, s1);
        snprintf(name, sizeof(name),
                 "refinement_layer.out_conv_layers.%d.out_conv2.weight", i);
        w = read_host(pf, name, 4, s2);
        snprintf(name, sizeof(name),
                 "refinement_layer.out_conv_layers.%d.batch_norm", i);
        fold(pf, name, inter, (long)inter * 9, w, b);
        m->f2_w[i] = upload(m, w, (size_t)inter * inter * 9, "out_conv2");
        m->f2_b[i] = upload(m, b, inter, "out_conv2 bias");
        G_free(w);
        G_free(b);
        snprintf(name, sizeof(name),
                 "refinement_layer.out_conv_layers.%d.out_conv3.weight", i);
        m->f3_w[i] = load_dev(m, pf, name, 4, s3);
    }
}

void eloftr_load(struct eloftr_model *m, struct ops *ops, struct pth_file *pf)
{
    int s, b, l = 0, cin = 1, i;

    memset(m, 0, sizeof(*m));
    m->ops = ops;
    m->ocl = ops->ocl;
    m->pos_scale = 1.0f;
    progress_begin(PG_LOAD, pth_bytes(pf, "") / 1073741824.0);
    for (s = 0; s < 4; s++)
        for (b = 0; b < stage_blocks[s]; b++, l++) {
            m->bcin[l] = cin;
            m->bcout[l] = stage_out[s];
            m->bstride[l] = b == 0 ? stage_stride[s] : 1;
            load_block(m, pf, l, s, b);
            cin = stage_out[s];
        }
    for (i = 0; i < ELOFTR_LAYERS; i++) {
        load_attn(m, pf, i, "self_attention", &m->self_attn[i]);
        load_attn(m, pf, i, "cross_attention", &m->cross_attn[i]);
    }
    load_fusion(m, pf);
    progress_end();
    m->k_row_lse = ocl_kernel(m->ocl, "eloftr_row_lse");
    m->k_col_lse = ocl_kernel(m->ocl, "eloftr_col_lse");
    m->k_conf_row = ocl_kernel(m->ocl, "eloftr_conf_row");
    m->k_conf_col = ocl_kernel(m->ocl, "eloftr_conf_col");
    m->k_fine = ocl_kernel(m->ocl, "eloftr_fine");
}

static cl_mem walloc(struct eloftr_model *m, size_t floats, const char *what)
{
    return ocl_alloc(m->ocl, floats * sizeof(float), what);
}

/* RoPE angles of the aggregated grid: position (i, j), 1-based and
 * multiplied by pos_scale, pair p of the 256 channels: i * f(p / 2) for
 * even p, j * f(p / 2) for odd p, f(k) = base^(-2k / ROPE_DIM)
 * (compute_embeddings of the reference, with rotate_half on interleaved
 * pairs; the authors' RoPEPositionEncodingSine with npe). */
static void rope_table(struct eloftr_model *m)
{
    int np = ELOFTR_DIM / 2, i, j, p;
    float *t = G_malloc((size_t)m->ah * m->aw * np * 2 * sizeof(float));

    for (i = 0; i < m->ah; i++)
        for (j = 0; j < m->aw; j++)
            for (p = 0; p < np; p++) {
                float f = (float)(1.0 / pow(ROPE_BASE,
                                            (double)(2 * (p / 2)) / ROPE_DIM));
                float a = (p % 2 ? (float)(j + 1) : (float)(i + 1)) *
                          m->pos_scale * f;
                float *q = t + (((size_t)i * m->aw + j) * np + p) * 2;

                q[0] = cosf(a);
                q[1] = sinf(a);
            }
    m->rope = upload(m, t, (size_t)m->ah * m->aw * np * 2, "rope");
    G_free(t);
}

void eloftr_alloc(struct eloftr_model *m, int H, int W)
{
    size_t hw = (size_t)H * W, n, na;
    int k;

    if (H % ELOFTR_MULTIPLE || W % ELOFTR_MULTIPLE)
        G_fatal_error(_("ELoFTR tile %d x %d is not a multiple of %d"), H, W,
                      ELOFTR_MULTIPLE);
    m->H = H;
    m->W = W;
    m->h = H / 8;
    m->w = W / 8;
    m->ah = m->h / 4;
    m->aw = m->w / 4;
    n = (size_t)m->h * m->w;
    na = (size_t)m->ah * m->aw;
    m->img = walloc(m, hw, "image");
    m->s_a = walloc(m, 16 * hw, "backbone");
    m->s_b = walloc(m, 16 * hw, "backbone");
    for (k = 0; k < 2; k++) {
        m->f1[k] = walloc(m, 16 * hw, "1/2 features");
        m->f2[k] = walloc(m, 8 * hw, "1/4 features");
        m->x[k] = walloc(m, ELOFTR_DIM * n, "1/8 features");
        m->fine[k] = walloc(m, ELOFTR_FINE_DIM * hw, "fine features");
    }
    m->t_q = walloc(m, ELOFTR_DIM * na, "aggregated");
    m->t_kv = walloc(m, ELOFTR_DIM * na, "aggregated");
    m->t_qn = walloc(m, ELOFTR_DIM * na, "tokens");
    m->t_kvn = walloc(m, ELOFTR_DIM * na, "tokens");
    m->t_Q = walloc(m, ELOFTR_DIM * na, "queries");
    m->t_K = walloc(m, ELOFTR_DIM * na, "keys");
    m->t_V = walloc(m, ELOFTR_DIM * na, "values");
    m->t_O = walloc(m, ELOFTR_DIM * na, "attention");
    m->t_O2 = walloc(m, ELOFTR_DIM * na, "attention");
    m->t_cat = walloc(m, 2 * ELOFTR_DIM * n, "concatenation");
    m->t_tok = walloc(m, 2 * ELOFTR_DIM * n, "tokens");
    m->t_h1 = walloc(m, MLP * n, "mlp");
    m->t_h2 = walloc(m, ELOFTR_DIM * n, "mlp");
    rope_table(m);
    /* Score blocks of at most 64 Mi floats. */
    m->blk_rows = (int)((size_t)64 * 1048576 / n);
    if (m->blk_rows > (int)n)
        m->blk_rows = (int)n;
    if (m->blk_rows < 1)
        m->blk_rows = 1;
    m->s_blk = walloc(m, (size_t)m->blk_rows * n, "scores");
    m->row_lse = walloc(m, n, "row lse");
    m->col_m = walloc(m, n, "column max");
    m->col_s = walloc(m, n, "column sum");
    m->row_max = walloc(m, n, "row max");
    m->col_max = walloc(m, n, "column max");
    m->row_arg = ocl_alloc(m->ocl, n * sizeof(int), "row argmax");
    m->col_arg = ocl_alloc(m->ocl, n * sizeof(int), "column argmax");
    m->fu_a = walloc(m, 32 * hw, "fusion");
    m->fu_b = walloc(m, 32 * hw, "fusion");
    m->fu_c = walloc(m, 32 * hw, "fusion");
    m->nmax = (int)n;
    m->m_idx = ocl_alloc(m->ocl, n * 4 * sizeof(int), "match cells");
    m->m_out = walloc(m, n * 4, "fine matches");
    m->mi = G_malloc(n * sizeof(int));
    m->mj = G_malloc(n * sizeof(int));
    m->mconf = G_malloc(n * sizeof(float));
    m->kp = G_malloc(n * 4 * sizeof(float));
}

static void conv3x3(struct eloftr_model *m, cl_mem in, int C, int H, int W,
                    cl_mem w, cl_mem b, int K, int stride, int act, cl_mem out)
{
    struct conv_geom g = ops_geom(3, stride, 1);

    ops_conv2d(m->ops, in, C, H, W, w, b, K, &g, NULL, NULL, NULL, act, LEAKY,
               out);
}

/* The RepVGG backbone on m->img (1 x H x W) of image k: 1/2 features
 * into f1[k], 1/4 into f2[k], 1/8 into x[k]. */
static void backbone(struct eloftr_model *m, int k)
{
    cl_mem in = m->img;
    int l = 0, s, b, Hc = m->H, Wc = m->W;

    for (s = 0; s < 4; s++)
        for (b = 0; b < stage_blocks[s]; b++, l++) {
            int st = m->bstride[l];
            int last = b == stage_blocks[s] - 1;
            cl_mem out;

            if (last && s == 1)
                out = m->f1[k];
            else if (last && s == 2)
                out = m->f2[k];
            else if (last && s == 3)
                out = m->x[k];
            else
                out = in == m->s_a ? m->s_b : m->s_a;
            conv3x3(m, in, m->bcin[l], Hc, Wc, m->bw[l], m->bb[l], m->bcout[l],
                    st, ACT_RELU, out);
            Hc /= st;
            Wc /= st;
            in = out;
        }
}

static void copy(struct eloftr_model *m, cl_mem src, size_t soff, cl_mem dst,
                 size_t doff, size_t floats)
{
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, src, dst, soff * sizeof(float),
                                  doff * sizeof(float), floats * sizeof(float),
                                  0, NULL, NULL),
              "clEnqueueCopyBuffer");
}

/* One aggregated attention block: X (256 x h x w) attends to Y, X
 * updated in place (EfficientLoFTRAggregatedAttention). */
static void agg_attn(struct eloftr_model *m, const struct eloftr_attn *a,
                     cl_mem X, cl_mem Y, int use_rope)
{
    struct ops *o = m->ops;
    struct conv_geom g = {4, 4, 4, 4, 0, 0, 1, 1, ELOFTR_DIM};
    int D = ELOFTR_DIM, n = m->h * m->w, na = m->ah * m->aw;
    int hd = D / HEADS;

    ops_conv2d(o, X, D, m->h, m->w, a->q_agg, NULL, D, &g, NULL, NULL, NULL,
               ACT_NONE, 0, m->t_q);
    ops_pool2d(o, Y, D, m->h, m->w, 4, 4, 0, 1, m->t_kv);
    ops_transpose(o, m->t_q, m->t_qn, 1, D, na);
    ops_transpose(o, m->t_kv, m->t_kvn, 1, D, na);
    ops_layernorm(o, na, D, m->t_qn, m->t_qn, a->norm_g, a->norm_b, EPS_LN,
                  ACT_NONE, 0);
    ops_layernorm(o, na, D, m->t_kvn, m->t_kvn, a->norm_g, a->norm_b, EPS_LN,
                  ACT_NONE, 0);
    ops_linear(o, na, D, D, m->t_qn, 0, a->wq, NULL, m->t_Q, 0, ACT_NONE, 0, 0);
    ops_linear(o, na, D, D, m->t_kvn, 0, a->wk, NULL, m->t_K, 0, ACT_NONE, 0,
               0);
    ops_linear(o, na, D, D, m->t_kvn, 0, a->wv, NULL, m->t_V, 0, ACT_NONE, 0,
               0);
    if (use_rope) {
        ops_rope(o, m->t_Q, 0, D, na, 1, D, na, m->rope, 1);
        ops_rope(o, m->t_K, 0, D, na, 1, D, na, m->rope, 1);
    }
    ops_attention(o, ops_mat(m->t_Q, 0, D), ops_mat(m->t_K, 0, D),
                  ops_mat(m->t_V, 0, D), ops_mat(m->t_O, 0, D), na, na, HEADS,
                  hd, 1.0f / sqrtf((float)hd), NULL);
    ops_linear(o, na, D, D, m->t_O, 0, a->wo, NULL, m->t_Q, 0, ACT_NONE, 0, 0);
    ops_transpose(o, m->t_Q, m->t_O2, 1, na, D);
    copy(m, X, 0, m->t_cat, 0, (size_t)D * n);
    ops_resize(o, m->t_O2, D, m->ah, m->aw, m->t_cat, m->h, m->w, D, 0, 4.0f);
    ops_transpose(o, m->t_cat, m->t_tok, 1, 2 * D, n);
    ops_linear(o, n, 2 * D, MLP, m->t_tok, 0, a->fc1, NULL, m->t_h1, 0,
               ACT_LEAKY, LEAKY, 0);
    ops_linear(o, n, MLP, D, m->t_h1, 0, a->fc2, NULL, m->t_h2, 0, ACT_NONE, 0,
               0);
    ops_layernorm(o, n, D, m->t_h2, m->t_h2, a->mlp_g, a->mlp_b, EPS_LN,
                  ACT_NONE, 0);
    ops_transpose(o, m->t_h2, m->t_cat, 1, n, D);
    ops_axpby(o, X, X, 1.0f, m->t_cat, 1.0f, (long)D * n, (long)D * n);
}

static void transformer(struct eloftr_model *m)
{
    int l;

    for (l = 0; l < ELOFTR_LAYERS; l++) {
        agg_attn(m, &m->self_attn[l], m->x[0], m->x[0], 1);
        agg_attn(m, &m->self_attn[l], m->x[1], m->x[1], 1);
        /* As the reference (and the original): image 1 attends to image
         * 0's features already updated by this layer's cross attention. */
        agg_attn(m, &m->cross_attn[l], m->x[0], m->x[1], 0);
        agg_attn(m, &m->cross_attn[l], m->x[1], m->x[0], 0);
    }
}

static void run(struct eloftr_model *m, cl_kernel k, int dims, const size_t *g,
                const size_t *l, const char *what)
{
    ops_run(m->ocl, k, dims, g, l, what);
}

/* Coarse dual-softmax matching by blocks of rows, then mutual maxima
 * above the threshold away from the border. */
static void coarse_match(struct eloftr_model *m)
{
    struct ops *o = m->ops;
    int D = ELOFTR_DIM, n = m->h * m->w, R = m->blk_rows, r0, pass, i;
    cl_int cn = n, cr0, cR;
    float alpha = 1.0f / (D * COARSE_T), *buf = G_malloc(n * sizeof(float));
    float *rmax = G_malloc(n * sizeof(float));
    int *rarg = G_malloc(n * sizeof(int)), *carg = G_malloc(n * sizeof(int));

    /* Token-major features of both images in t_tok. */
    ops_transpose(o, m->x[0], m->t_tok, 1, D, n);
    ops_transpose(o, m->x[1], m->t_h1, 1, D, n);
    for (i = 0; i < n; i++)
        buf[i] = -INFINITY;
    ocl_write(m->ocl, m->col_m, 0, buf, n * sizeof(float));
    for (i = 0; i < n; i++)
        buf[i] = 0;
    ocl_write(m->ocl, m->col_s, 0, buf, n * sizeof(float));
    for (i = 0; i < n; i++)
        buf[i] = -1;
    ocl_write(m->ocl, m->col_max, 0, buf, n * sizeof(float));
    for (i = 0; i < n; i++)
        carg[i] = -1;
    ocl_write(m->ocl, m->col_arg, 0, carg, n * sizeof(int));

    for (pass = 0; pass < 2; pass++)
        for (r0 = 0; r0 < n; r0 += R) {
            int rows = n - r0 < R ? n - r0 : R;
            size_t gr = (size_t)rows * 256, lr = 256;
            size_t gc = (size_t)(n + 63) / 64 * 64, lc = 64;
            cl_kernel k;

            ops_gemm(o, rows, n, D, ops_mat(m->t_tok, (long)r0 * D, D),
                     ops_mat(m->t_h1, 0, D), 1, ops_mat(m->s_blk, 0, n), 1, 1,
                     NULL, 0, NULL, 0, alpha, ACT_NONE, 0);
            cr0 = r0;
            cR = rows;
            if (pass == 0) {
                k = m->k_row_lse;
                SETARG(k, 0, m->s_blk);
                SETARG(k, 1, cn);
                SETARG(k, 2, cr0);
                SETARG(k, 3, m->row_lse);
                run(m, k, 1, &gr, &lr, "eloftr_row_lse");
                k = m->k_col_lse;
                SETARG(k, 0, m->s_blk);
                SETARG(k, 1, cR);
                SETARG(k, 2, cn);
                SETARG(k, 3, m->col_m);
                SETARG(k, 4, m->col_s);
                run(m, k, 1, &gc, &lc, "eloftr_col_lse");
            }
            else {
                k = m->k_conf_row;
                SETARG(k, 0, m->s_blk);
                SETARG(k, 1, cn);
                SETARG(k, 2, cr0);
                SETARG(k, 3, m->row_lse);
                SETARG(k, 4, m->col_m);
                SETARG(k, 5, m->col_s);
                SETARG(k, 6, m->row_max);
                SETARG(k, 7, m->row_arg);
                run(m, k, 1, &gr, &lr, "eloftr_conf_row");
                k = m->k_conf_col;
                SETARG(k, 0, m->s_blk);
                SETARG(k, 1, cR);
                SETARG(k, 2, cn);
                SETARG(k, 3, cr0);
                SETARG(k, 4, m->row_lse);
                SETARG(k, 5, m->col_m);
                SETARG(k, 6, m->col_s);
                SETARG(k, 7, m->col_max);
                SETARG(k, 8, m->col_arg);
                run(m, k, 1, &gc, &lc, "eloftr_conf_col");
            }
        }
    ocl_read(m->ocl, m->row_max, 0, rmax, n * sizeof(float));
    ocl_read(m->ocl, m->row_arg, 0, rarg, n * sizeof(int));
    ocl_read(m->ocl, m->col_arg, 0, carg, n * sizeof(int));
    m->nm = 0;
    for (i = 0; i < n; i++) {
        int j = rarg[i];
        int xi = i % m->w, yi = i / m->w, xj = j % m->w, yj = j / m->w;

        if (!(rmax[i] > COARSE_THR) || carg[j] != i)
            continue;
        if (xi < BORDER || yi < BORDER || xi >= m->w - BORDER ||
            yi >= m->h - BORDER || xj < BORDER || yj < BORDER ||
            xj >= m->w - BORDER || yj >= m->h - BORDER)
            continue;
        m->mi[m->nm] = i;
        m->mj[m->nm] = j;
        m->mconf[m->nm++] = rmax[i];
    }
    G_free(buf);
    G_free(rmax);
    G_free(rarg);
    G_free(carg);
}

/* Fine fusion of image k: 1/8 transformer output, 1/4 and 1/2 backbone
 * features into 64 channels at full resolution (fine[k]). */
static void fusion(struct eloftr_model *m, int k)
{
    struct ops *o = m->ops;
    int h = m->h, w = m->w;
    int H4 = 2 * h, W4 = 2 * w, H2 = 4 * h, W2 = 4 * w;

    ops_conv1x1(o, m->x[k], 0, 256, h * w, m->f_out, NULL, 256, m->fu_a, 0,
                ACT_NONE, 0);
    ops_resize(o, m->fu_a, 256, h, w, m->fu_b, H4, W4, 0, 0, 2.0f);
    /* Block 0 at 1/4: hidden 128, intermediate 256. */
    ops_conv1x1(o, m->f2[k], 0, 128, H4 * W4, m->f1_w[0], NULL, 256, m->fu_c, 0,
                ACT_NONE, 0);
    ops_axpby(o, m->fu_c, m->fu_c, 1.0f, m->fu_b, 1.0f, 256L * H4 * W4,
              256L * H4 * W4);
    conv3x3(m, m->fu_c, 256, H4, W4, m->f2_w[0], m->f2_b[0], 256, 1, ACT_LEAKY,
            m->fu_a);
    conv3x3(m, m->fu_a, 256, H4, W4, m->f3_w[0], NULL, 128, 1, ACT_NONE,
            m->fu_b);
    ops_resize(o, m->fu_b, 128, H4, W4, m->fu_c, H2, W2, 0, 0, 2.0f);
    /* Block 1 at 1/2: hidden 64, intermediate 128. */
    ops_conv1x1(o, m->f1[k], 0, 64, H2 * W2, m->f1_w[1], NULL, 128, m->fu_a, 0,
                ACT_NONE, 0);
    ops_axpby(o, m->fu_a, m->fu_a, 1.0f, m->fu_c, 1.0f, 128L * H2 * W2,
              128L * H2 * W2);
    conv3x3(m, m->fu_a, 128, H2, W2, m->f2_w[1], m->f2_b[1], 128, 1, ACT_LEAKY,
            m->fu_b);
    conv3x3(m, m->fu_b, 128, H2, W2, m->f3_w[1], NULL, 64, 1, ACT_NONE,
            m->fu_c);
    ops_resize(o, m->fu_c, 64, H2, W2, m->fine[k], m->H, m->W, 0, 0, 2.0f);
}

static void fine_match(struct eloftr_model *m)
{
    int *cells = G_malloc((m->nm + 1) * 4 * sizeof(int)), i;
    cl_int cH = m->H, cW = m->W, cM = m->nm;
    cl_float T = FINE_T;
    size_t g = (size_t)m->nm * 256, l = 256;
    cl_kernel k = m->k_fine;

    if (m->nm == 0) {
        G_free(cells);
        return;
    }
    for (i = 0; i < m->nm; i++) {
        cells[4 * i] = m->mi[i] % m->w;
        cells[4 * i + 1] = m->mi[i] / m->w;
        cells[4 * i + 2] = m->mj[i] % m->w;
        cells[4 * i + 3] = m->mj[i] / m->w;
    }
    ocl_write(m->ocl, m->m_idx, 0, cells, (size_t)m->nm * 4 * sizeof(int));
    SETARG(k, 0, m->fine[0]);
    SETARG(k, 1, m->fine[1]);
    SETARG(k, 2, cH);
    SETARG(k, 3, cW);
    SETARG(k, 4, m->m_idx);
    SETARG(k, 5, cM);
    SETARG(k, 6, T);
    SETARG(k, 7, m->m_out);
    run(m, k, 1, &g, &l, "eloftr_fine");
    ocl_read(m->ocl, m->m_out, 0, m->kp, (size_t)m->nm * 4 * sizeof(float));
    G_free(cells);
}

static void dump_dev(struct eloftr_model *m, const char *dir, int tile,
                     const char *what, cl_mem b, size_t floats)
{
    char path[GPATH_MAX];
    float *v = G_malloc(floats * sizeof(float));
    FILE *fp;

    ocl_read(m->ocl, b, 0, v, floats * sizeof(float));
    snprintf(path, sizeof(path), "%s/tile%03d_eloftr_%s.f32", dir, tile, what);
    if (!(fp = fopen(path, "wb")) ||
        fwrite(v, sizeof(float), floats, fp) != floats)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
    G_free(v);
}

void eloftr_match(struct eloftr_model *m, const float *grey0,
                  const float *grey1, const char *dump, int tile)
{
    size_t n = (size_t)m->h * m->w, hw = (size_t)m->H * m->W;
    int k;

    for (k = 0; k < 2; k++) {
        ocl_write(m->ocl, m->img, 0, k ? grey1 : grey0, hw * sizeof(float));
        backbone(m, k);
    }
    if (dump) {
        dump_dev(m, dump, tile, "backbone_0", m->x[0], ELOFTR_DIM * n);
        dump_dev(m, dump, tile, "backbone_1", m->x[1], ELOFTR_DIM * n);
    }
    transformer(m);
    if (dump) {
        dump_dev(m, dump, tile, "coarse_0", m->x[0], ELOFTR_DIM * n);
        dump_dev(m, dump, tile, "coarse_1", m->x[1], ELOFTR_DIM * n);
    }
    coarse_match(m);
    for (k = 0; k < 2; k++)
        fusion(m, k);
    if (dump) {
        dump_dev(m, dump, tile, "fine_0", m->fine[0], ELOFTR_FINE_DIM * hw);
        dump_dev(m, dump, tile, "fine_1", m->fine[1], ELOFTR_FINE_DIM * hw);
    }
    fine_match(m);
}

void eloftr_free(struct eloftr_model *m)
{
    cl_mem *all[] = {
        &m->img,     &m->s_a,     &m->s_b,     &m->f1[0], &m->f1[1],
        &m->f2[0],   &m->f2[1],   &m->x[0],    &m->x[1],  &m->fine[0],
        &m->fine[1], &m->t_q,     &m->t_kv,    &m->t_qn,  &m->t_kvn,
        &m->t_Q,     &m->t_K,     &m->t_V,     &m->t_O,   &m->t_O2,
        &m->t_cat,   &m->t_tok,   &m->t_h1,    &m->t_h2,  &m->rope,
        &m->s_blk,   &m->row_lse, &m->col_m,   &m->col_s, &m->row_max,
        &m->row_arg, &m->col_max, &m->col_arg, &m->fu_a,  &m->fu_b,
        &m->fu_c,    &m->m_idx,   &m->m_out,   &m->f_out};
    int i, l;

    for (i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
        ocl_release(all[i]);
    for (l = 0; l < ELOFTR_BLOCKS; l++) {
        ocl_release(&m->bw[l]);
        ocl_release(&m->bb[l]);
    }
    for (l = 0; l < ELOFTR_LAYERS; l++) {
        struct eloftr_attn *a[2] = {&m->self_attn[l], &m->cross_attn[l]};
        int j;

        for (j = 0; j < 2; j++) {
            ocl_release(&a[j]->q_agg);
            ocl_release(&a[j]->norm_g);
            ocl_release(&a[j]->norm_b);
            ocl_release(&a[j]->wq);
            ocl_release(&a[j]->wk);
            ocl_release(&a[j]->wv);
            ocl_release(&a[j]->wo);
            ocl_release(&a[j]->fc1);
            ocl_release(&a[j]->fc2);
            ocl_release(&a[j]->mlp_g);
            ocl_release(&a[j]->mlp_b);
        }
    }
    for (i = 0; i < 2; i++) {
        ocl_release(&m->f1_w[i]);
        ocl_release(&m->f2_w[i]);
        ocl_release(&m->f2_b[i]);
        ocl_release(&m->f3_w[i]);
    }
    if (m->k_fine) {
        clReleaseKernel(m->k_row_lse);
        clReleaseKernel(m->k_col_lse);
        clReleaseKernel(m->k_conf_row);
        clReleaseKernel(m->k_conf_col);
        clReleaseKernel(m->k_fine);
    }
    G_free(m->mi);
    G_free(m->mj);
    G_free(m->mconf);
    G_free(m->kp);
    memset(m, 0, sizeof(*m));
}
