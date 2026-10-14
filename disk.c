/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      DISK local features on an OpenCL device: the U-Net of
 *               kornia.feature.DISK with the published weights, heatmap
 *               non-maxima suppression and descriptor sampling.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "disk.h"
#include "progress.h"

/* Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64],
 * up=[64, 64, 64, 129]): down blocks 0-4 then up blocks 0-3. Each block
 * but the first normalizes (InstanceNorm2d) and gates (PReLU) its input
 * before the convolution; down blocks 1-4 average-pool first, up blocks
 * upsample the bottom input and concatenate the skip connection. */
static const int layer_cin[DISK_LAYERS] = {3, 16, 32, 64, 64, 128, 128, 96, 80};
static const int layer_cout[DISK_LAYERS] = {16, 32, 64, 64,           64,
                                            64, 64, 64, DISK_DESC + 1};

static cl_mem upload_tensor(struct disk_model *m, struct pth_file *pf,
                            const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *v = pth_read_f32(pf, t);
    cl_mem buf = ocl_upload(m->ocl, v, pth_numel(t) * sizeof(float), name);

    G_free(v);
    return buf;
}

void disk_load(struct disk_model *m, struct ops *ops, struct pth_file *pf)
{
    struct ocl_backend *ocl = ops->ocl;
    char name[128];
    int l;

    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->ops = ops;
    progress_begin(PG_LOAD, pth_bytes(pf, "") / 1073741824.0);
    for (l = 0; l < DISK_LAYERS; l++) {
        long wshape[4] = {layer_cout[l], layer_cin[l], 5, 5};
        long bshape[1] = {layer_cout[l]}, ashape[1] = {layer_cin[l]};
        const char *prefix;
        int idx;

        m->cin[l] = layer_cin[l];
        m->cout[l] = layer_cout[l];
        if (l < 5) {
            prefix = "unet.path_down.%d.1.%d.%s";
            idx = l;
        }
        else {
            prefix = "unet.path_up.%d.conv.%d.%s";
            idx = l - 5;
        }
        snprintf(name, sizeof(name), prefix, idx, 3, "weight");
        m->w[l] = upload_tensor(m, pf, name, 4, wshape);
        snprintf(name, sizeof(name), prefix, idx, 3, "bias");
        m->b[l] = upload_tensor(m, pf, name, 1, bshape);
        if (l > 0) {
            snprintf(name, sizeof(name), prefix, idx, 1, "weight");
            m->alpha[l] = upload_tensor(m, pf, name, 1, ashape);
        }
    }
    progress_end();

    m->k_nms = ocl_kernel(ocl, "nms5x5");
    m->k_gather = ocl_kernel(ocl, "gather_desc");
}

void disk_alloc(struct disk_model *m, int H, int W, int nmax)
{
    size_t px = (size_t)H * W;
    int k;

    if (H % DISK_MULTIPLE || W % DISK_MULTIPLE)
        G_fatal_error(_("DISK tile %d x %d is not a multiple of %d"), H, W,
                      DISK_MULTIPLE);
    m->H = H;
    m->W = W;
    m->nmax = nmax;
    m->input = ocl_alloc(m->ocl, 3 * px * sizeof(float), "input");
    for (k = 0; k < 5; k++)
        m->f[k] =
            ocl_alloc(m->ocl, layer_cout[k] * (px >> (2 * k)) * sizeof(float),
                      "down features");
    for (k = 0; k < 3; k++)
        m->u[k] = ocl_alloc(m->ocl, 64 * (px >> (2 * (3 - k))) * sizeof(float),
                            "up features");
    m->tmp = ocl_alloc(m->ocl, 16 * (px >> 2) * sizeof(float), "pooled");
    m->cat = ocl_alloc(m->ocl, 80 * px * sizeof(float), "concatenation");
    m->out = ocl_alloc(m->ocl, (DISK_DESC + 1) * px * sizeof(float), "output");
    m->mean = ocl_alloc(m->ocl, 128 * sizeof(float), "mean");
    m->rstd = ocl_alloc(m->ocl, 128 * sizeof(float), "rstd");
    m->mask = ocl_alloc(m->ocl, px, "nms mask");
    m->xy = ocl_alloc(m->ocl, 2 * nmax * sizeof(int), "keypoints");
    m->desc = ocl_alloc(m->ocl, (size_t)nmax * DISK_DESC * sizeof(float),
                        "descriptors");
    m->heat = G_malloc(px * sizeof(float));
    m->maskh = G_malloc(px);
}

static void run(struct disk_model *m, cl_kernel k, int dims, const size_t *g,
                const size_t *l, const char *what)
{
    ops_run(m->ocl, k, dims, g, l, what);
}

/* Instance-norm statistics of a (C, H, W) tensor into m->mean, m->rstd. */
static void stats(struct disk_model *m, cl_mem x, int C, int H, int W)
{
    ops_chan_stats(m->ops, x, C, H * W, 1e-5f, m->mean, m->rstd);
}

/* Convolution of layer l on in (cin, H, W) into out (cout, H, W), with
 * the input normalized and gated by the statistics in m->mean/rstd for
 * every layer but the first. */
static void conv(struct disk_model *m, int l, cl_mem in, int H, int W,
                 cl_mem out)
{
    struct conv_geom g = ops_geom(5, 1, 2);

    if (l > 0)
        ops_conv2d(m->ops, in, m->cin[l], H, W, m->w[l], m->b[l], m->cout[l],
                   &g, m->mean, m->rstd, m->alpha[l], ACT_NONE, 0, out);
    else
        ops_conv2d(m->ops, in, m->cin[l], H, W, m->w[l], m->b[l], m->cout[l],
                   &g, NULL, NULL, NULL, ACT_NONE, 0, out);
}

static void pool(struct disk_model *m, cl_mem in, int C, int H, int W,
                 cl_mem out)
{
    ops_avgpool2(m->ops, in, C, H, W, out);
}

/* 2x bilinear upsampling (scale_factor 2) into channels coff.. of out. */
static void upsample(struct disk_model *m, cl_mem in, int C, int h, int w,
                     cl_mem out, int coff)
{
    ops_resize(m->ops, in, C, h, w, out, 2 * h, 2 * w, coff, 0, 2.0f);
}

static int cmp_desc_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;

    return (x < y) - (x > y);
}

int disk_detect(struct disk_model *m, const float *img, int H, int W, int h,
                int w, int n, float thresh, const unsigned char *valid, int *xy,
                float *score, float *desc)
{
    size_t px = (size_t)H * W;
    cl_ulong off = (cl_ulong)DISK_DESC * px;
    int lev, j, x, y, ncand = 0, nk = 0;
    float *cand, cut = -INFINITY;

    if (H > m->H || W > m->W || (size_t)H * W > (size_t)m->H * m->W)
        G_fatal_error(_("DISK tile larger than its buffers"));
    if (n > m->nmax)
        n = m->nmax;
    ocl_write(m->ocl, m->input, 0, img, 3 * px * sizeof(float));

    /* Down path. */
    conv(m, 0, m->input, H, W, m->f[0]);
    for (lev = 1; lev < 5; lev++) {
        int c = layer_cout[lev - 1], hh = H >> lev, ww = W >> lev;
        /* The pooled input of level 1 fits m->tmp; deeper levels are
         * small enough to reuse the concatenation buffer. */
        cl_mem t = lev == 1 ? m->tmp : m->cat;

        pool(m, m->f[lev - 1], c, H >> (lev - 1), W >> (lev - 1), t);
        stats(m, t, c, hh, ww);
        conv(m, lev, t, hh, ww, m->f[lev]);
    }

    /* Up path: upsample the bottom input into the first channels of the
     * concatenation, copy the skip connection after it. */
    for (j = 0; j < 4; j++) {
        int l = 5 + j, lvl = 3 - j, hh = H >> lvl, ww = W >> lvl;
        cl_mem bot = j == 0 ? m->f[4] : m->u[j - 1];
        cl_mem hor = m->f[lvl];
        int cb = 64, ch = layer_cout[lvl];
        size_t plane = (size_t)hh * ww;

        upsample(m, bot, cb, hh / 2, ww / 2, m->cat, 0);
        ocl_check(clEnqueueCopyBuffer(
                      m->ocl->queue, hor, m->cat, 0, cb * plane * sizeof(float),
                      ch * plane * sizeof(float), 0, NULL, NULL),
                  "clEnqueueCopyBuffer");
        stats(m, m->cat, cb + ch, hh, ww);
        conv(m, l, m->cat, hh, ww, j < 3 ? m->u[j] : m->out);
    }

    /* Heatmap NMS on the h x w crop. */
    {
        size_t g[2] = {(size_t)w, (size_t)h};
        cl_kernel k = m->k_nms;

        clSetKernelArg(k, 0, sizeof(cl_mem), &m->out);
        clSetKernelArg(k, 1, sizeof(cl_ulong), &off);
        clSetKernelArg(k, 2, sizeof(int), &W);
        clSetKernelArg(k, 3, sizeof(int), &h);
        clSetKernelArg(k, 4, sizeof(int), &w);
        clSetKernelArg(k, 5, sizeof(float), &thresh);
        clSetKernelArg(k, 6, sizeof(cl_mem), &m->mask);
        run(m, k, 2, g, NULL, "nms5x5");
    }
    ocl_read(m->ocl, m->mask, 0, m->maskh, (size_t)h * w);
    ocl_read(m->ocl, m->out, off * sizeof(float), m->heat,
             (size_t)h * W * sizeof(float));

    /* The n highest candidates, kept in raster order: those above the
     * (n + 1)-th highest score, as heatmap_to_keypoints does (with n or
     * fewer candidates, it takes the lowest as that threshold, which
     * drops it). */
    cand = G_malloc(((size_t)h * w + 1) * sizeof(float));
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (m->maskh[(size_t)y * w + x] &&
                (!valid || valid[(size_t)y * w + x]))
                cand[ncand++] = m->heat[(size_t)y * W + x];
    if (ncand > 0) {
        qsort(cand, ncand, sizeof(float), cmp_desc_float);
        cut = cand[ncand > n ? n : ncand - 1];
    }
    G_free(cand);
    for (y = 0; y < h && nk < n; y++)
        for (x = 0; x < w && nk < n; x++) {
            size_t i = (size_t)y * w + x;
            float v = m->heat[(size_t)y * W + x];

            if (!m->maskh[i] || (valid && !valid[i]) || !(v > cut))
                continue;
            xy[2 * nk] = x;
            xy[2 * nk + 1] = y;
            score[nk++] = v;
        }

    if (nk > 0) {
        size_t g = nk;
        int dd = DISK_DESC;
        cl_kernel k = m->k_gather;

        ocl_write(m->ocl, m->xy, 0, xy, 2 * nk * sizeof(int));
        clSetKernelArg(k, 0, sizeof(cl_mem), &m->out);
        clSetKernelArg(k, 1, sizeof(int), &H);
        clSetKernelArg(k, 2, sizeof(int), &W);
        clSetKernelArg(k, 3, sizeof(int), &dd);
        clSetKernelArg(k, 4, sizeof(cl_mem), &m->xy);
        clSetKernelArg(k, 5, sizeof(int), &nk);
        clSetKernelArg(k, 6, sizeof(cl_mem), &m->desc);
        run(m, k, 1, &g, NULL, "gather_desc");
        ocl_read(m->ocl, m->desc, 0, desc,
                 (size_t)nk * DISK_DESC * sizeof(float));
    }
    return nk;
}

void disk_free(struct disk_model *m)
{
    int k;

    for (k = 0; k < DISK_LAYERS; k++) {
        ocl_release(&m->w[k]);
        ocl_release(&m->b[k]);
        ocl_release(&m->alpha[k]);
    }
    ocl_release(&m->input);
    for (k = 0; k < 5; k++)
        ocl_release(&m->f[k]);
    for (k = 0; k < 3; k++)
        ocl_release(&m->u[k]);
    ocl_release(&m->tmp);
    ocl_release(&m->cat);
    ocl_release(&m->out);
    ocl_release(&m->mean);
    ocl_release(&m->rstd);
    ocl_release(&m->mask);
    ocl_release(&m->xy);
    ocl_release(&m->desc);
    if (m->k_nms) {
        clReleaseKernel(m->k_nms);
        clReleaseKernel(m->k_gather);
    }
    G_free(m->heat);
    G_free(m->maskh);
    memset(m, 0, sizeof(*m));
}
