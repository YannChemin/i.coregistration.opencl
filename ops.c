/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Host side of the generic tensor kernels (ops.cl):
 *               argument setting, launch geometry, profiling, and the
 *               host helpers that go with them (RoPE tables, BatchNorm
 *               folding). The GEMM and LayerNorm wrappers follow
 *               i.sam3.opencl (same author, Unlicense).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "ops.h"

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

static struct {
    const char *what;
    double ms;
    long calls;
} prof[64];
static int nprof;

void ops_run(struct ocl_backend *ocl, cl_kernel k, int dims, const size_t *g,
             const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    int i;

    ocl_check(clEnqueueNDRangeKernel(ocl->queue, k, dims, NULL, g, l, 0, NULL,
                                     ocl->profile ? &ev : NULL),
              what);
    if (!ocl->profile)
        return;
    clWaitForEvents(1, &ev);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;
    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].calls++;
    }
}

void ops_report(void)
{
    int i;

    for (i = 0; i < nprof; i++)
        G_message("%-20s %10.1f ms %8ld calls", prof[i].what, prof[i].ms,
                  prof[i].calls);
}

static void run1d(struct ops *o, cl_kernel k, size_t n, const char *what)
{
    size_t l = 256, g = (n + l - 1) / l * l;

    ops_run(o->ocl, k, 1, &g, &l, what);
}

static size_t round_up(size_t v, size_t m)
{
    return (v + m - 1) / m * m;
}

/* Element counts passed as int to the kernels. */
static cl_int count(long n, const char *what)
{
    if (n < 0 || n > 0x7fffffffL)
        G_fatal_error(_("OpenCL: %s of %ld elements exceeds 2^31"), what, n);
    return (cl_int)n;
}

void ops_init(struct ops *o, struct ocl_backend *ocl)
{
    memset(o, 0, sizeof(*o));
    o->ocl = ocl;
    o->gemm = ocl_kernel(ocl, "gemm");
    o->gemm_big = ocl_kernel(ocl, "gemm_big");
    o->conv5x5 = ocl_kernel(ocl, "conv5x5");
    o->conv2d = ocl_kernel(ocl, "conv2d");
    o->chan_stats = ocl_kernel(ocl, "chan_stats");
    o->chan_affine = ocl_kernel(ocl, "chan_affine");
    o->prelu = ocl_kernel(ocl, "prelu");
    o->act_map = ocl_kernel(ocl, "act_map");
    o->layernorm = ocl_kernel(ocl, "layernorm");
    o->softmax_rows = ocl_kernel(ocl, "softmax_rows");
    o->attention = ocl_kernel(ocl, "attention");
    o->rope = ocl_kernel(ocl, "rope");
    o->resize_bilinear = ocl_kernel(ocl, "resize_bilinear");
    o->grid_sample = ocl_kernel(ocl, "grid_sample");
    o->unfold = ocl_kernel(ocl, "unfold");
    o->avgpool2 = ocl_kernel(ocl, "avgpool2");
    o->pool2d = ocl_kernel(ocl, "pool2d");
    o->transpose2d = ocl_kernel(ocl, "transpose2d");
    o->axpby = ocl_kernel(ocl, "axpby");
}

void ops_free(struct ops *o)
{
    cl_kernel *k = &o->gemm, *end = &o->axpby;

    if (!o->ocl)
        return;
    for (; k <= end; k++)
        if (*k)
            clReleaseKernel(*k);
    memset(o, 0, sizeof(*o));
}

struct opnd ops_mat(cl_mem buf, long off, int ld)
{
    struct opnd m = {buf, off, ld, 0, 0};

    return m;
}

struct opnd ops_bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd m = {buf, off, ld, s0, s1};

    return m;
}

void ops_gemm(struct ops *o, int M, int N, int K, struct opnd A, struct opnd B,
              int transb, struct opnd C, int nb, int nb1, cl_mem bias,
              int biasrow, cl_mem R, long roff, float alpha, int act,
              float slope)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_int cbr = biasrow;
    cl_long croff = roff;
    cl_float calpha = alpha, cslope = slope;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    cl_kernel k = big ? o->gemm_big : o->gemm;

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, cbr);
    SETARG(k, 22, R);
    SETARG(k, 23, croff);
    SETARG(k, 24, calpha);
    SETARG(k, 25, cact);
    SETARG(k, 26, cslope);
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    ops_run(o->ocl, k, 3, g, l, big ? "gemm_big" : "gemm");
}

void ops_linear(struct ops *o, int rows, int in, int out, cl_mem X, long xoff,
                cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, float slope,
                int residual)
{
    ops_gemm(o, rows, out, in, ops_mat(X, xoff, in), ops_mat(W, 0, in), 1,
             ops_mat(Y, yoff, out), 1, 1, b, 0, residual ? Y : NULL, yoff, 1.0f,
             act, slope);
}

void ops_conv1x1(struct ops *o, cl_mem X, long xoff, int C, int HW, cl_mem W,
                 cl_mem b, int K, cl_mem Y, long yoff, int act, float slope)
{
    ops_gemm(o, K, HW, C, ops_mat(W, 0, C), ops_mat(X, xoff, HW), 0,
             ops_mat(Y, yoff, HW), 1, 1, b, 1, NULL, 0, 1.0f, act, slope);
}

struct conv_geom ops_geom(int k, int s, int p)
{
    struct conv_geom g = {k, k, s, s, p, p, 1, 1, 1};

    return g;
}

int ops_conv_out(int n, int k, int s, int p, int d)
{
    return (n + 2 * p - d * (k - 1) - 1) / s + 1;
}

void ops_conv2d(struct ops *o, cl_mem in, int C, int H, int W, cl_mem w,
                cl_mem b, int K, const struct conv_geom *g, cl_mem mean,
                cl_mem rstd, cl_mem alpha, int act, float slope, cl_mem out)
{
    cl_int Ho = ops_conv_out(H, g->kh, g->sy, g->py, g->dy);
    cl_int Wo = ops_conv_out(W, g->kw, g->sx, g->px, g->dx);
    cl_int cC = C, cH = H, cW = W, cK = K, cact = act;
    cl_float cslope = slope;
    cl_kernel k;

    if (C % g->groups || K % g->groups)
        G_fatal_error(_("OpenCL: %d -> %d channels in %d groups"), C, K,
                      g->groups);
    count((long)C * H * W, "convolution input");
    count((long)K * Ho * Wo, "convolution output");
    if (g->kh == 5 && g->kw == 5 && g->sy == 1 && g->sx == 1 && g->py == 2 &&
        g->px == 2 && g->dy == 1 && g->dx == 1 && g->groups == 1 && b &&
        act == ACT_NONE && (!mean || alpha)) {
        cl_int norm = mean != NULL;
        cl_mem a = norm ? alpha : w, mn = norm ? mean : w, rs = norm ? rstd : w;
        size_t gs[3] = {round_up(W, 16), round_up(H, 16), (size_t)(K + 7) / 8};
        size_t ls[3] = {16, 16, 1};

        k = o->conv5x5;
        SETARG(k, 0, in);
        SETARG(k, 1, cC);
        SETARG(k, 2, cH);
        SETARG(k, 3, cW);
        SETARG(k, 4, w);
        SETARG(k, 5, b);
        SETARG(k, 6, cK);
        SETARG(k, 7, mn);
        SETARG(k, 8, rs);
        SETARG(k, 9, a);
        SETARG(k, 10, norm);
        SETARG(k, 11, out);
        ops_run(o->ocl, k, 3, gs, ls, "conv5x5");
        return;
    }
    {
        cl_int v[11] = {Ho,    Wo,    g->kh, g->kw, g->sy,    g->sx,
                        g->py, g->px, g->dy, g->dx, g->groups};
        size_t gs[3] = {round_up(Wo, 16), round_up(Ho, 4), (size_t)K};
        size_t ls[3] = {16, 4, 1};
        int i;

        k = o->conv2d;
        SETARG(k, 0, in);
        SETARG(k, 1, cC);
        SETARG(k, 2, cH);
        SETARG(k, 3, cW);
        SETARG(k, 4, w);
        SETARG(k, 5, b);
        SETARG(k, 6, cK);
        for (i = 0; i < 11; i++)
            SETARG(k, 7 + i, v[i]);
        SETARG(k, 18, mean);
        SETARG(k, 19, rstd);
        SETARG(k, 20, alpha);
        SETARG(k, 21, cact);
        SETARG(k, 22, cslope);
        SETARG(k, 23, out);
        ops_run(o->ocl, k, 3, gs, ls, "conv2d");
    }
}

void ops_chan_stats(struct ops *o, cl_mem x, int C, int HW, float eps,
                    cl_mem mean, cl_mem rstd)
{
    cl_int hw = HW;
    cl_float ce = eps;
    size_t g = (size_t)C * 256, l = 256;
    cl_kernel k = o->chan_stats;

    SETARG(k, 0, x);
    SETARG(k, 1, hw);
    SETARG(k, 2, ce);
    SETARG(k, 3, mean);
    SETARG(k, 4, rstd);
    ops_run(o->ocl, k, 1, &g, &l, "chan_stats");
}

void ops_chan_affine(struct ops *o, cl_mem X, cl_mem Y, int C, int HW,
                     cl_mem mean, cl_mem rstd, cl_mem g, cl_mem b, int act,
                     float slope)
{
    cl_int hw = HW, n = count((long)C * HW, "chan_affine"), cact = act;
    cl_float cs = slope;
    cl_kernel k = o->chan_affine;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, hw);
    SETARG(k, 3, n);
    SETARG(k, 4, mean);
    SETARG(k, 5, rstd);
    SETARG(k, 6, g);
    SETARG(k, 7, b);
    SETARG(k, 8, cact);
    SETARG(k, 9, cs);
    run1d(o, k, n, "chan_affine");
}

void ops_prelu(struct ops *o, cl_mem X, cl_mem Y, int C, int HW, cl_mem alpha)
{
    cl_int hw = HW, n = count((long)C * HW, "prelu");
    cl_kernel k = o->prelu;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, hw);
    SETARG(k, 3, n);
    SETARG(k, 4, alpha);
    run1d(o, k, n, "prelu");
}

void ops_act(struct ops *o, cl_mem X, cl_mem Y, long n, int act, float slope)
{
    cl_int cn = count(n, "act"), cact = act;
    cl_float cs = slope;
    cl_kernel k = o->act_map;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, cn);
    SETARG(k, 3, cact);
    SETARG(k, 4, cs);
    run1d(o, k, cn, "act");
}

/* Work-group size of the row kernels: up to 256, a power of two. */
static size_t row_group(int L)
{
    size_t l = 32;

    while (l < 256 && (int)l < L)
        l *= 2;
    return l;
}

void ops_layernorm(struct ops *o, long rows, int C, cl_mem X, cl_mem Y,
                   cl_mem g, cl_mem b, float eps, int act, float slope)
{
    cl_kernel k = o->layernorm;
    cl_int cC = C, cact = act;
    cl_float ceps = eps, cs = slope;
    size_t ls = row_group(C), gs = rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    SETARG(k, 7, cs);
    ops_run(o->ocl, k, 1, &gs, &ls, "layernorm");
}

void ops_softmax_rows(struct ops *o, cl_mem S, long rows, int L, float scale)
{
    cl_kernel k = o->softmax_rows;
    cl_int cL = L;
    cl_float cs = scale;
    size_t ls = row_group(L), gs = rows * ls;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    SETARG(k, 2, cs);
    ops_run(o->ocl, k, 1, &gs, &ls, "softmax_rows");
}

void ops_attention(struct ops *o, struct opnd Q, struct opnd K, struct opnd V,
                   struct opnd O, int Nq, int Nk, int nh, int d, float scale,
                   cl_mem kmask)
{
    cl_kernel k = o->attention;
    cl_int cNq = Nq, cNk = Nk, cd = d;
    cl_float cs = scale;
    size_t gs[3] = {16, round_up(Nq, 16), (size_t)nh}, ls[3] = {16, 16, 1};

    if (d < 1 || d > OPS_MAX_HEAD_DIM)
        G_fatal_error(_("OpenCL: attention head dimension %d not in 1..%d"), d,
                      OPS_MAX_HEAD_DIM);
    SETARG(k, 0, Q.buf);
    SETARG(k, 1, Q.off);
    SETARG(k, 2, Q.ld);
    SETARG(k, 3, K.buf);
    SETARG(k, 4, K.off);
    SETARG(k, 5, K.ld);
    SETARG(k, 6, V.buf);
    SETARG(k, 7, V.off);
    SETARG(k, 8, V.ld);
    SETARG(k, 9, O.buf);
    SETARG(k, 10, O.off);
    SETARG(k, 11, O.ld);
    SETARG(k, 12, cNq);
    SETARG(k, 13, cNk);
    SETARG(k, 14, cd);
    SETARG(k, 15, cs);
    SETARG(k, 16, kmask);
    ops_run(o->ocl, k, 3, gs, ls, "attention");
}

void ops_rope(struct ops *o, cl_mem X, long off, int ld, long rows, int nh,
              int d, int npos, cl_mem cs, int interleaved)
{
    cl_kernel k = o->rope;
    cl_long coff = off;
    cl_int cld = ld, cnh = nh, cd = d, cnp = npos, ci = interleaved;
    cl_int n = count(rows * nh * (d / 2), "rope");

    SETARG(k, 0, X);
    SETARG(k, 1, coff);
    SETARG(k, 2, cld);
    SETARG(k, 3, cnh);
    SETARG(k, 4, cd);
    SETARG(k, 5, cnp);
    SETARG(k, 6, cs);
    SETARG(k, 7, ci);
    SETARG(k, 8, n);
    run1d(o, k, n, "rope");
}

void ops_resize(struct ops *o, cl_mem in, int C, int h, int w, cl_mem out,
                int H, int W, int coff, int align_corners, float scale)
{
    cl_kernel k = o->resize_bilinear;
    cl_int ch = h, cw = w, cH = H, cW = W, cc = coff, ac = align_corners;
    cl_float sy = scale > 0 ? 1.0f / scale : (float)h / H;
    cl_float sx = scale > 0 ? 1.0f / scale : (float)w / W;
    size_t g[3] = {(size_t)W, (size_t)H, (size_t)C};

    SETARG(k, 0, in);
    SETARG(k, 1, ch);
    SETARG(k, 2, cw);
    SETARG(k, 3, out);
    SETARG(k, 4, cH);
    SETARG(k, 5, cW);
    SETARG(k, 6, cc);
    SETARG(k, 7, ac);
    SETARG(k, 8, sy);
    SETARG(k, 9, sx);
    ops_run(o->ocl, k, 3, g, NULL, "resize_bilinear");
}

void ops_grid_sample(struct ops *o, cl_mem in, int C, int H, int W, cl_mem grid,
                     int n, int align_corners, cl_mem out)
{
    cl_kernel k = o->grid_sample;
    cl_int cC = C, cH = H, cW = W, cn = n, ac = align_corners;
    size_t g[2] = {round_up(n, 64), (size_t)C}, l[2] = {64, 1};

    SETARG(k, 0, in);
    SETARG(k, 1, cC);
    SETARG(k, 2, cH);
    SETARG(k, 3, cW);
    SETARG(k, 4, grid);
    SETARG(k, 5, cn);
    SETARG(k, 6, ac);
    SETARG(k, 7, out);
    ops_run(o->ocl, k, 2, g, l, "grid_sample");
}

void ops_unfold(struct ops *o, cl_mem in, int C, int H, int W,
                const struct conv_geom *g, cl_mem out)
{
    cl_kernel k = o->unfold;
    cl_int Ho = ops_conv_out(H, g->kh, g->sy, g->py, 1);
    cl_int Wo = ops_conv_out(W, g->kw, g->sx, g->px, 1);
    cl_int v[11] = {C, H, W, g->kh, g->kw, g->sy, g->sx, g->py, g->px, Ho, Wo};
    size_t gs[2] = {round_up((size_t)Ho * Wo, 64), (size_t)C * g->kh * g->kw},
           ls[2] = {64, 1};
    int i;

    count((long)C * g->kh * g->kw * Ho * Wo, "unfold output");
    SETARG(k, 0, in);
    for (i = 0; i < 11; i++)
        SETARG(k, 1 + i, v[i]);
    SETARG(k, 12, out);
    ops_run(o->ocl, k, 2, gs, ls, "unfold");
}

void ops_avgpool2(struct ops *o, cl_mem in, int C, int H, int W, cl_mem out)
{
    cl_kernel k = o->avgpool2;
    cl_int cH = H, cW = W;
    size_t g[3] = {(size_t)W / 2, (size_t)H / 2, (size_t)C};

    SETARG(k, 0, in);
    SETARG(k, 1, cH);
    SETARG(k, 2, cW);
    SETARG(k, 3, out);
    ops_run(o->ocl, k, 3, g, NULL, "avgpool2");
}

void ops_pool2d(struct ops *o, cl_mem in, int C, int H, int W, int ks, int s,
                int p, int max, cl_mem out)
{
    cl_kernel k = o->pool2d;
    cl_int Ho = ops_conv_out(H, ks, s, p, 1), Wo = ops_conv_out(W, ks, s, p, 1);
    cl_int cH = H, cW = W, ck = ks, cs = s, cp = p, cm = max;
    size_t g[3] = {(size_t)Wo, (size_t)Ho, (size_t)C};

    SETARG(k, 0, in);
    SETARG(k, 1, cH);
    SETARG(k, 2, cW);
    SETARG(k, 3, Ho);
    SETARG(k, 4, Wo);
    SETARG(k, 5, ck);
    SETARG(k, 6, cs);
    SETARG(k, 7, cp);
    SETARG(k, 8, cm);
    SETARG(k, 9, out);
    ops_run(o->ocl, k, 3, g, NULL, max ? "maxpool" : "avgpool");
}

void ops_transpose(struct ops *o, cl_mem in, cl_mem out, int B, int R, int C)
{
    cl_kernel k = o->transpose2d;
    cl_int cR = R, cC = C;
    size_t g[3] = {round_up(C, 16), round_up(R, 16), (size_t)B};
    size_t l[3] = {16, 16, 1};

    count((long)B * R * C, "transpose");
    SETARG(k, 0, in);
    SETARG(k, 1, out);
    SETARG(k, 2, cR);
    SETARG(k, 3, cC);
    ops_run(o->ocl, k, 3, g, l, "transpose");
}

void ops_axpby(struct ops *o, cl_mem out, cl_mem a, float alpha, cl_mem b,
               float beta, long n, long bmod)
{
    cl_kernel k = o->axpby;
    cl_int cn = count(n, "axpby"), cm = bmod > 0 ? (cl_int)bmod : 1;
    cl_float ca = alpha, cb = beta;

    SETARG(k, 0, out);
    SETARG(k, 1, a);
    SETARG(k, 2, ca);
    SETARG(k, 3, b);
    SETARG(k, 4, cb);
    SETARG(k, 5, cn);
    SETARG(k, 6, cm);
    run1d(o, k, cn, "axpby");
}

void ops_fold_bn(float *w, float *b, int K, long per, const float *gamma,
                 const float *beta, const float *mean, const float *var,
                 float eps)
{
    int k;
    long i;

    for (k = 0; k < K; k++) {
        double s = gamma[k] / sqrt((double)var[k] + eps);

        for (i = 0; i < per; i++)
            w[(size_t)k * per + i] = (float)(w[(size_t)k * per + i] * s);
        b[k] = (float)((b[k] - mean[k]) * s + beta[k]);
    }
}
