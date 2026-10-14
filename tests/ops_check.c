/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Developer check of the generic tensor kernels (ops.cl):
 *               runs every operation on deterministic random inputs and
 *               writes inputs, parameters and outputs to a folder, which
 *               tests/ops_compare.py checks against PyTorch. Built by
 *               "make ops_check", not installed.
 *
 *               Usage: ops_check <folder> [device [platform]]
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "ocl_backend.h"
#include "ops.h"

static struct ocl_backend ocl;
static struct ops o;
static const char *dir;
static FILE *cases;
static unsigned long long rng = 88172645463325252ULL;

/* Uniform in lo..hi (xorshift64). */
static float urand(float lo, float hi)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return lo + (hi - lo) * (float)((rng >> 11) * (1.0 / 9007199254740992.0));
}

static float *rand_host(long n, float lo, float hi)
{
    float *v = G_malloc(n * sizeof(float));
    long i;

    for (i = 0; i < n; i++)
        v[i] = urand(lo, hi);
    return v;
}

static void save(const char *cname, const char *tname, const void *v,
                 size_t bytes)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/%s_%s.f32", dir, cname, tname);
    if (!(fp = fopen(path, "wb")) || fwrite(v, 1, bytes, fp) != bytes)
        G_fatal_error(_("Unable to write <%s>"), path);
    fclose(fp);
}

/* A random device tensor of n floats, also saved as <case>_<name>. */
static cl_mem input(const char *cname, const char *tname, long n, float lo,
                    float hi)
{
    float *v = rand_host(n, lo, hi);
    cl_mem m = ocl_upload(&ocl, v, n * sizeof(float), tname);

    save(cname, tname, v, n * sizeof(float));
    G_free(v);
    return m;
}

/* A zeroed device tensor of n floats. */
static cl_mem zeros(long n)
{
    float *v = G_calloc(n, sizeof(float));
    cl_mem m = ocl_upload(&ocl, v, n * sizeof(float), "zeros");

    G_free(v);
    return m;
}

static void output(const char *cname, const char *tname, cl_mem m, long n)
{
    float *v = G_malloc(n * sizeof(float));

    ocl_read(&ocl, m, 0, v, n * sizeof(float));
    save(cname, tname, v, n * sizeof(float));
    G_free(v);
}

/* One line of cases.txt: the case name, then key=value parameters. */
static void record(const char *cname, const char *fmt, ...)
{
    va_list ap;

    fprintf(cases, "%s", cname);
    if (fmt && *fmt) {
        fputc(' ', cases);
        va_start(ap, fmt);
        vfprintf(cases, fmt, ap);
        va_end(ap);
    }
    fputc('\n', cases);
}

static void case_gemm(const char *name, int M, int N, int K, int transb, int nb,
                      int act, int residual)
{
    long na = (long)nb * M * K, nbb = (long)nb * N * K, nc = (long)nb * M * N;
    cl_mem A = input(name, "a", na, -1, 1);
    cl_mem B = input(name, "b", nbb, -1, 1);
    cl_mem bias = input(name, "bias", N, -1, 1);
    cl_mem C = residual ? input(name, "r", nc, -1, 1) : zeros(nc);

    ops_gemm(&o, M, N, K, ops_bmat(A, 0, K, (long)M * K, 0),
             ops_bmat(B, 0, transb ? K : N, (long)N * K, 0), transb,
             ops_bmat(C, 0, N, (long)M * N, 0), nb, 1, bias, 0,
             residual ? C : NULL, 0, 0.5f, act, 0.1f);
    output(name, "out", C, nc);
    record(name,
           "op=gemm M=%d N=%d K=%d transb=%d nb=%d act=%d residual=%d "
           "alpha=0.5 slope=0.1",
           M, N, K, transb, nb, act, residual);
    ocl_release(&A);
    ocl_release(&B);
    ocl_release(&bias);
    ocl_release(&C);
}

static void case_conv1x1(const char *name, int C, int HW, int K)
{
    cl_mem X = input(name, "x", (long)C * HW, -1, 1);
    cl_mem W = input(name, "w", (long)K * C, -1, 1);
    cl_mem b = input(name, "bias", K, -1, 1);
    cl_mem Y = zeros((long)K * HW);

    ops_conv1x1(&o, X, 0, C, HW, W, b, K, Y, 0, ACT_RELU, 0);
    output(name, "out", Y, (long)K * HW);
    record(name, "op=conv1x1 C=%d HW=%d K=%d act=%d", C, HW, K, ACT_RELU);
    ocl_release(&X);
    ocl_release(&W);
    ocl_release(&b);
    ocl_release(&Y);
}

static void case_conv(const char *name, int C, int H, int W, int K, int k,
                      int s, int p, int d, int groups, int act, int norm)
{
    struct conv_geom g = {k, k, s, s, p, p, d, d, groups};
    int Ho = ops_conv_out(H, k, s, p, d), Wo = ops_conv_out(W, k, s, p, d);
    cl_mem X = input(name, "x", (long)C * H * W, -1, 1);
    cl_mem Wt = input(name, "w", (long)K * (C / groups) * k * k, -0.3f, 0.3f);
    cl_mem b = input(name, "bias", K, -1, 1);
    cl_mem Y = zeros((long)K * Ho * Wo);
    cl_mem mean = NULL, rstd = NULL, alpha = NULL;

    if (norm) {
        mean = input(name, "mean", C, -0.5f, 0.5f);
        rstd = input(name, "rstd", C, 0.5f, 2.0f);
        alpha = input(name, "alpha", C, 0.0f, 0.5f);
    }
    ops_conv2d(&o, X, C, H, W, Wt, b, K, &g, mean, rstd, alpha, act, 0.2f, Y);
    output(name, "out", Y, (long)K * Ho * Wo);
    record(name,
           "op=conv2d C=%d H=%d W=%d K=%d k=%d s=%d p=%d d=%d groups=%d "
           "act=%d slope=0.2 norm=%d",
           C, H, W, K, k, s, p, d, groups, act, norm);
    ocl_release(&X);
    ocl_release(&Wt);
    ocl_release(&b);
    ocl_release(&Y);
    ocl_release(&mean);
    ocl_release(&rstd);
    ocl_release(&alpha);
}

static void case_instnorm(const char *name, int C, int HW)
{
    cl_mem X = input(name, "x", (long)C * HW, -2, 3);
    cl_mem g = input(name, "g", C, 0.5f, 1.5f);
    cl_mem b = input(name, "bias", C, -1, 1);
    cl_mem mean = zeros(C), rstd = zeros(C), Y = zeros((long)C * HW);

    ops_chan_stats(&o, X, C, HW, 1e-5f, mean, rstd);
    ops_chan_affine(&o, X, Y, C, HW, mean, rstd, g, b, ACT_LEAKY, 0.01f);
    output(name, "out", Y, (long)C * HW);
    record(name, "op=instnorm C=%d HW=%d eps=1e-5 act=%d slope=0.01", C, HW,
           ACT_LEAKY);
    ocl_release(&X);
    ocl_release(&g);
    ocl_release(&b);
    ocl_release(&mean);
    ocl_release(&rstd);
    ocl_release(&Y);
}

static void case_layernorm(const char *name, int rows, int C)
{
    cl_mem X = input(name, "x", (long)rows * C, -2, 3);
    cl_mem g = input(name, "g", C, 0.5f, 1.5f);
    cl_mem b = input(name, "bias", C, -1, 1);
    cl_mem Y = zeros((long)rows * C);

    ops_layernorm(&o, rows, C, X, Y, g, b, 1e-6f, ACT_GELU, 0);
    output(name, "out", Y, (long)rows * C);
    record(name, "op=layernorm rows=%d C=%d eps=1e-6 act=%d", rows, C,
           ACT_GELU);
    ocl_release(&X);
    ocl_release(&g);
    ocl_release(&b);
    ocl_release(&Y);
}

static void case_softmax(const char *name, int rows, int L, float scale)
{
    cl_mem X = input(name, "x", (long)rows * L, -20, 20);

    ops_softmax_rows(&o, X, rows, L, scale);
    output(name, "out", X, (long)rows * L);
    record(name, "op=softmax rows=%d L=%d scale=%.9g", rows, L, scale);
    ocl_release(&X);
}

/* Attention on a fused qkv buffer (rows of 3 * nh * d: queries, keys,
 * values), queries from the first Nq rows, keys and values from all Nk
 * rows, with an optional key mask. */
static void case_attention(const char *name, int Nq, int Nk, int nh, int d,
                           int masked)
{
    int E = nh * d;
    cl_mem X = input(name, "qkv", (long)Nk * 3 * E, -2, 2);
    cl_mem O = zeros((long)Nq * E), mask = NULL;
    float scale = 1.0f / sqrtf(d);

    if (masked) {
        unsigned char *m = G_malloc(Nk);
        float *mf = G_malloc(Nk * sizeof(float));
        int i;

        for (i = 0; i < Nk; i++) {
            m[i] = urand(0, 1) > 0.3f;
            mf[i] = m[i];
        }
        mask = ocl_upload(&ocl, m, Nk, "mask");
        save(name, "mask", mf, Nk * sizeof(float));
        G_free(m);
        G_free(mf);
    }
    ops_attention(&o, ops_mat(X, 0, 3 * E), ops_mat(X, E, 3 * E),
                  ops_mat(X, 2 * E, 3 * E), ops_mat(O, 0, E), Nq, Nk, nh, d,
                  scale, mask);
    output(name, "out", O, (long)Nq * E);
    record(name, "op=attention Nq=%d Nk=%d nh=%d d=%d scale=%.9g masked=%d", Nq,
           Nk, nh, d, scale, masked);
    ocl_release(&X);
    ocl_release(&O);
    ocl_release(&mask);
}

static void case_rope(const char *name, int rows, int nh, int d, int npos,
                      int interleaved)
{
    int ld = nh * d + 6, f, i;
    cl_mem X = input(name, "x", (long)rows * ld + 3, -1, 1);
    float *ang = rand_host((long)npos * d / 2, -3.2f, 3.2f);
    float *cs = G_malloc((long)npos * d * sizeof(float));
    cl_mem T;

    for (i = 0; i < npos; i++)
        for (f = 0; f < d / 2; f++) {
            cs[(i * d / 2 + f) * 2] = cosf(ang[i * d / 2 + f]);
            cs[(i * d / 2 + f) * 2 + 1] = sinf(ang[i * d / 2 + f]);
        }
    save(name, "angles", ang, (long)npos * d / 2 * sizeof(float));
    T = ocl_upload(&ocl, cs, (long)npos * d * sizeof(float), "rope table");
    ops_rope(&o, X, 3, ld, rows, nh, d, npos, T, interleaved);
    output(name, "out", X, (long)rows * ld + 3);
    record(name,
           "op=rope rows=%d nh=%d d=%d npos=%d ld=%d off=3 "
           "interleaved=%d",
           rows, nh, d, npos, ld, interleaved);
    G_free(ang);
    G_free(cs);
    ocl_release(&X);
    ocl_release(&T);
}

static void case_resize(const char *name, int C, int h, int w, int H, int W,
                        int align, float scale)
{
    cl_mem X = input(name, "x", (long)C * h * w, -1, 1);
    cl_mem Y = zeros((long)(C + 2) * H * W);

    ops_resize(&o, X, C, h, w, Y, H, W, 2, align, scale);
    output(name, "out", Y, (long)(C + 2) * H * W);
    record(name,
           "op=resize C=%d h=%d w=%d H=%d W=%d coff=2 align=%d "
           "scale=%.9g",
           C, h, w, H, W, align, scale);
    ocl_release(&X);
    ocl_release(&Y);
}

static void case_grid_sample(const char *name, int C, int H, int W, int n,
                             int align)
{
    cl_mem X = input(name, "x", (long)C * H * W, -1, 1);
    cl_mem G = input(name, "grid", 2L * n, -1.2f, 1.2f);
    cl_mem Y = zeros((long)C * n);

    ops_grid_sample(&o, X, C, H, W, G, n, align, Y);
    output(name, "out", Y, (long)C * n);
    record(name, "op=grid_sample C=%d H=%d W=%d n=%d align=%d", C, H, W, n,
           align);
    ocl_release(&X);
    ocl_release(&G);
    ocl_release(&Y);
}

static void case_unfold(const char *name, int C, int H, int W, int k, int s,
                        int p)
{
    struct conv_geom g = ops_geom(k, s, p);
    int Ho = ops_conv_out(H, k, s, p, 1), Wo = ops_conv_out(W, k, s, p, 1);
    cl_mem X = input(name, "x", (long)C * H * W, -1, 1);
    cl_mem Y = zeros((long)C * k * k * Ho * Wo);

    ops_unfold(&o, X, C, H, W, &g, Y);
    output(name, "out", Y, (long)C * k * k * Ho * Wo);
    record(name, "op=unfold C=%d H=%d W=%d k=%d s=%d p=%d", C, H, W, k, s, p);
    ocl_release(&X);
    ocl_release(&Y);
}

static void case_pool(const char *name, int C, int H, int W, int k, int s,
                      int p, int max)
{
    int Ho = ops_conv_out(H, k, s, p, 1), Wo = ops_conv_out(W, k, s, p, 1);
    cl_mem X = input(name, "x", (long)C * H * W, -1, 1);
    cl_mem Y = zeros((long)C * Ho * Wo);

    if (k == 2 && s == 2 && p == 0 && !max)
        ops_avgpool2(&o, X, C, H, W, Y);
    else
        ops_pool2d(&o, X, C, H, W, k, s, p, max, Y);
    output(name, "out", Y, (long)C * Ho * Wo);
    record(name, "op=pool C=%d H=%d W=%d k=%d s=%d p=%d max=%d", C, H, W, k, s,
           p, max);
    ocl_release(&X);
    ocl_release(&Y);
}

static void case_transpose(const char *name, int B, int R, int C)
{
    cl_mem X = input(name, "x", (long)B * R * C, -1, 1);
    cl_mem Y = zeros((long)B * R * C);

    ops_transpose(&o, X, Y, B, R, C);
    output(name, "out", Y, (long)B * R * C);
    record(name, "op=transpose B=%d R=%d C=%d", B, R, C);
    ocl_release(&X);
    ocl_release(&Y);
}

static void case_elementwise(const char *name, int C, int HW)
{
    long n = (long)C * HW;
    cl_mem X = input(name, "x", n, -3, 3);
    cl_mem A = input(name, "alpha", C, 0, 0.5f);
    cl_mem B = input(name, "b", HW, -1, 1);
    cl_mem Y = zeros(n);
    int act;

    ops_prelu(&o, X, Y, C, HW, A);
    output(name, "prelu", Y, n);
    for (act = ACT_GELU; act <= ACT_GELU_TANH; act++) {
        char t[32];

        ops_act(&o, X, Y, n, act, 0.05f);
        snprintf(t, sizeof(t), "act%d", act);
        output(name, t, Y, n);
    }
    ops_axpby(&o, Y, X, 0.5f, B, -2.0f, n, HW);
    output(name, "axpby", Y, n);
    record(name, "op=elementwise C=%d HW=%d slope=0.05 a=0.5 b=-2", C, HW);
    ocl_release(&X);
    ocl_release(&A);
    ocl_release(&B);
    ocl_release(&Y);
}

/* BatchNorm folding (host): the convolution with the folded weights must
 * equal the convolution followed by the BatchNorm. */
static void case_fold_bn(const char *name, int C, int H, int W, int K)
{
    long per = (long)C * 9;
    float *w = rand_host(K * per, -0.3f, 0.3f), *b = rand_host(K, -1, 1);
    float *gm = rand_host(K, 0.5f, 1.5f), *bt = rand_host(K, -1, 1);
    float *mu = rand_host(K, -1, 1), *var = rand_host(K, 0.2f, 2.0f);
    struct conv_geom g = ops_geom(3, 1, 1);
    cl_mem X = input(name, "x", (long)C * H * W, -1, 1), Wd, Bd, Y;

    save(name, "w", w, K * per * sizeof(float));
    save(name, "bias", b, K * sizeof(float));
    save(name, "gamma", gm, K * sizeof(float));
    save(name, "beta", bt, K * sizeof(float));
    save(name, "mean", mu, K * sizeof(float));
    save(name, "var", var, K * sizeof(float));
    ops_fold_bn(w, b, K, per, gm, bt, mu, var, 1e-5f);
    Wd = ocl_upload(&ocl, w, K * per * sizeof(float), "w");
    Bd = ocl_upload(&ocl, b, K * sizeof(float), "b");
    Y = zeros((long)K * H * W);
    ops_conv2d(&o, X, C, H, W, Wd, Bd, K, &g, NULL, NULL, NULL, ACT_NONE, 0, Y);
    output(name, "out", Y, (long)K * H * W);
    record(name, "op=fold_bn C=%d H=%d W=%d K=%d eps=1e-5", C, H, W, K);
    G_free(w);
    G_free(b);
    G_free(gm);
    G_free(bt);
    G_free(mu);
    G_free(var);
    ocl_release(&X);
    ocl_release(&Wd);
    ocl_release(&Bd);
    ocl_release(&Y);
}

int main(int argc, char *argv[])
{
    char path[GPATH_MAX];

    G_no_gisinit();
    if (argc < 2)
        G_fatal_error(_("Usage: %s <folder> [device [platform]]"), argv[0]);
    dir = argv[1];
    snprintf(path, sizeof(path), "%s/cases.txt", dir);
    if (!(cases = fopen(path, "w")))
        G_fatal_error(_("Unable to write <%s>"), path);
    ocl_init(&ocl, argc > 2 ? argv[2] : "auto", argc > 3 ? argv[3] : NULL);
    ops_init(&o, &ocl);

    case_gemm("gemm_small_nt", 37, 29, 21, 0, 1, ACT_GELU, 0);
    case_gemm("gemm_small_t", 70, 65, 33, 1, 3, ACT_LEAKY, 1);
    case_gemm("gemm_big_t", 300, 200, 96, 1, 2, ACT_RELU, 1);
    case_gemm("gemm_big_nt", 256, 130, 64, 0, 1, ACT_NONE, 0);
    case_conv1x1("conv1x1", 48, 300, 40);
    case_conv("conv5x5_norm", 12, 37, 45, 10, 5, 1, 2, 1, 1, ACT_NONE, 1);
    case_conv("conv5x5_first", 3, 33, 20, 16, 5, 1, 2, 1, 1, ACT_NONE, 0);
    case_conv("conv3x3_s2_g2", 6, 37, 29, 8, 3, 2, 1, 1, 2, ACT_RELU, 0);
    case_conv("conv3x3_dil2", 5, 21, 26, 7, 3, 1, 2, 2, 1, ACT_LEAKY, 0);
    case_conv("conv_dw4_s4", 8, 32, 40, 8, 4, 4, 0, 1, 8, ACT_NONE, 0);
    case_conv("conv5x5_norm_relu", 6, 19, 23, 5, 5, 1, 2, 1, 1, ACT_RELU, 1);
    case_conv("conv3x3_im2col", 24, 37, 29, 20, 3, 1, 1, 1, 1, ACT_RELU, 0);
    /* Three im2col blocks of at most 128 output positions. */
    o.im2col_bytes = 16 * 9 * 128 * sizeof(float);
    case_conv("conv3x3_s2_im2col_blocks", 16, 33, 41, 12, 3, 2, 1, 1, 1,
              ACT_LEAKY, 0);
    o.im2col_bytes = (size_t)256 << 20;
    case_instnorm("instnorm", 7, 1000);
    case_layernorm("layernorm_96", 50, 96);
    case_layernorm("layernorm_1030", 9, 1030);
    case_softmax("softmax", 7, 300, 0.1f);
    case_softmax("softmax_short", 5, 20, 10.0f);
    case_attention("attention_d40", 50, 77, 3, 40, 0);
    case_attention("attention_d128_mask", 33, 300, 2, 128, 1);
    case_attention("attention_d32", 260, 260, 8, 32, 0);
    case_rope("rope_interleaved", 20, 2, 16, 10, 1);
    case_rope("rope_half", 20, 2, 16, 10, 0);
    case_resize("resize_x2", 3, 16, 24, 32, 48, 0, 2.0f);
    case_resize("resize_size", 2, 13, 17, 29, 31, 0, 0);
    case_resize("resize_align", 2, 13, 17, 29, 31, 1, 0);
    case_grid_sample("grid_sample", 3, 20, 25, 500, 0);
    case_grid_sample("grid_sample_align", 3, 20, 25, 500, 1);
    case_unfold("unfold", 4, 12, 10, 3, 2, 1);
    case_unfold("unfold_8", 2, 16, 16, 8, 8, 0);
    case_pool("avgpool2", 3, 16, 22, 2, 2, 0, 0);
    case_pool("maxpool4", 3, 16, 24, 4, 4, 0, 1);
    case_pool("avgpool3_pad", 2, 15, 17, 3, 2, 1, 0);
    case_transpose("transpose", 2, 37, 50);
    case_elementwise("elementwise", 5, 77);
    case_fold_bn("fold_bn", 4, 11, 13, 6);

    ocl_check(clFinish(ocl.queue), "clFinish");
    fclose(cases);
    ops_free(&o);
    ocl_free(&ocl);
    G_message(_("Operations written to <%s>"), dir);
    return 0;
}
