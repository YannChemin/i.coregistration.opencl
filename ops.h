/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_OPS_H
#define I_COREGISTRATION_OPENCL_OPS_H

#include "ocl_backend.h"

/* Activations, applied after the bias (see apply_act in ops.cl). */
#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2
#define ACT_LEAKY 3
#define ACT_GELU_TANH 4

/* Largest attention head dimension of ops_attention. */
#define OPS_MAX_HEAD_DIM 128

/* The generic kernels of ops.cl, created once per device. Tensors are
 * float32, channel-major (C, H, W) for image-like data and token-major
 * (tokens, channels) for sequences. */
struct ops {
    struct ocl_backend *ocl;
    cl_kernel gemm, gemm_big, conv5x5, conv2d, chan_stats, chan_affine, prelu,
        act_map, layernorm, softmax_rows, attention, rope, resize_bilinear,
        grid_sample, unfold, avgpool2, pool2d, transpose2d, axpby;
    /* Work buffer of the im2col convolutions, grown on demand up to
     * im2col_bytes (256 MiB by default) per block of output positions. */
    cl_mem scratch;
    size_t scratch_bytes, im2col_bytes;
};

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides (see the gemm kernel). */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

struct opnd ops_mat(cl_mem buf, long off, int ld);
struct opnd ops_bmat(cl_mem buf, long off, int ld, long s0, long s1);

void ops_init(struct ops *o, struct ocl_backend *ocl);
void ops_free(struct ops *o);

/* Enqueue kernel k (fatal on error). With I_COREGISTRATION_OPENCL_PROFILE
 * set, every launch is timed under the name what, and ops_report()
 * prints the totals. */
void ops_run(struct ocl_backend *ocl, cl_kernel k, int dims, const size_t *g,
             const size_t *l, const char *what);
void ops_report(void);

/* Batched C = act(alpha * A * op(B) + bias) + R over nb = nb0 * nb1
 * products (see the gemm kernel); bias per column, or per row with
 * biasrow set; R NULL for no residual. */
void ops_gemm(struct ops *o, int M, int N, int K, struct opnd A, struct opnd B,
              int transb, struct opnd C, int nb, int nb1, cl_mem bias,
              int biasrow, cl_mem R, long roff, float alpha, int act,
              float slope);

/* Token-major Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when
 * residual), W the PyTorch [out, in] weight. */
void ops_linear(struct ops *o, int rows, int in, int out, cl_mem X, long xoff,
                cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, float slope,
                int residual);

/* 1x1 convolution of a channel-major (C, HW) tensor into (K, HW):
 * W (K, C) times X, bias per output channel. */
void ops_conv1x1(struct ops *o, cl_mem X, long xoff, int C, int HW, cl_mem W,
                 cl_mem b, int K, cl_mem Y, long yoff, int act, float slope);

/* Geometry of a 2D convolution or pooling window. */
struct conv_geom {
    int kh, kw, sy, sx, py, px, dy, dx, groups;
};

/* Square kernel k, stride s, padding p, dilation 1, one group. */
struct conv_geom ops_geom(int k, int s, int p);

/* Output size of a convolution along one axis. */
int ops_conv_out(int n, int k, int s, int p, int d);

/* General convolution of in (C, H, W) into out (K, Ho, Wo), Ho and Wo
 * from the geometry. mean/rstd/alpha (NULL for none) normalize and gate
 * the input as in DISK; the 5x5 / stride 1 / padding 2 / one group case
 * runs the tiled conv5x5 kernel, other one-group convolutions without
 * normalization im2col and GEMM by blocks of output positions, the rest
 * the direct conv2d kernel. */
void ops_conv2d(struct ops *o, cl_mem in, int C, int H, int W, cl_mem w,
                cl_mem b, int K, const struct conv_geom *g, cl_mem mean,
                cl_mem rstd, cl_mem alpha, int act, float slope, cl_mem out);

/* Instance statistics of a (C, HW) tensor: mean and 1 / sqrt(var + eps)
 * per channel into mean, rstd (C floats each). */
void ops_chan_stats(struct ops *o, cl_mem x, int C, int HW, float eps,
                    cl_mem mean, cl_mem rstd);

/* Y = act((X - mean) * rstd * g + b) per channel; any of mean (with
 * rstd), g, b may be NULL. Y may alias X. */
void ops_chan_affine(struct ops *o, cl_mem X, cl_mem Y, int C, int HW,
                     cl_mem mean, cl_mem rstd, cl_mem g, cl_mem b, int act,
                     float slope);

void ops_prelu(struct ops *o, cl_mem X, cl_mem Y, int C, int HW, cl_mem alpha);
void ops_act(struct ops *o, cl_mem X, cl_mem Y, long n, int act, float slope);

/* Row LayerNorm of rows x C (token-major), g/b NULL for none. */
void ops_layernorm(struct ops *o, long rows, int C, cl_mem X, cl_mem Y,
                   cl_mem g, cl_mem b, float eps, int act, float slope);

/* In-place softmax(scale * x) over rows of L. */
void ops_softmax_rows(struct ops *o, cl_mem S, long rows, int L, float scale);

/* Blockwise scaled dot-product attention over nh heads of dimension d
 * (<= OPS_MAX_HEAD_DIM): O = softmax(scale Q K^T) V per head, token-major
 * operands, head h at columns h * d. kmask (Nk bytes, 0 = ignore the
 * key) may be NULL. */
void ops_attention(struct ops *o, struct opnd Q, struct opnd K, struct opnd V,
                   struct opnd O, int Nq, int Nk, int nh, int d, float scale,
                   cl_mem kmask);

/* Rotary encoding in place on rows x (nh heads of d) of X (row stride
 * ld, offset off), row r at position r % npos of the (cos, sin) table cs
 * (npos x d / 2 float pairs, built by each model for its own
 * convention). */
void ops_rope(struct ops *o, cl_mem X, long off, int ld, long rows, int nh,
              int d, int npos, cl_mem cs, int interleaved);

/* Bilinear resampling of in (C, h, w) into channels coff.. of out
 * (*, H, W). With align_corners 0, scale > 0 gives torch's scale_factor
 * (source step 1 / scale), scale 0 the size ratio h / H. */
void ops_resize(struct ops *o, cl_mem in, int C, int h, int w, cl_mem out,
                int H, int W, int coff, int align_corners, float scale);

/* Bilinear sampling of in (C, H, W) at n normalized (x, y) points of
 * grid into out (C, n), zero padding. */
void ops_grid_sample(struct ops *o, cl_mem in, int C, int H, int W, cl_mem grid,
                     int n, int align_corners, cl_mem out);

/* F.unfold of in (C, H, W) into out (C * kh * kw, Ho * Wo). */
void ops_unfold(struct ops *o, cl_mem in, int C, int H, int W,
                const struct conv_geom *g, cl_mem out);

/* 2x2 / stride 2 average pooling (DISK). */
void ops_avgpool2(struct ops *o, cl_mem in, int C, int H, int W, cl_mem out);

/* k x k pooling, stride s, padding p: maximum or average. */
void ops_pool2d(struct ops *o, cl_mem in, int C, int H, int W, int k, int s,
                int p, int max, cl_mem out);

/* (B, R, C) to (B, C, R). */
void ops_transpose(struct ops *o, cl_mem in, cl_mem out, int B, int R, int C);

/* out = alpha * a + beta * b[i % bmod] (b NULL: alpha * a). */
void ops_axpby(struct ops *o, cl_mem out, cl_mem a, float alpha, cl_mem b,
               float beta, long n, long bmod);

/* Fold an eval-mode BatchNorm (gamma, beta, running mean and variance,
 * eps) following a convolution into its weights w (K x per) and bias b
 * (K floats, written; the convolution's own bias, or zeros, on entry). */
void ops_fold_bn(float *w, float *b, int K, long per, const float *gamma,
                 const float *beta, const float *mean, const float *var,
                 float eps);

#endif /* I_COREGISTRATION_OPENCL_OPS_H */
