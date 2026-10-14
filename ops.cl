/* SPDX-License-Identifier: Unlicense */

/* Generic tensor kernels of i.coregistration.opencl, shared by every
 * matcher (see ops.h for the host side). OpenCL C 1.1, float32.
 *
 * Two layouts are used. Image-like tensors are channel-major (C, H, W),
 * as PyTorch's NCHW without the batch axis. Token sequences are
 * token-major (rows = tokens, columns = channels), which makes every
 * nn.Linear and 1x1 convolution on tokens a row-major GEMM against the
 * PyTorch [out, in] weight; transpose2d converts between the two.
 *
 * Element-wise kernels index with 32-bit integers where the host keeps
 * tensors below 2^31 elements: 64-bit division is emulated on GPUs. The
 * GEMM, LayerNorm and softmax kernels come from i.sam3.opencl (same
 * author, Unlicense). */

#define ACT_NONE 0
#define ACT_GELU 1  /* Exact (erf) GELU, torch's default. */
#define ACT_RELU 2
#define ACT_LEAKY 3 /* LeakyReLU, negative slope given as parameter. */
#define ACT_GELU_TANH 4

float apply_act(float v, const int act, const float slope)
{
    switch (act) {
    case ACT_GELU:
        return 0.5f * v * (1.0f + erf(v * 0.70710678118654752f));
    case ACT_RELU:
        return fmax(v, 0.0f);
    case ACT_LEAKY:
        return v >= 0.0f ? v : slope * v;
    case ACT_GELU_TANH:
        return 0.5f * v *
               (1.0f + tanh(0.79788456080286536f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

/* Batched GEMM: C = act(alpha * A * op(B) + bias) + R.
 *
 * A is M x K row-major (lda). With transb != 0, B is stored N x K (the
 * PyTorch Linear weight layout, B(k, n) = B[n * ldb + k]); otherwise B
 * is K x N (B(k, n) = B[k * ldb + n]). Batch index b = get_group_id(2)
 * is split into (b / nb1, b % nb1), each part with its own element
 * stride per operand; a stride of 0 shares an operand across that batch
 * axis. Element offsets aoff/boff/coff/roff select sub-matrices of the
 * buffers. bias is indexed by column (biasrow 0) or by row (biasrow 1,
 * for the channel-major W x X products of 1x1 convolutions). R, when
 * used, has C's layout and may alias C for in-place residual updates.
 *
 * Generic version: 64 x 64 output tile per work-group of 16 x 16 items,
 * each item computing a 4 x 4 register block. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm(const int M, const int N, const int K, __global const float *A,
     const long aoff, const int lda, const long sa0, const long sa1,
     __global const float *B, const long boff, const int ldb, const long sb0,
     const long sb1, const int transb, __global float *C, const long coff,
     const int ldc, const long sc0, const long sc1, const int nb1,
     __global const float *bias, const int biasrow, __global const float *R,
     const long roff, const float alpha, const int act, const float slope)
{
    __local float As[16][65];
    __local float Bs[16][65];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 64, m0 = get_group_id(1) * 64;
    const int b = get_group_id(2);
    const long b0 = b / nb1, b1 = b % nb1;
    float acc[4][4];
    int i, j, k, l;

    A += aoff + b0 * sa0 + b1 * sa1;
    B += boff + b0 * sb0 + b1 * sb1;
    C += coff + b0 * sc0 + b1 * sc1;
    if (R)
        R += roff + b0 * sc0 + b1 * sc1;

    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (l = 0; l < 4; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 4, c = idx & 15;
            const int gm = m0 + r, gk = k0 + c;

            As[c][r] = (gm < M && gk < K) ? A[(long)gm * lda + gk] : 0.0f;
            if (transb) {
                const int gn = n0 + r;

                Bs[c][r] = (gn < N && gk < K) ? B[(long)gn * ldb + gk] : 0.0f;
            }
            else {
                const int kk = idx >> 6, nn = idx & 63;
                const int gk2 = k0 + kk, gn = n0 + nn;

                Bs[kk][nn] =
                    (gn < N && gk2 < K) ? B[(long)gk2 * ldb + gn] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (k = 0; k < 16; k++) {
            float a[4], bb[4];

            for (i = 0; i < 4; i++)
                a[i] = As[k][ty + 16 * i];
            for (j = 0; j < 4; j++)
                bb[j] = Bs[k][tx + 16 * j];
            for (i = 0; i < 4; i++)
                for (j = 0; j < 4; j++)
                    acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = 0; i < 4; i++) {
        const int m = m0 + ty + 16 * i;

        if (m >= M)
            continue;
        for (j = 0; j < 4; j++) {
            const int n = n0 + tx + 16 * j;
            float v;

            if (n >= N)
                continue;
            v = alpha * acc[i][j];
            if (bias)
                v += bias[biasrow ? m : n];
            v = apply_act(v, act, slope);
            if (R)
                v += R[(long)m * ldc + n];
            C[(long)m * ldc + n] = v;
        }
    }
}

/* Epilogue of gemm_big: one row of an 8 x 8 register block. */
void store_row8(__global float *C, __global const float *R,
                __global const float *bias, const int biasrow, const int m,
                const int n, const int M, const int N, const int ldc,
                const float alpha, const int act, const float slope,
                const float4 lo, const float4 hi)
{
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    if (m >= M)
        return;
    for (int j = 0; j < 8; j++) {
        float x;

        if (n + j >= N)
            return;
        x = alpha * v[j];
        if (bias)
            x += bias[biasrow ? m : n + j];
        x = apply_act(x, act, slope);
        if (R)
            x += R[(long)m * ldc + n + j];
        C[(long)m * ldc + n + j] = x;
    }
}

/* Rank-1 update of the 8 x 8 block row r from A column value s. */
#define ROW_FMA(r, s)                                                          \
    c##r##0 = mad((float4)(s), b0, c##r##0);                                   \
    c##r##1 = mad((float4)(s), b1, c##r##1)

/* Same contract as gemm, for large, 16-byte aligned operands: 128 x 128
 * output tile per 16 x 16 work-group, 8 x 8 register block per item
 * held in explicit float4 accumulators (private arrays of that size end
 * up in scratch memory with Mesa, costing a factor of 70), float4 loads.
 * Requires K, lda, ldb and all A/B offsets and batch strides to be
 * multiples of 4 (checked by the host). */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm_big(const int M, const int N, const int K, __global const float *A,
         const long aoff, const int lda, const long sa0, const long sa1,
         __global const float *B, const long boff, const int ldb,
         const long sb0, const long sb1, const int transb, __global float *C,
         const long coff, const int ldc, const long sc0, const long sc1,
         const int nb1, __global const float *bias, const int biasrow,
         __global const float *R, const long roff, const float alpha,
         const int act, const float slope)
{
    __local float As[16][132];
    __local float Bs[16][132];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 128, m0 = get_group_id(1) * 128;
    const int b = get_group_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float4 c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c30 = 0.0f, c31 = 0.0f;
    float4 c40 = 0.0f, c41 = 0.0f, c50 = 0.0f, c51 = 0.0f;
    float4 c60 = 0.0f, c61 = 0.0f, c70 = 0.0f, c71 = 0.0f;

    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int l = 0; l < 2; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 2, kq = (idx & 3) * 4;
            const int gm = m0 + r, gk = k0 + kq;
            float4 v = 0.0f;

            if (gm < M && gk < K)
                v = vload4(0, A + (long)gm * lda + gk);
            As[kq][r] = v.x;
            As[kq + 1][r] = v.y;
            As[kq + 2][r] = v.z;
            As[kq + 3][r] = v.w;

            if (transb) {
                const int gn = n0 + r;

                v = 0.0f;
                if (gn < N && gk < K)
                    v = vload4(0, B + (long)gn * ldb + gk);
                Bs[kq][r] = v.x;
                Bs[kq + 1][r] = v.y;
                Bs[kq + 2][r] = v.z;
                Bs[kq + 3][r] = v.w;
            }
            else {
                const int kk = idx >> 5, nq = (idx & 31) * 4;
                const int gk2 = k0 + kk, gn = n0 + nq;

                v = 0.0f;
                if (gk2 < K) {
                    __global const float *bp = B + (long)gk2 * ldb + gn;

                    if (gn + 3 < N)
                        v = vload4(0, bp);
                    else {
                        if (gn < N)
                            v.x = bp[0];
                        if (gn + 1 < N)
                            v.y = bp[1];
                        if (gn + 2 < N)
                            v.z = bp[2];
                    }
                }
                vstore4(v, 0, &Bs[kk][nq]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            const float4 a0 = vload4(0, &As[k][ty * 8]);
            const float4 a1 = vload4(0, &As[k][ty * 8 + 4]);
            const float4 b0 = vload4(0, &Bs[k][tx * 8]);
            const float4 b1 = vload4(0, &Bs[k][tx * 8 + 4]);

            ROW_FMA(0, a0.x);
            ROW_FMA(1, a0.y);
            ROW_FMA(2, a0.z);
            ROW_FMA(3, a0.w);
            ROW_FMA(4, a1.x);
            ROW_FMA(5, a1.y);
            ROW_FMA(6, a1.z);
            ROW_FMA(7, a1.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#define STORE(r)                                                               \
    store_row8(C, R, bias, biasrow, m0 + ty * 8 + r, n0 + tx * 8, M, N, ldc,  \
               alpha, act, slope, c##r##0, c##r##1)
    STORE(0);
    STORE(1);
    STORE(2);
    STORE(3);
    STORE(4);
    STORE(5);
    STORE(6);
    STORE(7);
#undef STORE
}

/* Sum and max of val over a work-group of n items (power of two). */
float wg_sum(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

float wg_max(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

#define TS 16 /* Output tile side of conv5x5. */
#define OCB 8 /* Output channels per conv5x5 work-item. */
#define PS (TS + 4)

/* 5x5 convolution, stride 1, zero padding 2 ("same"), with bias: the
 * fast path of DISK's U-Net. With norm set, each input value x of
 * channel c is first normalized and gated as in DISK's Conv block,
 * prelu((x - mean[c]) * rstd[c], alpha[c]), and the zero padding applies
 * to that result. */
__kernel __attribute__((reqd_work_group_size(TS, TS, 1))) void
conv5x5(__global const float *in, const int C, const int H, const int W,
        __global const float *wt, __global const float *bias, const int K,
        __global const float *mean, __global const float *rstd,
        __global const float *alpha, const int norm, __global float *out)
{
    __local float patch[PS * PS];
    __local float wl[OCB * 25];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int x = get_global_id(0), y = get_global_id(1);
    const int x0 = get_group_id(0) * TS - 2, y0 = get_group_id(1) * TS - 2;
    const int k0 = get_group_id(2) * OCB;
    const int lid = ly * TS + lx;
    float acc[OCB];
    int c, i, o, ky, kx;

    for (o = 0; o < OCB; o++)
        acc[o] = 0.0f;
    for (c = 0; c < C; c++) {
        const __global float *src = in + (size_t)c * H * W;
        float m = 0.0f, r = 1.0f, a = 0.0f;

        if (norm) {
            m = mean[c];
            r = rstd[c];
            a = alpha[c];
        }
        for (i = lid; i < PS * PS; i += TS * TS) {
            const int px = x0 + i % PS, py = y0 + i / PS;
            float v = 0.0f;

            if (px >= 0 && px < W && py >= 0 && py < H) {
                v = src[(size_t)py * W + px];
                if (norm) {
                    v = (v - m) * r;
                    v = v >= 0.0f ? v : a * v;
                }
            }
            patch[i] = v;
        }
        for (i = lid; i < OCB * 25; i += TS * TS) {
            o = i / 25;
            wl[i] = k0 + o < K ? wt[((size_t)(k0 + o) * C + c) * 25 + i % 25]
                               : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (ky = 0; ky < 5; ky++)
            for (kx = 0; kx < 5; kx++) {
                const float v = patch[(ly + ky) * PS + lx + kx];

                for (o = 0; o < OCB; o++)
                    acc[o] += v * wl[o * 25 + ky * 5 + kx];
            }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < W && y < H)
        for (o = 0; o < OCB; o++)
            if (k0 + o < K)
                out[((size_t)(k0 + o) * H + y) * W + x] = acc[o] + bias[k0 + o];
}

/* General 2D convolution of a (C, H, W) tensor into a (K, Ho, Wo) one,
 * as torch.nn.functional.conv2d: kh x kw kernel, stride, zero padding,
 * dilation and groups (C and K multiples of groups; weights
 * (K, C / groups, kh, kw)), optional bias, activation after the bias.
 * With mean set, each input value is first normalized and gated,
 * prelu((x - mean[c]) * rstd[c], alpha[c]) (alpha NULL: no gate), before
 * the zero padding, as conv5x5. One work-item per output value: the
 * general path; fast paths come per model. */
__kernel void conv2d(__global const float *in, const int C, const int H,
                     const int W, __global const float *wt,
                     __global const float *bias, const int K, const int Ho,
                     const int Wo, const int kh, const int kw, const int sy,
                     const int sx, const int py, const int px, const int dy,
                     const int dx, const int groups,
                     __global const float *mean, __global const float *rstd,
                     __global const float *alpha, const int act,
                     const float slope, __global float *out)
{
    const int x = get_global_id(0), y = get_global_id(1), k = get_global_id(2);
    const int cg = C / groups, kg = K / groups, c0 = (k / kg) * cg;
    __global const float *w = wt + (size_t)k * cg * kh * kw;
    float acc = 0.0f;
    int c, i, j;

    if (x >= Wo || y >= Ho || k >= K)
        return;
    for (c = 0; c < cg; c++) {
        const __global float *src = in + (size_t)(c0 + c) * H * W;
        float m = 0.0f, r = 1.0f, a = 1.0f;

        if (mean) {
            m = mean[c0 + c];
            r = rstd[c0 + c];
            if (alpha)
                a = alpha[c0 + c];
        }
        for (i = 0; i < kh; i++) {
            const int iy = y * sy - py + i * dy;

            if (iy < 0 || iy >= H)
                continue;
            for (j = 0; j < kw; j++) {
                const int ix = x * sx - px + j * dx;
                float v;

                if (ix < 0 || ix >= W)
                    continue;
                v = src[(size_t)iy * W + ix];
                if (mean) {
                    v = (v - m) * r;
                    v = v >= 0.0f ? v : a * v;
                }
                acc += v * w[(c * kh + i) * kw + j];
            }
        }
    }
    if (bias)
        acc += bias[k];
    out[((size_t)k * Ho + y) * Wo + x] = apply_act(acc, act, slope);
}

/* Instance normalization statistics, one work-group of 256 per channel:
 * mean and 1 / sqrt(biased variance + eps), two passes over the plane
 * (torch.nn.InstanceNorm2d). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
chan_stats(__global const float *x, const int HW, const float eps,
           __global float *mean, __global float *rstd)
{
    __local float red[256];
    const int c = get_group_id(0), l = get_local_id(0);
    const __global float *p = x + (size_t)c * HW;
    float s = 0.0f, m;
    int i, h;

    for (i = l; i < HW; i += 256)
        s += p[i];
    red[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h)
            red[l] += red[l + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = red[0] / HW;
    barrier(CLK_LOCAL_MEM_FENCE);

    s = 0.0f;
    for (i = l; i < HW; i += 256) {
        const float d = p[i] - m;

        s += d * d;
    }
    red[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h)
            red[l] += red[l + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        mean[c] = m;
        rstd[c] = rsqrt(red[0] / HW + eps);
    }
}

/* Per-channel affine normalization of a (C, HW) tensor, in place or not:
 * y = act((x - mean[c]) * rstd[c] * g[c] + b[c]); mean/rstd NULL skip
 * the normalization, g/b NULL the affine part. With the instance
 * statistics of chan_stats this is InstanceNorm2d, with running
 * statistics BatchNorm2d (unfolded). */
__kernel void chan_affine(__global const float *X, __global float *Y,
                          const int HW, const int total,
                          __global const float *mean,
                          __global const float *rstd, __global const float *g,
                          __global const float *b, const int act,
                          const float slope)
{
    const int i = get_global_id(0), c = i / HW;
    float v;

    if (i >= total)
        return;
    v = X[i];
    if (mean)
        v = (v - mean[c]) * rstd[c];
    if (g)
        v *= g[c];
    if (b)
        v += b[c];
    Y[i] = apply_act(v, act, slope);
}

/* PReLU with one slope per channel of a (C, HW) tensor, in place or not. */
__kernel void prelu(__global const float *X, __global float *Y, const int HW,
                    const int total, __global const float *alpha)
{
    const int i = get_global_id(0);
    float v;

    if (i >= total)
        return;
    v = X[i];
    Y[i] = v >= 0.0f ? v : alpha[i / HW] * v;
}

/* Element-wise activation, in place or not. */
__kernel void act_map(__global const float *X, __global float *Y,
                      const int total, const int act, const float slope)
{
    const int i = get_global_id(0);

    if (i < total)
        Y[i] = apply_act(X[i], act, slope);
}

/* Row-wise LayerNorm over C channels (token-major), Y may alias X; g and
 * bt NULL for no affine part. The work-group size (a power of two up to
 * 256) is the local size chosen by the host. */
__kernel void layernorm(__global const float *X, __global float *Y,
                        __global const float *g, __global const float *bt,
                        const int C, const float eps, const int act,
                        const float slope)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    const long row = get_group_id(0);
    __global const float *x = X + row * C;
    __global float *y = Y + row * C;
    float s = 0.0f, mean, var;
    int c;

    for (c = lid; c < C; c += n)
        s += x[c];
    mean = wg_sum(s, scratch, n) / C;
    s = 0.0f;
    for (c = lid; c < C; c += n) {
        const float d = x[c] - mean;

        s += d * d;
    }
    var = wg_sum(s, scratch, n) / C;
    s = rsqrt(var + eps);
    for (c = lid; c < C; c += n) {
        float v = (x[c] - mean) * s;

        if (g)
            v = v * g[c] + bt[c];
        y[c] = apply_act(v, act, slope);
    }
}

/* In-place softmax of rows of L entries scaled by scale (softmax(scale *
 * x), scale = 1 / temperature), one work-group (a power of two up to
 * 256, chosen by the host) per row. */
__kernel void softmax_rows(__global float *S, const int L, const float scale)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    __global float *s = S + (long)get_group_id(0) * L;
    float m = -INFINITY, sum = 0.0f, v;
    int j;

    for (j = lid; j < L; j += n)
        m = fmax(m, s[j] * scale);
    m = wg_max(m, scratch, n);
    for (j = lid; j < L; j += n) {
        v = exp(s[j] * scale - m);
        s[j] = v;
        sum += v;
    }
    v = 1.0f / wg_sum(sum, scratch, n);
    for (j = lid; j < L; j += n)
        s[j] *= v;
}

#define AB 16   /* Queries and keys per attention block. */
#define AMAXD 8 /* Head dimension per work-item column: d <= AB * AMAXD. */

/* Scaled dot-product attention by blocks (flash-attention style), never
 * storing the score matrix: O = softmax(scale * Q K^T) V per head, with
 * a running maximum and sum per query. Q, K, V and O are token-major
 * with head h at columns h * d .. h * d + d - 1 of rows of ldq/ldk/ldv/
 * ldo floats, from element offsets qoff..ooff; d <= 128. Work-group
 * 16 x 16 per (block of 16 queries, head): item (j, i) computes the
 * score of query i with key j of the current key block, then updates
 * the output columns j, j + 16, ... of query i. With kmask set, key k is
 * ignored where kmask[k] is 0 (a query with no key left gets zeros). */
__kernel __attribute__((reqd_work_group_size(AB, AB, 1))) void
attention(__global const float *Q, const long qoff, const int ldq,
          __global const float *K, const long koff, const int ldk,
          __global const float *V, const long voff, const int ldv,
          __global float *O, const long ooff, const int ldo, const int Nq,
          const int Nk, const int d, const float scale,
          __global const uchar *kmask)
{
    __local float qs[AB][AB * AMAXD + 1];
    __local float ks[AB][AB * AMAXD + 1];
    __local float vs[AB][AB * AMAXD + 1];
    __local float p[AB][AB + 1];
    const int j = get_local_id(0), i = get_local_id(1);
    const int q0 = get_group_id(1) * AB, h = get_group_id(2);
    const int qi = q0 + i;
    float o[AMAXD], m = -INFINITY, l = 0.0f;
    int t, c;

    Q += qoff + h * d;
    K += koff + h * d;
    V += voff + h * d;
    for (t = 0; t < AMAXD; t++)
        o[t] = 0.0f;
    for (c = j; c < d; c += AB)
        qs[i][c] = qi < Nq ? Q[(long)qi * ldq + c] * scale : 0.0f;

    for (int k0 = 0; k0 < Nk; k0 += AB) {
        float s = -INFINITY, mn, corr;

        /* Row i of the work-group loads key and value k0 + i. */
        for (c = j; c < d; c += AB) {
            const int kk = k0 + i;

            ks[i][c] = kk < Nk ? K[(long)kk * ldk + c] : 0.0f;
            vs[i][c] = kk < Nk ? V[(long)kk * ldv + c] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k0 + j < Nk && (!kmask || kmask[k0 + j])) {
            s = 0.0f;
            for (c = 0; c < d; c++)
                s = mad(qs[i][c], ks[j][c], s);
        }
        p[i][j] = s;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Every item of row i sees the same scores: the same maximum. */
        mn = m;
        for (t = 0; t < AB; t++)
            mn = fmax(mn, p[i][t]);
        if (mn > -INFINITY) {
            float sum = 0.0f;

            corr = exp(m - mn);
            for (t = 0; t < AMAXD; t++)
                o[t] *= corr;
            for (int kk = 0; kk < AB; kk++) {
                const float e = exp(p[i][kk] - mn);

                sum += e;
                for (t = 0; t < AMAXD && j + AB * t < d; t++)
                    o[t] = mad(e, vs[kk][j + AB * t], o[t]);
            }
            l = l * corr + sum;
            m = mn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (qi < Nq)
        for (t = 0; t < AMAXD && j + AB * t < d; t++)
            O[ooff + (long)qi * ldo + h * d + j + AB * t] =
                l > 0.0f ? o[t] / l : 0.0f;
}

/* Rotary position encoding, in place on nh heads of d channels of
 * token-major rows (row stride ld, from element offset off). Row r uses
 * the angles of position r % npos: cs holds (cos, sin) for every
 * position and frequency, d / 2 per position. interleaved != 0 rotates
 * channel pairs (2f, 2f + 1) (complex view, as RoPE-ViTs and SAM), 0
 * rotates (f, f + d / 2) (rotate_half, as DINOv3 and LLaMA). */
__kernel void rope(__global float *X, const long off, const int ld,
                   const int nh, const int d, const int npos,
                   __global const float2 *cs, const int interleaved,
                   const int total)
{
    const int gid = get_global_id(0), nf = d / 2;
    const int f = gid % nf, h = (gid / nf) % nh, r = gid / (nf * nh);
    __global float *x = X + off + (long)r * ld + h * d;
    float2 rot;
    float a, b;
    int i0, i1;

    if (gid >= total)
        return;
    rot = cs[(r % npos) * nf + f];
    i0 = interleaved ? 2 * f : f;
    i1 = interleaved ? 2 * f + 1 : f + nf;
    a = x[i0];
    b = x[i1];
    x[i0] = a * rot.x - b * rot.y;
    x[i1] = a * rot.y + b * rot.x;
}

/* Bilinear resampling of a (C, h, w) tensor to (C, H, W), written into
 * channels coff.. of a (Ctot, H, W) tensor, as torch F.interpolate(mode=
 * "bilinear") with align_corners; without it, sy and sx are the scale
 * factors' inverses (h / H when the size is given, 1 / scale_factor when
 * the scale is: torch uses the latter when given a scale_factor). */
__kernel void resize_bilinear(__global const float *in, const int h,
                              const int w, __global float *out, const int H,
                              const int W, const int coff,
                              const int align_corners, const float sy,
                              const float sx)
{
    const int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    float fx, fy, ux, uy;
    int x0, y0, x1, y1;
    const __global float *p;

    if (x >= W || y >= H)
        return;
    if (align_corners) {
        ux = W > 1 ? x * (float)(w - 1) / (W - 1) : 0.0f;
        uy = H > 1 ? y * (float)(h - 1) / (H - 1) : 0.0f;
    }
    else {
        ux = fmax((x + 0.5f) * sx - 0.5f, 0.0f);
        uy = fmax((y + 0.5f) * sy - 0.5f, 0.0f);
    }
    x0 = (int)ux;
    y0 = (int)uy;
    x1 = min(x0 + 1, w - 1);
    y1 = min(y0 + 1, h - 1);
    fx = ux - x0;
    fy = uy - y0;
    p = in + (size_t)c * h * w;
    out[((size_t)(c + coff) * H + y) * W + x] =
        (1.0f - fy) * ((1.0f - fx) * p[y0 * w + x0] + fx * p[y0 * w + x1]) +
        fy * ((1.0f - fx) * p[y1 * w + x0] + fx * p[y1 * w + x1]);
}

/* Bilinear sampling of a (C, H, W) tensor at n points, as torch
 * F.grid_sample(mode="bilinear", padding_mode="zeros"): grid holds (x, y)
 * pairs normalized to -1..1, out is (C, n). */
__kernel void grid_sample(__global const float *in, const int C, const int H,
                          const int W, __global const float2 *grid,
                          const int n, const int align_corners,
                          __global float *out)
{
    const int i = get_global_id(0), c = get_global_id(1);
    const __global float *p = in + (size_t)c * H * W;
    float2 g;
    float ix, iy, fx, fy, v = 0.0f;
    int x0, y0;

    if (i >= n || c >= C)
        return;
    g = grid[i];
    if (align_corners) {
        ix = (g.x + 1.0f) * 0.5f * (W - 1);
        iy = (g.y + 1.0f) * 0.5f * (H - 1);
    }
    else {
        ix = ((g.x + 1.0f) * W - 1.0f) * 0.5f;
        iy = ((g.y + 1.0f) * H - 1.0f) * 0.5f;
    }
    x0 = (int)floor(ix);
    y0 = (int)floor(iy);
    fx = ix - x0;
    fy = iy - y0;
#define AT(yy, xx)                                                             \
    ((xx) >= 0 && (xx) < W && (yy) >= 0 && (yy) < H ? p[(yy) * W + (xx)]       \
                                                    : 0.0f)
    if (isfinite(ix) && isfinite(iy))
        v = (1.0f - fy) * ((1.0f - fx) * AT(y0, x0) + fx * AT(y0, x0 + 1)) +
            fy * ((1.0f - fx) * AT(y0 + 1, x0) + fx * AT(y0 + 1, x0 + 1));
#undef AT
    out[(size_t)c * n + i] = v;
}

/* Local windows of a (C, H, W) tensor, as torch F.unfold: kh x kw
 * windows, stride, zero padding, into a (C * kh * kw, L) tensor, row
 * c * kh * kw + i * kw + j, column l = oy * Wo + ox. */
__kernel void unfold(__global const float *in, const int C, const int H,
                     const int W, const int kh, const int kw, const int sy,
                     const int sx, const int py, const int px, const int Ho,
                     const int Wo, __global float *out)
{
    const int l = get_global_id(0), row = get_global_id(1);
    const int L = Ho * Wo, c = row / (kh * kw), i = (row / kw) % kh,
              j = row % kw;
    int iy, ix;

    if (l >= L || row >= C * kh * kw)
        return;
    iy = (l / Wo) * sy - py + i;
    ix = (l % Wo) * sx - px + j;
    out[(size_t)row * L + l] = iy >= 0 && iy < H && ix >= 0 && ix < W
                                   ? in[((size_t)c * H + iy) * W + ix]
                                   : 0.0f;
}

/* 2x2 average pooling, stride 2 (H and W even), of a (C, H, W) tensor. */
__kernel void avgpool2(__global const float *in, const int H, const int W,
                       __global float *out)
{
    const int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    const int w = W / 2, h = H / 2;
    const __global float *p;

    if (x >= w || y >= h)
        return;
    p = in + ((size_t)c * H + 2 * y) * W + 2 * x;
    out[((size_t)c * h + y) * w + x] = 0.25f * (p[0] + p[1] + p[W] + p[W + 1]);
}

/* k x k pooling of a (C, H, W) tensor into (C, Ho, Wo) with stride s and
 * zero padding pad: maximum (max != 0, padding ignored, as max_pool2d)
 * or average (padding counted, avg_pool2d's count_include_pad). */
__kernel void pool2d(__global const float *in, const int H, const int W,
                     const int Ho, const int Wo, const int k, const int s,
                     const int pad, const int max, __global float *out)
{
    const int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    const __global float *p = in + (size_t)c * H * W;
    float acc = max ? -INFINITY : 0.0f;
    int i, j;

    if (x >= Wo || y >= Ho)
        return;
    for (i = 0; i < k; i++) {
        const int iy = y * s - pad + i;

        if (iy < 0 || iy >= H)
            continue;
        for (j = 0; j < k; j++) {
            const int ix = x * s - pad + j;

            if (ix < 0 || ix >= W)
                continue;
            acc = max ? fmax(acc, p[iy * W + ix]) : acc + p[iy * W + ix];
        }
    }
    out[((size_t)c * Ho + y) * Wo + x] = max ? acc : acc / (k * k);
}

/* Batched transpose: (B, R, C) to (B, C, R), 16 x 16 tiles through local
 * memory; channel-major (C, HW) to token-major (HW, C) and back. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
transpose2d(__global const float *in, __global float *out, const int R,
            const int C)
{
    __local float t[16][17];
    const int b = get_global_id(2);
    const int c = get_global_id(0), r = get_global_id(1);
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int c0 = get_group_id(0) * 16, r0 = get_group_id(1) * 16;
    const size_t base = (size_t)b * R * C;

    if (r < R && c < C)
        t[ly][lx] = in[base + (size_t)r * C + c];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (r0 + lx < R && c0 + ly < C)
        out[base + (size_t)(c0 + ly) * R + r0 + lx] = t[lx][ly];
}

/* out[i] = alpha * a[i] + beta * b[i % bmod] (b NULL: alpha * a[i]);
 * out may alias a. Element-wise sums, scaling and broadcast biases. */
__kernel void axpby(__global float *out, __global const float *a,
                    const float alpha, __global const float *b,
                    const float beta, const int n, const int bmod)
{
    const int i = get_global_id(0);

    if (i >= n)
        return;
    out[i] = alpha * a[i] + (b ? beta * b[i % bmod] : 0.0f);
}
