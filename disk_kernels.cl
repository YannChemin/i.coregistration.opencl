/* SPDX-License-Identifier: Unlicense */

/* OpenCL kernels of i.coregistration.opencl: the DISK U-Net (5x5
 * convolutions with instance normalization and PReLU folded into the
 * input loads, 2x2 average pooling, bilinear 2x upsampling), the 5x5
 * non-maxima suppression of the heatmap and the descriptor gather.
 * Tensors are float32, channel-major (C, H, W). OpenCL C 1.1. */

#define TS 16 /* Output tile side of conv5x5. */
#define OCB 8 /* Output channels per conv5x5 work-item. */
#define PS (TS + 4)

/* 5x5 convolution, zero padding 2 ("same"), with bias. With norm set,
 * each input value x of channel c is first normalized and gated as in
 * DISK's Conv block, prelu((x - mean[c]) * rstd[c], alpha[c]), and the
 * zero padding applies to that result. */
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

/* Instance normalization statistics, one work-group of 256 per channel:
 * mean and 1 / sqrt(biased variance + 1e-5), two passes over the plane
 * (torch.nn.InstanceNorm2d without affine parameters). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
chan_stats(__global const float *x, const int HW, __global float *mean,
           __global float *rstd)
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
        rstd[c] = rsqrt(red[0] / HW + 1e-5f);
    }
}

/* 2x2 average pooling, stride 2 (H and W even). */
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

/* Bilinear 2x upsampling with align_corners=False (torch
 * F.interpolate(scale_factor=2, mode="bilinear")) of a (C, h, w) tensor
 * into channels coff.. of a (Ctot, 2h, 2w) tensor. */
__kernel void upsample2(__global const float *in, const int h, const int w,
                        __global float *out, const int coff)
{
    const int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    const int W = 2 * w, H = 2 * h;
    float sx, sy, fx, fy;
    int x0, y0, x1, y1;
    const __global float *p;

    if (x >= W || y >= H)
        return;
    sx = fmax((x + 0.5f) * 0.5f - 0.5f, 0.0f);
    sy = fmax((y + 0.5f) * 0.5f - 0.5f, 0.0f);
    x0 = (int)sx;
    y0 = (int)sy;
    x1 = min(x0 + 1, w - 1);
    y1 = min(y0 + 1, h - 1);
    fx = sx - x0;
    fy = sy - y0;
    p = in + (size_t)c * h * w;
    out[((size_t)(c + coff) * H + y) * W + x] =
        (1.0f - fy) * ((1.0f - fx) * p[y0 * w + x0] + fx * p[y0 * w + x1]) +
        fy * ((1.0f - fx) * p[y1 * w + x0] + fx * p[y1 * w + x1]);
}

/* Keypoint candidates of the heatmap (at element off, stride W) cropped
 * to h x w: the
 * cells strictly greater than their 24 neighbours in a 5x5 window and
 * than thresh; the 2-cell border is never a candidate (kornia nms2d). */
__kernel void nms5x5(__global const float *heat_all, const ulong off,
                     const int W, const int h, const int w,
                     const float thresh, __global uchar *mask)
{
    const int x = get_global_id(0), y = get_global_id(1);
    const __global float *heat = heat_all + off;
    float v;
    int dx, dy, keep = 1;

    if (x >= w || y >= h)
        return;
    if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) {
        mask[y * w + x] = 0;
        return;
    }
    v = heat[(size_t)y * W + x];
    for (dy = -2; dy <= 2 && keep; dy++)
        for (dx = -2; dx <= 2; dx++)
            if ((dx || dy) && !(v > heat[(size_t)(y + dy) * W + x + dx])) {
                keep = 0;
                break;
            }
    mask[y * w + x] = keep && v > thresh;
}

/* Descriptors (channels 0..D-1 of a (D + 1, H, W) output) at n keypoints
 * xy (x, y pairs), L2-normalized (F.normalize, eps 1e-12) into (n, D). */
__kernel void gather_desc(__global const float *feat, const int H, const int W,
                          const int D, __global const int *xy, const int n,
                          __global float *desc)
{
    const int k = get_global_id(0);
    const size_t plane = (size_t)H * W;
    size_t off;
    float ss = 0.0f, inv;
    int d;

    if (k >= n)
        return;
    off = (size_t)xy[2 * k + 1] * W + xy[2 * k];
    for (d = 0; d < D; d++) {
        const float v = feat[d * plane + off];

        ss += v * v;
    }
    inv = 1.0f / fmax(sqrt(ss), 1e-12f);
    for (d = 0; d < D; d++)
        desc[(size_t)k * D + d] = feat[d * plane + off] * inv;
}
