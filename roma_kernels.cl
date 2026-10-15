/* SPDX-License-Identifier: Unlicense */

/* RoMa's own kernels (see roma.c): Gaussian process kernel, coarse warp
 * from the anchor classes, sampling and local correlation at the
 * current warp, displacement input and warp update of the refiners.
 * Warps are channel-major (2, n): x then y, normalized to -1..1 with
 * align_corners=False (cell centres at -1 + (2 i + 1) / size). */

/* Euclidean norm of each row of X (rows x C). */
__kernel void roma_row_norm(__global const float *X, const int rows,
                            const int C, __global float *out)
{
    const int r = get_global_id(0);
    __global const float *x = X + (size_t)r * C;
    float s = 0.0f;
    int c;

    if (r >= rows)
        return;
    for (c = 0; c < C; c++)
        s += x[c] * x[c];
    out[r] = sqrt(s);
}

/* RoMa's CosKernel in place on the dot products S (N x M):
 * exp((S / (|x_i| |y_j| + eps) - 1) / T). */
__kernel void roma_cos_exp(__global float *S, __global const float *nx,
                           __global const float *ny, const int N, const int M,
                           const float T, const float eps)
{
    const int j = get_global_id(0), i = get_global_id(1);
    float c;

    if (i >= N || j >= M)
        return;
    c = S[(size_t)i * M + j] / (nx[i] * ny[j] + eps);
    S[(size_t)i * M + j] = exp((c - 1.0f) / T);
}

/* cls_to_flow_refine: per token (row of L, ld floats, the first C =
 * res * res the anchor logits, then the certainty logit), softmax over
 * the anchors, its mode and the four neighbours (indices clamped to
 * 0..C-1 as the reference, so that a clamped neighbour counts twice),
 * and their probability-weighted mean position. One work-group of 256
 * per token. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
roma_cls_to_flow(__global const float *L, const int ld, const int C,
                 const int res, const int n, __global float *flow,
                 __global float *cert)
{
    __local float vm[256], vs[256];
    __local int vi[256];
    const int t = get_group_id(0), l = get_local_id(0);
    __global const float *p = L + (size_t)t * ld;
    float m = -INFINITY, s = 0.0f;
    int arg = 0, j, k;

    for (j = l; j < C; j += 256)
        if (p[j] > m) {
            m = p[j];
            arg = j;
        }
    vm[l] = m;
    vi[l] = arg;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k = 128; k > 0; k >>= 1) {
        if (l < k && (vm[l + k] > vm[l] ||
                      (vm[l + k] == vm[l] && vi[l + k] < vi[l]))) {
            vm[l] = vm[l + k];
            vi[l] = vi[l + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = vm[0];
    for (j = l; j < C; j += 256)
        s += exp(p[j] - m);
    barrier(CLK_LOCAL_MEM_FENCE);
    vs[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k = 128; k > 0; k >>= 1) {
        if (l < k)
            vs[l] += vs[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        const int mode = vi[0];
        const int nb[5] = {mode - 1, mode, mode + 1, mode - res, mode + res};
        float fx = 0.0f, fy = 0.0f, tot = 0.0f;

        for (k = 0; k < 5; k++) {
            const int c = clamp(nb[k], 0, C - 1);
            const float pr = exp(p[c] - m) / vs[0];

            fx += pr * (-1.0f + (2 * (c % res) + 1) / (float)res);
            fy += pr * (-1.0f + (2 * (c / res) + 1) / (float)res);
            tot += pr;
        }
        flow[t] = fx / tot;
        flow[n + t] = fy / tot;
        cert[t] = p[C];
    }
}

/* Bilinear value of plane p (H x W) at normalized (x, y),
 * align_corners=False, zero padding (F.grid_sample). */
float bilinear0(__global const float *p, const int H, const int W,
                const float x, const float y)
{
    const float ix = ((x + 1.0f) * W - 1.0f) * 0.5f;
    const float iy = ((y + 1.0f) * H - 1.0f) * 0.5f;
    const int x0 = (int)floor(ix), y0 = (int)floor(iy);
    const float fx = ix - x0, fy = iy - y0;
    float v00, v01, v10, v11;

    if (!isfinite(ix) || !isfinite(iy))
        return 0.0f;
#define AT(yy, xx)                                                             \
    ((xx) >= 0 && (xx) < W && (yy) >= 0 && (yy) < H ? p[(yy) * W + (xx)]       \
                                                    : 0.0f)
    v00 = AT(y0, x0);
    v01 = AT(y0, x0 + 1);
    v10 = AT(y0 + 1, x0);
    v11 = AT(y0 + 1, x0 + 1);
#undef AT
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
           fy * ((1.0f - fx) * v10 + fx * v11);
}

/* x_hat: in (C, H, W) sampled at the warp flow (2, n) of the same grid
 * (n = H W), into channels coff.. of out (*, n). */
__kernel void roma_warp_sample(__global const float *in, const int C,
                               const int H, const int W,
                               __global const float *flow,
                               __global float *out, const int coff)
{
    const int i = get_global_id(0), c = get_global_id(1), n = H * W;

    if (i >= n || c >= C)
        return;
    out[(size_t)(coff + c) * n + i] =
        bilinear0(in + (size_t)c * n, H, W, flow[i], flow[n + i]);
}

/* Local correlation in the other image (RoMa's local_correlation): for
 * window offset k = (a, b) of (2r + 1)^2, a the row, the dot product of
 * f1 (C, n) / sqrt(C) with f2 (C, H, W) sampled (bilinear, zero
 * padding) at the warp shifted by (b - r, a - r) cells, into channels
 * coff + k of out. The offsets are whole cells, so every sample of a
 * cell shares its bilinear weights: the dot products are first taken at
 * the (2r + 2)^2 whole cells around the warp (local memory), then
 * combined with those weights. One work-group of 64 per cell; C <=
 * 512, r <= 7. */
#define CORR_WG 64
__kernel __attribute__((reqd_work_group_size(CORR_WG, 1, 1))) void
roma_local_corr(__global const float *f1, __global const float *f2,
                const int C, const int H, const int W, const int r,
                __global const float *flow, __global float *out,
                const int coff)
{
    __local float a1[512], S[256];
    const int i = get_group_id(0), l = get_local_id(0), n = H * W;
    const int side = 2 * r + 1, ps = 2 * r + 2;
    const float rc = rsqrt((float)C);
    const float ix = ((flow[i] + 1.0f) * W - 1.0f) * 0.5f;
    const float iy = ((flow[n + i] + 1.0f) * H - 1.0f) * 0.5f;
    const int ok = isfinite(ix) && isfinite(iy);
    const int x0 = ok ? (int)floor(ix) : 0, y0 = ok ? (int)floor(iy) : 0;
    const float fx = ok ? ix - x0 : 0.0f, fy = ok ? iy - y0 : 0.0f;
    int c, e, k;

    for (c = l; c < C; c += CORR_WG)
        a1[c] = f1[(size_t)c * n + i] * rc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (e = l; e < ps * ps; e += CORR_WG) {
        const int yy = y0 - r + e / ps, xx = x0 - r + e % ps;
        float s = 0.0f;

        if (ok && xx >= 0 && xx < W && yy >= 0 && yy < H) {
            __global const float *p = f2 + (size_t)yy * W + xx;

            for (c = 0; c < C; c++)
                s += a1[c] * p[(size_t)c * n];
        }
        S[e] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k = l; k < side * side; k += CORR_WG) {
        const int e0 = (k / side) * ps + k % side, e1 = e0 + ps;

        out[(size_t)(coff + k) * n + i] =
            (1.0f - fy) * ((1.0f - fx) * S[e0] + fx * S[e0 + 1]) +
            fy * ((1.0f - fx) * S[e1] + fx * S[e1 + 1]);
    }
}

/* Displacement input of a refiner: (fx, fy) (warp - own cell centre),
 * (2, n) for an H x W grid. */
__kernel void roma_disp(__global const float *flow, const int H, const int W,
                        const float fx, const float fy, __global float *out)
{
    const int i = get_global_id(0), n = H * W;
    float gx, gy;

    if (i >= n)
        return;
    gx = -1.0f + 1.0f / W + (i % W) * (2.0f - 2.0f / W) / (W - 1);
    gy = -1.0f + 1.0f / H + (i / W) * (2.0f - 2.0f / H) / (H - 1);
    out[i] = fx * (flow[i] - gx);
    out[n + i] = fy * (flow[n + i] - gy);
}

/* Warp and certainty update from the refiner output d (3, n): the
 * displacement in units of 1 / (4 Wf), 1 / (4 Hf) of the pass's full
 * size, times the scale. */
__kernel void roma_update(__global float *flow, __global float *cert,
                          __global const float *d, const int n,
                          const float scale, const float Wf, const float Hf)
{
    const int i = get_global_id(0);

    if (i >= n)
        return;
    flow[i] += scale * (d[i] / (4.0f * Wf));
    flow[n + i] += scale * (d[n + i] / (4.0f * Hf));
    cert[i] += d[2 * n + i];
}

/* RoMa v2's update from the refiner output d (6, n) on a W x H grid:
 * warp += (dx, dy) / (4 W, 4 H); confidence (4, n) += the overlap logit
 * and the precision (p00, p10, p11) of the Cholesky factor (softplus
 * diagonal, as torch's softplus, threshold 20). */
float softplus20(const float x) { return x > 20.0f ? x : log1p(exp(x)); }

__kernel void roma2_update(__global float *flow, __global float *conf,
                           __global const float *d, const int n,
                           const float W, const float H)
{
    const int i = get_global_id(0);
    float l00, l10, l11;

    if (i >= n)
        return;
    flow[i] += d[i] / (4.0f * W);
    flow[n + i] += d[n + i] / (4.0f * H);
    l00 = softplus20(d[3 * n + i]) + 1e-6f;
    l10 = d[4 * n + i];
    l11 = softplus20(d[5 * n + i]) + 1e-6f;
    conf[i] += d[2 * n + i];
    conf[n + i] += l00 * l00;
    conf[2 * n + i] += l00 * l10;
    conf[3 * n + i] += l10 * l10 + l11 * l11;
}

/* Depthwise 5 x 5 convolution, stride 1, zero padding 2, bias, ReLU
 * (the refiners' first convolution with its BatchNorm folded): one
 * work-group of 16 x 16 outputs of one channel, its 20 x 20 input patch
 * in local memory. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
roma_dwconv5(__global const float *in, const int H, const int W,
             __global const float *wt, __global const float *bias,
             __global float *out)
{
    __local float t[20][21];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int x0 = get_group_id(0) * 16, y0 = get_group_id(1) * 16;
    const int c = get_global_id(2), x = x0 + lx, y = y0 + ly;
    __global const float *p = in + (size_t)c * H * W;
    __global const float *w = wt + c * 25;
    float acc;
    int i, j, e;

    for (e = ly * 16 + lx; e < 400; e += 256) {
        const int yy = y0 - 2 + e / 20, xx = x0 - 2 + e % 20;

        t[e / 20][e % 20] =
            yy >= 0 && yy < H && xx >= 0 && xx < W ? p[yy * W + xx] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= W || y >= H)
        return;
    acc = bias[c];
    for (i = 0; i < 5; i++)
        for (j = 0; j < 5; j++)
            acc += t[ly + i][lx + j] * w[i * 5 + j];
    out[((size_t)c * H + y) * W + x] = fmax(acc, 0.0f);
}
