/* SPDX-License-Identifier: Unlicense */

/* DISK's own OpenCL kernels: the 5x5 non-maxima suppression of the
 * heatmap and the descriptor gather. The U-Net runs on the generic
 * kernels of ops.cl. Tensors are float32, channel-major (C, H, W).
 * OpenCL C 1.1. */

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
