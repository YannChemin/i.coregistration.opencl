/* SPDX-License-Identifier: Unlicense */

/* Efficient LoFTR's own kernels: the coarse dual-softmax matching by
 * blocks of rows (never storing the whole score matrix) and the
 * two-stage fine matching. The network itself runs on ops.cl. Follows
 * EfficientLoFTRForKeypointMatching (Hugging Face Transformers, Apache
 * 2.0), itself from the authors' code (Apache 2.0 until 2026-09-15).
 * OpenCL C 1.1. */

/* Log-sum-exp of each row of a block S (R x N, row-major), one
 * work-group of 256 per row, into lse[r0 + r]. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
eloftr_row_lse(__global const float *S, const int N, const int r0,
               __global float *lse)
{
    __local float red[256];
    const int r = get_group_id(0), l = get_local_id(0);
    __global const float *s = S + (long)r * N;
    float m = -INFINITY, sum = 0.0f;
    int j, h;

    for (j = l; j < N; j += 256)
        m = fmax(m, s[j]);
    red[l] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h)
            red[l] = fmax(red[l], red[l + h]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j = l; j < N; j += 256)
        sum += exp(s[j] - m);
    red[l] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h)
            red[l] += red[l + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        lse[r0 + r] = m + log(red[0]);
}

/* Running column maximum and sum of exponentials over blocks of rows:
 * one work-item per column, merging the R rows of this block. */
__kernel void eloftr_col_lse(__global const float *S, const int R,
                             const int N, __global float *cm,
                             __global float *cs)
{
    const int j = get_global_id(0);
    float m = -INFINITY, s = 0.0f, mo, mn;
    int r;

    if (j >= N)
        return;
    for (r = 0; r < R; r++)
        m = fmax(m, S[(long)r * N + j]);
    for (r = 0; r < R; r++)
        s += exp(S[(long)r * N + j] - m);
    mo = cm[j];
    mn = fmax(mo, m);
    cs[j] = (mo == -INFINITY ? 0.0f : cs[j] * exp(mo - mn)) + s * exp(m - mn);
    cm[j] = mn;
}

/* Dual-softmax confidence softmax_row * softmax_col of one score. */
float dual_conf(const float s, const float lr, const float cm, const float cs)
{
    return exp(s - lr) * exp(s - (cm + log(cs)));
}

/* Row maximum and first argmax of the confidence, one work-group per row
 * of the block. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
eloftr_conf_row(__global const float *S, const int N, const int r0,
                __global const float *lse, __global const float *cm,
                __global const float *cs, __global float *rmax,
                __global int *rarg)
{
    __local float vm[256];
    __local int vi[256];
    const int r = get_group_id(0), l = get_local_id(0);
    __global const float *s = S + (long)r * N;
    const float lr = lse[r0 + r];
    float best = -1.0f;
    int bi = 0, j, h;

    for (j = l; j < N; j += 256) {
        const float c = dual_conf(s[j], lr, cm[j], cs[j]);

        if (c > best) {
            best = c;
            bi = j;
        }
    }
    vm[l] = best;
    vi[l] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h &&
            (vm[l + h] > vm[l] || (vm[l + h] == vm[l] && vi[l + h] < vi[l]))) {
            vm[l] = vm[l + h];
            vi[l] = vi[l + h];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        rmax[r0 + r] = vm[0];
        rarg[r0 + r] = vi[0];
    }
}

/* Running column maximum and first argmax of the confidence over blocks
 * of rows (blocks in increasing row order, so ties keep the first). */
__kernel void eloftr_conf_col(__global const float *S, const int R,
                              const int N, const int r0,
                              __global const float *lse,
                              __global const float *cm,
                              __global const float *cs, __global float *cmax,
                              __global int *carg)
{
    const int j = get_global_id(0);
    float best, m, sm;
    int r, bi;

    if (j >= N)
        return;
    best = cmax[j];
    bi = carg[j];
    m = cm[j];
    sm = cs[j];
    for (r = 0; r < R; r++) {
        const float c = dual_conf(S[(long)r * N + j], lse[r0 + r], m, sm);

        if (c > best) {
            best = c;
            bi = r0 + r;
        }
    }
    cmax[j] = best;
    carg[j] = bi;
}

#define FW 8           /* Fine window of image 0. */
#define FW1 (FW + 2)   /* Fine window of image 1 (one cell of padding). */
#define FC 64          /* Fine feature channels. */
#define FS 8           /* Channels of the second stage (the last ones). */
#define NP (FW * FW)   /* 64 positions in image 0. */
#define NQ (FW1 * FW1) /* 100 positions in image 1. */

/* Fine feature c of image f (FC x H x W) at (x, y), zero outside. */
float fine_at(__global const float *f, const int H, const int W, const int c,
              const int x, const int y)
{
    if (x < 0 || y < 0 || x >= W || y >= H)
        return 0.0f;
    return f[((long)c * H + y) * W + x];
}

/* Both fine stages for one coarse match per work-group of 256: the
 * 8 x 8 window of image 0 at cell (x0, y0) against the 10 x 10 window of
 * image 1 around cell (x1, y1), scores on the first FC - FS channels
 * divided by FC (the authors' code scales by the full channel count;
 * Transformers by FC - FS), dual softmax, argmax over the inner 8 x 8
 * of image 1; then the 3 x 3 scores on the last FS channels (image 1's
 * features divided by sqrt(FS)), softmax with temperature T, spatial
 * expectation. As in the authors' code (and Transformers), the 3 x 3
 * window takes the inner 8 x 8 indices of the chosen position as 10 x 10
 * indices: it is centred one pixel up and left of that position, and
 * index -1 wraps to 9 (Python's negative indexing); the network was
 * trained so. Output per match: x0, y0, x1, y1 in pixels, in the
 * reference convention (window position minus FW / 2 - 0.5 from the
 * cell's first pixel). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
eloftr_fine(__global const float *f0, __global const float *f1, const int H,
            const int W, __global const int4 *cells, const int M,
            const float T, __global float4 *out)
{
    __local float sc[NP * NQ];
    __local float pm[NQ], ps[NQ], qm[NP], qs[NP];
    __local float vm[256];
    __local int vi[256];
    __local float s2[9];
    const int k = get_group_id(0), l = get_local_id(0);
    int4 cl;
    int e, p, q, c, h, bp, bq, qi, qj;
    float best = -1.0f, inv = 1.0f / FC;
    int bi = 0;

    if (k >= M)
        return;
    cl = cells[k];

    /* First stage scores. */
    for (e = l; e < NP * NQ; e += 256) {
        const int pp = e / NQ, qq = e % NQ;
        const int ax = cl.x * FW + pp % FW, ay = cl.y * FW + pp / FW;
        const int bx = cl.z * FW - 1 + qq % FW1, by = cl.w * FW - 1 + qq / FW1;
        float d = 0.0f;

        for (c = 0; c < FC - FS; c++)
            d = mad(fine_at(f0, H, W, c, ax, ay), fine_at(f1, H, W, c, bx, by),
                    d);
        sc[e] = d * inv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Softmax statistics over image 0's positions (per q) and over image
     * 1's positions (per p). */
    for (q = l; q < NQ; q += 256) {
        float m = -INFINITY, s = 0.0f;

        for (p = 0; p < NP; p++)
            m = fmax(m, sc[p * NQ + q]);
        for (p = 0; p < NP; p++)
            s += exp(sc[p * NQ + q] - m);
        pm[q] = m;
        ps[q] = s;
    }
    for (p = l; p < NP; p += 256) {
        float m = -INFINITY, s = 0.0f;

        for (q = 0; q < NQ; q++)
            m = fmax(m, sc[p * NQ + q]);
        for (q = 0; q < NQ; q++)
            s += exp(sc[p * NQ + q] - m);
        qm[p] = m;
        qs[p] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Argmax over p and the inner 8 x 8 of image 1, in the flattened
     * order p * 64 + i * 8 + j of the reference. */
    for (e = l; e < NP * FW * FW; e += 256) {
        const int pp = e / (FW * FW), ii = (e / FW) % FW, jj = e % FW;
        const int qq = (ii + 1) * FW1 + jj + 1;
        const float s = sc[pp * NQ + qq];
        const float v =
            exp(s - pm[qq]) / ps[qq] * (exp(s - qm[pp]) / qs[pp]);

        if (v > best) {
            best = v;
            bi = e;
        }
    }
    vm[l] = best;
    vi[l] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (h = 128; h > 0; h >>= 1) {
        if (l < h &&
            (vm[l + h] > vm[l] || (vm[l + h] == vm[l] && vi[l + h] < vi[l]))) {
            vm[l] = vm[l + h];
            vi[l] = vi[l + h];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bp = vi[0] / (FW * FW);
    bq = vi[0] % (FW * FW);
    qi = bq / FW;
    qj = bq % FW;

    /* Second stage: 3 x 3 around 10 x 10 indices (qi, qj) (see above). */
    if (l < 9) {
        const int ii = (qi + l / 3 - 1 + FW1) % FW1;
        const int jj = (qj + l % 3 - 1 + FW1) % FW1;
        const int ax = cl.x * FW + bp % FW, ay = cl.y * FW + bp / FW;
        const int bx = cl.z * FW - 1 + jj, by = cl.w * FW - 1 + ii;
        float d = 0.0f;

        for (c = FC - FS; c < FC; c++)
            d = mad(fine_at(f0, H, W, c, ax, ay), fine_at(f1, H, W, c, bx, by),
                    d);
        s2[l] = d * rsqrt((float)FS) / T;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) {
        float m = -INFINITY, s = 0.0f, ex = 0.0f, ey = 0.0f;
        int t;

        for (t = 0; t < 9; t++)
            m = fmax(m, s2[t]);
        for (t = 0; t < 9; t++)
            s += exp(s2[t] - m);
        for (t = 0; t < 9; t++) {
            const float w = exp(s2[t] - m) / s;

            ex += w * (t % 3 - 1);
            ey += w * (t / 3 - 1);
        }
        out[k] = (float4)(cl.x * FW + bp % FW - (FW / 2 - 0.5f),
                          cl.y * FW + bp / FW - (FW / 2 - 0.5f),
                          cl.z * FW + qj - (FW / 2 - 0.5f) + ex,
                          cl.w * FW + qi - (FW / 2 - 0.5f) + ey);
    }
}
