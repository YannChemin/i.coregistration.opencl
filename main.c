/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Co-register an imagery group to a reference group with
 *               DISK local features (on an OpenCL device) matched and
 *               filtered by AdaLAM, writing the matches as the group's
 *               control points for i.rectify (a translation of
 *               RS_coreg_LoFTR to C and OpenCL).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/imagery.h>
#include <grass/raster.h>

#include "adalam.h"
#include "disk.h"
#include "download.h"
#include "gcp.h"
#include "image.h"
#include "ocl_backend.h"
#include "progress.h"
#include "pth_loader.h"

/* Published DISK weights (both 4,375,832 bytes). */
#define DOWNLOAD_URL "https://raw.githubusercontent.com/cvlab-epfl/disk/master"
#define CHECKPOINT_SIZE 4375832LL

/* Refinement windows, in cells: correlation radius and search. */
#define REFINE_RADIUS 7
#define REFINE_SEARCH 2

/* The central folder of DISK checkpoints: $DISK_MODELS_DIR, by default
 * ~/models/disk. */
static char *default_weights_dir(void)
{
    const char *base = getenv("DISK_MODELS_DIR");
    char *dir;

    if (base && *base)
        dir = G_store(base);
    else
        G_asprintf(&dir, "%s/models/disk", G_home());
    return dir;
}

static int file_exists(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

/* Tile (x0, y0, w, h) of both images as DISK input: (3, H, W) floats in
 * 0..1 zero-padded to H x W, and the tile's validity. */
static void tile_input(const struct rgb_image *img, int x0, int y0, int w,
                       int h, int H, int W, float *out, unsigned char *valid)
{
    size_t plane = (size_t)H * W;
    int x, y, b;

    memset(out, 0, 3 * plane * sizeof(float));
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            size_t i = (size_t)(y0 + y) * img->cols + x0 + x;

            valid[(size_t)y * w + x] = img->valid[i];
            for (b = 0; b < 3; b++)
                out[b * plane + (size_t)y * W + x] = img->rgb[3 * i + b] / 255.0f;
        }
}

/* Developer aid: with I_COREGISTRATION_OPENCL_DUMP set to a folder, each
 * tile's DISK inputs (3 x H x W float32 per image), keypoints (x, y,
 * score, then the descriptor, float32 per keypoint) and AdaLAM matches
 * (i1 i2 dist) are written there, for tests/reference_compare.py. */
static void dump_tile(const char *dir, int t, const char *what,
                      const void *data, size_t bytes)
{
    char path[GPATH_MAX];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/tile%03d_%s", dir, t, what);
    if (!(fp = fopen(path, "wb")) || fwrite(data, 1, bytes, fp) != bytes)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fclose(fp);
}

static void dump_keypoints(const char *dir, int t, const char *what,
                           const int *xy, const float *score, const float *desc,
                           int n)
{
    size_t row = 3 + DISK_DESC;
    float *v = G_malloc((n + 1) * row * sizeof(float));
    int i;

    for (i = 0; i < n; i++) {
        v[i * row] = xy[2 * i];
        v[i * row + 1] = xy[2 * i + 1];
        v[i * row + 2] = score[i];
        memcpy(v + i * row + 3, desc + (size_t)i * DISK_DESC,
               DISK_DESC * sizeof(float));
    }
    dump_tile(dir, t, what, v, n * row * sizeof(float));
    G_free(v);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;

    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    if (n == 0)
        return 0;
    qsort(v, n, sizeof(double), cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *group, *subgroup, *reference, *ref_subgroup,
            *checkpoint, *weights_dir, *model, *points, *percentile,
            *features, *tile_size, *tile_overlap, *area_ratio,
            *search_expansion, *ransac_iters, *min_inliers, *min_confidence,
            *max_shift, *tolerance, *device, *platform;
    } opt;
    struct {
        struct Flag *mnn, *refine, *dry, *offline;
    } flag;
    struct Cell_head win;
    struct rgb_image src, ref;
    struct ocl_backend ocl;
    struct pth_file pf;
    struct disk_model disk;
    struct adalam_config cfg;
    struct gcp_list gl = {0, 0, NULL};
    char *ckpt = NULL;
    int stretch = 0, tile, overlap, step, nfeat, ntx, nty, tx, ty, t = 0;
    int H, W, nraw = 0, nshift = 0, ntol = 0, nref = 0;
    double lo = 0, hi = 100, max_shift = 0, tol = 0;
    float *img1, *img2, *k1, *k2, *s1, *s2, *d1, *d2;
    int *xy1, *xy2;
    unsigned char *v1, *v2;
    const char *dump;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("co-registration"));
    G_add_keyword(_("geometry"));
    G_add_keyword(_("ground control points"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("GPU"));
    module->description =
        _("Finds control points between an imagery group and a reference "
          "group with DISK local features matched and filtered by AdaLAM, on "
          "an OpenCL device, and writes them to the group for i.rectify.");

    opt.group = G_define_standard_option(G_OPT_I_GROUP);
    opt.group->description =
        _("Imagery group to co-register (three maps used as R, G, B); its "
          "control points are written");

    opt.subgroup = G_define_standard_option(G_OPT_I_SUBGROUP);
    opt.subgroup->required = NO;
    opt.subgroup->description =
        _("Subgroup of group holding the three maps to match");

    opt.reference = G_define_standard_option(G_OPT_I_GROUP);
    opt.reference->key = "reference";
    opt.reference->description =
        _("Reference imagery group (three maps used as R, G, B)");

    opt.ref_subgroup = G_define_standard_option(G_OPT_I_SUBGROUP);
    opt.ref_subgroup->key = "reference_subgroup";
    opt.ref_subgroup->required = NO;
    opt.ref_subgroup->description =
        _("Subgroup of reference holding the three maps to match");

    opt.points = G_define_standard_option(G_OPT_V_OUTPUT);
    opt.points->key = "points";
    opt.points->required = NO;
    opt.points->description =
        _("Name for output vector map of the control points (at their "
          "reference positions, with source position and shift)");

    opt.checkpoint = G_define_standard_option(G_OPT_F_INPUT);
    opt.checkpoint->key = "checkpoint";
    opt.checkpoint->required = NO;
    opt.checkpoint->description =
        _("DISK checkpoint (depth-save.pth or epipolar-save.pth); default: "
          "the one of model= in weights_dir=, downloaded if missing");
    opt.checkpoint->guisection = _("Model");

    opt.weights_dir = G_define_standard_option(G_OPT_M_DIR);
    opt.weights_dir->key = "weights_dir";
    opt.weights_dir->required = NO;
    opt.weights_dir->description =
        _("Folder of DISK checkpoints (default: $DISK_MODELS_DIR or "
          "~/models/disk)");
    opt.weights_dir->guisection = _("Model");

    opt.model = G_define_option();
    opt.model->key = "model";
    opt.model->type = TYPE_STRING;
    opt.model->options = "depth,epipolar";
    opt.model->answer = "depth";
    opt.model->description = _("DISK weights");
    G_asprintf((char **)&opt.model->descriptions,
               "depth;%s;epipolar;%s",
               _("Trained with depth supervision, more precise (as the "
                 "repository)"),
               _("Trained with epipolar supervision, keypoints everywhere "
                 "matchable"));
    opt.model->guisection = _("Model");

    opt.features = G_define_option();
    opt.features->key = "features";
    opt.features->type = TYPE_INTEGER;
    opt.features->answer = "2048";
    opt.features->options = "16-65536";
    opt.features->description = _("Keypoints per image and tile");
    opt.features->guisection = _("Model");

    opt.tile_size = G_define_option();
    opt.tile_size->key = "tile_size";
    opt.tile_size->type = TYPE_INTEGER;
    opt.tile_size->answer = "1024";
    opt.tile_size->options = "64-8192";
    opt.tile_size->description =
        _("Side of the tiles matched one by one, in cells (rounded up to a "
          "multiple of 16)");
    opt.tile_size->guisection = _("Model");

    opt.tile_overlap = G_define_option();
    opt.tile_overlap->key = "tile_overlap";
    opt.tile_overlap->type = TYPE_INTEGER;
    opt.tile_overlap->answer = "64";
    opt.tile_overlap->description = _("Overlap of neighbouring tiles, in cells");
    opt.tile_overlap->guisection = _("Model");

    opt.percentile = G_define_option();
    opt.percentile->key = "percentile";
    opt.percentile->type = TYPE_DOUBLE;
    opt.percentile->key_desc = "low,high";
    opt.percentile->required = NO;
    opt.percentile->description =
        _("Stretch every band between these percentiles (default: 8-bit bands "
          "as they are, others between 2 and 98)");

    opt.area_ratio = G_define_option();
    opt.area_ratio->key = "area_ratio";
    opt.area_ratio->type = TYPE_DOUBLE;
    opt.area_ratio->answer = "100";
    opt.area_ratio->description =
        _("AdaLAM: tile area over seed circle area (more seeds when higher)");
    opt.area_ratio->guisection = _("AdaLAM");

    opt.search_expansion = G_define_option();
    opt.search_expansion->key = "search_expansion";
    opt.search_expansion->type = TYPE_DOUBLE;
    opt.search_expansion->answer = "16";
    opt.search_expansion->description =
        _("AdaLAM: neighbourhood radius over seed radius");
    opt.search_expansion->guisection = _("AdaLAM");

    opt.ransac_iters = G_define_option();
    opt.ransac_iters->key = "ransac_iters";
    opt.ransac_iters->type = TYPE_INTEGER;
    opt.ransac_iters->answer = "256";
    opt.ransac_iters->options = "1-100000";
    opt.ransac_iters->description = _("AdaLAM: hypotheses per neighbourhood");
    opt.ransac_iters->guisection = _("AdaLAM");

    opt.min_inliers = G_define_option();
    opt.min_inliers->key = "min_inliers";
    opt.min_inliers->type = TYPE_INTEGER;
    opt.min_inliers->answer = "6";
    opt.min_inliers->description =
        _("AdaLAM: inliers a neighbourhood must keep");
    opt.min_inliers->guisection = _("AdaLAM");

    opt.min_confidence = G_define_option();
    opt.min_confidence->key = "min_confidence";
    opt.min_confidence->type = TYPE_DOUBLE;
    opt.min_confidence->answer = "200";
    opt.min_confidence->description =
        _("AdaLAM: threshold of the confidence-based RANSAC");
    opt.min_confidence->guisection = _("AdaLAM");

    opt.max_shift = G_define_option();
    opt.max_shift->key = "max_shift";
    opt.max_shift->type = TYPE_DOUBLE;
    opt.max_shift->required = NO;
    opt.max_shift->description =
        _("Drop matches shifted by more than this, in map units (objects "
          "that moved between dates)");
    opt.max_shift->guisection = _("Filter");

    opt.tolerance = G_define_option();
    opt.tolerance->key = "tolerance";
    opt.tolerance->type = TYPE_DOUBLE;
    opt.tolerance->required = NO;
    opt.tolerance->description =
        _("Drop matches whose shift differs from the median shift of their "
          "tile by more than this along either axis, in map units");
    opt.tolerance->guisection = _("Filter");

    opt.device = G_define_option();
    opt.device->key = "device";
    opt.device->type = TYPE_STRING;
    opt.device->options = "auto,gpu,cpu";
    opt.device->answer = "auto";
    opt.device->description = _("OpenCL device type");
    opt.device->guisection = _("OpenCL");

    opt.platform = G_define_option();
    opt.platform->key = "platform";
    opt.platform->type = TYPE_STRING;
    opt.platform->required = NO;
    opt.platform->description =
        _("Use only OpenCL platforms whose name contains this text (e.g. "
          "rusticl, Portable)");
    opt.platform->guisection = _("OpenCL");

    flag.mnn = G_define_flag();
    flag.mnn->key = 'm';
    flag.mnn->description =
        _("AdaLAM: take seeds only among mutual nearest neighbours");
    flag.mnn->guisection = _("AdaLAM");

    flag.refine = G_define_flag();
    flag.refine->key = 's';
    flag.refine->description =
        _("Refine matches to a fraction of a cell by normalized "
          "cross-correlation");
    flag.refine->guisection = _("Filter");

    flag.dry = G_define_flag();
    flag.dry->key = 'n';
    flag.dry->description =
        _("Do not write the control points to the group (report and "
          "points= only)");

    flag.offline = G_define_flag();
    flag.offline->key = 'o';
    flag.offline->description =
        _("Do not download a missing checkpoint (fail instead)");
    flag.offline->guisection = _("Model");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    if (flag.dry->answer && !opt.points->answer)
        G_fatal_error(_("With -n, give points= or nothing is written"));
    if (opt.percentile->answer) {
        if (!opt.percentile->answers[1])
            G_fatal_error(_("percentile= needs two values, low,high"));
        lo = atof(opt.percentile->answers[0]);
        hi = atof(opt.percentile->answers[1]);
        if (!(lo >= 0 && hi <= 100 && lo < hi))
            G_fatal_error(_("percentile= must satisfy 0 <= low < high <= 100"));
        stretch = 1;
    }
    tile = atoi(opt.tile_size->answer);
    tile = (tile + DISK_MULTIPLE - 1) / DISK_MULTIPLE * DISK_MULTIPLE;
    overlap = atoi(opt.tile_overlap->answer);
    if (overlap < 0 || overlap >= tile / 2)
        G_fatal_error(_("tile_overlap= must lie in 0..%d"), tile / 2 - 1);
    step = tile - overlap;
    nfeat = atoi(opt.features->answer);
    if (opt.max_shift->answer && (max_shift = atof(opt.max_shift->answer)) <= 0)
        G_fatal_error(_("max_shift= must be positive"));
    if (opt.tolerance->answer && (tol = atof(opt.tolerance->answer)) <= 0)
        G_fatal_error(_("tolerance= must be positive"));
    adalam_defaults(&cfg);
    cfg.area_ratio = atof(opt.area_ratio->answer);
    cfg.search_expansion = atof(opt.search_expansion->answer);
    cfg.ransac_iters = atoi(opt.ransac_iters->answer);
    cfg.min_inliers = atoi(opt.min_inliers->answer);
    cfg.min_confidence = atof(opt.min_confidence->answer);
    cfg.force_seed_mnn = flag.mnn->answer;
    if (cfg.area_ratio <= 0 || cfg.search_expansion <= 0)
        G_fatal_error(_("area_ratio= and search_expansion= must be positive"));

    /* The checkpoint, before any long reading. */
    if (opt.checkpoint->answer) {
        ckpt = G_store(opt.checkpoint->answer);
        if (!file_exists(ckpt))
            G_fatal_error(_("Checkpoint <%s> not found"), ckpt);
    }
    else {
        char *dir = opt.weights_dir->answer ? G_store(opt.weights_dir->answer)
                                            : default_weights_dir();

        G_asprintf(&ckpt, "%s/%s-save.pth", dir, opt.model->answer);
        if (!file_exists(ckpt)) {
            char *url;

            if (flag.offline->answer)
                G_fatal_error(_("Checkpoint <%s> not found and -o given"),
                              ckpt);
            G_asprintf(&url, "%s/%s-save.pth", DOWNLOAD_URL,
                       opt.model->answer);
            G_message(_("Downloading DISK weights <%s>..."), url);
            download_file(url, ckpt, CHECKPOINT_SIZE);
            G_free(url);
        }
        G_free(dir);
    }

    G_get_window(&win);
    ntx = win.cols <= tile ? 1 : 1 + (win.cols - tile + step - 1) / step;
    nty = win.rows <= tile ? 1 : 1 + (win.rows - tile + step - 1) / step;
    progress_plan(PG_READ, 2.0 * win.rows * win.cols / 1e6);
    progress_plan(PG_LOAD, CHECKPOINT_SIZE / 1073741824.0);
    progress_plan(PG_BUILD, 1);
    progress_plan(PG_TILE, ntx * nty);
    progress_plan(PG_MATCH, ntx * nty);
    progress_plan(PG_WRITE, 1);

    image_read_group(opt.group->answer, opt.subgroup->answer, stretch, lo, hi,
                     &src);
    image_read_group(opt.reference->answer, opt.ref_subgroup->answer, stretch,
                     lo, hi, &ref);

    progress_begin(PG_BUILD, 1);
    ocl_init(&ocl, opt.device->answer, opt.platform->answer);
    progress_end();
    pth_open(&pf, ckpt);
    disk_load(&disk, &ocl, &pf);
    pth_close(&pf);

    H = win.rows < tile ? (win.rows + 15) / 16 * 16 : tile;
    W = win.cols < tile ? (win.cols + 15) / 16 * 16 : tile;
    disk_alloc(&disk, H, W, nfeat);
    img1 = G_malloc(3 * (size_t)H * W * sizeof(float));
    img2 = G_malloc(3 * (size_t)H * W * sizeof(float));
    v1 = G_malloc((size_t)H * W);
    v2 = G_malloc((size_t)H * W);
    xy1 = G_malloc(2 * nfeat * sizeof(int));
    xy2 = G_malloc(2 * nfeat * sizeof(int));
    k1 = G_malloc(2 * nfeat * sizeof(float));
    k2 = G_malloc(2 * nfeat * sizeof(float));
    s1 = G_malloc(nfeat * sizeof(float));
    s2 = G_malloc(nfeat * sizeof(float));
    d1 = G_malloc((size_t)nfeat * DISK_DESC * sizeof(float));
    d2 = G_malloc((size_t)nfeat * DISK_DESC * sizeof(float));

    G_message(_("Matching %d tile(s) of %d x %d cells..."), ntx * nty, tile,
              tile);
    dump = getenv("I_COREGISTRATION_OPENCL_DUMP");
    if (dump && !*dump)
        dump = NULL;
    for (ty = 0; ty < nty; ty++)
        for (tx = 0; tx < ntx; tx++) {
            int x0 = tx * step, y0 = ty * step, w, h, n1, n2, nm, *pairs, i;
            int first = gl.n;
            float *dist;
            double *ddx, *ddy, mdx = 0, mdy = 0;

            /* The last tiles end on the region's edge. */
            if (x0 + tile > win.cols)
                x0 = win.cols > tile ? win.cols - tile : 0;
            if (y0 + tile > win.rows)
                y0 = win.rows > tile ? win.rows - tile : 0;
            w = win.cols - x0 < tile ? win.cols - x0 : tile;
            h = win.rows - y0 < tile ? win.rows - y0 : tile;
            t++;

            progress_begin(PG_TILE, 1);
            tile_input(&src, x0, y0, w, h, H, W, img1, v1);
            tile_input(&ref, x0, y0, w, h, H, W, img2, v2);
            n1 = disk_detect(&disk, img1, H, W, h, w, nfeat, 0.0f, v1, xy1, s1,
                             d1);
            n2 = disk_detect(&disk, img2, H, W, h, w, nfeat, 0.0f, v2, xy2, s2,
                             d2);
            progress_end();

            progress_begin(PG_MATCH, 1);
            for (i = 0; i < 2 * n1; i++)
                k1[i] = xy1[i];
            for (i = 0; i < 2 * n2; i++)
                k2[i] = xy2[i];
            nm = adalam_match(k1, d1, n1, k2, d2, n2, DISK_DESC, &cfg, &pairs,
                              &dist);
            nraw += nm;
            if (dump) {
                char info[GPATH_MAX];
                FILE *fp;

                dump_tile(dump, t, "src_input.f32", img1,
                          3 * (size_t)H * W * sizeof(float));
                dump_tile(dump, t, "ref_input.f32", img2,
                          3 * (size_t)H * W * sizeof(float));
                dump_keypoints(dump, t, "src_kp.f32", xy1, s1, d1, n1);
                dump_keypoints(dump, t, "ref_kp.f32", xy2, s2, d2, n2);
                snprintf(info, sizeof(info), "%s/tile%03d_matches.txt", dump, t);
                if (!(fp = fopen(info, "w")))
                    G_fatal_error(_("Unable to write debug dump <%s>"), info);
                fprintf(fp, "# x0 %d y0 %d w %d h %d H %d W %d\n", x0, y0, w,
                        h, H, W);
                for (i = 0; i < nm; i++)
                    fprintf(fp, "%d %d %.9g\n", pairs[2 * i], pairs[2 * i + 1],
                            dist[i]);
                fclose(fp);
            }
            G_debug(1, "tile %d (%d, %d): %d + %d keypoints, %d matches", t,
                    x0, y0, n1, n2, nm);

            for (i = 0; i < nm; i++) {
                struct gcp g;
                int a = pairs[2 * i], b = pairs[2 * i + 1];

                g.xs = x0 + xy1[2 * a];
                g.ys = y0 + xy1[2 * a + 1];
                g.xr = x0 + xy2[2 * b];
                g.yr = y0 + xy2[2 * b + 1];
                g.dist = dist[i];
                g.tile = t;
                g.refined = 0;
                g.n = 1;
                if (max_shift > 0 &&
                    hypot((g.xr - g.xs) * win.ew_res,
                          (g.yr - g.ys) * win.ns_res) > max_shift) {
                    nshift++;
                    continue;
                }
                gcp_add(&gl, &g);
            }
            G_free(pairs);
            G_free(dist);

            /* Matches far from their tile's median shift. */
            if (tol > 0 && gl.n > first) {
                int k = 0, n = gl.n - first;

                ddx = G_malloc(n * sizeof(double));
                ddy = G_malloc(n * sizeof(double));
                for (i = first; i < gl.n; i++) {
                    ddx[i - first] = gl.v[i].xr - gl.v[i].xs;
                    ddy[i - first] = gl.v[i].yr - gl.v[i].ys;
                }
                mdx = median(ddx, n);
                mdy = median(ddy, n);
                for (i = first; i < gl.n; i++) {
                    struct gcp *g = &gl.v[i];

                    if (fabs((g->xr - g->xs - mdx) * win.ew_res) > tol ||
                        fabs((g->yr - g->ys - mdy) * win.ns_res) > tol) {
                        ntol++;
                        continue;
                    }
                    gl.v[first + k++] = *g;
                }
                gl.n = first + k;
                G_free(ddx);
                G_free(ddy);
            }
            if (flag.refine->answer)
                for (i = first; i < gl.n; i++)
                    nref += gcp_refine(&gl.v[i], &src, &ref, REFINE_RADIUS,
                                       REFINE_SEARCH);
            progress_end();
        }

    disk_free(&disk);
    ocl_free(&ocl);
    G_free(img1);
    G_free(img2);
    G_free(v1);
    G_free(v2);
    G_free(xy1);
    G_free(xy2);
    G_free(k1);
    G_free(k2);
    G_free(s1);
    G_free(s2);
    G_free(d1);
    G_free(d2);

    progress_begin(PG_WRITE, 1);
    {
        int before = gl.n;

        gcp_merge(&gl);
        G_message(_("%d AdaLAM matches; %d dropped by max_shift=, %d by "
                    "tolerance=; %d refined; %d control points after "
                    "merging %d sharing a reference cell"),
                  nraw, nshift, ntol, nref, gl.n, before - gl.n);
    }
    if (gl.n > 0) {
        double *dx = G_malloc(gl.n * sizeof(double));
        double *dy = G_malloc(gl.n * sizeof(double));
        double mx, my, adx, ady;
        int i;

        for (i = 0; i < gl.n; i++) {
            dx[i] = (gl.v[i].xr - gl.v[i].xs) * win.ew_res;
            dy[i] = -(gl.v[i].yr - gl.v[i].ys) * win.ns_res;
        }
        mx = median(dx, gl.n);
        my = median(dy, gl.n);
        for (i = 0; i < gl.n; i++) {
            dx[i] = fabs(dx[i] - mx);
            dy[i] = fabs(dy[i] - my);
        }
        adx = median(dx, gl.n);
        ady = median(dy, gl.n);
        G_message(_("Shift source -> reference: median east %.3f, north "
                    "%.3f map units (median absolute deviation %.3f, %.3f)"),
                  mx, my, adx, ady);
        G_free(dx);
        G_free(dy);
    }
    else
        G_warning(_("No control point found"));

    if (!flag.dry->answer && gl.n > 0)
        gcp_write_group(&gl, opt.group->answer, &win);
    if (opt.points->answer)
        gcp_write_vector(&gl, opt.points->answer, &win);
    progress_end();
    progress_finish();

    image_free(&src);
    image_free(&ref);
    G_free(gl.v);
    G_free(ckpt);
    exit(EXIT_SUCCESS);
}
