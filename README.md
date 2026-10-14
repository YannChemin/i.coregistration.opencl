# i.coregistration.opencl

GRASS GIS addon that finds control points between an imagery group and a
reference group with DISK local features matched and filtered by AdaLAM,
on an OpenCL device, and writes them as the group's control points for
*i.rectify* (thin plate spline with `-t`) or *v.rectify*. It is a C and
OpenCL translation of
[RS_coreg_LoFTR](https://github.com/YannChemin/RS_coreg_LoFTR) (kornia
DISK + AdaLAM, then `gdalwarp -tps`): no Python or PyTorch at run time,
the published DISK weights are read directly (legacy PyTorch format).

See `i.coregistration.opencl.md` for the manual, `TODO.md` for the
planned matchers (Efficient LoFTR, RoMa v2, MatchAnything) and robust
estimators (MAGSAC++, Graph-Cut RANSAC, VSAC, SupeRANSAC, dense match
summarization), and `PLAN.md` for the order of the work.

## Layout

- `main.c`: options, tiles, filters, refinement, output.
- `matcher.h`, `matcher.c`: the matcher interface (device stage, host
  matching, dumps); `matcher_disk.c`: DISK + AdaLAM behind it.
- `ops.cl`, `ops.c`, `ops.h`: generic tensor kernels shared by the
  matchers (GEMM, convolutions, normalizations, activations, softmax,
  blockwise attention, RoPE, bilinear resizing, `grid_sample`, unfold,
  pooling, transpose).
- `models.c`: similarity, affinity, homography, polynomials of order 2
  and 3 (weighted least squares, leave-one-out residuals);
  `robust.c`: RANSAC, MAGSAC++, Graph-Cut RANSAC and their combination
  (PROSAC, independent inliers); `maxflow.c`: minimum cut (Dinic);
  `estimate.c`: estimation over control points, per tile and region.
- `disk.c`, `disk_kernels.cl`: the DISK network and its own kernels;
  `adalam.c`: AdaLAM; `gcp.c`: control points; `image.c`: input bands.
- `ocl_backend.c`, `pth_loader.c`, `download.c`, `progress.c`: device,
  checkpoints, downloads, progress (from i.sam.opencl).

## Build

```sh
make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL
```

(or `MODULE_TOPDIR=$HOME/dev/grass` for a source tree). Needs the OpenCL
headers and ICD loader, libcurl (download of the weights) and OpenMP.

## Weights

`depth-save.pth` (default) or `epipolar-save.pth` from
<https://github.com/cvlab-epfl/disk>, looked for in `$DISK_MODELS_DIR` or
`~/models/disk` and downloaded there when missing (`-o` to forbid).

## Use

```sh
g.region raster=esri.1
i.coregistration.opencl group=google reference=esri points=gcp_google \
    -s max_shift=5 tolerance=2
i.rectify -t group=google extension=_coreg method=linear -c
```

## Tests

```sh
I_COREGISTRATION_OPENCL_CHECKPOINT=~/models/disk/depth-save.pth \
    grass --tmp-project XY --exec python3 -m pytest tests
```

Without the checkpoint variable, only the argument checks run.

The generic kernels are checked against PyTorch with a separate program:

```sh
make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL ops_check
OBJ.*/ops_check /tmp/ops [gpu [platform]]     # on the OpenCL machine
python3 tests/ops_compare.py /tmp/ops          # where torch is installed
```

The robust estimation stage is checked on the CPU (synthetic models,
outliers and moved clusters; minimum cut against brute force;
leave-one-out against refits), and against OpenCV's USAC_MAGSAC:

```sh
make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL robust_check
OBJ.*/robust_check                 # also run by pytest when built
python3 tests/robust_compare.py    # needs OpenCV (cv2)
```

`tests/eval/` holds the evaluation harness (synthetic warps with a known
field, real image pairs with leave-one-out residuals and loop closure),
run on the GPU server by `tests/eval/run_server.sh`; see
`tests/eval/README.md`.

`tests/reference_compare.py` compares one tile with kornia (DISK
keypoints, scores and descriptors, AdaLAM matches) from the dump written
with `I_COREGISTRATION_OPENCL_DUMP=<folder>`. On a 1024 x 1024 basemap
tile, the keypoints were the same 2,048 in both images, scores within
7e-4, descriptors within 3e-6, and AdaLAM returned the same 538 matches.

## Licence

Unlicense (see `LICENSE`).
