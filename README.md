# i.coregistration.opencl

GRASS GIS addon that finds control points between an imagery group and a
reference group with DISK local features matched and filtered by AdaLAM,
on an OpenCL device, and writes them as the group's control points for
*i.rectify* (thin plate spline with `-t`) or *v.rectify*. It is a C and
OpenCL translation of
[RS_coreg_LoFTR](https://github.com/YannChemin/RS_coreg_LoFTR) (kornia
DISK + AdaLAM, then `gdalwarp -tps`): no Python or PyTorch at run time,
the published DISK weights are read directly (legacy PyTorch format).

See `i.coregistration.opencl.md` for the manual and `TODO.md` for the
planned matchers (Efficient LoFTR, RoMa v2, MatchAnything) and robust
estimators (MAGSAC++, Graph-Cut RANSAC, VSAC, SupeRANSAC, dense match
summarization).

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
`tests/reference_compare.py` compares one tile with kornia (DISK
keypoints, scores and descriptors, AdaLAM matches) from the dump written
with `I_COREGISTRATION_OPENCL_DUMP=<folder>`. On a 1024 x 1024 basemap
tile, the keypoints were the same 2,048 in both images, scores within
7e-4, descriptors within 3e-6, and AdaLAM returned the same 538 matches.

## Licence

Unlicense (see `LICENSE`).
