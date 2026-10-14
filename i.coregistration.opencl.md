## DESCRIPTION

*i.coregistration.opencl* finds control points between an imagery group
and a reference imagery group, and writes them as the group's control
points for *i.rectify* (or *v.rectify*). Keypoints and descriptors are
extracted with DISK, a learnt local feature, on an OpenCL device (GPU or
CPU), with weights read directly from the published PyTorch checkpoint;
they are matched by nearest neighbour and the matches filtered with
AdaLAM, which keeps those that agree with a local affine transformation
around confident seed matches. Neither Python nor PyTorch is needed at
run time.

The matching method is chosen with **matcher**. DISK with AdaLAM
(`disk`) is the only one so far; detector-free and dense matchers will
be added as further choices, with the same outputs.

The module is a translation to C and OpenCL of the RS_coreg_LoFTR
workflow (DISK features, AdaLAM matching, control points, thin plate
spline warp with *gdalwarp*), made for images that are already roughly
georeferenced in the same project: aerial and satellite basemaps of
different dates, a hyperspectral scene and a very high resolution
reference, and so on.

The **group** (the image to co-register) and the **reference** (or their
**subgroup** and **reference_subgroup**) must each contain exactly three
raster maps, used as red, green and blue in group order (see *i.group*).
Both are read over the current computational region, on its grid: set
the region to the reference (for instance `g.region raster=...`) and make
sure that the image to co-register covers it. Bands within 0..255 are
used as they are (as the repository reads its 8-bit images); others, or
all bands with **percentile**, are stretched linearly to 0..255. Cells
that are NULL in any band yield no keypoint.

### Tiles

DISK runs on tiles of **tile_size** cells (1024 by default), with an
overlap of **tile_overlap** cells, taken at the same place in both
images: since the images are already roughly aligned, each tile of the
image is matched against the same tile of the reference only. This keeps
the matches local, lets images of any size be processed in the memory of
one tile, and spreads the control points over the whole region. Each
tile yields up to **features** keypoints per image (2048, as the
repository).

### DISK

DISK (Tyszkiewicz et al., 2020) is a U-Net of 5x5 convolutions with
instance normalization and PReLU gates, whose output holds a 128-value
descriptor and a detection score for every cell. Keypoints are the cells
that are strict maxima of the score over 5x5 windows and positive, the
**features** highest of them; their descriptors are normalized to unit
length. This is the inference procedure of *kornia.feature.DISK*, and on
the same tile the keypoints, scores and descriptors agree with it to
single precision (see `tests/reference_compare.py`).

The **checkpoint** is one of the two published DISK models (4.2 MiB
each), chosen with **model**:

- `depth` ([depth-save.pth](https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth),
  default, as the repository): trained with depth supervision, more
  precise;
- `epipolar` ([epipolar-save.pth](https://raw.githubusercontent.com/cvlab-epfl/disk/master/epipolar-save.pth)):
  trained with epipolar supervision, with keypoints on everything that
  can be matched, thin structures and object edges included.

Without **checkpoint**, the model is looked for in **weights_dir** (by
default `$DISK_MODELS_DIR`, or `~/models/disk`) and downloaded there when
missing, unless the **-o** flag is given. Both the current PyTorch zip
format and the legacy format these files use are read.

### Matching and AdaLAM

Each keypoint of the image is matched to the nearest descriptor of the
reference, scored by the ratio of the squared distances to the nearest
and second nearest. AdaLAM (Cavalli et al., 2020) then picks seed
matches, the best-scored ones (ratio below 0.8 squared) within circles
whose area is the tile's keypoint extent divided by **area_ratio**,
gathers around each seed the matches lying within **search_expansion**
times that radius in both images, and fits a local affine transformation
to each neighbourhood with a confidence-based RANSAC of
**ransac_iters** hypotheses followed by a least-squares refit. Matches
of neighbourhoods with a confidence of at least **min_confidence** and
at least **min_inliers** inliers are kept. With **-m**, seeds are taken
among mutual nearest neighbours only. The defaults are those of the
repository (*kornia*'s with a search expansion of 16, 256 hypotheses and
no mutual nearest neighbour constraint), and the procedure follows
*kornia.feature.match_adalam* step by step: on the same keypoints, both
return the same matches. DISK keypoints have no orientation or scale, so
AdaLAM's orientation and scale agreement tests do not apply.

### Filters and refinement

Images of different dates show objects that moved: a vehicle, an
aircraft or a ship gives matches that agree with each other but not with
the ground. **max_shift** drops matches whose shift exceeds that
distance (in map units), and **tolerance** those whose shift differs
from the median shift of their tile by more than that along either axis.

DISK keypoints lie on cell centres, so control points are only as
precise as one cell. With **-s**, each match is refined in the reference
to a fraction of a cell: the 15x15-cell grey window of the image around
the keypoint is correlated (normalized cross-correlation) with reference
windows shifted by up to two cells, and the peak is fitted by a parabola
along each axis. Matches whose peak lies on the edge of the search area,
or whose windows are flat or meet NULL cells, keep their whole-cell
position.

Matches sharing a reference cell (found in two overlapping tiles, or by
two keypoints) are merged into one control point, at the mean of their
positions, as the repository does.

### Output

The control points are written to the group's POINTS file, source
east/north (the cell's position in the group's current georeferencing)
and reference east/north, all active, and the current project and mapset
become the group's target: *i.rectify* with **-t** (thin plate spline,
the repository's `gdalwarp -tps`) or a polynomial **order** then warps
the group onto the reference, and *v.rectify* moves vector maps drawn on
the image. With **-n**, the group is left as it is.

The **points** vector map holds the control points at their reference
positions, with the attribute columns:

- `src_east`, `src_north`: position in the image;
- `ref_east`, `ref_north`: position in the reference;
- `dx`, `dy`, `shift`: shift from the image to the reference, in map
  units;
- `dist`: AdaLAM inverse confidence of the neighbourhood (lower is
  better);
- `tile`: tile of the match;
- `refined`: 1 when refined to a fraction of a cell (**-s**);
- `matches`: matches merged into the point.

The module reports the number of matches, those dropped by each filter,
and the median shift with its median absolute deviation.

## NOTES

### OpenCL device

**device** selects the first GPU (`gpu`), the first CPU device (`cpu`),
or the first GPU and else the first CPU (`auto`); **platform** restricts
the search to platforms whose name contains the given text. A GPU is
strongly recommended. With Mesa's rusticl, set `RUSTICL_ENABLE` to the
driver, e.g. `RUSTICL_ENABLE=radeonsi`.

### Performance

DISK takes about 300 billion multiply-adds per million cells and per
image. On one 1024 x 1024 tile, a laptop CPU (PoCL, 4 cores) took about
four minutes for both images. The memory needed on the device grows with
the tile area: about 1.2 GiB for 1024-cell tiles.

On an AMD Radeon Pro WX 7100 GPU (Mesa rusticl), a 1024 x 1024 tile
took about four seconds for both images, matching included (25 tiles in
1 min 45 s). The first run after an installation is a few minutes
longer, while the driver compiles the kernels; it keeps them in its
cache afterwards. With **--verbose**, the time spent on the device and
in matching is reported at the end.

### Debugging aids

With the environment variable `I_COREGISTRATION_OPENCL_DUMP` set to a
folder, each tile's DISK inputs, keypoints with their scores and
descriptors, and AdaLAM matches are written there;
`tests/reference_compare.py` compares them with *kornia* in a Python
environment with PyTorch. With `I_COREGISTRATION_OPENCL_PROFILE` set,
every OpenCL kernel is timed and the totals per kernel are reported at
the end (this slows the run down).

## EXAMPLES

### Two basemaps of different dates

Co-register a Google basemap to an ESRI one, dropping matches on
aircraft that moved between the two dates (more than 5 m, or 2 m away
from their tile's median), with sub-cell refinement, then warp it:

```sh
i.group group=esri input=esri.1,esri.2,esri.3
i.group group=google input=google.1,google.2,google.3
g.region raster=esri.1 -p
i.coregistration.opencl group=google reference=esri points=gcp_google \
    -s max_shift=5 tolerance=2
i.rectify -t group=google extension=_coreg method=linear -c
```

### Hyperspectral scene to a very high resolution reference

Match a hyperspectral scene (three bands near red, green and blue)
against a WorldView-3 pansharpened RGB image, on the reference grid, and
look at the control points before warping:

```sh
i.group group=hymap_rgb input=hymap.28,hymap.17,hymap.8
i.group group=wv3_rgb input=wv3.5,wv3.3,wv3.2
g.region raster=wv3.5 -p
i.coregistration.opencl group=hymap_rgb reference=wv3_rgb \
    points=gcp_hymap -n
v.db.univar map=gcp_hymap column=shift
```

## REFERENCES

Cavalli, L., Larsson, V., Oswald, M. R., Sattler, T., Pollefeys, M.,
2020. Handcrafted outlier detection revisited. In: Computer Vision -
ECCV 2020, Lecture Notes in Computer Science 12364, Springer, 770-787.
[doi:10.1007/978-3-030-58529-7_45](https://doi.org/10.1007/978-3-030-58529-7_45)

Riba, E., Mishkin, D., Ponsa, D., Rublee, E., Bradski, G., 2020.
Kornia: an open source differentiable computer vision library for
PyTorch. In: Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision (WACV), 3674-3683.

Tyszkiewicz, M. J., Fua, P., Trulls, E., 2020. DISK: learning local
features with policy gradient. In: Advances in Neural Information
Processing Systems 33 (NeurIPS 2020), 14254-14265.

## SEE ALSO

*[i.group](i.group.md), [i.ortho.photo](i.ortho.photo.md),
[i.rectify](i.rectify.md), [m.transform](m.transform.md),
[v.rectify](v.rectify.md)*

## AUTHORS

Yann Chemin
