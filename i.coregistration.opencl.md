## DESCRIPTION

*i.coregistration.opencl* finds control points between an imagery group
and a reference imagery group, and writes them as the group's control
points for *i.rectify* (or *v.rectify*). The images are matched on an
OpenCL device (GPU or CPU) by a learnt matcher, with weights read
directly from their published files; the matches are refined to a
fraction of a cell by normalized cross-correlation, and a robust model
fitted per tile makes the remaining outliers (objects that moved
between the images, mismatches) inactive control points. Neither
Python nor PyTorch is needed at run time.

The matching method is chosen with **matcher**: Efficient LoFTR
(`eloftr`, the default), a detector-free matcher that finds matches over
the whole image, texture-poor areas included, or DISK keypoints matched
by nearest neighbour and filtered with AdaLAM (`disk`), which keeps the
matches that agree with a local affine transformation around confident
seed matches, or Efficient LoFTR with the weights of MatchAnything
(`matchanything`), trained for matching across imaging modalities. All
give the same outputs and go through the same filters, refinement and
robust estimation. On the evaluation pairs
(aerial basemaps of 0.5 m), Efficient LoFTR with the refinement gave 13
to 18 times as many control points as DISK, over every part of the
images, and smaller leave-one-out residuals; DISK did better between
images of very different radiometry and left fewer matches on moved
objects (see `tests/eval/README.md`).

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

The matcher runs on tiles of **tile_size** cells (1024 by default), with an
overlap of **tile_overlap** cells, taken at the same place in both
images: since the images are already roughly aligned, each tile of the
image is matched against the same tile of the reference only. This keeps
the matches local, lets images of any size be processed in the memory of
one tile, and spreads the control points over the whole region. With
`disk`, each tile yields up to **features** keypoints per image (2048,
as the repository).

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

### Efficient LoFTR

Efficient LoFTR (Wang et al., 2024) is a detector-free, semi-dense
matcher in the line of LoFTR (Sun et al., 2021). Both tiles, converted to
grey (0.2989 R + 0.5870 G + 0.1140 B), go through a RepVGG backbone (Ding
et al., 2021) whose 3x3, 1x1 and identity branches are folded into single
3x3 convolutions when the weights are read. Its features at one eighth
of the resolution are transformed by four layers of self- and
cross-attention between the two images, computed on tokens aggregated
over 4x4 cells (with a 2D rotary position encoding in the self-attention).
Every cell of 8x8 pixels of the image is then matched to the cell of the
reference with the highest dual-softmax confidence, when that confidence
exceeds 0.2 and the choice is mutual; the score matrix between all cells
of the two tiles is processed by blocks of rows, never stored whole.
Each such match is refined at full resolution, with features fused from
the coarse, quarter and half resolution levels: first to the best pair
of pixels between an 8x8 window of the image and a 10x10 window of the
reference, then to a fraction of a cell in the reference by the
expectation of a 3x3 correlation. The `dist` of a match is one minus its
coarse confidence.

The weights are those published by the authors, converted by Hugging
Face (`zju-community/efficientloftr`, Apache License 2.0, 61 MiB).
Without **checkpoint**, `model.safetensors` is looked for in
**weights_dir** (by default `$ELOFTR_MODELS_DIR`, or `~/models/eloftr`)
and downloaded there when missing, unless **-o** is given. Tile sides
are rounded up to multiples of 32 cells. AdaLAM and the options of the
DISK and AdaLAM sections do not apply. On the same tile, the coarse
matches are those of the authors' code, and the fine positions agree
with it within 1e-4 cell (99 % within 1e-6), but for float ties in the
first fine stage (see `tests/reference_compare_eloftr.py`).

### MatchAnything

MatchAnything, by the group that made Efficient LoFTR (a preprint, not
yet peer reviewed: <https://zju3dv.github.io/MatchAnything>), trains
Efficient LoFTR on
large sets of synthetic cross-modal pairs (visible against thermal,
depth and other renderings) so that one set of weights matches images
of different sensors; the network is unchanged. With
`matcher=matchanything`, the Efficient LoFTR above runs these weights,
with the rotary positions scaled by 832 over the tile's long side (the
authors' training resolution over the test one, as their code does). On
a 1024-cell tile, the coarse matches were those of Transformers'
network run with that scaling (13976 of 13976), and the fine positions
agreed within 1e-4 cell.

The weights converted by Hugging Face
(`zju-community/matchanything_eloftr`, `model.safetensors`, 61 MiB) are
published under the Apache License 2.0 on their model card; the
authors' own repository has since moved to their Project Registration
License (free for research, education and personal use, with a free
registration before use in a company's or organization's project).
Without **checkpoint**, they are looked for in **weights_dir** (by
default `$MATCHANYTHING_MODELS_DIR`, or `~/models/matchanything`) and
downloaded there when missing, unless **-o** is given.

On the Tucson evaluation (aerial basemaps of 0.5 m; an AVIRIS scene of
11.3 m against them, in true colour and in SWIR/NIR false colour), the
MatchAnything weights did as well as Efficient LoFTR's, not better:
the same accuracy on synthetic shifts, slightly larger residuals on the
basemaps, and on the cross-modal pairs a few percent more active
control points at the same residuals, but more matches left on moved
objects. Keep `eloftr` for images of the same kind, and try
`matchanything` where the modalities differ more than here (thermal,
SAR, maps).

### Filters and refinement

Images of different dates show objects that moved: a vehicle, an
aircraft or a ship gives matches that agree with each other but not with
the ground. **max_shift** drops matches whose shift exceeds that
distance (in map units), and **tolerance** those whose shift differs
from the median shift of their tile by more than that along either axis.

DISK keypoints lie on cell centres, so its matches are only as precise
as one cell. Efficient LoFTR refines its matches to a fraction of a
cell, but on aerial basemaps its estimate leans towards whole cells
(median error 0.3-0.4 cell on synthetic sub-cell shifts, against
0.1-0.2 cell after the refinement below). So, unless **-r** is given,
each match is refined in the reference
to a fraction of a cell: the 15x15-cell grey window of the image around
the keypoint is correlated (normalized cross-correlation) with reference
windows shifted by up to two cells, and the peak is fitted by a parabola
along each axis. Matches whose peak lies on the edge of the search area,
or whose windows are flat or meet NULL cells, keep their whole-cell
position. **-s**, which asked for this refinement before it became the
default, is deprecated.

Matches sharing a reference cell (found in two overlapping tiles, or by
two keypoints) are merged into one control point, at the mean of their
positions, as the repository does.

### Robust estimation

AdaLAM checks each match against its neighbourhood only. Objects that
moved between the two images, when they are large enough (an aircraft at
half a metre), give clusters of matches that agree with each other and
pass. So a model is fitted to the matches of each tile
(**tile_model**: an affinity by default, or a similarity or a
homography), after the filters and the refinement, and the matches that
do not fit it become inactive control points: they stay in the POINTS
file with status 0, which *i.rectify* and *v.rectify* skip, and in the
**points** map with `inlier` 0. Inliers and outliers are merged apart.
**estimator** chooses the method, `superansac` by default; `none` keeps
every match active (the behaviour of the RS_coreg_LoFTR workflow).

- `ransac` draws random minimal samples (Fischler and Bolles, 1981),
  keeps the model with the most inliers and refits it to them by least
  squares.
- `magsac` scores each model by the MAGSAC++ quality (Barath et al.,
  2020), which marginalizes the noise scale up to a maximum instead of
  counting points under a hard threshold, and refits the best model by
  least squares weighted by the MAGSAC++ weights, iterated; each point
  gets a weight between 0 and 1.
- `gcransac` labels inliers and outliers by a minimum graph cut with a
  spatial coherence term between neighbouring matches (Graph-Cut RANSAC,
  Barath and Matas, 2018; graph construction after Kolmogorov and Zabih,
  2004, maximum flow by Dinic, 1970), so that a coherent cluster is kept
  or rejected together.
- `superansac` combines them: samples drawn first among the best-ranked
  matches (PROSAC, Chum and Matas, 2005, by the matcher's confidence),
  MAGSAC++ quality, graph-cut local optimization, final weighted least
  squares.

Every estimator rejects samples that do not determine a model and
models that reverse the orientation or scale the image by more than a
factor of two, and counts the matches lying within 8 cells of each other
once (the independent inliers of Ivashechkin et al., 2021), so that a
dense cluster cannot outvote the ground. **threshold** is the largest
distance of an inlier to the model, in map units (3 cells by default);
the MAGSAC++ noise scale goes up to a third of it. A tile where no model
can be fitted has all its matches made inactive.

The threshold trades the outliers left against the good points lost.
On basemaps of half a metre, the default removed the matches of moved
aircraft and halved the 90th percentile of the leave-one-out residuals;
two cells removed more outliers but also points of genuine local
distortion that one model per tile cannot follow. Between a
hyperspectral scene of 11.3 m and basemaps averaged to its grid, the
cross-modal matches have a heavier tail: one to one and a half cells
brought the 90th percentile of the leave-one-out residuals from about
two cells to about a third of a cell, where three cells left it
unchanged.

With **region_model**, a second model is fitted to the active control points
over the whole region (a similarity, an affinity, a homography or a
polynomial of order 2 or 3), with the same estimator, and its outliers
become inactive too. The module then reports the root mean square of
the inlier residuals and, for the linear models, of their leave-one-out
residuals (each point against the least-squares model fitted without
it). Keep it for regions whose images differ by a smooth distortion:
over a mosaic with seams, a global polynomial rejects the points of
whole patches that are simply off by a little.

### Output

The control points are written to the group's POINTS file, source
east/north (the cell's position in the group's current georeferencing)
and reference east/north, active unless rejected by the robust
estimation, and the current project and mapset
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
- `dist`: AdaLAM inverse confidence of the neighbourhood with `disk`,
  one minus the coarse matching confidence with `eloftr` (lower is
  better);
- `tile`: tile of the match;
- `refined`: 1 when refined to a fraction of a cell by the
  cross-correlation (0 with **-r**);
- `matches`: matches merged into the point;
- `weight`: weight of the robust estimation, 0..1 (MAGSAC++ weight
  relative to a zero residual; 1 with **estimator**=`none`);
- `residual`: distance to the last fitted model, in map units (NULL
  with **estimator**=`none`);
- `inlier`: 1 for the active control points, 0 for the rejected ones.

The module reports the number of matches, those dropped by each filter
and by the robust estimation, and the median shift of the active
control points with its median absolute deviation.

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

Efficient LoFTR took about two seconds per 1024 x 1024 tile on the same
GPU, and its matching on the host is almost free (25 tiles in about
1 min 25 s with the refinement, 1 min 5 s with **-r**), for 15 to 20 times as many control
points as DISK, spread over the whole tile. It needs about 2 GiB of
device memory for 1024-cell tiles.

### Debugging aids

With the environment variable `I_COREGISTRATION_OPENCL_DUMP` set to a
folder, each tile's DISK inputs, keypoints with their scores and
descriptors, and AdaLAM matches are written there;
`tests/reference_compare.py` compares them with *kornia* in a Python
environment with PyTorch. With `eloftr` and `matchanything`, the grey
inputs, the backbone,
transformer and fine features and the matches are written, and
`tests/reference_compare_eloftr.py` compares them stage by stage with
*transformers*. With `I_COREGISTRATION_OPENCL_PROFILE` set,
every OpenCL kernel is timed and the totals per kernel are reported at
the end (this slows the run down).

## EXAMPLES

### Two basemaps of different dates

Co-register a Google basemap to an ESRI one, dropping matches on
aircraft that moved between the two dates (more than 5 m, or 2 m away
from their tile's median, then those that do not fit an affinity per
tile), then warp it (the thin plate spline
uses the active control points only):

```sh
i.group group=esri input=esri.1,esri.2,esri.3
i.group group=google input=google.1,google.2,google.3
g.region raster=esri.1 -p
i.coregistration.opencl group=google reference=esri points=gcp_google \
    max_shift=5 tolerance=2
i.rectify -t group=google extension=_coreg method=linear -c
```

### DISK keypoints

The same pair with DISK and AdaLAM, which give fewer control points
(none on tarmac, bare soil and other texture-poor areas) but fare better
between images of very different radiometry:

```sh
i.coregistration.opencl group=google reference=esri points=gcp_google \
    matcher=disk max_shift=5 tolerance=2
```

### Hyperspectral scene to a very high resolution reference

Match a hyperspectral scene (three bands near red, green and blue)
against a WorldView-3 pansharpened RGB image, on the reference grid,
with a robust estimation tighter than the default (cross-modal matches
have a heavier tail: here 5 m, about one and a half cells of the 3.3 m
scene), and look at the active control points before warping:

```sh
i.group group=hymap_rgb input=hymap.28,hymap.17,hymap.8
i.group group=wv3_rgb input=wv3.5,wv3.3,wv3.2
g.region raster=wv3.5 -p
i.coregistration.opencl group=hymap_rgb reference=wv3_rgb \
    points=gcp_hymap -n threshold=5
v.db.univar map=gcp_hymap column=shift where="inlier = 1"
```

### Thermal or SWIR scene with MatchAnything

Match a thermal (or short-wave infrared) composite against a visible
reference with MatchAnything's cross-modal weights
(downloaded into `~/models/matchanything` on the first run), with the same tight
threshold:

```sh
i.coregistration.opencl group=tir_rgb reference=wv3_rgb \
    points=gcp_tir -n matcher=matchanything threshold=5
```

## REFERENCES

Barath, D., Matas, J., 2018. Graph-Cut RANSAC. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
6733-6741.

Barath, D., Noskova, J., Ivashechkin, M., Matas, J., 2020. MAGSAC++, a
fast, reliable and accurate robust estimator. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
1304-1312.

Cavalli, L., Larsson, V., Oswald, M. R., Sattler, T., Pollefeys, M.,
2020. Handcrafted outlier detection revisited. In: Computer Vision -
ECCV 2020, Lecture Notes in Computer Science 12364, Springer, 770-787.
[doi:10.1007/978-3-030-58529-7_45](https://doi.org/10.1007/978-3-030-58529-7_45)

Chum, O., Matas, J., 2005. Matching with PROSAC - progressive sample
consensus. In: Proceedings of the IEEE Computer Society Conference on
Computer Vision and Pattern Recognition (CVPR), vol. 1, 220-226.

Ding, X., Zhang, X., Ma, N., Han, J., Ding, G., Sun, J., 2021. RepVGG:
making VGG-style ConvNets great again. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR),
13733-13742.

Dinic, E. A., 1970. Algorithm for solution of a problem of maximum flow
in networks with power estimation. Soviet Mathematics Doklady 11,
1277-1280.

Fischler, M. A., Bolles, R. C., 1981. Random sample consensus: a
paradigm for model fitting with applications to image analysis and
automated cartography. Communications of the ACM 24 (6), 381-395.

Ivashechkin, M., Barath, D., Matas, J., 2021. VSAC: efficient and
accurate estimator for H and F. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), 15243-15252.

Kolmogorov, V., Zabih, R., 2004. What energy functions can be minimized
via graph cuts? IEEE Transactions on Pattern Analysis and Machine
Intelligence 26 (2), 147-159.

Riba, E., Mishkin, D., Ponsa, D., Rublee, E., Bradski, G., 2020.
Kornia: an open source differentiable computer vision library for
PyTorch. In: Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision (WACV), 3674-3683.

Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X., 2021. LoFTR:
detector-free local feature matching with transformers. In: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 8922-8931.

Tyszkiewicz, M. J., Fua, P., Trulls, E., 2020. DISK: learning local
features with policy gradient. In: Advances in Neural Information
Processing Systems 33 (NeurIPS 2020), 14254-14265.

Wang, Y., He, X., Peng, S., Tan, D., Zhou, X., 2024. Efficient LoFTR:
semi-dense local feature matching with sparse-like speed. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 21666-21675.

## SEE ALSO

*[i.group](i.group.md), [i.ortho.photo](i.ortho.photo.md),
[i.rectify](i.rectify.md), [m.transform](m.transform.md),
[v.rectify](v.rectify.md)*

## AUTHORS

Yann Chemin
