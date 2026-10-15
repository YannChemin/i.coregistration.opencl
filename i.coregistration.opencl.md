## DESCRIPTION

*i.coregistration.opencl* finds control points between an imagery group
and a reference imagery group, and writes them as the group's control
points for *i.rectify* (or *v.rectify*). The images are matched on an
OpenCL device (GPU or CPU) by a learnt matcher, with weights read
directly from their published files; the matches are found (or refined
by normalized cross-correlation) to a fraction of a cell, and a robust model
fitted per tile makes the remaining outliers (objects that moved
between the images, mismatches) inactive control points. Neither
Python nor PyTorch is needed at run time.

The matching method is chosen with **matcher**: RoMa, a dense matcher
that estimates where every cell of the image lies in the reference,
with its certainty, followed by the finer refiners of RoMa v2
(`roma_v2ref`, the default) or alone (`roma`); Efficient LoFTR (`eloftr`), a
detector-free matcher that finds matches over
the whole image, texture-poor areas included; DISK keypoints matched
by nearest neighbour and filtered with AdaLAM (`disk`), which keeps the
matches that agree with a local affine transformation around confident
seed matches, or Efficient LoFTR with the weights of MatchAnything
(`matchanything`), trained for matching across imaging modalities. All
give the same outputs and go through the same filters and robust
estimation. On the evaluation pairs (aerial basemaps of 0.5 m, synthetic
warps of them, a hyperspectral scene of 11.3 m against them), RoMa was
the most accurate: errors of 0.03 to 0.08 cell on synthetic warps
against 0.09 to 0.40 for Efficient LoFTR, the smallest leave-one-out
residuals and loop closure on the basemaps, and the fewest outliers
between the hyperspectral scene and the basemaps, for a little over twice
the run time of Efficient LoFTR (see `tests/eval/README.md`); RoMa v2's
refiners made it as accurate or more on every test and about 20 %
faster. Efficient
LoFTR gave 13 to 18 times as many control points as DISK, over every
part of the images, and smaller residuals.

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

### RoMa

RoMa (Edstedt et al., 2024) is a dense matcher: it estimates, for every
cell of the image, where it lies in the reference (a warp) and how
certain that is. With `matcher=roma`, both tiles (in colour, normalized
with the ImageNet mean and deviation) are first resampled to 560 x 560
cells (antialiased bicubic) for a coarse pass. Their coarse features are
the patch tokens of the frozen DINOv2 ViT-L/14 vision transformer
(Oquab et al., 2024), one per 14 x 14 cells, and their fine features
those of VGG19 with batch normalization (Simonyan and Zisserman, 2015)
at 1/8 down to full resolution. A Gaussian process regression over the
reference's coarse features (as in DKM, Edstedt et al., 2023) gives each
coarse cell of the image an embedding of its likely position, from which
a transformer decoder chooses among 64 x 64 anchor positions. Refiners
at 1/16, 1/8, 1/4, 1/2 and full resolution then correct the warp and its
certainty, from the image's features, the reference's features sampled
at the current warp and, down to 1/4, their correlation in a window
around it. A second pass runs the refiners from 1/8 down again at the
tile's own size, from the first pass's warp, as RoMa's reference code
does when it upsamples its predictions; the certainty is then lowered
where the coarse one was low, and set to zero where the warp leaves the
reference.

The warp is turned into matches by blocks of 8 x 8 cells: in each block,
the cell of highest certainty, when that certainty is at least 0.5. The
`dist` of a match is one minus its certainty. The matches then go
through the same filters, refinement and robust estimation as the other
matchers. With **dense**, the warp itself is written as rasters (see
Output).

The weights are the authors' outdoor ones (`roma_outdoor.pth`, MIT
License, 425 MiB) and DINOv2's (`dinov2_vitl14_pretrain.pth`, Apache
License 2.0, 1.1 GiB). Without **checkpoint**, RoMa's are looked for in
**weights_dir** (by default `$ROMA_MODELS_DIR`, or `~/models/roma`);
DINOv2's are always looked for there. Both are downloaded when missing,
unless **-o** is given. Tile sides are rounded up to multiples of 8
cells. Everything runs in single precision (the authors' code uses half
precision on GPUs), the batch normalizations and DINOv2's layer scales
folded into the weights when they are read. With random weights, the
warp agreed with the authors' code within 1e-4 cell in the second pass,
and within 1e-3 cell for 99 % of the cells of the first pass (see
`tests/reference_compare_roma.py`).

On the Tucson evaluation, RoMa's own positions were the most accurate
of all matchers: median errors of 0.03 to 0.08 cell on synthetic warps
of an aerial basemap (Efficient LoFTR with the refinement: 0.09 to
0.40), a 90th percentile of the leave-one-out residuals of 0.25 to 0.32
m between basemaps of 0.5 m (0.59 to 0.87 m), and between a
hyperspectral scene of 11.3 m (true colour, or short-wave and near
infrared) and visible basemaps 3.2 to 3.7 m (4.7 to 7.4 m), with almost
no control point made inactive by the robust estimation. It gives fewer
control points than Efficient LoFTR (at most one per block) and only
one match on moved objects passed the robust estimation (16 for
Efficient LoFTR). Followed by RoMa v2's refiners (`roma_v2ref`, below),
it is the default matcher; Efficient LoFTR is about twice as fast for a
looser fit.

### RoMa with RoMa v2's refiners

RoMa v2, by the authors of RoMa (a preprint, not yet peer reviewed:
<https://github.com/Parskatt/RoMaV2>), matches on features of DINOv3,
whose licence brings terms this module does not take on (Meta's DINOv3
License Agreement: California law, terms that Meta may change, uses
excluded). Its refiners, however, see none of DINOv3: they refine a
coarse warp from VGG19-BN features of their own, at 1/4, 1/2 and full
resolution, and also give each cell a covariance of its position. With
`matcher=roma_v2ref`, RoMa runs as above down to its refiner at 1/8 of
the coarse pass (DINOv2's features, Gaussian process, coordinate
decoder), then RoMa v2's refiners run from that warp at 560 x 560 cells
and again at the tile's size, as RoMa v2 does with its own matcher; the
certainty is RoMa v2's overlap probability, zero where the warp leaves
the reference. On a 1024-cell tile, the warp agreed with the authors'
code within 1.5e-3 cell (99 % within 1e-4) and the certainty within
2e-5 (`tests/reference_compare_roma2.py`). Positions are never refined
by correlation, and **dense** works as with `roma`.

On the Tucson evaluation, it was as accurate as `roma` or more, and
about 20 % faster (its second pass runs three refiners instead of
four): median errors of 0.02 to 0.09 cell on the synthetic warps (90th
percentiles 0.05 to 0.16 cell against 0.07 to 0.18) with 6 to 25 % more
control points and 74 to 90 % of the cells certain (63 to 76 %); the
same residuals between basemaps (leave-one-out 90th percentile 0.23 to
0.31 m), though with 10 to 33 % fewer control points there; and between
the hyperspectral scene and the basemaps a 90th percentile of 3.0 to
3.2 m in true colour (3.2 to 3.5 m) and 3.4 to 3.7 m in SWIR/NIR (3.6
to 3.7 m), with 4 to 19 % more control points.

The refiners' weights are those of RoMa v2's release (MIT License),
without its DINOv3 backbone and its coarse matcher: the file
`romav2_refiners.pt` (20 MiB) holds only the `refiners` and
`refiner_features` tensors of the authors' `romav2.0.1.pt`, extracted
by a script published with it at
<https://github.com/YannChemin/romav2-refiners>. It is looked for in
`$ROMAV2_MODELS_DIR` or `~/models/romav2` and downloaded there when
missing (unless **-o**); RoMa's and DINOv2's weights are those of
`roma`, in RoMa's folder.

### MatchAnything's RoMa

MatchAnything (see above) also trained RoMa on its cross-modal pairs.
With `matcher=matchanything_roma`, the RoMa above runs those weights:
the same network and the same frozen DINOv2, but the images are given
in 0..1 without the ImageNet normalization, as the authors trained and
test them. On a 1024-cell tile, the warp agreed with the authors' code
within 5e-4 cell and the certainty within 5e-5
(`tests/reference_compare_roma.py --weights`). The positions are never
refined by correlation, and **dense** works as with `roma`.

On the Tucson evaluation these weights were less accurate than RoMa's
own, on every test: median errors of 0.08 to 0.13 cell on the synthetic
warps (RoMa 0.03 to 0.08), a 90th percentile of the leave-one-out
residuals of 0.33 to 0.47 m between basemaps (0.25 to 0.32 m), and of
3.5 to 4.5 m between the hyperspectral scene and the basemaps (3.2 to
3.7 m), although with about 30 % more control points there. Keep `roma`
for these images; try `matchanything_roma` where the modalities differ
more (thermal, SAR, maps), which is what it was trained for.

These weights are **not downloaded by the module**, because of their
licence. They are published only on the authors' Google Drive, as
`weights.zip` (461 MiB, with the Efficient LoFTR weights too), linked
from their repository's README:

- URL: <https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view>
- File to keep: `weights/matchanything_roma.ckpt` in the archive (425
  MiB, a PyTorch Lightning checkpoint).
- Place: **weights_dir**, by default `$MATCHANYTHING_MODELS_DIR` or
  `~/models/matchanything`, as `matchanything_roma.ckpt` (or give the
  file with **checkpoint**).
- DINOv2's weights are read from RoMa's folder (`$ROMA_MODELS_DIR` or
  `~/models/roma`), and downloaded there when missing (Apache License
  2.0, see RoMa).

For example:

```sh
mkdir -p ~/models/matchanything
cd $(mktemp -d)
# Download weights.zip from the URL above into this folder, then:
unzip weights.zip
mv weights/matchanything_roma.ckpt ~/models/matchanything/
```

Licence: the archive states none of its own. The authors' repository
(<https://github.com/zju3dv/MatchAnything>), which before held only a
README, has been under their Project Registration License 1.0 since
2026-09-15. Use for research, education, evaluation and personal
purposes is free, without registration. Use by a company or other
organization in any project (commercial or not, internal or not, from
development to production) is free too but must be registered before
it begins, through the Google form given in the licence (name of the
organization, a contact, the project and its intended use); permission
is then granted automatically, one registration per project. Copies
must carry the licence, and those who receive them must register in
turn. The licence names no governing law and has no clause on changing
its terms. Read it in that repository
before downloading, and register when it applies to you. This module
contains none of the authors' code or weights.

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
default, is deprecated. RoMa's positions are never refined: its warp is
finer than the correlation, which made its errors three to four times
larger on the synthetic warps.

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
- `inlier`: 1 for the active control points, 0 for the rejected ones;
- `sd`: with `roma_v2ref`, the standard deviation of the reference
  position predicted by RoMa v2's refiners, in map units (NULL with
  the other matchers; see below).

RoMa v2's refiners predict, for each cell, a 2 x 2 precision matrix P
of its position in the reference; `sd` is det(P)^(-1/4) (the geometric
mean of the two standard deviations, as the authors' code reports it),
converted to map units. On the Tucson evaluation it ranked the control
points by their accuracy, loosely: the third of the points of smallest
`sd` had a 90th percentile of leave-one-out residuals 13 to 31 % below
that of the third of largest `sd` between the hyperspectral scene and
the basemaps, and 1.8 to 2.8 times below it between basemaps (rank
correlations 0.14 to 0.35); its median was close to the median residual
(1.4 to 2.2 m against 1.2 to 1.6 m across modalities, 0.12 to 0.15 m
against 0.10 to 0.12 m between basemaps). On synthetic warps, where the
errors are a few hundredths of a cell, it was larger than the errors
(0.08 to 0.15 cell) and did not rank them. It is reported, not used by
the module: select or weight with it downstream, for instance with
*v.extract* on `sd`.

With the RoMa matchers, **dense** writes RoMa's warp over the whole
region as raster maps (single precision):

- `<dense>_dx`, `<dense>_dy`: shift east and north from each cell of
  the image to where it lies in the reference, in map units (the same
  sign as the `dx`, `dy` columns of **points**);
- `<dense>_certainty`: RoMa's certainty of that shift, 0..1;
- `<dense>_sd`: with `roma_v2ref`, the predicted standard deviation of
  the shift, in map units (as the `sd` column of **points**), blended
  as the shifts.

Neighbouring tiles are blended in their overlap (**tile_overlap**):
each tile's weight falls linearly to its edges over the overlap width,
and the shifts are averaged with those weights times the certainty. The
certainty is zero, and the shifts NULL, where the warp leaves the
reference or lands on a NULL cell of it; all are NULL where the
image is NULL. The rasters are the network's own estimate: none of the
filters, the refinement or the robust estimation applies to them, so a
vehicle that moved between the dates shows its own shift. On synthetic
warps of a 0.5 m aerial basemap, the cells of certainty 0.5 or more
(63 to 76 % of the cells) had median errors of 0.03 to 0.08 cell and a
90th percentile of 0.06 to 0.16 cell, the same as RoMa's control points;
the cells of certainty 0.1 or more (90 to 96 %) were as accurate. Mask
the shifts with the certainty before using them, for instance with
*r.mapcalc*. Writing them adds no measurable time.

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

RoMa took about 7 seconds per 1024 x 1024 tile on the same GPU (25
tiles in about 3 min 10 s), most of it in the matrix products of
DINOv2, VGG19 and the refiners, and needs about 3.5 GiB of device
memory by the sum of its buffers (its weights alone take 1.7 GiB). Reading its 1.5 GiB of weights takes
a few seconds at the start of each run. With RoMa v2's refiners
(`roma_v2ref`), a tile took about 5.5 seconds (25 tiles in about 2 min
25 s).

### Debugging aids

With the environment variable `I_COREGISTRATION_OPENCL_DUMP` set to a
folder, each tile's DISK inputs, keypoints with their scores and
descriptors, and AdaLAM matches are written there;
`tests/reference_compare.py` compares them with *kornia* in a Python
environment with PyTorch. With `eloftr` and `matchanything`, the grey
inputs, the backbone,
transformer and fine features and the matches are written, and
`tests/reference_compare_eloftr.py` compares them stage by stage with
*transformers*. With `roma`, the normalized inputs of both passes, the
DINOv2 tokens, the Gaussian process posterior, the coarse warp and the
warp and certainty after every refiner are written, and
`tests/reference_compare_roma.py` compares them with the authors' code.
With `I_COREGISTRATION_OPENCL_PROFILE` set,
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

### Faster matching with Efficient LoFTR

For a large mosaic where RoMa's run time (a little over twice) is too much,
Efficient LoFTR with the correlation refinement is the next most
accurate:

```sh
i.coregistration.opencl group=google reference=esri points=gcp_google \
    matcher=eloftr max_shift=5 tolerance=2
```

### Dense shift rasters

Write RoMa's dense shift with its certainty, without changing the group
(**-n**), and keep the shifts of the cells of certainty 0.5 or more:

```sh
i.coregistration.opencl group=google reference=esri dense=shift -n
r.mapcalc expression="shift_sure = if(shift_certainty >= 0.5, \
    sqrt(shift_dx^2 + shift_dy^2), null())"
r.univar map=shift_sure
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

Edstedt, J., Athanasiadis, I., Wadenbäck, M., Felsberg, M., 2023. DKM:
dense kernelized feature matching for geometry estimation. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 17765-17775.
[doi:10.1109/CVPR52729.2023.01704](https://doi.org/10.1109/CVPR52729.2023.01704)

Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., Felsberg, M., 2024.
RoMa: robust dense feature matching. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR),
19790-19800.

Fischler, M. A., Bolles, R. C., 1981. Random sample consensus: a
paradigm for model fitting with applications to image analysis and
automated cartography. Communications of the ACM 24 (6), 381-395.

Ivashechkin, M., Barath, D., Matas, J., 2021. VSAC: efficient and
accurate estimator for H and F. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), 15243-15252.

Kolmogorov, V., Zabih, R., 2004. What energy functions can be minimized
via graph cuts? IEEE Transactions on Pattern Analysis and Machine
Intelligence 26 (2), 147-159.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M.,
Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.,
2024. DINOv2: learning robust visual features without supervision.
Transactions on Machine Learning Research.

Riba, E., Mishkin, D., Ponsa, D., Rublee, E., Bradski, G., 2020.
Kornia: an open source differentiable computer vision library for
PyTorch. In: Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision (WACV), 3674-3683.

Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks for
large-scale image recognition. In: International Conference on Learning
Representations (ICLR 2015).

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
