# Evaluation harness

Scripts that measure what a matcher gives, so that every phase of
`PLAN.md` can be compared with the previous one. Not shipped with the
module; results go to `tests/eval/output/<tag>/` (not committed: the
images may be confidential).

## Scripts

- `eval_synthetic.py`: synthetic pairs with a known warp, built from one
  reference GeoTIFF (translation, affine, smooth field, moved objects,
  radiometric change). Errors of the control points against the true
  field, inlier ratios, coverage, matches on moved objects that pass,
  run time.
- `eval_basemaps.py`: real pairs without ground truth (basemaps of
  several providers or dates). Each image matched to the next and the
  last to the first: shift, leave-one-out residuals of polynomials of
  order 1 and 2, loop closure of the composed shift fields with three or
  more images.
- `prepare_hyperspectral.py`: the cross-modal pair. From a hyperspectral
  Zarr cube (hyp4u layout) and RGB basemaps: a true-colour composite of
  the bands nearest 640, 550 and 460 nm on the cube's grid, and each
  basemap averaged onto that grid, for `eval_basemaps.py`.
- `run_server.sh`: builds the working tree on the GPU server (in its own
  folder, the installed module untouched), uploads the images to a
  private `/tmp` folder, runs one evaluation detached, fetches the
  results and deletes the remote folder. Progress goes to MetaMonitor
  (https://github.com/YannChemin/MetaMonitor).

```sh
tests/eval/run_server.sh <tag> synthetic ref.tif [--size 2048] [--flags s]
tests/eval/run_server.sh <tag> basemaps a.tif b.tif c.tif [-- --flags s]
python3 tests/eval/prepare_hyperspectral.py cube.zarr esri.tif google.tif bing.tif --output <dir>
tests/eval/run_server.sh <tag> basemaps <dir>/hs_rgb.tif <dir>/*_hsgrid.tif -- --pairs star
```

Both scripts run inside a GRASS session (`grass --tmp-project
<image> --exec python3 ...`) and take `--module` to evaluate a build
other than the installed one.

## Baseline (v0: DISK + AdaLAM)

Recorded on 2026-10-14 on the WX 7100 (rusticl) with the phase 0 build,
whose outputs are bit-identical to v0 (same keypoints, descriptors,
matches and control points on 25 tiles of 1024 cells), on crops of the
Tucson basemaps (`output/baseline/` on the machine that ran it).

Synthetic pairs, 2048 x 2048 cells (0.5 m) of the ESRI basemap, `-s`,
9 tiles of 1024 cells; errors in cells against the true field, points
on or within 8 cells of moved objects excluded from the error and
inlier columns:

| scenario | points | median error | p90 error | within 1 cell | within 2 cells | coverage | object leaks | device s | matching s |
|---|---|---|---|---|---|---|---|---|---|
| translation | 4160 | 0.178 | 0.308 | 99.95 % | 100.00 % | 82 % | 0 | 11.1 | 57.8 |
| affine | 5239 | 0.116 | 0.253 | 99.98 % | 99.98 % | 84 % | 0 | 8.9 | 28.4 |
| smooth | 6643 | 0.12 | 0.244 | 99.94 % | 99.98 % | 84 % | 0 | 8.8 | 35.0 |
| objects | 7761 | 0.072 | 0.33 | 91.97 % | 92.58 % | 82 % | 998 | 8.9 | 31.8 |
| radiometric | 6035 | 0.273 | 0.544 | 98.49 % | 99.93 % | 82 % | 0 | 8.9 | 35.4 |

Three basemaps (ESRI, Google, Bing), 4072 x 4072 cells (0.5 m), `-s`,
25 tiles each; shifts and leave-one-out (LOO) residuals in metres:

| pair | points | median shift E, N | LOO order 1 median, p90 | LOO order 2 median, p90 | device s | matching s |
|---|---|---|---|---|---|---|
| esri -> google | 15810 | 0.353, 0.439 | 0.603, 2.593 | 0.583, 2.59 | 25.2 | 76.2 |
| google -> bing | 17252 | -0.028, -0.058 | 0.346, 1.448 | 0.268, 1.438 | 25.4 | 99.9 |
| bing -> esri | 15509 | -0.107, -0.39 | 0.976, 3.023 | 0.854, 3.105 | 25.6 | 77.4 |

Loop closure (ESRI -> Google -> Bing -> ESRI) over 185 of 256
blocks of 256 cells: median 0.212 m, 90th percentile 0.55 m.

Cross-modal: the Tucson AVIRIS scene (`~/RSDATA/cropped_tucson.zarr`,
11.3 m, bands 641.7, 551.5 and 461.6 nm) against the three basemaps
averaged onto its grid, 762 x 558 cells (one tile), `-s`; metres:

| pair | points | median shift E, N | median absolute deviation E, N | LOO order 1 median, p90 |
|---|---|---|---|---|
| AVIRIS -> ESRI | 965 | -1.83, -0.77 | 1.35, 1.47 | 2.72, 25.2 |
| AVIRIS -> Google | 1029 | -2.00, -0.21 | 1.47, 1.15 | 3.91, 29.6 |
| AVIRIS -> Bing | 986 | -1.52, -0.53 | 1.52, 1.17 | 2.48, 33.0 |
| ESRI -> Google (same grid) | 1005 | 0.37, 0.00 | 0.45, 0.48 | 0.85, 2.05 |
| Google -> Bing (same grid) | 742 | -0.17, -0.22 | 0.48, 0.22 | 0.63, 1.56 |
| Bing -> AVIRIS | 856 | 1.59, 0.51 | 1.46, 1.09 | 2.44, 20.4 |

Loop AVIRIS -> ESRI -> Google -> Bing -> AVIRIS, blocks of 64 cells:
median closure 0.93 m, 90th percentile 1.30 m, but over only 13 of 108
blocks (the others lack 5 points on one of the four legs).

Findings for the next phases:

- Without moved objects, v0 is accurate and clean: 0.12-0.27 cell
  median error with `-s`, under 0.1 % of points beyond 2 cells.
- Moved objects the size of aircraft are not stopped by AdaLAM: 998 of
  the 7761 points (13 %) lie on them with a wrong shift. Their presence
  also lets through about 7 % of wrong points on the ground (errors of
  2.5-23 cells, within the range of the object shifts), spread over
  every tile in proportion to the object matches there (correlation
  0.68 between tiles). Likely cause, not confirmed: AdaLAM's
  confidence-based inlier threshold widens in tile-sized neighbourhoods
  that hold coherent object clusters. This is phase 1's test case.
- On real basemaps the leave-one-out residuals have a heavy tail (90th
  percentile 1.4-3.1 m against medians of 0.3-1.0 m): the outliers and
  moved aircraft that a robust stage should weigh down.
- Cross-modal, DISK + AdaLAM already finds about 1000 control points
  between AVIRIS and each basemap. Their core agrees: the three
  references give the same shift within about 0.6 m (0.05 cell), and the
  reverse leg (Bing -> AVIRIS) has the opposite sign. But the tail is
  heavy: the 90th percentile of the leave-one-out residuals is 20-33 m
  (2-3 cells) against 2-3 m for the median, so more than 10 % of the
  points are wrong. A robust stage matters even more here, and this pair
  is the yardstick for MatchAnything (phase 3).
- Host-side matching (nearest neighbours, AdaLAM, `-s` refinement)
  takes 3 to 6 times as long as DISK on the GPU: the CPU stage is the
  bottleneck on the server.

## Phase 1: robust estimation (2026-10-14)

Same data and build settings as the baseline, with `estimator=` (per-tile
affinity unless stated). `superansac` is now the module's default: to
reproduce the v0 baseline, pass `--options estimator=none`. Synthetic pairs, scenario `objects` (the
others change little, see below); counts of control points:

| configuration | active | inactive | correct points lost | object leaks | ground within 1 cell | within 2 cells |
|---|---|---|---|---|---|---|
| v0 (none) | 7761 | 0 | 0 | 998 | 91.97 % | 92.58 % |
| ransac | 6482 | 1506 | 1 | 7 | 99.65 % | 99.78 % |
| magsac | 6482 | 1506 | 1 | 7 | 99.65 % | 99.78 % |
| gcransac | 6486 | 1494 | 1 | 7 | 99.43 % | 99.59 % |
| superansac | 6482 | 1506 | 1 | 7 | 99.65 % | 99.78 % |
| superansac, homography per tile | 6482 | 1506 | 1 | 7 | 99.65 % | 99.78 % |
| superansac + region_model=poly2 | 6482 | 1506 | 1 | 7 | 99.65 % | 99.78 % |
| superansac, threshold 1 m (2 cells) | 6460 | 1526 | 1 | 0 | 99.87 % | 100.00 % |

Correct points lost in the other scenarios: none with the default
threshold (one point in `smooth`); `region_model=poly2` lost 62 of
`smooth` (a global polynomial cannot follow its local bumps), threshold
1 m lost 273 of `smooth`.

Real basemaps (`superansac`), metres:

| configuration | pair | active | inactive | LOO order 1 median, p90 |
|---|---|---|---|---|
| v0 | esri -> google | 15810 | 0 | 0.603, 2.593 |
| v0 | google -> bing | 17252 | 0 | 0.346, 1.448 |
| v0 | bing -> esri | 15509 | 0 | 0.976, 3.023 |
| default threshold | esri -> google | 12536 | 3621 | 0.385, 1.102 |
| default threshold | google -> bing | 15638 | 1944 | 0.237, 0.869 |
| default threshold | bing -> esri | 11590 | 4234 | 0.441, 1.2 |
| threshold 1 m | esri -> google | 10893 | 5251 | 0.328, 0.786 |
| threshold 1 m | google -> bing | 14351 | 2987 | 0.154, 0.531 |
| threshold 1 m | bing -> esri | 10021 | 5981 | 0.349, 0.825 |
| + region_model=poly2 | esri -> google | 12383 | 3774 | 0.372, 1.067 |
| + region_model=poly2 | google -> bing | 15310 | 2272 | 0.182, 0.783 |
| + region_model=poly2 | bing -> esri | 11202 | 4622 | 0.371, 1.117 |

Loop closure ESRI -> Google -> Bing -> ESRI (median, 90th percentile,
metres): v0 0.212, 0.55; default threshold 0.154, 0.384; threshold
1 m 0.147, 0.357; with region_model=poly2 0.132, 0.374.

AVIRIS against the basemaps on its grid (`--pairs star`), metres:

| configuration | pair | active | inactive | LOO order 1 median, p90 |
|---|---|---|---|---|
| v0 | AVIRIS -> esri | 965 | 0 | 2.715, 25.17 |
| v0 | AVIRIS -> google | 1029 | 0 | 3.914, 29.584 |
| v0 | AVIRIS -> bing | 986 | 0 | 2.479, 32.987 |
| superansac, default (34 m) | AVIRIS -> esri | 934 | 32 | 2.621, 23.115 |
| superansac, default (34 m) | AVIRIS -> google | 949 | 96 | 2.12, 22.04 |
| superansac, default (34 m) | AVIRIS -> bing | 895 | 97 | 2.325, 22.545 |
| superansac, 17 m | AVIRIS -> esri | 794 | 173 | 1.935, 4.574 |
| superansac, 17 m | AVIRIS -> google | 829 | 215 | 1.741, 3.781 |
| superansac, 17 m | AVIRIS -> bing | 771 | 221 | 1.749, 4.471 |
| superansac, 11.3 m | AVIRIS -> esri | 782 | 185 | 1.939, 4.316 |
| superansac, 11.3 m | AVIRIS -> google | 823 | 221 | 1.684, 3.659 |
| superansac, 11.3 m | AVIRIS -> bing | 767 | 225 | 1.764, 4.358 |
| magsac, 11.3 m | AVIRIS -> esri | 782 | 185 | 1.939, 4.316 |
| magsac, 11.3 m | AVIRIS -> google | 823 | 221 | 1.684, 3.659 |
| magsac, 11.3 m | AVIRIS -> bing | 767 | 225 | 1.764, 4.358 |
| gcransac, 11.3 m | AVIRIS -> esri | 788 | 179 | 1.964, 4.403 |
| gcransac, 11.3 m | AVIRIS -> google | 827 | 216 | 1.74, 3.756 |
| gcransac, 11.3 m | AVIRIS -> bing | 776 | 216 | 1.765, 4.617 |

Conclusions:

- The per-tile model does the work: on these data the four estimators
  draw nearly the same line between inliers and outliers (Graph-Cut
  RANSAC keeps a few more wrong points next to good ones). superansac
  is as good as the best of them everywhere.
- Moved aircraft-sized objects: 998 wrong points down to 7 (0 at 2
  cells), for one correct point lost.
- Basemaps: the 90th percentile of the leave-one-out residuals is
  halved or better (2.6-3.0 m to 1.1-1.2 m) and the loop closes
  tighter, so the rejected points were inconsistent between the three
  sources, not local truth.
- Cross-modal: the default threshold (3 cells) is too loose; 1-1.5 cells
  takes the 90th percentile from about 2 cells to about 0.35 cell.
- region_model=poly2 helps little over the tiles and costs correct
  points where the distortion is local: keep it for smooth cases.
- The estimation adds no measurable time: the matching stage stays
  within its run-to-run variation (about a second per 9 or 25 tiles).

## Phase 2: Efficient LoFTR (2026-10-14)

Same data as phase 1, `matcher=eloftr` with the default estimator
(superansac, threshold 3 cells), without and with `-s`; the DISK rows
are phase 1's (`-s`, superansac). Synthetic pairs, errors in cells:

| scenario | matcher | active | median error | p90 error | within 1 cell | coverage | object leaks | wall s |
|---|---|---|---|---|---|---|---|---|
| translation | disk -s | 4160 | 0.178 | 0.308 | 99.95 % | 82 % | 0 | 69.9 |
| translation | eloftr | 68185 | 0.391 | 0.458 | 99.51 % | 100 % | 0 | 22.0 |
| translation | eloftr -s | 67795 | 0.181 | 0.367 | 99.79 % | 100 % | 0 | 31.1 |
| affine | disk -s | 5239 | 0.116 | 0.253 | 99.98 % | 84 % | 0 | 38.3 |
| affine | eloftr | 71438 | 0.34 | 0.566 | 99.53 % | 100 % | 0 | 21.8 |
| affine | eloftr -s | 71175 | 0.134 | 0.345 | 99.74 % | 100 % | 0 | 30.3 |
| smooth | disk -s | 6642 | 0.12 | 0.244 | 99.95 % | 84 % | 0 | 45.6 |
| smooth | eloftr | 74835 | 0.3 | 0.538 | 99.81 % | 100 % | 0 | 22.1 |
| smooth | eloftr -s | 74729 | 0.127 | 0.31 | 99.86 % | 100 % | 0 | 29.5 |
| objects | disk -s | 6482 | 0.064 | 0.199 | 99.65 % | 82 % | 7 | 42.5 |
| objects | eloftr | 66011 | 0.434 | 0.498 | 97.42 % | 100 % | 12 | 22.1 |
| objects | eloftr -s | 65848 | 0.089 | 0.296 | 99.54 % | 100 % | 16 | 30.9 |
| radiometric | disk -s | 6035 | 0.273 | 0.544 | 98.49 % | 82 % | 0 | 46.1 |
| radiometric | eloftr | 73560 | 0.63 | 0.833 | 95.50 % | 100 % | 0 | 22.0 |
| radiometric | eloftr -s | 72958 | 0.4 | 0.762 | 95.55 % | 100 % | 0 | 30.8 |

Real basemaps, metres:

| pair | matcher | active | inactive | median shift E, N | LOO order 1 median, p90 | wall s |
|---|---|---|---|---|---|---|
| esri -> google | disk -s | 12536 | 3621 | 0.318, 0.469 | 0.385, 1.102 | 107 |
| esri -> google | eloftr | 233814 | 23791 | 0.049, 0.5 | 0.339, 0.824 | 63 |
| esri -> google | eloftr -s | 232210 | 24052 | 0.188, 0.497 | 0.338, 0.868 | 83 |
| google -> bing | disk -s | 15638 | 1944 | -0.023, -0.051 | 0.237, 0.869 | 129 |
| google -> bing | eloftr | 262897 | 19610 | -0.003, -0.009 | 0.079, 0.56 | 64 |
| google -> bing | eloftr -s | 261708 | 19762 | -0.007, -0.033 | 0.146, 0.589 | 86 |
| bing -> esri | disk -s | 11590 | 4234 | -0.116, -0.444 | 0.441, 1.2 | 105 |
| bing -> esri | eloftr | 223866 | 31394 | -0.025, -0.492 | 0.36, 0.755 | 63 |
| bing -> esri | eloftr -s | 221786 | 31935 | -0.122, -0.48 | 0.327, 0.82 | 82 |

Loop closure ESRI -> Google -> Bing -> ESRI (median, 90th percentile,
metres): disk -s 0.154, 0.384 over 180 of 256 blocks; eloftr 0.026,
0.445 over 256; eloftr -s 0.066, 0.273 over 256.

Conclusions:

- The translation is faithful: on one Google/ESRI tile of 1024 cells,
  the same 14009 coarse matches as the authors' code (through
  Transformers' network), fine positions within 6e-5 pixel but for 3
  float ties (`tests/reference_compare_eloftr.py`). Transformers 4.57
  itself departs from the authors' fine matching (scale of the first
  stage; its first stage always returns window position 0); the module
  follows the authors.
- ELoFTR's own sub-cell estimate leans to whole cells on these images:
  on the translation (3.3, -2.7) its median shift is (3.03, -2.98). The
  second stage moves the points by about 0.1 pixel where 0.3 is needed;
  centring its 3 x 3 window on the chosen pixel (instead of the
  authors' indexing, one pixel up-left) makes it worse, so this is the
  weights' behaviour, not a translation error. The near-integer shifts
  also make the loop closure of ELoFTR without `-s` look better than it
  is. With `-s` the errors come down to those of DISK (0.09-0.18 cell
  median), radiometric changes excepted (0.40 against 0.27).
- 13-18 times as many control points, over all of every tile (coverage
  100 % against 82-84 %; loop closure on all 256 blocks against 180),
  and on real basemaps a lower leave-one-out 90th percentile (0.59-0.87 m
  against 0.87-1.20 m).
- Moved objects: superansac keeps 12-16 leaks out of about 7000 object
  matches (DISK: 7 out of about 1500).
- Speed: the device stage is about 2 s per 1024 tile, the host matching
  almost free (no AdaLAM): 63-86 s wall per basemap pair against
  105-129 s for DISK. The 3x3 convolutions now run as im2col + GEMM
  (16.2 s to 2.1 s per tile; bit-identical results); DISK still uses its
  conv5x5 kernel.
