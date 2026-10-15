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
tests/eval/run_server.sh <tag> synthetic ref.tif [--size 2048] [--flags r] [--options matcher=disk]
tests/eval/run_server.sh <tag> basemaps a.tif b.tif c.tif [-- --flags r --options matcher=disk]
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

Default since 2026-10-14: `matcher=eloftr` with the refinement, which no
longer needs `-s` (deprecated; `-r` skips it). The rows above marked
`-s` are what the module now does without flags; to reproduce the
phase 0 and 1 rows, pass `--options matcher=disk` (and `--flags r` for
those run without `-s`).

## Phase 3: MatchAnything (2026-10-14)

`matcher=matchanything`: the Efficient LoFTR network of phase 2 with
the weights `zju-community/matchanything_eloftr` (Hugging Face), and the
rotary positions scaled by 832 over the tile's long side ("npe", as the
authors' code; Transformers lacks it). Parity on one Google/ESRI tile
of 1024 cells (scale 0.8125) with `tests/reference_compare_eloftr.py`:
backbone and transformer within 1.3e-5 relative, 13976/13976 coarse
matches, fine positions within 6.2e-5 pixel.

Synthetic pairs (as phase 2, refinement on, superansac), cells:

| scenario | matcher | active | median error | p90 error | within 1 cell | object leaks |
|---|---|---|---|---|---|---|
| translation | eloftr | 67795 | 0.181 | 0.367 | 99.79 % | 0 |
| translation | eloftr + npe | 67710 | 0.181 | 0.366 | 99.81 % | 0 |
| translation | matchanything | 61790 | 0.179 | 0.37 | 99.66 % | 0 |
| affine | eloftr | 71175 | 0.134 | 0.345 | 99.74 % | 0 |
| affine | eloftr + npe | 70801 | 0.134 | 0.344 | 99.75 % | 0 |
| affine | matchanything | 70446 | 0.134 | 0.345 | 99.65 % | 0 |
| smooth | eloftr | 74729 | 0.127 | 0.31 | 99.86 % | 0 |
| smooth | eloftr + npe | 74526 | 0.127 | 0.31 | 99.86 % | 0 |
| smooth | matchanything | 77042 | 0.126 | 0.309 | 99.82 % | 0 |
| objects | eloftr | 65848 | 0.089 | 0.296 | 99.54 % | 16 |
| objects | eloftr + npe | 65924 | 0.088 | 0.295 | 99.56 % | 12 |
| objects | matchanything | 67524 | 0.088 | 0.294 | 99.32 % | 46 |
| radiometric | eloftr | 72958 | 0.4 | 0.762 | 95.55 % | 0 |
| radiometric | eloftr + npe | 73032 | 0.399 | 0.758 | 95.82 % | 0 |
| radiometric | matchanything | 70846 | 0.396 | 0.791 | 94.14 % | 0 |

Basemaps (4072-cell crops), metres; loop closure median, p90:

| pair | matcher | active | inactive | LOO order 1 median, p90 |
|---|---|---|---|---|
| esri -> google | eloftr | 232210 | 24052 | 0.338, 0.868 |
| esri -> google | eloftr + npe | 230941 | 24595 | 0.339, 0.872 |
| esri -> google | matchanything | 223465 | 25648 | 0.35, 0.861 |
| google -> bing | eloftr | 261708 | 19762 | 0.146, 0.589 |
| google -> bing | eloftr + npe | 261013 | 19882 | 0.147, 0.595 |
| google -> bing | matchanything | 256950 | 21379 | 0.15, 0.6 |
| bing -> esri | eloftr | 221786 | 31935 | 0.327, 0.82 |
| bing -> esri | eloftr + npe | 221062 | 32310 | 0.324, 0.822 |
| bing -> esri | matchanything | 210903 | 33042 | 0.331, 0.799 |

Loop closure: eloftr 0.066, 0.273; eloftr + npe 0.067, 0.28;
matchanything 0.077, 0.325.

Cross-modal: AVIRIS (11.3 m) against the three basemaps averaged onto
its grid (`--pairs star`, threshold 11.3 m, one tile of 768 x 576: npe
scale 1.083), in true colour (640, 550, 460 nm) and in SWIR/NIR false
colour (`prepare_hyperspectral.py --rgb 2200,1650,860`) against the
visible basemaps. Active points (inactive), LOO order 1 median and p90
in metres, for ESRI, Google and Bing:

| composite | matcher | ESRI | Google | Bing |
|---|---|---|---|---|
| true colour | disk | 782 (185) 1.94, 4.32 | 823 (221) 1.68, 3.66 | 767 (225) 1.76, 4.36 |
| true colour | eloftr | 5914 (320) 2.08, 5.17 | 5904 (325) 1.99, 4.69 | 5924 (298) 2.07, 4.97 |
| true colour | eloftr + npe | 5912 (322) 2.08, 5.15 | 5908 (321) 1.99, 4.70 | 5931 (295) 2.07, 4.97 |
| true colour | matchanything | 5918 (321) 2.01, 5.00 | 5973 (255) 2.02, 4.57 | 5951 (271) 2.05, 4.91 |
| true colour | matchanything, no npe | 5925 (316) 2.01, 5.02 | 5975 (253) 2.01, 4.58 | 5953 (273) 2.05, 4.92 |
| SWIR/NIR | disk | 763 (164) 2.80, 6.08 | 541 (304) 2.04, 6.29 | 468 (396) 2.56, 7.01 |
| SWIR/NIR | eloftr | 5678 (531) 2.49, 5.97 | 5289 (845) 2.44, 6.66 | 5003 (1111) 2.75, 7.39 |
| SWIR/NIR | matchanything | 5783 (438) 2.50, 6.03 | 5419 (767) 2.46, 6.64 | 5097 (1047) 2.65, 7.31 |
| SWIR/NIR, -r | disk | 455 (472) 2.79, 11.89 | 355 (474) 3.32, 11.90 | 314 (523) 1.59, 11.91 |
| SWIR/NIR, -r | eloftr | 4969 (1241) 2.03, 9.21 | 4629 (1509) 2.00, 9.82 | 4125 (1994) 2.08, 9.82 |
| SWIR/NIR, -r | matchanything | 5002 (1223) 1.59, 9.55 | 4629 (1558) 1.80, 10.07 | 4149 (2000) 1.84, 10.09 |

Conclusions:

- MatchAnything needs no new code: same tensors, same configuration but
  the npe size; the module is verified against Transformers with the
  scaling.
- The npe scaling changes nothing measurable at these tile sizes (scale
  0.81-1.08), for either weights: Efficient LoFTR keeps Transformers'
  unscaled positions, MatchAnything the authors' scaling.
- On these data MatchAnything is not better than Efficient LoFTR's
  outdoor weights: same synthetic accuracy, 3-5 % fewer basemap points
  and a looser loop closure, three times the moved-object leaks; on the
  cross-modal pairs 1-10 % more active points and 10-20 % fewer
  inactive ones at the same residuals. SWIR/NIR against visible is not
  a hard enough modality gap to show its training; thermal, SAR or map
  references would be (none available here).
- Both ELoFTR weights find 7-10 times DISK's control points across
  modalities, and DISK loses half of its points between SWIR/NIR and
  visible.
- The NCC refinement still helps across modalities (LOO p90 6-7 m
  against 9-10 m with `-r`): it stays on by default.
- Defaults unchanged: `matcher=eloftr`, refinement on.

## Phase 4: RoMa v1 (2026-10-14)

`matcher=roma`: RoMa's outdoor weights with DINOv2 ViT-L/14, coarse pass
at 560 x 560, fine pass at the tile's size, warp sampled at the most
certain cell of each 8 x 8 block when its certainty is at least 0.5.
Parity with the authors' code (`tests/reference_compare_roma.py`, CPU,
float32, native local correlation) on one Google/ESRI tile of 1024
cells: DINOv2 tokens within 6e-6 relative, Gaussian process posterior
6e-5, coarse warp within 1e-4 cell, final warp within 4.6e-4 cell (99 %
within 1e-4), certainty within 3e-5. With random weights (all
BatchNorm statistics randomized) the second pass agreed within 1.5e-4
cell. Device stage about 7 s per 1024 tile on the WX 7100 after three
optimizations (attention as two GEMMs around a softmax, the local
correlation from whole-cell dot products, a tiled depthwise
convolution: 10.1 s before, same results).

Synthetic pairs (as phase 2, superansac), cells; `roma -r` is what the
module now does for RoMa (its positions are never refined):

| scenario | matcher | points | median error | p90 error | within 1 cell | object leaks | wall s |
|---|---|---|---|---|---|---|---|
| translation | eloftr (refined) | 67795 | 0.181 | 0.367 | 99.79 % | 0 | 31.1 |
| translation | roma, refined | 84488 | 0.182 | 0.358 | 99.93 % | 0 | 69.0 |
| translation | roma -r | 84531 | 0.051 | 0.087 | 100.00 % | 0 | 68.7 |
| affine | eloftr (refined) | 71175 | 0.134 | 0.345 | 99.74 % | 0 | 30.3 |
| affine | roma, refined | 81387 | 0.131 | 0.336 | 99.75 % | 0 | 69.6 |
| affine | roma -r | 81610 | 0.032 | 0.069 | 100.00 % | 0 | 67.3 |
| smooth | eloftr (refined) | 74729 | 0.127 | 0.31 | 99.86 % | 0 | 29.5 |
| smooth | roma, refined | 76351 | 0.123 | 0.304 | 99.47 % | 0 | 68.4 |
| smooth | roma -r | 76459 | 0.045 | 0.096 | 100.00 % | 0 | 66.8 |
| objects | eloftr (refined) | 65848 | 0.089 | 0.296 | 99.54 % | 16 | 30.9 |
| objects | roma, refined | 73099 | 0.082 | 0.264 | 99.79 % | 3 | 68.3 |
| objects | roma -r | 75425 | 0.029 | 0.071 | 100.00 % | 1 | 66.7 |
| radiometric | eloftr (refined) | 72958 | 0.4 | 0.762 | 95.55 % | 0 | 30.8 |
| radiometric | roma, refined | 71302 | 0.353 | 0.669 | 98.73 % | 0 | 68.1 |
| radiometric | roma -r | 72728 | 0.084 | 0.176 | 100.00 % | 0 | 67.5 |

Basemaps (4072-cell crops, 25 tiles of 1024), metres:

| pair | matcher | active | inactive | LOO order 1 median, p90 | wall s |
|---|---|---|---|---|---|
| esri -> google | eloftr (refined) | 232210 | 24052 | 0.338, 0.868 | 83 |
| esri -> google | roma, refined | 183902 | 2870 | 0.251, 0.597 | 189 |
| esri -> google | roma -r | 187083 | 2770 | 0.136, 0.319 | 186 |
| google -> bing | eloftr (refined) | 261708 | 19762 | 0.146, 0.589 | 86 |
| google -> bing | roma, refined | 219350 | 5711 | 0.132, 0.428 | 189 |
| google -> bing | roma -r | 220921 | 5629 | 0.104, 0.246 | 185 |
| bing -> esri | eloftr (refined) | 221786 | 31935 | 0.327, 0.82 | 82 |
| bing -> esri | roma, refined | 191954 | 5083 | 0.237, 0.587 | 188 |
| bing -> esri | roma -r | 195453 | 5022 | 0.131, 0.299 | 184 |

Loop closure (median, p90, metres): eloftr refined 0.066, 0.273; roma
refined 0.058, 0.187; roma -r 0.075, 0.158. With tiles of 864 cells
(RoMa's own fine resolution; 36 tiles, 24 % slower) the results are the
same within a few millimetres (roma -r: LOO p90 0.313, 0.247, 0.297;
closure 0.072, 0.153): the default tile size stays 1024.

Cross-modal (as phase 3, one tile of 768 x 576, threshold 11.3 m),
active points (inactive), LOO order 1 median and p90 in metres, for
ESRI, Google and Bing:

| composite | matcher | ESRI | Google | Bing |
|---|---|---|---|---|
| true colour | eloftr (refined) | 5914 (320) 2.08, 5.17 | 5904 (325) 1.99, 4.69 | 5924 (298) 2.07, 4.97 |
| true colour | roma, refined | 5076 (37) 1.97, 4.64 | 5129 (28) 1.94, 4.39 | 5225 (40) 2.01, 4.60 |
| true colour | roma -r | 5120 (0) 1.69, 3.46 | 5165 (1) 1.63, 3.23 | 5269 (1) 1.62, 3.23 |
| SWIR/NIR | eloftr (refined) | 5678 (531) 2.49, 5.97 | 5289 (845) 2.44, 6.66 | 5003 (1111) 2.75, 7.39 |
| SWIR/NIR | roma, refined | 4814 (88) 2.40, 5.49 | 4654 (123) 2.34, 5.57 | 4773 (169) 2.52, 6.19 |
| SWIR/NIR | roma -r | 4901 (0) 1.78, 3.74 | 4780 (0) 1.81, 3.55 | 4948 (2) 1.89, 3.68 |

The cross-modal median shifts differ by matcher: east -1.4 to -2.1 m
with the refinement (and ELoFTR, DISK), -3.2 to -3.9 m with RoMa's own
positions, north up to +2 m (SWIR/NIR). That is 0.15-0.3 of an 11.3 m
cell; without ground truth on these pairs, which is closer is unknown
(on the synthetic pairs RoMa's own positions had no such bias).

Conclusions:

- The translation is faithful to the authors' code at every stage.
- RoMa's own warp is the most accurate of all matchers here: 0.03-0.08
  cell median error on synthetic warps (3-5 times better than any
  refined matcher), basemap LOO p90 0.25-0.32 m against 0.59-0.87 m for
  Efficient LoFTR, and the loop closure p90 down from 0.273 to 0.158 m;
  across modalities LOO p90 3.2-3.7 m against 4.7-7.4 m, with almost no
  inactive points.
- The NCC refinement makes RoMa worse (its errors 3-4 times larger):
  RoMa's positions are never refined.
- Moved objects: 1 leak (Efficient LoFTR 16, DISK 7).
- Fewer points than Efficient LoFTR (at most one per 8 x 8 block: 15-25
  % fewer on the basemaps, 10-15 % across modalities), far fewer
  inactive ones.
- Cost: about 7.4 s per 1024 tile in wall time (Efficient LoFTR about
  3.3 s with the refinement), 1.5 GiB of weights (DINOv2 1.1 GiB) and about 3.5 GiB of
  device memory.

Default since 2026-10-14 (late): `matcher=roma`, chosen on this
evaluation for long-term use; existing outputs change (control points
and their shifts, the Tucson report's included) and runs take about
2.2 times longer. `matcher=eloftr` gives the previous default.

### Dense rasters (4.4, 2026-10-15)

`dense=<base>` (roma only) writes `<base>_dx`, `<base>_dy` (map units,
signs as the points' `dx`, `dy`) and `<base>_certainty`; tiles blended
in their overlaps (linear weights over the overlap width, times the
certainty for the shifts), rows written as soon as final (only one row
of tiles held). `eval_synthetic.py --dense` scores every cell against
the known field (`p4_roma_dense`, same pairs as `p4_roma_r`; moved
objects apart):

| scenario | cells >= 0.5 | median, p90 | cells >= 0.1 | median, p90 | object cells certain and off (>= 0.5, >= 0.1) |
|---|---|---|---|---|---|
| translation | 76.0 % | 0.050, 0.080 | 95.4 % | 0.050, 0.082 | |
| affine | 69.8 % | 0.031, 0.061 | 95.1 % | 0.032, 0.063 | |
| smooth | 63.3 % | 0.046, 0.089 | 95.7 % | 0.045, 0.090 | |
| objects | 67.6 % | 0.027, 0.056 | 89.5 % | 0.027, 0.057 | 12.8 %, 34.0 % |
| radiometric | 65.0 % | 0.080, 0.161 | 92.3 % | 0.084, 0.173 | |

The dense field is as accurate as RoMa's control points (the sampled
best cells), and remains so down to certainty 0.1. On the moved
objects the field follows the object (its own shift, not the
background's): no filter applies to the rasters. Run time unchanged
(68 s per scenario, 9 tiles of 1024). On the 256-cell pytest texture
only about 40 % of the cells reach 0.5 (28 % with a single tile).

### MatchAnything's RoMa weights (2026-10-15)

`matcher=matchanything_roma`: RoMa v1's network with MatchAnything's
retrained weights (`matchanything_roma.ckpt` from the authors' Google
Drive `weights.zip`; same 603 tensors as `roma_outdoor.pth` under the
prefix `matcher.model.`, 541 of them different), the same frozen DINOv2,
inputs in 0..1 without the ImageNet normalization (as MatchAnything
trains and tests). Parity on one Google/ESRI tile
(`reference_compare_roma.py --weights`): final warp within 4.6e-4 cell,
certainty within 4.2e-5. Never downloaded by the module (PRL, see the
manual). Results (`p4_mar`, `p4_basemaps_mar`, `p4_hs_mar`,
`p4_swir_mar`) against RoMa's own weights (`*_roma_r`):

| test | matchanything_roma | roma |
|---|---|---|
| synthetic median / p90 (cells) | 0.08-0.13 / 0.16-0.25 | 0.03-0.08 / 0.07-0.18 |
| basemaps LOO p90 (m) | 0.33-0.47 | 0.25-0.32 |
| basemaps loop closure median, p90 (m) | 0.108, 0.180 | 0.075, 0.158 |
| true colour LOO p90 (m), points | 3.5-3.8, 6564-6614 | 3.2-3.5, 5120-5269 |
| SWIR/NIR LOO p90 (m), points | 4.2-4.5, 6423-6504 | 3.6-3.7, 4780-4948 |
| dense moved-object cells certain and off (>= 0.5) | 33.8 % | 12.8 % |

Less accurate everywhere, more points across modalities. The
cross-modal median east shift is the same as RoMa's (-3.1 to -3.8 m),
so that shift does not come from RoMa's own weights. Not a default;
kept as an option for modalities further apart (thermal, SAR, maps).

### RoMa with RoMa v2's refiners (2026-10-15)

The user does not accept the DINOv3 License, which RoMa v2's matcher
needs. RoMa v2's refiners (1/4, 1/2, 1/1) use only their own VGG19-BN,
so `matcher=roma_v2ref` runs RoMa v1 down to its 1/8 refiner of the
coarse pass, then RoMa v2's refiners at 560 and at the tile's size (as
RoMa v2's low and high resolutions; precision cleared before the second
pass). Weights: `romav2_refiners.pt` (20 MiB, the `refiners.*` and
`refiner_features.*` tensors of `romav2.0.1.pt`, MIT), published at
https://github.com/YannChemin/romav2-refiners (release
`v2.0.1-refiners`) and downloaded by the module. A PyTorch prototype on
one 1024 tile first compared three hand-over points (after RoMa's
coarse pass, after its 1/8, after its full result): all gave 5-17 %
lower median errors than RoMa alone; the 1/8 hand-over had the most
certain cells and the least work. Parity of the OpenCL version
(`reference_compare_roma2.py`): final warp within 1.4e-3 cell (p99
6e-5), overlap within 1.9e-5.

| test | roma_v2ref | roma |
|---|---|---|
| synthetic median (cells) | 0.024-0.087 | 0.029-0.084 |
| synthetic p90 (cells) | 0.049-0.162 | 0.071-0.176 |
| synthetic points | 81797-95882 | 72728-84531 |
| synthetic dense cells >= 0.5 | 74-90 % | 63-76 % |
| synthetic wall time per scenario (9 tiles of 1024) | 54.5 s | 68 s |
| moved-object leaks (points; dense cells >= 0.5) | 0; 10.6 % | 1; 12.8 % |
| basemaps LOO p90 (m) | 0.234-0.313 | 0.246-0.319 |
| basemaps points | 129744-199843 | 187083-220921 |
| basemaps loop closure median, p90 (m) | 0.062, 0.160 | 0.075, 0.158 |
| true colour LOO p90 (m), points | 2.96-3.19, 5743-6103 | 3.23-3.46, 5120-5269 |
| SWIR/NIR LOO p90 (m), points | 3.43-3.66, 5171-5589 | 3.55-3.74, 4780-4948 |

Better or equal everywhere, 20 % faster; fewer control points on the
basemaps (its overlap probability falls below 0.5 more often between
dates), more everywhere else. The cross-modal east shift is unchanged
(-3.0 to -3.9 m).

Default since 2026-10-15: `matcher=roma_v2ref` (the user's decision on
this evaluation); `matcher=roma` gives the previous default.
