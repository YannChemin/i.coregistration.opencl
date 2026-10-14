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
