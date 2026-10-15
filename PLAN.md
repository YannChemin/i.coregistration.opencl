# Implementation plan for TODO.md

Written 2026-10-14, after the DISK + AdaLAM translation (v0) matched
kornia exactly and ran on the Tucson basemaps (Google against ESRI:
252 tiles, 16 min on the WX 7100, 205,871 matches, median shift
-0.53 m east, 0.00 m north).

The order below puts first what helps every matcher and the current
Tucson work (a robust estimation stage), then the matchers by cost:
Efficient LoFTR, MatchAnything's ELoFTR weights, RoMa v2. Each phase ends
with a release that passes its reference comparison and the pytest
suite.

## Principles (all phases)

- **Faithful first, then extend.** Each network is translated so that,
  on one dumped tile, the OpenCL output matches the PyTorch reference
  (keypoints or matches identical, values within single precision), as
  done for DISK and AdaLAM (`tests/reference_compare.py`). Extensions
  (tiling, filters, robust stage) come after and stay optional.
- **One module, several matchers.** A `matcher=` option (`disk`,
  `eloftr`, `matchanything_eloftr`, `romav2`, later `roma`,
  `matchanything_roma`) and an `estimator=` option for the robust stage.
  Outputs stay the same: group POINTS, `points=`, and later dense rasters.
- **Weights read from the published checkpoints**, located in
  `~/models/<name>` (overridable), downloaded when the licence allows it,
  never redistributed. The zip and legacy PyTorch formats are already
  read; Lightning `.ckpt` files are zip archives with a `state_dict`
  wrapper (already handled).
- **Licences checked before translating** (see TODO.md): translate only
  from MIT, BSD or Apache 2.0 code, keep notices in the file headers
  (and state changes for Apache 2.0). Research-only or registration
  licences: implement from the papers instead, or keep out of the
  published addon.
- **GRASS conventions**: `G_percent` progress (MetaMonitor's GRASS hook),
  `G_fatal_error` on inconsistent input, md/html manual kept in sync,
  pytest tests that skip without weights or device, clang-format LLVM
  4-space Stroustrup.
- **Device budget**: the WX 7100 (16 GiB, 2 GiB per buffer under
  rusticl) is the target; every phase states its memory per tile and
  splits work so that no buffer exceeds 2 GiB.

## Phase 0. Shared infrastructure and evaluation (first)

Goal: the code base and the yardstick that every later phase needs.

0.1 **Tensor kernel library.** Move the generic kernels into one
`ops.cl` with a C wrapper (`ops.c`): NxN convolution with stride,
dilation, groups and fused pre-normalization (generalizing `conv5x5`),
1x1 convolution and linear layers (tiled matrix product), LayerNorm,
InstanceNorm and BatchNorm (folded at load), GELU/ReLU/LeakyReLU/PReLU,
softmax over rows, scaled dot-product attention by blocks
(flash-attention style: running max and sum, no full score matrix),
RoPE 1D/2D, bilinear upsampling, `grid_sample` (bilinear, zero padding,
align_corners as each model needs), unfold of local windows. Reuse the
matrix product, LayerNorm, softmax and attention kernels of
i.sam.opencl / i.sam3.opencl (same author, Unlicense).

0.2 **Matcher interface.** `struct matcher { load(); alloc(); match(tile_src,
tile_ref) -> matches (x1, y1, x2, y2, confidence) }`, with DISK+AdaLAM as
the first implementation; `main.c` loops over tiles once for all.

0.3 **Evaluation harness** (`tests/eval/`, scripts, not shipped):
- synthetic pairs with known warps: translation, affine, smooth TPS
  field of 0-3 cells, plus moved "objects" (pasted patches) and
  radiometric changes, on Tucson tiles;
- the three Tucson basemaps (three dates): leave-one-out residuals of the
  fitted model on held-out control points, the agreement of
  ESRI->Google->Bing->ESRI loops (should close to zero);
- the hyperspectral scene against a basemap (cross-modal, for
  MatchAnything);
- metrics: match count, inlier ratio, median and 90th percentile
  residual against the known field, run time per tile on the WX 7100.
- record v0 (DISK + AdaLAM) as the baseline.

Deliverable: refactor with no change of results (reference_compare and
pytest unchanged), eval report for v0.

Status (2026-10-14): done. `ops.cl`/`ops.c` (40 cases within 3e-6 of
PyTorch on the WX 7100, `tests/ops_check.c` + `tests/ops_compare.py`),
`matcher.h` with `matcher_disk.c` and a `matcher=` option, `tests/eval/`
(synthetic and real-pair scripts, `run_server.sh`). Dumps and control
points bit-identical to v0 on 25 tiles; pytest 5/5 on the server
(but against the installed v0 binary: the GRASS session puts the
installation's bin first; fixed in phase 1 with
I_COREGISTRATION_OPENCL_BINDIR); same speed (1 min 45 s for 25 tiles). Not done: a separate fast 3x3 or
im2col path for ELoFTR's convolutions (left to phase 2, when its sizes
are known). Cross-modal pair added afterwards (AVIRIS against the
three basemaps on the AVIRIS grid, `tests/eval/prepare_hyperspectral.py`).
Baseline and findings in
`tests/eval/README.md`; the main one for phase 1: moved aircraft-sized
objects pass AdaLAM (13 % of points) and bring about 7 % of wrong ground
points with them.

## Phase 1. Robust estimation stage (CPU, C)

Goal: a model fitted to the matches, with per-point weights and
residuals, so that outliers and moved objects stop reaching the thin
plate spline. Benefits DISK now and every later matcher.

1.1 **Models** (`models.c`): 2D similarity, affinity (3 points),
homography (4 points, normalized DLT), polynomials of order 1-3 (3, 6,
10 points; same form as i.rectify's, so that results can be checked
against `I_compute_georef_equations`), and a regularized TPS (lambda
from leave-one-out). Each with minimal and least-squares/weighted
solvers and a residual function.

1.2 **Scoring: MAGSAC++** (Barath et al., 2020; translate from
`superansac/include/scoring/magsac_scoring.h`, MIT, and `magsac`, BSD):
noise scale marginalized up to a maximum sigma, no hard threshold; the
final fit by iteratively reweighted least squares gives each control
point a weight in 0..1.

1.3 **Samplers**: PROSAC (matches ordered by AdaLAM inverse confidence
or matcher confidence), Progressive NAPSAC (local samples first, through
a grid neighbourhood; from superansac's samplers, MIT).

1.4 **Verification and degeneracy** (VSAC, Ivashechkin et al., 2021):
SPRT early rejection of bad models, independent inliers (matches closer
than a few cells count once, so that one roof or one aircraft cannot
outvote the ground), sample degeneracy checks (collinear or clustered
samples, condition number), model checks (scale and shear bounds, as
AdaLAM's det_thr). Implemented from the paper; OpenCV's USAC (Apache
2.0) as cross-check.

1.5 **Graph-cut local optimization** (Barath and Matas, 2018): the
binary labelling inlier/outlier with a spatial smoothness term is a
single s-t minimum cut. The bundled GCoptimization/maxflow code is
research-only, so write the max-flow from the published algorithm
(Boykov and Kolmogorov, 2004; the related US patent 6,744,923 has
expired) and the GC-RANSAC energy from the paper.

1.6 **Pipeline (SupeRANSAC recipe, Barath 2025)**: per tile, robust
affinity or homography (removes moved objects and mismatches that
AdaLAM let through); then over the region, robust polynomial (order
chosen by `order=`) with MAGSAC++ weights; optional TPS on the weighted
inliers. Options: `estimator=none|ransac|magsac|gcransac|superansac`,
`model=`, `max_sigma=`, `confidence=`. Outputs: control point status 0
for outliers in POINTS (i.rectify skips them), `weight`, `residual`,
`inlier` columns in `points=`, RMS and leave-one-out RMS in the report.

1.7 **Tests**: synthetic sets with known model, 30-80 % outliers and
coherent clusters (moved objects): the right inliers and model within
tolerance; same scores as superansac's Python bindings on fixed seeds
(built in a throwaway venv for the comparison only).

Effort: the largest C part (estimators, samplers, scoring, max-flow);
no device code. Done before any new network.

Status (2026-10-14): done, with these differences from the plan.
`models.c` (similarity, affinity, homography, poly2, poly3; weighted
least squares, exact leave-one-out), `robust.c` (RANSAC, MAGSAC++ in
closed form for 2D residuals, Graph-Cut RANSAC, superansac = PROSAC +
MAGSAC++ + graph-cut LO + IRLS; independent inliers; degeneracy and
plausibility checks), `maxflow.c`, `estimate.c`; options `estimator=`,
`tile_model=`, `region_model=` (not `model=`: taken by the DISK
weights), `threshold=` (map units; not `max_sigma=`: sigma_max is
threshold / k as in the authors' magsac); outliers stay in POINTS with
status 0; `weight`, `residual`, `inlier` columns. Checks:
`tests/robust_check.c` (synthetic models and outliers, brute-force
minimum cut, leave-one-out against refits), `tests/robust_compare.py`
(same inliers as OpenCV's USAC_MAGSAC on six sets, models as accurate or
better), pytest 7/7 on the server (moved-object test). Eval in
`tests/eval/README.md`. Differences: the max-flow is Dinic's, not
Boykov-Kolmogorov's (simpler; the graphs are small); no SPRT (the exact
early stop on the best quality instead) and no P-NAPSAC (PROSAC and
uniform only); no TPS model (i.rectify -t fits it on the active points);
superansac's MAGSAC++ scoring differs from the paper's (a reparametrized
quality), the paper's and the authors' magsac form is used; parity with
superansac's bindings was replaced by the OpenCV comparison. Default
`estimator=superansac` (the user's decision: the addon is built for long
term use, not for one report); `estimator=none` gives v0's behaviour.
Threshold default 3 cells (1-1.5 cells for cross-modal pairs).

## Phase 2. Efficient LoFTR (semi-dense)

Source: `~/dev/EfficientLoFTR` at the parent of commit `07e9c14` (last
Apache 2.0 version). Weights: `eloftr_outdoor.ckpt` (Google Drive link
in the README, or the Hugging Face model `zju-community/efficientloftr`);
check their licence before automatic download, else read a user-given
path only.

2.1 **Loader**: `state_dict` of the Lightning checkpoint; RepVGG blocks
re-parameterized at load (3x3 + 1x1 + identity branches and their
BatchNorms folded into one 3x3 convolution), for both `full` and `opt`
configurations.

2.2 **Backbone** (RepVGG, input grey 0..1, 1/1, 1/2 and 1/8 features):
3x3 convolutions with stride 1 and 2 and ReLU, from the Phase 0 kernels.

2.3 **Coarse transformer**: 4 x (self, cross) AG-RoPE layers at 1/8:
depthwise strided convolution (aggregation, size 4) and max pooling of
keys/values, q/k/v projections, 2D RoPE, attention over the aggregated
tokens (blockwise kernel), merge, MLP with LeakyReLU, LayerNorms,
bilinear upsampling of the message.

2.4 **Coarse matching**: dual-softmax over all coarse token pairs and
mutual nearest neighbours above a threshold. A 1024-cell tile has 16,384
coarse tokens per image: the full 16k x 16k score matrix (1 GiB) is too
large, so compute it by blocks in two passes (row and column log-sum-exp
first, then the confidences and mutual maxima), never stored whole.
Border masking as the reference.

2.5 **Fine matching** (`RESOLUTION = (8, 1)`, `FINE_WINDOW_SIZE = 8`):
FPN fusion of the 1/8, 1/2 and 1/1 features (`FinePreprocess`: 1x1 and
3x3 convolutions, bilinear upsampling), an 8 x 8 window of image 0 and a
10 x 10 window of image 1 unfolded at full resolution around each coarse
match; first stage, dual-softmax between the two windows (cell-level
match); second stage, a 3 x 3 local regression on the last 8 feature
channels (softmax with temperature, spatial expectation): matches to a
fraction of a cell, with confidence.

2.6 **Reference comparison**: dumps per stage (backbone features,
transformer output, coarse matches, fine offsets) against the PyTorch
model on the same tile; identical coarse matches, fine offsets within
1e-3 cell.

2.7 **Integration**: `matcher=eloftr` -> Phase 1 estimator (AdaLAM not
needed: matches are already mutual; optional). Expected gains over DISK:
matches on texture-poor desert and tarmac, sub-cell precision without
the NCC refinement. Eval against v0 on the Tucson basemaps.

Memory per 1024-cell tile: backbone features about 0.3 GiB per image,
transformer small (1/8 tokens), coarse matching blockwise.

Status (2026-10-14): done, with these differences from the plan.
Translated from Transformers 4.57's EfficientLoFTRForKeypointMatching,
whose names the published weights use (`zju-community/efficientloftr`,
safetensors, Apache 2.0: downloaded when missing, as DISK's), checked
against the authors' code at ffd4a46; `pth_loader.c` reads safetensors.
Files `eloftr.c/.h`, `eloftr_kernels.cl` (blockwise dual softmax, fine
matching), `matcher_eloftr.c/.h`; only one (outdoor) configuration.
Transformers departs from the authors after the network (coarse pairs,
fine first-stage scale and its gather): the module and
`tests/reference_compare_eloftr.py` follow the authors. Parity on one
Google/ESRI tile: 14009/14009 coarse matches, fine positions within
6e-5 pixel but for 3 float ties. The general convolution now runs as
im2col + GEMM by blocks (`ops.c`, checked by two new `ops_check`
cases, 41/41): 16.2 s to 2.1 s per tile. AdaLAM is not offered after
ELoFTR. pytest 8/8 on the server (new: ELoFTR recovers a known shift).
Eval in `tests/eval/README.md`: 13-18 times the control points, full
coverage, lower basemap LOO p90, faster; but ELoFTR's own sub-cell
estimate leans to whole cells (the weights' behaviour, checked), so
the NCC refinement is needed for DISK-level accuracy. The user then
made `matcher=eloftr` with the refinement the default: the refinement
runs unless `-r` is given, `-s` is kept as a deprecated no-op.

## Phase 3. MatchAnything (cross-modality, ELoFTR weights)

Source: code and weights in the Hugging Face Space
`LittleFrog/MatchAnything` (`imcui/third_party/MatchAnything`), under the
Project Registration License.

3.1 Diff its ELoFTR model against the Apache 2.0 Efficient LoFTR. If the
architecture is the same (expected: the paper changes the training, not
the network), MatchAnything needs no code translation: Phase 2's code
loads its weights. Only the weights fall under the PRL: personal research
use needs no registration; never download them automatically or ship
them.

3.2 If the architecture differs (input normalization, resolution, layer
count), translate the differences from the paper, not the PRL code.

3.3 Eval on cross-modal pairs: hyperspectral composite against the ESRI
basemap (Tucson), thermal or SAR if available; compare with DISK and
ELoFTR outdoor weights.

Status (2026-10-14): done. 3.1: the Hugging Face conversion
`zju-community/matchanything_eloftr` (model card: Apache 2.0; the
authors' GitHub repository is now under the PRL) has the same tensors
and configuration as Efficient LoFTR's but `embedding_size` 26 x 26: the
authors' "npe", RoPE positions scaled by 832 over the input's long side,
which Transformers ignores. 3.2: the only change is that scale
(`eloftr_model.pos_scale`), on for `matcher=matchanything`; weights read
from `checkpoint=` or `~/models/matchanything`, downloaded there
when missing (Apache 2.0 on the Hugging Face model card; at first read
from a path only, changed by the user the same day).
Parity with Transformers + the scaling: 13976/13976 coarse matches,
fine positions within 6.2e-5 pixel. 3.3: in `tests/eval/README.md`;
MatchAnything equals Efficient LoFTR on these data (slightly more
cross-modal points, more moved-object leaks); no thermal or SAR pair was
available. Defaults unchanged.

## Phase 4. RoMa v2 (dense)

Sources: `~/dev/RoMaV2` (MIT) with a DINOv3 ViT-L/16 backbone whose code
and weights are under Meta's DINOv3 License Agreement (gated download).
Decision needed before starting: accept the DINOv3 licence for personal
research (weights read from a user-given path, never downloaded or
shipped by the module, backbone written from the architecture described
in the DINOv3 paper rather than translated from Meta's code), or start
with RoMa v1 (`~/dev/RoMa`, MIT, DINOv2 Apache 2.0) and add v2 later.
The plan assumes RoMa v2 as asked, keeping the parts shared with RoMa v1.

4.1 **Backbone**: ViT-L/16 with 2D RoPE and register tokens, features of
layers 4, 11, 17, 23 (`features.py`); at 512 x 512 input, 1,024 tokens:
blockwise attention from Phase 0, ViT blocks as in i.sam3.opencl.

4.2 **Matcher**: multi-view ViT (vit_base, alternating self/cross
attention, RoPE) on both images' tokens, DPT head (`dpt.py`): coarse warp
and certainty at 1/16 (anchor 512 x 512).

4.3 **Fine features and refiners**: `FineFeatures` convolutions, then the
ConvRefiner cascade (1/8, 1/4, 1/2, 1/1): `grid_sample` of the other
image's features at the current warp, local correlation in a window
around it (the custom CUDA kernel `local_corr`: a fused OpenCL kernel
computing the correlation volume on the fly, without storing it), the
convolution blocks, warp and certainty updates.

4.4 **Outputs**: dense warp and certainty at full tile resolution. Write
`dx`, `dy` and `certainty` rasters over the region (tiles blended with
weights in the overlaps), and control points sampled where the certainty
is high (RoMa's balanced sampling) for Phase 1 and the POINTS file.

4.5 **Reference comparison** per stage against the PyTorch model, then
eval: dense field against the synthetic ground truth and against DISK and
ELoFTR on the basemaps.

Memory: ViT-L weights about 1.2 GiB in float32 (float16 storage with
float32 accumulation where the device allows), activations of a 512 x 512
tile well under 2 GiB per buffer.

4.6 **Later**: RoMa v1 (DINOv2 ViT-L/14, its own matcher and refiners,
same correlation and refiner kernels), then MatchAnything's RoMa weights
on it (same PRL treatment as Phase 3).

Status (2026-10-14): RoMa v1 done first, as the user chose
(`roma.c`, `roma_kernels.cl`, `matcher_roma.c`, `matcher=roma`):
coarse pass at 560 x 560 and fine pass at the tile's size as the
authors' match(), one direction (image to reference), float32, the
Gaussian process solved on the host in double precision. Parity with
the authors' code (`tests/reference_compare_roma.py`): final warp within
4.6e-4 cell, certainty within 3e-5. The dense warp is sampled into
matches (best cell of each 8 x 8 block, certainty >= 0.5) for the
existing filters and robust stage; 4.4's dense `dx`/`dy`/certainty
rasters are not written yet. Eval in `tests/eval/README.md`: the most
accurate matcher on every test (synthetic 0.03-0.08 cell, basemap LOO
p90 0.25-0.32 m, cross-modal 3.2-3.7 m), the NCC refinement makes it
worse (so never applied to it), 2.2 times Efficient LoFTR's run time.
Made the default (long-term addon rule). Open: RoMa v2 (DINOv3 licence
decision), MatchAnything's RoMa weights (Google Drive only, Project
Registration License: the user's decision, and their model code has a
configurable backbone to check), dense rasters (4.4), the symmetric
mode.

## Phase 5. Dense match summarization

Source: Astermark, Heyden and Larsson (CVPR 2025), no code. For the
dense matches of Phase 4 (millions per tile):

5.1 For models linear in their parameters (affinity, polynomials), the
least-squares contribution of a block of matches is exactly summarized by
its normal-equation terms (A^T W A, A^T W b) and residual sum: summarize
per block (e.g. 16 x 16 cells), keep the block centre and count.

5.2 For the homography, summarize the DLT constraints in a 9 x 9 matrix
per block as the paper does for the epipolar case.

5.3 Run Phase 1's estimators on the summaries (MAGSAC++ scoring adapted
to block residuals), then refit on the full matches of the inlier blocks.
Eval: same model and accuracy as with all matches, 10-100 times faster.

## Phase 6. Other improvements (interleaved when convenient)

- `exclude=` vector map: no keypoints and no matches inside areas
  (aircraft outlines of the other date).
- Displacement field output for v.rectify's missing TPS: rasters `dx`,
  `dy` from the fitted TPS or the dense warp, and a small companion
  (or v.rectify patch, GPL file) that moves vector maps through a field.
- DISK speed: both images in one batch, Winograd 5x5 convolutions,
  float16 storage on devices that support it.
- Report and manual: per-matcher sections, eval tables.

## Milestones

| Release | Content | Check |
|---|---|---|
| v0 (done) | DISK + AdaLAM, tiles, filters, `-s` | kornia identical, pytest 5/5 |
| v0.1 (done) | Phase 0: kernel library, matcher interface, eval harness | v0 results unchanged, baseline report |
| v0.2 (done) | Phase 1: robust estimation stage | synthetic outlier tests, OpenCV USAC parity |
| v0.3 (done) | Phase 2: Efficient LoFTR | PyTorch parity per stage, eval vs v0 |
| v0.4 (done) | Phase 3: MatchAnything ELoFTR weights | cross-modal eval |
| v0.5 (v1 done) | Phase 4: RoMa v1, then v2 (dense field) | PyTorch parity, dense eval |
| v0.6 | Phase 5: summarization | speed and accuracy vs full matches |

## Decisions from the user (2026-10-14)

1. Phase 4 starts with RoMa v1 (MIT, DINOv2 Apache 2.0); RoMa v2 and
   DINOv3 later.
2. Efficient LoFTR's and MatchAnything's weights are downloaded when
   missing (MatchAnything's at first by path only; downloaded since the
   user noticed the Hugging Face conversion is Apache 2.0).
3. MatchAnything ships in the published addon, with the licence terms
   in the manual. The user's own use is personal
   research (no registration).
4. Order: robust stage first (done).
5. Defaults follow the evaluation for long-term addon use: after phase
   4, `matcher=roma` (RoMa's positions never refined), decided under
   that standing rule while the user was away (2026-10-14 night);
   `matcher=eloftr` is the previous default.
