# i.coregistration.opencl: notes for a session taking over

Read `PLAN.md` (the implementation plan, phases 0-6) and `TODO.md` (the
items, sources, licences) first. This file holds what they do not: the
state, how to build, test and run, and what is still to be decided.

## State (2026-10-14)

- v0 done: DISK (U-Net on OpenCL) + AdaLAM (C, CPU), tiles, `max_shift=`,
  `tolerance=`, `-s` sub-cell NCC refinement, group POINTS + `points=`
  vector. Files: `main.c`, `disk.c/.h` + `disk_kernels.cl`, `adalam.c/.h`,
  `gcp.c/.h`, `image.c/.h`; reused from `~/dev/i.sam.opencl` (same author,
  Unlicense): `ocl_backend`, `pth_loader` (extended here to the legacy
  PyTorch format and to any single wrapper key), `download`, `progress`.
- Verified against kornia on one 1024 x 1024 Tucson tile (Google vs
  ESRI): same 2048 keypoints per image, heatmap within 7e-4, descriptors
  within 3e-6, AdaLAM 538/538 identical matches
  (`tests/reference_compare.py`). pytest: 5 passed.
- Production run: Tucson basemaps on the GPU server, about 17 min per
  basemap (252 tiles), about 210,000 matches each. The control points
  showed a sharp seam in the ESRI mosaic (south-eastern patch about 1.6 m
  off). That work belongs to the Tucson report, not to this repository
  (`~/Documents/LaTex/Report/20261008_Spectra_Forensic_Tucson_boneyard`,
  scripts 22_coreg_server.sh, 04c_coreg_field.py, basemap_field.py).
- The user commits (often, as "update"); leave changes uncommitted
  unless asked.
- Phase 0 done (2026-10-14): `ops.cl`/`ops.c`/`ops.h` (generic kernels,
  checked against PyTorch by `tests/ops_check.c` + `tests/ops_compare.py`:
  40/40 on the WX 7100), `matcher.h` interface with `matcher_disk.c`,
  `matcher=` option (one choice), `--verbose` stage timing,
  `tests/eval/` harness. Bit-identical to v0 on 25 Tucson tiles (all
  dumps and the control point table). Baseline numbers in
  `tests/eval/README.md`.
- Evaluation data (confidential, never in the repository): Tucson
  basemaps `esri_worldimagery_aoi.tif`, `google_satellite_aoi.tif`,
  `bing_aerial_aoi.tif` (0.5 m, 17200 x 12600) in the report's `data/`;
  AVIRIS cube `~/RSDATA/cropped_tucson.zarr` (424 bands, 11.3 m, read by
  `tests/eval/prepare_hyperspectral.py`). Phase 0 used 4072-cell crops
  of the basemaps (`gdal_translate -projwin 510500 3560500 512536
  3558464`).

- Phase 1 done (2026-10-14): robust estimation (`models.c`, `robust.c`,
  `maxflow.c`, `estimate.c`; options `estimator=` (default superansac),
  `tile_model=`, `region_model=`, `threshold=`. Checked by
  `tests/robust_check.c` (CPU, also run by pytest when built) and
  `tests/robust_compare.py` (OpenCV USAC_MAGSAC); pytest 7/7 on the
  server. Results in `tests/eval/README.md`: superansac per tile removes
  moved-object matches (998 -> 7) and halves the basemap LOO p90;
  cross-modal needs threshold 1-1.5 cells. The user chose superansac as
  the default: decisions follow long-term addon use, not one report;
  `estimator=none` reproduces v0 (and the reference comparison).

- Phase 2 done (2026-10-14): `matcher=eloftr` (Efficient LoFTR,
  Hugging Face safetensors weights in `~/models/eloftr` on both
  machines), im2col convolutions, `tests/reference_compare_eloftr.py`
  (run with the repository's `venv_ref`: torch 2.6 + transformers
  4.57), eval in `tests/eval/README.md`.
  `I_COREGISTRATION_OPENCL_ELOFTR=<model.safetensors>` enables its
  pytest. The user chose `matcher=eloftr` + refinement as the default:
  the NCC refinement now runs unless `-r`; `-s` is a deprecated no-op
  (warning) kept for existing scripts (the Tucson report's use it).

- Phase 3 done (2026-10-14): `matcher=matchanything` = the ELoFTR
  network with `zju-community/matchanything_eloftr` weights (in
  `~/models/matchanything` on both machines; downloaded when
  missing, Apache 2.0 on the model card) and RoPE positions scaled by 832 / tile long side (`npe`).
  Not better than eloftr on the Tucson data; defaults unchanged.
  `I_COREGISTRATION_OPENCL_MATCHANYTHING=<model.safetensors>` enables
  its pytest. The Space clone is `~/dev/MatchAnything_space`.

- Phase 4 RoMa v1 done (2026-10-14 night): `matcher=roma`, the new
  default (best on every eval, 2.2x ELoFTR's time; its positions are
  never NCC-refined). Weights `roma_outdoor.pth` + `dinov2_vitl14_pretrain.pth`
  in `~/models/roma` on both machines (with SHA256SUMS);
  `I_COREGISTRATION_OPENCL_ROMA=<that folder>` enables its pytest.
  Parity: `tests/reference_compare_roma.py <dump> ~/models/roma ~/dev/RoMa`
  run with `~/dev/RS_coreg_LoFTR/venv_kornia/bin/python` (not `-I`: einops
  is in the user site; loguru is stubbed). The pth loader now skips
  integer tensors (BatchNorm's num_batches_tracked) and unwraps
  `_rebuild_from_type_v2` (DINOv2's file).
- Dense rasters (2026-10-15): `dense=<base>` (roma only) writes
  `<base>_dx`, `_dy` (map units) and `_certainty`, tiles blended
  (`dense.c`; matchers expose a `field` callback). Eval:
  `eval_synthetic.py --dense` (tests/eval/README.md).
- `matcher=matchanything_roma` (2026-10-15): RoMa v1's code with
  MatchAnything's weights (`~/models/matchanything/matchanything_roma.ckpt`,
  never downloaded by the module, PRL; inputs not ImageNet-normalized;
  DINOv2 from `~/models/roma`). Parity 4.6e-4 cell; less accurate than
  `roma` on every Tucson test (tests/eval/README.md).
  `I_COREGISTRATION_OPENCL_MATCHANYTHING_ROMA=<ckpt>` enables its pytest.
- `matcher=roma_v2ref` (2026-10-15): RoMa v1 to its 1/8 refiner, then
  RoMa v2's refiners (no DINOv3: the user rejects its licence). Weights
  `romav2_refiners.pt` from https://github.com/YannChemin/romav2-refiners
  (public, created at the user's request; downloaded by the module into
  `$ROMAV2_MODELS_DIR` or `~/models/romav2`). Parity
  `tests/reference_compare_roma2.py` (venv_kornia, not `-I`): 1.4e-3
  cell. Best or equal on every Tucson test, 20 % faster than roma; the
  default since 2026-10-15 (user's decision). Its predicted standard
  deviation (det(P)^(-1/4) of RoMa v2's precision) is reported as
  `points=` column `sd` and `dense=` raster `_sd`; calibration in
  tests/eval/README.md (ranks real residuals loosely, too large on
  synthetic warps); used only by `max_sd=` (drop matches above it:
  0.2 m on the basemaps improved loop closure p90 0.160 -> 0.131 m).
  `I_COREGISTRATION_OPENCL_ROMAV2=<folder>` enables its pytest.
- `max_cycle=` (2026-10-15, RoMa matchers): the reverse warp too (twice the
  time), matches whose round trip misses by more are dropped; `points=`
  column `cycle`. Basemaps max_cycle=0.05-0.1 m: loop closure p90 0.160 ->
  0.120-0.131 m with more coverage than max_sd=. Option, not default.
  Swapped-role runs showed the AVIRIS 3-4 m east offset is in the data.
  The Tucson report now reads `~/RSDATA/cropped_tucson_coreg.zarr` (the
  cube translated by -3.745 m east, +1.220 m north to ESRI; report
  backup in its `backup/pre_romacoreg_2026-10-15`).

## Next step

Licence policy (user, 2026-10-15): the default run downloads its
permissively licensed weights itself; weights under particular licences
are opt-in matchers, never downloaded by the module: the manual and the
error message give the URL, the licence and the path. Next: RoMa v2
(`romav2.0.1.pt` from github.com/Parskatt/RoMaV2 release v2.0.1, in
`~/models/romav2` on both machines). The user does not accept the DINOv3
License (California law, Meta may change the terms): never load the
`f.*` tensors (DINOv3) of that file; done as `matcher=roma_v2ref`.
Next: phase 5.

## Build, test, run

- Local GRASS is `/usr/local/grass86` (8.6 dev); `~/dev/grass` is an older
  dist build, do not test against it.
  `make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL` installs into
  `/usr/local/grass86/bin`. Docs: the `.md` is the source, the `.html` is
  regenerated with pandoc (see the commands used in git history or:
  `pandoc -f gfm -t html --no-highlight --wrap=preserve`, then `<strong>`
  to `<b>`, code blocks to `<div class="code"><pre>`, `.md` links to
  `.html`); `make` fails without the `.html`.
- Tests: `I_COREGISTRATION_OPENCL_CHECKPOINT=~/models/disk/depth-save.pth
  I_COREGISTRATION_OPENCL_ELOFTR=~/models/eloftr/model.safetensors
  I_COREGISTRATION_OPENCL_MATCHANYTHING=~/models/matchanything/model.safetensors
  I_COREGISTRATION_OPENCL_ROMA=~/models/roma
  I_COREGISTRATION_OPENCL_MATCHANYTHING_ROMA=~/models/matchanything/matchanything_roma.ckpt
  I_COREGISTRATION_OPENCL_ROMAV2=~/models/romav2
  grass --tmp-project XY --exec python3 -m pytest tests`. To test a
  build that is not installed (the server builds), also set
  `I_COREGISTRATION_OPENCL_BINDIR=<build>/bin`: the GRASS session puts the
  installation's bin first in PATH, so a PATH prefix is not enough (the
  phase 0 pytest run tested the installed v0 by mistake). Build
  `make ... robust_check` too, or its test is skipped.
- Reference comparison: run the module with
  `I_COREGISTRATION_OPENCL_DUMP=<dir>`, then
  `~/dev/RS_coreg_LoFTR/venv_kornia/bin/python -I tests/reference_compare.py <dir> ~/models/disk/depth-save.pth`
  (that venv has kornia 0.8.3 over the system torch 2.6). Build one such
  comparison per new network, stage by stage.
- Laptop OpenCL is PoCL on the CPU only (one 1024 tile pair: about 4 min).
  The user wants every OpenCL run (tests included) on the GPU server, not
  on the laptop (faster, and it does not clutter the PC); build locally
  only to check compilation. The server is `yann@10.42.0.89`
  (debiantest, AMD Radeon Pro WX 7100, 16 GiB, Mesa rusticl: export
  `RUSTICL_ENABLE=radeonsi`; max buffer 2 GiB). Build there into
  `~/build/<module>_<tag>` (never overwrite the user's `~/dev` copies),
  `make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL`, adding
  `BIN=$HOME/<dir>/bin HTMLDIR=... MDDIR=... MANBASEDIR=...` so that the
  installed module (/usr/local/grass86/bin, used by the Tucson report)
  stays v0; `tests/eval/run_server.sh` does all this. Inside a GRASS
  session, put the build's bin first in PATH from within the session
  (`grass ... --exec bash -c 'PATH=...:$PATH ...'`): GRASS prepends its
  own bin. No torch on the server: run `ops_check` there and
  `tests/ops_compare.py` on the laptop. The first run after a kernel
  change takes about 3 min more (rusticl compiles, then caches). Model
  weights live in `~/models/<name>` on both machines (mirrored).
- Confidential data: anything copied to the server goes to a private
  `/tmp/<job>` folder (mode 700) and is deleted after the run; check no
  `.gislock` or dead `/tmp/grass8-*` is left. Detach long remote runs
  (`ssh -f -n ... setsid nohup ...`) and poll.

## Sources (cloned in ~/dev)

EfficientLoFTR (translate from the parent of commit 07e9c14, the last
Apache 2.0 version), RoMaV2 (MIT; its ViT backbone code and DINOv3
weights are under the DINOv3 License Agreement), RoMa (MIT, DINOv2),
MatchAnything (no code in the repository: code and weights are in the
Hugging Face Space LittleFrog/MatchAnything, Project Registration
License), LoFTR (Apache 2.0), magsac and graph-cut-ransac (new BSD, but
their bundled GCoptimization/maxflow are research-only: write the
max-flow from the Boykov-Kolmogorov paper), superansac (MIT, same
bundled graph-cut caveat), disk and kornia (Apache 2.0). Articles in
`references/`.

## Conventions

The user's global CLAUDE.md applies (GRASS addon conventions, Unlicense,
clang-format LLVM 4-space Stroustrup, pytest, MetaMonitor through
`G_percent`, fail loudly). References in docs: international
peer-reviewed only. The work is private and confidential.
