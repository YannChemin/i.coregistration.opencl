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
- The user commits; leave changes uncommitted unless asked. At handoff,
  uncommitted: README.md, TODO.md, PLAN.md, this file, a test fix.

## Next step

PLAN.md phase 0 (shared kernel library, matcher interface, evaluation
harness), then phase 1 (robust estimation). Ask the user about the four
decisions at the end of PLAN.md before phases 3-4 (DINOv3 licence for
RoMa v2; download of ELoFTR/MatchAnything weights; MatchAnything in the
published addon or not; order). Their answers are not known yet.

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
  grass --tmp-project XY --exec python3 -m pytest tests`.
- Reference comparison: run the module with
  `I_COREGISTRATION_OPENCL_DUMP=<dir>`, then
  `~/dev/RS_coreg_LoFTR/venv_kornia/bin/python -I tests/reference_compare.py <dir> ~/models/disk/depth-save.pth`
  (that venv has kornia 0.8.3 over the system torch 2.6). Build one such
  comparison per new network, stage by stage.
- Laptop OpenCL is PoCL on the CPU only (one 1024 tile pair: about 4 min).
  Real runs go to the GPU server `yann@10.42.0.89` (debiantest, AMD
  Radeon Pro WX 7100, 16 GiB, Mesa rusticl: export
  `RUSTICL_ENABLE=radeonsi`; max buffer 2 GiB). Build there into
  `~/build/<module>_<tag>` (never overwrite the user's `~/dev` copies),
  `make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL`. Model weights
  live in `~/models/<name>` on both machines (mirrored).
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
