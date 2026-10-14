# TODO

Base articles are in `references/` (PDFs as downloaded). Code links come
from the articles themselves, or from a search when they give none.

## Matchers to add (alternatives to DISK + AdaLAM)

The current module is the translation of RS_coreg_LoFTR, which (despite
its name) runs DISK keypoints and AdaLAM, so control points sit on whole
cells (refined with `-s`). The matchers below are detector-free or
dense: they return matches to a fraction of a cell directly, and the
dense ones a displacement for every cell. Each would be a `matcher=`
choice, weights read from the published checkpoint as for DISK.

1. **Efficient LoFTR** (semi-dense).
   - Wang, Y., He, X., Peng, S., Tan, D., Zhou, X., 2024. Efficient LoFTR:
     semi-dense local feature matching with sparse-like speed. CVPR 2024,
     21666-21675. `references/2403.04765v2.pdf`.
   - Code and weights: <https://github.com/zju3dv/EfficientLoFTR> (also in
     Hugging Face Transformers as `EfficientLoFTR`).
   - What to port: RepVGG backbone (3x3 convolutions, reparameterized at
     load time), aggregated attention with adaptive token selection
     (attention on pooled tokens), coarse dual-softmax matching, the
     two-stage fine correlation (sub-pixel). About 2.5x faster than LoFTR;
     the attention is the main new OpenCL work (softmax and matrix
     products; the i.sam.opencl kernels can be reused).
   - Benefit here: sub-cell matches without the NCC refinement, and
     matches on texture-poor ground (desert, tarmac) where DISK finds few
     keypoints.
   - The original LoFTR (Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X.,
     2021. LoFTR: detector-free local feature matching with transformers.
     CVPR 2021, <https://github.com/zju3dv/LoFTR>), the name of the
     repository, could come with it at little extra cost (same structure,
     full linear attention); its PDF is not in `references/` yet.

2. **RoMa v2** (dense).
   - Edstedt, J., Nordström, D., Zhang, Y., Bökman, G., Astermark, J.,
     Heyden, A., Larsson, V., Wadenbäck, M., Felsberg, M., Kahl, F., 2025.
     RoMa v2: harder better faster denser feature matching. arXiv
     2511.15706 (v2 and v3 in `references/`).
   - Code and weights: <https://github.com/Parskatt/RoMaV2>.
   - What to port: frozen DINOv3 ViT backbone (the ViT blocks of
     i.sam.opencl / i.sam3.opencl are close), the matcher, the decoupled
     refiners, whose memory-saving custom CUDA kernel needs an OpenCL
     equivalent; output: a warp and a certainty for every cell.
   - Benefit here: a dense displacement field between two dates with a
     certainty map, i.e. the co-registration itself rather than control
     points; best accuracy on hard pairs. Heaviest of the three (ViT-L
     sized backbone): GPU only, tiled.
   - Output: besides control points sampled where the certainty is high,
     write the field as two rasters (dx, dy) and the certainty.

3. **MatchAnything** (cross-modality).
   - He, X., Yu, H., Peng, S., Tan, D., Shen, Z., Bao, H., Zhou, X., 2025.
     MatchAnything: universal cross-modality image matching with
     large-scale pre-training. arXiv 2501.07556.
     `references/2501.07556v1.pdf`.
   - Code and weights: <https://github.com/zju3dv/MatchAnything> (project
     page <https://zju3dv.github.io/MatchAnything/>).
   - Not a new architecture: a pre-training framework on synthetic
     cross-modal pairs, applied to two existing matchers, ELoFTR
     (semi-dense) and RoMa (dense; the first RoMa, DINOv2 backbone, not
     RoMa v2), with one set of weights for all modalities.
   - Remote sensing tasks it was evaluated on: thermal-visible
     (satellite and aerial), visible-SAR, and visible-vectorized map.
   - Benefit here: different sensors and modalities (hyperspectral band
     composites against an RGB basemap, thermal, SAR, maps). The ELoFTR
     variant reuses item 1's code with other weights, so it comes nearly
     free once Efficient LoFTR is in; the RoMa variant needs the first
     RoMa's matcher besides item 2 (shared backbone family, different
     refiners).

## Robust estimation (after AdaLAM)

AdaLAM keeps locally affine-consistent matches but fits no model over
the image, and the repository's `cv2.findFundamentalMat(USAC_MAGSAC)`
only counted inliers for its plot (a fundamental matrix is the wrong
model for orthoimages anyway). The control points then go straight into a
thin plate spline, which reproduces every remaining outlier. A robust
global stage would help: per tile an affinity or homography, over the
region a polynomial (orders 1-3, as i.rectify) or a robust TPS, with the
residual of each point written to `points=` and to the report.

4. **MAGSAC++** for scoring and the final fit.
   - Barath, D., Noskova, J., Ivashechkin, M., Matas, J., 2020. MAGSAC++,
     a fast, reliable and accurate robust estimator. CVPR 2020,
     1304-1312. `references/1912.05909v1.pdf`.
   - Code: <https://github.com/danini/magsac> (also OpenCV `USAC_MAGSAC`).
   - Take: the threshold-free model quality (noise scale marginalized),
     the IRLS final fit (weights per point instead of an inlier cut-off:
     gives the per-point weight to write out), and the Progressive NAPSAC
     sampler (local samples first, then global), which suits tiles where
     nearby points share a local shift.

5. **Graph-Cut RANSAC** for spatially coherent inliers.
   - Barath, D., Matas, J., 2018. Graph-Cut RANSAC. CVPR 2018, 6733-6741
     (PDF not in `references/` yet).
   - Code: <https://github.com/danini/graph-cut-ransac>.
   - Take: the graph-cut local optimization, which labels neighbouring
     points together. Matches on one moved aircraft form a coherent
     cluster that disagrees with the surrounding ground: the spatial term
     separates them better than `max_shift=`/`tolerance=`.

6. **VSAC** for speed and degenerate tiles.
   - Ivashechkin, M., Barath, D., Matas, J., 2021. VSAC: efficient and
     accurate estimator for H and F. ICCV 2021, 15243-15252.
     `references/[2106.10240] VSAC_ Efficient and Accurate Estimator for H and F.pdf`.
   - Code: in OpenCV's USAC framework (`cv::USAC_*`); the paper cites
     <https://github.com/danini/magsac> and
     <https://github.com/danini/graph-cut-ransac> for the parts it builds on.
   - Take: independent inliers (points too close to each other count
     once; dense matches on one roof or one aircraft no longer outvote
     the rest), SPRT verification (adaptive, stops bad models early),
     local optimization run about once, and the dominant-plane handling,
     which matters on flat terrain where a homography and an affinity
     are nearly degenerate.

7. **SupeRANSAC** as the recipe for the whole stage.
   - Barath, D., 2025. SupeRANSAC: one RANSAC to rule them all. arXiv
     2506.04803. `references/2506.04803v1.pdf`.
   - Code: <https://github.com/danini/superansac>.
   - Take: its analysis of which details make RANSAC work per problem
     (homography and affine included): PROSAC / P-NAPSAC sampling,
     MAGSAC++ scoring, GC-RANSAC local optimization, a final IRLS, and
     degeneracy checks on samples and models. Implement that pipeline
     once (C, CPU; the hypotheses can run in OpenCL like AdaLAM's) for
     affinity, homography and the polynomial orders.

8. **Dense match summarization** before estimation (with item 2).
   - Astermark, J., Heyden, A., Larsson, V., 2025. Dense match
     summarization for faster two-view estimation. CVPR 2025, 1093-1102.
     `references/2506.02893v1.pdf`.
   - Code: none found (neither the article nor a search gives a
     repository).
   - Take: replace a dense set of matches by fewer summarized ones, each
     carrying the constraints of its neighbours (a 9x9 matrix for the
     epipolar case), for 10-100x faster RANSAC at the same accuracy. To
     adapt to the 2D models used here (affinity, homography, polynomial):
     summarize RoMa v2's millions of matches per block before items 4-7.

## Other improvements

- `exclude=` vector map: drop keypoints inside areas (aircraft or
  vehicle outlines of another date), complementing `max_shift=`.
- Per-point residuals of the global fit and leave-one-out residuals of
  the TPS in `points=`, and an RMS in the report.
- *v.rectify* has no thin plate spline: write a displacement field
  (dx, dy rasters from the TPS) so that vector maps can be moved with it,
  or add TPS to *v.rectify*.
- Run DISK on both images in one batch, and speed up the 5x5
  convolutions (Winograd, half precision where the device supports it).

## Note

The conversation shared at
<https://claude.ai/share/525baa9e-f3a2-4af6-941c-be180405e540> could not
be read from here (the page is rendered by JavaScript). Items 4-8 come
from the articles in `references/`; anything in that conversation not
covered here still needs adding (paste or save it into `references/`).
