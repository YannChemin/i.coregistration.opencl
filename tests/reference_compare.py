# SPDX-License-Identifier: Unlicense
"""Compare i.coregistration.opencl with the PyTorch reference (kornia).

Run the module with I_COREGISTRATION_OPENCL_DUMP=<dir>, then, in a Python
environment with torch and kornia (as RS_coreg_LoFTR's venv_kornia):

    python reference_compare.py <dir> [checkpoint] [tile]

For the tile (default 1), on the very input tensors the module used:

- DISK: kornia's DISK.from_pretrained("depth") (or the given checkpoint)
  with n = the module's keypoint count; keypoints compared as sets,
  heatmap scores and descriptors compared at the common keypoints;
- AdaLAM: kornia's match_adalam with RS_coreg_LoFTR's configuration
  (search_expansion 16, ransac_iters 256, force_seed_mnn off, LAFs of
  scale 96) on the module's own keypoints and descriptors, so that its
  matches can be compared one to one.

Not a pytest test: it needs torch, kornia and the DISK weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import kornia.feature as KF

D = 128


def load_kp(path):
    v = np.fromfile(path, dtype=np.float32).reshape(-1, 3 + D)
    return v[:, :2].astype(np.int64), v[:, 2], v[:, 3:]


def main():
    dump = Path(sys.argv[1])
    ckpt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    head = (dump / f"tile{t:03d}_matches.txt").read_text().splitlines()[0].split()
    info = dict(zip(head[1::2], map(int, head[2::2])))
    H, W, h, w = info["H"], info["W"], info["h"], info["w"]

    if ckpt:
        disk = KF.DISK()
        disk.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["extractor"])
        disk.eval()
    else:
        disk = KF.DISK.from_pretrained("depth")

    feats = {}
    for side in ("src", "ref"):
        img = np.fromfile(dump / f"tile{t:03d}_{side}_input.f32", dtype=np.float32)
        img = torch.from_numpy(img.reshape(1, 3, H, W))
        xy, score, desc = load_kp(dump / f"tile{t:03d}_{side}_kp.f32")
        feats[side] = (xy, score, desc)
        with torch.inference_mode():
            heat, dense = disk.heatmap_and_dense_descriptors(img)
            ref = disk(img, n=len(xy), window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
        heat = heat[0, 0, :h, :w].numpy()
        rxy = ref.keypoints.numpy().astype(np.int64)
        mine = {tuple(p) for p in xy}
        theirs = {tuple(p) for p in rxy}
        common = mine & theirs
        print(f"{side}: {len(xy)} keypoints, kornia {len(rxy)}, common {len(common)} "
              f"({100 * len(common) / max(len(theirs), 1):.2f} %)")
        hv = heat[xy[:, 1], xy[:, 0]]
        print(f"  heatmap at keypoints: max |diff| {np.abs(hv - score).max():.3e} "
              f"(scores {score.min():.3f}..{score.max():.3f})")
        dd = torch.nn.functional.normalize(dense[0, :, xy[:, 1], xy[:, 0]].T, dim=-1).numpy()
        cos = (dd * desc).sum(1)
        print(f"  descriptors: max |diff| {np.abs(dd - desc).max():.3e}, min cosine {cos.min():.6f}")

    k1, _, d1 = feats["src"]
    k2, _, d2 = feats["ref"]
    cfg = KF.adalam.get_adalam_default_config()
    cfg["force_seed_mnn"] = False
    cfg["search_expansion"] = 16
    cfg["ransac_iters"] = 256
    k1t = torch.from_numpy(k1).float()
    k2t = torch.from_numpy(k2).float()
    lafs1 = KF.laf_from_center_scale_ori(k1t[None], 96 * torch.ones(1, len(k1), 1, 1))
    lafs2 = KF.laf_from_center_scale_ori(k2t[None], 96 * torch.ones(1, len(k2), 1, 1))
    hw = torch.tensor([H, W])
    dists, idxs = KF.match_adalam(torch.from_numpy(d1), torch.from_numpy(d2), lafs1, lafs2,
                                  config=cfg, hw1=hw, hw2=hw)
    theirs = {tuple(p) for p in idxs.numpy().tolist()}
    m = np.loadtxt(dump / f"tile{t:03d}_matches.txt", ndmin=2)
    mine = {(int(a), int(b)) for a, b, _ in m}
    common = mine & theirs
    print(f"AdaLAM: module {len(mine)} matches, kornia {len(theirs)}, common {len(common)}; "
          f"only module {len(mine - theirs)}, only kornia {len(theirs - mine)}")


if __name__ == "__main__":
    main()
