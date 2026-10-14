# SPDX-License-Identifier: Unlicense
"""Compare i.coregistration.opencl's Efficient LoFTR with the PyTorch reference.

Run the module with matcher=eloftr and I_COREGISTRATION_OPENCL_DUMP=<dir>,
then, in a Python environment with torch and transformers >= 4.57 (as the
repository's venv_ref):

    python -I reference_compare_eloftr.py <dir> <weights folder> [tile]

The weights folder holds config.json and model.safetensors of
zju-community/efficientloftr or zju-community/matchanything_eloftr. For
the tile (default 1), on the very grey
images the module used, stage by stage against
EfficientLoFTRForKeypointMatching:

- backbone: the 1/8 RepVGG features of both images;
- coarse: the transformer output, with the RoPE positions scaled as the
  module did (pos_scale in the dump: the authors' "npe", which
  Transformers lacks);
- coarse matches: mutual maxima of the dual softmax above the threshold,
  away from the border, as the authors' CoarseMatching (pairs (i, j));
  compared as sets, with their confidences;
- fine: the fused full-resolution features;
- fine matches: the authors' two-stage fine matching on the module's
  coarse matches (so that the positions compare one to one), in pixels.

Transformers departs from the authors' code after the network: its
_get_matches_from_scores pairs the row and column maxima of different
cells (index k of image 0's list belongs to cell k of image 1, and the
converse), and its fine matching differs (see fine_match). Both are done
here as the authors do.

Not a pytest test: it needs torch, transformers and the weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import EfficientLoFTRForKeypointMatching

D = 256
FD = 64


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = float(np.abs(a - b).max())
    return m, m / max(float(np.abs(b).max()), 1e-30)


def report(name, a, b):
    m, r = rel(a, b)
    print(f"{name:22s} max |diff| {m:.3e}  relative {r:.3e}")


def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def rope(inv_freq, cfg, h, w, scale):
    """Transformers' rotary embedding with positions times scale (1:
    Transformers' own; 832 / long side: the authors' npe)."""
    k, s = cfg.q_aggregation_kernel_size, cfg.q_aggregation_stride
    eh, ew = (h - k) // s + 1, (w - k) // s + 1
    i = torch.arange(1, eh + 1, dtype=torch.float32)[:, None, None] * scale
    j = torch.arange(1, ew + 1, dtype=torch.float32)[None, :, None] * scale
    f = inv_freq.reshape(-1)
    emb = torch.zeros(1, eh, ew, D // 2)
    emb[..., 0::2] = i * f
    emb[..., 1::2] = j * f
    return (
        emb.cos().repeat_interleave(2, dim=-1),
        emb.sin().repeat_interleave(2, dim=-1),
    )


def coarse_pairs(model, feats):
    """Mutual nearest neighbours of the dual softmax (authors' convention)."""
    cfg = model.config
    _, _, c, h, w = feats.shape
    f = feats.permute(0, 1, 3, 4, 2).reshape(1, 2, -1, c) / c**0.5
    sim = f[:, 0] @ f[:, 1].transpose(-1, -2) / cfg.coarse_matching_temperature
    conf = torch.softmax(sim, 1) * torch.softmax(sim, 2)
    mask = conf > cfg.coarse_matching_threshold
    b = cfg.coarse_matching_border_removal
    m5 = mask.view(1, h, w, h, w)
    m5[:, :b] = False
    m5[:, :, :b] = False
    m5[:, :, :, :b] = False
    m5[:, :, :, :, :b] = False
    m5[:, -b:] = False
    m5[:, :, -b:] = False
    m5[:, :, :, -b:] = False
    m5[:, :, :, :, -b:] = False
    mask = m5.view(1, h * w, h * w)
    mask &= conf == conf.max(2, keepdim=True).values
    mask &= conf == conf.max(1, keepdim=True).values
    ok, j = mask.max(2)
    i = torch.nonzero(ok[0])[:, 0]
    j = j[0, i]
    return i.numpy(), j.numpy(), conf[0, i, j].numpy()


def fine_match(cfg, feat0, feat1, c0, c1):
    """Two-stage fine matching of the authors' FineMatching (commit ffd4a46).

    Transformers' version differs: it scales the first stage by the 56
    sliced channels instead of all 64, and its first stage gathers the
    window positions over the matches axis (always position 0).
    """
    m, ww, c = feat0.shape
    W = int(ww**0.5)
    sd = cfg.fine_matching_slice_dim
    f0, f1 = feat0[..., :-sd] / c**0.5, feat1[..., :-sd] / c**0.5
    ff0, ff1 = feat0[..., -sd:], feat1[..., -sd:] / sd**0.5
    conf = f0 @ f1.transpose(1, 2)
    conf_ff = ff0 @ ff1.transpose(1, 2)
    sm = torch.softmax(conf, 1) * torch.softmax(conf, 2)
    sm = sm.reshape(m, ww, W + 2, W + 2)[..., 1:-1, 1:-1].reshape(m, -1)
    idx = sm.max(-1).indices
    idx_l, idx_r = idx // ww, idx % ww
    pos = lambda k: torch.stack([k % W, k // W], -1).float() - W // 2 + 0.5
    k0 = c0 + pos(idx_l)
    k1 = c1 + pos(idx_r)
    # The 3 x 3 window: inner indices used as 10 x 10 ones, -1 wrapping.
    ii, jj = idx_r // W, idx_r % W
    d = torch.arange(-1, 2)
    ar = torch.arange(m)[:, None, None]
    sel = conf_ff.reshape(m, ww, W + 2, W + 2)[
        ar,
        idx_l[:, None, None],
        ii[:, None, None] + d[None, :, None],
        jj[:, None, None] + d[None, None, :],
    ].reshape(m, 9)
    heat = torch.softmax(sel / cfg.fine_matching_regress_temperature, -1).reshape(m, 3, 3)
    ex = (heat.sum(1) * d).sum(-1)
    ey = (heat.sum(2) * d).sum(-1)
    k1 = k1 + torch.stack([ex, ey], -1)
    return torch.cat([k0, k1], -1).numpy()


def main():
    dump = Path(sys.argv[1])
    wdir = sys.argv[2]
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    lines = (dump / f"tile{t:03d}_eloftr_matches.txt").read_text().splitlines()
    head = lines[0].split()
    info = dict(zip(head[1::2], map(float, head[2::2])))
    H, W = int(info["H"]), int(info["W"])
    pos_scale = info.get("pos_scale", 1.0)
    h, w = H // 8, W // 8
    rows = np.array([list(map(float, ln.split())) for ln in lines[1:]]).reshape(-1, 7)
    mi, mj = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
    mconf, mkp = rows[:, 2], rows[:, 3:]

    model = EfficientLoFTRForKeypointMatching.from_pretrained(wdir)
    model.config._attn_implementation = "eager"
    model.eval()
    grey = np.stack(
        [load(dump / f"tile{t:03d}_eloftr_grey_{k}.f32", (1, H, W)) for k in (0, 1)]
    )
    px = torch.from_numpy(grey)  # (2, 1, H, W)

    with torch.inference_mode():
        net = model.efficientloftr
        feats = net.backbone(px)
        coarse_in, residual = feats[-1], feats[:-1]
        for k in (0, 1):
            report(
                f"backbone_{k}",
                load(dump / f"tile{t:03d}_eloftr_backbone_{k}.f32", (D, h, w)),
                coarse_in[k].numpy(),
            )
        cos, sin = rope(net.rotary_emb.inv_freq, model.config, h, w, pos_scale)
        cos = cos.expand(2, -1, -1, -1).reshape(2, -1, D)
        sin = sin.expand(2, -1, -1, -1).reshape(2, -1, D)
        coarse = net.local_feature_transformer(
            coarse_in.reshape(1, 2, D, h, w), position_embeddings=(cos, sin)
        )
        for k in (0, 1):
            report(
                f"coarse_{k}",
                load(dump / f"tile{t:03d}_eloftr_coarse_{k}.f32", (D, h, w)),
                coarse[0, k].numpy(),
            )

        ri, rj, rconf = coarse_pairs(model, coarse)
        ref = {(a, b): c for a, b, c in zip(ri, rj, rconf)}
        mod = {(a, b): c for a, b, c in zip(mi, mj, mconf)}
        common = sorted(set(ref) & set(mod))
        print(
            f"coarse matches         module {len(mod)}, reference {len(ref)}, "
            f"common {len(common)}"
        )
        for name, only in (("module", set(mod) - set(ref)), ("reference", set(ref) - set(mod))):
            if only:
                print(
                    f"  only in {name}: {len(only)}, confidences "
                    + ", ".join(
                        f"{(mod if name == 'module' else ref)[p]:.4f}" for p in sorted(only)[:8]
                    )
                )
        if common:
            report(
                "match confidence",
                [mod[p] for p in common],
                [ref[p] for p in common],
            )

        # Fine features on the reference's own coarse features.
        rl = model.refinement_layer
        fine = rl.forward_pyramid(coarse.reshape(2, D, h, w) / D**0.5, list(reversed(residual)))
        for k in (0, 1):
            report(
                f"fine_{k}",
                load(dump / f"tile{t:03d}_eloftr_fine_{k}.f32", (FD, H, W)),
                fine[k].numpy(),
            )
        if not len(mi):
            return

        # Fine matching on the module's coarse matches.
        fw = model.config.fine_kernel_size
        f0 = torch.nn.functional.unfold(fine[0:1], kernel_size=fw, stride=8, padding=0)
        f0 = f0.reshape(FD, fw * fw, -1).permute(2, 1, 0)
        f1 = torch.nn.functional.unfold(fine[1:2], kernel_size=fw + 2, stride=8, padding=1)
        f1 = f1.reshape(FD, (fw + 2) ** 2, -1).permute(2, 1, 0)
        i_t, j_t = torch.from_numpy(mi), torch.from_numpy(mj)
        c0 = torch.stack([i_t % w, i_t // w], -1).float() * 8
        c1 = torch.stack([j_t % w, j_t // w], -1).float() * 8
        kp = fine_match(model.config, f0[i_t], f1[j_t], c0, c1)
        d = np.abs(mkp - kp)
        print(
            f"fine matches           {len(kp)}: max |diff| {d.max():.3e} px, "
            f"p99 {np.percentile(d, 99):.3e}, > 1e-3: {(d.max(1) > 1e-3).sum()}"
        )


if __name__ == "__main__":
    main()
