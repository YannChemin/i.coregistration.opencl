# SPDX-License-Identifier: Unlicense
"""Compare i.coregistration.opencl's RoMa with the authors' PyTorch model.

Run the module with matcher=roma and I_COREGISTRATION_OPENCL_DUMP=<dir>,
then, in a Python environment with torch, torchvision, kornia and
OpenCV (RoMa's imports; loguru is stubbed):

    python reference_compare_roma.py <dir> <weights folder> <RoMa source> [tile]
        [--weights matchanything_roma.ckpt]

The weights folder holds roma_outdoor.pth and dinov2_vitl14_pretrain.pth;
the RoMa source is a clone of https://github.com/Parskatt/RoMa (romatch,
commit 77f8d68). With --weights, RoMa's weights come from that file
instead, MatchAnything's Lightning checkpoint included (matcher=
matchanything_roma: the dumped inputs are then not normalized, which
the comparison inherits). For the tile (default 1), on the very normalized images
the module used (its coarse 560 x 560 pair and its full-size pair):

- dino: DINOv2's normalized patch tokens of both images;
- gp, cls: the Gaussian process posterior and the coarse warp and
  certainty of the coordinate decoder;
- p0_s16 .. p0_s1: warp and certainty after each refiner of the coarse
  pass, the reference running its own chain from the same images;
- p1_s8 .. p1_s1: the same for the fine pass, started from the module's
  coarse-pass warp (so that each pass compares on equal inputs);
- final: warp and certainty as RoMa's match() returns them.

Warp differences are given in cells of the scale (normalized difference
times half the side). The reference runs on the CPU in float32
(automatic mixed precision is off on the CPU), with RoMa's native local
correlation (use_custom_corr=False).

Not a pytest test: it needs torch, the weights and RoMa's source.
"""

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def stub_loguru():
    """RoMa logs through loguru, not needed here."""
    mod = types.ModuleType("loguru")

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    mod.logger = Logger()
    sys.modules.setdefault("loguru", mod)


def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def report(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    rel = d.max() / max(float(np.abs(b).max()), 1e-30)
    print(
        f"{name:16s} max |diff| {d.max():.3e}  p99 {np.percentile(d, 99):.3e}  "
        f"median {np.median(d):.3e}  relative {rel:.3e}"
    )


def report_flow(name, mod, ref):
    """Warp (2, h, w) difference in cells of its own grid."""
    h, w = ref.shape[-2:]
    d = np.abs(np.asarray(mod, np.float64) - np.asarray(ref, np.float64))
    d[0] *= w / 2
    d[1] *= h / 2
    d = d.max(0)
    print(
        f"{name:16s} max {d.max():.3e} cells  p99 {np.percentile(d, 99):.3e}  "
        f"median {np.median(d):.3e}  > 0.1: {(d > 0.1).mean() * 100:.3f} %"
    )


def load_weights(path):
    """RoMa's state_dict, unwrapped from MatchAnything's checkpoint."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    pre = "matcher.model."
    return {k.removeprefix(pre): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("weights_dir", type=Path)
    ap.add_argument("roma_source")
    ap.add_argument("tile", type=int, nargs="?", default=1)
    ap.add_argument("--weights", type=Path)
    args = ap.parse_args()
    dump, wdir, t = args.dump, args.weights_dir, args.tile
    stub_loguru()
    sys.path.insert(0, args.roma_source)
    torch.set_float32_matmul_precision("highest")
    from romatch.models.model_zoo.roma_models import roma_model
    from romatch.utils.utils import cls_to_flow_refine

    head = (dump / f"tile{t:03d}_roma.txt").read_text().split()
    info = dict(zip(head[0::2], map(int, head[1::2])))
    H, W, C = info["H"], info["W"], info["coarse"]
    pre = f"tile{t:03d}_roma_"

    model = roma_model(
        resolution=(C, C),
        upsample_preds=True,
        device="cpu",
        weights=load_weights(args.weights or wdir / "roma_outdoor.pth"),
        dinov2_weights=torch.load(
            wdir / "dinov2_vitl14_pretrain.pth", map_location="cpu"
        ),
        amp_dtype=torch.float32,
        use_custom_corr=False,
        symmetric=False,
        upsample_res=(H, W),
    ).eval()

    def img(what, h, w):
        return torch.from_numpy(load(dump / f"{pre}{what}.f32", (1, 3, h, w)))

    a_c, b_c = img("in_c_0", C, C), img("in_c_1", C, C)
    a_f, b_f = img("in_f_0", H, W), img("in_f_1", H, W)
    h16 = C // 14
    seen = {}
    model.decoder.gps["16"].register_forward_hook(
        lambda mod, inp, out: seen.__setitem__("gp", out)
    )
    model.decoder.embedding_decoder.register_forward_hook(
        lambda mod, inp, out: seen.__setitem__("cls", out)
    )

    with torch.inference_mode():
        vit = model.encoder.dinov2_vitl14[0]
        for k, x in enumerate((a_c, b_c)):
            tok = vit.forward_features(x)["x_norm_patchtokens"][0]
            report(
                f"dino_{k}",
                load(dump / f"{pre}dino_{k}.f32", (1024, h16 * h16)),
                tok.T.numpy(),
            )

        corr = model({"im_A": a_c, "im_B": b_c}, batched=True, scale_factor=1)
        report(
            "gp",
            load(dump / f"{pre}gp.f32", (h16 * h16, 512)),
            seen["gp"][0].reshape(512, -1).T.numpy(),
        )
        cls, cert = seen["cls"][0], seen["cls"][1]
        report_flow(
            "cls flow",
            load(dump / f"{pre}cls_flow.f32", (2, h16, h16)),
            cls_to_flow_refine(cls)[0].permute(2, 0, 1).numpy(),
        )
        report(
            "cls certainty",
            load(dump / f"{pre}cls_cert.f32", (h16, h16)),
            cert[0, 0].numpy(),
        )
        for s in (16, 8, 4, 2, 1):
            h, w = corr[s]["flow"].shape[-2:]
            report_flow(
                f"p0_s{s} flow",
                load(dump / f"{pre}p0_s{s}_flow.f32", (2, h, w)),
                corr[s]["flow"][0].numpy(),
            )
            report(
                f"p0_s{s} cert",
                load(dump / f"{pre}p0_s{s}_cert.f32", (h, w)),
                corr[s]["certainty"][0, 0].numpy(),
            )

        # Fine pass from the module's own coarse-pass result.
        start = {
            "flow": torch.from_numpy(load(dump / f"{pre}p0_s1_flow.f32", (1, 2, C, C))),
            "certainty": torch.from_numpy(
                load(dump / f"{pre}p0_s1_cert.f32", (1, 1, C, C))
            ),
        }
        sf = math.sqrt(H * W) / C
        fine = model(
            {"im_A": a_f, "im_B": b_f, "corresps": start},
            batched=True,
            upsample=True,
            scale_factor=sf,
        )
        for s in (8, 4, 2, 1):
            h, w = fine[s]["flow"].shape[-2:]
            report_flow(
                f"p1_s{s} flow",
                load(dump / f"{pre}p1_s{s}_flow.f32", (2, h, w)),
                fine[s]["flow"][0].numpy(),
            )
            report(
                f"p1_s{s} cert",
                load(dump / f"{pre}p1_s{s}_cert.f32", (h, w)),
                fine[s]["certainty"][0, 0].numpy(),
            )

        # RoMa's match() after both passes.
        c16 = torch.from_numpy(load(dump / f"{pre}p0_s16_cert.f32", (1, 1, h16, h16)))
        low = F.interpolate(c16, size=(H, W), align_corners=False, mode="bilinear")
        cert = fine[1]["certainty"] - 0.5 * low * (low < 0)
        cert = cert.sigmoid()[0, 0]
        flow = fine[1]["flow"][0]
        cert[(flow.abs() > 1).any(0)] = 0
        flow = flow.clamp(-1, 1)
        report_flow("final flow", load(dump / f"{pre}flow.f32", (2, H, W)), flow.numpy())
        report("final cert", load(dump / f"{pre}cert.f32", (H, W)), cert.numpy())


if __name__ == "__main__":
    main()
