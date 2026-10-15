# SPDX-License-Identifier: Unlicense
"""Compare matcher=roma_v2ref's RoMa v2 stages with the authors' code.

Run the module with matcher=roma_v2ref and I_COREGISTRATION_OPENCL_DUMP=<dir>,
then, in a Python environment with torch, torchvision and einops (not
run with -I when einops is in the user site):

    python reference_compare_roma2.py <dir> <romav2_refiners.pt> <RoMa v2 source> [tile]

The RoMa v2 source is the src folder of a clone of
https://github.com/Parskatt/RoMaV2 (romav2: its refiners and VGG19-BN
only; nothing of DINOv3 is loaded, and romav2_refiners.pt holds none).
RoMa's part (down to 1/8) is the same code as matcher=roma, compared by
reference_compare_roma.py. Here, on the module's own inputs:

- p0_v2s4 .. p0_v2s1: RoMa v2's refiners at 560 x 560, from the module's
  RoMa warp and certainty after 1/8;
- p1_v2s4 .. p1_v2s1: the same at the tile's size, from the module's
  p0_v2s1 (precision cleared first, as RoMa v2's second pass);
- final: warp and overlap probability.

Warp differences in cells of their grid. CPU, float32 (RoMa v2's bfloat16
autocast in the refiners is turned off).

Not a pytest test: it needs torch, the weights and RoMa v2's source.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def main():
    dump, weights, src = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    t = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    sys.path.insert(0, src)
    torch.set_float32_matmul_precision("highest")
    from romav2.features import VGG19BN
    from romav2.refiner import Block, Refiners

    sd = torch.load(weights, map_location="cpu", weights_only=True)
    refiners = Refiners(Refiners.Cfg())
    vgg = VGG19BN(4, False)
    refiners.load_state_dict(
        {k[len("refiners.") :]: v for k, v in sd.items() if k.startswith("refiners.")}
    )
    vgg.load_state_dict(
        {
            k[len("refiner_features.") :]: v
            for k, v in sd.items()
            if k.startswith("refiner_features.")
        }
    )
    for m in refiners.modules():
        if isinstance(m, Block):
            m.enable_amp = False
    refiners.eval()
    vgg.eval()

    head = (dump / f"tile{t:03d}_roma.txt").read_text().split()
    info = dict(zip(head[0::2], map(int, head[1::2])))
    H, W, C = info["H"], info["W"], info["coarse"]
    pre = f"tile{t:03d}_roma_"

    def feats(img):
        """VGG19-BN features before each pooling, on an image already
        normalized by the module (RoMa v2's forward normalizes itself)."""
        out, x, scale = {}, img, 1
        for layer in vgg.layers:
            if isinstance(layer, nn.MaxPool2d):
                out[scale] = x.permute(0, 2, 3, 1)
                scale *= 2
            x = layer(x)
        return out

    def img(what, h, w):
        return torch.from_numpy(load(dump / f"{pre}{what}.f32", (1, 3, h, w)))

    def bhwc_resize(x, h, w):
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    def chain(p, a, b, Hp, Wp, warp, conf, zero_prec):
        fa, fb = feats(a), feats(b)
        sf = torch.tensor((Wp / 512, Hp / 512))
        for ps, ref in refiners.items():
            s = int(ps)
            if zero_prec and s == 4:
                conf = conf.clone()
                conf[..., 1:] = 0
            h, w = Hp // s, Wp // s
            warp, conf = bhwc_resize(warp, h, w), bhwc_resize(conf, h, w)
            out = ref(
                f_A=fa[s], f_B=fb[s], prev_warp=warp, prev_confidence=conf,
                scale_factor=sf,
            )
            warp, conf = out["warp"], out["confidence"]
            report_flow(
                f"p{p}_v2s{s} flow",
                load(dump / f"{pre}p{p}_v2s{s}_flow.f32", (2, h, w)),
                warp[0].permute(2, 0, 1).numpy(),
            )
            report(
                f"p{p}_v2s{s} conf",
                load(dump / f"{pre}p{p}_v2s{s}_conf.f32", (4, h, w)),
                conf[0].permute(2, 0, 1).numpy(),
            )
        return warp, conf

    with torch.inference_mode():
        h8 = C // 8
        warp = torch.from_numpy(load(dump / f"{pre}p0_s8_flow.f32", (1, 2, h8, h8)))
        cert = torch.from_numpy(load(dump / f"{pre}p0_s8_cert.f32", (1, 1, h8, h8)))
        conf = torch.cat((cert, torch.zeros(1, 3, h8, h8)), 1)
        chain(
            0, img("in_c_0", C, C), img("in_c_1", C, C), C, C,
            warp.permute(0, 2, 3, 1), conf.permute(0, 2, 3, 1), False,
        )
        # Second pass from the module's own first-pass result.
        warp = torch.from_numpy(load(dump / f"{pre}p0_v2s1_flow.f32", (1, 2, C, C)))
        conf = torch.from_numpy(load(dump / f"{pre}p0_v2s1_conf.f32", (1, 4, C, C)))
        warp, conf = chain(
            1, img("in_f_0", H, W), img("in_f_1", H, W), H, W,
            warp.permute(0, 2, 3, 1), conf.permute(0, 2, 3, 1), True,
        )
        flow = warp[0].permute(2, 0, 1)
        over = conf[0, ..., 0].sigmoid()
        over[(flow.abs() > 1).any(0)] = 0
        flow = flow.clamp(-1, 1)
        report_flow("final flow", load(dump / f"{pre}flow.f32", (2, H, W)), flow.numpy())
        report("final cert", load(dump / f"{pre}cert.f32", (H, W)), over.numpy())


if __name__ == "__main__":
    main()
