# SPDX-License-Identifier: Unlicense
"""Check the generic OpenCL kernels (ops.cl) against PyTorch.

Run ``make ops_check`` and ``OBJ.<arch>/ops_check <dir> [device
[platform]]`` (on the OpenCL machine), then, wherever torch is installed:

    python3 tests/ops_compare.py <dir>

Every case of <dir>/cases.txt is recomputed with torch in float64 from
the saved inputs and compared with the device output: the largest
absolute difference, relative to the largest reference magnitude, must
stay below the case's tolerance. Prints one line per case and exits with
status 1 if any fails. Not a pytest test: it needs torch and an OpenCL
device, usually on different machines.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ACT_NONE, ACT_GELU, ACT_RELU, ACT_LEAKY, ACT_GELU_TANH = range(5)
# Single precision with sums of up to a few hundred terms.
TOL = 2e-5


def act(x, a, slope):
    if a == ACT_GELU:
        return F.gelu(x)
    if a == ACT_RELU:
        return F.relu(x)
    if a == ACT_LEAKY:
        return F.leaky_relu(x, slope)
    if a == ACT_GELU_TANH:
        return F.gelu(x, approximate="tanh")
    return x


def number(v):
    """An int, a float, or the string itself."""
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


class Case:
    def __init__(self, folder, line):
        words = line.split()
        self.name = words[0]
        self.p = {}
        for w in words[1:]:
            k, v = w.split("=", 1)
            self.p[k] = number(v)
        self.folder = folder

    def __getitem__(self, k):
        return self.p[k]

    def t(self, name, *shape):
        v = np.fromfile(self.folder / f"{self.name}_{name}.f32", dtype=np.float32)
        return (
            torch.from_numpy(v.astype(np.float64)).reshape(*shape)
            if shape
            else torch.from_numpy(v.astype(np.float64))
        )


def ref_gemm(c):
    M, N, K, nb = c["M"], c["N"], c["K"], c["nb"]
    a = c.t("a", nb, M, K)
    b = c.t("b", nb, N, K) if c["transb"] else c.t("b", nb, K, N).transpose(1, 2)
    v = c["alpha"] * a @ b.transpose(1, 2) + c.t("bias")
    v = act(v, c["act"], c["slope"])
    if c["residual"]:
        v = v + c.t("r", nb, M, N)
    return {"out": v}


def ref_conv1x1(c):
    C, HW, K = c["C"], c["HW"], c["K"]
    v = c.t("w", K, C) @ c.t("x", C, HW) + c.t("bias")[:, None]
    return {"out": act(v, c["act"], 0)}


def ref_conv2d(c):
    C, H, W, K, k, g = c["C"], c["H"], c["W"], c["K"], c["k"], c["groups"]
    x = c.t("x", 1, C, H, W)
    if c["norm"]:
        x = (x - c.t("mean")[None, :, None, None]) * c.t("rstd")[None, :, None, None]
        x = F.prelu(x, c.t("alpha"))
    y = F.conv2d(
        x,
        c.t("w", K, C // g, k, k),
        c.t("bias"),
        stride=c["s"],
        padding=c["p"],
        dilation=c["d"],
        groups=g,
    )
    return {"out": act(y, c["act"], c["slope"])}


def ref_instnorm(c):
    C, HW = c["C"], c["HW"]
    y = F.instance_norm(
        c.t("x", 1, C, HW), weight=c.t("g"), bias=c.t("bias"), eps=c["eps"]
    )
    return {"out": act(y, c["act"], c["slope"])}


def ref_layernorm(c):
    rows, C = c["rows"], c["C"]
    y = F.layer_norm(c.t("x", rows, C), (C,), c.t("g"), c.t("bias"), c["eps"])
    return {"out": act(y, c["act"], 0)}


def ref_softmax(c):
    return {"out": torch.softmax(c["scale"] * c.t("x", c["rows"], c["L"]), -1)}


def ref_attention(c):
    Nq, Nk, nh, d = c["Nq"], c["Nk"], c["nh"], c["d"]
    E = nh * d
    qkv = c.t("qkv", Nk, 3, nh, d)
    q, k, v = (qkv[:, i].transpose(0, 1) for i in range(3))
    mask = None
    if c["masked"]:
        mask = c.t("mask").bool()[None, None, :]
    o = F.scaled_dot_product_attention(
        q[:, :Nq][None], k[None], v[None], attn_mask=mask, scale=c["scale"]
    )
    return {"out": o[0].transpose(0, 1).reshape(Nq, E)}


def ref_rope(c):
    rows, nh, d, npos, ld, off = (
        c["rows"],
        c["nh"],
        c["d"],
        c["npos"],
        c["ld"],
        c["off"],
    )
    flat = c.t("x")
    x = flat[off : off + rows * ld].reshape(rows, ld)[:, : nh * d].reshape(rows, nh, d)
    ang = c.t("angles", npos, d // 2)[torch.arange(rows) % npos][:, None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    if c["interleaved"]:
        a, b = x[..., 0::2], x[..., 1::2]
        y = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).reshape(rows, nh, d)
    else:
        a, b = x[..., : d // 2], x[..., d // 2 :]
        y = torch.cat((a * cos - b * sin, a * sin + b * cos), -1)
    out = flat.clone()
    view = out[off : off + rows * ld].reshape(rows, ld)
    view[:, : nh * d] = y.reshape(rows, nh * d)
    return {"out": out}


def ref_resize(c):
    C, h, w, H, W, coff = c["C"], c["h"], c["w"], c["H"], c["W"], c["coff"]
    x = c.t("x", 1, C, h, w)
    if c["scale"] > 0:
        y = F.interpolate(
            x, scale_factor=c["scale"], mode="bilinear", align_corners=False
        )
    else:
        y = F.interpolate(
            x, size=(H, W), mode="bilinear", align_corners=bool(c["align"])
        )
    out = torch.zeros(C + coff, H, W, dtype=torch.float64)
    out[coff:] = y[0]
    return {"out": out}


def ref_grid_sample(c):
    C, H, W, n = c["C"], c["H"], c["W"], c["n"]
    y = F.grid_sample(
        c.t("x", 1, C, H, W),
        c.t("grid", 1, 1, n, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=bool(c["align"]),
    )
    return {"out": y.reshape(C, n)}


def ref_unfold(c):
    C, H, W, k, s, p = c["C"], c["H"], c["W"], c["k"], c["s"], c["p"]
    return {"out": F.unfold(c.t("x", 1, C, H, W), k, stride=s, padding=p)[0]}


def ref_pool(c):
    C, H, W, k, s, p = c["C"], c["H"], c["W"], c["k"], c["s"], c["p"]
    x = c.t("x", 1, C, H, W)
    y = (
        F.max_pool2d(x, k, s, p)
        if c["max"]
        else F.avg_pool2d(x, k, s, p, count_include_pad=True)
    )
    return {"out": y}


def ref_transpose(c):
    return {"out": c.t("x", c["B"], c["R"], c["C"]).transpose(1, 2)}


def ref_elementwise(c):
    C, HW = c["C"], c["HW"]
    x = c.t("x", C, HW)
    r = {"prelu": F.prelu(x[None], c.t("alpha"))[0]}
    for a in (ACT_GELU, ACT_RELU, ACT_LEAKY, ACT_GELU_TANH):
        r[f"act{a}"] = act(x, a, c["slope"])
    r["axpby"] = c["a"] * x + c["b"] * c.t("b")[None, :]
    return r


def ref_fold_bn(c):
    C, H, W, K = c["C"], c["H"], c["W"], c["K"]
    y = F.conv2d(c.t("x", 1, C, H, W), c.t("w", K, C, 3, 3), c.t("bias"), padding=1)
    y = F.batch_norm(
        y,
        c.t("mean"),
        c.t("var"),
        c.t("gamma"),
        c.t("beta"),
        training=False,
        eps=c["eps"],
    )
    return {"out": y}


REFS = {
    "gemm": ref_gemm,
    "conv1x1": ref_conv1x1,
    "conv2d": ref_conv2d,
    "instnorm": ref_instnorm,
    "layernorm": ref_layernorm,
    "softmax": ref_softmax,
    "attention": ref_attention,
    "rope": ref_rope,
    "resize": ref_resize,
    "grid_sample": ref_grid_sample,
    "unfold": ref_unfold,
    "pool": ref_pool,
    "transpose": ref_transpose,
    "elementwise": ref_elementwise,
    "fold_bn": ref_fold_bn,
}


def main():
    folder = Path(sys.argv[1])
    failed = 0
    for line in (folder / "cases.txt").read_text().splitlines():
        c = Case(folder, line)
        for out, ref in REFS[c["op"]](c).items():
            got = c.t(out).numpy()
            ref = ref.reshape(-1).numpy()
            if got.shape != ref.shape:
                print(f"FAIL {c.name}/{out}: {got.size} values, reference {ref.size}")
                failed += 1
                continue
            scale = max(np.abs(ref).max(), 1e-30)
            err = np.abs(got - ref).max() / scale
            ok = err <= TOL and np.isfinite(got).all()
            failed += not ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {c.name}/{out}: max |diff| / max |ref| = {err:.2e}"
            )
    print(f"{failed} failure(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
