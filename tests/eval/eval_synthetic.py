# SPDX-License-Identifier: Unlicense
"""Evaluate i.coregistration.opencl on synthetic pairs with a known warp.

Run inside a GRASS session whose project has the reference's CRS:

    grass --tmp-project ref.tif --exec python3 eval_synthetic.py ref.tif \
        --output <dir> [--module <path>] [--size 2048] [--flags s] [--tag v0]

The reference (an RGB GeoTIFF) is linked with r.external. In a centred
window of size x size cells, each scenario builds an image that shows at
cell p what the reference shows at f(p) = p + u(p) (bilinear sampling),
so the true reference position of every control point (xs, ys) is
f(xs, ys):

- translation: a sub-cell shift;
- affine: rotation, scale and shift about the window's centre;
- smooth: a smooth field (Gaussian bumps of up to 3 cells) and a shift;
- objects: a shift, plus 40 patches of the reference, 64-160 cells
  wide, pasted 12-30 cells away from where the field puts them (objects
  that moved between dates); matches whose source lies in a patch or
  within 8 cells of it are counted apart;
- radiometric: a shift with gamma, gain, offset, blur and noise.

Metrics per scenario: control points, error to the true position in
cells (median, 90th percentile), inlier ratios (error <= 1 and <= 2
cells), coverage (128 x 128 blocks holding an inlier), matches on moved
objects that pass (error > 2 cells), wall time and the module's own
stage timing. Results go to <dir>/synthetic.json and synthetic.md, the
control points with their true positions and errors (window cells) to
<dir>/points_<scenario>.csv.
Progress lines "k/n scenario" go to stdout. Not shipped with the module.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import grass.script as gs
import numpy as np
from grass.script import array as garray
from scipy import ndimage

SCENARIOS = ("translation", "affine", "smooth", "objects", "radiometric")
BLOCK = 128
# Cells around a pasted object whose control points count as on the
# object: their DISK window sees the pasted edge.
OBJECT_MARGIN = 8


def field(name, n, rng):
    """Displacement (u, v) in cells on an n x n grid, and the parameters."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2
    if name in ("translation", "objects", "radiometric"):
        t = {
            "translation": (3.3, -2.7),
            "objects": (2.0, 1.5),
            "radiometric": (-1.6, 2.4),
        }[name]
        return np.full((n, n), t[0]), np.full((n, n), t[1]), {"shift": t}
    if name == "affine":
        a, s, t = np.radians(0.3), 1.002, (1.5, 1.0)
        xx, yy = x - c, y - c
        u = s * (np.cos(a) * xx - np.sin(a) * yy) - xx + t[0]
        v = s * (np.sin(a) * xx + np.cos(a) * yy) - yy + t[1]
        return u, v, {"rotation_deg": 0.3, "scale": s, "shift": t}
    if name == "smooth":
        u = np.full((n, n), 1.2)
        v = np.full((n, n), -0.8)
        bumps = []
        for _ in range(6):
            cx, cy = rng.uniform(0, n, 2)
            sig = rng.uniform(n / 10, n / 5)
            au, av = rng.uniform(-3, 3, 2)
            g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig**2))
            u += au * g
            v += av * g
            bumps.append([round(float(b), 2) for b in (cx, cy, sig, au, av)])
        return u, v, {"shift": (1.2, -0.8), "bumps": bumps}
    raise ValueError(name)


def warp(ref, r0, c0, u, v):
    """The image showing at window cell p what ref shows at p + (u, v)."""
    n = u.shape[0]
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    coords = [r0 + y + v, c0 + x + u]
    return np.stack(
        [ndimage.map_coordinates(b, coords, order=1, mode="nearest") for b in ref]
    )


def paste_objects(img, ref, r0, c0, u, v, rng, count=40, sides=(64, 160)):
    """Paste reference patches 12-30 cells off the field (objects the size
    of aircraft at 0.5 m); returns the mask."""
    n = u.shape[0]
    mask = np.zeros((n, n), bool)
    for _ in range(count):
        side = int(rng.integers(sides[0], sides[1] + 1))
        y, x = rng.integers(side, n - 2 * side, 2)
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(12, 30)
        dy, dx = int(round(dist * np.sin(ang))), int(round(dist * np.cos(ang)))
        # The patch shows the reference around f(y, x) but lands at (y + dy, x + dx).
        sy = int(round(r0 + y + v[y, x]))
        sx = int(round(c0 + x + u[y, x]))
        img[:, y + dy : y + dy + side, x + dx : x + dx + side] = ref[
            :, sy : sy + side, sx : sx + side
        ]
        mask[y + dy : y + dy + side, x + dx : x + dx + side] = True
    return mask


def radiometry(img, rng):
    out = np.empty_like(img)
    for b in range(img.shape[0]):
        x = ndimage.gaussian_filter(img[b], 0.8)
        x = 255 * 1.15 * (np.clip(x, 0, 255) / 255) ** 0.75 - 12
        out[b] = x + rng.normal(0, 6, x.shape)
    return out


def run_module(module, group, points, flags, env):
    cmd = [
        module,
        f"group={group}",
        "reference=ref",
        f"points={points}",
        "-n",
        "--overwrite",
        "--verbose",
    ]
    if flags:
        cmd.append(f"-{flags}")
    t0 = time.monotonic()
    p = subprocess.run(
        cmd, env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
    )
    wall = time.monotonic() - t0
    if p.returncode:
        sys.stderr.write(p.stderr[-3000:])
        gs.fatal(f"{module} failed on <{group}>")
    info = {"wall_s": round(wall, 2)}
    m = re.search(
        r"Timing: device stage ([\d.]+) s, matching and filters ([\d.]+) s, for (\d+) tile",
        p.stderr,
    )
    if m:
        info.update(device_s=float(m[1]), match_s=float(m[2]), tiles=int(m[3]))
    m = re.search(r"Matching (\d+) tile", p.stderr)
    if m:
        info["tiles"] = int(m[1])
    return info


def read_points(name, region):
    """Source and reference positions in window cells (centres at integers)."""
    rows = gs.vector_db_select(
        name, columns="src_east,src_north,ref_east,ref_north,matches"
    )["values"]
    a = np.array([[float(x) for x in r] for r in rows.values()]).reshape(-1, 5)
    w, nn, ew, ns = region["w"], region["n"], region["ewres"], region["nsres"]
    xs, ys = (a[:, 0] - w) / ew - 0.5, (nn - a[:, 1]) / ns - 0.5
    xr, yr = (a[:, 2] - w) / ew - 0.5, (nn - a[:, 3]) / ns - 0.5
    return xs, ys, xr, yr, a[:, 4]


def score(xs, ys, xr, yr, u, v, objmask, n, merged, csv=None):
    tu = ndimage.map_coordinates(u, [ys, xs], order=1, mode="nearest")
    tv = ndimage.map_coordinates(v, [ys, xs], order=1, mode="nearest")
    err = np.hypot(xr - (xs + tu), yr - (ys + tv))
    on_obj = np.zeros(len(xs), bool)
    if objmask is not None:
        objmask = ndimage.binary_dilation(objmask, iterations=OBJECT_MARGIN)
        iy = np.clip(np.round(ys).astype(int), 0, n - 1)
        ix = np.clip(np.round(xs).astype(int), 0, n - 1)
        on_obj = objmask[iy, ix]
    if csv:
        np.savetxt(
            csv,
            np.column_stack([xs, ys, xr, yr, xs + tu, ys + tv, err, on_obj, merged]),
            fmt="%.4f",
            delimiter=",",
            header="xs,ys,xr,yr,x_true,y_true,err,on_object,matches",
            comments="",
        )
    ground = err[~on_obj]
    inl = ~on_obj & (err <= 1)
    blocks = {(int(y) // BLOCK, int(x) // BLOCK) for y, x in zip(ys[inl], xs[inl])}
    nblocks = ((n + BLOCK - 1) // BLOCK) ** 2
    q = (
        (lambda p: round(float(np.percentile(ground, p)), 3))
        if len(ground)
        else (lambda p: None)
    )
    return {
        "points": int(len(xs)),
        "median_err": q(50),
        "p90_err": q(90),
        "inliers_1": round(float(np.mean(ground <= 1)), 4) if len(ground) else None,
        "inliers_2": round(float(np.mean(ground <= 2)), 4) if len(ground) else None,
        "coverage": round(len(blocks) / nblocks, 4),
        "object_points": int(on_obj.sum()),
        "object_leaks": int((on_obj & (err > 2)).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("--output", required=True)
    ap.add_argument("--module", default="i.coregistration.opencl")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--flags", default="s")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GRASS_MESSAGE_FORMAT"] = "plain"
    gs.run_command(
        "r.external",
        input=args.reference,
        output="ref",
        flags="o",
        overwrite=True,
        quiet=True,
    )
    gs.run_command("i.group", group="ref", input="ref.1,ref.2,ref.3", quiet=True)
    gs.run_command("g.region", raster="ref.1")
    full = gs.region()
    ref = np.stack(
        [np.asarray(garray.array(f"ref.{b}", dtype=np.float32)) for b in (1, 2, 3)]
    )
    n = args.size
    r0, c0 = (full["rows"] - n) // 2, (full["cols"] - n) // 2
    if min(r0, c0) < 64:
        gs.fatal(
            f"Reference of {full['rows']} x {full['cols']} cells too small for a {n} window and its margin"
        )
    gs.run_command(
        "g.region",
        n=full["n"] - r0 * full["nsres"],
        s=full["n"] - (r0 + n) * full["nsres"],
        w=full["w"] + c0 * full["ewres"],
        e=full["w"] + (c0 + n) * full["ewres"],
    )
    region = gs.region()
    results = {
        "tag": args.tag,
        "reference": Path(args.reference).name,
        "size": n,
        "flags": args.flags,
        "module": args.module,
        "scenarios": {},
    }
    names = args.scenarios.split(",")
    for k, name in enumerate(names):
        print(f"{k}/{len(names)} {name}", flush=True)
        # Seeded by name: a scenario is the same when run alone.
        rng = np.random.default_rng(args.seed + zlib.crc32(name.encode()))
        u, v, params = field(name, n, rng)
        img = warp(ref, r0, c0, u, v)
        objmask = (
            paste_objects(img, ref, r0, c0, u, v, rng) if name == "objects" else None
        )
        if name == "radiometric":
            img = radiometry(img, rng)
        img = np.clip(np.round(img), 0, 255).astype(np.int32)
        for b in range(3):
            a = garray.array(dtype=np.int32)
            a[...] = img[b]
            a.write(f"img_{name}.{b + 1}", overwrite=True)
        gs.run_command(
            "i.group",
            group=f"img_{name}",
            input=",".join(f"img_{name}.{b}" for b in (1, 2, 3)),
            quiet=True,
        )
        info = run_module(args.module, f"img_{name}", f"gcp_{name}", args.flags, env)
        xs, ys, xr, yr, merged = read_points(f"gcp_{name}", region)
        res = score(
            xs, ys, xr, yr, u, v, objmask, n, merged, out / f"points_{name}.csv"
        )
        res.update(info)
        res["params"] = params
        results["scenarios"][name] = res
        print(
            f"{name}: {json.dumps({k2: v2 for k2, v2 in res.items() if k2 != 'params'})}",
            flush=True,
        )
    print(f"{len(names)}/{len(names)} done", flush=True)
    (out / "synthetic.json").write_text(json.dumps(results, indent=1) + "\n")
    cols = [
        "points",
        "median_err",
        "p90_err",
        "inliers_1",
        "inliers_2",
        "coverage",
        "object_leaks",
        "wall_s",
        "device_s",
        "match_s",
        "tiles",
    ]
    lines = [
        f"# Synthetic pairs ({args.tag}, {n} x {n} cells of {results['reference']}, flags -{args.flags})",
        "",
        "Errors in cells against the known field; inlier ratios and errors exclude points on moved objects.",
        "",
        "| scenario | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for name, r in results["scenarios"].items():
        lines.append(
            f"| {name} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        )
    (out / "synthetic.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
