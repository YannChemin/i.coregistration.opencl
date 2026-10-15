# SPDX-License-Identifier: Unlicense
"""Evaluate i.coregistration.opencl on real image pairs without ground truth.

Run inside a GRASS session whose project has the images' CRS:

    grass --tmp-project a.tif --exec python3 eval_basemaps.py a.tif b.tif [c.tif ...] \
        --output <dir> [--module <path>] [--flags r] [--tag v0]

The images (RGB GeoTIFFs over the same area, e.g. three basemaps or two
dates) are linked with r.external; the region is the first image's. Each
image is matched to the next, and the last to the first (A->B, B->C,
C->A), so with three or more images the shifts should close the loop.

Metrics per pair: control points, the shift (median and median absolute
deviation), and the leave-one-out residuals of a polynomial of order 1
and 2 fitted to the shifts (median, 90th percentile; the larger the
residual, the more the points disagree with a smooth model: outliers,
moved objects, local distortion of either image). Over the loop: the
closure of the composed shift fields (each a median shift per block of
--grid x --grid cells, 256 by default, bilinearly interpolated), median
and 90th percentile, in map units. With --pairs star, the first image
is matched to each of the others instead (a scene against several
references), without closure.
Results go to <dir>/basemaps.json and basemaps.md; progress lines
"k/n pair" to stdout. Not shipped with the module.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import grass.script as gs
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MIN_BLOCK = 5  # Control points a block needs.


def run_module(module, src, ref, points, flags, options, env):
    cmd = [
        module,
        f"group={src}",
        f"reference={ref}",
        f"points={points}",
        "-n",
        "--overwrite",
        "--verbose",
    ]
    if flags:
        cmd.append(f"-{flags}")
    cmd += options.split()
    t0 = time.monotonic()
    p = subprocess.run(
        cmd, env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
    )
    wall = time.monotonic() - t0
    if p.returncode:
        sys.stderr.write(p.stderr[-3000:])
        gs.fatal(f"{module} failed on <{src}> -> <{ref}>")
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


def read_points(name):
    """Active control points (all of them before the robust stage existed),
    the number of inactive ones, and the active points' predicted standard
    deviations (map units; NAN when the matcher gives none)."""
    cols = "src_east,src_north,ref_east,ref_north"
    columns = gs.vector_columns(name)
    has_inlier = "inlier" in columns
    has_sd = "sd" in columns
    if has_inlier:
        cols += ",inlier"
    if has_sd:
        cols += ",sd"
    rows = gs.vector_db_select(name, columns=cols)["values"]
    a = np.array(
        [[float(x) if x != "" else np.nan for x in r] for r in rows.values()]
    )
    a = a.reshape(-1, 4 + has_inlier + has_sd)
    active = a[:, 4] > 0 if has_inlier else np.ones(len(a), bool)
    sd = a[active, -1] if has_sd else np.full(int(active.sum()), np.nan)
    return a[active, :4], int((~active).sum()), sd


def sd_vs_residual(sd, res):
    """How well the predicted standard deviations rank the leave-one-out
    residuals: Spearman correlation, and the residuals' 90th percentile in
    the thirds of lowest and highest predicted standard deviation."""
    ok = np.isfinite(sd) & np.isfinite(res)
    if ok.sum() < 30:
        return {}
    s, e = sd[ok], res[ok]
    rs, re = np.argsort(np.argsort(s)), np.argsort(np.argsort(e))
    lo, hi = np.percentile(s, [100 / 3, 200 / 3])
    return {
        "sd_median": round(float(np.median(s)), 3),
        "sd_spearman": round(float(np.corrcoef(rs, re)[0, 1]), 3),
        "loo2_p90_low_sd": round(float(np.percentile(e[s <= lo], 90)), 3),
        "loo2_p90_high_sd": round(float(np.percentile(e[s >= hi], 90)), 3),
    }


def poly_terms(x, y, order):
    t = [np.ones_like(x)]
    for d in range(1, order + 1):
        t += [x ** (d - i) * y**i for i in range(d + 1)]
    return np.stack(t, 1)


def loo_residuals(pts, order, region):
    """Leave-one-out residuals (map units) of a polynomial of the shift."""
    cx, cy = (region["e"] + region["w"]) / 2, (region["n"] + region["s"]) / 2
    sc = max(region["e"] - region["w"], region["n"] - region["s"]) / 2
    A = poly_terms((pts[:, 0] - cx) / sc, (pts[:, 1] - cy) / sc, order)
    if len(pts) <= 2 * A.shape[1]:
        return None
    d = pts[:, 2:4] - pts[:, 0:2]
    Q, _ = np.linalg.qr(A)
    h = np.sum(Q * Q, 1)
    res = d - Q @ (Q.T @ d)
    loo = res / (1 - h)[:, None]
    return np.hypot(loo[:, 0], loo[:, 1])


def shift_grid(pts, region, grid):
    """Median shift per grid x grid block, NaN where too few points."""
    w, n = region["w"], region["n"]
    bx, by = grid * region["ewres"], grid * region["nsres"]
    nx = int(np.ceil((region["e"] - w) / bx))
    ny = int(np.ceil((n - region["s"]) / by))
    ix = np.clip(((pts[:, 0] - w) / bx).astype(int), 0, nx - 1)
    iy = np.clip(((n - pts[:, 1]) / by).astype(int), 0, ny - 1)
    g = np.full((ny, nx, 2), np.nan)
    for j in range(ny):
        for i in range(nx):
            sel = (ix == i) & (iy == j)
            if sel.sum() >= MIN_BLOCK:
                g[j, i] = np.median(pts[sel, 2:4] - pts[sel, 0:2], 0)
    xc = w + (np.arange(nx) + 0.5) * bx
    yc = n - (np.arange(ny) + 0.5) * by
    # RegularGridInterpolator wants increasing axes: north to south is decreasing.
    return RegularGridInterpolator(
        (yc[::-1], xc), g[::-1], bounds_error=False, fill_value=None
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--module", default="i.coregistration.opencl")
    ap.add_argument("--flags", default="")
    ap.add_argument(
        "--options", default="", help="more module options, e.g. estimator=magsac"
    )
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--grid", type=int, default=256, help="cells per block of the shift fields"
    )
    ap.add_argument(
        "--pairs",
        choices=("loop", "star"),
        default="loop",
        help="loop: each image to the next and the last to the first; "
        "star: the first image to each other one",
    )
    args = ap.parse_args()
    if len(args.images) < 2:
        gs.fatal("Give at least two images")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GRASS_MESSAGE_FORMAT"] = "plain"
    names = []
    for k, path in enumerate(args.images):
        name = f"im{k}"
        gs.run_command(
            "r.external", input=path, output=name, flags="o", overwrite=True, quiet=True
        )
        gs.run_command(
            "i.group", group=name, input=f"{name}.1,{name}.2,{name}.3", quiet=True
        )
        names.append(name)
    gs.run_command("g.region", raster="im0.1")
    region = gs.region()
    m = len(names)
    if args.pairs == "star":
        pairs = [(0, k) for k in range(1, m)]
    else:
        pairs = [(k, (k + 1) % m) for k in range(m if m > 2 else 1)]
    results = {
        "tag": args.tag,
        "images": [Path(p).name for p in args.images],
        "flags": args.flags,
        "module": args.module,
        "pairs": {},
    }
    fields = []
    for j, (a, b) in enumerate(pairs):
        label = f"{Path(args.images[a]).stem} -> {Path(args.images[b]).stem}"
        print(f"{j}/{len(pairs)} {label}", flush=True)
        info = run_module(
            args.module,
            names[a],
            names[b],
            f"gcp_{a}_{b}",
            args.flags,
            args.options,
            env,
        )
        pts, inactive, sd = read_points(f"gcp_{a}_{b}")
        d = pts[:, 2:4] - pts[:, 0:2]
        med = np.median(d, 0) if len(d) else np.zeros(2)
        r = {
            "points": int(len(pts)),
            "inactive": inactive,
            "shift_east": round(float(med[0]), 3),
            "shift_north": round(float(med[1]), 3),
            "mad_east": round(float(np.median(np.abs(d[:, 0] - med[0]))), 3)
            if len(d)
            else None,
            "mad_north": round(float(np.median(np.abs(d[:, 1] - med[1]))), 3)
            if len(d)
            else None,
        }
        for order in (1, 2):
            loo = loo_residuals(pts, order, region)
            r[f"loo{order}_median"] = (
                round(float(np.median(loo)), 3) if loo is not None else None
            )
            r[f"loo{order}_p90"] = (
                round(float(np.percentile(loo, 90)), 3) if loo is not None else None
            )
        loo = loo_residuals(pts, 2, region)
        if loo is not None:
            r.update(sd_vs_residual(sd, loo))
        r.update(info)
        results["pairs"][label] = r
        fields.append(shift_grid(pts, region, args.grid))
        print(f"{label}: {json.dumps(r)}", flush=True)
    if m > 2 and args.pairs == "loop":
        # Closure at block centres: follow the fields around the loop.
        bx, by = args.grid * region["ewres"], args.grid * region["nsres"]
        xs = np.arange(region["w"] + bx / 2, region["e"], bx)
        ys = np.arange(region["n"] - by / 2, region["s"], -by)
        X, Y = np.meshgrid(xs, ys)
        p0 = np.stack([X.ravel(), Y.ravel()], 1)
        p = p0.copy()
        for f in fields:
            p = p + f(np.stack([p[:, 1], p[:, 0]], 1))
        c = np.hypot(*(p - p0).T)
        ok = np.isfinite(c)
        results["loop"] = {
            "nodes": int(len(c)),
            "evaluated": int(ok.sum()),
            "closure_median": round(float(np.median(c[ok])), 3) if ok.any() else None,
            "closure_p90": round(float(np.percentile(c[ok], 90)), 3)
            if ok.any()
            else None,
        }
    print(f"{len(pairs)}/{len(pairs)} done", flush=True)
    (out / "basemaps.json").write_text(json.dumps(results, indent=1) + "\n")
    cols = [
        "points",
        "inactive",
        "shift_east",
        "shift_north",
        "mad_east",
        "mad_north",
        "loo1_median",
        "loo1_p90",
        "loo2_median",
        "loo2_p90",
        "sd_median",
        "sd_spearman",
        "loo2_p90_low_sd",
        "loo2_p90_high_sd",
        "wall_s",
        "device_s",
        "match_s",
        "tiles",
    ]
    lines = [
        f"# Image pairs ({args.tag}, flags -{args.flags} {args.options})",
        "",
        f"Region of {results['images'][0]}: {region['rows']} x {region['cols']} cells of "
        f"{region['nsres']:g} map units. Shifts and residuals in map units.",
        "",
        "| pair | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for label, r in results["pairs"].items():
        lines.append(
            f"| {label} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        )
    if "loop" in results:
        lp = results["loop"]
        lines += [
            "",
            f"Loop closure over {lp['evaluated']} of {lp['nodes']} blocks of {args.grid} cells: median "
            f"{lp['closure_median']}, 90th percentile {lp['closure_p90']} map units.",
        ]
    (out / "basemaps.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
