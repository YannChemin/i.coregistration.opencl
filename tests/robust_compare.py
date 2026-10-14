# SPDX-License-Identifier: Unlicense
"""Compare the robust estimation stage with OpenCV's USAC (MAGSAC++).

    make MODULE_TOPDIR=... robust_check
    python3 tests/robust_compare.py

Synthetic matches (a known homography or affinity, Gaussian noise of 0.3,
uniform outliers and clusters of matches moved together by 15-30 units)
go both to ``robust_check --fit`` (this module's estimators) and to
OpenCV's cv2.findHomography / cv2.estimateAffine2D with USAC_MAGSAC and
the same threshold. Printed per set: the largest error of each model
against the truth over the domain, the inlier recall and precision of
each, and the agreement of the two inlier sets (Jaccard index). Not a
pytest test: it needs OpenCV; exits 1 when this module's model is off by
more than 1 unit or worse than OpenCV's by more than 0.5.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIDE = 1000.0
NOISE = 0.3
THRESHOLD = 2.7  # 9 sigma of the noise, as in robust_check.


def truth(kind, x, y):
    if kind == "homography":
        d = 1 + 2e-5 * x - 1e-5 * y
        return (1.001 * x + 0.002 * y + 2) / d, (-0.001 * x + 0.999 * y - 3) / d
    return 1.002 * x + 0.003 * y + 1.5, -0.002 * x + 0.998 * y + 4.0


def make_set(rng, kind, n, outliers, clusters):
    x, y = rng.uniform(0, SIDE, n), rng.uniform(0, SIDE, n)
    X, Y = truth(kind, x, y)
    X, Y = X + rng.normal(0, NOISE, n), Y + rng.normal(0, NOISE, n)
    good = np.ones(n, bool)
    cost = rng.uniform(0, 0.6, n)
    out = rng.uniform(0, 1, n) < outliers
    X[out], Y[out] = rng.uniform(0, SIDE, out.sum()), rng.uniform(0, SIDE, out.sum())
    good[out] = False
    cost[out] = rng.uniform(0.3, 1, out.sum())
    per = n // 20
    for c in range(clusters):
        sel = slice(c * per, (c + 1) * per)
        cx, cy = rng.uniform(150, 850, 2)
        x[sel] = cx + rng.uniform(-30, 30, per)
        y[sel] = cy + rng.uniform(-30, 30, per)
        tx, ty = truth(kind, x[sel], y[sel])
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(15, 30)
        X[sel], Y[sel] = tx + d * np.cos(a), ty + d * np.sin(a)
        good[sel] = False
        cost[sel] = rng.uniform(0, 0.2, per)
    return np.column_stack([x, y, X, Y]), cost, good


def grid():
    g = np.linspace(0, SIDE, 11)
    return np.array([(x, y) for y in g for x in g])


def error_of_map(kind, mapped):
    tx, ty = truth(kind, mapped[:, 0], mapped[:, 1])
    return np.hypot(mapped[:, 2] - tx, mapped[:, 3] - ty).max()


def ours(binary, pts, cost, estimator, kind):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        np.savetxt(f, np.column_stack([pts, cost]), fmt="%.6f")
        path = f.name
    p = subprocess.run(
        [binary, "--fit", path, estimator, kind, str(THRESHOLD), str(SIDE)],
        capture_output=True,
        text=True,
        check=True,
    )
    Path(path).unlink()
    inl = np.array(
        [
            int(w[1])
            for w in (line.split() for line in p.stdout.splitlines())
            if w[0] == "inlier"
        ],
        bool,
    )
    m = np.array(
        [
            [float(v) for v in line.split()[1:]]
            for line in p.stdout.splitlines()
            if line.startswith("map")
        ]
    )
    return inl, m


def opencv(pts, kind):
    src, dst = pts[:, :2].astype(np.float64), pts[:, 2:].astype(np.float64)
    if kind == "homography":
        M, mask = cv2.findHomography(
            src, dst, cv2.USAC_MAGSAC, THRESHOLD, maxIters=5000, confidence=0.99
        )
        g = grid()
        h = np.column_stack([g, np.ones(len(g))]) @ M.T
        mapped = np.column_stack([g, h[:, :2] / h[:, 2:]])
    else:
        M, mask = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=THRESHOLD,
            maxIters=5000,
            confidence=0.99,
        )
        g = grid()
        mapped = np.column_stack([g, np.column_stack([g, np.ones(len(g))]) @ M.T])
    return mask.ravel().astype(bool), mapped


def rates(inl, good):
    tp = (inl & good).sum()
    return tp / good.sum(), tp / max(inl.sum(), 1)


def main():
    found = sorted(Path(__file__).parent.parent.glob("OBJ.*/robust_check"))
    if not found:
        sys.exit("build tests/robust_check.c first: make robust_check")
    rng = np.random.default_rng(7)
    failed = 0
    sets = [
        ("affine", 0.3, 0),
        ("affine", 0.6, 3),
        ("affine", 0.8, 0),
        ("homography", 0.3, 0),
        ("homography", 0.5, 3),
        ("homography", 0.7, 2),
    ]
    for kind, outl, clusters in sets:
        pts, cost, good = make_set(rng, kind, 1500, outl, clusters)
        cv_in, cv_map = opencv(pts, kind)
        cv_err = error_of_map(kind, cv_map)
        cv_r, cv_p = rates(cv_in, good)
        print(
            f"{kind:10s} outliers {outl:.0%} clusters {clusters}: OpenCV USAC_MAGSAC error {cv_err:.3f}, "
            f"recall {cv_r:.3f}, precision {cv_p:.3f}"
        )
        for est in ("ransac", "magsac", "gcransac", "superansac"):
            inl, m = ours(str(found[0]), pts, cost, est, kind)
            err = error_of_map(kind, m) if len(m) else np.inf
            r, p = rates(inl, good)
            jac = (inl & cv_in).sum() / max((inl | cv_in).sum(), 1)
            ok = err < 1 and err < cv_err + 0.5
            failed += not ok
            print(
                f"  {'ok  ' if ok else 'FAIL'} {est:10s} error {err:.3f}, recall {r:.3f}, precision {p:.3f}, "
                f"agreement with OpenCV {jac:.3f}"
            )
    print(f"{failed} failure(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
