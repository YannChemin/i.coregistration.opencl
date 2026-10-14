# SPDX-License-Identifier: Unlicense
"""Prepare a cross-modal evaluation pair: a hyperspectral cube against RGB
basemaps, on the hyperspectral grid.

    python3 prepare_hyperspectral.py cube.zarr basemap.tif [basemap.tif ...] \
        --output <dir> [--rgb 640,550,460] [--percentiles 2,98]

The cube is a Zarr group holding a (bands, rows, cols) reflectance array
with a "spatial:transform" and "proj:code" (as written by hyp4u), and the
band centres in "band" (nm). Over the first basemap's extent, snapped to
the cube's grid:

- <dir>/hs_rgb.tif: the bands nearest to the --rgb wavelengths, each
  stretched linearly between its --percentiles to 1..255 (0 = no data);
- <dir>/<basemap>_hsgrid.tif: each basemap averaged over the cells of the
  cube's grid (gdalwarp -r average: no aliasing from the 0.5 m cells).

The pair then goes to eval_basemaps.py. Not shipped with the module.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from osgeo import gdal, osr

gdal.UseExceptions()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cube")
    ap.add_argument("basemaps", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rgb", default="640,550,460")
    ap.add_argument("--percentiles", default="2,98")
    ap.add_argument("--array", default=None, help="array name (default: the 3D one)")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lo_p, hi_p = (float(v) for v in args.percentiles.split(","))

    g = zarr.open_group(args.cube, mode="r")
    name = args.array or next(k for k, a in g.arrays() if a.ndim == 3)
    cube = g[name]
    wl = g["band"][:].astype(float)
    a, b, x0, d, e, y0 = cube.attrs["spatial:transform"]
    if b or d:
        raise SystemExit("Rotated cube grids are not handled")
    srs = osr.SpatialReference()
    srs.SetFromUserInput(cube.attrs["proj:code"])

    # Extent of the first basemap, snapped outwards to the cube's grid.
    bm = gdal.Open(args.basemaps[0])
    bx0, bdx, _, by0, _, bdy = bm.GetGeoTransform()
    bx1, by1 = bx0 + bdx * bm.RasterXSize, by0 + bdy * bm.RasterYSize
    c0 = max(int(np.floor((bx0 - x0) / a)), 0)
    c1 = min(int(np.ceil((bx1 - x0) / a)), cube.shape[2])
    r0 = max(int(np.floor((by0 - y0) / e)), 0)
    r1 = min(int(np.ceil((by1 - y0) / e)), cube.shape[1])
    gx0, gy0 = x0 + c0 * a, y0 + r0 * e
    rows, cols = r1 - r0, c1 - c0
    gt = (gx0, a, 0.0, gy0, 0.0, e)

    bands = [int(np.argmin(np.abs(wl - float(w)))) for w in args.rgb.split(",")]
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(out / "hs_rgb.tif"), cols, rows, 3, gdal.GDT_Byte, ["COMPRESS=DEFLATE"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(srs.ExportToWkt())
    for k, bi in enumerate(bands):
        v = np.asarray(cube[bi, r0:r1, c0:c1], dtype=np.float64)
        ok = np.isfinite(v)
        lo, hi = np.percentile(v[ok], [lo_p, hi_p])
        s = np.zeros(v.shape, np.uint8)
        s[ok] = np.clip(np.round(1 + 254 * (v[ok] - lo) / (hi - lo)), 1, 255)
        band = ds.GetRasterBand(k + 1)
        band.WriteArray(s)
        band.SetNoDataValue(0)
        band.SetDescription(f"{wl[bi]:.1f} nm")
        print(f"band {k + 1}: {wl[bi]:.1f} nm, stretch {lo:.4f}..{hi:.4f}")
    ds = None
    print(f"hs_rgb.tif: {cols} x {rows} cells of {a:g} m, origin {gx0:.3f}, {gy0:.3f}")

    for path in args.basemaps:
        dst = out / f"{Path(path).stem}_hsgrid.tif"
        gdal.Warp(
            str(dst),
            path,
            outputBounds=(gx0, gy0 + e * rows, gx0 + a * cols, gy0),
            width=cols,
            height=rows,
            resampleAlg="average",
            dstSRS=srs.ExportToWkt(),
            creationOptions=["COMPRESS=DEFLATE"],
        )
        print(f"{dst.name}: averaged onto the same grid")


if __name__ == "__main__":
    main()
