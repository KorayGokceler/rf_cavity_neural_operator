"""Coverage report of generator (field-label) H5 file(s): per family counts, size / volume ranges,
frequency range, model DOF, deformation / anchor / low-β / port / cell-count shares and the
share of geometries with a near-degenerate pair among the stored modes.  Size is the equivalent
length V^(1/3); --extent rebuilds each model mesh (~1 s per sample) for the bounding-box size and aspect.

    python scripts/dataset_stats_3d.py part00.h5 [part01.h5 ...] [--csv stats.csv] [--extent]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def bbox_extent(g):
    """Sorted bounding-box side lengths [m] of the sample's model mesh."""
    from src.data_gen.field_labels import read_field_group
    from src.service.geometry import linear_mesh
    X = linear_mesh(read_field_group(g)[0])[0]
    return np.sort(X.max(0) - X.min(0))


def rows(paths, extent=False):
    for p in paths:
        with h5py.File(p, "r") as f:
            for k in f:
                g = f[k]
                if "freqs" not in g:
                    continue
                prm = json.loads(g.attrs.get("geom_params", "{}"))
                vol = float(g.attrs["volume"])
                ext = bbox_extent(g) if extent else np.array([np.nan, np.nan, vol ** (1 / 3)])
                fr = g["freqs"][:]
                yield {"shape_type": str(g.attrs["shape_type"]), "f1_GHz": fr[0], "f_last_GHz": fr[-1],
                       "size_mm": 1e3 * ext[-1], "aspect": ext[0] / ext[-1], "volume_cm3": 1e6 * vol,
                       "n_dof": int(g.attrs["n_dof_model"]), "deformed": prm.get("deform_lip", 0.0) > 0,
                       "anchor": prm.get("anchor", 0.0) > 0, "low_beta": prm.get("beta", 1.0) < 1.0,
                       "ports": prm.get("n_ports", 0.0) > 0, "n_cells": prm.get("n_cells", np.nan),
                       "degenerate": bool(np.any(fr[1:] / fr[:-1] - 1 < 1e-3))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("h5", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--extent", action="store_true", help="bounding-box size and aspect (rebuilds the meshes)")
    a = ap.parse_args()
    if a.extent:
        import torch  # noqa: F401 — before NGSolve (tests/conftest.py)
    df = pd.DataFrame(rows(a.h5, a.extent))
    if a.csv:
        df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby("shape_type")
    num = g[["f1_GHz", "size_mm", *(["aspect"] if a.extent else []), "n_dof"]].agg(["min", "median", "max"]).round(3)
    share = g[["deformed", "anchor", "low_beta", "ports", "degenerate"]].mean().round(2)
    share.insert(0, "n", g.size())
    print(f"{len(df)} geometries in {len(a.h5)} file(s)\n")
    print(share.to_string(), "\n")
    print(num.to_string(), "\n")
    if df["n_cells"].notna().any():
        print("elliptical cell counts:", df["n_cells"].dropna().astype(int).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
