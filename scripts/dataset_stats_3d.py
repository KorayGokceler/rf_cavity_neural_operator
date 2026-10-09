"""Coverage report of generator H5 file(s): per family counts, size / aspect / volume ranges,
frequency range, mesh size, deformation / anchor / low-β / port / cell-count shares and the
share of geometries with a near-degenerate pair among the stored modes.

    python scripts/dataset_stats_3d.py part00.h5 [part01.h5 ...] [--csv stats.csv]
"""
import argparse
import json

import h5py
import numpy as np
import pandas as pd


def rows(paths):
    for p in paths:
        with h5py.File(p, "r") as f:
            for k in f:
                g = f[k]
                if "freqs" not in g:
                    continue
                prm = json.loads(g.attrs.get("geom_params", "{}"))
                X = g["nodes"][:]
                ext = np.sort(X.max(0) - X.min(0))
                fr = g["freqs"][:]
                yield {"shape_type": str(g.attrs["shape_type"]), "f1_GHz": fr[0], "f_last_GHz": fr[-1],
                       "size_mm": 1e3 * ext[-1], "aspect": ext[0] / ext[-1], "volume_cm3": 1e6 * g.attrs["volume"],
                       "n_edges": int(g.attrs["n_edges"]), "deformed": prm.get("deform_lip", 0.0) > 0,
                       "anchor": prm.get("anchor", 0.0) > 0, "low_beta": prm.get("beta", 1.0) < 1.0,
                       "ports": prm.get("n_ports", 0.0) > 0, "n_cells": prm.get("n_cells", np.nan),
                       "degenerate": bool(np.any(fr[1:] / fr[:-1] - 1 < 1e-3))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("h5", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    df = pd.DataFrame(rows(a.h5))
    if a.csv:
        df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby("shape_type")
    num = g[["f1_GHz", "size_mm", "aspect", "n_edges"]].agg(["min", "median", "max"]).round(3)
    share = g[["deformed", "anchor", "low_beta", "ports", "degenerate"]].mean().round(2)
    share.insert(0, "n", g.size())
    print(f"{len(df)} geometries in {len(a.h5)} file(s)\n")
    print(share.to_string(), "\n")
    print(num.to_string(), "\n")
    if df["n_cells"].notna().any():
        print("elliptical cell counts:", df["n_cells"].dropna().astype(int).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
