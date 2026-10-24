"""Assumption (A) rank check (referee item M1, docs/35 check 1).

The upper-bound proposition needs  ker K = range G  on the free DOFs of V_h = HCurl(p) with PEC wall:
the curl-curl matrix K annihilates the discrete gradients G (K G = 0), and (A) says there is nothing else
in its kernel, i.e. dim ker K = rank G = (#interior H1(p+1) DOFs) + (#extra wall potentials).  "K G = 0"
alone only gives range G inside ker K; this script reports the rank statement on meshes small enough for
dense linear algebra, for p = 1, 2, 3:

    n, dim ker K (eigenvalues of K below --rel_tol * max, dense generalised eigh with M), rank G (SVD),
    rank Kp = G^T M G, #interior H1(p+1) DOFs, #extra wall shells, the spectral gap around the cut, the
    M-distance of ker K from range G, the smallest nonzero generalised eigenvalue vs the shift-invert
    label (highorder.solve_modes, the generator's solver) and vs the analytic pillbox TM010.

Meshes: pillbox (the 179-tet curved cylinder of tests/test_field_model.py), a spherical shell (two wall
shells: one extra potential), an annular cylinder (solid torus, b1 = 1), optionally a coarse pillbox_pipes
mesh rebuilt from the BREP of a stored sample (--pipes_h5).

    python scripts/paper/check_assumption_a.py [--orders 1 2 3] [--geoms pillbox shell annulus pipes]
        [--pipes_h5 lf_test.h5 --pipes_key sample_000024 --pipes_cs 0.4 --pipes_maxh 0.2] [--json out.json]
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset_converter_3d import normalise  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402

J01 = 2.404825557695773          # first zero of J0: TM010 of the pillbox, k = J01 / R


def build_mesh(name, args):
    """(mesh, analytic k^2 of the lowest mode or None) of a test geometry."""
    from netgen.occ import Axes, Cylinder, OCCGeometry, Sphere, Z
    from ngsolve import Mesh
    if name == "pillbox":
        geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
        geo.faces.name = "wall"
        mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.05))
        k2 = (J01 / 0.1) ** 2
    elif name == "shell":
        geo = Sphere((0, 0, 0), 0.1) - Sphere((0, 0, 0), 0.04)
        geo.faces.name = "wall"
        mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.06))
        k2 = None
    elif name == "annulus":
        geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1) - Cylinder(Axes((0, 0, -0.01), Z), r=0.04, h=0.12)
        geo.faces.name = "wall"
        mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.05))
        k2 = None
    elif name == "pipes":
        import h5py
        with h5py.File(args.pipes_h5, "r") as f, tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.brep")
            with open(path, "wb") as fh:
                fh.write(np.asarray(f[args.pipes_key]["brep"]).tobytes())
            mesh = ho.netgen_mesh(path, args.pipes_maxh, curve=int(args.curve), retries=0, fix_passes=0,
                                  min_jacobian_ratio=0.0, curvaturesafety=args.pipes_cs)
        return mesh, None
    else:
        raise ValueError(name)
    mesh.Curve(int(args.curve))
    return mesh, k2


def rank_report(mesh, p, k2_analytic, rel_tol, max_dense):
    Xv = np.array([tuple(q.p) for q in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)
    ops = ho.field_operators(mesh, order=p, scale=scale, center=center)
    free = ~np.asarray(ops["bnd"], bool)
    n = int(free.sum())
    out = {"p": p, "n_tets": int(mesh.ne), "n_dof": int(ops["n_dof"]), "n_free": n}
    if n > max_dense:
        out["skipped"] = f"n_free {n} > --max_dense {max_dense}"
        return out
    K = ops["K"][free][:, free].toarray()
    M = ops["M"][free][:, free].toarray()
    G = ops["G"][free].toarray()
    out["G_wall_rows_max"] = float(abs(ops["G"][~free]).max()) if (~free).any() else 0.0
    out["KG_over_K"] = float(np.abs(K @ G).max() / np.abs(K).max())
    n_shell_extra = len(ho.wall_shells(mesh)) - 1
    out["n_pot"] = int(G.shape[1])
    out["n_extra_wall_shells"] = int(n_shell_extra)
    out["n_interior_H1"] = int(G.shape[1] - n_shell_extra)
    sv = np.linalg.svd(G, compute_uv=False)
    out["rank_G"] = int((sv > 1e-10 * sv[0]).sum())
    out["sv_G_min_over_max"] = float(sv[-1] / sv[0])
    ev_kp = np.linalg.eigvalsh(ops["Kp"].toarray())
    out["rank_Kp"] = int((ev_kp > 1e-10 * ev_kp[-1]).sum())
    t0 = time.perf_counter()
    w, X = sla.eigh(K, M, driver="gvd")                       # X is M-orthonormal
    out["t_dense_eigh_s"] = time.perf_counter() - t0
    cut = rel_tol * w[-1]
    nz = int((w < cut).sum())
    out["dim_ker_K"] = nz
    out["zero_cluster_max_rel"] = float(w[:nz].max() / w[-1]) if nz else 0.0
    out["smallest_nonzero_rel"] = float(w[nz] / w[-1])
    out["negative_min_rel"] = float(w[0] / w[-1])
    for tol in (1e-12, 1e-8):                                 # the count does not depend on the cut
        out[f"dim_ker_K_at_{tol:g}"] = int((w < tol * w[-1]).sum())
    # ker K inside range G: M-orthogonal projection of the kernel vectors onto range G leaves nothing
    N = X[:, :nz]
    Kp = ops["Kp"].toarray()
    GtMN = G.T @ (M @ N)
    R = N - G @ np.linalg.solve(Kp, GtMN)
    out["ker_minus_rangeG_M_norm_max"] = float(np.sqrt(np.einsum("ik,ik->k", R, M @ R)).max())
    lam_min = float(w[nz])
    out["lam_min_nonzero_xi"] = lam_min
    ref = ho.solve_modes(mesh, 3, order=p)["lam"] * scale ** 2     # generator solver (shift-invert + projection)
    out["lam_label_xi_first3"] = [float(v) for v in ref]
    out["lam_min_nonzero_vs_label"] = float(lam_min / ref[0] - 1.0)
    out["lowest3_dense_vs_label_max_rel"] = float(np.abs(w[nz:nz + 3] / ref - 1.0).max())
    if k2_analytic is not None:
        out["lam_min_nonzero_vs_analytic"] = float(lam_min / (k2_analytic * scale ** 2) - 1.0)
    out["rank_condition_A"] = bool(nz == out["rank_G"] == out["n_pot"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--geoms", nargs="+", default=["pillbox", "shell", "annulus"],
                    choices=["pillbox", "shell", "annulus", "pipes"])
    ap.add_argument("--curve", type=int, default=2)
    ap.add_argument("--rel_tol", type=float, default=1e-10, help="kernel = eigenvalues below rel_tol * max")
    ap.add_argument("--max_dense", type=int, default=9000, help="skip meshes with more free DOFs")
    ap.add_argument("--pipes_h5", default=None)
    ap.add_argument("--pipes_key", default="sample_000024")
    ap.add_argument("--pipes_cs", type=float, default=0.4)
    ap.add_argument("--pipes_maxh", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import ngsolve
    ngsolve.SetNumThreads(args.threads)
    rows = []
    for name in args.geoms:
        mesh, k2 = build_mesh(name, args)
        for p in args.orders:
            r = rank_report(mesh, p, k2, args.rel_tol, args.max_dense)
            r["geom"] = name
            rows.append(r)
            if "skipped" in r:
                print(f"{name:<8} p{p} tets {r['n_tets']} n_free {r['n_free']}: skipped ({r['skipped']})", flush=True)
                continue
            print(f"{name:<8} p{p} tets {r['n_tets']:>5} n_free {r['n_free']:>5} | dim ker K {r['dim_ker_K']:>5} "
                  f"rank G {r['rank_G']:>5} (cols {r['n_pot']} = {r['n_interior_H1']} interior H1(p+1) + "
                  f"{r['n_extra_wall_shells']} extra) rank Kp {r['rank_Kp']:>5} | A holds: {r['rank_condition_A']} | "
                  f"gap {r['zero_cluster_max_rel']:.1e} | {r['smallest_nonzero_rel']:.1e} | "
                  f"ker-rangeG {r['ker_minus_rangeG_M_norm_max']:.1e} | lam_min vs label "
                  f"{r['lam_min_nonzero_vs_label']:+.1e}"
                  + (f" | vs analytic {r['lam_min_nonzero_vs_analytic']:+.2e}" if "lam_min_nonzero_vs_analytic" in r
                     else ""), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
