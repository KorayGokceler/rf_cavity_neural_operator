#!/usr/bin/env python
"""Measurements of the field display (docs/26_WEB_UI.md "Display pipeline", docs/31 §3.12).

    python scripts/measure_display.py accuracy [--maxh 0.02 0.012 0.008 0.006] [--n_points 5000]
    python scripts/measure_display.py serve --tets 10000 [--save payload.json]     # one process per size
    python scripts/measure_display.py js payload.json                               # browser-side slicing time

accuracy   pointwise error of the drawn E and H against the exact TM010 of a netgen pillbox (R = 40 mm,
           L = 50 mm, curve 3, p2 solution), in units of the peak E0 / H0, for src.viz.hcurl_display (P2 nodes + 8
           sub-tets per tet, linear in each sub-tet), sampled at random points through the same P1 interpolation the
           browser uses.  The display of the first web UI (Whitney-N0 `Set(dual)` + vertex average) was removed with
           docs/31 §6 item 15; its numbers stay in the "before" columns of docs/26 and docs/31.
serve      the API on a pillbox STEP meshed to about --tets curved tets (untrained model): geometry, predict
           (display stage), /fields payload raw and gzip, request times, peak RSS of the process.
js         src/fieldlab.ts (run by node) on a saved /fields payload: decode, slicePlane + sampleSlice per
           slider position, rasterize + sampleGrid (the 3D volume), in milliseconds.

`import torch` stays before ngsolve (segfault otherwise).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch  # noqa: F401  — before ngsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0  # noqa: E402
from scipy.special import j0, j1, jn_zeros  # noqa: E402

R, L = 0.040, 0.050
X01 = float(jn_zeros(0, 1)[0])


def pillbox(maxh, curve=3):
    from netgen.occ import Axes, Cylinder, OCCGeometry, Z
    from ngsolve import Mesh
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    m = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh))
    m.Curve(curve)
    return m


def tm010(pts):
    """Exact TM010 at U = 1 J (E = E0 J0(x01 r / R) ẑ, H = H0 J1(x01 r / R) φ̂): (E [N,3], H [N,3], E0, H0)."""
    e0 = np.sqrt(2.0 / (EPS0 * np.pi * R ** 2 * L)) / j1(X01)
    h0 = e0 * np.sqrt(EPS0 / MU0)
    r = np.hypot(pts[:, 0], pts[:, 1])
    arg = X01 * r / R
    E = np.zeros_like(pts)
    E[:, 2] = e0 * j0(arg)
    H = np.zeros_like(pts)
    rr = np.maximum(r, 1e-12)
    H[:, 0], H[:, 1] = -pts[:, 1] / rr * h0 * j1(arg), pts[:, 0] / rr * h0 * j1(arg)
    return E, H, e0, h0


def sample_points(n, seed=0):
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.uniform(0, 0.98 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-0.49 * L, 0.49 * L, n)], 1)


def new_display(mesh, U, f_ghz):
    """The p2 display: (P2 nodes [Nn,3] m, sub-tets, E, H, seconds)."""
    from src.viz import hcurl_display as hd
    t0 = time.perf_counter()
    dm = hd.display_mesh(mesh)
    fld = hd.nodal_fields(mesh, dm, U[:, None], np.asarray([f_ghz]), hd.mass_matrix(mesh, 2))
    return dm["x"], dm["tets"], fld["E"][0], fld["H"][0], time.perf_counter() - t0


def p1_at(nodes, tets, F, pts):
    from src.viz.meshsample import interp_vertex, locate
    tid, bary = locate(nodes, tets, pts)
    out = np.full((len(pts), 3), np.nan)
    ok = tid >= 0
    out[ok] = interp_vertex(tets, tid[ok], bary[ok], F)
    return out


def errors(approx, exact, peak):
    """(max, rms) of |approx − s·exact| / peak, s = ±1 (the eigenvector sign)."""
    s = 1.0 if np.nansum(approx * exact) >= 0 else -1.0
    e = np.linalg.norm(approx - s * exact, axis=1) / peak
    return float(np.nanmax(e)), float(np.sqrt(np.nanmean(e ** 2)))


def cmd_accuracy(a):
    from src.data_gen import highorder as ho
    from src.viz.meshsample import locate
    pts0 = sample_points(a.n_points * 3)
    rows = []
    for maxh in a.maxh:
        mesh = pillbox(maxh)
        sol = ho.solve_modes(mesh, 1, order=2)
        U, f = sol["gfs"][0].vec.FV().NumPy().copy(), float(sol["f_hz"][0]) / 1e9
        out = {"maxh": maxh, "tets": int(mesh.ne), "f_GHz": f}
        nodes, tets, E, H, sec = new_display(mesh, U, f)
        keep = locate(nodes, tets, pts0)[0] >= 0                       # sample points inside the display
        pts = pts0[keep][:a.n_points]
        Ex, Hx, e0, h0 = tm010(pts)
        ee = errors(p1_at(nodes, tets, E, pts), Ex, e0)
        he = errors(p1_at(nodes, tets, H, pts), Hx, h0)
        out.update(n_samples=int(len(pts)),
                   new={"E_max": ee[0], "E_rms": ee[1], "H_max": he[0], "H_rms": he[1], "seconds": sec})
        rows.append(out)
    print(json.dumps(rows, indent=1))
    print(f"\n{'tets':>7} {'E max':>9} {'E rms':>9} {'H max':>9} {'H rms':>9} {'stage s':>8}   (fractions of peak E0 / H0)")
    for o in rows:
        v = o["new"]
        print(f"{o['tets']:>7} {v['E_max']:9.2e} {v['E_rms']:9.2e} {v['H_max']:9.2e} {v['H_rms']:9.2e} {v['seconds']:8.2f}")


# ───────────────────────────── API payload ─────────────────────────────

def pillbox_step(d, length_mm=100.0):
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addCylinder(0, 0, -length_mm / 2, 0, 0, length_mm, 100)        # mm, R = 100
    gmsh.model.occ.synchronize()
    path = os.path.join(d, "pillbox.step")
    gmsh.write(path)
    gmsh.clear()
    return path


def _family_mesh(family, s_id, mesh_size, max_elements=None):
    """G.mesh_family with the API's 0.5 request value scaled to the real mesh size (RFCAV_MEASURE_MESH_SIZE):
    the API's minimum of 0.05 is too coarse to reach 30k tets (module-level: picklable for the spawned child)."""
    from src.service import geometry as G
    return G.mesh_family(family, s_id, float(os.environ["RFCAV_MEASURE_MESH_SIZE"]), max_elements)   # the child has the original


def cmd_serve(a):
    from fastapi.testclient import TestClient

    from src.service import api as A
    from src.service import geometry as G
    from src.service.model import ModelService
    A.Limits.min_mesh_size = 0.01            # a plain pillbox stays below 2k tets at the API's 0.05 minimum
    if a.family:
        G.mesh_family = _family_mesh
    client = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    out = {"target_tets": a.tets, "family": a.family or "pillbox (STEP upload)"}
    with tempfile.TemporaryDirectory() as d:
        step = None if a.family else pillbox_step(d, a.length_mm)

        def make(ms):
            t0 = time.perf_counter()
            if a.family:
                os.environ["RFCAV_MEASURE_MESH_SIZE"] = str(ms)
                r = client.post("/api/geometries/sample", json={"family": a.family, "id": a.id, "mesh_size": 0.5})
            else:
                with open(step, "rb") as fh:
                    r = client.post("/api/geometries", files={"file": ("cavity.step", fh)},
                                    data={"unit": "mm", "mesh_size": str(ms)})
            assert r.status_code == 200, r.text
            return r.json(), time.perf_counter() - t0

        ms = a.mesh_size
        g, t_geom = make(ms)
        for _ in range(6):                       # tets ∝ h⁻³ (roughly): correct the mesh size towards the target
            n = g["check"]["n_tets"]
            if abs(n / a.tets - 1) < a.tol and g["can_predict"]:
                break
            ms = min(0.5, max(0.005, ms * (n / a.tets) ** (1 / 3)))
            g, t_geom = make(ms)
    out.update(mesh_size=round(ms, 4), n_tets=g["check"]["n_tets"], geometry_s=round(t_geom, 2),
               surface_points=g["surface"]["n_points"], geometry_response_bytes=len(json.dumps(g)))
    t0 = time.perf_counter()
    r = client.post("/api/predictions", json={"geometry_id": g["id"]})
    assert r.status_code == 200, r.text
    p = r.json()
    out["predict_request_s"] = round(time.perf_counter() - t0, 2)
    out["time_s"] = {k: round(v, 3) for k, v in p["time_s"].items()}
    out["n_dof"] = p["n_dof"]
    pid = p["id"]
    t0 = time.perf_counter()
    raw = client.get(f"/api/predictions/{pid}/fields", headers={"Accept-Encoding": "identity"})
    out["fields_request_s"] = round(time.perf_counter() - t0, 2)
    out["fields_raw_bytes"] = len(raw.content)
    t0 = time.perf_counter()
    gz = client.get(f"/api/predictions/{pid}/fields", headers={"Accept-Encoding": "gzip"})
    out["fields_gzip_request_s"] = round(time.perf_counter() - t0, 2)
    out["fields_gzip_bytes"] = int(gz.headers.get("content-length", len(gz.content)))
    j = raw.json()
    out["fields_keys"] = sorted(j)
    if a.save:
        with open(a.save, "w") as fh:
            fh.write(raw.text)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(json.dumps(out, indent=1))


def cmd_js(a):
    runner = os.path.join(ROOT, "scripts", "measure_display_js.mjs")
    subprocess.run(["node", runner, a.payload], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("accuracy")
    s.add_argument("--maxh", type=float, nargs="+", default=[0.02, 0.012, 0.008, 0.006])
    s.add_argument("--n_points", type=int, default=5000)
    s.set_defaults(fn=cmd_accuracy)
    s = sub.add_parser("serve")
    s.add_argument("--tets", type=int, default=10000)
    s.add_argument("--mesh_size", type=float, default=0.1)
    s.add_argument("--length_mm", type=float, default=100.0, help="pillbox length (R = 100 mm)")
    s.add_argument("--family", default=None, help="a generator family sample instead of the pillbox STEP")
    s.add_argument("--id", type=int, default=3)
    s.add_argument("--tol", type=float, default=0.08, help="accepted relative deviation from --tets")
    s.add_argument("--save", default=None)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("js")
    s.add_argument("payload")
    s.set_defaults(fn=cmd_js)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
