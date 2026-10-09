"""Label-quality benchmark: current N0 labels and candidate high-order (NGSolve) labels against a
finer high-order reference, per geometry family — frequency AND fields.

For each sample (generator family + id, same geometry and smooth deformation as the dataset):

    N0 @ ms     the current label: Whitney N0 on the gmsh tet mesh at --mesh_size (dataset_generator_3d)
    N0 @ ms2    "more mesh" alternatives (--n0_mesh_sizes)
    p{p}c{c}    high-order candidates on the curved netgen mesh of the same CAD solid (--cand)
    p{p}c{c}→N0 that candidate interpolated to the N0 DOFs of the label mesh (u_e = ∫_e E·t): what
                the model would be trained on with high-order labels on the current mesh
    best N0     L2 projection of the reference onto the N0 space of the label mesh: the lower bound
                of the field error of ANY N0 field on that mesh (the model's representation limit)

all compared with the reference (--ref, default p = 4, geometry order 4, finer mesh).

Metrics per mode i < K (reference order; degenerate clusters — reference frequencies within
--cluster_tol — are fitted jointly, so rotations inside a cluster are not errors):
    df    f_i / f_ref,i − 1
    eE    ‖E_ref,i − Σ_j c_j E_j‖ / ‖E_ref,i‖ (L2 over the label mesh, best c over the candidate's
          cluster modes, sign / amplitude free)
    eH    the same c applied to curl E (the magnetic field)
    QoI   Q0, G, R/Q, T, Epk/Eacc, Bpk/Eacc of the fundamental (mode 0) and of the mode with the
          largest reference R/Q, relative to the reference (src.qoi conventions: copper, β = 1).

    python scripts/label_benchmark.py --families elliptical reentrant --ids 0 1 --k 6 \\
        --json bench.json --md bench.md
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.data_gen.dataset_generator_3d as gen  # noqa: E402
from src.data.dataset_converter_3d import (QOI_LABEL_SETTINGS, geometry_operators,  # noqa: E402
                                           qoi_operators_of)
from src.data_gen import highorder as ho  # noqa: E402
from src.qoi.analytic import fe_modes  # noqa: E402
from src.qoi.operators import QOI_LABELS, _local_basis, beam_axis, qoi_from_dofs  # noqa: E402
from src.viz.nedelec import _whitney_vectors, locate  # noqa: E402

QOI_SHOW = ("Q0", "G_ohm", "R_over_Q_ohm", "T_transit", "Epk_Eacc", "Bpk_Eacc_mT_per_MVm")
SHORT = {"Q0": "Q0", "G_ohm": "G", "R_over_Q_ohm": "R/Q", "T_transit": "T", "Epk_Eacc": "Epk/Eacc",
         "Bpk_Eacc_mT_per_MVm": "Bpk/Eacc"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--families", nargs="+", default=["elliptical", "reentrant", "pillbox_pipes", "ridged_box",
                                                      "composite", "hwr", "spoke", "dtl"])
    p.add_argument("--ids", type=int, nargs="+", default=[0], help="sample ids per family")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=6, help="modes compared (K)")
    p.add_argument("--mesh_size", type=float, default=0.10, help="label N0 mesh size (× V^(1/3))")
    p.add_argument("--n0_mesh_sizes", type=float, nargs="*", default=[0.07], help="extra N0 mesh sizes")
    p.add_argument("--cand", nargs="*", default=["3:3:3:0.5", "3:3:3:1"],
                   help="high-order candidates p:curve:maxh_factor:curvaturesafety (maxh = factor × label h; "
                        "netgen's curvaturesafety, default 2, refines curved faces whatever maxh is)")
    p.add_argument("--ref", default="4:4:3:1", help="reference p:curve:maxh_factor:curvaturesafety")
    p.add_argument("--deform", action="store_true", help="apply the generator's smooth deformation")
    p.add_argument("--cluster_tol", type=float, default=5e-3, help="relative gap that joins reference modes")
    p.add_argument("--quad_order", type=int, default=5, help="tet quadrature order of the field norms")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    return p.parse_args(argv)


def _spec(s):
    p, c, f, cs = s.split(":")
    return int(p), int(c), float(f), float(cs)


# ─────────────────────────── field sampling ────────────────────

def label_quadrature(X_phys, tets, order):
    """Points [Nt·q, 3] and weights of an order-`order` tet rule on the straight label mesh."""
    from ngsolve import TET, IntegrationRule
    ir = IntegrationRule(TET, order)
    ref = np.array([tuple(pt) for pt in ir.points])[:, :3]
    w = np.array(list(ir.weights))
    bary = np.column_stack([ref, 1.0 - ref.sum(1)])                  # [q, 4]
    V = X_phys[tets]                                                 # [Nt, 4, 3]
    pts = np.einsum("qi,tic->tqc", bary, V).reshape(-1, 3)
    d = np.linalg.det(V[:, 1:] - V[:, :1])
    wt = (np.abs(d)[:, None] * w[None, :]).reshape(-1)               # ref volume 1/6 · 6·vol
    return pts, wt


class N0Field:
    """Whitney field of a normalised N0 mesh (geom from fe_modes / geometry_operators) at physical
    points: E = Φ(ξ) u / s, curl E = C u / s² (DOFs are line integrals: coordinate-invariant)."""

    def __init__(self, geom, pts):
        X, tets, edges = geom["X"], geom["tets"], geom["edges"]
        s, c = float(geom["scale"]), np.asarray(geom["center"], dtype=np.float64)
        dof, loc, vol, grads, curls = _local_basis(X, tets, edges)
        tid, bary = locate(X, tets, (pts - c) / s)
        self.inside = tid >= 0
        ti = tid[self.inside]
        rows = np.repeat(np.flatnonzero(self.inside)[:, None], 6, 1)
        W = _whitney_vectors(grads[ti], loc[ti], bary[self.inside]) / s        # [n, 6, 3]
        Cv = curls[ti] / s ** 2
        shape = (len(pts), len(edges))
        self.Phi = [sp.csr_matrix((W[..., k].ravel(), (rows.ravel(), dof[ti].ravel())), shape=shape) for k in range(3)]
        self.Cur = [sp.csr_matrix((Cv[..., k].ravel(), (rows.ravel(), dof[ti].ravel())), shape=shape) for k in range(3)]

    def E(self, U):
        return np.stack([P @ U for P in self.Phi], axis=-1)                  # [N, K, 3]

    def H(self, U):
        return np.stack([P @ U for P in self.Cur], axis=-1)


def ho_fields(res, pts):
    """(E [N,K,3], curl E [N,K,3], inside [N]) of an NGSolve solve_modes result at points."""
    from ngsolve import curl
    loc = ho.locate(res["mesh"], pts)
    E = np.stack([ho.evaluate(g, loc) for g in res["gfs"]], axis=1)
    H = np.stack([ho.evaluate(curl(g), loc) for g in res["gfs"]], axis=1)
    return E, H, loc[1]


def clusters(f, tol):
    """Index groups of consecutive frequencies with relative gaps < tol."""
    groups, cur = [], [0]
    for i in range(1, len(f)):
        if f[i] / f[i - 1] - 1 < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    return groups + [cur]


def field_errors(ref, cand, w, groups, K):
    """eE, eH [K]: best cluster fit of every reference mode i < K by the candidate's modes of i's
    reference cluster (weighted L2 over the points inside both meshes)."""
    (ER, HR, inR), (EC, HC, inC) = ref, cand
    m = inR & inC
    ww = w * m
    eE, eH = np.full(K, np.nan), np.full(K, np.nan)
    for g in groups:
        g = [j for j in g if j < EC.shape[1]]
        A = (EC[:, g, :] * np.sqrt(ww)[:, None, None]).transpose(0, 2, 1).reshape(-1, len(g))
        Ah = HC[:, g, :].transpose(0, 2, 1).reshape(-1, len(g))
        for i in g:
            if i >= K:
                continue
            b = (ER[:, i, :] * np.sqrt(ww)[:, None]).reshape(-1)
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            eE[i] = np.linalg.norm(A @ c - b) / np.linalg.norm(b)
            hb = HR[:, i, :].reshape(-1)
            hr = (Ah @ c - hb) * np.repeat(np.sqrt(ww), 3)
            eH[i] = np.linalg.norm(hr) / np.linalg.norm(hb * np.repeat(np.sqrt(ww), 3))
    return eE, eH


def best_n0(field, ref, w, free):
    """L2 projection of the reference E onto the N0 space (wall DOFs 0) → DOFs [Ne, K]."""
    ER, _, inR = ref
    ww = sp.diags(w * inR * field.inside)
    Mq = sum(P.T @ ww @ P for P in field.Phi).tocsr()
    B = sum(P.T @ ww @ ER[:, :, k] for k, P in enumerate(field.Phi))
    U = np.zeros(B.shape)
    Mf = Mq[free][:, free].tocsc()
    U[free] = spsolve(Mf, B[free]).reshape(len(free), -1)
    return U


# ─────────────────────────── one sample ────────────────────────

def n0_label(fam, sid, ms, args, cad_path=None, dmap=None):
    """(geom, Y [Ne,K+2], f_hz, t, g) of the N0 solve at mesh size ms; dmap: apply this deformation
    (instead of drawing one) so every mesh discretises the same solid."""
    argv = ["--families", fam, "--mesh_size", str(ms), "--seed", str(args.seed),
            "--deform_prob", "1" if (args.deform and dmap is None) else "0"]
    gen.ARGS = gen.parse_args(argv)
    t0 = time.perf_counter()
    g = gen.mesh_sample(sid, cad_path=cad_path)
    nodes = g["nodes"]
    if dmap is not None:
        om, a, ph = dmap
        nodes = nodes + np.sin(nodes @ om.T + ph) @ a
    geom, Y, f = fe_modes(nodes, g["tets"], args.k + 2)
    geom["shape_type"] = g["shape_type"]
    return geom, Y, f, time.perf_counter() - t0, g


def rel(a, b):
    return np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64) - 1.0


def run_sample(fam, sid, args, tmp):
    K = args.k
    out = {"family": fam, "id": sid, "deform": bool(args.deform), "methods": {}}
    cad = os.path.join(tmp, f"{fam}_{sid}.brep")
    geom, Y, f0, t_lab, g = n0_label(fam, sid, args.mesh_size, args, cad_path=cad)
    if g.get("cad_path") is None:
        raise RuntimeError(f"{fam}: no CAD solid (discrete family)")
    dmap = g["deform"]
    h = g["h"]
    s, c = float(geom["scale"]), np.asarray(geom["center"])
    X_phys = np.asarray(geom["X"], dtype=np.float64) * s + c
    out.update(h=h, ne_label=int(len(geom["edges"])), params=g["params"])

    pr, cr, fr, csr = _spec(args.ref)
    t0 = time.perf_counter()
    mref = ho.netgen_mesh(cad, fr * h, curve=cr, deform=dmap, curvaturesafety=csr)
    ref = ho.solve_modes(mref, K + 2, order=pr)
    ref["mesh"] = mref
    out["ref"] = {"spec": args.ref, "f_hz": ref["f_hz"].tolist(), "t": time.perf_counter() - t0, **ref["info"]}
    print(f"  ref {args.ref}: ndof {ref['info']['ndof']} {out['ref']['t']:.1f}s  f/GHz "
          + " ".join(f"{x / 1e9:.4f}" for x in ref["f_hz"][:K]), flush=True)

    pts, w = label_quadrature(X_phys, geom["tets"], args.quad_order)
    R = ho_fields(ref, pts)
    groups = clusters(ref["f_hz"], args.cluster_tol)
    groups = [gr for gr in groups if gr[0] < K]
    out["clusters"] = groups

    axis = beam_axis(fam, geom["X"], s, c)
    ad = axis.get("axis_dir", "z")
    axis_dir = np.eye(3)["xyz".index(ad)] if isinstance(ad, str) else np.asarray(ad)
    axis_point = np.asarray(axis.get("axis_point", (0.0, 0.0, 0.0)))
    qref = ho.mode_qoi(mref, ref["gfs"][:K], ref["f_hz"][:K], axis_dir, axis_point,
                       sigma=QOI_LABEL_SETTINGS["sigma"], beta=QOI_LABEL_SETTINGS["beta"])
    i_acc = int(np.nanargmax(qref["R_over_Q_ohm"]))
    out["ref"]["qoi"] = {k: np.asarray(qref[k]).tolist() for k in QOI_LABELS}
    out["i_acc"] = i_acc

    def record(name, f, eE, eH, q, t, extra=None):
        dq = {k: [float(rel(q[k][i], qref[k][i])) for i in (0, i_acc)] for k in QOI_SHOW} if q else None
        out["methods"][name] = {"df": rel(f[:K], ref["f_hz"][:K]).tolist(), "eE": eE.tolist(), "eH": eH.tolist(),
                                "dqoi": dq, "t": t, **(extra or {})}
        print(f"  {name:14s} max|df| {np.abs(rel(f[:K], ref['f_hz'][:K])).max():.1e}  "
              f"eE med {np.nanmedian(eE):.1e} max {np.nanmax(eE):.1e}  eH med {np.nanmedian(eH):.1e}  "
              + ("  ".join(f"{SHORT[k]} {dq[k][0]:+.1e}" for k in QOI_SHOW) if dq else "") + f"  {t:.1f}s",
              flush=True)

    ops = qoi_operators_of(geom, M=geom["M"])
    lab = N0Field(geom, pts)
    q = qoi_from_dofs(ops, Y[:, :K], f0[:K] , **QOI_LABEL_SETTINGS)
    eE, eH = field_errors(R, (lab.E(Y), lab.H(Y), lab.inside), w, groups, K)
    record(f"N0@{args.mesh_size:g}", f0, eE, eH, q, t_lab)

    bnd = geometry_operators(geom["X"], geom["tets"])[0]["bnd_edge"]
    free = np.flatnonzero(~bnd)
    Ub = best_n0(lab, R, w, free)
    eE, eH = field_errors(R, (lab.E(Ub), lab.H(Ub), lab.inside), w, [[i] for i in range(K)], K)
    record("best N0", ref["f_hz"], eE, eH, qoi_from_dofs(ops, Ub[:, :K], ref["f_hz"][:K], **QOI_LABEL_SETTINGS), 0.0)

    for ms2 in args.n0_mesh_sizes:
        geom2, Y2, f2, t2, _ = n0_label(fam, sid, ms2, args, dmap=dmap)
        fld = N0Field(geom2, pts)
        ops2 = qoi_operators_of(geom2, M=geom2["M"])
        q2 = qoi_from_dofs(ops2, Y2[:, :K], f2[:K], **QOI_LABEL_SETTINGS)
        eE, eH = field_errors(R, (fld.E(Y2), fld.H(Y2), fld.inside), w, groups, K)
        record(f"N0@{ms2:g}", f2, eE, eH, q2, t2, {"ne": int(len(geom2["edges"]))})

    for spec in args.cand:
        p, cv, fac, cs = _spec(spec)
        t0 = time.perf_counter()
        mc = ho.netgen_mesh(cad, fac * h, curve=cv, deform=dmap, curvaturesafety=cs)
        rc = ho.solve_modes(mc, K + 2, order=p)
        rc["mesh"] = mc
        tc = time.perf_counter() - t0
        qc = ho.mode_qoi(mc, rc["gfs"][:K], rc["f_hz"][:K], axis_dir, axis_point,
                         sigma=QOI_LABEL_SETTINGS["sigma"], beta=QOI_LABEL_SETTINGS["beta"])
        eE, eH = field_errors(R, ho_fields(rc, pts), w, groups, K)
        name = f"p{p}c{cv}h{fac:g}s{cs:g}"
        record(name, rc["f_hz"], eE, eH, qc, tc, {"ndof": rc["info"]["ndof"]})
        Ui, frac = ho.edge_dofs(mc, rc["gfs"], X_phys, np.asarray(geom["edges"]))
        Ui[bnd] = 0.0
        qi = qoi_from_dofs(ops, Ui[:, :K], rc["f_hz"][:K], **QOI_LABEL_SETTINGS)
        eE, eH = field_errors(R, (lab.E(Ui), lab.H(Ui), lab.inside), w, groups, K)
        record(name + "→N0", rc["f_hz"], eE, eH, qi, tc, {"edge_inside": float(frac[~bnd].mean())})
    return out


# ─────────────────────────── report ────────────────────────────

def markdown(results, args):
    names = list(results[0]["methods"]) if results else []
    lines = [f"Reference: NGSolve {args.ref} (p:curve:maxh/h:curvaturesafety). K = {args.k}, label mesh {args.mesh_size}, "
             f"deform = {args.deform}.  df: max |f/f_ref − 1| over the K modes; eE / eH: median (max) relative "
             "L2 error of E / curl E; QoI: mode 0 relative error.", ""]
    head = "| family | id | method | max\\|df\\| | eE | eH | " + " | ".join(SHORT[k] for k in QOI_SHOW) + " | t [s] |"
    lines += [head, "|" + "---|" * (6 + len(QOI_SHOW) + 1)]
    for r in results:
        for n in names:
            m = r["methods"][n]
            eE, eH = np.asarray(m["eE"]), np.asarray(m["eH"])
            q = " | ".join(f"{m['dqoi'][k][0]:+.1e}" if m["dqoi"] else "" for k in QOI_SHOW)
            lines.append(f"| {r['family']} | {r['id']} | {n} | {np.abs(m['df']).max():.1e} | "
                         f"{np.nanmedian(eE):.1e} ({np.nanmax(eE):.1e}) | {np.nanmedian(eH):.1e} ({np.nanmax(eH):.1e}) | "
                         f"{q} | {m['t']:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    from ngsolve import SetNumThreads
    SetNumThreads(args.threads)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for fam in args.families:
            for sid in args.ids:
                print(f"{fam} #{sid}", flush=True)
                try:
                    results.append(run_sample(fam, sid, args, tmp))
                except Exception as e:                         # keep going: report the failure
                    print(f"  FAILED: {e}", flush=True)
                    results.append({"family": fam, "id": sid, "error": str(e)})
                if args.json:
                    with open(args.json, "w") as fh:
                        json.dump({"args": vars(args), "results": results}, fh, indent=1)
    ok = [r for r in results if "methods" in r]
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(markdown(ok, args))
    return results


if __name__ == "__main__":
    main()
