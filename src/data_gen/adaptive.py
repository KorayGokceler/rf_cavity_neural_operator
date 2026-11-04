"""Adaptive mesh refinement of the high-order cavity eigenmodes (docs/30).

Loop (one geometry, curved mesh from highorder.netgen_mesh):

    solve HCurl(p) eigenpairs → figures of merit → error indicator per element → mark (Dörfler θ) →
    bisect (netgen Refine: new boundary points on the CAD) → re-curve, re-deform, quality check → …

until the cluster frequencies AND the figures of merit change less than tol_f / tol_q between two
iterations, or the next mesh would exceed the DOF budget.

Indicator (flux recovery; unit-L2 modes, so ∫|curl E|² = λ):

    η_vol,T²  = Σ_k ( ‖curl E_k − R curl E_k‖²_T + λ_k ‖E_k − R E_k‖²_T ) / λ_k
    η_wall,T² = Σ_k ( ∫_{∂T∩wall} |curl E_k − R curl E_k|² / ∫_wall |curl E_k|²  +  same for E_k )

R = averaging interpolation into VectorH1(p): the jumps of the discrete fields are the error.
Σ η_vol² estimates the relative eigenvalue error (λ_h − λ ≈ ‖curl e‖² − λ‖e‖², second order in the
field error); the figures of merit (wall loss, R/Q, surface peaks) are FIRST order in the field error
near the wall, hence the squared relative wall term.  At the targets (f 1e-6 ↔ λ 2e-6, Q0 1e-3 ↔ 1e-6
squared) both are ~1e-6, so they add with weight 1.  Measured on the analytic copper pillbox (docs/30
§2): the wall term cuts the Q0 / B_pk errors 3× at equal DOF, and the estimate of the frequency
error is within 1.5× of the true error at every level.

Degenerate pairs: the convergence check uses cluster-invariant quantities (mean f, mean Q0, mean G,
Σ R/Q, max peaks), which do not depend on the rotation inside a pair.
"""
import time

import numpy as np

from src.data_gen import highorder as ho

DEFAULTS = {"theta": 0.5, "tol_f": 1e-6, "tol_q": 1e-3, "max_iter": 8, "max_ndof": 900_000, "min_iter": 2,
            "wall_weight": 1.0, "cluster_rel": 1e-3, "rq_floor": 1e-2, "min_jacobian_ratio": 0.05, "peaks": True,
            "rho": 0.25,
            # forced growth (docs/31 item 14): every remeshed pass must add >= min_growth·100 % tets (target
            # force_growth x the old count), else the loop stops 'stalled' instead of declaring convergence.
            # rq_abs_floor [ohm]: R/Q below it counts as zero (no relative change of numerical noise, blob: 1e59)
            "min_growth": 0.15, "force_growth": 1.3, "rq_abs_floor": 1e-3}
QNAMES = ("f_Hz", "Q0", "G_ohm", "R_over_Q_ohm", "E_pk_Vm", "B_pk_T")


def indicator(mesh, gfs, lam, order, wall=True):
    """(η_vol², η_wall²) per volume element (module docstring)."""
    from ngsolve import (BoundaryFromVolumeCF, FacetFESpace, GridFunction, InnerProduct, Integrate,
                         TaskManager, VectorH1, VOL, curl, ds, dx)
    V = VectorH1(mesh, order=order)
    rB, rE = GridFunction(V), GridFunction(V)
    ev, ew = np.zeros(mesh.ne), np.zeros(mesh.ne)
    if wall:
        chi = GridFunction(FacetFESpace(mesh, order=0))          # 1 on the wall facets
        chi.Set(1.0, definedon=mesh.Boundaries("wall"))
        deb = dx(element_boundary=True, bonus_intorder=4)
    with TaskManager():
        for g, lk in zip(gfs, lam):
            rB.Set(curl(g))
            rE.Set(g)
            dB, dE = curl(g) - rB, g - rE
            e = Integrate(InnerProduct(dB, dB) + float(lk) * InnerProduct(dE, dE), mesh, VOL,
                          element_wise=True, order=2 * order + 2)
            ev += np.asarray(e) / float(lk)
            if wall:
                cb, ce = BoundaryFromVolumeCF(curl(g)), BoundaryFromVolumeCF(g)
                nb = Integrate(InnerProduct(cb, cb) * ds("wall"), mesh)
                ne_ = Integrate(InnerProduct(ce, ce) * ds("wall"), mesh)
                wb = np.asarray(Integrate(chi * InnerProduct(dB, dB) * deb, mesh, element_wise=True))
                we = np.asarray(Integrate(chi * InnerProduct(dE, dE) * deb, mesh, element_wise=True))
                ew += wb / max(nb, 1e-300) + we / max(ne_, 1e-300)
    return ev, ew


def mark_dorfler(eta, theta):
    """Smallest set of elements carrying the fraction theta of Σ η² (bool [ne])."""
    idx = np.argsort(eta)[::-1]
    c = np.cumsum(eta[idx])
    n = int(np.searchsorted(c, theta * c[-1])) + 1
    flags = np.zeros(len(eta), bool)
    flags[idx[:n]] = True
    return flags


def refine(mesh, flags, curve, deform=None, min_jacobian_ratio=0.05, fix_passes=2):
    """Bisect the marked elements, re-curve (and re-deform); then bisect elements whose curved map is
    distorted (min/max det J < min_jacobian_ratio) up to fix_passes times."""
    from ngsolve import VOL, ElementId

    def bisect(fl):
        for e in range(mesh.ne):
            mesh.SetRefinementFlag(ElementId(VOL, e), bool(fl[e]))
        mesh.Refine()
        mesh.Curve(int(curve))
        if deform is not None:
            ho.set_deformation(mesh, *deform, order=int(curve))

    bisect(flags)
    for _ in range(fix_passes):
        r = ho.element_jacobian_ratio(mesh)
        bad = r < min_jacobian_ratio
        if not bad.any():
            break
        bisect(bad)


def element_geometry(mesh):
    """(centroids [ne, 3] of the straight tets, size h [ne] = edge of the regular tet of the same (curved)
    volume)."""
    from ngsolve import CF, VOL, Integrate
    vol = np.abs(np.asarray(Integrate(CF(1.0), mesh, VOL, element_wise=True)))
    X = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    T = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.ngmesh.Elements3D()], dtype=np.int64)
    return X[T].mean(1), (6 * np.sqrt(2) * vol) ** (1 / 3)


def size_field(h, eta, order, rho=0.25, shrink=0.3, grow=1.0):
    """New local size per element: equidistribute the error at rho × its current mean.  The error of the
    region of element T scales as (h_new/h)^(2p) (energy norm², p = polynomial order), so
        h_new = h · (rho · mean(η²) / η_T²)^(1/(2p)),  clipped to [shrink, grow] · h
    (grow > 1 would coarsen where the error is small; with the accumulating remesher it has no effect)."""
    eta = np.maximum(np.asarray(eta, float), 1e-300)
    fac = (rho * eta.mean() / eta) ** (1.0 / (2 * order))
    return h * np.clip(fac, shrink, grow)


def mesh_remesher(shape, maxh, curve=3, deform=None, curvaturesafety=2.0, grading=0.5, min_jacobian_ratio=0.05,
                  tries=3):
    """remesh(centroids, h_new) → new curved mesh of `shape` whose local size is at most h_new around each
    point (netgen RestrictH; grading smooths the transitions).  Distorted curved elements (det-J ratio below
    min_jacobian_ratio) get their size halved and the shape is remeshed (no bisection: its closure grows a
    graded mesh 30–100 new tets per marked one, docs/30 §3).  The restrictions ACCUMULATE over the calls
    (RestrictH is an upper bound, so the smallest request wins): remeshing from scratch with only the
    latest field oscillates — a region refined at one iteration is coarsened again at the next.  The
    coarse end comes from the coarse start mesh (low curvaturesafety), not from coarsening."""
    acc = {"cen": np.zeros((0, 3)), "h": np.zeros(0)}

    def remesh(cen, hn):
        from netgen.meshing import MeshingParameters
        from netgen.occ import OCCGeometry
        from ngsolve import Mesh
        acc["cen"] = np.vstack([acc["cen"], np.asarray(cen, float)])
        acc["h"] = np.r_[acc["h"], np.asarray(hn, float)]
        cen, hn = acc["cen"].copy(), acc["h"].copy()
        for t in range(tries):
            mp = MeshingParameters(maxh=float(maxh), curvaturesafety=float(curvaturesafety), grading=float(grading))
            for c, hh in zip(cen, hn):
                mp.RestrictH(x=float(c[0]), y=float(c[1]), z=float(c[2]), h=float(hh))
            mesh = Mesh(OCCGeometry(shape).GenerateMesh(mp=mp))
            mesh.Curve(int(curve))
            if deform is not None:
                ho.set_deformation(mesh, *deform, order=int(curve))
            r = ho.element_jacobian_ratio(mesh)
            if r.min() > min_jacobian_ratio:
                return mesh
            bad_c, hb = element_geometry(mesh)
            bad = r <= min_jacobian_ratio
            cen = np.vstack([cen, bad_c[bad]])
            hn = np.r_[hn, 0.5 * hb[bad]]
            acc["cen"], acc["h"] = cen.copy(), hn.copy()
        raise RuntimeError(f"remeshed curved mesh still has distorted elements (min det-J ratio {r.min():.2f})")

    def reset():                                  # netgen overshot: the accumulated restrictions cannot be undone
        acc["cen"], acc["h"] = np.zeros((0, 3)), np.zeros(0)
    remesh.reset = reset
    return remesh


def _bisect_rho(g, target, decreasing):
    """rho (log-bisection) with g(rho) = target for a monotone g; the end of the range when unreachable."""
    lo, hi = -25.0, 25.0                          # ln rho
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (g(np.exp(mid)) > target) == decreasing:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def predicted_count(h, hn):
    """Element count of a mesh of local size hn that replaces elements of size h (tets ∝ h⁻³)."""
    return float(np.sum((np.asarray(h, float) / np.asarray(hn, float)) ** 3))


def predicted_error_ratio(h, hn, eta, order):
    """Σ η' / Σ η after the elements shrink from h to hn (η ∝ h^(2·order), the model of size_field)."""
    eta = np.maximum(np.asarray(eta, float), 1e-300)
    return float(np.sum(eta * (np.asarray(hn, float) / np.asarray(h, float)) ** (2 * order)) / eta.sum())


def size_field_for_count(h, eta, order, n_target, shrink=0.25):
    """size_field scaled to a TARGET element count (the fix of the stall: aim at a count, not at rho·mean(η)):
    the equidistribution level rho is searched so that Σ(h/hn)³ = n_target; hn ≤ h always (n_target ≤ len(h)
    gives h itself; above the count of the all-shrink field the shrink-clipped field)."""
    h = np.asarray(h, float)
    if n_target <= len(h):
        return h.copy()
    rho = _bisect_rho(lambda r: predicted_count(h, size_field(h, eta, order, r, shrink)), n_target, True)
    return np.minimum(size_field(h, eta, order, rho, shrink), h)


def size_field_for_reduction(h, eta, order, ratio, shrink=0.25):
    """size_field (equidistributing: the per-element error is equalised, exponent 1/(2·order); not the count-optimal
    exponent 1/(2·order+3)) whose predicted error ratio (predicted_error_ratio) is `ratio` < 1; the
    shrink-clipped field when that ratio is out of reach."""
    h = np.asarray(h, float)
    if ratio >= 1.0:
        return h.copy()
    rho = _bisect_rho(lambda r: predicted_error_ratio(h, size_field(h, eta, order, r, shrink), eta, order), ratio, False)
    return np.minimum(size_field(h, eta, order, rho, shrink), h)


def grow_mesh(remesh, mesh, eta, order, n_target, n_hard, cen=None, h=None, min_growth=0.15, tries=3, shrink=0.25,
              log=None):
    """One remeshed pass with FORCED growth → (new mesh | None, status, info).  Aims at n_target tets (size_field_for_count
    from the per-element indicator eta); netgen's RestrictH does not follow the field exactly, so the REALISED count
    decides: < (1+min_growth)·old → retry with a stronger target (up to `tries` meshings, status 'stalled' when none
    grew); > n_hard (the budget) → rebuild the remesher (remesh.reset(): accumulated restrictions cannot be undone) and
    retry smaller, status 'budget' when no mesh in [(1+min_growth)·old, n_hard] was found.  Errors of remesh propagate.
    info: meshings, targets [per meshing], ne [realised per meshing]."""
    if cen is None or h is None:
        cen, h = element_geometry(mesh)
    ne = int(mesh.ne)
    lo = (1.0 + min_growth) * ne
    info = {"meshings": 0, "targets": [], "ne": []}
    if n_hard < lo:
        return None, "budget", info
    target, status = min(float(n_target), n_hard), "stalled"
    for _ in range(int(tries)):
        target = max(target, 1.05 * lo)
        hn = size_field_for_count(h, eta, order, min(target, n_hard), shrink)
        new = remesh(cen, hn)
        info["meshings"] += 1
        info["targets"].append(float(target))
        info["ne"].append(int(new.ne))
        if log is not None:
            log(f"grow_mesh: target {target:.0f} tets from {ne}, got {new.ne}")
        if new.ne > n_hard:
            if hasattr(remesh, "reset"):
                remesh.reset()
            target = max(0.9 * n_hard * min(target, n_hard) / new.ne, 1.05 * lo)
            status = "budget"
            continue
        if new.ne >= lo:
            return new, "ok", info
        status = "stalled"
        target = min(n_hard, target * max(1.2, 1.3 * lo / max(new.ne, 1)))   # realised growth too small: aim higher
    return None, status, info


def qoi_change(prev_inv, inv, s):
    """(df, dq) between two invariant_qoi tables: max relative change of the mean frequency, and of the
    other figures of merit (Q0, G, peaks when s['peaks'], Σ R/Q of the accelerating clusters).  Relative to
    max(|previous|, floor): R/Q below s['rq_abs_floor'] ohm is zero (no ratio of numerical noise: blob gave 1e59)."""
    floor = np.zeros(inv.shape[1])
    floor[3] = s["rq_abs_floor"]
    den = np.maximum(np.abs(prev_inv), floor)
    rel = np.where(den > 0, np.abs(inv - prev_inv) / np.where(den > 0, den, 1.0), 0.0)
    acc = (inv[:, 3] > s["rq_floor"] * inv[:, 3].max()) & (inv[:, 3] > s["rq_abs_floor"])
    cols = [1, 2, 4, 5] if s["peaks"] else [1, 2]            # sharp edges left: peaks are not converging
    return float(rel[:, 0].max()), float(max(rel[:, cols].max(), rel[acc, 3].max() if acc.any() else 0.0))


def clusters(f, rel=1e-3):
    """Consecutive near-degenerate groups of the ascending frequencies f."""
    out, cur = [], [0]
    for i in range(1, len(f)):
        if abs(f[i] / f[i - 1] - 1) < rel:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def invariant_qoi(f_hz, q, rel=1e-3):
    """[n_clusters, 6] in QNAMES order: mean f, mean Q0, mean G, Σ R/Q, max E_pk, max B_pk (U = 1 J)."""
    return np.array([[np.mean(f_hz[c]), np.mean(q["Q0"][c]), np.mean(q["G_ohm"][c]), np.sum(q["R_over_Q_ohm"][c]),
                      np.max(q["E_pk_Vm"][c]), np.max(q["B_pk_T"][c])] for c in clusters(f_hz, rel)])


def adaptive_modes(mesh, k, order=3, curve=3, deform=None, axis=((0, 0, 1), (0, 0, 0)), settings=None,
                   on_iter=None, log=None, remesh=None):
    """Adaptive refinement for the k lowest modes.  remesh (mesh_remesher(...)): every iteration builds a
    NEW mesh from the error-equidistributing size field (size_field scaled to a target count, grow_mesh: every
    pass is forced to add ≥ min_growth of the tets, inside the DOF budget; 'converged' needs the criteria AND a
    pass that really grew the mesh, else 'stalled'); remesh=None: bisection of the
    Dörfler-marked elements of `mesh` (modified in place; on graded meshes the closure explodes).

    axis: (axis_dir, axis_point[, L_acc]) of mode_qoi.
    Returns dict(mesh, lab (solve_modes of the final mesh, k + 1 modes), q (mode_qoi, k modes), hist
    (one record per iteration), stop ('converged' | 'budget' | 'max_iter' | 'stalled' [remesh: no growth])).  on_iter(mesh, lab, q, rec)
    is called after every solve (e.g. to keep an intermediate mesh)."""
    s = {**DEFAULTS, **(settings or {})}
    hist, prev, growth = [], None, 6.0        # new elements per marked element (bisection closure); measured
    for it in range(int(s["max_iter"])):
        t0 = time.perf_counter()
        lab = ho.solve_modes(mesh, k + 1, order=order)
        q = ho.mode_qoi(mesh, lab["gfs"][:k], lab["f_hz"][:k], *axis)
        inv = invariant_qoi(lab["f_hz"][:k], q, s["cluster_rel"])
        ev, ew = indicator(mesh, lab["gfs"][:k], lab["lam"][:k], order, wall=s["wall_weight"] > 0)
        rec = {"it": it, "ne": int(mesh.ne), "ndof": int(lab["fes"].ndof), "est_rel_f": 0.5 * float(ev.sum()) / k,
               "est_rel_wall": float(np.sqrt(ew.sum() / k)), "inv": inv}
        if prev is not None:
            rec["growth"] = float(mesh.ne) / prev["ne"]
        if prev is not None and prev["inv"].shape == inv.shape:
            rec["df"], rec["dq"] = qoi_change(prev["inv"], inv, s)
        rec["t"] = time.perf_counter() - t0
        hist.append(rec)
        if on_iter is not None:
            on_iter(mesh, lab, q, rec)
        if log is not None:
            log(f"adapt it {it}: ne {mesh.ne} ndof {rec['ndof']} est δf/f {rec['est_rel_f']:.1e} "
                f"wall {rec['est_rel_wall']:.1e}" + (f" Δf {rec['df']:.1e} Δq {rec['dq']:.1e}" if "df" in rec else ""))
        done = "df" in rec and it + 1 >= s["min_iter"] and rec["df"] < s["tol_f"] and rec["dq"] < s["tol_q"]
        eta = ev + s["wall_weight"] * ew
        if remesh is not None:
            # convergence counts only when the last pass really grew the mesh: two nearly identical meshes
            # always agree (docs/31 item 14); otherwise another (forced-growth) pass is tried
            done = done and rec.get("growth", 1.0) >= 1.0 + s["min_growth"]
            dof_per_el = rec["ndof"] / mesh.ne
            ne_max = s["max_ndof"] / dof_per_el / 1.25                # 25 % margin (grading, curvature)
            ret = lambda stop: {"mesh": mesh, "lab": lab, "q": q, "hist": hist, "stop": stop}  # noqa: E731
            if done:
                return ret("converged")
            if ne_max < (1.0 + s["min_growth"]) * mesh.ne:     # budget reached: no real refinement left
                return ret("budget")
            if it == s["max_iter"] - 1:
                return ret("max_iter")
            cen, h = element_geometry(mesh)
            ne_next = predicted_count(h, size_field(h, eta, order, s["rho"]))
            target = min(max(ne_next, s["force_growth"] * mesh.ne), ne_max)      # forced growth, inside the budget
            new, status, info = grow_mesh(remesh, mesh, eta, order, target, ne_max, cen=cen, h=h,
                                          min_growth=s["min_growth"])
            rec["meshings"] = info["meshings"]
            if new is None:                                           # 'stalled' | 'budget'
                return ret(status)
            from ngsolve import HCurl
            if HCurl(new, order=order, dirichlet="wall").ndof > s["max_ndof"]:   # the built space, not the prediction
                return ret("budget")
            mesh = new
            prev = rec
            continue
        flags = mark_dorfler(eta, s["theta"])
        # keep the next mesh inside the DOF budget: mark fewer (the largest η) when the predicted growth
        # would exceed it; stop when less than 2 % of the elements could still be refined
        n_allowed = int((s["max_ndof"] * mesh.ne / rec["ndof"] - mesh.ne) / growth)
        over = n_allowed < 0.02 * mesh.ne
        if not over and flags.sum() > n_allowed:
            flags = np.zeros(mesh.ne, bool)
            flags[np.argsort(eta)[::-1][:n_allowed]] = True
        if done or over or it == s["max_iter"] - 1:
            stop = "converged" if done else ("budget" if over else "max_iter")
            return {"mesh": mesh, "lab": lab, "q": q, "hist": hist, "stop": stop}
        ne0, nm = mesh.ne, int(flags.sum())
        refine(mesh, flags, curve, deform, s["min_jacobian_ratio"])
        growth = max(1.0, 1.2 * (mesh.ne - ne0) / max(nm, 1))      # 20 % margin
        prev = rec
