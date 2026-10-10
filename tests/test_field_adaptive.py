"""Adaptive refinement of the high-order labels (src/data_gen/adaptive.py) and the rounding of re-entrant
edges (cavity_shapes.round_reentrant_edges), docs/30.

File name: runs after test_data_gen_3d.py — a process holding NGSolve / netgen state must not fork the
generator's multiprocessing Pool (the forked child crashes); production parents never load netgen."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")
from scipy.special import jn_zeros  # noqa: E402

from src.data_gen import adaptive as ad  # noqa: E402

C0 = 299792458.0
R, L = 0.040, 0.050


def _pillbox(maxh):
    from netgen.occ import Axes, Cylinder, OCCGeometry, Z
    from ngsolve import Mesh
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    m = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh))
    m.Curve(3)
    return m


def test_mark_dorfler_and_clusters():
    eta = np.array([1.0, 8.0, 0.5, 0.5])
    assert ad.mark_dorfler(eta, 0.5).tolist() == [False, True, False, False]
    assert ad.mark_dorfler(eta, 0.9).sum() == 2
    assert ad.clusters(np.array([1.0, 2.0, 2.0005, 3.0])) == [[0], [1, 2], [3]]


def test_adaptive_pillbox_converges_to_the_analytic_tm010():
    f_tm010 = jn_zeros(0, 1)[0] * C0 / (2 * np.pi * R)
    mesh = _pillbox(0.02)
    errs = []
    res = ad.adaptive_modes(mesh, 1, order=2, axis=((0, 0, 1), (0, 0, 0)),
                            on_iter=lambda m, lab, q, rec: errs.append(abs(lab["f_hz"][0] / f_tm010 - 1)),
                            settings={"tol_f": 1e-12, "tol_q": 1e-12, "max_iter": 3, "max_ndof": 200_000})
    h = res["hist"]
    assert len(h) == 3 and res["stop"] == "max_iter"
    assert h[-1]["ne"] > h[0]["ne"]
    assert errs[-1] < 0.2 * errs[0]                              # refinement reduces the true error
    assert h[-1]["est_rel_f"] < h[0]["est_rel_f"]
    assert 0.2 < h[0]["est_rel_f"] / errs[0] < 5                 # the estimate tracks the true error
    assert h[-1]["df"] < 1e-3 and np.isfinite(h[-1]["dq"])


def test_remesh_mode_refines_where_the_error_is_and_respects_the_budget():
    from netgen.occ import Axes, Cylinder, Z
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    f_tm010 = jn_zeros(0, 1)[0] * C0 / (2 * np.pi * R)
    errs = []
    remesh = ad.mesh_remesher(cyl, 0.02, curve=3)
    res = ad.adaptive_modes(_pillbox(0.02), 1, order=2, remesh=remesh,
                            on_iter=lambda m, lab, q, rec: errs.append(abs(lab["f_hz"][0] / f_tm010 - 1)),
                            settings={"tol_f": 1e-12, "tol_q": 1e-12, "max_iter": 3, "max_ndof": 60_000})
    h = res["hist"]
    assert len(h) >= 2 and errs[-1] < 0.3 * errs[0]
    assert all(r["ndof"] <= 60_000 * 1.3 for r in h)              # predicted size kept inside the budget
    cen, size = ad.element_geometry(res["mesh"])
    assert cen.shape == (res["mesh"].ne, 3) and np.all(size > 0)
    hn = ad.size_field(np.ones(4), np.array([1.0, 1.0, 1.0, 100.0]), 2)
    assert hn[3] < hn[0] and np.all(hn <= 1.0) and np.all(hn >= 0.3)  # refine the large-error element


def test_budget_stops_refinement():
    mesh = _pillbox(0.02)
    res = ad.adaptive_modes(mesh, 1, order=2, settings={"tol_f": 1e-12, "tol_q": 1e-12, "max_iter": 6,
                                                         "max_ndof": 6_000})
    assert res["stop"] == "budget"
    assert all(r["ndof"] <= 6_000 * 1.5 for r in res["hist"])


def test_reentrant_edges_are_found_and_rounded():
    gmsh = pytest.importorskip("gmsh")
    from src.data_gen import cavity_shapes as cs
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("reentrant_test")
    try:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 2, 2, 1)
        out, _ = occ.cut([(3, box)], [(3, occ.addBox(1, 1, -1, 2, 2, 3))])     # L shape: one 270° edge
        occ.synchronize()
        vol = out[0][1]
        assert len(cs.reentrant_edges(vol, 1e-3)) == 1
        vol2, n_round, r, n_sharp = cs.round_reentrant_edges(occ, vol, 0.1)
        assert (n_round, n_sharp) == (1, 0) and r == pytest.approx(0.1)
        assert cs.reentrant_edges(vol2, 1e-3) == []
        box2 = occ.addBox(5, 5, 5, 1, 1, 1)                                     # convex only: untouched
        occ.synchronize()
        assert cs.round_reentrant_edges(occ, box2, 0.1)[1:] == (0, 0.0, 0)
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()
