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
        assert cs.round_reentrant_edges(occ, vol2, 0.1, fillet=False)[1:] == (0, 0.0, 0)
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def test_a_rounded_geometry_that_fails_is_labelled_sharp_not_redrawn():
    """Port-junction fillets fold the curved netgen mesh at their vertices whatever the mesh size (docs/30
    §3): the generator then labels the SAME draw with sharp edges (peaks NaN) instead of re-drawing it —
    re-drawing dropped every ported elliptical design.  Sample 524296 (seed 0, Sobol) draws two ports."""
    import os
    import tempfile
    pytest.importorskip("gmsh")
    from src.data_gen import cavity_shapes as cs
    from src.data_gen import dataset_generator_3d as gen
    old, old_fillet = gen.ARGS, cs.MIN_FILLET
    gen.ARGS = gen.parse_args(["--labels", "field", "--families", "elliptical", "--seed", "0", "--sampling", "sobol",
                               "--mesh_size", "0.3"])
    seen = []

    def post(path, h, dmap, st, prm):              # stands in for a curved mesh that folds on the fillets
        seen.append(dict(prm))
        if prm["n_rounded_edges"] > 0:
            raise RuntimeError("curved mesh has folded elements")
        return "labels"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            g = gen.mesh_sample(524296, cad_path=os.path.join(tmp, "s.brep"), post=post)
    finally:
        gen.ARGS, cs.MIN_FILLET = old, old_fillet
    assert len(seen) == 2 and g["post"] == "labels"
    rounded, sharp = seen
    assert rounded["n_rounded_edges"] > 0 and rounded["round_fallback"] == 0
    assert sharp["n_rounded_edges"] == 0 and sharp["n_sharp_edges"] == rounded["n_rounded_edges"]
    assert sharp["round_fallback"] == 1 and g["params"]["round_fallback"] == 1
    geo = [k for k in rounded if k not in ("n_rounded_edges", "r_rounded_edges", "n_sharp_edges", "round_fallback")]
    assert geo and all(rounded[k] == sharp[k] for k in geo)              # the same draw, only sharp


def test_single_level_gives_the_numbering_of_a_reload():
    """After a bisection (the curved-mesh repair) NGSolve keeps the hierarchy, whose topology still numbers
    the parents' edges and faces: HCurl would have DOFs a reload of the same mesh does not.  single_level
    rebuilds it so that the stored fields match the mesh FieldDataset / the UI rebuild."""
    from netgen.occ import Axes, Cylinder, Z
    from ngsolve import VOL, ElementId, HCurl
    from src.data_gen import highorder as ho
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    m = _pillbox(0.02)
    for e in range(m.ne):
        m.SetRefinementFlag(ElementId(VOL, e), e % 7 == 0)
    m.Refine()
    m.Curve(3)
    assert m.levels == 2
    s = ho.single_level(m, cyl, 3)
    assert s.levels == 1 and s.ne == m.ne and s.nv == m.nv
    assert HCurl(s, order=2).ndof < HCurl(m, order=2).ndof          # the parents' unused DOFs are gone
    assert ho.single_level(s, cyl, 3) is s


def _solved(maxh, order=2, k=3):
    from src.data_gen import highorder as ho
    mesh = _pillbox(maxh)
    return mesh, ho.solve_modes(mesh, k, order=order)


def test_transfer_to_model_with_one_point_search_equals_direct_integration():
    """transfer_to_model locates the model mesh's integration points once and reads the label fields from an
    IntegrationRuleSpace; integrating the label GridFunctions directly (one point search per form and mode,
    253 s of 664 s per sample at 430k label DOFs) must give the same projection."""
    from ngsolve import BilinearForm, HCurl, InnerProduct, Integrate, LinearForm, TaskManager
    from src.data_gen import field_labels as fl
    from src.data_gen import highorder as ho
    _, lab = _solved(0.016)
    gfs = lab["gfs"]
    mesh_model = _pillbox(0.027)
    U, err, rq, fes2 = fl.transfer_to_model(gfs, mesh_model, 2, 2)
    dxq = ho.volume_measure(2)
    fes_ref = HCurl(mesh_model, order=2, dirichlet="wall")
    u2, v2 = fes_ref.TnT()
    with TaskManager():
        A = BilinearForm(u2 * v2 * dxq, symmetric=True).Assemble()
        Ainv = A.mat.Inverse(fes_ref.FreeDofs(), inverse="sparsecholesky")
    A2 = ho._tosp(A.mat)
    assert fes2.ndof == fes_ref.ndof
    for k, g in enumerate(gfs):
        with TaskManager():
            rhs = LinearForm(InnerProduct(g, v2) * dxq).Assemble()
            x = A.mat.CreateColVector()
            x.data = Ainv * rhs.vec
        ref = x.FV().NumPy().copy()
        n3 = Integrate(InnerProduct(g, g) * dxq, mesh_model)
        assert np.abs(U[:, k] - ref).max() < 1e-10 * np.abs(ref).max()
        assert abs(err[k] - np.sqrt(max(0.0, 1 - ref @ (A2 @ ref) / n3))) < 1e-9
    assert np.all(err > 0) and np.all(err < 0.2) and np.all(rq > 0)


def test_project_to_model_unit_norm_shortcut_gives_the_same_projection():
    from src.data_gen import field_labels as fl
    mesh, lab = _solved(0.02)
    a = fl.project_to_model(mesh, {"fes": lab["fes"], "gfs": lab["gfs"]}, 1)
    b = fl.project_to_model(mesh, {"fes": lab["fes"], "gfs": lab["gfs"]}, 1, unit_norm=True)
    assert np.abs(a[0] - b[0]).max() < 1e-14 * np.abs(a[0]).max()
    assert np.abs(a[1] - b[1]).max() < 1e-7            # sqrt(1 - n2): the rounding of n3 = 1 ± 1e-15 is amplified


def test_kp_from_the_gradient_form_equals_the_galerkin_product():
    """solve_modes assembles Kp = ∇φ·∇ψ on H1(p+1) instead of the sparse product GᵀMG (4–6 s per solve)."""
    from ngsolve import BilinearForm, HCurl, TaskManager, grad
    from src.data_gen import highorder as ho
    mesh = _pillbox(0.02)
    fes = HCurl(mesh, order=3, dirichlet="wall")
    u, v = fes.TnT()
    dx = ho.volume_measure(3)
    with TaskManager():
        M = BilinearForm(u * v * dx, symmetric=True).Assemble()
        G, h1 = fes.CreateGradient()
        phi, psi = h1.TnT()
        Kp = BilinearForm(grad(phi) * grad(psi) * dx, symmetric=True).Assemble()
    Gs, Ms = ho._tosp(G), ho._tosp(M.mat)
    ref = (Gs.T @ Ms @ Gs).tocsr()
    free = np.where(np.asarray(list(h1.FreeDofs()), dtype=bool))[0]
    a, b = ref[free][:, free], ho._tosp(Kp.mat)[free][:, free]
    assert abs(a - b).max() < 1e-12 * abs(a).max()


def test_too_many_tets_is_rejected_before_curving_and_not_retried_with_a_finer_mesh(tmp_path, monkeypatch):
    from netgen.occ import Axes, Cylinder, Z
    from src.data_gen import field_labels as fl
    from src.data_gen import highorder as ho
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    brep = str(tmp_path / "s.brep")
    cyl.WriteBrep(brep)
    with pytest.raises(ho.TooManyElements, match="max_elements=50"):
        ho.netgen_mesh(brep, 0.02, curve=3, max_elements=50, split=False)
    mesh = ho.netgen_mesh(brep, 0.02, curve=3, max_elements=10 ** 6, split=False)
    assert mesh.ne > 50
    calls = []

    def too_fine(cad_path, h, deform, s):
        calls.append(s["curvaturesafety"])
        raise ho.TooManyElements("too many")

    monkeypatch.setattr(fl, "curved_mesh", too_fine)
    with pytest.raises(ho.TooManyElements):
        fl.label_sample(brep, 0.02, "pillbox", 2, settings={"adapt_cs": 1.5, "curvaturesafety": 2.0})
    assert calls == [1.5]                                   # cs 2 only has more tets
