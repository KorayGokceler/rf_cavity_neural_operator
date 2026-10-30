"""Drawing the HCurl(p2) field directly (src/viz/hcurl_display.py, docs/26 "Display pipeline"): the field against
the analytic pillbox TM010 / box TE101, the energy scale against NGSolve's Integrate, the P2 / sub-tet mesh, the
curved wall lattice, deformed samples, degenerate-pair alignment in the p2 space, and the exact /plane, /volume."""
import base64
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("ngsolve")
from scipy.constants import epsilon_0 as EPS0                                # noqa: E402
from scipy.special import j0, j1, jn_zeros                                   # noqa: E402

from src.data_gen import highorder as ho                                     # noqa: E402
from src.viz import hcurl_display as hd                                      # noqa: E402
from tests.test_field_adaptive import L, R, _pillbox                         # noqa: E402

X01 = float(jn_zeros(0, 1)[0])
C0 = 299792458.0


def _random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.uniform(0, 0.98 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-0.49 * L, 0.49 * L, n)], 1)


def _tm010(pts):
    """Exact TM010 at U = 1 J: (E, H, E0, H0)."""
    e0 = np.sqrt(2.0 / (EPS0 * np.pi * R ** 2 * L)) / j1(X01)
    h0 = e0 * np.sqrt(EPS0 / (4e-7 * np.pi))
    r = np.hypot(pts[:, 0], pts[:, 1])
    arg = X01 * r / R
    E = np.zeros_like(pts)
    E[:, 2] = e0 * j0(arg)
    H = np.zeros_like(pts)
    rr = np.maximum(r, 1e-12)
    H[:, 0], H[:, 1] = -pts[:, 1] / rr * h0 * j1(arg), pts[:, 0] / rr * h0 * j1(arg)
    return E, H, e0, h0


def _err(approx, exact, peak):
    """(max, rms) of |approx − s·exact| / peak with the eigenvector sign s = ±1."""
    s = 1.0 if np.nansum(approx * exact) >= 0 else -1.0
    e = np.linalg.norm(approx - s * exact, axis=1) / peak
    return float(e.max()), float(np.sqrt(np.mean(e ** 2)))


def _p1_energy(x, sub, E):
    """½ε0 ∫|E|² of the P1 field E (per node) on the sub-tets: V/20 (Σ|e_i|² + |Σ e_i|²) per tet."""
    P = x[sub]
    vol = np.abs(np.einsum("ni,ni->n", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])) / 6
    e = E.astype(float)[sub]
    return 0.5 * EPS0 * (vol / 20 * ((e ** 2).sum((1, 2)) + (e.sum(1) ** 2).sum(1))).sum()


@pytest.fixture(scope="module")
def tm010():
    """Pillbox R = 40 mm, L = 50 mm, curve 3, p2: lowest 3 modes (TM010 first), the display mesh and fields."""
    mesh = _pillbox(0.0055)                                    # ~6.7k tets
    sol = ho.solve_modes(mesh, 3, order=2)
    U = np.stack([g.vec.FV().NumPy().copy() for g in sol["gfs"]], 1)
    f = sol["f_hz"] / 1e9
    Mp = hd.mass_matrix(mesh, 2)
    dm = hd.display_mesh(mesh)
    fld = hd.nodal_fields(mesh, dm, U, f, Mp)
    return {"mesh": mesh, "U": U, "f": f, "Mp": Mp, "dm": dm, "fld": fld}


def test_exact_field_at_points_matches_tm010(tm010):
    """The p2 field itself (exact evaluation, what /plane and /volume return) against J0 / J1 at 5k interior points."""
    c = tm010
    pts = _random_points(5000)
    E, H, inside = hd.eval_points(c["mesh"], c["U"][:, :1], c["f"][:1], c["Mp"], pts)
    assert inside.all()
    Ex, Hx, e0, h0 = _tm010(pts)
    mx, rms = _err(E[0], Ex, e0)
    assert rms <= 4e-4 and mx <= 3e-3, (mx, rms)                  # measured 1.2e-4 / 6.0e-4 at 6.7k tets
    mx, rms = _err(H[0], Hx, h0)
    assert rms <= 4e-3 and mx <= 2e-2, (mx, rms)                  # curl of p2 is P1: measured 2.0e-3 / 1.4e-2


def test_display_nodes_match_tm010_and_carry_one_joule(tm010):
    """The drawn field: P1 on the sub-tets of the P2 nodes, sampled the way the browser does (barycentric)."""
    from src.viz.nedelec import interp_vertex, locate
    c = tm010
    dm, fld = c["dm"], c["fld"]
    pts = _random_points(5000, seed=1)
    tid, bary = locate(dm["x"], dm["tets"], pts)
    ok = tid >= 0
    assert ok.mean() > 0.97                                       # the sub-tets cover the curved solid
    Ex, Hx, e0, h0 = _tm010(pts[ok])
    E = interp_vertex(dm["tets"], tid[ok], bary[ok], fld["E"][0].astype(float))
    H = interp_vertex(dm["tets"], tid[ok], bary[ok], fld["H"][0].astype(float))
    mx, rms = _err(E, Ex, e0)
    assert mx <= 1e-2 and rms <= 2e-3, (mx, rms)                  # measured 5.5e-3 / 1.5e-3 at 6.7k tets
    mx, rms = _err(H, Hx, h0)
    assert mx <= 2e-2 and rms <= 4e-3, (mx, rms)                  # measured 9.1e-3 / 1.9e-3
    # energy: the scale is exact (mass matrix = NGSolve Integrate, below); the drawn P1 field loses a little
    # of it (interpolating a concave peak): -0.5 % at 6.7k tets, -1.3 % at 1.8k
    assert _p1_energy(dm["x"], dm["tets"], fld["E"][0]) == pytest.approx(1.0, abs=1e-2)


def test_energy_scale_matches_integrate(tm010):
    """U = ½ε0 ∫|E|² computed from the DOFs and the physical mass matrix is NGSolve's Integrate of the field (same
    rule: 1e-14; a higher rule: the curved-element quadrature error, ~5e-8), and the field operators' M is this / scale."""
    from ngsolve import GridFunction, HCurl, InnerProduct, Integrate
    c = tm010
    g = GridFunction(HCurl(c["mesh"], order=2, dirichlet="wall"))
    u0 = hd.field_energy(c["Mp"], c["U"])
    for k in range(3):
        g.vec.FV().NumPy()[:] = c["U"][:, k]
        exact8 = 0.5 * EPS0 * Integrate(InnerProduct(g, g), c["mesh"], order=8)
        exact12 = 0.5 * EPS0 * Integrate(InnerProduct(g, g), c["mesh"], order=12)
        assert u0[k] == pytest.approx(exact8, rel=1e-12)
        assert u0[k] == pytest.approx(exact12, rel=1e-6)
    a = hd.energy_scale(c["Mp"], c["U"])
    assert hd.field_energy(c["Mp"], c["U"] * a[None, :]) == pytest.approx(1.0, abs=1e-12)
    # the learned-field operators' M is the physical M / scale (highorder.field_operators): the service uses M·scale
    from src.data_gen import field_labels as fl
    from src.data.sparse import to_csr
    center, scale = fl.model_normalisation(c["mesh"])
    ops = ho.field_operators(c["mesh"], order=2, scale=scale, center=center)
    n = ops["n_dof"]
    Mops = to_csr(ops["M"], (n, n)) * ops["scale"]
    assert abs(Mops - c["Mp"]).max() <= 1e-12 * abs(c["Mp"]).max()


def test_display_mesh_is_conforming_and_positive(tm010):
    from ngsolve import BND, CF, Integrate, x, y, z
    c = tm010
    dm, mesh = c["dm"], c["mesh"]
    nt, nv = mesh.ne, mesh.nv
    assert dm["tets"].shape == (8 * nt, 4) and dm["x"].shape == (nv + mesh.nedge, 3) and dm["elem_nodes"].shape == (nt, 10)
    P = dm["x"][dm["tets"]]
    vol = np.einsum("ni,ni->n", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0]) / 6
    assert vol.min() > 0 and vol.sum() == pytest.approx(dm["volume"], rel=1e-12)
    assert vol.sum() == pytest.approx(Integrate(1.0, mesh), rel=1e-3)         # the P2 geometry: -6e-4 at 6.7k tets
    # conforming: every sub-tet face is shared by two sub-tets, except the 4 sub-faces of each wall face
    sub = dm["tets"]
    faces = np.sort(np.concatenate([sub[:, [0, 1, 2]], sub[:, [0, 1, 3]], sub[:, [0, 2, 3]], sub[:, [1, 2, 3]]]), 1)
    _, count = np.unique(faces, axis=0, return_counts=True)
    assert count.max() == 2 and int((count == 1).sum()) == 4 * len(list(mesh.Elements(BND)))
    # shared nodes: every element evaluates the same physical point (continuous curved map)
    pts = np.asarray(CF((x, y, z))(dm["mips"])).reshape(-1, 3)
    spread = np.abs(pts - dm["x"][dm["elem_nodes"].reshape(-1)]).max()
    assert spread < 1e-12
    # vertex nodes are the mesh vertices, in order (reference tet (1,0,0),(0,1,0),(0,0,1),(0,0,0) = vertices 0..3)
    V = np.array([tuple(v.point) for v in mesh.vertices])
    assert np.abs(dm["x"][:nv] - V).max() < 1e-12


def test_wall_lattice_is_curved_welded_outward_and_one_sided(tm010):
    from ngsolve import CF, x, y, z
    c = tm010
    mesh = c["mesh"]
    lat = hd.surface_lattice(mesh)
    assert lat["level"] == 4 and len(lat["tris"]) == 16 * lat["n_faces"]
    P = np.asarray(CF((x, y, z))(lat["mips"])).reshape(-1, 3)[lat["rows"]]
    assert np.abs(P - lat["x"][lat["group"]]).max() < 1e-14                   # welded copies agree to rounding
    X = lat["x"]
    side = np.abs(X[:, 2]) < L / 2 - 1e-6                                     # not on an end cap
    assert np.abs(np.hypot(X[side, 0], X[side, 1]) - R).max() <= 2e-4 * R     # measured 9.6e-5 R (linear facets: 2 % R)
    # outward orientation: area vectors point away from the axis / out of the end caps
    t = X[lat["tris"]]
    area = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    mid = t.mean(1)
    on_cap = np.abs(np.abs(mid[:, 2]) - L / 2) < 1e-9
    out = np.where(on_cap[:, None], np.stack([0 * mid[:, 0], 0 * mid[:, 0], np.sign(mid[:, 2])], 1),
                   np.stack([mid[:, 0], mid[:, 1], 0 * mid[:, 2]], 1))
    assert np.all(np.einsum("ij,ij->i", area, out) > 0)
    # closed surface: area ≈ 2πR L + 2πR²
    assert 0.5 * np.linalg.norm(area, axis=1).sum() == pytest.approx(2 * np.pi * R * L + 2 * np.pi * R ** 2, rel=1e-3)
    # vertex data onto the lattice: P1 interpolation of the face vertices' coordinates gives the flat facet point
    V = np.array([tuple(v.point) for v in mesh.vertices])
    flat = np.einsum("pk,pkc->pc", lat["w3"], V[lat["vid3"]])
    assert np.abs(flat - X).max() <= 0.02 * R                                 # within the curved sagitta
    # one-sided wall values: at the end caps E is normal (E_tan / E_n at rounding level), E_n = the TM010 profile
    vE, vH = hd.surface_vectors(mesh, lat, c["U"][:, :1], c["f"][:1], c["Mp"])
    Pc = P[:]                                                                 # copy positions
    cap = (np.abs(np.abs(Pc[:, 2]) - L / 2) < 1e-9) & (np.hypot(Pc[:, 0], Pc[:, 1]) < 0.97 * R)   # not the rim copies
    e = vE[0][cap].astype(float)
    assert np.abs(e[:, :2]).max() / np.abs(e[:, 2]).max() <= 1e-6             # float32 values; BoundaryFromVolumeCF: 3e-16
    en = np.abs(e[:, 2])
    r_cap = np.hypot(Pc[cap, 0], Pc[cap, 1])
    Ex, _, e0, _ = _tm010(np.column_stack([Pc[cap, 0], Pc[cap, 1], Pc[cap, 2]]))
    assert np.abs(en - np.abs(Ex[:, 2])).max() / e0 <= 3e-3 and r_cap.max() <= R * (1 + 1e-6)
    # the wall peak equals the labels' Epk (mode_qoi uses the same one-sided path): E_pk / E_acc
    zeta, ez, L_axis = hd.axis_profile(c["mesh"], c["dm"], c["U"][:, :1], c["Mp"], (0, 0, 1), (0, 0, 0))
    q = ho.mode_qoi(mesh, [_gf(mesh, c["U"][:, 0])], c["f"][:1] * 1e9)
    dt = (zeta[-1] - zeta[0]) / (len(zeta) - 1) * 1e-3
    e_acc = q["T_transit"][0] * np.nansum(np.abs(np.nan_to_num(ez[0]))) * dt / L_axis
    epk = hd.surface_magnitudes(lat, vE).max()
    assert epk / e_acc == pytest.approx(q["Epk_Eacc"][0], rel=5e-3)


def _gf(mesh, u):
    from ngsolve import GridFunction, HCurl
    g = GridFunction(HCurl(mesh, order=2, dirichlet="wall"))
    g.vec.FV().NumPy()[:] = u
    return g


def test_axis_profile_is_the_tm010_profile(tm010):
    c = tm010
    zeta, ez, L_axis = hd.axis_profile(c["mesh"], c["dm"], c["U"][:, :1], c["Mp"], (0, 0, 1), (0, 0, 0))
    assert L_axis == pytest.approx(L, rel=1e-3) and abs(zeta.mean()) < 1e-3   # relative to the volume centroid [mm]
    inside = np.isfinite(ez[0])
    assert inside.sum() >= 395
    e0 = _tm010(np.zeros((1, 3)))[2]
    assert np.abs(np.abs(ez[0][inside]) - e0).max() / e0 < 2e-3               # E_z on the axis = E0


def test_box_te101():
    """Straight box: the lowest mode is TE101, E_y = E0 sin(πx/a) sin(πz/d)."""
    from netgen.occ import Box, OCCGeometry, Pnt
    from ngsolve import Mesh
    a, b, d = 0.08, 0.04, 0.06
    box = Box(Pnt(0, 0, 0), Pnt(a, b, d))
    box.faces.name = "wall"
    mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.009))
    mesh.Curve(3)
    sol = ho.solve_modes(mesh, 1, order=2)
    U = sol["gfs"][0].vec.FV().NumPy().copy()[:, None]
    f = sol["f_hz"][:1] / 1e9
    assert f[0] * 1e9 == pytest.approx(0.5 * C0 * np.hypot(1 / a, 1 / d), rel=2e-3)
    Mp = hd.mass_matrix(mesh, 2)
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.05 * a, 0.95 * a, 3000), rng.uniform(0.05 * b, 0.95 * b, 3000),
                           rng.uniform(0.05 * d, 0.95 * d, 3000)])
    E, H, inside = hd.eval_points(mesh, U, f, Mp, pts)
    assert inside.all()
    e0 = np.sqrt(8.0 / (EPS0 * a * b * d))
    kx, kz = np.pi / a, np.pi / d
    Ex = np.zeros_like(pts)
    Ex[:, 1] = e0 * np.sin(kx * pts[:, 0]) * np.sin(kz * pts[:, 2])
    w = 2 * np.pi * f[0] * 1e9
    h0 = e0 / (w * 4e-7 * np.pi)                                              # H = curl E / (ω μ0), sign folded
    Hx = np.zeros_like(pts)
    Hx[:, 0] = h0 * kz * np.sin(kx * pts[:, 0]) * np.cos(kz * pts[:, 2])
    Hx[:, 2] = -h0 * kx * np.cos(kx * pts[:, 0]) * np.sin(kz * pts[:, 2])
    mx, rms = _err(E[0], Ex, e0)
    assert mx <= 1e-2 and rms <= 2e-3, (mx, rms)
    hpk = h0 * np.hypot(kx, kz)
    mx, rms = _err(H[0], Hx, hpk)
    assert mx <= 5e-2 and rms <= 1e-2, (mx, rms)
    dm = hd.display_mesh(mesh)
    fld = hd.nodal_fields(mesh, dm, U, f, Mp)
    ok = np.all((dm["x"] > 1e-9) & (dm["x"] < [a - 1e-9, b - 1e-9, d - 1e-9]), 1)
    xn = dm["x"][ok]
    En = np.zeros_like(xn)
    En[:, 1] = e0 * np.sin(kx * xn[:, 0]) * np.sin(kz * xn[:, 2])
    assert _err(fld["E"][0][ok].astype(float), En, e0)[0] <= 1e-2
    assert dm["volume"] == pytest.approx(a * b * d, rel=1e-12)                # straight box: the sub-tets fill it exactly


def test_deformed_sample_is_drawn_deformed():
    """SetDeformation does not move mesh.vertices: the P2 nodes and the wall lattice come from the mapped points."""
    from ngsolve import VOL, HCurl, Integrate, IntegrationRule
    mesh0 = _pillbox(0.012)
    mesh = _pillbox(0.012)
    rng = np.random.default_rng(5)
    om, amp, ph = rng.uniform(40, 90, (3, 3)), rng.uniform(-1, 1, (3, 3)) * 1.5e-3, rng.uniform(0, 6, 3)
    ho.set_deformation(mesh, om, amp, ph, order=3)
    dm0, dm = hd.display_mesh(mesh0), hd.display_mesh(mesh)
    shift = np.linalg.norm(dm["x"] - dm0["x"], axis=1)
    assert 5e-4 < shift.max() < 6e-3                                          # drawn deformed (|δ| ≲ 3 mm)
    V = np.array([tuple(v.point) for v in mesh.vertices])
    assert np.abs(V - np.array([tuple(v.point) for v in mesh0.vertices])).max() == 0      # the vertices did not move
    r0, r1 = dm0["volume"] / Integrate(1.0, mesh0) - 1, dm["volume"] / Integrate(1.0, mesh) - 1
    assert abs(r1) < 5e-3 and abs(r1 - r0) < 1e-3                              # P2 sub-tets fill the deformed solid as well
    P = dm["x"][dm["tets"]]
    assert (np.einsum("ni,ni->n", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0]) > 0).all()
    lat = hd.surface_lattice(mesh)
    # the deformed wall lattice lies on the deformed solid: its points are inside the display mesh's bbox and the
    # lattice moved like the nodes did
    lat0 = hd.surface_lattice(mesh0)
    assert np.abs(lat["x"].max(0) - dm["x"].max(0)).max() < 5e-4 and np.abs(lat["x"] - lat0["x"]).max() > 5e-4
    # exact evaluation on a deformed sample: NGSolve's point search ignores the deformation, hd.locate looks for the
    # preimage; the value at the element-local node (mips) equals the value at the same physical point (1 % inside
    # the element: the normal component jumps across element faces, so not exactly at the node)
    fes = HCurl(mesh, order=2, dirichlet="wall")
    u = rng.standard_normal(fes.ndof)
    u[~np.asarray(list(fes.FreeDofs()), bool)] = 0
    g = _gf(mesh, u)
    el = rng.choice(mesh.ne, 150, replace=False)
    nodes = dm["x"][dm["elem_nodes"][el]]                                      # [150, 10, 3]
    inner = 0.99 * nodes + 0.01 * nodes[:, :4].mean(1, keepdims=True)
    loc = hd.locate(mesh, inner.reshape(-1, 3), (om, amp, ph))
    assert loc[1].all()                                                        # (the plain search loses ~26 % of them)
    assert not ho.locate(mesh, inner.reshape(-1, 3))[1].all()
    val_loc = ho.evaluate(g, loc).reshape(150, 10, 3)
    mips = mesh.MapToAllElements(IntegrationRule([tuple(p) for p in hd.REF_P2], [0.1] * 10), VOL)
    val_map = np.asarray(g(mips)).reshape(mesh.ne, 10, 3)[el]
    err = np.abs(val_loc - val_map).max(2) / np.abs(val_map).max()
    assert np.median(err) < 5e-3 and np.quantile(err, 0.95) < 2e-2             # (the 1 % shift of a random p2 field)


def test_degenerate_cluster_is_aligned_in_p2_space_with_the_real_mass_matrix():
    """A cube's TE101 / TE011 / TM110 triple: any rotation of the triple inside its span is a valid eigenbasis; the
    model-vs-FE alignment (service.model.align) must work on p2 DOFs with the true HCurl mass matrix."""
    from netgen.occ import Box, OCCGeometry, Pnt
    from ngsolve import Mesh

    from src.service.model import align, degenerate_mask
    a = 0.05
    cube = Box(Pnt(0, 0, 0), Pnt(a, a, a))
    cube.faces.name = "wall"
    mesh = Mesh(OCCGeometry(cube).GenerateMesh(maxh=0.012))
    mesh.Curve(3)
    sol = ho.solve_modes(mesh, 4, order=2)
    T = np.stack([g.vec.FV().NumPy().copy() for g in sol["gfs"]], 1)
    f = sol["f_hz"] / 1e9
    assert f[3] > 1.2 * f[2] and np.ptp(f[:3]) / f[0] < 0.02                  # a triple, then TE111-like modes
    Mp = hd.mass_matrix(mesh, 2)
    rng = np.random.default_rng(7)
    Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]                          # a random rotation inside the triple
    Mt = T[:, :3] / np.sqrt(np.einsum("ik,ik->k", T[:, :3], Mp @ T[:, :3]))   # M-normalised
    P = np.column_stack([Mt @ Q * np.array([3.0, -0.5, 1.0]), T[:, 3] * -2.0])
    fc = np.where(degenerate_mask(f, 0.03), f[:3].mean(), f)                  # the triple is one cluster
    out, rl = align(P, T, Mp, fc, rel=0.03)
    assert rl.max() < 1e-8, rl                                                # the aligned span reproduces the FE modes
    assert np.abs(out - T).max() / np.abs(T).max() < 1e-8
    # the M-norm matters: the identity metric aligns a different (wrong) combination
    bad, rl_bad = align(P, T, np.eye(len(T)), fc, rel=0.03)
    assert rl_bad[:3].max() > 1e-3
    assert bad.shape == T.shape


# ───────────────────────── API: exact /plane and /volume ─────────────────────────

@pytest.fixture(scope="module")
def api_case(tmp_path_factory):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    gmsh = pytest.importorskip("gmsh")
    from fastapi.testclient import TestClient

    from src.service import api as A
    from src.service.model import ModelService
    d = tmp_path_factory.mktemp("hd_api")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addCylinder(0, 0, -50, 0, 0, 100, 100)         # mm, R = L = 100
    gmsh.model.occ.synchronize()
    gmsh.write(str(d / "pillbox.step"))
    gmsh.clear()
    client = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    with open(d / "pillbox.step", "rb") as fh:
        g = client.post("/api/geometries", files={"file": ("cavity.step", fh)},
                        data={"unit": "mm", "mesh_size": "0.3"}).json()
    p = client.post("/api/predictions", json={"geometry_id": g["id"]}).json()
    return client, g, p


def _f32(s):
    return np.frombuffer(base64.b64decode(s), np.float32)


def _oracle(client, gid, mode):
    """E, H functions of the predicted mode from NGSolve alone (GridFunction + Integrate energy + Faraday)."""
    from ngsolve import GridFunction, HCurl, InnerProduct, Integrate, curl

    from src.service.model import forward
    svc = client.app.state.service
    mesh, ops = svc._ops[gid]
    U, f, _, _ = forward(svc.lm, ops, svc.feature_indices, "", "cpu")
    g = GridFunction(HCurl(mesh, order=2, dirichlet="wall"))
    g.vec.FV().NumPy()[:] = U[:, mode]
    a = 1.0 / np.sqrt(0.5 * EPS0 * Integrate(InnerProduct(g, g), mesh, order=8))
    return mesh, g, a, 1.0 / (2 * np.pi * f[mode] * 1e9 * 4e-7 * np.pi), curl(g)


def test_api_plane_and_volume_are_exact_evaluations(api_case):
    client, g, p = api_case
    mesh, gf, a, fa, cg = _oracle(client, g["id"], 1)
    q = client.get(f"/api/predictions/{p['id']}/plane", params={"mode": 1, "axis": "x", "pos": 10.0, "res": 41}).json()
    P = _f32(q["points_b64"]).reshape(-1, 3).astype(float) * 1e-3
    assert np.abs(P[:, 0] - 0.010).max() < 1e-6
    E, H = _f32(q["E_b64"]).reshape(-1, 3), _f32(q["H_b64"]).reshape(-1, 3)
    loc = ho.locate(mesh, P)
    assert loc[1].all()
    Er = a * ho.evaluate(gf, loc)
    Hr = -a * fa * ho.evaluate(cg, loc)
    assert np.abs(E - Er).max() <= 2e-6 * np.abs(Er).max() and np.abs(H - Hr).max() <= 2e-6 * np.abs(Hr).max()   # float32
    v = client.get(f"/api/predictions/{p['id']}/volume", params={"mode": 1, "n": 6000}).json()
    nx, ny, nz = v["dims"]
    inside = np.frombuffer(base64.b64decode(v["inside_b64"]), np.uint8).astype(bool)
    E = _f32(v["E_b64"]).reshape(-1, 3)
    o, h = np.asarray(v["origin_mm"]) * 1e-3, v["spacing_mm"] * 1e-3
    ii = np.arange(nx * ny * nz)
    pts = np.stack([o[0] + h * (ii % nx), o[1] + h * ((ii // nx) % ny), o[2] + h * (ii // (nx * ny))], 1)
    loc = ho.locate(mesh, pts)
    assert np.array_equal(loc[1], inside) and 0 < inside.sum() < len(inside)
    Er = a * ho.evaluate(gf, loc)
    assert np.abs(E - Er).max() <= 2e-6 * np.abs(Er).max() and np.all(E[~inside] == 0)


def test_api_fields_schema_gzip_and_curved_wall(api_case):
    client, g, p = api_case
    r = client.get(f"/api/predictions/{p['id']}/fields", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
    fb = r.json()
    assert {"n_modes", "n_nodes", "n_cells", "n_tets", "n_surface", "f_GHz", "E_max", "H_max", "nodes_b64", "tets_b64",
            "E_b64", "H_b64", "surf_E_b64", "surf_H_b64", "axis"} <= set(fb)
    for k in ("E_b64", "H_b64", "surf_E_b64", "surf_H_b64"):
        assert np.isfinite(np.frombuffer(base64.b64decode(fb[k]), np.float16).astype(float)).all()
    assert fb["n_cells"] == 8 * fb["n_tets"] and fb["n_surface"] == g["surface"]["n_points"]
    tets = np.frombuffer(base64.b64decode(fb["tets_b64"]), np.uint32)
    assert tets.max() == fb["n_nodes"] - 1 and tets.min() == 0
    # the geometry's surface and the check's volume / bbox come from the curved solid
    wall = _f32(g["surface"]["points_b64"]).reshape(-1, 3)
    assert np.hypot(wall[:, 0], wall[:, 1]).max() == pytest.approx(100, abs=0.5)
    assert g["check"]["volume_cm3"] == pytest.approx(np.pi * 100 ** 2 * 100 * 1e-3, rel=5e-3)      # π R² L [cm³]
    assert g["check"]["bbox_mm"][1][0] == pytest.approx(100, abs=0.5)
