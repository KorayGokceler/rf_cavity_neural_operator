"""Learned-field model (option B): field_operators + field_collate + EigenspaceOperatorField on a
small curved pillbox (NGSolve)."""
import numpy as np
import pytest
import torch

pytest.importorskip("ngsolve")
from netgen.occ import Axes, Cylinder, OCCGeometry, Z  # noqa: E402
from ngsolve import Mesh  # noqa: E402

from src.data.dataset_converter_3d import normalise  # noqa: E402
from src.data.field_dataset import field_collate, field_item  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.models.eigenspace_operator_field import EigenspaceOperatorField  # noqa: E402
from src.models.hcurl import hcurl_grams  # noqa: E402
from src.training.lightning_module import span_residual  # noqa: E402

K = 3


@pytest.fixture(scope="module")
def sample():
    geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
    geo.faces.name = "wall"
    mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.05))
    mesh.Curve(2)
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)
    ops = ho.field_operators(mesh, order=2, scale=scale, center=center)
    r = ho.solve_modes(mesh, K, order=2)
    Y = np.column_stack([g.vec.FV().NumPy() for g in r["gfs"]])
    return mesh, ops, r, Y


def test_field_operators_reproduce_interpolation_and_spectrum(sample):
    mesh, ops, r, _ = sample
    s, c = ops["scale"], ops["center"]
    # a field sampled at the quadrature points maps to (nearly) its HCurl interpolant
    from ngsolve import CF, GridFunction, HCurl, sin, x, y, z
    f = lambda X: np.stack([np.sin(2 * X[..., 1]) * X[..., 2], X[..., 0] ** 2, X[..., 0] * X[..., 1]], -1)  # noqa: E731
    u = ops["C"] @ np.einsum("tlq,tqc->tcl", ops["P"], f(ops["qx"])).reshape(-1)
    xi = [(x - c[0]) / s, (y - c[1]) / s, (z - c[2]) / s]
    g = GridFunction(HCurl(mesh, order=2))
    g.Set(CF((sin(2 * xi[1]) * xi[2], xi[0] ** 2, xi[0] * xi[1])))
    ref = g.vec.FV().NumPy()
    assert np.linalg.norm(u - ref) / np.linalg.norm(ref) < 0.05
    # the normalised operators carry the eigenvalues λ_ξ = λ_x s²; Kp = GᵀMG exactly
    assert abs(ops["Kp"] - (ops["G"].T @ ops["M"] @ ops["G"])).max() < 1e-12 * abs(ops["Kp"]).max()
    Y = np.column_stack([gf.vec.FV().NumPy() for gf in r["gfs"]])
    rq = np.einsum("ik,ik->k", Y, ops["K"] @ Y) / np.einsum("ik,ik->k", Y, ops["M"] @ Y)
    np.testing.assert_allclose(rq, r["lam"] * s ** 2, rtol=1e-9)
    assert np.abs(Y[ops["bnd"]]).max() < 1e-12                       # PEC: wall DOFs of the labels are 0


def test_field_model_ritz_is_an_upper_bound_and_learns(sample):
    _, ops, r, Y = sample
    f_ghz = r["f_hz"] / 1e9
    batch = field_collate([field_item(ops, Y, f_ghz), field_item(ops, Y, f_ghz)])
    torch.manual_seed(0)
    m = 8
    model = EigenspaceOperatorField(val_dim=9, embed_dim=32, n_layers=1, n_heads=2, n_basis=m, num_field_modes=K,
                                    rff_dim=16, rff_length_scale=0.3)
    model.freq_stats = {"mean": float(f_ghz.mean()), "std": 1.0}
    lam_ref = torch.as_tensor(r["lam"] * ops["scale"] ** 2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        out = model(batch)
        assert out["basis"].shape == (2, ops["n_dof"], m)
        assert out["basis"][:, torch.from_numpy(ops["bnd"])].abs().max() == 0       # n × E = 0
        assert torch.allclose(out["basis"][0], out["basis"][1], atol=1e-5)          # same item, same basis
        assert bool((out["eigenvalues"].double() >= lam_ref * (1 - 1e-6)).all())   # Ritz ≥ discrete λ
        G_M, G_A, _ = hcurl_grams(out["basis"].double(), batch["Y_field"].double(),
                                  {"M_div": out["M_div"], "A_V": out["A_V"], "Z": out["Z"]}, batch)
        loss = span_residual(G_M, m).mean() + span_residual(G_A, m).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


def test_ritz_of_the_labels_returns_their_eigenvalues_and_mismatch_is_caught(sample):
    """The Ritz layer and the labels live in the same space: the labels as basis give back exactly
    their eigenvalues; labels from a differently curved mesh (same DOF count) are rejected."""
    from src.data.field_dataset import check_labels
    from src.models.hcurl import hcurl_ritz
    mesh, ops, r, Y = sample
    batch = field_collate([field_item(ops, Y, r["f_hz"] / 1e9)])
    lam, _, _ = hcurl_ritz(torch.as_tensor(Y)[None].double(), batch, K, mass_ridge=1e-14, drop_tol=1e-9,
                           tol=1e-12, maxiter=5000)
    np.testing.assert_allclose(lam[0].numpy(), r["lam"] * ops["scale"] ** 2, rtol=1e-10)
    geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
    geo.faces.name = "wall"
    m3 = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.05))
    m3.Curve(3)                                                    # the sample mesh is curved to order 2
    r3 = ho.solve_modes(m3, K, order=2)
    Y3 = np.column_stack([g.vec.FV().NumPy() for g in r3["gfs"]])
    assert Y3.shape == Y.shape
    with pytest.raises(ValueError, match="not eigenvectors"):
        check_labels(ops, Y3, r3["f_hz"] / 1e9)
