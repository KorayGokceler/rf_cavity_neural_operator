"""Kernel projection and Rayleigh–Ritz of the H(curl) eigenspace operator (src/models/hcurl.py) on
HCurl(p2) spaces of curved meshes (field_operators): Ritz exactness with gradient columns, gradient
contamination, the Kp solve and its gradient, the PEC / potential contract on a box, padding
invariance of the field model, cluster bookkeeping of the losses and the checkpoint guard."""
import math
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("ngsolve")
from netgen.occ import Axes, Box, Cylinder, OCCGeometry, Pnt, Z  # noqa: E402
from ngsolve import Mesh  # noqa: E402

from src.data.mesh_features import normalise  # noqa: E402
from src.data.field_dataset import field_collate, field_item  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.models.eigenspace_operator_field import EigenspaceOperatorField  # noqa: E402
from src.models.hcurl import hcurl_grams, hcurl_ritz, mode_rel_l2, project_basis, spmm  # noqa: E402
from src.training.lightning_module import CavityLightning, ritz_compliance, span_residual  # noqa: E402

K_MODES = 3                    # pillbox: TM010 + the TE111 pair
BOX = (1.0, 0.8, 0.6)


def _curved(geo, maxh, k):
    """(field_operators dict, solve_modes result, labels Y [n, k]) of a p2 space on a curved mesh."""
    geo.faces.name = "wall"
    mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=maxh))
    mesh.Curve(2)
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)
    ops = ho.field_operators(mesh, order=2, scale=scale, center=center)
    r = ho.solve_modes(mesh, k, order=2)
    return ops, r, np.column_stack([g.vec.FV().NumPy() for g in r["gfs"]])


def _item(ops, r, Y):
    it = field_item(ops, Y, r["f_hz"] / 1e9)
    it["FreqNext"] = torch.tensor(float("nan"))
    it["geom_id"] = torch.tensor([0])
    return it


@pytest.fixture(scope="module")
def pill():
    """(item, single-item batch, 4 gradient columns G·φ [1, n, 4], discrete λ_h [K] normalised)."""
    ops, r, Y = _curved(Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1), 0.05, K_MODES)
    item = _item(ops, r, Y)
    phi = np.random.default_rng(0).standard_normal((ops["n_pot"], 4))
    return item, field_collate([item]), torch.from_numpy(ops["G"] @ phi)[None], \
        torch.as_tensor(r["lam"] * ops["scale"] ** 2)


@pytest.fixture(scope="module")
def box():
    ops, r, Y = _curved(Box(Pnt(0, 0, 0), Pnt(*BOX)), 0.25, 6)
    return ops, r, Y


def _box_item(box):
    """The box as a K_MODES item (batched with the pillbox)."""
    ops, r, Y = box
    return _item(ops, dict(r, f_hz=r["f_hz"][:K_MODES]), Y[:, :K_MODES])


def _plain_ritz(V, batch, K):
    """Ritz with the UNPROJECTED mass Gram VᵀMV (no kernel projection)."""
    p = project_basis(V, batch, 1e-12)
    L = torch.linalg.cholesky(p["M_V"])
    Li = torch.linalg.inv(L)
    vals = torch.linalg.eigvalsh(Li @ p["A_V"] @ Li.transpose(-1, -2))
    return vals[:, :K]


def test_projected_ritz_exact_with_gradient_columns(pill):
    item, batch, Gphi, lam = pill
    T = item["Y_field"].double()[None]
    mix = torch.randn(K_MODES, K_MODES, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    V = torch.cat([T @ mix, Gphi], dim=-1)                              # exact span ⊕ 4 gradients
    theta, fields, proj = hcurl_ritz(V, batch, K_MODES, tol=1e-12)
    assert torch.allclose(theta[0], lam, rtol=1e-9)
    G_M, G_A, MT = hcurl_grams(V, T, proj, batch)
    m = V.shape[-1]
    assert span_residual(G_M, m).max() < 1e-6 and span_residual(G_A, m).max() < 1e-6   # (dead columns ≈ 1e-16 noise)
    assert mode_rel_l2(fields[0], T[0], MT[0], [[0], [1, 2]]).max() < 1e-6            # TE111 pair: subspace error


def test_gradient_contamination_breaks_only_the_unprojected_gram(pill):
    item, batch, Gphi, lam = pill
    T = item["Y_field"].double()[None]
    T = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    Gn = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()
    V = T + 0.3 * Gn[..., :K_MODES]                                      # 8 % gradient mass per column
    plain = _plain_ritz(V, batch, K_MODES)[0]
    assert (plain / lam - 1).min() < -0.02                             # Ritz values pulled ~8 % low
    theta, _, proj = hcurl_ritz(V, batch, K_MODES, tol=1e-12)
    assert torch.allclose(theta[0], lam, rtol=1e-9)
    # mass-norm span residual: projected ≈ 0, unprojected > 0
    G_M, _, _ = hcurl_grams(V, T, proj, batch)
    G_plain = G_M.clone()
    G_plain[:, :K_MODES, :K_MODES] = proj["M_V"]
    assert span_residual(G_M, K_MODES).max() < 1e-8 < 1e-3 < span_residual(G_plain, K_MODES).mean()   # (1e-9 = ridge bias)


def test_pure_gradient_column_gives_no_zero_ritz_value(pill):
    item, batch, Gphi, lam = pill
    T = item["Y_field"].double()[None]
    V = torch.cat([T, Gphi[..., :1]], dim=-1)
    assert _plain_ritz(V, batch, 1)[0, 0] < 1e-6 * lam[0]             # unprojected: θ = 0
    theta, _, proj = hcurl_ritz(V, batch, K_MODES + 1, tol=1e-12)
    assert torch.allclose(theta[0, :K_MODES], lam, rtol=1e-9)          # gradient direction dropped (θ = ∞)
    assert theta[0, -1] > 100 * lam[-1]
    # compliance tr(A⁻¹M) with an almost-gradient column (gradient + 1e-3·mode):
    # unprojected ~ 1/(1e-6·λ) (the unbounded reward), projected = the clean span's value
    Gn = Gphi[..., :1] / (Gphi[..., :1] * spmm(batch["M"], Gphi[..., :1])).sum(1, keepdim=True).sqrt()
    Tn = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    V = torch.cat([Tn[..., :-1], Gn + 1e-3 * Tn[..., -1:]], dim=-1)
    proj = project_basis(V, batch, 1e-12)
    clean = (1.0 / lam).sum()                                          # = tr(A⁻¹M) of span(T)
    assert ritz_compliance(proj["M_V"], proj["A_V"]) > 1e3 * clean
    assert torch.allclose(ritz_compliance(proj["M_div"], proj["A_V"]), clean, rtol=1e-5)


def test_kp_solve_gradient(pill):
    """Directional derivative of a loss through M_div (forward + backward Kp solves) against central
    finite differences."""
    _, batch, _, _ = pill
    g = torch.Generator().manual_seed(0)
    V = torch.randn(1, batch["DofMask"].shape[1], 3, dtype=torch.float64, generator=g, requires_grad=True)
    W = torch.randn(1, 3, 3, dtype=torch.float64, generator=g)
    f = lambda v: (project_basis(v, batch, 1e-13)["M_div"] * W).sum()    # noqa: E731
    f(V).backward()
    for _ in range(2):
        dV = torch.randn(V.shape, dtype=torch.float64, generator=g)
        fd = (f(V.detach() + 1e-6 * dV) - f(V.detach() - 1e-6 * dV)) / 2e-6
        assert torch.allclose((V.grad * dV).sum(), fd, rtol=1e-6)


def test_all_gradient_sample_gets_theta_above_spectrum(pill):
    """docs/21 B1: a sample whose span is pure gradient keeps no direction; its 'θ = ∞' must stay far
    above the physical spectrum (was 1.0)."""
    _, batch, Gphi, lam = pill
    theta, _, _ = hcurl_ritz(Gphi[..., :3], batch, 2, tol=1e-12)
    assert (theta > 100 * lam[-1]).all()
    assert torch.isfinite(theta).all()


def test_projection_forces_float64_and_warns_at_maxiter(pill):
    """docs/21 B3/B4: float32 input gives the float64 result; CG stopping at maxiter warns instead of
    returning silently."""
    item, batch, _, _ = pill
    V = item["Y_field"][None]
    p32 = project_basis(V.float(), batch, 1e-10)
    p64 = project_basis(V.double(), batch, 1e-10)
    assert p32["M_div"].dtype == torch.float64
    torch.testing.assert_close(p32["M_div"], p64["M_div"], rtol=1e-6, atol=1e-8)
    with pytest.warns(UserWarning, match="maxiter"):
        project_basis(V.double() + 0.1 * torch.randn_like(V.double()), batch, 1e-14, maxiter=2)


def test_box_spectrum_and_contract(box):
    """Curved-mesh PEC box: frequencies → the analytic spectrum, wall rows 0, G with zero wall rows,
    Kp = GᵀMG SPD, eigenvectors M-orthogonal to span(G)."""
    import scipy.sparse as sp

    from src.data_gen.dataset_generator_3d import box_spectrum
    ops, r, Y = box
    assert np.abs(r["lam"] / box_spectrum(*BOX)[:6] - 1).max() < 0.03
    bnd, M, G, Kp = ops["bnd"], ops["M"], ops["G"], ops["Kp"]
    assert bnd.sum() > 0 and np.abs(Y[bnd]).max() == 0
    assert abs(G[bnd]).sum() == 0 and G.shape == (ops["n_dof"], ops["n_pot"])
    assert abs(Kp - (G.T @ M @ G)).max() < 1e-12 * abs(Kp).max() and isinstance(Kp, sp.csr_matrix)
    assert np.linalg.eigvalsh(Kp.toarray()).min() > 0
    MY = M @ Y
    assert np.abs(G.T @ MY).max() < 1e-8 * np.abs(MY).max()


def test_kp_solve_is_plain_spd_pcg(pill, box):
    """Kp Z = GᵀMV on the valid block of each sample, Z = 0 on the padded potentials."""
    from src.models.hcurl import _pcg
    batch = field_collate([pill[0], _box_item(box)])
    V = torch.randn(2, batch["DofMask"].shape[1], 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    V = V * (batch["DofMask"] & ~batch["BndDof"]).unsqueeze(-1)
    Bm = spmm(batch["Gt"], spmm(batch["M"], V))
    diag = batch["Kp_diag"]
    dinv = torch.where(diag > 0, 1.0 / diag.clamp(min=1e-300), torch.zeros_like(diag)).unsqueeze(-1)
    valid = diag > 0
    assert 0 < valid[0].sum() < diag.shape[1] and bool(valid[1].all())   # the pillbox is padded
    Z, _ = _pcg(batch["Kp"], dinv, Bm, 1e-12, 3000)
    assert (Z[~valid] == 0).all()
    assert (spmm(batch["Kp"], Z) - Bm).abs().max() < 1e-9 * Bm.abs().max()


def _model(**kw):
    torch.manual_seed(0)
    m = EigenspaceOperatorField(9, embed_dim=16, n_layers=1, n_heads=2, n_basis=8, num_field_modes=K_MODES,
                                rff_dim=8, kp_tol=1e-12, **kw).eval()
    m.freq_stats = {"mean": 0.0, "std": 1.0}
    return m


def test_padding_invariance(pill, box):
    a, b = pill[0], _box_item(box)
    na, nb = a["M"].shape[0], b["M"].shape[0]
    assert nb > na and b["X"].shape[0] > a["X"].shape[0]                 # a gets padded
    model = _model()
    with torch.no_grad():
        o1, o2 = model(field_collate([a])), model(field_collate([b, a, b]))
    assert torch.allclose(o2["basis"][1, :na], o1["basis"][0], atol=1e-5)
    assert (o2["basis"][1, na:] == 0).all()
    assert torch.allclose(o2["eigenvalues"][1], o1["eigenvalues"][0], rtol=1e-5)
    F1, F2 = o1["field"][0], o2["field"][1, :na]
    assert torch.allclose(F2 * torch.sign((F1 * F2).sum(0)), F1, atol=1e-4)
    bnd = a["BndDof"]                                                    # wall rows exactly 0, padded too
    assert bnd.any() and (o1["field"][0, bnd] == 0).all() and (o2["field"][1, :na][bnd] == 0).all()
    assert (o2["field"][1, na:] == 0).all()
    with pytest.raises(ValueError, match="BndDof"):                       # a batch without the wall mask
        model({k: v for k, v in field_collate([a]).items() if k != "BndDof"})


def _module(**kw):
    torch.manual_seed(0)
    lm = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K_MODES, rff_dim=8,
                         eigenspace_kwargs={"n_layers": 1}, near_deg_rel_threshold=0.02, **kw)
    lm.freq_stats = {"mean": 0.0, "std": 1.0}
    return lm


def test_loss_and_finite_gradients(pill, box):
    batch = field_collate([pill[0], _box_item(box)])
    lm = _module(freq_weight=0.1)
    loss, preds, targets = lm._compute_loss_3d(batch, "train")
    assert torch.isfinite(loss) and preds.shape == targets.shape
    loss.backward()
    grads = [p.grad for p in lm.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_split_cluster_is_excluded():
    lm = _module()
    lm.near_deg_rel_threshold = 0.01
    batch = {"Y_freq": torch.tensor([[1.0, 2.0, 2.001, 3.0, 4.0, 4.002]]), "FreqNext": torch.tensor([4.003])}
    assert lm._clusters_3d(batch, 0, 6) == ([[0], [1, 2], [3]], [4, 5])
    assert lm._clusters_3d(batch, 0, 2) == ([[0]], [1])


def _all_split(batch, b):
    """Put every stored mode of sample b (and FreqNext) into one near-degenerate cluster."""
    f0 = batch["Y_freq"][b, 0].clone()
    batch["Y_freq"][b] = f0 + 1e-4 * torch.arange(batch["Y_freq"].shape[1])
    batch["FreqNext"][b] = f0 + 1e-4 * batch["Y_freq"].shape[1]


def _logged(lm, batch, prefix="val"):
    logs = {}
    lm.log = lambda name, value, **kw: logs.__setitem__(name, (float(value), kw.get("batch_size")))
    with torch.no_grad():
        lm._compute_loss_3d(batch, prefix)
    return logs


def test_split_clusters_do_not_poison_logged_metrics(pill):
    """Regression: a split-cluster (NaN) rel-L2 entry is excluded by weight.  Before, a batch whose
    entries were all NaN logged val/field_rel_l2 = NaN, which made the epoch mean NaN: EarlyStopping
    stopped training after one epoch and ModelCheckpoint never improved."""
    lm = _module()
    K = lm.model.num_field_modes
    batch = field_collate([pill[0], pill[0]])
    _all_split(batch, 0)
    assert lm._clusters_3d(batch, 0, K)[1] == list(range(K))
    split1 = lm._clusters_3d(batch, 1, K)[1]
    logs = _logged(lm, batch)
    v, n = logs["val/field_rel_l2"]
    assert math.isfinite(v) and n == K - len(split1)
    for k in range(K):                       # every key on every step (DDP), weight = #valid
        v, n = logs[f"val/mode_{k}_rel_l2"]
        assert math.isfinite(v) and n == int(k not in split1)
    # a batch with no valid entry: finite value with zero weight (no epoch-mean contribution)
    only = field_collate([pill[0]])
    _all_split(only, 0)
    logs = _logged(lm, only)
    assert logs["val/field_rel_l2"] == (0.0, 0)
    assert all(logs[f"val/mode_{k}_rel_l2"] == (0.0, 0) for k in range(K))


def test_removed_models_are_rejected_with_a_clear_error():
    with pytest.raises(ValueError, match="removed H formulation"):
        CavityLightning(val_dim=9, data_cfg={"field": "H"})
    with pytest.raises(ValueError, match="removed 2D model"):
        CavityLightning(val_dim=9, model_type="gnot")
    with pytest.raises(ValueError, match="eigenspace3d"):
        CavityLightning(val_dim=9, model_type="eigenspace3d")
    lm = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=10, num_field_modes=4, rff_dim=8,
                         eigenspace_kwargs={"n_layers": 1}, grid_dim=3, physics_freq=True, degeneracy_mode="hard",
                         qoi_weight=0.0, qoi_terms=("Q0",), qoi_peak_p=None, qoi_rq_floor=1e-2)
    assert lm.model_type == "field3d" and isinstance(lm.model, EigenspaceOperatorField)
    assert lm.hparams.hidden_dim == 16
    assert not any(k in lm.hparams for k in ("grid_dim", "qoi_weight", "qoi_terms"))   # legacy arguments ignored


def test_train_py_rejects_removed_options(monkeypatch):
    import train
    for override in ("model.type=eigenspace3d", "training.qoi_weight=1.0", "training.augment=true"):
        monkeypatch.setattr(sys, "argv", ["train.py", "--override", override])
        with pytest.raises(ValueError, match="eigenspace3d|removed N0 model"):
            train.main()
