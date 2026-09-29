"""3D Maxwell eigenspace operator (Whitney N0, H and E formulations): kernel
projection, Ritz exactness, orientation / padding invariance, dataset, training
wiring.  The PKL-based tests run for field ∈ {H, E} (the `pkl` fixture is
parametrized); E-only tests (wall mask, SPD Kp solve, lean rebuild, mixed
batches) follow at the end."""
import math
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("skfem")

from src.data.dataset_3d import (Maxwell3DDataset, item_from_geometry,      # noqa: E402
                                 maxwell3d_collate, to_csr)
from src.models.eigenspace_operator_3d import EigenspaceOperator3D            # noqa: E402
from src.models.hcurl import (hcurl_grams, hcurl_ritz, mode_rel_l2,           # noqa: E402
                              project_basis, spmm)
from src.models.spectral_no import _generalized_eigh                          # noqa: E402
from src.training.lightning_module import (GNOTLightning, ritz_compliance,    # noqa: E402
                                           span_residual)
from tests.maxwell3d_synth import C0, box_geometry, edge_dofs, make_dataset   # noqa: E402

K_MODES = 6


@pytest.fixture(scope="module", params=["H", "E"])
def pkl(request, tmp_path_factory):
    import pickle
    path = tmp_path_factory.mktemp("m3d") / f"synth3d_{request.param}.pkl"
    with open(path, "wb") as f:
        pickle.dump(make_dataset(n_geoms=5, n=4, n_modes=K_MODES, seed=0, field=request.param), f)
    return str(path)


STATS = {}   # field -> freq z-score stats of the synthetic PKL (set by the ds fixture)


@pytest.fixture(scope="module")
def ds(pkl):
    d = Maxwell3DDataset(pkl, split="test", train_ratio=0.0, val_ratio=0.0)   # all geometries
    STATS[d.field] = dict(d.stats)
    return d


def _lam(item, stats=None):
    """Discrete λ_h (normalised mesh): Rayleigh quotients of the stored eigenvectors
    (float32 DOFs → error O(eps32²)); consistent with the stored GHz, f = c√λ/(2π·scale)."""
    T = item["Y_field"].double().numpy()
    lam = (T * (item["K"] @ T)).sum(0) / (T * (item["M"] @ T)).sum(0)
    st = stats or STATS[item["field"]]
    f = item["Y_freq"].double().numpy() * st["std"] + st["mean"]
    assert np.allclose(C0 * np.sqrt(lam) / (2 * math.pi * float(item["Scale"])) / 1e9, f, rtol=1e-5)
    return torch.from_numpy(lam)


@pytest.fixture(scope="module")
def box(ds):
    item = ds[0]
    batch = maxwell3d_collate([item])
    G = to_csr(ds.geometry_pool[ds.active_geoms[0]]["G"])
    phi = np.random.default_rng(0).standard_normal((G.shape[1], 4))
    return item, batch, torch.from_numpy(G @ phi)[None], _lam(item)


def _plain_ritz(V, batch, K):
    """Ritz with the UNPROJECTED mass Gram VᵀMV (the 2D code, applied as is)."""
    p = project_basis(V, batch, 1e-12)
    vals, _ = _generalized_eigh(p["A_V"], p["M_V"], 1e-4)
    return vals[:, :K]


def test_projected_ritz_exact_with_gradient_columns(box):
    item, batch, Gphi, lam = box
    T = item["Y_field"].double()[None]
    mix = torch.randn(K_MODES, K_MODES, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    V = torch.cat([T @ mix, Gphi], dim=-1)                              # exact span ⊕ 4 gradients
    theta, fields, proj = hcurl_ritz(V, batch, K_MODES, tol=1e-12)
    assert torch.allclose(theta[0], lam, rtol=1e-9)
    G_M, G_A, MT = hcurl_grams(V, T, proj, batch)
    m = V.shape[-1]
    assert span_residual(G_M, m).max() < 1e-6 and span_residual(G_A, m).max() < 1e-6   # (dead columns ≈ 1e-16 noise)
    clusters = [[0], [1], [2], [3, 4], [5]] if torch.isclose(lam[3], lam[4], rtol=1e-6) else [[k] for k in range(6)]
    assert mode_rel_l2(fields[0], T[0], MT[0], clusters).max() < 1e-6


def test_gradient_contamination_breaks_only_the_unprojected_gram(box):
    item, batch, Gphi, lam = box
    T = item["Y_field"].double()[None]
    Gn = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()
    V = T + 0.3 * torch.cat([Gn, Gn[..., :2]], dim=-1)                 # 8 % gradient mass per column
    plain = _plain_ritz(V, batch, K_MODES)[0]
    assert (plain / lam - 1).min() < -0.02                             # research: Ritz values pulled 6–13 % low
    theta, _, proj = hcurl_ritz(V, batch, K_MODES, tol=1e-12)
    assert torch.allclose(theta[0], lam, rtol=1e-9)
    # mass-norm span residual: projected ≈ 0, unprojected > 0
    G_M, _, _ = hcurl_grams(V, T, proj, batch)
    G_plain = G_M.clone()
    G_plain[:, :K_MODES, :K_MODES] = proj["M_V"]
    assert span_residual(G_M, K_MODES).max() < 1e-8 < 1e-3 < span_residual(G_plain, K_MODES).mean()   # (1e-9 = ridge bias)


def test_pure_gradient_column_gives_no_zero_ritz_value(box):
    item, batch, Gphi, lam = box
    T = item["Y_field"].double()[None]
    V = torch.cat([T, Gphi[..., :1]], dim=-1)
    assert _plain_ritz(V, batch, 1)[0, 0] < 1e-6 * lam[0]             # unprojected: θ = 0
    theta, _, proj = hcurl_ritz(V, batch, K_MODES + 1, tol=1e-12)
    assert torch.allclose(theta[0, :K_MODES], lam, rtol=1e-9)          # gradient direction dropped (θ = ∞)
    assert theta[0, -1] > 100 * lam[-1]
    # compliance tr(A⁻¹M) with an almost-gradient column (gradient + 1e-3·mode):
    # unprojected ~ 1/(1e-6·λ) (the unbounded reward), projected = the clean span's value
    Gn = Gphi[..., :1] / (Gphi[..., :1] * spmm(batch["M"], Gphi[..., :1])).sum(1, keepdim=True).sqrt()
    V = torch.cat([T[..., :-1], Gn + 1e-3 * T[..., -1:]], dim=-1)
    proj = project_basis(V, batch, 1e-12)
    clean = (1.0 / lam).sum()                                          # = tr(A⁻¹M) of span(T)
    assert ritz_compliance(proj["M_V"], proj["A_V"]) > 1e3 * clean
    assert torch.allclose(ritz_compliance(proj["M_div"], proj["A_V"]), clean, rtol=1e-5)


def test_kp_solve_gradient(box):
    _, batch, _, _ = box
    """Directional derivative of a loss through M_div (forward + backward Kp solves)
    against central finite differences."""
    g = torch.Generator().manual_seed(0)
    V = torch.randn(1, batch["Edges"].shape[1], 3, dtype=torch.float64, generator=g, requires_grad=True)
    W = torch.randn(1, 3, 3, dtype=torch.float64, generator=g)
    f = lambda v: (project_basis(v, batch, 1e-13)["M_div"] * W).sum()    # noqa: E731
    f(V).backward()
    for _ in range(2):
        dV = torch.randn(V.shape, dtype=torch.float64, generator=g)
        fd = (f(V.detach() + 1e-6 * dV) - f(V.detach() - 1e-6 * dV)) / 2e-6
        assert torch.allclose((V.grad * dV).sum(), fd, rtol=1e-6)


def _flip(item, flip):
    """The same geometry with the edges in `flip` oriented high → low."""
    import scipy.sparse as sp
    s = np.ones(item["Edges"].shape[0])
    s[flip] = -1.0
    S = sp.diags(s)
    out = dict(item)
    E = item["Edges"].clone()
    E[flip] = E[flip].flip(-1)
    out.update(Edges=E, Y_field=item["Y_field"] * torch.from_numpy(s).float()[:, None],
               M=(S @ item["M"] @ S).tocsr(), K=(S @ item["K"] @ S).tocsr(), G=(S @ item["G"]).tocsr())
    return out, torch.from_numpy(s)


def _model(val_dim, stats=None, **kw):
    torch.manual_seed(0)
    m = EigenspaceOperator3D(val_dim, embed_dim=16, n_layers=1, n_heads=2, n_basis=10,
                             num_field_modes=4, rff_dim=8, kp_tol=1e-12, **kw).eval()
    m.freq_stats = dict(stats or {"mean": 0.0, "std": 1.0})
    return m


def test_edge_orientation_equivariance(ds, box):
    item = box[0]
    flip = np.arange(0, item["Edges"].shape[0], 3)
    item_f, s = _flip(item, flip)
    # targets: the canonical interpolant ∫_e v·t and the eigenvectors flip with the edge
    X = item["X"].double().numpy()
    field = lambda x: np.stack([np.sin(x[:, 1]), x[:, 0] * x[:, 2], np.cos(x[:, 0])], -1)   # noqa: E731
    assert np.allclose(edge_dofs(X, item_f["Edges"].numpy(), field), s.numpy() * edge_dofs(X, item["Edges"].numpy(), field))
    y = item_f["Y_field"][:, 0].double().numpy()
    lam0 = float(_lam(item)[0])
    free = ~item["BndEdge"].numpy() if "BndEdge" in item else slice(None)   # E: wall rows are constraints
    res = (item_f["K"] @ y - lam0 * (item_f["M"] @ y))[free]
    assert np.abs(res).max() < 1e-4 * np.abs(item_f["K"] @ y).max()
    # model: raw basis and Ritz fields flip exactly, eigenvalues unchanged
    model = _model(item["Input_funcs"].shape[-1], ds.stats)
    with torch.no_grad():
        o, of = model(maxwell3d_collate([item])), model(maxwell3d_collate([item_f]))
    sf = s.float()[None, :, None]
    assert torch.allclose(of["basis"], o["basis"] * sf, atol=1e-6)
    assert torch.allclose(of["eigenvalues"], o["eigenvalues"], rtol=1e-6)
    sign = torch.sign((of["field"] * o["field"] * sf).sum(1, keepdim=True))
    assert torch.allclose(of["field"], o["field"] * sf * sign, atol=1e-5)


def test_padding_invariance(ds):
    items = sorted((ds[i] for i in range(len(ds))), key=lambda it: it["Edges"].shape[0])
    a, b = items[0], items[-1]                                          # a gets padded
    assert b["Edges"].shape[0] > a["Edges"].shape[0] and b["X"].shape[0] > a["X"].shape[0]
    model = _model(a["Input_funcs"].shape[-1], ds.stats)
    with torch.no_grad():
        o1, o2 = model(maxwell3d_collate([a])), model(maxwell3d_collate([b, a, b]))
    ne = a["Edges"].shape[0]
    assert torch.allclose(o2["basis"][1, :ne], o1["basis"][0], atol=1e-6)
    assert (o2["basis"][1, ne:] == 0).all()
    assert torch.allclose(o2["eigenvalues"][1], o1["eigenvalues"][0], rtol=1e-6)
    F1, F2 = o1["field"][0], o2["field"][1, :ne]
    assert torch.allclose(F2 * torch.sign((F1 * F2).sum(0)), F1, atol=1e-5)
    if ds.field == "E":                                                 # wall rows exactly 0, padded too
        bnd = a["BndEdge"]
        assert bnd.any() and (o1["field"][0, bnd] == 0).all() and (o2["field"][1, :ne][bnd] == 0).all()
        assert (o2["field"][1, ne:] == 0).all()


def _module(val_dim, stats=None, **kw):
    torch.manual_seed(0)
    lm = GNOTLightning(val_dim=val_dim, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=10,
                       num_field_modes=4, rff_dim=8, model_type="eigenspace3d", physics_freq=True,
                       eigenspace_kwargs={"n_layers": 1}, near_deg_rel_threshold=0.02, **kw)
    lm.freq_stats = dict(stats or {"mean": 0.0, "std": 1.0})
    return lm


def test_loss_and_finite_gradients(ds):
    batch = maxwell3d_collate([ds[0], ds[1]])
    lm = _module(ds[0]["Input_funcs"].shape[-1], ds.stats, freq_weight=0.1)
    loss, preds, targets = lm._compute_loss(batch, "train")
    assert torch.isfinite(loss) and preds.shape == targets.shape
    loss.backward()
    grads = [p.grad for p in lm.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_dataset_and_collate(pkl, ds):
    tr = Maxwell3DDataset(pkl, split="train", train_ratio=0.6, val_ratio=0.2, augment=True)
    va = Maxwell3DDataset(pkl, split="val", train_ratio=0.6, val_ratio=0.2)
    te = Maxwell3DDataset(pkl, split="test", train_ratio=0.6, val_ratio=0.2)
    assert len(tr) == 3 and len(va) == 1 and len(te) == 1
    assert not set(tr.active_geoms) & set(va.active_geoms + te.active_geoms)
    assert tr.data_dims() == (9, K_MODES)
    it = ds[1]
    assert (it["Y_freq"][1:] >= it["Y_freq"][:-1]).all()
    items = [ds[0], ds[1], ds[2]]
    b = maxwell3d_collate(items)
    B, Nv, Ne = 3, max(i["X"].shape[0] for i in items), max(i["Edges"].shape[0] for i in items)
    assert b["X"].shape == (B, Nv, 3) and b["Y_field"].shape == (B, Ne, K_MODES)
    assert b["Edges"].shape == (B, Ne, 2) and b["EdgeMask"].sum() == sum(i["Edges"].shape[0] for i in items)
    assert b["M"].shape == (B * Ne, B * Ne) and b["G"].shape == (B * Ne, B * Nv) and b["Kp"].shape == (B * Nv, B * Nv)
    assert b["field"] == ds.field and b["KpNull"] == {"H": "const", "E": "none"}[ds.field]
    if ds.field == "E":
        assert b["BndEdge"].shape == (B, Ne) and b["BndEdge"].dtype == torch.bool
        assert not (b["BndEdge"] & ~b["EdgeMask"]).any()                       # padding False
        for k, i in enumerate(items):
            assert torch.equal(b["BndEdge"][k, :i["Edges"].shape[0]], i["BndEdge"])
            assert (i["Y_field"][i["BndEdge"]] == 0).all()                     # wall rows exactly 0
    else:
        assert "BndEdge" not in b
    MY = spmm(b["M"], b["Y_field"].double())
    for k, i in enumerate(items):
        ne = i["Edges"].shape[0]
        ref = i["M"] @ i["Y_field"].double().numpy()
        assert np.allclose(MY[k, :ne].numpy(), ref) and (MY[k, ne:] == 0).all()
        assert np.allclose((i["Y_field"].double().numpy() * ref).sum(0), 1.0, rtol=1e-4)   # unit M-norm targets
    # augmentation: rotated coordinates, same operators and targets
    ta, tb = tr[0], Maxwell3DDataset(pkl, split="train", train_ratio=0.6, val_ratio=0.2)[0]
    assert not torch.allclose(ta["X"], tb["X"])
    assert torch.allclose(ta["X"].norm(dim=-1), tb["X"].norm(dim=-1), atol=1e-5)
    assert torch.equal(ta["Y_field"], tb["Y_field"])


def test_fast_dev_run_train_py(pkl, tmp_path, monkeypatch):
    import train
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/eigenspace_3d.yaml", "--fast_dev_run", "--override",
        f"dataset.data_path={pkl}", "dataset.train_ratio=0.6", "dataset.val_ratio=0.2",
        "model.embed_dim=16", "model.n_basis=10", "model.num_field_modes=4", "model.rff_dim=8",
        "model.eigenspace.n_layers=1", "training.batch_size=2", "training.num_workers=0",
        f"training.log_dir={tmp_path}"])
    train.main()


def test_operators_rebuilt_when_missing(pkl, tmp_path):
    """convert_3d.py --no_operators: M, K, G, Kp are rebuilt from (X, tets)."""
    import pickle
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    if data["metadata"]["field"] == "E" and not _converter_has_field():
        pytest.skip("dataset_converter_3d.geometry_operators has no field argument yet")
    full = Maxwell3DDataset(pkl, split="test", train_ratio=0.0, val_ratio=0.0)
    for g in data["geometry_pool"].values():
        for k in ("M", "K", "G", "Kp"):
            del g[k]
    path = tmp_path / "noops.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    lean = Maxwell3DDataset(str(path), split="test", train_ratio=0.0, val_ratio=0.0)
    a, b = full[0], lean[0]
    for k in ("M", "K", "G", "Kp"):
        assert abs(a[k] - b[k]).max() < 1e-5 * abs(a[k]).max()


def test_split_cluster_is_excluded():
    lm = _module(9)
    lm.near_deg_rel_threshold = 0.01
    lm.freq_stats = {"mean": 0.0, "std": 1.0}
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
        lm._compute_loss(batch, prefix)
    return logs


def test_split_clusters_do_not_poison_logged_metrics(ds):
    """Regression: a split-cluster (NaN) rel-L2 entry is excluded by weight.  Before,
    a batch whose entries were all NaN logged val/field_rel_l2 = NaN, which made
    the epoch mean NaN: EarlyStopping stopped training after one epoch and
    ModelCheckpoint never improved (K=1, batch_size=1 on the smoke data)."""
    lm = _module(ds[0]["Input_funcs"].shape[-1], ds.stats)
    K = lm.model.num_field_modes
    batch = maxwell3d_collate([ds[0], ds[1]])
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
    only = maxwell3d_collate([ds[0]])
    _all_split(only, 0)
    logs = _logged(lm, only)
    assert logs["val/field_rel_l2"] == (0.0, 0)
    assert all(logs[f"val/mode_{k}_rel_l2"] == (0.0, 0) for k in range(K))


def test_all_gradient_sample_gets_theta_above_spectrum(box):
    """docs/21 B1: a sample whose span is pure gradient keeps no direction;
    its 'θ = ∞' must stay far above the physical spectrum (was 1.0)."""
    _, batch, Gphi, lam = box
    theta, fields, _ = hcurl_ritz(Gphi[..., :3], batch, 2, tol=1e-12)
    assert (theta > 100 * lam[-1]).all()
    assert torch.isfinite(theta).all()


def test_projection_forces_float64_and_warns_at_maxiter(box):
    """docs/21 B3/B4: float32 input gives the float64 result; CG stopping at
    maxiter warns instead of returning silently."""
    item, batch, _, _ = box
    V = item["Y_field"][None]
    p32 = project_basis(V.float(), batch, 1e-10)
    p64 = project_basis(V.double(), batch, 1e-10)
    assert p32["M_div"].dtype == torch.float64
    torch.testing.assert_close(p32["M_div"], p64["M_div"], rtol=1e-6, atol=1e-8)
    with pytest.warns(UserWarning, match="maxiter"):
        project_basis(V.double() + 0.1 * torch.randn_like(V.double()), batch, 1e-14, maxiter=2)


# ── E formulation (PEC wall edges essential, Kp SPD on its valid block) ─────────

def _converter_has_field():
    import inspect

    from src.data.dataset_converter_3d import geometry_operators
    return "field" in inspect.signature(geometry_operators).parameters


@pytest.fixture(scope="module")
def e_ds(tmp_path_factory):
    import pickle
    path = tmp_path_factory.mktemp("m3dE") / "synth3d_E_only.pkl"
    with open(path, "wb") as f:
        pickle.dump(make_dataset(n_geoms=3, n=4, n_modes=K_MODES, seed=2, field="E"), f)
    return Maxwell3DDataset(str(path), split="test", train_ratio=0.0, val_ratio=0.0)


def test_e_box_spectrum_and_contract():
    """Synthetic E box: frequencies → the analytic PEC box spectrum (same set as H),
    wall rows 0, G/Kp as in the contract (zero wall rows / zero columns after n_pot,
    SPD valid block), eigenvectors M-orthogonal to span(G)."""
    import scipy.sparse as sp

    from src.data_gen.dataset_generator_3d import box_spectrum
    dims = (1.0, 0.8, 0.6)
    ref = box_spectrum(*dims)[:6]
    geom, lam, Y = box_geometry(dims, n=6, n_modes=6, field="E")
    assert np.abs(lam / ref - 1).max() < 0.03
    bnd, n_pot = geom["bnd_edge"], geom["n_pot"]
    assert geom["field"] == "E" and geom["n_bnd_components"] == 1 and geom["betti1"] == 0
    assert (Y[bnd] == 0).all() and bnd.sum() > 0
    ne, nv = len(geom["edges"]), len(geom["X"])
    M, G, Kp = (to_csr(geom[k], s) for k, s in (("M", (ne, ne)), ("G", (ne, nv)), ("Kp", (nv, nv))))
    assert abs(G[bnd]).sum() == 0 and abs(G[:, n_pot:]).sum() == 0 and abs(Kp[n_pot:]).sum() == 0
    assert np.linalg.eigvalsh(Kp[:n_pot, :n_pot].toarray()).min() > 0
    assert np.abs(G.T @ (M @ Y)).max() < 1e-10
    assert abs(Kp - (G.T @ M @ G)).max() < 1e-12 and isinstance(Kp, sp.csr_matrix)


def test_e_kp_solve_is_plain_spd_pcg(e_ds):
    """KpNull='none': Kp Z = GᵀMV on the valid block (no mean-free step, which would
    be wrong for the SPD E system), Z = 0 on the zero columns; 'const' is not."""
    from src.models.hcurl import _pcg
    batch = maxwell3d_collate([e_ds[0]])
    assert batch["KpNull"] == "none"
    V = torch.randn(1, batch["Edges"].shape[1], 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    V = V * (~batch["BndEdge"]).unsqueeze(-1)
    Bm = spmm(batch["Gt"], spmm(batch["M"], V))
    diag = batch["Kp_diag"]
    dinv = torch.where(diag > 0, 1.0 / diag.clamp(min=1e-300), torch.zeros_like(diag)).unsqueeze(-1)
    valid = diag[0] > 0
    assert 0 < valid.sum() < diag.shape[1]
    Z, _ = _pcg(batch["Kp"], dinv, Bm, 1e-12, 2000, null="none")
    assert (Z[0, ~valid] == 0).all()
    assert (spmm(batch["Kp"], Z) - Bm).abs().max() < 1e-9 * Bm.abs().max()
    Zc, _ = _pcg(batch["Kp"], dinv, Bm, 1e-12, 2000, null="const")
    assert (spmm(batch["Kp"], Zc) - Bm).abs().max() > 1e-3 * Bm.abs().max()


def test_e_model_zero_on_wall_and_losses(e_ds):
    """E: raw basis and Ritz fields exactly 0 on PEC wall edges; the Ritz values are
    ≥ λ_h (min–max on the E space); train loss finite with finite gradients."""
    batch = maxwell3d_collate([e_ds[0], e_ds[1]])
    model = _model(e_ds[0]["Input_funcs"].shape[-1], e_ds.stats)
    with torch.no_grad():
        out = model(batch)
    wall = batch["BndEdge"]
    assert wall.any() and (out["basis"][wall] == 0).all() and (out["field"][wall] == 0).all()
    for b in range(2):
        lam = _lam(e_ds[b], e_ds.stats)
        assert (out["eigenvalues"][b].double() >= lam[:4] * (1 - 1e-6)).all()
    nob = {k: v for k, v in batch.items() if k != "BndEdge"}
    with pytest.raises(ValueError, match="BndEdge"):
        model(nob)


def test_mixed_field_batch_rejected(ds, e_ds):
    item_h = ds[0] if ds.field == "H" else None
    if item_h is None:
        pytest.skip("runs once, with the H dataset")
    with pytest.raises(ValueError, match="mixes fields"):
        maxwell3d_collate([item_h, e_ds[0]])


def test_e_item_from_geometry_matches_dataset(e_ds):
    """Active sampling path: a converter-style E geometry dict (field key, bnd_edge)
    gives the same model input as the dataset item."""
    g = e_ds.geometry_pool[e_ds.active_geoms[0]]
    it = item_from_geometry(g, feature_names=e_ds.feature_names, g_id=0)
    ref = e_ds[0]
    assert it["field"] == "E" and torch.equal(it["BndEdge"], ref["BndEdge"])
    b = maxwell3d_collate([it])
    assert b["KpNull"] == "none" and b["BndEdge"].shape == (1, ref["Edges"].shape[0])
    g_h = {k: v for k, v in g.items() if k not in ("field", "bnd_edge")}
    assert "BndEdge" not in item_from_geometry(g_h) and maxwell3d_collate([item_from_geometry(g_h)])["KpNull"] == "const"
    with pytest.raises(ValueError, match="bnd_edge"):
        item_from_geometry(dict(g_h, field="E"))


def _synth_e_operators(X, tets, field="H"):
    """Stand-in for geometry_operators(X, tets, field=...) built from the synthetic
    skfem operators (same mesh ⇒ same edges)."""
    from skfem import MeshTet

    from src.data.dataset_converter_3d import to_csr_tuple
    from tests.maxwell3d_synth import e_operators, n0_operators
    mesh = MeshTet(np.ascontiguousarray(np.asarray(X, np.float64).T), np.ascontiguousarray(np.asarray(tets).T))
    if field == "E":
        K, M, G, Kp, edges, bnd, n_pot, n_comp = e_operators(mesh)
        extra = {"bnd_edge": bnd, "n_pot": n_pot, "n_bnd_components": n_comp, "field": "E"}
    else:
        K, M, G, Kp, edges = n0_operators(mesh)
        extra = {}
    return {"edges": edges, "M": to_csr_tuple(M), "K": to_csr_tuple(K), "G": to_csr_tuple(G),
            "Kp": to_csr_tuple(Kp), **extra}, M


def _lean(e_ds, tmp_path, drop_bnd=False):
    import copy
    import pickle
    data = {"geometry_pool": copy.deepcopy(e_ds.geometry_pool), "samples": e_ds.samples_metadata,
            "metadata": e_ds.metadata}
    for g in data["geometry_pool"].values():
        for k in ("M", "K", "G", "Kp") + (("bnd_edge",) if drop_bnd else ()):
            del g[k]
    path = tmp_path / "lean_E.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path)


def test_e_lean_pkl_rebuild(e_ds, tmp_path, monkeypatch):
    """Lean E PKL: operators AND bnd_edge come from geometry_operators(X, tets, field='E');
    a converter without the field argument, or a rebuilt dict without 'bnd_edge',
    fails with a clear error."""
    import src.data.dataset_3d as d3
    path = _lean(e_ds, tmp_path, drop_bnd=True)
    monkeypatch.setattr(d3, "geometry_operators", _synth_e_operators)
    lean = Maxwell3DDataset(path, split="test", train_ratio=0.0, val_ratio=0.0)
    a, b = e_ds[0], lean[0]
    for k in ("M", "K", "G", "Kp"):
        assert abs(a[k] - b[k]).max() < 1e-5 * abs(a[k]).max()
    assert torch.equal(a["BndEdge"], b["BndEdge"])

    def no_bnd(X, tets, field="H"):
        ops, M = _synth_e_operators(X, tets, field)
        ops.pop("bnd_edge", None)
        return ops, M
    monkeypatch.setattr(d3, "geometry_operators", no_bnd)
    with pytest.raises(ValueError, match="bnd_edge"):
        Maxwell3DDataset(path, split="test", train_ratio=0.0, val_ratio=0.0)[0]
    monkeypatch.setattr(d3, "geometry_operators", lambda X, tets: _synth_e_operators(X, tets, "H"))
    with pytest.raises(ValueError, match="field"):
        Maxwell3DDataset(path, split="test", train_ratio=0.0, val_ratio=0.0)[0]


def test_e_converter_operators_match_contract(e_ds):
    """Integration check of the other side: the converter's E operators on a synthetic
    box equal the synthetic contract operators (skipped until field= exists)."""
    if not _converter_has_field():
        pytest.skip("dataset_converter_3d.geometry_operators has no field argument yet")
    from src.data.dataset_converter_3d import geometry_operators
    g = e_ds.geometry_pool[e_ds.active_geoms[0]]
    ops, _ = geometry_operators(np.asarray(g["X"], np.float64), np.asarray(g["tets"]), field="E")
    ne, nv = len(g["edges"]), len(g["X"])
    assert np.array_equal(ops["edges"], g["edges"])
    assert np.array_equal(np.asarray(ops["bnd_edge"], bool), g["bnd_edge"])
    shapes = {"M": (ne, ne), "K": (ne, ne), "G": (ne, nv), "Kp": (nv, nv)}
    for k in ("M", "K", "G", "Kp"):
        A, B = to_csr(ops[k], shapes[k]), to_csr(g[k], shapes[k])
        assert abs(A - B).max() < 1e-5 * max(abs(B).max(), 1e-12), k
    assert int(ops.get("n_pot", g["n_pot"])) == g["n_pot"]
