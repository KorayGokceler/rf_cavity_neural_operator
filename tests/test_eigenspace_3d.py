"""3D Maxwell eigenspace operator (H field, Whitney N0): kernel projection,
Ritz exactness, orientation / padding invariance, dataset, training wiring."""
import math
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("skfem")

from src.data.dataset_3d import Maxwell3DDataset, maxwell3d_collate, to_csr   # noqa: E402
from src.models.eigenspace_operator_3d import EigenspaceOperator3D            # noqa: E402
from src.models.hcurl import (hcurl_grams, hcurl_ritz, mode_rel_l2,           # noqa: E402
                              project_basis, spmm)
from src.models.spectral_no import _generalized_eigh                          # noqa: E402
from src.training.lightning_module import (GNOTLightning, ritz_compliance,    # noqa: E402
                                           span_residual)
from tests.maxwell3d_synth import C0, edge_dofs, make_dataset                 # noqa: E402

K_MODES = 6


@pytest.fixture(scope="module")
def pkl(tmp_path_factory):
    import pickle
    path = tmp_path_factory.mktemp("m3d") / "synth3d.pkl"
    with open(path, "wb") as f:
        pickle.dump(make_dataset(n_geoms=5, n=4, n_modes=K_MODES, seed=0), f)
    return str(path)


STATS = {}   # freq z-score stats of the synthetic PKL (set by the ds fixture)


@pytest.fixture(scope="module")
def ds(pkl):
    d = Maxwell3DDataset(pkl, split="test", train_ratio=0.0, val_ratio=0.0)   # all geometries
    STATS.update(d.stats)
    return d


def _lam(item):
    """Discrete λ_h (normalised mesh): Rayleigh quotients of the stored eigenvectors
    (float32 DOFs → error O(eps32²)); consistent with the stored GHz, f = c√λ/(2π·scale)."""
    T = item["Y_field"].double().numpy()
    lam = (T * (item["K"] @ T)).sum(0) / (T * (item["M"] @ T)).sum(0)
    f = item["Y_freq"].double().numpy() * STATS["std"] + STATS["mean"]
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


def _model(val_dim, **kw):
    torch.manual_seed(0)
    m = EigenspaceOperator3D(val_dim, embed_dim=16, n_layers=1, n_heads=2, n_basis=10,
                             num_field_modes=4, rff_dim=8, kp_tol=1e-12, **kw).eval()
    m.freq_stats = dict(STATS)
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
    assert np.abs(item_f["K"] @ y - lam0 * (item_f["M"] @ y)).max() < 1e-4 * np.abs(item_f["K"] @ y).max()
    # model: raw basis and Ritz fields flip exactly, eigenvalues unchanged
    model = _model(item["Input_funcs"].shape[-1])
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
    model = _model(a["Input_funcs"].shape[-1])
    with torch.no_grad():
        o1, o2 = model(maxwell3d_collate([a])), model(maxwell3d_collate([b, a, b]))
    ne = a["Edges"].shape[0]
    assert torch.allclose(o2["basis"][1, :ne], o1["basis"][0], atol=1e-6)
    assert (o2["basis"][1, ne:] == 0).all()
    assert torch.allclose(o2["eigenvalues"][1], o1["eigenvalues"][0], rtol=1e-6)
    F1, F2 = o1["field"][0], o2["field"][1, :ne]
    assert torch.allclose(F2 * torch.sign((F1 * F2).sum(0)), F1, atol=1e-5)


def _module(val_dim, **kw):
    torch.manual_seed(0)
    lm = GNOTLightning(val_dim=val_dim, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=10,
                       num_field_modes=4, rff_dim=8, model_type="eigenspace3d", physics_freq=True,
                       eigenspace_kwargs={"n_layers": 1}, near_deg_rel_threshold=0.02, **kw)
    lm.freq_stats = dict(STATS)
    return lm


def test_loss_and_finite_gradients(ds):
    batch = maxwell3d_collate([ds[0], ds[1]])
    lm = _module(ds[0]["Input_funcs"].shape[-1], freq_weight=0.1)
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
