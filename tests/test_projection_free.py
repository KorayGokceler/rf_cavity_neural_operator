"""Projection-free training and validation (training.projection: false, docs/38): the Grams without the kernel
projection, the train and validation steps without a Kp solve (one loss function for both), the projected test
step, the monitored metric, the unchanged default, the options' guards, and the datasets without G / Kp.
Pillbox + box on curved p2 meshes (NGSolve), as tests/test_hcurl_projection.py."""
import sys
import types

import numpy as np
import pytest
import torch

pytest.importorskip("ngsolve")
from netgen.occ import Axes, Box, Cylinder, OCCGeometry, Pnt, Z  # noqa: E402
from ngsolve import Mesh  # noqa: E402

from src.data.field_dataset import field_collate, field_item  # noqa: E402
from src.data.mesh_features import normalise  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.models import hcurl  # noqa: E402
from src.models.eigenspace_operator_field import EigenspaceOperatorField  # noqa: E402
from src.models.hcurl import hcurl_grams, hcurl_grams_plain, project_basis, spmm  # noqa: E402
from src.training.lightning_module import CavityLightning, _basis_conditioning_loss, span_residual  # noqa: E402

K_MODES = 3


def _build(geo, maxh, k):
    geo.faces.name = "wall"
    mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=maxh))
    mesh.Curve(2)
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)
    ops = ho.field_operators(mesh, order=2, scale=scale, center=center)
    r = ho.solve_modes(mesh, k, order=2)
    Y = np.column_stack([g.vec.FV().NumPy() for g in r["gfs"]])
    ops_np = ho.field_operators(mesh, order=2, scale=scale, center=center, potential=False)
    return mesh, ops, ops_np, r, Y


@pytest.fixture(scope="module")
def pill():
    return _build(Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1), 0.05, K_MODES)


@pytest.fixture(scope="module")
def box():
    return _build(Box(Pnt(0, 0, 0), Pnt(1.0, 0.8, 0.6)), 0.25, K_MODES)


def _item(ops, r, Y, **kw):
    it = field_item(ops, Y[:, :K_MODES], r["f_hz"][:K_MODES] / 1e9, **kw)
    it["FreqNext"] = torch.tensor(float("nan"))
    it["geom_id"] = torch.tensor([0])
    return it


def _module(**kw):
    torch.manual_seed(0)
    kw = {"near_deg_rel_threshold": 0.02, **kw}
    lm = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K_MODES, rff_dim=8,
                         eigenspace_kwargs={"n_layers": 1}, **kw)
    lm.freq_stats = {"mean": 0.0, "std": 1.0}
    return lm


def _capture(lm):
    """Replace lm.log: {name: (float value or None for a torchmetrics object, batch_size)}."""
    logs = {}

    def log(name, value, **kw):
        try:
            value = float(value.detach() if torch.is_tensor(value) else value)
        except (TypeError, ValueError):
            value = None
        logs[name] = (value, kw.get("batch_size"))
    lm.log = log
    return logs


class _CgCounter:
    """Counts the Kp CG solves (forward and backward of KpSolve)."""

    def __init__(self, monkeypatch):
        self.n, self.orig = 0, hcurl._pcg

        def counted(*a, **k):
            self.n += 1
            return self.orig(*a, **k)
        monkeypatch.setattr(hcurl, "_pcg", counted)


# ── operators without the kernel potentials ─────────────────────────────────────────────────────────

def test_operators_without_potentials_are_the_full_ones_minus_g_and_kp(pill):
    _, ops, ops_np, _, _ = pill
    assert "G" not in ops_np and "Kp" not in ops_np and ops_np["n_pot"] is None and ops["n_pot"] > 0
    assert set(ops) - set(ops_np) == {"G", "Kp"}
    for k in ("K", "M", "C"):
        assert (ops[k] != ops_np[k]).nnz == 0
    for k in ("P", "qx", "qw", "qfeat", "bnd", "bary", "X", "tets", "center"):
        assert np.array_equal(ops[k], ops_np[k]), k
    assert ops["scale"] == ops_np["scale"] and ops["n_dof"] == ops_np["n_dof"]
    # items / batches carry no G, Gt, Kp, Kp_diag, everything else as the full ones
    full = field_collate([_item(ops, pill[3], pill[4])])
    red = field_collate([_item(ops_np, pill[3], pill[4])])
    assert {"G", "Gt", "Kp", "Kp_diag"} <= set(full) and not ({"G", "Gt", "Kp", "Kp_diag"} & set(red))
    assert set(full) - set(red) == {"G", "Gt", "Kp", "Kp_diag"}
    for k in set(red):
        if torch.is_tensor(red[k]) and not red[k].is_sparse_csr:
            assert torch.allclose(red[k], full[k], rtol=0, atol=0, equal_nan=True), k
    for k in ("M", "K", "C"):
        assert torch.equal(red[k].values(), full[k].values()) and torch.equal(red[k].col_indices(), full[k].col_indices())
    with pytest.raises(ValueError, match="some items carry the kernel potentials"):
        field_collate([_item(ops, pill[3], pill[4]), _item(ops_np, pill[3], pill[4])])


# ── Grams ───────────────────────────────────────────────────────────────────────────────────────────

def _mixed_basis(batch, T, Gphi):
    """V = T + 0.3·(unit-mass gradients) + 2 random fields: the targets are (nearly) in span(V), with gradient content."""
    keep = (batch["DofMask"] & ~batch["BndDof"]).unsqueeze(-1)
    Tn = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    Gn = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()
    rnd = torch.randn(1, T.shape[1], 2, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    return torch.cat([Tn + 0.3 * Gn[..., :K_MODES], rnd * keep], -1) * keep


def test_plain_grams_against_projected_ones(pill):
    _, ops, _, r, Y = pill
    item = _item(ops, r, Y)
    batch = field_collate([item])
    T = item["Y_field"].double()[None]
    Gphi = torch.from_numpy(ops["G"] @ np.random.default_rng(0).standard_normal((ops["n_pot"], 4)))[None]
    V = _mixed_basis(batch, T, Gphi)
    m = V.shape[-1]
    proj = project_basis(V, batch, 1e-12)
    G_M, G_A, MT = hcurl_grams(V, T, proj, batch)
    P_M, P_A, MT2 = hcurl_grams_plain(V, T, batch)
    assert torch.equal(MT, MT2)
    # energy Grams: identical bit for bit (K G = 0: the projection never touches them)
    assert torch.equal(G_A, P_A)
    # mass: VV = VᵀMV (project_basis' M_V), TT identical, cross blocks equal up to the label precision (GᵀMT ~ 1e-8)
    assert torch.equal(P_M[:, :m, :m], proj["M_V"]) and torch.equal(P_M[:, m:, m:], G_M[:, m:, m:])
    cross = P_M[:, :m, m:]
    assert (cross - G_M[:, :m, m:]).abs().max() < 1e-6 * cross.abs().max()
    GtMT = spmm(batch["Gt"], MT)
    assert (GtMT.norm(dim=1) / MT.norm(dim=1)).max() < 1e-6                       # labels ⟂_M range(G)
    # the unprojected VV block dominates M_div (Schur complement): gradient mass of V counts
    gap = torch.linalg.eigvalsh(P_M[:, :m, :m] - G_M[:, :m, :m])
    assert gap.min() > -1e-10 * P_M[:, :m, :m].abs().max() and gap.max() > 1e-3
    # residuals: r_plain ≥ r_proj in the mass norm (cross blocks equal), equal in the energy norm
    r_proj, r_plain = span_residual(G_M, m), span_residual(P_M, m)
    assert (r_plain >= r_proj - 1e-12).all() and (r_plain > 1e-3).all() and (r_proj < 1e-6).all()
    assert torch.equal(span_residual(G_A, m), span_residual(P_A, m))


def test_plain_grams_match_a_scipy_reference(pill, box):
    """[V|T]ᵀ M [V|T] and [V|T]ᵀ K [V|T] with scipy matrices; a batch of two samples of different sizes (padding)."""
    items = [_item(pill[1], pill[3], pill[4]), _item(box[1], box[3], box[4])]
    batch = field_collate(items)
    assert items[0]["M"].shape[0] < items[1]["M"].shape[0]
    V = torch.randn(2, batch["DofMask"].shape[1], 5, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    V = (V * (batch["DofMask"] & ~batch["BndDof"]).unsqueeze(-1)).float()
    G_M, G_A, _ = hcurl_grams_plain(V, batch["Y_field"], batch)
    for b, (ops, it) in enumerate(zip((pill[1], box[1]), items, strict=True)):
        n = ops["n_dof"]
        VT = torch.cat([V[b, :n].double(), it["Y_field"].double()], -1).numpy()
        ref_M, ref_A = VT.T @ (ops["M"] @ VT), VT.T @ (ops["K"] @ VT)
        np.testing.assert_allclose(G_M[b].numpy(), ref_M, rtol=1e-10, atol=1e-11 * abs(ref_M).max())
        np.testing.assert_allclose(G_A[b].numpy(), ref_A, rtol=1e-10, atol=1e-11 * abs(ref_A).max())


def test_plain_grams_backward_with_and_without_precomputed_transposes(pill):
    """The exactly-symmetric M, K backward (SymmetricMK) equals torch.sparse.mm's own backward, bit for bit."""
    _, ops, _, r, Y = pill
    batch = field_collate([_item(ops, r, Y)])
    plain = {k: v for k, v in batch.items() if k != "SymmetricMK"}
    V0 = torch.randn(1, batch["DofMask"].shape[1], 4, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    W = torch.randn(1, 4 + K_MODES, 4 + K_MODES, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    grads = []
    for b in (batch, plain):
        V = V0.clone().requires_grad_()
        G_M, G_A, _ = hcurl_grams_plain(V, b["Y_field"], b)
        ((G_M * W).sum() + (G_A * W).sum()).backward()
        grads.append(V.grad)
    assert torch.equal(grads[0], grads[1]) and grads[0].abs().sum() > 0


# ── the training step ───────────────────────────────────────────────────────────────────────────────

def test_forward_basis_is_the_basis_of_forward(pill):
    _, ops, ops_np, r, Y = pill
    full, red = field_collate([_item(ops, r, Y)]), field_collate([_item(ops_np, r, Y)])
    torch.manual_seed(0)
    model = EigenspaceOperatorField(9, embed_dim=16, n_layers=1, n_heads=2, n_basis=8, num_field_modes=K_MODES,
                                    rff_dim=8).eval()
    model.freq_stats = {"mean": 0.0, "std": 1.0}
    with torch.no_grad():
        assert torch.equal(model(full)["basis"], model.forward_basis(full)["basis"])
        assert torch.equal(model.forward_basis(red)["basis"], model.forward_basis(full)["basis"])
        with pytest.raises(ValueError, match=r"needs batch keys \['G'"):          # the Ritz forward needs G, Kp
            model(red)
        with pytest.raises(ValueError, match="BndDof"):
            model.forward_basis({k: v for k, v in red.items() if k != "BndDof"})


def test_noproj_loss_is_span_plus_ortho_of_the_unprojected_grams(pill, box):
    """The training loss equals span_weight·mean√(½(r_M + r_A)) + ortho_weight·ortho of scipy-built unprojected Grams
    (independent of hcurl_grams_plain; unit mass-norm columns, ortho on the curl–curl correlations), the logged parts are consistent and there is no selfsup / freq / Ritz key."""
    items = [_item(pill[2], pill[3], pill[4]), _item(box[2], box[3], box[4])]
    batch = field_collate(items)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    lm.train()
    logs = _capture(lm)
    loss = lm._compute_loss_3d_plain(batch, "train")
    with torch.no_grad():
        V = lm.model.forward_basis(batch)["basis"].double()
    r_M, r_A, ortho = [], [], []
    m = V.shape[-1]
    for b, (ops, it) in enumerate(zip((pill[2], box[2]), items, strict=True)):
        n = ops["n_dof"]
        VT = torch.cat([V[b, :n], it["Y_field"].double()], -1).numpy()
        gm, ga = VT.T @ (ops["M"] @ VT), VT.T @ (ops["K"] @ VT)
        sc = np.concatenate([1.0 / np.sqrt(np.diag(gm)[:m]), np.ones(gm.shape[0] - m)])      # unit mass-norm columns
        gm, ga = gm * np.outer(sc, sc), ga * np.outer(sc, sc)
        r_M.append(span_residual(torch.from_numpy(gm)[None], m)[0])
        r_A.append(span_residual(torch.from_numpy(ga)[None], m)[0])
        c = ga[:m, :m] / np.sqrt(np.outer(np.diag(ga)[:m], np.diag(ga)[:m]))                 # curl–curl correlations
        ortho.append((c - np.diag(np.diag(c))).__pow__(2).sum() / (m * (m - 1)))
    r_M, r_A = torch.stack(r_M), torch.stack(r_A)
    span = torch.sqrt(0.5 * (r_M + r_A) + 1e-8).mean()
    expected = span + 0.01 * float(np.mean(ortho))
    assert torch.isfinite(loss) and abs(float(loss) - float(expected)) < 1e-4 * float(expected)
    assert set(logs) == {"train/span_loss", "train/ortho_loss", "train/span_rel_l2", "train/span_rel_l2_energy",
                         *(f"train/span_mode_{k}_rel_l2" for k in range(K_MODES)), "train/loss",
                         "train/mass_diag_ratio",
                         "train/grad_frac_lb", "train/grad_frac_lb_max"}
    assert abs(logs["train/span_loss"][0] - float(span)) < 1e-4 * float(span)
    assert abs(logs["train/ortho_loss"][0] - float(np.mean(ortho))) < 1e-4 * float(np.mean(ortho))
    assert logs["train/mass_diag_ratio"][0] >= 1.0
    assert all(bs == 2 for _, bs in logs.values())
    loss.backward()
    grads = [p.grad for p in lm.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


def test_noproj_train_and_val_steps_have_no_kp_solve_and_the_test_step_projects(pill, box, monkeypatch):
    red = field_collate([_item(pill[2], pill[3], pill[4]), _item(box[2], box[3], box[4])])
    full = field_collate([_item(pill[1], pill[3], pill[4]), _item(box[1], box[3], box[4])])
    assert not ({"G", "Gt", "Kp", "Kp_diag"} & set(red))
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    lm.optimizers = lambda: types.SimpleNamespace(param_groups=[{"lr": 1e-3}])
    logs = _capture(lm)
    cg = _CgCounter(monkeypatch)
    calls = []
    orig_apply = hcurl.KpSolve.apply
    monkeypatch.setattr(hcurl.KpSolve, "apply", staticmethod(lambda *a: calls.append(1) or orig_apply(*a)))
    loss = lm.training_step(red, 0)                     # forward + loss + backward without any Kp solve
    loss.backward()
    assert cg.n == 0 and not calls and "train/field_rel_l2" not in logs and "train/r2" not in logs
    assert "lr" in logs and "train/loss" in logs and "train/freq_loss" not in logs
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lm.parameters())
    # validation: the same reduced batch, the same loss function, the same keys, no Ritz key, no CG
    logs.clear()
    lm.eval()
    with torch.no_grad():
        lm.validation_step(red, 0)
    assert cg.n == 0 and not calls
    assert set(logs) == {"val/span_loss", "val/ortho_loss", "val/span_rel_l2", "val/span_rel_l2_energy",
                         *(f"val/span_mode_{k}_rel_l2" for k in range(K_MODES)), "val/loss", "val/mass_diag_ratio",
                         "val/grad_frac_lb", "val/grad_frac_lb_max"}
    # the test step projects: CG runs, the Ritz metrics are logged, and the training loss comes next to the projected one
    logs.clear()
    with torch.no_grad():
        lm.test_step(full, 0)
    assert cg.n > 0 and calls
    assert {"test/field_rel_l2", "test/freq_rel_err", "test/freq_mae_ghz", "test/grad_frac", "test/kp_cg_iters",
            "test/selfsup_loss", "test/mode_0_rel_l2", "test/loss", "test/loss_plain", "test/r2",
            "test/mae"} <= set(logs)
    assert "test/mass_diag_ratio" not in logs and not any(k.startswith("val/") for k in logs)
    with pytest.raises(ValueError, match=r"needs batch keys \['G'"):                # a test batch needs G, Kp
        lm.test_step(red, 0)


def test_noproj_val_loss_is_the_train_loss_on_the_same_batch(pill, box):
    """Like for like: one function computes both, so on the same batch, in eval mode with the same weights, every
    val/ value equals the train/ value (torch.equal on the logged floats), and the test step's loss_plain too."""
    items = [_item(pill[2], pill[3], pill[4]), _item(box[2], box[3], box[4])]
    batch = field_collate(items)
    full = field_collate([_item(pill[1], pill[3], pill[4]), _item(box[1], box[3], box[4])])
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    lm.optimizers = lambda: types.SimpleNamespace(param_groups=[{"lr": 1e-3}])
    lm.eval()
    logs = _capture(lm)
    with torch.no_grad():
        tl = lm.training_step(batch, 0)
        vl = lm.validation_step(batch, 0)
    assert torch.equal(tl, vl)
    tr = {k[len("train/"):]: v for k, v in logs.items() if k.startswith("train/")}
    va = {k[len("val/"):]: v for k, v in logs.items() if k.startswith("val/")}
    assert tr == va and "loss" in va and "mass_diag_ratio" in va and len(va) == 8 + K_MODES
    # the projected test batch (with G, Kp) gives the same plain loss: the plain loss never reads them
    with torch.no_grad():
        assert torch.equal(lm._compute_loss_3d_plain(full, None), lm._compute_loss_3d_plain(batch, None))
        lm.test_step(full, 0)
    assert logs["test/loss_plain"][0] == va["loss"][0]
    # the weights of the terms are the same function's (span_weight, ortho_weight): change one, both follow
    lm.ortho_weight = 0.0
    with torch.no_grad():
        l2 = lm.validation_step(batch, 0)
    assert abs(float(l2) - logs["val/span_loss"][0]) < 1e-6 and not torch.equal(l2, vl)


def test_monitor_follows_the_projection_option(tmp_path):
    """ModelCheckpoint, EarlyStopping and the ReduceLROnPlateau scheduler monitor val/field_rel_l2 by default and
    val/span_loss without the projection (the span part of the loss: val/loss adds the ortho regulariser); the file
    name carries the metric (src.training.checkpoint reads it)."""
    import train
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from src.training.lightning_module import monitor_metric
    assert monitor_metric(True) == "val/field_rel_l2" and monitor_metric(False) == "val/span_loss"
    tc = types.SimpleNamespace(log_dir=str(tmp_path), exp_name="e", patience=7)
    for projection, monitor, name in ((True, "val/field_rel_l2", "best-epoch={epoch:02d}-val_rel_l2={val/field_rel_l2:.4f}"),
                                      (False, "val/span_loss", "best-epoch={epoch:02d}-val_span_loss={val/span_loss:.4f}")):
        ckpt, stop = train._selection_callbacks(tc, projection)
        assert isinstance(ckpt, ModelCheckpoint) and isinstance(stop, EarlyStopping)
        assert ckpt.monitor == stop.monitor == monitor and ckpt.mode == stop.mode == "min"
        assert ckpt.filename == name and ckpt.save_top_k == 1 and ckpt.save_last and stop.patience == 7
        assert ckpt.dirpath == f"{tmp_path}/e" and ckpt.auto_insert_metric_name is False
        assert ckpt.format_checkpoint_name({"epoch": 3, monitor: torch.tensor(0.123456)}).endswith(
            ("val_rel_l2=0.1235.ckpt" if projection else "val_span_loss=0.1235.ckpt"))
    for projection, monitor in ((True, "val/field_rel_l2"), (False, "val/span_loss")):
        lm = _module(**({} if projection else dict(projection=False, freq_weight=0.0, selfsup_weight=0.0)),
                     scheduler="reducelr")
        lm.trainer = types.SimpleNamespace(check_val_every_n_epoch=1, max_epochs=3)
        cfg = lm.configure_optimizers()
        assert cfg["lr_scheduler"]["monitor"] == monitor


def test_resolve_checkpoint_reads_all_score_tags(tmp_path):
    import os

    from src.training.checkpoint import resolve_checkpoint
    d = tmp_path / "run"
    d.mkdir()
    for n in ("last.ckpt", "best-epoch=03-val_loss=0.2500.ckpt"):
        (d / n).write_bytes(b"0")
    assert resolve_checkpoint(str(d)).endswith("val_loss=0.2500.ckpt")
    (d / "best-epoch=02-val_loss=0.3000-v1.ckpt").write_bytes(b"0")
    assert resolve_checkpoint(str(d)).endswith("val_loss=0.2500.ckpt")                 # lowest within the kind
    assert resolve_checkpoint(str(d / "last.ckpt")).endswith("last.ckpt")
    # the current projection-free tag (val/span_loss) is read as a kind of its own; val_loss files stay readable
    sp = tmp_path / "span"
    sp.mkdir()
    for n in ("last.ckpt", "best-epoch=01-val_span_loss=0.4000.ckpt", "best-epoch=05-val_span_loss=0.3000-v1.ckpt"):
        (sp / n).write_bytes(b"0")
    assert resolve_checkpoint(str(sp)).endswith("best-epoch=05-val_span_loss=0.3000-v1.ckpt")
    # a run resumed with the other option: the scores are not comparable, the newest kind wins
    new = d / "best-epoch=09-val_rel_l2=0.9000.ckpt"
    new.write_bytes(b"0")
    os.utime(new, (4e9, 4e9))
    assert resolve_checkpoint(str(d)).endswith("val_rel_l2=0.9000.ckpt")
    only_last = tmp_path / "only"
    only_last.mkdir()
    (only_last / "last.ckpt").write_bytes(b"0")
    assert resolve_checkpoint(str(only_last)).endswith("last.ckpt")


def test_noproj_training_learns(pill):
    """The projection-free loss trains the trunk: the span error of the training Grams falls."""
    _, _, ops_np, r, Y = pill
    batch = field_collate([_item(ops_np, r, Y, with_ct=True)] * 2)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    opt = torch.optim.Adam(lm.parameters(), lr=3e-3)
    _capture(lm)
    losses = []
    for _ in range(25):
        loss = lm._compute_loss_3d_plain(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.8 * losses[0]


def test_default_projection_loss_is_composed_as_before(pill, box, monkeypatch):
    """projection: true (default): the loss is span + 0.01·selfsup + 0.01·ortho + 0.1·freq of the PROJECTED problem, with
    the Ritz metrics logged, exactly the terms of the logs."""
    batch = field_collate([_item(pill[1], pill[3], pill[4]), _item(box[1], box[3], box[4])])
    lm = _module()
    assert lm.projection is True and lm.hparams.projection is True
    logs = _capture(lm)
    loss, _, _ = lm._compute_loss_3d(batch, "train")
    parts = (1.0 * logs["train/span_loss"][0] + 0.01 * logs["train/selfsup_loss"][0]
             + 0.01 * logs["train/ortho_loss"][0] + 0.1 * logs["train/freq_loss"][0])
    assert abs(float(loss) - parts) < 1e-5 * abs(parts)
    assert {"train/field_rel_l2", "train/grad_frac", "train/kp_cg_iters", "train/selfsup_loss",
            "train/freq_rel_err"} <= set(logs)
    lm.optimizers = lambda: types.SimpleNamespace(param_groups=[{"lr": 1e-3}])
    cg = _CgCounter(monkeypatch)
    lm.training_step(batch, 0)
    assert cg.n > 0 and "train/r2" in logs


# ── gradient-content robustness of the projection-free loss (docs/38 §2.5–2.6) ──────────────────────

def _plain_setup(pill):
    _, ops, _, r, Y = pill
    item = _item(ops, r, Y)
    batch = field_collate([item])
    keep = (batch["DofMask"] & ~batch["BndDof"]).unsqueeze(-1)
    T = item["Y_field"].double()[None]
    G = torch.from_numpy(ops["G"].toarray()) if hasattr(ops["G"], "toarray") else torch.from_numpy(ops["G"])
    return ops, batch, T, G, keep


def test_plain_ortho_is_blind_to_gradient_content_but_sees_collinearity(pill):
    """3 clean columns + 3 redundant copies of them: adding independent gradient parts G·φ to the copies leaves
    the span-relevant (curl) part, so the plain-path ortho (curl–curl correlations) does not move, while the
    mass-Gram ortho falls (the gradient parts dilute the correlations: the reward the energy Gram removes)."""
    from src.training.lightning_module import _basis_conditioning_loss
    ops, batch, T, G, keep = _plain_setup(pill)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    Tn = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    phi = torch.randn(G.shape[1], K_MODES, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    Gphi = (G @ phi)[None]
    Gphi = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()
    res = {}
    for a in (0.0, 3.0):
        V = torch.cat([Tn, Tn + a * Gphi], -1) * keep                      # copies: same curl, a· gradient
        G_M, G_A, _ = hcurl_grams_plain(V.float(), batch["Y_field"], batch)
        res[a] = (None,
                  _basis_conditioning_loss(G_A[:, :6, :6], rel_floor=1e-12), _basis_conditioning_loss(G_M[:, :6, :6]))
        Gn, An, _ = lm._normalise_columns(G_M, G_A, 6)
        res[a] += (_basis_conditioning_loss(An[:, :6, :6], rel_floor=1e-12),)
    # energy-Gram ortho: the same at a = 0 and a = 3 (K G = 0; float32 basis → ~1e-6), and large (the copies are collinear)
    assert abs(float(res[3.0][1]) - float(res[0.0][1])) < 1e-5 and float(res[0.0][1]) > 0.1
    assert abs(float(res[3.0][3]) - float(res[0.0][3])) < 1e-5            # also after the unit-mass-norm column scaling
    # the mass-Gram ortho is diluted by the gradient parts (the probe): it must have fallen well below the energy one
    assert float(res[3.0][2]) < 0.3 * float(res[0.0][2])
    # a pure-gradient column (zero curl) neither blows up nor counts as collinear with the clean ones
    V = torch.cat([Tn, 1e3 * Gphi[..., :1]], -1) * keep
    G_M, G_A, _ = hcurl_grams_plain(V.float(), batch["Y_field"], batch)
    Gn, An, ratio = lm._normalise_columns(G_M, G_A, 4)
    o = _basis_conditioning_loss(An[:, :4, :4], rel_floor=1e-12)
    clean = _basis_conditioning_loss(An[:, :3, :3], rel_floor=1e-12)
    assert torch.isfinite(o) and float(o) <= float(clean) * 9 / 12 + 1e-6 and float(ratio) > 1e3


def test_plain_span_residual_survives_huge_gradient_columns(pill):
    """Probe: the clean columns (exact span of the targets, mass norm ~3) + two pure-gradient columns of mass norm 1e7 (Gram diagonal ratio 1e13).  The raw dead-column rule
    of the Jacobi-Cholesky (diag ≤ 1e-12·max) calls the clean columns dead (r_M = 1); the unit-mass-norm columns of
    the projection-free loss do not: r_M ≈ 0 for the represented targets, as without the gradient columns."""
    ops, batch, T, G, keep = _plain_setup(pill)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    Gphi = (G @ torch.randn(G.shape[1], 2, dtype=torch.float64, generator=torch.Generator().manual_seed(8)))[None]
    Gphi = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()        # unit mass norm, then ×1e7
    V = torch.cat([T, 1e7 * Gphi], -1) * keep
    G_M, G_A, _ = hcurl_grams_plain(V.float(), batch["Y_field"], batch)
    m = V.shape[-1]
    assert float(span_residual(G_M, m).max()) > 0.99                         # raw rule: clean columns "dead"
    Gn, An, ratio = lm._normalise_columns(G_M, G_A, m)
    assert float(span_residual(Gn, m).max()) < 1e-6 and float(span_residual(An, m).max()) < 1e-6
    assert float(ratio) > 1e12
    # and the same targets without the gradient columns give the same residuals
    V0 = T * keep
    G_M0, G_A0, _ = hcurl_grams_plain(V0.float(), batch["Y_field"], batch)
    assert float(span_residual(G_M0, 3).max()) < 1e-6


def test_plain_loss_terms_are_invariant_to_the_scale_of_every_column(pill):
    """span_residual is invariant to column scaling, its dead-column rule is not: in the plain path the V columns are
    scaled to unit mass norm first, so r_M, r_A and the (energy-Gram) ortho do not move when each column is multiplied
    by any factor in 1e-1.2 … 1e1.2 (basis: targets + gradient content + random fields).  Columns below 1e-3 of the
    curl-bearing columns' amplitude are dead by design (test_a_collapsed_column_is_dead_...), so the range is limited."""
    ops, batch, T, G, keep = _plain_setup(pill)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    Gphi = (G @ torch.randn(G.shape[1], 4, dtype=torch.float64, generator=torch.Generator().manual_seed(9)))[None]
    V = _mixed_basis(batch, T, Gphi)
    m = V.shape[-1]
    fac = 10.0 ** torch.tensor([-1.2, 1.2, 1.0, -1.0, 0.5])[:m].double()
    res = []
    for scale in (torch.ones(m, dtype=torch.float64), fac):
        G_M, G_A, ratio = lm._normalise_columns(*hcurl_grams_plain((V * scale).double(), batch["Y_field"], batch)[:2], m)
        res.append((span_residual(G_M, m), span_residual(G_A, m), _basis_conditioning_loss(G_A[:, :m, :m], rel_floor=1e-12),
                    ratio))
    for a, b in zip(res[0][:3], res[1][:3], strict=True):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-9), (a, b)
    print("r_M", res[0][0], "r_A", res[0][1], "ortho", res[0][2], "ratios", float(res[0][3]), float(res[1][3]))
    assert float(res[1][3]) > 10 * float(res[0][3])                  # the scale change moved the ratio, nothing else
    assert (res[0][0] > 1e-3).all() and (res[0][1] < 1e-6).all()      # nontrivial: gradient content costs mass-norm error


def test_a_collapsed_column_is_dead_and_a_live_columns_gradient_is_bounded(pill):
    """Review finding: a column that collapses to 1e-9 of the others' amplitude was blown up to unit norm by the exact
    1/√diag scale and its loss gradient on V grew ~1/‖V_i‖ (stalling gradient-clipped training).  The loss is invariant
    to each column's scale, so a scaling bound alone does not help (the gradient stays ∝ 1/amplitude); the column is
    dead (scale 0) when its mass norm² is below 1e-6 of the curl-bearing columns' largest, which bounds a live
    column's gradient by ~1e3 x a healthy one's.  A gradient-heavy column does not set the reference."""
    ops, batch, T, G, keep = _plain_setup(pill)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    Tn = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    rnd = torch.randn(1, T.shape[1], 3, dtype=torch.float64, generator=torch.Generator().manual_seed(11)) * keep
    rnd = rnd / (rnd * spmm(batch["M"], rnd)).sum(1, keepdim=True).sqrt()

    def column_grads(amp, exact=False):
        V = (torch.cat([Tn[..., :2], rnd[..., :1], amp * rnd[..., 1:2], rnd[..., 2:]], -1) * keep).double().requires_grad_()
        m = V.shape[-1]
        G_M, G_A, _ = hcurl_grams_plain(V, batch["Y_field"], batch)
        if exact:                                              # the earlier exact unit-norm scale
            d = torch.diagonal(G_M, dim1=-2, dim2=-1)[:, :m].rsqrt()
            sc = torch.cat([d, torch.ones(1, G_M.shape[1] - m, dtype=d.dtype)], -1)
            Gn, An = G_M * sc[:, :, None] * sc[:, None, :], G_A * sc[:, :, None] * sc[:, None, :]
        else:
            Gn, An, _ = lm._normalise_columns(G_M, G_A, m)
        lm._span_terms(Gn, An, m, None, None, selfsup=False, ortho="energy").backward()
        return V.grad[0].norm(dim=0)                            # per column [m]
    healthy = column_grads(1.0)
    assert torch.isfinite(healthy).all() and float(healthy.min()) > 0
    for amp in (1.0, 1e-1, 1e-2, 5e-3):                         # live: gradient ∝ 1/amp, bounded by ~1/amp_min
        g = column_grads(amp)
        assert torch.isfinite(g).all() and float(g[3] / healthy[3]) < 1.2 / amp, amp
        assert float(g[[0, 1, 2, 4]].max()) < 10 * float(healthy.max())      # the other columns are not affected
    for amp in (1e-4, 1e-6, 1e-9, 1e-20):                       # collapsed: dead, no gradient (the exact scale: 1/amp)
        g = column_grads(amp)
        assert float(g[3]) == 0.0 and torch.isfinite(g).all(), amp
        assert float(g[[0, 1, 2, 4]].max()) < 10 * float(healthy.max())
    assert float(column_grads(1e-9, exact=True)[3] / healthy[3]) > 1e6
    # the dead column drops out of the span residuals: same r_M, r_A as without it
    V = torch.cat([Tn[..., :2], rnd[..., :1], rnd[..., 2:]], -1) * keep
    V9 = torch.cat([Tn[..., :2], rnd[..., :1], 1e-9 * rnd[..., 1:2], rnd[..., 2:]], -1) * keep
    r = []
    for X in (V, V9):
        G_M, G_A, _ = hcurl_grams_plain(X.double(), batch["Y_field"], batch)
        Gn, An, _ = lm._normalise_columns(G_M, G_A, X.shape[-1])
        r.append(torch.stack([span_residual(Gn, X.shape[-1]), span_residual(An, X.shape[-1])]))
    assert torch.allclose(r[0], r[1], rtol=1e-6, atol=1e-9)


def test_grad_frac_lower_bound_is_a_lower_bound_and_exact_on_the_lowest_mode(pill):
    """grad_frac_lb = max(0, 1 − RQ_i/λ₁) from the unprojected Grams only.  With V_i = T_k + a·g (g a unit-mass gradient,
    M-orthogonal to T_k) the true gradient fraction is a²/(1 + a²); the bound is exact for the lowest mode (RQ = λ₁),
    smaller for higher modes (the bound cannot see the gradient mass behind curl content of RQ > λ₁), and never above the
    true fraction 1 − diag(M_div)/diag(VᵀMV) of the projected basis (random mixed columns, too)."""
    ops, batch, T, G, keep = _plain_setup(pill)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    Tn = T / (T * spmm(batch["M"], T)).sum(1, keepdim=True).sqrt()
    Gphi = (G @ torch.randn(G.shape[1], K_MODES, dtype=torch.float64, generator=torch.Generator().manual_seed(5)))[None]
    Gn = Gphi / (Gphi * spmm(batch["M"], Gphi)).sum(1, keepdim=True).sqrt()
    f = lambda a: a * a / (1 + a * a)                                         # noqa: E731
    for a in (0.0, 0.5, 2.0, 10.0):
        V = (Tn + a * Gn) * keep
        G_M, G_A, _ = hcurl_grams_plain(V, batch["Y_field"], batch)
        lb = lm._grad_frac_lower_bound(G_M, G_A, V.shape[-1])[0]
        proj = project_basis(V, batch, 1e-12)
        true = 1.0 - torch.diagonal(proj["M_div"], dim1=-2, dim2=-1)[0] / torch.diagonal(proj["M_V"], dim1=-2, dim2=-1)[0]
        assert (lb <= true + 1e-6).all(), (a, lb, true)                      # a lower bound (tolerance: the 1e-12 Cholesky tol)
        assert abs(float(lb[0]) - f(a)) < 1e-4, (a, float(lb[0]))              # exact on the lowest mode
        assert (lb <= f(a) + 1e-6).all()
        if a == 2.0:                                                         # a higher mode (RQ > λ₁): a strict bound
            assert float(lb[-1]) < f(a) - 0.01
    # random mixed basis: the bound never exceeds the projected basis's true gradient fraction
    V = _mixed_basis(batch, T, (G @ torch.randn(G.shape[1], 4, dtype=torch.float64, generator=torch.Generator().manual_seed(6)))[None])
    G_M, G_A, _ = hcurl_grams_plain(V, batch["Y_field"], batch)
    lb = lm._grad_frac_lower_bound(G_M, G_A, V.shape[-1])[0]
    proj = project_basis(V, batch, 1e-12)
    true = 1.0 - torch.diagonal(proj["M_div"], dim1=-2, dim2=-1)[0] / torch.diagonal(proj["M_V"], dim1=-2, dim2=-1)[0]
    assert (lb <= true + 1e-6).all() and float(lb.max()) > 0.05 and (lb >= 0).all()
    # a zero column counts 0, a pure-gradient column is ~1
    V = torch.cat([Tn[..., :1], torch.zeros_like(Tn[..., :1]), 3.0 * Gn[..., :1]], -1) * keep
    G_M, G_A, _ = hcurl_grams_plain(V.float(), batch["Y_field"], batch)
    lb = lm._grad_frac_lower_bound(G_M, G_A, 3)[0]
    assert float(lb[0]) < 1e-3 and float(lb[1]) == 0.0 and float(lb[2]) > 0.999


def test_noproj_logs_grad_frac_lb_every_step_for_train_and_val(pill):
    _, _, ops_np, r, Y = pill
    batch = field_collate([_item(ops_np, r, Y)])
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    logs = _capture(lm)
    with torch.no_grad():
        lm._compute_loss_3d_plain(batch, "train")
        lm._compute_loss_3d_plain(batch, "val")
    for pre in ("train", "val"):
        mean, mx = logs[f"{pre}/grad_frac_lb"], logs[f"{pre}/grad_frac_lb_max"]
        assert 0.0 <= mean[0] <= mx[0] <= 1.0 and mean[1] == mx[1] == 1
    assert logs["train/grad_frac_lb"] == logs["val/grad_frac_lb"]


def test_p3_labels_projected_onto_p2_of_the_same_mesh_are_m_orthogonal_to_the_gradients(pill):
    """docs/38 §3.2: the exact M-orthogonal projection of a p3 eigenvector onto the nested p2 space of the SAME mesh
    stays M-orthogonal to the p2 gradients ((g, P e₃) = (g, e₃) = 0); what remains is the dx₂ / dx₃ rule difference
    (measured ~1e-9 at 109 tets, falling with the mesh size; the p2 solve itself: 1e-16)."""
    from src.data_gen.field_labels import project_to_model
    mesh, ops, _, r2, Y2 = pill
    r3 = ho.solve_modes(mesh, K_MODES, order=3)
    U, err, _, _ = project_to_model(mesh, {"fes": r3["fes"], "gfs": r3["gfs"]}, 2, unit_norm=True)
    assert err.max() > 1e-3                                       # a real projection, not the p2 modes again

    def e(U):
        U = np.asarray(U, dtype=np.float64).copy()
        U[ops["bnd"]] = 0.0
        MT = ops["M"] @ U
        return np.linalg.norm(ops["G"].T @ MT, axis=0) / np.linalg.norm(MT, axis=0)
    assert e(Y2).max() < 1e-12 and e(U).max() < 1e-7


# ── guards ──────────────────────────────────────────────────────────────────────────────────────────

def test_selfsup_or_freq_weight_with_projection_off_is_an_error():
    kw = dict(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K_MODES, rff_dim=8,
              eigenspace_kwargs={"n_layers": 1}, projection=False)
    for bad in ({}, {"selfsup_weight": 0.01, "freq_weight": 0.0}, {"selfsup_weight": 0.0, "freq_weight": 0.1}):
        with pytest.raises(ValueError, match="selfsup_weight = 0 and freq_weight = 0"):
            CavityLightning(**{**kw, **bad})          # defaults: selfsup 0.01, freq 0.1
    lm = CavityLightning(**kw, selfsup_weight=0.0, freq_weight=0.0)
    assert lm.projection is False and lm.hparams.projection is False
    with pytest.raises(ValueError, match="true or false"):
        CavityLightning(val_dim=9, projection="no")


def test_span_norm_energy_with_projection_off_is_an_error():
    """The energy norm and the energy-Gram ortho are blind to gradients: nothing would keep V free of them."""
    kw = dict(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K_MODES, rff_dim=8,
              eigenspace_kwargs={"n_layers": 1}, selfsup_weight=0.0, freq_weight=0.0)
    with pytest.raises(ValueError, match="span_norm='energy'"):
        CavityLightning(projection=False, span_norm="energy", **kw)
    for ok in ("mass", "both"):
        assert CavityLightning(projection=False, span_norm=ok, **kw).span_norm == ok
    CavityLightning(projection=True, span_norm="energy", **{**kw, "selfsup_weight": 0.01, "freq_weight": 0.1})   # projected: allowed


def test_test_step_of_a_projection_free_run_runs_the_trunk_once(pill, monkeypatch):
    """test/loss_plain reuses the basis of the projected pass: one trunk evaluation per test step, not two."""
    full = field_collate([_item(pill[1], pill[3], pill[4])])
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0)
    logs = _capture(lm)
    calls = []
    orig = lm.model.field_basis
    monkeypatch.setattr(lm.model, "field_basis", lambda b: calls.append(1) or orig(b))
    with torch.no_grad():
        lm.test_step(full, 0)
        assert len(calls) == 1
        plain = lm._compute_loss_3d_plain(full, None)                       # alone (no basis handed in): runs the trunk itself
    assert len(calls) == 2
    assert abs(logs["test/loss_plain"][0] - float(plain)) < 1e-6 * float(plain)


def test_resume_with_another_projection_is_an_error(tmp_path, monkeypatch):
    import train
    ckpt = tmp_path / "last.ckpt"
    torch.save({"hyper_parameters": {"projection": True}}, ckpt)
    old = tmp_path / "old.ckpt"
    torch.save({"hyper_parameters": {}}, old)                                # before the option: projection true
    train._check_projection_matches(str(ckpt), True)
    train._check_projection_matches(str(old), True)
    with pytest.raises(ValueError, match="projection=True, but training.projection=False"):
        train._check_projection_matches(str(ckpt), False)
    with pytest.raises(ValueError, match="projection=True, but training.projection=False"):
        train._check_projection_matches(str(old), False)
    torch.save({"hyper_parameters": {"projection": False}}, tmp_path / "np.ckpt")
    train._check_projection_matches(str(tmp_path / "np.ckpt"), False)
    with pytest.raises(ValueError, match="projection=False, but training.projection=True"):
        train._check_projection_matches(str(tmp_path / "np.ckpt"), True)
    # and through main(): the error comes before any dataset is built
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/field_3d_noproj.yaml", "--resume", str(ckpt)])
    with pytest.raises(ValueError, match="--init_from"):
        train.main()


def test_checkpoint_without_the_key_loads_with_projection(tmp_path):
    """Checkpoints written before the option have no hparams['projection']: they load as projection=True; one with
    projection=False keeps it."""
    lm = _module()
    for projection in (None, False):
        hp = dict(lm.hparams)
        hp.pop("projection")
        if projection is False:
            hp.update(projection=False, selfsup_weight=0.0, freq_weight=0.0)
        path = tmp_path / f"c_{projection}.ckpt"
        torch.save({"state_dict": lm.state_dict(), "hyper_parameters": hp, "pytorch-lightning_version": "2.0.0"}, path)
        loaded = CavityLightning.load_from_checkpoint(str(path), map_location="cpu")
        assert loaded.projection is (projection is None)
        assert loaded.hparams.projection is (projection is None)


def test_noproj_config_differs_from_the_default_only_in_the_projection_keys():
    from src.config import load_config
    a, b = load_config("configs/field_3d.yaml"), load_config("configs/field_3d_noproj.yaml")
    diff = {f"{sec}.{k}" for sec in ("dataset", "model", "training") for k in {*a[sec], *b[sec]}
            if a[sec].get(k, "<absent>") != b[sec].get(k, "<absent>")}
    assert diff == {"training.projection", "training.selfsup_weight", "training.freq_weight", "training.exp_name"}
    assert a.training.projection is True and b.training.projection is False
    assert b.training.selfsup_weight == 0 and b.training.freq_weight == 0


def test_train_py_rejects_non_boolean_projection_and_nonzero_weights(monkeypatch):
    import train
    monkeypatch.setattr(sys, "argv", ["train.py", "--override", "training.projection=no"])
    with pytest.raises(ValueError, match="training.projection must be true or false"):
        train.main()
