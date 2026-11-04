"""Field labels (docs/29): generator → H5 → FieldDataset → field3d model, and the
TRUBA one-line orchestrator (run.sh --dry-run)."""
import json
import os
import pickle
import subprocess

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")
from src.data.field_dataset import FieldDataset, field_collate  # noqa: E402
from src.data_gen import cavity_shapes as cs  # noqa: E402
from src.data_gen import dataset_generator_3d as gen  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 4


@pytest.fixture(scope="module")
def field_h5(tmp_path_factory):
    """Calibration box + pillbox with field labels (p3 → p2), written like the generator does."""
    path = tmp_path_factory.mktemp("field") / "calib.h5"
    old_args = gen.ARGS
    gen.ARGS = gen.parse_args(["--mode", "calibration", "--mesh_size", "0.2",
                               "--n_eigen_modes", str(K), "--tol_f", "1e-4", "--tol_q", "3e-2",
                               "--max_ndof", "40000", "--model_max_elements", "60", "--label_order", "3"])
    gen._init_worker(gen.ARGS)
    try:
        with h5py.File(path, "w") as f:
            f.attrs["metadata"] = json.dumps(gen.file_metadata(gen.ARGS), default=str)
            for s_id in range(4):                         # even: box, odd: pillbox
                res = gen.generate_sample_data(s_id)
                assert res is not None and res["labels"] == "field"
                gen.write_sample(f, res)
    finally:
        gen.ARGS, cs.MIN_FILLET = old_args, 0.0          # module state other tests rely on
    return path


def test_generator_defaults():
    """Field labels are the only label type: fillets, healing and the long sample timeout are the defaults, the
    discrete freeform family stays in the parsed list (main() drops it); --no_* switch them off."""
    a = gen.parse_args([])
    assert a.min_fillet == 0.05 and a.heal and a.round_edges and a.sample_timeout == 3600 and "freeform" in a.families
    b = gen.parse_args(["--no_heal", "--no_round_edges", "--min_fillet", "0"])
    assert b.min_fillet == 0 and not b.heal and not b.round_edges
    # the fillet floor only ever enlarges a radius, up to the builder's own cap
    cs.MIN_FILLET = 0.05
    try:
        assert cs.fillet_floor(0.001, 0.1) == pytest.approx(0.005)
        assert cs.fillet_floor(0.001, 0.1, cap=0.003) == pytest.approx(0.003)
        assert cs.fillet_floor(0.02, 0.1) == pytest.approx(0.02)
    finally:
        cs.MIN_FILLET = 0.0


def test_field_samples_are_high_order_and_consistent(field_h5):
    with h5py.File(field_h5, "r") as f:
        g = f["sample_0001"]                                          # calibration pillbox
        assert g.attrs["labels"] == "field" and g.attrs["label_order"] == 3 and g.attrs["model_order"] == 2
        f_ref = g.attrs["freqs_analytic"]
        assert abs(g["freqs"][0] / f_ref[0] - 1) < 1e-4            # TM010 from p3 on the curved mesh
        assert np.all(np.asarray(g["proj_err"]) < 0.1)
        assert g["u_model"].shape == (int(g.attrs["n_dof_model"]), K)
        assert g["qoi"].shape == (K, len(str(g.attrs["qoi_names"]).split(",")))
        assert float(g["qoi"][0, 0]) > 1e3                          # Q0 of the copper TM010 cell
        assert g["qfeat_wall"].shape[0] == int(g.attrs["n_tets"]) and g["qfeat_wall"].shape[2] == 5
        assert str(g.attrs["qfeat_columns"]) == "dist_to_boundary,dir_bnd_x,dir_bnd_y,dir_bnd_z,torsion"


def test_stored_token_features_equal_the_computed_ones(field_h5, tmp_path):
    """A file with qfeat_wall (generator) and the same file without it (older data, loader computes them)
    give the same items bit for bit; scripts/add_field_features.py restores the stored array."""
    from src.data_gen.field_labels import add_wall_features
    old = tmp_path / "old.h5"
    with h5py.File(field_h5, "r") as src, h5py.File(old, "w") as dst:
        for k in src:
            src.copy(src[k], dst, name=k)
        for k in [k for k in dst if k.startswith("sample_")]:
            del dst[k]["qfeat_wall"]
    ds_new = FieldDataset(str(field_h5), split="train", train_ratio=1.0, val_ratio=0.0)
    ds_old = FieldDataset(str(old), split="train", train_ratio=1.0, val_ratio=0.0)
    for i in range(len(ds_new)):
        a, b = ds_new[i], ds_old[i]
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))
        assert all((a[k] != b[k]).nnz == 0 for k in ("K", "M", "G", "Kp", "C"))
    assert add_wall_features(str(old), log=lambda *a: None) == len(ds_new)
    assert add_wall_features(str(old), log=lambda *a: None) == 0               # already there: untouched
    with h5py.File(field_h5, "r") as f1, h5py.File(old, "r") as f2:
        for k in f1:
            if k.startswith("sample_"):
                assert np.array_equal(f1[k]["qfeat_wall"], f2[k]["qfeat_wall"])


def test_field_dataset_rebuilds_the_labelled_mesh(field_h5, tmp_path):
    ds = FieldDataset(str(field_h5), split="train", train_ratio=0.5, val_ratio=0.25)
    va = FieldDataset(str(field_h5), split="val", train_ratio=0.5, val_ratio=0.25)
    assert len(ds) == 2 and len(va) == 1 and ds.stats == va.stats  # val is z-scored with the train stats
    items = [ds[i] for i in range(len(ds))]
    b = field_collate(items)
    for k in ("Y_field", "Y_freq", "FreqNext", "geom_id", "Y_qoi", "C", "QP", "DofMask"):
        assert k in b, k
    from src.models.eigenspace_operator_field import EigenspaceOperatorField
    torch.manual_seed(0)
    m = EigenspaceOperatorField(val_dim=ds.data_dims()[0], embed_dim=16, n_layers=1, n_heads=2, n_basis=8,
                                num_field_modes=K, rff_dim=8)
    m.freq_stats = ds.stats
    out = m(b)
    assert out["freq"].shape == (2, K) and out["basis"].shape[1] == b["DofMask"].shape[1]
    # labels that no longer belong to the stored mesh are rejected
    bad = tmp_path / "bad.h5"
    with h5py.File(field_h5, "r") as src, h5py.File(bad, "w") as dst:
        for k in src:
            src.copy(src[k], dst, name=k)
        u = dst["sample_0001/u_model"]
        u[...] = np.roll(np.asarray(u), 7, axis=0)
    ds_bad = FieldDataset(str(bad), split="train", train_ratio=1.0, val_ratio=0.0)
    i = ds_bad.active_geoms.index(1)
    with pytest.raises(ValueError, match="do not match the rebuilt operators"):
        ds_bad[i]


def test_fast_dev_run_train_py(field_h5, tmp_path, monkeypatch):
    """train.py end to end on field-label H5 with its default config (configs/field_3d.yaml)."""
    import sys

    import train
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--fast_dev_run", "--override",
        f"dataset.data_path={field_h5}", "dataset.train_ratio=0.5", "dataset.val_ratio=0.25",
        "model.embed_dim=16", "model.n_heads=2", "model.n_basis=8", f"model.num_field_modes={K}", "model.rff_dim=8",
        "model.eigenspace.n_layers=1", "model.eigenspace.grad_checkpoint=false", "training.batch_size=1",
        "training.num_workers=0", f"training.log_dir={tmp_path}"])
    train.main()


_SMALL = ["model.embed_dim=16", "model.n_heads=2", "model.n_basis=8", f"model.num_field_modes={K}", "model.rff_dim=8",
          "model.eigenspace.n_layers=1", "model.eigenspace.grad_checkpoint=false", "training.batch_size=1",
          "training.num_workers=0"]


def test_noproj_fast_dev_run_train_py(field_h5, tmp_path, monkeypatch):
    """train.py with configs/field_3d_noproj.yaml (training.projection: false): the train AND val batches have no
    G / Kp and no Kp solve runs in their steps; only the test step projects (docs/38)."""
    import sys

    import train
    from src.models import hcurl
    from src.training.lightning_module import CavityLightning
    seen = {"batch_keys": {}, "proj": [], "plain": [], "cg_by_stage": {"train": 0, "val": 0, "test": 0}, "stage": None}
    orig_pcg = hcurl._pcg

    def stage(name, orig):
        def wrapped(self, batch, batch_idx):
            seen["stage"] = name
            seen["batch_keys"][name] = set(batch)
            return orig(self, batch, batch_idx)
        return wrapped

    def spy_proj(orig):
        def wrapped(self, batch, prefix, *a):
            seen["proj"].append(prefix)
            return orig(self, batch, prefix, *a)
        return wrapped

    def spy_plain(orig):
        def wrapped(self, batch, prefix="train", *a, **kw):
            seen["plain"].append(prefix)
            return orig(self, batch, prefix, *a, **kw)
        return wrapped

    def spy_pcg(*a, **k):
        seen["cg_by_stage"][seen["stage"]] += 1
        return orig_pcg(*a, **k)

    for name, attr in (("train", "training_step"), ("val", "validation_step"), ("test", "test_step")):
        monkeypatch.setattr(CavityLightning, attr, stage(name, getattr(CavityLightning, attr)))
    monkeypatch.setattr(CavityLightning, "_compute_loss_3d", spy_proj(CavityLightning._compute_loss_3d))
    monkeypatch.setattr(CavityLightning, "_compute_loss_3d_plain", spy_plain(CavityLightning._compute_loss_3d_plain))
    monkeypatch.setattr(hcurl, "_pcg", spy_pcg)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/field_3d_noproj.yaml", "--fast_dev_run", "--override",
        f"dataset.data_path={field_h5}", "dataset.train_ratio=0.5", "dataset.val_ratio=0.25", *_SMALL,
        f"training.log_dir={tmp_path}"])
    with pytest.warns(UserWarning, match="not eigenvectors of the model space"):      # the fixture's labels were transferred between meshes
        train.main()
    gram_keys = {"G", "Gt", "Kp", "Kp_diag"}
    assert set(seen["batch_keys"]) == {"train", "val", "test"}
    assert not (gram_keys & seen["batch_keys"]["train"]) and not (gram_keys & seen["batch_keys"]["val"])
    assert gram_keys <= seen["batch_keys"]["test"] and "Ct" in seen["batch_keys"]["train"]
    assert seen["cg_by_stage"]["train"] == 0 and seen["cg_by_stage"]["val"] == 0 and seen["cg_by_stage"]["test"] > 0
    assert seen["proj"] == ["test"]                                      # the Ritz loss runs only for the test split
    assert set(seen["plain"]) == {"train", "val", None}                  # None: test/loss_plain


def test_noproj_train_py_trains_selects_by_val_span_loss_and_tests_projected(field_h5, tmp_path, monkeypatch):
    """train.py end to end, 2 epochs, projection-free: the history has the like-for-like train/val keys and no
    Ritz key, the checkpoint is chosen by val/span_loss (file name, progress.json, resolve_checkpoint), and the final
    test of the best checkpoint is projected (Ritz metrics + test/loss_plain)."""
    import sys

    import train
    from src.training.checkpoint import resolve_checkpoint
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/field_3d_noproj.yaml", "--override",
        f"dataset.data_path={field_h5}", "dataset.train_ratio=0.5", "dataset.val_ratio=0.25", *_SMALL,
        f"training.log_dir={tmp_path}", "training.exp_name=np", "training.max_epochs=2", "training.patience=5"])
    with pytest.warns(UserWarning, match="not eigenvectors of the model space"):
        train.main()
    run = tmp_path / "np"
    prog = json.load(open(run / "progress.json"))
    assert prog["status"] == "finished" and len(prog["history"]) == 2
    h = prog["history"][-1]
    for pre in ("train", "val"):
        for key in ("loss", "span_loss", "ortho_loss", "span_rel_l2", "span_rel_l2_energy", "mass_diag_ratio", "grad_frac_lb",
                    *(f"span_mode_{k}_rel_l2" for k in range(K))):
            assert f"{pre}/{key}" in h, f"{pre}/{key}"
    assert not any(k.endswith(("field_rel_l2", "freq_rel_err", "grad_frac", "kp_cg_iters", "r2", "selfsup_loss"))
                   for k in h if k.split("/")[0] in ("train", "val"))
    best = prog["best"]
    assert best["monitor"] == "val/span_loss" and os.path.basename(best["path"]).startswith("best-epoch=")
    assert "-val_span_loss=" in best["path"] and "val_rel_l2" not in best["path"] and (run / "last.ckpt").exists()
    assert resolve_checkpoint(str(run)) == best["path"]
    assert abs(best["score"] - min(e["val/span_loss"] for e in prog["history"])) < 1e-9
    assert f"val_span_loss={best['score']:.4f}" in best["path"]
    t = prog["test"]                                                    # projected: the real quality
    for key in ("test/field_rel_l2", "test/freq_rel_err", "test/freq_mae_ghz", "test/grad_frac", "test/kp_cg_iters",
                "test/mode_0_rel_l2", "test/loss", "test/loss_plain", "test/span_rel_l2"):
        assert key in t, key
    assert t["test/kp_cg_iters"] > 0 and t["test/loss_plain"] > 0 and t["test/loss"] > 0


def test_dataset_without_potentials_cache_and_label_warning(field_h5, tmp_path):
    """FieldDataset(build_potential=False): items / batches without G, Kp; the operator cache of the two variants
    never mixes (a pickle without G, Kp is not read by a dataset that needs them; a full pickle serves both);
    the fixture's labels (p3 → p2 transferred between meshes: n_tets_label != n_tets) trigger the approximate-orthogonality warning, and only for the
    projection-free train and val splits."""
    import warnings

    from src.data.field_dataset import labels_are_model_eigenvectors
    kw = dict(train_ratio=0.5, val_ratio=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = FieldDataset(str(field_h5), split="train", **kw)                       # projection on: no warning
        FieldDataset(str(field_h5), split="test", build_potential=False, **kw)        # warns for train and val only
    with pytest.warns(UserWarning, match=r"1 of 1 val geometries have labels that are not eigenvectors"):
        FieldDataset(str(field_h5), split="val", build_potential=False, **kw)
    with pytest.warns(UserWarning, match=r"2 of 2 train geometries have labels that are not eigenvectors"):
        red = FieldDataset(str(field_h5), split="train", build_potential=False, **kw)
    assert red.active_geoms == full.active_geoms and red.stats == full.stats
    a, b = full[0], red[0]
    assert "G" in a and "Kp" in a and "G" not in b and "Kp" not in b
    for k in ("M", "K", "C"):
        assert (a[k] != b[k]).nnz == 0
    for k in ("X", "Input_funcs", "Y_field", "Y_freq", "QP", "BndDof"):
        assert torch.equal(a[k], b[k]), k
    assert not ({"G", "Gt", "Kp", "Kp_diag"} & set(field_collate([red[0], red[1]])))
    assert {"G", "Gt", "Kp", "Kp_diag"} <= set(field_collate([full[0], full[1]]))

    # cache, reduced variant first: its own file; the full variant rebuilds instead of reading it
    c1 = tmp_path / "c1"
    with pytest.warns(UserWarning):
        red1 = FieldDataset(str(field_h5), split="train", build_potential=False, cache_dir=str(c1), **kw)
    full1 = FieldDataset(str(field_h5), split="train", cache_dir=str(c1), **kw)
    g0 = red1.active_geoms[0]
    key = red1.index[g0][1]
    nopot, fullname = f"calib__{key}.nopot.ops2.pkl", f"calib__{key}.ops2.pkl"
    assert "G" not in red1[0]
    assert {p.name for p in c1.iterdir()} == {nopot}
    assert "G" in full1[0] and "Kp" in full1[0]                                         # not served the reduced pickle
    assert {p.name for p in c1.iterdir()} == {nopot, fullname}
    # items from the cache equal fresh ones: bit for bit for the full variant, to float32 rounding for the reduced one
    fresh_full = FieldDataset(str(field_h5), split="train", **kw)
    fresh_red = FieldDataset(str(field_h5), split="train", build_potential=False, **kw)
    for a, b in ((full1[0], fresh_full[0]),):                                            # cache file against no cache at all
        assert set(a) == set(b)
        for k in a:
            if sp.issparse(a[k]):
                assert a[k].shape == b[k].shape and np.array_equal(a[k].indptr, b[k].indptr)
                assert np.array_equal(a[k].indices, b[k].indices) and np.array_equal(a[k].data, b[k].data), k
            elif torch.is_tensor(a[k]):
                assert a[k].dtype == b[k].dtype and torch.allclose(a[k], b[k], rtol=0, atol=0, equal_nan=True), k
    ra, rb = red1[0], fresh_red[0]
    assert set(ra) == set(rb) and "G" not in ra
    for k in ("M", "K", "C", "Ct"):
        assert ra[k].shape == rb[k].shape and np.array_equal(ra[k].indices, rb[k].indices)
        assert np.abs(ra[k].data - rb[k].data).max() <= 6e-8 * np.abs(rb[k].data).max()
    assert (ra["K"] != ra["K"].T).nnz == 0 and (ra["M"] != ra["M"].T).nnz == 0
    for k in ("X", "Input_funcs", "Area", "QX", "QP", "Y_field", "Y_freq", "BndDof"):
        assert torch.equal(ra[k], rb[k]), k                                              # float32 items: no loss at all
    # a file of the old format (".ops.pkl") or of another version is never read; an unreadable one is rebuilt
    c4 = tmp_path / "c4"
    c4.mkdir()
    (c4 / f"calib__{key}.ops.pkl").write_bytes(b"not a pickle")
    (c4 / nopot).write_bytes(pickle.dumps({"format": 1, "stale": True}))
    with pytest.warns(UserWarning, match="another format"):
        FieldDataset(str(field_h5), split="train", build_potential=False, cache_dir=str(c4), **kw)[0]
    assert pickle.loads((c4 / nopot).read_bytes())["format"] == 2                                 # ... and replaced
    (c4 / nopot).write_bytes((c4 / nopot).read_bytes()[:500])
    with pytest.warns(UserWarning, match="unreadable"):
        FieldDataset(str(field_h5), split="train", build_potential=False, cache_dir=str(c4), **kw)[0]
    assert not [p for p in c4.iterdir() if p.name.endswith(".tmp")]
    assert "G" in FieldDataset(str(field_h5), split="train", cache_dir=str(c4), **kw)[0]
    assert (c4 / f"calib__{key}.ops.pkl").read_bytes() == b"not a pickle" and (c4 / fullname).exists()   # old name: untouched
    # full pickle first: the reduced variant reuses it (no .nopot file), G / Kp dropped
    c2 = tmp_path / "c2"
    full2 = FieldDataset(str(field_h5), split="train", cache_dir=str(c2), **kw)
    full2[0]
    with pytest.warns(UserWarning):
        red2 = FieldDataset(str(field_h5), split="train", build_potential=False, cache_dir=str(c2), **kw)
    it = red2[0]
    assert "G" not in it and "Kp" not in it and len(list(c2.iterdir())) == 1
    # a reduced pickle under the full name (copied by hand) is refused, not served
    c3 = tmp_path / "c3"
    c3.mkdir()
    (c3 / fullname).write_bytes((c1 / nopot).read_bytes())
    bad = FieldDataset(str(field_h5), split="train", cache_dir=str(c3), **kw)
    with pytest.raises(ValueError, match="without the kernel potentials"):
        bad[0]

    assert labels_are_model_eigenvectors({"label_order": 2, "model_order": 2, "n_tets": 100, "n_tets_label": 100})
    assert labels_are_model_eigenvectors({"label_order": 2, "model_order": 2})            # attrs of old files absent
    # p3 labels projected onto p2 of the SAME mesh: exact nested M-orthogonal projection (docs/38 §3.2) → no warning
    assert labels_are_model_eigenvectors({"label_order": 3, "model_order": 2, "n_tets": 100, "n_tets_label": 100})
    # transferred from another mesh, or of a lower order than the model space: only approximately orthogonal
    assert not labels_are_model_eigenvectors({"label_order": 2, "model_order": 2, "n_tets": 100, "n_tets_label": 400})
    assert not labels_are_model_eigenvectors({"label_order": 3, "model_order": 2, "n_tets": 100, "n_tets_label": 400})
    assert not labels_are_model_eigenvectors({"label_order": 2, "model_order": 3, "n_tets": 100, "n_tets_label": 100})


def test_field_datasets_in_the_ui_browser(field_h5):
    """The web UI lists field-label H5 as kind 'field' with the label quality, and opens a geometry on its
    model mesh: the label fields as stored (HCurl p2 DOFs, drawn directly — no Whitney N0 projection), with
    unit energy and the frequencies from their own Rayleigh quotient, and the stored p3 figures of merit."""
    from ngsolve import BilinearForm, HCurl, curl

    from src.data_gen import highorder as ho
    from src.service.datasets import FieldH5Dataset, discover
    ds = discover(str(field_h5))
    d = next(iter(ds.values()))
    assert isinstance(d, FieldH5Dataset) and d.kind == "field"
    rows = d.rows()
    assert len(rows) == 4 and {r["family"] for r in rows} == {"calib_box", "calib_pillbox"}
    r1 = next(r for r in rows if r["id"] == 1)
    assert r1["n_dof"] > 0 and r1["n_tets"] > 0 and r1["n_nodes"] > 0
    assert r1["field"]["adapt_stop"] in ("converged", "budget", "max_iter") and r1["field"]["adapt_iters"] >= 1
    rec = d.open(1)                                                   # calibration pillbox
    U, mesh = rec["U"], rec["mesh"]
    assert U.shape == (rec["n_dof"], K) == (r1["n_dof"], K) and np.isfinite(U).all()   # raw HCurl DOFs, no N0
    assert rec["Mp"].shape == (rec["n_dof"], rec["n_dof"]) and rec["order"] == 2
    assert set(rec["curved"]) >= {"points", "triangles", "vid3", "w3", "volume_m3", "bbox_m"}
    fes = HCurl(mesh, order=rec["order"], dirichlet="wall")
    u, v = fes.TnT()
    Kp = ho._tosp(BilinearForm(curl(u) * curl(v) * ho.volume_measure(rec["order"]), symmetric=True).Assemble().mat)
    lam = np.einsum("ik,ik->k", U, Kp @ U) / np.einsum("ik,ik->k", U, rec["Mp"] @ U)
    f_p2 = np.sqrt(lam) * 299792458.0 / (2 * np.pi) / 1e9
    # the stored p2 field is the label mode (a sign / DOF-order error would put it far off; the 57-tet model mesh
    # of the fixture costs the p2 projection of the p3 mode a few percent)
    assert np.all(f_p2 >= rec["f_GHz"] * (1 - 1e-3)) and np.all(np.abs(f_p2 / rec["f_GHz"] - 1) < 0.1)
    assert rec["qoi_labels"][0]["Q0"] > 1e3                           # stored p3 label of the copper TM010
    # the API shows the stored p3 figures of merit
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.service import api as A
    from src.service.model import ModelService
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={"f": d}))
    did = c.get("/api/datasets").json()[0]
    assert did["kind"] == "field"
    items = c.get(f"/api/datasets/{did['id']}/items").json()["items"]
    assert items[0]["field"]["n_tets_label"] >= items[0]["n_tets"] > 0
    out = c.post(f"/api/datasets/{did['id']}/items/1/open").json()
    assert out["truth"]["qoi_source"] == "p3_labels" and out["truth"]["n_dof"] == rec["n_dof"]
    assert out["truth"]["modes"][0]["Q0"] == pytest.approx(rec["qoi_labels"][0]["Q0"])
    assert out["check"]["volume_cm3"] == pytest.approx(rec["curved"]["volume_m3"] * 1e6)    # the curved solid's volume


def test_truba_one_line_run_dry_run(tmp_path):
    env = dict(os.environ, DATA_ROOT=str(tmp_path), CONTAINER=str(tmp_path / "none.sif"), USER="t")
    r = subprocess.run(["bash", "cluster/truba/run.sh", "all", "LABELS=field", "FAMILIES=elliptical,box",
                        "N_PER_FAMILY=1024", "EXP=f1", "THREADS=4", "--dry-run"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    log = r.stdout + r.stderr
    sb = [ln for ln in log.splitlines() if ln.startswith("[dry-run] sbatch")]
    assert len(sb) == 3                                     # 2 generation arrays (no conversion) + training
    assert "gen_shard.sbatch" in sb[0] and "gen_shard.sbatch" in sb[1]
    assert "--dependency=afterany:DRY1:DRY2" in sb[2] and "train.sbatch" in sb[2]
    assert "data field H5 of: elliptical" in log                   # the OOD family never trains the model
    r = subprocess.run(["bash", "cluster/truba/run.sh", "train", "NOT-A-VAR=1", "--dry-run"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


def test_multi_fidelity_items_split_and_loss_weights(field_h5, tmp_path):
    """docs/34: Fidelity comes from the H5 attrs (adaptive / p3 labels = HF, label_order == model_order without
    adaptivity = LF), train_extra geometries only ever go to the train split, hf_repeat lists the HF ones more
    often, and the per-sample loss weights scale the span term per sample."""
    from src.data.field_dataset import HF, LF, label_fidelity
    from src.training.lightning_module import CavityLightning
    kw = dict(train_ratio=0.5, val_ratio=0.25)
    ds = FieldDataset(str(field_h5), split="train", **kw)
    assert set(ds.fidelities) == {HF} and int(ds[0]["Fidelity"]) == HF
    assert label_fidelity({"label_order": 2, "model_order": 2}) == LF
    assert label_fidelity({"label_order": 2, "model_order": 2, "adapt_iters": 1}) == HF
    assert label_fidelity({"label_order": 3, "model_order": 2}) == HF
    extra = tmp_path / "extra.h5"
    with h5py.File(field_h5, "r") as src, h5py.File(extra, "w") as dst:
        for k in src:
            src.copy(src[k], dst, name=k)
    tr = FieldDataset(str(field_h5), split="train", train_extra=str(extra), **kw)
    va = FieldDataset(str(field_h5), split="val", train_extra=str(extra), **kw)
    te = FieldDataset(str(field_h5), split="test", train_extra=str(extra), **kw)
    assert len(tr) == len(ds) + 4 and len(va) == 1 and len(te) == 1
    assert all(g >= 4 for g in tr.active_geoms[len(ds):]) and all(g < 4 for g in va.active_geoms + te.active_geoms)
    rep = FieldDataset(str(field_h5), split="train", hf_repeat=3, **kw)
    assert len(rep) == 3 * len(ds)
    assert int(field_collate([ds[0], ds[1]])["Fidelity"].sum()) == 2
    # per-sample weights of the span term: a zero weight removes a sample, equal weights change nothing
    mod = CavityLightning(val_dim=9, num_field_modes=2, model_type="field3d", selfsup_weight=0.0, ortho_weight=0.0)
    torch.manual_seed(0)
    m, n = 3, 2
    A = torch.randn(2, m + n, m + n, dtype=torch.float64)
    G_M = A @ A.transpose(1, 2) + torch.eye(m + n, dtype=torch.float64)
    G_A = G_M * 0.5 + torch.eye(m + n, dtype=torch.float64)
    plain = mod._span_terms(G_M, G_A, m, "train")
    one = mod._span_terms(G_M, G_A, m, "train", torch.ones(2, dtype=torch.float64))
    assert torch.allclose(plain, one)
    only0 = mod._span_terms(G_M[:1], G_A[:1], m, "train")
    w0 = mod._span_terms(G_M, G_A, m, "train", torch.tensor([2.0, 0.0], dtype=torch.float64))
    assert torch.allclose(w0, only0)                       # mean_b(w_b l_b) = (2 l_0 + 0) / 2


def test_cache_compress_policies_give_the_same_items_and_train_py_runs_with_the_throughput_options(field_h5, tmp_path, monkeypatch):
    """dataset.cache_compress true | 'fast' | false: the cache files differ in size and codec, the items read back are bit for bit
    the same; a 2-epoch projection-free train.py run with the throughput options (diagnostic metrics every 2nd batch,
    prefetch_factor, non-persistent val workers, uncompressed cache, step profile) trains, logs the diagnostics on the
    epoch level, and writes the profile (docs/38 §11)."""
    import sys
    import warnings

    import train
    from src.data import ops_cache
    kw = dict(train_ratio=0.5, val_ratio=0.25, build_potential=False)
    items = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pol in (True, "fast", False):
            d = tmp_path / f"c_{pol}"
            ds = FieldDataset(str(field_h5), split="train", cache_dir=str(d), cache_compress=pol, **kw)
            ds[0]                                                       # writes the file (and returns its unpacked form)
            items[pol] = (ds[0], sorted(d.glob("*.nopot.ops2.pkl"))[0])  # second call: read from the file
    sizes = {pol: p.stat().st_size for pol, (_, p) in items.items()}
    assert sizes[False] > sizes["fast"] > sizes[True]
    codecs = {pol: {a["codec"] for q in ops_cache.read(str(p))["ops"].values() if q["kind"] == "csr"
                    for a in (q["indptr"], q["indices"], q["data"])} for pol, (_, p) in items.items()}
    assert codecs[False] == {"raw"} and "zlib" in codecs[True]
    ref = items[True][0]
    for pol in ("fast", False):
        it = items[pol][0]
        assert it.keys() == ref.keys()
        for k in ("M", "K", "C", "Ct"):
            for attr in ("indptr", "indices", "data"):
                assert np.array_equal(getattr(it[k], attr), getattr(ref[k], attr)), (pol, k, attr)
        for k in ("X", "Input_funcs", "Y_field", "Y_freq", "QP", "QX", "Area"):
            assert torch.equal(it[k], ref[k]), (pol, k)

    cache = tmp_path / "ops"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/field_3d_noproj.yaml", "--override",
        f"dataset.data_path={field_h5}", "dataset.train_ratio=0.5", "dataset.val_ratio=0.25", f"dataset.cache_dir={cache}",
        "dataset.cache_compress=false", *_SMALL, f"training.log_dir={tmp_path}", "training.exp_name=thr",
        "training.max_epochs=2", "training.patience=5", "training.diag_metrics_every_n_steps=2", "training.prefetch_factor=1",
        "training.val_persistent_workers=false", "training.num_workers=1", "training.val_num_workers=0",
        "training.profile_steps=2", "training.progress_bar_refresh_rate=10", "training.gpu_prefetch=true"])
    with pytest.warns(UserWarning, match="not eigenvectors of the model space"):
        train.main()
    run = tmp_path / "thr"
    prog = json.load(open(run / "progress.json"))
    assert prog["status"] == "finished" and len(prog["history"]) == 2
    h = prog["history"][-1]
    for key in ("train/loss", "train/span_loss", "train/ortho_loss", "train/span_rel_l2", "train/mass_diag_ratio",
                "train/grad_frac_lb", "val/span_rel_l2", "val/mass_diag_ratio", "val/grad_frac_lb", "val/span_loss"):
        assert key in h, key
    prof = json.load(open(run / "step_profile.json"))
    assert prof["n_batches"] >= 1 and prof["stages"]["trunk"]["mean"] > 0 and prof["gpu_prefetch"] is False   # no CUDA here
    files = list(cache.glob("*.ops2.pkl"))
    assert files and all(a["codec"] == "raw" for f in files for q in ops_cache.read(str(f))["ops"].values()
                         if q["kind"] == "csr" for a in (q["indptr"], q["indices"], q["data"]))
