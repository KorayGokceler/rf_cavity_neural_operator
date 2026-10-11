"""Field labels (docs/29): generator --labels field → H5 → FieldDataset → field3d model, and the
TRUBA one-line orchestrator (run.sh --dry-run)."""
import json
import os
import subprocess

import h5py
import numpy as np
import pytest
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
    gen.ARGS = gen.parse_args(["--labels", "field", "--mode", "calibration", "--mesh_size", "0.2",
                               "--n_eigen_modes", str(K), "--tol_f", "1e-4", "--tol_q", "3e-2",
                               "--max_ndof", "40000", "--model_max_elements", "60"])
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


def test_generator_defaults_per_label_kind():
    a = gen.parse_args(["--labels", "field"])
    assert a.min_fillet == 0.05 and a.heal and a.sample_timeout == 3600 and "freeform" not in a.families
    b = gen.parse_args([])
    assert b.min_fillet == 0.0 and not b.heal and b.sample_timeout == 300 and "freeform" in b.families
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


def test_field_datasets_in_the_ui_browser(field_h5):
    """The web UI lists field-label H5 as kind 'field' with the label quality, and opens a geometry on its
    model mesh: label fields on Whitney N0 for display (unit energy, frequencies from their own Rayleigh
    quotient), the stored p3 figures of merit."""
    from src.data.dataset_3d import rebuild_operators, to_csr
    from src.service.datasets import FieldH5Dataset, discover
    ds = discover(str(field_h5))
    d = next(iter(ds.values()))
    assert isinstance(d, FieldH5Dataset) and d.kind == "field"
    rows = d.rows()
    assert len(rows) == 4 and {r["family"] for r in rows} == {"calib_box", "calib_pillbox"}
    r1 = next(r for r in rows if r["id"] == 1)
    assert r1["n_edges"] > 0 and r1["n_tets"] > 0 and r1["n_nodes"] > 0
    assert r1["field"]["adapt_stop"] in ("converged", "budget", "max_iter") and r1["field"]["adapt_iters"] >= 1
    rec = d.open(1)                                                   # calibration pillbox
    g, U = rec["geom"], rec["U"]
    ne = len(g["edges"])
    assert U.shape == (ne, K) and np.isfinite(U).all()
    ops = rebuild_operators(g["X"], g["tets"])
    Kn, Mn = to_csr(ops["K"], (ne, ne)), to_csr(ops["M"], (ne, ne))
    lam = np.einsum("ik,ik->k", U, Kn @ U) / np.einsum("ik,ik->k", U, Mn @ U)
    f_n0 = np.sqrt(lam) * 299792458.0 / (2 * np.pi * g["scale"]) / 1e9
    # the N0 interpolant is the same mode (a sign / edge-order error would put it far off; the coarse
    # 57-tet model mesh of the fixture costs N0 up to ~15 %)
    assert abs(f_n0[0] / rec["f_GHz"][0] - 1) < 0.02 and np.all(np.abs(f_n0 / rec["f_GHz"] - 1) < 0.25)
    assert rec["qoi_labels"][0]["Q0"] > 1e3                           # stored p3 label of the copper TM010
    # the API shows the stored p3 figures of merit (not those of the N0 display fields)
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
    assert out["truth"]["qoi_source"] == "p3_labels"
    assert out["truth"]["modes"][0]["Q0"] == pytest.approx(rec["qoi_labels"][0]["Q0"])


def test_truba_one_line_run_dry_run(tmp_path):
    env = dict(os.environ, DATA_ROOT=str(tmp_path), CONDA_HOME=str(tmp_path / "noconda"), USER="t")
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
