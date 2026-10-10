"""Field labels (docs/29): generator --labels field → H5 → FieldDataset → field3d model, and the
TRUBA one-line orchestrator (run.sh --dry-run)."""
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
