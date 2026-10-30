"""scripts/predict_geometry.py (new-geometry prediction, end of the pipeline)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict_geometry as pg                                  # noqa: E402
from src.training.lightning_module import CavityLightning                     # noqa: E402

R, L = 0.1, 0.1                                     # pillbox [m]: TM010 at 2.405 c / (2π R)
F_TM010 = 2.404825557695773 * 299792458.0 / (2 * np.pi * R) / 1e9


@pytest.fixture(scope="module")
def pillbox_files(tmp_path_factory):
    gmsh = pytest.importorskip("gmsh")
    d = tmp_path_factory.mktemp("pillbox")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()                                                  # other tests leave models in the session
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addCylinder(0, 0, -50, 0, 0, 100, 100)        # mm
    gmsh.model.occ.synchronize()
    gmsh.write(str(d / "pillbox.step"))
    gmsh.option.setNumber("Mesh.MeshSizeMax", 30)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(d / "pillbox.msh"))
    gmsh.clear()
    return d


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    import pytorch_lightning as pl
    torch.manual_seed(0)
    m = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=10, num_field_modes=4, rff_dim=8,
                        eigenspace_kwargs={"n_layers": 1}, data_cfg={"field": "E", "feature_indices": None})
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.strategy.connect(m)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.ckpt"
    trainer.save_checkpoint(str(path))
    return path


@pytest.mark.parametrize("source", ["step", "family"])
def test_predict_geometry_end_to_end(tmp_path, monkeypatch, pillbox_files, ckpt, source):
    """CLI: CAD / generator geometry → curved mesh → field3d checkpoint → Ritz frequencies (upper bounds of the
    p2 FE eigenvalues of the same mesh, --fe), p2 figures of merit, prediction.json + modes.vtu."""
    from src.data_gen import cavity_shapes as cs
    from src.data_gen import dataset_generator_3d as gen
    monkeypatch.setattr(gen, "ARGS", gen.ARGS)                   # the CLI sets the generator's module state:
    monkeypatch.setattr(cs, "MIN_FILLET", cs.MIN_FILLET)         # restored for the other tests
    out = tmp_path / "pred"
    arg = ["--step", str(pillbox_files / "pillbox.step"), "--unit", "mm", "--mesh_size", "0.3"] if source == "step" \
        else ["--family", "box", "--id", "3", "--mesh_size", "0.2"]
    res = pg.main(["--checkpoint", str(ckpt), *arg, "--fe", "--out_dir", str(out), "--device", "cpu"])
    assert len(res["f_pred_GHz"]) == 4 and len(res["f_fe_GHz"]) == 4 and not res["untrained"]
    assert np.all(np.isfinite(res["f_pred_GHz"]))
    assert np.all(np.asarray(res["f_pred_GHz"]) >= np.asarray(res["f_fe_GHz"]) * (1 - 1e-6))   # Ritz upper bounds
    if source == "step":
        assert res["f_fe_GHz"][0] == pytest.approx(F_TM010, rel=0.01)       # p2 FE reference is the TM010
    assert res["qoi_source"] == "p2_model" and all(m["Q0"] is None or m["Q0"] > 0 for m in res["modes"])
    assert {"geometry", "mesh", "operators", "model", "qoi", "fe_solve"} <= set(res["time_s"])
    with open(out / "prediction.json") as f:
        assert json.load(f)["n_dof"] == res["n_dof"]
    assert (out / "modes.vtu").exists()
    import meshio                                                # the p2 fields on the P2 nodes of the curved mesh
    vtu = meshio.read(out / "modes.vtu")
    assert len(vtu.cells_dict["tetra"]) == 8 * res["n_tets"] and len(vtu.points) > res["n_tets"] // 2
    assert {"E_pred_0", "H_pred_3"} <= set(vtu.point_data) and np.isfinite(vtu.point_data["E_pred_0"]).all()
