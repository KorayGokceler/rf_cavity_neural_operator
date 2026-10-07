"""scripts/predict_geometry.py (new-geometry prediction, end of the pipeline) and the general /
per-family beam axis of the QoI operators."""
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
from src.data.dataset_converter_3d import extract_geometry_3d, qoi_operators_of   # noqa: E402
from src.qoi.operators import beam_axis, build_qoi_operators                # noqa: E402
from src.training.lightning_module import GNOTLightning                     # noqa: E402

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
    m = GNOTLightning(val_dim=9, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=10, num_field_modes=4, rff_dim=8,
                      model_type="eigenspace3d", physics_freq=True, eigenspace_kwargs={"n_layers": 1},
                      data_cfg={"field": "E", "feature_indices": None})
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.strategy.connect(m)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.ckpt"
    trainer.save_checkpoint(str(path))
    return path


def test_axis_dir_point_reproduces_axis_xy():
    rng = np.random.default_rng(0)
    from tests.maxwell3d_synth import box_geometry
    g = box_geometry(field="E", n=4, n_modes=2)[0]
    c = np.array([0.003, -0.002, 0.001])
    xy = (c[0] + 0.01 * rng.standard_normal(), c[1])
    a = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, "E", n_axis=31, axis_xy=xy)
    b = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, "E", n_axis=31,
                            axis_dir="z", axis_point=(xy[0], xy[1], 0.37))
    np.testing.assert_allclose(a["q"], b["q"])
    np.testing.assert_allclose(a["zeta"], b["zeta"])
    np.testing.assert_allclose(a["Az"].toarray(), b["Az"].toarray(), atol=1e-12)
    v = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, "E", n_axis=31,
                            axis_dir=(0, 0, -2.0), axis_point=(xy[0], xy[1], 0.0))    # reversed: ζ → −ζ, E_z → −E_z
    np.testing.assert_allclose(np.sort(-v["zeta"]), np.sort(a["zeta"]), atol=1e-12)
    assert v["L_axis"] == pytest.approx(a["L_axis"])


def test_beam_axis_per_family():
    X = np.array([[0, 0, -1.0], [0, 0, 0.5], [1, 0, 0], [0, 1, 0]])
    assert beam_axis("elliptical", X, 0.1, np.zeros(3)) == {}
    ax = beam_axis("hwr", X, 0.1, np.array([0, 0, 0.2]))
    assert ax["axis_dir"] == "x" and ax["axis_point"][2] == pytest.approx(0.2 + 0.1 * (-0.25))


@pytest.mark.parametrize("source", ["mesh", "step"])
def test_predict_geometry_end_to_end(tmp_path, pillbox_files, ckpt, source):
    out = tmp_path / "pred"
    arg = ["--mesh", str(pillbox_files / "pillbox.msh")] if source == "mesh" else \
        ["--step", str(pillbox_files / "pillbox.step"), "--mesh_size", "0.3"]
    res = pg.main(["--checkpoint", str(ckpt), *arg, "--unit", "mm", "--fe", "--out_dir", str(out),
                   "--device", "cpu", "--n_axis", "101"])
    assert res["field"] == "E" and len(res["f_pred_GHz"]) == 4 and len(res["f_fe_GHz"]) == 4
    assert np.all(np.isfinite(res["f_pred_GHz"]))
    assert res["f_fe_GHz"][0] == pytest.approx(F_TM010, rel=0.03)          # FE reference is the TM010
    assert res["qoi_fe"]["G_ohm"][0] == pytest.approx(2.405 * 376.73 / (2 * (1 + R / L)), rel=0.15)
    assert all(0.0 <= r <= 1.0 + 1e-9 for r in res["field_rel_l2"])
    assert {"mesh", "operators", "model", "qoi", "fe_solve"} <= set(res["time_s"])
    with open(out / "prediction.json") as f:
        assert json.load(f)["n_edges"] == res["n_edges"]
    assert (out / "modes.vtu").exists()


def test_predict_geometry_needs_field_or_rejects_mismatch(tmp_path, pillbox_files, ckpt):
    with pytest.raises(ValueError, match="trained on field E"):
        pg.main(["--checkpoint", str(ckpt), "--mesh", str(pillbox_files / "pillbox.msh"), "--unit", "mm",
                 "--field", "H", "--out_dir", str(tmp_path), "--no_vtu"])


def test_qoi_operators_of_uses_family_axis(pillbox_files):
    nodes, tets = pg.mesh_file(str(pillbox_files / "pillbox.msh"), "mm")
    geom, M = extract_geometry_3d(nodes, tets, "E")
    z = qoi_operators_of(dict(geom, shape_type="pillbox_pipes"), "E", M=M, n_axis=51)
    x = qoi_operators_of(dict(geom, shape_type="hwr"), "E", M=M, n_axis=51)
    assert z["L_axis"] * geom["scale"] == pytest.approx(L, rel=0.05)         # along z: the length
    assert x["L_axis"] * geom["scale"] == pytest.approx(2 * R, rel=0.05)     # along x: the diameter
