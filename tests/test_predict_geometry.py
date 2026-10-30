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
from src.service import geometry as G                                       # noqa: E402
from src.data.dataset_converter_3d import extract_geometry_3d, qoi_operators_of   # noqa: E402
from src.qoi.operators import beam_axis, build_qoi_operators                # noqa: E402
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


def test_axis_dir_point_reproduces_axis_xy():
    rng = np.random.default_rng(0)
    from tests.maxwell3d_synth import box_geometry
    g = box_geometry(n=4, n_modes=2)[0]
    c = np.array([0.003, -0.002, 0.001])
    xy = (c[0] + 0.01 * rng.standard_normal(), c[1])
    a = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, n_axis=31, axis_xy=xy)
    b = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, n_axis=31,
                            axis_dir="z", axis_point=(xy[0], xy[1], 0.37))
    np.testing.assert_allclose(a["q"], b["q"])
    np.testing.assert_allclose(a["zeta"], b["zeta"])
    np.testing.assert_allclose(a["Az"].toarray(), b["Az"].toarray(), atol=1e-12)
    v = build_qoi_operators(g["X"], g["tets"], g["edges"], g["scale"], c, n_axis=31,
                            axis_dir=(0, 0, -2.0), axis_point=(xy[0], xy[1], 0.0))    # reversed: ζ → −ζ, E_z → −E_z
    np.testing.assert_allclose(np.sort(-v["zeta"]), np.sort(a["zeta"]), atol=1e-12)
    assert v["L_axis"] == pytest.approx(a["L_axis"])


def test_beam_axis_per_family():
    X = np.array([[0, 0, -1.0], [0, 0, 0.5], [1, 0, 0], [0, 1, 0]])
    assert beam_axis("elliptical", X, 0.1, np.zeros(3)) == {}
    ax = beam_axis("hwr", X, 0.1, np.array([0, 0, 0.2]))
    assert ax["axis_dir"] == "x" and ax["axis_point"][2] == pytest.approx(0.2 + 0.1 * (-0.25))
    # ridged_box sits in [0, a] × [0, b] × [0, d]: the axis runs along x through the middle of the gap
    p = {"a": 0.08, "b": 0.05, "d": 0.04, "ridge_h": 0.01, "double": 1.0, "gap": 0.02}
    ax = beam_axis("ridged_box", X, 1.0, np.zeros(3), p)
    assert ax["axis_dir"] == "x" and ax["axis_point"] == pytest.approx((0.0, 0.025, 0.02))
    assert beam_axis("ridged_box", X, 1.0, np.zeros(3), {**p, "double": 0.0, "ridge_h": 0.02})["axis_point"][2] == 0.01


def test_acceleration_length_excludes_the_beam_pipes():
    from src.data_gen import cavity_shapes as cs
    from src.qoi.operators import acceleration_length
    p, halves, walls = cs.draw_elliptical(np.random.default_rng(4), n_cells=3)
    assert acceleration_length("elliptical", p) == pytest.approx(cs.elliptical_chain(halves, walls, 3)[1])
    assert acceleration_length("pillbox_pipes", {"Lc": 0.03, "Lpipe": 0.02}) == 0.03
    assert acceleration_length("composite", {"R0": 0.03}) is None and acceleration_length("elliptical", None) is None


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


def test_qoi_operators_of_uses_family_axis(pillbox_files):
    nodes, tets = G.mesh_file(str(pillbox_files / "pillbox.msh"), "mm")
    geom, M = extract_geometry_3d(nodes, tets)
    z = qoi_operators_of(dict(geom, shape_type="pillbox_pipes"), M=M, n_axis=51)
    x = qoi_operators_of(dict(geom, shape_type="hwr"), M=M, n_axis=51)
    assert z["L_axis"] * geom["scale"] == pytest.approx(L, rel=0.05)         # along z: the length
    assert x["L_axis"] * geom["scale"] == pytest.approx(2 * R, rel=0.05)     # along x: the diameter
