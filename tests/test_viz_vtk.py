"""Tests for src/viz/vtk_export.py (A3 of the 3D viz contract).

Independent of src/viz/nedelec.py (A1, written in parallel): `export_prediction` is exercised
with a stub `cell_H_fn` that returns a known constant field per mode, never importing nedelec.
"""
import json

import meshio
import numpy as np
import pytest

from src.data.dataset_converter_3d import edges_from_tets, tet_geometry
from src.viz.vtk_export import cell_to_point, export_prediction, write_modes_vtu

# ─────────────────────────── toy meshes ─────────────────────────

# Two tets sharing the face (1, 2, 3); vertex 0 belongs only to tet A, vertex 4 only to tet B.
_NODES = np.array([
    [0.0, 0.0, 0.0],   # 0 (only in A)
    [1.0, 0.0, 0.0],   # 1 (shared)
    [0.0, 1.0, 0.0],   # 2 (shared)
    [0.0, 0.0, 1.0],   # 3 (shared)
    [1.0, 1.0, 1.0],   # 4 (only in B)
])
_TETS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)


def _stub_cell_H_fn(base):
    """Returns a `cell_H_fn(X, tets, edges, u)` stub: constant vector `base*(k+1)` per mode k,
    on every cell, independent of `u` — keeps the test independent of A1/nedelec.py."""
    def fn(X, tets, edges, u):
        u = np.asarray(u)
        K = 1 if u.ndim == 1 else u.shape[1]
        n_tets = np.asarray(tets).shape[0]
        H = np.zeros((n_tets, 3, K))
        for k in range(K):
            H[:, :, k] = np.asarray(base) * (k + 1)
        return H
    return fn


# ─────────────────────────── write_modes_vtu ────────────────────

def test_write_modes_vtu_roundtrip(tmp_path):
    path = str(tmp_path / "mesh.vtu")
    vec = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])   # constant vector field, 2 cells
    scal = np.array([5.0, 5.0])
    fields = {"H_true_0": vec, "absH_true_0": scal}
    point_vec = np.tile([1.0, 2.0, 3.0], (5, 1))
    point_fields = {"H_true_0_pt": point_vec}

    write_modes_vtu(path, _NODES, _TETS, fields, point_fields)

    m = meshio.read(path)
    assert np.allclose(m.points, _NODES, atol=1e-6)
    assert m.cells[0].type == "tetra"
    assert np.array_equal(m.cells[0].data, _TETS)

    assert set(fields) <= set(m.cell_data)
    assert m.cell_data["H_true_0"][0].shape == (2, 3)
    assert np.allclose(m.cell_data["H_true_0"][0], vec, atol=1e-5)
    assert m.cell_data["absH_true_0"][0].shape == (2,)
    assert np.allclose(m.cell_data["absH_true_0"][0], scal, atol=1e-5)

    assert "H_true_0_pt" in m.point_data
    assert m.point_data["H_true_0_pt"].shape == (5, 3)
    assert np.allclose(m.point_data["H_true_0_pt"], point_vec, atol=1e-5)


def test_write_modes_vtu_no_optional_args(tmp_path):
    """point_fields=None, field_data=None must work and not write a sidecar."""
    path = tmp_path / "mesh2.vtu"
    write_modes_vtu(str(path), _NODES, _TETS, {"absH_true_0": np.array([1.0, 2.0])})
    assert path.exists()
    assert not path.with_suffix(".json").exists()


def test_write_modes_vtu_field_data_sidecar_and_nan(tmp_path):
    path = tmp_path / "mesh3.vtu"
    field_data = {
        "f_true_GHz": np.array([1.0, 2.0]),
        "f_pred_GHz": np.array([1.01, 2.02]),
        "rel_l2": np.array([0.01, np.nan]),
        "geom_id": 7,
        "shape_type": "pillbox",
    }
    write_modes_vtu(str(path), _NODES, _TETS, {"absH_true_0": np.array([1.0, 2.0])},
                     field_data=field_data)

    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    with open(sidecar) as fh:
        data = json.load(fh)
    assert data["geom_id"] == 7
    assert data["shape_type"] == "pillbox"
    assert data["f_true_GHz"] == [1.0, 2.0]
    assert data["rel_l2"][0] == pytest.approx(0.01)
    assert np.isnan(data["rel_l2"][1])

    # meshio itself must NOT have round-tripped field_data into the .vtu (documented decision).
    m = meshio.read(str(path))
    assert m.field_data == {}


def test_write_modes_vtu_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "bad.vtu"
    with pytest.raises(ValueError):
        write_modes_vtu(str(path), _NODES, _TETS, {"absH_true_0": np.array([1.0, 2.0, 3.0])})


# ─────────────────────────── cell_to_point ───────────────────────

def test_cell_to_point_constant_field_exact():
    """A constant cell field must map to that same constant at every vertex, for any weights."""
    c = np.array([2.0, -1.0, 0.5])
    cell_vals = np.tile(c, (2, 1))
    for weights in (None, np.array([1.0, 1.0]), np.array([0.1, 7.3])):
        out = cell_to_point(_TETS, 5, cell_vals, weights)
        assert out.shape == (5, 3)
        assert np.allclose(out, c, atol=1e-12)


def test_cell_to_point_volume_weighting_on_two_tets():
    vol, _ = tet_geometry(_NODES, _TETS)
    assert vol.shape == (2,)
    vA, vB = 3.0, -7.0
    cell_vals = np.array([vA, vB])

    out = cell_to_point(_TETS, 5, cell_vals, weights=vol)

    # exclusive vertices get exactly their tet's value
    assert out[0] == pytest.approx(vA)
    assert out[4] == pytest.approx(vB)
    # shared vertices (1, 2, 3) get the volume-weighted average of vA and vB
    expected_shared = (vol[0] * vA + vol[1] * vB) / (vol[0] + vol[1])
    assert out[1] == pytest.approx(expected_shared)
    assert out[2] == pytest.approx(expected_shared)
    assert out[3] == pytest.approx(expected_shared)

    # sanity: unequal volumes actually differ, so the test exercises real weighting
    assert not np.isclose(vol[0], vol[1])


def test_cell_to_point_unweighted_average_differs_from_weighted():
    vol, _ = tet_geometry(_NODES, _TETS)
    vA, vB = 3.0, -7.0
    cell_vals = np.array([vA, vB])
    weighted = cell_to_point(_TETS, 5, cell_vals, weights=vol)
    unweighted = cell_to_point(_TETS, 5, cell_vals, weights=None)
    # shared vertices: plain mean vs volume-weighted mean must differ (volumes are unequal)
    assert not np.isclose(weighted[1], unweighted[1])
    assert unweighted[1] == pytest.approx((vA + vB) / 2.0)


def test_cell_to_point_vertex_with_no_incident_cell_is_zero():
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    cell_vals = np.array([[9.0, 9.0, 9.0]])
    out = cell_to_point(tets, n_nodes=6, cell_vals=cell_vals)  # vertices 4, 5 unused
    assert np.allclose(out[4], 0.0)
    assert np.allclose(out[5], 0.0)


# ─────────────────────────── export_prediction ──────────────────

def _make_pred(seed=0, split_second_mode=True):
    rng = np.random.default_rng(seed)
    edges = edges_from_tets(_TETS)
    ne, K = len(edges), 2
    true = rng.standard_normal((ne, K))
    pred = true + 0.01 * rng.standard_normal((ne, K))
    err = pred - true
    rel_l2 = np.array([0.05, np.nan if split_second_mode else 0.2])
    return {
        "geom_id": 3, "shape_type": "toy",
        "X": _NODES, "tets": _TETS, "edges": edges,
        "scale": 2.0, "center": np.array([1.0, 2.0, 3.0]),
        "true": true, "pred": pred, "err": err,
        "f_true": np.array([1.0, 2.0]), "f_pred": np.array([1.001, 1.999]),
        "rel_l2": rel_l2, "split": np.array([False, split_second_mode]),
    }


def test_export_prediction_writes_expected_names_and_shapes(tmp_path):
    path = tmp_path / "geom_3.vtu"
    pred = dict(_make_pred(), field="H")                     # explicit letter: H_* array names
    base = np.array([1.0, 0.0, 0.0])
    stub = _stub_cell_H_fn(base)

    out = export_prediction(str(path), pred, cell_H_fn=stub)
    assert out == str(path)
    assert path.exists()

    m = meshio.read(str(path))
    K = pred["true"].shape[1]
    for k in range(K):
        for prefix in ("H_true", "H_pred", "H_err"):
            name = f"{prefix}_{k}"
            assert name in m.cell_data, name
            assert m.cell_data[name][0].shape == (2, 3)
            pt_name = f"{name}_pt"
            assert pt_name in m.point_data, pt_name
            assert m.point_data[pt_name].shape == (5, 3)
        for prefix in ("absH_true", "absH_pred", "absH_err"):
            name = f"{prefix}_{k}"
            assert name in m.cell_data, name
            assert m.cell_data[name][0].shape == (2,)

    # stub H is independent of u (constant per mode, ignores the passed-in DOFs) => true/pred/err
    # cell data are all identical to that constant.
    for k in range(K):
        expected = base * (k + 1)
        for prefix in ("H_true", "H_pred", "H_err"):
            assert np.allclose(m.cell_data[f"{prefix}_{k}"][0], expected, atol=1e-5)
        assert np.allclose(m.cell_data[f"absH_true_{k}"][0], np.linalg.norm(expected), atol=1e-5)


def test_export_prediction_coordinates_in_mm(tmp_path):
    path = tmp_path / "geom_coords.vtu"
    pred = _make_pred()
    stub = _stub_cell_H_fn(np.array([1.0, 1.0, 1.0]))
    export_prediction(str(path), pred, cell_H_fn=stub)

    m = meshio.read(str(path))
    expected_mm = (pred["X"] * pred["scale"] + pred["center"]) * 1e3
    assert np.allclose(m.points, expected_mm, atol=1e-2)


def test_export_prediction_sidecar_has_mode_info_and_handles_nan(tmp_path):
    path = tmp_path / "geom_nan.vtu"
    pred = _make_pred(split_second_mode=True)
    stub = _stub_cell_H_fn(np.array([0.0, 0.0, 1.0]))
    export_prediction(str(path), pred, cell_H_fn=stub)

    with open(path.with_suffix(".json")) as fh:
        data = json.load(fh)
    assert data["geom_id"] == 3
    assert data["shape_type"] == "toy"
    assert data["f_true_GHz"] == pytest.approx([1.0, 2.0])
    assert data["rel_l2"][0] == pytest.approx(0.05)
    assert np.isnan(data["rel_l2"][1])


def test_export_prediction_default_cell_H_fn_lazily_imports_nedelec(tmp_path, monkeypatch):
    """cell_H_fn=None must not blow up import of this module even if nedelec.py is missing;
    the lazy import only happens when actually called with the default."""
    import builtins
    import src.viz.vtk_export as vtk_export  # noqa: F401  (import must succeed regardless)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.viz.nedelec":
            raise ImportError("src.viz.nedelec not available in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    path = tmp_path / "geom_default.vtu"
    pred = _make_pred()
    with pytest.raises(ImportError):
        export_prediction(str(path), pred, cell_H_fn=None)


def test_export_prediction_names_follow_the_field(tmp_path):
    """E-formulation predictions (pred['field'] = 'E'): E_true_k, absE_*, E_*_pt; no H_* arrays."""
    path = tmp_path / "geom_E.vtu"
    pred = dict(_make_pred(), field="E")
    export_prediction(str(path), pred, cell_H_fn=_stub_cell_H_fn(np.array([0.0, 1.0, 0.0])))
    m = meshio.read(str(path))
    for k in range(pred["true"].shape[1]):
        for prefix in ("E_true", "E_pred", "E_err"):
            assert f"{prefix}_{k}" in m.cell_data and f"{prefix}_{k}_pt" in m.point_data
        assert f"absE_true_{k}" in m.cell_data
    assert not any(n.startswith(("H_", "absH_")) for n in list(m.cell_data) + list(m.point_data))
    with open(path.with_suffix(".json")) as fh:
        assert json.load(fh)["field"] == "E"
