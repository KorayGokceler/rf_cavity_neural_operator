"""Tests for src/viz/vtk_export.py (write_modes_vtu, cell_to_point)."""
import json

import meshio
import numpy as np
import pytest

from src.data.dataset_converter_3d import tet_geometry
from src.viz.vtk_export import cell_to_point, write_modes_vtu

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
