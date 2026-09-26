"""Export N0 (Whitney/edge-element) mode fields to ParaView-readable .vtu (A3, docs/viz3d contract).

Public API (see the contract in the shared scratchpad — `viz3d_contract.md`, section A3):
    write_modes_vtu(path, X_phys, tets, fields, point_fields=None, field_data=None) -> path
    cell_to_point(tets, n_nodes, cell_vals, weights=None) -> [Nv, ...]
    export_prediction(path, pred, cell_H_fn=None) -> path

Conventions
-----------
- Cells are linear tetrahedra ('tetra' in VTK/meshio); one .vtu per geometry (no multi-block /
  .pvd — the contract says that's optional and unnecessary here).
- All field data (cell and point) is written as float32 to keep files small; meshio's default
  writer for .vtu is binary + zlib-compressed XML, which is kept as-is (no format= override).
- Coordinates are written in millimetres: X_mm = (X_norm * scale + center) * 1e3, where X_norm
  are the NORMALISED mesh vertices used everywhere else in this codebase (see
  src/data/dataset_converter_3d.py). The field VECTORS (H_true/H_pred/H_err and their
  point-averaged '_pt' versions) are left in normalised-mesh DOF units — they are only ever
  compared to each other / used for relative display (glyph length, colour-by-|H|), so no
  rescaling is applied to them. This is intentional, not an oversight.
- `field_data` (run-level scalars/arrays such as f_true_GHz, f_pred_GHz, rel_l2, geom_id,
  shape_type) is NOT written into the .vtu itself. meshio 5.3.5's VTU writer does not round-trip
  `meshio.Mesh(..., field_data=...)` for the 'vtu' format — writing it and reading it back gives
  `{}` (verified empirically; VTK's own <FieldData> block is simply not emitted by meshio's vtu
  writer, unlike e.g. its vtk/xdmf writers). So instead, whenever `field_data` is given,
  `write_modes_vtu` writes a small JSON sidecar next to the .vtu, at the same path with its
  extension replaced by '.json' (e.g. 'geom_3.vtu' -> 'geom_3.json'). NaN values (e.g. `rel_l2`
  for split modes) are written as the (non-standard but Python-round-trippable) JSON literal
  `NaN`; `json.load` reads it back as `float('nan')`.
"""
import json
from pathlib import Path

import meshio
import numpy as np

from src.data.dataset_converter_3d import tet_geometry


def _sidecar_path(path):
    """The JSON sidecar path for a given .vtu path: same location, '.json' extension."""
    return str(Path(path).with_suffix(".json"))


def _to_jsonable(value):
    """numpy scalars/arrays -> plain Python (lists/floats/ints/str), NaN preserved as float('nan')."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def write_modes_vtu(path, X_phys, tets, fields, point_fields=None, field_data=None):
    """Write one .vtu of tetrahedral cells with per-cell and (optionally) per-point data.

    Args:
        path: output path, should end in '.vtu'.
        X_phys: [Nv,3] vertex coordinates (as written to the file verbatim — callers convert
            to mm etc. themselves; see `export_prediction`).
        tets: [Nt,4] int, 0-based vertex indices.
        fields: dict[name -> array [Nt] or [Nt,3] (or any [Nt, ...] shape)] cell data. Cast to
            float32.
        point_fields: optional dict[name -> array [Nv] or [Nv,3]] point data. Cast to float32.
        field_data: optional dict of run-level scalars/arrays (e.g. {'f_true_GHz': [...], ...}).
            NOT embedded in the .vtu (see module docstring) — written to a JSON sidecar next to
            `path` instead.

    Returns `path` (unchanged, for chaining).
    """
    X_phys = np.asarray(X_phys, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    n_nodes, n_tets = X_phys.shape[0], tets.shape[0]

    cell_data = {}
    for name, val in fields.items():
        val = np.asarray(val)
        if val.shape[0] != n_tets:
            raise ValueError(f"field {name!r}: shape[0]={val.shape[0]} != n_tets={n_tets}")
        cell_data[name] = [val.astype(np.float32)]

    point_data = {}
    for name, val in (point_fields or {}).items():
        val = np.asarray(val)
        if val.shape[0] != n_nodes:
            raise ValueError(f"point field {name!r}: shape[0]={val.shape[0]} != n_nodes={n_nodes}")
        point_data[name] = val.astype(np.float32)

    mesh = meshio.Mesh(
        points=X_phys,
        cells=[("tetra", tets)],
        point_data=point_data or None,
        cell_data=cell_data or None,
    )
    mesh.write(str(path))

    if field_data:
        sidecar = {k: _to_jsonable(v) for k, v in field_data.items()}
        with open(_sidecar_path(path), "w") as fo:
            json.dump(sidecar, fo, indent=1, allow_nan=True)

    return path


def cell_to_point(tets, n_nodes, cell_vals, weights=None):
    """Volume-weighted average of per-cell values onto vertices.

    For vertex v: out[v] = sum_{t incident to v} weights[t] * cell_vals[t]
                            / sum_{t incident to v} weights[t]
    (a plain unweighted average when `weights` is None). Vertices touched by no tet get 0.

    Args:
        tets: [Nt,4] int.
        n_nodes: number of vertices Nv (output length).
        cell_vals: [Nt, ...] array (scalar or vector/tensor per cell).
        weights: [Nt] array or None (default: uniform weights, i.e. a plain average). Callers
            wanting an exact volume-weighted average pass tet volumes, e.g. from
            `src.data.dataset_converter_3d.tet_geometry`.

    Returns:
        [n_nodes, ...] array, same trailing shape as `cell_vals`, dtype float64.
    """
    tets = np.asarray(tets, dtype=np.int64)
    cell_vals = np.asarray(cell_vals, dtype=np.float64)
    n_tets = tets.shape[0]
    trailing = cell_vals.shape[1:]

    w = np.ones(n_tets, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    contrib = cell_vals * w.reshape((n_tets,) + (1,) * len(trailing))

    num = np.zeros((n_nodes,) + trailing, dtype=np.float64)
    den = np.zeros(n_nodes, dtype=np.float64)
    for i in range(4):
        np.add.at(num, tets[:, i], contrib)
        np.add.at(den, tets[:, i], w)

    den_safe = np.where(den > 0, den, 1.0)
    out = num / den_safe.reshape((n_nodes,) + (1,) * len(trailing))
    out[den == 0] = 0.0
    return out


def export_prediction(path, pred, cell_H_fn=None):
    """Write one geometry's true/pred/error H-field modes (A2 `predict()` output) to a .vtu.

    Args:
        path: output .vtu path.
        pred: dict in the A2 contract format (`src.viz.predict.predict`):
            'X' [Nv,3] normalised vertices, 'tets' [Nt,4], 'edges' [Ne,2],
            'scale' float, 'center' [3], 'true'/'pred'/'err' [Ne,K] N0 DOFs,
            'f_true'/'f_pred'/'rel_l2' [K], 'geom_id', 'shape_type'.
        cell_H_fn: callable(X, tets, edges, u) -> H [Nt,3,K] (u: [Ne] or [Ne,K]), evaluating the
            N0 field at tet centroids. Default None means "use A1's `cell_field`" — imported
            lazily *inside this function*, not at module top level, so that this module can be
            imported (and its other functions tested) even before/without src/viz/nedelec.py
            existing. Pass a stub here in tests to stay independent of A1.

    Writes, per mode k in range(K):
        cell:  H_true_k, H_pred_k, H_err_k        [Nt,3] vectors
               absH_true_k, absH_pred_k, absH_err_k  [Nt]  scalars (Euclidean norm of the vector)
        point: H_true_k_pt, H_pred_k_pt, H_err_k_pt  [Nv,3] (volume-weighted cell_to_point average,
               for smooth ParaView glyphs/streamlines)
    plus a JSON sidecar (see module docstring) with 'f_true_GHz', 'f_pred_GHz', 'rel_l2', 'geom_id',
    'shape_type'.

    Coordinates are written in mm: (X*scale + center)*1e3. The field vectors themselves are left
    in normalised-mesh DOF units (see module docstring).

    Returns `path`.
    """
    if cell_H_fn is None:
        from src.viz.nedelec import cell_field as cell_H_fn

    X = np.asarray(pred["X"], dtype=np.float64)
    tets = np.asarray(pred["tets"], dtype=np.int64)
    edges = np.asarray(pred["edges"], dtype=np.int64)
    scale = float(pred["scale"])
    center = np.asarray(pred["center"], dtype=np.float64)

    true = np.asarray(pred["true"], dtype=np.float64)
    predicted = np.asarray(pred["pred"], dtype=np.float64)
    err = np.asarray(pred.get("err", predicted - true), dtype=np.float64)
    K = true.shape[1]
    n_nodes = X.shape[0]

    H_true = np.asarray(cell_H_fn(X, tets, edges, true))   # [Nt,3,K]
    H_pred = np.asarray(cell_H_fn(X, tets, edges, predicted))
    H_err = np.asarray(cell_H_fn(X, tets, edges, err))

    vol, _ = tet_geometry(X, tets)

    fields, point_fields = {}, {}
    for k in range(K):
        ht, hp, he = H_true[:, :, k], H_pred[:, :, k], H_err[:, :, k]
        fields[f"H_true_{k}"] = ht
        fields[f"H_pred_{k}"] = hp
        fields[f"H_err_{k}"] = he
        fields[f"absH_true_{k}"] = np.linalg.norm(ht, axis=1)
        fields[f"absH_pred_{k}"] = np.linalg.norm(hp, axis=1)
        fields[f"absH_err_{k}"] = np.linalg.norm(he, axis=1)

        point_fields[f"H_true_{k}_pt"] = cell_to_point(tets, n_nodes, ht, vol)
        point_fields[f"H_pred_{k}_pt"] = cell_to_point(tets, n_nodes, hp, vol)
        point_fields[f"H_err_{k}_pt"] = cell_to_point(tets, n_nodes, he, vol)

    X_mm = (X * scale + center) * 1e3

    field_data = {
        "f_true_GHz": np.asarray(pred["f_true"], dtype=np.float64),
        "f_pred_GHz": np.asarray(pred["f_pred"], dtype=np.float64),
        "rel_l2": np.asarray(pred["rel_l2"], dtype=np.float64),
        "geom_id": pred.get("geom_id"),
        "shape_type": pred.get("shape_type"),
    }

    return write_modes_vtu(path, X_mm, tets, fields, point_fields, field_data)
