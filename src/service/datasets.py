"""Dataset browsing for the development UI: generator H5 shards, listed and opened one geometry at a
time (mesh, model input features, FE modes, QoI labels, parameters).

Registry: RFCAV_DATA = paths separated by os.pathsep (':' on Linux); each path is an .h5 file or a
directory searched (depth ≤ 4) for directories holding *.h5 shards — one dataset per H5 directory
(the TRUBA layout h5/<family>/<family>_s*.h5 gives one dataset per family).

Field-label H5 (generator --labels field, docs/29–30; file metadata labels = 'field') open as
FieldH5Dataset: the table adds the label quality (adaptive levels, stop reason, error estimate, label
mesh, rounded / sharp edges), and a geometry opens on its model mesh with the label fields
interpolated onto Whitney N0 for display; the figures of merit shown are the stored p3 labels.
"""
import glob
import json
import os
import threading

import numpy as np

from src.data.dataset_converter_3d import (FEATURE_NAMES_3D, extract_geometry_3d, geom_params_of, h5_edges_to_canonical,
                                           require_e_field)

_PARAM_SKIP = {'geom_params', 'fem_element', 'field', 'shape_type'}


def _float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class H5Dataset:
    kind = 'h5'

    def __init__(self, files, name):
        self.files, self.name = sorted(files), name
        self._rows, self._index, self._lock = None, None, threading.Lock()

    def rows(self):
        """One summary row per sample (attributes only: cheap)."""
        import h5py
        with self._lock:
            if self._rows is None:
                rows, index = [], {}
                for fp in self.files:
                    with h5py.File(fp, 'r') as f:
                        require_e_field(json.loads(f.attrs.get('metadata', '{}') or '{}'))
                        for key in f:
                            if not key.startswith('sample_'):
                                continue
                            g = f[key]
                            sid = int(key.split('_')[-1])
                            fr = np.sort(np.asarray(g['freqs'][()], float))
                            rows.append({'id': sid, 'family': str(g.attrs.get('shape_type', '')),
                                         'n_nodes': int(g['nodes'].shape[0]), 'n_edges': int(g['edges'].shape[0]),
                                         'n_tets': int(g['tets'].shape[0]),
                                         'betti1': int(g.attrs.get('betti1', 0)),
                                         'deformed': 'deform_lip' in g.attrs,
                                         'f_GHz': [float(x) for x in fr]})
                            index[sid] = (fp, key)
                self._rows, self._index = rows, index
            return self._rows

    def open(self, sid):
        """Full record of sample sid (features are recomputed exactly as the converter does)."""
        import h5py
        self.rows()
        if sid not in self._index:
            raise KeyError(sid)
        fp, key = self._index[sid]
        with h5py.File(fp, 'r') as f:
            g = f[key]
            if 'e_edges' not in g:
                raise ValueError(f"{key}: no E-field DOFs (data of the removed H formulation?)")
            nodes, tets, edges_h5 = g['nodes'][()], g['tets'][()], g['edges'][()]
            V, freqs = np.asarray(g['e_edges'][()], float), np.asarray(g['freqs'][()], float)
            attrs = {k: g.attrs[k] for k in g.attrs}
        order = np.argsort(freqs, kind='stable')
        freqs, V = freqs[order], V[:, order]
        geom, M = extract_geometry_3d(nodes, tets)
        rows, sign = h5_edges_to_canonical(edges_h5, geom['n_nodes'], geom['edges'])
        U = np.zeros_like(V)
        U[rows] = sign[:, None] * V
        params = {k: _float(v) for k, v in attrs.items() if k not in _PARAM_SKIP and _float(v) is not None}
        geom.update(shape_type=str(attrs.get('shape_type', '')), params=geom_params_of(attrs))
        return {'id': sid, 'family': geom['shape_type'], 'nodes': np.asarray(nodes, float),
                'tets': np.asarray(tets, np.int64), 'geom': geom, 'M': M, 'U': U, 'f_GHz': freqs,
                'f_next': _float(attrs.get('freq_next')), 'params': params, 'qoi_labels': None,
                'features': FEATURE_NAMES_3D, 'source_file': os.path.basename(fp)}


_NG_LOCK = threading.Lock()          # NGSolve from the API's request threads: one call at a time


def is_field_h5(fp):
    """True for a field-label H5 (generator --labels field)."""
    import h5py
    try:
        with h5py.File(fp, 'r') as f:
            return json.loads(f.attrs.get('metadata', '{}') or '{}').get('labels') == 'field'
    except (OSError, ValueError):
        return False


def _attr(g, k, cast=float, default=None):
    v = g.attrs.get(k, None)
    if v is None:
        return default
    if isinstance(v, bytes):
        v = v.decode()
    try:
        out = cast(v)
    except (TypeError, ValueError):
        return default
    return None if isinstance(out, float) and not np.isfinite(out) else out


def whitney_dofs(mesh, U, model_order):
    """Model-space fields U [ndof, K] (HCurl(model_order, dirichlet='wall') of the curved mesh) → the
    linear mesh's vertices [Nv, 3], tets [Nt, 4] and Whitney N0 DOFs on NGSolve's edges [Ne, K] with
    those edges [Ne, 2] (tail, head) — the H5 N0 convention of the browser (h5_edges_to_canonical)."""
    from ngsolve import GridFunction, HCurl
    fes = HCurl(mesh, order=int(model_order), dirichlet='wall')
    if fes.ndof != U.shape[0]:
        raise ValueError(f"stored fields have {U.shape[0]} DOFs, the rebuilt HCurl(p{model_order}) space "
                         f"{fes.ndof}: different NGSolve version?")
    from src.service.geometry import linear_mesh
    fes0 = HCurl(mesh, order=0)
    g, g0 = GridFunction(fes), GridFunction(fes0)
    V = np.zeros((fes0.ndof, U.shape[1]))
    for k in range(U.shape[1]):
        g.vec.FV().NumPy()[:] = U[:, k]
        g0.Set(g, dual=True)          # projection-based (commuting) interpolation: plain Set is ~15 % off
        V[:, k] = g0.vec.FV().NumPy()
    nodes, tets = linear_mesh(mesh)
    edges = np.array([[v.nr for v in e.vertices] for e in mesh.edges], np.int64)
    return nodes, tets, edges, V


class FieldH5Dataset:
    kind = 'field'

    def __init__(self, files, name):
        self.files, self.name = sorted(files), name
        self._rows, self._index, self._lock = None, None, threading.Lock()

    def rows(self):
        """One row per sample from the attributes (no mesh is read); 'field' holds the label quality."""
        import h5py
        with self._lock:
            if self._rows is None:
                rows, index = [], {}
                for fp in self.files:
                    with h5py.File(fp, 'r') as f:
                        for key in f:
                            if not key.startswith('sample_'):
                                continue
                            g = f[key]
                            sid = int(key.split('_')[-1])
                            pe = np.asarray(g['proj_err'][()], float) if 'proj_err' in g else np.zeros(0)
                            rows.append({
                                'id': sid, 'family': str(g.attrs.get('shape_type', '')),
                                'n_nodes': _attr(g, 'n_vertices', int, 0), 'n_edges': _attr(g, 'n_dof_model', int, 0),
                                'n_tets': _attr(g, 'n_tets', int, 0), 'betti1': _attr(g, 'betti1', int, 0),
                                'deformed': 'deform_om' in g,
                                'f_GHz': [float(x) for x in np.sort(np.asarray(g['freqs'][()], float))],
                                'field': {'n_tets_label': _attr(g, 'n_tets_label', int),
                                          'n_dof_label': _attr(g, 'n_dof_label', int),
                                          'n_tets_start': _attr(g, 'n_tets_start', int),
                                          'adapt_iters': _attr(g, 'adapt_iters', int),
                                          'adapt_stop': _attr(g, 'adapt_stop', str),
                                          'est_rel_f': _attr(g, 'est_rel_f'), 'est_rel_wall': _attr(g, 'est_rel_wall'),
                                          'adapt_df': _attr(g, 'adapt_df'), 'adapt_dq': _attr(g, 'adapt_dq'),
                                          'rounded_edges': _attr(g, 'n_rounded_edges', lambda v: int(float(v))),
                                          'sharp_edges': _attr(g, 'sharp_edges', int, 0),
                                          'proj_err_max': float(pe.max()) if pe.size else None,
                                          't_labels_s': _attr(g, 't_labels')}})
                            index[sid] = (fp, key)
                self._rows, self._index = rows, index
            return self._rows

    def open(self, sid):
        """Model mesh + the label fields on Whitney N0 (display) + the stored p3 figures of merit; mesh_src (the
        stored curved mesh: .vol + BREP bytes, curve order, deformation) lets the field model predict on the
        same mesh (src.service.model.ModelService.predict)."""
        import h5py
        import torch  # noqa: F401 — before NGSolve: ngsolve loaded first breaks torch later (tests/conftest.py)

        from src.data_gen.field_labels import read_field_group
        self.rows()
        if sid not in self._index:
            raise KeyError(sid)
        fp, key = self._index[sid]
        with _NG_LOCK, h5py.File(fp, 'r') as f:
            g = f[key]
            mesh, rec = read_field_group(g)
            attrs = {k: g.attrs[k] for k in g.attrs}
            mesh_src = {'vol': np.asarray(g['vol']).tobytes(), 'brep': np.asarray(g['brep']).tobytes(),
                        'curve': int(attrs['curve']), 'model_order': int(rec['model_order']),
                        'deform': (tuple(np.asarray(g[k]) for k in ('deform_om', 'deform_a', 'deform_ph'))
                                   if 'deform_om' in g else None)}
            nodes, tets, edges_ng, V = whitney_dofs(mesh, rec['u_model'], rec['model_order'])
        order = np.argsort(rec['freqs'], kind='stable')
        freqs, V = rec['freqs'][order], V[:, order]
        geom, M = extract_geometry_3d(nodes, tets)
        rows, sign = h5_edges_to_canonical(edges_ng, geom['n_nodes'], geom['edges'])
        U = np.zeros((len(geom['edges']), V.shape[1]))
        U[rows] = sign[:, None] * V
        names = [n for n in rec['qoi_names'] if n]
        labels = [{n: _float(rec['qoi'][k, j]) for j, n in enumerate(names)} for k in order]
        params = {k: _float(v) for k, v in attrs.items() if k not in _PARAM_SKIP and _float(v) is not None}
        geom.update(shape_type=rec['shape_type'], params=geom_params_of(attrs))
        return {'id': sid, 'family': rec['shape_type'], 'nodes': nodes, 'tets': tets, 'geom': geom, 'M': M,
                'U': U, 'f_GHz': freqs, 'f_next': _float(rec['freq_next']), 'params': params,
                'qoi_labels': labels, 'features': FEATURE_NAMES_3D, 'source_file': os.path.basename(fp),
                'mesh_src': mesh_src, 'n_tets_model': int(mesh.ne)}


def discover(spec=None):
    """{name: dataset} from RFCAV_DATA (or `spec`)."""
    spec = os.environ.get('RFCAV_DATA', '') if spec is None else spec
    out = {}

    def add(ds):
        name, i = ds.name, 2
        while name in out:
            name, i = f"{ds.name}#{i}", i + 1
        ds.name = name
        out[name] = ds

    for root in [p for p in spec.split(os.pathsep) if p]:
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.isfile(root):
            if root.endswith('.h5'):
                add((FieldH5Dataset if is_field_h5(root) else H5Dataset)([root], os.path.basename(root)))
            continue
        h5_dirs = {}
        for depth in range(5):
            pat = os.path.join(root, *(['*'] * depth))
            for fp in glob.glob(os.path.join(pat, '*.h5')):
                h5_dirs.setdefault(os.path.dirname(fp), []).append(fp)
        for dname, files in sorted(h5_dirs.items()):
            rel = os.path.relpath(dname, root)
            field = is_field_h5(sorted(files)[0])
            add((FieldH5Dataset if field else H5Dataset)(
                files, (rel if rel != '.' else os.path.basename(root)) + ('/ (field)' if field else '/ (h5)')))
    return out
