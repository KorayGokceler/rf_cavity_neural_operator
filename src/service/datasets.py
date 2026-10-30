"""Dataset browsing for the development UI: generator field-label H5 shards (docs/29–30; file metadata
labels = 'field'), listed and opened one geometry at a time.

Registry: RFCAV_DATA = paths separated by os.pathsep (':' on Linux); each path is an .h5 file or a
directory searched (depth ≤ 4) for directories holding *.h5 shards — one dataset per H5 directory
(the TRUBA layout h5/<family>/<family>_s*.h5 gives one dataset per family). H5 files that are not
field-label H5 are skipped.

FieldH5Dataset: the table adds the label quality (adaptive levels, stop reason, error estimate, label
mesh, rounded / sharp edges), and a geometry opens on its model mesh with the label fields as stored (HCurl
p2 DOFs, drawn directly by src/viz/hcurl_display.py); the figures of merit shown are the stored p3 labels.
"""
import glob
import json
import os
import threading

import numpy as np

from src.data.mesh_features import geom_params_of

_PARAM_SKIP = {'geom_params', 'fem_element', 'field', 'shape_type'}


def _float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


_NG_LOCK = threading.Lock()          # NGSolve from the API's request threads: one call at a time


def is_field_h5(fp):
    """True for a field-label H5 (generator output)."""
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
                                'n_nodes': _attr(g, 'n_vertices', int, 0), 'n_dof': _attr(g, 'n_dof_model', int, 0),
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
        """Model mesh + the label fields as stored (HCurl DOFs of the model mesh, frequency order) + the stored p3
        figures of merit; mesh_src (the stored curved mesh: .vol + BREP bytes, curve order, deformation) lets the
        field model predict on the same mesh (src.service.model.ModelService.predict) and the display evaluate the
        fields; curved: the wall lattice, volume and bbox of the curved mesh (hcurl_display.curved_display); Mp: the
        physical HCurl mass matrix (energy, alignment with a prediction); mesh: the curved NGSolve mesh itself
        (in-process, for the display cache)."""
        import h5py
        import torch  # noqa: F401 — before NGSolve: ngsolve loaded first breaks torch later (tests/conftest.py)

        from src.data_gen.field_labels import read_field_group
        from src.service.geometry import linear_mesh
        from src.viz import hcurl_display as hd
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
            U = rec['u_model']
            from ngsolve import HCurl
            n_dof = HCurl(mesh, order=int(rec['model_order']), dirichlet='wall').ndof
            if n_dof != U.shape[0]:
                raise ValueError(f"stored fields have {U.shape[0]} DOFs, the rebuilt HCurl(p{rec['model_order']}) "
                                 f"space {n_dof}: different NGSolve version?")
            nodes, tets = linear_mesh(mesh)
            curved = hd.curved_display(mesh)
            Mp = hd.mass_matrix(mesh, rec['model_order'])
            n_tets = int(mesh.ne)
        order = np.argsort(rec['freqs'], kind='stable')
        freqs, U = rec['freqs'][order], U[:, order]
        names = [n for n in rec['qoi_names'] if n]
        labels = [{n: _float(rec['qoi'][k, j]) for j, n in enumerate(names)} for k in order]
        params = {k: _float(v) for k, v in attrs.items() if k not in _PARAM_SKIP and _float(v) is not None}
        return {'id': sid, 'family': rec['shape_type'], 'nodes': nodes, 'tets': tets, 'U': U, 'f_GHz': freqs,
                'f_next': _float(rec['freq_next']), 'params': params, 'geom_params': geom_params_of(attrs),
                'qoi_labels': labels, 'source_file': os.path.basename(fp), 'mesh_src': mesh_src,
                'n_tets_model': n_tets, 'n_dof': int(n_dof), 'order': int(rec['model_order']), 'curved': curved,
                'Mp': Mp, 'mesh': mesh}


def discover(spec=None):
    """{name: dataset} from RFCAV_DATA (or `spec`): field-label H5 only; other H5 directories / files are skipped."""
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
            if root.endswith('.h5') and is_field_h5(root):
                add(FieldH5Dataset([root], os.path.basename(root)))
            continue
        h5_dirs = {}
        for depth in range(5):
            pat = os.path.join(root, *(['*'] * depth))
            for fp in glob.glob(os.path.join(pat, '*.h5')):
                h5_dirs.setdefault(os.path.dirname(fp), []).append(fp)
        for dname, files in sorted(h5_dirs.items()):
            if not is_field_h5(sorted(files)[0]):
                continue
            rel = os.path.relpath(dname, root)
            add(FieldH5Dataset(files, (rel if rel != '.' else os.path.basename(root)) + '/ (field)'))
    return out
