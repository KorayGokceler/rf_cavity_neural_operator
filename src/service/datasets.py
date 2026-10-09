"""Dataset browsing for the development UI: generator H5 shards and converted training PKLs, listed
and opened one geometry at a time (mesh, model input features, FE modes, QoI labels, parameters).

Registry: RFCAV_DATA = paths separated by os.pathsep (':' on Linux); each path is a .pkl file, an
.h5 file, or a directory searched (depth ≤ 4) for *.pkl files and for directories holding *.h5
shards — one dataset per PKL and per H5 directory (the TRUBA layout h5/<family>/<family>_s*.h5
gives one dataset per family).
"""
import glob
import json
import os
import pickle
import threading

import numpy as np

from src.data.dataset_converter_3d import FEATURE_NAMES_3D, extract_geometry_3d, h5_edges_to_canonical, require_e_field

MAX_PKL_GB = float(os.environ.get('RFCAV_MAX_PKL_GB', 8))
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
        geom.update(shape_type=str(attrs.get('shape_type', '')))
        return {'id': sid, 'family': geom['shape_type'], 'nodes': np.asarray(nodes, float),
                'tets': np.asarray(tets, np.int64), 'geom': geom, 'M': M, 'U': U, 'f_GHz': freqs,
                'f_next': _float(attrs.get('freq_next')), 'params': params, 'qoi_labels': None,
                'features': FEATURE_NAMES_3D, 'source_file': os.path.basename(fp)}


class PKLDataset:
    kind = 'pkl'

    def __init__(self, path, name):
        self.path, self.name = path, name
        self._d, self._lock = None, threading.Lock()

    def _data(self):
        with self._lock:
            if self._d is None:
                gb = os.path.getsize(self.path) / 2 ** 30
                if gb > MAX_PKL_GB:
                    raise MemoryError(f"{self.name}: {gb:.1f} GB > RFCAV_MAX_PKL_GB={MAX_PKL_GB:g}; "
                                      "browse its H5 shards instead")
                with open(self.path, 'rb') as f:
                    d = pickle.load(f)
                require_e_field(d.get('metadata'))
                by = {}
                for s in d['samples']:
                    by.setdefault(s['geom_id'], []).append(s)
                d['_by_geom'] = {g: sorted(v, key=lambda s: float(s['Theta'][0])) for g, v in by.items()}
                self._d = d
            return self._d

    def rows(self):
        d = self._data()
        out = []
        for gid, g in d['geometry_pool'].items():
            ss = d['_by_geom'].get(gid, [])
            out.append({'id': int(gid), 'family': str(g.get('shape_type', '')), 'n_nodes': int(len(g['X'])),
                        'n_edges': int(len(g['edges'])), 'n_tets': int(len(g['tets'])),
                        'betti1': int(g.get('betti1', 0) or 0), 'deformed': None,
                        'f_GHz': [float(s['Theta'][1]) for s in ss]})
        return out

    def open(self, gid):
        from src.data.dataset_3d import rebuild_operators, to_csr
        d = self._data()
        key = next((k for k in d['geometry_pool'] if int(k) == int(gid)), None)
        if key is None:
            raise KeyError(gid)
        g = dict(d['geometry_pool'][key])
        ne = len(g['edges'])
        if 'M' not in g:                                      # lean PKL: rebuild the operators
            g.update(rebuild_operators(g['X'], g['tets']))
        M = to_csr(g['M'], (ne, ne))
        ss = d['_by_geom'].get(key, [])
        U = np.stack([np.asarray(s['Y'], float) for s in ss], 1) if ss else np.zeros((ne, 0))
        nodes = np.asarray(g['X'], float) * float(g['scale']) + np.asarray(g['center'], float).reshape(1, 3)
        names = list((d.get('metadata') or {}).get('feature_names') or FEATURE_NAMES_3D)
        labels = [s.get('qoi') for s in ss] if ss and 'qoi' in ss[0] else None
        return {'id': int(gid), 'family': str(g.get('shape_type', '')), 'nodes': nodes,
                'tets': np.asarray(g['tets'], np.int64), 'geom': g, 'M': M, 'U': U,
                'f_GHz': np.array([float(s['Theta'][1]) for s in ss]), 'f_next': _float(g.get('freq_next')),
                'params': {}, 'qoi_labels': labels, 'features': names, 'source_file': os.path.basename(self.path)}


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
            if root.endswith('.pkl'):
                add(PKLDataset(root, os.path.basename(root)))
            elif root.endswith('.h5'):
                add(H5Dataset([root], os.path.basename(root)))
            continue
        h5_dirs = {}
        for depth in range(5):
            pat = os.path.join(root, *(['*'] * depth))
            for fp in glob.glob(os.path.join(pat, '*.pkl')):
                add(PKLDataset(fp, os.path.relpath(fp, root)))
            for fp in glob.glob(os.path.join(pat, '*.h5')):
                h5_dirs.setdefault(os.path.dirname(fp), []).append(fp)
        for dname, files in sorted(h5_dirs.items()):
            rel = os.path.relpath(dname, root)
            add(H5Dataset(files, (rel if rel != '.' else os.path.basename(root)) + '/ (h5)'))
    return out
