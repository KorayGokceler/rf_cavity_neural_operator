"""How each model input feature is best shown (development UI) — not every feature is a scalar map:

  position  x, y, z            a coordinate system, not a field: the normalised frame (centroid, ±1 =
                                one `scale`) drawn as a 0.25-step grid of iso-lines on the cut, plus
                                the in-plane axes through the centroid.
  wall      dist_to_boundary   one vector feature d·n̂ (vector to the nearest wall point): arrows on the
            + dir_bnd_x/y/z    cut whose tips land on the wall, over the distance map with iso-distance
                                lines; the medial axis (where n̂ flips) shows as arrows splitting.
  torsion   torsion            a smooth scalar (−Δw = 1, w = 0 on ∂Ω): colour + iso-lines.
  mesh      node_volume        mesh information: the tets cut by the plane ("crinkle cut", edges
                                drawn), coloured by the lumped vertex volume the model sees.
  raw       any channel        the plain scalar on the cut (debugging).

Output (all lengths mm): plane {points [P,3], triangles [T,3], scalars [P]} | None,
cells {points, triangles, scalars} | None, segments [{role, points [S,2,3]}], range, signed, legend.
"""
import numpy as np

from src.data.dataset_converter_3d import FEATURE_NAMES_3D
from src.viz.nedelec import interp_vertex, locate, plane_grid

KINDS = ('position', 'wall', 'torsion', 'mesh', 'raw')
_AX = {'x': 0, 'y': 1, 'z': 2}


def _to_mm(geom, X):
    return (np.asarray(X) * geom['scale'] + np.asarray(geom['center']).reshape(1, 3)) * 1e3


def _grid(geom, axis, pos_mm, res):
    """Regular grid on the cut: (pts [res²,3] normalised, tid, bary, inside [res,res], U, V, iu, iv, off)."""
    ia = _AX[axis]
    c, s = float(np.asarray(geom['center']).reshape(3)[ia]), float(geom['scale'])
    off = 0.0 if pos_mm is None else (pos_mm * 1e-3 - c) / s
    pts, U, V, (iu, iv) = plane_grid(geom['X'], axis, off, res, pad=0.0)
    tid, bary = locate(geom['X'], geom['tets'], pts)
    return pts, tid, bary, (tid >= 0).reshape(U.shape), U, V, iu, iv, off


def _plane_mesh(geom, g, values):
    """Triangulated inside part of the grid + per-point values ([res²,…] → inside points)."""
    pts, tid, _, inside, U, *_ = g
    idx = -np.ones(inside.shape, np.int64)
    idx[inside] = np.arange(int(inside.sum()))
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    tri = np.concatenate([np.stack([a[full], b[full], d[full]], 1), np.stack([a[full], d[full], c[full]], 1)])
    sel = tid >= 0
    return {'points': _to_mm(geom, pts[sel]), 'triangles': tri, 'scalars': values[sel]}


def _vertex_values(geom, g, nodal):
    _, tid, bary, *_ = g
    return interp_vertex(geom['tets'], tid, bary, np.asarray(nodal, float))          # NaN outside


def _isolines(geom, g, Z, levels):
    """Contour segments [S,2,3] (mm) of the grid field Z [res,res] (NaN outside Ω) on the cut."""
    import contourpy
    _, _, _, _, U, V, iu, iv, off = g
    if not np.isfinite(Z).any():
        return np.zeros((0, 2, 3))
    gen = contourpy.contour_generator(U, V, np.ma.masked_invalid(Z), line_type=contourpy.LineType.Separate)
    segs = []
    for lv in levels:
        for line in gen.lines(float(lv)):
            if len(line) < 2:
                continue
            P = np.zeros((len(line), 3))
            P[:, iu], P[:, iv] = line[:, 0], line[:, 1]
            P[:, 3 - iu - iv] = off
            P = _to_mm(geom, P)
            segs.append(np.stack([P[:-1], P[1:]], 1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 3))


def _nice_levels(lo, hi, n=8):
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.array([])
    step = 10 ** np.floor(np.log10(span / n))
    step *= min((m for m in (1, 2, 2.5, 5, 10) if span / (m * step) <= n), default=10)
    return np.arange(np.ceil(lo / step) * step, hi, step)[1:] if lo == 0 else np.arange(np.ceil(lo / step) * step, hi, step)


def feature_view(geom, kind, axis='y', pos_mm=None, res=121, names=None, channel=None):
    """The best display of one feature group on an axis-aligned cut (see module docstring)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    names = list(names or FEATURE_NAMES_3D)
    F = np.asarray(geom['Input_funcs'], float)
    col = {n: F[:, i] for i, n in enumerate(names)}
    if kind == 'mesh':
        return _mesh_view(geom, col, axis, pos_mm)
    g = _grid(geom, axis, pos_mm, res)
    shape = g[3].shape
    out = {'kind': kind, 'plane': None, 'cells': None, 'segments': [], 'range': [0.0, 1.0], 'signed': False,
           'legend': ''}

    if kind == 'raw':
        if channel not in col:
            raise KeyError(channel)
        v = _vertex_values(geom, g, col[channel])
        out.update(plane=_plane_mesh(geom, g, v), range=[float(col[channel].min()), float(col[channel].max())],
                   signed=bool(col[channel].min() < 0), legend=channel)
        return out

    if kind == 'position':
        X = g[0]                                                     # grid points, normalised coordinates
        _, _, _, inside, U, V, iu, iv, off = g
        out['plane'] = _plane_mesh(geom, g, np.zeros(len(X)))
        out['neutral'] = True                                        # no colour map: the grid is the content
        lv = np.arange(-1.0, 1.0001, 0.25)
        for comp, role in ((iu, 'grid'), (iv, 'grid')):
            Z = np.where(inside, X[:, comp].reshape(shape), np.nan)
            out['segments'].append({'role': role, 'points': _isolines(geom, g, Z, lv[np.abs(lv) > 1e-9])})
            out['segments'].append({'role': 'axis', 'points': _isolines(geom, g, Z, [0.0])})
        c = np.zeros((1, 3))
        c[0, 3 - iu - iv] = off
        cm = _to_mm(geom, c)[0]
        L = float(geom['scale']) * 1e3 * 0.08
        mark = []
        for ax in (iu, iv):
            e = np.zeros(3)
            e[ax] = L
            mark.append([cm - e, cm + e])
        out['segments'].append({'role': 'marker', 'points': np.asarray(mark)})
        out.update(range=[0.0, 1.0], legend=f"normalised x, y, z: centroid = 0, ±1 = {geom['scale'] * 1e3:.1f} mm "
                                            "(grid every 0.25)")
        return out

    if kind == 'torsion':
        v = _vertex_values(geom, g, col['torsion'])
        out['plane'] = _plane_mesh(geom, g, v)
        Z = np.where(g[3], v.reshape(shape), np.nan)
        out['segments'].append({'role': 'iso', 'points': _isolines(geom, g, Z, np.arange(0.1, 0.95, 0.1))})
        out.update(range=[0.0, 1.0], legend='torsion w / max w (iso-lines every 0.1)')
        return out

    if kind == 'wall':
        d = _vertex_values(geom, g, col['dist_to_boundary'])
        n = _vertex_values(geom, g, np.stack([col['dir_bnd_x'], col['dir_bnd_y'], col['dir_bnd_z']], 1))
        out['plane'] = _plane_mesh(geom, g, d)
        Z = np.where(g[3], d.reshape(shape), np.nan)
        dmax = float(np.nanmax(col['dist_to_boundary']))
        out['segments'].append({'role': 'iso', 'points': _isolines(geom, g, Z, _nice_levels(0.0, dmax, 6))})
        # arrows on a coarser sub-grid: from the point to its nearest wall point (d · n̂, mm)
        step = max(1, res // 22)
        sub = np.zeros(shape, bool)
        sub[::step, ::step] = True
        sel = (sub & g[3]).ravel()
        P = _to_mm(geom, g[0][sel])
        nn = n[sel]
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        vec = nn * (d[sel] * float(geom['scale']) * 1e3)[:, None]
        tip = P + vec
        normal = np.eye(3)[_AX[axis]]
        side = np.cross(normal, nn)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        Lh = np.minimum(np.linalg.norm(vec, axis=1), float(geom['scale']) * 1e3 * 0.06)[:, None] * 0.35
        heads = [np.stack([tip, tip - Lh * nn + 0.6 * Lh * side], 1), np.stack([tip, tip - Lh * nn - 0.6 * Lh * side], 1)]
        out['segments'].append({'role': 'arrow', 'points': np.concatenate([np.stack([P, tip], 1), *heads])})
        out.update(range=[0.0, dmax], legend='distance to the wall (normalised) · arrows: vector to the nearest wall point')
        return out

    raise AssertionError(kind)


def _mesh_view(geom, col, axis, pos_mm):
    """Tets crossing the cut plane, faces drawn with edges, coloured by node_volume."""
    ia = _AX[axis]
    c, s = float(np.asarray(geom['center']).reshape(3)[ia]), float(geom['scale'])
    off = 0.0 if pos_mm is None else (pos_mm * 1e-3 - c) / s
    X, T = np.asarray(geom['X'], float), np.asarray(geom['tets'], np.int64)
    zc = X[T, ia]
    cut = (zc.min(1) <= off) & (zc.max(1) >= off)
    faces = T[cut][:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    vid, inv = np.unique(faces, return_inverse=True)
    return {'kind': 'mesh', 'plane': None, 'segments': [], 'signed': False,
            'cells': {'points': _to_mm(geom, X[vid]), 'triangles': inv.reshape(-1, 3),
                      'scalars': col['node_volume'][vid]},
            'range': [float(col['node_volume'].min()), 1.0],
            'legend': f"node_volume (lumped vertex volume / max) · {int(cut.sum())} tets cut by the plane"}
