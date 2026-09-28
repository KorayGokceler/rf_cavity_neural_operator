"""3D views of generated cavity geometries (boundary surface of the tet mesh).

    from src.viz.geometry import load_h5, gallery, interactive
    geoms = load_h5("maxwell3d_..._part00.h5", n=12)          # one per family first
    gallery(geoms).savefig("gallery.png")                     # matplotlib, two views each
    interactive(geoms[0]).show()                              # plotly, rotatable (Colab/Jupyter)

or from the shell: python scripts/show_geometries.py data.h5 [--n 12] [--out gallery.png]
"""
import json

import h5py
import numpy as np

from src.data.dataset_converter_3d import boundary_faces

FAMILY_CMAP = {'pillbox': 'Blues', 'axisym_cell': 'Oranges', 'blob': 'Greens', 'elliptical': 'Oranges',
               'reentrant': 'Purples', 'pillbox_pipes': 'Blues', 'freeform': 'Greens',
               'box': 'Greys', 'coax_qw': 'Reds', 'pillbox_port': 'PuBu', 'elliptical_long': 'YlOrBr',
               'junction': 'BuGn'}  # last row: out-of-distribution test families
_LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def load_h5(path, n=12, ids=None):
    """Geometries of a dataset_generator_3d H5 as dicts {id, shape_type, nodes [mm],
    tets, faces (outward boundary triangles), freqs [GHz], params}; ids=None →
    round-robin over the families so each appears."""
    with h5py.File(path, 'r') as f:
        keys = sorted(k for k in f.keys() if 'tets' in f[k])
        if ids is not None:
            keys = [k for k in keys if int(k.split('_')[-1]) in set(ids)]
        else:
            fam = {}
            for k in keys:
                fam.setdefault(str(f[k].attrs['shape_type']), []).append(k)
            rr = [ks[i] for i in range(max(map(len, fam.values()), default=0)) for ks in fam.values() if i < len(ks)]
            keys = sorted(rr[:n], key=lambda k: (list(fam).index(str(f[k].attrs['shape_type'])), k))
        out = []
        for k in keys:
            g = f[k]
            X = g['nodes'][:] * 1e3
            T = g['tets'][:]
            out.append({'id': int(k.split('_')[-1]), 'shape_type': str(g.attrs['shape_type']), 'nodes': X,
                        'tets': T, 'faces': boundary_faces(X, T), 'freqs': g['freqs'][:],
                        'params': json.loads(g.attrs.get('geom_params', '{}'))})
    return out


def _draw(ax, g, elev, azim, cut=False):
    """cut=True: only the far half (y > centre) seen from −y, i.e. the inside wall
    (nose cones, irises) of a half-sectioned cavity."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    X, P = g['nodes'], g['nodes'][g['faces']]
    if cut:
        P = P[P.mean(1)[:, 1] > 0.5 * (X[:, 1].min() + X[:, 1].max())]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    e, a = np.radians(elev), np.radians(azim)              # light follows the camera
    cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    shade = 0.3 + 0.7 * np.clip(np.abs(n @ (0.6 * cam + 0.4 * _LIGHT)) if cut else n @ (0.6 * cam + 0.4 * _LIGHT), 0, 1)
    cmap = plt.get_cmap(FAMILY_CMAP.get(g['shape_type'], 'Greys'))
    ax.add_collection3d(Poly3DCollection(P, facecolors=cmap(0.3 + 0.65 * shade),
                                         edgecolor=(0, 0, 0, 0.1), linewidths=0.2))
    lo, hi = X.min(0), X.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(c[2] - r, c[2] + r))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.tick_params(labelsize=6)


def gallery(geoms, views=((25, -60), 'cut'), ncols=4):
    """Figure: every geometry from each view: (elev, azim) or 'cut' (half section,
    inside wall).  Default: outside from above + half section."""
    import matplotlib.pyplot as plt
    nv = len(views)
    per_row = max(1, ncols // nv) * nv
    n = len(geoms) * nv
    rows = -(-n // per_row)
    fig = plt.figure(figsize=(4 * per_row, 3.8 * rows))
    for i, g in enumerate(geoms):
        ext = np.round(g['nodes'].max(0) - g['nodes'].min(0)).astype(int)
        for j, v in enumerate(views):
            ax = fig.add_subplot(rows, per_row, i * nv + j + 1, projection='3d')
            cut = v == 'cut'
            el, az = (12, -90) if cut else v
            _draw(ax, g, el, az, cut)
            view = 'half section' if cut else ('above' if el >= 0 else 'below')
            ax.set_title(f"#{g['id']} {g['shape_type']} ({view})\n{ext[0]}×{ext[1]}×{ext[2]} mm, "
                         f"f1 = {g['freqs'][0]:.2f} GHz", fontsize=8)
    fig.tight_layout()
    return fig


def interactive(g):
    """plotly Mesh3d of one geometry (drag to rotate); needs plotly (preinstalled in Colab)."""
    import plotly.graph_objects as go
    X, F = g['nodes'], g['faces']
    fig = go.Figure(go.Mesh3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                              color={'pillbox': '#4a90d9', 'pillbox_pipes': '#4a90d9', 'axisym_cell': '#e8883a',
                                     'elliptical': '#e8883a', 'reentrant': '#8e6cc0'}.get(g['shape_type'], '#4caf50'),
                              flatshading=True, opacity=1.0,
                              lighting=dict(ambient=0.35, diffuse=0.8, specular=0.2)))
    fig.update_layout(title=f"#{g['id']} {g['shape_type']} · f = {np.round(g['freqs'][:4], 3)} GHz",
                      scene=dict(aspectmode='data', xaxis_title='x [mm]', yaxis_title='y [mm]',
                                 zaxis_title='z [mm]'), margin=dict(l=0, r=0, t=40, b=0), height=520)
    return fig
