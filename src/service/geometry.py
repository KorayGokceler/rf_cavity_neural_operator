"""Geometry side of the prediction service: CAD / mesh / generator geometry → tet mesh [m], checks,
display surface — shared by scripts/predict_geometry.py (CLI) and src/service/api.py (web).

Torch-free on purpose: `run_isolated` executes the meshing of untrusted uploads (OCC / gmsh /
meshio parsers) in a separate spawned process with a time and memory limit.
"""
import multiprocessing as mp
import os
import traceback

import numpy as np

from src.data.dataset_converter_3d import boundary_faces, tet_geometry
from src.data_gen import dataset_generator_3d as gen

UNITS = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3}
CAD_EXT = ('.step', '.stp', '.brep', '.iges', '.igs')
MESH_EXT = ('.msh', '.vtu', '.vtk', '.mesh', '.inp', '.med')


def gen_args(field, mesh_size, n_modes=6, families=None):
    """Set the generator's module-level ARGS (meshing helpers read them)."""
    gen.ARGS = gen.parse_args(['--field', field, '--mesh_size', str(mesh_size), '--n_eigen_modes', str(n_modes),
                               *(['--families', *families] if families else [])])
    return gen.ARGS


def mesh_step(path, unit, mesh_size, vol_div=1.0, h_abs=None):
    """CAD file → (nodes [m], tets): OCC import, fuse to one volume, gmsh tets (generator settings,
    h = mesh_size · (V / vol_div)^(1/3) or h_abs [m])."""
    import gmsh
    gen._gmsh_start()
    gmsh.clear()
    gmsh.model.occ.importShapes(path)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if not vols:
        raise ValueError("no 3D volume in the CAD file (a closed solid is needed: the cavity's vacuum region)")
    if len(vols) > 1:
        gmsh.model.occ.fuse(vols[:1], vols[1:])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
    V = sum(gmsh.model.occ.getMass(d, t) for d, t in vols)                    # file units³
    h = (h_abs / UNITS[unit]) if h_abs else mesh_size * (V / vol_div) ** (1 / 3)
    nodes, tets = gen._mesh_current_model(h)
    gmsh.clear()
    return nodes * UNITS[unit], tets


def mesh_file(path, unit):
    """Tetrahedral mesh file (meshio) → (nodes [m], tets); unused nodes dropped."""
    import meshio
    m = meshio.read(path)
    tets = [c.data for c in m.cells if c.type == 'tetra']
    if not tets:
        raise ValueError("no 'tetra' cells in the mesh file (linear tets needed)")
    tets = np.concatenate(tets).astype(np.int64)
    used = np.unique(tets)
    remap = np.full(len(m.points), -1, np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(m.points[used, :3], np.float64) * UNITS[unit], remap[tets]


def mesh_upload(path, unit='mm', mesh_size=0.10, vol_div=1.0, h_abs=None, field='E'):
    """Any supported upload (by extension) → (nodes [m], tets)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in CAD_EXT:
        gen_args(field, mesh_size)
        return mesh_step(path, unit, mesh_size, vol_div, h_abs)
    if ext in MESH_EXT:
        return mesh_file(path, unit)
    raise ValueError(f"unsupported file type {ext!r} (CAD: {', '.join(CAD_EXT)}; mesh: {', '.join(MESH_EXT)})")


def mesh_family(family, s_id, field='E', mesh_size=0.10):
    """Generator geometry (family, id) → (nodes [m], tets, params): the same shape as in the dataset."""
    gen_args(field, mesh_size, families=[family])
    g = gen.mesh_sample(int(s_id))
    return g['nodes'], g['tets'], {k: float(v) for k, v in g['params'].items()}


def from_h5(path, sample):
    """Generator H5 group (name or index) → (nodes, tets, truth dict)."""
    import h5py
    with h5py.File(path, 'r') as f:
        keys = sorted(k for k in f if k.startswith('sample_'))
        key = keys[int(sample)] if str(sample).isdigit() and str(sample) not in keys else str(sample)
        g = f[key]
        ds = 'e_edges' if 'e_edges' in g else 'h_edges'
        return (g['nodes'][()], g['tets'][()], {'edges': g['edges'][()], 'vecs': g[ds][()],
                                                 'freqs': g['freqs'][()], 'field': 'E' if ds == 'e_edges' else 'H',
                                                 'shape_type': str(g.attrs.get('shape_type', '')), 'key': key})


# ─────────────────────────── checks + display surface ──────────────────

def check(nodes, tets, field='E'):
    """Mesh facts and the formulation's validity certificate (no exception: 'ok' + 'problems')."""
    nodes = np.asarray(nodes, np.float64)
    tets = np.asarray(tets, np.int64)
    problems = []
    try:
        topo = gen.mesh_topology(tets, len(nodes))
    except Exception as e:                      # e.g. a face shared by more than two tets
        return {'ok': False, 'problems': [f"invalid mesh: {e}"], 'n_nodes': int(len(nodes)),
                'n_tets': int(len(tets))}
    if topo['n_components'] != 1:
        problems.append(f"{topo['n_components']} disconnected pieces (one cavity volume expected)")
    if not topo['manifold']:
        problems.append("non-manifold mesh")
    if field == 'H' and not gen.is_topological_ball(topo):
        problems.append("the H-field model needs a topological ball (no handles / holes)")
    vol, _ = tet_geometry(nodes, tets)
    lo, hi = nodes.min(0), nodes.max(0)
    return {'ok': not problems, 'problems': problems, 'n_nodes': int(len(nodes)), 'n_tets': int(len(tets)),
            'volume_cm3': float(vol.sum() * 1e6), 'bbox_mm': [(lo * 1e3).tolist(), (hi * 1e3).tolist()],
            'betti1': int(topo['betti1']), 'n_bnd_components': int(topo['n_shells'])}


def surface(nodes, tets):
    """Display surface: (vertex ids into nodes [Nb], triangles into those [Nf,3], outward)."""
    f = boundary_faces(np.asarray(nodes, np.float64), np.asarray(tets, np.int64))
    vid, tri = np.unique(f, return_inverse=True)
    return vid, tri.reshape(-1, 3)


# ─────────────────────────── isolation for untrusted uploads ───────────

def _vm_size():
    """Current virtual address-space size [bytes] (Linux; 0 elsewhere)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmSize:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _child(conn, fn, args, kwargs, mem_mb):
    try:
        if mem_mb:                              # mem_mb on top of what the process has mapped already
            import resource
            lim = _vm_size() + int(mem_mb) * 2 ** 20
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        conn.send(('ok', fn(*args, **kwargs)))
    except BaseException as e:                  # noqa: BLE001 — report everything to the parent
        conn.send(('err', f"{type(e).__name__}: {e}", traceback.format_exc(limit=3)))
    finally:
        conn.close()


def run_isolated(fn, *args, timeout=120.0, mem_mb=4096, **kwargs):
    """fn(*args, **kwargs) in a fresh spawned process (own gmsh / OCC state) with a wall-time and
    address-space limit; returns its result or raises RuntimeError / TimeoutError."""
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, fn, args, kwargs, mem_mb), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f"geometry processing exceeded {timeout:.0f} s")
        msg = parent.recv()
    except EOFError:
        raise RuntimeError("geometry processing crashed (out of memory or invalid input)") from None
    finally:
        if p.is_alive():
            p.kill()
        p.join(5)
    if msg[0] == 'err':
        raise RuntimeError(msg[1])
    return msg[1]
