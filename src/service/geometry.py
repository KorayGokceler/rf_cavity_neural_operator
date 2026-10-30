"""Geometry side of the prediction service: CAD / mesh / generator geometry → tet mesh [m], checks,
display surface — shared by scripts/predict_geometry.py (CLI) and src/service/api.py (web).

The field3d model predicts on a CURVED netgen mesh of the CAD solid (docs/28–29), so every geometry that
can be predicted keeps its curved mesh as bytes (`mesh_src`: netgen .vol of the straight tets + the BREP
of the meshed solid [m] + curve order + deformation, the H5 field-label storage) and its display mesh is
the linear (vertex) mesh of that curved mesh — prediction, display and the dataset's FE truth share one
vertex / tet numbering.  Geometries without a CAD solid (mesh files, the discrete freeform family) are
meshed with gmsh / meshio for display and checks only (`blocker` says why they cannot be predicted).

Torch-free on purpose: `run_isolated` executes the meshing of untrusted uploads (OCC / gmsh / netgen /
meshio parsers) in a separate spawned process with a time and memory limit; results are plain arrays
and bytes (picklable), never netgen objects.
"""
import multiprocessing as mp
import os
import traceback

import numpy as np

from src.data.dataset_converter_3d import boundary_faces, tet_geometry
from src.data_gen import dataset_generator_3d as gen

UNITS = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3}
CAD_EXT = ('.step', '.stp', '.brep', '.iges', '.igs')
UNIT_FILE_EXT = ('.step', '.stp', '.iges', '.igs')    # declare their length unit: OCC converts on import
MESH_EXT = ('.msh', '.vtu', '.vtk', '.mesh', '.inp', '.med')


def unit_used(path, unit):
    """Effective length unit of a geometry file: 'file' for STEP / IGES (the unit the file declares, converted
    to metres on import; the user's `unit` is ignored), else `unit` (BREP has no unit — the generator's
    BREPs are in metres — and mesh files carry none)."""
    return 'file' if os.path.splitext(str(path))[1].lower() in UNIT_FILE_EXT else unit


def _occ_import(path, unit):
    """gmsh OCC import of a CAD file with STEP / IGES read in METRES: OpenCASCADE converts STEP / IGES to its
    target unit (millimetres by default) whatever unit the file declares, so the target is set to 'M' for
    the import and put back to OCC's default 'MM' afterwards (the option is process-wide).  BREP has no unit
    and is read as it is.  Returns the factor to metres still to apply for `unit` (1 for STEP / IGES)."""
    import gmsh
    step = unit_used(path, unit) == 'file'
    if step:
        gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
    try:
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
    finally:
        if step:
            gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    return 1.0 if step else UNITS[unit]


def linear_mesh(mesh):
    """(vertices [Nv,3] float64, tets [Nt,4] int64) of an NGSolve mesh: the straight mesh under a curved
    one, in NGSolve's vertex numbering (src.service.datasets.whitney_dofs uses the same)."""
    from ngsolve import VOL
    nodes = np.array([tuple(v.point) for v in mesh.vertices], np.float64)
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)], np.int64)
    return nodes, tets


def curved_from_cad(cad_path, h, deform=None, max_elements=None):
    """Curved netgen mesh of a CAD solid [m] with the field-label settings (field_labels.start_mesh: the
    adaptive labels' start mesh, maxh = maxh_factor·h, curvaturesafety 1.5 then 2) → dict(nodes, tets
    [linear mesh], mesh_src {vol, brep (bytes), curve, deform}, n_tets).  Raises when netgen fails, folds
    elements or exceeds max_elements (default field_labels.DEFAULTS['max_elements'])."""
    import tempfile

    from src.data_gen import field_labels as fl
    s = dict(fl.DEFAULTS, **({'max_elements': int(max_elements)} if max_elements else {}))
    mesh, shape, _ = fl.start_mesh(cad_path, h, deform, s, (s['adapt_cs'], s['curvaturesafety']))
    with tempfile.TemporaryDirectory() as tmp:
        brep = os.path.join(tmp, 's.brep')
        shape.WriteBrep(brep)
        brep_bytes = open(brep, 'rb').read()
    nodes, tets = linear_mesh(mesh)
    return {'nodes': nodes, 'tets': tets, 'n_tets': int(mesh.ne),
            'mesh_src': {'vol': fl._mesh_bytes(mesh), 'brep': brep_bytes, 'curve': int(s['curve']),
                         'deform': deform}}


def load_mesh_src(src):
    """The curved NGSolve mesh of a `mesh_src` dict (highorder.load_curved_mesh: identical numbering)."""
    import tempfile

    from src.data_gen import highorder as ho
    with tempfile.TemporaryDirectory() as tmp:
        vol, brep = os.path.join(tmp, 'm.vol'), os.path.join(tmp, 's.brep')
        with open(vol, 'wb') as fh:
            fh.write(src['vol'])
        with open(brep, 'wb') as fh:
            fh.write(src['brep'])
        return ho.load_curved_mesh(vol, brep, int(src['curve']), src.get('deform'))


def mesh_step(path, unit, mesh_size, vol_div=1.0, h_abs=None):
    """CAD file → (nodes [m], tets): OCC import, fuse to one volume, gmsh tets (generator settings,
    h = mesh_size · (V / vol_div)^(1/3) or h_abs [m]).  `unit` applies to BREP only (unit_used)."""
    import gmsh
    gen._gmsh_start()
    gmsh.clear()
    f = _occ_import(path, unit)
    vols = gmsh.model.getEntities(3)
    if not vols:
        raise ValueError("no 3D volume in the CAD file (a closed solid is needed: the cavity's vacuum region)")
    if len(vols) > 1:
        gmsh.model.occ.fuse(vols[:1], vols[1:])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
    V = sum(gmsh.model.occ.getMass(d, t) for d, t in vols)                    # model units³
    h = (h_abs / f) if h_abs else mesh_size * (V / vol_div) ** (1 / 3)
    nodes, tets = gen._mesh_current_model(h)
    gmsh.clear()
    return nodes * f, tets


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


def cad_to_brep(path, unit, out_path):
    """CAD file → one solid as BREP in metres at out_path (gmsh OCC import — STEP / IGES in the unit they
    declare, BREP scaled by `unit` — and fuse); returns (volume [m³], re-entrant edges left sharp).
    The sharp edges are counted as the generator counts them (cavity_shapes.round_reentrant_edges with
    fillet=False, radius MIN_FILLET-like 0.05·equivalent-sphere radius): E and H are singular there, so the
    surface peaks of such a solid are not converged (its figures of merit leave them out, as the labels)."""
    import gmsh

    from src.data_gen import cavity_shapes as cs
    gen._gmsh_start()
    gmsh.clear()
    f = _occ_import(path, unit)
    vols = gmsh.model.getEntities(3)
    if not vols:
        raise ValueError("no 3D volume in the CAD file (a closed solid is needed: the cavity's vacuum region)")
    if len(vols) > 1:
        gmsh.model.occ.fuse(vols[:1], vols[1:])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
    if f != 1.0:
        gmsh.model.occ.dilate(gmsh.model.getEntities(), 0.0, 0.0, 0.0, f, f, f)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
    V = sum(gmsh.model.occ.getMass(d, t) for d, t in vols)
    n_sharp = 0
    if len(vols) == 1:
        r_edge = 0.05 * (3 * V / (4 * np.pi)) ** (1 / 3)
        n_sharp = int(cs.round_reentrant_edges(gmsh.model.occ, vols[0][1], r_edge, fillet=False)[3])
    gmsh.write(str(out_path))
    gmsh.clear()
    return V, n_sharp


def mesh_upload(path, unit='mm', mesh_size=0.10, vol_div=1.0, h_abs=None, max_elements=None):
    """Any supported upload (by extension) → dict(nodes [m], tets, mesh_src | None, blocker | None, n_tets,
    unit_used, n_sharp_edges).  CAD: the solid in metres (STEP / IGES: the unit the file declares, `unit`
    ignored; BREP: `unit`), curved netgen mesh at h = mesh_size·(V / vol_div)^(1/3) (or h_abs [m]) — when
    that fails, the gmsh mesh for display and checks, with the reason as blocker.  Mesh files: no CAD
    surface, so no curved mesh (blocker)."""
    import tempfile
    ext = os.path.splitext(path)[1].lower()
    used = unit_used(path, unit)
    if ext in CAD_EXT:
        with tempfile.TemporaryDirectory() as tmp:
            brep = os.path.join(tmp, 'solid.brep')
            V, n_sharp = cad_to_brep(path, unit, brep)
            h = float(h_abs) if h_abs else mesh_size * (V / vol_div) ** (1 / 3)
            try:
                return dict(curved_from_cad(brep, h, max_elements=max_elements), blocker=None, unit_used=used,
                            n_sharp_edges=n_sharp)
            except Exception as e:                     # noqa: BLE001 — reported as the predict blocker
                blocker = f"no curved mesh of the CAD solid ({type(e).__name__}: {e})"
        nodes, tets = mesh_step(path, unit, mesh_size, vol_div, h_abs)
        return {'nodes': nodes, 'tets': tets, 'mesh_src': None, 'blocker': blocker, 'n_tets': int(len(tets)),
                'unit_used': used, 'n_sharp_edges': n_sharp}
    if ext in MESH_EXT:
        nodes, tets = mesh_file(path, unit)
        return {'nodes': nodes, 'tets': tets, 'mesh_src': None, 'n_tets': int(len(tets)), 'unit_used': used,
                'n_sharp_edges': None,
                'blocker': 'a mesh file has no CAD surface: the field model needs the curved geometry of a solid '
                           f"({', '.join(CAD_EXT)})"}
    raise ValueError(f"unsupported file type {ext!r} (CAD: {', '.join(CAD_EXT)}; mesh: {', '.join(MESH_EXT)})")


NO_CAD_BLOCKER = 'the {} family is a discrete surface without a CAD solid: the field model needs the curved geometry'


def mesh_family(family, s_id, mesh_size=0.10, max_elements=None):
    """Generator geometry (family, id) with the field-label settings (rounded re-entrant edges, healing, Sobol
    sampling, seed 0: the geometry of the field datasets) and its curved mesh → dict(nodes [m], tets, params,
    mesh_src, blocker, n_tets).  A curved mesh that fails re-draws the geometry, as the generator does.
    The discrete families (freeform) have no CAD: gmsh mesh for display only (blocker)."""
    import tempfile
    if family in gen.DISCRETE_BUILDERS:
        gen.ARGS = gen.parse_args(['--mesh_size', str(mesh_size), '--families', family])
        g = gen.mesh_sample(int(s_id))
        return {'nodes': g['nodes'], 'tets': g['tets'], 'params': {k: float(v) for k, v in g['params'].items()},
                'mesh_src': None, 'n_tets': int(len(g['tets'])), 'blocker': NO_CAD_BLOCKER.format(family)}
    gen.ARGS = gen.parse_args(['--mesh_size', str(mesh_size), '--families', family, '--sampling', 'sobol'])
    with tempfile.TemporaryDirectory() as tmp:
        g = gen.mesh_sample(int(s_id), cad_path=os.path.join(tmp, 'solid.brep'),
                            post=lambda path, h, dmap, _st, _prm: curved_from_cad(path, h, dmap, max_elements))
    return dict(g['post'], params={k: float(v) for k, v in g['params'].items()}, blocker=None)


# ─────────────────────────── checks + display surface ──────────────────

def check(nodes, tets):
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
