"""Drawing the HCurl(p2) field directly, without any Whitney-N0 projection (docs/26 "Display pipeline").

The web UI shows the same high-order field the model predicts and the generator stores.  Everything here
works on the CURVED NGSolve mesh (curve order 3, optional smooth deformation) and on HCurl(order, 'wall')
DOF vectors of that mesh, in physical units [m]:

    display_mesh(mesh)        the P2 nodes of every tet (4 vertices + 6 edge midpoints, mapped through the
                              curved element map, so deformed samples are drawn deformed) and 8 sub-tets per
                              tet (4 corner tets + the interior octahedron split along its shortest diagonal):
                              a conforming linear mesh on half the element size that carries the P2 geometry
    nodal_fields(...)         E and H at those nodes (exact p2 evaluation, averaged where elements meet), at
                              stored energy U = 1 J; the browser interpolates linearly inside the sub-tets
    surface_lattice(mesh)     the wall as a level-r triangle lattice on every curved boundary face (the curved
                              surface, not its linear facets), welded; surface_values: |E|, |H| there from
                              the adjacent volume element (BoundaryFromVolumeCF, as Epk / Bpk)
    axis_profile(...)         E along the beam axis (NaN outside)
    mass_matrix, field_energy U = ½ε0 ∫|E|² from the DOFs; eval_points: exact evaluation anywhere (/plane, /volume)

Reference tetrahedron of NGSolve: (1,0,0), (0,1,0), (0,0,1), (0,0,0) are the element's vertices 0..3
(highorder.field_operators); the reference triangle (1,0), (0,1), (0,0) the boundary element's vertices 0..2.

Honest limit: the drawn field is P1 on the sub-tets of the P2 nodes.  Its error is dominated by that
interpolation, ~1e-3 of the peak on fine meshes, ~10 times the p2 solution's own error there (6.7k tets: 1.5e-3 rms
against 1.2e-4); /plane and /volume (eval_points) are exact.  Averaging at the nodes smooths the real jump of the normal E between
elements (0.9 % at the wall), which is why the wall itself is drawn from the unaveraged boundary path.

`import torch` before ngsolve in any process that uses both (tests/conftest.py).
"""
import numpy as np
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0
from scipy.sparse import csr_matrix

LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))     # local node 4 + i = midpoint of LOCAL_EDGES[i]
_REF_VERTICES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float64)
REF_P2 = np.vstack([_REF_VERTICES, [0.5 * (_REF_VERTICES[a] + _REF_VERTICES[b]) for a, b in LOCAL_EDGES]])   # [10, 3]
CORNER_TETS = ((0, 4, 5, 6), (1, 4, 7, 8), (2, 5, 7, 9), (3, 6, 8, 9))
# interior octahedron: opposite midpoint pairs and the ring of the other four around each diagonal
DIAGONALS = (((4, 9), (5, 6, 8, 7)), ((5, 8), (4, 6, 9, 7)), ((6, 7), (4, 5, 9, 8)))
WALL_NAMES = ("wall",)
MAX_LATTICE_TRIANGLES = 150_000


# ───────────────────────── P2 nodes + sub-tets ─────────────────────────

def element_vertices(mesh):
    """tets [Nt, 4] int64 in NGSolve's vertex numbering and element order (= the order of MapToAllElements)."""
    from ngsolve import VOL
    return np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)], np.int64).reshape(-1, 4)


def _subtet_table(diag):
    """Sub-tets [8, 4] (local node ids 0..9) of one element given its diagonal choice (0..2)."""
    (d0, d1), ring = DIAGONALS[diag]
    return np.array(list(CORNER_TETS) + [(d0, d1, ring[i], ring[(i + 1) % 4]) for i in range(4)], np.int64)


def display_mesh(mesh):
    """P2 display mesh of a curved mesh (positions in metres).

    dict: x [Nn, 3] node positions (mapped through the element maps, shared nodes averaged), tets [8·Nt, 4]
    sub-tets (positively oriented), elem_nodes [Nt, 10] global node id of each element's local nodes
    (vertex i → i, edge e → Nv + e), avg (CSR [Nn, 10·Nt]: element-point values → node means), mips (the
    mapped points, in-process only), n_vertices, n_tets, centroid (volume centroid of the sub-tets), volume.
    """
    from ngsolve import VOL, CF, IntegrationRule, x as X_, y as Y_, z as Z_
    tets = element_vertices(mesh)
    nt, nv = len(tets), int(mesh.nv)
    ev = np.array([[v.nr for v in e.vertices] for e in mesh.edges], np.int64).reshape(-1, 2)
    ekey = ev.min(1) * nv + ev.max(1)
    order = np.argsort(ekey)
    skey = ekey[order]
    elem_nodes = np.empty((nt, 10), np.int64)
    elem_nodes[:, :4] = tets
    for i, (a, b) in enumerate(LOCAL_EDGES):
        va, vb = tets[:, a], tets[:, b]
        key = np.minimum(va, vb) * nv + np.maximum(va, vb)
        pos = np.searchsorted(skey, key)
        if np.any(pos >= len(skey)) or np.any(skey[np.minimum(pos, len(skey) - 1)] != key):
            raise ValueError("an element edge is not an edge of the mesh")
        elem_nodes[:, 4 + i] = nv + order[pos]
    n_nodes = nv + int(mesh.nedge)
    ir = IntegrationRule([tuple(p) for p in REF_P2], [1.0 / 10] * 10)
    mips = mesh.MapToAllElements(ir, VOL)
    pts = np.asarray(CF((X_, Y_, Z_))(mips), np.float64).reshape(nt * 10, 3)
    rows = elem_nodes.reshape(-1)
    avg = csr_matrix((np.ones(nt * 10), (rows, np.arange(nt * 10))), shape=(n_nodes, nt * 10))
    cnt = np.asarray(avg.sum(1)).ravel()
    avg = csr_matrix((1.0 / cnt[rows], (rows, np.arange(nt * 10))), shape=(n_nodes, nt * 10))
    x = avg @ pts
    # sub-tets, the interior diagonal along the shortest of the three midpoint pairs
    xe = x[elem_nodes]                                                        # [Nt, 10, 3]
    dlen = np.stack([np.linalg.norm(xe[:, a] - xe[:, b], axis=1) for (a, b), _ in DIAGONALS], 1)
    diag = dlen.argmin(1)
    tables = np.stack([_subtet_table(d) for d in range(3)])             # [3, 8, 4]
    local = tables[diag]                                                      # [Nt, 8, 4]
    sub = np.take_along_axis(elem_nodes[:, None, :].repeat(8, 1), local, axis=2).reshape(-1, 4)
    p = x[sub]
    vol = np.einsum("ni,ni->n", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0
    flip = vol < 0
    sub[flip] = sub[flip][:, [0, 1, 3, 2]]
    vol = np.abs(vol)
    return {"x": x, "tets": sub, "elem_nodes": elem_nodes, "avg": avg, "mips": mips, "n_vertices": nv,
            "n_tets": nt, "volume": float(vol.sum()),
            "centroid": (vol[:, None] * p.mean(1)).sum(0) / float(vol.sum())}


# ───────────────────────── fields at the nodes ─────────────────────────

def _gridfunction(mesh, order, u):
    from ngsolve import GridFunction, HCurl
    g = GridFunction(HCurl(mesh, order=int(order), dirichlet="wall"))
    if g.space.ndof != len(u):
        raise ValueError(f"{len(u)} DOFs for an HCurl(p{order}) space of {g.space.ndof}: a different mesh or NGSolve?")
    g.vec.FV().NumPy()[:] = u
    return g


def mass_matrix(mesh, order=2):
    """Physical mass matrix [n × n] (scipy CSR) of HCurl(order, 'wall'): ∫ u·v dx with the one explicit tet rule
    highorder.volume_measure — u^T M u = ∫|E|² dV [m³] for the DOF vector u (the field operators' M is this / scale)."""
    from ngsolve import BilinearForm, HCurl

    from src.data_gen import highorder as ho
    fes = HCurl(mesh, order=int(order), dirichlet="wall")
    u, v = fes.TnT()
    M = BilinearForm(u * v * ho.volume_measure(order), symmetric=True).Assemble()
    return ho._tosp(M.mat)


def field_energy(Mp, U):
    """Stored energy ½ε0 ∫|E|² [J] of each column of U [n, K] for the physical mass matrix Mp."""
    U = np.asarray(U, np.float64).reshape(len(U), -1)
    return 0.5 * EPS0 * np.einsum("ik,ik->k", U, Mp @ U)


def energy_scale(Mp, U):
    """a_k with E = a_k · E_raw at U = 1 J: [K]."""
    return 1.0 / np.sqrt(field_energy(Mp, U))


def _faraday(f_ghz):
    """1/(ω μ0) per mode: H = −(a/(ω μ0)) curl E_raw (E(t) = E cos φ, H(t) = H sin φ: the sign is folded in H)."""
    return 1.0 / (2 * np.pi * np.asarray(f_ghz, np.float64) * 1e9 * MU0)


def nodal_fields(mesh, dm, U, f_ghz, Mp, order=2):
    """E and H at the P2 nodes at U = 1 J.  U [n, K] HCurl DOFs (any normalisation), f_ghz [K], Mp = mass_matrix.
    dict: E, H [K, Nn, 3] float32 [V/m, A/m], a [K] (energy scale), plus the raw per-element values are dropped."""
    from ngsolve import curl
    U = np.asarray(U, np.float64).reshape(len(U), -1)
    a = energy_scale(Mp, U)
    fa = a * _faraday(f_ghz)
    E = np.empty((U.shape[1], dm["x"].shape[0], 3), np.float32)
    H = np.empty_like(E)
    for k in range(U.shape[1]):
        g = _gridfunction(mesh, order, U[:, k])
        e = np.asarray(g(dm["mips"]), np.float64).reshape(-1, 3)
        c = np.asarray(curl(g)(dm["mips"]), np.float64).reshape(-1, 3)
        E[k] = a[k] * (dm["avg"] @ e)
        H[k] = -fa[k] * (dm["avg"] @ c)
    return {"E": E, "H": H, "a": a}


def locate(mesh, pts, deform=None, n_iter=8):
    """highorder.locate for physical points; deform = the mesh's (om, a, ph) when SetDeformation is active.
    NGSolve's point search works in the undeformed geometry (and evaluates a field through the deformed element
    map there), so a deformed sample is searched at the preimage x_ref + δ(x_ref) = x (Newton, ‖∇δ‖ < 1)."""
    from src.data_gen import highorder as ho
    pts = np.ascontiguousarray(pts, np.float64)
    if deform is not None:
        om, a, ph = (np.asarray(v, np.float64) for v in deform)
        ref = pts.copy()
        for _ in range(n_iter):
            arg = ref @ om.T + ph                                                  # [N, n]
            res = ref + np.sin(arg) @ a - pts
            if np.abs(res).max() < 1e-13:
                break
            J = np.eye(3)[None] + np.einsum("nm,mc,md->ncd", np.cos(arg), a, om)   # I + ∇δ
            ref = ref - np.linalg.solve(J, res[:, :, None])[:, :, 0]
        pts = ref
    return ho.locate(mesh, pts)


def eval_points(mesh, U, f_ghz, Mp, pts, order=2, deform=None):
    """Exact E and H [V/m, A/m] of the modes U [n, K] at physical points pts [N, 3] (U = 1 J):
    (E [K, N, 3], H [K, N, 3], inside [N] bool; 0 outside the curved mesh).  deform: see locate."""
    from ngsolve import curl

    from src.data_gen import highorder as ho
    U = np.asarray(U, np.float64).reshape(len(U), -1)
    a = energy_scale(Mp, U)
    fa = a * _faraday(f_ghz)
    loc = locate(mesh, pts, deform)
    E = np.zeros((U.shape[1], len(pts), 3))
    H = np.zeros_like(E)
    for k in range(U.shape[1]):
        g = _gridfunction(mesh, order, U[:, k])
        E[k] = a[k] * ho.evaluate(g, loc)
        H[k] = -fa[k] * ho.evaluate(curl(g), loc)
    return E, H, loc[1]


# ───────────────────────── the curved wall ─────────────────────────

def lattice_level(n_faces, max_triangles=MAX_LATTICE_TRIANGLES):
    """Subdivision level r of the lattice: 4 up to ~9k faces, 3 above, 2 beyond that; ≤ max_triangles triangles."""
    for r in (4, 3, 2, 1):
        if r * r * n_faces <= max_triangles:
            return r
    return 1


def _lattice_topology(r):
    """(ref [(r+1)(r+2)/2, 2] points (i/r, j/r), tri [r², 3] indices) of the level-r lattice of the reference triangle."""
    idx = {}
    ref = []
    for i in range(r + 1):
        for j in range(r + 1 - i):
            idx[(i, j)] = len(ref)
            ref.append((i / r, j / r))
    tri = [(idx[i, j], idx[i + 1, j], idx[i, j + 1]) for i in range(r) for j in range(r - i)]
    tri += [(idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]) for i in range(r - 1) for j in range(r - 1 - i)]
    return np.array(ref), np.array(tri, np.int64)


def _weld(points, tol):
    """(group count, group id [N]) of points closer than tol (connected components of the pair graph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    n = len(points)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


def surface_lattice(mesh, level=None, names=WALL_NAMES):
    """The wall of the curved mesh as a welded triangle lattice.

    dict: x [Ns, 3] points [m] (mapped through the curved boundary elements), tris [T, 3] outward-oriented,
    vid3 [Ns, 3] / w3 [Ns, 3] (the three linear vertices of one boundary face holding the point and its barycentric
    weights: P1 interpolation of vertex data onto the lattice), group [Ncopy] (lattice point of each
    (face, local point) copy), rows [Ncopy] (row of that copy in the evaluation of all boundary elements),
    mips (in-process), level.  Faces of other boundary names (the internal "interface" cuts) are left out."""
    from ngsolve import BND, CF, IntegrationRule, specialcf, x as X_, y as Y_, z as Z_
    faces = list(mesh.Elements(BND))
    nf_all = len(faces)
    keep_names = set(names)
    wall = np.array([f.mat in keep_names for f in faces], bool) if keep_names else np.ones(nf_all, bool)
    if not wall.any():
        wall[:] = True                                   # an unnamed mesh: every boundary face
    nf = int(wall.sum())
    r = int(level or lattice_level(nf))
    ref, ltri = _lattice_topology(r)
    npl = len(ref)
    ir = IntegrationRule([tuple(p) for p in ref], [1.0 / npl] * npl)
    mips = mesh.MapToAllElements(ir, BND)
    P = np.asarray(CF((X_, Y_, Z_))(mips), np.float64).reshape(nf_all, npl, 3)[wall]
    nrm = np.asarray(specialcf.normal(3)(mips), np.float64).reshape(nf_all, npl, 3)[wall]
    verts = np.array([[v.nr for v in f.vertices] for f in faces], np.int64)[wall]        # [nf, 3]
    # orientation: the lattice triangles' area vectors against the outward normal of the face
    t = P[:, ltri]                                                                        # [nf, T, 3, 3]
    area = np.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0]).sum(1)              # [nf, 3]
    flip = np.einsum("fc,fc->f", area, nrm.mean(1)) < 0
    tri_f = np.broadcast_to(ltri, (nf,) + ltri.shape).copy()
    tri_f[flip] = tri_f[flip][:, :, [0, 2, 1]]
    flat = P.reshape(-1, 3)
    size = float(np.ptp(flat, axis=0).max())
    ng, group = _weld(flat, 1e-9 * max(size, 1e-30))
    tris = (tri_f + (np.arange(nf) * npl)[:, None, None]).reshape(-1, 3)
    tris = group[tris]
    w_ref = np.column_stack([ref[:, 0], ref[:, 1], 1 - ref.sum(1)])                       # weights of vertices 0, 1, 2
    first = np.unique(group, return_index=True)[1]                                        # first copy of each point
    face_of, loc_of = first // npl, first % npl
    xs = np.zeros((ng, 3))
    xs[group] = flat                                                                      # equal to 1e-9 size
    rows = (np.flatnonzero(wall)[:, None] * npl + np.arange(npl)[None, :]).reshape(-1)
    cnt = np.bincount(group, minlength=ng).astype(np.float64)
    mean = csr_matrix((1.0 / cnt[group], (group, np.arange(len(group)))), shape=(ng, len(group)))   # copies → mean
    return {"x": xs, "tris": tris, "vid3": verts[face_of], "w3": w_ref[loc_of], "group": group, "rows": rows,
            "mean": mean, "mips": mips, "level": r, "n_faces": nf, "n_points": int(ng)}


def curved_display(mesh, level=None):
    """Picklable summary of the curved mesh for the geometry record (computed where the mesh is loaded):
    the wall lattice (points [m], triangles, vid3, w3), the curved volume ∫1 dV [m³], its bbox [m] and level."""
    from ngsolve import Integrate
    lat = surface_lattice(mesh, level)
    return {"points": lat["x"], "triangles": lat["tris"], "vid3": lat["vid3"], "w3": lat["w3"],
            "volume_m3": float(Integrate(1.0, mesh)), "bbox_m": [lat["x"].min(0).tolist(), lat["x"].max(0).tolist()],
            "level": lat["level"]}


def surface_vectors(mesh, lat, U, f_ghz, Mp, order=2):
    """E and H vectors [V/m, A/m] at every (face, lattice point) copy of the wall at U = 1 J, from the adjacent volume
    element (BoundaryFromVolumeCF: the one-sided wall value Epk / Bpk use): (E, H) each [K, Ncopy, 3] float32."""
    from ngsolve import BoundaryFromVolumeCF, curl
    U = np.asarray(U, np.float64).reshape(len(U), -1)
    a = energy_scale(Mp, U)
    fa = a * _faraday(f_ghz)
    E = np.empty((U.shape[1], len(lat["rows"]), 3), np.float32)
    H = np.empty_like(E)
    for k in range(U.shape[1]):
        g = _gridfunction(mesh, order, U[:, k])
        E[k] = a[k] * np.asarray(BoundaryFromVolumeCF(g)(lat["mips"]), np.float64).reshape(-1, 3)[lat["rows"]]
        H[k] = -fa[k] * np.asarray(BoundaryFromVolumeCF(curl(g))(lat["mips"]), np.float64).reshape(-1, 3)[lat["rows"]]
    return E, H


def weld_mean(lat, copies):
    """Mean over the copies of each lattice point: copies [..., Ncopy] → [..., Ns]."""
    flat = np.asarray(copies, np.float64).reshape(-1, copies.shape[-1])
    return (lat["mean"] @ flat.T).T.reshape(copies.shape[:-1] + (lat["n_points"],))


def surface_magnitudes(lat, vec_a, vec_b=None):
    """|vec_a| (or |vec_a − vec_b|) per lattice point: [K, Ns] float32 (copy magnitudes averaged at welded points)."""
    d = vec_a if vec_b is None else vec_a.astype(np.float64) - vec_b
    return weld_mean(lat, np.linalg.norm(d, axis=-1)).astype(np.float32)


# ───────────────────────── beam axis ─────────────────────────

def _axis_points(mesh, dm, axis_dir, axis_point, n_axis, deform=None):
    """(unit direction, positions t [P] along it, cell length dt, located points) of n_axis cell midpoints over the
    extent of the (curved, deformed) mesh along the axis through axis_point."""
    d = np.asarray(axis_dir, np.float64)
    d = d / np.linalg.norm(d)
    p0 = np.asarray(axis_point, np.float64)
    t = dm["x"] @ d
    t0, t1 = float(t.min()), float(t.max())
    dt = (t1 - t0) / n_axis
    tt = t0 + (np.arange(n_axis) + 0.5) * dt
    pts = (p0 - (p0 @ d) * d)[None, :] + tt[:, None] * d[None, :]
    return d, tt, dt, locate(mesh, pts, deform)


def axis_chord(mesh, dm, axis_dir, axis_point, n_axis=401, deform=None):
    """Length [m] of the beam axis inside the cavity."""
    _, _, dt, loc = _axis_points(mesh, dm, axis_dir, axis_point, n_axis, deform)
    return float(loc[1].sum() * dt)


def axis_profile(mesh, dm, U, Mp, axis_dir, axis_point, order=2, n_axis=401, deform=None):
    """E along the beam axis at U = 1 J: (zeta_mm [P] relative to the volume centroid, Ez [K, P] V/m (NaN outside
    the cavity), L_axis [m] = length inside)."""
    from src.data_gen import highorder as ho
    d, tt, dt, loc = _axis_points(mesh, dm, axis_dir, axis_point, n_axis, deform)
    inside = loc[1]
    U = np.asarray(U, np.float64).reshape(len(U), -1)
    a = energy_scale(Mp, U)
    ez = np.full((U.shape[1], n_axis), np.nan)
    for k in range(U.shape[1]):
        ez[k, inside] = a[k] * (ho.evaluate(_gridfunction(mesh, order, U[:, k]), loc) @ d)[inside]
    return (tt - float(dm["centroid"] @ d)) * 1e3, ez, float(inside.sum() * dt)
