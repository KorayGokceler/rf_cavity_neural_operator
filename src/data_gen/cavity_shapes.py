"""Geometry families for the 3D generator: realistic RF cavities and free-form solids.

Axisymmetric cavities are built as a meridian profile in the (z, r) half-plane
(z = beam axis), revolved 2π about z with gmsh OCC.  Beam pipes are closed by flat
PEC end caps, as in a closed eigenmode solve.  Lengths in metres.

- elliptical:    TESLA-type elliptical cells (1–5) with beam pipes; real-design anchors, low-β
                 cells, different end half-cells, optional coupler ports on the pipes.  Half-cell =
                 iris ellipse (a, b) at the iris plane + equator ellipse (A, B) at the
                 equator, joined by their common tangent (the standard construction).
                 TESLA mid-cell: Req 103.3, Riris 35, A = B = 42, a 12, b 19, L 57.7 mm.
- reentrant:     nose-cone (klystron / IOT type) cavity with beam pipes, filleted corners.
- pillbox_pipes: pillbox with beam pipes, rounded iris edge and outer corners.
- ridged_box:    rectangular cavity with 1–2 full-length ridges (+ optional beam pipes), filleted.
- composite:     CSG tree: cylinder / ellipsoid body with attached rounded primitives and dents.
- freeform:      star-shaped superellipsoid × smooth random modulation × lobes, then a
                 bend / twist / taper warp (each an injective map, so the surface never
                 self-intersects and the solid stays a topological ball); returned as a
                 closed triangulated surface for gmsh's discrete-surface remeshing.

Handle families (HANDLE_FAMILIES; first Betti number b1 > 0 — need the E formulation, the H
formulation rejects them):

- hwr:           half-wave coaxial resonator (TEM λ/2, f ≈ c/2L): inner conductor touching both end
                 plates (straight or tapered, filleted junctions); 50 %: a transverse beam port through
                 the outer wall and the inner conductor at mid-length.  b1 = 1 (2 with the port).
- spoke:         cylindrical tank (axis z) with 1–2 spokes (circular / elliptic section) across the
                 diameter along x, touching the tank wall at both ends; 60 %: a beam bore along z
                 through the tank and every spoke (+ PEC-capped pipes), else pipe stubs at the end
                 plates only.  b1 = 1 per spoke (2 per spoke with the bore).
- dtl:           Alvarez-like tank with 1–3 drift tubes (annular rings with a beam bore and rounded
                 noses) on stems (cylinders along y) to the tank wall, beam pipes at the tank ends.
                 b1 = 1 per drift tube.

Out-of-distribution families (OOD_FAMILIES, never in the default training set):

- box:             rectangular cavity: flat faces, sharp edges (analytic spectrum).
- coax_qw:         quarter-wave coaxial resonator: inner post from one end cap with a
                   capacitive gap to the other (deep narrow annulus, blind hole → ball).
- pillbox_port:    pillbox with beam pipes and 1–2 radial side ports (breaks the axial
                   symmetry every training cavity has).
- elliptical_long: 6–9 elliptical cells (training: 1–5), same cell-shape ranges.
- junction:        L / T / cross of box arms (3D cross optional): non-star-shaped with
                   sharp re-entrant edges (training free-forms are smooth star shapes).
"""
import numpy as np
from scipy.optimize import least_squares

# ─────────────────────────── profile helpers ───────────────────

def filleted(corners, radii):
    """Closed polygon (z, r) corners with fillet radii (0 = sharp) → segment list
    [('line', p, q) | ('arc', p, centre, q)]; radii shrink to fit the edges."""
    P = np.asarray(corners, float)
    n = len(P)
    ends = []                                        # per corner: (entry point, arc or None, exit point)
    for i in range(n):
        p0, p, p1 = P[i - 1], P[i], P[(i + 1) % n]
        rho = float(radii[i])
        d1, d2 = p - p0, p1 - p
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        d1, d2 = d1 / l1, d2 / l2
        cos_t = np.clip(-d1 @ d2, -1.0, 1.0)
        theta = np.arccos(cos_t)                     # angle between the two edges at the corner
        if rho <= 0 or theta > np.pi - 1e-3:
            ends.append((p, None, p))
            continue
        t = min(rho / np.tan(theta / 2), 0.45 * l1, 0.45 * l2)
        rho = t * np.tan(theta / 2)
        bis = (d2 - d1) / np.linalg.norm(d2 - d1)
        ends.append((p - d1 * t, p + bis * rho / np.sin(theta / 2), p + d2 * t))
    segs = []
    for i in range(n):
        a_in, c, a_out = ends[i]
        if c is not None:
            segs.append(('arc', a_in, c, a_out))
        segs.append(('line', a_out, ends[(i + 1) % n][0]))
    return segs


def revolve_segments(occ, segs):
    """Build the (x = r, y = 0, z) profile from segments, revolve 2π about z."""
    pts = {}

    def pt(p):
        key = (round(float(p[0]), 12), round(float(p[1]), 12))
        if key not in pts:
            pts[key] = occ.addPoint(max(float(p[1]), 0.0), 0.0, float(p[0]))
        return pts[key]

    curves = []
    for s in segs:
        if s[0] == 'line':
            if np.linalg.norm(np.asarray(s[1]) - np.asarray(s[2])) > 1e-12:
                curves.append(occ.addLine(pt(s[1]), pt(s[2])))
        elif s[0] == 'arc':
            curves.append(occ.addCircleArc(pt(s[1]), pt(s[2]), pt(s[3])))
        else:                                            # ('spline', [points])
            curves.append(occ.addSpline([pt(p) for p in s[1]]))
    surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
    return next(t for d, t in out if d == 3)


# ─────────────────────────── elliptical cells ──────────────────

def half_cell_wall(Req, Riris, L, A, B, a, b, n=24):
    """Wall points (z, r) of a half-cell from the iris plane (z = 0, r = Riris) to the
    equator (z = L, r = Req): iris ellipse → common tangent → equator ellipse.
    None if no valid (monotone, non-overlapping) tangent exists."""
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])

    def P1(t):
        return ci + np.stack([a * np.sin(t), -b * np.cos(t)], -1)

    def P2(s):
        return ce + np.stack([-A * np.cos(s), B * np.sin(s)], -1)

    def T1(t):
        return np.array([a * np.cos(t), b * np.sin(t)])

    def T2(s):
        return np.array([A * np.sin(s), B * np.cos(s)])

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def res(x):
        t, s = x
        d = P2(s) - P1(t)
        return [cross(T1(t), T2(s)) / (a * A), cross(T1(t), d) / (a * L)]

    sol = least_squares(res, [np.pi / 2, 0.0], bounds=([1e-3, -np.pi / 2 + 1e-3], [np.pi - 1e-3, np.pi / 2]))
    t, s = sol.x
    if np.max(np.abs(sol.fun)) > 1e-9 or (P2(s) - P1(t)) @ T1(t) <= 0:
        return None
    wall = np.vstack([P1(np.linspace(0, t, n)), P2(np.linspace(s, np.pi / 2, n))])
    if np.any(np.diff(wall[:, 0]) < -1e-12) or np.any(np.diff(wall[:, 1]) < -1e-12):
        return None                                       # re-entrant wall: rejected
    return wall


HALF_KEYS = ("Req", "Riris", "L", "A", "B", "a", "b")

# Mid-cell half-cell geometries of real designs [m], used only as SAMPLING CENTRES (perturbed
# ±10 %, rescaled in size); labels always come from the FE solve, so small inaccuracies in these
# literature values do not affect the data.  TESLA: Aune et al., PRST-AB 3, 092001 (2000);
# ILC low-loss: approximate values of the Cornell/KEK LL cell (verify before citing).
ANCHORS = {
    "tesla": dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019),
    "ilc_ll": dict(Req=0.0986, Riris=0.030, L=0.0577, A=0.0501, B=0.0342, a=0.0076, b=0.0100),
}
N_CELLS, P_CELLS = [1, 2, 3, 4, 5], [0.35, 0.25, 0.2, 0.1, 0.1]     # training; OOD elliptical_long: 6–9


def _draw_half(rng, Req, Riris, beta):
    """Random half-cell (ratios around TESLA / LEP / CEBAF-like cells); L ∝ β (β < 1: shorter cells)."""
    h = {"Req": Req, "Riris": Riris, "L": Req * rng.uniform(0.45, 0.65) * beta}
    h["A"] = h["L"] * rng.uniform(0.6, 0.85)
    h["B"] = h["A"] * rng.uniform(0.85, 1.25)
    h["a"] = h["L"] * rng.uniform(0.15, 0.30)
    h["b"] = h["a"] * rng.uniform(1.0, 1.9)
    return h


def _valid_half(h):
    if h["Riris"] + h["b"] >= h["Req"] - h["B"]:
        return None
    return half_cell_wall(**{k: h[k] for k in HALF_KEYS})


def draw_elliptical(rng, n_cells=None):
    """Mid half-cell (random ratios, a real-design anchor, or a low-β cell), optional different
    end half-cells (larger end iris = beam-pipe radius, perturbed shape) and 1–5 cells.
    Returns (flat params, {'mid', 'e1', 'e2'} half-cell dicts, {name: wall})."""
    for _ in range(200):
        u = rng.uniform()
        if u < 0.3:                                        # real-design anchor, perturbed and rescaled
            name = str(rng.choice(sorted(ANCHORS)))
            s = rng.uniform(0.04, 0.11) / ANCHORS[name]["Req"]
            mid = {k: v * s * rng.uniform(0.9, 1.1) for k, v in ANCHORS[name].items()}
            beta, anchor = 1.0, float(1 + sorted(ANCHORS).index(name))
        else:
            beta = 1.0 if u < 0.75 else rng.uniform(0.5, 0.95)          # 25 %: low-β cells
            Req = rng.uniform(0.04, 0.11)
            mid, anchor = _draw_half(rng, Req, Req * rng.uniform(0.25, 0.42), beta), 0.0
        w_mid = _valid_half(mid)
        if w_mid is None:
            continue
        halves, walls = {"mid": mid}, {"mid": w_mid}
        for e in ("e1", "e2"):
            end = dict(mid)
            if rng.uniform() < 0.6:                        # end half-cell differs (as in real structures)
                end.update({k: mid[k] * rng.uniform(0.85, 1.15) for k in ("L", "A", "B", "a", "b")})
                end["Riris"] = mid["Riris"] * rng.uniform(1.0, 1.35)
            w = _valid_half(end)
            halves[e], walls[e] = (end, w) if w is not None else (dict(mid), w_mid)
        n = int(n_cells if n_cells is not None else rng.choice(N_CELLS, p=P_CELLS))
        p = {k: mid[k] for k in HALF_KEYS}
        for e in ("e1", "e2"):
            p.update({f"{e}_{k}": halves[e][k] for k in HALF_KEYS if k != "Req"})
        p.update({"n_cells": float(n), "beta": beta, "anchor": anchor,
                  "Lpipe1": halves["e1"]["Riris"] * rng.uniform(1.2, 2.5),
                  "Lpipe2": halves["e2"]["Riris"] * rng.uniform(1.2, 2.5)})
        return p, halves, walls
    raise RuntimeError("no valid elliptical cell parameters")


def elliptical_chain(halves, walls, n):
    """Wall points (z, r) of n cells: left half (iris → equator) + mirrored right half; the
    first left half is e1, the last right half e2, all others mid."""
    pts, z = [], 0.0
    for k in range(n):
        left, right = ("e1" if k == 0 else "mid"), ("e2" if k == n - 1 else "mid")
        wl, wr, Lr = walls[left], walls[right], halves[right]["L"]
        seg_l = wl + [z, 0]
        z_eq = z + halves[left]["L"]
        seg_r = np.c_[z_eq + Lr - wr[::-1, 0], wr[::-1, 1]]
        pts += [seg_l if k == 0 else seg_l[1:], seg_r[1:]]
        z = z_eq + Lr
    return np.vstack(pts), z


def elliptical_segments(p, halves, walls):
    """Profile: pipe (radius = e1 iris) – chain – pipe (radius = e2 iris), PEC-capped."""
    chain, z1 = elliptical_chain(halves, walls, int(p["n_cells"]))
    r1, r2 = halves["e1"]["Riris"], halves["e2"]["Riris"]
    L1, L2 = p.get("Lpipe1", p.get("Lpipe")), p.get("Lpipe2", p.get("Lpipe"))
    return [('line', (-L1, 0.0), (-L1, r1)), ('line', (-L1, r1), (0.0, r1)),
            ('spline', [tuple(q) for q in chain]),
            ('line', (z1, r2), (z1 + L2, r2)), ('line', (z1 + L2, r2), (z1 + L2, 0.0)),
            ('line', (z1 + L2, 0.0), (-L1, 0.0))], z1


def _add_pipe_ports(occ, vol, rng, pipes, params):
    """1–3 radial coupler ports (FPC / HOM-coupler stubs) on the beam pipes, fused; pipes =
    [(z_centre, radius, length)]."""
    tags = [vol]
    n = int(rng.integers(1, 4))
    for i in range(n):
        zc, rp, lp = pipes[int(rng.integers(0, len(pipes)))]
        rs = min(rp * rng.uniform(0.25, 0.6), 0.4 * lp)
        ls = rp * rng.uniform(0.8, 2.0)
        phi = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        x0 = 0.5 * rp * u + [0, 0, zc]
        tags.append(occ.addCylinder(*x0, *((0.5 * rp + ls) * u), rs))
        params.update({f"port{i}_r": rs, f"port{i}_len": ls, f"port{i}_z": zc, f"port{i}_phi": phi})
    params["n_ports"] = float(n)
    return _fuse_all(occ, tags), min(params[f"port{i}_r"] for i in range(n))


def build_elliptical(occ, rng, params=None, n_cells=None, shape_type="elliptical", port_prob=0.35):
    """params: mid half-cell keys (+ n_cells, Lpipe) → fixed geometry (tests); else random."""
    if params is not None:
        mid = {k: params[k] for k in HALF_KEYS}
        w = _valid_half(mid)
        if w is None:
            raise RuntimeError("invalid elliptical parameters")
        p, halves, walls = dict(params), {"mid": mid, "e1": mid, "e2": mid}, {"mid": w, "e1": w, "e2": w}
    else:
        p, halves, walls = draw_elliptical(rng, n_cells)
    segs, z1 = elliptical_segments(p, halves, walls)
    vol = revolve_segments(occ, segs)
    h_cap = min(halves["e1"]["Riris"], halves["mid"]["Riris"], halves["e2"]["Riris"])
    if rng is not None and rng.uniform() < port_prob:
        L1, L2 = p.get("Lpipe1", p.get("Lpipe")), p.get("Lpipe2", p.get("Lpipe"))
        _, rs_min = _add_pipe_ports(occ, vol, rng, [(-L1 / 2, halves["e1"]["Riris"], L1),
                                                   (z1 + L2 / 2, halves["e2"]["Riris"], L2)], p)
        h_cap = min(h_cap, rs_min)
    else:
        p["n_ports"] = 0.0
    return shape_type, dict(p, _h_cap=0.7 * h_cap, _vol_div=p["n_cells"])   # h from the per-cell volume


# ─────────────────────────── re-entrant / pillbox with pipes ───

def build_reentrant(occ, rng):
    for _ in range(100):
        R = rng.uniform(0.03, 0.07)
        Lc = R * rng.uniform(0.4, 1.0)
        rp = R * rng.uniform(0.10, 0.25)
        g = Lc * rng.uniform(0.15, 0.6)                 # nose-to-nose gap
        t_tip = R * rng.uniform(0.08, 0.2)              # nose tip wall thickness
        alpha = np.radians(rng.uniform(-25, 35))        # nose outer cone half-angle (< 0: mushroom nose)
        zt = (Lc - g) / 2
        rt = rp + t_tip
        rn = rt + zt * np.tan(alpha)
        if max(rn, rt) < 0.75 * R and rn > rp + 0.04 * R and zt > 0.05 * R:
            break
    else:
        raise RuntimeError("no valid re-entrant parameters")
    Lp = rp * rng.uniform(1.5, 4.0)
    rho_tip = t_tip * rng.uniform(0.2, 0.5)
    rho_root = min(R - rn, rn - rp) * rng.uniform(0.05, 0.3)
    rho_out = min(R - rn, Lc / 2) * rng.uniform(0.05, 0.6)   # large → toroidal outer wall
    C = [(-Lp, 0), (-Lp, rp), (zt, rp), (zt, rt), (0, rn), (0, R), (Lc, R), (Lc, rn),
         (Lc - zt, rt), (Lc - zt, rp), (Lc + Lp, rp), (Lc + Lp, 0)]
    rad = [0, 0, rho_tip, rho_tip, rho_root, rho_out, rho_out, rho_root, rho_tip, rho_tip, 0, 0]
    revolve_segments(occ, filleted(C, rad))
    params = {"R": R, "Lc": Lc, "rp": rp, "gap": g, "t_tip": t_tip, "alpha_deg": float(np.degrees(alpha)),
              "rn": rn, "Lpipe": Lp, "rho_tip": rho_tip, "rho_root": rho_root, "rho_out": rho_out}
    return "reentrant", dict(params, _h_cap=0.8 * min(rp, g, t_tip))


def build_pillbox_pipes(occ, rng):
    R = rng.uniform(0.03, 0.06)
    Lc = R * rng.uniform(0.4, 1.6)
    rp = R * rng.uniform(0.12, 0.35)
    Lp = rp * rng.uniform(1.5, 4.0)
    rho_iris = rp * rng.uniform(0.05, 0.4)
    rho_out = min(R - rp, Lc / 2) * rng.uniform(0.0, 0.5)
    C = [(-Lp, 0), (-Lp, rp), (0, rp), (0, R), (Lc, R), (Lc, rp), (Lc + Lp, rp), (Lc + Lp, 0)]
    revolve_segments(occ, filleted(C, [0, 0, rho_iris, rho_out, rho_out, rho_iris, 0, 0]))
    return "pillbox_pipes", {"R": R, "Lc": Lc, "rp": rp, "Lpipe": Lp, "rho_iris": rho_iris,
                             "rho_out": rho_out, "_h_cap": 0.8 * rp}


# ─────────────────────────── ridged rectangular cavity ─────────

def build_ridged_box(occ, rng):
    """Rectangular cavity with 1–2 full-length ridges (ridged-waveguide resonator), optional
    round beam pipes through the gap; all edges filleted (training shapes keep rounded edges —
    the sharp-edged plain box stays an OOD family)."""
    a, b, d = rng.uniform(0.04, 0.10), rng.uniform(0.03, 0.08), rng.uniform(0.02, 0.06)
    box = occ.addBox(0, 0, 0, a, b, d)
    w = b * rng.uniform(0.15, 0.45)
    double = rng.uniform() < 0.5
    hr = d * rng.uniform(0.12, 0.3 if double else 0.4)
    ridges = [occ.addBox(-1e-3, (b - w) / 2, d - hr, a + 2e-3, w, hr + 1e-3)]
    if double:
        ridges.append(occ.addBox(-1e-3, (b - w) / 2, -1e-3, a + 2e-3, w, hr + 1e-3))
    out, _ = occ.cut([(3, box)], [(3, r) for r in ridges])
    vol = [t for dd, t in out if dd == 3]
    if len(vol) != 1:
        raise RuntimeError(f"ridge cut produced {len(vol)} volumes")
    vol = vol[0]
    gap = d - hr * (2 if double else 1)
    params = {"a": a, "b": b, "d": d, "ridge_w": w, "ridge_h": hr, "double": float(double), "gap": gap}
    occ.synchronize()
    rho = min(hr, w, gap) * rng.uniform(0.08, 0.25)
    curves = [c for _, c in occ.getEntities(1)]
    out = occ.fillet([vol], curves, [rho])
    vol = next(t for dd, t in out if dd == 3)
    params["rho"] = rho
    if rng.uniform() < 0.4:                                     # beam pipes along x through the gap
        rp = min(gap, w) * rng.uniform(0.2, 0.4)
        zc = (hr if double else 0.0) + gap / 2
        Lp = rp * rng.uniform(1.5, 3.0)
        pipes = [occ.addCylinder(-Lp, b / 2, zc, Lp + rho, 0, 0, rp),
                 occ.addCylinder(a - rho, b / 2, zc, Lp + rho, 0, 0, rp)]
        vol = _fuse_all(occ, [vol] + pipes)
        params.update({"rp": rp, "Lpipe": Lp})
    return "ridged_box", dict(params, _h_cap=0.8 * min(gap, w, params.get("rp", gap)))


# ─────────────────────────── CSG composite (tree of primitives) ─

class _Prim:
    """Cylinder (local axis z, radius r, half-length hl) or ellipsoid (semi-axes s), placed by a
    centre and a rotation R (columns = local axes in world coordinates)."""

    def __init__(self, kind, centre, R, dims):
        self.kind, self.c, self.R, self.dims = kind, np.asarray(centre, float), np.asarray(R, float), dims

    def support(self, d):
        """Distance from the centre to the surface along the unit world direction d."""
        l = self.R.T @ d
        if self.kind == "ell":
            return 1.0 / np.sqrt(((l / self.dims) ** 2).sum())
        r, hl = self.dims
        rho = np.hypot(l[0], l[1])
        return min(r / rho if rho > 1e-12 else np.inf, hl / abs(l[2]) if abs(l[2]) > 1e-12 else np.inf)

    def width(self):
        return 2 * (min(self.dims) if self.kind == "ell" else min(self.dims))

    def occ(self, occ):
        if self.kind == "cyl":
            r, hl = self.dims
            ax = self.R[:, 2]
            return occ.addCylinder(*(self.c - hl * ax), *(2 * hl * ax), r)
        from scipy.spatial.transform import Rotation
        t = occ.addSphere(0, 0, 0, 1.0)
        occ.dilate([(3, t)], 0, 0, 0, *self.dims)
        rv = Rotation.from_matrix(self.R).as_rotvec()
        ang = np.linalg.norm(rv)
        if ang > 1e-12:
            occ.rotate([(3, t)], 0, 0, 0, *(rv / ang), ang)
        occ.translate([(3, t)], *self.c)
        return t


def _frame(axis, rng):
    """Random rotation whose third column is the unit vector axis."""
    axis = axis / np.linalg.norm(axis)
    t = rng.standard_normal(3)
    t -= t @ axis * axis
    t /= np.linalg.norm(t)
    return np.c_[t, np.cross(axis, t), axis]


def build_composite(occ, rng):
    """A body (cylinder or ellipsoid) with 1–4 attached rounded primitives (cylinders /
    ellipsoids, depth ≤ 2: children may carry grandchildren) fused, plus 0–2 ellipsoidal dents
    cut into the surface.  A tree of attachments keeps the solid a ball; overlaps that close a
    loop or cut through are rejected by the topology certificate."""
    R0 = rng.uniform(0.025, 0.05)
    if rng.uniform() < 0.5:
        body = _Prim("cyl", np.zeros(3), np.eye(3), (R0, R0 * rng.uniform(0.4, 1.2)))
    else:
        body = _Prim("ell", np.zeros(3), _frame(rng.standard_normal(3), rng), R0 * rng.uniform(0.5, 1.0, 3))
    prims, fuse, cut = [body], [], []
    n_child = int(rng.integers(1, 5))
    bw = body.width()
    for i in range(n_child):
        parent = prims[int(rng.integers(0, min(len(prims), 2)))]   # body or the first child (depth ≤ 2)
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        q = parent.c + parent.support(d) * d
        wcap = 0.9 * parent.width()                                # never wider than the parent
        if rng.uniform() < 0.6:                                    # stub / arm: radius ∝ body, not parent
            r = min(bw * rng.uniform(0.15, 0.35), wcap / 2)
            hl = R0 * rng.uniform(0.25, 0.7)
            ch = _Prim("cyl", q + d * hl * rng.uniform(0.3, 0.7), _frame(d, rng), (r, hl))
        else:                                                      # lobe
            sa = np.r_[np.minimum(bw * rng.uniform(0.15, 0.4, 2), wcap / 2), R0 * rng.uniform(0.3, 0.7)]
            ch = _Prim("ell", q + d * sa[2] * rng.uniform(0.2, 0.6), _frame(d, rng), sa)
        prims.append(ch)
        fuse.append(ch)
    for _ in range(int(rng.integers(0, 3))):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        q = body.c + body.support(d) * d
        cut.append(_Prim("ell", q, _frame(d, rng), body.width() * rng.uniform(0.1, 0.3, 3)))
    vol = _fuse_all(occ, [p.occ(occ) for p in [body] + fuse])
    for c in cut:
        out, _ = occ.cut([(3, vol)], [(3, c.occ(occ))])
        vols = [t for dd, t in out if dd == 3]
        if len(vols) != 1:
            raise RuntimeError(f"dent produced {len(vols)} volumes")
        vol = vols[0]
    params = {"R0": R0, "body_cyl": float(body.kind == "cyl"), "n_attached": float(len(fuse)),
              "n_dents": float(len(cut))}
    for i, pr in enumerate(fuse):
        params.update({f"c{i}_cyl": float(pr.kind == "cyl"), f"c{i}_w": pr.width(),
                       f"c{i}_x": pr.c[0], f"c{i}_y": pr.c[1], f"c{i}_z": pr.c[2]})
    return "composite", dict(params, _h_cap=0.8 * min(p.width() / 2 for p in prims))


# ─────────────────────────── free-form solids ──────────────────

_ICO = {}


def icosphere(level):
    """Unit icosphere (vertices [N,3], outward triangles [F,3]), cached."""
    if level in _ICO:
        return _ICO[level]
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                  [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(level):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        ue, inv = np.unique(e, axis=0, return_inverse=True)
        m = V[ue].mean(1)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        idx = len(V) + inv.reshape(3, -1).T
        V = np.vstack([V, m])
        a, b, c = F.T
        ab, bc, ca = idx.T
        F = np.vstack([np.c_[a, ab, ca], np.c_[b, bc, ab], np.c_[c, ca, bc], np.c_[ab, bc, ca]])
    _ICO[level] = (V, F)
    return V, F


def _superellipsoid_radius(u, ax, e1, e2):
    """Radius of the superellipsoid |x/a|^(2/e2)+|y/b|^(2/e2))^(e2/e1)+|z/c|^(2/e1) = 1 along unit u."""
    x, y, z = (np.abs(u) / ax + 1e-12).T
    return ((x ** (2 / e2) + y ** (2 / e2)) ** (e2 / e1) + z ** (2 / e1)) ** (-e1 / 2)


def freeform_surface(rng, level=5):
    """Closed, outward triangulated surface (P [N,3] metres, F [M,3]) and parameters."""
    V, F = icosphere(level)
    R0 = rng.uniform(0.03, 0.06)
    ax = R0 * np.r_[1.0, rng.uniform(0.45, 1.0, 2)][rng.permutation(3)]
    e1, e2 = np.exp(rng.uniform(np.log(0.35), np.log(1.6), 2))          # box-like … ellipsoid … pinched
    r = _superellipsoid_radius(V, ax, e1, e2)
    n_w = int(rng.integers(6, 20))                                        # smooth random modulation
    k = rng.uniform(1.0, 6.0, n_w)
    dirs = rng.standard_normal((n_w, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    amp = rng.uniform(0.03, 0.18)
    w = k ** -1.5
    coef = rng.standard_normal(n_w) * w * amp / np.sqrt((w ** 2).sum())
    noise = (coef * np.cos(V @ (dirs * k[:, None]).T + rng.uniform(0, 2 * np.pi, n_w))).sum(1)
    n_l = int(rng.integers(0, 5))                                         # protrusions (+) / dents (−)
    ld = rng.standard_normal((n_l, 3))
    ld /= np.linalg.norm(ld, axis=1, keepdims=True) + 1e-12
    lw, lh = rng.uniform(0.25, 0.7, n_l), rng.uniform(-0.45, 0.8, n_l)
    lobes = (lh * np.exp(-(1 - V @ ld.T) / lw ** 2)).sum(1) if n_l else 0.0
    P = V * (r * np.exp(noise + lobes))[:, None]
    P -= P.mean(0)
    zr = np.ptp(P[:, 2]) / 2 + 1e-12                                      # warp along z (injective maps)
    zn = P[:, 2] / zr
    bend = rng.uniform(-0.5, 0.5, 2) * zr
    P[:, 0] += bend[0] * zn ** 2
    P[:, 1] += bend[1] * zn ** 2
    twist = rng.uniform(-1.0, 1.0)
    ct, st = np.cos(twist * zn), np.sin(twist * zn)
    P[:, 0], P[:, 1] = ct * P[:, 0] - st * P[:, 1], st * P[:, 0] + ct * P[:, 1]
    taper = rng.uniform(-0.35, 0.35)
    P[:, :2] *= (1 + taper * zn)[:, None]
    params = {"R0": R0, "ax0": ax[0], "ax1": ax[1], "ax2": ax[2], "e1": e1, "e2": e2, "n_waves": float(n_w),
              "mod_amp": amp, "n_lobes": float(n_l), "bend_x": bend[0], "bend_y": bend[1], "twist": twist,
              "taper": taper}
    return P, F, params


def surface_volume(P, F):
    """Enclosed volume of a closed outward triangulation (divergence theorem)."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


# ─────────────────────────── out-of-distribution families ─────

OOD_FAMILIES = ("box", "coax_qw", "pillbox_port", "elliptical_long", "junction")
HANDLE_FAMILIES = ("hwr", "spoke", "dtl")


def _fuse_all(occ, tags):
    vol = tags[0]
    for t in tags[1:]:
        out, _ = occ.fuse([(3, vol)], [(3, t)])
        vols = [tg for d, tg in out if d == 3]
        if len(vols) != 1:
            raise RuntimeError(f"fuse produced {len(vols)} volumes")
        vol = vols[0]
    return vol


def build_box(occ, rng):
    a, b, d = rng.uniform(0.03, 0.10, 3)
    occ.addBox(0, 0, 0, a, b, d)
    return "box", {"a": a, "b": b, "d": d}


def build_coax_qw(occ, rng):
    Ro = rng.uniform(0.02, 0.05)
    L = Ro * rng.uniform(1.5, 4.0)
    ri = Ro * rng.uniform(0.2, 0.45)
    gap = L * rng.uniform(0.08, 0.3)
    outer = occ.addCylinder(0, 0, 0, 0, 0, L, Ro)
    post = occ.addCylinder(0, 0, 0, 0, 0, L - gap, ri)
    occ.cut([(3, outer)], [(3, post)])
    return "coax_qw", {"Ro": Ro, "L": L, "ri": ri, "gap": gap, "_h_cap": 0.8 * min(gap, Ro - ri)}


def build_pillbox_port(occ, rng):
    R = rng.uniform(0.03, 0.06)
    Lc = R * rng.uniform(0.6, 1.6)
    rp = R * rng.uniform(0.12, 0.3)
    Lp = rp * rng.uniform(1.5, 3.0)
    tags = [occ.addCylinder(0, 0, 0, 0, 0, Lc, R), occ.addCylinder(0, 0, -Lp, 0, 0, Lp + 1e-4, rp),
            occ.addCylinder(0, 0, Lc - 1e-4, 0, 0, Lp + 1e-4, rp)]
    n_ports = int(rng.integers(1, 3))
    params = {"R": R, "Lc": Lc, "rp": rp, "Lpipe": Lp, "n_ports": float(n_ports)}
    phi0 = rng.uniform(0, 2 * np.pi)
    for i in range(n_ports):
        rs = min(R, Lc) * rng.uniform(0.12, 0.3)
        zc = Lc / 2 + rng.uniform(-1, 1) * (Lc / 2 - rs) * 0.8
        phi = phi0 + i * rng.uniform(0.6, 1.0) * np.pi
        Ls = R * rng.uniform(0.4, 1.2)
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        x0 = 0.5 * R * u + [0, 0, zc]
        tags.append(occ.addCylinder(*x0, *((0.5 * R + Ls) * u), rs))
        params.update({f"p{i}_r": rs, f"p{i}_z": zc, f"p{i}_phi": phi, f"p{i}_len": Ls})
    _fuse_all(occ, tags)
    return "pillbox_port", dict(params, _h_cap=0.8 * min([rp] + [params[f"p{i}_r"] for i in range(n_ports)]))


def build_elliptical_long(occ, rng):
    """6–9 cells (training: 1–5), no ports; otherwise the training elliptical distribution."""
    return build_elliptical(occ, rng, n_cells=int(rng.integers(6, 10)), shape_type="elliptical_long",
                            port_prob=0.0)


def build_junction(occ, rng):
    w = rng.uniform(0.02, 0.04)                       # arm width
    d = w * rng.uniform(0.5, 1.2)                     # thickness (z)
    kind = str(rng.choice(["L", "T", "X"]))
    dirs = {"L": [0, 1], "T": [0, 1, 2], "X": [0, 1, 2, 3]}[kind]
    tags = [occ.addBox(-w / 2, -w / 2, 0, w, w, d)]   # hub
    params = {"w": w, "d": d, "kind": float("LTX".index(kind))}
    for i, q in enumerate(dirs):
        la = w * rng.uniform(1.0, 2.5)
        wa = w * rng.uniform(0.6, 1.0)
        c, s_ = np.cos(q * np.pi / 2), np.sin(q * np.pi / 2)
        x0 = [w / 2 * c - wa / 2 * abs(s_) - (la if c < -0.5 else 0), w / 2 * s_ - wa / 2 * abs(c) - (la if s_ < -0.5 else 0), 0]
        size = [la if abs(c) > 0.5 else wa, la if abs(s_) > 0.5 else wa, d]
        if c > 0.5 or s_ > 0.5:
            x0 = [w / 2 - 1e-4 if c > 0.5 else -wa / 2, w / 2 - 1e-4 if s_ > 0.5 else -wa / 2, 0]
        tags.append(occ.addBox(*x0, *size))
        params[f"arm{i}_len"], params[f"arm{i}_w"] = la, wa
    if rng.uniform() < 0.3:                           # vertical arm → 3D cross
        lz = w * rng.uniform(1.0, 2.0)
        tags.append(occ.addBox(-w / 2, -w / 2, d - 1e-4, w, w, lz))
        params["arm_z_len"] = lz
    _fuse_all(occ, tags)
    return "junction", dict(params, _h_cap=0.8 * min(d, w))


# ─────────────────────────── handle families (b1 > 0, E formulation) ─

def _check_volume(occ, vol, expected, tol=0.1):
    """Guard against a silently failed OCC boolean (e.g. a cut returning only a tool fragment):
    the solid's volume must match the analytic estimate (fillets / curved junctions neglected)."""
    occ.synchronize()
    v = occ.getMass(3, vol)
    if not abs(v / expected - 1) < tol:
        raise RuntimeError(f"boolean volume check failed: {v:.3e} vs expected {expected:.3e}")
    return v


def _cut_all(occ, vol, tools):
    out, _ = occ.cut([(3, vol)], [(3, t) for t in tools])
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        raise RuntimeError(f"cut produced {len(vols)} volumes")
    return vols[0]


def build_hwr(occ, rng):
    """Half-wave coaxial resonator: annular region Ri(z) < r < Ro, 0 < z < L, revolved about z; the
    inner conductor touches both end plates (TEM λ/2 mode, f ≈ c/2L; b1 = 1).  Inner radius
    straight or tapered (end radius ri, mid radius rm), fillets at the inner-conductor / end-plate
    junctions and the outer corners.  50 %: transverse beam port (along x, at z = L/2) through the
    outer wall AND the inner conductor, with PEC-capped pipes (b1 = 2)."""
    Ro = rng.uniform(0.03, 0.06)
    L = rng.uniform(0.08, 0.20)                                  # TEM f ≈ 0.75–1.9 GHz
    ri = Ro * rng.uniform(0.2, 0.5)
    taper = rng.uniform() < 0.5
    rm = float(np.clip(ri * rng.uniform(0.6, 1.5), 0.15 * Ro, 0.6 * Ro)) if taper else ri
    rho_in = min(Ro - max(ri, rm), L / 4) * rng.uniform(0.0, 0.4)
    rho_out = min(Ro - max(ri, rm), L / 4) * rng.uniform(0.0, 0.5)
    C = [(0, ri), (0, Ro), (L, Ro), (L, ri)] + ([(L / 2, rm)] if taper else [])
    rad = [rho_in, rho_out, rho_out, rho_in] + ([0.0] if taper else [])
    vol = revolve_segments(occ, filleted(C, rad))
    params = {"Ro": Ro, "L": L, "ri": ri, "rm": rm, "taper": float(taper), "rho_in": rho_in, "rho_out": rho_out}
    v_exp = np.pi * Ro ** 2 * L - np.pi * L / 3 * (ri ** 2 + ri * rm + rm ** 2)     # two frustums
    h_cap = Ro - max(ri, rm)
    if rng.uniform() < 0.5:
        rb = min(rm, ri) * rng.uniform(0.45, 0.7)
        Lp = rb * rng.uniform(1.5, 3.0)
        bore = occ.addCylinder(-(Ro + Lp), 0, L / 2, 2 * (Ro + Lp), 0, 0, rb)
        vol = _fuse_all(occ, [vol, bore])
        params.update({"rb": rb, "Lpipe": Lp})
        v_exp += np.pi * rb ** 2 * (2 * Lp + 2 * rm)
        h_cap = min(h_cap, rb)
    _check_volume(occ, vol, v_exp)
    params["beam_port"] = float("rb" in params)
    return "hwr", dict(params, _h_cap=0.8 * h_cap)


def build_spoke(occ, rng):
    """Spoke cavity: tank (radius Rt, length Lt, axis z) with n = 1–2 spokes along x across the full
    diameter (circular or elliptic section, semi-axes ay, az; the second spoke parallel or turned
    90° about z), each touching the tank wall at both ends (b1 = 1 per spoke).  60 %: beam bore of
    radius rp along z through the tank and every spoke, continued as PEC-capped pipes (the bore
    through a spoke adds a second loop: b1 = 2 per spoke); else pipe stubs at the end plates only."""
    Rt = rng.uniform(0.05, 0.10)                                  # spoke mode f ≈ c/(4Rt)·O(1)
    n = 1 if rng.uniform() < 0.6 else 2
    Lt = Rt * (rng.uniform(0.9, 1.4) if n == 1 else rng.uniform(1.5, 2.2))
    zs = Lt * (np.arange(1, n + 1) / (n + 1) + rng.uniform(-0.04, 0.04, n))
    spacing = np.diff(np.r_[0.0, zs, Lt]).min()
    params = {"Rt": Rt, "Lt": Lt, "n_spokes": float(n)}
    bore = rng.uniform() < 0.6
    tools, ay_min, az_min = [], np.inf, np.inf
    v_exp = np.pi * Rt ** 2 * Lt
    for i, z in enumerate(zs):
        az = min(Rt * rng.uniform(0.15, 0.28), 0.3 * spacing)    # half-width along the beam axis
        if bore:                                                  # racetrack-like: wider across the beam
            ay = az * rng.uniform(1.0, 1.6)
        else:
            ay = az * (rng.uniform(0.7, 1.6) if rng.uniform() < 0.6 else 1.0)
        ay = min(ay, 0.35 * Rt)
        v_exp -= np.pi * ay * az * 2 * Rt * (1 - (ay / Rt) ** 2 / 6)   # chord length ≈ 2√(Rt² − y²)
        phi = 0.0 if (i == 0 or rng.uniform() < 0.7) else np.pi / 2
        t = occ.addCylinder(-(Rt + 0.01), 0, 0, 2 * (Rt + 0.01), 0, 0, 1.0)
        occ.dilate([(3, t)], 0, 0, 0, 1.0, ay, az)
        if phi:
            occ.rotate([(3, t)], 0, 0, 0, 0, 0, 1, phi)
        occ.translate([(3, t)], 0, 0, z)
        tools.append(t)
        ay_min, az_min = min(ay_min, ay), min(az_min, az)
        params.update({f"s{i}_z": z, f"s{i}_ay": ay, f"s{i}_az": az, f"s{i}_phi": phi})
    vol = _cut_all(occ, occ.addCylinder(0, 0, 0, 0, 0, Lt, Rt), tools)
    rp = min(ay_min * rng.uniform(0.45, 0.65), 0.3 * Rt) if bore else Rt * rng.uniform(0.1, 0.25)
    Lp = rp * rng.uniform(1.5, 3.0)
    if bore:
        pipes = [occ.addCylinder(0, 0, -Lp, 0, 0, Lt + 2 * Lp, rp)]
        v_exp += np.pi * rp ** 2 * 2 * sum(params[f"s{i}_az"] for i in range(n))
    else:
        pipes = [occ.addCylinder(0, 0, -Lp, 0, 0, Lp + 1e-4, rp), occ.addCylinder(0, 0, Lt - 1e-4, 0, 0, Lp + 1e-4, rp)]
    v_exp += 2 * np.pi * rp ** 2 * Lp
    vol = _fuse_all(occ, [vol] + pipes)
    _check_volume(occ, vol, v_exp)
    gap = min(zs[0] - az_min, Lt - zs[-1] - az_min, spacing - 2 * az_min if n > 1 else np.inf)
    params.update({"rp": rp, "Lpipe": Lp, "bore": float(bore), "gap_min": gap})
    return "spoke", dict(params, _h_cap=0.8 * min(ay_min, az_min, gap, rp))


def build_dtl(occ, rng):
    """Alvarez-like drift-tube linac tank: cylinder (radius Rt, axis z) of n + 1 cells of length Lc
    with n = 1–3 drift tubes at the cell boundaries.  Drift tube = annular ring (bore rb, outer
    radius Rd, length ld, rounded noses; revolved filleted profile) held by a stem (cylinder along
    +y) to the tank wall; beam pipes of the bore radius at both end plates.  b1 = 1 per tube
    (the loop through the bore)."""
    Rt = rng.uniform(0.05, 0.10)                                  # TM010-like f ≈ c·2.405/(2πRt)
    n = int(rng.integers(1, 4))
    Lc = Rt * rng.uniform(0.5, 0.9)
    Lt = (n + 1) * Lc
    Rd = Rt * rng.uniform(0.2, 0.3)
    rb = Rd * rng.uniform(0.35, 0.5)
    ld = Lc * rng.uniform(0.35, 0.6)
    rs = min(ld / 2, Rd) * rng.uniform(0.35, 0.6)
    rho_o = min(Rd - rb, ld) * rng.uniform(0.2, 0.45)             # outer nose radius
    rho_i = min(Rd - rb, ld) * rng.uniform(0.05, 0.2)             # bore edge radius
    tank = _fuse_all(occ, [occ.addCylinder(0, 0, 0, 0, 0, Lt, Rt)]
                     + [occ.addCylinder(0, 0, -rb * 2.0, 0, 0, rb * 2.0 + 1e-4, rb * 1.15),
                        occ.addCylinder(0, 0, Lt - 1e-4, 0, 0, rb * 2.0 + 1e-4, rb * 1.15)])
    tools = []
    for i in range(n):
        zc = (i + 1) * Lc
        C = [(zc - ld / 2, rb), (zc + ld / 2, rb), (zc + ld / 2, Rd), (zc - ld / 2, Rd)]
        tools.append(revolve_segments(occ, filleted(C, [rho_i, rho_i, rho_o, rho_o])))
        y0 = 0.5 * (rb + Rd)
        tools.append(occ.addCylinder(0, y0, zc, 0, Rt + 0.01 - y0, 0, rs))
    vol = _cut_all(occ, tank, tools)
    y0 = 0.5 * (rb + Rd)
    _check_volume(occ, vol, np.pi * Rt ** 2 * Lt + 2 * np.pi * (1.15 * rb) ** 2 * 2 * rb
                  - n * (np.pi * (Rd ** 2 - rb ** 2) * ld + np.pi * rs ** 2 * (Rt - y0)), tol=0.12)
    params = {"Rt": Rt, "Lc": Lc, "Lt": Lt, "n_tubes": float(n), "Rd": Rd, "rb": rb, "ld": ld, "rs": rs,
              "rho_nose": rho_o, "rho_bore": rho_i, "rpipe": 1.15 * rb, "Lpipe": 2.0 * rb}
    return "dtl", dict(params, _h_cap=0.8 * min(1.2 * rb, Lc - ld, 1.5 * rs))
