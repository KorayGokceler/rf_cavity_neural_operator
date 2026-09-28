"""Geometry families for the 3D generator: realistic RF cavities and free-form solids.

Axisymmetric cavities are built as a meridian profile in the (z, r) half-plane
(z = beam axis), revolved 2π about z with gmsh OCC.  Beam pipes are closed by flat
PEC end caps, as in a closed eigenmode solve.  Lengths in metres.

- elliptical:    TESLA-type elliptical cells (1–3) with beam pipes.  Half-cell =
                 iris ellipse (a, b) at the iris plane + equator ellipse (A, B) at the
                 equator, joined by their common tangent (the standard construction).
                 TESLA mid-cell: Req 103.3, Riris 35, A = B = 42, a 12, b 19, L 57.7 mm.
- reentrant:     nose-cone (klystron / IOT type) cavity with beam pipes, filleted corners.
- pillbox_pipes: pillbox with beam pipes, rounded iris edge and outer corners.
- freeform:      star-shaped superellipsoid × smooth random modulation × lobes, then a
                 bend / twist / taper warp (each an injective map, so the surface never
                 self-intersects and the solid stays a topological ball); returned as a
                 closed triangulated surface for gmsh's discrete-surface remeshing.

Out-of-distribution families (OOD_FAMILIES, never in the default training set):

- box:             rectangular cavity: flat faces, sharp edges (analytic spectrum).
- coax_qw:         quarter-wave coaxial resonator: inner post from one end cap with a
                   capacitive gap to the other (deep narrow annulus, blind hole → ball).
- pillbox_port:    pillbox with beam pipes and 1–2 radial side ports (breaks the axial
                   symmetry every training cavity has).
- elliptical_long: 4–5 elliptical cells (training: 1–3), same cell-shape ranges.
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
    occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * np.pi)


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


def draw_elliptical(rng):
    """Random TESLA-type parameters (ratios around TESLA / LEP / CEBAF-like designs)."""
    for _ in range(100):
        Req = rng.uniform(0.04, 0.11)
        p = {"Req": Req, "Riris": Req * rng.uniform(0.25, 0.42), "L": Req * rng.uniform(0.45, 0.65)}
        p["A"] = p["L"] * rng.uniform(0.6, 0.85)
        p["B"] = p["A"] * rng.uniform(0.85, 1.25)
        p["a"] = p["L"] * rng.uniform(0.15, 0.30)
        p["b"] = p["a"] * rng.uniform(1.0, 1.9)
        if p["Riris"] + p["b"] >= p["Req"] - p["B"]:
            continue
        wall = half_cell_wall(**{k: p[k] for k in ("Req", "Riris", "L", "A", "B", "a", "b")})
        if wall is not None:
            p["n_cells"] = float(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            p["Lpipe"] = p["Riris"] * rng.uniform(1.0, 2.5)
            return p, wall
    raise RuntimeError("no valid elliptical cell parameters")


def elliptical_segments(p, wall):
    """Profile segments: pipe – n_cells × (half-cell + mirrored half-cell) – pipe."""
    L, Ri, Lp, n = p["L"], p["Riris"], p["Lpipe"], int(p["n_cells"])
    cell = np.vstack([wall, (np.array([2 * L, 0]) + np.array([-1, 1]) * wall[::-1])[1:]])
    chain = np.vstack([cell[(1 if k else 0):] + [2 * L * k, 0] for k in range(n)])
    z1 = 2 * L * n
    return [('line', (-Lp, 0.0), (-Lp, Ri)), ('line', (-Lp, Ri), (0.0, Ri)),
            ('spline', [tuple(q) for q in chain]),
            ('line', (z1, Ri), (z1 + Lp, Ri)), ('line', (z1 + Lp, Ri), (z1 + Lp, 0.0)),
            ('line', (z1 + Lp, 0.0), (-Lp, 0.0))]


def build_elliptical(occ, rng, params=None):
    p, wall = (params, half_cell_wall(**{k: params[k] for k in ("Req", "Riris", "L", "A", "B", "a", "b")})) \
        if params is not None else draw_elliptical(rng)
    if wall is None:
        raise RuntimeError("invalid elliptical parameters")
    revolve_segments(occ, elliptical_segments(p, wall))
    return "elliptical", dict(p, _h_cap=0.7 * p["Riris"])


# ─────────────────────────── re-entrant / pillbox with pipes ───

def build_reentrant(occ, rng):
    for _ in range(100):
        R = rng.uniform(0.03, 0.07)
        Lc = R * rng.uniform(0.4, 1.0)
        rp = R * rng.uniform(0.10, 0.25)
        g = Lc * rng.uniform(0.15, 0.6)                 # nose-to-nose gap
        t_tip = R * rng.uniform(0.08, 0.2)              # nose tip wall thickness
        alpha = np.radians(rng.uniform(0, 35))          # nose outer cone half-angle
        zt = (Lc - g) / 2
        rt = rp + t_tip
        rn = rt + zt * np.tan(alpha)
        if rn < 0.75 * R and zt > 0.05 * R:
            break
    else:
        raise RuntimeError("no valid re-entrant parameters")
    Lp = rp * rng.uniform(1.5, 4.0)
    rho_tip = t_tip * rng.uniform(0.2, 0.5)
    rho_root = (R - rn) * rng.uniform(0.05, 0.3)
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
    p, wall = draw_elliptical(rng)
    p["n_cells"] = float(rng.choice([4, 5]))
    revolve_segments(occ, elliptical_segments(p, wall))
    return "elliptical_long", dict(p, _h_cap=0.7 * p["Riris"])


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
