"""Standalone copy of the half-cell construction of src/data_gen/cavity_shapes.py (read-only reuse),
plus wall-angle evaluation. Lengths in mm or m (scale-free)."""
import numpy as np
from scipy.optimize import least_squares

def tangent(Req, Riris, L, A, B, a, b, allow_reentrant=False):
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a*np.sin(t), -b*np.cos(t)])
    P2 = lambda s: ce + np.array([-A*np.cos(s), B*np.sin(s)])
    T1 = lambda t: np.array([a*np.cos(t), b*np.sin(t)])
    T2 = lambda s: np.array([A*np.sin(s), B*np.cos(s)])
    cross = lambda u, v: u[0]*v[1]-u[1]*v[0]
    def res(x):
        t, s = x
        d = P2(s) - P1(t)
        return [cross(T1(t), T2(s))/(a*A), cross(T1(t), d)/(a*L)]
    sol = least_squares(res, [np.pi/2, 0.0], bounds=([1e-3, -np.pi/2+1e-3], [np.pi-1e-3, np.pi/2]))
    t, s = sol.x
    if np.max(np.abs(sol.fun)) > 1e-9 or (P2(s)-P1(t)) @ T1(t) <= 0:
        return None
    d = P2(s) - P1(t)
    # angle of the wall line w.r.t. +z, going from iris to equator (90 = vertical; >90 = re-entrant in OUR docs' convention)
    theta = np.degrees(np.arctan2(d[1], d[0]))
    n = 24
    wall = np.vstack([np.array([P1(x) for x in np.linspace(0, t, n)]), np.array([P2(x) for x in np.linspace(s, np.pi/2, n)])])
    reentrant = np.any(np.diff(wall[:, 0]) < -1e-12) or np.any(np.diff(wall[:, 1]) < -1e-12)
    zmin = wall[:, 0].min()   # must stay > 0 (metal between neighbouring cells) for a re-entrant wall
    return dict(theta=theta, alpha_shemelin=180.0-theta, reentrant=bool(reentrant), zmin=zmin,
                ok_ours=(not reentrant) and (Riris + b < Req - B))
