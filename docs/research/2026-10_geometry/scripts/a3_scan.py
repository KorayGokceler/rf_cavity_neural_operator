"""Parametric scan of the generator's reentrant profile outside its ranges (report a3, docs/33)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_axisym as ax
cs = ax.cs

def profile(R, Lc, rp, g, t_tip, alpha_deg, f_tip=0.5, f_root=0.2, f_out=0.5, Lp_rp=2.5):
    """Same construction as cavity_shapes.build_reentrant, with chosen (not drawn) parameters."""
    zt = (Lc - g) / 2
    rt = rp + t_tip
    rn = rt + zt * np.tan(np.radians(alpha_deg))
    Lp = rp * Lp_rp
    rho_tip = cs.fillet_floor(t_tip * f_tip, R, cap=0.5 * t_tip)
    rho_root = cs.fillet_floor(min(R - rn, rn - rp) * f_root, R, cap=0.45 * min(R - rn, rn - rp))
    rho_out = cs.fillet_floor(min(R - rn, Lc / 2) * f_out, R)
    C = [(-Lp, 0), (-Lp, rp), (zt, rp), (zt, rt), (0, rn), (0, R), (Lc, R), (Lc, rn),
         (Lc - zt, rt), (Lc - zt, rp), (Lc + Lp, rp), (Lc + Lp, 0)]
    rad = [0, 0, rho_tip, rho_tip, rho_root, rho_out, rho_out, rho_root, rho_tip, rho_tip, 0, 0]
    return cs.filleted(C, rad)

R = 0.05
print("Lc/R  g/Lc  rp/R tt/R alpha | R/lam  g/lam  Lc/lam | R/Q_linac  G    T")
for rpR in (0.15, 0.20):
    for LcR in (0.8, 1.0, 1.2, 1.4, 1.7):
        for gL in (0.4, 0.6, 0.75):
            for ttR, al in ((0.10, 20.0),):
                segs = profile(R, LcR * R, rpR * R, gL * LcR * R, ttR * R, al)
                q = ax.solve(segs, R, nev=1)[0][0]
                lam = ax.C0 / q["f"]
                print(f"{LcR:4.1f} {gL:5.2f} {rpR:5.2f} {ttR:4.2f} {al:5.1f} | {R/lam:5.3f} {gL*LcR*R/lam:6.3f} {LcR*R/lam:6.3f} | {q['RQ']:7.1f} {q['G']:6.1f} {q['T']:.3f}")
# strongly capacity-loaded, MAX IV-like ratios: rod (nose) radius 0.146 R, gap 0.122 R; nose = straight rod (alpha 0)
print("capacity-loaded (rt/R = 0.146, g/R = 0.122, alpha 0), rp/R 0.06 / tt 0.086:")
for LcR in (0.8, 1.2, 1.6, 2.0):
    segs = profile(R, LcR * R, 0.06 * R, 0.122 * R, 0.086 * R, 0.0, f_out=0.3)
    q = ax.solve(segs, R, nev=1)[0][0]
    lam = ax.C0 / q["f"]
    print(f"  Lc/R {LcR:3.1f}: R/lam {R/lam:.3f}  R/Q {q['RQ']:6.1f}  G {q['G']:5.1f}  T {q['T']:.3f}")
