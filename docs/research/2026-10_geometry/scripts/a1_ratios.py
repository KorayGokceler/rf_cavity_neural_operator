"""Dimensionless ratios of published / code-sourced mid half-cells vs. our random ranges (cavity_shapes._draw_half).
Order of the code-sourced lists: [A, B, a, b, Ri, L, Req] (mm)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_halfcell import tangent
C = 299.792458  # mm*GHz
D = {  # name: (A, B, a, b, Ri, L, Req, f_GHz, beta, provenance)
 "TESLA mid":        (42, 42, 12, 19, 35, 57.7, 103.3, 1.3, 1.0, "PRSTAB 3 092001 T.III"),
 "TESLA end1":       (40.3, 40.3, 10, 13.5, 39, 56.0, 103.3, 1.3, 1.0, "PRSTAB 3 092001 T.III"),
 "TESLA end2":       (42, 42, 9, 12.8, 39, 57.0, 103.3, 1.3, 1.0, "PRSTAB 3 092001 T.III"),
 "LL mid":           (50.052, 36.5, 7.6, 10.0, 30, 57.7, 98.58, 1.3, 1.0, "code+partial excerpt"),
 "ICHIRO mid":       (50.052, 34.222, 7.6, 9.945, 30, 57.7, 98.14, 1.3, 1.0, "code+partial excerpt"),
 "our ilc_ll anchor":(50.1, 34.2, 7.6, 10.0, 30, 57.7, 98.6, 1.3, 1.0, "cavity_shapes.ANCHORS"),
 "NLSF mid":         (47.152, 31.35, 10.5, 15.5, 32, 57.7, 98.58, 1.3, 1.0, "code"),
 "NLSF_RE mid":      (49, 35.30, 10.5, 17, 32, 57.7, 98.58, 1.3, 1.0, "code"),
 "Cornell ERL mid":  (43.99, 35.06, 12.53, 20.95, 35, 57.6524, 101.205, 1.3, 1.0, "code"),
 "Cornell ERL endL": (50.9, 45.3, 8.4, 11.5, 39, 59.988, 101.205, 1.3, 1.0, "code"),
 "Cornell ERL endR": (52.1, 47.9, 9.9, 11.3, 37, 62.665, 101.205, 1.3, 1.0, "code"),
 "FCC-ee C3794 mid": (73.52, 131.75, 106.25, 118.7, 150, 187, 369.63, 0.4008, 1.0, "code"),
 "FCC-ee C3795 mid": (62.22, 66.13, 30.22, 23.11, 72, 93.5, 171.2, 0.8016, 1.0, "code"),
 "FCC-ee C3795 end": (62.58, 57.54, 17.21, 12.0, 80.0, 93.795, 171.2, 0.8016, 1.0, "code"),
 "FCCUROS5 mid":     (67.72, 57.45, 21.75, 35.95, 60, 93.5, 166.591, 0.8016, 1.0, "code"),
}
R = {"Ri/Req": (0.25, 0.42), "L/Req": (0.45, 0.65), "A/L": (0.6, 0.85), "B/A": (0.85, 1.25), "a/L": (0.15, 0.30), "b/a": (1.0, 1.9)}
print(f"{'design':18s} {'f GHz':>6s} {'L/(lam/4)':>9s} " + " ".join(f"{k:>7s}" for k in R) + "   theta  alphaSh  ours?  outside")
for n, (A, B, a, b, Ri, L, Req, f, beta, prov) in D.items():
    lam4 = C / f / 4
    r = {"Ri/Req": Ri/Req, "L/Req": L/Req/beta, "A/L": A/L, "B/A": B/A, "a/L": a/L, "b/a": b/a}
    out = [k for k, v in r.items() if not (R[k][0] - 1e-9 <= v <= R[k][1] + 1e-9)]
    t = tangent(Req, Ri, L, A, B, a, b)
    th = "  n/a " if t is None else f"{t['theta']:6.1f}"
    al = "  n/a " if t is None else f"{t['alpha_shemelin']:6.1f}"
    ok = (t is not None) and t["ok_ours"]
    print(f"{n:18s} {f:6.3f} {L/lam4:9.3f} " + " ".join(f"{v:7.3f}" for v in r.values()) + f"  {th}  {al}  {str(ok):5s}  {','.join(out) if out else '-'}")
# 3.9 GHz (FNAL TM-2210 excerpt: Ri 15 mid, 20 end, Req 35.787)
lam4 = C/3.9/4
print("\n3.9 GHz FNAL: Ri/Req =", round(15/35.787, 3), " end Ri/Req =", round(20/35.787, 3), " end/mid iris =", round(20/15, 3),
      " L(=lam/4)/Req =", round(lam4/35.787, 3), " (lam/4 =", round(lam4, 2), "mm)")
print("LHeC 802 MHz Req = 327.95/2 =", 327.95/2, "mm; pillbox-equivalent 2.405c/(2 pi f) =", round(2.405*C/(2*np.pi*0.8016), 1), "mm")
print("TESLA Req / (2.405 c/(2 pi f)) =", round(103.3/(2.405*C/(2*np.pi*1.3)), 3))
