"""Derived numbers for the A1 report (all formulas stated; inputs from the report's tables)."""
import numpy as np
C = 299.792458  # mm GHz
# 1) end half-cell / mid half-cell ratios  [A, B, a, b, Ri, L]
mid_end = {
 "TESLA end1/mid": ([40.3, 40.3, 10, 13.5, 39, 56.0], [42, 42, 12, 19, 35, 57.7]),
 "TESLA end2/mid": ([42, 42, 9, 12.8, 39, 57.0], [42, 42, 12, 19, 35, 57.7]),
 "ERL endL/mid":   ([50.9, 45.3, 8.4, 11.5, 39, 59.988], [43.99, 35.06, 12.53, 20.95, 35, 57.6524]),
 "ERL endR/mid":   ([52.1, 47.9, 9.9, 11.3, 37, 62.665], [43.99, 35.06, 12.53, 20.95, 35, 57.6524]),
 "C3795 end/mid":  ([62.58, 57.54, 17.21, 12.0, 80.0, 93.795], [62.22, 66.13, 30.22, 23.11, 72, 93.5]),
}
print("1) end/mid ratios  A, B, a, b, Ri, L   (ours: A,B,a,b,L in 0.85-1.15; Ri in 1.0-1.35)")
for k, (e, m) in mid_end.items():
    r = np.array(e)/np.array(m)
    flag = ["A", "B", "a", "b", "Ri", "L"]
    out = [flag[i] for i in range(6) if (i != 4 and not 0.85 <= r[i] <= 1.15) or (i == 4 and not 1.0 <= r[i] <= 1.35)]
    print(f"  {k:15s}", np.round(r, 3), " outside:", out)
# 2) iris-tip and equator curvature radii (ellipse apex radii a^2/b and A^2/B) relative to Req
print("2) iris tip radius a^2/b, equator apex radius A^2/B (mm, /Req)")
for n, (A, B, a, b, Req) in {"TESLA": (42, 42, 12, 19, 103.3), "LL": (50.052, 36.5, 7.6, 10, 98.58), "ICHIRO": (50.052, 34.222, 7.6, 9.945, 98.14),
                            "NLSF": (47.152, 31.35, 10.5, 15.5, 98.58), "ERL mid": (43.99, 35.06, 12.53, 20.95, 101.205),
                            "C3795": (62.22, 66.13, 30.22, 23.11, 171.2), "C3794": (73.52, 131.75, 106.25, 118.7, 369.63)}.items():
    print(f"  {n:8s} iris {a*a/b:6.2f} mm ({a*a/b/Req:.3f} Req)   equator {A*A/B:6.1f} mm ({A*A/B/Req:.3f} Req)")
# 3) TM010 passband: f(theta) ~ fbar (1 - k/2 cos theta), theta = m pi / N; spacing pi vs (N-1)pi/N
print("3) pi-mode spacing  df = f k/2 (1 - cos(pi/N))  and field-flatness factor N^2/k(%)")
for name, f, k, Ns in [("TESLA-like", 1300., 0.0187, [1, 2, 3, 5, 6, 7, 9]), ("LL", 1300., 0.0152, [9]), ("ESS med", 704.42, 0.012, [6]),
                       ("SNS", 805., 0.016, [6]), ("CEBAF OC", 1497., 0.0329, [5]), ("CEBAF LL (C100)", 1497., 0.0149, [7])]:
    for N in Ns:
        if N == 1: continue
        df = f*k/2*(1-np.cos(np.pi/N))
        print(f"  {name:16s} N={N}: df(pi,(N-1)pi/N) = {df:6.3f} MHz ({df/f:.2e} rel),  N^2/k = {N*N/(100*k):5.1f}")
# 4) beam-pipe cutoffs
x01, x11 = 2.405, 1.841
print("4) pipe cutoffs TE11 / TM01 (GHz) and ratio to f0")
for n, rp, f0 in [("TESLA 39 mm", 39, 1.3), ("CEPC 78 mm (derived)", 78, 0.65), ("KEKB iris 110 mm", 110, 0.5089), ("CESR 120 mm", 120, 0.5), ("FCC C3794 150 mm", 150, 0.4008), ("3.9 GHz end 20 mm", 20, 3.9)]:
    fte, ftm = x11*C/(2*np.pi*rp), x01*C/(2*np.pi*rp)
    print(f"  {n:22s} TE11 {fte:6.3f}  TM01 {ftm:6.3f}   TE11/f0 {fte/f0:5.2f}  TM01/f0 {ftm/f0:5.2f}")
print("   CEPC pipe radius from published cutoffs: TM01 1471 MHz ->", round(x01*C/(2*np.pi*1.471), 1), "mm;  TE11 1126 MHz ->", round(x11*C/(2*np.pi*1.126), 1), "mm")
# 5) evanescent decay of TM010 (TM01 in pipe) and TE111 (TE11 in pipe) for TESLA-size pipe; our pipe length 1.2-2.5 rp
rp = 39.0
for lab, xc, f in [("TM010 in TM01 pipe", x01, 1.3), ("TE111 (~1.7 GHz) in TE11 pipe", x11, 1.7)]:
    kc, k = xc/rp, 2*np.pi*f/C
    alpha = np.sqrt(kc*kc - k*k)
    print(f"5) {lab}: decay length {1/alpha:5.1f} mm; amplitude at cap for pipe 1.2-2.5 rp: {np.exp(-alpha*1.2*rp):.3f} - {np.exp(-alpha*2.5*rp):.4f}")
# 6) coax stub with antenna and PEC cap: quarter-wave f = c/(4 l); our stub geometry l ~ (0.5 + 0.8..2.0) rp
for rp in [39.0]:
    for l in [(0.5+0.8)*rp, (0.5+2.0)*rp]:
        print(f"6) capped coax lambda/4 resonance for l = {l:5.1f} mm: {C/(4*l):5.2f} GHz (TESLA f0 = 1.30)")
# 7) hollow stub TE11 cutoff relative to f0 (TESLA-size): rs = 0.25..0.6 rp
for rs in [0.25*39, 0.6*39]:
    print(f"7) hollow stub r = {rs:5.1f} mm: TE11 cutoff {x11*C/(2*np.pi*rs):5.2f} GHz = {x11*C/(2*np.pi*rs)/1.3:4.1f} f0")
# 8) L/Req per beta from TESLA calibration and lambda/4 cells
print("8) CEBAF inner cells (MOP84 diameters): Ri/Req and (lambda/4)/Req at 1.497 GHz")
for n, Deq, Dir in [("OC", 187.0, 70.0), ("HG", 180.5, 61.4), ("LL", 174.0, 53.0)]:
    print(f"   {n}: Ri/Req = {Dir/Deq:.3f}, L/Req = {C/1.497/4/(Deq/2):.3f}")
print("   CEPC 650 MHz: L(lambda/4)/Req =", round(C/0.65/4/205, 3), "; pipe/Req =", round(78/205, 3))
print("   LHeC 802 MHz: L/Req =", round(C/0.8016/4/163.975, 3))
print("   HB650 Ri = 59 mm; beta 0.92 -> L = beta lambda/4 =", round(0.92*C/0.65/4, 1), "mm")
print("   SNS derived Epk/Eacc =", round(27.5/10.1, 2), round(35/15.6, 2), " Bpk/Eacc =", round(58/10.1, 2), round(76/15.6, 2))
print("   HB650 derived Bpk/Eacc =", round(73.1/18.8, 2))
