# Copy of deform_map logic (src/data_gen/dataset_generator_3d.py) to measure displacement magnitudes.
import numpy as np

def deform_map(X, rng, lip_max=0.5):
    diag = float(np.linalg.norm(X.max(0) - X.min(0)))
    n = int(rng.integers(3, 9))
    dirs = rng.standard_normal((n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    om = dirs * (2 * np.pi / (diag * rng.uniform(0.25, 1.5, n)))[:, None]
    a = rng.standard_normal((n, 3)); ph = rng.uniform(0, 2 * np.pi, n)
    lip = lip_max * rng.uniform(0.3, 1.0)
    a *= lip / (np.linalg.norm(a, axis=1) * np.linalg.norm(om, axis=1)).sum()
    return om, a, ph, lip, diag

# point cloud: a 1-cell TESLA-like cell (Req 103 mm, L 115 mm) with pipes of radius 35 mm, length 2*35 mm each side
rng0 = np.random.default_rng(0)
pts = []
for _ in range(20000):
    z = rng0.uniform(-0.0577 - 0.07, 0.0577 + 0.07)
    rmax = 0.1033 if abs(z) < 0.0577 else 0.035
    r = rmax * np.sqrt(rng0.uniform()); t = rng0.uniform(0, 2*np.pi)
    pts.append((r*np.cos(t), r*np.sin(t), z))
X = np.array(pts)
for lip_max in (0.5, 0.05, 0.01, 0.002):
    md, mdr = [], []
    for s in range(400):
        rng = np.random.default_rng([0, s, 1])
        om, a, ph, lip, diag = deform_map(X, rng, lip_max)
        d = np.sin(X @ om.T + ph) @ a
        disp = np.linalg.norm(d, axis=1)
        md.append(disp.max()); mdr.append(np.sqrt((disp**2).mean()))
    md, mdr = np.array(md), np.array(mdr)
    print(f"lip_max={lip_max}: diag={diag*1e3:.0f} mm; max|delta| median {np.median(md)*1e3:.2f} mm "
          f"(5-95%: {np.percentile(md,5)*1e3:.2f}-{np.percentile(md,95)*1e3:.2f} mm) = "
          f"{np.median(md)/diag*100:.2f}% of diag, {np.median(md)/0.1033*100:.2f}% of Req; "
          f"rms|delta| median {np.median(mdr)*1e3:.2f} mm")
