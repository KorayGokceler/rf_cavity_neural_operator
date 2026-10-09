"""E8: nodal (vector P1 Lagrange) elements for curl curl E = k² E in the PEC box 1×0.8×0.6.
Tangential components are set to 0 on each face (axis-aligned, so this is componentwise).
(a) pure curl–curl: the discrete kernel is NOT the gradient space → spurious non-zero modes
(b) regularised curl–curl + s·(div, div) (grad–div penalty): works on this convex box but
    converges to the wrong spectrum on domains with re-entrant edges (Costabel–Dauge)."""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import Basis, BilinearForm, ElementTetP1, ElementVector, condense
from skfem.helpers import curl, div, dot
from n0lib import box_mesh, box_spectrum

a, b, d = 1.0, 0.8, 0.6
ref = box_spectrum(a, b, d)[:8]
print("exact k²:", np.round(ref, 2))
for n in (8, 12, 16):
    m = box_mesh(a, b, d, n)
    bs = Basis(m, ElementVector(ElementTetP1()))
    Kc = BilinearForm(lambda u, v, w: dot(curl(u), curl(v))).assemble(bs)
    Kd = BilinearForm(lambda u, v, w: div(u) * div(v)).assemble(bs)
    M = BilinearForm(lambda u, v, w: dot(u, v)).assemble(bs)
    # tangential BC: on x-faces E_y = E_z = 0, etc.  DOF ordering of ElementVector: nodal (x, y, z) per node
    x = m.p
    D = []
    for comp in range(3):
        dofs = bs.nodal_dofs[comp]
        on = np.zeros(x.shape[1], bool)
        for ax, L in ((0, a), (1, b), (2, d)):
            if ax != comp:
                on |= (np.abs(x[ax]) < 1e-12) | (np.abs(x[ax] - L) < 1e-12)
        D.append(dofs[on])
    D = np.concatenate(D)
    for s, name in ((0.0, "curl-curl only"), (1.0, "curl-curl + div-div (s=1)")):
        A, Mc, _, _ = condense(Kc + s * Kd, M, D=D)
        sig = 0.5 * ref[0]
        vals = np.sort(spla.eigsh(A.tocsc(), k=40, M=Mc.tocsc(), sigma=sig, which='LM')[0])
        vals = vals[vals > 1e-6 * ref[0]][:16]
        spur = [v for v in vals if np.min(np.abs(v / ref[:8] - 1)) > 0.05 and v < ref[7] * 1.02]
        print(f"n={n:2d} dof={A.shape[0]:6d} {name:26s}: lowest non-zero k² {np.array2string(vals[:12], precision=1)}  "
              f"#values below λ8 not within 5% of an exact one: {len(spur)}")
