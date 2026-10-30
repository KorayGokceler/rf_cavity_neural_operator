"""p2 reference solves for the generator tests: gmsh OCC solid -> BREP [m] -> curved netgen mesh
(src.data_gen.highorder.netgen_mesh) -> HCurl p2 eigenmodes (highorder.solve_modes).

This is the production label path at the model order (curved mesh, one explicit integration rule, kernel
projection of the Ritz block), without adaptivity: the physical checks the generator tests make (analytic
spectra, handles, isolated conductors) run on the same discretisation the model and the labels use.
Import NGSolve only after torch (tests/conftest.py imports torch first).
"""
import numpy as np

from src.data_gen import highorder as ho

C0 = ho.C0


def write_brep(build, path):
    """Build a solid with `build(gmsh.model.occ)` in a private gmsh session and write its BREP [m] to `path`."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("p2_reference")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(3)) != 1:
            raise RuntimeError(f"{len(gmsh.model.getEntities(3))} volumes")
        gmsh.write(str(path))
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()
    return path


def curved_mesh(brep, maxh, curve=3, **meshing):
    """Curved netgen mesh of the BREP solid (boundary "wall"), with the label path's repair settings
    (field_labels.DEFAULTS: min Jacobian ratio 0.05, bisect below 0.1); `meshing` goes to GenerateMesh."""
    opts = {"min_jacobian_ratio": 0.05, "fix_below": 0.1, "retries": 1, "split": False, **meshing}
    return ho.netgen_mesh(str(brep), maxh, curve=curve, **opts)


def solve(mesh, k, order=2):
    """The k lowest physical modes of the curved mesh: dict(f_hz [k], lam [k] (1/m^2), U [n_dof, k]
    (unit M-norm HCurl DOFs, wall rows 0), fes, info) from highorder.solve_modes."""
    res = ho.solve_modes(mesh, k, order=order, tol=1e-10)
    res["U"] = np.column_stack([np.asarray(g.vec.FV().NumPy()).copy() for g in res["gfs"]])
    return res


def occ_modes(build, maxh, k, tmp_path, order=2, curve=3, **meshing):
    """build -> BREP -> curved mesh -> k modes: (mesh, solve_modes result)."""
    brep = write_brep(build, tmp_path / "solid.brep")
    mesh = curved_mesh(brep, maxh, curve=curve, **meshing)
    return mesh, solve(mesh, k, order)
