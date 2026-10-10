"""High-order field labels for the learned-field model (docs/29): CST-grade labels, cheap training.

One curved mesh per geometry, two polynomial orders on it:

    labels   HCurl(label_order = 3): eigenpairs (frequency, fields) and the cavity QoI computed
             directly from the high-order fields (highorder.mode_qoi)        — CST-grade accuracy
    model    HCurl(model_order = 2): the space the model's Ritz layer works in   — cheap training

HCurl(p2) ⊂ HCurl(p3) on the same mesh, so the label fields are carried to the model space by the
M-orthogonal projection  u₂ = argmin ‖E u₂ − u₃‖_M₃  (E = the exact p2 → p3 embedding): the best
representation the model can reach.  The frequencies stay the p3 ones (the targets the Ritz values
are trained towards); proj_err records how much of each label field lies outside the model space.

Storage (dataset_generator_3d --labels field, H5 group per geometry; read by load_field_sample):
    vol      uint8   the netgen .vol file of the straight mesh (vertex / element numbering)
    brep     uint8   the meshed OCC shape (BREP): curving on load
    deform_om / deform_a / deform_ph   the smooth deformation map (when the sample has one)
    freqs    [K]     label (p3) frequencies, GHz
    u_model  [n2, K] float32  projected label fields, model-order DOFs (wall rows 0)
    qoi      [K, Q]  label QoI (qoi_names attr), U = 1 J, copper, β = 1
    rq_model [K]     Rayleigh quotients of u_model in the model space [1/m²] (load-time check)
    proj_err [K]     ‖u₃ − E u₂‖_M / ‖u₃‖_M
    attrs    labels='field', label_order, model_order, curve, n_tets, n_dof_model, n_dof_label, …
"""
import os
import tempfile
import time

import numpy as np

from src.data_gen import highorder as ho

C0 = 299792458.0
# Curved-mesh settings measured in docs/29 §2 (pillbox with pipes, 8 % fillets, p3 labels):
#   curvaturesafety 1.5 → 9k tets but min det-J ratio 0.01: frequencies fine (1e-5), Q0 −50 %, surface
#   peaks off 100×; curvaturesafety 2 → 16k tets, ratio 0.07: Q0 / R/Q / peaks of the isolated modes
#   within 1e-6–5e-5 of the locally repaired mesh (ratio 0.35, 18k tets).  grading > 0.5 folds elements.
DEFAULTS = {"label_order": 3, "model_order": 2, "curve": 3, "maxh_factor": 3.0, "curvaturesafety": 2.0,
            "grading": 0.5, "fix_passes": 3, "fix_below": 0.1, "min_jacobian_ratio": 0.05, "retries": 1,
            "max_elements": 30000, "max_ndof": 900_000}


def curved_mesh(cad_path, h, deform=None, s=None):
    """Curved netgen mesh of the generator's CAD solid at maxh = maxh_factor·h → (mesh, shape)."""
    s = {**DEFAULTS, **(s or {})}
    mesh, shape = ho.netgen_mesh(cad_path, s["maxh_factor"] * h, curve=s["curve"], deform=deform,
                                 return_shape=True, retries=s["retries"], min_jacobian_ratio=s["min_jacobian_ratio"],
                                 fix_passes=s["fix_passes"], fix_below=s["fix_below"],
                                 curvaturesafety=s["curvaturesafety"], grading=s["grading"])
    if mesh.ne > s["max_elements"]:
        raise RuntimeError(f"curved mesh has {mesh.ne} tets > max_elements={s['max_elements']} "
                           "(small features: raise --min_fillet or --field_max_elements)")
    return mesh, shape


def project_to_model(mesh, label, model_order):
    """M-orthogonal projection of the label modes (solve_modes result: fes, gfs) onto
    HCurl(model_order) of the same mesh → (U [n_model, K], proj_err [K], rq_model [K], fes_model).

    Normal equations in the label space's inner product (its integration rule dx₃):
    M₂₂ u₂ = M₂₃ u₃ with M₂₂ = ∫ v₂·w₂ dx₃, M₂₃ = ∫ v₂·u₃ dx₃ (NGSolve sparse Cholesky on the
    model space only; nested spaces, so this is the exact best approximation).  proj_err =
    ‖u₃ − u₂‖ / ‖u₃‖ in L² (‖u₃ − u₂‖² = ‖u₃‖² − ‖u₂‖², since u₂ is the orthogonal projection).
    rq_model: Rayleigh quotients of u₂ with the MODEL operators (field_operators' integration rule
    volume_measure(model_order)) — the check the dataset repeats on the rebuilt mesh."""
    from ngsolve import BilinearForm, HCurl, InnerProduct, Integrate, TaskManager, curl
    fes3 = label["fes"]
    dx3 = ho.volume_measure(fes3.globalorder)
    fes2 = HCurl(mesh, order=int(model_order), dirichlet="wall")
    u2, v2 = fes2.TnT()
    with TaskManager():
        A = BilinearForm(u2 * v2 * dx3, symmetric=True).Assemble()
        B = BilinearForm(trialspace=fes3, testspace=fes2)
        B += fes3.TrialFunction() * v2 * dx3
        B.Assemble()
        Ainv = A.mat.Inverse(fes2.FreeDofs(), inverse="sparsecholesky")
    n, Kn = fes2.ndof, len(label["gfs"])
    U = np.zeros((n, Kn))
    rhs, x = A.mat.CreateColVector(), A.mat.CreateColVector()
    n3 = np.zeros(Kn)
    for k, g in enumerate(label["gfs"]):
        rhs.data = B.mat * g.vec
        x.data = Ainv * rhs
        U[:, k] = x.FV().NumPy()
        n3[k] = Integrate(InnerProduct(g, g) * dx3, mesh)
    dx2 = ho.volume_measure(model_order)
    with TaskManager():
        K2 = ho._tosp(BilinearForm(curl(u2) * curl(v2) * dx2, symmetric=True).Assemble().mat)
        M2 = ho._tosp(BilinearForm(u2 * v2 * dx2, symmetric=True).Assemble().mat)
    A2 = ho._tosp(A.mat)
    n2 = np.einsum("ik,ik->k", U, A2 @ U)
    err = np.sqrt(np.clip(1.0 - n2 / n3, 0.0, None))
    rq = np.einsum("ik,ik->k", U, K2 @ U) / np.einsum("ik,ik->k", U, M2 @ U)
    return U, err, rq, fes2


def _beam_axis(shape_type, mesh):
    from src.qoi.operators import beam_axis
    X = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    ax = beam_axis(shape_type, X, 1.0, np.zeros(3))
    d = ax.get("axis_dir", "z")
    return (np.eye(3)["xyz".index(d)] if isinstance(d, str) else np.asarray(d, float),
            np.asarray(ax.get("axis_point", (0.0, 0.0, 0.0)), float))


def label_sample(cad_path, h, shape_type, n_modes, deform=None, settings=None):
    """All field labels of one geometry (the generator's CAD solid at label mesh size h)."""
    from src.qoi.operators import QOI_LABELS
    s = {**DEFAULTS, **(settings or {})}
    t0 = time.perf_counter()
    mesh, shape = curved_mesh(cad_path, h, deform, s)
    t1 = time.perf_counter()
    lab = ho.solve_modes(mesh, n_modes + 1, order=s["label_order"], max_ndof=s["max_ndof"])
    K = n_modes
    axis_dir, axis_point = _beam_axis(shape_type, mesh)
    q = ho.mode_qoi(mesh, lab["gfs"][:K], lab["f_hz"][:K], axis_dir, axis_point)
    gfs_k = {"fes": lab["fes"], "gfs": lab["gfs"][:K]}
    U, err, rq, fes2 = project_to_model(mesh, gfs_k, s["model_order"])
    t2 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        vol, brep = os.path.join(tmp, "m.vol"), os.path.join(tmp, "s.brep")
        mesh.ngmesh.Save(vol)
        shape.WriteBrep(brep)
        vol_bytes, brep_bytes = open(vol, "rb").read(), open(brep, "rb").read()
    return {"vol": np.frombuffer(vol_bytes, np.uint8), "brep": np.frombuffer(brep_bytes, np.uint8),
            "deform": deform, "freqs": lab["f_hz"][:K] / 1e9, "freq_next": float(lab["f_hz"][K] / 1e9),
            "u_model": U.astype(np.float32), "rq_model": rq, "proj_err": err,
            "qoi": np.column_stack([np.asarray(q[n], float) for n in QOI_LABELS]),
            "qoi_names": list(QOI_LABELS),
            "attrs": {"labels": "field", "label_order": int(s["label_order"]), "model_order": int(s["model_order"]),
                      "curve": int(s["curve"]), "n_tets": int(mesh.ne), "n_dof_label": int(lab["fes"].ndof),
                      "n_dof_model": int(fes2.ndof), "jacobian_ratio": float(ho.jacobian_ratio(mesh)),
                      "maxh": float(s["maxh_factor"] * h), "t_curved_mesh": t1 - t0, "t_labels": t2 - t1}}


def write_field_group(g, res):
    """Write label_sample() output into an H5 group (dataset_generator_3d.write_sample)."""
    z = dict(compression="gzip", compression_opts=4)
    g.create_dataset("vol", data=res["vol"], **z)
    g.create_dataset("brep", data=res["brep"], **z)
    if res["deform"] is not None:
        for name, arr in zip(("deform_om", "deform_a", "deform_ph"), res["deform"], strict=True):
            g.create_dataset(name, data=np.asarray(arr, float))
    g.create_dataset("freqs", data=np.asarray(res["freqs"], float))
    g.create_dataset("u_model", data=res["u_model"], **z)
    for k in ("rq_model", "proj_err"):
        g.create_dataset(k, data=np.asarray(res[k], float))
    g.create_dataset("qoi", data=res["qoi"])
    g.attrs["qoi_names"] = ",".join(res["qoi_names"])
    g.attrs["freq_next"] = float(res["freq_next"])
    for k, v in res["attrs"].items():
        g.attrs[k] = v


def read_field_group(g):
    """(mesh, record) of one H5 field group: the curved mesh rebuilt exactly as it was labelled."""
    deform = None
    if "deform_om" in g:
        deform = tuple(np.asarray(g[k]) for k in ("deform_om", "deform_a", "deform_ph"))
    with tempfile.TemporaryDirectory() as tmp:
        vol, brep = os.path.join(tmp, "m.vol"), os.path.join(tmp, "s.brep")
        with open(vol, "wb") as fh:
            fh.write(np.asarray(g["vol"]).tobytes())
        with open(brep, "wb") as fh:
            fh.write(np.asarray(g["brep"]).tobytes())
        mesh = ho.load_curved_mesh(vol, brep, int(g.attrs["curve"]), deform)
    rec = {"freqs": np.asarray(g["freqs"], float), "u_model": np.asarray(g["u_model"], np.float64),
           "rq_model": np.asarray(g["rq_model"], float), "proj_err": np.asarray(g["proj_err"], float),
           "qoi": np.asarray(g["qoi"], float), "qoi_names": str(g.attrs.get("qoi_names", "")).split(","),
           "freq_next": float(g.attrs.get("freq_next", np.nan)),
           "model_order": int(g.attrs["model_order"]), "label_order": int(g.attrs["label_order"]),
           "shape_type": str(g.attrs.get("shape_type", ""))}
    return mesh, rec
