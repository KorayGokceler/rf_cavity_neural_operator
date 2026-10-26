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
    vol      uint8   the netgen .vol file of the MODEL mesh (straight tets + geometry info; with adaptive
                     labels the finest refinement level with ≤ model_max_elements tets, docs/30 §4)
    brep     uint8   the meshed OCC shape (BREP): curving on load
    deform_om / deform_a / deform_ph   the smooth deformation map (when the sample has one)
    freqs    [K]     label (p3) frequencies, GHz
    u_model  [n2, K] float32  projected label fields, model-order DOFs (wall rows 0)
    qoi      [K, Q]  label QoI (qoi_names attr), U = 1 J, copper, β = 1; L_acc from
                     src.qoi.operators.acceleration_length (axis chord without a standard length)
    axis_t   [P]     positions along the beam axis inside the label mesh [m] (attrs axis_dir, axis_point,
                     axis_dt; L_acc attr = the accelerating length, NaN = axis chord)
    axis_ez  [K, P]  E along the axis there [V/m] at U = 1 J (float32): T, V, R/Q at any β
    rq_model [K]     Rayleigh quotients of u_model in the model space [1/m²] (load-time check)
    proj_err [K]     ‖u₃ − E u₂‖_M / ‖u₃‖_M
    qfeat_wall [Nt, nq, 5] float32  the wall-distance / torsion columns of the model's token features at the
                     quadrature points of the stored model mesh (highorder.wall_features_for_mesh; attr
                     qfeat_columns).  Optional: the loader computes them (~1.3–2 s per 9k tets) when absent;
                     scripts/add_field_features.py adds them to existing files.
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
            "max_elements": 30000, "max_ndof": 900_000,
            # adaptive refinement (src/data_gen/adaptive.py, docs/30): start coarser (curvaturesafety 1.5,
            # falling back to `curvaturesafety` when that folds elements) and refine where the error is
            "adaptive": True, "adapt_cs": 1.5, "tol_f": 1e-6, "tol_q": 1e-3, "adapt_max_iter": 8, "theta": 0.5,
            # model mesh (docs/30 §4): the finest adaptive level with ≤ model_max_elements tets; the labels
            # of the final (finest) level are L2-projected onto its HCurl(model_order) space
            "model_max_elements": 10000}

PEAK_QOI = ("Epk_Eacc", "Bpk_Eacc_mT_per_MVm")


def curved_mesh(cad_path, h, deform=None, s=None):
    """Curved netgen mesh of the generator's CAD solid at maxh = maxh_factor·h → (mesh, shape)."""
    s = {**DEFAULTS, **(s or {})}
    mesh, shape = ho.netgen_mesh(cad_path, s["maxh_factor"] * h, curve=s["curve"], deform=deform,
                                 return_shape=True, retries=s["retries"], min_jacobian_ratio=s["min_jacobian_ratio"],
                                 fix_passes=s["fix_passes"], fix_below=s["fix_below"],
                                 curvaturesafety=s["curvaturesafety"], grading=s["grading"],
                                 max_elements=s["max_elements"])
    if mesh.ne > s["max_elements"]:
        raise RuntimeError(f"curved mesh has {mesh.ne} tets > max_elements={s['max_elements']} "
                           "(small features: raise --min_fillet or --field_max_elements)")
    return mesh, shape


def start_mesh(cad_path, h, deform, s, css):
    """Curved mesh at the first curvaturesafety of css that neither folds elements nor exceeds max_elements
    → (mesh, shape, cs); a finer value is not tried after a too-large mesh."""
    err, too_fine = None, np.inf
    for cs in dict.fromkeys(css):
        if cs >= too_fine:                            # a finer curvaturesafety only has more tets
            continue
        try:
            mesh, shape = curved_mesh(cad_path, h, deform, {**s, "curvaturesafety": cs, "retries": 0})
            return mesh, shape, cs
        except ho.TooManyElements as e:
            err, too_fine = e, cs
        except RuntimeError as e:                     # folded elements
            err = e
    raise err


def project_to_model(mesh, label, model_order, unit_norm=False):
    """M-orthogonal projection of the label modes (solve_modes result: fes, gfs) onto
    HCurl(model_order) of the same mesh → (U [n_model, K], proj_err [K], rq_model [K], fes_model).

    Normal equations in the label space's inner product (its integration rule dx₃):
    M₂₂ u₂ = M₂₃ u₃ with M₂₂ = ∫ v₂·w₂ dx₃, M₂₃ = ∫ v₂·u₃ dx₃ (NGSolve sparse Cholesky on the
    model space only; nested spaces, so this is the exact best approximation).  proj_err =
    ‖u₃ − u₂‖ / ‖u₃‖ in L² (‖u₃ − u₂‖² = ‖u₃‖² − ‖u₂‖², since u₂ is the orthogonal projection).
    rq_model: Rayleigh quotients of u₂ with the MODEL operators (field_operators' integration rule
    volume_measure(model_order)) — the check the dataset repeats on the rebuilt mesh.
    unit_norm: the label modes have unit norm in dx₃ (solve_modes normalises with the same rule), so the
    norm integrals are skipped (1.5 s of 5 s on 158k label DOFs; they give 1 ± 1e-15)."""
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
        n3[k] = 1.0 if unit_norm else Integrate(InnerProduct(g, g) * dx3, mesh)
    dx2 = ho.volume_measure(model_order)
    with TaskManager():
        K2 = ho._tosp(BilinearForm(curl(u2) * curl(v2) * dx2, symmetric=True).Assemble().mat)
        M2 = ho._tosp(BilinearForm(u2 * v2 * dx2, symmetric=True).Assemble().mat)
    A2 = ho._tosp(A.mat)
    n2 = np.einsum("ik,ik->k", U, A2 @ U)
    err = np.sqrt(np.clip(1.0 - n2 / n3, 0.0, None))
    rq = np.einsum("ik,ik->k", U, K2 @ U) / np.einsum("ik,ik->k", U, M2 @ U)
    return U, err, rq, fes2


def transfer_to_model(gfs, mesh_model, model_order, label_order):
    """M-orthogonal (L2) projection of label fields living on ANOTHER (finer) mesh onto
    HCurl(model_order) of mesh_model → (U [n_model, K], proj_err [K], rq_model [K], fes_model).  Both meshes
    are curved to the same CAD, so the domains agree to the geometry approximation.  proj_err is relative
    to ‖u₃‖ on the model domain; rq_model uses the model operators (field_operators' rule).

    The label fields are evaluated at the model mesh's integration points (the rule of dxq) with ONE point
    search for all modes: the physical points are located in the label mesh once and the values of every
    mode are put on the model mesh's IntegrationRuleSpace, where the linear forms and norms read them back
    at the same points.  (Integrating the label GridFunctions directly searches every point again for each
    form and mode: 253 s of 664 s per sample at 430k label DOFs, docs/31 §3.7; same result to 1e-14.)"""
    from ngsolve import (CF, TET, VOL, BilinearForm, GridFunction, HCurl, InnerProduct, Integrate,
                         IntegrationRule, LinearForm, TaskManager, curl, x, y, z)
    from ngsolve.comp import IntegrationRuleSpace
    dxq = ho.volume_measure(label_order)
    rule_order = 2 * int(label_order) + 4                  # the degree of volume_measure's rule
    fes2 = HCurl(mesh_model, order=int(model_order), dirichlet="wall")
    u2, v2 = fes2.TnT()
    with TaskManager():
        A = BilinearForm(u2 * v2 * dxq, symmetric=True).Assemble()
        Ainv = A.mat.Inverse(fes2.FreeDofs(), inverse="sparsecholesky")
    rule = IntegrationRule(TET, rule_order)
    pts = np.asarray(CF((x, y, z))(mesh_model.MapToAllElements(rule, VOL))).reshape(-1, 3)
    vals = np.zeros((len(gfs), len(pts), 3))               # outside the label mesh (geometry mismatch): 0
    for a in range(0, len(pts), 1 << 18):                  # chunks bound the memory of the located points
        located = ho.locate(gfs[0].space.mesh, pts[a:a + (1 << 18)])
        for k, g in enumerate(gfs):
            vals[k, a:a + (1 << 18)] = ho.evaluate(g, located)
    irs = IntegrationRuleSpace(mesh_model, order=rule_order // 2)
    if irs.ndof != mesh_model.ne * len(rule.points):
        raise RuntimeError("IntegrationRuleSpace does not carry the volume_measure rule")
    comps = [GridFunction(irs) for _ in range(3)]
    cf = CF(tuple(comps))
    U, n3 = np.zeros((fes2.ndof, len(gfs))), np.zeros(len(gfs))
    xk = A.mat.CreateColVector()
    for k in range(len(gfs)):
        for c in range(3):
            comps[c].vec.FV().NumPy()[:] = vals[k, :, c]
        with TaskManager():
            rhs = LinearForm(InnerProduct(cf, v2) * dxq).Assemble()
            xk.data = Ainv * rhs.vec
        U[:, k] = xk.FV().NumPy()
        n3[k] = Integrate(InnerProduct(cf, cf) * dxq, mesh_model)
    A2 = ho._tosp(A.mat)
    err = np.sqrt(np.clip(1.0 - np.einsum("ik,ik->k", U, A2 @ U) / n3, 0.0, None))
    dx2 = ho.volume_measure(model_order)
    with TaskManager():
        K2 = ho._tosp(BilinearForm(curl(u2) * curl(v2) * dx2, symmetric=True).Assemble().mat)
        M2 = ho._tosp(BilinearForm(u2 * v2 * dx2, symmetric=True).Assemble().mat)
    rq = np.einsum("ik,ik->k", U, K2 @ U) / np.einsum("ik,ik->k", U, M2 @ U)
    return U, err, rq, fes2


def _mesh_bytes(mesh):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.vol")
        mesh.ngmesh.Save(path)
        return open(path, "rb").read()


def _mesh_from_bytes(vol_bytes, shape, curve, deform):
    with tempfile.TemporaryDirectory() as tmp:
        vol, brep = os.path.join(tmp, "m.vol"), os.path.join(tmp, "s.brep")
        with open(vol, "wb") as fh:
            fh.write(vol_bytes)
        shape.WriteBrep(brep)
        return ho.load_curved_mesh(vol, brep, curve, deform)


def _check_reload(vol_bytes, shape, curve, deform, model_order, ndof):
    """The stored model mesh must rebuild the model space the fields were written in (FieldDataset and
    the UI rebuild it from the .vol + BREP): a mismatch fails the sample here, not the training later.
    Returns the reloaded mesh (the one the loader sees)."""
    from ngsolve import HCurl
    mesh = _mesh_from_bytes(vol_bytes, shape, curve, deform)
    n = HCurl(mesh, order=int(model_order), dirichlet="wall").ndof
    if n != ndof:
        raise RuntimeError(f"stored model mesh rebuilds HCurl(p{model_order}) with {n} DOFs, the fields have {ndof}")
    return mesh


def model_normalisation(mesh):
    """(center, scale) of the model's normalised geometry for a (reloaded) model mesh: what FieldDataset
    passes to highorder.field_operators."""
    from src.data.dataset_converter_3d import normalise
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()], dtype=np.float64)
    tets = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.ngmesh.Elements3D()], dtype=np.int64)
    _, center, scale = normalise(Xv, tets)
    return center, scale


def wall_feature_array(mesh, model_order):
    """qfeat_wall of a model mesh as loaded from the H5 (highorder.wall_features_for_mesh with the
    dataset's normalisation)."""
    center, scale = model_normalisation(mesh)
    return ho.wall_features_for_mesh(mesh, order=int(model_order), scale=scale, center=center)


def _beam_axis(shape_type, mesh, params=None):
    """(axis_dir, axis_point, L_acc) of mode_qoi for the family (src.qoi.operators)."""
    from src.qoi.operators import acceleration_length, axis_line, beam_axis
    X = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    d, p0 = axis_line(**beam_axis(shape_type, X, 1.0, np.zeros(3), params))
    return d, p0, acceleration_length(shape_type, params)


def label_sample(cad_path, h, shape_type, n_modes, deform=None, settings=None, sharp_edges=0, params=None):
    """All field labels of one geometry (the generator's CAD solid at label mesh size h; params: its
    geom_params, for the beam axis and the accelerating length).
    adaptive (default): adaptive refinement until the frequencies change < tol_f and the figures of merit
    < tol_q (docs/30).  sharp_edges > 0 (re-entrant edges the generator could not round): the surface
    peaks are singular there, so they are left out of the convergence check and stored as NaN."""
    from src.qoi.operators import QOI_LABELS
    s = {**DEFAULTS, **(settings or {})}
    K = n_modes
    t0 = time.perf_counter()
    adapt_attrs = {}
    if s["adaptive"]:
        from src.data_gen import adaptive as ad
        mesh, shape, cs = start_mesh(cad_path, h, deform, s, (s["adapt_cs"], s["curvaturesafety"]))
        t1 = time.perf_counter()
        ne0 = mesh.ne
        axis = _beam_axis(shape_type, mesh, params)
        snap = {}

        def keep_model_level(m, lab_, q_, rec):          # finest level within the model budget
            if m.ne <= s["model_max_elements"] or not snap:
                snap.update(vol=_mesh_bytes(m), ne=int(m.ne))
        remesh = ad.mesh_remesher(shape, s["maxh_factor"] * h, curve=s["curve"], deform=deform, curvaturesafety=cs,
                                  grading=s["grading"], min_jacobian_ratio=s["min_jacobian_ratio"])
        res = ad.adaptive_modes(mesh, K, order=s["label_order"], curve=s["curve"], deform=deform,
                                axis=axis, on_iter=keep_model_level, remesh=remesh,
                                settings={"tol_f": s["tol_f"], "tol_q": s["tol_q"], "max_iter": s["adapt_max_iter"],
                                          "max_ndof": s["max_ndof"], "theta": s["theta"], "peaks": sharp_edges == 0,
                                          "min_jacobian_ratio": s["min_jacobian_ratio"]})
        mesh, lab, q, last = res["mesh"], res["lab"], res["q"], res["hist"][-1]
        adapt_attrs = {"adapt_iters": len(res["hist"]), "adapt_stop": res["stop"], "n_tets_start": int(ne0),
                       "adapt_cs": float(cs), "est_rel_f": last["est_rel_f"], "est_rel_wall": last["est_rel_wall"],
                       "adapt_df": last.get("df", np.nan), "adapt_dq": last.get("dq", np.nan)}
    else:
        if s["label_order"] == s["model_order"]:          # low-fidelity labels: the adaptive path's start mesh
            mesh, shape, _ = start_mesh(cad_path, h, deform, s, (s["adapt_cs"], s["curvaturesafety"]))
        else:
            mesh, shape = curved_mesh(cad_path, h, deform, s)
        t1 = time.perf_counter()
        lab = ho.solve_modes(mesh, n_modes + 1, order=s["label_order"], max_ndof=s["max_ndof"])
        axis = _beam_axis(shape_type, mesh, params)
        q = ho.mode_qoi(mesh, lab["gfs"][:K], lab["f_hz"][:K], *axis)
    if sharp_edges:
        q = dict(q)
        for n in PEAK_QOI + ("E_pk_Vm", "B_pk_T"):
            if n in q:
                q[n] = np.full_like(np.asarray(q[n], float), np.nan)
    n_tets_label, model_mesh = int(mesh.ne), mesh
    if s["adaptive"] and snap["ne"] != mesh.ne:          # model mesh = a coarser level: L2 transfer
        model_mesh = _mesh_from_bytes(snap["vol"], shape, s["curve"], deform)
        U, err, rq, fes2 = transfer_to_model(lab["gfs"][:K], model_mesh, s["model_order"], s["label_order"])
    elif mesh.levels > 1:                                # a bisection hierarchy numbers its DOFs unlike its reload
        model_mesh = ho.single_level(mesh, shape, s["curve"], deform)
        U, err, rq, fes2 = transfer_to_model(lab["gfs"][:K], model_mesh, s["model_order"], s["label_order"])
    else:                                                # same mesh: exact nested projection
        U, err, rq, fes2 = project_to_model(mesh, {"fes": lab["fes"], "gfs": lab["gfs"][:K]}, s["model_order"],
                                            unit_norm=True)
    t2 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        brep = os.path.join(tmp, "s.brep")
        shape.WriteBrep(brep)
        brep_bytes = open(brep, "rb").read()
    vol_bytes = _mesh_bytes(model_mesh)
    reloaded = _check_reload(vol_bytes, shape, s["curve"], deform, s["model_order"], U.shape[0])
    wall_feat = wall_feature_array(reloaded, s["model_order"])         # the training loader reads it back
    return {"vol": np.frombuffer(vol_bytes, np.uint8), "brep": np.frombuffer(brep_bytes, np.uint8),
            "deform": deform, "freqs": lab["f_hz"][:K] / 1e9, "freq_next": float(lab["f_hz"][K] / 1e9),
            "u_model": U.astype(np.float32), "rq_model": rq, "proj_err": err, "qfeat_wall": wall_feat,
            "qoi": np.column_stack([np.asarray(q[n], float) for n in QOI_LABELS]),
            "qoi_names": list(QOI_LABELS), "axis_t": q["axis_t"], "axis_ez": q["axis_ez"].astype(np.float32),
            "attrs": {"labels": "field", "label_order": int(s["label_order"]), "model_order": int(s["model_order"]),
                      "curve": int(s["curve"]), "n_tets": int(model_mesh.ne), "n_vertices": int(model_mesh.nv),
                      "n_tets_label": n_tets_label,
                      "n_dof_label": int(lab["fes"].ndof), "n_dof_model": int(fes2.ndof),
                      "jacobian_ratio": float(ho.jacobian_ratio(model_mesh)),
                      "maxh": float(s["maxh_factor"] * h), "t_curved_mesh": t1 - t0, "t_labels": t2 - t1,
                      "sharp_edges": int(sharp_edges), **adapt_attrs,
                      "axis_dir": axis[0], "axis_point": axis[1],
                      "axis_dt": float(q["axis_dt"]), "L_acc": float(np.nan if axis[2] is None else axis[2])}}


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
    if res.get("qfeat_wall") is not None:
        write_wall_features(g, res["qfeat_wall"])
    g.create_dataset("axis_t", data=np.asarray(res["axis_t"], float))
    g.create_dataset("axis_ez", data=res["axis_ez"], **z)
    g.attrs["qoi_names"] = ",".join(res["qoi_names"])
    g.attrs["freq_next"] = float(res["freq_next"])
    for k, v in res["attrs"].items():
        g.attrs[k] = v


def write_wall_features(g, wall_feat):
    """Store qfeat_wall (docs/29): float32 [Nt, nq, 5], columns ho.WALL_FEATURES."""
    g.create_dataset("qfeat_wall", data=np.asarray(wall_feat, np.float32), compression="gzip", compression_opts=4)
    g.attrs["qfeat_columns"] = ",".join(ho.WALL_FEATURES)


def add_wall_features(path, overwrite=False, log=print):
    """Add qfeat_wall to every sample of a field-label H5 file that lacks it (rebuilds each model mesh
    once; no label is touched).  Returns the number of samples written."""
    import h5py
    n = 0
    with h5py.File(path, "r+") as f:
        for key in sorted(k for k in f.keys() if k.startswith("sample_")):
            g = f[key]
            if "qfeat_wall" in g:
                if not overwrite:
                    continue
                del g["qfeat_wall"]
            t = time.perf_counter()
            mesh, rec = read_field_group(g)
            write_wall_features(g, wall_feature_array(mesh, rec["model_order"]))
            n += 1
            log(f"{path}/{key}: qfeat_wall {g['qfeat_wall'].shape} ({time.perf_counter() - t:.1f} s)")
    return n


def export_step_files(path, step_dir, overwrite=False, log=print):
    """STEP copies [mm] of the stored CAD solids (the 'brep' of each sample) of a field-label H5 file, named like
    dataset_generator_3d --step_dir (<shape_type>_<id:08d>.step), for files generated without --step_dir.  Deformed
    samples are skipped as there: their labels belong to the warped mesh, not to the stored solid.
    Returns {'written', 'existing', 'deformed', 'no_cad'} counts."""
    import h5py
    n = dict(written=0, existing=0, deformed=0, no_cad=0)
    with h5py.File(path, "r") as f, tempfile.TemporaryDirectory() as tmp:
        for key in sorted(k for k in f.keys() if k.startswith("sample_")):
            g = f[key]
            if "brep" not in g:                          # N0 labels store no CAD
                n["no_cad"] += 1
                continue
            if "deform_om" in g:
                n["deformed"] += 1
                continue
            out = os.path.join(step_dir, f"{g.attrs['shape_type']}_{int(key.split('_', 1)[1]):08d}.step")
            if os.path.exists(out) and not overwrite:
                n["existing"] += 1
                continue
            brep = os.path.join(tmp, "s.brep")
            with open(brep, "wb") as fh:
                fh.write(np.asarray(g["brep"]).tobytes())
            os.makedirs(step_dir, exist_ok=True)
            ho.write_step(brep, out + ".part")
            os.replace(out + ".part", out)
            n["written"] += 1
    log(f"{path}: {n['written']} STEP written, {n['existing']} existing, {n['deformed']} deformed (no STEP), "
        f"{n['no_cad']} without CAD (N0 labels)")
    return n


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
           "shape_type": str(g.attrs.get("shape_type", "")), "wall_feat": None}
    if "qfeat_wall" in g:                      # optional: files written before docs/31 §3.7 do not have it
        cols = str(g.attrs.get("qfeat_columns", ""))
        if cols != ",".join(ho.WALL_FEATURES):
            raise ValueError(f"qfeat_wall columns '{cols}' differ from highorder.WALL_FEATURES; "
                             "scripts/add_field_features.py --overwrite rewrites them")
        rec["wall_feat"] = np.asarray(g["qfeat_wall"], np.float32)
    return mesh, rec
