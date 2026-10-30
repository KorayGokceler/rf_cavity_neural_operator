"""Multi-fidelity evaluation of a field3d checkpoint (docs/34): a model trained on LF labels, evaluated on the
stored geometries of an HF (or any field-label) H5 file, in the model space of the file ("native") and, because
the Ritz layer works in whatever HCurl space it is given, in a richer space of the same geometry:

    native   HCurl(model_order) on the stored model mesh; targets = the stored HF labels (p3 frequencies,
             projected p2 fields)
    p3       HCurl(3) on the same mesh; targets = a p3 solve on it (fields + frequencies)
    refine   HCurl(model_order) [or --order] on the uniformly refined mesh; targets = a solve there

For every geometry and space: model frequencies vs the stored HF label frequencies and vs the FE eigenvalues of
the evaluated space (the Ritz values must stay >= the latter), field rel-L2 vs the FE modes of the space, and
the cost of operators / forward.

    python scripts/eval_fidelity.py --ckpt training_logs/<exp>/last.ckpt --data 'hf.h5' --spaces native p3
        [--ids 0 1] [--threads 2] [--json out.json]
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.mesh_features import normalise  # noqa: E402
from src.data.field_dataset import field_collate, field_item, label_fidelity  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.data_gen.field_labels import read_field_group  # noqa: E402
from src.models.hcurl import mode_rel_l2  # noqa: E402
from src.training.lightning_module import CavityLightning  # noqa: E402

C0 = 299792458.0


def refined_mesh(g, mesh, curve, deform):
    """Uniform refinement of the stored curved mesh as a clean single-level mesh (ho.single_level)."""
    from netgen.occ import OCCGeometry
    with tempfile.TemporaryDirectory() as tmp:
        brep = os.path.join(tmp, "s.brep")
        with open(brep, "wb") as fh:
            fh.write(np.asarray(g["brep"]).tobytes())
        shape = OCCGeometry(brep).shape
        if len(shape.solids) == 1:
            shape = shape.solids[0]
        ho._name_faces(shape)
        mesh.Refine()                                     # `mesh` was read for this call only
        mesh.Curve(int(curve))
        return ho.single_level(mesh, shape, curve, deform)


def space_targets(mesh, order, K):
    """FE reference of the evaluated space: (f_GHz [K+1], T [n, K] DOF vectors, solve seconds)."""
    t0 = time.perf_counter()
    lab = ho.solve_modes(mesh, K + 1, order=order)
    T = np.column_stack([g.vec.FV().NumPy().copy() for g in lab["gfs"][:K]])
    return lab["f_hz"] / 1e9, T, time.perf_counter() - t0


def evaluate(model, g, space, order_override=None, fe_reference=True):
    mesh, rec = read_field_group(g)
    deform = tuple(np.asarray(g[k]) for k in ("deform_om", "deform_a", "deform_ph")) if "deform_om" in g else None
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()], dtype=np.float64)
    tets = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.ngmesh.Elements3D()], dtype=np.int64)
    _, center, scale = normalise(Xv, tets)
    K = model.num_field_modes
    f_hf = rec["freqs"][:K]
    order = int(order_override or rec["model_order"])
    mesh_e = mesh
    if space == "p3":
        order = 3
    elif space == "refine":
        mesh_e = refined_mesh(g, mesh, int(g.attrs["curve"]), deform)
    t0 = time.perf_counter()
    ops = ho.field_operators(mesh_e, order=order, scale=scale, center=center)
    t_ops = time.perf_counter() - t0
    if space == "native" and order == rec["model_order"]:
        T, f_next = rec["u_model"][:, :K], rec["freq_next"]
        f_fe, t_fe = None, 0.0
        if fe_reference:                                  # p2 floor: the FE eigenvalues of the model space
            f_fe, _, t_fe = space_targets(mesh_e, order, K)
    else:
        f_fe, T, t_fe = space_targets(mesh_e, order, K)
        f_next = f_fe[K]
    ops = {k: v for k, v in ops.items() if k not in ("bary", "X", "tets")}
    item = field_item(ops, T[:, :K], f_hf, None, str(g.attrs.get("shape_type", "")), check=False)
    item["FreqNext"] = torch.tensor(float(f_next), dtype=torch.float32)
    batch = field_collate([item])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(batch)
    t_fwd = time.perf_counter() - t0
    lam = out["eigenvalues"][0].double().numpy()
    f_pred = C0 * np.sqrt(lam) / (2 * np.pi * scale) / 1e9
    inside, split = model._clusters_3d(batch, 0, K)
    Fd, Tk = out["field"][0].double(), batch["Y_field"][0, :, :K].double()
    MT = torch.sparse.mm(batch["M"], Tk.reshape(-1, K))
    rl = mode_rel_l2(Fd, Tk, MT, inside).numpy()
    rl[split] = np.nan
    res = {"space": space, "order": order, "n_tets": int(mesh_e.ne), "n_dof": int(ops["n_dof"]),
           "f_pred": f_pred.tolist(), "f_hf": f_hf.tolist(),
           "rel_err_vs_hf": (f_pred / f_hf - 1).tolist(), "field_rel_l2": rl.tolist(),
           "t_ops": t_ops, "t_forward": t_fwd, "t_fe_ref": t_fe}
    if f_fe is not None:
        res["f_fe"] = f_fe[:K].tolist()
        res["rel_err_vs_fe"] = (f_pred / f_fe[:K] - 1).tolist()
        res["fe_floor_vs_hf"] = (f_fe[:K] / f_hf - 1).tolist()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="field-label H5 file")
    ap.add_argument("--spaces", nargs="+", default=["native"], choices=["native", "p3", "refine"])
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="sample indices in the file (default: all)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no_fe_reference", action="store_true", help="native space: skip the p2 FE solve")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import ngsolve
    ngsolve.SetNumThreads(args.threads)
    model = CavityLightning.load_from_checkpoint(args.ckpt, map_location="cpu", strict=False).eval()
    model.freq_stats = {"mean": 0.0, "std": 1.0}          # Y_freq / FreqNext below are raw GHz
    rows = []
    with h5py.File(args.data, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        for i, key in enumerate(keys):
            if args.ids is not None and i not in args.ids:
                continue
            for space in args.spaces:
                r = evaluate(model, f[key], space, fe_reference=not args.no_fe_reference)
                r.update(key=key, shape_type=str(f[key].attrs.get("shape_type", "")),
                         fidelity="HF" if label_fidelity(f[key].attrs) else "LF")
                rows.append(r)
                fmt = lambda a: " ".join(f"{100 * v:+7.3f}" for v in a)       # noqa: E731
                print(f"{key} {r['shape_type']:<14} {space:<7} p{r['order']} {r['n_tets']:>6} tets {r['n_dof']:>7} dof | "
                      f"f err vs HF [%] {fmt(r['rel_err_vs_hf'])}"
                      + (f" | vs FE of space {fmt(r['rel_err_vs_fe'])}" if "rel_err_vs_fe" in r else "")
                      + f" | field rel-L2 {' '.join(f'{v:5.3f}' for v in r['field_rel_l2'])}"
                      f" | ops {r['t_ops']:.1f}s fwd {r['t_forward']:.1f}s", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
