"""Stage timings of the field3d model on one stored geometry (docs/31 §3): data preparation (mesh rebuild,
field_operators, field_item + collate), inference without gradients and one training step (forward + loss,
backward), the Kp CG share of each, the dense FLOPs of the network and the peak memory.  CPU or CUDA.

    python scripts/profile_field_step.py <field-label H5> [--sample 1] [--device cuda] [--threads 4]
        [--reps 2] [--config configs/field_3d.yaml] [--grad_checkpoint] [--no_projection] [--cprofile]
        [--json out.json]

The model has the sizes of --config; the weights are random (the cost does not depend on them).  Every
timing is wall clock without a profiler; --cprofile adds a separate profiled run of field_operators.

--no_projection: the training step of training.projection: false (docs/38): operators and batch WITHOUT G / Kp,
trunk + unprojected Grams + span / ortho loss, no Kp CG, no Rayleigh–Ritz (selfsup_weight = freq_weight = 0).  The
training steps run first, on the reduced batch (the peak memory is theirs); the inference pass (with the
projection, as a test / prediction step; the validation step of a projection-free run needs no G / Kp either) follows on the full operators.
"""
import argparse
import json
import os
import resource
import sys
import time

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.models.hcurl as hcurl  # noqa: E402
from src.data.mesh_features import normalise  # noqa: E402
from src.data import matfree  # noqa: E402
from src.data.field_dataset import field_collate, field_collate_mf, field_item, field_item_matfree  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.data_gen.field_labels import read_field_group  # noqa: E402
from src.training.lightning_module import CavityLightning  # noqa: E402


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


class Clock:
    """Wall-clock stages; CUDA work is synchronised before each reading."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"

    def __call__(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


class OpMeter:
    """Wraps matfree.element_matrices and matfree.psi_map (what StepOperators calls inside the step): time and calls."""

    def __init__(self, clock):
        self.clock, self.orig = clock, (matfree.element_matrices, matfree.psi_map)
        self.reset()

    def reset(self):
        self.t, self.calls = 0.0, 0

    def __enter__(self):
        def timed(fn):
            def wrapped(*a, **k):
                t = self.clock()
                out = fn(*a, **k)
                self.t += self.clock() - t
                self.calls += 1
                return out
            return wrapped
        matfree.element_matrices, matfree.psi_map = (timed(f) for f in self.orig)
        return self

    def __exit__(self, *_exc):
        matfree.element_matrices, matfree.psi_map = self.orig


class CGMeter:
    """Wraps hcurl._pcg (the Kp solve of KpSolve, forward and backward): time and iterations."""

    def __init__(self, clock):
        self.clock, self.orig = clock, hcurl._pcg
        self.reset()

    def reset(self):
        self.t, self.iters, self.calls = 0.0, [], 0

    def __enter__(self):
        def timed(*a, **k):
            t = self.clock()
            x, it = self.orig(*a, **k)
            self.t += self.clock() - t
            self.iters.append(int(it))
            self.calls += 1
            return x, it
        hcurl._pcg = timed
        return self

    def __exit__(self, *_exc):
        hcurl._pcg = self.orig


def linear_flops(model, batch, forward=None):
    """Dense multiply-add FLOPs (2·in·out per row) of every nn.Linear in one forward (default model(batch))."""
    total = [0]
    hooks = [m.register_forward_hook(lambda m, i, o: total.__setitem__(
        0, total[0] + 2 * i[0].numel() * m.out_features))
        for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        (forward or model)(batch)
    for h in hooks:
        h.remove()
    return total[0]


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_model(cfg_path, val_dim, n_modes, projection=True, **eigenspace):
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    mc, tc = cfg["model"], cfg["training"]
    mc["eigenspace"] = {**mc["eigenspace"], **eigenspace}
    lm = CavityLightning(
        val_dim=val_dim, num_field_modes=min(int(mc["num_field_modes"]), n_modes), hidden_dim=mc["embed_dim"],
        n_heads=mc["n_heads"], n_basis=mc["n_basis"], rff_dim=mc["rff_dim"],
        rff_length_scale=mc["rff_length_scale"], dropout=mc["dropout"], eigenspace_kwargs=mc["eigenspace"],
        freq_weight=tc["freq_weight"] if projection else 0.0, span_weight=tc["span_weight"],
        selfsup_weight=tc["selfsup_weight"] if projection else 0.0,
        ortho_weight=tc["ortho_weight"], span_norm=tc["span_norm"], span_root=tc["span_root"],
        span_ridge=tc["span_ridge"], selfsup_form=tc["selfsup_form"], model_type="field3d", projection=projection)
    lm.freq_stats = {"mean": 3.0, "std": 1.0}
    lm.log = lambda *a, **k: None
    return lm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5")
    ap.add_argument("--sample", default="1", help="group index in sorted order, or a group name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--config", default="configs/field_3d.yaml")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true", help="model.eigenspace.grad_checkpoint = true")
    ap.add_argument("--no_projection", action="store_true",
                    help="measure the projection-free training step (training.projection: false); the inference "
                         "pass still projects")
    ap.add_argument("--matfree", action="store_true",
                    help="the matrix-free projection-free step (dataset.operators: matfree, docs/40 §10); implies --no_projection")
    ap.add_argument("--matfree_chunk", type=int, default=2048, help="training.matfree_chunk (elements per chunk)")
    ap.add_argument("--no_stored_features", action="store_true",
                    help="compute the token wall-distance / torsion features instead of reading qfeat_wall")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.matfree:
        a.no_projection = True

    import ngsolve
    torch.set_num_threads(a.threads)
    ngsolve.SetNumThreads(a.threads)
    dev = torch.device(a.device)
    clock = Clock(dev)
    res = {"device": str(dev), "threads": a.threads, "grad_checkpoint": a.grad_checkpoint,
           "projection": not a.no_projection, "operators": "matfree" if a.matfree else "assembled"}

    with h5py.File(a.h5, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        key = keys[int(a.sample)] if a.sample.isdigit() else a.sample
        t = clock()
        mesh, rec = read_field_group(f[key])
        res["t_mesh_rebuild"] = clock() - t
        attrs = dict(f[key].attrs)
    res["t_mesh_generator"] = float(attrs.get("t_mesh", np.nan)) + float(attrs.get("t_curved_mesh", np.nan))
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)

    wall = None if a.no_stored_features else rec.get("wall_feat")
    res["stored_features"] = wall is not None
    K = int(rec["u_model"].shape[1])

    def prepare(potential):
        """(ops, batch on the device, timings) with or without the kernel potentials G, Kp."""
        t = clock()
        ops = ho.field_operators(mesh, order=rec["model_order"], scale=scale, center=center, wall_feat=wall,
                                 potential=potential)
        t_ops = clock() - t
        t = clock()
        item = field_item(ops, rec["u_model"], rec["freqs"][:K], None, rec.get("shape_type", ""), check=False,
                          with_ct=True)
        t_item = clock()
        batch = field_collate([item])
        t_col = clock() - t_item
        t_item -= t
        t = clock()
        batch = to_device(batch, dev)
        return ops, batch, {"t_operators": t_ops, "t_item": t_item, "t_collate": t_col, "t_to_device": clock() - t}

    def prepare_matfree():
        """(info, batch on the device, timings): the matrix-free item without any global matrix."""
        t = clock()
        mrec = matfree.build_record(mesh, rec["model_order"], scale, center)
        t_rec = clock() - t
        t = clock()
        _, qw, qfeat = ho.token_inputs(mesh, rec["model_order"], scale, center, wall)
        t_tok = clock() - t
        t = clock()
        bundle = {"mf": mrec, "qfeat": qfeat.reshape(mesh.ne * matfree.N_TOKEN, -1).astype(np.float32),
                  "qw": qw.reshape(-1).astype(np.float32)}
        item = field_item_matfree(bundle, rec["u_model"], rec["freqs"][:K], None, rec.get("shape_type", ""))
        t_item = clock()
        batch = field_collate_mf([item])
        t_col = clock() - t_item
        t_item -= t
        t = clock()
        batch = to_device(batch, dev)
        batch["MF"] = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in r.items()} for r in batch["MF"]]
        info = {"n_dof": int(mrec["n_dof"]), "record_MB": matfree.record_nbytes(mrec) / 1e6,
                "item_MB": sum(v.numel() * v.element_size() for v in item.values() if torch.is_tensor(v)) / 1e6
                + sum(v.numel() * v.element_size() for v in item["MF"].values() if torch.is_tensor(v)) / 1e6}
        return info, batch, {"t_operators": t_rec + t_tok, "t_record": t_rec, "t_tokens": t_tok, "t_item": t_item,
                             "t_collate": t_col, "t_to_device": clock() - t}

    if a.matfree:
        mf_info, batch, tm = prepare_matfree()
        ops = {"n_dof": mf_info["n_dof"], "n_pot": None}
        res.update(tm)
        res["record_MB"], res["item_MB"] = mf_info["record_MB"], mf_info["item_MB"]
    else:
        ops, batch, tm = prepare(not a.no_projection)
        res.update(tm)
    res["rss_after_operators_gb"] = peak_rss_gb()
    if a.cprofile and not a.matfree:
        import cProfile
        import io
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        ho.field_operators(mesh, order=rec["model_order"], scale=scale, center=center)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(16)
        print("field_operators under cProfile, by own time:")
        for line in s.getvalue().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[1].replace(".", "").isdigit() and float(parts[1]) >= 0.05:
                print(f"   {float(parts[1]):6.2f} s  {parts[5][:100]}")

    lm = build_model(a.config, batch["Input_funcs"].shape[-1], K, projection=not a.no_projection,
                     **({"grad_checkpoint": True} if a.grad_checkpoint else {}))
    lm.to(dev)
    lm.matfree_chunk = a.matfree_chunk
    model = lm.model
    res.update(n_tets=int(mesh.ne), n_tokens=int(batch["X"].shape[1]), n_dof=int(ops["n_dof"]),
               n_pot=None if ops["n_pot"] is None else int(ops["n_pot"]), n_modes=lm.num_field_modes,
               n_params=int(sum(p.numel() for p in model.parameters())),
               nnz={} if a.matfree else {k: int(ops[k].nnz) for k in ("K", "M", "G", "Kp", "C") if k in ops})
    res["gflop_forward"] = linear_flops(model, batch, model.forward_basis if a.no_projection else None) / 1e9

    def mem_reset():
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def mem_peak():
        return torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else peak_rss_gb()

    def infer_pass(cg, batch):
        model.eval()
        out = []
        for _ in range(a.reps):
            cg.reset()
            mem_reset()
            t = clock()
            with torch.no_grad():
                model(batch)
            out.append({"t": clock() - t, "t_cg": cg.t, "cg_iters": cg.iters, "mem_gb": mem_peak()})
        return out

    def train_pass(cg, batch, om=None):
        model.train()
        out = []
        for _ in range(a.reps):
            lm.zero_grad(set_to_none=True)
            cg.reset()
            if om is not None:
                om.reset()
            mem_reset()
            t = clock()
            if a.no_projection:
                loss = lm._compute_loss_3d_plain(batch, "train")
            else:
                loss, _, _ = lm._compute_loss_3d(batch, "train")
            t1 = clock()
            t_cg_fwd = cg.t
            loss.backward()
            out.append({"t_forward": t1 - t, "t_backward": clock() - t1, "t_cg_forward": t_cg_fwd,
                        "t_cg_backward": cg.t - t_cg_fwd, "cg_iters": cg.iters, "calls_cg": cg.calls,
                        "mem_gb": mem_peak(), **({"t_step_operators": om.t} if om is not None else {})})
        return out

    with CGMeter(clock) as cg, OpMeter(clock) as om:
        if a.matfree:               # no assembled operator exists: no inference pass
            res["train"] = train_pass(cg, batch, om)
            res["infer"] = []
        elif a.no_projection:         # reduced batch first (its peak memory), then the projected inference pass
            res["train"] = train_pass(cg, batch)
            del batch, ops
            ops, batch, tm = prepare(True)
            res.update({f"{k}_full": v for k, v in tm.items()})
            res["infer"] = infer_pass(cg, batch)
        else:
            res["infer"] = infer_pass(cg, batch)
            res["train"] = train_pass(cg, batch)

    mem = "GPU allocated" if dev.type == "cuda" else "process RSS"
    print(f"{key}: {res['n_tets']} tets, {res['n_tokens']} tokens, {res['n_dof']} DOF, "
          f"{'no' if res['n_pot'] is None else res['n_pot']} potentials, "
          f"{res['n_modes']} modes; {res['n_params'] / 1e6:.3f} M parameters; {res['device']}, {a.threads} threads; "
          f"{'MATRIX-FREE projection-free training step' if a.matfree else 'PROJECTION-FREE training step' if a.no_projection else 'projected training step'}")
    if not a.matfree:
        print(f"nnz: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in res["nnz"].items()))
    print(f"data   mesh (generator, from CAD) {res['t_mesh_generator']:6.2f} s | rebuild from .vol {res['t_mesh_rebuild']:.2f} s")
    if a.matfree:
        print(f"       matfree record {res['t_record']:.2f} s + token inputs {res['t_tokens']:.2f} s (features "
              f"{'read from the H5' if res['stored_features'] else 'computed'}) | record {res['record_MB']:.1f} MB, item "
              f"{res['item_MB']:.1f} MB | peak RSS {res['rss_after_operators_gb']:.2f} GB")
    else:
        print(f"       field_operators            {res['t_operators']:6.2f} s (token features "
              f"{'read from the H5' if res['stored_features'] else 'computed'}) | peak RSS {res['rss_after_operators_gb']:.2f} GB")
    print(f"       field_item / collate       {res['t_item']:6.2f} / {res['t_collate']:.2f} s | to {res['device']} {res['t_to_device']:.2f} s")
    if a.no_projection and not a.matfree:
        print(f"       full operators (G, Kp) for the inference pass: field_operators {res['t_operators_full']:.2f} s, "
              f"collate {res['t_collate_full']:.2f} s")
    for i, r in enumerate(res["infer"]):
        print(f"infer  #{i} forward {r['t']:6.2f} s (CG {r['t_cg']:.2f} s, iterations {r['cg_iters']}) | "
              f"peak {mem} {r['mem_gb']:.2f} GB")
    for i, r in enumerate(res["train"]):
        print(f"train  #{i} forward+loss {r['t_forward']:6.2f} s (CG {r['t_cg_forward']:.2f}), backward "
              f"{r['t_backward']:6.2f} s (CG {r['t_cg_backward']:.2f}), CG calls {r['calls_cg']}, iterations {r['cg_iters']} | "
              f"peak {mem} {r['mem_gb']:.2f} GB"
              + (f" | on-device element matrices + psi maps {r['t_step_operators']:.2f} s" if "t_step_operators" in r else ""))
    print(f"network dense FLOPs, forward: {res['gflop_forward']:.1f} GFLOP")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
