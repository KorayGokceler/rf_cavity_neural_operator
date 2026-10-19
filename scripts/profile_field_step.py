"""Stage timings of the field3d model on one stored geometry (docs/31 §3): data preparation (mesh rebuild,
field_operators, field_item + collate), inference without gradients and one training step (forward + loss,
backward), the Kp CG share of each, the dense FLOPs of the network and the peak memory.  CPU or CUDA.

    python scripts/profile_field_step.py <field-label H5> [--sample 1] [--device cuda] [--threads 4]
        [--reps 2] [--config configs/field_3d.yaml] [--cprofile] [--json out.json]

The model has the sizes of --config; the weights are random (the cost does not depend on them).  Every
timing is wall clock without a profiler; --cprofile adds a separate profiled run of field_operators.
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
from src.data.dataset_converter_3d import normalise  # noqa: E402
from src.data.field_dataset import field_collate, field_item  # noqa: E402
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

    def __exit__(self, *exc):
        hcurl._pcg = self.orig


def linear_flops(model, batch):
    """Dense multiply-add FLOPs (2·in·out per row) of every nn.Linear in one forward."""
    total = [0]
    hooks = [m.register_forward_hook(lambda m, i, o: total.__setitem__(
        0, total[0] + 2 * i[0].numel() * m.out_features))
        for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()
    return total[0]


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_model(cfg_path, val_dim, n_modes):
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    mc, tc = cfg["model"], cfg["training"]
    lm = CavityLightning(
        val_dim=val_dim, num_field_modes=min(int(mc["num_field_modes"]), n_modes), hidden_dim=mc["embed_dim"],
        n_heads=mc["n_heads"], n_basis=mc["n_basis"], rff_dim=mc["rff_dim"],
        rff_length_scale=mc["rff_length_scale"], dropout=mc["dropout"], eigenspace_kwargs=mc["eigenspace"],
        freq_weight=tc["freq_weight"], span_weight=tc["span_weight"], selfsup_weight=tc["selfsup_weight"],
        ortho_weight=tc["ortho_weight"], span_norm=tc["span_norm"], span_root=tc["span_root"],
        span_ridge=tc["span_ridge"], selfsup_form=tc["selfsup_form"], model_type="field3d")
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
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import ngsolve
    torch.set_num_threads(a.threads)
    ngsolve.SetNumThreads(a.threads)
    dev = torch.device(a.device)
    clock = Clock(dev)
    res = {"device": str(dev), "threads": a.threads}

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

    t = clock()
    ops = ho.field_operators(mesh, order=rec["model_order"], scale=scale, center=center)
    res["t_operators"] = clock() - t
    res["rss_after_operators_gb"] = peak_rss_gb()
    if a.cprofile:
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

    K = int(rec["u_model"].shape[1])
    t = clock()
    item = field_item(ops, rec["u_model"], rec["freqs"][:K], None, rec.get("shape_type", ""), check=False)
    t_item = clock()
    batch = field_collate([item])
    res["t_item"], res["t_collate"] = t_item - t, clock() - t_item
    t = clock()
    batch = to_device(batch, dev)
    res["t_to_device"] = clock() - t

    lm = build_model(a.config, batch["Input_funcs"].shape[-1], K)
    lm.to(dev)
    model = lm.model
    res.update(n_tets=int(mesh.ne), n_tokens=int(batch["X"].shape[1]), n_dof=int(ops["n_dof"]),
               n_pot=int(ops["n_pot"]), n_modes=lm.num_field_modes,
               n_params=int(sum(p.numel() for p in model.parameters())),
               nnz={k: int(ops[k].nnz) for k in ("K", "M", "G", "Kp", "C")})
    res["gflop_forward"] = linear_flops(model, batch) / 1e9

    def mem_reset():
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def mem_peak():
        return torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else peak_rss_gb()

    with CGMeter(clock) as cg:
        model.eval()
        res["infer"] = []
        for _ in range(a.reps):
            cg.reset()
            mem_reset()
            t = clock()
            with torch.no_grad():
                model(batch)
            res["infer"].append({"t": clock() - t, "t_cg": cg.t, "cg_iters": cg.iters, "mem_gb": mem_peak()})
        model.train()
        res["train"] = []
        for _ in range(a.reps):
            lm.zero_grad(set_to_none=True)
            cg.reset()
            mem_reset()
            t = clock()
            loss, _, _ = lm._compute_loss_3d(batch, "train")
            t1 = clock()
            t_cg_fwd = cg.t
            loss.backward()
            res["train"].append({"t_forward": t1 - t, "t_backward": clock() - t1, "t_cg_forward": t_cg_fwd,
                                 "t_cg_backward": cg.t - t_cg_fwd, "cg_iters": cg.iters, "mem_gb": mem_peak()})

    mem = "GPU allocated" if dev.type == "cuda" else "process RSS"
    print(f"{key}: {res['n_tets']} tets, {res['n_tokens']} tokens, {res['n_dof']} DOF, {res['n_pot']} potentials, "
          f"{res['n_modes']} modes; {res['n_params'] / 1e6:.3f} M parameters; {res['device']}, {a.threads} threads")
    print(f"nnz: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in res["nnz"].items()))
    print(f"data   mesh (generator, from CAD) {res['t_mesh_generator']:6.2f} s | rebuild from .vol {res['t_mesh_rebuild']:.2f} s")
    print(f"       field_operators            {res['t_operators']:6.2f} s | peak RSS {res['rss_after_operators_gb']:.2f} GB")
    print(f"       field_item / collate       {res['t_item']:6.2f} / {res['t_collate']:.2f} s | to {res['device']} {res['t_to_device']:.2f} s")
    for i, r in enumerate(res["infer"]):
        print(f"infer  #{i} forward {r['t']:6.2f} s (CG {r['t_cg']:.2f} s, iterations {r['cg_iters']}) | "
              f"peak {mem} {r['mem_gb']:.2f} GB")
    for i, r in enumerate(res["train"]):
        print(f"train  #{i} forward+loss {r['t_forward']:6.2f} s (CG {r['t_cg_forward']:.2f}), backward "
              f"{r['t_backward']:6.2f} s (CG {r['t_cg_backward']:.2f}), iterations {r['cg_iters']} | "
              f"peak {mem} {r['mem_gb']:.2f} GB")
    print(f"network dense FLOPs, forward: {res['gflop_forward']:.1f} GFLOP")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
