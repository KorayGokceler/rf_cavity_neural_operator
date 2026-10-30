"""Geometries and reference labels for calibrating against CST (reports/CST eigenmode çözücüsü analizi.md, Ek 3).

Each case goes through the field-label generator unchanged (dataset_generator_3d: CAD, curved netgen mesh,
adaptive p3 labels, figures of merit) and writes
    <out>/<case>.step      the CAD solid [mm] that CST should import,
    <out>/labels.h5        the labels (one group per case, as the generator writes them).
Cases:
    calib_box       PEC box 100 × 80 × 60 mm (analytic; the hexahedral FIT mesh fits it exactly)
    calib_pillbox   pillbox R = 40 mm, L = 50 mm (analytic; the label solver's validation case)
    pillbox_pipes   pillbox with pipes, test geometry 24 of the multi-fidelity experiments (docs/34 §8)
    tesla_cell      one TESLA mid-cell (both half-cells = the mid-cell anchor) with beam pipes

    python scripts/cst_calibration_set.py --out <dir> [--cases ...] [--max_ndof 600000]
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import torch  # noqa: E402,F401  (torch before ngsolve)

from src.data_gen import cavity_shapes as cs  # noqa: E402
from src.data_gen import dataset_generator_3d as gen  # noqa: E402

TESLA_PIPE = 0.080                                       # beam-pipe length on each side [m]
TESLA = dict(cs.ANCHORS["tesla"], n_cells=1.0, Lpipe=TESLA_PIPE,
             e1_L=cs.ANCHORS["tesla"]["L"], e2_L=cs.ANCHORS["tesla"]["L"])   # L_acc = one cell (2 L)

CASES = {   # case: (generator arguments, sample id)
    "calib_box": (["--mode", "calibration"], 0),
    "calib_pillbox": (["--mode", "calibration"], 1),
    "pillbox_pipes": (["--families", "pillbox_pipes", "--seed", "0", "--sampling", "sobol",
                       "--deform_prob", "0", "--min_fillet", "0.12"], 24),
    "tesla_cell": (["--families", "elliptical", "--deform_prob", "0"], 0),
}


def run_case(name, out, common):
    argv, s_id = CASES[name]
    step_dir = os.path.join(out, "_step", name)
    old = gen.ARGS, cs.MIN_FILLET, gen.BUILDERS["elliptical"]
    gen.ARGS = gen.parse_args(["--step_dir", step_dir, *common, *argv])
    gen._init_worker(gen.ARGS)
    if name == "tesla_cell":                              # fixed TESLA parameters instead of a random draw
        gen.BUILDERS["elliptical"] = lambda occ, rng: cs.build_elliptical(occ, None, params=dict(TESLA))
    try:
        res = gen.generate_sample_data(s_id)
        if res is None:
            raise RuntimeError(f"{name}: the generator returned no sample")
        with h5py.File(os.path.join(out, "labels.h5"), "a") as f:
            if name in f:
                del f[name]
            sub = f.create_group(name)                     # write_sample names the group sample_<id>
            gen.write_sample(sub, res)
            sub.attrs["metadata"] = json.dumps(gen.file_metadata(gen.ARGS), default=str)
        (step,) = [os.path.join(step_dir, s) for s in os.listdir(step_dir) if s.endswith(".step")]
        os.replace(step, os.path.join(out, f"{name}.step"))
        os.rmdir(step_dir)
        if not os.listdir(os.path.dirname(step_dir)):
            os.rmdir(os.path.dirname(step_dir))
    finally:
        gen.ARGS, cs.MIN_FILLET, gen.BUILDERS["elliptical"] = old
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--max_ndof", type=int, default=600_000, help="adaptive label budget (~8 GB at 600k)")
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    common = ["--max_ndof", str(a.max_ndof), "--n_eigen_modes", str(a.n_modes), "--n_workers", "1",
              "--threads", str(a.threads)]
    for name in a.cases:
        res = run_case(name, a.out, common)
        print(f"{name}: {res['shape_type']}, f [GHz] = " + " ".join(f"{x:.6f}" for x in res["freqs"]),
              flush=True)


if __name__ == "__main__":
    main()
