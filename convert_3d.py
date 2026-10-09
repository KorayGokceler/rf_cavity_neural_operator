"""CLI: 3D H5 (src/data_gen/dataset_generator_3d.py) → PKL for the 3D Maxwell model.
E-field N0 DOFs ('e_edges'); metadata['field'] = 'E'.  Contract: src/data/dataset_converter_3d.py, docs/19_3D_DATA_PIPELINE.md.
Cavity QoI labels (Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc; docs/24_CAVITY_QOI.md) are stored per
sample unless --no_qoi; `--add_qoi IN.pkl` back-fills them into an existing PKL."""
import argparse
import os
import pickle

from src.data.dataset_converter_3d import RFCavity3DConverter, attach_qoi_labels

DEFAULT_OUT = "data/rf_cavity_3d.pkl"


def add_qoi(in_path, out_path=None):
    """Back-fill samples[i]['qoi'] + metadata['qoi'] into an existing PKL (in place by default)."""
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    attach_qoi_labels(data)
    out_path = out_path or in_path
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    print(f"QoI labels added: {out_path} ({len(data['samples'])} samples, "
          f"{data['metadata']['qoi']['n_failed']} geometries failed)")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert a 3D N0 (E- or H-field) H5 dataset to the training PKL.")
    p.add_argument("--h5_filepath", type=str, nargs="+", default=["rf_cavity_3d_dataset.h5"],
                   help="Input H5 file(s); shards from dataset_generator_3d.py --start_id (disjoint sample ids).")
    p.add_argument("--output_path", type=str, default=None, help=f"Output PKL file (default {DEFAULT_OUT}; "
                   "with --add_qoi: the input PKL, overwritten).")
    p.add_argument("--modes", type=int, nargs="+", default=None, help="Mode indices to include (default: all).")
    p.add_argument("--max_samples", type=int, default=None, help="Limit the number of geometries.")
    p.add_argument("--freq_mean", type=float, default=None, help="Manual override for the frequency mean.")
    p.add_argument("--freq_std", type=float, default=None, help="Manual override for the frequency std.")
    p.add_argument("--no_rayleigh_check", action="store_true", help="Skip the per-mode Rayleigh/frequency check.")
    p.add_argument("--no_operators", action="store_true",
                   help="Do not store M/K/G/Kp (rebuild with dataset_converter_3d.geometry_operators(X, tets, "
                        "field=metadata['field'])).")
    p.add_argument("--no_qoi", action="store_true",
                   help="Do not compute the per-sample cavity QoI labels (samples[i]['qoi'], metadata['qoi']).")
    p.add_argument("--add_qoi", type=str, default=None, metavar="PKL",
                   help="Only back-fill QoI labels into this existing PKL (no H5 conversion).")
    args = p.parse_args(argv)
    if args.add_qoi:
        return add_qoi(args.add_qoi, args.output_path)
    out = args.output_path or DEFAULT_OUT
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return RFCavity3DConverter(args.h5_filepath).convert_dataset(
        out, mode_indices=args.modes, max_samples=args.max_samples,
        freq_mean=args.freq_mean, freq_std=args.freq_std, check_rayleigh=not args.no_rayleigh_check,
        store_operators=not args.no_operators, compute_qoi=not args.no_qoi)


if __name__ == "__main__":
    main()
