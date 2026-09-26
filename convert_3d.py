"""CLI: 3D H5 (src/data_gen/dataset_generator_3d.py) → PKL for the 3D Maxwell model.
Contract: src/data/dataset_converter_3d.py, docs/19_3D_DATA_PIPELINE.md."""
import argparse
import os

from src.data.dataset_converter_3d import RFCavity3DConverter


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert a 3D N0 (H-field) H5 dataset to the training PKL.")
    p.add_argument("--h5_filepath", type=str, default="rf_cavity_3d_dataset.h5", help="Input H5 file.")
    p.add_argument("--output_path", type=str, default="data/rf_cavity_3d.pkl", help="Output PKL file.")
    p.add_argument("--modes", type=int, nargs="+", default=None, help="Mode indices to include (default: all).")
    p.add_argument("--max_samples", type=int, default=None, help="Limit the number of geometries.")
    p.add_argument("--freq_mean", type=float, default=None, help="Manual override for the frequency mean.")
    p.add_argument("--freq_std", type=float, default=None, help="Manual override for the frequency std.")
    p.add_argument("--no_rayleigh_check", action="store_true", help="Skip the per-mode Rayleigh/frequency check.")
    p.add_argument("--no_operators", action="store_true",
                   help="Do not store M/K/G/Kp (rebuild with dataset_converter_3d.geometry_operators(X, tets)).")
    args = p.parse_args(argv)
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    RFCavity3DConverter(args.h5_filepath).convert_dataset(
        args.output_path, mode_indices=args.modes, max_samples=args.max_samples,
        freq_mean=args.freq_mean, freq_std=args.freq_std, check_rayleigh=not args.no_rayleigh_check,
        store_operators=not args.no_operators)


if __name__ == "__main__":
    main()
