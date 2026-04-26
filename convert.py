import os
import argparse
from src.data.dataset_converter import RFCavityToGNOT

def main():
    parser = argparse.ArgumentParser(description="Convert H5 dataset to GNOT proper format.")
    parser.add_argument("--h5_filepath", type=str, default="rf_cavity_1000_dataset.h5", help="Path to input H5 file.")
    parser.add_argument("--output_path", type=str, default="data/gnot_dataset.pkl", help="Path to output file.")
    parser.add_argument("--output_format", type=str, default="pkl", choices=["pkl", "h5"], help="Output format (pkl or h5).")
    parser.add_argument("--modes", type=int, nargs='+', default=[0, 1, 2], help="Mode indices to include (e.g., 0 1 or 1).")
    
    # Frequency stats override
    parser.add_argument("--freq_mean", type=float, default=None, help="Manual override for frequency mean.")
    parser.add_argument("--freq_std", type=float, default=None, help="Manual override for frequency std.")
    
    args, _ = parser.parse_known_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting conversion from {args.h5_filepath} to {args.output_path} (Modes: {args.modes}, Format: {args.output_format})...")
    converter = RFCavityToGNOT(args.h5_filepath)
    converter.convert_dataset(
        output_filepath=args.output_path,
        mode_indices=args.modes,
        max_samples=None,
        format=args.output_format,
        freq_mean=args.freq_mean,
        freq_std=args.freq_std
    )
    print("Conversion finished.")

if __name__ == "__main__":
    main()
