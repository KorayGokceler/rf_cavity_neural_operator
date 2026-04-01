import os
import argparse
from src.data.dataset_converter import RFCavityToGNOT

def main():
    parser = argparse.ArgumentParser(description="Convert H5 dataset to GNOT PKL.")
    parser.add_argument("--h5_filepath", type=str, default="rf_cavity_1000_dataset.h5", help="Path to input H5 file.")
    parser.add_argument("--output_pkl", type=str, default="data/gnot_dataset.pkl", help="Path to output PKL file.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_pkl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting conversion from {args.h5_filepath} to {args.output_pkl}...")
    converter = RFCavityToGNOT(args.h5_filepath)
    converter.convert_dataset(
        output_filepath=args.output_pkl,
        mode_indices=[0, 1, 2],
        max_samples=None,
    )
    print("Conversion finished.")

if __name__ == "__main__":
    main()
