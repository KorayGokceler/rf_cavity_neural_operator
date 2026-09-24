"""Veri doğrulama scripti — her modun Y_field'ı gerçekten farklı mı kontrol et.

Kullanım:
    python scripts/check_mode_data.py --data_path data/gnot_dataset_5k.pkl [--n 50]
"""
import argparse
import pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", default="data/gnot_dataset.pkl", help="Converted PKL dataset")
    ap.add_argument("--n", type=int, default=50, help="Samples per mode used for statistics")
    args = ap.parse_args()

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]

    # Mod başına istatistikler (Theta[0] = mode slot index written by the converter)
    modes = sorted({int(s["Theta"][0]) for s in samples})
    for mode in modes:
        mode_samples = [s for s in samples if int(s["Theta"][0]) == mode]
        ys = [s["Y"] for s in mode_samples[:args.n]]

        means = [y.mean() for y in ys]
        stds = [y.std() for y in ys]
        pos_fracs = [(y > 0).mean() for y in ys]
        maxs = [y.max() for y in ys]
        mins = [y.min() for y in ys]

        print(f"\n=== MODE {mode} ({len(mode_samples)} samples) ===")
        print(f"  Y mean:     {np.mean(means):.4f} ± {np.std(means):.4f}")
        print(f"  Y std:      {np.mean(stds):.4f} ± {np.std(stds):.4f}")
        print(f"  Y max:      {np.mean(maxs):.4f} ± {np.std(maxs):.4f}")
        print(f"  Y min:      {np.mean(mins):.4f} ± {np.std(mins):.4f}")
        print(f"  % positive: {np.mean(pos_fracs)*100:.1f}% ± {np.std(pos_fracs)*100:.1f}%")

        # İlk 3 sample'ın detayı
        for i in range(min(3, len(mode_samples))):
            y = mode_samples[i]["Y"]
            gid = mode_samples[i]["geom_id"]
            freq = mode_samples[i]["Theta"][1]
            print(f"  Sample {i}: geom={gid}, freq={freq:.2f}GHz, "
                  f"shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}, "
                  f"min={y.min():.4f}, max={y.max():.4f}")


if __name__ == "__main__":
    main()
