"""
RF Cavity GNOT — Feature Encoding Ablation Study
=================================================
Runs the feature-ablation experiments sequentially to test which geometry
features matter:

  A) Sadece (x, y)                    → feature_indices [0, 1],       val_dim=2
  B) (x, y) + boundary bilgisi       → feature_indices [0, 1, 2, 3], val_dim=4

Kullanım:
  python run_ablation.py                 # hepsini sırayla çalıştır
  python run_ablation.py --only A        # sadece A
  python run_ablation.py --dry-run       # komutları yazdır, eğitim başlatma
  python run_ablation.py -- --override training.max_epochs=5   # train.py'ye ek argüman

Sonuçlar training_logs/ablation_* altına kaydedilir.
TensorBoard ile karşılaştırma:
  tensorboard --logdir training_logs
"""
import argparse
import subprocess
import sys
import time

CONFIGS = [
    ("A", "configs/ablation/A_xy_only.yaml",       "Sadece (x, y)"),
    ("B", "configs/ablation/B_xy_boundary.yaml",    "(x, y) + boundary"),
]


def run_experiment(label, config_path, description, extra_args=(), dry_run=False):
    print(f"\n{'='*60}")
    print(f"  ABLATION {label}: {description}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, "train.py", "--config", config_path, *extra_args]
    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return True, 0.0

    start = time.time()
    result = subprocess.run(cmd, cwd=".")
    elapsed = time.time() - start

    status = "✅ SUCCESS" if result.returncode == 0 else "❌ FAILED"
    print(f"\n{status} — Ablation {label} tamamlandı ({elapsed/60:.1f} dk)")
    return result.returncode == 0, elapsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the feature-ablation trainings.")
    parser.add_argument("--only", nargs="+", choices=[c[0] for c in CONFIGS],
                        help="Run only these experiments (e.g. --only A)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the train.py commands without running them")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to train.py (after `--`)")
    args = parser.parse_args(argv)
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args


def main(argv=None):
    args = parse_args(argv)
    print("🔬 RF Cavity GNOT — Feature Ablation Study")
    print("="*60)
    print("Feature referansı (Input_funcs, val_dim=12):")
    print("  [0] x_norm  [1] y_norm  [2] dist_boundary  [3] dir_bnd_x  [4] dir_bnd_y")
    print("  [5] node_area  [6] cos_principal  [7] sin_principal")
    print("  [8] dist_2nd_bnd  [9] dist_3rd_bnd  [10] curvature  [11] convexity")
    print("="*60)

    selected = [c for c in CONFIGS if not args.only or c[0] in args.only]
    results = []
    for label, config, desc in selected:
        success, elapsed = run_experiment(label, config, desc, args.extra, args.dry_run)
        results.append((label, desc, success, elapsed))

    # Summary
    print(f"\n\n{'='*60}")
    print("  ABLATION STUDY SONUÇLARI")
    print(f"{'='*60}")
    print(f"{'Exp':<4} {'Açıklama':<30} {'Durum':<10} {'Süre':<10}")
    print("-"*60)
    for label, desc, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f" {label:<3} {desc:<30} {status:<10} {elapsed/60:.1f} dk")

    print("\n📊 TensorBoard ile karşılaştırma:")
    print("   tensorboard --logdir training_logs")
    print("\n   Karşılaştırılacak metrikler:")
    print("   • val/field_rel_l2  (düşük = daha iyi)")
    print("   • val/loss")
    print("   • val/freq_mae_ghz")
    return 0 if all(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
