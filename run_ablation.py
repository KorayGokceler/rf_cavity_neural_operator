"""
RF Cavity GNOT — Feature Encoding Ablation Study
=================================================
Runs 4 experiments sequentially to test which geometry features matter:

  A) Sadece (x, y)                    → val_dim=2
  B) (x, y) + boundary bilgisi       → val_dim=4
  C) Tüm 6 feature, RFF kapalı       → val_dim=6, use_rff=False
  D) Tüm 6 feature + RFF (referans)  → val_dim=6, use_rff=True

Kullanım:
  python run_ablation.py

Sonuçlar training_logs/ablation_* altına kaydedilir.
TensorBoard ile karşılaştırma:
  tensorboard --logdir training_logs
"""
import subprocess
import sys
import time

CONFIGS = [
    ("A", "configs/ablation/A_xy_only.yaml",       "Sadece (x, y)"),
    ("B", "configs/ablation/B_xy_boundary.yaml",    "(x, y) + boundary"),
    ("C", "configs/ablation/C_full_no_rff.yaml",    "Full 6 feat, RFF OFF"),
    ("D", "configs/ablation/D_full_with_rff.yaml",  "Full 6 feat + RFF ON"),
]

def run_experiment(label, config_path, description):
    print(f"\n{'='*60}")
    print(f"  ABLATION {label}: {description}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, "train.py", "--config", config_path],
        cwd=".",
    )
    elapsed = time.time() - start
    
    status = "✅ SUCCESS" if result.returncode == 0 else "❌ FAILED"
    print(f"\n{status} — Ablation {label} tamamlandı ({elapsed/60:.1f} dk)")
    return result.returncode == 0, elapsed

def main():
    print("🔬 RF Cavity GNOT — Feature Ablation Study")
    print("="*60)
    print("Feature referansı:")
    print("  [0] x_norm  [1] y_norm  [2] dist_boundary")
    print("  [3] boundary_mask  [4] dist_center  [5] curvature")
    print("="*60)
    
    results = []
    for label, config, desc in CONFIGS:
        success, elapsed = run_experiment(label, config, desc)
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
    
    print(f"\n📊 TensorBoard ile karşılaştırma:")
    print(f"   tensorboard --logdir training_logs")
    print(f"\n   Karşılaştırılacak metrikler:")
    print(f"   • val/field_rel_l2  (düşük = daha iyi)")
    print(f"   • val/loss")
    print(f"   • val/freq_mae_ghz")

if __name__ == "__main__":
    main()
