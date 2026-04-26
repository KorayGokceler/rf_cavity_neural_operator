import os
import subprocess
import yaml
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run full GNOT pipeline (Data Generation -> Conversion -> Training).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--skip-gen", action="store_true", help="Skip data generation step")
    parser.add_argument("--skip-convert", action="store_true", help="Skip data conversion step")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    
    # Kalan argümanları yakala (Colab'da --override training.num_workers=2 gibi geçmek için)
    args, unknown_args = parser.parse_known_args()

    if not os.path.exists(args.config):
        print(f"❌ Hata: Config dosyası bulunamadı: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print("\n" + "═"*60)
    print("🚀 GNOT UÇTAN UCA EĞİTİM BORU HATTI (PIPELINE) 🚀")
    print("═"*60)
    
    # ==========================================
    # 1. VERİ ÜRETİMİ (DATA GENERATION)
    # ==========================================
    if not args.skip_gen:
        print("\n[1/3] VERİ ÜRETİMİ (DATA GENERATION) BAŞLIYOR...")
        gen_args = [
            "python", "src/data_gen/dataset_generator.py",
            "--h5_filename", cfg.get('data_gen', {}).get('h5_filename', 'rf_cavity_1000_dataset.h5'),
            "--plot_dir", cfg.get('data_gen', {}).get('plot_dir', 'dataset_plots'),
            "--n_total", str(cfg.get('data_gen', {}).get('n_total', 1000)),
            "--n_plot", str(cfg.get('data_gen', {}).get('n_plot', 100)),
            "--mode", cfg.get('data_gen', {}).get('generation_mode', 'random')
        ]
        
        try:
            subprocess.run(gen_args, check=True)
            print("✅ Veri üretimi başarılı!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Veri üretimi başarısız oldu: {e}")
            sys.exit(1)
    else:
        print("\n[1/3] ⏩ Veri üretimi atlandı (--skip-gen kullanıldı).")

    # ==========================================
    # 2. VERİ DÖNÜŞTÜRME (CONVERSION)
    # ==========================================
    if not args.skip_convert:
        print("\n" + "─"*60)
        print("[2/3] FİZİKSEL FEATURE ÇIKARIMI (DATA CONVERSION) BAŞLIYOR...")
        
        convert_args = [
            "python", "convert.py",
            "--h5_filepath", cfg.get('data_convert', {}).get('h5_filepath', 'rf_cavity_1000_dataset.h5'),
            "--output_path", cfg.get('data_convert', {}).get('output_path', 'data/gnot_dataset.pkl'),
            "--output_format", cfg.get('data_convert', {}).get('output_format', 'pkl'),
            "--modes"
        ] + [str(m) for m in cfg.get('data_convert', {}).get('mode_indices', [0, 1, 2])]
        
        try:
            subprocess.run(convert_args, check=True)
            print("✅ Veri dönüştürme başarılı!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Veri dönüştürme başarısız oldu: {e}")
            sys.exit(1)
    else:
        print("\n" + "─"*60)
        print("[2/3] ⏩ Veri dönüştürme atlandı (--skip-convert kullanıldı).")

    # ==========================================
    # 3. EĞİTİM (TRAINING)
    # ==========================================
    if not args.skip_train:
        print("\n" + "─"*60)
        print("[3/3] MODEL EĞİTİMİ (TRAINING) BAŞLIYOR...")
        train_args = ["python", "train.py", "--config", args.config] + unknown_args
        
        try:
            subprocess.run(train_args, check=True)
            print("✅ Model eğitimi tamamlandı!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Model eğitimi başarısız oldu: {e}")
            sys.exit(1)
    else:
        print("\n" + "─"*60)
        print("[3/3] ⏩ Model eğitimi atlandı (--skip-train kullanıldı).")

    print("\n" + "═"*60)
    print("🎉 GNOT PIPELINE BAŞARIYLA TAMAMLANDI! 🎉")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()
