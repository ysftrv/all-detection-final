"""
run_test.py — Tek komutla model yükle, test setini değerlendir, sonuçları kaydet.

Kullanım:
    python scripts/run_test.py --model resnet50
    python scripts/run_test.py --model baseline
    python scripts/run_test.py --all          # config.yaml'daki tüm modelleri çalıştır

Davranış:
    1. config.yaml ve argümanları yükle.
    2. İlgili model ağırlığını/joblib dosyasını experiments/{model}/ altından yükle.
    3. data/processed/test.csv üzerinde tahmin yap.
    4. Metrikleri ve görselleri experiments/{model}/ altına kaydet.
    5. Konsola özet tabloyu yazdır.

Çıkış kodu:
    0 — başarı
    1 — ağırlık/model dosyası bulunamadı
    2 — test CSV bulunamadı

NOT: Bu dosya eğitim YAPMAZ; sadece mevcut ağırlıkları test eder.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from src.evaluate import evaluate_single
from src.train import resolve_output_dir
from src.utils.config import load_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ALL_MODELS = ["baseline", "classical_ml", "alexnet", "vgg16", "resnet50"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kayitli modeli/modelleri test seti uzerinde degerlendir."
    )
    p.add_argument("--model", default=None,
                   help="baseline | classical_ml | alexnet | vgg16 | resnet50")
    p.add_argument("--all", action="store_true",
                   help="config.yaml'daki tum modelleri sirayla calistir.")
    p.add_argument("--config", default="config.yaml",
                   help="Alternatif yaml yolu (varsayilan: config.yaml).")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="cuda | cpu (varsayilan: otomatik algila).")
    p.add_argument("--output-dir", default=None,
                   help="Cikti kok dizini; config.phase2.output_dir'i ezer.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.all and args.model is None:
        log.error("--model belirtin ya da --all kullanin.")
        return 1

    cfg = load_config(args.config)

    # Cihaz secimi (kullanici belirtimi veya otomatik)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device(
        "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    log.info("Cihaz: %s", device)

    # Cikti dizini ve test CSV on kontrolu
    output_dir = resolve_output_dir(args.output_dir, cfg)
    proc_dir = Path(cfg["paths"]["data_processed"])
    test_csv = proc_dir / "test.csv"
    if not test_csv.exists():
        log.error("Test CSV bulunamadi: %s", test_csv)
        return 2

    models = (cfg.get("phase2", {}).get("models", ALL_MODELS)
              if args.all else [args.model])

    log.info("Cikti dizini: %s", output_dir.resolve())

    results = {}
    for m in models:
        log.info("=== Degerlendiriliyor: %s ===", m)
        exp_dir = output_dir / m
        weight_ok = (exp_dir / "weights.pth").exists() or (exp_dir / "model.joblib").exists()
        if not weight_ok:
            log.error("Agirlik/model dosyasi bulunamadi: %s (weights.pth / model.joblib)",
                      exp_dir)
            return 1
        try:
            results[m] = evaluate_single(m, cfg, output_dir=output_dir)
        except FileNotFoundError as e:
            log.error("%s", e)
            return 1

    # Ozet tablo
    print("\n" + "=" * 68)
    print(f"  {'Model':<15} {'Acc':>6} {'F1':>6} {'AUC':>6} "
          f"{'Spec':>7} {'Prec':>6} {'Recall':>6}")
    print("=" * 68)
    for m, r in results.items():
        print(f"  {m:<15} {r['accuracy']:>6.4f} {r['f1']:>6.4f} "
              f"{r['roc_auc']:>6.4f} {r['specificity']:>7.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
