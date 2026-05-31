"""
run_test.py — Tek komutla model yükle, test setini değerlendir, sonuçları kaydet.

Kullanım:
    python scripts/run_test.py --model resnet50 --aug true
    python scripts/run_test.py --model baseline
    python scripts/run_test.py --all          # config.yaml'daki tüm modelleri çalıştır

Davranış:
    1. config.yaml ve argümanları yükle.
    2. İlgili model ağırlığını experiments/weights/ altından yükle.
    3. data/processed/test.csv üzerinde tahmin yap.
    4. Metrikleri ve görselleri experiments/ altına kaydet.
    5. Konsola özet tabloyu yazdır.

Çıkış kodu:
    0 — başarı
    1 — ağırlık dosyası bulunamadı
    2 — test CSV bulunamadı

NOT: Bu dosya eğitim YAPMAZ; sadece mevcut ağırlıkları test eder.
     Yeniden üretilebilirlik için her çalıştırmaya timestamp eklenir.
"""

import argparse
import sys

# TODO (Faz 5):
#
#   parse_args():
#     --model   : alexnet | vgg16 | resnet50 | baseline | classical_ml
#     --aug     : true | false
#     --all     : tüm modelleri sırayla çalıştır
#     --config  : alternatif yaml yolu (varsayılan: config.yaml)
#     --device  : cuda | cpu (varsayılan: otomatik algıla)
#
#   load_model(model_name, aug_flag, config):
#     - Derin model → build_model + state_dict yükle + eval()
#     - Klasik model → BaselineClassifier.load() veya ClassicalMLClassifier.load()
#
#   run_single(model_name, aug_flag, config):
#     - load_model çağır.
#     - test DataLoader veya image_paths yükle.
#     - evaluate.py'den compute_metrics çağır.
#     - Metrikleri ve görselleri kaydet.
#     - Sonuç dict döndür.
#
#   print_summary_table(results):
#     - tabulate veya el yapımı format ile konsola Acc/F1/AUC tablosu yazdır.
#
#   main():
#     - args parse et; --all bayrağına göre run_single döngüsü veya tek çalıştırma.
#     - print_summary_table ile bitir.


def main():
    raise NotImplementedError("TODO (Faz 5): run_test.py henüz implemente edilmedi.")


if __name__ == "__main__":
    sys.exit(main() or 0)
