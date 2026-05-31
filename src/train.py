"""
train.py — Genel eğitim döngüsü (derin öğrenme modelleri için).

Kullanım:
    python -m src.train --model resnet50 --aug true
    python -m src.train --model alexnet  --aug false

Özellikler:
    - Epoch başına train loss/acc ve val loss/acc loglanır.
    - En iyi val-F1 ağırlıkları experiments/weights/{model}_{aug}.pth'e kaydedilir.
    - Eğitim/test süresi, donanım bilgisi ve hiperparametreler
      experiments/logs/{model}_{aug}_run_info.json'a yazılır.
    - Early stopping desteği (patience config.yaml'dan).
    - Baseline ve classical_ml için ayrı fit() çağrısı yapılır (DataLoader gerekmez).
"""

import argparse
import time

# TODO (Faz 4 — derin modeller) / (Faz 3 — baseline & classical_ml):
#
#   parse_args():
#     --model  : alexnet | vgg16 | resnet50 | baseline | classical_ml
#     --aug    : true | false   (config.yaml'daki augmentation.enabled'ı geçersiz kılar)
#     --config : isteğe bağlı alternatif yaml yolu
#
#   setup_logging(config, model_name, aug_flag) → log_dir:
#     - experiments/logs/ altında timestamp'li alt klasör yarat.
#     - Python logging'i hem dosyaya hem konsola bağla.
#
#   train_deep(model, train_loader, val_loader, config, device) → history dict:
#     - optimizer = AdamW(lr, weight_decay) kurulumu.
#     - scheduler seçimi (cosine / step / none).
#     - Epoch döngüsü: forward → loss → backward → optimizer.step.
#     - Val loop: torch.no_grad() içinde val loss/acc/F1.
#     - Early stopping sayacını güncelle; en iyi ağırlıkları kaydet.
#     - Her epoch sonunda history'ye loss/acc/f1 ekle.
#
#   train_classical(classifier, config):
#     - data/processed/train.csv'yi oku; görüntü yollarını ve etiketleri al.
#     - classifier.fit(X, y) çağır; kaydet.
#
#   log_run_info(config, model_name, aug_flag, elapsed_sec, device_info):
#     - Hiperparametreler, donanım, süre, tarih → JSON.
#
#   main():
#     - Argümanları parse et; config'i yükle; seed'i ayarla.
#     - Model türüne göre train_deep veya train_classical'ı çağır.
#     - Toplam süreyi ölç ve logla.


def main():
    raise NotImplementedError("TODO (Faz 3/4): train.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
