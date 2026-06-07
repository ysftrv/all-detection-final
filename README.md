# ALL Detection — Akut Lenfoblastik Lösemi İkili Sınıflandırması

Mikroskobik kan görüntülerinden ALL (lösemi) vs normal hücre sınıflandırması.  
**Veri seti:** C-NMC 2019 · **Çerçeve:** PyTorch · **Eğitim:** Google Colab (GPU)

---

## Proje Özeti

Bu proje üç farklı yaklaşımı karşılaştırmaktadır:

| Katman | Yöntem |
|--------|--------|
| 1 — Baseline | Otsu segmentasyon + el-yapımı özellikler + Lojistik Regresyon |
| 2 — Klasik ML | HOG + LBP + SVM |
| 3 — Derin Öğrenme | AlexNet / VGG-16 / ResNet-50 (transfer learning) |

Değerlendirme kriterleri: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC.  
Bölme stratejisi: **Hasta-bazlı (patient-disjoint)** — veri sızıntısını önlemek için
aynı hastanın görüntüleri yalnızca tek bir parçada (train/val/test) bulunur.

**En iyi sonuç:** ResNet-50 — Accuracy 0.8911, F1 0.9329, ROC-AUC 0.9029
(hasta-bazlı test seti, 11 hasta / 964 görüntü).

---

## Veri Seti

**C-NMC 2019** (Challenge on Classification of Normal vs Malignant Cells)

- **Kaynak:** Kaggle — `andrewmvd/leukemia-classification`
- **Sınıflar:** ALL (lösemi hücresi), HEM (normal hematopoietik hücre)
- **Boyut:** 10.661 görüntü, 73 gerçek hasta
- **Bölme:** Hasta-bazlı %70/15/15 → train 51 / val 11 / test 11 hasta
- **Lisans:** Araştırma ve akademik kullanım

---

## Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/ysftrv/all-detection-final.git
cd all-detection-final

# 2. Sanal ortam oluştur (önerilen)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Kaggle API anahtarını yapılandır
# ~/.kaggle/kaggle.json dosyasına {username, key} bilgilerini ekle
```

---

## Çalıştırma Talimatları

### Faz 1 — Veri Hazırlama

```bash
# Veri setini Kaggle'dan indir
PYTHONPATH=. python -m src.data.download

# Veri setini keşfet (istatistikler ve örnek görüntüler)
PYTHONPATH=. python -m src.data.explore

# Hasta-bazlı train/val/test bölmesi (data/processed/{train,val,test}.csv üretir)
PYTHONPATH=. python -m src.data.split
```

### Faz 2 — Model Eğitimi

```bash
PYTHONPATH=. python -m src.train --model baseline
PYTHONPATH=. python -m src.train --model classical_ml
PYTHONPATH=. python -m src.train --model alexnet
PYTHONPATH=. python -m src.train --model vgg16
PYTHONPATH=. python -m src.train --model resnet50
```

### Faz 3 — Değerlendirme

```bash
PYTHONPATH=. python -m src.evaluate --model all
```

---

## Değerlendirici İçin Hızlı Test (run_test.py)

Eğitilmiş modelleri tek komutla test seti üzerinde değerlendirir. **Eğitim YAPMAZ**,
yalnızca mevcut ağırlıkları test eder.

```bash
# Önce veri ve test bölmesi hazırlanmalı (bir kez):
PYTHONPATH=. python -m src.data.download   # C-NMC 2019 indir (Kaggle API gerekir)
PYTHONPATH=. python -m src.data.split      # hasta-bazlı test.csv üretir

# Tek model:
PYTHONPATH=. python scripts/run_test.py --model resnet50

# Tüm modeller (önerilen):
PYTHONPATH=. python scripts/run_test.py --all
```

**Argümanlar:** `--model {baseline|classical_ml|alexnet|vgg16|resnet50}`,
`--all`, `--config <yol>`, `--device {cuda|cpu}`, `--output-dir <yol>`.

**Çıkış kodları:** `0` başarı · `1` ağırlık/model dosyası yok · `2` test CSV yok.

Eğitilmiş ağırlıklar derin modeller için `experiments/{model}/weights.pth`,
klasik modeller için `experiments/{model}/model.joblib` altında bulunur. Sonuçlar
(metrics.json, predictions.csv, confusion matrix, ROC eğrisi) her modelin
`experiments/{model}/` klasörüne yazılır ve konsola özet tablo basılır.

---

## Donanım ve Hiperparametreler (Yeniden Üretilebilirlik)

- **Donanım:** NVIDIA L4 GPU (Google Colab)
- **Optimizer:** Adam (lr = 1e-4)
- **Batch size:** 32 · **Epoch:** 25 · **Loss:** Weighted Cross-Entropy
- **Overfitting önlemleri:** ReduceLROnPlateau, early stopping (val F1), dropout (0.5),
  L2 weight decay, AMP
- Her çalıştırmanın seed, donanım ve hiperparametre kaydı `experiments/{model}/run_info.json`
  dosyasında tutulur.

---

## Klasör Yapısı

```
all-detection-final/
├── config.yaml              # merkezi yapılandırma
├── requirements.txt
├── README.md
├── src/
│   ├── data/                # download, explore, split, dataset
│   ├── models/              # baseline, classical_ml, deep_models
│   ├── analysis/            # compare, ablation, error_analysis
│   ├── utils/               # config, metrics
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── run_test.py          # tek komutla test
├── experiments/             # her model: metrics.json, predictions.csv, weights, run_info
├── notebooks/
│   └── colab_runner.ipynb
└── docs/
    └── hardware.md
```
