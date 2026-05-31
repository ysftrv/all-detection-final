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

Değerlendirme kriterleri: Accuracy, Precision, Recall, F1 (macro), ROC-AUC.  
Augmentation ablasyonu: her model aug=True ve aug=False ile koşturulur.

---

## Veri Seti

**C-NMC 2019** (Challenge on Classification of Normal vs Malignant Cells)

- **Kaynak:** Kaggle — `andrewmvd/leukemia-classification`
- **Sınıflar:** ALL (lösemi hücresi), HEM (normal hematopoietik hücre)
- **Yapı:** Eğitim verisi 3 fold'a bölünmüş + ayrı test seti
- **Bölme stratejisi:** Hasta-bazlı (data leakage'ı önlemek için)
- **Lisans:** Araştırma ve akademik kullanım (ticari kullanım için kaynak ekibiyle iletişime geç)

---

## Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADI/all-detection-final.git
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
python -m src.data.download

# Veri setini keşfet (istatistikler ve örnek görüntüler)
python -m src.data.explore

# Hasta-bazlı train/val/test bölmesi
python -m src.data.split
```

### Faz 3 — Baseline & Klasik ML Eğitimi

```bash
python -m src.train --model baseline --aug false
python -m src.train --model classical_ml --aug false
```

### Faz 4 — Derin Öğrenme Eğitimi (Google Colab'da koştur)

```bash
# Colab'da:
# notebooks/colab_runner.ipynb dosyasını açıp sırayla çalıştır

# Yerel (GPU varsa):
python -m src.train --model alexnet  --aug true
python -m src.train --model vgg16    --aug true
python -m src.train --model resnet50 --aug true
```

### Faz 5 — Değerlendirme & Analiz

```bash
# Tüm modelleri değerlendir
python -m src.evaluate --model resnet50 --aug true

# Karşılaştırma tablosu ve grafikleri üret
python -m src.analysis.compare
python -m src.analysis.ablation
python -m src.analysis.error_analysis --model resnet50 --aug true
```

---

## run_test.py Kullanımı

Kayıtlı ağırlıklarla tek komutta değerlendirme:

```bash
# Tek model
python scripts/run_test.py --model resnet50 --aug true

# Baseline
python scripts/run_test.py --model baseline

# Config.yaml'daki tüm modelleri çalıştır
python scripts/run_test.py --all

# Özel config ve cihaz
python scripts/run_test.py --model vgg16 --aug false --config config.yaml --device cuda
```

Çıktılar `experiments/` altına kaydedilir:
- `weights/` — model ağırlıkları (.pth)
- `logs/`    — eğitim logları ve run bilgisi (JSON)
- `figures/` — karışıklık matrisi, ROC eğrisi, hata ızgaraları
- `tables/`  — metrik CSV dosyaları, karşılaştırma tablosu

---

## Donanım

Eğitim Google Colab T4 GPU üzerinde koşturulmuştur.  
Ayrıntılar için → [`docs/hardware.md`](docs/hardware.md)  
Otomatik kaydedilen donanım bilgisi → `experiments/logs/hardware_info.json`

---

## Sonuçlar

> *Faz 5 tamamlandıktan sonra bu tablo güncellenecektir.*

| Model | Aug | Accuracy | Precision | Recall | F1 | AUC |
|-------|-----|----------|-----------|--------|-----|-----|
| Baseline (LR) | — | — | — | — | — | — |
| Classical ML (SVM) | — | — | — | — | — | — |
| AlexNet | ✓ | — | — | — | — | — |
| AlexNet | ✗ | — | — | — | — | — |
| VGG-16 | ✓ | — | — | — | — | — |
| VGG-16 | ✗ | — | — | — | — | — |
| ResNet-50 | ✓ | — | — | — | — | — |
| ResNet-50 | ✗ | — | — | — | — | — |

---

## Proje Yapısı

```
all-detection-final/
├── config.yaml              # Merkezi konfigürasyon
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                 # Ham veri (git'e dahil edilmez)
│   └── processed/           # train/val/test CSV'leri
├── src/
│   ├── data/
│   │   ├── download.py      # Kaggle indirme
│   │   ├── explore.py       # EDA
│   │   ├── split.py         # Hasta-bazlı bölme
│   │   └── dataset.py       # PyTorch Dataset + transforms
│   ├── models/
│   │   ├── baseline.py      # Otsu + LR
│   │   ├── classical_ml.py  # HOG/LBP + SVM
│   │   └── deep_models.py   # AlexNet / VGG / ResNet
│   ├── train.py             # Eğitim döngüsü
│   ├── evaluate.py          # Metrik hesaplama
│   └── analysis/
│       ├── compare.py       # Model karşılaştırma tablosu
│       ├── ablation.py      # Augmentation ablasyonu
│       └── error_analysis.py# Hata görselleştirme
├── scripts/
│   └── run_test.py          # Tek komutla test
├── notebooks/
│   └── colab_runner.ipynb   # Google Colab eğitim notebook'u
├── experiments/
│   ├── weights/             # Model ağırlıkları (git'e dahil edilmez)
│   ├── logs/                # Eğitim logları
│   ├── figures/             # Görseller
│   └── tables/              # Metrik CSV'leri
└── docs/
    └── hardware.md          # Donanım kaydı
```
