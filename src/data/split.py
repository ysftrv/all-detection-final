"""
split.py — C-NMC görüntülerini hasta-bazlı (patient-level) train/val/test
           listelerine böl; bölme sonucunu data/processed/ altına CSV olarak kaydet.

Neden hasta-bazlı bölme?
    Aynı hastaya ait görüntüler train'de, test'te ayrı kümelere düşerse data
    leakage oluşur ve test metrikler gerçekçi olmaz.

Kullanım:
    python -m src.data.split

Çıktılar:
    data/processed/train.csv   (path, label, patient_id)
    data/processed/val.csv
    data/processed/test.csv
    data/processed/split_summary.txt   (her küme için sınıf dağılımı)
"""

# TODO (Faz 1):
#   1. config.yaml'ı yükle (seed, dataset.val_split, dataset.test_split,
#      dataset.patient_id_field).
#   2. data/raw/ altındaki görüntü yollarını tara; her görüntüye ait
#      patient_id'yi klasör adından veya dosya adından çıkar.
#      (C-NMC'de klasör adları zaten hasta ID'si olabilir — doğrula.)
#   3. Benzersiz hasta listesini al; sklearn.model_selection.GroupShuffleSplit
#      ile önce test'i ayır, kalan train/val olarak böl.
#   4. Her bölmedeki sınıf dengesini kontrol et; ciddi dengesizlik varsa uyar.
#   5. train.csv, val.csv, test.csv'yi yaz; split_summary.txt üret.
#   6. Yeniden üretilebilirlik için config seed'ini dosyaların başına ekle.


def main():
    raise NotImplementedError("TODO (Faz 1): split.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
