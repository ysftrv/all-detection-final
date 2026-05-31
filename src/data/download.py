"""
download.py — Kaggle'dan C-NMC 2019 veri setini indir ve data/raw/ altına aç.

Kullanım:
    python -m src.data.download            # config.yaml'dan yolu okur
    python -m src.data.download --out data/raw

Gereksinimler:
    - Kaggle API anahtarı: ~/.kaggle/kaggle.json  (ya da KAGGLE_USERNAME / KAGGLE_KEY env değişkenleri)
    - pip install kaggle

Çıktı (beklenen klasör yapısı — Faz 1'de doğrula):
    data/raw/
        C_NMC_2019/
            training_data/
                fold_0/  fold_1/  fold_2/
                    all/    hem/          (all = lösemi, hem = normal)
            testing_data/
                C-NMC_test_prelim_phase_data/
                    all/    hem/
"""

# TODO (Faz 1):
#   1. config.yaml'ı yükle (seed, paths.data_raw, dataset.kaggle_id).
#   2. kaggle.api.authenticate() ile Kaggle oturumu aç.
#   3. kaggle.api.dataset_download_files(dataset_id, path=raw_dir, unzip=True) çağır.
#   4. İndirilen klasör yapısını kontrol et; beklenen dizinler yoksa ValueError fırlat.
#   5. İndirme tarihini ve dosya sayısını experiments/logs/download_log.txt'e yaz.


def main():
    raise NotImplementedError("TODO (Faz 1): download.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
