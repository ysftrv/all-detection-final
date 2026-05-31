"""
error_analysis.py — Yanlış sınıflandırılan görüntüleri görselleştir ve örüntü bul.

Kullanım:
    python -m src.analysis.error_analysis --model resnet50 --aug true

Çıktılar:
    experiments/figures/{model}_false_positives.png   (Normal → ALL tahmini)
    experiments/figures/{model}_false_negatives.png   (ALL → Normal tahmini)
    experiments/tables/{model}_error_stats.csv        (hata oranı özeti)
"""

# TODO (Faz 5):
#   1. Test seti tahminlerini ve ground truth'ları yükle
#      (evaluate.py'nin kaydettiği predictions.csv'den).
#   2. False positive ve false negative indekslerini bul.
#   3. Her kategori için en "emin" hataları (yüksek confidence ama yanlış) seç.
#   4. matplotlib subplot ızgarasında görüntüleri göster; başlığa confidence yaz.
#   5. İsteğe bağlı: Grad-CAM ısı haritasını hata görseline bindir
#      (sadece derin modeller, Faz 6+ olarak işaretle).
#   6. Hata örüntüsü gözlemlerini (bulanıklık, aydınlatma, vb.) error_stats.csv'ye not et.


def find_errors(predictions_csv: str):
    """Yanlış sınıflandırılan örnekleri bul. → (false_pos_df, false_neg_df)"""
    raise NotImplementedError("TODO (Faz 5): find_errors implemente edilmedi.")


def plot_error_grid(image_paths, confidences, title: str, save_path: str, n: int = 16):
    """n adet hatayı ızgara görselinde göster."""
    raise NotImplementedError("TODO (Faz 5): plot_error_grid implemente edilmedi.")


def main():
    raise NotImplementedError("TODO (Faz 5): error_analysis.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
