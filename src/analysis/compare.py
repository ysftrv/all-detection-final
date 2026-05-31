"""
compare.py — Tüm modellerin metrik sonuçlarını tek bir karşılaştırma tablosuna topla.

Kullanım:
    python -m src.analysis.compare

Giriş:
    experiments/tables/*_metrics.csv  (evaluate.py'nin çıktıları)

Çıktılar:
    experiments/tables/comparison_table.csv   (LaTeX uyumlu format)
    experiments/figures/comparison_bar.png    (grouped bar chart: Accuracy / F1 / AUC)
    experiments/figures/comparison_radar.png  (radar/spider chart — isteğe bağlı)
"""

# TODO (Faz 5):
#   1. experiments/tables/ altındaki tüm *_metrics.csv dosyalarını glob ile bul.
#   2. Model adını ve augmentation bayrağını dosya adından parse et.
#   3. pandas DataFrame'e birleştir; sütunlar: Model, Aug, Accuracy, Precision,
#      Recall, F1, AUC.
#   4. Değerleri %'e çevir; en iyi değerleri bold olarak işaretle (LaTeX \textbf).
#   5. comparison_table.csv ve isteğe bağlı .tex dosyası kaydet.
#   6. matplotlib/seaborn ile grouped bar chart çiz; comparison_bar.png kaydet.


def build_comparison_table(results_dir: str):
    """*_metrics.csv dosyalarını oku → pandas DataFrame döndür."""
    raise NotImplementedError("TODO (Faz 5): build_comparison_table implemente edilmedi.")


def plot_comparison_bar(df, save_path: str):
    raise NotImplementedError("TODO (Faz 5): plot_comparison_bar implemente edilmedi.")


def main():
    raise NotImplementedError("TODO (Faz 5): compare.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
