"""
ablation.py — Augmentation ablasyon analizi: aug=True vs aug=False sonuçlarını karşılaştır.

Kullanım:
    python -m src.analysis.ablation

Giriş:
    experiments/tables/*_aug_true_metrics.csv
    experiments/tables/*_aug_false_metrics.csv

Çıktılar:
    experiments/tables/ablation_table.csv
    experiments/figures/ablation_delta.png   (F1 artışı bar chart)
    experiments/figures/learning_curves.png  (aug açık/kapalı train-val loss eğrileri)
"""

# TODO (Faz 5):
#   1. aug=True ve aug=False metrik CSV'lerini yükle; farkları hesapla (ΔF1, ΔAUC).
#   2. Her model için augmentation katkısını tablo ve grafik olarak göster.
#   3. Eğitim loglarından (experiments/logs/*_history.json) loss/acc eğrilerini oku.
#   4. Yanyana subplot ile aug açık vs kapalı learning curve'leri çiz.
#   5. Bulguları kısa metin özeti olarak ablation_table.csv'nin altına ekle.


def compute_ablation_delta(aug_true_csv: str, aug_false_csv: str):
    """İki metrik CSV arasındaki farkı hesapla. → pandas DataFrame"""
    raise NotImplementedError("TODO (Faz 5): compute_ablation_delta implemente edilmedi.")


def plot_learning_curves(log_dir: str, model_name: str, save_path: str):
    raise NotImplementedError("TODO (Faz 5): plot_learning_curves implemente edilmedi.")


def main():
    raise NotImplementedError("TODO (Faz 5): ablation.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
