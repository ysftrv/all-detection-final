"""
evaluate.py — Kayıtlı bir modeli test seti üzerinde değerlendir ve metrik raporunu üret.

Kullanım:
    python -m src.evaluate --model resnet50 --aug true
    python -m src.evaluate --model baseline

Çıktılar (experiments/ altına kaydedilir):
    figures/{model}_{aug}_confusion_matrix.png
    figures/{model}_{aug}_roc_curve.png
    tables/{model}_{aug}_metrics.csv      (accuracy, precision, recall, F1, AUC)
    tables/{model}_{aug}_report.txt       (sklearn classification_report)
"""

# TODO (Faz 5):
#
#   evaluate_model(model, test_loader_or_paths, labels, config, device) → metrics dict:
#     - Derin model için DataLoader üzerinde forward pass; klasik için predict().
#     - y_pred ve y_proba topla.
#
#   compute_metrics(y_true, y_pred, y_proba) → dict:
#     - accuracy_score, precision_score, recall_score, f1_score (macro & binary).
#     - roc_auc_score.
#     - confusion_matrix.
#
#   plot_confusion_matrix(cm, class_names, save_path):
#     - seaborn heatmap; normalize='true' versiyonu da kaydet.
#
#   plot_roc_curve(y_true, y_proba, save_path):
#     - sklearn.metrics.RocCurveDisplay ile AUC değerini eksen başlığına ekle.
#
#   save_metrics(metrics, model_name, aug_flag):
#     - CSV ve düz metin raporu yaz.
#
#   main():
#     - Argümanları parse et; modeli yükle; evaluate_model çağır; raporları kaydet.


def compute_metrics(y_true, y_pred, y_proba):
    """Tüm sınıflandırma metriklerini hesapla. → dict"""
    raise NotImplementedError("TODO (Faz 5): compute_metrics implemente edilmedi.")


def plot_confusion_matrix(cm, class_names, save_path: str):
    raise NotImplementedError("TODO (Faz 5): plot_confusion_matrix implemente edilmedi.")


def plot_roc_curve(y_true, y_proba, save_path: str):
    raise NotImplementedError("TODO (Faz 5): plot_roc_curve implemente edilmedi.")


def main():
    raise NotImplementedError("TODO (Faz 5): evaluate.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
