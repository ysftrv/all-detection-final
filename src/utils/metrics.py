"""
metrics.py — Tüm modeller için ortak metrik hesaplama, görselleştirme ve kaydetme.

Dışa aktarılanlar:
    CLASS_NAMES                : ["HEM", "ALL"]
    compute_metrics(y_true, y_pred, y_proba)  → dict
    save_metrics(metrics, out_dir)
    save_predictions(ids, y_true, y_pred, y_proba, out_dir)
    save_run_info(info, out_dir)
    plot_confusion_matrix(cm, class_names, save_path)
    plot_roc_curve(y_true, y_proba, save_path)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["HEM", "ALL"]  # 0 → HEM (normal), 1 → ALL (lösemi)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Sınıflandırma metriklerini hesapla (pozitif sınıf = ALL = 1).

    Girdi:
        y_true  : gerçek etiketler (0/1)
        y_pred  : tahmin edilen etiketler (0/1)
        y_proba : ALL sınıfı olasılığı (float, 0–1)
    Çıktı: dict — accuracy, precision, recall, f1, specificity, roc_auc, confusion_matrix
    """
    y_true  = np.array(y_true,  dtype=int)
    y_pred  = np.array(y_pred,  dtype=int)
    y_proba = np.array(y_proba, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity":      float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "roc_auc":          float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def save_metrics(metrics: dict, out_dir: Path) -> None:
    """Metrikleri experiments/{model}/metrics.json olarak kaydet (sabit şema)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_predictions(ids, y_true, y_pred, y_proba, out_dir: Path) -> None:
    """Tahminleri experiments/{model}/predictions.csv olarak kaydet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_id":   list(ids),
        "y_true":     list(y_true),
        "y_pred":     list(y_pred),
        "y_prob_ALL": list(y_proba),
    }).to_csv(out_dir / "predictions.csv", index=False)


def save_run_info(info: dict, out_dir: Path) -> None:
    """Çalıştırma meta bilgilerini experiments/{model}/run_info.json olarak kaydet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)


def plot_confusion_matrix(cm, class_names, save_path: Path) -> None:
    """Sayı ve normalize karışıklık matrisini yan yana kaydet."""
    cm       = np.array(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (count)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", ax=ax, vmin=0,
        )
        ax.set_title(title)
        ax.set_ylabel("Gerçek")
        ax.set_xlabel("Tahmin")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, save_path: Path) -> None:
    """ROC eğrisini AUC değeriyle kaydet."""
    y_true  = np.array(y_true,  dtype=int)
    y_proba = np.array(y_proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
