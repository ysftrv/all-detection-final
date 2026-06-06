"""
explore.py — C-NMC veri setini keşfet; istatistikler ve görseller üret.

Kullanım:
    python -m src.data.explore

Çıktılar:
    experiments/figures/class_distribution.png   — ALL vs HEM bar grafiği
    experiments/figures/sample_grid.png          — her sınıftan 8 örnek
    experiments/figures/images_per_patient.png   — hasta başına görüntü histogramı
    experiments/tables/data_summary.csv          — tam istatistik tablosu
"""

import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # GUI gerektirmez (Colab ve sunucu uyumlu)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_patient_id(stem: str) -> str:
    """
    C-NMC dosya adından GERÇEK hasta ID'sini çıkar.

    Format: UID_<hasta_no>_<hücre_no>_...
    Örnek:  UID_11_29_1_all → UID_11  (11 = hasta no, 29 = hücre/örnek no)

    Kural: Yalnızca ilk iki alt çizgi bölümü (UID öneki + hasta numarası) alınır.
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return stem


def parse_sample_id(stem: str) -> str:
    """Hücre/örnek numarasını çıkar: UID_11_29_... → 29"""
    parts = stem.split("_")
    return parts[2] if len(parts) >= 3 else ""


def scan_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    data/raw/ altını tarar; tüm görüntüler için bir DataFrame döndürür.

    Sütunlar: filepath, label (all=1 / hem=0), fold, patient_id, width, height
    """
    records = []
    image_exts = {".bmp", ".jpg", ".jpeg", ".png"}

    training_root = None
    for cand in raw_dir.rglob("training_data"):
        if cand.is_dir():
            training_root = cand
            break

    if training_root is None:
        raise FileNotFoundError(
            f"training_data/ klasörü bulunamadı: {raw_dir}\n"
            "Önce 'python -m src.data.download' çalıştırın."
        )

    for fold_dir in sorted(training_root.iterdir()):
        if not fold_dir.is_dir():
            continue
        fold_name = fold_dir.name
        for cls_dir in sorted(fold_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            if cls_name not in {"all", "hem"}:
                continue
            label = 1 if cls_name == "all" else 0
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in image_exts:
                    continue
                patient_id = parse_patient_id(img_path.stem)
                records.append({
                    "filepath": str(img_path),
                    "label": label,
                    "class_name": cls_name,
                    "fold": fold_name,
                    "patient_id": patient_id,
                    "sample_id": parse_sample_id(img_path.stem),
                })

    if not records:
        raise RuntimeError("Hiç görüntü bulunamadı. Klasör yapısını kontrol edin.")

    df = pd.DataFrame(records)

    # Boyut örneklemesi: her sınıftan en fazla 200 görüntünün boyutunu oku
    log.info("Görüntü boyutları örnekleniyor (%d görüntüden)…", min(400, len(df)))
    widths, heights = [], []
    sample_idx = (
        df.groupby("label")
        .apply(lambda g: g.sample(min(200, len(g)), random_state=42))
        .index.get_level_values(1)
    )
    for idx in sample_idx:
        try:
            with Image.open(df.loc[idx, "filepath"]) as im:
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            pass

    log.info(
        "Boyut istatistikleri (örneklem) — W: %d–%d (ort %.0f) | H: %d–%d (ort %.0f)",
        min(widths), max(widths), np.mean(widths),
        min(heights), max(heights), np.mean(heights),
    )

    return df


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df.groupby(["class_name", "fold"]).size().unstack(fill_value=0)
    ax = counts.T.plot(kind="bar", figsize=(8, 5), color=["#e74c3c", "#2ecc71"])
    ax.set_title("C-NMC 2019 — Sınıf Dağılımı (fold bazında)", fontsize=13)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Görüntü Sayısı")
    ax.legend(title="Sınıf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Kaydedildi: %s", out_path)


def plot_sample_grid(df: pd.DataFrame, out_path: Path, n_per_class: int = 8) -> None:
    fig, axes = plt.subplots(2, n_per_class, figsize=(2 * n_per_class, 5))
    for row_idx, (cls_label, cls_name, color) in enumerate(
        [(1, "ALL (Lösemi)", "#e74c3c"), (0, "HEM (Normal)", "#2ecc71")]
    ):
        samples = df[df["label"] == cls_label].sample(n_per_class, random_state=42)
        for col_idx, (_, row) in enumerate(samples.iterrows()):
            ax = axes[row_idx][col_idx]
            try:
                img = Image.open(row["filepath"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "?", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(cls_name, color=color, fontsize=9, rotation=90, labelpad=5)
    fig.suptitle("C-NMC 2019 — Örnek Görüntüler", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Kaydedildi: %s", out_path)


def plot_images_per_patient(df: pd.DataFrame, out_path: Path) -> None:
    per_patient = df.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_patient.values, bins=30, color="#3498db", edgecolor="white")
    ax.set_title("Hasta Başına Görüntü Sayısı Dağılımı", fontsize=13)
    ax.set_xlabel("Görüntü Sayısı / Hasta")
    ax.set_ylabel("Hasta Sayısı")
    ax.axvline(per_patient.mean(), color="red", linestyle="--", label=f"Ort: {per_patient.mean():.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Kaydedildi: %s", out_path)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    n_all = (df["label"] == 1).sum()
    n_hem = (df["label"] == 0).sum()
    n_patients = df["patient_id"].nunique()
    per_patient = df.groupby("patient_id").size()

    rows = [
        {"Metrik": "Toplam görüntü",             "Değer": total},
        {"Metrik": "ALL (lösemi) görüntü",        "Değer": int(n_all)},
        {"Metrik": "HEM (normal) görüntü",        "Değer": int(n_hem)},
        {"Metrik": "ALL oranı (%)",               "Değer": round(n_all / total * 100, 2)},
        {"Metrik": "Benzersiz hasta sayısı",       "Değer": int(n_patients)},
        {"Metrik": "Hasta başına görüntü — min",  "Değer": int(per_patient.min())},
        {"Metrik": "Hasta başına görüntü — maks", "Değer": int(per_patient.max())},
        {"Metrik": "Hasta başına görüntü — ort",  "Değer": round(float(per_patient.mean()), 1)},
        {"Metrik": "Hasta başına görüntü — std",  "Değer": round(float(per_patient.std()), 1)},
        {"Metrik": "Fold sayısı",                 "Değer": df["fold"].nunique()},
    ]
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        rows.append({"Metrik": f"  {fold} — ALL", "Değer": int((sub["label"] == 1).sum())})
        rows.append({"Metrik": f"  {fold} — HEM", "Değer": int((sub["label"] == 0).sum())})

    return pd.DataFrame(rows)


def main():
    from src.utils.config import load_config

    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_raw"])
    fig_dir = Path(cfg["paths"]["figures"])
    tbl_dir = Path(cfg["paths"]["tables"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    log.info("Veri seti taranıyor: %s", raw_dir)
    df = scan_dataset(raw_dir)

    # Grafikler
    plot_class_distribution(df, fig_dir / "class_distribution.png")
    plot_sample_grid(df, fig_dir / "sample_grid.png")
    plot_images_per_patient(df, fig_dir / "images_per_patient.png")

    # Özet tablo
    summary = build_summary(df)
    summary_path = tbl_dir / "data_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    log.info("Kaydedildi: %s", summary_path)

    # Konsol raporu
    print("\n" + "=" * 52)
    print("  C-NMC 2019 — Veri Seti Özeti")
    print("=" * 52)
    print(summary.to_string(index=False))
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
