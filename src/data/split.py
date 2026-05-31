"""
split.py — C-NMC görüntülerini HASTA-BAZLI train/val/test olarak böl.

Neden hasta-bazlı?
    Aynı hastanın görüntüleri farklı kümelere düşerse model hasta özelliklerini
    ezberler → test metrikleri gerçek dünya performansını yansıtmaz (data leakage).

Bölme stratejisi:
    1. GroupShuffleSplit ile hastaları önce train+val / test olarak ayır.
    2. Kalan train+val içinde tekrar GroupShuffleSplit → train / val.
    Oran hedefi: %70 train / %15 val / %15 test (hastalar bazında).

Kullanım:
    python -m src.data.split

Çıktılar:
    data/processed/train.csv  — sütunlar: filepath, label, patient_id, fold
    data/processed/val.csv
    data/processed/test.csv
    experiments/tables/split_summary.csv  — hasta & görüntü sayıları, sınıf dengesi
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.explore import scan_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _patient_level_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hasta-bazlı ikili bölme: (büyük küme, küçük küme).
    GroupShuffleSplit hasta ID'lerini gruplar halinde böler;
    aynı hastanın görüntüleri her zaman aynı tarafta kalır.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X = np.arange(len(df))
    groups = df["patient_id"].values
    main_idx, holdout_idx = next(gss.split(X, groups=groups))
    return df.iloc[main_idx].copy(), df.iloc[holdout_idx].copy()


def split_dataset(raw_dir: Path, out_dir: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    Bütün veri setini tara, hasta-bazlı böl, CSV'leri yaz.
    Döndürür: {"train": df, "val": df, "test": df}
    """
    seed = cfg["seed"]
    val_frac = cfg["dataset"]["val_split"]    # 0.15
    test_frac = cfg["dataset"]["test_split"]  # 0.15

    log.info("Veri seti taranıyor…")
    df = scan_dataset(raw_dir)
    log.info("Toplam: %d görüntü, %d benzersiz hasta.", len(df), df["patient_id"].nunique())

    # Adım 1: test ayır
    # test_frac oranında (tüm veri üzerinden) test seti oluştur
    trainval_df, test_df = _patient_level_split(df, test_size=test_frac, seed=seed)

    # Adım 2: val ayır (kalan train+val içinden)
    # val_frac_adjusted: trainval içinde val'ın hedef oranı
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    train_df, val_df = _patient_level_split(trainval_df, test_size=val_frac_adjusted, seed=seed)

    # Hasta sızıntısı kontrolü
    train_patients = set(train_df["patient_id"])
    val_patients   = set(val_df["patient_id"])
    test_patients  = set(test_df["patient_id"])

    assert train_patients.isdisjoint(val_patients),  "HATA: train–val hasta örtüşmesi var!"
    assert train_patients.isdisjoint(test_patients), "HATA: train–test hasta örtüşmesi var!"
    assert val_patients.isdisjoint(test_patients),   "HATA: val–test hasta örtüşmesi var!"
    log.info("Hasta sızıntısı kontrolü PASSED — örtüşme yok.")

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # CSV'leri yaz
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8")
        log.info("Kaydedildi: %s  (%d görüntü, %d hasta)", path, len(split_df), split_df["patient_id"].nunique())

    return splits


def build_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, df in splits.items():
        n_all = int((df["label"] == 1).sum())
        n_hem = int((df["label"] == 0).sum())
        n_patients = int(df["patient_id"].nunique())
        total = len(df)
        rows.append({
            "Split":             split_name,
            "Görüntü (toplam)":  total,
            "ALL":               n_all,
            "HEM":               n_hem,
            "ALL %":             round(n_all / total * 100, 1),
            "Hasta sayısı":      n_patients,
        })
    return pd.DataFrame(rows)


def main():
    from src.utils.config import load_config

    cfg = load_config()
    raw_dir  = Path(cfg["paths"]["data_raw"])
    out_dir  = Path(cfg["paths"]["data_processed"])
    tbl_dir  = Path(cfg["paths"]["tables"])
    tbl_dir.mkdir(parents=True, exist_ok=True)

    splits = split_dataset(raw_dir, out_dir, cfg)

    summary = build_split_summary(splits)
    summary_path = tbl_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    log.info("Kaydedildi: %s", summary_path)

    print("\n" + "=" * 62)
    print("  Hasta-Bazlı Bölme Özeti")
    print("=" * 62)
    print(summary.to_string(index=False))
    print("=" * 62)
    print("Hasta sızıntısı: YOK (assert ile doğrulandı)")
    print()


if __name__ == "__main__":
    main()
