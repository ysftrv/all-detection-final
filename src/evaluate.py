"""
evaluate.py — Kayıtlı modeli test seti üzerinde değerlendir, metrik raporları üret.

Kullanım:
    python -m src.evaluate --model resnet50 [--config config.yaml]
    python -m src.evaluate --model baseline
    python -m src.evaluate --model all

Çıktılar (experiments/{model}/ altına):
    metrics.json, predictions.csv, confusion_matrix.png, roc_curve.png

Bu modül scripts/run_test.py tarafından da import edilir (Faz 4 temiz arayüz).
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.baseline import BaselineClassifier, extract_features
from src.models.classical_ml import ClassicalMLClassifier, extract_hog_lbp
from src.models.deep_models import SUPPORTED_MODELS, build_model, unfreeze_all
from src.utils.config import load_config
from src.utils.metrics import (
    CLASS_NAMES,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
    save_predictions,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CLASSICAL_MODELS = {"baseline", "classical_ml"}


# ---------------------------------------------------------------------------
# Model yükleme yardımcıları
# ---------------------------------------------------------------------------

def load_classical_model(model_name: str, exp_dir: Path, config: dict):
    """
    Eğitilmiş klasik modeli joblib dosyasından yükle.

    Girdi : model_name ('baseline' | 'classical_ml'), exp_dir, config
    Çıktı : BaselineClassifier veya ClassicalMLClassifier
    """
    model_path = exp_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    if model_name == "baseline":
        return BaselineClassifier.load(model_path, config)
    return ClassicalMLClassifier.load(model_path, config)


def load_deep_model(model_name: str, exp_dir: Path, config: dict,
                    device: torch.device) -> torch.nn.Module:
    """
    Eğitilmiş derin modeli ağırlık dosyasından yükle ve eval moduna al.

    Girdi : model_name, exp_dir, config (phase2 bloğu içeren), device
    Çıktı : eval modunda nn.Module
    """
    weights_path = exp_dir / "weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    p2 = config.get("phase2", {})
    if model_name == "vgg16":
        variant = p2.get("vgg_variant", "vgg16")
    elif model_name == "resnet50":
        variant = p2.get("resnet_variant", "resnet50")
    else:
        variant = model_name

    model = build_model(variant, config)
    unfreeze_all(model)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Değerlendirme fonksiyonları
# ---------------------------------------------------------------------------

def evaluate_classical(clf, model_name: str, proc_dir: Path, config: dict,
                        exp_dir: Path, test_df: pd.DataFrame | None = None) -> dict:
    """
    Klasik model için test seti değerlendirmesi.

    Girdi:
        clf       : yüklenmiş BaselineClassifier veya ClassicalMLClassifier
        test_df   : None ise proc_dir/test.csv okunur
    Çıktı: metrics dict
    """
    if test_df is None:
        test_df = pd.read_csv(proc_dir / "test.csv")

    paths  = test_df["filepath"].tolist()
    y_true = test_df["label"].tolist()
    ids    = [Path(p).stem for p in paths]

    log.info("%s: %d görüntü üzerinde test değerlendirmesi…", model_name, len(paths))
    t0 = time.time()

    if model_name == "baseline":
        img_size = config.get("phase2", {}).get("image_size", 128)
        X = np.vstack([extract_features(p, config) for p in tqdm(paths, desc="features")]).astype(np.float32)
    else:
        img_size = config.get("phase2", {}).get("image_size", 128)
        X = np.vstack([extract_hog_lbp(p, img_size) for p in tqdm(paths, desc="features")]).astype(np.float32)

    proba = clf.predict_proba_from_X(X)[:, 1]
    preds = clf.predict_from_X(X)
    infer_time = time.time() - t0

    log.info("Çıkarım süresi: %.1f s", infer_time)

    metrics = compute_metrics(y_true, preds, proba)
    save_metrics(metrics, exp_dir)
    save_predictions(ids, y_true, preds, proba, exp_dir)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES,
                          exp_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, proba, exp_dir / "roc_curve.png")

    log.info("%s — F1=%.4f  AUC=%.4f  Acc=%.4f",
             model_name, metrics["f1"], metrics["roc_auc"], metrics["accuracy"])
    return metrics


def evaluate_deep(model: torch.nn.Module, model_name: str, proc_dir: Path,
                  config: dict, exp_dir: Path, device: torch.device,
                  test_df: pd.DataFrame | None = None) -> dict:
    """
    Derin model için test seti değerlendirmesi.

    Girdi:
        model    : eval modunda nn.Module
        test_df  : None ise proc_dir/test.csv okunur
    Çıktı: metrics dict
    """
    from src.data.dataset import ALLDataset, get_transforms
    from torch.utils.data import DataLoader
    import tempfile, os

    if test_df is None:
        test_df = pd.read_csv(proc_dir / "test.csv")

    # Geçici CSV ile DataLoader kur (test_df'deki satırlar)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    test_df.to_csv(tmp.name, index=False)
    tmp.close()

    try:
        p2       = config.get("phase2", {})
        cfg_size = config.copy()
        cfg_size.setdefault("dataset", {})["image_size"] = p2.get("image_size", 224)
        tf       = get_transforms(cfg_size, "test")
        ds       = ALLDataset(tmp.name, transform=tf)
        loader   = DataLoader(ds, batch_size=p2.get("batch_size", 32),
                              shuffle=False, num_workers=0)

        y_true_all, y_pred_all, y_proba_all = [], [], []
        log.info("%s: %d görüntü üzerinde test değerlendirmesi…", model_name, len(ds))
        t0 = time.time()

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="test inference"):
                imgs = imgs.to(device)
                out  = model(imgs)
                proba = torch.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
                y_true_all.extend(labels.tolist())
                y_pred_all.extend(preds.cpu().tolist())
                y_proba_all.extend(proba.cpu().tolist())

        infer_time = time.time() - t0
        log.info("Çıkarım süresi: %.1f s", infer_time)
    finally:
        os.unlink(tmp.name)

    ids = [Path(p).stem for p in test_df["filepath"].tolist()]

    metrics = compute_metrics(y_true_all, y_pred_all, y_proba_all)
    save_metrics(metrics, exp_dir)
    save_predictions(ids, y_true_all, y_pred_all, y_proba_all, exp_dir)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES,
                          exp_dir / "confusion_matrix.png")
    plot_roc_curve(y_true_all, y_proba_all, exp_dir / "roc_curve.png")

    log.info("%s — F1=%.4f  AUC=%.4f  Acc=%.4f",
             model_name, metrics["f1"], metrics["roc_auc"], metrics["accuracy"])
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kayıtlı modeli test seti üzerinde değerlendir.")
    p.add_argument("--model",  required=True,
                   help="Model adı: baseline | classical_ml | alexnet | vgg16 | resnet50 | all")
    p.add_argument("--config", default="config.yaml", help="Config dosyası yolu")
    return p.parse_args()


def evaluate_single(model_name: str, config: dict) -> dict:
    """
    Tek bir modeli değerlendir; metrics dict döndür.
    scripts/run_test.py bu fonksiyonu import eder.
    """
    exp_dir  = Path(config["paths"]["experiments"]) / model_name
    proc_dir = Path(config["paths"]["data_processed"])
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in CLASSICAL_MODELS:
        clf = load_classical_model(model_name, exp_dir, config)
        return evaluate_classical(clf, model_name, proc_dir, config, exp_dir)
    else:
        model = load_deep_model(model_name, exp_dir, config, device)
        return evaluate_deep(model, model_name, proc_dir, config, exp_dir, device)


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    p2 = cfg.get("phase2", {})
    if args.model == "all":
        models = p2.get("models", ["baseline", "classical_ml", "alexnet", "vgg16", "resnet50"])
    else:
        models = [args.model]

    results = {}
    for m in models:
        log.info("=== Değerlendiriliyor: %s ===", m)
        results[m] = evaluate_single(m, cfg)

    # Özet tablo
    print("\n" + "=" * 68)
    print(f"  {'Model':<15} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Spec':>7} {'Prec':>6} {'Recall':>6}")
    print("=" * 68)
    for m, r in results.items():
        print(f"  {m:<15} {r['accuracy']:>6.4f} {r['f1']:>6.4f} "
              f"{r['roc_auc']:>6.4f} {r['specificity']:>7.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
