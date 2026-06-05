"""
train.py — Tüm 5 modelin eğitim + test değerlendirme orkestratörü.

CLI:
    python -m src.train --model all        [--config config.yaml] [--smoke-test]
    python -m src.train --model resnet50   [--config config.yaml] [--smoke-test]
    python -m src.train --model baseline   [--smoke-test]

Çıktılar: experiments/{model_name}/ altında
    metrics.json, predictions.csv, run_info.json,
    confusion_matrix.png, roc_curve.png
    + derin modeller: weights.pth, weights_last.pth,
                      training_history.csv, training_history.png
    + klasik modeller: model.joblib
    + özellik cache: experiments/features/{model}_{split}.npz
"""

import argparse
import logging
import platform
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ALLDataset, get_transforms
from src.models.baseline import BaselineClassifier, extract_features
from src.models.classical_ml import ClassicalMLClassifier, extract_hog_lbp
from src.models.deep_models import build_model, unfreeze_all
from src.utils.config import load_config
from src.utils.metrics import (
    CLASS_NAMES,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
    save_predictions,
    save_run_info,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CLASSICAL = {"baseline", "classical_ml"}
DEEP      = {"alexnet", "vgg16", "resnet50"}

SMOKE_N = {"train": 40, "val": 20, "test": 20}


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """numpy, random ve torch tohumlarını eş zamanlı sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _real_data_ok(proc_dir: Path) -> bool:
    """Gerçek veri var ve ilk görüntü erişilebilir mi?"""
    csv = proc_dir / "train.csv"
    if not csv.exists():
        return False
    try:
        df = pd.read_csv(csv)
        return len(df) > 0 and Path(df.iloc[0]["filepath"]).exists()
    except Exception:
        return False


class _SmokeContext:
    """
    Smoke test için geçici sentetik veri yöneticisi.

    Gerçek veri varsa proc_dir'i aynen döndürür (subsample train.py'de yapılır).
    Yoksa 64×64 rastgele PNG + CSV üretir; çıkışta temizler.
    """

    def __init__(self, proc_dir: Path):
        self._real = proc_dir
        self._tmp_imgs = None
        self._tmp_proc = None

    def __enter__(self) -> Path:
        if _real_data_ok(self._real):
            return self._real

        from PIL import Image as PILImage  # lazy import

        self._tmp_imgs = Path(tempfile.mkdtemp(prefix="all_smoke_imgs_"))
        self._tmp_proc = Path(tempfile.mkdtemp(prefix="all_smoke_proc_"))
        rng = np.random.RandomState(42)

        for split, n in SMOKE_N.items():
            rows = []
            for i in range(n):
                label = i % 2
                fpath = self._tmp_imgs / f"{split}_{i:04d}.png"
                arr   = rng.randint(30, 220, (64, 64, 3), dtype=np.uint8)
                PILImage.fromarray(arr).save(str(fpath))
                rows.append({"filepath": str(fpath), "label": label,
                             "patient_id": f"P{i // 4:03d}", "fold": "f1"})
            pd.DataFrame(rows).to_csv(self._tmp_proc / f"{split}.csv", index=False)

        log.warning("Gerçek veri yok — sentetik smoke verisi: %s", self._tmp_proc)
        return self._tmp_proc

    def __exit__(self, *args) -> None:
        for d in (self._tmp_imgs, self._tmp_proc):
            if d and d.exists():
                shutil.rmtree(d, ignore_errors=True)


def _class_weights(labels) -> np.ndarray:
    """Ters-frekans sınıf ağırlıkları: shape (2,), idx 0=HEM, 1=ALL."""
    return compute_class_weight("balanced", classes=np.array([0, 1]),
                                y=np.array(labels))


def _build_run_info(model_name: str, cfg: dict, p2: dict, device: torch.device,
                    train_time: float, infer_time: float,
                    n_train: int, n_val: int, n_test: int,
                    smoke_test: bool = False) -> dict:
    """Seed, donanım, hiperparametre ve süre bilgilerini topla."""
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    cuda_ver  = torch.version.cuda or "N/A"
    return {
        "model_name":          model_name,
        "timestamp":           datetime.now().isoformat(),
        "smoke_test":          smoke_test,
        "seed":                p2.get("seed", cfg.get("seed", 42)),
        "device":              str(device),
        "gpu_name":            gpu_name,
        "torch_version":       torch.__version__,
        "cuda_version":        cuda_ver,
        "python_version":      platform.python_version(),
        "augmentation":        p2.get("augmentation", True),
        "pretrained":          p2.get("pretrained", True),
        "freeze_backbone":     p2.get("freeze_backbone", False),
        "hyperparameters":     {k: v for k, v in p2.items() if k != "models"},
        "train_size":          n_train,
        "val_size":            n_val,
        "test_size":           n_test,
        "train_time_sec":      round(train_time, 2),
        "inference_time_sec":  round(infer_time, 2),
    }


# ---------------------------------------------------------------------------
# Klasik model özellik çıkarımı (cache destekli)
# ---------------------------------------------------------------------------

def _get_features(model_name: str, split: str, df: pd.DataFrame,
                  cfg: dict, feat_dir: Path, use_cache: bool):
    """
    Özellik matrisini döndür; cache varsa yükle, yoksa üret ve kaydet.

    Çıktı: (X: np.ndarray float32, y: list[int], ids: list[str])
    """
    cache_path = feat_dir / f"{model_name}_{split}.npz"

    if use_cache and cache_path.exists():
        log.info("Cache yükleniyor: %s", cache_path)
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["y"].tolist(), d["ids"].tolist()

    paths  = df["filepath"].tolist()
    labels = df["label"].tolist()
    ids    = [Path(p).stem for p in paths]

    log.info("%s/%s: %d görüntüden özellik çıkarılıyor…", model_name, split, len(paths))
    p2       = cfg.get("phase2", {})
    img_size = p2.get("image_size", 128)

    if model_name == "baseline":
        X = np.vstack([
            extract_features(p, cfg)
            for p in tqdm(paths, desc=f"{model_name}/{split}")
        ]).astype(np.float32)
    else:
        X = np.vstack([
            extract_hog_lbp(p, img_size)
            for p in tqdm(paths, desc=f"{model_name}/{split}")
        ]).astype(np.float32)

    if use_cache:
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=np.array(labels, dtype=int),
                 ids=np.array(ids, dtype=object))
        log.info("Cache kaydedildi: %s", cache_path)

    return X, labels, ids


# ---------------------------------------------------------------------------
# Klasik model eğitimi
# ---------------------------------------------------------------------------

def train_classical(model_name: str, cfg: dict, smoke_test: bool = False) -> None:
    """
    Klasik model eğitim + test değerlendirme pipeline'ı.

    Girdi : model_name ('baseline' | 'classical_ml'), cfg (tüm config), smoke_test
    """
    p2      = cfg["phase2"]
    seed    = p2["seed"]
    _set_seed(seed)

    proc_dir_real = Path(cfg["paths"]["data_processed"])
    exp_dir       = Path(cfg["paths"]["experiments"]) / model_name
    feat_dir      = Path(cfg["paths"]["experiments"]) / "features"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with _SmokeContext(proc_dir_real) as proc_dir:
        train_df = pd.read_csv(proc_dir / "train.csv")
        val_df   = pd.read_csv(proc_dir / "val.csv")
        test_df  = pd.read_csv(proc_dir / "test.csv")

        if smoke_test:
            train_df = train_df.head(SMOKE_N["train"])
            val_df   = val_df.head(SMOKE_N["val"])
            test_df  = test_df.head(SMOKE_N["test"])

        use_cache = p2.get("use_cached_features", True) and not smoke_test

        X_train, y_train, _       = _get_features(model_name, "train", train_df, cfg, feat_dir, use_cache)
        _,       _,       _       = _get_features(model_name, "val",   val_df,   cfg, feat_dir, use_cache)
        X_test,  y_test,  ids_test = _get_features(model_name, "test",  test_df,  cfg, feat_dir, use_cache)

        if model_name == "baseline":
            clf = BaselineClassifier(cfg)
        else:
            clf = ClassicalMLClassifier(cfg)

        log.info("=== %s eğitimi başlıyor ===", model_name)
        t0 = time.time()
        clf.fit_from_X(X_train, y_train)
        train_time = time.time() - t0

        clf.save(exp_dir / "model.joblib")

        t1 = time.time()
        proba = clf.predict_proba_from_X(X_test)[:, 1]
        preds = clf.predict_from_X(X_test)
        infer_time = time.time() - t1

    metrics = compute_metrics(y_test, preds, proba)
    save_metrics(metrics, exp_dir)
    save_predictions(ids_test, y_test, preds, proba, exp_dir)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES,
                          exp_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, proba, exp_dir / "roc_curve.png")

    device = torch.device("cpu")
    run_info = _build_run_info(
        model_name, cfg, p2, device, train_time, infer_time,
        len(train_df), len(val_df), len(test_df), smoke_test,
    )
    save_run_info(run_info, exp_dir)

    log.info("=== %s TAMAMLANDI — F1=%.4f  AUC=%.4f  Acc=%.4f  Süre=%.0fs ===",
             model_name, metrics["f1"], metrics["roc_auc"],
             metrics["accuracy"], train_time)


# ---------------------------------------------------------------------------
# Derin model eğitim döngüsü
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """Tek eğitim epoch'u; ortalama loss döndür."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / max(len(loader.dataset), 1)


def _eval_epoch(model, loader, criterion, device):
    """
    Değerlendirme epoch'u.
    Döndürür: (avg_loss, f1, y_true_list, y_pred_list, y_proba_list)
    """
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all, y_proba_all = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out   = model(imgs)
            loss  = criterion(out, labels)
            proba = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            total_loss  += loss.item() * imgs.size(0)
            y_true_all.extend(labels.cpu().tolist())
            y_pred_all.extend(preds.cpu().tolist())
            y_proba_all.extend(proba.cpu().tolist())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    val_f1   = f1_score(y_true_all, y_pred_all, zero_division=0)
    return avg_loss, val_f1, y_true_all, y_pred_all, y_proba_all


def _plot_history(hist_df: pd.DataFrame, save_path: Path) -> None:
    """Eğitim loss ve val F1 eğrilerini kaydet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="train loss")
    ax1.plot(hist_df["epoch"], hist_df["val_loss"],   label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Kayıp Eğrisi"); ax1.legend()

    ax2.plot(hist_df["epoch"], hist_df["val_f1"], color="green", label="val F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1")
    ax2.set_title("Validation F1"); ax2.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_deep(model_name: str, cfg: dict, smoke_test: bool = False) -> None:
    """
    Derin model eğitim + test değerlendirme pipeline'ı.

    Girdi : model_name ('alexnet' | 'vgg16' | 'resnet50'), cfg, smoke_test
    """
    p2   = cfg["phase2"]
    seed = p2["seed"]
    _set_seed(seed)

    device = torch.device("cpu" if smoke_test else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    proc_dir_real = Path(cfg["paths"]["data_processed"])
    exp_dir       = Path(cfg["paths"]["experiments"]) / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Model variant seçimi
    if model_name == "vgg16":
        variant = p2.get("vgg_variant", "vgg16")
    elif model_name == "resnet50":
        variant = p2.get("resnet_variant", "resnet50")
    else:
        variant = model_name

    log.info("=== %s (%s) eğitimi başlıyor — cihaz: %s ===",
             model_name, variant, device)

    model = build_model(variant, cfg)
    if not p2.get("freeze_backbone", False):
        unfreeze_all(model)
    model = model.to(device)

    # Transform ayarları
    image_size = p2["image_size"]
    aug_cfg    = dict(cfg["augmentation"])
    aug_cfg["enabled"] = p2.get("augmentation", True)
    cfg_tf = {**cfg, "augmentation": aug_cfg,
              "dataset": {**cfg["dataset"], "image_size": image_size}}

    augment = p2.get("augmentation", True) and not smoke_test
    bs      = 8 if smoke_test else p2["batch_size"]
    nw      = 0 if smoke_test else p2["num_workers"]

    with _SmokeContext(proc_dir_real) as proc_dir:
        train_df = pd.read_csv(proc_dir / "train.csv")
        val_df   = pd.read_csv(proc_dir / "val.csv")
        test_df  = pd.read_csv(proc_dir / "test.csv")

        train_tf = get_transforms(cfg_tf, "train", augment=augment)
        val_tf   = get_transforms(cfg_tf, "val")

        train_ds_full = ALLDataset(proc_dir / "train.csv", transform=train_tf)
        val_ds_full   = ALLDataset(proc_dir / "val.csv",   transform=val_tf)
        test_ds_full  = ALLDataset(proc_dir / "test.csv",  transform=val_tf)

        if smoke_test:
            n_tr = min(SMOKE_N["train"], len(train_ds_full))
            n_va = min(SMOKE_N["val"],   len(val_ds_full))
            n_te = min(SMOKE_N["test"],  len(test_ds_full))
            train_ds = Subset(train_ds_full, list(range(n_tr)))
            val_ds   = Subset(val_ds_full,   list(range(n_va)))
            test_ds  = Subset(test_ds_full,  list(range(n_te)))
            train_df, val_df, test_df = (train_df.head(n_tr),
                                         val_df.head(n_va),
                                         test_df.head(n_te))
        else:
            train_ds, val_ds, test_ds = train_ds_full, val_ds_full, test_ds_full

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=(nw > 0))
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=(nw > 0))
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=(nw > 0))

        # Sınıf ağırlıkları
        cw   = _class_weights(train_df["label"].values)
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32).to(device)
        )

        # Optimizer + scheduler
        opt   = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=p2["learning_rate"], weight_decay=p2["weight_decay"],
        )
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=3, factor=0.5
        )

        use_amp = p2.get("use_amp", True) and device.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        epochs  = 1 if smoke_test else p2["epochs"]
        patience_max = p2["early_stopping_patience"]
        best_f1, patience_cnt = -1.0, 0
        history = []

        weights_best = exp_dir / "weights.pth"
        weights_last = exp_dir / "weights_last.pth"

        # --- Eğitim döngüsü ---
        t_train_start = time.time()
        for epoch in range(1, epochs + 1):
            tr_loss = _train_epoch(model, train_loader, crit, opt, scaler, device)
            va_loss, va_f1, *_ = _eval_epoch(model, val_loader, crit, device)
            sched.step(va_f1)

            history.append({"epoch": epoch, "train_loss": tr_loss,
                             "val_loss": va_loss, "val_f1": va_f1})
            log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f",
                     epoch, epochs, tr_loss, va_loss, va_f1)

            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), weights_best)
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience_max and not smoke_test:
                log.info("Early stopping — epoch %d  best val F1=%.4f", epoch, best_f1)
                break

        torch.save(model.state_dict(), weights_last)
        train_time = time.time() - t_train_start

        # Eğitim geçmişini kaydet
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(exp_dir / "training_history.csv", index=False)
        _plot_history(hist_df, exp_dir / "training_history.png")

        # En iyi ağırlıkla test değerlendirmesi (yoksa son epoch ağırlığını kullan)
        load_path = weights_best if weights_best.exists() else weights_last
        model.load_state_dict(torch.load(load_path, map_location=device))

        t_infer = time.time()
        _, _, y_true, y_pred, y_proba = _eval_epoch(model, test_loader, crit, device)
        infer_time = time.time() - t_infer

        ids = [Path(p).stem for p in test_df["filepath"].tolist()[:len(y_true)]]

    # Metrik ve dosyaları kaydet
    metrics = compute_metrics(y_true, y_pred, y_proba)
    save_metrics(metrics, exp_dir)
    save_predictions(ids, y_true, y_pred, y_proba, exp_dir)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES,
                          exp_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_proba, exp_dir / "roc_curve.png")

    run_info = _build_run_info(
        model_name, cfg, p2, device, train_time, infer_time,
        len(train_df), len(val_df), len(test_df), smoke_test,
    )
    save_run_info(run_info, exp_dir)

    log.info("=== %s TAMAMLANDI — F1=%.4f  AUC=%.4f  Acc=%.4f  Süre=%.0fs ===",
             model_name, metrics["f1"], metrics["roc_auc"],
             metrics["accuracy"], train_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALL Detection — model eğitim + değerlendirme"
    )
    p.add_argument(
        "--model", required=True,
        help="baseline | classical_ml | alexnet | vgg16 | resnet50 | all",
    )
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Küçük sentetik veriyle pipeline doğrulaması (CPU, 1 epoch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    p2   = cfg["phase2"]

    if args.model == "all":
        models = p2.get("models", ["baseline", "classical_ml", "alexnet", "vgg16", "resnet50"])
    else:
        models = [args.model]

    log.info("Eğitilecek modeller: %s  smoke_test=%s", models, args.smoke_test)

    t_total = time.time()
    for m in models:
        if m in CLASSICAL:
            train_classical(m, cfg, smoke_test=args.smoke_test)
        elif m in DEEP:
            train_deep(m, cfg, smoke_test=args.smoke_test)
        else:
            raise ValueError(f"Bilinmeyen model: '{m}'. Seçenekler: {sorted(CLASSICAL | DEEP)}")

    log.info("Toplam süre: %.1f dakika", (time.time() - t_total) / 60)


if __name__ == "__main__":
    main()
