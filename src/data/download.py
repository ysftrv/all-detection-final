"""
download.py — Kaggle'dan C-NMC 2019 veri setini indir ve data/raw/ altına aç.

Kullanım:
    python -m src.data.download

Kaggle kimlik bilgisi arama sırası:
    1. Proje kökündeki kaggle.json
    2. ~/.kaggle/kaggle.json  (Kaggle API standardı)
    3. KAGGLE_USERNAME / KAGGLE_KEY ortam değişkenleri

C-NMC Dosya adı formatı (indirme sonrası doğrulanmış):
    {UID}_{frameNo}_{cellNo}.bmp
    Örn: UID_H1_1_3.bmp  →  patient_id = "UID_H1"
    parse_patient_id() fonksiyonu bakınız.

Çıktı klasör yapısı (data/raw/ altında):
    training_data/
        fold_0/  fold_1/  fold_2/
            all/   hem/
    testing_data/
        C-NMC_test_prelim_phase_data/   (etiketsiz)
        C-NMC_test_prelim_phase_data_2/ (etiketsiz)
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Proje kökü: bu dosyanın konumuna göre hesaplanır (src/data/download.py → ../../)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _setup_kaggle_credentials() -> None:
    """
    Kaggle kimlik bilgisini birden fazla konumdan arar ve gerekirse
    ~/.kaggle/kaggle.json konumuna kopyalar.
    """
    kaggle_dir = Path.home() / ".kaggle"
    standard_path = kaggle_dir / "kaggle.json"

    # 1. Zaten standart konumda var mı?
    if standard_path.exists():
        log.info("Kaggle kimlik bilgisi ~/.kaggle/kaggle.json konumunda bulundu.")
        standard_path.chmod(0o600)
        return

    # 2. Proje kökünde kaggle.json var mı?
    project_cred = PROJECT_ROOT / "kaggle.json"
    if project_cred.exists():
        kaggle_dir.mkdir(exist_ok=True)
        shutil.copy2(project_cred, standard_path)
        standard_path.chmod(0o600)
        log.info("kaggle.json proje kökünden ~/.kaggle/ konumuna kopyalandı.")
        return

    # 3. Ortam değişkenleri var mı?
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        kaggle_dir.mkdir(exist_ok=True)
        creds = {
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_KEY"],
        }
        standard_path.write_text(json.dumps(creds))
        standard_path.chmod(0o600)
        log.info("Kaggle kimlik bilgisi ortam değişkenlerinden oluşturuldu.")
        return

    raise FileNotFoundError(
        "Kaggle kimlik bilgisi bulunamadı.\n"
        "Çözüm seçenekleri:\n"
        "  A) kaggle.json dosyasını proje köküne veya ~/.kaggle/ altına koyun.\n"
        "  B) KAGGLE_USERNAME ve KAGGLE_KEY ortam değişkenlerini tanımlayın.\n"
        "  Kaggle API token: https://www.kaggle.com/settings → 'Create New Token'"
    )


def _find_training_root(raw_dir: Path) -> Path | None:
    """data/raw/ içinde training_data/ klasörünü bulur (iç içe olabilir)."""
    for candidate in raw_dir.rglob("training_data"):
        if candidate.is_dir():
            return candidate.parent
    return None


def _verify_structure(dataset_root: Path) -> dict:
    """
    İndirilen veri setinin C-NMC yapısına uyup uymadığını kontrol eder.
    Döndürür: {'folds': [...], 'test_dirs': [...], 'total_images': int}
    """
    training = dataset_root / "training_data"
    if not training.exists():
        raise ValueError(f"Beklenen 'training_data/' klasörü bulunamadı: {dataset_root}")

    folds_found = sorted([d.name for d in training.iterdir() if d.is_dir()])
    if not folds_found:
        raise ValueError("training_data/ altında fold klasörü bulunamadı.")

    total = 0
    for fold_dir in training.iterdir():
        if not fold_dir.is_dir():
            continue
        for cls_dir in fold_dir.iterdir():
            if cls_dir.is_dir():
                total += sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".bmp", ".jpg", ".png"})

    testing = dataset_root / "testing_data"
    test_dirs = []
    if testing.exists():
        test_dirs = [d.name for d in testing.iterdir() if d.is_dir()]

    return {"folds": folds_found, "test_dirs": test_dirs, "total_images": total}


def download(raw_dir: str | None = None) -> Path:
    """
    Veri setini indir; zaten indirilmişse tekrar indirmez (idempotent).
    Döndürür: dataset_root (training_data/ ve testing_data/'ın ebeveyni).
    """
    from src.utils.config import load_config

    cfg = load_config()
    raw_path = Path(raw_dir or cfg["paths"]["data_raw"])
    raw_path.mkdir(parents=True, exist_ok=True)

    # İdempotent kontrol: training_data/ zaten var mı?
    existing_root = _find_training_root(raw_path)
    if existing_root is not None:
        log.info("Veri seti zaten mevcut: %s — indirme atlandı.", existing_root)
        info = _verify_structure(existing_root)
        log.info("Yapı: %s | Test klasörleri: %s | Toplam görüntü: ~%d",
                 info["folds"], info["test_dirs"], info["total_images"])
        return existing_root

    # Kaggle kimlik bilgisi hazırla
    _setup_kaggle_credentials()

    # Kaggle API ile indir
    import kaggle  # noqa: import burada — credentials hazır olduktan sonra
    kaggle.api.authenticate()

    dataset_id = cfg["dataset"]["kaggle_id"]
    log.info("İndiriliyor: %s → %s", dataset_id, raw_path)
    kaggle.api.dataset_download_files(dataset_id, path=str(raw_path), unzip=True)
    log.info("İndirme ve açma tamamlandı.")

    # Yapıyı doğrula
    dataset_root = _find_training_root(raw_path)
    if dataset_root is None:
        raise ValueError(
            f"İndirme sonrası training_data/ klasörü bulunamadı. "
            f"{raw_path} içeriği: {list(raw_path.iterdir())}"
        )

    info = _verify_structure(dataset_root)
    log.info("Yapı doğrulandı: %s | Test: %s | Toplam görüntü: ~%d",
             info["folds"], info["test_dirs"], info["total_images"])

    # İndirme logu
    logs_dir = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "folds": info["folds"],
        "test_dirs": info["test_dirs"],
        "total_images": info["total_images"],
    }
    log_file = logs_dir / "download_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    log.info("İndirme logu kaydedildi: %s", log_file)

    return dataset_root


def main():
    download()


if __name__ == "__main__":
    main()
