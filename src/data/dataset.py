"""
dataset.py — PyTorch Dataset ve veri dönüşüm pipeline'ı.

Dışa aktarılanlar:
    ALLDataset          : CSV tabanlı torch.utils.data.Dataset
    get_transforms()    : augmentation bayrağına göre train/val/test transform döndürür
    get_dataloaders()   : config.yaml'dan DataLoader üçlüsü (train, val, test) döndürür

Etiket sözleşmesi:
    label = 1  →  ALL  (lösemi)
    label = 0  →  HEM  (normal)

Ablation notu:
    get_dataloaders(augment=True/False) ile iki farklı train transform seti seçilir;
    bu ayrım Faz 5 augmentation ablasyon çalışması için zorunludur.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ALLDataset(Dataset):
    """
    CSV dosyasından C-NMC görüntülerini yükleyen Dataset.

    Args:
        csv_path: train.csv / val.csv / test.csv yollarından biri.
                  Beklenen sütunlar: filepath, label
        transform: torchvision transform (None ise ham PIL Image döner)
    """

    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        # Sütun adı kontrolü
        for col in ("filepath", "label"):
            if col not in self.df.columns:
                raise ValueError(f"CSV'de beklenen sütun yok: '{col}' — {csv_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    @property
    def labels(self):
        """Tüm etiketlerin listesi (WeightedRandomSampler için kullanışlı)."""
        return self.df["label"].tolist()


def get_transforms(config: dict, split: str, augment: bool = True):
    """
    split: 'train' | 'val' | 'test'
    augment: yalnızca split=='train' için geçerli; False ise aug uygulanmaz.

    Döndürür: torchvision.transforms.Compose
    """
    size = config["dataset"]["image_size"]
    aug_cfg = config["augmentation"]
    mean = aug_cfg["normalize"]["mean"]
    std  = aug_cfg["normalize"]["std"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if split == "train" and augment and aug_cfg.get("enabled", True):
        # --- Augmented train pipeline ---
        tf_list = [
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        ]
        if aug_cfg.get("horizontal_flip", True):
            tf_list.append(transforms.RandomHorizontalFlip())
        if aug_cfg.get("vertical_flip", False):
            tf_list.append(transforms.RandomVerticalFlip())
        rot = aug_cfg.get("rotation_degrees", 15)
        if rot > 0:
            tf_list.append(transforms.RandomRotation(rot))
        cj = aug_cfg.get("color_jitter", {})
        if cj:
            tf_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
            ))
        tf_list += [transforms.ToTensor(), normalize]
    else:
        # --- Baseline (no-aug) pipeline: val, test veya aug=False train ---
        tf_list = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(tf_list)


def get_dataloaders(
    config: dict,
    augment: bool = True,
    processed_dir: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    train/val/test DataLoader üçlüsü döndürür.

    Args:
        config:        load_config() ile yüklenmiş dict
        augment:       True → train setine augmentation uygula
        processed_dir: data/processed/ yolu (None ise config'den alınır)

    Döndürür: (train_loader, val_loader, test_loader)
    """
    proc_dir = Path(processed_dir or config["paths"]["data_processed"])
    bs = config["training"]["batch_size"]
    nw = config["training"]["num_workers"]

    train_ds = ALLDataset(
        proc_dir / "train.csv",
        transform=get_transforms(config, "train", augment=augment),
    )
    val_ds = ALLDataset(
        proc_dir / "val.csv",
        transform=get_transforms(config, "val"),
    )
    test_ds = ALLDataset(
        proc_dir / "test.csv",
        transform=get_transforms(config, "test"),
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader
