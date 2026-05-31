"""
dataset.py — PyTorch Dataset sınıfı ve veri dönüşüm pipeline'ı.

Dışa aktarılanlar:
    ALLDataset          : CSV tabanlı torch.utils.data.Dataset
    get_transforms()    : augmentation bayrağına göre train/val/test transform döndürür
    get_dataloaders()   : config.yaml'dan DataLoader üçlüsü (train, val, test) döndürür

Kullanım örneği:
    from src.data.dataset import get_dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(config, augment=True)
"""

from torch.utils.data import Dataset

# TODO (Faz 2):
#   ALLDataset.__init__:
#     - csv_path (str): train/val/test.csv yollarından biri
#     - transform (callable | None): torchvision transform
#     - Sütunlar: 'path' (görüntü yolu), 'label' (0=HEM/normal, 1=ALL)
#
#   ALLDataset.__getitem__:
#     - PIL.Image ile görüntüyü aç (RGB)
#     - transform uygula
#     - (tensor, label) döndür
#
#   get_transforms(config, split):
#     - split == 'train' ve config.augmentation.enabled == True ise
#       RandomHorizontalFlip, RandomRotation, ColorJitter + Normalize ekle
#     - val/test için sadece Resize + ToTensor + Normalize
#
#   get_dataloaders(config, augment):
#     - ALLDataset örnekleri yarat (train aug açık/kapalı bayrağa göre)
#     - DataLoader ile sar (batch_size, num_workers, shuffle=True sadece train)
#     - (train_loader, val_loader, test_loader) döndür


class ALLDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        raise NotImplementedError("TODO (Faz 2): ALLDataset.__init__ implemente edilmedi.")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def get_transforms(config, split: str):
    """split: 'train' | 'val' | 'test'"""
    raise NotImplementedError("TODO (Faz 2): get_transforms implemente edilmedi.")


def get_dataloaders(config, augment: bool = True):
    """config: dict yüklenmiş config.yaml; augment: train setine augmentation uygula."""
    raise NotImplementedError("TODO (Faz 2): get_dataloaders implemente edilmedi.")
