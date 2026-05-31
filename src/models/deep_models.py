"""
deep_models.py — Katman 3: Transfer öğrenme ile derin sinir ağı modelleri.

Desteklenen mimariler (hepsi ImageNet ön-eğitimli):
    - AlexNet  (torchvision.models.alexnet)
    - VGG-16   (torchvision.models.vgg16)
    - ResNet-50 (torchvision.models.resnet50)

Strateji:
    - Önceden eğitilmiş ağırlıkları dondur (freeze), yalnızca son tam-bağlantı
      katmanını 2-sınıflı çıkışa göre değiştir.
    - İsteğe bağlı fine-tuning: son N bloğu çöz.

Dışa aktarılanlar:
    build_model(name, config)  : model oluştur ve son katmanı uyarla
    freeze_backbone(model)     : conv katmanlarını dondur
    unfreeze_last_n(model, n)  : fine-tuning için son n bloğu aç
"""

import torch.nn as nn

# TODO (Faz 4):
#   build_model(name, config):
#     - torchvision.models'dan pretrained=True ile modeli yükle.
#     - Son katmanı 2 çıkışlı Linear ile değiştir (AlexNet → classifier[-1],
#       VGG → classifier[-1], ResNet → fc).
#     - freeze_backbone() çağır.
#     - Modeli döndür.
#
#   freeze_backbone(model):
#     - model.parameters() döngüsünde requires_grad=False.
#     - Son katman parametrelerini requires_grad=True olarak geri aç.
#
#   unfreeze_last_n(model, n):
#     - ResNet için layer4, layer3... sırasıyla; VGG için features[-n:]
#       gibi mimari-özel mantık gerekir.
#
#   Kayıt / yükleme:
#     - torch.save(model.state_dict(), path)  /  model.load_state_dict(...)
#     - Path şeması: experiments/weights/{model_name}_{aug_flag}.pth


SUPPORTED_MODELS = ["alexnet", "vgg16", "resnet50"]


def build_model(name: str, config=None) -> nn.Module:
    """name: 'alexnet' | 'vgg16' | 'resnet50'. Döndürür: son katmanı uyarlanmış nn.Module."""
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Desteklenmeyen model: {name}. Seçenekler: {SUPPORTED_MODELS}")
    raise NotImplementedError(f"TODO (Faz 4): build_model('{name}') implemente edilmedi.")


def freeze_backbone(model: nn.Module) -> None:
    """Tüm parametreleri dondur, son sınıflandırma katmanını aç."""
    raise NotImplementedError("TODO (Faz 4): freeze_backbone implemente edilmedi.")


def unfreeze_last_n(model: nn.Module, n: int) -> None:
    """Fine-tuning: son n bloğun gradyanını aç."""
    raise NotImplementedError("TODO (Faz 4): unfreeze_last_n implemente edilmedi.")
