"""
deep_models.py — AlexNet, VGG16, ResNet50 transfer öğrenme modelleri.

Son katman 2 çıkışlı Linear ile değiştirilir:
    AlexNet  → classifier[6]   (in=4096)
    VGG16/BN → classifier[6]   (in=4096)
    ResNet50 → fc              (in=2048)
    ResNet18 → fc              (in=512)

Dışa aktarılanlar:
    SUPPORTED_MODELS     : desteklenen mimari adları listesi
    build_model(name, config)   → nn.Module (son katmanı uyarlanmış)
    freeze_backbone(model)      : backbone dondur, son katman aç
    unfreeze_all(model)         : tüm parametreleri aç
"""

import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    VGG16_Weights,
)

SUPPORTED_MODELS = ["alexnet", "vgg16", "vgg16_bn", "resnet50", "resnet18"]


def build_model(name: str, config=None) -> nn.Module:
    """
    İstenen mimariyi yükle; son sınıflandırma katmanını 2 çıkışlı yap.

    config.phase2.pretrained      : True → ImageNet; False → sıfırdan
    config.phase2.freeze_backbone : True → backbone dondur (yalnızca head eğitilir)

    Girdi : name — SUPPORTED_MODELS içinden biri
    Çıktı : nn.Module
    """
    cfg = (config or {}).get("phase2", {})
    pretrained = cfg.get("pretrained", True)
    freeze_bb  = cfg.get("freeze_backbone", False)

    name = name.lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Desteklenmeyen model: '{name}'. Seçenekler: {SUPPORTED_MODELS}")

    if name == "alexnet":
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        model   = tvm.alexnet(weights=weights)
        in_f    = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, 2)

    elif name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model   = tvm.vgg16(weights=weights)
        in_f    = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, 2)

    elif name == "vgg16_bn":
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        model   = tvm.vgg16_bn(weights=weights)
        in_f    = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, 2)

    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model   = tvm.resnet50(weights=weights)
        in_f    = model.fc.in_features
        model.fc = nn.Linear(in_f, 2)

    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model   = tvm.resnet18(weights=weights)
        in_f    = model.fc.in_features
        model.fc = nn.Linear(in_f, 2)

    if freeze_bb:
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Tüm parametreleri dondur; ardından son sınıflandırma katmanını yeniden aç."""
    for p in model.parameters():
        p.requires_grad = False

    # ResNet: fc; AlexNet/VGG: classifier'ın son elemanı
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "classifier"):
        for p in model.classifier[-1].parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Tüm parametreleri eğitime aç (tam fine-tuning)."""
    for p in model.parameters():
        p.requires_grad = True
