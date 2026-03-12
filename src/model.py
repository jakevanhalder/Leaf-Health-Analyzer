"""
src/model.py — model factory for leaf disease classification.

Supported architectures
-----------------------
  "resnet50"        : torchvision ResNet-50, pretrained on ImageNet-1k
  "efficientnet_b0" : timm EfficientNet-B0, pretrained on ImageNet-1k

Both are adapted for two task types:
  task="single"  — softmax head, output size = num_classes (PlantVillage)
  task="multi"   — sigmoid head, output size = num_classes (Plant Pathology)

Public API
----------
build_model(arch, num_classes, task, pretrained, freeze_backbone) -> nn.Module
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torchvision import models


def _replace_head(backbone: nn.Module, in_features: int,
                  num_classes: int, task: str) -> nn.Module:
    """Swap the final linear layer for a task-appropriate head."""
    head = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def build_model(
    arch: str = "resnet50",
    num_classes: int = 38,
    task: str = "single",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create and return a model ready for training.

    Parameters
    ----------
    arch             : "resnet50" or "efficientnet_b0"
    num_classes      : number of output neurons (classes or labels)
    task             : "single" (cross-entropy) or "multi" (BCE)
    pretrained       : load ImageNet weights when True
    freeze_backbone  : freeze all layers except the final head when True

    Returns
    -------
    nn.Module
    """
    arch = arch.lower()

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model   = models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = _replace_head(model.fc, in_feat, num_classes, task)

    elif arch == "efficientnet_b0":
        model   = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                    num_classes=num_classes)
        # timm already replaces the head — re-init for clean training
        in_feat = model.classifier.in_features
        model.classifier = _replace_head(model.classifier, in_feat, num_classes, task)

    else:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         "Choose 'resnet50' or 'efficientnet_b0'.")

    if freeze_backbone:
        # Freeze everything, then unfreeze the head
        for param in model.parameters():
            param.requires_grad = False
        head = model.fc if arch == "resnet50" else model.classifier
        for param in head.parameters():
            param.requires_grad = True

    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

