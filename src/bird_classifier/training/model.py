"""EfficientNet-B3 model builder."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

from bird_classifier.config import (
    EFFICIENTNET_B3_WEIGHTS_FILENAME,
    MODELS_DIR,
    NUM_CLASSES,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    num_classes: int = NUM_CLASSES,
    weights_path: Path | None = None,
) -> nn.Module:
    """
    Build EfficientNet-B3 with ImageNet pretrained backbone and a fresh classification head.

    Loads the pretrained weights from weights_path, then replaces the 1000-class
    ImageNet head with a new Linear layer sized for num_classes. Used at the start
    of every training run and every Optuna trial.
    """
    if weights_path is None:
        weights_path = MODELS_DIR / EFFICIENTNET_B3_WEIGHTS_FILENAME

    model = efficientnet_b3(weights=None)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(
    checkpoint_path: Path | str,
    num_classes: int = NUM_CLASSES,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """
    Load a fine-tuned model checkpoint for evaluation or inference.

    Builds the EfficientNet-B3 architecture with the correct head size, then loads
    the saved state dict directly — skipping the pretrained weights since the
    checkpoint already contains the fully trained weights.
    """
    if device is None:
        device = get_device()

    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return model.to(device), device
