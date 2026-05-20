"""Tests for model builder and checkpoint round-trip."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

from bird_classifier.training.model import build_model, get_device, load_checkpoint


@pytest.fixture(scope="module")
def fake_imagenet_weights(tmp_path_factory):
    """A throwaway 'pretrained' weights file with random weights — same shape as the real one.

    Module-scoped so we only pay the EfficientNet-B3 instantiation cost once.
    """
    path = tmp_path_factory.mktemp("weights") / "imagenet.pth"
    torch.save(efficientnet_b3(weights=None).state_dict(), path)
    return path


@pytest.fixture(scope="module")
def finetuned_checkpoint(tmp_path_factory):
    """A throwaway fine-tuned checkpoint with a 4-class head (small for tests)."""
    path = tmp_path_factory.mktemp("ckpt") / "finetuned.pth"
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    torch.save(model.state_dict(), path)
    return path


def test_build_model_head_matches_num_classes(fake_imagenet_weights):
    """The classifier head must be sized to num_classes — otherwise the loss is undefined."""
    model = build_model(num_classes=42, weights_path=fake_imagenet_weights)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == 42


def test_build_model_backbone_uses_provided_pretrained_weights(fake_imagenet_weights):
    """A specific feature-extractor weight should match the saved file (proves load happened)."""
    model = build_model(num_classes=4, weights_path=fake_imagenet_weights)
    saved = torch.load(fake_imagenet_weights, map_location="cpu", weights_only=True)
    # Pick any features-stage parameter — must match the saved state dict
    assert torch.equal(
        model.features[0][0].weight,
        saved["features.0.0.weight"],
    )


def test_load_checkpoint_roundtrip(finetuned_checkpoint):
    """Save state_dict → load_checkpoint → state dict equals what we saved."""
    model, _ = load_checkpoint(finetuned_checkpoint, num_classes=4, device=torch.device("cpu"))
    saved = torch.load(finetuned_checkpoint, map_location="cpu", weights_only=True)

    loaded = model.state_dict()
    assert loaded.keys() == saved.keys()
    for key in saved:
        assert torch.equal(loaded[key], saved[key]), f"Mismatch at {key}"


def test_load_checkpoint_returns_model_on_target_device(finetuned_checkpoint):
    model, device = load_checkpoint(finetuned_checkpoint, num_classes=4, device=torch.device("cpu"))
    assert device.type == "cpu"
    assert next(model.parameters()).device.type == "cpu"


def test_load_checkpoint_raises_on_head_size_mismatch(finetuned_checkpoint):
    """A 4-class checkpoint loaded into a 10-class architecture must fail loudly, not silently."""
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(finetuned_checkpoint, num_classes=10, device=torch.device("cpu"))


def test_get_device_returns_torch_device():
    """get_device must always return a valid torch.device, regardless of CUDA availability."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}
