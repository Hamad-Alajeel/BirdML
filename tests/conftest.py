"""Shared pytest fixtures."""

import io

import pytest
import torch
import torch.nn as nn
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, Value
from PIL import Image


@pytest.fixture
def tiny_pil_image() -> Image.Image:
    """A 224x224 RGB PIL image with a deterministic gradient pattern."""
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for x in range(224):
        for y in range(224):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


@pytest.fixture
def tiny_png_bytes(tiny_pil_image) -> bytes:
    """The tiny image encoded as PNG bytes (used by the API upload test)."""
    buf = io.BytesIO()
    tiny_pil_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_dataset_sparse(tiny_pil_image) -> DatasetDict:
    """
    Synthetic HF DatasetDict with deliberately SPARSE label IDs ({0, 1, 5, 9}).

    Class 9 only appears in val/test, mimicking the real bird dataset's quirk
    where one class shows up only outside train. Used to exercise the
    sparse→dense remap in build_label_mapping.
    """
    features = Features({"image": HFImage(), "label": Value("int64")})
    train = Dataset.from_dict(
        {"image": [tiny_pil_image] * 6, "label": [0, 0, 1, 1, 5, 5]},
        features=features,
    )
    val = Dataset.from_dict(
        {"image": [tiny_pil_image] * 4, "label": [0, 1, 5, 9]},
        features=features,
    )
    test = Dataset.from_dict(
        {"image": [tiny_pil_image] * 4, "label": [0, 1, 5, 9]},
        features=features,
    )
    return DatasetDict({"train": train, "validation": val, "test": test})


@pytest.fixture
def synthetic_dataset_classlabel(tiny_pil_image) -> DatasetDict:
    """
    Synthetic HF DatasetDict with a ClassLabel feature (named classes).

    Used for build_idx_to_name which needs class_label.int2str(...).
    """
    class_names = ["sparrow", "robin", "eagle", "owl"]
    features = Features({"image": HFImage(), "label": ClassLabel(names=class_names)})
    train = Dataset.from_dict(
        {"image": [tiny_pil_image] * 4, "label": [0, 1, 2, 3]},
        features=features,
    )
    val = Dataset.from_dict(
        {"image": [tiny_pil_image] * 4, "label": [0, 1, 2, 3]},
        features=features,
    )
    test = Dataset.from_dict(
        {"image": [tiny_pil_image] * 4, "label": [0, 1, 2, 3]},
        features=features,
    )
    return DatasetDict({"train": train, "validation": val, "test": test})


class _StubClassifier(nn.Module):
    """Tiny deterministic classifier for tests that need a model without paying EfficientNet-B3's cost.

    Takes a [B, 3, H, W] tensor, global-average-pools to [B, 3], passes through a
    fixed-weight Linear to produce [B, num_classes] logits. Used by predict() and
    API tests where the only thing that matters is the output-shape contract.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(3, num_classes, bias=True)
        # Fix the weights so predictions are deterministic across test runs.
        with torch.no_grad():
            self.linear.weight.fill_(0.5)
            self.linear.bias.copy_(torch.linspace(-1.0, 1.0, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=(2, 3)))


@pytest.fixture
def stub_model() -> nn.Module:
    """A 10-class stub model (see _StubClassifier).

    10 classes is enough to satisfy `topk(5)` for the API's default `k=5` while
    still being small enough to keep softmax-sum-to-one checks tractable.
    """
    return _StubClassifier(num_classes=10)


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU so tests don't depend on CUDA availability."""
    return torch.device("cpu")
