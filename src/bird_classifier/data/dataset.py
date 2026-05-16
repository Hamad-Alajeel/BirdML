"""PyTorch Dataset wrapper and label utilities for the bird dataset."""

import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

def build_label_mapping(dataset) -> dict[int, int]:
    """
    Build mapping from original label IDs to dense label IDs.

    Example:
        original labels: 0, 1, 2, 4
        dense labels:    0, 1, 2, 3
    """

    unique_labels = set()

    for split in ("train", "validation", "test"):
        unique_labels.update(int(label) for label in dataset[split]["label"])

    return {
        original_label: dense_idx
        for dense_idx, original_label in enumerate(sorted(unique_labels))
    }

def build_idx_to_name(dataset) -> dict[int, str]:
    """Build mapping from DENSE label index to species name.

    Mirrors the compaction done by build_label_mapping so class_names.json[i]
    is the species the trained model actually predicts at dense output i.
    """
    class_label = dataset["train"].features["label"]
    label_to_idx = build_label_mapping(dataset)
    return {
        dense_idx: class_label.int2str(original_label)
        for original_label, dense_idx in label_to_idx.items()
    }


def save_idx_to_name(idx_to_name: dict[int, str], path: Path) -> None:
    """Save the index-to-name mapping to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx_to_name, indent=2))


def load_idx_to_name(path: Path) -> dict[int, str]:
    """Load the index-to-name mapping from a JSON file."""
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


class BirdDataset(Dataset):
    """
    Wrap a Hugging Face dataset split for PyTorch training.

    Converts:
    - PIL image -> transformed tensor
    - original label ID -> dense label ID
    """

    def __init__(
        self,
        hf_split,
        transform: Callable | None = None,
        label_to_idx: dict[int, int] | None = None,
    ) -> None:
        self.hf_split = hf_split
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.hf_split[idx]

        image = sample["image"].convert("RGB")
        label = int(sample["label"])

        if self.label_to_idx is not None:
            label = self.label_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label