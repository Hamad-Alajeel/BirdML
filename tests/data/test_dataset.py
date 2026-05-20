"""Tests for label mapping and BirdDataset — the seam that caused the eval bug."""

import json

import pytest
import torch
from torchvision import transforms

from bird_classifier.data.dataset import (
    BirdDataset,
    build_idx_to_name,
    build_label_mapping,
    load_idx_to_name,
    save_idx_to_name,
)


def test_build_label_mapping_is_deterministic(synthetic_dataset_sparse):
    """Same dataset → same mapping. This is the regression test for the container/notebook drift."""
    first = build_label_mapping(synthetic_dataset_sparse)
    second = build_label_mapping(synthetic_dataset_sparse)
    assert first == second


def test_build_label_mapping_compacts_sparse_labels(synthetic_dataset_sparse):
    """Sparse {0, 1, 5, 9} should compact to dense {0, 1, 2, 3}."""
    mapping = build_label_mapping(synthetic_dataset_sparse)
    assert mapping == {0: 0, 1: 1, 5: 2, 9: 3}


def test_build_label_mapping_unions_all_splits(synthetic_dataset_sparse):
    """Class 9 only appears in val/test — it must still be in the mapping."""
    mapping = build_label_mapping(synthetic_dataset_sparse)
    assert 9 in mapping, "Labels only present in val/test must be included"
    assert len(mapping) == 4


def test_build_label_mapping_dense_indices_are_contiguous(synthetic_dataset_sparse):
    """Dense indices must be 0..N-1 with no gaps — the model's head depends on this."""
    mapping = build_label_mapping(synthetic_dataset_sparse)
    dense_values = sorted(mapping.values())
    assert dense_values == list(range(len(mapping)))


def test_build_idx_to_name_aligns_with_label_mapping(synthetic_dataset_classlabel):
    """class_names.json[i] must name the species the model predicts at dense output i."""
    label_mapping = build_label_mapping(synthetic_dataset_classlabel)
    idx_to_name = build_idx_to_name(synthetic_dataset_classlabel)

    expected_names = ["sparrow", "robin", "eagle", "owl"]
    for original_label, dense_idx in label_mapping.items():
        assert idx_to_name[dense_idx] == expected_names[original_label]


def test_save_load_idx_to_name_roundtrip(tmp_path):
    """save → load must preserve the int-keyed dict (JSON would otherwise stringify keys)."""
    original = {0: "sparrow", 1: "robin", 2: "eagle"}
    path = tmp_path / "class_names.json"

    save_idx_to_name(original, path)
    loaded = load_idx_to_name(path)

    assert loaded == original
    assert all(isinstance(k, int) for k in loaded), "Keys must be int after load"


def test_save_idx_to_name_creates_parent_dir(tmp_path):
    """save should create the parent directory if it doesn't exist."""
    path = tmp_path / "nested" / "deep" / "class_names.json"
    save_idx_to_name({0: "sparrow"}, path)
    assert path.exists()


def test_bird_dataset_remaps_sparse_to_dense(synthetic_dataset_sparse):
    """BirdDataset must apply label_to_idx; raw sparse labels would mis-train the model."""
    label_to_idx = build_label_mapping(synthetic_dataset_sparse)
    ds = BirdDataset(
        hf_split=synthetic_dataset_sparse["test"],
        transform=transforms.ToTensor(),
        label_to_idx=label_to_idx,
    )

    _, label_9 = ds[3]  # last test sample has original label 9
    assert label_9 == 3  # dense index for 9 is 3


def test_bird_dataset_returns_tensor_and_int(synthetic_dataset_sparse):
    """Each item must be a (tensor, int) pair — the DataLoader downstream depends on this."""
    label_to_idx = build_label_mapping(synthetic_dataset_sparse)
    ds = BirdDataset(
        hf_split=synthetic_dataset_sparse["train"],
        transform=transforms.ToTensor(),
        label_to_idx=label_to_idx,
    )

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_bird_dataset_applies_transform(synthetic_dataset_sparse, tiny_pil_image):
    """The transform must be called on every __getitem__."""
    calls = []

    def tracking_transform(img):
        calls.append(img)
        return transforms.ToTensor()(img)

    ds = BirdDataset(
        hf_split=synthetic_dataset_sparse["train"],
        transform=tracking_transform,
        label_to_idx=build_label_mapping(synthetic_dataset_sparse),
    )

    _ = ds[0]
    _ = ds[1]
    assert len(calls) == 2


def test_bird_dataset_works_without_label_remap(synthetic_dataset_sparse):
    """When label_to_idx is None, labels pass through unchanged (used for raw exploration)."""
    ds = BirdDataset(
        hf_split=synthetic_dataset_sparse["test"],
        transform=transforms.ToTensor(),
        label_to_idx=None,
    )

    _, raw_label = ds[3]  # was originally 9
    assert raw_label == 9


def test_bird_dataset_length_matches_split(synthetic_dataset_sparse):
    ds = BirdDataset(hf_split=synthetic_dataset_sparse["train"])
    assert len(ds) == len(synthetic_dataset_sparse["train"])


def test_load_idx_to_name_handles_string_keys_from_json(tmp_path):
    """JSON stringifies dict keys — load must convert them back to int."""
    path = tmp_path / "names.json"
    # Write with raw string keys (what json.dumps produces from int keys)
    path.write_text(json.dumps({"0": "a", "1": "b", "42": "c"}))
    loaded = load_idx_to_name(path)
    assert loaded == {0: "a", 1: "b", 42: "c"}
