"""Tests for top_k_accuracy and the eval-mode path of run_epoch."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bird_classifier.training.engine import run_epoch, top_k_accuracy


def test_top_k_accuracy_perfect_prediction():
    """When every argmax matches the label, top-1 must be 1.0."""
    outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    assert top_k_accuracy(outputs, labels, k=1) == 1.0


def test_top_k_accuracy_zero_when_all_wrong():
    outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # argmax=0, label=2 → wrong
            [10.0, 0.0, 0.0],  # argmax=0, label=1 → wrong
        ]
    )
    labels = torch.tensor([2, 1])
    assert top_k_accuracy(outputs, labels, k=1) == 0.0


def test_top_k_accuracy_top5_lenient_than_top1():
    """A label that's the 3rd-highest logit should count for top-5 but not top-1."""
    # 5 classes — label 4 is 3rd highest
    outputs = torch.tensor([[5.0, 4.0, 3.5, 2.0, 4.5]])
    labels = torch.tensor([4])
    assert top_k_accuracy(outputs, labels, k=1) == 0.0
    assert top_k_accuracy(outputs, labels, k=5) == 1.0


def test_top_k_accuracy_matches_argmax_for_k_one():
    """top-1 must always equal (argmax == label).float().mean()."""
    torch.manual_seed(0)
    outputs = torch.randn(32, 100)
    labels = torch.randint(0, 100, (32,))

    expected = (outputs.argmax(dim=1) == labels).float().mean().item()
    assert top_k_accuracy(outputs, labels, k=1) == pytest_approx(expected)


def pytest_approx(value):
    """Small float-equality helper without importing pytest at module top."""
    import pytest

    return pytest.approx(value, abs=1e-6)


def test_run_epoch_eval_returns_three_floats():
    """run_epoch in eval mode must return (loss, top1, top5) — all finite floats."""
    torch.manual_seed(0)
    # Tiny classifier: 4 features → 3 classes
    # 5+ output classes so run_epoch's internal top_k_accuracy(k=5) doesn't out-of-range.
    model = nn.Linear(4, 6)
    images = torch.randn(8, 4)
    labels = torch.randint(0, 6, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    loss, top1, top5 = run_epoch(
        model, loader, device=torch.device("cpu"), training=False
    )
    for value in (loss, top1, top5):
        assert isinstance(value, float)
        assert value == value  # not NaN
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top5 <= 1.0


def test_run_epoch_eval_does_not_update_parameters():
    """Eval mode must NOT modify weights — otherwise we'd corrupt models during evaluation."""
    torch.manual_seed(0)
    # 5+ output classes so run_epoch's internal top_k_accuracy(k=5) doesn't out-of-range.
    model = nn.Linear(4, 6)
    images = torch.randn(8, 4)
    labels = torch.randint(0, 6, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    weights_before = model.weight.detach().clone()
    run_epoch(model, loader, device=torch.device("cpu"), training=False)
    weights_after = model.weight.detach().clone()

    assert torch.equal(weights_before, weights_after)


def test_run_epoch_training_does_update_parameters():
    """Sanity: training=True with an optimizer must change weights."""
    torch.manual_seed(0)
    model = nn.Linear(4, 6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images = torch.randn(8, 4)
    labels = torch.randint(0, 6, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    weights_before = model.weight.detach().clone()
    run_epoch(
        model, loader, device=torch.device("cpu"),
        optimizer=optimizer, training=True,
    )
    weights_after = model.weight.detach().clone()

    assert not torch.equal(weights_before, weights_after)
