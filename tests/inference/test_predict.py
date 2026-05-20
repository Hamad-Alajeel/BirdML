"""Tests for the predict() output contract — the API and frontend depend on it."""

import torch

from bird_classifier.inference.predict import predict


def test_predict_returns_exactly_k_items(tiny_pil_image, stub_model, cpu_device):
    results = predict(tiny_pil_image, stub_model, cpu_device, k=3)
    assert len(results) == 3


def test_predict_results_have_label_and_score(tiny_pil_image, stub_model, cpu_device):
    """Every entry must have an int label and a float score — the API contract."""
    results = predict(tiny_pil_image, stub_model, cpu_device, k=2)
    for entry in results:
        assert "label" in entry
        assert "score" in entry
        assert isinstance(entry["label"], int)
        assert isinstance(entry["score"], float)


def test_predict_scores_are_softmax_probabilities(tiny_pil_image, stub_model, cpu_device):
    """Scores must be probabilities in [0, 1] — softmax is applied inside predict().

    Pulling all 10 classes so the sum should be ≈ 1.0 (full softmax distribution).
    """
    results = predict(tiny_pil_image, stub_model, cpu_device, k=10)
    for entry in results:
        assert 0.0 <= entry["score"] <= 1.0
    assert sum(e["score"] for e in results) == pytest_approx(1.0, abs=1e-3)


def test_predict_results_sorted_by_score_descending(tiny_pil_image, stub_model, cpu_device):
    results = predict(tiny_pil_image, stub_model, cpu_device, k=5)
    scores = [e["score"] for e in results]
    assert scores == sorted(scores, reverse=True)


def test_predict_attaches_species_names_when_idx_to_name_provided(
    tiny_pil_image, stub_model, cpu_device
):
    """The 'name' field must be populated when idx_to_name is passed."""
    idx_to_name = {i: f"species_{i}" for i in range(10)}
    results = predict(tiny_pil_image, stub_model, cpu_device, idx_to_name=idx_to_name, k=5)

    for entry in results:
        assert "name" in entry
        assert entry["name"] == idx_to_name[entry["label"]]


def test_predict_omits_name_when_idx_to_name_is_none(tiny_pil_image, stub_model, cpu_device):
    """No 'name' key when idx_to_name isn't provided — keeps the response minimal."""
    results = predict(tiny_pil_image, stub_model, cpu_device, k=2)
    for entry in results:
        assert "name" not in entry


def test_predict_handles_non_rgb_input_image(stub_model, cpu_device):
    """Greyscale and RGBA inputs must work — predict() handles .convert('RGB') internally."""
    from PIL import Image

    greyscale = Image.new("L", (300, 300), color=128)
    rgba = Image.new("RGBA", (300, 300), color=(0, 0, 0, 128))

    grey_results = predict(greyscale, stub_model, cpu_device, k=2)
    rgba_results = predict(rgba, stub_model, cpu_device, k=2)

    assert len(grey_results) == 2
    assert len(rgba_results) == 2


def test_predict_runs_model_in_eval_mode(tiny_pil_image, stub_model, cpu_device):
    """predict() must put the model in eval mode (matters for BatchNorm / Dropout)."""
    stub_model.train()  # start in training mode
    predict(tiny_pil_image, stub_model, cpu_device, k=2)
    assert not stub_model.training


def pytest_approx(value, **kwargs):
    """Tiny shim — pytest.approx imported lazily to avoid top-level import in test names."""
    import pytest

    return pytest.approx(value, **kwargs)
