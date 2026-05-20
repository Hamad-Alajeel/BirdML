"""Tests for the FastAPI inference service.

The app's lifespan handler downloads the model from S3 at startup, which we
can't do in unit tests. Instead, we bypass the lifespan and inject our stub
model directly into the app's shared state dict.
"""

import io

import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(stub_model, cpu_device):
    """A TestClient with state populated as if the lifespan had run."""
    # Import inside the fixture so any state from a previous test is reset.
    from api import main

    main.state.clear()
    main.state["model"] = stub_model
    main.state["device"] = cpu_device
    main.state["idx_to_name"] = {i: f"species_{i}" for i in range(10)}

    # Construct TestClient WITHOUT a `with` block — the context-manager form
    # triggers FastAPI's lifespan, which downloads the real model from S3 and
    # overwrites our stub. Plain construction skips lifespan entirely.
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client

    main.state.clear()


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_top_k(api_client, tiny_png_bytes):
    response = api_client.post(
        "/predict",
        files={"file": ("bird.png", tiny_png_bytes, "image/png")},
        params={"k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 3


def test_predict_endpoint_returns_label_score_name(api_client, tiny_png_bytes):
    """The API surface matches the predict() contract."""
    response = api_client.post(
        "/predict",
        files={"file": ("bird.png", tiny_png_bytes, "image/png")},
    )
    body = response.json()
    for entry in body["predictions"]:
        assert set(entry.keys()) >= {"label", "score", "name"}
        assert isinstance(entry["label"], int)
        assert isinstance(entry["score"], float)
        assert isinstance(entry["name"], str)


def test_predict_endpoint_rejects_non_image(api_client):
    """A text upload should be rejected with HTTP 400, not 500."""
    response = api_client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_endpoint_default_k_is_five(api_client, tiny_png_bytes):
    """The default k=5 must be honored when no query param is passed."""
    response = api_client.post(
        "/predict",
        files={"file": ("bird.png", tiny_png_bytes, "image/png")},
    )
    body = response.json()
    assert len(body["predictions"]) == 5
