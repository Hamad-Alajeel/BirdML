"""Inference — single-image forward pass."""

from typing import Any

import torch
from PIL import Image

from bird_classifier.data.transforms import get_eval_transforms


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    idx_to_name: dict[int, str] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run inference on a single PIL image.

    Args:
        image: PIL image of any size or mode — will be converted and resized internally.
        model: Loaded EfficientNet-B3 model in eval mode.
        device: Device the model lives on.
        idx_to_name: Optional mapping from label index to species name.
        k: Number of top predictions to return.

    Returns:
        List of k dicts sorted by score descending, each with:
            'label' (int), 'score' (float 0-1), and 'name' (str) if idx_to_name provided.
    """
    transform = get_eval_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top_probs, top_indices = probs.topk(k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist()):
        entry: dict[str, Any] = {"label": idx, "score": round(prob, 4)}
        if idx_to_name:
            entry["name"] = idx_to_name.get(idx, str(idx))
        results.append(entry)

    return results
