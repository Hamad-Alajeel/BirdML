"""Evaluation metrics."""

import torch


def top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Return the fraction of samples where the correct label is in the top-k predictions."""
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()
