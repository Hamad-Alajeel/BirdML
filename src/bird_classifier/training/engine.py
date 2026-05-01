"""Training and evaluation loop utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Return the fraction of samples where the correct label is in the top-k predictions."""
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    training: bool = True,
    epoch_label: str = "",
) -> tuple[float, float, float]:
    """
    Run one full pass over loader in training or evaluation mode.

    Pass optimizer=None when training=False.

    Returns:
        avg_loss, top1_accuracy, top5_accuracy  (all averaged over batches)
    """
    model.train(training)
    criterion = nn.CrossEntropyLoss()

    total_loss = total_top1 = total_top5 = 0.0
    pbar = tqdm(loader, desc=epoch_label, leave=False)

    with torch.set_grad_enabled(training):
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_top1 += top_k_accuracy(outputs, labels, k=1)
            total_top5 += top_k_accuracy(outputs, labels, k=5)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_top1 / n, total_top5 / n
